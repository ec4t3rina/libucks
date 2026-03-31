"""WatchdogService — OS file events → DiffEvents on CentralAgent's queue.

Debounce window: 500 ms per file (last event wins).
Thread-safe: watchdog callbacks run in a background thread; events are
forwarded to the asyncio event loop via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import structlog
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from libucks.diff.diff_extractor import DiffExtractor
from libucks.parsing.grammar_registry import SUPPORTED_LANGUAGES

if TYPE_CHECKING:
    from libucks.central_agent import CentralAgent

log = structlog.get_logger(__name__)

_DEBOUNCE_SECONDS = 0.5
_TRACKED_EXTENSIONS = set(SUPPORTED_LANGUAGES.keys())


class _Handler(FileSystemEventHandler):
    def __init__(
        self,
        extractor: DiffExtractor,
        agent: "CentralAgent",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._extractor = extractor
        self._agent = agent
        self._loop = loop
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in _TRACKED_EXTENSIONS:
            return
        self._debounce(path)

    def _debounce(self, path: Path) -> None:
        key = str(path)
        with self._lock:
            existing = self._timers.pop(key, None)
            if existing:
                existing.cancel()
            timer = threading.Timer(_DEBOUNCE_SECONDS, self._fire, args=(path,))
            self._timers[key] = timer
            timer.start()

    def _fire(self, path: Path) -> None:
        with self._lock:
            self._timers.pop(str(path), None)

        log.info("watchdog.file_changed", path=str(path))
        try:
            events = self._extractor.extract(path)
        except Exception as exc:
            log.warning("watchdog.extract_error", path=str(path), error=str(exc))
            return

        for diff_event in events:
            asyncio.run_coroutine_threadsafe(
                self._agent.post(diff_event), self._loop
            )
            log.info("watchdog.event_posted", file=diff_event.file)


class WatchdogService:
    def __init__(
        self,
        repo_path: Path,
        agent: "CentralAgent",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._repo_path = repo_path
        self._extractor = DiffExtractor(repo_path)
        self._handler = _Handler(self._extractor, agent, loop)
        self._observer = Observer()

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self._repo_path), recursive=True)
        self._observer.start()
        log.info("watchdog.started", path=str(self._repo_path))

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        log.info("watchdog.stopped")
