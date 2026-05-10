"""Phase B.2 Testing Gate — test_watchdog_service.py

Tests _Handler (the FileSystemEventHandler subclass) in isolation.
No real git repo, no real watchdog Observer is started.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libucks.diff.diff_extractor import DiffExtractor
from libucks.models.events import DiffEvent, DiffHunk
from libucks.watchdog_service import _Handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


@pytest.fixture
def mock_extractor():
    return MagicMock(spec=DiffExtractor)


@pytest.fixture
def mock_agent():
    # CentralAgent.post is a coroutine — use MagicMock here so the call
    # inside _fire() doesn't error; run_coroutine_threadsafe is patched away
    # in tests that call _fire(), so the return value is never awaited.
    from libucks.central_agent import CentralAgent
    return MagicMock(spec=CentralAgent)


@pytest.fixture
def handler(mock_extractor, mock_agent, loop):
    return _Handler(mock_extractor, mock_agent, loop)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _modified_event(path: str, is_directory: bool = False):
    ev = MagicMock()
    ev.is_directory = is_directory
    ev.src_path = path
    return ev


def _make_diff_event(filepath: str) -> DiffEvent:
    hunk = DiffHunk(
        file=filepath, old_start=1, old_end=2,
        new_start=1, new_end=3,
        added_lines=["new line"], removed_lines=[],
    )
    return DiffEvent(file=filepath, hunks=[hunk], is_rename=False)


# ---------------------------------------------------------------------------
# Extension filtering
# ---------------------------------------------------------------------------

class TestExtensionFilter:
    def test_ds_store_does_not_create_timer(self, handler):
        with patch("libucks.watchdog_service.threading.Timer") as mock_timer_cls:
            handler.on_modified(_modified_event("/repo/.DS_Store"))
        mock_timer_cls.assert_not_called()

    def test_txt_file_does_not_create_timer(self, handler):
        with patch("libucks.watchdog_service.threading.Timer") as mock_timer_cls:
            handler.on_modified(_modified_event("/repo/README.txt"))
        mock_timer_cls.assert_not_called()

    def test_py_file_creates_timer(self, handler):
        mock_t = MagicMock(spec=threading.Timer)
        with patch("libucks.watchdog_service.threading.Timer", return_value=mock_t):
            handler.on_modified(_modified_event("/repo/module.py"))
        mock_t.start.assert_called_once()

    def test_directory_event_does_not_create_timer(self, handler):
        with patch("libucks.watchdog_service.threading.Timer") as mock_timer_cls:
            handler.on_modified(_modified_event("/repo/some_dir", is_directory=True))
        mock_timer_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Debounce logic
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_second_modification_of_same_file_cancels_first_timer(self, handler):
        t1 = MagicMock(spec=threading.Timer)
        t2 = MagicMock(spec=threading.Timer)

        with patch("libucks.watchdog_service.threading.Timer", side_effect=[t1, t2]):
            handler.on_modified(_modified_event("/repo/module.py"))
            handler.on_modified(_modified_event("/repo/module.py"))

        t1.cancel.assert_called_once()
        t2.start.assert_called_once()

    def test_two_different_files_get_independent_timers(self, handler):
        t1 = MagicMock(spec=threading.Timer)
        t2 = MagicMock(spec=threading.Timer)

        with patch("libucks.watchdog_service.threading.Timer", side_effect=[t1, t2]):
            handler.on_modified(_modified_event("/repo/module_a.py"))
            handler.on_modified(_modified_event("/repo/module_b.py"))

        t1.cancel.assert_not_called()
        t1.start.assert_called_once()
        t2.start.assert_called_once()

    def test_timer_started_with_correct_callable_and_path(self, handler):
        t = MagicMock(spec=threading.Timer)
        path = "/repo/module.py"

        with patch("libucks.watchdog_service.threading.Timer", return_value=t) as cls:
            handler.on_modified(_modified_event(path))

        pos_args = cls.call_args[0]   # (delay, callable)
        kw_args  = cls.call_args[1]   # {'args': (Path(path),)}
        assert pos_args[1] == handler._fire
        assert kw_args["args"] == (Path(path),)


# ---------------------------------------------------------------------------
# _fire(): extractor called and events posted
# ---------------------------------------------------------------------------

class TestFire:
    def test_fire_calls_extractor_with_path(self, handler, mock_extractor):
        mock_extractor.extract.return_value = []
        handler._fire(Path("/repo/module.py"))
        mock_extractor.extract.assert_called_once_with(Path("/repo/module.py"))

    def test_fire_posts_event_via_run_coroutine_threadsafe(self, handler, mock_extractor, loop):
        diff_ev = _make_diff_event("module.py")
        mock_extractor.extract.return_value = [diff_ev]

        with patch("libucks.watchdog_service.asyncio.run_coroutine_threadsafe") as mock_rct:
            handler._fire(Path("/repo/module.py"))

        mock_rct.assert_called_once()
        assert mock_rct.call_args[0][1] is loop

    def test_fire_posts_one_call_per_diff_event(self, handler, mock_extractor, loop):
        events = [_make_diff_event("a.py"), _make_diff_event("b.py")]
        mock_extractor.extract.return_value = events

        with patch("libucks.watchdog_service.asyncio.run_coroutine_threadsafe") as mock_rct:
            handler._fire(Path("/repo/module.py"))

        assert mock_rct.call_count == 2

    def test_fire_does_not_post_when_no_diff_events(self, handler, mock_extractor):
        mock_extractor.extract.return_value = []

        with patch("libucks.watchdog_service.asyncio.run_coroutine_threadsafe") as mock_rct:
            handler._fire(Path("/repo/module.py"))

        mock_rct.assert_not_called()


# ---------------------------------------------------------------------------
# _fire(): resilience
# ---------------------------------------------------------------------------

class TestFireResilience:
    def test_extractor_exception_does_not_propagate(self, handler, mock_extractor):
        mock_extractor.extract.side_effect = RuntimeError("git exploded")

        with patch("libucks.watchdog_service.asyncio.run_coroutine_threadsafe") as mock_rct:
            handler._fire(Path("/repo/module.py"))  # must not raise

        mock_rct.assert_not_called()
