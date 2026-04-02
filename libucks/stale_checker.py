"""StaleChecker — JIT (Just-In-Time) staleness detection for the query path.

Runs in <50ms before Librarians are called. If any of the four levels detects
that a bucket's content may be out of sync with the source files, it returns
a StaleCheckResult with is_stale=True so the orchestrator can fire a background
re-index while still answering the query with the stale data (eventual consistency).

Four levels (ordered cheapest → most specific):

  Level 1 — Process: Is the watcher PID recorded in _meta still alive?
             Uses os.kill(pid, 0); ESRCH means the process is gone.
             Budget: ~2ms.

  Level 2 — Git HEAD: Does the current HEAD differ from _meta.last_indexed_head?
             Uses a subprocess git rev-parse HEAD.
             If different, ALL queried buckets are flagged stale and we return early.
             Budget: ~15ms.

  Level 3 — File mtime: Are source files newer than the bucket's last_indexed_at?
             Uses os.stat().st_mtime on every unique source_file in each queried bucket.
             Detects uncommitted saves (the watcher debounce window, IDE saves).
             Budget: ~5ms.

  Level 4 — Chunk SHA: Does the bucket's index_head_sha differ from current HEAD?
             Detects per-bucket SHA drift (bucket was indexed at an older commit).
             Purely in-memory; current_head reused from Level 2.
             Budget: ~5ms.

The three module-level helpers (_process_is_alive, _get_current_head,
_get_file_mtime) are extracted for test-patchability without global mocking.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import structlog

if TYPE_CHECKING:
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Injectable helpers (patched in unit tests)
# ---------------------------------------------------------------------------

def _process_is_alive(pid: int) -> bool:
    """Return True if the process with *pid* exists on this machine."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists; we just lack permission to signal it
    except OSError:
        return False


def _get_current_head(repo_path: Path) -> Optional[str]:
    """Return current git HEAD SHA via subprocess, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_file_mtime(path: str) -> Optional[float]:
    """Return os.stat mtime for *path*, or None if the file cannot be stat'd."""
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StaleCheckResult:
    is_stale: bool
    stale_bucket_ids: List[str]
    level: int   # 0 = fresh; 1–4 = first level that triggered
    reason: str


# ---------------------------------------------------------------------------
# StaleChecker
# ---------------------------------------------------------------------------

class StaleChecker:
    """Checks the freshness of a set of bucket IDs before they are queried.

    Args:
        registry:  The live BucketRegistry (read-only; provides per-bucket metadata).
        store:     The BucketStore (used for Level 3 to read chunk source-file paths).
        repo_path: The repository root (used for Level 2 subprocess call).
    """

    def __init__(
        self,
        registry: "BucketRegistry",
        store: "BucketStore",
        repo_path: Path,
    ) -> None:
        self._registry = registry
        self._store = store
        self._repo_path = repo_path

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def check(self, bucket_ids: List[str]) -> StaleCheckResult:
        """Check *bucket_ids* for staleness. Returns within ~50ms."""

        # ---- Level 1: watcher process alive? ---------------------------------
        watcher_pid = self._registry._meta.get("watcher_pid")
        if watcher_pid is not None:
            try:
                pid_int = int(watcher_pid)
            except (TypeError, ValueError):
                pid_int = None
            if pid_int is not None and not _process_is_alive(pid_int):
                log.warning(
                    "stale_checker.watcher_down",
                    pid=pid_int,
                    level=1,
                )
                return StaleCheckResult(
                    is_stale=True,
                    stale_bucket_ids=list(bucket_ids),
                    level=1,
                    reason=f"watcher process (PID {pid_int}) is not running",
                )

        # ---- Level 2: git HEAD drift? ----------------------------------------
        current_head = _get_current_head(self._repo_path)
        last_indexed_head: Optional[str] = self._registry._meta.get("last_indexed_head")  # type: ignore[assignment]

        if current_head is not None and last_indexed_head is not None:
            if current_head != last_indexed_head:
                log.warning(
                    "stale_checker.head_drift",
                    from_sha=last_indexed_head[:8],
                    to_sha=current_head[:8],
                    level=2,
                )
                return StaleCheckResult(
                    is_stale=True,
                    stale_bucket_ids=list(bucket_ids),
                    level=2,
                    reason=(
                        f"global HEAD drifted: "
                        f"{last_indexed_head[:8]} → {current_head[:8]}"
                    ),
                )

        # ---- Levels 3 & 4: per-bucket checks ---------------------------------
        stale_ids: List[str] = []
        first_trigger_level: Optional[int] = None

        for bucket_id in bucket_ids:
            entry = self._registry._buckets.get(bucket_id)
            if entry is None:
                continue

            # Level 3 — file mtime vs last_indexed_at
            if entry.last_indexed_at is not None:
                try:
                    indexed_ts = datetime.fromisoformat(entry.last_indexed_at).timestamp()
                    if self._any_source_file_newer(bucket_id, indexed_ts):
                        stale_ids.append(bucket_id)
                        if first_trigger_level is None:
                            first_trigger_level = 3
                        log.warning(
                            "stale_checker.file_newer_than_index",
                            bucket_id=bucket_id,
                            level=3,
                        )
                        continue
                except (ValueError, TypeError):
                    pass  # malformed timestamp — skip Level 3 for this bucket

            # Level 4 — per-bucket index_head_sha vs current HEAD
            if (
                current_head is not None
                and entry.index_head_sha is not None
                and entry.index_head_sha != current_head
            ):
                stale_ids.append(bucket_id)
                if first_trigger_level is None:
                    first_trigger_level = 4
                log.warning(
                    "stale_checker.bucket_sha_drift",
                    bucket_id=bucket_id,
                    bucket_sha=entry.index_head_sha[:8],
                    current_sha=current_head[:8],
                    level=4,
                )

        if stale_ids:
            level = first_trigger_level or 3
            return StaleCheckResult(
                is_stale=True,
                stale_bucket_ids=stale_ids,
                level=level,
                reason=f"level-{level} staleness in {len(stale_ids)} bucket(s)",
            )

        return StaleCheckResult(
            is_stale=False,
            stale_bucket_ids=[],
            level=0,
            reason="all queried buckets are fresh",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _any_source_file_newer(self, bucket_id: str, indexed_ts: float) -> bool:
        """Return True if any source file in *bucket_id* has mtime > *indexed_ts*."""
        try:
            front_matter, _ = self._store.read(bucket_id)
        except FileNotFoundError:
            return False

        seen: set[str] = set()
        for chunk in front_matter.chunks:
            src = chunk.source_file
            if src in seen:
                continue
            seen.add(src)
            mtime = _get_file_mtime(src)
            if mtime is not None and mtime > indexed_ts:
                return True
        return False
