"""StartupRecovery — replay commits that arrived while libucks serve was offline.

Algorithm (runs once, synchronously, before the MCP stdio server starts):

  1. Read registry._meta["last_indexed_head"] — the HEAD SHA at last save.
  2. Run `git rev-parse HEAD` — the current HEAD.
  3. If they differ (gap detected):
       a. `git diff --name-only <last> <current>` — which files changed.
       b. For each file whose extension is tracked:
            - Resolve which bucket(s) own chunks from that file.
            - `DiffExtractor.extract_between(file, last, current)` — get the diff.
            - For each hunk, call `librarian.handle(UpdateEvent(…))`.
  4. Always return the current HEAD so the caller can update the baseline.

If `last_indexed_head` is None (first run after `libucks init`), no recovery is
attempted — the index was just built by INIT, so it is already current.  The
current HEAD is still returned so the caller can record it as the new baseline.

The two module-level git helpers (_git_rev_parse_head, _git_diff_name_only) are
deliberately extracted so tests can patch them without mocking subprocess globally.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import structlog

from libucks.diff.diff_extractor import DiffExtractor
from libucks.models.events import UpdateEvent
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.watchdog_service import _TRACKED_EXTENSIONS

if TYPE_CHECKING:
    from libucks.librarian import Librarian

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Injectable git helpers (patched in unit tests)
# ---------------------------------------------------------------------------

def _git_rev_parse_head(repo_path: Path) -> Optional[str]:
    """Return current git HEAD SHA, or None if git is unavailable."""
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


def _git_diff_name_only(repo_path: Path, from_sha: str, to_sha: str) -> List[str]:
    """Return list of repo-relative file paths changed between two SHAs."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--name-only", from_sha, to_sha],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().splitlines() if f]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# StartupRecovery
# ---------------------------------------------------------------------------

class StartupRecovery:
    def __init__(
        self,
        repo_path: Path,
        registry: BucketRegistry,
        store: BucketStore,
        librarians: Dict[str, "Librarian"],
        extractor: DiffExtractor,
    ) -> None:
        self._repo_path = repo_path
        self._registry = registry
        self._store = store
        self._librarians = librarians
        self._extractor = extractor

    def _find_buckets_for_file(self, rel_filepath: str) -> List[str]:
        """Return bucket IDs that own at least one chunk from the given file.

        Matching is done by resolving both paths to absolute form so that
        relative-vs-absolute mismatches (common when mixing git output with
        stored absolute paths) do not cause missed updates.
        """
        try:
            abs_target = (self._repo_path / rel_filepath).resolve()
        except Exception:
            return []

        matched: List[str] = []
        for bucket_id in self._store.list_all():
            try:
                front_matter, _ = self._store.read(bucket_id)
            except FileNotFoundError:
                continue
            for chunk in front_matter.chunks:
                try:
                    chunk_abs = Path(chunk.source_file).resolve()
                except Exception:
                    continue
                if chunk_abs == abs_target:
                    matched.append(bucket_id)
                    break  # one match per bucket is enough

        return matched

    async def run(self) -> Optional[str]:
        """Replay any commits missed while the server was offline.

        Returns the current git HEAD SHA if git is reachable, None otherwise.
        The caller MUST write this value into registry._meta["last_indexed_head"]
        and call registry.save() so the next startup has an accurate baseline.
        """
        current_head = _git_rev_parse_head(self._repo_path)
        if not current_head:
            log.warning("startup_recovery.git_unavailable", repo=str(self._repo_path))
            return None

        last_head: Optional[str] = self._registry._meta.get("last_indexed_head")

        if not last_head:
            log.info(
                "startup_recovery.no_baseline",
                current_head=current_head[:8],
                note="recording baseline; no recovery needed",
            )
            return current_head

        if last_head == current_head:
            log.info("startup_recovery.up_to_date", head=current_head[:8])
            return current_head

        log.info(
            "startup_recovery.gap_detected",
            from_sha=last_head[:8],
            to_sha=current_head[:8],
        )

        changed_files = _git_diff_name_only(self._repo_path, last_head, current_head)
        log.info("startup_recovery.changed_files_count", count=len(changed_files))

        recovered_updates = 0
        for rel_filepath in changed_files:
            suffix = Path(rel_filepath).suffix.lower()
            if suffix not in _TRACKED_EXTENSIONS:
                log.debug("startup_recovery.skip_extension", file=rel_filepath, suffix=suffix)
                continue

            bucket_ids = self._find_buckets_for_file(rel_filepath)
            if not bucket_ids:
                log.debug("startup_recovery.no_bucket_for_file", file=rel_filepath)
                continue

            try:
                diff_events = self._extractor.extract_between(
                    self._repo_path / rel_filepath,
                    last_head,
                    current_head,
                )
            except Exception as exc:
                log.warning(
                    "startup_recovery.extract_failed",
                    file=rel_filepath,
                    error=str(exc),
                )
                continue

            if not diff_events:
                log.debug("startup_recovery.empty_diff", file=rel_filepath)
                continue

            for bucket_id in bucket_ids:
                librarian = self._librarians.get(bucket_id)
                if librarian is None:
                    continue
                for diff_event in diff_events:
                    for hunk in diff_event.hunks:
                        update = UpdateEvent(bucket_id=bucket_id, hunk=hunk)
                        await librarian.handle(update)
                        recovered_updates += 1

        log.info(
            "startup_recovery.complete",
            recovered_updates=recovered_updates,
            to_sha=current_head[:8],
        )
        return current_head
