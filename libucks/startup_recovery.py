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
from libucks.models.events import CreateBucketEvent, DiffHunk, UpdateEvent
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.watchdog_service import _TRACKED_EXTENSIONS

if TYPE_CHECKING:
    from libucks.central_agent import CentralAgent
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.librarian import Librarian


# Cap unmatched-file content read for novelty embedding so a huge new file
# doesn't dominate startup; same order-of-magnitude as InitOrchestrator's
# _MAX_FILE_BYTES guard.
_UNMATCHED_READ_CAP = 50_000
_TOKENS_PER_CHAR = 0.25

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
        log.warning("git_rev_parse_head_failed", repo=str(repo_path),
                    returncode=result.returncode, stderr=result.stderr.strip()[:200])
    except Exception as exc:
        log.warning("git_rev_parse_head_errored", repo=str(repo_path), error=repr(exc))
    return None


def _git_show_toplevel(repo_path: Path) -> Optional[Path]:
    """Return the git repository root for repo_path.

    libucks repo_path may be a subdirectory of the git root (e.g. click's
    `src/click/` inside the `click/` checkout). git diff --name-only returns
    paths relative to the root, not to repo_path, so all path arithmetic that
    resolves diff paths to absolute paths MUST be anchored to the root.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        log.warning("git_show_toplevel_failed", repo=str(repo_path),
                    returncode=result.returncode, stderr=result.stderr.strip()[:200])
    except Exception as exc:
        log.warning("git_show_toplevel_errored", repo=str(repo_path), error=repr(exc))
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
        log.warning(
            "git_diff_failed",
            from_sha=from_sha, to_sha=to_sha, returncode=result.returncode,
            stderr=result.stderr.strip()[:200],
            consequence="treated as no-changes; bucket updates will be SKIPPED",
        )
    except Exception as exc:
        # Returning [] here is indistinguishable from "no files changed", so a
        # silent failure means the whole update pipeline quietly does nothing.
        log.warning(
            "git_diff_errored",
            from_sha=from_sha, to_sha=to_sha, error=repr(exc),
            consequence="treated as no-changes; bucket updates will be SKIPPED",
        )
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
        central_agent: Optional["CentralAgent"] = None,
        embedder: Optional["EmbeddingService"] = None,
        min_bucket_seed_tokens: int = 1_500,
    ) -> None:
        self._repo_path = repo_path
        self._registry = registry
        self._store = store
        self._librarians = librarians
        self._extractor = extractor
        # central_agent + embedder enable novelty detection on unmatched files.
        # Both are optional so existing unit tests that don't exercise the
        # novel-bucket path can construct StartupRecovery without them.
        self._central_agent = central_agent
        self._embedder = embedder
        self._min_bucket_seed_tokens = int(min_bucket_seed_tokens)
        # git diff returns paths relative to the git root, not repo_path. When
        # libucks is anchored at a subdirectory of the git root (e.g. click's
        # src/click/), naive `repo_path / rel_filepath` doubles a path component
        # and breaks every match. Cache the root once and resolve against it.
        _root = _git_show_toplevel(repo_path)
        self._git_root: Path = _root if _root is not None else repo_path

    def _find_buckets_for_file(self, rel_filepath: str) -> List[str]:
        """Return bucket IDs that own at least one chunk from the given file.

        Matching is done by resolving both paths to absolute form so that
        relative-vs-absolute mismatches (common when mixing git output with
        stored absolute paths) do not cause missed updates.

        rel_filepath is resolved against the git root, not repo_path, because
        git diff --name-only emits paths relative to the root.
        """
        try:
            abs_target = (self._git_root / rel_filepath).resolve()
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
                # Unmatched: file not owned by any existing bucket. Decide
                # whether to spawn a new bucket (substantial + novel content)
                # or route the content to the nearest existing bucket.
                routed = await self._handle_unmatched_file(rel_filepath)
                if routed:
                    recovered_updates += 1
                else:
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

    async def _handle_unmatched_file(self, rel_filepath: str) -> bool:
        """Handle a changed file that no existing bucket owns.

        Returns True if the file was dispatched (either enqueued as a new
        bucket or routed to the nearest existing one); False if it was
        skipped (deleted, empty, or no central_agent/embedder injected).
        """
        if self._central_agent is None or self._embedder is None:
            return False

        # rel_filepath is git-root-relative; resolve against the cached root.
        abs_path = self._git_root / rel_filepath
        if not abs_path.is_file():
            # Deleted file or stat failure — nothing to route.
            return False

        try:
            content = abs_path.read_text(errors="replace")
        except Exception as exc:
            log.warning("startup_recovery.unmatched_read_failed", file=rel_filepath, error=str(exc))
            return False

        if not content.strip():
            return False

        sample = content[:_UNMATCHED_READ_CAP]
        try:
            embedding = self._embedder.embed(sample)
        except Exception as exc:
            log.warning("startup_recovery.unmatched_embed_failed", file=rel_filepath, error=str(exc))
            return False

        est_tokens = max(1, int(len(content) * _TOKENS_PER_CHAR))
        line_count = max(1, content.count("\n") + 1)

        # Spawn a new bucket only for substantial novel subjects. Smaller or
        # cosine-near content is absorbed into the nearest existing bucket;
        # HealthMonitor's coherence-driven mitosis will split if pressure
        # builds. Prevents one-file fragmentation.
        if (
            est_tokens >= self._min_bucket_seed_tokens
            and self._central_agent.is_novel(embedding)
        ):
            event = CreateBucketEvent(
                seed_content=sample,
                source_file=rel_filepath,
                start_line=1,
                end_line=line_count,
            )
            await self._central_agent.create_bucket_queue.put(event)
            log.info(
                "startup_recovery.novel_file_queued",
                file=rel_filepath,
                tokens=est_tokens,
                lines=line_count,
            )
            return True

        top = self._central_agent.route(embedding, top_k=1)
        if not top or top[0] not in self._librarians:
            return False

        bucket_id = top[0]
        hunk = DiffHunk(
            file=rel_filepath,
            old_start=0,
            old_end=0,
            new_start=1,
            new_end=line_count,
            added_lines=sample.splitlines(),
            removed_lines=[],
        )
        await self._librarians[bucket_id].handle(UpdateEvent(bucket_id=bucket_id, hunk=hunk))
        log.info(
            "startup_recovery.unmatched_routed",
            file=rel_filepath,
            bucket_id=bucket_id,
            tokens=est_tokens,
        )
        return True
