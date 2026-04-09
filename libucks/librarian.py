"""Librarian — per-bucket async event handler (Phase 4 full implementation).

Responsibilities per event type:
  UpdateEvent      — reason(diff, prose) → write prose → recompute centroid
  TombstoneEvent   — strip purged chunk_ids from front-matter → re-render prose
  PathUpdateEvent  — update source_file on all matching ChunkMetadata (no AI call)
  QueryEvent       — reason(query, prose) → return Representation (no decode)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import structlog

from libucks.models.chunk import ChunkMetadata
from libucks.models.events import (
    PathUpdateEvent,
    QueryEvent,
    TombstoneEvent,
    UpdateEvent,
)
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.thinking.base import Representation, ThinkingStrategy

if TYPE_CHECKING:
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.mitosis import MitosisService

log = structlog.get_logger(__name__)


def _encode_centroid(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _read_chunk_content(meta: ChunkMetadata) -> str:
    """Read lines [start_line, end_line] from the chunk's source file."""
    try:
        lines = Path(meta.source_file).read_text(errors="replace").splitlines()
        return "\n".join(lines[meta.start_line - 1 : meta.end_line])
    except OSError:
        return ""


def _collect_source_text(front_matter, max_chars: int = 3000) -> str:
    """Concatenate actual code content from ChunkMetadata, up to max_chars."""
    parts = []
    total = 0
    for meta in front_matter.chunks:
        content = _read_chunk_content(meta)
        if not content:
            continue
        block = f"# {meta.source_file}\n{content}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n\n".join(parts)


def _get_head_sha(repo_path: Optional[Path]) -> str:
    """Return the current git HEAD SHA, or 'unknown' if unavailable."""
    if repo_path is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Librarian:
    def __init__(
        self,
        bucket_id: str,
        store: Optional[BucketStore] = None,
        registry: Optional[BucketRegistry] = None,
        strategy: Optional[ThinkingStrategy] = None,
        embedder: Optional["EmbeddingService"] = None,
        mitosis_threshold: int = 20_000,
        mitosis_service: Optional["MitosisService"] = None,
        repo_path: Optional[Path] = None,
    ) -> None:
        self.bucket_id = bucket_id
        self._store = store
        self._registry = registry
        self._strategy = strategy
        self._embedder = embedder
        self._mitosis_threshold = mitosis_threshold
        self._mitosis_service = mitosis_service
        self._repo_path = repo_path
        self.queue: asyncio.Queue[object] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle(self, event: object) -> Optional[Representation]:
        # Phase 2 stub path: no deps wired — just enqueue for test observation.
        if self._store is None or self._registry is None:
            log.debug("librarian.stub_enqueue", bucket_id=self.bucket_id, type=type(event).__name__)
            await self.queue.put(event)
            return None

        if isinstance(event, UpdateEvent):
            await self._handle_update(event)
        elif isinstance(event, TombstoneEvent):
            await self._handle_tombstone(event)
        elif isinstance(event, PathUpdateEvent):
            await self._handle_path_update(event)
        elif isinstance(event, QueryEvent):
            return await self._handle_query(event)
        else:
            log.warning("librarian.unknown_event", bucket_id=self.bucket_id, type=type(event).__name__)
        return None

    # ------------------------------------------------------------------
    # UpdateEvent
    # ------------------------------------------------------------------

    async def _handle_update(self, event: UpdateEvent) -> None:
        lock = self._registry.get_lock(self.bucket_id)
        async with lock:
            log.info("librarian.update.start", bucket_id=self.bucket_id)
            try:
                front_matter, current_prose = self._store.read(self.bucket_id)
            except FileNotFoundError:
                log.warning("librarian.update.bucket_missing", bucket_id=self.bucket_id)
                return

            diff_content = "\n".join(event.hunk.added_lines)
            prompt = (
                f"A code diff has been applied. Update the bucket summary to incorporate these changes.\n\n"
                f"Diff (added lines):\n{diff_content}"
            )

            try:
                result = await self._strategy.reason(prompt, current_prose)
                updated_prose = str(result)
            except Exception as exc:
                log.warning("librarian.update.reason_failed", bucket_id=self.bucket_id, error=str(exc))
                updated_prose = current_prose + f"\n\n<!-- update: {diff_content[:200]} -->"

            self._store.write_prose(self.bucket_id, updated_prose)
            log.info("librarian.update.prose_written", bucket_id=self.bucket_id)

            # Recompute centroid from chunk contents.
            chunk_contents = [_read_chunk_content(c) for c in front_matter.chunks]
            chunk_contents = [c for c in chunk_contents if c]
            if chunk_contents:
                embeddings = self._embedder.embed_batch(chunk_contents)
                centroid = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
            else:
                centroid = self._embedder.embed(updated_prose)

            # Blend title embedding into centroid (α=0.2) so the bucket's vector
            # identity stays anchored to its semantic aspect label.
            title_embed = self._embedder.embed(front_matter.domain_label)
            centroid = 0.8 * centroid + 0.2 * title_embed
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            new_token_count = sum(c.token_count for c in front_matter.chunks)

            await self._registry.register(
                self.bucket_id, centroid.astype(np.float32), new_token_count
            )

            # Phase 6-A: stamp chunks from the updated file with current HEAD SHA
            # and an indexed_at timestamp, then persist and save registry.
            head_sha = _get_head_sha(self._repo_path)
            now = _now_iso()
            updated_file = event.hunk.file
            for chunk in front_matter.chunks:
                # Match on suffix to handle relative vs absolute path variations.
                if chunk.source_file == updated_file or chunk.source_file.endswith(updated_file):
                    chunk.git_sha = head_sha
                    chunk.indexed_at = now

            front_matter.last_indexed_at = now
            front_matter.index_head_sha = head_sha
            self._store.write_front_matter(self.bucket_id, front_matter)

            self._registry.update_index_timestamp(self.bucket_id, now, head_sha)
            self._registry.save()

            log.info(
                "librarian.update.done",
                bucket_id=self.bucket_id,
                token_count=new_token_count,
                head_sha=head_sha,
            )

        # Check mitosis outside lock.
        if new_token_count > self._mitosis_threshold and self._mitosis_service:
            log.info("librarian.mitosis_triggered", bucket_id=self.bucket_id, token_count=new_token_count)
            asyncio.ensure_future(self._mitosis_service.split(self.bucket_id))

    # ------------------------------------------------------------------
    # TombstoneEvent
    # ------------------------------------------------------------------

    async def _handle_tombstone(self, event: TombstoneEvent) -> None:
        lock = self._registry.get_lock(self.bucket_id)
        async with lock:
            log.info("librarian.tombstone.start", bucket_id=self.bucket_id, chunk_ids=event.chunk_ids)
            try:
                front_matter, current_prose = self._store.read(self.bucket_id)
            except FileNotFoundError:
                return

            surviving = [c for c in front_matter.chunks if c.chunk_id not in event.chunk_ids]
            if len(surviving) == len(front_matter.chunks):
                return  # Nothing to purge.

            front_matter.chunks = surviving
            front_matter.token_count = sum(c.token_count for c in surviving)
            self._store.write_front_matter(self.bucket_id, front_matter)

            if surviving:
                purged_count = len(front_matter.chunks) - len(surviving) + (len(front_matter.chunks) - len(surviving))
                prompt = (
                    f"Some code has been deleted. Rewrite the summary omitting the deleted concepts. "
                    f"Chunk IDs removed: {event.chunk_ids}"
                )
                try:
                    result = await self._strategy.reason(prompt, current_prose)
                    self._store.write_prose(self.bucket_id, str(result))
                except Exception as exc:
                    log.warning("librarian.tombstone.reason_failed", bucket_id=self.bucket_id, error=str(exc))

            await self._registry.register(
                self.bucket_id,
                self._registry.get_all_centroids().get(
                    self.bucket_id, np.zeros(384, dtype=np.float32)
                ),
                front_matter.token_count,
            )
            log.info("librarian.tombstone.done", bucket_id=self.bucket_id, surviving=len(surviving))

    # ------------------------------------------------------------------
    # PathUpdateEvent
    # ------------------------------------------------------------------

    async def _handle_path_update(self, event: PathUpdateEvent) -> None:
        lock = self._registry.get_lock(self.bucket_id)
        async with lock:
            try:
                front_matter, _ = self._store.read(self.bucket_id)
            except FileNotFoundError:
                return

            updated = False
            for chunk in front_matter.chunks:
                if chunk.source_file == event.old_path:
                    chunk.source_file = event.new_path
                    updated = True

            if updated:
                self._store.write_front_matter(self.bucket_id, front_matter)
                log.info(
                    "librarian.path_update.done",
                    bucket_id=self.bucket_id,
                    old=event.old_path,
                    new=event.new_path,
                )

    # ------------------------------------------------------------------
    # QueryEvent
    # ------------------------------------------------------------------

    async def _handle_query(self, event: QueryEvent) -> Representation:
        # Read-only — no lock required.
        try:
            front_matter, prose = self._store.read(self.bucket_id)
        except FileNotFoundError:
            return ""

        # Use actual code content (same as training); fall back to prose if unreadable
        source_text = _collect_source_text(front_matter, max_chars=3000) or prose

        log.info("librarian.query", bucket_id=self.bucket_id, query=event.query[:60])
        try:
            result = await self._strategy.reason(event.query, source_text)
        except Exception as exc:
            log.warning("librarian.query.reason_failed", bucket_id=self.bucket_id, error=str(exc))
            result = prose  # Fall back to raw prose.

        # Architectural rule: do NOT call decode() here.
        return result
