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
    ) -> None:
        self.bucket_id = bucket_id
        self._store = store
        self._registry = registry
        self._strategy = strategy
        self._embedder = embedder
        self._mitosis_threshold = mitosis_threshold
        self._mitosis_service = mitosis_service
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

            new_token_count = sum(c.token_count for c in front_matter.chunks)

            await self._registry.register(
                self.bucket_id, centroid.astype(np.float32), new_token_count
            )
            log.info(
                "librarian.update.done",
                bucket_id=self.bucket_id,
                token_count=new_token_count,
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
            _, prose = self._store.read(self.bucket_id)
        except FileNotFoundError:
            return ""

        log.info("librarian.query", bucket_id=self.bucket_id, query=event.query[:60])
        try:
            result = await self._strategy.reason(event.query, prose)
        except Exception as exc:
            log.warning("librarian.query.reason_failed", bucket_id=self.bucket_id, error=str(exc))
            result = prose  # Fall back to raw prose.

        # Architectural rule: do NOT call decode() here.
        return result
