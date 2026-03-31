"""CentralAgent — embedding-based router and async event dispatcher.

Step 2.1 — routing math:
  route()    — returns bucket_ids sorted by cosine similarity descending.
  is_novel() — True if query cosine distance > novelty_threshold from all centroids.

Step 2.2 / Phase 4 — event loop:
  post(DiffEvent)   — enqueue an incoming diff for processing.
  run()             — continuous loop; process DiffEvents until cancelled.
  run_once()        — dequeue and process exactly one DiffEvent.
  clear_splitting() — clear the mitosis flag and drain the retry buffer.

Routing uses dot products because all centroids and query embeddings are
L2-normalised (dot product == cosine similarity for unit vectors).

embed_fn is injected so tests can drive routing deterministically without
loading a real embedding model.  Production code passes
EmbeddingService.get_instance().embed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import structlog

from libucks.config import Config
from libucks.models.events import (
    CreateBucketEvent,
    DiffEvent,
    DiffHunk,
    PathUpdateEvent,
    TombstoneEvent,
    UpdateEvent,
)
from libucks.storage.bucket_registry import BucketRegistry

if TYPE_CHECKING:
    from libucks.librarian import Librarian

log = structlog.get_logger(__name__)


class CentralAgent:
    def __init__(
        self,
        registry: BucketRegistry,
        config: Config,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self._registry = registry
        self._config = config
        self._embed_fn = embed_fn

        self._diff_queue: asyncio.Queue[DiffEvent] = asyncio.Queue()
        self.create_bucket_queue: asyncio.Queue[CreateBucketEvent] = asyncio.Queue()
        self._librarians: Dict[str, "Librarian"] = {}
        # Each entry is (bucket_id, UpdateEvent) — drained when splitting clears.
        self.retry_buffer: List[Tuple[str, UpdateEvent]] = []

    @property
    def diff_queue(self) -> asyncio.Queue[DiffEvent]:
        return self._diff_queue

    def register_librarian(self, bucket_id: str, librarian: "Librarian") -> None:
        self._librarians[bucket_id] = librarian
        log.debug("central_agent.librarian_registered", bucket_id=bucket_id)

    def unregister_librarian(self, bucket_id: str) -> None:
        self._librarians.pop(bucket_id, None)

    # ------------------------------------------------------------------
    # Routing math (Step 2.1)
    # ------------------------------------------------------------------

    def route(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        """Return up to top_k bucket_ids by cosine similarity descending."""
        centroids = self._registry.get_all_centroids()
        if not centroids:
            return []
        bucket_ids = list(centroids.keys())
        matrix = np.stack([centroids[bid] for bid in bucket_ids])  # (N, dim)
        similarities = matrix @ query_embedding                      # (N,)
        k = min(top_k, len(bucket_ids))
        top_indices = np.argsort(similarities)[::-1][:k]
        return [bucket_ids[int(i)] for i in top_indices]

    def is_novel(self, query_embedding: np.ndarray) -> bool:
        """True if max cosine similarity < (1 - novelty_threshold)."""
        centroids = self._registry.get_all_centroids()
        if not centroids:
            return True
        matrix = np.stack(list(centroids.values()))
        max_similarity = float(np.max(matrix @ query_embedding))
        return max_similarity < (1.0 - self._config.routing.novelty_threshold)

    # ------------------------------------------------------------------
    # Event loop (Step 2.2 / Phase 4)
    # ------------------------------------------------------------------

    async def post(self, event: DiffEvent) -> None:
        """Enqueue a DiffEvent for processing."""
        await self._diff_queue.put(event)
        log.debug("central_agent.event_queued", file=event.file, qsize=self._diff_queue.qsize())

    async def run(self) -> None:
        """Continuously process DiffEvents until cancelled."""
        log.info("central_agent.running")
        while True:
            await self.run_once()

    async def run_once(self) -> None:
        """Dequeue and process exactly one DiffEvent."""
        event = await self._diff_queue.get()
        log.info("central_agent.processing", file=event.file, hunks=len(event.hunks))
        try:
            await self._process(event)
        finally:
            self._diff_queue.task_done()

    async def clear_splitting(self, bucket_id: str) -> None:
        """Clear the mitosis flag for bucket_id and drain its retry buffer."""
        await self._registry.set_splitting(bucket_id, False)
        to_drain = [(bid, ev) for bid, ev in self.retry_buffer if bid == bucket_id]
        self.retry_buffer = [(bid, ev) for bid, ev in self.retry_buffer if bid != bucket_id]
        for _, ev in to_drain:
            if ev.bucket_id in self._librarians:
                log.info("central_agent.retry_drain", bucket_id=ev.bucket_id)
                await self._librarians[ev.bucket_id].handle(ev)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _process(self, event: DiffEvent) -> None:
        if event.is_rename:
            await self._handle_rename(event)
            return
        for hunk in event.hunks:
            if hunk.added_lines:
                await self._handle_added(hunk)
            if hunk.removed_lines:
                await self._handle_removed(hunk)

    async def _handle_rename(self, event: DiffEvent) -> None:
        path_event = PathUpdateEvent(
            old_path=event.old_path or "",
            new_path=event.new_path or "",
        )
        log.info("central_agent.rename", old=event.old_path, new=event.new_path)
        for librarian in self._librarians.values():
            await librarian.handle(path_event)

    async def _handle_added(self, hunk: DiffHunk) -> None:
        content = "\n".join(hunk.added_lines)
        embedding = self._embed(content)

        if self.is_novel(embedding):
            log.info("central_agent.novel_diff", lines=len(hunk.added_lines))
            await self.create_bucket_queue.put(CreateBucketEvent(seed_content=content))
            return

        top = self.route(embedding, top_k=1)
        if not top:
            return
        bucket_id = top[0]
        update = UpdateEvent(bucket_id=bucket_id, hunk=hunk)

        if self._registry.is_splitting(bucket_id):
            log.info("central_agent.buffered_splitting", bucket_id=bucket_id)
            self.retry_buffer.append((bucket_id, update))
        elif bucket_id in self._librarians:
            log.info("central_agent.routed", bucket_id=bucket_id, added=len(hunk.added_lines))
            await self._librarians[bucket_id].handle(update)

    async def _handle_removed(self, hunk: DiffHunk) -> None:
        content = "\n".join(hunk.removed_lines)
        embedding = self._embed(content)

        top = self.route(embedding, top_k=self._config.routing.top_k)
        bucket_ids = top if top else list(self._librarians.keys())

        tombstone = TombstoneEvent(chunk_ids=[], bucket_ids=bucket_ids)
        log.info("central_agent.tombstone", buckets=bucket_ids, removed=len(hunk.removed_lines))
        for bid in bucket_ids:
            if bid in self._librarians:
                await self._librarians[bid].handle(tombstone)

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(text)
        return np.zeros(384, dtype=np.float32)
