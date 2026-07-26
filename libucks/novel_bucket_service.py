"""NovelBucketService — drains create_bucket_queue, spawns new buckets for novel commits.

Producer: StartupRecovery (for the git-hook path) and CentralAgent._handle_added
(for the opt-in WatchdogService path) detect content that is both substantial
and cosine-distant from every existing centroid, then enqueue a CreateBucketEvent.

Consumer (this class): drains the queue, computes centroid via the same blend
formula as InitOrchestrator (0.8 chunk + 0.2 title), persists a single-chunk
bucket, and wires up a new Librarian on the CentralAgent. Best-effort prose
generation via strategy + translator; falls back to a placeholder if either is
absent or the generation raises.

bucket_id is derived from source identity (file + line range), not from content
text, so duplicate events for the same file region are idempotent — the second
enqueue is a no-op.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

from libucks.models.chunk import ChunkMetadata
from libucks.models.events import CreateBucketEvent
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_registry import encode_centroid as _encode_centroid
from libucks.storage.bucket_store import BucketStore

if TYPE_CHECKING:
    from libucks.central_agent import CentralAgent
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.thinking.base import ThinkingStrategy
    from libucks.translator import Translator

log = structlog.get_logger(__name__)

# Matches init_orchestrator: 0.8 chunk centroid + 0.2 title embedding.
_TITLE_BLEND_ALPHA = 0.2
_TOKENS_PER_CHAR = 0.25


def _rough_tokens(text: str) -> int:
    return max(1, int(len(text) * _TOKENS_PER_CHAR))


class NovelBucketService:
    def __init__(
        self,
        store: BucketStore,
        registry: BucketRegistry,
        embedder: "EmbeddingService",
        agent: "CentralAgent",
        strategy: Optional["ThinkingStrategy"] = None,
        translator: Optional["Translator"] = None,
        mitosis_threshold: int = 20_000,
        repo_path: Optional[Path] = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._agent = agent
        self._strategy = strategy
        self._translator = translator
        self._mitosis_threshold = mitosis_threshold
        self._repo_path = repo_path

    async def run(self) -> None:
        """Drain create_bucket_queue indefinitely."""
        log.info("novel_bucket.running")
        while True:
            event = await self._agent.create_bucket_queue.get()
            try:
                await self._create_bucket(event)
            except Exception as exc:
                log.warning("novel_bucket.failed", error=str(exc))
            finally:
                self._agent.create_bucket_queue.task_done()

    async def drain_pending(self) -> None:
        """Drain everything currently in the queue without blocking.

        Useful for tests that want to wait until all enqueued events are
        processed before asserting on registry state.
        """
        while not self._agent.create_bucket_queue.empty():
            event = await self._agent.create_bucket_queue.get()
            try:
                await self._create_bucket(event)
            except Exception as exc:
                log.warning("novel_bucket.failed", error=str(exc))
            finally:
                self._agent.create_bucket_queue.task_done()

    async def _create_bucket(self, event: CreateBucketEvent) -> None:
        content = event.seed_content
        if not content.strip():
            log.debug("novel_bucket.empty_content")
            return

        ident = (
            f"{event.source_file or 'novel'}"
            f":{event.start_line or 1}:{event.end_line or 1}"
        )
        bucket_id = hashlib.sha1(ident.encode()).hexdigest()[:8]

        # Idempotency: a duplicate event for the same file region is a no-op.
        if bucket_id in self._agent._librarians:
            log.debug("novel_bucket.already_exists", bucket_id=bucket_id)
            return

        chunk_centroid = self._embedder.embed(content).astype(np.float32)
        norm = float(np.linalg.norm(chunk_centroid))
        if norm > 0:
            chunk_centroid /= norm

        domain_label = (
            Path(event.source_file).stem
            if event.source_file
            else f"novel-{bucket_id}"
        )
        title_embed = self._embedder.embed(domain_label).astype(np.float32)
        centroid = (
            (1 - _TITLE_BLEND_ALPHA) * chunk_centroid
            + _TITLE_BLEND_ALPHA * title_embed
        )
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm

        prose = await self._generate_prose(content, domain_label)

        token_count = _rough_tokens(content)
        chunk_meta = ChunkMetadata(
            chunk_id=hashlib.sha1(ident.encode()).hexdigest()[:12],
            source_file=event.source_file or "<novel>",
            start_line=event.start_line or 1,
            end_line=event.end_line or 1,
            git_sha="novel",
            token_count=token_count,
        )

        self._store.create(
            bucket_id=bucket_id,
            domain_label=domain_label,
            centroid=_encode_centroid(centroid),
            chunks=[chunk_meta],
            prose=prose,
        )
        await self._registry.register(bucket_id, centroid, token_count)

        # Wire up a Librarian for the new bucket so subsequent queries route to it.
        from libucks.librarian import Librarian  # local import avoids circular dep
        lib = Librarian(
            bucket_id=bucket_id,
            store=self._store,
            registry=self._registry,
            strategy=self._strategy,
            embedder=self._embedder,
            mitosis_threshold=self._mitosis_threshold,
            repo_path=self._repo_path,
            translator=self._translator,
        )
        self._agent.register_librarian(bucket_id, lib)
        self._registry.save()

        log.info(
            "novel_bucket.created",
            bucket_id=bucket_id,
            domain=domain_label,
            source_file=event.source_file,
            tokens=token_count,
        )

    async def _generate_prose(self, content: str, domain_label: str) -> str:
        if self._strategy is None:
            return f"# {domain_label}\n\nNew bucket seeded from novel commit content.\n"
        try:
            prompt = f"Write a concise technical summary for: {domain_label}"
            latent = await self._strategy.reason(prompt, content[:2000])
            if self._translator is not None:
                return await self._translator.synthesize("", [latent])
            return f"# {domain_label}\n\n{content[:500]}"
        except Exception as exc:
            log.warning("novel_bucket.prose_failed", error=str(exc))
            return f"# {domain_label}\n\n{content[:500]}"
