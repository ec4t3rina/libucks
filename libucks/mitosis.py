"""MitosisService — splits an oversized bucket into two children using k-means.

Process (V1, manual trigger only — no automatic clustering):
  1. set_splitting(True) in registry to block new updates.
  2. Re-embed all chunks in the bucket.
  3. k-means k=2 (scikit-learn) on chunk embeddings.
  4. Create two child BucketStore entries.
  5. Instantiate two new Librarians and register them with CentralAgent.
  6. Deregister parent from registry + delete from store.
  7. clear_splitting() → CentralAgent drains retry buffer, re-routes events.

Invariant: len(child_A.chunks) + len(child_B.chunks) == len(parent.chunks).
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import structlog
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from libucks.models.chunk import ChunkMetadata
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore

if TYPE_CHECKING:
    from libucks.central_agent import CentralAgent
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.librarian import Librarian
    from libucks.thinking.base import ThinkingStrategy

log = structlog.get_logger(__name__)


def _encode_centroid(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _child_id(parent_id: str, label: int) -> str:
    raw = f"{parent_id}:child{label}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _read_chunk_content(meta: ChunkMetadata) -> str:
    try:
        lines = Path(meta.source_file).read_text(errors="replace").splitlines()
        return "\n".join(lines[meta.start_line - 1 : meta.end_line])
    except OSError:
        return meta.source_file  # fallback: just use filename as content


class MitosisService:
    def __init__(
        self,
        store: BucketStore,
        registry: BucketRegistry,
        embedder: "EmbeddingService",
        agent: "CentralAgent",
        strategy: "ThinkingStrategy",
        mitosis_threshold: int = 20_000,
    ) -> None:
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._agent = agent
        self._strategy = strategy
        self._mitosis_threshold = mitosis_threshold

    async def split(self, bucket_id: str) -> None:
        """Split *bucket_id* into two child buckets."""
        log.info("mitosis.start", bucket_id=bucket_id)

        # 1. Guard.
        await self._registry.set_splitting(bucket_id, True)

        try:
            front_matter, prose = self._store.read(bucket_id)
        except FileNotFoundError:
            log.warning("mitosis.bucket_missing", bucket_id=bucket_id)
            await self._registry.set_splitting(bucket_id, False)
            return

        chunks: List[ChunkMetadata] = front_matter.chunks
        if len(chunks) < 2:
            log.info("mitosis.too_few_chunks", bucket_id=bucket_id, count=len(chunks))
            await self._registry.set_splitting(bucket_id, False)
            return

        # 2. Re-embed all chunks.
        contents = [_read_chunk_content(c) for c in chunks]
        embeddings = self._embedder.embed_batch(contents)  # (N, D)

        # 3. k-means k=2.
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(embeddings)

        groups: List[List[int]] = [[], []]
        for idx, label in enumerate(labels):
            groups[int(label)].append(idx)

        # Edge case: all chunks assigned to one cluster — force a split.
        if not groups[0] or not groups[1]:
            mid = len(chunks) // 2
            groups = [list(range(mid)), list(range(mid, len(chunks)))]

        log.info(
            "mitosis.clustered",
            bucket_id=bucket_id,
            group_a=len(groups[0]),
            group_b=len(groups[1]),
        )

        child_ids: List[str] = []
        for label, indices in enumerate(groups):
            child_id = _child_id(bucket_id, label)
            child_chunks = [chunks[i] for i in indices]
            child_embeddings = embeddings[indices]

            centroid = np.mean(child_embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Derive a domain label from file stems in this group.
            stems = sorted({Path(c.source_file).stem for c in child_chunks})
            domain = f"{front_matter.domain_label} / {', '.join(stems[:3])}"

            # Generate prose for child via ThinkingStrategy (best-effort).
            child_content = "\n".join(_read_chunk_content(c) for c in child_chunks)
            prompt = f"Write a concise technical summary for these code chunks: domain={domain}"
            try:
                result = await self._strategy.reason(prompt, child_content[:2000])
                child_prose = str(result)
            except Exception as exc:
                log.warning("mitosis.reason_failed", child_id=child_id, error=str(exc))
                child_prose = f"# {domain}\n\n{child_content[:500]}"

            self._store.create(
                bucket_id=child_id,
                domain_label=domain,
                centroid=_encode_centroid(centroid),
                chunks=child_chunks,
                prose=child_prose,
            )
            token_count = sum(c.token_count for c in child_chunks)
            await self._registry.register(child_id, centroid.astype(np.float32), token_count)

            # Wire up a new Librarian for this child.
            from libucks.librarian import Librarian  # avoid circular import at module level

            child_librarian = Librarian(
                bucket_id=child_id,
                store=self._store,
                registry=self._registry,
                strategy=self._strategy,
                embedder=self._embedder,
                mitosis_threshold=self._mitosis_threshold,
                mitosis_service=self,
            )
            self._agent.register_librarian(child_id, child_librarian)
            child_ids.append(child_id)
            log.info("mitosis.child_created", child_id=child_id, domain=domain, chunks=len(child_chunks))

        # 6. Deregister parent.
        self._agent.unregister_librarian(bucket_id)
        try:
            await self._registry.deregister(bucket_id)
        except KeyError:
            pass
        try:
            self._store.delete(bucket_id)
        except FileNotFoundError:
            pass

        # 7. Clear splitting flag — CentralAgent drains retry buffer.
        # Guard with try/except: if the parent was deregistered in step 6 above
        # (e.g. it no longer exists in the registry), clearing its flag is a no-op.
        try:
            await self._agent.clear_splitting(bucket_id)
        except KeyError:
            pass
        self._registry.save()

        log.info("mitosis.complete", parent=bucket_id, children=child_ids)
