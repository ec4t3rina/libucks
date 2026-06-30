"""Per-chunk embedding cache + query-time scoring (Phase 3-B).

Embeds each ChunkMetadata's content with the same sentence-transformer as
bucket centroids, caches on disk per bucket, and exposes:
  - score_chunks(bucket_id, q) → [(meta, cos)] sorted desc — used by
    Translator._gather_verbatim to pick chunks by query relevance.
  - max_chunk_score(bucket_id, q) → float — used by CentralAgent.route to
    rerank a centroid-pre-filtered candidate set by best-chunk-cos.

Cache layout:
    <repo>/.libucks/chunk_emb_cache/<bucket_id>.pt
A single pickle per bucket, structure:
    {chunk_id: {"git_sha": str, "embedding": np.ndarray(float32, L2-norm)}}
Entries are refreshed when meta.git_sha differs from the cached value
(i.e. chunk content changed since last embed). Stale entries are pruned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from libucks.embeddings.embedding_service import EmbeddingService
from libucks.models.chunk import ChunkMetadata
from libucks.storage.bucket_store import BucketStore


def _read_chunk_content(meta: ChunkMetadata) -> str:
    try:
        lines = Path(meta.source_file).read_text(errors="replace").splitlines()
        return "\n".join(lines[meta.start_line - 1 : meta.end_line])
    except OSError:
        return ""


class ChunkRetriever:
    def __init__(
        self,
        cache_dir: Path,
        embedder: EmbeddingService,
        store: BucketStore,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._store = store
        self._mem_cache: Dict[str, Dict[str, Dict[str, object]]] = {}

    def score_chunks(
        self,
        bucket_id: str,
        query_embedding: np.ndarray,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Return [(meta, cos)] for this bucket's chunks, descending by cos."""
        chunks = self._chunks_for_bucket(bucket_id)
        cache = self._get_or_build(bucket_id, chunks)
        q = query_embedding.astype(np.float32)
        results: List[Tuple[ChunkMetadata, float]] = []
        for meta in chunks:
            entry = cache.get(meta.chunk_id)
            if entry is None:
                continue
            emb = entry["embedding"]
            results.append((meta, float(q @ emb)))
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    def max_chunk_score(self, bucket_id: str, query_embedding: np.ndarray) -> float:
        """Return the best chunk-cos for this bucket against the query."""
        scored = self.score_chunks(bucket_id, query_embedding)
        return scored[0][1] if scored else 0.0

    def _chunks_for_bucket(self, bucket_id: str) -> List[ChunkMetadata]:
        try:
            fm, _ = self._store.read(bucket_id)
        except FileNotFoundError:
            return []
        return list(fm.chunks)

    def _get_or_build(
        self,
        bucket_id: str,
        chunks: List[ChunkMetadata],
    ) -> Dict[str, Dict[str, object]]:
        if bucket_id in self._mem_cache:
            cached = self._mem_cache[bucket_id]
        else:
            cached = self._load_from_disk(bucket_id)

        dirty = False
        live_ids = set()
        for meta in chunks:
            live_ids.add(meta.chunk_id)
            entry = cached.get(meta.chunk_id)
            if entry is not None and entry.get("git_sha") == meta.git_sha:
                continue
            content = _read_chunk_content(meta)
            if not content:
                continue
            emb = self._embedder.embed(content)
            cached[meta.chunk_id] = {"git_sha": meta.git_sha, "embedding": emb}
            dirty = True

        for stale in list(cached.keys()):
            if stale not in live_ids:
                del cached[stale]
                dirty = True

        if dirty:
            self._save_to_disk(bucket_id, cached)
        self._mem_cache[bucket_id] = cached
        return cached

    def _load_from_disk(self, bucket_id: str) -> Dict[str, Dict[str, object]]:
        path = self._cache_dir / f"{bucket_id}.pt"
        if not path.exists():
            return {}
        try:
            return torch.load(path, weights_only=False)
        except Exception:
            return {}

    def _save_to_disk(self, bucket_id: str, cached: Dict[str, Dict[str, object]]) -> None:
        path = self._cache_dir / f"{bucket_id}.pt"
        torch.save(cached, path)
