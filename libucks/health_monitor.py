"""HealthMonitor — periodic quality guardian for the bucket index.

Runs every *interval* seconds (default 300 = 5 min). Each pass:

  1. Size check: token_count >= mitosis_threshold → MitosisService.split()
  2. Coherence check: mean chunk-to-centroid similarity < 0.55 → MitosisService.split()
  3. Merge pass: delegate to MergingService to find and collapse redundant pairs.

Coherence is the mean cosine similarity of each chunk embedding to the bucket
centroid — cheap to compute (one embed_batch + dot product), no pairwise loop.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

from libucks.mitosis import _read_chunk_content

if TYPE_CHECKING:
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.merging_service import MergingService
    from libucks.mitosis import MitosisService
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore

log = structlog.get_logger(__name__)

_COHERENCE_THRESHOLD = 0.55


class HealthMonitor:
    def __init__(
        self,
        registry: "BucketRegistry",
        store: "BucketStore",
        mitosis_service: "MitosisService",
        merging_service: "MergingService",
        embedder: "EmbeddingService",
        mitosis_threshold: int = 20_000,
        interval: int = 300,
    ) -> None:
        self._registry = registry
        self._store = store
        self._mitosis = mitosis_service
        self._merging = merging_service
        self._embedder = embedder
        self._mitosis_threshold = mitosis_threshold
        self._interval = interval

    async def run(self) -> None:
        """Loop forever, running a health check every *interval* seconds."""
        log.info("health_monitor.started", interval=self._interval)
        while True:
            await asyncio.sleep(self._interval)
            await self._check()

    async def _check(self) -> None:
        """Run one health pass over all registered buckets."""
        bucket_ids = list(self._registry.get_all_centroids().keys())
        log.info("health_monitor.check", bucket_count=len(bucket_ids))

        for bucket_id in bucket_ids:
            try:
                if self._registry.is_splitting(bucket_id):
                    continue
            except KeyError:
                continue  # bucket deregistered mid-pass

            # ---- Size check -------------------------------------------------
            try:
                tokens = self._registry.get_token_count(bucket_id)
            except KeyError:
                continue
            if tokens >= self._mitosis_threshold:
                log.warning(
                    "health_monitor.size_trigger",
                    bucket_id=bucket_id,
                    tokens=tokens,
                    threshold=self._mitosis_threshold,
                )
                await self._mitosis.split(bucket_id)
                continue  # bucket is gone; skip coherence

            # ---- Coherence check --------------------------------------------
            score = self._compute_coherence(bucket_id)
            if score is not None:
                # Store for status reporting
                entry = self._registry._buckets.get(bucket_id)
                if entry is not None:
                    entry.coherence_score = score
                if score < _COHERENCE_THRESHOLD:
                    log.warning(
                        "health_monitor.coherence_trigger",
                        bucket_id=bucket_id,
                        coherence=round(score, 3),
                    )
                    await self._mitosis.split(bucket_id)

        # ---- Merge pass -----------------------------------------------------
        await self._merging.run_merge_pass()

    def _compute_coherence(self, bucket_id: str) -> Optional[float]:
        """Return mean cosine similarity of chunk embeddings to the bucket centroid.

        Returns None if the bucket cannot be read or has fewer than 2 chunks.
        """
        try:
            front_matter, _ = self._store.read(bucket_id)
        except FileNotFoundError:
            return None

        chunks = front_matter.chunks
        if len(chunks) < 2:
            return 1.0  # a single-chunk bucket is trivially coherent

        contents = [_read_chunk_content(c) for c in chunks]
        try:
            embeddings = self._embedder.embed_batch(contents)  # (N, D)
        except Exception as exc:
            log.warning("health_monitor.embed_failed", bucket_id=bucket_id, error=str(exc))
            return None

        # L2-normalise rows
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        centroid = embeddings.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid /= c_norm

        return float(np.mean(embeddings @ centroid))
