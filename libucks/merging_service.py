"""MergingService — merge two semantically similar, under-filled buckets into one.

Merge conditions (all three must hold):
  1. cosine_similarity(A.centroid, B.centroid) > MERGE_SIMILARITY  (0.82)
  2. A.token_count + B.token_count < MERGE_TOKEN_LIMIT             (15 000)
  3. Neither bucket appears in _meta.merge_history within the last hour.

Merge protocol:
  - Absorbing bucket = higher token count (keeps its ID).
  - Dissolved bucket = smaller (deregistered and deleted).
  - New centroid = normalize(mean(embed_batch(all chunks from both))).
  - New prose = ThinkingStrategy.reason() over combined chunk content.
  - Anti-cycle: append to _meta.merge_history; prune entries > 24 h old on save.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional, Set

import numpy as np
import structlog

from libucks.mitosis import _read_chunk_content

if TYPE_CHECKING:
    from libucks.central_agent import CentralAgent
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.thinking.base import ThinkingStrategy
    from libucks.translator import Translator

log = structlog.get_logger(__name__)

MERGE_SIMILARITY: float = 0.82
MERGE_TOKEN_LIMIT: int = 15_000


def _encode_centroid(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


class MergingService:
    def __init__(
        self,
        registry: "BucketRegistry",
        store: "BucketStore",
        agent: "CentralAgent",
        embedder: "EmbeddingService",
        strategy: "ThinkingStrategy",
        translator: "Translator | None" = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._agent = agent
        self._embedder = embedder
        self._strategy = strategy
        self._translator = translator

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run_merge_pass(self) -> None:
        """Scan all bucket pairs and merge the first eligible pair found.

        One merge per pass avoids cascading consistency issues.
        """
        centroids = self._registry.get_all_centroids()
        bucket_ids = list(centroids.keys())
        recent = self._recent_merged_ids()

        for i, a in enumerate(bucket_ids):
            for b in bucket_ids[i + 1 :]:
                if self._should_merge(a, b, centroids, recent):
                    await self._merge(a, b)
                    return

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_merge(
        self,
        a: str,
        b: str,
        centroids: dict,
        recent: Set[str],
    ) -> bool:
        if a in recent or b in recent:
            return False
        sim = float(centroids[a] @ centroids[b])  # L2-normalised → cosine sim
        if sim <= MERGE_SIMILARITY:
            return False
        try:
            combined = self._registry.get_token_count(a) + self._registry.get_token_count(b)
        except KeyError:
            return False
        return combined < MERGE_TOKEN_LIMIT

    def _recent_merged_ids(self) -> Set[str]:
        """Return bucket IDs involved in any merge within the last hour."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        result: Set[str] = set()
        for entry in self._registry._meta.get("merge_history", []):
            try:
                merged_at_str: str = entry["merged_at"]
                merged_at = datetime.fromisoformat(merged_at_str)
                # Make offset-naive comparisons safe
                if merged_at.tzinfo is None:
                    merged_at = merged_at.replace(tzinfo=timezone.utc)
                if merged_at > cutoff:
                    for bid in entry.get("merged_bucket_ids", []):
                        result.add(bid)
            except Exception:
                pass
        return result

    async def _merge(self, a: str, b: str) -> None:
        try:
            tokens_a = self._registry.get_token_count(a)
            tokens_b = self._registry.get_token_count(b)
        except KeyError:
            return

        absorbing, dissolved = (a, b) if tokens_a >= tokens_b else (b, a)
        log.info("merging.start", absorbing=absorbing, dissolved=dissolved)

        try:
            fm_absorb, _ = self._store.read(absorbing)
            fm_dissolve, _ = self._store.read(dissolved)
        except FileNotFoundError as exc:
            log.warning("merging.read_failed", error=str(exc))
            return

        all_chunks = fm_absorb.chunks + fm_dissolve.chunks
        contents = [_read_chunk_content(c) for c in all_chunks]

        try:
            embeddings = self._embedder.embed_batch(contents)
        except Exception as exc:
            log.warning("merging.embed_failed", error=str(exc))
            return

        centroid = np.mean(embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm

        domain = f"{fm_absorb.domain_label} + {fm_dissolve.domain_label}"
        combined_content = "\n".join(contents)
        try:
            result = await self._strategy.reason(
                f"Summarize these merged code chunks: {domain}",
                combined_content[:2000],
            )
            if self._translator is not None:
                prose = await self._translator.synthesize("", [result])
            else:
                prose = f"# {domain}\n\n"
        except Exception as exc:
            log.warning("merging.reason_failed", error=str(exc))
            prose = f"# {domain}\n\n(merged bucket)"

        self._store.create(
            bucket_id=absorbing,
            domain_label=domain,
            centroid=_encode_centroid(centroid),
            chunks=all_chunks,
            prose=prose,
        )
        total_tokens = sum(c.token_count for c in all_chunks)
        await self._registry.register(absorbing, centroid, total_tokens)

        # Dissolve the smaller bucket
        self._agent.unregister_librarian(dissolved)
        try:
            await self._registry.deregister(dissolved)
        except KeyError:
            pass
        try:
            self._store.delete(dissolved)
        except FileNotFoundError:
            pass

        # Anti-cycle bookkeeping
        history: list = self._registry._meta.setdefault("merge_history", [])  # type: ignore[assignment]
        history.append(
            {
                "merged_bucket_ids": [absorbing, dissolved],
                "result_bucket_id": absorbing,
                "merged_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._prune_merge_history()
        self._registry.save()

        log.info(
            "merging.complete",
            absorbing=absorbing,
            dissolved=dissolved,
            total_tokens=total_tokens,
        )

    def _prune_merge_history(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        history: List[dict] = self._registry._meta.get("merge_history", [])  # type: ignore[assignment]
        kept = []
        for entry in history:
            try:
                merged_at_str: str = entry["merged_at"]
                merged_at = datetime.fromisoformat(merged_at_str)
                if merged_at.tzinfo is None:
                    merged_at = merged_at.replace(tzinfo=timezone.utc)
                if merged_at > cutoff:
                    kept.append(entry)
            except Exception:
                pass
        self._registry._meta["merge_history"] = kept
