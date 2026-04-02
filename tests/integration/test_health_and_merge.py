"""Integration tests for HealthMonitor (6-E) and MergingService (6-F).

Uses real registry/store objects with mocked embedder and strategy.
Proves that the size trigger, coherence trigger, and merge trigger all fire.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from libucks.health_monitor import HealthMonitor, _COHERENCE_THRESHOLD
from libucks.merging_service import MergingService, MERGE_SIMILARITY, MERGE_TOKEN_LIMIT
from libucks.mitosis import MitosisService
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry(tmp_path: Path) -> BucketRegistry:
    reg = BucketRegistry(tmp_path / "registry.json")
    return reg


@pytest.fixture()
def store(tmp_path: Path) -> BucketStore:
    return BucketStore(tmp_path / "buckets")


def _mock_embedder(dim: int = 8) -> MagicMock:
    """Return a fake EmbeddingService whose embed/embed_batch return deterministic unit vectors."""
    embedder = MagicMock()
    embedder.embed.side_effect = lambda text: np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    embedder.embed_batch.side_effect = lambda texts: np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)
    return embedder


def _mock_strategy() -> MagicMock:
    strategy = MagicMock()
    strategy.reason = AsyncMock(return_value="mock prose")
    return strategy


def _mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.unregister_librarian = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _seed(registry: BucketRegistry, bucket_id: str, tokens: int, centroid: np.ndarray) -> None:
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    await registry.register(bucket_id, centroid.astype(np.float32), tokens)


# ---------------------------------------------------------------------------
# 6-E: HealthMonitor size trigger
# ---------------------------------------------------------------------------

class TestHealthMonitorSizeTrigger:
    @pytest.mark.asyncio
    async def test_size_trigger_calls_mitosis_split(self, registry: BucketRegistry, store: BucketStore) -> None:
        """A bucket over the token threshold must trigger MitosisService.split()."""
        BIG_TOKEN_COUNT = 25_000  # above default 20k threshold
        bucket_id = "aabbccdd"
        centroid = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        await _seed(registry, bucket_id, BIG_TOKEN_COUNT, centroid)

        mitosis_svc = MagicMock(spec=MitosisService)
        mitosis_svc.split = AsyncMock()
        merging_svc = MagicMock(spec=MergingService)
        merging_svc.run_merge_pass = AsyncMock()

        monitor = HealthMonitor(
            registry=registry,
            store=store,
            mitosis_service=mitosis_svc,
            merging_service=merging_svc,
            embedder=_mock_embedder(),
            mitosis_threshold=20_000,
        )

        await monitor._check()

        mitosis_svc.split.assert_called_once_with(bucket_id)
        merging_svc.run_merge_pass.assert_called_once()


# ---------------------------------------------------------------------------
# 6-E: HealthMonitor coherence trigger
# ---------------------------------------------------------------------------

class TestHealthMonitorCoherenceTrigger:
    @pytest.mark.asyncio
    async def test_low_coherence_calls_mitosis_split(
        self, registry: BucketRegistry, store: BucketStore
    ) -> None:
        """A bucket whose chunks are incoherent must trigger MitosisService.split()."""
        bucket_id = "coh00001"
        centroid = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        await _seed(registry, bucket_id, 5_000, centroid)

        mitosis_svc = MagicMock(spec=MitosisService)
        mitosis_svc.split = AsyncMock()
        merging_svc = MagicMock(spec=MergingService)
        merging_svc.run_merge_pass = AsyncMock()

        # 4 orthogonal unit vectors → mean coherence = 1/√4 = 0.5 < threshold 0.55
        embedder = MagicMock()
        dim = 8
        e1 = np.zeros(dim, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(dim, dtype=np.float32); e2[1] = 1.0
        e3 = np.zeros(dim, dtype=np.float32); e3[2] = 1.0
        e4 = np.zeros(dim, dtype=np.float32); e4[3] = 1.0
        embedder.embed_batch.side_effect = lambda texts: np.stack([e1, e2, e3, e4])

        monitor = HealthMonitor(
            registry=registry,
            store=store,
            mitosis_service=mitosis_svc,
            merging_service=merging_svc,
            embedder=embedder,
            mitosis_threshold=20_000,
        )

        # Patch store.read to return a fake bucket with 2 chunks
        fake_chunk = MagicMock()
        fake_chunk.source_file = "/nonexistent/file.py"
        fake_chunk.start_line = 1
        fake_chunk.end_line = 10
        fake_fm = MagicMock()
        fake_fm.chunks = [fake_chunk, fake_chunk, fake_chunk, fake_chunk]

        with patch.object(store, "read", return_value=(fake_fm, "prose")):
            await monitor._check()

        mitosis_svc.split.assert_called_once_with(bucket_id)


# ---------------------------------------------------------------------------
# 6-F: MergingService merge trigger
# ---------------------------------------------------------------------------

class TestMergingService:
    @pytest.mark.asyncio
    async def test_merge_fires_when_similar_and_small(
        self, registry: BucketRegistry, store: BucketStore, tmp_path: Path
    ) -> None:
        """Two buckets with high centroid similarity and low combined tokens must be merged."""
        dim = 8
        # Both centroids point in nearly the same direction → similarity > MERGE_SIMILARITY
        c1 = np.array([1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c2 = np.array([1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)

        await _seed(registry, "bucket_a", 3_000, c1)
        await _seed(registry, "bucket_b", 4_000, c2)

        agent = _mock_agent()
        svc = MergingService(
            registry=registry,
            store=store,
            agent=agent,
            embedder=_mock_embedder(dim),
            strategy=_mock_strategy(),
        )

        # Verify _should_merge returns True for this pair
        centroids = registry.get_all_centroids()
        recent: set = set()
        assert svc._should_merge("bucket_a", "bucket_b", centroids, recent), (
            "Expected _should_merge to return True for similar, small buckets"
        )

    @pytest.mark.asyncio
    async def test_merge_blocked_by_token_limit(self, registry: BucketRegistry, store: BucketStore) -> None:
        """Two large buckets must not be merged even if similar."""
        dim = 8
        c = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        await _seed(registry, "big_a", 10_000, c.copy())
        await _seed(registry, "big_b", 10_000, c.copy())

        svc = MergingService(
            registry=registry,
            store=store,
            agent=_mock_agent(),
            embedder=_mock_embedder(dim),
            strategy=_mock_strategy(),
        )

        centroids = registry.get_all_centroids()
        assert not svc._should_merge("big_a", "big_b", centroids, set()), (
            "Merge should be blocked: combined tokens >= MERGE_TOKEN_LIMIT"
        )

    @pytest.mark.asyncio
    async def test_merge_blocked_by_anti_cycle(self, registry: BucketRegistry, store: BucketStore) -> None:
        """A bucket in recent merge_history must not be merged again within 1 hour."""
        from datetime import datetime, timezone

        dim = 8
        c = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        await _seed(registry, "cycle_a", 1_000, c.copy())
        await _seed(registry, "cycle_b", 1_000, c.copy())

        # Record a recent merge involving cycle_a
        registry._meta["merge_history"] = [
            {
                "merged_bucket_ids": ["cycle_a", "cycle_x"],
                "result_bucket_id": "cycle_a",
                "merged_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

        svc = MergingService(
            registry=registry,
            store=store,
            agent=_mock_agent(),
            embedder=_mock_embedder(dim),
            strategy=_mock_strategy(),
        )

        centroids = registry.get_all_centroids()
        recent = svc._recent_merged_ids()
        assert not svc._should_merge("cycle_a", "cycle_b", centroids, recent), (
            "Merge should be blocked by anti-cycle guard"
        )

    @pytest.mark.asyncio
    async def test_no_merge_when_dissimilar(self, registry: BucketRegistry, store: BucketStore) -> None:
        """Orthogonal centroids must not be merged."""
        dim = 8
        c1 = np.zeros(dim, dtype=np.float32); c1[0] = 1.0
        c2 = np.zeros(dim, dtype=np.float32); c2[1] = 1.0

        await _seed(registry, "orth_a", 1_000, c1)
        await _seed(registry, "orth_b", 1_000, c2)

        svc = MergingService(
            registry=registry,
            store=store,
            agent=_mock_agent(),
            embedder=_mock_embedder(dim),
            strategy=_mock_strategy(),
        )

        centroids = registry.get_all_centroids()
        assert not svc._should_merge("orth_a", "orth_b", centroids, set()), (
            "Orthogonal centroids should not be merged"
        )
