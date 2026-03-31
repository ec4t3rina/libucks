"""Phase 2 Testing Gate — test_central_agent_routing.py

Tests routing math (route / is_novel) for CentralAgent using deterministic
unit-vector centroids. No embedding model is loaded at any point.

Cosine similarity = dot(q, c) for L2-normalised vectors.
Orthogonal unit vectors give similarity 0.0; identical unit vectors give 1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from libucks.central_agent import CentralAgent
from libucks.config import Config, RoutingConfig
from libucks.storage.bucket_registry import BucketRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(n: int, dim: int = 384) -> np.ndarray:
    """Return a float32 unit vector with 1.0 at index n, 0 elsewhere."""
    v = np.zeros(dim, dtype=np.float32)
    v[n] = 1.0
    return v


def _vec_with_similarity(s: float, dim: int = 384) -> np.ndarray:
    """Return a float32 unit vector whose dot product with _unit_vec(0) equals s.

    Constructed as s*e0 + sqrt(1-s²)*e1 — guaranteed L2-normalised.
    """
    v = np.zeros(dim, dtype=np.float32)
    v[0] = s
    v[1] = float(np.sqrt(max(0.0, 1.0 - s ** 2)))
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    return BucketRegistry(tmp_path / "registry.json")


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def agent(registry, config):
    return CentralAgent(registry=registry, config=config)


# ---------------------------------------------------------------------------
# route()
# ---------------------------------------------------------------------------

class TestRoute:
    async def test_closest_centroid_is_rank_1(self, agent, registry):
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.register("bucket_b", _unit_vec(1), 100)

        result = agent.route(_unit_vec(0), top_k=2)

        assert result[0] == "bucket_a"

    async def test_second_closest_is_rank_2(self, agent, registry):
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.register("bucket_b", _unit_vec(1), 100)
        await registry.register("bucket_c", _unit_vec(2), 100)

        # Query close to bucket_b
        result = agent.route(_unit_vec(1), top_k=3)

        assert result[0] == "bucket_b"
        assert set(result) == {"bucket_a", "bucket_b", "bucket_c"}

    async def test_route_returns_exactly_top_k(self, agent, registry):
        for i in range(5):
            await registry.register(f"bucket_{i}", _unit_vec(i), 100)

        result = agent.route(_unit_vec(0), top_k=3)

        assert len(result) == 3

    async def test_route_fewer_buckets_than_top_k(self, agent, registry):
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.register("bucket_b", _unit_vec(1), 100)

        result = agent.route(_unit_vec(0), top_k=5)

        assert len(result) == 2

    async def test_route_empty_registry_returns_empty_list(self, agent):
        result = agent.route(_unit_vec(0), top_k=3)

        assert result == []

    async def test_equidistant_centroids_no_error(self, agent, registry):
        centroid = np.ones(384, dtype=np.float32) / np.sqrt(384)
        for i in range(3):
            await registry.register(f"bucket_{i}", centroid.copy(), 100)

        result = agent.route(centroid.copy(), top_k=3)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# is_novel()
# ---------------------------------------------------------------------------

class TestIsNovel:
    async def test_novel_orthogonal_embedding(self, agent, registry):
        """Similarity = 0.0, threshold = 0.35 → 0.0 < 0.65 → novel."""
        await registry.register("bucket_a", _unit_vec(0), 100)

        assert agent.is_novel(_unit_vec(1)) is True

    async def test_not_novel_identical_embedding(self, agent, registry):
        """Similarity = 1.0, threshold = 0.35 → 1.0 >= 0.65 → not novel."""
        await registry.register("bucket_a", _unit_vec(0), 100)

        assert agent.is_novel(_unit_vec(0)) is False

    async def test_novel_empty_registry(self, agent):
        """No centroids → everything is novel."""
        assert agent.is_novel(_unit_vec(0)) is True

    async def test_novelty_boundary_just_below_threshold(self, registry, tmp_path):
        """Similarity 0.6 < (1 - 0.35 = 0.65) → novel."""
        cfg = Config()
        agent = CentralAgent(registry=registry, config=cfg)
        await registry.register("bucket_a", _unit_vec(0), 100)

        query = _vec_with_similarity(0.6)  # dot with e0 == 0.6
        assert agent.is_novel(query) is True

    async def test_novelty_boundary_just_above_threshold(self, registry, tmp_path):
        """Similarity 0.7 >= (1 - 0.35 = 0.65) → not novel."""
        cfg = Config()
        agent = CentralAgent(registry=registry, config=cfg)
        await registry.register("bucket_a", _unit_vec(0), 100)

        query = _vec_with_similarity(0.7)  # dot with e0 == 0.7
        assert agent.is_novel(query) is False
