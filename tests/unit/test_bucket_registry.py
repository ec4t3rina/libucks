"""Phase 1 Testing Gate — test_bucket_registry.py

Tests in-memory state management, JSON persistence, and concurrent-write
safety for BucketRegistry.
"""

import asyncio
import base64
import json
from pathlib import Path

import numpy as np
import pytest

from libucks.storage.bucket_registry import BucketRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / ".libucks" / "registry.json"


@pytest.fixture
def registry(registry_path: Path) -> BucketRegistry:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    return BucketRegistry(registry_path)


def _centroid(seed: float = 0.1) -> np.ndarray:
    """Return a normalised float32 vector with a recognisable value."""
    arr = np.array([seed, seed * 2, seed * 3], dtype=np.float32)
    return arr / np.linalg.norm(arr)


# ---------------------------------------------------------------------------
# register() / get_all_centroids()
# ---------------------------------------------------------------------------

class TestRegister:
    async def test_registered_centroid_is_returned(self, registry: BucketRegistry):
        c = _centroid(0.1)
        await registry.register("bucket-a", c, token_count=100)
        centroids = registry.get_all_centroids()
        assert "bucket-a" in centroids
        np.testing.assert_array_almost_equal(centroids["bucket-a"], c)

    async def test_registered_token_count_is_stored(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=512)
        assert registry.get_token_count("bucket-a") == 512

    async def test_multiple_buckets_stored_independently(self, registry: BucketRegistry):
        c_a = _centroid(0.1)
        c_b = _centroid(0.9)
        await registry.register("bucket-a", c_a, token_count=100)
        await registry.register("bucket-b", c_b, token_count=200)
        centroids = registry.get_all_centroids()
        np.testing.assert_array_almost_equal(centroids["bucket-a"], c_a)
        np.testing.assert_array_almost_equal(centroids["bucket-b"], c_b)

    async def test_re_register_overwrites(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(0.1), token_count=100)
        new_c = _centroid(0.9)
        await registry.register("bucket-a", new_c, token_count=999)
        centroids = registry.get_all_centroids()
        np.testing.assert_array_almost_equal(centroids["bucket-a"], new_c)
        assert registry.get_token_count("bucket-a") == 999


# ---------------------------------------------------------------------------
# deregister()
# ---------------------------------------------------------------------------

class TestDeregister:
    async def test_deregistered_bucket_not_in_centroids(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        await registry.deregister("bucket-a")
        assert "bucket-a" not in registry.get_all_centroids()

    async def test_deregister_unknown_bucket_raises(self, registry: BucketRegistry):
        with pytest.raises(KeyError):
            await registry.deregister("ghost")

    async def test_other_buckets_unaffected_by_deregister(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(0.1), token_count=100)
        await registry.register("bucket-b", _centroid(0.9), token_count=200)
        await registry.deregister("bucket-a")
        assert "bucket-b" in registry.get_all_centroids()


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestPersistence:
    async def test_save_creates_registry_json(
        self, registry: BucketRegistry, registry_path: Path
    ):
        await registry.register("bucket-a", _centroid(), token_count=42)
        registry.save()
        assert registry_path.exists()

    async def test_load_restores_centroid_within_float32_tolerance(
        self, registry_path: Path
    ):
        # Populate and save
        r1 = BucketRegistry(registry_path)
        original = _centroid(0.3)
        await r1.register("bucket-a", original, token_count=300)
        r1.save()

        # Load into a fresh instance
        r2 = BucketRegistry(registry_path)
        r2.load()
        centroids = r2.get_all_centroids()
        assert "bucket-a" in centroids
        assert np.allclose(centroids["bucket-a"], original, atol=1e-6)

    async def test_load_restores_token_count(self, registry_path: Path):
        r1 = BucketRegistry(registry_path)
        await r1.register("bucket-a", _centroid(), token_count=777)
        r1.save()

        r2 = BucketRegistry(registry_path)
        r2.load()
        assert r2.get_token_count("bucket-a") == 777

    async def test_load_restores_multiple_buckets(self, registry_path: Path):
        r1 = BucketRegistry(registry_path)
        for i in range(5):
            await r1.register(f"bucket-{i}", _centroid(0.1 * (i + 1)), token_count=i * 10)
        r1.save()

        r2 = BucketRegistry(registry_path)
        r2.load()
        assert set(r2.get_all_centroids().keys()) == {f"bucket-{i}" for i in range(5)}

    async def test_save_persists_centroid_as_base64(
        self, registry: BucketRegistry, registry_path: Path
    ):
        await registry.register("bucket-a", _centroid(), token_count=1)
        registry.save()
        data = json.loads(registry_path.read_text())
        # The centroid must be stored as a base64 string, not a raw list of floats
        stored = data["bucket-a"]["centroid_embedding"]
        assert isinstance(stored, str)
        # Must be valid base64 that decodes to the right number of bytes
        decoded = base64.b64decode(stored)
        assert len(decoded) % 4 == 0  # multiple of 4 bytes (float32)

    async def test_load_on_empty_registry_is_noop(
        self, registry: BucketRegistry, registry_path: Path
    ):
        # No file exists yet — load() should be a no-op, not raise
        registry.load()
        assert registry.get_all_centroids() == {}

    async def test_save_then_register_then_reload(self, registry_path: Path):
        """Reload reflects only the saved state, not post-save registrations."""
        r1 = BucketRegistry(registry_path)
        await r1.register("bucket-a", _centroid(0.1), token_count=100)
        r1.save()
        await r1.register("bucket-b", _centroid(0.9), token_count=200)
        # bucket-b was NOT saved

        r2 = BucketRegistry(registry_path)
        r2.load()
        assert "bucket-a" in r2.get_all_centroids()
        assert "bucket-b" not in r2.get_all_centroids()


# ---------------------------------------------------------------------------
# set_splitting() / is_splitting()
# ---------------------------------------------------------------------------

class TestSplitting:
    async def test_default_splitting_is_false(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        assert registry.is_splitting("bucket-a") is False

    async def test_set_splitting_true_is_reflected(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        await registry.set_splitting("bucket-a", True)
        assert registry.is_splitting("bucket-a") is True

    async def test_set_splitting_false_clears_flag(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        await registry.set_splitting("bucket-a", True)
        await registry.set_splitting("bucket-a", False)
        assert registry.is_splitting("bucket-a") is False

    async def test_splitting_flag_independent_per_bucket(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(0.1), token_count=100)
        await registry.register("bucket-b", _centroid(0.9), token_count=200)
        await registry.set_splitting("bucket-a", True)
        assert registry.is_splitting("bucket-a") is True
        assert registry.is_splitting("bucket-b") is False

    async def test_set_splitting_unknown_bucket_raises(self, registry: BucketRegistry):
        with pytest.raises(KeyError):
            await registry.set_splitting("ghost", True)

    async def test_is_splitting_unknown_bucket_raises(self, registry: BucketRegistry):
        with pytest.raises(KeyError):
            registry.is_splitting("ghost")


# ---------------------------------------------------------------------------
# get_lock() — per-bucket asyncio.Lock
# ---------------------------------------------------------------------------

class TestLocks:
    async def test_get_lock_returns_asyncio_lock(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        lock = registry.get_lock("bucket-a")
        assert isinstance(lock, asyncio.Lock)

    async def test_same_bucket_returns_same_lock_instance(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        assert registry.get_lock("bucket-a") is registry.get_lock("bucket-a")

    async def test_different_buckets_have_different_locks(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(0.1), token_count=100)
        await registry.register("bucket-b", _centroid(0.9), token_count=200)
        assert registry.get_lock("bucket-a") is not registry.get_lock("bucket-b")

    async def test_lock_is_functional(self, registry: BucketRegistry):
        await registry.register("bucket-a", _centroid(), token_count=100)
        lock = registry.get_lock("bucket-a")
        async with lock:
            assert lock.locked()

    async def test_lock_survives_re_register(self, registry: BucketRegistry):
        """Re-registering a bucket must NOT replace its lock (ongoing holders would deadlock)."""
        await registry.register("bucket-a", _centroid(0.1), token_count=100)
        lock_before = registry.get_lock("bucket-a")
        await registry.register("bucket-a", _centroid(0.9), token_count=999)
        lock_after = registry.get_lock("bucket-a")
        assert lock_before is lock_after


# ---------------------------------------------------------------------------
# Concurrency — 100 concurrent register() calls
# ---------------------------------------------------------------------------

class TestConcurrency:
    async def test_100_concurrent_registers_no_corruption(self, registry: BucketRegistry):
        """asyncio.gather over 100 distinct bucket registrations must all land correctly."""
        n = 100
        centroids = [_centroid(0.01 * (i + 1)) for i in range(n)]

        await asyncio.gather(
            *[
                registry.register(f"bucket-{i}", centroids[i], token_count=i)
                for i in range(n)
            ]
        )

        all_centroids = registry.get_all_centroids()
        assert len(all_centroids) == n
        for i in range(n):
            assert f"bucket-{i}" in all_centroids
            np.testing.assert_array_almost_equal(all_centroids[f"bucket-{i}"], centroids[i])

    async def test_concurrent_set_splitting_no_corruption(self, registry: BucketRegistry):
        """Concurrent flag flips on different buckets must not corrupt each other."""
        n = 50
        for i in range(n):
            await registry.register(f"bucket-{i}", _centroid(0.01 * (i + 1)), token_count=i)

        # Flip all to True concurrently
        await asyncio.gather(
            *[registry.set_splitting(f"bucket-{i}", True) for i in range(n)]
        )
        for i in range(n):
            assert registry.is_splitting(f"bucket-{i}") is True

        # Flip all to False concurrently
        await asyncio.gather(
            *[registry.set_splitting(f"bucket-{i}", False) for i in range(n)]
        )
        for i in range(n):
            assert registry.is_splitting(f"bucket-{i}") is False
