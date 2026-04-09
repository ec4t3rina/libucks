"""Tests that InitOrchestrator uses init_bucket_size (not mitosis_threshold) for INIT clustering.

The core invariant: with init_bucket_size=2000 and mitosis_threshold=20000, a repo
with ~40000 raw tokens must produce ~20 clusters, NOT 2.  If the orchestrator were
accidentally using mitosis_threshold the cluster count would be 10x too low.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from libucks.thinking.base import ThinkingStrategy


def _make_strategy() -> MagicMock:
    strategy = MagicMock(spec=ThinkingStrategy)
    strategy.reason = AsyncMock(return_value="TITLE: Test module\n---\nSome prose.")
    return strategy


class TestInitBucketSize:
    async def test_n_clusters_uses_init_bucket_size_not_mitosis_threshold(self, tmp_path: Path):
        """With init_bucket_size=1000 and mitosis_threshold=20000, a small repo should
        produce significantly more clusters than if mitosis_threshold were used."""
        from libucks.init_orchestrator import InitOrchestrator, _n_clusters
        from libucks.config import RoutingConfig

        # _n_clusters(total_tokens, target_tokens, n_chunks)
        # With init_bucket_size=1000 and 10000 tokens → 10 clusters
        # With mitosis_threshold=20000 and 10000 tokens → 0, clamped to 1
        assert _n_clusters(10_000, 1_000, 100) == 10
        assert _n_clusters(10_000, 20_000, 100) == 1

    async def test_orchestrator_bucket_count_reflects_init_bucket_size(self, tmp_path: Path):
        """InitOrchestrator must create more buckets when init_bucket_size is small."""
        import shutil
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.config import Config, RoutingConfig
        from libucks.embeddings.embedding_service import EmbeddingService

        # Write enough Python files to produce significant raw token counts.
        for i in range(8):
            src = "\n\n".join([
                f"def func_{i}_{j}(x, y):\n    '''Function {j} in file {i}.'''\n    return x + y + {j}"
                for j in range(20)
            ])
            (tmp_path / f"module_{i}.py").write_text(src)

        def _make_embedder_mock():
            mock_emb = MagicMock()
            rng = np.random.default_rng(seed=42)

            def fake_embed_batch(texts):
                n = len(texts)
                embs = rng.standard_normal((n, 8)).astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                return embs / norms

            mock_emb.embed_batch.side_effect = fake_embed_batch
            mock_emb.embed.return_value = np.ones(8, dtype=np.float32) / np.sqrt(8)
            return mock_emb

        # Run with small init_bucket_size
        small_cfg = Config(routing=RoutingConfig(init_bucket_size=500, mitosis_threshold=20_000))
        with patch.object(Config, "load", return_value=small_cfg):
            with patch.object(EmbeddingService, "get_instance", return_value=_make_embedder_mock()):
                await InitOrchestrator(tmp_path, strategy=_make_strategy()).run()
        small_count = len(list((tmp_path / ".libucks" / "buckets").glob("*.md")))
        shutil.rmtree(tmp_path / ".libucks")

        # Run with large init_bucket_size (same as mitosis_threshold → few clusters)
        large_cfg = Config(routing=RoutingConfig(init_bucket_size=20_000, mitosis_threshold=20_000))
        with patch.object(Config, "load", return_value=large_cfg):
            with patch.object(EmbeddingService, "get_instance", return_value=_make_embedder_mock()):
                await InitOrchestrator(tmp_path, strategy=_make_strategy()).run()
        large_count = len(list((tmp_path / ".libucks" / "buckets").glob("*.md")))

        assert small_count > large_count, (
            f"Expected more buckets with init_bucket_size=500 ({small_count}) "
            f"than with init_bucket_size=20000 ({large_count})"
        )

    def test_default_init_bucket_size_gives_reasonable_click_scale_clusters(self):
        """Sanity check: with default init_bucket_size=2000 and click's ~95k tokens,
        we expect roughly 47 clusters (not 4 from the old mitosis_threshold formula)."""
        from libucks.init_orchestrator import _n_clusters

        click_tokens = 95_000
        n_chunks = 400  # click has ~400 functions/classes

        # Old formula (mitosis_threshold=20000): 95000 // 20000 = 4
        old_result = _n_clusters(click_tokens, 20_000, n_chunks)
        # New formula (init_bucket_size=2000): 95000 // 2000 = 47
        new_result = _n_clusters(click_tokens, 2_000, n_chunks)

        assert old_result == 4
        assert new_result == 47
