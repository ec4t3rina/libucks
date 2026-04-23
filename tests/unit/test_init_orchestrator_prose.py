"""Tests for INIT prose generation — bucket prose via Translator, centroid blending."""
from __future__ import annotations

import base64
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from libucks.thinking.base import ThinkingStrategy


def _make_mock_strategy() -> MagicMock:
    """strategy.reason() returns a dummy Tensor (as LatentStrategy does in V2.1)."""
    strategy = MagicMock(spec=ThinkingStrategy)
    strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))
    return strategy


def _make_mock_translator(prose: str = "Handles JWT validation.") -> MagicMock:
    """translator.synthesize() returns decoded prose string."""
    translator = MagicMock()
    translator.synthesize = AsyncMock(return_value=prose)
    return translator


def _decode_centroid(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    n = len(raw) // 4
    return np.array(struct.unpack(f"{n}f", raw), dtype=np.float32)


def _read_first_bucket(bucket_dir: Path) -> tuple[dict, str]:
    md_files = list(bucket_dir.glob("*.md"))
    assert len(md_files) >= 1, "Expected at least one bucket .md file"
    raw = md_files[0].read_text()
    _, yaml_block, prose = raw.split("---\n", 2)
    return yaml.safe_load(yaml_block), prose


class TestInitOrchestratorProse:
    async def test_prose_is_not_placeholder(self, tmp_path: Path):
        """BucketStore.create() must receive translator-decoded prose, not placeholder."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy()
        translator = _make_mock_translator("Handles JWT validation.")

        (tmp_path / "auth.py").write_text(
            "def authenticate(token):\n    '''Validate JWT.'''\n    return True\n"
        )

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy, translator=translator).run()

        fm, prose = _read_first_bucket(tmp_path / ".libucks" / "buckets")
        assert "Initial seed" not in prose
        assert "Handles JWT validation" in prose

    async def test_translator_synthesize_called_once_per_cluster(self, tmp_path: Path):
        """translator.synthesize() must be called exactly N times for N clusters."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy()
        translator = _make_mock_translator("Some prose.")

        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(2, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy, translator=translator).run()

        n_buckets = len(list((tmp_path / ".libucks" / "buckets").glob("*.md")))
        assert translator.synthesize.call_count == n_buckets

    async def test_translator_receives_latent_from_strategy(self, tmp_path: Path):
        """translator.synthesize() must be called with the tensor from strategy.reason()."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        sentinel = torch.ones(4, 8)
        strategy = _make_mock_strategy()
        strategy.reason = AsyncMock(return_value=sentinel)
        translator = _make_mock_translator("Decoded prose.")

        (tmp_path / "cli.py").write_text("def run(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy, translator=translator).run()

        call_args = translator.synthesize.call_args
        # synthesize("", [latent]) — second positional arg is the representations list
        representations = call_args[0][1]
        assert len(representations) == 1
        assert torch.equal(representations[0], sentinel)

    async def test_strategy_reason_called_once_per_cluster(self, tmp_path: Path):
        """strategy.reason() must be called exactly N times for N clusters."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy()
        translator = _make_mock_translator()

        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(2, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy, translator=translator).run()

        n_buckets = len(list((tmp_path / ".libucks" / "buckets").glob("*.md")))
        assert strategy.reason.call_count == n_buckets

    async def test_centroid_is_blended_with_fallback_label(self, tmp_path: Path):
        """Centroid must incorporate the fallback label's title embedding (0.2 weight)."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy()
        translator = _make_mock_translator("Core CLI logic.")

        (tmp_path / "cli.py").write_text("def run(): pass\n")

        chunk_embed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        title_embed = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = chunk_embed.reshape(1, -1)
            # embed() is called with the fallback_label (domain label) as the title
            mock_emb.embed.return_value = title_embed
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy, translator=translator).run()

        fm, _ = _read_first_bucket(tmp_path / ".libucks" / "buckets")
        centroid = _decode_centroid(fm["centroid_embedding"])

        # Pure chunk mean → [1, 0, 0, 0] after norm. Blended → dim[1] > 0.
        assert centroid[1] > 0.01, (
            f"Centroid dim[1] should reflect title embedding contribution, got {centroid}"
        )

    async def test_placeholder_prose_used_when_no_strategy(self, tmp_path: Path):
        """No strategy → placeholder prose, no translator calls."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        translator = _make_mock_translator()

        (tmp_path / "utils.py").write_text("def helper(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=None, translator=translator).run()

        translator.synthesize.assert_not_called()
        bucket_files = list((tmp_path / ".libucks" / "buckets").glob("*.md"))
        assert len(bucket_files) == 1

    async def test_prose_is_empty_string_when_no_translator(self, tmp_path: Path):
        """Strategy present but no translator → reason() is called but prose stays empty."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy()

        (tmp_path / "utils.py").write_text("def helper(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            # No translator injected
            await InitOrchestrator(tmp_path, strategy=strategy, translator=None).run()

        fm, prose = _read_first_bucket(tmp_path / ".libucks" / "buckets")
        assert prose == ""
