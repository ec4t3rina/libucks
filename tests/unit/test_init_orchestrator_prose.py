"""Tests for INIT prose generation — bucket prose, title parsing, centroid blending."""
from __future__ import annotations

import base64
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml

from libucks.thinking.base import ThinkingStrategy


def _make_mock_strategy(
    response: str = "TITLE: CLI decorators\n---\nThis module handles decorators.",
) -> MagicMock:
    strategy = MagicMock(spec=ThinkingStrategy)
    strategy.reason = AsyncMock(return_value=response)
    strategy.encode = AsyncMock(side_effect=lambda text: text)
    strategy.decode = AsyncMock(side_effect=lambda r: r)
    return strategy


def _decode_centroid(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    n = len(raw) // 4
    return np.array(struct.unpack(f"{n}f", raw), dtype=np.float32)


def _read_first_bucket(bucket_dir: Path) -> tuple[dict, str]:
    """Read the first .md file in bucket_dir and return (front_matter_dict, prose)."""
    md_files = list(bucket_dir.glob("*.md"))
    assert len(md_files) >= 1, "Expected at least one bucket .md file"
    raw = md_files[0].read_text()
    # Format: ---\n{yaml}---\n{prose}
    _, yaml_block, prose = raw.split("---\n", 2)
    return yaml.safe_load(yaml_block), prose


class TestInitOrchestratorProse:
    async def test_prose_is_not_placeholder(self, tmp_path: Path):
        """BucketStore.create() must receive non-placeholder prose."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy("TITLE: Auth middleware\n---\nHandles JWT validation.")

        (tmp_path / "auth.py").write_text(
            "def authenticate(token):\n    '''Validate JWT.'''\n    return True\n"
        )

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy).run()

        fm, prose = _read_first_bucket(tmp_path / ".libucks" / "buckets")
        assert "Initial seed" not in prose
        assert "Handles JWT validation" in prose

    async def test_domain_label_is_parsed_title(self, tmp_path: Path):
        """domain_label must be the TITLE line, not a file-stem string."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy("TITLE: JWT authentication handler\n---\nDetails here.")

        (tmp_path / "auth.py").write_text("def check(token):\n    return True\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy).run()

        fm, _ = _read_first_bucket(tmp_path / ".libucks" / "buckets")
        assert fm["domain_label"] == "JWT authentication handler"

    async def test_strategy_reason_called_once_per_cluster(self, tmp_path: Path):
        """strategy.reason() must be called exactly N times for N clusters."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy()

        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(2, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy).run()

        n_buckets = len(list((tmp_path / ".libucks" / "buckets").glob("*.md")))
        assert strategy.reason.call_count == n_buckets

    async def test_centroid_is_blended_not_pure_chunk_mean(self, tmp_path: Path):
        """Centroid must incorporate title embedding (0.2 weight), not be pure chunk mean."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy("TITLE: CLI handler\n---\nCore CLI logic.")

        (tmp_path / "cli.py").write_text("def run(): pass\n")

        chunk_embed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        title_embed = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = chunk_embed.reshape(1, -1)
            mock_emb.embed.return_value = title_embed
            mock_cls.return_value = mock_emb

            await InitOrchestrator(tmp_path, strategy=strategy).run()

        fm, _ = _read_first_bucket(tmp_path / ".libucks" / "buckets")
        centroid = _decode_centroid(fm["centroid_embedding"])

        # Pure chunk mean → [1, 0, 0, 0] after norm. Blended → dim[1] > 0.
        assert centroid[1] > 0.01, (
            f"Centroid dim[1] should reflect title embedding contribution, got {centroid}"
        )

    async def test_fallback_to_file_stem_label_on_missing_title(self, tmp_path: Path):
        """If LLM response lacks TITLE: line, fall back gracefully — no crash, bucket created."""
        from libucks.init_orchestrator import InitOrchestrator
        from libucks.embeddings.embedding_service import EmbeddingService

        strategy = _make_mock_strategy("Just some prose without a title line.")

        (tmp_path / "utils.py").write_text("def helper(): pass\n")

        with patch.object(EmbeddingService, "get_instance") as mock_cls:
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_emb.embed.return_value = np.random.rand(8).astype(np.float32)
            mock_cls.return_value = mock_emb

            # Must not raise
            await InitOrchestrator(tmp_path, strategy=strategy).run()

        bucket_files = list((tmp_path / ".libucks" / "buckets").glob("*.md"))
        assert len(bucket_files) == 1
