"""Tests for MultiPerspectiveDataGenerator.generate_curriculum_batch().

A curriculum batch item contains:
  - mixed_input: Tensor of shape (K, D) — CurriculumMixer.mix() output
  - target_ids:  LongTensor of token IDs — the target text tokenized (integer IDs)
  - r:           float in [0, 1] — the mixing rate used for this item

The batch is used by LoRAReceiverTrainer to fine-tune the Base model with
cross-entropy loss (teacher forcing) + L_sep.
"""
import math
import torch
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


K = 8
D = 64
VOCAB = 256


@pytest.fixture
def mock_latent_strategy():
    strategy = MagicMock()
    # encode() returns a (5, D) tensor
    strategy.encode = AsyncMock(return_value=torch.randn(5, D))
    # reason() is also awaited in generate_curriculum_batch()
    strategy.reason = AsyncMock(return_value=torch.randn(5, D))
    return strategy


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    # forward() returns (K, D)
    adapter.return_value = torch.randn(K, D)
    return adapter


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    # tokenize returns some fake ids
    tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    return tok


@pytest.fixture
def mock_embedding():
    """Fake embedding layer — maps token ID to D-dim vector."""
    emb = MagicMock()
    emb.return_value = torch.randn(K, D)   # returns (K, D) for any input
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    emb.parameters.side_effect = lambda: iter([mock_param])
    return emb


@pytest.fixture
def mock_text_strategy():
    ts = MagicMock()
    ts.reason = AsyncMock(return_value="This module handles routing logic.")
    return ts


@pytest.fixture
def generator(mock_latent_strategy, mock_adapter, mock_tokenizer, mock_embedding, mock_text_strategy):
    from libucks.thinking.training.data_generator import MultiPerspectiveDataGenerator
    with patch("libucks.thinking.training.data_generator.MultiPerspectiveDataGenerator.__init__",
               lambda self, *a, **kw: None):
        gen = MultiPerspectiveDataGenerator.__new__(MultiPerspectiveDataGenerator)
    gen._latent_strategy = mock_latent_strategy
    gen._registry = MagicMock()
    gen._store = MagicMock()
    gen._teacher_reason = AsyncMock(return_value="This module handles routing logic.")
    gen._store.read.return_value = ("id", "some prose about code")
    gen._registry.get_all_centroids.return_value = {}
    return gen


# ── generate_curriculum_batch() contract ────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_returns_mixed_input_and_target_ids(
    generator, mock_adapter, mock_tokenizer, mock_embedding
):
    """generate_curriculum_batch() returns a dict with 'mixed_input' and 'target_ids'."""
    result = await generator.generate_curriculum_batch(
        bucket_id="test_bucket",
        adapter=mock_adapter,
        tokenizer=mock_tokenizer,
        embedding=mock_embedding,
        output_len=K,
        hidden_dim=D,
    )
    assert "mixed_input" in result
    assert "target_ids" in result


@pytest.mark.asyncio
async def test_mixed_input_shape(
    generator, mock_adapter, mock_tokenizer, mock_embedding
):
    """mixed_input has shape (K, D)."""
    result = await generator.generate_curriculum_batch(
        bucket_id="test_bucket",
        adapter=mock_adapter,
        tokenizer=mock_tokenizer,
        embedding=mock_embedding,
        output_len=K,
        hidden_dim=D,
    )
    assert result["mixed_input"].shape == (K, D)


@pytest.mark.asyncio
async def test_target_ids_are_integer_tensor(
    generator, mock_adapter, mock_tokenizer, mock_embedding
):
    """target_ids must be a LongTensor (integer IDs, not floats)."""
    result = await generator.generate_curriculum_batch(
        bucket_id="test_bucket",
        adapter=mock_adapter,
        tokenizer=mock_tokenizer,
        embedding=mock_embedding,
        output_len=K,
        hidden_dim=D,
    )
    ids = result["target_ids"]
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype in (torch.long, torch.int32, torch.int64)


@pytest.mark.asyncio
async def test_r_in_result_is_float_in_range(
    generator, mock_adapter, mock_tokenizer, mock_embedding
):
    """Result must include 'r' — a float in [0, 1]."""
    result = await generator.generate_curriculum_batch(
        bucket_id="test_bucket",
        adapter=mock_adapter,
        tokenizer=mock_tokenizer,
        embedding=mock_embedding,
        output_len=K,
        hidden_dim=D,
    )
    assert "r" in result
    r = result["r"]
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


@pytest.mark.asyncio
async def test_r_sampled_uniformly_across_calls(
    generator, mock_adapter, mock_tokenizer, mock_embedding
):
    """Across many calls, r values should span the range [0, 1] (not all same)."""
    rs = []
    for _ in range(50):
        result = await generator.generate_curriculum_batch(
            bucket_id="test_bucket",
            adapter=mock_adapter,
            tokenizer=mock_tokenizer,
            embedding=mock_embedding,
            output_len=K,
            hidden_dim=D,
        )
        rs.append(result["r"])

    # With 50 samples from U[0,1], max - min should be > 0.3 with overwhelming probability
    assert max(rs) - min(rs) > 0.3, "r values appear to not be sampled uniformly"
