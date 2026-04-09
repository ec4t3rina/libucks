"""Phase 11: LatentCompressor tests.

All tests run on CPU with hidden_dim=64 for speed.

Key invariant: regardless of input sequence length L, the compressor always
produces exactly (compression_steps, hidden_dim) — the bottleneck shape.

The compressor training objective: MSE(Adapter(compressed), Adapter(full)).
After training, compressed latents should feed the adapter and produce outputs
that are close to those produced by the full (uncompressed) latents.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from libucks.thinking.compressor import LatentCompressor
from libucks.thinking.communication_adapter import CommunicationAdapter
from libucks.thinking.training.train_adapter import ContrastiveAdapterTrainer


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
COMPRESS_K = 8
OUTPUT_LEN = 8   # CommunicationAdapter output_len


def _make_compressor(k: int = COMPRESS_K) -> LatentCompressor:
    return LatentCompressor(hidden_dim=HIDDEN_DIM, compression_steps=k, num_heads=4)


def _make_adapter() -> CommunicationAdapter:
    return CommunicationAdapter(hidden_dim=HIDDEN_DIM, output_len=OUTPUT_LEN, num_heads=4)


def _make_trainer(adapter: CommunicationAdapter) -> ContrastiveAdapterTrainer:
    return ContrastiveAdapterTrainer(adapter, temperature=0.07, lr=1e-3, device="cpu")


# ---------------------------------------------------------------------------
# LatentCompressor — shape contract
# ---------------------------------------------------------------------------

class TestLatentCompressor:
    def test_output_shape_from_long_input(self):
        """Explicit requirement: (512, 2048) → (8, 2048)."""
        comp = LatentCompressor(hidden_dim=2048, compression_steps=8, num_heads=4)
        x = torch.randn(512, 2048)
        out = comp(x)
        assert out.shape == (8, 2048)

    def test_output_shape_from_short_input(self):
        comp = _make_compressor()
        x = torch.randn(5, HIDDEN_DIM)
        out = comp(x)
        assert out.shape == (COMPRESS_K, HIDDEN_DIM)

    def test_output_shape_from_unit_input(self):
        """Even a single token compresses to K steps."""
        comp = _make_compressor()
        x = torch.randn(1, HIDDEN_DIM)
        out = comp(x)
        assert out.shape == (COMPRESS_K, HIDDEN_DIM)

    def test_output_shape_invariant_to_input_length(self):
        """Core contract: output shape is independent of L."""
        comp = _make_compressor()
        for L in [3, 10, 50, 100, 500]:
            x = torch.randn(L, HIDDEN_DIM)
            out = comp(x)
            assert out.shape == (COMPRESS_K, HIDDEN_DIM), (
                f"Expected ({COMPRESS_K}, {HIDDEN_DIM}) for L={L}, got {out.shape}"
            )

    def test_five_times_size_reduction(self):
        """5x+ compression ratio required by the plan."""
        comp = _make_compressor(k=8)
        input_len = 50  # 50 × 64 → 8 × 64 is 6.25x
        x = torch.randn(input_len, HIDDEN_DIM)
        out = comp(x)
        ratio = input_len / out.shape[0]
        assert ratio >= 5, f"Expected >= 5x compression, got {ratio:.2f}x"

    def test_output_is_deterministic(self):
        """Same input → same output (eval mode, no dropout)."""
        comp = _make_compressor()
        comp.eval()
        x = torch.randn(30, HIDDEN_DIM)
        out1 = comp(x)
        out2 = comp(x)
        assert torch.allclose(out1, out2)

    def test_output_is_float_tensor(self):
        comp = _make_compressor()
        x = torch.randn(20, HIDDEN_DIM)
        out = comp(x)
        assert out.dtype == torch.float32

    def test_learnable_query_vectors_exist(self):
        comp = _make_compressor()
        assert hasattr(comp, "query_vectors")
        assert isinstance(comp.query_vectors, nn.Parameter)
        assert comp.query_vectors.shape == (1, COMPRESS_K, HIDDEN_DIM)

    def test_parameter_count_is_positive(self):
        comp = _make_compressor()
        total = sum(p.numel() for p in comp.parameters())
        assert total > 0

    def test_gradient_flows_to_query_vectors(self):
        """Gradient must reach the learned query bottleneck."""
        comp = _make_compressor()
        x = torch.randn(20, HIDDEN_DIM)
        out = comp(x)
        out.sum().backward()
        assert comp.query_vectors.grad is not None, (
            "No gradient reached query_vectors — backward pass is broken"
        )

    def test_accepts_batched_input(self):
        """(1, L, d) batched input should also work."""
        comp = _make_compressor()
        x = torch.randn(1, 30, HIDDEN_DIM)
        out = comp(x)
        assert out.shape == (COMPRESS_K, HIDDEN_DIM)

    def test_compression_steps_attribute(self):
        comp = LatentCompressor(hidden_dim=64, compression_steps=4, num_heads=4)
        assert comp.compression_steps == 4


# ---------------------------------------------------------------------------
# Compressor training via ContrastiveAdapterTrainer
# ---------------------------------------------------------------------------

class TestCompressorTrainingStep:
    def test_train_compressor_step_returns_float(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        comp = _make_compressor()
        opt = AdamW(comp.parameters(), lr=1e-3)

        full_latents = [torch.randn(50, HIDDEN_DIM)]
        loss = trainer.train_compressor_step(comp, opt, full_latents)
        assert isinstance(loss, float)

    def test_train_compressor_step_is_finite(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        comp = _make_compressor()
        opt = AdamW(comp.parameters(), lr=1e-3)

        full_latents = [torch.randn(50, HIDDEN_DIM) for _ in range(3)]
        loss = trainer.train_compressor_step(comp, opt, full_latents)
        assert math.isfinite(loss)

    def test_adapter_weights_unchanged_during_compressor_training(self):
        """Only compressor params should be updated — adapter stays frozen."""
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        comp = _make_compressor()
        opt = AdamW(comp.parameters(), lr=1e-3)

        initial = {n: p.clone() for n, p in adapter.named_parameters()}
        full_latents = [torch.randn(40, HIDDEN_DIM)]

        for _ in range(5):
            trainer.train_compressor_step(comp, opt, full_latents)

        for n, p in adapter.named_parameters():
            assert torch.equal(p, initial[n]), (
                f"Adapter parameter '{n}' changed during compressor training"
            )

    def test_compressor_weights_are_modified(self):
        """After a training step, compressor params must change."""
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        comp = _make_compressor()
        opt = AdamW(comp.parameters(), lr=1e-3)

        initial_q = comp.query_vectors.clone()
        full_latents = [torch.randn(40, HIDDEN_DIM)]
        trainer.train_compressor_step(comp, opt, full_latents)

        assert not torch.equal(comp.query_vectors, initial_q), (
            "query_vectors unchanged after a training step"
        )

    def test_compressor_loss_decreases_over_steps(self):
        """MSE loss should trend downward as compressor learns to mimic full latents."""
        torch.manual_seed(7)
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        comp = _make_compressor()
        opt = AdamW(comp.parameters(), lr=5e-3)

        # Fixed latents — compressor should overfit to these quickly
        full_latents = [torch.randn(50, HIDDEN_DIM)]

        losses = []
        for _ in range(60):
            losses.append(trainer.train_compressor_step(comp, opt, full_latents))

        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        assert last_10 < first_10, (
            f"Compressor loss should decrease: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )


# ---------------------------------------------------------------------------
# Config — compression_steps field
# ---------------------------------------------------------------------------

class TestCompressionConfig:
    def test_default_compression_steps_is_8(self):
        from libucks.config import ModelConfig
        cfg = ModelConfig()
        assert cfg.compression_steps == 8

    def test_compression_steps_zero_is_allowed(self):
        """0 means disabled — should not raise."""
        from libucks.config import ModelConfig
        cfg = ModelConfig(compression_steps=0)
        assert cfg.compression_steps == 0

    def test_compression_steps_custom_value(self):
        from libucks.config import ModelConfig
        cfg = ModelConfig(compression_steps=16)
        assert cfg.compression_steps == 16

    def test_compression_steps_preserved_through_merge(self):
        """_merge() must forward compression_steps from TOML data."""
        from libucks.config import _merge, ModelConfig
        cfg = _merge(ModelConfig, {"compression_steps": 4})
        assert cfg.compression_steps == 4


# ---------------------------------------------------------------------------
# LatentStrategy — compressor integration
# ---------------------------------------------------------------------------

class TestLatentStrategyCompressorIntegration:
    """Verify reason() compresses when a LatentCompressor is injected."""

    HIDDEN_DIM = 64
    SEQ_LEN = 20

    def _make_mock_mgr(self):
        mgr = MagicMock()
        mgr.device = "cpu"

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, self.SEQ_LEN)),
            "attention_mask": torch.ones(1, self.SEQ_LEN, dtype=torch.long),
        }
        mgr.get_tokenizer.return_value = mock_tokenizer

        hidden_states = tuple(
            torch.randn(1, self.SEQ_LEN, self.HIDDEN_DIM) for _ in range(5)
        )
        mock_output = MagicMock()
        mock_output.hidden_states = hidden_states
        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mgr.get_model.return_value = mock_model

        return mgr

    async def test_reason_without_compressor_returns_full_seq_len(self):
        from libucks.thinking.latent_strategy import LatentStrategy
        strategy = LatentStrategy(model_manager=self._make_mock_mgr())
        result = await strategy.reason("q", "ctx")
        assert result.shape == (self.SEQ_LEN, self.HIDDEN_DIM)

    async def test_reason_with_compressor_returns_compressed_shape(self):
        from libucks.thinking.latent_strategy import LatentStrategy
        comp = LatentCompressor(hidden_dim=self.HIDDEN_DIM, compression_steps=4, num_heads=4)
        strategy = LatentStrategy(model_manager=self._make_mock_mgr(), compressor=comp)
        result = await strategy.reason("q", "ctx")
        assert result.shape == (4, self.HIDDEN_DIM), (
            f"Expected (4, {self.HIDDEN_DIM}), got {result.shape}"
        )

    async def test_encode_is_not_affected_by_compressor(self):
        """encode() should never compress — raw hidden states only."""
        from libucks.thinking.latent_strategy import LatentStrategy
        comp = LatentCompressor(hidden_dim=self.HIDDEN_DIM, compression_steps=4, num_heads=4)
        strategy = LatentStrategy(model_manager=self._make_mock_mgr(), compressor=comp)
        result = await strategy.encode("hello")
        # encode() should return full SEQ_LEN, not compressed
        assert result.shape == (self.SEQ_LEN, self.HIDDEN_DIM)
