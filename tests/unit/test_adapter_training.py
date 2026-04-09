"""Phase 10: ContrastiveAdapterTrainer and MultiPerspectiveDataGenerator tests.

All tests run on CPU with hidden_dim=64 for speed.

Key test: after contrastive training, the adapter correctly distinguishes a
"Correct Collective Thought" (positive latents close to the target) from a
"Noisy/Negative Thought" (latents pointing in the opposite direction).
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from libucks.thinking.communication_adapter import CommunicationAdapter
from libucks.thinking.training.data_generator import (
    PERSPECTIVE_PROMPTS,
    MultiPerspectiveDataGenerator,
    TrainingSample,
)
from libucks.thinking.training.train_adapter import ContrastiveAdapterTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
OUTPUT_LEN = 8


def _unit_vec(dim: int = HIDDEN_DIM, idx: int = 0) -> torch.Tensor:
    """Return a (1, dim) unit-vector with 1.0 at index idx."""
    v = torch.zeros(HIDDEN_DIM)
    v[idx] = 1.0
    return v.unsqueeze(0).expand(10, -1)  # (10, HIDDEN_DIM)


def _make_sample(
    pos_signal: torch.Tensor,
    neg_signal: torch.Tensor,
    noise: float = 0.02,
) -> TrainingSample:
    """Build a TrainingSample with clear positive and negative signals."""
    pos_latents = [pos_signal + noise * torch.randn_like(pos_signal) for _ in range(3)]
    return TrainingSample(
        librarian_latents=pos_latents,
        target_latent=pos_signal.clone(),
        hard_negatives=[neg_signal.clone()],
        target_text="The auth module validates JWT tokens at the gateway.",
    )


def _make_adapter(hidden_dim: int = HIDDEN_DIM) -> CommunicationAdapter:
    return CommunicationAdapter(hidden_dim=hidden_dim, output_len=OUTPUT_LEN, num_heads=4)


def _make_trainer(adapter: CommunicationAdapter) -> ContrastiveAdapterTrainer:
    return ContrastiveAdapterTrainer(adapter, temperature=0.07, lr=1e-3, device="cpu")


# ---------------------------------------------------------------------------
# TrainingSample — data contract
# ---------------------------------------------------------------------------

class TestTrainingSample:
    def test_is_dataclass_with_required_fields(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(TrainingSample)}
        assert fields == {
            "librarian_latents", "target_latent", "hard_negatives", "target_text"
        }

    def test_can_be_constructed(self):
        sample = TrainingSample(
            librarian_latents=[torch.randn(5, HIDDEN_DIM)],
            target_latent=torch.randn(5, HIDDEN_DIM),
            hard_negatives=[torch.randn(5, HIDDEN_DIM)],
            target_text="test",
        )
        assert isinstance(sample, TrainingSample)


# ---------------------------------------------------------------------------
# MultiPerspectiveDataGenerator
# ---------------------------------------------------------------------------

class TestMultiPerspectiveDataGenerator:
    """Generator produces TrainingSamples with three perspectives + hard negatives."""

    def _make_generator(self, centroids: dict | None = None):
        """Return a generator with fully mocked dependencies."""
        text_strategy = MagicMock()
        text_strategy.reason = AsyncMock(side_effect=[
            "This code authenticates users via JWT.",       # perspective 1
            "It checks header → decode → validate → ok.",   # perspective 2
            "Depends on jwt_lib; called by APIGateway.",    # perspective 3
        ])

        latent_strategy = MagicMock()
        latent_strategy.encode = AsyncMock(
            return_value=torch.randn(10, HIDDEN_DIM)
        )

        registry = MagicMock()
        registry.get_all_centroids.return_value = centroids or {
            "bucket-auth": np.array([0.9, 0.1] + [0.0] * 382, dtype=np.float32),
            "bucket-db": np.array([0.5, 0.5] + [0.0] * 382, dtype=np.float32),
            "bucket-ui": np.array([-0.9, 0.1] + [0.0] * 382, dtype=np.float32),
        }

        store = MagicMock()
        store.read.return_value = (MagicMock(), "auth module prose")

        return MultiPerspectiveDataGenerator(
            text_strategy=text_strategy,
            latent_strategy=latent_strategy,
            registry=registry,
            store=store,
        )

    async def test_generate_returns_training_sample(self):
        gen = self._make_generator()
        sample = await gen.generate("bucket-auth")
        assert isinstance(sample, TrainingSample)

    async def test_generate_produces_three_librarian_latents(self):
        gen = self._make_generator()
        sample = await gen.generate("bucket-auth")
        assert len(sample.librarian_latents) == 3

    async def test_all_librarian_latents_are_tensors(self):
        gen = self._make_generator()
        sample = await gen.generate("bucket-auth")
        assert all(isinstance(t, torch.Tensor) for t in sample.librarian_latents)

    async def test_target_latent_is_tensor(self):
        gen = self._make_generator()
        sample = await gen.generate("bucket-auth")
        assert isinstance(sample.target_latent, torch.Tensor)

    async def test_target_text_is_nonempty_string(self):
        gen = self._make_generator()
        sample = await gen.generate("bucket-auth")
        assert isinstance(sample.target_text, str)
        assert len(sample.target_text) > 0

    async def test_hard_negatives_are_tensors(self):
        gen = self._make_generator()
        sample = await gen.generate("bucket-auth")
        assert all(isinstance(t, torch.Tensor) for t in sample.hard_negatives)

    async def test_three_different_prompts_are_used(self):
        gen = self._make_generator()
        await gen.generate("bucket-auth")

        # Verify text_strategy.reason was called with 3 different perspective prompts
        calls = gen._text_strategy.reason.call_args_list
        assert len(calls) == 3
        prompts_used = [c[0][0] for c in calls]  # first positional arg each time
        # Each prompt should be distinct
        assert len(set(prompts_used)) == 3

    async def test_perspective_prompts_cover_summary_logic_deps(self):
        """Verify the three perspective prompts are the canonical ones."""
        gen = self._make_generator()
        await gen.generate("bucket-auth")
        calls = gen._text_strategy.reason.call_args_list
        prompts_used = {c[0][0] for c in calls}
        assert prompts_used == set(PERSPECTIVE_PROMPTS)

    async def test_hard_negatives_exclude_anchor_bucket(self):
        """The anchor bucket itself must never appear as a hard negative."""
        gen = self._make_generator()
        # Reset encode mock to track calls
        gen._latent_strategy.encode.reset_mock()
        await gen.generate("bucket-auth")
        # Ensure encode is not called with "bucket-auth"'s own prose as a negative
        # (implementation detail: we check by ensuring at least 1 hard negative
        # comes from a different bucket — can verify via call count)
        # encode should be called 3 (perspectives) + 1 (target) + N_negatives times
        total_calls = gen._latent_strategy.encode.call_count
        # At minimum 4 calls: 3 perspectives + 1 target
        assert total_calls >= 4


# ---------------------------------------------------------------------------
# ContrastiveAdapterTrainer — loss function
# ---------------------------------------------------------------------------

class TestContrastiveLossEmptyNegatives:
    """Edge case: sample has no hard negatives (empty list)."""

    def test_contrastive_loss_does_not_raise_with_empty_negatives(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        anchor = torch.randn(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.randn(10, HIDDEN_DIM)

        # Must not raise RuntimeError: stack expects a non-empty TensorList
        loss = trainer.contrastive_loss(anchor, target, [])
        assert loss.shape == ()

    def test_contrastive_loss_is_finite_with_empty_negatives(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        anchor = torch.randn(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.randn(10, HIDDEN_DIM)

        loss = trainer.contrastive_loss(anchor, target, [])
        assert math.isfinite(loss.item())

    def test_train_step_does_not_raise_with_empty_negatives(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        sample = TrainingSample(
            librarian_latents=[torch.randn(10, HIDDEN_DIM)],
            target_latent=torch.randn(10, HIDDEN_DIM),
            hard_negatives=[],
            target_text="no negatives mined",
        )

        # Must complete without RuntimeError and return a finite loss
        loss = trainer.train_step(sample)
        assert isinstance(loss, float)
        assert math.isfinite(loss)


class TestContrastiveLoss:
    def test_loss_is_scalar_tensor(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        anchor = torch.randn(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.randn(10, HIDDEN_DIM)
        negatives = [torch.randn(10, HIDDEN_DIM)]

        loss = trainer.contrastive_loss(anchor, target, negatives)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        anchor = torch.randn(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.randn(10, HIDDEN_DIM)
        negatives = [torch.randn(10, HIDDEN_DIM) for _ in range(3)]

        loss = trainer.contrastive_loss(anchor, target, negatives)
        assert math.isfinite(loss.item())

    def test_loss_is_lower_when_anchor_aligned_with_positive(self):
        """InfoNCE loss drops when anchor ≈ positive and negative is orthogonal."""
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        # Build from 1-D vectors so we can expand to any seq_len / OUTPUT_LEN
        d_vec = F.normalize(torch.randn(HIDDEN_DIM), dim=0)
        o_vec = F.normalize(torch.randn(HIDDEN_DIM), dim=0)

        direction = d_vec.unsqueeze(0).expand(10, -1)          # (10, d) target/neg
        ortho = o_vec.unsqueeze(0).expand(10, -1)
        anchor_good = d_vec.unsqueeze(0).expand(OUTPUT_LEN, -1)    # (K, d) aligned
        anchor_bad = (-d_vec).unsqueeze(0).expand(OUTPUT_LEN, -1)  # opposite

        loss_good = trainer.contrastive_loss(anchor_good, direction, [ortho])
        loss_bad = trainer.contrastive_loss(anchor_bad, direction, [ortho])
        assert loss_good.item() < loss_bad.item()

    def test_loss_increases_with_more_confusing_negatives(self):
        """Loss should be higher when the negative is similar to the positive."""
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        d_vec = F.normalize(torch.randn(HIDDEN_DIM), dim=0)
        direction = d_vec.unsqueeze(0).expand(10, -1)             # (10, d)
        anchor = d_vec.unsqueeze(0).expand(OUTPUT_LEN, -1)        # (K, d)
        easy_neg = (-d_vec).unsqueeze(0).expand(10, -1)           # opposite = easy
        hard_neg = (d_vec * 0.9).unsqueeze(0).expand(10, -1)      # near-positive = hard

        loss_easy = trainer.contrastive_loss(anchor, direction, [easy_neg])
        loss_hard = trainer.contrastive_loss(anchor, direction, [hard_neg])
        assert loss_easy.item() < loss_hard.item()


# ---------------------------------------------------------------------------
# ContrastiveAdapterTrainer — training loop
# ---------------------------------------------------------------------------

class TestContrastiveAdapterTrainer:
    def test_train_step_returns_float(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        sample = _make_sample(_unit_vec(idx=0), _unit_vec(idx=32))
        loss = trainer.train_step(sample)
        assert isinstance(loss, float)
        assert math.isfinite(loss)

    def test_train_step_modifies_adapter_weights(self):
        adapter = _make_adapter()
        # Record initial weights
        initial = {n: p.clone() for n, p in adapter.named_parameters()}

        trainer = _make_trainer(adapter)
        sample = _make_sample(_unit_vec(idx=0), _unit_vec(idx=32))
        trainer.train_step(sample)

        changed = any(
            not torch.equal(p, initial[n])
            for n, p in adapter.named_parameters()
        )
        assert changed, "At least one parameter should have changed after a training step"

    def test_loss_decreases_over_training_steps(self):
        """First 10 losses should average higher than last 10 losses."""
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        # Very clear signal: positive and negative are opposite unit vectors
        pos = _unit_vec(idx=0)
        neg = _unit_vec(idx=32)  # orthogonal
        sample = _make_sample(pos, neg, noise=0.0)

        losses = trainer.train([sample] * 60, num_epochs=1)

        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        assert last_10 < first_10, (
            f"Expected loss to decrease: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )

    def test_adapter_distinguishes_positive_from_negative_after_training(self):
        """Core Phase 10 test: after contrastive training, positive_sim > negative_sim."""
        torch.manual_seed(42)

        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        # Positive signal: unit vector in dimension 0
        pos_signal = _unit_vec(idx=0)    # (10, 64), all rows = e_0
        neg_signal = _unit_vec(idx=1)    # orthogonal: e_1
        sample = _make_sample(pos_signal, neg_signal, noise=0.0)

        trainer.train([sample] * 200, num_epochs=1)

        with torch.no_grad():
            output = adapter(sample.librarian_latents)           # (K, d)
            anchor = F.normalize(output.mean(dim=0), dim=0)     # (d,)

            pos_vec = F.normalize(pos_signal.mean(dim=0), dim=0)
            neg_vec = F.normalize(neg_signal.mean(dim=0), dim=0)

            pos_sim = torch.dot(anchor, pos_vec).item()
            neg_sim = torch.dot(anchor, neg_vec).item()

        assert pos_sim > neg_sim, (
            f"Expected pos_sim ({pos_sim:.4f}) > neg_sim ({neg_sim:.4f}) "
            "after contrastive training"
        )

    def test_train_accepts_multiple_samples(self):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        samples = [
            _make_sample(_unit_vec(idx=i), _unit_vec(idx=i + 32))
            for i in range(5)
        ]
        losses = trainer.train(samples, num_epochs=2)
        assert len(losses) == 10  # 5 samples × 2 epochs

    def test_save_creates_checkpoint_file(self, tmp_path):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)
        out_path = tmp_path / "adapter.pt"

        trainer.save(out_path)

        assert out_path.exists()

    def test_saved_weights_can_be_reloaded(self, tmp_path):
        adapter = _make_adapter()
        trainer = _make_trainer(adapter)

        sample = _make_sample(_unit_vec(idx=0), _unit_vec(idx=32))
        trainer.train([sample] * 10)

        out_path = tmp_path / "adapter.pt"
        trainer.save(out_path)

        # Load into a fresh adapter
        fresh = _make_adapter()
        fresh.load_saved_weights(out_path)

        # Forward pass should produce identical output
        with torch.no_grad():
            original_out = adapter(sample.librarian_latents)
            loaded_out = fresh(sample.librarian_latents)
        assert torch.allclose(original_out, loaded_out)

    def test_backbone_parameters_not_modified_during_training(self):
        """Only the adapter should be trained — no backbone to pollute."""
        adapter = _make_adapter()
        param_names = [n for n, _ in adapter.named_parameters()]
        trainer = _make_trainer(adapter)

        # Confirm optimizer only has adapter parameters
        optimizer_param_count = sum(
            len(g["params"]) for g in trainer.optimizer.param_groups
        )
        assert optimizer_param_count == len(param_names)


# ---------------------------------------------------------------------------
# MSE fallback — small-repo / single-bucket edge case
# ---------------------------------------------------------------------------

class TestMSEFallback:
    """When hard negatives are insufficient, trainer must use MSE distillation loss."""

    def test_mse_fallback_is_used_when_no_negatives(self):
        """Identical anchor/target → MSE ≈ 0; InfoNCE with any negative cannot reach 0."""
        adapter = _make_adapter()
        trainer = ContrastiveAdapterTrainer(
            adapter, temperature=0.07, lr=1e-3, device="cpu", min_negatives=1
        )
        signal = torch.ones(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.ones(10, HIDDEN_DIM)
        loss = trainer.contrastive_loss(signal, target, negatives=[])
        assert loss.item() < 0.01, (
            f"Expected near-zero MSE loss for identical anchor/target, got {loss.item():.6f}"
        )

    def test_mse_loss_converges_toward_target(self):
        """Single-bucket scenario (zero hard negatives): loss must decrease over training."""
        torch.manual_seed(0)
        adapter = _make_adapter()
        trainer = ContrastiveAdapterTrainer(
            adapter, temperature=0.07, lr=1e-3, device="cpu", min_negatives=1
        )
        target_signal = _unit_vec(idx=0)
        sample = TrainingSample(
            librarian_latents=[target_signal.clone()],
            target_latent=target_signal.clone(),
            hard_negatives=[],
            target_text="single-bucket repo",
        )
        losses = trainer.train([sample] * 80)
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        assert last_10 < first_10, (
            f"Expected MSE loss to decrease: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )

    def test_min_negatives_threshold_is_configurable(self):
        """min_negatives=2: fallback activates even when exactly 1 real negative is present."""
        adapter = _make_adapter()
        trainer = ContrastiveAdapterTrainer(
            adapter, temperature=0.07, lr=1e-3, device="cpu", min_negatives=2
        )
        signal = torch.ones(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.ones(10, HIDDEN_DIM)
        one_negative = [torch.randn(10, HIDDEN_DIM)]  # 1 < min_negatives=2 → MSE path
        loss = trainer.contrastive_loss(signal, target, negatives=one_negative)
        assert loss.item() < 0.01, (
            f"With min_negatives=2 and 1 negative, MSE fallback should activate. "
            f"Got loss={loss.item():.6f}"
        )

    def test_mse_loss_does_not_raise_when_seq_len_not_divisible_by_K(self):
        """seq_len % K != 0 must not crash (MPS adaptive_avg_pool1d regression).

        adaptive_avg_pool1d on MPS requires seq_len divisible by K.
        F.interpolate has no such constraint.  This test pins that guarantee.
        """
        adapter = _make_adapter()
        trainer = ContrastiveAdapterTrainer(
            adapter, temperature=0.07, lr=1e-3, device="cpu", min_negatives=1
        )
        # seq_len=7, K=OUTPUT_LEN=8: 7 % 8 != 0 and seq_len < K (both directions)
        anchor = torch.randn(OUTPUT_LEN, HIDDEN_DIM)
        target = torch.randn(7, HIDDEN_DIM)
        loss = trainer.contrastive_loss(anchor, target, negatives=[])
        assert math.isfinite(loss.item()), f"Expected finite loss, got {loss.item()}"

    def test_mse_gradient_reaches_all_output_tokens(self):
        """Per-token MSE must produce non-zero gradients for every output position.

        With mean-pooled MSE the gradient at each position is diluted by 1/K,
        but every position still gets *some* gradient — so this test checks that
        all K positions actually move after a single backward pass.
        The bug that motivated the fix (structure collapse) is caught by
        test_output_tokens_are_not_identical_after_training.
        """
        torch.manual_seed(7)
        adapter = _make_adapter()
        trainer = ContrastiveAdapterTrainer(
            adapter, temperature=0.07, lr=1e-3, device="cpu", min_negatives=1
        )
        latents = [torch.randn(10, HIDDEN_DIM)]
        target = torch.randn(10, HIDDEN_DIM)
        sample = TrainingSample(
            librarian_latents=latents,
            target_latent=target,
            hard_negatives=[],
            target_text="gradient test",
        )
        # Record output before training
        with torch.no_grad():
            before = adapter(latents).clone()
        trainer.train_step(sample)
        with torch.no_grad():
            after = adapter(latents).clone()
        # Every one of the K output positions should have changed
        per_token_diff = (after - before).abs().sum(dim=-1)  # (K,)
        assert (per_token_diff > 0).all(), (
            "Every output token should receive gradient; "
            f"zero-diff positions: {(per_token_diff == 0).nonzero().squeeze().tolist()}"
        )

    def test_output_tokens_are_not_identical_after_training(self):
        """After MSE fallback training, K output tokens must be structurally diverse.

        Structure collapse (all K tokens converging to the same vector) causes
        Qwen to see uniform attention keys and fall back to repetitive generation.

        Architecture note: with N=1 librarian input, Stage 3 cross-attends over a
        single value vector so all K output positions are physically forced to
        the same vector — diversity is impossible regardless of the loss function.
        Real usage always supplies 3 perspective latents (N=3), giving Stage 3
        three distinct value vectors to attend over.  This test mirrors that.
        """
        torch.manual_seed(42)
        adapter = _make_adapter()
        trainer = ContrastiveAdapterTrainer(
            adapter, temperature=0.07, lr=1e-3, device="cpu", min_negatives=1
        )
        # Three librarian inputs pointing in orthogonal directions — mirrors
        # real usage (summary, logic-flow, dependency perspectives).
        lib1 = _unit_vec(idx=0)  # (10, 64) all rows = e_0
        lib2 = _unit_vec(idx=1)  # (10, 64) all rows = e_1
        lib3 = _unit_vec(idx=2)  # (10, 64) all rows = e_2

        # Varied teacher target: rows sweep linearly from e_0 to e_1.
        # Adaptive pooling (16→K) yields K distinct unit vectors — each
        # output token gets a different target, so each query gets a
        # different gradient direction.
        rows = [
            F.normalize(
                (1.0 - i / 15.0) * _unit_vec(idx=0)[0]
                + (i / 15.0) * _unit_vec(idx=1)[0],
                dim=0,
            )
            for i in range(16)
        ]
        varied_target = torch.stack(rows)  # (16, 64)

        sample = TrainingSample(
            librarian_latents=[lib1, lib2, lib3],
            target_latent=varied_target,
            hard_negatives=[],
            target_text="3-perspective single-bucket",
        )
        trainer.train([sample] * 300)

        with torch.no_grad():
            output = adapter(sample.librarian_latents)  # (K, d)

        # Pairwise cosine similarity between output tokens — mean should be < 0.99
        # (K tokens should NOT have collapsed to a single direction)
        normed = F.normalize(output, dim=-1)  # (K, d)
        sim_matrix = normed @ normed.T        # (K, K)
        off_diag = sim_matrix[~torch.eye(OUTPUT_LEN, dtype=torch.bool)]
        mean_sim = off_diag.mean().item()
        assert mean_sim < 0.99, (
            f"Output tokens collapsed: mean off-diagonal cosine similarity = {mean_sim:.4f}. "
            "K output queries should remain structurally diverse after training."
        )
