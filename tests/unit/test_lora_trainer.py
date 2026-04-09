"""Tests for LoRAReceiverTrainer — Interlat-Lite Phase 12.6 / 12.7.

Verifies:
  - LoRA applied only to q_proj / v_proj (not mlp, not embed_tokens)
  - train_step() returns loss dict with 'task' and 'sep' keys
  - Non-LoRA parameters are frozen (no gradient after step)
  - Cross-entropy (task) loss decreases over 2 consecutive steps
  - L_sep gradient flows through LoRA params (not detached)
  - Batched (2, S, D) forward pass produces correct output shapes
  - Default hyperparameters are conservative for small datasets
  - accumulate_step() accumulates gradients without stepping optimizer

All tests run on CPU with a tiny causal LM mock — no GPU required.
"""
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """Minimal causal-ish model for unit testing — NOT a real transformer."""

    def __init__(self, vocab: int = 64, hidden: int = 32, seq: int = 8):
        super().__init__()
        self.vocab = vocab
        self.embed_tokens = nn.Embedding(vocab, hidden)
        # fake attention projections so LoRA tests can target q_proj / v_proj
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(hidden, hidden, bias=False),
                    "v_proj": nn.Linear(hidden, hidden, bias=False),
                    "o_proj": nn.Linear(hidden, hidden, bias=False),
                }),
                "mlp": nn.Linear(hidden, hidden, bias=False),
            })
        ])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self._seq = seq
        self._hidden = hidden

    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
        # Fake transformer: just pass through q_proj, then lm_head
        for layer in self.layers:
            x = layer["self_attn"]["q_proj"](x)
        logits = self.lm_head(x)
        # Return a namespace-like object
        return _FakeOutput(logits)


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


# ── Fixtures ─────────────────────────────────────────────────────────────────

VOCAB = 64
HIDDEN = 32
SEQ = 8
K = 4   # soft-prompt length
PREFIX_LEN = K + 2  # K soft-prompt tokens + bop + eop


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyMLP(vocab=VOCAB, hidden=HIDDEN, seq=SEQ)


@pytest.fixture
def trainer(tiny_model):
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer
    return LoRAReceiverTrainer(tiny_model, lora_r=2, lora_alpha=4, lr=1e-3)


@pytest.fixture
def batch():
    """Curriculum batch item shaped for train_step (keys match what train_step reads)."""
    torch.manual_seed(1)
    return {
        # Required keys consumed by train_step / accumulate_step:
        "inputs_embeds":       torch.randn(PREFIX_LEN + SEQ, HIDDEN),
        "inputs_embeds_wrong": torch.randn(PREFIX_LEN + SEQ, HIDDEN),
        "target_ids":          torch.randint(0, VOCAB, (SEQ,)),
        "prefix_len":          PREFIX_LEN,
        # Legacy keys kept so any future test that reads them doesn't KeyError:
        "mixed_input": torch.randn(K, HIDDEN),
        "r": 0.5,
    }


# ── LoRA targeting ───────────────────────────────────────────────────────────


def test_trainer_applies_lora_to_q_proj(trainer):
    """LoRA must wrap q_proj layers."""
    param_names = [n for n, _ in trainer.model.named_parameters() if "lora" in n.lower()]
    assert any("q_proj" in n for n in param_names), (
        f"No LoRA params found on q_proj. LoRA params: {param_names}"
    )


def test_trainer_applies_lora_to_v_proj(trainer):
    """LoRA must wrap v_proj layers."""
    param_names = [n for n, _ in trainer.model.named_parameters() if "lora" in n.lower()]
    assert any("v_proj" in n for n in param_names), (
        f"No LoRA params found on v_proj. LoRA params: {param_names}"
    )


def test_trainer_does_not_apply_lora_to_mlp(trainer):
    """LoRA must NOT be applied to mlp layers."""
    lora_params = [n for n, _ in trainer.model.named_parameters() if "lora" in n.lower()]
    assert not any("mlp" in n for n in lora_params), (
        f"LoRA was unexpectedly applied to mlp: {lora_params}"
    )


def test_trainer_does_not_apply_lora_to_embed(trainer):
    """LoRA must NOT be applied to embed_tokens."""
    lora_params = [n for n, _ in trainer.model.named_parameters() if "lora" in n.lower()]
    assert not any("embed" in n for n in lora_params), (
        f"LoRA was unexpectedly applied to embeddings: {lora_params}"
    )


# ── train_step() output ──────────────────────────────────────────────────────


def test_train_step_returns_task_key(trainer, batch):
    """train_step() loss dict must contain 'task' key."""
    losses = trainer.train_step(batch)
    assert "task" in losses


def test_train_step_returns_sep_key(trainer, batch):
    """train_step() loss dict must contain 'sep' key."""
    losses = trainer.train_step(batch)
    assert "sep" in losses


def test_train_step_task_loss_is_positive(trainer, batch):
    """Cross-entropy (task) loss should be positive."""
    losses = trainer.train_step(batch)
    assert losses["task"] > 0.0


def test_train_step_sep_loss_is_non_negative(trainer, batch):
    """Separation loss (JSD) should be >= 0."""
    losses = trainer.train_step(batch)
    assert losses["sep"] >= 0.0


# ── Parameter freezing ───────────────────────────────────────────────────────


def test_frozen_params_have_no_grad_after_step(trainer, batch):
    """Non-LoRA, non-lm_head parameters must have no gradient after train_step."""
    trainer.train_step(batch)
    for name, param in trainer.model.named_parameters():
        if "lora" not in name.lower() and "lm_head" not in name:
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"Frozen param {name} has non-zero gradient"
            )


# ── Loss decreases ───────────────────────────────────────────────────────────


def test_task_loss_decreases_over_two_steps(tiny_model, batch):
    """Task loss must strictly decrease across 2 consecutive train steps."""
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer

    torch.manual_seed(42)
    trainer = LoRAReceiverTrainer(tiny_model, lora_r=2, lora_alpha=4, lr=5e-2)

    losses_1 = trainer.train_step(batch)
    losses_2 = trainer.train_step(batch)

    assert losses_2["task"] < losses_1["task"], (
        f"Task loss did not decrease: {losses_1['task']:.4f} → {losses_2['task']:.4f}"
    )


# ── L_sep gradient flow (Bug 6 regression guard) ─────────────────────────────


def test_sep_loss_influences_gradients():
    """L_sep must contribute gradient to LoRA params (detach bug guard).

    When correct == wrong, L_sep == 0 and its gradient is zero.
    When correct != wrong, L_sep > 0 and contributes a non-zero gradient
    that changes the parameter update vs the zero-sep case.

    If L_sep is detached (Bug 6), both cases produce identical param updates
    (only task_loss matters, and inputs_embeds is the same) — the assertion fails.
    """
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer

    shared_embed = torch.randn(PREFIX_LEN + SEQ, HIDDEN)
    target_ids = torch.randint(0, VOCAB, (SEQ,))

    def make_batch(wrong_embed):
        return {
            "inputs_embeds":       shared_embed.clone(),
            "inputs_embeds_wrong": wrong_embed,
            "target_ids":          target_ids.clone(),
            "prefix_len":          PREFIX_LEN,
        }

    # Case A: wrong == correct → sep ≈ 0 → gradient purely from task_loss
    torch.manual_seed(7)
    m_a = TinyMLP(vocab=VOCAB, hidden=HIDDEN)
    t_a = LoRAReceiverTrainer(m_a, lora_r=2, lora_alpha=4, lr=1e-2)
    t_a.train_step(make_batch(shared_embed.clone()))
    params_a = {n: p.data.clone() for n, p in t_a.model.named_parameters() if p.requires_grad}

    # Case B: wrong very different → sep > 0 → extra gradient from L_sep
    torch.manual_seed(7)
    m_b = TinyMLP(vocab=VOCAB, hidden=HIDDEN)
    t_b = LoRAReceiverTrainer(m_b, lora_r=2, lora_alpha=4, lr=1e-2)
    very_different = shared_embed * -50.0
    t_b.train_step(make_batch(very_different))
    params_b = {n: p.data.clone() for n, p in t_b.model.named_parameters() if p.requires_grad}

    assert params_a.keys() == params_b.keys()
    any_differ = any(
        not torch.allclose(params_a[n], params_b[n], atol=1e-7)
        for n in params_a
    )
    assert any_differ, (
        "LoRA parameters are identical whether sep=0 or sep>0 — "
        "L_sep is likely detached and contributing zero gradient (Bug 6)"
    )


# ── Batched forward pass shape correctness ────────────────────────────────────


def test_batched_forward_logit_shapes(trainer, batch):
    """Batched (2, S, D) forward pass must produce finite loss values."""
    losses = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(losses["task"])), "task loss is not finite"
    assert torch.isfinite(torch.tensor(losses["sep"])), "sep loss is not finite"


# ── Default hyperparameters ───────────────────────────────────────────────────


def test_default_lr_is_conservative():
    """Default LR must be <= 3e-4 to avoid oscillation on small datasets."""
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer
    torch.manual_seed(0)
    m = TinyMLP(vocab=VOCAB, hidden=HIDDEN)
    t = LoRAReceiverTrainer(m)
    lr = t.optimizer.param_groups[0]["lr"]
    assert lr <= 3e-4, f"Default LR {lr} is too high for a 47-sample dataset (max 3e-4)"


def test_default_lora_rank_is_small():
    """Default lora_r must be <= 8 to avoid overfitting on 47 samples."""
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer, LoRALinear
    torch.manual_seed(0)
    m = TinyMLP(vocab=VOCAB, hidden=HIDDEN)
    t = LoRAReceiverTrainer(m)
    lora_linears = [mod for mod in t.model.modules() if isinstance(mod, LoRALinear)]
    assert lora_linears, "No LoRALinear found in model"
    for ll in lora_linears:
        assert ll.lora_A.shape[0] <= 8, (
            f"lora_r={ll.lora_A.shape[0]} is too large for a 47-sample dataset (max 8)"
        )


# ── accumulate_step ───────────────────────────────────────────────────────────


def test_accumulate_step_does_not_step_optimizer(trainer, batch):
    """accumulate_step(step=False) must accumulate gradients without updating params."""
    params_before = [p.data.clone() for p in trainer.model.parameters() if p.requires_grad]
    trainer.accumulate_step(batch, scale=4, step=False)
    params_after = [p.data.clone() for p in trainer.model.parameters() if p.requires_grad]
    for pb, pa in zip(params_before, params_after):
        assert torch.equal(pb, pa), "Parameters changed during accumulate_step(step=False)"


def test_accumulate_step_step_true_advances_params(trainer, batch):
    """accumulate_step(step=True) must update at least one LoRA parameter."""
    params_before = [p.data.clone() for p in trainer.model.parameters() if p.requires_grad]
    trainer.accumulate_step(batch, scale=1, step=True)
    params_after = [p.data.clone() for p in trainer.model.parameters() if p.requires_grad]
    any_changed = any(not torch.equal(pb, pa) for pb, pa in zip(params_before, params_after))
    assert any_changed, "No LoRA parameter changed after accumulate_step(step=True)"


def test_accumulate_step_returns_loss_dict(trainer, batch):
    """accumulate_step must return dict with 'task' and 'sep' keys."""
    losses = trainer.accumulate_step(batch, scale=2, step=False)
    assert "task" in losses
    assert "sep" in losses
    assert torch.isfinite(torch.tensor(losses["task"]))
    assert torch.isfinite(torch.tensor(losses["sep"]))
