"""Tests for L_sep — Conditional Thought Separation Loss.

Simplified version of Interlat's L_sep:
    L_sep = -mean(JSD(p_correct, p_wrong))

where:
  p_correct = model logits given the *correct* latent for this query
  p_wrong   = model logits given a *mismatched* latent from a different query

A positive L_sep means the distributions differ (good — the model is using the
latent signal). Negating it as a loss term rewards divergence.

JSD is the Jensen-Shannon divergence (symmetric, bounded in [0, log 2]):
    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5*(P+Q)
"""
import torch
import pytest
from libucks.thinking.training.losses import separation_loss


VOCAB = 100
SEQ = 10


@pytest.fixture
def uniform_logits():
    """Logits for a near-uniform distribution — low entropy difference."""
    return torch.zeros(SEQ, VOCAB)


@pytest.fixture
def peaked_logits():
    """Logits with a strong peak at token 0 — high confidence."""
    t = torch.full((SEQ, VOCAB), -10.0)
    t[:, 0] = 10.0
    return t


def test_lsep_positive_when_distributions_differ(uniform_logits, peaked_logits):
    """L_sep > 0 when the two logit sequences have different distributions."""
    loss = separation_loss(uniform_logits, peaked_logits)
    assert loss.item() > 0.0


def test_lsep_zero_when_same_distribution(uniform_logits):
    """L_sep ≈ 0 when both logit sequences are identical."""
    loss = separation_loss(uniform_logits, uniform_logits.clone())
    assert loss.item() < 1e-5


def test_lsep_returns_scalar(uniform_logits, peaked_logits):
    """separation_loss returns a scalar (0-dim) tensor."""
    loss = separation_loss(uniform_logits, peaked_logits)
    assert loss.dim() == 0


def test_lsep_gradient_flows(uniform_logits, peaked_logits):
    """Gradient must flow through both logit arguments."""
    lc = uniform_logits.clone().requires_grad_(True)
    lw = peaked_logits.clone().requires_grad_(True)
    loss = separation_loss(lc, lw)
    loss.backward()
    assert lc.grad is not None and lw.grad is not None
    assert lc.grad.abs().sum().item() > 0
    assert lw.grad.abs().sum().item() > 0


def test_lsep_symmetric(uniform_logits, peaked_logits):
    """separation_loss(a, b) == separation_loss(b, a) (JSD is symmetric)."""
    l1 = separation_loss(uniform_logits, peaked_logits)
    l2 = separation_loss(peaked_logits, uniform_logits)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_lsep_non_negative(uniform_logits, peaked_logits):
    """-JSD >= 0 since JSD <= 0 natively; check loss >= 0."""
    loss = separation_loss(uniform_logits, peaked_logits)
    assert loss.item() >= 0.0


def test_lsep_batch_input(uniform_logits, peaked_logits):
    """Works with batch-first input (B, T, V)."""
    lc = uniform_logits.unsqueeze(0).expand(4, -1, -1)   # (4, T, V)
    lw = peaked_logits.unsqueeze(0).expand(4, -1, -1)
    loss = separation_loss(lc, lw)
    assert loss.dim() == 0


def test_lsep_larger_difference_gives_larger_loss():
    """More divergent distributions → higher L_sep."""
    # slightly different
    a = torch.zeros(SEQ, VOCAB)
    b_small = torch.zeros(SEQ, VOCAB)
    b_small[:, 0] = 1.0  # small perturbation
    b_large = torch.zeros(SEQ, VOCAB)
    b_large[:, 0] = 100.0  # large perturbation

    loss_small = separation_loss(a, b_small)
    loss_large = separation_loss(a, b_large)
    assert loss_large.item() > loss_small.item()
