"""Tests for CommunicationAdapter Interlat-Lite updates.

Changes verified:
  - Default num_heads raised from 4 → 8 (per Interlat paper)
  - New frame(soft_prompt, bop_embed, eop_embed) method:
      output = [e(<bop>), h_1, ..., h_K, e(<eop>)]   shape (K+2, d)
"""
import torch
import pytest
from libucks.thinking.communication_adapter import CommunicationAdapter


D = 64
K = 8  # output_len


@pytest.fixture
def adapter():
    return CommunicationAdapter(hidden_dim=D, output_len=K)


@pytest.fixture
def soft_prompt():
    torch.manual_seed(0)
    return torch.randn(K, D)


@pytest.fixture
def bop_embed():
    torch.manual_seed(10)
    return torch.randn(D)


@pytest.fixture
def eop_embed():
    torch.manual_seed(11)
    return torch.randn(D)


# ── num_heads default ───────────────────────────────────────────────────────


def test_default_num_heads_is_8(adapter):
    """Default adapter must use 8 attention heads for all layers."""
    assert adapter.pool_attn.num_heads == 8
    for layer in adapter.inter_attn_layers:
        assert layer.num_heads == 8
    assert adapter.output_attn.num_heads == 8


def test_custom_num_heads_respected():
    a = CommunicationAdapter(hidden_dim=D, output_len=K, num_heads=4)
    assert a.pool_attn.num_heads == 4


# ── frame() shape ───────────────────────────────────────────────────────────


def test_frame_total_length(adapter, soft_prompt, bop_embed, eop_embed):
    """Framed output length == K + 2."""
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)
    assert framed.shape[0] == K + 2


def test_frame_hidden_dim_preserved(adapter, soft_prompt, bop_embed, eop_embed):
    """Framed output hidden dim == D."""
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)
    assert framed.shape[1] == D


# ── frame() content ─────────────────────────────────────────────────────────


def test_frame_prepends_bop_token(adapter, soft_prompt, bop_embed, eop_embed):
    """First row of framed output equals bop_embed."""
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)
    assert torch.allclose(framed[0], bop_embed)


def test_frame_appends_eop_token(adapter, soft_prompt, bop_embed, eop_embed):
    """Last row of framed output equals eop_embed."""
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)
    assert torch.allclose(framed[-1], eop_embed)


def test_frame_middle_is_soft_prompt(adapter, soft_prompt, bop_embed, eop_embed):
    """Rows 1..(K) of framed output equal the original soft_prompt."""
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)
    assert torch.allclose(framed[1:-1], soft_prompt)


def test_frame_does_not_mutate_soft_prompt(adapter, soft_prompt, bop_embed, eop_embed):
    """frame() must not modify soft_prompt in-place."""
    sp_copy = soft_prompt.clone()
    adapter.frame(soft_prompt, bop_embed, eop_embed)
    assert torch.allclose(soft_prompt, sp_copy)


def test_frame_1d_embeds_broadcast(adapter, soft_prompt):
    """frame() accepts 1-D bop/eop vectors of shape (D,)."""
    bop = torch.randn(D)
    eop = torch.randn(D)
    framed = adapter.frame(soft_prompt, bop, eop)
    assert framed.shape == (K + 2, D)


# ── forward() still works ───────────────────────────────────────────────────


def test_forward_then_frame_end_to_end(adapter, bop_embed, eop_embed):
    """forward() + frame() pipeline produces correct final shape."""
    reps = [torch.randn(5, D), torch.randn(3, D)]
    soft = adapter(reps)                          # (K, D)
    framed = adapter.frame(soft, bop_embed, eop_embed)  # (K+2, D)
    assert framed.shape == (K + 2, D)
