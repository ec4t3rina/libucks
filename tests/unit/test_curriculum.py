"""Tests for CurriculumMixer.

Formula (from Interlat §3.3):
    H^(r) = [e_1, ..., e_{floor(r·K)}] ⊕ [h_{floor(r·K)+1}, ..., h_K]
             ← token embeddings →        ←      latents           →

So:
  r=0 → all latents   (0 token positions)
  r=1 → all tokens    (K token positions)

Training samples r ~ U[0,1] uniformly each step, forcing the model to handle
any mixture — this is the curriculum (not a fixed schedule).
"""
import math
import torch
import pytest
from libucks.thinking.curriculum import CurriculumMixer


K = 8
D = 64


@pytest.fixture
def latents():
    torch.manual_seed(0)
    return torch.randn(K, D)


@pytest.fixture
def tok_embeds():
    torch.manual_seed(1)
    return torch.randn(K, D)


def test_mix_r0_returns_all_latents(latents, tok_embeds):
    """r=0 → floor(0*K)=0 token positions → output equals latents entirely."""
    out = CurriculumMixer.mix(latents, tok_embeds, r=0.0)
    assert torch.allclose(out, latents)


def test_mix_r1_returns_all_tokens(latents, tok_embeds):
    """r=1 → floor(1*K)=K token positions → output equals tok_embeds entirely."""
    out = CurriculumMixer.mix(latents, tok_embeds, r=1.0)
    assert torch.allclose(out, tok_embeds)


def test_mix_midpoint(latents, tok_embeds):
    """r=0.5 → first floor(0.5*K) rows are tok_embeds, rest are latents."""
    r = 0.5
    split = math.floor(r * K)  # 4
    out = CurriculumMixer.mix(latents, tok_embeds, r=r)
    assert torch.allclose(out[:split], tok_embeds[:split])
    assert torch.allclose(out[split:], latents[split:])


def test_mix_shape_preserved(latents, tok_embeds):
    """Output shape (K, D) is invariant to r."""
    for r in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out = CurriculumMixer.mix(latents, tok_embeds, r=r)
        assert out.shape == (K, D), f"shape wrong for r={r}"


def test_mix_dtype_preserved(latents, tok_embeds):
    """Output dtype matches input dtype."""
    out = CurriculumMixer.mix(latents, tok_embeds, r=0.5)
    assert out.dtype == latents.dtype


def test_mix_device_preserved(latents, tok_embeds):
    """Output stays on CPU when inputs are on CPU."""
    out = CurriculumMixer.mix(latents, tok_embeds, r=0.5)
    assert out.device == latents.device


def test_mix_does_not_mutate_inputs(latents, tok_embeds):
    """mix() must not modify the input tensors in-place."""
    lat_copy = latents.clone()
    tok_copy = tok_embeds.clone()
    CurriculumMixer.mix(latents, tok_embeds, r=0.5)
    assert torch.allclose(latents, lat_copy)
    assert torch.allclose(tok_embeds, tok_copy)


def test_mix_r_boundary_values(latents, tok_embeds):
    """r outside [0,1] raises ValueError."""
    with pytest.raises(ValueError):
        CurriculumMixer.mix(latents, tok_embeds, r=-0.1)
    with pytest.raises(ValueError):
        CurriculumMixer.mix(latents, tok_embeds, r=1.1)


def test_mix_shape_mismatch_raises(latents):
    """Mismatched shape raises ValueError."""
    bad = torch.randn(K + 2, D)
    with pytest.raises(ValueError):
        CurriculumMixer.mix(latents, bad, r=0.5)
