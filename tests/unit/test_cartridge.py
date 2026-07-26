"""CM-A.0 contract tests for the trainable KV-prefix cartridge.

Written test-first (TDD): defines the KVPrefixCartridge API that CM-A.1's
`libucks/cache_augmentation/cartridge.py` must satisfy. These are fast tensor-
mechanics checks — no 3B model load. The real frozen-model context-distillation
step (KL decreases; base-model grads stay None) lives in
tests/integration/test_cartridge_smoke.py.

Qwen 2.5-3B KV geometry (from Phase 4-C): 36 layers, 2 KV heads, head_dim 128.
"""
from __future__ import annotations

import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge

N_LAYERS = 4          # tiny stand-in; real use passes 36
N_KV_HEADS = 2
HEAD_DIM = 128
PREFIX_LEN = 8        # real use P=64


def _make_flat_kv(seq_len: int = 16, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Fake extract_bucket_kv output for init tests."""
    flat: dict[str, torch.Tensor] = {}
    for i in range(N_LAYERS):
        flat[f"layer_{i}_K"] = torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM, dtype=dtype)
        flat[f"layer_{i}_V"] = torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM, dtype=dtype)
    flat["_meta_seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
    flat["_meta_n_layers"] = torch.tensor([N_LAYERS], dtype=torch.int32)
    return flat


def _make_cartridge() -> KVPrefixCartridge:
    return KVPrefixCartridge(
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,
        prefix_len=PREFIX_LEN, head_dim=HEAD_DIM, dtype=torch.float32,
    )


def test_parameter_shapes_and_trainable():
    cart = _make_cartridge()
    params = list(cart.parameters())
    # One K + one V trainable tensor per layer.
    assert len(params) == 2 * N_LAYERS
    for p in params:
        assert p.requires_grad
        assert tuple(p.shape) == (1, N_KV_HEADS, PREFIX_LEN, HEAD_DIM)


def test_to_dynamic_cache_shape():
    cart = _make_cartridge()
    cache = cart.to_dynamic_cache(device=torch.device("cpu"))
    assert hasattr(cache, "layers")
    assert len(cache.layers) == N_LAYERS
    # Each layer's cached K is the prefix length along the seq dim (dim=2).
    assert cache.layers[0].keys.shape[2] == PREFIX_LEN
    assert cache.layers[0].keys.shape[1] == N_KV_HEADS


def test_init_from_extracted_kv_copies_prefix():
    cart = _make_cartridge()
    flat = _make_flat_kv(seq_len=16)
    cart.init_from_extracted_kv(flat)
    # After init, layer-0 trainable K equals the first PREFIX_LEN positions
    # of the extracted layer-0 K.
    got = dict(cart.named_parameters())
    k0 = got["k.0"] if "k.0" in got else got["k_prefix.0"]
    expected = flat["layer_0_K"][:, :, :PREFIX_LEN, :]
    assert torch.allclose(k0.detach(), expected, atol=1e-5)


def test_init_from_short_kv_pads_or_truncates():
    """Buckets shorter than PREFIX_LEN must still init without crashing."""
    cart = _make_cartridge()
    flat = _make_flat_kv(seq_len=PREFIX_LEN // 2)  # shorter than prefix
    cart.init_from_extracted_kv(flat)  # should not raise
    for p in cart.parameters():
        assert tuple(p.shape) == (1, N_KV_HEADS, PREFIX_LEN, HEAD_DIM)


def test_grads_flow_into_prefix():
    cart = _make_cartridge()
    cache = cart.to_dynamic_cache(device=torch.device("cpu"))
    loss = sum(layer.keys.sum() + layer.values.sum() for layer in cache.layers)
    loss.backward()
    for p in cart.parameters():
        assert p.grad is not None


def test_save_load_roundtrip(tmp_path):
    cart = _make_cartridge()
    flat = _make_flat_kv()
    cart.init_from_extracted_kv(flat)
    path = tmp_path / "bucket.cartridge.safetensors"
    cart.save(path)

    reloaded = KVPrefixCartridge(
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,
        prefix_len=PREFIX_LEN, head_dim=HEAD_DIM, dtype=torch.float32,
    )
    reloaded.load(path)
    for a, b in zip(cart.parameters(), reloaded.parameters()):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6)


# ---------------------------------------------------------------------------
# CM-B.0b: load() geometry validation.
#
# save() has always written n_layers / n_kv_heads / prefix_len / head_dim as
# safetensors metadata, but load() never read it and went straight to
# `param.data.copy_(flat[...])`. Tensor.copy_ BROADCASTS, so a file with a
# smaller-but-broadcastable geometry loads silently and corrupts the cartridge
# instead of raising. Verified: copying a (1,1,P,D) source into a (1,2,P,D)
# destination succeeds and duplicates head 0 across both heads.
#
# This matters here specifically because the project runs two receiver sizes
# (Qwen2.5-3B: 36 layers/2 heads, Qwen2.5-0.5B: 24 layers) and PREFIX_LEN is
# duplicated across scripts under a "MUST match" comment. A mismatched load
# must fail loudly.
# ---------------------------------------------------------------------------

import pytest


def test_load_rejects_prefix_len_mismatch(tmp_path):
    saved = KVPrefixCartridge(
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,
        prefix_len=PREFIX_LEN * 2, head_dim=HEAD_DIM, dtype=torch.float32,
    )
    path = tmp_path / "c.safetensors"
    saved.save(path)

    with pytest.raises(ValueError, match="prefix_len"):
        _make_cartridge().load(path)


def test_load_rejects_layer_count_mismatch(tmp_path):
    saved = KVPrefixCartridge(
        n_layers=N_LAYERS * 2, n_kv_heads=N_KV_HEADS,
        prefix_len=PREFIX_LEN, head_dim=HEAD_DIM, dtype=torch.float32,
    )
    path = tmp_path / "c.safetensors"
    saved.save(path)

    # Without validation this silently loads only the first N_LAYERS layers.
    with pytest.raises(ValueError, match="n_layers"):
        _make_cartridge().load(path)


def test_load_rejects_kv_head_mismatch_instead_of_broadcasting(tmp_path):
    saved = KVPrefixCartridge(
        n_layers=N_LAYERS, n_kv_heads=1,
        prefix_len=PREFIX_LEN, head_dim=HEAD_DIM, dtype=torch.float32,
    )
    path = tmp_path / "c.safetensors"
    saved.save(path)

    # copy_ would broadcast 1 head across 2 and duplicate it silently.
    with pytest.raises(ValueError, match="n_kv_heads"):
        _make_cartridge().load(path)


def test_load_error_names_both_geometries(tmp_path):
    """The message must say what was expected and what was found — a bare
    'shape mismatch' is what made the 3B/0.5B confusion expensive to trace."""
    saved = KVPrefixCartridge(
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,
        prefix_len=PREFIX_LEN * 2, head_dim=HEAD_DIM, dtype=torch.float32,
    )
    path = tmp_path / "c.safetensors"
    saved.save(path)

    with pytest.raises(ValueError) as ei:
        _make_cartridge().load(path)
    msg = str(ei.value)
    assert str(PREFIX_LEN) in msg and str(PREFIX_LEN * 2) in msg
    assert str(path.name) in msg or str(path) in msg


def test_load_still_accepts_matching_geometry(tmp_path):
    """Guard against the validation being too strict."""
    a = _make_cartridge()
    with torch.no_grad():
        a.k[0].add_(1.234)
    path = tmp_path / "c.safetensors"
    a.save(path)

    b = _make_cartridge()
    b.load(path)
    assert torch.allclose(a.k[0], b.k[0])


def test_load_tolerates_missing_metadata(tmp_path):
    """Cartridges written before CM-B.0b carry metadata, but a hand-built file
    may not. Absent metadata must fall back to shape checks, not crash."""
    from safetensors.torch import save_file

    src = _make_cartridge()
    flat = {}
    for i in range(N_LAYERS):
        flat[f"k_{i}"] = src.k[i].detach().clone()
        flat[f"v_{i}"] = src.v[i].detach().clone()
    path = tmp_path / "nometa.safetensors"
    save_file(flat, str(path))          # no metadata= argument

    _make_cartridge().load(path)        # must not raise
