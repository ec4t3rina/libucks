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
