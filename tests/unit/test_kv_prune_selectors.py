"""Contract for training-free KV-cache selection.

The arms only mean something if each selector does what its name claims and if
position ORDER is preserved. The cache's keys carry RoPE-encoded positional
information, so returning indices out of order would hand the model a scrambled
document while still reporting a plausible-looking score — a silent-corruption
failure of exactly the kind this project keeps hitting.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "cm_kv_prune",
    Path(__file__).resolve().parents[2] / "scripts/cm_kv_prune.py",
)
kp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kp)

SEQ = 20
LAYERS = 3
HEADS = 2
DIM = 4


def _flat(seq: int = SEQ) -> dict[str, torch.Tensor]:
    out = {}
    for i in range(LAYERS):
        out[f"layer_{i}_K"] = torch.randn(1, HEADS, seq, DIM)
        out[f"layer_{i}_V"] = torch.randn(1, HEADS, seq, DIM)
    out["_meta_seq_len"] = torch.tensor(seq)
    out["_meta_n_layers"] = torch.tensor(LAYERS)
    return out


class TestLayerKeys:
    def test_ignores_meta_scalars(self):
        assert kp.layer_keys(_flat()) == [
            ("layer_0_K", "layer_0_V"),
            ("layer_1_K", "layer_1_V"),
            ("layer_2_K", "layer_2_V"),
        ]


class TestSelectors:
    @pytest.mark.parametrize("how", kp.SELECTORS)
    def test_returns_exactly_p_positions(self, how):
        idx = kp.select_indices(_flat(), how, 8)
        assert len(idx) == 8, f"{how} returned {len(idx)}"

    @pytest.mark.parametrize("how", kp.SELECTORS)
    def test_indices_are_in_range(self, how):
        idx = kp.select_indices(_flat(), how, 8)
        assert all(0 <= i < SEQ for i in idx)

    @pytest.mark.parametrize("how", kp.SELECTORS)
    def test_indices_are_unique(self, how):
        idx = kp.select_indices(_flat(), how, 8)
        assert len(set(idx)) == len(idx)

    @pytest.mark.parametrize("how", kp.SELECTORS)
    def test_position_order_is_preserved(self, how):
        """RoPE-encoded keys carry order; scrambling it corrupts the document."""
        idx = kp.select_indices(_flat(), how, 8)
        assert idx == sorted(idx), f"{how} returned positions out of order"

    @pytest.mark.parametrize("how", kp.SELECTORS)
    def test_p_larger_than_sequence_is_clamped(self, how):
        idx = kp.select_indices(_flat(seq=5), how, 50)
        assert len(idx) == 5 and idx == sorted(idx)

    def test_first_takes_the_head(self):
        assert kp.select_indices(_flat(), "kv_first", 4) == [0, 1, 2, 3]

    def test_last_takes_the_tail(self):
        assert kp.select_indices(_flat(), "kv_last", 4) == [16, 17, 18, 19]

    def test_stride_spans_both_endpoints(self):
        """The point of stride is covering the tail that kv_first cannot reach."""
        idx = kp.select_indices(_flat(), "kv_stride", 5)
        assert idx[0] == 0 and idx[-1] == SEQ - 1

    def test_norm_picks_the_largest_keys(self):
        flat = _flat()
        for i in range(LAYERS):
            flat[f"layer_{i}_K"] = torch.zeros(1, HEADS, SEQ, DIM)
            flat[f"layer_{i}_K"][:, :, [3, 11, 17], :] = 9.0
        assert kp.select_indices(flat, "kv_norm", 3) == [3, 11, 17]

    def test_unknown_selector_raises(self):
        with pytest.raises(ValueError):
            kp.select_indices(_flat(), "kv_nonsense", 4)


class TestCartridgeFromSelection:
    def _template(self, p=8):
        from libucks.cache_augmentation.cartridge import KVPrefixCartridge

        return KVPrefixCartridge(n_layers=LAYERS, n_kv_heads=HEADS, prefix_len=p,
                                 head_dim=DIM, dtype=torch.float32)

    def test_slots_are_the_real_cache_positions_verbatim(self):
        flat = _flat()
        idx = [2, 5, 9]
        c = kp.cartridge_from_selection(flat, idx, self._template(), torch.device("cpu"))
        assert c.prefix_len == 3
        expected = flat["layer_1_K"].float()[:, :, idx, :]
        assert torch.allclose(c.k[1], expected), (
            "the arm must contain the REAL activations, not an approximation — "
            "that is the entire premise of a training-free selection"
        )

    def test_values_are_selected_too_not_just_keys(self):
        flat = _flat()
        idx = [1, 4]
        c = kp.cartridge_from_selection(flat, idx, self._template(), torch.device("cpu"))
        assert torch.allclose(c.v[0], flat["layer_0_V"].float()[:, :, idx, :])

    def test_geometry_matches_the_template_except_prefix_len(self):
        c = kp.cartridge_from_selection(_flat(), [0, 1, 2], self._template(),
                                        torch.device("cpu"))
        t = self._template()
        assert (c.n_layers, c.n_kv_heads, c.head_dim) == (t.n_layers, t.n_kv_heads, t.head_dim)

    def test_kv_first_reproduces_init_from_extracted_kv(self):
        """kv_first must be identical to the untrained warm start, or the claim
        'cartridge - kv_first is what training adds' is false."""
        flat = _flat()
        p = 8
        warm = self._template(p)
        warm.init_from_extracted_kv(flat)
        sel = kp.cartridge_from_selection(
            flat, kp.select_indices(flat, "kv_first", p), self._template(p),
            torch.device("cpu"),
        )
        for i in range(LAYERS):
            assert torch.allclose(warm.k[i], sel.k[i])
            assert torch.allclose(warm.v[i], sel.v[i])
