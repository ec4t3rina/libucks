"""Query-aware KV selection — the mechanism this project never tested.

WHY. Every selector in cm_kv_prune is query-AGNOSTIC: kv_first/kv_last/kv_stride are
positional, kv_norm is ||K|| magnitude — and kv_norm's own comment concedes it is "a
cheap stand-in for attention importance that needs no second forward pass." The
literature does not use a stand-in. SnapKV / H2O / CompressKV keep the positions the
ACTUAL QUERY attends to and report 97-99% of full-cache accuracy at 3-19% budget.
CM-B.0i measured 8% of ceiling at 3% budget with a dumb selector and concluded no
compression mechanism exists. That conclusion was never tested against an informed
selector.

WHAT kv_attn ACTUALLY MEASURES, stated precisely so the result is not oversold:
scoring requires a forward pass over the FULL cache, so this arm saves nothing at
measurement time. It answers "given the query, do P well-chosen positions suffice?"

  - If YES: the information is concentrated, selection is the mechanism, and this is
    what SnapKV delivers in production (it selects during a prefill it must do anyway).
  - If NO: no subset of that size suffices, and compression at that ratio is
    impossible for this content by ANY selection method. That is a far stronger and
    more general negative than "kv_first is bad at it".

Using the query to select is not leakage — the query is available at inference; only
the question is used, never the answer.

POSITION ORDER IS LOAD-BEARING. Selected positions must stay in ascending document
order. The cache's keys carry RoPE phase from their original positions; reordering
them scrambles the relative geometry the model decodes against. Every selector in
cm_kv_prune sorts for this reason and kv_attn must too.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

N_LAYERS, N_HEADS, HEAD_DIM, SEQ = 3, 2, 4, 10


def _kp():
    spec = importlib.util.spec_from_file_location(
        "cm_kv_prune", ROOT / "scripts/cm_kv_prune.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flat(seq: int = SEQ) -> dict[str, torch.Tensor]:
    """Cache layout matching extract_bucket_kv: layer_<i>_K / layer_<i>_V."""
    f = {}
    for i in range(N_LAYERS):
        # Distinct per position so index_select errors are visible.
        base = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1)
        f[f"layer_{i}_K"] = base.expand(1, N_HEADS, seq, HEAD_DIM).clone() + i * 100
        f[f"layer_{i}_V"] = base.expand(1, N_HEADS, seq, HEAD_DIM).clone() - i * 100
    f["_meta_seq_len"] = torch.tensor(seq)
    f["_meta_n_layers"] = torch.tensor(N_LAYERS)
    return f


class _AttnModel(torch.nn.Module):
    """Returns per-layer attention with a controlled argmax over cache positions.

    `hot[layer]` is the cache position that layer attends to most. A correct global
    selector aggregates across layers; a correct per-layer selector keeps each
    layer's own favourite.
    """

    def __init__(self, hot: list[int], q_len: int = 2, seq: int = SEQ) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.hot, self.q_len, self.seq = hot, q_len, seq
        self.asked_for_attentions: list[bool] = []
        self.saw_past: list[bool] = []

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                use_cache=None, output_attentions=False, **kw):
        self.asked_for_attentions.append(bool(output_attentions))
        self.saw_past.append(past_key_values is not None)
        attns = []
        for layer in range(N_LAYERS):
            a = torch.full((1, N_HEADS, self.q_len, self.seq + self.q_len), 0.01)
            a[:, :, :, self.hot[layer]] = 0.9      # the position this layer wants
            a[:, :, :, self.seq:] = 0.5            # the query's own positions
            attns.append(a)
        logits = torch.zeros(1, input_ids.shape[1], 8)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values,
                               attentions=tuple(attns))


class _Tok:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        ids = torch.zeros((1, 2), dtype=torch.long)
        return {"input_ids": ids} if return_tensors else {"input_ids": [0, 0]}

    def decode(self, ids, skip_special_tokens=True):
        return "x"


class TestExists:
    def test_selector_is_exported(self):
        assert hasattr(_kp(), "select_indices_attn")

    def test_per_layer_builder_is_exported(self):
        assert hasattr(_kp(), "cartridge_from_per_layer_selection")


class TestPicksWhatTheQueryAttendsTo:
    def test_global_selection_includes_every_layers_favourite(self):
        kp = _kp()
        hot = [7, 3, 5]
        idx = kp.select_indices_attn(
            _AttnModel(hot), _Tok(), _flat(), p=3, query="q?",
            device=torch.device("cpu"))
        assert sorted(idx) == sorted(hot), (
            "aggregating over layers must surface all three hot positions"
        )

    def test_single_budget_picks_the_globally_hottest(self):
        kp = _kp()
        # layer 0 and 1 both want position 4, layer 2 wants 9 -> 4 wins on sum
        idx = kp.select_indices_attn(
            _AttnModel([4, 4, 9]), _Tok(), _flat(), p=1, query="q?",
            device=torch.device("cpu"))
        assert idx == [4]

    def test_beats_positional_selection_on_a_late_fact(self):
        """The whole point: kv_first cannot reach position 9 at p=1, kv_attn can."""
        kp = _kp()
        attn = kp.select_indices_attn(
            _AttnModel([9, 9, 9]), _Tok(), _flat(), p=1, query="q?",
            device=torch.device("cpu"))
        first = kp.select_indices(_flat(), "kv_first", 1)
        assert attn == [9] and first == [0]


class TestQueryPositionsAreNotSelectable:
    def test_only_cache_positions_are_returned(self):
        """Attention to the query's own tokens is highest in the stub for positions
        >= seq. Selecting those would index off the end of the cache."""
        kp = _kp()
        idx = kp.select_indices_attn(
            _AttnModel([1, 2, 3]), _Tok(), _flat(), p=4, query="q?",
            device=torch.device("cpu"))
        assert all(0 <= i < SEQ for i in idx), f"out-of-cache index in {idx}"


class TestOrderAndBudget:
    def test_indices_are_ascending(self):
        kp = _kp()
        idx = kp.select_indices_attn(
            _AttnModel([8, 1, 5]), _Tok(), _flat(), p=3, query="q?",
            device=torch.device("cpu"))
        assert idx == sorted(idx), "RoPE phase requires document order"

    def test_no_duplicates(self):
        kp = _kp()
        idx = kp.select_indices_attn(
            _AttnModel([2, 2, 2]), _Tok(), _flat(), p=3, query="q?",
            device=torch.device("cpu"))
        assert len(idx) == len(set(idx))

    def test_budget_over_seq_len_clamps(self):
        kp = _kp()
        idx = kp.select_indices_attn(
            _AttnModel([1, 2, 3]), _Tok(), _flat(), p=999, query="q?",
            device=torch.device("cpu"))
        assert len(idx) == SEQ

    def test_respects_the_budget(self):
        kp = _kp()
        idx = kp.select_indices_attn(
            _AttnModel([1, 2, 3]), _Tok(), _flat(), p=4, query="q?",
            device=torch.device("cpu"))
        assert len(idx) == 4


class TestUsesTheCacheAndAsksForAttention:
    def test_attention_output_is_requested(self):
        kp = _kp()
        m = _AttnModel([1, 2, 3])
        kp.select_indices_attn(m, _Tok(), _flat(), p=2, query="q?",
                              device=torch.device("cpu"))
        assert any(m.asked_for_attentions), "scores require output_attentions=True"

    def test_the_full_cache_is_attached(self):
        """Scoring against no cache would rank nothing."""
        kp = _kp()
        m = _AttnModel([1, 2, 3])
        kp.select_indices_attn(m, _Tok(), _flat(), p=2, query="q?",
                              device=torch.device("cpu"))
        assert m.saw_past and m.saw_past[0] is True


class TestScoresAreReusableAcrossBudgets:
    """Scoring costs a full-cache forward and does not depend on p. A sweep over
    several budgets, or wanting both global and per-layer selection, must be able
    to pay for that forward once."""

    def test_scores_shape_is_layers_by_seq(self):
        kp = _kp()
        s = kp.attn_scores(_AttnModel([1, 2, 3]), _Tok(), _flat(), "q?",
                           device=torch.device("cpu"))
        assert tuple(s.shape) == (N_LAYERS, SEQ)

    def test_scores_exclude_query_positions(self):
        """seq columns only — the stub gives the query's own positions weight."""
        kp = _kp()
        s = kp.attn_scores(_AttnModel([1, 2, 3]), _Tok(), _flat(), "q?",
                           device=torch.device("cpu"))
        assert s.shape[1] == SEQ

    def test_one_forward_serves_every_budget(self):
        kp = _kp()
        m = _AttnModel([9, 9, 9])
        s = kp.attn_scores(m, _Tok(), _flat(), "q?", device=torch.device("cpu"))
        calls = len(m.saw_past)
        for p in (1, 2, 5):
            assert len(kp.select_from_scores(s, p)) == p
        assert len(m.saw_past) == calls, "selection must not re-run the model"

    def test_select_from_scores_matches_the_one_shot_helper(self):
        kp = _kp()
        flat, hot = _flat(), [7, 3, 5]
        s = kp.attn_scores(_AttnModel(hot), _Tok(), flat, "q?",
                           device=torch.device("cpu"))
        for per in (False, True):
            assert kp.select_from_scores(s, 2, per_layer=per) == \
                kp.select_indices_attn(_AttnModel(hot), _Tok(), flat, 2, "q?",
                                       device=torch.device("cpu"), per_layer=per)


class TestSilentDegenerationIsImpossible:
    """SDPA and flash kernels do not produce attention weights. If transformers
    hands back None instead of falling back to eager, every score is zero and
    `topk` returns positions 0..p-1 — kv_attn would BE kv_first while being
    reported as query-aware, i.e. it would manufacture the null result it exists
    to test for. This must be an error, never a silent pass.
    """

    def _model_returning(self, attentions):
        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 2)

            def forward(self, input_ids=None, past_key_values=None, **kw):
                return SimpleNamespace(
                    logits=torch.zeros(1, input_ids.shape[1], 8),
                    past_key_values=past_key_values, attentions=attentions)
        return _M()

    def test_empty_tuple_raises(self):
        """THE ACTUAL OBSERVED FAILURE. transformers 5.4.0 with SDPA returns
        `attentions=()` — an empty tuple, not None, and it does NOT fall back to
        eager. Verified directly against gpt2. So the real-world path into silent
        kv_first degeneration is an empty tuple, and it must be caught."""
        kp = _kp()
        with pytest.raises(RuntimeError, match="no attention weights"):
            kp.select_indices_attn(self._model_returning(()), _Tok(), _flat(),
                                   p=2, query="q?", device=torch.device("cpu"))

    def test_none_attentions_raise(self):
        kp = _kp()
        with pytest.raises(RuntimeError, match="no attention weights"):
            kp.select_indices_attn(self._model_returning(None), _Tok(), _flat(),
                                   p=2, query="q?", device=torch.device("cpu"))

    def test_a_none_layer_raises(self):
        kp = _kp()
        good = torch.rand(1, N_HEADS, 2, SEQ + 2)
        with pytest.raises(RuntimeError, match="no attention weights"):
            kp.select_indices_attn(
                self._model_returning((good, None, good)), _Tok(), _flat(),
                p=2, query="q?", device=torch.device("cpu"))

    def test_layer_count_mismatch_raises(self):
        kp = _kp()
        good = torch.rand(1, N_HEADS, 2, SEQ + 2)
        with pytest.raises(RuntimeError, match="cache layers"):
            kp.select_indices_attn(
                self._model_returning((good, good)), _Tok(), _flat(),
                p=2, query="q?", device=torch.device("cpu"))


class TestPerLayerSelection:
    def test_each_layer_keeps_its_own_favourite(self):
        kp = _kp()
        hot = [7, 3, 5]
        per = kp.select_indices_attn(
            _AttnModel(hot), _Tok(), _flat(), p=1, query="q?",
            device=torch.device("cpu"), per_layer=True)
        assert per == [[7], [3], [5]], (
            "per-layer selection is what SnapKV does; layers attend differently"
        )

    def test_returns_one_list_per_layer_each_of_length_p(self):
        kp = _kp()
        per = kp.select_indices_attn(
            _AttnModel([1, 2, 3]), _Tok(), _flat(), p=3, query="q?",
            device=torch.device("cpu"), per_layer=True)
        assert len(per) == N_LAYERS and all(len(x) == 3 for x in per)

    def test_per_layer_lists_are_ascending(self):
        kp = _kp()
        per = kp.select_indices_attn(
            _AttnModel([9, 0, 4]), _Tok(), _flat(), p=2, query="q?",
            device=torch.device("cpu"), per_layer=True)
        assert all(x == sorted(x) for x in per)


class TestPerLayerBuilder:
    def test_each_layer_gets_its_own_positions(self):
        kp = _kp()
        flat = _flat()
        tmpl = kp.KVPrefixCartridge(n_layers=N_LAYERS, n_kv_heads=N_HEADS,
                                    prefix_len=2, head_dim=HEAD_DIM,
                                    dtype=torch.float32)
        per = [[0, 1], [4, 5], [8, 9]]
        c = kp.cartridge_from_per_layer_selection(flat, per, tmpl,
                                                 torch.device("cpu"))
        for i, want in enumerate(per):
            got = c.k[i][0, 0, :, 0].tolist()
            assert got == [float(w) + i * 100 for w in want], (
                f"layer {i} holds {got}, expected positions {want}"
            )

    def test_prefix_len_matches_the_budget(self):
        kp = _kp()
        tmpl = kp.KVPrefixCartridge(n_layers=N_LAYERS, n_kv_heads=N_HEADS,
                                    prefix_len=2, head_dim=HEAD_DIM,
                                    dtype=torch.float32)
        c = kp.cartridge_from_per_layer_selection(
            _flat(), [[1, 2], [3, 4], [5, 6]], tmpl, torch.device("cpu"))
        assert c.prefix_len == 2

    def test_ragged_selection_is_rejected(self):
        """Unequal per-layer counts cannot form a rectangular cartridge, and
        silently padding would fabricate positions."""
        kp = _kp()
        tmpl = kp.KVPrefixCartridge(n_layers=N_LAYERS, n_kv_heads=N_HEADS,
                                    prefix_len=2, head_dim=HEAD_DIM,
                                    dtype=torch.float32)
        with pytest.raises(ValueError, match="same length"):
            kp.cartridge_from_per_layer_selection(
                _flat(), [[1, 2], [3], [5, 6]], tmpl, torch.device("cpu"))

    def test_wrong_layer_count_is_rejected(self):
        kp = _kp()
        tmpl = kp.KVPrefixCartridge(n_layers=N_LAYERS, n_kv_heads=N_HEADS,
                                    prefix_len=2, head_dim=HEAD_DIM,
                                    dtype=torch.float32)
        with pytest.raises(ValueError, match="layer"):
            kp.cartridge_from_per_layer_selection(
                _flat(), [[1, 2], [3, 4]], tmpl, torch.device("cpu"))
