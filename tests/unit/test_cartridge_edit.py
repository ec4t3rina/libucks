"""CM-B Stage 1, step 3 — the three cartridge repair methods.

The claim under test in Stage 1: when one chunk of a bucket changes, a cheap
warm-started repair lands close to a full re-distill (~7,200 s) at a fraction
of the cost. Three methods, cheapest first:

  continue  — warm-start from the existing cartridge, retrain only on queries
              whose teacher answer actually changed. The honest baseline.
  slots     — retrain only the prefix slots the edit touches. THE RESEARCH BET:
              a cartridge has no index, so whether the knowledge about one
              chunk is localised at all is the open question.
  lowrank   — freeze the cartridge, learn a small additive correction.
              Fallback if `slots` shows no localisation.

Everything here is CPU-only and model-free. The plumbing that decides *what*
to retrain is separable from the forward passes that do it, and it is where
the experiment can silently produce a meaningless number — so it is the part
that gets pinned.

Two failure modes these tests exist to prevent:

  * A repair that trains on zero changed queries "succeeds" instantly and
    reports a wonderful cost ratio while having done nothing. That is the
    staleness-floor trap from docs/cm-b-plan.md, one step earlier.
  * A "slot-localized" method that in fact updates every slot is just
    continue-training wearing a hat, and would make the headline claim false.
"""
from __future__ import annotations

import pytest
import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.cache_augmentation.cartridge_edit import (
    REPAIR_METHODS,
    LowRankDelta,
    NoChangeDetected,
    RepairResult,
    SlotMask,
    changed_queries,
    slots_for_char_span,
)

N_LAYERS, N_KV_HEADS, PREFIX_LEN, HEAD_DIM = 2, 2, 8, 4


@pytest.fixture
def cart() -> KVPrefixCartridge:
    torch.manual_seed(0)
    return KVPrefixCartridge(
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,
        prefix_len=PREFIX_LEN, head_dim=HEAD_DIM,
    )


def _fill_grads(c: KVPrefixCartridge, per_slot: list[float]) -> None:
    """Give every parameter a gradient whose magnitude varies by slot."""
    g = torch.tensor(per_slot).view(1, 1, -1, 1)
    for i in range(c.n_layers):
        c.k[i].grad = g.expand_as(c.k[i]).clone()
        c.v[i].grad = g.expand_as(c.v[i]).clone()


# ---------------------------------------------------------------------------
# Which queries actually changed
# ---------------------------------------------------------------------------

class TestChangedQueries:
    def test_returns_only_queries_whose_answer_moved(self):
        old = {"q1": "alpha", "q2": "beta", "q3": "gamma"}
        new = {"q1": "alpha", "q2": "BETA!", "q3": "gamma"}
        assert changed_queries(old, new) == ["q2"]

    def test_result_is_sorted_for_reproducibility(self):
        old = {"z": "1", "a": "1", "m": "1"}
        new = {"z": "2", "a": "2", "m": "2"}
        assert changed_queries(old, new) == ["a", "m", "z"]

    def test_a_query_only_in_the_new_set_counts_as_changed(self):
        assert changed_queries({"q1": "a"}, {"q1": "a", "q2": "b"}) == ["q2"]

    def test_a_query_dropped_after_the_edit_is_ignored(self):
        """It has no post-edit target, so it cannot supply a training signal."""
        assert changed_queries({"q1": "a", "q2": "b"}, {"q1": "a"}) == []

    def test_identical_answers_yield_nothing(self):
        same = {"q1": "a", "q2": "b"}
        assert changed_queries(same, dict(same)) == []

    def test_whitespace_only_differences_do_not_count(self):
        """Otherwise decode jitter inflates the changed set and the cost."""
        assert changed_queries({"q": "the answer"}, {"q": "  the answer\n"}) == []


class TestNoChangeIsLoud:
    """An edit that disturbs nothing must not look like a cheap repair."""

    def test_zero_changed_queries_raises(self):
        with pytest.raises(NoChangeDetected) as exc:
            changed_queries({"q": "a"}, {"q": "a"}, require_change=True)
        assert "staleness floor" in str(exc.value).lower()

    def test_the_error_names_how_many_queries_were_compared(self):
        with pytest.raises(NoChangeDetected, match="3"):
            changed_queries(
                {"a": "x", "b": "y", "c": "z"},
                {"a": "x", "b": "y", "c": "z"},
                require_change=True,
            )


# ---------------------------------------------------------------------------
# Slot masking — the research bet
# ---------------------------------------------------------------------------

class TestSlotMask:
    def test_all_slots_is_equivalent_to_no_masking(self, cart):
        m = SlotMask.all_slots(cart)
        assert m.n_trainable_slots == PREFIX_LEN

    def test_explicit_slots_are_the_only_trainable_ones(self, cart):
        m = SlotMask.from_slots(cart, [1, 3])
        assert m.n_trainable_slots == 2
        assert sorted(m.slots) == [1, 3]

    def test_out_of_range_slot_is_rejected(self, cart):
        with pytest.raises(ValueError, match="out of range"):
            SlotMask.from_slots(cart, [PREFIX_LEN])

    def test_empty_slot_set_is_rejected(self, cart):
        """Training zero slots is a no-op dressed as a repair."""
        with pytest.raises(ValueError, match="at least one"):
            SlotMask.from_slots(cart, [])

    def test_masking_zeroes_gradients_outside_the_selection(self, cart):
        _fill_grads(cart, [1.0] * PREFIX_LEN)
        SlotMask.from_slots(cart, [2, 5]).apply(cart)
        for i in range(cart.n_layers):
            for p in (cart.k[i], cart.v[i]):
                kept = p.grad[:, :, [2, 5], :]
                assert torch.all(kept != 0), "selected slots must keep gradient"
                dropped = [s for s in range(PREFIX_LEN) if s not in (2, 5)]
                assert torch.all(p.grad[:, :, dropped, :] == 0)

    def test_masking_is_harmless_when_a_param_has_no_grad(self, cart):
        SlotMask.from_slots(cart, [0]).apply(cart)  # must not raise

    def test_unmasked_slots_do_not_move_under_an_optimizer_step(self, cart):
        """The property that makes 'slot-localized' true rather than a label."""
        before = [p.detach().clone() for p in cart.k]
        opt = torch.optim.SGD(cart.parameters(), lr=1.0)
        _fill_grads(cart, [1.0] * PREFIX_LEN)
        SlotMask.from_slots(cart, [4]).apply(cart)
        opt.step()

        for i in range(cart.n_layers):
            moved = ~torch.isclose(cart.k[i], before[i])
            assert moved[:, :, 4, :].all(), "slot 4 should have been updated"
            others = [s for s in range(PREFIX_LEN) if s != 4]
            assert not moved[:, :, others, :].any(), "frozen slots moved"

    def test_gradient_selection_picks_the_largest_slots(self, cart):
        per_slot = [0.1, 9.0, 0.1, 7.0, 0.1, 0.1, 0.1, 0.1]
        _fill_grads(cart, per_slot)
        assert sorted(SlotMask.from_gradient(cart, top_k=2).slots) == [1, 3]

    def test_gradient_selection_needs_gradients(self, cart):
        with pytest.raises(ValueError, match="no gradients"):
            SlotMask.from_gradient(cart, top_k=2)

    def test_gradient_top_k_is_clamped_to_prefix_len(self, cart):
        _fill_grads(cart, [1.0] * PREFIX_LEN)
        assert SlotMask.from_gradient(cart, top_k=999).n_trainable_slots == PREFIX_LEN


class TestPositionalSlotMapping:
    """`init_from_extracted_kv` copies the first P positions of the bucket's
    real KV, so before training slot i corresponds to source token i. Whether
    that survives distillation is exactly what Stage 1 measures — this is the
    mapping the hypothesis needs, not proof that it holds."""

    def test_char_end_is_exclusive(self):
        """[100, 200) over 800 chars / 8 slots is slot 1 alone — 100 chars each,
        and an end exactly on the boundary belongs to the preceding slot."""
        assert slots_for_char_span(100, 200, 800, 8) == [1]

    def test_a_span_crossing_a_boundary_covers_both_slots(self):
        assert slots_for_char_span(150, 250, 800, 8) == [1, 2]

    def test_pad_widens_the_selection_symmetrically(self):
        """The experiment passes pad=1: missing the one slot that mattered
        would bias the measurement against the localisation hypothesis."""
        assert slots_for_char_span(100, 200, 800, 8, pad=1) == [0, 1, 2]

    def test_pad_is_clamped_at_the_edges(self):
        assert slots_for_char_span(0, 50, 800, 8, pad=2) == [0, 1, 2]
        assert slots_for_char_span(790, 800, 800, 8, pad=2) == [5, 6, 7]

    def test_pad_defaults_to_zero(self):
        """Widening must be a visible choice — it weakens the localized claim."""
        assert slots_for_char_span(150, 250, 800, 8) == \
               slots_for_char_span(150, 250, 800, 8, pad=0)

    def test_negative_pad_is_rejected(self):
        with pytest.raises(ValueError, match="pad"):
            slots_for_char_span(100, 200, 800, 8, pad=-1)

    def test_span_at_the_start_maps_to_the_first_slot(self):
        assert slots_for_char_span(0, 50, 800, 8) == [0]

    def test_span_past_the_prefix_still_yields_a_valid_slot(self):
        """Buckets are longer than P, so late edits must not fall off the end."""
        slots = slots_for_char_span(790, 800, 800, 8)
        assert slots and max(slots) < 8

    def test_whole_file_span_covers_every_slot(self):
        assert slots_for_char_span(0, 800, 800, 8) == list(range(8))

    def test_empty_corpus_is_rejected(self):
        with pytest.raises(ValueError, match="total_chars"):
            slots_for_char_span(0, 10, 0, 8)

    def test_inverted_span_is_rejected(self):
        with pytest.raises(ValueError, match="char_start"):
            slots_for_char_span(200, 100, 800, 8)


# ---------------------------------------------------------------------------
# Low-rank delta — the fallback
# ---------------------------------------------------------------------------

class TestLowRankDelta:
    def test_starts_as_an_exact_no_op(self, cart):
        """A repair that begins by perturbing the cartridge is not a repair."""
        delta = LowRankDelta(cart, rank=2)
        for i in range(cart.n_layers):
            dk, dv = delta.delta_for_layer(i)
            assert torch.all(dk == 0)
            assert torch.all(dv == 0)

    def test_shapes_match_the_cartridge(self, cart):
        delta = LowRankDelta(cart, rank=2)
        for i in range(cart.n_layers):
            dk, dv = delta.delta_for_layer(i)
            assert dk.shape == cart.k[i].shape
            assert dv.shape == cart.v[i].shape

    def test_is_much_smaller_than_the_cartridge(self, cart):
        """The whole point is cost; a delta as big as the prefix buys nothing."""
        delta = LowRankDelta(cart, rank=1)
        cart_params = sum(p.numel() for p in cart.parameters())
        assert delta.n_parameters < cart_params

    def test_becomes_non_zero_once_trained(self, cart):
        delta = LowRankDelta(cart, rank=2)
        opt = torch.optim.SGD(delta.parameters(), lr=0.5)
        loss = sum(delta.delta_for_layer(i)[0].sum() for i in range(cart.n_layers))
        # A zero-init product has zero grad on both factors; the implementation
        # must break that symmetry or the delta can never learn anything.
        loss.backward()
        opt.step()
        assert any(
            torch.any(delta.delta_for_layer(i)[0] != 0) for i in range(cart.n_layers)
        ), "delta stayed identically zero — factor initialisation is degenerate"

    def test_rejects_non_positive_rank(self, cart):
        with pytest.raises(ValueError, match="rank"):
            LowRankDelta(cart, rank=0)

    def test_the_base_cartridge_stays_frozen(self, cart):
        delta = LowRankDelta(cart, rank=2)
        assert all(not p.requires_grad for p in cart.parameters()), (
            "LowRankDelta must freeze the cartridge it wraps"
        )
        assert all(p.requires_grad for p in delta.parameters())


# ---------------------------------------------------------------------------
# Result accounting
# ---------------------------------------------------------------------------

class TestRepairResult:
    def test_cost_ratio_against_a_full_redistill(self):
        r = RepairResult(method="continue", seconds=720.0, n_queries=12,
                         n_trainable_params=100, final_kl=0.2, kl_history=[0.5, 0.2])
        assert r.cost_ratio(full_redistill_seconds=7200.0) == pytest.approx(0.10)

    def test_cost_ratio_rejects_a_zero_baseline(self):
        r = RepairResult(method="continue", seconds=1.0, n_queries=1,
                         n_trainable_params=1, final_kl=0.1, kl_history=[0.1])
        with pytest.raises(ValueError, match="full_redistill_seconds"):
            r.cost_ratio(full_redistill_seconds=0.0)

    def test_zero_query_result_is_flagged_uninformative(self):
        r = RepairResult(method="slots", seconds=0.1, n_queries=0,
                         n_trainable_params=10, final_kl=0.0, kl_history=[])
        assert r.is_uninformative
        assert "no queries" in r.caveat.lower()

    def test_a_normal_result_is_not_flagged(self):
        r = RepairResult(method="slots", seconds=10.0, n_queries=5,
                         n_trainable_params=10, final_kl=0.2, kl_history=[0.4, 0.2])
        assert not r.is_uninformative
        assert r.caveat == ""

    def test_every_declared_method_has_a_name(self):
        assert set(REPAIR_METHODS) == {"continue", "slots", "lowrank"}
