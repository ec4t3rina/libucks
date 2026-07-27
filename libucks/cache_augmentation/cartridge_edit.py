"""CM-B Stage 1 — repairing a stale cartridge instead of rebuilding it.

A cartridge is distilled once from a bucket's corpus. When a commit changes
that corpus the cartridge is stale, and today the only remedy is to throw it
away and re-distill — measured at ~7,200 s per bucket. Cartridges at Scale
(arXiv 2606.04557) states the same gap in its own words ("updating or adding a
single document requires re-encoding the entire KV cache") and does not solve
it.

Three repair methods, cheapest first:

    continue   Warm-start from the existing cartridge and retrain only on the
               queries whose teacher answer actually changed. Every parameter
               stays trainable. The honest baseline — if this already wins,
               there is no research result, only good engineering.

    slots      Retrain only the prefix slots the edit touches, freezing the
               rest. THE RESEARCH BET. A cartridge has no index: whatever the
               model learned about chunk c3 is smeared across 2.36M numbers
               with no map back to source. `init_from_extracted_kv` copies the
               first P positions of the bucket's real KV, so a positional
               correspondence exists *before* training. Whether any of it
               survives distillation is the open question this measures.

    lowrank    Freeze the cartridge entirely and learn a small additive
               correction. Fallback if `slots` shows no localisation.

This module deliberately contains no forward passes. Deciding *what* to
retrain is separable from running the training, is where the experiment can
silently produce a meaningless number, and is CPU-testable — so it lives here
and is pinned by tests/unit/test_cartridge_edit.py. The orchestration that
feeds these into CartridgeTrainer lives in scripts/cm_edit_experiment.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from libucks.cache_augmentation.cartridge import KVPrefixCartridge

REPAIR_METHODS: tuple[str, ...] = ("continue", "slots", "lowrank")


class NoChangeDetected(Exception):
    """Raised when an edit moved no teacher answer at all.

    This is the staleness-floor trap, caught early. If nothing changed there is
    nothing to repair, and any repair method will report a wonderful cost ratio
    for having done no work. docs/cm-b-plan.md pre-commits to reporting that
    outcome as *insensitivity*, not as a mechanism — so it must be loud, never
    silently folded into a cheap-looking success.
    """


# ---------------------------------------------------------------------------
# Which queries the edit actually disturbed
# ---------------------------------------------------------------------------

def changed_queries(
    old_answers: dict[str, str],
    new_answers: dict[str, str],
    *,
    require_change: bool = False,
) -> list[str]:
    """Queries whose teacher answer differs before vs after the edit.

    Only these carry a training signal: for every other query the pre-edit
    cartridge is already correct, and retraining on them spends the budget the
    experiment is trying to measure.

    Comparison is whitespace-normalised. Greedy decoding is deterministic, but
    trailing-newline and indentation jitter still shows up between runs, and
    counting that as a change would inflate both the changed set and the repair
    cost — biasing the headline number in the wrong direction.

    Queries absent from `new_answers` are ignored: without a post-edit target
    there is nothing to distil toward.

    Set `require_change=True` at experiment call sites to turn "the edit did
    nothing" into an error instead of a suspiciously cheap success.
    """
    def norm(s: str) -> str:
        return " ".join(s.split())

    changed = sorted(
        q for q, new in new_answers.items()
        if q not in old_answers or norm(old_answers[q]) != norm(new)
    )

    if require_change and not changed:
        raise NoChangeDetected(
            f"the edit changed none of {len(new_answers)} teacher answers. "
            "This is the staleness floor: the cartridge was never disturbed, so "
            "any 'repair' measured here is free by construction and the "
            "datapoint is uninformative. Use a larger edit, or report this as "
            "insensitivity."
        )
    return changed


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------

def slots_for_char_span(
    char_start: int,
    char_end: int,
    total_chars: int,
    prefix_len: int,
    *,
    pad: int = 0,
) -> list[int]:
    """Map a character span in the bucket's verbatim onto prefix slots.

    The hypothesis, not a proof of it. `init_from_extracted_kv` warm-starts the
    cartridge by copying the first `prefix_len` positions of the bucket's real
    extracted KV, so at initialisation slot i holds the KV of source position i.
    This function assumes that correspondence still holds after distillation;
    Stage 1 exists to find out whether it does.

    `char_end` is EXCLUSIVE, matching Python slice convention: a span landing
    exactly on a slot boundary belongs to the preceding slot, so
    [100, 200) over 800 chars and 8 slots is slot 1 alone, not slots 1-2.

    The mapping is proportional, not tokeniser-exact. Slot granularity is
    coarse — P=128 over a ~4,000-char bucket is ~30 chars per slot — so exact
    token offsets would be false precision.

    `pad` widens the result by that many slots on each side, clamped to the
    prefix. Default 0, so the function does exactly what its name says. The
    experiment should pass pad=1: missing the one slot that mattered would make
    slot-localized repair look worse than it is, biasing the measurement
    against the hypothesis, and a couple of extra slots costs little. Padding
    is opt-in and visible in the caller rather than baked in here, because
    every extra slot weakens the "localized" claim and that trade-off should be
    a recorded choice, not a default nobody sees.
    """
    if total_chars <= 0:
        raise ValueError(f"total_chars must be positive, got {total_chars}")
    if char_start > char_end:
        raise ValueError(
            f"char_start ({char_start}) must not exceed char_end ({char_end})"
        )
    if prefix_len <= 0:
        raise ValueError(f"prefix_len must be positive, got {prefix_len}")
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")

    span = max(0.0, min(1.0, char_start / total_chars))
    end = max(0.0, min(1.0, char_end / total_chars))

    lo = int(span * prefix_len)
    hi = int(end * prefix_len)
    # char_end is exclusive: an end exactly on a boundary belongs to the
    # preceding slot.
    if hi > lo and end * prefix_len == hi:
        hi -= 1
    lo = min(lo, prefix_len - 1)
    hi = min(max(hi, lo), prefix_len - 1)

    lo = max(0, lo - pad)
    hi = min(prefix_len - 1, hi + pad)
    return list(range(lo, hi + 1))


@dataclass(frozen=True)
class SlotMask:
    """The set of prefix slots a repair is allowed to modify.

    Enforced by zeroing gradients outside the selection after backward and
    before `optimizer.step()`. Zeroing gradients rather than detaching keeps
    the forward pass identical to an unmasked run, so the KL numbers stay
    comparable across methods — the whole point of the comparison.

    Note this is gradient masking, not true freezing: an optimizer with
    momentum or weight decay can still nudge a zero-gradient parameter. Use
    SGD without momentum, or AdamW with weight_decay=0, for a clean
    measurement. test_unmasked_slots_do_not_move_under_an_optimizer_step pins
    the guarantee for plain SGD.
    """

    slots: tuple[int, ...]
    prefix_len: int

    @property
    def n_trainable_slots(self) -> int:
        return len(self.slots)

    @property
    def fraction(self) -> float:
        """Share of the prefix this repair may touch — the cost story."""
        return len(self.slots) / self.prefix_len

    # -- constructors ---------------------------------------------------

    @classmethod
    def all_slots(cls, cartridge: KVPrefixCartridge) -> "SlotMask":
        """Every slot trainable — equivalent to no masking. The `continue`
        baseline, expressed through the same machinery so the two paths differ
        in exactly one variable."""
        return cls(tuple(range(cartridge.prefix_len)), cartridge.prefix_len)

    @classmethod
    def from_slots(cls, cartridge: KVPrefixCartridge, slots) -> "SlotMask":
        chosen = sorted(set(int(s) for s in slots))
        if not chosen:
            raise ValueError(
                "a SlotMask needs at least one trainable slot; an empty mask is "
                "a no-op that would report as a free repair"
            )
        bad = [s for s in chosen if not 0 <= s < cartridge.prefix_len]
        if bad:
            raise ValueError(
                f"slot(s) {bad} out of range for prefix_len={cartridge.prefix_len}"
            )
        return cls(tuple(chosen), cartridge.prefix_len)

    @classmethod
    def from_gradient(cls, cartridge: KVPrefixCartridge, top_k: int) -> "SlotMask":
        """Select the slots that actually respond to the changed queries.

        The empirical counterpart to `slots_for_char_span`. Run a backward pass
        over the post-edit queries, then call this: it ranks slots by summed
        gradient magnitude across every layer and both K and V. If the
        positional hypothesis holds, this should agree with the char-span
        mapping — and comparing the two IS a result either way.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        scores = torch.zeros(cartridge.prefix_len)
        seen = False
        for i in range(cartridge.n_layers):
            for params in (cartridge.k, cartridge.v):
                g = params[i].grad
                if g is None:
                    continue
                seen = True
                # (1, H, P, D) -> (P,)
                scores += g.detach().abs().sum(dim=(0, 1, 3)).cpu().float()
        if not seen:
            raise ValueError(
                "no gradients on the cartridge — run a backward pass over the "
                "post-edit queries before selecting slots by gradient"
            )

        k = min(top_k, cartridge.prefix_len)
        chosen = torch.topk(scores, k).indices.tolist()
        return cls(tuple(sorted(int(s) for s in chosen)), cartridge.prefix_len)

    # -- application ----------------------------------------------------

    def apply(self, cartridge: KVPrefixCartridge) -> None:
        """Zero gradients outside the selection. Call after backward, before step."""
        keep = torch.zeros(self.prefix_len, dtype=torch.bool)
        keep[list(self.slots)] = True
        for i in range(cartridge.n_layers):
            for params in (cartridge.k, cartridge.v):
                g = params[i].grad
                if g is None:
                    continue
                g[:, :, ~keep.to(g.device), :] = 0


# ---------------------------------------------------------------------------
# Low-rank delta
# ---------------------------------------------------------------------------

class LowRankDelta(nn.Module):
    """A frozen cartridge plus a learned low-rank additive correction.

    Per layer and per tensor: `delta = A @ B` with A (P, r) and B (r, D),
    broadcast across heads. Parameter count drops from P*H*D to r*(P+D) per
    tensor — at P=128, H=2, D=128, r=4 that is 32,768 -> 1,024, a 32x cut.

    A is zero-initialised and B is random, so the product starts at exactly
    zero: the wrapped cartridge is untouched until training begins. A repair
    that starts by perturbing the artifact is not measuring repair.

    Initialising BOTH factors to zero would be the obvious way to guarantee a
    no-op, and it is a trap: d(AB)/dA = B = 0 and d(AB)/dB = A = 0, so the
    delta is stuck at zero forever and "the fallback method didn't help" would
    be an artifact of the initialisation. Pinned by
    test_becomes_non_zero_once_trained.
    """

    def __init__(self, cartridge: KVPrefixCartridge, rank: int = 4) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.rank = rank
        self.n_layers = cartridge.n_layers
        self.prefix_len = cartridge.prefix_len
        self.head_dim = cartridge.head_dim
        self.n_kv_heads = cartridge.n_kv_heads

        # The cartridge is the frozen base; only the factors train.
        for p in cartridge.parameters():
            p.requires_grad_(False)

        P, D = cartridge.prefix_len, cartridge.head_dim

        def factors() -> nn.ParameterList:
            return nn.ParameterList(
                [nn.Parameter(torch.zeros(P, rank)) for _ in range(self.n_layers)]
            )

        def seeds() -> nn.ParameterList:
            return nn.ParameterList(
                [nn.Parameter(torch.randn(rank, D) * 0.02) for _ in range(self.n_layers)]
            )

        self.k_a, self.k_b = factors(), seeds()
        self.v_a, self.v_b = factors(), seeds()

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def delta_for_layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Additive corrections (dk, dv), shaped like the cartridge's tensors."""
        dk = (self.k_a[i] @ self.k_b[i]).unsqueeze(0).unsqueeze(0)
        dv = (self.v_a[i] @ self.v_b[i]).unsqueeze(0).unsqueeze(0)
        shape = (1, self.n_kv_heads, self.prefix_len, self.head_dim)
        return dk.expand(shape), dv.expand(shape)

    def apply_to(self, cartridge: KVPrefixCartridge) -> None:
        """Fold the learned delta into the cartridge in place, then reset it.

        Used once the repair has converged, so the result is an ordinary
        cartridge that any existing loader can read.
        """
        with torch.no_grad():
            for i in range(self.n_layers):
                dk, dv = self.delta_for_layer(i)
                cartridge.k[i].data += dk.to(cartridge.k[i].dtype)
                cartridge.v[i].data += dv.to(cartridge.v[i].dtype)
                self.k_a[i].zero_()
                self.v_a[i].zero_()


# ---------------------------------------------------------------------------
# Result accounting
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """One repair attempt: what it cost and what it bought.

    `final_kl` is KL(repaired ‖ fully-re-distilled) — the primary metric,
    chosen because it needs no generation and therefore carries none of the
    decode-loop variance that manufactured Phase 4-C's phantom win.
    """

    method: str
    seconds: float
    n_queries: int
    n_trainable_params: int
    final_kl: float
    kl_history: list[float] = field(default_factory=list)

    def cost_ratio(self, full_redistill_seconds: float) -> float:
        """Repair cost as a fraction of a full re-distill. The headline number."""
        if full_redistill_seconds <= 0:
            raise ValueError(
                "full_redistill_seconds must be positive — without a measured "
                f"baseline the cost ratio is meaningless (got {full_redistill_seconds})"
            )
        return self.seconds / full_redistill_seconds

    @property
    def is_uninformative(self) -> bool:
        """True when this row must not be read as a success."""
        return self.n_queries == 0

    @property
    def caveat(self) -> str:
        if self.n_queries == 0:
            return (
                "no queries changed — the edit did not disturb the cartridge, so "
                "this repair was free by construction. Report as insensitivity, "
                "not as a result."
            )
        return ""
