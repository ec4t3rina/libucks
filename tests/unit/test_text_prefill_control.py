"""The text-in-prompt ceiling control must be exact, cheap, and non-accumulating.

WHY THIS ARM EXISTS. CM-B.0i's ceiling was 13/26 — barely half, with the ENTIRE
bucket in cache. Every compression number in the log is a fraction of that, so if
the ceiling is itself a harness artifact the whole log is measured against a bent
ruler. The control feeds the same bucket text as prompt tokens read with full
attention. Causal attention makes it mathematically IDENTICAL to a full-cache
prefix, so any gap localises a bug in the cache path — the bf16 -> CPU -> float32
-> index_select -> bf16 round-trip, or the separator junction.

WHY IT IS BUILT THIS WAY. The naive version (one 4,626-token forward per fixture)
exhausted 16 GB plus 20 GB of swap and wedged after 2 of 26 fixtures: with
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 the allocator has no ceiling, so 26 large
transient attention workspaces went to swap instead of erroring.

The verbatim prefix is identical across all 26 prompts; only the trailing question
differs. So it is prefilled ONCE into a live DynamicCache (never round-tripped)
and each fixture forwards only its question against a COPY.

THE HAZARD THIS FILE GUARDS. DynamicCache is mutated IN PLACE by the decode loop.
Hand `generate_answer` the same cache object twice and fixture N+1's prefix
silently contains fixture N's decoded answer — every fixture after the first
corrupted, with no error and a plausible-looking score. So the API takes a
FACTORY, not a cache: freshness is structural, not a caller convention.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer


class _Tok:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        self.last_text = text
        return {"input_ids": torch.zeros((1, max(1, len(text) // 4)), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "ANS:" + ",".join(str(i) for i in ids)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.attn_widths: list[int] = []
        self.saw_past: list[object] = []

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                use_cache=None, **kw):
        self.attn_widths.append(int(attention_mask.shape[1]))
        self.saw_past.append(past_key_values)
        logits = torch.full((1, input_ids.shape[1], 32), -1e9)
        logits[0, -1, 0] = 1.0          # EOS immediately: one forward per call
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _cart():
    return KVPrefixCartridge(n_layers=1, n_kv_heads=1, prefix_len=4, head_dim=2,
                             dtype=torch.float32)


class _Factory:
    """Stands in for 'clone the master prefilled cache'."""

    def __init__(self, prefix_len: int = 100) -> None:
        self.prefix_len = prefix_len
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return object(), self.prefix_len      # a DISTINCT object every call


def _trainer(model=None):
    return CartridgeTrainer(model or _Model(), _Tok())


class TestSignature:
    def test_prefix_factory_is_accepted(self):
        sig = inspect.signature(CartridgeTrainer.generate_answer)
        assert "prefix_factory" in sig.parameters
        assert sig.parameters["prefix_factory"].default is None


class TestFreshnessIsStructural:
    def test_factory_is_called_once_per_generation(self):
        f = _Factory()
        t = _trainer()
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        assert f.calls == 1
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        assert f.calls == 2, (
            "a cache reused across calls accumulates the previous answer into the "
            "next fixture's prefix"
        )

    def test_each_generation_receives_a_distinct_cache_object(self):
        m = _Model()
        f = _Factory()
        t = _trainer(m)
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        first, second = m.saw_past[0], m.saw_past[1]
        assert first is not second

    def test_prefix_width_does_not_grow_across_calls(self):
        """The observable symptom of accumulation."""
        m = _Model()
        f = _Factory()
        t = _trainer(m)
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        assert m.attn_widths[0] == m.attn_widths[1]

    def test_trainer_does_not_retain_the_cache(self):
        f = _Factory()
        t = _trainer()
        t.generate_answer(None, "q?", prefix_factory=f, max_new_tokens=1)
        leaked = [k for k, v in vars(t).items()
                  if type(v).__name__ in ("DynamicCache", "object")]
        assert not leaked, f"cache retained on the trainer: {leaked}"


class TestMutuallyExclusiveWithOtherContextPaths:
    def test_cartridge_and_factory_together_raise(self):
        """Both would attach a prefix; the second silently wins."""
        with pytest.raises(ValueError, match="prefix_factory"):
            _trainer().generate_answer(_cart(), "q?", prefix_factory=_Factory(),
                                       max_new_tokens=1)

    def test_verbatim_and_factory_together_raise(self):
        """Context would be counted twice — once as cache, once as prompt text —
        which is not the control and not any other arm either."""
        with pytest.raises(ValueError, match="verbatim"):
            _trainer().generate_answer(None, "q?", verbatim="some text",
                                       prefix_factory=_Factory(), max_new_tokens=1)


class TestAttentionAccounting:
    def test_mask_covers_factory_prefix_plus_query(self):
        m = _Model()
        t = _trainer(m)
        t.generate_answer(None, "q?", prefix_factory=_Factory(prefix_len=100),
                          max_new_tokens=1)
        q_len = int(_Tok()(t._q_text("q?", ""))["input_ids"].shape[1])
        assert m.attn_widths[0] == 100 + q_len

    def test_only_the_question_tail_is_tokenised(self):
        """The prefix is already in the cache. Re-tokenising the verbatim here
        would double it and blow the memory budget this design exists to fix."""
        t = _trainer()
        t.generate_answer(None, "what is X?", prefix_factory=_Factory(),
                          max_new_tokens=1)
        assert t.tokenizer.last_text == t._q_text("what is X?", "")
        assert "verbatim" not in t.tokenizer.last_text


class TestSweepSplitIsVerified:
    """Prefilling head + forwarding tail only equals a single-shot prompt if the
    tokenisation is split-invariant at the junction. BPE is not split-invariant in
    general, so the sweep must CHECK rather than assume."""

    def _src(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[1].parent
                / "scripts/cm_kv_sweep.py").read_text()

    def test_sweep_asserts_token_exactness_of_the_split(self):
        src = self._src()
        assert "token-exact" in src, (
            "cm_kv_sweep must verify head_ids + tail_ids == full prompt ids, or "
            "the control is not comparable to a single-shot text prompt"
        )

    def test_sweep_uses_the_factory_api(self):
        assert "prefix_factory" in self._src()
