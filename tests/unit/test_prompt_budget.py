"""Silent prompt truncation must be impossible.

This project has now been misled three times by a cap that quietly dropped
content instead of complaining:

  1. `_collect_source_text` truncated instead of slicing on overflow, so buckets
     carried less verbatim than the log claimed (fixed 2026-07-03).
  2. `cm_kv_sweep` inherited `max_chars=4096`, capping a 20,013-char bucket at
     1,021 tokens — caught 45s into a run only because `seq_len=1021` printed.
  3. `generate_answer` tokenises with `truncation=True, max_length=3500` and
     Qwen's `truncation_side='right'`. The stratified bucket is 4,599 tokens, so
     a text-in-prompt arm would silently lose its last ~1,100 tokens — exactly
     where the late stratified fixtures live. That arm exists to check whether
     the 13/26 full-cache ceiling is real; a truncated version of it would
     "confirm" the ceiling by crippling the control.

(3) is the dangerous one, because it fails toward a false positive: the text arm
would score low, agree with the cache arm, and be read as evidence the ceiling is
model-bound when it was really harness-bound.

So the contract is: a prompt over budget RAISES. The caller must widen the budget
deliberately or shorten the context deliberately. No third option.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from libucks.thinking.training.cartridge_trainer import (
    MAX_PROMPT_TOKENS,
    CartridgeTrainer,
)


class _LenTok:
    """Token count tracks text length, so budget overflow is expressible.

    The stub in test_generate_answer_floor returns a fixed 3 tokens regardless of
    input, which cannot exercise a cap.
    """

    eos_token_id = 0
    truncation_side = "right"

    def __init__(self, chars_per_token: int = 1) -> None:
        self.cpt = chars_per_token
        self.last_kwargs: dict = {}

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        self.last_text = text
        self.last_kwargs = {"truncation": truncation, "max_length": max_length}
        n = max(1, len(text) // self.cpt)
        return {"input_ids": torch.zeros((1, n), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "ANS:" + ",".join(str(i) for i in ids)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.prompt_widths: list[int] = []
        self._seq = [11, 0]

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                use_cache=None, **kw):
        self.prompt_widths.append(int(input_ids.shape[1]))
        tok = self._seq[min(len(self.prompt_widths) - 1, len(self._seq) - 1)]
        logits = torch.full((1, input_ids.shape[1], 32), -1e9)
        logits[0, -1, tok] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _trainer(tok=None, model=None):
    return CartridgeTrainer(model or _Model(), tok or _LenTok())


class TestBudgetConstantIsShared:
    def test_default_matches_the_historical_literal(self):
        """3500 was hardcoded in 6 places. Changing the default silently would
        re-scope every prior measurement, so pin it."""
        assert MAX_PROMPT_TOKENS == 3500

    def test_generate_answer_exposes_the_budget(self):
        sig = inspect.signature(CartridgeTrainer.generate_answer)
        assert "max_prompt_tokens" in sig.parameters
        assert sig.parameters["max_prompt_tokens"].default == MAX_PROMPT_TOKENS


class TestOverflowIsLoud:
    def test_overlong_prompt_raises(self):
        t = _trainer(_LenTok())
        with pytest.raises(ValueError) as e:
            t.generate_answer(None, "q?", verbatim="x" * 200, max_prompt_tokens=32)
        assert "32" in str(e.value), "the error must name the budget it exceeded"

    def test_error_reports_the_actual_token_count(self):
        """A message that says only 'too long' forces the caller to guess how
        much to raise the budget by."""
        t = _trainer(_LenTok())
        with pytest.raises(ValueError, match=r"\b2\d\d\b"):
            t.generate_answer(None, "q?", verbatim="y" * 250, max_prompt_tokens=16)

    def test_no_truncation_flag_is_passed_to_the_tokenizer(self):
        """Truncation must not be available as a fallback: if the tokenizer is
        allowed to cut, the guard can be bypassed by a future edit."""
        t = _trainer(_LenTok())
        t.generate_answer(None, "q?", max_new_tokens=1)
        assert not t.tokenizer.last_kwargs.get("truncation")


class TestWithinBudgetIsUnchanged:
    def test_short_prompt_still_decodes(self):
        t = _trainer(_LenTok())
        assert t.generate_answer(None, "q?", max_new_tokens=1) == "ANS:11"

    def test_raised_budget_admits_the_full_context(self):
        """The whole point: the text arm must be able to see all 4,599 tokens."""
        m = _Model()
        t = _trainer(_LenTok(), m)
        t.generate_answer(None, "q?", verbatim="z" * 4599,
                          max_prompt_tokens=8192, max_new_tokens=1)
        assert m.prompt_widths[0] > 4599, (
            "full verbatim must reach the model uncut when the budget allows it"
        )

    def test_exactly_at_budget_is_allowed(self):
        """Off-by-one here would silently shave the last token of a context that
        was deliberately sized to fit."""
        t = _trainer(_LenTok())
        text = "w" * 64
        n = t.tokenizer(t._q_text("q?", text))["input_ids"].shape[1]
        t.generate_answer(None, "q?", verbatim=text, max_prompt_tokens=n,
                          max_new_tokens=1)


class TestSweepNeverBuildsAnOverlongPrompt:
    """This started life as "the sweep must RAISE its budget above 3500", which
    was right for the first design: the text arm put the whole bucket in each
    prompt, so the cap had to move. That design exhausted 16 GB plus 20 GB of swap
    and wedged after 2 of 26 fixtures.

    The arm now prefills the shared prefix once and each fixture forwards only its
    question tail, so no prompt is ever long and the default cap is never in play.
    The invariant worth guarding therefore inverted: the sweep must NOT widen the
    budget, because needing to would mean it had gone back to stuffing the bucket
    into every prompt. See test_text_prefill_control.py for the positive contract.
    """

    def _src(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[1].parent
                / "scripts/cm_kv_sweep.py").read_text()

    def test_sweep_does_not_widen_the_prompt_budget(self):
        assert "max_prompt_tokens=" not in self._src(), (
            "widening the budget means the bucket is back in the prompt; the "
            "one-time prefill exists so that is unnecessary"
        )

    def test_sweep_reports_what_the_naive_prompt_would_have_cost(self):
        """Keep the number in the log even though it is no longer used — it is the
        evidence that the default cap would have truncated by ~24%."""
        assert "text_need" in self._src()
