"""Query-set composition must be observable, not inferred.

CM-B.0b distilled bc6b90e2 on 84 model-written questions and 36 silent template
fill-ins, then the log entry credited the whole run to fact-probing queries.
`generate_self_study_queries` always tops up to `n`, so the caller's
`if not qs` guard could never fire — the shortfall was structurally invisible.

These tests pin the contract that makes it visible.
"""
from __future__ import annotations

import pytest

from libucks.thinking.training.self_study import generate_self_study_queries

TEXT = (
    "# agents.py\n"
    "RELAY_PROBABILITY = 0.8\n"
    "CONFIRMATIONS_REQUIRED = 2\n"
    "class Agent:\n"
    "    def relay(self, message): ...\n"
    "    def garble(self, message): ...\n"
) * 20


class _FakeTok:
    """Minimal tokenizer stand-in; no chat template, so the plain path is used."""

    chat_template = None
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        import torch

        return {"input_ids": torch.zeros((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return self._reply

    def apply_chat_template(self, *a, **k):  # pragma: no cover - chat_template is None
        raise AssertionError("should not be reached")


class _FakeModel:
    """Yields a fixed number of distinct questions, then nothing."""

    def __init__(self, n_questions: int) -> None:
        self._n = n_questions
        self._emitted = 0

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def generate(self, ids, **kw):
        import torch

        return torch.zeros((1, 8), dtype=torch.long)


def _wire(n_questions: int):
    """A (model, tokenizer) pair whose decode yields `n_questions` distinct lines."""
    tok = _FakeTok()
    model = _FakeModel(n_questions)
    lines = [f"What is the value of constant number {i}?" for i in range(n_questions)]
    tok._reply = "\n".join(lines)
    return model, tok


class TestStatsAreReported:
    def test_stats_dict_is_populated_when_passed(self):
        stats: dict = {}
        generate_self_study_queries(TEXT, 10, stats=stats)
        assert set(stats) >= {"model", "template", "requested"}

    def test_template_only_run_reports_zero_model_queries(self):
        stats: dict = {}
        qs = generate_self_study_queries(TEXT, 10, stats=stats)
        assert len(qs) == 10
        assert stats["model"] == 0
        assert stats["template"] == 10
        assert stats["requested"] == 10

    def test_counts_sum_to_the_returned_length(self):
        stats: dict = {}
        qs = generate_self_study_queries(TEXT, 12, stats=stats)
        assert stats["model"] + stats["template"] == len(qs)

    def test_partial_model_shortfall_is_visible(self):
        """The CM-B.0b case: model supplies some, templates silently pad the rest."""
        model, tok = _wire(6)
        stats: dict = {}
        qs = generate_self_study_queries(TEXT, 20, model=model, tokenizer=tok, stats=stats)
        assert len(qs) == 20
        assert stats["model"] == 6, "model contribution must be reported, not inferred"
        assert stats["template"] == 14, "the silent padding must be countable"

    def test_stats_is_optional_and_omitting_it_changes_nothing(self):
        a = generate_self_study_queries(TEXT, 8)
        b = generate_self_study_queries(TEXT, 8, stats={})
        assert a == b


class TestExistingBehaviourUnchanged:
    def test_still_returns_exactly_n(self):
        assert len(generate_self_study_queries(TEXT, 15)) == 15

    def test_queries_are_distinct(self):
        qs = generate_self_study_queries(TEXT, 15)
        assert len(qs) == len({q.lower() for q in qs})

    @pytest.mark.parametrize("n", [1, 5, 30])
    def test_various_sizes(self, n):
        assert len(generate_self_study_queries(TEXT, n)) == n
