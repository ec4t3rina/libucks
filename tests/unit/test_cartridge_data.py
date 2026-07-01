"""CM-A.1 unit tests for self-study query generation (no model load)."""
from __future__ import annotations

from libucks.thinking.training.self_study import (
    _extract_identifiers,
    generate_self_study_queries,
)

_CODE = (
    "PANIC_RADIUS = 3\n"
    "def relay_message(agent, message):\n"
    "    if agent.state == AgentState.STRANDED:\n"
    "        return None\n"
    "    return propagate(message, PANIC_RADIUS)\n"
)


def test_extract_identifiers_finds_symbols():
    idents = _extract_identifiers(_CODE)
    assert "PANIC_RADIUS" in idents
    assert "relay_message" in idents or "propagate" in idents
    # stopwords excluded
    assert "return" not in idents
    assert "None" not in idents


def test_template_only_returns_exact_count():
    qs = generate_self_study_queries(_CODE, 12)  # no model → templates
    assert len(qs) == 12
    assert all(q.endswith("?") or q.endswith(".") for q in qs)
    assert all(len(q) >= 8 for q in qs)


def test_queries_are_deduped():
    qs = generate_self_study_queries(_CODE, 20)
    assert len(qs) == len(set(q.lower() for q in qs))


def test_empty_ish_bucket_still_yields_queries():
    qs = generate_self_study_queries("x = 1", 5)
    assert len(qs) == 5  # falls back to "this module" token
