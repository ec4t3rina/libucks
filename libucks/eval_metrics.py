"""Grounding metric for the eval harness (Stage 0a).

`_grounding_score` originally did plain case-insensitive substring matching. That
under-counts: a model answering "0.8" fails a fixture expecting "80%", and one
answering "two" fails a fixture expecting "2". Three of CM-A.2's 18 failures are
that artifact rather than wrong answers.

The fix is deliberately conservative, because the whole point is to keep old and
new numbers comparable:

  1. The LITERAL keyword keeps plain substring matching, exactly as before. No
     previously-passing fixture can start failing.
  2. ADDED variants match on word boundaries only. Substring matching for
     generated variants would be actively wrong — keyword "1" produces variant
     "one", and "one" is a substring of "money".

Rule 1 preserves the metric's known false-positive behaviour (keyword "2" hits
inside "120"). That is intentional: fixing it would move every historical
baseline at the same time as this change, and then neither number would be
interpretable.
"""
from __future__ import annotations

import re

# Word <-> digit for the small integers that realistically appear as fixture
# keywords. Deliberately stops at 20; beyond that, keywords are written as digits.
_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_DIGIT_TO_WORD = {v: k for k, v in _WORD_TO_DIGIT.items()}

_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*%$")
_DECIMAL_RE = re.compile(r"^\d*\.\d+$")
_INTEGER_RE = re.compile(r"^\d+$")


def _fmt(value: float) -> str:
    """Format a number without float noise: 0.8 -> '0.8', 30.000000004 -> '30'."""
    return f"{value:g}"


def keyword_variants(kw: str) -> set[str]:
    """Return the lowercased keyword plus any equivalent surface forms.

    >>> sorted(keyword_variants("80%"))
    ['0.8', '80%']
    >>> sorted(keyword_variants("two"))
    ['2', 'two']
    """
    literal = kw.strip().lower()
    out = {literal}

    m = _PERCENT_RE.match(literal)
    if m:
        out.add(_fmt(float(m.group(1)) / 100.0))
        return out

    if _DECIMAL_RE.match(literal):
        out.add(f"{_fmt(float(literal) * 100.0)}%")
        return out

    if _INTEGER_RE.match(literal):
        word = _DIGIT_TO_WORD.get(str(int(literal)))
        if word:
            out.add(word)
        return out

    digit = _WORD_TO_DIGIT.get(literal)
    if digit:
        out.add(digit)

    return out


def _variant_hit(answer_lower: str, variant: str) -> bool:
    """Word-boundary match for a generated variant.

    Boundary-anchored so "one" (variant of "1") does not fire inside "money",
    and "0.8" (variant of "80%") does not fire inside "10.85".
    """
    return re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", answer_lower) is not None


def keyword_hit(answer: str, kw: str) -> bool:
    """True if `kw` — literally, or via an equivalent form — appears in `answer`."""
    answer_lower = answer.lower()
    literal = kw.strip().lower()
    if literal and literal in answer_lower:
        return True
    return any(
        _variant_hit(answer_lower, v)
        for v in keyword_variants(kw)
        if v != literal
    )


def grounding_score(answer: str, answer_keywords: list[str]) -> bool:
    """True when at least 50% of `answer_keywords` are present in `answer`.

    Same threshold and same intent as the original `_grounding_score`; only the
    per-keyword matching is widened.
    """
    if not answer_keywords:
        return False
    hits = sum(1 for kw in answer_keywords if keyword_hit(answer, kw))
    return hits >= len(answer_keywords) / 2.0
