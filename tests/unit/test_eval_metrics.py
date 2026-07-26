"""Contract for the eval grounding metric (Stage 0a).

The original `_grounding_score` did plain case-insensitive substring matching, so
a model that answered "0.8" failed a fixture expecting "80%", and one that
answered "two" failed a fixture expecting "2". Three of CM-A.2's 18 failures are
that artifact, not wrong answers.

This module pins the fix. Two rules, deliberately conservative:

  1. The LITERAL keyword keeps plain substring matching, unchanged, so existing
     numbers stay comparable.
  2. ADDED variants (percent<->decimal, number word<->digit) match on word
     boundaries only. Without that, keyword "1" would generate variant "one",
     which is a substring of "money".
"""
from __future__ import annotations

from libucks.eval_metrics import grounding_score, keyword_variants


class TestKeywordVariants:
    def test_plain_word_has_only_itself(self):
        assert keyword_variants("verbatim") == {"verbatim"}

    def test_percent_gains_decimal_form(self):
        assert "0.8" in keyword_variants("80%")

    def test_decimal_gains_percent_form(self):
        assert "30%" in keyword_variants("0.3")

    def test_digit_gains_word_form(self):
        assert "two" in keyword_variants("2")

    def test_word_gains_digit_form(self):
        assert "2" in keyword_variants("two")

    def test_fractional_percent_converts(self):
        assert "0.125" in keyword_variants("12.5%")

    def test_variants_are_lowercase(self):
        assert all(v == v.lower() for v in keyword_variants("COMPLIANT"))


class TestGroundingScore:
    def test_empty_keywords_is_false(self):
        assert grounding_score("anything at all", []) is False

    def test_literal_substring_still_matches(self):
        # Unchanged behaviour: plain substring, no word boundary required.
        assert grounding_score("the COMPLIANT agent", ["compliant"]) is True

    def test_half_the_keywords_passes(self):
        assert grounding_score("alpha and beta", ["alpha", "beta", "gamma", "delta"]) is True

    def test_below_half_fails(self):
        assert grounding_score("alpha only", ["alpha", "beta", "gamma", "delta"]) is False

    def test_percent_keyword_matches_decimal_answer(self):
        assert grounding_score("relay probability is 0.8 for it", ["80%"]) is True

    def test_digit_keyword_matches_word_answer(self):
        assert grounding_score("at least two distinct sources", ["2"]) is True

    def test_word_number_variant_respects_word_boundaries(self):
        # "one" is a substring of "money" — must NOT count as a hit for "1".
        assert grounding_score("he made money", ["1"]) is False

    def test_variant_does_not_fire_inside_larger_number(self):
        # Variant matching is boundary-anchored: "0.8" must not match "10.85".
        assert grounding_score("the value 10.85 appears", ["80%"]) is False


class TestRealCMA2Cases:
    """Regression cases taken verbatim from tests/eval/results/cm/."""

    def test_echoswarm_01_was_a_false_negative(self):
        answer = (
            "The relay probability for a Compliant agent is 0.8. It modifies the "
            "message by randomly selecting a subset of the received message and "
            "relaying it to the neighbors."
        )
        # "COMPLIANT" already matched; "80%" now matches via 0.8 -> 2/3 >= 50%.
        assert grounding_score(answer, ["80%", "verbatim", "COMPLIANT"]) is True

    def test_echoswarm_02_was_a_false_negative(self):
        answer = (
            "The Skeptical agent will relay the message if it has received the "
            "message from at least two different sources."
        )
        assert grounding_score(answer, ["confirmations", "2", "SKEPTICAL"]) is True

    def test_echoswarm_06_is_still_a_genuine_failure(self):
        answer = (
            "A single Simulation.tick() executes the following sub-steps: "
            "1. Update the state of all agents. 2. Update the state of all intersections."
        )
        assert grounding_score(
            answer, ["_relay_messages", "_move_agents", "_spread_panic"]
        ) is False

    def test_echoswarm_10_is_still_a_genuine_failure(self):
        answer = (
            "The three CERC pillars that Hermes follows are: C (Crisis), "
            "E (Economic), and R (Reputation)."
        )
        assert grounding_score(answer, ["FRAMING", "CLARITY", "CONTENT"]) is False
