"""Exclude fixtures the base model can answer with no memory at all.

The extension set's floor is 4/16: the 3B answers x09, x11, x12 and x15 correctly
cold, because questions like "what are the five states an agent can be in?" invite
guesses that happen to be right. Any score on the full 16 therefore includes items
that measure nothing about the cartridge.

The subset must be decided ONCE across all observed floor runs, not per run.
Choosing it per run would move the denominator between arms and make the scores
non-comparable — the same failure mode as changing the fixture file mid-track. A
fixture is treated as leaky if the floor answered it in ANY run: a question the
model can sometimes guess is not a clean test even when it fails once.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cm_leak_filter",
    Path(__file__).resolve().parents[2] / "scripts/cm_leak_filter.py",
)
clf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clf)


def _run(**verdicts):
    """A floor-results shape: {id: (floor, random, cartridge)}."""
    return {"per_question": [
        {"id": i, "floor": {"grounded": f}, "random": {"grounded": r},
         "cartridge": {"grounded": c}}
        for i, (f, r, c) in verdicts.items()
    ]}


class TestLeakySetIsUnionAcrossRuns:
    def test_fixture_leaking_in_one_run_is_excluded(self):
        a = _run(x1=(True, False, True), x2=(False, False, True))
        b = _run(x1=(False, False, True), x2=(False, False, True))
        assert clf.leaky_ids([a, b]) == {"x1"}

    def test_nothing_leaks_when_floor_never_scores(self):
        a = _run(x1=(False, False, True), x2=(False, False, False))
        assert clf.leaky_ids([a]) == set()

    def test_union_not_intersection(self):
        """Intersection would keep an item that leaked in 2 of 3 runs."""
        runs = [
            _run(x1=(True, False, True), x2=(False, False, True)),
            _run(x1=(True, False, True), x2=(True, False, True)),
        ]
        assert clf.leaky_ids(runs) == {"x1", "x2"}

    def test_empty_input_is_not_an_error(self):
        assert clf.leaky_ids([]) == set()


class TestRestrictedScore:
    def test_drops_only_the_leaky_items(self):
        r = _run(x1=(True, False, True), x2=(False, False, True), x3=(False, False, False))
        got = clf.restricted(r, {"x1"})
        assert got["n"] == 2
        assert got["cartridge"] == 1
        assert got["floor"] == 0

    def test_c_minus_floor_is_recomputed_on_the_subset(self):
        """The point of the exercise: a +1 that came entirely from a leaky item
        must fall to 0 once that item is removed."""
        r = _run(leaky=(True, True, True), clean=(False, False, False))
        full = clf.restricted(r, set())
        sub = clf.restricted(r, {"leaky"})
        assert full["c_minus_floor"] == 0     # floor 1, cartridge 1
        assert sub["n"] == 1
        assert sub["c_minus_floor"] == 0
        assert sub["cartridge"] == 0

    def test_a_real_gain_survives_filtering(self):
        r = _run(leaky=(True, True, True), clean=(False, False, True))
        sub = clf.restricted(r, {"leaky"})
        assert sub["cartridge"] == 1 and sub["floor"] == 0
        assert sub["c_minus_floor"] == 1

    def test_excluding_everything_yields_zero_not_a_crash(self):
        r = _run(x1=(True, False, True))
        sub = clf.restricted(r, {"x1"})
        assert sub["n"] == 0
        assert sub["c_minus_floor"] == 0

    def test_ids_absent_from_the_run_are_ignored(self):
        r = _run(x1=(False, False, True))
        sub = clf.restricted(r, {"not_present"})
        assert sub["n"] == 1


class TestRealData:
    """Against the committed floor snapshots, the known four must be identified."""

    def test_extension_set_leaks_exactly_the_four_known_fixtures(self):
        import json

        root = Path(__file__).resolve().parents[2]
        paths = sorted((root / "tests/eval/results/cm").glob("echoswarm_floor_*ext16.json"))
        if not paths:
            pytest.skip("no extension floor snapshots present")
        runs = [json.loads(p.read_text()) for p in paths]
        leaky = clf.leaky_ids(runs)
        assert {"echoswarm_x09", "echoswarm_x11", "echoswarm_x12",
                "echoswarm_x15"} <= leaky, f"got {sorted(leaky)}"

    def test_original_8_set_does_not_leak(self):
        import json

        root = Path(__file__).resolve().parents[2]
        paths = sorted((root / "tests/eval/results/cm").glob("echoswarm_floor_*orig8.json"))
        if not paths:
            pytest.skip("no orig8 floor snapshots present")
        runs = [json.loads(p.read_text()) for p in paths]
        assert clf.leaky_ids(runs) == set(), (
            "the original 8 had floor 0/8 in every run; if this fails the floor "
            "is no longer zero and every c-floor number needs revisiting"
        )
