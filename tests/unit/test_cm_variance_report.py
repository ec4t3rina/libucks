"""Contract for the variance aggregator.

The whole point of this script is to stop a noisy delta being read as signal, so
its own arithmetic has to be right — particularly the distinction between score
spread and per-fixture churn, which can disagree.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cm_variance_report",
    Path(__file__).resolve().parents[2] / "scripts/cm_variance_report.py",
)
cvr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cvr)


def _draw(label, verdicts: dict[str, bool]):
    return {"label": label, "score": sum(verdicts.values()), "n": len(verdicts),
            "verdicts": verdicts}


class TestSummarise:
    def test_identical_draws_have_zero_spread(self):
        v = {"a": True, "b": False}
        s = cvr.summarise([_draw("s1", v), _draw("s2", dict(v))])
        assert s["range"] == 0
        assert s["unstable"] == []

    def test_score_spread_is_reported(self):
        s = cvr.summarise([
            _draw("s1", {"a": True, "b": True}),
            _draw("s2", {"a": False, "b": False}),
        ])
        assert s["min"] == 0 and s["max"] == 2 and s["range"] == 2

    def test_churn_is_caught_even_when_the_score_is_identical(self):
        """Two fixtures swapping leaves the headline unchanged and the
        per-fixture claim worthless. This is the case a score-only summary
        would miss."""
        s = cvr.summarise([
            _draw("s1", {"a": True, "b": False}),
            _draw("s2", {"a": False, "b": True}),
        ])
        assert s["range"] == 0, "headline score is stable"
        assert sorted(s["unstable"]) == ["a", "b"], "but both fixtures flipped"

    def test_partitions_are_disjoint_and_complete(self):
        s = cvr.summarise([
            _draw("s1", {"a": True, "b": False, "c": True}),
            _draw("s2", {"a": True, "b": False, "c": False}),
        ])
        assert s["always"] == ["a"]
        assert s["never"] == ["b"]
        assert s["unstable"] == ["c"]
        assert len(s["always"]) + len(s["never"]) + len(s["unstable"]) == 3

    def test_stdev_is_none_for_a_single_draw(self):
        s = cvr.summarise([_draw("s1", {"a": True})])
        assert s["stdev"] is None
        assert s["range"] == 0


class TestVerdict:
    def test_single_draw_refuses_to_claim_anything(self):
        s = cvr.summarise([_draw("s1", {"a": True})])
        assert "no error bar" in " ".join(cvr.verdict(s, None)).lower()

    def test_delta_inside_the_spread_is_called_noise(self):
        s = cvr.summarise([
            _draw("s1", {"a": True, "b": True, "c": True, "d": False}),
            _draw("s2", {"a": False, "b": False, "c": False, "d": False}),
        ])
        txt = " ".join(cvr.verdict(s, claim_delta=2))
        assert "INSIDE" in txt and "noise" in txt

    def test_delta_beyond_the_spread_survives(self):
        s = cvr.summarise([
            _draw("s1", {"a": True, "b": False}),
            _draw("s2", {"a": True, "b": False}),
        ])
        txt = " ".join(cvr.verdict(s, claim_delta=3))
        assert "EXCEEDS" in txt

    def test_two_draws_are_flagged_provisional(self):
        s = cvr.summarise([_draw("s1", {"a": True}), _draw("s2", {"a": True})])
        assert any("provisional" in ln.lower() for ln in cvr.verdict(s, None))


class TestLoadDraw:
    def test_reads_the_real_results_shape(self, tmp_path):
        p = tmp_path / "echoswarm_cartridge_A2_s1.json"
        p.write_text(json.dumps({"per_question": [
            {"id": "echoswarm_01", "grounded": True},
            {"id": "echoswarm_02", "grounded": False},
        ]}))
        d = cvr.load_draw(p)
        assert d["score"] == 1 and d["n"] == 2
        assert d["verdicts"] == {"echoswarm_01": True, "echoswarm_02": False}

    def test_label_strips_the_shared_prefix(self, tmp_path):
        p = tmp_path / "echoswarm_cartridge_A2_s2.json"
        p.write_text(json.dumps({"per_question": []}))
        assert cvr.load_draw(p)["label"] == "s2"
