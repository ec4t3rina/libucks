"""The stratified set must be diagnostic, not confirmatory.

The bc6b90e2 extension set produced an apparently clean compression curve for
prefix truncation. It was an artifact: the scores tracked the number of fixtures
whose keywords fall inside the first P tokens almost exactly (11 vs 11 at P=256,
9 vs 8 at P=128), because those questions clustered in the head of the file. A
prefix-truncation arm wins such a set by construction — it cannot know anything
past P, so a set that only asks about the head cannot distinguish truncation from
compression.

These tests hold the 95c8e099 set to the property that fixes that: target facts
spread across the whole token range, so the positional-availability ceiling at
small P is near zero and any arm exceeding it is doing real work.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STRAT = ROOT / "tests/eval/fixtures/echoswarm_qa_95c8e099_strat.json"
REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
TARGET = "95c8e099"
GENERIC = {"int", "bool", "str", "float", "none", "true", "false", "list", "dict"}

pytestmark = pytest.mark.skipif(
    not (REPO / ".libucks" / "registry.json").exists(),
    reason="echoswarm test repo not present",
)


def _fx() -> list[dict]:
    return json.loads(STRAT.read_text())["fixtures"]


def _verbatim() -> str:
    """FULL bucket text — raw cache extraction is not bound by the 4096 cap that
    the distillation teacher's context window imposed."""
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.thinking.training.data_generator import _collect_source_text

    d = REPO / ".libucks"
    reg = BucketRegistry(d / "registry.json")
    reg.load()
    store = BucketStore(d / "buckets")
    bid = next(b for b in reg.get_all_centroids() if b.startswith(TARGET))
    fm, _ = store.read(bid)
    return _collect_source_text(fm, max_chars=10 ** 9) or ""


def _first_hit_tokens(f: dict, vl: str, cpt: float) -> list[int]:
    """Approx token position of each keyword's first occurrence, ascending."""
    pos = [vl.find(k.lower()) for k in f["answer_keywords"]]
    return sorted(int(p / cpt) for p in pos if p >= 0)


def _min_P(f: dict, vl: str, cpt: float) -> int | None:
    """Smallest prefix length from which >=50% of the keywords are available.

    grounding_score needs `hits >= len/2`, i.e. CEIL(len/2) hits — so the index is
    ceil(len/2)-1, not int(len/2)-1. The int() form is off by one for odd keyword
    counts and understates min_P, making fixtures look more head-available than
    they are. Corrected after it hid two late fixtures from the gap check.

    Approximate in both directions: char->token conversion is uniform, and
    keyword_hit normalises forms (80% <-> 0.8, two <-> 2) that a literal substring
    scan misses. Use it to compare fixture sets, not as a hard bound.
    """
    import math

    toks = _first_hit_tokens(f, vl, cpt)
    idx = math.ceil(len(f["answer_keywords"]) / 2) - 1
    return toks[idx] + 1 if len(toks) > idx >= 0 else None


class TestWellFormed:
    def test_enough_fixtures_to_resolve(self):
        """n=8 was proven to have no resolving power: it scored 2/8 at P=32 and
        2/8 at P=384, which cannot both be signal."""
        assert len(_fx()) >= 16

    def test_ids_unique_and_namespaced(self):
        ids = [f["id"] for f in _fx()]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("echoswarm_y") for i in ids), (
            "a distinct id prefix keeps results from colliding with the x/plain sets"
        )

    @pytest.mark.parametrize("field", [
        "id", "question", "expected_bucket_keywords",
        "answer_keywords", "ground_truth_answer", "needs_multi_bucket",
    ])
    def test_required_fields(self, field):
        assert not [f.get("id", "?") for f in _fx() if field not in f]

    def test_no_empty_keyword_lists(self):
        assert not [f["id"] for f in _fx() if not f["answer_keywords"]]


class TestNotConfirmatory:
    def test_echo_alone_cannot_score(self):
        offenders = []
        for f in _fx():
            q = f["question"].lower()
            echoed = [k for k in f["answer_keywords"] if k.lower() in q]
            if len(echoed) >= len(f["answer_keywords"]) / 2.0:
                offenders.append(f"{f['id']}: {echoed}")
        assert not offenders, "\n  " + "\n  ".join(offenders)

    def test_generic_type_names_not_load_bearing(self):
        offenders = []
        for f in _fx():
            kws = f["answer_keywords"]
            g = [k for k in kws if k.lower() in GENERIC]
            if len(g) >= len(kws) / 2.0:
                offenders.append(f"{f['id']}: {g}")
        assert not offenders, "\n  " + "\n  ".join(offenders)


class TestAnswerable:
    def test_every_fixture_reaches_the_threshold_from_the_full_verbatim(self):
        v = _verbatim().lower()
        bad = []
        for f in _fx():
            kws = f["answer_keywords"]
            hits = sum(1 for k in kws if k.lower() in v)
            if hits < len(kws) / 2.0:
                bad.append(f"{f['id']}: {hits}/{len(kws)}")
        assert not bad, "unanswerable even from the whole bucket:\n  " + "\n  ".join(bad)


class TestPositionalStratification:
    """The property that makes this set able to distinguish truncation from
    compression."""

    def _spread(self):
        v = _verbatim()
        cpt = len(v) / max(1, len(v) // 4)   # ~chars per token
        vl = v.lower()
        return [(f["id"], _min_P(f, vl, cpt)) for f in _fx()], len(v) // 4

    def test_every_fixture_has_a_reachable_position(self):
        spread, _ = self._spread()
        assert not [i for i, p in spread if p is None]

    def test_facts_are_not_clustered_in_the_head(self):
        """The bc6b90e2 failure: 9 of 16 answerable from the first 128 tokens."""
        spread, total = self._spread()
        head = total // 8          # first 12.5% of the file
        n_head = sum(1 for _, p in spread if p <= head)
        assert n_head <= len(spread) * 0.35, (
            f"{n_head}/{len(spread)} fixtures answerable from the first {head} "
            f"tokens — a prefix-truncation arm would win by construction"
        )

    def test_a_substantial_share_needs_the_tail(self):
        spread, total = self._spread()
        half = total // 2
        n_tail = sum(1 for _, p in spread if p > half)
        assert n_tail >= len(spread) * 0.25, (
            f"only {n_tail}/{len(spread)} fixtures require more than the first "
            f"half of the file; truncation would not be penalised"
        )

    def test_positions_span_the_document(self):
        spread, total = self._spread()
        ps = sorted(p for _, p in spread if p is not None)
        assert ps[0] < total * 0.15, "nothing in the opening"
        assert ps[-1] > total * 0.7, "nothing in the closing"

    def test_no_large_positional_gap(self):
        """A gap leaves a P range where the ceiling is flat, which is what made
        the earlier curve look like saturation."""
        spread, total = self._spread()
        ps = sorted(p for _, p in spread if p is not None)
        gaps = [(b - a, a, b) for a, b in zip(ps, ps[1:])]
        worst = max(gaps) if gaps else (0, 0, 0)
        assert worst[0] <= total * 0.25, (
            f"gap of {worst[0]} tokens between {worst[1]} and {worst[2]} "
            f"(document ~{total} tokens)"
        )


class TestExplicitBucket:
    """Routing is deliberately NOT part of this experiment.

    14 of the first 20 questions routed to other buckets, because simulation.py is
    the orchestrator and overlaps semantically with most of the repo. That is a
    fact about retrieval, not about whether the bucket's cache carries its content
    — and routing is already measured separately (Phase 1: 1/15 -> 14/15). Letting
    the router decide here would make a cache result depend on retrieval quality
    and silently drop most of the fixtures.
    """

    def test_every_fixture_declares_its_bucket(self):
        missing = [f["id"] for f in _fx() if not f.get("bucket")]
        assert not missing, f"no explicit bucket on {missing}"

    def test_all_declare_the_same_target(self):
        assert {f["bucket"] for f in _fx()} == {TARGET}

    def test_the_declared_bucket_exists(self):
        from libucks.storage.bucket_registry import BucketRegistry

        reg = BucketRegistry(REPO / ".libucks" / "registry.json")
        reg.load()
        assert any(b.startswith(TARGET) for b in reg.get_all_centroids())

    def test_the_sweep_honours_the_field(self):
        src = (ROOT / "scripts/cm_kv_sweep.py").read_text()
        assert 'f.get("bucket")' in src, (
            "cm_kv_sweep must use the declared bucket, or these fixtures get "
            "routed away and silently dropped"
        )
