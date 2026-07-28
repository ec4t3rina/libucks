"""A fixture that cannot be answered from the bucket is measuring nothing.

Five of the original 25 echoswarm fixtures are structurally unreachable — their
answer keywords live in verbatim the 4096-char cap discards, or in a bucket the
question does not route to. Those five cap the eval at 20/25 and quietly inflate
the apparent failure rate of every path.

The extension set was written against the actual distilled text, so these tests
hold it to the standard the original set was never checked against. They are
also a guard for future fixtures: adding an unanswerable one now fails here
rather than showing up as a mystery zero months later.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json"
MAIN = ROOT / "tests/eval/fixtures/echoswarm_qa.json"
REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
TARGET = "bc6b90e2"

pytestmark = pytest.mark.skipif(
    not (REPO / ".libucks" / "registry.json").exists(),
    reason="echoswarm test repo not present",
)


def _ext() -> list[dict]:
    return json.loads(EXT.read_text())["fixtures"]


def _kept_verbatim() -> str:
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.thinking.training.data_generator import _collect_source_text

    d = REPO / ".libucks"
    reg = BucketRegistry(d / "registry.json")
    reg.load()
    store = BucketStore(d / "buckets")
    bid = next(b for b in reg.get_all_centroids() if b.startswith(TARGET))
    fm, _ = store.read(bid)
    return (_collect_source_text(fm, max_chars=4096) or "").lower()


class TestWellFormed:
    def test_file_parses_and_is_non_trivial(self):
        fx = _ext()
        assert len(fx) >= 12, "too few to move the noise floor meaningfully"

    def test_ids_are_unique_within_the_set(self):
        ids = [f["id"] for f in _ext()]
        assert len(ids) == len(set(ids))

    def test_ids_do_not_collide_with_the_original_set(self):
        """Colliding ids would silently overwrite results when both are scored."""
        main_ids = {f["id"] for f in json.loads(MAIN.read_text())["fixtures"]}
        assert not (main_ids & {f["id"] for f in _ext()})

    @pytest.mark.parametrize("field", [
        "id", "question", "expected_bucket_keywords",
        "answer_keywords", "ground_truth_answer", "needs_multi_bucket",
    ])
    def test_every_fixture_has_the_required_field(self, field):
        missing = [f.get("id", "?") for f in _ext() if field not in f]
        assert not missing, f"{field} missing from {missing}"

    def test_questions_are_distinct(self):
        qs = [f["question"].strip().lower() for f in _ext()]
        assert len(qs) == len(set(qs))

    def test_keyword_lists_are_non_empty(self):
        """grounding_score returns False for an empty list — a silent zero."""
        bad = [f["id"] for f in _ext() if not f["answer_keywords"]]
        assert not bad, f"empty answer_keywords in {bad}"


class TestAnswerable:
    """The check the original 25 never got."""

    def test_every_fixture_clears_the_50_percent_bar_from_kept_verbatim(self):
        kept = _kept_verbatim()
        unreachable = []
        for f in _ext():
            kws = f["answer_keywords"]
            hits = sum(1 for k in kws if k.lower() in kept)
            if hits < len(kws) / 2.0:
                unreachable.append(f"{f['id']}: {hits}/{len(kws)} in kept verbatim")
        assert not unreachable, (
            "fixtures that cannot reach the grounding threshold even with a "
            "perfect answer:\n  " + "\n  ".join(unreachable)
        )

    def test_reports_how_much_headroom_each_fixture_has(self):
        """Not a pass/fail on its own — surfaces fixtures sitting exactly on the
        boundary, where one keyword drifting out of the verbatim silently makes
        them impossible."""
        kept = _kept_verbatim()
        marginal = []
        for f in _ext():
            kws = f["answer_keywords"]
            hits = sum(1 for k in kws if k.lower() in kept)
            if hits == len(kws) / 2.0:
                marginal.append(f["id"])
        assert len(marginal) <= 2, (
            f"too many fixtures sit exactly on the 50% boundary: {marginal}"
        )


@pytest.mark.slow
class TestRouting:
    """A fixture that routes elsewhere measures the wrong bucket. Three of the
    original 25 fail this way."""

    def test_every_fixture_routes_to_the_target_bucket(self):
        import numpy as np

        from libucks.config import Config
        from libucks.embeddings.embedding_service import EmbeddingService
        from libucks.storage.bucket_registry import BucketRegistry

        cfg = Config.load(REPO)
        reg = BucketRegistry(REPO / ".libucks" / "registry.json")
        reg.load()
        emb = EmbeddingService.get_instance(cfg.model.embedding_model)
        cent = reg.get_all_centroids()
        bids = list(cent)
        mat = np.stack([cent[b] for b in bids])

        misrouted = []
        for f in _ext():
            got = bids[int((mat @ emb.embed(f["question"])).argmax())]
            if not got.startswith(TARGET):
                misrouted.append(f"{f['id']} -> {got[:8]}")
        assert not misrouted, (
            "fixtures routing away from " + TARGET + ":\n  " + "\n  ".join(misrouted)
        )
