"""CM-B bug sweep: the merge limit must never exceed the mitosis threshold.

HealthMonitor._check runs, every 5 minutes and in this order:

    for each bucket:  size-split if tokens >= mitosis_threshold
                      coherence-split if coherence < 0.55
    then:             MergingService.run_merge_pass()

MergingService merges any cosine-similar pair whose combined token count is
under its merge limit. If that limit is allowed to sit ABOVE mitosis_threshold,
a merge can produce a bucket that the very next health pass immediately splits.

The anti-cycle guard does not save us. MergingService records the *pre-merge*
bucket IDs in _meta["merge_history"] and refuses to re-merge them for an hour,
but MitosisService mints brand-new sha1 child IDs (`mitosis._child_id`). The
children are therefore absent from the cooldown set, and their centroids are
still ~the originals, so they are immediately merge-eligible again. Split ->
merge -> split, forever, each cycle burning LLM calls to regenerate prose and
invalidating every affected cartridge.

At the shipped defaults (mitosis 20_000, merge 15_000) this cannot fire, and a
scan of the real echoswarm/libugry registries confirms it: the single
merge-eligible pair yields a merged coherence of 0.675, well above the 0.55
split threshold. But mitosis_threshold is a documented, user-tunable knob
(ARCHITECTURE.md:472, README.md:373) with no stated lower bound, and
`test_config.py` itself constructs one at 10_000 — below the old hardcoded
15_000 merge limit. Nothing anywhere enforced the ordering.

Fix under test: the merge limit is DERIVED from mitosis_threshold rather than
hardcoded, so the invariant holds structurally for any configuration.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from libucks.merging_service import (
    MERGE_TOKEN_LIMIT,
    MERGE_TOKEN_RATIO,
    MergingService,
)


def _svc(mitosis_threshold: int | None = None) -> MergingService:
    registry = MagicMock()
    registry._meta = {}
    kwargs = {}
    if mitosis_threshold is not None:
        kwargs["mitosis_threshold"] = mitosis_threshold
    return MergingService(
        registry=registry,
        store=MagicMock(),
        agent=MagicMock(),
        embedder=MagicMock(),
        strategy=MagicMock(),
        **kwargs,
    )


class TestMergeLimitDerivation:
    def test_default_reproduces_the_historical_constant(self):
        """0.75 * 20_000 == 15_000 — the shipped behaviour is unchanged."""
        assert _svc()._merge_token_limit == MERGE_TOKEN_LIMIT == 15_000

    @pytest.mark.parametrize("threshold", [1_000, 8_000, 10_000, 20_000, 100_000])
    def test_merge_limit_is_always_below_mitosis_threshold(self, threshold: int):
        """The invariant: a merge can never produce an instantly-splittable bucket."""
        assert _svc(threshold)._merge_token_limit < threshold

    def test_low_threshold_shrinks_the_merge_limit(self):
        """The exact configuration test_config.py constructs (10_000)."""
        assert _svc(10_000)._merge_token_limit == 7_500

    def test_ratio_leaves_headroom(self):
        assert 0.0 < MERGE_TOKEN_RATIO < 1.0


class TestShouldMergeRespectsDerivedLimit:
    def _centroids(self) -> dict:
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        return {"a": v, "b": v.copy()}

    def _run(self, mitosis_threshold: int, tokens_each: int) -> bool:
        svc = _svc(mitosis_threshold)
        svc._registry.get_token_count.return_value = tokens_each
        return svc._should_merge("a", "b", self._centroids(), set())

    def test_pair_that_would_oscillate_is_refused(self):
        """Combined 12_854 tokens under a 10_000 mitosis threshold.

        These are the real token counts of echoswarm buckets 0817f15e+f1a6c60e,
        the only merge-eligible pair in that repo. Under the old hardcoded
        15_000 limit this merge was allowed, and the merged bucket immediately
        exceeded a 10_000 mitosis threshold — the cycle.
        """
        assert self._run(10_000, 6_427) is False

    def test_same_pair_still_merges_at_the_default_threshold(self):
        """No regression: at mitosis 20_000 the pair is still eligible."""
        assert self._run(20_000, 6_427) is True

    def test_boundary_exactly_at_the_limit_is_refused(self):
        assert self._run(20_000, 7_500) is False
