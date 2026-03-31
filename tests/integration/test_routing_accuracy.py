"""Phase 2 Testing Gate — test_routing_accuracy.py

Needle-in-a-Haystack: 20 software domains × 5 probe + 5 adversarial queries.

Accuracy gates:
  top-1 accuracy on probe queries   ≥ 90 %   (100 cases)
  top-3 accuracy on adversarial     ≥ 98 %   (100 cases)

Marked @pytest.mark.slow because the EmbeddingService loads the
sentence-transformers model once for the session.  Exclude from fast
CI runs with:  pytest -m "not slow"

Test strategy
-------------
1. Load needle_cases.json (text only — no pre-computed numbers).
2. Session-scoped fixture uses EmbeddingService to:
   a. Compute per-domain centroids: mean of embed_batch(centroid_texts), L2-normalised.
   b. Embed all 200 probe queries and 200 adversarial queries.
3. Register all 20 centroids in a BucketRegistry.
4. Call CentralAgent.route() with the pre-computed query embeddings.
5. Assert aggregate accuracy meets the gate thresholds.
   Individual case results are printed for diagnostic visibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from libucks.central_agent import CentralAgent
from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.storage.bucket_registry import BucketRegistry

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "routing" / "needle_cases.json"
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class DomainCase(NamedTuple):
    domain_id: str
    domain_label: str
    centroid: np.ndarray        # L2-normalised, shape (384,)
    probe_embeddings: list      # list of np.ndarray (384,)
    adversarial_embeddings: list


class RoutingFixture(NamedTuple):
    domains: list[DomainCase]
    registry: BucketRegistry
    agent: CentralAgent


# ---------------------------------------------------------------------------
# Session-scoped fixture: load model once, embed everything
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def routing_fixture(tmp_path_factory) -> RoutingFixture:
    raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    service = EmbeddingService.get_instance()

    tmp = tmp_path_factory.mktemp("routing_accuracy")
    registry = BucketRegistry(tmp / "registry.json")
    config = Config()

    domains: list[DomainCase] = []

    for d in raw["domains"]:
        # Centroid: mean of L2-normalised centroid_text embeddings, re-normalised.
        centroid_matrix = service.embed_batch(d["centroid_texts"])  # (N, 384)
        centroid = centroid_matrix.mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm

        probe_embs = [service.embed(q) for q in d["probe_queries"]]
        adv_embs = [service.embed(q) for q in d["adversarial_queries"]]

        domains.append(DomainCase(
            domain_id=d["domain_id"],
            domain_label=d["domain_label"],
            centroid=centroid,
            probe_embeddings=probe_embs,
            adversarial_embeddings=adv_embs,
        ))

    # Register all centroids synchronously (asyncio.run inside sync fixture).
    import asyncio
    async def _register_all():
        for dom in domains:
            await registry.register(dom.domain_id, dom.centroid, 1000)
    asyncio.run(_register_all())

    agent = CentralAgent(registry=registry, config=config)
    return RoutingFixture(domains=domains, registry=registry, agent=agent)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _route_top_k(agent: CentralAgent, embedding: np.ndarray, k: int) -> list[str]:
    return agent.route(embedding, top_k=k)


# ---------------------------------------------------------------------------
# Probe queries — top-1 accuracy gate ≥ 90 %
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_probe_top1_accuracy(routing_fixture: RoutingFixture):
    """For each domain × probe query, the correct bucket must be rank-1.

    Gate: at least 90 % of 100 cases correct (allows 10 failures).
    With well-separated domains this should reach 100 %.
    """
    failures: list[str] = []
    correct = 0
    total = 0

    for dom in routing_fixture.domains:
        for i, emb in enumerate(dom.probe_embeddings):
            result = _route_top_k(routing_fixture.agent, emb, k=1)
            total += 1
            if result and result[0] == dom.domain_id:
                correct += 1
            else:
                rank1 = result[0] if result else "∅"
                failures.append(
                    f"  FAIL probe  domain={dom.domain_id!r} query_idx={i} "
                    f"got_rank1={rank1!r}"
                )

    accuracy = correct / total
    if failures:
        print(f"\nProbe top-1 failures ({len(failures)}/{total}):")
        for f in failures:
            print(f)

    assert accuracy >= 0.90, (
        f"Probe top-1 accuracy {accuracy:.1%} < 90 % gate "
        f"({correct}/{total} correct)"
    )


# ---------------------------------------------------------------------------
# Adversarial queries — top-3 accuracy gate ≥ 98 %
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_adversarial_top3_accuracy(routing_fixture: RoutingFixture):
    """For each domain × adversarial query, the correct bucket must appear in top-3.

    Gate: at least 98 % of 100 cases correct (allows 2 failures).
    Adversarial queries use adjacent-domain vocabulary to stress the router.
    """
    failures: list[str] = []
    correct = 0
    total = 0

    for dom in routing_fixture.domains:
        for i, emb in enumerate(dom.adversarial_embeddings):
            result = _route_top_k(routing_fixture.agent, emb, k=3)
            total += 1
            if dom.domain_id in result:
                correct += 1
            else:
                failures.append(
                    f"  FAIL adv    domain={dom.domain_id!r} query_idx={i} "
                    f"top3={result!r}"
                )

    accuracy = correct / total
    if failures:
        print(f"\nAdversarial top-3 failures ({len(failures)}/{total}):")
        for f in failures:
            print(f)

    assert accuracy >= 0.98, (
        f"Adversarial top-3 accuracy {accuracy:.1%} < 98 % gate "
        f"({correct}/{total} correct)"
    )


# ---------------------------------------------------------------------------
# Summary printout (always runs, non-asserting)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_accuracy_summary(routing_fixture: RoutingFixture, capsys):
    """Print a per-domain accuracy table for diagnostic visibility."""
    rows = []
    for dom in routing_fixture.domains:
        probe_hits = sum(
            1 for emb in dom.probe_embeddings
            if (r := _route_top_k(routing_fixture.agent, emb, k=1)) and r[0] == dom.domain_id
        )
        adv_hits = sum(
            1 for emb in dom.adversarial_embeddings
            if dom.domain_id in _route_top_k(routing_fixture.agent, emb, k=3)
        )
        rows.append((dom.domain_label, probe_hits, adv_hits))

    with capsys.disabled():
        print("\n\n── Routing accuracy by domain ──────────────────────────────────")
        print(f"{'Domain':<48}  probe top-1  adv top-3")
        print("─" * 70)
        for label, p, a in rows:
            print(f"{label:<48}  {p}/5          {a}/5")
        probe_total = sum(r[1] for r in rows)
        adv_total = sum(r[2] for r in rows)
        print("─" * 70)
        print(f"{'TOTAL':<48}  {probe_total}/100       {adv_total}/100")
        print(f"{'ACCURACY':<48}  {probe_total/100:.0%}         {adv_total/100:.0%}")
        print("────────────────────────────────────────────────────────────────\n")
