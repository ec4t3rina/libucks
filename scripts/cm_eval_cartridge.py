"""CM-A.2 cartridge eval — latent-alone grounding across all echoswarm fixtures.

For each fixture: route to its bucket, load that bucket's distilled cartridge,
generate an answer from the LATENT ALONE (no verbatim), score keyword grounding.
Reports the CM-A.2 gate number: cartridge latent-alone grounding /25.

Reference baselines (Phase 4-C fairness eval, re-scored under CM-B.0a):
  hybrid 11/25 · text_clean 4/25 · no_context 3/25 · cache_aug_no_verbatim 3/25
Gate: cartridge latent-alone >= 8/25 (= no_context 3 + 5).

WARNING — those baselines ran on a 0.5B receiver (echoswarm has no
.libucks/config.toml, so Config falls back to the 0.5B default) while this script
hardcodes 3B below. Cartridge-vs-baseline is therefore CROSS-MODEL and must not be
quoted as a comparison until a 3B no_context/text_clean baseline exists.

Run: uv run python scripts/cm_eval_cartridge.py
     uv run python scripts/cm_eval_cartridge.py --buckets bc6b90e2,40615ba9,fe7ded0d
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# MUST precede the torch import. Proven causally necessary for any distill/eval
# run on this machine (CM-A.2 r4 wedged without it; a clean probe passed with it).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch

from libucks.config import Config
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.eval_metrics import EVAL_MAX_NEW_TOKENS, grounding_score
from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RESULTS = Path(__file__).resolve().parent.parent / "tests/eval/results/cm"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
# Fallback only, for reporting. The real P is read from each cartridge file via
# KVPrefixCartridge.read_geometry, so this no longer has to match the distiller.
PREFIX_LEN = 128


def _log(m: str) -> None:
    print(f"[cm_eval] {m}", file=sys.stderr, flush=True)


def _grounded(answer: str, keywords: list[str]) -> bool:
    """Shared CM-B.0a metric — do NOT reintroduce a local substring copy here.

    This script used to carry its own plain-substring scorer, which is the metric
    that under-counted CM-A.2 by 3 fixtures (7/25 -> 10/25 once corrected). Any
    number produced with a local copy is not comparable to the logged results.
    """
    return grounding_score(answer, keywords)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--buckets",
        default="",
        help="comma-separated bucket id prefixes; evaluate only fixtures routing to these",
    )
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument(
        "--fixtures",
        default="",
        help="alternate fixture JSON (e.g. tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json). "
             "Changes the denominator — scores are NOT comparable across fixture sets.",
    )
    args = ap.parse_args()

    cfg = Config.load(REPO)
    libucks_dir = REPO / ".libucks"
    cart_dir = libucks_dir / "kv_cache"
    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    store = BucketStore(libucks_dir / "buckets")
    embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    # A DIFFERENT fixture file means a different denominator. Scores from one are
    # not comparable with scores from another, so the choice is explicit and the
    # filename is echoed into the results.
    fixtures_path = Path(args.fixtures) if args.fixtures else FIXTURES
    if not fixtures_path.is_absolute():
        fixtures_path = Path(__file__).resolve().parent.parent / fixtures_path
    fixtures = json.loads(fixtures_path.read_text())["fixtures"]
    if args.fixtures:
        _log(f"fixture set: {fixtures_path.name} ({len(fixtures)} fixtures) — "
             f"NOT comparable with the default 25-fixture numbers")

    centroids = registry.get_all_centroids()
    bids = list(centroids.keys())
    mat = np.stack([centroids[b] for b in bids])

    if args.buckets:
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        kept = []
        for fx in fixtures:
            bid = bids[int((mat @ embedder.embed(fx["question"])).argmax())]
            if any(bid.startswith(w) for w in want):
                kept.append(fx)
        if not kept:
            sys.exit(f"[cm_eval] --buckets matched no fixtures: {sorted(want)}")
        _log(f"restricted to {len(kept)}/{len(fixtures)} fixtures routing to {sorted(want)}")
        fixtures = kept

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(RECEIVER_ID, dtype=torch.bfloat16).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok)

    loaded: dict[str, KVPrefixCartridge] = {}

    def _get_cart(bid: str):
        if bid in loaded:
            return loaded[bid]
        p = cart_dir / f"{bid}.cartridge.safetensors"
        if not p.exists():
            loaded[bid] = None
            return None
        # Read P off the file rather than assuming PREFIX_LEN. The two scripts
        # used to restate the constant at each other under reciprocal "MUST
        # match" comments; this makes a P sweep evaluable without editing code,
        # and a stale constant can no longer silently mis-shape the cartridge.
        saved_p = KVPrefixCartridge.read_geometry(p)["prefix_len"]
        if saved_p != PREFIX_LEN:
            _log(f"{bid[:8]} — cartridge was trained at P={saved_p}, not the "
                 f"script default {PREFIX_LEN}; using the file's value")
        c = KVPrefixCartridge.for_model(model, prefix_len=saved_p, dtype=torch.float32)
        c.load(p)
        c.to(device)
        loaded[bid] = c
        return c

    grounded = 0
    multi_grounded = 0
    multi_total = 0
    missing = 0
    rows = []
    for i, fx in enumerate(fixtures, 1):
        bid = bids[int((mat @ embedder.embed(fx["question"])).argmax())]
        is_multi = bool(fx.get("needs_multi_bucket", False))
        multi_total += int(is_multi)
        cart = _get_cart(bid)
        if cart is None:
            missing += 1
            ans = ""
            g = False
        else:
            ans = trainer.generate_answer(cart, fx["question"], max_new_tokens=EVAL_MAX_NEW_TOKENS, verbatim="")
            g = _grounded(ans, fx["answer_keywords"])
        grounded += int(g)
        multi_grounded += int(g and is_multi)
        rows.append({"id": fx["id"], "bucket": bid[:8], "grounded": g,
                     "multi": is_multi, "kw": fx["answer_keywords"], "answer": ans})
        _log(f"[{i:2d}/{len(fixtures)}] {fx['id']} b={bid[:8]} grounded={g} :: {ans[:90]!r}")

    n = len(fixtures)
    _log("================ CARTRIDGE EVAL ================")
    _log(f"cartridge latent-alone grounding: {grounded}/{n}  (multi {multi_grounded}/{multi_total})")
    _log(f"missing cartridges (ungrounded): {missing}")
    if args.buckets:
        # Subset run: the /25 gate does not apply. CM-A.2's score on the three
        # CM-B.0b spot-check buckets was 2/13 raw, 5/13 under the CM-B.0a metric.
        _log("SUBSET RUN — the >=8/25 gate does not apply.")
        _log("CM-A.2 baseline on these fixtures: 2/13 old metric, 5/13 CM-B.0a metric")
        _log(f"CM-B.0b bar (>= 5/13 is neutral, > 5/13 is improvement): {grounded}/{n}")
    else:
        _log("reference (0.5B receiver — CROSS-MODEL, not a valid comparison): "
             "hybrid 11/25 · text_clean 4/25 · no_context 3/25 · cache_aug_no_verbatim 3/25")
        _log(f"GATE (cartridge latent-alone >= 8/25): {'PASS' if grounded >= 8 else 'FAIL'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"echoswarm_cartridge_A2{('_' + args.tag) if args.tag else ''}.json"
    (RESULTS / name).write_text(json.dumps(
        {"grounding": grounded, "n": n, "multi_grounding": multi_grounded,
         "multi_total": multi_total, "missing": missing,
         "buckets_filter": args.buckets or None, "metric": "cm-b.0a",
         "per_question": rows}, indent=2))
    _log(f"wrote {RESULTS / name}")


if __name__ == "__main__":
    main()
