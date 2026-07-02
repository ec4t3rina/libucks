"""CM-A.2 cartridge eval — latent-alone grounding across all echoswarm fixtures.

For each fixture: route to its bucket, load that bucket's distilled cartridge,
generate an answer from the LATENT ALONE (no verbatim), score keyword grounding.
Reports the CM-A.2 gate number: cartridge latent-alone grounding /25.

Reference baselines (from docs/archive/phase-4c fairness eval):
  hybrid 11/25 · text_clean 4/25 · no_context 3/25 · cache_aug_no_verbatim 2/25
Gate: cartridge latent-alone >= 8/25 (= no_context 3 + 5).

Run: uv run python scripts/cm_eval_cartridge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from libucks.config import Config
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RESULTS = Path(__file__).resolve().parent.parent / "tests/eval/results/cm"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
PREFIX_LEN = 128          # MUST match cm_distill_buckets.py


def _log(m: str) -> None:
    print(f"[cm_eval] {m}", file=sys.stderr, flush=True)


def _grounded(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    a = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in a) >= len(keywords) / 2.0


def main() -> None:
    cfg = Config.load(REPO)
    libucks_dir = REPO / ".libucks"
    cart_dir = libucks_dir / "kv_cache"
    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    store = BucketStore(libucks_dir / "buckets")
    embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    fixtures = json.loads(FIXTURES.read_text())["fixtures"]

    centroids = registry.get_all_centroids()
    bids = list(centroids.keys())
    mat = np.stack([centroids[b] for b in bids])

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
        c = KVPrefixCartridge.for_model(model, prefix_len=PREFIX_LEN, dtype=torch.float32)
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
            ans = trainer.generate_answer(cart, fx["question"], max_new_tokens=64, verbatim="")
            g = _grounded(ans, fx["answer_keywords"])
        grounded += int(g)
        multi_grounded += int(g and is_multi)
        rows.append({"id": fx["id"], "bucket": bid[:8], "grounded": g,
                     "multi": is_multi, "kw": fx["answer_keywords"], "answer": ans})
        _log(f"[{i:2d}/25] {fx['id']} b={bid[:8]} grounded={g} :: {ans[:90]!r}")

    n = len(fixtures)
    _log("================ CM-A.2 CARTRIDGE EVAL ================")
    _log(f"cartridge latent-alone grounding: {grounded}/{n}  (multi {multi_grounded}/{multi_total})")
    _log(f"missing cartridges (ungrounded): {missing}")
    _log("reference: hybrid 11/25 · text_clean 4/25 · no_context 3/25 · cache_aug_no_verbatim 2/25")
    _log(f"GATE (cartridge latent-alone >= 8/25): {'PASS' if grounded >= 8 else 'FAIL'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "echoswarm_cartridge_A2.json").write_text(json.dumps(
        {"grounding": grounded, "n": n, "multi_grounding": multi_grounded,
         "multi_total": multi_total, "missing": missing, "per_question": rows}, indent=2))
    _log(f"wrote {RESULTS / 'echoswarm_cartridge_A2.json'}")


if __name__ == "__main__":
    main()
