"""CM-A.1 single-bucket proof.

Distill ONE echoswarm bucket into a KV-prefix cartridge via context
distillation, then test whether the frozen 3B can answer that bucket's real
fixtures from the LATENT ALONE (no verbatim). This is the fast falsification of
the Cartridge Memory hypothesis before scaling to all buckets (CM-A.2).

Prints, for the busiest bucket:
  - init (warm-start KV, no distill) latent-alone grounding
  - distilled latent-alone grounding   <-- the number that matters
  - KL trajectory

Run: uv run python scripts/cm_proof_single_bucket.py
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
from libucks.thinking.training.data_generator import _collect_source_text
from libucks.cache_augmentation.kv_extract import extract_bucket_kv
from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer
from libucks.thinking.training.self_study import generate_self_study_queries

import os

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
QGEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # instruct model for fact-probing query gen
# Env-tunable levers (defaults = original CM-A.1 run for reproducibility).
PREFIX_LEN = int(os.environ.get("CM_PREFIX_LEN", "64"))
N_QUERIES = int(os.environ.get("CM_NQUERIES", "128"))
EPOCHS = int(os.environ.get("CM_EPOCHS", "2"))
MODEL_QUERIES = os.environ.get("CM_MODEL_QUERIES", "0") == "1"
LR = float(os.environ.get("CM_LR", "1e-2"))


def _log(m: str) -> None:
    print(f"[cm_proof] {m}", file=sys.stderr, flush=True)


def _grounded(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    a = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in a) >= len(keywords) / 2.0


def main() -> None:
    cfg = Config.load(REPO)
    libucks_dir = REPO / ".libucks"
    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    store = BucketStore(libucks_dir / "buckets")
    embedder = EmbeddingService.get_instance(cfg.model.embedding_model)

    fixtures = json.loads(FIXTURES.read_text())["fixtures"]

    # --- route each fixture to its nearest-centroid bucket ---
    centroids = registry.get_all_centroids()
    bids = list(centroids.keys())
    mat = np.stack([centroids[b] for b in bids])  # (N, D), normalized
    by_bucket: dict[str, list] = {}
    for fx in fixtures:
        q_emb = embedder.embed(fx["question"])
        bid = bids[int((mat @ q_emb).argmax())]
        by_bucket.setdefault(bid, []).append(fx)

    chosen = max(by_bucket, key=lambda b: len(by_bucket[b]))
    chosen_fx = by_bucket[chosen]
    _log(f"routed fixture distribution: {sorted((len(v)) for v in by_bucket.values())}")
    _log(f"chosen bucket {chosen[:12]} with {len(chosen_fx)} fixtures")

    fm, _prose = store.read(chosen)
    verbatim = _collect_source_text(fm, max_chars=4096) or ""
    _log(f"verbatim chars: {len(verbatim)}")
    if len(verbatim) < 50:
        _log("verbatim too short — aborting proof")
        return

    # --- load frozen 3B ---
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(RECEIVER_ID, dtype=torch.bfloat16).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)

    trainer = CartridgeTrainer(
        model, tok, temperature=2.0, alpha_ce=0.3,
        max_answer_tokens=48, max_verbatim_chars=4096,
    )

    # --- warm-start cartridge from the bucket's real extracted KV ---
    cart = KVPrefixCartridge.for_model(model, prefix_len=PREFIX_LEN, dtype=torch.float32)
    flat = extract_bucket_kv(model, tok, verbatim, max_tokens=1024)
    cart.init_from_extracted_kv(flat)
    cart.to(device)

    def _eval(tag: str) -> int:
        n = 0
        for fx in chosen_fx:
            ans = trainer.generate_answer(cart, fx["question"], max_new_tokens=64, verbatim="")
            g = _grounded(ans, fx["answer_keywords"])
            n += int(g)
            _log(f"  [{tag}] grounded={g} kw={fx['answer_keywords']} :: {ans[:110]!r}")
        return n

    _log("=== BEFORE distillation (warm-start KV, latent-alone) ===")
    init_grounded = _eval("init")

    qgen_model = qgen_tok = None
    if MODEL_QUERIES:
        _log(f"loading query-gen model {QGEN_ID} ...")
        qgen_model = AutoModelForCausalLM.from_pretrained(QGEN_ID, dtype=torch.float32).eval().to(device)
        qgen_tok = AutoTokenizer.from_pretrained(QGEN_ID)
    queries = generate_self_study_queries(verbatim, N_QUERIES, model=qgen_model, tokenizer=qgen_tok)
    if qgen_model is not None:
        del qgen_model  # free before distillation
        if device.type == "mps":
            torch.mps.empty_cache()
    _log(f"distilling on {len(queries)} queries (model_queries={MODEL_QUERIES}), P={PREFIX_LEN}, {EPOCHS} epochs, lr={LR} ...")
    _log(f"sample queries: {queries[:3]}")
    res = trainer.distill_bucket(cart, verbatim, queries, epochs=EPOCHS, lr=LR)

    _log("=== AFTER distillation (latent-alone) ===")
    distilled_grounded = _eval("distilled")

    _log("================ CM-A.1 PROOF RESULT ================")
    _log(f"bucket={chosen[:12]} fixtures={len(chosen_fx)}")
    _log(f"KL trajectory: init={res['init_mean_kl']:.3f} -> final={res['final_mean_kl']:.3f}  (epochs {res['epoch_mean_kl']})")
    _log(f"latent-alone grounding: init(warm-start)={init_grounded}/{len(chosen_fx)}  distilled={distilled_grounded}/{len(chosen_fx)}")
    gate = distilled_grounded >= 3 and distilled_grounded >= init_grounded
    _log(f"GATE (distilled>=3 and >=init): {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
