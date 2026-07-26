"""Batch distiller — distill a KV-prefix cartridge for every echoswarm bucket
that any fixture routes to, saving each to disk.

Resumable: skips buckets whose cartridge already exists, so a sleep/crash just
requires re-running (it continues where it left off). A per-epoch checkpoint
(<bucket_id>.cartridge.ckpt.safetensors) is written during training and removed
on success, so an MPS wedge leaves at most one lost epoch behind — the final
.cartridge.safetensors only appears once all epochs completed.

CM-B.0b FIX: this script used to call `generate_self_study_queries(..., model=None)`,
which skips `_model_queries` entirely and falls through to generic templates
(self_study.py:178-185). CM-A.1 had already established that templated queries were
the bottleneck (2/8 templated -> 4/8 fact-probing), so CM-A.2 unknowingly re-ran the
losing configuration while its log recorded "fact-probing self-study". A query-gen
model is now loaded and used, mirroring cm_proof_single_bucket.py:121-128.

Query generation happens BEFORE the 3B receiver is loaded, and the generator is freed
first, so the two models are never resident at the same time — on 16 GB that matters.

Cartridge saved to: <repo>/.libucks/kv_cache/<bucket_id>.cartridge.safetensors

Run (overnight, kept awake, detached — a non-detached run dies with its parent shell):
  nohup caffeinate -dimsu uv run python scripts/cm_distill_buckets.py \
      > cm_redistill.log 2>&1 &

Only the CM-B.0b spot-check buckets:
  ... scripts/cm_distill_buckets.py --buckets bc6b90e2,40615ba9,fe7ded0d --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# MUST precede the torch import below. Without it, distill runs on 16 GB Apple
# Silicon wedge in an uninterruptible MPS allocator wait — proven causally
# necessary in CM-A.2 (r4 wedged without it; a clean probe passed with it in 62 s).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

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

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
QGEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # fact-probing query generator (CM-B.0b)
PREFIX_LEN = 128          # MUST match cm_eval_cartridge.py
N_QUERIES = int(os.environ.get("CM_NQUERIES", "120"))
EPOCHS = int(os.environ.get("CM_EPOCHS", "4"))
LR = float(os.environ.get("CM_LR", "1e-2"))


def _log(m: str) -> None:
    print(f"[cm_distill] {m}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--buckets",
        default="",
        help="comma-separated bucket id prefixes to restrict to (e.g. bc6b90e2,40615ba9)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-distill even if a cartridge already exists (needed to re-run CM-A.2 buckets)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved work list and exit without loading any model",
    )
    args = ap.parse_args()

    cfg = Config.load(REPO)
    libucks_dir = REPO / ".libucks"
    cart_dir = libucks_dir / "kv_cache"
    cart_dir.mkdir(parents=True, exist_ok=True)
    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    store = BucketStore(libucks_dir / "buckets")
    embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    fixtures = json.loads(FIXTURES.read_text())["fixtures"]

    # buckets that any fixture routes to (top-1 centroid)
    centroids = registry.get_all_centroids()
    bids = list(centroids.keys())
    mat = np.stack([centroids[b] for b in bids])
    routed: set[str] = set()
    for fx in fixtures:
        routed.add(bids[int((mat @ embedder.embed(fx["question"])).argmax())])
    routed_list = sorted(routed)
    if args.buckets:
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        routed_list = [b for b in routed_list if any(b.startswith(w) for w in want)]
        unmatched = {w for w in want if not any(b.startswith(w) for b in routed_list)}
        if unmatched:
            sys.exit(f"[cm_distill] --buckets did not match any routed bucket: {sorted(unmatched)}")
    _log(f"{len(routed_list)} fixture-routed buckets to distill: {[b[:8] for b in routed_list]}")

    # ---- work list: resolve verbatim + honour resume/--force before loading models ----
    todo: list[tuple[str, str]] = []
    skipped, failed = 0, 0
    for bid in routed_list:
        if (cart_dir / f"{bid}.cartridge.safetensors").exists() and not args.force:
            _log(f"{bid[:8]} — cartridge exists, skip (use --force to re-distill)")
            skipped += 1
            continue
        fm, _ = store.read(bid)
        verbatim = _collect_source_text(fm, max_chars=4096) or ""
        if len(verbatim) < 50:
            _log(f"{bid[:8]} — verbatim too short ({len(verbatim)}), skip")
            failed += 1
            continue
        todo.append((bid, verbatim))

    if not todo:
        _log(f"=== nothing to do: skipped={skipped} failed={failed} ===")
        return

    if args.dry_run:
        _log(f"DRY RUN — would distill {len(todo)} bucket(s), "
             f"{N_QUERIES} queries each, P={PREFIX_LEN}, {EPOCHS}ep, lr={LR}:")
        for bid, verbatim in todo:
            _log(f"  {bid[:8]}  verbatim={len(verbatim)} chars")
        _log(f"query generator: {QGEN_ID} | receiver: {RECEIVER_ID}")
        _log(f"skipped={skipped} failed={failed}")
        return

    # ---- Phase 1: fact-probing query generation (0.5B resident; 3B not yet loaded) ----
    # CM-B.0b: this is the fix. Previously `model=None` here silently produced
    # templated queries — the configuration CM-A.1 had already shown to fail.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Query generation runs on CPU, deliberately. transformers' generate() aborts
    # the whole process on MPS for prompts of this length:
    #     MPSNDArray.mm:761: total bytes of NDArray > 2**32
    # It is a hard Metal assertion, not a Python exception, so it cannot be caught
    # or retried. Verified 2026-07-27 across greedy / sample / top-k / top-p and
    # both fp32 and bf16 — all four crash on MPS at a ~780-token prompt; CPU
    # completes in ~10 s. Distillation itself is unaffected because
    # CartridgeTrainer._teacher_generate uses a manual decode loop, not generate().
    qgen_device = torch.device("cpu")
    _log(f"loading query generator {QGEN_ID} on {qgen_device} (MPS generate() is broken; see note) ...")
    qgen_model = AutoModelForCausalLM.from_pretrained(
        QGEN_ID, dtype=torch.float32
    ).eval().to(qgen_device)
    qgen_tok = AutoTokenizer.from_pretrained(QGEN_ID)
    queries_by_bucket: dict[str, list[str]] = {}
    for n, (bid, verbatim) in enumerate(todo, 1):
        qs = generate_self_study_queries(
            verbatim, N_QUERIES, model=qgen_model, tokenizer=qgen_tok
        )
        queries_by_bucket[bid] = qs
        _log(f"[qgen {n}/{len(todo)}] {bid[:8]} — {len(qs)} queries")

    # Free the generator before the 3B loads: never hold both resident on 16 GB.
    del qgen_model, qgen_tok
    if device.type == "mps":
        torch.mps.empty_cache()
    _log("query generator freed")

    # Fail loudly rather than silently distilling on templates again — the exact
    # regression CM-B.0b exists to fix.
    template_only = [b for b, qs in queries_by_bucket.items() if not qs]
    if template_only:
        sys.exit(f"[cm_distill] query generation produced nothing for {template_only}; aborting")

    # ---- Phase 2: distillation (3B resident) ----
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(RECEIVER_ID, dtype=torch.bfloat16).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok, temperature=2.0, alpha_ce=0.3,
                               max_answer_tokens=32, max_verbatim_chars=4096)

    done = 0
    for n, (bid, verbatim) in enumerate(todo, 1):
        out_path = cart_dir / f"{bid}.cartridge.safetensors"
        try:
            t0 = time.time()
            cart = KVPrefixCartridge.for_model(model, prefix_len=PREFIX_LEN, dtype=torch.float32)
            cart.init_from_extracted_kv(extract_bucket_kv(model, tok, verbatim, max_tokens=1024))
            cart.to(device)
            queries = queries_by_bucket[bid]
            _log(f"[{n}/{len(todo)}] {bid[:8]} — distilling ({len(queries)} q, P={PREFIX_LEN}, {EPOCHS}ep) ...")
            ckpt_path = cart_dir / f"{bid}.cartridge.ckpt.safetensors"
            res = trainer.distill_bucket(cart, verbatim, queries, epochs=EPOCHS, lr=LR,
                                         checkpoint_path=ckpt_path)
            cart.save(out_path)
            ckpt_path.unlink(missing_ok=True)
            _log(f"[{n}/{len(todo)}] {bid[:8]} — saved. KL {res['init_mean_kl']:.3f}->{res['final_mean_kl']:.3f} "
                 f"({time.time()-t0:.0f}s)")
            done += 1
        except Exception as exc:
            _log(f"[{n}/{len(todo)}] {bid[:8]} — FAILED: {exc!r}")
            failed += 1

    _log(f"=== DONE: distilled={done} skipped={skipped} failed={failed} / {len(routed_list)} ===")


if __name__ == "__main__":
    main()
