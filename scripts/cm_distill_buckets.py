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
# Prefix length to TRAIN at. This is the source of truth: it is written into
# each cartridge's safetensors metadata, and cm_eval_cartridge reads it back via
# KVPrefixCartridge.read_geometry. It used to be restated there under reciprocal
# "MUST match" comments, which made a P sweep impossible without editing both.
PREFIX_LEN = int(os.environ.get("CM_PREFIX_LEN", "128"))
N_QUERIES = int(os.environ.get("CM_NQUERIES", "120"))
EPOCHS = int(os.environ.get("CM_EPOCHS", "4"))
LR = float(os.environ.get("CM_LR", "1e-2"))
# Verbatim budget, in characters. Feeds THREE places that must agree: the text
# collected from chunks, the teacher's context window, and the KV extraction
# length. They were three separate literals; a mismatch means the teacher answers
# from text the cartridge never encoded.
#
# Also the lever for the slice-on-overflow question. Since 30ee434 an overflowing
# block is truncated rather than dropped, so the verbatim now ends mid-statement
# ("if self.agent_") where it used to end at a chunk boundary ("return None").
# Setting this to a value that lands exactly on a boundary reproduces the old
# behaviour: for bc6b90e2 that is 3868 (vs 4132 today).
VERBATIM_CHARS = int(os.environ.get("CM_VERBATIM_CHARS", "4096"))
# Tokens to extract for the warm-start KV. ~4 chars/token, and it only has to
# cover PREFIX_LEN positions, but keeping it proportional avoids silently
# warm-starting from a fraction of a raised verbatim budget.
EXTRACT_TOKENS = int(os.environ.get("CM_EXTRACT_TOKENS", str(max(1024, VERBATIM_CHARS // 4))))
# Teacher answer budget. CM-A.1-retry (the only run that ever passed, 4/8 on
# bc6b90e2) used 48; CM-A.2 and CM-B.0b both used 32, which cannot fit a
# multi-identifier answer — echoswarm_11 needs six JSON keys. Default stays 32
# so this lever is inert unless set: silently changing a default is the exact
# mistake that made CM-A.2 re-run the losing configuration.
MAX_ANSWER_TOKENS = int(os.environ.get("CM_MAX_ANSWER_TOKENS", "32"))
# Abort if template fill-in exceeds this share of the query set. CM-B.0b ran
# bc6b90e2 at 30% templates and fe7ded0d at 33% without any signal. 0.35 lets
# the observed CM-B.0b runs through unchanged while catching a real collapse;
# tighten once a run demonstrates the generator can sustain a lower rate.
MAX_TEMPLATE_FRACTION = float(os.environ.get("CM_MAX_TEMPLATE_FRACTION", "0.35"))
# Use the 0.5B instruct model to write queries (1) or the curated fact-probing
# templates (0).
#
# READ THIS BEFORE CHANGING IT. `_TEMPLATES` has been the *fact-probing* set
# since bdba4ae (2026-07-02) — the very commit whose message reads "Fact-probing
# self-study queries (vs generic templates): latent-alone 2/8 -> 4/8". The 2/8
# loser was the GENERIC template set, deleted in that same commit. So
# `model=None` does NOT mean "the losing configuration"; it means the winning
# one. A comment added 2026-07-27 (12ca750) claimed otherwise and is what
# motivated CM-B.0b, which then displaced proven templates with 0.5B output and
# fell 5/8 -> 1/8. Default 1 preserves CM-B.0b behaviour; set 0 to reproduce
# CM-A.1-retry.
MODEL_QUERIES = os.environ.get("CM_MODEL_QUERIES", "1") == "1"
# Query-order seed. Unset reproduces the historical unseeded behaviour, under
# which no two runs of this script ever saw the same order — which is why every
# number in this track (2/8 vs 4/8, 7/25 vs 10/25, 5/8 vs 1/8) is a single
# sample with no error bar. Set it to make a run replayable; vary it, holding
# everything else fixed, to finally measure run-to-run variance.
SEED = int(os.environ["CM_SEED"]) if os.environ.get("CM_SEED") else None
# Promote the lowest-mean-KL epoch instead of the last one. Every run in this
# track shipped the last epoch; CM-B.0d's curves rose at epoch 2 in all three
# seeds, so the shipped cartridge was repeatedly not the best trained. Default 0
# preserves the historical protocol — a best-of-N run is NOT comparable with a
# last-epoch run, and the log says so when this is on.
KEEP_BEST = os.environ.get("CM_KEEP_BEST", "0") == "1"


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
        verbatim = _collect_source_text(fm, max_chars=VERBATIM_CHARS) or ""
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
    qgen_model = qgen_tok = None
    if MODEL_QUERIES:
        qgen_device = torch.device("cpu")
        _log(f"loading query generator {QGEN_ID} on {qgen_device} (MPS generate() is broken; see note) ...")
        qgen_model = AutoModelForCausalLM.from_pretrained(
            QGEN_ID, dtype=torch.float32
        ).eval().to(qgen_device)
        qgen_tok = AutoTokenizer.from_pretrained(QGEN_ID)
    else:
        _log("query generator DISABLED — using curated fact-probing templates "
             "(the CM-A.1-retry configuration)")
    queries_by_bucket: dict[str, list[str]] = {}
    qgen_stats: dict[str, dict] = {}
    for n, (bid, verbatim) in enumerate(todo, 1):
        st: dict = {}
        qs = generate_self_study_queries(
            verbatim, N_QUERIES, model=qgen_model, tokenizer=qgen_tok, stats=st
        )
        queries_by_bucket[bid] = qs
        qgen_stats[bid] = st
        _log(f"[qgen {n}/{len(todo)}] {bid[:8]} — {len(qs)} queries "
             f"(model={st['model']} template={st['template']})")

    # Free the generator before the 3B loads: never hold both resident on 16 GB.
    del qgen_model, qgen_tok
    if device.type == "mps":
        torch.mps.empty_cache()
    if MODEL_QUERIES:
        _log("query generator freed")

    # Catch an UNINTENDED collapse to templates. The former guard tested
    # `not qs`, which can NEVER be true — generate_self_study_queries always
    # tops up to n — so CM-B.0b passed it while distilling bc6b90e2 on 30%
    # templates. Test the composition instead.
    #
    # Skipped when MODEL_QUERIES=0, where 100% templates is the requested
    # configuration, not a failure.
    too_templated = {} if not MODEL_QUERIES else {
        b: s for b, s in qgen_stats.items()
        if s["template"] > MAX_TEMPLATE_FRACTION * max(1, s["model"] + s["template"])
    }
    if too_templated:
        detail = ", ".join(f"{b[:8]}: {s['model']} model / {s['template']} template"
                           for b, s in sorted(too_templated.items()))
        sys.exit(
            f"[cm_distill] query generation fell back to templates beyond "
            f"{MAX_TEMPLATE_FRACTION:.0%} for: {detail}. Templates are the "
            f"configuration CM-A.1 showed to fail (2/8 vs 4/8). Raise "
            f"CM_MAX_TEMPLATE_FRACTION to override deliberately, and say so in the log."
        )

    # ---- Phase 2: distillation (3B resident) ----
    # Echo the full recipe. CM-A.2's log entry claimed a configuration the run
    # did not execute; a reader of this log must never have to infer it again.
    _log(f"RECIPE: P={PREFIX_LEN} queries={N_QUERIES} epochs={EPOCHS} lr={LR} "
         f"max_answer_tokens={MAX_ANSWER_TOKENS} verbatim={VERBATIM_CHARS} extract={EXTRACT_TOKENS} "
         f"qgen={QGEN_ID if MODEL_QUERIES else 'fact-probing templates'} "
         f"receiver={RECEIVER_ID} seed={SEED if SEED is not None else 'UNSEEDED'} "
         f"keep_best={KEEP_BEST}")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(RECEIVER_ID, dtype=torch.bfloat16).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok, temperature=2.0, alpha_ce=0.3,
                               max_answer_tokens=MAX_ANSWER_TOKENS, max_verbatim_chars=VERBATIM_CHARS)

    done = 0
    for n, (bid, verbatim) in enumerate(todo, 1):
        out_path = cart_dir / f"{bid}.cartridge.safetensors"
        try:
            t0 = time.time()
            cart = KVPrefixCartridge.for_model(model, prefix_len=PREFIX_LEN, dtype=torch.float32)
            ckpt_path = cart_dir / f"{bid}.cartridge.ckpt.safetensors"
            side_path = Path(str(ckpt_path) + ".json")

            # Resume from a wedged run's per-epoch checkpoint. Before CM-B.0b the
            # checkpoint was written but never read, so a wedge cost the whole
            # ~2 h bucket instead of one epoch.
            epochs_done = 0
            if ckpt_path.exists() and side_path.exists():
                try:
                    epochs_done = int(json.loads(side_path.read_text())["epochs_done"])
                    cart.load(ckpt_path)   # validates geometry; raises on mismatch
                    _log(f"[{n}/{len(todo)}] {bid[:8]} — RESUMING from checkpoint "
                         f"({epochs_done}/{EPOCHS} epochs already done)")
                except Exception as exc:
                    _log(f"[{n}/{len(todo)}] {bid[:8]} — checkpoint unusable ({exc!r}); "
                         "starting from warm-start init")
                    epochs_done = 0
                    cart = KVPrefixCartridge.for_model(model, prefix_len=PREFIX_LEN,
                                                       dtype=torch.float32)

            if epochs_done == 0:
                cart.init_from_extracted_kv(
                    extract_bucket_kv(model, tok, verbatim, max_tokens=EXTRACT_TOKENS)
                )
            cart.to(device)

            remaining = EPOCHS - epochs_done
            if remaining <= 0:
                _log(f"[{n}/{len(todo)}] {bid[:8]} — checkpoint already has all "
                     f"{EPOCHS} epochs; saving as final")
                cart.save(out_path)
                ckpt_path.unlink(missing_ok=True)
                side_path.unlink(missing_ok=True)
                done += 1
                continue

            queries = queries_by_bucket[bid]
            _log(f"[{n}/{len(todo)}] {bid[:8]} — distilling ({len(queries)} q, "
                 f"P={PREFIX_LEN}, {remaining}ep{f' of {EPOCHS}, resumed' if epochs_done else ''}) ...")
            best_path = (cart_dir / f"{bid}.cartridge.best.safetensors") if KEEP_BEST else None
            res = trainer.distill_bucket(cart, verbatim, queries, epochs=remaining, lr=LR,
                                         checkpoint_path=ckpt_path, seed=SEED,
                                         best_path=best_path)
            cart.save(out_path)
            # Promote the best epoch over the last one when asked. Every run so far
            # shipped the LAST epoch, and CM-B.0d's epoch means rose at epoch 2 in
            # all three seeds (e.g. 5.754, 3.207, 4.753, 2.987), so the promoted
            # cartridge was repeatedly not the best one trained. Opt-in, because
            # best-of-N selection makes a run non-comparable with the ones that
            # took the last epoch — say so in the log when using it.
            if KEEP_BEST and best_path is not None and best_path.exists():
                be, bkl = res.get("best_epoch"), res.get("best_mean_kl")
                last_kl = res.get("final_mean_kl")
                if be is not None and be != remaining - 1:
                    import shutil
                    shutil.copyfile(best_path, out_path)
                    _log(f"[{n}/{len(todo)}] {bid[:8]} — PROMOTED epoch {be} "
                         f"(mean_kl={bkl:.4f}) over last epoch {remaining-1} "
                         f"(mean_kl={last_kl:.4f}). This run is NOT comparable with "
                         f"last-epoch runs.")
                else:
                    _log(f"[{n}/{len(todo)}] {bid[:8]} — best epoch WAS the last "
                         f"({be}); nothing promoted")
                best_path.unlink(missing_ok=True)
                Path(str(best_path) + ".json").unlink(missing_ok=True)
            ckpt_path.unlink(missing_ok=True)
            side_path.unlink(missing_ok=True)
            resumed = " (RESUMED — init_mean_kl is the first resumed epoch, "
            resumed += "not the true pre-training value)" if epochs_done else ""
            _log(f"[{n}/{len(todo)}] {bid[:8]} — saved. KL {res['init_mean_kl']:.3f}->{res['final_mean_kl']:.3f} "
                 f"({time.time()-t0:.0f}s){resumed if epochs_done else ''}")
            done += 1
        except Exception as exc:
            _log(f"[{n}/{len(todo)}] {bid[:8]} — FAILED: {exc!r}")
            failed += 1

    _log(f"=== DONE: distilled={done} skipped={skipped} failed={failed} / {len(routed_list)} ===")


if __name__ == "__main__":
    main()
