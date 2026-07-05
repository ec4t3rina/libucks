"""CM-A.2 batch distiller — distill a KV-prefix cartridge for every echoswarm
bucket that any fixture routes to, saving each to disk.

Resumable: skips buckets whose cartridge already exists, so a sleep/crash just
requires re-running (it continues where it left off). A per-epoch checkpoint
(<bucket_id>.cartridge.ckpt.safetensors) is written during training and removed
on success, so an MPS wedge leaves at most one lost epoch behind — the final
.cartridge.safetensors only appears once all epochs completed.

Cartridge saved to: <repo>/.libucks/kv_cache/<bucket_id>.cartridge.safetensors

Run (overnight, kept awake):
  caffeinate -dimsu uv run python scripts/cm_distill_buckets.py
"""
from __future__ import annotations

import json
import sys
import time
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

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
PREFIX_LEN = 128          # MUST match cm_eval_cartridge.py
N_QUERIES = 120
EPOCHS = 4
LR = 1e-2


def _log(m: str) -> None:
    print(f"[cm_distill] {m}", file=sys.stderr, flush=True)


def main() -> None:
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
    _log(f"{len(routed_list)} fixture-routed buckets to distill: {[b[:8] for b in routed_list]}")

    # load frozen 3B once
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(RECEIVER_ID, dtype=torch.bfloat16).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok, temperature=2.0, alpha_ce=0.3,
                               max_answer_tokens=32, max_verbatim_chars=4096)

    done, skipped, failed = 0, 0, 0
    for n, bid in enumerate(routed_list, 1):
        out_path = cart_dir / f"{bid}.cartridge.safetensors"
        if out_path.exists():
            _log(f"[{n}/{len(routed_list)}] {bid[:8]} — cartridge exists, skip")
            skipped += 1
            continue
        try:
            fm, _ = store.read(bid)
            verbatim = _collect_source_text(fm, max_chars=4096) or ""
            if len(verbatim) < 50:
                _log(f"[{n}/{len(routed_list)}] {bid[:8]} — verbatim too short ({len(verbatim)}), skip")
                failed += 1
                continue
            t0 = time.time()
            cart = KVPrefixCartridge.for_model(model, prefix_len=PREFIX_LEN, dtype=torch.float32)
            cart.init_from_extracted_kv(extract_bucket_kv(model, tok, verbatim, max_tokens=1024))
            cart.to(device)
            queries = generate_self_study_queries(verbatim, N_QUERIES, model=None)
            _log(f"[{n}/{len(routed_list)}] {bid[:8]} — distilling ({len(queries)} q, P={PREFIX_LEN}, {EPOCHS}ep) ...")
            ckpt_path = cart_dir / f"{bid}.cartridge.ckpt.safetensors"
            res = trainer.distill_bucket(cart, verbatim, queries, epochs=EPOCHS, lr=LR,
                                         checkpoint_path=ckpt_path)
            cart.save(out_path)
            ckpt_path.unlink(missing_ok=True)
            _log(f"[{n}/{len(routed_list)}] {bid[:8]} — saved. KL {res['init_mean_kl']:.3f}->{res['final_mean_kl']:.3f} "
                 f"({time.time()-t0:.0f}s)")
            done += 1
        except Exception as exc:
            _log(f"[{n}/{len(routed_list)}] {bid[:8]} — FAILED: {exc!r}")
            failed += 1

    _log(f"=== DONE: distilled={done} skipped={skipped} failed={failed} / {len(routed_list)} ===")


if __name__ == "__main__":
    main()
