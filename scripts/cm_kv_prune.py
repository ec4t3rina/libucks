"""Training-free KV-cache selection vs. a distilled cartridge.

THE QUESTION. Distillation has to *relearn* the bucket's identifiers from a few
thousand supervised tokens; a slice of the REAL KV cache already contains their
activations. If some training-free selection of the real cache matches or beats a
90-minute distill, then cartridges are the wrong mechanism for this problem and the
simpler answer wins.

That is not a hypothetical concern. CM-B.0f showed P was the binding constraint,
and bc6b90e2's verbatim is only ~1009 tokens — so at P=384 the "compression" is
2.6x and at P=768 it is 1.3x. As P approaches seq_len a learned prefix stops
compressing the cache and starts merely reparameterising it, and cache pruning
(SnapKV / H2O / StreamingLLM in the literature) is the established, gradient-free
way to do that.

ARMS, all at the same P as the trained cartridge, all sharing one decode loop:

  floor       no prefix                        the guessing baseline
  kv_first    real cache, first P positions    == what init_from_extracted_kv does,
                                               i.e. the UNTRAINED warm start whose
                                               score has never been measured here
  kv_last     real cache, last P               recency
  kv_stride   real cache, every k-th position   uniform coverage of the document
  kv_norm     real cache, top P by ||K||        cheap importance proxy, no attention
                                               rollout needed
  cartridge   the distilled prefix              the incumbent

Every arm is loaded into a KVPrefixCartridge and generated through
CartridgeTrainer.generate_answer, so tokenisation, prompt, greedy selection and
mask construction are identical across arms by construction — the same discipline
that made cm_floor's deltas attributable.

Run: uv run python scripts/cm_kv_prune.py --buckets bc6b90e2
     uv run python scripts/cm_kv_prune.py --buckets bc6b90e2 \
         --fixtures tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.cache_augmentation.kv_extract import extract_bucket_kv
from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.eval_metrics import EVAL_MAX_NEW_TOKENS, grounding_score
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer
from libucks.thinking.training.data_generator import _collect_source_text

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RESULTS = Path(__file__).resolve().parent.parent / "tests/eval/results/cm"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
SELECTORS = ("kv_first", "kv_last", "kv_stride", "kv_norm")
ARMS = ("floor",) + SELECTORS + ("cartridge",)


def _log(m: str) -> None:
    print(f"[cm_kv_prune] {m}", file=sys.stderr, flush=True)


def layer_keys(flat: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """[(K key, V key)] in layer order, ignoring the _meta_* scalars."""
    n = sum(1 for k in flat if k.endswith("_K") and not k.startswith("_meta"))
    return [(f"layer_{i}_K", f"layer_{i}_V") for i in range(n)]


def select_indices(flat: dict[str, torch.Tensor], how: str, p: int) -> list[int]:
    """Positions of the real cache to keep. Deterministic for every selector."""
    kk, _ = layer_keys(flat)[0]
    seq = int(flat[kk].shape[2])
    p = min(p, seq)
    if how == "kv_first":
        return list(range(p))
    if how == "kv_last":
        return list(range(seq - p, seq))
    if how == "kv_stride":
        # Uniform coverage including both endpoints, so the tail of the document
        # is represented — the failure mode kv_first cannot avoid.
        return sorted({int(round(i * (seq - 1) / max(1, p - 1))) for i in range(p)})
    if how == "kv_norm":
        # ||K|| summed over layers and heads. A cheap stand-in for attention
        # importance that needs no second forward pass. Position order is
        # preserved after selection: shuffling positions would scramble the
        # relative ordering the model's RoPE-encoded keys still carry.
        score = None
        for kkey, _v in layer_keys(flat):
            k = flat[kkey].float()               # (1, heads, seq, dim)
            n = k.norm(dim=-1).sum(dim=1)[0]     # (seq,)
            score = n if score is None else score + n
        top = torch.topk(score, k=p).indices.tolist()
        return sorted(top)
    raise ValueError(f"unknown selector {how!r}")


def cartridge_from_selection(flat, idx: list[int], template: KVPrefixCartridge,
                             device) -> KVPrefixCartridge:
    """Build a cartridge whose slots ARE the chosen real-cache positions."""
    c = KVPrefixCartridge(
        n_layers=template.n_layers, n_kv_heads=template.n_kv_heads,
        prefix_len=len(idx), head_dim=template.head_dim, dtype=torch.float32,
    )
    sel = torch.tensor(idx, dtype=torch.long)
    with torch.no_grad():
        for i, (kkey, vkey) in enumerate(layer_keys(flat)):
            c.k[i].copy_(flat[kkey].float().index_select(2, sel))
            c.v[i].copy_(flat[vkey].float().index_select(2, sel))
    return c.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buckets", default="", help="comma-separated bucket id prefixes")
    ap.add_argument("--fixtures", default="", help="alternate fixture JSON")
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument("--prefix-len", type=int, default=0,
                    help="P for the training-free arms; defaults to the trained "
                         "cartridge's own P so the arms are matched on capacity")
    args = ap.parse_args()

    fx_path = Path(args.fixtures) if args.fixtures else DEFAULT_FIXTURES
    if not fx_path.is_absolute():
        fx_path = Path(__file__).resolve().parent.parent / fx_path
    fixtures = json.loads(fx_path.read_text())["fixtures"]

    cfg = Config.load(REPO)
    d = REPO / ".libucks"
    cart_dir = d / "kv_cache"
    reg = BucketRegistry(d / "registry.json")
    reg.load()
    store = BucketStore(d / "buckets")
    emb = EmbeddingService.get_instance(cfg.model.embedding_model)
    cent = reg.get_all_centroids()
    bids = list(cent)
    mat = np.stack([cent[b] for b in bids])

    routed = [(f, bids[int((mat @ emb.embed(f["question"])).argmax())]) for f in fixtures]
    if args.buckets:
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        routed = [(f, b) for f, b in routed if any(b.startswith(w) for w in want)]
    if not routed:
        sys.exit("[cm_kv_prune] no fixtures matched")
    buckets = sorted({b for _, b in routed})
    _log(f"{len(routed)}/{len(fixtures)} fixtures from {fx_path.name}, "
         f"{len(buckets)} bucket(s)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=torch.bfloat16
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok)

    prepared: dict[str, dict] = {}
    for bid in buckets:
        p_file = cart_dir / f"{bid}.cartridge.safetensors"
        if not p_file.exists():
            _log(f"{bid[:8]} — no trained cartridge; skipping bucket")
            continue
        geo = KVPrefixCartridge.read_geometry(p_file)
        trained = KVPrefixCartridge(dtype=torch.float32, **geo)
        trained.load(p_file)
        trained.to(device)
        P = args.prefix_len or geo["prefix_len"]

        fm, _ = store.read(bid)
        verbatim = _collect_source_text(fm, max_chars=4096) or ""
        flat = extract_bucket_kv(model, tok, verbatim, max_tokens=max(1024, P))
        seq = int(flat[layer_keys(flat)[0][0]].shape[2])
        _log(f"{bid[:8]} — P={P}, real cache seq_len={seq} "
             f"({seq / max(1, P):.2f}x compression)")

        arms = {"cartridge": trained}
        for how in SELECTORS:
            idx = select_indices(flat, how, P)
            arms[how] = cartridge_from_selection(flat, idx, trained, device)
        prepared[bid] = {"arms": arms, "P": P, "seq": seq}
        del flat
        if device.type == "mps":
            torch.mps.empty_cache()

    if not prepared:
        sys.exit("[cm_kv_prune] no bucket had a trained cartridge to compare against")

    scores = {a: 0 for a in ARMS}
    rows = []
    for n, (f, bid) in enumerate(routed, 1):
        if bid not in prepared:
            continue
        kw = f["answer_keywords"]
        row = {"id": f["id"], "bucket": bid, "kw": kw}
        for a in ARMS:
            cart = None if a == "floor" else prepared[bid]["arms"][a]
            ans = trainer.generate_answer(cart, f["question"],
                                          max_new_tokens=EVAL_MAX_NEW_TOKENS,
                                          verbatim="")
            g = grounding_score(ans, kw)
            scores[a] += int(g)
            row[a] = {"grounded": g, "answer": ans}
        rows.append(row)
        _log(f"[{n:2}/{len(routed)}] {f['id']:14} "
             + "  ".join(f"{a}={int(row[a]['grounded'])}" for a in ARMS))

    total = len(rows)
    _log("=" * 66)
    for a in ARMS:
        delta = scores[a] - scores["floor"]
        _log(f"{a:>10}: {scores[a]}/{total}   vs floor {delta:+d}")
    best_free = max(SELECTORS, key=lambda a: scores[a])
    _log("-" * 66)
    if scores[best_free] > scores["cartridge"]:
        _log(f"VERDICT: training-free {best_free} ({scores[best_free]}/{total}) BEATS the "
             f"distilled cartridge ({scores['cartridge']}/{total}). A 90-minute distill "
             f"is being outperformed by slicing the real cache.")
    elif scores[best_free] == scores["cartridge"]:
        _log(f"VERDICT: training-free {best_free} MATCHES the cartridge "
             f"({scores['cartridge']}/{total}) — distillation is buying nothing over "
             f"cache selection at this P.")
    else:
        _log(f"VERDICT: the cartridge ({scores['cartridge']}/{total}) beats the best "
             f"training-free arm ({best_free} {scores[best_free]}/{total}); "
             f"distillation earns its cost. n={total}, single draw.")
    _log("Note: kv_first is exactly the untrained warm start init_from_extracted_kv "
         "produces, so cartridge - kv_first is what the gradient steps actually add.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"echoswarm_kvprune{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({
        "fixtures_file": fx_path.name, "n": total,
        "prefix_len": {b: prepared[b]["P"] for b in prepared},
        "seq_len": {b: prepared[b]["seq"] for b in prepared},
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "metric": "grounding_score (CM-B.0a)",
        "scores": scores, "per_question": rows,
    }, indent=2))
    _log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
