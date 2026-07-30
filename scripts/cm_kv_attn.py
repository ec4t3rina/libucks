"""Does INFORMED selection compress where dumb selection did not?

THE STATE OF PLAY. CM-B.0i swept prefix budgets on 26 position-stratified fixtures
and found score ∝ fraction retained: 1/26 at P=128 (35.9×), 6/26 at P=1024 (4.5×),
13/26 with the whole 4,599-token cache. No plateau, no knee. CM-B.0j then confirmed
that 13/26 ceiling is REAL and not a harness artifact — text-in-prompt scores 14/26,
and the 5 disagreements are bidirectional. So the yardstick is trustworthy and the
negative stands.

But every selector tested was query-AGNOSTIC. kv_first/kv_last/kv_stride are
positional; kv_norm ranks by ‖K‖ and its own comment concedes it is "a cheap stand-in
for attention importance". The literature does not use a stand-in: SnapKV, H2O and
CompressKV keep the positions the ACTUAL QUERY attends to and report 97–99% of
full-cache accuracy at 3–19% budget. We measured 8% of ceiling at 3% budget with a
dumb selector and concluded no mechanism exists. That inference was never tested.

WHAT A RESULT HERE WOULD MEAN — both directions are informative.

  kv_attn ≫ kv_first   The information is concentrated and selection is the
                       mechanism. This is what SnapKV delivers in production, where
                       selection rides along on a prefill you must do anyway.
  kv_attn ≈ kv_first   No subset of that size suffices, so compression at that ratio
                       is impossible for this content by ANY selection method. A far
                       stronger and more general negative than "kv_first is bad".

HONEST SCOPE. Scoring needs the full cache resident, so this arm saves nothing at
measurement time — it buys attention-COMPUTE compression, not STORAGE compression,
and it is not a precomputable per-bucket cartridge. Using the query to select is not
leakage: the query is available at inference and only the question is used, never the
answer.

TWO THINGS THIS SCRIPT MUST GET RIGHT, both learned the hard way:

1. `attn_implementation="eager"`. transformers 5.4.0 with SDPA returns
   `attentions=()` and does NOT fall back. Unguarded, every score is zero, topk
   returns 0..p-1, and kv_attn silently BECOMES kv_first while being reported as
   query-aware — it would manufacture the null result it exists to test for.
   `select_indices_attn` raises rather than allow that; this script loads eager so it
   never fires.
2. Chunked extraction. Eager attention over 4,599 tokens in one forward materialises
   a (1, heads, 4599, 4599) matrix per layer. CM-B.0j also measured that one long
   forward permanently caches ~4.5 GB that `torch.mps.empty_cache()` will not return,
   making the real requirement 12.3 GB against 7.0 GB of tensors.

MEMORY. Even chunked this needs ~7 GB, most of it the 3B itself. On a 16 GB machine
that means closing the big editors first — CM-B.0j took four launch attempts to learn
that, and `cur`/`driver` are logged per fixture so starvation is distinguishable from
a leak at a glance.

Run: uv run python scripts/cm_kv_attn.py --buckets 95c8e099 \
         --fixtures tests/eval/fixtures/echoswarm_qa_95c8e099_strat.json \
         --prefix-lens 128,256,1024 --tag strat26
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
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
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = ROOT / "tests/eval/fixtures/echoswarm_qa.json"
RESULTS = ROOT / "tests/eval/results/cm"
RECEIVER_ID = "Qwen/Qwen2.5-3B"
# Per-budget arms. kv_first/kv_norm are query-AGNOSTIC (positional, magnitude);
# kv_attn/kv_attn_L are query-aware, global and per-layer.
SELECTORS = ("kv_first", "kv_norm", "kv_attn", "kv_attn_L")


def _log(m: str) -> None:
    print(f"[cm_kv_attn] {m}", file=sys.stderr, flush=True)


def _parse_lens(s: str) -> list[int]:
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError(f"prefix length must be positive, got {v}")
        out.add(v)
    if not out:
        raise ValueError("no prefix lengths given")
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buckets", default="")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prefix-lens", default="128,256,1024")
    ap.add_argument("--chunk-tokens", type=int, default=512,
                    help="cache is built in segments of this size; required because "
                         "eager attention over the whole bucket at once is huge")
    ap.add_argument("--verbatim-chars", type=int, default=10 ** 9)
    ap.add_argument("--selectors", default=",".join(SELECTORS),
                    help="which per-budget arms to run. Default is all four. A "
                         "sweep over many budgets should usually narrow this: "
                         "arms = 3 + len(selectors) * len(prefix_lens), and every "
                         "arm costs one generation per fixture, so four selectors "
                         "over six budgets is 702 generations.")
    args = ap.parse_args()

    sel = [s.strip() for s in args.selectors.split(",") if s.strip()]
    unknown = [s for s in sel if s not in SELECTORS]
    if unknown:
        sys.exit(f"[cm_kv_attn] unknown selector(s) {unknown}; "
                 f"choose from {list(SELECTORS)}")
    if not sel:
        sys.exit("[cm_kv_attn] no selectors given")

    lens = _parse_lens(args.prefix_lens)
    fx_path = Path(args.fixtures) if args.fixtures else DEFAULT_FIXTURES
    if not fx_path.is_absolute():
        fx_path = ROOT / fx_path
    fixtures = json.loads(fx_path.read_text())["fixtures"]

    cfg = Config.load(REPO)
    d = REPO / ".libucks"
    reg = BucketRegistry(d / "registry.json")
    reg.load()
    store = BucketStore(d / "buckets")
    emb = EmbeddingService.get_instance(cfg.model.embedding_model)
    cent = reg.get_all_centroids()
    bids = list(cent)
    mat = np.stack([cent[b] for b in bids])

    def _target(f):
        declared = f.get("bucket")
        if declared:
            for b in bids:
                if b.startswith(declared):
                    return b
            sys.exit(f"[cm_kv_attn] {f['id']} declares unknown bucket {declared!r}")
        return bids[int((mat @ emb.embed(f["question"])).argmax())]

    routed = [(f, _target(f)) for f in fixtures]
    if args.buckets:
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        routed = [(f, b) for f, b in routed if any(b.startswith(w) for w in want)]
    if not routed:
        sys.exit("[cm_kv_attn] no fixtures matched")
    buckets = sorted({b for _, b in routed})
    _log(f"{len(routed)}/{len(fixtures)} fixtures from {fx_path.name}; P {lens}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # EAGER IS MANDATORY — see the module docstring. SDPA yields attentions=() on
    # transformers 5.4.0 and kv_attn would silently degrade to kv_first.
    _log(f"loading {RECEIVER_ID} on {device} with attn_implementation=eager ...")
    model = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok)

    def _mem() -> str:
        if device.type != "mps":
            return ""
        parts = []
        for lbl, fn in (("cur", getattr(torch.mps, "current_allocated_memory", None)),
                        ("driver", getattr(torch.mps, "driver_allocated_memory", None))):
            if fn is not None:
                try:
                    parts.append(f"{lbl}={fn() / 2 ** 20:.0f}MB")
                except Exception:
                    pass
        return " ".join(parts)

    spec = importlib.util.spec_from_file_location(
        "cm_kv_prune", Path(__file__).resolve().parent / "cm_kv_prune.py")
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)

    prep: dict[str, dict] = {}
    for bid in buckets:
        fm, _ = store.read(bid)
        verbatim = _collect_source_text(fm, max_chars=args.verbatim_chars) or ""
        flat = extract_bucket_kv(model, tok, verbatim,
                                 max_tokens=max(1024, max(lens), 8192),
                                 chunk_tokens=args.chunk_tokens)
        seq = int(flat[kp.layer_keys(flat)[0][0]].shape[2])
        tmpl = KVPrefixCartridge(
            n_layers=len(kp.layer_keys(flat)),
            n_kv_heads=flat[kp.layer_keys(flat)[0][0]].shape[1],
            prefix_len=1, head_dim=flat[kp.layer_keys(flat)[0][0]].shape[3],
            dtype=torch.float32)

        p_file = d / "kv_cache" / f"{bid}.cartridge.safetensors"
        trained = None
        if p_file.exists():
            geo = KVPrefixCartridge.read_geometry(p_file)
            trained = KVPrefixCartridge(dtype=torch.float32, **geo)
            trained.load(p_file)
            trained.to(device)

        # Query-AGNOSTIC arms and the full cache do not depend on the query, so
        # build them once. kv_norm is the strongest dumb baseline and the fair
        # comparison for kv_attn: both rank positions, one just cannot see the query.
        agnostic = {}
        for P in lens:
            for how in ("kv_first", "kv_norm"):
                if how in sel:
                    agnostic[(how, P)] = kp.cartridge_from_selection(
                        flat, kp.select_indices(flat, how, P), tmpl, device)
        full = kp.cartridge_from_selection(
            flat, list(range(seq)), tmpl, device)

        prep[bid] = {"flat": flat, "seq": seq, "tmpl": tmpl, "trained": trained,
                     "agnostic": agnostic, "full": full}
        _log(f"{bid[:8]} — seq_len={seq} (chunked at {args.chunk_tokens}); "
             f"built {len(agnostic)} agnostic arms + full cache  {_mem()}")

    arms = ["floor", "full_cache"]
    if any(prep[b]["trained"] is not None for b in buckets):
        arms.append("cartridge")
    for P in lens:
        arms += [f"{s}@{P}" for s in sel]
    scores = dict.fromkeys(arms, 0)
    _log(f"{len(arms)} arms x {len(routed)} fixtures = "
         f"{len(arms) * len(routed)} generations; selectors {sel}")
    rows = []

    for n, (f, bid) in enumerate(routed, 1):
        kw, q = f["answer_keywords"], f["question"]
        pk = prep[bid]
        # ONE scoring forward per fixture, reused for every budget and both modes.
        t0 = time.perf_counter()
        s = kp.attn_scores(model, tok, pk["flat"], q, device=device, trainer=trainer)
        _log(f"  [{n:2}/{len(routed)}] {f['id']} scored in "
             f"{time.perf_counter() - t0:.1f}s  {_mem()}")

        plan = [("floor", None), ("full_cache", pk["full"])]
        if pk["trained"] is not None:
            plan.append(("cartridge", pk["trained"]))
        for P in lens:
            for how in sel:
                if how in ("kv_first", "kv_norm"):
                    cart = pk["agnostic"][(how, P)]
                elif how == "kv_attn":
                    cart = kp.cartridge_from_selection(
                        pk["flat"], kp.select_from_scores(s, P), pk["tmpl"], device)
                else:                                    # kv_attn_L
                    cart = kp.cartridge_from_per_layer_selection(
                        pk["flat"], kp.select_from_scores(s, P, per_layer=True),
                        pk["tmpl"], device)
                plan.append((f"{how}@{P}", cart))

        row = {"id": f["id"], "bucket": bid, "kw": kw}
        for name, cart in plan:
            t0 = time.perf_counter()
            ans = trainer.generate_answer(cart, q,
                                          max_new_tokens=EVAL_MAX_NEW_TOKENS,
                                          verbatim="")
            g = grounding_score(ans, kw)
            scores[name] += int(g)
            row[name] = {"grounded": g, "answer": ans}
            # Per-ARM heartbeat, not per-fixture. A 15-arm fixture can run many
            # minutes, and with only a per-fixture line a healthy-but-slow run is
            # indistinguishable from a wedged one — a stall watchdog killed a
            # perfectly good sweep for exactly that reason. Emitting here makes the
            # log grow steadily and yields per-arm timings for future estimates.
            _log(f"    {f['id']} {name:16} g={int(g)} "
                 f"{time.perf_counter() - t0:5.1f}s")
        row["mem"] = _mem()
        rows.append(row)
        _log(f"[{n:2}/{len(routed)}] {f['id']:14} "
             + " ".join(f"{k}={int(row[k]['grounded'])}" for k, _ in plan)
             + f"  {row['mem']}")
        if device.type == "mps":
            torch.mps.empty_cache()

    total = len(rows)
    seq = prep[buckets[0]]["seq"]
    _log("=" * 78)
    _log(f"floor       {scores['floor']}/{total}")
    _log(f"full cache  {scores['full_cache']}/{total}   (the CEILING, 1.0x)")
    if "cartridge" in scores:
        _log(f"cartridge   {scores['cartridge']}/{total}")
    _log("-" * 78)
    aware = [s for s in sel if s.startswith("kv_attn")]
    dumb = [s for s in sel if not s.startswith("kv_attn")]
    hdr = f"{'P':>6}{'ratio':>9}" + "".join(f"{s:>13}" for s in sel)
    if aware and dumb:
        hdr += f"{'aware-dumb':>12}"
    _log(hdr)

    def _best(names, P):
        return max((scores[f"{s}@{P}"] for s in names), default=0)

    for P in lens:
        line = f"{P:>6}{seq / P:>8.1f}x" + "".join(
            f"{scores[f'{s}@{P}']:>8}/{total}" for s in sel)
        if aware and dumb:
            line += f"{_best(aware, P) - _best(dumb, P):>+12}"
        _log(line)
    _log("-" * 78)
    best = max(lens, key=lambda P: _best(aware or sel, P))
    bv = _best(aware or sel, best)
    ceil = scores["full_cache"]
    _log(f"best query-aware: P={best} at {bv}/{total} = "
         f"{(100 * bv / ceil) if ceil else 0:.0f}% of the {ceil}/{total} ceiling "
         f"at {seq / best:.1f}x compression")
    _log("CM-B.0i's dumb selector reached 8% of ceiling at 35.9x. Above that is a "
         "mechanism; level with it is a much stronger negative.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"echoswarm_kvattn{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({
        "fixtures_file": fx_path.name, "n": total, "seq_len": seq,
        "prefix_lens": lens, "chunk_tokens": args.chunk_tokens,
        "attn_implementation": "eager",
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "metric": "grounding_score (CM-B.0a)",
        "scores": scores, "per_question": rows,
    }, indent=2))
    _log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
