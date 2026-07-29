"""How far can a raw KV-cache prefix be truncated before it stops working?

CM-B.0h found that `kv_first` — the first P positions of the REAL cache, with no
training at all — beats the distilled cartridge on 3 of 4 measurements (13/16 vs
8/16 at P=768). Since kv_first is provably identical to init_from_extracted_kv,
that means 98 minutes of gradient descent DEGRADES the cartridge's own starting
point. It also showed graceful degradation: 12/16 at P=384 vs 13/16 at P=768, so
halving the budget cost one fixture.

This maps the whole curve. It is the compression/accuracy tradeoff this project
has been trying to find since Phase 4-C, and it is now cheap because nothing is
trained.

EFFICIENCY. floor and cartridge do not depend on P, so they are generated ONCE.
The real cache is extracted once per bucket and re-sliced per P. Only the kv_first
arm repeats, which turns an O(P values x arms) sweep into O(P values).

DETERMINISM. The training-free arms have no seed and no optimiser, so their only
variance is MPS decode noise. That is a categorical improvement over the
cartridges, whose fixture scores spread by 3-4 across seeds.

Run: uv run python scripts/cm_kv_sweep.py --buckets bc6b90e2 \
         --prefix-lens 32,64,128,256,384,512,768
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


def _log(m: str) -> None:
    print(f"[cm_kv_sweep] {m}", file=sys.stderr, flush=True)


def parse_lens(s: str) -> list[int]:
    """Ascending, deduplicated, positive. Rejects junk loudly rather than
    silently sweeping a subset of what the caller asked for."""
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
    ap.add_argument("--buckets", default="", help="comma-separated bucket id prefixes")
    ap.add_argument("--fixtures", default="", help="alternate fixture JSON")
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument("--prefix-lens", default="32,64,128,256,384,512,768",
                    help="comma-separated P values for the kv_first arm")
    ap.add_argument("--verbatim-chars", type=int, default=10 ** 9,
                    help="bucket text budget. Defaults to UNBOUNDED: raw cache "
                         "extraction is not limited by the teacher context window "
                         "that motivated the 4096 cap elsewhere.")
    args = ap.parse_args()

    lens = parse_lens(args.prefix_lens)
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

    # A fixture may pin its bucket. The stratified set does, because routing is a
    # separate concern already measured elsewhere: 14 of its 20 questions route
    # away from simulation.py, which would silently drop them and make a
    # cache-content result depend on retrieval quality.
    def _target(f):
        declared = f.get("bucket")
        if declared:
            for b in bids:
                if b.startswith(declared):
                    return b
            sys.exit(f"[cm_kv_sweep] fixture {f['id']} declares unknown bucket {declared!r}")
        return bids[int((mat @ emb.embed(f["question"])).argmax())]

    routed = [(f, _target(f)) for f in fixtures]
    n_pinned = sum(1 for f in fixtures if f.get("bucket"))
    if n_pinned:
        _log(f"{n_pinned}/{len(fixtures)} fixtures pin their bucket explicitly "
             f"(routing bypassed by design)")
    if args.buckets:
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        routed = [(f, b) for f, b in routed if any(b.startswith(w) for w in want)]
    if not routed:
        sys.exit("[cm_kv_sweep] no fixtures matched")
    buckets = sorted({b for _, b in routed})
    _log(f"{len(routed)}/{len(fixtures)} fixtures from {fx_path.name}; P sweep {lens}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=torch.bfloat16
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok)

    # Reuse the tested selector/builder from cm_kv_prune rather than duplicating
    # the slicing logic — a second copy is how the two grounding scorers drifted.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cm_kv_prune", Path(__file__).resolve().parent / "cm_kv_prune.py")
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)

    prep: dict[str, dict] = {}
    for bid in buckets:
        p_file = cart_dir / f"{bid}.cartridge.safetensors"
        trained = None
        if p_file.exists():
            geo = KVPrefixCartridge.read_geometry(p_file)
            trained = KVPrefixCartridge(dtype=torch.float32, **geo)
            trained.load(p_file)
            trained.to(device)
        fm, _ = store.read(bid)
        # FULL bucket text, not the 4096-char cap. That cap existed for the
        # distillation teacher's context window; raw cache extraction has no such
        # constraint, and inheriting it silently limited a 20,013-char bucket to
        # its first 1,021 tokens — defeating the point of choosing a large bucket
        # and putting most of the stratified fixtures outside the cache entirely.
        verbatim = _collect_source_text(fm, max_chars=args.verbatim_chars) or ""
        flat = extract_bucket_kv(model, tok, verbatim, max_tokens=max(1024, max(lens)))
        seq = int(flat[kp.layer_keys(flat)[0][0]].shape[2])
        arms = {}
        for P in lens:
            idx = kp.select_indices(flat, "kv_first", P)
            arms[P] = kp.cartridge_from_selection(flat, idx, trained or KVPrefixCartridge(
                n_layers=len(kp.layer_keys(flat)),
                n_kv_heads=flat[kp.layer_keys(flat)[0][0]].shape[1],
                prefix_len=P,
                head_dim=flat[kp.layer_keys(flat)[0][0]].shape[3],
                dtype=torch.float32), device)
        prep[bid] = {"arms": arms, "trained": trained, "seq": seq,
                     "trained_P": trained.prefix_len if trained else None}
        _log(f"{bid[:8]} — real cache seq_len={seq}; built {len(lens)} kv_first arms")
        del flat
        if device.type == "mps":
            torch.mps.empty_cache()

    scores = {"floor": 0, "cartridge": 0, **{f"kv_first@{P}": 0 for P in lens}}
    rows = []
    for n, (f, bid) in enumerate(routed, 1):
        kw = f["answer_keywords"]
        row = {"id": f["id"], "bucket": bid, "kw": kw}
        plan = [("floor", None)]
        if prep[bid]["trained"] is not None:
            plan.append(("cartridge", prep[bid]["trained"]))
        plan += [(f"kv_first@{P}", prep[bid]["arms"][P]) for P in lens]
        for name, cart in plan:
            ans = trainer.generate_answer(cart, f["question"],
                                          max_new_tokens=EVAL_MAX_NEW_TOKENS,
                                          verbatim="")
            g = grounding_score(ans, kw)
            scores[name] += int(g)
            row[name] = {"grounded": g, "answer": ans}
        rows.append(row)
        _log(f"[{n:2}/{len(routed)}] {f['id']:14} "
             + " ".join(f"{k.replace('kv_first@','P')}={int(row[k]['grounded'])}"
                        for k, _ in plan))

    total = len(rows)
    seq = prep[buckets[0]]["seq"]
    _log("=" * 70)
    _log(f"floor (no memory):      {scores['floor']}/{total}")
    if prep[buckets[0]]["trained"] is not None:
        tp = prep[buckets[0]]["trained_P"]
        _log(f"distilled cartridge:    {scores['cartridge']}/{total}   (P={tp}, ~98 min to train)")
    _log("-" * 70)
    _log(f"{'P':>6}{'compression':>14}{'score':>10}{'vs floor':>10}")
    for P in lens:
        s = scores[f"kv_first@{P}"]
        _log(f"{P:>6}{seq / P:>13.2f}x{s:>7}/{total}{s - scores['floor']:>+10}")
    best = max(lens, key=lambda P: scores[f"kv_first@{P}"])
    _log("-" * 70)
    _log(f"best training-free: P={best} at {scores[f'kv_first@{best}']}/{total} "
         f"({seq / best:.2f}x compression), cost = ONE forward pass")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"echoswarm_kvsweep{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({
        "fixtures_file": fx_path.name, "n": total, "seq_len": seq,
        "prefix_lens": lens, "trained_P": prep[buckets[0]]["trained_P"],
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "metric": "grounding_score (CM-B.0a)",
        "scores": scores, "per_question": rows,
    }, indent=2))
    _log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
