"""How far can a raw KV-cache prefix be truncated before it stops working?

HISTORY — read this before trusting any curve this script prints. CM-B.0h ran it
on the bc6b90e2 extension set and reported graceful degradation (13/16 at P=768,
12/16 at P=384, 8/16 at P=128 = 7.9x compression), concluding that raw cache
prefixes compress and that training degrades its own warm start. CM-B.0i
RETRACTED both conclusions: those fixtures clustered in the head of the file, so
`kv_first` won by construction — the scores tracked the count of fixtures whose
keywords fall inside the first P tokens, not any compression property. On the
position-stratified 95c8e099 set the curve flattened into plain proportional
decay (8% of ceiling at 36x, 46% at 4.5x, no plateau anywhere).

So this script maps a curve; it does NOT establish that the curve means
compression. That requires fixtures whose target facts are spread across the
whole document — see tests/unit/test_fixture_stratification.py.

THE CEILING CONTROL (added 2026-07-30). CM-B.0i's ceiling was 13/26: barely half,
with the ENTIRE bucket in cache. Every compression number is a fraction of that,
so if the ceiling is itself an artifact, the whole log is measured against a bent
ruler. The `text` arm settles it by feeding the same bucket text as prompt tokens
the model reads with full attention:

    text ~= kv_first@<full>   -> ceiling is real (model- or fixture-bound)
    text >> kv_first@<full>   -> the cache path is lossy; redo the sweep

It is the one arm here whose prompt is large, and therefore the only one that
could be silently truncated. generate_answer now raises instead; the budget is
computed from the actual fixtures below and logged.

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
from libucks.thinking.training.cartridge_trainer import (
    MAX_PROMPT_TOKENS,
    CartridgeTrainer,
)
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

    def _mem() -> str:
        """Per-fixture MPS accounting.

        Three runs of this sweep died after 0-2 fixtures and I diagnosed it as a
        per-fixture leak by INFERENCE, then found that two of the three had
        launched with half the memory they needed — so the inference was
        unsupported. This settles it with data instead: if `driver` is flat across
        fixtures there is no leak and we are simply memory-starved; if it climbs,
        there is a real bug and no amount of headroom will save the run.

        `current` is what tensors hold; `driver` is what PyTorch has taken from the
        OS and, with PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0, never has to give back.
        The gap between them is cached-but-unused, which is the suspect.
        """
        if device.type != "mps":
            return ""
        out = []
        for label, fn in (("cur", getattr(torch.mps, "current_allocated_memory", None)),
                          ("driver", getattr(torch.mps, "driver_allocated_memory", None))):
            if fn is not None:
                try:
                    out.append(f"{label}={fn() / 2 ** 20:.0f}MB")
                except Exception:
                    pass
        return " ".join(out)

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
                     "verbatim": verbatim,
                     "trained_P": trained.prefix_len if trained else None}
        _log(f"{bid[:8]} — real cache seq_len={seq}; built {len(lens)} kv_first arms")
        del flat
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- ceiling control: prefill the shared verbatim ONCE -------------------
    # The naive form of this arm (one full-length forward per fixture) exhausted
    # 16 GB plus 20 GB of swap and wedged after 2 of 26 fixtures. With
    # PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 — mandatory here — the allocator has no
    # ceiling, so 26 large transient attention workspaces swapped instead of
    # raising. The verbatim prefix is IDENTICAL across every prompt and only the
    # trailing question differs, so it is prefilled once and each fixture forwards
    # just its own tail against a fresh copy: one big forward, not 26.
    from transformers.cache_utils import DynamicCache

    text_need = max(
        len(tok(trainer._q_text(f["question"], prep[bid]["verbatim"]))["input_ids"])
        for f, bid in routed
    )
    _log(f"text arm: a single-shot prompt would be {text_need} tokens "
         f"(the {MAX_PROMPT_TOKENS} default would have truncated: "
         f"{'YES' if text_need > MAX_PROMPT_TOKENS else 'no'}); prefilling the "
         f"shared prefix once instead")

    for bid in buckets:
        head = prep[bid]["verbatim"] + "\n\n"
        head_ids = tok(head)["input_ids"]
        # token-exact or nothing: prefilling `head` and forwarding only the tail
        # equals the single-shot prompt ONLY if tokenisation is split-invariant at
        # the junction. BPE is not split-invariant in general, so check every
        # fixture rather than assume, and die loudly if any disagrees — a silent
        # off-by-a-token at the seam would shift every position by one and make
        # the control incomparable to the arm it is meant to validate.
        for f, b in routed:
            if b != bid:
                continue
            full = tok(trainer._q_text(f["question"], prep[bid]["verbatim"]))["input_ids"]
            tail = tok(trainer._q_text(f["question"], ""))["input_ids"]
            if list(full) != list(head_ids) + list(tail):
                sys.exit(
                    f"[cm_kv_sweep] tokenisation is not split-invariant at the "
                    f"verbatim/question junction for {f['id']}: "
                    f"{len(full)} != {len(head_ids)}+{len(tail)}. The one-time "
                    f"prefill would not equal a single-shot prompt; do not "
                    f"interpret this arm."
                )
        head_pt = tok(head, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            o = model(input_ids=head_pt,
                      attention_mask=torch.ones_like(head_pt), use_cache=True)
        # Keep only the tensors. Holding the DynamicCache itself would keep the
        # whole model output graph alive alongside it.
        prep[bid]["text_kv"] = [(L.keys.detach(), L.values.detach())
                                for L in o.past_key_values.layers]
        prep[bid]["text_len"] = int(head_pt.shape[1])
        del o
        if device.type == "mps":
            torch.mps.empty_cache()
        _log(f"{bid[:8]} — text control prefilled: {prep[bid]['text_len']} tokens "
             f"(verbatim + separator), token-exact split verified on "
             f"{sum(1 for _, b in routed if b == bid)} fixtures  {_mem()}")

    def _text_factory(bid: str):
        """Fresh DynamicCache per generation.

        `.clone()` is deliberate. DynamicCache currently replaces its per-layer
        tensors on update rather than mutating them, so the master would probably
        survive being shared — but 'probably' rests on a transformers internal,
        and the failure mode is silent: fixture N's decoded answer would become
        part of fixture N+1's prefix. 170 MB of transient copy is the cheaper
        side of that trade.
        """
        kv, n = prep[bid]["text_kv"], prep[bid]["text_len"]

        def factory():
            c = DynamicCache()
            for i, (k, v) in enumerate(kv):
                c.update(k.clone(), v.clone(), layer_idx=i)
            return c, n

        return factory

    factories = {bid: _text_factory(bid) for bid in buckets}

    scores = {"floor": 0, "text": 0, "cartridge": 0,
              **{f"kv_first@{P}": 0 for P in lens}}
    rows = []
    for n, (f, bid) in enumerate(routed, 1):
        kw = f["answer_keywords"]
        row = {"id": f["id"], "bucket": bid, "kw": kw}
        # (name, cartridge, prefix_factory). Every arm shares one decode loop; the
        # arms differ only in where the prefix came from. `text` gets it from a
        # live prefill, `kv_first@P` from the serialised round-trip — which is
        # precisely the difference under test.
        plan = [("floor", None, None), ("text", None, factories[bid])]
        if prep[bid]["trained"] is not None:
            plan.append(("cartridge", prep[bid]["trained"], None))
        plan += [(f"kv_first@{P}", prep[bid]["arms"][P], None) for P in lens]
        for name, cart, fac in plan:
            ans = trainer.generate_answer(cart, f["question"],
                                          max_new_tokens=EVAL_MAX_NEW_TOKENS,
                                          verbatim="", prefix_factory=fac)
            g = grounding_score(ans, kw)
            scores[name] += int(g)
            row[name] = {"grounded": g, "answer": ans}
        row["mem"] = _mem()
        rows.append(row)
        _log(f"[{n:2}/{len(routed)}] {f['id']:14} "
             + " ".join(f"{k.replace('kv_first@','P')}={int(row[k]['grounded'])}"
                        for k, _, _ in plan)
             + f"  {row['mem']}")

    total = len(rows)
    seq = prep[buckets[0]]["seq"]
    _log("=" * 70)
    _log(f"floor (no memory):      {scores['floor']}/{total}")
    _log(f"text in prompt:         {scores['text']}/{total}   "
         f"(CEILING CONTROL: live prefill of {prep[buckets[0]]['text_len']} "
         f"tokens, no serialisation round-trip)")
    full = max(lens)
    if full >= seq:
        cache_full = scores[f"kv_first@{full}"]
        gap = scores["text"] - cache_full
        _log(f"whole cache (P={full}):    {cache_full}/{total}")
        verdict = ("cache path is LOSSY — redo the sweep" if gap >= 3
                   else "ceiling is REAL — model/fixture-bound, not harness-bound"
                   if abs(gap) <= 2 else "cache BEATS text — unexpected, investigate")
        _log(f"  text - whole cache = {gap:+d}  ->  {verdict}")
    else:
        _log(f"  (no full-cache arm: max P={full} < seq_len={seq}; "
             f"add a P >= {seq} to compare against the ceiling control)")
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
        "text_single_shot_tokens": text_need,
        "text_prefill_tokens": prep[buckets[0]]["text_len"],
        "metric": "grounding_score (CM-B.0a)",
        "scores": scores, "per_question": rows,
    }, indent=2))
    _log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
