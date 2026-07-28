"""No-context floor for the cartridge eval — the number that makes the rest mean something.

CM-B.0d leaves bc6b90e2's distilled cartridge at 0.67/8 (original fixtures) and
4.33/16 (extension set), each over 3 seeds. Neither is interpretable alone. Two
things could produce 4.33/16 with no memory involved at all:

  * the question leaks its own answer — "what are the five states an agent can be
    in?" in an evacuation-simulation context invites guesses like
    WAITING/INFORMED/EVACUATING/SAFE, which happen to be correct;
  * the base 3B already knows enough about this kind of code to score.

So this measures three arms on the SAME questions with the SAME decoder:

  floor       no prefix at all             -> what the model produces cold
  random      untrained random-init prefix -> does DISTILLATION do anything, or
                                              does any prefix of this shape help?
  cartridge   the distilled prefix         -> the claim

`cartridge - floor` is the only defensible statement of what the latent channel
contributes. `cartridge - random` separates trained content from the mere
presence of 128 extra attendable positions.

All three arms call CartridgeTrainer.generate_answer, which takes cartridge=None
for the floor, so tokenisation, prompt template, greedy selection and EOS
handling are shared by construction rather than by careful copying.

Run: uv run python scripts/cm_floor.py --buckets bc6b90e2
     uv run python scripts/cm_floor.py --buckets bc6b90e2 \
         --fixtures tests/eval/fixtures/echoswarm_qa_bc6b90e2_ext.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# MUST precede the torch import. Proven causally necessary for any MPS run here.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.eval_metrics import EVAL_MAX_NEW_TOKENS, grounding_score
from libucks.storage.bucket_registry import BucketRegistry
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer

REPO = Path("/Users/ecaterina/Developer/test-repos/echoswarm")
DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "tests/eval/fixtures/echoswarm_qa.json"
RESULTS = Path(__file__).resolve().parent.parent / "tests/eval/results/cm"
RECEIVER_ID = "Qwen/Qwen2.5-3B"


def _log(m: str) -> None:
    print(f"[cm_floor] {m}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buckets", default="", help="comma-separated bucket id prefixes")
    ap.add_argument("--fixtures", default="", help="alternate fixture JSON")
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS,
                    help="answer budget; defaults to the shared "
                         "eval_metrics.EVAL_MAX_NEW_TOKENS so this arm and the "
                         "cartridge eval cannot drift apart")
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
    emb = EmbeddingService.get_instance(cfg.model.embedding_model)
    cent = reg.get_all_centroids()
    bids = list(cent)
    mat = np.stack([cent[b] for b in bids])

    routed = [(f, bids[int((mat @ emb.embed(f["question"])).argmax())]) for f in fixtures]
    if args.buckets:
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        routed = [(f, b) for f, b in routed if any(b.startswith(w) for w in want)]
    if not routed:
        sys.exit("[cm_floor] no fixtures matched")
    _log(f"{len(routed)}/{len(fixtures)} fixtures from {fx_path.name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"loading {RECEIVER_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        RECEIVER_ID, dtype=torch.bfloat16
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(RECEIVER_ID)
    trainer = CartridgeTrainer(model, tok)

    trained: dict[str, KVPrefixCartridge | None] = {}
    randinit: dict[str, KVPrefixCartridge | None] = {}

    def _trained(bid: str):
        if bid not in trained:
            p = cart_dir / f"{bid}.cartridge.safetensors"
            if not p.exists():
                trained[bid] = None
            else:
                geo = KVPrefixCartridge.read_geometry(p)
                c = KVPrefixCartridge(dtype=torch.float32, **geo)
                c.load(p)
                trained[bid] = c.to(device)
        return trained[bid]

    def _random(bid: str):
        """Same geometry as the trained cartridge, never distilled."""
        if bid not in randinit:
            t = _trained(bid)
            if t is None:
                randinit[bid] = None
            else:
                c = KVPrefixCartridge(
                    n_layers=t.n_layers, n_kv_heads=t.n_kv_heads,
                    prefix_len=t.prefix_len, head_dim=t.head_dim,
                    dtype=torch.float32,
                )
                randinit[bid] = c.to(device)
        return randinit[bid]

    arms = {"floor": 0, "random": 0, "cartridge": 0}
    rows = []
    for n, (f, bid) in enumerate(routed, 1):
        kw = f["answer_keywords"]
        got = {}
        for arm in ("floor", "random", "cartridge"):
            cart = None if arm == "floor" else (
                _random(bid) if arm == "random" else _trained(bid)
            )
            if arm != "floor" and cart is None:
                got[arm] = (False, "<no cartridge on disk>")
                continue
            ans = trainer.generate_answer(
                cart, f["question"], max_new_tokens=args.max_new_tokens, verbatim=""
            )
            g = grounding_score(ans, kw)
            arms[arm] += int(g)
            got[arm] = (g, ans)
        rows.append({"id": f["id"], "bucket": bid, "kw": kw,
                     **{a: {"grounded": got[a][0], "answer": got[a][1]} for a in got}})
        _log(f"[{n:2}/{len(routed)}] {f['id']:14} "
             f"floor={int(got['floor'][0])} random={int(got['random'][0])} "
             f"cartridge={int(got['cartridge'][0])}")

    total = len(routed)
    _log("=" * 60)
    for a in ("floor", "random", "cartridge"):
        _log(f"{a:>10}: {arms[a]}/{total}")
    _log(f"cartridge - floor  = {arms['cartridge'] - arms['floor']:+d}"
         "   <- what the latent channel contributes")
    _log(f"cartridge - random = {arms['cartridge'] - arms['random']:+d}"
         "   <- what DISTILLATION contributes over an untrained prefix")
    if arms["cartridge"] <= arms["floor"]:
        _log("VERDICT: the cartridge does not beat having no memory at all. On this "
             "bucket the latent channel is inert.")
    elif arms["cartridge"] <= arms["random"]:
        _log("VERDICT: the cartridge does not beat an UNTRAINED prefix of the same "
             "shape. Any gain is from extra attendable positions, not learned content.")
    else:
        _log("VERDICT: the cartridge beats both floors on this fixture set. "
             f"Single-arm draw, n={total}; confirm across seeds before quoting.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"echoswarm_floor{('_' + args.tag) if args.tag else ''}.json"
    out = RESULTS / name
    out.write_text(json.dumps({
        "fixtures_file": fx_path.name, "n": total,
        "max_new_tokens": args.max_new_tokens,
        "metric": "grounding_score (CM-B.0a)",
        "scores": arms, "per_question": rows,
    }, indent=2))
    _log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
