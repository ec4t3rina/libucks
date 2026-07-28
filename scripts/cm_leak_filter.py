"""Re-score floor results with question-leaking fixtures excluded.

CM-B.0e measured the extension set's floor at 4/16 — the base 3B answers four of
those questions correctly with no memory at all, because items like "what are the
five states an agent can be in?" invite guesses that happen to be right. Any score
on the full 16 therefore mixes cartridge performance with the model's prior.

Uses only the committed floor snapshots, so this needs no GPU and no re-generation.

The leaky set is decided ONCE across every floor run available, not per run.
Per-run selection would move the denominator between arms — the same mistake as
changing the fixture file mid-track. A fixture counts as leaky if the floor
answered it in ANY run: a question the model can sometimes guess is not a clean
test even on a run where it happened to miss.

Run: uv run python scripts/cm_leak_filter.py
     uv run python scripts/cm_leak_filter.py --glob 'echoswarm_floor_b0g*ext16.json'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

RESULTS = Path(__file__).resolve().parent.parent / "tests/eval/results/cm"
ARMS = ("floor", "random", "cartridge")


def leaky_ids(runs: Iterable[dict[str, Any]]) -> set[str]:
    """Fixture ids the floor arm ever answered — the union over all runs."""
    out: set[str] = set()
    for r in runs:
        for q in r.get("per_question", []):
            if q.get("floor", {}).get("grounded"):
                out.add(q["id"])
    return out


def restricted(run: dict[str, Any], exclude: set[str]) -> dict[str, Any]:
    """Per-arm scores over the fixtures NOT in `exclude`, plus c-floor."""
    kept = [q for q in run.get("per_question", []) if q["id"] not in exclude]
    scores = {a: sum(1 for q in kept if q.get(a, {}).get("grounded")) for a in ARMS}
    return {
        "n": len(kept),
        **scores,
        "c_minus_floor": scores["cartridge"] - scores["floor"],
        "dropped": sorted(q["id"] for q in run.get("per_question", [])
                          if q["id"] in exclude),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="echoswarm_floor_*ext16.json",
                    help="floor snapshots to treat as runs of one fixture set")
    args = ap.parse_args()

    paths = sorted(RESULTS.glob(args.glob))
    if not paths:
        print(f"no floor snapshots matched {args.glob} in {RESULTS}", file=sys.stderr)
        return 2
    runs = [(p.stem.replace("echoswarm_floor_", ""), json.loads(p.read_text()))
            for p in paths]

    leaky = leaky_ids(r for _, r in runs)
    n_full = len(runs[0][1].get("per_question", []))
    print(f"{len(paths)} floor run(s), {n_full} fixtures each")
    print(f"question-leaking fixtures (floor answered them cold in >=1 run): "
          f"{len(leaky)}")
    for i in sorted(leaky):
        print(f"    {i}")
    if not leaky:
        print("  none — scores on this set are already clean")
    print()

    w = max((len(t) for t, _ in runs), default=8)
    print(f"{'run':<{w}}  {'FULL SET':>22}   {'LEAK-FREE SUBSET':>26}")
    print(f"{'':<{w}}  {'floor rand cart  c-fl':>22}   {'n  floor rand cart  c-fl':>26}")
    tot_full = tot_sub = 0
    for tag, r in runs:
        f = restricted(r, set())
        s = restricted(r, leaky)
        tot_full += f["c_minus_floor"]
        tot_sub += s["c_minus_floor"]
        print(f"{tag:<{w}}  "
              f"{f['floor']:>5} {f['random']:>4} {f['cartridge']:>4} "
              f"{f['c_minus_floor']:>+5}   "
              f"{s['n']:>4} {s['floor']:>5} {s['random']:>4} {s['cartridge']:>4} "
              f"{s['c_minus_floor']:>+5}")
    k = len(runs)
    print(f"{'mean c-floor':<{w}}  {tot_full / k:>+22.2f}   {tot_sub / k:>+26.2f}")
    print()
    if leaky and tot_sub <= 0 < tot_full:
        print("  The full-set gain came entirely from fixtures the model can guess. "
              "On the leak-free subset the cartridge adds nothing.")
    elif leaky:
        print("  Gain survives on the leak-free subset — that is the number to quote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
