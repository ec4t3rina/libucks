"""Aggregate repeated draws of one config into the error bar this track lacks.

Every headline in the cartridge work is a single sample compared against another
single sample — 2/8 vs 4/8, 7/25 vs 10/25, 5/8 vs 1/8 — because nothing in the
pipeline was seeded until 2026-07-28 and no config was ever run twice. This
reads the per-draw result JSONs and reports what the spread actually is, so a
delta can finally be judged against noise instead of assumed to be signal.

Two numbers matter and they are different:
  * score spread   — how much the headline N/8 moves between draws
  * fixture churn  — how many INDIVIDUAL fixtures flip verdict between draws.
    Churn can be high while the score looks stable (two fixtures swapping),
    which still means a per-fixture claim is not reproducible.

Run: uv run python scripts/cm_variance_report.py
     uv run python scripts/cm_variance_report.py --glob 'echoswarm_cartridge_A2_s*.json'
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent.parent / "tests/eval/results/cm"


def load_draw(path: Path) -> dict[str, Any]:
    """One results JSON -> {label, score, n, verdicts: {fixture_id: bool}}."""
    d = json.loads(path.read_text())
    per_q = d.get("per_question", [])
    return {
        "label": path.stem.replace("echoswarm_cartridge_A2_", "") or path.stem,
        "score": sum(1 for q in per_q if q.get("grounded")),
        "n": len(per_q),
        "verdicts": {q["id"]: bool(q.get("grounded")) for q in per_q},
    }


def summarise(draws: list[dict[str, Any]]) -> dict[str, Any]:
    """Score stats plus per-fixture churn across draws.

    `unstable` are fixtures that are not unanimous across draws — the ones whose
    individual pass/fail cannot be quoted from a single run.
    """
    scores = [d["score"] for d in draws]
    ids: list[str] = []
    for d in draws:
        for i in d["verdicts"]:
            if i not in ids:
                ids.append(i)

    unstable, always, never = [], [], []
    for i in ids:
        vals = [d["verdicts"][i] for d in draws if i in d["verdicts"]]
        if len(set(vals)) > 1:
            unstable.append(i)
        elif vals and vals[0]:
            always.append(i)
        else:
            never.append(i)

    return {
        "n_draws": len(draws),
        "scores": scores,
        "n_fixtures": draws[0]["n"] if draws else 0,
        "mean": statistics.mean(scores) if scores else 0.0,
        # Sample stdev needs >= 2 points; with 2-3 draws the RANGE is the more
        # honest statistic, so report both and lead with range in the verdict.
        "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
        "min": min(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "range": (max(scores) - min(scores)) if scores else 0,
        "always": always,
        "never": never,
        "unstable": unstable,
    }


def verdict(s: dict[str, Any], claim_delta: float | None) -> list[str]:
    """Plain-language read on whether a claimed delta survives the noise."""
    out: list[str] = []
    if s["n_draws"] < 2:
        out.append("Only one draw — no error bar is possible. Nothing here is a result yet.")
        return out
    if s["n_draws"] < 3:
        out.append("Two draws give a range, not a distribution. Treat as provisional.")
    out.append(
        f"Score range across {s['n_draws']} draws: {s['min']}-{s['max']}/"
        f"{s['n_fixtures']} (spread {s['range']})."
    )
    if s["unstable"]:
        out.append(
            f"{len(s['unstable'])} fixture(s) flip verdict between draws "
            f"({', '.join(s['unstable'])}) — per-fixture claims about these are "
            f"not reproducible from a single run."
        )
    else:
        out.append("No fixture flipped verdict — per-fixture results are stable.")
    if claim_delta is not None:
        if claim_delta <= s["range"]:
            out.append(
                f"A claimed delta of {claim_delta:g} is INSIDE the observed spread "
                f"({s['range']}). It is not distinguishable from noise at this sample size."
            )
        else:
            out.append(
                f"A claimed delta of {claim_delta:g} EXCEEDS the observed spread "
                f"({s['range']}), so it survives this (small-sample) noise check."
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="echoswarm_cartridge_A2_s*.json",
                    help="result files to treat as repeated draws of ONE config")
    ap.add_argument("--claim-delta", type=float, default=None,
                    help="a delta you want to claim (e.g. 3 for 1/8 -> 4/8); "
                         "reported against the observed spread")
    args = ap.parse_args()

    paths = sorted(RESULTS.glob(args.glob))
    if not paths:
        print(f"no result files matched {args.glob} in {RESULTS}", file=sys.stderr)
        return 2

    draws = [load_draw(p) for p in paths]
    widths = [len(d["label"]) for d in draws]
    w = max(widths) if widths else 8
    print(f"{'draw':<{w}}  score")
    for d in draws:
        print(f"{d['label']:<{w}}  {d['score']}/{d['n']}")

    s = summarise(draws)
    print()
    print(f"mean {s['mean']:.2f}  range {s['min']}-{s['max']}  spread {s['range']}"
          + (f"  stdev {s['stdev']:.2f}" if s["stdev"] is not None else ""))
    print(f"always grounded: {len(s['always'])}   never: {len(s['never'])}   "
          f"unstable: {len(s['unstable'])}")
    print()
    for line in verdict(s, args.claim_delta):
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
