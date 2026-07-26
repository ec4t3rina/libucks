#!/usr/bin/env python3
"""Re-score stored eval answers under the Stage 0a grounding metric.

Zero compute: every answer was already generated and saved. This only re-applies
the scoring function, so it isolates "how much of the reported failure was a
metric artifact" from "how much was the model".

Scores BOTH metrics side by side for EVERY path, because the normalization lifts
baselines too (no_context, text_clean, hybrid). Reporting only the cartridge
delta would be dishonest — the gate is an absolute bar, but the gap is the story.

Sources:
  tests/eval/results/phase4c/echoswarm_4c6_fairness.json  — 7 paths, full text
  tests/eval/results/cm/echoswarm_cartridge_A2.json       — the CM-A.2 cartridge

Usage:
    python scripts/cm_rescore_grounding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libucks.eval_metrics import grounding_score, keyword_hit  # noqa: E402

FAIRNESS = REPO_ROOT / "tests/eval/results/phase4c/echoswarm_4c6_fairness.json"
CARTRIDGE = REPO_ROOT / "tests/eval/results/cm/echoswarm_cartridge_A2.json"
FIXTURES = REPO_ROOT / "tests/eval/fixtures/echoswarm_qa.json"

GATE = 8  # CM-A.2 primary gate: latent-alone grounding >= 8/25


def _old_metric(answer: str, keywords: list[str]) -> bool:
    """The original _grounding_score: plain case-insensitive substring, >=50%."""
    if not keywords:
        return False
    ans = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in ans)
    return hits >= len(keywords) / 2.0


def _load_keywords() -> dict[str, list[str]]:
    fx = json.loads(FIXTURES.read_text())["fixtures"]
    return {f["id"]: f["answer_keywords"] for f in fx}


def _flip_detail(answer: str, keywords: list[str]) -> str:
    """Show which keywords the new metric newly credits."""
    gained = [
        kw for kw in keywords
        if keyword_hit(answer, kw) and kw.lower() not in answer.lower()
    ]
    return ", ".join(gained)


def main() -> None:
    kw_by_id = _load_keywords()
    print(f"{'path':28} {'old':>7} {'new':>7} {'delta':>6}")
    print("-" * 52)

    flips: list[tuple[str, str, str]] = []

    # ---- the 7 paths from the Phase 4-C fairness eval -------------------
    if FAIRNESS.exists():
        rows = json.loads(FAIRNESS.read_text())[0]["per_question"]
        paths = sorted({p for r in rows for p in r["answers"]})
        for path in paths:
            old = new = 0
            for r in rows:
                entry = r["answers"].get(path)
                if not entry:
                    continue
                kws = kw_by_id.get(r["id"], [])
                text = entry.get("text", "")
                o, n = _old_metric(text, kws), grounding_score(text, kws)
                old += o
                new += n
                if n and not o:
                    flips.append((path, r["id"], _flip_detail(text, kws)))
            mark = "  <-- gate" if path == "cache_aug_no_verbatim" else ""
            print(f"{path:28} {old:>4}/{len(rows):<2} {new:>4}/{len(rows):<2} {new-old:>+6}{mark}")

    # ---- the CM-A.2 cartridge run ---------------------------------------
    if CARTRIDGE.exists():
        d = json.loads(CARTRIDGE.read_text())
        rows = d["per_question"]
        old = new = 0
        for r in rows:
            kws = r.get("kw") or kw_by_id.get(r["id"], [])
            text = r.get("answer", "")
            o, n = _old_metric(text, kws), grounding_score(text, kws)
            old += o
            new += n
            if n and not o:
                flips.append(("cartridge (CM-A.2)", r["id"], _flip_detail(text, kws)))
        print("-" * 52)
        print(f"{'cartridge (CM-A.2)':28} {old:>4}/{len(rows):<2} {new:>4}/{len(rows):<2} {new-old:>+6}  <-- gate")
        print(f"\nreported in log: {d['grounding']}/{d['n']}  (old-metric reproduction: {old}/{len(rows)})")
        print(f"gate >= {GATE}/{d['n']}: old {'PASS' if old >= GATE else 'FAIL'} "
              f"-> new {'PASS' if new >= GATE else 'FAIL'}")

    if flips:
        print(f"\n{len(flips)} fixture(s) flipped to grounded:")
        for path, fid, gained in flips:
            print(f"  {path:22} {fid:14} newly credited: {gained}")


if __name__ == "__main__":
    main()
