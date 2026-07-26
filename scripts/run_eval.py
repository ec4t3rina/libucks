#!/usr/bin/env python3
"""Reproduce the headline eval end-to-end.

The headline number is libugry hybrid grounding 19.5 +/- 1.7 / 30 (4-run mean,
Qwen2.5-3B receiver). A single run lands anywhere in ~18-21/30; the spread is
real, so `--runs 4` is what the README figure means.

Two things this wrapper exists to enforce, because forgetting either has cost
whole nights of compute:

  1. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 is set before torch is imported.
     Without it, distill/eval runs on 16 GB Apple Silicon wedge in an
     uninterruptible MPS allocator wait (see docs/cartridges-log.md, CM-A.2
     redistill r4 vs r6 — a clean-process probe wedged without the env var and
     passed with it).
  2. The receiver model is read from the target repo's .libucks/config.toml and
     printed up front. A repo with no config.toml silently falls back to the
     0.5B default in libucks/config.py, which is NOT the configuration that
     produced the headline number.

Usage:
    python scripts/run_eval.py --repo /path/to/libugry --fixtures libugry
    python scripts/run_eval.py --repo /path/to/libugry --fixtures libugry --runs 4
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

# MUST precede any torch import, including transitive ones via libucks.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _report_config(repo: Path) -> None:
    """Print the receiver the eval will actually use, and warn if it is the default."""
    cfg_path = repo / ".libucks" / "config.toml"
    sys.path.insert(0, str(REPO_ROOT))
    from libucks.config import Config  # noqa: E402  (after env var is set)

    cfg = Config.load(repo)
    print(f"[run_eval] repo      : {repo}")
    print(f"[run_eval] encoder   : {cfg.model.local_model}")
    print(f"[run_eval] receiver  : {cfg.model.base_model}")
    print(f"[run_eval] quantized : {cfg.model.quantization}")
    if not cfg_path.exists():
        print(
            "[run_eval] WARNING: no .libucks/config.toml in this repo — the receiver "
            "above is the libucks/config.py DEFAULT, not a pinned choice. The headline "
            "19.5/30 used Qwen2.5-3B. Cross-config comparisons are not like-for-like.",
            file=sys.stderr,
        )
    lora = repo / ".libucks" / "lora_receiver.pt"
    if not lora.exists():
        print(
            f"[run_eval] WARNING: {lora} missing — the LoRA receiver is untrained, so "
            "the text/hybrid paths will not reflect the headline configuration.",
            file=sys.stderr,
        )


def _run_once(fixtures: str, run_idx: int, total: int) -> int | None:
    """Run the eval once. Returns hybrid grounding count, or None if unparseable."""
    print(f"\n[run_eval] ===== run {run_idx}/{total} =====", flush=True)
    env = {**os.environ, "LIBUCKS_EVAL_REPOS": fixtures}
    proc = subprocess.run(
        [
            "uv", "run", "pytest", "-m", "eval",
            "tests/eval/test_latent_vs_baseline.py", "-v", "-s",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(proc.stderr)
    print(proc.stdout)

    # The harness prints e.g. "[eval] hybrid      : grounding 19/30 (multi 5/10) cos 0.612"
    for line in (proc.stdout + proc.stderr).splitlines():
        if "[eval]" in line and "hybrid" in line and "grounding" in line:
            for tok in line.split():
                if "/" in tok and tok.split("/")[0].isdigit():
                    return int(tok.split("/")[0])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path, help="target repo with .libucks/")
    ap.add_argument(
        "--fixtures",
        required=True,
        choices=["libugry", "echoswarm", "click"],
        help="which fixture set / eval repo to run (sets LIBUCKS_EVAL_REPOS)",
    )
    ap.add_argument("--runs", type=int, default=1, help="repeat N times and report mean/stdev")
    args = ap.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / ".libucks").is_dir():
        sys.exit(f"[run_eval] no .libucks/ in {repo} — run `libucks init --local {repo}` first")

    _report_config(repo)

    scores: list[int] = []
    for i in range(1, args.runs + 1):
        got = _run_once(args.fixtures, i, args.runs)
        if got is None:
            print(f"[run_eval] run {i}: could not parse a hybrid grounding line", file=sys.stderr)
            continue
        scores.append(got)
        print(f"[run_eval] run {i}: hybrid grounding {got}")

    if not scores:
        sys.exit("[run_eval] no runs produced a parseable score")

    print("\n[run_eval] ===== summary =====")
    print(f"[run_eval] runs   : {scores}")
    print(f"[run_eval] mean   : {statistics.mean(scores):.1f}")
    if len(scores) > 1:
        print(f"[run_eval] stdev  : {statistics.stdev(scores):.1f}")
    print("[run_eval] reference: libugry hybrid 19.5 +/- 1.7 / 30 (Phase 4-A, 4-run mean, 3B)")


if __name__ == "__main__":
    main()
