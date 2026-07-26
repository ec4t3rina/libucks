#!/usr/bin/env python3
"""Is a trained cartridge's prefix localizable? — CPU-only structural probe.

Stage 1's research bet is *slot-localized repair*: when one chunk changes, retrain
only the cartridge slots that carry it instead of the whole prefix. That is only
possible if the P slots are differentiated. If distillation collapsed them into P
near-identical vectors, there is nothing to localize and the bet is dead — worth
knowing before building the machinery.

This reads a cartridge off disk and measures, per layer:

  slot_cos_mean   mean pairwise cosine between the P slot vectors.
                  ~1.0 => collapsed, all slots carry the same thing (bad).
                  low  => differentiated, slots are distinguishable (good).
  eff_rank        effective rank via the entropy of the normalized singular value
                  spectrum, exp(H). 1.0 => rank-1 collapse. Near P => slots span
                  the space. This is the honest version of "are they different",
                  since low pairwise cosine can still hide a low-dimensional
                  structure.
  drift           only with --compare: mean per-slot cosine between two cartridges,
                  e.g. a CM-A.2 backup vs its CM-B.0b re-distill. Shows how far
                  training moved each slot.

No model load, no MPS, safe to run while a distill occupies the GPU.

Usage:
    python scripts/cm_probe_slot_structure.py <cartridge.safetensors> [...]
    python scripts/cm_probe_slot_structure.py new.safetensors --compare old.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def _slots(tensors: dict[str, torch.Tensor], kind: str) -> list[torch.Tensor]:
    """Return per-layer (P, n_kv_heads*head_dim) matrices for kind in {'k','v'}.

    KVPrefixCartridge.save writes flat keys "k_<layer>" / "v_<layer>", each
    (1, n_kv_heads, P, head_dim). Heads are folded into the feature axis so each
    of the P slots becomes one vector.
    """
    out: list[torch.Tensor] = []
    idx = 0
    while f"{kind}_{idx}" in tensors:
        t = tensors[f"{kind}_{idx}"]          # (1, n_kv_heads, P, head_dim)
        t = t.squeeze(0).permute(1, 0, 2)     # (P, n_kv_heads, head_dim)
        out.append(t.reshape(t.shape[0], -1).float())
        idx += 1
    return out


def _mean_pairwise_cos(m: torch.Tensor) -> float:
    n = torch.nn.functional.normalize(m, dim=-1)
    sim = n @ n.T
    p = sim.shape[0]
    off = (sim.sum() - sim.diagonal().sum()) / (p * (p - 1))
    return float(off)


def _effective_rank(m: torch.Tensor) -> float:
    s = torch.linalg.svdvals(m)
    s = s / (s.sum() + 1e-12)
    h = -(s * (s + 1e-12).log()).sum()
    return float(h.exp())


def _report(path: Path, compare: Path | None) -> None:
    tensors = load_file(str(path))
    other = load_file(str(compare)) if compare else None

    print(f"\n=== {path.name} ===")
    for kind in ("k", "v"):
        mats = _slots(tensors, kind)
        if not mats:
            print(f"  {kind}: no layers found (keys look like: {sorted(tensors)[:3]})")
            continue
        p = mats[0].shape[0]
        cos = [_mean_pairwise_cos(m) for m in mats]
        rank = [_effective_rank(m) for m in mats]
        print(f"  {kind}: {len(mats)} layers, P={p} slots, dim={mats[0].shape[1]}")
        print(f"     slot_cos_mean  min {min(cos):+.3f}  mean {sum(cos)/len(cos):+.3f}  max {max(cos):+.3f}")
        print(f"     eff_rank       min {min(rank):6.1f}  mean {sum(rank)/len(rank):6.1f}  max {max(rank):6.1f}   (of {p})")

        if other is not None:
            om = _slots(other, kind)
            if len(om) == len(mats):
                drift = []
                for a, b in zip(mats, om):
                    na = torch.nn.functional.normalize(a, dim=-1)
                    nb = torch.nn.functional.normalize(b, dim=-1)
                    drift.append(float((na * nb).sum(-1).mean()))
                print(f"     drift vs {compare.name}: mean per-slot cos "
                      f"{sum(drift)/len(drift):+.3f} (1.0 = unchanged)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cartridges", nargs="+", type=Path)
    ap.add_argument("--compare", type=Path, default=None,
                    help="second cartridge to measure per-slot drift against")
    args = ap.parse_args()

    for c in args.cartridges:
        if not c.exists():
            print(f"missing: {c}", file=sys.stderr)
            continue
        _report(c, args.compare)

    print("\ninterpretation:")
    print("  slot_cos_mean near 1.0 with eff_rank near 1  -> collapsed; slot-localized")
    print("     repair is not viable, fall back to continue-training / low-rank delta.")
    print("  low slot_cos_mean with eff_rank >> 1         -> differentiated; localization")
    print("     is worth testing against source positions once the GPU is free.")


if __name__ == "__main__":
    main()
