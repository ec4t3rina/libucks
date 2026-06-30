"""Diagnose CommunicationAdapter health by measuring cross-bucket cosine.

Run after Phase 1 training to detect adapter collapse (deaf adapter:
cross-bucket cos ~= 1.0). Reads cached librarian latents from
.libucks/latent_cache/ so no teacher API calls are made.

Usage:
    python scripts/diagnose_adapter.py /path/to/repo [--n-samples N]

Exit codes: 0 = healthy, 1 = degraded, 2 = collapsed.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from libucks.config import Config
from libucks.thinking.communication_adapter import CommunicationAdapter


def main(repo_path: Path, n_samples: int = 10, seed: int = 0) -> int:
    random.seed(seed)
    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"
    cache_dir = bucket_dir / "latent_cache"

    hidden_dim = AutoConfig.from_pretrained(cfg.model.local_model).hidden_size
    base_dim = AutoConfig.from_pretrained(cfg.model.base_model).hidden_size
    adapter = CommunicationAdapter(hidden_dim=hidden_dim, output_dim=base_dim,
                                   output_len=cfg.model.output_len)
    adapter.load_saved_weights(bucket_dir / "adapter.pt")
    adapter.eval()

    cached = sorted(cache_dir.glob("*.pt"))
    if len(cached) < 3:
        print(
            f"Only {len(cached)} cached latents in {cache_dir}; "
            "need >= 3 to compute cross-bucket cos."
        )
        return 3
    sampled = random.sample(cached, min(n_samples, len(cached)))

    pooled_outputs: list[tuple[str, torch.Tensor]] = []
    for p in sampled:
        latents = torch.load(p, map_location="cpu", weights_only=True)
        latents = [t.float() for t in latents]
        with torch.no_grad():
            out = adapter(latents)
        pooled = F.normalize(out.mean(dim=0), dim=-1)
        pooled_outputs.append((p.stem, pooled))

    names = [n for n, _ in pooled_outputs]
    vecs = torch.stack([v for _, v in pooled_outputs])
    cos = vecs @ vecs.T

    n = len(names)
    off_diag_mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = cos[off_diag_mask]
    mean = off_diag.mean().item()
    max_v = off_diag.max().item()
    min_v = off_diag.min().item()

    print(f"N={n} buckets  base_dim={base_dim}  K={adapter.output_len}")
    print(f"cross-bucket cos  mean={mean:.4f}  min={min_v:.4f}  max={max_v:.4f}")
    if mean > 0.85:
        print(f"\nVERDICT: COLLAPSED (mean cos {mean:.4f} > 0.85 threshold).")
        print("Retrain Phase 1 before touching --sep-lambda.")
        return 2
    if mean > 0.70:
        print(f"\nVERDICT: degraded (mean cos {mean:.4f} between 0.70 and 0.85).")
        print("Borderline -- investigate before assuming Phase 2 will work.")
        return 1
    print(f"\nVERDICT: healthy (mean cos {mean:.4f} <= 0.70).")
    print("Adapter looks fine; --sep-lambda hypothesis is defensible.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", type=Path)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(main(args.repo_path, args.n_samples, args.seed))
