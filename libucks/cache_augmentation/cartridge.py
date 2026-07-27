"""Trainable KV-prefix cartridge (Cartridge Memory, CM-A).

A per-bucket learnable prefix of key/value tensors, prefix-tuning on a frozen
receiver. This is the artifact CM-A distills — replacing the from-scratch
coprocessor (Phase 4-C, negative result) with the Cartridges recipe
(arXiv 2506.06266): a small trainable KV cache the frozen model reads as
`past_key_values`.

Per-layer parameters have the receiver's KV geometry:
    k[i], v[i]: (1, n_kv_heads, prefix_len, head_dim)
For Qwen 2.5-3B: n_layers=36, n_kv_heads=2, head_dim=128; prefix_len (P) = 64.

Only these prefix tensors are trainable; the receiver stays frozen. At decode
time `to_dynamic_cache` yields a `transformers.DynamicCache` usable as the
augmentation prefix in `decode.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class KVPrefixCartridge(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        prefix_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.prefix_len = prefix_len
        self.head_dim = head_dim
        self._dtype = dtype

        shape = (1, n_kv_heads, prefix_len, head_dim)
        self.k = nn.ParameterList(
            [nn.Parameter(torch.randn(shape, dtype=dtype) * init_std) for _ in range(n_layers)]
        )
        self.v = nn.ParameterList(
            [nn.Parameter(torch.randn(shape, dtype=dtype) * init_std) for _ in range(n_layers)]
        )

    # ------------------------------------------------------------------
    @classmethod
    def for_model(
        cls,
        model: torch.nn.Module,
        *,
        prefix_len: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> "KVPrefixCartridge":
        """Build a cartridge whose KV geometry matches `model`'s attention config."""
        cfg = model.config
        n_layers = int(cfg.num_hidden_layers)
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
        head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )
        return cls(
            n_layers=n_layers, n_kv_heads=n_kv_heads,
            prefix_len=prefix_len, head_dim=head_dim, dtype=dtype,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_from_extracted_kv(self, flat: dict[str, torch.Tensor]) -> None:
        """Warm-start the prefix from a bucket's real extracted KV (the flat
        dict produced by `kv_extract.extract_bucket_kv`).

        Copies the first `prefix_len` positions of each layer's K/V. If the
        bucket is shorter than `prefix_len`, copies what exists and leaves the
        remaining (random-init) positions as trainable slack.
        """
        for i in range(self.n_layers):
            self._copy_prefix(self.k[i], flat[f"layer_{i}_K"])
            self._copy_prefix(self.v[i], flat[f"layer_{i}_V"])

    def _copy_prefix(self, param: nn.Parameter, src: torch.Tensor) -> None:
        # src: (1, n_kv_heads, T, head_dim)
        seq_len = src.shape[2]
        n = min(seq_len, self.prefix_len)
        param.data[:, :, :n, :] = src[:, :, :n, :].to(dtype=param.dtype, device=param.device)

    # ------------------------------------------------------------------
    def to_dynamic_cache(self, device: torch.device, dtype: torch.dtype | None = None) -> Any:
        """Build a DynamicCache holding the trainable prefix (grad-preserving)."""
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i in range(self.n_layers):
            k = self.k[i].to(device=device)
            v = self.v[i].to(device=device)
            if dtype is not None:
                k = k.to(dtype=dtype)
                v = v.to(dtype=dtype)
            cache.update(k, v, layer_idx=i)
        return cache

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        flat: dict[str, torch.Tensor] = {}
        for i in range(self.n_layers):
            flat[f"k_{i}"] = self.k[i].detach().cpu().contiguous()
            flat[f"v_{i}"] = self.v[i].detach().cpu().contiguous()
        meta = {
            "n_layers": str(self.n_layers),
            "n_kv_heads": str(self.n_kv_heads),
            "prefix_len": str(self.prefix_len),
            "head_dim": str(self.head_dim),
        }
        save_file(flat, str(path), metadata=meta)

    @staticmethod
    def read_geometry(path: str | Path) -> dict[str, int]:
        """Return the geometry a cartridge file was saved with.

        `PREFIX_LEN = 128` is duplicated in cm_distill_buckets.py and
        cm_eval_cartridge.py under reciprocal "MUST match" comments — the manual
        cross-file invariant this class's own load() docstring calls "a question
        of when, not if". Callers should read the geometry off the file instead
        of restating it, which also lets one eval handle cartridges trained at
        different P.

        Raises ValueError if the file carries no geometry metadata: guessing is
        exactly the silent-corruption path load() exists to prevent.
        """
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        keys = ("n_layers", "n_kv_heads", "prefix_len", "head_dim")
        missing = [k for k in keys if k not in meta]
        if missing:
            raise ValueError(
                f"{path} has no geometry metadata ({missing} absent). It predates "
                f"metadata-writing saves; re-save it or construct the cartridge "
                f"explicitly rather than inferring its shape."
            )
        return {k: int(meta[k]) for k in keys}

    def load(self, path: str | Path) -> None:
        """Load a saved cartridge, refusing any geometry that isn't an exact match.

        The validation is not optional politeness. `Tensor.copy_` broadcasts, so a
        file saved with n_kv_heads=1 loads into a 2-head cartridge *silently*,
        duplicating head 0 across both heads and corrupting the prefix with no
        error. A file with more layers than this object silently loads a prefix of
        them. libucks runs two receiver geometries (Qwen2.5-3B: 36 layers/2 heads;
        Qwen2.5-0.5B: 24 layers) and PREFIX_LEN is duplicated across scripts under a
        "MUST match" comment, so a mismatch is a question of when, not if.

        `save` has always written the geometry as safetensors metadata; this reads
        it. Files without metadata fall back to per-tensor shape comparison.
        """
        from safetensors import safe_open
        from safetensors.torch import load_file

        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}

        expected = {
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "prefix_len": self.prefix_len,
            "head_dim": self.head_dim,
        }
        mismatches = [
            f"{field}: this cartridge is {want}, file has {meta[field]}"
            for field, want in expected.items()
            if meta.get(field) is not None and int(meta[field]) != want
        ]
        if mismatches:
            raise ValueError(
                f"cartridge geometry mismatch loading {path.name} — "
                + "; ".join(mismatches)
                + ". Refusing to load: copy_ would broadcast or partially fill "
                "and corrupt the prefix silently."
            )

        flat = load_file(str(path))

        n_in_file = sum(1 for key in flat if key.startswith("k_"))
        if n_in_file != self.n_layers:
            raise ValueError(
                f"cartridge geometry mismatch loading {path.name} — n_layers: "
                f"this cartridge is {self.n_layers}, file has {n_in_file}."
            )

        for i in range(self.n_layers):
            for kind, params in (("k", self.k), ("v", self.v)):
                src = flat.get(f"{kind}_{i}")
                if src is None:
                    raise ValueError(
                        f"cartridge file {path.name} is missing tensor {kind}_{i} "
                        f"(n_layers: this cartridge is {self.n_layers})."
                    )
                if tuple(src.shape) != tuple(params[i].shape):
                    raise ValueError(
                        f"cartridge geometry mismatch loading {path.name} — "
                        f"{kind}_{i}: this cartridge is {tuple(params[i].shape)}, "
                        f"file has {tuple(src.shape)}."
                    )

        with torch.no_grad():
            for i in range(self.n_layers):
                self.k[i].data.copy_(flat[f"k_{i}"].to(self.k[i].dtype))
                self.v[i].data.copy_(flat[f"v_{i}"].to(self.v[i].dtype))
