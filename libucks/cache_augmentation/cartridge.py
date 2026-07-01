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

    def load(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        flat = load_file(str(path))
        with torch.no_grad():
            for i in range(self.n_layers):
                self.k[i].data.copy_(flat[f"k_{i}"].to(self.k[i].dtype))
                self.v[i].data.copy_(flat[f"v_{i}"].to(self.v[i].dtype))
