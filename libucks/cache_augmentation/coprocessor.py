"""Coprocessor — reads a bucket's KV cache + learned soft tokens, emits z.

Architecture (lightweight variant of DeepMind 2412.17747):

  bucket KV cache (36 layers × 2 KV heads × T × 128, both K and V)
                          │
                          ▼
              flatten per-layer + concat K,V
              learned softmax over 36 layers
                          │
                          ▼
              project 512 → 2048 (kv_proj)
                          │
                          ▼ context tokens (T, 2048)
                          │
   (K=64, 2048) ──►  N × CoprocessorBlock  ──►  z (K=64, 2048)
   learnable soft     self-attn + cross-attn
   tokens             + FFN, pre-norm

This is a per-bucket coprocessor. Multi-bucket fusion (4-C.4) takes
N coprocessor outputs z_b and combines them into z_fused.

Defaults sized for MPS: ~150-200M params (4 blocks, 8 heads, FFN×2).
DeepMind paper used a full Qwen-3B-shaped coprocessor, which gave +3
GSM8K points over from-scratch and +5 over LoRA-128. We accept a few
points of ceiling for an order of magnitude less memory.

Output is in float32 by default for training numerical stability; cast
to receiver dtype at the boundary in decode.py (Phase 4-C.4).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class CoprocessorBlock(nn.Module):
    """Pre-norm transformer block: self-attn + cross-attn to bucket context + FFN."""

    def __init__(
        self,
        hidden_dim: int = 2048,
        n_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_xa = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, H) — soft tokens through this block
            ctx: (B, T, H) — bucket context tokens (one per source position)
        Returns:
            (B, K, H)
        """
        # Self-attn
        q = self.norm_sa(x)
        sa, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + sa
        # Cross-attn
        q = self.norm_xa(x)
        ca, _ = self.cross_attn(q, ctx, ctx, need_weights=False)
        x = x + ca
        # FFN
        q = self.norm_ffn(x)
        x = x + self.ffn(q)
        return x


class Coprocessor(nn.Module):
    """Per-bucket coprocessor: bucket KV cache + K soft tokens → z (K, H)."""

    def __init__(
        self,
        hidden_dim: int = 2048,          # matches Qwen 2.5-3B
        n_kv_heads: int = 2,             # Qwen 2.5-3B GQA
        head_dim: int = 128,             # Qwen 2.5-3B
        n_source_layers: int = 36,       # Qwen 2.5-3B
        K: int = 64,                     # soft-prompt budget (matches Phase 4-A)
        n_blocks: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        # Per-token feature dim from a single source layer: K + V flattened
        # across the 2 KV heads → 2*head_dim + 2*head_dim = 4*head_dim = 512.
        per_layer_feat = 2 * n_kv_heads * head_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.n_source_layers = n_source_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        # Learned per-layer blending of the 36 source layers.
        self.layer_blend = nn.Parameter(torch.zeros(n_source_layers))

        # Project the per-position context feature to hidden_dim.
        self.kv_proj = nn.Linear(per_layer_feat, hidden_dim)
        self.norm_ctx = nn.LayerNorm(hidden_dim)

        # Learnable soft tokens (K, hidden_dim).
        self.soft_tokens = nn.Parameter(torch.randn(K, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                CoprocessorBlock(hidden_dim, n_heads, ffn_mult)
                for _ in range(n_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, bucket_kv: dict[str, torch.Tensor]) -> torch.Tensor:
        """Turn a flat bucket-KV dict into per-position context tokens.

        Input keys: layer_<i>_K, layer_<i>_V where each tensor has shape
        (1, n_kv_heads, T, head_dim).

        Returns: (1, T, hidden_dim) context tensor on the same device as the
        coprocessor's parameters.
        """
        device = next(self.parameters()).device
        compute_dtype = next(self.parameters()).dtype

        # Stack all layers' (K, V) into (n_layers, 1, n_kv_heads, T, head_dim).
        keys = []
        values = []
        for i in range(self.n_source_layers):
            k = bucket_kv[f"layer_{i}_K"].to(device=device, dtype=compute_dtype)
            v = bucket_kv[f"layer_{i}_V"].to(device=device, dtype=compute_dtype)
            keys.append(k)
            values.append(v)
        # (L, 1, H_kv, T, D)
        K_stack = torch.stack(keys, dim=0)
        V_stack = torch.stack(values, dim=0)
        T = K_stack.shape[3]

        # Reshape to (L, 1, T, H_kv * D) for both K and V, then concat → (L, 1, T, 2*H_kv*D).
        # transpose to (L, 1, T, H_kv, D) → flatten last two dims.
        K_flat = K_stack.transpose(2, 3).reshape(self.n_source_layers, 1, T, -1)
        V_flat = V_stack.transpose(2, 3).reshape(self.n_source_layers, 1, T, -1)
        per_layer = torch.cat([K_flat, V_flat], dim=-1)  # (L, 1, T, 4*D)

        # Learned softmax blending across source layers → (1, T, 4*D).
        w = torch.softmax(self.layer_blend, dim=0).view(-1, 1, 1, 1)
        blended = (per_layer * w.to(compute_dtype)).sum(dim=0)  # (1, T, 4*D)

        # Project to hidden_dim and norm.
        ctx = self.kv_proj(blended)
        ctx = self.norm_ctx(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, bucket_kv: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bucket_kv: flat dict from kv_extract.extract_bucket_kv — one
                bucket's KV cache.
        Returns:
            z: (1, K, hidden_dim) — the bucket's latent embedding output.
        """
        ctx = self._build_context(bucket_kv)  # (1, T, H)

        # Soft tokens broadcast to batch dim 1.
        compute_dtype = next(self.parameters()).dtype
        x = self.soft_tokens.unsqueeze(0).to(compute_dtype)  # (1, K, H)

        for block in self.blocks:
            x = block(x, ctx)
        return self.norm_out(x)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
