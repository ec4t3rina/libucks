"""CrossBucketFusion — combine N per-bucket coprocessor outputs into z_fused.

This is libucks's multi-agent extension to DeepMind's single-context cache
augmentation. Each bucket's `Coprocessor` produces z_b of shape (1, K=64,
hidden_dim=2048); the router picks top_k buckets (default 3); this module
fuses {z_1, ..., z_k} into a single z_fused of the same shape.

Architecture (intentionally simpler than the existing CommunicationAdapter):
  1. Stack: concat z_b tensors along token dim → (1, N*K, hidden_dim) context
  2. Output queries: K learnable slot tokens (the fusion's "what to extract")
  3. N transformer blocks: self-attn among output queries + cross-attn to the
     N*K context tokens + FFN. Each output query can attend to any token in
     any bucket — no per-bucket masking — which is the whole point of fusion.
  4. Output norm → z_fused (1, K, hidden_dim)

We do NOT warm-start from the existing adapter.pt because the input shape is
fundamentally different (the old adapter pools each Librarian rep across L_i
positions then operates on N pooled summaries; here each bucket's output is
already a K-position structured sequence whose per-slot identity is meaningful
and should NOT be collapsed). Training from scratch is the correct choice.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from libucks.cache_augmentation.coprocessor import CoprocessorBlock


class CrossBucketFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        K: int = 64,
        n_blocks: int = 2,
        n_heads: int = 8,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = K

        # Output queries — slot identities for the final K-token z_fused.
        self.output_queries = nn.Parameter(torch.randn(1, K, hidden_dim) * 0.02)

        # Reuse the coprocessor block: self-attn + cross-attn + FFN.
        self.blocks = nn.ModuleList(
            [
                CoprocessorBlock(hidden_dim, n_heads, ffn_mult)
                for _ in range(n_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, bucket_z: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bucket_z: list of N tensors, each shape (1, K, hidden_dim).
                Order is the routed-bucket order (top-k by routing score).

        Returns:
            z_fused: (1, K, hidden_dim) — the final soft prompt that will
                be appended into the receiver's KV cache at decode time.
        """
        if not bucket_z:
            raise ValueError("bucket_z must be non-empty")

        # Stack across buckets → context (1, N*K, hidden_dim).
        # Each bucket's K tokens are concatenated as separate positions; the
        # cross-attention then sees all N*K positions and can route attention
        # to whichever bucket-and-slot is most relevant per output query.
        ctx = torch.cat(bucket_z, dim=1)

        # Output queries flow through the fusion blocks attending to the
        # concatenated bucket context.
        compute_dtype = next(self.parameters()).dtype
        x = self.output_queries.to(compute_dtype)
        # Match ctx dtype (it could come from a coprocessor in fp32 or bf16).
        ctx = ctx.to(compute_dtype)

        for block in self.blocks:
            x = block(x, ctx)
        return self.norm_out(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
