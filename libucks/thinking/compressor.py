"""LatentCompressor — bottleneck layer for hidden-state sequences.

Compresses a variable-length sequence (L, d) into K fixed steps (K, d) using
K learned query vectors that cross-attend over the full input sequence.

This is architecturally similar to the Perceiver / DETR object-query approach:
the K query vectors act as a fixed-size bottleneck, attending to the most
relevant parts of the full sequence and discarding redundant positional noise.

After training (via ContrastiveAdapterTrainer.train_compressor_step), the
compressor learns to preserve the information that matters to the downstream
CommunicationAdapter — yielding the same Translator output from K tokens as
from the full L tokens, with L/K speedup for all subsequent Librarian calls.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentCompressor(nn.Module):
    """Compress (L, d) hidden-state sequences to (K, d).

    Args:
        hidden_dim:         Hidden state dimension — must match the backbone.
        compression_steps:  K, the fixed output sequence length (default 8).
        num_heads:          Attention heads for cross-attention (default 4).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        compression_steps: int = 8,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compression_steps = compression_steps

        # K learned query vectors — the compression bottleneck
        self.query_vectors = nn.Parameter(torch.randn(1, compression_steps, hidden_dim))

        # Single cross-attention layer: queries attend over the full sequence
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress hidden states to K fixed steps.

        Args:
            x: Tensor of shape (L, hidden_dim) or (1, L, hidden_dim).
               L may vary across calls.

        Returns:
            Tensor of shape (compression_steps, hidden_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)          # (1, L, d)
        q = self.query_vectors.to(x.device)  # (1, K, d)
        out, _ = self.cross_attn(q, x, x)   # (1, K, d)
        out = self.norm(out)
        return out.squeeze(0)               # (K, d)
