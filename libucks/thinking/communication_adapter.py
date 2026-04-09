"""CommunicationAdapter — neural switchboard for merging Librarian latents.

Aggregates N variable-length hidden-state tensors (one per Librarian bucket)
into a single fixed-length soft-prompt for the Translator to decode.

Architecture (three stages):
  1. Attentive Pooling  — squash each Librarian's (L_i, d) → (d,) via a
                          learned query vector attending over token positions.
  2. Inter-Librarian    — N summary vectors attend to each other, letting
     Self-Attention       cross-bucket relationships emerge.
  3. Output Projection  — K learned output queries cross-attend over the N
                          refined summaries, producing the (K, d) soft-prompt.

K (output_len) defaults to 32. d (hidden_dim) must match the backbone model.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CommunicationAdapter(nn.Module):
    """Merge N Librarian hidden-state tensors into a (output_len, hidden_dim) soft-prompt.

    Args:
        hidden_dim:  Dimension of the backbone model's hidden states (default 2048).
        output_len:  Number of soft-prompt tokens to produce (default 32).
        num_heads:   Attention heads for all internal attention layers (default 4).
        num_inter_layers: Depth of the inter-Librarian self-attention stack (default 2).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        output_len: int = 32,
        num_heads: int = 8,
        num_inter_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_len = output_len

        # --- Stage 1: Attentive Pooling ---
        # Learned query attends over each Librarian's token dimension
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # --- Stage 2: Inter-Librarian Self-Attention ---
        self.inter_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
             for _ in range(num_inter_layers)]
        )
        self.inter_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_inter_layers)]
        )

        # --- Stage 3: Output Projection ---
        # K learned output queries cross-attend over the N summaries
        self.output_queries = nn.Parameter(torch.randn(1, output_len, hidden_dim))
        self.output_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        """Merge Librarian tensors into a single soft-prompt tensor.

        Args:
            representations: List of N tensors, each shape (L_i, hidden_dim).
                             L_i may differ between Librarians.

        Returns:
            torch.Tensor of shape (output_len, hidden_dim).

        Raises:
            ValueError: If representations is empty.
        """
        if not representations:
            raise ValueError("representations must be non-empty")

        device = representations[0].device

        # ------------------------------------------------------------------ #
        # Stage 1: Attentive Pooling — (L_i, d) → (d,) per Librarian
        # ------------------------------------------------------------------ #
        summaries: List[torch.Tensor] = []
        pool_q = self.pool_query.to(device)  # (1, 1, d)

        for rep in representations:
            if rep.dim() == 2:
                rep = rep.unsqueeze(0)        # (1, L_i, d)
            # query: (1, 1, d)  key/value: (1, L_i, d)  → attn_out: (1, 1, d)
            attn_out, _ = self.pool_attn(pool_q, rep, rep)
            summaries.append(attn_out[0, 0])  # (d,)

        # Stack N summaries → (1, N, d)
        x = torch.stack(summaries, dim=0).unsqueeze(0)  # (1, N, d)

        # ------------------------------------------------------------------ #
        # Stage 2: Inter-Librarian Self-Attention — (1, N, d) → (1, N, d)
        # ------------------------------------------------------------------ #
        for attn, norm in zip(self.inter_attn_layers, self.inter_norms):
            residual = x
            attn_out, _ = attn(x, x, x)
            x = norm(residual + attn_out)

        # ------------------------------------------------------------------ #
        # Stage 3: Output Projection — (1, K, d) soft-prompt
        # ------------------------------------------------------------------ #
        out_q = self.output_queries.to(device)       # (1, K, d)
        output, _ = self.output_attn(out_q, x, x)   # (1, K, d)
        output = self.output_norm(output)

        return output.squeeze(0)   # (K, d)

    def frame(
        self,
        soft_prompt: torch.Tensor,
        bop_embed: torch.Tensor,
        eop_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Wrap a soft-prompt with <bop> and <eop> boundary embeddings.

        Produces:   [e(<bop>), h_1, ..., h_K, e(<eop>)]
        Shape:      (output_len + 2, hidden_dim)

        Args:
            soft_prompt: Adapter output, shape (output_len, hidden_dim).
            bop_embed:   Embedding vector for the <bop> special token, shape (hidden_dim,).
            eop_embed:   Embedding vector for the <eop> special token, shape (hidden_dim,).

        Returns:
            Framed tensor of shape (output_len + 2, hidden_dim).
        """
        bop = bop_embed.view(1, -1)   # (1, D)
        eop = eop_embed.view(1, -1)   # (1, D)
        return torch.cat([bop, soft_prompt, eop], dim=0)

    def load_saved_weights(self, path: "Path") -> None:
        """Load adapter weights from a checkpoint file if it exists.

        Args:
            path: Path to a .pt file saved by ContrastiveAdapterTrainer.save().
                  Silently ignored if the file does not exist.
        """
        from pathlib import Path as _Path
        p = _Path(path)
        if p.exists():
            state = torch.load(p, map_location="cpu", weights_only=True)
            self.load_state_dict(state)
