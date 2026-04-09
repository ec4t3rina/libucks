"""CurriculumMixer — stochastic token-to-latent mixing for Interlat-Lite training.

Implements the mixing formula from Interlat §3.3:

    H^(r) = [e_1, ..., e_{floor(r·K)}] ⊕ [h_{floor(r·K)+1}, ..., h_K]
             ← token embeddings →         ←        latents          →

r is sampled uniformly from [0, 1] at each training step.
  r = 0  →  pure latents   (model sees only adapter output)
  r = 1  →  pure tokens    (model sees only plan token embeddings)

Sampling uniformly forces the receiver to handle any mixture, bridging the gap
between the token manifold (which the base model already understands) and the
latent manifold (which it must learn to interpret).
"""
import math
import torch


class CurriculumMixer:
    """Static utility — no state required."""

    @staticmethod
    def mix(
        latents: torch.Tensor,
        tok_embeds: torch.Tensor,
        r: float,
    ) -> torch.Tensor:
        """Return a mixed soft-prompt of shape (K, D).

        Args:
            latents:    Adapter output, shape (K, D).
            tok_embeds: Token embeddings for the text plan, shape (K, D).
            r:          Mixing rate in [0, 1].
                        r=0 → all latents; r=1 → all token embeddings.

        Returns:
            Mixed tensor, shape (K, D), same dtype and device as *latents*.

        Raises:
            ValueError: if r ∉ [0, 1] or shapes don't match.
        """
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"r must be in [0, 1], got {r}")
        if latents.shape != tok_embeds.shape:
            raise ValueError(
                f"Shape mismatch: latents {latents.shape} vs tok_embeds {tok_embeds.shape}"
            )

        K = latents.shape[0]
        split = math.floor(r * K)  # number of leading token-embedding rows

        if split == 0:
            return latents.clone()
        if split == K:
            return tok_embeds.clone()

        return torch.cat(
            [tok_embeds[:split].clone(), latents[split:].clone()],
            dim=0,
        )
