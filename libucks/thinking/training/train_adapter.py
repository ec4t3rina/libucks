"""ContrastiveAdapterTrainer — InfoNCE loss with MPS/CUDA mixed-precision.

The CommunicationAdapter is trained to maximize cosine similarity between its
output and the correct "target latent" from the V1 teacher, while minimizing
similarity to "hard negative" latents from topically-adjacent buckets.

Loss function (InfoNCE):
    L = -log( exp(sim(a, p) / τ) / Σ exp(sim(a, k) / τ) )

where a = pool(adapter(positive_latents)), p = pool(target_latent),
and the denominator sums over [p] + all hard negatives.

Mixed-precision strategy:
  CUDA  — autocast float16 + GradScaler (prevents gradient underflow)
  MPS   — autocast float16 (no GradScaler; Apple MLIR backend is stable)
  CPU   — float32, no autocast (bfloat16 not beneficial for small adapters)

Small-repo MSE fallback:
  When fewer than `min_negatives` hard negatives are available (e.g. a
  single-bucket repository), InfoNCE degenerates — the only "negative" would
  be a random unit-vector, producing a flat gradient signal.  In that case
  `contrastive_loss()` delegates to `_mse_distillation_loss()`, which
  minimises MSE between the mean-pooled adapter output and the mean-pooled
  teacher target in latent space.  Unlike InfoNCE, MSE retains magnitude
  signal and converges reliably without negatives.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from libucks.thinking.communication_adapter import CommunicationAdapter
from libucks.thinking.training.data_generator import TrainingSample


class ContrastiveAdapterTrainer:
    """Trains CommunicationAdapter with InfoNCE contrastive loss.

    Args:
        adapter:     The CommunicationAdapter instance to optimise.
        temperature: InfoNCE temperature τ — lower = sharper distribution.
        lr:          AdamW learning rate.
        device:      'auto' | 'cpu' | 'cuda' | 'mps'.
    """

    def __init__(
        self,
        adapter: CommunicationAdapter,
        temperature: float = 0.07,
        lr: float = 1e-4,
        device: str = "auto",
        min_negatives: int = 1,
    ) -> None:
        self.adapter = adapter
        self.temperature = temperature
        self.min_negatives = min_negatives
        self.device = self._resolve_device(device)
        self.adapter.to(self.device)
        self.optimizer = AdamW(adapter.parameters(), lr=lr)
        self._scaler = self._make_scaler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contrastive_loss(
        self,
        anchor: torch.Tensor,       # (K, d) — adapter output
        target: torch.Tensor,       # (seq_len_t, d) — positive
        negatives: List[torch.Tensor],  # each (seq_len_n, d)
    ) -> torch.Tensor:
        """InfoNCE loss between the adapter's output and the teacher's target.

        All tensors are mean-pooled to (d,) and L2-normalised before the
        similarity computation so the loss is purely directional.
        """
        a = self._pool(anchor.to(self.device))
        p = self._pool(target.to(self.device).clone().detach())

        pos_sim = torch.dot(a, p) / self.temperature

        # If fewer than min_negatives hard negatives were mined (e.g. a tiny
        # repo with only 1-2 buckets), InfoNCE degenerates — fall back to a
        # direct MSE alignment between adapter output and teacher target.
        if len(negatives) < self.min_negatives:
            return self._mse_distillation_loss(anchor, target)
        effective_negatives = negatives

        neg_sims = torch.stack([
            torch.dot(a, self._pool(neg.to(self.device).clone().detach())) / self.temperature
            for neg in effective_negatives
        ])

        # log-sum-exp over [positive, negatives]
        all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        return -pos_sim + torch.logsumexp(all_sims, dim=0)

    def train_step(self, sample: TrainingSample) -> float:
        """Single optimisation step. Returns the scalar loss value."""
        self.optimizer.zero_grad()

        # .clone().detach() is required — not just .detach().
        # Tensors from LatentStrategy are produced under torch.inference_mode(),
        # which permanently disables their version counters. A bare .detach()
        # keeps the inference-mode flag on the storage, so autograd still raises:
        #   "RuntimeError: Inference tensors cannot be saved for backward."
        # .clone() allocates fresh storage outside inference_mode; .detach()
        # then makes it a leaf node. The adapter's own parameters remain in the
        # graph — we deliberately do not propagate into the frozen Qwen backbone.
        latents = [t.to(self.device).clone().detach() for t in sample.librarian_latents]

        with self._autocast_ctx():
            output: torch.Tensor = self.adapter(latents)
            loss = self.contrastive_loss(
                output, sample.target_latent, sample.hard_negatives
            )

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.device == "mps":
            torch.mps.empty_cache()

        return loss.item()

    def train(
        self,
        samples: List[TrainingSample],
        num_epochs: int = 1,
    ) -> List[float]:
        """Train over a list of samples for num_epochs. Returns per-step losses."""
        losses: list[float] = []
        for _ in range(num_epochs):
            for sample in samples:
                losses.append(self.train_step(sample))
        return losses

    def train_compressor_step(
        self,
        compressor: "LatentCompressor",
        compressor_optimizer: torch.optim.Optimizer,
        full_latents: List[torch.Tensor],
    ) -> float:
        """Single optimisation step for the LatentCompressor.

        Objective: MSE(Adapter(compressed), Adapter(full)).
        The adapter is treated as a frozen reference — only compressor
        parameters are updated.

        Args:
            compressor:           The LatentCompressor to train.
            compressor_optimizer: Optimizer bound to compressor.parameters().
            full_latents:         List of full-length Librarian tensors.

        Returns:
            Scalar loss value (float).
        """
        compressor.to(self.device)
        compressor_optimizer.zero_grad()

        # Frozen reference: adapter output on full (uncompressed) latents
        with torch.no_grad():
            full_inputs = [t.to(self.device) for t in full_latents]
            full_out = self.adapter(full_inputs)  # (K, d)

        # Compressed path: pass each latent through the compressor first
        compressed_inputs = [compressor(t.to(self.device)) for t in full_latents]
        compressed_out = self.adapter(compressed_inputs)  # (K, d)

        loss = F.mse_loss(compressed_out, full_out.detach())
        loss.backward()
        compressor_optimizer.step()

        return loss.item()

    def save(self, path: Path) -> None:
        """Save adapter state_dict to path."""
        torch.save(self.adapter.state_dict(), path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mse_distillation_loss(
        self,
        anchor: torch.Tensor,  # (K, d) — adapter output
        target: torch.Tensor,  # (seq_len, d) — teacher target
    ) -> torch.Tensor:
        """Per-token, scale-invariant MSE fallback for small-repo training.

        Two design choices motivated by observed failure modes:

        1.  Adaptive pool teacher (seq_len, d) → (K, d) instead of mean-pooling
            both tensors to (d,).  Mean-pooling collapses K distinct output
            positions into one vector, diluting gradients by 1/K (= 1/32) and
            driving all K output queries toward the same target — causing Qwen to
            see K identical soft-prompt tokens and fall into repetitive generation.
            Per-token targets give each output query a distinct gradient signal.

        2.  L2-normalise both tensors before MSE.  mse(norm(a), norm(t)) equals
            2*(1 − cosine_similarity) — scale-invariant and purely directional.
            Qwen's cross-attention cares about the direction of soft-prompt key/
            value vectors, not their absolute scale, so this matches the
            downstream objective.  It also prevents the optimizer from spending
            its budget correcting magnitude mismatches between Qwen's hidden-state
            scale (~10–50/dim) and the adapter's random-init scale (~1–5/dim).

        .clone().detach() on the target preserves the inference_mode autograd
        contract (same as contrastive_loss).
        """
        K = anchor.shape[0]
        a = anchor.to(self.device)                          # (K, d)
        t = target.to(self.device).clone().detach()         # (seq_len, d)

        # Resample teacher from seq_len → K via linear interpolation.
        # F.interpolate(mode='linear') has no divisibility constraint.
        # F.adaptive_avg_pool1d would crash on MPS when seq_len % K != 0:
        #   "RuntimeError: Adaptive pool MPS: input sizes must be divisible
        #    by output sizes."
        # Both ops expect (N, C, L); we treat d as channels.
        t_K = F.interpolate(
            t.T.unsqueeze(0),   # (1, d, seq_len)
            size=K,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T          # (K, d)

        # L2-normalise: mse on unit vectors = 2*(1 − cos_sim) per element.
        a_n = F.normalize(a, dim=-1)    # (K, d)
        t_n = F.normalize(t_K, dim=-1)  # (K, d)

        return F.mse_loss(a_n, t_n)

    @staticmethod
    def _pool(tensor: torch.Tensor) -> torch.Tensor:
        """Mean-pool (seq_len, d) → (d,), L2-normalised."""
        return F.normalize(tensor.mean(dim=0), dim=0)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _make_scaler(self):
        """GradScaler for CUDA only; None on MPS/CPU."""
        if self.device == "cuda":
            return torch.cuda.amp.GradScaler()
        return None

    def _autocast_ctx(self):
        """Return the appropriate autocast context for the current device."""
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.device == "mps":
            try:
                return torch.autocast(device_type="mps", dtype=torch.float16)
            except Exception:
                pass
        return contextlib.nullcontext()
