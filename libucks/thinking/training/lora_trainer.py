"""LoRAReceiverTrainer — fine-tunes the Base receiver model for latent injection.

Implements Interlat-Lite training (Phase 12.6 / 12.7):
  L_total = L_task - λ_sep * L_sep

where:
  L_task: cross-entropy (teacher forcing) on target text given injected latents
  L_sep:  JSD(logits_correct, logits_wrong) — positive reward for using the latent

LoRA is applied only to q_proj and v_proj attention layers, keeping the model
compact and avoiding catastrophic forgetting of language priors.

For production use with HuggingFace models, set use_peft=True (requires peft>=0.18).
For unit tests with non-standard model architectures, the built-in lightweight
LoRALinear injector is used automatically.

Phase 12.7 changes:
  - Two sequential forward passes replace the batched (2, S, D) call.
    Wrong path runs under no_grad — halves peak activation memory on MPS.
  - L_sep gradient flows through correct-path logits only (wrong logits detached).
  - accumulate_step() supports gradient accumulation over multiple buckets.
  - Default lora_r=4, lora_alpha=8, lr=2e-4 (conservative for 47-sample datasets).
  - Scalars extracted before backward(); graph refs deleted immediately after.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from libucks.thinking.training.losses import separation_loss

try:
    from mps_bitsandbytes import Linear4bit as _Linear4bit
except ImportError:
    _Linear4bit = None  # type: ignore[assignment,misc]

# Type tuple used in isinstance checks — includes Linear4bit when available.
_LINEAR_TYPES: tuple = (nn.Linear,) if _Linear4bit is None else (nn.Linear, _Linear4bit)


# ── Lightweight manual LoRA ──────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank delta.

    W_effective = W_base + (lora_B @ lora_A) * scaling
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = linear
        d_in = linear.in_features
        d_out = linear.out_features
        # Freeze the pre-trained base weights
        for p in self.base.parameters():
            p.requires_grad_(False)
        # LoRA matrices — float32 regardless of model dtype so AdamW state stays
        # float32. Float16 exp_avg_sq overflows (max ~65504) when gradients are
        # large on an untrained model, corrupting the model after epoch 1.
        # In forward(), matrices are cast to x.dtype so the computation stays
        # float16 (fast), but gradients flow back as float32 to the params.
        base_device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, d_in, device=base_device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=base_device))
        self.lora_scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.to(x.dtype).T) @ self.lora_B.to(x.dtype).T
        return base_out + lora_out * self.lora_scaling


def _inject_lora_into_module_dict(
    module: nn.ModuleDict, targets: tuple[str, ...], r: int, alpha: float
) -> None:
    """Recursively wrap target Linear layers inside a ModuleDict."""
    for key in list(module.keys()):
        child = module[key]
        if key in targets and isinstance(child, _LINEAR_TYPES):
            module[key] = LoRALinear(child, r, alpha)
        elif isinstance(child, (nn.ModuleDict, nn.ModuleList, nn.Module)):
            _inject_lora(child, targets, r, alpha)


def _inject_lora(module: nn.Module, targets: tuple[str, ...], r: int, alpha: float) -> None:
    """Walk the full module tree and replace target nn.Linear layers with LoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ModuleDict):
            for key in list(child.keys()):
                sub = child[key]
                if key in targets and isinstance(sub, _LINEAR_TYPES):
                    child[key] = LoRALinear(sub, r, alpha)
                else:
                    _inject_lora(sub, targets, r, alpha)
        elif name in targets and isinstance(child, _LINEAR_TYPES):
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            _inject_lora(child, targets, r, alpha)


# ── Trainer ──────────────────────────────────────────────────────────────────

_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_SEP_LAMBDA = 0.1   # weight for separation loss term


class LoRAReceiverTrainer:
    """Fine-tunes a Base model to interpret framed latent injections.

    Args:
        model:      Base causal LM (nn.Module).  Modified in-place.
        lora_r:     LoRA rank (default 4 — conservative for ≤100 samples).
        lora_alpha: LoRA scaling factor (default 8.0).
        lr:         AdamW learning rate (default 2e-4 — standard LoRA range).
    """

    def __init__(
        self,
        model: nn.Module,
        lora_r: int = 4,
        lora_alpha: float = 4.0,
        lr: float = 2e-4,
        warmup_steps: int = 0,
    ) -> None:
        _inject_lora(model, _LORA_TARGETS, lora_r, lora_alpha)
        self.model = model

        # Freeze everything except LoRA params — embed_tokens and lm_head stay
        # frozen (token recycling: <|im_start|>/<|im_end|> are used as frame
        # boundaries so no new embeddings are needed).
        for name, param in self.model.named_parameters():
            param.requires_grad_("lora_" in name)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        # eps=1e-6 (vs default 1e-8): extra margin against near-zero variance in fp32.
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, eps=1e-6)
        # Linear warmup: ramp lr from 10% → 100% over warmup_steps optimizer steps.
        # start_factor=1.0 when warmup_steps=0 keeps the scheduler a no-op.
        _warmup = max(1, warmup_steps)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1 if warmup_steps > 0 else 1.0,
            end_factor=1.0,
            total_iters=_warmup,
        )
        # Tracks position within current gradient-accumulation cycle.
        self._accum_count = 0

    def _forward_and_losses(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two sequential forward passes and return (task_loss, sep_loss, total).

        Correct path runs with full gradients. Wrong path runs under no_grad so
        its activations are never added to the computation graph — this halves
        peak activation memory vs the previous batched (2, S, D) approach while
        preserving the L_sep signal (gradient still flows through correct-path
        logits toward the wrong-path distribution, just not back through the
        wrong-path LoRA weights).
        """
        inputs_embeds: torch.Tensor = batch["inputs_embeds"]       # (S, D)
        target_ids: torch.Tensor = batch["target_ids"].long()      # (T,)
        prefix_len: int = int(batch["prefix_len"])                 # K+2

        inputs_embeds_wrong: torch.Tensor = batch.get(
            "inputs_embeds_wrong",
            torch.zeros_like(inputs_embeds),
        )

        # ── Correct path — full gradient graph ────────────────────────────── #
        out_correct = self.model(inputs_embeds=inputs_embeds.unsqueeze(0))
        logits_all = out_correct.logits.squeeze(0)   # (S, V)
        del out_correct  # release 36-layer hidden_states immediately

        # ── Wrong path — NO gradient (halves activation memory) ───────────── #
        with torch.no_grad():
            out_wrong = self.model(inputs_embeds=inputs_embeds_wrong.unsqueeze(0))
            logits_all_wrong = out_wrong.logits.squeeze(0).detach()  # (S, V)
            del out_wrong

        # ── Cross-entropy task loss ───────────────────────────────────────── #
        T = target_ids.shape[0]
        logits_tgt       = logits_all[prefix_len - 1: prefix_len - 1 + T]        # (T, V)
        logits_wrong_tgt = logits_all_wrong[prefix_len - 1: prefix_len - 1 + T]  # (T, V)

        task_loss = F.cross_entropy(logits_tgt.float(), target_ids)

        # ── L_sep: separation loss ────────────────────────────────────────── #
        # Gradient flows through correct-path logits only (wrong_tgt is detached).
        sep = separation_loss(logits_tgt, logits_wrong_tgt)

        total = task_loss - _SEP_LAMBDA * sep
        return task_loss, sep, total

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run one full gradient step (forward + backward + optimizer update).

        Args:
            batch: Dict with keys:
                'inputs_embeds'       (S, D) — framed prefix + target token embeds
                'inputs_embeds_wrong' (S, D) — wrong-latent prefix + same target embeds
                'target_ids'          (T,)   — integer token IDs for CE target
                'prefix_len'          int    — K+2 (number of framed-prefix positions)

        Returns:
            Dict with 'task' and 'sep' loss values (floats).
        """
        self.model.train()
        self.optimizer.zero_grad()
        self._accum_count = 0

        task_loss, sep, total = self._forward_and_losses(batch)
        task_val, sep_val = task_loss.item(), sep.item()  # extract before backward

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        del task_loss, sep, total  # break computation graph refs

        return {"task": task_val, "sep": sep_val}

    def accumulate_step(
        self, batch: Dict[str, Any], scale: int = 1, step: bool = True
    ) -> Dict[str, float]:
        """Gradient-accumulation variant of train_step.

        Divides the loss by `scale` before backward so that accumulating
        `scale` calls is equivalent to one full-batch gradient step.
        Only calls optimizer.step() (and resets the accumulation counter)
        when `step=True`.

        Args:
            batch:  Same dict format as train_step.
            scale:  Number of accumulation steps in this cycle (loss divisor).
            step:   If True, clip gradients and advance the optimizer.

        Returns:
            Dict with 'task' and 'sep' loss values (floats, unscaled for logging).
        """
        self.model.train()

        # Zero gradients at the start of each accumulation cycle.
        if self._accum_count == 0:
            self.optimizer.zero_grad()
        self._accum_count += 1

        task_loss, sep, total = self._forward_and_losses(batch)
        task_val, sep_val = task_loss.item(), sep.item()  # extract before backward

        (total / scale).backward()
        del task_loss, sep, total  # break computation graph refs

        if step:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            self._accum_count = 0

        return {"task": task_val, "sep": sep_val}
