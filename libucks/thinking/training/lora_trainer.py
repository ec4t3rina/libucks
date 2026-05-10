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

from libucks.thinking.training.losses import separation_loss, alignment_loss, margin_separation_loss

try:
    from mps_bitsandbytes import Linear4bit as _Linear4bit
except ImportError:
    _Linear4bit = None  # type: ignore[assignment,misc]

# Type tuple used in isinstance checks — includes Linear4bit when available.
_LINEAR_TYPES: tuple = (nn.Linear,) if _Linear4bit is None else (nn.Linear, _Linear4bit)


# ── Lightweight manual LoRA ──────────────────────────────────────────────────

class _LoRADeltaFn(torch.autograd.Function):
    """MPS-safe LoRA delta: float32 compute, native-dtype output, correct gradients.

    Root cause: on MPS, float32→float16 downcasts (.to(dtype)) do NOT register
    ToCopyBackward, silently dropping grad_fn and severing lora_A/lora_B from the
    computation graph. Float16→float32 upcasts (.float()) DO work (ToCopyBackward0).
    Any approach that ends with a f32→f16 downcast — whether in LoRALinear.forward
    or in downstream attention ops — kills the grad chain.

    Fix: custom autograd.Function that computes in float32 but manually manages
    backward. The forward returns x.dtype (float16) — correct values, correct shape —
    and the backward provides exact gradient formulas without relying on MPS autograd.
    """

    @staticmethod
    def forward(ctx, x, lora_A, lora_B, scaling):  # type: ignore[override]
        x_f32 = x.float()
        h = x_f32 @ lora_A.t()              # (S, r)  float32
        out = (h @ lora_B.t()) * scaling    # (S, d_out) float32
        ctx.save_for_backward(x_f32, lora_A, lora_B)
        ctx.scaling = scaling
        return out.to(x.dtype)              # float16 — no autograd; custom backward below

    @staticmethod
    def backward(ctx, grad_output):         # type: ignore[override]
        x_f32, lora_A, lora_B = ctx.saved_tensors
        g = grad_output.float()             # float16 → float32 for numerics
        sc = ctx.scaling

        # out = (x @ lora_A.T) @ lora_B.T * sc
        # x may be 3D (B, S, d_in) in attention; flatten all batch dims for grad formulas.
        orig_shape = x_f32.shape
        d_in = lora_A.shape[1]
        d_out = lora_B.shape[0]

        x_2d = x_f32.reshape(-1, d_in)    # (N, d_in)
        g_2d = g.reshape(-1, d_out)        # (N, d_out)
        h_2d = x_2d @ lora_A.t()          # (N, r) — recompute; cheaper than saving

        # ∂L/∂lora_B = (∂L/∂out).T @ h * sc
        grad_lora_B = g_2d.t() @ h_2d * sc      # (d_out, r)

        # ∂L/∂h = ∂L/∂out @ lora_B * sc
        grad_h_2d = g_2d @ lora_B * sc           # (N, r)

        # ∂L/∂lora_A = (∂L/∂h).T @ x
        grad_lora_A = grad_h_2d.t() @ x_2d      # (r, d_in)

        # ∂L/∂x = ∂L/∂h @ lora_A
        grad_x = (grad_h_2d @ lora_A).reshape(orig_shape).to(grad_output.dtype)

        return grad_x, grad_lora_A, grad_lora_B, None


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank delta.

    W_effective = W_base + (lora_B @ lora_A) * scaling
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float,
                 device: "torch.device | None" = None) -> None:
        super().__init__()
        self.base = linear
        d_in = linear.in_features
        d_out = linear.out_features
        # Freeze the pre-trained base weights
        for p in self.base.parameters():
            p.requires_grad_(False)
        # LoRA matrices stay float32 so AdamW state stays float32.
        # Float16 exp_avg_sq overflows (~65504 max) on an untrained model.
        # Device is passed explicitly from LoRAReceiverTrainer so that
        # quantized layers (Linear4bit) whose weight buffer may live on CPU
        # still get lora_A/lora_B on the model's MPS computation device.
        if device is None:
            device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, d_in, device=device) * 0.01)
        # Non-zero lora_B init: lora_B=0 causes ∂loss/∂lora_A = (lora_B^T @ g) @ x^T = 0
        # at step 1, so lora_A never receives Q=0 (soft_prompt) gradient until lora_B has
        # grown from Q=1 (query token) updates — by then soft_prompts are in lora_B's null
        # space and lora_A stays frozen for Q=0 inputs indefinitely.
        # Scale 0.001 keeps the initial LoRA perturbation ~1e-5 × input norm (negligible),
        # while ensuring both parameters get non-zero gradient from the very first step.
        self.lora_B = nn.Parameter(torch.randn(d_out, r, device=device) * 0.001)
        self.lora_scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_delta = _LoRADeltaFn.apply(x, self.lora_A, self.lora_B, self.lora_scaling)
        return base_out + lora_delta


def _inject_lora_into_module_dict(
    module: nn.ModuleDict, targets: tuple[str, ...], r: int, alpha: float,
    device: "torch.device | None" = None,
) -> None:
    """Recursively wrap target Linear layers inside a ModuleDict."""
    for key in list(module.keys()):
        child = module[key]
        if key in targets and isinstance(child, _LINEAR_TYPES):
            module[key] = LoRALinear(child, r, alpha, device=device)
        elif isinstance(child, (nn.ModuleDict, nn.ModuleList, nn.Module)):
            _inject_lora(child, targets, r, alpha, device=device)


def _inject_lora(module: nn.Module, targets: tuple[str, ...], r: int, alpha: float,
                 device: "torch.device | None" = None) -> None:
    """Walk the full module tree and replace target nn.Linear layers with LoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ModuleDict):
            for key in list(child.keys()):
                sub = child[key]
                if key in targets and isinstance(sub, _LINEAR_TYPES):
                    child[key] = LoRALinear(sub, r, alpha, device=device)
                else:
                    _inject_lora(sub, targets, r, alpha, device=device)
        elif name in targets and isinstance(child, _LINEAR_TYPES):
            setattr(module, name, LoRALinear(child, r, alpha, device=device))
        else:
            _inject_lora(child, targets, r, alpha, device=device)


# ── Trainer ──────────────────────────────────────────────────────────────────

# o_proj is required for gradient flow on MPS with mps_bitsandbytes quantization:
# 4-bit layers return requires_grad=False outputs (no autograd through input).
# q_proj/v_proj lora_delta IS in the attention computation (attn_out.rg=True),
# but o_proj (4-bit) BLOCKS it from the residual.  Adding o_proj LoRA bridges
# the gap: lora_delta_o(attn_out) is added directly to the residual, connecting
# loss → h → lora_delta_o → attn_out → q → lora_delta_q → lora_A_q.
_LORA_TARGETS = ("q_proj", "v_proj", "o_proj")
# λ=1.0 with margin loss: sep starts at ~2.0 nats when model ignores latent,
# contributing 2.0 / (task_q0 ≈ 8.7) ≈ 23% of the gradient magnitude.
# Unlike JSD (which collapses to 0 when model ignores latent), margin loss
# always has a gradient, so λ=1.0 is safe from the chicken-and-egg failure.
_SEP_LAMBDA   = 0.1
_ALIGN_LAMBDA = 0.05


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
        total_steps: int = 0,
    ) -> None:
        # Determine the model's computation device from non-quantized parameters
        # (embed_tokens, layernorms, lm_head) before LoRA injection.  For 4-bit
        # models, quantized linear weights are uint8 buffers that may report a
        # different device than the one used for float activations.
        _pre_params = list(model.parameters())
        lora_device = _pre_params[0].device if _pre_params else torch.device("cpu")
        _inject_lora(model, _LORA_TARGETS, lora_r, lora_alpha, device=lora_device)
        self.model = model

        # Freeze everything first, then explicitly re-enable only LoRA params.
        # Using isinstance(LoRALinear) instead of "lora_" string matching avoids
        # silent failure when HuggingFace renames internal modules — name-based
        # matching would freeze everything and produce no grad_fn on any forward
        # pass, causing "does not require grad" on every .backward() call.
        for param in self.model.parameters():
            param.requires_grad_(False)

        trainable: list[nn.Parameter] = []
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
                trainable.extend([module.lora_A, module.lora_B])

        if not trainable:
            raise RuntimeError(
                "LoRA injection produced zero trainable parameters. "
                "Check that the base model has q_proj / v_proj nn.Linear layers "
                "and that _inject_lora is traversing the full module tree."
            )
        # eps=1e-6 (vs default 1e-8): extra margin against near-zero variance in fp32.
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, eps=1e-6)
        # Warmup → cosine decay: ramp lr from 10% → 100% over warmup_steps, then
        # cosine-anneal to 1% over the remainder. The flat post-warmup schedule
        # (previous LinearLR) kept lr fully open through all epochs, causing
        # late-epoch divergence once the model had memorised the training set.
        _decay_steps = max(1, total_steps - warmup_steps) if total_steps > warmup_steps else 1
        _warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1 if warmup_steps > 0 else 1.0,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        _cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=_decay_steps,
            eta_min=lr * 0.1,   # floor at 10% of peak, not 1% — avoids lr→0 stagnation
        )
        if warmup_steps > 0 and total_steps > warmup_steps:
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[_warmup_sched, _cosine_sched],
                milestones=[warmup_steps],
            )
        else:
            # No total_steps provided or warmup covers everything — fall back to
            # warmup-only (safe no-op for unit tests with synthetic models).
            self.scheduler = _warmup_sched
        # Tracks position within current gradient-accumulation cycle.
        self._accum_count = 0

    def _forward_and_losses(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Three sequential forward passes returning (task_loss, sep_loss, align_loss, total).

        Correct path — full gradients.
        Wrong path   — no_grad (halves peak activation memory).
        Plan path    — no_grad; query+target only, no latent frame.  Provides the
                       L_align anchor: KL(p_correct || p_plan) penalises drift to
                       idiosyncratic tokens that would otherwise exploit L_sep.
                       Skipped when 'inputs_embeds_plan' is absent from batch (query dropped).
        """
        inputs_embeds: torch.Tensor = batch["inputs_embeds"]       # (S, D)
        target_ids: torch.Tensor = batch["target_ids"].long()      # (T,)
        prefix_len: int = int(batch["prefix_len"])                 # K+2+Q

        # On MPS, F.linear only creates grad_fn when the INPUT (not just the
        # weight) requires grad.  inputs_embeds is built from no_grad embedding
        # lookups, so it has requires_grad=False.  Forcing it to a grad-tracked
        # leaf ensures x.requires_grad=True at every transformer layer, making
        # the LoRA computation (F.linear(x.float(), lora_A)) produce grad_fn
        # and allowing loss.backward() to compute lora_A/lora_B gradients.
        # The optimizer only updates lora_A/lora_B; inputs_embeds.grad is discarded.
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        # Plan path inputs — present only when query dropout kept the query (Q > 0).
        inputs_embeds_plan: torch.Tensor | None = batch.get("inputs_embeds_plan", None)
        plan_prefix_len: int = int(batch.get("plan_prefix_len", 0))
        has_plan = inputs_embeds_plan is not None and plan_prefix_len > 0

        if target_ids.shape[0] == 0:
            raise ValueError("target_ids is empty — skip this bucket")

        # use_cache=False: transformers ≥4.46 creates a DynamicCache(config=…)
        # inside the forward when use_cache=True (the default).  In train() mode
        # the cache's get_seq_length() can return a stale value that makes
        # cache_position = arange(S, 2S) — an S-length range starting at S — so
        # the model processes 0 "new" tokens, producing hidden_states of shape
        # (1, 0, D).  Qwen2Attention then does .view(*shape[:-1], -1, head_dim)
        # = .view(1, 0, -1, 64) on a 0-element tensor, which fails because -1 is
        # ambiguous.  Passing use_cache=False bypasses the cache entirely.

        # ── Q=0 wrong path FIRST — no_grad, no saved activations ─────────── #
        # On MPS, running a second forward through the same model after the first
        # forward's saved-for-backward tensors have been released (via del out_correct)
        # can cause the GPU allocator to reuse those activation buffers — silently
        # zeroing all LoRA gradients.  Running the no_grad wrong path FIRST means
        # the correct path's backward buffers are never at risk.
        logits_wrong_tgt: torch.Tensor | None = None
        if not has_plan and _SEP_LAMBDA != 0.0:
            inputs_embeds_wrong: torch.Tensor = batch.get(
                "inputs_embeds_wrong", torch.zeros_like(inputs_embeds)
            )
            with torch.no_grad():
                out_wrong = self.model(inputs_embeds=inputs_embeds_wrong.unsqueeze(0),
                                       use_cache=False)
                logits_all_wrong = out_wrong.logits.squeeze(0).detach()  # (S, V)
                del out_wrong
            # MPS: flush all pending GPU ops from the wrong-path before the
            # correct-path forward allocates its saved-for-backward buffers.
            if inputs_embeds.device.type == "mps":
                torch.mps.synchronize()

        # ── Correct path — full gradient graph ────────────────────────────── #
        out_correct = self.model(inputs_embeds=inputs_embeds.unsqueeze(0),
                                 use_cache=False)
        logits_all = out_correct.logits.squeeze(0)   # (S, V)
        del out_correct  # release 36-layer hidden_states immediately

        # ── Cross-entropy task loss ───────────────────────────────────────── #
        T = target_ids.shape[0]
        logits_tgt = logits_all[prefix_len - 1: prefix_len - 1 + T]  # (T, V)

        task_loss = F.cross_entropy(logits_tgt.float(), target_ids)

        if not has_plan and _SEP_LAMBDA != 0.0:
            # ── Q=0 step: latent is the model's only context ──────────────── #
            logits_wrong_tgt = logits_all_wrong[prefix_len - 1: prefix_len - 1 + T]
            # Margin ranking loss for gradient: ReLU(margin - (c - w)).
            # Gradient is non-zero even when model ignores latent — avoids
            # JSD chicken-and-egg collapse to sep=0.0000.
            _sep_loss = separation_loss(logits_tgt, logits_wrong_tgt)
            # Logged sep = actual gap (correct_logit - wrong_logit for y[0]).
            # Starts near 0 when model ignores latent; rises above 2.0 when margin satisfied.
            _y0 = target_ids[0].long()
            sep = (logits_tgt[0, _y0].float() - logits_wrong_tgt[0, _y0].float().detach()).detach()
            align = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
            total = task_loss + _SEP_LAMBDA * _sep_loss
        else:
            # ── Q>0 step: query present, task loss only ───────────────────── #
            # When the query is visible, L_sep at position 0 fights CE whenever
            # the hardest-negative bucket shares the answer's first token.
            # Task signal alone is sufficient; the latent is trained implicitly
            # via the Q=0 steps that fire on the other 50% of samples.
            sep   = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
            align = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
            total = task_loss

        return task_loss, sep, align, total

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

        # inference_mode(False) is required: torch.enable_grad() cannot override
        # an active torch.inference_mode() context — they use separate C++ flags.
        # inference_mode(False) is a no-op when inference_mode is already off,
        # so this guard is safe to apply unconditionally.
        with torch.inference_mode(False), torch.enable_grad():
            task_loss, sep, align, total = self._forward_and_losses(batch)
            task_val, sep_val, align_val = task_loss.item(), sep.item(), align.item()
            total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        del task_loss, sep, align, total

        return {"task": task_val, "sep": sep_val, "align": align_val}

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

        with torch.inference_mode(False), torch.enable_grad():
            task_loss, sep, align, total = self._forward_and_losses(batch)
            task_val, sep_val, align_val = task_loss.item(), sep.item(), align.item()
            (total / scale).backward()
        del task_loss, sep, align, total

        if step:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            self._accum_count = 0

        return {"task": task_val, "sep": sep_val, "align": align_val}
