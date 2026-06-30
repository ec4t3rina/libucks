"""Phase 4-C.5 — Cache augmentation trainer.

Trains the Coprocessor + CrossBucketFusion on per-bucket Q&A pairs, with the
Qwen-3B receiver FROZEN throughout (per DeepMind 2412.17747 §2.2). Loss is
plain cross-entropy on the answer tokens given the augmented KV cache.

Why this is much simpler than the existing LoRA trainer:
- No LoRA injection on the receiver — frozen base.
- No L_sep margin loss — DeepMind's setup uses LM loss only and it converges.
- Single forward + backward per training example (no wrong-path no_grad pass).
- All gradient flow lives in the coproc + fusion path; the receiver only
  contributes a forward (no_grad equivalent — params have requires_grad=False).

Pipeline per training step:
    bucket_kv (frozen, precomputed)  ─►  Coprocessor  ─►  z_b
                                                            │
                       optional N-bucket fusion ◄───────────┘
                                                            │
                                                          z_fused
                                                            │
    "Q: <query>\\nA:" text  ──►  receiver(use_cache=True) ──►  C_input
                                                            │
    z_fused (as inputs_embeds) ──►  receiver(use_cache=True) ──►  C_z
                                                            │
                              concat C_input + C_z  ──►  C_aug
                                                            │
              receiver(answer_input_ids, past_kv=C_aug) ──►  logits
                                                            │
                                                          CE loss
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
from libucks.cache_augmentation.coprocessor import Coprocessor
from libucks.cache_augmentation.fusion import CrossBucketFusion


def _log(msg: str) -> None:
    print(f"[libucks:cache_aug_train] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class CacheAugTrainSample:
    bucket_id: str
    query: str
    answer: str


# The data_generator falls back to this question when the teacher API returns
# malformed output or the bucket has empty source. Pairs with this question
# carry no useful training signal — answer is just the bucket source name.
_STUB_QUESTION_PREFIX = "Explain concisely what this code does"


def load_qa_pairs(qa_cache_path: Path) -> list[CacheAugTrainSample]:
    """Read qa_cache.json (produced by the existing data_generator) into a
    flat list of training samples. Each pair is a [query, answer] list.

    Drops generic-stub pairs (see _STUB_QUESTION_PREFIX) — they're a
    data-generator fallback signal that the teacher couldn't produce a
    real Q&A for the bucket.
    """
    data = json.loads(qa_cache_path.read_text())
    samples: list[CacheAugTrainSample] = []
    stubs_dropped = 0
    for bid, entry in data.items():
        for pair in entry.get("pairs", []):
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            q, a = pair[0], pair[1]
            if not q or not a:
                continue
            if str(q).startswith(_STUB_QUESTION_PREFIX):
                stubs_dropped += 1
                continue
            samples.append(CacheAugTrainSample(bucket_id=bid, query=str(q), answer=str(a)))
    _log(f"load_qa_pairs: {len(samples)} kept, {stubs_dropped} generic stubs dropped")
    return samples


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CacheAugTrainer:
    """End-to-end trainer for coproc + fusion against frozen Qwen-3B."""

    def __init__(
        self,
        base_model,
        tokenizer,
        coprocessor: Coprocessor,
        fusion: CrossBucketFusion,
        bucket_kv_cache: BucketKVCache,
        bucket_chunks: dict[str, list],
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        # Phase 4-C.5.5 curriculum knobs:
        store=None,                                              # BucketStore (for verbatim)
        text_ratio_choices: tuple = (0.0, 0.25, 0.5, 0.75, 1.0),
        max_verbatim_chars: int = 2400,
    ) -> None:
        self.model = base_model
        self.tokenizer = tokenizer
        self.coproc = coprocessor
        self.fusion = fusion
        self.kv_cache = bucket_kv_cache
        self.bucket_chunks = bucket_chunks
        self.store = store
        self.text_ratio_choices = tuple(text_ratio_choices) or (0.0,)
        self.max_verbatim_chars = max_verbatim_chars

        # Freeze the receiver — only coproc + fusion train.
        for p in self.model.parameters():
            p.requires_grad_(False)

        trainable = list(self.coproc.parameters()) + list(self.fusion.parameters())
        # foreach=False: on MPS, AdamW's fused foreach kernel can no-op .step()
        # despite valid grads (sibling MPS bug to the LoRA input-grad-chain one
        # in feedback_mps_lora_grad.md). Forcing the slow per-tensor path is
        # the only way to make .step() actually apply.
        self.optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay, eps=1e-6, foreach=False
        )

        def _lr_lambda(step: int) -> float:
            # 1-indexed warmup: step 0 starts at 1/warmup_steps (not 0), so
            # the FIRST optimizer.step() runs with non-zero lr. LambdaLR
            # initializes lr to lambda(0) at construction time; a 0-indexed
            # warmup would waste step 0 at lr=0.
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        self.device = next(self.model.parameters()).device
        self.receiver_dtype = next(self.model.parameters()).dtype

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _build_z_fused_grad(self, sample: CacheAugTrainSample) -> Optional[torch.Tensor]:
        """Run coproc + fusion on a single bucket's cached KV. Returns z_fused
        with grad tracked through coproc + fusion params. None if cache miss."""
        flat = self.kv_cache.load(
            sample.bucket_id, self.bucket_chunks.get(sample.bucket_id, []), device="cpu"
        )
        if flat is None:
            return None
        # coproc handles the device cast internally via _build_context.
        z_b = self.coproc(flat)
        z_fused = self.fusion([z_b.to(self.device)])
        return z_fused

    def _sample_verbatim(self, sample: CacheAugTrainSample) -> tuple[str, float]:
        """Return (verbatim_text, text_ratio) for this step's prompt.

        Samples text_ratio ~ Uniform(self.text_ratio_choices). Returns the
        bucket's source text truncated to text_ratio × max_verbatim_chars.
        Returns ("", 0.0) when store is None (legacy mode), text_ratio is 0,
        or the bucket source can't be loaded."""
        if self.store is None:
            return "", 0.0
        r = random.choice(self.text_ratio_choices)
        if r <= 0.0:
            return "", 0.0
        try:
            from libucks.thinking.training.data_generator import _collect_source_text
            fm, _ = self.store.read(sample.bucket_id)
            src = _collect_source_text(fm, max_chars=self.max_verbatim_chars * 2) or ""
            verbatim = src[: int(r * self.max_verbatim_chars)]
            return verbatim, r
        except Exception:
            return "", 0.0

    def step(self, sample: CacheAugTrainSample) -> Optional[dict[str, float]]:
        """One training step on a single (bucket, query, answer) sample.

        Returns metrics dict, or None if the bucket cache is missing."""
        self.coproc.train()
        self.fusion.train()

        # ALL grad-path forwards inside enable_grad to defeat any ambient
        # inference_mode from upstream callers (HF fixtures, pytest, etc.).
        with torch.inference_mode(False), torch.enable_grad():
            z_fused = self._build_z_fused_grad(sample)
        if z_fused is None:
            return None
        assert z_fused.requires_grad, "z_fused lost grad before forward — check upstream context"

        # Build the prompt KV (frozen, no_grad).
        verbatim, text_ratio = self._sample_verbatim(sample)
        if verbatim:
            prompt = f"{verbatim}\n\nQuestion: {sample.query.strip()}\nAnswer:"
        else:
            prompt = f"Question: {sample.query.strip()}\nAnswer:"
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3500
        ).to(self.device)
        with torch.no_grad():
            out_in = self.model(input_ids=enc["input_ids"], use_cache=True)
            cache_input = out_in.past_key_values
            del out_in

        # Build z's KV WITH gradient tracking (this is where grad flows back).
        with torch.inference_mode(False), torch.enable_grad():
            z_for_receiver = z_fused.to(dtype=self.receiver_dtype)
            out_z = self.model(inputs_embeds=z_for_receiver, use_cache=True)
            cache_z = out_z.past_key_values
            del out_z
            assert cache_z.layers[0].keys.requires_grad, "cache_z lost grad after receiver forward"

            # Concat layer-by-layer. The cache_z keys/values retain grad; cache_input
            # is detached.
            from transformers.cache_utils import DynamicCache
            aug_cache = DynamicCache()
            for i in range(len(cache_input.layers)):
                k_in = cache_input.layers[i].keys.detach()
                v_in = cache_input.layers[i].values.detach()
                k_z = cache_z.layers[i].keys
                v_z = cache_z.layers[i].values
                aug_cache.update(torch.cat([k_in, k_z], dim=2), torch.cat([v_in, v_z], dim=2), layer_idx=i)
            assert aug_cache.layers[0].keys.requires_grad, "aug_cache lost grad after concat"

        # Forward over the answer tokens with the augmented cache.
        # Targets = answer_ids; inputs = answer_ids shifted by one (or just feed
        # answer_ids and predict the same positions; HF causal LM auto-shifts).
        ans_enc = self.tokenizer(
            sample.answer.strip(), return_tensors="pt",
            truncation=True, max_length=256, add_special_tokens=False,
        ).to(self.device)
        answer_ids = ans_enc["input_ids"]
        if answer_ids.shape[1] == 0:
            return None

        # Feed answer_ids preceded by a single "bridge" token (last input token)
        # so the first answer token has a position to attend from. Simpler: just
        # feed answer_ids; HF's logits[t] predicts answer_ids[t+1].
        # We shift inside: input = answer_ids[:-1], target = answer_ids[1:].
        # For very short answers (1 token), skip.
        if answer_ids.shape[1] < 2:
            return None
        in_ids = answer_ids[:, :-1]
        tgt_ids = answer_ids[:, 1:]
        attn_mask_total = torch.ones(
            1,
            enc["input_ids"].shape[1] + z_for_receiver.shape[1] + in_ids.shape[1],
            dtype=torch.long, device=self.device,
        )

        with torch.inference_mode(False), torch.enable_grad():
            out_ans = self.model(
                input_ids=in_ids,
                past_key_values=aug_cache,
                attention_mask=attn_mask_total,
                use_cache=False,
            )
            logits = out_ans.logits  # (1, T-1, V)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                tgt_ids.view(-1),
            )
            self.optimizer.zero_grad()
            loss.backward()
            # Pytest can leak ambient inference_mode into backward, locking the
            # grad tensors so optimizer.step() silently no-ops. Defensive clone.
            for p in list(self.coproc.parameters()) + list(self.fusion.parameters()):
                if p.grad is not None and p.grad.is_inference():
                    p.grad = p.grad.clone()

        torch.nn.utils.clip_grad_norm_(
            list(self.coproc.parameters()) + list(self.fusion.parameters()),
            max_norm=1.0,
        )
        # MPS: optimizer.step must run inside enable_grad as well — pytest's
        # inference_mode shadow extends here in some torch versions.
        with torch.inference_mode(False), torch.enable_grad():
            self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": float(loss.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "n_tokens": int(tgt_ids.numel()),
            "text_ratio": float(text_ratio),
        }

    # ------------------------------------------------------------------
    # Multi-step driver
    # ------------------------------------------------------------------

    def train_epoch(self, samples: list[CacheAugTrainSample]) -> dict[str, float]:
        """Run one full pass over the samples in random order."""
        order = list(range(len(samples)))
        random.shuffle(order)
        _log(f"epoch start: n_samples={len(samples)} lr={self.optimizer.param_groups[0]['lr']:.2e}")
        losses = []
        skipped = 0
        for i, idx in enumerate(order):
            metrics = self.step(samples[idx])
            if metrics is None:
                skipped += 1
                continue
            losses.append(metrics["loss"])
            if (i + 1) % 10 == 0:
                _log(f"  step {i+1}/{len(order)}: loss={metrics['loss']:.3f} lr={metrics['lr']:.2e}")
        return {
            "mean_loss": sum(losses) / max(1, len(losses)),
            "n_steps": len(losses),
            "skipped": skipped,
        }
