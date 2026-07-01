"""Cartridge Memory (CM-A) — context-distillation trainer.

Trains a per-bucket KVPrefixCartridge so the FROZEN receiver, given only the
cartridge prefix (no verbatim), reproduces the next-token distribution it
produces WITH the full verbatim context. This is the objective libucks was
missing (Phase 4-C used cosine/CE only); per arXiv 2605.28889 / Cartridges
2506.06266 it is what makes a latent channel carry facts.

Per distillation step, for one (verbatim, query):
    teacher  = frozen model, context [verbatim, "Question: q\\nAnswer:"]
               → greedy-generate answer a AND capture per-token logits  (no_grad)
    student  = frozen model, prefix = cartridge KV, input ["Question: q\\nAnswer:", a]
               → logits at the answer positions                          (grad → cartridge)
    loss     = KL(teacher ‖ student) over a  (+ optional CE on a)
    backward → cartridge prefix only (receiver params requires_grad=False)

MPS correctness patterns (inference_mode/enable_grad guards, foreach=False,
defensive grad.clone, step under enable_grad) are copied verbatim from
cache_aug_trainer.py, which validated them on this exact hardware.
"""
from __future__ import annotations

import math
import random
import sys
from typing import Any, Optional

import torch
import torch.nn.functional as F

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.losses import distillation_loss


def _log(msg: str) -> None:
    print(f"[libucks:cartridge_train] {msg}", file=sys.stderr, flush=True)


class CartridgeTrainer:
    def __init__(
        self,
        base_model,
        tokenizer,
        *,
        temperature: float = 2.0,
        alpha_ce: float = 0.3,
        max_answer_tokens: int = 48,
        max_verbatim_chars: int = 3000,
    ) -> None:
        self.model = base_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.max_answer_tokens = max_answer_tokens
        self.max_verbatim_chars = max_verbatim_chars

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = next(self.model.parameters()).device
        self.receiver_dtype = next(self.model.parameters()).dtype
        self.eos_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    def _q_text(self, query: str, verbatim: str = "") -> str:
        if verbatim:
            return f"{verbatim}\n\nQuestion: {query.strip()}\nAnswer:"
        return f"Question: {query.strip()}\nAnswer:"

    @torch.no_grad()
    def _teacher_generate(
        self, verbatim: str, query: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Greedy-generate the teacher's answer from full context and capture
        the aligned per-token logits. Returns (answer_ids (A,), logits (A, V))
        or None if the teacher emits no tokens."""
        with torch.inference_mode(False), torch.no_grad():
            ctx = self._q_text(query, verbatim[: self.max_verbatim_chars])
            enc = self.tokenizer(
                ctx, return_tensors="pt", truncation=True, max_length=3500
            ).to(self.device)
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=True,
            )
            cache = out.past_key_values
            next_logits = out.logits[:, -1, :]  # predicts answer[0]

            answer_ids: list[int] = []
            teacher_logits: list[torch.Tensor] = []
            for _ in range(self.max_answer_tokens):
                teacher_logits.append(next_logits[0].detach().clone())
                nxt = int(next_logits[0].argmax().item())
                if nxt == self.eos_id:
                    break
                answer_ids.append(nxt)
                step_in = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=step_in, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                next_logits = out.logits[:, -1, :]

            a = len(answer_ids)
            if a == 0:
                return None
            # teacher_logits[j] is the distribution that produced answer[j].
            ans_logits = torch.stack(teacher_logits[:a], dim=0).float()  # (A, V)
            ans_ids = torch.tensor(answer_ids, dtype=torch.long, device=self.device)
            return ans_ids, ans_logits

    def _student_logits(
        self, cartridge: KVPrefixCartridge, query: str, answer_ids: torch.Tensor
    ) -> torch.Tensor:
        """Logits at the answer positions from the cartridge-conditioned forward.
        Grad flows into the cartridge prefix. Must be called under enable_grad."""
        prefix_cache = cartridge.to_dynamic_cache(self.device, dtype=self.receiver_dtype)
        q_ids = self.tokenizer(
            self._q_text(query), return_tensors="pt", truncation=True, max_length=1024
        )["input_ids"].to(self.device)
        fed = torch.cat([q_ids, answer_ids.view(1, -1)], dim=1)  # (1, Lq + A)
        total_len = cartridge.prefix_len + fed.shape[1]
        attn = torch.ones(1, total_len, dtype=torch.long, device=self.device)

        out = self.model(
            input_ids=fed,
            past_key_values=prefix_cache,
            attention_mask=attn,
            use_cache=False,
        )
        logits = out.logits  # (1, Lq + A, V)
        lq = q_ids.shape[1]
        a = answer_ids.shape[0]
        # logits[:, lq-1 : lq-1+A] predict answer[0..A-1].
        return logits[0, lq - 1 : lq - 1 + a, :]

    # ------------------------------------------------------------------
    def step(
        self, cartridge, optimizer, scheduler, verbatim: str, query: str
    ) -> Optional[dict[str, float]]:
        gen = self._teacher_generate(verbatim, query)
        if gen is None:
            return None
        answer_ids, teacher_ans_logits = gen

        with torch.inference_mode(False), torch.enable_grad():
            student_ans_logits = self._student_logits(cartridge, query, answer_ids)
            kl = distillation_loss(student_ans_logits, teacher_ans_logits, self.temperature)
            loss = kl
            if self.alpha_ce > 0:
                ce = F.cross_entropy(student_ans_logits.float(), answer_ids)
                loss = kl + self.alpha_ce * ce
            optimizer.zero_grad()
            loss.backward()
            # MPS/pytest can leave grads as inference tensors → step() no-ops.
            for p in cartridge.parameters():
                if p.grad is not None and p.grad.is_inference():
                    p.grad = p.grad.clone()

        torch.nn.utils.clip_grad_norm_(cartridge.parameters(), max_norm=1.0)
        with torch.inference_mode(False), torch.enable_grad():
            optimizer.step()
        scheduler.step()

        return {
            "loss": float(loss.item()),
            "kl": float(kl.item()),
            "n_ans": int(answer_ids.shape[0]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

    # ------------------------------------------------------------------
    def distill_bucket(
        self,
        cartridge: KVPrefixCartridge,
        verbatim: str,
        queries: list[str],
        *,
        epochs: int = 2,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> dict[str, Any]:
        """Distill one bucket's cartridge over its self-study queries."""
        cartridge.train()
        cartridge.to(self.device)
        total_steps = max(1, epochs * len(queries))
        warmup = max(1, total_steps // 10)

        optimizer = torch.optim.AdamW(
            cartridge.parameters(), lr=lr, weight_decay=weight_decay,
            eps=1e-6, foreach=False,
        )

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        history: list[float] = []
        first_kls: list[float] = []
        last_kls: list[float] = []
        for ep in range(epochs):
            order = list(range(len(queries)))
            random.shuffle(order)
            ep_kls: list[float] = []
            skipped = 0
            for i, idx in enumerate(order):
                m = self.step(cartridge, optimizer, scheduler, verbatim, queries[idx])
                if m is None:
                    skipped += 1
                    continue
                ep_kls.append(m["kl"])
                if (i + 1) % 20 == 0:
                    _log(f"  ep{ep} step {i+1}/{len(order)}: kl={m['kl']:.3f} loss={m['loss']:.3f} lr={m['lr']:.2e}")
            mean_kl = sum(ep_kls) / max(1, len(ep_kls))
            history.append(mean_kl)
            if ep == 0:
                first_kls = ep_kls
            last_kls = ep_kls
            _log(f"epoch {ep}: mean_kl={mean_kl:.4f} steps={len(ep_kls)} skipped={skipped}")

        return {
            "epoch_mean_kl": history,
            "init_mean_kl": sum(first_kls) / max(1, len(first_kls)),
            "final_mean_kl": sum(last_kls) / max(1, len(last_kls)),
            "n_queries": len(queries),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_answer(
        self, cartridge: KVPrefixCartridge, query: str,
        *, max_new_tokens: int = 64, verbatim: str = "",
    ) -> str:
        """Greedy-decode an answer from the cartridge prefix (latent-alone when
        verbatim=""). Used for the CM-A.1 proof and eval."""
        with torch.inference_mode(False), torch.no_grad():
            cartridge.eval()
            prefix_cache = cartridge.to_dynamic_cache(self.device, dtype=self.receiver_dtype)
            q_ids = self.tokenizer(
                self._q_text(query, verbatim), return_tensors="pt",
                truncation=True, max_length=3500,
            )["input_ids"].to(self.device)
            cur_len = cartridge.prefix_len + q_ids.shape[1]
            attn = torch.ones(1, cur_len, dtype=torch.long, device=self.device)
            out = self.model(
                input_ids=q_ids, past_key_values=prefix_cache,
                attention_mask=attn, use_cache=True,
            )
            cache = out.past_key_values
            next_logits = out.logits[:, -1, :]

            gen: list[int] = []
            for _ in range(max_new_tokens):
                nxt = int(next_logits[0].argmax().item())
                if nxt == self.eos_id:
                    break
                gen.append(nxt)
                cur_len += 1
                attn = torch.ones(1, cur_len, dtype=torch.long, device=self.device)
                step_in = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                out = self.model(
                    input_ids=step_in, past_key_values=cache,
                    attention_mask=attn, use_cache=True,
                )
                cache = out.past_key_values
                next_logits = out.logits[:, -1, :]

            return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
