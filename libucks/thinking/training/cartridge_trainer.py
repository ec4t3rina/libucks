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

import faulthandler
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.losses import distillation_loss

# MPS runs have wedged mid-step inside Metal waitUntilCompleted (CM-A.1 step
# ~180; CM-A.2 fe7ded0d epoch 1) with no traceback. Each precompute/train
# iteration re-arms this watchdog; a step stalling past the timeout dumps all
# thread stacks to stderr every interval instead of hanging silently for hours.
_STALL_DUMP_SECS = 300


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
        max_answer_tokens: int = 32,
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

    @torch.no_grad()
    def _teacher_forced_logits(
        self, verbatim: str, query: str, answer_ids: torch.Tensor
    ) -> torch.Tensor:
        """Teacher logits over a KNOWN answer via a SINGLE forward (teacher-forced).

        The teacher is frozen + deterministic, so once the greedy answer is known
        (precomputed once) its per-position logits are constant across epochs and
        recoverable with one forward — vs. 48 sequential forwards for greedy gen.
        Returns (A, V) aligned to answer_ids[0..A-1]."""
        with torch.inference_mode(False), torch.no_grad():
            ctx = self._q_text(query, verbatim[: self.max_verbatim_chars])
            ctx_ids = self.tokenizer(
                ctx, return_tensors="pt", truncation=True, max_length=3500
            )["input_ids"].to(self.device)
            fed = torch.cat([ctx_ids, answer_ids.view(1, -1)], dim=1)
            out = self.model(input_ids=fed, use_cache=False)
            lc = ctx_ids.shape[1]
            a = answer_ids.shape[0]
            # logits[lc-1 : lc-1+A] predict answer[0..A-1].
            return out.logits[0, lc - 1 : lc - 1 + a, :].detach().float()

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
    def _optimize(
        self, cartridge, optimizer, scheduler, query: str,
        answer_ids: torch.Tensor, teacher_ans_logits: torch.Tensor,
    ) -> dict[str, float]:
        """Student forward + KL(+CE) distill + step, given precomputed teacher
        target. Shared by step() (greedy) and _step_cached (teacher-forced)."""
        with torch.inference_mode(False), torch.enable_grad():
            student_ans_logits = self._student_logits(cartridge, query, answer_ids)
            kl = distillation_loss(student_ans_logits, teacher_ans_logits, self.temperature)
            loss = kl
            if self.alpha_ce > 0:
                ce = F.cross_entropy(student_ans_logits.float(), answer_ids)
                loss = kl + self.alpha_ce * ce
            optimizer.zero_grad()
            loss.backward()
            for p in cartridge.parameters():
                if p.grad is not None and p.grad.is_inference():
                    p.grad = p.grad.clone()

        torch.nn.utils.clip_grad_norm_(cartridge.parameters(), max_norm=1.0)
        with torch.inference_mode(False), torch.enable_grad():
            optimizer.step()
        scheduler.step()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return {
            "loss": float(loss.item()),
            "kl": float(kl.item()),
            "n_ans": int(answer_ids.shape[0]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

    def _step_cached(
        self, cartridge, optimizer, scheduler, verbatim: str, query: str,
        answer_ids: torch.Tensor,
    ) -> dict[str, float]:
        """Training step with a precomputed teacher answer — teacher logits via
        one forward (fast path used across all epochs)."""
        teacher_ans_logits = self._teacher_forced_logits(verbatim, query, answer_ids)
        return self._optimize(cartridge, optimizer, scheduler, query, answer_ids, teacher_ans_logits)

    def step(
        self, cartridge, optimizer, scheduler, verbatim: str, query: str
    ) -> Optional[dict[str, float]]:
        gen = self._teacher_generate(verbatim, query)
        if gen is None:
            return None
        answer_ids, teacher_ans_logits = gen
        return self._optimize(cartridge, optimizer, scheduler, query, answer_ids, teacher_ans_logits)

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
        checkpoint_path: str | Path | None = None,
        best_path: str | Path | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Distill one bucket's cartridge over its self-study queries.

        Teacher answers are precomputed ONCE (greedy) and reused across epochs —
        the teacher is frozen+deterministic, so per-epoch regeneration is wasted
        work. Training steps then get teacher logits via a single teacher-forced
        forward (fast) instead of 48-step greedy generation.

        With checkpoint_path set, the cartridge is saved after every epoch and a
        `<checkpoint_path>.json` sidecar records how many epochs it contains, so
        an MPS wedge loses at most one epoch rather than the whole bucket.

        Resuming is the CALLER's job: load the checkpoint into the cartridge and
        pass the remaining epoch count. See `scripts/cm_distill_buckets.py`.
        Until CM-B.0b nothing ever read the checkpoint back, so the guarantee in
        this docstring was not actually delivered — a wedge cost the full ~2 h
        bucket. Note that a resumed run's `init_mean_kl` is the first RESUMED
        epoch, not the true pre-training value.

        With `best_path` set, the lowest-mean-KL epoch is ALSO retained there,
        with a `<best_path>.json` sidecar naming the winning epoch. Training is
        not otherwise affected. This is opt-in because selecting the best of N
        epochs is a protocol change: a run that does it is not comparable to one
        that does not, and CM-A.1-retry (the reference result) took the last
        epoch. CM-B.0b needed this — bc6b90e2's epoch means were 5.4433,
        4.1816, 5.2721, 5.2284, and it promoted the 5.2284.

        `seed` makes the per-epoch query shuffle reproducible. Without it the
        global `random` is used and no two runs see the same order, which is why
        this track has no error bars: 2/8 vs 4/8, 7/25 vs 10/25 and 5/8 vs 1/8
        are all single samples with unmeasured variance. Vary the seed with
        everything else fixed to measure it. Note this seeds the ORDER only —
        cartridge init uses `torch.randn` separately, and warm-starting from
        extracted KV overwrites it anyway."""
        cartridge.train()
        cartridge.to(self.device)
        try:
            return self._distill_bucket_inner(
                cartridge, verbatim, queries,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                checkpoint_path=checkpoint_path, best_path=best_path,
                seed=seed,
            )
        finally:
            faulthandler.cancel_dump_traceback_later()

    def _distill_bucket_inner(
        self,
        cartridge: KVPrefixCartridge,
        verbatim: str,
        queries: list[str],
        *,
        epochs: int,
        lr: float,
        weight_decay: float,
        checkpoint_path: str | Path | None,
        best_path: str | Path | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        # --- precompute teacher answers once (greedy) ---
        _log(f"precomputing teacher answers for {len(queries)} queries ...")
        qa: list[tuple[str, torch.Tensor]] = []
        for i, q in enumerate(queries):
            faulthandler.dump_traceback_later(_STALL_DUMP_SECS, repeat=True, exit=False)
            gen = self._teacher_generate(verbatim, q)
            if gen is None:
                continue
            qa.append((q, gen[0]))  # (query, answer_ids); discard greedy logits
            if self.device.type == "mps":
                torch.mps.empty_cache()
            if (i + 1) % 10 == 0:
                _log(f"  precomputed {i+1}/{len(queries)}")
        _log(f"precomputed {len(qa)} usable (query, answer) pairs")
        if not qa:
            return {"epoch_mean_kl": [], "init_mean_kl": 0.0, "final_mean_kl": 0.0,
                    "n_queries": 0, "best_epoch": None, "best_mean_kl": None}

        total_steps = max(1, epochs * len(qa))
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
        best_mean_kl = float("inf")
        best_epoch: int | None = None
        # A dedicated Random when seeded, so reproducibility does not depend on
        # global RNG state that any other import could have advanced.
        rng = random.Random(seed) if seed is not None else random
        for ep in range(epochs):
            order = list(range(len(qa)))
            rng.shuffle(order)
            ep_kls: list[float] = []
            for i, idx in enumerate(order):
                faulthandler.dump_traceback_later(_STALL_DUMP_SECS, repeat=True, exit=False)
                q, ans = qa[idx]
                m = self._step_cached(cartridge, optimizer, scheduler, verbatim, q, ans)
                ep_kls.append(m["kl"])
                if self.device.type == "mps" and (i + 1) % 20 == 0:
                    # Flush the Metal command queue at a controlled point; the
                    # observed wedge is an unbounded waitUntilCompleted deep in
                    # a step after long uninterrupted queueing.
                    torch.mps.synchronize()
                if (i + 1) % 40 == 0:
                    _log(f"  ep{ep} step {i+1}/{len(order)}: kl={m['kl']:.3f} loss={m['loss']:.3f} lr={m['lr']:.2e}")
            mean_kl = sum(ep_kls) / max(1, len(ep_kls))
            history.append(mean_kl)
            if ep == 0:
                first_kls = ep_kls
            last_kls = ep_kls
            _log(f"epoch {ep}: mean_kl={mean_kl:.4f} steps={len(ep_kls)}")
            if checkpoint_path is not None:
                cartridge.save(checkpoint_path)
                # Sidecar records how many epochs the checkpoint actually
                # contains. Without it a resume cannot know how many remain and
                # would silently re-run all of them — which is why the
                # checkpoint was write-only and the "loses at most one epoch"
                # claim was false until CM-B.0b.
                Path(str(checkpoint_path) + ".json").write_text(
                    json.dumps({"epochs_done": ep + 1, "mean_kl": mean_kl})
                )
                _log(f"  checkpoint saved (epoch {ep}, {ep + 1} done) -> {checkpoint_path}")

            # Retain the best epoch SEPARATELY. It must not share
            # `checkpoint_path`: that file's sidecar promises `epochs_done`
            # epochs of training, and skipping a write on a worse epoch would
            # leave the two out of step and rewind a resume.
            if best_path is not None and mean_kl < best_mean_kl:
                best_mean_kl, best_epoch = mean_kl, ep
                cartridge.save(best_path)
                Path(str(best_path) + ".json").write_text(
                    json.dumps({"best_epoch": ep, "best_mean_kl": mean_kl})
                )
                _log(f"  best so far (epoch {ep}, mean_kl={mean_kl:.4f}) -> {best_path}")

        return {
            "epoch_mean_kl": history,
            "init_mean_kl": sum(first_kls) / max(1, len(first_kls)),
            "final_mean_kl": sum(last_kls) / max(1, len(last_kls)),
            "n_queries": len(qa),
            "best_epoch": best_epoch,
            "best_mean_kl": None if best_epoch is None else best_mean_kl,
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

            text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            if self.device.type == "mps":
                torch.mps.empty_cache()
            return text
