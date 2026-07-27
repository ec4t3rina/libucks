"""Interlat-Lite training loss functions.

separation_loss       — simplified L_sep from Interlat §3.2 (JSD-based).
margin_separation_loss — direct margin ranking on first target token (no chicken-and-egg).
alignment_loss        — L_align from Interlat §3.3 (prevents L_sep exploitation).
distillation_loss     — context distillation (Cartridge Memory, CM-A): KL from a
                        full-context teacher into a latent-conditioned student.
"""
import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Context-distillation loss: KL(teacher ‖ student) over answer positions.

    The teacher sees the full verbatim context; the student sees only the
    trained latent (cartridge) prefix. Minimising this trains the latent to
    make the frozen model reproduce the teacher's full-context next-token
    distribution — the objective that (per arXiv 2605.28889 / Cartridges
    2506.06266) makes a latent memory channel actually carry facts, where the
    cosine/margin objectives did not.

    Args:
        student_logits: (P, V) logits at the answer positions from the
            cartridge-conditioned forward. Grad flows through these.
        teacher_logits: (P, V) logits at the same answer positions from the
            full-context forward. Detached internally (teacher is fixed).
        temperature: softmax temperature T; loss scaled by T² (Hinton distillation).

    Returns:
        Scalar tensor ≥ 0. Zero when the student matches the teacher.
    """
    t = max(1e-6, float(temperature))
    log_student = F.log_softmax(student_logits.float() / t, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float().detach() / t, dim=-1)
    # F.kl_div(log_student, teacher_probs) = Σ teacher·(log teacher − log student) = KL(teacher‖student)
    return F.kl_div(log_student, teacher_probs, reduction="batchmean") * (t * t)


def margin_separation_loss(
    logits_correct: torch.Tensor,
    logits_wrong: torch.Tensor,
    target_ids: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Margin ranking loss: correct-latent logit for y[0] must exceed wrong-latent by margin.

    ⚠️ NEVER WIRED IN. Added by 03a9db5 ("fix adapter collapse via L_xsep +
    L_sep"), imported by lora_trainer.py:33, and called from nowhere — verified
    against the full git history, not just the current tree. It also has no
    test; tests/unit/test_lsep_loss.py covers only `separation_loss`.

    That matters because of what it was written to fix. `separation_loss` (JSD)
    is the loss actually in use (lora_trainer.py:348), and JSD has ZERO gradient
    when the two distributions coincide — exactly the sep=0.0000 collapse
    CLAUDE.md tells you to stop the run over. This function was built to escape
    that state, per the docstring below. The project's shipped mitigation is
    instead query_dropout_rate=0.5, a different lever.

    So: if sep=0.0000 recurs and query dropout does not rescue it, swapping
    this in is the obvious next experiment — but it IS an experiment. It
    changes the LoRA objective, so it needs its own gate and a retrain, and
    every existing lora_receiver.pt was trained without it.

    L_sep = ReLU(margin - (logit_c[y0] - logit_w[y0]))

    Unlike JSD, this produces a non-zero gradient even when the model ignores
    the latent (chicken-and-egg avoided): when logit_c ≈ logit_w, gradient
    is -1 on logit_c, directly pushing the model to assign higher probability
    to the correct answer given the correct latent.

    Args:
        logits_correct: Logits from the correct-latent path.  Shape (T, V).
        logits_wrong:   Logits from the wrong-latent path.  Shape (T, V). Must be detached.
        target_ids:     Target token IDs.  Shape (T,).
        margin:         Minimum required logit gap (default 2.0 ≈ 7× probability ratio).

    Returns:
        Scalar tensor. Zero once the margin is satisfied; margin when ignored entirely.
    """
    y0 = target_ids[0].long()
    c = logits_correct[0, y0].float()
    w = logits_wrong[0, y0].float().detach()
    return F.relu(torch.tensor(margin, device=c.device, dtype=c.dtype) - (c - w))


def separation_loss(
    logits_correct: torch.Tensor,
    logits_wrong: torch.Tensor,
) -> torch.Tensor:
    """Compute L_sep = -mean(JSD(p_correct, p_wrong)).

    Rewards the model for producing *different* distributions when given the
    correct vs. a mismatched latent — i.e., for actually using the latent signal.

    Args:
        logits_correct: Logits given the correct latent.  Shape (..., V).
        logits_wrong:   Logits given a mismatched latent.  Shape (..., V).

    Returns:
        Scalar tensor.  Always >= 0.  Zero when distributions are identical.

    Note on sign: JSD ∈ [0, log 2]. We negate it so the caller can add it
    to the task loss with ``L_total = L_task + λ * L_sep`` (both minimised).
    Minimising -JSD maximises divergence between correct and wrong logits.
    """
    # Upcast to float32 before softmax: float16 exp() overflows at logit ~9
    # (exp(11) ≈ 60000 ≈ float16 max), producing inf → NaN in log ops.
    logits_correct = logits_correct.float()
    logits_wrong = logits_wrong.float()

    p = F.softmax(logits_correct, dim=-1)
    q = F.softmax(logits_wrong, dim=-1)
    m = 0.5 * (p + q)

    # KL(P||M) and KL(Q||M) — clamp to avoid log(0)
    eps = 1e-8
    kl_pm = (p * (torch.log(p.clamp(min=eps)) - torch.log(m.clamp(min=eps)))).sum(dim=-1)
    kl_qm = (q * (torch.log(q.clamp(min=eps)) - torch.log(m.clamp(min=eps)))).sum(dim=-1)

    jsd = 0.5 * kl_pm + 0.5 * kl_qm  # (...) — one value per position
    # Return positive JSD so callers can inspect "how separated" the distributions are.
    # Training code uses: L_total = L_task - λ_sep * separation_loss(...)
    return jsd.mean()


def alignment_loss(
    logits_correct: torch.Tensor,
    logits_plan: torch.Tensor,
) -> torch.Tensor:
    """KL(p_correct || p_plan) — Interlat L_align: anchor latent path to plan distribution.

    Prevents L_sep exploitation: when the model shifts probability to idiosyncratic tokens
    to maximise JSD, KL(p_garbage || p_plan) becomes large (~5–10 nats), overwhelming the
    small L_sep incentive and forcing the distribution back toward coherent output.

    Args:
        logits_correct: Logits from the latent-conditioned correct path.  Shape (..., V).
        logits_plan:    Logits from the query-only plan path.  Shape (..., V).
                        Caller must detach() before passing — no gradient flows through it.

    Returns:
        Scalar tensor >= 0.  Near zero when distributions match; large when the latent
        path has drifted far from what the model predicts from the query alone.
    """
    log_q = F.log_softmax(logits_plan.float(), dim=-1)   # detached plan distribution
    p = F.softmax(logits_correct.float(), dim=-1)         # latent-conditioned distribution
    # F.kl_div(log_q, p, 'batchmean') = KL(p || q) = Σ p*(log p − log q) / N
    return F.kl_div(log_q, p, reduction="batchmean")
