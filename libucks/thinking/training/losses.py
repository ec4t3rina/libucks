"""Interlat-Lite training loss functions.

separation_loss       — simplified L_sep from Interlat §3.2 (JSD-based).
margin_separation_loss — direct margin ranking on first target token (no chicken-and-egg).
alignment_loss        — L_align from Interlat §3.3 (prevents L_sep exploitation).
"""
import torch
import torch.nn.functional as F


def margin_separation_loss(
    logits_correct: torch.Tensor,
    logits_wrong: torch.Tensor,
    target_ids: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Margin ranking loss: correct-latent logit for y[0] must exceed wrong-latent by margin.

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
