"""Interlat-Lite training loss functions.

separation_loss — simplified L_sep from Interlat §3.2.
"""
import torch
import torch.nn.functional as F


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
    # Training code uses: L_total = L_task - λ * separation_loss(...)
    return jsd.mean()
