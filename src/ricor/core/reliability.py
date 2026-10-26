"""Factorized reliability with evidential uncertainty (paper Eqs. 4-6)."""

from __future__ import annotations

import torch


def beta_from_evidence(e_pos: torch.Tensor, e_neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (4): alpha = 1 + e+, beta = 1 + e-."""
    return 1.0 + e_pos, 1.0 + e_neg


def beta_mean_var(alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean r_bar (Eq. 4) and variance sigma^2 (Eq. 5) of Beta(alpha, beta)."""
    s = alpha + beta
    mean = alpha / s
    var = alpha * beta / (s.pow(2) * (s + 1.0))
    return mean, var


def reliability(
    p_ch: torch.Tensor,
    r_bar: torch.Tensor,
    sigma: torch.Tensor,
    lambda_u: float,
    split: bool = True,
    uncertainty: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (6): rho_bar = p_ch * r_bar and the lower bound rho_lower.

    ``split=False`` reproduces the joint-head ablation, where ``r_bar`` already encodes
    channel and content and ``p_ch`` is ignored. ``uncertainty=False`` reproduces the
    point-estimate ablation (rho_lower = rho_bar).
    """
    ch = p_ch if split else torch.ones_like(p_ch)
    rho_bar = ch * r_bar
    if not uncertainty:
        return rho_bar, rho_bar
    rho_lower = ch * torch.clamp(r_bar - lambda_u * sigma, min=0.0)
    return rho_bar, rho_lower


def beta_kl_to_uniform(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """KL(Beta(alpha, beta) || Beta(1, 1)), used by the evidential regularizer in Eq. (12)."""
    s = alpha + beta
    return (
        torch.lgamma(s)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(s))
        + (beta - 1.0) * (torch.digamma(beta) - torch.digamma(s))
    )


def reliability_loss(
    p_ch: torch.Tensor,
    y_ch: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y_cr: torch.Tensor,
    beta_kl: float,
) -> torch.Tensor:
    """Eq. (12): Brier loss for the channel, evidential Brier + variance, and KL shrinkage.

    ``(alpha~, beta~)`` remove the evidence that supports the correct label, so only
    misleading evidence is pulled toward the uniform prior.
    """
    r_bar, var = beta_mean_var(alpha, beta)
    channel = (y_ch - p_ch).pow(2).mean()
    content = ((y_cr - r_bar).pow(2) + var).mean()
    alpha_t = y_cr + (1.0 - y_cr) * alpha
    beta_t = (1.0 - y_cr) + y_cr * beta
    kl = beta_kl_to_uniform(alpha_t, beta_t).mean()
    return channel + content + beta_kl * kl
