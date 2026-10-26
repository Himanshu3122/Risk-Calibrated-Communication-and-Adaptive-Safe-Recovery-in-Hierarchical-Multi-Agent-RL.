"""Decision sensitivity and the risk-conditioned gate (paper Eqs. 7-8)."""

from __future__ import annotations

import torch


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for categorical distributions given by logits along the last axis."""
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def decision_sensitivity(kl: torch.Tensor, tail_cost: torch.Tensor) -> torch.Tensor:
    """Eq. (7): psi = KL(pi(.|M) || pi(.|M_-j)) * E_{a ~ pi(.|M)} Q^c_alpha(h, a).

    ``kl`` has one entry per (receiver, sender); ``tail_cost`` is the receiver's expected
    CVaR cost of the decision it would take with the message, broadcast over senders.
    """
    return kl * tail_cost.clamp(min=0.0)


def gate_weights(
    rho_lower: torch.Tensor,
    psi: torch.Tensor,
    tau0: float,
    kappa: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (8): tau = tau0 + kappa * psi and w = sigmoid((rho_lower - tau) / T_g)."""
    tau = tau0 + kappa * psi
    return torch.sigmoid((rho_lower - tau) / temperature), tau
