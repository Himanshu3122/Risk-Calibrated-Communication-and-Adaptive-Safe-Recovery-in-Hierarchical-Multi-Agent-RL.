"""Operating modes, hysteresis, and the fixed-threshold switching rule (paper Sec. IV-B, Eq. 10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Mode(IntEnum):
    NORMAL = 0
    SELECTIVE = 1
    VERIFY = 2
    FALLBACK = 3


N_MODES = 4
SUMMARY_DIM = 6


def mode_summary(
    rho_lower: np.ndarray, psi: np.ndarray, sigma: np.ndarray, outage: int, since_switch: int, hysteresis: int
) -> np.ndarray:
    """xi_t: mean/min rho_lower, max psi, mean sigma, outage length, time since last switch."""
    if rho_lower.size == 0:
        rho_lower = psi = sigma = np.zeros(1)
    return np.array(
        [
            float(rho_lower.mean()),
            float(rho_lower.min()),
            float(psi.max()),
            float(sigma.mean()),
            min(outage, 50) / 50.0,
            min(since_switch, 4 * hysteresis) / (4 * hysteresis),
        ],
        dtype=np.float32,
    )


def admissible(current: Mode, since_switch: int, hysteresis: int, verify_enabled: bool = True) -> np.ndarray:
    """Mask of admissible next modes: switches need ``hysteresis`` steps, except into fallback."""
    mask = np.zeros(N_MODES, dtype=bool)
    mask[current] = True
    mask[Mode.FALLBACK] = True
    if since_switch >= hysteresis:
        mask[:] = True
    if not verify_enabled:
        mask[Mode.VERIFY] = current == Mode.VERIFY
    return mask


def mode_reward(
    team_reward: float, switched: bool, n_queries: int, cost: float, lam: float, switch_cost: float, query_cost: float
) -> float:
    """Eq. (10): r^mu = R - c_sw 1[switch] - c_v n_q - lambda C."""
    return team_reward - switch_cost * float(switched) - query_cost * n_queries - lam * cost


@dataclass
class FixedRule:
    """Threshold switching used by the Ours-fixed ablation (triggers of paper Fig. 2)."""

    tau: float
    psi_v: float
    sigma_v: float
    outage_limit: int
    hysteresis: int
    calm_steps: int = field(default=0)

    def choose(self, current: Mode, xi: np.ndarray, outage: int, since_switch: int, verify: bool) -> Mode:
        mean_rho, min_rho, max_psi, mean_sigma = xi[:4]
        if outage > self.outage_limit:
            target = Mode.FALLBACK
        elif min_rho < self.tau:
            self.calm_steps = 0
            target = Mode.VERIFY if verify and max_psi > self.psi_v and mean_sigma > self.sigma_v else Mode.SELECTIVE
        else:
            self.calm_steps += 1
            target = Mode.NORMAL if self.calm_steps >= self.hysteresis else current
        if not admissible(current, since_switch, self.hysteresis, verify)[target]:
            return current
        return target
