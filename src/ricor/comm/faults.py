"""Fault taxonomy F1-F6, C1-C3, U1-U3 (paper Table II).

Evaluation parameters follow Table II. Where the table leaves a quantity open (the share
of messages affected by corruption or replay, the Gilbert-Elliott bad-state fraction,
training ranges), the values are assumptions A8-A10 in ``docs/assumptions.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaultSpec:
    """Channel and content fault parameters; all-zero means clean communication."""

    loss_p: float = 0.0  # Bernoulli loss (F1)
    ge_p_gb: float = 0.0  # Gilbert-Elliott good->bad (F2)
    ge_p_bg: float = 1.0  # Gilbert-Elliott bad->good; mean burst = 1/ge_p_bg
    ge_loss_bad: float = 1.0
    delay_min: int = 0  # uniform delay in steps (F3)
    delay_max: int = 0
    corrupt_rate: float = 0.0  # share of messages corrupted (F4)
    corrupt_sigma: float = 0.0
    corrupt_fields: float = 0.3  # share of fields perturbed in a corrupted message
    stale_rate: float = 0.0  # share of messages replaced by an old report (F5)
    stale_min: int = 10
    stale_max: int = 30
    compromised: int = 0  # number of compromised regional agents (F6)
    mislead_eps: float = 0.0  # l_inf budget of the misleading perturbation
    mislead_ramp: int = 0  # slow-drift adversary: steps over which eps ramps up from 0 (U4)
    multihop_frac: float = 0.0  # share of uplinks relayed over an extra hop (U2)
    multihop_delay: int = 2
    multihop_loss: float = 0.15

    @property
    def is_clean(self) -> bool:
        return self == FaultSpec()


def compose(*specs: FaultSpec) -> FaultSpec:
    """Combine faults: independent losses multiply survival, other fields take the stronger value."""
    out = FaultSpec()
    for s in specs:
        merged = {}
        for f in dataclasses.fields(FaultSpec):
            a, b = getattr(out, f.name), getattr(s, f.name)
            if f.name == "loss_p":
                merged[f.name] = 1.0 - (1.0 - a) * (1.0 - b)
            elif f.name == "ge_p_bg":
                merged[f.name] = min(a, b)
            elif f.name in {"stale_min", "corrupt_fields", "ge_loss_bad", "multihop_delay", "multihop_loss"}:
                merged[f.name] = b if getattr(s, f.name) != getattr(FaultSpec(), f.name) else a
            else:
                merged[f.name] = max(a, b)
        out = FaultSpec(**merged)
    return out


def _ge(mean_burst: float, bad_fraction: float) -> dict[str, float]:
    p_bg = 1.0 / mean_burst
    return {"ge_p_gb": bad_fraction * p_bg / (1.0 - bad_fraction), "ge_p_bg": p_bg}


def _isolated(
    loss: float = 0.3,
    burst: float = 14.0,
    bad: float = 0.25,
    delay_max: int = 6,
    sigma: float = 0.5,
    stale: float = 0.5,
    k: int = 2,
    eps: float = 0.4,
) -> dict[str, FaultSpec]:
    return {
        "F1": FaultSpec(loss_p=loss),
        "F2": FaultSpec(**_ge(burst, bad)),
        "F3": FaultSpec(delay_min=0, delay_max=delay_max),
        "F4": FaultSpec(corrupt_rate=0.5, corrupt_sigma=sigma),
        "F5": FaultSpec(stale_rate=stale),
        "F6": FaultSpec(compromised=k, mislead_eps=eps),
    }


def evaluation_spec(name: str, loss_override: float | None = None) -> FaultSpec:
    """Evaluation-time fault for a Table II condition name (``clean`` included)."""
    f = _isolated() if loss_override is None else _isolated(loss=loss_override)
    table = {
        "clean": FaultSpec(),
        **f,
        "C1": compose(f["F1"], f["F4"]),
        "C2": compose(f["F2"], f["F6"]),
        "C3": compose(f["F3"], f["F5"], f["F6"]),
        # U1: same families as training but at unseen rates (assumption A10).
        "U1": compose(
            FaultSpec(**_ge(20.0, 0.35)),
            FaultSpec(corrupt_rate=0.6, corrupt_sigma=0.8),
            FaultSpec(compromised=2, mislead_eps=0.5),
        ),
        "U2": FaultSpec(multihop_frac=0.25),
        "U3": FaultSpec(compromised=3, mislead_eps=0.4),
        # U4 (added in revision): slow-drift adversary that ramps eps over 20 steps so that the
        # step-to-step change of each report stays small (evades temporal-consistency cues).
        "U4": FaultSpec(compromised=2, mislead_eps=0.4, mislead_ramp=20),
        # Compromised-fraction sweep (with F6 = 2 and U3 = 3 compromised agents).
        "K1": FaultSpec(compromised=1, mislead_eps=0.4),
        "K4": FaultSpec(compromised=4, mislead_eps=0.4),
    }
    if name not in table:
        raise KeyError(f"unknown fault condition {name!r}; expected one of {sorted(table)}")
    return table[name]


ISOLATED = ("F1", "F2", "F3", "F4", "F5", "F6")
COMPOUND = ("C1", "C2", "C3")
HELD_OUT = ("U1", "U2", "U3")
EXTRA = ("U4", "K1", "K4")


def sample_training_spec(stage: int, rng: np.random.Generator, n_regions: int) -> FaultSpec:
    """Curriculum sample: stage 0 clean, 1 isolated F1-F6, 2 compound C1-C3 (randomized rates)."""
    if stage == 0:
        return FaultSpec()
    max_k = max(1, min(2, n_regions // 2))
    f = _isolated(
        loss=float(rng.uniform(0.1, 0.4)),
        burst=float(rng.uniform(8.0, 20.0)),
        bad=float(rng.uniform(0.1, 0.3)),
        delay_max=int(rng.integers(1, 7)),
        sigma=float(rng.uniform(0.3, 0.6)),
        stale=float(rng.uniform(0.2, 0.5)),
        k=int(rng.integers(1, max_k + 1)),
        eps=float(rng.uniform(0.2, 0.4)),
    )
    if stage == 1:
        return f[ISOLATED[int(rng.integers(len(ISOLATED)))]]
    combos = [("F1", "F4"), ("F2", "F6"), ("F3", "F5", "F6")]
    return compose(*(f[n] for n in combos[int(rng.integers(len(combos)))]))
