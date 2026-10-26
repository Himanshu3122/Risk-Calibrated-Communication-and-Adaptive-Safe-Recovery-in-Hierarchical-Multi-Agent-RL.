"""Statistics across seeds: bootstrap CIs, Welch's t-test, Holm correction (paper Sec. VI-E)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over the given units (use one value per seed)."""
    x = np.asarray(values, float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(x.mean()), float(lo), float(hi)


def welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Welch's unequal-variance t-test plus Hedges' g effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return {"t": float("nan"), "p": float("nan"), "g": float("nan")}
    res = sps.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
    g = d * (1 - 3 / (4 * (a.size + b.size) - 9))
    return {"t": float(res.statistic), "p": float(res.pvalue), "g": float(g)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (the paper reports none; recommended in PROJECT_REPORT)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
