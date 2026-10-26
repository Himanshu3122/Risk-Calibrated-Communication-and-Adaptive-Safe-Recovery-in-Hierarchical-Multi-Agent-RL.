"""Evaluation metrics of paper Sec. VI-E."""

from __future__ import annotations

import numpy as np


def ece(prob: np.ndarray, label: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins (Naeini et al. 2015)."""
    prob, label = np.asarray(prob, float), np.asarray(label, float)
    if prob.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(prob[m].mean() - label[m].mean())
    return float(total)


def brier(prob: np.ndarray, label: np.ndarray) -> float:
    prob, label = np.asarray(prob, float), np.asarray(label, float)
    return float(np.mean((prob - label) ** 2)) if prob.size else float("nan")


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (ties get average ranks)."""
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def far_frr(prob: np.ndarray, label: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """False acceptance (unreliable accepted) and false rejection (reliable rejected) rates in %."""
    prob, label = np.asarray(prob, float), np.asarray(label).astype(bool)
    accept = prob >= threshold
    far = 100.0 * float((accept & ~label).sum() / max((~label).sum(), 1))
    frr = 100.0 * float((~accept & label).sum() / max(label.sum(), 1))
    return far, frr


DIAGNOSIS = ("clean", "channel", "content")


def diagnosis_confusion(p_ch: np.ndarray, r_bar: np.ndarray, y_ch: np.ndarray, y_cr: np.ndarray) -> np.ndarray:
    """Row-normalized 3-way confusion (true x predicted, %): clean / channel failure / content failure.

    The paper's Fig. 6 separates corrupted from misleading content; this model has no
    four-way classifier, so content faults are pooled (assumption A14).
    """
    true = np.where(np.asarray(y_ch) < 0.5, 1, np.where(np.asarray(y_cr) < 0.5, 2, 0))
    pred = np.where(np.asarray(p_ch) < 0.5, 1, np.where(np.asarray(r_bar) < 0.5, 2, 0))
    mat = np.zeros((3, 3))
    for t, p in zip(true, pred, strict=True):
        mat[t, p] += 1
    rows = mat.sum(1, keepdims=True)
    return 100.0 * np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)


def recovery_time(curve: np.ndarray, window: tuple[int, int], smoothing: int = 10, level: float = 0.95) -> float:
    """Steps after fault removal until the smoothed per-step reward regains ``level`` of its pre-fault value.

    Rewards in HierFire are non-positive, so "regaining 95%" is measured on the reward
    shifted by the pre-fault minimum (assumption A15). Returns NaN if it never recovers.
    """
    start, end = window
    if start <= 2 * smoothing or end >= len(curve):
        raise ValueError("recovery window must leave room for smoothing on both sides")
    # Trailing (causal) moving average: smooth[t] uses rewards up to t only, so recovery is never
    # detected before it happens. The first ``smoothing`` steps (partial windows) are excluded
    # from the pre-fault baseline.
    curve = np.asarray(curve, float)
    csum = np.concatenate([[0.0], np.cumsum(curve)])
    idx = np.arange(1, len(curve) + 1)
    lo_idx = np.maximum(0, idx - smoothing)
    smooth = (csum[idx] - csum[lo_idx]) / (idx - lo_idx)
    base = smooth[smoothing:start]
    lo = min(float(smooth.min()), float(base.min()))
    pre = float(base.mean()) - lo
    if pre <= 0:
        return float("nan")
    for t in range(end, len(curve)):
        if smooth[t] - lo >= level * pre:
            return float(t - end)
    return float("nan")


def degradation(clean: float, faulty: float) -> float:
    """Relative APR loss from clean to faulty conditions, in %."""
    return 100.0 * (clean - faulty) / clean if clean else float("nan")


def reliability_report(rec: dict[str, np.ndarray], bins: int = 15) -> dict[str, float | list[list[float]]]:
    """ECE, Brier, AUROC of rho_bar against y = y_ch * y_cr; FAR/FRR of the gate's actual acceptance.

    A message counts as accepted when its gate weight w (Eq. 8, including the risk term) is at
    least 0.5, i.e., at the operating threshold rather than at a fixed cut on rho_bar.
    """
    if rec["rho_bar"].size == 0:
        return {}
    y = rec["y_ch"] * rec["y_cr"]
    far, frr = far_frr(rec["w"], y, threshold=0.5)
    return {
        "ece": ece(rec["rho_bar"], y, bins),
        "brier": brier(rec["rho_bar"], y),
        "auroc": auroc(rec["rho_bar"], y),
        "far": far,
        "frr": frr,
        "n_messages": int(y.size),
        "confusion": diagnosis_confusion(rec["p_ch"], rec["r_bar"], rec["y_ch"], rec["y_cr"]).tolist(),
    }
