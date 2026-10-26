"""Aggregate per-seed evaluation files into paper-style tables (Table IV layout).

Expected layout: ``<root>/<method>/seed<k>/eval.json`` as written by ``scripts/evaluate.py``.
The statistical unit is the seed: each seed contributes its mean over episodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ricor.comm.faults import COMPOUND, HELD_OUT, ISOLATED
from ricor.evaluation.stats import bootstrap_ci, holm, welch
from ricor.utils import read_json

GROUPS = {"clean": ("clean",), "isolated": ISOLATED, "compound": COMPOUND, "held_out": HELD_OUT}
CVAR_ALPHA = 0.9


def episode_cvar(values: list[float], alpha: float = CVAR_ALPHA) -> float:
    """Empirical upper-tail CVaR_alpha: mean of the values at or above the alpha-quantile."""
    x = np.asarray(values, float)
    if x.size == 0:
        return float("nan")
    q = np.quantile(x, alpha)
    return float(x[x >= q].mean())


def collect(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Map method name -> list of per-seed eval.json payloads."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(Path(root).glob("*/seed*/eval.json")):
        out.setdefault(f.parent.parent.name, []).append(read_json(f))
    return out


def condition_metric(ev: dict[str, Any], cond: str, metric: str) -> float:
    """One seed's value of ``metric`` on ``cond``; ``cvar_sv`` is CVaR_0.9 of per-episode safety cost."""
    c = ev["conditions"][cond]
    if metric == "cvar_sv":
        return episode_cvar([e["sv"] for e in c["episodes"]])
    return float(c["summary"][metric])


def group_metric(ev: dict[str, Any], group: tuple[str, ...], metric: str) -> float:
    vals = [condition_metric(ev, c, metric) for c in group if c in ev["conditions"]]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(
    results: dict[str, list[dict[str, Any]]],
    reference: str | None = None,
    metrics: tuple[str, ...] = ("apr", "sv", "cvar_sv"),
    n_boot: int = 10000,
) -> dict[str, Any]:
    """Per method x group x metric: seed-level mean with 95% bootstrap CI; Welch vs reference (Holm-adjusted)."""
    table: dict[str, Any] = {}
    tests: list[tuple[str, str, str, float]] = []
    for method, evs in results.items():
        table[method] = {}
        for g, conds in GROUPS.items():
            for m in metrics:
                per_seed = [group_metric(ev, conds, m) for ev in evs]
                mean, lo, hi = bootstrap_ci(np.array(per_seed), n_boot)
                table[method][f"{g}.{m}"] = {"mean": mean, "ci": [lo, hi], "n_seeds": len(per_seed)}
                if reference and method != reference and reference in results:
                    ref = [group_metric(ev, conds, m) for ev in results[reference]]
                    w = welch(np.array(per_seed), np.array(ref))
                    table[method][f"{g}.{m}"]["welch_vs_" + reference] = w
                    tests.append((method, g, m, w["p"]))
    adjusted = holm([t[3] for t in tests]) if tests else []
    for (method, g, m, _), p_adj in zip(tests, adjusted, strict=True):
        table[method][f"{g}.{m}"]["welch_vs_" + reference]["p_holm"] = p_adj
    return table


def to_markdown(table: dict[str, Any], metrics: tuple[str, ...] = ("apr", "sv")) -> str:
    cols = [f"{g}.{m}" for g in GROUPS for m in metrics]
    lines = ["| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for method, row in table.items():
        cells = []
        for c in cols:
            v = row.get(c)
            if not v or np.isnan(v["mean"]):
                cells.append("n/a")
            else:
                half = (v["ci"][1] - v["ci"][0]) / 2
                cells.append(f"{v['mean']:.2f} ± {half:.2f} (n={v['n_seeds']})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
