"""Build every results table, plot-data file, and number macro of the revised manuscript from run outputs.

Usage:
    python scripts/make_paper_artifacts.py --root runs/revision --out ../results

Reads ``<root>/<method>/seed<k>/eval.json``, ``train_history.json``, and ``checkpoint.json``; writes LaTeX
tables (``tab_*.tex``), pgfplots data files (``*.dat``), a macro file (``numbers.tex``) with every
number quoted in the prose, and ``artifacts_summary.json`` with the same values for auditing.
No value in these files is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ricor.comm.faults import COMPOUND, HELD_OUT, ISOLATED
from ricor.evaluation.aggregate import condition_metric, group_metric
from ricor.evaluation.stats import bootstrap_ci, holm, welch

GROUPS = {"clean": ("clean",), "isolated": ISOLATED, "compound": COMPOUND, "held_out": HELD_OUT}
LABEL = {
    "h_mappo": "H-MAPPO",
    "h_mappo_drop": "H-MAPPO+Drop",
    "admac_h": "ADMAC-H",
    "beta_rep": "Beta-Rep",
    "hpsf": "HPSF",
    "ricor": r"\method{}",
    "no_comm": "No-Comm (bound)",
    "oracle_gate": "Oracle gate (bound)",
    "ricor_no_split": "w/o channel/content split",
    "ricor_no_uncertainty": "w/o uncertainty (point est.)",
    "ricor_kappa0": r"w/o risk conditioning ($\kappa=0$)",
    "ricor_fixed": "w/o learned modes (fixed rule)",
    "ricor_no_verify": "w/o verify mode",
    "ricor_no_shield": "w/o safety shield",
}
MAIN = ["h_mappo", "h_mappo_drop", "admac_h", "beta_rep", "hpsf", "ricor"]
BOUNDS = ["no_comm", "oracle_gate"]
ABL = ["ricor_no_split", "ricor_no_uncertainty", "ricor_kappa0", "ricor_fixed", "ricor_no_verify", "ricor_no_shield"]
MACRO_NAME = {
    "h_mappo": "Hmappo",
    "h_mappo_drop": "Drop",
    "admac_h": "Admac",
    "beta_rep": "Brep",
    "hpsf": "Hpsf",
    "ricor": "Ricor",
    "no_comm": "Nocomm",
    "oracle_gate": "Oracle",
    "ricor_no_split": "AblSplit",
    "ricor_no_uncertainty": "AblUnc",
    "ricor_kappa0": "AblKappa",
    "ricor_fixed": "AblFixed",
    "ricor_no_verify": "AblVerify",
    "ricor_no_shield": "AblShield",
}
GROUP_MACRO = {"clean": "Clean", "isolated": "Iso", "compound": "Comp", "held_out": "Held"}
METRIC_MACRO = {"apr": "Apr", "sv": "Sv", "cvar_sv": "Cvar", "ba": "Ba"}
DIGIT_WORDS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def load(root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(root.glob("*/seed*/eval.json")):
        ev = json.loads(f.read_text(encoding="utf-8"))
        hist = f.with_name("train_history.json")
        ev["_history"] = json.loads(hist.read_text(encoding="utf-8")) if hist.is_file() else []
        ev["_seed_dir"] = str(f.parent)
        out.setdefault(f.parent.parent.name, []).append(ev)
    return out


def ci(values: list[float]) -> dict[str, float]:
    x = np.array([v for v in values if not math.isnan(v)], float)
    if x.size == 0:
        return {"mean": float("nan"), "half": float("nan"), "n": 0}
    m, lo, hi = bootstrap_ci(x, 10000)
    return {"mean": m, "half": (hi - lo) / 2, "lo": lo, "hi": hi, "n": int(x.size)}


def fmt(v: float, d: int) -> str:
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def tnum(v: float, d: int) -> str:
    """Number for a text-mode table cell, with a true minus sign."""
    text = f"{v:.{d}f}"
    return rf"$-${text[1:]}" if text.startswith("-") else text


def cell(s: dict[str, float], d: int, bold: bool = False) -> str:
    if s["n"] == 0:
        return "--"
    body = tnum(s["mean"], d)
    if bold:
        body = rf"\textbf{{{body}}}"
    # Percentile intervals over three seeds are asymmetric, so print both endpoints.
    return rf"\ci{{{body}}}{{{tnum(s['lo'], d)}}}{{{tnum(s['hi'], d)}}}"


class Macros:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.values: dict[str, Any] = {}

    def add(self, name: str, value: float | int | str, d: int = 1) -> None:
        # TeX control-sequence names cannot contain digits: spell them out (60 -> SixZero, F3 -> FThree).
        name = "".join(DIGIT_WORDS.get(ch, ch) for ch in name)
        text = value if isinstance(value, str) else fmt(float(value), d)
        if not isinstance(value, str) and d == 0 and abs(float(value)) >= 1000:
            text = f"{int(round(float(value))):,}"  # thousands separator, as in the prose ("6,000")
        if text.startswith("-"):
            text = rf"\ensuremath{{{text}}}"  # typeset a true minus sign in text and math
        self.lines.append(rf"\newcommand{{\R{name}}}{{{text}}}")
        self.values[name] = value if isinstance(value, str) else float(value)


def per_seed(evs: list[dict[str, Any]], group: tuple[str, ...], metric: str) -> list[float]:
    return [group_metric(ev, group, metric) for ev in evs]


def main_table(res: dict, mac: Macros, out: Path, methods: list[str], name: str) -> None:
    metrics = (("apr", 1, max), ("sv", 2, min), ("cvar_sv", 2, min))
    stats = {m: {(g, k): ci(per_seed(res[m], conds, k)) for g, conds in GROUPS.items() for k, _, _ in metrics}
             for m in methods if m in res}
    best = {}
    for g in GROUPS:
        for k, _, pick in metrics:
            vals = {m: stats[m][(g, k)]["mean"] for m in stats if m in MAIN}
            vals = {m: v for m, v in vals.items() if not math.isnan(v)}
            best[(g, k)] = pick(vals, key=vals.get) if vals else None
    cvars = [stats[m][(g, "cvar_sv")]["mean"] for m in stats for g in GROUPS]
    budget = 0.5
    mac.add("CvarRatioMin", min(cvars) / budget, 1)
    mac.add("CvarRatioMax", max(cvars) / budget, 1)
    rows = []
    for m in methods:
        if m not in stats:
            continue
        cells = []
        for g in GROUPS:
            for k, d, _ in metrics:
                cells.append(cell(stats[m][(g, k)], d, bold=best[(g, k)] == m))
                mac.add(f"{MACRO_NAME[m]}{GROUP_MACRO[g]}{METRIC_MACRO[k]}", stats[m][(g, k)]["mean"], d)
                mac.add(f"{MACRO_NAME[m]}{GROUP_MACRO[g]}{METRIC_MACRO[k]}Ci", stats[m][(g, k)]["half"], d)
        mac.add(f"NSeedsMain{MACRO_NAME[m]}", len(res[m]), 0)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
        if m == "ricor" and BOUNDS[0] in methods:
            rows.append(r"\midrule")
    (out / f"{name}.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def ba_table(res: dict, mac: Macros, out: Path) -> None:
    """Burned area (% of burnable cells burnt or burning at episode end) per condition group."""
    rows = []
    for m in MAIN + BOUNDS:
        if m not in res:
            continue
        cells = []
        for g, conds in GROUPS.items():
            s = ci(per_seed(res[m], conds, "ba"))
            cells.append(cell(s, 1))
            mac.add(f"{MACRO_NAME[m]}{GROUP_MACRO[g]}Ba", s["mean"], 1)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
        if m == "ricor":
            rows.append(r"\midrule")
    (out / "tab_ba.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def tests_table(res: dict, mac: Macros, out: Path) -> dict:
    """Welch tests of RiCoR vs every baseline on APR, SV, CVaR per group; Holm over the whole family."""
    ref = res["ricor"]
    tests = []
    for m in ["h_mappo", "h_mappo_drop", "admac_h", "beta_rep", "hpsf"]:
        if m not in res:
            continue
        for g, conds in GROUPS.items():
            for k in ("apr", "sv", "cvar_sv"):
                w = welch(np.array(per_seed(ref, conds, k)), np.array(per_seed(res[m], conds, k)))
                diff = float(np.mean(per_seed(ref, conds, k)) - np.mean(per_seed(res[m], conds, k)))
                tests.append({"method": m, "group": g, "metric": k, "diff": diff, **w})
    adj = holm([t["p"] if not math.isnan(t["p"]) else 1.0 for t in tests])
    for t, p in zip(tests, adj, strict=True):
        t["p_holm"] = p
    rows = []
    for m in ["h_mappo", "h_mappo_drop", "admac_h", "beta_rep", "hpsf"]:
        sel = [t for t in tests if t["method"] == m]
        if not sel:
            continue
        cells = []
        for g in GROUPS:
            for k in ("apr", "sv", "cvar_sv"):
                t = next(x for x in sel if x["group"] == g and x["metric"] == k)
                d = 1 if k == "apr" else 2
                star = r"$^{*}$" if t["p_holm"] < 0.05 else ""
                cells.append(f"${t['diff']:+.{d}f}${star}")
                mac.add(f"Diff{MACRO_NAME[m]}{GROUP_MACRO[g]}{METRIC_MACRO[k]}", t["diff"], d)
                mac.add(f"P{MACRO_NAME[m]}{GROUP_MACRO[g]}{METRIC_MACRO[k]}", fmt_p(t["p_holm"]))
                mac.add(f"G{MACRO_NAME[m]}{GROUP_MACRO[g]}{METRIC_MACRO[k]}", t["g"], 2)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    (out / "tab_tests.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    n_sig = sum(t["p_holm"] < 0.05 for t in tests)
    mac.add("NTests", len(tests), 0)
    mac.add("NSig", n_sig, 0)
    valid = [t for t in tests if not math.isnan(t["p"])]
    mac.add("MaxAbsG", max(abs(t["g"]) for t in valid), 1)
    mac.add("MinRawP", min(t["p"] for t in valid), 3)
    mac.add("MinHolmP", min(t["p_holm"] for t in valid), 2)
    return {"tests": tests}


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    return r"<0.001" if p < 0.001 else f"{p:.3f}"


def calibration_table(res: dict, mac: Macros, out: Path) -> None:
    rows = []
    for m in ["admac_h", "beta_rep", "ricor_no_uncertainty", "ricor_no_split", "ricor_kappa0", "ricor"]:
        if m not in res:
            continue
        cal = [ev["calibration"].get("pooled", {}) for ev in res[m]]
        cal = [c for c in cal if c]
        if not cal:
            continue
        cells = []
        for k, d in (("ece", 3), ("brier", 3), ("auroc", 3), ("far", 1), ("frr", 1)):
            s = ci([c[k] for c in cal])
            cells.append(cell(s, d))
            mac.add(f"Cal{MACRO_NAME[m]}{k.capitalize()}", s["mean"], d)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    (out / "tab_calibration.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if "ricor" in res:
        mats = np.array([ev["calibration"]["pooled"]["confusion"] for ev in res["ricor"]])
        mean = mats.mean(0)
        names = ["Clean", "Channel", "Content"]
        rows = [f"{names[i]} & " + " & ".join(f"{v:.1f}" for v in mean[i]) + r" \\" for i in range(3)]
        (out / "tab_confusion.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
        for i, a in enumerate(["Clean", "Channel", "Content"]):
            mac.add(f"Conf{a}", mean[i, i], 1)


def recovery_and_modes(res: dict, mac: Macros, out: Path) -> None:
    rows = []
    for m in ["h_mappo", "h_mappo_drop", "admac_h", "beta_rep", "hpsf", "ricor_fixed", "ricor"]:
        if m not in res:
            continue
        evs = res[m]
        deg = ci([100 * (group_metric(e, ("clean",), "apr") - group_metric(e, COMPOUND, "apr"))
                  / group_metric(e, ("clean",), "apr") for e in evs])
        rec = [e.get("recovery", {}).get("recovery_steps", float("nan")) for e in evs]
        rec = [float("nan") if r is None else float(r) for r in rec]
        if m == "ricor":
            # Per-seed recovery time and pre-fault level (mean smoothed reward before the window).
            for e, r in zip(evs, rec, strict=True):
                curve = np.asarray(e["recovery"]["mean_reward_curve"], float)
                cs = np.concatenate([[0.0], np.cumsum(curve)])
                idx = np.arange(1, len(curve) + 1)
                sm = (cs[idx] - cs[np.maximum(0, idx - 10)]) / (idx - np.maximum(0, idx - 10))
                mac.add(f"RecSeed{'ABC'[e['seed']]}", r, 0)
                mac.add(f"PreSeed{'ABC'[e['seed']]}", float(sm[10:e["recovery"]["window"][0]].mean()), 3)
        never = sum(math.isnan(r) for r in rec)
        rs = ci(rec)
        occ = np.array([np.mean([e["conditions"][c]["summary"]["mode_occupancy"] for c in COMPOUND], 0)
                        for e in evs])
        fb = 100 * occ[:, 3].mean()
        modes = m in ("ricor", "ricor_fixed")
        rows.append(
            f"{LABEL[m]} & {cell(deg, 1)} & {cell(rs, 1)} & {never}/{len(evs)} & "
            + (f"{fb:.1f} & {100 * occ[:, 2].mean():.1f}" if modes else "-- & --")
            + r" \\"
        )
        mac.add(f"Deg{MACRO_NAME[m]}", deg["mean"], 1)
        mac.add(f"Rec{MACRO_NAME[m]}", rs["mean"], 1)
        mac.add(f"RecNever{MACRO_NAME[m]}", never, 0)
        if modes:
            mac.add(f"Fb{MACRO_NAME[m]}", fb, 1)
    (out / "tab_recovery.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Mode occupancy of RiCoR per condition group (Table: mode usage).
    if "ricor" in res:
        rows = []
        for g, conds in list(GROUPS.items()) + [("adaptive", ("U4",))]:
            occ = np.array([np.mean([e["conditions"][c]["summary"]["mode_occupancy"] for c in conds
                                     if c in e["conditions"]], 0) for e in res["ricor"]])
            name = {"clean": "Clean", "isolated": "Isolated (F1--F6)", "compound": "Compound (C1--C3)",
                    "held_out": "Held-out (U1--U3)", "adaptive": "Slow drift (U4)"}[g]
            rows.append(f"{name} & " + " & ".join(f"{100 * v:.1f}" for v in occ.mean(0)) + r" \\")
            for i, mname in enumerate(["N", "S", "V", "F"]):
                mac.add(f"Mode{GROUP_MACRO.get(g, 'Adapt')}{mname}", 100 * occ.mean(0)[i], 1)
        (out / "tab_modes.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Recovery curves: seed-mean per-step reward, trailing 10-step average.
    for m in ["h_mappo", "hpsf", "admac_h", "ricor"]:
        if m in res and all("recovery" in e for e in res[m]):
            curve = np.mean([e["recovery"]["mean_reward_curve"] for e in res[m]], 0)
            k = 10
            cs = np.concatenate([[0.0], np.cumsum(curve)])
            idx = np.arange(1, len(curve) + 1)
            lo = np.maximum(0, idx - k)
            sm = (cs[idx] - cs[lo]) / (idx - lo)
            _dat(out / f"recovery_{m}.dat", ["t", "r"], [(t, v) for t, v in enumerate(sm)])
    if "ricor" in res:
        w = res["ricor"][0]["recovery"]["window"]
        mac.add("RecStart", w[0], 0)
        mac.add("RecEnd", w[1], 0)


def ablation_table(res: dict, mac: Macros, out: Path) -> None:
    ref = res.get("ricor")
    rows = []
    tests = []
    for m in ["ricor"] + ABL:
        if m not in res:
            continue
        evs = res[m]
        row = {}
        for key, conds, k in (("apr", COMPOUND, "apr"), ("sv", COMPOUND, "sv"), ("cvar", COMPOUND, "cvar_sv"),
                              ("aprU", HELD_OUT, "apr")):
            row[key] = ci(per_seed(evs, conds, k))
            if m != "ricor" and ref:
                w = welch(np.array(per_seed(evs, conds, k)), np.array(per_seed(ref, conds, k)))
                tests.append((m, key, w))
        cal = [e["calibration"].get("pooled", {}).get("ece", float("nan")) for e in evs]
        row["ece"] = ci(cal)
        rows.append((m, row))
    adj = holm([t[2]["p"] if not math.isnan(t[2]["p"]) else 1.0 for t in tests]) if tests else []
    sig = {(t[0], t[1]): p < 0.05 for t, p in zip(tests, adj, strict=True)}
    pmap = {(t[0], t[1]): p for t, p in zip(tests, adj, strict=True)}
    lines = []
    for m, row in rows:
        cells = []
        for key, d in (("apr", 1), ("sv", 2), ("cvar", 2), ("ece", 3), ("aprU", 1)):
            star = r"$^{*}$" if sig.get((m, key)) else ""
            cells.append(cell(row[key], d) + star)
            mac.add(f"{MACRO_NAME[m]}Abl{key.capitalize()}", row[key]["mean"], d)
            if (m, key) in pmap:
                mac.add(f"P{MACRO_NAME[m]}Abl{key.capitalize()}", fmt_p(pmap[(m, key)]))
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
        if m == "ricor":
            lines.append(r"\midrule")
    (out / "tab_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mac.add("NAblTests", len(tests), 0)
    mac.add("NAblSig", sum(sig.values()), 0)


def compromised_table(res: dict, mac: Macros, out: Path) -> None:
    conds = [("K1", "1"), ("F6", "2"), ("U3", "3"), ("K4", "4"), ("U4", "2 (drift)")]
    lines = []
    for m in ["admac_h", "beta_rep", "hpsf", "ricor", "ricor_no_verify", "oracle_gate"]:
        if m not in res:
            continue
        cells = []
        for c, _ in conds:
            a = ci([condition_metric(e, c, "apr") for e in res[m] if c in e["conditions"]])
            s = ci([condition_metric(e, c, "sv") for e in res[m] if c in e["conditions"]])
            cells.append(f"{fmt(a['mean'], 1)} / {fmt(s['mean'], 2)}")
            mac.add(f"K{MACRO_NAME[m]}{c}Apr", a["mean"], 1)
            mac.add(f"K{MACRO_NAME[m]}{c}Sv", s["mean"], 2)
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    (out / "tab_compromised.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_seed_modes(res: dict, mac: Macros, out: Path) -> None:
    """Per-seed clean/compound APR and mode occupancy for the mode-switching variants."""
    lines = []
    for m in ["ricor", "ricor_fixed"]:
        if m not in res:
            continue
        mac.add(f"MeanClean{MACRO_NAME[m]}", float(np.mean([group_metric(e, ("clean",), "apr") for e in res[m]])), 1)
        for e in sorted(res[m], key=lambda e: e["seed"]):
            c = e["conditions"]
            occ = 100 * np.array(c["clean"]["summary"]["mode_occupancy"])
            occc = 100 * np.mean([c[k]["summary"]["mode_occupancy"] for k in COMPOUND], 0)
            apr_c = group_metric(e, ("clean",), "apr")
            apr_k = group_metric(e, COMPOUND, "apr")
            lines.append(
                f"{LABEL[m]} & {e['seed']} & {apr_c:.1f} & {apr_k:.1f} & "
                + " / ".join(f"{v:.0f}" for v in occ) + " & " + " / ".join(f"{v:.0f}" for v in occc) + r" \\"
            )
            mac.add(f"Seed{MACRO_NAME[m]}{'ABC'[e['seed']]}CleanApr", apr_c, 1)
            for i, mn in enumerate("NSVF"):
                mac.add(f"Seed{MACRO_NAME[m]}{'ABC'[e['seed']]}Clean{mn}", occ[i], 1)
                mac.add(f"Seed{MACRO_NAME[m]}{'ABC'[e['seed']]}Comp{mn}", occc[i], 1)
    (out / "tab_seedmodes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def overhead_table(res: dict, mac: Macros, out: Path) -> None:
    lines = []
    for m in MAIN + BOUNDS:
        if m not in res:
            continue
        evs = res[m]
        params = _params(evs[0]["_seed_dir"])
        lat = ci([group_metric(e, COMPOUND, "latency_ms") for e in evs])
        train_h = ci([sum(h["seconds"] for h in e["_history"]) / 3600 for e in evs])
        comm = ci([group_metric(e, COMPOUND, "comm_kb") for e in evs])
        commc = ci([group_metric(e, ("clean",), "comm_kb") for e in evs])
        lines.append(f"{LABEL[m]} & {cell(lat, 2)} & {cell(train_h, 2)} & "
                     f"{fmt(commc['mean'], 0)} & {fmt(comm['mean'], 0)}" + r" \\")
        mac.add(f"Params{MACRO_NAME[m]}", params, 3)
        mac.add(f"Lat{MACRO_NAME[m]}", lat["mean"], 2)
        mac.add(f"Train{MACRO_NAME[m]}", train_h["mean"], 2)
        mac.add(f"Comm{MACRO_NAME[m]}", comm["mean"], 0)
        mac.add(f"CommClean{MACRO_NAME[m]}", commc["mean"], 0)
    (out / "tab_overhead.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _params(seed_dir: str) -> float:
    import torch

    from ricor.training import load_system

    system, _ = load_system(Path(seed_dir) / "checkpoint.pt")
    used = sum(p.numel() for p in system.parameters())
    del torch
    return used / 1e6


def sweeps_and_curves(res: dict, mac: Macros, out: Path) -> None:
    for m in MAIN + BOUNDS:
        if m not in res:
            continue
        evs = [e for e in res[m] if e.get("loss_sweep")]
        if evs:
            keys = sorted(evs[0]["loss_sweep"], key=float)
            pts = []
            for k in keys:
                s = ci([e["loss_sweep"][k]["apr"] for e in evs])
                pts.append((100 * float(k), s["mean"], s["half"]))
                mac.add(f"Sweep{MACRO_NAME[m]}{int(round(100 * float(k)))}", s["mean"], 1)
            _dat(out / f"sweep_{m}.dat", ["loss", "apr", "ci"], pts)
        pts = []
        for i, c in enumerate(ISOLATED):
            s = ci([condition_metric(e, c, "apr") for e in res[m]])
            pts.append((i, s["mean"], s["half"]))
            mac.add(f"Fault{MACRO_NAME[m]}{c}", s["mean"], 1)
        _dat(out / f"perfault_{m}.dat", ["fault", "apr", "ci"], pts)
        hists = [e["_history"] for e in res[m] if e["_history"]]
        if hists:
            n = min(len(h) for h in hists)
            steps_per_it = None
            for key in ("ret", "apr", "sv", "lambda", "cvar_cost"):
                arr = np.array([[h[i][key] for i in range(n)] for h in hists])
                mean = arr.mean(0)
                k = 5
                sm = np.convolve(mean, np.ones(k) / k, mode="valid")
                _dat(out / f"learn_{m}_{key}.dat", ["it", key], [(i + k - 1, v) for i, v in enumerate(sm)])
            mac.add(f"Iters{MACRO_NAME[m]}", n, 0)
            del steps_per_it


def _dat(path: Path, header: list[str], rows: list[tuple]) -> None:
    lines = [" ".join(header)] + [" ".join(f"{v:.6g}" for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/revision")
    p.add_argument("--out", default="../results")
    args = p.parse_args()
    res = load(Path(args.root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mac = Macros()
    mac.add("NSeeds", min(len(v) for v in res.values()), 0)
    mac.add("NSeedsMax", max(len(v) for v in res.values()), 0)
    any_ev = next(iter(res.values()))[0]
    mac.add("NEpisodes", any_ev["eval_config"]["episodes_per_condition"], 0)
    # Scale of training-episode returns (all iterations of all runs), used to judge the mode-selector penalties.
    rets = np.array([h["ret"] for evs in res.values() for e in evs for h in e["_history"]])
    steps = any_ev["eval_config"].get("episode_length", None) or 200
    for name, q in (("RetMin", 0), ("RetPTen", 10), ("RetMedian", 50), ("RetPNinety", 90), ("RetMax", 100)):
        mac.add(name, float(np.percentile(rets, q)), 1 if name != "RetMax" else 2)
    mac.add("RetPTenStep", float(np.percentile(rets, 10)) / steps, 3)
    mac.add("RetPNinetyStep", float(np.percentile(rets, 90)) / steps, 3)
    mac.add("NRetIters", len(rets), 0)
    # Training CVaR at the end of training (mean of the last 5 iterations of each run) relative to d = 0.5.
    end_cvar = [np.mean([h["cvar_cost"] for h in e["_history"][-5:]]) for evs in res.values() for e in evs if e["_history"]]
    mac.add("TrainCvarRatioMin", min(end_cvar) / 0.5, 0)
    mac.add("TrainCvarRatioMax", max(end_cvar) / 0.5, 0)
    main_table(res, mac, out, MAIN + BOUNDS, "tab_main")
    ba_table(res, mac, out)
    extra = {}
    if "ricor" in res:
        extra = tests_table(res, mac, out)
        calibration_table(res, mac, out)
        ablation_table(res, mac, out)
    recovery_and_modes(res, mac, out)
    per_seed_modes(res, mac, out)
    compromised_table(res, mac, out)
    overhead_table(res, mac, out)
    sweeps_and_curves(res, mac, out)
    (out / "numbers.tex").write_text("\n".join(mac.lines) + "\n", encoding="utf-8")
    summary = {"n_seeds": {m: len(v) for m, v in res.items()}, "macros": mac.values, **extra}
    (out / "artifacts_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"wrote {len(mac.lines)} macros for {len(res)} methods to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
