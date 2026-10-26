"""Evaluation protocol: fault conditions, loss sweep, recovery, calibration (paper Sec. VI-E, VII)."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from ricor.comm.faults import evaluation_spec
from ricor.config import ExperimentConfig
from ricor.evaluation.metrics import recovery_time, reliability_report
from ricor.scenarios import split_seeds
from ricor.system import RiCoRSystem
from ricor.utils import get_logger

log = get_logger("ricor.evaluation")


def _channel_seed(tag: str, seed: int, i: int) -> int:
    return (zlib.crc32(tag.encode()) + 1_000_003 * seed + i) % (2**31 - 1)


def run_condition(
    system: RiCoRSystem,
    cfg: ExperimentConfig,
    condition: str,
    episodes: int,
    batch: int,
    loss_override: float | None = None,
    record_reliability: bool = True,
) -> dict[str, Any]:
    """Evaluate ``episodes`` test maps under one fault condition; returns per-episode and pooled metrics."""
    maps = split_seeds(cfg.scenarios)[cfg.eval.split][:episodes]
    if len(maps) < episodes:
        raise ValueError(f"split {cfg.eval.split!r} has only {len(maps)} maps, {episodes} requested")
    spec = evaluation_spec(condition, loss_override)
    tag = condition if loss_override is None else f"{condition}@{loss_override}"
    per_ep: list[dict[str, Any]] = []
    recs: dict[str, list[np.ndarray]] = {}
    for start in range(0, episodes, batch):
        chunk = maps[start : start + batch]
        seeds = [_channel_seed(tag, cfg.seed, start + i) for i in range(len(chunk))]
        roll = system.run_episodes(
            chunk, [spec] * len(chunk), seeds, stochastic=False, record_reliability=record_reliability
        )
        per_ep += roll.episode_metrics
        for k, v in roll.reliability_records.items():
            recs.setdefault(k, []).append(v)
    pooled = {k: np.concatenate(v) for k, v in recs.items()}
    summary = {
        k: float(np.mean([m[k] for m in per_ep]))
        for k in (
            "apr",
            "ba",
            "sv",
            "return",
            "comm_kb",
            "queries",
            "shield_interventions",
            "latency_ms",
            "entrapment",
            "drops_near_crew",
        )
    }
    summary["mode_occupancy"] = np.mean([m["mode_occupancy"] for m in per_ep], 0).tolist()
    return {"condition": tag, "summary": summary, "episodes": per_ep, "reliability": pooled}


def run_recovery(system: RiCoRSystem, cfg: ExperimentConfig, episodes: int, batch: int) -> dict[str, Any]:
    """Inject ``recovery_condition`` inside ``recovery_window`` and measure recovery time (Fig. 7, Table VI)."""
    ec = cfg.eval
    window = tuple(ec.recovery_window)
    if window[1] + ec.recovery_smoothing >= cfg.env.episode_length:
        raise ValueError("recovery_window must end before episode_length - recovery_smoothing")
    maps = split_seeds(cfg.scenarios)[ec.split][:episodes]
    spec = evaluation_spec(ec.recovery_condition)
    curves = []
    for start in range(0, len(maps), batch):
        chunk = maps[start : start + batch]
        seeds = [_channel_seed("recovery", cfg.seed, start + i) for i in range(len(chunk))]
        roll = system.run_episodes(chunk, [spec] * len(chunk), seeds, stochastic=False, windows=[window] * len(chunk))
        curves.append(roll.step_rewards)
    curve = np.concatenate(curves, axis=1).mean(1)
    return {
        "condition": ec.recovery_condition,
        "window": list(window),
        "mean_reward_curve": curve.tolist(),
        "recovery_steps": recovery_time(curve, window, ec.recovery_smoothing),
    }


def evaluate(
    system: RiCoRSystem,
    cfg: ExperimentConfig,
    batch: int | None = None,
    include_sweep: bool = True,
    include_recovery: bool = True,
) -> dict[str, Any]:
    """Full protocol for one trained system (one method, one seed)."""
    ec = cfg.eval
    mc = cfg.method
    batch = batch or cfg.train.n_envs
    system.eval()
    # Calibration is reported only for methods whose gate reads a reliability score.
    scored = mc.gating and not mc.oracle_gate and not mc.no_comm
    results: dict[str, Any] = {"conditions": {}, "loss_sweep": {}, "calibration": {}}
    pooled: dict[str, list[np.ndarray]] = {}
    for cond in ec.conditions:
        log.info("evaluating %s on %s (%d episodes)", cfg.method.name, cond, ec.episodes_per_condition)
        r = run_condition(system, cfg, cond, ec.episodes_per_condition, batch, record_reliability=scored)
        results["conditions"][cond] = {"summary": r["summary"], "episodes": r["episodes"]}
        if scored:
            results["calibration"][cond] = reliability_report(r["reliability"], ec.ece_bins)
            for k, v in r["reliability"].items():
                pooled.setdefault(k, []).append(v)
    if scored:
        results["calibration"]["pooled"] = reliability_report(
            {k: np.concatenate(v) for k, v in pooled.items()}, ec.ece_bins
        )
    if include_sweep:
        for p in ec.loss_sweep:
            r = run_condition(
                system, cfg, "F1", ec.episodes_per_condition, batch, loss_override=p, record_reliability=False
            )
            results["loss_sweep"][str(p)] = r["summary"]
    if include_recovery:
        results["recovery"] = run_recovery(system, cfg, ec.recovery_episodes, batch)
    return results
