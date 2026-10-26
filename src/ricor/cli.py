"""Command-line entry points (``ricor-train``, ``ricor-evaluate``, ``ricor-infer``, ``ricor-aggregate``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ricor.comm.faults import evaluation_spec
from ricor.config import ConfigError, deep_merge, load_config, parse_override
from ricor.scenarios import split_seeds
from ricor.utils import get_logger, make_run_dir, safe_child, set_seed, write_json

log = get_logger("ricor.cli")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="experiment YAML (e.g. configs/experiments/smoke.yaml)")
    p.add_argument("--method", default=None, help="method YAML to merge (e.g. configs/methods/ricor.yaml)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", action="append", default=[], metavar="a.b=value", help="override a config value")


def _load(args: argparse.Namespace) -> Any:
    overrides: dict[str, Any] = {}
    if args.method:
        import yaml

        with open(args.method, encoding="utf-8") as fh:
            overrides = deep_merge(overrides, {"method": yaml.safe_load(fh) or {}})
    if args.seed is not None:
        overrides = deep_merge(overrides, {"seed": args.seed})
    for text in args.set:
        overrides = deep_merge(overrides, parse_override(text))
    return load_config(args.config, overrides)


def train_main(argv: list[str] | None = None) -> int:
    from ricor.training import Trainer

    p = argparse.ArgumentParser(description="Train RiCoR or a baseline")
    _common(p)
    args = p.parse_args(argv)
    try:
        cfg = _load(args)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    run_dir = make_run_dir(cfg.output_dir, cfg.name, cfg.method.name, cfg.seed)
    write_json(run_dir / "config.json", cfg.to_dict())
    path = Trainer(cfg, run_dir).train()
    print(path)
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    from ricor.evaluation.runner import evaluate
    from ricor.training import load_system

    p = argparse.ArgumentParser(description="Evaluate a checkpoint under the paper's fault conditions")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--config", default=None, help="optional YAML whose eval/env-independent settings override the checkpoint's"
    )
    p.add_argument("--set", action="append", default=[], metavar="a.b=value")
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--no-recovery", action="store_true")
    p.add_argument("--out", default=None, help="output JSON (default: eval.json next to the checkpoint)")
    args = p.parse_args(argv)
    system, cfg = load_system(args.checkpoint)
    if args.config or args.set:
        from ricor.config import from_dict

        data = cfg.to_dict()
        if args.config:
            import yaml

            with open(args.config, encoding="utf-8") as fh:
                data = deep_merge(data, {"eval": (yaml.safe_load(fh) or {}).get("eval", {})})
        for text in args.set:
            data = deep_merge(data, parse_override(text))
        cfg = from_dict(data)
        system.cfg = cfg
    set_seed(cfg.seed)
    results = evaluate(system, cfg, include_sweep=not args.no_sweep, include_recovery=not args.no_recovery)
    results["method"], results["seed"] = cfg.method.name, cfg.seed
    results["eval_config"] = cfg.to_dict()["eval"]
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("eval.json")
    write_json(out, results)
    print(out)
    return 0


def infer_main(argv: list[str] | None = None) -> int:
    from ricor.training import load_system

    p = argparse.ArgumentParser(description="Run one episode with a checkpoint and write a step trace")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", default="C2", help="fault condition (clean, F1..F6, C1..C3, U1..U3)")
    p.add_argument("--map-index", type=int, default=0, help="index into the test split")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    system, cfg = load_system(args.checkpoint)
    maps = split_seeds(cfg.scenarios)["test"]
    if not 0 <= args.map_index < len(maps):
        log.error("map-index must lie in [0, %d)", len(maps))
        return 2
    try:
        spec = evaluation_spec(args.condition)
    except KeyError as exc:
        log.error("%s", exc)
        return 2
    roll = system.run_episodes(
        [maps[args.map_index]], [spec], [args.map_index], stochastic=False, record_reliability=True
    )
    rec = roll.reliability_records
    trace = {
        "method": cfg.method.name,
        "condition": args.condition,
        "map_seed": maps[args.map_index],
        "episode": roll.episode_metrics[0],
        "step_rewards": roll.step_rewards[:, 0].tolist(),
        "messages": {k: {"mean": float(np.mean(v)) if len(v) else None, "n": int(len(v))} for k, v in rec.items()},
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(f"infer_{args.condition}.json")
    write_json(out, trace)
    print(out)
    return 0


def aggregate_main(argv: list[str] | None = None) -> int:
    from ricor.evaluation.aggregate import collect, summarize, to_markdown

    p = argparse.ArgumentParser(description="Aggregate eval.json files across methods and seeds")
    p.add_argument("--root", required=True, help="directory containing <method>/seed<k>/eval.json")
    p.add_argument("--reference", default=None, help="method for Welch tests (e.g. hpsf)")
    p.add_argument("--out", default=None, help="output directory (default: <root>)")
    args = p.parse_args(argv)
    results = collect(args.root)
    if not results:
        log.error("no eval.json files under %s", args.root)
        return 2
    first = next(iter(results.values()))[0]
    n_boot = int(first.get("eval_config", {}).get("bootstrap_samples", 10000))
    table = summarize(results, args.reference, n_boot=n_boot)
    out_dir = Path(args.out) if args.out else Path(args.root)
    write_json(safe_child(out_dir, "summary.json"), table)
    (out_dir / "summary.md").write_text(to_markdown(table), encoding="utf-8")
    print(out_dir / "summary.md")
    return 0


def scenarios_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the map-seed split manifest")
    _common(p)
    p.add_argument("--out", default="data/scenarios")
    args = p.parse_args(argv)
    cfg = _load(args)
    split = split_seeds(cfg.scenarios)
    path = safe_child(args.out, f"{cfg.name}_manifest.json")
    write_json(path, {"scenarios": cfg.to_dict()["scenarios"], "splits": split})
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(train_main())
