"""Run the reduced-budget study of the revised manuscript: train + evaluate every method and seed.

Usage:
    python scripts/run_study.py --workers 9 [--seeds 0 1 2] [--ablation-seeds 0 1 2] [--only ricor hpsf]

Jobs whose ``eval.json`` already exists are skipped, so the script can be restarted. Each job is a
separate process (one PyTorch thread each). Logs go to ``<output>/<name>/<method>/seed<k>/*.log``.
Main methods and reference bounds get the full protocol (all conditions, loss sweep, recovery);
ablations get all conditions and recovery but no loss sweep.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["ricor", "h_mappo", "h_mappo_drop", "admac_h", "hpsf", "oracle_gate", "no_comm", "beta_rep"]
ABLATIONS = {
    "no_split": "ricor_no_split",
    "no_uncertainty": "ricor_no_uncertainty",
    "kappa0": "ricor_kappa0",
    "fixed_modes": "ricor_fixed",
    "no_verify": "ricor_no_verify",
    "no_shield": "ricor_no_shield",
}


def jobs(
    config: str, seeds: list[int], ablation_seeds: list[int], only: list[str] | None
) -> list[tuple[str, str, int, bool]]:
    out = []
    for m in MAIN:
        for s in seeds:
            out.append((str(ROOT / "configs" / "methods" / f"{m}.yaml"), m, s, True))
    for a, name in ABLATIONS.items():
        for s in ablation_seeds:
            out.append((str(ROOT / "configs" / "methods" / "ablations" / f"{a}.yaml"), name, s, False))
    if only:
        out = [j for j in out if j[1] in only]
    # Interleave seeds first so that every method has early seeds finished.
    return sorted(out, key=lambda j: (j[2], j[1]))



def training_active(method_yaml: str, seed: int) -> bool:
    """True if a train.py process for this method and seed is already running (e.g. left by an earlier runner)."""
    if sys.platform != "win32":
        return False
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True, check=False,
    ).stdout
    pat = re.compile(re.escape(Path(method_yaml).name) + r"['\"]?\s+--seed\s+['\"]?" + str(seed) + r"\b")
    return any("train.py" in line and pat.search(line) for line in out.splitlines())


def run(job: tuple[str, str, int, bool], config: str, out_root: Path, name: str) -> str:
    method_yaml, method, seed, full = job
    run_dir = out_root / name / method / f"seed{seed}"
    if (run_dir / "eval.json").is_file():
        return f"skip {method} seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RICOR_TORCH_THREADS": "1"}
    t0 = time.time()
    ckpt = run_dir / "checkpoint.pt"
    # Adopt a training run that is still in progress instead of starting a duplicate.
    while not ckpt.is_file() and training_active(method_yaml, seed):
        time.sleep(60)
    if not ckpt.is_file():
        with open(run_dir / "train.log", "w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", config, "--method", method_yaml,
                 "--seed", str(seed), "--set", f"output_dir={out_root}"],
                stdout=log, stderr=subprocess.STDOUT, env=env, check=True, cwd=ROOT,
            )
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--checkpoint", str(ckpt)]
    if not full:
        cmd.append("--no-sweep")
    with open(run_dir / "eval.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, cwd=ROOT)
    return f"done {method} seed{seed} in {(time.time() - t0) / 60:.1f} min"


def keep_awake() -> None:
    """On Windows, ask the OS not to sleep while this process runs (released automatically on exit)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> int:
    keep_awake()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs" / "experiments" / "revision.yaml"))
    p.add_argument("--name", default="revision")
    p.add_argument("--output", default=str(ROOT / "runs"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    todo = jobs(args.config, args.seeds, args.ablation_seeds, args.only)
    print(f"{len(todo)} jobs, {args.workers} workers", flush=True)
    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j, args.config, Path(args.output), args.name) for j in todo]
        for f in futures:
            try:
                print(f.result(), flush=True)
            except subprocess.CalledProcessError as exc:
                failures += 1
                print(f"FAILED: {exc}", flush=True)
    print(f"finished with {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
