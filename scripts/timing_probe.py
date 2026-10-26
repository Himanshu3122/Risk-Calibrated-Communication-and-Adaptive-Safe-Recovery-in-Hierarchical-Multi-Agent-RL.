"""Timing probe behind the compute estimate in the manuscript (Sec. 6.4) and docs/reproducibility.md.

Measures, for one method, the wall-clock seconds of one training iteration split into rollout
(simulation + inference) and PPO update, and reports both per environment step.

Usage (from the Github/ folder, one thread):
    python scripts/timing_probe.py ricor paper 100 16
Arguments: method name (configs/methods/<name>.yaml), geometry ("paper" = 48x48 grid, 8 regions;
"reduced" = 32x32 grid), episode length T, parallel environments B.

Measured on 2026-10-25 (Intel Core Ultra 5 225U, torch 2.5.1+cpu, one thread):
    ricor   paper   T=100 B=16: rollout 7.16 ms/env-step, update 13.6 s per 1,600 steps (~8.5 ms/env-step)
    ricor   reduced T=100 B=16: rollout 4.74 ms/env-step, update 13.2 s
    h_mappo reduced T=100 B=16: rollout 2.33 ms/env-step, update 7.7 s
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ricor.comm.faults import sample_training_spec
from ricor.config import load_config
from ricor.training import Trainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    torch.set_num_threads(1)
    method, geom, T, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with open(ROOT / "configs" / "methods" / f"{method}.yaml", encoding="utf-8") as fh:
        mcfg = yaml.safe_load(fh)
    env = {"episode_length": T}
    if geom == "reduced":
        env.update(height=32, width=32, region_rows=4, region_cols=2, n_drops=4, assets_min=8, assets_max=14,
                   ignitions_min=2, ignitions_max=3)
    cfg = load_config(ROOT / "configs" / "experiments" / "full.yaml",
                      {"method": mcfg, "env": env, "train": {"n_envs": B, "total_steps": B * T}})
    tr = Trainer(cfg, tempfile.mkdtemp(prefix="ricor_probe_"))
    rng = np.random.default_rng(0)
    maps = rng.choice(tr.splits["train"], size=B).tolist()
    faults = [sample_training_spec(2, rng, tr.system.N) for _ in range(B)]
    t0 = time.perf_counter()
    tr.system.eval()
    roll = tr.system.run_episodes(maps, faults, list(range(B)), stochastic=True, training=True)
    t1 = time.perf_counter()
    tr.system.train()
    tr.update(roll.data, 0.5)
    t2 = time.perf_counter()
    n = B * T
    print(f"{method} {geom} T={T} B={B}: rollout {t1 - t0:.1f}s ({(t1 - t0) / n * 1000:.2f} ms/env-step), "
          f"update {t2 - t1:.1f}s ({(t2 - t1) / n * 1000:.2f} ms/env-step), "
          f"params {sum(p.numel() for p in tr.system.parameters()) / 1e6:.3f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
