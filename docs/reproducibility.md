# Reproducibility

Two different things are often both called "reproducibility":

| Kind | Status |
|---|---|
| **Software pipeline:** the code runs the same way for anyone | **Yes.** Pinned requirements, YAML configs, deterministic map split, seeded runs, run metadata (versions + git SHA), and 52 automated tests including an end-to-end smoke test |
| **Scientific results of the revised manuscript** | **Yes, at the reduced budget.** Every number, table, and plot in the revised manuscript is generated from the runs below by `scripts/make_paper_artifacts.py` |
| **Planned paper-scale study** (10⁷ steps, 8 seeds, 1,200 test episodes) | **Not run** (compute; see below). The first submission's numbers were illustrative targets and have been removed from the manuscript |

## Recipe for the reduced-budget study (revised manuscript)

```bash
python scripts/run_study.py --workers 9                     # 14 configurations x 3 seeds
python scripts/make_paper_artifacts.py --root runs/revision --out ../results
python scripts/honest_majority.py                           # Table "honest majority" (exact combinatorics)
```

`configs/experiments/revision.yaml` keeps the paper geometry (48×48 grid, 8 regions, 3 crews) and the Table III hyperparameters, and reduces: episode length 400 → 200, parallel environments 32 → 16, training budget 10⁷ → 2×10⁵ environment steps (63 PPO iterations), seeds 8 → 3, test episodes 1,200 → 30 per condition and seed. `run_study.py` is resumable (finished jobs with an `eval.json` are skipped), runs each job in its own single-threaded process, and asks Windows not to sleep while it runs.

Measured on an Intel Core Ultra 5 225U laptop (12 cores, 16 GB, CPU only) with 9 concurrent jobs: 65–140 s per PPO iteration depending on method and load. Per-job durations are logged in `runs/revision_study.log`.

## Recipe for the planned paper-scale study (not run)

```bash
for m in ricor h_mappo h_mappo_drop admac_h hpsf beta_rep oracle_gate no_comm; do
  for s in 0 1 2 3 4 5 6 7; do
    python scripts/train.py --config configs/experiments/full.yaml --method configs/methods/$m.yaml --seed $s
    python scripts/evaluate.py --checkpoint runs/full/$m/seed$s/checkpoint.pt
  done
done
python scripts/aggregate.py --root runs/full --reference ricor
```

## Compute estimate

A timing probe at paper geometry (16 parallel environments, 100-step episodes, one thread) measured, for RiCoR, about 7.2 ms per environment step for simulation and inference and about 8.5 ms per environment step for the PPO update (10 epochs, 4 minibatches). 10⁷ steps therefore need more than 40 CPU-hours per method-seed; H-MAPPO is roughly a third cheaper. The figure of 14.7 h on two GPUs in the first submission was an illustrative target, not a measurement.

## Known sources of divergence from the first submission

1. Fire dynamics and map generator (A1); safety-cost counting (A4); report fields (A5).
2. Network sizes: about 0.41 M parameters at paper scale (the first submission's table listed an illustrative 1.86 M).
3. Unreported loss weights (A7), heuristic adversary (A9), fault rates (A8, A10), verification protocol (A12, revised).
4. Baseline reimplementations (A16–A18) and the added Beta-reputation baseline (A31).
5. HCN not implemented and no longer in the manuscript (A28).

## Determinism

`set_seed` seeds Python, NumPy, and PyTorch and requests deterministic kernels (`warn_only=True`). Rollouts are deterministic for a fixed seed on the same platform and library versions. Bit-identical results across operating systems or BLAS builds are not guaranteed.
