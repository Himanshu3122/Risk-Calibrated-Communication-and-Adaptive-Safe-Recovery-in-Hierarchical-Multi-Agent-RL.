# Evaluation

## Command

```bash
python scripts/evaluate.py --checkpoint runs/<name>/<method>/seed<k>/checkpoint.pt [--no-sweep] [--no-recovery] [--set eval.episodes_per_condition=100]
```

It writes `eval.json` next to the checkpoint, unless `--out` is given. Policies act greedily: crew actions use the arg-max, and drops use largest-remainder rounding of the coordinator's probabilities.

## Protocol (paper Sec. VI-E)

| Part | Implementation |
|---|---|
| Conditions | `eval.conditions`: clean, F1–F6, C1–C3, U1–U3 (Table II parameters, `comm/faults.py::evaluation_spec`) |
| Episodes | The first `episodes_per_condition` maps of the test split (1,200 in the paper config). The same maps are used for every method and seed |
| Loss sweep | F1 at each `eval.loss_sweep` rate (Fig. 4) |
| Recovery | `recovery_condition` (C2) active only inside `recovery_window` (default [100, 250]); mean per-step reward curve and recovery time (Fig. 7, Table VI) |
| Calibration | Per condition and pooled: ECE (15 bins), Brier, AUROC, FAR, FRR of `ρ̄` against `y_ch · y_cr`, plus a 3-way diagnosis confusion matrix |

## Metrics in `eval.json`

Per condition, `summary` holds the episode means and `episodes` holds the per-episode values:

- `apr`: asset protection rate (%)
- `ba`: burned area (%)
- `sv`: safety violations per episode (entrapment onsets + drops near crews)
- `entrapment`, `drops_near_crew`: the two components of `sv`
- `return`: team reward
- `comm_kb`: bytes transmitted, including verification queries
- `queries`: number of verification queries
- `shield_interventions`: number of shield overrides
- `mode_occupancy`: share of steps in N, S, V, F
- `latency_ms`: wall-clock decision time per step for the whole team. This is the Python reference loop, not an optimized deployment

## Aggregation and statistics

```bash
python scripts/aggregate.py --root runs/<name> --reference hpsf
```

This reads every `<method>/seed<k>/eval.json` and writes `summary.json` and `summary.md`. For each method and each group (clean, isolated = F1–F6, compound = C1–C3, held-out = U1–U3), it reports the seed-level mean with a 95% bootstrap CI (the unit is the seed), and Welch's t-test with Hedges' g against the reference method, Holm-corrected across all comparisons.

## Baselines

See [training.md](training.md). The baselines are reimplementations with documented approximations (assumptions A16–A18), not the original authors' code.

## Not provided as scripts

The κ sweep (Fig. 8) and the scalability study (Table VIII) are config sweeps: `--set method.kappa=...`, or `--set env.region_rows=... env.region_cols=...`. Run them, then aggregate.
