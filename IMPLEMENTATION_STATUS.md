# Implementation Status

Source of truth: `main.tex` (methods) and `PROJECT_REPORT.md` (engineering specification). Assumption IDs (A1–A29) refer to [docs/assumptions.md](docs/assumptions.md).

## Fully Implemented

Implemented as the paper specifies, with no assumption that changes the meaning.

- **Equations (4), (5), (6):** evidential Beta posterior, variance, factorized reliability and lower confidence bound, including the joint-head and point-estimate ablation switches (`core/reliability.py`).
- **Equation (8):** risk-conditioned threshold and sigmoid gate (`core/gate.py`).
- **Equation (9):** multi-head gated attention with log-weight logits and a null slot (`models.GatedAttention`).
- **Equation (10):** mode-selector reward with switching and query costs; hysteresis with always-admissible fallback (`core/modes.py`).
- **Equation (11):** one-step shield with most-probable-safe replacement and a fallback action (`core/shield.py`).
- **Equation (12):** reliability loss with KL shrinkage of misleading evidence (`core/reliability.py`, `training.py`).
- **Equations (13), (14):** PPO-Lagrangian objective, empirical CVaR, projected λ update (`training.py`). The loss weights c_val, c_c, c_r are assumed (A7).
- **Equation (2) channel model:** Gilbert–Elliott loss, Bernoulli loss, delay queues, observable metadata (arrival, age, gap), content transforms, fault labels (`comm/channel.py`).
- **Environment geometry:** 48×48 grid, 8 regions of 12×24, 3 crews each, 6 drops every K = 5 steps, 400 steps, 12–20 assets, the two safety predicates (`envs/hierfire.py`, `configs/env/hierfire.yaml`).
- **Map library and split:** 6,000 maps, 4,200 / 600 / 1,200 by seed (`scenarios.py`).
- **Curriculum:** clean → isolated → compound; temperature calibration on validation maps.
- **Table III hyperparameters** (`configs/training/full.yaml`, `configs/methods/ricor.yaml`).
- **Metrics:** APR, BA, SV, ECE (15 bins), Brier, AUROC, FAR, FRR, degradation, communication volume, latency, mode occupancy.
- **Statistics:** seed-level bootstrap CIs, Welch's t-test (plus Hedges' g and Holm correction, which go beyond the paper).
- **Evaluation protocol:** 13 conditions, loss sweep, recovery window, pooled calibration.
- **Ablation switches** for all six Table VII variants.
- **Checkpointing, run metadata, CLI, configs, tests, CI.**

## Partially Implemented

Present and working, but based on assumptions or approximations.

| Component | Gap | Assumption |
|---|---|---|
| Fire dynamics and map generator | Constants and distributions not published | A1 |
| Crew actions, drop mechanics, report fields | Underspecified | A2, A3, A5 |
| Safety-cost counting | Event definition | A4 |
| Network architectures | Only widths given | A6 |
| Decision sensitivity, Eq. (7) | max over a factored joint action approximated per crew; reference weights for the "full" policy | A11, A19 |
| Distributional cost critics | Parameterization not given | A11 |
| F6 misleading adversary | Fixed heuristic (now stated as such in the manuscript); slow-drift variant U4 added | A9 |
| Fault rates (F2 bad-state share, F4/F5 shares, U1/U2 details) | Now stated in the manuscript (Table 4) | A8, A10 |
| Verification (mode V) | Border-view protocol defined in revision and stated in the manuscript | A12, A30 |
| Mode effects S and F; outage measure | Described qualitatively | A20 |
| Credibility inputs | Described qualitatively | A24 |
| Baselines ADMAC-H, HPSF, H-MAPPO+Drop | Reimplementations without original code | A16–A18 |
| Recovery time | Reward sign handling | A15 |
| Diagnosis confusion | 3-way instead of the paper's 4-way | A14 |

## Not Implemented

| Component | Why |
|---|---|
| HCN benchmark (Sec. VI-A, generalization paragraph) | Observation, reward, and fault details are not specified (A28) |
| Pretrained weights | None exist |
| Paper-scale study (10⁷ steps, 8 seeds) | Compute. The revised manuscript instead reports the reduced-budget study (`scripts/run_study.py`, `configs/experiments/revision.yaml`) |
| κ sweep and N-scalability study | Not run in the revision and removed from the manuscript; expressible as config sweeps (`--set`) |
| Learned, gate-aware adversary; ensemble/GP uncertainty baseline | Not implemented (listed as limitations in the manuscript) |

## Added in revision (2026-10-25)

- Privileged information removed from execution: decentralized coordinator cost critic (`RiCoRSystem.coord_local`, `coord_cost`), border-view verification (`HierFireEnv.border_view`), per-receiver credibility median (`_receiver_median`); test `test_execution_never_reads_the_global_state`.
- Eq. 7 tail term = expected CVaR of the induced action (regional) / coordinator-local critic at expected allocation (coordinator).
- Audit fixes: discounted cost-critic targets, per-region drop targets, verification queries no longer advance the Gilbert–Elliott state, decision-only latency, causal recovery smoothing, FAR/FRR at the operating gate, calibration omitted for methods without a reliability score, joint-head calibration label, dead `load_metadata` removed, `bootstrap_samples` honored by `ricor-aggregate`.
- New configurations `oracle_gate`, `no_comm`, `beta_rep`; new conditions U4 (slow drift), K1, K4; CVaR₀.₉ of per-episode cost in aggregation.
- Scripts: `run_study.py`, `make_paper_artifacts.py`, `honest_majority.py`.
| Web API / UI | Not useful for an offline research pipeline |

## Technical Debt

- **Speed:** the episode loop and channel are Python and NumPy per environment, and the shield runs per crew. About 0.02 s per env step at paper scale on CPU. Vectorizing across environments is the main optimization.
- **Leave-one-out sensitivity** runs M + 1 regional and N + 1 coordinator forward passes per step (O(N²) per team step).
- **Reliability heads are detached from the policy loss.** This is equivalent to Eq. (13) under A25, but prevents end-to-end shaping of the gate.
- **PPO trains on pre-shield actions (A22).** Feeding the shield mask into the logits is an alternative worth testing.
- **Calibration** uses a fixed temperature grid [0.25, 4]. It hits the boundary on barely trained models.
- **Checkpoints use `torch.load(weights_only=False)`** because they embed the config.
- **The Dockerfile has not been built** (Docker was unavailable during validation).
- **Validated only on Windows 11 with Python 3.12 and CPU.** Linux runs only in CI, and CI has not been executed (there is no remote repository).

## Recommended Next Steps

1. Vectorize HierFire and the channel across environments (or port them to JAX) to make paper-scale training feasible.
2. Run the full study (5 methods + 6 ablations × 8 seeds) and replace the manuscript's illustrative targets with measured results.
3. Implement HCN once its specification is settled.
4. Replace the heuristic F6 adversary with a learned adversary, and add a gate-aware adaptive attacker.
5. Add an ensemble-based uncertainty baseline to test the evidential head (Bengs et al., 2022).
6. Report the measured CVaR₀.₉ alongside mean SV (manuscript item HR-03).

## Research Reproducibility

- **Can be reproduced:** the software pipeline, including every stage from map generation to aggregated statistics, on the smoke configuration (see SMOKE_TEST_REPORT.md).
- **Can be reproduced (revised manuscript):** every table, plot, and number, from `runs/revision` via `scripts/make_paper_artifacts.py`.
- **Not reproduced:** the first submission's illustrative values (never measured; removed) and the planned paper-scale study.

## Production Readiness Assessment

Scores are out of 10. This is research software, not a deployable system.

| Aspect | Score | Justification |
|---|---|---|
| Architecture | 7 | Clean separation of env, channel, core equations, models, system, training, and evaluation; every method is a config |
| Code quality | 7 | Type hints, docstrings citing equations, ruff-clean. `system.py` is long and loop-heavy |
| Maintainability | 7 | Config-driven, documented assumptions, traceability map, 44 tests |
| Scalability | 3 | Python-level loops; paper scale is slow on CPU |
| Reliability | 6 | Deterministic seeding, validated configs, tested CLI error paths. Only one platform validated |
| Security | 5 | No secrets, safe path joins, input validation. Unsafe-by-design checkpoint loading is documented. Not audited |
| Reproducibility | 6 (software) / 1 (science) | Pipeline reproducible; the paper's results do not exist to reproduce |
| Documentation | 8 | README, nine docs (including the paper mapping and assumptions), status, smoke report |
