# Paper-to-Repository Mapping

Source: `main.tex` (RiCoR manuscript). Status: **Full** = implemented as specified; **Approx.** = implemented with an assumption from [assumptions.md](assumptions.md); **Missing** = not implemented.

**Important:** the manuscript's quantitative results are illustrative targets, not measurements (manuscript note HR-01). Nothing in this repository reproduces them, and no result here should be read as confirming them.

## Problem formulation (Sec. III)

| Paper | Repository | Status | Notes |
|---|---|---|---|
| Eq. (1) hierarchical Dec-POMDP with cost and goals | `envs/hierfire.py`, `system.py` | Full | Reward and cost are returned separately; goals are issued every K steps |
| Eq. (2) channel: loss ℓ(z), Gilbert–Elliott state, delay δ, content transform φ_f, metadata η = (a, δ, s) | `comm/channel.py::Channel.transmit` | Full | Labels y_ch and y_cr are returned for supervision only (A27) |
| Threat model: channel vs content failures, ≤ k compromised regionals, uncompromised coordinator, ℓ∞ budget ε | `comm/channel.py`, `comm/faults.py` | Approx. | Adversary is a heuristic, not an optimizer (A9) |
| Eq. (3) CVaR-constrained objective | `training.py::Trainer.update` | Full | Lagrangian relaxation with an empirical CVaR of episode costs |
| Safety predicates: entrapment (spread prob. > 0.6, occupying or adjacent) and drops within one cell | `HierFireEnv._unsafe_cells`, `HierFireEnv.step` | Approx. | Onset counting; standing in a burning cell counts as unsafe (A4) |

## Architecture (Sec. IV, Fig. 1)

| Paper | Repository | Status | Notes |
|---|---|---|---|
| Communication monitor (metadata) | `Channel.transmit` → `Delivery` | Full | |
| Channel estimator: GRU over η → p̂_ch | `models.ChannelEstimator` | Full | |
| Credibility network → evidence e⁺, e⁻ | `models.CredibilityNet`, `RiCoRSystem.cred_features` | Approx. | Inputs per A24 |
| Risk-conditioned gate | `core/gate.py`, `RiCoRSystem._sensitivity` | Approx. | Tail-cost decomposition (A11); reference weights (A19) |
| Gated multi-head attention aggregation | `models.GatedAttention` (regional), `models.CoordinatorPolicy` | Approx. | Coordinator uses per-region gated slots (A6) |
| Four modes N/S/V/F with hysteresis (Fig. 2) | `core/modes.py`, `RiCoRSystem._apply_mode` | Approx. | Mode effects per A20; verification per A12 |
| Coordinator issuing goals every K steps | `RiCoRSystem.run_episodes` | Full | Goals = drop allocation vector |
| Regional policies | `models.RegionalPolicy` | Approx. | Shared parameters, one head per crew (A6) |
| One-step safety shield | `core/shield.py` | Full | Local-patch predicate; fallback = move away from heat |

## Mathematical model (Sec. V)

| Equation | Repository | Status |
|---|---|---|
| (4) α = 1 + e⁺, β = 1 + e⁻, r̄ | `core/reliability.py::beta_from_evidence`, `beta_mean_var` | Full |
| (5) Beta variance σ² | `beta_mean_var` | Full |
| (6) ρ̄ = p̂_ch·r̄, ρ_lower = p̂_ch·max(0, r̄ − λ_u σ) | `reliability` | Full (+ ablation switches) |
| (7) ψ = KL(π(·\|M) ‖ π(·\|M₋ⱼ))·max_a Q^c_α | `core/gate.py::decision_sensitivity`, `RiCoRSystem._sensitivity` | Approx. (A11, A19) |
| (8) τ = τ₀ + κψ, w = σ((ρ_lower − τ)/T_g) | `gate_weights` | Full |
| (9) gated attention | `GatedAttention` (log w added to the logits; null slot) | Full |
| (10) r^μ = R − c_sw·1[switch] − c_v·n_q − λC | `core/modes.py::mode_reward` | Full |
| (11) shield | `core/shield.py::shield` | Full |
| Proposition 1 | Documented in `core/shield.py`; its assumptions are not guaranteed in HierFire | n/a |
| (12) reliability loss with KL shrinkage | `reliability_loss`, `Trainer._reliability_loss` | Full |
| (13) total PPO-Lagrangian loss | `Trainer._ppo_losses` | Full (weights per A7) |
| (14) λ ← [λ + η_λ(CVaR − d)]₊ | `Trainer.update` | Full (plus `lambda_max` clip) |
| Algorithm 1 (execution) | `RiCoRSystem.run_episodes` | Full |
| Training procedure (curriculum, temperature scaling) | `Trainer.train`, `Trainer.calibrate` | Full (stage fractions configurable) |

## Experimental setup (Sec. VI)

| Paper | Repository | Status | Notes |
|---|---|---|---|
| HierFire 48×48, N = 8 regions of 12×24, 3 crews, 6 drops, K = 5, 400 steps, 12–20 assets | `configs/env/hierfire.yaml` | Full | Fire dynamics approximated (A1) |
| HCN (MPE, 1 leader + 6 followers) | — | **Missing** | A28 |
| 6,000 maps split 4,200 / 600 / 1,200 by seed | `scenarios.py`, `configs/scenarios/paper.yaml` | Full | Maps generated from seeds; generator per A1 |
| Fault table F1–F6, C1–C3, U1–U3 | `comm/faults.py::evaluation_spec` | Approx. | Open rates per A8, A10 |
| Curriculum clean → isolated → compound | `sample_training_spec`, `Trainer.stage` | Full | Stage fractions [0.2, 0.4, 0.4] assumed |
| Baselines H-MAPPO, H-MAPPO+Drop, ADMAC-H, HPSF | `configs/methods/*.yaml` | Approx. | A16–A18 |
| Hyperparameters (Table III) | `configs/training/full.yaml`, `configs/methods/ricor.yaml` | Full | Except unreported loss weights (A7) |
| 10⁷ steps, 8 seeds | `configs/training/full.yaml`; seeds via `--seed` | Full | Not executed here (compute) |
| Hardware (2× RTX A6000, EPYC 7543) | — | n/a | The code runs on CPU; `train.device` selects the device |

## Evaluation (Sec. VI-E, VII)

| Paper | Repository | Status |
|---|---|---|
| APR, BA, SV | `HierFireEnv.asset_protection_rate`, `burned_area`; cost sum | Full |
| ECE (15 bins), Brier, AUROC, FAR, FRR | `evaluation/metrics.py` | Full |
| Diagnosis confusion (Fig. 6) | `diagnosis_confusion` | Approx. (3-way, A14) |
| Degradation, recovery time (95% of pre-fault reward), mode usage | `metrics.degradation`, `recovery_time`, episode `mode_occupancy` | Approx. (A15) |
| Packet-loss sweep 0–0.6 (Fig. 4) | `runner.evaluate` → `loss_sweep` | Full |
| Recovery curve with fault window [100, 250] (Fig. 7) | `runner.run_recovery` | Full |
| Communication volume, inference latency, training time | `comm_kb`, `latency_ms` per episode; `seconds` in `train_history.json` | Full (latency is wall-clock of the Python loop, not an optimized deployment) |
| 95% bootstrap CIs; Welch's t-test across seeds | `evaluation/stats.py`, `evaluation/aggregate.py` | Full (+ Holm, A26) |
| κ sweep (Fig. 8) | `--set method.kappa=...` per run, then `ricor-aggregate` | Supported via configs, no dedicated script |
| Scalability N ∈ {4, 8, 16, 24} (Table VIII) | `--set env.region_rows=...` | Supported via configs, no dedicated script |
| Table VII ablations | `configs/methods/ablations/*.yaml` | Full (as switches) |
| Supplementary run with 4/8 compromised agents | — | Not scripted; FaultSpec(compromised=4) can express it |

## Not reproduced

All tables and figures with numbers (Tables IV–IX, Figs. 4–8) and every statistical claim in Sections VII–VIII. See [reproducibility.md](reproducibility.md).
