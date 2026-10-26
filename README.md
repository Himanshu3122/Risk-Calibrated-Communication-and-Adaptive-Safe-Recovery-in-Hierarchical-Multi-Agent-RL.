# RiCoR: Risk-Calibrated Communication and Adaptive Safe Recovery for Hierarchical MARL

Research implementation of *"Risk-Calibrated Communication and Adaptive Safe Recovery in Hierarchical Multi-Agent Reinforcement Learning under Compound Communication Failures"*.

> **Read this first.** The first submission of the manuscript contained illustrative planning targets, not measurements. The revised manuscript reports only results measured with this code in a **reduced-budget study** (`configs/experiments/revision.yaml`: 2×10⁵ environment steps per run, 200-step episodes, 3 seeds per method, bound, and ablation, 30 test episodes per condition and seed), run with `scripts/run_study.py` and turned into tables and numbers by `scripts/make_paper_artifacts.py`. The planned paper-scale study (`configs/experiments/full.yaml`, 10⁷ steps, 8 seeds) has **not** been run.
>
> **Outcome of the reduced-budget study:** RiCoR did **not** outperform the baselines. It had the lowest asset protection rate in every condition group, HPSF was the strongest method, no difference survived Holm correction, and RiCoR's learned mode selector settled in degenerate modes in two of three seeds. See the revised manuscript, Section 7, and `../results/`.
>
> **Revision changes (2026-10-25).** Execution no longer uses privileged information: the coordinator's risk term uses a decentralized critic over the reports it actually received, verification replies come from neighbors' own border observations, and credibility features use only the receiver's own inbox (`tests/integration/test_system.py::test_execution_never_reads_the_global_state`). Further fixes: Eq. 7 tail term evaluated under the induced action distribution, discounted cost-critic targets, per-region drop-target knowledge, decision-only latency, causal recovery smoothing, FAR/FRR at the operating gate, CVaR₀.₉ reporting. New configurations: `oracle_gate` and `no_comm` (bounds), `beta_rep` (Beta-reputation baseline); new conditions U4 (slow-drift adversary), K1, K4 (compromised-fraction sweep).

## What it does

A coordinator and *N* regional agents fight a simulated wildfire while their messages pass through a channel that can lose, delay, corrupt, replay, or falsify them. RiCoR decides how much to trust each message from three signals:

1. **Channel health:** a GRU over arrival, age, and sequence-gap metadata gives `p_ch`.
2. **Content credibility with uncertainty:** an evidential Beta posterior gives `r̄` and `σ`. Together these form a lower confidence bound `ρ_lower = p_ch · max(0, r̄ − λ_u σ)`.
3. **Decision risk:** how much removing the message would change the policy (both policies use the κ = 0 reference gate), weighted by the expected CVaR cost of the action the message induces, from receiver-local critics, gives `ψ`.

Messages are gated by `w = sigmoid((ρ_lower − τ0 − κψ) / T_g)`. A learned selector switches the team between **Normal / Selective / Verify / Fallback** modes, and a **one-step shield** filters every crew action. Training is MAPPO with a Lagrangian CVaR safety constraint and a clean → isolated → compound fault curriculum.

## Key contributions implemented

| Paper contribution | Code |
|---|---|
| Factorized channel × content reliability with evidential uncertainty (Eqs. 4–6) | `src/ricor/core/reliability.py`, `models.ChannelEstimator`, `models.CredibilityNet` |
| Risk-conditioned gating via CVaR cost critic (Eqs. 7–9) | `src/ricor/core/gate.py`, `system.RiCoRSystem._sensitivity`, `models.GatedAttention` |
| Four-mode recovery with switching and verification costs (Eq. 10) | `src/ricor/core/modes.py`, `system.RiCoRSystem._apply_mode` |
| One-step action shield (Eq. 11) | `src/ricor/core/shield.py` |
| PPO-Lagrangian training, reliability loss, λ update (Eqs. 12–14) | `src/ricor/training.py` |
| Fault taxonomy F1–F6, C1–C3, U1–U3 (Table II) | `src/ricor/comm/faults.py`, `src/ricor/comm/channel.py` |
| Baselines, reference bounds, and ablations | `configs/methods/` (incl. `beta_rep`, `oracle_gate`, `no_comm`) |
| Border-view verification protocol | `HierFireEnv.border_view`, `Channel.verification_reply`, `RiCoRSystem._apply_mode` |
| Reduced-budget study and paper artifacts | `scripts/run_study.py`, `scripts/make_paper_artifacts.py`, `scripts/honest_majority.py` |
| Evaluation protocol, statistics (Sec. VI-E) | `src/ricor/evaluation/` |

Status per component: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) · Paper mapping: [docs/paper_implementation_mapping.md](docs/paper_implementation_mapping.md) · Assumptions: [docs/assumptions.md](docs/assumptions.md)

## Repository structure

```text
Github/
├── configs/
│   ├── experiments/   full.yaml (paper settings), smoke.yaml (tiny CPU check)
│   ├── env/ scenarios/ training/ evaluation/
│   └── methods/       ricor, h_mappo, h_mappo_drop, admac_h, hpsf, ablations/
├── src/ricor/         package (env, channel, core equations, models, system, training, evaluation, CLI)
├── scripts/           train.py, evaluate.py, infer.py, aggregate.py, generate_scenarios.py, smoke_test.py
├── tests/             unit/, integration/, smoke/
├── docs/              architecture, setup, training, evaluation, inference, reproducibility, troubleshooting, mapping, assumptions
├── IMPLEMENTATION_STATUS.md
└── SMOKE_TEST_REPORT.md
```

## Installation

Python 3.10–3.13 (PyTorch has no wheels for 3.14+ at the time of writing). Validated with Python 3.12 on CPU.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements-dev.txt
pip install --no-deps -e .
```

Details and GPU notes: [docs/setup.md](docs/setup.md).

## Quick start (smoke test, about 1 minute on CPU)

```bash
pytest                                   # 52 tests: unit, integration, smoke
python scripts/smoke_test.py             # train -> evaluate -> infer -> aggregate on configs/experiments/smoke.yaml
```

The smoke configuration uses a 16×16 grid, 4 regions, and 30-step episodes. Its numbers only show that the code runs; they are **not** research results.

## Training

```bash
# One method, one seed, paper settings (expensive: see docs/reproducibility.md)
python scripts/train.py --config configs/experiments/full.yaml --method configs/methods/ricor.yaml --seed 0

# Baseline or ablation
python scripts/train.py --config configs/experiments/full.yaml --method configs/methods/hpsf.yaml --seed 0
python scripts/train.py --config configs/experiments/full.yaml --method configs/methods/ablations/kappa0.yaml --seed 0

# Override any value
python scripts/train.py --config configs/experiments/smoke.yaml --method configs/methods/ricor.yaml --set method.kappa=2.0
```

Outputs go to `runs/<experiment>/<method>/seed<k>/`: `checkpoint.pt`, `checkpoint.json` (config + environment metadata), `config.json`, `train_history.json`. See [docs/training.md](docs/training.md).

## Evaluation and aggregation

```bash
python scripts/evaluate.py --checkpoint runs/full/ricor/seed0/checkpoint.pt          # -> eval.json
python scripts/aggregate.py --root runs/full --reference hpsf                         # -> summary.md / summary.json
```

Evaluation covers all 13 conditions, the packet-loss sweep, the recovery protocol, and calibration (ECE, Brier, AUROC, FAR, FRR). Aggregation reports seed-level means with 95% bootstrap CIs and Welch's t-tests with Holm correction. See [docs/evaluation.md](docs/evaluation.md).

## Reduced-budget study of the revised manuscript

```bash
python scripts/run_study.py --workers 9            # 14 configurations x 3 seeds; resumable; keeps Windows awake
python scripts/make_paper_artifacts.py --root runs/revision --out ../results
```

`run_study.py` trains and evaluates every method, bound, and ablation (`runs/revision/<method>/seed<k>/`). `make_paper_artifacts.py` writes the manuscript's tables (`tab_*.tex`), plot data (`*.dat`), and every number quoted in the text (`numbers.tex`, `artifacts_summary.json`).

## Inference

```bash
python scripts/infer.py --checkpoint runs/full/ricor/seed0/checkpoint.pt --condition C2 --map-index 0
```

Writes a per-episode trace (metrics, per-step rewards, message-reliability summary). See [docs/inference.md](docs/inference.md).

## Configuration

YAML files compose through `include:`. Unknown keys are rejected. Every research-critical value (grid, regions, faults, hyperparameters, seeds, splits, evaluation sizes) lives in `configs/`, and `--set section.key=value` overrides anything. The schema is in `src/ricor/config.py`.

## Data

There is no external dataset. Maps are generated procedurally from seeds: 6,000 maps split 4,200/600/1,200 for the paper configuration. `scripts/generate_scenarios.py` writes the split manifest. The generator's distributions are not published in the paper and are approximated here ([assumption A1](docs/assumptions.md)).

## Reproducibility

- **Software pipeline:** reproducible. Configs, seeds, deterministic map splits, run metadata (versions, git SHA), and pinned requirements are provided. A tested smoke workload runs end to end.
- **Scientific results:** the revised manuscript's results come from the reduced-budget study above and can be regenerated with the two commands shown (CPU; many hours on a 12-core laptop with 9 workers, per-job times in `runs/revision_study.log`). The planned paper-scale study has not been run: it needs about 15 ms per environment step for RiCoR on CPU (simulation plus PPO update), i.e., more than 40 CPU-hours per method-seed. HCN is not implemented and is no longer part of the manuscript.

## Limitations

- The HierFire dynamics, adversary, verification protocol, and baseline reimplementations are approximations (see [docs/assumptions.md](docs/assumptions.md)).
- HCN (the secondary benchmark) is not implemented.
- The episode loop is Python-level and single-process, so training at paper scale is slow.
- Proposition 1's safety guarantee depends on assumptions that HierFire does not guarantee, so violations occur.
- The reliability heads need injected-fault labels, which a real deployment would not have.

## Citation

```bibtex
@misc{ricor2026,
  title  = {Risk-Calibrated Communication and Adaptive Safe Recovery in Hierarchical Multi-Agent
            Reinforcement Learning under Compound Communication Failures},
  author = {Himanshu},
  year   = {2026},
  note   = {Manuscript in revision; results from a reduced-budget study}
}
```

## License

MIT. See [LICENSE](LICENSE).
