# Smoke Test Report

Everything below was executed on 2026-10-25 during repository construction. Nothing here is a research result: smoke runs use a 16×16 grid, 4 regions, 30-step episodes, and 2 training iterations.

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| System Python | 3.15.1 (no PyTorch wheels available) |
| Validation Python | 3.12.15 (venv created with uv 0.13.5) |
| Packages | torch 2.5.1+cpu, numpy 2.2.6, scipy 1.18.1, PyYAML 6.0.3, pytest 9.1.1, ruff 0.17.2 |
| Hardware | CPU only. No GPU used |
| Package install | `uv pip install -e Github --no-deps` → `ricor==0.1.0` installed |

## Results

| Test | Status | Command | Observed result | Notes |
|---|---|---|---|---|
| PyTorch on system Python | **BLOCKED** | `python -m pip index versions torch` | `No matching distribution found for torch` | Python 3.15 is unsupported. Worked around with a Python 3.12 venv |
| Dependency install | PASS | `uv pip install --index-url https://download.pytorch.org/whl/cpu ... torch==2.5.1 numpy scipy pyyaml pytest ruff` | Installed | `requirements*.txt` pinned to these versions |
| Import validation | PASS | `python -c "import ricor, ricor.cli, ricor.system, ricor.training, ricor.evaluation.runner, ricor.evaluation.aggregate"` | `imports ok 0.1.0` | |
| Lint (first run) | FAIL → fixed | `ruff check src scripts tests` | 48 findings (missing trailing newlines, long lines, ambiguous `l`, unused locals, `zip` without `strict`) | Fixed with `ruff --fix`, `ruff format`, and manual edits |
| Lint (final) | PASS | `ruff check src scripts tests`; `ruff format --check ...` | `All checks passed!`; `42 files already formatted` | |
| Config validation | PASS | `pytest tests/unit/test_config.py` | 4 passed | Paper values from Table III and Sec. VI checked against `configs/experiments/full.yaml`; unknown keys and inconsistent splits rejected |
| Unit tests | PASS | `pytest tests/unit` | 31 passed in 22.2 s | Equations 4–8, 10–12, channel/faults, environment, shield, modes, metrics, statistics |
| Integration tests (first run) | FAIL → fixed | `pytest tests/integration -x` | `RuntimeError: view size is not compatible with input tensor's size and stride` in the HPSF path | NumPy fancy indexing gave non-contiguous arrays. Fixed with `np.ascontiguousarray` in `RiCoRSystem._t` |
| Integration tests (second run) | FAIL → fixed | same | Same error in `Trainer._predictor_loss` | Replaced `.view` with `.reshape` in `training.py`; rollout buffers stored contiguous |
| Integration tests (final) | PASS | `pytest tests/integration` | 11 passed in 18.8 s | 8 method configs run faulty episodes; training update; checkpoint round trip gives identical weights and identical evaluation; HPSF predictor trains |
| Smoke tests (pytest) | PASS | `pytest tests/smoke` | 2 passed in 30.1 s | CLI train → evaluate (13 conditions + sweep + recovery) → infer → aggregate; bad `--set` returns exit code 2 |
| Full suite (final) | PASS | `pytest` | 44 passed in 38.3 s | |
| End-to-end smoke script | PASS | `python scripts/smoke_test.py --methods ricor h_mappo h_mappo_drop admac_h hpsf` | `[smoke] OK`, 80.1 s wall | Run twice; both runs printed identical per-method numbers (deterministic seeding) |
| Scenario manifest | PASS | `python scripts/generate_scenarios.py --config configs/experiments/smoke.yaml` | `smoke_manifest.json` written | |
| Training script (2nd seed) | PASS | `python scripts/train.py ... --method configs/methods/{ricor,hpsf}.yaml --seed 1` | Checkpoints written | Weights differ from seed 0 (checked) |
| Ablation training | PASS | `python scripts/train.py ... --method configs/methods/ablations/<a>.yaml` for all six ablations | All exit 0, 6.4–7.3 s each | |
| Evaluation script | PASS | `python scripts/evaluate.py --checkpoint ... --no-sweep --no-recovery` | `eval.json` written | |
| Inference script | PASS | `python scripts/infer.py --checkpoint ... --condition U1` | `infer_U1.json` written | |
| Inference error path | PASS | `python scripts/infer.py ... --condition Z9` | Exit 2, `unknown fault condition 'Z9'` | |
| Aggregation with statistics | PASS | `python scripts/aggregate.py --root ... --reference hpsf` | `summary.md`/`summary.json` with bootstrap CIs, Welch t, Hedges' g, Holm p | Two seeds for ricor and hpsf |
| Console entry points | PASS | `ricor-train --help` (and evaluate, infer, aggregate, scenarios) | Usage printed, exit 0 | |
| Paper-scale timing probe | PASS | 2 RiCoR episodes × 50 steps on `configs/experiments/full.yaml` | 0.019–0.021 s per env step; 0.368 M parameters | Basis of the compute estimate in docs/reproducibility.md |
| Docker build | **NOT RUN** | `docker build -t ricor .` | Docker not installed on the build machine | Dockerfile unverified |
| GitHub Actions CI | **NOT RUN** | `.github/workflows/ci.yml` | No remote repository | Workflow mirrors the commands above |
| Full paper-scale training | **NOT RUN** | `scripts/train.py --config configs/experiments/full.yaml ...` | Not attempted | ~55 h of rollouts per method-seed on this CPU (estimate from the probe) |

## Fixes applied during validation

1. **Fidelity fix before testing.** Entrapment first counted adjacency to *burning* cells as unsafe. The paper's predicate refers to cells whose *spread probability* exceeds 0.6 (burning cells have spread probability 0). With the original rule, crews could never suppress fire safely. The environment and the shield were corrected, and `tests/unit/test_env.py::test_entrapment_follows_paper_predicate` covers it.
2. Non-contiguous tensor views (two places), described above.
3. Deterministic evaluation first sent all drops to the arg-max region. It now uses largest-remainder rounding (assumption A13).
4. NumPy 2 removed `ndarray.ptp`; replaced by `np.ptp`.

## Runtime observations

- The untrained smoke models are not calibrated. In the smoke `eval.json`, FRR = 100% (the gate rejects every message) and AUROC is about 0.49. Calibration temperatures hit the grid boundary (4.0). This is expected after two iterations.
- `comm_kb` counts bytes sent, including lost messages, so it is identical across loss conditions for methods without verification.
- Seeds 0 and 1 of RiCoR produced different clean APR and SV but identical compound APR on the two smoke test maps. Asset losses are discrete events on the same maps, so this is not a seeding bug (weights were confirmed to differ).

## Remaining issues and dependency concerns

- PyTorch does not support the system's Python 3.15; use 3.10–3.13.
- `torch.load(weights_only=False)` is required for checkpoints that embed configs.
- Only Windows has been validated locally.
