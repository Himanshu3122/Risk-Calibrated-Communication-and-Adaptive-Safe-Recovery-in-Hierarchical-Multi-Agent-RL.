# Training

## Command

```bash
python scripts/train.py --config <experiment.yaml> --method <method.yaml> [--seed K] [--set a.b=value ...]
```

or the installed entry point `ricor-train` with the same arguments.

| Experiment | Purpose |
|---|---|
| `configs/experiments/full.yaml` | Paper settings: 48×48 grid, 8 regions, 400-step episodes, 32 envs, 10⁷ steps, Table III hyperparameters |
| `configs/experiments/smoke.yaml` | 16×16 grid, 4 regions, 30-step episodes, 2 envs, 2 iterations. Pipeline check only |

| Method file | Meaning |
|---|---|
| `methods/ricor.yaml` | Full RiCoR |
| `methods/h_mappo.yaml` | Hierarchical MAPPO, trained without faults |
| `methods/h_mappo_drop.yaml` | + message dropout and noise |
| `methods/admac_h.yaml` | Single reliability score per message (ADMAC-style approximation) |
| `methods/hpsf.yaml` | Missing-information predictor + shield (HPSF approximation) |
| `methods/ablations/*.yaml` | Table VII: `no_split`, `no_uncertainty`, `kappa0`, `fixed_modes` (Ours-fixed), `no_verify`, `no_shield` |

## What happens

1. `iterations = ceil(total_steps / (n_envs × episode_length))`. Each iteration plays `n_envs` complete episodes on training maps.
2. Curriculum by progress (`train.curriculum`, default `[0.2, 0.4, 0.4]`): clean → isolated F1–F6 (randomized rates) → compound C1–C3. H-MAPPO always trains clean.
3. PPO-Lagrangian update (Eqs. 12–14). λ rises while the empirical CVaR₀.₉ of episode cost exceeds `cost_budget` (d = 0.5), and is clipped at `lambda_max`.
4. Temperature calibration of `p_ch` and `r̄` on validation maps.
5. Checkpoint written.

## Outputs (`runs/<name>/<method>/seed<k>/`)

| File | Content |
|---|---|
| `checkpoint.pt` | State dict (all networks, temperatures, λ) + config + metadata |
| `checkpoint.json` | Config and environment metadata (Python, torch, numpy, platform, git SHA, timestamp) |
| `config.json` | Resolved configuration |
| `train_history.json` | Per iteration: losses, return, APR, SV, λ, CVaR of cost, stage, seconds |

Set `train.checkpoint_every: N` to keep intermediate checkpoints every N iterations.

## Seeds

The paper uses 8 seeds. Run `--seed 0` through `--seed 7` for each method. Seeds control PyTorch and NumPy initialization, map sampling, and channel randomness. The map split itself depends only on `scenarios.base_seed`.

## Cost

Measured on the build machine (CPU): a paper-size rollout costs about 0.02 s per environment step. 10⁷ steps is therefore about 55 h of rollouts per method-seed, plus PPO updates. Plan parallel runs across machines; see [reproducibility.md](reproducibility.md).
