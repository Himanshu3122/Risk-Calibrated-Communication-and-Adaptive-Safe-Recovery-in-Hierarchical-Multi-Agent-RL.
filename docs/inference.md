# Inference

```bash
python scripts/infer.py --checkpoint runs/<name>/<method>/seed<k>/checkpoint.pt --condition U1 --map-index 3
```

This runs one deterministic episode on test map `map-index` under the named fault condition and writes `infer_<condition>.json` next to the checkpoint:

| Key | Content |
|---|---|
| `episode` | APR, BA, SV, return, communication volume, queries, shield interventions, mode occupancy, latency |
| `step_rewards` | Team reward at every step |
| `messages` | Mean and count of `rho_bar`, `p_ch`, `r_bar`, gate weight `w`, and the labels over delivered messages |

An invalid condition or map index exits with code 2 and an error message.

## Programmatic use

```python
from ricor.training import load_system
from ricor.comm.faults import evaluation_spec
from ricor.scenarios import split_seeds

system, cfg = load_system("runs/smoke/ricor/seed0/checkpoint.pt")
maps = split_seeds(cfg.scenarios)["test"][:2]
roll = system.run_episodes(maps, [evaluation_spec("C2")] * 2, channel_seeds=[0, 1], stochastic=False,
                           record_reliability=True)
print(roll.episode_metrics[0]["apr"], roll.reliability_records["w"].mean())
```

There is no web API. RiCoR is an offline research pipeline, and the paper describes no service.
