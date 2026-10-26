# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for torch` | Python version newer than PyTorch supports (e.g. 3.15) | Use Python 3.10–3.13, e.g. `uv venv --python 3.12` |
| `ConfigError: unknown keys in method: [...]` | Typo in a YAML file or `--set` | Check field names in `src/ricor/config.py` |
| `ConfigError: train/val/test sizes must sum to n_maps` | Inconsistent `scenarios` section | Make the split sizes sum to `n_maps` |
| `split 'test' has only K maps, M requested` | `eval.episodes_per_condition` exceeds the test split | Lower it, or enlarge `scenarios.n_test` |
| `recovery_window must ...` | Window too close to the episode start or end | Keep `start > recovery_smoothing` and `end + recovery_smoothing < episode_length` |
| Calibration temperature at 4.0 or 0.25 | Grid boundary reached, typical for barely trained models (as in the smoke run) | Train longer, or widen the grid in `Trainer.calibrate` |
| FRR = 100% in a smoke `eval.json` | The untrained gate rejects everything | Expected after a 2-iteration smoke run. Not a result |
| Training is very slow at paper scale | Python-level episode loop, about 0.02 s/step | Run seeds in parallel processes. Vectorizing the env and channel is listed as technical debt |
| `torch.load` warning or error about `weights_only` | Checkpoints store the config, so they are loaded with `weights_only=False` | Load only checkpoints you trust (see SECURITY.md) |
| Windows console shows `Â±` in `summary.md` | Console encoding, not file content | Open the file in a UTF-8 editor |
