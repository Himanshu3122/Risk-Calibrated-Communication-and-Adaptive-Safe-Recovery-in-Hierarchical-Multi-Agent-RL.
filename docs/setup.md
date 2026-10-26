# Setup

## Requirements

- Python 3.10–3.13. PyTorch publishes no wheels for newer interpreters. This repository was validated with Python 3.12.15 on Windows 11 (CPU).
- About 1 GB of disk space for PyTorch (CPU build).
- No GPU is required for the smoke workload. `train.device: cuda` enables a GPU for training, but the episode loop is CPU-bound.

## Clean environment (pip)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements-dev.txt
pip install --no-deps -e .
```

For a CUDA build, replace the first install with the command from https://pytorch.org for your CUDA version.

## Clean environment (uv)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.5.1 -r requirements-dev.txt
uv pip install --python .venv --no-deps -e .
```

## Verify

```bash
python -c "import ricor; print(ricor.__version__)"
pytest -q
python scripts/smoke_test.py
```

## Docker (CPU)

```bash
docker build -t ricor .
docker run --rm ricor                      # runs scripts/smoke_test.py
docker run --rm ricor pytest -q
```

The Dockerfile was written but **not built** during validation, because Docker was unavailable on the build machine.

## Environment variables

See `.env.example`. `RICOR_LOG_LEVEL` controls logging verbosity, and `OMP_NUM_THREADS` limits PyTorch CPU threads. No secrets are needed.
