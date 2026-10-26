# Contributing

Thank you for helping. This is a research codebase, so the bar for changes is scientific
traceability as much as code quality.

## Workflow

1. Create a virtual environment with Python 3.10-3.13 and run `make dev`.
2. Make your change on a branch. Keep research-critical values in `configs/`, not in code.
3. Run `make lint` and `make test` (the smoke tests take about a minute on a laptop CPU).
4. Open a pull request that explains what changed and why.

## Rules for research changes

- If you implement something the paper does not specify, add an entry to `docs/assumptions.md`
  and update `docs/paper_implementation_mapping.md`.
- Never commit numbers as results unless they come from runs you can reproduce from a config,
  seed list, and git SHA. The manuscript's own numbers are illustrative targets, not results.
- Add or update tests for every behavioral change, especially in `src/ricor/core/` (paper equations).

## Style

Ruff (`pyproject.toml`) is the linter. Use type hints and docstrings that cite the paper's
equation or section numbers where relevant.