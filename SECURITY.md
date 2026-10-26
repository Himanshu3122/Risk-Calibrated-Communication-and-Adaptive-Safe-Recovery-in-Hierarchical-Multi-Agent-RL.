# Security Policy

This repository is research software. It has not been security-audited and is not intended for
deployment on real communication networks or vehicles.

## Reporting a vulnerability

Please do not open a public issue for security problems. Contact the maintainer privately
through the email listed in the paper's author block, and include steps to reproduce.

## Scope notes

- Checkpoints are loaded with `torch.load(..., weights_only=False)` because they contain the
  experiment configuration. Only load checkpoints you created or trust.
- The code needs no secrets or network access at run time.