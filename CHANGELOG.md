# Changelog

## 0.1.0 (2026-10-25)

First implementation of the RiCoR research pipeline from `main.tex`:

- HierFire simulator, fault-prone channel with the F1-F6 / C1-C3 / U1-U3 taxonomy
- Factorized evidential reliability, risk-conditioned gate, four-mode recovery, one-step shield
- PPO-Lagrangian training with CVaR constraint, curriculum, temperature calibration
- Baselines (H-MAPPO, H-MAPPO+Drop, ADMAC-H, HPSF) and Table VII ablations as config files
- Evaluation protocol, statistics, aggregation, smoke-test configuration, CI