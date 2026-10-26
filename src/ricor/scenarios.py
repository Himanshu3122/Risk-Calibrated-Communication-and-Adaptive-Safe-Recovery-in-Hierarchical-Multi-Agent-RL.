"""Procedural map library split by seed (paper Sec. VI-B).

Maps are generated on demand from their seed by :meth:`HierFireEnv.reset`, so the library
is fully described by its seed list. The paper's generator distributions (fuel, terrain,
wind, ignition) are not published; ours are assumption A1.
"""

from __future__ import annotations

import numpy as np

from ricor.config import ScenarioConfig


def split_seeds(cfg: ScenarioConfig) -> dict[str, list[int]]:
    """Deterministic train/val/test map-seed split."""
    cfg.validate()
    seeds = cfg.base_seed + np.arange(cfg.n_maps)
    perm = np.random.default_rng(cfg.base_seed).permutation(seeds)
    a, b = cfg.n_train, cfg.n_train + cfg.n_val
    return {
        "train": perm[:a].tolist(),
        "val": perm[a:b].tolist(),
        "test": perm[b:].tolist(),
    }
