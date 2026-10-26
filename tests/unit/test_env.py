import numpy as np

from ricor.envs.hierfire import N_CREW_ACTIONS, REPORT_DIM, HierFireEnv


def make(cfg):
    env = HierFireEnv(cfg.env)
    env.reset(123)
    return env


def test_reset_is_deterministic(smoke_cfg):
    a, b = make(smoke_cfg), make(smoke_cfg)
    assert np.array_equal(a.fuel, b.fuel) and np.array_equal(a.crews, b.crews)
    for _ in range(5):
        acts = np.zeros((a.n_regions, a.n_crews), int)
        assert a.step(acts, None)[0] == b.step(acts, None)[0]
    assert np.array_equal(a.intensity, b.intensity)


def test_shapes_and_metric_ranges(smoke_cfg):
    env = make(smoke_cfg)
    assert env.reports().shape == (4, REPORT_DIM)
    assert env.crew_patches().shape == (4, 3, 3, 5, 5)
    assert env.global_state().shape == (4 * REPORT_DIM + 1,)
    rng = np.random.default_rng(0)
    while not env.done:
        env.step(rng.integers(0, N_CREW_ACTIONS, (4, 3)), None)
    assert 0.0 <= env.asset_protection_rate() <= 100.0 and 0.0 <= env.burned_area() <= 100.0


def test_regions_tile_the_grid_and_neighbors_are_symmetric(smoke_cfg):
    env = make(smoke_cfg)
    assert env.region_neighbors(0) == [2, 1]
    for r in range(env.n_regions):
        assert all(r in env.region_neighbors(q) for q in env.region_neighbors(r))


def test_drop_lands_after_latency_and_costs_nearby_crew(smoke_cfg):
    env = make(smoke_cfg)
    env.intensity[:] = 0.0
    env.intensity[2, 2] = 1.0
    env.crews[0, 0] = (2, 3)  # adjacent to the target
    alloc = np.zeros(env.n_regions)
    alloc[0] = 1
    stay = np.zeros((env.n_regions, env.n_crews), int)
    _, info = env.step(stay, alloc)
    assert info.drop_events == 0 and env.pending_drops
    for _ in range(smoke_cfg.env.drop_latency - 1):
        _, info = env.step(stay, None)
    _, info = env.step(stay, None)
    assert info.drop_events >= 1 and env.intensity[2, 2] == 0.0


def test_entrapment_follows_paper_predicate(smoke_cfg):
    env = make(smoke_cfg)
    env.intensity[:] = 0.0
    env.spread[:] = 0.0
    env.spread[5, 6] = 0.9  # neighbor above threshold -> unsafe
    unsafe = env._unsafe_cells()
    assert unsafe[5, 5] and unsafe[5, 6] and not unsafe[0, 0]
    env.spread[:] = 0.0
    env.intensity[8, 8] = 1.0  # adjacency to a burning cell alone is safe
    unsafe = env._unsafe_cells()
    assert unsafe[8, 8] and not unsafe[8, 9]
