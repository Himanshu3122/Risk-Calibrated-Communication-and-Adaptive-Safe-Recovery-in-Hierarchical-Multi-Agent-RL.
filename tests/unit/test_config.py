import pytest

from ricor.config import ConfigError, from_dict, load_config, parse_override

from ..conftest import ROOT, SMOKE


def test_smoke_config_loads_with_includes(smoke_cfg):
    assert smoke_cfg.env.height == 16 and smoke_cfg.env.n_regions == 4
    assert smoke_cfg.scenarios.n_train + smoke_cfg.scenarios.n_val + smoke_cfg.scenarios.n_test == 40
    assert smoke_cfg.train.n_envs == 2


def test_full_config_matches_paper_table_iii():
    cfg = load_config(ROOT / "configs" / "experiments" / "full.yaml")
    assert (cfg.env.height, cfg.env.width, cfg.env.n_regions) == (48, 48, 8)
    assert (cfg.env.height // cfg.env.region_rows, cfg.env.width // cfg.env.region_cols) == (12, 24)
    assert cfg.env.episode_length == 400 and cfg.env.coordinator_period == 5 and cfg.env.n_drops == 6
    t = cfg.train
    assert (t.actor_lr, t.critic_lr, t.lagrange_lr, t.gamma, t.gae_lambda, t.clip) == (
        5e-4,
        1e-3,
        5e-3,
        0.99,
        0.95,
        0.2,
    )
    assert (t.ppo_epochs, t.minibatches, t.n_envs, t.cvar_alpha, t.cost_budget) == (10, 4, 32, 0.9, 0.5)
    assert (cfg.scenarios.n_train, cfg.scenarios.n_val, cfg.scenarios.n_test) == (4200, 600, 1200)
    assert cfg.eval.episodes_per_condition == 1200


def test_overrides_and_unknown_keys():
    cfg = load_config(SMOKE, parse_override("method.kappa=2.5"))
    assert cfg.method.kappa == 2.5
    with pytest.raises(ConfigError):
        load_config(SMOKE, {"method": {"kapa": 1.0}})
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_validation_rejects_inconsistent_values(smoke_cfg):
    data = smoke_cfg.to_dict()
    data["scenarios"]["n_test"] = 99
    with pytest.raises(ConfigError):
        from_dict(data)
    data = smoke_cfg.to_dict()
    data["env"]["height"] = 15
    with pytest.raises(ConfigError):
        from_dict(data)
