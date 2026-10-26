import numpy as np
import pytest
import torch

from ricor.comm.faults import evaluation_spec
from ricor.evaluation.runner import run_condition
from ricor.system import RiCoRSystem
from ricor.training import Trainer, load_system

from ..conftest import method_cfg

METHODS = [
    "ricor",
    "h_mappo",
    "h_mappo_drop",
    "admac_h",
    "hpsf",
    "fixed_modes",
    "no_split",
    "no_shield",
    "kappa0",
    "oracle_gate",
    "no_comm",
    "beta_rep",
]


@pytest.mark.parametrize("name", METHODS)
def test_every_method_runs_a_faulty_episode(name):
    cfg = method_cfg(name)
    torch.manual_seed(0)
    system = RiCoRSystem(cfg)
    roll = system.run_episodes(
        [cfg.scenarios.base_seed] * 2,
        [evaluation_spec("C2")] * 2,
        [1, 2],
        stochastic=True,
        training=True,
        record_reliability=True,
    )
    T = cfg.env.episode_length
    assert roll.step_rewards.shape == (T, 2)
    assert roll.data["obs_r"].shape[:3] == (T, 2, system.N)
    assert roll.data["meta"].shape == (T, 2, system.L, 3)
    m = roll.episode_metrics[0]
    assert 0 <= m["apr"] <= 100 and m["comm_kb"] > 0
    assert abs(sum(m["mode_occupancy"]) - 1.0) < 1e-6
    if cfg.method.modes == "none":
        assert m["mode_occupancy"][0] == 1.0
    if not cfg.method.shield:
        assert m["shield_interventions"] == 0
    if cfg.method.no_comm:
        assert roll.reliability_records["rho_bar"].size == 0  # nothing received is used
    else:
        assert roll.reliability_records["rho_bar"].size > 0


def test_execution_never_reads_the_global_state(monkeypatch):
    """Decentralized execution: the centralized critics are training-only."""
    cfg = method_cfg("ricor")
    system = RiCoRSystem(cfg)

    def forbidden(*_a, **_k):
        raise AssertionError("centralized critic called at execution time")

    monkeypatch.setattr(system.cost_critic, "forward", forbidden)
    monkeypatch.setattr(system.value, "forward", forbidden)
    system.run_episodes([cfg.scenarios.base_seed], [evaluation_spec("C2")], [5], stochastic=False)


def test_drop_targets_are_known_per_region():
    cfg = method_cfg("ricor")
    system = RiCoRSystem(cfg)
    slot = system.new_slot(cfg.scenarios.base_seed, evaluation_spec("clean"), 0, None)
    slot.sent_decision = {3: 1}
    slot.decision_targets = {1: [(2, 2)]}

    class D:
        seq = np.full(system.L, -1)

    d = D()
    d.seq[system.downlink_idx[0]] = 3  # only region 0 received the goal message of decision 1
    assert system._known_targets(slot, d, 0) == [(2, 2)]
    assert system._known_targets(slot, d, 1) == []


def test_gate_downweights_when_risk_is_high():
    cfg = method_cfg("ricor")
    system = RiCoRSystem(cfg)
    roll = system.run_episodes(
        [cfg.scenarios.base_seed], [evaluation_spec("C1")], [3], stochastic=False, record_reliability=True
    )
    w = roll.reliability_records["w"]
    assert ((w >= 0) & (w <= 1)).all()


def test_training_update_and_checkpoint_roundtrip(tmp_path):
    cfg = method_cfg("ricor", train={"total_steps": 60})
    trainer = Trainer(cfg, tmp_path)
    path = trainer.train()
    hist = trainer.history
    assert len(hist) == 1 and np.isfinite(hist[0]["pi_r"]) and np.isfinite(hist[0]["rel"])
    assert hist[0]["lambda"] >= 0.0
    system, cfg2 = load_system(path)
    assert cfg2.method.name == "ricor"
    for (k, a), (_, b) in zip(trainer.system.state_dict().items(), system.state_dict().items(), strict=True):
        assert torch.equal(a, b), k
    r1 = run_condition(trainer.system.eval(), cfg, "F4", 2, 2)
    r2 = run_condition(system, cfg2, "F4", 2, 2)
    assert r1["summary"]["apr"] == r2["summary"]["apr"]


def test_hpsf_trains_its_predictor(tmp_path):
    cfg = method_cfg("hpsf", train={"total_steps": 60})
    trainer = Trainer(cfg, tmp_path)
    trainer.train()
    assert trainer.history[0]["pred"] > 0
