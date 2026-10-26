import numpy as np

from ricor.core.modes import FixedRule, Mode, admissible, mode_reward, mode_summary
from ricor.core.shield import fallback_action, shield, unsafe_actions


def patch(spread=None, intensity=None):
    p = np.zeros((3, 5, 5), np.float32)
    for (y, x), v in (spread or {}).items():
        p[1, y, x] = v
    for (y, x), v in (intensity or {}).items():
        p[0, y, x] = v
    return p


def test_unsafe_moves_and_drop_proximity():
    p = patch(spread={(0, 2): 0.9})  # two cells north of the crew
    unsafe = unsafe_actions(p, np.zeros((5, 5), bool), 0.6, 1)
    assert unsafe[1] and not unsafe[2]  # moving north is adjacent to the hot cell
    drops = np.zeros((5, 5), bool)
    drops[4, 2] = True
    unsafe = unsafe_actions(patch(), drops, 0.6, 1)
    assert unsafe[2] and not unsafe[1]


def test_shield_keeps_safe_proposals_and_replaces_unsafe_ones():
    p = patch(spread={(0, 2): 0.9})
    probs = np.full(8, 1 / 8)
    assert shield(2, probs, p, np.zeros((5, 5), bool), 0.6, 1) == (2, False)
    a, hit = shield(1, probs, p, np.zeros((5, 5), bool), 0.6, 1)
    assert hit and a != 1
    assert fallback_action(patch(intensity={(0, 2): 1.0})) == 2  # move away (south)


def test_hysteresis_and_fallback_admissibility():
    m = admissible(Mode.NORMAL, since_switch=1, hysteresis=5)
    assert m[Mode.NORMAL] and m[Mode.FALLBACK] and not m[Mode.SELECTIVE]
    assert admissible(Mode.NORMAL, 5, 5).all()
    assert not admissible(Mode.NORMAL, 9, 5, verify_enabled=False)[Mode.VERIFY]


def test_fixed_rule_and_mode_reward():
    rule = FixedRule(tau=0.5, psi_v=0.5, sigma_v=0.1, outage_limit=3, hysteresis=0)
    xi = mode_summary(np.array([0.2, 0.9]), np.array([0.1, 0.0]), np.array([0.0, 0.0]), 0, 10, 5)
    assert rule.choose(Mode.NORMAL, xi, 0, 10, True) == Mode.SELECTIVE
    assert rule.choose(Mode.NORMAL, xi, outage=5, since_switch=10, verify=True) == Mode.FALLBACK
    assert mode_reward(1.0, True, 2, 0.5, 2.0, 0.05, 0.02) == 1.0 - 0.05 - 0.04 - 1.0
