import numpy as np

from ricor.comm.channel import FAULT_CORRUPT, FAULT_MISLEAD, FAULT_STALE, Channel
from ricor.comm.faults import FaultSpec, compose, evaluation_spec, sample_training_spec

PEERS = [[1, 2], [0, 3], [0, 3], [1, 2]]


def run(spec, steps=200, seed=0, window=None):
    ch = Channel(4, PEERS, 16, 4)
    ch.reset(spec, seed=seed, window=window)
    rng = np.random.default_rng(1)
    out = []
    for t in range(steps):
        out.append(ch.transmit(t, rng.random((4, 16)).astype(np.float32), np.zeros((4, 4), np.float32)))
    return ch, out


def test_clean_channel_delivers_every_message_fresh():
    ch, out = run(FaultSpec(), steps=20)
    assert all(d.arrived.all() and (d.age == 0).all() and d.y_ch.all() and d.y_cr.all() for d in out)


def test_bernoulli_loss_rate_matches_spec():
    _, out = run(FaultSpec(loss_p=0.3), steps=400)
    rate = 1 - np.mean([d.arrived.mean() for d in out[1:]])
    assert abs(rate - 0.3) < 0.03


def test_gilbert_elliott_produces_bursts():
    spec = evaluation_spec("F2")
    _, out = run(spec, steps=600)
    lost = np.array([~d.arrived for d in out])
    runs = []
    for col in lost.T:
        n = 0
        for v in col:
            if v:
                n += 1
            elif n:
                runs.append(n)
                n = 0
    assert np.mean(runs) > 3.0  # mean burst 14 in spec; i.i.d. 25% loss would give ~1.3


def test_delay_produces_aged_messages_and_labels():
    _, out = run(FaultSpec(delay_min=3, delay_max=3), steps=20)
    assert (out[-1].age == 3).all() and not out[-1].y_ch.any()


def test_content_faults_are_labelled():
    ch, out = run(FaultSpec(corrupt_rate=1.0, corrupt_sigma=0.5), steps=5)
    assert (out[-1].fault_type[ch.uplink_idx] == FAULT_CORRUPT).all() and not out[-1].y_cr[ch.uplink_idx].any()
    assert out[-1].y_cr[ch.downlink_idx].all()  # goals are not content-corrupted
    ch, out = run(FaultSpec(stale_rate=1.0), steps=40)
    assert (out[-1].fault_type[ch.uplink_idx] == FAULT_STALE).all()
    ch, out = run(FaultSpec(compromised=2, mislead_eps=0.4), steps=5)
    bad = [i for i in ch.uplink_idx if ch.links[i][0] - 1 in ch.compromised]
    assert len(bad) == 2 and (out[-1].fault_type[bad] == FAULT_MISLEAD).all()


def test_fault_window_switches_faults_off():
    _, out = run(FaultSpec(loss_p=1.0), steps=30, window=(10, 20))
    assert out[5].arrived.all() and not out[15].arrived.any() and out[25].arrived.all()


def test_verification_reply_uses_border_view_and_leaves_channel_state():
    ch = Channel(4, PEERS, 16, 4)
    ch.reset(evaluation_spec("F2"), seed=3)
    for t in range(30):
        ch.transmit(t, np.zeros((4, 16), np.float32), np.zeros((4, 4), np.float32))
    ge, rng_state = ch.ge_bad.copy(), ch.rng.bit_generator.state
    view = np.array([4.0, 0.7], np.float32)
    replies = [ch.verification_reply(30, 1, view) for _ in range(50)]
    assert (ch.ge_bad == ge).all() and ch.rng.bit_generator.state == rng_state
    assert all(r is None or np.allclose(r, view) for r in replies)


def test_slow_drift_adversary_ramps_up():
    ch = Channel(4, PEERS, 16, 4)
    ch.reset(evaluation_spec("U4"), seed=0)
    eps = [ch.mislead_eps(t) for t in range(30)]
    assert eps[0] < eps[10] < eps[25] and abs(eps[25] - 0.4) < 1e-9


def test_compose_and_curriculum():
    c = compose(FaultSpec(loss_p=0.5), FaultSpec(loss_p=0.5, compromised=2))
    assert abs(c.loss_p - 0.75) < 1e-9 and c.compromised == 2
    assert evaluation_spec("C1") == compose(evaluation_spec("F1"), evaluation_spec("F4"))
    rng = np.random.default_rng(0)
    assert sample_training_spec(0, rng, 4).is_clean
    assert not sample_training_spec(2, rng, 4).is_clean
