import numpy as np
import pytest

from ricor.evaluation.metrics import auroc, brier, diagnosis_confusion, ece, far_frr, recovery_time
from ricor.evaluation.stats import bootstrap_ci, holm, welch


def test_calibration_metrics():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y) == 0.0 and brier(p, y) == 0.0
    assert ece(np.full(4, 0.5), y) == pytest.approx(0.0)
    assert ece(np.full(4, 0.9), np.array([0, 0, 0, 0])) == pytest.approx(0.9)


def test_auroc_and_rates():
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(0.5)
    far, frr = far_frr(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 0, 1, 0]))
    assert (far, frr) == (50.0, 50.0)


def test_diagnosis_confusion_rows_sum_to_100():
    m = diagnosis_confusion(
        np.array([0.9, 0.1, 0.9]), np.array([0.9, 0.9, 0.1]), np.array([1, 0, 1]), np.array([1, 1, 0])
    )
    assert np.allclose(np.diag(m), 100.0)


def test_recovery_time_detects_return_to_baseline():
    curve = np.full(100, -0.1)
    curve[30:60] = -1.0
    curve[60:70] = np.linspace(-1.0, -0.1, 10)
    assert recovery_time(curve, (30, 60), smoothing=3) == pytest.approx(9.0, abs=3)
    # Causal smoothing: a jump back to baseline is detected no earlier than it happens.
    step = np.full(100, -0.1)
    step[30:70] = -1.0
    assert recovery_time(step, (30, 60), smoothing=5) >= 10
    never = np.full(100, -0.1)
    never[30:] = -1.0
    assert np.isnan(recovery_time(never, (30, 60), smoothing=3))


def test_bootstrap_welch_holm():
    mean, lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]), n_boot=2000)
    assert mean == 2.5 and lo < mean < hi
    w = welch(np.array([1.0, 1.1, 0.9, 1.0]), np.array([2.0, 2.1, 1.9, 2.0]))
    assert w["p"] < 0.001 and w["g"] < 0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
