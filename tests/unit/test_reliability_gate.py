import torch

from ricor.core.gate import categorical_kl, decision_sensitivity, gate_weights
from ricor.core.reliability import beta_from_evidence, beta_kl_to_uniform, beta_mean_var, reliability, reliability_loss


def test_beta_moments_match_closed_form():
    a, b = beta_from_evidence(torch.tensor([3.0]), torch.tensor([1.0]))
    mean, var = beta_mean_var(a, b)
    assert torch.allclose(mean, torch.tensor([4.0 / 6.0]))
    assert torch.allclose(var, torch.tensor([4.0 * 2.0 / (36.0 * 7.0)]))


def test_no_evidence_is_uniform_with_max_uncertainty():
    a, b = beta_from_evidence(torch.zeros(1), torch.zeros(1))
    mean, var = beta_mean_var(a, b)
    assert torch.allclose(mean, torch.tensor([0.5])) and torch.allclose(var, torch.tensor([1.0 / 12.0]))
    assert torch.allclose(beta_kl_to_uniform(a, b), torch.zeros(1), atol=1e-6)


def test_lower_bound_and_ablations():
    p, r, s = torch.tensor([0.9]), torch.tensor([0.8]), torch.tensor([0.1])
    rho_bar, rho_lower = reliability(p, r, s, lambda_u=1.0)
    assert torch.allclose(rho_bar, torch.tensor([0.72])) and torch.allclose(rho_lower, torch.tensor([0.63]))
    _, joint_lower = reliability(p, r, s, 1.0, split=False)
    assert torch.allclose(joint_lower, torch.tensor([0.7]))
    point_bar, point_lower = reliability(p, r, s, 1.0, uncertainty=False)
    assert torch.equal(point_bar, point_lower)
    _, clamped = reliability(p, torch.tensor([0.05]), torch.tensor([0.2]), 1.0)
    assert clamped.item() == 0.0


def test_gate_threshold_rises_with_risk():
    rho = torch.tensor([0.8, 0.8])
    psi = torch.tensor([0.0, 0.5])
    w, tau = gate_weights(rho, psi, tau0=0.5, kappa=1.0, temperature=0.1)
    assert torch.allclose(tau, torch.tensor([0.5, 1.0]))
    assert w[0] > 0.9 and w[1] < 0.2
    w0, _ = gate_weights(rho, psi, 0.5, 0.0, 0.1)
    assert torch.allclose(w0[0], w0[1])


def test_sensitivity_is_zero_for_identical_policies_and_scales_with_cost():
    logits = torch.randn(4, 8)
    assert torch.allclose(categorical_kl(logits, logits), torch.zeros(4), atol=1e-6)
    kl = categorical_kl(logits, torch.zeros(4, 8))
    assert (kl >= 0).all()
    assert torch.allclose(decision_sensitivity(kl, torch.full((4,), 2.0)), 2 * kl)


def test_reliability_loss_prefers_correct_evidence():
    y = torch.tensor([1.0, 0.0])
    good = reliability_loss(torch.tensor([0.9, 0.1]), y, torch.tensor([9.0, 1.0]), torch.tensor([1.0, 9.0]), y, 0.05)
    bad = reliability_loss(torch.tensor([0.1, 0.9]), y, torch.tensor([1.0, 9.0]), torch.tensor([9.0, 1.0]), y, 0.05)
    assert good < bad
