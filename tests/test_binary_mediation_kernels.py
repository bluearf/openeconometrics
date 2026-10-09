"""Kernel-level likelihood derivatives, full covariance, units and ML refusals."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import special
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.decomposition import binary_kernels as kernel


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sample(link="logit", outcome="gaussian", n=250, seed=128):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, 2))
    a = rng.binomial(1, .5, n).astype(float)
    eta = -.3 + .65 * a + .3 * c[:, 0] - .25 * c[:, 1]
    probability = special.expit(eta) if link == "logit" else special.ndtr(eta)
    m = rng.binomial(1, probability).astype(float)
    eta_y = -.15 + .3 * a + .55 * m - .25 * a * m + .2 * c[:, 0] + .15 * c[:, 1]
    if outcome == "gaussian":
        y = eta_y + rng.normal(size=n) * .8
    elif outcome == "poisson":
        y = rng.poisson(np.exp(eta_y)).astype(float)
    else:
        probability_y = special.expit(eta_y) if outcome == "logit" else special.ndtr(eta_y)
        y = rng.binomial(1, probability_y).astype(float)
    return tuple(torch.tensor(v, dtype=torch.float64, device="cpu") for v in (y, m, a, c))


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("outcome", ["gaussian", "logit", "probit", "poisson"])
@pytest.mark.parametrize("interaction", [False, True])
def test_full_analytic_scores_hessian_inverse_and_paired_hc0(link, outcome, interaction):
    y, m, a, c = sample(link, outcome)
    result = kernel.fit_joint(y, m, a, c, mediator_link=link, outcome_model=outcome,
                              interaction=interaction, covariance="HC0")
    theta = torch.tensor(result["theta"], dtype=torch.float64, device="cpu")

    def likelihood(point):
        return kernel.row_loglikelihood(point, y, m, a, c, mediator_link=link,
                                        outcome_model=outcome, interaction=interaction)

    jac = torch.autograd.functional.jacobian(likelihood, theta).sum(1)
    information = -torch.autograd.functional.hessian(lambda p: likelihood(p).sum(), theta)
    observed = torch.tensor(result["information"], dtype=torch.float64, device="cpu")
    scores = torch.tensor(result["scores"], dtype=torch.float64, device="cpu")
    bread = torch.tensor(result["bread"], dtype=torch.float64, device="cpu")
    covariance = torch.tensor(result["covariance"], dtype=torch.float64, device="cpu")
    torch.testing.assert_close(scores, jac, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(observed, information, rtol=2e-11, atol=2e-10)
    torch.testing.assert_close(bread, torch.linalg.inv(information), rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(covariance, bread @ jac.T @ jac @ bread, rtol=2e-10, atol=2e-10)
    qm = len(c[0]) + 2
    assert torch.count_nonzero(observed[:qm, qm:]) == 0
    assert float((jac[:, :qm].T @ jac[:, qm:]).abs().max()) > 1e-4
    assert result["convergence"]["converged"]
    assert result["parameter_slices"] == {"mediator": [0, qm], "outcome": [qm, len(theta)]}
    np.testing.assert_allclose(result["row_loglikelihood"], likelihood(theta).tolist(), rtol=1e-13, atol=1e-13)
    json.dumps(result, allow_nan=False)
    restored = kernel.evaluate_joint(theta, y, m, a, c, mediator_link=link,
                                     outcome_model=outcome, interaction=interaction, covariance="HC0")
    for key in ("theta", "scores", "information", "bread", "covariance", "row_loglikelihood"):
        assert result[key] == restored[key]


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("outcome", ["gaussian", "logit", "probit", "poisson"])
def test_original_control_units_affine_reparameterization_full_matrices(link, outcome):
    y, m, a, c = sample(link, outcome)
    baseline = kernel.fit_joint(y, m, a, c, mediator_link=link, outcome_model=outcome, covariance="OIM")
    scales = torch.tensor([1e-4, 3e5], dtype=torch.float64, device="cpu")
    shifts = torch.tensor([.05, -2e6], dtype=torch.float64, device="cpu")
    changed = kernel.fit_joint(y, m, a, c * scales + shifts,
                              mediator_link=link, outcome_model=outcome, covariance="OIM")
    p = len(baseline["theta"])
    transform = torch.eye(p, dtype=torch.float64, device="cpu")
    qm = 4
    for start, cstart in ((0, 2), (qm, 4)):
        transform[start, start+cstart:start+cstart+2] = -shifts / scales
        for j in range(2):
            transform[start+cstart+j, start+cstart+j] = 1 / scales[j]
    base_theta = torch.tensor(baseline["theta"], dtype=torch.float64, device="cpu")
    cov = torch.tensor(baseline["covariance"], dtype=torch.float64, device="cpu")
    torch.testing.assert_close(torch.tensor(changed["theta"], dtype=torch.float64, device="cpu"),
                               transform @ base_theta, rtol=3e-8, atol=3e-8)
    torch.testing.assert_close(torch.tensor(changed["covariance"], dtype=torch.float64, device="cpu"),
                               transform @ cov @ transform.T, rtol=3e-8, atol=3e-8)
    np.testing.assert_allclose(changed["row_loglikelihood"], baseline["row_loglikelihood"], rtol=3e-9, atol=3e-9)


def test_gaussian_normalized_ml_log_sigma_is_full_nuisance_not_residual_df_scale():
    y, m, a, c = sample()
    result = kernel.fit_joint(y, m, a, c, covariance="OIM")
    theta = torch.tensor(result["theta"], dtype=torch.float64, device="cpu")
    xm, xy = kernel._designs(m, a, c, True)
    beta = torch.linalg.lstsq(xy, y, driver="gelsd").solution
    sigma2 = (y - xy @ beta).square().mean()
    assert theta[-1] == pytest.approx(float(sigma2.log() / 2), rel=1e-11)
    expected = -.5 * len(y) * (1 + math.log(2 * math.pi) + float(sigma2.log()))
    assert result["equation_log_likelihood"][1] == pytest.approx(expected, rel=1e-11)
    assert result["information"][-1][-1] == pytest.approx(2 * len(y), rel=1e-11)
    assert result["parameter_names"][-1] == "outcome:log_sigma"
    assert result["covariance"] == result["bread"]


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_complete_separation_is_never_reported_as_finite_stationary_mle(link):
    y, _, a, c = sample()
    m = (c[:, 0] > 0).to(torch.float64)
    with pytest.raises(AnalysisError):
        kernel.fit_joint(y, m, a, c, mediator_link=link, max_iterations=100)


def test_poisson_zero_group_separation_is_not_silently_dropped_or_ridged():
    _, m, a, c = sample()
    y = (1 - m) * 2
    with pytest.raises(AnalysisError):
        kernel.fit_joint(y, m, a, c, outcome_model="poisson", max_iterations=100)


@pytest.mark.parametrize("failure", ["fractional_count", "negative_count", "all_zero_count", "nonbinary_m", "one_arm", "nan", "collinear", "constant_control", "perfect_gaussian", "oversized", "work", "bool_input"])
def test_input_information_and_resource_refusals_are_structured(failure):
    y, m, a, c = sample()
    options = {}
    if failure in {"fractional_count", "negative_count", "all_zero_count"}:
        y = torch.ones_like(y)
        options["outcome_model"] = "poisson"
        if failure == "fractional_count":
            y[0] = .5
        elif failure == "negative_count":
            y[0] = -1
        else:
            y.zero_()
    elif failure == "nonbinary_m":
        m[0] = 2
    elif failure == "one_arm":
        a.zero_()
    elif failure == "nan":
        c[0, 0] = float("nan")
    elif failure == "collinear":
        c[:, 1] = c[:, 0] * 2
    elif failure == "constant_control":
        c[:, 1] = 2
    elif failure == "perfect_gaussian":
        y = .5 + a + m + .25 * a * m
    elif failure == "oversized":
        y, m, a, c = sample(n=4097)
    elif failure == "work":
        y, m, a, _ = sample(n=4096)
        c = torch.randn((4096, 10), dtype=torch.float64, device="cpu")
        options["max_iterations"] = 1000
    else:
        a = a.to(torch.bool)
    with pytest.raises(AnalysisError):
        kernel.fit_joint(y, m, a, c, **options)


@pytest.mark.parametrize("options", [dict(covariance="HC1"), dict(max_iterations=True), dict(max_iterations=0), dict(tolerance=0.), dict(tolerance=float("nan")), dict(interaction=1), dict(mediator_link="cloglog"), dict(outcome_model="linear")])
def test_invalid_options_refused(options):
    with pytest.raises(AnalysisError):
        kernel.fit_joint(*sample(), **options)


def test_host_global_torch_defaults_do_not_change_native_dtype_device_or_thread_count():
    inputs = sample("probit", "poisson")
    baseline = kernel.fit_joint(*inputs, mediator_link="probit", outcome_model="poisson")
    before_dtype = torch.get_default_dtype()
    before_device = torch.get_default_device()
    before_threads = torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        changed = kernel.fit_joint(*inputs, mediator_link="probit", outcome_model="poisson")
        assert changed == baseline
        assert torch.get_default_dtype() == torch.float32
        assert str(torch.get_default_device()) == "meta"
        assert torch.get_num_threads() == before_threads
    finally:
        torch.set_default_dtype(before_dtype)
        torch.set_default_device(before_device)


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("outcome", ["gaussian", "logit", "probit", "poisson"])
def test_saved_stationarity_recomputes_score_and_parameter_step_without_fitting(link, outcome, monkeypatch):
    y, m, a, c = sample(link, outcome)
    fitted = kernel.fit_joint(y, m, a, c, mediator_link=link, outcome_model=outcome)
    theta = torch.tensor(fitted["theta"], dtype=torch.float64, device="cpu")

    def no_fit(*args, **kwargs):
        raise AssertionError("Stationarity replay must not optimize.")

    monkeypatch.setattr(kernel, "_glm_fit", no_fit)
    monkeypatch.setattr(kernel, "_gaussian_fit", no_fit)
    check = kernel.stationarity_check(theta, y, m, a, c, mediator_link=link, outcome_model=outcome)
    assert check["mediator"]["stationary"] and check["outcome"]["stationary"]
    for j in (1, 4 + 1, len(theta) - 1):
        changed = theta.clone()
        changed[j] += .15
        # Recomputing a complete positive-information state at arbitrary theta
        # is possible; it must not turn that state into an accepted ML fit.
        kernel.evaluate_joint(changed, y, m, a, c, mediator_link=link, outcome_model=outcome)
        with pytest.raises(AnalysisError, match="stationarity"):
            kernel.stationarity_check(changed, y, m, a, c, mediator_link=link, outcome_model=outcome)


def test_stationarity_step_refuses_tiny_scores_on_a_separated_likelihood():
    n = 240
    a = torch.arange(n, device="cpu").remainder(2).to(torch.float64)
    c = torch.linspace(-1, 1, n, dtype=torch.float64, device="cpu")[:, None]
    m = (c[:, 0] > 0).to(torch.float64)
    y = torch.sin(torch.arange(n, dtype=torch.float64, device="cpu"))
    # A logistic threshold with a large slope has a small score/decrement but
    # an appreciable parameter step. No caller metadata can certify it finite.
    theta = torch.tensor([0., 0., 1000., 0., 0., 0., 0., 0., 0.], dtype=torch.float64, device="cpu")
    with pytest.raises(AnalysisError):
        kernel.stationarity_check(theta, y, m, a, c, tolerance=1e-5)


def test_high_gaussian_outcome_offset_refuses_unrestorable_parameter_representation():
    y, m, a, c = sample()
    with pytest.raises(AnalysisError, match="Original-unit parameter representation") as failure:
        kernel.fit_joint(y + 1e8, m, a, c)
    assert failure.value.code == "numerical_failure"
