"""Independent numerical expectations for the single float64 tensor engine."""


import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, ndtr
from scipy.stats import norm

from openecon.engines.contracts import KernelError
from openecon.engines import torch_engine


import torch


@pytest.fixture
def ols_inputs():
    rng = np.random.default_rng(230)
    x = np.column_stack([np.ones(240), rng.normal(size=(240, 3))])
    y = x @ np.array([0.5, 2.0, -1.0, 0.25]) + rng.normal(size=240) * (0.5 + x[:, 1] ** 2)
    return x, y, np.repeat(np.arange(24), 10)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3", "cluster"])
def test_ols_matches_independent_normal_equations_and_sandwich(ols_inputs, covariance):
    x, y, groups = ols_inputs
    actual = torch_engine.solve("ols", x, y, covariance, groups)
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    if covariance == "nonrobust":
        variance = bread * (residual @ residual) / (n - k)
    elif covariance == "cluster":
        cluster_score = np.stack([x[groups == g].T @ residual[groups == g] for g in np.unique(groups)])
        count = len(cluster_score)
        variance = bread @ cluster_score.T @ cluster_score @ bread * count / (count - 1) * (n - 1) / (n - k)
    else:
        score_residual = residual
        if covariance == "HC3":
            score_residual = residual / (1 - np.sum((x @ bread) * x, axis=1))
        score = x * score_residual[:, None]
        variance = bread @ score.T @ score @ bread
        if covariance == "HC1":
            variance *= n / (n - k)
    assert_allclose(actual.parameters, beta, rtol=2e-12, atol=2e-12)
    assert_allclose(actual.covariance, variance, rtol=2e-12, atol=2e-12)
    assert_allclose(actual.fitted, x @ beta, rtol=2e-12, atol=2e-12)
    assert actual.diagnostics["dtype"] == "float64"
    assert actual.diagnostics["autograd"] is False
    assert actual.solver == "torch_qr"
    assert actual.log_likelihood == pytest.approx(-0.5 * n * (np.log(2 * np.pi) + 1 + np.log(residual @ residual / n)))


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("covariance", ["nonrobust", "cluster"])
def test_binary_known_exact_mle_and_observed_information(estimator, covariance):
    # Binomial proportions .25, .50, .75 at equally spaced predictors yield
    # exact symmetric logit/probit MLEs without an estimator reference package.
    x = np.column_stack([np.ones(120), np.repeat([-1., 0., 1.], 40)])
    y = np.concatenate([np.r_[np.ones(success), np.zeros(40 - success)] for success in [10, 20, 30]])
    groups = np.tile(np.arange(10), 12)
    beta = np.array([0., np.log(3) if estimator == "logit" else norm.ppf(0.75)])
    eta = x @ beta
    p = expit(eta) if estimator == "logit" else ndtr(eta)
    if estimator == "logit":
        scores = y - p
        weights = p * (1 - p)
    else:
        signed = 2 * y - 1
        t = signed * eta
        ratio = norm.pdf(t) / norm.cdf(t)
        scores = signed * ratio
        weights = ratio * (ratio + t)
    bread = np.linalg.inv(x.T @ (x * weights[:, None]))
    variance = bread
    if covariance == "cluster":
        sums = np.stack([x[groups == g].T @ scores[groups == g] for g in np.unique(groups)])
        variance = bread @ sums.T @ sums @ bread * 10 / 9 * 119 / 118
    actual = torch_engine.solve(estimator, x, y, covariance, groups)
    assert_allclose(actual.parameters, beta, rtol=1e-9, atol=1e-10)
    assert_allclose(actual.covariance, variance, rtol=1e-9, atol=1e-11)
    assert_allclose(actual.fitted, p, rtol=1e-10, atol=1e-10)
    assert actual.log_likelihood == pytest.approx(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    assert actual.diagnostics["converged"]
    assert actual.iterations < 15


def test_normalization_preserves_extreme_units_and_offsets(ols_inputs):
    x, y, _ = ols_inputs
    transformed = x.copy()
    transformed[:, 1] = 1e12 + x[:, 1] * 1e8
    transformed[:, 2] *= 1e-12
    actual = torch_engine.solve("ols", transformed, y, "HC3")
    baseline = torch_engine.solve("ols", x, y, "HC3")
    mapping = np.diag([1., 1e-8, 1e12, 1.])
    mapping[0, 1] = -1e4
    assert_allclose(actual.parameters, mapping @ np.asarray(baseline.parameters), rtol=1e-10, atol=1e-8)
    assert_allclose(actual.covariance, mapping @ np.asarray(baseline.covariance) @ mapping.T, rtol=1e-10, atol=1e-8)
    assert_allclose(actual.fitted, baseline.fitted, rtol=1e-9, atol=1e-10)
    assert actual.condition_number < 2


def test_no_intercept_ols(ols_inputs):
    x, y, _ = ols_inputs
    x = x[:, 1:]
    actual = torch_engine.solve("ols", x, y, "nonrobust", intercept=False)
    assert_allclose(actual.parameters, np.linalg.lstsq(x, y, rcond=None)[0], rtol=1e-12)


def test_single_observation_clusters_equal_hc1(ols_inputs):
    x, y, _ = ols_inputs
    cluster = torch_engine.solve("ols", x, y, "cluster", np.arange(len(y)))
    robust = torch_engine.solve("ols", x, y, "HC1")
    assert_allclose(cluster.covariance, robust.covariance, rtol=1e-12, atol=1e-13)


def test_rank_deficiency_rejected(ols_inputs):
    x, y, _ = ols_inputs
    x[:, 3] = x[:, 1] + 2 * x[:, 2]
    with pytest.raises(KernelError, match="rank deficient") as error:
        torch_engine.solve("ols", x, y, "nonrobust")
    assert error.value.code == "singular_design"


def test_binary_nonconvergence_not_returned_as_success():
    x = np.column_stack([np.ones(40), np.linspace(-2, 2, 40)])
    y = np.tile([0., 1., 0., 1., 1.], 8)
    with pytest.raises(KernelError) as error:
        torch_engine.solve("logit", x, y, "nonrobust", max_iter=1)
    assert error.value.code == "nonconvergence"


def test_mps_is_rejected_without_precision_fallback(ols_inputs):
    x, y, _ = ols_inputs
    with pytest.raises(KernelError, match="float64") as error:
        torch_engine.solve("ols", x, y, "HC1", device="mps")
    assert error.value.code == "unsupported_device"


def test_unavailable_cuda_is_rejected(ols_inputs, monkeypatch):
    x, y, _ = ols_inputs
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(KernelError, match="no CUDA device") as error:
        torch_engine.solve("ols", x, y, "HC1", device="cuda")
    assert error.value.code == "device_unavailable"


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_tail_likelihood_score_and_information_remain_finite(estimator):
    z = torch.tensor([[-1e5], [-40.], [40.], [1e5]], dtype=torch.float64)
    y = torch.ones(4, dtype=torch.float64)
    theta = torch.ones(1, dtype=torch.float64)
    ll, score, weight, fitted = torch_engine._binary_terms(torch, estimator, z, y, theta)
    for value in [ll, score, weight, fitted]:
        assert torch.isfinite(value).all()
    assert (weight >= 0).all()
    if estimator == "probit":
        assert weight[0].item() == pytest.approx(1 - 1e-10)
        assert score[0].item() == pytest.approx(1e5 + 1e-5)
    else:
        assert score[0].item() == 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA hardware is available for real-device validation")
@pytest.mark.parametrize("estimator", ["ols", "logit", "probit"])
def test_cuda_matches_cpu_float64_on_real_device(estimator, ols_inputs):
    x, y, groups = ols_inputs
    if estimator != "ols":
        y = np.random.default_rng(232).binomial(1, expit(x @ np.array([0.1, 0.5, -0.2, 0.3]))).astype(float)
    cpu = torch_engine.solve(estimator, x, y, "cluster", groups, device="cpu")
    cuda = torch_engine.solve(estimator, x, y, "cluster", groups, device="cuda")
    assert cuda.parameters.device.type == "cuda"
    assert_allclose(cuda.parameters.cpu(), cpu.parameters, rtol=2e-9, atol=2e-10)
    assert_allclose(cuda.covariance.cpu(), cpu.covariance, rtol=2e-9, atol=2e-10)
    assert cuda.diagnostics["device"] == "cuda"
    assert cuda.diagnostics["dtype"] == "float64"
