"""Independent identities, difficult units, and a development-only oracle."""

from types import SimpleNamespace

import numpy as np
import torch
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from openecon.engines.contracts import KernelError
from openecon.engines.torch_engine import _binary_terms as _terms, solve as _solve


def solve(*args, **kwargs):
    result = _solve(*args, **kwargs)
    # NumPy lives only in this independent numerical-oracle test boundary.
    return SimpleNamespace(parameters=np.asarray(result.parameters), covariance=np.asarray(result.covariance),
                           fitted=np.asarray(result.fitted), log_likelihood=result.log_likelihood,
                           condition_number=result.condition_number, diagnostics=result.diagnostics)


def _binary_terms(estimator, eta, y):
    design = torch.eye(len(eta), dtype=torch.float64)
    terms = _terms(torch, estimator, design, torch.as_tensor(y), torch.as_tensor(eta))
    return tuple(np.asarray(term) for term in terms)



@pytest.fixture
def design():
    rng = np.random.default_rng(429)
    x = np.column_stack([np.ones(280), rng.normal(size=(280, 3))])
    y = x @ np.array([1.3, -.4, .2, .8]) + rng.normal(size=280) * (1 + x[:, 1] ** 2)
    return x, y, np.repeat(np.arange(28), 10)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3", "cluster"])
def test_ols_against_explicit_linear_algebra(design, covariance):
    x, y, groups = design
    fitted = solve("ols", x, y, covariance, groups=groups)
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    parameters = np.linalg.solve(x.T @ x, x.T @ y)
    residual = y - x @ parameters
    if covariance == "nonrobust":
        variance = bread * (residual @ residual) / (n - k)
    elif covariance == "cluster":
        scores = np.array([x[groups == g].T @ residual[groups == g] for g in np.unique(groups)])
        variance = bread @ (scores.T @ scores) @ bread * (28 / 27) * (n - 1) / (n - k)
    else:
        weights = residual**2
        if covariance == "HC1":
            weights *= n / (n - k)
        else:
            leverage = np.einsum("ij,jk,ik->i", x, bread, x)
            weights /= (1 - leverage)**2
        variance = bread @ (x.T @ (weights[:, None] * x)) @ bread
    assert_allclose(fitted.parameters, parameters, rtol=1e-12, atol=1e-13)
    assert_allclose(fitted.covariance, variance, rtol=1e-11, atol=1e-13)
    assert_allclose(fitted.fitted, x @ parameters, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3", "cluster"])
def test_ols_unit_change_keeps_fit_and_transforms_covariance(design, covariance):
    x, y, groups = design
    original = solve("ols", x, y, covariance, groups=groups)
    multiplier = np.array([1., 1e12, 1e-12, 1e6])
    offset = np.array([0., 5e12, 2e-12, -1e6])
    changed = solve("ols", x * multiplier + offset, y, covariance, groups=groups)
    transform = np.diag(1 / multiplier)
    transform[0, 1:] = -offset[1:] / multiplier[1:]
    assert_allclose(changed.parameters, transform @ original.parameters, rtol=1e-11, atol=1e-13)
    assert_allclose(changed.covariance, transform @ original.covariance @ transform.T, rtol=1e-10, atol=1e-13)
    assert_allclose(changed.fitted, original.fitted, rtol=1e-11, atol=1e-13)
    assert changed.condition_number == pytest.approx(original.condition_number, rel=1e-12)
    assert changed.diagnostics["condition_number_basis"] == "centered_and_scaled_design"


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_derivatives_match_finite_differences(estimator):
    eta = np.array([-12., -8., -3., -.2, 0., .3, 3., 8., 12.])
    y = np.array([1., 1., 0., 1., 0., 1., 0., 0., 0.])
    step = 1e-4
    ll, score, weight, _ = _binary_terms(estimator, eta, y)
    upper = _binary_terms(estimator, eta + step, y)
    lower = _binary_terms(estimator, eta - step, y)
    difference = []
    for index in range(len(eta)):
        changed = eta.copy()
        changed[index] += step
        ll_upper = _binary_terms(estimator, changed, y)[0]
        changed[index] -= 2 * step
        ll_lower = _binary_terms(estimator, changed, y)[0]
        difference.append((ll_upper - ll_lower) / (2 * step))
    assert_allclose(score, difference, rtol=5e-8, atol=1e-8)
    assert_allclose(weight, -(upper[1] - lower[1]) / (2 * step), rtol=5e-8, atol=1e-8)
    assert np.isfinite(ll).all()
    assert (weight > 0).all()


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_extreme_tails_remain_finite(estimator):
    eta = np.array([-1e5, -40., 40., 1e5])
    y = np.array([1., 1., 0., 0.])
    ll, score, weight, fitted = _binary_terms(estimator, eta, y)
    assert np.isfinite(ll).all()
    assert np.isfinite(score).all()
    assert np.isfinite(weight).all()
    assert (weight >= 0).all()
    assert np.logical_and(fitted >= 0, fitted <= 1).all()
    if estimator == "probit":
        assert_allclose(weight[[0, 3]], np.ones(2), rtol=1e-9)
        assert_allclose(score[[0, 3]], [1e5, -1e5], rtol=1e-9)


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("covariance", ["nonrobust", "cluster"])
def test_binary_matches_development_oracle(design, estimator, covariance):
    # statsmodels is an optional development oracle, never a runtime dependency.
    sm = pytest.importorskip("statsmodels.api")
    x, _, groups = design
    rng = np.random.default_rng(456)
    y = rng.binomial(1, expit(x @ np.array([-.3, .7, -.2, .1])))
    actual = solve(estimator, x, y, covariance, groups=groups)
    cls = sm.Logit if estimator == "logit" else sm.Probit
    kwargs = {"groups": groups, "use_correction": True, "df_correction": True} if covariance == "cluster" else {}
    expected = cls(y, x).fit(disp=False, tol=1e-12, maxiter=100,
                            cov_type=covariance, use_t=False, **({"cov_kwds": kwargs} if kwargs else {}))
    assert_allclose(actual.parameters, expected.params, rtol=1e-9, atol=1e-11)
    assert_allclose(actual.covariance, expected.cov_params(), rtol=1e-9, atol=1e-11)
    assert actual.log_likelihood == pytest.approx(expected.llf, rel=1e-12)
    assert actual.diagnostics["gradient_max_per_observation"] < 1e-10


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_predictor_units_do_not_affect_optimization(design, estimator):
    x, _, groups = design
    y = np.random.default_rng(97).binomial(1, expit(x @ np.array([-.1, .4, .7, -.3])))
    original = solve(estimator, x, y, "cluster", groups=groups)
    multiplier = np.array([1., 1e10, 1e-10, 1.])
    offset = np.array([0., 3e10, -2e-10, 100.])
    transformed = solve(estimator, x * multiplier + offset, y, "cluster", groups=groups)
    transform = np.diag(1 / multiplier)
    transform[0, 1:] = -offset[1:] / multiplier[1:]
    assert_allclose(transformed.parameters, transform @ original.parameters, rtol=1e-10)
    assert_allclose(transformed.covariance, transform @ original.covariance @ transform.T, rtol=1e-10)
    assert_allclose(transformed.fitted, original.fitted, rtol=1e-11)


@pytest.mark.parametrize("estimator", ["ols", "logit", "probit"])
def test_rank_deficient_design_is_rejected(design, estimator):
    x, y, _ = design
    x[:, 3] = x[:, 1] + 2 * x[:, 2]
    if estimator != "ols":
        y = (y > y.mean()).astype(float)
    with pytest.raises(KernelError) as caught:
        solve(estimator, x, y, "nonrobust")
    assert caught.value.code == "singular_design"


def test_hc3_rejects_unit_leverage():
    x = np.column_stack([np.ones(6), [1, 0, 0, 0, 0, 0]])
    with pytest.raises(KernelError) as caught:
        solve("ols", x, np.array([1., 2., 1., 4., 5., 3.]), "HC3")
    assert caught.value.code == "undefined_hc3"


def test_binary_nonconvergence_is_an_error(design):
    x, _, _ = design
    y = np.random.default_rng(7).binomial(1, expit(x @ np.array([1., .3, -.7, .1])))
    with pytest.raises(KernelError) as caught:
        solve("logit", x, y, "nonrobust", max_iter=1)
    assert caught.value.code == "nonconvergence"


def test_no_intercept_uses_uncentered_design(design):
    x, y, _ = design
    x = x[:, 1:]
    actual = solve("ols", x, y, "nonrobust", intercept=False)
    expected = np.linalg.lstsq(x, y, rcond=None)[0]
    assert_allclose(actual.parameters, expected, rtol=1e-12)
    assert actual.diagnostics["condition_number_basis"] == "scaled_design"
    assert actual.diagnostics["column_centers"] == [0., 0., 0.]


def test_singleton_clusters_reduce_to_hc1(design):
    x, y, _ = design
    hc1 = solve("ols", x, y, "HC1")
    clustered = solve("ols", x, y, "cluster", groups=np.arange(len(y)))
    assert_allclose(clustered.covariance, hc1.covariance, rtol=1e-12)
