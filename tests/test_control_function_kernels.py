"""Independent derivatives and complete two-stage covariance, plus admission."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import expit, gammaln
from scipy.stats import norm

from openecon.econometrics.control_function.kernels import (
    KINDS, certificate_work, evaluate_joint, fit_joint, mean_link,
)
from openecon.engines.contracts import KernelError
from openecon.resources import use_workspace_budget


def data(kind, n=160):
    rng = np.random.default_rng(1729)
    v = rng.normal(size=(n, 5))
    z = np.column_stack((np.ones(n), v[:, 0], v[:, 1]))
    d = .3 + .4*v[:, 0] + .8*v[:, 1] + v[:, 2]
    x = np.column_stack((z[:, :2], d))
    eta = .1 + .15*v[:, 0] + .2*d + .3*v[:, 2]
    if kind == "gaussian":
        y = eta + v[:, 3]
    elif kind in {"logit", "probit", "cloglog"}:
        probability = (expit(eta) if kind == "logit" else norm.cdf(eta)
                       if kind == "probit" else -np.expm1(-np.exp(eta)))
        y = (rng.uniform(size=n) < probability).astype(float)
    elif kind == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif kind == "fractional_logit":
        y = expit(eta + .5*v[:, 3])
        y[:2] = [0, 1]
    else:
        y = np.exp(eta + .45*v[:, 3])
    return tuple(torch.tensor(a, dtype=torch.float64, device="cpu") for a in (z, x, d, y))


def quantities(kind, y, eta):
    """Scalar criterion and derivatives independent of production GLM code."""
    if kind == "gaussian":
        return -.5*(y-eta)**2, y-eta, -np.ones_like(y)
    if kind in {"logit", "fractional_logit"}:
        p = expit(eta)
        return y*eta-np.logaddexp(0, eta), y-p, -p*(1-p)
    if kind == "probit":
        p, c, density = norm.cdf(eta), norm.sf(eta), norm.pdf(eta)
        a, b = density/p, density/c
        s = y*a-(1-y)*b
        return y*norm.logcdf(eta)+(1-y)*norm.logsf(eta), s, -a*b+s*(-eta-a+b)
    if kind == "cloglog":
        t = np.exp(eta)
        p = -np.expm1(-t)
        a, b = t*np.exp(-t)/p, t
        s = y*a-(1-y)*b
        return y*np.log(p)-(1-y)*t, s, -a*b+s*(1-t-a+b)
    mu = np.exp(eta)
    if kind == "poisson":
        return y*eta-mu-gammaln(y+1), y-mu, -mu
    if kind == "gamma":
        return -y/mu-eta, y/mu-1, -y/mu
    return -.5*y/mu**2+1/mu, y/mu**2-1/mu, 1/mu-2*y/mu**2


def independent_scores(theta, z, x, d, y, kind):
    kz = z.shape[1]
    gamma, beta = theta[:kz], theta[kz:]
    residual = d-z@gamma
    q = np.column_stack((x, residual))
    scalar = quantities(kind, y, q@beta)[1]
    return np.column_stack((z*residual[:, None], q*scalar[:, None]))


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("clustered", [False, True])
def test_full_observed_jacobian_general_sandwich_and_scalar_oracle(kind, clustered):
    z, x, d, y = data(kind)
    codes = torch.arange(len(d), dtype=torch.int64, device="cpu") // 8 if clustered else None
    fit = fit_joint(z, x, d, y, kind, cluster_codes=codes)
    zn, xn, dn, yn = (a.numpy() for a in (z, x, d, y))
    theta = np.concatenate((fit["gamma"].numpy(), fit["beta"].numpy()))
    rows = independent_scores(theta, zn, xn, dn, yn, kind)
    bread = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        upper, lower = theta.copy(), theta.copy()
        upper[j] += 1e-5
        lower[j] -= 1e-5
        bread[:, j] = -(independent_scores(upper, zn, xn, dn, yn, kind).sum(0)
                        - independent_scores(lower, zn, xn, dn, yn, kind).sum(0))/2e-5
    np.testing.assert_allclose(fit["bread"], bread, atol=2e-7, rtol=2e-8)
    np.testing.assert_allclose(fit["row_scores"], rows, atol=3e-12, rtol=3e-12)
    if clustered:
        units = np.zeros((20, rows.shape[1]))
        np.add.at(units, codes.numpy(), rows)
        rows = units
    meat = rows.T@rows
    covariance = np.linalg.solve(bread, np.linalg.solve(bread, meat).T).T
    np.testing.assert_allclose(fit["meat"], meat, atol=2e-10, rtol=3e-12)
    np.testing.assert_allclose(fit["joint_covariance"], covariance, atol=1e-9, rtol=1e-7)
    q, beta = fit["design"].numpy(), fit["beta"].numpy()
    criterion, score, h = quantities(kind, yn, q@beta)
    np.testing.assert_allclose(fit["scalar_score"], score, atol=3e-12)
    np.testing.assert_allclose(fit["scalar_derivative"], h, atol=3e-12)
    assert fit["criterion"].item() == pytest.approx(criterion.sum(), abs=2e-10)
    assert np.max(abs(bread-bread.T)) > .1
    assert np.max(abs(covariance[:3, 3:])) > 1e-5
    assert fit["optimizer"]["converged"] is True


def test_off_stationary_cross_derivative_keeps_design_derivative_term():
    z, x, d, y = data("gaussian")
    fit = fit_joint(z, x, d, y, "gaussian")
    gamma, beta = fit["gamma"]+.015, fit["beta"]+.03
    result = evaluate_joint(z, x, d, y, "gaussian", gamma, beta)
    naive = beta[-1]*(result["design"]*result["scalar_derivative"][:, None]).T@z
    difference = result["bread"][3:, :3]-naive
    np.testing.assert_allclose(difference[:-1], 0, atol=1e-12)
    np.testing.assert_allclose(difference[-1], result["scalar_score"]@z, atol=1e-12)
    assert difference[-1].abs().max() > 1
    theta = torch.cat((gamma, beta)).numpy()
    arrays = [a.numpy() for a in (z, x, d, y)]
    jacobian = np.empty((7, 7))
    for j in range(7):
        upper, lower = theta.copy(), theta.copy()
        upper[j] += 1e-5
        lower[j] -= 1e-5
        jacobian[:, j] = -(independent_scores(upper, *arrays, "gaussian").sum(0)
                          - independent_scores(lower, *arrays, "gaussian").sum(0))/2e-5
    np.testing.assert_allclose(result["bread"], jacobian, atol=2e-8)
    assert result["optimizer"]["method"] == "evaluation"


def test_inverse_gaussian_observed_derivative_changes_sign():
    z, x, d, y = data("inverse_gaussian")
    fit = fit_joint(z, x, d, y, "inverse_gaussian")
    # Retain enough negative curvature globally but make one row positive.
    y = y.clone()
    y[0] = fit["fitted"][0]/4
    evaluated = evaluate_joint(z, x, d, y, "inverse_gaussian", fit["gamma"], fit["beta"])
    assert evaluated["scalar_derivative"][0] > 0
    assert (evaluated["scalar_derivative"] < 0).any()


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_singleton_cluster_meat_is_exact_hc0_and_replay_without_refit(kind):
    z, x, d, y = data(kind)
    iid = fit_joint(z, x, d, y, kind)
    cluster = evaluate_joint(z, x, d, y, kind, iid["gamma"], iid["beta"],
                             cluster_codes=torch.arange(len(d), dtype=torch.int64))
    torch.testing.assert_close(iid["joint_covariance"], cluster["joint_covariance"], rtol=0, atol=0)
    torch.testing.assert_close(iid["meat"], cluster["meat"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_native_mean_link_independent_derivative_and_ambient_state(kind):
    z, x, d, y = data(kind)
    eta = torch.linspace(-2, 2, 13, dtype=torch.float64, device="cpu")
    expected = (eta.numpy() if kind == "gaussian" else expit(eta.numpy())
                if kind in {"logit", "fractional_logit"} else norm.cdf(eta.numpy())
                if kind == "probit" else -np.expm1(-np.exp(eta.numpy()))
                if kind == "cloglog" else np.exp(eta.numpy()))
    before_rng = torch.get_rng_state().clone()
    before_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            fit = fit_joint(z, x, d, y, kind)
            mu, derivative = mean_link(kind, eta)
            assert torch.empty(0).device.type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert all(value.device.type == "cpu" and value.dtype == torch.float64
                   for value in fit.values() if isinstance(value, torch.Tensor))
        assert torch.equal(torch.get_rng_state(), before_rng)
    finally:
        torch.set_default_dtype(before_dtype)
    np.testing.assert_allclose(mu, expected, atol=2e-15)
    step = 1e-5
    plus, _ = mean_link(kind, eta+step)
    minus, _ = mean_link(kind, eta-step)
    np.testing.assert_allclose(derivative, (plus-minus)/(2*step), atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("kind,bad", [
    ("logit", .5), ("probit", -.1), ("cloglog", 2),
    ("fractional_logit", 1.01), ("poisson", .5), ("poisson", -1),
    ("gamma", 0), ("inverse_gaussian", -1),
])
def test_outcome_domain_refusal(kind, bad):
    z, x, d, y = data(kind)
    y[0] = bad
    with pytest.raises(KernelError, match="outcomes"):
        fit_joint(z, x, d, y, kind)


@pytest.mark.parametrize("kind", ["logit", "probit", "cloglog", "fractional_logit", "poisson"])
def test_constant_boundary_separation_refuses(kind):
    z, x, d, y = data(kind)
    with pytest.raises(KernelError) as caught:
        fit_joint(z, x, d, torch.zeros_like(y), kind)
    assert caught.value.code == "separation_detected"


def test_complete_separation_and_rank_failure_refuse():
    z, x, d, y = data("logit")
    with pytest.raises(KernelError):
        fit_joint(z, x, d, (d > 0).double(), "logit")
    z[:, 2] = z[:, 1]
    with pytest.raises(KernelError) as caught:
        fit_joint(z, x, d, y, "logit")
    assert caught.value.code == "rank_deficient"


def test_preallocation_work_workspace_dimensions_and_numeric_types(monkeypatch):
    z, x, d, y = data("gaussian")
    import openecon.econometrics.control_function.kernels as kernels
    monkeypatch.setattr(kernels, "least_squares", lambda *a, **k: pytest.fail("allocated QR before admission"))
    with pytest.raises(KernelError) as caught:
        fit_joint(z, x, d, y, "gaussian", max_work=1)
    assert caught.value.code == "work_limit"
    with use_workspace_budget(1), pytest.raises(Exception, match="workspace"):
        fit_joint(*data("gaussian", 5000), "gaussian")
    with pytest.raises(KernelError, match="CPU float64"):
        fit_joint(z.float(), x, d, y, "gaussian")
    with pytest.raises(KernelError, match="CPU float64"):
        fit_joint(z.to("meta"), x, d, y, "gaussian")
    with pytest.raises(KernelError) as caught:
        fit_joint(*data("gaussian", 5001), "gaussian")
    assert caught.value.code == "unsupported_dimensions"


def test_no_silent_design_drop_cluster_recode_or_link_clipping():
    z, x, d, y = data("gaussian")
    broken = x.clone()
    broken[0, 1] += 1
    with pytest.raises(KernelError, match="prefix"):
        fit_joint(z, broken, d, y, "gaussian")
    with pytest.raises(KernelError, match="dense"):
        fit_joint(z, x, d, y, "gaussian", cluster_codes=torch.arange(len(d), dtype=torch.int64)+1)
    with pytest.raises(KernelError, match="more clusters"):
        fit_joint(z, x, d, y, "gaussian", cluster_codes=torch.arange(len(d), dtype=torch.int64)%4)
    with pytest.raises(KernelError, match="link domain"):
        mean_link("poisson", torch.tensor([1000.], dtype=torch.float64))


def test_zero_residual_or_outcome_variance_refuses_singular_joint_covariance():
    z, x, d, y = data("gaussian")
    zero_d = z@torch.tensor([.2, .4, .7], dtype=torch.float64)
    zero_x = torch.column_stack((x[:, :2], zero_d))
    with pytest.raises(KernelError):
        fit_joint(z, zero_x, zero_d, y, "gaussian")
    with pytest.raises(KernelError, match="positive|singular|precision"):
        fit_joint(z, x, d, torch.zeros_like(y), "gaussian")


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("exogenous", [False, True])
def test_no_intercept_with_empty_or_nonempty_shared_prefix(kind, exogenous):
    z, x, d, y = data(kind)
    if exogenous:
        z, x = z[:, 1:], x[:, 1:]
    else:
        z, x = z[:, 2:], d[:, None]
    fit = fit_joint(z, x, d, y, kind)
    replay = evaluate_joint(z, x, d, y, kind, fit["gamma"], fit["beta"])
    torch.testing.assert_close(fit["joint_covariance"], replay["joint_covariance"], rtol=0, atol=0)
    np.testing.assert_allclose(fit["gamma"], np.linalg.lstsq(z.numpy(), d.numpy(), rcond=None)[0], atol=2e-12)
    assert fit["optimizer"]["converged"] is True


def test_cloglog_small_tail_at_finite_stationary_fit_is_not_separation():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/"scripts/verify_control_function_oracles.py"
    spec = importlib.util.spec_from_file_location("cf_tail_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    inputs = oracle.prepare_inputs(oracle.fixture("cloglog"))
    z, x, d, y = (torch.tensor(inputs[name], dtype=torch.float64) for name in ("z", "x", "d", "y"))
    fitted = fit_joint(z, x, d, y, "cloglog")
    expected = oracle.fit_oracle("cloglog", oracle.fixture("cloglog"))
    np.testing.assert_allclose(fitted["beta"], expected["beta"], atol=2e-8)
    assert np.min(np.exp(-np.exp(fitted["design"].numpy()@fitted["beta"].numpy()))) < 1e-10


def separated_data(quasi=False):
    rng = np.random.default_rng(637)
    n = 180
    w = np.where(np.arange(n) % 2, 1., -1.) * (1 + .01*rng.uniform(size=n))
    if quasi:
        w[:40] = 0
    v, u = rng.normal(size=(2, n))
    d = .3 + .5*w + .8*v + u
    z = np.column_stack((np.ones(n), w, v))
    x = np.column_stack((np.ones(n), w, d))
    y = (w > 0).astype(float)
    if quasi:
        y[:40] = np.arange(40) % 2
    return tuple(torch.tensor(a, dtype=torch.float64, device="cpu") for a in (z, x, d, y))


@pytest.mark.parametrize("kind", ["logit", "probit", "cloglog", "fractional_logit"])
@pytest.mark.parametrize("quasi", [False, True])
def test_globally_verified_separation_refuses_even_loose_optimizer_tolerance(kind, quasi, monkeypatch):
    import openecon.econometrics.control_function.kernels as kernels
    z, x, d, y = separated_data(quasi)
    if quasi and kind == "fractional_logit":
        y[:40] = torch.linspace(.2, .8, 40, dtype=torch.float64)
    gamma = torch.tensor(np.linalg.lstsq(z.numpy(), d.numpy(), rcond=None)[0], dtype=torch.float64)
    q = torch.column_stack((x, d-z@gamma))
    # A hand-specified recession direction verifies every original row.
    direction = torch.tensor([0., 1., 0., 0.], dtype=torch.float64)
    signed = (2*(y == 1).double()-1)*(q@direction)
    assert (signed >= 0).all() and (signed > 0).any()
    if quasi:
        assert (q[:40]@direction == 0).all()
    else:
        assert (signed > 0).all()
    monkeypatch.setattr(kernels, "maximize_newton", lambda *a, **k: pytest.fail("separated outcome entered Newton"))
    for call in (
        lambda: fit_joint(z, x, d, y, kind, max_iterations=200, tolerance=1e-4),
        lambda: evaluate_joint(z, x, d, y, kind, gamma, torch.zeros(4, dtype=torch.float64)),
    ):
        with pytest.raises(KernelError) as caught:
            call()
        assert caught.value.code == "separation_detected"


def test_fractional_interiors_pin_the_direction_and_retain_finite_corner_fit():
    z, x, d, y = separated_data()
    y[:30] = torch.linspace(.2, .8, 30, dtype=torch.float64)
    fit = fit_joint(z, x, d, y, "fractional_logit")
    assert fit["optimizer"]["converged"] is True
    assert fit["optimizer"]["separation_dual_upper_bound"] <= 1e-8
    assert fit["optimizer"]["separation_peak_constraints"] <= fit["optimizer"]["separation_constraint_capacity"]
    assert fit["resource_plan"]["buffers"]["separation_scaled_design_and_replay"] > 0
    criterion, score, derivative = quantities("fractional_logit", y.numpy(), fit["design"].numpy()@fit["beta"].numpy())
    np.testing.assert_allclose(fit["scalar_score"], score, atol=2e-12)
    np.testing.assert_allclose(fit["scalar_derivative"], derivative, atol=2e-12)
    assert fit["criterion"].item() == pytest.approx(criterion.sum(), abs=2e-11)


def test_certificate_budget_refuses_fit_and_semantic_replay_before_allocation(monkeypatch):
    import openecon.econometrics.control_function.kernels as kernels
    z, x, d, y = data("logit")
    gamma = torch.tensor(np.linalg.lstsq(z.numpy(), d.numpy(), rcond=None)[0], dtype=torch.float64)
    cost = certificate_work(len(d), x.shape[1]+1, "logit")
    assert cost > 0 and certificate_work(len(d), x.shape[1]+1, "gaussian") == 0
    monkeypatch.setattr(kernels, "least_squares", lambda *a, **k: pytest.fail("allocated QR before certificate budget"))
    monkeypatch.setattr(kernels, "certify_separation", lambda *a, **k: pytest.fail("ran certificate before budget"))
    for call in (
        lambda: fit_joint(z, x, d, y, "logit", max_work=cost-1),
        lambda: evaluate_joint(z, x, d, y, "logit", gamma, torch.zeros(4, dtype=torch.float64), max_work=cost-1),
    ):
        with pytest.raises(KernelError) as caught:
            call()
        assert caught.value.code == "work_limit"
