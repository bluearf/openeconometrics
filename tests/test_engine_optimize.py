"""Independent oracles for the maximum-likelihood optimizers and covariance helpers.

Likelihoods and their analytic derivatives are written here in torch; solutions,
Hessians and covariances are compared with statsmodels, SciPy and closed forms.
"""

import ast
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.optimize import (
    OptimResult,
    check_derivatives,
    information_inverse,
    maximize_bfgs,
    maximize_newton,
    numerical_gradient,
    numerical_hessian,
    opg,
)


def tensor(values):
    return torch.as_tensor(np.asarray(values, dtype=float), dtype=torch.float64)


def logit_objective(x, y):
    x, y = tensor(x), tensor(y)

    def fn(theta):
        eta = x @ theta
        value = (y * torch.nn.functional.logsigmoid(eta)
                 + (1 - y) * torch.nn.functional.logsigmoid(-eta)).sum()
        p = torch.sigmoid(eta)
        return value, x.T @ (y - p), -(x.T @ (x * (p * (1 - p))[:, None]))

    return fn


def poisson_objective(x, y):
    x, y = tensor(x), tensor(y)

    def value_fn(theta):
        eta = x @ theta
        return (y * eta - torch.exp(eta) - torch.lgamma(y + 1)).sum()

    def fn(theta):
        mu = torch.exp(x @ theta)
        return value_fn(theta), x.T @ (y - mu), -(x.T @ (x * mu[:, None]))

    return fn, value_fn


def normal_objective(y):
    """theta = (mu, log sigma)."""
    y = tensor(y)
    n = len(y)

    def fn(theta):
        error = y - theta[0]
        precision = torch.exp(-2 * theta[1])
        quadratic = (error * error).sum() * precision
        linear = error.sum() * precision
        value = -0.5 * n * math.log(2 * math.pi) - n * theta[1] - 0.5 * quadratic
        gradient = torch.stack([linear, quadratic - n])
        hessian = torch.stack([torch.stack([-n * precision, -2 * linear]),
                               torch.stack([-2 * linear, -2 * quadratic])])
        return value, gradient, hessian

    return fn


def negative_binomial_objective(x, y):
    """NB2 with theta = (beta, log alpha), Var(y) = mu + alpha mu^2 (Stata's nbreg /lnalpha)."""
    x, y = tensor(x), tensor(y)

    def fn(theta):
        beta, a = theta[:-1], torch.exp(-theta[-1])          # a = 1 / alpha
        eta = x @ beta
        mu = torch.exp(eta)
        total = a + mu
        value = (torch.lgamma(y + a) - torch.lgamma(a) - torch.lgamma(y + 1)
                 + a * (torch.log(a) - torch.log(total)) + y * (eta - torch.log(total))).sum()
        d_a = (torch.digamma(y + a) - torch.digamma(a) + torch.log(a) - torch.log(total)
               + (mu - y) / total)
        d_aa = (torch.polygamma(1, y + a) - torch.polygamma(1, a) + 1 / a - 1 / total
                - (mu - y) / total**2)
        gradient = torch.cat([x.T @ ((y - mu) * a / total), (-a * d_a).sum()[None]])
        h_bb = -(x.T @ (x * (mu * a * (a + y) / total**2)[:, None]))
        h_bt = x.T @ (-(y - mu) * a * mu / total**2)
        h_tt = (a * d_a + a * a * d_aa).sum()
        hessian = torch.cat([torch.cat([h_bb, h_bt[:, None]], dim=1),
                             torch.cat([h_bt, h_tt[None]])[None, :]], dim=0)
        return value, gradient, hessian

    return fn


def rosenbrock_objective(theta):
    """Negated Rosenbrock function: maximum 0 at (1, 1)."""
    a, b = theta[0], theta[1]
    value = -((1 - a) ** 2 + 100 * (b - a * a) ** 2)
    gradient = torch.stack([2 * (1 - a) + 400 * a * (b - a * a), -200 * (b - a * a)])
    hessian = torch.stack([torch.stack([-2 + 400 * b - 1200 * a * a, 400 * a]),
                           torch.stack([400 * a, torch.full_like(a, -200.0)])])
    return value, gradient, hessian


@pytest.fixture(scope="module")
def logit_data():
    rng = np.random.default_rng(20261002)
    x = np.column_stack([np.ones(1500), rng.normal(size=(1500, 4))])
    beta = np.array([0.3, 1.0, -0.8, 0.5, 0.0])
    y = (rng.uniform(size=1500) < 1 / (1 + np.exp(-x @ beta))).astype(float)
    return x, y


@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(7)
    x = np.column_stack([np.ones(3000), rng.normal(size=(3000, 3))])
    y = rng.poisson(np.exp(x @ np.array([5.0, 1.0, -0.8, 0.5]))).astype(float)
    return x, y


@pytest.fixture(scope="module")
def normal_data():
    return np.random.default_rng(11).normal(50.0, 5.0, size=500)


@pytest.fixture(scope="module")
def count_data():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(2000), rng.normal(size=(2000, 2))])
    mu = np.exp(x @ np.array([1.0, 0.5, -0.3]))
    alpha = 0.7
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(float)
    return x, y


@pytest.fixture(scope="module")
def negative_binomial_reference(count_data):
    x, y = count_data
    model = sm.NegativeBinomial(y, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(method="newton", maxiter=200, tol=1e-13, disp=0)
    return model, fit


def covariance(result):
    return information_inverse(-result.hessian).numpy()


# --- Newton-Raphson ---------------------------------------------------------


def test_newton_logit_matches_statsmodels(logit_data):
    x, y = logit_data
    start = torch.zeros(5, dtype=torch.float64)
    result = maximize_newton(logit_objective(x, y), start)
    reference = sm.Logit(y, x).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    assert isinstance(result, OptimResult) and result.converged
    assert_allclose(result.theta.numpy(), reference.params, rtol=1e-9, atol=1e-10)
    assert_allclose(covariance(result), reference.cov_params(), rtol=1e-9, atol=1e-13)
    assert result.value == pytest.approx(reference.llf, rel=1e-13)
    assert result.method == "newton_cholesky_marquardt_backtracking"
    assert result.theta.dtype == torch.float64 and not result.theta.requires_grad
    assert_allclose(result.hessian.numpy(), result.hessian.numpy().T, rtol=0, atol=0)
    assert torch.equal(start, torch.zeros(5, dtype=torch.float64))        # theta0 untouched
    diagnostics = result.diagnostics
    assert diagnostics["gradient_max"] <= 1e-8 * abs(result.value)
    assert diagnostics["scaled_gradient"] <= 1e-10
    assert diagnostics["nonconcave_iterations"] == 0 and diagnostics["backtracks"] == 0
    assert diagnostics["function_evaluations"] == result.iterations + 1
    assert diagnostics["value_history"][-1] == result.value
    assert len(diagnostics["value_history"]) == result.iterations + 1
    assert np.all(np.diff(diagnostics["value_history"]) >= 0)
    assert "converged" in diagnostics["message"]
    result.theta[0] = 0.0                                # results are ordinary, writable tensors


def test_newton_large_logit_takes_few_iterations_without_extra_evaluations():
    rng = np.random.default_rng(1)
    n, k = 20000, 50
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x @ (0.3 * rng.normal(size=k))))).astype(float)
    fn = logit_objective(x, y)
    started = time.perf_counter()
    result = maximize_newton(fn, torch.zeros(k, dtype=torch.float64))
    elapsed = time.perf_counter() - started
    assert result.converged and result.iterations <= 8
    assert result.diagnostics["function_evaluations"] == result.iterations + 1
    assert result.diagnostics["backtracks"] == 0
    assert elapsed < 5.0
    reference = sm.Logit(y, x).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    assert_allclose(result.theta.numpy(), reference.params, rtol=1e-9, atol=1e-11)
    assert_allclose(np.sqrt(np.diag(covariance(result))), reference.bse, rtol=1e-9)


def test_newton_poisson_backtracks_through_overflowing_trials(poisson_data):
    x, y = poisson_data
    fn, value_fn = poisson_objective(x, y)
    trial_values = []

    def recording(theta):
        output = fn(theta)
        trial_values.append(float(output[0]))
        return output

    # From zero the first Newton step is the OLS fit of y - 1: exp() overflows.
    result = maximize_newton(recording, torch.zeros(4, dtype=torch.float64))
    assert not all(math.isfinite(value) for value in trial_values)
    assert result.converged and result.diagnostics["backtracks"] > 0
    reference = sm.GLM(y, x, family=sm.families.Poisson()).fit(tol=1e-14, maxiter=500)
    assert_allclose(result.theta.numpy(), reference.params, rtol=1e-9, atol=1e-10)
    assert_allclose(covariance(result), reference.cov_params(), rtol=1e-8, atol=1e-15)
    assert result.value == pytest.approx(reference.llf, rel=1e-12)

    cheap = maximize_newton(fn, torch.zeros(4, dtype=torch.float64), value_fn=value_fn)
    assert_allclose(cheap.theta.numpy(), result.theta.numpy(), rtol=1e-12, atol=1e-14)
    assert cheap.iterations == result.iterations
    # Rejected trials are evaluated without derivatives once value_fn is available.
    assert cheap.diagnostics["derivative_evaluations"] \
        < result.diagnostics["derivative_evaluations"]


def test_newton_normal_far_start_crosses_a_nonconcave_region(normal_data):
    y = normal_data
    fn = normal_objective(y)
    start = torch.zeros(2, dtype=torch.float64)
    _, _, hessian = fn(start)
    assert np.linalg.eigvalsh(hessian.numpy()).max() > 0          # indefinite at the start
    result = maximize_newton(fn, start)
    n, sigma = len(y), y.std()
    assert result.converged and result.diagnostics["nonconcave_iterations"] > 0
    assert_allclose(result.theta.numpy(), [y.mean(), math.log(sigma)], rtol=1e-11)
    assert_allclose(covariance(result), np.diag([sigma**2 / n, 1 / (2 * n)]), rtol=1e-9,
                    atol=1e-12)
    expected = -0.5 * n * (math.log(2 * math.pi) + 1) - n * math.log(sigma)
    assert result.value == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("start", [
    [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, -8.0], [3.0, -2.0, 2.0, 4.0],
])
def test_newton_negative_binomial_log_alpha(count_data, negative_binomial_reference, start):
    x, y = count_data
    model, reference = negative_binomial_reference
    result = maximize_newton(negative_binomial_objective(x, y), tensor(start))
    assert result.converged and result.iterations <= 25
    beta, alpha = result.theta[:-1].numpy(), math.exp(result.theta[-1])
    assert_allclose(np.r_[beta, alpha], reference.params, rtol=1e-8)
    assert result.value == pytest.approx(reference.llf, rel=1e-12)
    # statsmodels parameterizes alpha itself: H_log = J' H_alpha J at the optimum.
    jacobian = np.diag(np.r_[np.ones(3), alpha])
    expected = jacobian @ model.hessian(np.r_[beta, alpha]) @ jacobian
    assert_allclose(result.hessian.numpy(), expected, rtol=1e-7, atol=1e-7)
    assert_allclose(jacobian @ covariance(result) @ jacobian, reference.cov_params(), rtol=1e-6)


def test_newton_negative_binomial_starts_not_concave(count_data):
    x, y = count_data
    fn = negative_binomial_objective(x, y)
    start = tensor([0.0, 0.0, 0.0, -8.0])
    assert np.linalg.eigvalsh(fn(start)[2].numpy()).max() > 0
    result = maximize_newton(fn, start)
    assert result.converged and result.diagnostics["nonconcave_iterations"] > 0
    assert result.diagnostics["concave"] is True


@pytest.mark.parametrize("scales", [[1, 1e-4, 1, 1e5], [1, 1e-3, 1e3, 1e6], [1, 1e-8, 1, 1e9]])
def test_newton_ill_scaled_logit_from_zero(scales):
    rng = np.random.default_rng(5)
    z = np.column_stack([np.ones(4000), rng.normal(size=(4000, 3))])
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z @ [0.3, 1.0, -1.0, 0.5]))).astype(float)
    scales = np.asarray(scales, dtype=float)
    reference = sm.Logit(y, z).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    result = maximize_newton(logit_objective(z * scales, y), torch.zeros(4, dtype=torch.float64))
    assert result.converged and result.iterations <= 8
    assert result.diagnostics["backtracks"] == 0
    assert_allclose(result.theta.numpy() * scales, reference.params, rtol=1e-9)
    assert_allclose(covariance(result) * np.outer(scales, scales), reference.cov_params(),
                    rtol=1e-7, atol=1e-12)


def test_newton_converges_at_float64_resolution_when_unit_dependent_rules_cannot_hold():
    rng = np.random.default_rng(11)
    z = np.column_stack([np.ones(5000), rng.normal(size=(5000, 3))])
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-z @ [0.3, 1.0, -1.0, 0.5]))).astype(float)
    scales = np.array([1.0, 1e-13, 1.0, 1e13])
    result = maximize_newton(logit_objective(z * scales, y), torch.zeros(4, dtype=torch.float64))
    reference = sm.Logit(y, z).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    assert result.converged
    # The raw gradient of the 1e13-scaled column is pure rounding noise above gradient_tol.
    assert result.diagnostics["gradient_max"] > 1e-8 * abs(result.value)
    assert "resolution" in result.diagnostics["message"]
    assert result.diagnostics["scaled_gradient"] <= 1e-10
    assert_allclose(result.theta.numpy() * scales, reference.params, rtol=1e-9)


def test_newton_accepts_steps_on_slope_when_the_objective_is_noisy(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    clean = maximize_newton(fn, torch.zeros(5, dtype=torch.float64))

    def noisy(theta):
        value, gradient, hessian = fn(theta)
        # A large cancelling constant leaves rounding noise of about 1e-7 in the value,
        # far above the gain of the last Newton steps; the derivatives stay exact.
        return (value + 1e9) - 1e9, gradient, hessian

    result = maximize_newton(noisy, torch.zeros(5, dtype=torch.float64))
    assert result.converged and result.iterations <= clean.iterations + 2
    assert result.diagnostics["backtracks"] == 0
    assert_allclose(result.theta.numpy(), clean.theta.numpy(), rtol=1e-9, atol=1e-11)
    bfgs = maximize_bfgs(noisy, torch.zeros(5, dtype=torch.float64))
    assert bfgs.converged and bfgs.diagnostics["function_evaluations"] < 40
    assert_allclose(bfgs.theta.numpy(), clean.theta.numpy(), rtol=1e-6, atol=1e-7)


def test_newton_single_parameter_and_start_at_the_maximum():
    def fn(theta):
        d = theta - 3
        return -(d**4).sum() - (d**2).sum(), -4 * d**3 - 2 * d, (-12 * d**2 - 2).reshape(1, 1)

    result = maximize_newton(fn, tensor([10.0]))
    assert result.converged and result.theta.shape == (1,)
    assert result.theta.item() == pytest.approx(3.0, abs=1e-12)
    assert result.hessian.shape == (1, 1)
    again = maximize_newton(fn, result.theta)
    assert again.converged and again.iterations == 0
    assert again.diagnostics["function_evaluations"] == 1
    assert again.diagnostics["last_step_max"] == 0.0


def test_newton_accepts_float_values_and_reports_through_callback(normal_data):
    fn = normal_objective(normal_data)
    seen = []

    def as_float(theta):
        value, gradient, hessian = fn(theta)
        return float(value), gradient, hessian

    result = maximize_newton(
        as_float, tensor([40.0, 2.0]),
        callback=lambda i, theta, value: seen.append((i, theta.clone(), value)))
    assert [entry[0] for entry in seen] == list(range(1, result.iterations + 1))
    assert seen[-1][2] == result.value and torch.equal(seen[-1][1], result.theta)
    assert result.diagnostics["value_history"][1:] == [entry[2] for entry in seen]


def test_newton_reports_nonconvergence(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    with pytest.raises(KernelError) as error:
        maximize_newton(fn, torch.zeros(5, dtype=torch.float64), max_iter=1)
    assert error.value.code == "nonconvergence"
    result = maximize_newton(fn, torch.zeros(5, dtype=torch.float64), max_iter=1,
                             raise_on_failure=False)
    assert not result.converged and result.iterations == 1
    assert "did not converge in 1 iterations" in result.diagnostics["message"]
    assert result.value > float(fn(torch.zeros(5, dtype=torch.float64))[0])


def test_newton_never_converges_where_the_hessian_is_not_negative_definite():
    def saddle(theta):
        value = theta[0] ** 2 - theta[1] ** 2
        hessian = torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
        return value, torch.stack([2 * theta[0], -2 * theta[1]]), hessian

    # A stationary saddle: the gradient test alone would call this converged.
    with pytest.raises(KernelError) as error:
        maximize_newton(saddle, torch.zeros(2, dtype=torch.float64))
    assert error.value.code == "nonconvergence" and "not negative definite" in str(error.value)
    result = maximize_newton(saddle, torch.zeros(2, dtype=torch.float64), raise_on_failure=False)
    assert not result.converged and result.diagnostics["scaled_gradient"] is None

    def double_well(theta):
        x = theta[0]
        return -(x * x - 1) ** 2, (-4 * x * (x * x - 1)).reshape(1), (4 - 12 * x * x).reshape(1, 1)

    result = maximize_newton(double_well, tensor([0.1]))           # convex around the start
    assert result.converged and result.diagnostics["nonconcave_iterations"] >= 1
    assert result.theta.item() == pytest.approx(1.0, abs=1e-9)


def test_newton_reports_a_monotone_likelihood_as_nonconvergence():
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(400), rng.normal(size=400)])
    y = (x[:, 1] > 0).astype(float)                       # perfectly separated
    with pytest.raises(KernelError) as error:
        maximize_newton(logit_objective(x, y), torch.zeros(2, dtype=torch.float64), max_iter=60)
    assert error.value.code == "nonconvergence"


def test_newton_stops_when_derivatives_are_inconsistent(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)

    def wrong(theta):
        value, gradient, hessian = fn(theta)
        return value, -gradient, hessian

    evaluations = []
    with pytest.raises(KernelError) as error:
        maximize_newton(lambda theta: evaluations.append(1) or wrong(theta),
                        tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
    assert error.value.code == "nonconvergence"
    assert len(evaluations) < 150                         # no 200-iteration grind


def test_newton_invalid_start_and_options(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)

    def log_objective(theta):
        return torch.log(theta[0]), 1 / theta, (-1 / theta**2).reshape(1, 1)

    with pytest.raises(KernelError) as error:
        maximize_newton(log_objective, tensor([-1.0]))
    assert error.value.code == "invalid_start"
    for bad in (tensor([math.nan] * 5), torch.zeros((5, 1), dtype=torch.float64),
                torch.zeros(0, dtype=torch.float64)):
        with pytest.raises(KernelError) as error:
            maximize_newton(fn, bad)
        assert error.value.code == "invalid_start"
    for options in ({"max_iter": 0}, {"max_iter": 2.5}, {"gradient_tol": 0.0},
                    {"step_tol": math.inf}, {"scaled_gradient_tol": -1.0}):
        with pytest.raises(KernelError) as error:
            maximize_newton(fn, torch.zeros(5, dtype=torch.float64), **options)
        assert error.value.code == "invalid_solver_options"
    with pytest.raises(KernelError) as error:
        maximize_newton(lambda theta: (theta.sum(), theta, theta),
                        torch.zeros(2, dtype=torch.float64))
    assert error.value.code == "invalid_objective"


# --- BFGS -------------------------------------------------------------------


def first_order(fn):
    return lambda theta: fn(theta)[:2]


def test_bfgs_logit_matches_statsmodels_with_a_numerical_hessian(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    start = torch.zeros(5, dtype=torch.float64)
    result = maximize_bfgs(first_order(fn), start)
    reference = sm.Logit(y, x).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    assert result.converged and result.method == "bfgs_strong_wolfe"
    assert torch.equal(start, torch.zeros(5, dtype=torch.float64))
    assert_allclose(result.theta.numpy(), reference.params, rtol=1e-6, atol=1e-7)
    assert_allclose(covariance(result), reference.cov_params(), rtol=1e-6, atol=1e-12)
    # The reported Hessian is the true one at the solution, not the BFGS approximation.
    assert_allclose(result.hessian.numpy(), fn(result.theta)[2].numpy(), rtol=1e-9, atol=1e-9)
    assert result.diagnostics["hessian_source"] == "numerical_hessian"
    assert result.diagnostics["gradient_max"] <= 1e-8 * abs(result.value)
    assert result.diagnostics["function_evaluations"] >= result.iterations + 1


def test_bfgs_uses_the_supplied_hessian_function(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    result = maximize_bfgs(fn, torch.zeros(5, dtype=torch.float64),
                           hessian_fn=lambda theta: fn(theta)[2])
    assert result.converged and result.diagnostics["hessian_source"] == "hessian_fn"
    assert_allclose(result.hessian.numpy(), fn(result.theta)[2].numpy(), rtol=1e-14)


def test_bfgs_poisson_handles_overflowing_trials(poisson_data):
    x, y = poisson_data
    fn, _ = poisson_objective(x, y)
    result = maximize_bfgs(first_order(fn), torch.zeros(4, dtype=torch.float64))
    reference = sm.GLM(y, x, family=sm.families.Poisson()).fit(tol=1e-14, maxiter=500)
    assert result.converged
    assert_allclose(result.theta.numpy(), reference.params, rtol=1e-6, atol=1e-8)
    assert_allclose(covariance(result), reference.cov_params(), rtol=1e-5, atol=1e-14)
    # Late steps gain less than the rounding noise of this likelihood; the slope-based
    # acceptance keeps the search from grinding through bisections.
    assert result.diagnostics["function_evaluations"] < 60


def test_bfgs_normal_from_a_far_nonconcave_start(normal_data):
    y = normal_data
    result = maximize_bfgs(normal_objective(y), torch.zeros(2, dtype=torch.float64))
    assert result.converged
    assert_allclose(result.theta.numpy(), [y.mean(), math.log(y.std())], rtol=1e-7)
    assert_allclose(covariance(result), np.diag([y.var() / len(y), 1 / (2 * len(y))]),
                    rtol=1e-6, atol=1e-9)


def test_bfgs_negative_binomial_matches_newton_and_scipy(count_data, negative_binomial_reference):
    x, y = count_data
    fn = negative_binomial_objective(x, y)
    _, reference = negative_binomial_reference
    result = maximize_bfgs(fn, torch.zeros(4, dtype=torch.float64))
    newton = maximize_newton(fn, torch.zeros(4, dtype=torch.float64))
    assert result.converged
    assert_allclose(result.theta.numpy(), newton.theta.numpy(), rtol=1e-6, atol=1e-7)
    assert_allclose(result.hessian.numpy(), newton.hessian.numpy(), rtol=1e-6, atol=1e-5)
    assert_allclose(np.r_[result.theta[:-1].numpy(), math.exp(result.theta[-1])], reference.params,
                    rtol=1e-6)
    oracle = minimize(lambda t: -float(fn(tensor(t))[0]), np.zeros(4),
                      jac=lambda t: -fn(tensor(t))[1].numpy(), method="BFGS",
                      options={"gtol": 1e-7})
    assert_allclose(result.theta.numpy(), oracle.x, rtol=1e-5, atol=1e-6)
    assert result.value >= -oracle.fun - 1e-8


def test_bfgs_negated_rosenbrock():
    start = tensor([-1.2, 1.0])
    result = maximize_bfgs(rosenbrock_objective, start)
    assert result.converged and result.iterations < 60
    assert_allclose(result.theta.numpy(), [1.0, 1.0], rtol=1e-8)
    assert result.value == pytest.approx(0.0, abs=1e-15)
    assert_allclose(result.hessian.numpy(), [[-802.0, 400.0], [400.0, -200.0]], rtol=1e-8)
    oracle = minimize(lambda t: -float(rosenbrock_objective(tensor(t))[0]), start.numpy(),
                      jac=lambda t: -rosenbrock_objective(tensor(t))[1].numpy(), method="BFGS")
    assert_allclose(result.theta.numpy(), oracle.x, rtol=1e-5)
    # Comparable effort to SciPy's Wolfe-search BFGS on the classic test problem.
    assert result.diagnostics["function_evaluations"] <= 2 * oracle.nfev


@pytest.mark.parametrize("k, condition", [(10, 1e2), (30, 1e4), (60, 1e6)])
def test_bfgs_ill_conditioned_quadratic(k, condition):
    rng = np.random.default_rng(100 + k)
    basis, _ = np.linalg.qr(rng.normal(size=(k, k)))
    matrix = basis @ np.diag(np.logspace(0, np.log10(condition), k)) @ basis.T
    matrix = (matrix + matrix.T) / 2
    linear = rng.normal(size=k)
    a, b = tensor(matrix), tensor(linear)

    def fn(theta):
        return -0.5 * theta @ a @ theta + b @ theta, b - a @ theta

    result = maximize_bfgs(fn, torch.zeros(k, dtype=torch.float64))
    solution = np.linalg.solve(matrix, linear)
    assert result.converged and result.diagnostics["gradient_max"] <= 1e-8
    assert_allclose(result.theta.numpy(), solution, rtol=1e-6, atol=1e-8)
    assert_allclose(result.hessian.numpy(), -matrix, rtol=1e-8, atol=1e-8 * condition)
    assert result.iterations <= 4 * k


def test_bfgs_single_parameter_and_callback():
    def fn(theta):
        d = theta - 3
        return -(d**4).sum() - (d**2).sum(), -4 * d**3 - 2 * d

    seen = []
    result = maximize_bfgs(fn, tensor([10.0]), callback=lambda i, theta, value: seen.append(i))
    assert result.converged and result.theta.item() == pytest.approx(3.0, abs=1e-8)
    assert result.hessian.shape == (1, 1) and result.hessian.item() == pytest.approx(-2.0, rel=1e-8)
    assert seen == list(range(1, result.iterations + 1))


def test_bfgs_initial_inverse_hessian_rescues_an_ill_scaled_problem():
    rng = np.random.default_rng(5)
    z = np.column_stack([np.ones(4000), rng.normal(size=(4000, 3))])
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z @ [0.3, 1.0, -1.0, 0.5]))).astype(float)
    scales = np.array([1.0, 1e-8, 1.0, 1e9])
    fn = logit_objective(z * scales, y)
    start = torch.zeros(4, dtype=torch.float64)
    plain = maximize_bfgs(first_order(fn), start, raise_on_failure=False)
    reference = sm.Logit(y, z).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    # Steepest ascent cannot cross 1e17 units: the BFGS iteration stalls and only the
    # Newton polishing steps on the true Hessian reach the maximum.
    assert plain.converged and plain.diagnostics["polish_iterations"] > 0
    assert "polishing" in plain.diagnostics["message"]
    assert_allclose(plain.theta.numpy() * scales, reference.params, rtol=1e-6, atol=1e-7)
    result = maximize_bfgs(first_order(fn), start,
                           initial_inverse_hessian=information_inverse(-fn(start)[2]))
    assert result.converged and result.iterations < plain.iterations + 40
    assert result.diagnostics["polish_iterations"] == 0
    assert_allclose(result.theta.numpy() * scales, reference.params, rtol=1e-6, atol=1e-7)
    assert_allclose(covariance(result) * np.outer(scales, scales), reference.cov_params(),
                    rtol=1e-6, atol=1e-12)


def test_bfgs_failures(logit_data):
    x, y = logit_data
    fn = first_order(logit_objective(x, y))
    with pytest.raises(KernelError) as error:
        maximize_bfgs(fn, torch.zeros(5, dtype=torch.float64), max_iter=1)
    assert error.value.code == "nonconvergence"
    result = maximize_bfgs(fn, torch.zeros(5, dtype=torch.float64), max_iter=1,
                           raise_on_failure=False)
    assert not result.converged and result.iterations == 1
    assert result.hessian.shape == (5, 5) and bool(torch.isfinite(result.hessian).all())

    def double_well(theta):
        return -((theta * theta - 1) ** 2).sum(), -4 * theta * (theta * theta - 1)

    # The gradient vanishes at 0 but the Hessian there is +4: a minimum, not a maximum.
    with pytest.raises(KernelError) as error:
        maximize_bfgs(double_well, tensor([0.0]))
    assert error.value.code == "nonconvergence" and "not negative definite" in str(error.value)
    assert maximize_bfgs(double_well, tensor([0.1])).theta.item() == pytest.approx(1.0, abs=1e-8)

    with pytest.raises(KernelError) as error:
        maximize_bfgs(lambda theta: (torch.log(theta[0]), 1 / theta), tensor([-1.0]))
    assert error.value.code == "invalid_start"
    with pytest.raises(KernelError) as error:
        maximize_bfgs(fn, torch.zeros(5, dtype=torch.float64), step_tol=0.0)
    assert error.value.code == "invalid_solver_options"
    with pytest.raises(KernelError) as error:
        maximize_bfgs(fn, torch.zeros(5, dtype=torch.float64),
                      initial_inverse_hessian=torch.eye(3, dtype=torch.float64))
    assert error.value.code == "invalid_solver_options"


# --- numerical derivatives --------------------------------------------------


def test_numerical_gradient_and_hessian_match_analytic_derivatives(count_data):
    x, y = count_data
    fn = negative_binomial_objective(x, y)
    theta = tensor([0.5, 0.2, 0.1, 0.3])
    value, gradient, hessian = fn(theta)
    numeric_gradient = numerical_gradient(lambda t: fn(t)[0], theta)
    numeric_hessian = numerical_hessian(lambda t: fn(t)[1], theta)
    assert numeric_gradient.dtype == torch.float64 and numeric_gradient.shape == (4,)
    assert_allclose(numeric_gradient.numpy(), gradient.numpy(), rtol=1e-9,
                    atol=1e-11 * abs(float(value)))
    assert_allclose(numeric_hessian.numpy(), hessian.numpy(), rtol=1e-9,
                    atol=1e-9 * float(hessian.abs().max()))
    assert torch.equal(numeric_hessian, numeric_hessian.T)
    # Plain central differences (levels=1) are cheaper and visibly less accurate.
    plain = numerical_hessian(lambda t: fn(t)[1], theta, levels=1)
    assert_allclose(plain.numpy(), hessian.numpy(), rtol=1e-6,
                    atol=1e-6 * float(hessian.abs().max()))
    plain_gradient = numerical_gradient(lambda t: fn(t)[0], theta, levels=1)
    assert_allclose(plain_gradient.numpy(), gradient.numpy(), rtol=1e-6, atol=1e-6)


def test_numerical_derivative_evaluation_counts(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    theta = tensor([0.1, 0.5, -0.4, 0.2, 0.3])
    calls = []
    numerical_gradient(lambda t: calls.append(1) or fn(t)[0], theta, levels=1)
    assert len(calls) == 2 * 5
    calls.clear()
    numerical_hessian(lambda t: calls.append(1) or fn(t)[1], theta, levels=1)
    assert len(calls) == 2 * 5
    calls.clear()
    numerical_hessian(lambda t: calls.append(1) or fn(t)[1], theta)
    assert len(calls) <= 8 * 5                        # three or four step sizes per coordinate
    calls.clear()
    numerical_gradient(lambda t: calls.append(1) or fn(t)[0], theta, relative_step=1e-3)
    assert len(calls) <= 8 * 5 + 1


def test_numerical_derivatives_at_the_optimum_and_under_bad_scaling():
    rng = np.random.default_rng(5)
    z = np.column_stack([np.ones(4000), rng.normal(size=(4000, 3))])
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z @ [0.3, 1.0, -1.0, 0.5]))).astype(float)
    for scales in ([1.0, 1.0, 1.0, 1.0], [1.0, 1e-4, 1.0, 1e5], [1.0, 1e-8, 1.0, 1e9]):
        scales = tensor(scales)
        fn = logit_objective(z * scales.numpy(), y)
        for theta in (tensor([0.1, 0.5, -0.4, 0.2]) / scales,
                      maximize_newton(fn, torch.zeros(4, dtype=torch.float64)).theta):
            value, gradient, hessian = fn(theta)
            unit = scales[:, None] * scales                 # compare in standardized units
            numeric_hessian = numerical_hessian(lambda t: fn(t)[1], theta)
            assert_allclose((numeric_hessian / unit).numpy(), (hessian / unit).numpy(),
                            rtol=1e-8, atol=1e-9 * float((hessian / unit).abs().max()))
            numeric_gradient = numerical_gradient(lambda t: fn(t)[0], theta)
            assert_allclose((numeric_gradient / scales).numpy(), (gradient / scales).numpy(),
                            rtol=1e-8, atol=1e-10 * abs(float(value)))


def test_numerical_derivatives_shrink_steps_that_leave_the_domain():
    weights = tensor([1e-6, 2.0])
    theta = tensor([1e-6, 2.0])                       # the default step would make theta[0] < 0

    def value_fn(point):
        return (weights * torch.log(point) - point).sum()

    def gradient_fn(point):
        return weights / point - 1

    assert_allclose(numerical_gradient(value_fn, theta).numpy(), [0.0, 0.0], atol=1e-8)
    assert_allclose(numerical_hessian(gradient_fn, theta).numpy(),
                    np.diag((-weights / theta**2).numpy()), rtol=1e-8, atol=1e-12)
    with pytest.raises(KernelError) as error:
        numerical_gradient(lambda point: torch.log(-point.abs()).sum(), theta)
    assert error.value.code == "numerical_failure"
    with pytest.raises(KernelError) as error:
        numerical_hessian(lambda point: point.sum(), theta)       # not a gradient
    assert error.value.code == "invalid_objective"
    with pytest.raises(KernelError) as error:
        numerical_gradient(value_fn, theta, relative_step=0.0)
    assert error.value.code == "invalid_solver_options"


def test_numerical_derivatives_polynomial_exactness():
    matrix = tensor([[4.0, 1.0, -2.0], [1.0, 3.0, 0.5], [-2.0, 0.5, 5.0]])
    theta = tensor([0.3, -1.7, 250.0])

    def value_fn(point):
        return -0.5 * point @ matrix @ point + point.sum()

    assert_allclose(numerical_gradient(value_fn, theta).numpy(), (1 - matrix @ theta).numpy(),
                    rtol=1e-9, atol=1e-12 * abs(float(value_fn(theta))))
    assert_allclose(numerical_hessian(lambda point: 1 - matrix @ point, theta).numpy(),
                    -matrix.numpy(), rtol=1e-9, atol=1e-9)


def test_check_derivatives_accepts_correct_and_flags_wrong_derivatives(count_data):
    x, y = count_data
    fn = negative_binomial_objective(x, y)
    theta = tensor([0.5, 0.2, 0.1, 0.3])
    report = check_derivatives(fn, theta)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    assert report["hessian_asymmetry"] < 1e-9 * float(fn(theta)[2].abs().max())
    assert report["gradient_max_abs_error"] >= 0 and report["hessian_max_abs_error"] >= 0

    def wrong_gradient(point):
        value, gradient, hessian = fn(point)
        gradient = gradient.clone()
        gradient[-1] *= 1.01                              # a 1% error in the log-alpha score
        return value, gradient, hessian

    def wrong_hessian(point):
        value, gradient, hessian = fn(point)
        hessian = hessian.clone()
        hessian[0, -1] = hessian[-1, 0] = -hessian[0, -1]   # sign error in a cross term
        return value, gradient, hessian

    assert check_derivatives(wrong_gradient, theta)["gradient_max_rel_error"] > 1e-4
    flagged = check_derivatives(wrong_hessian, theta)
    assert flagged["gradient_max_rel_error"] < 1e-8 and flagged["hessian_max_rel_error"] > 1e-4
    gradient_only = check_derivatives(lambda point: fn(point)[:2], theta)
    assert gradient_only["gradient_max_rel_error"] < 1e-8
    assert gradient_only["hessian_max_rel_error"] is None


# --- covariance helpers -----------------------------------------------------


def test_information_inverse_matches_numpy_and_is_scale_robust():
    rng = np.random.default_rng(9)
    design = rng.normal(size=(200, 6)) * np.array([1e-6, 1.0, 1e5, 3.0, 1e-3, 1e8])
    information = design.T @ design
    inverse = information_inverse(tensor(information))
    scale = np.sqrt(np.diag(information))
    expected = np.linalg.inv(information / np.outer(scale, scale)) / np.outer(scale, scale)
    assert_allclose(inverse.numpy(), expected, rtol=1e-10)
    assert torch.equal(inverse, inverse.T) and inverse.dtype == torch.float64
    assert_allclose(information_inverse(tensor([[4.0]])).numpy(), [[0.25]])


@pytest.mark.parametrize("matrix", [
    [[1.0, 2.0], [2.0, 1.0]],                    # indefinite
    [[1.0, 1.0], [1.0, 1.0]],                    # singular
    [[-1.0, 0.0], [0.0, -2.0]],                  # a Hessian passed without the sign flip
    [[0.0, 0.0], [0.0, 1.0]],
    [[math.nan, 0.0], [0.0, 1.0]],
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],          # not square
])
def test_information_inverse_rejects_matrices_that_are_not_positive_definite(matrix):
    with pytest.raises(KernelError) as error:
        information_inverse(tensor(matrix))
    assert error.value.code == "singular_information"


def test_information_inverse_rejects_numerically_collinear_information():
    rng = np.random.default_rng(4)
    design = rng.normal(size=(100, 3))
    design[:, 2] = design[:, 0] + design[:, 1] * (1 + 1e-15)
    with pytest.raises(KernelError):
        information_inverse(tensor(design.T @ design))


def test_opg_and_sandwich_against_statsmodels(logit_data):
    x, y = logit_data
    result = maximize_newton(logit_objective(x, y), torch.zeros(5, dtype=torch.float64))
    scores = tensor(x) * (tensor(y) - torch.sigmoid(tensor(x) @ result.theta))[:, None]
    meat = opg(scores)
    assert_allclose(meat.numpy(), scores.numpy().T @ scores.numpy(), rtol=1e-13)
    assert_allclose(scores.sum(dim=0).numpy(), result.gradient.numpy(), atol=1e-9)
    bread = information_inverse(-result.hessian)
    robust = sm.Logit(y, x).fit(method="newton", tol=1e-14, disp=0, cov_type="HC0")
    assert_allclose((bread @ meat @ bread).numpy(), robust.cov_params(), rtol=1e-8, atol=1e-14)
    with pytest.raises(KernelError):
        opg(torch.zeros(3, dtype=torch.float64))


# --- module contract --------------------------------------------------------


def test_module_imports_only_torch_and_the_standard_library():
    tree = ast.parse(Path(optimize.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "dataclasses", "math", "typing", "torch"}
    source = Path(optimize.__file__).read_text()
    assert "requires_grad" not in source and "autograd" not in source.replace("no autograd", "")


def test_entry_points_work_inside_inference_mode_and_upcast_starts(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    reference = maximize_newton(fn, torch.zeros(5, dtype=torch.float64))
    with torch.inference_mode():
        newton = maximize_newton(fn, torch.zeros(5, dtype=torch.float32))
        bfgs = maximize_bfgs(fn, [0, 0, 0, 0, 0])
        report = check_derivatives(fn, newton.theta + 0.1)
        inverse = information_inverse(-newton.hessian)
    assert newton.theta.dtype == torch.float64 and torch.equal(newton.theta, reference.theta)
    assert bfgs.converged and bfgs.theta.dtype == torch.float64
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    assert_allclose(inverse.numpy(), covariance(reference), rtol=1e-12)


def test_optimizers_do_not_build_autograd_graphs(logit_data):
    x, y = logit_data
    fn = logit_objective(x, y)
    start = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    for result in (maximize_newton(fn, start), maximize_bfgs(fn, start)):
        assert result.converged
        for item in (result.theta, result.gradient, result.hessian):
            assert item.dtype == torch.float64 and not item.requires_grad
        assert isinstance(result.value, float) and isinstance(result.iterations, int)
