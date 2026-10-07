"""Adversarial stress tests for openecon.engines.optimize.

Independent of the implementer's tests: closed forms, brute-force loops and NumPy /
SciPy / statsmodels are the oracles. Every converged BFGS result must satisfy the
scaled-gradient rule with the true Hessian, exactly as a Newton result does.
"""

import ast
import math
import time
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
    check_derivatives,
    information_inverse,
    maximize_bfgs,
    maximize_newton,
    numerical_gradient,
    numerical_hessian,
    opg,
)

F64 = torch.float64


def tensor(values):
    return torch.as_tensor(np.asarray(values, dtype=float), dtype=F64)


def logit_objective(x, y):
    x, y = tensor(x), tensor(y)

    def fn(theta):
        eta = x @ theta
        value = (y * torch.nn.functional.logsigmoid(eta)
                 + (1 - y) * torch.nn.functional.logsigmoid(-eta)).sum()
        p = torch.sigmoid(eta)
        return value, x.T @ (y - p), -(x.T @ (x * (p * (1 - p))[:, None]))

    return fn


def normal_objective(y):
    """theta = (mu, log sigma); closed-form maximum (mean, log of the ML sigma)."""
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


def quadratic_objective(matrix, linear):
    """-0.5 theta' A theta + b' theta with maximum A^-1 b."""
    a, b = tensor(matrix), tensor(linear)

    def fn(theta):
        return -0.5 * theta @ a @ theta + b @ theta, b - a @ theta, -a

    return fn


def first_order(fn):
    return lambda theta: fn(theta)[:2]


def numpy_logit_newton(x, y, iterations=30):
    """Independent IRLS oracle in NumPy."""
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ beta))
        gradient = x.T @ (y - p)
        information = x.T @ (x * (p * (1 - p))[:, None])
        beta = beta + np.linalg.solve(information, gradient)
    return beta, np.linalg.inv(information)


def scaled_gradient(result):
    """g'(-H)^-1 g recomputed independently with NumPy."""
    g, h = result.gradient.numpy(), result.hessian.numpy()
    return float(g @ np.linalg.solve(-h, g))


def assert_true_maximum(result, tol=1e-10):
    assert result.converged and result.diagnostics["concave"] is True
    assert np.all(np.linalg.eigvalsh(-result.hessian.numpy()) > 0)
    assert scaled_gradient(result) <= tol
    assert result.diagnostics["scaled_gradient"] <= tol


# --- Newton-Raphson: magnitudes and degenerate problems -----------------------


@pytest.mark.parametrize("sigma, n", [(1e-6, 2000), (1.0, 7), (1e6, 100000)])
def test_newton_normal_closed_form_across_variance_magnitudes(sigma, n):
    y = np.random.default_rng(3).normal(50.0, sigma, size=n)
    fn = normal_objective(y)
    result = maximize_newton(fn, tensor([0.0, 0.0]))
    assert_true_maximum(result)
    sigma_ml = math.sqrt(((y - y.mean()) ** 2).mean())
    # step_tol = 1e-10 is relative to max(1, |theta_j|): that is the precision contract.
    assert_allclose(result.theta.numpy(), [y.mean(), math.log(sigma_ml)], rtol=1e-10, atol=0)
    expected = -0.5 * n * (math.log(2 * math.pi) + 1) - n * math.log(sigma_ml)
    assert result.value == pytest.approx(expected, rel=1e-13)
    cov = information_inverse(-result.hessian).numpy()
    assert_allclose(cov, np.diag([sigma_ml**2 / n, 1 / (2 * n)]), rtol=1e-8,
                    atol=1e-12 * cov.diagonal().max())


def test_newton_objective_of_magnitude_1e12_converges_to_the_closed_form():
    rng = np.random.default_rng(8)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    matrix = basis @ np.diag([1e2, 1e3, 1e4, 1e5, 1e6, 1e7]) @ basis.T
    matrix = (matrix + matrix.T) / 2
    linear = rng.normal(size=6) * 1e6
    fn = quadratic_objective(matrix, linear)
    start = tensor(rng.normal(size=6) * 1e6)
    result = maximize_newton(fn, start)
    solution = np.linalg.solve(matrix, linear)
    assert abs(result.diagnostics["value_history"][0]) > 1e15
    assert abs(result.value) > 1e9 and result.iterations <= 3
    assert_true_maximum(result)
    assert_allclose(result.theta.numpy(), solution, rtol=1e-12, atol=0)


@pytest.mark.parametrize("magnitude", [1e2, 1e4, 1e6, 1e8])
def test_newton_saturated_logit_starts_where_the_hessian_vanishes(magnitude):
    """eta ~ 1e8 makes p(1-p) exactly 0 (or 1e-300): the Hessian carries no information."""
    x, y = _logit_data(600, 3, seed=21)
    fn = logit_objective(x, y)
    start = tensor([magnitude, -magnitude, magnitude])
    _, _, hessian = fn(start)
    if magnitude >= 1e4:
        assert float(hessian.abs().max()) < 1e-90
    result = maximize_newton(fn, start)
    beta, _ = numpy_logit_newton(x, y)
    assert_true_maximum(result)
    assert result.iterations <= 40
    assert_allclose(result.theta.numpy(), beta, rtol=1e-9, atol=1e-12)
    if magnitude >= 1e4:
        assert result.diagnostics["nonconcave_iterations"] > 0


def test_newton_intercept_only_logit_with_three_observations():
    fn = logit_objective(np.ones((3, 1)), [1.0, 0.0, 0.0])
    result = maximize_newton(fn, tensor([2.5]))
    assert_true_maximum(result)
    assert result.theta.item() == pytest.approx(math.log(0.5), abs=1e-14)
    assert result.value == pytest.approx(math.log(1 / 3) + 2 * math.log(2 / 3), rel=1e-14)
    assert information_inverse(-result.hessian).item() == pytest.approx(1 / (3 * 2 / 9), rel=1e-12)


def test_newton_normal_with_n_barely_above_k():
    y = [1.0, 2.0, 4.0]
    result = maximize_newton(normal_objective(y), tensor([0.0, 0.0]))
    assert_true_maximum(result)
    sigma_ml = math.sqrt(np.var(y))
    assert_allclose(result.theta.numpy(), [np.mean(y), math.log(sigma_ml)], rtol=1e-10)


def test_newton_unidentified_parameter_is_reported_quickly_as_nonconvergence():
    x, y = _logit_data(500, 3, seed=4)
    fn = logit_objective(x, y)
    identified = maximize_newton(fn, torch.zeros(3, dtype=F64))

    def padded(theta):
        value, gradient, hessian = fn(theta[:3])
        gradient = torch.cat([gradient, torch.zeros(1, dtype=F64)])
        full = torch.zeros(4, 4, dtype=F64)
        full[:3, :3] = hessian
        return value, gradient, full

    with pytest.raises(KernelError) as error:
        maximize_newton(padded, torch.zeros(4, dtype=F64))
    assert error.value.code == "nonconvergence" and "unidentified" in str(error.value)
    result = maximize_newton(padded, torch.zeros(4, dtype=F64), raise_on_failure=False)
    assert not result.converged and result.iterations < 30
    assert result.diagnostics["concave"] is False
    assert result.diagnostics["scaled_gradient"] is None
    # The identified coefficients still reached their maximum; the spare one never moved.
    assert_allclose(result.theta[:3].numpy(), identified.theta.numpy(), rtol=1e-8, atol=1e-10)
    assert result.theta[3].item() == 0.0
    assert "nan" not in result.diagnostics["message"]


def test_newton_duplicated_regressor_is_nonconvergence_not_a_false_maximum():
    x, y = _logit_data(400, 2, seed=5)
    duplicated = np.column_stack([x, x[:, 1]])
    result = maximize_newton(logit_objective(duplicated, y), torch.zeros(3, dtype=F64),
                             raise_on_failure=False)
    assert not result.converged and result.diagnostics["concave"] is False
    assert result.iterations < 40
    with pytest.raises(KernelError):
        information_inverse(-result.hessian)


def test_newton_zero_variance_outcome_runs_the_variance_to_the_boundary():
    fn = normal_objective(np.full(50, 3.0))              # sigma -> 0, lnsigma -> -inf
    result = maximize_newton(fn, tensor([0.0, 0.0]), max_iter=40, raise_on_failure=False)
    assert not result.converged
    assert result.theta[1].item() < -5.0
    assert result.diagnostics["message"]


@pytest.mark.parametrize("kind", ["complete", "quasi", "poisson_all_zero"])
def test_newton_monotone_likelihoods_never_converge(kind):
    rng = np.random.default_rng(12)
    if kind == "poisson_all_zero":
        x = np.column_stack([np.ones(100), rng.normal(size=100)])
        xt = tensor(x)
        zero = torch.zeros(100, dtype=F64)

        def fn(theta):
            mu = torch.exp(xt @ theta)
            return -mu.sum(), xt.T @ (zero - mu), -(xt.T @ (xt * mu[:, None]))
        k = 2
    else:
        z = rng.normal(size=300)
        if kind == "quasi":
            # Quasi-complete separation: 30 observations sit exactly at the boundary x = 0
            # with mixed outcomes, so the intercept is identified but the slope diverges.
            z[:30] = 0.0
        x = np.column_stack([np.ones(300), z])
        y = (z > 0).astype(float)
        if kind == "quasi":
            y[:30] = rng.integers(0, 2, size=30)
        fn = logit_objective(x, y)
        k = 2
    result = maximize_newton(fn, torch.zeros(k, dtype=F64), max_iter=80, raise_on_failure=False)
    assert not result.converged
    assert np.all(np.diff(result.diagnostics["value_history"]) >= -1e-9)


def test_newton_poisson_with_large_counts_and_objective_noise(seed=9):
    rng = np.random.default_rng(seed)
    n = 50000
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = rng.poisson(np.exp(x @ np.array([6.0, 0.4, -0.3]))).astype(float)
    xt, yt = tensor(x), tensor(y)
    constant = float(torch.lgamma(yt + 1).sum())

    def fn(theta):
        eta = xt @ theta
        mu = torch.exp(eta)
        return (yt * eta - mu).sum() - constant, xt.T @ (yt - mu), -(xt.T @ (xt * mu[:, None]))

    result = maximize_newton(fn, tensor([math.log(y.mean()), 0.0, 0.0]))
    assert abs(result.value) > 1e5
    assert_true_maximum(result)
    reference = sm.GLM(y, x, family=sm.families.Poisson()).fit(tol=1e-14, maxiter=200)
    assert_allclose(result.theta.numpy(), reference.params, rtol=1e-10, atol=1e-12)


# --- Newton-Raphson: API, shapes, dtypes, immutability ------------------------


def test_newton_rejects_malformed_objectives_with_kernel_errors():
    fn = normal_objective(np.arange(10.0))
    start = tensor([1.0, 0.5])
    malformed = {
        "two_outputs": lambda t: fn(t)[:2],
        "not_a_tuple": lambda t: fn(t)[0],
        "gradient_column": lambda t: (fn(t)[0], fn(t)[1][:, None], fn(t)[2]),
        "gradient_too_long": lambda t: (fn(t)[0], torch.cat([fn(t)[1], fn(t)[1]]), fn(t)[2]),
        "hessian_vector": lambda t: (fn(t)[0], fn(t)[1], fn(t)[1]),
        "hessian_batched": lambda t: (fn(t)[0], fn(t)[1], fn(t)[2][None]),
        "value_vector": lambda t: (fn(t)[1], fn(t)[1], fn(t)[2]),
        "value_string": lambda t: ("x", fn(t)[1], fn(t)[2]),
    }
    for name, bad in malformed.items():
        with pytest.raises(KernelError) as error:
            maximize_newton(bad, start)
        assert error.value.code == "invalid_objective", name
    with pytest.raises(KernelError) as error:
        maximize_bfgs(lambda t: fn(t)[0], start)
    assert error.value.code == "invalid_objective"
    with pytest.raises(KernelError) as error:
        maximize_bfgs(lambda t: (fn(t)[0], fn(t)[1][None]), start)
    assert error.value.code == "invalid_objective"
    with pytest.raises(KernelError) as error:
        maximize_bfgs(fn, start, hessian_fn=lambda t: fn(t)[1])
    assert error.value.code == "invalid_objective"
    for bad_start in ("abc", [[0.0, 0.0]], [], [0.0, math.inf], tensor([1.0, 2.0]) * 1j):
        with pytest.raises(KernelError) as error:
            maximize_newton(fn, bad_start)
        assert error.value.code == "invalid_start"


def test_newton_and_bfgs_accept_four_outputs_and_extra_outputs_are_ignored():
    fn = normal_objective(np.random.default_rng(1).normal(size=40))
    four = lambda t: (*fn(t), "extra")  # noqa: E731
    assert maximize_newton(four, tensor([0.0, 0.0])).converged
    assert maximize_bfgs(four, tensor([0.0, 0.0])).converged


def test_python_list_starts_are_exact_float64():
    def fn(theta):
        d = theta - 0.1
        return -(d * d).sum(), -2 * d, torch.full((1, 1), -2.0, dtype=F64)

    result = maximize_newton(fn, [0.1])                   # 0.1 is not a float32 number
    assert result.iterations == 0 and result.theta.item() == 0.1
    bfgs = maximize_bfgs(first_order(fn), [0.1])
    assert bfgs.iterations == 0 and bfgs.theta.item() == 0.1
    assert_allclose(numerical_gradient(lambda t: (t * 3).sum(), [0.1, 0.7]).numpy(), [3.0, 3.0],
                    rtol=1e-10)
    assert torch.equal(maximize_newton(fn, np.array([0.1])).theta, tensor([0.1]))
    assert maximize_newton(fn, [0]).theta.dtype == F64            # integer list


def test_float32_derivatives_are_upcast_and_results_are_float64():
    """float32 derivatives are coerced; the objective VALUE itself must stay float64,
    since numerical differentiation assumes float64 rounding in the function."""
    x, y = _logit_data(300, 3, seed=6)
    fn = logit_objective(x, y)
    exact = maximize_newton(fn, torch.zeros(3, dtype=F64))

    def single(theta):
        value, gradient, hessian = fn(theta)
        return value, gradient.to(torch.float32), hessian.to(torch.float32)

    result = maximize_newton(single, torch.zeros(3, dtype=F64))
    for item in (result.theta, result.gradient, result.hessian):
        assert item.dtype == F64
    assert_allclose(result.theta.numpy(), exact.theta.numpy(), rtol=1e-5, atol=1e-6)
    bfgs = maximize_bfgs(first_order(single), torch.zeros(3, dtype=F64),
                         hessian_fn=lambda t: single(t)[2])
    assert bfgs.hessian.dtype == F64 and bfgs.gradient.dtype == F64
    report = check_derivatives(single, tensor([0.1, 0.2, 0.3]))
    assert report["gradient_max_rel_error"] < 1e-5


def test_numpy_returning_objectives_are_coerced():
    y = np.random.default_rng(2).normal(size=25)
    fn = normal_objective(y)

    def as_numpy(theta):
        value, gradient, hessian = fn(theta)
        return float(value), gradient.numpy(), hessian.numpy()

    result = maximize_newton(as_numpy, tensor([0.0, 0.0]))
    assert_true_maximum(result)
    assert result.theta[0].item() == pytest.approx(y.mean(), abs=1e-10)


def test_objectives_that_reuse_output_buffers_are_safe():
    x, y = _logit_data(500, 4, seed=7)
    fn = logit_objective(x, y)
    fresh_newton = maximize_newton(fn, torch.zeros(4, dtype=F64))
    fresh_bfgs = maximize_bfgs(first_order(fn), torch.zeros(4, dtype=F64))
    buffers = (torch.zeros((), dtype=F64), torch.zeros(4, dtype=F64), torch.zeros(4, 4, dtype=F64))

    def reusing(theta):
        value, gradient, hessian = fn(theta)
        buffers[0].copy_(value)
        buffers[1].copy_(gradient)
        buffers[2].copy_(hessian)
        return buffers

    result = maximize_newton(reusing, torch.zeros(4, dtype=F64))
    assert torch.equal(result.theta, fresh_newton.theta)
    assert torch.equal(result.gradient, fresh_newton.gradient)
    bfgs = maximize_bfgs(lambda t: reusing(t)[:2], torch.zeros(4, dtype=F64))
    assert bfgs.converged and bfgs.iterations == fresh_bfgs.iterations
    assert_allclose(bfgs.theta.numpy(), fresh_bfgs.theta.numpy(), rtol=0, atol=1e-12)
    numeric = numerical_hessian(lambda t: reusing(t)[1], fresh_newton.theta)
    assert_allclose(numeric.numpy(), fresh_newton.hessian.numpy(), rtol=1e-8, atol=1e-9)
    assert buffers[1] is not result.gradient and buffers[1] is not bfgs.gradient


def test_inputs_are_not_modified_and_runs_are_deterministic():
    x, y = _logit_data(800, 5, seed=10)
    fn = logit_objective(x, y)
    start = tensor([0.3, -0.2, 0.1, 0.0, 0.5])
    start_copy = start.clone()
    returned = []

    def recording(theta):
        output = fn(theta)
        returned.append(tuple(item.clone() for item in output))
        return output

    first = maximize_newton(recording, start)
    second = maximize_newton(fn, start)
    assert torch.equal(start, start_copy)
    assert torch.equal(first.theta, second.theta) and first.value == second.value
    assert torch.equal(first.hessian, second.hessian)
    assert first.diagnostics["value_history"] == second.diagnostics["value_history"]
    for stored, (value, gradient, hessian) in zip(returned, returned):
        assert torch.equal(stored[1], gradient) and torch.equal(stored[2], hessian)
    bfgs_a = maximize_bfgs(first_order(fn), start)
    bfgs_b = maximize_bfgs(first_order(fn), start)
    assert torch.equal(bfgs_a.theta, bfgs_b.theta) and torch.equal(bfgs_a.hessian, bfgs_b.hessian)
    assert torch.equal(start, start_copy)
    information = -first.hessian
    information_copy = information.clone()
    information_inverse(information)
    assert torch.equal(information, information_copy)
    theta_copy = first.theta.clone()
    numerical_gradient(lambda t: fn(t)[0], first.theta)
    numerical_hessian(lambda t: fn(t)[1], first.theta)
    check_derivatives(fn, first.theta)
    assert torch.equal(first.theta, theta_copy)


def test_callback_exceptions_propagate_and_can_abort_the_iteration():
    fn = normal_objective(np.random.default_rng(1).normal(size=40))

    class Stop(Exception):
        pass

    def abort(iteration, theta, value):
        raise Stop

    with pytest.raises(Stop):
        maximize_newton(fn, tensor([0.0, 0.0]), callback=abort)
    with pytest.raises(Stop):
        maximize_bfgs(fn, tensor([0.0, 0.0]), callback=abort)


def test_newton_large_n_is_fast_with_one_evaluation_per_iteration():
    rng = np.random.default_rng(99)
    n, k = 1_000_000, 10
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x @ (0.3 * rng.normal(size=k))))).astype(float)
    fn = logit_objective(x, y)
    started = time.perf_counter()
    result = maximize_newton(fn, torch.zeros(k, dtype=F64))
    elapsed = time.perf_counter() - started
    assert result.converged and elapsed < 5.0
    assert result.diagnostics["function_evaluations"] == result.iterations + 1
    assert_true_maximum(result)
    beta, cov = numpy_logit_newton(x, y, iterations=8)
    assert_allclose(result.theta.numpy(), beta, rtol=1e-9, atol=1e-12)
    assert_allclose(information_inverse(-result.hessian).numpy(), cov, rtol=1e-8)
    started = time.perf_counter()
    bfgs = maximize_bfgs(first_order(fn), torch.zeros(k, dtype=F64),
                         hessian_fn=lambda t: fn(t)[2])
    assert time.perf_counter() - started < 5.0
    assert_true_maximum(bfgs)
    assert_allclose(bfgs.theta.numpy(), beta, rtol=1e-6, atol=1e-8)


# --- BFGS: unit-dependence of the gradient rule ------------------------------


@pytest.mark.parametrize("sigma, n", [(1e4, 100000), (1e5, 100000), (1e6, 1000000)])
@pytest.mark.parametrize("with_hessian", [True, False])
def test_bfgs_wide_normal_reaches_the_true_maximum(sigma, n, with_hessian):
    """The gradient rule max|g| <= 1e-8 |f| alone stops 0.1 SE short of the mean here."""
    y = np.random.default_rng(1).normal(50.0, sigma, size=n)
    fn = normal_objective(y)
    start = tensor([0.0, math.log(sigma)])
    hessian_fn = (lambda t: fn(t)[2]) if with_hessian else None
    result = maximize_bfgs(first_order(fn), start, hessian_fn=hessian_fn)
    assert_true_maximum(result)
    se = sigma / math.sqrt(n)
    assert abs(result.theta[0].item() - y.mean()) / se < 1e-5
    sigma_ml = math.sqrt(((y - y.mean()) ** 2).mean())
    assert_allclose(result.theta.numpy(), [y.mean(), math.log(sigma_ml)], rtol=1e-9, atol=1e-9)
    assert result.diagnostics["function_evaluations"] < 80


def test_bfgs_polishing_is_counted_and_reported():
    y = np.random.default_rng(1).normal(50.0, 1e6, size=1000000)
    fn = normal_objective(y)
    seen = []
    result = maximize_bfgs(first_order(fn), tensor([0.0, math.log(1e6)]),
                           hessian_fn=lambda t: fn(t)[2],
                           callback=lambda i, theta, value: seen.append(i))
    assert result.converged
    polish = result.diagnostics["polish_iterations"]
    assert seen == list(range(1, result.iterations + 1))
    assert len(result.diagnostics["value_history"]) == result.iterations + 1
    assert np.all(np.diff(result.diagnostics["value_history"]) >= -1e-9 * abs(result.value))
    if polish:
        assert "polishing" in result.diagnostics["message"]
    else:
        assert result.diagnostics["message"].startswith("converged")


def test_bfgs_converged_results_always_satisfy_the_scaled_gradient_rule():
    rng = np.random.default_rng(31)
    problems = []
    for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        x, y = _logit_data(2000, 3, seed=int(scale * 7 + 11) % 1000)
        problems.append(("logit", logit_objective(x * [1.0, scale, 1.0], y), 3))
    for sigma in (1e-4, 1e2, 1e5):
        problems.append(("normal", normal_objective(rng.normal(1.0, sigma, size=5000)), 2))
    basis, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    matrix = basis @ np.diag(np.logspace(-4, 4, 8)) @ basis.T
    problems.append(("quadratic", quadratic_objective((matrix + matrix.T) / 2, rng.normal(size=8)),
                     8))
    for name, fn, k in problems:
        result = maximize_bfgs(first_order(fn), torch.zeros(k, dtype=F64),
                               hessian_fn=lambda t, fn=fn: fn(t)[2])
        assert_true_maximum(result)
        newton = maximize_newton(fn, torch.zeros(k, dtype=F64))
        se = np.sqrt(np.diag(information_inverse(-newton.hessian).numpy()))
        assert np.max(np.abs(result.theta.numpy() - newton.theta.numpy()) / se) < 1e-4, name


def test_bfgs_from_the_maximum_and_from_a_saddle():
    fn = normal_objective(np.random.default_rng(5).normal(size=200))
    optimum = maximize_newton(fn, tensor([0.0, 0.0])).theta
    result = maximize_bfgs(first_order(fn), optimum)
    assert result.converged and result.iterations == 0
    assert result.diagnostics["function_evaluations"] == 1

    def saddle(theta):
        return theta[0] ** 2 - theta[1] ** 2, torch.stack([2 * theta[0], -2 * theta[1]])

    with pytest.raises(KernelError) as error:
        maximize_bfgs(saddle, torch.zeros(2, dtype=F64))
    assert error.value.code == "nonconvergence"
    result = maximize_bfgs(saddle, torch.zeros(2, dtype=F64), raise_on_failure=False)
    assert not result.converged and result.diagnostics["concave"] is False
    assert result.diagnostics["scaled_gradient"] is None


def test_bfgs_restarts_from_a_non_positive_definite_initial_matrix():
    x, y = _logit_data(1000, 4, seed=8)
    fn = logit_objective(x, y)
    result = maximize_bfgs(first_order(fn), torch.zeros(4, dtype=F64),
                           initial_inverse_hessian=-torch.eye(4, dtype=F64))
    assert_true_maximum(result)
    assert result.diagnostics["restarts"] >= 1
    beta, _ = numpy_logit_newton(x, y)
    assert_allclose(result.theta.numpy(), beta, rtol=1e-6, atol=1e-8)
    with pytest.raises(KernelError) as error:
        maximize_bfgs(first_order(fn), torch.zeros(4, dtype=F64),
                      initial_inverse_hessian=torch.full((4, 4), math.nan, dtype=F64))
    assert error.value.code == "invalid_solver_options"


def test_bfgs_log_barrier_domain_with_non_finite_trials():
    weights = tensor([1.0, 5.0, 0.01])

    def fn(theta):
        if bool((theta <= 0).any()):
            return torch.tensor(-math.inf, dtype=F64), torch.full_like(theta, math.nan)
        return (weights * torch.log(theta) - theta).sum(), weights / theta - 1

    result = maximize_bfgs(fn, tensor([0.01, 0.01, 0.01]))
    assert_true_maximum(result)
    assert_allclose(result.theta.numpy(), weights.numpy(), rtol=1e-7)
    assert_allclose(result.hessian.numpy(), np.diag(-1 / weights.numpy()), rtol=1e-6)
    newton = maximize_newton(lambda t: (*fn(t), torch.diag(-weights / t**2)),
                             tensor([3.0, 3.0, 3.0]))
    assert_allclose(newton.theta.numpy(), weights.numpy(), rtol=1e-12)


def test_bfgs_k1_matches_scipy_and_the_closed_form():
    def fn(theta):
        return -(theta**4).sum() + 3 * theta.sum(), -4 * theta**3 + 3

    result = maximize_bfgs(fn, tensor([-7.0]))
    assert_true_maximum(result)
    root = (3 / 4) ** (1 / 3)
    assert result.theta.item() == pytest.approx(root, rel=1e-8)
    assert result.hessian.item() == pytest.approx(-12 * root**2, rel=1e-6)
    oracle = minimize(lambda t: -float(fn(tensor(t))[0]), [-7.0],
                      jac=lambda t: -fn(tensor(t))[1].numpy(), method="BFGS")
    assert result.theta.item() == pytest.approx(oracle.x[0], rel=1e-5)


def test_bfgs_max_iter_raises_before_the_hessian_but_returns_one_otherwise():
    x, y = _logit_data(500, 3, seed=1)
    fn = logit_objective(x, y)
    hessian_calls = []

    def counting(theta):
        hessian_calls.append(1)
        return fn(theta)[2]

    with pytest.raises(KernelError):
        maximize_bfgs(first_order(fn), torch.zeros(3, dtype=F64), max_iter=2, hessian_fn=counting)
    assert not hessian_calls
    result = maximize_bfgs(first_order(fn), torch.zeros(3, dtype=F64), max_iter=2,
                           hessian_fn=counting, raise_on_failure=False)
    assert not result.converged and result.iterations == 2 and len(hessian_calls) == 1
    assert result.diagnostics["polish_iterations"] == 0


# --- numerical derivatives ----------------------------------------------------


@pytest.mark.parametrize("theta", [700.0, -700.0, 1e-8, 1e8, 0.0])
def test_numerical_gradient_exp_and_cubic_across_magnitudes(theta):
    point = tensor([theta])
    if theta < 700.5:
        expected = math.exp(theta)
        assert_allclose(numerical_gradient(lambda t: torch.exp(t).sum(), point).numpy(),
                        [expected], rtol=1e-9)
    expected = 3 * theta**2
    assert_allclose(numerical_gradient(lambda t: (t**3).sum(), point).numpy(), [expected],
                    rtol=1e-9, atol=1e-12)
    assert_allclose(numerical_hessian(lambda t: 3 * t**2, point).numpy(), [[6 * theta]],
                    rtol=1e-9, atol=1e-12)


def test_numerical_hessian_with_a_zero_column_and_a_dominant_entry():
    def gradient_fn(theta):
        return torch.stack([2 * theta[0] + 1e6 * theta[2], torch.zeros((), dtype=F64),
                            1e6 * theta[0] + 1e12 * theta[2]])

    hessian = numerical_hessian(gradient_fn, tensor([0.3, -2.0, 7.0]))
    expected = np.array([[2.0, 0.0, 1e6], [0.0, 0.0, 0.0], [1e6, 0.0, 1e12]])
    assert_allclose(hessian.numpy(), expected, rtol=1e-9, atol=1e-6)
    assert torch.equal(hessian, hessian.T)


def test_numerical_gradient_of_a_function_rough_at_every_scale_stays_finite():
    point = tensor([0.3])

    def rough(t):
        return (t**3 + (t - 0.3).abs() ** 1.5).sum()

    calls = []
    gradient = numerical_gradient(lambda t: calls.append(1) or rough(t), point)
    assert bool(torch.isfinite(gradient).all())
    assert gradient.item() == pytest.approx(0.27, rel=2e-3)
    assert len(calls) <= 1 + 4 * 12 + 2 * 8


def test_numerical_derivatives_reject_wrong_shapes_and_coerce_outputs():
    theta = tensor([0.5, 1.5])
    with pytest.raises(KernelError) as error:
        numerical_hessian(lambda t: torch.cat([t, t]), theta)
    assert error.value.code == "invalid_objective"
    with pytest.raises(KernelError) as error:
        numerical_gradient(lambda t: t, theta)
    assert error.value.code == "invalid_objective"
    with pytest.raises(KernelError) as error:
        numerical_gradient(lambda t: (t.sum(), t), theta)
    assert error.value.code == "invalid_objective"
    with pytest.raises(KernelError) as error:
        numerical_gradient(lambda t: t.sum(), tensor([[0.5, 1.5]]))
    assert error.value.code == "invalid_parameters"
    for levels in (0, -1, 2.5, True):
        with pytest.raises(KernelError) as error:
            numerical_gradient(lambda t: t.sum(), theta, levels=levels)
        assert error.value.code == "invalid_solver_options"
    # [k, 1] and [1, k] gradients and Python floats are accepted.
    column = numerical_hessian(lambda t: (3 * t**2)[:, None], theta)
    row = numerical_hessian(lambda t: (3 * t**2)[None, :], theta)
    assert_allclose(column.numpy(), np.diag(6 * theta.numpy()), rtol=1e-9)
    assert torch.equal(column, row)
    assert_allclose(numerical_gradient(lambda t: float(t.sum()), theta).numpy(), [1.0, 1.0],
                    rtol=1e-10)
    plain = numerical_gradient(lambda t: float((t**2).sum()), theta, levels=1)
    assert_allclose(plain.numpy(), 2 * theta.numpy(), rtol=1e-8)


def test_numerical_hessian_cost_is_linear_in_k_and_accurate():
    rng = np.random.default_rng(17)
    n, k = 5000, 60
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x @ (0.2 * rng.normal(size=k))))).astype(float)
    fn = logit_objective(x, y)
    theta = maximize_newton(fn, torch.zeros(k, dtype=F64)).theta
    calls = []
    started = time.perf_counter()
    hessian = numerical_hessian(lambda t: calls.append(1) or fn(t)[1], theta)
    elapsed = time.perf_counter() - started
    assert len(calls) <= 8 * k and elapsed < 3.0
    exact = fn(theta)[2]
    assert float((hessian - exact).abs().max() / exact.abs().max()) < 1e-9


def test_check_derivatives_k1_and_asymmetric_hessians():
    def fn(theta):
        return (theta**3).sum(), 3 * theta**2, torch.diag(6 * theta)

    report = check_derivatives(fn, tensor([1.3]))
    assert report["gradient_max_rel_error"] < 1e-10 and report["hessian_max_rel_error"] < 1e-10
    assert report["hessian_asymmetry"] == 0.0

    def skewed(theta):
        hessian = torch.tensor([[-2.0, 1.0], [-1.0, -2.0]], dtype=F64)
        return -(theta**2).sum() + 0.0 * theta[0] * theta[1], -2 * theta, hessian

    report = check_derivatives(skewed, tensor([0.4, -0.2]))
    assert report["hessian_asymmetry"] == pytest.approx(2.0)
    assert report["hessian_max_rel_error"] < 1e-9       # the symmetric part is right
    with pytest.raises(KernelError) as error:
        check_derivatives(lambda t: (t.sum(),), tensor([0.4]))
    assert error.value.code == "invalid_objective"


# --- information_inverse and opg ---------------------------------------------


def test_information_inverse_extreme_magnitudes_and_k1():
    rng = np.random.default_rng(23)
    design = rng.normal(size=(50, 4))
    base = design.T @ design
    for magnitude in (1e-150, 1e-300, 1e150, 1e300):
        scaled = base * magnitude
        inverse = information_inverse(tensor(scaled)).numpy()
        expected = np.linalg.inv(base) / magnitude
        assert_allclose(inverse, expected, rtol=1e-9)
    for magnitude in (1e-320, 1e-310):
        with pytest.raises(KernelError) as error:
            information_inverse(tensor(base * magnitude))
        assert error.value.code == "singular_information"
    assert information_inverse(tensor([[1e-300]])).item() == pytest.approx(1e300, rel=1e-12)
    assert information_inverse(np.array([[2.0, 0.5], [0.5, 1.0]])).dtype == F64
    with pytest.raises(KernelError) as error:
        information_inverse(tensor([[1.0, 2.0], [2.0, 1.0]]) * 1e300)
    assert error.value.code == "singular_information"
    with pytest.raises(KernelError) as error:
        information_inverse(torch.zeros((0, 0), dtype=F64))
    assert error.value.code == "singular_information"
    with pytest.raises(KernelError) as error:
        information_inverse(tensor([4.0]))
    assert error.value.code == "singular_information"


def test_information_inverse_near_singular_matrix_matches_scaled_numpy_solve():
    rng = np.random.default_rng(24)
    design = rng.normal(size=(300, 5))
    design[:, 4] = design[:, 0] + 1e-5 * rng.normal(size=300)     # R-squared ~ 1 - 1e-10
    information = design.T @ design
    inverse = information_inverse(tensor(information)).numpy()
    assert_allclose(inverse @ information, np.eye(5), atol=1e-5)
    scale = 1 / np.sqrt(np.diag(information))
    expected = np.linalg.solve(information * np.outer(scale, scale), np.eye(5)) * np.outer(scale,
                                                                                            scale)
    # Condition number ~1e10: either factorization is only accurate to ~1e-6 relative.
    assert_allclose(inverse, expected, rtol=1e-5)


def test_opg_matches_a_brute_force_loop_and_handles_shapes():
    rng = np.random.default_rng(25)
    scores = rng.normal(size=(37, 3)) * np.array([1e-5, 1.0, 1e5])
    expected = np.zeros((3, 3))
    for row in scores:
        expected += np.outer(row, row)
    assert_allclose(opg(tensor(scores)).numpy(), expected, rtol=1e-13)
    assert opg(tensor(scores[:, :1])).shape == (1, 1)
    assert opg(tensor(scores[:1])).shape == (3, 3)
    assert np.linalg.matrix_rank(opg(tensor(scores[:1])).numpy()) == 1
    assert opg(torch.ones(4, 2, dtype=torch.float32)).dtype == F64
    assert_allclose(opg(torch.ones(4, 2, dtype=torch.int64)).numpy(), np.full((2, 2), 4.0))
    for bad in (torch.zeros(5, dtype=F64), torch.zeros(5, 2, 1, dtype=F64), np.zeros((5, 2))):
        with pytest.raises(KernelError) as error:
            opg(bad)
        assert error.value.code == "invalid_scores"


def test_opg_on_a_million_rows_is_fast_and_small():
    scores = torch.randn(1_000_000, 5, dtype=F64, generator=torch.Generator().manual_seed(0))
    started = time.perf_counter()
    meat = opg(scores)
    assert time.perf_counter() - started < 1.0
    assert meat.shape == (5, 5) and torch.equal(meat, meat.T)
    assert_allclose(meat.numpy(), scores.numpy().T @ scores.numpy(), rtol=1e-12)


# --- module contract ----------------------------------------------------------


def test_runtime_imports_and_no_autograd_or_observation_loops():
    source = Path(optimize.__file__).read_text()
    tree = ast.parse(source)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                roots.add(node.module.split(".")[0])
            else:
                assert node.module in {"contracts"}
    assert roots <= {"__future__", "dataclasses", "math", "typing", "torch"}
    for forbidden in ("numpy", "scipy", "statsmodels", "pandas", "sklearn", "torch.func",
                      "torch.autograd", "requires_grad", "enable_grad", "backward("):
        assert forbidden not in source.replace("no autograd", ""), forbidden


def test_results_are_plain_tensors_without_grad_even_when_grad_mode_is_on():
    x, y = _logit_data(200, 2, seed=2)
    fn = logit_objective(x, y)
    start = torch.zeros(2, dtype=F64, requires_grad=True)
    with torch.enable_grad():
        newton = maximize_newton(fn, start)
        bfgs = maximize_bfgs(first_order(fn), start)
        gradient = numerical_gradient(lambda t: fn(t)[0], start)
        hessian = numerical_hessian(lambda t: fn(t)[1], start)
    for item in (newton.theta, newton.gradient, newton.hessian, bfgs.theta, bfgs.hessian,
                 gradient, hessian):
        assert not item.requires_grad and item.grad_fn is None
    assert start.grad is None


def _logit_data(n, k, seed):
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    beta = 0.5 * rng.normal(size=k)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x @ beta))).astype(float)
    return x, y
