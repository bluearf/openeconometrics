"""Independent SciPy constrained/support oracles; NumPy/SciPy are dev-only.

The oracle enumerates small sign/support models and uses a separate external
quasi-Newton optimizer; it never calls production score/CD helpers.  Finite
difference derivatives additionally certify the reported objective's KKT.
"""

import itertools
import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.special import expit

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.regularized.glm_kernels import (
    BACKTRACKS,
    INNER_SWEEPS,
    solve_path,
    solver_work,
)


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def sample(family, n=91):
    random = np.random.default_rng(3217)
    x = random.normal(size=(n, 2))
    x[:, 1] += 0.3 * x[:, 0]
    eta = 0.2 + 0.65 * x[:, 0] - 0.4 * x[:, 1]
    y = random.binomial(1, expit(eta)) if family == "binomial" else random.poisson(np.exp(eta))
    weights = random.integers(1, 5, size=n).astype(float)
    return x, y.astype(float), weights


def reference_terms(beta, design, y, weights, family):
    eta = design @ beta
    if family == "binomial":
        mean = expit(eta)
        loss = np.dot(weights, np.logaddexp(0, eta) - y * eta)
        curvature = weights * mean * (1 - mean)
    else:
        mean = np.exp(eta)
        loss = np.dot(weights, mean - y * eta)
        curvature = weights * mean
    gradient = design.T @ (weights * (mean - y))
    hessian = design.T @ (curvature[:, None] * design)
    return loss, gradient, hessian


def independent_oracle(x, y, weights, family, penalty, ratio, factors, intercept):
    """Enumerated sign/support plus SciPy L-BFGS-B, unlike Newton/CD runtime."""
    weights = weights / weights.sum()
    design = np.column_stack([np.ones(len(y)), x]) if intercept else x
    loads = np.r_[0.0, factors] if intercept else np.array(factors)
    p = x.shape[1]
    if penalty * ratio == 0:

        def smooth(beta):
            loss, gradient, hessian = reference_terms(beta, design, y, weights, family)
            ridge = penalty * (1 - ratio) * loads
            return (
                loss + 0.5 * np.dot(ridge, beta**2),
                gradient + ridge * beta,
                hessian + np.diag(ridge),
            )

        fit = minimize(
            lambda b: smooth(b)[0],
            np.zeros(design.shape[1]),
            jac=lambda b: smooth(b)[1],
            hess=lambda b: smooth(b)[2],
            method="trust-exact",
            options={"gtol": 1e-12, "maxiter": 1000},
        )
        assert np.max(np.abs(smooth(fit.x)[1])) < 1e-8
        return fit.x, float(smooth(fit.x)[0])
    candidates = []
    for signs in itertools.product([-1, 0, 1], repeat=p):
        active_slopes = np.flatnonzero(signs)
        active = np.r_[0, active_slopes + 1] if intercept else active_slopes
        sign = np.r_[0.0, signs] if intercept else np.array(signs, dtype=float)
        coefficients = np.zeros(design.shape[1])
        l1 = penalty * ratio * loads
        ridge = penalty * (1 - ratio) * loads

        def smooth(b):
            beta = np.zeros(design.shape[1])
            beta[active] = b
            loss, gradient, _ = reference_terms(beta, design, y, weights, family)
            return (
                loss + np.dot(l1 * sign, beta) + 0.5 * np.dot(ridge, beta**2),
                (gradient + l1 * sign + ridge * beta)[active],
            )

        if len(active):
            bounds = [
                (None, None)
                if (intercept and j == 0)
                else ((0.0, None) if sign[j] > 0 else (None, 0.0))
                for j in active
            ]
            fit = minimize(
                lambda b: smooth(b)[0],
                np.zeros(len(active)),
                jac=lambda b: smooth(b)[1],
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000, "maxls": 50},
            )
            coefficients[active] = fit.x
        loss, gradient, _ = reference_terms(coefficients, design, y, weights, family)
        gradient += ridge * coefficients
        residual = np.where(
            np.abs(coefficients) > 1e-8,
            np.abs(gradient + l1 * np.sign(coefficients)),
            np.maximum(np.abs(gradient) - l1, 0),
        )
        if residual.max(initial=0) <= 1e-7:
            objective = (
                loss + np.dot(l1, np.abs(coefficients)) + 0.5 * np.dot(ridge, coefficients**2)
            )
            candidates.append((objective, coefficients))
    assert candidates, "Independent sign/support optimizer found no KKT-valid optimum."
    objective, coefficients = min(candidates, key=lambda entry: entry[0])
    return coefficients, float(objective)


def fit(x, y, weights, family, ratio=0.5, factors=None, intercept=True, path=None, **kwargs):
    return solve_path(
        tensor(x),
        tensor(y),
        tensor(weights),
        [0.8, 0.12, 0.0] if path is None else path,
        family=family,
        l1_ratio=ratio,
        penalty_factors=tensor([1.0] * x.shape[1] if factors is None else factors),
        intercept=intercept,
        max_iterations=kwargs.pop("max_iterations", 200),
        tolerance=kwargs.pop("tolerance", 1e-10),
        **kwargs,
    )


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("factors", [[1.0, 1.0], [0.0, 3.7], [0.3, 4.1]])
def test_path_matches_independent_support_optima_and_finite_difference_kkt(
    family, ratio, intercept, factors
):
    x, y, weights = sample(family)
    fits = fit(x, y, weights, family, ratio, factors, intercept)
    wn = weights / weights.sum()
    design = np.column_stack([np.ones(len(y)), x]) if intercept else x
    loads = np.r_[0.0, factors] if intercept else np.array(factors)
    for record in fits:
        lam = record["penalty"]
        actual = (
            np.r_[record["constant"], record["coefficients"]]
            if intercept
            else np.array(record["coefficients"])
        )
        oracle, objective = independent_oracle(
            x, y, weights, family, lam, ratio, factors, intercept
        )
        np.testing.assert_allclose(actual, oracle, atol=8e-7, rtol=8e-7)
        assert record["objective"] == pytest.approx(objective, abs=2e-12)
        assert record["converged"] is True and record["kkt_max"] <= 1e-10
        assert record["free_newton_correction"] <= 1e-5
        assert record["inner_sweeps"] <= record["iterations"] * INNER_SWEEPS
        assert record["backtracks"] < record["iterations"] * BACKTRACKS or record["iterations"] == 0

        # Central finite differences of a NumPy smooth objective, with exact
        # signed/zero L1 subgradient rather than a mirrored Torch score.
        def smooth(beta):
            loss = reference_terms(beta, design, y, wn, family)[0]
            return loss + lam * (1 - ratio) * np.dot(loads, beta**2) / 2

        derivative = np.zeros_like(actual)
        for j in range(len(actual)):
            perturbation = np.zeros_like(actual)
            perturbation[j] = 1e-5
            derivative[j] = (smooth(actual + perturbation) - smooth(actual - perturbation)) / 2e-5
        l1 = lam * ratio * loads
        residual = np.where(
            actual != 0, abs(derivative + l1 * np.sign(actual)), np.maximum(abs(derivative) - l1, 0)
        )
        assert np.max(residual, initial=0) < 2e-8


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_literal_frequency_replication_and_extreme_uniform_weight_rescaling(family, ratio):
    x, y, weights = sample(family, n=57)
    original = fit(x, y, weights, family, ratio, factors=[0.0, 2.4], path=[0.15])[-1]
    repeat = weights.astype(int)
    replicated = fit(
        np.repeat(x, repeat, axis=0),
        np.repeat(y, repeat),
        np.ones(repeat.sum()),
        family,
        ratio,
        factors=[0.0, 2.4],
        path=[0.15],
    )[-1]
    for record in [replicated] + [
        fit(x, y, weights * scale, family, ratio, factors=[0.0, 2.4], path=[0.15])[-1]
        for scale in [1e-280, 1e307]
    ]:
        np.testing.assert_allclose(record["coefficients"], original["coefficients"], atol=1e-9)
        assert record["constant"] == pytest.approx(original["constant"], abs=1e-9)
        assert record["objective"] == pytest.approx(original["objective"], abs=2e-12)


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_penalty_factors_are_literal_for_l1_and_l2_without_hidden_rescaling(family, ratio):
    x, y, weights = sample(family)
    a = fit(x, y, weights, family, ratio, factors=[1.8, 3.2], path=[0.1])[-1]
    b = fit(x, y, weights, family, ratio, factors=[0.6, 3.2 / 3], path=[0.3])[-1]
    c = fit(x, y, weights, family, ratio, factors=[0.6, 3.2 / 3], path=[0.1])[-1]
    np.testing.assert_allclose(a["coefficients"], b["coefficients"], atol=1e-9)
    assert a["constant"] == pytest.approx(b["constant"], abs=1e-9)
    assert a["objective"] == pytest.approx(b["objective"], abs=1e-12)
    assert not np.allclose(a["coefficients"], c["coefficients"], atol=1e-4)


@pytest.mark.parametrize(
    "family,y", [("binomial", [0.0, 0.0, 1.0, 1.0, 1.0]), ("poisson", [0.0, 1.0, 3.0, 2.0, 4.0])]
)
def test_intercept_only_analytic_solution(family, y):
    y, weights = np.array(y), np.array([1.0, 3.0, 2.0, 4.0, 1.0])
    record = fit(np.empty((len(y), 0)), y, weights, family, factors=[], path=[0.4])[-1]
    mean = np.dot(weights, y) / weights.sum()
    expected = math.log(mean / (1 - mean)) if family == "binomial" else math.log(mean)
    assert record["constant"] == pytest.approx(expected, abs=1e-13)
    assert record["coefficients"] == [] and record["kkt_max"] <= 1e-10


@pytest.mark.parametrize(
    "family,x,y,factors,penalty",
    [
        ("binomial", [[-2.0], [-1.0], [1.0], [2.0]], [0.0, 0.0, 1.0, 1.0], [1.0], 0.0),
        ("binomial", [[-2.0], [-1.0], [1.0], [2.0]], [0.0, 0.0, 1.0, 1.0], [0.0], 0.2),
        ("binomial", [[-1.0], [0.0], [0.0], [1.0]], [0.0, 0.0, 1.0, 1.0], [1.0], 0.0),
        ("poisson", [[0.0], [0.0], [1.0], [1.0]], [1.0, 2.0, 0.0, 0.0], [0.0], 0.2),
    ],
)
def test_divergent_free_directions_never_pass_on_small_score_alone(family, x, y, factors, penalty):
    with pytest.raises(AnalysisError, match="separation|unpenalized|identified"):
        fit(
            np.array(x),
            np.array(y),
            np.ones(len(y)),
            family,
            factors=factors,
            path=[penalty],
            tolerance=1e-8,
        )


@pytest.mark.parametrize(
    "family,y", [("binomial", [1.0] * 6), ("binomial", [0.0] * 6), ("poisson", [0.0] * 6)]
)
def test_degenerate_intercept_outcome_refused(family, y):
    with pytest.raises(AnalysisError, match="both outcome classes|All-zero"):
        fit(np.arange(6.0)[:, None], np.array(y), np.ones(6), family, path=[0.2])


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_identification_and_iteration_limit_do_not_return_approximate_fits(family):
    x, y, weights = sample(family)
    duplicate = np.column_stack([x[:, 0], x[:, 0]])
    with pytest.raises(AnalysisError, match="not numerically identified"):
        fit(duplicate, y, weights, family, factors=[0.0, 0.0], path=[0.2])
    with pytest.raises(AnalysisError, match="not numerically identified"):
        fit(duplicate, y, weights, family, path=[0.2, 0.0])
    with pytest.raises(AnalysisError, match="after 1 Newton"):
        fit(x, y, weights, family, path=[0.1], max_iterations=1)


@pytest.mark.parametrize(
    "change",
    [
        "dtype",
        "nan",
        "zero_weight",
        "negative_factor",
        "weight_underflow",
        "ascending",
        "repeat",
        "negative_penalty",
        "bad_tolerance",
        "bad_iterations",
        "binary_fraction",
    ],
)
def test_strict_kernel_input_guards(change):
    x, y, weights = sample("binomial")
    arguments = dict(
        z=tensor(x),
        y=tensor(y),
        weights=tensor(weights),
        lambdas=[0.2, 0.1],
        family="binomial",
        l1_ratio=0.5,
        penalty_factors=tensor([1.0, 1.0]),
        intercept=True,
        max_iterations=200,
        tolerance=1e-10,
    )
    if change == "dtype":
        arguments["z"] = arguments["z"].float()
    if change == "nan":
        arguments["z"][0, 0] = math.nan
    if change == "zero_weight":
        arguments["weights"][0] = 0.0
    if change == "negative_factor":
        arguments["penalty_factors"][0] = -1.0
    if change == "weight_underflow":
        arguments["weights"][[0, 1]] = tensor([1e-300, 1e300])
    if change == "ascending":
        arguments["lambdas"] = [0.1, 0.2]
    if change == "repeat":
        arguments["lambdas"] = [0.1, 0.1]
    if change == "negative_penalty":
        arguments["lambdas"] = [-0.1]
    if change == "bad_tolerance":
        arguments["tolerance"] = 1e-15
    if change == "bad_iterations":
        arguments["max_iterations"] = True
    if change == "binary_fraction":
        arguments["y"][0] = 0.5
    with pytest.raises(AnalysisError):
        solve_path(**arguments)


def test_work_bound_includes_all_bounded_solver_loops():
    n, p, outer, candidates = 101, 4, 200, 3
    q = p + 1
    assert solver_work(n, p, outer, candidates) > candidates * outer * (
        n * q**2 + INNER_SWEEPS * q**2 + BACKTRACKS * n + q**3
    )
    assert solver_work(n, p, outer, candidates + 1) > solver_work(n, p, outer, candidates)


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_identified_extreme_correlation_finishes_bounded_path_with_independent_kkt(family, ratio):
    # Regression for discovered CD mixing failures at correlation ~.999.
    # The objective remains convex, so independently evaluated KKT certifies
    # the optimum even when a large sign/support enumeration is impractical.
    random = np.random.default_rng(188)
    x = random.normal(size=(300, 8)) * math.sqrt(0.001)
    x += random.normal(size=(300, 1)) * math.sqrt(0.999)
    eta = 0.2 + 0.7 * x[:, 0] - 0.3 * x[:, 1]
    y = random.binomial(1, expit(eta)) if family == "binomial" else random.poisson(np.exp(eta))
    design = np.column_stack([np.ones(len(y)), x])
    loads = np.r_[0.0, np.ones(8)]
    for record in fit(x, y, np.ones(len(y)), family, ratio, path=[1.0, 0.1, 0.001, 0.0]):
        beta = np.r_[record["constant"], record["coefficients"]]
        lam = record["penalty"]
        loss, gradient, _ = reference_terms(beta, design, y, np.ones(len(y)) / len(y), family)
        smooth_gradient = gradient + lam * (1 - ratio) * loads * beta
        l1 = lam * ratio * loads
        residual = np.where(
            beta != 0,
            abs(smooth_gradient + l1 * np.sign(beta)),
            np.maximum(abs(smooth_gradient) - l1, 0),
        )
        assert residual.max() < 2e-10
        assert record["objective"] == pytest.approx(
            loss + np.dot(l1, abs(beta)) + lam * (1 - ratio) * np.dot(loads, beta**2) / 2, abs=2e-12
        )
        assert record["converged"] is True
