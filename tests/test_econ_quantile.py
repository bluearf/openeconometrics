"""Independent oracles for quantile regression (qreg) and its solver.

Coefficients are checked against the exact linear program solved by SciPy's
HiGHS (the gold standard for the vertex solution) and against statsmodels'
QuantReg; covariances against the formulas of Stata's qreg written out in
NumPy (sparsity by fitted values, residual quantiles and kernel density; the
Powell sandwich; the Parente-Santos Silva cluster covariance); weights against
duplicated rows.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import integrate, stats
from scipy.optimize import linprog
from statsmodels.regression import quantile_regression as smq

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.quantile import kernels
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle

X_COLS = ["x1", "x2", "x3"]


def lp_quantile(x, y, tau, w=None):
    """min tau w'u + (1 - tau) w'v  s.t.  x b + u - v = y, u, v >= 0  (HiGHS)."""
    n, k = x.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    cost = np.concatenate([np.zeros(k), tau * w, (1 - tau) * w])
    a_eq = np.hstack([x, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    solution = linprog(cost, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    assert solution.status == 0
    return solution.x[:k], solution.fun


def make_data(seed=11, n=240):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.exponential(size=n),
                          "x3": rng.uniform(-1, 1, size=n)})
    noise = rng.standard_t(4, size=n) * (1 + 0.4 * frame.x2)
    frame["y"] = 1 + 0.5 * frame.x1 - 0.3 * frame.x2 + 0.8 * frame.x3 + noise
    frame["g"] = rng.integers(0, 30, size=n)
    frame["f"] = rng.integers(1, 4, size=n).astype(float)
    frame["w"] = rng.uniform(0.3, 2.5, size=n)
    frame["sector"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def design(frame, cols=X_COLS):
    return np.column_stack([np.ones(len(frame)), frame[cols].to_numpy()])


def params(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


# ---- oracles written from the Stata formulas ---------------------------------------------


def bandwidth(tau, n, rule):
    z, za = stats.norm.ppf(tau), stats.norm.ppf(0.975)
    if rule == "hsheather":
        shape = 1.5 * stats.norm.pdf(z) ** 2 / (2 * z * z + 1)
        return n ** (-1 / 3) * za ** (2 / 3) * shape ** (1 / 3)
    if rule == "bofinger":
        return n ** (-0.2) * (4.5 * stats.norm.pdf(z) ** 4 / (2 * z * z + 1) ** 2) ** 0.2
    return za * np.sqrt(tau * (1 - tau) / n)


KERNELS = {
    "epanechnikov": lambda u: np.where(np.abs(u) < np.sqrt(5),
                                       0.75 * (1 - u ** 2 / 5) / np.sqrt(5), 0),
    "epan2": lambda u: np.where(np.abs(u) < 1, 0.75 * (1 - u ** 2), 0),
    "biweight": lambda u: np.where(np.abs(u) < 1, 15 / 16 * (1 - u ** 2) ** 2, 0),
    "cosine": lambda u: np.where(np.abs(u) < 0.5, 1 + np.cos(2 * np.pi * u), 0),
    "gaussian": stats.norm.pdf,
    "parzen": lambda u: np.where(np.abs(u) <= 0.5, 4 / 3 - 8 * u ** 2 + 8 * np.abs(u) ** 3,
                                 np.where(np.abs(u) <= 1, 8 * (1 - np.abs(u)) ** 3 / 3, 0)),
    "rectangle": lambda u: np.where(np.abs(u) < 1, 0.5, 0),
    "triangle": lambda u: np.where(np.abs(u) < 1, 1 - np.abs(u), 0),
}


def stata_quantile(values, p):
    return np.quantile(values, p, method="averaged_inverted_cdf")


def kernel_scale(resid, tau, h):
    iqr = stata_quantile(resid, 0.75) - stata_quantile(resid, 0.25)
    return min(resid.std(ddof=1), iqr / 1.34) * (stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h))


def oracle_covariance(x, y, tau, kind, *, density="fitted", rule="hsheather",
                      kernel="epanechnikov", groups=None):
    n = len(y)
    beta, _ = lp_quantile(x, y, tau)
    resid = y - x @ beta
    resid[np.abs(resid) < 1e-9] = 0.0
    h = bandwidth(tau, n, rule)
    xtx_inv = np.linalg.inv(x.T @ x)
    if kind == "nonrobust":
        if density == "fitted":
            high, low = lp_quantile(x, y, tau + h)[0], lp_quantile(x, y, tau - h)[0]
            sparsity = x.mean(axis=0) @ (high - low) / (2 * h)
        elif density == "residual":
            sparsity = (stata_quantile(resid, tau + h) - stata_quantile(resid, tau - h)) / (2 * h)
        else:
            c = kernel_scale(resid, tau, h)
            sparsity = 1 / (KERNELS[kernel](resid / c).sum() / (n * c))
        return tau * (1 - tau) * sparsity ** 2 * xtx_inv, sparsity, h
    c = kernel_scale(resid, tau, h)
    d_inv = np.linalg.inv((x * (KERNELS[kernel](resid / c) / c)[:, None]).T @ x)
    if kind == "robust":
        meat = tau * (1 - tau) * x.T @ x
    else:
        psi = tau - (resid <= 0)
        meat = np.zeros((x.shape[1], x.shape[1]))
        for g in np.unique(groups):
            score = (x[groups == g] * psi[groups == g][:, None]).sum(axis=0)
            meat += np.outer(score, score)
    return d_inv @ meat @ d_inv, None, h


# ---- the solver ----------------------------------------------------------------------------


def test_solver_matches_linear_program_on_random_problems():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(8, 250))
        k = int(rng.integers(1, min(6, n - 1)))
        x = np.column_stack([np.ones(n)] + [rng.normal(size=n) * 10 ** rng.uniform(-2, 2)
                                             + rng.uniform(-5, 5) for _ in range(k - 1)])
        noise = (rng.standard_t(2, size=n) if trial % 2
                 else rng.normal(size=n) * (1 + np.abs(x[:, -1])))
        y = x @ rng.normal(size=k) + noise
        tau = float(rng.choice([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.33]))
        w = rng.uniform(0.2, 3, size=n) if trial % 3 == 0 else None
        beta, objective = lp_quantile(x, y, tau, w)
        fit = kernels.quantile_regression(torch.tensor(x), torch.tensor(y), tau,
                                          None if w is None else torch.tensor(w))
        assert_allclose(fit.objective, objective, rtol=1e-10, atol=1e-10)
        if fit.unique:
            assert_allclose(fit.beta.numpy(), beta, rtol=1e-7, atol=1e-8)
        # A vertex: the fitted hyperplane interpolates exactly k observations.
        assert int((fit.resid == 0).sum()) == k == len(fit.basis)
        assert_allclose(y[fit.basis.numpy()], x[fit.basis.numpy()] @ fit.beta.numpy(),
                        rtol=1e-9, atol=1e-9)


def test_solver_simplex_fallback_reaches_the_same_vertex():
    rng = np.random.default_rng(3)
    for trial in range(12):
        n, k = int(rng.integers(30, 200)), int(rng.integers(2, 5))
        x = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(k - 1)])
        y = x @ rng.normal(size=k) + rng.standard_t(3, size=n)
        tau = float(rng.choice([0.2, 0.5, 0.8]))
        beta, objective = lp_quantile(x, y, tau)
        problem = kernels.prepare(torch.tensor(x), torch.tensor(y))
        # No interior-point iterations at all: pure simplex from the least-squares fit.
        fit = kernels.solve(problem, tau, max_iterations=trial % 3, max_pivots=10_000)
        assert fit.pivots > 0 or trial % 3
        assert_allclose(fit.objective, objective, rtol=1e-10)
        assert_allclose(fit.beta.numpy(), beta, rtol=1e-7, atol=1e-8)
        reference = kernels.solve(problem, tau)
        assert reference.pivots == 0
        assert_allclose(reference.beta.numpy(), fit.beta.numpy(), rtol=1e-9, atol=1e-10)


def test_solver_degenerate_designs_and_ties():
    rng = np.random.default_rng(1)
    nonunique = 0
    for trial in range(40):
        n = int(rng.integers(10, 150))
        x = np.column_stack([np.ones(n), rng.integers(0, 2, size=n), rng.integers(0, 3, size=n)]
                            ).astype(float)
        if np.linalg.matrix_rank(x) < 3:
            continue
        y = rng.integers(0, 6, size=n).astype(float)
        if trial % 4 == 0:
            x, y = np.vstack([x, x]), np.concatenate([y, y])
        tau = float(rng.choice([0.25, 0.5, 0.75]))
        _, objective = lp_quantile(x, y, tau)
        fit = kernels.quantile_regression(torch.tensor(x), torch.tensor(y), tau)
        assert_allclose(fit.objective, objective, rtol=1e-10, atol=1e-10)
        nonunique += not fit.unique
    assert nonunique > 0


def lp_solution_width(x, y, tau, w):
    """Largest spread of any coefficient over the set of linear-programming minimizers."""
    n, k = x.shape
    cost = np.concatenate([np.zeros(k), tau * w, (1 - tau) * w])
    a_eq = np.hstack([x, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    best = linprog(cost, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    width = 0.0
    for j in range(k):
        for sign in (1.0, -1.0):
            target = np.zeros(k + 2 * n)
            target[j] = sign
            extreme = linprog(target, A_eq=a_eq, b_eq=y, A_ub=cost[None, :],
                              b_ub=[best.fun * (1 + 1e-10) + 1e-10], bounds=bounds, method="highs")
            if extreme.status != 0:
                return best.fun, None
            width = max(width, abs(sign * extreme.fun - best.x[j]))
    return best.fun, width


def test_solver_ties_and_the_uniqueness_flag_are_sound():
    # Discrete outcomes and regressors, duplicated rows and integer weights: degenerate
    # vertices and frequently several minimizers. The objective must be the LP minimum and
    # `unique` must never be claimed for a solution set with positive width.
    rng = np.random.default_rng(101)
    flagged = {(True, True): 0, (False, True): 0, (False, False): 0}
    pivots = 0
    for trial in range(70):
        n = int(rng.integers(5, 40))
        k = int(rng.integers(1, 5))
        columns = [np.ones(n)] + [rng.integers(0, 3, size=n).astype(float) if rng.uniform() < 0.6
                                  else np.round(rng.normal(size=n)) for _ in range(k - 1)]
        x = np.column_stack(columns)
        y = (rng.integers(0, 4, size=n).astype(float) if trial % 3
             else np.round(rng.normal(size=n), 1))
        if trial % 5 == 0:
            x, y = np.vstack([x, x, x[:3]]), np.concatenate([y, y, y[:3]])
        if np.linalg.matrix_rank(x) < x.shape[1] or len(y) <= x.shape[1]:
            continue
        w = rng.integers(1, 4, size=len(y)).astype(float) if trial % 2 else np.ones(len(y))
        tau = float(rng.choice([0.25, 0.5, 0.75, 1 / 3, 0.2]))
        objective, width = lp_solution_width(x, y, tau, w)
        if width is None:
            continue
        problem = kernels.prepare(torch.tensor(x), torch.tensor(y),
                                  torch.tensor(w) if trial % 2 else None)
        fit = kernels.solve(problem, tau)
        assert_allclose(fit.objective, objective, rtol=1e-9, atol=1e-9)
        truly_unique = width < 1e-6
        assert not (fit.unique and not truly_unique), trial
        flagged[(fit.unique, truly_unique)] += 1
        # Tied vertices are normally certified by the interior dual point without pivoting ...
        assert fit.pivots <= 3
        # ... and the simplex path (no interior-point iterations at all) must agree: it
        # walks through the degenerate vertices with the perturbation rule.
        simplex = kernels.solve(problem, tau, max_iterations=0, max_pivots=10_000)
        assert_allclose(simplex.objective, objective, rtol=1e-9, atol=1e-9)
        assert not (simplex.unique and not truly_unique), trial
        if truly_unique:
            assert_allclose(simplex.beta.numpy(), fit.beta.numpy(), rtol=1e-8, atol=1e-9)
        pivots += simplex.pivots
    assert flagged[(True, True)] > 20 and flagged[(False, False)] > 3
    # The flag is conservative only in rare tied cases.
    assert flagged[(False, True)] <= 3
    assert pivots > 100        # degenerate pivots were exercised


def test_solver_outcome_with_tiny_variation_around_a_large_level():
    rng = np.random.default_rng(6)
    for trial in range(12):
        n, k = int(rng.integers(20, 120)), int(rng.integers(1, 6))
        x = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(k - 1)])
        signal = x @ rng.normal(size=k) + rng.normal(size=n)
        tau = float(rng.choice([0.05, 0.5, 0.9]))
        beta, objective = lp_quantile(x, signal, tau)
        fit = kernels.quantile_regression(torch.tensor(x), torch.tensor(1e3 + 1e-9 * signal), tau)
        # The level is absorbed by the intercept; slopes and loss scale by 1e-9 (the data
        # themselves carry only four significant digits of the signal).
        assert_allclose(fit.objective, objective * 1e-9, rtol=2e-3)
        assert_allclose(fit.beta.numpy()[1:], beta[1:] * 1e-9, rtol=0, atol=2e-12)


def test_solver_large_problem_satisfies_the_optimality_conditions():
    # Too large for the dense LP oracle: verify the Kuhn-Tucker conditions directly.
    rng = np.random.default_rng(12)
    n, k = 200_000, 8
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = x @ np.linspace(-1, 1, k) + rng.standard_t(3, size=n) * (1 + 0.5 * np.abs(x[:, 1]))
    w = rng.uniform(0.5, 2, size=n)
    for tau, weights in ((0.5, None), (0.1, w), (0.9, None)):
        fit = kernels.quantile_regression(torch.tensor(x), torch.tensor(y), tau,
                                          None if weights is None else torch.tensor(weights))
        beta, basis = fit.beta.numpy(), fit.basis.numpy()
        resid = y - x @ beta
        outside = np.ones(n, dtype=bool)
        outside[basis] = False
        assert np.abs(resid[basis]).max() < 1e-9 and np.abs(resid[outside]).min() > 1e-9
        scale = np.ones(n) if weights is None else weights
        psi = scale * (tau - (resid < 0))
        # sum_{i in h} lambda_j w_j x_j = - sum_{i not in h} w_i psi_i x_i, tau-1 < lambda < tau.
        score = (x[outside] * psi[outside][:, None]).sum(axis=0)
        multipliers = np.linalg.solve(x[basis].T, -score) / scale[basis]
        assert multipliers.min() > tau - 1 and multipliers.max() < tau
        assert fit.unique and fit.pivots == 0 and fit.iterations < 40
        assert_allclose(fit.objective, (scale * np.where(resid < 0, (tau - 1) * resid,
                                                         tau * resid)).sum(), rtol=1e-12)


def test_solver_scaling_offsets_and_exact_fit():
    rng = np.random.default_rng(4)
    n = 300
    year = 2000 + rng.integers(0, 20, size=n).astype(float)
    big = rng.normal(size=n) * 1e6
    y = 5 + 3 * (year - 2000) + big * 1e-6 + rng.normal(size=n)
    beta, objective = lp_quantile(np.column_stack([np.ones(n), year - 2000, big / 1e6]), y, 0.3)
    fit = kernels.quantile_regression(torch.tensor(np.column_stack([np.ones(n), year, big])),
                                      torch.tensor(y), 0.3)
    assert_allclose(fit.objective, objective, rtol=1e-9)
    assert_allclose(fit.beta.numpy()[1:], beta[1:] / np.array([1, 1e6]), rtol=1e-7)
    exact = kernels.quantile_regression(torch.tensor(np.column_stack([np.ones(n), year])),
                                        torch.tensor(2 + 0.5 * year), 0.5)
    assert exact.exact_fit and exact.objective == 0
    assert_allclose(exact.beta.numpy(), [2, 0.5], rtol=1e-8, atol=1e-6)


def test_solver_rejects_invalid_input():
    x = torch.ones((5, 1), dtype=torch.float64)
    y = torch.arange(5, dtype=torch.float64)
    for tau in (0.0, 1.0, -0.1, 1.5, True):
        with pytest.raises(KernelError) as caught:
            kernels.quantile_regression(x, y, tau)
        assert caught.value.code == "invalid_quantile"
    with pytest.raises(KernelError) as caught:
        kernels.quantile_regression(torch.ones((2, 2), dtype=torch.float64), y[:2], 0.5)
    assert caught.value.code == "insufficient_observations"
    with pytest.raises(KernelError) as caught:
        kernels.quantile_regression(torch.ones((5, 2), dtype=torch.float64), y, 0.5)
    assert caught.value.code == "singular_design"
    with pytest.raises(KernelError) as caught:
        kernels.quantile_regression(x, y, 0.5, torch.tensor([1.0, 1, 0, 1, 1], dtype=torch.float64))
    assert caught.value.code == "invalid_weights"
    with pytest.raises(KernelError) as caught:
        kernels.quantile_regression(x, torch.tensor([1.0, 2, float("nan"), 3, 4]), 0.5)
    assert caught.value.code in {"non_finite_values", "invalid_design"}


def test_kernels_and_bandwidths_match_independent_definitions():
    grid = torch.linspace(-3, 3, 601, dtype=torch.float64)
    for name, function in KERNELS.items():
        assert_allclose(kernels.kernel_values(grid, name).numpy(), function(grid.numpy()),
                        rtol=1e-13, atol=1e-15)
        area, _ = integrate.quad(lambda u, f=function: float(f(np.array(u))), -8, 8, limit=400,
                                 points=[-np.sqrt(5), -1, -0.5, 0.5, 1, np.sqrt(5)])
        assert_allclose(area, 1.0, rtol=1e-8)
    # statsmodels implements four of them under other names.
    pairs = (("epan2", "epa"), ("cosine", "cos"), ("gaussian", "gau"), ("parzen", "par"))
    for ours, theirs in pairs:
        assert_allclose(kernels.kernel_values(grid, ours).numpy(),
                        smq.kernels[theirs](grid.numpy()), rtol=1e-12, atol=1e-15)
    for tau in (0.1, 0.5, 0.83):
        for n in (50, 1000):
            assert_allclose(kernels.bandwidth(tau, n, "hsheather"), smq.hall_sheather(n, tau),
                            rtol=1e-11)
            if tau == 0.5:
                # statsmodels evaluates the density at 2 z; the two agree only at the median
                # (z = 0). Koenker's bandwidth.rq uses phi(z), as coded in the oracle above.
                assert_allclose(kernels.bandwidth(tau, n, "bofinger"), smq.bofinger(n, tau),
                                rtol=1e-11)
            assert_allclose(kernels.bandwidth(tau, n, "chamberlain"), smq.chamberlain(n, tau),
                            rtol=1e-11)
            for rule in kernels.BANDWIDTHS:
                assert_allclose(kernels.bandwidth(tau, n, rule), bandwidth(tau, n, rule),
                                rtol=1e-11)
    with pytest.raises(KernelError):
        kernels.bandwidth(0.5, 100, "silverman")
    with pytest.raises(KernelError):
        kernels.kernel_values(grid, "uniform")


def stata_raw_quantile(values, p):
    """Order statistic int(p (N + 1)) (at least the first): qreg's "about" value."""
    ordered = np.sort(values)
    return ordered[min(max(int(np.floor(p * (len(ordered) + 1) + 1e-9)), 1), len(ordered)) - 1]


def test_empirical_quantiles_follow_stata_definitions():
    rng = np.random.default_rng(8)
    for n in (7, 10, 40, 101):
        values = rng.normal(size=n)
        tensor = torch.tensor(values)
        for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.3):
            assert_allclose(kernels.stata_quantile(tensor, p),
                            np.quantile(values, p, method="averaged_inverted_cdf"), rtol=1e-13)
            assert_allclose(kernels.raw_quantile(tensor, p), stata_raw_quantile(values, p),
                            rtol=1e-13)
        counts = rng.integers(1, 4, size=n)
        expanded = np.repeat(values, counts)
        weights = torch.tensor(counts, dtype=torch.float64)
        for p in (0.2, 0.5, 0.75):
            assert_allclose(kernels.stata_quantile(tensor, p, weights),
                            np.quantile(expanded, p, method="averaged_inverted_cdf"), rtol=1e-13)
            assert_allclose(kernels.raw_quantile(tensor, p, weights),
                            stata_raw_quantile(expanded, p), rtol=1e-13)


# ---- qreg: coefficients and header statistics ---------------------------------------------------


@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_qreg_coefficients_and_header(data, tau):
    x, y = design(data), data.y.to_numpy()
    beta, objective = lp_quantile(x, y, tau)
    result = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau)
    assert [c.term for c in result.coefficients] == ["Intercept", *X_COLS]
    assert_allclose(params(result), beta, rtol=1e-8, atol=1e-9)
    reference = sm.QuantReg(y, x).fit(q=tau, max_iter=5000, p_tol=1e-10)
    assert_allclose(params(result), reference.params, rtol=2e-4, atol=2e-4)
    raw = np.quantile(y, tau, method="inverted_cdf")
    raw_sum = np.where(y - raw < 0, (tau - 1) * (y - raw), tau * (y - raw)).sum()
    metrics = result.metrics
    assert metrics["quantile"] == tau
    assert_allclose(metrics["sum_adev"], objective, rtol=1e-10)
    assert_allclose(metrics["sum_rdev"], raw_sum, rtol=1e-12)
    assert_allclose(metrics["raw_quantile"], raw, rtol=1e-13)
    assert_allclose(metrics["pseudo_r_squared"], 1 - objective / raw_sum, rtol=1e-9)
    assert_allclose(metrics["pseudo_r_squared"], reference.prsquared, rtol=1e-6)
    assert metrics["df_resid"] == len(y) - 4 and metrics["df_model"] == 3
    assert result.nobs == len(y) and result.tests == {}
    assert result.title == ("Median regression" if tau == 0.5 else f"{tau:g} Quantile regression")
    assert result.inference["use_t"] and result.inference["df_inference"] == len(y) - 4
    # t inference with N - K degrees of freedom.
    coefficient = result.coefficients[1]
    assert_allclose(coefficient.p_value,
                    2 * stats.t.sf(abs(coefficient.statistic), len(y) - 4), rtol=1e-8)
    half = stats.t.ppf(0.975, len(y) - 4) * coefficient.std_error
    assert_allclose([coefficient.ci_low, coefficient.ci_high],
                    [coefficient.estimate - half, coefficient.estimate + half], rtol=1e-8)
    assert result.provenance["solver"] == "frisch_newton_interior_point"
    assert result.provenance["solver_diagnostics"]["exact_vertex"] is True
    assert result.provenance["stata_parity_validated"] is False


def test_qreg_engel_data_published_values():
    # Engel's food expenditure data (Koenker and Bassett 1982), shipped with statsmodels.
    # Reference values: R's quantreg (rq / summary.rq, as printed in its vignette).
    engel = sm.datasets.engel.load_pandas().data
    published = {0.05: (124.88004, 0.34336), 0.25: (95.48354, 0.47410), 0.5: (81.48225, 0.56018),
                 0.75: (62.39659, 0.64401), 0.95: (64.10396, 0.70907)}
    for tau, (intercept, slope) in published.items():
        result = oe.qreg(data=engel, y="foodexp", x=["income"], quantile=tau)
        assert_allclose(params(result), [intercept, slope], rtol=0, atol=6e-6)
        assert result.extra["unique_solution"]
    # summary(rq, se = "ker"): Powell sandwich with a Gaussian kernel. R scales the bandwidth
    # by IQR / 1.34 with interpolated quartiles (we follow Stata: 1.34, order statistics).
    robust = oe.qreg(data=engel, y="foodexp", x=["income"], covariance="robust",
                     kernel="gaussian")
    assert_allclose(errors(robust), [30.21532, 0.03732], rtol=5e-3)
    # summary(rq, se = "iid"): sparsity from residual quantiles (R drops the zero residuals and
    # fits a local slope; the plain difference quotient used here is close).
    residual = oe.qreg(data=engel, y="foodexp", x=["income"], density="residual")
    assert_allclose(errors(residual), [13.23908, 0.01192], rtol=0.03)
    # The Powell formula of summary.rq coded with R's own scale conventions reproduces R.
    y = engel.foodexp.to_numpy()
    x = np.column_stack([np.ones(len(y)), engel.income.to_numpy()])
    u = y - x @ params(robust)
    h = bandwidth(0.5, len(y), "hsheather")
    c = (stats.norm.ppf(0.5 + h) - stats.norm.ppf(0.5 - h)) * min(
        u.std(ddof=1), (np.quantile(u, 0.75) - np.quantile(u, 0.25)) / 1.34)
    d_inv = np.linalg.inv((x * (stats.norm.pdf(u / c) / c)[:, None]).T @ x)
    assert_allclose(np.sqrt(np.diag(0.25 * d_inv @ x.T @ x @ d_inv)), [30.21532, 0.03732],
                    rtol=1e-4)


# ---- qreg: covariance estimators -----------------------------------------------------------


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.8])
@pytest.mark.parametrize("density", ["fitted", "residual", "kernel"])
@pytest.mark.parametrize("rule", ["hsheather", "bofinger", "chamberlain"])
def test_qreg_iid_covariance(data, tau, density, rule):
    x, y = design(data), data.y.to_numpy()
    expected, sparsity, h = oracle_covariance(x, y, tau, "nonrobust", density=density, rule=rule)
    result = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau, density=density, bandwidth=rule)
    assert_allclose(covariance(result), expected, rtol=1e-7, atol=1e-12)
    assert_allclose(result.metrics["sparsity"], sparsity, rtol=1e-7)
    assert_allclose(result.metrics["density"], 1 / sparsity, rtol=1e-7)
    assert_allclose(result.metrics["bandwidth"], h, rtol=1e-11)
    assert result.extra["density_method"] == density and result.extra["bandwidth_method"] == rule
    assert density in result.inference["correction"] and rule in result.inference["correction"]
    assert (result.metrics["kernel_bandwidth"] is not None) == (density == "kernel")


def test_qreg_default_is_fitted_hall_sheather(data):
    default = oe.qreg(data=data, y="y", x=X_COLS)
    explicit = oe.qreg(data=data, y="y", x=X_COLS, covariance="nonrobust", density="fitted",
                       bandwidth="hsheather")
    assert default.spec.covariance == "nonrobust"
    assert default.extra["density_method"] == "fitted"
    assert_allclose(covariance(default), covariance(explicit), rtol=0, atol=0)
    # Giving a kernel with the i.i.d. covariance selects the kernel density (vce(iid, kernel())).
    implied = oe.qreg(data=data, y="y", x=X_COLS, kernel="parzen")
    assert implied.extra["density_method"] == "kernel" and implied.extra["kernel"] == "parzen"
    # The bandwidth does not depend on the confidence level of the reported intervals.
    other = oe.qreg(data=data, y="y", x=X_COLS, alpha=0.2)
    assert_allclose(covariance(other), covariance(default), rtol=0, atol=0)
    assert other.coefficients[1].ci_high < default.coefficients[1].ci_high


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_qreg_kernel_density_and_robust_covariance_per_kernel(data, kernel):
    x, y = design(data), data.y.to_numpy()
    tau = 0.4
    expected, sparsity, _ = oracle_covariance(x, y, tau, "nonrobust", density="kernel",
                                              kernel=kernel)
    iid = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau, density="kernel", kernel=kernel)
    assert_allclose(covariance(iid), expected, rtol=1e-7)
    assert_allclose(iid.metrics["sparsity"], sparsity, rtol=1e-7)
    expected, _, _ = oracle_covariance(x, y, tau, "robust", kernel=kernel)
    robust = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau, covariance="robust", kernel=kernel)
    assert_allclose(covariance(robust), expected, rtol=1e-7)
    assert robust.extra["density_method"] == "kernel" and robust.extra["kernel"] == kernel
    assert "Powell" in robust.inference["correction"]


@pytest.mark.parametrize("rule", ["hsheather", "bofinger", "chamberlain"])
def test_qreg_robust_and_cluster_covariance(data, rule):
    x, y = design(data), data.y.to_numpy()
    for tau in (0.3, 0.5):
        expected, _, _ = oracle_covariance(x, y, tau, "robust", rule=rule)
        robust = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau, covariance="robust",
                         density="kernel", bandwidth=rule)
        assert_allclose(covariance(robust), expected, rtol=1e-7)
        expected, _, _ = oracle_covariance(x, y, tau, "cluster", rule=rule,
                                           groups=data.g.to_numpy())
        cluster = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau, cluster="g", bandwidth=rule)
        assert cluster.spec.covariance == "cluster"
        assert_allclose(covariance(cluster), expected, rtol=1e-7)
        assert cluster.inference["cluster_count"] == data.g.nunique()
        assert cluster.inference["df_inference"] == len(y) - 4
        assert "Parente" in cluster.inference["correction"]
        assert_allclose(params(cluster), params(robust), rtol=0, atol=0)


def test_qreg_kernel_iid_is_close_to_statsmodels(data):
    # statsmodels' "iid" covariance is the kernel sparsity estimate with slightly different
    # scale conventions (np.std(y), IQR/1.34, interpolated quartiles): close, not identical.
    x, y = design(data), data.y.to_numpy()
    for tau in (0.25, 0.5):
        reference = sm.QuantReg(y, x).fit(q=tau, vcov="iid", kernel="epa", max_iter=5000,
                                          p_tol=1e-10)
        result = oe.qreg(data=data, y="y", x=X_COLS, quantile=tau, density="kernel", kernel="epan2")
        assert_allclose(errors(result), reference.bse, rtol=0.03)


# ---- qreg: weights -------------------------------------------------------------------------


@pytest.mark.parametrize("options", [
    {}, {"density": "residual"}, {"density": "kernel"}, {"covariance": "robust"},
    {"covariance": "robust", "kernel": "gaussian", "bandwidth": "bofinger"}, {"cluster": "g"},
])
def test_qreg_frequency_weights_equal_duplicated_rows(data, options):
    expanded = data.loc[data.index.repeat(data.f.astype(int))].reset_index(drop=True)
    weighted = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.4, weights="f",
                       weight_type="fweight", **options)
    plain = oe.qreg(data=expanded, y="y", x=X_COLS, quantile=0.4, **options)
    assert weighted.nobs == len(expanded) == plain.nobs
    # Duplicated rows make the vertex degenerate, not the solution ambiguous.
    assert plain.extra["unique_solution"] and weighted.extra["unique_solution"]
    assert not any("not unique" in warning for warning in plain.warnings)
    assert_allclose(params(weighted), params(plain), rtol=1e-8, atol=1e-10)
    assert_allclose(covariance(weighted), covariance(plain), rtol=1e-6, atol=1e-12)
    for name in ("sum_adev", "sum_rdev", "raw_quantile", "pseudo_r_squared", "sparsity"):
        assert_allclose(weighted.metrics[name], plain.metrics[name], rtol=1e-7)
    assert weighted.metrics["df_resid"] == len(expanded) - 4


def test_qreg_analytic_and_sampling_weights(data):
    x, y, w = design(data), data.y.to_numpy(), data.w.to_numpy()
    beta, objective = lp_quantile(x, y, 0.6, w)
    aweight = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.6, weights="w", weight_type="aweight")
    assert_allclose(params(aweight), beta, rtol=1e-8, atol=1e-9)
    scaled = len(y) / w.sum()
    assert_allclose(aweight.metrics["sum_adev"], objective * scaled, rtol=1e-9)
    # The scale of analytic weights is irrelevant.
    rescaled = oe.qreg(data=data.assign(w=data.w * 37.5), y="y", x=X_COLS, quantile=0.6,
                       weights="w", weight_type="aweight")
    assert_allclose(covariance(rescaled), covariance(aweight), rtol=1e-8)
    assert_allclose(params(rescaled), params(aweight), rtol=1e-9)
    # i.i.d. covariance with the residual sparsity: weighted quantiles and (X'WX)^-1.
    wn = w * scaled
    resid = y - x @ beta
    resid[np.abs(resid) < 1e-9] = 0
    h = bandwidth(0.6, len(y), "hsheather")
    order = np.argsort(resid)
    cumulative = np.cumsum(wn[order])

    def weighted_quantile(p):
        return resid[order][np.searchsorted(cumulative, p * cumulative[-1], side="right")]

    sparsity = (weighted_quantile(0.6 + h) - weighted_quantile(0.6 - h)) / (2 * h)
    residual = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.6, weights="w",
                       weight_type="aweight", density="residual")
    assert_allclose(covariance(residual),
                    0.24 * sparsity ** 2 * np.linalg.inv((x * wn[:, None]).T @ x), rtol=1e-7)
    # pweights: robust by default, squared weights in the meat (kernel density here).
    pweight = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.6, weights="w", weight_type="pweight",
                      density="kernel")
    assert pweight.spec.covariance == "robust"
    assert_allclose(params(pweight), beta, rtol=1e-8, atol=1e-9)
    mean = (wn * resid).sum() / len(y)
    sd = np.sqrt((wn * (resid - mean) ** 2).sum() / (len(y) - 1))
    iqr = weighted_quantile(0.75) - weighted_quantile(0.25)
    c = min(sd, iqr / 1.34) * (stats.norm.ppf(0.6 + h) - stats.norm.ppf(0.6 - h))
    d_inv = np.linalg.inv((x * (wn * KERNELS["epanechnikov"](resid / c) / c)[:, None]).T @ x)
    expected = d_inv @ (0.24 * (x * (wn ** 2)[:, None]).T @ x) @ d_inv
    assert_allclose(covariance(pweight), expected, rtol=1e-7)
    robust_aweight = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.6, weights="w",
                             weight_type="aweight", covariance="robust", density="kernel")
    assert_allclose(covariance(robust_aweight), expected, rtol=1e-9)
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=data, y="y", x=X_COLS, weights="w", weight_type="pweight",
                covariance="nonrobust")
    assert caught.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=data, y="y", x=X_COLS, weights="w", weight_type="iweight")
    assert caught.value.code == "invalid_spec"


def test_qreg_zero_and_invalid_weights(data):
    frame = data.assign(f=np.where(np.arange(len(data)) % 5 == 0, 0.0, data.f))
    result = oe.qreg(data=frame, y="y", x=X_COLS, weights="f", weight_type="fweight")
    kept = frame[frame.f > 0]
    reference = oe.qreg(data=kept.reset_index(drop=True), y="y", x=X_COLS, weights="f",
                        weight_type="fweight")
    assert result.nobs == int(kept.f.sum()) and result.dropped_rows == len(frame) - len(kept)
    assert result.sample_positions == kept.index.tolist()
    assert any("zero weight" in warning for warning in result.warnings)
    assert_allclose(params(result), params(reference), rtol=0, atol=0)
    assert_allclose(covariance(result), covariance(reference), rtol=0, atol=0)
    cases = ((-data.f, "negative_weights"), (data.f + 0.5, "noninteger_frequency_weights"))
    for column, code in cases:
        with pytest.raises(AnalysisError) as caught:
            oe.qreg(data=data.assign(f=column), y="y", x=X_COLS, weights="f",
                    weight_type="fweight")
        assert caught.value.code == code


# ---- qreg: design handling, failures, rendering --------------------------------------------


def test_qreg_categorical_intercept_and_no_regressors(data):
    result = oe.qreg(data=data, y="y", x=["x1", "sector"], categorical=["sector"], quantile=0.3)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "sector[b]", "sector[c]"]
    dummies = np.column_stack([np.ones(len(data)), data.x1, data.sector == "b", data.sector == "c"]
                              ).astype(float)
    assert_allclose(params(result), lp_quantile(dummies, data.y.to_numpy(), 0.3)[0], rtol=1e-8)
    without = oe.qreg(data=data, y="y", x=["x1", "x2"], intercept=False, quantile=0.3)
    assert [c.term for c in without.coefficients] == ["x1", "x2"]
    expected = lp_quantile(data[["x1", "x2"]].to_numpy(), data.y.to_numpy(), 0.3)[0]
    assert_allclose(params(without), expected, rtol=1e-8)
    assert without.metrics["df_model"] == 2
    # No regressors: the coefficient is the raw quantile of y.
    constant = oe.qreg(data=data, y="y", quantile=0.3, density="residual")
    y = np.sort(data.y.to_numpy())
    estimate = constant.coefficients[0].estimate
    assert y[int(np.ceil(0.3 * len(y))) - 1] <= estimate <= y[int(0.3 * len(y))]
    assert_allclose(constant.metrics["pseudo_r_squared"], 0, atol=1e-12)
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=data, y="y", intercept=False)
    assert caught.value.code == "no_regressors"


def test_qreg_collinear_terms_are_omitted(data):
    frame = data.assign(twice=2 * data.x1, constant=3.0)
    result = oe.qreg(data=frame, y="y", x=["x1", "twice", "x2", "constant"], quantile=0.5)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert result.provenance["omitted_terms"] == ["twice", "constant"]
    assert any("collinearity" in warning for warning in result.warnings)
    reference = oe.qreg(data=data, y="y", x=["x1", "x2"], quantile=0.5)
    assert_allclose(params(result), params(reference), rtol=1e-9)
    assert_allclose(covariance(result), covariance(reference), rtol=1e-8)


def test_qreg_missing_data_policy(data):
    frame = data.copy()
    frame.loc[[3, 17, 40], "x2"] = np.nan
    frame.loc[5, "y"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=frame, y="y", x=X_COLS)
    assert caught.value.code == "missing_values"
    dropped = oe.qreg(data=frame, y="y", x=X_COLS, missing="drop")
    complete = frame.dropna(subset=["y", *X_COLS])
    reference = oe.qreg(data=complete.reset_index(drop=True), y="y", x=X_COLS)
    assert dropped.nobs == len(complete) and dropped.dropped_rows == 4
    assert_allclose(params(dropped), params(reference), rtol=1e-10)
    assert dropped.sample_positions == complete.index.tolist()


def test_qreg_nonunique_solution_is_flagged():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"d": np.repeat([0.0, 1.0], 20)})
    frame["y"] = rng.normal(size=40) + frame.d
    result = oe.qreg(data=frame, y="y", x=["d"], density="residual")
    assert result.extra["unique_solution"] is False
    assert any("not unique" in warning for warning in result.warnings)
    # Any optimal vertex attains the linear-programming minimum.
    x = np.column_stack([np.ones(40), frame.d])
    assert_allclose(result.metrics["sum_adev"], lp_quantile(x, frame.y.to_numpy(), 0.5)[1],
                    rtol=1e-10)


def test_qreg_error_codes(data):
    for quantile in (0, 1, 1.5, -0.2, 50):
        with pytest.raises(AnalysisError) as caught:
            oe.qreg(data=data, y="y", x=X_COLS, quantile=quantile)
        assert caught.value.code == "invalid_quantile"
    cases = [
        ({"kernel": "uniform"}, "invalid_spec"),
        ({"bandwidth": "silverman"}, "invalid_spec"),
        ({"density": "histogram"}, "invalid_spec"),
        ({"covariance": "HC1"}, "invalid_spec"),
        ({"covariance": "robust", "density": "residual"}, "invalid_spec"),
        ({"density": "residual", "kernel": "gaussian"}, "invalid_spec"),
        ({"covariance": "cluster"}, "invalid_spec"),
        ({"x": "x1"}, "invalid_spec"),
        ({"x": ["x1", "nope"]}, "missing_columns"),
    ]
    for options, code in cases:
        arguments = {"data": data, "y": "y", "x": X_COLS, **options}
        with pytest.raises(AnalysisError) as caught:
            oe.qreg(**arguments)
        assert caught.value.code == code, options
    small = data.iloc[:30]
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=small, y="y", x=X_COLS, quantile=0.02)
    assert caught.value.code == "bandwidth_out_of_range"
    assert "bsqreg" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=data.iloc[:4], y="y", x=X_COLS)
    assert caught.value.code == "insufficient_observations"
    exact = data.assign(y=1 + 2 * data.x1)
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=exact, y="y", x=["x1", "x2"])
    assert caught.value.code == "perfect_fit"
    # A heavily discrete outcome: zero residual spread in the density window.
    discrete = pd.DataFrame({"x": np.arange(60.0), "y": np.r_[np.zeros(50), np.ones(10)]})
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=discrete, y="y", x=["x"], density="residual")
    assert caught.value.code in {"degenerate_sparsity", "perfect_fit"}
    with pytest.raises(ValidationError):
        ModelSpec(estimator="qreg", outcome="y", predictors=X_COLS, options={"kernel": "uniform"})
    with pytest.raises(AnalysisError) as caught:
        oe.qreg(data=data.assign(g=1), y="y", x=X_COLS, cluster="g")
    assert caught.value.code == "insufficient_clusters"


def test_qreg_round_trip_rendering_and_public_api(data):
    result = oe.qreg(data=data, y="y", x=X_COLS, quantile=0.25, covariance="robust")
    assert isinstance(result, ResultBundle)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    json.dumps(result.model_dump())
    summary = result.summary()
    assert "0.25 Quantile regression" in summary and "pseudo_r_squared" in summary
    assert "sparsity" in summary and "x2" in summary
    latex = result.to_latex()
    assert "x1" in latex and "tabular" in latex
    assert len(result.predictions) == len(data) and result.predictions[0]["row"] == 0
    spec = ModelSpec(estimator="qreg", outcome="y", predictors=X_COLS, covariance="robust",
                     options={"quantile": 0.25})
    again = oe.fit(spec, data=data)
    assert_allclose(params(again), params(result), rtol=0, atol=0)
    assert_allclose(covariance(again), covariance(result), rtol=0, atol=0)
    listing = oe.capabilities()["estimators"]["qreg"]
    assert listing["family"] == "quantile" and listing["stata"] == ["qreg"]
    assert listing["covariances"] == ["nonrobust", "robust", "cluster"]
    assert listing["weights"] == ["aweight", "fweight", "pweight"]
    assert listing["inference"] == "Student t"
    for name in ("qreg", "bsqreg", "sqreg", "iqreg"):
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
