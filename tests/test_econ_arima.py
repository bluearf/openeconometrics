"""Independent oracles for the ARIMA estimator (oe.arima).

The exact likelihood is checked against (i) the multivariate normal density of
the series with its dense n-by-n ARMA covariance matrix (autocovariances from
the MA(infinity) weights, Cholesky factorization in NumPy), which shares no
code or algebra with the closed-form kernel, and (ii) statsmodels' Kalman
filter. Estimates are compared with a brute-force SciPy maximization of the
dense likelihood and with statsmodels SARIMAX; covariances with numerical
Hessians and numerical per-observation scores of the dense likelihood. The
conditional estimator is checked against an explicit residual recursion.
"""

import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats
from scipy.linalg import cholesky, solve_discrete_lyapunov, solve_triangular, toeplitz
from scipy.optimize import minimize
from scipy.signal import lfilter
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tsa.statespace.sarimax import SARIMAX

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.arima import filters
from openecon.econometrics.arima.kernels import ArmaLikelihood, starting_values
from openecon.engines.optimize import check_derivatives, numerical_hessian
from openecon.models import ModelSpec


# ---- independent oracles --------------------------------------------------------------------


def expand(phi, theta, sphi=(), stheta=(), s=1):
    """Expanded AR and MA lag polynomials (increasing powers)."""
    ar, ma = np.r_[1.0, -np.asarray(phi, float)], np.r_[1.0, np.asarray(theta, float)]
    sar, sma = np.zeros(len(sphi) * s + 1), np.zeros(len(stheta) * s + 1)
    sar[0] = sma[0] = 1.0
    for j, value in enumerate(sphi, start=1):
        sar[j * s] = -value
    for j, value in enumerate(stheta, start=1):
        sma[j * s] = value
    return np.convolve(ar, sar), np.convolve(ma, sma)


def dense_loglike(u, ar, ma, sigma, terms=6000):
    """Per-observation exact Gaussian log likelihood from the dense covariance matrix."""
    n = len(u)
    impulse = np.zeros(terms)
    impulse[0] = 1.0
    psi = lfilter(ma, ar, impulse)
    gamma = sigma ** 2 * np.array([psi[:terms - k] @ psi[k:] if k < terms else 0.0
                                   for k in range(n)])
    lower = cholesky(toeplitz(gamma), lower=True)
    e = solve_triangular(lower, u, lower=True)
    diagonal = np.diag(lower)
    contributions = -0.5 * math.log(2 * math.pi) - np.log(diagonal) - 0.5 * e ** 2
    return contributions, e * diagonal, diagonal ** 2


def css_residuals(u, ar, ma):
    """Conditional residuals by the explicit recursion with zero presample values."""
    e = np.zeros(len(u))
    for t in range(len(u)):
        value = sum(ar[i] * u[t - i] for i in range(min(t, len(ar) - 1) + 1))
        value -= sum(ma[j] * e[t - j] for j in range(1, min(t, len(ma) - 1) + 1))
        e[t] = value
    return e


def simulate(n, phi=(), theta=(), seed=0, burn=300):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + burn)
    return lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi)], e)[burn:]


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


@pytest.fixture(scope="module")
def armax():
    n = 140
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n).cumsum() / 4,
                          "t": np.arange(1990, 1990 + n)})
    frame["y"] = 1.5 + 0.8 * frame["x1"] - 0.5 * frame["x2"] + simulate(n, [0.6], [0.35], seed=3)
    return frame


@pytest.fixture(scope="module")
def armax_fits(armax):
    return {kind: oe.arima(data=armax, y="y", x=["x1", "x2"], order=(1, 0, 1), time="t",
                           covariance=kind) for kind in ("opg", "nonrobust", "robust")}


def armax_dense(params, frame):
    x = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    ar, ma = expand([params[3]], [params[4]])
    return dense_loglike(frame["y"].to_numpy() - x @ params[:3], ar, ma, params[5])[0]


# ---- filters ---------------------------------------------------------------------------------


def test_filters_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 500))
    for coefficients, unit in [([0.5], 1), ([-0.9, 0.3], 1), ([0.6], 12), ([-1.0], 1),
                               ([0.3, -0.2, 0.5, 0.1], 1), ([-0.5, 0.2], 4)]:
        denominator = np.zeros(len(coefficients) * unit + 1)
        denominator[0] = 1.0
        denominator[unit::unit] = coefficients
        out = filters.inverse_filter(torch.from_numpy(x), coefficients, unit).numpy()
        assert_allclose(out, lfilter([1.0], denominator, x, axis=-1), rtol=1e-10, atol=1e-10)
        applied = filters.apply_polynomial(torch.from_numpy(x), list(denominator)).numpy()
        assert_allclose(applied, lfilter(denominator, [1.0], x, axis=-1), rtol=1e-12, atol=1e-12)
    transition = np.array([[0.5, 1.0, 0.0], [0.2, 0.0, 1.0], [-0.1, 0.0, 0.0]])
    q = np.outer([1.0, 0.4, 0.2], [1.0, 0.4, 0.2])
    powers = filters.transition_powers(torch.from_numpy(transition))
    assert_allclose(filters.lyapunov(powers, torch.from_numpy(q)).numpy(),
                    solve_discrete_lyapunov(transition, q), rtol=1e-12)
    assert filters.transition_powers(torch.tensor([[1.01]], dtype=torch.float64)) is None
    roots = filters.inverse_roots([0.5, 0.2, -0.1]).numpy()
    expected = np.roots([1.0, -0.5, -0.2, 0.1])
    assert_allclose(np.sort(np.abs(roots)), np.sort(np.abs(expected)), rtol=1e-12)
    series = torch.from_numpy(rng.normal(size=60))
    assert_allclose(filters.difference(series, 1, 1, 4).numpy(),
                    np.diff(series.numpy()[4:] - series.numpy()[:-4]), atol=1e-14)
    assert filters.difference_polynomial(1, 1, 4) == [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    assert filters.shift(torch.arange(4.0, dtype=torch.float64), 2).tolist() == [0, 0, 0, 1]


def test_inverse_filter_stays_accurate_when_roots_meet_under_squaring():
    """Even, nearly even and seasonal-type dense polynomials with roots at the unit circle.

    Cyclic reduction divides by c(L) c(-L); roots rho and -rho (or roots that meet after
    repeated squaring) would make that polynomial's roots multiple and lose most digits.
    Folding even polynomials, the residual check with iterative refinement and the
    companion-form fallback keep the result at the accuracy of the plain recursion.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 60_000))
    x[1] *= 1e6                                         # rows of very different scale
    hw = [-0.95] + [0.0] * 10 + [-0.9, 0.855]           # (1 - .95 L)(1 - .9 L^12)
    dense = [0.05 + 5e-6 - 1] + [5e-6] * 10 + [5e-6 + 0.0475 - 1, 1 - 0.05 - 0.0475]
    for coefficients in ([0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, -0.98], [1e-7, 0.0, 0.0, -1.0],
                         [0.0, -1.9, 0.0, 0.9025], [0.0, -1.0], [1e-3, -0.98], hw, dense):
        expected = lfilter([1.0], np.r_[1.0, coefficients], x, axis=-1)
        out = filters.inverse_filter(torch.from_numpy(x), coefficients).numpy()
        scale = np.abs(expected).max(axis=1, keepdims=True)
        assert float((np.abs(out - expected) / scale).max()) < 1e-10, coefficients
        direct = filters._companion_filter(torch.from_numpy(x), [1.0, *coefficients], 1).numpy()
        assert float((np.abs(direct - expected) / scale).max()) < 1e-10, coefficients
    # The same polynomial with a lag unit: (1 - 0.9 L^12) is a first-order filter in L^12.
    strided = filters.inverse_filter(torch.from_numpy(x), [-0.9], 12).numpy()
    assert_allclose(strided, lfilter([1.0], np.r_[1.0, np.zeros(11), -0.9], x, axis=-1),
                    rtol=1e-11, atol=1e-11)
    assert filters._fold([1.0, 0.0, 0.0, 0.0, -1.0], 1) == ([1.0, -1.0], 4)
    assert filters._fold([1.0, 0.5, 0.0], 3) == ([1.0, 0.5], 3)
    explosive = filters.inverse_filter(torch.from_numpy(x[:1, :4000]), [-3.0, 1.5])
    assert not bool(torch.isfinite(explosive).all())    # callers treat this as infeasible


# ---- the likelihood kernel ------------------------------------------------------------------

CASES = [  # p, q, P, Q, s, ARMA parameters
    (1, 0, 0, 0, 0, [0.6]),
    (0, 1, 0, 0, 0, [0.6]),
    (1, 1, 0, 0, 0, [0.6, 0.4]),
    (2, 2, 0, 0, 0, [0.5, -0.3, 0.4, 0.2]),
    (1, 1, 1, 1, 12, [0.5, 0.3, 0.4, -0.5]),
    (0, 1, 0, 1, 4, [-0.4, -0.6]),
    (2, 0, 1, 0, 4, [0.4, 0.2, 0.5]),
    (0, 0, 0, 0, 0, []),
]


def kernel_case(p, q, sp, sq, s, arma, n=90, exact=True):
    rng = np.random.default_rng(100 + p + 3 * q + 5 * sp + 7 * sq)
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = simulate(n, [0.5], [0.3], seed=p + q) + x @ np.array([1.0, 2.0])
    like = ArmaLikelihood(torch.from_numpy(y), torch.from_numpy(x), p=p, q=q, seasonal_p=sp,
                          seasonal_q=sq, period=s, exact=exact)
    theta = np.r_[1.1, 1.9, arma, 1.3]
    return like, theta, y, x


@pytest.mark.parametrize("case", CASES)
def test_exact_likelihood_equals_dense_gaussian_density(case):
    p, q, sp, sq, s, arma = case
    like, theta, y, x = kernel_case(*case)
    ar, ma = expand(arma[:p], arma[p:p + q], arma[p + q:p + q + sp], arma[p + q + sp:], max(s, 1))
    contributions, innovations, variances = dense_loglike(y - x @ theta[:2], ar, ma, theta[-1])
    value, _ = like.full(torch.from_numpy(theta))
    assert value == pytest.approx(contributions.sum(), rel=1e-11)
    decomposition = like.decompose(torch.from_numpy(theta), scores=False)
    assert_allclose(decomposition.innovations.numpy(), innovations, rtol=1e-8, atol=1e-10)
    assert_allclose(decomposition.variance_ratio.numpy(), variances / theta[-1] ** 2, rtol=1e-9)


@pytest.mark.parametrize("case", CASES[:6])
def test_likelihoods_match_statsmodels_kalman_filter(case):
    p, q, sp, sq, s, arma = case
    like, theta, y, x = kernel_case(*case, n=150)
    model = SARIMAX(y, exog=x, order=(p, 0, q), seasonal_order=(sp, 0, sq, s))
    params = np.r_[theta[:-1], theta[-1] ** 2]
    assert like.full(torch.from_numpy(theta))[0] == pytest.approx(model.loglike(params), rel=1e-11)
    filtered = model.filter(params)
    decomposition = like.decompose(torch.from_numpy(theta), scores=False)
    assert_allclose(decomposition.innovations.numpy(), filtered.forecasts_error[0], atol=1e-10)
    # Conditional likelihood: presample state at zero, i.e. state covariance sigma^2 R R'.
    conditional, _, _, _ = kernel_case(*case, n=150, exact=False)
    loading = model.ssm["selection"][:, 0]
    model.ssm.initialize_known(np.zeros(model.k_states),
                               theta[-1] ** 2 * np.outer(loading, loading))
    assert conditional.full(torch.from_numpy(theta))[0] == pytest.approx(model.loglike(params),
                                                                        rel=1e-11)
    ar, ma = expand(arma[:p], arma[p:p + q], arma[p + q:p + q + sp], arma[p + q + sp:], max(s, 1))
    residuals = css_residuals(y - x @ theta[:2], ar, ma)
    assert_allclose(conditional.decompose(torch.from_numpy(theta)).innovations.numpy(), residuals,
                    atol=1e-10)


def test_marginally_noninvertible_and_infeasible_points():
    like, theta, y, x = kernel_case(0, 1, 0, 0, 0, [-1.0], n=80)
    for value in (-1.0, -1.01):          # unit root and a marginally non-invertible MA(1)
        theta[2] = value
        ar, ma = expand([], [value])
        dense = dense_loglike(y - x @ theta[:2], ar, ma, theta[-1], terms=3)[0].sum()
        assert like.full(torch.from_numpy(theta))[0] == pytest.approx(dense, rel=1e-9)
    theta[2] = -1.5                      # clearly non-invertible: rejected, not evaluated
    value, gradient = like.full(torch.from_numpy(theta))
    assert value == -math.inf and bool(torch.isnan(gradient).all())
    ar_like, ar_theta, _, _ = kernel_case(1, 0, 0, 0, 0, [1.0])
    assert ar_like.full(torch.from_numpy(ar_theta))[0] == -math.inf      # non-stationary AR
    assert ar_like.concentrated(torch.from_numpy(ar_theta[:-1]))[0] == -math.inf
    css, css_theta, _, _ = kernel_case(1, 0, 0, 0, 0, [1.0], exact=False)
    assert math.isfinite(css.full(torch.from_numpy(css_theta))[0])       # CSS allows a unit root


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("case", CASES[:7])
def test_analytic_derivatives_and_scores(case, exact):
    like, theta, _, _ = kernel_case(*case, exact=exact)
    point = torch.from_numpy(theta)
    full = check_derivatives(like.full, point)
    concentrated = check_derivatives(like.concentrated, point[:-1])
    assert full["gradient_max_rel_error"] < 1e-8
    assert concentrated["gradient_max_rel_error"] < 1e-8
    decomposition = like.decompose(point)
    value, gradient = like.full(point)
    assert_allclose(decomposition.scores.sum(dim=0).numpy(), gradient.numpy(), rtol=1e-8, atol=1e-8)
    sigma = theta[-1]
    v, f = decomposition.innovations.numpy(), decomposition.variance_ratio.numpy()
    per_observation = -0.5 * np.log(2 * np.pi * sigma ** 2 * f) - v ** 2 / (2 * sigma ** 2 * f)
    assert per_observation.sum() == pytest.approx(value, rel=1e-11)

    # Per-observation scores against numerical derivatives of the log-likelihood contributions.
    def contributions(trial):
        d = like.decompose(torch.from_numpy(trial), scores=False)
        vv, ff = d.innovations.numpy(), d.variance_ratio.numpy()
        return -0.5 * np.log(2 * np.pi * trial[-1] ** 2 * ff) - vv ** 2 / (2 * trial[-1] ** 2 * ff)

    numerical = approx_fprime(theta, contributions, centered=True)
    assert_allclose(decomposition.scores.numpy(), numerical, rtol=2e-6, atol=2e-6)


def test_full_hessian_from_concentrated_hessian():
    like, theta, y, x = kernel_case(1, 1, 0, 0, 0, [0.6, 0.4], n=150)
    start = starting_values(torch.from_numpy(y), torch.from_numpy(x), like.sizes, like.period)
    from openecon.engines.optimize import maximize_bfgs

    best = maximize_bfgs(like.concentrated, start)
    pieces = like.evaluate(best.theta)
    sigma = math.sqrt(pieces.ss / like.n)
    point = torch.cat([best.theta, torch.tensor([sigma], dtype=torch.float64)])
    assembled = like.full_hessian(best.theta, best.hessian)
    direct = numerical_hessian(lambda trial: like.full(trial)[1], point)
    assert_allclose(assembled.numpy(), direct.numpy(), rtol=1e-6, atol=1e-6)
    assert float(like.full(point)[1].abs().max()) < 1e-6


def test_starting_values_are_feasible_and_close():
    y = simulate(4000, [0.5, -0.2], [0.4], seed=5)
    x = np.ones((4000, 1))
    start = starting_values(torch.from_numpy(y), torch.from_numpy(x), (2, 1, 0, 0), 0).numpy()
    assert_allclose(start[1:], [0.5, -0.2, 0.4], atol=0.08)
    explosive = np.cumsum(np.cumsum(simulate(300, seed=6)))
    start = starting_values(torch.from_numpy(explosive), torch.from_numpy(np.ones((300, 1))),
                            (2, 1, 0, 0), 0).tolist()
    assert filters.spectral_radius(start[1:3]) <= 0.95 + 1e-12


# ---- estimates against brute force and statsmodels ------------------------------------------


def test_exact_ml_matches_bruteforce_dense_likelihood(armax, armax_fits):
    fit = armax_fits["nonrobust"]
    objective = lambda params: -armax_dense(params, armax).sum()  # noqa: E731
    best = minimize(objective, estimates(fit) + 0.02, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 6000, "maxfev": 6000})
    assert_allclose(estimates(fit), best.x, rtol=2e-5, atol=2e-6)
    assert fit.metrics["log_likelihood"] == pytest.approx(-best.fun, abs=1e-8)
    assert fit.metrics["sigma"] == pytest.approx(best.x[-1], rel=1e-5)
    assert [c.term for c in fit.coefficients] == ["Intercept", "x1", "x2", "ARMA:L1.ar",
                                                  "ARMA:L1.ma", "/sigma"]
    assert [c.equation for c in fit.coefficients] == ["y", "y", "y", "ARMA", "ARMA", None]
    # Observed information: numerical Hessian of the independent dense likelihood.
    hessian = approx_hess(estimates(fit), lambda params: armax_dense(params, armax).sum())
    assert_allclose(np.array(fit.covariance_matrix), np.linalg.inv(-hessian), rtol=2e-4, atol=1e-8)
    # OPG and robust: numerical per-observation scores of the dense likelihood.
    scores = approx_fprime(estimates(fit), lambda params: armax_dense(params, armax), centered=True)
    opg = np.linalg.inv(scores.T @ scores)
    assert_allclose(np.array(armax_fits["opg"].covariance_matrix), opg, rtol=2e-5, atol=1e-9)
    n = len(armax)
    bread = np.linalg.inv(-hessian)
    sandwich = n / (n - 1) * bread @ (scores.T @ scores) @ bread
    assert_allclose(np.array(armax_fits["robust"].covariance_matrix), sandwich, rtol=5e-4,
                    atol=1e-8)
    assert armax_fits["opg"].spec.covariance == "opg"
    assert "N/(N-1)" in armax_fits["robust"].inference["correction"]
    assert armax_fits["robust"].inference["small_sample_correction"] == pytest.approx(n / (n - 1))


def test_default_covariance_is_opg_and_matches_statsmodels(armax, armax_fits):
    default = oe.arima(data=armax, y="y", x=["x1", "x2"], order=(1, 0, 1), time="t")
    assert default.spec.covariance == "opg"
    assert default.inference["use_t"] is False and default.inference["distribution"] == "normal"
    x = np.column_stack([np.ones(len(armax)), armax["x1"], armax["x2"]])
    model = SARIMAX(armax["y"].to_numpy(), exog=x, order=(1, 0, 1))
    params = estimates(default)
    params[-1] **= 2
    assert default.metrics["log_likelihood"] == pytest.approx(model.loglike(params), rel=1e-11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False, maxiter=500)
        assert_allclose(estimates(default)[:5], fitted.params[:5], rtol=2e-3, atol=2e-4)
        assert default.metrics["log_likelihood"] >= fitted.llf - 1e-9
        opg = model.smooth(params, cov_type="opg")
        approx = model.smooth(params, cov_type="approx")
    sigma = estimates(default)[-1]
    assert_allclose(errors(default)[:5], opg.bse[:5], rtol=1e-4)
    assert errors(default)[5] == pytest.approx(opg.bse[5] / (2 * sigma), rel=1e-4)   # delta method
    assert_allclose(errors(armax_fits["nonrobust"])[:5], approx.bse[:5], rtol=1e-4)
    z = stats.norm.ppf(0.975)
    for c in default.coefficients[:5]:
        assert c.statistic == pytest.approx(c.estimate / c.std_error)
        assert c.p_value == pytest.approx(2 * stats.norm.sf(abs(c.statistic)), rel=1e-9)
        assert (c.ci_low, c.ci_high) == pytest.approx((c.estimate - z * c.std_error,
                                                       c.estimate + z * c.std_error))
    sigma_row = default.coefficients[-1]             # Stata: one-sided test, interval cut at zero
    assert sigma_row.p_value == pytest.approx(stats.norm.sf(sigma_row.statistic), rel=1e-9)
    assert sigma_row.ci_low >= 0.0


def test_differencing_regressors_and_drift():
    n = 160
    rng = np.random.default_rng(31)
    x1 = rng.normal(size=n).cumsum()
    y = np.cumsum(0.3 + simulate(n, [0.4], [-0.5], seed=8)) + 0.8 * x1
    frame = pd.DataFrame({"y": y, "x1": x1})
    fit = oe.arima(data=frame, y="y", x=["x1"], order=(1, 1, 1), covariance="nonrobust")
    assert fit.nobs == n - 1 and fit.metrics["n_differenced"] == n - 1
    assert fit.nobs_original == n and fit.dropped_rows == 1
    assert fit.sample_positions == list(range(1, n))
    assert [c.equation for c in fit.coefficients[:2]] == ["D.y", "D.y"]
    assert any("Differencing uses the first 1 observation" in w for w in fit.warnings)
    x = np.column_stack([np.ones(n - 1), np.diff(x1)])

    def negative(params):
        ar, ma = expand([params[2]], [params[3]])
        return -dense_loglike(np.diff(y) - x @ params[:2], ar, ma, params[4])[0].sum()

    best = minimize(negative, estimates(fit) + 0.01, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 5000, "maxfev": 5000})
    assert_allclose(estimates(fit), best.x, rtol=5e-5, atol=5e-6)
    assert fit.metrics["log_likelihood"] == pytest.approx(-best.fun, abs=1e-8)
    hessian = approx_hess(estimates(fit), lambda params: -negative(params))
    assert_allclose(errors(fit), np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=1e-4)
    # Fitted values are one-step predictions of the level; residuals are the innovations.
    ar, ma = expand([estimates(fit)[2]], [estimates(fit)[3]])
    innovations = dense_loglike(np.diff(y) - x @ estimates(fit)[:2], ar, ma, estimates(fit)[4])[1]
    rows = [row["row"] for row in fit.predictions]
    assert rows == sorted(rows) and rows[0] == 1 and rows[-1] == n - 1
    for row in fit.predictions[:5] + fit.predictions[-5:]:
        assert row["observed"] == pytest.approx(y[row["row"]])
        assert row["residual"] == pytest.approx(innovations[row["row"] - 1], abs=1e-7)
    second = oe.arima(data=frame, y="y", order=(0, 2, 1))
    assert second.coefficients[0].equation == "D2.y" and second.nobs == n - 2


def test_seasonal_airline_model_and_multiplicative_ar():
    rng = np.random.default_rng(41)
    n, s = 180, 12
    w = lfilter(np.convolve([1, -0.4], np.r_[1, np.zeros(11), -0.6]), [1.0], rng.normal(size=n))
    y = lfilter([1.0], np.convolve([1, -1], np.r_[1, np.zeros(11), -1]), w)
    frame = pd.DataFrame({"y": y})
    fit = oe.arima(data=frame, y="y", order=(0, 1, 1), seasonal=(0, 1, 1), period=s,
                   constant=False, covariance="nonrobust")
    assert [c.term for c in fit.coefficients] == ["ARMA:L1.ma", "ARMA12:L1.ma", "/sigma"]
    assert [c.equation for c in fit.coefficients] == ["ARMA", "ARMA12", None]
    assert fit.nobs == n - 13 and fit.title == "ARIMA(0,1,1)x(0,1,1)[12] regression"
    assert fit.extra["model_order"] == {"p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1,
                                        "period": 12}
    differenced = np.diff(y[s:] - y[:-s])

    def negative(params):
        ar, ma = expand([], [params[0]], [], [params[1]], s)
        return -dense_loglike(differenced, ar, ma, params[2], terms=40)[0].sum()

    best = minimize(negative, estimates(fit) + 0.01, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
    assert_allclose(estimates(fit), best.x, rtol=5e-5, atol=5e-6)
    hessian = approx_hess(estimates(fit), lambda params: -negative(params))
    assert_allclose(errors(fit), np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=2e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, s),
                            simple_differencing=True)
        params = estimates(fit)
        params[-1] **= 2
        assert fit.metrics["log_likelihood"] == pytest.approx(reference.loglike(params), rel=1e-10)

    quarterly = simulate(240, seed=43)
    ar_full = np.convolve([1, -0.5], [1, 0, 0, 0, -0.6])
    z = lfilter([1.0], ar_full, np.r_[np.random.default_rng(44).normal(size=200), quarterly])[200:]
    sar = oe.arima(data=pd.DataFrame({"y": z}), y="y", order=(1, 0, 0), seasonal=(1, 0, 0),
                   period=4, covariance="nonrobust")
    assert [c.term for c in sar.coefficients] == ["Intercept", "ARMA:L1.ar", "ARMA4:L1.ar",
                                                  "/sigma"]

    def negative_sar(params):
        ar, ma = expand([params[1]], [], [params[2]], [], 4)
        return -dense_loglike(z - params[0], ar, ma, params[3])[0].sum()

    best = minimize(negative_sar, estimates(sar) + 0.01, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
    assert_allclose(estimates(sar), best.x, rtol=5e-5, atol=5e-6)
    assert sar.extra["seasonal"] == {"ar": [pytest.approx(best.x[2], rel=1e-4)], "ma": [],
                                     "period": 4}


def test_conditional_sum_of_squares_matches_explicit_recursion(armax):
    fit = oe.arima(data=armax, y="y", x=["x1", "x2"], order=(1, 0, 1), time="t", method="css",
                   covariance="nonrobust")
    x = np.column_stack([np.ones(len(armax)), armax["x1"], armax["x2"]])
    y, n = armax["y"].to_numpy(), len(armax)

    def loglike(params):
        e = css_residuals(y - x @ params[:3], *expand([params[3]], [params[4]]))
        return -0.5 * n * math.log(2 * math.pi * params[5] ** 2) - e @ e / (2 * params[5] ** 2)

    best = minimize(lambda params: -loglike(params), estimates(fit) + 0.01, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 6000, "maxfev": 6000})
    assert_allclose(estimates(fit), best.x, rtol=2e-5, atol=2e-6)
    assert fit.metrics["log_likelihood"] == pytest.approx(-best.fun, abs=1e-8)
    e = css_residuals(y - x @ estimates(fit)[:3], *expand([estimates(fit)[3]], [estimates(fit)[4]]))
    assert fit.metrics["sigma"] == pytest.approx(math.sqrt(e @ e / n), rel=1e-9)
    hessian = approx_hess(estimates(fit), loglike)
    assert_allclose(errors(fit), np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=2e-4)
    assert fit.title.endswith("(conditional)") and fit.extra["method"] == "css"
    assert "conditional" in fit.provenance["likelihood"]
    assert fit.extra["last_state"]["disturbance_covariance"] is None
    exact = oe.arima(data=armax, y="y", x=["x1", "x2"], order=(1, 0, 1), time="t")
    assert abs(exact.metrics["log_likelihood"] - fit.metrics["log_likelihood"]) > 1e-3


def test_metrics_tests_and_diagnostics(armax, armax_fits):
    fit = armax_fits["opg"]
    n, k = len(armax), 6
    ll = fit.metrics["log_likelihood"]
    assert fit.metrics["aic"] == pytest.approx(-2 * ll + 2 * k)
    assert fit.metrics["bic"] == pytest.approx(-2 * ll + k * math.log(n))
    assert list(fit.metrics) == ["log_likelihood", "aic", "bic", "sigma", "n_differenced",
                                 "iterations"]
    b, v = estimates(fit)[1:5], np.array(fit.covariance_matrix)[1:5, 1:5]
    wald = fit.tests["model"]
    assert wald["statistic"] == pytest.approx(b @ np.linalg.solve(v, b), rel=1e-9)
    assert wald["df"] == 4 and wald["p_value"] == pytest.approx(stats.chi2.sf(wald["statistic"], 4))
    x = np.column_stack([np.ones(n), armax["x1"], armax["x2"]])
    ar, ma = expand([estimates(fit)[3]], [estimates(fit)[4]])
    residuals = dense_loglike(armax["y"].to_numpy() - x @ estimates(fit)[:3], ar, ma,
                              estimates(fit)[5])[1]
    lags = min(n // 2 - 2, 40)
    box = fit.tests["ljung_box"]
    reference = acorr_ljungbox(residuals, lags=[lags], model_df=0)
    adjusted = acorr_ljungbox(residuals, lags=[lags], model_df=2)
    assert box["lags"] == lags and box["df"] == lags and box["df_adjusted"] == lags - 2
    assert box["statistic"] == pytest.approx(reference["lb_stat"].iloc[0], rel=1e-7)
    assert box["p_value"] == pytest.approx(reference["lb_pvalue"].iloc[0], rel=1e-6)
    assert box["p_value_adjusted"] == pytest.approx(adjusted["lb_pvalue"].iloc[0], rel=1e-6)
    jb = sm_jarque_bera(residuals)
    assert fit.tests["jarque_bera"]["statistic"] == pytest.approx(jb[0], rel=1e-6)
    assert fit.tests["jarque_bera"]["p_value"] == pytest.approx(jb[1], rel=1e-6)
    arch = het_arch(residuals, nlags=1)
    assert fit.tests["arch_lm"]["statistic"] == pytest.approx(arch[0], rel=1e-6)
    custom = oe.arima(data=armax, y="y", x=["x1", "x2"], order=(1, 0, 1), time="t", ljung_lags=7)
    assert custom.tests["ljung_box"]["lags"] == 7
    assert custom.tests["ljung_box"]["statistic"] == pytest.approx(
        acorr_ljungbox(residuals, lags=[7])["lb_stat"].iloc[0], rel=1e-7)
    roots = fit.extra["roots"]
    assert roots["ar"][0]["modulus"] == pytest.approx(abs(estimates(fit)[3]))
    assert roots["ma"][0]["real"] == pytest.approx(-estimates(fit)[4])
    assert fit.extra["ar"] == [pytest.approx(estimates(fit)[3])]
    state = fit.extra["last_state"]
    assert state["last_period"] == int(armax["t"].iloc[-1])
    assert state["process"] == [pytest.approx((armax["y"].to_numpy() - x @ estimates(fit)[:3])[-1])]
    assert state["disturbances"] == [pytest.approx(residuals[-1], abs=1e-7)]
    assert fit.provenance["stata_parity_validated"] is False
    assert fit.provenance["optimizer"]["gradient"] == "analytic"
    assert "numerical" in fit.provenance["optimizer"]["hessian"]


def test_several_starts_escape_local_maxima():
    """Over-fitted mixed models have several local maxima; the best start must win."""
    cases = [(simulate(300, [0.5, -0.3], [0.4], seed=7), (3, 0, 3)),
             (simulate(150, [0.6], [0.3], seed=5), (2, 0, 2))]
    for y, order in cases:
        fit = oe.arima(data=pd.DataFrame({"y": y}), y="y", order=order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = SARIMAX(y, exog=np.ones(len(y)), order=order).fit(disp=False, maxiter=500)
        assert fit.metrics["log_likelihood"] >= reference.llf - 1e-5
        starts = fit.provenance["optimizer"]["starts"]
        assert {"hannan_rissanen", "ols_white_noise", "hannan_rissanen_identity"} <= set(starts)
        values = [value for value in starts.values() if value is not None]
        assert max(values) - min(values) > 0.1                 # the starts end in different basins
        assert fit.metrics["log_likelihood"] == pytest.approx(max(values), abs=1e-5)
        assert fit.provenance["optimizer"]["starting_values"] in starts
    pure = oe.arima(data=pd.DataFrame({"y": cases[1][0]}), y="y", order=(2, 0, 0))
    assert not pure.provenance["optimizer"].get("starts")         # a pure AR model: one start


def test_hessian_in_two_parts_for_many_regressors(monkeypatch):
    rng = np.random.default_rng(17)
    n = 400
    x = rng.normal(size=(n, 5))
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(5)])
    frame["y"] = 1 + x @ np.array([0.5, -1.0, 0.25, 2.0, 0.0]) + simulate(n, [0.6], [0.3], seed=18)
    names = [f"x{i}" for i in range(5)]
    split = oe.arima(data=frame, y="y", x=names, order=(1, 0, 1), covariance="nonrobust")
    assert "ARMA block" in split.provenance["optimizer"]["hessian"]
    from openecon.econometrics.arima import maximize as module

    monkeypatch.setattr(module, "_SPLIT_HESSIAN", 10 ** 9)
    full = oe.arima(data=frame, y="y", x=names, order=(1, 0, 1), covariance="nonrobust")
    assert "ARMA block" not in full.provenance["optimizer"]["hessian"]
    assert_allclose(estimates(split), estimates(full), rtol=1e-7, atol=1e-9)
    assert_allclose(np.array(split.covariance_matrix), np.array(full.covariance_matrix),
                    rtol=1e-6, atol=1e-10)
    design = np.column_stack([np.ones(n), x])

    def loglike(params):
        ar, ma = expand([params[6]], [params[7]])
        residual = frame["y"].to_numpy() - design @ params[:6]
        return dense_loglike(residual, ar, ma, params[8])[0].sum()

    hessian = approx_hess(estimates(split), loglike)
    assert_allclose(errors(split), np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=2e-4)


# ---- sample handling ----------------------------------------------------------------------------


def test_time_order_gaps_and_missing_values(armax, armax_fits):
    shuffled = armax.sample(frac=1.0, random_state=2)
    again = oe.arima(data=shuffled, y="y", x=["x1", "x2"], order=(1, 0, 1), time="t")
    assert_allclose(estimates(again), estimates(armax_fits["opg"]), rtol=1e-12)
    assert again.provenance["sample_order"] == "sorted by t"
    unordered = oe.arima(data=shuffled, y="y", x=["x1", "x2"], order=(1, 0, 1))
    assert abs(estimates(unordered)[3] - estimates(again)[3]) > 0.05     # row order is time order
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.drop(index=[70]), y="y", order=(1, 0, 0), time="t")
    assert error.value.code == "time_gaps"
    repeated = armax.copy()
    repeated.loc[5, "t"] = repeated.loc[6, "t"]
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=repeated, y="y", order=(1, 0, 0), time="t")
    assert error.value.code == "repeated_time_values"
    holes = armax.copy()
    holes.loc[50, "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=holes, y="y", order=(1, 0, 0))
    assert error.value.code == "missing_values"
    for time_column in (None, "t"):
        with pytest.raises(AnalysisError) as error:
            oe.arima(data=holes, y="y", order=(1, 0, 0), missing="drop", time=time_column)
        assert error.value.code == "time_gaps"
    ends = armax.copy()
    ends.loc[0, "y"] = np.nan
    ends.loc[len(ends) - 1, "x1"] = np.nan
    trimmed = oe.arima(data=ends, y="y", x=["x1"], order=(1, 0, 0), missing="drop", time="t")
    reference = oe.arima(data=armax.iloc[1:-1], y="y", x=["x1"], order=(1, 0, 0), time="t")
    assert trimmed.nobs == len(armax) - 2 and trimmed.dropped_rows == 2
    assert_allclose(estimates(trimmed), estimates(reference), rtol=1e-12)
    dated = armax.assign(date=pd.date_range("2001-01-01", periods=len(armax), freq="MS"))
    by_date = oe.arima(data=dated.sample(frac=1.0, random_state=3), y="y", x=["x1", "x2"],
                       order=(1, 0, 1), time="date")
    assert_allclose(estimates(by_date), estimates(armax_fits["opg"]), rtol=1e-12)
    assert by_date.extra["last_state"]["last_period"] is None
    assert any("consecutive periods" in w for w in by_date.warnings)


def test_collinear_and_categorical_regressors(armax):
    frame = armax.assign(x3=2 * armax["x1"] - armax["x2"], trend=np.arange(len(armax)),
                         group=np.where(np.arange(len(armax)) % 3 == 0, "a",
                                        np.where(np.arange(len(armax)) % 3 == 1, "b", "c")))
    fit = oe.arima(data=frame, y="y", x=["x1", "x2", "x3"], order=(1, 0, 0), time="t")
    assert fit.provenance["omitted_terms"] == ["x3"]
    assert any("Omitted because of collinearity: x3" in w for w in fit.warnings)
    reference = oe.arima(data=frame, y="y", x=["x1", "x2"], order=(1, 0, 0), time="t")
    assert_allclose(estimates(fit), estimates(reference), rtol=1e-10)
    # A linear trend is a constant after differencing: omitted next to the drift.
    differenced = oe.arima(data=frame, y="y", x=["x1", "trend"], order=(0, 1, 1), time="t")
    assert differenced.provenance["omitted_terms"] == ["trend"]
    categorical = oe.arima(data=frame, y="y", x=["x1", "group"], categorical=["group"],
                           order=(1, 0, 0), time="t")
    assert [c.term for c in categorical.coefficients][:4] == ["Intercept", "x1", "group[b]",
                                                              "group[c]"]
    dummies = frame.assign(b=(frame["group"] == "b") * 1.0, c=(frame["group"] == "c") * 1.0)
    manual = oe.arima(data=dummies, y="y", x=["x1", "b", "c"], order=(1, 0, 0), time="t")
    assert_allclose(estimates(categorical), estimates(manual), rtol=1e-10)
    assert categorical.provenance["categorical_encoding"]["group"]["reference"] == "a"


def test_constant_and_parameter_free_models(armax):
    y = armax["y"].to_numpy()
    n = len(y)
    white = oe.arima(data=armax, y="y", order=(0, 0, 0), covariance="nonrobust")
    assert estimates(white) == pytest.approx([y.mean(), y.std()])
    assert errors(white) == pytest.approx([y.std() / math.sqrt(n), y.std() / math.sqrt(2 * n)],
                                          rel=1e-6)
    assert "model" not in white.tests
    bare = oe.arima(data=armax, y="y", order=(0, 0, 0), constant=False)
    assert [c.term for c in bare.coefficients] == ["/sigma"]
    assert bare.metrics["sigma"] == pytest.approx(math.sqrt(np.mean(y ** 2)))
    assert bare.metrics["iterations"] == 0 and bare.spec.intercept is False
    no_constant = oe.arima(data=armax, y="y", x=["x1"], order=(1, 0, 0), constant=False)
    assert [c.term for c in no_constant.coefficients] == ["x1", "ARMA:L1.ar", "/sigma"]
    via_option = oe.fit(ModelSpec(estimator="arima", outcome="y", predictors=["x1"],
                                  options={"order": [1, 0, 0], "constant": False}), data=armax)
    assert_allclose(estimates(via_option), estimates(no_constant), rtol=1e-12)


def test_overdifferenced_series_reaches_the_invertibility_boundary():
    frame = pd.DataFrame({"y": simulate(250, seed=77)})          # white noise, differenced once
    fit = oe.arima(data=frame, y="y", order=(0, 1, 1), constant=False)
    theta = fit.coefficients[0].estimate
    assert -1.0 - 1e-6 <= theta < -0.9
    assert any("boundary of the invertibility" in w for w in fit.warnings)
    # At the MA unit root the theta score is an exact multiple of the sigma score (the
    # likelihood is symmetric in theta -> 1/theta), so the OPG is singular: the observed
    # information is reported instead and the substitution is recorded.
    assert abs(theta + 1.0) < 1e-6                      # the pile-up at the unit root
    assert fit.spec.covariance == "opg" and fit.inference["covariance"] == "nonrobust"
    assert fit.inference["requested_covariance"] == "opg"
    assert any("scores are linearly dependent" in w for w in fit.warnings)
    oim = oe.arima(data=frame, y="y", order=(0, 1, 1), constant=False, covariance="nonrobust")
    assert_allclose(errors(fit), errors(oim), rtol=1e-8)
    assert "requested_covariance" not in oim.inference
    # The sandwich would give theta a zero variance there: the same substitution.
    robust = oe.arima(data=frame, y="y", order=(0, 1, 1), constant=False, covariance="robust")
    assert robust.inference["covariance"] == "nonrobust"
    assert robust.inference["requested_covariance"] == "robust"
    assert_allclose(errors(robust), errors(oim), rtol=1e-8)
    differenced = np.diff(frame["y"].to_numpy())
    dense = dense_loglike(differenced, *expand([], [theta]), fit.metrics["sigma"], terms=3)[0].sum()
    assert fit.metrics["log_likelihood"] == pytest.approx(dense, rel=1e-9)
    grid = [dense_loglike(differenced, *expand([], [value]), fit.metrics["sigma"], terms=3)[0].sum()
            for value in np.linspace(-0.999, -0.5, 60)]
    assert fit.metrics["log_likelihood"] >= max(grid) - 1e-6
    near = oe.arima(data=pd.DataFrame({"y": np.cumsum(simulate(300, seed=78)) * 0.01
                                       + simulate(300, [0.995], seed=79)}), y="y", order=(1, 0, 0))
    assert any("unit root" in w for w in near.warnings)


# ---- failure contract ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, code", [
    ({"order": (1, 0)}, "invalid_order"),
    ({"order": (1, -1, 0)}, "invalid_order"),
    ({"order": "110"}, "invalid_order"),
    ({"order": (1, 4, 0)}, "invalid_order"),
    ({"order": (1, 0, 0), "seasonal": (1, 0, 0)}, "invalid_order"),
    ({"order": (1, 0, 0), "period": 12}, "invalid_order"),
    ({"order": (1, 0, 0), "seasonal": (1, 0, 1), "period": 365}, "model_too_large"),
    ({"order": (1, 0, 0), "tolerance": 0.0}, "invalid_option"),
    ({"order": (1, 0, 0), "ljung_lags": 5000}, "invalid_lags"),
    ({"order": (1, 0, 0), "x": "x1"}, "invalid_spec"),
    ({"order": (1, 0, 0), "x": ["nope"]}, "missing_columns"),
])
def test_error_codes(armax, kwargs, code):
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax, y="y", **kwargs)
    assert error.value.code == code


def test_degenerate_samples_and_invalid_specs(armax):
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.iloc[:6], y="y", order=(2, 1, 2))
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.iloc[:10], y="y", order=(0, 1, 1), seasonal=(0, 1, 1), period=12)
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.assign(y=3.0), y="y", order=(1, 0, 0))
    assert error.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.assign(y=np.arange(len(armax)) * 2.0), y="y", order=(1, 1, 0))
    assert error.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.assign(y=2 * armax["x1"] + 1), y="y", x=["x1"], order=(1, 0, 0))
    assert error.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as error:
        oe.arima(data=armax.assign(y="a"), y="y", order=(1, 0, 0))
    assert error.value.code == "non_numeric_column"
    for bad in ({"method": "exact"}, {"covariance": "HC1"}, {"max_iterations": 0},
                {"period": 1, "seasonal": (1, 0, 0)}, {"alpha": 1.5}):
        # A specification mistake: pydantic's ValidationError, or AnalysisError('invalid_spec')
        # when the shared make_spec wraps it.
        with pytest.raises((ValidationError, AnalysisError)) as error:
            oe.arima(data=armax, y="y", order=(1, 0, 0), **bad)
        assert getattr(error.value, "code", "invalid_spec") == "invalid_spec"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="arima", outcome="y")                       # order is required
    with pytest.raises(ValidationError):
        ModelSpec(estimator="arima", outcome="y", options={"order": [1, 0, 0]}, panel="id")
    with pytest.raises(ValidationError):
        ModelSpec(estimator="arima", outcome="y", options={"order": [1, 0, 0]}, weights="w",
                  weight_type="aweight")


# ---- persistence, rendering and the public surface ---------------------------------------------


def test_json_roundtrip_summary_latex_and_registry(armax_fits):
    fit = armax_fits["opg"]
    assert type(fit).model_validate_json(fit.model_dump_json()) == fit
    text = fit.summary()
    assert "ARIMA(1,0,1) regression" in text and "[ARMA]" in text and "ARMA:L1.ma" in text
    assert "/sigma" in text and "Ljung-Box" in text and "log_likelihood" in text
    latex = fit.to_latex()
    assert "ARMA:L1.ar" in latex and "tabular" in latex
    info = registry.get("arima")
    assert info.function == "arima" and info.default_covariance == "opg" and info.inference == "z"
    assert info.stata == ("arima",) and info.option("order").required
    capability = oe.capabilities()["estimators"]["arima"]
    assert capability["covariances"] == ["opg", "nonrobust", "robust"]
    exports = registry.public_exports()
    for name in ("arima", "forecast", "corrgram", "wntestq", "jarque_bera", "archlm", "tssmooth"):
        assert name in exports and callable(getattr(oe, name))
    spec = ModelSpec(estimator="arima", outcome="y", predictors=["x1", "x2"], time="t",
                     options={"order": [1, 0, 1]})
    assert spec.covariance == "opg"


def test_large_sample_runs_fast():
    n = 60_000
    y = np.cumsum(0.1 + simulate(n, [0.5, -0.2], [0.4, 0.2], seed=9))
    start = time.perf_counter()
    fit = oe.arima(data=pd.DataFrame({"y": y}), y="y", order=(2, 1, 2))
    elapsed = time.perf_counter() - start
    assert_allclose(estimates(fit)[1:5], [0.5, -0.2, 0.4, 0.2], atol=0.05)
    assert fit.metrics["sigma"] == pytest.approx(1.0, abs=0.02)
    assert elapsed < 10.0
