"""Impulse responses, variance decompositions and forecasts of oe.var against oracles.

The statsmodels VAR supplies simple and orthogonalized responses with their
asymptotic standard errors, the variance decomposition and forecasts. Every
delta-method standard error (simple, orthogonalized, generalized, FEVD) is also
checked against a numerical Jacobian of an independently written NumPy
implementation, and the forecast covariance against Lutkepohl's formula summed
observation by observation.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.tsa.api import VAR

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.var import impulse, kernels

NAMES = ["a", "b", "c"]
K, P, STEPS = 3, 2, 6


def make_data(seed=1, n=260):
    rng = np.random.default_rng(seed)
    mix = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, -0.3, 1.0]])
    e = rng.normal(size=(n, K)) @ mix.T
    a1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.3, 0.1], [0.0, -0.2, 0.4]])
    a2 = np.array([[-0.1, 0.0, 0.1], [0.0, 0.1, 0.0], [0.1, 0.0, -0.2]])
    y = np.zeros((n, K))
    for t in range(2, n):
        y[t] = 0.3 + a1 @ y[t - 1] + a2 @ y[t - 2] + e[t]
    frame = pd.DataFrame(y, columns=NAMES)
    frame["x"] = rng.normal(size=n)
    frame["period"] = np.arange(1, n + 1)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def fitted(data):
    return oe.var(data=data, y=NAMES, lags=P, dfk=True, irf_steps=STEPS)


@pytest.fixture(scope="module")
def reference(data):
    return VAR(data[NAMES]).fit(P, trend="c")


def lag_matrices(result):
    """A_1..A_p [p, K, K] from the reported coefficients."""
    m = result.extra["layout"]["n_regressors"]
    coef = np.array([c.estimate for c in result.coefficients]).reshape(K, m)
    return np.stack([coef[:, [v * P + j for v in range(K)]] for j in range(P)])


def ma(a, steps):
    phi = [np.eye(K)]
    for i in range(1, steps + 1):
        phi.append(sum(phi[i - j] @ a[j - 1] for j in range(1, min(i, len(a)) + 1)))
    return np.array(phi)


def vech(matrix):
    return np.concatenate([matrix[c:, c] for c in range(K)])


def unvech(vector):
    out = np.zeros((K, K))
    position = 0
    for c in range(K):
        out[c:, c] = vector[position:position + K - c]
        out[c, c:] = out[c:, c]
        position += K - c
    return out


def responses(theta, kind, steps=STEPS):
    """Independent implementation: theta = (vec(A_1..A_p), vech(Sigma)) -> stacked responses."""
    a = theta[:K * K * P].reshape(P, K, K).transpose(0, 2, 1)    # column-stacked blocks
    sigma = unvech(theta[K * K * P:])
    phi = ma(a, steps)
    if kind == "simple":
        return phi.ravel()
    if kind == "generalized":
        return (phi @ sigma / np.sqrt(np.diag(sigma))).ravel()
    theta_i = phi @ np.linalg.cholesky(sigma)
    if kind == "orthogonalized":
        return theta_i.ravel()
    squares = np.cumsum(theta_i ** 2, axis=0)
    share = squares / squares.sum(axis=2, keepdims=True)
    return np.concatenate([np.zeros((1, K, K)), share[:-1]]).ravel()


def jacobian(function, theta, step=1e-6):
    base = function(theta)
    out = np.zeros((len(base), len(theta)))
    for j in range(len(theta)):
        h = step * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        out[:, j] = (function(up) - function(down)) / (2 * h)
    return out


def parameter_covariance(result):
    """Covariance of (vec(A_1..A_p), vech(Sigma)) implied by the fit."""
    m = result.extra["layout"]["n_regressors"]
    t = result.nobs
    cov = np.array(result.covariance_matrix)
    # vec(A_1..A_p): column c = (j-1) K + v, row i  ->  position i*m + v*P + (j-1).
    index = [i * m + v * P + j for j in range(P) for v in range(K) for i in range(K)]
    sigma = np.array(result.extra["sigma"])
    rows = [(r, c) for c in range(K) for r in range(c, K)]
    cov_sigma = np.array([[(sigma[r1, r2] * sigma[c1, c2] + sigma[r1, c2] * sigma[c1, r2]) / t
                           for (r2, c2) in rows] for (r1, c1) in rows])
    size = len(index) + len(rows)
    full = np.zeros((size, size))
    full[:len(index), :len(index)] = cov[np.ix_(index, index)]
    full[len(index):, len(index):] = cov_sigma
    a = lag_matrices(result)
    theta = np.concatenate([np.concatenate([a[j].T.ravel() for j in range(P)]), vech(sigma)])
    return theta, full


def test_point_estimates_match_statsmodels_and_numpy(fitted, reference):
    record = fitted.extra["irf"]
    irf = reference.irf(STEPS)
    assert_allclose(record["simple"], irf.irfs, atol=1e-10)
    assert_allclose(record["orthogonalized"], irf.orth_irfs, atol=1e-10)
    a = lag_matrices(fitted)
    sigma = np.array(fitted.extra["sigma"])
    phi = ma(a, STEPS)
    assert_allclose(record["simple"], phi, atol=1e-12)
    assert_allclose(record["generalized"], phi @ sigma / np.sqrt(np.diag(sigma)), atol=1e-12)
    # The generalized response to the first variable equals the orthogonalized one.
    assert_allclose(np.array(record["generalized"])[:, :, 0],
                    np.array(record["orthogonalized"])[:, :, 0], atol=1e-12)
    decomposition = reference.fevd(STEPS).decomp                  # [response, horizon, impulse]
    ours = np.array(record["fevd"])
    assert_allclose(ours[0], 0.0)
    assert_allclose(ours[1:].transpose(1, 0, 2), decomposition, atol=1e-10)
    assert_allclose(ours[1:].sum(axis=2), 1.0, atol=1e-12)
    assert record["variables"] == NAMES and record["steps"] == STEPS


def test_standard_errors_match_statsmodels(fitted, reference):
    irf = reference.irf(STEPS)
    se = fitted.extra["irf"]["se"]
    assert_allclose(se["simple"], irf.stderr(orth=False), atol=1e-10)
    assert_allclose(se["orthogonalized"], irf.stderr(orth=True), atol=1e-10)


@pytest.mark.parametrize("kind", ["simple", "orthogonalized", "generalized", "fevd"])
@pytest.mark.parametrize("options", [dict(dfk=True), dict(), dict(covariance="robust")])
def test_standard_errors_match_numerical_delta_method(data, kind, options):
    result = oe.var(data=data, y=NAMES, lags=P, irf_steps=STEPS, **options)
    theta, covariance = parameter_covariance(result)
    jac = jacobian(lambda value: responses(value, kind), theta)
    expected = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", jac, covariance, jac), 0, None))
    ours = np.array(result.extra["irf"]["se"][kind]).ravel()
    assert_allclose(ours, expected, rtol=2e-5, atol=1e-8)
    assert_allclose(np.array(result.extra["irf"]["fevd" if kind == "fevd" else kind]).ravel(),
                    responses(theta, kind), atol=1e-10)


def test_responses_are_invariant_to_the_units_of_the_variables(data, fitted):
    scale = np.array([1e5, 1.0, 1e-4])
    scaled = data.copy()
    scaled[NAMES] = data[NAMES].to_numpy() * scale
    result = oe.var(data=scaled, y=NAMES, lags=P, dfk=True, irf_steps=STEPS)
    ours, base = result.extra["irf"], fitted.extra["irf"]
    by_row = scale[:, None]
    assert_allclose(ours["simple"], np.array(base["simple"]) * by_row / scale, rtol=1e-7,
                    atol=1e-12)
    assert_allclose(ours["orthogonalized"], np.array(base["orthogonalized"]) * by_row, rtol=1e-7)
    assert_allclose(ours["generalized"], np.array(base["generalized"]) * by_row, rtol=1e-7)
    assert_allclose(ours["fevd"], base["fevd"], rtol=1e-7, atol=1e-12)
    assert_allclose(ours["se"]["simple"], np.array(base["se"]["simple"]) * by_row / scale,
                    rtol=1e-6, atol=1e-12)
    assert_allclose(ours["se"]["orthogonalized"], np.array(base["se"]["orthogonalized"]) * by_row,
                    rtol=1e-6)
    assert_allclose(ours["se"]["generalized"], np.array(base["se"]["generalized"]) * by_row,
                    rtol=1e-6)
    assert_allclose(ours["se"]["fevd"], base["se"]["fevd"], rtol=1e-6, atol=1e-12)
    forecast, reference = oe.forecast(result, 4), oe.forecast(fitted, 4)
    assert_allclose(forecast.forecast.to_numpy().reshape(K, 4),
                    reference.forecast.to_numpy().reshape(K, 4) * by_row, rtol=1e-7)
    assert_allclose(forecast.std_error.to_numpy().reshape(K, 4),
                    reference.std_error.to_numpy().reshape(K, 4) * by_row, rtol=1e-6)
    assert result.metrics["aic_per_obs"] == pytest.approx(
        fitted.metrics["aic_per_obs"] + 2 * np.log(scale).sum(), rel=1e-9)
    for name in ("lm_autocorrelation_L1", "normality", "granger_all_b", "lag_exclusion_L2"):
        assert result.tests[name]["statistic"] == pytest.approx(fitted.tests[name]["statistic"],
                                                                rel=1e-6)


def test_matrix_helpers():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(K, K))
    f = g + g.T
    lk, dk, kk = (m.numpy() for m in (impulse.elimination_matrix(K), impulse.duplication_matrix(K),
                                       impulse.commutation_matrix(K)))
    assert_allclose(lk @ f.ravel(order="F"), vech(f))
    assert_allclose(dk @ vech(f), f.ravel(order="F"))
    assert_allclose(kk @ g.ravel(order="F"), g.T.ravel(order="F"))
    a = torch.as_tensor(rng.normal(size=(P, K, K)) * 0.3)
    companion = kernels.companion(a).numpy()
    assert_allclose(companion[:K, :K], a[0].numpy())
    assert_allclose(companion[:K, K:], a[1].numpy())
    assert_allclose(companion[K:, :K], np.eye(K))
    phi = kernels.ma_matrices(a, 5).numpy()
    power = np.linalg.matrix_power(companion, 5)[:K, :K]
    assert_allclose(phi[5], power, atol=1e-12)
    positions = impulse.alpha_positions(K, P, 7).numpy()
    assert sorted(positions.tolist()) == sorted(i * 7 + c for i in range(K) for c in range(K * P))
    assert positions[0] == 0 and positions[1] == 7 and positions[K] == 2      # (lag 1, variable b)


def test_irf_table(fitted, data):
    table = oe.irf(fitted, steps=10)
    assert list(table.columns) == ["impulse", "response", "step", "irf", "std_error", "ci_low",
                                   "ci_high", "fevd", "fevd_std_error"]
    assert len(table) == K * K * 11 and table.attrs["kind"] == "orthogonalized"
    stored = np.array(fitted.extra["irf"]["orthogonalized"])
    block = table[(table.impulse == "b") & (table.response == "c")]
    assert_allclose(block.irf.to_numpy()[:STEPS + 1], stored[:, 2, 1], atol=1e-12)
    assert_allclose(block.std_error.to_numpy()[:STEPS + 1],
                    np.array(fitted.extra["irf"]["se"]["orthogonalized"])[:, 2, 1], atol=1e-12)
    assert_allclose(block.fevd.to_numpy()[:STEPS + 1],
                    np.array(fitted.extra["irf"]["fevd"])[:, 2, 1], atol=1e-12)
    z = stats.norm.ppf(0.975)
    assert_allclose(table.ci_high, table.irf + z * table.std_error, rtol=1e-9)
    narrow = oe.irf(fitted, steps=2, kind="simple", alpha=0.10)
    assert "fevd" not in narrow.columns
    assert_allclose(narrow.ci_low, narrow.irf - stats.norm.ppf(0.95) * narrow.std_error, rtol=1e-9)
    generalized = oe.irf(fitted, steps=STEPS, kind="generalized")
    assert_allclose(generalized[(generalized.impulse == "c") & (generalized.response == "a")]
                    .std_error, np.array(fitted.extra["irf"]["se"]["generalized"])[:, 0, 2],
                    atol=1e-12)
    # A reloaded result gives the same table.
    again = type(fitted).model_validate_json(fitted.model_dump_json())
    assert_allclose(oe.irf(again, steps=4).irf, oe.irf(fitted, steps=4).irf, atol=1e-13)
    assert str(table.to_latex())
    for options, code in ((dict(kind="structural"), "invalid_option"),
                          (dict(steps=0), "invalid_steps"), (dict(steps=10_000), "invalid_steps"),
                          (dict(alpha=0.0), "invalid_alpha")):
        with pytest.raises(AnalysisError) as error:
            oe.irf(fitted, **options)
        assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        oe.irf(oe.arima(data=data, y="a", order=(1, 0, 0)))
    assert error.value.code == "invalid_result"


def test_forecast_matches_statsmodels(fitted, reference, data):
    steps = 8
    table = oe.forecast(fitted, steps)
    assert list(table.columns) == ["variable", "step", "forecast", "std_error", "ci_low",
                                   "ci_high"]
    ours = table.forecast.to_numpy().reshape(K, steps).T
    assert_allclose(ours, reference.forecast(data[NAMES].to_numpy()[-P:], steps), atol=1e-10)
    total = reference.mse(steps) + reference._omega_forc_cov(steps) / reference.nobs
    assert_allclose(table.std_error.to_numpy().reshape(K, steps).T,
                    np.sqrt(np.diagonal(total, axis1=1, axis2=2)), rtol=1e-9)
    z = stats.norm.ppf(0.975)
    assert_allclose(table.ci_low, table.forecast - z * table.std_error, rtol=1e-9)
    assert "coefficient uncertainty" in table.attrs["mse_formula"]
    assert oe.var_forecast(fitted, steps).equals(table)


def test_forecast_parameter_uncertainty_matches_the_sum_over_observations(data):
    """(1/T) sum_t G_t V G_t' with G_t the numerical derivative of the forecast in the data."""
    result = oe.var(data=data, y=NAMES, lags=P, covariance="robust")
    steps = 4
    y = data[NAMES].to_numpy()
    m = result.extra["layout"]["n_regressors"]
    coef = np.array([c.estimate for c in result.coefficients])
    cov = np.array(result.covariance_matrix)
    sigma = np.array(result.extra["sigma"])

    def path(flat, start):
        b = flat.reshape(K, m)
        recent = [start[-1], start[-2]]
        out = []
        for _ in range(steps):
            z = np.concatenate([[recent[j][v] for j in range(P)] for v in range(K)] + [[1.0]])
            value = b @ z
            out.append(value)
            recent = [value, recent[0]]
        return np.concatenate(out)

    total = np.zeros((steps, K, K))
    t = result.nobs
    for row in range(P - 1, len(y) - 1):                      # Z_t for every estimation row
        jac = jacobian(lambda flat: path(flat, y[row - P + 1:row + 1]), coef, step=1e-5)
        for h in range(steps):
            block = jac[h * K:(h + 1) * K]
            total[h] += block @ cov @ block.T / t
    phi = ma(lag_matrices(result), steps - 1)
    innovation = np.cumsum(phi @ sigma @ phi.transpose(0, 2, 1), axis=0)
    expected = np.sqrt(np.diagonal(innovation + total, axis1=1, axis2=2))
    table = oe.forecast(result, steps)
    assert_allclose(table.std_error.to_numpy().reshape(K, steps).T, expected, rtol=1e-6)
    assert_allclose(table.forecast.to_numpy().reshape(K, steps).T,
                    path(coef, y[-P:]).reshape(steps, K), atol=1e-10)


def test_forecast_with_exogenous_trend_and_new_origin(data):
    result = oe.var(data=data, y=NAMES, x=["x"], time="period", lags=P, trend=True)
    steps = 5
    future = pd.DataFrame({"x": np.linspace(-1, 1, steps)})
    table = oe.forecast(result, steps, exog=future)
    m = result.extra["layout"]["n_regressors"]
    coef = np.array([c.estimate for c in result.coefficients]).reshape(K, m)
    y = data[NAMES].to_numpy()
    recent = [y[-1], y[-2]]
    expected = []
    for h in range(steps):
        z = np.concatenate([[recent[j][v] for j in range(P)] for v in range(K)]
                           + [[future.x[h], len(y) + h + 1, 1.0]])
        value = coef @ z
        expected.append(value)
        recent = [value, recent[0]]
    assert_allclose(table.forecast.to_numpy().reshape(K, steps).T, np.array(expected), atol=1e-9)
    assert list(table.period[:steps]) == list(range(len(y) + 1, len(y) + steps + 1))
    assert "innovations only" in table.attrs["mse_formula"]
    phi = ma(lag_matrices(result), steps - 1)
    sigma = np.array(result.extra["sigma"])
    innovation = np.cumsum(phi @ sigma @ phi.transpose(0, 2, 1), axis=0)
    assert_allclose(table.std_error.to_numpy().reshape(K, steps).T,
                    np.sqrt(np.diagonal(innovation, axis1=1, axis2=2)), rtol=1e-9)
    # A new origin: forecasting from the first 200 rows of the same series.
    earlier = oe.forecast(result, 2, data=data.iloc[:200], exog=future.iloc[:2])
    recent = [y[199], y[198]]
    z = np.concatenate([[recent[j][v] for j in range(P)] for v in range(K)]
                       + [[future.x[0], 201.0, 1.0]])
    assert_allclose(earlier.forecast.to_numpy().reshape(K, 2).T[0], coef @ z, atol=1e-9)
    assert list(earlier.period[:2]) == [201, 202]
    for options, code in ((dict(), "missing_exog"), (dict(exog=future.iloc[:2]), "invalid_exog"),
                          (dict(exog=pd.DataFrame({"w": np.zeros(steps)})), "missing_columns"),
                          (dict(exog=future.assign(x=np.nan)), "missing_values")):
        with pytest.raises(AnalysisError) as error:
            oe.forecast(result, steps, **options)
        assert error.value.code == code
    plain = oe.var(data=data, y=NAMES, lags=1)
    for options, code in ((dict(steps=0), "invalid_steps"), (dict(steps=2, exog=future),
                                                             "invalid_exog"),
                          (dict(steps=2, alpha=2), "invalid_alpha")):
        with pytest.raises(AnalysisError) as error:
            oe.var_forecast(plain, **options)
        assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        oe.var_forecast(oe.arima(data=data, y="a", order=(1, 0, 0)), 3)
    assert error.value.code == "invalid_result"


def test_explosive_horizon_is_reported_not_returned():
    a = torch.as_tensor(np.array([[[40.0, 0.0], [0.0, 30.0]]]))
    with pytest.raises(Exception) as error:
        kernels.ma_matrices(a, 400)
    assert getattr(error.value, "code", "") == "numerical_failure"
