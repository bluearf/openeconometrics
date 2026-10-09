"""Independent recurrence, analytic, NumPy/SciPy test-only ARFIMA oracles."""

import copy
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.fractional import kernels as k
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def binomial_weights(d, size):
    values = [1.]
    for lag in range(1, size):
        values.append(values[-1] * (lag - 1 - d) / lag)
    return np.array(values)


def data_fixture(n=512, *, d=.25, ar=.35, ma=.2, mean=1.2, sigma=.8, terms=96):
    rng = np.random.default_rng(58017)
    innovation = rng.normal(0, sigma, n)
    denominator = np.convolve(binomial_weights(d, terms), [1., -ar])
    y = mean + lfilter([1., ma], denominator, innovation)
    return y


@pytest.mark.parametrize("d", [-1., -.35, 0., .3, 1.])
def test_fractional_weights_and_independent_convolution(d):
    y = np.random.default_rng(81).normal(size=1000)
    result = oe.fracdiff(y, d, terms=128)
    weights = binomial_weights(d, 128)
    np.testing.assert_allclose(result.attrs["filter_weights"], weights, atol=1e-15)
    np.testing.assert_allclose(result.difference, np.convolve(y, weights)[:len(y)], atol=2e-14)
    trimmed = oe.fracdiff(y, d, terms=128, initial="drop")
    assert trimmed.position.tolist() == list(range(127, len(y)))
    np.testing.assert_allclose(trimmed.difference, result.difference.iloc[127:], atol=1e-14)


def test_tolerance_is_explicit_not_tail_guarantee():
    result = oe.fracdiff(np.arange(300.), .3, terms=128, tolerance=.001)
    assert abs(result.attrs["first_omitted_coefficient"]) <= .001
    assert result.attrs["terms"] < 128
    with pytest.raises(AnalysisError, match="terms cannot satisfy"):
        oe.fracdiff(np.arange(300.), .3, terms=128, tolerance=1e-12)


@pytest.mark.parametrize("n", [3, 65, 513, 4096])
def test_formal_inverse_matches_independent_recursive_filter(n):
    coefficients = torch.tensor([1., .5, -.2], dtype=torch.float64)
    impulse = np.zeros(n)
    impulse[0] = 1
    np.testing.assert_allclose(k.inverse(coefficients, n), lfilter([1.], coefficients.numpy(), impulse), atol=3e-14)


def test_gph_full_inference_against_independent_periodogram_ols():
    y = data_fixture(n=4096, ar=0, ma=0, terms=1024, mean=0)
    m = 72
    result = oe.gph(y, bandwidth=m, alpha=.1)
    frequencies = 2 * np.pi * np.arange(1, m + 1) / len(y)
    spectrum = np.abs(np.fft.rfft(y - y.mean())[1:m + 1]) ** 2 / (2 * np.pi * len(y))
    x = np.column_stack([np.ones(m), -np.log(4 * np.sin(frequencies / 2) ** 2)])
    beta = np.linalg.lstsq(x, np.log(spectrum), rcond=None)[0]
    covariance = np.linalg.inv(x.T @ x) * np.pi ** 2 / 6
    np.testing.assert_allclose(result["estimate"].estimate, beta, atol=1e-12)
    np.testing.assert_allclose(result["estimate"].attrs["covariance_matrix"], covariance, atol=1e-12)
    se = np.sqrt(covariance.diagonal())
    np.testing.assert_allclose(result["estimate"].p_value, 2 * norm.sf(np.abs(beta / se)), atol=1e-12)
    np.testing.assert_allclose(result["estimate"].ci_low, beta - norm.isf(.05) * se, atol=1e-12)
    assert abs(result.attrs["d"] - .25) < .16
    assert result["regression"].attrs["df"] == m - 2
    assert result["estimate"].attrs["df"] is None


def test_white_noise_d_zero_exact_analytic_fit_and_uncertainty():
    y = np.random.default_rng(291).normal(2, .7, 700)
    result = oe.arfima({"y": y}, "y", d=0., terms=128)
    sigma = np.std(y, ddof=0)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], [y.mean(), sigma], atol=1e-10)
    np.testing.assert_allclose(result.covariance_matrix, np.diag([sigma ** 2 / len(y), sigma ** 2 / (2 * len(y))]), atol=1e-12)
    likelihood = -.5 * len(y) * (np.log(2 * np.pi * sigma ** 2) + 1)
    assert result.metrics["log_likelihood"] == pytest.approx(likelihood, abs=1e-10)


def test_likelihood_gradient_matches_independent_recurrence_and_difference():
    y = data_fixture()
    point = np.array([1.1, .24, .31, .15, .9])
    def oracle(theta):
        mean, d, ar, ma, sigma = theta
        transformed = np.convolve(y - mean, binomial_weights(d, 96))[:len(y)]
        error = lfilter([1., -ar], [1., ma], transformed)[8:]
        return -.5 * (len(error) * np.log(2 * np.pi * sigma ** 2) + np.sum(error ** 2) / sigma ** 2)
    theta = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    def native(t):
        return k.loglike(torch.tensor(y, dtype=torch.float64), t[0], t[1], t[2:3], t[3:4], t[4], 96, 8)
    value = native(theta)
    gradient = torch.autograd.grad(value, theta)[0]
    assert float(value.detach()) == pytest.approx(oracle(point), abs=1e-10)
    differences = []
    for axis in range(5):
        step = np.eye(5)[axis] * 1e-5
        differences.append((oracle(point + step) - oracle(point - step)) / 2e-5)
    np.testing.assert_allclose(gradient, differences, rtol=1e-7, atol=2e-7)


def test_joint_fit_full_covariance_and_restored_forecast_oracles():
    y = data_fixture(n=640)
    result = oe.arfima({"y": y, "t": np.arange(len(y))}, "y", ar=1, ma=1, terms=96, burn=8, time="t")
    params = np.array([c.estimate for c in result.coefficients])
    def objective(theta):
        mean, d, ar, ma, sigma = theta
        transformed = np.convolve(y - mean, binomial_weights(d, 96))[:len(y)]
        errors = lfilter([1., -ar], [1., ma], transformed)[8:]
        return .5 * (len(errors) * np.log(2 * np.pi * sigma ** 2) + np.sum(errors ** 2) / sigma ** 2)
    independent = minimize(objective, [1.2, .2, .3, .2, .8], method="BFGS", options={"gtol": 1e-5})
    assert independent.fun == pytest.approx(objective(params), abs=1e-7)
    np.testing.assert_allclose(params, independent.x, atol=2e-5)
    # Mixed second differences of independent scalar objective, including sigma.
    h = 2e-4
    hessian = np.empty((5, 5))
    for i in range(5):
        for j in range(5):
            ei, ej = np.eye(5)[i] * h, np.eye(5)[j] * h
            hessian[i, j] = (objective(params + ei + ej) - objective(params + ei - ej)
                             - objective(params - ei + ej) + objective(params - ei - ej)) / (4 * h * h)
    np.testing.assert_allclose(result.covariance_matrix, np.linalg.inv(hessian), rtol=4e-4, atol=1e-6)
    assert result.sample_positions == list(range(8, len(y)))
    assert result.nobs == len(y) - 8 and result.dropped_rows == 8
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    forecast = oe.forecast(restored, 12)
    pd.testing.assert_frame_equal(forecast, oe.forecast(result, 12))
    assert forecast.time.tolist() == list(range(len(y), len(y) + 12))
    state = restored.extra["fractional_state"]
    polynomial = np.convolve(binomial_weights(state["d"], state["terms"]), [1., -state["ar"][0]])
    impulse = np.zeros(12)
    impulse[0] = 1
    psi = lfilter([1., state["ma"][0]], polynomial, impulse)
    matrix = np.zeros((12, 12))
    for row in range(12):
        matrix[row, :row + 1] = psi[:row + 1][::-1]
    covariance = state["sigma"] ** 2 * matrix @ matrix.T
    np.testing.assert_allclose(forecast.attrs["covariance_matrix"], covariance, atol=1e-12)
    history, innovations = list(state["history"]), list(state["innovations"])
    means = []
    for _ in range(12):
        new = -np.dot(polynomial[1:], history[-(len(polynomial) - 1):][::-1]) + state["ma"][0] * innovations[-1]
        history.append(new)
        innovations.append(0.)
        means.append(state["mean"] + new)
    np.testing.assert_allclose(forecast.forecast, means, atol=1e-12)
    assert "parameter" in forecast.attrs["uncertainty"]
    assert "\\begin{tabular}" in result.to_latex() and "\\begin{tabular}" in forecast.to_latex()
    changed = copy.deepcopy(restored)
    changed.extra["fractional_state"]["mean"] += 1
    with pytest.raises(AnalysisError, match="saved ARFIMA state"):
        oe.forecast(changed, 5)


def test_d_zero_recovers_existing_conditional_arma():
    y = data_fixture(n=600, d=0, terms=128)
    fractional = oe.arfima({"y": y}, "y", ar=1, ma=1, d=0., terms=128)
    integer = oe.arima(data={"y": y}, y="y", order=(1, 0, 1), method="css", covariance="nonrobust")
    assert fractional.metrics["log_likelihood"] == pytest.approx(integer.metrics["log_likelihood"], abs=1e-7)
    np.testing.assert_allclose([c.estimate for c in fractional.coefficients], [c.estimate for c in integer.coefficients], atol=1e-5)
    np.testing.assert_allclose(oe.forecast(fractional, 10).forecast, oe.forecast(integer, 10).forecast, atol=1e-5)


@pytest.mark.parametrize("values", [[1., None, 3.], [1., math.nan, 3.], [True, 1., 2.], [1., math.inf, 3.], [[1.], [2.]]])
def test_missing_and_invalid_values_never_repaired(values):
    with pytest.raises(AnalysisError):
        oe.fracdiff(values, .2, terms=2)


def test_options_calendar_dataset_and_resource_guards():
    with pytest.raises(AnalysisError, match="boolean outcomes"):
        oe.arfima({"y": [True, False] * 150}, "y", d=0., terms=64)
    for kwargs in [{"ar": 4}, {"d": .7}, {"method": "ml"}, {"covariance": "robust"}, {"terms": True}, {"burn": -1}]:
        with pytest.raises(AnalysisError):
            oe.arfima({"y": list(range(300))}, "y", **kwargs)
    with pytest.raises(AnalysisError, match="calendar"):
        oe.fracdiff({"y": [1., 2., 3.], "t": [1, 3, 4]}, .2, column="y", time="t", terms=2)
    with pytest.raises(AnalysisError):
        oe.arfima({"y": [*range(100), math.nan, *range(100)]}, "y", terms=64, missing="drop")
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget"):
        oe.fracdiff(torch.ones(50000, dtype=torch.float64), .2, terms=64)
    with pytest.raises(AnalysisError, match="periodogram"):
        oe.gph([1.] * 128)
    assert ModelSpec(estimator="arfima", outcome="y").covariance == "nonrobust"
    cap = oe.capabilities()["fractional_memory"]
    assert cap["dataset_support"] is False and cap["devices"] == ["cpu"]


def test_nonfinite_filter_and_rehashed_incompatible_state_rejected():
    with pytest.raises(AnalysisError, match="overflowed"):
        oe.fracdiff([1e308] * 4, -1., terms=4)
    result = oe.arfima({"y": data_fixture(n=320)}, "y", terms=96, d=.25)
    from openecon.econometrics.fractional.models import _state_hash
    result.extra["fractional_state"]["mean"] += 1
    result.extra["fractional_state_sha256"] = _state_hash(result.extra["fractional_state"])
    with pytest.raises(AnalysisError, match="saved ARFIMA state"):
        oe.forecast(result, 3)


def test_long_series_has_linear_memory_and_no_dense_observation_square():
    n = 120001
    y = torch.sin(torch.arange(n, dtype=torch.float64) / 101)
    output = oe.fracdiff(y, .2, terms=1024)
    assert len(output) == n
    assert output.attrs["workspace"]["estimated_workspace_bytes"] < n * 300
    positions = [0, 9, 5000, n - 1]
    weights = binomial_weights(.2, 1024)
    for position in positions:
        length = min(position + 1, len(weights))
        expected = np.dot(y.numpy()[position - length + 1:position + 1][::-1], weights[:length])
        assert output.difference.iloc[position] == pytest.approx(expected, abs=1e-13)
