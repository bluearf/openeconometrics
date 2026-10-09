"""Independent process-precision/SVD oracles for fixed-shape temporal GLS."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.temporal import gls
from openecon.engines.contracts import KernelError


METHODS = ("chow_lin", "fernandez", "litterman")
ROUTES = (("Y", "Q", 4), ("Y", "M", 12), ("Q", "M", 3))


def inputs(route=ROUTES[0], aggregation="sum", m=8, intercept=True):
    low_frequency, high_frequency, ratio = route
    start = "2000" if low_frequency == "Y" else "2000Q1"
    low_periods = list(map(str, pd.period_range(start, periods=m, freq="Y-DEC" if low_frequency == "Y" else "Q-DEC")))
    high_periods = list(map(str, pd.period_range("2000Q1" if high_frequency == "Q" else "2000-01", periods=m*ratio,
                                               freq="Q-DEC" if high_frequency == "Q" else "M")))
    t = np.arange(m*ratio, dtype=float)
    x = np.column_stack((.2*t+.3*np.sin(.7*t), np.cos(.31*t)+.13*np.sin(.87*t)))
    X = np.column_stack((np.ones(len(t)), x)) if intercept else x
    C = np.zeros((m, m*ratio))
    for row in range(m):
        if aggregation == "sum":
            C[row, row*ratio:(row+1)*ratio] = 1
        elif aggregation == "mean":
            C[row, row*ratio:(row+1)*ratio] = 1/ratio
        else:
            C[row, row*ratio+(ratio-1 if aggregation == "last" else 0)] = 1
    y = C @ X @ (np.array([.35, 1.7, -.55]) if intercept else np.array([1.7, -.55]))
    y += .45*np.sin(np.arange(m)*1.7)+.17*np.cos(np.arange(m)*.43)
    arguments = dict(low_periods=low_periods, high_periods=high_periods,
                     low_frequency=low_frequency, high_frequency=high_frequency,
                     aggregation=aggregation, intercept=intercept)
    return y, x, X, C, arguments


def oracle(method, y, X, C, rho, alpha):
    """Build V independently by precision, then fit GLS via NumPy SVD.

    This does not use the production causal factor, column scaling, QR,
    native distributions or any openecon helper.  It checks the complete
    joint latent covariance with the direct Gaussian projection expression.
    """
    n, k = X.shape
    m = len(y)
    D = np.eye(n)-np.eye(n, k=-1)
    if method == "chow_lin":
        precision = np.diag(np.r_[1., np.repeat(1.+rho*rho, n-2), 1.])
        precision += np.diag(np.repeat(-rho, n-1), k=1)+np.diag(np.repeat(-rho, n-1), k=-1)
    else:
        H = np.eye(n)-(rho if method == "litterman" else 0.)*np.eye(n, k=-1)
        precision = D.T @ H.T @ H @ D
    V = np.linalg.inv(precision)
    W = C @ V @ C.T
    Z = C @ X
    L = np.linalg.cholesky(W)
    white_z, white_y = np.linalg.solve(L, Z), np.linalg.solve(L, y)
    beta = np.linalg.lstsq(white_z, white_y, rcond=None)[0]
    residual = y-Z@beta
    sse = residual @ np.linalg.solve(W, residual)
    sigma2, sigma2_ml = sse/(m-k), sse/m
    bread = np.linalg.inv(white_z.T@white_z)
    cv = sigma2*bread
    distribution = np.linalg.solve(W, C@V).T
    values = X@beta+distribution@residual
    unmatched = X-distribution@Z
    high_cv = sigma2*(V-distribution@C@V+unmatched@bread@unmatched.T)
    high_cv = (high_cv+high_cv.T)/2
    se = np.sqrt(cv.diagonal())
    t = beta/se
    critical = stats.t.isf(alpha/2, m-k)
    log_likelihood = stats.multivariate_normal.logpdf(y, mean=Z@beta, cov=sigma2_ml*W)
    return dict(beta=beta, coefficient_covariance=cv, high_covariance=high_cv,
                values=values, fitted=Z@beta, residual=residual, weighted_sse=sse,
                sigma2=sigma2, sigma2_ml=sigma2_ml, log_likelihood=log_likelihood,
                se=se, statistic=t, p_value=2*stats.t.sf(abs(t), m-k),
                ci_lower=beta-critical*se, ci_upper=beta+critical*se, critical=critical,
                df=m-k)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("aggregation", ("sum", "mean", "first", "last"))
@pytest.mark.parametrize("route", ROUTES)
def test_all_calendars_aggregations_complete_gls_covariance_inference(method, aggregation, route):
    y, x, X, C, arguments = inputs(route, aggregation)
    rho, alpha = (-.37 if method == "chow_lin" else .61), .1
    supplied = {} if method == "fernandez" else dict(rho=rho)
    result = getattr(gls, method)(y, x, alpha=alpha, **arguments, **supplied)
    expected = oracle(method, y, X, C, rho, alpha)
    coefficients = result["coefficients"]
    np.testing.assert_allclose(coefficients["coefficient"], expected["beta"], rtol=2e-8, atol=2e-9)
    for column, name in (("standard_error", "se"), ("t", "statistic"), ("p_value", "p_value"),
                         ("ci_lower", "ci_lower"), ("ci_upper", "ci_upper")):
        np.testing.assert_allclose(coefficients[column], expected[name], rtol=4e-8, atol=2e-9)
    assert list(coefficients["term"]) == ["_cons", "x1", "x2"]
    assert list(result["coefficient_covariance"].index) == ["_cons", "x1", "x2"]
    assert list(result["coefficient_covariance"].columns) == ["_cons", "x1", "x2"]
    for name in ("coefficient_covariance", "high_covariance"):
        np.testing.assert_allclose(result[name], expected[name], rtol=6e-8, atol=4e-9)
    assert list(result["high_covariance"].index) == arguments["high_periods"]
    assert list(result["high_covariance"].columns) == arguments["high_periods"]
    values = result["series"]["value"].to_numpy()
    np.testing.assert_allclose(values, expected["values"], rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(C@values, y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["aggregation"]["observed"], y)
    np.testing.assert_allclose(result["aggregation"]["reconstructed"], y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["low_fit"]["fitted"], expected["fitted"], rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(result["low_fit"]["residual"], expected["residual"], rtol=2e-8, atol=2e-9)
    for name in ("weighted_sse", "sigma2", "sigma2_ml", "log_likelihood"):
        assert result["fit"][name].iloc[0] == pytest.approx(expected[name], rel=3e-8, abs=2e-9)
        assert result.attrs[name] == result["fit"][name].iloc[0]
    assert result.attrs["df_residual"] == expected["df"] == 5
    high_cv = result["high_covariance"].to_numpy()
    np.testing.assert_allclose(C@high_cv@C.T, np.zeros((len(y), len(y))), atol=1e-10)
    assert np.linalg.eigvalsh(high_cv).min() >= -1e-10
    intervals = result["latent_intervals"]
    se = np.sqrt(np.maximum(high_cv.diagonal(), 0))
    np.testing.assert_allclose(intervals["standard_error"], se, atol=1e-13)
    np.testing.assert_allclose(intervals["ci_lower"], values-expected["critical"]*se, atol=1e-11)
    np.testing.assert_allclose(intervals["ci_upper"], values+expected["critical"]*se, atol=1e-11)
    if aggregation in ("first", "last"):
        observed = C.argmax(1)
        np.testing.assert_array_equal(high_cv[observed], np.zeros((len(y), len(values))))
        np.testing.assert_array_equal(intervals["standard_error"].iloc[observed], np.zeros(len(y)))
        np.testing.assert_array_equal(values[observed], y)
    assert result.attrs["scale_convention"] == "innovation variance"
    assert "fixed" in result.attrs["inference"]
    assert "forecast" in result.attrs["uncertainty"]
    assert result.attrs["complete_inputs_saved"]


@pytest.mark.parametrize("aggregation", ("sum", "mean", "first", "last"))
def test_litterman_rho_zero_is_fernandez_for_every_table(aggregation):
    y, x, _, _, arguments = inputs(aggregation=aggregation)
    fernandez = gls.fernandez(y, x, **arguments)
    litterman = gls.litterman(y, x, rho=0, **arguments)
    for name in fernandez:
        if name != "settings":
            pd.testing.assert_frame_equal(fernandez[name], litterman[name], check_exact=False, rtol=1e-13, atol=1e-13)
    for name in ("sigma2", "sigma2_ml", "weighted_sse", "log_likelihood", "df_residual"):
        assert fernandez.attrs[name] == litterman.attrs[name]


@pytest.mark.parametrize("method", METHODS)
def test_no_intercept_and_indicator_unit_response_scaling(method):
    y, x, X, C, arguments = inputs(intercept=False)
    kwargs = {} if method == "fernandez" else dict(rho=.42)
    original = getattr(gls, method)(y, x, **arguments, **kwargs)
    expected = oracle(method, y, X, C, .42, .05)
    np.testing.assert_allclose(original["coefficient_covariance"], expected["coefficient_covariance"], rtol=2e-9, atol=1e-10)
    assert list(original["coefficients"]["term"]) == ["x1", "x2"]
    scale_y, scale_x = 1e-20, np.array([1e15, 1e-15])
    scaled = getattr(gls, method)(y*scale_y, x*scale_x, **arguments, **kwargs)
    np.testing.assert_allclose(scaled["series"]["value"]/scale_y, original["series"]["value"], rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(scaled["coefficients"]["coefficient"].to_numpy()*scale_x/scale_y,
                               original["coefficients"]["coefficient"], rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(scaled["coefficients"]["t"], original["coefficients"]["t"], rtol=2e-9)
    np.testing.assert_allclose(scaled["high_covariance"]/(scale_y*scale_y), original["high_covariance"], rtol=4e-9, atol=1e-10)
    assert scaled.attrs["sigma2"]/(scale_y*scale_y) == pytest.approx(original.attrs["sigma2"], rel=2e-9)
    assert scaled.attrs["log_likelihood"] == pytest.approx(original.attrs["log_likelihood"]-len(y)*math.log(scale_y), rel=2e-9)


@pytest.mark.parametrize("method", METHODS)
def test_complete_json_inputs_replay_and_latex(method):
    y, x, _, _, arguments = inputs(aggregation="mean", m=5)
    kwargs = {} if method == "fernandez" else dict(rho=-.4)
    result = getattr(gls, method)(y, x, **arguments, **kwargs)
    restored = json.loads(json.dumps(result.attrs, allow_nan=False))
    replay_arguments = {name: restored[name] for name in ("low_periods", "high_periods", "low_frequency", "high_frequency",
                                                        "aggregation", "as_of", "low_releases", "high_releases",
                                                        "device", "weights", "intercept", "alpha")}
    if method != "fernandez":
        replay_arguments["rho"] = restored["rho"]
    replay = getattr(gls, method)(restored["low"], restored["indicator"], **replay_arguments)
    assert replay.attrs == result.attrs
    for name in result:
        pd.testing.assert_frame_equal(result[name], replay[name])
        latex = result[name].to_latex()
        assert any(f"\\begin{{{environment}}}" in latex and f"\\end{{{environment}}}" in latex
                   for environment in ("tabular", "longtable"))
    latex = result.to_latex()
    assert latex.count("\\begin{tabular}")+latex.count("\\begin{longtable}") == len(result)
    assert json.loads(result["settings"].set_index("setting").loc["indicator", "json"]) == x.tolist()


@pytest.mark.parametrize("method", ("chow_lin", "litterman"))
@pytest.mark.parametrize("rho", (-.96, .96, np.nan, np.inf, True, "0.5", None))
def test_invalid_fixed_rho_is_refused(method, rho):
    y, x, _, _, arguments = inputs()
    with pytest.raises(AnalysisError, match="rho must"):
        getattr(gls, method)(y, x, rho=rho, **arguments)


@pytest.mark.parametrize("method", ("chow_lin", "litterman"))
@pytest.mark.parametrize("rho", (-.95, .95))
def test_rho_boundaries_match_innovation_process_oracle(method, rho):
    y, x, X, C, arguments = inputs()
    result = getattr(gls, method)(y, x, rho=rho, **arguments)
    expected = oracle(method, y, X, C, rho, .05)
    np.testing.assert_allclose(result["coefficients"]["coefficient"], expected["beta"], rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(result["high_covariance"], expected["high_covariance"], rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("name,value", (("intercept", 1), ("alpha", True), ("alpha", 0),
                                        ("alpha", 1), ("alpha", np.nan), ("alpha", 1e-9)))
def test_inference_options_are_explicit(name, value):
    y, x, _, _, arguments = inputs()
    with pytest.raises(AnalysisError):
        gls.chow_lin(y, x, rho=.3, **(arguments | {name: value}))


@pytest.mark.parametrize("method", METHODS)
def test_no_residual_df_perfect_fit_and_aggregated_collinearity_refused(method):
    y, x, X, C, arguments = inputs()
    kwargs = {} if method == "fernandez" else dict(rho=.3)
    with pytest.raises(AnalysisError, match="Residual innovation variance"):
        getattr(gls, method)(C@X@np.array([1., 2., 3.]), x, **arguments, **kwargs)
    with pytest.raises(AnalysisError, match="more low-frequency observations"):
        small_y, small_x, _, _, small_arguments = inputs(m=2)
        getattr(gls, method)(small_y, small_x, **small_arguments, **kwargs)
    # Indicators differ within each year but have the same aggregated values:
    # high-frequency independence cannot identify a low-frequency GLS fit.
    independent_high = np.column_stack((x[:, 0], x[:, 0]+np.tile([-1., 1., -1., 1.], len(y))))
    with pytest.raises(AnalysisError, match="collinear or ill-conditioned"):
        getattr(gls, method)(y, independent_high, **arguments, **kwargs)
    with pytest.raises(AnalysisError, match="zero or constant"):
        getattr(gls, method)(y, np.ones(len(x)), **arguments, **kwargs)


def test_decomposition_and_native_distribution_failures_are_analysis_errors(monkeypatch):
    y, x, _, _, arguments = inputs()

    def decomposition_failure(*args, **kwargs):
        raise torch.linalg.LinAlgError("test decomposition refusal")

    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "qr", decomposition_failure)
        with pytest.raises(AnalysisError, match="matrix decomposition"):
            gls.chow_lin(y, x, rho=.3, **arguments)

    def native_failure(*args, **kwargs):
        raise KernelError("numerical_failure", "test native distribution refusal")

    monkeypatch.setattr(gls, "t_isf", native_failure)
    with pytest.raises(AnalysisError, match="native distribution refusal"):
        gls.chow_lin(y, x, rho=.3, **arguments)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("aggregation", ("first", "last"))
def test_exact_zero_low_benchmarks_have_exact_observed_values_and_zero_uncertainty(method, aggregation):
    y, x, _, C, arguments = inputs(aggregation=aggregation)
    y[[0, 3, 5]] = 0.
    kwargs = {} if method == "fernandez" else dict(rho=.4)
    result = getattr(gls, method)(y, x, **arguments, **kwargs)
    positions = C.argmax(1)
    np.testing.assert_array_equal(result["series"]["value"].iloc[positions], y)
    np.testing.assert_array_equal(result["high_covariance"].iloc[positions], np.zeros((len(y), len(x))))
    np.testing.assert_array_equal(result["aggregation"]["error"], np.zeros(len(y)))
    np.testing.assert_array_equal(result["latent_intervals"]["ci_lower"].iloc[positions], y)
    np.testing.assert_array_equal(result["latent_intervals"]["ci_upper"].iloc[positions], y)


@pytest.mark.parametrize("method", METHODS)
def test_cpu_float64_contract_overrides_global_torch_defaults(method):
    y, x, _, _, arguments = inputs()
    kwargs = {} if method == "fernandez" else dict(rho=.4)
    baseline = getattr(gls, method)(y, x, **arguments, **kwargs)
    previous_dtype, previous_device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = getattr(gls, method)(y, x, **arguments, **kwargs)
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)
    for name in baseline:
        pd.testing.assert_frame_equal(baseline[name], result[name])
    assert result.attrs["device"] == "cpu"
    assert result.attrs["precision"] == "float64"
