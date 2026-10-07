"""Independent PCA/ADF/generalized-eigenvalue oracles for Bai--Ng PANIC."""

from io import StringIO
import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.linalg import eigh
from scipy.special import log_ndtr
from scipy.stats import norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.unitroot import panic
from openecon.econometrics.unitroot.panic_kernels import (
    MQ_CRITICAL, common_sequence, idiosyncratic_probabilities,
)
from openecon.frame import DataFrame


def panel_data(seed=522, units=13, periods=77, factors=3, *, stationary=False):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(periods, factors)).cumsum(axis=0)
    innovations = rng.normal(size=(units, periods)) * np.linspace(.4, 1.5, units)[:, None]
    if stationary:
        common = rng.normal(size=(periods, factors))
        idio = innovations.copy()
        for t in range(1, periods):
            idio[:, t] += .35 * idio[:, t - 1]
    else:
        idio = innovations.cumsum(axis=1)
    loadings = rng.normal(size=(units, factors)) * 1.5
    levels = loadings @ common.T + idio + np.arange(units)[:, None] * 7
    return pd.DataFrame({"id": np.repeat(np.arange(units), periods),
                         "t": np.tile(np.arange(periods), units), "y": levels.ravel()})


def run(frame, **options):
    return panic.xtpanic(frame, "y", "id", "t", **options)


def oracle_decomposition(frame, factors, trend):
    y = frame.pivot(index="t", columns="id", values="y").to_numpy()
    differences = np.diff(y, axis=0)
    if trend == "trend":
        differences -= differences.mean(axis=0)
    # Independent small N-by-N covariance eigendecomposition, not Torch SVD.
    eigenvalues, eigenvectors = np.linalg.eigh(differences.T @ differences)
    vectors = eigenvectors[:, -factors:][:, ::-1]
    singular = np.sqrt(eigenvalues[-factors:][::-1])
    scores = differences @ vectors * (np.sqrt(len(differences)) / singular)
    loadings = vectors * (singular / np.sqrt(len(differences)))
    residuals = differences - scores @ loadings.T
    return scores.cumsum(axis=0), loadings, residuals.cumsum(axis=0).T


def oracle_adf(values, lag, trend):
    levels = np.asarray(values)
    if levels.ndim == 1:
        levels = levels[None, :]
    answer = []
    for y in levels:
        dy = np.diff(y)
        count = len(dy) - lag
        columns = [y[lag:-1]]
        columns += [dy[lag - j:len(dy) - j] for j in range(1, lag + 1)]
        if trend != "none":
            columns.append(np.ones(count))
        if trend == "trend":
            columns.append(np.arange(1, count + 1))
        x = np.column_stack(columns)
        b = np.linalg.lstsq(x, dy[lag:], rcond=None)[0]
        residual = dy[lag:] - x @ b
        variance = residual @ residual / (count - x.shape[1])
        # Explicit inverse is allowed in this small independent oracle.
        se = np.sqrt(variance * np.linalg.inv(x.T @ x)[0, 0])
        answer.append(b[0] / se)
    return np.array(answer)


def oracle_mq(levels, total, trend, method, bandwidth, order):
    length, r = levels.shape
    full_x = np.ones((length + 1, 1))
    if trend == "trend":
        full_x = np.column_stack([full_x, np.arange(length + 1)])
    deterministic = np.linalg.lstsq(full_x[1:], levels, rcond=None)[0]
    projected = np.vstack([np.zeros((1, r)), levels]) - full_x @ deterministic
    _, rotation = np.linalg.eigh(projected[1:].T @ projected[1:])
    result = []
    for m in range(r, 0, -1):
        z = projected @ rotation[:, -m:]
        if method == "mqc":
            lagged, current = z[:-1], z[1:]
            a = np.linalg.lstsq(lagged, current, rcond=None)[0]
            residual = current - lagged @ a
            one_sided = np.zeros((m, m))
            for lag in range(1, bandwidth + 1):
                # Paper orientation xi_{t-j} xi_t'; symmetric correction
                # must match irrespective of orientation convention.
                one_sided += (1 - lag / (bandwidth + 1)) * residual[:-lag].T @ residual[lag:] / total
            numerator = .5 * (current.T @ lagged + lagged.T @ current - total * (one_sided + one_sided.T))
        else:
            d = np.diff(z, axis=0)
            if order:
                x = np.column_stack([d[order - lag:-lag] for lag in range(1, order + 1)])
                a = np.linalg.lstsq(x, d[order:], rcond=None)[0]
                filtered = z[order:].copy()
                for lag in range(1, order + 1):
                    filtered -= z[order - lag:len(z) - lag] @ a[(lag - 1) * m:lag * m]
            else:
                filtered = z
            lagged, current = filtered[:-1], filtered[1:]
            numerator = .5 * (current.T @ lagged + lagged.T @ current)
        eigenvalue = eigh(numerator, lagged.T @ lagged, eigvals_only=True)[0]
        result.append(total * (eigenvalue - 1))
    return np.array(result)


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("factors", [1, 2, 3])
@pytest.mark.parametrize("lag", [0, 2])
def test_pca_projection_and_component_adf_independent_oracle(trend, factors, lag):
    frame = panel_data()
    before = frame.copy(deep=True)
    actual = run(frame, factors=factors, trend=trend, lags=lag, inference="none", components=True)
    common, loadings, idio = oracle_decomposition(frame, factors, trend)
    assert isinstance(actual, TableSet)
    assert all(isinstance(value, DataFrame) for value in actual.values())
    estimated_factors = actual["factors"].to_numpy()
    estimated_loadings = actual["loadings"].filter(like="factor_").to_numpy()
    assert_allclose(estimated_factors @ estimated_loadings.T, common @ loadings.T, atol=7e-11)
    estimated_idio = actual["idiosyncratic_levels"]["component"].to_numpy().reshape(idio.shape)
    assert_allclose(estimated_idio, idio, atol=7e-11)
    assert_allclose(actual["idiosyncratic"].statistic, oracle_adf(idio, lag, "none"), atol=4e-11)
    # F'F/(T-1)=I in the differenced score normalization.
    scores = np.diff(np.vstack([np.zeros((1, factors)), estimated_factors]), axis=0)
    assert_allclose(scores.T @ scores / len(scores), np.eye(factors), atol=2e-14)
    assert actual["idiosyncratic"].nobs.tolist() == [77 - 2 - lag] * 13
    if factors == 1:
        assert_allclose(actual["common"].statistic, oracle_adf(estimated_factors.T, lag, trend), atol=2e-11)
    if trend == "trend":
        assert_allclose(estimated_idio[:, -1], np.zeros(13), atol=4e-11)
        assert_allclose(estimated_factors[-1], np.zeros(factors), atol=2e-12)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("method,bandwidth,order", [("mqc", 0, 1), ("mqc", 4, 1), ("mqf", 0, 0), ("mqf", 0, 2)])
def test_common_mq_all_dimensions_generalized_eigen_oracle(trend, method, bandwidth, order):
    frame = panel_data()
    options = {"bandwidth": bandwidth} if method == "mqc" else {"var_lags": order}
    actual = run(frame, factors=3, trend=trend, method=method, inference="none", **options)
    common = oracle_decomposition(frame, 3, trend)[0]
    expected = oracle_mq(common, 77, trend, method, bandwidth, order)
    assert actual["common"].stochastic_trends_null.tolist() == [3, 2, 1]
    assert_allclose(actual["common"].statistic, expected, atol=2e-11)
    assert actual["common"].p_value.isna().all()
    assert actual["common"].reject.isna().all()
    assert actual.attrs["stochastic_trends"] is None


@pytest.mark.parametrize("factors", [1, 3])
@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_sign_and_orthogonal_factor_rotation_invariance(monkeypatch, factors, trend):
    frame = panel_data()
    baseline = run(frame, factors=factors, trend=trend, inference="none")
    original = panic.decompose
    rotation = np.linalg.qr(np.random.default_rng(18).normal(size=(factors, factors)))[0]
    def rotated(*args):
        scores, loadings, idio, spectrum, scale = original(*args)
        q = torch.tensor(rotation, dtype=torch.float64)
        return scores @ q, loadings @ q, idio, spectrum, scale
    monkeypatch.setattr(panic, "decompose", rotated)
    altered = run(frame, factors=factors, trend=trend, inference="none")
    assert_allclose(altered["common"].statistic, baseline["common"].statistic, atol=3e-11)
    assert_allclose(altered["idiosyncratic"].statistic, baseline["idiosyncratic"].statistic, atol=3e-11)


@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_panel_permutation_global_scale_and_block_invariance(trend):
    frame = panel_data()
    baseline = run(frame, factors=3, trend=trend, inference="none", block_size=1)
    shuffled = frame.sample(frac=1, random_state=57)
    shuffled["y"] = shuffled["y"] * -1e-100 + shuffled["id"] * 2e-100
    altered = run(shuffled, factors=3, trend=trend, inference="none", block_size=512)
    assert_allclose(altered["common"].statistic, baseline["common"].statistic, atol=3e-10)
    assert_allclose(altered["idiosyncratic"].statistic, baseline["idiosyncratic"].statistic, atol=3e-10)


def test_trend_case_removes_individual_linear_trends_constant_case_does_not():
    frame = panel_data()
    altered = frame.copy()
    altered["y"] += altered.t * (.3 + altered.id * .4)
    base = run(frame, factors=2, trend="trend", inference="none")
    other = run(altered, factors=2, trend="trend", inference="none")
    assert_allclose(other["common"].statistic, base["common"].statistic, atol=3e-10)
    assert_allclose(other["idiosyncratic"].statistic, base["idiosyncratic"].statistic, atol=3e-10)
    constant = run(frame, factors=2, inference="none")
    drifted = run(altered, factors=2, inference="none")
    assert abs(float(constant["common"].statistic.iloc[0] - drifted["common"].statistic.iloc[0])) > .1


def test_component_inference_domains_and_independence_opt_in():
    frame = panel_data()
    constant = run(frame, pooling="independent")
    statistic = constant["idiosyncratic"].statistic.to_numpy()
    # Independent explicitly transcribed MacKinnon no-constant response polynomial.
    left = .6344 + 1.2378 * statistic + .032496 * statistic**2
    right = .4797 + .93557 * statistic - .06999 * statistic**2 + .033066 * statistic**3
    reference = norm.cdf(np.where(statistic <= -1.04, left, right))
    assert_allclose(constant["idiosyncratic"].p_value, reference, rtol=4e-13)
    fisher = -2 * np.log(reference).sum()
    standardized = (fisher - 26) / math.sqrt(52)
    assert constant["pooled"].statistic.iloc[0] == pytest.approx(standardized, abs=3e-12)
    assert constant["pooled"].p_value.iloc[0] == pytest.approx(norm.sf(standardized), abs=3e-14)
    assert constant["pooled"].attrs["independence_assumed"] is True
    assert "pooled" not in run(frame)
    trend = run(frame, trend="trend")
    assert trend["common"].p_value.notna().all()
    assert trend["idiosyncratic"].p_value.isna().all()
    assert trend["idiosyncratic"].reject.isna().all()
    assert "Brownian-bridge" in trend["idiosyncratic"].attrs["distribution"]
    assert any("not a Dickey-Fuller" in note for note in trend.attrs["notes"])


def test_stable_extreme_response_log_probabilities_no_tail_clipping():
    values = np.array([-18., -14., -10., -8., -3., -1., 1.])
    polynomial = np.where(values <= -1.04, .6344 + 1.2378 * values + .032496 * values**2,
                          .4797 + .93557 * values - .06999 * values**2 + .033066 * values**3)
    probabilities, logs = idiosyncratic_probabilities(torch.tensor(values))
    assert_allclose(logs, log_ndtr(polynomial), atol=3e-13)
    assert_allclose(probabilities, np.exp(log_ndtr(polynomial)), rtol=3e-13)
    assert probabilities[0] > 0
    lower, log_lower = idiosyncratic_probabilities(torch.tensor([-19.041], dtype=torch.float64))
    assert lower[0] == 0 and torch.isneginf(log_lower[0])


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("level,index", [(.01, 0), (.05, 1), (.10, 2)])
def test_published_quantiles_and_sequential_stopping_rule(trend, level, index):
    frame = panel_data(stationary=True)
    result = run(frame, factors=3, trend=trend, level=level, method="mqf", var_lags=0)
    common = result["common"]
    for row in common.to_dict("records"):
        m = row["stochastic_trends_null"]
        assert row["critical_1%"] == MQ_CRITICAL[trend][m - 1][0]
        assert row["critical_5%"] == MQ_CRITICAL[trend][m - 1][1]
        assert row["critical_10%"] == MQ_CRITICAL[trend][m - 1][2]
        assert row["reject"] == (row["statistic"] < MQ_CRITICAL[trend][m - 1][index])
    stop = common.iloc[-1]
    assert bool(common.iloc[:-1].reject.all())
    assert result.attrs["stochastic_trends"] == (0 if stop.reject else stop.stochastic_trends_null)
    assert MQ_CRITICAL["trend"][4][2] == -55.286


def test_beyond_published_dimension_no_inference_or_extrapolation():
    frame = panel_data(units=17, periods=90, factors=7)
    with pytest.raises(AnalysisError, match="at most six"):
        run(frame, factors=7)
    result = run(frame, factors=7, inference="none")
    assert result["common"].stochastic_trends_null.tolist() == list(range(7, 0, -1))
    assert result["common"].filter(like="critical_").isna().all().all()


@pytest.mark.parametrize("alteration,code", [
    ("missing", "missing_values"), ("duplicate", "repeated_time_values"),
    ("unbalanced", "unbalanced_panel"), ("time_gap", "time_gaps"),
    ("fractional", "invalid_time"), ("duplicate_columns", "duplicate_columns"),
    ("non_numeric", "non_numeric_column"), ("infinite", "non_finite_values"),
])
def test_sample_guards(alteration, code):
    frame = panel_data()
    if alteration == "missing":
        frame.loc[5, "y"] = np.nan
    elif alteration == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif alteration == "unbalanced":
        frame = frame.iloc[1:]
    elif alteration == "time_gap":
        frame.t *= 2
    elif alteration == "fractional":
        frame.t = frame.t + .5
    elif alteration == "duplicate_columns":
        frame.columns = ["id", "t", "t"]
    elif alteration == "non_numeric":
        frame.y = "text"
    else:
        frame.loc[3, "y"] = np.inf
    with pytest.raises(AnalysisError) as caught:
        run(frame)
    assert caught.value.code == code


@pytest.mark.parametrize("option,value", [
    ("factors", 0), ("factors", True), ("factors", 13), ("factors", 33),
    ("lags", -1), ("lags", "unknown"), ("lags", 65), ("lags", 70),
    ("trend", "none"), ("method", "other"), ("inference", "table"),
    ("pooling", True), ("pooling", "weak"), ("components", 1),
    ("block_size", 513), ("memory_mb", 0), ("memory_mb", 1025),
    ("level", .025), ("level", True), ("bandwidth", True),
    ("var_lags", 17),
])
def test_option_guards(option, value):
    with pytest.raises(AnalysisError):
        run(panel_data(), **{option: value})


@pytest.mark.parametrize("options", [
    {"factors": 1, "method": "mqf"}, {"factors": 1, "bandwidth": 0},
    {"factors": 1, "var_lags": 0}, {"factors": 2, "method": "mqf", "bandwidth": 0},
    {"factors": 2, "method": "mqc", "var_lags": 0},
    {"trend": "trend", "pooling": "independent"},
    {"inference": "none", "pooling": "independent"},
    {"factors": 2, "bandwidth": 76},
    {"factors": 4, "method": "mqf", "var_lags": 16, "lags": 30},
])
def test_unused_or_unavailable_options_are_rejected(options):
    with pytest.raises(AnalysisError):
        run(panel_data(), **options)


@pytest.mark.parametrize("kind", ["constant", "exact_common", "rank", "tied"])
def test_nonidentified_decomposition_guard(kind):
    frame = panel_data(units=7, periods=31)
    if kind == "constant":
        frame.y = 1.
    elif kind in {"exact_common", "rank"}:
        frame.y = frame.t.astype(float) * (frame.id + 1)
    else:
        # Exact equal singular values at the requested factor boundary.
        increments = np.zeros((30, 7))
        increments[:7] = np.eye(7)
        y = np.vstack([np.zeros((1, 7)), increments.cumsum(0)]).T
        frame.y = y.ravel()
    with pytest.raises(AnalysisError) as caught:
        run(frame, factors=2 if kind == "rank" else 1)
    assert caught.value.code in {"constant_series", "factor_rank", "factor_boundary", "degenerate_idiosyncratic"}


def test_dense_pca_workspace_and_work_guards_before_svd(monkeypatch):
    frame = panel_data(units=100, periods=200)
    def forbidden(*args, **kwargs):
        raise AssertionError("PCA must not start before workspace checks")
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    with pytest.raises(AnalysisError) as caught:
        run(frame, memory_mb=1)
    assert caught.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as caught:
        panic._workspace(1000, 1000, 2, 0, 1, 128, 1024)
    assert caught.value.code == "work_limit"


def test_default_device_scope_restores_context_and_no_rng_or_dtype_mutation():
    frame = panel_data()
    rng = torch.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    with torch.device("meta"):
        result = run(frame, factors=2)
        assert torch.empty(1).device.type == "meta"
    assert torch.equal(rng, torch.get_rng_state())
    assert dtype == torch.get_default_dtype()
    assert result.attrs["provenance"]["device"] == "cpu"


@pytest.mark.parametrize("dtype", ["int64", "uint64", "datetime"])
def test_exact_large_integer_and_regular_datetime_period_identity(dtype):
    frame = panel_data()
    if dtype == "datetime":
        frame.t = pd.date_range("2001-01-01", periods=77, freq="MS").take(frame.t)
    else:
        frame.t = (frame.t.astype(dtype) + (2**53 + 101 if dtype == "int64" else 2**63 + 101)).astype(dtype)
    baseline = run(panel_data(), factors=2)
    actual = run(frame, factors=2)
    assert_allclose(actual["common"].statistic, baseline["common"].statistic, atol=3e-11)


def test_json_and_latex_all_tables_and_typed_panel_label_components():
    frame = panel_data(units=3)
    frame.id = frame.id.map({0: 1, 1: "1", 2: "other"})
    result = run(frame, components=True)
    json.dumps(result.attrs, allow_nan=False)
    assert result.attrs["provenance"]["stata_parity_validated"] is False
    assert result.attrs["workspace"]["estimated_tensor_workspace_bytes"] <= 64 * 1024**2
    for name, value in result.items():
        json.dumps(value.attrs, allow_nan=False)
        restored = pd.read_json(StringIO(value.to_json()), convert_dates=False, convert_axes=False)
        assert restored.shape == value.shape
        latex = value.to_latex()
        assert "\\begin{tabular}" in latex or "\\begin{longtable}" in latex
    assert "\\begin{tabular}" in result.to_latex()
    assert not result["idiosyncratic_levels"].columns.has_duplicates
    assert set(type(v).__name__ for v in result["idiosyncratic_levels"].panel) == {"int", "str"}


def test_individual_unit_mean_removal_is_not_accidentally_applied_in_constant_pca():
    frame = panel_data()
    actual = run(frame, factors=2, components=True, inference="none")
    factors, loadings, _ = oracle_decomposition(frame, 2, "constant")
    demeaned, demeaned_loadings, _ = oracle_decomposition(frame, 2, "trend")
    reconstructed = actual["factors"].to_numpy() @ actual["loadings"].filter(like="factor_").to_numpy().T
    assert_allclose(reconstructed, factors @ loadings.T, atol=6e-11)
    assert np.linalg.norm(reconstructed - demeaned @ demeaned_loadings.T) > 1


def test_generalized_eigen_statistic_is_not_nonsymmetric_eigen_or_wrong_tail():
    rng = np.random.default_rng(21)
    levels = rng.normal(size=(101, 3)).cumsum(0)
    rows, selected = common_sequence(torch.tensor(levels), 102, "constant", "mqc", 3, 1, "asymptotic", .05)
    expected = oracle_mq(levels, 102, "constant", "mqc", 3, 1)
    assert_allclose([r["statistic"] for r in rows], expected[:len(rows)], atol=4e-11)
    assert selected is not None
