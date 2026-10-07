"""Independent oracles for oe.fcast_eval and oe.dm_test (post-estimation family).

Every statistic is recomputed in NumPy from its textbook / EViews definition;
the Diebold-Mariano statistic is coded independently for both long-run
variance kernels (the uniform kernel reproduces the algorithm of R's
forecast::dm.test), with SciPy reference distributions.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError


def make_series(seed=3, h=60):
    rng = np.random.default_rng(seed)
    y = 10 + np.cumsum(rng.normal(size=h)) * 0.5 + rng.normal(size=h)
    f1 = y + rng.normal(scale=0.8, size=h) + 0.2
    f2 = y + rng.normal(scale=1.0, size=h)
    naive = np.r_[y[0], y[:-1]]
    return pd.DataFrame({"y": y, "f1": f1, "f2": f2, "naive": naive})


@pytest.fixture(scope="module")
def series():
    return make_series()


def eviews_statistics(y, f, m=1, naive=None):
    e = y - f
    h = len(y)
    mse = np.mean(e ** 2)
    sf, sy = np.std(f), np.std(y)
    r = np.corrcoef(f, y)[0, 1]
    u2 = np.sqrt(np.sum(((f[1:] - y[1:]) / y[:-1]) ** 2) / np.sum(((y[1:] - y[:-1]) / y[:-1]) ** 2))
    scale = np.mean(np.abs(y - naive)) if naive is not None else np.mean(np.abs(y[m:] - y[:-m]))
    return {
        "n": h, "mean_error": e.mean(), "mae": np.mean(np.abs(e)), "mse": mse,
        "rmse": np.sqrt(mse), "mape": 100 * np.mean(np.abs(e / y)),
        "smape": 100 * np.mean(np.abs(f - y) / ((np.abs(f) + np.abs(y)) / 2)),
        "theil_u1": np.sqrt(mse) / (np.sqrt(np.mean(f ** 2)) + np.sqrt(np.mean(y ** 2))),
        "theil_u2": u2,
        "bias_proportion": (f.mean() - y.mean()) ** 2 / mse,
        "variance_proportion": (sf - sy) ** 2 / mse,
        "covariance_proportion": 2 * (1 - r) * sf * sy / mse,
        "mase": np.mean(np.abs(e)) / scale,
    }


def test_fcast_eval_matches_definitions(series):
    result = oe.fcast_eval("y", "f1", data=series)
    expected = eviews_statistics(series["y"].to_numpy(), series["f1"].to_numpy())
    for name, value in expected.items():
        assert_allclose(result.loc[name, "value"], value, rtol=1e-11, err_msg=name)
    proportions = result.loc[["bias_proportion", "variance_proportion",
                              "covariance_proportion"], "value"].sum()
    assert_allclose(proportions, 1.0, rtol=1e-12)
    assert result.attrs["n"] == len(series)
    assert result.attrs["error_definition"] == "actual - forecast"
    assert list(result.columns) == ["value"]


def test_fcast_eval_inputs_seasonal_and_benchmark(series):
    y, f = series["y"].to_numpy(), series["f2"].to_numpy()
    seasonal = oe.fcast_eval(list(y), torch.as_tensor(f), seasonal_period=4)
    expected = eviews_statistics(y, f, m=4)
    assert_allclose(seasonal.loc["mase", "value"], expected["mase"], rtol=1e-12)
    assert seasonal.attrs["seasonal_period"] == 4
    bench = oe.fcast_eval(series["y"], series["f2"], naive=series["naive"])
    expected = eviews_statistics(y, f, naive=series["naive"].to_numpy())
    assert_allclose(bench.loc["mase", "value"], expected["mase"], rtol=1e-12)
    assert bench.attrs["mase_scale_source"] == "MAE of the benchmark forecast"


def test_fcast_eval_undefined_statistics_and_missing():
    y = np.array([0.0, 1.0, 2.0, 1.5, 3.0])
    f = np.array([0.0, 1.2, 1.8, 1.4, 2.5])
    result = oe.fcast_eval(y, f)
    assert math.isnan(result.loc["mape", "value"])
    assert math.isnan(result.loc["smape", "value"])
    assert math.isnan(result.loc["theil_u2", "value"])
    notes = " ".join(result.attrs["notes"])
    assert "MAPE is undefined" in notes and "Theil's U2" in notes
    exact = oe.fcast_eval([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert exact.loc["rmse", "value"] == 0 and math.isnan(exact.loc["bias_proportion", "value"])
    holes = pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0, 5.0], "f": [1.1, 2.1, 3.0, np.nan, 5.2]})
    with pytest.raises(AnalysisError) as error:
        oe.fcast_eval("y", "f", data=holes)
    assert error.value.code == "missing_values"
    dropped = oe.fcast_eval("y", "f", data=holes, missing="drop")
    assert dropped.attrs["n"] == 3
    expected = eviews_statistics(np.array([1.0, 2.0, 5.0]), np.array([1.1, 2.1, 5.2]))
    assert_allclose(dropped.loc["rmse", "value"], expected["rmse"])


def test_fcast_eval_errors(series):
    cases = [
        (lambda: oe.fcast_eval([1.0, 2.0], [1.0]), "length_mismatch"),
        (lambda: oe.fcast_eval("y", "f1"), "invalid_data"),
        (lambda: oe.fcast_eval("y", "nope", data=series), "missing_columns"),
        (lambda: oe.fcast_eval(["a", "b"], [1.0, 2.0]), "invalid_data"),
        (lambda: oe.fcast_eval([1.0], [1.0]), "insufficient_observations"),
        (lambda: oe.fcast_eval([1.0, 2.0], [1.0, 2.0], seasonal_period=0), "invalid_spec"),
        (lambda: oe.fcast_eval([1.0, 2.0], [1.0, 2.0], missing="ignore"), "invalid_spec"),
        (lambda: oe.fcast_eval(3.0, [1.0]), "invalid_data"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, code


def dm_numpy(y, f1, f2, h, loss, kernel, harvey):
    e1, e2 = y - f1, y - f2
    d = e1 ** 2 - e2 ** 2 if loss == "squared" else np.abs(e1) - np.abs(e2)
    n = len(d)
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    weights = [1.0] + [2 * (1 - k / h if kernel == "bartlett" else 1.0) for k in range(1, h)]
    lrv = np.dot(weights, gamma)
    dm = d.mean() / np.sqrt(lrv / n)
    if harvey:
        dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        return dm, 2 * stats.t.sf(abs(dm), n - 1)
    return dm, 2 * stats.norm.sf(abs(dm))


@pytest.mark.parametrize("h,loss,kernel,harvey", [
    (1, "squared", "bartlett", True), (4, "squared", "bartlett", True),
    (3, "absolute", "uniform", True), (2, "squared", "uniform", False),
    (5, "absolute", "bartlett", False)])
def test_dm_test_matches_hand_formula(series, h, loss, kernel, harvey):
    result = oe.dm_test("y", "f1", "f2", data=series, horizon=h, loss=loss, kernel=kernel,
                        harvey=harvey)
    dm, p = dm_numpy(series["y"].to_numpy(), series["f1"].to_numpy(), series["f2"].to_numpy(),
                     h, loss, kernel, harvey)
    row = result.iloc[0]
    assert_allclose(row["statistic"], dm, rtol=1e-11)
    assert_allclose(row["p_value"], p, rtol=1e-9)
    assert result.attrs["distribution"] == ("t" if harvey else "normal")
    if harvey:
        assert row["df"] == len(series) - 1
    assert row["n"] == len(series)


def test_dm_test_sign_and_power():
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    good = y + rng.normal(scale=0.5, size=400)
    bad = y + rng.normal(scale=1.5, size=400)
    result = oe.dm_test(y, bad, good)
    assert result.iloc[0]["statistic"] > 0 and result.iloc[0]["p_value"] < 1e-6
    symmetric = oe.dm_test(y, good, bad)
    assert_allclose(symmetric.iloc[0]["statistic"], -result.iloc[0]["statistic"])


def test_dm_test_errors(series):
    y = series["y"].to_numpy()
    cases = [
        (lambda: oe.dm_test(y, y + 1, y - 1), "constant_loss_differential"),
        (lambda: oe.dm_test(y, y, y, loss="huber"), "invalid_spec"),
        (lambda: oe.dm_test(y, y, y, kernel="parzen"), "invalid_spec"),
        (lambda: oe.dm_test(y, y, y, horizon=0), "invalid_spec"),
        (lambda: oe.dm_test(y[:3], y[:3], y[:3], horizon=3), "insufficient_observations"),
        (lambda: oe.dm_test(y, y[:-1], y), "length_mismatch"),
        (lambda: oe.dm_test(y, y, y, harvey="yes"), "invalid_spec"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, code


def test_rendering_and_access(series):
    table = oe.fcast_eval("y", "f1", data=series)
    assert "theil_u1" in table.to_string() and "tabular" in str(table.to_latex())
    dm = oe.dm_test("y", "f1", "f2", data=series)
    assert "Diebold" in dm.to_string()
    assert "fcast_eval" in dir(oe) and "dm_test" in dir(oe)


def test_dm_test_missing_policy(series):
    holes = series.copy()
    holes.loc[[3, 10], "f2"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.dm_test("y", "f1", "f2", data=holes)
    assert error.value.code == "missing_values"
    result = oe.dm_test("y", "f1", "f2", data=holes, missing="drop", horizon=2)
    kept = holes.dropna()
    dm, p = dm_numpy(kept["y"].to_numpy(), kept["f1"].to_numpy(), kept["f2"].to_numpy(), 2,
                     "squared", "bartlett", True)
    assert_allclose(result.iloc[0]["statistic"], dm, rtol=1e-11)
    assert result.iloc[0]["n"] == len(series) - 2
    assert any("Excluded 2" in note for note in result.attrs["notes"])
