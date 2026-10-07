"""Independent QR/normal-equation oracles for Pesaran CADF/CIPS."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot.cips import xtcips
from openecon.econometrics.unitroot.cips_tables import GRID, TRUNCATION, VALUES, critical_values


def make_panel(n=12, t=43, seed=4242, rho=1.0):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(n, t - 1)) + rng.uniform(0.4, 1.5, size=(n, 1)) * rng.normal(
        size=(1, t - 1)
    )
    y = np.zeros((n, t))
    for j in range(1, t):
        y[:, j] = rho * y[:, j - 1] + shocks[:, j - 1]
    return pd.DataFrame(
        {"id": np.repeat(np.arange(n), t), "t": np.tile(np.arange(t), n), "y": y.ravel()}
    )


def oracle(data, order, trend, start=None):
    cube = data.sort_values(["id", "t"]).y.to_numpy().reshape(data.id.nunique(), -1)
    mean = cube.mean(axis=0)
    dy = np.diff(cube, axis=1)
    dm = np.diff(mean)
    start = order if start is None else start
    answer = []
    for row in range(len(cube)):
        target = dy[row, start:]
        columns = [cube[row, start:-1], mean[start:-1], dm[start:]]
        columns += [dm[start - j : len(mean) - 1 - j] for j in range(1, order + 1)]
        columns += [dy[row, start - j : len(mean) - 1 - j] for j in range(1, order + 1)]
        if trend != "none":
            columns.append(np.ones(len(target)))
        if trend == "trend":
            columns.append(np.arange(start + 1, len(mean)))
        x = np.column_stack(columns)
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        residual = target - x @ beta
        ssr = residual @ residual
        se = np.sqrt(ssr / (len(target) - len(columns)) * np.linalg.inv(x.T @ x)[0, 0])
        answer.append((beta[0] / se, beta[0], se, ssr, len(target), len(columns)))
    return np.array(answer)


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_all_designs_equal_independent_per_unit_ols(trend, order):
    frame = make_panel()
    expected = oracle(frame, order, trend)
    result = xtcips(frame, "y", "id", "t", lags=order, trend=trend, individual=True)
    np.testing.assert_allclose(result.cadf, expected[:, 0], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result.level_coefficient, expected[:, 1], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result.std_error, expected[:, 2], atol=2e-11, rtol=2e-11)
    assert result.attrs["cips"] == pytest.approx(expected[:, 0].mean(), abs=2e-11)
    clipped = np.clip(expected[:, 0], *TRUNCATION[trend])
    np.testing.assert_allclose(result.cadf_truncated, clipped, atol=2e-11, rtol=2e-11)
    assert result.attrs["statistic"] == pytest.approx(clipped.mean(), abs=2e-11)
    assert result.attrs["p_value"] is None
    assert all(result.nobs == len(frame) // 12 - 1 - order)


@pytest.mark.parametrize("method", ["aic", "bic", "hqic"])
@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
def test_lag_selection_equal_independent_common_holdback(method, trend):
    frame = make_panel(seed=552)
    maximum = 3
    scores = []
    for p in range(maximum + 1):
        fit = oracle(frame, p, trend, start=maximum)
        nobs = fit[0, 4]
        penalty = {"aic": 2, "bic": math.log(nobs), "hqic": 2 * math.log(math.log(nobs))}[method]
        scores.append(nobs * np.log(fit[:, 3] / nobs) + penalty * fit[:, 5])
    chosen = np.argmin(np.array(scores), axis=0)
    result = xtcips(
        frame, "y", "id", "t", lags=method, maxlag=maximum, trend=trend, individual=True
    )
    np.testing.assert_array_equal(result.lags, chosen)
    for p in np.unique(chosen):
        expected = oracle(frame, int(p), trend)
        np.testing.assert_allclose(
            result.cadf.to_numpy()[chosen == p], expected[chosen == p, 0], atol=2e-10, rtol=2e-10
        )


def test_tables_orientation_parenthesized_cells_and_interpolation():
    # Published TableII(b), N20,T30; TableII(c), sparse parentheses at T15,N100.
    assert critical_values("cips", "constant", 20, 30, False) == {
        "1%": -2.38,
        "5%": -2.20,
        "10%": -2.11,
    }
    assert critical_values("cips", "constant", 10, 10, False)["1%"] == -2.97
    assert critical_values("cips", "constant", 10, 10, True)["1%"] == -2.85
    assert (
        critical_values("cips", "none", 100, 10, True)["1%"] == -1.71
    )  # blank bracket retains raw
    assert (
        critical_values("cips", "trend", 100, 15, True)["1%"] == -2.74
    )  # sparse bracket retains raw
    assert critical_values("cips", "trend", 200, 15, True)["1%"] == -2.70
    assert critical_values("cadf", "trend", 20, 30, False)["1%"] == -4.68
    q = critical_values("cips", "constant", 25, 25, False)
    for level in ("1%", "5%", "10%"):
        corners = [
            critical_values("cips", "constant", n, t, False)[level]
            for n in (20, 30)
            for t in (20, 30)
        ]
        assert q[level] == pytest.approx(sum(corners) / 4, abs=1e-15)
    for kind in VALUES:
        for trend in VALUES[kind]:
            for trunc in (False, True):
                variant = "truncated" if trunc else "raw"
                for ti, t in enumerate(GRID):
                    for ni, n in enumerate(GRID):
                        out = critical_values(kind, trend, n, t, trunc)
                        assert out["1%"] <= out["5%"] <= out["10%"]
                        assert all(
                            out[level] == VALUES[kind][trend][level][variant][ti][ni]
                            for level in out
                        )


@pytest.mark.parametrize("n,t", [(9, 31), (12, 10), (201, 31), (12, 202)])
def test_no_extrapolated_quantiles(n, t):
    frame = make_panel(n=n, t=t)
    with pytest.raises(AnalysisError) as e:
        xtcips(frame, "y", "id", "t")
    assert e.value.code == "critical_values_unavailable"
    stat = xtcips(frame, "y", "id", "t", inference="none")
    assert stat.attrs["p_value"] is None
    assert stat.critical_5pct.isna().all() and stat.reject_5pct.isna().all()


def test_input_unchanged_sorting_offsets_and_common_scale():
    frame = make_panel()
    original = frame.copy(deep=True)
    baseline = xtcips(frame, "y", "id", "t", lags=2)
    pd.testing.assert_frame_equal(frame, original)
    altered = frame.copy()
    altered.y = altered.y + altered.id * 1e6
    shifted = xtcips(altered.sample(frac=1, random_state=34), "y", "id", "t", lags=2, block_size=1)
    assert shifted.attrs["statistic"] == pytest.approx(baseline.attrs["statistic"], abs=3e-10)
    altered.y *= 1e110
    assert xtcips(altered, "y", "id", "t", lags=2).attrs["statistic"] == pytest.approx(
        baseline.attrs["statistic"], abs=4e-10
    )


@pytest.mark.parametrize(
    "dtype,first", [("int64", -(2**63) + 3), ("int64", 2**63 - 50), ("uint64", 2**64 - 50)]
)
def test_large_integer_dates_retain_exact_one_step_identity(dtype, first):
    frame = make_panel(t=31)
    frame.t = pd.Series([first + j for j in range(31)] * 12, dtype=dtype)
    actual = xtcips(frame, "y", "id", "t")
    expected = xtcips(make_panel(t=31), "y", "id", "t")
    assert actual.attrs["statistic"] == expected.attrs["statistic"]
    frame.loc[frame.t == first + 15, "t"] = first + 40
    with pytest.raises(AnalysisError) as e:
        xtcips(frame, "y", "id", "t")
    assert e.value.code == "time_gaps"


@pytest.mark.parametrize("freq", ["MS", "B", "D"])
def test_regular_datetime_calendar_and_gaps(freq):
    frame = make_panel(t=31)
    dates = pd.date_range("2000-01-01", periods=31, freq=freq)
    frame.t = np.tile(dates, 12)
    assert math.isfinite(xtcips(frame, "y", "id", "t").attrs["statistic"])
    frame = frame[frame.t != dates[15]]
    with pytest.raises(AnalysisError) as e:
        xtcips(frame, "y", "id", "t")
    assert e.value.code == "time_gaps"


@pytest.mark.parametrize(
    "option,value",
    [
        ("lags", True),
        ("lags", -1),
        ("lags", "foo"),
        ("maxlag", 2),
        ("trend", "foo"),
        ("truncated", 1),
        ("individual", "yes"),
        ("inference", "normal"),
        ("block_size", 0),
        ("memory_mb", 1025),
        ("lags", 100),
    ],
)
def test_invalid_options(option, value):
    with pytest.raises(AnalysisError):
        xtcips(make_panel(), "y", "id", "t", **{option: value})


@pytest.mark.parametrize(
    "defect,code",
    [
        ("missing", "missing_values"),
        ("duplicate", "repeated_time_values"),
        ("unbalanced", "unbalanced_panel"),
        ("constant", "collinear_regressors"),
        ("boolean_time", "invalid_time"),
        ("fraction_time", "invalid_time"),
        ("float_precision", "invalid_time"),
    ],
)
def test_sample_guards(defect, code):
    frame = make_panel()
    if defect == "missing":
        frame.loc[0, "y"] = None
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect == "unbalanced":
        frame = frame.iloc[1:]
    elif defect == "constant":
        frame.y = 3.0
    elif defect == "boolean_time":
        frame.t = True
    elif defect == "fraction_time":
        frame.t = frame.t + 0.25
    elif defect == "float_precision":
        frame.t = frame.t.astype(float) + 2**54
    with pytest.raises(AnalysisError) as e:
        xtcips(frame, "y", "id", "t")
    assert e.value.code == code


def test_metadata_json_latex_no_rng_or_dtype_mutation():
    frame = make_panel()
    rng = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    result = xtcips(frame, "y", "id", "t", lags=1)
    json.dumps(result.attrs, allow_nan=False)
    assert result.attrs["critical_T"] == 42
    assert result.attrs["provenance"]["stata_parity_validated"] is False
    assert "\\begin{tabular}" in result.to_latex(notes=result.attrs["notes"])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert dtype == torch.get_default_dtype()


def test_workspace_guard_and_block_equivalence():
    frame = make_panel(n=30, t=43)
    large = xtcips(frame, "y", "id", "t", lags=2, block_size=128)
    small = xtcips(frame, "y", "id", "t", lags=2, block_size=1, memory_mb=1)
    assert small.attrs["statistic"] == pytest.approx(large.attrs["statistic"], abs=1e-12)
    assert small.attrs["workspace"]["estimated_tensor_workspace_bytes"] <= 1024**2
    with pytest.raises(AnalysisError) as e:
        xtcips(make_panel(t=10001), "y", "id", "t", inference="none", memory_mb=1)
    assert e.value.code == "workspace_too_small"


def test_public_registry_and_xtunitroot_dispatch():
    import openecon as oe

    frame = make_panel()
    expected = xtcips(frame, "y", "id", "t", lags=1)
    actual = oe.xtunitroot(frame, "y", "id", "t", test="cips", lags=1)
    pd.testing.assert_frame_equal(actual, expected)
    individual = oe.xtunitroot(frame, "y", "id", "t", test="cadf", lags=1)
    assert len(individual) == 12
    pd.testing.assert_frame_equal(oe.xtcips(frame, "y", "id", "t", lags=1), expected)
    for option, value in (
        ("demean", True),
        ("kernel_lags", 1),
        ("robust", True),
        ("lrv", "null"),
        ("method", "pperron"),
        ("altt", True),
    ):
        with pytest.raises(AnalysisError) as e:
            oe.xtunitroot(frame, "y", "id", "t", test="cips", **{option: value})
        assert e.value.code == "invalid_option"


def test_cpu_scope_does_not_change_callers_default_device():
    frame = make_panel()
    expected = xtcips(frame, "y", "id", "t")
    with torch.device("meta"):
        actual = xtcips(frame, "y", "id", "t")
        assert torch.empty(0).device.type == "meta"
    pd.testing.assert_frame_equal(actual, expected)
