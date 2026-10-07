"""oe.tsfilter against statsmodels hpfilter / bkfilter / cffilter and explicit Hamilton OLS."""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.hp_filter import hpfilter

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsmodels.filters import hp_trend


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(5)
    n = 240
    y = np.cumsum(rng.normal(size=n)) + 3 * np.sin(np.arange(n) / 4) + 50
    return pd.DataFrame({"y": y, "q": np.arange(n) + 1}).sample(frac=1, random_state=2)


def ordered(df):
    return df.sort_values("q").y.to_numpy()


@pytest.mark.parametrize("smooth", [6.25, 1600.0, 129600.0])
def test_hp_matches_statsmodels(df, smooth):
    out = oe.tsfilter(df, "y", method="hp", time="q", smooth=smooth)
    cycle, trend = hpfilter(ordered(df), smooth)
    assert_allclose(out.cycle, cycle, atol=1e-8 * np.abs(ordered(df)).max())
    assert_allclose(out.trend, trend, rtol=1e-10)
    assert_allclose(out.observed, ordered(df))
    assert list(out.period) == list(range(1, len(df) + 1))
    assert out.attrs["smooth"] == smooth and out.attrs["method"] == "hp"


def test_hp_solves_the_normal_equations():
    y = np.random.default_rng(1).normal(size=50).cumsum()
    import torch
    tau = hp_trend(torch.tensor(y), 100.0).numpy()
    K = np.diff(np.eye(50), n=2, axis=0)
    assert_allclose((np.eye(50) + 100.0 * K.T @ K) @ tau, y, atol=1e-10)


@pytest.mark.parametrize("low,high,order", [(6, 32, 12), (2, 8, 3), (18, 96, 36)])
def test_bk_matches_statsmodels(df, low, high, order):
    out = oe.tsfilter(df, "y", method="bk", time="q", minperiod=low, maxperiod=high,
                      smaorder=order)
    reference = bkfilter(ordered(df), low, high, order)
    assert out.cycle.isna().sum() == 2 * order
    assert_allclose(out.cycle.to_numpy()[order:-order], reference, atol=1e-11)
    assert out.attrs["lost_start"] == order and out.attrs["lost_end"] == order


@pytest.mark.parametrize("drift", [True, False])
@pytest.mark.parametrize("low,high", [(6, 32), (2, 12)])
def test_cf_matches_statsmodels(df, drift, low, high):
    out = oe.tsfilter(df, "y", method="cf", time="q", minperiod=low, maxperiod=high, drift=drift)
    cycle, _ = cffilter(ordered(df), low, high, drift)
    assert_allclose(out.cycle, cycle, atol=1e-10)
    assert_allclose(out.trend + out.cycle, ordered(df), rtol=1e-13)


@pytest.mark.parametrize("h,p", [(8, 4), (2, 1), (24, 12)])
def test_hamilton_matches_explicit_ols(df, h, p):
    y = ordered(df)
    n = len(y)
    out = oe.tsfilter(df, "y", method="hamilton", time="q", hamilton_h=h, hamilton_p=p)
    lost = h + p - 1
    t = np.arange(p - 1, n - h)
    X = np.column_stack([np.ones(len(t))] + [y[t - j] for j in range(p)])
    beta = np.linalg.lstsq(X, y[t + h], rcond=None)[0]
    assert out.cycle.isna().sum() == lost
    assert_allclose(out.cycle.to_numpy()[lost:], y[t + h] - X @ beta, atol=1e-9)
    assert_allclose(out.trend.to_numpy()[lost:], X @ beta, rtol=1e-11)
    assert_allclose(list(out.attrs["coefficients"].values()), beta, rtol=1e-8)


def test_errors_and_attrs(df):
    for kwargs, code in [({"method": "bw"}, "invalid_option"),
                         ({"method": "hp", "smooth": -1.0}, "invalid_option"),
                         ({"method": "bk", "minperiod": 32, "maxperiod": 6}, "invalid_option"),
                         ({"method": "bk", "minperiod": 1.5}, "invalid_option"),
                         ({"method": "bk", "smaorder": 200}, "insufficient_observations"),
                         ({"method": "cf", "drift": "yes"}, "invalid_option"),
                         ({"method": "hamilton", "hamilton_h": 0}, "invalid_option")]:
        with pytest.raises(AnalysisError) as err:
            oe.tsfilter(df, "y", time="q", **kwargs)
        assert err.value.code == code
    holed = df.copy()
    holed.iloc[3, 0] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.tsfilter(holed, "y", time="q")
    assert err.value.code == "missing_values"
    with pytest.raises(AnalysisError) as err:
        oe.tsfilter(df.assign(q=df.q * 2), "y", time="q")
    assert err.value.code == "time_gaps"
    with pytest.raises(AnalysisError) as err:
        oe.tsfilter(df, "nope")
    assert err.value.code == "missing_columns"
    out = oe.tsfilter(df, "y", method="hamilton")
    json.dumps(out.attrs)
    assert list(out.columns) == ["period", "observed", "trend", "cycle"]
    assert "Hamilton" in out.attrs["title"]
    dated = df.assign(q=pd.date_range("2000-01-01", periods=len(df), freq="QS").to_numpy()[
        df.q.to_numpy() - 1])
    out = oe.tsfilter(dated, "y", time="q")
    assert_allclose(out.cycle, hpfilter(ordered(df), 1600.0)[0], atol=1e-8 * 100)
