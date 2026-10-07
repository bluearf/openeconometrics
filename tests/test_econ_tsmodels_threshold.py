"""oe.threshold against explicit NumPy grid searches and statsmodels OLS."""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 240
    q = rng.normal(size=n)
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    y = np.where(q <= -0.5, 2 - x, np.where(q <= 0.6, 0.5 * x, -1 + 1.5 * x)) + 0.7 * w \
        + rng.normal(size=n) * (1 + 0.3 * (q > 0))
    return pd.DataFrame({"y": y, "x": x, "w": w, "q": q, "t": np.arange(n)})


def design(df, thresholds, varying, common):
    q = df.q.to_numpy()
    region = sum((q > g).astype(int) for g in thresholds)
    cols = [df[c].to_numpy() if c != "Intercept" else np.ones(len(df)) for c in common]
    for r in range(len(thresholds) + 1):
        ind = (region == r).astype(float)
        cols += [ind * (df[c].to_numpy() if c != "Intercept" else 1.0) for c in varying]
    return np.column_stack(cols)


def ssr_of(df, thresholds, varying, common):
    X = design(df, thresholds, varying, common)
    y = df.y.to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return e @ e


def admissible(df, fixed, trim, extra=0):
    q = np.sort(df.q.to_numpy())
    n = len(q)
    minimum = max(math.ceil(trim * n), extra)
    positions = [i for i in range(1, n) if q[i] > q[i - 1]]
    bounds = sorted({0, n, *fixed})
    return [i for i in positions if all(abs(i - b) >= minimum for b in bounds)], q


def numpy_search(df, fixed_thresholds, varying, common, trim=0.1):
    q = np.sort(df.q.to_numpy())
    fixed = [int(np.searchsorted(q, g, side="right")) for g in fixed_thresholds]
    candidates, q = admissible(df, fixed, trim, len(varying) + 1)
    values = [(ssr_of(df, sorted(fixed_thresholds + [q[i - 1]]), varying, common), q[i - 1])
              for i in candidates]
    return min(values), values


@pytest.mark.parametrize("varying,common", [(["Intercept", "x", "w"], []),
                                            (["Intercept", "x"], ["w"]), (["x"], ["Intercept", "w"])])
def test_single_threshold_matches_brute_force(df, varying, common):
    fit = oe.threshold(data=df, y="y", x=["x", "w"], threshold_var="q", regions=varying)
    (best, gamma), _ = numpy_search(df, [], varying, common)
    assert_allclose(fit.extra["thresholds"], [gamma])
    assert_allclose(fit.metrics["ssr"], best, rtol=1e-10)
    X = design(df, [gamma], varying, common)
    ref = sm.OLS(df.y.to_numpy(), X).fit()
    assert_allclose([c.estimate for c in fit.coefficients], ref.params, rtol=1e-8, atol=1e-10)
    assert_allclose([c.std_error for c in fit.coefficients], ref.bse, rtol=1e-8)
    assert fit.inference["df_inference"] == len(df) - X.shape[1]
    expected = common + [f"region{r}:{c}" for r in (1, 2) for c in varying]
    assert [c.term for c in fit.coefficients] == expected
    n = len(df)
    assert_allclose(fit.metrics["bic"], n * np.log(best / n) + X.shape[1] * np.log(n))


@pytest.mark.parametrize("cov", ["robust", "HC1", "HC2", "HC3"])
def test_robust_covariances(df, cov):
    fit = oe.threshold(data=df, y="y", x=["x", "w"], threshold_var="q", covariance=cov)
    X = design(df, fit.extra["thresholds"], ["Intercept", "x", "w"], [])
    ref = sm.OLS(df.y.to_numpy(), X).fit(cov_type="HC1" if cov == "robust" else cov)
    assert_allclose([c.std_error for c in fit.coefficients], ref.bse, rtol=1e-8)


def test_two_thresholds_sequential_with_refinement(df):
    varying, common = ["Intercept", "x"], ["w"]
    fit = oe.threshold(data=df, y="y", x=["x", "w"], threshold_var="q", regions=varying,
                       nthresholds=2)
    (_, first), _ = numpy_search(df, [], varying, common)
    (_, second), _ = numpy_search(df, [first], varying, common)
    thresholds = sorted([first, second])
    for _ in range(10):                                 # Bai (1997) refinement, NumPy version
        changed = False
        for index in range(2):
            others = thresholds[:index] + thresholds[index + 1:]
            (_, new), _ = numpy_search(df, others, varying, common)
            if new != thresholds[index]:
                thresholds = sorted(others + [new])
                changed = True
        if not changed:
            break
    assert_allclose(fit.extra["thresholds"], thresholds)
    assert_allclose(fit.metrics["ssr"], ssr_of(df, thresholds, varying, common), rtol=1e-10)
    assert min(fit.extra["region_sizes"]) >= math.ceil(0.1 * len(df))
    assert [row["thresholds"] for row in fit.extra["by_number_of_thresholds"]] == [0, 1, 2]
    assert_allclose(fit.extra["thresholds"], [-0.5, 0.6], atol=0.15)


def test_confidence_set_and_ties():
    rng = np.random.default_rng(7)
    n = 300
    q = rng.integers(0, 40, size=n).astype(float)               # many ties
    x = rng.normal(size=n)
    y = 0.3 * (q > 20) + x + rng.normal(size=n)
    data = pd.DataFrame({"y": y, "x": x, "q": q})
    fit = oe.threshold(data=data, y="y", x=["x"], threshold_var="q", trim=0.15)
    (best, gamma), values = numpy_search(data, [], ["Intercept", "x"], [], trim=0.15)
    assert_allclose(fit.extra["thresholds"], [gamma])
    lr = np.array([n * (v - best) / best for v, _ in values])
    accepted = np.array([g for _, g in values])[lr <= -2 * np.log(1 - np.sqrt(0.95))]
    band = fit.extra["threshold_confidence_set"]
    assert_allclose([band["low"], band["high"]], [accepted.min(), accepted.max()])
    assert fit.extra["candidates"] == len(values)
    assert fit.extra["thresholds"][0] in set(q)


def test_bootstrap_matches_numpy_replication(df):
    fit = oe.threshold(data=df, y="y", x=["x"], threshold_var="q", bootstrap=30, seed=5)
    test = fit.tests["threshold_effect"]
    varying, common = ["Intercept", "x"], []
    n = len(df)
    (s1, gamma), _ = numpy_search(df, [], varying, common)
    X0 = np.column_stack([np.ones(n), df.x])
    y = df.y.to_numpy()
    s0 = np.sum((y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]) ** 2)
    statistic = n * (s0 - s1) / s1
    assert_allclose(test["statistic"], statistic, rtol=1e-9)
    order = np.argsort(df.q.to_numpy(), kind="stable")
    resid = (y - design(df, [gamma], varying, common)
             @ np.linalg.lstsq(design(df, [gamma], varying, common), y, rcond=None)[0])
    draws = torch.randn((n, 30), generator=torch.Generator().manual_seed(5),
                        dtype=torch.float64).numpy()
    exceed = 0
    frame = df.iloc[order].reset_index(drop=True)
    for r in range(30):
        star = frame.assign(y=draws[:, r] * resid[order])
        null = np.sum((star.y - X0[order] @ np.linalg.lstsq(X0[order], star.y, rcond=None)[0]) ** 2)
        (alt, _), _ = numpy_search(star, [], varying, common)
        exceed += n * (null - alt) / alt >= statistic
    assert_allclose(test["p_value"], exceed / 30)
    assert test["reps"] == 30 and test["distribution"] == "bootstrap"


def test_errors_missing_and_rendering(df):
    for kwargs, code in [({"trim": 0.6}, "invalid_option"), ({"regions": ["z"]}, "invalid_option"),
                         ({"regions": []}, "invalid_option"),
                         ({"nthresholds": 5, "trim": 0.2}, "insufficient_observations"),
                         ({"nthresholds": 9}, "invalid_spec")]:
        with pytest.raises(AnalysisError) as err:
            oe.threshold(data=df, y="y", x=["x"], threshold_var="q", **kwargs)
        assert err.value.code == code
    with pytest.raises(AnalysisError) as err:
        oe.threshold(data=df, y="y", x=["x"], threshold_var="nope")
    assert err.value.code == "missing_columns"
    holed = df.copy()
    holed.loc[10, "q"] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.threshold(data=holed, y="y", x=["x"], threshold_var="q")
    assert err.value.code == "missing_values"
    fit = oe.threshold(data=holed, y="y", x=["x"], threshold_var="q", missing="drop", time="t")
    assert fit.nobs == len(df) - 1
    tar = oe.threshold(data=df, y="y", x=["q", "x"], threshold_var="q")
    assert "region1:q" in [c.term for c in tar.coefficients]
    assert type(fit).model_validate_json(fit.model_dump_json()) == fit
    assert "region2:x" in fit.summary() and "region1" in fit.to_latex()
    assert fit.provenance["stata_parity_validated"] is False
