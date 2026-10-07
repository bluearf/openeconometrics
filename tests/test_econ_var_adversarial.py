"""Hostile inputs for the var family: every call works or raises a helpful AnalysisError.

Part 3 of the verification suite. Each case is either fitted (and its result
is then checked to be finite and JSON-clean) or refused with a stable error
code; nothing may escape as a raw exception or as NaN/inf in a result. The
regression tests at the end pin the defects fixed during verification.
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle

N = 120


def stationary():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(N, 3))
    y = np.zeros((N, 3))
    a = np.array([[0.5, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.2, 0.4]])
    for t in range(1, N):
        y[t] = 0.2 + a @ y[t - 1] + e[t]
    frame = pd.DataFrame(y, columns=["a", "b", "c"])
    frame["t"] = np.arange(N)
    frame["x"] = rng.normal(size=N)
    frame["g"] = np.array(["u", "v", "w"])[np.arange(N) % 3]
    return frame


def integrated():
    rng = np.random.default_rng(1)
    walk = rng.normal(size=(N, 3)).cumsum(axis=0)
    frame = pd.DataFrame(walk, columns=["a", "b", "c"])
    frame["b"] = 0.5 * frame["a"] + rng.normal(size=N)
    frame["t"] = np.arange(N)
    return frame


BASE, WALK = stationary(), integrated()
RNG = np.random.default_rng(5)


def check_clean(out):
    """A returned object is finite, JSON-serializable and free of NaN / Infinity tokens."""
    if isinstance(out, ResultBundle):
        text = out.model_dump_json()
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)
        assert all(math.isfinite(c.estimate) and c.std_error > 0 for c in out.coefficients)
        assert all(value is None or isinstance(value, str) or math.isfinite(value)
                   for value in out.metrics.values())
    else:
        values = out.select_dtypes("number").to_numpy(dtype=float)
        assert not np.isinf(values).any()
        json.dumps(out.attrs)


def run(function, expected, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if expected == "ok":
            check_clean(function(**options))
            return
        with pytest.raises(AnalysisError) as error:
            function(**options)
    codes = {expected} if isinstance(expected, str) else set(expected)
    assert error.value.code in codes, (error.value.code, str(error.value))
    assert len(str(error.value)) > 20                    # says what is wrong and what to change


VAR_CASES = {
    "empty table": (dict(data=BASE.iloc[:0], y=["a", "b"]), "empty_data"),
    "one row": (dict(data=BASE.iloc[:1], y=["a", "b"]), "insufficient_observations"),
    "n = lags + 1": (dict(data=BASE.iloc[:3], y=["a", "b"]), "insufficient_observations"),
    "T <= parameters": (dict(data=BASE.iloc[:5], y=["a", "b", "c"], lags=1),
                        "insufficient_observations"),
    "T = parameters + 1": (dict(data=BASE.iloc[:6], y=["a", "b", "c"], lags=1),
                           "singular_residual_covariance"),
    "smallest feasible sample": (dict(data=BASE.iloc[:7], y=["a", "b"], lags=1), "ok"),
    "small with two degrees of freedom": (dict(data=BASE.iloc[:9], y=["a", "b"], small=True),
                                          "ok"),
    "all-missing column": (dict(data=BASE.assign(a=np.nan), y=["a", "b"]), "missing_values"),
    "all-missing column, drop": (dict(data=BASE.assign(a=np.nan), y=["a", "b"], missing="drop"),
                                 "empty_sample"),
    "constant outcome": (dict(data=BASE.assign(a=3.0), y=["a", "b"]), "collinear_system"),
    "constant outcome, no constant": (dict(data=BASE.assign(a=3.0), y=["a", "b"],
                                           constant=False), "collinear_system"),
    "zero outcome": (dict(data=BASE.assign(a=0.0), y=["a", "b"]), "collinear_system"),
    "perfectly collinear variables": (dict(data=BASE.assign(d=BASE.a + BASE.b),
                                           y=["a", "b", "d"]), "collinear_system"),
    "text in an endogenous column": (dict(data=BASE.assign(a=BASE.a.astype(str)), y=["a", "b"]),
                                     "non_numeric_column"),
    "infinite value": (dict(data=BASE.assign(a=np.where(np.arange(N) == 5, np.inf, BASE.a)),
                            y=["a", "b"]), "non_finite_values"),
    "magnitude 1e-8": (dict(data=BASE.assign(a=BASE.a * 1e-8, b=BASE.b * 1e-8), y=["a", "b"]),
                       "ok"),
    "magnitude 1e8": (dict(data=BASE.assign(a=BASE.a * 1e8, b=BASE.b * 1e8), y=["a", "b"]),
                      "ok"),
    "magnitudes 1e-8 and 1e8, robust": (dict(data=BASE.assign(a=BASE.a * 1e-8, b=BASE.b * 1e8),
                                             y=["a", "b"], covariance="robust"), "ok"),
    "level 1e8": (dict(data=BASE.assign(a=BASE.a + 1e8), y=["a", "b"]), "ok"),
    "level 1e8 without a constant": (dict(data=BASE.assign(a=BASE.a + 1e8), y=["a", "b"],
                                          constant=False), "collinear_system"),
    "gap in time": (dict(data=BASE.drop(index=[10]), y=["a", "b"], time="t"), "time_gaps"),
    "duplicate time": (dict(data=pd.concat([BASE, BASE.iloc[[3]]]), y=["a", "b"], time="t"),
                       "repeated_time_values"),
    "fractional time": (dict(data=BASE.assign(t=BASE.t + 0.5), y=["a", "b"], time="t"),
                        "invalid_time"),
    "text time": (dict(data=BASE.assign(t=BASE.t.astype(str)), y=["a", "b"], time="t"),
                  "invalid_time"),
    "missing time at the start, drop": (dict(data=BASE.assign(
        t=np.where(np.arange(N) == 0, np.nan, BASE.t)), y=["a", "b"], time="t", missing="drop"),
        "ok"),
    "missing value inside, drop": (dict(data=BASE.assign(
        a=np.where(np.arange(N) == 60, np.nan, BASE.a)), y=["a", "b"], missing="drop"),
        "time_gaps"),
    "rows in reverse order": (dict(data=BASE.iloc[::-1], y=["a", "b"], time="t"), "ok"),
    "more lags than the sample allows": (dict(data=BASE, y=["a", "b"], lags=59),
                                         "insufficient_observations"),
    "lags = True": (dict(data=BASE, y=["a", "b"], lags=True), "invalid_spec"),
    "lags as text": (dict(data=BASE, y=["a", "b"], lags="2"), "invalid_spec"),
    "lags = 0": (dict(data=BASE, y=["a", "b"], lags=0), "invalid_spec"),
    "negative lags": (dict(data=BASE, y=["a", "b"], lags=-1), "invalid_spec"),
    "maxlag = 0": (dict(data=BASE, y=["a", "b"], maxlag=0), "ok"),
    "maxlag beyond the sample": (dict(data=BASE, y=["a", "b"], maxlag=100), "ok"),
    "irf_steps at its bound": (dict(data=BASE, y=["a", "b"], irf_steps=200), "ok"),
    "irf_steps above its bound": (dict(data=BASE, y=["a", "b"], irf_steps=201), "invalid_spec"),
    "no stored responses": (dict(data=BASE, y=["a", "b"], irf_kinds=[]), "ok"),
    "unknown response kind": (dict(data=BASE, y=["a", "b"], irf_kinds=["structural"]),
                              "invalid_option"),
    "lm_lags at its bound": (dict(data=BASE, y=["a", "b"], lm_lags=24), "ok"),
    "lm_lags above its bound": (dict(data=BASE, y=["a", "b"], lm_lags=25), "invalid_spec"),
    "alpha = 0": (dict(data=BASE, y=["a", "b"], alpha=0.0), "invalid_spec"),
    "alpha = 1": (dict(data=BASE, y=["a", "b"], alpha=1.0), "invalid_spec"),
    "constant exogenous column": (dict(data=BASE.assign(one=1.0), y=["a", "b"], x=["one"]),
                                  "ok"),
    "exogenous lag of an endogenous variable": (dict(
        data=BASE.assign(la=BASE.a.shift(1).fillna(0.0)), y=["a", "b"], x=["la"], lags=1),
        "collinear_system"),
    "text exogenous column": (dict(data=BASE, y=["a", "b"], x=["g"]), "non_numeric_column"),
    "categorical exogenous column": (dict(data=BASE, y=["a", "b"], x=["g"], categorical=["g"]),
                                     "ok"),
    "categorical that is not a regressor": (dict(data=BASE, y=["a", "b"], x=["x"],
                                                 categorical=["g"]), "invalid_spec"),
    "weights are not supported": (dict(data=BASE, y=["a", "b"], covariance="cluster"),
                                  "invalid_spec"),
    "explosive process": (dict(data=pd.DataFrame({
        "a": 1.5 ** np.arange(60) + RNG.normal(size=60), "b": RNG.normal(size=60)}),
        y=["a", "b"], lags=1), "ok"),
    "deterministic exponential": (dict(data=pd.DataFrame({
        "a": 3.0 ** np.arange(100.0), "b": RNG.normal(size=100)}), y=["a", "b"], lags=1),
        "perfect_fit"),
    "deterministic trend as a variable": (dict(data=pd.DataFrame({
        "a": np.arange(50.0), "b": RNG.normal(size=50)}), y=["a", "b"], lags=1), "perfect_fit"),
    "identity among the variables": (dict(data=BASE.assign(
        d=BASE.a.shift(1).fillna(0.0) * 2.0 + 1.0), y=["a", "d"], lags=1),
        {"perfect_fit", "collinear_system"}),
    "dict of lists": (dict(data={"a": list(BASE.a), "b": list(BASE.b)}, y=["a", "b"]), "ok"),
    "integer columns": (dict(data=pd.DataFrame({"a": RNG.integers(0, 10, 100),
                                                "b": RNG.integers(0, 10, 100)}), y=["a", "b"]),
                        "ok"),
    "y as a tuple": (dict(data=BASE, y=("a", "b")), "ok"),
    "y as a string": (dict(data=BASE, y="a"), "invalid_spec"),
    "y empty": (dict(data=BASE, y=[]), "invalid_spec"),
    "y with a non-string": (dict(data=BASE, y=["a", 3]), "invalid_spec"),
    "unknown column": (dict(data=BASE, y=["a", "zz"]), "missing_columns"),
    "trend with a short sample": (dict(data=BASE.iloc[:12], y=["a", "b"], lags=1, trend=True),
                                  "ok"),
}


@pytest.mark.parametrize("label", list(VAR_CASES))
def test_var_hostile_inputs(label):
    options, expected = VAR_CASES[label]
    run(oe.var, expected, **options)


VARSOC_CASES = {
    "plain": (dict(data=BASE, y=["a", "b"], maxlag=4), "ok"),
    "maxlag = 0": (dict(data=BASE, y=["a", "b"], maxlag=0), "ok"),
    "maxlag too large": (dict(data=BASE, y=["a", "b"], maxlag=58), "insufficient_observations"),
    "maxlag beyond the sample": (dict(data=BASE, y=["a", "b"], maxlag=500),
                                 "insufficient_observations"),
    "maxlag as text": (dict(data=BASE, y=["a", "b"], maxlag="3"), "invalid_lags"),
    "maxlag fractional": (dict(data=BASE, y=["a", "b"], maxlag=2.5), "invalid_lags"),
    "collinear variables": (dict(data=BASE.assign(d=BASE.a * 2), y=["a", "d"], maxlag=2),
                            {"singular_residual_covariance", "collinear_system", "perfect_fit"}),
    "deterministic variable": (dict(data=pd.DataFrame({
        "a": np.arange(60.0), "b": RNG.normal(size=60)}), y=["a", "b"], maxlag=2),
        {"perfect_fit", "singular_residual_covariance"}),
    "alpha out of range": (dict(data=BASE, y=["a", "b"], alpha=1), "invalid_alpha"),
    "gap in time": (dict(data=BASE.drop(index=[30]), y=["a", "b"], time="t"), "time_gaps"),
}


@pytest.mark.parametrize("label", list(VARSOC_CASES))
def test_varsoc_hostile_inputs(label):
    options, expected = VARSOC_CASES[label]
    run(oe.varsoc, expected, **options)


VECRANK_CASES = {
    "plain": (dict(data=WALK, y=["a", "b", "c"]), "ok"),
    "one lag": (dict(data=WALK, y=["a", "b", "c"], lags=1), "ok"),
    "lags = 0": (dict(data=WALK, y=["a", "b", "c"], lags=0), "invalid_spec"),
    "too many lags": (dict(data=WALK, y=["a", "b", "c"], lags=60), "insufficient_observations"),
    "a single variable": (dict(data=WALK, y=["a"]), "ok"),
    "tiny sample": (dict(data=WALK.iloc[:8], y=["a", "b"], lags=2), "insufficient_observations"),
    "one row": (dict(data=WALK.iloc[:1], y=["a", "b"]), "insufficient_observations"),
    "constant column": (dict(data=WALK.assign(a=1.0), y=["a", "b"]),
                        {"collinear_system", "singular_residual_covariance"}),
    "deterministic trend column": (dict(data=WALK.assign(a=np.arange(N) * 1.0), y=["a", "b"]),
                                   {"collinear_system", "singular_residual_covariance"}),
    "stationary data": (dict(data=BASE, y=["a", "b", "c"]), "ok"),
    "max_rank beyond K": (dict(data=WALK, y=["a", "b", "c"], max_rank=10), "ok"),
    "max_rank fractional": (dict(data=WALK, y=["a", "b", "c"], max_rank=1.5), "invalid_option"),
    "unknown trend": (dict(data=WALK, y=["a", "b", "c"], trend="quadratic"), "invalid_spec"),
    "magnitudes 1e8 and 1e-8": (dict(data=WALK.assign(a=WALK.a * 1e8, b=WALK.b * 1e-8),
                                     y=["a", "b", "c"]), "ok"),
    "level 1e9": (dict(data=WALK.assign(a=WALK.a + 1e9), y=["a", "b", "c"]), "ok"),
    "level 1e9, restricted constant": (dict(data=WALK.assign(a=WALK.a + 1e9), y=["a", "b", "c"],
                                            trend="rconstant"), "ok"),
    "text column": (dict(data=WALK.assign(a="x"), y=["a", "b"]), "non_numeric_column"),
    "missing inside": (dict(data=WALK.assign(a=np.where(np.arange(N) == 50, np.nan, WALK.a)),
                            y=["a", "b"], missing="drop"), "time_gaps"),
}


@pytest.mark.parametrize("label", list(VECRANK_CASES))
def test_vecrank_hostile_inputs(label):
    options, expected = VECRANK_CASES[label]
    run(oe.vecrank, expected, **options)


VEC_CASES = {
    "rank = K": (dict(data=WALK, y=["a", "b", "c"], rank=3), "invalid_rank"),
    "rank = 0": (dict(data=WALK, y=["a", "b", "c"], rank=0), "invalid_spec"),
    "rank = True": (dict(data=WALK, y=["a", "b", "c"], rank=True), "invalid_spec"),
    "one variable": (dict(data=WALK, y=["a"]), "invalid_spec"),
    "alpha out of range": (dict(data=WALK, y=["a", "b", "c"], alpha=5), "invalid_spec"),
    "no LM test": (dict(data=WALK, y=["a", "b", "c"], lm_lags=0), "ok"),
    "stationary data, rank 2": (dict(data=BASE, y=["a", "b", "c"], rank=2), "ok"),
    "robust covariance is not offered": (dict(data=WALK, y=["a", "b", "c"],
                                                   covariance="robust"), "invalid_spec"),
    "magnitudes 1e8 and 1e-8": (dict(data=WALK.assign(a=WALK.a * 1e8, b=WALK.b * 1e-8),
                                     y=["a", "b", "c"]), "ok"),
    "level 1e9": (dict(data=WALK.assign(a=WALK.a + 1e9), y=["a", "b", "c"]), "ok"),
    "level 1e9, restricted constant": (dict(data=WALK.assign(a=WALK.a + 1e9), y=["a", "b", "c"],
                                            trend="rconstant"), "ok"),
    "collinear variables": (dict(data=WALK.assign(d=WALK.a * 3.0), y=["a", "b", "d"]),
                            {"collinear_system", "singular_residual_covariance"}),
    "gap in time": (dict(data=WALK.drop(index=[40]), y=["a", "b", "c"], time="t"), "time_gaps"),
}
for _trend in ("none", "rconstant", "constant", "rtrend", "trend"):
    VEC_CASES[f"{_trend}: default"] = (dict(data=WALK, y=["a", "b", "c"], trend=_trend), "ok")
    VEC_CASES[f"{_trend}: one lag, rank 2"] = (dict(data=WALK, y=["a", "b", "c"], trend=_trend,
                                                    lags=1, rank=2), "ok")
    VEC_CASES[f"{_trend}: twelve observations"] = (dict(data=WALK.iloc[:12], y=["a", "b"],
                                                        trend=_trend), "ok")
    VEC_CASES[f"{_trend}: too few observations"] = (dict(data=WALK.iloc[:7], y=["a", "b"],
                                                         trend=_trend),
                                                    "insufficient_observations")


@pytest.mark.parametrize("label", list(VEC_CASES))
def test_vec_hostile_inputs(label):
    options, expected = VEC_CASES[label]
    run(oe.vec, expected, **options)


def test_post_estimation_hostile_inputs():
    fitted = oe.var(data=BASE, y=["a", "b"], lags=2)
    model = oe.vec(data=WALK, y=["a", "b", "c"], lags=2, rank=1)
    cases = [
        (oe.forecast, (fitted, 5), {}, "ok"), (oe.forecast, (fitted, 0), {}, "invalid_steps"),
        (oe.forecast, (fitted, 2.5), {}, "invalid_steps"),
        (oe.forecast, (fitted, True), {}, "invalid_steps"),
        (oe.forecast, (fitted, 5000), {}, "ok"), (oe.forecast, (fitted, 5001), {}, "invalid_steps"),
        (oe.forecast, (fitted, 3), dict(data=BASE.iloc[:1]), "insufficient_observations"),
        (oe.forecast, (fitted, 3), dict(data=BASE[["a"]]), "missing_columns"),
        (oe.forecast, (fitted, 3), dict(data=BASE.assign(a=np.nan)), "missing_values"),
        (oe.forecast, (fitted, 3), dict(alpha=0), "invalid_alpha"),
        (oe.forecast, (fitted, 3), dict(exog=BASE[["x"]].iloc[:3]), "invalid_exog"),
        (oe.forecast, (None, 3), {}, "invalid_result"),
        (oe.var_forecast, (model, 3), {}, "invalid_result"),
        (oe.vec_forecast, (fitted, 3), {}, "invalid_result"),
        (oe.forecast, (model, 5), {}, "ok"), (oe.forecast, (model, 5000), {}, "ok"),
        (oe.forecast, (model, 3), dict(data=WALK.iloc[:1]), "insufficient_observations"),
        (oe.irf, (fitted,), dict(steps=500), "ok"), (oe.irf, (fitted,), dict(steps=501),
                                                     "invalid_steps"),
        (oe.irf, (fitted,), dict(steps="3"), "invalid_steps"),
        (oe.irf, (fitted,), dict(kind="cumulative"), "invalid_option"),
        (oe.irf, (fitted,), dict(alpha=1.0), "invalid_alpha"),
        (oe.irf, (None,), {}, "invalid_result"),
        (oe.irf, (model,), dict(steps=20, kind="generalized"), "ok"),
    ]
    for function, arguments, options, expected in cases:
        run(lambda f=function, a=arguments, o=options: f(*a, **o), expected)


# ---- regression tests for the defects fixed during verification -------------------------------

def test_an_exactly_fitted_equation_is_refused_instead_of_reporting_rounding_noise():
    """Before the fix: standard errors of 3e-17 and a log likelihood of 1500 were returned."""
    frame = pd.DataFrame({"a": np.arange(50.0), "b": np.random.default_rng(0).normal(size=50)})
    with pytest.raises(AnalysisError) as error:
        oe.var(data=frame, y=["a", "b"], lags=1)
    assert error.value.code == "perfect_fit"
    assert "a" in str(error.value) and "exogenous" in str(error.value)
    with pytest.raises(AnalysisError) as error:
        oe.varsoc(data=frame, y=["a", "b"], maxlag=2)
    assert error.value.code == "perfect_fit"
    # A noisy trend is an ordinary (unstable) VAR, not a perfect fit.
    noisy = frame.assign(a=frame.a + np.random.default_rng(1).normal(size=50) * 1e-3)
    check_clean(oe.var(data=noisy, y=["a", "b"], lags=1))
    # An equation that becomes exact only at a longer lag does not stop the fit itself.
    rng = np.random.default_rng(2)
    b = rng.normal(size=80)
    a = np.zeros(80)
    a[3:] = 0.5 * b[:-3]                                  # a_t = 0.5 b_(t-3) exactly
    a[:3] = rng.normal(size=3)
    result = oe.var(data=pd.DataFrame({"a": a, "b": b}), y=["a", "b"], lags=1, maxlag=4)
    assert "lag_order_selection" not in result.extra
    assert any("Lag-order selection statistics were not computed" in w for w in result.warnings)


def test_restricted_constant_is_accurate_for_series_with_a_large_level():
    """Before the fix a level of 1e6 changed the log likelihood by 0.06 and 1e9 was refused."""
    frame = integrated()[["a", "b", "c"]]
    base = oe.vec(data=frame, y=["a", "b", "c"], lags=3, rank=1, trend="rconstant")
    for level in (1e6, 1e9):
        moved = oe.vec(data=frame + level, y=["a", "b", "c"], lags=3, rank=1, trend="rconstant")
        tolerance = 1e-9 if level == 1e6 else 1e-5
        assert moved.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                                abs=tolerance * 100)
        assert_allclose(moved.extra["beta_matrix"], base.extra["beta_matrix"], rtol=tolerance,
                        atol=tolerance)
        assert_allclose(moved.extra["alpha"], base.extra["alpha"], rtol=tolerance * 10,
                        atol=tolerance)
        beta = np.array(base.extra["beta_matrix"])[:, 0]
        constant = base.extra["ce_constant"][0] - beta.sum() * level
        assert moved.extra["ce_constant"][0] == pytest.approx(constant, rel=tolerance)
        # The standard error of the restricted constant grows with the level, as it must.
        entry = moved.extra["beta"][0]["coefficients"][-1]
        assert entry["variable"] == "_cons" and entry["std_error"] > 1e3
        table = oe.vecrank(data=frame + level, y=["a", "b", "c"], lags=3, trend="rconstant")
        reference = oe.vecrank(data=frame, y=["a", "b", "c"], lags=3, trend="rconstant")
        assert_allclose(table["trace"].to_numpy()[:3], reference["trace"].to_numpy()[:3],
                        rtol=tolerance * 100)


def test_johansen_normalization_does_not_depend_on_the_units():
    """Before the fix variables 16 orders of magnitude apart were 'not normalizable'."""
    frame = integrated()[["a", "b", "c"]]
    scale = np.array([1e8, 1.0, 1e-8])
    base = oe.vec(data=frame, y=["a", "b", "c"], lags=2, rank=1)
    other = oe.vec(data=frame * scale, y=["a", "b", "c"], lags=2, rank=1)
    b0, b1 = np.array(base.extra["beta_matrix"])[:, 0], np.array(other.extra["beta_matrix"])[:, 0]
    assert_allclose(b1 * scale / scale[0], b0, rtol=1e-8)
    assert abs(b1[2]) > 1e12                              # a legitimate, very large coefficient
    assert_allclose(np.array(other.extra["alpha"])[:, 0] / scale * scale[0],
                    np.array(base.extra["alpha"])[:, 0], rtol=1e-8)
    # Rank 2 in such units is either fitted cleanly or refused with a message that says why.
    check_clean(oe.vec(data=frame * scale, y=["a", "b", "c"], lags=2, rank=2, trend="rconstant"))
    try:
        check_clean(oe.vec(data=frame * scale, y=["a", "b", "c"], lags=2, rank=2))
    except AnalysisError as exc:
        assert exc.code == "singular_adjustment"
    from openecon.econometrics.var.vec import _project

    with pytest.raises(AnalysisError) as error:           # alpha without full column rank
        _project(torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float64),
                 torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64))
    assert error.value.code == "singular_adjustment" and "Rescale" in str(error.value)
    alpha = torch.tensor([[1.0, 0.5], [2.0, -1.0], [0.3, 0.2]], dtype=torch.float64)
    v = torch.tensor([0.4, -0.2, 0.9], dtype=torch.float64)
    assert_allclose(_project(alpha, v).numpy(),
                    np.linalg.lstsq(alpha.numpy(), v.numpy(), rcond=None)[0], rtol=1e-12)
    # A genuinely impossible normalization is still reported: y1 does not cointegrate.
    rng = np.random.default_rng(3)
    walk = rng.normal(size=400).cumsum()
    other_walk = rng.normal(size=400).cumsum()
    exact = pd.DataFrame({"z": other_walk, "p": walk, "q": walk})
    exact["q"] = exact.q + rng.normal(size=400)
    result = oe.vec(data=exact, y=["z", "p", "q"], lags=2, rank=1)
    assert abs(result.extra["beta_matrix"][1][0]) > 5     # normalized on the wrong variable
    check_clean(result)


def test_vec_equation_table_uses_stata_conventions():
    """Before the fix: centered R-squared and a Wald test that left out the constant."""
    frame = integrated()[["a", "b", "c"]] + np.arange(N)[:, None] * 0.3     # drifting levels
    result = oe.vec(data=frame, y=["a", "b", "c"], lags=2, rank=1)
    t, d = int(result.metrics["T"]), int(result.metrics["df_eq"])
    for row in result.extra["equations"]:
        assert row["df"] == row["parms"] == 5             # _ce1, three lagged differences, constant
        assert row["statistic"] == pytest.approx(
            (t - d) * row["r_squared"] / (1 - row["r_squared"]), rel=1e-6)
        assert row["r_squared"] > row["r_squared_centered"]
    for equation in result.extra["beta"]:
        assert {"parms", "statistic", "df", "p_value"} <= set(equation)
