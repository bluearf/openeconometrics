"""Independent oracles for oe.ardl: statsmodels ARDL/UECM, NumPy lag search and EC algebra."""

import itertools

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.tsa.ardl.pss_critical_values import crit_vals

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsmodels import bounds


def make_data(seed=0, n=220):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).cumsum(0)
    w = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (0.2 + 0.5 * y[t - 1] + 0.1 * y[t - 2] + 0.3 * x[t, 0] - 0.2 * x[t - 1, 1]
                + 0.4 * w[t] + rng.normal())
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "w": w,
                         "t": np.arange(n) + 1900})


@pytest.fixture(scope="module")
def df():
    return make_data()


def coef(fit):
    return np.array([c.estimate for c in fit.coefficients])


def ses(fit):
    return np.array([c.std_error for c in fit.coefficients])


def reorder_sm(params, trend):
    """statsmodels order (const, trend, y lags, x lags) equals ours (Intercept, trend, ...)."""
    return np.asarray(params)


@pytest.mark.parametrize("trend,sm_trend", [("constant", "c"), ("trend", "ct"), ("none", "n")])
def test_levels_match_statsmodels(df, trend, sm_trend):
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[2, 1, 2], trend=trend, time="t")
    ref = ARDL(df.y, 2, df[["x1", "x2"]], {"x1": 1, "x2": 2}, trend=sm_trend).fit()
    assert_allclose(coef(fit), reorder_sm(ref.params, sm_trend), rtol=1e-8, atol=1e-10)
    assert_allclose(ses(fit), reorder_sm(ref.bse, sm_trend), rtol=1e-7)
    assert fit.nobs == len(df) - 2 and fit.inference["df_inference"] == fit.nobs - len(ref.params)
    assert_allclose(fit.metrics["log_likelihood"], ref.llf, rtol=1e-10)
    expected = ["L.y", "L2.y", "x1", "L.x1", "x2", "L.x2", "L2.x2"]
    head = {"constant": ["Intercept"], "trend": ["Intercept", "trend"], "none": []}[trend]
    assert [c.term for c in fit.coefficients] == head + expected


@pytest.mark.parametrize("cov", ["robust", "HC1", "HC2", "HC3"])
def test_robust_covariances(df, cov):
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[1, 1, 0], exog=["w"], covariance=cov)
    n = len(df)
    y = df.y.to_numpy()
    X = np.column_stack([np.ones(n - 1), y[:-1], df.x1[1:], df.x1[:-1], df.x2[1:], df.w[1:]])
    ref = sm.OLS(y[1:], X).fit(cov_type="HC1" if cov == "robust" else cov)
    assert_allclose(coef(fit), ref.params, rtol=1e-9)
    assert_allclose(ses(fit), ref.bse, rtol=1e-8)
    assert [c.term for c in fit.coefficients][-1] == "w"


@pytest.mark.parametrize("case,trend,restricted,sm_trend", [
    (1, "none", False, "n"), (2, "constant", True, "c"), (3, "constant", False, "c"),
    (4, "trend", True, "ct"), (5, "trend", False, "ct")])
def test_bounds_f_matches_statsmodels_uecm(df, case, trend, restricted, sm_trend):
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[2, 1, 2], trend=trend,
                  restricted=restricted)
    uecm = UECM(df.y, 2, df[["x1", "x2"]], {"x1": 1, "x2": 2}, trend=sm_trend).fit()
    reference = uecm.bounds_test(case=case)
    test = fit.tests["bounds_f"]
    assert test["case"] == case and test["k"] == 2
    assert_allclose(test["statistic"], reference.stat, rtol=1e-8)
    assert test["df"] == 3 + (case in (2, 4))
    assert test["p_value"] is None
    assert test["critical_values"] == bounds.critical_values("F", case, 2)
    if case in (1, 3, 5):
        t_ref = uecm.tvalues.iloc[list(uecm.params.index).index("y.L1")]
        assert_allclose(fit.tests["bounds_t"]["statistic"], t_ref, rtol=1e-8)
    else:
        assert "bounds_t" not in fit.tests


def test_bounds_decisions():
    crit = bounds.critical_values("F", 3, 1)
    assert crit["5%"] == {"I0": 4.94, "I1": 5.73}
    assert bounds.decide("F", 6.0, crit)["5%"].startswith("reject")
    assert bounds.decide("F", 5.0, crit)["5%"] == "inconclusive"
    assert bounds.decide("F", 4.0, crit)["5%"] == "do not reject H0"
    t_crit = bounds.critical_values("t", 3, 1)
    assert bounds.decide("t", -3.3, t_crit)["5%"].startswith("reject")
    assert bounds.decide("t", -3.0, t_crit)["5%"] == "inconclusive"
    assert bounds.critical_values("t", 2, 1) is None and bounds.critical_values("F", 3, 11) is None


def test_typed_tables_agree_with_independent_simulation():
    """Every typed F bound (k >= 1) lies within Monte Carlo error of a 32M-draw simulation."""
    worst = [0.0, 0.0, 0.0]
    for case in range(1, 6):
        for k in range(1, 11):
            row = bounds.F_BOUNDS[case][k]
            low, high = crit_vals[(k, case, False)], crit_vals[(k, case, True)]
            for i in range(3):                                  # 90, 95, 99 percentiles
                worst[i] = max(worst[i], abs(row[2 * i] - low[i]), abs(row[2 * i + 1] - high[i]))
    # PSS used 40,000 replications: the 1% quantiles carry the largest Monte Carlo error.
    assert worst[0] < 0.05 and worst[1] < 0.05 and worst[2] < 0.15
    for case in (1, 3, 5):                                  # t: I(0) bound constant in k, I(1) grows
        table = np.array(bounds.T_BOUNDS[case])
        assert np.all(table[:, 0] == table[0, 0]) and np.all(np.diff(table[:, 1]) < 0)


def ec_oracle(df, p, q, case):
    """OLS of D.y on y_(t-1), x_t (Stata's ec form), lagged differences and deterministics."""
    y = df.y.to_numpy()
    X = df[["x1", "x2"]].to_numpy()
    n = len(y)
    m = max(p, *q)
    rows = np.arange(m, n)
    cols, names = [], []
    if case in (3, 4, 5):
        cols.append(np.ones(len(rows)))
        names.append("SR:Intercept")
    if case == 2:
        cols.append(np.ones(len(rows)))
        names.append("const_r")
    cols.append(y[rows - 1])
    names.append("ADJ")
    for j in range(2):
        cols.append(X[rows, j])
        names.append(f"xlevel{j}")
    for i in range(1, p):
        cols.append(y[rows - i] - y[rows - i - 1])
        names.append(f"SR:dy{i}")
    for j in range(2):
        for lag_ in range(q[j]):
            cols.append(X[rows - lag_, j] - X[rows - lag_ - 1, j])
            names.append(f"SR:dx{j}_{lag_}")
    Z = np.column_stack(cols)
    ref = sm.OLS(y[rows] - y[rows - 1], Z).fit()
    b, V = ref.params, ref.cov_params()
    a = names.index("ADJ")
    lr = []
    targets = ([names.index("const_r")] if case == 2 else []) + [names.index("xlevel0"),
                                                                  names.index("xlevel1")]
    for idx in targets:
        value = -b[idx] / b[a]
        grad = np.zeros(len(b))
        grad[idx] = -1 / b[a]
        grad[a] = b[idx] / b[a] ** 2
        lr.append((value, np.sqrt(grad @ V @ grad)))
    return ref, names, lr


@pytest.mark.parametrize("case,trend,restricted", [(3, "constant", False), (2, "constant", True)])
def test_error_correction_form(df, case, trend, restricted):
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[2, 1, 2], trend=trend,
                  restricted=restricted, ec=True)
    ref, names, lr = ec_oracle(df, 2, [1, 2], case)
    est = {c.term: (c.estimate, c.std_error) for c in fit.coefficients}
    a = names.index("ADJ")
    assert_allclose(est["ADJ:L.y"], (ref.params[a], ref.bse[a]), rtol=1e-8)
    lr_terms = (["LR:Intercept"] if case == 2 else []) + ["LR:x1", "LR:x2"]
    for term, (value, se) in zip(lr_terms, lr):
        assert_allclose(est[term], (value, se), rtol=1e-7)
    mapping = {"SR:LD.y": "SR:dy1", "SR:D.x1": "SR:dx0_0", "SR:D.x2": "SR:dx1_0",
               "SR:LD.x2": "SR:dx1_1"}
    if case == 3:
        mapping["SR:Intercept"] = "SR:Intercept"
    for ours, theirs in mapping.items():
        i = names.index(theirs)
        assert_allclose(est[ours], (ref.params[i], ref.bse[i]), rtol=1e-7, atol=1e-12)
    assert set(est) == {"ADJ:L.y", *lr_terms, *mapping}
    assert_allclose(fit.metrics["r_squared"], ref.rsquared, rtol=1e-9)
    assert_allclose(fit.extra["speed_of_adjustment"], ref.params[a], rtol=1e-9)
    levels = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[2, 1, 2], trend=trend,
                     restricted=restricted)
    long_run = {row["term"]: row for row in levels.extra["long_run"]}
    assert_allclose(long_run["x1"]["estimate"], est["LR:x1"][0], rtol=1e-10)
    assert_allclose(long_run["x1"]["std_error"], est["LR:x1"][1], rtol=1e-10)


def numpy_select(df, maxlags, ic, exog=False):
    y = df.y.to_numpy()
    X = df[["x1", "x2"]].to_numpy()
    n = len(y)
    m = max(maxlags)
    rows = np.arange(m, n)
    best = None
    for combo in itertools.product(range(1, maxlags[0] + 1), range(maxlags[1] + 1),
                                   range(maxlags[2] + 1)):
        cols = [np.ones(len(rows))] + ([df.w.to_numpy()[rows]] if exog else [])
        cols += [y[rows - i] for i in range(1, combo[0] + 1)]
        for j in range(2):
            cols += [X[rows - lag_, j] for lag_ in range(combo[j + 1] + 1)]
        Z = np.column_stack(cols)
        resid = y[rows] - Z @ np.linalg.lstsq(Z, y[rows], rcond=None)[0]
        nn = len(rows)
        ll = -nn / 2 * (1 + np.log(2 * np.pi) + np.log(resid @ resid / nn))
        value = -2 * ll + (2 if ic == "aic" else np.log(nn)) * Z.shape[1]
        if best is None or value < best[0] - 1e-9:
            best = (value, list(combo))
    return best


@pytest.mark.parametrize("ic", ["aic", "bic"])
@pytest.mark.parametrize("maxlags,exog", [([3, 3, 3], False), ([2, 3, 1], True)])
def test_lag_selection_matches_brute_force(df, ic, maxlags, exog):
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], maxlags=maxlags if len(set(maxlags)) > 1
                  else maxlags[0], ic=ic, exog=["w"] if exog else None)
    value, combo = numpy_select(df, maxlags, ic, exog)
    selection = fit.extra["lag_selection"]
    assert selection["selected"] == combo
    assert selection["models"] == maxlags[0] * (maxlags[1] + 1) * (maxlags[2] + 1)
    assert_allclose(selection["best"][0][ic], value, rtol=1e-9)
    assert fit.nobs == len(df) - max(maxlags)                 # common sample
    assert_allclose(fit.metrics[ic], value, rtol=1e-9)
    assert fit.extra["ardl_order"] == "ARDL(" + ",".join(map(str, combo)) + ")"


def test_distributed_lag_and_errors(df):
    dl = oe.ardl(data=df, y="y", x=["x1"], lags=[0, 2])
    assert "bounds_f" not in dl.tests and [c.term for c in dl.coefficients][:2] == ["Intercept", "x1"]
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df, y="y", x=["x1"], lags=[0, 2], ec=True)
    assert err.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[1, 1])
    assert err.value.code == "invalid_lags"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df, y="y", x=["x1"], lags=[1, 1], trend="none", restricted=True)
    assert err.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df.assign(c=3.0), y="y", x=["x1", "c"], lags=[1, 1, 1])
    assert err.value.code == "collinear_lags"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df, y="y", x=["x1", "x2", "w", "t"], maxlags=30)
    assert err.value.code in {"lag_grid_too_large", "insufficient_observations"}
    big = pd.DataFrame(np.random.default_rng(2).normal(size=(3000, 7)),
                       columns=["y", *[f"x{i}" for i in range(6)]])
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=big, y="y", x=[f"x{i}" for i in range(6)], maxlags=8)
    assert err.value.code == "lag_grid_too_large"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df, y="y", x=["x1"], ic="hq")
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=df, y="y", x=["x1"], exog=["x1"])
    assert err.value.code == "invalid_spec"


def test_missing_and_time_order(df):
    shuffled = df.sample(frac=1, random_state=4).reset_index(drop=True)
    a = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[1, 1, 1], time="t")
    b = oe.ardl(data=shuffled, y="y", x=["x1", "x2"], lags=[1, 1, 1], time="t")
    assert_allclose(coef(a), coef(b), rtol=1e-12)
    holed = df.copy()
    holed.loc[len(df) - 1, "x2"] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=holed, y="y", x=["x1", "x2"], lags=[1, 1, 1])
    assert err.value.code == "missing_values"
    fit = oe.ardl(data=holed, y="y", x=["x1", "x2"], lags=[1, 1, 1], missing="drop")
    assert fit.nobs == len(df) - 2
    holed.loc[50, "x1"] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=holed, y="y", x=["x1", "x2"], lags=[1, 1, 1], missing="drop")
    assert err.value.code == "time_gaps"


def test_round_trip_and_rendering(df):
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], maxlags=2, ec=True, time="t")
    assert type(fit).model_validate_json(fit.model_dump_json()) == fit
    text = fit.summary()
    assert "bounds F" in text and "ADJ" in text
    assert "LR" in fit.to_latex()
    assert fit.provenance["stata_parity_validated"] is False
