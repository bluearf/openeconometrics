"""Independent ARDL/OLS oracles and analytic multiplier derivative checks."""

import itertools

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from statsmodels.tsa.ardl import ARDL

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import make_spec
from openecon.models import ResultBundle


def data(seed=90, n=260):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).cumsum() + 12
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    delta = np.diff(x, prepend=x[0])
    xp, xn = np.maximum(delta, 0).cumsum(), np.minimum(delta, 0).cumsum()
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (.3 + .45*y[t-1] + .07*y[t-2] + .4*xp[t] - .1*xp[t-1]
                + .15*xn[t] + .06*xn[t-1] + .2*z[t] + .3*w[t] + rng.normal())
    return pd.DataFrame(dict(y=y, x=x, z=z, w=w, t=np.arange(n))), xp, xn


@pytest.mark.parametrize("cov", ["nonrobust", "robust", "HC1", "HC2", "HC3"])
def test_levels_against_independent_ols_and_ardl(cov):
    df, xp, xn = data()
    r = oe.nardl(data=df, y="y", x=["x", "z"], asymmetric=["x"],
                 lags=[2, 1, 0], exog=["w"], covariance=cov, ec=False)
    X = np.column_stack([np.ones(len(df)-2), df.y.to_numpy()[1:-1], df.y.to_numpy()[:-2],
                         xp[2:], xp[1:-1], xn[2:], xn[1:-1], df.z[2:], df.w[2:]])
    ref = sm.OLS(df.y[2:].to_numpy(), X).fit(cov_type="HC1" if cov == "robust" else cov)
    assert_allclose([c.estimate for c in r.coefficients], ref.params, rtol=1e-9, atol=1e-10)
    assert_allclose(r.covariance_matrix, ref.cov_params(), rtol=1e-8, atol=1e-10)
    assert r.extra["lags"] == {"y": 2, "x_positive": 1, "x_negative": 1, "z": 0}
    assert r.spec.predictors == ["x", "z"]
    assert r.provenance["input_columns"] == ["y", "x", "z", "w"]
    assert r.nobs == len(df)-2
    if cov == "nonrobust":
        expanded = pd.DataFrame(dict(x_positive=xp, x_negative=xn, z=df.z))
        ardl = ARDL(df.y, 2, expanded, {"x_positive": 1, "x_negative": 1, "z": 0},
                    fixed=df[["w"]]).fit()
        assert_allclose([c.estimate for c in r.coefficients], ardl.params, rtol=1e-9)


def test_ec_long_run_and_covariance_delta_oracle():
    df, xp, xn = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[2, 1], covariance="HC3")
    X = np.column_stack([np.ones(len(df)-2), df.y.to_numpy()[1:-1], df.y.to_numpy()[:-2],
                         xp[2:], xp[1:-1], xn[2:], xn[1:-1]])
    ref = sm.OLS(df.y[2:].to_numpy(), X).fit(cov_type="HC3")
    b, V = ref.params, ref.cov_params()
    a = 1-b[1]-b[2]
    expected = [b[1]+b[2]-1, (b[3]+b[4])/a, (b[5]+b[6])/a, -b[2], -b[4], -b[6], b[0]]
    J = np.zeros((7, 7))
    J[0, 1:3] = 1
    J[1, 3:5] = 1/a
    J[1, 1:3] = (b[3]+b[4])/a**2
    J[2, 5:7] = 1/a
    J[2, 1:3] = (b[5]+b[6])/a**2
    J[3, 2] = J[4, 4] = J[5, 6] = -1
    J[6, 0] = 1
    assert_allclose([c.estimate for c in r.coefficients], expected, rtol=1e-9)
    assert_allclose(r.covariance_matrix, J @ V @ J.T, rtol=1e-8, atol=1e-10)
    assert [c.term for c in r.coefficients][:3] == ["ADJ:L.y", "LR:x_positive", "LR:x_negative"]


@pytest.mark.parametrize("orders", [[2, 1], [2, 2, 1]])
def test_wald_symmetry_independent_matrix_oracle(orders):
    df, xp, xn = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=orders, ec=False, covariance="HC1")
    p, qp, qn = [2, 1, 1] if len(orders) == 2 else orders
    m, n = max(p, qp, qn), len(df)
    columns = [np.ones(n-m)] + [df.y.to_numpy()[m-i:n-i] for i in range(1, p+1)]
    columns += [xp[m-i:n-i] for i in range(qp+1)] + [xn[m-i:n-i] for i in range(qn+1)]
    ref = sm.OLS(df.y[m:].to_numpy(), np.column_stack(columns)).fit(cov_type="HC1")
    long = np.zeros(len(columns))
    sr = long.copy()
    startp, startn = 1+p, 1+p+qp+1
    long[startp:startn] = 1
    long[startn:] = -1
    sr[startp], sr[startn] = 1, -1
    for lag in range(2, qp+1):
        sr[startp+lag] = -(lag-1)
    for lag in range(2, qn+1):
        sr[startn+lag] = lag-1
    for name, R in (("long_run", long[None]), ("short_run", sr[None]),
                    ("joint", np.stack([long, sr]))):
        expected = ref.f_test(R)
        actual = r.tests[f"symmetry_{name}"]
        assert_allclose(actual["statistic"], expected.fvalue, rtol=1e-8)
        # statsmodels robust f_test still uses the same residual F denominator.
        assert_allclose(actual["p_value"], expected.pvalue, rtol=1e-8)


def test_no_short_run_symmetry_without_distributed_lags():
    df, _, _ = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[1, 0])
    assert "symmetry_short_run" not in r.tests
    assert "symmetry_long_run" in r.tests


def test_lag_selection_matches_independent_exhaustive_search():
    df, xp, xn = data(n=100)
    r = oe.nardl(data=df, y="y", x=["x"], maxlags=[2, 1], ic="bic", ec=False)
    candidates = []
    n, holdback = len(df), 2
    for p, qp, qn in itertools.product(range(1, 3), range(2), range(2)):
        cols = [np.ones(n-holdback)] + [df.y.to_numpy()[holdback-i:n-i] for i in range(1, p+1)]
        cols += [xp[holdback-i:n-i] for i in range(qp+1)]
        cols += [xn[holdback-i:n-i] for i in range(qn+1)]
        fit = sm.OLS(df.y[holdback:].to_numpy(), np.column_stack(cols)).fit()
        candidates.append((fit.bic, [p, qp, qn]))
    selected = min(candidates, key=lambda c: c[0])
    assert r.extra["lag_selection"]["selected"] == selected[1]
    assert_allclose(r.metrics["bic"], selected[0], rtol=1e-10)
    assert r.extra["partial_sum_origin"]["row"] == 0


def test_multipliers_and_standard_errors_finite_difference_oracle():
    df, _, _ = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[2, 1])
    multipliers = oe.nardl_multipliers(r, steps=80)
    terms = list(r.extra["levels_coefficients"])
    beta = np.array(list(r.extra["levels_coefficients"].values()))
    V = np.array(r.extra["levels_covariance"])

    def simulated(b):
        # Independent step-response simulation, no OpenEcon recurrence/helper.
        output = np.zeros((81, 2))
        phi = [b[terms.index("L.y")], b[terms.index("L2.y")]]
        for j, name in enumerate(("x_positive", "x_negative")):
            q = r.extra["lags"][name]
            dl = [b[terms.index(name if i == 0 else f"L.{name}")] for i in range(q+1)]
            for h in range(81):
                value = sum(dl[i] for i in range(q+1) if h-i >= 0)
                for i in range(2):
                    if h-i-1 >= 0:
                        value += phi[i]*output[h-i-1, j]
                output[h, j] = value
        return np.column_stack([output, output[:, 0]-output[:, 1]])

    expected = simulated(beta)
    J = np.zeros((81, 3, len(beta)))
    for j in range(len(beta)):
        step = 1e-5
        up, down = beta.copy(), beta.copy()
        up[j] += step
        down[j] -= step
        J[:, :, j] = (simulated(up)-simulated(down))/(2*step)
    expected_se = np.sqrt(np.einsum("hsk,kl,hsl->hs", J, V, J))
    for i, name in enumerate(("positive", "negative", "difference")):
        assert_allclose(multipliers[name], expected[:, i], rtol=1e-10, atol=1e-10)
        assert_allclose(multipliers[f"{name}_std_error"], expected_se[:, i], rtol=1e-7)
    assert_allclose(multipliers.positive.iloc[-1], r.extra["long_run"][0]["estimate"], rtol=1e-10)
    assert multipliers.attrs["autoregressive_stable"] is True
    assert "\\end{longtable}" in multipliers.to_latex()


def test_generic_fit_json_and_latex_are_reproducible():
    df, _, _ = data()
    spec = make_spec("nardl", outcome="y", predictors=["x"], time="t", options={"lags": [2, 1]})
    r = oe.fit(spec, data=df)
    saved = ResultBundle.model_validate_json(r.model_dump_json())
    assert saved.spec == spec
    assert "tabular" in saved.to_latex()
    assert_allclose(oe.nardl_multipliers(saved).positive, oe.nardl_multipliers(r).positive)
    assert r.provenance["stata_parity_validated"] is False
    for name in ("bounds_f", "bounds_t"):
        assert r.tests[name]["critical_values"] is None
        assert r.tests[name]["decision"] is None and r.tests[name]["p_value"] is None


def test_sorted_rows_and_missing_edges_preserve_original_positions():
    df, _, _ = data()
    df.loc[0, "x"] = np.nan
    shuffled = df.sample(frac=1, random_state=9).reset_index(drop=True)
    r = oe.nardl(data=shuffled, y="y", x=["x"], time="t", lags=[2, 1], missing="drop")
    clean = oe.nardl(data=df.iloc[1:].reset_index(drop=True), y="y", x=["x"],
                     time="t", lags=[2, 1])
    assert_allclose([c.estimate for c in r.coefficients], [c.estimate for c in clean.coefficients])
    assert r.nobs_original == len(df) and r.dropped_rows == 3
    assert shuffled.iloc[r.sample_positions].t.tolist() == list(range(3, len(df)))
    assert r.extra["partial_sum_origin"]["row"] == int(shuffled.index[shuffled.t == 1][0])


@pytest.mark.parametrize("change,code", [
    ({"asymmetric": []}, "invalid_spec"), ({"asymmetric": ["z"]}, "invalid_spec"),
    ({"asymmetric": ["x", "x"]}, "invalid_spec"), ({"lags": [0, 1]}, "invalid_lags"),
    ({"lags": [1, 1, 1, 1]}, "invalid_lags"), ({"trend": "none"}, "invalid_option"),
    ({"covariance": "cluster"}, "invalid_spec"),
    ({"exog": ["x"]}, "invalid_spec"),
])
def test_rejected_options(change, code):
    df, _, _ = data()
    options = dict(data=df, y="y", x=["x"], lags=[2, 1])
    options.update(change)
    with pytest.raises(AnalysisError) as exc:
        oe.nardl(**options)
    assert exc.value.code == code


@pytest.mark.parametrize("kind,code", [("monotone", "unidentified_asymmetry"),
                                        ("gap", "time_gaps"), ("missing", "time_gaps"),
                                        ("duplicate", "repeated_time_values")])
def test_data_guards(kind, code):
    df, _, _ = data()
    if kind == "monotone":
        df.x = np.arange(len(df))
    elif kind == "gap":
        df = df.drop(index=50)
    elif kind == "missing":
        df.loc[50, "x"] = np.nan
    else:
        df.loc[50, "t"] = 49
    with pytest.raises(AnalysisError) as exc:
        oe.nardl(data=df, y="y", x=["x"], time="t", lags=[2, 1], missing="drop")
    assert exc.value.code == code


def test_generated_name_collision():
    df, _, _ = data()
    df["x_positive"] = df.z
    with pytest.raises(AnalysisError, match="collide"):
        oe.nardl(data=df, y="y", x=["x", "x_positive"], asymmetric=["x"], lags=[2, 1, 1])


@pytest.mark.parametrize("steps", [-1, True, 1.5, 1001])
def test_invalid_multiplier_horizons(steps):
    df, _, _ = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[2, 1])
    with pytest.raises(AnalysisError, match="steps"):
        oe.nardl_multipliers(r, steps=steps)


def test_malformed_saved_multiplier_state():
    df, _, _ = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[2, 1])
    r.extra.pop("levels_covariance")
    with pytest.raises(AnalysisError) as exc:
        oe.nardl_multipliers(r)
    assert exc.value.code == "invalid_result"


@pytest.mark.parametrize("alpha", [0, 1, True, float("nan"), "0.05"])
def test_invalid_multiplier_alpha(alpha):
    df, _, _ = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[2, 1])
    with pytest.raises(AnalysisError, match="alpha"):
        oe.nardl_multipliers(r, alpha=alpha)


def test_partial_sum_overflow_has_actionable_error():
    df, _, _ = data()
    df.x = np.resize([1e308, -1e308], len(df))
    with pytest.raises(AnalysisError, match="rescale") as exc:
        oe.nardl(data=df, y="y", x=["x"], lags=[2, 1])
    assert exc.value.code == "non_finite_design"


@pytest.mark.parametrize("bad", [None, 0, -1, True, float("inf")])
def test_saved_multiplier_inference_cannot_silently_switch_to_normal(bad):
    df, _, _ = data()
    r = oe.nardl(data=df, y="y", x=["x"], lags=[2, 1])
    r.inference["df_inference"] = bad
    with pytest.raises(AnalysisError) as exc:
        oe.nardl_multipliers(r)
    assert exc.value.code == "invalid_inference"


def test_datetime_interval_explicitly_preserves_gaps():
    df, _, _ = data()
    df.t = pd.date_range("2020-01-01", periods=len(df), freq="D")
    r = oe.nardl(data=df, y="y", x=["x"], time="t", time_delta="1D", lags=[2, 1])
    reference = oe.nardl(data=df.drop(columns="t"), y="y", x=["x"], lags=[2, 1])
    assert_allclose([c.estimate for c in r.coefficients], [c.estimate for c in reference.coefficients])
    with pytest.raises(AnalysisError) as exc:
        oe.nardl(data=df, y="y", x=["x"], time="t", lags=[2, 1])
    assert exc.value.code == "invalid_time_delta"
    with pytest.raises(AnalysisError) as exc:
        oe.nardl(data=df.drop(index=50), y="y", x=["x"], time="t", time_delta="1D", lags=[2, 1])
    assert exc.value.code == "time_gaps"


@pytest.mark.parametrize("delta", ["0D", "-1D", "invalid"])
def test_invalid_datetime_delta(delta):
    df, _, _ = data()
    df.t = pd.date_range("2020-01-01", periods=len(df), freq="D")
    with pytest.raises(AnalysisError) as exc:
        oe.nardl(data=df, y="y", x=["x"], time="t", time_delta=delta, lags=[2, 1])
    assert exc.value.code == "invalid_time_delta"
