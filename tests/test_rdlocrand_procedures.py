"""rdlocrand procedures: invariances, NumPy oracles, window logic and the failure contract.

The oracles enumerate assignments independently with NumPy/itertools; the
invariances (shifting the running variable and cutoff together, row order,
outcome reflection) hold exactly because units are put in a canonical order
before any draw.
"""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def make_data(n=240, seed=4, effect=0.6):
    rng = np.random.default_rng(seed)
    r = np.round(rng.uniform(-1, 1, n), 4)
    t = ((r >= 0) & (rng.random(n) < 0.85)) | ((r < 0) & (rng.random(n) < 0.1))
    x1 = np.round(rng.normal(size=n) + 0.4 * r, 4)
    x2 = np.round(rng.normal(size=n) + 1.2 * (np.abs(r) > 0.6) * np.sign(r), 4)
    y = np.round(1 + r + effect * t + rng.normal(size=n), 4)
    return pd.DataFrame({"r": r, "y": y, "t": t.astype(float), "x1": x1, "x2": x2})


DATA = make_data()


def window(data, wl, wr, cutoff=0.0):
    inside = (data.r.round(8) >= round(wl, 8)) & (data.r.round(8) <= round(wr, 8))
    w = data[inside]
    return w.y.to_numpy(), (w.r >= cutoff).to_numpy(), w


def assignments(n, n1):
    rows = np.zeros((math.comb(n, n1), n), bool)
    for index, chosen in enumerate(itertools.combinations(range(n), n1)):
        rows[index, list(chosen)] = True
    return rows


def exact_diffmeans(y, d, taus):
    """Exact fixed-margins p-values of the difference in means for every null effect."""
    a = assignments(len(y), int(d.sum()))
    outcomes = y[:, None] - np.asarray(taus)[None, :] * d[:, None]
    values = a @ outcomes / a.sum(1, keepdims=True) - (~a) @ outcomes / (~a).sum(1, keepdims=True)
    observed = outcomes[d].mean(0) - outcomes[~d].mean(0)
    return list((np.abs(values) >= np.abs(observed) - 1e-10).mean(0))


# ---- invariances --------------------------------------------------------------------------------


def test_shifting_running_variable_and_cutoff_together_changes_nothing():
    shifted = DATA.assign(r=DATA.r + 1024.0)
    a = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.2, wr=0.25, statistic="all", reps=2000)
    b = oe.rdrandinf(data=shifted, y="y", running="r", cutoff=1024.0, wl=1023.8, wr=1024.25,
                     statistic="all", reps=2000)
    pd.testing.assert_frame_equal(a["tests"], b["tests"])
    w1 = oe.rdwinselect(data=DATA, running="r", covariates=["x1", "x2"], reps=300, nwindows=4)
    w2 = oe.rdwinselect(data=shifted, running="r", cutoff=1024.0, covariates=["x1", "x2"], reps=300,
                        nwindows=4)
    assert np.allclose(w2["windows"]["w_right"] - 1024.0, w1["windows"]["w_right"], atol=1e-9)
    assert list(w1["windows"]["p_value"]) == list(w2["windows"]["p_value"])


def test_row_order_does_not_change_monte_carlo_results():
    shuffled = DATA.sample(frac=1.0, random_state=9).reset_index(drop=True)
    for function, options in (
            (oe.rdrandinf, dict(y="y", wl=-0.3, wr=0.3, statistic="all", reps=1500)),
            (oe.rdrandinf, dict(y="y", wl=-0.3, wr=0.3, fuzzy="t", bernoulli=True, reps=1500)),
            (oe.rdsensitivity, dict(y="y", wlist=[0.2, 0.3], tlist=[0, 0.5, 1], reps=800))):
        a, b = function(data=DATA, running="r", **options), function(data=shuffled, running="r", **options)
        for name in a:
            pd.testing.assert_frame_equal(a[name], b[name])
    a = oe.rdwinselect(data=DATA, running="r", covariates=["x1", "x2"], statistic="ksmirnov", reps=400,
                       nwindows=4)
    b = oe.rdwinselect(data=shuffled, running="r", covariates=["x1", "x2"], statistic="ksmirnov",
                       reps=400, nwindows=4)
    pd.testing.assert_frame_equal(a["windows"], b["windows"])


def test_reflecting_the_outcome_leaves_two_sided_p_values_unchanged():
    reflected = DATA.assign(y=-DATA.y)
    a = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.15, wr=0.15, statistic="all", reps=3000)
    b = oe.rdrandinf(data=reflected, y="y", running="r", wl=-0.15, wr=0.15, statistic="all", reps=3000)
    assert np.allclose(a["tests"]["p_value"], b["tests"]["p_value"])
    assert np.allclose(a["tests"]["statistic"] * [-1, 1, -1], b["tests"]["statistic"])


# ---- NumPy oracles ------------------------------------------------------------------------------


def test_sensitivity_grid_matches_exact_enumeration():
    taus = [-1.0, 0.0, 0.5, 1.5]
    res = oe.rdsensitivity(data=DATA, y="y", running="r", wlist=[0.02, 0.02], wlist_left=[-0.02, -0.03],
                           tlist=taus, method="exact")
    for j, (wl, wr) in enumerate(((-0.02, 0.02), (-0.03, 0.02))):
        y, d, _ = window(DATA, wl, wr)
        expected = exact_diffmeans(y, d, taus)
        got = res["pvalues"][res["pvalues"]["window"] == j + 1]["p_value"].tolist()
        assert np.allclose(got, expected, atol=1e-14)
    assert list(res["windows"]["method"]) == ["exact", "exact"]


def test_confidence_set_is_the_accepted_grid_and_may_be_disconnected():
    grid = [-2 + 0.1 * i for i in range(51)]
    y, d, _ = window(DATA, -0.03, 0.02)
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.03, wr=0.02, ci=0.1, ci_grid=grid,
                       method="exact")
    expected = exact_diffmeans(y, d, grid)
    assert np.allclose(res["ci_grid"]["p_value"], expected, atol=1e-14)
    accepted = [tau for tau, p in zip(grid, expected) if p >= 0.1]
    rows = res["ci"]
    assert rows["ci_low"].iloc[0] == pytest.approx(accepted[0]) and rows["ci_high"].iloc[-1] == pytest.approx(accepted[-1])
    covered = [tau for tau in grid if any(lo - 1e-12 <= tau <= hi + 1e-12 for lo, hi in zip(rows["ci_low"], rows["ci_high"]))]
    assert np.allclose(covered, accepted)


def test_interference_interval_uses_quantiles_of_the_exact_law():
    y, d, _ = window(DATA, -0.03, 0.02)
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.03, wr=0.02, interfci=0.1, method="exact")
    a = assignments(len(y), int(d.sum()))
    values = a @ y / a.sum(1) - (~a) @ y / (~a).sum(1)
    observed = y[d].mean() - y[~d].mean()
    low, high = np.quantile(values, [0.05, 0.95])
    assert res.attrs["interference_ci_low"] == pytest.approx(observed - high, rel=1e-10)
    assert res.attrs["interference_ci_high"] == pytest.approx(observed - low, rel=1e-10)


def test_rosenbaum_bounds_at_gamma_one_equal_the_bernoulli_half_test():
    res = oe.rdrbounds(data=DATA, y="y", running="r", wlist=[0.02], expgamma=[1.0], statistic="diffmeans",
                       method="exact")
    base = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.02, wr=0.02, bernoulli=0.5, method="exact")
    p = base["tests"]["p_value"][0]
    assert res["bounds"]["lower_bound"][0] == pytest.approx(p, abs=1e-14)
    assert res["bounds"]["upper_bound"][0] == pytest.approx(p, abs=1e-14)


def test_rosenbaum_bounds_bracket_and_widen():
    res = oe.rdrbounds(data=DATA, y="y", running="r", wlist=[0.02], expgamma=[1.5, 3.0],
                       statistic="diffmeans", method="exact", fmpval=True)
    bounds = res["bounds"]
    assert all(bounds["lower_bound"] <= bounds["upper_bound"])
    assert bounds["upper_bound"][1] >= bounds["upper_bound"][0] - 1e-12
    assert res["pvalues"]["p_value_fixed_margins"][0] is not None
    one = oe.rdrbounds(data=DATA, y="y", running="r", wlist=[0.02], expgamma=[2.0], bound="upper",
                       method="exact", statistic="diffmeans")
    assert one["bounds"]["lower_bound"][0] is None
    assert one["bounds"]["upper_bound"][0] == pytest.approx(
        oe.rdrbounds(data=DATA, y="y", running="r", wlist=[0.02], expgamma=[2.0], method="exact",
                     statistic="diffmeans")["bounds"]["upper_bound"][0])


# ---- window selection ------------------------------------------------------------------------------


def test_recommended_window_is_the_last_passing_nested_window():
    res = oe.rdwinselect(data=DATA, running="r", covariates=["x1", "x2"], approx=True, wmin=0.2,
                         wstep=0.1, nwindows=8, level=0.1)
    p = list(res["windows"]["p_value"])
    assert p[0] >= 0.1 and min(p) < 0.1
    last = next(j for j, value in enumerate(p) if value < 0.1) - 1
    assert res.attrs["recommended_window"] == last + 1
    assert res.attrs["recommended_right"] == pytest.approx(res["windows"]["w_right"][last])
    balance = res["balance"]
    assert len(balance) == 2 * len(p)
    assert np.allclose(balance.groupby("window")["p_value"].min().to_numpy(), p)
    integrated = oe.rdrandinf(data=DATA, y="y", running="r", covariates=["x1", "x2"], approx=True,
                              wmin=0.2, wstep=0.1, nwindows=8, level=0.1, reps=500)
    assert integrated.attrs["window_selection"] == "rdwinselect"
    assert integrated.attrs["window_right"] == pytest.approx(res.attrs["recommended_right"])
    assert "windows" in integrated


def test_no_covariates_gives_binomial_tests_without_recommendation():
    res = oe.rdwinselect(data=DATA, running="r", nwindows=3)
    assert res.attrs["recommended_left"] is None and "balance" not in res
    assert all(res["windows"]["binomial_p_value"].notna())
    assert any("covariate" in note for note in res.attrs["notes"])


def test_missing_covariates_are_excluded_per_window_or_dropped():
    data = DATA.copy()
    data.loc[[3, 10, 20], "x1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.rdwinselect(data=data, running="r", covariates=["x1", "x2"], approx=True)
    assert error.value.code == "missing_values"
    kept = oe.rdwinselect(data=data, running="r", covariates=["x1", "x2"], approx=True, missing="drop",
                          wmin=0.3, wstep=0.1, nwindows=3)
    dropped = oe.rdwinselect(data=data, running="r", covariates=["x1", "x2"], approx=True, dropmissing=True,
                             wmin=0.3, wstep=0.1, nwindows=3)
    clean = oe.rdwinselect(data=data.dropna(), running="r", covariates=["x1", "x2"], approx=True,
                           wmin=0.3, wstep=0.1, nwindows=3)
    assert kept.attrs["n_missing_covariates"] == 3 and dropped.attrs["n_dropped"] == 3
    pd.testing.assert_frame_equal(dropped["windows"], clean["windows"])
    assert np.allclose(kept["windows"]["p_value"], clean["windows"]["p_value"])


def test_missing_outcomes_follow_the_missing_policy():
    data = DATA.copy()
    data.loc[[0, 1], "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.rdrandinf(data=data, y="y", running="r", wl=-0.3, wr=0.3)
    assert error.value.code == "missing_values"
    res = oe.rdrandinf(data=data, y="y", running="r", wl=-0.3, wr=0.3, missing="drop", reps=300)
    ref = oe.rdrandinf(data=data.dropna(), y="y", running="r", wl=-0.3, wr=0.3, reps=300)
    assert res.attrs["n_dropped"] == 2
    pd.testing.assert_frame_equal(res["tests"], ref["tests"])


# ---- failure contract -----------------------------------------------------------------------------


@pytest.mark.parametrize("function, options, code", [
    (oe.rdrandinf, dict(wl=0.0, wr=0.3), "insufficient_observations"),
    (oe.rdrandinf, dict(wl=-0.001, wr=0.002), "insufficient_observations"),
    (oe.rdrandinf, dict(wl=0.3, wr=-0.3), "invalid_window"),
    (oe.rdrandinf, dict(wl=0.1, wr=0.3), "invalid_window"),
    (oe.rdrandinf, dict(wl=-0.3), "invalid_option"),
    (oe.rdrandinf, dict(cutoff=5.0), "invalid_cutoff"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, kernel="gaussian"), "invalid_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, kernel="triangular", statistic="ranksum"), "unsupported_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, p=1, statistic="ksmirnov"), "unsupported_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, method="exact"), "work_budget_exceeded"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, ci_grid=[0, 1]), "invalid_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, interfci=0.05, bernoulli=True), "unsupported_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, bernoulli=1.5), "invalid_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, statistic="all", ci=0.05), "unsupported_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, covariates=["x1"]), "invalid_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, p=1, fuzzy="t", ci=0.05), "invalid_option"),
    (oe.rdrandinf, dict(wl=-0.3, wr=0.3, reps=0), "invalid_option"),
    (oe.rdrandinf, dict(y="missing_column", wl=-0.3, wr=0.3), "missing_columns"),
    (oe.rdsensitivity, dict(p=1), "invalid_option"),
    (oe.rdsensitivity, dict(wlist=[0.2], tlist=[0], ci=[-0.3, 0.3]), "invalid_option"),
    (oe.rdsensitivity, dict(wlist=[-0.2], tlist=[0]), "invalid_window"),
    (oe.rdrbounds, dict(gamma=[0.5], expgamma=[2]), "invalid_option"),
    (oe.rdrbounds, dict(expgamma=[0.5]), "invalid_option"),
])
def test_failure_contract(function, options, code):
    options = {"y": "y", **options}
    with pytest.raises(AnalysisError) as error:
        function(data=DATA, running="r", **options)
    assert error.value.code == code


@pytest.mark.parametrize("options, code", [
    (dict(wmasspoints=True, wobs=3), "invalid_option"),
    (dict(obsmin=5, wmin=0.2), "invalid_option"),
    (dict(wobs=5, wstep=0.1), "invalid_option"),
    (dict(wmin=[0.1, 0.2]), "invalid_option"),
    (dict(statistic="hotelling", covariates=None), "invalid_option"),
    (dict(statistic="ranksum", p=1), "unsupported_option"),
    (dict(cutoff=-1.5), "invalid_cutoff"),
    (dict(obsmin=200), "insufficient_observations"),
    (dict(level=1.0), "invalid_option"),
])
def test_window_selection_failure_contract(options, code):
    options = {"covariates": ["x1", "x2"], **options}
    with pytest.raises(AnalysisError) as error:
        oe.rdwinselect(data=DATA, running="r", **options)
    assert error.value.code == code


def test_no_recommended_window_is_an_error_inside_rdrandinf():
    data = DATA.assign(x1=DATA.x1 + 3.0 * (DATA.r >= 0))
    with pytest.raises(AnalysisError) as error:
        oe.rdrandinf(data=data, y="y", running="r", covariates=["x1"], approx=True)
    assert error.value.code == "no_recommended_window"


def test_bernoulli_probability_column_is_validated():
    with pytest.raises(AnalysisError) as error:
        oe.rdrandinf(data=DATA.assign(prob=1.5), y="y", running="r", wl=-0.2, wr=0.2, bernoulli="prob")
    assert error.value.code == "invalid_probabilities"


# ---- results ---------------------------------------------------------------------------------------


def test_results_are_json_safe_and_render(tmp_path):
    results = [
        oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.2, wr=0.2, statistic="all", reps=300),
        oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.3, wr=0.3, fuzzy="t", fuzzy_statistic="tsls", ci=0.05),
        oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.3, wr=0.3, p=1, ci=0.05),
        oe.rdwinselect(data=DATA, running="r", covariates=["x1", "x2"], reps=200, nwindows=3),
        oe.rdsensitivity(data=DATA, y="y", running="r", wlist=[0.2, 0.3], tlist=[0, 1], reps=200),
        oe.rdrbounds(data=DATA, y="y", running="r", wlist=[0.04], expgamma=[1.5], reps=200),
    ]
    for index, result in enumerate(results):
        restored = json.loads(json.dumps(result.attrs))
        assert restored == result.attrs
        assert result.attrs["stata_parity_validated"] is False
        text, latex = str(result), result.to_latex()
        assert result.title in text and "tabular" in latex
        for frame in result.values():
            json.loads(frame.to_json(orient="records"))
        path = tmp_path / f"rd-result-{index}.json"
        path.write_text(oe.summary_state(result))
        recovered = oe.restore_summary(path.read_text())
        assert recovered.title == result.title and recovered.attrs == result.attrs
        assert set(recovered) == set(result)
        for name, frame in result.items():
            pd.testing.assert_frame_equal(recovered[name], frame, check_dtype=False)


def test_large_sample_rows_have_no_randomization_p_value():
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.3, wr=0.3, fuzzy="t", fuzzy_statistic="tsls")
    row = res["tests"].iloc[0]
    assert row["p_value"] is None or math.isnan(row["p_value"])
    assert res.attrs["inference"] == "large-sample" and res.attrs["method"] is None
    assert res.attrs["vce"] == "HC1"


def test_constant_outcome_in_the_window_gives_unit_p_values():
    data = DATA.assign(y=np.where(DATA.r.abs() <= 0.1, 2.0, DATA.y))
    res = oe.rdrandinf(data=data, y="y", running="r", wl=-0.1, wr=0.1, statistic="all", reps=500)
    tests = res["tests"]
    assert list(tests["p_value"]) == [1.0, 1.0, 1.0]
    assert list(tests["asymptotic_p_value"]) == [1.0, 1.0, 1.0]
    assert tests["statistic"].tolist() == [0.0, 0.0, 0.0]
    assert res.attrs["monte_carlo_standard_errors"] == [0.0, 0.0, 0.0]
