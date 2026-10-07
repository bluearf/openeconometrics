"""Oracles for didregress and eventstudy: statsmodels OLS on explicit dummy variables.

The two-way fixed-effects estimates are compared with least squares on group
and time dummies, the covariances with statsmodels' cluster / HC1 / classical
estimators (the cluster factor rescaled from statsmodels' K = all dummies to
OpenEcon's K = covariates + non-nested absorbed effects), the parallel-trends
and Granger tests with Wald F tests computed by hand from augmented dummy
regressions.
"""

import json
import time

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle


def make_panel(seed=0, groups=36, periods=7, start=2004, repeated=1, staggered=False):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), periods * repeated)
    t = np.tile(np.repeat(np.arange(2000, 2000 + periods), repeated), groups)
    treated = g < groups // 3
    first = np.where(treated, start, np.nan)
    if staggered:
        first = np.where(treated, np.where(g % 2 == 0, start, start + 2), np.nan)
    d = (~np.isnan(first) & (t >= np.nan_to_num(first, nan=1e9))).astype(int)
    x = rng.normal(size=len(g))
    y = rng.normal(size=groups)[g] + 0.2 * (t - 2000) + 1.0 * d + 0.5 * x + rng.normal(size=len(g))
    frame = pd.DataFrame({"y": y, "d": d, "g": g, "t": t, "x": x, "first": first})
    frame["state"] = frame.g // 4
    frame["w"] = rng.uniform(0.5, 2, size=groups)[g]
    frame["f"] = rng.integers(1, 3, size=len(g)).astype(float)
    return frame


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def coef(result, term):
    row = next(c for c in result.coefficients if c.term == term)
    return row.estimate, row.std_error


def test_didregress_matches_dummy_regression_with_group_clusters(panel):
    result = oe.didregress(data=panel, y="y", treatment="d", group="g", time="t", x=["x"])
    model = smf.ols("y ~ d + x + C(g) + C(t)", panel).fit(cov_type="cluster",
                                                          cov_kwds={"groups": panel.g})
    n, k_all, k = len(panel), len(model.params), 2 + panel.t.nunique()
    scale = np.sqrt((n - k_all) / (n - k))           # group effects nested in clusters are free
    est, se = coef(result, "ATET:r1vs0.d")
    assert_allclose(est, model.params["d"], rtol=1e-9)
    assert_allclose(se, model.bse["d"] * scale, rtol=1e-7)
    assert_allclose(coef(result, "x")[1], model.bse["x"] * scale, rtol=1e-7)
    assert result.inference["df_inference"] == panel.g.nunique() - 1
    t_stat = est / se
    assert_allclose(result.coefficients[0].p_value,
                    2 * stats.t.sf(abs(t_stat), panel.g.nunique() - 1), rtol=1e-8)
    assert_allclose(result.metrics["r_squared"], model.rsquared, rtol=1e-9)
    assert result.metrics["n_groups"] == 36 and result.metrics["n_periods"] == 7


def test_other_covariances_and_nested_clusters(panel):
    for covariance, sm_kind in (("HC1", "HC1"), ("nonrobust", "nonrobust")):
        result = oe.didregress(data=panel, y="y", treatment="d", group="g", time="t", x=["x"],
                               covariance=covariance)
        model = smf.ols("y ~ d + x + C(g) + C(t)", panel).fit(cov_type=sm_kind)
        assert_allclose(coef(result, "ATET:r1vs0.d")[1], model.bse["d"], rtol=1e-7)
        assert result.inference["df_inference"] == model.df_resid
    clustered = oe.didregress(data=panel, y="y", treatment="d", group="g", time="t", x=["x"],
                              covariance="cluster", cluster="state")
    model = smf.ols("y ~ d + x + C(g) + C(t)", panel).fit(cov_type="cluster",
                                                          cov_kwds={"groups": panel.state})
    n, k_all, k = len(panel), len(model.params), 2 + panel.t.nunique()
    assert_allclose(coef(clustered, "ATET:r1vs0.d")[1],
                    model.bse["d"] * np.sqrt((n - k_all) / (n - k)), rtol=1e-7)
    assert clustered.inference["cluster_count"] == panel.state.nunique()


def test_weights_and_repeated_cross_sections():
    frame = make_panel(seed=4, repeated=3)
    result = oe.didregress(data=frame, y="y", treatment="d", group="g", time="t",
                           weights="w", weight_type="aweight", covariance="HC1")
    model = smf.wls("y ~ d + C(g) + C(t)", frame, weights=frame.w).fit(cov_type="HC1")
    assert_allclose(coef(result, "ATET:r1vs0.d")[0], model.params["d"], rtol=1e-9)
    assert_allclose(coef(result, "ATET:r1vs0.d")[1], model.bse["d"], rtol=1e-7)
    freq = oe.didregress(data=frame, y="y", treatment="d", group="g", time="t", weights="f",
                         weight_type="fweight")
    expanded = frame.loc[frame.index.repeat(frame.f.astype(int))].reset_index(drop=True)
    plain = oe.didregress(data=expanded, y="y", treatment="d", group="g", time="t")
    assert_allclose(coef(freq, "ATET:r1vs0.d"), coef(plain, "ATET:r1vs0.d"), rtol=1e-8)
    assert freq.nobs == len(expanded)


def wald_f(model, names, groups, n, k):
    """Cluster Wald F with OpenEcon's small-sample factor (K = covariates + T)."""
    cov = model.cov_params().loc[names, names].to_numpy() * (n - len(model.params)) / (n - k)
    b = model.params[names].to_numpy()
    return float(b @ np.linalg.solve(cov, b)) / len(names)


def test_parallel_trends_and_granger_tests_match_augmented_regressions(panel):
    result = oe.didregress(data=panel, y="y", treatment="d", group="g", time="t", x=["x"])
    frame = panel.copy()
    period = frame.t - 2000
    treated = (frame.g < 12).astype(float)
    frame["trend"] = treated * period * (frame.t < 2004)
    model = smf.ols("y ~ d + x + trend + C(g) + C(t)", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame.g})
    n, periods = len(frame), frame.t.nunique()
    expected = wald_f(model, ["trend"], frame.g, n, 3 + periods)
    assert_allclose(result.tests["parallel_trends"]["statistic"], expected, rtol=1e-7)
    assert result.tests["parallel_trends"]["df2"] == 35
    for lead in (1, 2, 3):
        frame[f"lead{lead}"] = treated * (frame.t == 2004 - lead)
    model = smf.ols("y ~ d + x + lead1 + lead2 + lead3 + C(g) + C(t)", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame.g})
    expected = wald_f(model, ["lead1", "lead2", "lead3"], frame.g, n, 5 + periods)
    assert_allclose(result.tests["granger"]["statistic"], expected, rtol=1e-7)
    assert result.tests["granger"]["df"] == 3
    assert result.extra["adoption"] == "common" and result.extra["treated_groups"] == 12


def test_staggered_adoption_warns_and_skips_common_timing_tests():
    frame = make_panel(seed=2, staggered=True)
    result = oe.didregress(data=frame, y="y", treatment="d", group="g", time="t")
    assert result.extra["adoption"] == "staggered"
    assert "parallel_trends" not in result.tests and "tests_note" in result.extra
    assert any("Goodman-Bacon" in w for w in result.warnings)


def event_oracle(frame, events, reference=-1, lo=None, hi=None):
    rel = frame.t - frame["first"]
    if lo is not None:
        rel = rel.clip(lower=lo)
    if hi is not None:
        rel = rel.clip(upper=hi)
    data = frame.copy()
    names = []
    for e in events:
        if e == reference:
            continue
        name = f"ev{'m' if e < 0 else 'p'}{abs(e)}"
        data[name] = ((rel == e) & frame["first"].notna()).astype(float)
        names.append(name)
    model = smf.ols("y ~ " + " + ".join(names + ["x"]) + " + C(g) + C(t)", data).fit(
        cov_type="cluster", cov_kwds={"groups": data.g})
    return model, names


def test_eventstudy_matches_relative_time_dummy_regression(panel):
    result = oe.eventstudy(data=panel, y="y", group="g", time="t", treatment_time="first",
                           x=["x"])
    events = list(range(-4, 3))
    model, names = event_oracle(panel, events)
    n, k = len(panel), len(names) + 1 + panel.t.nunique()
    scale = np.sqrt((n - len(model.params)) / (n - k))
    terms = ["lead4", "lead3", "lead2", "lag0", "lag1", "lag2"]
    got = np.array([coef(result, t) for t in terms])
    assert_allclose(got[:, 0], model.params[names].to_numpy(), rtol=1e-8)
    assert_allclose(got[:, 1], model.bse[names].to_numpy() * scale, rtol=1e-7)
    expected = wald_f(model, names[:3], panel.g, n, k)
    assert_allclose(result.tests["pretrends"]["statistic"], expected, rtol=1e-7)
    table = result.extra["event_table"]
    assert [row["relative_time"] for row in table] == events
    reference = next(row for row in table if row["reference"])
    assert reference["relative_time"] == -1 and reference["estimate"] == 0.0
    post = result.extra["average_post_effect"]
    assert_allclose(post["estimate"], np.mean(got[3:, 0]), rtol=1e-10)


def test_eventstudy_bins_end_points_and_handles_staggered_cohorts():
    frame = make_panel(seed=9, periods=9, staggered=True)
    result = oe.eventstudy(data=frame, y="y", group="g", time="t", treatment_time="first",
                           x=["x"], leads=2, lags=2)
    model, names = event_oracle(frame, list(range(-2, 3)), lo=-2, hi=2)
    got = [coef(result, t)[0] for t in ("lead2", "lag0", "lag1", "lag2")]
    assert_allclose(got, model.params[names].to_numpy(), rtol=1e-8)
    table = result.extra["event_table"]
    assert table[0]["binned"] and table[-1]["binned"]
    assert result.extra["cohorts"] == [2004, 2006]
    assert any("Sun and Abraham" in w or "Goodman-Bacon" in w for w in result.warnings)


def test_failure_contract(panel):
    with pytest.raises(AnalysisError) as caught:
        oe.didregress(data=panel.assign(d=panel.d * 2), y="y", treatment="d", group="g",
                      time="t")
    assert caught.value.code == "invalid_treatment"
    everyone = panel.assign(d=(panel.t >= 2004).astype(int))
    with pytest.raises(AnalysisError) as caught:
        oe.didregress(data=everyone, y="y", treatment="d", group="g", time="t")
    assert caught.value.code == "no_within_variation"
    varying = panel.assign(first=np.where(panel.index % 2 == 0, 2003, panel["first"]))
    with pytest.raises(AnalysisError) as caught:
        oe.eventstudy(data=varying, y="y", group="g", time="t", treatment_time="first")
    assert caught.value.code == "invalid_treatment"
    with pytest.raises(AnalysisError) as caught:
        oe.eventstudy(data=panel, y="y", group="g", time="t", treatment_time="first",
                      reference=-9)
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.eventstudy(data=panel.assign(first=np.nan), y="y", group="g", time="t",
                      treatment_time="first")
    assert caught.value.code == "invalid_treatment"
    with pytest.raises(AnalysisError) as caught:
        oe.didregress(data=panel, y="y", treatment="d", group="g", time="t",
                      covariance="robust", cluster="state")
    assert caught.value.code == "invalid_spec"
    missing = panel.copy()
    missing.loc[[1, 5], "x"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.didregress(data=missing, y="y", treatment="d", group="g", time="t", x=["x"])
    assert caught.value.code == "missing_values"
    dropped = oe.didregress(data=missing, y="y", treatment="d", group="g", time="t", x=["x"],
                            missing="drop")
    assert dropped.nobs == len(panel) - 2
    # a never-treated group's missing treatment_time is not a missing value
    es = oe.eventstudy(data=panel, y="y", group="g", time="t", treatment_time="first")
    assert es.nobs == len(panel)


def test_serialization_rendering_and_exports(panel):
    did = oe.didregress(data=panel, y="y", treatment="d", group="g", time="t", x=["x"])
    es = oe.eventstudy(data=panel, y="y", group="g", time="t", treatment_time="first")
    for result in (did, es):
        assert ResultBundle.model_validate_json(result.model_dump_json()) == result
        json.loads(result.model_dump_json())
        assert "begin{tabular}" in str(result.to_latex())
        assert result.provenance["stata_parity_validated"] is False
    assert "Difference-in-differences regression — y" in did.summary()
    assert "estat ptrends" in did.summary() and "Joint pre-trend test" in es.summary()
    assert callable(oe.didregress) and callable(oe.eventstudy)
    capability = oe.capabilities()["estimators"]["didregress"]
    assert capability["default_covariance"] == "robust" and capability["inference"] == "Student t"


def test_dense_large_panel_runs_in_seconds(monkeypatch):
    # Keep the original resident-kernel throughput contract. Disk FE projections
    # and full-source DID diagnostics have separate Dataset verification.
    from openecon.econometrics import streaming_registry
    monkeypatch.setattr(streaming_registry, "supports_spec", lambda spec: False)
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1024")
    rng = np.random.default_rng(1)
    groups, periods = 50_000, 20
    g = np.repeat(np.arange(groups), periods)
    t = np.tile(np.arange(periods), groups)
    first = np.where(g % 3 == 0, 10.0, np.nan)
    d = (t >= np.nan_to_num(first, nan=1e9)).astype(int)
    x = rng.normal(size=(len(g), 5))
    y = rng.normal(size=groups)[g] + 0.1 * t + d + x.sum(1) + rng.normal(size=len(g))
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(5)]).assign(
        y=y, d=d, g=g, t=t, first=first)
    start = time.perf_counter()
    did = oe.didregress(data=frame, y="y", treatment="d", group="g", time="t",
                        x=[f"x{i}" for i in range(5)])
    es = oe.eventstudy(data=frame, y="y", group="g", time="t", treatment_time="first",
                       leads=4, lags=4)
    assert time.perf_counter() - start < 40
    assert abs(coef(did, "ATET:r1vs0.d")[0] - 1) < 0.02
    assert es.nobs == len(frame)
