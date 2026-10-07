"""areg / reghdfe against explicit dummy-variable regressions (numpy, statsmodels)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pydantic import ValidationError

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ModelSpec, ResultBundle


@pytest.fixture
def panel():
    rng = np.random.default_rng(77)
    firms, years = 24, 10
    n = firms * years
    frame = pd.DataFrame({
        "firm": np.repeat(np.arange(firms), years), "year": np.tile(np.arange(years), firms),
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "w": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.0, size=n),
        "region": np.repeat(np.arange(6), years * 4),
    })
    frame["y"] = 0.3 * frame.firm + 0.2 * frame.year + 1.5 * frame.x1 - 0.7 * frame.x2 \
        + rng.normal(size=n) * (1 + 0.5 * frame.x1.abs())
    frame["firm_level"] = frame.firm * 2.0          # no within-firm variation
    return frame


def dummies(frame, *columns, prefix=True):
    blocks = [pd.get_dummies(frame[name], drop_first=True, dtype=float, prefix=name if prefix else None)
              for name in columns]
    return pd.concat([pd.Series(1.0, index=frame.index, name="const"), frame[["x1", "x2"]], *blocks], axis=1)


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def test_areg_matches_the_dummy_variable_regression(panel):
    result = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm")
    x = dummies(panel, "firm")
    reference = sm.OLS(panel.y, x).fit()
    n, groups = len(panel), 24
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert_allclose(estimates(result)[1:], reference.params.values[1:3], rtol=1e-12)
    assert_allclose(errors(result)[1:], reference.bse.values[1:3], rtol=1e-12)
    # Stata's _cons: grand mean of y minus grand means of x times b.
    assert result.coefficients[0].estimate == pytest.approx(
        panel.y.mean() - panel[["x1", "x2"]].mean().values @ reference.params.values[1:3], rel=1e-12)
    assert result.metrics["df_resid"] == n - 2 - groups == reference.df_resid
    assert result.metrics["df_absorbed"] == groups - 1 and result.metrics["n_groups"] == groups
    assert result.metrics["r_squared"] == pytest.approx(reference.rsquared, rel=1e-12)
    assert result.metrics["adjusted_r_squared"] == pytest.approx(reference.rsquared_adj, rel=1e-12)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(reference.scale), rel=1e-12)
    within = panel.groupby("firm")[["y", "x1", "x2"]].transform(lambda s: s - s.mean())
    tss_within = (within.y**2).sum()
    assert result.metrics["r_squared_within"] == pytest.approx(1 - reference.ssr / tss_within, rel=1e-12)
    assert result.tests["model"]["statistic"] == pytest.approx(
        reference.f_test(np.eye(x.shape[1])[1:3]).fvalue, rel=1e-10)
    assert result.tests["model"]["df"] == 2 and result.tests["model"]["df2"] == reference.df_resid
    pooled = sm.OLS(panel.y, x.iloc[:, :3]).fit()
    absorbed = result.tests["absorbed"]
    assert absorbed["df"] == groups - 1 and absorbed["df2"] == reference.df_resid
    assert absorbed["statistic"] == pytest.approx(
        ((pooled.ssr - reference.ssr) / (groups - 1)) / (reference.ssr / reference.df_resid), rel=1e-10)
    assert_allclose([p["fitted"] for p in result.predictions[:5]], reference.fittedvalues.values[:5])
    # reghdfe's shape: one level is the reference absorbed into the reported constant.
    assert result.extra["absorbed"] == [{"column": "firm", "levels": 24, "redundant": 1, "nested": False}]
    assert result.extra["constant_degree_of_freedom_added"] is True
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    assert "F test of absorbed firm effects" in result.summary()


def test_areg_robust_and_cluster_count_the_absorbed_effects(panel):
    x = dummies(panel, "firm")
    n, groups = len(panel), 24
    robust = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", covariance="robust")
    reference = sm.OLS(panel.y, x).fit(cov_type="HC1")
    assert robust.spec.covariance == "HC1"
    assert_allclose(errors(robust)[1:], reference.bse.values[1:3], rtol=1e-12)
    assert robust.inference["small_sample_correction"] == pytest.approx(n / (n - 2 - groups))
    assert "absorbed" not in robust.tests
    clustered = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", cluster="firm")
    reference = sm.OLS(panel.y, x).fit(cov_type="cluster", cov_kwds={"groups": panel.firm})
    assert_allclose(errors(clustered)[1:], reference.bse.values[1:3], rtol=1e-12)
    # Documented areg-vs-xtreg difference: areg keeps K_total = k + G even when nested.
    factor = groups / (groups - 1) * (n - 1) / (n - 2 - groups)
    assert clustered.inference["small_sample_correction"] == pytest.approx(factor)
    assert clustered.inference["df_inference"] == groups - 1
    xtreg_like = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm"], cluster="firm")
    ratio = np.sqrt((n - 2 - groups) / (n - 3))
    assert_allclose(errors(xtreg_like) * (1 / ratio), errors(clustered)[1:], rtol=1e-12)
    two_way = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", cluster=["firm", "year"])
    assert two_way.inference["df_inference"] == 9 and two_way.inference["cluster_counts"] == [24, 10]


def test_areg_weights_match_duplicated_rows_and_wls(panel):
    duplicated = panel.loc[panel.index.repeat(panel.w.astype(int))].reset_index(drop=True)
    weighted = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", weights="w", weight_type="fweight")
    plain = oe.areg(data=duplicated, y="y", x=["x1", "x2"], absorb="firm")
    assert weighted.nobs == plain.nobs == int(panel.w.sum())
    assert_allclose(estimates(weighted), estimates(plain), rtol=1e-11)
    assert_allclose(errors(weighted), errors(plain), rtol=1e-11)
    for name in ("r_squared", "r_squared_within", "rmse", "df_resid"):
        assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-11), name
    assert weighted.tests["absorbed"]["statistic"] == pytest.approx(plain.tests["absorbed"]["statistic"], rel=1e-10)
    analytic = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", weights="aw", weight_type="aweight")
    reference = sm.WLS(panel.y, dummies(panel, "firm"), weights=panel.aw).fit()
    assert_allclose(estimates(analytic)[1:], reference.params.values[1:3], rtol=1e-12)
    assert_allclose(errors(analytic)[1:], reference.bse.values[1:3], rtol=1e-12)
    mean = lambda column: np.average(panel[column], weights=panel.aw)  # noqa: E731
    assert analytic.coefficients[0].estimate == pytest.approx(
        mean("y") - np.array([mean("x1"), mean("x2")]) @ reference.params.values[1:3], rel=1e-12)
    sampling = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", weights="aw", weight_type="pweight")
    reference = sm.WLS(panel.y, dummies(panel, "firm"), weights=panel.aw).fit(cov_type="HC1")
    assert sampling.spec.covariance == "HC1"
    assert_allclose(errors(sampling)[1:], reference.bse.values[1:3], rtol=1e-12)


def test_areg_omits_regressors_without_within_variation_and_rejects_bad_input(panel):
    result = oe.areg(data=panel, y="y", x=["x1", "firm_level", "x2"], absorb="firm")
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert result.provenance["omitted_terms"] == ["firm_level"]
    assert any("absorbed fixed effects" in warning for warning in result.warnings)
    with pytest.raises(AnalysisError) as caught:
        oe.areg(data=panel, y="y", x=["x1"], absorb=["firm"])
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.areg(data=panel.assign(one=1), y="y", x=["x1"], absorb="one")
    assert caught.value.code == "insufficient_groups"
    with pytest.raises(AnalysisError) as caught:
        oe.areg(data=panel, y="y", x=["x1"], absorb="firm", weights="aw", weight_type="pweight",
                covariance="nonrobust")
    assert caught.value.code == "unsupported_covariance"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="areg", outcome="y", predictors=["x1"])
    with pytest.raises(ValidationError):
        ModelSpec(estimator="areg", outcome="y", predictors=["x1"], columns={"absorb": "firm"}, intercept=False)
    with pytest.raises(ValidationError):
        ModelSpec(estimator="areg", outcome="y", predictors=["x1"], columns={"absorb": "firm"}, covariance="HC3")
    # One observation per firm: the effects use every degree of freedom.
    with pytest.raises(AnalysisError) as caught:
        oe.areg(data=panel[panel.year < 1], y="y", x=["x1", "x2"], absorb="firm")
    assert caught.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as caught:
        oe.areg(data=panel.assign(y=lambda d: d.firm + 2 * d.x1), y="y", x=["x1"], absorb="firm")
    assert caught.value.code == "perfect_fit"


def test_reghdfe_two_way_matches_dummies_and_counts_redundant_effects(panel):
    result = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"])
    x = dummies(panel, "firm", "year")
    reference = sm.OLS(panel.y, x).fit()
    n = len(panel)
    assert [c.term for c in result.coefficients] == ["x1", "x2"]
    assert_allclose(estimates(result), reference.params.values[1:3], rtol=1e-11)
    assert_allclose(errors(result), reference.bse.values[1:3], rtol=1e-11)
    assert result.metrics["df_absorbed"] == 24 + 10 - 1 and result.metrics["df_resid"] == reference.df_resid
    assert result.metrics["r_squared"] == pytest.approx(reference.rsquared, rel=1e-11)
    assert result.metrics["adjusted_r_squared"] == pytest.approx(reference.rsquared_adj, rel=1e-11)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(reference.scale), rel=1e-11)
    # Within sums of squares from the explicit two-way projection.
    d = x.iloc[:, [0, *range(3, x.shape[1])]].to_numpy()
    projector = np.eye(n) - d @ np.linalg.pinv(d)
    tss_within = panel.y.to_numpy() @ projector @ panel.y.to_numpy()
    assert result.metrics["r_squared_within"] == pytest.approx(1 - reference.ssr / tss_within, rel=1e-10)
    assert result.metrics["adjusted_r_squared_within"] == pytest.approx(
        1 - (reference.ssr / reference.df_resid) / (tss_within / (n - 33)), rel=1e-10)
    assert result.tests["model"]["statistic"] == pytest.approx(
        reference.f_test(np.eye(x.shape[1])[1:3]).fvalue, rel=1e-9)
    assert result.tests["model"]["df2"] == reference.df_resid
    assert result.extra["absorbed"] == [
        {"column": "firm", "levels": 24, "redundant": 0, "nested": False},
        {"column": "year", "levels": 10, "redundant": 1, "nested": False}]
    assert result.extra["converged"] and result.extra["method"] == "symmetric_kaczmarz_cg"
    assert result.metrics["n_singletons_dropped"] == 0 and result.inference["intercept"] is False
    assert_allclose([p["fitted"] for p in result.predictions[:4]], reference.fittedvalues.values[:4], rtol=1e-9)
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    assert "Model F test (slopes)" in result.summary() and r"\text{Within }R^{2}" in str(result.to_latex())
    one_way = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm"])
    reference = sm.OLS(panel.y, dummies(panel, "firm")).fit()
    assert_allclose(errors(one_way), reference.bse.values[1:3], rtol=1e-12)
    assert one_way.extra["method"] == "within" and one_way.metrics["df_absorbed"] == 24


def test_reghdfe_cluster_degrees_of_freedom_follow_the_nested_rule(panel):
    n = len(panel)
    x = dummies(panel, "firm")
    # One dimension nested in the cluster: K = k + 1, the xtreg, fe convention.
    result = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm"], cluster="firm")
    reference = sm.OLS(panel.y, x).fit(cov_type="cluster", cov_kwds={"groups": panel.firm})
    assert_allclose(errors(result), reference.bse.values[1:3] * np.sqrt((n - 2 - 24) / (n - 3)), rtol=1e-12)
    assert result.metrics["df_absorbed"] == 1 and result.inference["df_inference"] == 23
    assert result.extra["absorbed"][0]["nested"] and result.extra["constant_degree_of_freedom_added"]
    assert result.inference["small_sample_correction"] == pytest.approx(24 / 23 * (n - 1) / (n - 3))
    # Firm nested in region: the firm levels do not count, the constant is added back.
    result = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm"], cluster="region")
    reference = sm.OLS(panel.y, x).fit(cov_type="cluster", cov_kwds={"groups": panel.region})
    assert_allclose(errors(result), reference.bse.values[1:3] * np.sqrt((n - 2 - 24) / (n - 3)), rtol=1e-12)
    assert result.inference["df_inference"] == 5
    # Two dimensions, one nested: year's 10 levels count (it now absorbs the constant).
    result = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster="firm")
    x2 = dummies(panel, "firm", "year")
    reference = sm.OLS(panel.y, x2).fit(cov_type="cluster", cov_kwds={"groups": panel.firm})
    assert_allclose(errors(result), reference.bse.values[1:3] * np.sqrt((n - x2.shape[1]) / (n - 12)), rtol=1e-11)
    assert result.metrics["df_absorbed"] == 10 and not result.extra["constant_degree_of_freedom_added"]
    # Two-way clustering uses the smallest G for the factor and the inference df. Nesting is
    # checked against EVERY cluster column: firm nests in cluster firm and year in cluster
    # year, so neither counts and the constant is added back (K = k + 1), as reghdfe marks
    # each fixed effect "nested within cluster" against each cluster variable.
    two_way = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster=["firm", "year"])
    assert two_way.inference["df_inference"] == 9 and two_way.inference["cluster_counts"] == [24, 10]
    assert two_way.metrics["df_absorbed"] == 1 and two_way.extra["constant_degree_of_freedom_added"]
    assert [a["nested"] for a in two_way.extra["absorbed"]] == [True, True]
    assert two_way.inference["small_sample_correction"] == pytest.approx(10 / 9 * (n - 1) / (n - 3))
    reordered = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster=["year", "firm"])
    assert_allclose(errors(reordered), errors(two_way), rtol=1e-13)
    assert reordered.metrics["df_absorbed"] == 1
    robust = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"], covariance="robust")
    reference = sm.OLS(panel.y, x2).fit(cov_type="HC1")
    assert_allclose(errors(robust), reference.bse.values[1:3], rtol=1e-11)


def test_reghdfe_drops_singletons_iteratively_and_can_keep_them(panel):
    frame = panel.copy()
    # Firm 23 keeps one observation; that observation is also the only one in year 10.
    frame = frame[(frame.firm != 23) | (frame.year == 0)].copy()
    frame.loc[(frame.firm == 23), "year"] = 10
    frame.loc[len(frame)] = frame.iloc[0]       # a duplicated row keeps firm 0 / year 0 safe
    frame.loc[frame.index[-1], "firm"] = 99     # ...but is itself a firm singleton
    result = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"])
    assert result.metrics["n_singletons_dropped"] == 2 and result.nobs == len(frame) - 2
    assert any(warning.startswith("Dropped 2 singleton observation(s)") for warning in result.warnings)
    kept = frame[(frame.firm != 23) & (frame.firm != 99)]
    reference = sm.OLS(kept.y, dummies(kept, "firm", "year")).fit()
    assert_allclose(estimates(result), reference.params.values[1:3], rtol=1e-11)
    assert_allclose(errors(result), reference.bse.values[1:3], rtol=1e-11)
    assert sorted(result.sample_positions) == kept.index.tolist()
    full = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], drop_singletons=False)
    assert full.nobs == len(frame) and full.metrics["n_singletons_dropped"] == 0
    assert_allclose(estimates(full), estimates(result), rtol=1e-10)   # singletons are fitted exactly
    assert full.metrics["df_resid"] == result.metrics["df_resid"]     # ...and cost one df each


def test_reghdfe_weights_omissions_and_errors(panel):
    duplicated = panel.loc[panel.index.repeat(panel.w.astype(int))].reset_index(drop=True)
    weighted = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"], weights="w",
                          weight_type="fweight", cluster="region")
    plain = oe.reghdfe(data=duplicated, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster="region")
    assert weighted.nobs == plain.nobs == int(panel.w.sum())
    assert_allclose(estimates(weighted), estimates(plain), rtol=1e-10)
    assert_allclose(errors(weighted), errors(plain), rtol=1e-10)
    assert weighted.metrics["df_resid"] == plain.metrics["df_resid"]
    analytic = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm", "year"], weights="aw",
                          weight_type="aweight")
    reference = sm.WLS(panel.y, dummies(panel, "firm", "year"), weights=panel.aw).fit()
    assert_allclose(estimates(analytic), reference.params.values[1:3], rtol=1e-10)
    assert_allclose(errors(analytic), reference.bse.values[1:3], rtol=1e-10)
    sampling = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm"], weights="aw", weight_type="pweight")
    reference = sm.WLS(panel.y, dummies(panel, "firm"), weights=panel.aw).fit(cov_type="HC1")
    assert_allclose(errors(sampling), reference.bse.values[1:3], rtol=1e-11)
    omitted = oe.reghdfe(data=panel.assign(const=1.0), y="y", x=["x1", "firm_level", "const", "x2"],
                         absorb=["firm", "year"])
    assert [c.term for c in omitted.coefficients] == ["x1", "x2"]
    assert omitted.provenance["omitted_terms"] == ["firm_level", "const"]
    categorical = oe.reghdfe(data=panel.assign(group=np.tile(["u", "v", "w"], 80)), y="y",
                             x=["x1", "group"], absorb=["firm"], categorical=["group"])
    assert [c.term for c in categorical.coefficients] == ["x1", "group[v]", "group[w]"]
    for arguments, code in [
        ({"x": ["firm_level"]}, "empty_design"),
        ({"x": ["x1"], "tolerance": 0.0}, "invalid_option"),
        ({"x": ["x1"], "tolerance": 1e-8, "max_iterations": 1}, "absorption_nonconvergence"),
        ({"x": ["x1"], "absorb": ["firm", "idx"]}, "empty_sample"),
    ]:
        frame = panel.assign(idx=np.arange(len(panel)))
        with pytest.raises(AnalysisError) as caught:
            oe.reghdfe(data=frame, y="y", **{"absorb": ["firm", "year"], **arguments})
        assert caught.value.code == code, arguments
    with pytest.raises(AnalysisError) as caught:
        oe.reghdfe(data=panel, y="y", x=["x1"], absorb="firm")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="reghdfe", outcome="y", predictors=["x1"], columns={"absorb": ["firm"]})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="reghdfe", outcome="y", predictors=["x1"], intercept=False)
    with pytest.raises(ValidationError):
        ModelSpec(estimator="reghdfe", outcome="y", predictors=["x1"], intercept=False,
                  columns={"absorb": ["firm"]}, options={"max_iterations": 0})


def test_reghdfe_scales_to_many_levels():
    rng = np.random.default_rng(11)
    n, firms, years = 150_000, 15_000, 300
    frame = pd.DataFrame({"firm": rng.integers(0, firms, size=n), "year": rng.integers(0, years, size=n),
                          "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    firm_effect, year_effect = rng.normal(size=firms), rng.normal(size=years)
    frame["y"] = firm_effect[frame.firm] + year_effect[frame.year] + frame.x1 - 0.5 * frame.x2 \
        + rng.normal(size=n)
    result = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster="firm")
    assert abs(result.coefficients[0].estimate - 1) < 0.02 and result.extra["converged"]
    assert result.metrics["df_absorbed"] == 300 and result.nobs == n - result.metrics["n_singletons_dropped"]


# ---------------------------------------------------------------- regression tests (repairs)


def test_reghdfe_nesting_is_checked_against_every_cluster_column(panel):
    """Regression: absorb(firm) with cluster(firm year) and cluster(year firm) gave different
    standard errors because only the first cluster column was tested for nesting."""
    n = len(panel)
    first = oe.reghdfe(data=panel, y="y", x=["x1"], absorb=["firm"], cluster=["firm", "year"])
    second = oe.reghdfe(data=panel, y="y", x=["x1"], absorb=["firm"], cluster=["year", "firm"])
    assert_allclose(errors(first), errors(second), rtol=1e-13)
    for result in (first, second):
        assert result.metrics["df_absorbed"] == 1 and result.metrics["df_resid"] == n - 2
        assert result.extra["absorbed"] == [{"column": "firm", "levels": 24, "redundant": 24, "nested": True}]
        assert result.extra["constant_degree_of_freedom_added"] is True
    # Explicit oracle: within-firm scores, Cameron-Gelbach-Miller meat, K = k + 1, G_min = 10.
    d = pd.get_dummies(panel.firm, dtype=float).to_numpy()
    x, y = panel[["x1"]].to_numpy(), panel.y.to_numpy()
    xw = x - d @ np.linalg.lstsq(d, x, rcond=None)[0]
    yw = y - d @ np.linalg.lstsq(d, y, rcond=None)[0]
    scores = xw * (yw - xw @ np.linalg.lstsq(xw, yw, rcond=None)[0])[:, None]

    def meat(labels):
        sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
        return sums.T @ sums

    both = panel.firm.astype(str) + "/" + panel.year.astype(str)
    combined = meat(panel.firm) + meat(panel.year) - meat(both)
    variance = combined[0, 0] / (xw.T @ xw)[0, 0] ** 2 * 10 / 9 * (n - 1) / (n - 2)
    assert variance > 0 and errors(first)[0] == pytest.approx(np.sqrt(variance), rel=1e-10)
    # Nested only in the SECOND cluster column (firm within region): still not counted.
    late = oe.reghdfe(data=panel, y="y", x=["x1"], absorb=["firm"], cluster=["year", "region"])
    assert late.extra["absorbed"][0]["nested"] is True and late.metrics["df_absorbed"] == 1
    # Partial nesting: firm nests in region, year does not: year's 10 levels count, no constant.
    partly = oe.reghdfe(data=panel, y="y", x=["x1"], absorb=["firm", "year"], cluster="region")
    assert [a["nested"] for a in partly.extra["absorbed"]] == [True, False]
    assert [a["redundant"] for a in partly.extra["absorbed"]] == [24, 0]
    assert partly.metrics["df_absorbed"] == 10 and not partly.extra["constant_degree_of_freedom_added"]
    # Nesting spread over two cluster columns (firm in firm, year in year) with a third,
    # non-nested dimension: only that one counts, in either cluster order.
    for clusters in (["firm", "year"], ["year", "firm"]):
        mixed = oe.reghdfe(data=panel, y="y", x=["x1"], absorb=["firm", "year", "region"], cluster=clusters)
        assert [a["nested"] for a in mixed.extra["absorbed"]] == [True, True, False]
        assert [a["redundant"] for a in mixed.extra["absorbed"]] == [24, 10, 0]
        assert mixed.metrics["df_absorbed"] == 6 and not mixed.extra["constant_degree_of_freedom_added"]
    # areg records the flag but keeps counting the absorbed effects (K_total = k + G).
    counted = oe.areg(data=panel, y="y", x=["x1"], absorb="firm", cluster=["year", "region"])
    assert counted.extra["absorbed"] == [{"column": "firm", "levels": 24, "redundant": 1, "nested": True}]
    assert counted.inference["k_total"] == 25
    assert oe.areg(data=panel, y="y", x=["x1"], absorb="firm", cluster="year").extra["absorbed"][0]["nested"] is False


def test_regressor_with_large_between_variation_is_kept(panel):
    """Regression: x = 1e4 * firm + x1 has unit within variation; the absorbed screen must keep
    it (and still omit an exactly absorbed column)."""
    frame = panel.assign(shifted=1e4 * panel.firm + panel.x1)
    reference = sm.OLS(panel.y, dummies(panel, "firm")).fit()      # the slope of shifted is that of x1
    result = oe.areg(data=frame, y="y", x=["shifted", "x2"], absorb="firm")
    assert [c.term for c in result.coefficients] == ["Intercept", "shifted", "x2"]
    assert result.provenance["omitted_terms"] == []
    assert_allclose(estimates(result)[1:], reference.params.values[1:3], rtol=1e-8)
    assert_allclose(errors(result)[1:], reference.bse.values[1:3], rtol=1e-8)
    two_way = oe.reghdfe(data=frame, y="y", x=["shifted", "firm_level", "x2"], absorb=["firm", "year"])
    assert [c.term for c in two_way.coefficients] == ["shifted", "x2"]
    assert two_way.provenance["omitted_terms"] == ["firm_level"]
    reference = sm.OLS(panel.y, dummies(panel, "firm", "year")).fit()
    assert_allclose(estimates(two_way), reference.params.values[1:3], rtol=1e-8)
    assert_allclose(errors(two_way), reference.bse.values[1:3], rtol=1e-8)


def test_exact_fit_guard_is_not_triggered_by_the_level_of_the_outcome(panel):
    """Regression: y = 1e9 + 2 x + 1e-3 e is a well-conditioned fit, not a perfect one, while a
    true exact fit is rejected at any level of the outcome."""
    noise = 1e-3 * np.random.default_rng(3).normal(size=len(panel))
    frame = panel.assign(small=2 * panel.x1 + noise, y=1e9 + 2 * panel.x1 + noise)
    reference = sm.OLS(frame.small, dummies(panel, "firm")[["const", "x1", *[f"firm_{i}" for i in range(1, 24)]]]).fit()
    for fit_one in (lambda outcome: oe.areg(data=frame, y=outcome, x=["x1"], absorb="firm"),
                    lambda outcome: oe.reghdfe(data=frame, y=outcome, x=["x1"], absorb=["firm"])):
        slope = next(c for c in fit_one("y").coefficients if c.term == "x1")
        assert slope.estimate == pytest.approx(reference.params["x1"], rel=1e-6)
        assert slope.std_error == pytest.approx(reference.bse["x1"], rel=1e-2)
    two_way = oe.reghdfe(data=frame, y="y", x=["x1"], absorb=["firm", "year"])
    centered = oe.reghdfe(data=frame, y="small", x=["x1"], absorb=["firm", "year"])
    assert two_way.coefficients[0].std_error == pytest.approx(centered.coefficients[0].std_error, rel=1e-2)
    for offset in (0.0, 1e10):
        exact = panel.assign(y=offset + panel.firm + 0.5 * panel.year + 2 * panel.x1)
        for fit_one in (lambda: oe.areg(data=exact.assign(y=exact.y - 0.5 * exact.year), y="y", x=["x1"],
                                        absorb="firm"),
                        lambda: oe.reghdfe(data=exact, y="y", x=["x1"], absorb=["firm", "year"])):
            with pytest.raises(AnalysisError) as caught:
                fit_one()
            assert caught.value.code == "perfect_fit" and "exactly" in str(caught.value)


def test_reghdfe_exact_fit_in_an_unbalanced_design_is_rejected():
    """An exact fit after iterative demeaning leaves residuals at the demeaning tolerance, not at
    rounding level: it must still be reported as a perfect fit, never as tiny standard errors."""
    rng = np.random.default_rng(12)
    n = 400
    frame = pd.DataFrame({"a": rng.integers(0, 30, size=n), "b": rng.integers(0, 12, size=n),
                          "x": rng.normal(size=n)})
    frame["y"] = 0.3 * frame.a - 0.2 * frame.b + 2 * frame.x
    with pytest.raises(AnalysisError) as caught:
        oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b"])
    assert caught.value.code == "perfect_fit"
    noisy = oe.reghdfe(data=frame.assign(y=frame.y + 1e-3 * rng.normal(size=n)), y="y", x=["x"], absorb=["a", "b"])
    assert noisy.coefficients[0].estimate == pytest.approx(2, abs=1e-3)
    assert noisy.metrics["r_squared_within"] > 0.999999


def test_frequency_weights_count_in_the_singleton_rule_and_levels_need_observations(panel):
    """Regression: a row with fweight >= 2 is not a singleton and protects its levels, so the
    weighted fit equals the duplicated-row fit also when singletons cascade."""
    rng = np.random.default_rng(8)
    n = 90
    frame = pd.DataFrame({"a": rng.integers(0, 30, size=n), "b": rng.integers(0, 12, size=n),
                          "x": rng.normal(size=n), "fw": rng.integers(1, 4, size=n).astype(float),
                          "cl": rng.integers(0, 9, size=n)})
    frame["y"] = frame.x + 0.1 * frame.a - 0.2 * frame.b + rng.normal(size=n)
    duplicated = frame.loc[frame.index.repeat(frame.fw.astype(int))].reset_index(drop=True)
    unweighted = oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b"])
    for extra in ({}, {"cluster": "cl"}, {"covariance": "robust"}):
        weighted = oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b"], weights="fw",
                              weight_type="fweight", **extra)
        rows = oe.reghdfe(data=duplicated, y="y", x=["x"], absorb=["a", "b"], **extra)
        assert 0 < weighted.metrics["n_singletons_dropped"] == rows.metrics["n_singletons_dropped"]
        assert weighted.metrics["n_singletons_dropped"] < unweighted.metrics["n_singletons_dropped"]
        assert weighted.nobs == rows.nobs
        assert_allclose(estimates(weighted), estimates(rows), rtol=1e-10)
        assert_allclose(errors(weighted), errors(rows), rtol=1e-10)
        assert weighted.metrics == pytest.approx(rows.metrics, rel=1e-10)
    # every dropped row had weight one; weighted rows always stay
    kept = set(weighted.sample_positions)
    assert all(index in kept for index in frame.index[frame.fw >= 2])
    # One row per absorbed level leaves nothing to estimate, whatever the frequency weights say.
    single = panel[panel.year < 1].assign(two=2.0)
    for arguments in ({}, {"weights": "two", "weight_type": "fweight"}):
        with pytest.raises(AnalysisError) as caught:
            oe.areg(data=single, y="y", x=["x1", "x2"], absorb="firm", **arguments)
        assert caught.value.code == "insufficient_observations"


def test_model_test_is_flagged_when_clusters_are_fewer_than_slopes():
    """With G - 1 < number of slopes the cluster covariance cannot test all slopes: the reduced
    test is kept but marked (Stata prints a missing F)."""
    rng = np.random.default_rng(2)
    n, k = 120, 8
    frame = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"z{i}" for i in range(k)])
    frame["firm"] = np.repeat(np.arange(24), 5)
    frame["g"] = np.repeat(np.arange(4), 30)
    frame["y"] = frame.z0 + 0.1 * frame.firm + rng.normal(size=n)
    columns = [f"z{i}" for i in range(k)]
    for result in (oe.areg(data=frame, y="y", x=columns, absorb="firm", cluster="g"),
                   oe.reghdfe(data=frame, y="y", x=columns, absorb=["firm"], cluster="g")):
        test = result.tests["model"]
        assert test["rank_deficient"] is True and test["restrictions"] == k and test["df"] == 3
        assert "rank-reduced to 3 of 8" in test["label"] and test["df2"] == 3
        assert any("Stata would report a missing F" in warning for warning in result.warnings)
    plain = oe.areg(data=frame, y="y", x=columns, absorb="firm", cluster="firm")
    assert "rank_deficient" not in plain.tests["model"] and plain.tests["model"]["df"] == k
