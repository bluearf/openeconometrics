"""t tests, variance tests and one-way ANOVA against SciPy / statsmodels and hand algebra."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
from statsmodels.stats.oneway import anova_oneway

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.frame import DataFrame


def _two_groups(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"y": rng.normal(1.0, 2.0, 60), "x": rng.normal(0.5, 1.0, 60),
                          "g": np.repeat(["a", "b"], [25, 35])})
    frame.loc[3, "y"] = np.nan
    return frame


def _four_groups(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = [12, 20, 9, 15]
    y = rng.normal(size=56) * np.repeat([1, 2, 1.5, 0.7], sizes) \
        + np.repeat([0, 0.8, 1.5, 0.2], sizes)
    return pd.DataFrame({"y": y, "g": np.repeat(["a", "b", "c", "d"], sizes)})


def _samples(frame: pd.DataFrame) -> list[np.ndarray]:
    return [group["y"].to_numpy() for _, group in frame.groupby("g")]


# ---- public surface ---------------------------------------------------------------


def test_functions_are_exported_and_return_tables():
    from openecon.econometrics import registry

    exports = registry.public_exports()
    for name in ("ttest", "sdtest", "oneway", "anova", "rm_anova", "manova", "correlate",
                 "pcorr", "describe"):
        assert exports[name][0].startswith("openecon.econometrics.stats.")
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
    result = oe.ttest(_two_groups(), "y", by="g")
    assert isinstance(result, TableSet)
    assert all(isinstance(table, DataFrame) for table in result.values())
    assert set(result) == {"statistics", "test", "levene", "effect_sizes"}


def test_results_render_and_serialize():
    result = oe.oneway(_four_groups(), "y", "g", posthoc=["tukey", "holm"])
    text = str(result)
    assert "One-way ANOVA of y by g" in text and "[posthoc_tukey]" in text
    latex = result.to_latex()
    assert latex.count(r"\begin{tabular}") == len(result) and "posthoc\\_tukey" in latex
    restored = json.loads(json.dumps(result.attrs))
    assert restored == result.attrs and restored["groups"] == ["a", "b", "c", "d"]
    for table in result.values():
        payload = json.loads(table.to_json())
        assert set(payload) == set(table.columns)
    # Undefined cells are missing numbers, never text or infinities.
    holm = result["posthoc_holm"]
    assert holm["ci_low"].dtype == np.float64 and holm["ci_low"].isna().all()
    for table in result.values():
        numeric = table.select_dtypes("number").to_numpy(dtype=float)
        assert not np.isinf(numeric).any()


def test_accepts_mappings_and_records():
    frame = _two_groups().dropna()
    expected = oe.ttest(frame, "y", by="g").attrs["statistic"]
    assert oe.ttest(frame.to_dict("list"), "y", by="g").attrs["statistic"] == expected
    assert oe.ttest(frame.to_dict("records"), "y", by="g").attrs["statistic"] == expected


# ---- t tests ----------------------------------------------------------------------


def test_one_sample_ttest_matches_scipy_and_hand_formulas():
    frame = _two_groups()
    y = frame["y"].dropna().to_numpy()
    result = oe.ttest(frame, "y", mu=0.5)
    reference = stats.ttest_1samp(y, 0.5)
    row = result["test"].loc["one_sample"]
    assert row["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert row["p_value"] == pytest.approx(reference.pvalue, rel=1e-11)
    assert row["df"] == 58 and result.attrs["n"] == 59 and result.attrs["n_missing"] == 1
    low, high = reference.confidence_interval(0.95)
    assert row["ci_low"] == pytest.approx(low - 0.5) and row["ci_high"] == pytest.approx(high - 0.5)
    greater = stats.ttest_1samp(y, 0.5, alternative="greater").pvalue
    assert row["p_greater"] == pytest.approx(greater)
    assert row["p_less"] == pytest.approx(stats.ttest_1samp(y, 0.5, alternative="less").pvalue)
    assert result.attrs["one_sided_p_value"] == pytest.approx(reference.pvalue / 2)
    d = (y.mean() - 0.5) / y.std(ddof=1)
    factor = math.exp(special.gammaln(29.0) - special.gammaln(28.5)) / math.sqrt(29.0)
    assert result.attrs["cohens_d"] == pytest.approx(d)
    assert result.attrs["hedges_g"] == pytest.approx(d * factor)
    stat = result["statistics"].loc["y"]
    assert stat["std_error"] == pytest.approx(stats.sem(y))
    interval = stats.t.interval(0.9, 58, y.mean(), stats.sem(y))
    ninety = oe.ttest(frame, "y", alpha=0.10)["statistics"].loc["y"]
    assert (ninety["ci_low"], ninety["ci_high"]) == pytest.approx(interval)


def test_paired_ttest_matches_scipy():
    frame = _two_groups()
    complete = frame.dropna()
    result = oe.ttest(frame, "y", paired_with="x")
    reference = stats.ttest_rel(complete["y"], complete["x"])
    row = result["test"].loc["paired"]
    assert row["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert row["p_value"] == pytest.approx(reference.pvalue, rel=1e-11)
    assert list(result["statistics"].index) == ["y", "x", "difference"]
    r, p = stats.pearsonr(complete["y"], complete["x"])
    assert result.attrs["correlation"] == pytest.approx(r)
    assert result.attrs["correlation_p_value"] == pytest.approx(p)
    difference = (complete["y"] - complete["x"]).to_numpy()
    assert result.attrs["cohens_d"] == pytest.approx(difference.mean() / difference.std(ddof=1))
    shifted = oe.ttest(frame, "y", paired_with="x", mu=0.25)
    assert shifted.attrs["statistic"] == pytest.approx(
        stats.ttest_1samp(difference, 0.25).statistic)


def test_independent_ttest_pooled_welch_levene_and_effect_sizes():
    frame = _two_groups()
    complete = frame.dropna()
    a = complete.loc[complete["g"] == "a", "y"].to_numpy()
    b = complete.loc[complete["g"] == "b", "y"].to_numpy()
    result = oe.ttest(frame, "y", by="g")
    pooled, welch = stats.ttest_ind(a, b), stats.ttest_ind(a, b, equal_var=False)
    test = result["test"]
    for label, reference in (("equal_variances", pooled), ("unequal_variances", welch)):
        assert test.loc[label, "statistic"] == pytest.approx(reference.statistic, rel=1e-12)
        assert test.loc[label, "p_value"] == pytest.approx(reference.pvalue, rel=1e-11)
        assert test.loc[label, "df"] == pytest.approx(reference.df, rel=1e-12)
        low, high = reference.confidence_interval(0.95)
        assert test.loc[label, "ci_low"] == pytest.approx(low)
        assert test.loc[label, "ci_high"] == pytest.approx(high)
    levene = stats.levene(a, b, center="mean")
    assert result["levene"].loc["levene_mean", "statistic"] == pytest.approx(levene.statistic)
    assert result["levene"].loc["levene_mean", "p_value"] == pytest.approx(levene.pvalue)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    df = n1 + n2 - 2
    factor = math.exp(special.gammaln(df / 2) - special.gammaln((df - 1) / 2)) / math.sqrt(df / 2)
    assert result.attrs["cohens_d"] == pytest.approx(d)
    assert result.attrs["hedges_g"] == pytest.approx(d * factor)
    assert factor == pytest.approx(1 - 3 / (4 * df - 1), abs=2e-5)     # the usual approximation
    assert result.attrs["glass_delta"] == pytest.approx((a.mean() - b.mean()) / b.std(ddof=1))
    assert list(result["statistics"].index) == ["a", "b", "combined"]
    assert result["statistics"].loc["combined", "n"] == 59
    shifted = oe.ttest(frame, "y", by="g", mu=0.3)["test"].loc["equal_variances"]
    assert shifted["statistic"] == pytest.approx(
        (a.mean() - b.mean() - 0.3) / (sp * math.sqrt(1 / n1 + 1 / n2)))


def test_textbook_two_sample_example():
    data = {"score": [5.1, 4.9, 6.2, 5.8, 6.9, 7.4, 6.6, 7.1], "group": list("aaaabbbb")}
    result = oe.ttest(data, "score", by="group")
    # means 5.5 and 7.0, pooled variance (1.10 + 0.34) / 6 = 0.24, se = sqrt(0.24 / 2)
    assert result.attrs["statistic"] == pytest.approx(-1.5 / math.sqrt(0.12))
    assert result.attrs["df"] == 6


def test_numeric_and_boolean_group_labels_sort_ascending():
    frame = _two_groups().dropna()
    frame["code"] = np.where(frame["g"] == "a", 10, 2)
    result = oe.ttest(frame, "y", by="code")
    assert result.attrs["groups"] == [2, 10]
    assert result.attrs["statistic"] == pytest.approx(-oe.ttest(frame, "y", by="g")
                                                      .attrs["statistic"])


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"y": "y", "by": "g", "paired_with": "x"}, "invalid_spec"),
    ({"y": "nope"}, "missing_columns"),
    ({"y": "g"}, "non_numeric_column"),
    ({"y": "y", "by": "three"}, "invalid_groups"),
    ({"y": "y", "alpha": 1.5}, "invalid_option"),
    ({"y": "y", "mu": float("nan")}, "invalid_option"),
    ({"y": "y", "missing": "raise"}, "missing_values"),
    ({"y": "y", "missing": "ignore"}, "invalid_option"),
    ({"y": "constant"}, "zero_variance"),
    ({"y": "constant", "by": "g"}, "zero_variance"),
    ({"y": ["y"]}, "invalid_spec"),
    ({"y": "infinite"}, "non_finite_values"),
])
def test_ttest_error_codes(kwargs, code):
    frame = _two_groups()
    frame["three"] = np.tile(["p", "q", "r"], 20)
    frame["constant"] = 3.0
    frame["infinite"] = np.where(np.arange(60) == 5, np.inf, 1.0)
    with pytest.raises(AnalysisError) as error:
        oe.ttest(frame, **kwargs)
    assert error.value.code == code


def test_ttest_needs_enough_observations():
    with pytest.raises(AnalysisError) as error:
        oe.ttest({"y": [1.0]}, "y")
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.ttest({"y": [1.0, 2.0, 3.0], "g": ["a", "b", "b"]}, "y", by="g")
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.ttest({"y": [None, None]}, "y")
    assert error.value.code in {"empty_sample", "non_numeric_column"}
    with pytest.raises(AnalysisError) as error:
        oe.ttest([], "y")
    assert error.value.code in {"empty_data", "missing_columns"}


# ---- variance tests ---------------------------------------------------------------


def _levene_oracle(samples: list[np.ndarray], centers: list[float]) -> tuple[float, float]:
    z = [np.abs(sample - center) for sample, center in zip(samples, centers, strict=True)]
    return tuple(stats.f_oneway(*z))


def test_sdtest_two_groups_matches_f_ratio_and_robvar():
    frame = _two_groups()
    complete = frame.dropna()
    a = complete.loc[complete["g"] == "a", "y"].to_numpy()
    b = complete.loc[complete["g"] == "b", "y"].to_numpy()
    result = oe.sdtest(frame, "y", by="g")
    ratio = result["variance_ratio"].loc["f"]
    f = a.var(ddof=1) / b.var(ddof=1)
    assert ratio["statistic"] == pytest.approx(f)
    assert (ratio["df1"], ratio["df2"]) == (len(a) - 1, len(b) - 1)
    lower = stats.f.cdf(f, len(a) - 1, len(b) - 1)
    assert ratio["p_less"] == pytest.approx(lower)
    assert ratio["p_greater"] == pytest.approx(1 - lower)
    assert ratio["p_value"] == pytest.approx(2 * min(lower, 1 - lower))
    robust = result["robust"]
    w0, w50 = stats.levene(a, b, center="mean"), stats.levene(a, b, center="median")
    assert robust.loc["w0", "statistic"] == pytest.approx(w0.statistic)
    assert robust.loc["w50", "statistic"] == pytest.approx(w50.statistic)
    assert robust.loc["w50", "p_value"] == pytest.approx(w50.pvalue)
    # W10: deviations of ALL observations from the mean of the middle 90% (Stata robvar).
    trimmed = [np.sort(s)[int(0.05 * len(s)):len(s) - int(0.05 * len(s))].mean() for s in (a, b)]
    statistic, p_value = _levene_oracle([a, b], trimmed)
    assert robust.loc["w10", "statistic"] == pytest.approx(statistic)
    assert robust.loc["w10", "p_value"] == pytest.approx(p_value)
    assert result.attrs["statistic"] == pytest.approx(w0.statistic)
    by_median = oe.sdtest(frame, "y", by="g", center="median")
    assert by_median.attrs["p_value"] == pytest.approx(w50.pvalue)
    assert oe.sdtest(frame, "y", by="g", center="trimmed").attrs["statistic"] \
        == pytest.approx(statistic)


def test_sdtest_many_groups_and_one_sample():
    frame = _four_groups()
    samples = _samples(frame)
    result = oe.sdtest(frame, "y", by="g")
    assert "variance_ratio" not in result and result.attrs["notes"]
    for label, center in (("w0", "mean"), ("w50", "median")):
        reference = stats.levene(*samples, center=center)
        assert result["robust"].loc[label, "statistic"] == pytest.approx(reference.statistic)
        assert result["robust"].loc[label, "p_value"] == pytest.approx(reference.pvalue)
    y = frame["y"].to_numpy()
    one = oe.sdtest(frame, "y", sd=1.2)
    chi2 = (len(y) - 1) * y.var(ddof=1) / 1.2 ** 2
    assert one.attrs["statistic"] == pytest.approx(chi2) and one.attrs["df"] == len(y) - 1
    lower = stats.chi2.cdf(chi2, len(y) - 1)
    assert one["chi2"].loc["chi2", "p_less"] == pytest.approx(lower)
    assert one.attrs["p_value"] == pytest.approx(2 * min(lower, 1 - lower))


@pytest.mark.parametrize(("kwargs", "code"), [
    ({}, "invalid_spec"), ({"by": "g", "sd": 1.0}, "invalid_spec"),
    ({"sd": -1.0}, "invalid_option"),
    ({"by": "g", "center": "mode"}, "invalid_option"), ({"by": "one"}, "invalid_groups"),
    ({"by": "single"}, "insufficient_observations")])
def test_sdtest_error_codes(kwargs, code):
    frame = _four_groups()
    frame["one"] = "same"
    frame["single"] = np.where(np.arange(len(frame)) == 0, "lonely", "rest")
    with pytest.raises(AnalysisError) as error:
        oe.sdtest(frame, "y", **kwargs)
    assert error.value.code == code


# ---- one-way ANOVA ----------------------------------------------------------------


def test_oneway_hand_computed_example():
    data = {"y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "g": list("aaabbbccc")}
    anova = oe.oneway(data, "y", "g")["anova"]
    # group means 2, 3, 6 around 11/3: SS_between = 26, SS_within = 6, F = 13.
    assert anova.loc["between", ["ss", "df", "ms", "statistic"]].tolist() \
        == pytest.approx([26.0, 2, 13.0, 13.0])
    assert anova.loc["within", ["ss", "df", "ms"]].tolist() == pytest.approx([6.0, 6, 1.0])
    assert anova.loc["total", ["ss", "df"]].tolist() == pytest.approx([32.0, 8])


def test_oneway_tables_match_scipy_and_statsmodels():
    frame = _four_groups()
    samples = _samples(frame)
    result = oe.oneway(frame, "y", "g")
    reference = stats.f_oneway(*samples)
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-11)
    assert (result.attrs["df1"], result.attrs["df2"]) == (3, 52)
    descriptives = result["descriptives"]
    assert list(descriptives.index) == ["a", "b", "c", "d", "Total"]
    for label, sample in zip("abcd", samples, strict=True):
        row = descriptives.loc[label]
        assert row["n"] == len(sample) and row["mean"] == pytest.approx(sample.mean())
        assert row["std_dev"] == pytest.approx(sample.std(ddof=1))
        interval = stats.t.interval(0.95, len(sample) - 1, sample.mean(), stats.sem(sample))
        assert (row["ci_low"], row["ci_high"]) == pytest.approx(interval)
        assert (row["min"], row["max"]) == (sample.min(), sample.max())
    assert descriptives.loc["Total", "std_dev"] == pytest.approx(frame["y"].std())
    homogeneity = result["homogeneity"]
    for label, center in (("levene_mean", "mean"), ("levene_median", "median")):
        check = stats.levene(*samples, center=center)
        assert homogeneity.loc[label, "statistic"] == pytest.approx(check.statistic)
        assert homogeneity.loc[label, "p_value"] == pytest.approx(check.pvalue)
    bartlett = stats.bartlett(*samples)
    assert homogeneity.loc["bartlett", "statistic"] == pytest.approx(bartlett.statistic)
    assert homogeneity.loc["bartlett", "p_value"] == pytest.approx(bartlett.pvalue)
    # Satterthwaite df of the median-based test: (sum u_i)^2 / sum u_i^2 / (n_i - 1).
    u = np.array([((np.abs(s - np.median(s)) - np.abs(s - np.median(s)).mean()) ** 2).sum()
                  for s in samples])
    adjusted = u.sum() ** 2 / (u ** 2 / (np.array([len(s) for s in samples]) - 1)).sum()
    row = homogeneity.loc["levene_median_adjusted_df"]
    assert row["df2"] == pytest.approx(adjusted)
    assert row["p_value"] == pytest.approx(stats.f.sf(row["statistic"], 3, adjusted))
    welch = anova_oneway(samples, use_var="unequal", welch_correction=True)
    assert result["robust"].loc["welch", "statistic"] == pytest.approx(welch.statistic)
    assert result["robust"].loc["welch", "df2"] == pytest.approx(welch.df_denom)
    assert result["robust"].loc["welch", "p_value"] == pytest.approx(welch.pvalue)
    brown = anova_oneway(samples, use_var="bf")
    assert result["robust"].loc["brown_forsythe", "statistic"] == pytest.approx(brown.statistic)
    assert result["robust"].loc["brown_forsythe", "df2"] == pytest.approx(brown.df_denom)
    assert result["robust"].loc["brown_forsythe", "p_value"] == pytest.approx(brown.pvalue2)
    ssb = result["anova"].loc["between", "ss"]
    sst, msw = result["anova"].loc["total", "ss"], result["anova"].loc["within", "ms"]
    effects = result["effect_sizes"]["estimate"]
    assert effects["eta_squared"] == pytest.approx(ssb / sst)
    assert effects["epsilon_squared"] == pytest.approx((ssb - 3 * msw) / sst)
    assert effects["omega_squared"] == pytest.approx((ssb - 3 * msw) / (sst + msw))
    assert "robust" not in oe.oneway(frame, "y", "g", welch=False)


def test_two_group_oneway_equals_squared_t():
    frame = _two_groups()
    assert oe.oneway(frame, "y", "g").attrs["statistic"] == pytest.approx(
        oe.ttest(frame, "y", by="g").attrs["statistic"] ** 2)


def test_posthoc_tukey_games_howell_and_dunnett_match_scipy():
    frame = _four_groups()
    samples = _samples(frame)
    result = oe.oneway(frame, "y", "g", posthoc=["tukey", "games_howell", "dunnett"])
    upper = np.triu_indices(4, 1)
    tukey = stats.tukey_hsd(*samples)
    table = result["posthoc_tukey"]
    assert list(zip(table["group_i"], table["group_j"], strict=True))[:3] \
        == [("a", "b"), ("a", "c"), ("a", "d")]
    np.testing.assert_allclose(table["mean_difference"], tukey.statistic[upper], rtol=1e-12)
    np.testing.assert_allclose(table["p_value"], tukey.pvalue[upper], rtol=1e-8)
    interval = tukey.confidence_interval(0.95)
    np.testing.assert_allclose(table["ci_low"], interval.low[upper], rtol=1e-8)
    np.testing.assert_allclose(table["ci_high"], interval.high[upper], rtol=1e-8)
    games = stats.tukey_hsd(*samples, equal_var=False)
    table = result["posthoc_games_howell"]
    np.testing.assert_allclose(table["p_value"], games.pvalue[upper], rtol=1e-8)
    interval = games.confidence_interval(0.95)
    np.testing.assert_allclose(table["ci_low"], interval.low[upper], rtol=1e-8)
    n = np.array([len(s) for s in samples])
    v = np.array([s.var(ddof=1) for s in samples])
    i, j = upper
    welch_df = (v[i] / n[i] + v[j] / n[j]) ** 2 / ((v[i] / n[i]) ** 2 / (n[i] - 1)
                                                   + (v[j] / n[j]) ** 2 / (n[j] - 1))
    np.testing.assert_allclose(table["df"], welch_df, rtol=1e-12)
    # SciPy's Dunnett uses randomized quasi-Monte Carlo (about 1e-4 accurate).
    dunnett = stats.dunnett(*samples[1:], control=samples[0], random_state=1)
    table = result["posthoc_dunnett"]
    assert list(table["group_j"]) == ["a", "a", "a"] and table.attrs["control"] == "a"
    np.testing.assert_allclose(table["p_value"], dunnett.pvalue, atol=2e-3)
    np.testing.assert_allclose(table["ci_low"], dunnett.confidence_interval().low, atol=1e-2)
    last = oe.oneway(frame, "y", "g", posthoc=["dunnett"], control="d")["posthoc_dunnett"]
    assert list(last["group_i"]) == ["a", "b", "c"] and set(last["group_j"]) == {"d"}


def test_posthoc_t_based_adjustments():
    frame = _four_groups()
    samples = _samples(frame)
    result = oe.oneway(frame, "y", "g", posthoc=["lsd", "bonferroni", "sidak", "scheffe", "holm"],
                       alpha=0.1)
    n = np.array([len(s) for s in samples])
    means = np.array([s.mean() for s in samples])
    mse = result["anova"].loc["within", "ms"]
    i, j = np.triu_indices(4, 1)
    se = np.sqrt(mse * (1 / n[i] + 1 / n[j]))
    t = (means[i] - means[j]) / se
    raw = 2 * stats.t.sf(np.abs(t), 52)
    lsd = result["posthoc_lsd"]
    np.testing.assert_allclose(lsd["statistic"], t)
    np.testing.assert_allclose(lsd["std_error"], se)
    np.testing.assert_allclose(lsd["p_value"], raw)
    np.testing.assert_allclose(lsd["ci_high"], means[i] - means[j] + stats.t.ppf(0.95, 52) * se)
    bonferroni = result["posthoc_bonferroni"]
    np.testing.assert_allclose(bonferroni["p_value"], np.minimum(1, 6 * raw))
    np.testing.assert_allclose(bonferroni["ci_low"],
                               means[i] - means[j] - stats.t.ppf(1 - 0.1 / 12, 52) * se)
    sidak = result["posthoc_sidak"]
    np.testing.assert_allclose(sidak["p_value"], 1 - (1 - raw) ** 6)
    level = 1 - (1 - 0.1) ** (1 / 6)
    np.testing.assert_allclose(sidak["ci_low"],
                               means[i] - means[j] - stats.t.ppf(1 - level / 2, 52) * se)
    scheffe = result["posthoc_scheffe"]
    np.testing.assert_allclose(scheffe["p_value"], stats.f.sf(t ** 2 / 3, 3, 52))
    np.testing.assert_allclose(
        scheffe["ci_high"], means[i] - means[j] + np.sqrt(3 * stats.f.ppf(0.9, 3, 52)) * se)
    order = np.argsort(raw)
    holm = np.empty(6)
    holm[order] = np.minimum(1, np.maximum.accumulate((6 - np.arange(6)) * raw[order]))
    np.testing.assert_allclose(result["posthoc_holm"]["p_value"], holm)
    from statsmodels.stats.multitest import multipletests
    np.testing.assert_allclose(holm, multipletests(raw, method="holm")[1])


def test_oneway_missing_values_and_small_groups():
    frame = _four_groups()
    frame.loc[[0, 30], "y"] = np.nan
    frame.loc[5, "g"] = None
    result = oe.oneway(frame, "y", "g")
    assert result.attrs["n_missing"] == 3 and result.attrs["n"] == 53
    assert result.attrs["statistic"] == pytest.approx(
        stats.f_oneway(*_samples(frame.dropna())).statistic)
    with pytest.raises(AnalysisError) as error:
        oe.oneway(frame, "y", "g", missing="raise")
    assert error.value.code == "missing_values"
    tiny = {"y": [1.0, 2.0, 4.0, 3.0, 7.0], "g": ["a", "a", "b", "b", "c"]}
    result = oe.oneway(tiny, "y", "g", posthoc=["tukey"])
    assert np.isnan(result["descriptives"].loc["c", "std_dev"])
    assert np.isnan(result["robust"].loc["welch", "statistic"]) and result.attrs["notes"]
    assert result["posthoc_tukey"]["p_value"].between(0, 1).all()
    with pytest.raises(AnalysisError) as error:
        oe.oneway(tiny, "y", "g", posthoc=["games_howell"])
    assert error.value.code == "zero_variance"


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"posthoc": "tukey"}, "invalid_spec"), ({"posthoc": ["duncan"]}, "invalid_option"),
    ({"posthoc": ["dunnett"], "control": "zzz"}, "invalid_option"),
    ({"control": "a"}, "invalid_option"), ({"welch": "yes"}, "invalid_option"),
    ({"alpha": 0}, "invalid_option")])
def test_oneway_option_errors(kwargs, code):
    with pytest.raises(AnalysisError) as error:
        oe.oneway(_four_groups(), "y", "g", **kwargs)
    assert error.value.code == code


def test_oneway_degenerate_designs():
    with pytest.raises(AnalysisError) as error:
        oe.oneway({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "a"]}, "y", "g")
    assert error.value.code == "invalid_groups"
    with pytest.raises(AnalysisError) as error:
        oe.oneway({"y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}, "y", "g")
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.oneway({"y": [1.0, 1.0, 3.0, 3.0], "g": ["a", "a", "b", "b"]}, "y", "g")
    assert error.value.code == "zero_variance"


def test_many_groups_tukey_runs_fast_and_matches_scipy_spot_checks():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"g": np.repeat(np.arange(40), 6)})
    frame["y"] = rng.normal(size=240) + 0.02 * frame["g"]
    table = oe.oneway(frame, "y", "g", posthoc=["tukey"])["posthoc_tukey"]
    assert len(table) == 780
    q = np.abs(table["statistic"].to_numpy()[:5]) * math.sqrt(2)
    np.testing.assert_allclose(table["p_value"].to_numpy()[:5],
                               stats.studentized_range.sf(q, 40, 200), atol=1e-9)
