"""Independent oracles for t tests, variance tests, one-way ANOVA with post hoc
comparisons, correlations and descriptives (verification pass).

Every expected value is derived here from explicit NumPy algebra or taken from
SciPy / statsmodels; none of the implementation's helpers is used.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

import openecon as oe

RTOL = 1e-9


def _hedges(df: float) -> float:
    return math.exp(special.gammaln(df / 2) - special.gammaln((df - 1) / 2)) / math.sqrt(df / 2)


def _sample(seed: int = 0, n: int = 57) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "y": rng.normal(10.0, 2.0, size=n),
        "z": rng.normal(9.5, 3.0, size=n),
        "g": rng.choice(["ctl", "trt"], size=n, p=[0.4, 0.6]),
        "k": rng.choice(["k1", "k2", "k3", "k4"], size=n, p=[0.15, 0.35, 0.3, 0.2]),
    })
    frame["z"] += 0.5 * frame["y"]
    frame.loc[frame["g"] == "trt", "y"] *= 1.6
    frame.loc[rng.choice(n, 6, replace=False), "y"] = np.nan
    frame.loc[rng.choice(n, 5, replace=False), "z"] = np.nan
    return frame


# ---- t tests ----------------------------------------------------------------------------


@pytest.mark.parametrize("mu, alpha", [(0.0, 0.05), (9.0, 0.1), (-3.5, 0.01)])
def test_one_sample_t(mu, alpha):
    frame = _sample()
    x = frame["y"].dropna().to_numpy()
    result = oe.ttest(frame, "y", mu=mu, alpha=alpha)
    oracle = stats.ttest_1samp(x, mu)
    row = result["test"].loc["one_sample"]
    assert row["statistic"] == pytest.approx(oracle.statistic, rel=RTOL)
    assert row["df"] == len(x) - 1 == oracle.df
    assert row["p_value"] == pytest.approx(oracle.pvalue, rel=1e-8)
    assert row["p_less"] == pytest.approx(stats.ttest_1samp(x, mu, alternative="less").pvalue,
                                          rel=1e-8)
    assert row["p_greater"] == pytest.approx(
        stats.ttest_1samp(x, mu, alternative="greater").pvalue, rel=1e-8)
    interval = oracle.confidence_interval(1 - alpha)
    assert row["mean_difference"] == pytest.approx(x.mean() - mu, rel=RTOL)
    assert row["ci_low"] == pytest.approx(interval.low - mu, rel=1e-8)
    assert row["ci_high"] == pytest.approx(interval.high - mu, rel=1e-8)
    assert row["std_error"] == pytest.approx(x.std(ddof=1) / math.sqrt(len(x)), rel=RTOL)
    described = result["statistics"].loc["y"]
    assert described["n"] == len(x)
    assert described["mean"] == pytest.approx(x.mean(), rel=RTOL)
    assert described["std_dev"] == pytest.approx(x.std(ddof=1), rel=RTOL)
    assert described["ci_low"] == pytest.approx(interval.low, rel=1e-8)
    assert described["ci_high"] == pytest.approx(interval.high, rel=1e-8)
    d = (x.mean() - mu) / x.std(ddof=1)
    assert result.attrs["cohens_d"] == pytest.approx(d, rel=RTOL)
    assert result.attrs["hedges_g"] == pytest.approx(d * _hedges(len(x) - 1), rel=RTOL)
    assert result.attrs["n"] == len(x) and result.attrs["n_missing"] == frame["y"].isna().sum()
    assert result.attrs["one_sided_p_value"] == pytest.approx(oracle.pvalue / 2, rel=1e-8)


def test_paired_t_uses_complete_pairs():
    frame = _sample(seed=3)
    pairs = frame[["y", "z"]].dropna()
    a, b = pairs["y"].to_numpy(), pairs["z"].to_numpy()
    result = oe.ttest(frame, "y", paired_with="z", mu=0.25, alpha=0.1)
    oracle = stats.ttest_rel(a - 0.25, b)
    row = result["test"].loc["paired"]
    assert row["statistic"] == pytest.approx(oracle.statistic, rel=RTOL)
    assert row["df"] == len(a) - 1
    assert row["p_value"] == pytest.approx(oracle.pvalue, rel=1e-8)
    interval = oracle.confidence_interval(0.9)
    assert row["ci_low"] == pytest.approx(interval.low, rel=1e-8)
    assert row["ci_high"] == pytest.approx(interval.high, rel=1e-8)
    table = result["statistics"]
    assert list(table.index) == ["y", "z", "difference"]
    assert table.loc["y", "mean"] == pytest.approx(a.mean(), rel=RTOL)
    assert table.loc["z", "std_dev"] == pytest.approx(b.std(ddof=1), rel=RTOL)
    assert table.loc["difference", "mean"] == pytest.approx((a - b).mean(), rel=RTOL)
    assert (table["n"] == len(a)).all()
    r = stats.pearsonr(a, b)
    assert result.attrs["correlation"] == pytest.approx(r.statistic, rel=RTOL)
    assert result.attrs["correlation_p_value"] == pytest.approx(r.pvalue, rel=1e-8)
    d = ((a - b).mean() - 0.25) / (a - b).std(ddof=1)
    assert result.attrs["cohens_d"] == pytest.approx(d, rel=RTOL)
    assert result.attrs["hedges_g"] == pytest.approx(d * _hedges(len(a) - 1), rel=RTOL)
    assert result.attrs["n_missing"] == len(frame) - len(pairs)


@pytest.mark.parametrize("mu", [0.0, -2.0])
def test_independent_samples_t(mu):
    frame = _sample(seed=5, n=64)
    used = frame[["y", "g"]].dropna()
    first = used.loc[used["g"] == "ctl", "y"].to_numpy()
    second = used.loc[used["g"] == "trt", "y"].to_numpy()
    result = oe.ttest(frame, "y", by="g", mu=mu, alpha=0.02)
    test = result["test"]
    for name, equal in (("equal_variances", True), ("unequal_variances", False)):
        oracle = stats.ttest_ind(first - mu, second, equal_var=equal)
        row = test.loc[name]
        assert row["statistic"] == pytest.approx(oracle.statistic, rel=RTOL)
        assert row["df"] == pytest.approx(oracle.df, rel=RTOL)
        assert row["p_value"] == pytest.approx(oracle.pvalue, rel=1e-8)
        interval = oracle.confidence_interval(0.98)
        assert row["ci_low"] == pytest.approx(interval.low, rel=1e-8)
        assert row["ci_high"] == pytest.approx(interval.high, rel=1e-8)
        less = stats.ttest_ind(first - mu, second, equal_var=equal, alternative="less")
        assert row["p_less"] == pytest.approx(less.pvalue, rel=1e-8)
        assert row["p_greater"] == pytest.approx(1 - less.pvalue, rel=1e-8)
        assert row["mean_difference"] == pytest.approx(first.mean() - second.mean() - mu,
                                                       rel=RTOL)
    n1, n2 = len(first), len(second)
    assert test.loc["equal_variances", "df"] == n1 + n2 - 2
    levene = stats.levene(first, second, center="mean")
    assert result["levene"].loc["levene_mean", "statistic"] == pytest.approx(levene.statistic,
                                                                            rel=RTOL)
    assert result["levene"].loc["levene_mean", "p_value"] == pytest.approx(levene.pvalue,
                                                                          rel=1e-8)
    assert result["levene"].loc["levene_mean", "df2"] == n1 + n2 - 2
    pooled = math.sqrt(((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1))
                       / (n1 + n2 - 2))
    gap = first.mean() - second.mean() - mu
    assert result.attrs["cohens_d"] == pytest.approx(gap / pooled, rel=RTOL)
    assert result.attrs["hedges_g"] == pytest.approx(gap / pooled * _hedges(n1 + n2 - 2),
                                                     rel=RTOL)
    assert result.attrs["glass_delta"] == pytest.approx(gap / second.std(ddof=1), rel=RTOL)
    assert result.attrs["glass_delta_first"] == pytest.approx(gap / first.std(ddof=1), rel=RTOL)
    table = result["statistics"]
    assert list(table.index) == ["ctl", "trt", "combined"]
    assert list(table["n"]) == [n1, n2, n1 + n2]
    both = np.concatenate([first, second])
    assert table.loc["combined", "std_dev"] == pytest.approx(both.std(ddof=1), rel=RTOL)
    assert table.loc["ctl", "ci_high"] == pytest.approx(
        first.mean() + stats.t.ppf(0.99, n1 - 1) * first.std(ddof=1) / math.sqrt(n1), rel=1e-8)


def test_t_tests_are_invariant_to_row_order_and_affine_changes_of_units():
    frame = _sample(seed=8)
    base = oe.ttest(frame, "y", by="g")
    shuffled = oe.ttest(frame.sample(frac=1.0, random_state=4), "y", by="g")
    scaled = oe.ttest(frame.assign(y=frame["y"] * 1e8 + 3e9), "y", by="g")
    tiny = oe.ttest(frame.assign(y=frame["y"] * 1e-8), "y", by="g")
    for other, factor in ((shuffled, 1.0), (scaled, 1e8), (tiny, 1e-8)):
        np.testing.assert_allclose(other["test"][["statistic", "df", "p_value"]],
                                   base["test"][["statistic", "df", "p_value"]], rtol=1e-7)
        np.testing.assert_allclose(other["test"]["mean_difference"],
                                   base["test"]["mean_difference"] * factor, rtol=1e-7)
        np.testing.assert_allclose(other["levene"]["statistic"], base["levene"]["statistic"],
                                   rtol=1e-6)
        assert other.attrs["cohens_d"] == pytest.approx(base.attrs["cohens_d"], rel=1e-7)
    # Swapping the labels reverses the difference; duplicating every row keeps the means.
    swapped = oe.ttest(frame.assign(g=frame["g"].map({"ctl": "z", "trt": "a"})), "y", by="g")
    assert swapped.attrs["statistic"] == pytest.approx(-base.attrs["statistic"], rel=RTOL)
    assert swapped.attrs["p_less"] == pytest.approx(base.attrs["p_greater"], rel=1e-8)
    doubled = oe.ttest(pd.concat([frame, frame], ignore_index=True), "y", by="g")
    assert doubled["statistics"].loc["ctl", "mean"] == pytest.approx(
        base["statistics"].loc["ctl", "mean"], rel=RTOL)
    assert doubled["statistics"].loc["ctl", "n"] == 2 * base["statistics"].loc["ctl", "n"]


# ---- tests of variances -------------------------------------------------------------------


def _w_statistic(groups: list[np.ndarray], centers: list[float]) -> tuple[float, int, int]:
    """One-way ANOVA F of the absolute deviations from the given centres."""
    z = [np.abs(g - c) for g, c in zip(groups, centers, strict=True)]
    f, _ = stats.f_oneway(*z)
    return float(f), len(groups) - 1, sum(len(g) for g in groups) - len(groups)


def test_variance_ratio_and_robust_tests_two_groups():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({"g": np.repeat(["a", "b"], [47, 63])})
    frame["y"] = np.where(frame["g"] == "a", rng.normal(0, 1, 110), rng.standard_t(4, 110) * 1.5)
    result = oe.sdtest(frame, "y", by="g", center="trimmed")
    first = frame.loc[frame["g"] == "a", "y"].to_numpy()
    second = frame.loc[frame["g"] == "b", "y"].to_numpy()
    f = first.var(ddof=1) / second.var(ddof=1)
    row = result["variance_ratio"].loc["f"]
    assert row["statistic"] == pytest.approx(f, rel=RTOL)
    assert (row["df1"], row["df2"]) == (46, 62)
    lower, upper = stats.f.cdf(f, 46, 62), stats.f.sf(f, 46, 62)
    assert row["p_less"] == pytest.approx(lower, rel=1e-8)
    assert row["p_greater"] == pytest.approx(upper, rel=1e-8)
    assert row["p_value"] == pytest.approx(2 * min(lower, upper), rel=1e-8)
    robust = result["robust"]
    for name, center in (("w0", "mean"), ("w50", "median")):
        oracle = stats.levene(first, second, center=center)
        assert robust.loc[name, "statistic"] == pytest.approx(oracle.statistic, rel=1e-8)
        assert robust.loc[name, "p_value"] == pytest.approx(oracle.pvalue, rel=1e-7)
    # W10: ALL observations, deviations from the mean of the middle 90% (floor(0.05 n) cut
    # from each tail). SciPy's 'trimmed' variant drops the trimmed observations instead.
    centers = []
    for group in (first, second):
        cut = int(math.floor(0.05 * len(group)))
        centers.append(np.sort(group)[cut:len(group) - cut].mean())
    statistic, df1, df2 = _w_statistic([first, second], centers)
    assert robust.loc["w10", "statistic"] == pytest.approx(statistic, rel=1e-8)
    assert robust.loc["w10", "p_value"] == pytest.approx(stats.f.sf(statistic, df1, df2),
                                                         rel=1e-7)
    assert result.attrs["statistic"] == pytest.approx(statistic, rel=1e-8)
    assert (result.attrs["df1"], result.attrs["df2"]) == (1, 108)


def test_robust_variance_tests_many_groups_and_one_sample_chi_square():
    frame = _sample(seed=20, n=140)
    used = frame[["y", "k"]].dropna()
    groups = [used.loc[used["k"] == level, "y"].to_numpy() for level in sorted(used["k"].unique())]
    result = oe.sdtest(frame, "y", by="k", center="median")
    assert "variance_ratio" not in result
    for name, center in (("w0", "mean"), ("w50", "median")):
        oracle = stats.levene(*groups, center=center)
        assert result["robust"].loc[name, "statistic"] == pytest.approx(oracle.statistic,
                                                                       rel=1e-8)
        assert result["robust"].loc[name, "p_value"] == pytest.approx(oracle.pvalue, rel=1e-7)
    assert result.attrs["statistic"] == result["robust"].loc["w50", "statistic"]
    x = frame["y"].dropna().to_numpy()
    one = oe.sdtest(frame, "y", sd=3.0)
    chi2 = (len(x) - 1) * x.var(ddof=1) / 9.0
    lower, upper = stats.chi2.cdf(chi2, len(x) - 1), stats.chi2.sf(chi2, len(x) - 1)
    row = one["chi2"].loc["chi2"]
    assert row["statistic"] == pytest.approx(chi2, rel=RTOL)
    assert row["df"] == len(x) - 1
    assert row["p_less"] == pytest.approx(lower, rel=1e-8)
    assert row["p_greater"] == pytest.approx(upper, rel=1e-8)
    assert row["p_value"] == pytest.approx(min(1.0, 2 * min(lower, upper)), rel=1e-8)


# ---- one-way ANOVA ---------------------------------------------------------------------


def _groups(frame: pd.DataFrame, y: str = "y", by: str = "k") -> tuple[list, list[np.ndarray]]:
    used = frame[[y, by]].dropna()
    labels = sorted(used[by].unique())
    return labels, [used.loc[used[by] == label, y].to_numpy() for label in labels]


def test_oneway_tables():
    frame = _sample(seed=30, n=150)
    labels, groups = _groups(frame)
    result = oe.oneway(frame, "y", "k", alpha=0.1)
    sizes = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    total, k = sizes.sum(), len(groups)
    f, p = stats.f_oneway(*groups)
    everything = np.concatenate(groups)
    ssb = (sizes * (means - everything.mean()) ** 2).sum()
    ssw = ((sizes - 1) * variances).sum()
    table = result["anova"]
    assert table.loc["between", "ss"] == pytest.approx(ssb, rel=RTOL)
    assert table.loc["within", "ss"] == pytest.approx(ssw, rel=RTOL)
    assert table.loc["total", "ss"] == pytest.approx(((everything - everything.mean()) ** 2).sum(),
                                                     rel=RTOL)
    assert list(table["df"]) == [k - 1, total - k, total - 1]
    assert table.loc["between", "statistic"] == pytest.approx(f, rel=RTOL)
    assert table.loc["between", "p_value"] == pytest.approx(p, rel=1e-8)
    assert result.attrs["rmse"] == pytest.approx(math.sqrt(ssw / (total - k)), rel=RTOL)
    described = result["descriptives"]
    assert list(described.index) == [*labels, "Total"]
    for label, group in zip(labels, groups, strict=True):
        row = described.loc[label]
        half = stats.t.ppf(0.95, len(group) - 1) * group.std(ddof=1) / math.sqrt(len(group))
        assert row["n"] == len(group)
        assert row["mean"] == pytest.approx(group.mean(), rel=RTOL)
        assert row["std_dev"] == pytest.approx(group.std(ddof=1), rel=RTOL)
        assert row["ci_low"] == pytest.approx(group.mean() - half, rel=1e-8)
        assert row["ci_high"] == pytest.approx(group.mean() + half, rel=1e-8)
        assert (row["min"], row["max"]) == (group.min(), group.max())
    assert described.loc["Total", "std_dev"] == pytest.approx(everything.std(ddof=1), rel=RTOL)
    # Homogeneity of variances.
    homogeneity = result["homogeneity"]
    for name, center in (("levene_mean", "mean"), ("levene_median", "median")):
        oracle = stats.levene(*groups, center=center)
        assert homogeneity.loc[name, "statistic"] == pytest.approx(oracle.statistic, rel=1e-8)
        assert homogeneity.loc[name, "p_value"] == pytest.approx(oracle.pvalue, rel=1e-7)
        assert (homogeneity.loc[name, "df1"], homogeneity.loc[name, "df2"]) == (k - 1, total - k)
    z = [np.abs(g - np.median(g)) for g in groups]
    u = np.array([((v - v.mean()) ** 2).sum() for v in z])
    adjusted = u.sum() ** 2 / (u ** 2 / (sizes - 1)).sum()
    row = homogeneity.loc["levene_median_adjusted_df"]
    assert row["df2"] == pytest.approx(adjusted, rel=1e-8)
    assert row["p_value"] == pytest.approx(
        stats.f.sf(homogeneity.loc["levene_median", "statistic"], k - 1, adjusted), rel=1e-7)
    bartlett = stats.bartlett(*groups)
    assert homogeneity.loc["bartlett", "statistic"] == pytest.approx(bartlett.statistic,
                                                                    rel=1e-8)
    assert homogeneity.loc["bartlett", "p_value"] == pytest.approx(bartlett.pvalue, rel=1e-7)
    assert homogeneity.loc["bartlett", "df1"] == k - 1
    # Robust tests of equal means: statsmodels and the textbook formulas.
    welch = anova_oneway(groups, use_var="unequal", welch_correction=True)
    robust = result["robust"]
    assert robust.loc["welch", "statistic"] == pytest.approx(welch.statistic, rel=1e-8)
    assert robust.loc["welch", "df2"] == pytest.approx(welch.df_denom, rel=1e-8)
    assert robust.loc["welch", "p_value"] == pytest.approx(welch.pvalue, rel=1e-7)
    weights = sizes / variances
    centre = (weights * means).sum() / weights.sum()
    lam = ((1 - weights / weights.sum()) ** 2 / (sizes - 1)).sum()
    expected = (weights * (means - centre) ** 2).sum() / (k - 1) \
        / (1 + 2 * (k - 2) * lam / (k * k - 1))
    assert robust.loc["welch", "statistic"] == pytest.approx(expected, rel=RTOL)
    assert robust.loc["welch", "df2"] == pytest.approx((k * k - 1) / (3 * lam), rel=RTOL)
    bf = anova_oneway(groups, use_var="bf")
    assert robust.loc["brown_forsythe", "statistic"] == pytest.approx(bf.statistic, rel=1e-8)
    assert robust.loc["brown_forsythe", "df2"] == pytest.approx(bf.df_denom, rel=1e-8)
    parts = (1 - sizes / total) * variances
    assert robust.loc["brown_forsythe", "statistic"] == pytest.approx(ssb / parts.sum(), rel=RTOL)
    df2 = parts.sum() ** 2 / (parts ** 2 / (sizes - 1)).sum()
    assert robust.loc["brown_forsythe", "df2"] == pytest.approx(df2, rel=RTOL)
    assert robust.loc["brown_forsythe", "p_value"] == pytest.approx(
        stats.f.sf(ssb / parts.sum(), k - 1, df2), rel=1e-7)
    # Effect sizes (SPSS 27+).
    msw, sst = ssw / (total - k), ssb + ssw
    assert result.attrs["eta_squared"] == pytest.approx(ssb / sst, rel=RTOL)
    assert result.attrs["epsilon_squared"] == pytest.approx((ssb - (k - 1) * msw) / sst, rel=RTOL)
    assert result.attrs["omega_squared"] == pytest.approx((ssb - (k - 1) * msw) / (sst + msw),
                                                          rel=RTOL)
    assert result.attrs["n_missing"] == frame[["y", "k"]].isna().any(axis=1).sum()


def test_two_group_robust_tests_reduce_to_welch_t():
    frame = _sample(seed=33, n=80)
    result = oe.oneway(frame, "y", "g")
    welch = oe.ttest(frame, "y", by="g")["test"].loc["unequal_variances"]
    for name in ("welch", "brown_forsythe"):
        row = result["robust"].loc[name]
        assert row["statistic"] == pytest.approx(welch["statistic"] ** 2, rel=1e-9)
        assert row["df2"] == pytest.approx(welch["df"], rel=1e-9)
        assert row["p_value"] == pytest.approx(welch["p_value"], rel=1e-8)


def test_posthoc_comparisons():
    frame = _sample(seed=41, n=170)
    labels, groups = _groups(frame)
    alpha = 0.05
    methods = ["tukey", "bonferroni", "sidak", "scheffe", "lsd", "games_howell", "holm"]
    result = oe.oneway(frame, "y", "k", posthoc=methods, alpha=alpha)
    sizes = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    k, total = len(groups), sizes.sum()
    df = total - k
    mse = ((sizes - 1) * variances).sum() / df
    pairs = list(itertools.combinations(range(k), 2))
    m = len(pairs)
    gap = np.array([means[i] - means[j] for i, j in pairs])
    se = np.array([math.sqrt(mse * (1 / sizes[i] + 1 / sizes[j])) for i, j in pairs])
    t = gap / se
    raw = 2 * stats.t.sf(np.abs(t), df)
    for method in methods:
        table = result[f"posthoc_{method}"]
        assert list(table["group_i"]) == [labels[i] for i, _ in pairs]
        assert list(table["group_j"]) == [labels[j] for _, j in pairs]
        np.testing.assert_allclose(table["mean_difference"], gap, rtol=RTOL)
        if method != "games_howell":
            np.testing.assert_allclose(table["std_error"], se, rtol=RTOL)
            np.testing.assert_allclose(table["statistic"], t, rtol=RTOL)
            assert (table["df"] == df).all()
    expected = {
        "lsd": (raw, stats.t.ppf(1 - alpha / 2, df)),
        "bonferroni": (multipletests(raw, method="bonferroni")[1],
                       stats.t.ppf(1 - alpha / (2 * m), df)),
        "sidak": (multipletests(raw, method="sidak")[1],
                  stats.t.ppf(1 - (1 - (1 - alpha) ** (1 / m)) / 2, df)),
        "scheffe": (stats.f.sf(t ** 2 / (k - 1), k - 1, df),
                    math.sqrt((k - 1) * stats.f.ppf(1 - alpha, k - 1, df))),
        "holm": (multipletests(raw, method="holm")[1], None),
    }
    for method, (p_values, critical) in expected.items():
        table = result[f"posthoc_{method}"]
        np.testing.assert_allclose(table["p_value"], p_values, rtol=1e-7)
        if critical is None:
            assert table["ci_low"].isna().all() and table["ci_high"].isna().all()
        else:
            np.testing.assert_allclose(table["ci_low"], gap - critical * se, rtol=1e-7)
            np.testing.assert_allclose(table["ci_high"], gap + critical * se, rtol=1e-7)
    # Tukey-Kramer against SciPy.
    oracle = stats.tukey_hsd(*groups)
    interval = oracle.confidence_interval(1 - alpha)
    table = result["posthoc_tukey"]
    for row, (i, j) in zip(table.itertuples(), pairs, strict=True):
        assert row.p_value == pytest.approx(oracle.pvalue[i, j], abs=5e-10, rel=1e-6)
        assert row.ci_low == pytest.approx(interval.low[i, j], rel=1e-6)
        assert row.ci_high == pytest.approx(interval.high[i, j], rel=1e-6)
    # Games-Howell: separate variances, Welch df, studentized range with k groups.
    table = result["posthoc_games_howell"]
    for row, (i, j) in zip(table.itertuples(), pairs, strict=True):
        a, b = variances[i] / sizes[i], variances[j] / sizes[j]
        se_pair = math.sqrt(a + b)
        df_pair = (a + b) ** 2 / (a ** 2 / (sizes[i] - 1) + b ** 2 / (sizes[j] - 1))
        q = abs(means[i] - means[j]) / se_pair * math.sqrt(2)
        assert row.std_error == pytest.approx(se_pair, rel=RTOL)
        assert row.df == pytest.approx(df_pair, rel=RTOL)
        assert row.p_value == pytest.approx(stats.studentized_range.sf(q, k, df_pair),
                                            abs=5e-10, rel=1e-6)
        half = stats.studentized_range.ppf(1 - alpha, k, df_pair) / math.sqrt(2) * se_pair
        assert row.ci_low == pytest.approx(means[i] - means[j] - half, rel=1e-6)
        assert row.ci_high == pytest.approx(means[i] - means[j] + half, rel=1e-6)


@pytest.mark.parametrize("control", ["k1", "k3"])
def test_dunnett_against_scipy(control):
    frame = _sample(seed=44, n=120)
    labels, groups = _groups(frame)
    position = labels.index(control)
    result = oe.oneway(frame, "y", "k", posthoc=["dunnett"], control=control)
    table = result["posthoc_dunnett"]
    others = [g for i, g in enumerate(groups) if i != position]
    oracle = stats.dunnett(*others, control=groups[position],
                           random_state=np.random.default_rng(1))
    assert list(table["group_j"]) == [control] * (len(labels) - 1)
    assert list(table["group_i"]) == [label for label in labels if label != control]
    np.testing.assert_allclose(table["statistic"], oracle.statistic, rtol=1e-9)
    # SciPy integrates the multivariate t by randomized quasi-Monte Carlo (about 1e-4).
    np.testing.assert_allclose(table["p_value"], oracle.pvalue, atol=2e-3)
    interval = oracle.confidence_interval(0.95)
    np.testing.assert_allclose(table["ci_low"], interval.low, rtol=5e-3)
    np.testing.assert_allclose(table["ci_high"], interval.high, rtol=5e-3)
    assert table.attrs["control"] == control


def test_dunnett_with_one_treatment_is_the_pooled_t_test():
    frame = _sample(seed=45, n=70)
    table = oe.oneway(frame, "y", "g", posthoc=["dunnett"], control="trt")["posthoc_dunnett"]
    pooled = oe.ttest(frame, "y", by="g")["test"].loc["equal_variances"]
    row = table.iloc[0]
    assert (row["group_i"], row["group_j"]) == ("ctl", "trt")
    assert row["statistic"] == pytest.approx(pooled["statistic"], rel=1e-9)
    assert row["p_value"] == pytest.approx(pooled["p_value"], rel=1e-8)
    assert row["ci_low"] == pytest.approx(pooled["ci_low"], rel=1e-8)
    assert row["ci_high"] == pytest.approx(pooled["ci_high"], rel=1e-8)


def test_oneway_invariances():
    frame = _sample(seed=50, n=110)
    methods = ["tukey", "games_howell", "scheffe"]
    base = oe.oneway(frame, "y", "k", posthoc=methods)
    shuffled = oe.oneway(frame.sample(frac=1.0, random_state=9), "y", "k", posthoc=methods)
    scaled = oe.oneway(frame.assign(y=frame["y"] * 1e-8 - 4.0), "y", "k", posthoc=methods)
    huge = oe.oneway(frame.assign(y=frame["y"] * 1e8 + 1e12), "y", "k", posthoc=methods)
    for other in (shuffled, scaled, huge):
        assert other.attrs["statistic"] == pytest.approx(base.attrs["statistic"], rel=1e-6)
        for name in ("homogeneity", "robust", "effect_sizes"):
            np.testing.assert_allclose(other[name].to_numpy(dtype=float),
                                       base[name].to_numpy(dtype=float), rtol=2e-5,
                                       equal_nan=True)
        for method in methods:
            np.testing.assert_allclose(other[f"posthoc_{method}"]["p_value"],
                                       base[f"posthoc_{method}"]["p_value"], rtol=1e-5,
                                       atol=1e-12)


# ---- correlations -----------------------------------------------------------------------


def _correlated(seed: int = 60, n: int = 90, ties: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    frame = pd.DataFrame({"a": base + rng.normal(size=n), "b": -0.5 * base + rng.normal(size=n),
                          "c": rng.exponential(size=n) + 0.3 * base, "d": rng.normal(size=n)})
    if ties:
        frame = (frame * 2).round()
    for name, count in (("a", 7), ("b", 9), ("d", 4)):
        frame.loc[rng.choice(n, count, replace=False), name] = np.nan
    return frame


_PAIR = {
    "pearson": lambda x, y: tuple(stats.pearsonr(x, y)),
    "spearman": lambda x, y: tuple(stats.spearmanr(x, y)),
    "kendall": lambda x, y: tuple(stats.kendalltau(x, y, variant="b", method="asymptotic")),
}


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("pairwise", [True, False])
def test_correlation_matrices(method, ties, pairwise):
    frame = _correlated(ties=ties)
    names = ["a", "b", "c", "d"]
    result = oe.correlate(frame, names, method=method, pairwise=pairwise, ci=True, alpha=0.1)
    complete = frame[names].dropna()
    assert result.attrs["n_complete"] == len(complete)
    assert result.attrs["missing"] == ("pairwise" if pairwise else "listwise")
    factor, offset = {"pearson": (1.0, 3), "spearman": (1.06, 3), "kendall": (0.437, 4)}[method]
    intervals = result["intervals"].set_index(["var_i", "var_j"])
    for i, j in itertools.combinations(names, 2):
        used = frame[[i, j]].dropna() if pairwise else complete
        r, p = _PAIR[method](used[i].to_numpy(), used[j].to_numpy())
        assert result["coefficients"].loc[i, j] == pytest.approx(r, rel=1e-9, abs=1e-13)
        assert result["coefficients"].loc[j, i] == result["coefficients"].loc[i, j]
        assert result["p_values"].loc[i, j] == pytest.approx(p, rel=1e-7, abs=1e-300)
        assert result["n"].loc[i, j] == len(used)
        half = stats.norm.ppf(0.95) * math.sqrt(factor / (len(used) - offset))
        row = intervals.loc[(i, j)]
        assert row["ci_low"] == pytest.approx(math.tanh(math.atanh(r) - half), rel=1e-8)
        assert row["ci_high"] == pytest.approx(math.tanh(math.atanh(r) + half), rel=1e-8)
    for name in names:
        assert result["coefficients"].loc[name, name] == 1.0
        assert result["n"].loc[name, name] == (frame[name].notna().sum() if pairwise
                                               else len(complete))
    if method == "pearson":
        used = frame[["a", "b"]].dropna()
        interval = stats.pearsonr(used["a"], used["b"]).confidence_interval(0.9)
        if pairwise:
            assert intervals.loc[("a", "b"), "ci_low"] == pytest.approx(interval.low, rel=1e-8)
            assert intervals.loc[("a", "b"), "ci_high"] == pytest.approx(interval.high, rel=1e-8)


def test_kendall_with_heavy_ties_and_small_samples_against_pair_counting():
    rng = np.random.default_rng(71)
    for n in (3, 4, 7, 30):
        x = rng.integers(0, 3, size=n).astype(float)
        y = rng.integers(0, 4, size=n).astype(float)
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        concordant = discordant = tied_x = tied_y = 0
        for i, j in itertools.combinations(range(n), 2):
            sign = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            concordant += sign > 0
            discordant += sign < 0
            tied_x += x[i] == x[j]
            tied_y += y[i] == y[j]
        pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / math.sqrt((pairs - tied_x) * (pairs - tied_y))
        result = oe.correlate({"x": x, "y": y}, ["x", "y"], method="kendall")
        assert result["coefficients"].loc["x", "y"] == pytest.approx(tau, rel=1e-12, abs=1e-14)
        oracle = stats.kendalltau(x, y, variant="b", method="asymptotic")
        assert result["p_values"].loc["x", "y"] == pytest.approx(oracle.pvalue, rel=1e-8)


def test_correlations_are_invariant_to_units_and_row_order():
    frame = _correlated(seed=77)
    names = ["a", "b", "c"]
    for method in ("pearson", "spearman", "kendall"):
        base = oe.correlate(frame, names, method=method)
        changed = frame.sample(frac=1.0, random_state=3).assign(
            a=lambda t: t["a"] * 1e8 + 5e9, b=lambda t: t["b"] * 1e-8, c=lambda t: 3 - t["c"])
        other = oe.correlate(changed, names, method=method)
        sign = np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
        np.testing.assert_allclose(other["coefficients"].to_numpy(dtype=float),
                                   base["coefficients"].to_numpy(dtype=float) * sign,
                                   rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(other["p_values"].to_numpy(dtype=float),
                                   base["p_values"].to_numpy(dtype=float), rtol=1e-5,
                                   equal_nan=True)


def _residual(target: np.ndarray, others: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(target)), others])
    return target - design @ np.linalg.lstsq(design, target, rcond=None)[0]


def test_partial_correlations_are_correlations_of_residuals():
    frame = _correlated(seed=81, n=75)
    used = frame.dropna()
    y = used["a"].to_numpy()
    table = oe.pcorr(frame, "a", ["b", "c"], controls=["d"])
    assert table.attrs["n"] == len(used) and table.attrs["n_missing"] == len(frame) - len(used)
    df = len(used) - 4
    for name in ("b", "c"):
        rest = used[[v for v in ("b", "c", "d") if v != name]].to_numpy()
        x_resid = _residual(used[name].to_numpy(), rest)
        partial = np.corrcoef(_residual(y, rest), x_resid)[0, 1]
        semi = np.corrcoef(y, x_resid)[0, 1]
        t = partial * math.sqrt(df / (1 - partial ** 2))
        row = table.loc[name]
        assert row["partial_corr"] == pytest.approx(partial, rel=1e-9)
        assert row["semipartial_corr"] == pytest.approx(semi, rel=1e-9)
        assert row["partial_corr_sq"] == pytest.approx(partial ** 2, rel=1e-9)
        assert row["semipartial_corr_sq"] == pytest.approx(semi ** 2, rel=1e-9)
        assert row["statistic"] == pytest.approx(t, rel=1e-9)
        assert row["df"] == df
        assert row["p_value"] == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-8)
    full = _residual(y, used[["b", "c", "d"]].to_numpy())
    assert table.attrs["r_squared"] == pytest.approx(
        1 - (full ** 2).sum() / ((y - y.mean()) ** 2).sum(), rel=1e-9)
    # Without controls a single x gives the plain correlation.
    simple = oe.pcorr(used, "a", ["b"])
    r = stats.pearsonr(used["a"], used["b"])
    assert simple.loc["b", "partial_corr"] == pytest.approx(r.statistic, rel=1e-9)
    assert simple.loc["b", "semipartial_corr"] == pytest.approx(r.statistic, rel=1e-9)
    assert simple.loc["b", "p_value"] == pytest.approx(r.pvalue, rel=1e-8)
    # Badly scaled regressors change nothing (the level 10 keeps nine digits of b * 1e-6).
    scaled = oe.pcorr(frame.assign(b=frame["b"] * 1e-6 + 10.0, d=frame["d"] * 1e8), "a",
                      ["b", "c"], controls=["d"])
    np.testing.assert_allclose(scaled.to_numpy(dtype=float), table.to_numpy(dtype=float),
                               rtol=1e-6)


# ---- descriptives -----------------------------------------------------------------------


def test_describe_moments_and_percentiles():
    rng = np.random.default_rng(90)
    frame = pd.DataFrame({"u": rng.gamma(2.0, 2.0, size=83), "v": rng.normal(size=83),
                          "g": rng.choice(["p", "q", "r"], size=83)})
    frame.loc[rng.choice(83, 8, replace=False), "u"] = np.nan
    requested = ["n", "missing", "mean", "sd", "se", "variance", "cv", "sum", "min", "max",
                 "range", "ci", "median", "iqr", "p10", "p37.5", "p90", "skewness",
                 "se_skewness", "kurtosis", "se_kurtosis"]
    for method, numpy_method in (("stata", "averaged_inverted_cdf"), ("haverage", "weibull")):
        for by in (None, "g"):
            table = oe.describe(frame, ["u", "v"], by=by, stats=requested,
                                percentile_method=method, alpha=0.1)
            groups = [(None, frame)] if by is None else list(frame.groupby("g"))
            for label, part in groups:
                for name in ("u", "v"):
                    x = part[name].dropna().to_numpy()
                    n = len(x)
                    row = table.loc[name] if by is None else \
                        table[(table["variable"] == name) & (table["g"] == label)].iloc[0]
                    half = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / math.sqrt(n)
                    quartiles = np.percentile(x, [25, 75], method=numpy_method)
                    expected = {
                        "n": n, "missing": len(part) - n, "mean": x.mean(),
                        "std_dev": x.std(ddof=1), "std_error": x.std(ddof=1) / math.sqrt(n),
                        "variance": x.var(ddof=1), "cv": x.std(ddof=1) / x.mean(),
                        "sum": x.sum(), "min": x.min(), "max": x.max(),
                        "range": x.max() - x.min(), "ci_low": x.mean() - half,
                        "ci_high": x.mean() + half,
                        "p50": np.percentile(x, 50, method=numpy_method),
                        "iqr": quartiles[1] - quartiles[0],
                        "p10": np.percentile(x, 10, method=numpy_method),
                        "p37.5": np.percentile(x, 37.5, method=numpy_method),
                        "p90": np.percentile(x, 90, method=numpy_method),
                        "skewness": stats.skew(x, bias=False),
                        "kurtosis": stats.kurtosis(x, bias=False),
                        # Exact standard errors of G1 and G2 under normality.
                        "se_skewness": math.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))),
                        "se_kurtosis": math.sqrt(24 * n * (n - 1) ** 2
                                                 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))),
                    }
                    for key, value in expected.items():
                        assert row[key] == pytest.approx(value, rel=1e-9), (method, by, key)
    stata = oe.describe(frame, ["u"], stats=["skewness", "kurtosis", "se_skewness"],
                        moments="stata")
    x = frame["u"].dropna().to_numpy()
    assert stata.loc["u", "skewness"] == pytest.approx(stats.skew(x, bias=True), rel=1e-9)
    assert stata.loc["u", "kurtosis"] == pytest.approx(
        stats.kurtosis(x, fisher=False, bias=True), rel=1e-9)
    assert np.isnan(stata.loc["u", "se_skewness"])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 20, 21])
def test_percentile_definitions_for_every_small_sample_size(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.normal(size=n))
    percents = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    names = [f"p{p}" for p in percents]
    stata = oe.describe({"x": x}, ["x"], stats=names, percentile_method="stata")
    spss = oe.describe({"x": x}, ["x"], stats=names, percentile_method="haverage")
    for p, name in zip(percents, names, strict=True):
        # Stata (summarize, detail): P = n p / 100; mean of x_(P), x_(P+1) if P is an integer.
        position = n * p / 100
        if abs(position - round(position)) < 1e-9:
            j = int(round(position))
            value = (x[max(j - 1, 0)] + x[min(j, n - 1)]) / 2
        else:
            value = x[int(math.floor(position))]
        assert stata.loc["x", name] == pytest.approx(value, rel=1e-12, abs=1e-15), (n, p)
        # SPSS HAVERAGE: weighted average at (n + 1) p / 100.
        position = (n + 1) * p / 100
        j, g = int(math.floor(position)), position - math.floor(position)
        if j < 1:
            value = x[0]
        elif j >= n:
            value = x[-1]
        else:
            value = (1 - g) * x[j - 1] + g * x[j]
        assert spss.loc["x", name] == pytest.approx(value, rel=1e-12, abs=1e-15), (n, p)


def test_describe_invariances_and_listwise_option():
    rng = np.random.default_rng(95)
    frame = pd.DataFrame({"u": rng.normal(size=60), "v": rng.normal(size=60)})
    frame.loc[:9, "v"] = np.nan
    base = oe.describe(frame, ["u"], stats=["mean", "sd", "skewness", "kurtosis", "p25"])
    moved = oe.describe(frame.assign(u=frame["u"] * 1e8 + 1e11), ["u"],
                        stats=["mean", "sd", "skewness", "kurtosis", "p25"])
    assert moved.loc["u", "std_dev"] == pytest.approx(base.loc["u", "std_dev"] * 1e8, rel=1e-7)
    assert moved.loc["u", "skewness"] == pytest.approx(base.loc["u", "skewness"], rel=1e-5)
    assert moved.loc["u", "kurtosis"] == pytest.approx(base.loc["u", "kurtosis"], rel=1e-5)
    listwise = oe.describe(frame, ["u", "v"], stats=["n", "mean"], listwise=True)
    assert list(listwise["n"]) == [50, 50]
    assert listwise.loc["u", "mean"] == pytest.approx(frame["u"].iloc[10:].mean(), rel=1e-12)
    variablewise = oe.describe(frame, ["u", "v"], stats=["n", "missing"])
    assert list(variablewise["n"]) == [60, 50] and list(variablewise["missing"]) == [0, 10]
