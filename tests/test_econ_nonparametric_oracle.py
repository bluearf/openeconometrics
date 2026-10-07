"""Nonparametric family: independent oracles (verification pass).

Every reference value in this file is derived without the family's own
formulas: complete enumeration of permutation / sign-flip distributions,
finite-population (permutation) moments of rank statistics, ANOVA-on-ranks
identities, SciPy / statsmodels where the same convention exists, and worked
examples printed in the Stata manuals and in Hanley and McNeil (1982).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats import contingency_tables as sm_tables
from statsmodels.stats import proportion as sm_proportion

import openecon as oe

# The fuel data of the Stata manuals ([R] ranksum, [R] signrank): 12 cars, two treatments.
MPG1 = [20, 23, 21, 25, 18, 17, 18, 24, 20, 24, 23, 19]
MPG2 = [24, 25, 21, 22, 23, 18, 17, 28, 24, 27, 21, 23]
# Hanley and McNeil (1982), Table I / Stata's hanley.dta: ratings 1..5 of 58 normal and
# 51 abnormal cases.
HM_NORMAL = [33, 6, 6, 11, 2]
HM_ABNORMAL = [3, 2, 2, 11, 33]


def ranks_of(values) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=float))


def expand(table: np.ndarray) -> pd.DataFrame:
    """One row per observation of an [R, C] table of integer counts."""
    rows, columns = np.nonzero(np.ones_like(table))
    repeats = table.reshape(-1).astype(int)
    return pd.DataFrame({"r": np.repeat(rows, repeats), "c": np.repeat(columns, repeats)})


# ---- two independent samples -------------------------------------------------------------


def test_ranksum_reproduces_the_stata_manual_example():
    data = {"mpg": MPG1 + MPG2, "treat": [0] * 12 + [1] * 12}
    result = oe.ranksum(data, "mpg", "treat")
    attrs = result.attrs
    # [R] ranksum: rank sums 128 and 172, expected 150, unadjusted variance 300.00,
    # adjustment for ties -4.04, adjusted variance 295.96, z = -1.279, Prob > |z| = 0.2010.
    assert result["ranks"]["rank_sum"].tolist() == [128.0, 172.0]
    assert result["ranks"]["expected"].tolist() == [150.0, 150.0]
    assert attrs["variance_unadjusted"] == pytest.approx(300.0)
    assert attrs["variance"] - attrs["variance_unadjusted"] == pytest.approx(-4.04, abs=0.005)
    assert attrs["variance"] == pytest.approx(295.96, abs=0.005)
    assert attrs["z"] == pytest.approx(-1.279, abs=5e-4)
    assert attrs["p_value"] == pytest.approx(0.2010, abs=5e-5)


@pytest.mark.parametrize("seed, n1, n2, decimals", [(1, 5, 7, None), (2, 9, 4, 0), (3, 30, 45, 1),
                                                    (4, 3, 3, None), (5, 60, 2, 0)])
def test_ranksum_moments_are_the_permutation_moments_of_the_rank_sum(seed, n1, n2, decimals):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n1 + n2) * 2
    if decimals is not None:
        y = np.round(y, decimals)
    g = np.r_[np.zeros(n1), np.ones(n2)]
    order = rng.permutation(n1 + n2)
    y, g = y[order], g[order]
    attrs = oe.ranksum({"y": y, "g": g}, "y", "g").attrs
    ranks = ranks_of(y)
    total = n1 + n2
    # Sampling n1 of N ranks without replacement: mean n1 * mean(r),
    # variance n1 n2 / N * S^2 with S^2 the (N - 1)-divisor variance of the ranks.
    mean = n1 * ranks.mean()
    variance = n1 * n2 / total * ranks.var(ddof=1)
    t1 = ranks[g == 0].sum()
    z = (t1 - mean) / math.sqrt(variance)
    assert attrs["variance"] == pytest.approx(variance, rel=1e-12)
    assert attrs["z"] == pytest.approx(z, rel=1e-10, abs=1e-12)
    assert attrs["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
    # U by direct pair counting.
    first, second = y[g == 0], y[g == 1]
    greater = (first[:, None] > second[None, :]).sum() + 0.5 * (first[:, None]
                                                                == second[None, :]).sum()
    assert attrs["u1"] == pytest.approx(greater) and attrs["u2"] == pytest.approx(
        n1 * n2 - greater)
    assert attrs["u"] == pytest.approx(min(greater, n1 * n2 - greater))
    assert attrs["porder"] == pytest.approx(greater / (n1 * n2))
    assert attrs["effect_r"] == pytest.approx(z / math.sqrt(total))
    # W: rank sum of the smaller sample (the first group when sizes are equal).
    assert attrs["w"] == pytest.approx(t1 if n1 <= n2 else ranks[g == 1].sum())
    cc = np.sign(t1 - mean) * max(abs(t1 - mean) - 0.5, 0.0) / math.sqrt(variance)
    assert attrs["z_continuity"] == pytest.approx(cc, rel=1e-10, abs=1e-12)
    reference = stats.mannwhitneyu(first, second, method="asymptotic", use_continuity=True)
    assert attrs["p_value_continuity"] == pytest.approx(reference.pvalue, rel=1e-9)


@pytest.mark.parametrize("values, n1", [
    ([3.1, 0.2, 5.5, 4.0, 1.7, 2.2, 9.0, 6.1, 7.3], 4),          # no ties
    ([1, 2, 2, 3, 3, 3, 4, 5, 5], 4),                            # ties
    ([2, 2, 2, 2, 7, 7, 1, 1], 3),                               # heavy ties
    ([1, 2, 2, 3, 3, 3, 4, 5, 5], 6),                            # first group larger, ties
    ([3.1, 0.2, 5.5, 4.0, 1.7, 2.2, 9.0, 6.1, 7.3], 7),          # first group larger
])
def test_ranksum_exact_p_is_the_complete_enumeration(values, n1):
    values = np.asarray(values, dtype=float)
    total = values.size
    g = np.r_[np.zeros(n1), np.ones(total - n1)]
    attrs = oe.ranksum({"y": values, "g": g}, "y", "g", exact=True).attrs
    ranks = ranks_of(values)
    centre = n1 * (total + 1) / 2.0
    observed = ranks[:n1].sum()
    sums = np.array([ranks[list(subset)].sum()
                     for subset in itertools.combinations(range(total), n1)])
    assert attrs["p_exact_lower"] == pytest.approx(np.mean(sums <= observed + 1e-9))
    assert attrs["p_exact_upper"] == pytest.approx(np.mean(sums >= observed - 1e-9))
    assert attrs["p_exact"] == pytest.approx(
        np.mean(np.abs(sums - centre) >= abs(observed - centre) - 1e-9))
    if len(set(values)) == total:
        # Without ties the null distribution is symmetric: SPSS's 2 * one-tailed p.
        assert attrs["p_exact"] == pytest.approx(
            min(1.0, 2 * min(attrs["p_exact_lower"], attrs["p_exact_upper"])))
        assert attrs["p_exact"] == pytest.approx(
            stats.mannwhitneyu(values[:n1], values[n1:], method="exact").pvalue)


def test_ranksum_default_exact_rule_and_group_order():
    rng = np.random.default_rng(6)
    # 20 x 20 = 400 pairs: exact; 20 x 21: asymptotic only; ties: asymptotic only.
    a = {"y": rng.normal(size=40), "g": ["b"] * 20 + ["a"] * 20}
    assert oe.ranksum(a, "y", "g").attrs["exact"] is True
    assert oe.ranksum(a, "y", "g").attrs["groups"] == ["a", "b"]
    b = {"y": rng.normal(size=41), "g": [0] * 20 + [1] * 21}
    assert oe.ranksum(b, "y", "g").attrs["exact"] is False
    assert "p_exact" not in oe.ranksum(b, "y", "g").attrs
    tied = {"y": [1.0, 2, 2, 3, 4, 5], "g": [0, 0, 0, 1, 1, 1]}
    assert oe.ranksum(tied, "y", "g").attrs["exact"] is False
    # The first group is the smaller sorted level: its rank sum drives the sign of z.
    first = oe.ranksum({"y": [1.0, 2, 3, 10, 11, 12], "g": [5, 5, 5, 9, 9, 9]}, "y", "g").attrs
    assert first["z"] < 0 and first["u1"] == 0.0 and first["porder"] == 0.0


@pytest.mark.parametrize("seed, n1, n2, alpha", [(7, 6, 8, 0.05), (8, 4, 4, 0.10), (9, 12, 9, 0.01),
                                                 (10, 3, 2, 0.05)])
def test_hodges_lehmann_interval_has_the_exact_coverage_property(seed, n1, n2, alpha):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=n1) + 1.0, rng.normal(size=n2)
    attrs = oe.ranksum({"v": np.r_[x, y], "g": [0] * n1 + [1] * n2}, "v", "g", alpha=alpha).attrs
    differences = np.sort((x[:, None] - y[None, :]).reshape(-1))
    assert attrs["hl_estimate"] == pytest.approx(np.median(differences))
    assert attrs["hl_ci_method"] == "exact"
    # The interval (D_(q), D_(M + 1 - q)) covers with probability 1 - 2 P(U <= q - 1). The
    # reported q is the largest whose coverage still exceeds 1 - alpha (at least 1).
    u_values = np.array([sum(1 for i in subset for j in range(n1 + n2)
                             if j not in subset and j < i)
                         for subset in itertools.combinations(range(n1 + n2), n1)])
    cdf = np.array([np.mean(u_values <= u) for u in range(n1 * n2 + 1)])
    q = int(np.searchsorted(differences, attrs["hl_ci_low"])) + 1
    assert differences[q - 1] == attrs["hl_ci_low"]
    assert differences[n1 * n2 - q] == attrs["hl_ci_high"]
    assert q == 1 or cdf[q - 1] < alpha / 2 + 1e-12
    assert cdf[q] >= alpha / 2 - 1e-12 or q == 1


def test_hodges_lehmann_normal_interval_follows_hollander_wolfe():
    rng = np.random.default_rng(11)
    n1, n2 = 40, 35
    x, y = rng.normal(size=n1) + 0.3, rng.normal(size=n2)
    attrs = oe.ranksum({"v": np.r_[x, y], "g": [0] * n1 + [1] * n2}, "v", "g").attrs
    differences = np.sort((x[:, None] - y[None, :]).reshape(-1))
    assert attrs["hl_ci_method"] == "normal"
    # C_alpha = n1 n2 / 2 - z sqrt(n1 n2 (N + 1) / 12), rounded to the closest integer.
    c = int(round(n1 * n2 / 2 - stats.norm.isf(0.025) * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)))
    assert attrs["hl_ci_low"] == differences[c - 1]
    assert attrs["hl_ci_high"] == differences[n1 * n2 - c]
    assert attrs["hl_ci_low"] < attrs["hl_estimate"] < attrs["hl_ci_high"]


# ---- k independent samples ---------------------------------------------------------------


@pytest.mark.parametrize("seed, sizes, decimals", [(21, (6, 9, 5), None), (22, (4, 4, 4, 4), 0),
                                                   (23, (30, 2, 11), 1), (24, (2, 2), None)])
def test_kwallis_is_the_anova_on_ranks_identity(seed, sizes, decimals):
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(loc=0.4 * j, size=size) for j, size in enumerate(sizes)])
    if decimals is not None:
        y = np.round(y, decimals)
    g = np.repeat(np.arange(len(sizes)), sizes)
    result = oe.kwallis({"y": y, "g": g}, "y", "g", pairwise=True, adjust="holm")
    attrs = result.attrs
    ranks = ranks_of(y)
    total, k = y.size, len(sizes)
    means = np.array([ranks[g == j].mean() for j in range(k)])
    between = sum(size * (mean - ranks.mean()) ** 2 for size, mean in zip(sizes, means))
    # H (tie-corrected) = (N - 1) SS_between / SS_total of the ranks.
    h = (total - 1) * between / ((ranks - ranks.mean()) ** 2).sum()
    assert attrs["statistic"] == pytest.approx(h, rel=1e-10)
    assert attrs["df"] == k - 1
    assert attrs["p_value"] == pytest.approx(stats.chi2.sf(h, k - 1), rel=1e-10)
    assert attrs["statistic"] == pytest.approx(
        stats.kruskal(*[y[g == j] for j in range(k)]).statistic, rel=1e-10)
    unadjusted = 12 / (total * (total + 1)) * sum(
        ranks[g == j].sum() ** 2 / sizes[j] for j in range(k)) - 3 * (total + 1)
    assert attrs["statistic_unadjusted"] == pytest.approx(unadjusted, rel=1e-9, abs=1e-12)
    assert attrs["epsilon_squared"] == pytest.approx(h / (total - 1))
    if total > k:
        assert attrs["eta_squared"] == pytest.approx((h - k + 1) / (total - k))
    # Dunn: the variance of a difference of mean ranks under the permutation distribution.
    s2 = ranks.var(ddof=1)
    pairs = list(itertools.combinations(range(k), 2))
    z = np.array([(means[i] - means[j]) / math.sqrt(s2 * (1 / sizes[i] + 1 / sizes[j]))
                  for i, j in pairs])
    p = 2 * stats.norm.sf(np.abs(z))
    table = result["pairwise"]
    assert table["z"].to_numpy() == pytest.approx(z, rel=1e-10)
    assert table["p_value"].to_numpy() == pytest.approx(p, rel=1e-10)
    order = np.argsort(p)
    holm = np.empty_like(p)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - position) * p[index]))
        holm[index] = running
    assert table["p_adjusted"].to_numpy() == pytest.approx(holm, rel=1e-10)
    bonferroni = oe.kwallis({"y": y, "g": g}, "y", "g", pairwise=True)["pairwise"]
    assert bonferroni["p_adjusted"].to_numpy() == pytest.approx(np.minimum(1.0, len(p) * p))


def test_kwallis_with_two_groups_is_the_squared_ranksum_z():
    rng = np.random.default_rng(25)
    data = {"y": np.round(rng.normal(size=31), 1), "g": rng.integers(0, 2, size=31)}
    kw, rs = oe.kwallis(data, "y", "g").attrs, oe.ranksum(data, "y", "g").attrs
    assert kw["statistic"] == pytest.approx(rs["z"] ** 2, rel=1e-10)
    assert kw["p_value"] == pytest.approx(rs["p_value"], rel=1e-9)


@pytest.mark.parametrize("ties, scipy_ties", [("below", "below"), ("above", "above"),
                                              ("drop", "ignore")])
def test_median_test_against_scipy_and_a_hand_table(ties, scipy_ties):
    rng = np.random.default_rng(26)
    y = np.round(rng.normal(size=47), 0)
    g = rng.integers(0, 3, size=47)
    result = oe.median_test({"y": y, "g": g}, "y", "g", ties=ties)
    reference = stats.median_test(*[y[g == j] for j in range(3)], ties=scipy_ties,
                                  correction=False)
    assert result.attrs["median"] == reference.median
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-10)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert result["counts"]["above"].tolist() == reference.table[0].tolist()
    assert result["counts"]["not_above"].tolist() == reference.table[1].tolist()
    assert result.attrs["n"] == int(reference.table.sum())


def test_median_test_two_groups_continuity_and_fisher():
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    g = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    result = oe.median_test({"y": y, "g": g}, "y", "g")
    # median 6; above: g0 has 1 (7), g1 has 4; not above: g0 4, g1 2.
    table = np.array([[1, 4], [4, 2]])
    assert result["counts"][["above", "not_above"]].to_numpy().T.tolist() == table.tolist()
    assert result.attrs["statistic_continuity"] == pytest.approx(
        stats.chi2_contingency(table, correction=True).statistic)
    assert result.attrs["p_exact"] == pytest.approx(stats.fisher_exact(table).pvalue)


@pytest.mark.parametrize("values, sizes", [
    ([1.5, 0.3, 2.2, 4.0, 3.1, 5.6, 7.7], (2, 2, 3)),            # no ties
    ([1, 1, 2, 2, 2, 3, 4, 4], (3, 2, 3)),                       # ties
    ([0, 1, 0, 1, 1, 0, 1], (2, 3, 2)),                          # binary outcome
])
def test_jonckheere_moments_are_the_exact_permutation_moments(values, sizes):
    values = np.asarray(values, dtype=float)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    attrs = oe.jonckheere({"y": values, "g": groups}, "y", "g").attrs

    def statistic(labels: np.ndarray) -> float:
        total = 0.0
        for i, j in itertools.combinations(range(len(sizes)), 2):
            low, high = values[labels == i], values[labels == j]
            total += (high[None, :] > low[:, None]).sum() + 0.5 * (high[None, :]
                                                                   == low[:, None]).sum()
        return total

    assert attrs["statistic"] == pytest.approx(statistic(groups))
    # Every distinct assignment of the group labels to the observations, equally likely.
    arrangements = set(itertools.permutations(groups.tolist()))
    draws = np.array([statistic(np.array(labels)) for labels in arrangements])
    assert attrs["mean"] == pytest.approx(draws.mean(), rel=1e-12)
    assert attrs["std_dev"] == pytest.approx(draws.std(ddof=0), rel=1e-10)
    z = (statistic(groups) - draws.mean()) / draws.std(ddof=0)
    assert attrs["z"] == pytest.approx(z, rel=1e-10, abs=1e-12)
    assert attrs["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
    assert attrs["p_increasing"] == pytest.approx(stats.norm.sf(z), rel=1e-10)
    assert attrs["p_decreasing"] == pytest.approx(stats.norm.cdf(z), rel=1e-10)


def test_jonckheere_order_option_reverses_the_alternative():
    rng = np.random.default_rng(27)
    y = np.round(rng.normal(size=40) + np.repeat([0.0, 0.5, 1.0, 1.5], 10), 1)
    g = np.repeat(["low", "mid", "high", "top"], 10)
    data = {"y": y, "g": g}
    up = oe.jonckheere(data, "y", "g", order=["low", "mid", "high", "top"]).attrs
    down = oe.jonckheere(data, "y", "g", order=["top", "high", "mid", "low"]).attrs
    assert up["z"] > 2 and down["z"] == pytest.approx(-up["z"])
    assert up["statistic"] + down["statistic"] == pytest.approx(2 * up["mean"])
    codes = np.repeat([0, 1, 2, 3], 10)
    kendall = stats.kendalltau(codes, y, method="asymptotic")
    assert up["p_value"] == pytest.approx(kendall.pvalue, rel=1e-8)


# ---- paired samples ----------------------------------------------------------------------


def test_signrank_and_signtest_reproduce_the_stata_manual_example():
    data = {"mpg1": MPG1, "mpg2": MPG2}
    result = oe.signrank(data, "mpg1", "mpg2", zero_method="adjust")
    attrs = result.attrs
    # [R] signrank: positive 3 obs, sum ranks 13.5; negative 8, 63.5; zero 1; expected 38.5;
    # adjusted variance 160.62; z = -1.973; Prob > |z| = 0.0485.
    assert (attrs["n_positive"], attrs["n_negative"], attrs["n_zero"]) == (3, 8, 1)
    assert (attrs["t_plus"], attrs["t_minus"]) == (13.5, 63.5)
    assert attrs["expected"] == 38.5
    assert attrs["variance"] == pytest.approx(160.62, abs=0.006)
    # unadjusted 162.50, ties -1.62, zeros -0.25
    assert attrs["variance"] == pytest.approx(162.50 - 1.625 - 0.25)
    assert attrs["z"] == pytest.approx(-1.973, abs=5e-4)
    assert attrs["p_value"] == pytest.approx(0.0485, abs=5e-5)
    sign = oe.signtest(data, "mpg1", "mpg2").attrs
    # [R] signtest: Pr(#positive >= 3) = 0.9673, Pr(#negative >= 8) = 0.1133, two-sided 0.2266.
    assert sign["p_positive"] == pytest.approx(0.9673, abs=5e-5)
    assert sign["p_negative"] == pytest.approx(0.1133, abs=5e-5)
    assert sign["p_value"] == pytest.approx(0.2266, abs=5e-5)


@pytest.mark.parametrize("d, zero_method", [
    ([1.2, -0.4, 2.2, 3.1, -0.9, 0.7, 5.0, -2.6], "drop"),
    ([1, -1, 2, 2, -3, 0, 4, 0, 5, -2], "drop"),
    ([1, -1, 2, 2, -3, 0, 4, 0, 5, -2], "adjust"),
    ([0, 0, 0, 3, -1, 1, 1], "adjust"),
])
def test_signrank_is_the_complete_sign_flip_distribution(d, zero_method):
    d = np.asarray(d, dtype=float)
    attrs = oe.signrank({"d": d}, "d", zero_method=zero_method, exact=True).attrs
    if zero_method == "drop":
        d = d[d != 0]
    ranks = ranks_of(np.abs(d))
    used = ranks[d != 0]
    positive = d[d != 0] > 0
    t_plus = used[positive].sum()
    sums = np.array([used[np.array(signs, dtype=bool)].sum()
                     for signs in itertools.product([0, 1], repeat=used.size)])
    assert attrs["t_plus"] == pytest.approx(t_plus)
    assert attrs["expected"] == pytest.approx(sums.mean())
    assert attrs["variance"] == pytest.approx(sums.var(ddof=0), rel=1e-12)
    z = (t_plus - sums.mean()) / sums.std(ddof=0)
    assert attrs["z"] == pytest.approx(z, rel=1e-10, abs=1e-12)
    assert attrs["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
    assert attrs["p_exact_lower"] == pytest.approx(np.mean(sums <= t_plus + 1e-9))
    assert attrs["p_exact_upper"] == pytest.approx(np.mean(sums >= t_plus - 1e-9))
    assert attrs["p_exact"] == pytest.approx(
        np.mean(np.abs(sums - sums.mean()) >= abs(t_plus - sums.mean()) - 1e-9))
    assert attrs["statistic"] == pytest.approx(min(t_plus, used.sum() - t_plus))


def test_signrank_matches_scipy_conventions_and_the_spss_variance():
    rng = np.random.default_rng(31)
    a, b = np.round(rng.normal(size=60), 1), np.round(rng.normal(size=60) + 0.3, 1)
    data = {"a": a, "b": b}
    for zero_method, scipy_name in (("drop", "wilcox"), ("adjust", "pratt")):
        attrs = oe.signrank(data, "a", "b", zero_method=zero_method).attrs
        reference = stats.wilcoxon(a, b, zero_method=scipy_name, correction=False,
                                   method="asymptotic")
        assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-9)
        assert attrs["statistic"] == pytest.approx(reference.statistic)
    # SPSS: n(n+1)/4 and n(n+1)(2n+1)/24 - sum(t^3 - t)/48 over the nonzero differences.
    d = (a - b)[a != b]
    n = d.size
    _, counts = np.unique(np.abs(d), return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24 - ((counts ** 3 - counts).sum()) / 48
    attrs = oe.signrank(data, "a", "b").attrs
    assert attrs["expected"] == pytest.approx(n * (n + 1) / 4)
    assert attrs["variance"] == pytest.approx(variance, rel=1e-12)
    # One-sample form with a hypothesized median equals the paired form on y - mu.
    one = oe.signrank({"a": a}, "a", mu=0.25).attrs
    two = oe.signrank({"a": a, "m": np.full(60, 0.25)}, "a", "m").attrs
    assert one["z"] == pytest.approx(two["z"]) and one["n_zero"] == two["n_zero"]


def test_signrank_default_exact_rule():
    rng = np.random.default_rng(32)
    d25, d26 = rng.normal(size=25), rng.normal(size=26)
    assert oe.signrank({"d": d25}, "d").attrs["exact"] is True
    assert oe.signrank({"d": d26}, "d").attrs["exact"] is False
    assert oe.signrank({"d": [1.0, -1.0, 2.0, 3.0]}, "d").attrs["exact"] is False   # tie
    reference = stats.wilcoxon(d25, method="exact")
    assert oe.signrank({"d": d25}, "d").attrs["p_exact"] == pytest.approx(reference.pvalue,
                                                                         rel=1e-10)


@pytest.mark.parametrize("n_positive, n_negative, n_zero", [(3, 8, 1), (0, 5, 0), (14, 14, 3),
                                                             (40, 22, 0), (1, 0, 4)])
def test_signtest_is_binomial(n_positive, n_negative, n_zero):
    d = np.r_[np.ones(n_positive), -np.ones(n_negative), np.zeros(n_zero)]
    attrs = oe.signtest({"d": d}, "d").attrs
    n = n_positive + n_negative
    assert attrs["n"] == n and attrs["n_zero"] == n_zero
    assert attrs["p_positive"] == pytest.approx(stats.binom.sf(n_positive - 1, n, 0.5))
    assert attrs["p_negative"] == pytest.approx(stats.binom.sf(n_negative - 1, n, 0.5))
    assert attrs["p_value"] == pytest.approx(
        min(1.0, 2 * stats.binom.sf(max(n_positive, n_negative) - 1, n, 0.5)))
    assert attrs["p_value"] == pytest.approx(stats.binomtest(n_positive, n, 0.5).pvalue)
    # SPSS (n > 25): Z = (max(n+, n-) - n/2 - 1/2) / (sqrt(n) / 2), signed by n+ - n-.
    z = max(0.0, max(n_positive, n_negative) - 0.5 * n - 0.5) / (0.5 * math.sqrt(n))
    assert abs(attrs["z"]) == pytest.approx(z)
    assert attrs["p_value_normal"] == pytest.approx(2 * stats.norm.sf(z))


def test_mcnemar_and_symmetry_against_statsmodels_and_hand_algebra():
    rng = np.random.default_rng(33)
    a = rng.integers(0, 2, size=90)
    b = np.where(rng.random(90) < 0.3, 1 - a, a)
    b[:7] = 1
    result = oe.mcnemar({"a": a, "b": b}, "a", "b")
    table = pd.crosstab(a, b).to_numpy()
    n12, n21 = table[0, 1], table[1, 0]
    attrs = result.attrs
    assert (attrs["n12"], attrs["n21"]) == (n12, n21)
    assert attrs["statistic"] == pytest.approx((n12 - n21) ** 2 / (n12 + n21))
    assert attrs["statistic_continuity"] == pytest.approx(
        (abs(n12 - n21) - 1) ** 2 / (n12 + n21))
    exact = sm_tables.mcnemar(table, exact=True)
    assert attrs["p_exact"] == pytest.approx(exact.pvalue, rel=1e-10)
    assert attrs["p_value"] == pytest.approx(
        sm_tables.mcnemar(table, exact=False, correction=False).pvalue, rel=1e-10)
    assert attrs["p_value_continuity"] == pytest.approx(
        sm_tables.mcnemar(table, exact=False, correction=True).pvalue, rel=1e-10)

    k = 4
    r1 = rng.integers(0, k, size=200)
    r2 = np.where(rng.random(200) < 0.5, r1, rng.integers(0, k, size=200))
    result = oe.symmetry({"a": r1, "b": r2}, "a", "b")
    counts = pd.crosstab(r1, r2).to_numpy().astype(float)
    square = sm_tables.SquareTable(counts, shift_zeros=False)
    bowker = square.symmetry(method="bowker")
    assert result.attrs["statistic"] == pytest.approx(bowker.statistic, rel=1e-10)
    assert result.attrs["df"] == bowker.df
    assert result.attrs["p_value"] == pytest.approx(bowker.pvalue, rel=1e-10)
    # Stuart-Maxwell by dropping the last category.
    d = (counts.sum(1) - counts.sum(0))[:-1]
    v = -(counts + counts.T)
    np.fill_diagonal(v, counts.sum(1) + counts.sum(0) - 2 * np.diag(counts))
    statistic = d @ np.linalg.solve(v[:-1, :-1], d)
    assert result.attrs["mh_statistic"] == pytest.approx(statistic, rel=1e-9)
    assert result.attrs["mh_df"] == k - 1
    assert result.attrs["mh_statistic"] == pytest.approx(
        square.homogeneity(method="stuart_maxwell").statistic, rel=1e-9)
    # SPSS's ordinal marginal-homogeneity z from the raw pairs.
    gap = (r1 - r2).astype(float)
    z = gap.sum() / math.sqrt((gap ** 2).sum())
    assert result.attrs["mh_z"] == pytest.approx(z, rel=1e-12)
    assert result.attrs["mh_z_p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)


# ---- k related samples -------------------------------------------------------------------


@pytest.mark.parametrize("seed, n, k, decimals", [(41, 12, 4, None), (42, 9, 3, 0), (43, 30, 6, 0),
                                                  (44, 2, 5, None)])
def test_friedman_is_the_two_way_rank_anova(seed, n, k, decimals):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k)) + 0.3 * np.arange(k)
    if decimals is not None:
        x = np.round(x, decimals)
    names = [f"c{j}" for j in range(k)]
    result = oe.friedman(dict(zip(names, x.T)), names, pairwise=True, adjust="none")
    ranks = np.apply_along_axis(stats.rankdata, 1, x)
    column_sums = ranks.sum(0)
    # Q = (k - 1) sum_j (R_j - n (k + 1) / 2)^2 / (sum r_ij^2 - n k (k + 1)^2 / 4):
    # the tie-corrected statistic from the within-row rank variances.
    q = (k - 1) * ((column_sums - n * (k + 1) / 2) ** 2).sum() \
        / ((ranks ** 2).sum() - n * k * (k + 1) ** 2 / 4)
    assert result.attrs["statistic"] == pytest.approx(q, rel=1e-10)
    assert result.attrs["p_value"] == pytest.approx(stats.chi2.sf(q, k - 1), rel=1e-10)
    assert result.attrs["kendall_w"] == pytest.approx(q / (n * (k - 1)), rel=1e-10)
    if k >= 3:
        assert result.attrs["statistic"] == pytest.approx(
            stats.friedmanchisquare(*x.T).statistic, rel=1e-10)
    assert result["ranks"]["rank_sum"].to_numpy() == pytest.approx(column_sums)
    pairs = list(itertools.combinations(range(k), 2))
    z = [(column_sums[i] - column_sums[j]) / n / math.sqrt(k * (k + 1) / (6 * n))
         for i, j in pairs]
    assert result["pairwise"]["z"].to_numpy() == pytest.approx(np.array(z), rel=1e-10)
    assert result["pairwise"]["p_adjusted"].to_numpy() == pytest.approx(
        2 * stats.norm.sf(np.abs(z)), rel=1e-10)


def test_kendall_w_without_ties_is_the_variance_ratio_of_rank_sums():
    rng = np.random.default_rng(45)
    n, k = 7, 5
    x = rng.normal(size=(n, k))
    names = list("abcde")
    attrs = oe.friedman(dict(zip(names, x.T)), names).attrs
    sums = np.apply_along_axis(stats.rankdata, 1, x).sum(0)
    w = 12 * ((sums - sums.mean()) ** 2).sum() / (n ** 2 * (k ** 3 - k))
    assert attrs["kendall_w"] == pytest.approx(w, rel=1e-12)


def test_cochran_q_against_the_definition_and_mcnemar():
    rng = np.random.default_rng(46)
    x = (rng.random((40, 4)) < np.array([0.3, 0.5, 0.6, 0.4])).astype(int)
    names = ["t1", "t2", "t3", "t4"]
    result = oe.cochran_q(dict(zip(names, x.T)), names)
    k = 4
    columns, rows = x.sum(0), x.sum(1)
    q = k * (k - 1) * ((columns - columns.mean()) ** 2).sum() / (k * rows.sum()
                                                                  - (rows ** 2).sum())
    assert result.attrs["statistic"] == pytest.approx(q, rel=1e-12)
    assert result.attrs["p_value"] == pytest.approx(stats.chi2.sf(q, 3), rel=1e-10)
    reference = sm_tables.cochrans_q(x)
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-10)
    two = oe.cochran_q({"a": x[:, 0], "b": x[:, 1]}, ["a", "b"]).attrs
    assert two["statistic"] == pytest.approx(
        oe.mcnemar({"a": x[:, 0], "b": x[:, 1]}, "a", "b").attrs["statistic"])
    text = np.where(x == 1, "yes", "no")
    labelled = oe.cochran_q(dict(zip(names, text.T)), names, positive="yes").attrs
    assert labelled["statistic"] == pytest.approx(q, rel=1e-12)
    flipped = oe.cochran_q(dict(zip(names, text.T)), names, positive="no").attrs
    assert flipped["statistic"] == pytest.approx(q, rel=1e-12)       # Q is symmetric in 0/1


# ---- one-sample distribution tests -------------------------------------------------------


@pytest.mark.parametrize("seed, n", [(51, 1), (52, 6), (53, 40), (54, 400)])
def test_ksmirnov_one_sample_against_scipy_and_stata_one_sided_formulas(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.2, scale=1.3, size=n)
    result = oe.ksmirnov({"x": x}, "x", params=[0.0, 1.0])
    attrs = result.attrs
    ordered = np.sort(x)
    cdf = stats.norm.cdf(ordered)
    i = np.arange(1, n + 1)
    d_plus, d_minus = np.max(i / n - cdf), np.max(cdf - (i - 1) / n)
    assert attrs["d_plus"] == pytest.approx(d_plus, rel=1e-12)
    assert attrs["d_minus"] == pytest.approx(-d_minus, rel=1e-12)
    assert attrs["statistic"] == pytest.approx(max(d_plus, d_minus), rel=1e-12)
    reference = stats.ks_1samp(x, stats.norm.cdf, method="exact")
    assert attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert attrs["p_exact"] == pytest.approx(reference.pvalue, rel=1e-8, abs=1e-14)
    assert attrs["p_value"] == attrs["p_exact"]
    asymptotic = stats.kstwobign.sf(math.sqrt(n) * max(d_plus, d_minus))
    assert attrs["p_asymptotic"] == pytest.approx(asymptotic, rel=1e-10, abs=1e-300)
    # Stata: one-sided p = exp(-2 n D^2).
    assert result.loc["positive", "p_value"] == pytest.approx(math.exp(-2 * n * d_plus ** 2))
    assert result.loc["negative", "p_value"] == pytest.approx(math.exp(-2 * n * d_minus ** 2))
    assert attrs["z"] == pytest.approx(math.sqrt(n) * max(d_plus, d_minus))


def test_ksmirnov_estimated_parameters_follow_spss_estimates():
    rng = np.random.default_rng(55)
    x = rng.gamma(2.0, 1.5, size=80)
    normal = oe.ksmirnov({"x": x}, "x").attrs
    assert normal["params"] == pytest.approx([x.mean(), x.std(ddof=1)])
    assert normal["estimated"] is True and normal["exact"] is False
    d = stats.ks_1samp(x, stats.norm(x.mean(), x.std(ddof=1)).cdf).statistic
    assert normal["statistic"] == pytest.approx(d, rel=1e-12)
    # Dallal and Wilkinson (1986), valid for p < 0.1.
    dw = math.exp(-7.01256 * d * d * (80 + 2.78019) + 2.99587 * d * math.sqrt(80 + 2.78019)
                  - 0.122119 + 0.974598 / math.sqrt(80) + 1.67997 / 80)
    if dw <= 0.1:
        assert normal["p_lilliefors"] == pytest.approx(dw, rel=1e-12)
    assert normal["p_value"] == normal["p_lilliefors"]
    uniform = oe.ksmirnov({"x": x}, "x", distribution="uniform").attrs
    assert uniform["params"] == pytest.approx([x.min(), x.max()])
    assert uniform["statistic"] == pytest.approx(
        stats.ks_1samp(x, stats.uniform(x.min(), x.max() - x.min()).cdf).statistic, rel=1e-12)
    exponential = oe.ksmirnov({"x": x}, "x", distribution="exponential").attrs
    assert exponential["params"] == pytest.approx([x.mean()])
    assert exponential["statistic"] == pytest.approx(
        stats.ks_1samp(x, stats.expon(scale=x.mean()).cdf).statistic, rel=1e-12)
    assert exponential["p_value"] == pytest.approx(
        stats.kstwobign.sf(math.sqrt(80) * exponential["statistic"]), rel=1e-10)


def test_lilliefors_p_value_against_simulation():
    # Monte Carlo null distribution of D with estimated mean and sd (n = 20): the
    # Dallal-Wilkinson p must match the simulated tail in its range of validity.
    rng = np.random.default_rng(56)
    n, reps = 20, 40_000
    draws = np.sort(rng.normal(size=(reps, n)), axis=1)
    z = (draws - draws.mean(1, keepdims=True)) / draws.std(1, ddof=1, keepdims=True)
    cdf = stats.norm.cdf(z)
    i = np.arange(1, n + 1)
    d = np.maximum((i / n - cdf).max(1), (cdf - (i - 1) / n).max(1))
    from openecon.econometrics.nonparametric.distribution import lilliefors_p_value

    for quantile in (0.90, 0.95, 0.99):
        value = np.quantile(d, quantile)
        assert lilliefors_p_value(float(value), n) == pytest.approx(1 - quantile, rel=0.12)
    # p > 0.1: the Stephens branch stays within a few points of the simulated tail.
    for quantile in (0.3, 0.6, 0.8):
        value = np.quantile(d, quantile)
        assert lilliefors_p_value(float(value), n) == pytest.approx(1 - quantile, abs=0.04)


def test_ksmirnov_poisson_statistic_by_brute_force():
    rng = np.random.default_rng(57)
    x = rng.poisson(3.2, size=70)
    attrs = oe.ksmirnov({"x": x}, "x", distribution="poisson").attrs
    mean = x.mean()
    assert attrs["params"] == pytest.approx([mean])
    values = np.arange(0, x.max() + 2)
    ecdf = np.array([(x <= v).mean() for v in values])
    ecdf_before = np.array([(x < v).mean() for v in values])
    theoretical = stats.poisson.cdf(values, mean)
    theoretical_before = stats.poisson.cdf(values - 1, mean)
    observed = np.isin(values, x)
    d_plus = np.max((ecdf - theoretical)[observed])
    d_minus = np.max((theoretical_before - ecdf_before)[observed])
    assert attrs["d_plus"] == pytest.approx(max(d_plus, 0.0), abs=1e-12)
    assert attrs["d_minus"] == pytest.approx(-max(d_minus, 0.0), abs=1e-12)
    assert attrs["p_value"] == pytest.approx(
        stats.kstwobign.sf(math.sqrt(70) * attrs["statistic"]), rel=1e-10)


@pytest.mark.parametrize("seed, n1, n2, decimals", [(58, 7, 9, None), (59, 30, 30, None),
                                                    (60, 25, 40, 0), (61, 1, 5, None)])
def test_ksmirnov_two_sample_against_scipy(seed, n1, n2, decimals):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n1), rng.normal(size=n2) + 0.5
    if decimals is not None:
        a, b = np.round(a, decimals), np.round(b, decimals)
    result = oe.ksmirnov({"v": np.r_[a, b], "g": [0] * n1 + [1] * n2}, "v", by="g")
    attrs = result.attrs
    grid = np.unique(np.r_[a, b])
    f1 = np.array([(a <= v).mean() for v in grid])
    f2 = np.array([(b <= v).mean() for v in grid])
    d_plus, d_minus = max((f1 - f2).max(), 0.0), max((f2 - f1).max(), 0.0)
    d = max(d_plus, d_minus)
    assert attrs["d_plus"] == pytest.approx(d_plus, abs=1e-12)
    assert attrs["d_minus"] == pytest.approx(-d_minus, abs=1e-12)
    m = n1 * n2 / (n1 + n2)
    assert attrs["p_asymptotic"] == pytest.approx(stats.kstwobign.sf(math.sqrt(m) * d),
                                                  rel=1e-10)
    assert result.loc["positive", "p_value"] == pytest.approx(math.exp(-2 * m * d_plus ** 2))
    assert result.loc["negative", "p_value"] == pytest.approx(math.exp(-2 * m * d_minus ** 2))
    if decimals is None:
        reference = stats.ks_2samp(a, b, method="exact")
        assert attrs["exact"] is True
        assert attrs["p_exact"] == pytest.approx(reference.pvalue, rel=1e-9)
        assert attrs["p_value"] == attrs["p_exact"]
    else:
        assert attrs["exact"] is False and attrs["p_value"] == attrs["p_asymptotic"]


def test_smirnov_exact_distribution_by_complete_enumeration():
    n1, n2 = 4, 5
    values = np.arange(n1 + n2, dtype=float)
    statistics = []
    for subset in itertools.combinations(range(n1 + n2), n1):
        mask = np.zeros(n1 + n2, dtype=bool)
        mask[list(subset)] = True
        f1 = np.cumsum(mask) / n1
        f2 = np.cumsum(~mask) / n2
        statistics.append(np.abs(f1 - f2).max())
    statistics = np.array(statistics)
    for subset in ((0, 1, 2, 3), (0, 2, 5, 8), (1, 3, 4, 6)):
        g = np.ones(n1 + n2, dtype=int)
        g[list(subset)] = 0
        attrs = oe.ksmirnov({"v": values, "g": g}, "v", by="g").attrs
        assert attrs["p_exact"] == pytest.approx(
            np.mean(statistics >= attrs["statistic"] - 1e-12))


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 11, 12, 13, 25, 100, 999, 2000, 5000])
def test_swilk_against_scipy_shapiro(n):
    rng = np.random.default_rng(100 + n)
    for x in (rng.normal(size=n), rng.exponential(size=n), np.round(rng.normal(size=n), 1)):
        if np.ptp(x) == 0:
            continue
        attrs = oe.swilk({"x": x}, "x").attrs
        reference = stats.shapiro(x)
        assert attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-9)
        assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=2e-6, abs=1e-12)
        if attrs["z"] is not None and 1e-12 < attrs["p_value"] < 1:
            assert stats.norm.sf(attrs["z"]) == pytest.approx(attrs["p_value"], rel=1e-9)


def test_sfrancia_definition_and_royston_normalization():
    rng = np.random.default_rng(71)
    for n in (5, 12, 60, 800):
        x = rng.lognormal(size=n)
        attrs = oe.sfrancia({"x": x}, "x").attrs
        scores = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        w = np.corrcoef(np.sort(x), scores)[0, 1] ** 2
        assert attrs["statistic"] == pytest.approx(w, rel=1e-10)
        u = math.log(n)
        v = math.log(u)
        z = (math.log(1 - w) - (-1.2725 + 1.0521 * (v - u))) / (1.0308 - 0.26758 * (v + 2 / u))
        assert attrs["z"] == pytest.approx(z, rel=1e-8)
        assert attrs["p_value"] == pytest.approx(stats.norm.sf(z), rel=1e-8)
    # Under normality the p-values are roughly uniform (the normalization is calibrated).
    p = [oe.sfrancia({"x": rng.normal(size=50)}, "x").attrs["p_value"] for _ in range(300)]
    assert 0.02 < np.mean(np.array(p) < 0.05) < 0.09
    assert 0.4 < np.mean(p) < 0.6


def test_swilk_is_calibrated_under_normality():
    rng = np.random.default_rng(72)
    for n in (8, 40):
        p = np.array([oe.swilk({"x": rng.normal(size=n)}, "x").attrs["p_value"]
                      for _ in range(400)])
        assert 0.02 < np.mean(p < 0.05) < 0.09 and 0.42 < p.mean() < 0.58


@pytest.mark.parametrize("seed, n", [(73, 8), (74, 20), (75, 75), (76, 1000)])
def test_sktest_against_scipy(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.gamma(3.0, size=n)
    result = oe.sktest({"x": x}, "x")
    attrs = result.attrs
    assert attrs["skewness"] == pytest.approx(stats.skew(x), rel=1e-10)
    assert attrs["kurtosis"] == pytest.approx(stats.kurtosis(x, fisher=False), rel=1e-10)
    skew = stats.skewtest(x)
    assert attrs["z_skewness"] == pytest.approx(skew.statistic, rel=1e-9)
    assert attrs["p_skewness"] == pytest.approx(skew.pvalue, rel=1e-8)
    if n >= 20:
        kurt = stats.kurtosistest(x)
        assert attrs["z_kurtosis"] == pytest.approx(kurt.statistic, rel=1e-9)
        assert attrs["p_kurtosis"] == pytest.approx(kurt.pvalue, rel=1e-8)
        assert attrs["chi2_unadjusted"] == pytest.approx(stats.normaltest(x).statistic, rel=1e-9)
    jb = stats.jarque_bera(x)
    assert attrs["jarque_bera"] == pytest.approx(jb.statistic, rel=1e-10)
    assert attrs["p_jarque_bera"] == pytest.approx(jb.pvalue, rel=1e-9)
    assert attrs["p_unadjusted"] == pytest.approx(
        stats.chi2.sf(attrs["z_skewness"] ** 2 + attrs["z_kurtosis"] ** 2, 2), rel=1e-9)
    assert attrs["p_value"] == pytest.approx(stats.chi2.sf(attrs["statistic"], 2), rel=1e-9)


def test_sktest_royston_adjustment_matches_the_stata_manual_and_is_calibrated():
    from openecon.econometrics.nonparametric.normality import royston_adjusted_chi2

    # [R] sktest (auto.dta, n = 74): mpg 13.13 -> 10.95 (p 0.0042); trunk 4.05 -> 4.19 (0.1228).
    assert royston_adjusted_chi2(13.13, 74) == pytest.approx(10.95, abs=0.006)
    assert royston_adjusted_chi2(4.05, 74) == pytest.approx(4.19, abs=0.006)
    # The adjustment exists because K2 over-rejects in small samples: under normality the
    # adjusted test must be closer to its nominal 5% than the unadjusted one.
    rng = np.random.default_rng(77)
    adjusted, raw = [], []
    for _ in range(1500):
        attrs = oe.sktest({"x": rng.normal(size=20)}, "x").attrs
        adjusted.append(attrs["p_value"] < 0.05)
        raw.append(attrs["p_unadjusted"] < 0.05)
    assert abs(np.mean(adjusted) - 0.05) <= abs(np.mean(raw) - 0.05) + 0.005
    assert 0.03 < np.mean(adjusted) < 0.075


@pytest.mark.parametrize("n1, n2", [(3, 4), (5, 5), (2, 7), (1, 6)])
def test_runs_exact_distribution_by_complete_enumeration(n1, n2):
    counts: dict[int, int] = {}
    for subset in itertools.combinations(range(n1 + n2), n1):
        mask = np.zeros(n1 + n2, dtype=bool)
        mask[list(subset)] = True
        runs = 1 + int((mask[1:] != mask[:-1]).sum())
        counts[runs] = counts.get(runs, 0) + 1
    total = sum(counts.values())
    expected = 2 * n1 * n2 / (n1 + n2) + 1
    # A series with the first n1 values above the cut and one swap.
    series = np.r_[np.full(n1, 5.0), np.full(n2, 1.0)]
    if n1 > 1:
        series[[0, -1]] = series[[-1, 0]]
    attrs = oe.runtest({"y": series}, "y", threshold=3.0).attrs
    r = attrs["runs"]
    assert r == 1 + int((np.diff((series >= 3).astype(int)) != 0).sum())
    assert attrs["expected"] == pytest.approx(expected)
    mean = sum(k * v for k, v in counts.items()) / total
    variance = sum((k - mean) ** 2 * v for k, v in counts.items()) / total
    assert attrs["expected"] == pytest.approx(mean)
    assert attrs["variance"] == pytest.approx(variance, rel=1e-12)
    assert attrs["p_exact_lower"] == pytest.approx(
        sum(v for k, v in counts.items() if k <= r) / total)
    assert attrs["p_exact_upper"] == pytest.approx(
        sum(v for k, v in counts.items() if k >= r) / total)
    assert attrs["p_exact"] == pytest.approx(
        sum(v for k, v in counts.items() if abs(k - expected) >= abs(r - expected) - 1e-9)
        / total)


def test_runtest_conventions_of_stata_and_spss():
    rng = np.random.default_rng(78)
    y = np.round(rng.normal(size=60), 1)
    median = np.median(y)
    for ties, above in (("above", y >= median), ("below", y > median)):
        attrs = oe.runtest({"y": y}, "y", ties=ties).attrs
        n1, n0 = int(above.sum()), int((~above).sum())
        runs = 1 + int((above[1:] != above[:-1]).sum())
        mu = 2 * n1 * n0 / 60 + 1
        sigma = math.sqrt(2 * n1 * n0 * (2 * n1 * n0 - 60) / (60 ** 2 * 59))
        assert (attrs["n_above"], attrs["n_below"], attrs["runs"]) == (n1, n0, runs)
        assert attrs["continuity"] is False                  # N >= 50: no correction (SPSS)
        assert attrs["z"] == pytest.approx((runs - mu) / sigma, rel=1e-12)
        assert attrs["p_value"] == pytest.approx(2 * stats.norm.sf(abs(attrs["z"])), rel=1e-10)
        assert attrs["exact"] is False
    short = y[:30]
    attrs = oe.runtest({"y": short}, "y", threshold="mean").attrs
    above = short >= short.mean()
    n1, n0 = int(above.sum()), int((~above).sum())
    runs = 1 + int((above[1:] != above[:-1]).sum())
    mu = 2 * n1 * n0 / 30 + 1
    sigma = math.sqrt(2 * n1 * n0 * (2 * n1 * n0 - 30) / (30 ** 2 * 29))
    gap = runs - mu
    corrected = 0.0 if abs(gap) < 0.5 else (gap + 0.5 if gap < 0 else gap - 0.5)
    assert attrs["continuity"] is True and attrs["exact"] is True       # N < 50 (SPSS)
    assert attrs["z"] == pytest.approx(corrected / sigma, abs=1e-12)
    assert attrs["threshold"] == pytest.approx(short.mean())
    dropped = oe.runtest({"y": y}, "y", ties="drop").attrs
    kept = y[y != median]
    assert dropped["n"] == kept.size
    assert dropped["runs"] == 1 + int(((kept > median)[1:] != (kept > median)[:-1]).sum())


@pytest.mark.parametrize("k, n, p", [(0, 9, 0.5), (9, 9, 0.2), (3, 20, 0.3), (17, 40, 0.5),
                                     (5, 12, 0.75), (1, 1, 0.4), (250, 1000, 0.27)])
def test_bitest_against_scipy_binomtest(k, n, p):
    y = np.r_[np.ones(k), np.zeros(n - k)]
    attrs = oe.bitest({"y": y}, "y", p=p, alpha=0.1).attrs       # 0/1 data: the event is 1
    reference = stats.binomtest(k, n, p)
    assert attrs["successes"] == k and attrs["n"] == n
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-9)
    assert attrs["p_lower"] == pytest.approx(stats.binom.cdf(k, n, p), rel=1e-9)
    assert attrs["p_upper"] == pytest.approx(stats.binom.sf(k - 1, n, p), rel=1e-9)
    interval = reference.proportion_ci(confidence_level=0.9, method="exact")
    assert attrs["ci_low"] == pytest.approx(interval.low, rel=1e-8, abs=1e-12)
    assert attrs["ci_high"] == pytest.approx(interval.high, rel=1e-8, abs=1e-12)


def test_prtest_against_statsmodels_and_hand_formulas():
    rng = np.random.default_rng(79)
    y = (rng.random(140) < 0.42).astype(int)
    g = rng.integers(0, 2, size=140)
    one = oe.prtest({"y": y}, "y", p=0.35)
    n, phat = 140, y.mean()
    z = (phat - 0.35) / math.sqrt(0.35 * 0.65 / n)
    assert one.attrs["z"] == pytest.approx(z, rel=1e-12)
    reference = sm_proportion.proportions_ztest(y.sum(), n, value=0.35, prop_var=0.35)
    assert one.attrs["z"] == pytest.approx(reference[0], rel=1e-10)
    assert one.attrs["p_value"] == pytest.approx(reference[1], rel=1e-9)
    assert one.attrs["p_lower"] == pytest.approx(stats.norm.cdf(z), rel=1e-9)
    assert one.attrs["p_upper"] == pytest.approx(stats.norm.sf(z), rel=1e-9)
    se = math.sqrt(phat * (1 - phat) / n)
    critical = stats.norm.isf(0.025)
    assert one.loc["y", ["std_error", "ci_low", "ci_high"]].tolist() == pytest.approx(
        [se, phat - critical * se, phat + critical * se])
    two = oe.prtest({"y": y, "g": g}, "y", by="g", alpha=0.01)
    counts = np.array([y[g == 0].sum(), y[g == 1].sum()])
    sizes = np.array([(g == 0).sum(), (g == 1).sum()])
    reference = sm_proportion.proportions_ztest(counts, sizes)
    assert two.attrs["z"] == pytest.approx(reference[0], rel=1e-10)
    assert two.attrs["p_value"] == pytest.approx(reference[1], rel=1e-9)
    p1, p2 = counts / sizes
    se = math.sqrt(p1 * (1 - p1) / sizes[0] + p2 * (1 - p2) / sizes[1])
    critical = stats.norm.isf(0.005)
    assert two.loc["diff", ["proportion", "std_error", "ci_low", "ci_high"]].tolist() == \
        pytest.approx([p1 - p2, se, p1 - p2 - critical * se, p1 - p2 + critical * se])
    # The two-sample z squared is the Pearson chi-square of the 2x2 table.
    chi2 = stats.chi2_contingency(pd.crosstab(g, y).to_numpy(), correction=False).statistic
    assert two.attrs["z"] ** 2 == pytest.approx(chi2, rel=1e-10)


def test_chi2gof_against_scipy():
    rng = np.random.default_rng(80)
    y = rng.choice(list("abcd"), size=120, p=[0.1, 0.2, 0.3, 0.4])
    observed = np.array([(y == level).sum() for level in "abcd"])
    equal = oe.chi2gof({"y": y}, "y")
    reference = stats.chisquare(observed)
    assert equal.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert equal.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-9)
    shares = np.array([1.0, 2.0, 3.0, 4.0])
    weighted = oe.chi2gof({"y": y}, "y", expected=list(shares))
    reference = stats.chisquare(observed, 120 * shares / shares.sum())
    assert weighted.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert weighted.attrs["df"] == 3
    mapped = oe.chi2gof({"y": y}, "y", expected={"d": 4, "c": 3, "b": 2, "a": 1, "e": 2})
    full = np.r_[observed, 0]
    reference = stats.chisquare(full, 120 * np.array([1, 2, 3, 4, 2.0]) / 12)
    assert mapped.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert mapped.attrs["df"] == 4 and mapped.attrs["k"] == 5
    # Frequency weights equal expanded rows.
    compact = oe.chi2gof({"y": list("abcd"), "w": observed}, "y", weights="w",
                         expected=[1, 2, 3, 4])
    assert compact.attrs["statistic"] == pytest.approx(weighted.attrs["statistic"], rel=1e-12)
    frequencies = weighted["frequencies"]
    fitted = 120 * shares / shares.sum()
    assert frequencies["std_residual"].to_numpy() == pytest.approx(
        (observed - fitted) / np.sqrt(fitted))


# ---- cross-tabulation --------------------------------------------------------------------

# Tables without tied maxima (lambda is not differentiable at a tie).
TABLES = [
    np.array([[17, 20, 1, 21], [12, 13, 16, 7], [24, 2, 7, 10]]),
    np.array([[14, 10], [4, 9]]),
    np.array([[1, 2, 4, 24], [5, 16, 19, 6], [7, 11, 8, 25], [5, 22, 20, 21]]),
    np.array([[3, 10, 16], [12, 16, 17], [16, 2, 24], [14, 22, 7], [9, 23, 5]]),
    np.array([[30, 0, 4], [2, 11, 0], [0, 6, 9]]),                    # zero cells
]


def pair_counts(p: np.ndarray) -> tuple[float, float]:
    concordant = discordant = 0.0
    rows, columns = p.shape
    for i in range(rows):
        for j in range(columns):
            concordant += p[i, j] * (p[:i, :j].sum() + p[i + 1:, j + 1:].sum())
            discordant += p[i, j] * (p[:i, j + 1:].sum() + p[i + 1:, :j].sum())
    return concordant, discordant


def association(p: np.ndarray) -> dict[str, float]:
    """Every measure of ``crosstab`` written from its textbook definition (proportions)."""
    p = p / p.sum()
    rows, columns = p.shape
    r, c = p.sum(1), p.sum(0)
    concordant, discordant = pair_counts(p)
    untied_r, untied_c = 1 - (r ** 2).sum(), 1 - (c ** 2).sum()
    m = min(rows, columns)
    out = {
        "gamma": (concordant - discordant) / (concordant + discordant),
        "kendall_tau_b": (concordant - discordant) / math.sqrt(untied_r * untied_c),
        "kendall_tau_c": m * (concordant - discordant) / (m - 1),
        "somers_d_symmetric": 2 * (concordant - discordant) / (untied_r + untied_c),
        "somers_d_row": (concordant - discordant) / untied_c,
        "somers_d_column": (concordant - discordant) / untied_r,
        "lambda_column": (p.max(1).sum() - c.max()) / (1 - c.max()),
        "lambda_row": (p.max(0).sum() - r.max()) / (1 - r.max()),
        "lambda_symmetric": (p.max(1).sum() + p.max(0).sum() - c.max() - r.max())
        / (2 - c.max() - r.max()),
        "goodman_kruskal_tau_column": ((p ** 2 / r[:, None]).sum() - (c ** 2).sum())
        / (1 - (c ** 2).sum()),
        "goodman_kruskal_tau_row": ((p ** 2 / c[None, :]).sum() - (r ** 2).sum())
        / (1 - (r ** 2).sum()),
    }

    def entropy(v: np.ndarray) -> float:
        v = v[v > 0]
        return float(-(v * np.log(v)).sum())

    h_r, h_c, h_rc = entropy(r), entropy(c), entropy(p.reshape(-1))
    out["uncertainty_symmetric"] = 2 * (h_r + h_c - h_rc) / (h_r + h_c)
    out["uncertainty_row"] = (h_r + h_c - h_rc) / h_r
    out["uncertainty_column"] = (h_r + h_c - h_rc) / h_c

    def correlation(x: np.ndarray, y: np.ndarray) -> float:
        mx, my = (r * x).sum(), (c * y).sum()
        return float((p * np.outer(x - mx, y - my)).sum()
                     / math.sqrt((r * (x - mx) ** 2).sum() * (c * (y - my) ** 2).sum()))

    out["pearson_r"] = correlation(np.arange(rows, dtype=float), np.arange(columns, dtype=float))
    out["spearman_rho"] = correlation(np.cumsum(r) - r / 2, np.cumsum(c) - c / 2)
    if rows == columns:
        index = np.arange(rows)
        gap = np.abs(index[:, None] - index[None, :]) / (rows - 1)
        for name, w in (("kappa", np.eye(rows)), ("kappa_linear", 1 - gap),
                        ("kappa_quadratic", 1 - gap ** 2)):
            observed, chance = (w * p).sum(), (w * np.outer(r, c)).sum()
            out[name] = (observed - chance) / (1 - chance)
    return out


def multinomial_ase(counts: np.ndarray, name: str) -> float:
    """Delta-method standard error under multinomial sampling, by numerical differentiation."""
    total = counts.sum()
    p = counts / total
    gradient = np.zeros_like(p)
    step = 1e-6
    for index in np.ndindex(*p.shape):
        up, down = p.copy(), p.copy()
        up[index] += step
        if p[index] > 0:
            down[index] -= step
            gradient[index] = (association(up)[name] - association(down)[name]) / (2 * step)
        else:
            gradient[index] = (association(up)[name] - association(p)[name]) / step
    variance = ((p * gradient ** 2).sum() - (p * gradient).sum() ** 2) / total
    return math.sqrt(max(variance, 0.0))


@pytest.mark.parametrize("counts", TABLES, ids=["3x4", "2x2", "4x4", "5x3", "zeros"])
def test_crosstab_measures_and_ase1_from_definitions(counts):
    result = oe.crosstab(expand(counts), "r", "c")
    measures = result["measures"]
    reference = association(counts.astype(float))
    two_by_two = counts.shape == (2, 2)
    for name, value in reference.items():
        if two_by_two and name in ("kappa_linear", "kappa_quadratic"):
            assert name not in measures.index          # identical to kappa for 2 categories
            continue
        assert measures.loc[name, "value"] == pytest.approx(value, rel=1e-10, abs=1e-12), name
        if (counts == 0).any() and name.startswith(("lambda", "uncertainty")):
            continue                                   # one-sided derivative at empty cells
        assert measures.loc[name, "ase"] == pytest.approx(
            multinomial_ase(counts.astype(float), name), rel=2e-5, abs=1e-9), name
    total = counts.sum()
    chi2 = stats.chi2_contingency(counts, correction=False).statistic
    assert abs(measures.loc["phi", "value"]) == pytest.approx(math.sqrt(chi2 / total))
    assert measures.loc["cramers_v", "value"] == pytest.approx(
        stats.contingency.association(counts, method="cramer"), rel=1e-10)
    assert measures.loc["contingency_coefficient", "value"] == pytest.approx(
        stats.contingency.association(counts, method="pearson"), rel=1e-10)
    frame = expand(counts)
    assert measures.loc["kendall_tau_b", "value"] == pytest.approx(
        stats.kendalltau(frame.r, frame.c, variant="b").statistic, rel=1e-10)
    assert measures.loc["kendall_tau_c", "value"] == pytest.approx(
        stats.kendalltau(frame.r, frame.c, variant="c").statistic, rel=1e-10)
    assert measures.loc["somers_d_column", "value"] == pytest.approx(
        stats.somersd(frame.r, frame.c).statistic, rel=1e-10)       # d(Y | X): column dependent
    assert measures.loc["somers_d_row", "value"] == pytest.approx(
        stats.somersd(frame.c, frame.r).statistic, rel=1e-10)
    assert measures.loc["spearman_rho", "value"] == pytest.approx(
        stats.spearmanr(frame.r, frame.c).statistic, rel=1e-10)
    pearson = stats.pearsonr(frame.r, frame.c)
    assert measures.loc["pearson_r", "value"] == pytest.approx(pearson.statistic, rel=1e-10)
    assert measures.loc["pearson_r", "p_value"] == pytest.approx(pearson.pvalue, rel=1e-8)
    t = pearson.statistic * math.sqrt((total - 2) / (1 - pearson.statistic ** 2))
    assert measures.loc["pearson_r", "t"] == pytest.approx(t, rel=1e-9)


@pytest.mark.parametrize("counts", TABLES[:4], ids=["3x4", "2x2", "4x4", "5x3"])
def test_crosstab_null_standard_errors_follow_the_published_formulas(counts):
    measures = oe.crosstab(expand(counts), "r", "c")["measures"]
    f = counts.astype(float)
    total = f.sum()
    rows, columns = f.shape
    r, c = f.sum(1), f.sum(0)
    # C_ij - D_ij for every cell by direct summation.
    difference = np.zeros_like(f)
    for i in range(rows):
        for j in range(columns):
            difference[i, j] = (f[:i, :j].sum() + f[i + 1:, j + 1:].sum()
                                - f[:i, j + 1:].sum() - f[i + 1:, :j].sum())
    p_minus_q = (f * difference).sum()
    null = math.sqrt((f * difference ** 2).sum() - p_minus_q ** 2 / total)
    d_r, d_c = total ** 2 - (r ** 2).sum(), total ** 2 - (c ** 2).sum()
    concordant, discordant = pair_counts(f)
    m = min(rows, columns)
    ase0 = {"gamma": 2 * null / (concordant + discordant),
            "kendall_tau_b": 2 * null / math.sqrt(d_r * d_c),
            "kendall_tau_c": 2 * m * null / ((m - 1) * total ** 2),
            "somers_d_symmetric": 4 * null / (d_r + d_c),
            "somers_d_row": 2 * null / d_c, "somers_d_column": 2 * null / d_r}
    for name, error in ase0.items():
        t = measures.loc[name, "value"] / error
        assert measures.loc[name, "t"] == pytest.approx(t, rel=1e-9), name
        assert measures.loc[name, "p_value"] == pytest.approx(2 * stats.norm.sf(abs(t)),
                                                              rel=1e-8), name
    # tau-c: ASE1 equals ASE0 (SPSS algorithms).
    assert measures.loc["kendall_tau_c", "ase"] == pytest.approx(ase0["kendall_tau_c"], rel=1e-10)
    # SciPy's tau-b test uses the permutation variance given the margins; the ASE0 above
    # plugs the observed cells into the multinomial variance (Brown and Benedetti 1977), so
    # the two p-values agree only roughly.
    frame = expand(counts)
    scipy_p = stats.kendalltau(frame.r, frame.c, method="asymptotic").pvalue
    assert measures.loc["kendall_tau_b", "p_value"] == pytest.approx(scipy_p, abs=0.08)
    # Goodman-Kruskal tau: (W - 1)(C - 1) tau ~ chi2((R - 1)(C - 1)) (Light and Margolin).
    df = (rows - 1) * (columns - 1)
    tau = measures.loc["goodman_kruskal_tau_column", "value"]
    assert measures.loc["goodman_kruskal_tau_column", "p_value"] == pytest.approx(
        stats.chi2.sf((total - 1) * (columns - 1) * tau, df), rel=1e-9)
    tau = measures.loc["goodman_kruskal_tau_row", "value"]
    assert measures.loc["goodman_kruskal_tau_row", "p_value"] == pytest.approx(
        stats.chi2.sf((total - 1) * (rows - 1) * tau, df), rel=1e-9)
    # Lambda: Goodman and Kruskal's closed-form ASE1.
    modal = int(np.argmax(c))
    explained = f.max(1).sum()
    in_modal = sum(f[i].max() for i in range(rows) if int(np.argmax(f[i])) == modal)
    ase1 = math.sqrt((total - explained) * (explained + c[modal] - 2 * in_modal)
                     / (total - c[modal]) ** 3)
    assert measures.loc["lambda_column", "ase"] == pytest.approx(ase1, rel=1e-10, abs=1e-12)
    # Uncertainty coefficients carry the likelihood-ratio p; phi / V / C the Pearson p.
    g2 = stats.chi2_contingency(counts, correction=False, lambda_="log-likelihood")
    assert measures.loc["uncertainty_symmetric", "p_value"] == pytest.approx(g2.pvalue, rel=1e-9)
    pearson = stats.chi2_contingency(counts, correction=False)
    assert measures.loc["cramers_v", "p_value"] == pytest.approx(pearson.pvalue, rel=1e-9)


@pytest.mark.parametrize("counts", TABLES, ids=["3x4", "2x2", "4x4", "5x3", "zeros"])
def test_crosstab_tests_against_scipy(counts):
    result = oe.crosstab(expand(counts), "r", "c", expected=True, residuals=True,
                         percentages="all")
    tests = result["tests"]
    pearson = stats.chi2_contingency(counts, correction=False)
    assert tests.loc["pearson", ["statistic", "df", "p_value"]].tolist() == pytest.approx(
        [pearson.statistic, pearson.dof, pearson.pvalue], rel=1e-9)
    g2 = stats.chi2_contingency(counts, correction=False, lambda_="log-likelihood")
    assert tests.loc["likelihood_ratio", ["statistic", "p_value"]].tolist() == pytest.approx(
        [g2.statistic, g2.pvalue], rel=1e-9)
    frame = expand(counts)
    total = counts.sum()
    linear = (total - 1) * stats.pearsonr(frame.r, frame.c).statistic ** 2
    assert tests.loc["linear_by_linear", "statistic"] == pytest.approx(linear, rel=1e-9)
    assert tests.loc["linear_by_linear", "p_value"] == pytest.approx(stats.chi2.sf(linear, 1),
                                                                    rel=1e-9)
    assert result["expected"].to_numpy() == pytest.approx(pearson.expected_freq)
    f, fitted = counts.astype(float), pearson.expected_freq
    assert result["residuals"].to_numpy() == pytest.approx((f - fitted) / np.sqrt(fitted))
    adjusted = (f - fitted) / np.sqrt(fitted * np.outer(1 - f.sum(1) / total,
                                                        1 - f.sum(0) / total))
    assert result["adjusted_residuals"].to_numpy() == pytest.approx(adjusted)
    assert result["row_percent"].to_numpy()[:-1, :-1] == pytest.approx(
        100 * f / f.sum(1, keepdims=True))
    assert result["column_percent"].to_numpy()[:-1, :-1] == pytest.approx(
        100 * f / f.sum(0, keepdims=True))
    assert result["total_percent"].to_numpy()[:-1, :-1] == pytest.approx(100 * f / total)
    assert result["counts"].to_numpy()[-1, -1] == total
    assert result.attrs["cells_below_5"] == int((fitted < 5).sum())
    assert result.attrs["min_expected"] == pytest.approx(fitted.min())
    if counts.shape == (2, 2):
        yates = stats.chi2_contingency(counts, correction=True)
        assert tests.loc["continuity", ["statistic", "p_value"]].tolist() == pytest.approx(
            [yates.statistic, yates.pvalue], rel=1e-9)
        assert tests.loc["fisher_exact", "p_value"] == pytest.approx(
            stats.fisher_exact(counts).pvalue, rel=1e-9)
        assert tests.loc["fisher_exact_less", "p_value"] == pytest.approx(
            stats.fisher_exact(counts, alternative="less").pvalue, rel=1e-9)
        assert tests.loc["fisher_exact_greater", "p_value"] == pytest.approx(
            stats.fisher_exact(counts, alternative="greater").pvalue, rel=1e-9)
    else:
        assert "fisher_exact" not in tests.index           # r x c only on request


def all_tables(rows: list[int], columns: list[int]):
    """Every nonnegative integer table with the given margins (recursive enumeration)."""
    if len(rows) == 1:
        yield [list(columns)]
        return

    def first_rows(remaining: int, caps: list[int]):
        if len(caps) == 1:
            if remaining <= caps[0]:
                yield [remaining]
            return
        for value in range(min(remaining, caps[0]) + 1):
            for rest in first_rows(remaining - value, caps[1:]):
                yield [value, *rest]

    for line in first_rows(rows[0], columns):
        left = [c - v for c, v in zip(columns, line)]
        for rest in all_tables(rows[1:], left):
            yield [line, *rest]


@pytest.mark.parametrize("counts", [np.array([[3, 1, 0], [1, 2, 2], [0, 1, 4]]),
                                    np.array([[2, 0, 1, 3], [1, 4, 0, 1]]),
                                    np.array([[5, 1], [0, 3], [2, 2], [1, 0]])])
def test_fisher_freeman_halton_by_independent_enumeration(counts):
    result = oe.crosstab(expand(counts), "r", "c", exact=True)
    rows, columns = counts.sum(1).tolist(), counts.sum(0).tolist()
    total = counts.sum()
    constant = (sum(math.lgamma(v + 1) for v in rows) + sum(math.lgamma(v + 1) for v in columns)
                - math.lgamma(total + 1))

    def probability(table) -> float:
        return math.exp(constant - sum(math.lgamma(v + 1) for line in table for v in line))

    observed = probability(counts.tolist())
    probabilities = np.array([probability(table) for table in all_tables(rows, columns)])
    assert probabilities.sum() == pytest.approx(1.0, rel=1e-10)
    p = probabilities[probabilities <= observed * (1 + 1e-7)].sum()
    assert result.attrs["exact_method"] == "enumeration"
    assert result.attrs["p_exact"] == pytest.approx(p, rel=1e-9)
    assert result["tests"].loc["fisher_exact", "p_value"] == pytest.approx(p, rel=1e-9)


def test_fisher_monte_carlo_covers_the_exact_value():
    from openecon.econometrics.nonparametric import tables as table_kernels
    import torch

    counts = np.array([[6, 2, 1, 4], [1, 5, 3, 2], [2, 2, 7, 1]])
    exact = oe.crosstab(expand(counts), "r", "c", exact=True).attrs["p_exact"]
    p, low, high = table_kernels.fisher_monte_carlo(
        torch.as_tensor(counts, dtype=torch.float64), 20_000, 11)
    assert low <= exact <= high and abs(p - exact) < 0.01
    again = table_kernels.fisher_monte_carlo(torch.as_tensor(counts, dtype=torch.float64),
                                             20_000, 11)
    assert again == (p, low, high)                               # seeded: reproducible
    half = stats.norm.isf(0.005) * math.sqrt(p * (1 - p) / 20_000)
    assert (low, high) == pytest.approx((p - half, p + half))


def test_two_by_two_risk_and_stratified_analysis_against_statsmodels():
    rng = np.random.default_rng(91)
    strata = rng.integers(2, 30, size=(6, 2, 2))
    frames = []
    for index, table in enumerate(strata):
        frames.append(expand(table).assign(s=index))
    data = pd.concat(frames, ignore_index=True)
    result = oe.crosstab(data, "r", "c", layer="s", alpha=0.1)
    reference = sm_tables.StratifiedTable(strata.transpose(1, 2, 0).astype(float))
    cmh = result["cmh"]
    plain = reference.test_null_odds(correction=False)
    corrected = reference.test_null_odds(correction=True)
    assert cmh.loc["cmh", ["statistic", "p_value"]].tolist() == pytest.approx(
        [plain.statistic, plain.pvalue], rel=1e-9)
    assert cmh.loc["cmh_continuity", ["statistic", "p_value"]].tolist() == pytest.approx(
        [corrected.statistic, corrected.pvalue], rel=1e-9)
    a = strata[:, 0, 0].astype(float)
    n = strata.sum((1, 2)).astype(float)
    r1, c1 = strata[:, 0].sum(1), strata[:, :, 0].sum(1)
    cochran = (a - r1 * c1 / n).sum() ** 2 / (r1 * (n - r1) * c1 * (n - c1) / n ** 3).sum()
    assert cmh.loc["cochran", "statistic"] == pytest.approx(cochran, rel=1e-10)
    breslow_day = reference.test_equal_odds(adjust=False)
    tarone = reference.test_equal_odds(adjust=True)
    assert cmh.loc["breslow_day", ["statistic", "df", "p_value"]].tolist() == pytest.approx(
        [breslow_day.statistic, 5, breslow_day.pvalue], rel=1e-8)
    assert cmh.loc["tarone", ["statistic", "p_value"]].tolist() == pytest.approx(
        [tarone.statistic, tarone.pvalue], rel=1e-8)
    common = result["common_odds_ratio"].loc["mantel_haenszel"]
    assert common["value"] == pytest.approx(reference.oddsratio_pooled, rel=1e-10)
    assert common["std_error_log"] == pytest.approx(reference.logodds_pooled_se, rel=1e-10)
    low, high = reference.oddsratio_pooled_confint(alpha=0.1)
    assert (common["ci_low"], common["ci_high"]) == pytest.approx((low, high), rel=1e-9)
    wald = reference.logodds_pooled / reference.logodds_pooled_se
    assert common["z"] == pytest.approx(wald, rel=1e-9)
    assert common["p_value"] == pytest.approx(2 * stats.norm.sf(abs(wald)), rel=1e-8)
    # Each layer's risk block against Table2x2.
    risk = result["risk"]
    for index, table in enumerate(strata):
        two = sm_tables.Table2x2(table.astype(float))
        prefix = f"s={index} | "
        assert risk.loc[prefix + "odds_ratio", "value"] == pytest.approx(two.oddsratio)
        assert risk.loc[prefix + "odds_ratio", ["ci_low", "ci_high"]].tolist() == pytest.approx(
            list(two.oddsratio_confint(0.1)), rel=1e-9)
        assert risk.loc[prefix + "risk_ratio_column_1", "value"] == pytest.approx(two.riskratio)
        assert risk.loc[prefix + "risk_ratio_column_1",
                        ["ci_low", "ci_high"]].tolist() == pytest.approx(
            list(two.riskratio_confint(0.1)), rel=1e-9)
        flipped = sm_tables.Table2x2(table[:, ::-1].astype(float))
        assert risk.loc[prefix + "risk_ratio_column_2", "value"] == pytest.approx(
            flipped.riskratio)
    # The total block is the pooled table.
    pooled = strata.sum(0)
    assert result["tests"].loc["total | pearson", "statistic"] == pytest.approx(
        stats.chi2_contingency(pooled, correction=False).statistic, rel=1e-10)
    assert result.attrs["statistic"] == pytest.approx(
        stats.chi2_contingency(pooled, correction=False).statistic, rel=1e-10)


def test_kappa_against_statsmodels_including_null_errors():
    from statsmodels.stats.inter_rater import cohens_kappa

    counts = np.array([[21, 4, 1, 0], [5, 17, 6, 2], [1, 7, 15, 4], [0, 2, 5, 10]])
    measures = oe.crosstab(expand(counts), "r", "c")["measures"]
    for name, wt in (("kappa", None), ("kappa_linear", "linear"),
                     ("kappa_quadratic", "quadratic")):
        reference = cohens_kappa(counts, wt=wt) if wt else cohens_kappa(counts)
        row = measures.loc[name]
        assert row["value"] == pytest.approx(reference.kappa, rel=1e-10)
        assert row["ase"] == pytest.approx(reference.std_kappa, rel=1e-8)
        assert row["t"] == pytest.approx(reference.kappa / reference.std_kappa0, rel=1e-8)
        assert row["p_value"] == pytest.approx(reference.pvalue_two_sided, rel=1e-7)


def test_crosstab_invariances_weights_permutations_and_transposition():
    rng = np.random.default_rng(92)
    counts = rng.integers(0, 12, size=(3, 4))
    counts[0, 0] = 0
    long = expand(counts).sample(frac=1.0, random_state=1).reset_index(drop=True)
    compact = pd.DataFrame({"r": np.repeat(np.arange(3), 4), "c": np.tile(np.arange(4), 3),
                            "w": counts.reshape(-1)})
    a = oe.crosstab(long, "r", "c", exact=True)
    b = oe.crosstab(compact, "r", "c", weights="w", exact=True)
    for name in ("counts", "tests", "measures"):
        left, right = a[name].to_numpy(dtype=float), b[name].to_numpy(dtype=float)
        assert np.allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True), name
    assert a.attrs["p_exact"] == pytest.approx(b.attrs["p_exact"], rel=1e-12)
    # Transposition swaps the roles of row and column.
    t = oe.crosstab(long.rename(columns={"r": "c", "c": "r"}), "r", "c")["measures"]
    m = a["measures"]
    for left, right in (("lambda_row", "lambda_column"), ("somers_d_row", "somers_d_column"),
                        ("uncertainty_row", "uncertainty_column"),
                        ("goodman_kruskal_tau_row", "goodman_kruskal_tau_column")):
        assert t.loc[left, ["value", "ase"]].tolist() == pytest.approx(
            m.loc[right, ["value", "ase"]].tolist(), rel=1e-10)
    for name in ("gamma", "kendall_tau_b", "kendall_tau_c", "spearman_rho", "cramers_v"):
        assert t.loc[name, "value"] == pytest.approx(m.loc[name, "value"], rel=1e-10)
    # Scaling the weights changes N but not the measures of association.
    scaled = oe.crosstab(compact.assign(w=compact.w * 2.5), "r", "c", weights="w")
    assert scaled["measures"]["value"].to_numpy() == pytest.approx(
        b["measures"]["value"].to_numpy(), rel=1e-10)
    assert scaled.attrs["statistic"] == pytest.approx(2.5 * b.attrs["statistic"], rel=1e-10)
    assert "fisher_exact" not in scaled["tests"].index      # non-integer counts: no exact test


def test_tabulate_against_pandas():
    rng = np.random.default_rng(93)
    x = pd.Series(rng.choice(["a", "b", "c", None], size=200, p=[0.4, 0.3, 0.2, 0.1]))
    table = oe.tabulate({"x": x}, "x")
    counts = x.value_counts().sort_index()
    valid = counts.sum()
    assert table["count"].tolist() == [*counts.tolist(), 200 - valid]
    assert table["percent"].tolist() == pytest.approx([*(100 * counts / 200), 100 * (200 - valid)
                                                       / 200])
    assert table["valid_percent"].tolist()[:-1] == pytest.approx((100 * counts / valid).tolist())
    assert table["cumulative_percent"].tolist()[:-1] == pytest.approx(
        (100 * counts.cumsum() / valid).tolist())
    assert table.attrs["n"] == valid and table.attrs["n_missing"] == 200 - valid
    assert table.attrs["mode"] == counts.idxmax()
    w = rng.integers(1, 5, size=200)
    weighted = oe.tabulate({"x": x, "w": w}, "x", weights="w")
    expected = pd.Series(w).groupby(x).sum().sort_index()
    assert weighted["count"].tolist()[:-1] == expected.tolist()
    assert weighted.attrs["n_total"] == int(w.sum())


# ---- ROC ---------------------------------------------------------------------------------


def hanley_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "disease": np.r_[np.zeros(58, dtype=int), np.ones(51, dtype=int)],
        "rating": np.r_[np.repeat(np.arange(1, 6), HM_NORMAL),
                        np.repeat(np.arange(1, 6), HM_ABNORMAL)]})


def test_roc_reproduces_stata_roctab_and_hanley_mcneil_1982():
    result = oe.roc(hanley_frame(), "disease", "rating")
    attrs = result.attrs
    # [R] roctab (hanley.dta): Obs 109, area 0.8932, Std. err. 0.0307, 95% CI 0.83295 0.95339.
    assert attrs["n"] == 109
    assert attrs["auc"] == pytest.approx(0.8932, abs=5e-5)
    assert attrs["std_error"] == pytest.approx(0.0307, abs=5e-5)
    assert result["auc"].loc["delong", ["ci_low", "ci_high"]].tolist() == pytest.approx(
        [0.83295, 0.95339], abs=6e-6)
    # Hanley and McNeil (1982): W = 0.893, SE = 0.032 from their Table II sums.
    normal, abnormal = np.array(HM_NORMAL, float), np.array(HM_ABNORMAL, float)
    above = np.array([abnormal[j + 1:].sum() for j in range(5)])
    below = np.array([normal[:j].sum() for j in range(5)])
    w = (normal * above + 0.5 * normal * abnormal).sum() / (58 * 51)
    q1 = (normal * (above ** 2 + above * abnormal + abnormal ** 2 / 3)).sum() / (58 * 51 ** 2)
    q2 = (abnormal * (below ** 2 + below * normal + normal ** 2 / 3)).sum() / (58 ** 2 * 51)
    se = math.sqrt((w * (1 - w) + 50 * (q1 - w * w) + 57 * (q2 - w * w)) / (58 * 51))
    assert attrs["auc"] == pytest.approx(w, rel=1e-12)
    assert attrs["se_hanley_mcneil"] == pytest.approx(se, rel=1e-12)
    assert round(attrs["auc"], 3) == 0.893 and round(attrs["se_hanley_mcneil"], 3) == 0.032
    # SPSS's Asymptotic Sig. uses sd = sqrt((n1 + n0 + 1) / (12 n1 n0)); ours is tie-corrected.
    untied = math.sqrt(110 / (12 * 58 * 51))
    assert attrs["se_null_uncorrected"] == pytest.approx(untied, rel=1e-12)
    assert attrs["z_uncorrected"] == pytest.approx((w - 0.5) / untied, rel=1e-12)
    assert attrs["p_value_uncorrected"] == pytest.approx(
        2 * stats.norm.sf((w - 0.5) / untied), rel=1e-8)
    ranksum = oe.ranksum(hanley_frame(), "rating", "disease").attrs
    assert attrs["z"] == pytest.approx(-ranksum["z"], rel=1e-12)
    assert attrs["se_null"] < attrs["se_null_uncorrected"]
    # Coordinates: cutoff 3.5 classifies ratings 4-5 as positive.
    curve = result["curve"].set_index("threshold")
    assert curve.loc[3.5, "sensitivity"] == pytest.approx(44 / 51)
    assert curve.loc[3.5, "specificity"] == pytest.approx(45 / 58)


@pytest.mark.parametrize("seed, n, decimals", [(95, 40, None), (96, 120, 0), (97, 9, 1)])
def test_roc_against_brute_force_pairs(seed, n, decimals):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 3, dtype=int), np.zeros(n - n // 3, dtype=int)]
    s = rng.normal(size=n) + 0.8 * y
    if decimals is not None:
        s = np.round(s, decimals)
    result = oe.roc({"y": y, "s": s}, "y", "s", alpha=0.2)
    attrs = result.attrs
    positive, negative = s[y == 1], s[y == 0]
    psi = (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
    n1, n0 = psi.shape
    auc = psi.mean()
    assert attrs["auc"] == pytest.approx(auc, rel=1e-12)
    delong = math.sqrt(psi.mean(1).var(ddof=1) / n1 + psi.mean(0).var(ddof=1) / n0)
    assert attrs["std_error"] == pytest.approx(delong, rel=1e-10)
    # Hanley-McNeil Q1, Q2 by counting triples; a tied pair counts 1/3 when both are tied.
    greater = (positive[:, None] > negative[None, :]).astype(float)
    tied = (positive[:, None] == negative[None, :]).astype(float)
    q1 = np.mean([(greater[:, j].sum() ** 2 + greater[:, j].sum() * tied[:, j].sum()
                   + tied[:, j].sum() ** 2 / 3) for j in range(n0)]) / n1 ** 2
    q2 = np.mean([(greater[i].sum() ** 2 + greater[i].sum() * tied[i].sum()
                   + tied[i].sum() ** 2 / 3) for i in range(n1)]) / n0 ** 2
    hanley = math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
                        + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    assert attrs["se_hanley_mcneil"] == pytest.approx(hanley, rel=1e-10)
    critical = stats.norm.isf(0.1)
    assert result["auc"].loc["delong", ["ci_low", "ci_high"]].tolist() == pytest.approx(
        [max(0.0, auc - critical * delong), min(1.0, auc + critical * delong)])
    # The curve: sensitivity / specificity at every cutoff, and the trapezoidal area.
    curve = result["curve"]
    for _, line in curve.iterrows():
        predicted = s >= line["threshold"]
        assert line["sensitivity"] == pytest.approx(predicted[y == 1].mean())
        assert line["specificity"] == pytest.approx((~predicted)[y == 0].mean())
    x, yy = curve["one_minus_specificity"].to_numpy(), curve["sensitivity"].to_numpy()
    assert -np.trapezoid(yy, x) == pytest.approx(auc, rel=1e-12)
    youden = (curve["sensitivity"] + curve["specificity"] - 1).max()
    assert attrs["youden_index"] == pytest.approx(youden)
    reference = stats.mannwhitneyu(positive, negative, use_continuity=False, method="asymptotic")
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-9)


def test_roccomp_against_brute_force_delong():
    rng = np.random.default_rng(98)
    n = 150
    y = (rng.random(n) < 0.4).astype(int)
    s1 = np.round(rng.normal(size=n) + 1.0 * y, 1)
    s2 = np.round(0.6 * s1 + rng.normal(size=n) + 0.3 * y, 1)
    s3 = rng.normal(size=n) + 0.2 * y
    result = oe.roccomp({"y": y, "a": s1, "b": s2, "c": s3}, "y", ["a", "b", "c"])
    scores = np.column_stack([s1, s2, s3])
    positive, negative = scores[y == 1], scores[y == 0]
    n1, n0 = len(positive), len(negative)
    v10 = np.empty((n1, 3))
    v01 = np.empty((n0, 3))
    for k in range(3):
        psi = (positive[:, k][:, None] > negative[:, k][None, :]) \
            + 0.5 * (positive[:, k][:, None] == negative[:, k][None, :])
        v10[:, k], v01[:, k] = psi.mean(1), psi.mean(0)
    areas = v10.mean(0)
    covariance = np.cov(v10, rowvar=False) / n1 + np.cov(v01, rowvar=False) / n0
    assert np.array(list(result.attrs["auc"].values())) == pytest.approx(areas, rel=1e-12)
    assert np.array(result.attrs["covariance"]) == pytest.approx(covariance, rel=1e-9)
    contrast = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])        # another basis of contrasts
    difference = contrast @ areas
    chi2 = difference @ np.linalg.solve(contrast @ covariance @ contrast.T, difference)
    assert result.attrs["statistic"] == pytest.approx(chi2, rel=1e-9)
    assert result.attrs["df"] == 2
    assert result.attrs["p_value"] == pytest.approx(stats.chi2.sf(chi2, 2), rel=1e-8)
    pair = result["pairwise"].loc["a - c"]
    se = math.sqrt(covariance[0, 0] + covariance[2, 2] - 2 * covariance[0, 2])
    assert pair["z"] == pytest.approx((areas[0] - areas[2]) / se, rel=1e-9)
    assert pair["p_value"] == pytest.approx(2 * stats.norm.sf(abs(pair["z"])), rel=1e-8)
    # Two scores: chi2(1) = z^2 of the pair.
    two = oe.roccomp({"y": y, "a": s1, "b": s2}, "y", ["a", "b"])
    assert two.attrs["statistic"] == pytest.approx(two["pairwise"].loc["a - b", "z"] ** 2,
                                                   rel=1e-9)
    # The single-score standard error is the DeLong error of oe.roc.
    assert result["auc"].loc["a", "std_error"] == pytest.approx(
        oe.roc({"y": y, "a": s1}, "y", "a").attrs["std_error"], rel=1e-10)


# ---- invariances shared by the rank procedures ---------------------------------------------


def test_rank_procedures_are_invariant_to_monotone_transformations_and_row_order():
    rng = np.random.default_rng(99)
    n = 48
    y = np.round(rng.normal(size=n), 1)
    g2, g3 = rng.integers(0, 2, size=n), rng.integers(0, 3, size=n)
    frame = pd.DataFrame({"y": y, "g2": g2, "g3": g3, "x": np.round(rng.normal(size=n), 1)})
    moved = frame.assign(y=np.exp(frame.y / 3.0) * 1e4)             # strictly increasing map
    shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
    for call in (lambda d: oe.ranksum(d, "y", "g2"), lambda d: oe.kwallis(d, "y", "g3"),
                 lambda d: oe.jonckheere(d, "y", "g3"), lambda d: oe.median_test(d, "y", "g3"),
                 lambda d: oe.roc(d, "g2", "y"), lambda d: oe.ksmirnov(d, "y", by="g2")):
        base = call(frame).attrs
        for other in (call(moved).attrs, call(shuffled).attrs):
            assert other["p_value"] == pytest.approx(base["p_value"], rel=1e-10)


def test_missing_rows_are_deleted_listwise_and_counted():
    rng = np.random.default_rng(101)
    n = 60
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n),
                          "g": rng.integers(0, 2, size=n).astype(float),
                          "k": rng.integers(0, 3, size=n).astype(float)})
    holes = frame.copy()
    holes.loc[[3, 17], "y"] = np.nan
    holes.loc[[5], "g"] = np.nan
    holes.loc[[8, 17], "x"] = np.nan
    holes.loc[[11], "k"] = np.nan
    pairs = [
        (lambda d: oe.ranksum(d, "y", "g"), ["y", "g"]),
        (lambda d: oe.kwallis(d, "y", "k"), ["y", "k"]),
        (lambda d: oe.signrank(d, "y", "x"), ["y", "x"]),
        (lambda d: oe.friedman(d, ["y", "x", "k"]), ["y", "x", "k"]),
        (lambda d: oe.ksmirnov(d, "y"), ["y"]),
        (lambda d: oe.swilk(d, "y"), ["y"]),
        (lambda d: oe.crosstab(d, "g", "k"), ["g", "k"]),
        (lambda d: oe.roccomp(d, "g", ["y", "x"]), ["g", "y", "x"]),
    ]
    for call, used in pairs:
        complete = holes.dropna(subset=used)
        a, b = call(holes).attrs, call(complete).attrs
        assert a["n_dropped"] == len(holes) - len(complete) and b["n_dropped"] == 0
        assert a["n"] == b["n"] == len(complete)
        assert a["p_value"] == pytest.approx(b["p_value"], rel=1e-12)
        with pytest.raises(oe.AnalysisError) as error:
            _raise_on_missing(call, holes)
        assert error.value.code == "missing_values"


def _raise_on_missing(call, data):
    """Re-run ``call`` with missing='raise' by wrapping the public functions."""
    import functools

    names = ("ranksum", "kwallis", "signrank", "friedman", "ksmirnov", "swilk", "crosstab",
             "roccomp")
    originals = {name: getattr(oe, name) for name in names}
    try:
        for name, function in originals.items():
            setattr(oe, name, functools.partial(function, missing="raise"))
        return call(data)
    finally:
        for name, function in originals.items():
            setattr(oe, name, function)
