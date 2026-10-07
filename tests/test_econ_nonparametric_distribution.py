"""Nonparametric family: one-sample distribution tests and normality tests.

Oracles: scipy.stats (ks_1samp, ks_2samp, kstwo, shapiro, binomtest, chisquare,
skewtest, kurtosistest, normaltest, jarque_bera), scipy.special.kolmogorov,
statsmodels (lilliefors, runstest_1samp, proportions_ztest) and explicit NumPy.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.proportion import proportions_ztest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.nonparametric import exact as exact_kernels
from openecon.econometrics.nonparametric.distribution import clopper_pearson, lilliefors_p_value
from openecon.econometrics.nonparametric.normality import (
    normal_scores, royston_adjusted_chi2, shapiro_wilk_weights,
)

# ---- reference distributions -------------------------------------------------------------


@pytest.mark.parametrize("z", [0.05, 0.3, 0.5, 0.9, 1.0, 1.18, 1.5, 2.5, 4.0, 6.0])
def test_kolmogorov_limiting_distribution(z):
    assert exact_kernels.kolmogorov_sf(z) == pytest.approx(special.kolmogorov(z), rel=1e-12,
                                                           abs=1e-300)


@pytest.mark.parametrize("n, d", [(1, 0.7), (2, 0.3), (5, 0.3), (7, 0.07), (10, 0.9),
                                  (20, 0.2), (50, 0.11), (100, 0.1), (100, 0.2), (140, 0.3),
                                  (60, 0.5), (30, 0.02)])
def test_exact_kolmogorov_distribution_matches_kstwo(n, d):
    assert exact_kernels.kolmogorov_sf_exact(n, d) == pytest.approx(stats.kstwo.sf(d, n),
                                                                    rel=1e-9, abs=1e-14)


def test_exact_kolmogorov_matrix_and_tail_formula_agree_where_both_apply():
    # s = n d^2 just beyond the switch point: the Durbin matrix and 2 * one-sided agree.
    n, d = 120, 0.18
    matrix = 1.0 - exact_kernels.kolmogorov_cdf_matrix(n, d)
    tail = 2.0 * exact_kernels.kolmogorov_one_sided_sf(n, d)
    assert matrix == pytest.approx(tail, rel=1e-9)
    assert exact_kernels.kolmogorov_one_sided_sf(n, d) == pytest.approx(stats.ksone.sf(d, n),
                                                                        rel=1e-10)
    # Large n: close to the limiting law.
    assert exact_kernels.kolmogorov_sf_exact(4000, 0.02) == pytest.approx(
        special.kolmogorov(math.sqrt(4000) * 0.02), abs=0.01)


@pytest.mark.parametrize("n1, n2", [(4, 5), (6, 6), (12, 9), (40, 30)])
def test_exact_two_sample_distribution_matches_scipy(n1, n2):
    rng = np.random.default_rng(n1 * 100 + n2)
    x, y = rng.normal(size=n1), rng.normal(size=n2) + 0.6
    reference = stats.ks_2samp(x, y, method="exact")
    h = round(reference.statistic * n1 * n2)
    assert exact_kernels.smirnov_exact(n1, n2, h) == pytest.approx(reference.pvalue, rel=1e-10)
    assert exact_kernels.smirnov_exact(n1, n2, 0) == 1.0


def test_runs_distribution_matches_enumeration():
    n1, n2 = 4, 6
    counts = np.zeros(12)
    for positions in itertools.combinations(range(n1 + n2), n1):
        sequence = np.zeros(n1 + n2, dtype=int)
        sequence[list(positions)] = 1
        counts[1 + np.count_nonzero(np.diff(sequence))] += 1
    support, pmf = exact_kernels.runs_distribution(n1, n2)
    np.testing.assert_allclose(pmf.numpy(), counts[support.numpy().astype(int)] / counts.sum(),
                               atol=1e-14)
    assert support.tolist() == list(range(2, 10))


# ---- ksmirnov ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(31)
    return pd.DataFrame({"x": rng.normal(1.0, 2.0, size=40), "e": rng.exponential(3.0, size=40),
                         "g": rng.integers(0, 2, size=40), "k": rng.poisson(3.0, size=40)})


def test_ksmirnov_one_sample_with_given_parameters(sample):
    x = sample["x"].to_numpy()
    result = oe.ksmirnov(sample, "x", params=[1.0, 2.0])
    assert not isinstance(result, TableSet) and result.index.tolist() == ["positive", "negative",
                                                                        "combined"]
    attrs = result.attrs
    exact = stats.ks_1samp(x, stats.norm(1, 2).cdf, method="exact")
    assert attrs["statistic"] == pytest.approx(exact.statistic, rel=1e-12)
    assert attrs["p_exact"] == pytest.approx(exact.pvalue, rel=1e-9)
    assert attrs["p_value"] == attrs["p_exact"] and attrs["exact"] is True
    assert attrs["p_asymptotic"] == pytest.approx(
        special.kolmogorov(math.sqrt(40) * exact.statistic), rel=1e-12)
    greater = stats.ks_1samp(x, stats.norm(1, 2).cdf, alternative="greater").statistic
    less = stats.ks_1samp(x, stats.norm(1, 2).cdf, alternative="less").statistic
    assert attrs["d_plus"] == pytest.approx(greater) and attrs["d_minus"] == pytest.approx(-less)
    assert result.loc["positive", "p_value"] == pytest.approx(math.exp(-2 * 40 * greater ** 2))
    assert "p_exact" not in oe.ksmirnov(sample, "x", params=[1.0, 2.0], exact=False).attrs


@pytest.mark.parametrize("distribution, params, frozen", [
    ("exponential", [3.0], stats.expon(scale=3.0)),
    ("uniform", [0.0, 20.0], stats.uniform(0.0, 20.0)),
])
def test_ksmirnov_other_continuous_distributions(sample, distribution, params, frozen):
    attrs = oe.ksmirnov(sample, "e", distribution=distribution, params=params).attrs
    reference = stats.ks_1samp(sample["e"].to_numpy(), frozen.cdf, method="exact")
    assert attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert attrs["p_exact"] == pytest.approx(reference.pvalue, rel=1e-9)


def test_ksmirnov_estimated_normal_uses_lilliefors(sample):
    x = sample["x"].to_numpy()
    attrs = oe.ksmirnov(sample, "x").attrs
    statistic, _ = lilliefors(x, dist="norm", pvalmethod="approx")
    assert attrs["estimated"] is True and attrs["exact"] is False
    assert attrs["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert attrs["params"] == pytest.approx([x.mean(), x.std(ddof=1)])
    assert attrs["p_value"] == attrs["p_lilliefors"] < attrs["p_asymptotic"]
    with pytest.raises(AnalysisError) as error:
        oe.ksmirnov(sample, "x", exact=True)
    assert error.value.code == "exact_unavailable"


def test_lilliefors_p_value_against_statsmodels():
    rng = np.random.default_rng(32)
    checked_small = 0
    for n in (8, 20, 50, 100, 300, 2000):
        for skewed in (False, True):
            x = rng.exponential(size=n) if skewed else rng.normal(size=n)
            statistic, approx = lilliefors(x, dist="norm", pvalmethod="approx")
            _, tabulated = lilliefors(x, dist="norm", pvalmethod="table")
            p = lilliefors_p_value(statistic, n)
            if approx < 0.1:                     # Dallal-Wilkinson range: same formula
                assert p == pytest.approx(approx, rel=1e-9)
                checked_small += 1
            else:                                # outside it: within simulation-table accuracy
                assert p == pytest.approx(tabulated, abs=0.08)
    assert checked_small >= 4
    assert lilliefors_p_value(0.01, 50) == 1.0


def test_ksmirnov_poisson_uses_both_sides_of_every_jump(sample):
    k = sample["k"].to_numpy()
    result = oe.ksmirnov(sample, "k", distribution="poisson")
    lam = k.mean()
    grid = np.arange(0, k.max() + 2)
    empirical = np.array([(k <= value).mean() for value in grid])
    theoretical = stats.poisson.cdf(grid, lam)
    assert result.attrs["d_plus"] == pytest.approx((empirical - theoretical).max(), rel=1e-10)
    assert result.attrs["d_minus"] == pytest.approx(-(theoretical - empirical).max(), rel=1e-10)
    assert result.attrs["statistic"] == pytest.approx(np.abs(empirical - theoretical).max())
    assert result.attrs["exact"] is False and len(result.attrs["notes"]) == 2
    given = oe.ksmirnov(sample, "k", distribution="poisson", params=[2.5]).attrs
    theoretical = stats.poisson.cdf(grid, 2.5)
    assert given["statistic"] == pytest.approx(np.abs(empirical - theoretical).max())


def test_ksmirnov_two_sample(sample):
    x = sample["x"].to_numpy()
    first, second = x[sample.g == 0], x[sample.g == 1]
    result = oe.ksmirnov(sample, "x", by="g")
    reference = stats.ks_2samp(first, second, method="exact")
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert attrs["p_exact"] == pytest.approx(reference.pvalue, rel=1e-10)
    scale = math.sqrt(len(first) * len(second) / 40)
    assert attrs["z"] == pytest.approx(scale * reference.statistic)
    assert attrs["p_asymptotic"] == pytest.approx(special.kolmogorov(attrs["z"]), rel=1e-12)
    assert attrs["d_plus"] == pytest.approx(
        stats.ks_2samp(first, second, alternative="greater").statistic)
    assert attrs["d_minus"] == pytest.approx(
        -stats.ks_2samp(first, second, alternative="less").statistic)
    tied = sample.assign(x=np.round(sample.x))
    tied_result = oe.ksmirnov(tied, "x", by="g").attrs
    assert tied_result["ties"] is True and tied_result["exact"] is False
    assert tied_result["statistic"] == pytest.approx(
        stats.ks_2samp(tied.x[tied.g == 0], tied.x[tied.g == 1]).statistic)
    assert oe.ksmirnov(tied, "x", by="g", exact=True).attrs["notes"]


# ---- runs test -----------------------------------------------------------------------------


@pytest.mark.parametrize("correction", [True, False])
def test_runtest_matches_statsmodels(correction):
    rng = np.random.default_rng(33)
    series = rng.normal(size=30)
    result = oe.runtest({"y": series}, "y", threshold="mean", continuity=correction)
    z, p = runstest_1samp(series, cutoff="mean", correction=correction)
    assert result.attrs["z"] == pytest.approx(z, rel=1e-12)
    assert result.attrs["p_value"] == pytest.approx(p, rel=1e-10)
    default = oe.runtest({"y": series}, "y", threshold="mean")
    assert default.attrs["continuity"] is True                 # N < 50
    long = oe.runtest({"y": rng.normal(size=80)}, "y")
    assert long.attrs["continuity"] is False and long.attrs["exact"] is False


def test_runtest_exact_thresholds_and_ties():
    result = oe.runtest({"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, "y")
    assert result.attrs["runs"] == 2 and result.attrs["threshold"] == 5.5
    assert result.attrs["p_exact"] == pytest.approx(4 / 252)
    assert result.attrs["p_exact_lower"] == pytest.approx(2 / 252)
    series = [3, 1, 3, 5, 3, 2, 6, 3, 1, 7, 2, 5]
    above = oe.runtest({"y": series}, "y", threshold=3)
    below = oe.runtest({"y": series}, "y", threshold=3, ties="below")
    dropped = oe.runtest({"y": series}, "y", threshold=3, ties="drop")
    flags = np.array(series) >= 3
    assert above.attrs["n_above"] == flags.sum()
    assert above.attrs["runs"] == 1 + np.count_nonzero(np.diff(flags))
    assert below.attrs["n_above"] == (np.array(series) > 3).sum()
    assert dropped.attrs["n"] == 8


# ---- binomial, proportions, chi-square -------------------------------------------------------


@pytest.mark.parametrize("p", [0.5, 0.42, 0.9])
def test_bitest_matches_scipy(p):
    rng = np.random.default_rng(34)
    y = (rng.random(37) < 0.3).astype(int)
    k = int(y.sum())
    result = oe.bitest({"y": y}, "y", p=p, alpha=0.10)
    reference = stats.binomtest(k, 37, p)
    attrs = result.attrs
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-9)
    assert attrs["p_lower"] == pytest.approx(stats.binomtest(k, 37, p, alternative="less").pvalue,
                                             rel=1e-10)
    assert attrs["p_upper"] == pytest.approx(
        stats.binomtest(k, 37, p, alternative="greater").pvalue, rel=1e-10)
    interval = reference.proportion_ci(confidence_level=0.90, method="exact")
    assert (attrs["ci_low"], attrs["ci_high"]) == pytest.approx((interval.low, interval.high),
                                                                rel=1e-9)
    assert result.loc["y", "expected"] == pytest.approx(37 * p)


def test_bitest_edges_and_positive_level():
    assert clopper_pearson(0, 10, 0.05) == pytest.approx((0.0, 1 - 0.025 ** 0.1))
    assert clopper_pearson(10, 10, 0.05) == pytest.approx((0.025 ** 0.1, 1.0))
    all_ones = oe.bitest({"y": [1, 1, 1, 1]}, "y", p=0.5).attrs
    assert all_ones["successes"] == 4 and all_ones["p_value"] == pytest.approx(0.125)
    labels = oe.bitest({"y": ["no", "yes", "yes", "no", "no"]}, "y", positive="no").attrs
    assert labels["successes"] == 3 and labels["positive"] == "no"
    default = oe.bitest({"y": ["no", "yes", "yes", "no", "no"]}, "y").attrs
    assert default["successes"] == 2 and default["positive"] == "yes"


def test_prtest_one_and_two_samples_match_statsmodels():
    rng = np.random.default_rng(35)
    y = (rng.random(60) < 0.35).astype(int)
    g = rng.integers(0, 2, size=60)
    one = oe.prtest({"y": y}, "y", p=0.42)
    z, p = proportions_ztest(y.sum(), 60, 0.42, prop_var=0.42)
    assert one.attrs["z"] == pytest.approx(z) and one.attrs["p_value"] == pytest.approx(p)
    phat = y.mean()
    se = math.sqrt(phat * (1 - phat) / 60)
    assert one.loc["y", "ci_low"] == pytest.approx(phat - stats.norm.isf(0.025) * se)
    assert one.loc["y", "z"] == pytest.approx(z) and one.loc["y", "p_value"] == pytest.approx(p)
    assert one.attrs["p_lower"] + one.attrs["p_upper"] == pytest.approx(1.0)
    assert oe.prtest({"y": y}, "y").attrs["p"] == 0.5
    two = oe.prtest({"y": y, "g": g}, "y", by="g")
    counts = [y[g == 0].sum(), y[g == 1].sum()]
    sizes = [(g == 0).sum(), (g == 1).sum()]
    z, p = proportions_ztest(counts, sizes)
    assert two.attrs["z"] == pytest.approx(z) and two.attrs["p_value"] == pytest.approx(p)
    p1, p2 = counts[0] / sizes[0], counts[1] / sizes[1]
    unpooled = math.sqrt(p1 * (1 - p1) / sizes[0] + p2 * (1 - p2) / sizes[1])
    assert two.loc["diff", "std_error"] == pytest.approx(unpooled)
    assert two.loc["diff", "proportion"] == pytest.approx(p1 - p2)
    assert two.index.tolist() == ["0", "1", "diff"]


def test_chi2gof_matches_scipy_with_expected_shares_and_weights():
    rng = np.random.default_rng(36)
    c = rng.integers(1, 5, size=90)
    observed = np.bincount(c)[1:]
    equal = oe.chi2gof({"c": c}, "c")
    reference = stats.chisquare(observed)
    assert equal.attrs["statistic"] == pytest.approx(reference.statistic)
    assert equal.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    shares = np.array([1.0, 2.0, 3.0, 4.0])
    listed = oe.chi2gof({"c": c}, "c", expected=[1, 2, 3, 4])
    reference = stats.chisquare(observed, shares / shares.sum() * 90)
    assert listed.attrs["statistic"] == pytest.approx(reference.statistic)
    mapped = oe.chi2gof({"c": c}, "c", expected={4: 4, 3: 3, 2: 2, 1: 1})
    assert mapped.attrs["statistic"] == pytest.approx(reference.statistic)
    extra = oe.chi2gof({"c": c}, "c", expected={1: 1, 2: 1, 3: 1, 4: 1, 5: 0.1})
    assert extra.attrs["k"] == 5 and extra["frequencies"].loc["5", "observed"] == 0
    assert extra.attrs["cells_below_5"] == 1 and "less than 5" in extra.attrs["notes"][0]
    weighted = oe.chi2gof({"c": [1, 2, 3, 4], "w": observed.astype(float)}, "c", weights="w")
    assert weighted.attrs["statistic"] == pytest.approx(equal.attrs["statistic"])
    assert isinstance(equal, TableSet) and list(equal) == ["frequencies", "tests"]


# ---- normality -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 11, 12, 13, 30, 200, 1500, 5000])
def test_swilk_matches_scipy_shapiro(n):
    rng = np.random.default_rng(n)
    x = rng.gamma(3.0, size=n) if n % 2 else rng.normal(size=n)
    result = oe.swilk({"x": x}, "x")
    reference = stats.shapiro(x)
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-10)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-7, abs=1e-12)
    if n > 3:
        assert result.attrs["p_value"] == pytest.approx(stats.norm.sf(result.attrs["z"]))
    assert result.loc["x", "n"] == n


def test_shapiro_wilk_weights_are_antisymmetric_and_normalized():
    for n in (3, 5, 6, 20, 101):
        weights = shapiro_wilk_weights(n)
        np.testing.assert_allclose(weights.numpy(), -weights.flip(0).numpy(), atol=1e-14)
        assert float((weights ** 2).sum()) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(normal_scores(7).numpy(),
                               stats.norm.ppf((np.arange(1, 8) - 0.375) / 7.25), atol=1e-13)


def test_sfrancia_is_the_squared_correlation_with_normal_scores():
    rng = np.random.default_rng(37)
    x = rng.gamma(4.0, size=60)
    result = oe.sfrancia({"x": x}, "x")
    scores = stats.norm.ppf((np.arange(1, 61) - 0.375) / 60.25)
    w = np.corrcoef(np.sort(x), scores)[0, 1] ** 2
    assert result.attrs["statistic"] == pytest.approx(w, rel=1e-12)
    u = math.log(60)
    v = math.log(u)
    z = (math.log(1 - w) - (-1.2725 + 1.0521 * (v - u))) / (1.0308 - 0.26758 * (v + 2 / u))
    assert result.attrs["z"] == pytest.approx(z, rel=1e-9)
    assert result.attrs["p_value"] == pytest.approx(stats.norm.sf(z), rel=1e-9)
    # Normal data are rarely rejected, heavy skew is.
    assert oe.sfrancia({"x": rng.normal(size=200)}, "x").attrs["p_value"] > 0.01
    assert oe.sfrancia({"x": rng.exponential(size=200)}, "x").attrs["p_value"] < 1e-6
    assert oe.sfrancia({"x": x}, "x").attrs["statistic"] > oe.swilk({"x": x}, "x").attrs[
        "statistic"] - 0.05


def test_sktest_matches_scipy_components():
    rng = np.random.default_rng(38)
    x = rng.gamma(4.0, size=75)
    result = oe.sktest({"x": x}, "x")
    attrs = result.attrs
    assert attrs["skewness"] == pytest.approx(stats.skew(x))
    assert attrs["kurtosis"] == pytest.approx(stats.kurtosis(x, fisher=False))
    skew, kurt = stats.skewtest(x), stats.kurtosistest(x)
    assert attrs["z_skewness"] == pytest.approx(skew.statistic, rel=1e-10)
    assert attrs["p_skewness"] == pytest.approx(skew.pvalue, rel=1e-9)
    assert attrs["z_kurtosis"] == pytest.approx(kurt.statistic, rel=1e-10)
    assert attrs["p_kurtosis"] == pytest.approx(kurt.pvalue, rel=1e-9)
    omnibus = stats.normaltest(x)
    assert attrs["chi2_unadjusted"] == pytest.approx(omnibus.statistic, rel=1e-10)
    assert attrs["p_unadjusted"] == pytest.approx(omnibus.pvalue, rel=1e-9)
    jb = stats.jarque_bera(x)
    assert attrs["jarque_bera"] == pytest.approx(jb.statistic, rel=1e-10)
    assert attrs["p_jarque_bera"] == pytest.approx(jb.pvalue, rel=1e-9)
    assert result.index.tolist() == ["skewness", "kurtosis", "joint_adjusted", "joint",
                                     "jarque_bera"]
    assert attrs["statistic"] == pytest.approx(royston_adjusted_chi2(omnibus.statistic, 75))
    # The statistic is invariant to location and scale.
    shifted = oe.sktest({"x": 1e6 + 1e3 * x}, "x").attrs
    assert shifted["chi2_unadjusted"] == pytest.approx(attrs["chi2_unadjusted"], rel=1e-6)


def test_royston_adjustment_reproduces_the_stata_manual_example_and_is_monotone():
    # [R] sktest, auto data (n = 74): mpg has chi2 13.13 unadjusted and 10.95 adjusted
    # (p = 0.0042); trunk has 4.05 and 4.19 (p = 0.1228).
    for unadjusted, adjusted, p in ((13.13, 10.95, 0.0042), (4.05, 4.19, 0.1228)):
        value = royston_adjusted_chi2(unadjusted, 74)
        assert value == pytest.approx(adjusted, abs=0.006)
        assert stats.chi2.sf(value, 2) == pytest.approx(p, abs=6e-5)
    for n in (10, 50, 500):
        grid = np.linspace(0.0, 40.0, 401)
        values = np.array([royston_adjusted_chi2(value, n) for value in grid])
        assert values[0] == 0.0 and np.all(np.diff(values) >= -1e-9)
        assert np.max(np.abs(np.diff(values))) < 0.5        # the three branches meet


# ---- failure contract ----------------------------------------------------------------------


def test_error_codes(sample):
    cases = [
        (lambda: oe.ksmirnov(sample, "x", distribution="gamma"), "invalid_option"),
        (lambda: oe.ksmirnov(sample, "x", params=[1.0]), "invalid_option"),
        (lambda: oe.ksmirnov(sample, "x", params=[0.0, -1.0]), "invalid_option"),
        (lambda: oe.ksmirnov(sample, "x", by="g", params=[0, 1]), "invalid_option"),
        (lambda: oe.ksmirnov(sample, "x", by="k"), "invalid_groups"),
        (lambda: oe.ksmirnov({"x": [2.0, 2.0, 2.0]}, "x"), "no_variation"),
        (lambda: oe.ksmirnov(sample, "x", distribution="poisson"), "invalid_values"),
        (lambda: oe.ksmirnov(sample, "k", distribution="poisson", params=[2.0], exact=True),
         "exact_unavailable"),
        (lambda: oe.runtest({"y": [1.0, 1.0, 1.0]}, "y"), "no_variation"),
        (lambda: oe.runtest(sample, "x", threshold="mode"), "invalid_option"),
        (lambda: oe.runtest(sample, "x", ties="split"), "invalid_option"),
        (lambda: oe.bitest(sample, "g", p=1.0), "invalid_option"),
        (lambda: oe.bitest(sample, "k"), "not_binary"),
        (lambda: oe.bitest(sample, "g", positive=5), "invalid_option"),
        (lambda: oe.prtest(sample, "g", by="g"), "invalid_spec"),
        (lambda: oe.prtest({"y": [1, 1, 1, 1], "g": [0, 0, 1, 1]}, "y", by="g"),
         "no_variation"),
        (lambda: oe.prtest({"y": [1, 0, 1, 0], "g": [0, 0, 1, 1]}, "y", by="g", p=0.3),
         "invalid_option"),
        (lambda: oe.chi2gof({"c": [1, 1, 1]}, "c"), "no_variation"),
        (lambda: oe.chi2gof({"c": [1, 2, 3]}, "c", expected=[1, 2]), "invalid_option"),
        (lambda: oe.chi2gof({"c": [1, 2, 3]}, "c", expected={1: 1, 2: 1}), "invalid_option"),
        (lambda: oe.chi2gof({"c": [1, 2, 3]}, "c", expected=[1, 0, 1]), "invalid_option"),
        (lambda: oe.chi2gof({"c": [1, 2], "w": [1.0, -1.0]}, "c", weights="w"),
         "invalid_weights"),
        (lambda: oe.swilk({"x": [1.0, 2.0]}, "x"), "sample_size"),
        (lambda: oe.swilk({"x": np.arange(5001.0)}, "x"), "sample_size"),
        (lambda: oe.swilk({"x": [1.0, 1.0, 1.0, 1.0]}, "x"), "no_variation"),
        (lambda: oe.sfrancia({"x": [1.0, 2.0, 3.0, 4.0]}, "x"), "sample_size"),
        (lambda: oe.sktest({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}, "x"), "sample_size"),
        (lambda: oe.sktest({"x": ["a"] * 9}, "x"), "non_numeric_column"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code, str(error.value))


def test_missing_values_are_dropped_and_reported(sample):
    frame = sample.copy()
    frame.loc[[1, 2, 3], "x"] = np.nan
    result = oe.swilk(frame, "x")
    assert result.attrs["n"] == 37 and result.attrs["n_dropped"] == 3
    assert result.attrs["statistic"] == pytest.approx(
        stats.shapiro(frame["x"].dropna()).statistic, rel=1e-10)
    with pytest.raises(AnalysisError) as error:
        oe.sktest(frame, "x", missing="raise")
    assert error.value.code == "missing_values"


def test_results_render_and_have_json_safe_attrs(sample):
    results = [oe.ksmirnov(sample, "x"), oe.ksmirnov(sample, "x", by="g"),
               oe.runtest(sample, "x"), oe.bitest(sample, "g"), oe.prtest(sample, "g"),
               oe.chi2gof(sample, "k"), oe.swilk(sample, "x"), oe.sfrancia(sample, "x"),
               oe.sktest(sample, "x")]
    for result in results:
        attrs = json.loads(json.dumps(result.attrs))
        assert attrs == result.attrs
        assert "\\begin{tabular}" in str(result.to_latex())
        assert "p_value" in str(result)
        for value in attrs.values():
            assert not (isinstance(value, float) and not math.isfinite(value))
