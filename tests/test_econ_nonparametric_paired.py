"""Nonparametric family: related-sample tests against SciPy, statsmodels and enumeration."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable, cochrans_q
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.nonparametric import exact as exact_kernels


@pytest.fixture(scope="module")
def pairs():
    rng = np.random.default_rng(21)
    a = rng.normal(size=15)
    b = a + rng.normal(size=15) + 0.5
    return pd.DataFrame({"a": a, "b": b}), a, b


@pytest.fixture(scope="module")
def tied_pairs():
    rng = np.random.default_rng(22)
    a = np.round(rng.normal(size=14) * 2)
    b = np.round(a + rng.normal(size=14) + 0.4)
    return pd.DataFrame({"a": a, "b": b}), a, b


# ---- signrank ----------------------------------------------------------------------------


def test_signrank_matches_scipy_without_ties(pairs):
    data, a, b = pairs
    result = oe.signrank(data, "b", "a")
    attrs = result.attrs
    exact = stats.wilcoxon(b, a, method="exact")
    normal = stats.wilcoxon(b, a, method="asymptotic", correction=False)
    assert attrs["statistic"] == pytest.approx(exact.statistic)
    assert attrs["exact"] is True and attrs["p_exact"] == pytest.approx(exact.pvalue, rel=1e-12)
    assert attrs["p_value"] == pytest.approx(normal.pvalue, rel=1e-10)
    assert attrs["t_plus"] + attrs["t_minus"] == pytest.approx(15 * 16 / 2)
    assert attrs["expected"] == pytest.approx(15 * 16 / 4)
    assert attrs["variance"] == pytest.approx(15 * 16 * 31 / 24)
    greater = stats.wilcoxon(b, a, method="exact", alternative="greater").pvalue
    assert attrs["p_exact_upper"] == pytest.approx(greater, rel=1e-12)
    assert attrs["effect_r"] == pytest.approx(attrs["z"] / math.sqrt(15))
    assert result["ranks"].loc["positive", "n"] == (b > a).sum()


@pytest.mark.parametrize("zero_method, scipy_name", [("drop", "wilcox"), ("adjust", "pratt")])
def test_signrank_zero_handling_and_ties_match_scipy(tied_pairs, zero_method, scipy_name):
    data, a, b = tied_pairs
    assert (a == b).any()
    attrs = oe.signrank(data, "b", "a", zero_method=zero_method).attrs
    reference = stats.wilcoxon(b, a, zero_method=scipy_name, method="asymptotic",
                               correction=False)
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert attrs["statistic"] == pytest.approx(reference.statistic)
    assert attrs["ties"] is True and attrs["exact"] is False and "p_exact" not in attrs
    assert attrs["n_zero"] == int((a == b).sum())


@pytest.mark.parametrize("zero_method", ["drop", "adjust"])
def test_signrank_forced_exact_with_ties_is_the_sign_flip_distribution(tied_pairs, zero_method):
    data, a, b = tied_pairs
    d = b - a
    attrs = oe.signrank(data, "b", "a", zero_method=zero_method, exact=True).attrs
    if zero_method == "drop":
        d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))[d != 0]
    signs = np.sign(d[d != 0])
    observed = ranks[signs > 0].sum()
    flips = np.array(list(itertools.product([0, 1], repeat=len(ranks))))
    sums = flips @ ranks
    centre = ranks.sum() / 2
    assert attrs["p_exact"] == pytest.approx(
        np.mean(np.abs(sums - centre) >= abs(observed - centre) - 1e-9), rel=1e-10)
    assert attrs["p_exact_upper"] == pytest.approx(np.mean(sums >= observed - 1e-9), rel=1e-10)


def test_signrank_one_sample_with_mu_and_default_exact_rule():
    rng = np.random.default_rng(23)
    x = rng.normal(loc=1.0, size=40)
    attrs = oe.signrank({"x": x}, "x", mu=0.5).attrs
    reference = stats.wilcoxon(x - 0.5, method="asymptotic", correction=False)
    assert attrs["exact"] is False                          # more than 25 differences
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    forced = oe.signrank({"x": x}, "x", mu=0.5, exact=True).attrs
    assert forced["p_exact"] == pytest.approx(stats.wilcoxon(x - 0.5, method="exact").pvalue,
                                              rel=1e-10)


def test_signed_sum_distribution_matches_enumeration():
    scores = [1, 2, 2, 5, 7, 7, 9]
    pmf = exact_kernels.signed_sum_distribution(torch.tensor(scores)).numpy()
    sums = np.array(list(itertools.product([0, 1], repeat=7))) @ np.array(scores)
    np.testing.assert_allclose(pmf, np.bincount(sums, minlength=len(pmf)) / 2 ** 7, atol=1e-15)


# ---- sign test -----------------------------------------------------------------------------


def test_signtest_is_the_exact_binomial(tied_pairs):
    data, a, b = tied_pairs
    result = oe.signtest(data, "b", "a")
    d = b - a
    plus, minus = int((d > 0).sum()), int((d < 0).sum())
    attrs = result.attrs
    assert (attrs["n_positive"], attrs["n_negative"], attrs["n_zero"]) == \
        (plus, minus, int((d == 0).sum()))
    assert attrs["p_value"] == pytest.approx(stats.binomtest(plus, plus + minus, 0.5).pvalue,
                                             rel=1e-10)
    assert attrs["p_positive"] == pytest.approx(
        stats.binomtest(plus, plus + minus, 0.5, alternative="greater").pvalue, rel=1e-10)
    assert attrs["p_negative"] == pytest.approx(
        stats.binomtest(plus, plus + minus, 0.5, alternative="less").pvalue, rel=1e-10)
    z = (abs(plus - minus) - 1) / math.sqrt(plus + minus) * np.sign(plus - minus)
    assert attrs["z"] == pytest.approx(z)
    assert attrs["p_value_normal"] == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
    shifted = oe.signtest(data, "b", "a", mu=10.0).attrs
    assert shifted["n_positive"] == 0 and shifted["p_negative"] == pytest.approx(
        0.5 ** shifted["n"])


# ---- McNemar and symmetry ------------------------------------------------------------------


def test_mcnemar_matches_statsmodels():
    rng = np.random.default_rng(24)
    x = rng.integers(0, 2, size=80)
    y = np.where(rng.random(80) < 0.7, x, 1 - x)
    y[:25] = 1
    result = oe.mcnemar({"x": x, "y": y}, "x", "y")
    table = pd.crosstab(x, y).to_numpy()
    attrs = result.attrs
    assert (attrs["n12"], attrs["n21"]) == (table[0, 1], table[1, 0])
    assert abs(table[0, 1] - table[1, 0]) > 1
    plain = sm_mcnemar(table, exact=False, correction=False)
    corrected = sm_mcnemar(table, exact=False, correction=True)
    assert attrs["statistic"] == pytest.approx(plain.statistic)
    assert attrs["p_value"] == pytest.approx(plain.pvalue, rel=1e-10)
    assert attrs["statistic_continuity"] == pytest.approx(corrected.statistic)
    assert attrs["p_value_continuity"] == pytest.approx(corrected.pvalue, rel=1e-10)
    assert attrs["p_exact"] == pytest.approx(sm_mcnemar(table, exact=True).pvalue, rel=1e-10)
    assert result["counts"].loc["total", "total"] == 80
    assert result["counts"].iloc[:2, :2].to_numpy().tolist() == table.tolist()
    # Equal discordant counts: the corrected statistic is 0 (R's rule), never 1 / (n12 + n21).
    balanced = oe.mcnemar({"a": [0, 0, 1, 1, 0, 1], "b": [1, 1, 0, 0, 0, 1]}, "a", "b").attrs
    assert balanced["statistic"] == 0.0 and balanced["statistic_continuity"] == 0.0
    assert balanced["p_exact"] == pytest.approx(1.0)


def test_mcnemar_without_discordant_pairs_and_text_levels():
    result = oe.mcnemar({"a": ["y", "n", "y", "n"], "b": ["y", "n", "y", "n"]}, "a", "b")
    assert result.attrs["statistic"] is None and result.attrs["p_exact"] == 1.0
    assert result.attrs["levels"] == ["n", "y"] and result.attrs["notes"]
    assert json.loads(json.dumps(result.attrs)) == result.attrs


def test_symmetry_matches_statsmodels_and_reduces_to_mcnemar():
    rng = np.random.default_rng(25)
    x = rng.integers(1, 5, size=300)
    y = np.clip(x + rng.integers(-3, 4, size=300), 1, 4)
    result = oe.symmetry({"x": x, "y": y}, "x", "y")
    table = pd.crosstab(x, y).to_numpy()
    square = SquareTable(table, shift_zeros=False)
    bowker = square.symmetry()
    assert result.attrs["statistic"] == pytest.approx(bowker.statistic, rel=1e-12)
    assert result.attrs["df"] == 6 == bowker.df                 # every pair is non-empty
    assert result.attrs["p_value"] == pytest.approx(bowker.pvalue, rel=1e-10)
    homogeneity = square.homogeneity()
    assert result.attrs["mh_statistic"] == pytest.approx(homogeneity.statistic, rel=1e-10)
    assert result.attrs["mh_df"] == 3
    assert result.attrs["mh_p_value"] == pytest.approx(homogeneity.pvalue, rel=1e-9)
    d = x - y
    assert result.attrs["mh_z"] == pytest.approx(d.sum() / math.sqrt((d ** 2).sum()))
    # K = 2: Bowker, Stuart-Maxwell and the squared ordinal z are McNemar's chi2.
    u = rng.integers(0, 2, size=60)
    v = np.where(rng.random(60) < 0.6, u, 1 - u)
    two = oe.symmetry({"u": u, "v": v}, "u", "v").attrs
    plain = oe.mcnemar({"u": u, "v": v}, "u", "v").attrs["statistic"]
    assert two["statistic"] == pytest.approx(plain) and two["df"] == 1
    assert two["mh_statistic"] == pytest.approx(plain)
    assert two["mh_z"] ** 2 == pytest.approx(plain)


def test_symmetry_degrees_of_freedom_skip_empty_pairs_and_diagonal_tables():
    data = {"a": [1, 1, 2, 2, 3, 3, 1, 2], "b": [1, 2, 2, 1, 3, 3, 1, 2]}
    result = oe.symmetry(data, "a", "b")
    assert result.attrs["df"] == 1                          # only the (1, 2) pair is observed
    assert result.attrs["mh_df"] == 1
    assert "marginal_homogeneity_ordinal" in result["tests"].index
    diagonal = oe.symmetry({"a": ["x", "y", "x"], "b": ["x", "y", "x"]}, "a", "b")
    assert diagonal.attrs["df"] == 0 and diagonal.attrs["p_value"] is None
    assert diagonal.attrs["notes"]


# ---- Friedman and Cochran ------------------------------------------------------------------


def test_friedman_matches_scipy_with_ties_and_kendall_w():
    rng = np.random.default_rng(26)
    wide = np.round(rng.normal(size=(14, 4)) + np.arange(4) * 0.5)
    frame = pd.DataFrame(wide, columns=list("abcd"))
    result = oe.friedman(frame, list("abcd"), pairwise=True)
    reference = stats.friedmanchisquare(*wide.T)
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert result.attrs["kendall_w"] == pytest.approx(reference.statistic / (14 * 3))
    ranks = stats.rankdata(wide, axis=1)
    np.testing.assert_allclose(result["ranks"]["mean_rank"], ranks.mean(0))
    se = math.sqrt(4 * 5 / (6 * 14))
    first = result["pairwise"].iloc[0]
    assert first["z"] == pytest.approx((ranks[:, 0].mean() - ranks[:, 1].mean()) / se)
    assert first["p_adjusted"] == pytest.approx(min(1.0, 6 * first["p_value"]))
    assert len(result["pairwise"]) == 6


def test_friedman_listwise_deletion_and_two_columns_equal_sign_test():
    rng = np.random.default_rng(27)
    frame = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    frame.loc[[2, 9], "b"] = np.nan
    result = oe.friedman(frame, ["a", "b", "c"])
    assert result.attrs["n"] == 18 and result.attrs["n_dropped"] == 2
    complete = frame.dropna().to_numpy()
    assert result.attrs["statistic"] == pytest.approx(
        stats.friedmanchisquare(*complete.T).statistic, rel=1e-12)
    two = oe.friedman(frame, ["a", "c"]).attrs
    plus = int((frame.a > frame.c).sum())
    assert two["statistic"] == pytest.approx((2 * plus - 20) ** 2 / 20)


def test_cochran_q_matches_statsmodels_and_mcnemar():
    rng = np.random.default_rng(28)
    binary = (rng.random((40, 3)) < np.array([0.3, 0.5, 0.7])).astype(int)
    frame = pd.DataFrame(binary, columns=list("abc"))
    result = oe.cochran_q(frame, list("abc"))
    reference = cochrans_q(binary)
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert result["frequencies"]["success"].tolist() == binary.sum(0).tolist()
    two = oe.cochran_q(frame, ["a", "b"]).attrs["statistic"]
    assert two == pytest.approx(oe.mcnemar(frame, "a", "b").attrs["statistic"])
    labels = frame.replace({0: "no", 1: "yes"})
    assert oe.cochran_q(labels, list("abc")).attrs["statistic"] == pytest.approx(
        reference.statistic)
    flipped = oe.cochran_q(labels, list("abc"), positive="no")
    assert flipped.attrs["statistic"] == pytest.approx(reference.statistic)
    assert flipped["frequencies"]["success"].tolist() == (40 - binary.sum(0)).tolist()


# ---- failure contract and rendering --------------------------------------------------------


def test_error_codes():
    data = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [3, 1, 2], "s": list("xyz")}
    cases = [
        (lambda: oe.signrank(data, "a", "b"), "no_variation"),
        (lambda: oe.signtest(data, "a", "b"), "no_variation"),
        (lambda: oe.signrank(data, "a", "s"), "non_numeric_column"),
        (lambda: oe.signrank(data, "a", mu="zero"), "invalid_option"),
        (lambda: oe.signrank(data, "a", "c", zero_method="split"), "invalid_option"),
        (lambda: oe.signrank(data, "a", 3), "invalid_spec"),
        (lambda: oe.mcnemar(data, "a", "c"), "not_binary"),
        (lambda: oe.mcnemar(data, "a", "nope"), "missing_columns"),
        (lambda: oe.symmetry({"a": [1, 1], "b": [1, 1]}, "a", "b"), "no_variation"),
        (lambda: oe.friedman(data, "a"), "invalid_spec"),
        (lambda: oe.friedman(data, ["a"]), "invalid_spec"),
        (lambda: oe.friedman(data, ["a", "a"]), "invalid_spec"),
        (lambda: oe.friedman({"a": [1, 1], "b": [1, 1]}, ["a", "b"]), "no_variation"),
        (lambda: oe.friedman(data, ["a", "s"]), "non_numeric_column"),
        (lambda: oe.cochran_q(data, ["a", "c"]), "not_binary"),
        (lambda: oe.cochran_q({"a": [1, 1], "b": [1, 1]}, ["a", "b"]), "no_variation"),
        (lambda: oe.cochran_q({"a": [0, 1], "b": [1, 1]}, ["a", "b"], positive=7),
         "invalid_option"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code, str(error.value))


def test_missing_policy_is_listwise_over_the_pair(pairs):
    data, a, b = pairs
    frame = data.copy()
    frame.loc[0, "a"] = np.nan
    result = oe.signrank(frame, "b", "a")
    assert result.attrs["n"] == 14 and result.attrs["n_dropped"] == 1
    assert result.attrs["p_exact"] == pytest.approx(
        stats.wilcoxon(b[1:], a[1:], method="exact").pvalue, rel=1e-12)
    with pytest.raises(AnalysisError) as error:
        oe.signtest(frame, "b", "a", missing="raise")
    assert error.value.code == "missing_values"


def test_results_render_and_have_json_safe_attrs(pairs, tied_pairs):
    wide = pd.DataFrame({"a": [1, 2, 1, 3, 2], "b": [2, 3, 3, 4, 3], "c": [3, 1, 2, 5, 4]})
    binary = pd.DataFrame({"t1": [1, 1, 0, 1, 0, 1], "t2": [1, 0, 0, 1, 0, 0],
                           "t3": [0, 0, 0, 1, 0, 0]})
    results = [oe.signrank(pairs[0], "b", "a"), oe.signtest(tied_pairs[0], "b", "a"),
               oe.mcnemar(binary, "t1", "t2"), oe.symmetry(wide, "a", "b"),
               oe.friedman(wide, ["a", "b", "c"], pairwise=True),
               oe.cochran_q(binary, ["t1", "t2", "t3"])]
    for result in results:
        assert isinstance(result, TableSet)
        assert json.loads(json.dumps(result.attrs)) == result.attrs
        assert "[tests]" in str(result) and "\\begin{tabular}" in result.to_latex()
