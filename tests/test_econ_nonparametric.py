"""Nonparametric family: independent-sample rank tests, public API and failure contract.

Oracles: scipy.stats (mannwhitneyu, kruskal, median_test, kendalltau) and explicit
NumPy / itertools enumeration. SciPy is used only here, never by the package.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import TableSet
from openecon.econometrics.nonparametric import EXPORTS, ESTIMATORS
from openecon.econometrics.nonparametric import exact as exact_kernels
from openecon.econometrics.nonparametric.common import adjust_p_values, midranks, row_midranks
from openecon.econometrics.nonparametric.independent import ordered_pair_counts

ALL_NAMES = (
    "ranksum", "kwallis", "median_test", "jonckheere", "signrank", "signtest", "mcnemar",
    "symmetry", "friedman", "cochran_q", "ksmirnov", "runtest", "bitest", "prtest", "chi2gof",
    "swilk", "sfrancia", "sktest", "crosstab", "tabulate", "roc", "roccomp",
)


@pytest.fixture(scope="module")
def two_groups():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=9), rng.normal(size=12) + 0.8
    return pd.DataFrame({"v": np.r_[x, y], "g": ["a"] * 9 + ["b"] * 12}), x, y


@pytest.fixture(scope="module")
def four_groups():
    rng = np.random.default_rng(12)
    g = rng.integers(0, 4, size=70)
    v = np.round(rng.normal(size=70) + 0.4 * g, 1)          # rounded: many ties
    return pd.DataFrame({"v": v, "g": g}), v, g


# ---- manifest and exports -------------------------------------------------------------


def test_manifest_is_torch_free_and_exports_resolve():
    assert ESTIMATORS == ()
    assert set(EXPORTS) == set(ALL_NAMES)
    exports = registry.public_exports()
    for name in ALL_NAMES:
        assert exports[name] == tuple(EXPORTS[name].split(":"))
        function = getattr(oe, name)
        assert callable(function) and function.__doc__ and "Example" in function.__doc__
        assert name in dir(oe)


def test_manifest_import_does_not_load_the_tensor_runtime():
    code = ("import sys; import openecon.econometrics.nonparametric as m; "
            "assert len(m.EXPORTS) == 22; "
            "assert 'torch' not in sys.modules and 'pandas' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ranking_kernels_match_scipy():
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(size=200), 1)
    ranks, sizes = midranks(torch.as_tensor(x))
    np.testing.assert_allclose(ranks.numpy(), stats.rankdata(x), rtol=0, atol=1e-12)
    assert int(sizes.sum()) == 200
    wide = np.round(rng.normal(size=(40, 5)), 0)
    ranks, ties = row_midranks(torch.as_tensor(wide))
    np.testing.assert_allclose(ranks.numpy(), stats.rankdata(wide, axis=1), atol=1e-12)
    expected = [sum(c ** 3 - c for c in np.unique(row, return_counts=True)[1]) for row in wide]
    np.testing.assert_allclose(ties.numpy(), expected)


def test_p_value_adjustments():
    p = [0.01, 0.04, 0.03, 0.20]
    assert adjust_p_values(p, "none") == p
    np.testing.assert_allclose(adjust_p_values(p, "bonferroni"), [0.04, 0.16, 0.12, 0.80])
    np.testing.assert_allclose(adjust_p_values(p, "holm"), [0.04, 0.09, 0.09, 0.20])


# ---- ranksum ---------------------------------------------------------------------------


def test_ranksum_matches_scipy_asymptotic_and_exact(two_groups):
    data, x, y = two_groups
    result = oe.ranksum(data, "v", "g")
    assert isinstance(result, TableSet) and list(result) == ["ranks", "tests"]
    attrs = result.attrs
    plain = stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=False)
    corrected = stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=True)
    exact = stats.mannwhitneyu(x, y, method="exact")
    assert attrs["u1"] == pytest.approx(plain.statistic)
    assert attrs["u1"] + attrs["u2"] == pytest.approx(9 * 12)
    assert attrs["p_value"] == pytest.approx(plain.pvalue, rel=1e-10)
    assert attrs["p_value_continuity"] == pytest.approx(corrected.pvalue, rel=1e-10)
    assert attrs["exact"] is True and attrs["p_exact"] == pytest.approx(exact.pvalue, rel=1e-10)
    less = stats.mannwhitneyu(x, y, method="exact", alternative="less").pvalue
    greater = stats.mannwhitneyu(x, y, method="exact", alternative="greater").pvalue
    assert attrs["p_exact_lower"] == pytest.approx(less, rel=1e-10)
    assert attrs["p_exact_upper"] == pytest.approx(greater, rel=1e-10)
    ranks = stats.rankdata(np.r_[x, y])
    assert result["ranks"]["rank_sum"].tolist() == pytest.approx([ranks[:9].sum(),
                                                                  ranks[9:].sum()])
    assert attrs["porder"] == pytest.approx(plain.statistic / (9 * 12))
    assert attrs["effect_r"] == pytest.approx(attrs["z"] / math.sqrt(21))
    assert attrs["w"] == pytest.approx(ranks[:9].sum())          # smaller group


def test_ranksum_hodges_lehmann_matches_brute_force(two_groups):
    data, x, y = two_groups
    attrs = oe.ranksum(data, "v", "g", alpha=0.10).attrs
    differences = np.sort((x[:, None] - y[None, :]).ravel())
    assert attrs["hl_estimate"] == pytest.approx(np.median(differences))
    # R's wilcox.test rule: q = qwilcox(alpha/2); CI = (D_(q), D_(mn + 1 - q)), with the
    # null distribution of U enumerated over all C(21, 9) rank subsets.
    subsets = np.array(list(itertools.combinations(range(1, 22), 9)))
    u = subsets.sum(1) - 45
    cdf = np.cumsum(np.bincount(u, minlength=9 * 12 + 1)) / len(u)
    q = max(int(np.searchsorted(cdf, 0.05)), 1)
    assert attrs["hl_ci_low"] == pytest.approx(differences[q - 1])
    assert attrs["hl_ci_high"] == pytest.approx(differences[len(differences) - q])
    assert attrs["hl_ci_low"] < attrs["hl_estimate"] < attrs["hl_ci_high"]
    assert attrs["hl_ci_method"] == "exact"


def test_rank_sum_distribution_matches_enumeration():
    scores = [1, 2, 2, 5, 7, 7, 9, 12]
    pmf = exact_kernels.rank_sum_distribution(torch.tensor(scores), 3).numpy()
    sums = [sum(combo) for combo in itertools.combinations(scores, 3)]
    expected = np.bincount(sums, minlength=len(pmf)) / len(sums)
    np.testing.assert_allclose(pmf, expected, atol=1e-15)


def test_ranksum_with_ties_uses_tie_corrected_variance_and_conditional_exact():
    rng = np.random.default_rng(3)
    x, y = np.round(rng.normal(size=7)), np.round(rng.normal(size=6) + 1)
    data = {"v": np.r_[x, y], "g": [0] * 7 + [1] * 6}
    default = oe.ranksum(data, "v", "g")
    reference = stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=False)
    assert default.attrs["ties"] is True and default.attrs["exact"] is False
    assert "p_exact" not in default.attrs
    assert default.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert default.attrs["variance"] < default.attrs["variance_unadjusted"]
    forced = oe.ranksum(data, "v", "g", exact=True).attrs
    ranks = stats.rankdata(np.r_[x, y])
    observed, centre = ranks[:7].sum(), 7 * 14 / 2
    sums = np.array([ranks[list(combo)].sum() for combo in itertools.combinations(range(13), 7)])
    assert forced["p_exact"] == pytest.approx(
        np.mean(np.abs(sums - centre) >= abs(observed - centre) - 1e-9), rel=1e-10)
    assert forced["p_exact_lower"] == pytest.approx(np.mean(sums <= observed + 1e-9), rel=1e-10)
    assert forced["hl_ci_method"] == "normal"


def test_ranksum_switches_to_asymptotic_for_larger_samples_and_handles_missing():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"v": rng.normal(size=60), "g": rng.integers(0, 2, size=60)})
    frame.loc[[3, 17], "v"] = np.nan
    frame["g"] = frame["g"].astype(object)
    frame.loc[5, "g"] = None
    result = oe.ranksum(frame, "v", "g")
    assert result.attrs["n"] == 57 and result.attrs["n_dropped"] == 3
    assert result.attrs["exact"] is False
    complete = frame.dropna()
    reference = stats.mannwhitneyu(complete.v[complete.g == 0], complete.v[complete.g == 1],
                                   method="asymptotic", use_continuity=False)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    with pytest.raises(AnalysisError) as error:
        oe.ranksum(frame, "v", "g", missing="raise")
    assert error.value.code == "missing_values"


# ---- kwallis ---------------------------------------------------------------------------


def test_kwallis_matches_scipy_and_dunn_formula(four_groups):
    data, v, g = four_groups
    result = oe.kwallis(data, "v", "g", pairwise=True, adjust="holm")
    reference = stats.kruskal(*[v[g == j] for j in range(4)])
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert attrs["statistic_unadjusted"] < attrs["statistic"]
    n = len(v)
    assert attrs["epsilon_squared"] == pytest.approx(reference.statistic / (n - 1))
    assert attrs["eta_squared"] == pytest.approx((reference.statistic - 3) / (n - 4))
    ranks = stats.rankdata(v)
    counts = np.unique(v, return_counts=True)[1]
    sigma2 = n * (n + 1) / 12 - (counts ** 3 - counts).sum() / (12 * (n - 1))
    pairwise = result["pairwise"]
    raw = []
    for (i, j), (_, row) in zip(itertools.combinations(range(4), 2), pairwise.iterrows()):
        difference = ranks[g == i].mean() - ranks[g == j].mean()
        se = math.sqrt(sigma2 * (1 / (g == i).sum() + 1 / (g == j).sum()))
        assert row["z"] == pytest.approx(difference / se, rel=1e-12)
        assert row["p_value"] == pytest.approx(2 * stats.norm.sf(abs(difference / se)), rel=1e-9)
        raw.append(row["p_value"])
    np.testing.assert_allclose(pairwise["p_adjusted"], adjust_p_values(raw, "holm"))
    assert list(result) == ["ranks", "tests", "pairwise"]
    assert "pairwise" not in oe.kwallis(data, "v", "g")


# ---- median test -------------------------------------------------------------------------


@pytest.mark.parametrize("ties", ["below", "above", "drop"])
def test_median_test_matches_scipy(four_groups, ties):
    data, v, g = four_groups
    scipy_ties = {"below": "below", "above": "above", "drop": "ignore"}[ties]
    reference = stats.median_test(*[v[g == j] for j in range(4)], ties=scipy_ties)
    result = oe.median_test(data, "v", "g", ties=ties)
    assert result.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-12)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert result.attrs["median"] == pytest.approx(reference.median)
    assert result["counts"]["above"].tolist() == reference.table[0].tolist()


def test_median_test_two_groups_adds_continuity_and_fisher(two_groups):
    data, x, y = two_groups
    result = oe.median_test(data, "v", "g")
    corrected = stats.median_test(x, y, correction=True)
    assert result.attrs["statistic_continuity"] == pytest.approx(corrected.statistic, rel=1e-12)
    assert result.attrs["p_exact"] == pytest.approx(stats.fisher_exact(corrected.table).pvalue,
                                                    rel=1e-10)
    assert result["tests"].index.tolist() == ["pearson", "continuity", "fisher_exact"]


# ---- Jonckheere-Terpstra -----------------------------------------------------------------


def test_jonckheere_matches_brute_force_and_kendall(four_groups):
    data, v, g = four_groups
    result = oe.jonckheere(data, "v", "g")
    statistic = sum(((v[g == j][None, :] > v[g == i][:, None])
                     + 0.5 * (v[g == j][None, :] == v[g == i][:, None])).sum()
                    for i, j in itertools.combinations(range(4), 2))
    assert result.attrs["statistic"] == pytest.approx(statistic)
    sizes = np.bincount(g)
    assert result.attrs["mean"] == pytest.approx((len(v) ** 2 - (sizes ** 2).sum()) / 4)
    # J - E(J) = S / 2 and Var(J) = Var(S) / 4: the z of Kendall's tau-b test.
    kendall = stats.kendalltau(g, v, method="asymptotic")
    assert result.attrs["p_value"] == pytest.approx(kendall.pvalue, rel=1e-9)
    assert result.attrs["z"] == pytest.approx(stats.norm.isf(kendall.pvalue / 2), rel=1e-8)
    assert result.attrs["p_increasing"] == pytest.approx(kendall.pvalue / 2, rel=1e-9)
    reverse = oe.jonckheere(data, "v", "g", order=[3, 2, 1, 0])
    assert reverse.attrs["z"] == pytest.approx(-result.attrs["z"])
    assert reverse["groups"].index.tolist() == ["3", "2", "1", "0"]


def test_ordered_pair_counts_blocks_agree_with_direct_count():
    rng = np.random.default_rng(5)
    groups = torch.as_tensor(rng.integers(0, 3, size=300))
    values = torch.as_tensor(np.round(rng.normal(size=300), 1))
    concordant, discordant = ordered_pair_counts(groups, values, 3)
    g, v = groups.numpy(), values.numpy()
    lower = g[:, None] < g[None, :]
    assert concordant == (lower & (v[:, None] < v[None, :])).sum()
    assert discordant == (lower & (v[:, None] > v[None, :])).sum()


# ---- failure contract --------------------------------------------------------------------


def test_error_codes():
    data = {"v": [1.0, 2.0, 3.0, 4.0], "g": [0, 0, 1, 1], "s": list("abcd"), "k": [0, 1, 2, 2]}
    cases = [
        (lambda: oe.ranksum(data, "v", "missing"), "missing_columns"),
        (lambda: oe.ranksum(data, "s", "g"), "non_numeric_column"),
        (lambda: oe.ranksum(data, "v", "k"), "invalid_groups"),
        (lambda: oe.ranksum(data, "v", "v"), "invalid_spec"),
        (lambda: oe.ranksum(data, ["v"], "g"), "invalid_spec"),
        (lambda: oe.ranksum(data, "v", "g", exact="yes"), "invalid_option"),
        (lambda: oe.ranksum(data, "v", "g", alpha=1.5), "invalid_option"),
        (lambda: oe.ranksum(data, "v", "g", missing="keep"), "invalid_option"),
        (lambda: oe.ranksum({"v": [1.0, 1.0, 1.0], "g": [0, 1, 1]}, "v", "g"), "no_variation"),
        (lambda: oe.ranksum({"v": [], "g": []}, "v", "g"), "empty_data"),
        (lambda: oe.ranksum({"v": [None, None], "g": [0, 1]}, "v", "g"), "empty_sample"),
        (lambda: oe.ranksum("not data", "v", "g"), "invalid_data"),
        (lambda: oe.kwallis({"v": [1.0, 2.0], "g": [1, 1]}, "v", "g"), "invalid_groups"),
        (lambda: oe.kwallis({"v": [1.0, 1.0], "g": [0, 1]}, "v", "g"), "no_variation"),
        (lambda: oe.kwallis(data, "v", "g", adjust="sidak"), "invalid_option"),
        (lambda: oe.median_test(data, "v", "g", ties="split"), "invalid_option"),
        (lambda: oe.median_test({"v": [1.0, 1.0, 1.0], "g": [0, 1, 1]}, "v", "g"),
         "no_variation"),
        (lambda: oe.jonckheere(data, "v", "g", order=[0]), "invalid_option"),
        (lambda: oe.jonckheere({"v": [1.0, 2.0], "g": [0, 1]}, "v", "g"), "too_few_observations"),
        (lambda: oe.jonckheere({"v": [1.0, 1.0, 1.0], "g": [0, 1, 1]}, "v", "g"), "no_variation"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code, str(error.value))
        assert str(error.value)


def test_exact_request_beyond_the_enumeration_bound_is_refused():
    rng = np.random.default_rng(6)
    data = {"v": rng.normal(size=1400), "g": np.repeat([0, 1], 700)}
    with pytest.raises(AnalysisError) as error:
        oe.ranksum(data, "v", "g", exact=True)
    assert error.value.code == "exact_unavailable" and "exact=False" in str(error.value)


def test_non_finite_and_duplicate_columns_are_refused():
    with pytest.raises(AnalysisError) as error:
        oe.ranksum({"v": [1.0, float("inf"), 2.0], "g": [0, 1, 1]}, "v", "g")
    assert error.value.code == "non_finite_values"
    frame = pd.DataFrame([[1.0, 2.0, 0]], columns=["v", "v", "g"])
    with pytest.raises(AnalysisError) as error:
        oe.ranksum(frame, "v", "g")
    assert error.value.code == "duplicate_columns"


# ---- inputs, rendering, JSON safety --------------------------------------------------------


def test_inputs_as_records_mapping_and_categorical_order():
    records = [{"v": float(i), "g": "low" if i < 5 else "high"} for i in range(10)]
    by_records = oe.ranksum(records, "v", "g")
    assert by_records.attrs["groups"] == ["high", "low"]            # sorted labels
    frame = pd.DataFrame(records)
    frame["g"] = pd.Categorical(frame["g"], categories=["low", "high", "unused"])
    ordered = oe.ranksum(frame, "v", "g")
    assert ordered.attrs["groups"] == ["low", "high"]               # declared order
    assert ordered.attrs["z"] == pytest.approx(-by_records.attrs["z"])
    assert ordered.attrs["p_exact"] == pytest.approx(by_records.attrs["p_exact"])


def test_results_are_tables_with_json_safe_attrs_and_latex(two_groups, four_groups):
    results = [oe.ranksum(two_groups[0], "v", "g"),
               oe.kwallis(four_groups[0], "v", "g", pairwise=True),
               oe.median_test(two_groups[0], "v", "g"), oe.jonckheere(four_groups[0], "v", "g")]
    for result in results:
        assert isinstance(result, TableSet) and result.title
        attrs = json.loads(json.dumps(result.attrs))
        assert attrs == result.attrs and 0.0 <= attrs["p_value"] <= 1.0
        text = str(result)
        assert "[tests]" in text and "p_value" in text
        assert "\\begin{tabular}" in result.to_latex()
        for frame in result.values():
            assert type(frame).__module__ == "openecon.frame"
            round_trip = pd.read_json(io.StringIO(frame.to_json()))
            assert round_trip.shape == frame.shape


def test_large_sample_runs_in_linear_time():
    rng = np.random.default_rng(7)
    n = 200_000
    data = pd.DataFrame({"v": rng.normal(size=n), "g": rng.integers(0, 2, size=n),
                         "k": rng.integers(0, 5, size=n)})
    result = oe.ranksum(data, "v", "g")
    reference = stats.mannwhitneyu(data.v[data.g == 0], data.v[data.g == 1],
                                   method="asymptotic", use_continuity=False)
    assert result.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-8)
    assert "hl_estimate" not in result.attrs                 # more than 1e6 pairs
    k = oe.kwallis(data, "v", "k")
    assert k.attrs["statistic"] == pytest.approx(
        stats.kruskal(*[data.v[data.k == j] for j in range(5)]).statistic, rel=1e-9)
    assert math.isfinite(oe.jonckheere(data, "v", "k").attrs["z"])
