"""Nonparametric family: crosstab and tabulate.

Oracles: scipy.stats (chi2_contingency, fisher_exact, kendalltau, somersd, spearmanr,
pearsonr, contingency.association), statsmodels (cohens_kappa, Table2x2,
StratifiedTable), brute-force enumeration, and the numerical delta method for every
asymptotic standard error.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
from statsmodels.stats.inter_rater import cohens_kappa

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.nonparametric import measures as measure_kernels
from openecon.econometrics.nonparametric import tables as table_kernels

# A table without zero cells and without ties among row / column maxima, so that every
# measure (lambda included) is differentiable at it.
SMOOTH = np.array([[21.0, 9.0, 6.0, 2.0], [8.0, 17.0, 11.0, 5.0], [3.0, 7.0, 14.0, 19.0]])


def expand(table: np.ndarray, rows=None, columns=None) -> pd.DataFrame:
    """One observation per unit count of a table."""
    r, c = table.shape
    rows = list(range(1, r + 1)) if rows is None else rows
    columns = list(range(1, c + 1)) if columns is None else columns
    counts = table.astype(int).ravel()
    return pd.DataFrame({"r": np.repeat(np.repeat(rows, c), counts),
                         "c": np.repeat(np.tile(columns, r), counts)})


def all_measures(counts: torch.Tensor) -> dict:
    chi2, _ = table_kernels.pearson_chi2(counts)
    scores_r = torch.arange(1.0, counts.shape[0] + 1, dtype=torch.float64)
    scores_c = torch.arange(1.0, counts.shape[1] + 1, dtype=torch.float64)
    out = {}
    out.update(measure_kernels.nominal_measures(counts, chi2,
                                                table_kernels.likelihood_ratio(counts)))
    out.update(measure_kernels.ordinal_measures(counts))
    out.update(measure_kernels.correlations(counts, scores_r, scores_c))
    return out


# ---- tables --------------------------------------------------------------------------------


def test_counts_percentages_expected_and_residuals():
    data = expand(SMOOTH)
    result = oe.crosstab(data, "r", "c", expected=True, percentages="all", residuals=True)
    assert isinstance(result, TableSet)
    assert list(result) == ["counts", "expected", "row_percent", "column_percent",
                            "total_percent", "residuals", "adjusted_residuals", "tests",
                            "measures"]
    counts = result["counts"]
    assert counts.index.tolist() == ["1", "2", "3", "total"]
    assert counts.columns.tolist() == ["1", "2", "3", "4", "total"]
    np.testing.assert_array_equal(counts.iloc[:3, :4].to_numpy(), SMOOTH)
    np.testing.assert_array_equal(counts["total"].to_numpy()[:3], SMOOTH.sum(1))
    assert counts.loc["total", "total"] == SMOOTH.sum()
    n = SMOOTH.sum()
    expected = np.outer(SMOOTH.sum(1), SMOOTH.sum(0)) / n
    np.testing.assert_allclose(result["expected"].to_numpy(), expected)
    np.testing.assert_allclose(result["row_percent"].iloc[:3, :4].to_numpy(),
                               100 * SMOOTH / SMOOTH.sum(1, keepdims=True))
    np.testing.assert_allclose(result["row_percent"].loc["total"].to_numpy()[:4],
                               100 * SMOOTH.sum(0) / n)
    np.testing.assert_allclose(result["column_percent"].iloc[:3, :4].to_numpy(),
                               100 * SMOOTH / SMOOTH.sum(0, keepdims=True))
    np.testing.assert_allclose(result["total_percent"].iloc[:3, :4].to_numpy(), 100 * SMOOTH / n)
    standardized = (SMOOTH - expected) / np.sqrt(expected)
    np.testing.assert_allclose(result["residuals"].to_numpy(), standardized)
    leverage = np.outer(1 - SMOOTH.sum(1) / n, 1 - SMOOTH.sum(0) / n)
    np.testing.assert_allclose(result["adjusted_residuals"].to_numpy(),
                               standardized / np.sqrt(leverage))
    only_rows = oe.crosstab(data, "r", "c", percentages=["row"])
    assert "row_percent" in only_rows and "column_percent" not in only_rows


def test_chi_square_tests_match_scipy():
    result = oe.crosstab(expand(SMOOTH), "r", "c")
    tests = result["tests"]
    chi2, p, df, _ = stats.chi2_contingency(SMOOTH, correction=False)
    assert tests.loc["pearson", "statistic"] == pytest.approx(chi2, rel=1e-12)
    assert tests.loc["pearson", "p_value"] == pytest.approx(p, rel=1e-9)
    assert tests.loc["pearson", "df"] == df == 6
    g2, p_g2, _, _ = stats.chi2_contingency(SMOOTH, lambda_="log-likelihood")
    assert tests.loc["likelihood_ratio", "statistic"] == pytest.approx(g2, rel=1e-12)
    assert tests.loc["likelihood_ratio", "p_value"] == pytest.approx(p_g2, rel=1e-9)
    data = expand(SMOOTH)
    r = stats.pearsonr(data.r, data.c).statistic
    assert tests.loc["linear_by_linear", "statistic"] == pytest.approx((len(data) - 1) * r * r)
    assert result.attrs["statistic"] == pytest.approx(chi2)
    assert result.attrs["cells_below_5"] == 0 and "notes" not in result.attrs
    assert result.attrs["min_expected"] == pytest.approx(38 * 26 / 122)
    assert "fisher_exact" not in tests.index                 # r x c exact only on request
    sparse = oe.crosstab(expand(np.array([[6, 1, 1], [1, 6, 2], [8, 9, 30]])), "r", "c")
    assert sparse.attrs["cells_below_5"] == 6
    assert "6 cell(s) (66.7%) have expected count less than 5" in sparse.attrs["notes"][0]
    assert "minimum expected count is 1.88" in sparse.attrs["notes"][0]


def test_two_by_two_tests_and_risk_match_scipy_and_statsmodels():
    table = np.array([[12, 5], [7, 14]])
    result = oe.crosstab(expand(table, [0, 1], [0, 1]), "r", "c", alpha=0.10)
    tests = result["tests"]
    assert tests.loc["continuity", "statistic"] == pytest.approx(
        stats.chi2_contingency(table, correction=True)[0], rel=1e-12)
    for name, alternative in (("fisher_exact", "two-sided"), ("fisher_exact_less", "less"),
                              ("fisher_exact_greater", "greater")):
        assert tests.loc[name, "p_value"] == pytest.approx(
            stats.fisher_exact(table, alternative=alternative).pvalue, rel=1e-10)
    assert result.attrs["p_exact"] == tests.loc["fisher_exact", "p_value"]
    reference = Table2x2(table)
    risk = result["risk"]
    assert risk.loc["odds_ratio", "value"] == pytest.approx(reference.oddsratio)
    assert risk.loc["odds_ratio", ["ci_low", "ci_high"]].tolist() == pytest.approx(
        list(reference.oddsratio_confint(alpha=0.10)))
    assert risk.loc["risk_ratio_column_1", "value"] == pytest.approx(reference.riskratio)
    assert risk.loc["risk_ratio_column_1", ["ci_low", "ci_high"]].tolist() == pytest.approx(
        list(reference.riskratio_confint(alpha=0.10)))
    flipped = Table2x2(table[:, ::-1])
    assert risk.loc["risk_ratio_column_2", "value"] == pytest.approx(flipped.riskratio)
    measures = result["measures"]
    data = expand(table, [0, 1], [0, 1])
    assert measures.loc["phi", "value"] == pytest.approx(stats.pearsonr(data.r, data.c).statistic)
    negative = oe.crosstab(expand(table[::-1], [0, 1], [0, 1]), "r", "c")["measures"]
    assert negative.loc["phi", "value"] == pytest.approx(-measures.loc["phi", "value"])
    assert "risk" not in oe.crosstab(expand(SMOOTH), "r", "c")
    assert "fisher_exact" not in oe.crosstab(data, "r", "c", exact=False)["tests"].index


# ---- measures ------------------------------------------------------------------------------


def test_measure_values_match_scipy_and_definitions():
    data = expand(SMOOTH)
    measures = oe.crosstab(data, "r", "c")["measures"]
    value = measures["value"]
    association = stats.contingency.association
    assert value["cramers_v"] == pytest.approx(association(SMOOTH.astype(int), method="cramer"))
    assert value["contingency_coefficient"] == pytest.approx(
        association(SMOOTH.astype(int), method="pearson"))
    chi2 = stats.chi2_contingency(SMOOTH, correction=False)[0]
    assert value["phi"] == pytest.approx(math.sqrt(chi2 / SMOOTH.sum()))
    assert value["kendall_tau_b"] == pytest.approx(stats.kendalltau(data.r, data.c).statistic)
    assert value["kendall_tau_c"] == pytest.approx(
        stats.kendalltau(data.r, data.c, variant="c").statistic)
    column_dependent = stats.somersd(SMOOTH.astype(int))
    row_dependent = stats.somersd(SMOOTH.astype(int).T)
    assert value["somers_d_column"] == pytest.approx(column_dependent.statistic)
    assert value["somers_d_row"] == pytest.approx(row_dependent.statistic)
    assert measures.loc["somers_d_column", "p_value"] == pytest.approx(column_dependent.pvalue,
                                                                       rel=1e-8)
    assert value["somers_d_symmetric"] == pytest.approx(
        2 / (1 / column_dependent.statistic + 1 / row_dependent.statistic))
    spearman, pearson = stats.spearmanr(data.r, data.c), stats.pearsonr(data.r, data.c)
    assert value["spearman_rho"] == pytest.approx(spearman.statistic)
    assert measures.loc["spearman_rho", "p_value"] == pytest.approx(spearman.pvalue, rel=1e-8)
    assert value["pearson_r"] == pytest.approx(pearson.statistic)
    assert measures.loc["pearson_r", "p_value"] == pytest.approx(pearson.pvalue, rel=1e-8)
    # Pair counts by brute force.
    r, c = data.r.to_numpy(), data.c.to_numpy()
    product = np.sign(r[:, None] - r[None, :]) * np.sign(c[:, None] - c[None, :])
    concordant, discordant = (product > 0).sum(), (product < 0).sum()
    assert value["gamma"] == pytest.approx((concordant - discordant) / (concordant + discordant))
    # lambda, Goodman-Kruskal tau and the uncertainty coefficient from their definitions.
    n = SMOOTH.sum()
    rows, columns = SMOOTH.sum(1), SMOOTH.sum(0)
    assert value["lambda_column"] == pytest.approx(
        (SMOOTH.max(1).sum() - columns.max()) / (n - columns.max()))
    assert value["lambda_row"] == pytest.approx(
        (SMOOTH.max(0).sum() - rows.max()) / (n - rows.max()))
    assert value["lambda_symmetric"] == pytest.approx(
        (SMOOTH.max(1).sum() + SMOOTH.max(0).sum() - columns.max() - rows.max())
        / (2 * n - columns.max() - rows.max()))
    tau_column = (n * (SMOOTH ** 2 / rows[:, None]).sum() - (columns ** 2).sum()) \
        / (n * n - (columns ** 2).sum())
    assert value["goodman_kruskal_tau_column"] == pytest.approx(tau_column)
    assert measures.loc["goodman_kruskal_tau_column", "p_value"] == pytest.approx(
        stats.chi2.sf((n - 1) * 3 * tau_column, 6), rel=1e-8)

    def entropy(v):
        return -(v / n * np.log(v / n)).sum()

    mutual = entropy(rows) + entropy(columns) - entropy(SMOOTH)
    assert value["uncertainty_column"] == pytest.approx(mutual / entropy(columns))
    assert value["uncertainty_row"] == pytest.approx(mutual / entropy(rows))
    assert value["uncertainty_symmetric"] == pytest.approx(
        2 * mutual / (entropy(rows) + entropy(columns)))
    g2 = stats.chi2_contingency(SMOOTH, lambda_="log-likelihood")
    assert measures.loc["uncertainty_row", "p_value"] == pytest.approx(g2[1], rel=1e-8)


def test_every_asymptotic_standard_error_equals_the_numerical_delta_method():
    counts = torch.tensor(SMOOTH, dtype=torch.float64)
    base = all_measures(counts)
    checked = []
    for name, row in base.items():
        if row[1] is None:
            continue
        gradient = torch.zeros_like(counts)
        step = 1e-5
        for i, j in itertools.product(range(counts.shape[0]), range(counts.shape[1])):
            up, down = counts.clone(), counts.clone()
            up[i, j] += step
            down[i, j] -= step
            gradient[i, j] = (all_measures(up)[name][0] - all_measures(down)[name][0]) / (2 * step)
        numerical = math.sqrt(float((counts * gradient ** 2).sum()))
        assert row[1] == pytest.approx(numerical, rel=2e-6), name
        # degree-zero homogeneity: sum f * d theta / d f = 0
        assert float((counts * gradient).sum()) == pytest.approx(0.0, abs=1e-6), name
        checked.append(name)
    assert len(checked) == 16 and {"lambda_symmetric", "spearman_rho", "uncertainty_row",
                                   "goodman_kruskal_tau_row", "kendall_tau_b"} <= set(checked)


def test_ordinal_null_standard_errors_follow_the_spss_formula():
    counts = torch.tensor(SMOOTH, dtype=torch.float64)
    concordant, discordant = measure_kernels.concordance(counts)
    f = SMOOTH
    c, d = concordant.numpy(), discordant.numpy()
    p, q = (f * c).sum(), (f * d).sum()
    n = f.sum()
    spread = math.sqrt((f * (c - d) ** 2).sum() - (p - q) ** 2 / n)
    rows = measure_kernels.ordinal_measures(counts)
    assert rows["gamma"][2] == pytest.approx((p - q) / (p + q) / (2 * spread / (p + q)))
    d_r, d_c = n * n - (f.sum(1) ** 2).sum(), n * n - (f.sum(0) ** 2).sum()
    assert rows["kendall_tau_b"][2] == pytest.approx((p - q) / (2 * spread))
    assert rows["kendall_tau_c"][1] == pytest.approx(2 * 3 / (2 * n * n) * spread)
    assert rows["somers_d_column"][0] == pytest.approx((p - q) / d_r)
    assert rows["somers_d_row"][0] == pytest.approx((p - q) / d_c)
    # brute-force C_ij for one interior cell
    assert c[1, 1] == f[0, 0] + f[2, 2] + f[2, 3]
    assert d[1, 1] == f[0, 2] + f[0, 3] + f[2, 0]


@pytest.mark.parametrize("weighting, wt", [("none", None), ("linear", "linear"),
                                           ("quadratic", "quadratic")])
def test_kappa_matches_statsmodels(weighting, wt):
    square = np.array([[20, 5, 2], [4, 15, 6], [1, 7, 25]])
    reference = cohens_kappa(square, wt=wt)
    value, ase, z, p = measure_kernels.kappa(torch.tensor(square, dtype=torch.float64), weighting)
    assert value == pytest.approx(reference.kappa, rel=1e-12)
    assert ase == pytest.approx(reference.std_kappa, rel=1e-10)
    assert z == pytest.approx(reference.z_value, rel=1e-10)
    assert p == pytest.approx(reference.pvalue_two_sided, rel=1e-8)
    measures = oe.crosstab(expand(square), "r", "c")["measures"]
    name = "kappa" if weighting == "none" else f"kappa_{weighting}"
    assert measures.loc[name, "value"] == pytest.approx(reference.kappa)
    assert measures.loc[name, "ase"] == pytest.approx(reference.std_kappa)


def test_kappa_requires_the_same_categories():
    different = expand(np.array([[5, 2], [3, 6]]), ["a", "b"], ["x", "y"])
    assert "kappa" not in oe.crosstab(different, "r", "c")["measures"].index
    same = expand(np.array([[5, 2], [3, 6]]), ["a", "b"], ["a", "b"])
    measures = oe.crosstab(same, "r", "c")["measures"]
    assert "kappa" in measures.index and "kappa_linear" not in measures.index


# ---- exact tests for r x c tables ----------------------------------------------------------


def brute_force_fisher(table: np.ndarray) -> float:
    rows, columns = table.sum(1), table.sum(0)
    constant = sum(math.lgamma(v + 1) for v in rows) + sum(math.lgamma(v + 1) for v in columns) \
        - math.lgamma(table.sum() + 1)

    def log_probability(t):
        return constant - sum(math.lgamma(v + 1) for v in t.ravel())

    observed, total = log_probability(table), 0.0
    r, c = table.shape
    ranges = [range(int(min(rows[i], columns[j])) + 1) for i in range(r - 1) for j in range(c - 1)]
    for cells in itertools.product(*ranges):
        t = np.zeros((r, c))
        t[:r - 1, :c - 1] = np.array(cells).reshape(r - 1, c - 1)
        t[:r - 1, c - 1] = rows[:r - 1] - t[:r - 1, :c - 1].sum(1)
        t[r - 1, :] = columns - t[:r - 1, :].sum(0)
        if (t < 0).any():
            continue
        if log_probability(t) <= observed + 1e-7:
            total += math.exp(log_probability(t))
    return total


@pytest.mark.parametrize("table", [
    np.array([[3, 1, 0], [1, 4, 2], [0, 2, 5]]),
    np.array([[2, 0, 1, 3], [1, 4, 0, 1]]),
    np.array([[4, 1], [1, 3], [0, 5], [2, 2]]),
])
def test_fisher_freeman_halton_enumeration_matches_brute_force(table):
    counts = torch.tensor(table, dtype=torch.float64)
    reference = brute_force_fisher(table)
    assert table_kernels.fisher_enumerate(counts) == pytest.approx(reference, rel=1e-10)
    result = oe.crosstab(expand(table), "r", "c", exact=True)
    assert result.attrs["exact_method"] == "enumeration"
    assert result.attrs["p_exact"] == pytest.approx(reference, rel=1e-10)
    assert result["tests"].loc["fisher_exact", "p_value"] == pytest.approx(reference, rel=1e-10)


def test_fisher_enumeration_equals_hypergeometric_for_two_by_two():
    table = np.array([[7, 2], [3, 9]])
    counts = torch.tensor(table, dtype=torch.float64)
    assert table_kernels.fisher_enumerate(counts) == pytest.approx(
        stats.fisher_exact(table).pvalue, rel=1e-10)


def test_monte_carlo_exact_test_is_seeded_and_close_to_the_exact_value():
    table = np.array([[3, 1, 0], [1, 4, 2], [0, 2, 5]])
    counts = torch.tensor(table, dtype=torch.float64)
    exact = brute_force_fisher(table)
    p, low, high = table_kernels.fisher_monte_carlo(counts, 40_000, 7)
    assert low <= p <= high and abs(p - exact) < 4 * math.sqrt(exact * (1 - exact) / 40_000)
    assert table_kernels.fisher_monte_carlo(counts, 40_000, 7)[0] == p          # reproducible
    assert table_kernels.fisher_monte_carlo(counts, 40_000, 8)[0] != p
    # A table too large to enumerate falls back to Monte Carlo and records its settings.
    rng = np.random.default_rng(41)
    big = pd.DataFrame({"r": rng.integers(0, 5, size=400), "c": rng.integers(0, 6, size=400)})
    result = oe.crosstab(big, "r", "c", exact=True, exact_reps=2000, seed=3)
    attrs = result.attrs
    assert attrs["exact_method"] == "monte_carlo" and attrs["exact_reps"] == 2000
    assert attrs["exact_seed"] == 3
    assert attrs["p_exact_ci_low"] <= attrs["p_exact"] <= attrs["p_exact_ci_high"]
    assert abs(attrs["p_exact"] - attrs["p_value"]) < 0.08       # near the chi-square p
    again = oe.crosstab(big, "r", "c", exact=True, exact_reps=2000, seed=3)
    assert again.attrs["p_exact"] == attrs["p_exact"]
    with pytest.raises(AnalysisError) as error:
        oe.crosstab(big, "r", "c", exact=True, exact_reps=1_000_000)
    assert error.value.code == "exact_unavailable"


# ---- layers --------------------------------------------------------------------------------


def test_layered_two_by_two_tables_match_statsmodels():
    rng = np.random.default_rng(42)
    s = rng.integers(0, 4, size=400)
    x = rng.integers(0, 2, size=400)
    y = (rng.random(400) < 0.25 + 0.25 * x + 0.06 * s).astype(int)
    result = oe.crosstab({"x": x, "y": y, "s": s}, "x", "y", layer="s", alpha=0.05)
    reference = StratifiedTable([pd.crosstab(x[s == k], y[s == k]).to_numpy() for k in range(4)])
    cmh = result["cmh"]
    plain, corrected = reference.test_null_odds(False), reference.test_null_odds(True)
    assert cmh.loc["cmh", "statistic"] == pytest.approx(plain.statistic, rel=1e-10)
    assert cmh.loc["cmh", "p_value"] == pytest.approx(plain.pvalue, rel=1e-8)
    assert cmh.loc["cmh_continuity", "statistic"] == pytest.approx(corrected.statistic, rel=1e-10)
    homogeneity = reference.test_equal_odds(adjust=False)
    tarone = reference.test_equal_odds(adjust=True)
    assert cmh.loc["breslow_day", "statistic"] == pytest.approx(homogeneity.statistic, rel=1e-8)
    assert cmh.loc["breslow_day", "df"] == 3
    assert cmh.loc["tarone", "statistic"] == pytest.approx(tarone.statistic, rel=1e-8)
    assert cmh.loc["tarone", "p_value"] == pytest.approx(tarone.pvalue, rel=1e-7)
    tables = np.array([pd.crosstab(x[s == k], y[s == k]).to_numpy() for k in range(4)], float)
    a, n = tables[:, 0, 0], tables.sum((1, 2))
    r1, c1 = tables[:, 0].sum(1), tables[:, :, 0].sum(1)
    excess = (a - r1 * c1 / n).sum()
    binomial = (r1 * (n - r1) * c1 * (n - c1) / n ** 3).sum()
    assert cmh.loc["cochran", "statistic"] == pytest.approx(excess ** 2 / binomial)
    odds = result["common_odds_ratio"].loc["mantel_haenszel"]
    assert odds["value"] == pytest.approx(reference.oddsratio_pooled, rel=1e-12)
    assert odds["std_error_log"] == pytest.approx(reference.logodds_pooled_se, rel=1e-10)
    assert [odds["ci_low"], odds["ci_high"]] == pytest.approx(
        list(reference.oddsratio_pooled_confint()), rel=1e-9)
    # Per-layer blocks and the total block.
    counts = result["counts"]
    assert counts.index[:3].tolist() == ["s=0 | 0", "s=0 | 1", "s=0 | total"]
    assert counts.index[-3:].tolist() == ["total | 0", "total | 1", "total | total"]
    layer_table = pd.crosstab(x[s == 2], y[s == 2]).to_numpy()
    assert counts.loc[["s=2 | 0", "s=2 | 1"], ["0", "1"]].to_numpy().tolist() == \
        layer_table.tolist()
    assert result["tests"].loc["s=2 | pearson", "statistic"] == pytest.approx(
        stats.chi2_contingency(layer_table, correction=False)[0])
    assert result["tests"].loc["total | pearson", "statistic"] == pytest.approx(
        stats.chi2_contingency(pd.crosstab(x, y).to_numpy(), correction=False)[0])
    assert result.attrs["statistic"] == result["tests"].loc["total | pearson", "statistic"]
    assert result.attrs["layer_levels"] == [0, 1, 2, 3] and result.attrs["strata_used"] == 4


def test_layer_with_a_constant_variable_and_larger_tables():
    data = {"x": [0, 0, 1, 1, 0, 1, 0, 1, 0, 0], "y": [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            "s": ["a", "a", "a", "a", "a", "a", "b", "b", "c", "c"]}
    result = oe.crosstab(data, "x", "y", layer="s")
    notes = " ".join(result.attrs["notes"])
    assert "s=c | No statistics are computed" in notes
    assert "1 layer(s) with a zero margin" in notes
    assert result.attrs["strata"] == 3 and result.attrs["strata_used"] == 2
    assert not any(label.startswith("s=c | ") for label in result["tests"].index)
    wide = oe.crosstab({"x": [1, 2, 3, 1, 2, 3], "y": [0, 1, 0, 1, 0, 1], "s": [0, 0, 0, 1, 1, 1]},
                       "x", "y", layer="s")
    assert "cmh" not in wide and "2x2 tables only" in " ".join(wide.attrs["notes"])


# ---- weights, labels, degenerate tables ----------------------------------------------------


def test_frequency_weights_reproduce_expanded_data():
    cells = [(i, j) for i in range(3) for j in range(4)]
    weighted = pd.DataFrame({"r": [i + 1 for i, _ in cells], "c": [j + 1 for _, j in cells],
                             "w": SMOOTH.ravel()})
    aggregated = oe.crosstab(weighted, "r", "c", weights="w")
    expanded = oe.crosstab(expand(SMOOTH), "r", "c")
    pd.testing.assert_frame_equal(aggregated["counts"], expanded["counts"])
    pd.testing.assert_frame_equal(aggregated["tests"], expanded["tests"])
    pd.testing.assert_frame_equal(aggregated["measures"], expanded["measures"])
    fractional = oe.crosstab(weighted.assign(w=weighted.w / 4), "r", "c", weights="w",
                             exact=True)
    assert fractional["counts"].loc["1", "1"] == pytest.approx(5.25)
    assert "fisher_exact" not in fractional["tests"].index
    assert "integer counts" in " ".join(fractional.attrs["notes"])
    assert fractional["measures"].loc["gamma", "value"] == pytest.approx(
        expanded["measures"].loc["gamma", "value"])
    zero = weighted.copy()
    zero.loc[zero.r == 3, "w"] = 0.0                       # an empty row is trimmed for the tests
    trimmed = oe.crosstab(zero, "r", "c", weights="w")
    assert trimmed["tests"].loc["pearson", "df"] == 3
    assert trimmed["tests"].loc["pearson", "statistic"] == pytest.approx(
        stats.chi2_contingency(SMOOTH[:2], correction=False)[0])


def test_text_categories_missing_values_and_constant_variables():
    frame = pd.DataFrame({"r": ["lo", "hi", "lo", None, "hi", "lo", "hi", "lo"],
                          "c": ["u", "v", "v", "u", None, "u", "v", "u"]})
    result = oe.crosstab(frame, "r", "c")
    assert result.attrs["n"] == 6 and result.attrs["n_dropped"] == 2
    assert result.attrs["row_levels"] == ["hi", "lo"]
    assert result["counts"].loc["lo", "u"] == 3
    with pytest.raises(AnalysisError) as error:
        oe.crosstab(frame, "r", "c", missing="raise")
    assert error.value.code == "missing_values"
    constant = oe.crosstab({"r": [1, 1, 1], "c": [0, 1, 1]}, "r", "c")
    assert list(constant) == ["counts"] and "No statistics" in constant.attrs["notes"][0]
    assert "statistic" not in constant.attrs
    ordered = frame.dropna().copy()
    ordered["r"] = pd.Categorical(ordered["r"], categories=["lo", "hi"])
    assert oe.crosstab(ordered, "r", "c").attrs["row_levels"] == ["lo", "hi"]


def test_error_codes():
    data = {"r": [1, 2, 1, 2], "c": [0, 1, 1, 0], "w": [1.0, 2.0, -1.0, 1.0]}
    cases = [
        (lambda: oe.crosstab(data, "r", "nope"), "missing_columns"),
        (lambda: oe.crosstab(data, "r", "r"), "invalid_spec"),
        (lambda: oe.crosstab(data, "r", ["c"]), "invalid_spec"),
        (lambda: oe.crosstab(data, "r", "c", weights="w"), "invalid_weights"),
        (lambda: oe.crosstab(data, "r", "c", percentages="cell"), "invalid_option"),
        (lambda: oe.crosstab(data, "r", "c", exact="yes"), "invalid_option"),
        (lambda: oe.crosstab(data, "r", "c", exact_reps=5), "invalid_option"),
        (lambda: oe.crosstab(data, "r", "c", alpha=0), "invalid_option"),
        (lambda: oe.crosstab({"r": np.arange(400), "c": np.arange(400)}, "r", "c"),
         "too_many_categories"),
        (lambda: oe.tabulate(data, "nope"), "missing_columns"),
        (lambda: oe.tabulate(data, "r", weights="w"), "invalid_weights"),
        (lambda: oe.tabulate({"x": [None, None]}, "x"), "empty_sample"),
        (lambda: oe.tabulate({"x": [1, None]}, "x", missing="raise"), "missing_values"),
        (lambda: oe.tabulate({"x": []}, "x"), "empty_data"),
        (lambda: oe.tabulate(data, "r", weights="r"), "invalid_spec"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code, str(error.value))


# ---- tabulate ------------------------------------------------------------------------------


def test_tabulate_frequencies_percentages_and_missing_row():
    frame = pd.DataFrame({"x": ["b", "a", "b", None, "c", "b", "a", None],
                          "w": [1.0, 2.0, 1.0, 3.0, 0.5, 1.5, 1.0, 1.0]})
    result = oe.tabulate(frame, "x")
    assert not isinstance(result, TableSet)
    assert result.index.tolist() == ["a", "b", "c", "missing"]
    assert result["count"].tolist() == [2, 3, 1, 2]
    np.testing.assert_allclose(result["percent"], [25.0, 37.5, 12.5, 25.0])
    np.testing.assert_allclose(result["valid_percent"][:3], [100 / 3, 50.0, 100 / 6])
    np.testing.assert_allclose(result["cumulative_percent"][:3], [100 / 3, 250 / 3, 100.0])
    assert math.isnan(result.loc["missing", "valid_percent"])
    assert result.attrs == {"n": 6, "n_missing": 2, "n_total": 8, "k": 3, "mode": "b",
                            "n_dropped": 0, "label": "Frequencies of x"}
    unweighted = frame.assign(w=[1.0, None, 1.0, 3.0, 0.5, 1.5, 1.0, 1.0])
    assert oe.tabulate(unweighted, "x", weights="w").attrs["n_dropped"] == 1
    with pytest.raises(AnalysisError) as error:
        oe.tabulate(unweighted, "x", weights="w", missing="raise")
    assert error.value.code == "missing_values"
    weighted = oe.tabulate(frame, "x", weights="w")
    np.testing.assert_allclose(weighted["count"], [3.0, 3.5, 0.5, 4.0])
    assert weighted.attrs["n"] == pytest.approx(7.0) and weighted.attrs["mode"] == "b"
    numeric = oe.tabulate({"x": [3, 1, 2, 3, 3]}, "x")
    assert numeric.index.tolist() == ["1", "2", "3"] and numeric.attrs["mode"] == 3
    assert "missing" not in numeric.index
    reference = pd.Series([3, 1, 2, 3, 3]).value_counts().sort_index()
    assert numeric["count"].tolist() == reference.tolist()


def test_results_render_and_have_json_safe_attrs():
    rng = np.random.default_rng(43)
    data = pd.DataFrame({"r": rng.integers(0, 2, size=80), "c": rng.integers(0, 2, size=80),
                         "s": rng.integers(0, 3, size=80)})
    for result in (oe.crosstab(data, "r", "c", layer="s", expected=True, percentages="all",
                               residuals=True), oe.crosstab(expand(SMOOTH), "r", "c"),
                   oe.tabulate(data, "s")):
        attrs = json.loads(json.dumps(result.attrs))
        assert attrs == result.attrs
        assert "\\begin{tabular}" in str(result.to_latex())
        assert str(result)


def test_one_million_rows_are_tabulated_quickly():
    rng = np.random.default_rng(44)
    n = 1_000_000
    data = pd.DataFrame({"r": rng.integers(0, 5, size=n), "c": rng.integers(0, 6, size=n),
                         "s": rng.integers(0, 3, size=n)})
    result = oe.crosstab(data, "r", "c", layer="s")
    table = pd.crosstab(data.r, data.c).to_numpy()
    assert result.attrs["statistic"] == pytest.approx(
        stats.chi2_contingency(table, correction=False)[0], rel=1e-9)
    assert oe.tabulate(data, "r")["count"].tolist() == table.sum(1).tolist()
