"""Correlations, partial correlations and descriptive statistics against SciPy / NumPy."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats.correlation import inversions, kendall_tau_b, midranks


def _frame(seed: int = 11, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"a": rng.normal(size=n)})
    frame["b"] = 0.5 * frame["a"] + rng.normal(size=n)
    frame["c"] = np.round(frame["b"] + rng.normal(size=n))          # many ties
    frame["d"] = rng.integers(0, 4, n).astype(float)                # heavy ties
    frame.loc[rng.choice(n, 20, replace=False), "b"] = np.nan
    frame.loc[rng.choice(n, 15, replace=False), "d"] = np.nan
    return frame


_ORACLES = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_pairwise_and_listwise_correlations_match_scipy(method):
    frame = _frame()
    names = list("abcd")
    pairwise = oe.correlate(frame, names, method=method)
    listwise = oe.correlate(frame, names, method=method, pairwise=False)
    complete = frame.dropna()
    for i, j in ((i, j) for i in names for j in names if i < j):
        both = frame[[i, j]].dropna()
        reference = _ORACLES[method](both[i], both[j])
        assert pairwise["coefficients"].loc[i, j] == pytest.approx(reference[0], abs=1e-13)
        assert pairwise["p_values"].loc[i, j] == pytest.approx(reference[1], rel=1e-9, abs=1e-300)
        assert pairwise["n"].loc[i, j] == len(both)
        assert pairwise["coefficients"].loc[j, i] == pairwise["coefficients"].loc[i, j]
        reference = _ORACLES[method](complete[i], complete[j])
        assert listwise["coefficients"].loc[i, j] == pytest.approx(reference[0], abs=1e-13)
        assert listwise["p_values"].loc[i, j] == pytest.approx(reference[1], rel=1e-9, abs=1e-300)
        assert listwise["n"].loc[i, j] == len(complete)
    assert np.allclose(np.diag(pairwise["coefficients"]), 1.0)
    assert pairwise["p_values"].loc["a", "a"] != pairwise["p_values"].loc["a", "a"]   # missing
    assert pairwise["n"].loc["b", "b"] == 180
    assert pairwise.attrs["missing"] == "pairwise" and listwise.attrs["missing"] == "listwise"
    assert pairwise.attrs["n_complete"] == len(complete)


def test_fisher_z_intervals():
    frame = _frame()
    z = stats.norm.ppf(0.95)
    for method, (factor, offset) in {"pearson": (1.0, 3), "spearman": (1.06, 3),
                                     "kendall": (0.437, 4)}.items():
        table = oe.correlate(frame, ["a", "b", "c"], method=method, ci=True, alpha=0.1)["intervals"]
        assert list(table.columns) == ["var_i", "var_j", "coefficient", "ci_low", "ci_high", "n"]
        row = table.iloc[0]
        half = z * math.sqrt(factor / (row["n"] - offset))
        assert row["ci_low"] == pytest.approx(math.tanh(math.atanh(row["coefficient"]) - half))
        assert row["ci_high"] == pytest.approx(math.tanh(math.atanh(row["coefficient"]) + half))
    reference = stats.pearsonr(*frame[["a", "c"]].to_numpy().T).confidence_interval(0.95)
    table = oe.correlate(frame, ["a", "c"], ci=True)["intervals"]
    assert (table.loc[0, "ci_low"], table.loc[0, "ci_high"]) == pytest.approx(tuple(reference))


def test_rank_kernels_against_brute_force():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 17, 64, 257):
        values = rng.integers(0, 9, n)
        brute = sum(int(values[i] > values[j]) for i in range(n) for j in range(i + 1, n))
        assert inversions(torch.as_tensor(values, dtype=torch.int64)) == brute
        x = torch.as_tensor(rng.integers(0, 5, n).astype(float))
        np.testing.assert_allclose(midranks(x).numpy(), stats.rankdata(x.numpy()))
    x = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    assert kendall_tau_b(x, x) == (None, None)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    tau, z = kendall_tau_b(x, -x)
    assert tau == -1.0 and z == pytest.approx(-6 / math.sqrt(4 * 3 * 13 / 18))


def test_constant_columns_and_perfect_correlation():
    data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "k": [3.0, 3.0, 3.0, 3.0, 3.0]}
    for method in ("pearson", "spearman", "kendall"):
        result = oe.correlate(data, ["x", "y", "k"], method=method, ci=True)
        assert result["coefficients"].loc["x", "y"] == pytest.approx(1.0)
        assert np.isnan(result["coefficients"].loc["x", "k"])
        assert np.isnan(result["coefficients"].loc["k", "k"])
        assert np.isnan(result["p_values"].loc["x", "k"])
        assert json.loads(json.dumps(result.attrs))["method"] == method
    assert oe.correlate(data, ["x", "y"])["p_values"].loc["x", "y"] == 0.0


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"columns": ["a"]}, "invalid_spec"), ({"columns": "a"}, "invalid_spec"),
    ({"columns": ["a", "a"]}, "invalid_spec"), ({"columns": ["a", "text"]}, "non_numeric_column"),
    ({"columns": ["a", "zzz"]}, "missing_columns"),
    ({"columns": ["a", "b"], "method": "phi"}, "invalid_option"),
    ({"columns": ["a", "b"], "pairwise": 1}, "invalid_option"),
    ({"columns": ["a", "b"], "alpha": 2}, "invalid_option")])
def test_correlate_error_codes(kwargs, code):
    frame = _frame()
    frame["text"] = "t"
    with pytest.raises(AnalysisError) as error:
        oe.correlate(frame, **kwargs)
    assert error.value.code == code


def test_correlate_needs_two_complete_rows():
    with pytest.raises(AnalysisError) as error:
        oe.correlate({"a": [1.0, None, 3.0], "b": [None, 2.0, None]}, ["a", "b"], pairwise=False)
    assert error.value.code == "insufficient_observations"


def test_partial_and_semipartial_correlations():
    frame = _frame()
    complete = frame.dropna()
    table = oe.pcorr(frame, "a", ["b", "c"], controls=["d"])
    fit = sm.OLS(complete["a"], sm.add_constant(complete[["b", "c", "d"]])).fit()
    for name in ("b", "c"):
        t = fit.tvalues[name]
        assert table.loc[name, "statistic"] == pytest.approx(t)
        assert table.loc[name, "partial_corr"] == pytest.approx(
            t / math.sqrt(t ** 2 + fit.df_resid))
        assert table.loc[name, "p_value"] == pytest.approx(fit.pvalues[name])
        others = [other for other in ("b", "c", "d") if other != name]
        reduced = sm.OLS(complete["a"], sm.add_constant(complete[others])).fit()
        # The squared semipartial correlation is the R^2 lost by dropping the variable.
        assert table.loc[name, "semipartial_corr_sq"] == pytest.approx(
            fit.rsquared - reduced.rsquared)
        assert table.loc[name, "partial_corr_sq"] == pytest.approx(
            (fit.rsquared - reduced.rsquared) / (1 - reduced.rsquared))
        # Definition: correlation of the residuals of y and x on the other regressors.
        ry = sm.OLS(complete["a"], sm.add_constant(complete[others])).fit().resid
        rx = sm.OLS(complete[name], sm.add_constant(complete[others])).fit().resid
        assert table.loc[name, "partial_corr"] == pytest.approx(np.corrcoef(ry, rx)[0, 1])
        assert table.loc[name, "semipartial_corr"] == pytest.approx(
            np.corrcoef(complete["a"], rx)[0, 1])
    assert table.attrs["n"] == len(complete)
    assert table.attrs["n_missing"] == len(frame) - len(complete)
    assert table.attrs["r_squared"] == pytest.approx(fit.rsquared)
    assert table.loc["b", "df"] == fit.df_resid
    simple = oe.pcorr(complete, "a", ["c"])
    assert simple.loc["c", "partial_corr"] == pytest.approx(stats.pearsonr(complete["a"],
                                                                           complete["c"])[0])
    assert r"\begin{tabular}" in str(table.to_latex())


def test_pcorr_error_codes():
    frame = _frame().dropna()
    frame["twin"] = 3 * frame["c"]
    with pytest.raises(AnalysisError) as error:
        oe.pcorr(frame, "a", ["c", "twin"])
    assert error.value.code == "collinear_design"
    with pytest.raises(AnalysisError) as error:
        oe.pcorr(frame.iloc[:3], "a", ["b", "c"], controls=["d"])
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.pcorr(frame, "a", ["a"])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.pcorr(frame.assign(e=2 * frame["c"] + 1), "e", ["c"])
    assert error.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as error:
        oe.pcorr(_frame(), "a", ["b"], missing="raise")
    assert error.value.code == "missing_values"


# ---- descriptives -----------------------------------------------------------------


def _describe_frame() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    frame = pd.DataFrame({"x": rng.gamma(2.0, 1.0, 101), "z": np.round(rng.normal(size=101), 1),
                          "g": rng.choice(["a", "b", "c"], 101)})
    frame.loc[[3, 7, 50], "x"] = np.nan
    return frame


def test_describe_defaults_match_numpy_and_scipy():
    frame = _describe_frame()
    table = oe.describe(frame, ["x", "z"])
    assert list(table.columns) == ["n", "mean", "std_dev", "std_error", "ci_low", "ci_high", "min",
                                   "p25", "p50", "p75", "max", "skewness", "kurtosis"]
    for name in ("x", "z"):
        values = frame[name].dropna().to_numpy()
        row = table.loc[name]
        assert row["n"] == len(values) and row["mean"] == pytest.approx(values.mean())
        assert row["std_dev"] == pytest.approx(values.std(ddof=1))
        assert row["std_error"] == pytest.approx(stats.sem(values))
        interval = stats.t.interval(0.95, len(values) - 1, values.mean(), stats.sem(values))
        assert (row["ci_low"], row["ci_high"]) == pytest.approx(interval)
        assert (row["min"], row["max"]) == (values.min(), values.max())
        # Stata's default percentile is the averaged inverted CDF (SAS definition 5).
        expected = np.percentile(values, [25, 50, 75], method="averaged_inverted_cdf")
        assert row[["p25", "p50", "p75"]].tolist() == pytest.approx(expected.tolist())
        assert row["skewness"] == pytest.approx(stats.skew(values, bias=False))
        assert row["kurtosis"] == pytest.approx(stats.kurtosis(values, bias=False))
    assert table.attrs["missing"] == "variable-wise"
    everything = oe.describe(frame)
    assert list(everything.index) == ["x", "z"]


def test_describe_by_group_options_and_standard_errors():
    frame = _describe_frame()
    stats_list = ["n", "missing", "mean", "sd", "se", "variance", "cv", "sum", "range", "median",
                  "p10", "p90", "p2.5", "iqr", "skewness", "se_skewness", "kurtosis", "se_kurtosis"]
    table = oe.describe(frame, ["x", "z"], by="g", stats=stats_list,
                        percentile_method="haverage")
    assert list(table.columns[:2]) == ["variable", "g"] and len(table) == 6
    for (variable, group), row in table.set_index(["variable", "g"]).iterrows():
        column = frame.loc[frame["g"] == group, variable]
        values = column.dropna().to_numpy()
        n = len(values)
        assert row["n"] == n and row["missing"] == column.isna().sum()
        assert row["variance"] == pytest.approx(values.var(ddof=1))
        assert row["cv"] == pytest.approx(values.std(ddof=1) / values.mean())
        assert row["sum"] == pytest.approx(values.sum())
        assert row["range"] == pytest.approx(values.max() - values.min())
        # SPSS HAVERAGE = weighted average at (n + 1) p: NumPy's "weibull" method.
        expected = np.percentile(values, [50, 10, 90, 2.5, 25, 75], method="weibull")
        assert row[["p50", "p10", "p90", "p2.5"]].tolist() == pytest.approx(expected[:4].tolist())
        assert row["iqr"] == pytest.approx(expected[5] - expected[4])
        assert row["se_skewness"] == pytest.approx(
            math.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
        assert row["se_kurtosis"] == pytest.approx(
            2 * row["se_skewness"] * math.sqrt((n * n - 1) / ((n - 3) * (n + 5))))
    stata = oe.describe(frame, ["x"], by="g", stats=["skewness", "kurtosis", "se_skewness"],
                        moments="stata").set_index("g")
    for group in "abc":
        values = frame.loc[frame["g"] == group, "x"].dropna().to_numpy()
        assert stata.loc[group, "skewness"] == pytest.approx(stats.skew(values))
        assert stata.loc[group, "kurtosis"] == pytest.approx(stats.kurtosis(values, fisher=False))
        assert np.isnan(stata.loc[group, "se_skewness"])
    listwise = oe.describe(frame, ["x", "z"], listwise=True, stats=["n"])
    assert listwise["n"].tolist() == [98, 98]


def test_percentile_definitions_on_a_hand_example():
    data = {"x": [2.0, 4.0, 4.0, 5.0, 7.0, 9.0, 10.0, 12.0]}
    stata = oe.describe(data, ["x"], stats=["p25", "p50", "p75", "p10", "p99"]).loc["x"]
    # n p / 100 = 2, 4, 6 are integers: average of neighbours; 0.8 and 7.92 round up.
    assert stata.tolist() == [4.0, 6.0, 9.5, 2.0, 12.0]
    spss = oe.describe(data, ["x"], stats=["p25", "p50", "p75", "p10", "p99"],
                       percentile_method="haverage").loc["x"]
    # (n + 1) p / 100 = 2.25, 4.5, 6.75, 0.9, 8.91.
    assert spss.tolist() == pytest.approx([4.0, 6.0, 9.75, 2.0, 12.0])


def test_describe_small_groups_and_errors():
    data = {"x": [1.0, 2.0, 5.0, None], "g": ["a", "a", "b", "c"]}
    table = oe.describe(data, ["x"], by="g").set_index("g")
    assert table.loc["a", "n"] == 2 and np.isnan(table.loc["a", "skewness"])
    assert table.loc["b", "n"] == 1 and np.isnan(table.loc["b", "std_dev"])
    assert table.loc["b", "p50"] == 5.0 and np.isnan(table.loc["b", "ci_low"])
    assert table.loc["c", "n"] == 0 and np.isnan(table.loc["c", "mean"])
    assert not np.isinf(table.select_dtypes("number").to_numpy(dtype=float)).any()
    json.loads(table.to_json())
    for kwargs, code in (({"stats": ["mode"]}, "invalid_option"),
                         ({"stats": ["p100"]}, "invalid_option"),
                         ({"percentile_method": "r7"}, "invalid_option"),
                         ({"moments": "sas"}, "invalid_option"),
                         ({"by": "nope"}, "missing_columns"),
                         ({"columns": ["g"]}, "non_numeric_column")):
        with pytest.raises(AnalysisError) as error:
            oe.describe(data, **({"columns": ["x"]} | kwargs))
        assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        oe.describe({"g": ["a", "b"]})
    assert error.value.code == "invalid_spec"
