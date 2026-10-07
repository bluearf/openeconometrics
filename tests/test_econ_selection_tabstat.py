"""oe.tabstat against NumPy / SciPy / statsmodels and Stata's documented formulas."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.frame import DataFrame

ALL = ["n", "mean", "sum", "min", "max", "range", "sd", "variance", "cv", "semean", "skewness",
       "kurtosis", "median", "p10", "p90", "iqr", "gmedian", "harmonic", "geometric"]


def _data(seed: int = 0, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.gamma(2.0, 2.0, n), "z": rng.normal(size=n).round(1),
        "k": rng.integers(1, 7, n).astype(float),
        "g": rng.choice(["north", "south", "west"], n),
        "w": rng.uniform(0.5, 2.0, n), "f": rng.integers(1, 5, n)})


def _stata_percentile(x, w, p):
    """Stata's summarize, detail percentile with weights (explicit loop)."""
    order = np.argsort(x, kind="stable")
    x, w = np.asarray(x)[order], np.asarray(w, dtype=float)[order]
    cumulative = np.cumsum(w)
    target = cumulative[-1] * p / 100
    for i in range(len(x)):
        if cumulative[i] > target + 1e-9 * cumulative[-1]:
            if i > 0 and abs(cumulative[i - 1] - target) <= 1e-9 * cumulative[-1]:
                return (x[i - 1] + x[i]) / 2
            return x[i]
    return x[-1]


def _grouped_median(x, w=None):
    """Textbook grouped median: classes limited halfway to the neighbouring values."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    values = np.unique(x)
    freq = np.array([w[x == value].sum() for value in values])
    if len(values) == 1:
        return values[0]
    half, cumulative = freq.sum() / 2, np.cumsum(freq)
    i = int(np.argmax(cumulative >= half - 1e-12))
    down = values[i] - values[i - 1] if i > 0 else values[i + 1] - values[i]
    up = values[i + 1] - values[i] if i < len(values) - 1 else down
    below = cumulative[i - 1] if i > 0 else 0.0
    return values[i] - down / 2 + (half - below) / freq[i] * (down + up) / 2


def _expected(x, w=None, kind=None):
    """Every statistic of one group from first principles."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    total = w.sum()
    n = total if kind in (None, "fweight") else float(len(x))
    mean = (w * x).sum() / total
    m2, m3, m4 = (((w * (x - mean) ** r).sum() / total) for r in (2, 3, 4))
    variance = m2 * n / (n - 1)
    sd = math.sqrt(variance)
    q25, q50, q75 = (_stata_percentile(x, w, p) for p in (25, 50, 75))
    return {
        "n": n, "mean": mean, "sum": (w * x).sum(), "min": x.min(), "max": x.max(),
        "range": x.max() - x.min(), "std_dev": sd, "variance": variance, "cv": sd / mean,
        "std_error": sd / math.sqrt(n), "skewness": m3 / m2 ** 1.5, "kurtosis": m4 / m2 ** 2,
        "median": q50, "p10": _stata_percentile(x, w, 10), "p90": _stata_percentile(x, w, 90),
        "iqr": q75 - q25, "grouped_median": _grouped_median(x, w),
        "harmonic_mean": total / (w / x).sum() if (x > 0).all() else np.nan,
        "geometric_mean": math.exp((w * np.log(x)).sum() / total) if (x > 0).all() else np.nan,
    }


def _assert_row(row, expected, rel=1e-10):
    for name, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(row[name]), name
        else:
            assert row[name] == pytest.approx(value, rel=rel, abs=1e-12), name


# ---- public surface ---------------------------------------------------------------


def test_tabstat_is_exported_and_returns_a_table():
    from openecon.econometrics import registry

    assert registry.public_exports()["tabstat"][0] == "openecon.econometrics.selection.tabstat"
    assert "MEANS" in oe.tabstat.__doc__
    table = oe.tabstat(_data(), ["x", "z"])
    assert isinstance(table, DataFrame)
    assert list(table.index) == ["x", "z"]
    assert list(table.columns) == ["n", "mean", "std_dev", "min", "max"]      # the default
    assert table.attrs["missing"] == "variable-wise" and table.attrs["n"] == 240
    assert json.loads(json.dumps(table.attrs)) == table.attrs
    assert r"\begin{tabular}" in str(table.to_latex())
    result = oe.tabstat(_data(), ["x"], by="g", anova=True)
    assert isinstance(result, TableSet)
    assert list(result) == ["statistics", "anova", "association"]
    assert "Summary statistics by g" in str(result)
    assert result.to_latex().count(r"\begin{tabular}") == 3
    assert json.loads(json.dumps(result.attrs)) == result.attrs


# ---- unweighted -------------------------------------------------------------------


def test_unweighted_statistics_match_numpy_and_scipy():
    frame = _data()
    table = oe.tabstat(frame, ["x", "z", "k"], stats=ALL)
    for name in ("x", "z", "k"):
        values = frame[name].to_numpy()
        _assert_row(table.loc[name], _expected(values))
        row = table.loc[name]
        assert row["std_dev"] == pytest.approx(values.std(ddof=1), rel=1e-12)
        assert row["skewness"] == pytest.approx(stats.skew(values), rel=1e-10)
        assert row["kurtosis"] == pytest.approx(stats.kurtosis(values, fisher=False), rel=1e-10)
        assert row["std_error"] == pytest.approx(stats.sem(values), rel=1e-12)
        for column, p in (("p10", 10), ("median", 50), ("p90", 90)):
            assert row[column] == pytest.approx(
                np.percentile(values, p, method="averaged_inverted_cdf"), rel=1e-12, abs=1e-12)
    assert table.loc["x", "geometric_mean"] == pytest.approx(stats.gmean(frame["x"]), rel=1e-12)
    assert table.loc["x", "harmonic_mean"] == pytest.approx(stats.hmean(frame["x"]), rel=1e-12)
    assert math.isnan(table.loc["z", "geometric_mean"])       # z has non-positive values


def test_q_expands_and_any_percentile_is_available():
    frame = _data()
    table = oe.tabstat(frame, ["x"], stats=["q", "p2.5", "p99"])
    assert list(table.columns) == ["p25", "p50", "p75", "p2.5", "p99"]
    values = frame["x"].to_numpy()
    for column, p in (("p25", 25), ("p50", 50), ("p75", 75), ("p2.5", 2.5), ("p99", 99)):
        assert table.loc["x", column] == pytest.approx(
            np.percentile(values, p, method="averaged_inverted_cdf"), rel=1e-12)
    # Even n: the median averages the two middle values; integer P = n p / 100 averages too.
    even = oe.tabstat({"v": [1.0, 2.0, 3.0, 4.0]}, ["v"], stats=["median", "p25", "p75"])
    assert even.loc["v"].tolist() == [2.5, 1.5, 3.5]
    odd = oe.tabstat({"v": [5.0, 1.0, 3.0]}, ["v"], stats=["median", "p25"])
    assert odd.loc["v"].tolist() == [3.0, 1.0]


def test_grouped_median_textbook_examples():
    table = oe.tabstat({"v": [1, 2, 2, 3, 3, 3, 4, 4, 5]}, ["v"], stats=["gmedian", "median"])
    assert table.loc["v", "grouped_median"] == pytest.approx(3.0)      # 2.5 + (4.5 - 3) / 3
    table = oe.tabstat({"v": [1, 1, 2, 2, 2, 2, 3]}, ["v"], stats=["gmedian"])
    assert table.loc["v", "grouped_median"] == pytest.approx(1.875)    # 1.5 + (3.5 - 2) / 4
    # Class midpoints 35, 45, 55 (width 10): 40 + (5 - 3) / 4 * 10.
    ages = [35] * 3 + [45] * 4 + [55] * 3
    table = oe.tabstat({"age": ages}, ["age"], stats=["gmedian"])
    assert table.loc["age", "grouped_median"] == pytest.approx(45.0)
    ages = [35] * 3 + [45] * 4 + [55] * 5
    table = oe.tabstat({"age": ages}, ["age"], stats=["gmedian"])
    assert table.loc["age", "grouped_median"] == pytest.approx(40 + (6 - 3) / 4 * 10)
    constant = oe.tabstat({"v": [7.0, 7.0, 7.0]}, ["v"], stats=["gmedian"])
    assert constant.loc["v", "grouped_median"] == 7.0


def test_spss_moments_match_bias_corrected_scipy():
    frame = _data()
    table = oe.tabstat(frame, ["x"], by="g", moments="spss",
                       stats=["skewness", "se_skewness", "kurtosis", "se_kurtosis"])
    for label, group in frame.groupby("g"):
        values = group["x"].to_numpy()
        n = len(values)
        row = table[table["g"] == label].iloc[0]
        assert row["skewness"] == pytest.approx(stats.skew(values, bias=False), rel=1e-10)
        assert row["kurtosis"] == pytest.approx(stats.kurtosis(values, bias=False), rel=1e-10)
        se_skew = math.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        assert row["se_skewness"] == pytest.approx(se_skew, rel=1e-12)
        assert row["se_kurtosis"] == pytest.approx(
            math.sqrt(4 * (n * n - 1) * se_skew ** 2 / ((n - 3) * (n + 5))), rel=1e-12)
    assert table.attrs["moments"] == "spss"


# ---- groups -----------------------------------------------------------------------


def test_by_groups_and_total_rows():
    frame = _data()
    table = oe.tabstat(frame, ["x", "z"], by="g", stats=ALL)
    assert list(table.columns[:2]) == ["variable", "g"]
    assert list(table["g"]) == ["north", "south", "west", "Total"] * 2
    assert table.attrs["groups"] == ["north", "south", "west"] and table.attrs["by"] == "g"
    for name in ("x", "z"):
        block = table[table["variable"] == name].set_index("g")
        for label, group in frame.groupby("g"):
            _assert_row(block.loc[label], _expected(group[name].to_numpy()))
        _assert_row(block.loc["Total"], _expected(frame[name].to_numpy()))
    reference = frame.groupby("g")["x"].agg(["count", "mean", "std", "min", "max", "median"])
    block = table[table["variable"] == "x"].set_index("g").loc[reference.index]
    np.testing.assert_allclose(block[["n", "mean", "std_dev", "min", "max", "median"]]
                               .to_numpy(dtype=float), reference.to_numpy(), rtol=1e-12)
    without = oe.tabstat(frame, ["x"], by="g", total=False)
    assert list(without["g"]) == ["north", "south", "west"]
    # Numeric and categorical group labels keep their order.
    numbered = oe.tabstat(frame.assign(code=frame["g"].map({"north": 3, "south": 1, "west": 2})),
                          ["x"], by="code", total=False)
    assert list(numbered["code"]) == [1, 2, 3]
    ordered = frame.assign(g=pd.Categorical(frame["g"], categories=["west", "north", "south"]))
    assert list(oe.tabstat(ordered, ["x"], by="g", total=False)["g"]) == ["west", "north",
                                                                           "south"]


def test_anova_and_eta_match_scipy():
    frame = _data()
    result = oe.tabstat(frame, ["x", "z"], by="g", anova=True)
    for name in ("x", "z"):
        samples = [group[name].to_numpy() for _, group in frame.groupby("g")]
        reference = stats.f_oneway(*samples)
        block = result["anova"][result["anova"]["variable"] == name].set_index("source")
        assert block.loc["between", "statistic"] == pytest.approx(reference.statistic, rel=1e-10)
        assert block.loc["between", "p_value"] == pytest.approx(reference.pvalue, rel=1e-9)
        assert list(block["df"]) == [2, 237, 239]
        grand = frame[name].mean()
        between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
        within = sum(((s - s.mean()) ** 2).sum() for s in samples)
        assert block.loc["between", "ss"] == pytest.approx(between, rel=1e-10)
        assert block.loc["within", "ss"] == pytest.approx(within, rel=1e-12)
        assert block.loc["total", "ss"] == pytest.approx(((frame[name] - grand) ** 2).sum(),
                                                         rel=1e-12)
        assert block.loc["within", "ms"] == pytest.approx(within / 237, rel=1e-12)
        eta_squared = between / (between + within)
        row = result["association"].loc[name]
        assert row["eta_squared"] == pytest.approx(eta_squared, rel=1e-10)
        assert row["eta"] == pytest.approx(math.sqrt(eta_squared), rel=1e-10)
    # Cross-check with OpenEcon's own one-way ANOVA.
    oneway = oe.oneway(frame, "x", "g")
    block = result["anova"][result["anova"]["variable"] == "x"].set_index("source")
    assert block.loc["between", "statistic"] == pytest.approx(oneway.attrs["statistic"],
                                                              rel=1e-10)


# ---- weights ----------------------------------------------------------------------


def test_frequency_weights_equal_replicated_rows():
    frame = _data()
    expanded = frame.loc[frame.index.repeat(frame["f"])].reset_index(drop=True)
    weighted = oe.tabstat(frame, ["x", "z", "k"], by="g", stats=ALL, weights="f",
                          weight_type="fweight", anova=True)
    replicated = oe.tabstat(expanded, ["x", "z", "k"], by="g", stats=ALL, anova=True)
    for name in weighted:
        left = weighted[name].select_dtypes("number").to_numpy(dtype=float)
        right = replicated[name].select_dtypes("number").to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
    table = weighted["statistics"]
    block = table[table["variable"] == "x"].set_index("g")
    for label, group in expanded.groupby("g"):
        values = group["x"].to_numpy()
        assert block.loc[label, "n"] == len(values)
        assert block.loc[label, "std_dev"] == pytest.approx(values.std(ddof=1), rel=1e-11)
        assert block.loc[label, "median"] == pytest.approx(
            np.percentile(values, 50, method="averaged_inverted_cdf"), rel=1e-12)
        assert block.loc[label, "skewness"] == pytest.approx(stats.skew(values), rel=1e-10)
    for label, group in frame.groupby("g"):
        _assert_row(block.loc[label], _expected(group["x"], group["f"], "fweight"))
    assert weighted.attrs["weights"] == {"column": "f", "type": "fweight"}
    spss = oe.tabstat(frame, ["x"], stats=["skewness", "kurtosis"], weights="f",
                      weight_type="fweight", moments="spss")
    assert spss.loc["x", "skewness"] == pytest.approx(stats.skew(expanded["x"], bias=False),
                                                      rel=1e-10)
    assert spss.loc["x", "kurtosis"] == pytest.approx(stats.kurtosis(expanded["x"], bias=False),
                                                      rel=1e-10)


def test_analytic_weights_follow_stata_summarize():
    frame = _data()
    table = oe.tabstat(frame, ["x", "z"], by="g", stats=ALL, weights="w")
    assert table.attrs["weights"] == {"column": "w", "type": "aweight"}      # the default type
    for name in ("x", "z"):
        block = table[table["variable"] == name].set_index("g")
        for label, group in [*frame.groupby("g"), ("Total", frame)]:
            x, w = group[name].to_numpy(), group["w"].to_numpy()
            _assert_row(block.loc[label], _expected(x, w, "aweight"))
            # statsmodels with the weights rescaled to sum to the number of rows.
            reference = DescrStatsW(x, weights=w * len(x) / w.sum(), ddof=1)
            assert block.loc[label, "n"] == len(x)
            assert block.loc[label, "mean"] == pytest.approx(reference.mean, rel=1e-12)
            assert block.loc[label, "variance"] == pytest.approx(reference.var, rel=1e-11)
            assert block.loc[label, "std_dev"] == pytest.approx(reference.std, rel=1e-11)
            # Stata's r(sum) = r(mean) * r(sum_w): the weighted total with the raw weights.
            assert block.loc[label, "sum"] == pytest.approx((w * x).sum(), rel=1e-12)
    # Analytic weights are scale free (except the weighted total); equal weights reproduce
    # the unweighted table.
    scaled = oe.tabstat(frame.assign(w=frame["w"] * 37.5), ["x", "z"], by="g", stats=ALL,
                        weights="w", weight_type="aweight")
    free = [name for name in table.select_dtypes("number").columns if name != "sum"]
    np.testing.assert_allclose(scaled[free].to_numpy(dtype=float),
                               table[free].to_numpy(dtype=float), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(scaled["sum"], 37.5 * table["sum"], rtol=1e-12)
    equal = oe.tabstat(frame.assign(w=2.5), ["x", "z"], by="g", stats=ALL, weights="w")
    plain = oe.tabstat(frame, ["x", "z"], by="g", stats=ALL)
    np.testing.assert_allclose(equal[free].to_numpy(dtype=float),
                               plain[free].to_numpy(dtype=float),
                               rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(equal["sum"], 2.5 * plain["sum"], rtol=1e-12)


def test_weighted_anova_uses_normalized_weights():
    frame = _data()
    result = oe.tabstat(frame, ["x"], by="g", weights="w", anova=True)
    v = frame["w"].to_numpy() * len(frame) / frame["w"].sum()
    x = frame["x"].to_numpy()
    grand = (v * x).sum() / v.sum()
    between = within = 0.0
    for label in ("north", "south", "west"):
        mask = (frame["g"] == label).to_numpy()
        mean = (v[mask] * x[mask]).sum() / v[mask].sum()
        between += v[mask].sum() * (mean - grand) ** 2
        within += (v[mask] * (x[mask] - mean) ** 2).sum()
    block = result["anova"].set_index("source")
    assert block.loc["between", "ss"] == pytest.approx(between, rel=1e-10)
    assert block.loc["within", "ss"] == pytest.approx(within, rel=1e-11)
    f = (between / 2) / (within / 237)
    assert block.loc["between", "statistic"] == pytest.approx(f, rel=1e-10)
    assert block.loc["between", "p_value"] == pytest.approx(stats.f.sf(f, 2, 237), rel=1e-9)
    assert result["association"].loc["x", "eta_squared"] == pytest.approx(
        between / (between + within), rel=1e-10)


def test_zero_and_missing_weights_leave_the_sample():
    frame = _data()
    frame.loc[:4, "w"] = 0.0
    frame.loc[5:7, "w"] = np.nan
    table = oe.tabstat(frame, ["x"], weights="w", stats=["n", "mean"])
    kept = frame.iloc[8:]
    assert table.loc["x", "n"] == len(kept) and table.attrs["n"] == len(kept)
    assert table.loc["x", "mean"] == pytest.approx(np.average(kept["x"], weights=kept["w"]))
    assert any("zero weight" in note for note in table.attrs["notes"])


# ---- missing values and degenerate groups -----------------------------------------


def test_missing_values_are_variable_wise_unless_listwise():
    frame = _data()
    frame.loc[:9, "x"] = np.nan
    frame.loc[5:19, "z"] = np.nan
    frame.loc[230:, "g"] = None
    table = oe.tabstat(frame, ["x", "z"], stats=["n", "mean"])
    assert table.loc["x", "n"] == 230 and table.loc["z", "n"] == 225
    assert table.loc["x", "mean"] == pytest.approx(frame["x"].mean())
    listwise = oe.tabstat(frame, ["x", "z"], stats=["n", "mean"], listwise=True)
    complete = frame.dropna(subset=["x", "z"])
    assert listwise.loc["x", "n"] == listwise.loc["z", "n"] == len(complete) == 220
    assert listwise.loc["z", "mean"] == pytest.approx(complete["z"].mean())
    assert listwise.attrs["missing"] == "listwise" and listwise.attrs["n"] == 220
    grouped = oe.tabstat(frame, ["x"], by="g", stats=["n"])
    assert grouped.set_index("g").loc["Total", "n"] == frame.loc[:229, "x"].notna().sum()
    assert grouped.attrs["n"] == 230


def test_undefined_statistics_are_missing_not_errors():
    frame = pd.DataFrame({"v": [4.0, 1.0, 2.0, 3.0, np.nan, np.nan, 5.0, 5.0],
                          "g": ["one", "two", "two", "two", "none", "none", "flat", "flat"]})
    table = oe.tabstat(frame, ["v"], by="g",
                       stats=["n", "mean", "sd", "skewness", "median", "gmedian", "semean"])
    block = table.set_index("g")
    assert block.loc["one", "n"] == 1 and block.loc["one", "mean"] == 4.0
    assert math.isnan(block.loc["one", "std_dev"]) and math.isnan(block.loc["one", "skewness"])
    assert block.loc["one", "median"] == 4.0 and block.loc["one", "grouped_median"] == 4.0
    assert block.loc["none", "n"] == 0
    assert block.loc["none", ["mean", "std_dev", "median", "grouped_median"]].isna().all()
    assert block.loc["flat", "std_dev"] == 0.0 and math.isnan(block.loc["flat", "skewness"])
    assert block.loc["Total", "n"] == 6
    spss = oe.tabstat(frame, ["v"], by="g", moments="spss",
                      stats=["skewness", "kurtosis", "se_kurtosis"]).set_index("g")
    assert math.isnan(spss.loc["two", "kurtosis"]) and not math.isnan(spss.loc["two", "skewness"])
    result = oe.tabstat(frame, ["v"], by="g", anova=True)
    block = result["anova"].set_index("source")
    assert block.loc["between", "df"] == 2 and block.loc["within", "df"] == 3   # 3 groups with data
    numeric = result["statistics"].select_dtypes("number").to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_large_offsets_keep_the_moments():
    rng = np.random.default_rng(8)
    base = rng.gamma(2.0, 1.0, 500)
    frame = pd.DataFrame({"x": base, "big": 1e9 + base, "g": rng.choice(["a", "b"], 500)})
    table = oe.tabstat(frame, ["x", "big"], by="g", stats=["sd", "skewness", "kurtosis"])
    small = table[table["variable"] == "x"].select_dtypes("number").to_numpy(dtype=float)
    large = table[table["variable"] == "big"].select_dtypes("number").to_numpy(dtype=float)
    np.testing.assert_allclose(large, small, rtol=1e-6)


# ---- failures ---------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, code", [
    ({"columns": "x"}, "invalid_spec"),
    ({"columns": []}, "invalid_spec"),
    ({"columns": ["x", "x"]}, "invalid_spec"),
    ({"columns": ["nope"]}, "missing_columns"),
    ({"columns": ["g"]}, "non_numeric_column"),
    ({"columns": ["x"], "by": "x"}, "invalid_spec"),
    ({"columns": ["x"], "by": ["g"]}, "invalid_spec"),
    ({"columns": ["x"], "stats": "mean"}, "invalid_spec"),
    ({"columns": ["x"], "stats": ["mean", "mode"]}, "invalid_option"),
    ({"columns": ["x"], "stats": ["p0"]}, "invalid_option"),
    ({"columns": ["x"], "stats": ["sd", "std_dev"]}, "invalid_spec"),
    ({"columns": ["x"], "stats": ["q", "p50"]}, "invalid_spec"),
    ({"columns": ["x"], "moments": "sas"}, "invalid_option"),
    ({"columns": ["x"], "listwise": 1}, "invalid_option"),
    ({"columns": ["x"], "anova": True}, "invalid_spec"),
    ({"columns": ["x"], "weight_type": "fweight"}, "invalid_spec"),
    ({"columns": ["x"], "weights": "w", "weight_type": "pweight"}, "invalid_option"),
    ({"columns": ["x"], "weights": "w", "weight_type": "fweight"},
     "noninteger_frequency_weights"),
    ({"columns": ["x"], "weights": "g"}, "non_numeric_column"),
    ({"columns": ["x"], "by": "g", "stats": ["mean", "n"], "weights": "neg"}, "negative_weights"),
])
def test_invalid_arguments_raise_analysis_errors(kwargs, code):
    frame = _data(n=30)
    frame["neg"] = -1.0
    with pytest.raises(AnalysisError) as error:
        oe.tabstat(frame, **kwargs)
    assert error.value.code == code
    assert str(error.value)


def test_degenerate_data_raise_analysis_errors():
    frame = _data(n=30)
    with pytest.raises(AnalysisError) as error:
        oe.tabstat(frame.iloc[:0], ["x"])
    assert error.value.code == "empty_data"
    with pytest.raises(AnalysisError) as error:
        oe.tabstat(frame.assign(g=None), ["x"], by="g")
    assert error.value.code == "empty_sample"
    with pytest.raises(AnalysisError) as error:
        oe.tabstat(frame.assign(x=np.inf), ["x"])
    assert error.value.code == "non_finite_values"
    with pytest.raises(AnalysisError) as error:
        oe.tabstat(frame.assign(w=0.0), ["x"], weights="w")
    assert error.value.code == "empty_sample"
    with pytest.raises(AnalysisError) as error:
        oe.tabstat(frame.rename(columns={"g": "mean"}), ["x"], by="mean")
    assert error.value.code == "invalid_spec"
    # A variable without any observation is reported with n = 0.
    table = oe.tabstat(frame.assign(z=np.nan), ["x", "z"], stats=["n", "mean", "median"])
    assert table.loc["z", "n"] == 0 and math.isnan(table.loc["z", "mean"])
    records = frame[["x", "g"]].to_dict("records")
    assert oe.tabstat(records, ["x"]).loc["x", "n"] == 30
