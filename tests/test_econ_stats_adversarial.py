"""Numerical robustness, input types, invariances and documentation of the stats family."""

from __future__ import annotations

import ast
import doctest
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM, anova_lm

import openecon as oe
from openecon.analysis_contracts import AnalysisError

_EXPORTS = ("ttest", "sdtest", "oneway", "anova", "rm_anova", "manova", "correlate", "pcorr",
            "describe")
_LEVEL = 1e9


def _data(seed: int = 21, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"g": rng.choice(["a", "b", "c"], n), "h": rng.choice([True, False], n),
                          "x": rng.normal(size=n)})
    frame["e"] = rng.normal(size=n) + 0.5 * (frame["g"] == "b") + 0.2 * frame["x"]
    frame["e2"] = rng.normal(size=n) - 0.3 * frame["h"]
    frame["y"] = _LEVEL + frame["e"]            # the stored values are exact sums
    frame["e"] = frame["y"] - _LEVEL            # ... so this difference is exact too
    frame["y2"] = _LEVEL + frame["e2"]
    frame["e2"] = frame["y2"] - _LEVEL
    frame["big_x"] = 1e6 * frame["x"] + 3e8
    return frame


def test_large_levels_cost_no_digits():
    frame = _data()
    samples = [group["e"].to_numpy() for _, group in frame.groupby("g")]
    assert oe.oneway(frame, "y", "g").attrs["statistic"] == pytest.approx(
        stats.f_oneway(*samples).statistic, rel=1e-11)
    a, b = frame.loc[~frame["h"], "e"], frame.loc[frame["h"], "e"]
    independent = oe.ttest(frame, "y", by="h")
    assert independent.attrs["groups"] == [False, True]
    assert independent.attrs["statistic"] == pytest.approx(stats.ttest_ind(a, b).statistic,
                                                           rel=1e-11)
    assert independent["statistics"].loc["False", "mean"] == pytest.approx(_LEVEL + a.mean(),
                                                                           rel=1e-15)
    one = oe.ttest(frame, "y", mu=_LEVEL + 0.125)             # exactly representable
    assert one.attrs["statistic"] == pytest.approx(stats.ttest_1samp(frame["e"], 0.125).statistic,
                                                   rel=1e-11)
    paired = oe.ttest(frame, "y", paired_with="y2")
    assert paired.attrs["statistic"] == pytest.approx(
        stats.ttest_rel(frame["e"], frame["e2"]).statistic, rel=1e-11)
    levene = oe.sdtest(frame, "y", by="g")["robust"].loc["w50", "statistic"]
    assert levene == pytest.approx(stats.levene(*samples, center="median").statistic, rel=1e-10)
    reference = anova_lm(smf.ols("e ~ x + C(g, Sum) * C(h, Sum)", frame).fit(), typ=3)
    table = oe.anova(frame, "y", ["g", "h"], covariates=["big_x"], emmeans=["g"])
    np.testing.assert_allclose(
        table["anova"].loc[["big_x", "g", "h", "g#h", "error"], "ss"],
        reference.loc[["x", "C(g, Sum)", "C(h, Sum)", "C(g, Sum):C(h, Sum)", "Residual"],
                      "sum_sq"], rtol=1e-9)
    centred = oe.anova(frame, "e", ["g", "h"], covariates=["x"], emmeans=["g"])
    np.testing.assert_allclose(table["emmeans_g"]["mean"] - _LEVEL, centred["emmeans_g"]["mean"],
                               atol=1e-6)
    np.testing.assert_allclose(table["emmeans_g"]["std_error"],
                               centred["emmeans_g"]["std_error"], rtol=1e-9)
    # The intercept and the uncorrected total are on the original scale.
    raw = smf.ols("y ~ C(g, Sum)", frame).fit()
    intercept = oe.anova(frame, "y", ["g"])["anova"]
    assert intercept.loc["Intercept", "statistic"] == pytest.approx(raw.tvalues["Intercept"] ** 2,
                                                                    rel=1e-6)
    assert intercept.loc["total", "ss"] == pytest.approx((frame["y"] ** 2).sum(), rel=1e-14)
    multivariate = oe.manova(frame, ["y", "y2"], ["g"])["multivariate"].set_index(
        ["effect", "test"])
    check = MANOVA.from_formula("e + e2 ~ C(g, Sum)", frame).mv_test().results["C(g, Sum)"]["stat"]
    assert multivariate.loc[("g", "wilks"), "value"] == pytest.approx(
        float(check.loc["Wilks' lambda", "Value"]), rel=1e-10)
    partial = oe.pcorr(frame, "y", ["big_x"], controls=["y2"])
    centred = oe.pcorr(frame, "e", ["x"], controls=["e2"])
    assert partial.loc["big_x", "partial_corr"] == pytest.approx(centred.loc["x", "partial_corr"],
                                                                 rel=1e-9)
    described = oe.describe(frame, ["y"], stats=["mean", "sd", "skewness", "kurtosis", "min"])
    assert described.loc["y", "std_dev"] == pytest.approx(frame["e"].std(), rel=1e-12)
    assert described.loc["y", "skewness"] == pytest.approx(stats.skew(frame["e"], bias=False),
                                                           rel=1e-10)
    assert described.loc["y", "min"] == frame["y"].min()
    correlation = oe.correlate(frame, ["y", "big_x"])["coefficients"].loc["y", "big_x"]
    assert correlation == pytest.approx(np.corrcoef(frame["e"], frame["x"])[0, 1], rel=1e-10)


def test_tiny_real_variation_is_not_mistaken_for_a_constant():
    # 1e9 +- 1e-4 varies by about a thousand units in the last place: a real sample.
    steps = np.array([-3.0, 1.0, 2.0, -1.0, 4.0, -2.0, 0.0, 5.0]) * 2.0 ** -13
    frame = pd.DataFrame({"y": _LEVEL + steps, "g": list("aabbccdd")})
    exact = stats.ttest_1samp(steps, 2.0 ** -14)
    result = oe.ttest(frame, "y", mu=_LEVEL + 2.0 ** -14)
    assert result.attrs["statistic"] == pytest.approx(exact.statistic, rel=1e-11)
    groups = [steps[i:i + 2] for i in range(0, 8, 2)]
    assert oe.oneway(frame, "y", "g").attrs["statistic"] == pytest.approx(
        stats.f_oneway(*groups).statistic, rel=1e-11)
    for constant in (0.1, 1e9 + 0.1, -7.3e-8):
        with pytest.raises(AnalysisError) as error:
            oe.ttest({"y": [constant] * 7}, "y", mu=1.0)
        assert error.value.code == "zero_variance"
        with pytest.raises(AnalysisError) as error:
            oe.oneway({"y": [constant] * 6, "g": list("aabbcc")}, "y", "g")
        assert error.value.code == "zero_variance"


def test_repeated_measures_with_a_large_level():
    rng = np.random.default_rng(3)
    rows = list(itertools.product(range(12), range(4)))
    frame = pd.DataFrame(rows, columns=["id", "t"])
    frame["e"] = rng.normal(size=12)[frame["id"]] + 0.3 * frame["t"] + rng.normal(size=48)
    frame["y"] = _LEVEL + frame["e"]
    frame["e"] = frame["y"] - _LEVEL
    result = oe.rm_anova(frame, "y", "id", ["t"])
    reference = AnovaRM(frame, "e", "id", within=["t"]).fit().anova_table
    row = result["within"].iloc[0]
    assert row["statistic"] == pytest.approx(reference.loc["t", "F Value"], rel=1e-10)
    centred = oe.rm_anova(frame, "e", "id", ["t"])
    np.testing.assert_allclose(result["sphericity"], centred["sphericity"], rtol=1e-8)
    assert result["between"].loc["Intercept", "ss"] == pytest.approx(
        4 * 12 * frame["y"].mean() ** 2, rel=1e-12)
    assert result["descriptives"].loc[0, "mean"] == pytest.approx(
        frame.loc[frame["t"] == 0, "y"].mean(), rel=1e-15)


def test_scale_and_row_order_invariance():
    frame = _data()
    base = oe.anova(frame, "e", ["g", "h"], covariates=["x"])["anova"]
    for factor in (1e-9, 1e7):
        scaled = frame.assign(e=frame["e"] * factor, x=frame["x"] / factor)
        table = oe.anova(scaled, "e", ["g", "h"], covariates=["x"])["anova"]
        np.testing.assert_allclose(table["statistic"].dropna(), base["statistic"].dropna(),
                                   rtol=1e-9)
    shuffled = frame.sample(frac=1.0, random_state=5)
    np.testing.assert_allclose(oe.anova(shuffled, "e", ["g", "h"], covariates=["x"])["anova"]["ss"],
                               base["ss"], rtol=1e-10)
    first = oe.correlate(frame, ["e", "x", "e2"], method="kendall")["coefficients"]
    second = oe.correlate(shuffled, ["e", "x", "e2"], method="kendall")["coefficients"]
    np.testing.assert_allclose(first, second, rtol=1e-13)
    one = oe.oneway(frame, "e", "g", posthoc=["tukey"])["posthoc_tukey"]
    two = oe.oneway(shuffled, "e", "g", posthoc=["tukey"])["posthoc_tukey"]
    np.testing.assert_allclose(one["p_value"], two["p_value"], rtol=1e-9)


def test_input_types_nullable_categorical_boolean_and_integer():
    frame = pd.DataFrame({
        "y": pd.array([1, 2, None, 4, 5, 7, 8, 9, 3, 6], dtype="Int64"),
        "g": pd.Categorical(list("aabbccddab")),
        "flag": pd.array([True, False, True, False, True, False, True, None, True, False],
                         dtype="boolean"),
        "level": [10, 2, 10, 2, 10, 2, 10, 2, 10, 2]})
    result = oe.oneway(frame, "y", "g")
    values = frame.dropna(subset=["y"])
    samples = [group["y"].to_numpy(dtype=float) for _, group in values.groupby("g", observed=True)]
    assert result.attrs["n"] == 9 and result.attrs["groups"] == ["a", "b", "c", "d"]
    assert result.attrs["statistic"] == pytest.approx(stats.f_oneway(*samples).statistic)
    by_flag = oe.ttest(frame, "y", by="flag")
    assert by_flag.attrs["n"] == 8 and by_flag.attrs["n_missing"] == 2
    by_level = oe.ttest(frame, "y", by="level")
    assert by_level.attrs["groups"] == [2, 10]
    described = oe.describe(frame, ["y", "level"], by="g")
    assert described["n"].sum() == 19
    assert oe.correlate(frame, ["y", "level"])["n"].loc["y", "level"] == 9
    text = pd.DataFrame({"y": ["1", "2", "3"], "g": ["a", "b", "a"]})
    for call in (lambda: oe.oneway(text, "y", "g"), lambda: oe.describe(text, ["y"]),
                 lambda: oe.correlate(text, ["y", "g"]), lambda: oe.anova(text, "y", ["g"])):
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == "non_numeric_column"
    with pytest.raises(AnalysisError) as error:
        oe.ttest("not a table", "y")
    assert error.value.code == "invalid_data"
    with pytest.raises(AnalysisError) as error:
        oe.oneway(pd.DataFrame({"y": [1.0, 2.0], "g": ["a", "b"]}).iloc[:0], "y", "g")
    assert error.value.code == "empty_data"


def test_input_tables_are_not_modified():
    frame = _data()
    frame.loc[3, "e"] = np.nan
    snapshot = frame.copy(deep=True)
    oe.ttest(frame, "e", by="h")
    oe.oneway(frame, "e", "g", posthoc=["tukey"])
    oe.anova(frame, "e", ["g", "h"], covariates=["x"], emmeans=["g"])
    oe.manova(frame, ["e", "e2"], ["g"])
    oe.correlate(frame, ["e", "x"], method="spearman")
    oe.describe(frame, ["e", "x"], by="g")
    pd.testing.assert_frame_equal(frame, snapshot)


def test_moderately_large_samples_run_quickly():
    rng = np.random.default_rng(9)
    n = 200_000
    frame = pd.DataFrame({"a": rng.integers(0, 4, n), "b": rng.integers(0, 3, n),
                          "x": rng.normal(size=n), "y2": rng.normal(size=n)})
    frame["y"] = rng.normal(size=n) + 0.02 * frame["a"] + 0.1 * frame["x"]
    start = time.perf_counter()
    oe.ttest(frame[frame["a"] < 2], "y", by="a")
    oe.sdtest(frame, "y", by="a")
    oe.oneway(frame, "y", "a", posthoc=["tukey", "games_howell", "dunnett"])
    table = oe.anova(frame, "y", ["a", "b"], covariates=["x"], emmeans=["a"])["anova"]
    oe.manova(frame, ["y", "y2"], ["a", "b"])
    oe.correlate(frame, ["y", "x", "y2"], method="kendall")
    oe.correlate(frame, ["y", "x", "y2"], method="spearman")
    oe.pcorr(frame, "y", ["x"], controls=["y2"])
    oe.describe(frame, ["y", "x"], by="a")
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0
    fit = smf.ols("y ~ x + C(a, Sum) * C(b, Sum)", frame).fit()
    assert table.loc["a", "statistic"] == pytest.approx(
        anova_lm(fit, typ=3).loc["C(a, Sum)", "F"], rel=1e-8)


@pytest.mark.parametrize("name", _EXPORTS)
def test_docstring_examples_run(name):
    function = getattr(oe, name)
    assert "Example" in function.__doc__ and "Parameters" in function.__doc__
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    tests = finder.find(function, name, globs={})
    assert tests and tests[0].examples
    for test in tests:
        result = runner.run(test)
        assert result.failed == 0 and result.attempted >= 3


def test_documentation_covers_every_function_and_the_manifest_is_light():
    page = (Path(__file__).resolve().parents[1] / "docs" / "econometrics" / "stats.md").read_text()
    for name in _EXPORTS:
        assert f"`{name}`" in page and f"oe.{name}(" in page
    for heading in ("Limitations", "Conventions that are not certain", "Performance"):
        assert heading in page
    import openecon.econometrics.stats as manifest

    planning = {"power_mean", "power_proportion", "power_correlation", "precision_mean",
                "power_paired_mean", "power_two_proportions", "power_two_correlations", "power_slope", "power_logrank", "power_mcnemar", "precision_mean_unknown", "precision_variance"}
    power_designs = {"power_tmean", "power_ttwomeans", "power_tpaired", "power_anova",
                     "power_regression", "power_cluster_mean", "power_gof", "power_independence"}
    assert manifest.ESTIMATORS == () and set(manifest.EXPORTS) == (
        set(_EXPORTS) | planning | power_designs | {"planning_scenarios", "planning_plot"}
    )
    designs_page = (Path(manifest.__file__).resolve().parents[4]
                    / "docs" / "econometrics" / "power-designs.md").read_text()
    for name in power_designs:
        assert f"{name}(" in designs_page
    planning_page = (Path(manifest.__file__).resolve().parents[4]
                     / "docs" / "econometrics" / "planning.md").read_text()
    for name in planning:
        assert f"oe.{name}" in planning_page
    source = Path(manifest.__file__).read_text()
    assert "import torch" not in source and "import pandas" not in source


def test_kernels_use_no_forbidden_libraries():
    package = Path(oe.__file__).parent / "econometrics" / "stats"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [(node.module or "").split(".")[0]]
            else:
                imported = []
            assert not set(imported) & {"numpy", "scipy", "statsmodels", "sklearn", "autograd"}, path.name
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"autograd", "requires_grad"}, path.name
