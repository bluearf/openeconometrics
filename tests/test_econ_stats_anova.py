"""Factorial ANOVA / ANCOVA, repeated measures and MANOVA against statsmodels and NumPy."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import patsy
import pytest
import statsmodels.formula.api as smf
from scipy import stats
from scipy.linalg import helmert
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM, anova_lm

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def _factorial(seed: int = 3, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.choice(["u", "v", "w"], n, p=[0.5, 0.3, 0.2]),
        "b": rng.choice(["p", "q"], n, p=[0.35, 0.65]),
        "c": rng.choice([1, 2, 3, 4], n), "x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["y"] = 1 + 0.5 * (frame["a"] == "v") - 0.4 * (frame["b"] == "q") + 0.3 * frame["x"] \
        + 0.2 * (frame["a"] == "w") * (frame["b"] == "q") + 0.1 * frame["c"] + rng.normal(size=n)
    return frame


def _compare(table: pd.DataFrame, reference: pd.DataFrame, names: dict[str, str]) -> None:
    for ours, theirs in names.items():
        assert table.loc[ours, "ss"] == pytest.approx(reference.loc[theirs, "sum_sq"], rel=1e-9)
        assert table.loc[ours, "df"] == reference.loc[theirs, "df"]
        if theirs != "Residual":
            assert table.loc[ours, "statistic"] == pytest.approx(reference.loc[theirs, "F"],
                                                                 rel=1e-9)
            assert table.loc[ours, "p_value"] == pytest.approx(reference.loc[theirs, "PR(>F)"],
                                                               rel=1e-8, abs=1e-300)


_NAMES = {"a": "C(a, Sum)", "b": "C(b, Sum)", "a#b": "C(a, Sum):C(b, Sum)", "x": "x",
          "error": "Residual"}


def test_type_three_and_two_match_statsmodels_on_unbalanced_ancova():
    frame = _factorial()
    fit = smf.ols("y ~ x + C(a, Sum) * C(b, Sum)", frame).fit()
    third = oe.anova(frame, "y", ["a", "b"], covariates=["x"])
    _compare(third["anova"], anova_lm(fit, typ=3), {**_NAMES, "Intercept": "Intercept"})
    second = oe.anova(frame, "y", ["a", "b"], covariates=["x"], ss_type=2)
    _compare(second["anova"], anova_lm(fit, typ=2), _NAMES)
    table = third["anova"]
    assert list(table.index) == ["corrected_model", "Intercept", "x", "a", "b", "a#b", "error",
                                 "total", "corrected_total"]
    assert third.attrs["r_squared"] == pytest.approx(fit.rsquared)
    assert third.attrs["adjusted_r_squared"] == pytest.approx(fit.rsquared_adj)
    assert third.attrs["rmse"] == pytest.approx(np.sqrt(fit.mse_resid))
    assert table.loc["corrected_model", "statistic"] == pytest.approx(fit.fvalue)
    assert table.loc["corrected_model", "p_value"] == pytest.approx(fit.f_pvalue, rel=1e-8)
    assert table.loc["total", "ss"] == pytest.approx((frame["y"] ** 2).sum())
    assert table.loc["corrected_total", "ss"] == pytest.approx(fit.centered_tss)
    eta = table.loc["a", "ss"] / (table.loc["a", "ss"] + table.loc["error", "ss"])
    assert table.loc["a", "partial_eta_squared"] == pytest.approx(eta)
    assert third.attrs["df_resid"] == fit.df_resid and third.attrs["df_model"] == fit.df_model


def test_type_one_is_sequential_in_model_order():
    frame = _factorial()
    y = frame["y"].to_numpy()
    blocks = [np.ones((len(frame), 1)), frame[["x"]].to_numpy(),
              np.asarray(patsy.dmatrix("0 + C(a, Sum)", frame))[:, :0]]
    full = np.asarray(patsy.dmatrix("x + C(a, Sum) * C(b, Sum)", frame))
    info = patsy.dmatrix("x + C(a, Sum) * C(b, Sum)", frame).design_info
    order = ["Intercept", "x", "C(a, Sum)", "C(b, Sum)", "C(a, Sum):C(b, Sum)"]
    blocks = [full[:, info.slice(name)] for name in order]

    def ssr(columns: list[np.ndarray]) -> float:
        design = np.column_stack(columns)
        return float(((y - design @ np.linalg.lstsq(design, y, rcond=None)[0]) ** 2).sum())

    table = oe.anova(frame, "y", ["a", "b"], covariates=["x"], ss_type=1)["anova"]
    previous = float((y ** 2).sum())
    for label, stop in zip(["Intercept", "x", "a", "b", "a#b"], range(1, 6), strict=True):
        current = ssr(blocks[:stop])
        assert table.loc[label, "ss"] == pytest.approx(previous - current, rel=1e-9)
        previous = current
    assert table.loc["error", "ss"] == pytest.approx(previous)
    assert table.loc["Intercept", "ss"] == pytest.approx(len(y) * y.mean() ** 2)


def test_three_way_factorial_with_interaction_list_and_main_effects_only():
    frame = _factorial()
    fit = smf.ols("y ~ C(a, Sum) * C(b, Sum) * C(c, Sum)", frame).fit()
    table = oe.anova(frame, "y", ["a", "b", "c"])["anova"]
    reference = anova_lm(fit, typ=3)
    _compare(table, reference, {
        "a": "C(a, Sum)", "c": "C(c, Sum)", "a#c": "C(a, Sum):C(c, Sum)",
        "b#c": "C(b, Sum):C(c, Sum)", "a#b#c": "C(a, Sum):C(b, Sum):C(c, Sum)",
        "error": "Residual"})
    partial = smf.ols("y ~ C(a, Sum) + C(b, Sum) + C(c, Sum) + C(a, Sum):C(c, Sum)", frame).fit()
    for spec in ([["a", "c"]], ["a#c"], ["a*c"], [("a", "c")]):
        table = oe.anova(frame, "y", ["a", "b", "c"], interactions=spec)["anova"]
        _compare(table, anova_lm(partial, typ=3), {"a": "C(a, Sum)", "b": "C(b, Sum)",
                                                   "a#c": "C(a, Sum):C(c, Sum)"})
    main = smf.ols("y ~ C(a, Sum) + C(b, Sum) + C(c, Sum)", frame).fit()
    table = oe.anova(frame, "y", ["a", "b", "c"], interactions="none")["anova"]
    _compare(table, anova_lm(main, typ=2), {"a": "C(a, Sum)", "b": "C(b, Sum)", "c": "C(c, Sum)"})


def test_type_two_containment_follows_the_spss_and_sas_rule():
    # "a" is not contained in "a#x" (different covariates), so its Type II sum of
    # squares is adjusted for a#x; "x" is contained in a#x and is not.
    frame = _factorial()
    table = oe.anova(frame, "y", ["a", "b"], covariates=["x"], ss_type=2,
                     interactions=[["a", "b"], ["a", "x"]])["anova"]
    design = patsy.dmatrix("x + C(a, Sum) * C(b, Sum) + C(a, Sum):x", frame)
    info, full = design.design_info, np.asarray(design)
    y = frame["y"].to_numpy()

    def ssr(names: list[str]) -> float:
        columns = np.column_stack([full[:, info.slice(name)] for name in names])
        return float(((y - columns @ np.linalg.lstsq(columns, y, rcond=None)[0]) ** 2).sum())

    a, b, ab, ax = "C(a, Sum)", "C(b, Sum)", "C(a, Sum):C(b, Sum)", "C(a, Sum):x"
    assert table.loc["a", "ss"] == pytest.approx(
        ssr(["Intercept", "x", b, ax]) - ssr(["Intercept", "x", a, b, ax]))
    assert table.loc["x", "ss"] == pytest.approx(
        ssr(["Intercept", a, b, ab]) - ssr(["Intercept", "x", a, b, ab]))
    assert table.loc["a#x", "ss"] == pytest.approx(
        ssr(["Intercept", "x", a, b, ab]) - ssr(["Intercept", "x", a, b, ab, ax]))


def test_balanced_design_gives_identical_types_and_textbook_sums():
    cells = list(itertools.product("uv", "pqr"))
    rng = np.random.default_rng(8)
    frame = pd.DataFrame([(a, b) for a, b in cells for _ in range(5)], columns=["a", "b"])
    frame["y"] = rng.normal(size=len(frame)) + (frame["a"] == "v") * 0.7
    tables = [oe.anova(frame, "y", ["a", "b"], ss_type=kind)["anova"] for kind in (1, 2, 3)]
    for other in tables[1:]:
        np.testing.assert_allclose(other.loc[["a", "b", "a#b"], "ss"],
                                   tables[0].loc[["a", "b", "a#b"], "ss"], rtol=1e-10)
    grand = frame["y"].mean()
    ss_a = 15 * ((frame.groupby("a")["y"].mean() - grand) ** 2).sum()
    ss_b = 10 * ((frame.groupby("b")["y"].mean() - grand) ** 2).sum()
    assert tables[2].loc["a", "ss"] == pytest.approx(ss_a)
    assert tables[2].loc["b", "ss"] == pytest.approx(ss_b)


def test_one_factor_anova_equals_oneway_and_homogeneity_of_slopes():
    frame = _factorial()
    table = oe.anova(frame, "y", ["a"])["anova"]
    one = oe.oneway(frame, "y", "a")
    assert table.loc["a", "statistic"] == pytest.approx(one.attrs["statistic"])
    assert table.loc["a", "partial_eta_squared"] == pytest.approx(one.attrs["eta_squared"])
    fit = smf.ols("y ~ x + C(a, Sum) + C(a, Sum):x", frame).fit()
    slopes = oe.anova(frame, "y", ["a"], covariates=["x"], homogeneity_of_slopes=True)["anova"]
    _compare(slopes, anova_lm(fit, typ=3), {"a": "C(a, Sum)", "x": "x", "a#x": "C(a, Sum):x"})
    listed = oe.anova(frame, "y", ["a"], covariates=["x"], interactions=[["a", "x"]])["anova"]
    np.testing.assert_allclose(listed["ss"], slopes["ss"], rtol=1e-12)
    regression = oe.anova(frame, "y", covariates=["x", "z"])["anova"]
    plain = smf.ols("y ~ x + z", frame).fit()
    assert regression.loc["x", "statistic"] == pytest.approx(plain.tvalues["x"] ** 2)


def test_estimated_marginal_means_and_pairwise_comparisons():
    frame = _factorial()
    fit = smf.ols("y ~ C(a, Sum) * C(b, Sum) * C(c, Sum) + x", frame).fit()
    result = oe.anova(frame, "y", ["a", "b", "c"], covariates=["x"], emmeans=["a", ["a", "b"]],
                      adjust="sidak")
    grid = pd.DataFrame(list(itertools.product("uvw", "pq", [1, 2, 3, 4])), columns=["a", "b", "c"])
    grid["x"] = frame["x"].mean()
    design = np.asarray(patsy.dmatrix(fit.model.data.design_info, grid))
    rows = {level: design[(grid["a"] == level).to_numpy()].mean(0) for level in "uvw"}
    emm = result["emmeans_a"].set_index("a")
    for level, vector in rows.items():
        test = fit.t_test(vector)
        assert emm.loc[level, "mean"] == pytest.approx(float(test.effect[0]))
        assert emm.loc[level, "std_error"] == pytest.approx(float(test.sd[0, 0]))
        low, high = test.conf_int()[0]
        assert (emm.loc[level, "ci_low"], emm.loc[level, "ci_high"]) == pytest.approx((low, high))
    pairs = result["pairwise_a"]
    assert pairs.attrs["adjustment"] == "sidak"
    for _, row in pairs.iterrows():
        test = fit.t_test(rows[row["level_i"]] - rows[row["level_j"]])
        assert row["mean_difference"] == pytest.approx(float(test.effect[0]))
        assert row["statistic"] == pytest.approx(float(test.tvalue[0, 0]))
        raw = float(test.pvalue)
        assert row["p_value"] == pytest.approx(1 - (1 - raw) ** 3)
        half = stats.t.ppf(1 - (1 - 0.95 ** (1 / 3)) / 2, fit.df_resid) * float(test.sd[0, 0])
        assert row["ci_high"] - row["mean_difference"] == pytest.approx(half)
    cell = result["emmeans_a#b"]
    assert len(cell) == 6 and list(cell.columns[:2]) == ["a", "b"]
    vector = design[((grid["a"] == "w") & (grid["b"] == "q")).to_numpy()].mean(0)
    assert cell.iloc[5]["mean"] == pytest.approx(float(fit.t_test(vector).effect[0]))
    bonferroni = oe.anova(frame, "y", ["a", "b"], emmeans=["a"])["pairwise_a"]
    lsd = oe.anova(frame, "y", ["a", "b"], emmeans=["a"], adjust="lsd")["pairwise_a"]
    np.testing.assert_allclose(bonferroni["p_value"], np.minimum(1, 3 * lsd["p_value"]))


def test_anova_missing_policy_and_serialization():
    frame = _factorial()
    frame.loc[[1, 5], "y"] = np.nan
    frame.loc[7, "a"] = None
    frame.loc[9, "x"] = np.nan
    result = oe.anova(frame, "y", ["a", "b"], covariates=["x"])
    complete = frame.dropna(subset=["y", "a", "b", "x"])
    assert result.attrs["n_missing"] == 4 and result.attrs["n"] == len(complete)
    again = oe.anova(complete, "y", ["a", "b"], covariates=["x"])
    np.testing.assert_allclose(result["anova"]["ss"], again["anova"]["ss"])
    with pytest.raises(AnalysisError) as error:
        oe.anova(frame, "y", ["a", "b"], covariates=["x"], missing="raise")
    assert error.value.code == "missing_values"
    assert json.loads(json.dumps(result.attrs))["terms"] == ["Intercept", "x", "a", "b", "a#b"]
    assert "Type III" in str(result) and r"\begin{tabular}" in result.to_latex()


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"factors": ["a", "hole"]}, "empty_cells"),
    ({"factors": ["a"], "covariates": ["x", "twin"]}, "collinear_design"),
    ({"factors": ["a", "same"]}, "single_level"),
    ({"factors": ["a"], "ss_type": 4}, "invalid_option"),
    ({"factors": ["a"], "interactions": "all"}, "invalid_option"),
    ({"factors": ["a", "b"], "interactions": [["a", "nope"]]}, "invalid_spec"),
    ({"factors": ["a", "b"], "interactions": [["a", "b"], ["b", "a"]]}, "invalid_spec"),
    ({"factors": "a"}, "invalid_spec"), ({"factors": []}, "invalid_spec"),
    ({"factors": ["a"], "emmeans": ["b"]}, "invalid_spec"),
    ({"factors": ["a"], "emmeans": "a"}, "invalid_spec"),
    ({"factors": ["a"], "adjust": "tukey"}, "invalid_option"),
    ({"factors": ["a"], "covariates": ["b"]}, "non_numeric_column"),
    ({"factors": ["a"], "within": ["b"]}, "invalid_spec"),
    ({"factors": ["a", "a"]}, "invalid_spec"),
    ({"factors": ["a"], "covariates": ["y"]}, "invalid_spec"),
])
def test_anova_error_codes(kwargs, code):
    frame = _factorial()
    frame["hole"] = np.where((frame["a"] == "u"), "only", np.where(frame["b"] == "p", "s", "t"))
    frame["twin"] = 2 * frame["x"] + 1
    frame["same"] = "k"
    with pytest.raises(AnalysisError) as error:
        oe.anova(frame, "y", **kwargs)
    assert error.value.code == code


def test_main_effects_model_allows_empty_cells_and_saturated_model_is_rejected():
    frame = _factorial()
    frame = frame[~((frame["a"] == "w") & (frame["b"] == "p"))]
    fit = smf.ols("y ~ C(a, Sum) + C(b, Sum)", frame).fit()
    table = oe.anova(frame, "y", ["a", "b"], interactions="none")["anova"]
    _compare(table, anova_lm(fit, typ=2), {"a": "C(a, Sum)", "b": "C(b, Sum)"})
    tiny = {"y": [1.0, 2.0, 3.0, 4.5], "a": ["u", "u", "v", "v"], "b": ["p", "q", "p", "q"]}
    with pytest.raises(AnalysisError) as error:
        oe.anova(tiny, "y", ["a", "b"])
    assert error.value.code == "no_residual_df"
    exact = {"y": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "a": list("uuvvww")}
    with pytest.raises(AnalysisError) as error:
        oe.anova(exact, "y", ["a"])
    assert error.value.code == "perfect_fit"


# ---- repeated measures ------------------------------------------------------------


def _repeated(seed: int = 5, subjects: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = list(itertools.product(range(subjects), range(3), range(4)))
    frame = pd.DataFrame(rows, columns=["id", "A", "B"])
    effect = rng.normal(size=subjects)
    frame["y"] = effect[frame["id"]] + 0.3 * frame["A"] + 0.2 * frame["B"] * (frame["A"] == 1) \
        + rng.normal(size=len(frame)) * (1 + 0.3 * frame["B"])
    frame["g"] = np.where(frame["id"] < 5, "x", np.where(frame["id"] < 10, "y", "z"))
    return frame.sample(frac=1.0, random_state=1).reset_index(drop=True)


def _assumed(result, source: str) -> pd.Series:
    table = result["within"]
    return table[(table["source"] == source)
                 & (table["correction"] == "sphericity_assumed")].iloc[0]


def test_rm_anova_two_within_factors_matches_statsmodels():
    frame = _repeated()
    result = oe.rm_anova(frame, "y", "id", ["A", "B"])
    reference = AnovaRM(frame, "y", "id", within=["A", "B"]).fit().anova_table
    for ours, theirs in (("A", "A"), ("B", "B"), ("A#B", "A:B")):
        row = _assumed(result, ours)
        assert row["statistic"] == pytest.approx(reference.loc[theirs, "F Value"], rel=1e-10)
        assert row["df"] == reference.loc[theirs, "Num DF"]
        assert row["p_value"] == pytest.approx(reference.loc[theirs, "Pr > F"], rel=1e-9)
        assert _assumed(result, f"error({ours})")["df"] == reference.loc[theirs, "Den DF"]
    assert result.attrs["n_subjects"] == 14 and result.attrs["within_cells"] == 12
    assert set(result) == {"within", "sphericity", "between", "descriptives"}
    cell = result["descriptives"]
    first = frame[(frame["A"] == 0) & (frame["B"] == 0)]["y"]
    assert cell.iloc[0][["n", "mean", "std_dev"]].tolist() == pytest.approx(
        [14, first.mean(), first.std()])


def test_sphericity_statistics_match_covariance_formulas():
    frame = _repeated()
    one = frame[frame["A"] == 0]
    result = oe.rm_anova(one, "y", "id", ["B"])
    wide = one.pivot(index="id", columns="B", values="y").to_numpy()
    n, p = wide.shape
    d = p - 1
    cov = np.cov(wide, rowvar=False)
    # Box's epsilon from the covariance matrix (Greenhouse and Geisser 1959).
    gg = (p * (np.diag(cov).mean() - cov.mean())) ** 2 / (
        d * ((cov ** 2).sum() - 2 * p * (cov.mean(1) ** 2).sum() + p ** 2 * cov.mean() ** 2))
    hf = min(1.0, (n * d * gg - 2) / (d * (n - 1 - d * gg)))
    contrasts = helmert(p).T
    reduced = contrasts.T @ cov @ contrasts
    w = np.linalg.det(reduced) / (np.trace(reduced) / d) ** d
    chi2 = -(n - 1 - (2 * d * d + d + 2) / (6 * d)) * np.log(w)
    row = result["sphericity"].loc["B"]
    assert row["epsilon_gg"] == pytest.approx(gg) and row["epsilon_hf"] == pytest.approx(hf)
    assert row["epsilon_lb"] == pytest.approx(1 / d)
    assert row["mauchly_w"] == pytest.approx(w) and row["chi2"] == pytest.approx(chi2)
    assert row["df"] == d * (d + 1) // 2 - 1
    assert row["p_value"] == pytest.approx(stats.chi2.sf(chi2, row["df"]))
    table = result["within"]
    f = _assumed(result, "B")["statistic"]
    for name, eps in (("greenhouse_geisser", gg), ("huynh_feldt", hf), ("lower_bound", 1 / d)):
        effect = table[(table["source"] == "B") & (table["correction"] == name)].iloc[0]
        error = table[(table["source"] == "error(B)") & (table["correction"] == name)].iloc[0]
        assert effect["df"] == pytest.approx(d * eps) and error["df"] == pytest.approx(
            d * (n - 1) * eps)
        assert effect["statistic"] == pytest.approx(f)
        assert effect["p_value"] == pytest.approx(stats.f.sf(f, d * eps, d * (n - 1) * eps))
    eta = _assumed(result, "B")
    assert eta["partial_eta_squared"] == pytest.approx(
        eta["ss"] / (eta["ss"] + _assumed(result, "error(B)")["ss"]))


def test_two_level_within_factor_is_a_paired_t_test():
    frame = _repeated()
    two = frame[(frame["A"] == 0) & (frame["B"] < 2)]
    result = oe.rm_anova(two, "y", "id", ["B"])
    wide = two.pivot(index="id", columns="B", values="y")
    t = stats.ttest_rel(wide[0], wide[1])
    assert _assumed(result, "B")["statistic"] == pytest.approx(t.statistic ** 2)
    assert _assumed(result, "B")["p_value"] == pytest.approx(t.pvalue)
    row = result["sphericity"].loc["B"]
    assert row["epsilon_gg"] == 1.0 and row["mauchly_w"] == 1.0 and np.isnan(row["p_value"])


def test_split_plot_with_unequal_groups_matches_contrast_regressions():
    frame = _repeated()
    one = frame[frame["A"] == 0]
    result = oe.rm_anova(one, "y", "id", ["B"], between=["g"])
    wide = one.pivot(index="id", columns="B", values="y")
    groups = one.drop_duplicates("id").set_index("id")["g"].loc[wide.index]
    contrasts = helmert(4).T
    transformed = wide.to_numpy() @ contrasts
    ss = {"Intercept": 0.0, "C(g, Sum)": 0.0, "Residual": 0.0}
    for column in range(3):
        data = pd.DataFrame({"t": transformed[:, column], "g": groups.to_numpy()})
        table = anova_lm(smf.ols("t ~ C(g, Sum)", data).fit(), typ=3)
        for key in ss:
            ss[key] += table.loc[key, "sum_sq"]
    assert _assumed(result, "B")["ss"] == pytest.approx(ss["Intercept"])
    assert _assumed(result, "B#g")["ss"] == pytest.approx(ss["C(g, Sum)"])
    assert _assumed(result, "error(B)")["ss"] == pytest.approx(ss["Residual"])
    assert _assumed(result, "B#g")["df"] == 6 and _assumed(result, "error(B)")["df"] == 33
    f = (ss["C(g, Sum)"] / 6) / (ss["Residual"] / 33)
    assert _assumed(result, "B#g")["statistic"] == pytest.approx(f)
    means = pd.DataFrame({"m": wide.mean(1).to_numpy() * 2.0, "g": groups.to_numpy()})
    between = anova_lm(smf.ols("m ~ C(g, Sum)", means).fit(), typ=3)
    table = result["between"]
    assert table.loc["g", "ss"] == pytest.approx(between.loc["C(g, Sum)", "sum_sq"])
    assert table.loc["g", "statistic"] == pytest.approx(between.loc["C(g, Sum)", "F"])
    assert table.loc["Intercept", "ss"] == pytest.approx(between.loc["Intercept", "sum_sq"])
    assert table.loc["error", "df"] == 11
    # Pooled within-group covariance drives the sphericity statistics; S = 14, v = 11.
    centred = wide.to_numpy() - wide.groupby(groups.to_numpy()).transform("mean").to_numpy()
    error = contrasts.T @ centred.T @ centred @ contrasts
    roots = np.linalg.eigvalsh(error)
    gg = roots.sum() ** 2 / (3 * (roots ** 2).sum())
    assert result["sphericity"].loc["B", "epsilon_gg"] == pytest.approx(gg)
    assert result["sphericity"].loc["B", "epsilon_hf"] == pytest.approx(
        min(1.0, (14 * 3 * gg - 2) / (3 * (11 - 3 * gg))))
    via_anova = oe.anova(one, "y", ["g"], within=["B"], subject="id")
    np.testing.assert_allclose(via_anova["within"]["ss"], result["within"]["ss"])


def test_two_within_factors_with_a_between_factor_match_contrast_regressions():
    rng = np.random.default_rng(3)
    subjects = 15
    rows = list(itertools.product(range(subjects), range(2), range(3)))
    frame = pd.DataFrame(rows, columns=["id", "A", "B"])
    frame["g"] = np.where(frame["id"] < 4, "x", np.where(frame["id"] < 9, "y", "z"))
    frame["y"] = rng.normal(size=subjects)[frame["id"]] + 0.4 * frame["A"] + 0.2 * frame["B"] \
        + 0.3 * (frame["g"] == "y") * frame["B"] + rng.normal(size=len(frame))
    result = oe.rm_anova(frame, "y", "id", ["A", "B"], between=["g"])
    wide = frame.pivot(index="id", columns=["A", "B"], values="y")
    wide = wide[sorted(wide.columns)]
    groups = frame.drop_duplicates("id").set_index("id")["g"].loc[wide.index].to_numpy()
    h_a, h_b = helmert(2).T, helmert(3).T
    one_a, one_b = np.ones((2, 1)) / np.sqrt(2), np.ones((3, 1)) / np.sqrt(3)
    contrasts = {"A": np.kron(h_a, one_b), "B": np.kron(one_a, h_b), "A#B": np.kron(h_a, h_b)}
    for name, matrix in contrasts.items():
        transformed = wide.to_numpy() @ matrix
        ss = {"Intercept": 0.0, "C(g, Sum)": 0.0, "Residual": 0.0}
        for column in range(transformed.shape[1]):
            data = pd.DataFrame({"t": transformed[:, column], "g": groups})
            table = anova_lm(smf.ols("t ~ C(g, Sum)", data).fit(), typ=3)
            for key in ss:
                ss[key] += table.loc[key, "sum_sq"]
        assert _assumed(result, name)["ss"] == pytest.approx(ss["Intercept"])
        assert _assumed(result, f"{name}#g")["ss"] == pytest.approx(ss["C(g, Sum)"])
        assert _assumed(result, f"error({name})")["ss"] == pytest.approx(ss["Residual"])
        d = matrix.shape[1]
        assert _assumed(result, f"{name}#g")["df"] == 2 * d
        assert _assumed(result, f"error({name})")["df"] == 12 * d
    assert result["sphericity"].loc["A", "epsilon_gg"] == 1.0
    assert list(result["sphericity"].index) == ["A", "B", "A#B"]


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda f: f.iloc[1:], "unbalanced_design"),
    (lambda f: pd.concat([f, f.iloc[:1]]), "unbalanced_design"),
    (lambda f: f.assign(y=f["y"].where(f.index != 3)), "unbalanced_design"),
    (lambda f: f.assign(g=np.where(f.index == 0, "other", f["g"])), "invalid_design"),
    (lambda f: f.assign(B=0), "single_level"),
])
def test_rm_anova_error_codes(mutate, code):
    frame = _repeated()
    frame = frame[frame["A"] == 0].reset_index(drop=True)
    with pytest.raises(AnalysisError) as error:
        oe.rm_anova(mutate(frame), "y", "id", ["B"], between=["g"])
    assert error.value.code == code
    if code == "unbalanced_design":
        assert "mixed model" in str(error.value)


# ---- MANOVA -----------------------------------------------------------------------


def _multivariate(seed: int = 7, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"a": rng.choice(["u", "v", "w"], n), "b": rng.choice(["p", "q"], n),
                          "x": rng.normal(size=n)})
    noise = rng.normal(size=(n, 3)) @ np.array([[1, 0.3, 0.2], [0, 1, 0.4], [0, 0, 1]])
    frame["y1"] = noise[:, 0] + 0.5 * (frame["a"] == "v") + 0.3 * frame["x"]
    frame["y2"] = noise[:, 1] - 0.4 * (frame["b"] == "q")
    frame["y3"] = noise[:, 2] + 0.3 * (frame["a"] == "w") * (frame["b"] == "q")
    return frame


def test_manova_statistics_match_statsmodels():
    frame = _multivariate()
    result = oe.manova(frame, ["y1", "y2", "y3"], ["a", "b"], covariates=["x"])
    reference = MANOVA.from_formula("y1 + y2 + y3 ~ C(a, Sum) * C(b, Sum) + x", frame).mv_test()
    names = {"Intercept": "Intercept", "a": "C(a, Sum)", "b": "C(b, Sum)",
             "a#b": "C(a, Sum):C(b, Sum)", "x": "x"}
    rows = {"wilks": "Wilks' lambda", "pillai": "Pillai's trace",
            "hotelling": "Hotelling-Lawley trace", "roy": "Roy's greatest root"}
    table = result["multivariate"].set_index(["effect", "test"])
    p, v = 3, result.attrs["df_resid"]
    for ours, theirs in names.items():
        stat = reference.results[theirs]["stat"]
        q = 2 if ours in ("a", "a#b") else 1
        s = min(p, q)
        for test, label in rows.items():
            row = table.loc[(ours, test)]
            assert row["value"] == pytest.approx(float(stat.loc[label, "Value"]), rel=1e-9)
            if test != "hotelling" or s == 1:
                assert row["statistic"] == pytest.approx(float(stat.loc[label, "F Value"]),
                                                         rel=1e-8)
                assert row["df1"] == pytest.approx(float(stat.loc[label, "Num DF"]))
                assert row["df2"] == pytest.approx(float(stat.loc[label, "Den DF"]))
                assert row["p_value"] == pytest.approx(float(stat.loc[label, "Pr > F"]),
                                                       rel=1e-7)
        hotelling = table.loc[(ours, "hotelling")]
        m, n = (abs(p - q) - 1) / 2, (v - p - 1) / 2
        f = 2 * (s * n + 1) * hotelling["value"] / (s ** 2 * (2 * m + s + 1))
        assert hotelling["statistic"] == pytest.approx(f)
        assert (hotelling["df1"], hotelling["df2"]) == pytest.approx(
            (s * (2 * m + s + 1), 2 * (s * n + 1)))
        assert table.loc[(ours, "wilks"), "f_type"] == "exact"
        assert table.loc[(ours, "roy"), "f_type"] == ("exact" if s == 1 else "upper_bound")
        assert table.loc[(ours, "pillai"), "partial_eta_squared"] == pytest.approx(
            table.loc[(ours, "pillai"), "value"] / s)


def test_manova_univariate_tables_and_single_outcome_reduce_to_anova():
    frame = _multivariate()
    result = oe.manova(frame, ["y1", "y2", "y3"], ["a", "b"], covariates=["x"])
    univariate = result["univariate"]
    for outcome in ("y1", "y3"):
        single = oe.anova(frame, outcome, ["a", "b"], covariates=["x"])["anova"]
        block = univariate[univariate["outcome"] == outcome].set_index("source")
        np.testing.assert_allclose(block["ss"], single["ss"], rtol=1e-10)
        np.testing.assert_allclose(block["p_value"].dropna(), single["p_value"].dropna(),
                                   rtol=1e-9)
    one = oe.manova(frame, ["y1"], ["a", "b"], covariates=["x"])["multivariate"]
    anova = oe.anova(frame, "y1", ["a", "b"], covariates=["x"])["anova"]
    wilks = one[(one["effect"] == "a") & (one["test"] == "wilks")].iloc[0]
    assert wilks["statistic"] == pytest.approx(anova.loc["a", "statistic"])
    assert wilks["p_value"] == pytest.approx(anova.loc["a", "p_value"])


def test_box_m_matches_explicit_formulas():
    frame = _multivariate()
    result = oe.manova(frame, ["y1", "y2", "y3"], ["a", "b"])
    outcomes = ["y1", "y2", "y3"]
    cells = [group[outcomes].to_numpy() for _, group in frame.groupby(["a", "b"])]
    g, p = len(cells), 3
    df = np.array([len(cell) - 1 for cell in cells])
    pooled = sum((len(cell) - 1) * np.cov(cell, rowvar=False) for cell in cells) / df.sum()
    m = df.sum() * np.linalg.slogdet(pooled)[1] - sum(
        (len(cell) - 1) * np.linalg.slogdet(np.cov(cell, rowvar=False))[1] for cell in cells)
    c1 = ((1 / df).sum() - 1 / df.sum()) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    c2 = ((1 / df ** 2).sum() - 1 / df.sum() ** 2) * (p - 1) * (p + 2) / (6 * (g - 1))
    df1 = p * (p + 1) * (g - 1) / 2
    df2 = (df1 + 2) / abs(c2 - c1 ** 2)
    assert c2 > c1 ** 2
    f = m * (1 - c1 - df1 / df2) / df1
    row = result["box_m"].loc["box_m"]
    assert row["statistic"] == pytest.approx(m)
    assert row["f"] == pytest.approx(f) and row["df1"] == df1 and row["df2"] == pytest.approx(df2)
    assert row["p_value"] == pytest.approx(stats.f.sf(f, df1, df2))
    assert row["chi2"] == pytest.approx(m * (1 - c1))
    assert row["chi2_p_value"] == pytest.approx(stats.chi2.sf(m * (1 - c1), df1))
    small = frame.iloc[:14]
    sparse = oe.manova(small, outcomes, ["a"])
    assert ("box_m" in sparse) or sparse.attrs["notes"]


def test_manova_error_codes_and_missing_values():
    frame = _multivariate()
    frame["copy"] = frame["y1"] + frame["y2"]
    with pytest.raises(AnalysisError) as error:
        oe.manova(frame, ["y1", "y2", "copy"], ["a"])
    assert error.value.code == "singular_error_matrix"
    with pytest.raises(AnalysisError) as error:
        oe.manova(frame, "y1", ["a"])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.manova(frame.iloc[:7], ["y1", "y2", "y3"], ["a", "b"])
    assert error.value.code in {"no_residual_df", "insufficient_observations", "empty_cells"}
    frame.loc[0, "y2"] = np.nan
    result = oe.manova(frame, ["y1", "y2"], ["a"])
    assert result.attrs["n_missing"] == 1 and result.attrs["n"] == 119
    assert json.loads(json.dumps(result.attrs))["outcomes"] == ["y1", "y2"]
    assert "[multivariate]" in str(result) and r"\begin{tabular}" in result.to_latex()
