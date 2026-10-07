"""Regression tests of the defects fixed in the verification pass of the stats family,
and adversarial inputs: each must work correctly or raise AnalysisError with a code."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats import srange


def _frame(seed: int = 0, n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.normal(size=n),
        "g": rng.integers(0, 2, n), "h": rng.integers(0, 3, n),
        "id": np.arange(n) // 3, "t": np.arange(n) % 3,
    })


def _code(call) -> str:
    with pytest.raises(AnalysisError) as caught:
        call()
    assert str(caught.value)                      # a message that says what to change
    return caught.value.code


def _finite(result) -> None:
    """No infinity anywhere in a result (undefined cells are missing, never inf)."""
    tables = result.values() if isinstance(result, dict) else [result]
    for table in tables:
        numbers = table.select_dtypes("number").to_numpy(dtype=float)
        assert not np.isinf(numbers).any()


# ---- fixes: raw exceptions became AnalysisError -------------------------------------------


def test_one_sample_sd_test_with_an_extreme_hypothesized_sd():
    frame = _frame()
    assert _code(lambda: oe.sdtest(frame, "y", sd=1e-300)) == "invalid_option"
    huge = oe.sdtest(frame, "y", sd=1e300)
    assert huge.attrs["statistic"] == 0.0 and huge.attrs["p_value"] == 0.0


@pytest.mark.parametrize("value", [3.0, "3", True, None, 0, 4])
def test_ss_type_must_be_an_integer_one_two_or_three(value):
    assert _code(lambda: oe.anova(_frame(), "y", ["g"], ss_type=value)) == "invalid_option"


def test_describe_rejects_repeated_statistics_after_alias_resolution_and_bare_strings():
    frame = _frame()
    for requested in (["sd", "std_dev"], ["median", "p50"], ["mean", "mean"], ["se", "std_error"]):
        assert _code(lambda r=requested: oe.describe(frame, ["y"], stats=r)) == "invalid_spec"
    assert _code(lambda: oe.describe(frame, ["y"], stats="mean")) == "invalid_spec"
    assert _code(lambda: oe.oneway(frame, "y", "h", posthoc="tukey")) == "invalid_spec"
    with pytest.raises(AnalysisError, match="list of methods"):
        oe.oneway(frame, "y", "h", posthoc="tukey")


def test_repeated_measures_with_constant_outcome_or_zero_error_terms():
    frame = _frame()
    for constant in (1.0, 0.1, 0.0):
        assert _code(lambda v=constant: oe.rm_anova(frame.assign(y=v), "y", "id", ["t"])) \
            == "zero_variance"
    # y depends on the subject only: the within error is exactly zero.
    subject_only = oe.rm_anova(frame.assign(y=frame["id"] * 1.5), "y", "id", ["t"])
    within = subject_only["within"]
    assert within["statistic"].isna().all() and within["p_value"].isna().all()
    assert (within["ss"] == 0.0).all()
    assert subject_only["sphericity"][["mauchly_w", "epsilon_gg", "epsilon_hf"]].isna().all().all()
    assert any("error term of 't'" in note for note in subject_only.attrs["notes"])
    assert subject_only["between"].loc["Intercept", "statistic"] > 0
    # y depends on time only: effect without error, eta squared is one, F is undefined.
    time_only = oe.rm_anova(frame.assign(y=frame["t"] * 2.0), "y", "id", ["t"])
    row = time_only["within"].iloc[0]
    assert row["ss"] == pytest.approx(16 * 2 * 4.0) and np.isnan(row["statistic"])
    assert row["partial_eta_squared"] == 1.0
    assert len(time_only.attrs["notes"]) == 2
    _finite(subject_only)
    _finite(time_only)


def test_ipsative_scores_have_no_between_subject_error_but_valid_within_tests():
    # Ranks within each subject: every subject mean is the same.
    frame = _frame(seed=4)
    ranked = frame.assign(y=frame.groupby("id")["y"].rank())
    result = oe.rm_anova(ranked, "y", "id", ["t"])
    assert np.isnan(result["between"].loc["Intercept", "statistic"])
    assert result["between"].loc["error", "ss"] == 0.0
    within = result["within"].set_index(["source", "correction"])
    wide = ranked.pivot(index="id", columns="t", values="y").to_numpy()
    s, p = wide.shape
    ss_time = s * ((wide.mean(0) - wide.mean()) ** 2).sum()
    ss_error = ((wide - wide.mean()) ** 2).sum() - ss_time
    f = ss_time / (p - 1) / (ss_error / ((p - 1) * (s - 1)))
    assert within.loc[("t", "sphericity_assumed"), "statistic"] == pytest.approx(f, rel=1e-9)
    assert any("subject means" in note for note in result.attrs["notes"])


# ---- fixes: silent overflow and underflow ---------------------------------------------------


@pytest.mark.parametrize("scale", [1e100, 1e-90, 1e8, 1e-8])
def test_higher_moments_and_correlations_do_not_depend_on_the_scale(scale):
    frame = _frame(seed=2, n=60)
    names = ["skewness", "kurtosis", "se_skewness", "se_kurtosis"]
    for moments in ("spss", "stata"):
        base = oe.describe(frame, ["y", "x"], stats=names, moments=moments)
        scaled = oe.describe(frame * scale, ["y", "x"], stats=names, moments=moments)
        np.testing.assert_allclose(scaled.to_numpy(dtype=float), base.to_numpy(dtype=float),
                                   rtol=1e-9, equal_nan=True)
    assert not base[["skewness", "kurtosis"]].isna().any().any()
    for method in ("pearson", "spearman"):
        base = oe.correlate(frame, ["y", "x", "z"], method=method)["coefficients"]
        scaled = oe.correlate(frame * scale, ["y", "x", "z"], method=method)["coefficients"]
        np.testing.assert_allclose(scaled.to_numpy(dtype=float), base.to_numpy(dtype=float),
                                   rtol=1e-9)
    paired = oe.ttest(frame * scale, "y", paired_with="x")
    assert paired.attrs["correlation"] == pytest.approx(
        stats.pearsonr(frame["y"], frame["x"]).statistic, rel=1e-9)
    assert paired.attrs["statistic"] == pytest.approx(
        stats.ttest_rel(frame["y"], frame["x"]).statistic, rel=1e-9)


def test_perfect_correlations_of_integers_stay_exact():
    data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [3.0, 5.0, 7.0, 9.0, 11.0],
            "w": [10.0, 8.0, 6.0, 4.0, 2.0]}
    result = oe.correlate(data, ["x", "y", "w"])
    assert result["coefficients"].loc["x", "y"] == 1.0
    assert result["coefficients"].loc["x", "w"] == -1.0
    assert result["p_values"].loc["x", "y"] == 0.0


def test_values_too_small_or_too_large_for_squares_are_refused_with_advice():
    frame = _frame()
    tiny, huge = frame.assign(y=frame["y"] * 1e-120), frame.assign(y=frame["y"] * 1e160)
    calls = [lambda d: oe.ttest(d, "y"), lambda d: oe.ttest(d, "y", by="g"),
             lambda d: oe.sdtest(d, "y", by="g"), lambda d: oe.oneway(d, "y", "h"),
             lambda d: oe.anova(d, "y", ["g"]), lambda d: oe.describe(d, ["y"]),
             lambda d: oe.correlate(d, ["y", "x"]), lambda d: oe.pcorr(d, "y", ["x"]),
             lambda d: oe.rm_anova(d, "y", "id", ["t"]),
             lambda d: oe.manova(d, ["y", "x"], ["g"])]
    for call in calls:
        for data in (tiny, huge):
            with pytest.raises(AnalysisError, match="rescale") as caught:
                call(data)
            assert caught.value.code == "non_finite_values"
    # A column of exact zeros is a constant, not a scaling problem.
    assert _code(lambda: oe.ttest(frame.assign(y=0.0), "y")) == "zero_variance"
    # Tiny values next to ordinary ones are fine.
    mixed = frame.assign(y=np.where(frame.index < 5, 1e-200, frame["y"]))
    assert math.isfinite(oe.ttest(mixed, "y").attrs["statistic"])


def test_one_sample_t_far_from_and_close_to_the_hypothesized_mean():
    frame = _frame(seed=6)
    x = frame["y"].to_numpy()
    se = x.std(ddof=1) / math.sqrt(len(x))
    for mu in (1e140, -1e200, 1e300):
        result = oe.ttest(frame, "y", mu=mu)
        assert result.attrs["statistic"] == pytest.approx((x.mean() - mu) / se, rel=1e-12)
        assert result["statistics"].loc["y", "mean"] == pytest.approx(x.mean(), rel=1e-12)
        assert result["statistics"].loc["y", "std_dev"] == pytest.approx(x.std(ddof=1), rel=1e-12)
        assert result.attrs["p_value"] == 0.0
    assert _code(lambda: oe.ttest(frame, "y", mu=1.7e308)) == "invalid_option"
    # A large common level with mu next to it keeps every digit the data have (values on
    # a grid of 1/1024 so that adding 2^30 is exact).
    grid = np.round(frame["y"] * 1024) / 1024
    shifted = oe.ttest(frame.assign(y=grid + 2.0 ** 30), "y", mu=2.0 ** 30 + 0.25)
    base = oe.ttest(frame.assign(y=grid), "y", mu=0.25)
    assert shifted.attrs["statistic"] == pytest.approx(base.attrs["statistic"], rel=1e-9)
    assert shifted.attrs["p_value"] == pytest.approx(base.attrs["p_value"], rel=1e-8)
    paired = oe.ttest(frame.assign(w=frame["x"] + 2.0 ** 30, y=grid + 2.0 ** 30), "y",
                      paired_with="w", mu=1e150)
    assert math.isfinite(paired.attrs["statistic"])


# ---- fixes: silently ignored options ----------------------------------------------------------


def test_anova_hands_repeated_measures_over_without_dropping_options():
    frame = _frame()
    direct = oe.rm_anova(frame, "y", "id", ["t"])
    through = oe.anova(frame, "y", within=["t"], subject="id")
    pd.testing.assert_frame_equal(through["within"], direct["within"])
    assert _code(lambda: oe.anova(frame, "y", within=["t"], subject="id", ss_type=2)) \
        == "invalid_spec"
    assert _code(lambda: oe.anova(frame, "y", within=["t"], subject="id",
                                  interactions="none")) == "invalid_spec"
    assert _code(lambda: oe.anova(frame, "y", within=["t"], subject="id", covariates=["x"])) \
        == "invalid_spec"
    assert _code(lambda: oe.anova(frame, "y", within=["t"])) == "invalid_spec"
    holed = frame.assign(y=frame["y"].where(frame.index != 4))
    assert _code(lambda: oe.anova(holed, "y", within=["t"], subject="id", missing="raise")) \
        == "missing_values"
    assert _code(lambda: oe.rm_anova(holed, "y", "id", ["t"], missing="raise")) \
        == "missing_values"
    assert _code(lambda: oe.rm_anova(holed, "y", "id", ["t"])) == "unbalanced_design"
    assert _code(lambda: oe.rm_anova(frame, "y", "id", ["t"], missing="ignore")) \
        == "invalid_option"


def test_ambiguous_term_names_and_clashing_result_columns_are_rejected():
    frame = _frame()
    renamed = frame.rename(columns={"x": "total", "g": "error", "h": "mean"})
    assert _code(lambda: oe.anova(renamed, "y", ["error"])) == "invalid_spec"
    assert _code(lambda: oe.anova(renamed, "y", ["mean"], covariates=["total"])) == "invalid_spec"
    assert _code(lambda: oe.manova(renamed, ["y", "z"], ["error"])) == "invalid_spec"
    assert _code(lambda: oe.anova(frame.rename(columns={"g": "Intercept"}), "y",
                                  ["Intercept"])) == "invalid_spec"
    hashed = frame.assign(**{"g#h": frame["g"] * 3 + frame["h"]})
    assert _code(lambda: oe.anova(hashed, "y", ["g", "h", "g#h"])) == "invalid_spec"
    assert list(oe.anova(hashed, "y", ["g#h"])["anova"].index)[2] == "g#h"
    assert _code(lambda: oe.anova(renamed, "y", ["mean"], emmeans=["mean"])) == "invalid_spec"
    assert _code(lambda: oe.describe(renamed, ["y"], by="mean")) == "invalid_spec"
    assert _code(lambda: oe.rm_anova(frame.rename(columns={"t": "n"}), "y", "id", ["n"])) \
        == "invalid_spec"


# ---- fixes: critical values for very small alpha -----------------------------------------------


def test_upper_tail_quantiles_of_the_range_distributions():
    # Two groups: the studentized range is sqrt(2) |t|, and one Dunnett comparison is |t|.
    for p, df in ((1e-12, 37.0), (1e-50, 12.5), (1e-200, 60.0), (0.05, 3.0), (0.9, 7.0)):
        expected = stats.t.isf(p / 2, df)
        assert float(srange.qtukey_upper(p, 2, df)[0]) == pytest.approx(
            expected * math.sqrt(2), rel=1e-9)
        assert float(srange.qdunnett_upper(p, [math.sqrt(0.5)], df)[0]) == pytest.approx(
            expected, rel=1e-9)
    for p, k, df in ((0.05, 4, 20.0), (0.01, 10, 8.0), (0.3, 3, 100.0)):
        assert float(srange.qtukey_upper(p, k, df)[0]) == pytest.approx(
            float(srange.qtukey(1 - p, k, df)[0]), rel=1e-10)
        assert float(srange.qtukey_upper(p, k, df)[0]) == pytest.approx(
            stats.studentized_range.isf(p, k, df), rel=1e-8)
    assert float(srange.qdunnett_upper(0.05, [0.7, 0.6, 0.5], 25.0)[0]) == pytest.approx(
        float(srange.qdunnett(0.95, [0.7, 0.6, 0.5], 25.0)[0]), rel=1e-10)


@pytest.mark.parametrize("alpha", [1e-12, 1e-40, 0.5, 0.999])
def test_posthoc_intervals_for_extreme_alpha_agree_with_the_t_interval_for_two_groups(alpha):
    frame = _frame(seed=9)
    # control=1 makes Dunnett's single comparison group 0 - group 1, like the others.
    result = oe.oneway(frame, "y", "g", posthoc=["tukey", "dunnett", "lsd", "scheffe"],
                       alpha=alpha, control=1)
    pooled = oe.ttest(frame, "y", by="g", alpha=alpha)["test"].loc["equal_variances"]
    for method in ("tukey", "dunnett", "lsd", "scheffe"):
        row = result[f"posthoc_{method}"].iloc[0]
        assert row["ci_low"] == pytest.approx(pooled["ci_low"], rel=1e-7)
        assert row["ci_high"] == pytest.approx(pooled["ci_high"], rel=1e-7)
        assert row["p_value"] == pytest.approx(pooled["p_value"], rel=1e-7)
    _finite(result)


def test_alpha_beyond_double_precision_is_an_option_error_not_a_distribution_error():
    frame = _frame(seed=9)
    for method in ("tukey", "games_howell", "dunnett"):
        try:
            result = oe.oneway(frame, "y", "h", posthoc=[method], alpha=1e-300)
        except AnalysisError as exc:
            assert exc.code == "invalid_option" and "alpha" in str(exc)
        else:
            _finite(result)
            assert (result[f"posthoc_{method}"]["ci_high"]
                    > result[f"posthoc_{method}"]["ci_low"]).all()


# ---- adversarial sweep --------------------------------------------------------------------


_F = _frame()
_ERRORS = [
    (lambda: oe.ttest(pd.DataFrame({"y": []}), "y"), "empty_data"),
    (lambda: oe.ttest(pd.DataFrame({"y": [1.0]}), "y"), "insufficient_observations"),
    (lambda: oe.ttest(_F.assign(y=np.nan), "y"), "empty_sample"),
    (lambda: oe.ttest(_F.assign(y=3.0), "y", mu=3), "zero_variance"),
    (lambda: oe.ttest(_F.assign(y="a"), "y"), "non_numeric_column"),
    (lambda: oe.ttest(_F.assign(y=pd.date_range("2020", periods=len(_F))), "y"),
     "non_numeric_column"),
    (lambda: oe.ttest(_F.assign(y=np.inf), "y"), "non_finite_values"),
    (lambda: oe.ttest(_F, "y", by="h"), "invalid_groups"),
    (lambda: oe.ttest(_F, "y", by="nope"), "missing_columns"),
    (lambda: oe.ttest(_F, "y", by="y"), "invalid_spec"),
    (lambda: oe.ttest(_F, "y", by="g", paired_with="x"), "invalid_spec"),
    (lambda: oe.ttest(_F, "y", alpha=0), "invalid_option"),
    (lambda: oe.ttest(_F, "y", mu=float("nan")), "invalid_option"),
    (lambda: oe.ttest(_F, "y", missing="keep"), "invalid_option"),
    (lambda: oe.ttest([1.0, 2.0], "y"), "invalid_data"),
    (lambda: oe.ttest(pd.DataFrame({"y": [1.0, 2, 3, 4], "g": [0, 1, 1, 1]}), "y", by="g"),
     "insufficient_observations"),
    (lambda: oe.ttest(pd.DataFrame({"y": [1.0, 1, 2, 2], "g": [0, 0, 1, 1]}), "y", by="g"),
     "zero_variance"),
    (lambda: oe.sdtest(_F, "y"), "invalid_spec"),
    (lambda: oe.sdtest(_F, "y", sd=0), "invalid_option"),
    (lambda: oe.sdtest(_F, "y", by="g", sd=1), "invalid_spec"),
    (lambda: oe.sdtest(_F.assign(k=1), "y", by="k"), "invalid_groups"),
    (lambda: oe.sdtest(_F, "y", by="id", sd=None, center="mean", missing="x"), "invalid_option"),
    (lambda: oe.sdtest(_F, "y", by="g", center="mode"), "invalid_option"),
    (lambda: oe.oneway(_F.assign(k=1), "y", "k"), "invalid_groups"),
    (lambda: oe.oneway(_F.assign(k=np.arange(len(_F))), "y", "k"), "insufficient_observations"),
    (lambda: oe.oneway(_F.assign(y=2.0), "y", "h"), "zero_variance"),
    (lambda: oe.oneway(_F, "y", "h", posthoc=["duncan"]), "invalid_option"),
    (lambda: oe.oneway(_F, "y", "h", control=0), "invalid_option"),
    (lambda: oe.oneway(_F, "y", "h", posthoc=["dunnett"], control=9), "invalid_option"),
    (lambda: oe.oneway(pd.DataFrame({"y": [1.0, 2, 3, 4, 6], "g": [0, 1, 1, 2, 2]}), "y", "g",
                       posthoc=["games_howell"]), "zero_variance"),
    (lambda: oe.anova(_F, "y"), "invalid_spec"),
    (lambda: oe.anova(_F, "y", "g"), "invalid_spec"),
    (lambda: oe.anova(_F.assign(k=1), "y", ["k"]), "single_level"),
    (lambda: oe.anova(_F.assign(k=np.arange(len(_F))), "y", ["k"]), "no_residual_df"),
    (lambda: oe.anova(_F.assign(k=_F["g"]), "y", ["g", "k"]), "empty_cells"),
    (lambda: oe.anova(_F.assign(k=_F["g"]), "y", ["g", "k"], interactions="none"),
     "collinear_design"),
    (lambda: oe.anova(_F.assign(k=1.0), "y", ["g"], covariates=["k"]), "collinear_design"),
    (lambda: oe.anova(_F.assign(k=2 * _F["x"]), "y", ["g"], covariates=["x", "k"]),
     "collinear_design"),
    (lambda: oe.anova(_F.assign(y=_F["g"] * 2.0), "y", ["g"]), "perfect_fit"),
    (lambda: oe.anova(_F, "y", ["g"], emmeans=["x"]), "invalid_spec"),
    (lambda: oe.anova(_F, "y", ["g"], emmeans=["g"], adjust="tukey"), "invalid_option"),
    (lambda: oe.anova(_F, "y", ["g", "h"], interactions=[["g", "nope"]]), "invalid_spec"),
    (lambda: oe.rm_anova(_F, "y", "id", "t"), "invalid_spec"),
    (lambda: oe.rm_anova(_F.iloc[1:], "y", "id", ["t"]), "unbalanced_design"),
    (lambda: oe.rm_anova(pd.concat([_F, _F.iloc[:1]]), "y", "id", ["t"]), "unbalanced_design"),
    (lambda: oe.rm_anova(_F[_F["id"] == 0], "y", "id", ["t"]), "no_residual_df"),
    (lambda: oe.rm_anova(_F, "y", "id", ["t"], between=["g"]), "invalid_design"),
    (lambda: oe.rm_anova(_F[_F["t"] == 0], "y", "id", ["t"]), "single_level"),
    (lambda: oe.manova(_F, "y", ["g"]), "invalid_spec"),
    (lambda: oe.manova(_F.assign(w=_F["y"] + _F["x"]), ["y", "x", "w"], ["g"]),
     "singular_error_matrix"),
    (lambda: oe.manova(_F.assign(w=_F["g"] * 1.0), ["y", "w"], ["g"]), "perfect_fit"),
    (lambda: oe.manova(_F.head(4), ["y", "x", "z"], ["g"]), "insufficient_observations"),
    (lambda: oe.correlate(_F, ["y"]), "invalid_spec"),
    (lambda: oe.correlate(_F, ["y", "x"], method="phi"), "invalid_option"),
    (lambda: oe.correlate(_F.assign(x=np.nan), ["y", "x"], pairwise=False),
     "insufficient_observations"),
    (lambda: oe.pcorr(_F, "y", ["x"], controls=["x"]), "invalid_spec"),
    (lambda: oe.pcorr(_F.assign(k=1.0), "y", ["x", "k"]), "collinear_design"),
    (lambda: oe.pcorr(_F.assign(k=_F["x"] + _F["z"]), "k", ["x", "z"]), "perfect_fit"),
    (lambda: oe.pcorr(_F.head(3), "y", ["x", "z"]), "insufficient_observations"),
    (lambda: oe.describe(pd.DataFrame({"s": ["a", "b"]})), "invalid_spec"),
    (lambda: oe.describe(_F, ["y"], stats=["p100"]), "invalid_option"),
    (lambda: oe.describe(_F, ["y"], stats=[]), "invalid_spec"),
    (lambda: oe.describe(_F.assign(k=np.nan), ["y"], by="k"), "empty_sample"),
    (lambda: oe.describe(_F, ["y"], percentile_method="r7"), "invalid_option"),
]


@pytest.mark.parametrize("index", range(len(_ERRORS)))
def test_invalid_inputs_raise_analysis_errors(index):
    call, code = _ERRORS[index]
    assert _code(call) == code


def test_degenerate_but_valid_inputs_give_finite_tables():
    frame = _frame(seed=12)
    tiny = pd.DataFrame({"y": [1.0, 2.5, 2.0, 4.0, 3.5, 7.0], "g": [0, 0, 1, 1, 2, 2],
                         "x": [0.3, 1.0, 0.2, 0.9, 0.1, 0.8]})
    results = [
        oe.ttest(pd.DataFrame({"y": [1.0, 2.0]}), "y"),
        oe.ttest(pd.DataFrame({"y": [1.0, 1, 1, 2, 3, 5], "g": [0, 0, 0, 1, 1, 1]}), "y", by="g"),
        oe.sdtest(frame.assign(y=3.0), "y", by="g"),
        oe.sdtest(frame.assign(y=3.0), "y", sd=2.0),
        oe.oneway(tiny, "y", "g", posthoc=["tukey", "games_howell", "dunnett", "holm"]),
        oe.oneway(pd.DataFrame({"y": [1.0, 2, 3, 4, 6], "g": [0, 1, 1, 2, 2]}), "y", "g",
                  posthoc=["tukey", "scheffe"]),
        oe.anova(tiny, "y", ["g"], covariates=["x"], emmeans=["g"]),
        oe.anova(frame, "y", covariates=["x"]),
        oe.rm_anova(frame[frame["id"] < 2], "y", "id", ["t"]),
        oe.manova(frame, ["y"], ["g"]),
        oe.correlate(frame.assign(k=1.0, m=np.nan), ["y", "k", "m"], ci=True),
        oe.correlate(frame.head(2), ["y", "x"], method="kendall", ci=True),
        oe.describe(frame.assign(k=np.nan), ["k", "y"]),
        oe.describe(frame.head(1), ["y"]),
        oe.describe(frame.assign(k=2.0), ["k"]),
        oe.describe(frame, ["y", "x"], listwise=True, by="h"),
    ]
    for result in results:
        _finite(result)
    # Undefined pieces are missing cells with their reason noted, not numbers.
    assert np.isnan(results[5]["robust"].loc["welch", "statistic"])
    assert results[5].attrs["notes"]
    assert np.isnan(results[10]["coefficients"].loc["y", "k"])
    assert results[12].loc["k", "n"] == 0 and np.isnan(results[12].loc["k", "mean"])


def test_nullable_categorical_and_boolean_inputs():
    data = pd.DataFrame({
        "y": pd.array([1.5, 2.5, None, 4.0, 5.0, 7.0, 3.0], dtype="Float64"),
        "flag": pd.array([True, False, True, None, False, True, False], dtype="boolean"),
        "level": pd.Categorical(["a", "b", "a", "b", None, "a", "b"], categories=["z", "b", "a"]),
        "text": pd.array(["u", "v", "u", "v", "u", None, "v"], dtype="string"),
    })
    by_level = oe.ttest(data, "y", by="level")
    assert by_level.attrs["groups"] == ["b", "a"]            # category order, unused level dropped
    assert by_level.attrs["n"] == 5 and by_level.attrs["n_missing"] == 2
    assert oe.ttest(data, "y", by="text").attrs["groups"] == ["u", "v"]
    assert oe.ttest(data, "y", by="flag").attrs["groups"] == [False, True]
    described = oe.describe(data, ["y", "flag"], stats=["n", "mean"])
    assert described.loc["y", "n"] == 6 and described.loc["flag", "mean"] == pytest.approx(0.5)
