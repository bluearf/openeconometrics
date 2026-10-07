"""Independent oracles for the linear-model procedures of the stats family.

Verification pass: nothing here reuses the implementation's formulas. Factorial
models are checked through the CELL-MEANS parametrization (hypothesis matrices on
cell means, NumPy pseudo-inverses), residual-sum-of-squares differences of
over-parametrized dummy designs, classical sums-of-squares formulas, closed-form
multivariate F tests and a 60-digit mpmath solve for a badly scaled covariate.
"""

from __future__ import annotations

import itertools
import math

import mpmath
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe

RTOL = 1e-8


# ---- independent linear-model machinery ---------------------------------------------


def _dummies(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    """Indicator columns of every observed combination of ``names`` (one column if empty)."""
    if not names:
        return np.ones((len(frame), 1))
    keys = frame[list(names)].astype(str).agg("|".join, axis=1)
    return pd.get_dummies(keys).to_numpy(dtype=float)


def _columns(frame: pd.DataFrame, term: tuple[tuple[str, ...], tuple[str, ...]]) -> np.ndarray:
    """Over-parametrized columns of a term (factors, covariates): dummies times covariates."""
    factors, covariates = term
    block = _dummies(frame, factors)
    for name in covariates:
        block = block * frame[name].to_numpy(dtype=float)[:, None]
    return block


def _rss(frame: pd.DataFrame, y: str, terms: list) -> float:
    design = np.hstack([_columns(frame, term) for term in terms])
    outcome = frame[y].to_numpy(dtype=float)
    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    return float(((outcome - design @ beta) ** 2).sum())


def _contains(big, small) -> bool:
    """SPSS / SAS containment: same covariates, factors a proper superset."""
    return set(big[1]) == set(small[1]) and set(small[0]) < set(big[0])


def _cell_means_fit(frame: pd.DataFrame, y: str, factors: list[str], covariates: list[str]):
    """Cell-means model y = mu_cell + b'x + e on a full factorial with every cell observed."""
    levels = [sorted(frame[name].unique()) for name in factors]
    cells = list(itertools.product(*levels))
    position = {cell: i for i, cell in enumerate(cells)}
    index = np.array([position[tuple(row)] for row in frame[factors].itertuples(index=False)])
    design = np.zeros((len(frame), len(cells) + len(covariates)))
    design[np.arange(len(frame)), index] = 1.0
    for j, name in enumerate(covariates):
        design[:, len(cells) + j] = frame[name].to_numpy(dtype=float)
    outcome = frame[y].to_numpy(dtype=float)
    gram_inverse = np.linalg.inv(design.T @ design)
    beta = gram_inverse @ design.T @ outcome
    resid = outcome - design @ beta
    df_error = len(frame) - design.shape[1]
    return levels, beta, gram_inverse, float(resid @ resid), df_error


def _effect_matrix(levels: list[list], subset: tuple[int, ...], extra: int) -> np.ndarray:
    """Rows on the cell means: differences over the factors in ``subset``, averages elsewhere."""
    matrix = np.ones((1, 1))
    for i, values in enumerate(levels):
        size = len(values)
        if i in subset:
            piece = np.hstack([np.eye(size - 1), -np.ones((size - 1, 1))])
        else:
            piece = np.full((1, size), 1.0 / size)
        matrix = np.kron(matrix, piece)
    return np.hstack([matrix, np.zeros((matrix.shape[0], extra))])


def _wald_ss(matrix: np.ndarray, beta: np.ndarray, gram_inverse: np.ndarray) -> float:
    value = matrix @ beta
    return float(value @ np.linalg.solve(matrix @ gram_inverse @ matrix.T, value))


def _factorial(seed: int = 5, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.choice(["a1", "a2", "a3"], size=n, p=[0.5, 0.3, 0.2]),
        "b": rng.choice(["b1", "b2"], size=n, p=[0.35, 0.65]),
        "c": rng.choice([10, 20, 30], size=n),
        "x": rng.normal(2.0, 1.5, size=n),
        "w": rng.exponential(1.0, size=n),
    })
    effect = {"a1": 0.0, "a2": 0.6, "a3": -0.4}
    frame["y"] = (frame["a"].map(effect) + 0.5 * (frame["b"] == "b2") + 0.02 * frame["c"]
                  + 0.4 * frame["x"] - 0.2 * frame["w"]
                  + 0.5 * (frame["a"] == "a2") * (frame["b"] == "b2") + rng.normal(size=n))
    frame["y2"] = 0.3 * frame["y"] + rng.normal(size=n) + 0.3 * (frame["a"] == "a3")
    frame["y3"] = rng.normal(size=n) - 0.2 * frame["x"]
    return frame


# ---- anova: Type III from the cell-means model ---------------------------------------


@pytest.mark.parametrize("factors, covariates", [
    (["a", "b"], []),
    (["a", "b"], ["x", "w"]),
    (["a", "b", "c"], ["x"]),
])
def test_type_three_equals_cell_means_hypotheses(factors, covariates):
    frame = _factorial()
    result = oe.anova(frame, "y", factors, covariates=covariates)
    table = result["anova"]
    levels, beta, gram_inverse, sse, df_error = _cell_means_fit(frame, "y", factors, covariates)
    cells = int(np.prod([len(v) for v in levels]))
    extra = len(covariates)
    mse = sse / df_error
    assert table.loc["error", "ss"] == pytest.approx(sse, rel=RTOL)
    assert table.loc["error", "df"] == df_error
    expected = {"Intercept": _effect_matrix(levels, (), extra)}
    for order in range(1, len(factors) + 1):
        for subset in itertools.combinations(range(len(factors)), order):
            expected["#".join(factors[i] for i in subset)] = _effect_matrix(levels, subset, extra)
    for j, name in enumerate(covariates):
        row = np.zeros((1, cells + extra))
        row[0, cells + j] = 1.0
        expected[name] = row
    assert set(expected) | {"corrected_model", "error", "total", "corrected_total"} \
        == set(table.index)
    for name, matrix in expected.items():
        ss = _wald_ss(matrix, beta, gram_inverse)
        df = matrix.shape[0]
        f = ss / df / mse
        row = table.loc[name]
        assert row["ss"] == pytest.approx(ss, rel=RTOL), name
        assert row["df"] == df
        assert row["ms"] == pytest.approx(ss / df, rel=RTOL)
        assert row["statistic"] == pytest.approx(f, rel=RTOL)
        assert row["p_value"] == pytest.approx(stats.f.sf(f, df, df_error), rel=1e-7, abs=1e-300)
        assert row["partial_eta_squared"] == pytest.approx(ss / (ss + sse), rel=RTOL)
    outcome = frame["y"].to_numpy()
    sst = float(((outcome - outcome.mean()) ** 2).sum())
    assert table.loc["corrected_total", "ss"] == pytest.approx(sst, rel=RTOL)
    assert table.loc["corrected_total", "df"] == len(frame) - 1
    assert table.loc["total", "ss"] == pytest.approx(float((outcome ** 2).sum()), rel=RTOL)
    assert table.loc["total", "df"] == len(frame)
    model_df = cells + extra - 1
    assert table.loc["corrected_model", "ss"] == pytest.approx(sst - sse, rel=RTOL)
    assert table.loc["corrected_model", "df"] == model_df
    f_model = (sst - sse) / model_df / mse
    assert result.attrs["statistic"] == pytest.approx(f_model, rel=RTOL)
    assert result.attrs["p_value"] == pytest.approx(stats.f.sf(f_model, model_df, df_error),
                                                    rel=1e-7)
    assert result.attrs["r_squared"] == pytest.approx(1 - sse / sst, rel=RTOL)
    assert result.attrs["adjusted_r_squared"] == pytest.approx(
        1 - mse / (sst / (len(frame) - 1)), rel=RTOL)
    assert result.attrs["rmse"] == pytest.approx(math.sqrt(mse), rel=RTOL)
    assert result.attrs["df_model"] == model_df and result.attrs["df_resid"] == df_error


# ---- anova: Type I and Type II from residual sums of squares --------------------------


def _model_terms(factors, covariates, interactions, slopes=False):
    terms = [((), ())] + [((), (x,)) for x in covariates] + [((f,), ()) for f in factors]
    terms += interactions
    if slopes:
        terms += [((f,), (x,)) for x in covariates for f in factors]
    return terms


def _name(term) -> str:
    return "#".join([*term[0], *term[1]]) or "Intercept"


@pytest.mark.parametrize("covariates, interactions, option", [
    (["x"], [(("a", "b"), ())], "full"),
    (["x"], [], "none"),
    (["x"], [(("a", "b"), ()), (("a",), ("x",))], [["a", "b"], ["a", "x"]]),
    # Separate slopes in every cell, written with all three separators.
    (["x"], [(("b",), ("x",)), (("a",), ("x",)), (("a", "b"), ("x",)), (("a", "b"), ())],
     ["b#x", ["a", "x"], "a*b*x", "a:b"]),
    # Products of covariates; a#x#w without a#x is not hierarchical in the covariates, so
    # the design keeps the raw covariates.
    (["x", "w"], [((), ("x", "w"))], [["x", "w"]]),
    (["x", "w"], [((), ("x", "w")), (("a",), ("x", "w"))], [["x", "w"], ["a", "x", "w"]]),
    (["x", "w"], [(("a",), ("x",)), ((), ("x", "w")), (("a",), ("w",)), (("a",), ("x", "w"))],
     ["a#x", "x#w", "a#w", "a#x#w"]),
])
def test_type_one_and_two_equal_differences_of_residual_sums_of_squares(covariates, interactions,
                                                                        option):
    frame = _factorial(seed=11, n=120)
    factors = ["a", "b"]
    terms = _model_terms(factors, covariates, interactions)
    full = _rss(frame, "y", terms)
    sequential = oe.anova(frame, "y", factors, covariates=covariates, interactions=option,
                          ss_type=1)["anova"]
    hierarchical = oe.anova(frame, "y", factors, covariates=covariates, interactions=option,
                            ss_type=2)["anova"]
    for table in (sequential, hierarchical):
        assert table.loc["error", "ss"] == pytest.approx(full, rel=RTOL)
    outcome = frame["y"].to_numpy()
    for position, term in enumerate(terms):
        if position == 0:
            type_one = len(frame) * outcome.mean() ** 2
        else:
            type_one = _rss(frame, "y", terms[:position]) - _rss(frame, "y", terms[:position + 1])
        assert sequential.loc[_name(term), "ss"] == pytest.approx(type_one, rel=1e-7), _name(term)
        # Type II: adjusted for every term that does not contain it. The intercept is
        # contained in the pure factor effects, not in terms that involve a covariate.
        members = [other for other in terms if not _contains(other, term)]
        without = [other for other in members if other != term]
        type_two = _rss(frame, "y", without) - _rss(frame, "y", members)
        assert hierarchical.loc[_name(term), "ss"] == pytest.approx(type_two, rel=1e-7), \
            _name(term)
    # The sequential sums of squares add up to the uncorrected total.
    names = [_name(term) for term in terms] + ["error"]
    assert sequential.loc[names, "ss"].sum() == pytest.approx(float((outcome ** 2).sum()),
                                                              rel=1e-9)


@pytest.mark.parametrize("option, missing", [
    ([["a", "b", "c"]], "a#b"),
    ([["a", "b"], ["a", "b", "c"]], "a#c"),
    ([["a", "b", "x"]], "a#x"),
    ([["a", "b"], ["a", "x"], ["a", "b", "x"]], "b#x"),
])
def test_interactions_without_their_lower_order_terms_are_refused(option, missing):
    # SPSS / Stata would fit cell indicators for such a term (a larger model); products of
    # sum-to-zero columns would silently fit something else.
    from openecon.analysis_contracts import AnalysisError

    frame = _factorial(seed=11, n=120)
    with pytest.raises(AnalysisError, match="needs the effects it contains") as caught:
        oe.anova(frame, "y", ["a", "b", "c"], covariates=["x"], interactions=option)
    assert caught.value.code == "invalid_spec" and missing in str(caught.value)
    with pytest.raises(AnalysisError, match="needs the effects it contains"):
        oe.manova(frame, ["y", "y2"], ["a", "b", "c"], covariates=["x"], interactions=option)


def test_sequential_sums_of_squares_need_a_marginal_order():
    from openecon.analysis_contracts import AnalysisError

    frame = _factorial(seed=11, n=120)
    scrambled = ["a#b#x", "a#x", "b#x"]
    with pytest.raises(AnalysisError, match="must come after") as caught:
        oe.anova(frame, "y", ["a", "b"], covariates=["x"], interactions=scrambled, ss_type=1)
    assert caught.value.code == "invalid_spec"
    # The order is irrelevant for Type II and III: same table as the marginal order.
    ordered = ["a#x", "b#x", "a#b#x"]
    for ss_type in (2, 3):
        one = oe.anova(frame, "y", ["a", "b"], covariates=["x"], interactions=scrambled,
                       ss_type=ss_type)["anova"]
        two = oe.anova(frame, "y", ["a", "b"], covariates=["x"], interactions=ordered,
                       ss_type=ss_type)["anova"]
        np.testing.assert_allclose(one.loc[two.index, "ss"], two["ss"], rtol=1e-9)
    oe.anova(frame, "y", ["a", "b"], covariates=["x"], interactions=ordered, ss_type=1)


def test_type_two_intercept_is_adjusted_for_covariates_only():
    frame = _factorial(seed=3, n=90)
    table = oe.anova(frame, "y", ["a", "b"], covariates=["x", "w"], ss_type=2)["anova"]
    # Intercept tested in the regression of y on the covariates alone (raw origin).
    design = np.column_stack([np.ones(len(frame)), frame["x"], frame["w"]])
    outcome = frame["y"].to_numpy()
    gram_inverse = np.linalg.inv(design.T @ design)
    beta = gram_inverse @ design.T @ outcome
    assert table.loc["Intercept", "ss"] == pytest.approx(beta[0] ** 2 / gram_inverse[0, 0],
                                                         rel=RTOL)


# ---- anova: heterogeneous slopes are tested at covariate = 0 (raw origin) ----------------


def test_heterogeneous_slopes_model_tests_factors_at_the_raw_origin():
    frame = _factorial(seed=21, n=140)
    result = oe.anova(frame, "y", ["a"], covariates=["x"], homogeneity_of_slopes=True)
    table = result["anova"]
    # Independent parametrization: a separate intercept and slope for each level of a.
    levels = sorted(frame["a"].unique())
    design = np.zeros((len(frame), 2 * len(levels)))
    for i, level in enumerate(levels):
        rows = (frame["a"] == level).to_numpy()
        design[rows, i] = 1.0
        design[rows, len(levels) + i] = frame.loc[rows, "x"]
    outcome = frame["y"].to_numpy()
    gram_inverse = np.linalg.inv(design.T @ design)
    beta = gram_inverse @ design.T @ outcome
    sse = float(((outcome - design @ beta) ** 2).sum())
    size = len(levels)
    difference = np.hstack([np.eye(size - 1), -np.ones((size - 1, 1))])
    average = np.full((1, size), 1.0 / size)
    zeros = np.zeros_like
    expected = {
        "a": np.hstack([difference, zeros(difference)]),          # equal intercepts at x = 0
        "a#x": np.hstack([zeros(difference), difference]),        # equal slopes
        "x": np.hstack([zeros(average), average]),                # average slope
        "Intercept": np.hstack([average, zeros(average)]),        # average intercept at x = 0
    }
    assert table.loc["error", "ss"] == pytest.approx(sse, rel=RTOL)
    for name, matrix in expected.items():
        assert table.loc[name, "ss"] == pytest.approx(_wald_ss(matrix, beta, gram_inverse),
                                                      rel=RTOL), name
    # Shifting the covariate changes the test of a (another origin), not that of the slopes.
    shifted = oe.anova(frame.assign(x=frame["x"] + 3.0), "y", ["a"], covariates=["x"],
                       homogeneity_of_slopes=True)["anova"]
    assert shifted.loc["a#x", "ss"] == pytest.approx(table.loc["a#x", "ss"], rel=RTOL)
    assert shifted.loc["x", "ss"] == pytest.approx(table.loc["x", "ss"], rel=RTOL)
    assert abs(shifted.loc["a", "ss"] / table.loc["a", "ss"] - 1.0) > 1e-3


def _mp_fit(design: list[list], outcome: list) -> tuple:
    """Normal equations solved with 60 significant digits: (beta, (X'X)^{-1}, sse)."""
    x, y = mpmath.matrix(design), mpmath.matrix(outcome)
    gram_inverse = (x.T * x) ** -1
    beta = gram_inverse * (x.T * y)
    resid = y - x * beta
    return beta, gram_inverse, (resid.T * resid)[0]


def test_covariate_with_a_huge_level_loses_no_digits():
    rng = np.random.default_rng(8)
    n = 60
    frame = pd.DataFrame({"g": rng.choice(["p", "q", "r"], size=n), "u": rng.normal(size=n)})
    frame["x"] = 1e8 + frame["u"]
    frame["y"] = 2.0 + 0.7 * frame["u"] + (frame["g"] == "q") + rng.normal(size=n)
    result = oe.anova(frame, "y", ["g"], covariates=["x"], emmeans=["g"])
    table = result["anova"]
    centred = oe.anova(frame.assign(x=frame["x"] - 1e8), "y", ["g"], covariates=["x"],
                       emmeans=["g"])
    for name in ("g", "x", "error", "corrected_model"):
        assert table.loc[name, "ss"] == pytest.approx(centred["anova"].loc[name, "ss"], rel=1e-9)
    for name in ("emmeans_g", "pairwise_g"):
        np.testing.assert_allclose(
            result[name].select_dtypes("number").to_numpy(),
            centred[name].select_dtypes("number").to_numpy(), rtol=1e-8, atol=1e-10)
    # Intercept at x = 0 (1e8 standard deviations away), against exact arithmetic.
    mpmath.mp.dps = 60
    levels = sorted(frame["g"].unique())
    design = [[mpmath.mpf(float(g == level)) for level in levels] + [mpmath.mpf(float(x))]
              for g, x in zip(frame["g"], frame["x"], strict=True)]
    beta, gram_inverse, sse = _mp_fit(design, [mpmath.mpf(float(v)) for v in frame["y"]])
    row = mpmath.matrix([[mpmath.mpf(1) / 3] * 3 + [0]])
    value = (row * beta)[0]
    intercept_ss = value ** 2 / (row * gram_inverse * row.T)[0]
    assert table.loc["Intercept", "ss"] == pytest.approx(float(intercept_ss), rel=1e-7)
    assert table.loc["error", "ss"] == pytest.approx(float(sse), rel=1e-9)
    slope = mpmath.matrix([[0, 0, 0, 1]])
    slope_ss = (slope * beta)[0] ** 2 / (slope * gram_inverse * slope.T)[0]
    assert table.loc["x", "ss"] == pytest.approx(float(slope_ss), rel=1e-7)
    # Type I and II agree with the centred covariate too; with slopes the raw origin is kept.
    for ss_type in (1, 2):
        one = oe.anova(frame, "y", ["g"], covariates=["x"], ss_type=ss_type)["anova"]
        two = oe.anova(frame.assign(x=frame["x"] - 1e8), "y", ["g"], covariates=["x"],
                       ss_type=ss_type)["anova"]
        for name in ("g", "x", "error"):
            assert one.loc[name, "ss"] == pytest.approx(two.loc[name, "ss"], rel=1e-9)


# ---- estimated marginal means ---------------------------------------------------------


@pytest.mark.parametrize("interactions", ["full", "none"])
def test_marginal_means_equal_averaged_predictions_of_a_dummy_regression(interactions):
    frame = _factorial(seed=14, n=130)
    factors, covariates = ["a", "b"], ["x"]
    result = oe.anova(frame, "y", factors, covariates=covariates, interactions=interactions,
                      emmeans=["a", "b", ["a", "b"]], adjust="sidak", alpha=0.1)
    terms = _model_terms(factors, covariates, [(("a", "b"), ())] if interactions == "full" else [])
    design = np.hstack([_columns(frame, term) for term in terms])
    outcome = frame["y"].to_numpy()
    pseudo = np.linalg.pinv(design.T @ design)
    beta = pseudo @ design.T @ outcome
    rank = np.linalg.matrix_rank(design)
    df_error = len(frame) - rank
    mse = float(((outcome - design @ beta) ** 2).sum()) / df_error
    levels = {name: sorted(frame[name].unique()) for name in factors}
    grid = pd.DataFrame(list(itertools.product(levels["a"], levels["b"])), columns=factors)
    grid["x"] = frame["x"].mean()
    grid["y"] = 0.0
    both = pd.concat([frame[["a", "b", "x", "y"]], grid], ignore_index=True)
    rows = np.hstack([_columns(both, term) for term in terms])[len(frame):]
    critical = stats.t.ppf(0.95, df_error)
    for name in factors:
        table = result[f"emmeans_{name}"]
        vectors = np.array([rows[(grid[name] == level).to_numpy()].mean(axis=0)
                            for level in levels[name]])
        means = vectors @ beta
        errors = np.sqrt(mse * np.einsum("ij,jk,ik->i", vectors, pseudo, vectors))
        np.testing.assert_allclose(table["mean"], means, rtol=RTOL)
        np.testing.assert_allclose(table["std_error"], errors, rtol=RTOL)
        np.testing.assert_allclose(table["ci_low"], means - critical * errors, rtol=1e-7)
        np.testing.assert_allclose(table["ci_high"], means + critical * errors, rtol=1e-7)
        assert (table["df"] == df_error).all()
        pairs = result[f"pairwise_{name}"]
        count = len(levels[name]) * (len(levels[name]) - 1) // 2
        level_alpha = 1 - (1 - 0.1) ** (1 / count)
        for k, (i, j) in enumerate(itertools.combinations(range(len(levels[name])), 2)):
            contrast = vectors[i] - vectors[j]
            estimate = contrast @ beta
            se = math.sqrt(mse * contrast @ pseudo @ contrast)
            p = 2 * stats.t.sf(abs(estimate / se), df_error)
            row = pairs.iloc[k]
            assert (row["level_i"], row["level_j"]) == (levels[name][i], levels[name][j])
            assert row["mean_difference"] == pytest.approx(estimate, rel=RTOL, abs=1e-12)
            assert row["std_error"] == pytest.approx(se, rel=RTOL)
            assert row["statistic"] == pytest.approx(estimate / se, rel=RTOL, abs=1e-10)
            assert row["p_value"] == pytest.approx(1 - (1 - p) ** count, rel=1e-7)
            half = stats.t.ppf(1 - level_alpha / 2, df_error) * se
            assert row["ci_low"] == pytest.approx(estimate - half, rel=1e-7, abs=1e-10)
            assert row["ci_high"] == pytest.approx(estimate + half, rel=1e-7, abs=1e-10)
    cells = result["emmeans_a#b"]
    np.testing.assert_allclose(cells["mean"], rows @ beta, rtol=RTOL)
    np.testing.assert_allclose(cells["std_error"],
                               np.sqrt(mse * np.einsum("ij,jk,ik->i", rows, pseudo, rows)),
                               rtol=RTOL)


def test_marginal_means_without_covariates_are_unweighted_means_of_cell_means():
    frame = _factorial(seed=2, n=100)
    result = oe.anova(frame, "y", ["a", "b"], emmeans=["a"], adjust="lsd")
    cell = frame.groupby(["a", "b"])["y"].agg(["mean", "size"])
    mse = result["anova"].loc["error", "ms"]
    for level, row in zip(sorted(frame["a"].unique()), result["emmeans_a"].itertuples(),
                          strict=True):
        means, sizes = cell.loc[level, "mean"], cell.loc[level, "size"]
        assert row.mean == pytest.approx(means.mean(), rel=RTOL)
        assert row.std_error == pytest.approx(math.sqrt(mse * (1 / sizes).sum()) / len(means),
                                              rel=RTOL)


# ---- manova ----------------------------------------------------------------------------


def _sscp(frame, outcomes, factors, covariates):
    """(E, {effect: (H, df)}, df_error) from the cell-means model, all outcomes at once."""
    levels = [sorted(frame[name].unique()) for name in factors]
    cells = list(itertools.product(*levels))
    position = {cell: i for i, cell in enumerate(cells)}
    index = np.array([position[tuple(row)] for row in frame[factors].itertuples(index=False)])
    design = np.zeros((len(frame), len(cells) + len(covariates)))
    design[np.arange(len(frame)), index] = 1.0
    for j, name in enumerate(covariates):
        design[:, len(cells) + j] = frame[name]
    outcome = frame[outcomes].to_numpy(dtype=float)
    gram_inverse = np.linalg.inv(design.T @ design)
    beta = gram_inverse @ design.T @ outcome
    resid = outcome - design @ beta
    hypotheses = {}
    names = {(): "Intercept"}
    for order in range(1, len(factors) + 1):
        for subset in itertools.combinations(range(len(factors)), order):
            names[subset] = "#".join(factors[i] for i in subset)
    for subset, name in names.items():
        matrix = _effect_matrix(levels, subset, len(covariates))
        value = matrix @ beta
        hypotheses[name] = (value.T @ np.linalg.solve(matrix @ gram_inverse @ matrix.T, value),
                            matrix.shape[0])
    for j, name in enumerate(covariates):
        row = np.zeros((1, design.shape[1]))
        row[0, len(cells) + j] = 1.0
        value = row @ beta
        hypotheses[name] = (value.T @ value / (row @ gram_inverse @ row.T), 1)
    return resid.T @ resid, hypotheses, len(frame) - design.shape[1]


def _multivariate_oracle(h, e, q, v):
    """The four statistics and their F approximations, written from Rencher (2002, ch. 6)."""
    p = e.shape[0]
    roots = np.sort(np.linalg.eigvals(np.linalg.solve(e, h)).real)[::-1]
    s = min(p, q)
    roots = np.clip(roots[:s], 0.0, None)
    m, n = (abs(p - q) - 1) / 2, (v - p - 1) / 2
    out = {}
    wilks = np.linalg.det(e) / np.linalg.det(e + h)
    if p * p + q * q - 5 > 0:
        t = math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5))
    else:
        t = 1.0
    w = v + q - (p + q + 1) / 2
    df1, df2 = p * q, w * t - (p * q - 2) / 2
    f = (1 - wilks ** (1 / t)) / wilks ** (1 / t) * df2 / df1
    out["wilks"] = (wilks, f, df1, df2)
    pillai = float(np.trace(h @ np.linalg.inv(h + e)))
    df1, df2 = s * (2 * m + s + 1), s * (2 * n + s + 1)
    out["pillai"] = (pillai, (2 * n + s + 1) / (2 * m + s + 1) * pillai / (s - pillai), df1, df2)
    trace = float(np.trace(np.linalg.solve(e, h)))
    df1, df2 = s * (2 * m + s + 1), 2 * (s * n + 1)
    out["hotelling"] = (trace, 2 * (s * n + 1) * trace / (s * s * (2 * m + s + 1)), df1, df2)
    r = max(p, q)
    out["roy"] = (roots[0], roots[0] * (v - r + q) / r, r, v - r + q)
    return out


@pytest.mark.parametrize("outcomes, factors, covariates", [
    (["y", "y2"], ["a"], []),
    (["y", "y2", "y3"], ["a", "b"], ["x"]),
    (["y", "y2", "y3"], ["a", "c"], []),
])
def test_manova_matches_cell_means_sscp_matrices(outcomes, factors, covariates):
    frame = _factorial(seed=9, n=160)
    result = oe.manova(frame, outcomes, factors, covariates=covariates)
    e, hypotheses, v = _sscp(frame, outcomes, factors, covariates)
    table = result["multivariate"]
    assert result.attrs["df_resid"] == v
    assert set(table["effect"]) == set(hypotheses)
    for name, (h, q) in hypotheses.items():
        expected = _multivariate_oracle(h, e, q, v)
        for test, (value, f, df1, df2) in expected.items():
            row = table[(table["effect"] == name) & (table["test"] == test)].iloc[0]
            assert row["value"] == pytest.approx(value, rel=1e-7), (name, test)
            assert row["statistic"] == pytest.approx(f, rel=1e-7), (name, test)
            assert row["df1"] == pytest.approx(df1) and row["df2"] == pytest.approx(df2)
            assert row["p_value"] == pytest.approx(stats.f.sf(f, df1, df2), rel=1e-6, abs=1e-300)
    # Univariate follow-ups are the diagonals of the same matrices.
    univariate = result["univariate"]
    for j, outcome in enumerate(outcomes):
        block = univariate[univariate["outcome"] == outcome].set_index("source")
        assert block.loc["error", "ss"] == pytest.approx(e[j, j], rel=RTOL)
        for name, (h, _) in hypotheses.items():
            assert block.loc[name, "ss"] == pytest.approx(h[j, j], rel=1e-7)


def test_two_group_manova_is_hotellings_t_squared():
    frame = _factorial(seed=4, n=70)
    outcomes = ["y", "y2", "y3"]
    result = oe.manova(frame, outcomes, ["b"])
    first = frame.loc[frame["b"] == "b1", outcomes].to_numpy()
    second = frame.loc[frame["b"] == "b2", outcomes].to_numpy()
    n1, n2, p = len(first), len(second), len(outcomes)
    pooled = ((n1 - 1) * np.cov(first.T) + (n2 - 1) * np.cov(second.T)) / (n1 + n2 - 2)
    gap = first.mean(0) - second.mean(0)
    t_squared = n1 * n2 / (n1 + n2) * gap @ np.linalg.solve(pooled, gap)
    f = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * t_squared
    p_value = stats.f.sf(f, p, n1 + n2 - p - 1)
    table = result["multivariate"]
    rows = table[table["effect"] == "b"].set_index("test")
    for test in ("pillai", "wilks", "hotelling", "roy"):
        assert rows.loc[test, "statistic"] == pytest.approx(f, rel=1e-8)
        assert rows.loc[test, "df1"] == p and rows.loc[test, "df2"] == n1 + n2 - p - 1
        assert rows.loc[test, "p_value"] == pytest.approx(p_value, rel=1e-7)
        assert rows.loc[test, "f_type"] == "exact"
    assert rows.loc["hotelling", "value"] == pytest.approx(t_squared / (n1 + n2 - 2), rel=1e-8)
    # SPSS's multivariate partial eta squared.
    wilks = rows.loc["wilks", "value"]
    assert rows.loc["wilks", "partial_eta_squared"] == pytest.approx(1 - wilks, rel=1e-8)
    assert rows.loc["pillai", "partial_eta_squared"] == pytest.approx(1 - wilks, rel=1e-8)


def test_wilks_f_is_exact_for_two_outcomes():
    # p = 2: F = [(1 - sqrt L) / sqrt L] (N - g - 1) / (g - 1) on 2(g-1), 2(N-g-1) df.
    frame = _factorial(seed=6, n=90)
    result = oe.manova(frame, ["y", "y2"], ["a"])
    groups = [frame.loc[frame["a"] == level, ["y", "y2"]].to_numpy()
              for level in sorted(frame["a"].unique())]
    within = sum((g - g.mean(0)).T @ (g - g.mean(0)) for g in groups)
    everything = frame[["y", "y2"]].to_numpy()
    total = (everything - everything.mean(0)).T @ (everything - everything.mean(0))
    wilks = np.linalg.det(within) / np.linalg.det(total)
    count, g = len(frame), len(groups)
    f = (1 - math.sqrt(wilks)) / math.sqrt(wilks) * (count - g - 1) / (g - 1)
    table = result["multivariate"]
    row = table[(table["effect"] == "a") & (table["test"] == "wilks")].iloc[0]
    assert row["value"] == pytest.approx(wilks, rel=1e-9)
    assert row["statistic"] == pytest.approx(f, rel=1e-8)
    assert (row["df1"], row["df2"]) == (2 * (g - 1), 2 * (count - g - 1))
    assert row["f_type"] == "exact"


def test_box_m_from_determinants():
    frame = _factorial(seed=12, n=200)
    outcomes = ["y", "y2", "y3"]
    box = oe.manova(frame, outcomes, ["a", "b"])["box_m"].loc["box_m"]
    groups = [g[outcomes].to_numpy() for _, g in frame.groupby(["a", "b"])]
    p, k = len(outcomes), len(groups)
    sizes = np.array([len(g) for g in groups])
    covs = [np.cov(g.T) for g in groups]
    pooled = sum((n - 1) * s for n, s in zip(sizes, covs, strict=True)) / (sizes.sum() - k)
    m = (sizes.sum() - k) * np.log(np.linalg.det(pooled)) \
        - sum((n - 1) * np.log(np.linalg.det(s)) for n, s in zip(sizes, covs, strict=True))
    a1 = (np.sum(1 / (sizes - 1)) - 1 / (sizes.sum() - k)) * (2 * p * p + 3 * p - 1) \
        / (6 * (p + 1) * (k - 1))
    a2 = (np.sum(1 / (sizes - 1) ** 2) - 1 / (sizes.sum() - k) ** 2) * (p - 1) * (p + 2) \
        / (6 * (k - 1))
    df1 = p * (p + 1) * (k - 1) / 2
    df2 = (df1 + 2) / abs(a2 - a1 ** 2)
    assert a2 > a1 ** 2
    f = m * (1 - a1 - df1 / df2) / df1
    assert box["statistic"] == pytest.approx(m, rel=1e-8)
    assert box["chi2"] == pytest.approx(m * (1 - a1), rel=1e-8)
    assert box["chi2_p_value"] == pytest.approx(stats.chi2.sf(m * (1 - a1), df1), rel=1e-7)
    assert box["f"] == pytest.approx(f, rel=1e-8)
    assert box["df1"] == df1 and box["df2"] == pytest.approx(df2, rel=1e-9)
    assert box["p_value"] == pytest.approx(stats.f.sf(f, df1, df2), rel=1e-7)


# ---- repeated measures -----------------------------------------------------------------


def _long(seed: int, sizes: list[int], levels: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    subject = 0
    for group, size in enumerate(sizes):
        for _ in range(size):
            base = rng.normal()
            for t in range(levels):
                rows.append((subject, f"g{group}", t,
                             base + 0.3 * t + 0.25 * group * t + rng.normal(scale=1 + 0.3 * t)))
            subject += 1
    frame = pd.DataFrame(rows, columns=["id", "g", "t", "y"])
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _box_epsilon(cov: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from the covariance matrix itself (Box 1954)."""
    p = cov.shape[0]
    grand, diagonal, rows = cov.mean(), np.trace(cov) / p, cov.mean(axis=1)
    return p ** 2 * (diagonal - grand) ** 2 / (
        (p - 1) * ((cov ** 2).sum() - 2 * p * (rows ** 2).sum() + p ** 2 * grand ** 2))


def test_one_way_repeated_measures_from_classical_sums_of_squares():
    frame = _long(31, [14], 4)
    result = oe.rm_anova(frame, "y", "id", ["t"])
    wide = frame.pivot(index="id", columns="t", values="y").to_numpy()
    s, p = wide.shape
    grand = wide.mean()
    ss_time = s * ((wide.mean(0) - grand) ** 2).sum()
    ss_subject = p * ((wide.mean(1) - grand) ** 2).sum()
    ss_error = ((wide - grand) ** 2).sum() - ss_time - ss_subject
    f = ss_time / (p - 1) / (ss_error / ((p - 1) * (s - 1)))
    within = result["within"].set_index(["source", "correction"])
    row = within.loc[("t", "sphericity_assumed")]
    assert row["ss"] == pytest.approx(ss_time, rel=RTOL)
    assert row["df"] == p - 1
    assert row["statistic"] == pytest.approx(f, rel=RTOL)
    assert row["p_value"] == pytest.approx(stats.f.sf(f, p - 1, (p - 1) * (s - 1)), rel=1e-7)
    assert row["partial_eta_squared"] == pytest.approx(ss_time / (ss_time + ss_error), rel=RTOL)
    error = within.loc[("error(t)", "sphericity_assumed")]
    assert error["ss"] == pytest.approx(ss_error, rel=RTOL)
    assert error["df"] == (p - 1) * (s - 1)
    between = result["between"]
    assert between.loc["error", "ss"] == pytest.approx(ss_subject, rel=RTOL)
    assert between.loc["Intercept", "ss"] == pytest.approx(s * p * grand ** 2, rel=RTOL)
    # Sphericity: Box's epsilon from the covariance matrix, Mauchly's W from the
    # eigenvalues of the doubly centred covariance matrix (no contrast basis needed).
    cov = np.cov(wide.T)
    gg = _box_epsilon(cov)
    d = p - 1
    centring = np.eye(p) - 1.0 / p
    roots = np.sort(np.linalg.eigvalsh(centring @ cov @ centring))[1:]
    w = np.prod(roots) / (roots.sum() / d) ** d
    chi2 = -((s - 1) - (2 * d * d + d + 2) / (6 * d)) * math.log(w)
    hf = min(1.0, (s * d * gg - 2) / (d * (s - 1 - d * gg)))
    sphericity = result["sphericity"].loc["t"]
    assert sphericity["epsilon_gg"] == pytest.approx(gg, rel=1e-8)
    assert sphericity["epsilon_hf"] == pytest.approx(hf, rel=1e-8)
    assert sphericity["epsilon_lb"] == pytest.approx(1 / d)
    assert sphericity["mauchly_w"] == pytest.approx(w, rel=1e-8)
    assert sphericity["chi2"] == pytest.approx(chi2, rel=1e-8)
    assert sphericity["df"] == d * (d + 1) // 2 - 1
    assert sphericity["p_value"] == pytest.approx(stats.chi2.sf(chi2, d * (d + 1) / 2 - 1),
                                                  rel=1e-7)
    for name, eps in (("greenhouse_geisser", gg), ("huynh_feldt", hf), ("lower_bound", 1 / d)):
        corrected = within.loc[("t", name)]
        assert corrected["df"] == pytest.approx(eps * d, rel=1e-8)
        assert corrected["statistic"] == pytest.approx(f, rel=RTOL)
        assert corrected["p_value"] == pytest.approx(
            stats.f.sf(f, eps * d, eps * d * (s - 1)), rel=1e-6)
        assert within.loc[("error(t)", name), "df"] == pytest.approx(eps * d * (s - 1), rel=1e-8)


def test_balanced_split_plot_from_classical_sums_of_squares():
    frame = _long(17, [9, 9, 9], 3)
    result = oe.rm_anova(frame, "y", "id", ["t"], between=["g"])
    groups = sorted(frame["g"].unique())
    wide = {g: frame[frame["g"] == g].pivot(index="id", columns="t", values="y").to_numpy()
            for g in groups}
    stacked = np.vstack(list(wide.values()))
    n, p, a = 9, 3, len(groups)
    grand = stacked.mean()
    group_means = np.array([wide[g].mean() for g in groups])
    time_means = stacked.mean(0)
    cell_means = np.array([wide[g].mean(0) for g in groups])
    ss_group = n * p * ((group_means - grand) ** 2).sum()
    ss_subjects = p * sum(((wide[g].mean(1) - wide[g].mean()) ** 2).sum() for g in groups)
    ss_time = a * n * ((time_means - grand) ** 2).sum()
    ss_inter = n * ((cell_means - group_means[:, None] - time_means[None, :] + grand) ** 2).sum()
    ss_error = ((stacked - grand) ** 2).sum() - ss_group - ss_subjects - ss_time - ss_inter
    within = result["within"].set_index(["source", "correction"])
    between = result["between"]
    df_error = (p - 1) * a * (n - 1)
    checks = [(within.loc[("t", "sphericity_assumed")], ss_time, p - 1, ss_error, df_error),
              (within.loc[("t#g", "sphericity_assumed")], ss_inter, (p - 1) * (a - 1), ss_error,
               df_error),
              (between.loc["g"], ss_group, a - 1, ss_subjects, a * (n - 1))]
    for row, ss, df, error, error_df in checks:
        f = ss / df / (error / error_df)
        assert row["ss"] == pytest.approx(ss, rel=RTOL)
        assert row["df"] == df
        assert row["statistic"] == pytest.approx(f, rel=RTOL)
        assert row["p_value"] == pytest.approx(stats.f.sf(f, df, error_df), rel=1e-7)
    assert within.loc[("error(t)", "sphericity_assumed"), "ss"] == pytest.approx(ss_error,
                                                                               rel=RTOL)
    assert between.loc["error", "ss"] == pytest.approx(ss_subjects, rel=RTOL)
    # Epsilon from the pooled within-group covariance matrix; SPSS's Huynh-Feldt uses N.
    pooled = sum((n - 1) * np.cov(wide[g].T) for g in groups) / (a * (n - 1))
    gg = _box_epsilon(pooled)
    d, subjects, v = p - 1, a * n, a * (n - 1)
    sphericity = result["sphericity"].loc["t"]
    assert sphericity["epsilon_gg"] == pytest.approx(gg, rel=1e-8)
    assert sphericity["epsilon_hf"] == pytest.approx(
        min(1.0, (subjects * d * gg - 2) / (d * (v - d * gg))), rel=1e-8)


def test_unbalanced_split_plot_uses_unweighted_group_means():
    frame = _long(23, [6, 11, 8], 3)
    result = oe.rm_anova(frame, "y", "id", ["t"], between=["g"])
    groups = sorted(frame["g"].unique())
    wide = {g: frame[frame["g"] == g].pivot(index="id", columns="t", values="y").to_numpy()
            for g in groups}
    sizes = np.array([len(wide[g]) for g in groups])
    p, a = 3, len(groups)
    # Any orthonormal basis of the within-subject contrasts gives the same traces.
    basis = np.linalg.qr(np.column_stack([np.ones(p), [1.0, 5.0, -2.0], [0.3, -1.0, 4.0]]))[0]
    basis = basis[:, 1:]
    scores = {g: wide[g] @ basis for g in groups}
    means = np.array([scores[g].mean(0) for g in groups])
    error = sum(((scores[g] - scores[g].mean(0)) ** 2).sum() for g in groups)
    df_error = (p - 1) * (sizes.sum() - a)
    # Type III main effect of t: the UNWEIGHTED average over groups of the contrast means.
    ss_time = (means.mean(0) ** 2).sum() / ((1 / sizes).sum() / a ** 2)
    weighted = (sizes[:, None] * means).sum(0) / sizes.sum()
    ss_inter = (sizes[:, None] * (means - weighted) ** 2).sum()
    within = result["within"].set_index(["source", "correction"])
    for name, ss, df in (("t", ss_time, p - 1), ("t#g", ss_inter, (p - 1) * (a - 1))):
        row = within.loc[(name, "sphericity_assumed")]
        f = ss / df / (error / df_error)
        assert row["ss"] == pytest.approx(ss, rel=RTOL)
        assert row["statistic"] == pytest.approx(f, rel=RTOL)
        assert row["p_value"] == pytest.approx(stats.f.sf(f, df, df_error), rel=1e-7)
    assert within.loc[("error(t)", "sphericity_assumed"), "ss"] == pytest.approx(error, rel=RTOL)
    # Between subjects: one-way ANOVA of the subject means times sqrt(p) (Type III = usual F).
    subject_means = [wide[g].mean(1) for g in groups]
    f, p_value = stats.f_oneway(*subject_means)
    assert result["between"].loc["g", "statistic"] == pytest.approx(f, rel=RTOL)
    assert result["between"].loc["g", "p_value"] == pytest.approx(p_value, rel=1e-7)
    # Cell descriptives.
    described = result["descriptives"]
    for g in groups:
        for t in range(p):
            row = described[(described["g"] == g) & (described["t"] == t)].iloc[0]
            assert row["n"] == len(wide[g])
            assert row["mean"] == pytest.approx(wide[g][:, t].mean(), rel=RTOL)
            assert row["std_dev"] == pytest.approx(wide[g][:, t].std(ddof=1), rel=RTOL)


def test_two_within_factors_from_classical_sums_of_squares():
    rng = np.random.default_rng(40)
    s, p, q = 10, 3, 2
    cube = rng.normal(size=(s, p, q)) + rng.normal(size=(s, 1, 1)) \
        + 0.4 * np.arange(p)[None, :, None] + 0.3 * np.arange(q)[None, None, :]
    rows = [(i, f"u{j}", f"v{k}", cube[i, j, k]) for i in range(s) for j in range(p)
            for k in range(q)]
    frame = pd.DataFrame(rows, columns=["id", "u", "v", "y"]).sample(frac=1.0, random_state=1)
    result = oe.rm_anova(frame, "y", "id", ["u", "v"])
    grand = cube.mean()
    subj = cube.mean(axis=(1, 2))
    u, v = cube.mean(axis=(0, 2)), cube.mean(axis=(0, 1))
    uv = cube.mean(axis=0)
    su, sv = cube.mean(axis=2), cube.mean(axis=1)
    ss_u = s * q * ((u - grand) ** 2).sum()
    ss_v = s * p * ((v - grand) ** 2).sum()
    ss_uv = s * ((uv - u[:, None] - v[None, :] + grand) ** 2).sum()
    ss_su = q * ((su - subj[:, None] - u[None, :] + grand) ** 2).sum()
    ss_sv = p * ((sv - subj[:, None] - v[None, :] + grand) ** 2).sum()
    ss_subj = p * q * ((subj - grand) ** 2).sum()
    ss_suv = ((cube - grand) ** 2).sum() - ss_u - ss_v - ss_uv - ss_su - ss_sv - ss_subj
    within = result["within"].set_index(["source", "correction"])
    for name, ss, df, error in (("u", ss_u, p - 1, ss_su), ("v", ss_v, q - 1, ss_sv),
                                ("u#v", ss_uv, (p - 1) * (q - 1), ss_suv)):
        row = within.loc[(name, "sphericity_assumed")]
        f = ss / df / (error / (df * (s - 1)))
        assert row["ss"] == pytest.approx(ss, rel=RTOL)
        assert row["df"] == df
        assert within.loc[(f"error({name})", "sphericity_assumed"), "ss"] == pytest.approx(
            error, rel=RTOL)
        assert row["statistic"] == pytest.approx(f, rel=RTOL)
        assert row["p_value"] == pytest.approx(stats.f.sf(f, df, df * (s - 1)), rel=1e-7)
    # A two-level factor has a single contrast: no sphericity test, all epsilons one.
    assert result["sphericity"].loc["v", "epsilon_gg"] == 1.0
    assert np.isnan(result["sphericity"].loc["v", "chi2"])
    # Epsilon of u from the covariance matrix of the u means of each subject.
    assert result["sphericity"].loc["u", "epsilon_gg"] == pytest.approx(
        _box_epsilon(np.cov(su.T)), rel=1e-8)
