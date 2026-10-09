"""Adversarial inputs and regression tests for the multivariate family.

Every invalid input must raise AnalysisError with a snake_case code; every
accepted degenerate input must give finite, JSON-safe output. The regression
tests at the end pin defects found by the verification pass.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import optimize as sopt

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import kmeans

RNG = np.random.default_rng(0)
X = RNG.normal(size=(60, 4))
NAMES = list("abcd")
DF = pd.DataFrame(X + (np.arange(60) // 20)[:, None], columns=NAMES)
DF["g"] = np.repeat(["u", "v", "w"], 20)
CAT = pd.DataFrame({"r": RNG.choice(["a", "b", "c"], 100), "c": RNG.choice(list("xyzw"), 100)})


def _with(**columns) -> pd.DataFrame:
    return DF.assign(**columns)


CONSTANT = _with(a=1.0)
COLLINEAR = _with(d=DF["a"] + DF["b"])
STRINGS = _with(a="x")
INFINITE = _with(a=[np.inf] + [0.0] * 59)
HOLE = _with(a=[np.nan] + list(DF["a"][1:]))
DUPLICATES = pd.DataFrame({"a": [1.0] * 10 + [2.0] * 10, "b": [0.0] * 20})

ERRORS = [
    ("pca empty", lambda: oe.pca(DF.iloc[:0], NAMES), "empty_data"),
    ("pca one row", lambda: oe.pca(DF.iloc[:1], NAMES), "insufficient_observations"),
    ("pca all missing", lambda: oe.pca(_with(a=np.nan), NAMES), "empty_sample"),
    ("pca constant", lambda: oe.pca(CONSTANT, NAMES), "constant_column"),
    ("pca strings", lambda: oe.pca(STRINGS, NAMES), "non_numeric_column"),
    ("pca infinite", lambda: oe.pca(INFINITE, NAMES), "non_finite_values"),
    ("pca huge", lambda: oe.pca(_with(a=1e200), NAMES), "non_finite_values"),
    ("pca bare string", lambda: oe.pca(DF, "a"), "invalid_spec"),
    ("pca repeated", lambda: oe.pca(DF, ["a", "a"]), "invalid_spec"),
    ("pca absent", lambda: oe.pca(DF, ["a", "zz"]), "missing_columns"),
    ("pca one column", lambda: oe.pca(DF, ["a"]), "invalid_spec"),
    ("pca components 0", lambda: oe.pca(DF, NAMES, components=0), "invalid_option"),
    ("pca components bool", lambda: oe.pca(DF, NAMES, components=True), "invalid_option"),
    ("pca mineigen high", lambda: oe.pca(DF, NAMES, mineigen=1e3), "no_components"),
    ("pca matrix", lambda: oe.pca(DF, NAMES, matrix="spearman"), "invalid_option"),
    ("pca raise", lambda: oe.pca(HOLE, NAMES, missing="raise"), "missing_values"),
    ("pca missing option", lambda: oe.pca(DF, NAMES, missing="pairwise"), "invalid_option"),
    ("pca no data", lambda: oe.pca(None, NAMES), "invalid_data"),
    ("pca_scores foreign", lambda: oe.pca_scores(oe.alpha(DF, NAMES), DF), "invalid_result"),
    ("pca_scores absent", lambda: oe.pca_scores(oe.pca(DF, NAMES), DF[["a"]]),
     "missing_columns"),
    ("factor n <= p", lambda: oe.factor(DF.iloc[:4], NAMES), "singular_matrix"),
    ("factor collinear", lambda: oe.factor(COLLINEAR, NAMES), "singular_matrix"),
    ("factor collinear ml", lambda: oe.factor(COLLINEAR, NAMES, method="ml"),
     "singular_matrix"),
    ("factor two columns", lambda: oe.factor(DF, ["a", "b"]), "invalid_spec"),
    ("factor ipf limit", lambda: oe.factor(DF, NAMES, method="ipf", max_iterations=1),
     "no_convergence"),
    ("factor method", lambda: oe.factor(DF, NAMES, method="minres"), "invalid_option"),
    ("factor rotate", lambda: oe.factor(DF, NAMES, rotate="unknown"), "invalid_option"),
    ("factor power", lambda: oe.factor(DF, NAMES, rotate="promax", power=0.5),
     "invalid_option"),
    ("factor gamma", lambda: oe.factor(DF, NAMES, rotate="oblimin", gamma=2),
     "invalid_option"),
    ("factor kaiser", lambda: oe.factor(DF, NAMES, kaiser="yes"), "invalid_option"),
    ("factor scores", lambda: oe.factor(DF, NAMES, scores="anderson"), "invalid_option"),
    ("factor tolerance", lambda: oe.factor(DF, NAMES, tolerance=0.0), "invalid_option"),
    ("factor few rows", lambda: oe.factor(DF.iloc[:2], NAMES, method="pcf"),
     "insufficient_observations"),
    ("factor_scores foreign", lambda: oe.factor_scores(oe.pca(DF, NAMES), DF),
     "invalid_result"),
    ("factortest collinear", lambda: oe.factortest(COLLINEAR, NAMES), "singular_matrix"),
    ("alpha one item", lambda: oe.alpha(DF, ["a"]), "invalid_spec"),
    ("alpha zero scale variance", lambda: oe.alpha(pd.DataFrame({"a": X[:, 0], "b": -X[:, 0]}),
                                                   ["a", "b"]), "degenerate_scale"),
    ("alpha constant", lambda: oe.alpha(CONSTANT, NAMES), "constant_column"),
    ("alpha reverse unknown", lambda: oe.alpha(DF, NAMES, reverse=["z"]), "invalid_spec"),
    ("alpha reverse string", lambda: oe.alpha(DF, NAMES, reverse="a"), "invalid_spec"),
    ("alpha model", lambda: oe.alpha(DF, NAMES, model="parallel"), "invalid_option"),
    ("alpha one row", lambda: oe.alpha(DF.iloc[:1], NAMES), "insufficient_observations"),
    ("kmeans k > n", lambda: oe.cluster_kmeans(DF.iloc[:3], NAMES, 4), "too_many_clusters"),
    ("kmeans k float", lambda: oe.cluster_kmeans(DF, NAMES, 2.0), "invalid_option"),
    ("kmeans duplicate start", lambda: oe.cluster_kmeans(DUPLICATES, ["a", "b"], 3),
     "duplicate_centers"),
    ("kmeans++ too few distinct", lambda: oe.cluster_kmeans(DUPLICATES, ["a", "b"], 3,
                                                           init="kmeans++", seed=1),
     "duplicate_centers"),
    ("kmeans standardize constant", lambda: oe.cluster_kmeans(DUPLICATES, ["a", "b"], 2,
                                                              standardize=True),
     "constant_column"),
    ("kmeans init shape", lambda: oe.cluster_kmeans(DF, NAMES, 2, init=[[1, 2]]),
     "invalid_option"),
    ("kmeans init name", lambda: oe.cluster_kmeans(DF, NAMES, 2, init="far"), "invalid_option"),
    ("kmeans init nan", lambda: oe.cluster_kmeans(DF, NAMES, 2, init=[[np.nan] * 4, [0] * 4]),
     "invalid_option"),
    ("kmeans seed", lambda: oe.cluster_kmeans(DF, NAMES, 2, init="random", seed=-1),
     "invalid_option"),
    ("kmeans tolerance", lambda: oe.cluster_kmeans(DF, NAMES, 2, tolerance=2.0),
     "invalid_option"),
    ("cluster_assign foreign", lambda: oe.cluster_assign(oe.pca(DF, NAMES), DF),
     "invalid_result"),
    ("hierarchical one row", lambda: oe.cluster_hierarchical(DF.iloc[:1], NAMES),
     "insufficient_observations"),
    ("hierarchical guard", lambda: oe.cluster_hierarchical(DF, NAMES, max_n=10),
     "too_many_observations"),
    ("hierarchical ward manhattan", lambda: oe.cluster_hierarchical(DF, NAMES,
                                                                    metric="manhattan"),
     "invalid_option"),
    ("hierarchical correlation one var", lambda: oe.cluster_hierarchical(
        DF, ["a"], linkage="average", metric="correlation"), "constant_row"),
    ("hierarchical linkage", lambda: oe.cluster_hierarchical(DF, NAMES, linkage="mcquitty"),
     "invalid_option"),
    ("cut neither", lambda: oe.cluster_cut(oe.cluster_hierarchical(DF, NAMES)),
     "invalid_option"),
    ("cut both", lambda: oe.cluster_cut(oe.cluster_hierarchical(DF, NAMES), k=2, height=1.0),
     "invalid_option"),
    ("cut k > n", lambda: oe.cluster_cut(oe.cluster_hierarchical(DF, NAMES), k=61),
     "invalid_option"),
    ("cut short data", lambda: oe.cluster_cut(oe.cluster_hierarchical(DF, NAMES), k=2,
                                              data=DF.iloc[:10]), "invalid_data"),
    ("discrim one group", lambda: oe.discrim(_with(g="u"), "g", NAMES), "invalid_groups"),
    ("discrim collinear", lambda: oe.discrim(COLLINEAR, "g", NAMES), "singular_matrix"),
    ("discrim constant within", lambda: oe.discrim(
        _with(d=(np.arange(60) // 20).astype(float)), "g", NAMES), "singular_matrix"),
    ("discrim group as variable", lambda: oe.discrim(DF, "a", ["a", "b"]), "invalid_spec"),
    ("discrim priors length", lambda: oe.discrim(DF, "g", NAMES, priors=[0.5, 0.5]),
     "invalid_option"),
    ("discrim priors zero", lambda: oe.discrim(DF, "g", NAMES, priors=[0.0, 0.5, 0.5]),
     "invalid_option"),
    ("discrim priors name", lambda: oe.discrim(DF, "g", NAMES, priors="size"),
     "invalid_option"),
    ("discrim singleton loo", lambda: oe.discrim(_with(g=["z"] + list(DF["g"][1:])), "g",
                                                 NAMES, loo=True),
     "insufficient_observations"),
    ("discrim qda small group", lambda: oe.discrim(
        DF.iloc[list(range(3)) + list(range(20, 60))], "g", NAMES, method="qda"),
     "singular_matrix"),
    ("discrim strings", lambda: oe.discrim(STRINGS, "g", NAMES), "non_numeric_column"),
    ("discrim_predict foreign", lambda: oe.discrim_predict(oe.pca(DF, NAMES), DF),
     "invalid_result"),
    ("canon overlap", lambda: oe.canon(DF, x=["a", "b"], y=["b", "c"]), "invalid_spec"),
    ("canon collinear", lambda: oe.canon(COLLINEAR, x=["a", "b", "d"], y=["c"]),
     "singular_matrix"),
    ("canon few rows", lambda: oe.canon(DF.iloc[:4], x=["a", "b"], y=["c"]),
     "insufficient_observations"),
    ("canon constant", lambda: oe.canon(_with(c=1.0), x=["a", "b"], y=["c"]),
     "constant_column"),
    ("canon empty set", lambda: oe.canon(DF, x=[], y=["c"]), "invalid_spec"),
    ("mds both inputs", lambda: oe.mds(DF, NAMES, distances=[[0, 1], [1, 0]]), "invalid_spec"),
    ("mds no input", lambda: oe.mds(), "invalid_spec"),
    ("mds asymmetric", lambda: oe.mds(distances=[[0, 1, 2], [1.5, 0, 1], [2, 1, 0]]),
     "invalid_distances"),
    ("mds negative", lambda: oe.mds(distances=[[0, -1], [-1, 0]]), "invalid_distances"),
    ("mds non-square", lambda: oe.mds(distances=[[0, 1, 2], [1, 0, 1]]), "invalid_distances"),
    ("mds missing", lambda: oe.mds(distances=[[0, np.nan], [np.nan, 0]]), "invalid_distances"),
    ("mds text", lambda: oe.mds(distances=[["a", "b"], ["c", "d"]]), "invalid_distances"),
    ("mds diagonal", lambda: oe.mds(distances=[[1, 1], [1, 0]]), "invalid_distances"),
    ("mds zero", lambda: oe.mds(distances=np.zeros((4, 4))), "too_many_dimensions"),
    ("mds too many dims", lambda: oe.mds(DF, ["a"], dimensions=2), "too_many_dimensions"),
    ("mds guard", lambda: oe.mds(DF, NAMES, max_n=10), "too_many_observations"),
    ("mds method", lambda: oe.mds(DF, NAMES, method="alscal"), "invalid_option"),
    ("ca one row category", lambda: oe.ca(CAT.assign(r="a"), "r", "c"), "degenerate_table"),
    ("ca same column", lambda: oe.ca(CAT, "r", "r"), "invalid_spec"),
    ("ca negative weights", lambda: oe.ca(CAT.assign(w=-1.0), "r", "c", weights="w"),
     "invalid_weights"),
    ("ca zero weights", lambda: oe.ca(CAT.assign(w=0.0), "r", "c", weights="w"),
     "degenerate_table"),
    ("ca text weights", lambda: oe.ca(CAT.assign(w="1"), "r", "c", weights="w"),
     "non_numeric_column"),
    ("ca independence", lambda: oe.ca(pd.DataFrame({"r": list("aabb"), "c": list("xyxy")}),
                                      "r", "c"), "degenerate_table"),
    ("ca dimensions", lambda: oe.ca(CAT, "r", "c", dimensions=0), "invalid_option"),
]


@pytest.mark.parametrize("label,call,code", ERRORS, ids=[item[0] for item in ERRORS])
def test_invalid_inputs_raise_analysis_errors(label, call, code):
    with pytest.raises(AnalysisError) as info:
        call()
    assert info.value.code == code
    assert len(str(info.value)) > 20                       # a message that says what to change


def _finite_and_json_safe(result) -> None:
    tables = result.values() if isinstance(result, dict) else [result]
    for table in tables:
        values = table.select_dtypes("number").to_numpy(dtype=float)
        assert not np.isinf(values).any()
    json.dumps(result.attrs, allow_nan=False)


DEGENERATE = [
    ("pca two rows", lambda: oe.pca(DF.iloc[:2], NAMES)),
    ("pca n < p", lambda: oe.pca(DF.iloc[:3], NAMES)),
    ("pca collinear", lambda: oe.pca(COLLINEAR, NAMES)),
    ("pca boolean column", lambda: oe.pca(_with(a=DF["a"] > 1), NAMES)),
    ("pca records", lambda: oe.pca(DF.to_dict("records"), NAMES)),
    ("factor pcf collinear rotated", lambda: oe.factor(COLLINEAR, NAMES, method="pcf",
                                                       factors=2, rotate="varimax")),
    ("factor ml saturated", lambda: oe.factor(DF, ["a", "b", "c"], method="ml", factors=1)),
    ("factor ml too many", lambda: oe.factor(DF, NAMES, method="ml", factors=3)),
    ("factor single rotated", lambda: oe.factor(DF, NAMES, factors=1, rotate="promax")),
    ("factor pcf two columns", lambda: oe.factor(DF, ["a", "b"], method="pcf")),
    ("factortest minimal", lambda: oe.factortest(DF.iloc[:5], NAMES)),
    ("alpha two items split", lambda: oe.alpha(DF, ["a", "b"], model="split")),
    ("alpha two rows guttman", lambda: oe.alpha(DF.iloc[:2], NAMES, model="guttman")),
    ("alpha collinear guttman", lambda: oe.alpha(COLLINEAR, NAMES, model="guttman")),
    ("kmeans k = 1", lambda: oe.cluster_kmeans(DF, NAMES, 1)),
    ("kmeans k = n", lambda: oe.cluster_kmeans(DF.iloc[:5], NAMES, 5)),
    ("kmeans one iteration", lambda: oe.cluster_kmeans(DF, NAMES, 3, max_iterations=1)),
    ("kmeans huge", lambda: oe.cluster_kmeans(DF[NAMES] * 1e100, NAMES, 3)),
    ("kmeans tiny", lambda: oe.cluster_kmeans(DF[NAMES] * 1e-100, NAMES, 3)),
    ("hierarchical two rows", lambda: oe.cluster_hierarchical(DF.iloc[:2], NAMES)),
    ("hierarchical identical", lambda: oe.cluster_hierarchical(DUPLICATES.iloc[:6], ["a", "b"],
                                                               linkage="centroid")),
    ("discrim singleton group", lambda: oe.discrim(_with(g=["z"] + list(DF["g"][1:])), "g",
                                                   NAMES)),
    ("discrim one variable loo", lambda: oe.discrim(DF, "g", ["a"], method="qda", loo=True)),
    ("discrim mixed labels", lambda: oe.discrim(_with(g=[1] * 20 + ["x"] * 20 + [2.5] * 20),
                                                "g", NAMES)),
    ("discrim boolean groups", lambda: oe.discrim(_with(g=np.arange(60) < 30), "g", NAMES)),
    ("canon perfect", lambda: oe.canon(_with(c=DF["a"] * 2 + 1), x=["a", "b"], y=["c"])),
    ("canon minimal n", lambda: oe.canon(DF.iloc[:5], x=["a", "b"], y=["c"])),
    ("mds two objects", lambda: oe.mds(distances=[[0, 1], [1, 0]], dimensions=1)),
    ("mds duplicate rows smacof", lambda: oe.mds(pd.concat([DF.iloc[:5]] * 2), NAMES,
                                                 method="smacof")),
    ("ca more dimensions than exist", lambda: oe.ca(CAT, "r", "c", dimensions=5)),
    ("ca huge weights", lambda: oe.ca(CAT.assign(w=1e140), "r", "c", weights="w")),
]


@pytest.mark.parametrize("label,call", DEGENERATE, ids=[item[0] for item in DEGENERATE])
def test_degenerate_but_valid_inputs_give_finite_json_safe_results(label, call):
    _finite_and_json_safe(call())


def test_correlation_based_results_are_invariant_to_extreme_units():
    scaled = DF.assign(a=DF["a"] * 1e-8, b=DF["b"] * 1e8 + 1e9)
    for call in (lambda d: oe.pca(d, NAMES)["loadings"],
                 lambda d: oe.factor(d, NAMES, method="ml", factors=1)["loadings"],
                 lambda d: oe.alpha(d, NAMES, standardized=True)["items"],
                 lambda d: oe.discrim(d, "g", NAMES)["structure_matrix"],
                 lambda d: oe.canon(d, x=["a", "b"], y=["c", "d"])["correlations"]):
        np.testing.assert_allclose(call(scaled).to_numpy(dtype=float),
                                   call(DF).to_numpy(dtype=float), rtol=1e-6, atol=1e-8)
    a = oe.cluster_kmeans(DF, NAMES, 3, standardize=True)
    b = oe.cluster_kmeans(scaled, NAMES, 3, standardize=True)
    np.testing.assert_allclose(a["centers"].to_numpy(), b["centers"].to_numpy(), atol=1e-7)


def test_missing_rows_are_dropped_listwise_and_reported_everywhere():
    holes = DF.copy()
    holes.loc[[2, 30], "b"] = np.nan
    holes.loc[45, "g"] = None
    for result in (oe.pca(holes, NAMES), oe.factor(holes, NAMES, factors=1),
                   oe.factortest(holes, NAMES), oe.alpha(holes, NAMES),
                   oe.cluster_kmeans(holes, NAMES, 2), oe.cluster_hierarchical(holes, NAMES),
                   oe.canon(holes, x=["a", "b"], y=["c", "d"]), oe.mds(holes, NAMES)):
        assert result.attrs["n_missing"] == 2 and result.attrs["n"] == 58
    assert oe.discrim(holes, "g", NAMES).attrs["n_missing"] == 3
    cut = oe.cluster_cut(oe.cluster_hierarchical(holes, NAMES), k=3, data=holes)
    assert len(cut) == 60 and cut["cluster"].isna().sum() == 2


# ---- regression tests for defects fixed by the verification pass --------------------


def test_regression_standardized_alpha_reports_standardized_interitem_covariance():
    # The average inter-item covariance used to be that of the raw items even when the
    # analysis was of standardized items (whose covariances are the correlations).
    result = oe.alpha(DF, NAMES, standardized=True)
    rbar = result["scale"].loc["average_interitem_correlation", "value"]
    assert result["scale"].loc["average_interitem_covariance", "value"] == pytest.approx(rbar)


def _spss_start(x: np.ndarray, k: int) -> np.ndarray:
    c = x[:k].copy()
    for row in x[k:]:
        d = ((c - row) ** 2).sum(1)
        cc = ((c[:, None] - c[None]) ** 2).sum(2)
        np.fill_diagonal(cc, np.inf)
        a, b = np.unravel_index(np.argmin(cc), cc.shape)
        q = int(np.argmin(d))
        if d[q] > cc[a, b]:
            c[a if d[a] < d[b] else b] = row
        elif np.delete(d, q).min() > cc[q].min():
            c[q] = row
    return c


@pytest.mark.parametrize("kind", ["integers", "sorted", "zigzag", "continuous"])
def test_regression_spss_start_paths_agree_on_dense_replacements(kind, monkeypatch):
    # Sorted data replace a centre at almost every case. That used to cost one block of
    # tensor operations per case (about 70 s for a million rows); runs of replacements
    # are now stepped on Python floats. Both paths must reproduce the case-by-case rule.
    rng = np.random.default_rng(4)
    n = 1500
    x = {"integers": rng.integers(0, 5, size=(n, 3)).astype(float),
         "sorted": np.column_stack([np.arange(n, dtype=float), rng.integers(0, 3, n)]),
         "zigzag": (np.where(np.arange(n) % 2, 1.0, -1.0) * np.arange(n))[:, None],
         "continuous": np.column_stack([np.sort(rng.normal(size=n)), rng.normal(size=n)])}[kind]
    tensor = torch.as_tensor(x)
    for k in (2, 3, 5):
        expected = _spss_start(x, k)
        np.testing.assert_array_equal(kmeans.spss_initial(tensor, k).numpy(), expected)
        monkeypatch.setattr(kmeans, "_STEPWISE_WORK", 0)
        np.testing.assert_array_equal(kmeans.spss_initial(tensor, k).numpy(), expected)
        monkeypatch.undo()


def test_regression_spss_start_on_sorted_data_is_fast():
    n = 200_000
    x = torch.cat([torch.arange(n, dtype=torch.float64)[:, None],
                   torch.randn(n, 2, dtype=torch.float64, generator=torch.Generator()
                               .manual_seed(0))], 1)
    start = time.perf_counter()
    kmeans.spss_initial(x, 4)
    assert time.perf_counter() - start < 6.0            # was about 14 s before the fix


def test_regression_explicit_centres_are_in_variable_units_when_standardizing():
    x = DF[NAMES].to_numpy()
    mean, std = x.mean(0), x.std(0, ddof=1)
    raw = x[[0, 25, 50]]
    a = oe.cluster_kmeans(DF, NAMES, 3, init=raw.tolist(), standardize=True)
    b = oe.cluster_kmeans(DF.iloc[[0, 25, 50] + [i for i in range(60) if i not in (0, 25, 50)]],
                          NAMES, 3, standardize=True)
    np.testing.assert_allclose(a["initial_centers"].to_numpy(), (raw - mean) / std, atol=1e-12)
    np.testing.assert_allclose(a["centers"].to_numpy(), b["centers"].to_numpy(), atol=1e-10)


def test_regression_ml_heywood_note_names_the_uniqueness_bound():
    rng = np.random.default_rng(0)
    f = rng.normal(size=200)
    frame = pd.DataFrame({"a": f + 0.01 * rng.normal(size=200), "b": f + 0.8 * rng.normal(size=200),
                          "c": f + rng.normal(size=200), "d": rng.normal(size=200) + 0.3 * f})
    result = oe.factor(frame, list("abcd"), method="ml", factors=1)
    assert result.attrs["heywood"] == ["a"]
    assert "lower bound 0.005" in " ".join(result.attrs["notes"])
    # The bounded ML solution: brute-force minimization of the full discrepancy with the
    # uniquenesses constrained to [0.005, inf).
    r = np.corrcoef(frame.to_numpy().T)

    def discrepancy(theta):
        lam, psi = theta[:4, None], theta[4:]
        sigma = lam @ lam.T + np.diag(psi)
        return (np.linalg.slogdet(sigma)[1] + np.trace(np.linalg.solve(sigma, r))
                - np.linalg.slogdet(r)[1] - 4)

    fit = sopt.minimize(discrepancy, np.r_[np.full(4, 0.6), np.full(4, 0.5)], method="L-BFGS-B",
                        bounds=[(None, None)] * 4 + [(0.005, None)] * 4,
                        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
    np.testing.assert_allclose(result["communalities"]["uniqueness"], fit.x[4:], atol=1e-5)
    assert result.attrs["discrepancy"] == pytest.approx(fit.fun, abs=1e-8)


@pytest.mark.parametrize("linkage", ["centroid", "median"])
def test_tied_distances_still_merge_a_closest_pair(linkage):
    # With ties the merge order of the non-reducible linkages may differ from SciPy's
    # tie-breaking, so the oracle replays the definition: every merge joins a pair of
    # clusters whose representatives (centroids, or Gower medians) are closest.
    grid = pd.DataFrame({"a": np.repeat(np.arange(5.0), 5), "b": np.tile(np.arange(5.0), 5)})
    x = grid.to_numpy()
    n = len(x)
    tree = oe.cluster_hierarchical(grid, ["a", "b"], linkage=linkage,
                                   metric="sqeuclidean")["dendrogram"]
    rep, size = {i: x[i] for i in range(n)}, {i: 1 for i in range(n)}
    for stage, (left, right, height) in enumerate(zip(tree["left"], tree["right"],
                                                      tree["height"], strict=True)):
        points = np.array(list(rep.values()))
        gaps = ((points[:, None] - points[None]) ** 2).sum(2)
        np.fill_diagonal(gaps, np.inf)
        merged = ((rep[left] - rep[right]) ** 2).sum()
        assert merged == pytest.approx(gaps.min(), abs=1e-12)
        assert height == pytest.approx(merged, abs=1e-12)
        if linkage == "centroid":
            rep[n + stage] = (size[left] * rep[left] + size[right] * rep[right]) / (
                size[left] + size[right])
        else:
            rep[n + stage] = (rep[left] + rep[right]) / 2
        size[n + stage] = size[left] + size[right]
        del rep[left], rep[right]


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_tied_distances_reducible_linkages_merge_a_closest_pair(linkage):
    # Rounded data have many tied distances; the tree is then not unique and may differ
    # from SciPy's, but every merge (in the reported, height-sorted order) must join a
    # closest pair of the current clusters under the linkage's definition.
    rng = np.random.default_rng(12)
    x = np.round(rng.normal(size=(30, 2)) * np.array([3.0, 1.0]))
    tree = oe.cluster_hierarchical(pd.DataFrame(x, columns=["a", "b"]), ["a", "b"],
                                   linkage=linkage)["dendrogram"]
    gaps = np.sqrt(((x[:, None] - x[None]) ** 2).sum(2))
    members = {i: [i] for i in range(len(x))}

    def between(a, b):
        block = gaps[np.ix_(members[a], members[b])]
        if linkage == "single":
            return block.min()
        if linkage == "complete":
            return block.max()
        if linkage == "average":
            return block.mean()
        na, nb = len(members[a]), len(members[b])
        diff = x[members[a]].mean(0) - x[members[b]].mean(0)
        return math.sqrt(2 * na * nb / (na + nb) * (diff ** 2).sum())

    for stage, (left, right, height) in enumerate(zip(tree["left"], tree["right"],
                                                      tree["height"], strict=True)):
        keys = list(members)
        closest = min(between(a, b) for i, a in enumerate(keys) for b in keys[i + 1:])
        assert between(left, right) == pytest.approx(closest, abs=1e-10)
        assert height == pytest.approx(between(left, right), abs=1e-10)
        members[len(x) + stage] = members.pop(left) + members.pop(right)


def test_merge_order_keeps_dependencies_when_a_height_rounds_down():
    from openecon.econometrics.multivariate import hierarchical

    # Chain order: (0 into 1) at 0.3, then cluster 1 with 2 at 0.3 minus one ulp (a
    # rounded Lance-Williams average of tied distances), then (3 into 4) at 0.2.
    lower = math.nextafter(0.3, 0.0)
    merges = [(0, 1, 0.3), (2, 1, lower), (3, 4, 0.2)]
    ordered = hierarchical.height_order(merges)
    assert ordered == [(3, 4, 0.2), (0, 1, 0.3), (2, 1, lower)]
    rows, _ = hierarchical.label_merges(ordered, 5)
    assert [row[:2] for row in rows] == [[3, 4], [0, 1], [2, 6]]


def test_kmeans_iteration_limit_and_seed_are_reported():
    result = oe.cluster_kmeans(DF, NAMES, 3, max_iterations=1)
    assert result.attrs["iterations"] == 1
    if not result.attrs["converged"]:
        assert any("max_iterations" in note for note in result.attrs["notes"])
    drawn = oe.cluster_kmeans(DF, NAMES, 3, init="random")
    again = oe.cluster_kmeans(DF, NAMES, 3, init="random", seed=drawn.attrs["seed"])
    np.testing.assert_array_equal(drawn["centers"].to_numpy(), again["centers"].to_numpy())


def test_attrs_never_hold_non_finite_numbers():
    for result in (oe.factor(DF, ["a", "b", "c"], method="ml", factors=1),
                   oe.canon(_with(c=DF["a"] * 2 + 1), x=["a", "b"], y=["c"]),
                   oe.mds(DF, NAMES, method="smacof", max_iterations=2)):
        for value in result.attrs.values():
            if isinstance(value, float):
                assert math.isfinite(value)
