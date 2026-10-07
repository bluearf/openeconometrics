"""Discriminant analysis and canonical correlation against explicit NumPy / statsmodels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import linalg as sla
from scipy import stats
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.multivariate.manova import MANOVA

import openecon as oe
from openecon.analysis_contracts import AnalysisError

NAMES = ["w", "x", "y", "z"]


def _groups(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = [(40, [0, 0, 0, 0], [1, 1.5, 0.7, 1]), (55, [1.5, 0.5, 0, 0.3], [1, 1, 1, 1.2]),
              (35, [0.5, 2, 1, 0], [1.3, 1, 0.8, 1])]
    x = np.vstack([rng.normal(size=(n, 4)) * scale + mean for n, mean, scale in blocks])
    frame = pd.DataFrame(x * [1, 10, 0.1, 1] + [0, 50, 0, -3], columns=NAMES)
    frame["g"] = np.repeat(["a", "b", "c"], [40, 55, 35])
    return frame.sample(frac=1.0, random_state=1).reset_index(drop=True)


def _pooled(x: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.unique(g)
    means = np.array([x[g == label].mean(0) for label in labels])
    within = sum((x[g == label] - means[i]).T @ (x[g == label] - means[i])
                 for i, label in enumerate(labels))
    return labels, means, within


def _lda_predict(train: np.ndarray, g: np.ndarray, test: np.ndarray, priors) -> np.ndarray:
    labels, means, within = _pooled(train, g)
    inverse = np.linalg.inv(within / (len(train) - len(labels)))
    scores = np.array([[-0.5 * (row - m) @ inverse @ (row - m) + math.log(pr)
                        for m, pr in zip(means, priors, strict=True)] for row in test])
    return scores


def _qda_predict(train: np.ndarray, g: np.ndarray, test: np.ndarray, priors) -> np.ndarray:
    scores = []
    for label, prior in zip(np.unique(g), priors, strict=True):
        block = train[g == label]
        mean, cov = block.mean(0), np.cov(block.T)
        inverse = np.linalg.inv(cov)
        scores.append([-0.5 * np.linalg.slogdet(cov)[1] - 0.5 * (row - mean) @ inverse
                       @ (row - mean) + math.log(prior) for row in test])
    return np.array(scores).T


def _table(actual: np.ndarray, predicted: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.array([[((actual == a) & (predicted == b)).sum() for b in labels] for a in labels])


# ---- linear discriminant analysis -------------------------------------------------


def test_lda_canonical_functions_match_generalized_eigenproblem():
    frame = _groups()
    frame.loc[5, "x"] = np.nan
    complete = frame.dropna()
    x, g = complete[NAMES].to_numpy(), complete["g"].to_numpy()
    n, p, groups = len(x), 4, 3
    labels, means, within = _pooled(x, g)
    total = (x - x.mean(0)).T @ (x - x.mean(0))
    between = total - within
    roots, vectors = sla.eigh(between, within)
    roots, vectors = roots[::-1][:2], vectors[:, ::-1][:, :2]
    sw = within / (n - groups)
    vectors = vectors / np.sqrt(np.diag(vectors.T @ sw @ vectors))
    result = oe.discrim(frame, "g", NAMES)
    assert result.attrs["n"] == n and result.attrs["n_missing"] == 1

    canonical = result["canonical_functions"]
    np.testing.assert_allclose(canonical["eigenvalue"], roots, rtol=1e-9)
    np.testing.assert_allclose(canonical["percent"], 100 * roots / roots.sum(), rtol=1e-9)
    np.testing.assert_allclose(canonical["canonical_correlation"], np.sqrt(roots / (1 + roots)),
                               rtol=1e-9)
    for k in range(2):
        lam = np.prod(1 / (1 + roots[k:]))
        chi2 = -(n - 1 - (p + groups) / 2) * math.log(lam)
        df = (p - k) * (groups - k - 1)
        row = canonical.iloc[k]
        assert row["wilks_lambda"] == pytest.approx(lam, rel=1e-9)
        assert row["chi2"] == pytest.approx(chi2, rel=1e-9) and row["df"] == df
        assert row["p_value"] == pytest.approx(stats.chi2.sf(chi2, df), rel=1e-7)

    raw = result["unstandardized_coefficients"]
    ours = raw.loc[NAMES].to_numpy()
    signs = np.sign((ours * vectors).sum(0))
    np.testing.assert_allclose(ours, vectors * signs, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(raw.loc["Intercept"], -x.mean(0) @ ours, rtol=1e-8)
    standardized = result["standardized_coefficients"].to_numpy()
    np.testing.assert_allclose(standardized, ours * np.sqrt(np.diag(sw))[:, None], rtol=1e-9)
    assert (standardized[np.abs(standardized).argmax(0), np.arange(2)] > 0).all()
    scores = x @ ours + raw.loc["Intercept"].to_numpy()
    np.testing.assert_allclose(scores.mean(0), 0, atol=1e-9)
    centroids = np.array([scores[g == label].mean(0) for label in labels])
    np.testing.assert_allclose(result["centroids"].to_numpy(), centroids, atol=1e-9)
    # Structure matrix: pooled within-groups correlations of variables and functions.
    dx = x - means[np.searchsorted(labels, g)]
    ds = scores - centroids[np.searchsorted(labels, g)]
    structure = (dx.T @ ds) / np.sqrt(np.outer((dx ** 2).sum(0), (ds ** 2).sum(0)))
    np.testing.assert_allclose(result["structure_matrix"].to_numpy(), structure, atol=1e-9)
    # The pooled within-group variance of every function is one.
    np.testing.assert_allclose((ds ** 2).sum(0) / (n - groups), 1.0, rtol=1e-9)


def test_lda_descriptives_equality_tests_and_box_m():
    frame = _groups(seed=2)
    x, g = frame[NAMES].to_numpy(), frame["g"].to_numpy()
    n, p = x.shape
    labels, means, within = _pooled(x, g)
    result = oe.discrim(frame, "g", NAMES)
    np.testing.assert_allclose(result["group_means"].to_numpy(), means, rtol=1e-10)
    statistics = result["group_statistics"].set_index(["group", "variable"])
    assert statistics.loc[("b", "x"), "std_dev"] == pytest.approx(x[g == "b", 1].std(ddof=1))
    assert statistics.loc[("Total", "z"), "mean"] == pytest.approx(x[:, 3].mean())
    assert statistics.loc[("a", "w"), "n"] == 40
    for j, name in enumerate(NAMES):
        f, p_value = stats.f_oneway(*[x[g == label, j] for label in labels])
        row = result["tests_of_equality"].loc[name]
        assert row["statistic"] == pytest.approx(f, rel=1e-9)
        assert row["p_value"] == pytest.approx(p_value, rel=1e-7)
        assert row["wilks_lambda"] == pytest.approx(1 / (1 + f * 2 / (n - 3)), rel=1e-9)
    # Box's M by hand.
    counts = np.array([(g == label).sum() for label in labels])
    log_dets = [np.linalg.slogdet(np.cov(x[g == label].T))[1] for label in labels]
    log_pooled = np.linalg.slogdet(within / (n - 3))[1]
    m = (n - 3) * log_pooled - ((counts - 1) * log_dets).sum()
    c1 = ((1 / (counts - 1)).sum() - 1 / (n - 3)) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * 2)
    c2 = ((1 / (counts - 1) ** 2).sum() - 1 / (n - 3) ** 2) * (p - 1) * (p + 2) / (6 * 2)
    df1 = p * (p + 1) * 2 / 2
    df2 = (df1 + 2) / (c2 - c1 ** 2)
    f = m * (1 - c1 - df1 / df2) / df1
    box = result["box_m"].iloc[0]
    assert box["statistic"] == pytest.approx(m, rel=1e-9)
    assert box["f"] == pytest.approx(f, rel=1e-9) and box["df1"] == df1
    assert box["df2"] == pytest.approx(df2, rel=1e-9)
    assert box["p_value"] == pytest.approx(stats.f.sf(f, df1, df2), rel=1e-7)
    assert box["chi2"] == pytest.approx(m * (1 - c1), rel=1e-9)
    np.testing.assert_allclose(result["log_determinants"]["log_determinant"],
                               [*log_dets, log_pooled], rtol=1e-9)


@pytest.mark.parametrize("priors", ["equal", "proportional", [0.5, 0.3, 0.2]])
def test_lda_classification_resubstitution_and_leave_one_out(priors):
    frame = _groups(seed=3)
    x, g = frame[NAMES].to_numpy(), frame["g"].to_numpy()
    labels = np.unique(g)
    n = len(x)
    counts = np.array([(g == label).sum() for label in labels])
    prior = {"equal": np.full(3, 1 / 3), "proportional": counts / n}.get(
        priors if isinstance(priors, str) else "", np.array(priors if not isinstance(
            priors, str) else [0.0] * 3))
    result = oe.discrim(frame, "g", NAMES, priors=priors, loo=True)
    np.testing.assert_allclose(result["groups"]["prior"], prior, rtol=1e-12)
    scores = _lda_predict(x, g, x, prior)
    expected = _table(g, labels[scores.argmax(1)], labels)
    table = result["classification_table"]
    assert (table[list(labels)].to_numpy() == expected).all()
    np.testing.assert_allclose(table["percent_correct"], 100 * np.diag(expected) / counts)
    assert result.attrs["percent_correct"] == pytest.approx(100 * np.trace(expected) / n)
    loo = np.array([labels[_lda_predict(np.delete(x, i, 0), np.delete(g, i), x[i:i + 1],
                                        prior).argmax(1)[0]] for i in range(n)])
    expected_loo = _table(g, loo, labels)
    assert (result["classification_table_loo"][list(labels)].to_numpy() == expected_loo).all()
    assert result.attrs["percent_correct_loo"] == pytest.approx(
        100 * np.trace(expected_loo) / n)

    # Fisher's classification functions and the posterior probabilities.
    functions = result["classification_functions"]
    _, means, within = _pooled(x, g)
    inverse = np.linalg.inv(within / (n - 3))
    np.testing.assert_allclose(functions.loc[NAMES].to_numpy(), inverse @ means.T, rtol=1e-7)
    np.testing.assert_allclose(
        functions.loc["Intercept"],
        -0.5 * np.einsum("gi,ij,gj->g", means, inverse, means) + np.log(prior), rtol=1e-7)
    predicted = oe.discrim_predict(result, frame)
    assert (predicted["predicted"].to_numpy() == labels[scores.argmax(1)]).all()
    posterior = np.exp(scores - scores.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    np.testing.assert_allclose(predicted[[f"posterior_{label}" for label in labels]].to_numpy(),
                               posterior, atol=1e-9)


# ---- quadratic discriminant analysis ----------------------------------------------


def test_qda_classification_and_prediction():
    frame = _groups(seed=4)
    x, g = frame[NAMES].to_numpy(), frame["g"].to_numpy()
    labels = np.unique(g)
    n = len(x)
    prior = np.full(3, 1 / 3)
    result = oe.discrim(frame, "g", NAMES, method="qda", loo=True)
    assert "canonical_functions" not in result and "classification_functions" not in result
    scores = _qda_predict(x, g, x, prior)
    assert (result["classification_table"][list(labels)].to_numpy()
            == _table(g, labels[scores.argmax(1)], labels)).all()
    loo = np.array([labels[_qda_predict(np.delete(x, i, 0), np.delete(g, i), x[i:i + 1],
                                        prior).argmax(1)[0]] for i in range(n)])
    assert (result["classification_table_loo"][list(labels)].to_numpy()
            == _table(g, loo, labels)).all()
    covariances = result["group_covariances"]
    block = covariances[covariances["group"] == "b"][NAMES].to_numpy()
    np.testing.assert_allclose(block, np.cov(x[g == "b"].T), rtol=1e-9)

    new = frame.iloc[:25].copy()
    new.loc[new.index[2], "w"] = np.nan
    predicted = oe.discrim_predict(result, new)
    assert pd.isna(predicted.iloc[2]["predicted"]) and predicted.iloc[2].isna().all()
    rows = [i for i in range(25) if i != 2]
    posterior = np.exp(scores[rows] - scores[rows].max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    np.testing.assert_allclose(
        predicted.iloc[rows][[f"posterior_{label}" for label in labels]].to_numpy(dtype=float),
        posterior, atol=1e-9)
    assert (predicted.iloc[rows]["predicted"].to_numpy() == labels[scores[rows].argmax(1)]).all()


def test_discrim_two_groups_numeric_labels_and_errors():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"g": np.repeat([2, 1], 30), "a": rng.normal(size=60),
                          "b": rng.normal(size=60)})
    frame.loc[frame["g"] == 2, "a"] += 2.0
    result = oe.discrim(frame, "g", ["a", "b"])
    assert list(result["groups"].index) == [1, 2] and len(result["canonical_functions"]) == 1
    assert list(result["classification_table"].columns) == ["1", "2", "n", "percent_correct"]
    assert set(oe.discrim_predict(result, frame)["predicted"]) == {1, 2}
    text = str(result)
    assert "[canonical_functions]" in text and "[classification_table]" in text
    assert result.to_latex().count(r"\begin{tabular}") == len(result)

    def code(*args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            oe.discrim(*args, **kwargs)
        return error.value.code

    assert code(frame, "g", ["a", "b"], method="knn") == "invalid_option"
    assert code(frame, "g", ["a", "b"], priors=[0.5, 0.6]) == "invalid_option"
    assert code(frame, "g", ["a", "b"], priors="size") == "invalid_option"
    assert code(frame, ["g"], ["a", "b"]) == "invalid_spec"
    assert code(frame, "g", ["a", "g"]) == "invalid_spec"
    assert code(frame.assign(g=1), "g", ["a", "b"]) == "invalid_groups"
    assert code(frame.assign(c=frame["a"] * 2), "g", ["a", "b", "c"]) == "singular_matrix"
    assert code(frame.assign(s="t"), "g", ["a", "s"]) == "non_numeric_column"
    tiny = frame.iloc[[0, 1, 2, 30, 31, 32]]
    assert code(tiny, "g", ["a", "b"], method="qda", loo=True) == "insufficient_observations"
    assert code(frame.iloc[[0, 1, 30, 31]], "g", ["a", "b"], method="qda") == "singular_matrix"
    with pytest.raises(AnalysisError) as error:
        oe.discrim_predict(result, frame[["a"]])
    assert error.value.code == "missing_columns"


# ---- canonical correlation --------------------------------------------------------


def _two_sets(seed: int = 0, n: int = 200) -> tuple[pd.DataFrame, list, list]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    data = {"x1": z[:, 0] + rng.normal(size=n), "x2": 5 * (z[:, 1] + rng.normal(size=n)),
            "x3": rng.normal(size=n) + 10, "y1": z[:, 0] + 0.5 * rng.normal(size=n),
            "y2": 0.1 * (z[:, 0] - z[:, 1] + rng.normal(size=n))}
    return pd.DataFrame(data), ["x1", "x2", "x3"], ["y1", "y2"]


def test_canon_matches_covariance_eigenproblem_and_statsmodels():
    frame, xs, ys = _two_sets()
    frame.loc[9, "y2"] = np.nan
    complete = frame.dropna()
    x, y = complete[xs].to_numpy(), complete[ys].to_numpy()
    n, p = len(x), 3
    result = oe.canon(frame, x=xs, y=ys)
    assert result.attrs["n"] == n and result.attrs["n_missing"] == 1
    sxx, syy = np.cov(x.T), np.cov(y.T)
    sxy = np.cov(np.column_stack([x, y]).T)[:p, p:]
    rho2 = np.sort(np.linalg.eigvals(np.linalg.solve(sxx, sxy) @ np.linalg.solve(syy, sxy.T))
                   .real)[::-1][:2]
    correlations = result["correlations"]
    np.testing.assert_allclose(correlations["correlation"], np.sqrt(rho2), rtol=1e-9)
    np.testing.assert_allclose(correlations["correlation"], CanCorr(y, x).cancorr, rtol=1e-9)
    np.testing.assert_allclose(correlations["eigenvalue"], rho2 / (1 - rho2), rtol=1e-8)

    raw = result["raw_coefficients"]
    a, b = raw.loc[xs].drop(columns="set").to_numpy(), raw.loc[ys].drop(columns="set").to_numpy()
    u, v = (x - x.mean(0)) @ a, (y - y.mean(0)) @ b
    np.testing.assert_allclose(np.cov(u.T), np.eye(2), atol=1e-9)        # unit variance
    np.testing.assert_allclose(np.cov(v.T), np.eye(2), atol=1e-9)
    cross = np.corrcoef(np.column_stack([u, v]).T)[:2, 2:]
    np.testing.assert_allclose(cross, np.diag(np.sqrt(rho2)), atol=1e-9)
    standardized = result["standardized_coefficients"]
    np.testing.assert_allclose(standardized.loc[xs].drop(columns="set").to_numpy(),
                               a * x.std(0, ddof=1)[:, None], rtol=1e-9)
    ours = standardized.loc[xs].drop(columns="set").to_numpy()
    assert (ours[np.abs(ours).argmax(0), np.arange(2)] > 0).all()
    loadings = result["loadings"]
    for j, name in enumerate(xs):
        for k in range(2):
            assert loadings.loc[name, f"Canon{k + 1}"] == pytest.approx(
                np.corrcoef(x[:, j], u[:, k])[0, 1], abs=1e-9)
            assert loadings.loc[name, f"Cross{k + 1}"] == pytest.approx(
                np.corrcoef(x[:, j], v[:, k])[0, 1], abs=1e-9)
    for j, name in enumerate(ys):
        assert loadings.loc[name, "Canon2"] == pytest.approx(
            np.corrcoef(y[:, j], v[:, 1])[0, 1], abs=1e-9)
        assert loadings.loc[name, "Cross1"] == pytest.approx(
            np.corrcoef(y[:, j], u[:, 0])[0, 1], abs=1e-9)
    assert list(loadings["set"]) == ["x", "x", "x", "y", "y"]
    redundancy = result["redundancy"]
    x_load = loadings.loc[xs, ["Canon1", "Canon2"]].to_numpy(dtype=float)
    np.testing.assert_allclose(redundancy["x_variance"], (x_load ** 2).mean(0), rtol=1e-10)
    np.testing.assert_allclose(redundancy["x_redundancy"], (x_load ** 2).mean(0) * rho2,
                               rtol=1e-9)
    # With as many variates as variables, the y set is fully reproduced.
    assert redundancy["y_variance"].sum() == pytest.approx(1.0)


def test_canon_tests_match_statsmodels_manova_and_hand_formulas():
    frame, xs, ys = _two_sets(seed=2, n=120)
    x, y = frame[xs].to_numpy(), frame[ys].to_numpy()
    n, p, q = len(x), 3, 2
    result = oe.canon(frame, x=xs, y=ys)
    exog = np.column_stack([np.ones(n), x])
    hypothesis = np.hstack([np.zeros((3, 1)), np.eye(3)])
    reference = MANOVA(y, exog).mv_test([("slopes", hypothesis)]).results["slopes"]["stat"]
    tests = result["tests"]
    for ours, theirs in (("wilks", "Wilks' lambda"), ("pillai", "Pillai's trace"),
                         ("roy", "Roy's greatest root")):
        assert tests.loc[ours, "value"] == pytest.approx(reference.loc[theirs, "Value"], rel=1e-8)
        assert tests.loc[ours, "statistic"] == pytest.approx(reference.loc[theirs, "F Value"],
                                                             rel=1e-8)
        assert tests.loc[ours, "df1"] == pytest.approx(reference.loc[theirs, "Num DF"])
        assert tests.loc[ours, "df2"] == pytest.approx(reference.loc[theirs, "Den DF"])
        assert tests.loc[ours, "p_value"] == pytest.approx(reference.loc[theirs, "Pr > F"],
                                                           rel=1e-6)
    rho2 = result["correlations"]["squared_correlation"].to_numpy()
    roots = rho2 / (1 - rho2)
    assert tests.loc["hotelling", "value"] == pytest.approx(
        reference.loc["Hotelling-Lawley trace", "Value"], rel=1e-8)
    s, m, h = 2, (abs(p - q) - 1) / 2, (n - 1 - q - p - 1) / 2
    f = 2 * (s * h + 1) * roots.sum() / (s * s * (2 * m + s + 1))
    assert tests.loc["hotelling", "statistic"] == pytest.approx(f, rel=1e-9)
    assert tests.loc["hotelling", "p_value"] == pytest.approx(
        stats.f.sf(f, s * (2 * m + s + 1), 2 * (s * h + 1)), rel=1e-7)
    assert list(tests["f_type"]) == ["exact", "approximate", "approximate", "upper_bound"]

    correlations = result["correlations"]
    first = correlations.iloc[0]                 # the first sequential test is Wilks' test
    assert first["f"] == pytest.approx(tests.loc["wilks", "statistic"], rel=1e-12)
    assert first["df2"] == pytest.approx(tests.loc["wilks", "df2"])
    w = n - 1 - (p + q + 1) / 2
    for k in range(2):
        lam = np.prod(1 - rho2[k:])
        pk, qk = p - k, q - k
        row = correlations.iloc[k]
        assert row["wilks_lambda"] == pytest.approx(lam, rel=1e-10)
        assert row["chi2"] == pytest.approx(-w * math.log(lam), rel=1e-10)
        assert row["chi2_df"] == pk * qk
        assert row["chi2_p_value"] == pytest.approx(stats.chi2.sf(-w * math.log(lam), pk * qk),
                                                    rel=1e-7)
        t = math.sqrt((pk * pk * qk * qk - 4) / (pk * pk + qk * qk - 5)) \
            if pk * pk + qk * qk - 5 > 0 else 1.0
        df2 = w * t - (pk * qk - 2) / 2
        f = (1 - lam ** (1 / t)) / lam ** (1 / t) * df2 / (pk * qk)
        assert row["f"] == pytest.approx(f, rel=1e-9) and row["df2"] == pytest.approx(df2)
        assert row["p_value"] == pytest.approx(stats.f.sf(f, pk * qk, df2), rel=1e-7)
    # Swapping the sets leaves the correlations and tests unchanged.
    swapped = oe.canon(frame, x=ys, y=xs)
    np.testing.assert_allclose(swapped["correlations"]["correlation"],
                               correlations["correlation"], rtol=1e-10)
    np.testing.assert_allclose(swapped["tests"]["statistic"], tests["statistic"], rtol=1e-9)


def test_canon_single_variable_is_multiple_correlation_and_errors():
    frame, xs, ys = _two_sets(seed=3, n=60)
    result = oe.canon(frame, x=xs, y=["y1"])
    design = np.column_stack([np.ones(60), frame[xs].to_numpy()])
    fitted = design @ np.linalg.lstsq(design, frame["y1"].to_numpy(), rcond=None)[0]
    r2 = np.corrcoef(fitted, frame["y1"])[0, 1] ** 2
    assert result["correlations"]["squared_correlation"].iloc[0] == pytest.approx(r2, rel=1e-9)
    f = r2 / (1 - r2) * (60 - 4) / 3                            # the regression F test
    assert result["tests"].loc["wilks", "statistic"] == pytest.approx(f, rel=1e-9)
    assert set(result["tests"]["f_type"]) == {"exact"}
    assert "Canonical correlations" in str(result)

    # An exact linear relation between the sets: rho = 1 and no test statistic.
    exact = oe.canon(frame.assign(y3=2 * frame["x1"] - frame["x2"]), x=xs, y=["y1", "y3"])
    assert exact["correlations"]["correlation"].iloc[0] == 1.0
    assert np.isnan(exact["correlations"]["f"].iloc[0])
    assert np.isnan(exact["tests"].loc["wilks", "statistic"])
    assert np.isfinite(exact["correlations"]["f"].iloc[1])
    assert "equals 1" in exact.attrs["notes"][0]

    def code(*args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            oe.canon(*args, **kwargs)
        return error.value.code

    assert code(frame, x=xs, y=["x1"]) == "invalid_spec"
    assert code(frame, x="x1", y=ys) == "invalid_spec"
    assert code(frame, x=xs, y=[]) == "invalid_spec"
    assert code(frame.assign(x4=frame["x1"] - frame["x2"]), x=[*xs, "x4"], y=ys) \
        == "singular_matrix"
    assert code(frame.iloc[:6], x=xs, y=ys) == "insufficient_observations"
    assert code(frame.assign(y1=1.0), x=xs, y=ys) == "constant_column"


def test_discrim_small_group_omits_box_m_with_a_note():
    frame = _groups(seed=6)
    keep = frame[frame["g"] != "c"].index.tolist() + frame[frame["g"] == "c"].index.tolist()[:1]
    small = frame.loc[keep]
    result = oe.discrim(small, "g", NAMES)
    assert "box_m" not in result and "Box's M" in result.attrs["notes"][0]
    statistics = result["group_statistics"].set_index(["group", "variable"])
    assert np.isnan(statistics.loc[("c", "w"), "std_dev"]) and statistics.loc[("c", "w"), "n"] == 1
    with pytest.raises(AnalysisError) as error:
        oe.discrim(small, "g", NAMES, loo=True)
    assert error.value.code == "insufficient_observations"
