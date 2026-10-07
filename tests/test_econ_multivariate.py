"""Principal components, factor analysis and reliability against NumPy / statsmodels oracles."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import optimize as sopt
from scipy import stats
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors
from statsmodels.multivariate.pca import PCA

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import extraction, rotation
from openecon.engines.optimize import check_derivatives
from openecon.frame import DataFrame


def _factor_data(n: int = 400, seed: int = 0, noise: float = 0.6) -> tuple[pd.DataFrame, list]:
    rng = np.random.default_rng(seed)
    loadings = np.array([[.8, .1, .0], [.7, .2, .1], [.75, .0, .2], [.1, .8, .0], [.0, .7, .2],
                         [.2, .6, .1], [.1, .1, .8], [.0, .2, .7], [.2, .0, .6]])
    x = rng.normal(size=(n, 3)) @ loadings.T + noise * rng.normal(size=(n, 9))
    names = [f"x{j}" for j in range(1, 10)]
    return pd.DataFrame(x * rng.uniform(0.5, 3.0, 9) + rng.normal(size=9), columns=names), names


def _corr(frame: pd.DataFrame, names: list) -> np.ndarray:
    return np.corrcoef(frame[names].dropna().to_numpy().T)


def _eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    return values[::-1], vectors[:, ::-1]


def _match(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Largest column difference after matching columns up to order and sign."""
    worst = 0.0
    for j in range(ours.shape[1]):
        worst = max(worst, min(min(np.abs(ours[:, j] - theirs[:, k]).max(),
                                   np.abs(ours[:, j] + theirs[:, k]).max())
                               for k in range(theirs.shape[1])))
    return worst


# ---- public surface ---------------------------------------------------------------


def test_functions_are_exported_and_documented():
    from openecon.econometrics import registry

    exports = registry.public_exports()
    for name in ("pca", "pca_scores", "factor", "factortest", "factor_scores", "alpha",
                 "cluster_kmeans", "cluster_assign", "cluster_hierarchical", "cluster_cut",
                 "discrim", "discrim_predict", "canon", "mds", "ca"):
        assert exports[name][0].startswith("openecon.econometrics.multivariate.")
        function = getattr(oe, name)
        assert callable(function) and "Example" in function.__doc__


def test_docstring_examples_run():
    import doctest
    from importlib import import_module

    for name in ("pca", "factor", "reliability", "kmeans", "hierarchical", "discrim", "canon",
                 "scaling"):
        module = import_module(f"openecon.econometrics.multivariate.{name}")
        outcome = doctest.testmod(module)
        assert outcome.attempted > 0 and outcome.failed == 0, name


def test_results_render_and_serialize():
    frame, names = _factor_data()
    result = oe.factor(frame, names, method="ml", factors=3, rotate="promax")
    assert isinstance(result, TableSet)
    assert all(isinstance(table, DataFrame) for table in result.values())
    text = str(result)
    assert "Factor analysis (maximum likelihood)" in text and "[rotated_loadings]" in text
    latex = result.to_latex()
    assert latex.count(r"\begin{tabular}") == len(result)
    json.dumps(result.attrs)                                   # attrs are JSON-safe
    for table in result.values():
        json.loads(table.to_json())


# ---- principal components ---------------------------------------------------------


def test_pca_correlation_matches_numpy_and_statsmodels():
    frame, names = _factor_data()
    frame.loc[5, "x2"] = np.nan
    result = oe.pca(frame, names)
    r = _corr(frame, names)
    values, vectors = _eig(r)
    table = result["eigenvalues"]
    np.testing.assert_allclose(table["eigenvalue"], values, rtol=1e-12)
    np.testing.assert_allclose(table["proportion"], values / 9, rtol=1e-12)
    np.testing.assert_allclose(table["cumulative"], np.cumsum(values) / 9, rtol=1e-12)
    np.testing.assert_allclose(table["difference"][:-1], -np.diff(values), rtol=1e-9)
    assert math.isnan(table["difference"].iloc[-1])
    ours = result["eigenvectors"].drop(columns="unexplained").to_numpy()
    assert _match(ours, vectors) < 1e-10
    # Sign rule: the entry of largest absolute value of every eigenvector is positive.
    assert (ours[np.abs(ours).argmax(0), np.arange(9)] > 0).all()
    np.testing.assert_allclose(result["loadings"].to_numpy(), ours * np.sqrt(values), rtol=1e-10)
    np.testing.assert_allclose(result["eigenvectors"]["unexplained"], 0, atol=1e-10)
    assert result.attrs["n"] == 399 and result.attrs["n_missing"] == 1
    assert result.attrs["trace"] == pytest.approx(9.0) and result.attrs["rho"] == 1.0
    # statsmodels' PCA of the standardized data has eigenvalues n * (ours).
    complete = frame[names].dropna().to_numpy()
    reference = PCA(complete, standardize=True, method="eig")
    np.testing.assert_allclose(reference.eigenvals / 399, values, rtol=1e-9)
    desc = result["descriptives"]
    np.testing.assert_allclose(desc["mean"], complete.mean(0), rtol=1e-12)
    np.testing.assert_allclose(desc["std_dev"], complete.std(0, ddof=1), rtol=1e-12)


def test_pca_retention_rules_and_covariance_matrix():
    frame, names = _factor_data()
    values, vectors = _eig(_corr(frame, names))
    kaiser = oe.pca(frame, names, mineigen=1.0)
    keep = int((values > 1).sum())
    assert kaiser.attrs["components"] == keep
    assert list(kaiser["loadings"].columns) == [f"Comp{i}" for i in range(1, keep + 1)]
    assert kaiser.attrs["rho"] == pytest.approx(values[:keep].sum() / 9)
    loadings = vectors[:, :keep] * np.sqrt(values[:keep])
    np.testing.assert_allclose(kaiser["communalities"]["extraction"], (loadings ** 2).sum(1),
                               rtol=1e-10)
    np.testing.assert_allclose(kaiser["eigenvectors"]["unexplained"],
                               1 - (loadings ** 2).sum(1), rtol=1e-9)
    assert oe.pca(frame, names, components=2).attrs["components"] == 2
    assert oe.pca(frame, names, components=4, mineigen=1.0).attrs["components"] == keep

    s = np.cov(frame[names].to_numpy().T)
    cov_values, cov_vectors = _eig(s)
    result = oe.pca(frame, names, matrix="covariance", components=3)
    np.testing.assert_allclose(result["eigenvalues"]["eigenvalue"], cov_values, rtol=1e-10)
    ours = result["eigenvectors"].drop(columns="unexplained").to_numpy()
    assert _match(ours, cov_vectors[:, :3]) < 1e-9
    assert result.attrs["trace"] == pytest.approx(np.trace(s))
    np.testing.assert_allclose(result["communalities"]["initial"], np.diag(s), rtol=1e-10)
    np.testing.assert_allclose(
        result["eigenvectors"]["unexplained"],
        np.diag(s) - ((cov_vectors[:, :3] ** 2) * cov_values[:3]).sum(1), rtol=1e-8)


def test_pca_scores():
    frame, names = _factor_data(n=150, seed=3)
    frame.loc[7, "x4"] = np.nan
    result = oe.pca(frame, names, components=3)
    scores = oe.pca_scores(result, frame)
    assert list(scores.index) == list(frame.index) and scores.loc[7].isna().all()
    complete = frame[names].dropna()
    z = (complete - complete.mean()) / complete.std(ddof=1)
    expected = z.to_numpy() @ result["eigenvectors"].drop(columns="unexplained").to_numpy()
    np.testing.assert_allclose(scores.dropna().to_numpy(), expected, atol=1e-10)
    np.testing.assert_allclose(scores.var(ddof=1), result["eigenvalues"]["eigenvalue"][:3],
                               rtol=1e-10)
    unit = oe.pca_scores(result, frame, normalize=True)
    np.testing.assert_allclose(unit.var(ddof=1), 1.0, rtol=1e-10)

    covariance = oe.pca(frame, names, matrix="covariance", components=2)
    vectors = covariance["eigenvectors"].drop(columns="unexplained").to_numpy()
    centred = oe.pca_scores(covariance, frame).dropna().to_numpy()
    np.testing.assert_allclose(centred, (complete - complete.mean()).to_numpy() @ vectors,
                               atol=1e-10)
    raw = oe.pca_scores(covariance, frame, center=False).dropna().to_numpy()
    np.testing.assert_allclose(raw, complete.to_numpy() @ vectors, atol=1e-10)
    # New data are scored with the estimation-sample moments.
    new = pd.DataFrame({name: [1.0, 2.0] for name in names})
    assert oe.pca_scores(result, new).shape == (2, 3)


# ---- factor extraction ------------------------------------------------------------


def test_principal_factors_and_pcf_match_numpy():
    frame, names = _factor_data()
    r = _corr(frame, names)
    smc = 1 - 1 / np.diag(np.linalg.inv(r))
    reduced = r.copy()
    np.fill_diagonal(reduced, smc)
    values, vectors = _eig(reduced)
    result = oe.factor(frame, names, method="pf")
    m = int((values > 5e-6).sum())
    assert result.attrs["factors"] == m
    table = result["eigenvalues"]
    np.testing.assert_allclose(table["eigenvalue"], values, atol=1e-10)
    np.testing.assert_allclose(table["proportion"], values / smc.sum(), atol=1e-10)
    np.testing.assert_allclose(table["initial_eigenvalue"], _eig(r)[0], rtol=1e-10)
    expected = vectors[:, :m] * np.sqrt(values[:m])
    assert _match(result["loadings"].to_numpy(), expected) < 1e-9
    comm = result["communalities"]
    np.testing.assert_allclose(comm["initial"], smc, rtol=1e-10)
    np.testing.assert_allclose(comm["extraction"], (expected ** 2).sum(1), rtol=1e-9)
    np.testing.assert_allclose(comm["uniqueness"], 1 - (expected ** 2).sum(1), rtol=1e-9)
    variance = result["variance"]
    np.testing.assert_allclose(variance["variance"], (expected ** 2).sum(0), rtol=1e-9)
    np.testing.assert_allclose(variance["proportion"], (expected ** 2).sum(0) / 9, rtol=1e-9)

    two = oe.factor(frame, names, method="pf", factors=2)
    assert two.attrs["factors"] == 2 and two["loadings"].shape == (9, 2)

    pcf = oe.factor(frame, names, method="pcf")
    pc_values, pc_vectors = _eig(r)
    keep = int((pc_values > 1).sum())
    assert pcf.attrs["factors"] == keep
    assert _match(pcf["loadings"].to_numpy(),
                  pc_vectors[:, :keep] * np.sqrt(pc_values[:keep])) < 1e-9
    np.testing.assert_allclose(pcf["communalities"]["initial"], 1.0)


def test_iterated_principal_factors_match_numpy_and_statsmodels():
    frame, names = _factor_data()
    r = _corr(frame, names)
    h = 1 - 1 / np.diag(np.linalg.inv(r))
    iterations = 0
    while True:                                   # the SPSS PAF rule, written independently
        reduced = r.copy()
        np.fill_diagonal(reduced, h)
        values, vectors = _eig(reduced)
        loadings = vectors[:, :3] * np.sqrt(values[:3])
        new = (loadings ** 2).sum(1)
        iterations += 1
        if np.abs(new - h).max() < 1e-3:
            h = new
            break
        h = new
    reduced = r.copy()
    np.fill_diagonal(reduced, h)
    values, vectors = _eig(reduced)
    result = oe.factor(frame, names, method="ipf", factors=3)
    assert result.attrs["iterations"] == iterations
    assert _match(result["loadings"].to_numpy(), vectors[:, :3] * np.sqrt(values[:3])) < 1e-9
    np.testing.assert_allclose(result["eigenvalues"]["eigenvalue"], values, atol=1e-9)

    tight = oe.factor(frame, names, method="ipf", factors=3, tolerance=1e-11,
                      max_iterations=5000)
    reference = Factor(corr=r, n_factor=3, method="pa").fit(maxiter=5000, tol=1e-12)
    assert _match(tight["loadings"].to_numpy(), reference.loadings) < 1e-6
    with pytest.raises(AnalysisError) as error:
        oe.factor(frame, names, method="ipf", factors=3, max_iterations=2)
    assert error.value.code == "no_convergence" and "max_iterations" in str(error.value)


def _ml_brute_force(r: np.ndarray, m: int, start: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimize the full ML discrepancy over (L, psi) with SciPy, independently of Joreskog."""
    p = r.shape[0]
    log_det_r = np.linalg.slogdet(r)[1]

    def discrepancy(theta: np.ndarray) -> float:
        loadings = theta[:p * m].reshape(p, m)
        sigma = loadings @ loadings.T + np.diag(np.exp(theta[p * m:]))
        return np.linalg.slogdet(sigma)[1] + np.trace(r @ np.linalg.inv(sigma)) - log_det_r - p

    best = sopt.minimize(discrepancy, start, method="BFGS", options={"gtol": 1e-9})
    return best.fun, np.exp(best.x[p * m:])


def test_maximum_likelihood_matches_statsmodels_and_brute_force():
    frame, names = _factor_data()
    r = _corr(frame, names)
    n, p, m = len(frame), 9, 3
    result = oe.factor(frame, names, method="ml", factors=m)
    reference = Factor(corr=r, n_factor=m, method="ml", nobs=n).fit()
    ours = result["loadings"].to_numpy()
    assert _match(ours, np.real(reference.loadings)) < 2e-5
    psi = result["communalities"]["uniqueness"].to_numpy()
    np.testing.assert_allclose(psi, reference.uniqueness, atol=2e-5)
    start = np.concatenate([ours.ravel() + 0.01, np.log(psi * 1.1)])
    value, brute_psi = _ml_brute_force(r, m, start)
    assert result.attrs["discrepancy"] == pytest.approx(value, abs=1e-8)
    np.testing.assert_allclose(psi, brute_psi, atol=1e-4)
    # Bartlett-corrected likelihood-ratio tests.
    fit = result["fit"]
    df = ((p - m) ** 2 - (p + m)) / 2
    chi2 = (n - 1 - (2 * p + 5) / 6 - 2 * m / 3) * result.attrs["discrepancy"]
    assert fit.loc["model", "statistic"] == pytest.approx(chi2, rel=1e-10)
    assert fit.loc["model", "df"] == df
    assert fit.loc["model", "p_value"] == pytest.approx(stats.chi2.sf(chi2, df), rel=1e-8)
    independence = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(r)[1]
    assert fit.loc["independence", "statistic"] == pytest.approx(independence, rel=1e-10)
    assert fit.loc["independence", "df"] == p * (p - 1) / 2
    # The loadings are in canonical form: L' Psi^{-1} L is diagonal.
    inner = ours.T @ (ours / psi[:, None])
    assert np.abs(inner - np.diag(np.diag(inner))).max() < 1e-5
    # Too many factors are capped at the largest identified model, with a note.
    capped = oe.factor(frame, names, method="ml", factors=8)
    assert capped.attrs["factors"] <= 5 and "retained instead" in capped.attrs["notes"][0]


def test_ml_discrepancy_derivatives_are_analytic():
    frame, names = _factor_data(seed=4)
    r = torch.tensor(_corr(frame, names))
    theta = torch.log(torch.linspace(0.2, 0.7, 9, dtype=torch.float64))
    report = check_derivatives(lambda t: extraction.ml_discrepancy(t, r, 2), theta)
    assert report["gradient_max_rel_error"] < 1e-8


def test_rotation_criterion_gradients_are_analytic():
    rng = np.random.default_rng(3)
    start = torch.tensor(rng.normal(size=12))

    def oblimin(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value, gradient = rotation._oblimin_value(theta.reshape(4, 3), -0.4)
        return torch.tensor(value, dtype=torch.float64), gradient.reshape(-1)

    assert check_derivatives(oblimin, start)["gradient_max_rel_error"] < 1e-8

    def orthomax(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = theta.reshape(4, 3)
        value = (z.pow(4).sum() - 1.5 / 4 * z.square().sum(0).square().sum()) / 4
        return value, rotation._asymmetry(z, 1.5)[1].reshape(-1)

    assert check_derivatives(orthomax, start)["gradient_max_rel_error"] < 1e-8


def test_ml_heywood_case_is_reported():
    rng = np.random.default_rng(1)
    n = 60
    f = rng.normal(size=n)
    x = np.column_stack([f + 0.01 * rng.normal(size=n)] + [
        0.6 * f + 0.8 * rng.normal(size=n) for _ in range(4)])
    frame = pd.DataFrame(x, columns=list("abcde"))
    result = oe.factor(frame, list("abcde"), method="ml", factors=2)
    assert result.attrs["heywood"] and "Heywood" in " ".join(result.attrs["notes"])
    bound = result["communalities"].loc[result.attrs["heywood"], "uniqueness"]
    np.testing.assert_allclose(bound, extraction.PSI_FLOOR)
    with pytest.raises(AnalysisError) as error:
        oe.factor(frame.assign(f=frame["a"] + frame["b"]), list("abcdef"), method="pf")
    assert error.value.code == "singular_matrix"


# ---- rotations --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["varimax", "quartimax"])
def test_orthogonal_rotations_match_statsmodels(method):
    frame, names = _factor_data()
    base = oe.factor(frame, names, method="pf", factors=3)
    a = base["loadings"].to_numpy()
    raw = oe.factor(frame, names, method="pf", factors=3, rotate=method, kaiser=False)
    expected, _ = rotate_factors(a, method, tol=1e-13, max_tries=5000)
    assert _match(raw["rotated_loadings"].to_numpy(), expected) < 1e-7
    matrix = raw["rotation_matrix"].to_numpy()
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(a @ matrix, raw["rotated_loadings"].to_numpy(), atol=1e-10)
    # Kaiser normalization: rotate the rows scaled to unit length, then scale back.
    scale = np.sqrt((a ** 2).sum(1))
    normalized, _ = rotate_factors(a / scale[:, None], method, tol=1e-13, max_tries=5000)
    kaiser = oe.factor(frame, names, method="pf", factors=3, rotate=method)
    assert _match(kaiser["rotated_loadings"].to_numpy(), normalized * scale[:, None]) < 1e-7
    rotated = kaiser["rotated_loadings"].to_numpy()
    ss = (rotated ** 2).sum(0)
    assert (np.diff(ss) <= 1e-12).all()                         # ordered by variance
    assert (rotated[np.abs(rotated).argmax(0), np.arange(3)] > 0).all()
    np.testing.assert_allclose(kaiser["variance"]["rotated_variance"], ss, rtol=1e-10)
    np.testing.assert_allclose(kaiser["variance"]["rotated_cumulative"].iloc[-1],
                               kaiser["variance"]["cumulative"].iloc[-1], rtol=1e-10)
    assert "structure" not in kaiser and "factor_correlations" not in kaiser


def _orthomax(loadings: np.ndarray, gamma: float) -> float:
    p = loadings.shape[0]
    return (loadings ** 4).sum() - gamma / p * ((loadings ** 2).sum(0) ** 2).sum()


def test_equamax_maximizes_the_orthomax_criterion():
    frame, names = _factor_data()
    a = oe.factor(frame, names, method="pf", factors=3)["loadings"].to_numpy()
    result = oe.factor(frame, names, method="pf", factors=3, rotate="equamax", kaiser=False)
    ours = _orthomax(result["rotated_loadings"].to_numpy(), 1.5)

    def rotation_matrix(angles: np.ndarray) -> np.ndarray:
        out = np.eye(3)
        for (i, j), angle in zip([(0, 1), (0, 2), (1, 2)], angles, strict=True):
            step = np.eye(3)
            step[[i, i, j, j], [i, j, i, j]] = [np.cos(angle), -np.sin(angle), np.sin(angle),
                                                np.cos(angle)]
            out = out @ step
        return out

    rng = np.random.default_rng(0)
    best = max(-sopt.minimize(lambda t: -_orthomax(a @ rotation_matrix(t), 1.5),
                              rng.uniform(-np.pi, np.pi, 3), method="BFGS",
                              options={"gtol": 1e-10}).fun for _ in range(12))
    assert ours == pytest.approx(best, rel=1e-9)
    # With two factors gamma = m/2 = 1: equamax is varimax.
    two = oe.factor(frame, names, method="pf", factors=2, rotate="equamax")
    var = oe.factor(frame, names, method="pf", factors=2, rotate="varimax")
    np.testing.assert_allclose(two["rotated_loadings"], var["rotated_loadings"], atol=1e-9)


@pytest.mark.parametrize("gamma", [0.0, -0.5])
def test_oblimin_matches_statsmodels(gamma):
    frame, names = _factor_data()
    a = oe.factor(frame, names, method="pf", factors=3)["loadings"].to_numpy()
    result = oe.factor(frame, names, method="pf", factors=3, rotate="oblimin", gamma=gamma,
                       kaiser=False)
    expected, _ = rotate_factors(a, "oblimin", gamma, "oblique", tol=1e-12, max_tries=20000)
    pattern = result["rotated_loadings"].to_numpy()
    assert _match(pattern, expected) < 5e-6
    phi = result["factor_correlations"].to_numpy()
    np.testing.assert_allclose(np.diag(phi), 1.0)
    np.testing.assert_allclose(pattern @ phi @ pattern.T, a @ a.T, atol=1e-9)
    np.testing.assert_allclose(result["structure"].to_numpy(), pattern @ phi, atol=1e-12)
    matrix = result["rotation_matrix"].to_numpy()
    np.testing.assert_allclose(a @ matrix, pattern, atol=1e-10)
    np.testing.assert_allclose(np.linalg.inv(matrix.T @ matrix), phi, atol=1e-9)
    np.testing.assert_allclose(result["variance"]["rotated_variance"],
                               ((pattern @ phi) ** 2).sum(0), rtol=1e-10)
    assert "rotated_cumulative" not in result["variance"]
    # The criterion is stationary: random oblique perturbations do not lower it.
    value = rotation._oblimin_value(torch.tensor(pattern), gamma)[0]
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = np.linalg.inv(matrix).T + 1e-4 * rng.normal(size=(3, 3))
        t = t / np.sqrt((t ** 2).sum(0))
        trial = a @ np.linalg.inv(t).T
        assert rotation._oblimin_value(torch.tensor(trial), gamma)[0] >= value - 1e-12


def _promax_numpy(a: np.ndarray, power: float, kaiser: bool) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt((a ** 2).sum(1)) if kaiser else np.ones(a.shape[0])
    normalized, _ = rotate_factors(a / scale[:, None], "varimax", tol=1e-13, max_tries=5000)
    varimax = normalized * scale[:, None]
    base = normalized if kaiser else varimax
    target = np.sign(base) * np.abs(base) ** power
    u = np.linalg.lstsq(varimax, target, rcond=None)[0]
    u = u * np.sqrt(np.diag(np.linalg.inv(u.T @ u)))
    return varimax @ u, np.linalg.inv(u.T @ u)


@pytest.mark.parametrize(("kaiser", "power"), [(True, 4.0), (False, 3.0)])
def test_promax_matches_the_hendrickson_white_formulas(kaiser, power):
    frame, names = _factor_data()
    a = oe.factor(frame, names, method="ml", factors=3)["loadings"].to_numpy()
    result = oe.factor(frame, names, method="ml", factors=3, rotate="promax", kaiser=kaiser,
                       power=power)
    pattern, phi = _promax_numpy(a, power, kaiser)
    ours = result["rotated_loadings"].to_numpy()
    assert _match(ours, pattern) < 1e-7
    ours_phi = result["factor_correlations"].to_numpy()
    np.testing.assert_allclose(np.sort(np.abs(ours_phi[np.triu_indices(3, 1)])),
                               np.sort(np.abs(phi[np.triu_indices(3, 1)])), atol=1e-7)
    np.testing.assert_allclose(ours @ ours_phi @ ours.T, a @ a.T, atol=1e-9)
    assert result.attrs["power"] == power and result.attrs["kaiser"] is kaiser


def test_single_factor_rotation_is_identity():
    frame, names = _factor_data()
    result = oe.factor(frame, names, method="pf", factors=1, rotate="varimax")
    np.testing.assert_allclose(result["rotated_loadings"], result["loadings"])
    assert "cannot be rotated" in " ".join(result.attrs["notes"])


# ---- factor scores and adequacy tests ---------------------------------------------


def test_factor_scores_regression_and_bartlett():
    frame, names = _factor_data(n=200, seed=2)
    frame.loc[3, "x1"] = np.nan
    r = _corr(frame, names)
    complete = frame[names].dropna()
    z = ((complete - complete.mean()) / complete.std(ddof=1)).to_numpy()
    for rotate in (None, "varimax", "oblimin"):
        result = oe.factor(frame, names, method="ml", factors=3, rotate=rotate)
        pattern = result["rotated_loadings" if rotate else "loadings"].to_numpy()
        phi = result["factor_correlations"].to_numpy() if rotate == "oblimin" else np.eye(3)
        expected = np.linalg.solve(r, pattern @ phi)
        np.testing.assert_allclose(result["score_coefficients"].to_numpy(), expected, atol=1e-8)
        scores = oe.factor_scores(result, frame)
        assert scores.loc[3].isna().all() and len(scores) == len(frame)
        np.testing.assert_allclose(scores.dropna().to_numpy(), z @ expected, atol=1e-8)
        assert result.attrs["scores"] == "regression"

        bartlett = oe.factor(frame, names, method="ml", factors=3, rotate=rotate,
                             scores="bartlett")
        psi = bartlett["communalities"]["uniqueness"].to_numpy()
        weighted = pattern / psi[:, None]
        expected = weighted @ np.linalg.inv(pattern.T @ weighted)
        np.testing.assert_allclose(bartlett["score_coefficients"].to_numpy(), expected,
                                   atol=1e-7)
        # Bartlett scores are conditionally unbiased: B'L = I.
        np.testing.assert_allclose(expected.T @ pattern, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(oe.factor_scores(bartlett, frame).dropna().to_numpy(),
                                   z @ expected, atol=1e-7)


def test_factortest_kmo_and_bartlett_by_hand():
    frame, names = _factor_data(n=120, seed=5)
    frame.loc[0, "x9"] = np.nan
    result = oe.factortest(frame, names)
    r = _corr(frame, names)
    inverse = np.linalg.inv(r)
    partial = -inverse / np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
    off = ~np.eye(9, dtype=bool)
    r2 = np.where(off, r ** 2, 0).sum(1)
    a2 = np.where(off, partial ** 2, 0).sum(1)
    np.testing.assert_allclose(result["kmo"][:-1], r2 / (r2 + a2), rtol=1e-10)
    assert result.loc["Overall", "kmo"] == pytest.approx(r2.sum() / (r2.sum() + a2.sum()))
    np.testing.assert_allclose(result["smc"][:-1], 1 - 1 / np.diag(inverse), rtol=1e-10)
    n = 119
    chi2 = -(n - 1 - (2 * 9 + 5) / 6) * np.log(np.linalg.det(r))
    assert result.attrs["statistic"] == pytest.approx(chi2, rel=1e-10)
    assert result.attrs["df"] == 36
    assert result.attrs["p_value"] == pytest.approx(stats.chi2.sf(chi2, 36), rel=1e-8)
    assert result.attrs["determinant"] == pytest.approx(np.linalg.det(r), rel=1e-10)
    assert result.attrs["n"] == n and result.attrs["n_missing"] == 1


# ---- reliability ------------------------------------------------------------------


def _items(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, list]:
    rng = np.random.default_rng(seed)
    trait = rng.normal(size=n)
    data = {f"q{j}": np.round(3 + (0.5 + 0.1 * j) * trait + rng.normal(size=n)) for j in
            range(1, 6)}
    return pd.DataFrame(data), list(data)


def _alpha(matrix: np.ndarray) -> float:
    k = matrix.shape[0]
    return k / (k - 1) * (1 - np.trace(matrix) / matrix.sum())


def test_alpha_scale_and_item_statistics_by_hand():
    frame, names = _items()
    frame.loc[2, "q3"] = np.nan
    x = frame.dropna().to_numpy()
    n, k = x.shape
    s, r = np.cov(x.T), np.corrcoef(x.T)
    result = oe.alpha(frame, names)
    scale = result["scale"]["value"]
    assert scale["alpha"] == pytest.approx(_alpha(s), rel=1e-12)
    rbar = r[~np.eye(k, dtype=bool)].mean()
    assert scale["standardized_alpha"] == pytest.approx(k * rbar / (1 + (k - 1) * rbar))
    assert scale["average_interitem_correlation"] == pytest.approx(rbar)
    assert scale["average_interitem_covariance"] == pytest.approx(s[~np.eye(k, dtype=bool)].mean())
    total = x.sum(1)
    assert scale["scale_mean"] == pytest.approx(total.mean())
    assert scale["scale_variance"] == pytest.approx(total.var(ddof=1))
    assert scale["n_items"] == k and result.attrs["n"] == n and result.attrs["n_missing"] == 1
    items = result["items"]
    for j, name in enumerate(names):
        rest = total - x[:, j]
        others = np.delete(np.arange(k), j)
        assert items.loc[name, "mean"] == pytest.approx(x[:, j].mean())
        assert items.loc[name, "std_dev"] == pytest.approx(x[:, j].std(ddof=1))
        assert items.loc[name, "item_rest_correlation"] == pytest.approx(
            np.corrcoef(x[:, j], rest)[0, 1], rel=1e-10)
        assert items.loc[name, "item_test_correlation"] == pytest.approx(
            np.corrcoef(x[:, j], total)[0, 1], rel=1e-10)
        assert items.loc[name, "scale_mean_if_deleted"] == pytest.approx(rest.mean())
        assert items.loc[name, "scale_variance_if_deleted"] == pytest.approx(rest.var(ddof=1))
        assert items.loc[name, "alpha_if_deleted"] == pytest.approx(
            _alpha(s[np.ix_(others, others)]), rel=1e-10)
        beta = np.linalg.lstsq(np.column_stack([np.ones(n), x[:, others]]), x[:, j],
                               rcond=None)[0]
        fitted = np.column_stack([np.ones(n), x[:, others]]) @ beta
        assert items.loc[name, "squared_multiple_correlation"] == pytest.approx(
            1 - ((x[:, j] - fitted) ** 2).sum() / ((x[:, j] - x[:, j].mean()) ** 2).sum(),
            rel=1e-9)


def test_alpha_standardized_reverse_split_and_guttman():
    frame, names = _items(seed=1)
    x = frame.to_numpy()
    k = x.shape[1]
    s, r = np.cov(x.T), np.corrcoef(x.T)
    std = oe.alpha(frame, names, standardized=True)
    assert std.attrs["alpha"] == pytest.approx(_alpha(r))
    assert std.attrs["alpha"] == pytest.approx(std.attrs["standardized_alpha"])
    assert std["scale"].loc["scale_variance", "value"] == pytest.approx(r.sum())

    flipped = frame.assign(q2=-frame["q2"])
    reverse = oe.alpha(flipped, names, reverse=["q2"])
    assert reverse.attrs["alpha"] == pytest.approx(_alpha(s))
    assert reverse.attrs["reversed"] == ["q2"]
    assert oe.alpha(flipped, names).attrs["alpha"] < reverse.attrs["alpha"]

    split = oe.alpha(frame, names, model="split")["split"]["value"]
    first, second = x[:, :3].sum(1), x[:, 3:].sum(1)
    rho = np.corrcoef(first, second)[0, 1]
    total_variance = x.sum(1).var(ddof=1)
    assert split["n_items_part1"] == 3 and split["n_items_part2"] == 2
    assert split["correlation_between_forms"] == pytest.approx(rho)
    assert split["spearman_brown_equal_length"] == pytest.approx(2 * rho / (1 + rho))
    weight = 3 * 2 / 25
    unequal = (-rho ** 2 + math.sqrt(rho ** 4 + 4 * rho ** 2 * (1 - rho ** 2) * weight)) \
        / (2 * (1 - rho ** 2) * weight)
    assert split["spearman_brown_unequal_length"] == pytest.approx(unequal)
    assert split["guttman_split_half"] == pytest.approx(
        2 * (total_variance - first.var(ddof=1) - second.var(ddof=1)) / total_variance)
    assert split["alpha_part1"] == pytest.approx(_alpha(s[:3, :3]))
    assert split["alpha_part2"] == pytest.approx(_alpha(s[3:, 3:]))

    guttman = oe.alpha(frame, names, model="guttman")["guttman"]["value"]
    off = s - np.diag(np.diag(s))
    lambda1 = 1 - np.trace(s) / s.sum()
    assert guttman["lambda1"] == pytest.approx(lambda1)
    assert guttman["lambda2"] == pytest.approx(
        lambda1 + math.sqrt(k / (k - 1) * (off ** 2).sum()) / s.sum())
    assert guttman["lambda3"] == pytest.approx(_alpha(s))
    assert guttman["lambda4"] == pytest.approx(split["guttman_split_half"])
    assert guttman["lambda5"] == pytest.approx(
        lambda1 + 2 * math.sqrt((off ** 2).sum(0).max()) / s.sum())
    assert guttman["lambda6"] == pytest.approx(1 - (1 / np.diag(np.linalg.inv(s))).sum() / s.sum())
    two = oe.alpha(frame, names[:2])
    assert two["items"]["alpha_if_deleted"].isna().all()


# ---- failure contract -------------------------------------------------------------


def test_error_codes():
    frame, names = _factor_data(n=40)

    def code(function, *args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            function(*args, **kwargs)
        return error.value.code

    assert code(oe.pca, frame, "x1") == "invalid_spec"
    assert code(oe.pca, frame, ["x1"]) == "invalid_spec"
    assert code(oe.pca, frame, ["x1", "x1"]) == "invalid_spec"
    assert code(oe.pca, frame, ["x1", "nope"]) == "missing_columns"
    assert code(oe.pca, frame.assign(s="a"), ["x1", "s"]) == "non_numeric_column"
    assert code(oe.pca, frame.assign(k=1.0), ["x1", "k"]) == "constant_column"
    assert code(oe.pca, frame, names, matrix="spearman") == "invalid_option"
    assert code(oe.pca, frame, names, components=0) == "invalid_option"
    assert code(oe.pca, frame, names, mineigen=50.0) == "no_components"
    assert code(oe.pca, frame.iloc[:0], names) == "empty_data"
    assert code(oe.pca, 3, names) == "invalid_data"
    holes = frame.copy()
    holes.loc[0, "x1"] = np.nan
    assert code(oe.pca, holes, names, missing="raise") == "missing_values"
    assert code(oe.pca, holes.assign(x1=np.nan), names) == "empty_sample"
    assert code(oe.pca, frame.assign(x1=np.inf), names) == "non_finite_values"
    assert code(oe.pca_scores, {"a": 1}, frame) == "invalid_result"
    assert code(oe.pca_scores, oe.pca(frame, names), frame[names[:3]]) == "missing_columns"
    assert code(oe.factor, frame, names, method="minres") == "invalid_option"
    assert code(oe.factor, frame, names, rotate="geomin") == "invalid_option"
    assert code(oe.factor, frame, names, scores="anderson") == "invalid_option"
    assert code(oe.factor, frame, names, kaiser="yes") == "invalid_option"
    assert code(oe.factor, frame, names[:2]) == "invalid_spec"
    assert code(oe.factor, frame, names, mineigen=50.0) == "no_factors"
    assert code(oe.factor_scores, oe.pca(frame, names), frame) == "invalid_result"
    assert code(oe.factortest, frame.assign(d=frame["x1"] * 2), [*names, "d"]) == "singular_matrix"
    assert code(oe.alpha, frame, ["x1"]) == "invalid_spec"
    assert code(oe.alpha, frame, names, reverse=["zz"]) == "invalid_spec"
    assert code(oe.alpha, frame, names, model="parallel") == "invalid_option"
    assert code(oe.alpha, frame.assign(neg=-frame["x1"]), ["x1", "neg"]) == "degenerate_scale"


def test_singular_correlation_matrix_paths():
    frame, names = _factor_data(n=60)
    frame["sum"] = frame["x1"] + frame["x2"] - frame["x3"]
    columns = [*names, "sum"]

    def code(function, *args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            function(*args, **kwargs)
        return error.value.code

    # PCA and principal-component factors do not need an inverse and run unchanged.
    assert oe.pca(frame, columns)["eigenvalues"]["eigenvalue"].iloc[-1] == pytest.approx(
        0.0, abs=1e-12)
    result = oe.factor(frame, columns, method="pcf", rotate="varimax")
    assert "score_coefficients" not in result and result.attrs["scores"] is None
    assert "singular" in " ".join(result.attrs["notes"])
    assert code(oe.factor_scores, result, frame) == "scores_unavailable"
    assert code(oe.factor, frame, columns, method="pcf", scores="regression") == "singular_matrix"
    for method in ("pf", "ipf", "ml"):
        assert code(oe.factor, frame, columns, method=method) == "singular_matrix"
    # Reliability still reports alpha; the squared multiple correlations are left blank.
    alpha = oe.alpha(frame, columns)
    assert alpha["items"]["squared_multiple_correlation"].isna().all()
    assert np.isfinite(alpha.attrs["alpha"]) and "singular" in alpha.attrs["notes"][0]


def test_large_common_level_does_not_cost_precision():
    frame, names = _factor_data(n=300, seed=9)
    shifted = frame + 1e7
    base = oe.factor(frame, names, method="ml", factors=3, rotate="varimax")
    moved = oe.factor(shifted, names, method="ml", factors=3, rotate="varimax")
    np.testing.assert_allclose(moved["rotated_loadings"], base["rotated_loadings"], atol=1e-7)
    np.testing.assert_allclose(oe.pca(shifted, names)["eigenvalues"]["eigenvalue"],
                               oe.pca(frame, names)["eigenvalues"]["eigenvalue"], rtol=1e-8)
    np.testing.assert_allclose(oe.alpha(shifted, names).attrs["alpha"],
                               oe.alpha(frame, names).attrs["alpha"], rtol=1e-8)
