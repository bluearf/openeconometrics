"""Independent oracles for oe.pca, oe.factor, oe.factortest and oe.alpha.

Written by the verification pass without reusing the implementation's formulas:
every expected value comes from explicit NumPy algebra on the raw data,
brute-force SciPy minimization of independently written criteria, or
invariances (row permutations, rescaling, listwise deletion).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import linalg as sla
from scipy import optimize as sopt
from scipy import stats

import openecon as oe

RTOL = 1e-7


def _data(n: int = 300, seed: int = 11, scale: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Two-factor data with badly scaled, shifted variables."""
    rng = np.random.default_rng(seed)
    loadings = np.array([[.8, .1], [.7, .2], [.75, 0], [.1, .8], [0, .7], [.2, .65], [.4, .4]])
    x = rng.normal(size=(n, 2)) @ loadings.T + 0.55 * rng.normal(size=(n, 7))
    if scale:
        x = x * np.array([1.0, 1e3, 1e-3, 7.0, 0.2, 50.0, 1.0]) + np.array(
            [3.0, -2e4, 1e-2, 100.0, 0.0, 5.0, -1.0])
    names = [f"x{j}" for j in range(1, 8)]
    return pd.DataFrame(x, columns=names), names


def _desc_eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, v = np.linalg.eigh(a)
    return w[::-1], v[:, ::-1]


def _sign_largest(v: np.ndarray) -> np.ndarray:
    """Columns signed so the entry of largest |value| is positive."""
    idx = np.abs(v).argmax(0)
    s = np.sign(v[idx, np.arange(v.shape[1])])
    s[s == 0] = 1
    return v * s


def _same_columns(ours: np.ndarray, theirs: np.ndarray, atol: float) -> None:
    """Each column of ``ours`` equals some column of ``theirs`` up to sign."""
    assert ours.shape == theirs.shape
    used = set()
    for j in range(ours.shape[1]):
        best, arg = np.inf, None
        for k in range(theirs.shape[1]):
            if k in used:
                continue
            gap = min(np.abs(ours[:, j] - theirs[:, k]).max(),
                      np.abs(ours[:, j] + theirs[:, k]).max())
            if gap < best:
                best, arg = gap, k
        used.add(arg)
        assert best < atol, (j, best)


# ---- principal components -----------------------------------------------------------


@pytest.mark.parametrize("matrix", ["correlation", "covariance"])
def test_pca_against_explicit_eigen_decomposition(matrix):
    frame, names = _data()
    x = frame[names].to_numpy()
    n, p = x.shape
    target = np.corrcoef(x.T) if matrix == "correlation" else np.cov(x.T)
    w, v = _desc_eigh(target)
    v = _sign_largest(v)
    result = oe.pca(frame, names, matrix=matrix)
    keep = int((w > 1e-5).sum())
    assert result.attrs["components"] == keep and result.attrs["n"] == n
    eig = result["eigenvalues"]
    np.testing.assert_allclose(eig["eigenvalue"], w, rtol=RTOL, atol=1e-12 * w[0])
    np.testing.assert_allclose(eig["difference"].to_numpy()[:-1], w[:-1] - w[1:], rtol=RTOL,
                               atol=1e-12 * w[0])
    assert np.isnan(eig["difference"].to_numpy()[-1])
    np.testing.assert_allclose(eig["proportion"], w / np.trace(target), rtol=RTOL, atol=1e-14)
    np.testing.assert_allclose(eig["cumulative"], np.cumsum(w) / np.trace(target), rtol=RTOL)
    vectors = result["eigenvectors"].drop(columns="unexplained").to_numpy()
    np.testing.assert_allclose(vectors, v[:, :keep], atol=1e-9)
    loadings = v[:, :keep] * np.sqrt(w[:keep])
    np.testing.assert_allclose(result["loadings"].to_numpy(), loadings, rtol=1e-8,
                               atol=1e-9 * math.sqrt(w[0]))
    extraction = (loadings ** 2).sum(1)
    np.testing.assert_allclose(result["communalities"]["initial"], np.diag(target), rtol=RTOL)
    np.testing.assert_allclose(result["communalities"]["extraction"], extraction, rtol=1e-8)
    np.testing.assert_allclose(result["eigenvectors"]["unexplained"],
                               np.diag(target) - extraction, atol=1e-8 * w[0])
    assert result.attrs["trace"] == pytest.approx(np.trace(target), rel=1e-12)
    assert result.attrs["rho"] == pytest.approx(w[:keep].sum() / np.trace(target), rel=1e-12)
    np.testing.assert_allclose(result["descriptives"]["mean"], x.mean(0), rtol=1e-12)
    np.testing.assert_allclose(result["descriptives"]["std_dev"], x.std(0, ddof=1), rtol=1e-12)


def test_pca_retention_and_scores_against_numpy():
    frame, names = _data(seed=3)
    x = frame[names].to_numpy()
    r = np.corrcoef(x.T)
    w, v = _desc_eigh(r)
    v = _sign_largest(v)
    result = oe.pca(frame, names, mineigen=1.0)
    keep = int((w > 1.0).sum())
    assert result.attrs["components"] == keep
    assert oe.pca(frame, names, components=1).attrs["components"] == 1
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    scores = oe.pca_scores(result, frame).to_numpy()
    np.testing.assert_allclose(scores, z @ v[:, :keep], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scores.var(0, ddof=1), w[:keep], rtol=1e-9)
    unit = oe.pca_scores(result, frame, normalize=True).to_numpy()
    np.testing.assert_allclose(unit.var(0, ddof=1), np.ones(keep), rtol=1e-9)
    # Covariance PCA: centred scores (default) and raw scores (center=False).
    cov = oe.pca(frame, names, matrix="covariance")
    wc, vc = _desc_eigh(np.cov(x.T))
    vc = _sign_largest(vc)[:, :cov.attrs["components"]]
    np.testing.assert_allclose(oe.pca_scores(cov, frame).to_numpy(), (x - x.mean(0)) @ vc,
                               rtol=1e-7, atol=1e-7 * math.sqrt(wc[0]))
    np.testing.assert_allclose(oe.pca_scores(cov, frame, center=False).to_numpy(), x @ vc,
                               rtol=1e-7, atol=1e-7 * np.abs(x @ vc).max())


def test_pca_invariances_permutation_scaling_and_listwise_deletion():
    frame, names = _data(seed=8)
    base = oe.pca(frame, names)
    permuted = frame.sample(frac=1.0, random_state=4)
    shuffled = oe.pca(permuted, names)
    for table in ("eigenvalues", "eigenvectors", "loadings"):
        np.testing.assert_allclose(shuffled[table].to_numpy(dtype=float),
                                   base[table].to_numpy(dtype=float), atol=1e-10)
    rescaled = frame.copy()
    rescaled[names] = rescaled[names] * np.arange(1, 8) * 1e4 - 3e5
    np.testing.assert_allclose(oe.pca(rescaled, names)["loadings"].to_numpy(),
                               base["loadings"].to_numpy(), atol=1e-8)
    holes = frame.copy()
    holes.iloc[[1, 5, 9], [0, 3, 6]] = np.nan
    dropped = oe.pca(holes, names)
    clean = oe.pca(holes.dropna(), names)
    assert dropped.attrs["n_missing"] == 3 and dropped.attrs["n"] == len(frame) - 3
    np.testing.assert_allclose(dropped["loadings"].to_numpy(), clean["loadings"].to_numpy(),
                               atol=1e-12)
    scores = oe.pca_scores(dropped, holes)
    assert scores.iloc[[1, 5, 9]].isna().all().all() and len(scores) == len(frame)


# ---- factor extraction ----------------------------------------------------------------


def _smc(r: np.ndarray) -> np.ndarray:
    # R^2 of each variable on the others by least squares on the correlation matrix.
    p = r.shape[0]
    out = np.empty(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        beta = np.linalg.solve(r[np.ix_(others, others)], r[others, j])
        out[j] = r[j, others] @ beta
    return out


def test_principal_factors_and_pcf_against_numpy():
    frame, names = _data()
    r = np.corrcoef(frame[names].to_numpy().T)
    smc = _smc(r)
    reduced = r.copy()
    np.fill_diagonal(reduced, smc)
    w, v = _desc_eigh(reduced)
    result = oe.factor(frame, names, method="pf", factors=2)
    loadings = _sign_largest(v[:, :2] * np.sqrt(w[:2]))
    np.testing.assert_allclose(result["loadings"].to_numpy(), loadings, atol=1e-9)
    np.testing.assert_allclose(result["communalities"]["initial"], smc, rtol=1e-9)
    np.testing.assert_allclose(result["communalities"]["extraction"], (loadings ** 2).sum(1),
                               rtol=1e-9)
    np.testing.assert_allclose(result["communalities"]["uniqueness"],
                               1 - (loadings ** 2).sum(1), rtol=1e-9)
    eig = result["eigenvalues"]
    np.testing.assert_allclose(eig["eigenvalue"], w, atol=1e-10)
    np.testing.assert_allclose(eig["proportion"], w / smc.sum(), atol=1e-10)
    np.testing.assert_allclose(eig["initial_eigenvalue"], _desc_eigh(r)[0], atol=1e-10)
    # Default retention (Stata): eigenvalues of the reduced matrix above 5e-6.
    assert oe.factor(frame, names).attrs["factors"] == int((w > 5e-6).sum())
    pcf = oe.factor(frame, names, method="pcf")
    wr, vr = _desc_eigh(r)
    m = int((wr > 1).sum())
    assert pcf.attrs["factors"] == m
    np.testing.assert_allclose(pcf["loadings"].to_numpy(), _sign_largest(vr[:, :m] * np.sqrt(
        wr[:m])), atol=1e-9)
    np.testing.assert_allclose(pcf["communalities"]["initial"], np.ones(len(names)))


def _ipf_loop(r: np.ndarray, m: int, tol: float, max_iter: int) -> np.ndarray:
    h = _smc(r)
    for _ in range(max_iter):
        reduced = r.copy()
        np.fill_diagonal(reduced, h)
        w, v = _desc_eigh(reduced)
        lam = v[:, :m] * np.sqrt(np.maximum(w[:m], 0))
        new = (lam ** 2).sum(1)
        if np.abs(new - h).max() < tol:
            return new
        h = new
    raise AssertionError("oracle did not converge")


def test_iterated_principal_factors_reach_the_fixed_point():
    frame, names = _data(seed=21)
    r = np.corrcoef(frame[names].to_numpy().T)
    result = oe.factor(frame, names, method="ipf", factors=2, tolerance=1e-13,
                       max_iterations=20000)
    h = _ipf_loop(r, 2, 1e-14, 20000)
    np.testing.assert_allclose(result["communalities"]["extraction"], h, atol=1e-10)
    # At the fixed point the loadings are the leading eigenpairs of R with diag(h).
    lam = result["loadings"].to_numpy()
    reduced = r.copy()
    np.fill_diagonal(reduced, (lam ** 2).sum(1))
    w, v = _desc_eigh(reduced)
    _same_columns(lam, v[:, :2] * np.sqrt(w[:2]), 1e-9)
    # SPSS defaults (25 iterations, 0.001) stay within the tolerance of the fixed point.
    default = oe.factor(frame, names, method="ipf", factors=2)
    assert default.attrs["iterations"] <= 25
    np.testing.assert_allclose(default["communalities"]["extraction"], h, atol=5e-3)


def _ml_discrepancy(params: np.ndarray, r: np.ndarray, m: int) -> float:
    p = r.shape[0]
    lam = params[:p * m].reshape(p, m)
    psi = np.exp(params[p * m:])
    sigma = lam @ lam.T + np.diag(psi)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return 1e10
    return logdet + np.trace(np.linalg.solve(sigma, r)) - np.linalg.slogdet(r)[1] - p


@pytest.mark.parametrize("m", [1, 2])
def test_ml_factor_against_brute_force_full_discrepancy(m):
    frame, names = _data(n=500, seed=2)
    x = frame[names].to_numpy()
    n, p = x.shape
    r = np.corrcoef(x.T)
    result = oe.factor(frame, names, method="ml", factors=m)
    # Brute force over loadings AND log-uniquenesses, from several starts.
    rng = np.random.default_rng(0)
    best = None
    for start in range(3):
        lam0 = np.linalg.eigh(r)[1][:, ::-1][:, :m] * 0.7 + 0.05 * rng.normal(size=(p, m))
        x0 = np.concatenate([lam0.ravel(), np.log(np.full(p, 0.5))])
        fit = sopt.minimize(_ml_discrepancy, x0, args=(r, m), method="BFGS",
                            options={"gtol": 1e-11, "maxiter": 20000})
        if best is None or fit.fun < best.fun:
            best = fit
    psi = np.exp(best.x[p * m:])
    lam = best.x[:p * m].reshape(p, m)
    np.testing.assert_allclose(result["communalities"]["uniqueness"], psi, atol=2e-5)
    np.testing.assert_allclose(result["uniqueness"]["uniqueness"], psi, atol=2e-5)
    ours = result["loadings"].to_numpy()
    np.testing.assert_allclose(ours @ ours.T, lam @ lam.T, atol=2e-5)
    assert result.attrs["discrepancy"] == pytest.approx(best.fun, abs=1e-9)
    # Our loadings, evaluated in the independent discrepancy, are at least as good.
    mine = np.concatenate([ours.ravel(), np.log(result["communalities"]["uniqueness"])])
    assert _ml_discrepancy(mine, r, m) <= best.fun + 1e-10
    # Bartlett-corrected LR tests (SPSS / Stata factor, ml).
    fmin = _ml_discrepancy(mine, r, m)
    df = ((p - m) ** 2 - (p + m)) / 2
    chi2 = (n - 1 - (2 * p + 5) / 6 - 2 * m / 3) * fmin
    fit_table = result["fit"]
    assert fit_table.loc["model", "df"] == df
    assert fit_table.loc["model", "statistic"] == pytest.approx(chi2, rel=1e-6)
    assert fit_table.loc["model", "p_value"] == pytest.approx(stats.chi2.sf(chi2, df), rel=1e-5)
    indep = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(r)[1]
    assert fit_table.loc["independence", "statistic"] == pytest.approx(indep, rel=1e-10)
    assert fit_table.loc["independence", "df"] == p * (p - 1) / 2
    # ML loadings are in canonical form: L' Psi^{-1} L diagonal.
    gram = ours.T @ (ours / result["communalities"]["uniqueness"].to_numpy()[:, None])
    assert np.abs(gram - np.diag(np.diag(gram))).max() < 1e-6 * np.abs(gram).max()


def test_ml_factor_is_invariant_to_scale_and_row_order():
    frame, names = _data(n=400, seed=9)
    a = oe.factor(frame, names, method="ml", factors=2)
    b = oe.factor(frame.iloc[::-1] * 3.0 + 1.0, names, method="ml", factors=2)
    np.testing.assert_allclose(a["communalities"].to_numpy(), b["communalities"].to_numpy(),
                               atol=1e-7)
    np.testing.assert_allclose(a["loadings"].to_numpy(), b["loadings"].to_numpy(), atol=1e-6)


# ---- rotations ------------------------------------------------------------------------


def _orthomax_value(lam: np.ndarray, gamma: float) -> float:
    p = lam.shape[0]
    sq = lam ** 2
    return float((sq ** 2).sum() - gamma / p * (sq.sum(0) ** 2).sum())


def _orthogonal(angles: np.ndarray, m: int) -> np.ndarray:
    skew = np.zeros((m, m))
    skew[np.triu_indices(m, 1)] = angles
    return sla.expm(skew - skew.T)


def _brute_orthomax(a: np.ndarray, gamma: float, kaiser: bool) -> np.ndarray:
    h = np.sqrt((a ** 2).sum(1)) if kaiser else np.ones(a.shape[0])
    an = a / h[:, None]
    m = a.shape[1]
    best = None
    rng = np.random.default_rng(1)
    for start in range(12):
        x0 = rng.uniform(-np.pi, np.pi, m * (m - 1) // 2) if start else np.zeros(m * (m - 1) // 2)
        fit = sopt.minimize(lambda t: -_orthomax_value(an @ _orthogonal(t, m), gamma), x0,
                            method="BFGS", options={"gtol": 1e-12})
        if best is None or fit.fun < best.fun - 1e-12:
            best = fit
    return (an @ _orthogonal(best.x, m)) * h[:, None]


@pytest.mark.parametrize("rotate,gamma", [("varimax", 1.0), ("quartimax", 0.0),
                                          ("equamax", None)])
@pytest.mark.parametrize("kaiser", [True, False])
def test_orthogonal_rotations_against_brute_force_maximization(rotate, gamma, kaiser):
    frame, names = _data(n=400, seed=4)
    rng = np.random.default_rng(3)
    extra = rng.normal(size=(400, 2))
    frame = frame.assign(x8=frame["x1"] * 0.01 + extra[:, 0], x9=extra[:, 1] + frame["x4"] * 0.1)
    names = [*names, "x8", "x9"]
    base = oe.factor(frame, names, method="ml", factors=3)
    a = base["loadings"].to_numpy()
    result = oe.factor(frame, names, method="ml", factors=3, rotate=rotate, kaiser=kaiser)
    g = 3 / 2 if gamma is None else gamma
    expected = _brute_orthomax(a, g, kaiser)
    ours = result["rotated_loadings"].to_numpy()
    _same_columns(ours, expected, 1e-6)
    m_ = result["rotation_matrix"].to_numpy()
    np.testing.assert_allclose(m_.T @ m_, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(a @ m_, ours, atol=1e-10)
    # Order: decreasing sums of squared loadings; sign: largest |loading| positive.
    ss = (ours ** 2).sum(0)
    assert np.all(np.diff(ss) <= 1e-12)
    assert np.all(ours[np.abs(ours).argmax(0), range(3)] > 0)
    variance = result["variance"]
    np.testing.assert_allclose(variance["rotated_variance"], ss, rtol=1e-10)
    np.testing.assert_allclose(variance["rotated_cumulative"], np.cumsum(ss) / len(names),
                               rtol=1e-10)


def _oblimin_value(lam: np.ndarray, gamma: float) -> float:
    p = lam.shape[0]
    sq = lam ** 2
    total = 0.0
    m = lam.shape[1]
    for s in range(m):
        for t in range(m):
            if s != t:
                total += (sq[:, s] * sq[:, t]).sum() - gamma / p * sq[:, s].sum() * sq[:, t].sum()
    return total / 4.0


def _brute_oblimin(a: np.ndarray, gamma: float, kaiser: bool) -> tuple[np.ndarray, np.ndarray]:
    h = np.sqrt((a ** 2).sum(1)) if kaiser else np.ones(a.shape[0])
    an = a / h[:, None]
    m = a.shape[1]

    def unpack(t):
        tt = t.reshape(m, m)
        return tt / np.sqrt((tt ** 2).sum(0))

    def value(t):
        tt = unpack(t)
        return _oblimin_value(an @ np.linalg.inv(tt).T, gamma)

    fit = sopt.minimize(value, np.eye(m).ravel(), method="BFGS", options={"gtol": 1e-13})
    tt = unpack(fit.x)
    return (an @ np.linalg.inv(tt).T) * h[:, None], tt.T @ tt


@pytest.mark.parametrize("gamma", [0.0, -0.5, 0.3])
@pytest.mark.parametrize("kaiser", [True, False])
def test_oblimin_against_brute_force_minimization(gamma, kaiser):
    frame, names = _data(n=500, seed=6)
    base = oe.factor(frame, names, method="ipf", factors=2, tolerance=1e-10,
                     max_iterations=5000)
    a = base["loadings"].to_numpy()
    result = oe.factor(frame, names, method="ipf", factors=2, tolerance=1e-10,
                       max_iterations=5000, rotate="oblimin", gamma=gamma, kaiser=kaiser)
    pattern, phi = _brute_oblimin(a, gamma, kaiser)
    ours = result["rotated_loadings"].to_numpy()
    _same_columns(ours, pattern, 2e-6)
    ours_phi = result["factor_correlations"].to_numpy()
    assert abs(abs(ours_phi[0, 1]) - abs(phi[0, 1])) < 2e-6
    # Reproduced common variance unchanged; structure = pattern Phi.
    np.testing.assert_allclose(ours @ ours_phi @ ours.T, a @ a.T, atol=1e-10)
    np.testing.assert_allclose(result["structure"].to_numpy(), ours @ ours_phi, atol=1e-12)
    m_ = result["rotation_matrix"].to_numpy()
    np.testing.assert_allclose(np.linalg.inv(m_.T @ m_), ours_phi, atol=1e-10)
    np.testing.assert_allclose(a @ m_, ours, atol=1e-10)


def _promax_oracle(a: np.ndarray, kappa: float, kaiser: bool) -> tuple[np.ndarray, np.ndarray]:
    """Varimax, then Hendrickson-White with the SPSS row-normalized target under Kaiser."""
    v = _brute_orthomax(a, 1.0, kaiser)
    h = np.sqrt((v ** 2).sum(1)) if kaiser else np.ones(v.shape[0])
    base = v / h[:, None]
    target = np.abs(base) ** kappa * np.sign(base)
    u = np.linalg.lstsq(v, target, rcond=None)[0]
    d = np.diag(np.linalg.inv(u.T @ u))
    u = u * np.sqrt(d)
    return v @ u, np.linalg.inv(u.T @ u)


@pytest.mark.parametrize("kaiser,power", [(True, 4.0), (False, 3.0), (True, 2.5)])
def test_promax_against_explicit_hendrickson_white(kaiser, power):
    frame, names = _data(n=500, seed=12)
    base = oe.factor(frame, names, method="pf", factors=2)
    a = base["loadings"].to_numpy()
    result = oe.factor(frame, names, method="pf", factors=2, rotate="promax", power=power,
                       kaiser=kaiser)
    pattern, phi = _promax_oracle(a, power, kaiser)
    _same_columns(result["rotated_loadings"].to_numpy(), pattern, 1e-7)
    assert abs(abs(result["factor_correlations"].iloc[0, 1]) - abs(phi[0, 1])) < 1e-7
    ours = result["rotated_loadings"].to_numpy()
    phi_ours = result["factor_correlations"].to_numpy()
    np.testing.assert_allclose(ours @ phi_ours @ ours.T, a @ a.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(phi_ours), 1.0)


# ---- factor scores, KMO, Bartlett ------------------------------------------------------


@pytest.mark.parametrize("rotate", [None, "varimax", "promax"])
def test_factor_scores_against_explicit_formulas(rotate):
    frame, names = _data(n=300, seed=31)
    x = frame[names].to_numpy()
    r = np.corrcoef(x.T)
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    reg = oe.factor(frame, names, method="ml", factors=2, rotate=rotate)
    lam = reg["rotated_loadings" if rotate else "loadings"].to_numpy()
    phi = reg["factor_correlations"].to_numpy() if rotate == "promax" else np.eye(2)
    coef = np.linalg.solve(r, lam @ phi)
    np.testing.assert_allclose(reg["score_coefficients"].to_numpy(), coef, atol=1e-9)
    np.testing.assert_allclose(oe.factor_scores(reg, frame).to_numpy(), z @ coef, atol=1e-8)
    bart = oe.factor(frame, names, method="ml", factors=2, rotate=rotate, scores="bartlett")
    psi = bart["communalities"]["uniqueness"].to_numpy()
    w = lam / psi[:, None]
    coef_b = w @ np.linalg.inv(lam.T @ w)
    np.testing.assert_allclose(bart["score_coefficients"].to_numpy(), coef_b, atol=1e-8)
    np.testing.assert_allclose(coef_b.T @ lam, np.eye(2), atol=1e-10)     # conditionally unbiased
    np.testing.assert_allclose(oe.factor_scores(bart, frame).to_numpy(), z @ coef_b, atol=1e-7)


def test_factortest_kmo_and_bartlett_against_numpy():
    frame, names = _data(n=200, seed=40)
    x = frame[names].to_numpy()
    n, p = x.shape
    r = np.corrcoef(x.T)
    inv = np.linalg.inv(r)
    partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    off = ~np.eye(p, dtype=bool)
    r2 = np.where(off, r ** 2, 0)
    a2 = np.where(off, partial ** 2, 0)
    msa = r2.sum(1) / (r2.sum(1) + a2.sum(1))
    kmo = r2.sum() / (r2.sum() + a2.sum())
    result = oe.factortest(frame, names)
    np.testing.assert_allclose(result["kmo"].to_numpy()[:-1], msa, rtol=1e-10)
    assert result.attrs["kmo"] == pytest.approx(kmo, rel=1e-12)
    assert result.loc["Overall", "kmo"] == pytest.approx(kmo, rel=1e-12)
    np.testing.assert_allclose(result["smc"].to_numpy()[:-1], _smc(r), rtol=1e-9)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(r))
    assert result.attrs["statistic"] == pytest.approx(chi2, rel=1e-10)
    assert result.attrs["df"] == p * (p - 1) / 2
    assert result.attrs["p_value"] == pytest.approx(stats.chi2.sf(chi2, p * (p - 1) / 2),
                                                    rel=1e-8)
    assert result.attrs["determinant"] == pytest.approx(np.linalg.det(r), rel=1e-10)


# ---- reliability ----------------------------------------------------------------------


def _items(n: int = 120, k: int = 6, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    trait = rng.normal(size=n)
    x = np.clip(np.round(3 + trait[:, None] * rng.uniform(0.4, 1.2, k)
                         + rng.normal(size=(n, k)) * 0.9), 1, 5)
    names = [f"q{j}" for j in range(1, k + 1)]
    return pd.DataFrame(x, columns=names), names


def _cronbach(x: np.ndarray) -> float:
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(0, ddof=1).sum() / x.sum(1).var(ddof=1))


def test_alpha_scale_and_item_table_from_raw_sums():
    frame, names = _items()
    x = frame[names].to_numpy()
    n, k = x.shape
    result = oe.alpha(frame, names)
    total = x.sum(1)
    corr = np.corrcoef(x.T)
    rbar = corr[~np.eye(k, dtype=bool)].mean()
    cov = np.cov(x.T)
    scale = result["scale"]["value"]
    assert scale["alpha"] == pytest.approx(_cronbach(x), rel=1e-12)
    assert scale["standardized_alpha"] == pytest.approx(k * rbar / (1 + (k - 1) * rbar), rel=1e-12)
    assert scale["n_items"] == k
    assert scale["average_interitem_covariance"] == pytest.approx(
        cov[~np.eye(k, dtype=bool)].mean(), rel=1e-12)
    assert scale["average_interitem_correlation"] == pytest.approx(rbar, rel=1e-12)
    assert scale["scale_mean"] == pytest.approx(total.mean(), rel=1e-12)
    assert scale["scale_variance"] == pytest.approx(total.var(ddof=1), rel=1e-12)
    assert scale["scale_std_dev"] == pytest.approx(total.std(ddof=1), rel=1e-12)
    items = result["items"]
    for j, name in enumerate(names):
        rest = total - x[:, j]
        others = np.delete(x, j, axis=1)
        row = items.loc[name]
        assert row["mean"] == pytest.approx(x[:, j].mean(), rel=1e-12)
        assert row["std_dev"] == pytest.approx(x[:, j].std(ddof=1), rel=1e-12)
        assert row["item_test_correlation"] == pytest.approx(np.corrcoef(x[:, j], total)[0, 1],
                                                             rel=1e-10)
        assert row["item_rest_correlation"] == pytest.approx(np.corrcoef(x[:, j], rest)[0, 1],
                                                             rel=1e-10)
        assert row["scale_mean_if_deleted"] == pytest.approx(rest.mean(), rel=1e-12)
        assert row["scale_variance_if_deleted"] == pytest.approx(rest.var(ddof=1), rel=1e-10)
        assert row["alpha_if_deleted"] == pytest.approx(_cronbach(others), rel=1e-10)
        design = np.column_stack([np.ones(n), others])
        resid = x[:, j] - design @ np.linalg.lstsq(design, x[:, j], rcond=None)[0]
        r2 = 1 - (resid ** 2).sum() / ((x[:, j] - x[:, j].mean()) ** 2).sum()
        assert row["squared_multiple_correlation"] == pytest.approx(r2, rel=1e-9)


def test_alpha_standardized_and_reverse_scoring():
    frame, names = _items(seed=3)
    x = frame[names].to_numpy()
    k = x.shape[1]
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    std = oe.alpha(frame, names, standardized=True)
    assert std.attrs["alpha"] == pytest.approx(_cronbach(z), rel=1e-12)
    assert std.attrs["alpha"] == pytest.approx(std.attrs["standardized_alpha"], rel=1e-12)
    rbar = np.corrcoef(x.T)[~np.eye(k, dtype=bool)].mean()
    # Inter-item covariances of standardized items are their correlations.
    assert std["scale"].loc["average_interitem_covariance", "value"] == pytest.approx(rbar)
    assert std["scale"].loc["scale_variance", "value"] == pytest.approx(z.sum(1).var(ddof=1))
    flipped = frame.copy()
    flipped["q2"] = 6 - flipped["q2"]
    back = oe.alpha(flipped, names, reverse=["q2"])
    plain = oe.alpha(frame, names)
    assert back.attrs["alpha"] == pytest.approx(plain.attrs["alpha"], rel=1e-12)
    np.testing.assert_allclose(back["items"]["item_rest_correlation"],
                               plain["items"]["item_rest_correlation"], rtol=1e-10)


@pytest.mark.parametrize("k", [5, 6])
def test_alpha_split_half_and_guttman_from_raw_data(k):
    frame, names = _items(k=k, seed=k)
    x = frame[names].to_numpy()
    k1 = (k + 1) // 2
    s1, s2, total = x[:, :k1].sum(1), x[:, k1:].sum(1), x.sum(1)
    r = np.corrcoef(s1, s2)[0, 1]
    split = oe.alpha(frame, names, model="split")["split"]["value"]
    assert split["correlation_between_forms"] == pytest.approx(r, rel=1e-10)
    assert split["spearman_brown_equal_length"] == pytest.approx(2 * r / (1 + r), rel=1e-10)
    # Unequal-length Spearman-Brown solves the Spearman-Brown equation for the
    # reliability of the whole test from forms of lengths k1 and k2 (Horst 1951).
    k2 = k - k1
    if k1 == k2:
        assert split["spearman_brown_unequal_length"] == pytest.approx(2 * r / (1 + r))
    else:
        w = k1 * k2 / k ** 2
        sb = (-r * r + math.sqrt(r ** 4 + 4 * r * r * (1 - r * r) * w)) / (2 * (1 - r * r) * w)
        assert split["spearman_brown_unequal_length"] == pytest.approx(sb, rel=1e-10)
    guttman_half = 2 * (1 - (s1.var(ddof=1) + s2.var(ddof=1)) / total.var(ddof=1))
    assert split["guttman_split_half"] == pytest.approx(guttman_half, rel=1e-10)
    assert split["alpha_part1"] == pytest.approx(_cronbach(x[:, :k1]), rel=1e-10)
    assert split["alpha_part2"] == pytest.approx(_cronbach(x[:, k1:]), rel=1e-10)
    g = oe.alpha(frame, names, model="guttman")["guttman"]["value"]
    cov = np.cov(x.T)
    vt = total.var(ddof=1)
    off = cov - np.diag(np.diag(cov))
    l1 = 1 - np.trace(cov) / vt
    assert g["lambda1"] == pytest.approx(l1, rel=1e-10)
    assert g["lambda2"] == pytest.approx(l1 + math.sqrt(k / (k - 1) * (off ** 2).sum()) / vt,
                                         rel=1e-10)
    assert g["lambda3"] == pytest.approx(_cronbach(x), rel=1e-10)
    assert g["lambda4"] == pytest.approx(guttman_half, rel=1e-10)
    assert g["lambda5"] == pytest.approx(l1 + 2 * math.sqrt((off ** 2).sum(0).max()) / vt,
                                         rel=1e-10)
    resid_var = []
    for j in range(k):
        design = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        e = x[:, j] - design @ np.linalg.lstsq(design, x[:, j], rcond=None)[0]
        resid_var.append((e ** 2).sum() / (len(x) - 1))
    assert g["lambda6"] == pytest.approx(1 - sum(resid_var) / vt, rel=1e-9)


def test_alpha_listwise_deletion_matches_complete_cases():
    frame, names = _items(seed=9)
    holes = frame.copy()
    holes.iloc[[0, 4, 7], [1, 2, 3]] = np.nan
    a = oe.alpha(holes, names)
    b = oe.alpha(holes.dropna(), names)
    assert a.attrs["n_missing"] == 3 and a.attrs["n"] == len(frame) - 3
    np.testing.assert_allclose(a["items"].to_numpy(dtype=float), b["items"].to_numpy(dtype=float),
                               rtol=1e-12)
