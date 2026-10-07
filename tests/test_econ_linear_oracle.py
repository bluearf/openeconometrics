"""Independent numerical oracles for the linear family: areg / reghdfe / cnsreg.

Every reference value here is derived directly from the textbook or Stata
"Methods and formulas" definition and coded in NumPy on small designs:
column-equilibrated QR least squares, cluster score sums and
Cameron-Gelbach-Miller inclusion-exclusion (with the eigenvalue adjustment
of an indefinite meat), dummy-variable (LSDV) regressions with the matrix
rank for the fixed-effect estimators, duplicated rows for frequency weights
and the Lagrangian closed form for constrained least squares. statsmodels is
used only where an identical convention exists. Nothing is read from the
implementation: a disagreement is a finding about the implementation unless
the oracle itself is shown to be wrong.

Plain OLS/WLS (``oe.regress`` / ``oe.newey``) is no longer estimated by this
family: those names delegate to ``oe.ols`` (``openecon.linear_ols``), whose
own test-suite carries its oracles; the delegation is tested in
``tests/test_econ_linear.py``.
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError

# ---------------------------------------------------------------------------- helpers


def coef(result):
    return np.array([c.estimate for c in result.coefficients])


def se(result):
    return np.array([c.std_error for c in result.coefficients])


def terms(result):
    return [c.term for c in result.coefficients]


def close(actual, expected, rtol=1e-9):
    expected = np.asarray(expected, dtype=float)
    scale = float(np.abs(expected).max()) if expected.size else 1.0
    assert_allclose(np.asarray(actual, dtype=float), expected, rtol=rtol, atol=rtol * max(scale, 1e-300))


def wls(x, y, w=None):
    """Explicit weighted least squares: (b, residuals, (X'WX)^-1) by column-equilibrated QR."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    scale = np.sqrt((w[:, None] * x * x).sum(axis=0))
    q, r = np.linalg.qr(np.sqrt(w)[:, None] * x / scale)
    b = np.linalg.solve(r, q.T @ (np.sqrt(w) * y)) / scale
    r_inv = np.linalg.solve(r, np.eye(len(scale)))
    inv = (r_inv @ r_inv.T) / np.outer(scale, scale)
    return b, y - x @ b, inv


def dummies(codes):
    levels = np.unique(codes)
    return (np.asarray(codes)[:, None] == levels[None, :]).astype(float)


def partial_out(d, values):
    """Residual of ``values`` off the column space of the dummy block ``d`` (the within transform)."""
    return values - d @ np.linalg.lstsq(d, values, rcond=None)[0]


def cluster_meat(codes, scores):
    _, index = np.unique(np.asarray(codes), return_inverse=True)
    totals = np.zeros((index.max() + 1, scores.shape[1]))
    np.add.at(totals, index, scores)
    return totals.T @ totals, index.max() + 1


def two_way_meat(first, second, scores):
    """Cameron-Gelbach-Miller: M_1 + M_2 - M_12 and the smallest one-way group count."""
    pairs = pd.factorize(pd.Series(list(zip(np.asarray(first).tolist(), np.asarray(second).tolist()))))[0]
    m1, g1 = cluster_meat(first, scores)
    m2, g2 = cluster_meat(second, scores)
    m12, _ = cluster_meat(pairs, scores)
    return m1 + m2 - m12, min(g1, g2)




def psd_truncate(a):
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    return (vectors * np.clip(values, 0, None)) @ vectors.T


def generalized_wald(b, v):
    """b' V^+ b and rank(V), on the correlation scale so badly scaled regressors lose no digits."""
    spread = np.sqrt(np.diag(v))
    values, vectors = np.linalg.eigh((v + v.T) / 2 / np.outer(spread, spread))
    keep = values > 1e-12 * values.max()
    projected = vectors[:, keep].T @ (b / spread)
    return float((projected ** 2 / values[keep]).sum()), int(keep.sum())


def check_inference(result, estimates, covariance, df, rtol=1e-9, alpha=0.05):
    """Estimates, standard errors, covariance, t statistics, p-values and intervals."""
    estimates = np.asarray(estimates, float)
    covariance = np.asarray(covariance, float)
    errors = np.sqrt(np.diag(covariance))
    close(coef(result), estimates, rtol)
    close(se(result), errors, rtol)
    close(np.asarray(result.covariance_matrix), covariance, rtol)
    t = estimates / errors
    close([c.statistic for c in result.coefficients], t, rtol)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.t.sf(np.abs(t), df), rtol=1e-6, atol=1e-300)
    critical = stats.t.ppf(1 - alpha / 2, df)
    close([c.ci_low for c in result.coefficients], estimates - critical * errors, 1e-8)
    close([c.ci_high for c in result.coefficients], estimates + critical * errors, 1e-8)
    assert result.inference["use_t"] is True and result.inference["distribution"] == "t"
    assert result.inference["df_inference"] == df


def check_f(test, statistic, df1, df2):
    assert test["distribution"] == "F" and test["df"] == df1 and test["df2"] == df2
    assert_allclose(test["statistic"], statistic, rtol=1e-8)
    assert_allclose(test["p_value"], stats.f.sf(statistic, df1, df2), rtol=1e-6, atol=1e-300)


def model_f(b, v, slopes, df2):
    """Wald F of the selected coefficients being zero: (b' V^-1 b / q, q)."""
    bs, vs = b[slopes], v[np.ix_(slopes, slopes)]
    statistic, rank = generalized_wald(bs, vs)
    return statistic / rank, rank


def duplicate(frame, column):
    return frame.loc[np.repeat(np.arange(len(frame)), frame[column].astype(int))].reset_index(drop=True)


# ---------------------------------------------------------------------------- data


@pytest.fixture(scope="module")
def panel():
    """Balanced 40 firms x 8 years with a non-nested cluster variable."""
    rng = np.random.default_rng(77)
    firms, years = 40, 8
    n = firms * years
    frame = pd.DataFrame({
        "firm": np.repeat(np.arange(firms), years), "year": np.tile(np.arange(years), firms),
        "x1": rng.normal(size=n), "x2": rng.normal(size=n), "region": rng.integers(0, 9, size=n),
        "aw": rng.uniform(0.3, 2.5, size=n), "fw": rng.integers(1, 3, size=n).astype(float),
        "sec": rng.choice(["m", "s"], size=n),
    })
    frame["y"] = 0.1 * frame.firm + 0.3 * frame.year + 1.5 * frame.x1 - frame.x2 \
        + rng.normal(size=n) * (1 + 0.5 * frame.x1 ** 2)
    return frame


@pytest.fixture(scope="module")
def unbalanced():
    """Unbalanced two-way design with few clusters (the inclusion-exclusion meat is indefinite)."""
    rng = np.random.default_rng(11)
    n = 120
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "firm": rng.integers(0, 12, size=n),
                          "year": rng.integers(0, 5, size=n), "cl": rng.integers(0, 7, size=n),
                          "aw": rng.uniform(0.3, 2.5, size=n)})
    frame["y"] = 0.3 * frame.firm + 0.7 * frame.year + 1.5 * frame.x1 - frame.x2 \
        + rng.normal(size=n) * (1 + frame.x1.abs())
    return frame


# ====================================================== two-way cluster with an indefinite meat


def _psd_variants(inv, meat, factor):
    return [inv @ psd_truncate(meat) @ inv * factor, psd_truncate(inv @ meat @ inv * factor)]


def test_two_way_cluster_indefinite_meat_gets_the_cgm_eigenvalue_adjustment(unbalanced):
    """With few clusters M_1 + M_2 - M_12 need not be positive semidefinite; reghdfe/cgmreg
    then zero the negative eigenvalues (Cameron, Gelbach and Miller 2011, eq. 2.13) and warn.
    A fit must never fail with a bare 'invalid_covariance'."""
    rng = np.random.default_rng(5)
    m = 40
    frame = pd.DataFrame({"x1": rng.normal(size=m), "x2": rng.normal(size=m), "a": rng.integers(0, 4, size=m),
                          "b": rng.integers(0, 4, size=m), "c": rng.integers(0, 3, size=m)})
    frame["y"] = frame.x1 + 0.3 * frame.c + rng.normal(size=m) * (1 + frame.x1 ** 2)
    y = frame.y.to_numpy()
    # cnsreg (x1 + x2 = 1): the sandwich lives in the K - q dimensional free space b = b_p + N g
    x = np.column_stack([np.ones(m), frame.x1, frame.x2])
    r_matrix, r_value = np.array([[0.0, 1.0, 1.0]]), np.array([1.0])
    basis = np.linalg.svd(r_matrix)[2][1:].T           # orthonormal basis N of null(R)
    b_p = np.linalg.lstsq(r_matrix, r_value, rcond=None)[0]
    g, u, inv = wls(x @ basis, y - x @ b_p)
    meat, g_min = two_way_meat(frame.a, frame.b, (x @ basis) * u[:, None])
    assert np.linalg.eigvalsh(meat).min() < 0          # the oracle meat is indefinite here
    factor = g_min / (g_min - 1) * (m - 1) / (m - 2)
    result = oe.cnsreg(data=frame, y="y", x=["x1", "x2"], cluster=["a", "b"],
                       constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1}])
    assert result.inference["psd_adjusted"] is True
    assert any("definite" in warning.lower() or "cameron" in warning.lower() for warning in result.warnings)
    close(coef(result), b_p + basis @ g, 1e-10)
    assert any(np.allclose(se(result), np.sqrt(np.diag(basis @ v @ basis.T)), rtol=1e-8)
               for v in _psd_variants(inv, meat, factor))
    # reghdfe absorbing a third variable: the within scores give an indefinite meat as well
    d = dummies(frame.c)
    xs = frame[["x1", "x2"]].to_numpy()
    bw, uw, invw = wls(partial_out(d, xs), partial_out(d, y))
    meat, g_min = two_way_meat(frame.a, frame.b, partial_out(d, xs) * uw[:, None])
    assert np.linalg.eigvalsh(meat).min() < 0
    factor = g_min / (g_min - 1) * (m - 1) / (m - 2 - d.shape[1])
    result = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["c"], cluster=["a", "b"])
    assert result.inference["psd_adjusted"] is True and result.metrics["n_singletons_dropped"] == 0
    assert any("definite" in warning.lower() or "cameron" in warning.lower() for warning in result.warnings)
    close(coef(result), bw, 1e-10)
    assert any(np.allclose(se(result), np.sqrt(np.diag(v)), rtol=1e-8) for v in _psd_variants(invw, meat, factor))
    # areg with the absorbed variable as one cluster dimension: the constant's score sums vanish
    # within that dimension, so the [0, 0] entry of the inclusion-exclusion meat is M_2 - M_12
    n = len(unbalanced)
    d = dummies(unbalanced.firm)
    xs = unbalanced[["x1", "x2"]].to_numpy()
    y = unbalanced.y.to_numpy()
    within = partial(partial_out, d)
    xt = np.column_stack([np.ones(n), within(xs) + xs.mean(axis=0)])
    bt, ut, invt = wls(xt, within(y) + y.mean())
    meat, g_min = two_way_meat(unbalanced.firm, unbalanced.cl, xt * ut[:, None])
    assert np.linalg.eigvalsh(meat).min() < 0
    factor = g_min / (g_min - 1) * (n - 1) / (n - 2 - 12)
    result = oe.areg(data=unbalanced, y="y", x=["x1", "x2"], absorb="firm", cluster=["firm", "cl"])
    assert result.inference["psd_adjusted"] is True
    close(coef(result), bt, 1e-10)
    assert any(np.allclose(se(result), np.sqrt(np.diag(v)), rtol=1e-8) for v in _psd_variants(invt, meat, factor))


# ============================================================================ areg


def test_areg_nonrobust_matches_lsdv_and_reports_statas_constant(unbalanced):
    frame = unbalanced
    n, g, k = len(frame), 12, 2
    xs = frame[["x1", "x2"]].to_numpy()
    y = frame.y.to_numpy()
    d = dummies(frame.firm)
    b_lsdv, u, inv_lsdv = wls(np.column_stack([xs, d]), y)
    rss = u @ u
    df = n - k - g
    result = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm")
    assert terms(result) == ["Intercept", "x1", "x2"]
    within = partial(partial_out, d)
    xw, yw = within(xs), within(y)
    # Stata: regress (y~ + ybar) on [1, X~ + xbar]; _cons = ybar - xbar'b, slopes = within slopes
    xt = np.column_stack([np.ones(n), xw + xs.mean(axis=0)])
    bt, ut, invt = wls(xt, yw + y.mean())
    close(bt[1:], b_lsdv[:k], 1e-12)
    assert bt[0] == pytest.approx(y.mean() - xs.mean(axis=0) @ b_lsdv[:k], rel=1e-12)
    v = invt * rss / df
    assert v[1:, 1:] == pytest.approx(inv_lsdv[:k, :k] * rss / df, rel=1e-10)
    check_inference(result, bt, v, df)
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / tss
    expected = {"r_squared": r2, "adjusted_r_squared": 1 - (1 - r2) * (n - 1) / df,
                "r_squared_within": 1 - rss / (yw @ yw), "rmse": np.sqrt(rss / df), "df_model": k,
                "df_resid": df, "df_absorbed": g - 1, "n_groups": g}
    assert list(result.metrics) == list(expected)
    for name, value in expected.items():
        assert result.metrics[name] == pytest.approx(value, rel=1e-11), name
    check_f(result.tests["model"], (((yw @ yw) - rss) / k) / (rss / df), k, df)
    _, pooled, _ = wls(np.column_stack([np.ones(n), xs]), y)
    check_f(result.tests["absorbed"], ((pooled @ pooled - rss) / (g - 1)) / (rss / df), g - 1, df)
    # same shape as reghdfe: levels - redundant = df_absorbed (one level is the reported constant)
    assert result.nobs == n
    assert result.extra["absorbed"] == [{"column": "firm", "levels": g, "redundant": 1, "nested": False}]
    rows = [p["row"] for p in result.predictions]
    close([p["fitted"] for p in result.predictions], (y - u)[rows], 1e-10)


def test_areg_robust_and_cluster_use_the_full_parameter_count(unbalanced, panel):
    frame = unbalanced
    n, g, k = len(frame), 12, 2
    xs = frame[["x1", "x2"]].to_numpy()
    y = frame.y.to_numpy()
    d = dummies(frame.firm)
    within = partial(partial_out, d)
    xt = np.column_stack([np.ones(n), within(xs) + xs.mean(axis=0)])
    bt, ut, invt = wls(xt, within(y) + y.mean())
    scores = xt * ut[:, None]
    k_total = k + g
    # HC1 = White times N/(N - K_total); slopes equal the LSDV sandwich block
    robust = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", covariance="robust")
    v = invt @ (scores.T @ scores) @ invt * n / (n - k_total)
    check_inference(robust, bt, v, n - k_total)
    assert robust.spec.covariance == "HC1"
    assert robust.inference["small_sample_correction"] == pytest.approx(n / (n - k_total))
    reference = sm.OLS(y, np.column_stack([xs, d])).fit(cov_type="HC1")
    close(se(robust)[1:], reference.bse[:k], 1e-9)
    assert "absorbed" not in robust.tests
    f_stat, _ = model_f(bt, v, [1, 2], n - k_total)
    check_f(robust.tests["model"], f_stat, 2, n - k_total)
    # cluster on the absorbed variable: areg still counts the G absorbed effects (unlike xtreg, fe)
    nested = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", cluster="firm")
    meat, groups = cluster_meat(frame.firm, scores)
    v = invt @ meat @ invt * groups / (groups - 1) * (n - 1) / (n - k_total)
    check_inference(nested, bt, v, groups - 1)
    xtreg_style = invt @ meat @ invt * groups / (groups - 1) * (n - 1) / (n - k - 1)
    close(se(nested)[1:], np.sqrt(np.diag(xtreg_style)[1:]) * np.sqrt((n - k - 1) / (n - k_total)), 1e-12)
    f_stat, _ = model_f(bt, v, [1, 2], groups - 1)
    check_f(nested.tests["model"], f_stat, 2, groups - 1)
    # cluster on another variable
    other = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", cluster="cl")
    meat, groups = cluster_meat(frame.cl, scores)
    check_inference(other, bt, invt @ meat @ invt * groups / (groups - 1) * (n - 1) / (n - k_total), groups - 1)
    reference = sm.OLS(y, np.column_stack([xs, d])).fit(cov_type="cluster", cov_kwds={"groups": frame.cl})
    close(se(other)[1:], reference.bse[:k], 1e-9)
    # two-way cluster on the balanced panel (positive definite meat): G_min - 1 degrees of freedom
    m = len(panel)
    dp = dummies(panel.firm)
    xp = panel[["x1", "x2"]].to_numpy()
    yp = panel.y.to_numpy()
    within_p = partial(partial_out, dp)
    xtp = np.column_stack([np.ones(m), within_p(xp) + xp.mean(axis=0)])
    btp, utp, invtp = wls(xtp, within_p(yp) + yp.mean())
    meat, g_min = two_way_meat(panel.firm, panel.year, xtp * utp[:, None])
    two_way = oe.areg(data=panel, y="y", x=["x1", "x2"], absorb="firm", cluster=["firm", "year"])
    check_inference(two_way, btp, invtp @ meat @ invtp * g_min / (g_min - 1) * (m - 1) / (m - 2 - 40), g_min - 1)


def test_areg_weights_against_weighted_lsdv_and_replication(unbalanced):
    frame = unbalanced.assign(fw=np.random.default_rng(4).integers(1, 4, size=len(unbalanced)).astype(float))
    n, g, k = len(frame), 12, 2
    xs = frame[["x1", "x2"]].to_numpy()
    y = frame.y.to_numpy()
    d = dummies(frame.firm)
    w = frame.aw.to_numpy() * n / frame.aw.sum()
    b_lsdv, u, inv_lsdv = wls(np.column_stack([xs, d]), y, w)
    rss = w @ u ** 2
    df = n - k - g
    result = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", weights="aw", weight_type="aweight")
    close(coef(result)[1:], b_lsdv[:k], 1e-11)
    close(se(result)[1:], np.sqrt(np.diag(inv_lsdv)[:k] * rss / df), 1e-11)
    ybar, xbar = (w * y).sum() / n, (w[:, None] * xs).sum(axis=0) / n
    assert coef(result)[0] == pytest.approx(ybar - xbar @ b_lsdv[:k], rel=1e-11)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(rss / df), rel=1e-11)
    assert result.metrics["r_squared"] == pytest.approx(1 - rss / (w @ (y - ybar) ** 2), rel=1e-11)
    _, pooled, _ = wls(np.column_stack([np.ones(n), xs]), y, w)
    check_f(result.tests["absorbed"], ((w @ pooled ** 2 - rss) / (g - 1)) / (rss / df), g - 1, df)
    robust = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", weights="aw", weight_type="aweight",
                     covariance="HC1")
    z = np.column_stack([xs, d])
    scores = z * (w * u)[:, None]
    close(se(robust)[1:], np.sqrt(np.diag(inv_lsdv @ (scores.T @ scores) @ inv_lsdv)[:k] * n / df), 1e-11)
    sampling = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "HC1"
    close(se(sampling), se(robust), 1e-12)
    for extra in ({}, {"cluster": "cl"}, {"covariance": "HC1"}):
        weighted = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", weights="fw", weight_type="fweight",
                           **extra)
        rows = oe.areg(data=duplicate(frame, "fw"), y="y", x=["x1", "x2"], absorb="firm", **extra)
        assert weighted.nobs == rows.nobs == int(frame.fw.sum())
        close(coef(weighted), coef(rows), 1e-12)
        close(se(weighted), se(rows), 1e-11)
        assert weighted.metrics == pytest.approx(rows.metrics, rel=1e-10)
        for name in weighted.tests:
            assert weighted.tests[name]["statistic"] == pytest.approx(rows.tests[name]["statistic"], rel=1e-10)
            assert weighted.tests[name]["df2"] == rows.tests[name]["df2"]


def test_areg_omits_absorbed_and_collinear_regressors_and_handles_labels(panel):
    frame = panel.assign(level=panel.firm * 0.5, shifted=panel.x1 + panel.firm, group=panel.firm.map(lambda i: f"f{i}"))
    result = oe.areg(data=frame, y="y", x=["x1", "level", "shifted", "x2", "sec"], absorb="group", categorical=["sec"])
    assert terms(result) == ["Intercept", "x1", "x2", "sec[s]"]
    assert result.provenance["omitted_terms"] == ["level", "shifted"]
    plain = oe.areg(data=frame, y="y", x=["x1", "x2", "sec"], absorb="firm", categorical=["sec"])
    close(coef(result), coef(plain), 1e-12)
    close(se(result), se(plain), 1e-12)
    shuffled = oe.areg(data=frame.sample(frac=1, random_state=2).reset_index(drop=True), y="y", x=["x1", "x2", "sec"],
                       absorb="group", categorical=["sec"], cluster="region")
    ordered = oe.areg(data=frame, y="y", x=["x1", "x2", "sec"], absorb="group", categorical=["sec"], cluster="region")
    close(coef(shuffled), coef(ordered), 1e-12)
    close(se(shuffled), se(ordered), 1e-12)
    holes = frame.copy()
    holes.loc[[0, 9, 50], "x1"] = np.nan
    dropped = oe.areg(data=holes, y="y", x=["x1", "x2"], absorb="firm", missing="drop")
    close(coef(dropped), coef(oe.areg(data=holes.dropna(), y="y", x=["x1", "x2"], absorb="firm")), 1e-13)
    assert dropped.nobs == len(frame) - 3
    with pytest.raises(AnalysisError) as info:
        oe.areg(data=frame.assign(one=1), y="y", x=["x1"], absorb="one")
    assert info.value.code == "insufficient_groups"
    with pytest.raises(AnalysisError) as info:
        oe.areg(data=frame, y="y", x=["x1"], absorb=["firm"])
    assert info.value.code == "invalid_spec"


# ============================================================================ reghdfe


def test_reghdfe_two_way_connected_design_matches_lsdv(unbalanced):
    frame = unbalanced
    n, k = len(frame), 2
    xs = frame[["x1", "x2"]].to_numpy()
    y = frame.y.to_numpy()
    d1, d2 = dummies(frame.firm), dummies(frame.year)
    z = np.column_stack([xs, d1, d2[:, 1:]])          # full rank: 2 + 12 + 4
    b, u, inv = wls(z, y)
    rss = u @ u
    df_absorbed = 12 + 5 - 1
    df = n - k - df_absorbed
    assert np.linalg.matrix_rank(np.column_stack([d1, d2])) == df_absorbed
    result = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"])
    assert terms(result) == ["x1", "x2"]
    check_inference(result, b[:k], inv[:k, :k] * rss / df, df)
    p = np.column_stack([d1, d2])
    yw = y - p @ np.linalg.lstsq(p, y, rcond=None)[0]
    tss = ((y - y.mean()) ** 2).sum()
    r2, r2w = 1 - rss / tss, 1 - rss / (yw @ yw)
    expected = {"r_squared": r2, "adjusted_r_squared": 1 - (1 - r2) * (n - 1) / df, "r_squared_within": r2w,
                "adjusted_r_squared_within": 1 - (rss / df) / ((yw @ yw) / (n - df_absorbed)),
                "rmse": np.sqrt(rss / df), "df_model": k, "df_resid": df, "df_absorbed": df_absorbed,
                "n_singletons_dropped": 0}
    assert list(result.metrics) == list(expected)
    for name, value in expected.items():
        assert result.metrics[name] == pytest.approx(value, rel=1e-10), name
    check_f(result.tests["model"], (((yw @ yw) - rss) / k) / (rss / df), k, df)
    absorbed = result.extra["absorbed"]
    assert [(a["column"], a["levels"], a["redundant"], a["nested"]) for a in absorbed] == \
        [("firm", 12, 0, False), ("year", 5, 1, False)]
    assert result.extra["converged"] is True and result.extra["iterations"] >= 1
    rows = [p["row"] for p in result.predictions]
    close([p["fitted"] for p in result.predictions], (y - u)[rows], 1e-9)
    close([p["residual"] for p in result.predictions], u[rows], 1e-9)
    robust = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], covariance="robust")
    scores = z * u[:, None]
    v = inv @ (scores.T @ scores) @ inv * n / (n - z.shape[1])
    check_inference(robust, b[:k], v[:k, :k], df)
    close(se(robust), sm.OLS(y, z).fit(cov_type="HC1").bse[:k], 1e-9)


def test_reghdfe_degrees_of_freedom_in_disconnected_and_three_way_designs():
    rng = np.random.default_rng(31)
    n = 100
    group = rng.integers(0, 2, size=n)
    frame = pd.DataFrame({"a": rng.integers(0, 5, size=n) + 10 * group, "b": rng.integers(0, 4, size=n) + 10 * group,
                          "x": rng.normal(size=n)})
    frame["y"] = frame.x + 0.3 * frame.a - 0.2 * frame.b + rng.normal(size=n)
    result = oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b"])
    z = np.column_stack([frame.x, dummies(frame.a), dummies(frame.b)])
    rank = np.linalg.matrix_rank(z)
    b = np.linalg.lstsq(z, frame.y.to_numpy(), rcond=None)[0]
    u = frame.y.to_numpy() - z @ b
    variance = np.linalg.pinv(z.T @ z)[0, 0] * (u @ u) / (n - rank)
    assert result.metrics["df_absorbed"] == rank - 1 == 16
    assert result.metrics["df_resid"] == n - rank
    assert [a["redundant"] for a in result.extra["absorbed"]] == [0, 2]      # two mobility groups
    check_inference(result, b[:1], [[variance]], n - rank)
    # three dimensions: pairwise mobility groups (one each here) reproduce the rank of the dummy block
    rng = np.random.default_rng(41)
    n = 400
    frame = pd.DataFrame({"a": rng.integers(0, 20, size=n), "b": rng.integers(0, 15, size=n),
                          "c": rng.integers(0, 10, size=n), "x": rng.normal(size=n)})
    frame["y"] = frame.x + 0.1 * frame.a + 0.2 * frame.b - 0.3 * frame.c + rng.normal(size=n)
    result = oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b", "c"])
    z = np.column_stack([frame.x, dummies(frame.a), dummies(frame.b), dummies(frame.c)])
    rank = np.linalg.matrix_rank(z)
    b = np.linalg.lstsq(z, frame.y.to_numpy(), rcond=None)[0]
    u = frame.y.to_numpy() - z @ b
    variance = np.linalg.pinv(z.T @ z)[0, 0] * (u @ u) / (n - rank)
    assert result.metrics["df_absorbed"] == rank - 1 == 20 + 15 + 10 - 2
    check_inference(result, b[:1], [[variance]], n - rank, rtol=1e-8)
    assert [a["redundant"] for a in result.extra["absorbed"]] == [0, 1, 1]


def test_reghdfe_singletons_are_dropped_iteratively():
    rng = np.random.default_rng(21)
    n = 80
    a, b = rng.integers(0, 30, size=n), rng.integers(0, 25, size=n)
    frame = pd.DataFrame({"a": a, "b": b, "x": rng.normal(size=n)})
    frame["y"] = frame.x + 0.1 * a + rng.normal(size=n)
    keep = np.ones(n, dtype=bool)
    while True:
        counts_a, counts_b = pd.Series(a[keep]).value_counts(), pd.Series(b[keep]).value_counts()
        single = keep & (np.isin(a, counts_a[counts_a == 1].index) | np.isin(b, counts_b[counts_b == 1].index))
        if not single.any():
            break
        keep &= ~single
    assert 0 < (~keep).sum() < n // 2
    result = oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b"])
    assert result.metrics["n_singletons_dropped"] == (~keep).sum() == result.extra["singletons_dropped"]
    assert result.nobs == keep.sum() and result.sample_positions == np.flatnonzero(keep).tolist()
    assert any(f"{(~keep).sum()} singleton" in warning for warning in result.warnings)
    sub = frame[keep]
    for drop, data in ((True, sub), (False, frame)):
        fit = result if drop else oe.reghdfe(data=frame, y="y", x=["x"], absorb=["a", "b"], drop_singletons=False)
        z = np.column_stack([data.x, dummies(data.a), dummies(data.b)])
        rank = np.linalg.matrix_rank(z)
        beta = np.linalg.lstsq(z, data.y.to_numpy(), rcond=None)[0]
        u = data.y.to_numpy() - z @ beta
        variance = np.linalg.pinv(z.T @ z)[0, 0] * (u @ u) / (len(data) - rank)
        check_inference(fit, beta[:1], [[variance]], len(data) - rank, rtol=1e-8)
        assert fit.metrics["df_resid"] == len(data) - rank
    assert fit.metrics["n_singletons_dropped"] == 0 and fit.nobs == n


def test_reghdfe_cluster_conventions(unbalanced, panel):
    frame = unbalanced
    n, k, g = len(frame), 2, 12
    xs = frame[["x1", "x2"]].to_numpy()
    y = frame.y.to_numpy()
    d = dummies(frame.firm)
    within = partial(partial_out, d)
    xw, yw = within(xs), within(y)
    bw, uw, invw = wls(xw, yw)
    scores = xw * uw[:, None]
    # absorb(firm) cluster(firm): the nested effects cost nothing, the constant counts: K = k + 1 (xtreg, fe)
    nested = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm"], cluster="firm")
    meat, groups = cluster_meat(frame.firm, scores)
    v = invw @ meat @ invw * groups / (groups - 1) * (n - 1) / (n - k - 1)
    check_inference(nested, bw, v, groups - 1)
    assert nested.metrics["df_resid"] == n - k - 1 and nested.metrics["df_absorbed"] == 1
    assert nested.extra["absorbed"][0]["nested"] is True and nested.extra["constant_degree_of_freedom_added"] is True
    assert nested.inference["small_sample_correction"] == pytest.approx(groups / (groups - 1) * (n - 1) / (n - k - 1))
    reference = sm.OLS(yw, np.column_stack([np.ones(n), xw])).fit(cov_type="cluster", cov_kwds={"groups": frame.firm})
    close(se(nested), reference.bse[1:], 1e-9)
    areg_nested = oe.areg(data=frame, y="y", x=["x1", "x2"], absorb="firm", cluster="firm")
    close(se(areg_nested)[1:], se(nested) * np.sqrt((n - k - 1) / (n - k - g)), 1e-12)
    # absorb(firm year) cluster(cl): nothing nested, K = k + 12 + 5 - 1 (statsmodels LSDV cluster)
    d2 = dummies(frame.year)
    z = np.column_stack([xs, d, d2[:, 1:]])
    other = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster="cl")
    reference = sm.OLS(y, z).fit(cov_type="cluster", cov_kwds={"groups": frame.cl})
    close(coef(other), reference.params[:k], 1e-11)
    close(se(other), reference.bse[:k], 1e-9)
    assert other.metrics["df_resid"] == n - z.shape[1] and other.inference["df_inference"] == 6
    # absorb(firm year) cluster(firm): firm nested (not counted), year counted in full - the family's
    # documented convention, K = k + G_year (equivalently G_year - 1 effects plus the constant)
    partially_nested = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], cluster="firm")
    p = np.column_stack([d, d2])
    xw2 = xs - p @ np.linalg.lstsq(p, xs, rcond=None)[0]
    yw2 = y - p @ np.linalg.lstsq(p, y, rcond=None)[0]
    bw2, uw2, invw2 = wls(xw2, yw2)
    meat, groups = cluster_meat(frame.firm, xw2 * uw2[:, None])
    assert partially_nested.metrics["df_absorbed"] == 5 and partially_nested.metrics["df_resid"] == n - k - 5
    v = invw2 @ meat @ invw2 * groups / (groups - 1) * (n - 1) / (n - k - 5)
    check_inference(partially_nested, bw2, v, groups - 1)
    assert [a["nested"] for a in partially_nested.extra["absorbed"]] == [True, False]
    # two-way cluster(firm year) on the balanced panel, absorbing firm: K = k + G_year? no - year is not
    # absorbed here, so K = k + 1 (firm nested in the first cluster dimension) and df = G_min - 1
    m = len(panel)
    dp = dummies(panel.firm)
    xp = panel[["x1", "x2"]].to_numpy()
    yp = panel.y.to_numpy()
    within_p = partial(partial_out, dp)
    bp, up, invp = wls(within_p(xp), within_p(yp))
    meat, g_min = two_way_meat(panel.firm, panel.year, within_p(xp) * up[:, None])
    two_way = oe.reghdfe(data=panel, y="y", x=["x1", "x2"], absorb=["firm"], cluster=["firm", "year"])
    check_inference(two_way, bp, invp @ meat @ invp * g_min / (g_min - 1) * (m - 1) / (m - k - 1), g_min - 1)
    assert two_way.metrics["df_resid"] == m - k - 1


def test_reghdfe_weights_against_weighted_lsdv_and_replication():
    rng = np.random.default_rng(51)
    n = 150
    frame = pd.DataFrame({"a": rng.integers(0, 12, size=n), "b": rng.integers(0, 6, size=n),
                          "x1": rng.normal(size=n), "x2": rng.normal(size=n), "aw": rng.uniform(0.2, 4, size=n),
                          "fw": rng.integers(1, 4, size=n).astype(float), "cl": rng.integers(0, 8, size=n)})
    frame["y"] = frame.x1 - frame.x2 + 0.1 * frame.a + 0.3 * frame.b + rng.normal(size=n) * (1 + frame.x1.abs())
    y = frame.y.to_numpy()
    w = frame.aw.to_numpy() * n / frame.aw.sum()
    z = np.column_stack([frame.x1, frame.x2, dummies(frame.a), dummies(frame.b)[:, 1:]])
    b, u, inv = wls(z, y, w)
    rss = w @ u ** 2
    df = n - z.shape[1]
    result = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["a", "b"], weights="aw", weight_type="aweight")
    check_inference(result, b[:2], inv[:2, :2] * rss / df, df)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(rss / df), rel=1e-11)
    ybar = (w * y).sum() / n
    assert result.metrics["r_squared"] == pytest.approx(1 - rss / (w @ (y - ybar) ** 2), rel=1e-11)
    scores = z * (w * u)[:, None]
    robust = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["a", "b"], weights="aw", weight_type="aweight",
                        covariance="HC1")
    close(se(robust), np.sqrt(np.diag(inv @ (scores.T @ scores) @ inv)[:2] * n / df), 1e-11)
    clustered = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["a", "b"], weights="aw", weight_type="pweight",
                           cluster="cl")
    meat, groups = cluster_meat(frame.cl, scores)
    v = inv @ meat @ inv * groups / (groups - 1) * (n - 1) / df
    check_inference(clustered, b[:2], v[:2, :2], groups - 1)
    for extra in ({}, {"cluster": "cl"}, {"covariance": "HC1"}):
        weighted = oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["a", "b"], weights="fw",
                              weight_type="fweight", **extra)
        rows = oe.reghdfe(data=duplicate(frame, "fw"), y="y", x=["x1", "x2"], absorb=["a", "b"], **extra)
        assert weighted.nobs == rows.nobs == int(frame.fw.sum())
        close(coef(weighted), coef(rows), 1e-12)
        close(se(weighted), se(rows), 1e-11)
        assert weighted.metrics == pytest.approx(rows.metrics, rel=1e-10)


def test_reghdfe_frequency_weighted_rows_are_not_singletons():
    """A row with fweight 3 stands for three observations: reghdfe (ftools drop_singletons with
    fweights) keeps it, and the result must equal the duplicated-row data set."""
    rng = np.random.default_rng(9)
    n = 60
    firm = rng.integers(0, 15, size=n)
    firm[0] = 99                                   # alone in its level, but weight 3
    frame = pd.DataFrame({"firm": firm, "year": rng.integers(0, 4, size=n), "x": rng.normal(size=n),
                          "fw": rng.integers(1, 3, size=n).astype(float)})
    frame.loc[0, "fw"] = 3.0
    frame["y"] = 0.5 * frame.x + 0.3 * frame.year + rng.normal(size=n)
    weighted = oe.reghdfe(data=frame, y="y", x=["x"], absorb=["firm", "year"], weights="fw", weight_type="fweight")
    rows = oe.reghdfe(data=duplicate(frame, "fw"), y="y", x=["x"], absorb=["firm", "year"])
    assert rows.metrics["n_singletons_dropped"] >= 1          # a weight-one singleton still goes
    assert weighted.metrics["n_singletons_dropped"] == rows.metrics["n_singletons_dropped"]
    assert 0 in weighted.sample_positions
    assert weighted.nobs == rows.nobs
    close(coef(weighted), coef(rows), 1e-12)
    close(se(weighted), se(rows), 1e-11)
    assert weighted.metrics == pytest.approx(rows.metrics, rel=1e-10)


def test_reghdfe_omits_absorbed_regressors_options_and_errors(panel):
    frame = panel.assign(level=panel.firm * 1.0, label=panel.sec)
    result = oe.reghdfe(data=frame, y="y", x=["x1", "level", "x2", "sec"], absorb=["firm", "year"], categorical=["sec"])
    assert terms(result) == ["x1", "x2", "sec[s]"] and result.provenance["omitted_terms"] == ["level"]
    assert "Intercept" not in terms(result)
    permuted = frame.sample(frac=1, random_state=8).reset_index(drop=True)
    shuffled = oe.reghdfe(data=permuted, y="y", x=["x1", "x2", "sec"], absorb=["firm", "year"],
                          categorical=["sec"], cluster="region")
    ordered = oe.reghdfe(data=frame, y="y", x=["x1", "x2", "sec"], absorb=["firm", "year"], categorical=["sec"],
                         cluster="region")
    close(coef(shuffled), coef(ordered), 1e-11)
    close(se(shuffled), se(ordered), 1e-11)
    with pytest.raises(AnalysisError) as info:
        oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], max_iterations=1, tolerance=1e-14)
    assert info.value.code == "absorption_nonconvergence"
    for bad in ({"tolerance": 0.0}, {"tolerance": 1.5}):
        with pytest.raises(AnalysisError) as info:
            oe.reghdfe(data=frame, y="y", x=["x1", "x2"], absorb=["firm", "year"], **bad)
        assert info.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as info:
        oe.reghdfe(data=frame, y="y", x=["level"], absorb=["firm"])
    assert info.value.code == "empty_design"
    with pytest.raises(AnalysisError) as info:
        oe.reghdfe(data=frame, y="y", x=["x1"], absorb=["firm"], weights="aw", weight_type="pweight",
                   covariance="nonrobust")
    assert info.value.code == "unsupported_covariance"


# ============================================================================ cnsreg


def lagrangian(x, y, r_matrix, r_value, w=None):
    """Constrained least squares: b_c = b - (X'WX)^-1 R'[R (X'WX)^-1 R']^-1 (R b - r) and the
    restricted inverse C = A - A R'(R A R')^-1 R A with A = (X'WX)^-1."""
    b, _, inv = wls(x, y, w)
    middle = np.linalg.solve(r_matrix @ inv @ r_matrix.T, np.eye(len(r_value)))
    bc = b - inv @ r_matrix.T @ middle @ (r_matrix @ b - r_value)
    c = inv - inv @ r_matrix.T @ middle @ r_matrix @ inv
    return bc, y - x @ bc, c


@pytest.fixture(scope="module")
def cns():
    rng = np.random.default_rng(77)
    n = 70
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n) * 50, "x3": rng.normal(size=n),
                          "sec": rng.choice(["a", "b", "c"], size=n), "cl": rng.integers(0, 9, size=n),
                          "aw": rng.uniform(0.3, 3, size=n), "fw": rng.integers(1, 4, size=n).astype(float)})
    frame["y"] = 2 + frame.x1 + 0.02 * frame.x2 - frame.x3 + (frame.sec == "b") * 0.5 \
        + rng.normal(size=n) * (1 + frame.x1 ** 2)
    return frame


CONSTRAINTS = [{"terms": {"x1": 1, "x3": 1}, "value": 0}, {"terms": {"Intercept": 1, "sec[b]": -2}, "value": 1}]
R_MATRIX = np.array([[0, 1, 0, 1, 0, 0], [1, 0, 0, 0, -2, 0]], dtype=float)
R_VALUE = np.array([0.0, 1.0])


def cns_design(frame):
    return np.column_stack([np.ones(len(frame)), frame.x1, frame.x2, frame.x3, frame.sec == "b",
                            frame.sec == "c"]).astype(float)


def test_cnsreg_matches_the_lagrangian_solution(cns):
    n, k, q = len(cns), 6, 2
    x, y = cns_design(cns), cns.y.to_numpy()
    bc, uc, c = lagrangian(x, y, R_MATRIX, R_VALUE)
    rss, df = uc @ uc, n - k + q
    result = oe.cnsreg(data=cns, y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"], constraints=CONSTRAINTS)
    assert terms(result) == ["Intercept", "x1", "x2", "x3", "sec[b]", "sec[c]"]
    assert R_MATRIX @ coef(result) == pytest.approx(R_VALUE, abs=1e-12)
    check_inference(result, bc, c * rss / df, df)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(rss / df), rel=1e-11)
    assert result.metrics["df_resid"] == df and result.metrics["df_constraints"] == q
    assert "r_squared" not in result.metrics
    statistic, rank = generalized_wald(bc[1:], c[1:, 1:] * rss / df)
    assert rank == k - q == 4 and result.metrics["df_model"] == rank
    check_f(result.tests["model"], statistic / rank, rank, df)
    assert result.extra["constrained_terms"] == {}
    assert result.extra["constraint_matrix"]["R"] == R_MATRIX.tolist()
    rows = [p["row"] for p in result.predictions]
    close([p["fitted"] for p in result.predictions], (x @ bc)[rows], 1e-11)
    # the constrained RSS exceeds the unconstrained one and the solution is the minimum on R b = r
    _, u_free, _ = wls(x, y)
    assert rss > u_free @ u_free
    for shift in np.linalg.svd(R_MATRIX)[2][q:]:
        assert ((y - x @ (bc + 1e-3 * shift)) ** 2).sum() >= rss


def test_cnsreg_robust_cluster_and_weights(cns):
    n, k, q = len(cns), 6, 2
    x, y = cns_design(cns), cns.y.to_numpy()
    bc, uc, c = lagrangian(x, y, R_MATRIX, R_VALUE)
    df = n - k + q
    scores = x * uc[:, None]
    robust = oe.cnsreg(data=cns, y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"], constraints=CONSTRAINTS,
                       covariance="robust")
    check_inference(robust, bc, c @ (scores.T @ scores) @ c * n / df, df)
    assert robust.inference["small_sample_correction"] == pytest.approx(n / df)
    clustered = oe.cnsreg(data=cns, y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"], constraints=CONSTRAINTS,
                          cluster="cl")
    meat, groups = cluster_meat(cns.cl, scores)
    v = c @ meat @ c * groups / (groups - 1) * (n - 1) / df
    check_inference(clustered, bc, v, groups - 1)
    f_stat, rank = model_f(bc, v, [1, 2, 3, 4, 5], groups - 1)
    check_f(clustered.tests["model"], f_stat, rank, groups - 1)
    two_way = oe.cnsreg(data=cns, y="y", x=["x1", "x2"], constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1}],
                        cluster=["cl", "sec"])
    x2 = x[:, :3]
    b2, u2, c2 = lagrangian(x2, y, np.array([[0, 1, 1.0]]), np.array([1.0]))
    meat, g_min = two_way_meat(cns.cl, cns.sec, x2 * u2[:, None])
    check_inference(two_way, b2, c2 @ meat @ c2 * g_min / (g_min - 1) * (n - 1) / (n - 2), g_min - 1, rtol=1e-8)
    w = cns.aw.to_numpy() * n / cns.aw.sum()
    bw, uw, cw = lagrangian(x, y, R_MATRIX, R_VALUE, w)
    rss = w @ uw ** 2
    analytic = oe.cnsreg(data=cns, y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"], constraints=CONSTRAINTS,
                         weights="aw", weight_type="aweight")
    check_inference(analytic, bw, cw * rss / df, df)
    assert analytic.metrics["rmse"] == pytest.approx(np.sqrt(rss / df), rel=1e-11)
    sampling = oe.cnsreg(data=cns, y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"], constraints=CONSTRAINTS,
                         weights="aw", weight_type="pweight")
    scores = x * (w * uw)[:, None]
    assert sampling.spec.covariance == "HC1"
    check_inference(sampling, bw, cw @ (scores.T @ scores) @ cw * n / df, df)
    for extra in ({}, {"cluster": "cl"}):
        weighted = oe.cnsreg(data=cns, y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"],
                             constraints=CONSTRAINTS, weights="fw", weight_type="fweight", **extra)
        rows = oe.cnsreg(data=duplicate(cns, "fw"), y="y", x=["x1", "x2", "x3", "sec"], categorical=["sec"],
                         constraints=CONSTRAINTS, **extra)
        assert weighted.nobs == rows.nobs == int(cns.fw.sum())
        close(coef(weighted), coef(rows), 1e-12)
        close(se(weighted), se(rows), 1e-11)
        assert weighted.metrics == pytest.approx(rows.metrics, rel=1e-10)
        assert weighted.tests["model"]["statistic"] == pytest.approx(rows.tests["model"]["statistic"], rel=1e-10)


def test_cnsreg_constraint_reduction_fixed_terms_and_errors(cns):
    n = len(cns)
    x, y = cns_design(cns)[:, :3], cns.y.to_numpy()
    unit_sum = {"terms": {"x1": 1, "x2": 1}, "value": 1}
    redundant = oe.cnsreg(data=cns, y="y", x=["x1", "x2"],
                          constraints=[unit_sum, {"terms": {"x1": 2, "x2": 2}, "value": 2}])
    assert any("redundant constraint(s) 2" in warning for warning in redundant.warnings)
    assert redundant.metrics["df_constraints"] == 1 and redundant.metrics["df_resid"] == n - 3 + 1
    bc, uc, c = lagrangian(x, y, np.array([[0, 1, 1.0]]), np.array([1.0]))
    check_inference(redundant, bc, c * (uc @ uc) / (n - 2), n - 2, rtol=1e-8)
    # a coefficient fixed by the constraints has no variance: reported in extra, not in the table
    fixed = oe.cnsreg(data=cns, y="y", x=["x1", "x2"], constraints=[{"terms": {"x1": 1}, "value": 2.0}])
    assert terms(fixed) == ["Intercept", "x2"] and fixed.extra["constrained_terms"] == {"x1": 2.0}
    bc, uc, c = lagrangian(x, y, np.array([[0, 1, 0.0]]), np.array([2.0]))
    keep = [0, 2]
    check_inference(fixed, bc[keep], c[np.ix_(keep, keep)] * (uc @ uc) / (n - 2), n - 2)
    assert fixed.metrics["df_model"] == 1 and fixed.tests["model"]["df"] == 1
    close(fixed.covariance_matrix, c[np.ix_(keep, keep)] * (uc @ uc) / (n - 2), 1e-9)
    noint = oe.cnsreg(data=cns, y="y", x=["x1", "x2"], intercept=False, constraints=[{"terms": {"x1": 1}, "value": 1}])
    bc, uc, c = lagrangian(x[:, 1:], y, np.array([[1.0, 0.0]]), np.array([1.0]))
    assert terms(noint) == ["x2"] and noint.extra["constrained_terms"] == {"x1": 1.0}
    check_inference(noint, bc[1:], c[1:, 1:] * (uc @ uc) / (n - 1), n - 1)
    frame = cns.assign(x4=2 * cns.x1)
    cases = [
        ([unit_sum, {"terms": {"x1": 2, "x2": 2}, "value": 3}], "inconsistent_constraints"),
        ([{"terms": {}, "value": 1}], "inconsistent_constraints"),
        ([{"terms": {"zz": 1}, "value": 1}], "invalid_constraint"),
        ([{"terms": {"x1": "one"}, "value": 1}], "invalid_constraint"),
        ([{"terms": {"x1": 1}, "value": 1}, {"terms": {"x2": 1}, "value": 1}, {"terms": {"Intercept": 1}, "value": 1}],
         "invalid_constraint"),
        ([], "invalid_constraint"),
        ([{"terms": {"x4": 1}, "value": 1}], "invalid_constraint"),
    ]
    for constraints, code in cases:
        with pytest.raises(AnalysisError) as info:
            oe.cnsreg(data=frame, y="y", x=["x1", "x4", "x2"], constraints=constraints)
        assert info.value.code == code, constraints


# ============================================================================ scale


def test_dense_one_million_rows_fit_in_seconds(monkeypatch):
    # This existing benchmark measures the resident-table kernels. Replay IO
    # and spill performance have separate physical-file measurements.
    from openecon.econometrics import streaming_registry
    monkeypatch.setattr(streaming_registry, "supports_spec", lambda spec: False)
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1024")
    rng = np.random.default_rng(0)
    n, k = 1_000_000, 10
    x = rng.normal(size=(n, k))
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(k)])
    frame["firm"] = rng.integers(0, 100_000, size=n)
    frame["year"] = rng.integers(0, 1000, size=n)
    frame["y"] = x @ np.arange(1, k + 1) + rng.normal(size=n)
    columns = [f"x{i}" for i in range(k)]
    seconds = {}
    start = time.perf_counter()
    # the true coefficients are 1, 2, ..., 10, so x0 - 0.5 x1 = 0 holds in the population
    constrained = oe.cnsreg(data=frame, y="y", x=columns, cluster="firm",
                            constraints=[{"terms": {"x0": 1, "x1": -0.5}, "value": 0}])
    seconds["cnsreg"] = time.perf_counter() - start
    start = time.perf_counter()
    one_way = oe.areg(data=frame, y="y", x=columns, absorb="firm", cluster="firm")
    seconds["areg"] = time.perf_counter() - start
    start = time.perf_counter()
    absorbed = oe.reghdfe(data=frame, y="y", x=columns, absorb=["firm", "year"], cluster="firm")
    seconds["reghdfe"] = time.perf_counter() - start
    close(coef(constrained)[1:], np.arange(1, k + 1), 2e-2)
    assert coef(constrained)[1] == pytest.approx(0.5 * coef(constrained)[2], rel=1e-12)
    close(coef(one_way)[1:], np.arange(1, k + 1), 2e-2)
    close(coef(absorbed), np.arange(1, k + 1), 2e-2)
    assert absorbed.metrics["df_absorbed"] > 0 and absorbed.extra["converged"] is True
    assert one_way.metrics["n_groups"] == frame.firm.nunique()
    assert max(seconds.values()) < 20, seconds
