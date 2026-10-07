"""Independent oracles for the iv family (ivregress, xtivreg, ivreghdfe).

Everything here is derived from the textbook / Stata "Methods and formulas"
statements with explicit dense NumPy algebra on square-root-weighted data
(n-by-n projection matrices, generalized eigenproblems by SciPy, dummy-variable
regressions) and never calls the implementation's kernels. Frequency weights
are checked against physically duplicated rows.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.stats as st
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError

X, ENDOG, INST = ["x1", "x2"], ["p1", "p2"], ["z1", "z2", "z3", "z4"]


# ---- data ------------------------------------------------------------------------------------


def make_data(seed=7, n=320, groups=37):
    rng = np.random.default_rng(seed)
    firm = np.sort(rng.integers(0, groups, size=n))          # unbalanced clusters
    frame = pd.DataFrame({
        "firm": firm, "region": rng.integers(0, 9, size=n),
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 2, size=n),
        "z1": rng.normal(size=n), "z2": rng.normal(size=n), "z3": rng.normal(size=n),
        "z4": rng.exponential(size=n), "sector": rng.choice(["k", "m", "s"], size=n),
        "t": rng.permutation(n), "fw": rng.integers(1, 5, size=n).astype(float),
        "aw": rng.uniform(0.3, 3.0, size=n),
    })
    v1, v2 = rng.normal(size=n), rng.normal(size=n)
    shock = rng.normal(size=groups)[firm]
    frame["p1"] = 0.8 * frame.z1 + 0.4 * frame.z2 - 0.3 * frame.z4 + 0.5 * frame.x1 + v1
    frame["p2"] = 0.6 * frame.z3 - 0.5 * frame.z2 + 0.2 * frame.x2 + v2 + 0.3 * shock
    noise = rng.normal(size=n) * (0.5 + frame.z1.abs()) + 0.8 * shock
    frame["y"] = (2 + 1.5 * frame.p1 - 0.7 * frame.p2 + frame.x1 - 0.5 * frame.x2
                  + 0.9 * v1 - 0.6 * v2 + noise)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def arrays(frame, x=X, endog=ENDOG, inst=INST, intercept=True, y="y"):
    cols = [np.ones(len(frame))] if intercept else []
    cols += [frame[c].to_numpy(float) for c in x]
    x1 = np.column_stack(cols) if cols else np.empty((len(frame), 0))
    return (frame[y].to_numpy(float), x1, frame[list(endog)].to_numpy(float),
            frame[list(inst)].to_numpy(float))


def coef(result):
    return np.array([c.estimate for c in result.coefficients])


def vcov(result):
    return np.asarray(result.covariance_matrix)


# ---- oracle: estimators ----------------------------------------------------------------------


def proj(a):
    """Dense orthogonal projector on the column space of a (n-by-n; small n only)."""
    return a @ np.linalg.pinv(a)


def group_meat(scores, labels):
    out = np.zeros((scores.shape[1],) * 2)
    for value in pd.unique(labels):
        total = scores[labels == value].sum(axis=0)
        out += np.outer(total, total)
    return out


def meat_of(scores, vce, *, cluster=None, time=None, lags=None, kernel="bartlett", fw=None):
    """Sum-form meat of score rows (already weighted): White, 1-2 way cluster or HAC."""
    if vce == "robust":
        return (scores * (1 if fw is None else fw)[..., None]).T @ scores if fw is not None \
            else scores.T @ scores
    if vce == "cluster":
        if len(cluster) == 1:
            return group_meat(scores, cluster[0])
        a, b = cluster
        both = np.array([f"{i}|{j}" for i, j in zip(a, b)])
        return group_meat(scores, a) + group_meat(scores, b) - group_meat(scores, both)
    order = np.argsort(time)
    s, tt = scores[order], np.asarray(time)[order]
    meat = s.T @ s
    for lag in range(1, lags + 1):
        if kernel == "bartlett":
            weight = 1 - lag / (lags + 1)
        elif kernel == "truncated":
            weight = 1.0
        else:                                                   # parzen
            ratio = lag / (lags + 1)
            weight = 1 - 6 * ratio ** 2 + 6 * ratio ** 3 if ratio <= 0.5 \
                else 2 * (1 - ratio) ** 3
        gamma = np.zeros_like(meat)
        for i in range(len(s)):                                 # pairs exactly `lag` apart
            j = np.searchsorted(tt, tt[i] - lag)
            if j < len(tt) and tt[j] == tt[i] - lag:
                gamma += np.outer(s[i], s[j])
        meat += weight * (gamma + gamma.T)
    return meat


def group_count(cluster):
    return min(len(pd.unique(c)) for c in cluster)


def oracle(y, x1, x2, z2, *, w=None, method="2sls", vce="nonrobust", small=False, cluster=None,
           time=None, lags=None, kernel="bartlett", wmatrix="robust", center=False, igmm=False,
           absorbed=0, group_factor=True):
    """Coefficients, covariance and inference of ivregress by dense algebra.

    ``w`` are analytic/sampling weights (rescaled to sum to n here). ``absorbed``
    adds partialled-out parameters to K in every small-sample factor.
    """
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float) * n / np.sum(w)
    sw = np.sqrt(w)
    xs, zs, ys = np.column_stack([x1, x2]) * sw[:, None], \
        np.column_stack([x1, z2]) * sw[:, None], y * sw
    x = np.column_stack([x1, x2])
    k, ell = xs.shape[1], zs.shape[1]
    kk = k + absorbed
    pz = proj(zs)
    mz = np.eye(n) - pz
    out = {"n": n, "k": k, "l": ell}

    def k_class(kappa):
        h = xs.T @ (np.eye(n) - kappa * mz)
        b = np.linalg.solve(h @ xs, h @ ys)
        return b, np.linalg.inv(h @ xs), h.T

    kappa = 1.0
    if method == "liml":
        m1 = np.eye(n) - proj(xs[:, :x1.shape[1]]) if x1.shape[1] else np.eye(n)
        yy = np.column_stack([ys, xs[:, x1.shape[1]:]])
        kappa = scipy.linalg.eigh(yy.T @ m1 @ yy, yy.T @ mz @ yy, eigvals_only=True)[0]
        out["kappa"] = kappa
    b, bread, score_x = k_class(kappa)
    factor = 1.0
    if vce == "cluster":
        g = group_count(cluster)
        factor = (g / (g - 1) if group_factor else 1.0) * ((n - 1) / (n - kk) if small else 1.0)
        out["df_t"] = g - 1
    else:
        factor = n / (n - kk) if small and vce != "nonrobust" else 1.0
        out["df_t"] = n - kk
    opts = {"cluster": cluster, "time": time, "lags": lags, "kernel": kernel}
    if method == "gmm":
        u1 = ys - xs @ b

        def s_of(resid, kind):
            if kind == "unadjusted":
                return (resid @ resid / n) * (zs.T @ zs) / n
            rows = zs * resid[:, None]
            if center:                                  # unweighted designs only
                rows = rows - rows.mean(axis=0)
            return meat_of(rows, kind, **opts) / n

        a = xs.T @ zs
        resid, iterations = u1, 0
        while True:
            s = s_of(resid, wmatrix)
            wm = np.linalg.inv(s)
            new = np.linalg.solve(a @ wm @ a.T, a @ wm @ zs.T @ ys)
            iterations += 1
            done = iterations > 1 and np.max(np.abs(new - b) / (np.abs(b) + 1)) < 1e-10
            b = new
            if not igmm or done:
                break
            resid = ys - xs @ b
        u = ys - xs @ b
        gbar = zs.T @ u / n
        out["j"] = n * gbar @ wm @ gbar
        out["iterations"] = iterations
        binv = np.linalg.inv(a @ wm @ a.T)
        kind = {"nonrobust": "unadjusted"}.get(vce, vce)
        if kind == wmatrix:
            # efficient GMM; `small` multiplies by N/(N-K) for the unadjusted type as well
            v = n * binv * (n / (n - kk) if small and vce == "nonrobust" else factor)
        else:
            if kind == "unadjusted":
                shat = (u @ u / ((n - kk) if small else n)) * (zs.T @ zs) / n
            else:
                shat = meat_of(zs * u[:, None], kind, **opts) / n * factor
            v = n * binv @ a @ wm @ shat @ wm @ a.T @ binv
    else:
        u = ys - xs @ b
        if vce == "nonrobust":
            v = bread * (u @ u) / ((n - kk) if small else n)
        else:
            v = bread @ meat_of(score_x * u[:, None], vce, **opts) @ bread * factor
    out.update({"b": b, "v": v, "u": (y - x @ b), "rss": float(u @ u), "bread": bread})
    return out


def inference(b, v, df=None, alpha=0.05):
    se = np.sqrt(np.diag(v))
    stat = b / se
    if df is None:
        p, crit = 2 * st.norm.sf(np.abs(stat)), st.norm.ppf(1 - alpha / 2)
    else:
        p, crit = 2 * st.t.sf(np.abs(stat), df), st.t.ppf(1 - alpha / 2, df)
    return se, stat, p, b - crit * se, b + crit * se


def check(result, ref, *, small, rtol=1e-7, vtol=1e-7):
    """Coefficients, covariance, tests of coefficients and the model test."""
    b, v = ref["b"], ref["v"]
    assert_allclose(coef(result), b, rtol=rtol, atol=1e-10)
    assert_allclose(vcov(result), v, rtol=vtol, atol=1e-12 * np.abs(v).max())
    df = ref["df_t"] if small else None
    se, stat, p, low, high = inference(coef(result), v, df)
    rows = result.coefficients
    assert_allclose([c.std_error for c in rows], se, rtol=vtol)
    assert_allclose([c.statistic for c in rows], stat, rtol=1e-6)
    assert_allclose([c.p_value for c in rows], p, rtol=1e-6, atol=1e-290)
    assert_allclose([c.ci_low for c in rows], low, rtol=1e-6, atol=1e-9)
    assert_allclose([c.ci_high for c in rows], high, rtol=1e-6, atol=1e-9)
    assert result.inference["use_t"] is small
    assert result.inference["df_inference"] == (df if small else None)
    slopes = [i for i, c in enumerate(rows) if c.term != "Intercept"]
    wald = b[slopes] @ np.linalg.solve(v[np.ix_(slopes, slopes)], b[slopes])
    test = result.tests["model"]
    assert test["df"] == len(slopes)
    if small:
        assert test["distribution"] == "F" and test["df2"] == df
        assert test["statistic"] == pytest.approx(wald / len(slopes), rel=1e-6)
        assert test["p_value"] == pytest.approx(
            st.f.sf(wald / len(slopes), len(slopes), df), rel=1e-6, abs=1e-290)
    else:
        assert test["distribution"] == "chi2"
        assert test["statistic"] == pytest.approx(wald, rel=1e-6)
        assert test["p_value"] == pytest.approx(st.chi2.sf(wald, len(slopes)), rel=1e-6,
                                                abs=1e-290)


def iv(frame, **options):
    options = {"x": X, "endog": ENDOG, "instruments": INST, **options}
    return oe.ivregress(data=frame, y="y", **options)


# ---- ivregress: estimators x covariances x small ---------------------------------------------


@pytest.mark.parametrize("method,vce,small", list(itertools.product(
    ["2sls", "liml"], ["nonrobust", "robust", "cluster", "twoway", "hac"], [False, True])))
def test_k_class_estimators_under_every_covariance(data, method, vce, small):
    y, x1, x2, z2 = arrays(data)
    options, ref_options = {}, {}
    if vce == "cluster":
        options["cluster"] = "firm"
        ref_options["cluster"] = [data.firm.to_numpy()]
    elif vce == "twoway":
        options["cluster"] = ["firm", "region"]
        ref_options["cluster"] = [data.firm.to_numpy(), data.region.to_numpy()]
    elif vce == "hac":
        options.update(covariance="hac", lags=3, time="t")
        ref_options.update(time=data.t.to_numpy(), lags=3)
    else:
        options["covariance"] = vce
    kind = "cluster" if vce == "twoway" else vce
    result = iv(data, method=method, small=small, **options)
    ref = oracle(y, x1, x2, z2, method=method, vce=kind, small=small, **ref_options)
    check(result, ref, small=small)
    n, k = len(y), 5
    tss = ((y - y.mean()) ** 2).sum()
    rss = ref["u"] @ ref["u"]
    assert result.metrics["r_squared"] == pytest.approx(1 - rss / tss, rel=1e-9)
    assert result.metrics["adjusted_r_squared"] == pytest.approx(
        1 - (rss / tss) * (n - 1) / (n - k), rel=1e-9)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(rss / (n - k if small else n)),
                                                   rel=1e-9)
    assert result.metrics["df_model"] == k - 1 and result.metrics["df_resid"] == n - k
    assert result.metrics["n_instruments"] == 4 and result.metrics["n_endogenous"] == 2
    if method == "liml":
        assert result.metrics["kappa"] == pytest.approx(ref["kappa"], rel=1e-9)
        assert list(result.metrics)[:4] == ["r_squared", "adjusted_r_squared", "rmse", "kappa"]
    assert result.nobs == n


# ---- ivregress: GMM ---------------------------------------------------------------------------


GMM_CASES = [
    # wmatrix, vce, small, igmm, center
    ("robust", "robust", False, False, False),
    ("robust", "robust", True, False, False),
    ("robust", "robust", False, True, False),
    ("robust", "robust", False, False, True),
    ("robust", "nonrobust", False, False, False),
    ("robust", "nonrobust", True, False, False),
    ("robust", "hac", False, False, False),
    ("unadjusted", "nonrobust", False, False, False),
    ("unadjusted", "robust", False, False, False),
    ("unadjusted", "robust", True, False, False),
    ("unadjusted", "cluster", False, False, False),
    ("cluster", "cluster", False, False, False),
    ("cluster", "cluster", True, False, False),
    ("cluster", "cluster", False, True, True),
    ("hac", "hac", False, False, False),
    ("hac", "hac", True, True, False),
    ("hac", "robust", False, False, False),
]


@pytest.mark.parametrize("wmatrix,vce,small,igmm,center", GMM_CASES)
def test_gmm_weight_matrices_and_covariances(data, wmatrix, vce, small, igmm, center):
    y, x1, x2, z2 = arrays(data)
    options = {"wmatrix": wmatrix, "small": small, "igmm": igmm, "center": center}
    ref_options = {}
    if "cluster" in (wmatrix, vce):
        options["cluster"] = "firm"
        ref_options["cluster"] = [data.firm.to_numpy()]
    if "hac" in (wmatrix, vce):
        options.update(lags=2, time="t", kernel="parzen")
        ref_options.update(time=data.t.to_numpy(), lags=2, kernel="parzen")
    if vce != "cluster":
        options["covariance"] = vce
    result = iv(data, method="gmm", **options)
    ref = oracle(y, x1, x2, z2, method="gmm", vce=vce, small=small, wmatrix=wmatrix, igmm=igmm,
                 center=center, **ref_options)
    check(result, ref, small=small, rtol=1e-7 if not igmm else 1e-6,
          vtol=1e-7 if not igmm else 1e-6)
    j = result.tests["hansen_j"]
    assert j["df"] == 2 and j["distribution"] == "chi2"
    assert j["statistic"] == pytest.approx(ref["j"], rel=1e-6)
    assert j["p_value"] == pytest.approx(st.chi2.sf(ref["j"], 2), rel=1e-6)
    assert result.metrics["j"] == pytest.approx(ref["j"], rel=1e-6)
    assert result.extra["gmm"]["wmatrix"] == wmatrix
    if igmm:
        assert result.extra["gmm"]["iterations"] == ref["iterations"] > 2
        assert result.extra["gmm"]["converged"] is True
    else:
        assert result.extra["gmm"]["iterations"] == 1


def test_gmm_defaults_follow_stata(data):
    # ivregress gmm: wmatrix(robust) and a VCE of the weight-matrix type.
    result = iv(data, method="gmm")
    assert result.spec.covariance == "robust" and result.extra["gmm"]["wmatrix"] == "robust"
    clustered = iv(data, method="gmm", cluster="firm")
    assert clustered.spec.covariance == "cluster"
    assert clustered.extra["gmm"]["wmatrix"] == "cluster"
    unadjusted = iv(data, method="gmm", wmatrix="unadjusted")
    assert unadjusted.spec.covariance == "nonrobust"
    hac = iv(data, method="gmm", wmatrix="hac", lags=1, time="t")
    assert hac.spec.covariance == "hac" and hac.extra["gmm"]["wmatrix"] == "hac"


# ---- oracle: diagnostics ----------------------------------------------------------------------


def diagnostics(y, x1, x2, z2, *, w=None, vce="nonrobust", cluster=None, time=None, lags=None,
                kernel="bartlett", absorbed=0, constant=True, tss=None):
    """estat firststage / overid / endogenous and ivreg2's weak-identification statistics.

    Textbook definitions on square-root-weighted data; regress-style factors for the
    auxiliary robust regressions; ``absorbed`` enters every residual degrees of freedom.
    """
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float) * n / np.sum(w)
    sw = np.sqrt(w)
    x1s, x2s, z2s, ys = x1 * sw[:, None], x2 * sw[:, None], z2 * sw[:, None], y * sw
    xs, zs = np.column_stack([x1s, x2s]), np.column_stack([x1s, z2s])
    k1, q, l2 = x1.shape[1], x2.shape[1], z2.shape[1]
    k, ell = k1 + q, k1 + l2
    eye = np.eye(n)
    p1 = proj(x1s) if k1 else np.zeros((n, n))
    pz = proj(zs)
    m1, mz = eye - p1, eye - pz
    opts = {"cluster": cluster, "time": time, "lags": lags, "kernel": kernel}
    df1 = n - ell - absorbed
    out = {"df_first": df1}

    def regress_factor(params):
        if vce == "cluster":
            g = group_count(cluster)
            return g / (g - 1) * (n - 1) / (n - params - absorbed), g - 1
        return n / (n - params - absorbed), n - params - absorbed

    # first stage -----------------------------------------------------------------------------
    xhat = pz @ xs
    shea_iv, shea_ols = np.linalg.inv(xhat.T @ xhat), np.linalg.inv(xs.T @ xs)
    rows = []
    for j in range(q):
        target = x2s[:, j]
        e = mz @ target
        rss, rss1 = e @ e, target @ m1 @ target
        if tss is None:
            raw = x2[:, j]
            total = (w * (raw - (w * raw).sum() / w.sum()) ** 2).sum() if constant \
                else (w * raw ** 2).sum()
        else:
            total = tss[j]
        r2 = 1 - rss / total
        row = {"r_squared": r2,
               "adjusted_r_squared": 1 - (1 - r2) * (n - int(constant) - absorbed) / df1,
               "partial_r_squared": 1 - rss / rss1,
               "shea_partial_r_squared": shea_ols[k1 + j, k1 + j] / shea_iv[k1 + j, k1 + j]}
        if vce == "nonrobust":
            f = ((rss1 - rss) / l2) / (rss / df1)
            row.update(f_statistic=f, df=l2, df2=df1, p_value=st.f.sf(f, l2, df1))
        else:
            pi = np.linalg.pinv(zs) @ target
            bread = np.linalg.inv(zs.T @ zs)
            factor, df2 = regress_factor(ell)
            v = bread @ meat_of(zs * e[:, None], vce, **opts) @ bread * factor
            f = pi[k1:] @ np.linalg.solve(v[k1:, k1:], pi[k1:]) / l2
            row.update(f_statistic=f, df=l2, df2=df2, p_value=st.f.sf(f, l2, df2))
        rows.append(row)
    out["first_stage"] = rows

    # weak identification ------------------------------------------------------------------------
    sigma_vv = x2s.T @ mz @ x2s / df1
    root = np.linalg.inv(scipy.linalg.sqrtm(sigma_vv).real)
    zp = m1 @ z2s
    gmat = root @ x2s.T @ zp @ np.linalg.solve(zp.T @ zp, zp.T @ x2s) @ root / l2
    out["cragg_donald"] = np.linalg.eigvalsh((gmat + gmat.T) / 2)[0]
    if vce != "nonrobust":
        # Kleibergen-Paap (2006) rk Wald statistic for rank q-1 of the reduced form.
        yp, v = m1 @ x2s, mz @ x2s
        ry, rz = np.linalg.cholesky(yp.T @ yp).T, np.linalg.cholesky(zp.T @ zp).T
        theta = np.linalg.solve(rz.T, zp.T @ yp) @ np.linalg.inv(ry)           # [l2, q]
        u_, _, vt = np.linalg.svd(theta)
        v_ = vt.T
        r = q - 1
        u12, u22 = u_[:r, r:], u_[r:, r:]
        v12, v22 = v_[:r, r:], v_[r:, r:]
        a_perp = np.vstack([u12, u22]) @ np.linalg.inv(u22) @ scipy.linalg.sqrtm(
            u22 @ u22.T).real
        b_perp = scipy.linalg.sqrtm(v22 @ v22.T).real @ np.linalg.inv(v22.T) @ np.hstack(
            [v12.T, v22.T])
        lam = (a_perp.T @ theta @ b_perp.T).flatten(order="F")
        # rows of vec(Z'V): v_i (x) z_i, mapped to vec(theta) by (Ry^-T (x) Rz^-T)
        cross = np.einsum("ij,ik->ijk", v, zp).reshape(n, q * l2)              # v (x) z
        cross = cross @ np.kron(np.linalg.inv(ry), np.linalg.inv(rz))
        cross = cross @ np.kron(b_perp, a_perp.T).T
        omega = meat_of(cross, vce, **opts)
        rk = lam @ np.linalg.solve(omega, lam)
        out["kp_rk"] = rk
        if vce == "cluster":
            g = group_count(cluster)
            out["kp_f"] = rk / (n - 1) * df1 * (g - 1) / g / l2
        else:
            out["kp_f"] = rk / n * df1 / l2

    # 2SLS residuals ------------------------------------------------------------------------
    b = np.linalg.solve(xhat.T @ xs, xhat.T @ ys)
    u = ys - xs @ b
    over = l2 - q
    if over:
        sargan = n * (u @ pz @ u) / (u @ u)
        out["sargan"] = sargan
        out["basmann"] = (u @ pz @ u) / (u @ mz @ u / df1)
        if vce != "nonrobust":
            # Wooldridge (1995): residuals of `over` extra instruments on Xhat, times u.
            extra = z2s[:, :over]
            kres = extra - xhat @ np.linalg.lstsq(xhat, extra, rcond=None)[0]
            mom = kres * u[:, None]
            total = mom.sum(axis=0)
            out["overid_score"] = total @ np.linalg.solve(meat_of(mom, vce, **opts), total)
            if vce == "robust":
                ones = np.ones(n)
                fit = mom @ np.linalg.lstsq(mom, ones, rcond=None)[0]
                assert np.isclose(out["overid_score"], n - ((ones - fit) ** 2).sum())
        m_1 = np.eye(n) - p1
        yy = np.column_stack([ys, x2s])
        kappa = scipy.linalg.eigh(yy.T @ m_1 @ yy, yy.T @ mz @ yy, eigvals_only=True)[0]
        out["anderson_rubin"] = n * np.log(kappa)
        out["basmann_f"] = (kappa - 1) * df1 / over

    # endogeneity ---------------------------------------------------------------------------
    b_ols = np.linalg.lstsq(xs, ys, rcond=None)[0]
    ue = ys - xs @ b_ols
    pzy = proj(np.column_stack([zs, x2s]))
    quad = ue @ pzy @ ue - u @ pz @ u                 # Stata: u_e'P_ZY u_e - u_c'P_Z u_c
    df2 = n - k - q - absorbed
    out["durbin"] = quad / (ue @ ue / n)
    out["wu_hausman"] = (quad / q) / ((ue @ ue - quad) / df2)
    out["wu_hausman_df2"] = df2
    if vce != "nonrobust":
        vhat = mz @ x2s
        rres = vhat - xs @ np.linalg.lstsq(xs, vhat, rcond=None)[0]
        mom = rres * ue[:, None]
        total = mom.sum(axis=0)
        out["endog_score"] = total @ np.linalg.solve(meat_of(mom, vce, **opts), total)
        aug = np.column_stack([xs, vhat])
        coef_aug = np.linalg.lstsq(aug, ys, rcond=None)[0]
        resid = ys - aug @ coef_aug
        bread = np.linalg.inv(aug.T @ aug)
        factor, df_aug = regress_factor(k + q)
        vaug = bread @ meat_of(aug * resid[:, None], vce, **opts) @ bread * factor
        gamma = coef_aug[k:]
        out["endog_regression_f"] = gamma @ np.linalg.solve(vaug[k:, k:], gamma) / q
        out["endog_regression_df2"] = df_aug
    return out


def check_first_stage(result, ref, terms=ENDOG):
    rows = result.extra["first_stage"]
    assert [row["term"] for row in rows] == list(terms)
    for row, expected in zip(rows, ref["first_stage"]):
        for name, value in expected.items():
            assert row[name] == pytest.approx(value, rel=2e-7, abs=1e-290), name


def check_entry(entry, statistic, df, df2=None, *, rel=1e-6):
    assert entry["statistic"] == pytest.approx(statistic, rel=rel)
    assert entry["df"] == df
    if df2 is None:
        assert entry["distribution"] == "chi2"
        assert entry["p_value"] == pytest.approx(st.chi2.sf(statistic, df), rel=1e-6, abs=1e-290)
    else:
        assert entry["distribution"] == "F" and entry["df2"] == df2
        assert entry["p_value"] == pytest.approx(st.f.sf(statistic, df, df2), rel=1e-6,
                                                 abs=1e-290)


DIAG_CASES = {
    "nonrobust": ({}, {}),
    "robust": ({"covariance": "robust"}, {}),
    "cluster": ({"cluster": "firm"}, {"cluster": "firm"}),
    "hac": ({"covariance": "hac", "lags": 2, "time": "t"}, {"time": "t", "lags": 2}),
}


def diag_reference(frame, vce, ref_options, **kwargs):
    extra = {}
    if "cluster" in ref_options:
        extra["cluster"] = [frame[ref_options["cluster"]].to_numpy()]
    if "time" in ref_options:
        extra.update(time=frame[ref_options["time"]].to_numpy(), lags=ref_options["lags"])
    return diagnostics(*arrays(frame), vce=vce, **extra, **kwargs)


@pytest.mark.parametrize("vce", list(DIAG_CASES))
@pytest.mark.parametrize("small", [False, True])
def test_2sls_postestimation_statistics(data, vce, small):
    options, ref_options = DIAG_CASES[vce]
    result = iv(data, small=small, **options)
    ref = diag_reference(data, vce, ref_options)
    n = len(data)
    check_first_stage(result, ref)
    tests = result.tests
    cd = tests["cragg_donald"]
    assert cd["statistic"] == pytest.approx(ref["cragg_donald"], rel=1e-7)
    assert (cd["df"], cd["df2"]) == (4, n - 7)
    if vce == "nonrobust":
        check_entry(tests["overid_sargan"], ref["sargan"], 2)
        check_entry(tests["overid_basmann"], ref["basmann"], 2)
        check_entry(tests["endog_durbin"], ref["durbin"], 2)
        check_entry(tests["endog_wu_hausman"], ref["wu_hausman"], 2, n - 5 - 2)
        assert "kleibergen_paap_rk_f" not in tests and "overid_score" not in tests
    else:
        check_entry(tests["overid_score"], ref["overid_score"], 2)
        check_entry(tests["endog_robust_score"], ref["endog_score"], 2)
        check_entry(tests["endog_robust_regression"], ref["endog_regression_f"], 2,
                    ref["endog_regression_df2"])
        kp = tests["kleibergen_paap_rk_f"]
        assert kp["statistic"] == pytest.approx(ref["kp_f"], rel=1e-6)
        assert kp["rk_wald_chi2"] == pytest.approx(ref["kp_rk"], rel=1e-6)
        assert kp["rk_df"] == 3
        assert "overid_sargan" not in tests and "endog_durbin" not in tests


def test_liml_overidentification_tests(data):
    result = iv(data, method="liml")
    ref = diag_reference(data, "nonrobust", {})
    n = len(data)
    check_entry(result.tests["anderson_rubin"], ref["anderson_rubin"], 2)
    check_entry(result.tests["basmann_f"], ref["basmann_f"], 2, n - 7)
    check_first_stage(result, ref)
    assert result.tests["cragg_donald"]["statistic"] == pytest.approx(ref["cragg_donald"],
                                                                      rel=1e-7)


def test_single_endogenous_kleibergen_paap_equals_the_robust_first_stage_f(data):
    # With one endogenous regressor the rk Wald F is the robust (cluster) first-stage F.
    for options in ({"covariance": "robust"}, {"cluster": "firm"}):
        result = oe.ivregress(data=data, y="y", x=X + ["p2"], endog=["p1"], instruments=INST,
                              **options)
        first = result.extra["first_stage"][0]
        assert result.tests["kleibergen_paap_rk_f"]["statistic"] == pytest.approx(
            first["f_statistic"], rel=1e-8)
        # nonrobust: Cragg-Donald is the classical first-stage F
        plain = oe.ivregress(data=data, y="y", x=X + ["p2"], endog=["p1"], instruments=INST)
        assert plain.tests["cragg_donald"]["statistic"] == pytest.approx(
            plain.extra["first_stage"][0]["f_statistic"], rel=1e-9)


# ---- weights -----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_analytic_weights(data, method, vce, small):
    if method == "gmm" and vce == "nonrobust":
        options = {"wmatrix": "unadjusted"}
    else:
        options = {}
    y, x1, x2, z2 = arrays(data)
    ref_options = {"cluster": [data.firm.to_numpy()]} if vce == "cluster" else {}
    call = {"cluster": "firm"} if vce == "cluster" else {"covariance": vce}
    result = iv(data, method=method, small=small, weights="aw", weight_type="aweight",
                **call, **options)
    wmatrix = {"nonrobust": "unadjusted"}.get(vce, vce)
    ref = oracle(y, x1, x2, z2, w=data.aw.to_numpy(), method=method, vce=vce, small=small,
                 wmatrix=wmatrix, **ref_options)
    check(result, ref, small=small)
    w = data.aw.to_numpy() * len(y) / data.aw.sum()
    tss = (w * (y - (w * y).sum() / w.sum()) ** 2).sum()
    rss = (w * ref["u"] ** 2).sum()
    assert result.metrics["r_squared"] == pytest.approx(1 - rss / tss, rel=1e-9)
    assert result.nobs == len(y)
    # scaling the weights changes nothing
    again = iv(data.assign(aw=data.aw * 37.5), method=method, small=small, weights="aw",
               weight_type="aweight", **call, **options)
    assert_allclose(coef(again), coef(result), rtol=1e-10)
    assert_allclose(vcov(again), vcov(result), rtol=1e-9)


@pytest.mark.parametrize("vce", ["robust", "cluster"])
def test_sampling_weights_equal_analytic_weights_with_a_robust_covariance(data, vce):
    call = {"cluster": "firm"} if vce == "cluster" else {"covariance": vce}
    for method in ("2sls", "liml", "gmm"):
        pw = iv(data, method=method, weights="aw", weight_type="pweight", **call)
        aw = iv(data, method=method, weights="aw", weight_type="aweight", **call)
        assert_allclose(coef(pw), coef(aw), rtol=1e-12)
        assert_allclose(vcov(pw), vcov(aw), rtol=1e-12)
    assert iv(data, weights="aw", weight_type="pweight").spec.covariance == "robust"
    with pytest.raises(AnalysisError) as error:
        iv(data, weights="aw", weight_type="pweight", covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"


def duplicated(frame, column="fw"):
    return frame.loc[frame.index.repeat(frame[column].astype(int))].reset_index(drop=True)


FWEIGHT_CASES = [
    ("2sls", {}), ("2sls", {"covariance": "robust"}), ("2sls", {"cluster": "firm"}),
    ("2sls", {"cluster": ["firm", "region"]}), ("2sls", {"small": True}),
    ("2sls", {"covariance": "robust", "small": True}), ("2sls", {"cluster": "firm", "small": True}),
    ("liml", {}), ("liml", {"covariance": "robust"}), ("liml", {"cluster": "firm"}),
    ("gmm", {}), ("gmm", {"cluster": "firm"}), ("gmm", {"wmatrix": "unadjusted"}),
    ("gmm", {"igmm": True}), ("gmm", {"center": True}), ("gmm", {"covariance": "nonrobust"}),
    ("gmm", {"wmatrix": "unadjusted", "covariance": "robust", "small": True}),
]


def same_statistics(a, b, rel=1e-7):
    """Every numeric entry of tests and extra['first_stage'] agrees."""
    assert set(a.tests) == set(b.tests)
    for name in a.tests:
        for key, value in a.tests[name].items():
            other = b.tests[name][key]
            if isinstance(value, float):
                assert other == pytest.approx(value, rel=rel, abs=1e-12), (name, key)
            else:
                assert other == value, (name, key)
    for row_a, row_b in zip(a.extra.get("first_stage", []), b.extra.get("first_stage", [])):
        for key, value in row_a.items():
            if isinstance(value, float):
                assert row_b[key] == pytest.approx(value, rel=rel, abs=1e-12), key
            else:
                assert row_b[key] == value, key
    for key, value in a.metrics.items():
        if isinstance(value, float):
            assert b.metrics[key] == pytest.approx(value, rel=rel, abs=1e-12), key


@pytest.mark.parametrize("method,options", FWEIGHT_CASES)
def test_frequency_weights_equal_duplicated_rows(data, method, options):
    weighted = iv(data, method=method, weights="fw", weight_type="fweight", **options)
    expanded = iv(duplicated(data), method=method, **options)
    assert weighted.nobs == expanded.nobs == int(data.fw.sum())
    assert_allclose(coef(weighted), coef(expanded), rtol=1e-9)
    assert_allclose(vcov(weighted), vcov(expanded), rtol=1e-8)
    assert_allclose([c.p_value for c in weighted.coefficients],
                    [c.p_value for c in expanded.coefficients], rtol=1e-6, atol=1e-290)
    same_statistics(weighted, expanded)


@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
def test_weighted_postestimation_statistics(data, vce):
    options, ref_options = DIAG_CASES[vce]
    result = iv(data, weights="aw", weight_type="aweight", **options)
    ref = diag_reference(data, vce, ref_options, w=data.aw.to_numpy())
    check_first_stage(result, ref)
    tests = result.tests
    assert tests["cragg_donald"]["statistic"] == pytest.approx(ref["cragg_donald"], rel=1e-7)
    if vce == "nonrobust":
        check_entry(tests["overid_sargan"], ref["sargan"], 2)
        check_entry(tests["overid_basmann"], ref["basmann"], 2)
        check_entry(tests["endog_durbin"], ref["durbin"], 2)
        check_entry(tests["endog_wu_hausman"], ref["wu_hausman"], 2, len(data) - 7)
    else:
        check_entry(tests["overid_score"], ref["overid_score"], 2)
        check_entry(tests["endog_robust_score"], ref["endog_score"], 2)
        check_entry(tests["endog_robust_regression"], ref["endog_regression_f"], 2,
                    ref["endog_regression_df2"])
        assert tests["kleibergen_paap_rk_f"]["statistic"] == pytest.approx(ref["kp_f"], rel=1e-6)
    liml = iv(data, method="liml", weights="aw", weight_type="aweight", **options)
    check_entry(liml.tests["anderson_rubin"], ref["anderson_rubin"], 2)
    check_entry(liml.tests["basmann_f"], ref["basmann_f"], 2, len(data) - 7)


# ---- designs ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
def test_no_exogenous_regressors_and_no_constant(data, method):
    y, _, x2, z2 = arrays(data)
    empty = np.empty((len(y), 0))
    for small in (False, True):
        result = oe.ivregress(data=data, y="y", endog=ENDOG, instruments=INST, method=method,
                              intercept=False, covariance="robust", small=small)
        ref = oracle(y, empty, x2, z2, method=method, vce="robust", small=small)
        check(result, ref, small=small)
        assert [c.term for c in result.coefficients] == ENDOG
        # uncentered R-squared without a constant
        assert result.metrics["r_squared"] == pytest.approx(
            1 - ref["u"] @ ref["u"] / (y @ y), rel=1e-9)
        assert result.metrics["adjusted_r_squared"] == pytest.approx(
            1 - (ref["u"] @ ref["u"] / (y @ y)) * len(y) / (len(y) - 2), rel=1e-9)
        assert result.tests["model"]["df"] == 2
    # constant only
    y, x1, x2, z2 = arrays(data, x=[])
    result = oe.ivregress(data=data, y="y", endog=ENDOG, instruments=INST, method=method)
    kind = "robust" if method == "gmm" else "nonrobust"
    check(result, oracle(y, x1, x2, z2, method=method, vce=kind), small=False)
    ref = diagnostics(y, x1, x2, z2, vce=kind)
    check_first_stage(result, ref)


def test_exact_identification_equivalences(data):
    exact = {"endog": ENDOG, "instruments": ["z1", "z3"], "x": X}
    tsls = oe.ivregress(data=data, y="y", **exact)
    liml = oe.ivregress(data=data, y="y", method="liml", **exact)
    gmm_u = oe.ivregress(data=data, y="y", method="gmm", wmatrix="unadjusted", **exact)
    gmm_r = oe.ivregress(data=data, y="y", method="gmm", **exact)
    robust = oe.ivregress(data=data, y="y", covariance="robust", **exact)
    y, x1, x2, z2 = arrays(data, inst=["z1", "z3"])
    # the exactly identified IV estimator (Z'X)^-1 Z'y
    xmat, zmat = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    b = np.linalg.solve(zmat.T @ xmat, zmat.T @ y)
    for result in (tsls, liml, gmm_u, gmm_r, robust):
        assert_allclose(coef(result), b, rtol=1e-9)
    assert liml.metrics["kappa"] == pytest.approx(1.0, abs=1e-9)
    assert_allclose(vcov(liml), vcov(tsls), rtol=1e-9)
    assert_allclose(vcov(gmm_u), vcov(tsls), rtol=1e-9)
    assert_allclose(vcov(gmm_r), vcov(robust), rtol=1e-8)        # efficient = sandwich here
    for result, name in ((tsls, "overid_sargan"), (liml, "anderson_rubin"),
                         (gmm_r, "hansen_j"), (robust, "overid_score")):
        entry = result.tests[name]
        assert entry["statistic"] is None and "exactly identified" in entry["note"]
    ref = diagnostics(y, x1, x2, z2)
    check_first_stage(tsls, ref)
    check_entry(tsls.tests["endog_durbin"], ref["durbin"], 2)
    check_entry(tsls.tests["endog_wu_hausman"], ref["wu_hausman"], 2, len(y) - 7)
    assert tsls.tests["cragg_donald"]["statistic"] == pytest.approx(ref["cragg_donald"],
                                                                    rel=1e-7)


def test_instruments_that_are_the_regressors_reproduce_ols(data):
    frame = data.assign(c1=data.p1, c2=data.p2)
    y, x1, x2, _ = arrays(data)
    xmat = np.column_stack([x1, x2])
    b = np.linalg.lstsq(xmat, y, rcond=None)[0]
    u = y - xmat @ b
    n, k = xmat.shape
    xtx_inv = np.linalg.inv(xmat.T @ xmat)
    for method in ("2sls", "liml", "gmm"):
        extra = {"wmatrix": "unadjusted"} if method == "gmm" else {}
        result = oe.ivregress(data=frame, y="y", x=X, endog=ENDOG, instruments=["c1", "c2"],
                              method=method, small=True, **extra)
        assert_allclose(coef(result), b, rtol=1e-9)
        assert_allclose(vcov(result), xtx_inv * (u @ u) / (n - k), rtol=1e-8)
        assert result.inference["df_inference"] == n - k


def test_categorical_exogenous_regressors(data):
    result = oe.ivregress(data=data, y="y", x=["x1", "sector"], categorical=["sector"],
                          endog=ENDOG, instruments=INST, covariance="robust")
    dummies = np.column_stack([(data.sector == "m").to_numpy(float),
                               (data.sector == "s").to_numpy(float)])
    y, _, x2, z2 = arrays(data)
    x1 = np.column_stack([np.ones(len(y)), data.x1.to_numpy(), dummies])
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "sector[m]", "sector[s]", "p1", "p2"]
    check(result, oracle(y, x1, x2, z2, vce="robust"), small=False)
    check_first_stage(result, diagnostics(y, x1, x2, z2, vce="robust"))
    with pytest.raises(AnalysisError) as error:
        oe.ivregress(data=data, y="y", x=["x1"], categorical=["z1"], endog=ENDOG,
                     instruments=INST)
    assert error.value.code == "invalid_spec"


def test_collinear_columns_are_omitted_like_stata(data):
    frame = data.assign(x3=2 * data.x1 - data.x2, z5=data.z1 + data.z2 - 3 * data.x1,
                        z6=data.z4, p3=data.p1 - 2 * data.x2)
    result = oe.ivregress(data=frame, y="y", x=["x1", "x2", "x3"], endog=["p1", "p2", "p3"],
                          instruments=["z1", "z2", "z5", "z3", "z4", "z6"], small=True)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", "p1", "p2"]
    assert result.provenance["omitted_terms"] == ["x3", "p3"]
    assert result.extra["instruments"] == ["z1", "z2", "z3", "z4"]
    assert result.extra["omitted_instruments"] == ["z5", "z6"]
    assert any("x3" in w for w in result.warnings) and any("z5" in w for w in result.warnings)
    check(result, oracle(*arrays(data), small=True), small=True)
    assert result.metrics["n_instruments"] == 4 and result.metrics["df_resid"] == len(data) - 5
    # the order condition is checked after the screen
    with pytest.raises(AnalysisError) as error:
        oe.ivregress(data=frame, y="y", x=["x1"], endog=["p1", "p2"],
                     instruments=["z4", "z6"])
    assert error.value.code == "underidentified"
    with pytest.raises(AnalysisError) as error:
        oe.ivregress(data=frame.assign(p9=frame.x1 * 3), y="y", x=["x1"], endog=["p9"],
                     instruments=["z4", "z6"])
    assert error.value.code == "no_endogenous_regressors"


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
def test_row_order_and_units_do_not_matter(data, method):
    base = iv(data, method=method, cluster="firm")
    shuffled = iv(data.sample(frac=1.0, random_state=3).reset_index(drop=True), method=method,
                  cluster="firm")
    assert_allclose(coef(shuffled), coef(base), rtol=1e-9)
    assert_allclose(vcov(shuffled), vcov(base), rtol=1e-8)
    same_statistics(base, shuffled, rel=1e-6)
    # rescaling: y*1e6, x1*1e-6, p1*1e5, z1*1e-7, z4*1e8
    scaled = data.assign(y=data.y * 1e6, x1=data.x1 * 1e-6, p1=data.p1 * 1e5, z1=data.z1 * 1e-7,
                         z4=data.z4 * 1e8)
    result = iv(scaled, method=method, cluster="firm")
    scale = 1e6 / np.array([1.0, 1e-6, 1.0, 1e5, 1.0])
    assert_allclose(coef(result), coef(base) * scale, rtol=1e-7)
    assert_allclose(vcov(result), vcov(base) * np.outer(scale, scale), rtol=1e-6)
    assert_allclose([c.statistic for c in result.coefficients],
                    [c.statistic for c in base.coefficients], rtol=1e-6)
    for name, entry in base.tests.items():
        if isinstance(entry.get("statistic"), float):
            assert result.tests[name]["statistic"] == pytest.approx(entry["statistic"],
                                                                    rel=1e-5), name
    assert result.metrics["r_squared"] == pytest.approx(base.metrics["r_squared"], rel=1e-9)


def test_missing_values_are_dropped_only_on_request(data):
    frame = data.copy()
    frame.loc[[3, 50, 51], "z2"] = np.nan
    frame.loc[[7, 120], "p1"] = np.nan
    frame.loc[200, "y"] = np.nan
    frame.loc[201, "firm"] = np.nan
    with pytest.raises(AnalysisError) as error:
        iv(frame)
    assert error.value.code == "missing_values"
    result = iv(frame, missing="drop", covariance="robust")
    kept = frame.dropna(subset=["y", *X, *ENDOG, *INST])
    assert result.nobs == len(kept) == 314 and result.dropped_rows == 6
    check(result, oracle(*arrays(kept), vce="robust"), small=False)
    assert result.sample_positions == kept.index.tolist()
    clustered = iv(frame, missing="drop", cluster="firm")
    kept = kept.dropna(subset=["firm"])
    assert clustered.nobs == len(kept) == 313
    check(clustered, oracle(*arrays(kept), vce="cluster", cluster=[kept.firm.to_numpy()]),
          small=False)


def test_tiny_samples(data):
    tiny = make_data(seed=11, n=9, groups=4)
    y, x1, x2, z2 = arrays(tiny, x=["x1"], endog=["p1"], inst=["z1", "z2"])
    result = oe.ivregress(data=tiny, y="y", x=["x1"], endog=["p1"], instruments=["z1", "z2"],
                          small=True)
    check(result, oracle(y, x1, x2, z2, small=True), small=True)
    assert result.metrics["df_resid"] == 6
    ref = diagnostics(y, x1, x2, z2)
    check_first_stage(result, ref, terms=["p1"])
    check_entry(result.tests["overid_sargan"], ref["sargan"], 1)
    check_entry(result.tests["endog_wu_hausman"], ref["wu_hausman"], 1, 9 - 3 - 1)
    # n <= K: no residual degrees of freedom
    for rows in (1, 2, 3, 4):
        with pytest.raises(AnalysisError) as error:
            oe.ivregress(data=tiny.head(rows), y="y", x=["x1"], endog=["p1"],
                         instruments=["z1", "z2"])
        assert error.value.code in {"insufficient_observations", "underidentified",
                                    "no_endogenous_regressors"}, rows


def test_hac_within_panels_and_with_gaps():
    rng = np.random.default_rng(5)
    units, periods = 12, 15
    frame = pd.DataFrame({"id": np.repeat(np.arange(units), periods),
                          "t": np.tile(np.arange(periods), units)})
    frame = frame[~((frame.t == 6) & (frame.id % 3 == 0))].reset_index(drop=True)  # gaps
    n = len(frame)
    frame["z1"], frame["z2"] = rng.normal(size=n), rng.normal(size=n)
    v = rng.normal(size=n)
    frame["p"] = frame.z1 - 0.5 * frame.z2 + v
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal()
    frame["y"] = 1 + frame.p + 0.7 * v + e
    result = oe.ivregress(data=frame.sample(frac=1, random_state=1), y="y", endog=["p"],
                          instruments=["z1", "z2"], covariance="hac", lags=2, panel="id",
                          time="t", small=True)
    y = frame.y.to_numpy()
    x1 = np.ones((n, 1))
    x2, z2 = frame[["p"]].to_numpy(), frame[["z1", "z2"]].to_numpy()
    base = oracle(y, x1, x2, z2, small=True)
    xs, zs = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    xhat = proj(zs) @ xs
    scores = xhat * base["u"][:, None]
    meat = np.zeros((2, 2))
    for unit in range(units):                 # Newey-West within each panel, gaps respected
        rows = (frame.id == unit).to_numpy()
        meat += meat_of(scores[rows], "hac", time=frame.t.to_numpy()[rows], lags=2)
    bread = np.linalg.inv(xhat.T @ xhat)
    v_ref = bread @ meat @ bread * n / (n - 2)
    assert_allclose(coef(result), base["b"], rtol=1e-9)
    assert_allclose(vcov(result), v_ref, rtol=1e-8)
    assert result.inference["df_inference"] == n - 2
    with pytest.raises(AnalysisError) as error:
        oe.ivregress(data=frame, y="y", endog=["p"], instruments=["z1", "z2"],
                     covariance="hac", time="t", panel="id")
    assert error.value.code == "invalid_spec" and "lags" in str(error.value)


# ---- xtivreg ---------------------------------------------------------------------------------


def make_panel(seed=101, units=41, periods=7):
    """Unbalanced panel with gaps, a time-invariant regressor and unsorted rows."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"id": np.repeat(np.arange(units) * 3 + 100, periods),
                          "year": np.tile(np.arange(1990, 1990 + periods), units)})
    n = len(frame)
    a = np.repeat(rng.normal(size=units), periods)
    frame["state"] = (frame.id // 3) % 8
    frame["g"] = np.repeat(rng.normal(size=units), periods)          # time invariant
    frame["x1"] = rng.normal(size=n) + 0.6 * a
    frame["z1"] = rng.normal(size=n) + 0.4 * a
    frame["z2"] = rng.normal(size=n) - 0.2 * a
    frame["z3"] = rng.normal(size=n)
    v1, v2 = rng.normal(size=n), rng.normal(size=n)
    frame["p1"] = 0.9 * frame.z1 + 0.5 * frame.z2 + 0.3 * frame.x1 + a + v1
    frame["p2"] = 0.7 * frame.z3 - 0.4 * frame.z1 + 0.5 * a + v2
    frame["y"] = (1 + 1.5 * frame.p1 - 0.8 * frame.p2 + 0.6 * frame.x1 + 0.4 * frame.g + a
                  + 0.7 * v1 - 0.5 * v2 + rng.normal(size=n) * (1 + 0.4 * np.abs(frame.z2)))
    keep = rng.uniform(size=n) > 0.12                                  # unbalanced, with gaps
    keep &= ~((frame.id == 100) & (frame.year > 1990))                 # one singleton panel
    return frame[keep].sample(frac=1.0, random_state=4).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


PX, PENDOG, PINST = ["x1"], ["p1", "p2"], ["z1", "z2", "z3"]


def xt(frame, **options):
    options = {"x": PX, "endog": PENDOG, "instruments": PINST, "panel": "id", "time": "year",
               **options}
    return oe.xtivreg(data=frame, y="y", **options)


def demean_by(values, ids):
    values = np.asarray(values, float)
    return values - pd.DataFrame(values).groupby(np.asarray(ids)).transform("mean").to_numpy(
        ).reshape(values.shape)


def means_by(values, ids):
    return pd.DataFrame(np.asarray(values, float)).groupby(np.asarray(ids), sort=True).mean(
        ).to_numpy()


def corr2(a, b):
    return np.corrcoef(a, b)[0, 1] ** 2


def xt_arrays(frame, x=PX, endog=PENDOG, inst=PINST):
    frame = frame.sort_values(["id", "year"]).reset_index(drop=True)
    return (frame, frame.y.to_numpy(float), frame[list(x)].to_numpy(float),
            frame[list(endog)].to_numpy(float), frame[list(inst)].to_numpy(float))


@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_xtivreg_fe_is_dummy_variable_two_stage_least_squares(panel, vce, small):
    frame, y, x1, x2, z2 = xt_arrays(panel)
    ids = frame.id.to_numpy()
    n_obs, n = len(y), frame.id.nunique()
    k = 3
    call = {"cluster": "state"} if vce == "cluster" else {"covariance": vce}
    result = xt(panel, model="fe", small=small, **call)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "p1", "p2"]
    # (1) LSDV: slopes, RSS and the conventional slope covariance
    dummies = (ids[:, None] == np.unique(ids)[None, :]).astype(float)
    lsdv = oracle(y, np.column_stack([dummies, x1]), x2, z2, small=True)
    assert_allclose(coef(result)[1:], lsdv["b"][n:], rtol=1e-8)
    rss = lsdv["rss"]
    assert result.metrics["sigma_e"] == pytest.approx(np.sqrt(rss / (n_obs - n - k)), rel=1e-9)
    # (2) Stata's transformed regression: panel means removed, grand means added back
    def tilde(values):
        return demean_by(values, ids) + np.asarray(values).mean(axis=0)
    cluster = {"robust": [ids], "cluster": [frame.state.to_numpy()]}.get(vce)
    ref = oracle(tilde(y), np.column_stack([np.ones(n_obs), tilde(x1)]), tilde(x2), tilde(z2),
                 vce="nonrobust" if vce == "nonrobust" else "cluster", small=True,
                 cluster=cluster, absorbed=n - 1 if vce == "nonrobust" else 0)
    if vce == "nonrobust":
        assert_allclose(ref["v"][1:, 1:], lsdv["v"][n:, n:], rtol=1e-8)
        ref["df_t"] = n_obs - n - k
    check(result, ref, small=bool(small))
    assert result.inference["use_t"] is bool(small)          # Stata's default: z and chi2
    slopes = ref["b"][1:]
    xall = np.column_stack([x1, x2])
    assert ref["b"][0] == pytest.approx(y.mean() - xall.mean(axis=0) @ slopes, rel=1e-9)
    # (3) reported statistics
    ybar, xbar = means_by(y, ids)[:, 0], means_by(xall, ids)
    u = ybar - xbar @ slopes
    sigma_u, sigma_e = u.std(ddof=1), np.sqrt(rss / (n_obs - n - k))
    assert result.metrics["sigma_u"] == pytest.approx(sigma_u, rel=1e-9)
    assert result.metrics["rho"] == pytest.approx(sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2),
                                                  rel=1e-9)
    codes = np.searchsorted(np.unique(ids), ids)
    assert result.metrics["corr_u_xb"] == pytest.approx(
        np.corrcoef(u[codes], xall @ slopes)[0, 1], rel=1e-8)          # over the N observations
    tss_w = (demean_by(y, ids) ** 2).sum()
    assert result.metrics["r_squared_within"] == pytest.approx(1 - rss / tss_w, rel=1e-9)
    assert result.metrics["r_squared_between"] == pytest.approx(corr2(xbar @ slopes, ybar),
                                                                rel=1e-9)
    assert result.metrics["r_squared_overall"] == pytest.approx(corr2(xall @ slopes, y),
                                                                rel=1e-9)
    assert result.metrics["n_groups"] == n and result.metrics["df_resid"] == n_obs - n - k
    sizes = frame.groupby("id").size()
    assert (result.metrics["t_min"], result.metrics["t_max"]) == (sizes.min(), sizes.max())
    assert result.metrics["t_avg"] == pytest.approx(n_obs / n)
    if vce == "nonrobust":
        pooled = oracle(y, np.column_stack([np.ones(n_obs), x1]), x2, z2)
        f = ((pooled["rss"] - rss) / (n - 1)) / (rss / (n_obs - n - k))
        check_entry(result.tests["fixed_effects"], f, n - 1, n_obs - n - k)
    else:
        assert result.inference["cluster_count"] == len(pd.unique(cluster[0]))


@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_xtivreg_fd_differences_consecutive_periods(panel, vce, small):
    frame, y, x1, x2, z2 = xt_arrays(panel)
    ids, year = frame.id.to_numpy(), frame.year.to_numpy()
    ok = np.r_[False, (ids[1:] == ids[:-1]) & (year[1:] - year[:-1] == 1)]

    def d(values):
        values = np.asarray(values, float)
        return (values - np.roll(values, 1, axis=0))[ok]

    rows = int(ok.sum())
    call = {"cluster": "state"} if vce == "cluster" else {"covariance": vce}
    result = xt(panel, model="fd", small=small, **call)
    cluster = {"robust": [ids[ok]], "cluster": [frame.state.to_numpy()[ok]]}.get(vce)
    ref = oracle(d(y), np.column_stack([np.ones(rows), d(x1)]), d(x2), d(z2), small=True,
                 vce="nonrobust" if vce == "nonrobust" else "cluster", cluster=cluster)
    check(result, ref, small=bool(small))
    assert result.nobs == rows and result.metrics["df_resid"] == rows - 4
    dy = d(y)
    assert result.metrics["r_squared"] == pytest.approx(
        1 - ref["rss"] / ((dy - dy.mean()) ** 2).sum(), rel=1e-9)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(ref["rss"] / (rows - 4)), rel=1e-9)
    assert result.metrics["n_groups"] == len(np.unique(ids[ok]))
    assert result.extra["dropped_for_differencing"] == len(y) - rows


@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_xtivreg_be_is_two_stage_least_squares_on_panel_means(panel, vce, small):
    frame, y, x1, x2, z2 = xt_arrays(panel, x=["x1", "g"])
    ids = frame.id.to_numpy()
    n = frame.id.nunique()
    yb, x1b, x2b, z2b = means_by(y, ids)[:, 0], means_by(x1, ids), means_by(x2, ids), \
        means_by(z2, ids)
    call = {"cluster": "state"} if vce == "cluster" else {"covariance": vce}
    result = xt(panel, model="be", x=["x1", "g"], small=small, **call)
    state = frame.groupby("id").state.first().to_numpy()
    ref = oracle(yb, np.column_stack([np.ones(n), x1b]), x2b, z2b, small=True, vce=vce,
                 cluster=[state] if vce == "cluster" else None)
    check(result, ref, small=bool(small))
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "g", "p1", "p2"]
    assert result.metrics["df_resid"] == n - 5 and result.nobs == len(y)
    slopes = ref["b"][1:]
    xall = np.column_stack([x1, x2])
    assert result.metrics["r_squared_between"] == pytest.approx(
        1 - ref["rss"] / ((yb - yb.mean()) ** 2).sum(), rel=1e-9)
    assert result.metrics["r_squared_within"] == pytest.approx(
        corr2(demean_by(xall, ids) @ slopes, demean_by(y, ids)), rel=1e-9)
    assert result.metrics["r_squared_overall"] == pytest.approx(corr2(xall @ slopes, y),
                                                                rel=1e-9)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(ref["rss"] / (n - 5)), rel=1e-9)


def g2sls(frame, y, x1, x2, z2, *, ec2sls=False):
    """Balestra-Varadharajan-Krishnakumar G2SLS / Baltagi EC2SLS by dense algebra."""
    ids = frame.id.to_numpy()
    n_obs, n = len(y), frame.id.nunique()
    sizes = frame.groupby("id").size().to_numpy().astype(float)
    codes = np.searchsorted(np.unique(ids), ids)
    kk = 1 + x1.shape[1] + x2.shape[1]
    # sigma_e: within 2SLS of the time-varying columns
    varying = [j for j in range(x1.shape[1]) if np.ptp(demean_by(x1[:, j], ids)) > 1e-9]
    xw = demean_by(x1[:, varying], ids)
    fe = oracle(demean_by(y, ids), xw, demean_by(x2, ids), demean_by(z2, ids))
    k_w = len(varying) + x2.shape[1]
    sigma_e2 = fe["rss"] / (n_obs - n - k_w)
    # sigma_u: between 2SLS on the N observations (each panel mean appears T_i times)
    def spread(values):
        values = np.asarray(values, float)
        return values - demean_by(values, ids)
    be = oracle(spread(y), np.column_stack([np.ones(n_obs), spread(x1)]), spread(x2),
                spread(z2))
    # Swamy-Arora adapted to unbalanced panels (Baltagi and Chang 2000; xtivreg's default):
    # r = trace{(Zb'Zb)^-1 Zb' Z_mu Z_mu' Zb} with Z_mu the N-by-n panel indicator matrix
    zb = np.column_stack([np.ones(n_obs), spread(x1), spread(x2)])
    z_mu = (codes[:, None] == np.arange(n)[None, :]).astype(float)
    r = np.trace(np.linalg.solve(zb.T @ zb, zb.T @ z_mu @ z_mu.T @ zb))
    sigma_u2 = max(0.0, (be["rss"] - (n - kk) * sigma_e2) / (n_obs - r))
    theta = 1 - np.sqrt(sigma_e2 / (sizes * sigma_u2 + sigma_e2))
    th = theta[codes]

    def quasi(values):
        values = np.asarray(values, float)
        mean = values - demean_by(values, ids)
        return values - (th[:, None] if values.ndim == 2 else th) * mean

    ones = np.ones(n_obs)
    xs = np.column_stack([quasi(ones), quasi(x1), quasi(x2)])
    ys = quasi(y)
    if ec2sls:
        exo = np.column_stack([x1, z2])
        zs = np.column_stack([ones, demean_by(exo, ids), exo - demean_by(exo, ids)])
        # drop the within part of time-invariant columns (all zero)
        zs = zs[:, np.abs(zs).max(axis=0) > 1e-9]
    else:
        zs = np.column_stack([quasi(ones), quasi(x1), quasi(z2)])
    pz = proj(zs)
    b = np.linalg.solve(xs.T @ pz @ xs, xs.T @ pz @ ys)
    u = ys - xs @ b
    return {"b": b, "u": u, "rss": float(u @ u), "xhat": pz @ xs, "sigma_e2": sigma_e2,
            "sigma_u2": sigma_u2, "theta": theta, "kk": kk, "n": n, "ids": ids,
            "bread": np.linalg.inv(xs.T @ pz @ xs)}


@pytest.mark.parametrize("ec2sls", [False, True])
@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_xtivreg_re_g2sls_and_ec2sls(panel, ec2sls, vce, small):
    frame, y, x1, x2, z2 = xt_arrays(panel, x=["x1", "g"])
    ref = g2sls(frame, y, x1, x2, z2, ec2sls=ec2sls)
    n_obs, kk = len(y), ref["kk"]
    call = {"cluster": "state"} if vce == "cluster" else {"covariance": vce}
    result = xt(panel, model="re", x=["x1", "g"], ec2sls=ec2sls, small=small, **call)
    if vce == "nonrobust":
        # the conventional VCE of the 2SLS regression on the transformed data
        ref["v"], ref["df_t"] = ref["bread"] * ref["rss"] / (n_obs - kk), n_obs - kk
    else:
        labels = ref["ids"] if vce == "robust" else frame.state.to_numpy()
        g = len(np.unique(labels))
        meat = group_meat(ref["xhat"] * ref["u"][:, None], labels)
        ref["v"] = ref["bread"] @ meat @ ref["bread"] * g / (g - 1) * (n_obs - 1) / (n_obs - kk)
        ref["df_t"] = g - 1
    check(result, ref, small=bool(small))
    assert result.inference["use_t"] is bool(small)
    metrics = result.metrics
    assert metrics["sigma_e"] == pytest.approx(np.sqrt(ref["sigma_e2"]), rel=1e-9)
    assert metrics["sigma_u"] == pytest.approx(np.sqrt(ref["sigma_u2"]), rel=1e-8)
    assert metrics["rho"] == pytest.approx(
        ref["sigma_u2"] / (ref["sigma_u2"] + ref["sigma_e2"]), rel=1e-8)
    assert metrics["theta"] is None                                    # unbalanced panel
    assert result.extra["theta"]["max"] == pytest.approx(ref["theta"].max(), rel=1e-8)
    assert result.extra["theta"]["min"] == pytest.approx(ref["theta"].min(), rel=1e-8)
    slopes = ref["b"][1:]
    xall, ids = np.column_stack([x1, x2]), ref["ids"]
    assert metrics["r_squared_within"] == pytest.approx(
        corr2(demean_by(xall, ids) @ slopes, demean_by(y, ids)), rel=1e-8)
    assert metrics["r_squared_between"] == pytest.approx(
        corr2(means_by(xall, ids) @ slopes, means_by(y, ids)[:, 0]), rel=1e-8)
    assert metrics["r_squared_overall"] == pytest.approx(corr2(xall @ slopes, y), rel=1e-8)
    assert result.title.startswith("EC2SLS" if ec2sls else "G2SLS")


def test_xtivreg_re_on_a_balanced_panel():
    rng = np.random.default_rng(8)
    units, periods = 30, 5
    frame = pd.DataFrame({"id": np.repeat(np.arange(units), periods),
                          "year": np.tile(np.arange(periods), units)})
    a = np.repeat(rng.normal(size=units), periods)
    n = len(frame)
    frame["z1"], frame["z2"] = rng.normal(size=n), rng.normal(size=n) + 0.3 * a
    v = rng.normal(size=n)
    frame["x1"] = rng.normal(size=n)
    frame["p1"] = frame.z1 + 0.6 * frame.z2 + v + 0.5 * a
    frame["y"] = 2 + frame.p1 - frame.x1 + a + 0.5 * v + rng.normal(size=n)
    sorted_frame, y, x1, x2, z2 = xt_arrays(frame, endog=["p1"], inst=["z1", "z2"])
    for ec2sls in (False, True):
        ref = g2sls(sorted_frame, y, x1, x2, z2, ec2sls=ec2sls)
        result = oe.xtivreg(data=frame, y="y", x=["x1"], endog=["p1"], instruments=["z1", "z2"],
                            panel="id", time="year", model="re", ec2sls=ec2sls)
        ref["v"], ref["df_t"] = ref["bread"] * ref["rss"] / (n - 3), n - 3
        check(result, ref, small=False)
        assert result.metrics["theta"] == pytest.approx(ref["theta"][0], rel=1e-9)
        # balanced panels: the Swamy-Arora formula sigma_u^2 = RSS_b/(n-K) - sigma_e^2/T
        assert result.metrics["sigma_u"] == pytest.approx(np.sqrt(ref["sigma_u2"]), rel=1e-9)


def test_xtivreg_re_with_columns_that_do_not_vary_between_panels():
    # A trend and a period-only instrument in a balanced panel are collinear with the
    # constant at the panel-mean level: the between regression behind sigma_u leaves them out.
    rng = np.random.default_rng(8)
    units, periods = 30, 5
    frame = pd.DataFrame({"id": np.repeat(np.arange(units), periods),
                          "year": np.tile(np.arange(periods), units)})
    n = len(frame)
    a = np.repeat(rng.normal(size=units), periods)
    frame["z1"], frame["z2"] = rng.normal(size=n), rng.normal(size=n) + 0.3 * a
    frame["zt"] = np.tile(rng.normal(size=periods), units)
    frame["trend"] = frame.year + 1990.0
    frame["x1"] = rng.normal(size=n)
    v = rng.normal(size=n)
    frame["p1"] = frame.z1 + 0.6 * frame.z2 + 0.3 * frame.zt + v + 0.5 * a
    frame["y"] = 2 + frame.p1 - frame.x1 + 0.1 * frame.trend + a + 0.5 * v + rng.normal(size=n)
    result = oe.xtivreg(data=frame, y="y", x=["x1", "trend"], endog=["p1"],
                        instruments=["z1", "z2", "zt"], panel="id", time="year", model="re")
    ids = frame.id.to_numpy()
    y = frame.y.to_numpy()
    x1, x2 = frame[["x1", "trend"]].to_numpy(), frame[["p1"]].to_numpy()
    z2 = frame[["z1", "z2", "zt"]].to_numpy()
    fe = oracle(demean_by(y, ids), demean_by(x1, ids), demean_by(x2, ids), demean_by(z2, ids))
    sigma_e2 = fe["rss"] / (n - units - 3)
    be = oracle(means_by(y, ids)[:, 0],
                np.column_stack([np.ones(units), means_by(x1[:, :1], ids)]),
                means_by(x2, ids), means_by(z2[:, :2], ids))             # K_b = 3
    sigma_u2 = be["rss"] / (units - 3) - sigma_e2 / periods
    components = result.extra["variance_components"]
    assert components["between_regressors"] == 3 and components["df_between"] == units - 3
    assert result.metrics["sigma_e"] == pytest.approx(np.sqrt(sigma_e2), rel=1e-9)
    assert result.metrics["sigma_u"] == pytest.approx(np.sqrt(sigma_u2), rel=1e-8)
    theta = 1 - np.sqrt(sigma_e2 / (periods * sigma_u2 + sigma_e2))
    assert result.metrics["theta"] == pytest.approx(theta, rel=1e-8)

    def quasi(values):
        values = np.asarray(values, float)
        return values - theta * (values - demean_by(values, ids))

    ones = np.ones(n)
    xs = np.column_stack([quasi(ones), quasi(x1), quasi(x2)])
    zs = np.column_stack([quasi(ones), quasi(x1), quasi(z2)])
    pz = proj(zs)
    b = np.linalg.solve(xs.T @ pz @ xs, xs.T @ pz @ quasi(y))
    assert_allclose(coef(result), b, rtol=1e-7)
    with pytest.raises(AnalysisError) as error:
        oe.xtivreg(data=frame, y="y", x=["x1"], endog=["p1"], instruments=["zt"], panel="id",
                   time="year", model="re")
    assert error.value.code == "underidentified" and "between" in str(error.value)


# ---- ivreghdfe -------------------------------------------------------------------------------


def make_firms(seed=23, firms=45, years=8):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"firm": np.repeat(np.arange(firms), years),
                          "year": np.tile(np.arange(2001, 2001 + years), firms)})
    n = len(frame)
    frame["industry"] = frame.firm % 9
    a, d = rng.normal(size=firms)[frame.firm], rng.normal(size=years)[frame.year - 2001]
    frame["x1"] = rng.normal(size=n) + 0.5 * a
    frame["z1"] = rng.normal(size=n) + 0.3 * a + 0.2 * d
    frame["z2"] = rng.normal(size=n)
    frame["z3"] = rng.normal(size=n) - 0.3 * d
    v1, v2 = rng.normal(size=n), rng.normal(size=n)
    frame["p1"] = 0.8 * frame.z1 + 0.5 * frame.z2 + 0.2 * frame.x1 + a + d + v1
    frame["p2"] = 0.6 * frame.z3 - 0.3 * frame.z2 + 0.5 * a + v2
    frame["y"] = (1.2 * frame.p1 - 0.9 * frame.p2 + 0.5 * frame.x1 + a + d + 0.6 * v1 - 0.4 * v2
                  + rng.normal(size=n) * (1 + 0.5 * np.abs(frame.z1)))
    frame["fw"] = rng.integers(1, 4, size=n).astype(float)
    frame["aw"] = rng.uniform(0.4, 2.5, size=n)
    keep = rng.uniform(size=n) > 0.1
    return frame[keep].reset_index(drop=True)


@pytest.fixture(scope="module")
def firms():
    return make_firms()


HX, HENDOG, HINST = ["x1"], ["p1", "p2"], ["z1", "z2", "z3"]


def hd(frame, **options):
    options = {"x": HX, "endog": HENDOG, "instruments": HINST, "absorb": ["firm", "year"],
               **options}
    return oe.ivreghdfe(data=frame, y="y", **options)


def dummy_matrix(frame, columns):
    """Full-column-rank dummy block of the absorbed dimensions (dense; small data only)."""
    blocks = [(frame[c].to_numpy()[:, None] == np.unique(frame[c])[None, :]).astype(float)
              for c in columns]
    full = np.column_stack(blocks)
    q, r, pivots = scipy.linalg.qr(full, mode="economic", pivoting=True)
    rank = int((np.abs(np.diag(r)) > 1e-9 * np.abs(r[0, 0])).sum())
    return full[:, np.sort(pivots[:rank])]


def residualize(values, dummies, w=None):
    values = np.asarray(values, float)
    sw = np.ones(len(dummies)) if w is None else np.sqrt(w)
    target = values * (sw[:, None] if values.ndim == 2 else sw)
    fitted = dummies @ np.linalg.lstsq(dummies * sw[:, None], target, rcond=None)[0]
    return values - fitted


HDFE_CASES = [
    # options, oracle vce, cluster columns, df_absorbed rule
    ({}, "nonrobust", None),
    ({"covariance": "robust"}, "robust", None),
    ({"cluster": "firm"}, "cluster", ["firm"]),
    ({"cluster": "industry"}, "cluster", ["industry"]),
    ({"cluster": ["firm", "year"]}, "cluster", ["firm", "year"]),
]


def absorbed_df(frame, absorb, cluster):
    """reghdfe: levels minus redundant; dimensions nested in a cluster column cost nothing."""
    def nested(dim):
        return cluster is not None and any(
            frame.groupby(dim)[c].nunique().max() == 1 for c in cluster)
    free = [dim for dim in absorb if not nested(dim)]
    if not free:
        return 1                                            # the constant
    rank = dummy_matrix(frame, free).shape[1]
    return rank


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
@pytest.mark.parametrize("options,vce,cluster", HDFE_CASES)
@pytest.mark.parametrize("small", [True, False])
def test_ivreghdfe_is_dummy_variable_iv(firms, method, options, vce, cluster, small):
    result = hd(firms, method=method, small=small, **options)
    dummies = dummy_matrix(firms, ["firm", "year"])
    y, x1, x2, z2 = (residualize(firms[c].to_numpy(float), dummies)
                     for c in ("y", HX, HENDOG, HINST))
    n, k = len(firms), 3
    df_a = absorbed_df(firms, ["firm", "year"], cluster)
    labels = None if cluster is None else [firms[c].to_numpy() for c in cluster]
    wmatrix = {"nonrobust": "unadjusted"}.get(vce, vce)
    ref = oracle(y, x1, x2, z2, method=method, vce=vce, small=small, cluster=labels,
                 absorbed=df_a, wmatrix=wmatrix, group_factor=small)
    assert [c.term for c in result.coefficients] == ["x1", "p1", "p2"]
    check(result, ref, small=small, rtol=1e-6, vtol=1e-6)
    assert result.metrics["df_absorbed"] == df_a
    assert result.metrics["df_resid"] == n - k - df_a
    if vce != "cluster":
        # the same numbers from the regression with explicit dummies
        raw = [firms[c].to_numpy(float) for c in ("y", HX, HENDOG, HINST)]
        lsdv = oracle(raw[0], np.column_stack([dummies, raw[1]]), raw[2], raw[3],
                      method=method, vce=vce, small=small, wmatrix=wmatrix)
        d = dummies.shape[1]
        assert df_a == d
        assert_allclose(coef(result), lsdv["b"][d:], rtol=1e-6)
        if not (method == "gmm" and vce == "robust"):
            # (efficient GMM with dummies in the moment set is a different estimator)
            assert_allclose(vcov(result), lsdv["v"][d:, d:], rtol=1e-6)
    rss = ref["rss"]
    yraw = firms.y.to_numpy()
    assert result.metrics["r_squared"] == pytest.approx(
        1 - rss / ((yraw - yraw.mean()) ** 2).sum(), rel=1e-7)
    assert result.metrics["r_squared_within"] == pytest.approx(1 - rss / (y @ y), rel=1e-7)
    assert result.metrics["rmse"] == pytest.approx(
        np.sqrt(rss / (n - k - df_a if small else n)), rel=1e-7)
    if method == "liml":
        assert result.metrics["kappa"] == pytest.approx(ref["kappa"], rel=1e-7)
    if method == "gmm":
        assert result.metrics["j"] == pytest.approx(ref["j"], rel=1e-6)


@pytest.mark.parametrize("options,vce,cluster", HDFE_CASES[:3])
def test_ivreghdfe_postestimation_statistics(firms, options, vce, cluster):
    result = hd(firms, **options)
    dummies = dummy_matrix(firms, ["firm", "year"])
    y, x1, x2, z2 = (residualize(firms[c].to_numpy(float), dummies)
                     for c in ("y", HX, HENDOG, HINST))
    df_a = absorbed_df(firms, ["firm", "year"], cluster)
    labels = None if cluster is None else [firms[c].to_numpy() for c in cluster]
    ref = diagnostics(y, x1, x2, z2, vce=vce, cluster=labels, absorbed=df_a, constant=False,
                      tss=[(x2[:, j] ** 2).sum() for j in range(2)])
    n = len(firms)
    check_first_stage(result, ref, terms=HENDOG)
    tests = result.tests
    assert tests["cragg_donald"]["statistic"] == pytest.approx(ref["cragg_donald"], rel=1e-6)
    assert tests["cragg_donald"]["df2"] == n - 4 - df_a
    if vce == "nonrobust":
        check_entry(tests["overid_sargan"], ref["sargan"], 1)
        check_entry(tests["overid_basmann"], ref["basmann"], 1)
        check_entry(tests["endog_durbin"], ref["durbin"], 2)
        check_entry(tests["endog_wu_hausman"], ref["wu_hausman"], 2, n - 3 - 2 - df_a)
    else:
        check_entry(tests["overid_score"], ref["overid_score"], 1)
        check_entry(tests["endog_robust_score"], ref["endog_score"], 2)
        check_entry(tests["endog_robust_regression"], ref["endog_regression_f"], 2,
                    ref["endog_regression_df2"])
        assert tests["kleibergen_paap_rk_f"]["statistic"] == pytest.approx(ref["kp_f"],
                                                                           rel=1e-6)


@pytest.mark.parametrize("vce", ["nonrobust", "robust", "cluster"])
def test_ivreghdfe_weights(firms, vce):
    call = {"cluster": "industry"} if vce == "cluster" else {"covariance": vce}
    labels = [firms.industry.to_numpy()] if vce == "cluster" else None
    w = firms.aw.to_numpy()
    dummies = dummy_matrix(firms, ["firm", "year"])
    y, x1, x2, z2 = (residualize(firms[c].to_numpy(float), dummies, w)
                     for c in ("y", HX, HENDOG, HINST))
    df_a = absorbed_df(firms, ["firm", "year"], ["industry"] if vce == "cluster" else None)
    result = hd(firms, weights="aw", weight_type="aweight", **call)
    ref = oracle(y, x1, x2, z2, w=w, vce=vce, small=True, cluster=labels, absorbed=df_a)
    check(result, ref, small=True, rtol=1e-6, vtol=1e-6)
    # frequency weights replicate rows (also for the singleton rule)
    weighted = hd(firms, weights="fw", weight_type="fweight", **call)
    expanded = hd(duplicated(firms), **call)
    assert weighted.nobs == expanded.nobs == int(firms.fw.sum())
    assert_allclose(coef(weighted), coef(expanded), rtol=1e-7)
    assert_allclose(vcov(weighted), vcov(expanded), rtol=1e-6)
    same_statistics(weighted, expanded, rel=1e-5)
    if vce != "nonrobust":
        pw = hd(firms, weights="aw", weight_type="pweight", **call)
        assert_allclose(vcov(pw), vcov(result), rtol=1e-10)


# ---- equivalences across estimators ----------------------------------------------------------


@pytest.mark.parametrize("model", ["fe", "re", "be", "fd"])
@pytest.mark.parametrize("vce", ["nonrobust", "robust"])
def test_xtivreg_with_self_instrumented_regressors_is_xtreg(panel, model, vce):
    # Instruments that reproduce the "endogenous" regressors turn every xtivreg model into
    # the corresponding xtreg model: coefficients, covariance and reported statistics.
    frame = panel.assign(c1=panel.p1, c2=panel.p2)
    iv_fit = oe.xtivreg(data=frame, y="y", x=["x1"], endog=["p1", "p2"],
                        instruments=["c1", "c2"], panel="id", time="year", model=model,
                        covariance=vce, small=model != "re")
    # xtivreg's random-effects default is the Swamy-Arora estimator (xtreg, re sa)
    ls_fit = oe.xtreg(data=frame, y="y", x=["x1", "p1", "p2"], panel="id", time="year",
                      model=model, covariance=vce, sa=model == "re")
    assert [c.term for c in iv_fit.coefficients] == [c.term for c in ls_fit.coefficients]
    assert_allclose(coef(iv_fit), coef(ls_fit), rtol=1e-9)
    assert_allclose(vcov(iv_fit), vcov(ls_fit), rtol=1e-8)
    assert_allclose([c.p_value for c in iv_fit.coefficients],
                    [c.p_value for c in ls_fit.coefficients], rtol=1e-6, atol=1e-290)
    for name in ("sigma_u", "sigma_e", "rho", "theta", "corr_u_xb", "r_squared_within",
                 "r_squared_between", "r_squared_overall", "r_squared", "rmse"):
        if name in iv_fit.metrics and name in ls_fit.metrics \
                and iv_fit.metrics[name] is not None:
            assert iv_fit.metrics[name] == pytest.approx(ls_fit.metrics[name], rel=1e-8), name


def test_ivreghdfe_with_one_dimension_matches_xtivreg_fe(firms):
    args = {"y": "y", "x": HX, "endog": HENDOG, "instruments": HINST}
    absorbed = oe.ivreghdfe(data=firms, absorb=["firm"], **args)
    within = oe.xtivreg(data=firms, panel="firm", time="year", model="fe", small=True, **args)
    assert_allclose(coef(absorbed), coef(within)[1:], rtol=1e-9)
    assert_allclose(vcov(absorbed), vcov(within)[1:, 1:], rtol=1e-8)
    assert absorbed.metrics["df_resid"] == within.metrics["df_resid"]
    assert absorbed.metrics["r_squared_within"] == pytest.approx(
        within.metrics["r_squared_within"], rel=1e-9)
    clustered = oe.ivreghdfe(data=firms, absorb=["firm"], cluster="firm", **args)
    robust = oe.xtivreg(data=firms, panel="firm", time="year", model="fe", small=True,
                        covariance="robust", **args)
    assert_allclose(vcov(clustered), vcov(robust)[1:, 1:], rtol=1e-8)
    assert clustered.inference["df_inference"] == robust.inference["df_inference"]
    # ivreghdfe without fixed-effect structure beyond a constant is ivregress with small
    constant = oe.ivreghdfe(data=firms.assign(one=1), absorb=["one"], covariance="robust",
                            **args)
    plain = oe.ivregress(data=firms, small=True, covariance="robust", **args)
    assert_allclose(coef(constant), coef(plain)[1:], rtol=1e-9)
    assert_allclose(vcov(constant), vcov(plain)[1:, 1:], rtol=1e-8)
    for name in ("overid_score", "endog_robust_score", "endog_robust_regression",
                 "cragg_donald", "kleibergen_paap_rk_f"):
        assert constant.tests[name]["statistic"] == pytest.approx(
            plain.tests[name]["statistic"], rel=1e-7), name


@pytest.mark.parametrize("options", [
    {}, {"covariance": "robust"}, {"cluster": "firm"}, {"cluster": ["firm", "year"]},
    {"cluster": "industry"}, {"weights": "aw", "weight_type": "aweight"},
    {"weights": "fw", "weight_type": "fweight", "cluster": "firm"},
])
def test_ivreghdfe_with_self_instrumented_regressors_is_reghdfe(firms, options):
    frame = firms.assign(c1=firms.p1, c2=firms.p2)
    iv_fit = oe.ivreghdfe(data=frame, y="y", x=HX, endog=HENDOG, instruments=["c1", "c2"],
                          absorb=["firm", "year"], **options)
    ls_fit = oe.reghdfe(data=frame, y="y", x=[*HX, *HENDOG], absorb=["firm", "year"], **options)
    assert_allclose(coef(iv_fit), coef(ls_fit), rtol=1e-9)
    assert_allclose(vcov(iv_fit), vcov(ls_fit), rtol=1e-8)
    assert iv_fit.inference["df_inference"] == ls_fit.inference["df_inference"]
    for name in ("df_absorbed", "df_resid", "r_squared", "r_squared_within", "rmse",
                 "adjusted_r_squared", "adjusted_r_squared_within", "n_singletons_dropped"):
        assert iv_fit.metrics[name] == pytest.approx(ls_fit.metrics[name], rel=1e-9), name
    assert iv_fit.extra["absorbed"] == ls_fit.extra["absorbed"]


@pytest.mark.parametrize("model", ["fe", "re", "be", "fd"])
def test_xtivreg_row_order_and_units_do_not_matter(panel, model):
    base = xt(panel, model=model, covariance="robust")
    shuffled = xt(panel.sample(frac=1.0, random_state=9).reset_index(drop=True), model=model,
                  covariance="robust")
    assert_allclose(coef(shuffled), coef(base), rtol=1e-9)
    assert_allclose(vcov(shuffled), vcov(base), rtol=1e-8)
    scaled = panel.assign(y=panel.y * 1e5, p1=panel.p1 * 1e-4, z1=panel.z1 * 1e6,
                          id=panel.id.astype(str) + "-unit")
    result = xt(scaled, model=model, covariance="robust")
    scale = 1e5 / np.array([1.0, 1.0, 1e-4, 1.0])
    assert_allclose(coef(result), coef(base) * scale, rtol=1e-7)
    assert_allclose(vcov(result), vcov(base) * np.outer(scale, scale), rtol=1e-6)


# ---- failure contract ------------------------------------------------------------------------


def code_of(call):
    with pytest.raises(AnalysisError) as error:
        call()
    assert str(error.value)                      # a message that says what to change
    return error.value.code


def test_ivregress_adversarial_inputs(data):
    assert code_of(lambda: iv(data.head(0))) == "empty_data"
    # too few rows are reported as such, not as collinearity
    assert code_of(lambda: iv(data.head(1))) == "insufficient_observations"
    assert code_of(lambda: iv(data.head(3))) == "insufficient_observations"
    assert code_of(lambda: iv(data.head(7))) == "insufficient_observations"       # n = L
    small = iv(data.head(8))                                                      # n = L + 1
    assert small.tests.keys() == {"model"} and "not computed" in small.warnings[0]
    assert code_of(lambda: iv(data.head(8), method="liml")) == "insufficient_observations"
    assert code_of(lambda: iv(data.assign(z1=np.nan), missing="drop")) == "empty_sample"
    assert code_of(lambda: iv(data.assign(y=3.0))) == "constant_outcome"
    for method in ("2sls", "gmm"):
        assert code_of(lambda: iv(data.assign(y=2 * data.p1 + 1), method=method)) \
            == "perfect_fit"
    assert code_of(lambda: iv(data.assign(z1=1.0, z2=2.0, z3=3.0))) == "underidentified"
    assert code_of(lambda: iv(data.assign(c=1), cluster="c")) == "insufficient_clusters"
    assert code_of(lambda: iv(data.assign(c=np.arange(len(data)) % 5), cluster="c",
                              method="gmm")) == "singular_weight_matrix"
    assert code_of(lambda: iv(data.assign(w=-1.0), weights="w", weight_type="aweight")) \
        == "negative_weights"
    assert code_of(lambda: iv(data.assign(w=1.5), weights="w", weight_type="fweight")) \
        == "noninteger_frequency_weights"
    assert code_of(lambda: iv(data.assign(w=0.0), weights="w", weight_type="aweight")) \
        == "empty_sample"
    assert code_of(lambda: iv(data.assign(p1="a"))) == "non_numeric_column"
    assert code_of(lambda: iv(data.assign(x1=np.where(np.arange(len(data)) == 0, np.inf,
                                                      data.x1)))) == "non_finite_values"
    assert code_of(lambda: iv(data.assign(t=0), covariance="hac", lags=1, time="t")) \
        == "repeated_time_values"
    assert code_of(lambda: iv(data, endog="p1")) == "invalid_spec"
    assert code_of(lambda: iv(data, instruments=["z1", "y"])) == "invalid_spec"
    assert code_of(lambda: iv(data, x=["x1", "p1"])) == "invalid_spec"
    assert code_of(lambda: iv(data, wmatrix="robust")) == "invalid_spec"
    assert code_of(lambda: iv(data, lags=2)) == "invalid_spec"
    assert code_of(lambda: iv(data, method="gmm", wmatrix="cluster")) == "invalid_spec"
    # zero weights are excluded and recorded; cluster = observation is the robust meat
    w = (np.arange(len(data)) % 3).astype(float)
    weighted = iv(data.assign(w=w), weights="w", weight_type="aweight")
    assert weighted.nobs == int((w > 0).sum()) and "zero weight" in weighted.warnings[0]
    kept = data[w > 0]
    check(weighted, oracle(*arrays(kept), w=w[w > 0]), small=False)
    each = iv(data.assign(c=np.arange(len(data))), cluster="c")
    n = len(data)
    assert_allclose(vcov(each), vcov(iv(data, covariance="robust")) * n / (n - 1), rtol=1e-9)
    # extreme magnitudes and offsets
    shifted = iv(data.assign(p1=data.p1 + 1e9, z1=data.z1 + 1e9))
    assert_allclose(coef(shifted)[1:], coef(iv(data))[1:], rtol=1e-5)
    # weak instruments still give finite output, and say so through the first stage
    noise = data.assign(q1=np.random.default_rng(0).normal(size=len(data)))
    weak = oe.ivregress(data=noise, y="y", endog=["p1"], instruments=["q1"])
    assert weak.tests["cragg_donald"]["statistic"] < 5
    assert np.isfinite(vcov(weak)).all()


def test_xtivreg_adversarial_inputs(panel):
    one = panel[panel.id == panel.id.iloc[0]]
    singles = panel.drop_duplicates("id")
    assert code_of(lambda: xt(panel.head(0))) == "empty_data"
    assert code_of(lambda: xt(singles, model="fe")) == "insufficient_observations"
    assert code_of(lambda: xt(singles, model="fd")) == "empty_sample"
    assert code_of(lambda: xt(singles, model="re")) == "insufficient_observations"
    assert code_of(lambda: xt(one, model="be")) == "insufficient_observations"
    assert code_of(lambda: xt(one, model="re")) == "insufficient_observations"
    assert code_of(lambda: xt(one, model="fd")) == "insufficient_observations"
    assert xt(singles, model="be").nobs == len(singles)            # between needs no repeats
    for model in ("fe", "fd", "be", "re"):
        assert code_of(lambda: xt(panel.assign(y=1.0), model=model)) == "constant_outcome"
        assert code_of(lambda: xt(pd.concat([panel, panel.head(2)]), model=model)) \
            == "repeated_time_values"
    constant_within = panel.assign(y=panel.id.astype(float))
    assert code_of(lambda: xt(constant_within, model="fe")) == "constant_outcome"
    assert code_of(lambda: xt(constant_within, model="re")) == "perfect_fit"
    assert code_of(lambda: xt(panel, model="fd", time=None)) == "invalid_spec"
    assert code_of(lambda: xt(panel, model="fe", ec2sls=True)) == "invalid_spec"
    assert code_of(lambda: xt(panel, model="be", cluster="year")) \
        == "cluster_varies_within_panel"
    assert code_of(lambda: xt(panel.assign(year=panel.year + 0.5), model="fd")) \
        == "invalid_time"
    invariant = panel.assign(z1=panel.g, z2=2 * panel.g, z3=panel.g ** 2)
    assert code_of(lambda: xt(invariant, model="fe")) == "underidentified"
    assert code_of(lambda: xt(invariant, model="re")) == "underidentified"
    assert code_of(lambda: xt(panel.assign(c=1), model="fe", cluster="c")) \
        == "insufficient_clusters"
    # a time-invariant endogenous regressor is omitted from fe with a warning
    result = xt(panel.assign(p2=panel.g), model="fe")
    assert result.provenance["omitted_terms"] == ["p2"]
    # the cluster-robust variance of the fe constant is xbar'V xbar: flagged when it is noise
    centered = panel.copy()
    for column in ("x1", "p1", "p2", "z1", "z2", "z3"):
        centered[column] -= centered[column].mean()
    flagged = xt(centered, model="fe", covariance="robust")
    assert flagged.coefficients[0].std_error < 1e-12
    assert any("zero up to rounding" in warning for warning in flagged.warnings)
    assert_allclose(coef(flagged)[1:], coef(xt(panel, model="fe"))[1:], rtol=1e-9)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "p1"]
    # clusters that do not nest the panels: allowed, flagged
    crossed = xt(panel, model="re", cluster="year")
    assert any("not nested" in warning for warning in crossed.warnings)
    crossed = xt(panel, model="fe", cluster="year")
    assert any("not nested" in warning for warning in crossed.warnings)


def test_ivreghdfe_adversarial_inputs(firms):
    assert code_of(lambda: hd(firms.head(0))) == "empty_data"
    assert code_of(lambda: hd(firms.head(4))) == "empty_sample"
    assert code_of(lambda: hd(firms.head(4), drop_singletons=False)) \
        == "insufficient_observations"
    each = firms.assign(i=np.arange(len(firms)))
    assert code_of(lambda: hd(each, absorb=["i"])) == "empty_sample"
    assert code_of(lambda: hd(each, absorb=["i"], drop_singletons=False)) \
        == "insufficient_observations"
    assert code_of(lambda: hd(firms.assign(y=2.0))) == "constant_outcome"
    assert code_of(lambda: hd(firms.assign(y=firms.firm * 1.0 + firms.year * 2.0))) \
        == "perfect_fit"
    assert code_of(lambda: hd(firms.assign(y=2 * firms.p1 + firms.firm))) == "perfect_fit"
    assert code_of(lambda: hd(firms.assign(z1=firms.firm * 1.0, z2=firms.year * 1.0))) \
        == "underidentified"
    assert code_of(lambda: hd(firms.assign(p1=firms.firm * 1.0, p2=firms.year * 2.0))) \
        == "no_endogenous_regressors"
    assert code_of(lambda: hd(firms, tolerance=0.0)) == "invalid_option"
    assert code_of(lambda: hd(firms, max_iterations=1)) == "absorption_nonconvergence"
    assert code_of(lambda: hd(firms, weights="aw", weight_type="pweight",
                              covariance="nonrobust")) == "unsupported_covariance"
    assert code_of(lambda: hd(firms.assign(c=1), cluster="c")) == "insufficient_clusters"
    assert code_of(lambda: hd(firms, absorb="firm")) == "invalid_spec"
    absorbed = hd(firms.assign(x1=firms.firm * 2.0))
    assert absorbed.provenance["omitted_terms"] == ["x1"]
    assert [c.term for c in absorbed.coefficients] == ["p1", "p2"]
    # a frequency-weighted row is never a singleton: identical to the duplicated data
    extra = firms.iloc[[0]].assign(firm=999, fw=3.0)
    frame = pd.concat([firms, extra], ignore_index=True)
    weighted = hd(frame, absorb=["firm"], weights="fw", weight_type="fweight")
    expanded = hd(duplicated(frame), absorb=["firm"])
    assert weighted.nobs == expanded.nobs == int(frame.fw.sum())
    assert weighted.metrics["n_singletons_dropped"] == 0
    assert_allclose(coef(weighted), coef(expanded), rtol=1e-9)
    assert_allclose(vcov(weighted), vcov(expanded), rtol=1e-8)
    assert weighted.metrics["df_absorbed"] == expanded.metrics["df_absorbed"]
    single = pd.concat([firms, extra.assign(fw=1.0)], ignore_index=True)
    dropped = hd(single, absorb=["firm"], weights="fw", weight_type="fweight")
    assert dropped.metrics["n_singletons_dropped"] == 1


# ---- identities between statistics, rendering --------------------------------------------------


def test_overidentification_identities(data):
    # Wooldridge's robust score test is the two-step GMM criterion with the robust weight
    # matrix from the 2SLS residuals; the unadjusted weight matrix gives Sargan's statistic.
    robust = iv(data, covariance="robust")
    gmm = iv(data, method="gmm")
    assert gmm.tests["hansen_j"]["statistic"] == pytest.approx(
        robust.tests["overid_score"]["statistic"], rel=1e-8)
    clustered = iv(data, cluster="firm")
    gmm = iv(data, method="gmm", cluster="firm")
    assert gmm.tests["hansen_j"]["statistic"] == pytest.approx(
        clustered.tests["overid_score"]["statistic"], rel=1e-8)
    plain = iv(data)
    unadjusted = iv(data, method="gmm", wmatrix="unadjusted")
    assert unadjusted.tests["hansen_j"]["statistic"] == pytest.approx(
        plain.tests["overid_sargan"]["statistic"], rel=1e-9)
    assert_allclose(coef(unadjusted), coef(plain), rtol=1e-12)
    # LIML: kappa >= 1 and the Anderson-Rubin statistic is N ln(kappa)
    liml = iv(data, method="liml")
    assert liml.metrics["kappa"] > 1
    assert liml.tests["anderson_rubin"]["statistic"] == pytest.approx(
        len(data) * np.log(liml.metrics["kappa"]), rel=1e-10)


@pytest.mark.parametrize("fit", [
    lambda d, p, f: iv(d, method="gmm", cluster="firm", igmm=True),
    lambda d, p, f: iv(d, method="liml", covariance="hac", lags=2, time="t", small=True),
    lambda d, p, f: iv(d, endog=["p1"], instruments=["z1"], x=X),            # exactly identified
    lambda d, p, f: xt(p, model="fe"),
    lambda d, p, f: xt(p, model="re", ec2sls=True, covariance="robust"),
    lambda d, p, f: xt(p, model="fd", small=True),
    lambda d, p, f: hd(f, cluster=["firm", "year"]),
    lambda d, p, f: hd(f, method="gmm", covariance="robust", small=False),
])
def test_results_are_json_safe_and_render(data, panel, firms, fit):
    import json
    import math

    result = fit(data, panel, firms)
    payload = json.loads(result.model_dump_json())

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, float):
            assert math.isfinite(value)

    walk(payload)
    assert payload["provenance"]["stata_parity_validated"] is False
    assert payload["provenance"]["family"] == "iv"
    terms = [c["term"] for c in payload["coefficients"]]
    assert len(set(terms)) == len(terms)
    for entry in payload["tests"].values():
        assert {"statistic", "df", "p_value", "distribution", "label"} <= set(entry)
    text = result.summary()
    assert all(term in text for term in terms)
    rebuilt = type(result).model_validate_json(result.model_dump_json())
    assert rebuilt.coefficients == result.coefficients
    again = oe.fit(result.spec, data={"gmm": data, "liml": data, "2sls": data}.get(
        result.extra.get("method"), None) if result.spec.estimator == "ivregress"
        else panel if result.spec.estimator == "xtivreg" else firms)
    assert_allclose(coef(again), coef(result), rtol=1e-12)
    assert_allclose(vcov(again), vcov(result), rtol=1e-12)
