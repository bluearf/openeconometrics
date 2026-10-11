"""IV extensions (JIVE/UJIVE, Lewbel, shift-share AKM, MOP weak-IV test) against independent oracles.

Oracles: brute-force leave-one-out regressions and dense hat matrices in NumPy
(JIVE1/JIVE2/UJIVE, transcribing Kolesar's ManyIV ``jive.fit``), explicit
stacked-moment sandwiches with finite-difference Jacobians (Lewbel), NumPy
transcriptions of the authors' R ShiftShareSE ``reg_ss.fit``/``ivreg_ss.fit``
and Stata ``weakivtest`` Mata code (SciPy noncentral chi-square), plus committed
fixtures from ``scripts/generate_iv_extension_oracles.{py,R}`` (linearmodels 7.0,
R AER/sandwich).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle

FIXTURES = Path(__file__).parent / "fixtures" / "iv_extensions"


def coef(result, term):
    return next(c for c in result.coefficients if c.term == term)


def cov_block(result, terms):
    names = [c.term for c in result.coefficients]
    index = [names.index(t) for t in terms]
    return np.asarray(result.covariance_matrix)[np.ix_(index, index)]


# ---- JIVE / UJIVE ----------------------------------------------------------------------------


def jive_data(seed=11, n=160, groups=20):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 6))
    v = rng.normal(size=n)
    frame = pd.DataFrame(z, columns=[f"z{j}" for j in range(6)])
    frame["x1"] = rng.normal(size=n)
    frame["d"] = z @ np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0]) + 0.5 * frame.x1 + v
    frame["d2"] = z @ np.array([0.0, 0.1, 0.3, 0.4, 0.2, 0.1]) + rng.normal(size=n)
    scale = 1 + 0.5 * np.abs(z[:, 0])
    frame["y"] = 1 + 0.5 * frame.d - frame.x1 + 0.8 * v + scale * rng.normal(size=n)
    frame["g"] = np.repeat(np.arange(groups), n // groups)
    return frame


INSTR = [f"z{j}" for j in range(6)]


def loo_prediction(q, target):
    """Brute-force leave-one-out OLS prediction of every column of target from q."""
    out = np.empty_like(target, dtype=float)
    for i in range(len(q)):
        keep = np.arange(len(q)) != i
        coefs = np.linalg.lstsq(q[keep], target[keep], rcond=None)[0]
        out[i] = q[i] @ coefs
    return out


def iv_sandwich(a, x, u, kind, groups=None, small=False):
    n, k = x.shape
    bread = np.linalg.inv(a.T @ x)
    if kind == "nonrobust":
        s2 = u @ u / ((n - k) if small else n)
        return s2 * bread @ (a.T @ a) @ bread.T
    rows = a * u[:, None]
    if kind == "robust":
        meat = rows.T @ rows * (n / (n - k) if small else 1.0)
    else:
        sums = np.vstack([rows[groups == g].sum(0) for g in np.unique(groups)])
        count = len(sums)
        meat = sums.T @ sums * count / (count - 1) * ((n - 1) / (n - k) if small else 1.0)
    return bread @ meat @ bread.T


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_jive1_matches_brute_force_leave_one_out(kind, small):
    d = jive_data()
    q = np.column_stack([np.ones(len(d)), d.x1, d[INSTR]])
    x = np.column_stack([np.ones(len(d)), d.x1, d.d, d.d2])
    xhat = loo_prediction(q, x)
    assert_allclose(xhat[:, :2], x[:, :2], atol=1e-10)  # exogenous columns are reproduced
    beta = np.linalg.solve(xhat.T @ x, xhat.T @ d.y)
    u = d.y.to_numpy() - x @ beta
    v = iv_sandwich(xhat, x, u, kind, d.g.to_numpy(), small)
    r = oe.jive(data=d, y="y", x=["x1"], endog=["d", "d2"], instruments=INSTR, method="jive1",
                covariance=kind, cluster="g" if kind == "cluster" else None, small=small)
    assert [c.term for c in r.coefficients] == ["Intercept", "x1", "d", "d2"]
    assert_allclose([c.estimate for c in r.coefficients], beta, rtol=1e-10)
    assert_allclose(r.covariance_matrix, v, rtol=1e-9, atol=1e-14)
    assert r.inference["use_t"] is small
    if small:
        dfi = 19 if kind == "cluster" else len(d) - 4
        assert r.inference["df_inference"] == dfi


def test_jive2_matches_aik_formula_with_dense_hat_matrix():
    d = jive_data(seed=5)
    q = np.column_stack([np.ones(len(d)), d.x1, d[INSTR]])
    x = np.column_stack([np.ones(len(d)), d.x1, d.d])
    hat = q @ np.linalg.solve(q.T @ q, q.T)
    h = np.diag(hat)
    xhat = (hat @ x - h[:, None] * x) / (1 - 1 / len(d))
    beta = np.linalg.solve(xhat.T @ x, xhat.T @ d.y)
    u = d.y.to_numpy() - x @ beta
    r = oe.jive(data=d, y="y", x=["x1"], endog=["d"], instruments=INSTR, method="jive2")
    assert_allclose([c.estimate for c in r.coefficients], beta, rtol=1e-10)
    assert_allclose(r.covariance_matrix, iv_sandwich(xhat, x, u, "robust"), rtol=1e-9)
    assert_allclose(r.metrics["max_leverage"], h.max(), rtol=1e-10)


def manyiv_ujive(y, d, z, w):
    """NumPy transcription of ManyIV tests/testthat/test_ujive.R::jive.fit (UJIVEold row)."""
    x = np.column_stack([z, w])
    proj = lambda a, b: a @ np.linalg.solve(a.T @ a, a.T @ b)  # noqa: E731
    dx = np.einsum("ij,ji->i", x, np.linalg.solve(x.T @ x, x.T))
    dw = np.einsum("ij,ji->i", w, np.linalg.solve(w.T @ w, w.T))
    zhat = (proj(x, d) - dx * d) / (1 - dx)
    dhat = zhat - (proj(w, d) - dw * d) / (1 - dw)
    beta = dhat @ y / (dhat @ d)
    eps = (y - proj(w, y)) - (d - proj(w, d)) * beta
    return beta, dhat, eps


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster"])
def test_ujive_matches_manyiv_transcription_and_brute_force(kind):
    d = jive_data(seed=3)
    w = np.column_stack([np.ones(len(d)), d.x1])
    z = d[INSTR].to_numpy()
    beta, dhat, eps = manyiv_ujive(d.y.to_numpy(), d.d.to_numpy(), z, w)
    # Independent route: the two leave-one-out regressions, observation by observation.
    brute = loo_prediction(np.column_stack([w, z]), d.d.to_numpy()[:, None])[:, 0] \
        - loo_prediction(w, d.d.to_numpy()[:, None])[:, 0]
    assert_allclose(brute, dhat, rtol=1e-9, atol=1e-12)
    n = len(d)
    if kind == "nonrobust":
        var = eps @ eps / n * (dhat @ dhat) / (dhat @ d.d) ** 2
    elif kind == "robust":
        var = np.sum((eps * dhat) ** 2) / (dhat @ d.d) ** 2  # ManyIV se_text
    else:
        sums = np.array([np.sum((eps * dhat)[d.g == g]) for g in range(20)])
        var = sums @ sums * 20 / 19 / (dhat @ d.d) ** 2
    r = oe.jive(data=d, y="y", x=["x1"], endog=["d"], instruments=INSTR, method="ujive",
                covariance=kind, cluster="g" if kind == "cluster" else None)
    assert [c.term for c in r.coefficients] == ["d"]
    assert_allclose(coef(r, "d").estimate, beta, rtol=1e-10)
    assert_allclose(coef(r, "d").std_error ** 2, var, rtol=1e-9)
    assert r.extra["partialled_out"] == ["Intercept", "x1"]


def test_jive1_recovers_ols_when_instruments_reproduce_the_regressor():
    d = jive_data()
    d["dcopy"] = d.d
    r = oe.jive(data=d, y="y", x=["x1"], endog=["d"], instruments=["dcopy", "z0"])
    x = np.column_stack([np.ones(len(d)), d.x1, d.d])
    ols = np.linalg.lstsq(x, d.y, rcond=None)[0]
    assert_allclose([c.estimate for c in r.coefficients], ols, rtol=1e-9)


def test_jive_failure_contract_and_persistence():
    d = jive_data()
    with pytest.raises(AnalysisError) as error:
        oe.jive(data=d, y="y", endog=["d", "d2"], instruments=INSTR, method="ujive")
    assert error.value.code == "unsupported_domain"
    lone = d.copy()
    lone["dummy"] = 0.0
    lone.loc[7, "dummy"] = 1.0  # an instrument that only one observation has: leverage one
    for method in ("jive1", "ujive"):
        with pytest.raises(AnalysisError) as error:
            oe.jive(data=lone, y="y", endog=["d"], instruments=["z0", "dummy"], method=method)
        assert error.value.code == "unit_leverage"
    jive2 = oe.jive(data=lone, y="y", endog=["d"], instruments=["z0", "dummy"], method="jive2")
    assert jive2.metrics["max_leverage"] == pytest.approx(1.0)
    with pytest.raises(AnalysisError) as error:
        oe.jive(data=d, y="y", endog=["d", "d2"], instruments=["z0"])
    assert error.value.code == "underidentified"
    with pytest.raises(AnalysisError) as error:
        oe.jive(data=d.assign(y=np.nan), y="y", endog=["d"], instruments=INSTR)
    assert error.value.code == "missing_values"
    with pytest.raises(AnalysisError) as error:
        oe.jive(data=d, y="y", endog=["d"], instruments=INSTR, method="liml")
    assert error.value.code == "invalid_spec"
    holes = d.copy()
    holes.loc[[3, 9], "z2"] = np.nan
    r = oe.jive(data=holes, y="y", x=["x1"], endog=["d"], instruments=INSTR, missing="drop")
    full = oe.jive(data=holes.dropna().reset_index(drop=True), y="y", x=["x1"], endog=["d"],
                   instruments=INSTR)
    assert r.nobs == len(d) - 2 and 3 not in r.sample_positions
    assert_allclose(r.covariance_matrix, full.covariance_matrix, rtol=1e-12)
    restored = ResultBundle.model_validate_json(r.model_dump_json())
    assert restored.covariance_matrix == r.covariance_matrix
    assert restored.extra["leave_out_unit"] == "observation"
    assert "JIVE1" in r.summary() and "\\begin{tabular}" in r.to_latex()


# ---- Lewbel generated instruments --------------------------------------------------------------


@pytest.fixture(scope="module")
def lewbel():
    payload = json.loads((FIXTURES / "lewbel.json").read_text())
    return pd.DataFrame(payload["data"]), payload


LEWBEL = {"y": "y", "x": ["x1", "x2"], "endog": ["p"]}


@pytest.mark.parametrize("name, external", [("generated", None), ("generated_external", ["zext"])])
def test_lewbel_ignored_first_step_matches_linearmodels_and_aer(lewbel, name, external):
    frame, payload = lewbel
    reference, aer = payload["python"][name], payload["r"][name]
    runs = {
        "unadjusted": ({"covariance": "nonrobust"}, reference["unadjusted"]["cov"]),
        "robust": ({}, reference["robust"]["cov"]),
        "robust_small": ({"small": True}, reference["robust_small"]["cov"]),
        "cluster_small": ({"cluster": "g", "small": True}, reference["cluster_small"]["cov"]),
        "cluster_aer_hc1": ({"cluster": "g", "small": True}, aer["cluster_hc1"]),
        "cluster_aer_hc0": ({"cluster": "g"}, aer["cluster_hc0"]),
        "robust_aer_hc0": ({}, aer["hc0"]),
    }
    for label, (options, expected) in runs.items():
        r = oe.ivlewbel(data=frame, **LEWBEL, instruments=external, first_step="ignored", **options)
        assert [c.term for c in r.coefficients] == ["Intercept", "x1", "x2", "p"]
        assert_allclose([c.estimate for c in r.coefficients], reference["robust"]["params"],
                        rtol=1e-10, err_msg=label)
        assert_allclose([c.estimate for c in r.coefficients], aer["params"], rtol=1e-10)
        assert_allclose(r.covariance_matrix, expected, rtol=1e-8, atol=1e-14, err_msg=label)
    r = oe.ivlewbel(data=frame, **LEWBEL, instruments=external, first_step="ignored")
    gmm = oe.ivlewbel(data=frame, **LEWBEL, instruments=external, method="gmm",
                      first_step="ignored")
    assert_allclose([c.estimate for c in gmm.coefficients], reference["gmm_robust"]["params"],
                    rtol=1e-9)
    assert r.extra["generated_instruments"] == ["p_x1_g", "p_x2_g"]
    assert "hansen_j" in gmm.tests and "overid_score" in r.tests


def test_lewbel_breusch_pagan_matches_statsmodels(lewbel):
    frame, payload = lewbel
    r = oe.ivlewbel(data=frame, **LEWBEL)
    bp = r.tests["lewbel_bp_p"]
    assert_allclose(bp["statistic"], payload["python"]["koenker_bp"]["statistic"], rtol=1e-10)
    assert_allclose(bp["p_value"], payload["python"]["koenker_bp"]["p_value"], rtol=1e-6)
    assert bp["df"] == 2


def _lewbel_stacked(frame, external, cluster=None):
    """Stacked first-step + 2SLS estimating equations; sandwich with numerical Jacobian."""
    n = len(frame)
    x1 = np.column_stack([np.ones(n), frame.x1, frame.x2])
    z = frame[["x1", "x2"]].to_numpy()
    ext = frame[external].to_numpy() if external else np.empty((n, 0))
    p, y = frame.p.to_numpy(), frame.y.to_numpy()
    x = np.column_stack([x1, p])

    def inst(b2, mu):
        return np.column_stack([x1, ext, (z - mu) * (p - x1 @ b2)[:, None]])

    b2, mu = np.linalg.solve(x1.T @ x1, x1.T @ p), z.mean(0)
    h = inst(b2, mu)
    fitted = h @ np.linalg.lstsq(h, x, rcond=None)[0]
    theta = np.linalg.solve(fitted.T @ x, fitted.T @ y)
    a0 = x.T @ h @ np.linalg.inv(h.T @ h)
    sizes = [3, 2, 4]

    def split(params):
        return np.split(params, np.cumsum(sizes)[:-1])

    def psi_eta(params):
        b, m, _ = split(params)
        return np.column_stack([x1 * (p - x1 @ b)[:, None], z - m])

    def moments(params):
        b, m, t = split(params)
        return inst(b, m) * (y - x @ t)[:, None]

    def psi(params):
        return np.column_stack([psi_eta(params), moments(params) @ a0.T])

    def jacobian(function, params, columns):
        out = np.empty((function(params).shape[1], len(columns)))
        for k, j in enumerate(columns):
            step = 1e-5 * max(1.0, abs(params[j]))
            up, down = params.copy(), params.copy()
            up[j] += step
            down[j] -= step
            out[:, k] = (function(up).sum(0) - function(down).sum(0)) / (2 * step)
        return out

    groups = None if cluster is None else np.unique(frame[cluster])

    def sums(rows):
        if cluster is None:
            return rows.T @ rows
        totals = np.vstack([rows[frame[cluster] == g].sum(0) for g in groups])
        return totals.T @ totals

    factor = 1.0 if cluster is None else len(groups) / (len(groups) - 1)
    params = np.concatenate([b2, mu, theta])
    inverse = np.linalg.inv(jacobian(psi, params, range(9)))
    v2sls = (inverse @ sums(psi(params)) @ inverse.T)[5:, 5:] * factor
    # Efficient GMM with the corrected moment covariance at the 2SLS residuals.
    m_eta = jacobian(moments, params, range(5))
    j_eta = jacobian(psi_eta, params, range(5))
    s = sums(moments(params) - psi_eta(params) @ np.linalg.solve(j_eta, m_eta.T))
    hx, hy = h.T @ x, h.T @ y
    bread = np.linalg.inv(hx.T @ np.linalg.solve(s, hx))
    tgmm = bread @ hx.T @ np.linalg.solve(s, hy)
    g = hy - hx @ tgmm
    return theta, v2sls, tgmm, bread * factor, float(g @ np.linalg.solve(s, g))


@pytest.mark.parametrize("external", [None, ["zext"]])
@pytest.mark.parametrize("cluster", [None, "g"])
def test_lewbel_corrected_covariance_matches_stacked_moment_sandwich(lewbel, external, cluster):
    frame, _ = lewbel
    theta, v2sls, tgmm, vgmm, j = _lewbel_stacked(frame, external, cluster)
    r = oe.ivlewbel(data=frame, **LEWBEL, instruments=external, cluster=cluster)
    assert_allclose([c.estimate for c in r.coefficients], theta, rtol=1e-10)
    assert_allclose(r.covariance_matrix, v2sls, rtol=2e-6, atol=1e-12)
    ignored = oe.ivlewbel(data=frame, **LEWBEL, instruments=external, cluster=cluster,
                          first_step="ignored")
    assert not np.allclose(r.covariance_matrix, ignored.covariance_matrix, rtol=1e-3)
    g = oe.ivlewbel(data=frame, **LEWBEL, instruments=external, cluster=cluster, method="gmm")
    assert_allclose([c.estimate for c in g.coefficients], tgmm, rtol=1e-6)
    assert_allclose(g.covariance_matrix, vgmm, rtol=2e-6, atol=1e-12)
    assert_allclose(g.metrics["j"], j, rtol=1e-5, atol=1e-9)
    assert g.tests["hansen_j"]["df"] == len(g.extra["instruments"]) - 1


def test_lewbel_two_endogenous_regressors_and_generator_subset(lewbel):
    frame, _ = lewbel
    rng = np.random.default_rng(4)
    data = frame.assign(q=frame.p * 0.3 + np.exp(0.4 * frame.x2) * rng.normal(size=len(frame)),
                        x3=rng.normal(size=len(frame)))
    r = oe.ivlewbel(data=data, y="y", x=["x1", "x2", "x3"], endog=["p", "q"],
                    generators=["x1", "x2"], first_step="ignored")
    assert r.extra["generated_instruments"] == ["p_x1_g", "p_x2_g", "q_x1_g", "q_x2_g"]
    x1 = np.column_stack([np.ones(len(data)), data.x1, data.x2, data.x3])
    targets = data[["p", "q"]].to_numpy()
    resid = targets - x1 @ np.linalg.lstsq(x1, targets, rcond=None)[0]
    zc = data[["x1", "x2"]].to_numpy() - data[["x1", "x2"]].to_numpy().mean(0)
    h = np.column_stack([x1, zc * resid[:, [0]], zc * resid[:, [1]]])
    x = np.column_stack([x1, data.p, data.q])
    fitted = h @ np.linalg.lstsq(h, x, rcond=None)[0]
    beta = np.linalg.solve(fitted.T @ x, fitted.T @ data.y)
    assert_allclose([c.estimate for c in r.coefficients], beta, rtol=1e-9)


def test_lewbel_failure_contract_and_persistence(lewbel):
    frame, _ = lewbel
    for options, code in [
        ({"covariance": "nonrobust"}, "unsupported_covariance"),
        ({"generators": ["zext"]}, "invalid_spec"),
        ({"generators": ["p"]}, "invalid_spec"),
        ({"method": "liml"}, "invalid_spec"),
        ({"first_step": "partial"}, "invalid_spec"),
    ]:
        with pytest.raises(AnalysisError) as error:
            oe.ivlewbel(data=frame, **LEWBEL, **options)
        assert error.value.code == code, options
    with pytest.raises(AnalysisError) as error:
        oe.ivlewbel(data=frame.assign(p=1 + 2 * frame.x1), **LEWBEL)
    assert error.value.code == "no_endogenous_regressors"
    r = oe.ivlewbel(data=frame, **LEWBEL, small=True)
    assert r.inference["use_t"] and r.inference["df_inference"] == len(frame) - 4
    restored = ResultBundle.model_validate_json(r.model_dump_json())
    assert restored.extra["first_step"] == "corrected"
    assert restored.covariance_matrix == r.covariance_matrix
    assert "Lewbel" in r.summary() and "\\begin{tabular}" in r.to_latex()


# ---- shift-share AKM / AKM0 ------------------------------------------------------------------


def shift_share_data(seed=8, n=120, sectors=40, strength=0.8):
    rng = np.random.default_rng(seed)
    raw = rng.gamma(0.6, size=(n, sectors)) * (rng.random((n, sectors)) < 0.5)
    raw[:, 0] += 0.05
    w = raw / raw.sum(1, keepdims=True) * rng.uniform(0.5, 1.0, size=(n, 1))  # incomplete shares
    g = rng.normal(size=sectors)
    sector_shock = rng.normal(size=sectors)
    common = w @ sector_shock                     # shock-level error component
    xs = w @ g
    units = [f"r{i:03d}" for i in range(n)]
    frame = pd.DataFrame({"unit": units, "c1": rng.normal(size=n), "pop": rng.uniform(1, 3, n),
                          "state": np.repeat(np.arange(30), n // 30)})
    frame["share_sum"] = w.sum(1)
    frame["d"] = 1 + strength * xs + 0.5 * frame.c1 + common + rng.normal(size=n)
    frame["y"] = 2 + 0.7 * frame.d - 0.3 * frame.c1 + 0.8 * common + rng.normal(size=n)
    rows, cols = np.nonzero(w)
    shares = pd.DataFrame({"unit": np.asarray(units)[rows], "sector": cols, "share": w[rows, cols]})
    shocks = pd.DataFrame({"sector": np.arange(sectors), "shock": g,
                           "group": np.arange(sectors) // 4})
    return frame, shares, shocks, w, g


def shiftshare_reference(y, w_shares, g, z, *, d=None, weights=None, region=None, sector=None,
                         beta0=0.0, alpha=0.05):
    """NumPy transcription of ShiftShareSE reg_ss.fit / ivreg_ss.fit (R/ols.R, R/iv.R)."""
    from scipy import stats

    n = len(y)
    wt = np.ones(n) if weights is None else weights
    xs = w_shares @ g
    mm = np.column_stack([xs, z])

    def wls(xm, v):
        sw = np.sqrt(wt)
        coef = np.linalg.lstsq(xm * sw[:, None], v * sw, rcond=None)[0]
        return coef, v - xm @ coef

    # ShiftShareSE drop_collinear: keep the pivot order of a rank-revealing QR on W.
    keep = []
    for j in range(w_shares.shape[1]):
        trial = w_shares[:, keep + [j]]
        if np.linalg.matrix_rank(trial, tol=1e-9 * np.linalg.norm(trial)) == len(keep) + 1:
            keep.append(j)
    wk = w_shares[:, keep]
    ddx = wls(z, xs)[1]
    hx = wls(wk, ddx)[0]
    cv = stats.norm.ppf(1 - alpha / 2)
    out = {}
    r1c, r1 = wls(mm, y)
    p = np.linalg.matrix_rank(mm)
    if d is None:
        beta, resid, ddy, exposure = r1c[0], r1, wls(z, y)[1], ddx
        rx = np.sum(wt * ddx**2)
        out["homoskedastic"] = np.sum(wt * resid**2) / (n - p) / rx
        u = wt * resid * ddx
        out["ehw"] = n / (n - p) * (u @ u) / rx**2
    else:
        r2c, r2 = wls(mm, d)
        beta = r1c[0] / r2c[0]
        resid = r1 - r2 * beta
        ddy, exposure = wls(z, y)[1], wls(z, d)[1]
        rx = np.sum(wt * exposure * ddx)
        out["homoskedastic"] = np.sum(wt * resid**2) / n / (np.sum(wt * ddx**2) * r2c[0] ** 2)
        u = wt * resid * ddx
        out["ehw"] = (u @ u) / rx**2
    if region is not None:
        labels = np.unique(region)
        sums = np.array([u[region == c].sum() for c in labels])
        factor = len(labels) / (len(labels) - 1) * (n - 1) / (n - p) if d is None else 1.0
        out["region_cluster"] = factor * (sums @ sums) / rx**2
    c_r = hx * (wk.T @ (wt * resid))
    c_r0 = hx * (wk.T @ (wt * (ddy - exposure * beta0)))
    c_w = hx * (wk.T @ (wt * exposure))
    if sector is not None:
        labels = sector[keep]
        c_r, c_r0, c_w = (np.array([v[labels == s].sum() for s in np.unique(labels)])
                          for v in (c_r, c_r0, c_w))
    out["akm"] = np.sum(c_r**2) / rx**2
    se0 = np.sqrt(np.sum(c_r0**2) / rx**2)
    q = rx**2 / cv**2 - np.sum(c_w**2)
    q2 = np.sum(c_r * c_w) / q
    mid = beta - q2
    dis = q2**2 + np.sum(c_r**2) / q
    return beta, out, se0, (q, mid, dis), keep


SS = {"y": "y", "unit": "unit"}


@pytest.mark.parametrize("model", ["ols", "iv"])
@pytest.mark.parametrize("weighted", [False, True])
def test_shift_share_matches_shiftshareSE_transcription(model, weighted):
    from scipy import stats

    frame, shares, shocks, w, g = shift_share_data(strength=3.0)
    z = np.column_stack([np.ones(len(frame)), frame.c1, frame.share_sum])
    beta, var, se0, (q, mid, dis), keep = shiftshare_reference(
        frame.y.to_numpy(), w, g, z, d=frame.d.to_numpy() if model == "iv" else None,
        weights=frame["pop"].to_numpy() if weighted else None, region=frame.state.to_numpy(),
        sector=shocks.group.to_numpy(), beta0=0.25)
    out = oe.shift_share_iv(data=frame, **SS, shares=shares, shocks=shocks,
                            endog="d" if model == "iv" else None, x=["c1", "share_sum"],
                            weights="pop" if weighted else None, cluster="state",
                            shock_cluster="group", beta0=0.25)
    table = out["inference"].set_index("method")
    assert out.attrs["model"] == model and len(keep) == 40
    assert_allclose(table.estimate, beta, rtol=1e-10)
    for method in ("homoskedastic", "ehw", "region_cluster", "akm"):
        assert_allclose(table.loc[method, "std_error"] ** 2, var[method], rtol=1e-9)
        se = np.sqrt(var[method])
        assert_allclose(table.loc[method, "p_value"],
                        2 * stats.norm.sf(abs(beta - 0.25) / se), rtol=1e-8)
    assert_allclose(table.loc["akm0", "p_value"], 2 * stats.norm.sf(abs(beta - 0.25) / se0),
                    rtol=1e-8)
    assert q > 0 and table.loc["akm0", "ci_type"] == "interval"
    assert_allclose([table.loc["akm0", "ci_low"], table.loc["akm0", "ci_high"]],
                    [mid - np.sqrt(dis), mid + np.sqrt(dis)], rtol=1e-9)
    assert_allclose(table.loc["akm0", "std_error"], np.sqrt(dis) / stats.norm.ppf(0.975),
                    rtol=1e-9)
    assert out.attrs["shock_clusters"] == 10 and out.attrs["region_clusters"] == 30
    assert not any("Share sums vary" in note for note in out.attrs["notes"])


def test_shift_share_akm_without_shock_clusters_and_unbounded_akm0():
    frame, shares, shocks, w, g = shift_share_data(seed=21, strength=0.02)
    z = np.column_stack([np.ones(len(frame)), frame.c1])
    beta, var, se0, (q, mid, dis), _ = shiftshare_reference(
        frame.y.to_numpy(), w, g, z, d=frame.d.to_numpy())
    out = oe.shift_share_iv(data=frame, **SS, shares=shares, shocks=shocks, endog="d", x=["c1"])
    table = out["inference"].set_index("method")
    assert_allclose(table.loc["akm", "std_error"] ** 2, var["akm"], rtol=1e-9)
    assert q < 0 and dis > 0
    assert table.loc["akm0", "ci_type"] == "complement_of_interval"
    assert np.isnan(table.loc["akm0", "std_error"])
    (low_ray, high_ray) = out.attrs["akm0_set"]
    assert low_ray[0] is None and high_ray[1] is None
    assert_allclose([low_ray[1], high_ray[0]], [mid - np.sqrt(dis), mid + np.sqrt(dis)], rtol=1e-9)
    assert any("Share sums vary" in note for note in out.attrs["notes"])
    assert "region_cluster" not in set(table.index)


def test_shift_share_one_sector_per_region_akm_is_sector_clustering():
    rng = np.random.default_rng(3)
    n, sectors = 90, 30
    member = np.repeat(np.arange(sectors), 3)
    g = rng.normal(size=sectors)
    frame = pd.DataFrame({"unit": np.arange(n), "y": rng.normal(size=n) + g[member]})
    shares = pd.DataFrame({"unit": np.arange(n), "sector": member, "share": 1.0})
    shocks = pd.DataFrame({"sector": np.arange(sectors), "shock": g})
    out = oe.shift_share_iv(data=frame, **SS, shares=shares, shocks=shocks)
    xs = g[member]
    ddx = xs - xs.mean()
    resid = frame.y - frame.y.mean() - ddx * (ddx @ frame.y) / (ddx @ ddx)
    sums = np.array([np.sum((ddx * resid)[member == s]) for s in range(sectors)])
    table = out["inference"].set_index("method")
    assert_allclose(table.loc["akm", "std_error"] ** 2, sums @ sums / (ddx @ ddx) ** 2, rtol=1e-9)
    assert out.attrs["effective_shock_count"] == pytest.approx(sectors)


def test_shift_share_collinear_sectors_instrument_table_and_failures():
    frame, shares, shocks, w, g = shift_share_data(seed=5)
    twin = shares[shares.sector == 3].assign(sector=99)
    shares2 = pd.concat([shares, twin], ignore_index=True)
    shocks2 = pd.concat([shocks, pd.DataFrame({"sector": [99], "shock": [0.4], "group": [9]})],
                        ignore_index=True)
    w2 = np.column_stack([w, w[:, 3]])
    g2 = np.append(g, 0.4)
    z = np.column_stack([np.ones(len(frame)), frame.c1])
    beta, var, _, _, keep = shiftshare_reference(frame.y.to_numpy(), w2, g2, z,
                                                 d=frame.d.to_numpy())
    out = oe.shift_share_iv(data=frame, **SS, shares=shares2, shocks=shocks2, endog="d", x=["c1"])
    assert keep == list(range(40)) and out.attrs["omitted_sectors"] == ["99"]
    assert_allclose(out["inference"].set_index("method").loc["akm", "std_error"] ** 2, var["akm"],
                    rtol=1e-9)
    bartik = oe.shift_share_instrument(shares=shares, shocks=shocks)
    merged = shares.merge(shocks, on="sector")
    expected = (merged.share * merged.shock).groupby(merged.unit, sort=False).sum()
    assert_allclose(bartik.set_index("unit").loc[expected.index, "shift_share"], expected,
                    rtol=1e-12)
    cases = [
        (dict(data=pd.concat([frame, frame.iloc[:1]])), "duplicate_units"),
        (dict(shares=shares[shares.unit != "r005"]), "missing_shares"),
        (dict(shocks=shocks[shocks.sector != 7]), "missing_shocks"),
        (dict(shares=shares.assign(share=-shares.share)), "negative_shares"),
        (dict(shares=pd.concat([shares, shares.iloc[:1]])), "duplicate_shares"),
        (dict(data=frame.assign(y=np.where(np.arange(len(frame)) == 2, np.nan, frame.y))),
         "missing_values"),
        (dict(shocks=shocks.assign(shock=1.0), x=["c1", "share_sum"]), "collinear_shift_share"),
        (dict(alpha=1.5), "invalid_option"),
        (dict(max_cells=10), "resource_limit"),
    ]
    for override, code in cases:
        options = dict(data=frame, **SS, shares=shares, shocks=shocks, endog="d", x=["c1"])
        options.update(override)
        with pytest.raises(AnalysisError) as error:
            oe.shift_share_iv(**options)
        assert error.value.code == code, override
    holes = frame.assign(y=np.where(np.arange(len(frame)) == 2, np.nan, frame.y))
    dropped = oe.shift_share_iv(data=holes, **SS, shares=shares, shocks=shocks, endog="d",
                                x=["c1"], missing="drop")
    assert dropped.attrs["nobs"] == len(frame) - 1 and 2 not in dropped.attrs["sample_positions"]
    assert dropped.attrs["unmatched_share_rows"] == int((shares.unit == "r002").sum())
    restored = oe.restore_summary(oe.summary_state(dropped))
    assert restored.attrs["akm0_set"] == dropped.attrs["akm0_set"]
    pd.testing.assert_frame_equal(restored["inference"], dropped["inference"], check_dtype=False)
    assert "AKM" in str(dropped) and "tabular" in dropped.to_latex()


# ---- Montiel Olea-Pflueger weak-instrument test -------------------------------------------------


def mop_data(seed=0, n=400, k=4):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k)) * np.linspace(1, 3, k)
    v = rng.normal(size=n) * (1 + np.abs(z[:, 0]))
    frame = pd.DataFrame(z, columns=[f"z{j}" for j in range(k)])
    frame["c"] = rng.normal(size=n)
    frame["x"] = z @ np.linspace(0.12, 0.0, k) + 0.3 * frame.c + v
    frame["y"] = 1 + frame.x + 0.6 * v + rng.normal(size=n) * (1 + 0.5 * np.abs(z[:, -1]))
    frame["g"] = np.repeat(np.arange(n // 8), 8)
    frame["t"] = np.arange(n)
    return frame


def weakivtest_reference(frame, k, kind, lags=None, taus=(0.05, 0.1, 0.2, 0.3), alpha=0.05):
    """NumPy/SciPy transcription of weakivtest.ado (avar W, Bmaxfunction, BmaxLIML, cpatnaikgen)."""
    from scipy import optimize, stats

    names = [f"z{j}" for j in range(k)]
    n = len(frame)
    c = np.column_stack([np.ones(n), frame.c])
    part = lambda a: a - c @ np.linalg.lstsq(c, a, rcond=None)[0]  # noqa: E731
    zp, xp, yp = part(frame[names].to_numpy()), part(frame.x.to_numpy()), part(frame.y.to_numpy())
    vals, vecs = np.linalg.eigh(zp.T @ zp / n)       # symmetric orthonormalization (Zs'Zs = nI)
    zs = zp @ vecs @ np.diag(vals ** -0.5) @ vecs.T
    fit = lambda a: zs @ np.linalg.solve(zs.T @ zs, zs.T @ a)  # noqa: E731
    v1, v2 = xp - fit(xp), yp - fit(yp)
    df = n - k - 2
    omega = np.column_stack([v2, v1]).T @ np.column_stack([v2, v1]) / df
    rows = np.column_stack([zs * v2[:, None], zs * v1[:, None]])
    if kind == "nonrobust":
        u = np.column_stack([v2, v1])
        s = np.kron(u.T @ u / n, zs.T @ zs / n)
    elif kind == "robust":
        s = rows.T @ rows / n
    elif kind == "cluster":
        sums = np.vstack([rows[frame.g == g].sum(0) for g in np.unique(frame.g)])
        s = sums.T @ sums / n
    else:
        s = rows.T @ rows
        for lag in range(1, lags + 1):
            gamma = rows[lag:].T @ rows[:-lag]
            s += (1 - lag / (lags + 1)) * (gamma + gamma.T)
        s /= n
    w = s * n / df
    w1, w12, w2 = w[:k, :k], w[:k, k:], w[k:, k:]
    groups = len(np.unique(frame.g))
    adjust = n / (n - 1) * (groups - 1) / groups if kind == "cluster" else 1.0
    f_cd = (fit(xp) @ fit(xp) / k) / (v1 @ v1 / df)
    f_eff = adjust * f_cd * k * omega[1, 1] / np.trace(w2)

    def bmax_tsls(beta):
        s12, s1 = w12 - beta * w2, w1 - 2 * beta * w12 + beta**2 * w2
        eig = np.linalg.eigvalsh(0.5 * (s12 + s12.T))
        den = np.sqrt(np.trace(w2) * np.trace(s1))
        return max(abs(np.trace(s12) - 2 * eig[0]), abs(np.trace(s12) - 2 * eig[-1])) / den

    def bmax_liml(beta):
        s12, s1 = w12 - beta * w2, w1 - 2 * beta * w12 + beta**2 * w2
        sig12 = omega[0, 1] - beta * omega[1, 1]
        sig1 = omega[0, 0] - 2 * beta * omega[0, 1] + beta**2 * omega[1, 1]
        matrix = 2 * s12 - sig12 / sig1 * s1
        eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        lead = np.trace(s12) - sig12 / sig1 * np.trace(s1)
        den = np.sqrt(np.trace(w2) * np.trace(s1))
        return max(abs(lead - eig[0]), abs(lead - eig[-1])) / den

    def supremum(function):
        grid = np.concatenate([np.sinh(np.linspace(-14, 14, 20001)), [-1e15, 1e15]])
        values = np.array([function(b) for b in grid])
        j = int(np.argmax(values[:-2]))
        best = max(values)
        lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 3)]
        refined = optimize.minimize_scalar(lambda b: -function(b), bounds=(lo, hi),
                                           method="bounded", options={"xatol": 1e-12})
        return max(best, -refined.fun)

    def cpatnaikgen(x):
        lam = np.sort(np.linalg.eigvalsh(w2))
        lam = lam / lam.sum()
        ek = 2 * (1 + 2 * x) / (2 * np.sum(lam**2) + 4 * x * lam.max())
        return stats.ncx2.ppf(1 - alpha, ek, ek * x) / ek, ek

    b_tsls, b_liml = supremum(bmax_tsls), supremum(bmax_liml)
    critical = {(name, t): cpatnaikgen(bound / t)
                for t in taus for name, bound in (("tsls", b_tsls), ("liml", b_liml),
                                                  ("simplified", 1.0))}
    return f_eff, b_tsls, b_liml, critical


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster", "hac"])
def test_weakivtest_matches_transcribed_stata_algorithm(kind):
    frame = mop_data()
    options = {"cluster": "g"} if kind == "cluster" else (
        {"time": "t", "lags": 3} if kind == "hac" else {})
    f_eff, b_tsls, b_liml, critical = weakivtest_reference(frame, 4, kind, lags=3)
    out = oe.weakivtest(data=frame, y="y", x=["c"], endog=["x"],
                        instruments=[f"z{j}" for j in range(4)], covariance=kind, **options)
    assert_allclose(out.attrs["effective_f"], f_eff, rtol=1e-10)
    assert_allclose([out.attrs["b_tsls"], out.attrs["b_liml"]], [b_tsls, b_liml], rtol=1e-8)
    assert out.attrs["b_tsls"] <= 1 + 1e-12
    details = out["details"].set_index(["estimator", "tau"])
    for (name, t), (value, k_eff) in critical.items():
        assert_allclose(details.loc[(name, t), "critical_value"], value, rtol=1e-8)
        assert_allclose(details.loc[(name, t), "k_eff"], k_eff, rtol=1e-8)
        assert details.loc[(name, t), "reject_weak"] == (f_eff > value)
    plain = oe.effective_f(data=frame, y="y", x=["c"], endog=["x"],
                           instruments=[f"z{j}" for j in range(4)], covariance=kind, **options)
    assert_allclose(out.attrs["effective_f_unadjusted"],
                    plain["diagnostic"].effective_f.iloc[0], rtol=1e-10)


def test_weakivtest_homoskedastic_one_instrument_and_scale_invariance():
    from scipy import stats

    frame = mop_data(seed=4)
    names = [f"z{j}" for j in range(4)]
    iid = oe.weakivtest(data=frame, y="y", x=["c"], endog=["x"], instruments=names,
                        covariance="nonrobust")
    cragg = oe.ivregress(data=frame, y="y", x=["c"], endog=["x"], instruments=names)
    assert_allclose(iid.attrs["effective_f"], cragg.tests["cragg_donald"]["statistic"], rtol=1e-10)
    scaled = oe.weakivtest(data=frame.assign(z1=1e3 * frame.z1, z2=frame.z2 - 5 * frame.z3),
                           y="y", x=["c"], endog=["x"], instruments=names)
    base = oe.weakivtest(data=frame, y="y", x=["c"], endog=["x"], instruments=names)
    assert_allclose(scaled["details"].critical_value, base["details"].critical_value, rtol=1e-8)
    assert_allclose(scaled.attrs["effective_f"], base.attrs["effective_f"], rtol=1e-10)
    single = oe.weakivtest(data=frame, y="y", x=["c"], endog=["x"], instruments=["z0"],
                           tau=[0.05, 0.1, 0.2])
    assert_allclose([single.attrs["b_tsls"], single.attrs["b_liml"]], [1.0, 1.0], rtol=1e-12)
    table = single["critical_values"].set_index("tau")
    # Windmeijer (2025, Sec. 4): with one instrument c = upper-alpha quantile of chi2_1(1/tau).
    for t in (0.05, 0.1, 0.2):
        expected = stats.ncx2.ppf(0.95, 1, 1 / t)
        assert_allclose(table.loc[t, ["tsls", "liml", "simplified"]], expected, rtol=1e-10)
    assert_allclose(table.loc[0.1, "tsls"], 23.1085, atol=5e-5)


def test_weakivtest_failure_contract_and_persistence():
    frame = mop_data()
    names = [f"z{j}" for j in range(4)]
    frame["x2"] = frame.x + frame.z1
    for options, code in [
        ({"endog": ["x", "x2"]}, "unsupported_target"),
        ({"tau": [0.0]}, "invalid_option"),
        ({"tau": "abc"}, "invalid_option"),
        ({"alpha": 2.0}, "invalid_option"),
        ({"covariance": "hac", "time": "t"}, "invalid_spec"),
        ({"covariance": "HC3"}, "unsupported_covariance"),
    ]:
        arguments = {"data": frame, "y": "y", "x": ["c"], "endog": ["x"], "instruments": names}
        arguments.update(options)
        with pytest.raises(AnalysisError) as error:
            oe.weakivtest(**arguments)
        assert error.value.code == code, options
    out = oe.weakivtest(data=frame, y="y", x=["c"], endog=["x"], instruments=names,
                        tau=[1 / 3.33])
    restored = oe.restore_summary(oe.summary_state(out))
    assert restored.attrs["w_matrix"] == out.attrs["w_matrix"]
    pd.testing.assert_frame_equal(restored["details"], out["details"], check_dtype=False)
    assert "effective_f" in str(out) and "tabular" in out.to_latex()
