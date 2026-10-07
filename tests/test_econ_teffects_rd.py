"""Independent NumPy oracle for rdrobust / rdbwselect.

The oracle re-implements the rdrobust R functions in plain loop-style NumPy:
rdrobust_kweight, rdrobust_res (the per-observation nearest-neighbour loop with
duplicate blocks, HC0-HC3), rdrobust_vce (with clusters), rdrobust_bw and the
mserd / msetwo / msesum / comb / cer bandwidth rules, and the main
conventional / bias-corrected / robust estimator for sharp and fuzzy designs.
A simulation checks the coverage of the robust bias-corrected interval.
"""

import json
import time

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle

# ---- oracle -----------------------------------------------------------------------


def kweight(x, c, h, kernel):
    u = (x - c) / h
    inside = np.abs(u) <= 1
    if kernel == "epanechnikov":
        return 0.75 * (1 - u ** 2) * inside / h
    if kernel == "uniform":
        return 0.5 * inside / h
    return (1 - np.abs(u)) * inside / h


def qrinv(a):
    return np.linalg.inv(a.T @ a)


def residuals(X, Y, fitted, hii, vce, matches, d):
    n = len(X)
    if vce == "nn":
        res = np.empty_like(Y)
        for pos in range(n):
            dups = int(np.sum(X == X[pos]))
            dupsid = int(np.sum(X[:pos + 1] == X[pos]))
            rpos, lpos = dups - dupsid, dupsid - 1
            while lpos + rpos < min(matches, n - 1):
                if pos - lpos - 1 < 0:
                    rpos += int(np.sum(X == X[pos + rpos + 1]))
                elif pos + rpos + 1 >= n:
                    lpos += int(np.sum(X == X[pos - lpos - 1]))
                else:
                    dl=X[pos]-X[pos-lpos-1]
                    dr=X[pos+rpos+1]-X[pos]
                    tolerance=max(dl,dr)*np.sqrt(np.finfo(float).eps)
                    if dl-dr>tolerance:
                        rpos+=int(np.sum(X==X[pos+rpos+1]))
                    elif dr-dl>tolerance:
                        lpos+=int(np.sum(X==X[pos-lpos-1]))
                    else:
                        rpos+=int(np.sum(X==X[pos+rpos+1]))
                        lpos+=int(np.sum(X==X[pos-lpos-1]))
            window = np.arange(pos - lpos, pos + rpos + 1)
            j = len(window) - 1
            res[pos] = np.sqrt(j / (j + 1)) * (Y[pos] - (Y[window].sum(0) - Y[pos]) / j)
        return res
    factor = {"hc0": np.ones(n), "hc1": np.full(n, np.sqrt(n / (n - d))),
              "hc2": None, "hc3": None}[vce]
    if vce == "hc2":
        factor = np.sqrt(1 / (1 - hii))
    elif vce == "hc3":
        factor = 1 / (1 - hii)
    return factor[:, None] * (Y - fitted)


def vce_meat(rx, res, s, cluster):
    scalar = res @ s
    if cluster is None:
        scores = rx * scalar[:, None]
        return scores.T @ scores
    groups = np.unique(cluster)
    g, (n, k) = len(groups), rx.shape
    meat = np.zeros((k, k))
    for label in groups:
        t = (rx[cluster == label] * scalar[cluster == label, None]).sum(0)
        meat += np.outer(t, t)
    return (n - 1) / (n - k) * g / (g - 1) * meat


def poly(dx, order):
    return np.column_stack([dx ** j for j in range(order + 1)])


def leverage(R, inv, w):
    return np.array([R[i] @ inv @ (R[i] * w[i]) for i in range(len(R))])


def bw_constants(X, Y, C, c, o, nu, o_b, h_v, h_b, scale, vce, matches, kernel):
    w = kweight(X, c, h_v, kernel)
    ind = w > 0
    eX, eY, eW = X[ind], Y[ind], w[ind]
    R = poly(eX - c, o)
    inv = qrinv(R * np.sqrt(eW)[:, None])
    beta = inv @ (R * eW[:, None]).T @ eY
    hii = leverage(R, inv, eW) if vce in ("hc2", "hc3") else None
    res = residuals(eX, eY, R @ beta, hii, vce, matches, o + 1)
    s = np.zeros(Y.shape[1])
    s[0] = 1
    v_v = (inv @ vce_meat(R * eW[:, None], res, s, None if C is None else C[ind]) @ inv)[nu, nu]
    v = (R * eW[:, None]).T @ ((eX - c) / h_v) ** (o + 1)
    b_const = (h_v ** np.arange(o + 1) * (inv @ v))[nu]
    w = kweight(X, c, h_b, kernel)
    ind = w > 0
    eX, eY, eW = X[ind], Y[ind], w[ind]
    R = poly(eX - c, o_b)
    inv_b = qrinv(R * np.sqrt(eW)[:, None])
    beta_b = inv_b @ (R * eW[:, None]).T @ eY
    reg = 0.0
    if scale > 0:
        hii = leverage(R, inv_b, eW) if vce in ("hc2", "hc3") else None
        res = residuals(eX, eY, R @ beta_b, hii, vce, matches, o_b + 1)
        v_b = (inv_b @ vce_meat(R * eW[:, None], res, s, None if C is None else C[ind])
               @ inv_b)[o + 1, o + 1]
        reg = 3 * b_const ** 2 * v_b
    B = np.sqrt(2 * (o + 1 - nu)) * b_const * (beta_b[o + 1] @ s)
    V = (2 * nu + 1) * h_v ** (2 * nu + 1) * v_v
    return V, B, scale * 2 * (o + 1 - nu) * reg, 1 / (2 * o + 3)


def quantile2(x, prob):
    s, n = np.sort(x), len(x)
    g = n * prob
    j = int(np.floor(g))
    return s[j] if g - j > 1e-12 else (s[j - 1] + s[j]) / 2


def bandwidths(x, Y, C, c, p, q, kernel, vce, matches, method, scale=1.0):
    order = np.argsort(x, kind="stable")
    x, Y = x[order], Y[order]
    C = None if C is None else C[order]
    left, right = x < c, x >= c
    sides = [(x[left], Y[left], None if C is None else C[left]),
             (x[right], Y[right], None if C is None else C[right])]
    n = len(x)
    C_c = {"triangular": 2.576, "epanechnikov": 2.34, "uniform": 1.843}[kernel]
    sd = x.std(ddof=1)
    iqr = quantile2(x, 0.75) - quantile2(x, 0.25)
    c_bw = C_c * min(sd, iqr / 1.349) * n ** -0.2
    bw_max_l, bw_max_r = abs(c - x.min()), abs(c - x.max())
    bw_max = max(bw_max_l, bw_max_r)
    c_bw = min(c_bw, bw_max)
    ranges = (abs(c - x[left].min()) + 1e-8, abs(c - x[right].max()) + 1e-8)

    def const(side, o, nu, o_b, h_b, sc):
        X, YY, CC = sides[side]
        return bw_constants(X, YY, CC, c, o, nu, o_b, c_bw, h_b, sc, vce, matches, kernel)

    dl = const(0, q + 1, q + 1, q + 2, ranges[0], 0)
    dr = const(1, q + 1, q + 1, q + 2, ranges[1], 0)
    out = {}
    for name, sign in (("mserd", -1), ("msesum", 1)):
        d = min(((dl[0] + dr[0]) / (dr[1] + sign * dl[1]) ** 2) ** dl[3], bw_max)
        bl, br = const(0, q, p + 1, q + 1, d, scale), const(1, q, p + 1, q + 1, d, scale)
        b = min(((bl[0] + br[0]) / ((br[1] + sign * bl[1]) ** 2 + scale * (bl[2] + br[2])))
                ** bl[3], bw_max)
        hl, hr = const(0, p, 0, q, b, scale), const(1, p, 0, q, b, scale)
        h = min(((hl[0] + hr[0]) / ((hr[1] + sign * hl[1]) ** 2 + scale * (hl[2] + hr[2])))
                ** hl[3], bw_max)
        out[name] = ((h, h), (b, b))
    caps = (bw_max_l, bw_max_r)
    d = [min((k[0] / k[1] ** 2) ** k[3], caps[i]) for i, k in enumerate((dl, dr))]
    bk = [const(i, q, p + 1, q + 1, d[i], scale) for i in (0, 1)]
    b = [min((k[0] / (k[1] ** 2 + scale * k[2])) ** k[3], caps[i]) for i, k in enumerate(bk)]
    hk = [const(i, p, 0, q, b[i], scale) for i in (0, 1)]
    h = [min((k[0] / (k[1] ** 2 + scale * k[2])) ** k[3], caps[i]) for i, k in enumerate(hk)]
    out["msetwo"] = (tuple(h), tuple(b))
    out["msecomb1"] = tuple((min(out["mserd"][j][0], out["msesum"][j][0]),) * 2 for j in (0, 1))
    out["msecomb2"] = tuple(tuple(sorted([out["mserd"][j][s], out["msesum"][j][s],
                                          out["msetwo"][j][s]])[1] for s in (0, 1))
                            for j in (0, 1))
    base = method.replace("cer", "mse")
    hh, bb = out[base]
    if method.startswith("cer"):
        f = n ** (-(p / ((3 + p) * (3 + 2 * p))))
        hh = (hh[0] * f, hh[1] * f)
    return hh, bb


def rd_oracle(x, y, c, h, b, p=1, q=2, kernel="triangular", vce="nn", matches=3, t=None,
              cluster=None):
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    Y = y[:, None] if t is None else np.column_stack([y, t[order]])
    C = None if cluster is None else cluster[order]
    fits = []
    for side, mask in enumerate((x < c, x >= c)):
        X, YY = x[mask], Y[mask]
        CC = None if C is None else C[mask]
        w_h, w_b = kweight(X, c, h[side], kernel), kweight(X, c, b[side], kernel)
        ind = (w_b > 0) if b[side] >= h[side] else (w_h > 0)
        eX, eY, Wh, Wb = X[ind], YY[ind], w_h[ind], w_b[ind]
        u = (eX - c) / h[side]
        Rq = poly(eX - c, q)
        Rp = Rq[:, :p + 1]
        L = (Rp * Wh[:, None]).T @ u ** (p + 1)
        inv_q, inv_p = qrinv(Rq * np.sqrt(Wb)[:, None]), qrinv(Rp * np.sqrt(Wh)[:, None])
        e = np.zeros(q + 1)
        e[p + 1] = 1
        projection = (Rq @ inv_q * Wb[:, None]).T
        Q = ((Rp * Wh[:, None]).T - h[side] ** (p + 1) * np.outer(L, e) @ projection).T
        beta_p = inv_p @ (Rp * Wh[:, None]).T @ eY
        beta_q = inv_q @ (Rq * Wb[:, None]).T @ eY
        beta_bc = inv_p @ Q.T @ eY
        hii = leverage(Rp, inv_p, Wh) if vce in ("hc2", "hc3") else None
        res_h = residuals(eX, eY, Rp @ beta_p, hii, vce, matches, p + 1)
        res_b = res_h if vce == "nn" else residuals(eX, eY, Rq @ beta_q, hii, vce, matches, q + 1)
        fits.append((beta_p, beta_bc, inv_p, Rp * Wh[:, None], Q, res_h, res_b,
                     None if CC is None else CC[ind]))
    jp = fits[1][0][0] - fits[0][0][0]
    jbc = fits[1][1][0] - fits[0][1][0]
    if t is None:
        s = np.array([1.0])
        tau_cl, tau_bc = jp[0], jbc[0]
    else:
        s = np.array([1 / jp[1], -jp[0] / jp[1] ** 2])
        tau_cl = jp[0] / jp[1]
        tau_bc = tau_cl - s @ (jp - jbc)
    v_cl = sum((f[2] @ vce_meat(f[3], f[5], s, f[7]) @ f[2])[0, 0] for f in fits)
    v_rb = sum((f[2] @ vce_meat(f[4], f[6], s, f[7]) @ f[2])[0, 0] for f in fits)
    return tau_cl, tau_bc, np.sqrt(v_cl), np.sqrt(v_rb)


def make_rd(seed=1, n=500, ties=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    if ties:
        x = np.round(x, 2)
    y = 0.5 + x - 0.6 * x ** 2 + 0.8 * (x >= 0) + rng.normal(scale=0.4, size=n)
    t = (np.where(x >= 0, rng.uniform(size=n) < 0.8, rng.uniform(size=n) < 0.15)).astype(float)
    yf = 0.3 + x + 1.2 * t + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x": x, "t": t, "yf": yf, "cl": rng.integers(0, 60, n)})


@pytest.fixture(scope="module")
def rd():
    return make_rd()


def rows(result):
    c = result.coefficients
    return np.array([c[0].estimate, c[1].estimate, c[0].std_error, c[2].std_error])


@pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
@pytest.mark.parametrize("vce", ["nn", "hc0", "hc1", "hc2", "hc3"])
def test_estimates_match_oracle_with_given_bandwidths(rd, kernel, vce):
    result = oe.rdrobust(data=rd, y="y", running="x", h=0.35, b=0.6, kernel=kernel, vce=vce)
    expected = rd_oracle(rd.x.to_numpy(), rd.y.to_numpy(), 0.0, (0.35, 0.35), (0.6, 0.6),
                         kernel=kernel, vce=vce)
    assert_allclose(rows(result), expected, rtol=1e-9)
    assert [c.term for c in result.coefficients] == ["Conventional", "Bias-corrected", "Robust"]
    assert result.coefficients[1].std_error == result.coefficients[0].std_error


def test_quadratic_asymmetric_bandwidths_fuzzy_and_clusters(rd):
    result = oe.rdrobust(data=rd, y="y", running="x", p=2, q=3, h=[0.5, 0.4], b=[0.8, 0.7],
                         cutoff=0.05)
    expected = rd_oracle(rd.x.to_numpy(), rd.y.to_numpy(), 0.05, (0.5, 0.4), (0.8, 0.7), p=2, q=3)
    assert_allclose(rows(result), expected, rtol=1e-9)
    fuzzy = oe.rdrobust(data=rd, y="yf", running="x", fuzzy="t", h=0.4, rho=0.8)
    expected = rd_oracle(rd.x.to_numpy(), rd.yf.to_numpy(), 0.0, (0.4, 0.4), (0.5, 0.5),
                         t=rd.t.to_numpy())
    assert_allclose(rows(fuzzy), expected, rtol=1e-9)
    assert fuzzy.extra["design"] == "fuzzy" and "first_stage" in fuzzy.extra
    for vce in ("nn", "hc1"):
        clustered = oe.rdrobust(data=rd, y="y", running="x", h=0.4, b=0.6, vce=vce,
                                covariance="cluster", cluster="cl")
        expected = rd_oracle(rd.x.to_numpy(), rd.y.to_numpy(), 0.0, (0.4, 0.4), (0.6, 0.6),
                             vce=vce, cluster=rd.cl.to_numpy())
        assert_allclose(rows(clustered), expected, rtol=1e-9)


def test_tied_running_values_use_duplicate_blocks():
    data = make_rd(seed=3, n=400, ties=True)
    assert data.x.duplicated().sum() > 100
    result = oe.rdrobust(data=data, y="y", running="x", h=0.3, b=0.5)
    expected = rd_oracle(data.x.to_numpy(), data.y.to_numpy(), 0.0, (0.3, 0.3), (0.5, 0.5))
    assert_allclose(rows(result), expected, rtol=1e-9)


@pytest.mark.parametrize("method", ["mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
                                    "cerrd", "certwo"])
def test_bandwidth_selection_matches_oracle(rd, method):
    result = oe.rdrobust(data=rd, y="y", running="x", bwselect=method)
    h, b = bandwidths(rd.x.to_numpy(), rd.y.to_numpy()[:, None], None, 0.0, 1, 2, "triangular",
                      "nn", 3, method)
    got = [result.metrics[k] for k in ("h_left", "h_right", "b_left", "b_right")]
    assert_allclose(got, [h[0], h[1], b[0], b[1]], rtol=1e-8)
    expected = rd_oracle(rd.x.to_numpy(), rd.y.to_numpy(), 0.0, h, b)
    assert_allclose(rows(result), expected, rtol=1e-7)


def test_bandwidth_selection_other_kernels_vce_and_orders(rd):
    for kernel, vce, p in (("epanechnikov", "hc1", 1), ("uniform", "hc3", 1),
                           ("triangular", "nn", 2)):
        result = oe.rdrobust(data=rd, y="y", running="x", kernel=kernel, vce=vce, p=p)
        h, b = bandwidths(rd.x.to_numpy(), rd.y.to_numpy()[:, None], None, 0.0, p, p + 1,
                          kernel, vce, 3, "mserd")
        assert_allclose([result.metrics["h_left"], result.metrics["b_left"]], [h[0], b[0]],
                        rtol=1e-8)
    clustered = oe.rdrobust(data=rd, y="y", running="x", covariance="cluster", cluster="cl")
    h, b = bandwidths(rd.x.to_numpy(), rd.y.to_numpy()[:, None], rd.cl.to_numpy(), 0.0, 1, 2,
                      "triangular", "nn", 3, "mserd")
    assert_allclose([clustered.metrics["h_left"], clustered.metrics["b_left"]], [h[0], b[0]],
                    rtol=1e-8)


def test_robust_interval_covers_in_simulation():
    rng = np.random.default_rng(21)
    covered, estimates = [], []
    for _ in range(150):
        n = 800
        x = rng.uniform(-1, 1, n)
        y = 0.5 + x - 0.6 * x ** 2 + 1.5 * x ** 3 * (x >= 0) + 0.8 * (x >= 0) \
            + rng.normal(scale=0.4, size=n)
        fit = oe.rdrobust(data=pd.DataFrame({"y": y, "x": x}), y="y", running="x")
        robust = fit.coefficients[2]
        covered.append(robust.ci_low <= 0.8 <= robust.ci_high)
        estimates.append(robust.estimate)
    assert np.mean(covered) > 0.88
    assert abs(np.mean(estimates) - 0.8) < 0.03


def test_failure_contract(rd):
    cases = [
        ({"cutoff": 5.0}, "invalid_cutoff"),
        ({"p": 2, "q": 2}, "invalid_spec"),
        ({"b": 0.5}, "invalid_spec"),
        ({"h": -1.0}, "invalid_spec"),
        ({"h": [0.3, 0.2, 0.1]}, "invalid_spec"),
        ({"h": 0.3, "b": 0.4, "rho": 1.0}, "invalid_spec"),
        ({"h": 0.01}, "insufficient_observations"),
        ({"kernel": "gaussian"}, "invalid_spec"),
        ({"covariance": "nonrobust"}, "invalid_spec"),
    ]
    for update, code in cases:
        arguments = {"data": rd, "y": "y", "running": "x", **update}
        with pytest.raises(AnalysisError) as caught:
            oe.rdrobust(**arguments)
        assert caught.value.code == code, update
    flat = rd.assign(t=0.0)
    with pytest.raises(AnalysisError) as caught:
        oe.rdrobust(data=flat, y="yf", running="x", fuzzy="t", h=0.4)
    assert caught.value.code == "weak_first_stage"
    missing = rd.copy()
    missing.loc[[0, 7], "y"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.rdrobust(data=missing, y="y", running="x", h=0.4)
    assert caught.value.code == "missing_values"
    dropped = oe.rdrobust(data=missing, y="y", running="x", h=0.4, missing="drop")
    assert dropped.nobs == len(rd) - 2


def test_serialization_rendering_and_dense_large_sample(rd, monkeypatch):
    result = oe.rdrobust(data=rd, y="y", running="x")
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.loads(result.model_dump_json())
    assert "Regression discontinuity: sharp local polynomial" in result.summary()
    assert "Robust" in str(result.to_latex())
    assert result.provenance["stata_parity_validated"] is False
    assert callable(oe.rdrobust)
    # Preserve the original resident-kernel throughput check. Disk bandwidth
    # selection, nearest-neighbor halos and serialization have Dataset tests.
    from openecon.econometrics import streaming_registry
    monkeypatch.setattr(streaming_registry, "supports_spec", lambda spec: False)
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1024")
    rng = np.random.default_rng(0)
    n = 1_000_000
    x = rng.uniform(-1, 1, n)
    y = x + 0.5 * (x >= 0) + rng.normal(size=n)
    start = time.perf_counter()
    big = oe.rdrobust(data=pd.DataFrame({"y": y, "x": x}), y="y", running="x")
    assert time.perf_counter() - start < 30
    assert abs(big.coefficients[2].estimate - 0.5) < 0.05
