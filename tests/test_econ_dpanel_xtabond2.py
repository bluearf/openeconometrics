"""xtabond2-faithful NumPy oracle for dynamic panel GMM (oe.xtdpd, oe.xtabond, oe.xtdpdsys).

This oracle is written from the algorithm of Roodman's ``xtabond2`` Mata code
(``xtabond2_mata()``, ``_H``, ``_xform``, ``_Orthog``, ``_MakeGMMinsts``,
``_ARTests``), not from OpenEcon's formulas. It reproduces xtabond2's own data
layout: every panel lives on the full grid of T periods; the transformed
equation and (system GMM) the level equation occupy T rows each; forward
orthogonal deviations are stored one period late; instruments and residuals of
unused rows are zero; H is the T x T (2T x 2T) matrix of ``_H``; inverses are
explicit. Conventions taken from that code:

- ``sig2 = e1'e1 / (2 - (orthogonal | h == 1)) / wttot`` from the one-step
  residuals of the transformed equation (the level equation when h = 1 in
  system GMM); ``wttot`` counts the transformed rows in system GMM with h > 1
  and the reported observations otherwise; two-step recomputes it from e2.
- Sargan = e1'Z (sum Z_i'HZ_i)^-1 Z'e1 / sig2; Hansen = e2'Z S^-1 Z'e2 with
  S = sum Z_i'e1_i e1_i'Z_i (also after one-step robust).
- Windmeijer: V2 + D V1r D' + (D + D) V2, symmetrized.
- small: V * wttot / (wttot - k) after one-step nonrobust, otherwise
  V * (N - 1) / (N - k) * G / (G - 1); df = N - k after one-step nonrobust,
  otherwise G - (constant present); sig2 * wttot / (wttot - k).
- AR(m): robust form after robust or two-step; after one-step nonrobust the
  homoskedastic form with H = sig2 _H(h, diff, diff) and the cross covariance
  sig2 _H(h, diff, transform) (zero for the level rows of system GMM); always
  the covariance before the small-sample factor and sig2 after it.
- difference-in-Sargan / Hansen: re-estimation with the block of S (of
  sig2 sum Z_i'HZ_i after one-step nonrobust) for the remaining instruments.
- GMM-style: transformed rows get levels at lags lo..hi; eq(both) adds the
  difference at lag lo - 1 (lag 0 when lo = 0) for the level rows; eq(level)
  gives the level rows the differences at lags lo..hi. IV-style: transformed
  like the regressors (passthru: the level at the transformed row's date).

The model Wald test is the slopes-only test of official Stata (xtabond2's own
statistic also includes the constant in the quadratic form; see the docs).
With orthogonal deviations xtabond2 builds the transformed-level block of H
(h = 3) and the homoskedastic AR cross covariance from the deviation matrix of
a panel spanning all periods; the oracle does the same (``xform``). On panels
with gaps the homoskedastic AR test is not compared (xtabond2 also pairs lagged
residuals with gap periods there; documented), nor is ``passthru`` (an absent
date drops xtabond2's row, OpenEcon uses zero).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe


def make_panel(seed, n=60, periods=7, kind="balanced"):
    """Heteroskedastic AR(1) panel with exogenous x, predetermined w and an extra instrument z."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = rng.normal()
        y, w_prev, e_prev = 2 * u + rng.normal(), rng.normal(), 0.0
        for t in range(-3, periods):
            e = rng.normal() * (0.5 + 0.5 * (i % 3))
            x = rng.normal() + 0.4 * u
            w = 0.5 * w_prev + 0.3 * e_prev + rng.normal()
            y = 0.5 * y + 0.4 * x - 0.3 * w + u + e
            if t >= 0:
                rows.append({"id": i, "t": 2000 + t, "y": y, "x": x, "w": w,
                             "z": rng.normal() + 0.5 * x})
            w_prev, e_prev = w, e
    df = pd.DataFrame(rows)
    period = df.t.to_numpy() - 2000
    if kind == "late":            # panels start late; every panel reaches the last period
        start = rng.integers(0, 3, size=n)
        df = df[period >= start[df.id.to_numpy()]]
    elif kind == "gaps":          # gaps and early ends
        drop = (rng.random(len(df)) < 0.07) & (period >= 2)
        end = periods - rng.integers(0, 3, size=n)
        df = df[~drop & (period < end[df.id.to_numpy()])]
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---- xtabond2 building blocks -------------------------------------------------

def xform(kind, periods):
    """``_xform``: rows = periods, column c = transformed observation stored at c."""
    m = np.zeros((periods, periods))
    for c in range(1, periods):
        if kind == "diff":
            m[c, c], m[c - 1, c] = 1.0, -1.0
        else:
            later = periods - c
            m[c - 1, c], m[c:, c] = 1.0, -1.0 / later
            m[:, c] /= np.sqrt(1.0 + 1.0 / later)
    return m


def h_matrix(h, left, right, system, periods):
    ml = np.eye(periods) if h == 1 else xform(left, periods)
    mr = np.eye(periods) if h == 1 else xform(right, periods)
    if h == 3 or not system:
        if system:
            ml, mr = np.hstack([ml, np.eye(periods)]), np.hstack([mr, np.eye(periods)])
        return ml.T @ mr
    out = np.zeros((2 * periods, 2 * periods))
    out[:periods, :periods], out[periods:, periods:] = ml.T @ mr, np.eye(periods)
    return out


def difference(v):
    out = np.full_like(v, np.nan)
    out[1:] = v[1:] - v[:-1]
    return out


def orthogonal(v, complete):
    """``_Orthog`` with forward = 1: deviation of period t stored at t + 1."""
    out = np.full_like(v, np.nan)
    total, count = np.zeros(v.shape[1:]), 0
    for t in range(len(v) - 1, -1, -1):
        if count:
            out[t + 1] = np.sqrt(1 - 1 / (count + 1)) * (v[t] - total / count)
        if complete[t]:
            count += 1
            total = total + v[t]
    return out


def lagged(v, lag):
    out = np.full_like(v, np.nan)
    if lag < len(v):
        out[lag:] = v[:len(v) - lag]
    return out


def zero(v):
    return np.where(np.isfinite(v), v, 0.0)


def ginv(matrix):
    """Generalized inverse (xtabond2 uses invsym) and rank, on the correlation scale.

    With system GMM and h(3), H = N N' has rank T, so sum Z_i'HZ_i = B'B can be
    singular although Z has full column rank. The moments Z'u = B'r and Z'X = B'X_l
    lie in the range of B', so the estimator, the Sargan statistic and the AR
    tests do not depend on which generalized inverse is used.
    """
    diag = np.diag(matrix).copy()
    scale = np.where(diag > 0, 1 / np.sqrt(np.where(diag > 0, diag, 1)), 1.0)
    scaled = matrix * scale[:, None] * scale[None, :]
    values, vectors = np.linalg.eigh((scaled + scaled.T) / 2)
    keep = values > 1e-12 * values.max()
    inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    return inverse * scale[:, None] * scale[None, :], int(keep.sum())


def xtabond2(df, *, y="y", x=(), lags=1, gmm=None, iv=None, system=False, orth=False, h=3,
             constant=True, time_dummies=False, collapse=False, twostep=False, robust=False,
             small=False, artests=2):
    x = list(x)
    if gmm is None:
        gmm = [{"columns": [y], "lags": [2, None]}]
    if iv is None:
        styled = {name for group in gmm for name in group["columns"]}
        exog = [name for name in x if name not in styled]
        iv = [{"columns": exog, "equation": "diff"}] if exog else []
    gmm = [(list(g["columns"]), g.get("lags", [1, None])[0], g.get("lags", [1, None])[1],
            g.get("equation", "both") if system else "diff", g.get("collapse", collapse))
           for g in gmm]
    iv = [(list(g["columns"]), g.get("equation", "both") if system else "diff",
           g.get("passthru", False)) for g in iv]
    cons = constant and system
    tmin = int(df.t.min())
    periods = int(df.t.max()) - tmin + 1
    names = sorted({y, *x, *(c for g in gmm for c in g[0]), *(c for g in iv for c in g[0])})
    grids = []
    for _, block in df.groupby("id", sort=True):
        grid = {name: np.full(periods, np.nan) for name in names}
        cells = block.t.to_numpy() - tmin
        for name in names:
            grid[name][cells] = block[name].to_numpy(dtype=float)
        lagcols = [lagged(grid[y], lag) for lag in range(1, lags + 1)]
        x0 = np.column_stack([*lagcols, *(grid[name] for name in x)]) if (lags or x) else \
            np.zeros((periods, 0))
        ivbase = [grid[c] for g in iv for c in g[0]]
        complete = np.isfinite(grid[y]) & np.isfinite(x0).all(axis=1)
        for base in ivbase:
            complete &= np.isfinite(base)
        grids.append({"grid": grid, "x0": x0, "complete": complete})
    level_periods = sorted({t for g in grids for t in np.flatnonzero(g["complete"])})
    dummy_periods = level_periods[1:] if time_dummies else []
    transform = (lambda v, c: orthogonal(v, c)) if orth else (lambda v, c: difference(v))
    keys = {}

    def key(k):
        return keys.setdefault(k, len(keys))

    panels = []
    for g in grids:
        grid, complete = g["grid"], g["complete"]
        dummies = np.column_stack([(np.arange(periods) == p).astype(float)
                                   for p in dummy_periods]) if dummy_periods else \
            np.zeros((periods, 0))
        x0 = np.column_stack([*([np.ones(periods)] if cons else []), g["x0"], dummies])
        x0[~complete] = np.nan
        y0 = np.where(complete, grid[y], np.nan)
        xt, yt = transform(x0, complete), transform(y0[:, None], complete)[:, 0]
        source_ok = lagged(complete.astype(float), 1) == 1 if orth else complete
        use_t = source_ok & np.isfinite(xt).all(axis=1) & np.isfinite(yt)
        zt, zl = {}, {}
        for gi, (cols, lo, hi, eq, coll) in enumerate(gmm):
            for name in cols:
                v = grid[name]
                if eq in ("diff", "both"):
                    for r in range(periods):
                        top = periods if hi is None else hi
                        for lag in range(lo, top + 1):
                            if 0 <= r - lag and np.isfinite(v[r - lag]):
                                k = (gi, name, "d", lag) if coll else (gi, name, "d", r, lag)
                                zt[(r, key(k))] = v[r - lag]
                dv = difference(v)
                if system and eq == "both":
                    lag = lo - 1 if lo >= 1 else 0
                    for t in range(periods):
                        if 0 <= t - lag and np.isfinite(dv[t - lag]):
                            k = (gi, name, "L") if coll else (gi, name, "L", t)
                            zl[(t, key(k))] = dv[t - lag]
                if system and eq == "level":
                    for t in range(periods):
                        top = periods if hi is None else hi
                        for lag in range(lo, top + 1):
                            if 0 <= t - lag and np.isfinite(dv[t - lag]):
                                k = (gi, name, "l", lag) if coll else (gi, name, "l", t, lag)
                                zl[(t, key(k))] = dv[t - lag]
        for gi, (cols, eq, passthru) in enumerate(iv):
            for name in cols:
                v = grid[name]
                k = key(("iv", gi, name))
                vt = v if passthru else transform(v[:, None], complete)[:, 0]
                for r in range(periods):
                    if eq in ("diff", "both"):
                        if use_t[r] and not np.isfinite(vt[r]):
                            use_t[r] = False
                        if np.isfinite(vt[r]):
                            zt[(r, k)] = vt[r]
                    if system and eq != "diff":
                        zl[(r, k)] = v[r]
        if dummy_periods:
            dcols = slice(x0.shape[1] - len(dummy_periods), x0.shape[1])
            for j, p in enumerate(dummy_periods):
                k = key(("time", p))
                for r in range(periods):
                    if system:
                        zl[(r, k)] = float(r == p)
                    elif np.isfinite(xt[r, dcols][j]):
                        zt[(r, k)] = xt[r, dcols][j]
        if cons:
            k = key(("cons",))
            for r in range(periods):
                zl[(r, k)] = 1.0
        panels.append({"x0": x0, "y0": y0, "xt": xt, "yt": yt, "use_t": use_t,
                       "use_l": complete.copy(), "zt": zt, "zl": zl, "complete": complete})
    width = len(keys)
    kx = panels[0]["x0"].shape[1]
    for p in panels:
        zt, zl = np.zeros((periods, width)), np.zeros((periods, width))
        for (r, j), value in p["zt"].items():
            zt[r, j] = value
        for (r, j), value in p["zl"].items():
            zl[r, j] = value
        zt[~p["use_t"]], zl[~p["use_l"]] = 0.0, 0.0
        xt, yt = np.where(p["use_t"][:, None], zero(p["xt"]), 0.0), np.where(p["use_t"],
                                                                              zero(p["yt"]), 0.0)
        xl, yl = np.where(p["use_l"][:, None], zero(p["x0"]), 0.0), np.where(p["use_l"],
                                                                              zero(p["y0"]), 0.0)
        if system:
            p.update(z=np.vstack([zt, zl]), x=np.vstack([xt, xl]), y=np.concatenate([yt, yl]))
        else:
            p.update(z=zt, x=xt, y=yt)
    used = np.flatnonzero(np.any([np.any(p["z"] != 0, axis=0) for p in panels], axis=0))
    labels = [k for k, _ in sorted(keys.items(), key=lambda item: item[1])]
    labels = [labels[j] for j in used]
    for p in panels:
        p["z"] = p["z"][:, used]
    h_full = h_matrix(h, "orth" if orth else "diff", "orth" if orth else "diff", system, periods)
    return estimate(panels, labels, h_full, periods, kx, system=system, orth=orth, h=h,
                    cons=cons, twostep=twostep, robust=robust, small=small, artests=artests)


def estimate(panels, labels, h_full, periods, k, *, system, orth, h, cons, twostep, robust,
             small, artests):
    zx = sum(p["z"].T @ p["x"] for p in panels)
    zy = sum(p["z"].T @ p["y"] for p in panels)
    s1 = sum(p["z"].T @ h_full @ p["z"] for p in panels)
    a1raw, j0 = ginv(s1)
    v1raw = np.linalg.inv(zx.T @ a1raw @ zx)
    b1 = v1raw @ zx.T @ a1raw @ zy
    nobs = sum(int(p["use_l"].sum() if system else p["use_t"].sum()) for p in panels)
    n_t = sum(int(p["use_t"].sum()) for p in panels)
    groups = sum(bool(p["use_l"].any() if system else p["use_t"].any()) for p in panels)
    wttot = n_t if (system and h > 1) else nobs
    divisor = 1 if (orth or h == 1) else 2

    def sig2_of(beta):
        total = 0.0
        for p in panels:
            e = p["y"] - p["x"] @ beta
            if system:
                e = e[periods:] if h == 1 else e[:periods]
            total += e @ e
        return total / divisor / wttot

    sig2 = sig2_of(b1)
    v1, a1 = v1raw * sig2, a1raw / sig2
    for p in panels:
        p["e1"] = p["y"] - p["x"] @ b1
        p["g1"] = p["z"].T @ p["e1"]
    ze1 = sum(p["g1"] for p in panels)
    out = {"b1": b1, "sig2": sig2, "sargan": ze1 @ a1 @ ze1, "labels": labels,
           "n_inst": len(labels), "j0": j0, "k": k, "nobs": nobs, "groups": groups}
    onestepnonrobust = not twostep and not robust
    s = sum(np.outer(p["g1"], p["g1"]) for p in panels)
    if onestepnonrobust:
        s = s1 * sig2
    p1 = v1 @ zx.T @ a1
    v1r = p1 @ s @ p1.T
    hansen = None
    if not onestepnonrobust:
        a2, _ = ginv(s)
        v2 = np.linalg.inv(zx.T @ a2 @ zx)
        b2 = v2 @ zx.T @ a2 @ zy
        for p in panels:
            p["e2"] = p["y"] - p["x"] @ b2
        ze2 = sum(p["z"].T @ p["e2"] for p in panels)
        hansen = ze2 @ a2 @ ze2
        if twostep:
            sig2 = sig2_of(b2)
        a2ze = a2 @ ze2
        d = sum((p["g1"] @ a2ze) * (p["z"].T @ p["x"])
                + np.outer(p["g1"], a2ze @ (p["z"].T @ p["x"])) for p in panels)
        d = v2 @ zx.T @ a2 @ d
        v2r = v2 + d @ v1r @ d.T + (d + d) @ v2
        out.update(b2=b2, hansen=hansen, v2=v2, a2=a2)
    if twostep:
        b, v_plain, a_step, resid = b2, v2, a2, "e2"
        v = v2r if robust else v2
    else:
        b, v_plain, a_step, resid = b1, v1, a1, "e1"
        v = v1r if robust else v1
    v_ar = v
    v = (v + v.T) / 2
    m2vzxa = -2 * v_plain @ zx.T @ a_step
    if small:
        factor = wttot / (wttot - k) if onestepnonrobust else \
            (nobs - 1) / (nobs - k) * groups / (groups - 1)
        v = v * factor
        sig2 = sig2 * wttot / (wttot - k)
        out["df_r"] = (nobs - k) if onestepnonrobust else groups - int(cons)
    out.update(b=b, v=v, sig2_reported=sig2)
    # Arellano-Bond tests (_ARTests)
    h_ar = h_matrix(1 if h == 1 else h, "diff", "diff", False, periods) * sig2
    psit = h_matrix(h, "diff", "orth" if orth else "diff", False, periods) * sig2
    ar = []
    for lag in range(1, artests + 1):
        total, whw, zhw, tmp = 0.0, 0.0, 0.0, 0.0
        for p in panels:
            if orth:
                w = zero(difference(p["y0"] - p["x0"] @ b))
                px = zero(difference(p["x0"]))
            else:
                w = p[resid][:periods]
                px = p["x"][:periods]
            wl = zero(lagged(w, lag))
            a_i = w @ wl
            total += a_i
            tmp = tmp + px.T @ wl
            if onestepnonrobust:
                wli = wl * p["use_t"]
                whw += wli @ h_ar @ wli
                psiw = psit.T @ wli
                zhw = zhw + p["z"][:periods].T @ psiw
            else:
                whw += a_i ** 2
                zhw = zhw + p["z"].T @ p[resid] * a_i
        if total == 0:
            ar.append(None)
            continue
        variance = whw + tmp @ (m2vzxa @ zhw + v_ar @ tmp)
        ar.append(total / np.sqrt(variance) if variance > 0 else np.nan)
    out["ar"] = ar
    # difference-in-Sargan / Hansen
    out["s"] = s
    out["zx"], out["zy"] = zx, zy
    out["over"] = hansen if hansen is not None else out["sargan"]
    return out


def restricted(orc, members):
    rest = [j for j in range(orc["n_inst"]) if j not in set(members)]
    app, _ = ginv(orc["s"][np.ix_(rest, rest)])
    zx, zy = orc["zx"][rest], orc["zy"][rest]
    beta = np.linalg.solve(zx.T @ app @ zx, zx.T @ app @ zy)
    g = zy - zx @ beta
    return g @ app @ g


# ---- helpers -------------------------------------------------------------------

def table(result):
    return {c.term: c for c in result.coefficients}


def ordered(result, orc_terms):
    t = table(result)
    return np.array([t[name].estimate for name in orc_terms]), \
        np.array([t[name].std_error for name in orc_terms])


def terms_of(result):
    """Oracle column order: Intercept, L1.y.., x.., time dummies (as OpenEcon names them)."""
    return [c.term for c in result.coefficients]


def compare(result, orc, *, rtol=1e-7, ar=True):
    terms = terms_of(result)
    assert len(terms) == orc["k"]
    est = np.array([c.estimate for c in result.coefficients])
    se = np.array([c.std_error for c in result.coefficients])
    assert_allclose(est, orc["b"], rtol=rtol, atol=1e-10)
    assert_allclose(np.array(result.covariance_matrix), orc["v"], rtol=rtol * 10,
                    atol=1e-12 * np.abs(orc["v"]).max())
    assert_allclose(se, np.sqrt(np.diag(orc["v"])), rtol=rtol * 10)
    assert result.metrics["n_instruments"] == orc["j0"]
    assert result.extra["instrument_columns"] == orc["n_inst"]
    assert result.metrics["n_groups"] == orc["groups"]
    assert result.nobs == orc["nobs"]
    sargan = result.tests["sargan"]
    assert sargan["df"] == orc["j0"] - orc["k"]
    assert_allclose(sargan["statistic"], orc["sargan"], rtol=rtol * 10)
    assert_allclose(sargan["p_value"], stats.chi2.sf(orc["sargan"], sargan["df"]), rtol=1e-6)
    if "hansen" in orc:
        assert_allclose(result.tests["hansen"]["statistic"], orc["hansen"], rtol=rtol * 10)
    else:
        assert "hansen" not in result.tests
    for m, expected in enumerate(orc["ar"] if ar else [], start=1):
        got = result.tests[f"ar{m}"]
        if expected is None or not np.isfinite(expected):
            # no pairs, or a negative homoskedastic variance (xtabond2 reports missing)
            assert got["statistic"] is None and got["p_value"] is None
            continue
        assert_allclose(got["statistic"], expected, rtol=rtol * 100)
        assert_allclose(got["p_value"], 2 * stats.norm.sf(abs(expected)), rtol=1e-5, atol=1e-300)
    assert_allclose(result.metrics["sigma_e"], np.sqrt(orc["sig2_reported"]), rtol=rtol * 10)


SPECS = {
    "diff": {},
    "diff_h1": {"h": 1},
    "diff_h2": {"h": 2},
    "diff_lags2_pre": {"lags": 2, "x": ["x", "w"],
                       "gmm": [{"columns": ["y"], "lags": [2, 4]},
                               {"columns": ["w"], "lags": [1, 3]}]},
    "diff_collapse_iv": {"x": ["x", "w"], "collapse": True,
                         "gmm": [{"columns": ["y"], "lags": [2, None]},
                                 {"columns": ["w"], "lags": [2, 3]}],
                         "iv": [{"columns": ["x", "z"]}]},
    "diff_dummies": {"time_dummies": True},
    "sys": {"system": True},
    "sys_h1": {"system": True, "h": 1},
    "sys_h2": {"system": True, "h": 2},
    "sys_nocons": {"system": True, "constant": False, "artests": 3},
    "sys_level_groups": {"system": True, "x": ["x", "w"],
                         "gmm": [{"columns": ["y"], "lags": [2, 3], "equation": "diff"},
                                 {"columns": ["y"], "lags": [1, 1], "equation": "level"},
                                 {"columns": ["w"], "lags": [0, 1], "equation": "level",
                                  "collapse": True}],
                         "iv": [{"columns": ["x"], "equation": "both"},
                                {"columns": ["z"], "equation": "level"}]},
    "sys_singular_h3": {"system": True,
                        "gmm": [{"columns": ["y"], "lags": [2, 3], "equation": "diff"},
                                {"columns": ["y"], "lags": [1, 2], "equation": "level"}]},
    "sys_lo0_both": {"system": True, "x": ["x", "w"],
                     "gmm": [{"columns": ["y"], "lags": [2, 3]},
                             {"columns": ["w"], "lags": [0, 1], "collapse": True}]},
    "sys_dummies_collapse": {"system": True, "time_dummies": True, "collapse": True},
}
ORTH_SPECS = {
    "orth": {"orthogonal": True},
    "orth_h1": {"orthogonal": True, "h": 1},
    "orth_passthru": {"orthogonal": True, "x": ["x"],
                      "iv": [{"columns": ["x", "z"], "passthru": True}]},
    "orth_sys": {"orthogonal": True, "system": True},
    "orth_sys_h2": {"orthogonal": True, "system": True, "h": 2},
    "orth_sys_h1_dummies": {"orthogonal": True, "system": True, "h": 1, "time_dummies": True},
    "orth_sys_passthru_diff": {"orthogonal": True, "system": True, "x": ["x"],
                               "iv": [{"columns": ["x"], "equation": "diff", "passthru": True},
                                      {"columns": ["z"], "equation": "level"}]},
}
VARIANTS = [{}, {"robust": True}, {"twostep": True}, {"twostep": True, "robust": True}]


def oracle_args(spec, variant, small):
    keys = {"lags": "lags", "gmm": "gmm", "iv": "iv", "system": "system", "orthogonal": "orth",
            "h": "h", "constant": "constant", "time_dummies": "time_dummies",
            "collapse": "collapse", "artests": "artests"}
    out = {keys[name]: value for name, value in spec.items() if name in keys}
    out["x"] = spec.get("x", ["x"])
    out.update(twostep=variant.get("twostep", False), robust=variant.get("robust", False),
               small=small)
    return out


def run(df, spec, variant, small=False):
    arguments = dict(data=df, y="y", x=["x"], panel="id", time="t", small=small)
    arguments.update(spec)
    arguments.update(variant)
    return oe.xtdpd(**arguments)


@pytest.fixture(scope="module")
def panels():
    return {"balanced": make_panel(1), "gaps": make_panel(2, n=70, kind="gaps"),
            "late": make_panel(3, n=60, kind="late")}


@pytest.mark.parametrize("variant", VARIANTS, ids=["one", "one_robust", "two", "two_robust"])
@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("design", ["balanced", "gaps"])
def test_matches_xtabond2_algorithm(panels, design, name, variant):
    df, spec = panels[design], SPECS[name]
    result = run(df, spec, variant)
    orc = xtabond2(df, **oracle_args(spec, variant, False))
    compare(result, orc)


@pytest.mark.parametrize("variant", VARIANTS, ids=["one", "one_robust", "two", "two_robust"])
@pytest.mark.parametrize("name", list(ORTH_SPECS))
@pytest.mark.parametrize("design", ["balanced", "late"])
def test_orthogonal_deviations_match_xtabond2_algorithm(panels, design, name, variant):
    df, spec = panels[design], ORTH_SPECS[name]
    result = run(df, spec, variant)
    orc = xtabond2(df, **oracle_args(spec, variant, False))
    compare(result, orc)


@pytest.mark.parametrize("variant", VARIANTS, ids=["one", "one_robust", "two", "two_robust"])
@pytest.mark.parametrize("name", ["orth", "orth_h1", "orth_sys", "orth_sys_h2",
                                  "orth_sys_h1_dummies"])
def test_orthogonal_deviations_with_gaps_and_early_ends_match_xtabond2(panels, name, variant):
    """Unbalanced panels: xtabond2's full-span deviation weights in H (h = 3, system GMM).

    Everything matches except the homoskedastic AR test after one-step nonrobust
    estimation, where xtabond2 pairs lagged residuals with gap periods (documented).
    """
    df, spec = panels["gaps"], ORTH_SPECS[name]
    result = run(df, spec, variant)
    orc = xtabond2(df, **oracle_args(spec, variant, False))
    compare(result, orc, ar=bool(variant))


@pytest.mark.parametrize("variant", VARIANTS, ids=["one", "one_robust", "two", "two_robust"])
@pytest.mark.parametrize("name", ["diff", "sys", "sys_h1", "orth_sys"])
def test_small_sample_conventions_match_xtabond2(panels, name, variant):
    df = panels["balanced"]
    spec = {**SPECS, **ORTH_SPECS}[name]
    result = run(df, spec, variant, small=True)
    orc = xtabond2(df, **oracle_args(spec, variant, True))
    compare(result, orc)
    assert result.inference["use_t"] is True
    assert result.inference["df_inference"] == orc["df_r"]
    t = np.array([c.statistic for c in result.coefficients])
    p = np.array([c.p_value for c in result.coefficients])
    assert_allclose(p, 2 * stats.t.sf(np.abs(t), orc["df_r"]), rtol=1e-8)
    low = np.array([c.ci_low for c in result.coefficients])
    se = np.sqrt(np.diag(orc["v"]))
    assert_allclose(low, orc["b"] - stats.t.ppf(0.975, orc["df_r"]) * se, rtol=1e-8, atol=1e-12)
    model = result.tests["model"]
    slopes = [j for j, c in enumerate(result.coefficients) if c.term != "Intercept"]
    block = orc["v"][np.ix_(slopes, slopes)]
    wald = orc["b"][slopes] @ np.linalg.solve(block, orc["b"][slopes])
    assert model["distribution"] == "F" and model["df"] == len(slopes)
    assert model["df2"] == orc["df_r"]
    assert_allclose(model["statistic"], wald / len(slopes), rtol=1e-8)
    assert_allclose(model["p_value"], stats.f.sf(wald / len(slopes), len(slopes), orc["df_r"]),
                    rtol=1e-6)


def test_large_sample_z_inference_and_wald(panels):
    df = panels["gaps"]
    result = run(df, SPECS["sys"], {"twostep": True, "robust": True})
    orc = xtabond2(df, **oracle_args(SPECS["sys"], {"twostep": True, "robust": True}, False))
    z = orc["b"] / np.sqrt(np.diag(orc["v"]))
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.norm.sf(np.abs(z)),
                    rtol=1e-8)
    slopes = [1, 2]
    wald = orc["b"][slopes] @ np.linalg.solve(orc["v"][np.ix_(slopes, slopes)], orc["b"][slopes])
    assert result.tests["model"]["distribution"] == "chi2"
    assert_allclose(result.tests["model"]["statistic"], wald, rtol=1e-8)
    assert "df_resid" not in result.metrics and result.inference["use_t"] is False


def _subsets(labels):
    groups = {}
    for j, label in enumerate(labels):
        if label[0] == "cons":
            continue
        name = f"iv{label[1] + 1}" if label[0] == "iv" else (
            "time" if label[0] == "time" else f"gmm{label[0] + 1}")
        groups.setdefault(name, []).append(j)
    level = [j for j, label in enumerate(labels)
             if isinstance(label[0], int) and label[2] in ("L", "l")]
    if level:
        groups["level"] = level
    return groups


@pytest.mark.parametrize("variant", VARIANTS, ids=["one", "one_robust", "two", "two_robust"])
def test_difference_in_sargan_and_hansen_match_xtabond2(panels, variant):
    df, spec = panels["balanced"], SPECS["sys_level_groups"]
    result = run(df, spec, variant)
    orc = xtabond2(df, **oracle_args(spec, variant, False))
    kind = "sargan" if not variant else "hansen"
    checked = 0
    for name, members in _subsets(orc["labels"]).items():
        rest = orc["n_inst"] - len(members)
        if rest < orc["k"]:
            continue
        expected = restricted(orc, members)
        excl, diff = result.tests[f"{kind}_excl_{name}"], result.tests[f"diff_{kind}_{name}"]
        assert excl["df"] == rest - orc["k"] and diff["df"] == len(members)
        assert_allclose(excl["statistic"], expected, rtol=1e-6)
        assert_allclose(diff["statistic"], orc["over"] - expected, rtol=1e-6, atol=1e-8)
        assert_allclose(diff["p_value"], stats.chi2.sf(max(orc["over"] - expected, 0),
                                                       len(members)), rtol=1e-5, atol=1e-12)
        checked += 1
    assert checked >= 4
    other = "hansen" if kind == "sargan" else "sargan"
    assert not any(key.startswith(f"diff_{other}") for key in result.tests)


def test_wrappers_build_xtabond_and_xtdpdsys_instrument_sets(panels):
    df = panels["balanced"]
    ab = oe.xtabond(data=df, y="y", x=["x"], panel="id", time="t", predetermined=["w"],
                    maxldep=2, maxlags=2, twostep=True, robust=True)
    orc = xtabond2(df, x=["x", "w"], gmm=[{"columns": ["y"], "lags": [2, 3]},
                                          {"columns": ["w"], "lags": [1, 2]}],
                   iv=[{"columns": ["x"], "equation": "diff"}], twostep=True, robust=True)
    compare(ab, orc)
    bb = oe.xtdpdsys(data=df, y="y", x=["x"], panel="id", time="t", endogenous=["w"])
    orc = xtabond2(df, x=["x", "w"], system=True,
                   gmm=[{"columns": ["y"], "lags": [2, None]},
                        {"columns": ["w"], "lags": [2, None]}],
                   iv=[{"columns": ["x"], "equation": "diff"}])
    compare(bb, orc)
