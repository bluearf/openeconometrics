"""Independent dense NumPy oracle for dynamic panel GMM (xtdpd, xtabond, xtdpdsys).

The oracle shares no code with OpenEcon: it loops over panels in Python,
builds every panel's transformation matrix M_i (first differences or forward
orthogonal deviations written out explicitly), its instrument block Z_i from
dictionaries keyed by (group, variable, period, lag) and its H_i matrix (with
xtabond2's full-span deviation weights in the transformed-level block), then
applies the textbook formulas of Arellano and Bond (1991), Blundell and Bond
(1998), Windmeijer (2005) and Roodman (2009) with explicit inverses:

    W1 = (sum Z_i'H_i Z_i)^-1,  b1 = (A'W1A)^-1 A'W1 b,  A = sum Z_i'X_i,  b = sum Z_i'y_i
    W2 = (sum Z_i'e1_i e1_i'Z_i)^-1, b2 likewise, Windmeijer D column by column,
    AR(m) = sum_i a_i / sqrt(v) with Arellano and Bond's variance v.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe


def simulate(seed=0, n=80, periods=7, gaps=True, rho=0.5):
    """AR(1) panel with an exogenous x (correlated with u_i) and a predetermined w."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        u = rng.normal()
        y_prev, w_prev, e_prev = u / (1 - rho) + rng.normal(), rng.normal(), 0.0
        start = int(rng.integers(0, 2)) if gaps else 0
        for t in range(-4, periods):
            e = rng.normal()
            x = rng.normal() + 0.3 * u
            w = 0.5 * w_prev + 0.4 * e_prev + rng.normal()
            y = rho * y_prev + 0.3 * x - 0.2 * w + u + e
            if t >= start:
                records.append({"id": 100 + i, "year": 2000 + t, "y": y, "x": x, "w": w,
                                "z": rng.normal() + 0.5 * x})
            y_prev, w_prev, e_prev = y, w, e
    frame = pd.DataFrame(records)
    if gaps:
        candidates = np.flatnonzero(frame.year.to_numpy() >= 2003)
        frame = frame.drop(index=rng.choice(candidates, size=len(frame) // 25, replace=False))
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _norm_groups(gmm, iv, y, x, lags, system):
    if gmm is None:
        gmm = [{"columns": [y], "lags": [2, None]}] if lags >= 1 else []
    if iv is None:
        styled = {v for g in gmm for v in g["columns"]}
        exog = [v for v in x if v not in styled]
        iv = [{"columns": exog, "equation": "diff"}] if exog else []
    out_g, out_i = [], []
    for g in gmm:
        eq = g.get("equation", "both")
        eq = "diff" if not system else eq
        lo, hi = g.get("lags", [1, None])
        out_g.append((list(g["columns"]), lo, hi, eq, g.get("collapse", False)))
    for v in iv:
        eq = v.get("equation", "both")
        eq = "diff" if not system else eq
        out_i.append((list(v["columns"]), eq, v.get("passthru", False)))
    return out_g, out_i


def oracle(df, *, y="y", x=("x",), lags=1, gmm=None, iv=None, system=False, orthogonal=False,
           h=3, constant=True, time_dummies=False, collapse=False, artests=2):
    x = list(x)
    gmm_groups, iv_groups = _norm_groups(gmm, iv, y, x, lags, system)
    gmm_groups = [(c, lo, hi, eq, col or collapse) for c, lo, hi, eq, col in gmm_groups]
    df = df.sort_values(["id", "year"])
    tmin = int(df.year.min())
    span = int(df.year.max()) - tmin + 1
    panels = []
    for _, block in df.groupby("id", sort=True):
        data = {int(r.year) - tmin: r for r in block.itertuples(index=False)}
        level = [t for t in sorted(data) if all((t - lag) in data for lag in range(1, lags + 1))]
        panels.append((data, level))
    level_periods = sorted({t for _, level in panels for t in level})
    dummy_periods = level_periods[1:] if time_dummies else []
    constant = constant and system
    keys, blocks = {}, []

    def key_index(key):
        if key not in keys:
            keys[key] = len(keys)
        return keys[key]

    for data, level in panels:
        nl = len(level)
        if nl == 0:
            continue
        xl = []
        for t in level:
            row = [1.0] if constant else []
            row += [getattr(data[t - lag], y) for lag in range(1, lags + 1)]
            row += [getattr(data[t], v) for v in x]
            row += [1.0 if t == p else 0.0 for p in dummy_periods]
            xl.append(row)
        xl = np.array(xl, dtype=float)
        yl = np.array([getattr(data[t], y) for t in level])
        rows_m, storage, source = [], [], []
        for j, t in enumerate(level):
            m = np.zeros(nl)
            if orthogonal:
                later = nl - 1 - j
                if later == 0:
                    continue
                c = np.sqrt(later / (later + 1))
                m[j] = c
                m[j + 1:] = -c / later
                rows_m.append(m), storage.append(t + 1), source.append(j)
            else:
                if j == 0 or level[j - 1] != t - 1:
                    continue
                m[j], m[j - 1] = 1.0, -1.0
                rows_m.append(m), storage.append(t), source.append(j)
        mm = np.array(rows_m).reshape(-1, nl)
        # Covariance of the transformed rows with the level errors. xtabond2 takes it from the
        # deviation matrix F_g of a complete panel of `span` periods (weights c and -c/n with
        # n = span - 1 - t), not from this panel's own weights; for differences it is M.
        fg = mm.copy()
        if orthogonal:
            fg = np.zeros_like(mm)
            for r, j0 in enumerate(source):
                t0 = level[j0]
                n_g = span - 1 - t0
                c_g = np.sqrt(n_g / (n_g + 1))
                for j, q in enumerate(level):
                    fg[r, j] = c_g if q == t0 else (-c_g / n_g if q > t0 else 0.0)
        nt = len(rows_m)
        # instruments: dictionaries key -> value per stacked row
        zrows = [dict() for _ in range(nt + (nl if system else 0))]
        for gi, (cols, lo, hi, eq, col) in enumerate(gmm_groups):
            for v in cols:
                if eq in ("diff", "both"):
                    for r, ts in enumerate(storage):
                        top = ts if hi is None else hi
                        for lag in range(lo, top + 1):
                            if (ts - lag) in data:
                                key = ("g", gi, v, "d", lag) if col else ("g", gi, v, "d", ts, lag)
                                zrows[r][key_index(key)] = getattr(data[ts - lag], v)
                if system and eq in ("level", "both"):
                    # eq(both): the difference at lag lo - 1 (0 when lo = 0); eq(level): the
                    # differences at lags lo..hi (xtabond2 counts lags of the difference).
                    for j, t in enumerate(level):
                        lags_l = [max(lo - 1, 0)] if eq == "both" else \
                            range(lo, (t if hi is None else hi) + 1)
                        for d in lags_l:
                            if (t - d) in data and (t - d - 1) in data:
                                if eq == "both":
                                    key = ("g", gi, v, "l") if col else ("g", gi, v, "l", t)
                                else:
                                    key = ("g", gi, v, "l", d) if col else ("g", gi, v, "l", t, d)
                                zrows[nt + j][key_index(key)] = (getattr(data[t - d], v)
                                                                 - getattr(data[t - d - 1], v))
        for ii, (cols, eq, passthru) in enumerate(iv_groups):
            for v in cols:
                vl = np.array([getattr(data[t], v) for t in level])
                key = key_index(("i", ii, v))
                if eq in ("diff", "both"):
                    for r in range(nt):
                        if passthru:        # the level at the transformed row's date
                            ts = storage[r]
                            zrows[r][key] = getattr(data[ts], v) if ts in data else 0.0
                        else:
                            zrows[r][key] = mm[r] @ vl
                if system and eq in ("level", "both"):
                    for j in range(nl):
                        zrows[nt + j][key] = vl[j]
        for p in dummy_periods:
            dl = np.array([1.0 if t == p else 0.0 for t in level])
            key = key_index(("t", p))
            if system:
                for j in range(nl):
                    zrows[nt + j][key] = dl[j]
            else:
                for r in range(nt):
                    zrows[r][key] = mm[r] @ dl
        if constant:
            key = key_index(("c",))
            for j in range(nl):
                zrows[nt + j][key] = 1.0
        xt, yt = mm @ xl, mm @ yl
        if system:
            xs, ys = np.vstack([xt, xl]), np.concatenate([yt, yl])
            n_stack = np.vstack([mm, np.eye(nl)])
            if h == 1:
                hh = np.eye(nt + nl)
            elif h == 2:
                hh = np.block([[mm @ mm.T, np.zeros((nt, nl))], [np.zeros((nl, nt)), np.eye(nl)]])
            else:
                hh = np.block([[mm @ mm.T, fg], [fg.T, np.eye(nl)]])
        else:
            xs, ys, n_stack = xt, yt, mm
            hh = np.eye(nt) if h == 1 else mm @ mm.T
        # first differences of consecutive level observations (for the AR tests)
        drows, dper = [], []
        for j, t in enumerate(level):
            if j > 0 and level[j - 1] == t - 1:
                m = np.zeros(nl)
                m[j], m[j - 1] = 1.0, -1.0
                drows.append(m), dper.append(t)
        dmat = np.array(drows).reshape(-1, nl)
        blocks.append(dict(zrows=zrows, x=xs, y=ys, h=hh, nt=nt, nl=nl, xl=xl, yl=yl, fg=fg,
                           n=n_stack, d=dmat, dper=dper, storage=storage))
    width = len(keys)
    for blk in blocks:
        z = np.zeros((len(blk["zrows"]), width))
        for r, entries in enumerate(blk["zrows"]):
            for j, value in entries.items():
                z[r, j] = value
        blk["z"] = z
    used = np.flatnonzero(np.any([np.any(b["z"] != 0, axis=0) for b in blocks], axis=0))
    names = [k for k, j in sorted(keys.items(), key=lambda kv: kv[1])]
    names = [names[j] for j in used]
    for blk in blocks:
        blk["z"] = blk["z"][:, used]
    return _estimate(blocks, names, orthogonal, artests, h=h, system=system)


def _gmm(a, b, w):
    bread = np.linalg.inv(a.T @ w @ a)
    beta = bread @ a.T @ w @ b
    return beta, bread


def _ar(blocks, beta, order, *, p, v, robust, resid_key=None, sigma2=None, h=3):
    num, s_aa = 0.0, 0.0
    xw = 0.0
    gza = 0.0
    hom_first, hom_cross = 0.0, 0.0
    pairs = 0
    for blk in blocks:
        r_l = blk["yl"] - blk["xl"] @ beta
        de, dx = blk["d"] @ r_l, blk["d"] @ blk["xl"]
        w = np.zeros(len(de))
        for k, t in enumerate(blk["dper"]):
            if (t - order) in blk["dper"]:
                w[k] = de[blk["dper"].index(t - order)]
                pairs += 1
        a_i = w @ de
        num += a_i
        s_aa += a_i ** 2
        xw = xw + dx.T @ w
        u_i = blk["y"] - blk["x"] @ beta if resid_key is None else blk[resid_key]
        gza = gza + blk["z"].T @ u_i * a_i
        # Homoskedastic form (xtabond2 _ARTests): sigma2 H on the differenced errors and their
        # covariance with the transformed equation only (zero for the level rows).
        nt = blk["nt"]
        if h == 1:
            hom_first += w @ w
            dated = np.array([w[blk["dper"].index(ts)] if ts in blk["dper"] else 0.0
                              for ts in blk["storage"]])
            hom_cross = hom_cross + blk["z"][:nt].T @ dated
        else:
            dw = blk["d"].T @ w
            hom_first += dw @ dw
            hom_cross = hom_cross + blk["z"][:nt].T @ blk["fg"] @ dw
    if pairs == 0:
        return None
    if robust:
        var = s_aa - 2 * xw @ p @ gza + xw @ v @ xw
    else:
        var = sigma2 * hom_first - 2 * sigma2 * xw @ p @ hom_cross + xw @ v @ xw
    return num / np.sqrt(var)


def _estimate(blocks, names, orthogonal, artests, *, h=3, system=False):
    a = sum(b["z"].T @ b["x"] for b in blocks)
    bb = sum(b["z"].T @ b["y"] for b in blocks)
    zhz = sum(b["z"].T @ b["h"] @ b["z"] for b in blocks)
    w1 = np.linalg.inv(zhz)
    beta1, bread1 = _gmm(a, bb, w1)
    for b in blocks:
        b["e1"] = b["y"] - b["x"] @ beta1
        b["g1"] = b["z"].T @ b["e1"]
    nt = sum(b["nt"] for b in blocks)
    nl = sum(b["nl"] for b in blocks)
    # xtabond2: e'e / (2 - (orthogonal | h == 1)) / wttot from the transformed residuals (the
    # level residuals for h = 1 in system GMM); wttot = level count in system GMM with h = 1.
    c = 1.0 if (orthogonal or h == 1) else 2.0
    if system and h == 1:
        ssr, count = sum(b["e1"][b["nt"]:] @ b["e1"][b["nt"]:] for b in blocks), nl
    else:
        ssr, count = sum(b["e1"][:b["nt"]] @ b["e1"][:b["nt"]] for b in blocks), nt
    sigma2 = ssr / (c * count)
    omega = sum(np.outer(b["g1"], b["g1"]) for b in blocks)
    p1 = bread1 @ a.T @ w1
    v1r = p1 @ omega @ p1.T
    w2 = np.linalg.inv(omega)
    beta2, bread2 = _gmm(a, bb, w2)
    for b in blocks:
        b["e2"] = b["y"] - b["x"] @ beta2
    g2 = sum(b["z"].T @ b["e2"] for b in blocks)
    k = a.shape[1]
    dmat = np.zeros((k, k))
    for j in range(k):
        d_omega = -sum(b["z"].T @ (np.outer(b["x"][:, j], b["e1"]) + np.outer(b["e1"], b["x"][:, j]))
                       @ b["z"] for b in blocks)
        dmat[:, j] = -bread2 @ a.T @ w2 @ d_omega @ w2 @ g2
    vw = bread2 + dmat @ bread2 + bread2 @ dmat.T + dmat @ v1r @ dmat.T
    g1 = sum(b["g1"] for b in blocks)
    p2 = bread2 @ a.T @ w2
    out = dict(beta1=beta1, beta2=beta2, v1=sigma2 * bread1, v1r=v1r, v2=bread2, vw=vw,
               sigma2=sigma2, sargan=g1 @ w1 @ g1 / sigma2, hansen=g2 @ w2 @ g2,
               n_instruments=len(names), names=names, nt=nt, k=k, a=a, b=bb, omega=omega,
               blocks=blocks, w1=w1, w2=w2, p1=p1, p2=p2)
    out["ar_one_nonrobust"] = [_ar(blocks, beta1, m, p=p1, v=sigma2 * bread1, robust=False,
                                   resid_key="e1", sigma2=sigma2, h=h)
                               for m in range(1, artests + 1)]
    out["ar_one_robust"] = [_ar(blocks, beta1, m, p=p1, v=v1r, robust=True, resid_key="e1")
                            for m in range(1, artests + 1)]
    out["ar_two"] = [_ar(blocks, beta2, m, p=p2, v=bread2, robust=True, resid_key="e2")
                     for m in range(1, artests + 1)]
    out["ar_two_robust"] = [_ar(blocks, beta2, m, p=p2, v=vw, robust=True, resid_key="e2")
                            for m in range(1, artests + 1)]
    return out


def restricted_hansen(orc, members):
    """Hansen J without the instrument columns ``members`` (block of the full Omega)."""
    rest = [j for j in range(orc["n_instruments"]) if j not in set(members)]
    w = np.linalg.inv(orc["omega"][np.ix_(rest, rest)])
    a, b = orc["a"][rest], orc["b"][rest]
    beta, _ = _gmm(a, b, w)
    g = b - a @ beta
    return g @ w @ g


def coefs(result):
    return np.array([c.estimate for c in result.coefficients])


def ses(result):
    return np.array([c.std_error for c in result.coefficients])


def stat(result, name):
    return result.tests[name]["statistic"]


@pytest.fixture(scope="module")
def panel():
    return simulate()


def fit(panel, **options):
    base = dict(data=panel, y="y", x=["x"], panel="id", time="year")
    base.update(options)
    return oe.xtdpd(**base)


def oracle_args(options):
    keys = ("lags", "gmm", "iv", "system", "orthogonal", "h", "constant", "time_dummies",
            "collapse", "artests")
    out = {k: options[k] for k in keys if k in options}
    if "x" in options:
        out["x"] = options["x"]
    return out


CASES = [
    {},
    {"orthogonal": True},
    {"system": True},
    {"system": True, "orthogonal": True},
    {"system": True, "h": 2},
    {"system": True, "h": 1},
    {"h": 1},
    {"collapse": True},
    {"system": True, "collapse": True},
    {"lags": 2, "x": ["x", "w"],
     "gmm": [{"columns": ["y"], "lags": [2, 4]}, {"columns": ["w"], "lags": [1, 2]}]},
    {"system": True, "x": ["x", "w"], "time_dummies": True,
     "gmm": [{"columns": ["y"], "lags": [2, 3]}, {"columns": ["w"], "lags": [1, 1],
                                                  "collapse": True}],
     "iv": [{"columns": ["x"], "equation": "both"}, {"columns": ["z"], "equation": "level"}]},
    {"time_dummies": True, "iv": [{"columns": ["x", "z"], "passthru": True}]},
    {"orthogonal": True, "time_dummies": True, "collapse": True, "artests": 3},
    {"system": True, "constant": False, "artests": 3},
    {"system": True, "orthogonal": True,
     "iv": [{"columns": ["x"], "equation": "diff", "passthru": True},
            {"columns": ["x"], "equation": "level"}],
     "gmm": [{"columns": ["y"], "lags": [2, None], "equation": "diff"},
             {"columns": ["y"], "lags": [2, 2], "equation": "level"}]},
    {"orthogonal": True, "iv": [{"columns": ["x", "z"], "passthru": True}], "h": 1},
    {"system": True, "x": ["x", "w"],
     "gmm": [{"columns": ["y"], "lags": [2, 3], "equation": "diff"},
             {"columns": ["w"], "lags": [1, 2], "equation": "level"},
             {"columns": ["w"], "lags": [0, 1], "equation": "both", "collapse": True}]},
]


@pytest.mark.parametrize("options", CASES)
def test_one_step_matches_dense_oracle(panel, options):
    orc = oracle(panel, **oracle_args(options))
    nonrobust = fit(panel, **options)
    assert nonrobust.metrics["n_instruments"] == orc["n_instruments"]
    assert nonrobust.extra["n_obs_transformed"] == orc["nt"]
    assert_allclose(coefs(nonrobust), orc["beta1"], rtol=1e-9, atol=1e-11)
    assert_allclose(ses(nonrobust), np.sqrt(np.diag(orc["v1"])), rtol=1e-8)
    assert_allclose(stat(nonrobust, "sargan"), orc["sargan"], rtol=1e-8)
    assert_allclose(nonrobust.extra["sigma2_one_step"], orc["sigma2"], rtol=1e-10)
    for m, expected in enumerate(orc["ar_one_nonrobust"], start=1):
        assert_allclose(stat(nonrobust, f"ar{m}"), expected, rtol=1e-7)
    robust = fit(panel, robust=True, **options)
    assert_allclose(coefs(robust), orc["beta1"], rtol=1e-9, atol=1e-11)
    assert_allclose(ses(robust), np.sqrt(np.diag(orc["v1r"])), rtol=1e-8)
    assert_allclose(stat(robust, "hansen"), orc["hansen"], rtol=1e-7)
    for m, expected in enumerate(orc["ar_one_robust"], start=1):
        assert_allclose(stat(robust, f"ar{m}"), expected, rtol=1e-7)


@pytest.mark.parametrize("options", CASES)
def test_two_step_and_windmeijer_match_dense_oracle(panel, options):
    orc = oracle(panel, **oracle_args(options))
    two = fit(panel, twostep=True, **options)
    assert_allclose(coefs(two), orc["beta2"], rtol=1e-8, atol=1e-10)
    assert_allclose(ses(two), np.sqrt(np.diag(orc["v2"])), rtol=1e-7)
    assert_allclose(stat(two, "hansen"), orc["hansen"], rtol=1e-7)
    for m, expected in enumerate(orc["ar_two"], start=1):
        assert_allclose(stat(two, f"ar{m}"), expected, rtol=1e-6)
    corrected = fit(panel, twostep=True, robust=True, **options)
    assert_allclose(coefs(corrected), orc["beta2"], rtol=1e-8, atol=1e-10)
    assert_allclose(ses(corrected), np.sqrt(np.diag(orc["vw"])), rtol=1e-7)
    assert corrected.extra["windmeijer"] is True
    for m, expected in enumerate(orc["ar_two_robust"], start=1):
        assert_allclose(stat(corrected, f"ar{m}"), expected, rtol=1e-6)


def test_difference_in_hansen_matches_restricted_estimation(panel):
    options = {"system": True, "x": ["x", "w"],
               "gmm": [{"columns": ["y"], "lags": [2, 3]}, {"columns": ["w"], "lags": [1, 2]}],
               "iv": [{"columns": ["x"], "equation": "diff"}]}
    orc = oracle(panel, **oracle_args(options))
    result = fit(panel, twostep=True, robust=True, **options)
    names = orc["names"]
    subsets = {
        "level": [j for j, key in enumerate(names) if key[0] == "g" and key[3] == "l"],
        "gmm1": [j for j, key in enumerate(names) if key[:2] == ("g", 0)],
        "gmm2": [j for j, key in enumerate(names) if key[:2] == ("g", 1)],
        "iv1": [j for j, key in enumerate(names) if key[:2] == ("i", 0)],
    }
    for name, members in subsets.items():
        restricted = restricted_hansen(orc, members)
        excl, diff = result.tests[f"hansen_excl_{name}"], result.tests[f"diff_hansen_{name}"]
        assert excl["df"] == orc["n_instruments"] - len(members) - orc["k"]
        assert diff["df"] == len(members)
        assert_allclose(excl["statistic"], restricted, rtol=1e-6)
        assert_allclose(diff["statistic"], orc["hansen"] - restricted, rtol=1e-6, atol=1e-8)
