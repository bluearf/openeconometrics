"""Independent oracles for the mixed family (verification pass).

1. Stata parity: the [XT] xtpoisson manual prints re (gamma and normal), fe and pa
   output for McCullagh and Nelder's ships data (Table 6.2, 34 observations with
   positive service). The data are reconstructed below and every printed digit is
   compared.
2. xtgee: a from-scratch NumPy GEE with explicit per-panel correlation matrices and
   the correlation estimators exactly as printed in Stata's [XT] xtgee Methods and
   formulas (pooled pairs for exchangeable; per-panel 1/n_i moments for ar / stationary /
   nonstationary / unstructured; upper-left blocks of one max(n_i) matrix; panels with
   n_i <= g dropped), plus invariances.
3. mixed: a dense explicit-V likelihood (NumPy) evaluated at OpenEcon's estimates
   (value, first-order conditions, brute-force SciPy maximum), numerical Hessians for
   the variance-parameter standard errors, numerical per-group scores for Stata's
   robust covariance, and invariances (row order, rescaling, frequency weights).
4. GLMMs and xt models: a NumPy Gauss-Hermite likelihood (non-adaptive, the identical
   rule for intmethod='ghermite'), first-order conditions at the estimates, numerical
   Hessians, and statsmodels conditional models.
"""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, gammaln, logsumexp, roots_hermite
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe

# ---- shared helpers ---------------------------------------------------------------------


def terms(result):
    return {c.term: c for c in result.coefficients}


def printed(value, text, units=0.5):
    """|value - printed| within ``units`` of the last printed digit (0.5: exact rounding)."""
    decimals = len(text.split(".")[1]) if "." in text else 0
    assert abs(value - float(text)) <= units * 10 ** -decimals * 1.0001, (value, text)


# ---- 1. Stata manual parity: ships data ------------------------------------------------------

_SHIPS = """A 60 60 127 0;A 60 75 63 0;A 65 60 1095 3;A 65 75 1095 4;A 70 60 1512 6;
A 70 75 3353 18;A 75 60 0 0;A 75 75 2244 11;B 60 60 44882 39;B 60 75 17176 29;
B 65 60 28609 58;B 65 75 20370 53;B 70 60 7064 12;B 70 75 13099 44;B 75 60 0 0;
B 75 75 7117 18;C 60 60 1179 1;C 60 75 552 1;C 65 60 781 0;C 65 75 676 1;C 70 60 783 6;
C 70 75 1948 2;C 75 60 0 0;C 75 75 274 1;D 60 60 251 0;D 60 75 105 0;D 65 60 288 0;
D 65 75 192 0;D 70 60 349 2;D 70 75 1208 11;D 75 60 0 0;D 75 75 2051 4;E 60 60 45 0;
E 60 75 0 0;E 65 60 789 7;E 65 75 437 7;E 70 60 1157 5;E 70 75 2161 12;E 75 60 0 0;
E 75 75 542 1"""
SHIP_X = ["op_75_79", "co_65_69", "co_70_74", "co_75_79"]


@pytest.fixture(scope="module")
def ships():
    rows = [r.split() for r in _SHIPS.replace("\n", "").split(";")]
    frame = pd.DataFrame(rows, columns=["ship", "year", "period", "service", "accident"])
    for name in ("year", "period", "service", "accident"):
        frame[name] = frame[name].astype(float)
    frame = frame[frame.service > 0].reset_index(drop=True)
    frame["op_75_79"] = (frame.period == 75) * 1.0
    for start in (65, 70, 75):
        frame[f"co_{start}_{start + 4}"] = (frame.year == start) * 1.0
    return frame


def check_irr(result, expected, units=0.5):
    """expected: {term: (IRR, se(IRR))} as printed (se of IRR = IRR * se(b))."""
    table = terms(result)
    for name, (irr, se) in expected.items():
        estimate = math.exp(table[name].estimate)
        printed(estimate, irr, units)
        printed(estimate * table[name].std_error, se, units)


def test_xtpoisson_re_gamma_reproduces_stata_manual(ships):
    result = oe.xtpoisson(data=ships, y="accident", x=SHIP_X, panel="ship", exposure="service")
    assert result.nobs == 34 and result.metrics["n_groups"] == 5
    printed(result.metrics["log_likelihood"], "-74.811217")
    check_irr(result, {"op_75_79": ("1.466305", ".1734005"), "co_65_69": ("2.032543", ".304083"),
                       "co_70_74": ("2.356853", ".3999259"), "co_75_79": ("1.641913", ".3811398"),
                       "Intercept": (".0013724", ".0002992")})
    lnalpha = terms(result)["/lnalpha"]
    printed(lnalpha.estimate, "-2.368406")
    printed(lnalpha.std_error, ".8474597")
    printed(lnalpha.ci_low, "-4.029397")
    printed(lnalpha.ci_high, "-.7074155")
    alpha = result.extra["alpha"]
    for key, text in (("estimate", ".0936298"), ("std_error", ".0793475"),
                      ("ci_low", ".0177851"), ("ci_high", ".4929165")):
        printed(alpha[key], text)
    printed(result.tests["model"]["statistic"], "50.90")
    assert result.tests["model"]["df"] == 4
    printed(result.tests["alpha"]["statistic"], "10.61")
    printed(result.tests["alpha"]["p_value"], "0.001")


def test_xtpoisson_re_normal_reproduces_stata_manual(ships):
    result = oe.xtpoisson(data=ships, y="accident", x=SHIP_X, panel="ship", exposure="service",
                          normal=True)
    printed(result.metrics["log_likelihood"], "-74.780982")
    check_irr(result, {"op_75_79": ("1.466677", ".1734403"), "co_65_69": ("2.032604", ".3040933"),
                       "co_70_74": ("2.357045", ".3998397"), "co_75_79": ("1.646935", ".3820235"),
                       "Intercept": (".0013075", ".0002775")})
    lnsig2u = terms(result)["/lnsig2u"]
    printed(lnsig2u.estimate, "-2.351868")
    printed(lnsig2u.std_error, ".8586262")
    printed(lnsig2u.ci_low, "-4.034745")
    printed(lnsig2u.ci_high, "-.6689918")
    # Stata freezes its adaptive quadrature once the log likelihood changes by < 1e-6, so
    # derived quantities may differ in the 7th significant digit: 2 units of the last digit.
    sigma = result.extra["sigma_u"]
    for key, text in (("estimate", ".3085306"), ("std_error", ".1324562"),
                      ("ci_low", ".1330045"), ("ci_high", ".7156988")):
        printed(sigma[key], text, units=2)
    printed(result.tests["model"]["statistic"], "50.95")
    printed(result.tests["sigma_u"]["statistic"], "10.67")
    assert result.extra["integration"]["points"] == 12


def test_xtpoisson_fe_reproduces_stata_manual(ships):
    result = oe.xtpoisson(data=ships, y="accident", x=SHIP_X, panel="ship", exposure="service",
                          model="fe")
    printed(result.metrics["log_likelihood"], "-54.641859")
    check_irr(result, {"op_75_79": ("1.468831", ".1737218"), "co_65_69": ("2.008002", ".3004803"),
                       "co_70_74": ("2.26693", ".384865"), "co_75_79": ("1.573695", ".3669393")})
    # Stata's xtpoisson, fe reports a Wald chi2 (e(chi2type) = "Wald"), not LR.
    assert result.tests["model"]["label"].startswith("Wald")
    printed(result.tests["model"]["statistic"], "48.44")
    assert result.metrics["group_size_min"] == 6 and result.metrics["group_size_max"] == 7


def test_xtpoisson_pa_robust_reproduces_stata_manual(ships):
    """Exchangeable GEE on unbalanced panels (7,7,7,7,6) with 5 clusters: checks the pooled
    exchangeable estimator and the G/(G-1) factor of the semi-robust covariance."""
    result = oe.xtpoisson(data=ships, y="accident", x=SHIP_X, panel="ship", exposure="service",
                          model="pa", covariance="robust")
    # Stata's printed iteration log stops at a coefficient tolerance of 4.4e-7; OpenEcon
    # iterates to 1e-10, hence up to 4 units of the 7th significant digit (6e-7 relative).
    check_irr(result, {"op_75_79": ("1.483299", ".1197901"), "co_65_69": ("2.038477", ".1809524"),
                       "co_70_74": ("2.643467", ".4093947"), "co_75_79": ("1.876656", ".33075"),
                       "Intercept": (".0010255", ".0000721")}, units=4)
    printed(result.tests["model"]["statistic"], "252.94")
    assert result.metrics["scale"] == 1.0


# ---- 2. xtgee: Stata's Methods and formulas in NumPy ------------------------------------------


def gee_panel(seed=0, panels=70):
    """Panels of 1..6 consecutive periods starting at different calendar times."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 7, size=panels)
    starts = rng.integers(0, 4, size=panels)
    ids = np.repeat(np.arange(panels), sizes)
    t = np.concatenate([s + np.arange(n) for s, n in zip(starts, sizes)])
    n = len(ids)
    frame = pd.DataFrame({"id": ids, "t": t, "x": rng.normal(size=n),
                          "w": rng.normal(size=n) * 3 + 10})
    u = rng.normal(size=panels)[ids]
    e = np.zeros(n)
    for i in range(n):                       # AR(1) noise within panels (test data only)
        e[i] = rng.normal() + (0.5 * e[i - 1] if i and ids[i] == ids[i - 1] else 0.0)
    frame["y"] = 1 + 0.5 * frame.x - 0.1 * frame.w + 0.7 * u + e
    frame["c"] = rng.poisson(np.exp(0.2 + 0.3 * frame.x + 0.4 * u))
    frame["b"] = (rng.random(n) < expit(0.3 * frame.x + u)).astype(float)
    return frame


_LINK = {"gaussian": (lambda e: e, lambda e: np.ones_like(e), lambda m: np.ones_like(m)),
         "poisson": (np.exp, np.exp, lambda m: m),
         "binomial": (expit, lambda e: expit(e) * expit(-e), lambda m: m * (1 - m))}


def stata_gee(frame, y, xcols, family, corr, order=1, nmp=False):
    """GEE with the correlation estimators printed in Stata's [XT] xtgee manual."""
    lag = {"ar1": 1, "stationary": order, "nonstationary": order}.get(corr, 0)
    frame = frame.sort_values(["id", "t"])
    if lag:
        frame = frame[frame.groupby("id").id.transform("size") > lag]
    frame = frame.reset_index(drop=True)
    X = np.column_stack([np.ones(len(frame)), frame[xcols].to_numpy()])
    Y = frame[y].to_numpy(float)
    N, p = X.shape
    panels = [np.flatnonzero(frame.id.to_numpy() == i) for i in frame.id.unique()]
    m, T = len(panels), max(len(i) for i in panels)
    inverse, derivative, variance = _LINK[family]

    def pieces(beta):
        eta = X @ beta
        mu = inverse(eta)
        sd = np.sqrt(variance(mu))
        return (Y - mu) / sd, X * (derivative(eta) / sd)[:, None]

    def working(r):
        if corr == "independent":
            return np.eye(T)
        if corr == "exchangeable":
            pairs = sum(r[i].sum() ** 2 - (r[i] ** 2).sum() for i in panels)
            a = (pairs / sum(len(i) * (len(i) - 1) for i in panels)) / ((r ** 2).sum() / N)
            return (1 - a) * np.eye(T) + a * np.ones((T, T))
        lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
        scale = sum((r[i] ** 2).sum() / len(i) for i in panels)
        if corr in {"ar1", "stationary"}:
            rho = [sum(r[i][:-k] @ r[i][k:] / len(i) for i in panels if len(i) > k) / scale
                   for k in range(1, (1 if corr == "ar1" else order) + 1)]
            if corr == "ar1":
                return rho[0] ** lags
            R = np.eye(T)
            for k, value in enumerate(rho, start=1):
                R[lags == k] = value
            return R
        alpha = np.zeros((T, T))
        for i in panels:                                    # r = 0 beyond n_i
            padded = np.zeros(T)
            padded[:len(i)] = r[i]
            alpha += np.outer(padded, padded)
        sizes = np.array([len(i) for i in panels])
        counts = np.array([[(sizes > max(a, b)).sum() for b in range(T)] for a in range(T)])
        alpha = m * alpha / counts / scale
        keep = (lags > 0) & ((lags <= order) if corr == "nonstationary" else True)
        return np.where(keep, alpha, 0.0) + np.eye(T)

    def assemble(beta, R):
        r, d = pieces(beta)
        bread, scores = np.zeros((p, p)), []
        for i in panels:
            inv = np.linalg.inv(R[:len(i), :len(i)])       # upper-left block
            bread += d[i].T @ inv @ d[i]
            scores.append(d[i].T @ inv @ r[i])
        return bread, np.array(scores)

    beta = np.zeros(p)
    if family == "gaussian":
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    for structure in ("independent", corr):
        for _ in range(500):
            r, _ = pieces(beta)
            R = np.eye(T) if structure == "independent" else working(r)
            bread, scores = assemble(beta, R)
            step = np.linalg.solve(bread, scores.sum(0))
            beta = beta + step
            if np.abs(step).max() < 1e-13:
                break
    r, _ = pieces(beta)
    R = working(r)
    bread, scores = assemble(beta, R)
    phi = (r ** 2).sum() / (N - p if nmp else N)
    return {"beta": beta, "R": R, "phi": phi, "bread": bread, "scores": scores, "N": N, "m": m}


@pytest.fixture(scope="module")
def unbalanced():
    return gee_panel()


@pytest.mark.parametrize("corr,order", [("exchangeable", 1), ("independent", 1), ("ar1", 1),
                                        ("stationary", 2), ("nonstationary", 2),
                                        ("unstructured", 1)])
@pytest.mark.parametrize("family,y", [("gaussian", "y"), ("poisson", "c"), ("binomial", "b")])
def test_xtgee_matches_stata_formulas(unbalanced, corr, order, family, y):
    options = {"corr_order": order} if corr in {"stationary", "nonstationary"} else {}
    try:
        ref = stata_gee(unbalanced, y, ["x", "w"], family, corr, order)
    except np.linalg.LinAlgError:
        pytest.skip("reference working correlation singular")
    if np.linalg.eigvalsh(ref["R"]).min() <= 0:
        with pytest.raises(oe.AnalysisError) as error:
            oe.xtgee(data=unbalanced, y=y, x=["x", "w"], panel="id", time="t", family=family,
                     corr=corr, **options)
        assert error.value.code == "working_correlation_not_pd"
        return
    fit = oe.xtgee(data=unbalanced, y=y, x=["x", "w"], panel="id", time="t", family=family,
                   corr=corr, **options)
    robust = oe.xtgee(data=unbalanced, y=y, x=["x", "w"], panel="id", time="t", family=family,
                      corr=corr, covariance="robust", **options)
    assert fit.nobs == ref["N"] and fit.metrics["n_groups"] == ref["m"]
    assert_allclose([c.estimate for c in fit.coefficients], ref["beta"], rtol=1e-8, atol=1e-11)
    scale = ref["phi"] if family == "gaussian" else 1.0
    assert_allclose(fit.metrics["scale"], scale, rtol=1e-9)
    bread_inv = np.linalg.inv(ref["bread"])
    assert_allclose(np.array(fit.covariance_matrix), scale * bread_inv, rtol=1e-7)
    m = ref["m"]
    sandwich = m / (m - 1) * bread_inv @ ref["scores"].T @ ref["scores"] @ bread_inv
    assert_allclose(np.array(robust.covariance_matrix), sandwich, rtol=1e-7)
    T = ref["R"].shape[0]
    if corr != "independent":
        assert_allclose(np.array(fit.extra["working_correlation"]), ref["R"][:T, :T],
                        rtol=1e-8, atol=1e-12)
    if corr in {"exchangeable", "ar1"}:
        assert_allclose(fit.extra["alpha"][0], ref["R"][0, 1], rtol=1e-8)
    wald = fit.tests["model"]
    b = np.array([c.estimate for c in fit.coefficients])[1:]
    v = np.array(fit.covariance_matrix)[1:, 1:]
    assert_allclose(wald["statistic"], b @ np.linalg.solve(v, b), rtol=1e-8)


def test_xtgee_short_panels_are_dropped_as_stata(unbalanced):
    fit = oe.xtgee(data=unbalanced, y="y", x=["x"], panel="id", time="t", corr="stationary",
                   corr_order=3)
    sizes = unbalanced.groupby("id").size()
    assert fit.metrics["n_groups"] == int((sizes > 3).sum())
    assert fit.nobs == int(sizes[sizes > 3].sum())
    assert any("dropped because of too few observations" in w for w in fit.warnings)
    # Exchangeable and unstructured keep every panel (singletons included).
    for corr in ("exchangeable", "unstructured"):
        try:
            kept = oe.xtgee(data=unbalanced, y="y", x=["x"], panel="id", time="t", corr=corr)
        except oe.AnalysisError as error:
            assert error.code == "working_correlation_not_pd"
            continue
        assert kept.nobs == len(unbalanced) and kept.metrics["n_groups"] == len(sizes)


def test_xtgee_alpha_ignores_nmp_and_ar1_is_yule_walker():
    frame = gee_panel(seed=3, panels=120)
    for corr in ("exchangeable", "ar1"):
        plain = oe.xtgee(data=frame, y="y", x=["x"], panel="id", time="t", corr=corr)
        nmp = oe.xtgee(data=frame, y="y", x=["x"], panel="id", time="t", corr=corr, nmp=True)
        assert_allclose(nmp.extra["alpha"], plain.extra["alpha"], rtol=1e-12)
        n, p = nmp.nobs, 2
        assert_allclose(nmp.metrics["scale"], plain.metrics["scale"] * n / (n - p), rtol=1e-12)
        assert_allclose(np.array(nmp.covariance_matrix),
                        np.array(plain.covariance_matrix) * n / (n - p), rtol=1e-9)
    # Balanced panels: the AR(1) estimate is sum r_t r_t+1 / sum r_t^2 (divisor n_i, not
    # n_i - 1), computed here from the Pearson residuals at the reported coefficients.
    balanced = frame[frame.groupby("id").id.transform("size") >= 4]
    balanced = balanced[balanced.groupby("id").cumcount() < 4]
    fit = oe.xtgee(data=balanced, y="y", x=["x"], panel="id", time="t", corr="ar1")
    b = [c.estimate for c in fit.coefficients]
    r = (balanced.y - b[0] - b[1] * balanced.x).to_numpy().reshape(-1, 4)
    assert_allclose(fit.extra["alpha"][0], (r[:, :-1] * r[:, 1:]).sum() / (r ** 2).sum(),
                    rtol=1e-9)


def test_xtgee_invariances(unbalanced):
    base = oe.xtgee(data=unbalanced, y="c", x=["x", "w"], panel="id", time="t",
                    family="poisson", corr="ar1", covariance="robust")
    shuffled = unbalanced.sample(frac=1.0, random_state=4).reset_index(drop=True)
    again = oe.xtgee(data=shuffled, y="c", x=["x", "w"], panel="id", time="t",
                     family="poisson", corr="ar1", covariance="robust")
    assert_allclose([c.estimate for c in again.coefficients],
                    [c.estimate for c in base.coefficients], rtol=1e-10)
    assert_allclose(again.covariance_matrix, base.covariance_matrix, rtol=1e-9)
    scaled = oe.xtgee(data=unbalanced.assign(w=unbalanced.w * 1000), y="c", x=["x", "w"],
                      panel="id", time="t", family="poisson", corr="ar1", covariance="robust")
    assert_allclose(terms(scaled)["w"].estimate * 1000, terms(base)["w"].estimate, rtol=1e-8)
    assert_allclose(terms(scaled)["w"].std_error * 1000, terms(base)["w"].std_error, rtol=1e-7)
    assert_allclose(terms(scaled)["Intercept"].estimate, terms(base)["Intercept"].estimate,
                    rtol=1e-8)
    # A panel's calendar start does not matter: R is indexed by position within the panel.
    moved = oe.xtgee(data=unbalanced.assign(t=unbalanced.t + 7 * (unbalanced.id % 3)), y="c",
                     x=["x", "w"], panel="id", time="t", family="poisson", corr="ar1",
                     covariance="robust")
    assert_allclose([c.estimate for c in moved.coefficients],
                    [c.estimate for c in base.coefficients], rtol=1e-10)


# ---- 3. mixed: dense explicit-V likelihood --------------------------------------------------


def mixed_data(seed=0, schools=12, classes=3, size=5):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(schools), classes * size)
    k = np.tile(np.repeat(np.arange(classes), size), schools)
    n = len(s)
    frame = pd.DataFrame({"s": s, "k": k, "x": rng.normal(size=n),
                          "w": rng.normal(size=n) * 2 + 5})
    cls = s * classes + k
    frame["y"] = (1 + 0.6 * frame.x - 0.2 * frame.w + 0.8 * rng.normal(size=schools)[s]
                  + 0.7 * rng.normal(size=schools * classes)[cls]
                  + 0.5 * rng.normal(size=schools * classes)[cls] * frame.x
                  + rng.normal(size=n))
    frame["f"] = rng.integers(1, 4, size=n).astype(float)
    frame["pair"] = s // 2
    return frame


def dense_parts(frame, xcols, zcols, low, top, G, g, s2, intercept=True):
    """Per-top-group (X_s, y_s, V_s) of the linear mixed model with explicit V."""
    X = frame[xcols].to_numpy(float)
    if intercept:
        X = np.column_stack([np.ones(len(frame)), X])
    Z = np.column_stack([np.ones(len(frame)) if c == "_cons" else frame[c].to_numpy(float)
                         for c in zcols])
    Y = frame["y"].to_numpy(float)
    lows = frame[low].to_numpy() if top is None else (frame[top].astype(str) + "/"
                                                       + frame[low].astype(str)).to_numpy()
    tops = lows if top is None else frame[top].to_numpy()
    parts = []
    for label in pd.unique(tops):
        i = np.flatnonzero(tops == label)
        same = lows[i][:, None] == lows[i][None, :]
        V = s2 * np.eye(len(i)) + same * (Z[i] @ G @ Z[i].T) + g
        parts.append((X[i], Y[i], V))
    return parts


def dense_group_loglik(parts, beta):
    out = []
    for X, Y, V in parts:
        r = Y - X @ beta
        out.append(-0.5 * (len(Y) * math.log(2 * math.pi) + np.linalg.slogdet(V)[1]
                           + r @ np.linalg.solve(V, r)))
    return np.array(out)


def dense_profile(parts, reml=False):
    xvx = sum(X.T @ np.linalg.solve(V, X) for X, Y, V in parts)
    xvy = sum(X.T @ np.linalg.solve(V, Y) for X, Y, V in parts)
    beta = np.linalg.solve(xvx, xvy)
    value = dense_group_loglik(parts, beta).sum()
    if reml:
        value += -0.5 * np.linalg.slogdet(xvx)[1] + 0.5 * len(beta) * math.log(2 * math.pi)
    return value, beta, xvx


SLOPE_TERMS = ["/var(_cons[s])", "/var(x[k])", "/var(_cons[k])", "/cov(x,_cons[k])",
               "/var(Residual)"]


def slope_parts(frame, phi):
    g, vx, vc, cov, s2 = phi
    G = np.array([[vx, cov], [cov, vc]])
    return dense_parts(frame, ["x", "w"], ["x", "_cons"], "k", "s", G, g, s2)


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_mixed_two_level_slopes_against_dense_likelihood(method):
    frame = mixed_data()
    reml = method == "reml"
    fit = oe.mixed(data=frame, y="y", x=["x", "w"], group=["s", "k"], random=["x"],
                   covstructure="unstructured", method=method)
    table = terms(fit)
    phi = np.array([table[t].estimate for t in SLOPE_TERMS])
    value, beta, xvx = dense_profile(slope_parts(frame, phi), reml)
    assert_allclose(fit.metrics["log_likelihood"], value, rtol=1e-11)
    assert_allclose([table[t].estimate for t in ("Intercept", "x", "w")], beta, rtol=1e-9)
    assert_allclose([table[t].std_error for t in ("Intercept", "x", "w")],
                    np.sqrt(np.diag(np.linalg.inv(xvx))), rtol=1e-9)

    def profile(p):
        return dense_profile(slope_parts(frame, p), reml)[0]

    # First-order conditions in the variance metric, and the observed information there
    # (the delta-method covariance of the reported variances equals inv(-H) at the maximum).
    gradient = approx_fprime(phi, profile, centered=True)
    assert np.abs(gradient * phi).max() < 1e-5
    cov = np.linalg.inv(-approx_hess(phi, profile))
    reported = np.array(fit.covariance_matrix)[3:, 3:]
    assert_allclose(reported, cov, rtol=2e-4, atol=1e-7)
    assert_allclose(np.array(fit.covariance_matrix)[:3, 3:], 0.0)     # block diagonal
    k = 3 + 5
    assert_allclose(fit.metrics["aic"], -2 * value + 2 * k, rtol=1e-12)
    assert_allclose(fit.metrics["bic"], -2 * value + math.log(len(frame)) * k, rtol=1e-12)


def _robust_oracle(frame, parts_of, phi, beta, clusters=None):
    """Stata's mixed vce(robust): block-diagonal OIM bread, group scores of (b, phi)."""
    parts = parts_of(phi)
    k_b, k_p = len(beta), len(phi)
    _, _, xvx = dense_profile(parts)
    profile_hessian = approx_hess(phi, lambda p: dense_profile(parts_of(p))[0])
    bread = np.zeros((k_b + k_p, k_b + k_p))
    bread[:k_b, :k_b] = np.linalg.inv(xvx)
    bread[k_b:, k_b:] = np.linalg.inv(-profile_hessian)

    def groups(params):
        return dense_group_loglik(parts_of(params[k_b:]), params[:k_b])

    scores = approx_fprime(np.r_[beta, phi], groups, centered=True)
    if clusters is not None:
        scores = pd.DataFrame(scores).groupby(clusters).sum().to_numpy()
    count = scores.shape[0]
    return count / (count - 1) * bread @ scores.T @ scores @ bread


def test_mixed_robust_matches_numerical_group_scores():
    frame = mixed_data(seed=2, schools=30, classes=1, size=7)
    fit = oe.mixed(data=frame, y="y", x=["x", "w"], group="s", random=["x"],
                   covariance="robust")
    table = terms(fit)
    names = ["/var(x[s])", "/var(_cons[s])", "/var(Residual)"]
    phi = np.array([table[t].estimate for t in names])
    beta = np.array([table[t].estimate for t in ("Intercept", "x", "w")])

    def parts_of(p):
        return dense_parts(frame, ["x", "w"], ["x", "_cons"], "s", None,
                           np.diag(p[:2]), 0.0, p[2])

    expected = _robust_oracle(frame, parts_of, phi, beta)
    assert_allclose(np.array(fit.covariance_matrix), expected, rtol=2e-4, atol=1e-9)
    assert "lr_vs_linear" not in fit.tests
    assert fit.inference["cluster_count"] == 30


def test_mixed_two_level_cluster_matches_numerical_group_scores():
    frame = mixed_data(seed=5, schools=16, classes=3, size=4)
    fit = oe.mixed(data=frame, y="y", x=["x"], group=["s", "k"], cluster="pair")
    table = terms(fit)
    names = ["/var(_cons[s])", "/var(_cons[k])", "/var(Residual)"]
    phi = np.array([table[t].estimate for t in names])
    beta = np.array([table[t].estimate for t in ("Intercept", "x")])

    def parts_of(p):
        return dense_parts(frame, ["x"], ["_cons"], "k", "s", np.array([[p[1]]]), p[0], p[2])

    clusters = frame.groupby("s").pair.first().to_numpy()
    expected = _robust_oracle(frame, parts_of, phi, beta, clusters=clusters)
    assert_allclose(np.array(fit.covariance_matrix), expected, rtol=2e-4, atol=1e-9)
    assert fit.inference["cluster_count"] == 8


def test_mixed_frequency_weights_equal_duplicated_rows():
    frame = mixed_data(seed=7)
    expanded = frame.loc[frame.index.repeat(frame.f.astype(int))].reset_index(drop=True)
    for options in ({"method": "reml"}, {"covariance": "robust"}):
        weighted = oe.mixed(data=frame, y="y", x=["x", "w"], group=["s", "k"], weights="f",
                            weight_type="fweight", **options)
        plain = oe.mixed(data=expanded, y="y", x=["x", "w"], group=["s", "k"], **options)
        assert weighted.nobs == plain.nobs == len(expanded)
        assert_allclose(weighted.metrics["log_likelihood"], plain.metrics["log_likelihood"],
                        rtol=1e-10)
        assert_allclose([c.estimate for c in weighted.coefficients],
                        [c.estimate for c in plain.coefficients], rtol=1e-6)
        assert_allclose(weighted.covariance_matrix, plain.covariance_matrix, rtol=1e-5,
                        atol=1e-12)


def test_mixed_invariances():
    frame = mixed_data(seed=9)
    base = oe.mixed(data=frame, y="y", x=["x", "w"], group="s", random=["x"],
                    covstructure="unstructured")
    shuffled = oe.mixed(data=frame.sample(frac=1.0, random_state=1), y="y", x=["x", "w"],
                        group="s", random=["x"], covstructure="unstructured")
    assert_allclose(shuffled.metrics["log_likelihood"], base.metrics["log_likelihood"],
                    rtol=1e-12)
    assert_allclose([c.estimate for c in shuffled.coefficients],
                    [c.estimate for c in base.coefficients], rtol=1e-6, atol=1e-9)
    # A regressor with a huge level and scale (not a random slope) changes nothing else.
    moved = oe.mixed(data=frame.assign(w=frame.w * 1e4 + 1e8), y="y", x=["x", "w"], group="s",
                     random=["x"], covstructure="unstructured")
    assert_allclose(moved.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-10)
    assert_allclose(terms(moved)["w"].estimate * 1e4, terms(base)["w"].estimate, rtol=1e-6)
    assert_allclose(terms(moved)["x"].estimate, terms(base)["x"].estimate, rtol=1e-6)
    assert_allclose(terms(moved)["/var(x[s])"].estimate, terms(base)["/var(x[s])"].estimate,
                    rtol=1e-5)
    # Scaling y by 10: fixed effects x10, variances x100, log likelihood - N ln 10.
    scaled = oe.mixed(data=frame.assign(y=frame.y * 10), y="y", x=["x", "w"], group="s",
                      random=["x"], covstructure="unstructured")
    assert_allclose(scaled.metrics["log_likelihood"],
                    base.metrics["log_likelihood"] - len(frame) * math.log(10), rtol=1e-10)
    for term in ("x", "w", "Intercept"):
        assert_allclose(terms(scaled)[term].estimate, 10 * terms(base)[term].estimate,
                        rtol=1e-6)
    for term in ("/var(x[s])", "/var(_cons[s])", "/var(Residual)", "/cov(x,_cons[s])"):
        assert_allclose(terms(scaled)[term].estimate, 100 * terms(base)[term].estimate,
                        rtol=1e-5)


# ---- 4. GLMMs and xt models: NumPy Gauss-Hermite likelihood ---------------------------------


def glmm_data(seed=0, groups=50, size=6, slope_sd=0.6):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), size)
    n = len(g)
    frame = pd.DataFrame({"g": g, "x": rng.normal(size=n), "w": rng.normal(size=n) + 3})
    u0, u1 = rng.normal(size=groups), slope_sd * rng.normal(size=groups)
    eta = -0.2 + 0.7 * frame.x - 0.3 * (frame.w - 3) + u0[g] + u1[g] * frame.x
    frame["y"] = (rng.random(n) < expit(eta)).astype(float)
    frame["e"] = rng.uniform(0.5, 2.0, size=n)
    frame["c"] = rng.poisson(frame.e * np.exp(0.1 + 0.4 * frame.x + 0.5 * u0[g]
                                               + 0.4 * u1[g] * frame.x))
    frame["cl"] = g // 5
    return frame


def gh_groups(frame, family, y, xcols, beta, sds, zcols, points, offset=None):
    """Per-group log likelihoods by the K-point (product) Gauss-Hermite rule."""
    X = np.column_stack([np.ones(len(frame)), frame[xcols].to_numpy()])
    eta0 = X @ beta + (0.0 if offset is None else offset)
    nodes, weights = roots_hermite(points)
    q = len(zcols)
    grid = np.array(np.meshgrid(*[nodes] * q, indexing="ij")).reshape(q, -1).T
    log_w = (np.log(np.array(np.meshgrid(*[weights] * q, indexing="ij")).reshape(q, -1))
             .sum(0) - 0.5 * q * math.log(math.pi))
    Z = np.column_stack([np.ones(len(frame)) if c == "_cons" else frame[c].to_numpy()
                         for c in zcols])
    eta = eta0[:, None] + math.sqrt(2) * (Z * np.asarray(sds)) @ grid.T
    Y = frame[y].to_numpy(float)[:, None]
    if family == "logit":
        log_f = -np.logaddexp(0, -(2 * Y - 1) * eta)
    elif family == "probit":
        from scipy.stats import norm
        log_f = norm.logcdf((2 * Y - 1) * eta)
    else:
        log_f = Y * eta - np.exp(eta) - gammaln(Y + 1)
    sums = pd.DataFrame(log_f).groupby(frame.g.to_numpy() if "g" in frame else
                                       frame.id.to_numpy()).sum().to_numpy()
    return logsumexp(sums + log_w[None, :], axis=1)


@pytest.mark.parametrize("family,command,y", [("logit", "melogit", "y"),
                                              ("probit", "meprobit", "y"),
                                              ("poisson", "mepoisson", "c")])
def test_ghermite_is_the_numpy_rule_and_estimates_solve_it(family, command, y):
    frame = glmm_data(seed=1)
    fit = getattr(oe, command)(data=frame, y=y, x=["x", "w"], group="g", random=["x"],
                               intmethod="ghermite", intpoints=9)
    table = terms(fit)
    beta = np.array([table[t].estimate for t in ("Intercept", "x", "w")])
    sds = np.sqrt([table["/var(x[g])"].estimate, table["/var(_cons[g])"].estimate])

    def loglik(params):
        return gh_groups(frame, family, y, ["x", "w"], params[:3], np.exp(params[3:]),
                         ["x", "_cons"], 9).sum()

    params = np.r_[beta, np.log(sds)]
    assert_allclose(fit.metrics["log_likelihood"], loglik(params), rtol=1e-12)
    assert np.abs(approx_fprime(params, loglik, centered=True)).max() < 1e-5
    # Observed information in (b, ln sd), delta method to the variances.
    cov = np.linalg.inv(-approx_hess(params, loglik))
    jac = np.diag(np.r_[1, 1, 1, 2 * sds ** 2])
    # Precision of the numerical Hessian (OpenEcon's is analytic): 5e-4 relative.
    assert_allclose(np.array(fit.covariance_matrix), jac @ cov @ jac.T, rtol=5e-4, atol=1e-8)


def test_adaptive_quadrature_converges_to_the_integral():
    frame = glmm_data(seed=2, slope_sd=0.0)
    fit = oe.meprobit(data=frame, y="y", x=["x", "w"], group="g", intpoints=25)
    table = terms(fit)
    beta = np.array([table[t].estimate for t in ("Intercept", "x", "w")])
    sd = math.sqrt(table["/var(_cons[g])"].estimate)
    exact = gh_groups(frame, "probit", "y", ["x", "w"], beta, [sd], ["_cons"], 120).sum()
    assert_allclose(fit.metrics["log_likelihood"], exact, rtol=1e-10)
    default = oe.meprobit(data=frame, y="y", x=["x", "w"], group="g")       # 7 points
    assert_allclose(default.metrics["log_likelihood"], exact, rtol=1e-5)


def test_mepoisson_robust_matches_numerical_group_scores():
    frame = glmm_data(seed=3)
    offset = np.log(frame.e.to_numpy())
    fit = oe.mepoisson(data=frame, y="c", x=["x"], group="g", random=["x"], exposure="e",
                       intmethod="ghermite", intpoints=15, covariance="robust")
    table = terms(fit)
    beta = np.array([table[t].estimate for t in ("Intercept", "x")])
    sds = np.sqrt([table["/var(x[g])"].estimate, table["/var(_cons[g])"].estimate])
    params = np.r_[beta, np.log(sds)]

    def groups(p):
        return gh_groups(frame, "poisson", "c", ["x"], p[:2], np.exp(p[2:]), ["x", "_cons"],
                         15, offset)

    scores = approx_fprime(params, groups, centered=True)
    bread = np.linalg.inv(-approx_hess(params, lambda p: groups(p).sum()))
    count = scores.shape[0]
    cov = count / (count - 1) * bread @ scores.T @ scores @ bread
    jac = np.diag(np.r_[1, 1, 2 * sds ** 2])
    assert_allclose(np.array(fit.covariance_matrix), jac @ cov @ jac.T, rtol=2e-4, atol=1e-10)
    assert "lr_vs_pooled" not in fit.tests
    clustered = oe.mepoisson(data=frame, y="c", x=["x"], group="g", random=["x"],
                             exposure="e", intmethod="ghermite", intpoints=15, cluster="cl")
    s = pd.DataFrame(scores).groupby(frame.groupby("g").cl.first().to_numpy()).sum().to_numpy()
    cov = len(s) / (len(s) - 1) * bread @ s.T @ s @ bread
    assert_allclose(np.array(clustered.covariance_matrix), jac @ cov @ jac.T, rtol=2e-4,
                    atol=1e-10)


def test_glmm_invariances():
    frame = glmm_data(seed=4, slope_sd=0.0)
    base = oe.melogit(data=frame, y="y", x=["x", "w"], group="g")
    shuffled = oe.melogit(data=frame.sample(frac=1.0, random_state=2), y="y", x=["x", "w"],
                          group="g")
    assert_allclose(shuffled.metrics["log_likelihood"], base.metrics["log_likelihood"],
                    rtol=1e-11)
    assert_allclose([c.estimate for c in shuffled.coefficients],
                    [c.estimate for c in base.coefficients], rtol=1e-7, atol=1e-10)
    moved = oe.melogit(data=frame.assign(w=frame.w * 1e3 + 1e6), y="y", x=["x", "w"], group="g")
    assert_allclose(moved.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-10)
    assert_allclose(terms(moved)["w"].estimate * 1e3, terms(base)["w"].estimate, rtol=1e-6)
    assert_allclose(terms(moved)["/var(_cons[g])"].estimate,
                    terms(base)["/var(_cons[g])"].estimate, rtol=1e-6)


def test_xtlogit_re_against_exact_integral_and_fe_against_statsmodels():
    from statsmodels.discrete.conditional_models import ConditionalLogit

    frame = glmm_data(seed=5, groups=80, size=5, slope_sd=0.0).rename(columns={"g": "id"})
    fit = oe.xtlogit(data=frame, y="y", x=["x", "w"], panel="id")
    table = terms(fit)
    beta = np.array([table[t].estimate for t in ("Intercept", "x", "w")])
    lnsig2u = table["/lnsig2u"].estimate

    def loglik(params):
        return gh_groups(frame, "logit", "y", ["x", "w"], params[:3],
                         [math.exp(params[3] / 2)], ["_cons"], 100).sum()

    params = np.r_[beta, lnsig2u]
    assert_allclose(fit.metrics["log_likelihood"], loglik(params), rtol=1e-8)
    assert np.abs(approx_fprime(params, loglik, centered=True)).max() < 1e-4
    se = np.sqrt(np.diag(np.linalg.inv(-approx_hess(params, loglik))))
    assert_allclose([c.std_error for c in fit.coefficients], se, rtol=1e-4)
    # Conditional fixed-effects logit on unbalanced panels.
    unbalanced = frame[~((frame.id % 4 == 0) & (frame.groupby("id").cumcount() >= 3))]
    fe = oe.xtlogit(data=unbalanced, y="y", x=["x", "w"], panel="id", model="fe")
    sums = unbalanced.groupby("id").y.agg(["sum", "size"])
    keep = sums[(sums["sum"] > 0) & (sums["sum"] < sums["size"])].index
    used = unbalanced[unbalanced.id.isin(keep)]
    ref = ConditionalLogit(used.y, used[["x", "w"]], groups=used.id).fit(disp=0,
                                                                         method="newton")
    assert_allclose([c.estimate for c in fe.coefficients], ref.params, rtol=1e-7)
    assert_allclose([c.std_error for c in fe.coefficients], ref.bse, rtol=1e-6)
    assert_allclose(fe.metrics["log_likelihood"], ref.llf, rtol=1e-10)
    # Stata's e(ll_0): the conditional likelihood at b = 0 is -sum ln C(n_i, k_i).
    counts = used.groupby("id").y.agg(["sum", "size"])
    ll0 = -(gammaln(counts["size"] + 1) - gammaln(counts["sum"] + 1)
            - gammaln(counts["size"] - counts["sum"] + 1)).sum()
    assert_allclose(fe.tests["model"]["statistic"], 2 * (ref.llf - ll0), rtol=1e-9)
    assert_allclose(fe.metrics["pseudo_r_squared"], 1 - ref.llf / ll0, rtol=1e-10)
    assert fe.metrics["n_groups_dropped"] == unbalanced.id.nunique() - len(keep)


def test_xtpoisson_fe_wald_and_robust():
    frame = glmm_data(seed=6, groups=60, size=5, slope_sd=0.0).rename(columns={"g": "id"})
    fit = oe.xtpoisson(data=frame, y="c", x=["x", "w"], panel="id", model="fe", exposure="e")
    b = np.array([c.estimate for c in fit.coefficients])
    v = np.array(fit.covariance_matrix)
    assert fit.tests["model"]["distribution"] == "chi2" and fit.tests["model"]["df"] == 2
    assert_allclose(fit.tests["model"]["statistic"], b @ np.linalg.solve(v, b), rtol=1e-10)
    # Conditional Poisson = Poisson with panel dummies (exposure as offset).
    import statsmodels.api as sm

    dummies = pd.get_dummies(frame.id).to_numpy(float)
    glm = sm.GLM(frame.c, np.column_stack([frame[["x", "w"]].to_numpy(), dummies]),
                 family=sm.families.Poisson(), offset=np.log(frame.e)).fit(tol=1e-13)
    assert_allclose(b, glm.params[:2], rtol=1e-8)
