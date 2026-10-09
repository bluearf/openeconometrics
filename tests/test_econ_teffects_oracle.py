"""Independent verification oracles for the teffects family (verify-and-repair pass).

Written independently of the implementation and of the implementer's tests:

* ra / ipw / ipwra / aipw: the complete stacked moment system is written out
  in NumPy from Stata's Methods and formulas and solved JOINTLY from naive
  starting values with ``scipy.optimize.root`` (no statsmodels fits, no
  sequential plug-in); the sandwich uses a Richardson-extrapolated numerical
  Jacobian. The full covariance (effects, potential-outcome means and the
  auxiliary-equation blocks reported in ``extra``) is compared. The RA-linear
  ATE is also checked against its closed-form influence function.
* Invariances: frequency weights equal duplicated rows, pweights are scale
  free, row order is irrelevant, affine rescaling of covariates leaves the
  effects and their standard errors unchanged.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, special

import openecon as oe
from openecon.analysis_contracts import AnalysisError

# ---- data -------------------------------------------------------------------------


def design_data(seed=11, n=500, levels=2):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.gamma(2.0, 1.0, size=n)
    g = rng.integers(0, 3, size=n)                       # categorical covariate
    if levels == 2:
        prob = special.expit(-0.3 + 0.7 * a - 0.25 * (b - 2) + 0.3 * (g == 2))
        d = (rng.uniform(size=n) < prob).astype(int)
    else:
        eta = np.column_stack([np.zeros(n), 0.1 + 0.6 * a, -0.2 + 0.3 * (b - 2)])
        prob = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
        d = (rng.uniform(size=n)[:, None] > prob.cumsum(1)).sum(1)
    y = 2 + 1.2 * d + 0.8 * a + 0.3 * b + 0.5 * (g == 1) + 0.6 * d * a + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "d": d, "a": a, "b": b, "g": g})
    frame["ybin"] = (y > np.quantile(y, 0.45)).astype(float)
    frame["ycnt"] = rng.poisson(np.exp(0.2 + 0.4 * a + 0.3 * (d > 0)))
    frame["pw"] = rng.uniform(0.3, 3.0, size=n)
    frame["fw"] = rng.integers(1, 4, size=n)
    frame["cl"] = rng.integers(0, 45, size=n)
    return frame


@pytest.fixture(scope="module")
def frame():
    return design_data()


def coefs(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]),
            np.array(result.covariance_matrix))


# ---- the stacked moment system (written from Stata's Methods and formulas) --------


def _cdf(model, eta):
    if model == "logit":
        return special.expit(eta)
    if model == "probit":
        return special.ndtr(eta)
    if model == "poisson":
        return np.exp(eta)
    return eta


def _score_resid(model, y, eta):
    """d log f / d eta of the outcome / treatment (quasi-)likelihood."""
    if model == "probit":
        upper = np.exp(-0.5 * eta ** 2 - special.log_ndtr(eta)) / np.sqrt(2 * np.pi)
        lower = np.exp(-0.5 * eta ** 2 - special.log_ndtr(-eta)) / np.sqrt(2 * np.pi)
        return y * upper - (1 - y) * lower
    return y - _cdf(model, eta)


class Stacked:
    """psi_i(theta) of ra / ipw / ipwra / aipw; theta = (gamma, b_0..b_{L-1}, tau)."""

    def __init__(self, method, estimand, omodel, tmodel, y, d, X, Z, levels):
        self.method, self.estimand, self.omodel, self.tmodel = method, estimand, omodel, tmodel
        self.y, self.d, self.X, self.Z, self.L = y, d, X, Z, levels
        self.use_t = method in {"ipw", "ipwra", "aipw"}
        self.use_o = method in {"ra", "ipwra", "aipw"}
        self.kz = Z.shape[1] if self.use_t else 0
        self.ko = X.shape[1] if self.use_o else 0
        self.ng = (levels - 1) * self.kz
        self.size = self.ng + levels * self.ko + levels

    def split(self, theta):
        gamma = theta[:self.ng].reshape(self.L - 1, self.kz) if self.use_t else None
        betas = [theta[self.ng + j * self.ko:self.ng + (j + 1) * self.ko] for j in range(self.L)]
        return gamma, betas, theta[-self.L:]

    def probabilities(self, gamma):
        if self.L == 2:
            eta = self.Z @ gamma[0]
            p1 = _cdf(self.tmodel, eta)
            p0 = special.ndtr(-eta) if self.tmodel == "probit" else 1 - p1
            return np.column_stack([p0, p1])
        eta = np.column_stack([np.zeros(len(self.Z)), self.Z @ gamma.T])
        eta -= eta.max(1, keepdims=True)
        return np.exp(eta) / np.exp(eta).sum(1, keepdims=True)

    def omega(self, p):
        if self.estimand == "atet":
            return p[:, [1]] / p
        return 1 / p

    def treatment_block(self, gamma):
        if self.L == 2:
            return [self.Z * _score_resid(self.tmodel, (self.d == 1) * 1.0,
                                          self.Z @ gamma[0])[:, None]]
        p = self.probabilities(gamma)
        return [self.Z * ((self.d == s) - p[:, s])[:, None] for s in range(1, self.L)]

    def outcome_block(self, betas, omega):
        cols = []
        for level, b in enumerate(betas):
            weight = omega[:, level] if self.method == "ipwra" else 1.0
            r = _score_resid(self.omodel, self.y, self.X @ b)
            cols.append(self.X * ((self.d == level) * weight * r)[:, None])
        return cols

    def mean_block(self, betas, tau, p, omega):
        sel = (self.d == 1) * 1.0 if self.estimand == "atet" else np.ones(len(self.y))
        cols = []
        for level in range(self.L):
            ind = (self.d == level) * 1.0
            if self.method in {"ra", "ipwra"}:
                mu = _cdf(self.omodel, self.X @ betas[level])
                cols.append(sel * (mu - tau[level]))
            elif self.method == "ipw":
                cols.append(ind * omega[:, level] * (self.y - tau[level]))
            else:
                mu = _cdf(self.omodel, self.X @ betas[level])
                cols.append(ind * (self.y - mu) / p[:, level] + mu - tau[level])
        return [np.column_stack(cols)]

    def psi(self, theta):
        gamma, betas, tau = self.split(theta)
        blocks = []
        if self.use_t:
            p = self.probabilities(gamma)
            omega = self.omega(p)
            blocks += self.treatment_block(gamma)
        else:
            p = omega = np.ones((len(self.y), self.L))
        if self.use_o:
            blocks += self.outcome_block(betas, omega)
        blocks += self.mean_block(betas, tau, p, omega)
        return np.column_stack(blocks)


def jacobian(fun, theta):
    """Richardson-extrapolated central differences of a vector function."""
    out = []
    for j in range(len(theta)):
        h = 1e-4 * max(1.0, abs(theta[j]))
        def diff(step):
            up, down = theta.copy(), theta.copy()
            up[j] += step
            down[j] -= step
            return (fun(up) - fun(down)) / (2 * step)
        out.append((4 * diff(h / 2) - diff(h)) / 3)
    return np.column_stack(out)


def solve_stacked(system, w):
    """Joint root of sum_i w_i psi_i(theta) = 0 from naive starting values."""
    y, d, L = system.y, system.d, system.L
    start = []
    if system.use_t:
        start += [0.0] * system.ng
    if system.use_o:
        for level in range(L):
            b = np.zeros(system.ko)
            m = (w * y * (d == level)).sum() / (w * (d == level)).sum()
            b[0] = {"linear": m, "logit": special.logit(m), "probit": special.ndtri(m),
                    "poisson": np.log(m)}[system.omodel]
            start += list(b)
    start += [float((w * y).sum() / w.sum())] * L
    theta = np.array(start, dtype=float)

    def total(th):
        return (w[:, None] * system.psi(th)).sum(0) / w.sum()

    # damped Newton with a numerical Jacobian (the system is smooth and well posed)
    for _ in range(200):
        value = total(theta)
        if np.abs(value).max() < 1e-13:
            break
        step = np.linalg.solve(jacobian(total, theta), value)
        scale = 1.0
        while scale > 1e-4:
            trial = theta - scale * step
            if np.all(np.isfinite(total(trial))) and \
                    np.abs(total(trial)).max() < np.abs(value).max() * (1 - 1e-4 * scale) + 1e-15:
                break
            scale /= 2
        theta = theta - scale * step
    root = optimize.root(total, theta, method="hybr", options={"xtol": 1e-14})
    theta = root.x if np.abs(total(root.x)).max() <= np.abs(total(theta)).max() else theta
    assert np.abs(total(theta)).max() < 1e-10
    return theta


def oracle_stacked(df, method, estimand="ate", omodel="linear", tmodel="logit", *,
                   y="y", x=("a", "b"), tx=None, wcol=None, wtype=None, cluster=None,
                   control=None, categorical=()):
    tx = x if tx is None else tx
    labels = sorted(df["d"].unique())
    if control is not None:
        labels.remove(control)
        labels.insert(0, control)
    d = np.array([labels.index(v) for v in df["d"]])
    L = len(labels)

    def matrix(names):
        cols = [np.ones(len(df))]
        for name in names:
            if name in categorical:
                levels = sorted(df[name].unique())
                cols += [(df[name] == v).to_numpy(float) for v in levels[1:]]
            else:
                cols.append(df[name].to_numpy(float))
        return np.column_stack(cols)

    X, Z = matrix(x), matrix(tx)
    w = np.ones(len(df)) if wcol is None else df[wcol].to_numpy(float)
    system = Stacked(method, estimand, omodel, tmodel, df[y].to_numpy(float), d, X, Z, L)
    theta = solve_stacked(system, w)
    rows = system.psi(theta)
    A = jacobian(lambda th: (w[:, None] * system.psi(th)).sum(0), theta)
    if cluster is not None:
        codes = pd.factorize(df[cluster])[0]
        totals = np.zeros((codes.max() + 1, len(theta)))
        np.add.at(totals, codes, w[:, None] * rows)
        B = totals.T @ totals
    elif wtype == "fweight":
        B = (w[:, None] * rows).T @ rows
    else:
        B = (w[:, None] * rows).T @ (w[:, None] * rows)
    Ainv = np.linalg.inv(A)
    V = Ainv @ B @ Ainv.T
    if estimand == "pomeans":
        T = np.eye(L)
    else:
        T = np.zeros((L, L))
        for level in range(1, L):
            T[level - 1, level], T[level - 1, 0] = 1, -1
        T[-1, 0] = 1
    k = system.size
    tau = theta[-L:]
    return {"params": T @ tau, "cov": T @ V[k - L:, k - L:] @ T.T, "theta": theta, "V": V,
            "system": system, "labels": labels, "p": system.probabilities(system.split(theta)[0])
            if system.use_t else None}


def fit_te(df, method, estimand="ate", omodel=None, tmodel=None, **kw):
    options = {}
    if omodel is not None and method in {"ra", "ipwra", "aipw"}:
        options["omodel"] = omodel
    if tmodel is not None and method in {"ipw", "ipwra", "aipw"}:
        options["tmodel"] = tmodel
    return oe.teffects(data=df, y=kw.pop("y", "y"), treatment="d", x=kw.pop("x", ["a", "b"]),
                       method=method, estimand=estimand, **options, **kw)


OUTCOME_COLUMN = {"linear": "y", "logit": "ybin", "probit": "ybin", "poisson": "ycnt"}
GRID = []
for _method in ("ra", "ipw", "ipwra", "aipw"):
    for _estimand in ("ate", "atet", "pomeans"):
        if _method == "aipw" and _estimand == "atet":
            continue
        for _omodel in (("linear",) if _method == "ipw" else ("linear", "logit", "probit",
                                                              "poisson")):
            for _tmodel in (("logit",) if _method == "ra" else ("logit", "probit")):
                if _method in {"ipwra", "aipw"} and _omodel in {"logit", "poisson"} \
                        and _tmodel == "probit" and _estimand != "ate":
                    continue                  # keep the grid fast; ate covers the pair
                GRID.append((_method, _estimand, _omodel, _tmodel))


@pytest.mark.parametrize("method,estimand,omodel,tmodel", GRID)
def test_stacked_grid_matches_joint_root_oracle(frame, method, estimand, omodel, tmodel):
    y = OUTCOME_COLUMN[omodel]
    result = fit_te(frame, method, estimand, omodel, tmodel, y=y)
    ref = oracle_stacked(frame, method, estimand, omodel, tmodel, y=y)
    est, se, cov = coefs(result)
    assert_allclose(est, ref["params"], rtol=1e-8, atol=1e-10)
    assert_allclose(cov, ref["cov"], rtol=1e-6, atol=1e-12)
    # z inference, Stata's reporting of teffects
    c = result.coefficients[0]
    assert result.inference["use_t"] is False
    assert_allclose(c.p_value, 2 * special.ndtr(-abs(c.estimate / c.std_error)), rtol=1e-10)
    assert_allclose(c.ci_high - c.ci_low, 2 * special.ndtri(0.975) * c.std_error, rtol=1e-10)
    # auxiliary equations: uncentred coefficients with their stacked-sandwich errors
    system, V, theta = ref["system"], ref["V"], ref["theta"]
    aux = result.extra["auxiliary_equations"]
    if system.use_t:
        rows = aux["TME1"]
        assert_allclose([r["estimate"] for r in rows], theta[:system.kz], rtol=1e-7, atol=1e-9)
        assert_allclose([r["std_error"] for r in rows],
                        np.sqrt(np.diag(V)[:system.kz]), rtol=1e-6)
    if system.use_o:
        for level in range(2):
            rows = aux[f"OME{level}"]
            block = slice(system.ng + level * system.ko, system.ng + (level + 1) * system.ko)
            assert_allclose([r["estimate"] for r in rows], theta[block], rtol=1e-7, atol=1e-9)
            assert_allclose([r["std_error"] for r in rows], np.sqrt(np.diag(V)[block]),
                            rtol=1e-6)
    if system.use_t:
        p1 = ref["p"][:, 1]
        assert_allclose(result.metrics["propensity_min"], p1.min(), rtol=1e-8)
        assert_allclose(result.metrics["propensity_max"], p1.max(), rtol=1e-8)


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_pweights_and_cluster_match_oracle(frame, method):
    result = fit_te(frame, method, weights="pw", weight_type="pweight")
    ref = oracle_stacked(frame, method, wcol="pw", wtype="pweight")
    est, _, cov = coefs(result)
    assert_allclose(est, ref["params"], rtol=1e-8)
    assert_allclose(cov, ref["cov"], rtol=1e-6)
    result = fit_te(frame, method, "atet" if method != "aipw" else "ate",
                    covariance="cluster", cluster="cl", weights="pw", weight_type="pweight")
    ref = oracle_stacked(frame, method, "atet" if method != "aipw" else "ate", wcol="pw",
                         wtype="pweight", cluster="cl")
    est, _, cov = coefs(result)
    assert_allclose(est, ref["params"], rtol=1e-8)
    assert_allclose(cov, ref["cov"], rtol=1e-6)
    assert result.inference["cluster_count"] == frame["cl"].nunique()


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_frequency_weights_equal_duplicated_rows(frame, method):
    weighted = fit_te(frame, method, weights="fw", weight_type="fweight")
    expanded = frame.loc[frame.index.repeat(frame["fw"])].reset_index(drop=True)
    plain = fit_te(expanded, method)
    a, b = coefs(weighted), coefs(plain)
    assert_allclose(a[0], b[0], rtol=1e-9)
    assert_allclose(a[2], b[2], rtol=1e-7)
    assert weighted.nobs == plain.nobs == int(frame["fw"].sum())
    ref = oracle_stacked(frame, method, wcol="fw", wtype="fweight")
    assert_allclose(a[2], ref["cov"], rtol=1e-6)


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_multivalued_treatment_with_control_level_and_categorical(method):
    df = design_data(seed=5, n=700, levels=3)
    for estimand in ("ate", "pomeans"):
        result = fit_te(df, method, estimand, x=["a", "b", "g"], categorical=["g"], control=1)
        ref = oracle_stacked(df, method, estimand, x=("a", "b", "g"), categorical=("g",),
                             control=1)
        est, _, cov = coefs(result)
        assert_allclose(est, ref["params"], rtol=1e-8, atol=1e-10)
        assert_allclose(cov, ref["cov"], rtol=1e-6, atol=1e-12)
    terms = [c.term for c in result.coefficients]
    assert terms == ["POmeans:1.d", "POmeans:0.d", "POmeans:2.d"]
    result = fit_te(df, method, "ate", control=1)
    assert [c.term for c in result.coefficients] == ["ATE:r0vs1.d", "ATE:r2vs1.d", "POmean:1.d"]


def test_ra_linear_ate_closed_form_influence_function(frame):
    """ATE_RA = xbar'(b1 - b0); its influence function written by hand."""
    X = np.column_stack([np.ones(len(frame)), frame[["a", "b"]].to_numpy()])
    y, d = frame["y"].to_numpy(), frame["d"].to_numpy()
    n = len(y)
    xbar = X.mean(0)
    phi = np.zeros(n)
    betas = []
    for level, sign in ((0, -1), (1, 1)):
        m = d == level
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        betas.append(b)
        e = y - X @ b
        inv = np.linalg.inv(X[m].T @ X[m])
        phi += sign * m * (X @ inv @ xbar) * e * n
    ate = xbar @ (betas[1] - betas[0])
    phi += X @ (betas[1] - betas[0]) - ate
    result = fit_te(frame, "ra")
    assert_allclose(result.coefficients[0].estimate, ate, rtol=1e-10)
    assert_allclose(result.coefficients[0].std_error, np.sqrt((phi ** 2).sum()) / n, rtol=1e-9)
    # POmean:0 = xbar'b0 with influence X b0 - POM0 + n xbar'(X0'X0)^-1 x e
    m = d == 0
    e = y - X @ betas[0]
    phi0 = X @ betas[0] - xbar @ betas[0] + m * (X @ np.linalg.inv(X[m].T @ X[m]) @ xbar) * e * n
    assert_allclose(result.coefficients[1].std_error, np.sqrt((phi0 ** 2).sum()) / n, rtol=1e-9)


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_invariance_to_scale_offset_permutation_and_pweight_scale(frame, method):
    base = coefs(fit_te(frame, method, weights="pw", weight_type="pweight"))
    moved = frame.copy()
    moved["a"] = 1e4 * moved["a"] + 1e6
    moved["b"] = 1e-4 * moved["b"] - 3.0
    moved["pw"] = 1e3 * moved["pw"]
    moved = moved.sample(frac=1.0, random_state=4).reset_index(drop=True)
    other = coefs(fit_te(moved, method, weights="pw", weight_type="pweight"))
    assert_allclose(other[0], base[0], rtol=1e-8)
    assert_allclose(other[2], base[2], rtol=1e-6)


def test_missing_drop_equals_complete_case_fit(frame):
    holes = frame.copy()
    holes.loc[[3, 17, 99], "a"] = np.nan
    holes.loc[[5], "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        fit_te(holes, "aipw")
    assert error.value.code == "missing_values"
    dropped = fit_te(holes, "aipw", missing="drop")
    complete = fit_te(holes.dropna(subset=["a", "y"]).reset_index(drop=True), "aipw")
    assert_allclose(coefs(dropped)[2], coefs(complete)[2], rtol=1e-10)
    assert dropped.nobs == len(frame) - 4


# ---- nnmatch / psmatch: brute force from the matched sets --------------------------


def matching_data(seed=21, n=300):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n) * 2 + 1
    d = (rng.uniform(size=n) < special.expit(0.5 * a - 0.3 * (b - 1))).astype(int)
    y = 1 + 1.5 * d + a + 0.4 * b + 0.7 * d * a + rng.normal(size=n) * (1 + 0.4 * np.abs(a))
    df = pd.DataFrame({"y": y, "d": d, "a": a, "b": b})
    df["i1"] = rng.integers(0, 5, size=n).astype(float)      # integer grid: exact ties
    df["i2"] = rng.integers(0, 3, size=n).astype(float)
    return df


@pytest.fixture(scope="module")
def mdata():
    return matching_data()


def matched_sets(dist, m):
    """Boolean [nq, np]: every pool unit within the m-th smallest distance (ties kept)."""
    kth = np.sort(dist, axis=1)[:, m - 1]
    return dist <= kth[:, None]


def brute_matching(df, dist_fn, estimand, m=1, nn=2, biasadj=None, both=False):
    """Matching estimator and Abadie-Imbens variance written from the matched sets.

    The variance is assembled as V^E + V^tau(X) (Abadie and Imbens 2006, section 4):
    V^E sums (weight of unit i)^2 sigma2_i and V^tau(X) is the mean squared imputed effect
    minus its expected conditional variance, sigma2_i + sum_{j in J(i)} sigma2_j / #J(i)^2,
    so the K' bookkeeping of the implementation is never used.
    """
    y, d = df["y"].to_numpy(float), df["d"].to_numpy()
    n = len(y)
    groups = [np.flatnonzero(d == 0), np.flatnonzero(d == 1)]
    directions = [1] if estimand == "atet" and not both else [0, 1]
    sets = {}
    K = np.zeros(n)
    for level in directions:
        q, p = groups[level], groups[1 - level]
        mask = matched_sets(dist_fn(q, p), m)
        sets[level] = mask
        K[p] += (mask / mask.sum(1, keepdims=True)).sum(0)
    beta = {}
    if biasadj is not None:
        XB = np.column_stack([np.ones(n), df[biasadj].to_numpy(float)])
        for level in directions:
            p = groups[1 - level]
            sw = np.sqrt(K[p])
            beta[1 - level] = np.linalg.lstsq(XB[p] * sw[:, None], y[p] * sw, rcond=None)[0]
    imputed = np.column_stack([y, y])
    for level in directions:
        q, p = groups[level], groups[1 - level]
        for row, i in enumerate(q):
            js = p[sets[level][row]]
            values = y[js]
            if biasadj is not None:
                values = values + (XB[i] - XB[js]) @ beta[1 - level]
            imputed[i, 1 - level] = values.mean()
    sigma2 = np.zeros(n)
    for level in (0, 1):
        g = groups[level]
        dist = dist_fn(g, g)
        np.fill_diagonal(dist, np.inf)
        mask = matched_sets(dist, nn)
        size = mask.sum(1)
        sigma2[g] = size / (size + 1) * (y[g] - (mask * y[g]).sum(1) / size) ** 2
    effect = imputed[:, 1] - imputed[:, 0]
    expected = {}
    for level in directions:
        q, p = groups[level], groups[1 - level]
        mask = sets[level]
        size = mask.sum(1)
        expected[level] = sigma2[q] + (mask * sigma2[p]).sum(1) / size ** 2
    if estimand == "ate":
        tau = effect.mean()
        weight = 1 + K
        v_e = np.sum(weight ** 2 * sigma2) / n ** 2
        v_x = (np.sum((effect - tau) ** 2) - np.sum(expected[0]) - np.sum(expected[1])) / n ** 2
    else:
        t, c = groups[1], groups[0]
        tau = effect[t].mean()
        v_e = (np.sum(sigma2[t]) + np.sum(K[c] ** 2 * sigma2[c])) / len(t) ** 2
        v_x = (np.sum((effect[t] - tau) ** 2) - np.sum(expected[1])) / len(t) ** 2
    return tau, v_e + v_x, effect, sigma2


def metric_distance(df, columns, metric):
    X = df[columns].to_numpy(float)
    S = np.cov(X, rowvar=False).reshape(len(columns), len(columns))
    weight = {"mahalanobis": np.linalg.inv(S), "ivariance": np.diag(1 / np.diag(S)),
              "euclidean": np.eye(len(columns))}[metric]

    def dist(q, p):
        diff = X[q][:, None, :] - X[p][None, :, :]
        return np.sqrt(np.einsum("qpk,kl,qpl->qp", diff, weight, diff).clip(0))
    return dist


@pytest.mark.parametrize("metric", ["mahalanobis", "ivariance", "euclidean"])
@pytest.mark.parametrize("estimand", ["ate", "atet"])
def test_nnmatch_matches_brute_force(mdata, metric, estimand):
    for m, nn in ((1, 2), (3, 4)):
        result = oe.teffects(data=mdata, y="y", treatment="d", x=["a", "b"], method="nnmatch",
                             estimand=estimand, metric=metric, neighbors=m, vce_neighbors=nn)
        tau, var, _, _ = brute_matching(mdata, metric_distance(mdata, ["a", "b"], metric),
                                        estimand, m=m, nn=nn)
        assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
        assert_allclose(result.coefficients[0].std_error ** 2, var, rtol=1e-9)
        assert result.coefficients[0].term == f"{estimand.upper()}:r1vs0.d"


@pytest.mark.parametrize("estimand", ["ate", "atet"])
def test_nnmatch_ties_on_integer_grid_and_bias_adjustment(mdata, estimand):
    dist = metric_distance(mdata, ["i1", "i2"], "euclidean")
    for m in (1, 2):
        result = oe.teffects(data=mdata, y="y", treatment="d", x=["i1", "i2"],
                             method="nnmatch", estimand=estimand, metric="euclidean",
                             neighbors=m, biasadj=["a", "i1"])
        tau, var, _, _ = brute_matching(mdata, dist, estimand, m=m, biasadj=["a", "i1"])
        assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
        assert_allclose(result.coefficients[0].std_error ** 2, var, rtol=1e-9)
        assert result.extra["matches"]["units_with_ties"] > 0


def logit_fit(X, d):
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = special.expit(X @ beta)
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X, X.T @ (d - p))
        beta += step
        if np.abs(step).max() < 1e-14:
            break
    return beta


@pytest.mark.parametrize("estimand", ["ate", "atet"])
def test_psmatch_matches_abadie_imbens_2016_from_the_paper(mdata, estimand):
    """Abadie and Imbens (2016, Econometrica) Theorem 1 / 2 with their section-4 estimators."""
    m, L = 1, 3
    result = oe.teffects(data=mdata, y="y", treatment="d", tx=["a", "b"], method="psmatch",
                         estimand=estimand, neighbors=m, vce_neighbors=L)
    X = np.column_stack([np.ones(len(mdata)), mdata[["a", "b"]].to_numpy()])
    d = mdata["d"].to_numpy()
    y = mdata["y"].to_numpy()
    theta = logit_fit(X, d)
    ps = special.expit(X @ theta)
    f = ps * (1 - ps)

    def dist(q, p):
        return np.abs(ps[q][:, None] - ps[p][None, :])
    tau, var, effect, _ = brute_matching(mdata, dist, estimand, m=m, nn=L, both=True)
    info_inv = np.linalg.inv((X * (ps * (1 - ps))[:, None]).T @ X / len(d))
    # conditional covariances cov(X, Y | p, W = w), matching on p within group w (L units,
    # the unit itself excluded when it belongs to w), divisor L - 1
    n = len(d)
    cov = np.zeros((2, n, X.shape[1]))
    for w in (0, 1):
        pool = np.flatnonzero(d == w)
        D = np.abs(ps[:, None] - ps[pool][None, :])
        D[pool, np.arange(len(pool))] = np.inf
        mask = matched_sets(D, L)
        for i in range(n):
            js = pool[mask[i]]
            cov[w, i] = np.cov(X[js].T, y[js])[-1, :-1]
    if estimand == "ate":
        c = np.mean(f[:, None] * (cov[1] / ps[:, None] + cov[0] / (1 - ps)[:, None]), axis=0)
        adjusted = var - c @ info_inv @ c / n
    else:
        share = d.mean()
        c = np.mean(f[:, None] * (cov[1] + (ps / (1 - ps))[:, None] * cov[0]), axis=0) / share
        dtau = np.mean(X * (f * (effect - tau))[:, None], axis=0) / share
        adjusted = var - c @ info_inv @ c / n + dtau @ info_inv @ dtau / n
    assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
    assert_allclose(result.coefficients[0].std_error ** 2, adjusted, rtol=1e-8)
    assert_allclose(result.extra["variance"]["abadie_imbens"], var, rtol=1e-9)
    tme = result.extra["auxiliary_equations"]["TME1"]
    assert_allclose([r["estimate"] for r in tme], theta, rtol=1e-8)


# ---- didregress / eventstudy: explicit dummy-variable least squares in NumPy -------


def did_panel(seed=31, groups=24, periods=7, staggered=False, repeated=1, unequal=False):
    rng = np.random.default_rng(seed)
    times = np.array([1990, 1992, 1995, 1996, 1997, 2001, 2002, 2005, 2006])[:periods] \
        if unequal else np.arange(2000, 2000 + periods)
    g = np.repeat(np.arange(groups), periods * repeated)
    t = np.tile(np.repeat(times, repeated), groups)
    treated = g < groups // 3
    first = np.where(treated, times[3], np.nan)
    if staggered:
        first = np.where(treated, np.where(g % 2 == 0, times[2], times[4]), np.nan)
    d = (~np.isnan(first) & (t >= np.nan_to_num(first, nan=1e9))).astype(int)
    x = rng.normal(size=len(g))
    y = rng.normal(size=groups)[g] + 0.1 * (t - times[0]) + 0.9 * d + 0.4 * x \
        + rng.normal(size=len(g)) * (1 + 0.5 * treated[g])
    df = pd.DataFrame({"y": y, "d": d, "g": g, "t": t, "x": x, "first": first})
    df["region"] = df["g"] // 3
    df["aw"] = rng.uniform(0.5, 2.0, size=len(g))
    df["fw"] = rng.integers(1, 4, size=len(g))
    return df


def dummies(codes):
    levels = np.unique(codes)
    return (codes[:, None] == levels[None, 1:]).astype(float)


def ols_dummies(df, regressors, weights=None):
    """Coefficients of y on regressors + group and time dummies + constant, and pieces."""
    R = np.column_stack([df[c].to_numpy(float) if isinstance(c, str) else c for c in regressors])
    X = np.column_stack([R, np.ones(len(df)), dummies(df["g"].to_numpy()),
                         dummies(df["t"].to_numpy())])
    y = df["y"].to_numpy(float)
    w = np.ones(len(df)) if weights is None else weights
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    resid = y - X @ beta
    bread = np.linalg.inv((X * w[:, None]).T @ X)
    return beta, resid, X, bread, w


def cluster_cov(X, resid, bread, w, codes, k_dof, n=None):
    n = len(resid) if n is None else n
    scores = X * (w * resid)[:, None]
    _, inverse = np.unique(codes, return_inverse=True)
    totals = np.zeros((inverse.max() + 1, X.shape[1]))
    np.add.at(totals, inverse, scores)
    G = totals.shape[0]
    factor = G / (G - 1) * (n - 1) / (n - k_dof)
    return factor * bread @ (totals.T @ totals) @ bread, G


def by_term(result):
    return {c.term: c for c in result.coefficients}


@pytest.mark.parametrize("covariance", ["robust", "cluster", "HC1", "nonrobust"])
def test_didregress_matches_numpy_dummy_regression(covariance):
    df = did_panel()
    kw = {"cluster": "region"} if covariance == "cluster" else {}
    result = oe.didregress(data=df, y="y", treatment="d", group="g", time="t", x=["x"],
                           covariance=covariance, **kw)
    beta, resid, X, bread, w = ols_dummies(df, ["d", "x"])
    n, T, G = len(df), df["t"].nunique(), df["g"].nunique()
    if covariance in {"robust", "cluster"}:
        # groups nest in their own clusters and in regions: they cost no degrees of freedom
        V, clusters = cluster_cov(X, resid, bread, w, df["g" if covariance == "robust"
                                                         else "region"].to_numpy(), 2 + T)
        df_inf = clusters - 1
    else:
        k = X.shape[1]
        if covariance == "HC1":
            V = n / (n - k) * bread @ ((X * resid[:, None]).T @ (X * resid[:, None])) @ bread
        else:
            V = resid @ resid / (n - k) * bread
        df_inf = n - (G + T - 1 + 2)
    terms = by_term(result)
    est = np.array([terms["ATET:r1vs0.d"].estimate, terms["x"].estimate])
    assert_allclose(est, beta[:2], rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), V[:2, :2], rtol=1e-8)
    assert result.inference["df_inference"] == df_inf
    c = terms["ATET:r1vs0.d"]
    from scipy import stats
    assert_allclose(c.p_value, 2 * stats.t.sf(abs(c.statistic), df_inf), rtol=1e-9)
    assert_allclose(c.ci_high - c.estimate, stats.t.ppf(0.975, df_inf) * c.std_error,
                    rtol=1e-9)
    assert_allclose(result.metrics["rmse"],
                    np.sqrt(resid @ resid / result.metrics["df_resid"]), rtol=1e-9)


def test_didregress_weights_and_frequency_duplication():
    df = did_panel(seed=4, repeated=2)
    for wtype in ("aweight", "pweight"):
        result = oe.didregress(data=df, y="y", treatment="d", group="g", time="t", x=["x"],
                               weights="aw", weight_type=wtype)
        w = df["aw"].to_numpy(float)
        beta, resid, X, bread, w = ols_dummies(df, ["d", "x"], w)
        V, _ = cluster_cov(X, resid, bread, w, df["g"].to_numpy(), 2 + df["t"].nunique())
        assert_allclose(np.array(result.covariance_matrix), V[:2, :2], rtol=1e-8)
        assert_allclose(result.coefficients[0].estimate, beta[0], rtol=1e-9)
    # aweights with the classical covariance: weights normalized to sum N (Stata)
    result = oe.didregress(data=df, y="y", treatment="d", group="g", time="t",
                           weights="aw", weight_type="aweight", covariance="nonrobust")
    w = df["aw"].to_numpy(float)
    w = w * len(w) / w.sum()
    beta, resid, X, bread, w = ols_dummies(df, ["d"], w)
    V = (w * resid ** 2).sum() / (len(df) - X.shape[1]) * bread
    assert_allclose(result.coefficients[0].std_error ** 2, V[0, 0], rtol=1e-8)
    weighted = oe.didregress(data=df, y="y", treatment="d", group="g", time="t", x=["x"],
                             weights="fw", weight_type="fweight")
    expanded = df.loc[df.index.repeat(df["fw"])].reset_index(drop=True)
    plain = oe.didregress(data=expanded, y="y", treatment="d", group="g", time="t", x=["x"])
    assert_allclose(np.array(weighted.covariance_matrix), np.array(plain.covariance_matrix),
                    rtol=1e-8)
    assert weighted.nobs == plain.nobs


@pytest.mark.parametrize("unequal", [False, True])
def test_parallel_trends_and_granger_from_augmented_regressions(unequal):
    df = did_panel(seed=8, unequal=unequal)
    result = oe.didregress(data=df, y="y", treatment="d", group="g", time="t", x=["x"])
    times = np.sort(df["t"].unique())
    start = times[3]
    treated = df["first"].notna().to_numpy()
    t = df["t"].to_numpy(float)
    pre = (t < start)
    T = len(times)
    # estat ptrends: a linear trend in the time variable for treated groups before treatment
    trend = treated * t * pre
    beta, resid, X, bread, w = ols_dummies(df, ["d", "x", trend])
    V, G = cluster_cov(X, resid, bread, w, df["g"].to_numpy(), 3 + T)
    F = beta[2] ** 2 / V[2, 2]
    test = result.tests["parallel_trends"]
    assert_allclose(test["statistic"], F, rtol=1e-7)
    assert (test["df"], test["df2"]) == (1, G - 1)
    # estat granger: treated-group indicators of the pre-treatment periods but the first
    leads = [treated * (t == times[j]) for j in range(1, 3)]
    beta, resid, X, bread, w = ols_dummies(df, ["d", "x", *leads])
    V, G = cluster_cov(X, resid, bread, w, df["g"].to_numpy(), 4 + T)
    block = slice(2, 4)
    F = beta[block] @ np.linalg.solve(V[block, block], beta[block]) / 2
    test = result.tests["granger"]
    assert_allclose(test["statistic"], F, rtol=1e-7)
    assert (test["df"], test["df2"]) == (2, G - 1)
    assert result.extra["first_treated_periods"] == [int(start)]


@pytest.mark.parametrize("staggered", [False, True])
def test_eventstudy_matches_numpy_relative_time_dummies(staggered):
    df = did_panel(seed=12, staggered=staggered, periods=8)
    for leads, lags in ((None, None), (2, 2)):
        result = oe.eventstudy(data=df, y="y", group="g", time="t", treatment_time="first",
                               x=["x"], leads=leads, lags=lags)
        rel = df["t"].to_numpy(float) - df["first"].to_numpy(float)
        treated = df["first"].notna().to_numpy()
        observed = rel[treated]
        lo = -leads if leads is not None else int(observed.min())
        hi = lags if lags is not None else int(observed.max())
        binned = np.clip(np.nan_to_num(rel, nan=0.0), lo, hi)
        events = [e for e in range(lo, hi + 1) if e != -1]
        cols = [(treated & (binned == e)).astype(float) for e in events]
        beta, resid, X, bread, w = ols_dummies(df, [*cols, "x"])
        # K = slopes (event indicators and x) + time effects; group effects nest in clusters
        k = len(events) + 1
        V, G = cluster_cov(X, resid, bread, w, df["g"].to_numpy(), k + df["t"].nunique())
        terms = by_term(result)
        names = [f"lead{-e}" if e < 0 else f"lag{e}" for e in events]
        assert_allclose([terms[name].estimate for name in names], beta[:len(events)],
                        rtol=1e-8, atol=1e-10)
        assert_allclose([terms[name].std_error ** 2 for name in names],
                        np.diag(V)[:len(events)], rtol=1e-7)
        pre = [i for i, e in enumerate(events) if e < 0]
        F = beta[pre] @ np.linalg.solve(V[np.ix_(pre, pre)], beta[pre]) / len(pre)
        assert_allclose(result.tests["pretrends"]["statistic"], F, rtol=1e-7)
        post = [i for i, e in enumerate(events) if e >= 0]
        avg = beta[post].mean()
        se = np.sqrt(np.ones(len(post)) @ V[np.ix_(post, post)] @ np.ones(len(post))) / len(post)
        assert_allclose(result.extra["average_post_effect"]["estimate"], avg, rtol=1e-8)
        assert_allclose(result.extra["average_post_effect"]["std_error"], se, rtol=1e-7)
        table = {row["relative_time"]: row for row in result.extra["event_table"]}
        assert table[-1]["reference"] and table[-1]["estimate"] == 0.0


# ---- csdid: R did / DRDID influence functions re-written in NumPy ------------------


def cs_panel(seed=41, units=160, periods=6):
    rng = np.random.default_rng(seed)
    times = np.arange(2001, 2001 + periods)
    cohort = rng.choice([2003.0, 2004.0, 2006.0, np.nan], size=units, p=[0.2, 0.2, 0.2, 0.4])
    cohort[:3] = 2001.0                    # treated in the first period: dropped by did
    cohort[3:5] = 2010.0                   # treated after the last period: never treated
    x1 = rng.normal(size=units)
    alpha = rng.normal(size=units) + 0.5 * x1
    rows = []
    for i in range(units):
        x2 = rng.normal(size=periods)
        for j, t in enumerate(times):
            treated = not np.isnan(cohort[i]) and t >= cohort[i]
            effect = (1.0 + 0.3 * (t - cohort[i])) if treated else 0.0
            y = alpha[i] + 0.2 * j + 0.4 * x1[i] * j / periods + effect + rng.normal()
            rows.append((i, t, y, cohort[i], x1[i], x2[j]))
    return pd.DataFrame(rows, columns=["id", "t", "y", "first", "x1", "x2"])


def drdid_panel_if(method, dy, X, D):
    """att and influence function (n_gt scale, mean form) of DRDID's panel estimators."""
    n = len(dy)
    if method in {"ipw", "dr"}:
        gamma = logit_fit(X, D)
        ps = special.expit(X @ gamma)
        hessian_inv = np.linalg.inv((X * (ps * (1 - ps))[:, None]).T @ X / n)
        lin_ps = (X * (D - ps)[:, None]) @ hessian_inv
    if method in {"reg", "dr"}:
        c = D == 0
        coef = np.linalg.lstsq(X[c], dy[c], rcond=None)[0]
        out = X @ coef
        wols = (1 - D)
        xpx_inv = np.linalg.inv((X * wols[:, None]).T @ X / n)
        lin_ols = (X * (wols * (dy - out))[:, None]) @ xpx_inv
    w_t = D
    if method == "reg":
        w_c = D
        att_t, att_c = w_t * dy, w_c * out
        eta_t, eta_c = att_t.mean() / w_t.mean(), att_c.mean() / w_c.mean()
        inf_t = (att_t - w_t * eta_t) / w_t.mean()
        inf_c = (att_c - w_c * eta_c + lin_ols @ np.mean(w_c[:, None] * X, 0)) / w_c.mean()
        return eta_t - eta_c, inf_t - inf_c
    w_c = ps * (1 - D) / (1 - ps)
    if method == "ipw":
        att_t, att_c = w_t * dy, w_c * dy
        eta_t, eta_c = att_t.mean() / w_t.mean(), att_c.mean() / w_c.mean()
        inf_t = (att_t - w_t * eta_t) / w_t.mean()
        m2 = np.mean((w_c * (dy - eta_c))[:, None] * X, 0)
        inf_c = (att_c - w_c * eta_c + lin_ps @ m2) / w_c.mean()
        return eta_t - eta_c, inf_t - inf_c
    att_t, att_c = w_t * (dy - out), w_c * (dy - out)
    eta_t, eta_c = att_t.mean() / w_t.mean(), att_c.mean() / w_c.mean()
    inf_t = (att_t - w_t * eta_t - lin_ols @ np.mean(w_t[:, None] * X, 0)) / w_t.mean()
    m2 = np.mean((w_c * (dy - out - eta_c))[:, None] * X, 0)
    m3 = np.mean(w_c[:, None] * X, 0)
    inf_c = (att_c - w_c * eta_c + lin_ps @ m2 - lin_ols @ m3) / w_c.mean()
    return eta_t - eta_c, inf_t - inf_c


def did_att_gt(df, method, control, base, covariates):
    """R did::att_gt (panel, analytic) with first-period-treated units dropped."""
    first = df.groupby("id")["first"].first()
    times = np.sort(df["t"].unique())
    G = first.to_numpy().copy()
    G[G > times[-1]] = np.nan                      # treated after the sample: never treated
    keep = ~(G <= times[0])                        # treated in the first period: dropped
    ids = first.index.to_numpy()[keep]
    G = G[keep]
    wide = df.pivot(index="id", columns="t", values="y").loc[ids].to_numpy()
    n = len(ids)
    cohorts = np.unique(G[~np.isnan(G)])
    Gz = np.nan_to_num(G, nan=0.0)
    out = []
    for g in cohorts:
        gi = int(np.searchsorted(times, g))
        index = range(len(times)) if base == "universal" else range(1, len(times))
        for ti in index:
            if base == "universal" and ti == gi - 1:
                continue
            b = gi - 1 if (ti >= gi or base == "universal") else ti - 1
            later = times[max(ti, b)]
            if control == "never":
                ctrl = Gz == 0
            else:
                ctrl = (Gz == 0) | (Gz > later)
            ctrl &= Gz != g
            sub = (Gz == g) | ctrl
            dy = wide[sub, ti] - wide[sub, b]
            D = (Gz[sub] == g).astype(float)
            if covariates:
                cov = df[df["t"] == times[b]].set_index("id").loc[ids[sub], covariates]
                X = np.column_stack([np.ones(sub.sum()), cov.to_numpy(float)])
            else:
                X = np.ones((sub.sum(), 1))
            att, inf = drdid_panel_if(method, dy, X, D)
            phi = np.zeros(n)
            phi[sub] = inf / sub.sum()
            out.append((g, times[ti], att, phi, ti >= gi))
    return out, Gz, n


def aggte(results, Gz, n):
    """did::aggte simple / dynamic / group / calendar with the weight influence (wif)."""
    att = np.array([r[2] for r in results])
    phi = np.column_stack([r[3] for r in results])
    group = np.array([r[0] for r in results])
    period = np.array([r[1] for r in results])
    post = np.array([r[4] for r in results])
    cohorts = np.unique(group)
    pg_of = {g: np.mean(Gz == g) for g in cohorts}
    pgg = np.array([pg_of[g] for g in group])

    def weighted(keepers):
        w = pgg[keepers] / pgg[keepers].sum()
        if1 = np.column_stack([((Gz == group[k]) - pgg[k]) / pgg[keepers].sum()
                               for k in keepers])
        if2 = np.sum(np.column_stack([(Gz == group[k]) - pgg[k] for k in keepers]), 1)[:, None] \
            * (pgg[keepers] / pgg[keepers].sum() ** 2)[None, :]
        wif = (if1 - if2) / n
        return w @ att[keepers], phi[:, keepers] @ w + wif @ att[keepers]

    se = lambda f: float(np.sqrt(np.sum(f ** 2)))     # noqa: E731
    out = {}
    est, f = weighted(np.flatnonzero(post))
    out["simple"] = (est, se(f))
    event = np.array([(p - g) for g, p in zip(group, period)])
    dyn = {}
    for e in np.unique(event):
        dyn[int(e)] = weighted(np.flatnonzero(event == e))
    out["dynamic"] = {e: (v, se(f)) for e, (v, f) in dyn.items()}
    pos = [e for e in dyn if e >= 0]
    out["dynamic_overall"] = (np.mean([dyn[e][0] for e in pos]),
                              se(np.mean([dyn[e][1] for e in pos], axis=0)))
    gvals, gphis = [], []
    for g in cohorts:
        keep = np.flatnonzero((group == g) & post)
        gvals.append(att[keep].mean())
        gphis.append(phi[:, keep].mean(1))
    out["group"] = {g: (v, se(f)) for g, v, f in zip(cohorts, gvals, gphis)}
    pg = np.array([pg_of[g] for g in cohorts])
    wif_rows = ((Gz[:, None] == cohorts[None, :]) - pg[None, :]) / pg.sum() \
        - np.sum((Gz[:, None] == cohorts[None, :]) - pg[None, :], 1)[:, None] * pg / pg.sum() ** 2
    overall = (pg / pg.sum()) @ np.array(gvals)
    f = np.column_stack(gphis) @ (pg / pg.sum()) + (wif_rows / n) @ np.array(gvals)
    out["group_overall"] = (overall, se(f))
    cal = {}
    for t in np.unique(period[post]):
        cal[t] = weighted(np.flatnonzero((period == t) & post))
    out["calendar"] = {t: (v, se(f)) for t, (v, f) in cal.items()}
    out["calendar_overall"] = (np.mean([v for v, _ in cal.values()]),
                               se(np.mean([f for _, f in cal.values()], axis=0)))
    return out


@pytest.mark.parametrize("method,control,base,covariates", [
    ("dr", "never", "varying", []), ("reg", "notyet", "universal", []),
    ("dr", "never", "varying", ["x1", "x2"]), ("ipw", "notyet", "varying", ["x1"]),
    ("reg", "never", "universal", ["x1", "x2"]), ("dr", "notyet", "universal", ["x1"]),
])
def test_csdid_matches_did_drdid_formulas(method, control, base, covariates):
    df = cs_panel()
    result = oe.csdid(data=df, y="y", group="id", time="t", treatment_time="first",
                      x=covariates or None, method=method, control=control, base=base)
    ref, Gz, n = did_att_gt(df, method, control, base, covariates)
    terms = [f"ATT({g:g},{t:g})" for g, t, *_ in ref]
    got = {c.term: c for c in result.coefficients}
    assert list(got) == terms
    assert_allclose([got[t].estimate for t in terms], [r[2] for r in ref], rtol=1e-8,
                    atol=1e-10)
    phi = np.column_stack([r[3] for r in ref])
    assert_allclose(np.array(result.covariance_matrix), phi.T @ phi, rtol=1e-7, atol=1e-12)
    agg = aggte(ref, Gz, n)
    extra = result.extra
    assert_allclose([extra["simple"]["estimate"], extra["simple"]["std_error"]],
                    agg["simple"], rtol=1e-7)
    for row in extra["dynamic"]:
        assert_allclose([row["estimate"], row["std_error"]], agg["dynamic"][row["label"]],
                        rtol=1e-7)
    for key in ("dynamic_overall", "group_overall", "calendar_overall"):
        assert_allclose([extra[key]["estimate"], extra[key]["std_error"]], agg[key],
                        rtol=1e-7)
    for row in extra["group"]:
        assert_allclose([row["estimate"], row["std_error"]], agg["group"][row["label"]],
                        rtol=1e-7)
    for row in extra["calendar"]:
        assert_allclose([row["estimate"], row["std_error"]], agg["calendar"][row["label"]],
                        rtol=1e-7)


# ---- rdrobust: statsmodels WLS identities and set-based nearest-neighbour residuals --


def rd_data(seed=51, n=900, ties=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n)
    if ties:
        x = np.round(x, 2)
    t = (rng.uniform(size=n) < np.where(x >= 0, 0.8, 0.2)).astype(float)
    y = 0.5 + 0.8 * x - 0.6 * x ** 2 + 0.7 * (x >= 0) + rng.normal(size=n) * (0.4 + 0.2 * x)
    yf = 0.5 + 0.8 * x + 1.5 * t + rng.normal(size=n) * 0.5
    return pd.DataFrame({"y": y, "x": x, "t": t, "yf": yf,
                         "cl": rng.integers(0, 60, size=n)})


def kernel_w(u, kernel):
    inside = np.abs(u) <= 1
    k = {"triangular": 1 - np.abs(u), "epanechnikov": 0.75 * (1 - u ** 2),
         "uniform": np.full_like(u, 0.5)}[kernel]
    return np.where(inside, k, 0.0)


def side_wls(df, side, h, order, kernel, cov_type, y="y", cluster=False):
    import statsmodels.api as sm
    x = df["x"].to_numpy()
    mask = (x < 0) if side == "left" else (x >= 0)
    w = kernel_w(x[mask] / h, kernel) / h
    keep = w > 0
    X = np.vander(x[mask][keep], order + 1, increasing=True)
    model = sm.WLS(df[y].to_numpy()[mask][keep], X, weights=w[keep])
    if cluster:
        return model.fit(cov_type="cluster",
                         cov_kwds={"groups": df["cl"].to_numpy()[mask][keep]})
    return model.fit(cov_type=cov_type)


@pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
@pytest.mark.parametrize("vce", ["hc0", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("p", [1, 2])
def test_rd_rows_equal_local_polynomial_wls_when_b_equals_h(kernel, vce, p):
    """rho = 1: Conventional = order-p WLS, Bias-corrected/Robust = order-(p+1) WLS (CCF 2018)."""
    df = rd_data()
    h = 0.45
    result = oe.rdrobust(data=df, y="y", running="x", h=h, p=p, kernel=kernel, vce=vce)
    est, se, _ = coefs(result)
    lo = [side_wls(df, s, h, p, kernel, vce.upper()) for s in ("left", "right")]
    hi = [side_wls(df, s, h, p + 1, kernel, "HC0" if vce == "hc0" else "HC1")
          for s in ("left", "right")]
    assert_allclose(est[0], lo[1].params[0] - lo[0].params[0], rtol=1e-9)
    assert_allclose(est[1], hi[1].params[0] - hi[0].params[0], rtol=1e-9)
    assert_allclose(est[2], est[1], rtol=1e-12)
    assert_allclose(se[0] ** 2, lo[1].bse[0] ** 2 + lo[0].bse[0] ** 2, rtol=1e-8)
    assert_allclose(se[1], se[0], rtol=1e-12)
    if vce in {"hc0", "hc1"}:
        # HC2/HC3 robust rows reuse the order-p leverages (rdrobust_res), unlike OLS HC2/3
        assert_allclose(se[2] ** 2, hi[1].bse[0] ** 2 + hi[0].bse[0] ** 2, rtol=1e-8)
    n_h = [int((kernel_w(df["x"][m].to_numpy() / h, kernel) > 0).sum())
           for m in (df["x"] < 0, df["x"] >= 0)]
    assert [result.metrics["n_h_left"], result.metrics["n_h_right"]] == n_h


def test_rd_cluster_hc0_equals_statsmodels_cluster_wls():
    df = rd_data(seed=7)
    h = 0.5
    result = oe.rdrobust(data=df, y="y", running="x", h=h, vce="hc0", covariance="cluster",
                         cluster="cl")
    _, se, _ = coefs(result)
    lo = [side_wls(df, s, h, 1, "triangular", None, cluster=True) for s in ("left", "right")]
    hi = [side_wls(df, s, h, 2, "triangular", None, cluster=True) for s in ("left", "right")]
    assert_allclose(se[0] ** 2, lo[0].bse[0] ** 2 + lo[1].bse[0] ** 2, rtol=1e-8)
    # rdrobust_vce's (n-1)/(n-k) takes k = ncol(Q) = p + 1 for the robust variance, while the
    # order-(p+1) WLS counts p + 2 coefficients
    rescale = [(fit.nobs - 3) / (fit.nobs - 2) for fit in hi]
    assert_allclose(se[2] ** 2, rescale[0] * hi[0].bse[0] ** 2 + rescale[1] * hi[1].bse[0] ** 2,
                    rtol=1e-8)


def nn_residual_sets(x, y, J):
    """Set definition: own ties, then whole blocks of tied values nearest first (both
    blocks when equidistant) until at least min(J, n - 1) other observations."""
    n = len(x)
    values = np.unique(x)
    out = np.empty(n)
    for i in range(n):
        target = min(J, n - 1)
        chosen = {x[i]}
        count = np.sum(x == x[i]) - 1
        remaining = [v for v in values if v != x[i]]
        while count < target:
            dist = np.array([abs(v - x[i]) for v in remaining])
            nearest = dist.min()
            # rdrobust's NN definition treats floating point equidistance with
            # a relative sqrt(eps) tolerance; rounded mass points need this too.
            add = [v for v, dv in zip(remaining, dist)
                   if abs(dv-nearest) <= max(dv, nearest)*np.sqrt(np.finfo(float).eps)]
            for v in add:
                chosen.add(v)
                count += np.sum(x == v)
                remaining.remove(v)
        members = np.isin(x, list(chosen))
        others = members.copy()
        others[i] = False
        size = others.sum()
        out[i] = np.sqrt(size / (size + 1)) * (y[i] - y[others].mean())
    return out


@pytest.mark.parametrize("ties", [False, True])
def test_rd_nearest_neighbour_variance_from_set_definition(ties):
    df = rd_data(seed=3, n=500, ties=ties)
    h, b, J = 0.35, 0.6, 3
    result = oe.rdrobust(data=df, y="y", running="x", h=h, b=b, nnmatch=J)
    _, se, _ = coefs(result)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    var_cl = var_rb = 0.0
    for side in ("left", "right"):
        m = (x < 0) if side == "left" else (x >= 0)
        xs, ys = x[m], y[m]
        inside = np.abs(xs) < max(h, b) + 0 * xs      # union of the h and b windows
        inside = (kernel_w(xs / h, "triangular") > 0) | (kernel_w(xs / b, "triangular") > 0)
        xs, ys = xs[inside], ys[inside]
        eps = nn_residual_sets(xs, ys, J)
        wh = kernel_w(xs / h, "triangular") / h
        wb = kernel_w(xs / b, "triangular") / b
        Rp, Rq = np.vander(xs, 2, increasing=True), np.vander(xs, 3, increasing=True)
        Gp_inv = np.linalg.inv((Rp * wh[:, None]).T @ Rp)
        Gq_inv = np.linalg.inv((Rq * wb[:, None]).T @ Rq)
        ell = (Rp * wh[:, None]) @ Gp_inv[:, 0]
        var_cl += np.sum(ell ** 2 * eps ** 2)
        # bias-corrected smoother weights: subtract h^2 * L'Gp^-1 e0 times the weights
        # of the quadratic coefficient of the order-q fit at bandwidth b
        L = (Rp * wh[:, None]).T @ (xs / h) ** 2
        quad = (Rq * wb[:, None]) @ Gq_inv[:, 2]
        ell_bc = ell - h ** 2 * (Gp_inv[0] @ L) * quad
        var_rb += np.sum(ell_bc ** 2 * eps ** 2)
    assert_allclose(se[0] ** 2, var_cl, rtol=1e-9)
    assert_allclose(se[2] ** 2, var_rb, rtol=1e-9)


PILOT = {"triangular": 2.576, "epanechnikov": 2.34, "uniform": 1.843}


def rd_residuals(xs, ys, R, w, invG, vce, J, d):
    if ys.ndim == 2:
        return np.column_stack([rd_residuals(xs, ys[:, j], R, w, invG, vce, J, d)
                                for j in range(ys.shape[1])])
    if vce == "nn":
        return nn_residual_sets(xs, ys, J)
    resid = ys - R @ (invG @ ((R * w[:, None]).T @ ys))
    if vce == "hc0":
        return resid
    if vce == "hc1":
        return resid * np.sqrt(len(ys) / (len(ys) - d))
    hii = np.einsum("ij,jk,ik->i", R, invG, R * w[:, None])
    return resid / (np.sqrt(1 - hii) if vce == "hc2" else (1 - hii))


def rd_meat(rows, res, s, clusters):
    """rdrobust_vce: sum (s'e_i)^2 x_i x_i' or the cluster sums with (n-1)/(n-k) G/(G-1)."""
    e = res @ s if res.ndim == 2 else res * s[0]
    scores = rows * e[:, None]
    if clusters is None:
        return scores.T @ scores
    labels, inverse = np.unique(clusters, return_inverse=True)
    totals = np.zeros((len(labels), rows.shape[1]))
    np.add.at(totals, inverse, scores)
    n, k = rows.shape
    return (n - 1) / (n - k) * len(labels) / (len(labels) - 1) * totals.T @ totals


def bw_constants(xs, ys, o, nu, o_b, h_v, h_b, scale, kernel, vce, J, cl=None):
    """rdrobust_bw: ys [n] (sharp) or [n, 2] (outcome, treatment: fuzzy)."""
    w = kernel_w(xs / h_v, kernel) / h_v
    keep = w > 0
    xv, yv, wv = xs[keep], ys[keep], w[keep]
    cv = None if cl is None else cl[keep]
    R = np.vander(xv, o + 1, increasing=True)
    invG = np.linalg.inv((R * wv[:, None]).T @ R)
    beta_v = invG @ ((R * wv[:, None]).T @ yv)
    s = np.array([1.0]) if ys.ndim == 1 else \
        np.array([1 / beta_v[nu, 1], -beta_v[nu, 0] / beta_v[nu, 1] ** 2])
    res = rd_residuals(xv, yv, R, wv, invG, vce, J, o + 1)
    V_V = (invG @ rd_meat(R * wv[:, None], res, s, cv) @ invG)[nu, nu]
    BConst = h_v ** nu * (invG @ ((R * wv[:, None]).T @ (xv / h_v) ** (o + 1)))[nu]
    wb = kernel_w(xs / h_b, kernel) / h_b
    keep = wb > 0
    xb, yb, wbb = xs[keep], ys[keep], wb[keep]
    cb = None if cl is None else cl[keep]
    RB = np.vander(xb, o_b + 1, increasing=True)
    invGB = np.linalg.inv((RB * wbb[:, None]).T @ RB)
    beta_b = invGB @ ((RB * wbb[:, None]).T @ yb)
    reg = 0.0
    if scale > 0:
        res_b = rd_residuals(xb, yb, RB, wbb, invGB, vce, J, o_b + 1)
        meat_b = rd_meat(RB * wbb[:, None], res_b, s, cb)
        reg = 3 * BConst ** 2 * (invGB @ meat_b @ invGB)[o + 1, o + 1]
    top = beta_b[o + 1] @ s if ys.ndim == 2 else beta_b[o + 1]
    return {"V": (2 * nu + 1) * h_v ** (2 * nu + 1) * V_V,
            "B": np.sqrt(2 * (o + 1 - nu)) * BConst * top,
            "R": scale * 2 * (o + 1 - nu) * reg, "rate": 1 / (2 * o + 3)}


def type2_quantile(x, prob):
    s = np.sort(x)
    g = len(s) * prob
    j = int(np.floor(g))
    return s[j] if g - j > 1e-12 else (s[j - 1] + s[j]) / 2


def rdbwselect(df, p, q, kernel, vce, J, method, scale=1.0, y="y", fuzzy=None, cluster=None):
    x = df["x"].to_numpy()
    y = df[y].to_numpy() if fuzzy is None else df[[y, fuzzy]].to_numpy()
    cl = None if cluster is None else df[cluster].to_numpy()
    xl, yl, xr, yr = x[x < 0], y[x < 0], x[x >= 0], y[x >= 0]
    cls = (None, None) if cl is None else (cl[x < 0], cl[x >= 0])
    range_l, range_r = abs(xl.min()), abs(xr.max())
    bw_max = max(range_l, range_r)
    iqr = type2_quantile(x, 0.75) - type2_quantile(x, 0.25)
    c_bw = min(PILOT[kernel] * min(np.std(x, ddof=1), iqr / 1.349) * len(x) ** -0.2, bw_max)

    def const(side, o, nu, o_b, h_b, s):
        xs, ys = (xl, yl) if side == 0 else (xr, yr)
        return bw_constants(xs, ys, o, nu, o_b, c_bw, h_b, s, kernel, vce, J, cls[side])

    dl = const(0, q + 1, q + 1, q + 2, range_l + 1e-8, 0)
    dr = const(1, q + 1, q + 1, q + 2, range_r + 1e-8, 0)
    out = {}

    def pooled(sign):
        d = min(((dl["V"] + dr["V"]) / (dr["B"] + sign * dl["B"]) ** 2) ** dl["rate"], bw_max)
        bl, br = const(0, q, p + 1, q + 1, d, scale), const(1, q, p + 1, q + 1, d, scale)
        b = min(((bl["V"] + br["V"]) / ((br["B"] + sign * bl["B"]) ** 2
                                         + scale * (bl["R"] + br["R"]))) ** bl["rate"], bw_max)
        hl, hr = const(0, p, 0, q, b, scale), const(1, p, 0, q, b, scale)
        h = min(((hl["V"] + hr["V"]) / ((hr["B"] + sign * hl["B"]) ** 2
                                         + scale * (hl["R"] + hr["R"]))) ** hl["rate"], bw_max)
        return (h, h), (b, b)

    def two():
        d = (min((dl["V"] / dl["B"] ** 2) ** dl["rate"], range_l),
             min((dr["V"] / dr["B"] ** 2) ** dl["rate"], range_r))
        bl, br = const(0, q, p + 1, q + 1, d[0], scale), const(1, q, p + 1, q + 1, d[1], scale)
        b = (min((bl["V"] / (bl["B"] ** 2 + scale * bl["R"])) ** bl["rate"], range_l),
             min((br["V"] / (br["B"] ** 2 + scale * br["R"])) ** bl["rate"], range_r))
        hl, hr = const(0, p, 0, q, b[0], scale), const(1, p, 0, q, b[1], scale)
        h = (min((hl["V"] / (hl["B"] ** 2 + scale * hl["R"])) ** hl["rate"], range_l),
             min((hr["V"] / (hr["B"] ** 2 + scale * hr["R"])) ** hl["rate"], range_r))
        return h, b

    base = method.replace("cer", "mse")
    if base in {"mserd", "msecomb1", "msecomb2"}:
        out["mserd"] = pooled(-1)
    if base in {"msesum", "msecomb1", "msecomb2"}:
        out["msesum"] = pooled(+1)
    if base in {"msetwo", "msecomb2"}:
        out["msetwo"] = two()
    if base == "msecomb1":
        h = min(out["mserd"][0][0], out["msesum"][0][0])
        b = min(out["mserd"][1][0], out["msesum"][1][0])
        out[base] = ((h, h), (b, b))
    if base == "msecomb2":
        med = [[float(np.median([out[k][which][side] for k in ("mserd", "msesum", "msetwo")]))
                for side in (0, 1)] for which in (0, 1)]
        out[base] = (tuple(med[0]), tuple(med[1]))
    h, b = out[base]
    if method.startswith("cer"):
        count = len(x) if cl is None else len(np.unique(cls[0])) + len(np.unique(cls[1]))
        factor = count ** (-(p / ((3 + p) * (3 + 2 * p))))
        h = (h[0] * factor, h[1] * factor)
    return h, b


@pytest.mark.parametrize("method", ["mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
                                    "cerrd", "certwo", "cersum", "cercomb1", "cercomb2"])
def test_rd_bandwidth_selectors_match_independent_rdbwselect(method):
    df = rd_data(seed=9, n=600)
    result = oe.rdrobust(data=df, y="y", running="x", bwselect=method)
    h, b = rdbwselect(df, 1, 2, "triangular", "nn", 3, method)
    got = [result.metrics[k] for k in ("h_left", "h_right", "b_left", "b_right")]
    assert_allclose(got, [*h, *b], rtol=1e-8)


@pytest.mark.parametrize("kernel,vce,p", [("epanechnikov", "hc1", 1), ("uniform", "hc3", 1),
                                          ("triangular", "hc0", 2), ("triangular", "nn", 2)])
def test_rd_bandwidth_selection_other_kernels_orders_and_vce(kernel, vce, p):
    df = rd_data(seed=19, n=700)
    result = oe.rdrobust(data=df, y="y", running="x", kernel=kernel, vce=vce, p=p)
    h, b = rdbwselect(df, p, p + 1, kernel, vce, 3, "mserd")
    got = [result.metrics[k] for k in ("h_left", "h_right", "b_left", "b_right")]
    assert_allclose(got, [*h, *b], rtol=1e-8)


def test_csdid_drops_units_treated_in_or_before_the_first_period():
    """Regression: those units were kept in the sample counts and a cohort before the first
    period raised invalid_treatment; did drops them."""
    df = cs_panel(seed=43)
    df.loc[df["id"] == 5, "first"] = 1990.0         # treated before the panel starts
    result = oe.csdid(data=df, y="y", group="id", time="t", treatment_time="first")
    ref, Gz, n = did_att_gt(df, "dr", "never", "varying", [])
    assert result.metrics["n_groups"] == n == df["id"].nunique() - 4
    assert result.nobs == n * df["t"].nunique()
    assert any("treated in or before the first period" in w for w in result.warnings)
    phi = np.column_stack([r[3] for r in ref])
    assert_allclose(np.array(result.covariance_matrix), phi.T @ phi, rtol=1e-7, atol=1e-12)
    assert_allclose(result.extra["simple"]["std_error"], aggte(ref, Gz, n)["simple"][1],
                    rtol=1e-7)


@pytest.mark.parametrize("method", ["mserd", "msetwo", "cerrd"])
def test_rd_fuzzy_bandwidth_uses_the_linearized_ratio(method):
    """rdrobust's default fuzzy selector: rdrobust_bw with s = (1/tau_T, -tau_Y/tau_T^2)."""
    df = rd_data(seed=23, n=700)
    result = oe.rdrobust(data=df, y="yf", running="x", fuzzy="t", bwselect=method)
    h, b = rdbwselect(df, 1, 2, "triangular", "nn", 3, method, y="yf", fuzzy="t")
    got = [result.metrics[k] for k in ("h_left", "h_right", "b_left", "b_right")]
    assert_allclose(got, [*h, *b], rtol=1e-8)
    assert result.extra["bandwidth_equation"] == "fuzzy"
    sharp = oe.rdrobust(data=df, y="yf", running="x", fuzzy="t", bwselect=method, sharpbw=True)
    h, b = rdbwselect(df, 1, 2, "triangular", "nn", 3, method, y="yf")
    got = [sharp.metrics[k] for k in ("h_left", "h_right", "b_left", "b_right")]
    assert_allclose(got, [*h, *b], rtol=1e-8)


def test_rd_fuzzy_perfect_compliance_and_cluster_cer_bandwidth():
    df = rd_data(seed=29, n=700)
    df["t1"] = np.where(df["x"] >= 0, df["t"], 0.0)       # one-sided noncompliance
    result = oe.rdrobust(data=df, y="yf", running="x", fuzzy="t1")
    h, b = rdbwselect(df, 1, 2, "triangular", "nn", 3, "mserd", y="yf")
    assert_allclose([result.metrics["h_left"], result.metrics["b_left"]], [h[0], b[0]],
                    rtol=1e-8)
    assert "perfect compliance" in result.extra["bandwidth_equation"]
    # regression: the CER rate counts clusters (per side) when the variance is clustered
    clustered = oe.rdrobust(data=df, y="y", running="x", bwselect="cerrd", covariance="cluster",
                            cluster="cl")
    h, b = rdbwselect(df, 1, 2, "triangular", "nn", 3, "cerrd", cluster="cl")
    got = [clustered.metrics[k] for k in ("h_left", "h_right", "b_left", "b_right")]
    assert_allclose(got, [*h, *b], rtol=1e-8)


@pytest.mark.parametrize("column", ["a", "i1"])
@pytest.mark.parametrize("estimand", ["ate", "atet"])
def test_nnmatch_single_covariate_sorted_search_matches_brute_force(mdata, column, estimand):
    """One covariate takes the sorted O(n log n) path; i1 has heavy ties."""
    for m, nn in ((1, 2), (4, 3)):
        result = oe.teffects(data=mdata, y="y", treatment="d", x=[column], method="nnmatch",
                             estimand=estimand, neighbors=m, vce_neighbors=nn,
                             biasadj=["b"])
        tau, var, _, _ = brute_matching(mdata, metric_distance(mdata, [column], "mahalanobis"),
                                        estimand, m=m, nn=nn, biasadj=["b"])
        assert_allclose(result.coefficients[0].estimate, tau, rtol=1e-10)
        assert_allclose(result.coefficients[0].std_error ** 2, var, rtol=1e-9)
        assert result.provenance["neighbor_search"] == "sorted one-dimensional"
