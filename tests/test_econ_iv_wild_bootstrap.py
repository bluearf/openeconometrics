"""Independent NumPy/statsmodels oracles for the IV WRE, multiway WCR and score bootstraps.

Every observed statistic is recomputed with explicit NumPy algebra (full 2SLS
sandwiches, explicit restricted fits, cluster score loops) or statsmodels,
and every bootstrap distribution is recomputed in a NumPy loop from the SAME
multiplier matrix, regenerated from the documented private-generator stream.
"""

import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2
import statsmodels.api as sm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.iv import wild_bootstrap as wb
from openecon.econometrics.postest import score_bootstrap as sb
from openecon.econometrics.postest import wild_multipliers as wm
from openecon.econometrics.postest import wild_multiway as mw


@pytest.fixture(autouse=True, scope="module")
def owned_test_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def multipliers(kind, reps, groups, seed):
    return torch.cat(list(wm.blocks(kind, reps, groups, seed, reps))).numpy()


def boot_p(observed, boot, tail="upper"):
    """Plus-one p-value; draws within relative 1e-10 of the observed value count as ties."""
    boot = np.asarray(boot)
    slack = 1e-10 * max(1.0, abs(observed))
    upper = (1 + (boot >= observed - slack).sum()) / (len(boot) + 1)
    lower = (1 + (boot <= observed + slack).sum()) / (len(boot) + 1)
    return {"upper": upper, "lower": lower, "equal_tail": min(1, 2 * min(upper, lower)),
            "symmetric": (1 + (np.abs(boot) >= abs(observed) - slack).sum()) / (len(boot) + 1)}[tail]


# ---- IV data and oracles ----------------------------------------------------------------


def iv_data(n=240, groups=16, seed=1, strength=.5, beta=1.0, unequal=False):
    rng = np.random.default_rng(seed)
    if unequal:
        sizes = np.maximum(2, np.round(np.geomspace(2, 40, groups))).astype(int)
        g = np.repeat(np.arange(groups), sizes)[:n]
        n = len(g)
    else:
        g = np.repeat(np.arange(groups), n // groups)
    cg = rng.normal(size=groups)[g]
    z1, z2, w = rng.normal(size=(3, n))
    v = rng.normal(size=n) * (1 + .5 * np.abs(z1)) + .5 * cg
    u = .6 * v + rng.normal(size=n) + .5 * cg
    y2 = .3 + strength * z1 + .3 * strength * z2 + .4 * w + v
    y1 = 1 + beta * y2 + .5 * w + u
    return pd.DataFrame(dict(y1=y1, y2=y2, z1=z1, z2=z2, w=w, g=g))


class IVOracle:
    def __init__(self, frame, cluster):
        n = len(frame)
        self.n = n
        self.x1 = np.column_stack([np.ones(n), frame.w])
        self.x2 = frame[["z1", "z2"]].to_numpy()
        self.x = np.column_stack([self.x1, self.x2])
        self.y1, self.y2 = frame.y1.to_numpy(), frame.y2.to_numpy()
        self.codes = pd.factorize(frame.g, sort=False)[0] if cluster else np.arange(n)
        self.groups = self.codes.max() + 1
        self.cluster = cluster

    def factor(self, k):
        n, g = self.n, self.groups
        return g / (g - 1) * (n - 1) / (n - k) if self.cluster else n / (n - k)

    def tsls(self, y1, y2, beta0):
        """Full 2SLS sandwich on [y2, X1] with fitted first stage."""
        zhat = np.column_stack([self.x @ np.linalg.lstsq(self.x, y2, rcond=None)[0], self.x1])
        z = np.column_stack([y2, self.x1])
        b = np.linalg.solve(zhat.T @ z, zhat.T @ y1)
        u = y1 - z @ b
        a = np.linalg.inv(zhat.T @ zhat)
        s = np.zeros((self.groups, z.shape[1]))
        np.add.at(s, self.codes, zhat * u[:, None])
        v = self.factor(z.shape[1]) * a @ s.T @ s @ a
        return b[0], math.sqrt(v[0, 0]), (b[0] - beta0) / math.sqrt(v[0, 0])

    def wre(self, beta0, weights):
        yt = self.y1 - beta0 * self.y2
        gamma = np.linalg.lstsq(self.x1, yt, rcond=None)[0]
        u1 = yt - self.x1 @ gamma
        coef = np.linalg.lstsq(np.column_stack([self.x, u1]), self.y2, rcond=None)[0]
        fitted2 = self.x @ coef[:-1]
        u2 = self.y2 - fitted2
        out = []
        for w in weights:
            v = w[self.codes]
            y2s = fitted2 + u2 * v
            y1s = beta0 * y2s + self.x1 @ gamma + u1 * v
            out.append(self.tsls(y1s, y2s, beta0)[2])
        return np.array(out)

    def ar(self, outcome):
        b = np.linalg.lstsq(self.x, outcome, rcond=None)[0]
        e = outcome - self.x @ b
        a = np.linalg.inv(self.x.T @ self.x)
        s = np.zeros((self.groups, self.x.shape[1]))
        np.add.at(s, self.codes, self.x * e[:, None])
        v = self.factor(self.x.shape[1]) * a @ s.T @ s @ a
        idx = [2, 3]
        return b[idx] @ np.linalg.solve(v[np.ix_(idx, idx)], b[idx])

    def ar_boot(self, outcome, weights):
        gamma = np.linalg.lstsq(self.x1, outcome, rcond=None)[0]
        u1 = outcome - self.x1 @ gamma
        return np.array([self.ar(self.x1 @ gamma + u1 * w[self.codes]) for w in weights])


def geometry(frame, cluster):
    return wb.iv_geometry(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                          cluster=cluster, intercept=True, missing="raise")


@pytest.mark.parametrize("cluster", [None, "g"])
@pytest.mark.parametrize("kind", ["rademacher", "webb", "mammen"])
def test_wre_t_statistics_match_full_sandwich_oracle_with_identical_multipliers(cluster, kind):
    frame = iv_data(unequal=cluster is not None)
    g = geometry(frame, cluster)
    oracle = IVOracle(frame, cluster)
    weights = multipliers(kind, 99, g["groups"], 5)
    beta, se, t, ok = wb.tsls_t(g, g["y1"], g["y2"], .8)
    assert bool(ok)
    assert_allclose([float(beta), float(se), float(t)], oracle.tsls(oracle.y1, oracle.y2, .8), rtol=1e-10)
    draws, good = wb.wre_draws(g, .8, torch.tensor(weights))
    assert bool(good.all())
    assert_allclose(draws.numpy(), oracle.wre(.8, weights), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("cluster", [None, "g"])
def test_public_wre_and_ar_p_values_reproduce_oracle_from_regenerated_multipliers(cluster):
    frame = iv_data(seed=3)
    oracle = IVOracle(frame, cluster)
    reps, seed = 199, 11
    groups = oracle.groups
    weights = multipliers("webb", reps, groups, seed)
    for tail in ("symmetric", "equal_tail", "upper", "lower"):
        out = oe.iv_wild_test(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                              null=1.1, cluster=cluster, wild="webb", reps=reps, seed=seed, tail=tail)
        row = out["test"].iloc[0]
        t = oracle.tsls(oracle.y1, oracle.y2, 1.1)[2]
        boot = oracle.wre(1.1, weights)
        expected = boot_p(t, boot, tail)
        assert_allclose(row.statistic, t, rtol=1e-10)
        assert row.bootstrap_p_value == pytest.approx(expected)
    ar = oe.iv_wild_test(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                         null=1.1, cluster=cluster, wild="webb", reps=reps, seed=seed, method="ar")
    row = ar["test"].iloc[0]
    outcome = oracle.y1 - 1.1 * oracle.y2
    stat = oracle.ar(outcome)
    boot = oracle.ar_boot(outcome, weights)
    assert_allclose(row.statistic, stat, rtol=1e-9)
    assert_allclose(row.asymptotic_p_value, chi2.sf(stat, 2), rtol=1e-8)
    assert row.bootstrap_p_value == pytest.approx(boot_p(stat, boot))
    json.dumps(ar.attrs, allow_nan=False)
    assert "\\begin{table}" in ar.to_latex()


def test_ar_confidence_set_brackets_limit_test_and_weak_instrument_unboundedness():
    frame = iv_data(seed=7, strength=.6)
    reps, seed, alpha = 199, 2, .1
    out = oe.iv_wild_confidence_set(data=frame, y="y1", endog="y2", instruments=["z1", "z2"],
                                    x=["w"], cluster="g", method="ar", reps=reps, seed=seed,
                                    alpha=alpha, points=21, refine=10)
    oracle = IVOracle(frame, "g")
    weights = multipliers("rademacher", reps, oracle.groups, seed)

    def p(beta0):
        outcome = oracle.y1 - beta0 * oracle.y2
        stat = oracle.ar(outcome)
        return boot_p(stat, oracle.ar_boot(outcome, weights))

    grid = out["grid"]
    for row in grid.iloc[::5].itertuples():
        assert row.p_value == pytest.approx(p(row.null))
    for boundary in out.attrs["boundaries"]:
        assert p(boundary["accepted_side"]) > alpha >= p(boundary["rejected_side"])
        gap = out.attrs["maximum_grid_gap"]
        assert boundary["bracket_width"] <= gap / 2**10 * (1 + 1e-9)
    # Limit beta0 -> infinity: wild Wald test of the instruments in the first stage.
    limit = oracle.ar(oracle.y2)
    limit_p = boot_p(limit, oracle.ar_boot(oracle.y2, weights))
    assert out.attrs["ar_limit_p_value"] == pytest.approx(limit_p) and limit_p <= alpha
    component = out["components"].iloc[0]
    assert component.lower_status == component.upper_status == "certified bracket"
    weak = iv_data(seed=8, strength=0.0)
    unbounded = oe.iv_wild_confidence_set(data=weak, y="y1", endog="y2", instruments=["z1", "z2"],
                                          x=["w"], method="ar", reps=199, points=11, refine=4,
                                          grid=[-3., -1., 0., 1., 2., 4.])
    assert unbounded.attrs["ar_limit_p_value"] > .05
    statuses = set(unbounded["components"].lower_status) | set(unbounded["components"].upper_status)
    assert any("unbounded" in status for status in statuses)
    json.dumps(unbounded.attrs, allow_nan=False)


def test_wald_confidence_set_uses_shared_multipliers_and_reports_open_tails():
    frame = iv_data(seed=9)
    out = oe.iv_wild_confidence_set(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                                    method="wald", reps=199, points=15, width=8.0, refine=8)
    assert out.attrs["shared_multipliers"] and out.attrs["ar_limit_p_value"] is None
    component = out["components"].iloc[0]
    assert component.lower is not None and component.upper is not None
    estimate = oe.iv_wild_test(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                               null=0.0, reps=199)["test"].estimate.iloc[0]
    assert component.lower < estimate < component.upper
    narrow = oe.iv_wild_confidence_set(data=frame, y="y1", endog="y2", instruments=["z1", "z2"],
                                       x=["w"], method="wald", reps=199, grid=[estimate - .01, estimate + .01],
                                       refine=2)
    assert "not determined beyond grid" in narrow["components"].lower_status.iloc[0]


def test_iv_wild_failure_contract(monkeypatch):
    frame = iv_data()
    base = dict(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"], null=1.0, reps=99)
    for change, code in (({"endog": ["y2", "w"]}, "unsupported_spec"), ({"method": "liml"}, "invalid_option"),
                         ({"wild": "normal"}, "invalid_option"), ({"reps": 10}, "invalid_option"),
                         ({"null": float("nan")}, "invalid_restrictions"), ({"tail": "two"}, "invalid_option"),
                         ({"cluster": ["g", "w"]}, "invalid_cluster"), ({"max_work": 10}, "work_budget_exceeded")):
        with pytest.raises(AnalysisError) as error:
            oe.iv_wild_test(**{**base, **change})
        assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        oe.iv_wild_test(**{**base, "data": frame.assign(y1=np.nan)})
    assert error.value.code == "missing_values"
    original = wb.tsls_t

    def broken(g, y1, y2, beta0):
        beta, se, t, ok = original(g, y1, y2, beta0)
        return beta, se, t, ok if y1.ndim == 1 else torch.zeros_like(ok)

    monkeypatch.setattr(wb, "tsls_t", broken)
    with pytest.raises(AnalysisError) as error:
        oe.iv_wild_test(**base)
    assert error.value.code == "bootstrap_failure"


# ---- multiway WCR -------------------------------------------------------------------------


def multiway_data(n=500, seed=4, k_extra=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"a": rng.integers(0, 10, n), "b": rng.integers(0, 7, n)})
    ea, eb = rng.normal(size=10)[frame.a], rng.normal(size=7)[frame.b]
    frame["x1"] = rng.normal(size=n) + .5 * ea
    frame["x2"] = rng.normal(size=n) + .5 * eb
    for j in range(k_extra):
        frame[f"e{j}"] = rng.normal(size=n) + rng.normal(size=10)[frame.a]
    frame["y"] = 1 + .3 * frame.x1 + ea + eb + rng.normal(size=n)
    return frame


def v3_oracle(x, u, frame):
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    meat = np.zeros((k, k))
    for codes, sign in ((frame.a.to_numpy(), 1), (frame.b.to_numpy(), 1),
                        (frame.a.to_numpy() * 1000 + frame.b.to_numpy(), -1)):
        labels, inverse = np.unique(codes, return_inverse=True)
        s = np.zeros((len(labels), k))
        np.add.at(s, inverse, x * u[:, None])
        meat += sign * len(labels) * (n - 1) / ((len(labels) - 1) * (n - k)) * s.T @ s
    v = bread @ meat @ bread
    w, vectors = np.linalg.eigh(v)
    if w.min() < -64 * k * np.finfo(float).eps * np.abs(w).max():
        v = (vectors * np.maximum(w, 0)) @ vectors.T
    return v


def test_three_term_crve_matches_statsmodels_two_way_cluster():
    frame = multiway_data()
    fit = sm.OLS(frame.y, sm.add_constant(frame[["x1", "x2"]])).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([frame.a, frame.b])})
    result = oe.ols(data=frame, y="y", x=["x1", "x2"])
    g = mw.multiway_geometry(result, frame, ["a", "b"])
    cov, adjusted = mw.multiway_variance(g, (g["y"] - g["x"] @ g["beta"])[None])
    order = [g["terms"].index(t) for t in ("Intercept", "x1", "x2")]
    expected = fit.cov_params().to_numpy()
    assert not bool(adjusted[0])
    assert_allclose(cov[0].numpy()[np.ix_(order, order)], expected, rtol=1e-9)


@pytest.mark.parametrize("level,name", [(0, "a"), (1, "b"), (2, "intersection"), (None, "observation")])
def test_wcr_draws_match_oracle_for_every_bootstrap_clustering(level, name):
    frame = multiway_data()
    result = oe.ols(data=frame, y="y", x=["x1", "x2"])
    g = mw.multiway_geometry(result, frame, ["a", "b"])
    x, y = g["x"].numpy(), frame.y.to_numpy()
    j = g["terms"].index("x2")
    groups = g["n"] if level is None else g["dims"][level][1]
    weights = multipliers("mammen", 99, groups, 3)
    r = torch.zeros((1, g["k"]), dtype=torch.float64)
    r[0, j] = 1
    stat, ok, _ = mw.wcr_draws(g, r, torch.tensor([.1], dtype=torch.float64), torch.tensor(weights), level)
    bread = np.linalg.inv(x.T @ x)
    rr = r.numpy()
    bh = bread @ x.T @ y
    bt = bh - bread @ rr.T @ np.linalg.solve(rr @ bread @ rr.T, rr @ bh - .1)
    ut = y - x @ bt
    codes = {0: pd.factorize(frame.a, sort=False)[0], 1: pd.factorize(frame.b, sort=False)[0],
             2: g["dims"][2][0].numpy(), None: np.arange(len(y))}[level]
    expected = []
    for w in weights:
        ys = x @ bt + ut * w[codes]
        bs = bread @ x.T @ ys
        v = v3_oracle(x, ys - x @ bs, frame)
        expected.append((bs[j] - .1) / math.sqrt(v[j, j]))
    assert bool(ok.all())
    assert_allclose(stat.numpy(), expected, rtol=1e-9, atol=1e-12)
    out = oe.wild_multiway_test(result, data=frame, null={"x2": .1}, clusters=["a", "b"],
                                bootstrap_cluster=name, reps=99, seed=3, wild="mammen")
    observed = (bh[j] - .1) / math.sqrt(v3_oracle(x, y - x @ bh, frame)[j, j])
    row = out["test"].iloc[0]
    assert_allclose(row.statistic, observed, rtol=1e-10)
    assert row.bootstrap_p_value == pytest.approx(
        boot_p(observed, expected, "symmetric"))
    assert out.attrs["bootstrap_cluster"] == name


def test_multiway_joint_wald_truncation_and_fewest_default():
    frame = multiway_data(n=120, seed=6, k_extra=8)
    names = ["x1", "x2", *[f"e{j}" for j in range(8)]]
    result = oe.ols(data=frame, y="y", x=names)
    out = oe.wild_multiway_test(result, data=frame, null={"x1": .3, "x2": 0.0}, clusters=["a", "b"],
                                reps=99)
    assert out.attrs["bootstrap_cluster"] == "b"            # 7 < 10 clusters
    g = mw.multiway_geometry(result, frame, ["a", "b"])
    x, y = g["x"].numpy(), frame.y.to_numpy()
    bh = np.linalg.lstsq(x, y, rcond=None)[0]
    v = v3_oracle(x, y - x @ bh, frame)
    idx = [g["terms"].index("x1"), g["terms"].index("x2")]
    d = bh[idx] - np.array([.3, 0.])
    assert_allclose(out["test"].statistic.iloc[0], d @ np.linalg.solve(v[np.ix_(idx, idx)], d), rtol=1e-8)
    assert out.attrs["observed_truncated"] is True         # k=11 > min(G, H): V3 not PSD
    json.dumps(out.attrs, allow_nan=False)
    with pytest.raises(AnalysisError):
        oe.wild_multiway_test(result, data=frame, null={"x1": .3}, clusters=["a"], reps=99)
    with pytest.raises(AnalysisError):
        oe.wild_multiway_test(result, data=frame, null={"x1": .3}, clusters=["a", "b"],
                              bootstrap_cluster="c", reps=99)
    with pytest.raises(AnalysisError):
        oe.wild_multiway_test(result, data=frame.iloc[::-1], null={"x1": .3}, clusters=["a", "b"], reps=99)


# ---- score bootstrap -------------------------------------------------------------------------


def binary_data(n=400, seed=12, model="logit"):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "c": rng.integers(0, 25, n)})
    noise = rng.logistic(size=n) if model == "logit" else rng.normal(size=n)
    frame["d"] = (.2 + .7 * frame.x1 + rng.normal(size=25)[frame.c] * .3 + noise > 0).astype(float)
    return frame


def score_oracle(frame, model, theta, codes, groups, idx, weights):
    x = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    y = frame.d.to_numpy()
    eta = x @ theta
    if model == "logit":
        p = 1 / (1 + np.exp(-eta))
        grad, w = y - p, p * (1 - p)
    else:
        from scipy.stats import norm
        cdf, pdf = norm.cdf(eta), norm.pdf(eta)
        l1, l0 = pdf / cdf, pdf / (1 - cdf)
        grad = y * l1 - (1 - y) * l0
        w = y * l1 * (l1 + eta) + (1 - y) * l0 * (l0 - eta)
    s = np.zeros((groups, 3))
    np.add.at(s, codes, x * grad[:, None])
    a = np.linalg.inv((x * w[:, None]).T @ x)[idx]
    out = []
    for v in [np.ones(groups)] + list(weights):
        proj = (s * v[:, None]) @ a.T
        total = proj.sum(0)
        centred = proj - proj.mean(0)
        cov = groups / (groups - 1) * centred.T @ centred
        out.append((total @ np.linalg.solve(cov, total), total[0] / math.sqrt(cov[0, 0])))
    return out


@pytest.mark.parametrize("model,cls", [("logit", sm.Logit), ("probit", sm.Probit)])
@pytest.mark.parametrize("cluster", [None, "c"])
def test_score_bootstrap_lm_matches_statsmodels_fits_and_numpy_draws(model, cls, cluster):
    frame = binary_data(model=model)
    reps, seed = 199, 21
    out = oe.score_bootstrap_test(data=frame, y="d", x=["x1", "x2"], null={"x2": .1}, model=model,
                                  cluster=cluster, reps=reps, seed=seed, wild="rademacher")
    estimates = out["estimates"].set_index("term")
    full = cls(frame.d, sm.add_constant(frame[["x1", "x2"]])).fit(disp=0, tol=1e-12)
    link = sm.families.links.Logit() if model == "logit" else sm.families.links.Probit()
    restricted = sm.GLM(frame.d.to_numpy(), sm.add_constant(frame[["x1"]]).to_numpy(),
                        family=sm.families.Binomial(link=link),
                        offset=(.1 * frame.x2).to_numpy()).fit(tol=1e-13)
    assert_allclose(estimates.unrestricted[["Intercept", "x1", "x2"]], full.params, rtol=1e-7, atol=1e-9)
    assert_allclose(estimates.restricted[["Intercept", "x1"]], restricted.params, rtol=1e-7, atol=1e-9)
    assert estimates.restricted["x2"] == .1
    codes = pd.factorize(frame.c, sort=False)[0] if cluster else np.arange(len(frame))
    groups = codes.max() + 1
    weights = multipliers("rademacher", reps, groups, seed)
    theta = estimates.restricted[["Intercept", "x1", "x2"]].to_numpy()
    values = score_oracle(frame, model, theta, codes, groups, [2], weights)
    observed, boot = values[0], values[1:]
    row = out["test"].iloc[0]
    assert_allclose([row.quadratic_form, row.statistic], observed, rtol=1e-6)
    signed = np.array([b[1] for b in boot])
    assert row.bootstrap_p_value == pytest.approx(
        boot_p(observed[1], signed, "symmetric"))
    assert_allclose(row.asymptotic_p_value, chi2.sf(observed[0], 1), rtol=1e-5)


def test_score_bootstrap_wald_route_joint_test_and_contracts():
    frame = binary_data(model="logit")
    reps, seed = 199, 4
    out = oe.score_bootstrap_test(data=frame, y="d", x=["x1", "x2"], null={"x1": .5, "x2": 0.0},
                                  method="wald", cluster="c", reps=reps, seed=seed, wild="webb")
    full = sm.Logit(frame.d, sm.add_constant(frame[["x1", "x2"]])).fit(disp=0, tol=1e-12)
    codes = pd.factorize(frame.c, sort=False)[0]
    weights = multipliers("webb", reps, 25, seed)
    x = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    theta = full.params.to_numpy()
    p = 1 / (1 + np.exp(-x @ theta))
    s = np.zeros((25, 3))
    np.add.at(s, codes, x * (frame.d.to_numpy() - p)[:, None])
    a = np.linalg.inv((x * (p * (1 - p))[:, None]).T @ x)[[1, 2]]
    proj = s @ a.T
    centred = proj - proj.mean(0)
    cov = 25 / 24 * centred.T @ centred
    delta = theta[[1, 2]] - np.array([.5, 0.])
    wald = delta @ np.linalg.solve(cov, delta)
    boot = []
    for v in weights:
        pv = proj * v[:, None]
        c = pv - pv.mean(0)
        boot.append(pv.sum(0) @ np.linalg.solve(25 / 24 * c.T @ c, pv.sum(0)))
    row = out["test"].iloc[0]
    assert_allclose(row.statistic, wald, rtol=1e-6)
    assert row.bootstrap_p_value == pytest.approx(boot_p(wald, boot))
    json.dumps(out.attrs, allow_nan=False)
    assert "\\begin{table}" in out.to_latex()
    with pytest.raises(AnalysisError) as error:
        oe.score_bootstrap_test(data=frame.assign(d=frame.d * 2), y="d", x=["x1"], null={"x1": 0.0})
    assert error.value.code == "invalid_outcome"
    separated = frame.assign(d=(frame.x1 > 0).astype(float))
    with pytest.raises(AnalysisError) as error:
        oe.score_bootstrap_test(data=separated, y="d", x=["x1", "x2"], null={"x2": 0.0}, reps=99)
    assert error.value.code == "convergence_failure"
    for change in ({"model": "cloglog"}, {"method": "score"}, {"null": {"zz": 0.0}}, {"reps": 5},
                   {"null": {"x1": 0.0, "x2": 0.0}, "tail": "lower"}, {"max_work": 1}):
        with pytest.raises(AnalysisError):
            oe.score_bootstrap_test(**{"data": frame, "y": "d", "x": ["x1", "x2"], "null": {"x1": 0.0},
                                       "reps": 99, **change})


# ---- multipliers ------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["rademacher", "webb", "mammen"])
def test_multiplier_laws_and_block_invariant_stream(kind):
    whole = torch.cat(list(wm.blocks(kind, 4000, 50, 9, 4000)))
    pieces = torch.cat(list(wm.blocks(kind, 4000, 50, 9, 37)))
    assert torch.equal(whole, pieces)
    values = whole.numpy().ravel()
    assert abs(values.mean()) < .02 and abs(values.var() - 1) < .02
    state = torch.random.get_rng_state().clone()
    list(wm.blocks(kind, 10, 5, 1, 3))
    assert torch.equal(state, torch.random.get_rng_state())
    assert whole.dtype == torch.float64
    support = {"rademacher": [-1.0, 1.0], "webb": [-math.sqrt(1.5), -1.0, -math.sqrt(.5), math.sqrt(.5), 1.0,
                                                    math.sqrt(1.5)],
               "mammen": [(1 - math.sqrt(5)) / 2, (1 + math.sqrt(5)) / 2]}[kind]
    assert_allclose(np.unique(values), support, rtol=0, atol=0)


@pytest.mark.parametrize("model", ["logit", "probit"])
def test_analytic_link_derivatives_match_finite_differences_and_stay_finite(model):
    eta = torch.tensor([-38., -9., -2.5, -.3, 0., .7, 3., 8.5, 37.], dtype=torch.float64)
    for outcome in (0.0, 1.0):
        y = torch.full_like(eta, outcome)
        loglik, gradient, weight = sb.link_terms(model, eta, y)
        assert bool(torch.isfinite(loglik).all() & torch.isfinite(gradient).all() & torch.isfinite(weight).all())
        assert bool((weight > 0).all())
        h = 1e-5
        up, down = sb.link_terms(model, eta + h, y), sb.link_terms(model, eta - h, y)
        moderate = eta.abs() < 10
        assert_allclose((up[0] - down[0])[moderate] / (2 * h), gradient[moderate], rtol=1e-6, atol=1e-8)
        assert_allclose(-(up[1] - down[1])[moderate] / (2 * h), weight[moderate], rtol=1e-5, atol=1e-8)


def test_portable_summary_state_restores_wild_results():
    frame = iv_data()
    results = [
        oe.iv_wild_test(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"], null=1.0,
                        cluster="g", reps=99),
        oe.iv_wild_confidence_set(data=frame, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                                  reps=99, points=7, refine=3),
        oe.score_bootstrap_test(data=binary_data(), y="d", x=["x1", "x2"], null={"x2": 0.0}, reps=99),
    ]
    data = multiway_data()
    results.append(oe.wild_multiway_test(oe.ols(data=data, y="y", x=["x1", "x2"]), data=data,
                                         null={"x2": 0.0}, clusters=["a", "b"], reps=99))
    for result in results:
        restored = oe.restore_summary(oe.summary_state(result))
        assert restored.attrs == json.loads(json.dumps(result.attrs))
        for name, table in result.items():
            pd.testing.assert_frame_equal(restored[name].reset_index(drop=True),
                                          table.reset_index(drop=True), check_dtype=False)
        assert str(result)


def test_committed_size_power_receipt_is_complete():
    from pathlib import Path
    path = Path(__file__).parent / "fixtures" / "iv_wild_bootstrap" / "size_power.json"
    record = json.loads(path.read_text())
    assert (Path(__file__).parents[1] / "scripts" / "generate_iv_wild_bootstrap_oracles.py").exists()
    for cell in record["cells"]:
        assert cell["planned_datasets"] == cell["completed_datasets"] and not cell["failed_calls"]
        low, high = cell["bootstrap_wilson95"]
        if "null" in cell["cell"]:
            assert low <= .05 <= high <= .10, cell["cell"]


def test_tie_guard_counts_exact_reproductions_of_the_statistic_as_exceedances():
    draws = torch.tensor([1.0 - 1e-15, -1.0, .5, 2.0], dtype=torch.float64)
    assert wm.tail_p_values(1.0, draws, "symmetric") == pytest.approx(4 / 5)
    assert wm.tail_p_values(1.0, draws, "upper") == pytest.approx(3 / 5)
    assert wm.tail_p_values(-1.0, draws, "lower") == pytest.approx(2 / 5)
