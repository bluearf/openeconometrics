"""Oracles for oe.csdid (Callaway-Sant'Anna group-time ATTs).

Without covariates ATT(g,t) is a difference in mean changes whose influence-
function standard error equals the HC0 standard error of the treatment dummy
in an OLS of the change on a constant and the dummy (statsmodels). With
covariates the dr / ipw / reg estimators are checked against NumPy
M-estimation with a numerically differentiated Jacobian, and the
aggregations against a NumPy re-implementation of did::aggte's weights and
their influence functions.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from statsmodels.tools.numdiff import approx_fprime

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle


def make_panel(seed=0, n=240, periods=6):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n), periods)
    t = np.tile(np.arange(1, periods + 1), n)
    first = rng.choice([3.0, 4.0, 5.0, np.nan], size=n, p=[0.25, 0.25, 0.2, 0.3])
    x = rng.normal(size=n)
    G = first[unit]
    on = ~np.isnan(G) & (t >= np.nan_to_num(G, nan=99))
    effect = on * (1 + 0.5 * (t - np.nan_to_num(G, nan=0)))
    noise = rng.normal(size=n * periods)
    y = rng.normal(size=n)[unit] + 0.3 * t + 0.4 * x[unit] * t + effect + noise
    return pd.DataFrame({"y": y, "id": unit, "t": t, "first": G, "x": x[unit]})


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def wide(frame):
    y = frame.pivot(index="id", columns="t", values="y").to_numpy()
    first = frame.groupby("id")["first"].first().to_numpy()
    x = frame.groupby("id")["x"].first().to_numpy()
    return y, first, x


def pairs(first, periods, control, base):
    out = []
    for g in sorted(set(first[~np.isnan(first)])):
        gi = periods.index(g)
        for ti in range(len(periods)):
            if base == "universal" and ti == gi - 1:
                continue
            b = gi - 1 if (ti >= gi or base == "universal") else ti - 1
            if b < 0:
                continue
            cohort = first == g
            if control == "never":
                ctrl = np.isnan(first)
            else:
                gidx = np.array([periods.index(v) if not np.isnan(v) else 99 for v in first])
                ctrl = (gidx > max(ti, b)) & ~cohort
            out.append((g, periods[ti], ti, b, cohort, ctrl))
    return out


@pytest.mark.parametrize("control", ["never", "notyet"])
@pytest.mark.parametrize("base", ["varying", "universal"])
def test_unconditional_att_matches_difference_in_means_and_hc0(panel, control, base):
    result = oe.csdid(data=panel, y="y", group="id", time="t", treatment_time="first",
                      control=control, base=base)
    y, first, _ = wide(panel)
    periods = sorted(panel.t.unique())
    expected = pairs(first, periods, control, base)
    assert [c.term for c in result.coefficients] == [f"ATT({g:g},{t:g})"
                                                     for g, t, *_ in expected]
    for coef, (g, t, ti, b, cohort, ctrl) in zip(result.coefficients, expected, strict=True):
        dy = y[:, ti] - y[:, b]
        rows = cohort | ctrl
        ols = sm.OLS(dy[rows], sm.add_constant(cohort[rows].astype(float))).fit(cov_type="HC0")
        assert_allclose(coef.estimate, ols.params[1], rtol=1e-9, atol=1e-12)
        assert_allclose(coef.std_error, ols.bse[1], rtol=1e-8)


def m_oracle(dy, x, d, method):
    xs = sm.add_constant(x)
    k = xs.shape[1]
    gamma = sm.Logit(d, xs).fit(disp=0, tol=1e-12, method="newton").params \
        if method != "reg" else np.zeros(k)
    beta = sm.OLS(dy[d == 0], xs[d == 0]).fit().params if method != "ipw" else np.zeros(k)

    def etas(g, b):
        resid = dy - xs @ b
        e1 = np.sum(d * resid) / d.sum()
        w = (1 - d) * np.exp(xs @ g)
        e0 = np.sum(w * resid) / w.sum()
        return e1, e0

    e1, e0 = etas(gamma, beta)
    theta = np.concatenate([gamma, beta, [e1, e0]])

    def psi(th):
        g, b, h1, h0 = th[:k], th[k:2 * k], th[-2], th[-1]
        p = 1 / (1 + np.exp(-xs @ g))
        resid = dy - xs @ b
        w = (1 - d) * np.exp(xs @ g)
        cols = []
        if method != "reg":
            cols.append(xs * (d - p)[:, None])
        if method != "ipw":
            cols.append(xs * ((1 - d) * resid)[:, None])
        cols.append((d * (resid - h1))[:, None])
        if method != "reg":
            cols.append((w * (resid - h0))[:, None])
        return np.column_stack(cols)

    keep = np.ones(len(theta), bool)
    if method == "reg":
        keep[:k] = False
        keep[-1] = False
    if method == "ipw":
        keep[k:2 * k] = False
    full = theta.copy()

    def summed(sub):
        th = full.copy()
        th[keep] = sub
        return psi(th).sum(0)

    jac = approx_fprime(theta[keep], summed, centered=True)
    contrast = np.zeros(keep.sum())
    contrast[-1 if method == "reg" else -2] = 1
    if method != "reg":
        contrast[-1] = -1
    phi = -psi(theta) @ np.linalg.solve(jac.T, contrast)
    att = e1 - (e0 if method != "reg" else 0)
    return att, np.sqrt(np.sum(phi ** 2))


@pytest.mark.parametrize("method", ["dr", "ipw", "reg"])
def test_covariate_estimators_match_numerical_m_estimation(panel, method):
    result = oe.csdid(data=panel, y="y", group="id", time="t", treatment_time="first",
                      x=["x"], method=method)
    y, first, x = wide(panel)
    periods = sorted(panel.t.unique())
    for coef, (g, t, ti, b, cohort, ctrl) in zip(result.coefficients,
                                                 pairs(first, periods, "never", "varying"),
                                                 strict=True):
        rows = cohort | ctrl
        att, se = m_oracle((y[:, ti] - y[:, b])[rows], x[rows], cohort[rows].astype(float),
                           method)
        assert_allclose(coef.estimate, att, rtol=1e-7, atol=1e-10)
        assert_allclose(coef.std_error, se, rtol=1e-5)


def test_aggregations_match_numpy_with_weight_influence(panel):
    result = oe.csdid(data=panel, y="y", group="id", time="t", treatment_time="first")
    y, first, _ = wide(panel)
    periods = sorted(panel.t.unique())
    n = len(first)
    info = pairs(first, periods, "never", "varying")
    att = np.array([c.estimate for c in result.coefficients])
    phi = np.zeros((n, len(info)))
    for j, (g, t, ti, b, cohort, ctrl) in enumerate(info):
        dy = y[:, ti] - y[:, b]
        phi[cohort, j] = (dy[cohort] - dy[cohort].mean()) / cohort.sum()
        phi[ctrl, j] = -(dy[ctrl] - dy[ctrl].mean()) / ctrl.sum()
    cohorts = [g for g, *_ in info]
    shares = {g: np.mean(first == g) for g in set(cohorts)}
    share_phi = {g: ((first == g) - shares[g]) / n for g in set(cohorts)}

    def aggregate(select):
        raw = np.array([shares[g] if s else 0.0 for g, s in zip(cohorts, select, strict=True)])
        raw_phi = np.column_stack([share_phi[g] if s else np.zeros(n)
                                   for g, s in zip(cohorts, select, strict=True)])
        total = raw.sum()
        w = raw / total
        w_phi = (raw_phi * total - raw[None, :] * raw_phi.sum(1, keepdims=True)) / total ** 2
        return w @ att, phi @ w + w_phi @ att

    post = [t >= g for g, t, *_ in info]
    est, inf = aggregate(post)
    assert_allclose(result.extra["simple"]["estimate"], est, rtol=1e-10)
    assert_allclose(result.extra["simple"]["std_error"], np.sqrt(np.sum(inf ** 2)), rtol=1e-8)
    events = [t - g for g, t, *_ in info]
    dyn = []
    for row in result.extra["dynamic"]:
        est, inf = aggregate([e == row["label"] for e in events])
        assert_allclose(row["estimate"], est, rtol=1e-10)
        assert_allclose(row["std_error"], np.sqrt(np.sum(inf ** 2)), rtol=1e-8)
        if row["label"] >= 0:
            dyn.append(inf)
    overall = np.mean(dyn, axis=0)
    assert_allclose(result.extra["dynamic_overall"]["std_error"], np.sqrt(np.sum(overall ** 2)),
                    rtol=1e-8)
    for row in result.extra["calendar"]:
        est, _ = aggregate([p and t == row["label"] for p, (g, t, *_) in zip(post, info,
                                                                              strict=True)])
        assert_allclose(row["estimate"], est, rtol=1e-10)
    group_means = {g: np.mean([a for a, (gg, t, *_) in zip(att, info, strict=True)
                               if gg == g and t >= g]) for g in shares}
    expected = sum(group_means[g] * shares[g] for g in shares) / sum(shares.values())
    assert_allclose(result.extra["group_overall"]["estimate"], expected, rtol=1e-10)
    pre = [i for i, (g, t, *_) in enumerate(info) if t < g]
    cov = phi.T @ phi
    stat = att[pre] @ np.linalg.solve(cov[np.ix_(pre, pre)], att[pre])
    assert_allclose(result.tests["pretrends"]["statistic"], stat, rtol=1e-7)


def test_failure_contract_and_serialization(panel):
    with pytest.raises(AnalysisError) as caught:
        oe.csdid(data=panel.iloc[1:], y="y", group="id", time="t", treatment_time="first")
    assert caught.value.code == "unbalanced_panel"
    varying = panel.assign(first=np.where(panel.index % 6 == 0, 4.0, panel["first"]))
    with pytest.raises(AnalysisError) as caught:
        oe.csdid(data=varying, y="y", group="id", time="t", treatment_time="first")
    assert caught.value.code == "invalid_treatment"
    everyone = panel.assign(first=panel["first"].fillna(4.0))
    with pytest.raises(AnalysisError) as caught:
        oe.csdid(data=everyone, y="y", group="id", time="t", treatment_time="first")
    assert caught.value.code == "invalid_treatment"
    notyet = oe.csdid(data=everyone, y="y", group="id", time="t", treatment_time="first",
                      control="notyet")
    assert notyet.extra["control_group"] == "notyet"
    off = panel.assign(first=panel["first"] + 0.5)
    with pytest.raises(AnalysisError) as caught:
        oe.csdid(data=off, y="y", group="id", time="t", treatment_time="first")
    assert caught.value.code == "invalid_treatment"
    with pytest.raises(AnalysisError) as caught:
        oe.csdid(data=panel, y="y", group="id", time="t", treatment_time="first", method="aipw")
    assert caught.value.code == "invalid_spec"
    result = oe.csdid(data=panel, y="y", group="id", time="t", treatment_time="first", x=["x"])
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.loads(result.model_dump_json())
    assert "Callaway-Sant'Anna group-time ATT" in result.summary()
    assert "begin{tabular}" in str(result.to_latex())
    assert result.metrics["simple_att"] == result.extra["simple"]["estimate"]
    assert callable(oe.csdid)
