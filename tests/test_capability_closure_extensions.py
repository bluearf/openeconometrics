"""Independent matrix, likelihood, counterfactual and window oracles."""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.optimize as so
import scipy.stats as st
from numpy.testing import assert_allclose
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec, ResultBundle


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def iv_data():
    r = np.random.default_rng(6419)
    n = 300
    z = r.normal(size=(n, 3))
    c = r.normal(size=n)
    v = r.normal(size=n)
    x = z @ [1, 0.7, 0.2] + 0.2 * c + v
    y = 1 + 2 * x - 0.5 * c + 0.7 * v + (1 + 0.6 * abs(z[:, 0])) * r.normal(size=n)
    return pd.DataFrame(
        dict(
            y=y,
            x=x,
            c=c,
            z1=z[:, 0],
            z2=z[:, 1],
            z3=z[:, 2],
            g=np.repeat(np.arange(60), 5),
            t=np.arange(n),
        )
    )


@pytest.mark.parametrize(
    "cluster,center", [(False, False), (False, True), (True, False), (True, True)]
)
def test_cue_independent_optimization_full_covariance_and_j(cluster, center):
    d = iv_data()
    n = len(d)
    x = np.column_stack((np.ones(n), d.c, d.x))
    z = np.column_stack((np.ones(n), d.c, d[["z1", "z2", "z3"]]))
    y = d.y.to_numpy()

    def state(b):
        m = z * (y - x @ b)[:, None]
        g = m.mean(0)
        if center:
            m = m - g
        if cluster:
            m = np.vstack([m[d.g == j].sum(0) for j in d.g.unique()])
        return g, m.T @ m / n

    def objective(b):
        g, s = state(b)
        return n * g @ np.linalg.solve(s, g)

    pz = z @ np.linalg.solve(z.T @ z, z.T @ x)
    start = np.linalg.solve(x.T @ pz, pz.T @ y)
    oracle = so.minimize(objective, start, method="BFGS", options={"gtol": 1e-7})
    b = oe.ivcue(
        data=d,
        y="y",
        x=["c"],
        endog=["x"],
        instruments=["z1", "z2", "z3"],
        center=center,
        cluster="g" if cluster else None,
    )
    assert_allclose(estimates(b), oracle.x, rtol=1e-6, atol=3e-7)
    _, s = state(oracle.x)
    jac = -z.T @ x / n
    v = np.linalg.inv(jac.T @ np.linalg.solve(s, jac)) / n
    assert_allclose(b.covariance_matrix, v, rtol=2e-6, atol=1e-8)
    assert_allclose(b.metrics["j"], objective(oracle.x), rtol=1e-9)
    assert_allclose(b.tests["hansen_j"]["p_value"], st.chi2.sf(objective(oracle.x), 2), rtol=1e-9)
    assert b.extra["jacobian_rank"] == 3
    restored = ResultBundle.model_validate_json(b.model_dump_json())
    assert restored.covariance_matrix == b.covariance_matrix


def test_cue_just_identified_scaling_and_fail_closed():
    d = iv_data()
    b = oe.ivcue(data=d, y="y", endog=["x"], instruments=["z1"])
    reference = oe.ivregress(data=d, y="y", endog=["x"], instruments=["z1"], covariance="robust")
    assert_allclose(estimates(b), estimates(reference), atol=2e-8)
    assert_allclose(b.covariance_matrix, reference.covariance_matrix, rtol=1e-7)
    d2 = d.assign(x=d.x * 100, z1=d.z1 * 0.1)
    scaled = oe.ivcue(data=d2, y="y", endog=["x"], instruments=["z1"])
    assert_allclose(estimates(scaled) * [1, 100], estimates(b), atol=2e-8)
    with pytest.raises(AnalysisError):
        oe.ivcue(data=d, y="y", endog=["x"], instruments=["z1"], max_work=1)
    with pytest.raises(AnalysisError):
        oe.ivcue(data=d.assign(z2=d.z1), y="y", endog=["x"], instruments=["z1", "z2"])
    with pytest.raises(AnalysisError):
        oe.ivcue(data=d, y="y", endog=["x"], instruments=["z1", "z2", "z3"], max_iterations=1)


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster", "hac"])
def test_effective_f_independent_residualized_trace_formula(kind):
    d = iv_data()
    c = np.column_stack((np.ones(len(d)), d.c))
    z = d[["z1", "z2", "z3"]].to_numpy()
    z -= c @ np.linalg.lstsq(c, z, rcond=None)[0]
    x = d.x.to_numpy() - c @ np.linalg.lstsq(c, d.x, rcond=None)[0]
    fitted = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    v = x - fitted
    rows = z * v[:, None]
    if kind == "nonrobust":
        meat = (v @ v) / len(d) * (z.T @ z)
    elif kind == "cluster":
        sums = np.vstack([rows[d.g == j].sum(0) for j in d.g.unique()])
        meat = sums.T @ sums
    else:
        meat = rows.T @ rows
        if kind == "hac":
            for lag in range(1, 4):
                cross = rows[lag:].T @ rows[:-lag]
                meat += (1 - lag / 4) * (cross + cross.T)
    options = (
        {"cluster": "g"} if kind == "cluster" else {"time": "t", "lags": 3} if kind == "hac" else {}
    )
    b = oe.effective_f(
        data=d,
        y="y",
        x=["c"],
        endog=["x"],
        instruments=["z1", "z2", "z3"],
        covariance=kind,
        **options,
    )
    oracle = fitted @ fitted / np.trace(np.linalg.solve(z.T @ z, meat))
    assert_allclose(b["diagnostic"].effective_f.iloc[0], oracle, rtol=2e-12)
    assert b.attrs["critical_values"] is None and b.attrs["rejection"] is None


def panel_data():
    r = np.random.default_rng(534)
    sizes = r.integers(7, 12, size=45)
    i = np.repeat(np.arange(45), sizes)
    u = r.normal(size=45)
    w = r.normal(size=45)
    x1 = r.normal(size=len(i)) + w[i]
    x2 = r.normal(size=len(i)) + u[i]
    z2 = 0.6 * w[i] + 0.7 * u[i] + r.normal(size=45)[i]
    z1 = r.normal(size=45)[i]
    y = 1 + x1 + 2 * x2 - 0.7 * z2 + 0.3 * z1 + u[i] + r.normal(size=len(i))
    return pd.DataFrame(
        dict(y=y, x1=x1, x2=x2, z1=z1, z2=z2, i=i, t=np.concatenate([np.arange(t) for t in sizes]))
    )


def panel_means(x, d):
    means = np.vstack([x[d.i == i].mean(0) for i in d.i.unique()])
    return means, means[d.i.to_numpy()]


def test_cre_independent_fgls_joint_cr1_mundlak_sample_means():
    d = panel_data()
    d.loc[0, "y"] = np.nan
    clean = d.dropna().reset_index(drop=True)
    g = clean.i.nunique()
    n = len(clean)
    base = clean[["x1", "x2", "z1", "z2"]].to_numpy()
    means, expanded = panel_means(base, clean)
    design = np.column_stack((np.ones(n), base, expanded[:, :2]))
    dm, de = panel_means(design, clean)
    ym, ye = panel_means(clean.y.to_numpy()[:, None], clean)
    wx = base - expanded
    wy = clean.y.to_numpy() - ye[:, 0]
    within = np.linalg.lstsq(wx, wy, rcond=None)[0]
    sigma_e2 = np.square(wy - wx @ within).sum() / (n - g - 2)
    group_fit = np.linalg.lstsq(dm, ym[:, 0], rcond=None)[0]
    group_rss = np.square(ym[:, 0] - dm @ group_fit).sum()
    sizes = clean.groupby("i").size().to_numpy()
    harmonic = g / np.sum(1 / sizes)
    sigma_u2 = max(0, group_rss / (g - np.linalg.matrix_rank(dm)) - sigma_e2 / harmonic)
    theta = 1 - np.sqrt(sigma_e2 / (sizes * sigma_u2 + sigma_e2))
    xt = design - theta[clean.i.to_numpy(), None] * de
    yt = clean.y.to_numpy() - theta[clean.i.to_numpy()] * ye[:, 0]
    beta = np.linalg.lstsq(xt, yt, rcond=None)[0]
    bread = np.linalg.inv(xt.T @ xt)
    scores = xt * (yt - xt @ beta)[:, None]
    sums = np.vstack([scores[clean.i == i].sum(0) for i in clean.i.unique()])
    v = bread @ (sums.T @ sums) @ bread * g / (g - 1) * (n - 1) / (n - design.shape[1])
    b = oe.cre(
        data=d,
        y="y",
        x=["x1", "x2", "z1", "z2"],
        means=["x1", "x2"],
        panel="i",
        time="t",
        missing="drop",
    )
    assert_allclose(estimates(b), beta, rtol=2e-10)
    assert_allclose(b.covariance_matrix, v, rtol=2e-9, atol=2e-11)
    test = beta[-2:] @ np.linalg.solve(v[-2:, -2:], beta[-2:]) / 2
    assert_allclose(b.tests["mundlak"]["statistic"], test, rtol=1e-10)
    assert_allclose(b.extra["panel_prediction"]["unit_means"], means[:, :2], rtol=1e-12)
    assert 0 not in b.sample_positions
    restored = ResultBundle.model_validate_json(b.model_dump_json())
    target = d.iloc[1:4].assign(y=99999)
    saved = oe.panel_structural_predict(restored, data=target)["predictions"]
    pred = np.column_stack((np.ones(3), target[["x1", "x2", "z1", "z2"]], means[target.i, :2]))
    assert_allclose(saved["mean"], pred @ beta, rtol=1e-10)
    assert_allclose(saved.std_error, np.sqrt(np.einsum("ij,jk,ik->i", pred, v, pred)), rtol=1e-10)
    with pytest.raises(AnalysisError):
        oe.panel_structural_predict(restored, data=target.assign(i=100))


@pytest.mark.parametrize("robust", [False, True])
def test_hausman_taylor_independent_unbalanced_instruments_and_full_v(robust):
    d = panel_data()
    n = len(d)
    g = d.i.nunique()
    tv = d[["x1", "x2"]].to_numpy()
    zi = np.column_stack((np.ones(n), d.z1, d.z2))
    tvm, tve = panel_means(tv, d)
    zm, ze = panel_means(zi, d)
    ym, ye = panel_means(d.y.to_numpy()[:, None], d)
    wx = tv - tve
    wy = d.y.to_numpy() - ye[:, 0]
    bw = np.linalg.lstsq(wx, wy, rcond=None)[0]
    sigma_e2 = np.square(wy - wx @ bw).sum() / (n - g - 2)
    iv = np.column_stack((zm[:, :2], tvm[:, :1]))
    q = np.linalg.qr(iv)[0]
    group_y = ym[:, 0] - tvm @ bw
    bg = np.linalg.lstsq(q.T @ zm, q.T @ group_y, rcond=None)[0]
    sizes = d.groupby("i").size().to_numpy()
    sigma_u2 = max(0, np.square(group_y - zm @ bg).sum() / (g - 3) - sigma_e2 * np.mean(1 / sizes))
    theta = 1 - np.sqrt(sigma_e2 / (sizes * sigma_u2 + sigma_e2))
    raw = np.column_stack((tv, zi))
    bar = np.column_stack((tvm, zm))
    xt = raw - theta[d.i, None] * bar[d.i]
    yt = d.y.to_numpy() - theta[d.i] * ym[d.i, 0]
    instruments = np.column_stack((wx, (1 - theta[d.i, None]) * iv[d.i]))
    q = np.linalg.qr(instruments)[0]
    xhat = q @ (q.T @ xt)
    bread = np.linalg.inv(xt.T @ xhat)
    beta = bread @ xhat.T @ yt
    v = sigma_e2 * bread
    if robust:
        rows = xhat * (yt - xt @ beta)[:, None]
        sums = np.vstack([rows[d.i == j].sum(0) for j in d.i.unique()])
        v = bread @ (sums.T @ sums) @ bread * g / (g - 1) * (n - 1) / (n - raw.shape[1])
    b = oe.xthtaylor(
        data=d,
        y="y",
        x1=["x1"],
        x2=["x2"],
        z1=["z1"],
        z2=["z2"],
        panel="i",
        covariance="robust" if robust else "nonrobust",
    )
    assert_allclose(estimates(b), beta, rtol=2e-10)
    assert_allclose(b.covariance_matrix, v, rtol=2e-9, atol=1e-11)
    assert_allclose(b.extra["theta"], theta, rtol=2e-12)
    bad = d.copy()
    bad.loc[0, "z2"] += 1
    with pytest.raises(AnalysisError):
        oe.xthtaylor(data=bad, y="y", x1=["x1"], z2=["z2"], panel="i")


def mediation_data():
    r = np.random.default_rng(763)
    n = 250
    a = r.normal(size=n)
    c = r.normal(size=n)
    m = 1 + 0.7 * a + 0.2 * c + r.normal(size=n)
    y = 0.3 + 1.2 * a + 1.3 * m + 0.5 * a * m + 0.1 * c + r.normal(size=n)
    return pd.DataFrame(dict(a=a, m=m, y=y, c=c, g=np.repeat(np.arange(50), 5)))


@pytest.mark.parametrize("cluster", [False, True])
def test_interaction_mediation_independent_counterfactuals_joint_scores_delta(cluster):
    d = mediation_data()
    n = len(d)
    xm = np.column_stack((np.ones(n), d.a, d.c))
    xy = np.column_stack((np.ones(n), d.a, d.m, d.a * d.m, d.c))
    bm = np.linalg.lstsq(xm, d.m, rcond=None)[0]
    by = np.linalg.lstsq(xy, d.y, rcond=None)[0]
    beta = np.r_[bm, by]
    im = xm * (d.m.to_numpy() - xm @ bm)[:, None] @ np.linalg.inv(xm.T @ xm)
    iy = xy * (d.y.to_numpy() - xy @ by)[:, None] @ np.linalg.inv(xy.T @ xy)
    influence = np.column_stack((im, iy))
    if cluster:
        sums = np.vstack([influence[d.g == j].sum(0) for j in d.g.unique()])
        v = sums.T @ sums * 50 / 49 * (n - 1) / (n - 5)
    else:
        influence *= np.r_[np.repeat(np.sqrt(n / (n - 3)), 3), np.repeat(np.sqrt(n / (n - 5)), 5)]
        v = influence.T @ influence

    def targets(b):
        mb, yb = b[:3], b[3:]

        def natural(a, ap):
            m = mb @ [1, ap, d.c.mean()]
            return yb @ [1, a, m, a * m, d.c.mean()]

        y00, y10, y01, y11 = natural(0, 0), natural(1, 0), natural(0, 1), natural(1, 1)
        return np.array([y10 - y00, y11 - y01, y01 - y00, y11 - y10, y11 - y00])

    eps = 1e-5
    jac = np.column_stack(
        [(targets(beta + e) - targets(beta - e)) / (2 * eps) for e in np.eye(len(beta)) * eps]
    )
    b = oe.mediation_interaction(
        data=d, y="y", x=["a"], mediator="m", controls=["c"], cluster="g" if cluster else None
    )
    assert_allclose(estimates(b), targets(beta), rtol=1e-11)
    assert_allclose(b.extra["equation_covariance"], v, rtol=1e-10, atol=1e-12)
    assert_allclose(b.covariance_matrix, jac @ v @ jac.T, rtol=1e-8, atol=1e-10)
    assert_allclose(estimates(b)[-1], estimates(b)[0] + estimates(b)[3], rtol=1e-12)
    with pytest.raises(AnalysisError):
        oe.mediation_interaction(data=d, y="y", x=["a"], mediator="m", interpretation="causal")


@pytest.mark.parametrize("signs", [("+", "+"), ("+", "-"), ("-", "+"), ("-", "-")])
def test_asymmetric_partial_sum_wald_independent_full_var_covariance(signs):
    r = np.random.default_rng(555)
    data = pd.DataFrame(
        dict(y=r.normal(size=120).cumsum(), x=r.normal(size=120).cumsum(), t=np.arange(120))
    )
    level = np.column_stack((data.y, data.x))
    dif = np.diff(level, axis=0)
    partial = np.vstack(
        (
            np.zeros(2),
            np.cumsum(
                np.column_stack(
                    [
                        np.maximum(dif[:, j], 0) if s == "+" else np.minimum(dif[:, j], 0)
                        for j, s in enumerate(signs)
                    ]
                ),
                axis=0,
            ),
        )
    )
    p = 3
    x = np.column_stack(
        (np.ones(len(data) - p), *[partial[p - j : len(data) - j] for j in range(1, p + 1)])
    )
    y = partial[p:]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    u = y - x @ beta
    v = np.kron(u.T @ u / len(y), np.linalg.inv(x.T @ x))
    indices = [2, 4]
    bv = beta[indices, 0]
    vs = v[np.ix_(indices, indices)]
    stat = bv @ np.linalg.solve(vs, bv)
    b = oe.asymcausality(
        data=data.sample(frac=1, random_state=5),
        y="y",
        x="x",
        time="t",
        lags=2,
        dmax=1,
        bootstrap=0,
        signs=signs,
    )
    assert_allclose(b["tests"].statistic.iloc[0], stat, rtol=2e-9)
    assert_allclose(b.attrs["covariance_matrix"], v, rtol=1e-8, atol=1e-9)
    assert_allclose(b["tests"].asymptotic_p_value.iloc[0], st.chi2.sf(stat, 2), rtol=1e-9)


def test_asymmetric_joint_null_draws_seed_budget_and_calendar():
    r = np.random.default_rng(12)
    d = pd.DataFrame(
        dict(y=r.normal(size=80).cumsum(), x=r.normal(size=80).cumsum(), t=np.arange(80))
    )
    state = torch.random.get_rng_state().clone()
    one = oe.asymcausality(data=d, y="y", x="x", time="t", bootstrap=49, seed=11)
    two = oe.asymcausality(data=d, y="y", x="x", time="t", bootstrap=49, seed=11)
    assert torch.equal(state, torch.random.get_rng_state())
    assert one.attrs["bootstrap_statistics"] == two.attrs["bootstrap_statistics"]
    assert len(one.attrs["bootstrap_statistics"]) == 49
    assert one.attrs["null_coefficients"][2][0] == 0
    value = one["tests"].statistic.iloc[0]
    expected = (sum(s >= value for s in one.attrs["bootstrap_statistics"]) + 1) / 50
    assert one["tests"].bootstrap_p_value.iloc[0] == expected
    with pytest.raises(AnalysisError):
        oe.asymcausality(data=d, y="y", x="x", bootstrap=49, max_work=1)
    with pytest.raises(AnalysisError):
        oe.asymcausality(data=d.drop(20), y="y", x="x", time="t", bootstrap=0)


@pytest.mark.parametrize("intercept", [True, False])
def test_recursive_updated_qr_independent_fits_uncertainty_and_restoration(intercept):
    r = np.random.default_rng(45)
    n = 90
    d = pd.DataFrame(dict(x=r.normal(size=n), z=r.normal(size=n), t=np.arange(n)))
    d["y"] = 1 + 2 * d.x - 0.4 * d.z + r.normal(size=n)
    spec = ModelSpec(
        estimator="ols", outcome="y", predictors=["x", "z"], time="t", intercept=intercept
    )
    b = oe.recursive_ols(spec, data=d, minimum=15, step=7)
    for origin in b.attrs["origins"]:
        stop = origin["stop_exclusive"]
        x = d[["x", "z"]].iloc[:stop].to_numpy()
        if intercept:
            x = np.column_stack((np.ones(stop), x))
        y = d.y.iloc[:stop].to_numpy()
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        rss = np.square(y - x @ beta).sum()
        v = np.linalg.inv(x.T @ x) * rss / (stop - x.shape[1])
        assert_allclose([c["estimate"] for c in origin["coefficients"]], beta, atol=1e-12)
        assert_allclose(origin["covariance_matrix"], v, rtol=1e-11)
        restored = ResultBundle.model_validate_json(json.dumps(origin["result"]))
        assert restored.id == origin["result_id"]
        assert_allclose(
            restored.coefficients[0].p_value,
            2 * st.t.sf(abs(beta[0] / np.sqrt(v[0, 0])), stop - x.shape[1]),
            rtol=1e-10,
        )
    changed = d.copy()
    changed.loc[50:, "y"] += 999
    again = oe.recursive_ols(spec, data=changed, minimum=15, step=7)
    assert again.attrs["origins"][0]["coefficients"] == b.attrs["origins"][0]["coefficients"]


def test_reverse_windows_fit_correct_suffixes_and_no_forward_lookahead_claim():
    r = np.random.default_rng(56)
    d = pd.DataFrame(dict(x=r.normal(size=40), y=r.normal(size=40), t=np.arange(40)))
    spec = ModelSpec(estimator="ols", outcome="y", predictors=["x"], time="t")
    b = oe.rolling(spec, data=d, window=12, step=7, expanding=True, reverse=True)
    assert [(o["start"], o["stop_exclusive"]) for o in b.attrs["origins"]] == [
        (28, 40),
        (21, 40),
        (14, 40),
        (7, 40),
        (0, 40),
    ]
    for o in b.attrs["origins"]:
        subset = d.iloc[o["start"] :]
        beta = np.linalg.lstsq(
            np.column_stack((np.ones(len(subset)), subset.x)), subset.y, rcond=None
        )[0]
        assert_allclose([c["estimate"] for c in o["coefficients"]], beta, rtol=1e-10)
    with pytest.raises(AnalysisError):
        oe.rolling(spec, data=d, window=12, reverse=True, forecast_steps=1)


def test_rolling_selection_is_trained_per_origin_and_future_outcomes_only_score():
    r = np.random.default_rng(56)
    d = pd.DataFrame(dict(y=r.normal(size=80), t=np.arange(80)))
    spec = ModelSpec(estimator="arima", outcome="y", time="t", options={"order": [0, 0, 0]})
    settings = {"d": 0, "max_p": 0, "max_q": 0, "constant": True}
    b = oe.rolling(
        spec, data=d, window=30, step=30, selection=settings, forecast_steps=2, evaluate=True
    )
    changed = d.copy()
    changed.loc[30:, "y"] += 3
    again = oe.rolling(
        spec, data=changed, window=30, step=30, selection=settings, forecast_steps=2, evaluate=True
    )
    assert b.attrs["origins"][0]["coefficients"] == again.attrs["origins"][0]["coefficients"]
    assert b.attrs["origins"][0]["auto_selection"]["candidate_count"] == 1
    assert b["evaluation"].target_position.iloc[0] == 30
    assert b["evaluation"].forecast.iloc[0] == again["evaluation"].forecast.iloc[0]
    assert b["evaluation"].observed.iloc[0] != again["evaluation"].observed.iloc[0]
    assert_allclose(b["evaluation"].squared_error, b["evaluation"].error ** 2)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_fairlie_binary_reference_matching_and_first_bootstrap_independently(link):
    r = np.random.default_rng(19)
    n = 240
    groups = np.repeat(["A", "B"], n // 2)
    x = r.normal(size=n) + (groups == "A") * 0.7
    probability = st.norm.cdf(0.3 + x) if link == "probit" else 1 / (1 + np.exp(-(0.3 + x)))
    d = pd.DataFrame(dict(y=r.binomial(1, probability), x=x, g=groups))

    def independent(rows):
        selected = d.iloc[rows]
        xx = np.column_stack((np.ones(len(rows)), selected.x))
        y = selected.y.to_numpy()

        def objective(b):
            eta = xx @ b
            if link == "logit":
                return np.sum(np.logaddexp(0, eta) - y * eta)
            return -np.sum(np.where(y == 1, st.norm.logcdf(eta), st.norm.logsf(eta)))

        beta = so.minimize(objective, np.zeros(2), method="BFGS", options={"gtol": 1e-7}).x
        predicted = st.norm.cdf(xx @ beta) if link == "probit" else 1 / (1 + np.exp(-(xx @ beta)))
        a = selected.g.to_numpy() == "A"
        gap = y[a].mean() - y[~a].mean()
        explained = predicted[a].mean() - predicted[~a].mean()
        return beta, np.array([gap, explained, gap - explained, explained])

    rng_state = torch.random.get_rng_state().clone()
    b = oe.fairlie(
        data=d,
        y="y",
        x=["x"],
        group="g",
        groups=["A", "B"],
        link=link,
        reps=49,
        matching_reps=1,
        seed=3,
    )
    beta, values = independent(np.arange(n))
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert_allclose(b.extra["reference_parameters"], beta, atol=2e-6)
    assert_allclose(estimates(b), values, atol=2e-7)
    _, first = independent(b.extra["bootstrap_first_positions"])
    assert_allclose(b.extra["bootstrap_estimates"][0], first, atol=3e-7)
    assert_allclose(
        b.covariance_matrix, np.cov(np.array(b.extra["bootstrap_estimates"]).T), rtol=1e-12
    )
    assert abs(b.extra["matching_mc_remainder"]) < 1e-12
    assert ResultBundle.model_validate_json(b.model_dump_json()).extra["seed"] == 3
    with pytest.raises(AnalysisError):
        oe.fairlie(data=d, y="y", x=["x"], group="g", groups=["B", "C"], reps=49)


def test_asymmetric_first_joint_null_replication_independently():
    r = np.random.default_rng(54)
    d = pd.DataFrame(dict(y=r.normal(size=90).cumsum(), x=r.normal(size=90).cumsum()))
    b = oe.asymcausality(data=d, y="y", x="x", bootstrap=49, seed=13)
    difference = np.diff(d[["y", "x"]].to_numpy(), axis=0)
    partial = np.vstack((np.zeros(2), np.maximum(difference, 0).cumsum(0)))
    p = 2
    design = np.column_stack(
        (np.ones(len(d) - p), *[partial[p - lag : len(d) - lag] for lag in range(1, p + 1)])
    )
    null = np.array(b.attrs["null_coefficients"])
    residual = partial[p:] - design @ null
    residual -= residual.mean(0)
    noise = residual[b.attrs["bootstrap_first_indices"]]
    simulated = partial.copy()
    for t in range(p, len(simulated)):
        row = np.r_[1, *[simulated[t - lag] for lag in range(1, p + 1)]]
        simulated[t] = row @ null + noise[t - p]
    x = np.column_stack(
        (np.ones(len(d) - p), *[simulated[p - lag : len(d) - lag] for lag in range(1, p + 1)])
    )
    y = simulated[p:, 0]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    variance = np.square(y - x @ beta).sum() / len(y) * np.linalg.inv(x.T @ x)
    oracle = beta[2] ** 2 / variance[2, 2]
    assert_allclose(b.attrs["bootstrap_statistics"][0], oracle, rtol=2e-9)


def test_panel_windows_cut_calendars_and_retain_unbalanced_original_rows():
    d = panel_data().sample(frac=1, random_state=14).reset_index(drop=True)
    spec = ModelSpec(
        estimator="xtreg",
        outcome="y",
        predictors=["x1", "x2"],
        panel="i",
        time="t",
        options={"model": "pooled"},
    )
    windows = oe.rolling_panel(spec, data=d, window=5, step=3, expanding=True)
    for record in windows.attrs["origins"]:
        selected = d[(d.t >= record["start_period"]) & (d.t <= record["end_period"])]
        x = np.column_stack((np.ones(len(selected)), selected.x1, selected.x2))
        beta = np.linalg.lstsq(x, selected.y, rcond=None)[0]
        residual = selected.y - x @ beta
        v = np.linalg.inv(x.T @ x) * np.sum(residual**2) / (len(selected) - 3)
        assert_allclose([c["estimate"] for c in record["coefficients"]], beta, rtol=1e-11)
        assert_allclose(record["covariance_matrix"], v, rtol=1e-10)
        assert set(record["input_positions"]) == set(selected.index)
        assert max(d.iloc[record["sample_positions"]].t) == record["end_period"]
    changed = d.copy()
    changed.loc[changed.t > 4, "y"] += 900
    again = oe.rolling_panel(spec, data=changed, window=5, step=3, expanding=True)
    assert windows.attrs["origins"][0]["coefficients"] == again.attrs["origins"][0]["coefficients"]
    reverse = oe.rolling_panel(spec, data=d, window=5, step=3, expanding=True, reverse=True)
    assert reverse.attrs["origins"][0]["start_period"] == 6
    with pytest.raises(AnalysisError):
        oe.rolling_panel(spec, data=d[d.t != 4], window=5)
