"""Independent oracles for oe.vecrank, oe.vec and oe.vec_forecast.

Johansen's reduced-rank regression is re-implemented in NumPy/SciPy with
explicit inverses and a generalized symmetric eigenproblem for all five trend
specifications; statsmodels' coint_johansen (no deterministic terms and
unrestricted constant) and VECM (all five cases) are second oracles. The
statsmodels deterministic codes map to Stata's trend() as

    none -> "n",  rconstant -> "ci",  constant -> "co",  rtrend -> "coli",  trend -> "colo".

statsmodels uses the ML divisor T; OpenEcon follows Stata's T - d, so standard
errors differ by sqrt(T / (T - d)).
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import linalg, stats
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.var import critical_values
from openecon.econometrics.var.johansen import select_rank
from openecon.models import ModelSpec

NAMES = ["y1", "y2", "y3"]
TRENDS = ["none", "rconstant", "constant", "rtrend", "trend"]
DETERMINISTIC = {"none": "n", "rconstant": "ci", "constant": "co", "rtrend": "coli",
                 "trend": "colo"}


def make_data(seed=5, n=300):
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n).cumsum(), rng.normal(size=n).cumsum()
    frame = pd.DataFrame({
        "y1": w1 + rng.normal(size=n),
        "y2": 0.5 * w1 + rng.normal(size=n) + 1.0,
        "y3": w2 + 0.3 * w1 + rng.normal(size=n),
    })
    frame["period"] = np.arange(1, n + 1)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def blocks(y, p, trend):
    """Z0, Z1, Z2 of Stata's vec Methods and formulas (trend = 1-based position)."""
    n, k = y.shape
    dy = np.diff(y, axis=0)
    z0 = dy[p - 1:]
    z1 = y[p - 1:n - 1]
    lagged = [dy[p - 1 - i:n - 1 - i, v] for v in range(k) for i in range(1, p)]
    t = n - p
    index = np.arange(p + 1, n + 1, dtype=float)
    z2 = np.column_stack(lagged) if lagged else np.empty((t, 0))
    if trend == "rconstant":
        z1 = np.column_stack([z1, np.ones(t)])
    if trend == "rtrend":
        z1 = np.column_stack([z1, index])
    if trend == "trend":
        z2 = np.column_stack([z2, index])
    if trend in ("constant", "rtrend", "trend"):
        z2 = np.column_stack([z2, np.ones(t)])
    return z0, z1, z2, index


def johansen_oracle(y, p, trend):
    z0, z1, z2, index = blocks(y, p, trend)
    t, k = z0.shape
    if z2.shape[1]:
        project = np.eye(t) - z2 @ np.linalg.solve(z2.T @ z2, z2.T)
        r0, r1 = project @ z0, project @ z1
    else:
        r0, r1 = z0, z1
    s00, s01, s11 = r0.T @ r0 / t, r0.T @ r1 / t, r1.T @ r1 / t
    values, vectors = linalg.eigh(s01.T @ np.linalg.inv(s00) @ s01, s11)
    order = np.argsort(values)[::-1]
    values, vectors = values[order][:k], vectors[:, order][:, :k]
    return dict(t=t, k=k, z0=z0, z1=z1, z2=z2, s00=s00, s01=s01, s11=s11, values=values,
                vectors=vectors, index=index)


@pytest.mark.parametrize("trend", TRENDS)
def test_vecrank_matches_numpy_johansen(data, trend):
    p = 3
    table = oe.vecrank(data=data, y=NAMES, lags=p, trend=trend)
    o = johansen_oracle(data[NAMES].to_numpy(), p, trend)
    t, k = o["t"], o["k"]
    assert_allclose(table.attrs["eigenvalues"], o["values"], rtol=1e-8)
    assert_allclose(table["eigenvalue"].to_numpy()[1:], o["values"], rtol=1e-8)
    logs = np.log(1 - o["values"])
    assert_allclose(table["trace"].to_numpy()[:k], [-t * logs[r:].sum() for r in range(k)],
                    rtol=1e-8)
    assert_allclose(table["max"].to_numpy()[:k], -t * logs, rtol=1e-8)
    base = -0.5 * t * (k * (np.log(2 * np.pi) + 1) + np.log(np.linalg.det(o["s00"])))
    ll = [base - 0.5 * t * logs[:r].sum() for r in range(k + 1)]
    assert_allclose(table["ll"], ll, rtol=1e-10)
    m1, m2 = o["z1"].shape[1], o["z2"].shape[1]
    parms = [k * m2 + (k + m1 - r) * r for r in range(k + 1)]
    assert list(table["parms"]) == parms
    assert_allclose(table["aic"], [(-2 * a + 2 * b) / t for a, b in zip(ll, parms, strict=True)],
                    rtol=1e-10)
    assert_allclose(table["sbic"], [(-2 * a + np.log(t) * b) / t
                                    for a, b in zip(ll, parms, strict=True)], rtol=1e-10)
    assert_allclose(table["hqic"], [(-2 * a + 2 * np.log(np.log(t)) * b) / t
                                    for a, b in zip(ll, parms, strict=True)], rtol=1e-10)
    assert table.attrs["n_obs"] == t and table.attrs["trend"] == trend
    # Critical values are those of the matching Osterwald-Lenum table.
    for r in range(k):
        for statistic in ("trace", "max"):
            for level in (5, 1):
                assert table[f"{statistic}_cv{level}"][r] == critical_values.lookup(
                    trend, k - r, statistic, level)
    expected = next((r for r in range(k) if table["trace"][r] <= table["trace_cv5"][r]), k)
    assert table.attrs["selected_rank"] == expected == 1


@pytest.mark.parametrize("trend, order", [("none", -1), ("constant", 0)])
def test_vecrank_matches_statsmodels(data, trend, order):
    table = oe.vecrank(data=data, y=NAMES, lags=3, trend=trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = coint_johansen(data[NAMES].to_numpy(), order, 2)
    assert_allclose(table.attrs["eigenvalues"], reference.eig, rtol=1e-8)
    assert_allclose(table["trace"].to_numpy()[:3], reference.lr1, rtol=1e-8)
    assert_allclose(table["max"].to_numpy()[:3], reference.lr2, rtol=1e-8)


def test_critical_value_tables():
    # Values printed in the Stata manual ([TS] vecrank, three variables, trend(constant)).
    assert [critical_values.lookup("constant", d, "trace", 5) for d in (3, 2, 1)] == [
        29.68, 15.41, 3.76]
    assert [critical_values.lookup("constant", d, "trace", 1) for d in (3, 2, 1)] == [
        35.65, 20.04, 6.65]
    assert [critical_values.lookup("constant", d, "max", 5) for d in (3, 2, 1)] == [
        20.97, 14.07, 3.76]
    assert [critical_values.lookup("constant", d, "max", 1) for d in (3, 2, 1)] == [
        25.52, 18.63, 6.65]
    assert critical_values.lookup("rconstant", 2, "trace", 5) == 19.96
    assert critical_values.lookup("rtrend", 2, "trace", 5) == 25.32
    assert critical_values.lookup("none", 2, "trace", 5) == 12.53
    assert critical_values.lookup("trend", 2, "trace", 5) == 18.17
    assert critical_values.lookup("constant", 12, "trace", 5) is None
    assert critical_values.lookup("none", 12, "max", 5) is None
    assert critical_values.lookup("trend", 0, "max", 5) is None
    for trend in TRENDS:
        size = critical_values.available(trend)
        for statistic in ("trace", "max"):
            five = [critical_values.lookup(trend, d, statistic, 5) for d in range(1, size + 1)]
            one = [critical_values.lookup(trend, d, statistic, 1) for d in range(1, size + 1)]
            assert all(b > a for a, b in zip(five, five[1:], strict=False))     # grow with K - r
            assert all(b > a for a, b in zip(one, one[1:], strict=False))
            assert all(o > f for o, f in zip(one, five, strict=True))           # 1% above 5%
        # With one common trend the two statistics coincide.
        assert critical_values.lookup(trend, 1, "trace", 5) == critical_values.lookup(
            trend, 1, "max", 5)
        # The trace statistic is the sum of maximum-eigenvalue type terms: larger for d > 1.
        assert critical_values.lookup(trend, 2, "trace", 5) > critical_values.lookup(
            trend, 2, "max", 5)


def test_critical_values_agree_with_a_simulation_of_the_limit_distribution():
    """Trace quantiles of the limiting functional for two common trends (T = 400 steps)."""
    rng = np.random.default_rng(2024)
    steps, reps, d = 400, 4000, 2
    eps = rng.standard_normal((reps, steps, d))
    lag = np.concatenate([np.zeros((reps, 1, d)), np.cumsum(eps, axis=1)[:, :-1]], axis=1)
    grid = np.arange(1, steps + 1) / steps
    cases = {
        "none": lag,
        "rconstant": np.concatenate([lag, np.ones((reps, steps, 1))], axis=2),
        "rtrend": np.concatenate([lag - lag.mean(axis=1, keepdims=True),
                                  np.broadcast_to((grid - grid.mean())[None, :, None],
                                                  (reps, steps, 1))], axis=2),
    }
    demeaned = lag - lag.mean(axis=1, keepdims=True)
    demeaned[:, :, -1] = grid - grid.mean()
    cases["constant"] = demeaned
    for trend, f in cases.items():
        sef = np.einsum("bti,btj->bij", eps, f)
        sff = np.einsum("bti,btj->bij", f, f)
        trace = np.trace(sef @ np.linalg.solve(sff, sef.transpose(0, 2, 1)), axis1=1, axis2=2)
        assert np.quantile(trace, 0.95) == pytest.approx(
            critical_values.lookup(trend, d, "trace", 5), abs=0.9), trend
        assert np.quantile(trace, 0.99) == pytest.approx(
            critical_values.lookup(trend, d, "trace", 1), abs=2.0), trend


def test_untabulated_dimensions_give_no_critical_value_and_no_selected_rank():
    rng = np.random.default_rng(9)
    names = [f"z{i}" for i in range(12)]
    frame = pd.DataFrame(rng.normal(size=(600, 12)).cumsum(axis=0), columns=names)
    table = oe.vecrank(data=frame, y=names, lags=2, trend="none")
    assert np.isnan(table["trace_cv5"][0]) and np.isnan(table["max_cv1"][0])   # K - r = 12
    assert table["trace_cv5"][1] == 255.27 and table["max_cv1"][11] == 6.51
    assert table.attrs["selected_rank"] is None           # the rule would need K - r = 12
    assert np.isfinite(table["trace"].to_numpy()[:12]).all()
    eight = oe.vecrank(data=frame, y=names[:8], lags=2, trend="none")
    assert eight["trace_cv5"][0] == 141.20 and eight.attrs["selected_rank"] is not None
    default = oe.vecrank(data=frame, y=names[:8], lags=2)
    assert default["trace_cv5"][0] == 156.00 and default.attrs["selected_rank"] is not None
    single = oe.vecrank(data=frame, y=["z0"], lags=2)
    assert list(single["rank"]) == [0, 1] and single["trace_cv5"][0] == 3.76
    assert single["trace"][0] == pytest.approx(single["max"][0])


def test_rank_selection_rule():
    rows = [{"rank": 0, "trace": 50.0, "trace_cv5": 29.68}, {"rank": 1, "trace": 10.0,
                                                           "trace_cv5": 15.41},
            {"rank": 2, "trace": 1.0, "trace_cv5": 3.76}, {"rank": 3, "trace": None,
                                                         "trace_cv5": None}]
    assert select_rank(rows, "trace", 5, 3) == 1
    rows[1]["trace"] = rows[2]["trace"] = 40.0
    assert select_rank(rows, "trace", 5, 3) == 3
    rows[1]["trace_cv5"] = None                       # beyond the tabulated dimensions
    assert select_rank(rows, "trace", 5, 3) is None


def vec_oracle(y, p, r, trend):
    """Stata's vec estimates by explicit matrix algebra."""
    o = johansen_oracle(y, p, trend)
    t, k = o["t"], o["k"]
    z0, z1, z2 = o["z0"], o["z1"], o["z2"]
    m1, m2 = z1.shape[1], z2.shape[1]
    raw = o["vectors"][:, :r]
    beta = raw @ np.linalg.inv(raw[:r])
    alpha = o["s01"] @ beta @ np.linalg.inv(beta.T @ o["s11"] @ beta)
    omega = o["s00"] - alpha @ beta.T @ o["s01"].T
    d = (k * m2 + (k + m1 - r) * r) // k
    ce = z1 @ beta
    w = np.column_stack([ce, z2])
    psi = np.linalg.lstsq(w, z0, rcond=None)[0].T
    mu = rho = None
    has_constant, has_trend = trend in ("constant", "rtrend", "trend"), trend == "trend"
    if has_constant:
        mu = np.linalg.solve(alpha.T @ alpha, alpha.T @ psi[:, -1])
        ce = ce + mu
    if has_trend:
        rho = np.linalg.solve(alpha.T @ alpha, alpha.T @ psi[:, -2])
        ce = ce + np.outer(o["index"], rho)
    w = np.column_stack([ce, z2])
    coef = np.linalg.lstsq(w, z0, rcond=None)[0].T
    cov = np.kron(omega * t / (t - d), np.linalg.inv(w.T @ w))
    beta_cov = np.kron(np.linalg.inv(alpha.T @ np.linalg.inv(omega) @ alpha),
                       np.linalg.inv(o["s11"][r:, r:])) / (t - d)
    logs = np.log(1 - o["values"])
    ll = -0.5 * t * (k * (np.log(2 * np.pi) + 1) + np.log(np.linalg.det(o["s00"])) + logs[:r].sum())
    return dict(alpha=alpha, beta=beta, omega=omega, d=d, coef=coef, cov=cov, mu=mu, rho=rho,
                beta_se=np.sqrt(np.diag(beta_cov)).reshape(r, m1 - r).T, ll=ll, t=t,
                parms=k * m2 + (k + m1 - r) * r, w=w, z0=z0)


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("rank", [1, 2])
def test_vec_matches_numpy_and_statsmodels(data, trend, rank):
    p = 3
    result = oe.vec(data=data, y=NAMES, lags=p, rank=rank, trend=trend)
    y = data[NAMES].to_numpy()
    o = vec_oracle(y, p, rank, trend)
    k, t, d = 3, o["t"], o["d"]
    estimate = np.array([c.estimate for c in result.coefficients])
    assert_allclose(estimate, o["coef"].ravel(), rtol=1e-6, atol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), o["cov"], rtol=1e-6, atol=1e-12)
    assert_allclose(result.extra["alpha"], o["alpha"], rtol=1e-7, atol=1e-10)
    assert_allclose(result.extra["beta_matrix"], o["beta"][:k], rtol=1e-7, atol=1e-10)
    assert_allclose(result.extra["omega"], o["omega"], rtol=1e-8)
    metrics = result.metrics
    assert_allclose(metrics["log_likelihood"], o["ll"], rtol=1e-10)
    assert (metrics["rank"], metrics["n_lags"], metrics["df_eq"], metrics["T"]) == (rank, p, d, t)
    assert metrics["df_model"] == o["parms"]
    assert_allclose(metrics["aic_per_obs"], (-2 * o["ll"] + 2 * o["parms"]) / t, rtol=1e-10)
    assert_allclose(metrics["sbic_per_obs"], (-2 * o["ll"] + np.log(t) * o["parms"]) / t,
                    rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * o["ll"] + np.log(t) * o["parms"], rtol=1e-10)
    assert_allclose(metrics["det_sigma_ml"], np.linalg.det(o["omega"]), rtol=1e-8)
    assert not result.inference["use_t"] and result.nobs == t
    assert result.inference["small_sample_correction"] == pytest.approx(t / (t - d))
    # Cointegrating equations: normalization, free elements and their standard errors.
    for i, equation in enumerate(result.extra["beta"]):
        entries = {e["variable"]: e for e in equation["coefficients"]}
        assert equation["equation"] == f"_ce{i + 1}"
        for j, name in enumerate(NAMES):
            assert entries[name]["estimate"] == pytest.approx(o["beta"][j, i], rel=1e-7, abs=1e-10)
            if j < rank:
                assert entries[name]["estimate"] == (1.0 if j == i else 0.0)
                assert entries[name]["std_error"] is None
            else:
                error = o["beta_se"][j - rank, i]
                assert entries[name]["std_error"] == pytest.approx(error, rel=1e-6)
                z = entries[name]["estimate"] / error
                assert entries[name]["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)),
                                                                 rel=1e-6, abs=1e-300)
        if trend == "rconstant":
            assert entries["_cons"]["std_error"] == pytest.approx(o["beta_se"][-1, i], rel=1e-6)
        elif trend == "none":
            assert "_cons" not in entries
        else:
            assert entries["_cons"]["estimate"] == pytest.approx(o["mu"][i], rel=1e-6)
            assert entries["_cons"]["std_error"] is None
        if trend == "rtrend":
            assert entries["_trend"]["std_error"] == pytest.approx(o["beta_se"][-1, i], rel=1e-6)
        if trend == "trend":
            assert entries["_trend"]["estimate"] == pytest.approx(o["rho"][i], rel=1e-5, abs=1e-9)
            assert entries["_trend"]["std_error"] is None
    # statsmodels VECM: the same ML estimates; its covariance uses the divisor T.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = VECM(y, k_ar_diff=p - 1, coint_rank=rank,
                         deterministic=DETERMINISTIC[trend]).fit()
    assert_allclose(result.extra["alpha"], reference.alpha, rtol=1e-6, atol=1e-9)
    assert_allclose(result.extra["beta_matrix"], reference.beta, rtol=1e-6, atol=1e-9)
    assert_allclose(metrics["log_likelihood"], np.real(reference.llf), rtol=1e-9)
    assert_allclose(result.extra["omega"], reference.sigma_u, rtol=1e-7)
    scale = np.sqrt(t / (t - d))
    rows = {c.term: c for c in result.coefficients}
    ours_alpha = np.array([[rows[f"D_{n}:L._ce{i + 1}"].std_error for i in range(rank)]
                           for n in NAMES])
    assert_allclose(ours_alpha, reference.stderr_alpha * scale, rtol=1e-6)
    gamma = np.array([[[rows[f"D_{n}:{'LD' if i == 1 else f'L{i}D'}.{v}"].estimate
                        for v in NAMES] for n in NAMES] for i in range(1, p)])
    assert_allclose(np.hstack(list(gamma)), reference.gamma, rtol=1e-6, atol=1e-9)
    free = np.array([[e["std_error"] for e in eq["coefficients"] if e["variable"] in NAMES[rank:]]
                     for eq in result.extra["beta"]]).T
    # statsmodels fills its (free rows x r) table of standard errors column by column from a
    # vector that is ordered row by row; undo that before comparing (it only matters when
    # both dimensions exceed one).
    theirs = reference.stderr_coint[rank:]
    theirs = theirs.ravel(order="F").reshape(theirs.shape)
    assert_allclose(free, theirs[:k - rank] * scale, rtol=1e-6)
    if trend in ("rconstant", "rtrend"):
        inside = result.extra["ce_constant" if trend == "rconstant" else "ce_trend"]
        assert_allclose(inside, reference.det_coef_coint.ravel(), rtol=1e-6)


def test_vec_terms_tests_and_structure(data):
    result = oe.vec(data=data, y=NAMES, lags=3, rank=1, trend="trend")
    terms = [c.term for c in result.coefficients[:9]]
    assert terms == ["D_y1:L._ce1", "D_y1:LD.y1", "D_y1:L2D.y1", "D_y1:LD.y2", "D_y1:L2D.y2",
                     "D_y1:LD.y3", "D_y1:L2D.y3", "D_y1:trend", "D_y1:Intercept"]
    assert {c.equation for c in result.coefficients} == {"D_y1", "D_y2", "D_y3"}
    o = vec_oracle(data[NAMES].to_numpy(), 3, 1, "trend")
    estimate = np.array([c.estimate for c in result.coefficients])
    cov = np.array(result.covariance_matrix)
    m = 9
    for j in (1, 2):
        index = [i * m + 1 + v * 2 + j - 1 for i in range(3) for v in range(3)]
        wald = estimate[index] @ np.linalg.solve(cov[np.ix_(index, index)], estimate[index])
        test = result.tests[f"lag_exclusion_L{j}D"]
        assert_allclose(test["statistic"], wald, rtol=1e-7)
        assert test["df"] == 9
    resid = o["z0"] - o["w"] @ o["coef"].T
    t = o["t"]
    whitened = np.linalg.solve(np.linalg.cholesky(o["omega"]), resid.T).T
    b1, b2 = (whitened ** 3).mean(0), (whitened ** 4).mean(0)
    joint = t * (b1 ** 2).sum() / 6 + t * ((b2 - 3) ** 2).sum() / 24
    assert_allclose(result.tests["normality"]["statistic"], joint, rtol=1e-7)
    lagged = np.zeros_like(resid)
    lagged[1:] = resid[:-1]
    w2 = np.column_stack([o["w"], lagged])
    u2 = o["z0"] - w2 @ np.linalg.lstsq(w2, o["z0"], rcond=None)[0]
    lm = (t - (m + 3) - 0.5) * np.log(np.linalg.det(o["omega"]) / np.linalg.det(u2.T @ u2 / t))
    assert_allclose(result.tests["lm_autocorrelation_L1"]["statistic"], lm, rtol=1e-6)
    # Equation table, stability and the stored rank tests.
    record = result.extra["equations"][0]
    assert record["equation"] == "D_y1" and record["parms"] == m
    assert_allclose(record["rmse"], np.sqrt((resid[:, 0] ** 2).sum() / (t - o["d"])), rtol=1e-8)
    stability = result.extra["stability"]
    moduli = [e["modulus"] for e in stability["eigenvalues"]]
    assert stability["unit_moduli_imposed"] == 2 and len(moduli) == 9
    assert_allclose(moduli[:2], 1.0, atol=1e-8)
    assert stability["stable"] is True and moduli[2] < 1.0
    table = oe.vecrank(data=data, y=NAMES, lags=3, trend="trend")
    assert_allclose([r["trace"] for r in result.extra["rank_test"][:3]],
                    table["trace"].to_numpy()[:3], rtol=1e-12)
    assert result.extra["selected_rank"] == table.attrs["selected_rank"]
    # The VAR representation reproduces the fitted differences.
    a = np.array(result.extra["var_representation"]["A"])
    c0 = np.array(result.extra["var_representation"]["constant"])
    c1 = np.array(result.extra["var_representation"]["trend"])
    y = data[NAMES].to_numpy()
    row = 10
    level = sum(a[i] @ y[row - 1 - i] for i in range(3)) + c0 + c1 * (row + 1)
    assert_allclose(level - y[row - 1], (o["w"] @ o["coef"].T)[row - 3], atol=1e-8)
    assert result.predictions[0]["row"] == 3
    assert_allclose(result.predictions[0]["observed"], y[3, 0] - y[2, 0], atol=1e-12)


def test_vec_lags_one_and_two_variables():
    frame = make_data(seed=8, n=200)
    result = oe.vec(data=frame, y=["y1", "y2"], lags=1, rank=1)
    assert [c.term for c in result.coefficients] == ["D_y1:L._ce1", "D_y1:Intercept",
                                                     "D_y2:L._ce1", "D_y2:Intercept"]
    o = vec_oracle(frame[["y1", "y2"]].to_numpy(), 1, 1, "constant")
    assert_allclose([c.estimate for c in result.coefficients], o["coef"].ravel(), rtol=1e-7)
    assert_allclose([c.std_error for c in result.coefficients], np.sqrt(np.diag(o["cov"])),
                    rtol=1e-7)
    assert not any(name.startswith("lag_exclusion") for name in result.tests)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = VECM(frame[["y1", "y2"]].to_numpy(), k_ar_diff=0, coint_rank=1,
                         deterministic="co").fit()
    assert_allclose(result.extra["beta_matrix"], reference.beta, rtol=1e-7)


def test_johansen_estimate_maximizes_the_likelihood():
    """Brute-force maximization of the concentrated likelihood over beta (K = 2, r = 1)."""
    frame = make_data(seed=21, n=250)
    y = frame[["y1", "y2"]].to_numpy()
    o = johansen_oracle(y, 2, "constant")
    t = o["t"]

    def negative(parameter):
        beta = np.array([[1.0], [parameter[0]]])
        middle = beta.T @ o["s11"] @ beta
        omega = o["s00"] - o["s01"] @ beta @ np.linalg.solve(middle, beta.T @ o["s01"].T)
        return 0.5 * t * np.log(np.linalg.det(omega))

    best = minimize(negative, [-1.0], method="BFGS", options={"gtol": 1e-10})
    result = oe.vec(data=frame, y=["y1", "y2"], lags=2, rank=1)
    assert result.extra["beta_matrix"][1][0] == pytest.approx(best.x[0], abs=1e-5)
    ll = -best.fun - 0.5 * t * 2 * (np.log(2 * np.pi) + 1)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-9)


@pytest.mark.parametrize("trend", ["none", "constant", "rtrend"])
def test_vec_forecast(data, trend):
    result = oe.vec(data=data, y=NAMES, time="period", lags=3, rank=1, trend=trend)
    steps = 6
    table = oe.forecast(result, steps)
    y = data[NAMES].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = VECM(y, k_ar_diff=2, coint_rank=1, deterministic=DETERMINISTIC[trend]).fit()
    ours = table.forecast.to_numpy().reshape(3, steps).T
    assert_allclose(ours, reference.predict(steps=steps), rtol=1e-6, atol=1e-7)
    a = np.array(result.extra["var_representation"]["A"])
    phi = [np.eye(3)]
    for i in range(1, steps):
        phi.append(sum(phi[i - j] @ a[j - 1] for j in range(1, min(i, 3) + 1)))
    phi = np.array(phi)
    t, d = result.metrics["T"], result.metrics["df_eq"]
    omega = np.array(result.extra["omega"])
    mse = np.cumsum(phi @ omega @ phi.transpose(0, 2, 1), axis=0) * t / (t - d)
    assert_allclose(table.std_error.to_numpy().reshape(3, steps).T,
                    np.sqrt(np.diagonal(mse, axis1=1, axis2=2)), rtol=1e-8)
    assert list(table.period[:steps]) == list(range(len(y) + 1, len(y) + steps + 1))
    assert oe.vec_forecast(result, steps).equals(table)
    earlier = oe.vec_forecast(result, 1, data=data.iloc[:150])
    c0 = np.array(result.extra["var_representation"]["constant"])
    c1 = np.array(result.extra["var_representation"]["trend"])
    expected = sum(a[i] @ y[149 - i] for i in range(3)) + c0 + c1 * 151
    assert_allclose(earlier.forecast.to_numpy(), expected, atol=1e-8)
    # Impulse responses of the VAR representation (no standard errors, as in Stata).
    responses = oe.irf(result, steps=4)
    assert responses.std_error.isna().all()
    assert "not available" in responses.attrs["standard_errors"]
    block = responses[(responses.impulse == "y1") & (responses.response == "y2")]
    assert_allclose(block.irf.to_numpy(), (phi[:5] @ np.linalg.cholesky(omega))[:, 1, 0],
                    atol=1e-9)
    with pytest.raises(AnalysisError) as error:
        oe.vec_forecast(oe.var(data=data, y=NAMES, lags=1), 3)
    assert error.value.code == "invalid_result"


def test_vec_and_vecrank_error_codes(data):
    cases = [
        (dict(y=NAMES, rank=3), "invalid_rank"),
        (dict(y=NAMES, rank=0), "invalid_spec"),
        (dict(y=["y1"]), "invalid_spec"),
        (dict(y=NAMES, trend="quadratic"), "invalid_spec"),
        (dict(y=NAMES, lags=0), "invalid_spec"),
        (dict(y="y1"), "invalid_spec"),
        (dict(y=["y1", "nope"]), "missing_columns"),
        (dict(y=NAMES, lags=120), "insufficient_observations"),
        (dict(y=NAMES, time="y1"), "invalid_spec"),
    ]
    for options, code in cases:
        with pytest.raises(AnalysisError) as error:
            oe.vec(data=data, **options)
        assert error.value.code == code, options
    twin = data.assign(y4=data.y1 * 3.0)
    for function, options in ((oe.vec, dict(rank=1)), (oe.vecrank, {})):
        with pytest.raises(AnalysisError) as error:
            function(data=twin, y=["y1", "y2", "y4"], lags=2, **options)
        assert error.value.code in {"collinear_system", "singular_residual_covariance"}
    with pytest.raises(AnalysisError) as error:
        oe.vecrank(data=data, y=NAMES, max_rank=-1)
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.vecrank(data=data.drop(index=40), y=NAMES, time="period")
    assert error.value.code == "time_gaps"
    holes = data.copy()
    holes.loc[0, "y2"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.vecrank(data=holes, y=NAMES)
    assert error.value.code == "missing_values"
    trimmed = oe.vecrank(data=holes, y=NAMES, missing="drop")
    assert trimmed.attrs["n_obs"] == len(data) - 3
    short = oe.vecrank(data=data, y=NAMES, max_rank=1)
    assert list(short["rank"]) == [0, 1]
    with pytest.raises(ValidationError):
        ModelSpec(estimator="vec", outcome="y1", predictors=["y2"], columns={"system": NAMES})
    with pytest.raises(AnalysisError) as error:
        oe.fit(ModelSpec(estimator="vec", outcome="y2", columns={"system": NAMES}), data=data)
    assert error.value.code == "invalid_spec"


def test_vec_round_trip_summary_and_latex(data):
    result = oe.vec(data=data, y=NAMES, lags=2, rank=1)
    again = type(result).model_validate_json(result.model_dump_json())
    assert again == result
    json.loads(result.model_dump_json())
    assert_allclose(oe.forecast(again, 3).forecast, oe.forecast(result, 3).forecast, atol=1e-12)
    text = result.summary()
    assert "Vector error-correction model" in text and "[D_y2]" in text and "L._ce1" in text
    assert "Jarque-Bera" in text and "log_likelihood" in text
    assert "LD.y3" in str(result.to_latex())
    assert result.provenance["stata_parity_validated"] is False
    assert result.provenance["trend_specification"] == "constant"
    spec = ModelSpec(estimator="vec", outcome="y1", columns={"system": NAMES},
                     options={"lags": 2, "rank": 1})
    direct = oe.fit(spec, data=data)
    assert_allclose([c.estimate for c in direct.coefficients],
                    [c.estimate for c in result.coefficients], rtol=1e-12)
    shuffled = data.sample(frac=1.0, random_state=2)
    by_time = oe.vec(data=shuffled, y=NAMES, time="period", lags=2, rank=1)
    assert_allclose([c.estimate for c in by_time.coefficients],
                    [c.estimate for c in result.coefficients], rtol=1e-9, atol=1e-12)
    table = oe.vecrank(data=data, y=NAMES)
    assert str(table.to_latex()) and "Osterwald-Lenum" in table.attrs["critical_values"]
    assert oe.capabilities()["estimators"]["vec"]["options"]["trend"]["choices"] == TRENDS
