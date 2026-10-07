"""Independent oracles for oe.stcox and oe.stcurve.

statsmodels PHReg (Breslow/Efron ties, strata, delayed entry, offset, score
residuals, baseline hazard, Lin-Wei robust covariance via ``groups``) is the
main oracle. Note: PHReg counts a record whose entry equals a failure time as
at risk at that time, whereas Stata (and OpenEcon) use ``t0 < t <= t``; oracle
data with entries equal to failure times shift the entries by 1e-9 for PHReg.
The exact partial likelihood, weighted likelihoods, Schoenfeld-residual PH
tests and Harrell's C are recomputed by explicit O(n^2) loops in NumPy and
brute-force SciPy maximization.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.numdiff import approx_hess

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival.cox_kernels import CoxObjective, ExactObjective
from openecon.econometrics.survival.data import RiskSets
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle

X = ["x1", "x2", "x3"]


def make_data(seed=3, n=400, grid=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    t = np.round(rng.exponential(np.exp(-x @ [0.5, -0.3, 0.2])) * grid) / grid + 1 / grid
    c = np.round(rng.exponential(1.5, n) * grid) / grid + 1 / grid
    d = (t <= c).astype(float)
    t = np.minimum(t, c)
    frame = pd.DataFrame({"t": t, "d": d, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2],
                          "s": rng.integers(0, 3, n), "off": 0.2 * rng.normal(size=n),
                          "cl": rng.integers(0, 40, n), "w": rng.integers(1, 4, n),
                          "pw": rng.uniform(0.5, 2.0, n)})
    frame["t0"] = np.where(rng.random(n) < 0.3, frame.t * rng.uniform(0.1, 0.8, n), 0.0)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def est(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_matches_phreg_with_strata_entry_and_offset(data, ties):
    result = oe.stcox(data=data, time="t", failure="d", x=X, entry="t0", strata="s",
                      offset="off", ties=ties)
    model = PHReg(data.t, data[X], status=data.d, entry=data.t0, strata=data.s,
                  offset=data.off, ties=ties).fit()
    b, se = est(result)
    assert_allclose(b, model.params, rtol=1e-8, atol=1e-10)
    assert_allclose(se, model.bse, rtol=1e-8)
    assert result.metrics["log_likelihood"] == pytest.approx(model.llf, rel=1e-11)
    null = PHReg(data.t, data[X], status=data.d, entry=data.t0, strata=data.s,
                 offset=data.off, ties=ties).loglike(np.zeros(3))
    assert result.metrics["log_likelihood_null"] == pytest.approx(null, rel=1e-11)
    lr = 2 * (model.llf - null)
    assert result.tests["model"]["statistic"] == pytest.approx(lr, rel=1e-9)
    assert result.inference["distribution"] == "normal"
    hr = result.extra["hazard_ratios"]["x1"]
    assert hr["hazard_ratio"] == pytest.approx(np.exp(b[0]))
    assert hr["std_error"] == pytest.approx(np.exp(b[0]) * se[0])
    assert result.metrics["n_failures"] == data.d.sum()
    assert result.metrics["time_at_risk"] == pytest.approx((data.t - data.t0).sum())


def test_score_test_and_derivatives(data):
    result = oe.stcox(data=data, time="t", failure="d", x=X)
    model = PHReg(data.t, data[X], status=data.d)
    g, h = model.score(np.zeros(3)), model.hessian(np.zeros(3))
    assert result.tests["score"]["statistic"] == pytest.approx(g @ np.linalg.solve(-h, g),
                                                               rel=1e-10)
    tensor = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    risk = RiskSets.build(tensor(data.t), tensor(data.t0), tensor(data.d), torch.tensor(data.s))
    theta = tensor([0.3, -0.2, 0.1])
    for ties, weights in (("breslow", tensor(data.pw)), ("efron", None)):
        report = check_derivatives(CoxObjective(tensor(data[X]), risk, weights,
                                                tensor(data.off), ties), theta)
        assert report["gradient_max_rel_error"] < 1e-7
        assert report["hessian_max_rel_error"] < 1e-7
    exact = ExactObjective(tensor(data[X]), risk, tensor(data.d), None)
    report = check_derivatives(exact, theta)
    assert report["gradient_max_rel_error"] < 1e-7 and report["hessian_max_rel_error"] < 1e-7


def exact_loglike(beta, t, d, x):
    """Exact partial likelihood by enumerating the subsets of each risk set."""
    eta = x @ beta
    total = 0.0
    for time in np.unique(t[d == 1]):
        risk = np.flatnonzero(t >= time)
        dead = np.flatnonzero((t == time) & (d == 1))
        denominator = sum(np.exp(eta[list(c)].sum()) for c in itertools.combinations(risk,
                                                                                     len(dead)))
        total += eta[dead].sum() - np.log(denominator)
    return total


def test_exact_partial_likelihood_matches_enumeration():
    rng = np.random.default_rng(11)
    t = np.repeat(np.arange(1.0, 13.0), [3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2])
    frame = pd.DataFrame({"t": t, "d": (rng.random(t.size) < 0.8).astype(float),
                          "x1": rng.normal(size=t.size) - 0.1 * t,
                          "x2": rng.normal(size=t.size)})
    result = oe.stcox(data=frame, time="t", failure="d", x=["x1", "x2"], ties="exactp")
    x, t, d = frame[["x1", "x2"]].to_numpy(), frame.t.to_numpy(), frame.d.to_numpy()
    fitted = minimize(lambda b: -exact_loglike(b, t, d, x), np.zeros(2), method="BFGS",
                      options={"gtol": 1e-10})
    b, se = est(result)
    assert_allclose(b, fitted.x, atol=1e-6)
    hessian = approx_hess(fitted.x, lambda v: exact_loglike(v, t, d, x))
    assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=1e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun, rel=1e-10)
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=frame, time="t", failure="d", x=["x1"], ties="exactp",
                 covariance="robust")
    assert error.value.code == "invalid_spec"


def test_robust_and_cluster_covariances_follow_stata(data):
    n = len(data)
    robust = oe.stcox(data=data, time="t", failure="d", x=X, covariance="robust")
    model = PHReg(data.t, data[X], status=data.d).fit(groups=np.arange(n))
    assert_allclose(est(robust)[1], model.bse * np.sqrt(n / (n - 1)), rtol=1e-8)
    assert "N/(N-1)" in robust.inference["correction"]
    # Efron: statsmodels' score residuals are Breslow-only, so the sandwich is rebuilt
    # from the explicit Efron residual loop and PHReg's Efron Hessian.
    efron = oe.stcox(data=data, time="t", failure="d", x=X, covariance="robust", ties="efron")
    b = est(efron)[0]
    bread = np.linalg.inv(-PHReg(data.t, data[X], status=data.d, ties="efron").hessian(b))
    w = efron_residuals(b, data.t.to_numpy(), data.d.to_numpy(), data[X].to_numpy())
    expected = n / (n - 1) * bread @ (w.T @ w) @ bread
    assert_allclose(np.array(efron.covariance_matrix), expected, rtol=1e-7)
    cluster = oe.stcox(data=data, time="t", failure="d", x=X, cluster="cl")
    model = PHReg(data.t, data[X], status=data.d).fit(groups=data.cl.to_numpy())
    g = data.cl.nunique()
    assert_allclose(est(cluster)[1], model.bse * np.sqrt(g / (g - 1)), rtol=1e-8)
    assert robust.tests["model"]["label"].startswith("Wald")


def multi_record(frame):
    """Split every subject at a covariate change: same subject, two records."""
    first = frame.assign(stop=frame.t / 2, event=0.0, start=0.0, z=frame.x3)
    second = frame.assign(stop=frame.t, event=frame.d, start=frame.t / 2, z=frame.x3 + 1)
    out = pd.concat([first, second]).sort_index(kind="stable").reset_index()
    return out.rename(columns={"index": "pid"})


def test_multiple_records_with_id_cluster_on_subject(data):
    long = multi_record(data)
    result = oe.stcox(data=long, time="stop", failure="event", entry="start", id="pid",
                      x=["x1", "x2", "z"], covariance="robust")
    oracle = PHReg(long.stop, long[["x1", "x2", "z"]], status=long.event,
                   entry=long.start + 1e-9)
    model = oracle.fit()
    b, se = est(result)
    assert_allclose(b, model.params, rtol=1e-8)
    g = long.pid.nunique()
    residuals = np.nan_to_num(oracle.score_residuals(model.params))   # NaN: never at risk
    summed = pd.DataFrame(residuals).groupby(long.pid.to_numpy()).sum().to_numpy()
    bread = np.linalg.inv(-oracle.hessian(model.params))
    expected = g / (g - 1) * bread @ (summed.T @ summed) @ bread
    assert_allclose(se, np.sqrt(np.diag(expected)), rtol=1e-7)
    assert result.metrics["n_subjects"] == g and result.nobs == len(long)
    assert result.inference["cluster_column"] == "pid"
    assert result.metrics["concordance"] is None
    overlapping = long.assign(start=0.0)
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=overlapping, time="stop", failure="event", entry="start", id="pid",
                 x=["x1"])
    assert error.value.code == "overlapping_records"


def weighted_breslow(beta, t, d, x, w):
    eta = x @ beta
    total = 0.0
    for time in np.unique(t[d == 1]):
        risk = t >= time
        dead = (t == time) & (d == 1)
        total += (w[dead] * eta[dead]).sum() - w[dead].sum() * np.log(
            (w[risk] * np.exp(eta[risk])).sum())
    return total


def test_weights_follow_stata(data):
    expanded = data.loc[data.index.repeat(data.w)].reset_index(drop=True)
    fw = oe.stcox(data=data, time="t", failure="d", x=X, weights="w", weight_type="fweight")
    plain = oe.stcox(data=expanded, time="t", failure="d", x=X)
    assert_allclose(est(fw)[0], est(plain)[0], rtol=1e-9)
    assert_allclose(est(fw)[1], est(plain)[1], rtol=1e-9)
    assert fw.nobs == len(expanded)
    x, t, d = data[X].to_numpy(), data.t.to_numpy(), data.d.to_numpy()
    for kind in ("iweight", "pweight"):
        result = oe.stcox(data=data, time="t", failure="d", x=X, weights="pw", weight_type=kind,
                          covariance="robust")
        w = data.pw.to_numpy() * (len(data) / data.pw.sum() if kind == "pweight" else 1.0)
        fitted = minimize(lambda b: -weighted_breslow(b, t, d, x, w), np.zeros(3),
                          method="BFGS", options={"gtol": 1e-9})
        assert_allclose(est(result)[0], fitted.x, atol=1e-6)
        assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun, rel=1e-9)
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=data, time="t", failure="d", x=X, weights="w", weight_type="fweight",
                 ties="efron")
    assert error.value.code == "unsupported_weights"
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=data, time="t", failure="d", x=X, weights="pw", weight_type="pweight",
                 covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"


def efron_residuals(beta, t, d, x):
    """Lin-Wei score residuals under Efron ties by an explicit loop."""
    n, k = x.shape
    u = np.exp(x @ beta)
    out = np.zeros((n, k))
    for time in np.unique(t[d == 1]):
        risk = np.flatnonzero(t >= time)
        dead = np.flatnonzero((t == time) & (d == 1))
        c = len(dead)
        mean_sum = np.zeros(k)
        for r in range(c):
            weight = np.ones(n)
            weight[dead] = 1 - r / c
            phi = (u[risk] * weight[risk]).sum()
            xbar = (u[risk] * weight[risk]) @ x[risk] / phi
            mean_sum += xbar
            out[risk] -= (u[risk] * weight[risk] / phi)[:, None] * (x[risk] - xbar)
        out[dead] += x[dead] - mean_sum / c
    return out


def test_score_and_schoenfeld_residuals(data):
    small = data.iloc[:120]
    tensor = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    risk = RiskSets.build(tensor(small.t), None, tensor(small.d))
    beta = np.array([0.3, -0.2, 0.1])
    efron = CoxObjective(tensor(small[X]), risk, None, None, "efron")
    residuals, _ = efron.residuals(tensor(beta))
    expected = efron_residuals(beta, small.t.to_numpy(), small.d.to_numpy(),
                               small[X].to_numpy())
    assert_allclose(residuals.numpy(), expected, atol=1e-12)
    breslow = CoxObjective(tensor(small[X]), risk, None, None, "breslow")
    model = PHReg(small.t, small[X], status=small.d)
    assert_allclose(breslow.residuals(tensor(beta))[0].numpy(), model.score_residuals(beta),
                    atol=1e-12)


def ph_oracle(frame, beta, cov, transform, ties="breslow"):
    """estat phtest formulas with Schoenfeld residuals from an explicit loop."""
    t, d, x = frame.t.to_numpy(), frame.d.to_numpy(), frame[X].to_numpy()
    u = np.exp(x @ beta)
    rows, times = [], []
    for i in np.flatnonzero(d == 1):
        risk = t >= t[i]
        if ties == "breslow":
            mean = u[risk] @ x[risk] / u[risk].sum()
        rows.append(x[i] - mean)
        times.append(t[i])
    r, times = np.array(rows), np.array(times)
    if transform == "identity":
        g = times
    elif transform == "log":
        g = np.log(times)
    elif transform == "rank":
        g = stats.rankdata(times)
    else:
        grid = np.unique(t[d == 1])
        surv = np.cumprod([1 - ((t == s) & (d == 1)).sum() / (t >= s).sum() for s in grid])
        g = 1 - surv[np.searchsorted(grid, times)]
    dd = len(times)
    gc = g - g.mean()
    vu = cov @ (gc @ r)
    per = dd * vu ** 2 / (np.diag(cov) * (gc ** 2).sum())
    glob = (gc @ r) @ vu * dd / (gc ** 2).sum()
    scaled = beta + dd * r @ cov
    rho = [np.corrcoef(scaled[:, p], g)[0, 1] for p in range(len(beta))]
    return per, glob, rho


@pytest.mark.parametrize("transform", ["identity", "log", "km", "rank"])
def test_proportional_hazards_tests(data, transform):
    result = oe.stcox(data=data, time="t", failure="d", x=X, phtest=transform)
    b = est(result)[0]
    cov = np.array(result.covariance_matrix)
    per, glob, rho = ph_oracle(data, b, cov, transform)
    for p, term in enumerate(X):
        assert result.tests[f"ph_{term}"]["statistic"] == pytest.approx(per[p], rel=1e-8)
        assert result.tests[f"ph_{term}"]["rho"] == pytest.approx(rho[p], rel=1e-8)
    assert result.tests["ph_global"]["statistic"] == pytest.approx(glob, rel=1e-8)
    assert result.tests["ph_global"]["df"] == 3


def harrell_oracle(eta, t, d, strata):
    pairs = concordant = ties = 0
    n = len(t)
    for i in range(n):
        for j in range(n):
            if i == j or strata[i] != strata[j] or d[i] != 1:
                continue
            if t[i] < t[j] or (t[i] == t[j] and d[j] == 0):
                pairs += 1
                concordant += eta[i] > eta[j]
                ties += eta[i] == eta[j]
    return (concordant + ties / 2) / pairs, pairs, ties


def test_harrell_concordance_and_baseline(data):
    small = data.iloc[:150].assign(x1=lambda f: np.round(f.x1, 1))
    result = oe.stcox(data=small, time="t", failure="d", x=["x1"], strata="s")
    b = est(result)[0]
    expected, pairs, tied = harrell_oracle(small.x1.to_numpy() * b[0], small.t.to_numpy(),
                                           small.d.to_numpy(), small.s.to_numpy())
    assert result.metrics["concordance"] == pytest.approx(expected, rel=1e-12)
    assert result.extra["concordance"]["pairs"] == pairs
    assert result.extra["concordance"]["tied_predictions"] == tied
    base = result.extra["baseline"]
    t, d, x, s = (small[c].to_numpy() for c in ("t", "d", "x1", "s"))
    for stratum in range(3):
        mask = np.array(base["stratum"]) == stratum         # labels of the strata column
        member = s == stratum
        times = np.unique(t[member & (d == 1)])
        # Breslow: H0(t) = sum_{t_j <= t} d_j / sum_{R_j} exp(x b), right-continuous
        jumps = [((t == u) & (d == 1) & member).sum()
                 / np.exp(b[0] * x[member & (t >= u)]).sum() for u in times]
        assert_allclose(np.array(base["time"])[mask], times)
        assert_allclose(np.array(base["cumulative_hazard"])[mask], np.cumsum(jumps), rtol=1e-10)
        assert_allclose(np.array(base["survivor"])[mask], np.exp(-np.cumsum(jumps)), rtol=1e-10)


def test_tvc_equals_explicit_episode_splitting(data):
    result = oe.stcox(data=data, time="t", failure="d", x=["x1", "x2"], tvc=["x3"],
                      strata="s", texp="log")
    rows = []
    for i, row in data.iterrows():
        fts = np.unique(data.t[(data.d == 1) & (data.s == row.s)])
        previous = 0.0
        for tau in fts[fts <= row.t]:
            rows.append((previous, tau, row.d if tau == row.t else 0.0, row.x1, row.x2,
                         row.x3 * np.log(tau), row.s))
            previous = tau
    split = np.array(rows)
    model = PHReg(split[:, 1], split[:, 3:6], status=split[:, 2], entry=split[:, 0] + 1e-9,
                  strata=split[:, 6]).fit()
    b, se = est(result)
    assert_allclose(b, model.params, rtol=1e-8, atol=1e-10)
    assert_allclose(se, model.bse, rtol=1e-7)
    assert [c.term for c in result.coefficients] == ["x1", "x2", "tvc:x3"]
    assert [c.equation for c in result.coefficients] == ["main", "main", "tvc"]
    robust = oe.stcox(data=data, time="t", failure="d", x=["x1", "x2"], tvc=["x3"],
                      covariance="robust")
    assert robust.metrics["n_subjects"] == len(data)


def test_collinearity_missing_values_and_failures(data):
    frame = data.assign(dup=2 * data.x1, sconst=data.s * 1.5)
    result = oe.stcox(data=frame, time="t", failure="d", x=["x1", "dup", "sconst"], strata="s")
    assert [c.term for c in result.coefficients] == ["x1"]
    assert set(result.provenance["omitted_terms"]) == {"dup", "sconst"}
    missing = data.copy()
    missing.loc[[1, 2], "x1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=missing, time="t", failure="d", x=X)
    assert error.value.code == "missing_values"
    dropped = oe.stcox(data=missing, time="t", failure="d", x=X, missing="drop")
    assert dropped.nobs == len(data) - 2
    for frame, code in ((data.assign(d=data.d * 2), "invalid_failure_indicator"),
                        (data.assign(t=-data.t), "invalid_survival_time"),
                        (data.assign(d=0.0), "no_failures")):
        with pytest.raises(AnalysisError) as error:
            oe.stcox(data=frame, time="t", failure="d", x=X)
        assert error.value.code == code
    separated = data.assign(sep=(data.d == 0).astype(float))
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=separated, time="t", failure="d", x=["sep"])
    assert error.value.code in {"separation_detected", "nonconvergence"}
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=data, time="t", failure="d", x="x1")
    assert error.value.code == "invalid_spec"


def test_stcurve_serialization_and_rendering(data):
    result = oe.stcox(data=data, time="t", failure="d", x=X, ties="efron")
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result and isinstance(result, ResultBundle)
    assert "x1" in result.summary() and "tabular" in result.to_latex()
    full = oe.stcurve(result, data=data, at={"x1": 1.0, "x2": 0.0, "x3": 0.0})
    stored = oe.stcurve(result, at={"x1": 1.0, "x2": 0.0, "x3": 0.0})
    assert_allclose(full.survivor, stored.survivor, rtol=1e-12)
    b = est(result)[0]
    base = oe.stcurve(result, data=data, at={"x1": 0.0, "x2": 0.0, "x3": 0.0})
    assert_allclose(full.cumulative_hazard, base.cumulative_hazard * np.exp(b[0]), rtol=1e-12)
    mean = oe.stcurve(result)
    assert mean.attrs["at"]["x2"] == pytest.approx(data.x2.mean())
    with pytest.raises(AnalysisError) as error:
        oe.stcurve(result, at={"nope": 1.0})
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.stcurve(result, data=data.iloc[:100])
    assert error.value.code == "data_mismatch"
