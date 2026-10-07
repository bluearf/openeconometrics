"""Independent oracles for the linear system estimators (sureg, mvreg, reg3).

Every estimate is compared with explicit stacked-system algebra in NumPy:
Kronecker products (Sigma^-1 kron I) and (Sigma^-1 kron P_Z) on small data,
restricted GLS through a Lagrangian, OLS equation by equation through
statsmodels, and the textbook Breusch-Pagan statistic.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle

EQS = [{"y": "y1", "x": ["x1", "x2"]}, {"y": "y2", "x": ["x1", "x3"]},
       {"y": "y3", "x": ["x2", "x3", "x4"]}]


def make_sur(seed=1, n=150, offset=0.0):
    rng = np.random.default_rng(seed)
    x1, x2, x3, x4 = rng.normal(size=(4, n))
    cov = [[1.0, 0.5, 0.3], [0.5, 2.0, -0.4], [0.3, -0.4, 1.5]]
    e = rng.multivariate_normal(np.zeros(3), cov, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2 + offset, "x3": x3, "x4": x4})
    frame["y1"] = 1 + x1 + 0.5 * x2 + e[:, 0]
    frame["y2"] = -1 + 2 * x1 - x3 + e[:, 1]
    frame["y3"] = 0.5 - x2 + 0.5 * x3 + 0.2 * x4 + e[:, 2]
    frame["w"] = rng.integers(1, 4, size=n).astype(float)
    frame["g"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    return frame


@pytest.fixture(scope="module")
def sur():
    return make_sur()


def designs(frame, eqs):
    xs = [np.column_stack([np.ones(len(frame)), *(frame[c].to_numpy() for c in eq["x"])])
          for eq in eqs]
    ys = [frame[eq["y"]].to_numpy() for eq in eqs]
    return xs, ys


def stacked(xs):
    n, k = len(xs[0]), sum(x.shape[1] for x in xs)
    big = np.zeros((n * len(xs), k))
    col = 0
    for i, x in enumerate(xs):
        big[i * n:(i + 1) * n, col:col + x.shape[1]] = x
        col += x.shape[1]
    return big


def gls(xs, ys, sigma, proj=None, w=None):
    n = len(ys[0])
    a = np.eye(n) if proj is None else proj
    if w is not None:
        a = np.sqrt(w)[:, None] * a * np.sqrt(w)[None, :]
    big, y = stacked(xs), np.concatenate(ys)
    omega = np.kron(np.linalg.inv(sigma), a)
    v = np.linalg.inv(big.T @ omega @ big)
    return v @ big.T @ omega @ y, v


def residuals(xs, ys, beta):
    out, col = [], 0
    for x, y in zip(xs, ys):
        out.append(y - x @ beta[col:col + x.shape[1]])
        col += x.shape[1]
    return np.column_stack(out)


def ols_sigma(xs, ys, divisor, w=None):
    w = np.ones(len(ys[0])) if w is None else w
    betas = [np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
             for x, y in zip(xs, ys)]
    e = residuals(xs, ys, np.concatenate(betas))
    return (e * w[:, None]).T @ e / divisor, np.concatenate(betas)


def est(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


def test_sureg_two_step_matches_kronecker_gls(sur):
    xs, ys = designs(sur, EQS)
    n = len(sur)
    sigma, _ = ols_sigma(xs, ys, n)
    beta, v = gls(xs, ys, sigma)
    result = oe.sureg(data=sur, equations=EQS)
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-10, atol=1e-12)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), v, rtol=1e-8, atol=1e-14)
    assert [c.term for c in result.coefficients][:3] == ["y1:Intercept", "y1:x1", "y1:x2"]
    assert result.coefficients[4].equation == "y2"
    assert result.inference["use_t"] is False
    # Per-equation statistics from the final residuals.
    e = residuals(xs, ys, beta)
    for i, eq in enumerate(EQS):
        rss = e[:, i] @ e[:, i]
        tss = ((ys[i] - ys[i].mean()) ** 2).sum()
        assert_allclose(result.metrics[f"{eq['y']}:rmse"], np.sqrt(rss / n), rtol=1e-10)
        assert_allclose(result.metrics[f"{eq['y']}:r_squared"], 1 - rss / tss, rtol=1e-10)
    # Breusch-Pagan on the OLS residual correlations used by the GLS step.
    corr = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
    lm = n * (corr[1, 0] ** 2 + corr[2, 0] ** 2 + corr[2, 1] ** 2)
    assert_allclose(result.tests["breusch_pagan"]["statistic"], lm, rtol=1e-10)
    assert result.tests["breusch_pagan"]["df"] == 3
    assert_allclose(result.tests["breusch_pagan"]["p_value"], stats.chi2.sf(lm, 3), rtol=1e-8)
    assert_allclose(np.array(result.extra["sigma"]), sigma, rtol=1e-10)
    # Equation Wald test of the slopes.
    idx = [1, 2]
    wald = beta[idx] @ np.linalg.solve(v[np.ix_(idx, idx)], beta[idx])
    assert_allclose(result.tests["y1:model"]["statistic"], wald, rtol=1e-8)
    s_ml = e.T @ e / n
    ll = -n / 2 * (3 * (1 + np.log(2 * np.pi)) + np.log(np.linalg.det(s_ml)))
    assert_allclose(result.metrics["log_likelihood"], ll, rtol=1e-10)


def test_sureg_dfk_small_and_iterated(sur):
    xs, ys = designs(sur, EQS)
    n = len(sur)
    k = np.array([x.shape[1] for x in xs])
    sigma, _ = ols_sigma(xs, ys, np.sqrt(np.outer(n - k, n - k)))
    beta, v = gls(xs, ys, sigma)
    result = oe.sureg(data=sur, equations=EQS, dfk=True, small=True)
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-10)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-9)
    assert result.inference["use_t"] and result.inference["df_inference"] == n - 3
    c = result.coefficients[1]
    assert_allclose(c.p_value, 2 * stats.t.sf(abs(c.statistic), n - 3), rtol=1e-8)
    assert result.tests["y2:model"]["distribution"] == "F"
    assert result.tests["y2:model"]["df2"] == n - 3
    e = residuals(xs, ys, beta)
    assert_allclose(result.metrics["y3:rmse"], np.sqrt(e[:, 2] @ e[:, 2] / (n - 4)), rtol=1e-10)
    # Iterated SUR = Gaussian ML: iterate the NumPy recursion to its fixed point.
    sigma, _ = ols_sigma(xs, ys, n)
    for _ in range(500):
        beta, v = gls(xs, ys, sigma)
        e = residuals(xs, ys, beta)
        new = e.T @ e / n
        if np.abs(new - sigma).max() < 1e-14:
            break
        sigma = new
    result = oe.sureg(data=sur, equations=EQS, iterate=True)
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-7, atol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-6)
    assert result.extra["iterations"] > 2
    assert result.title.endswith("(iterated)")


def test_sureg_ml_maximizes_the_gaussian_likelihood(sur):
    from scipy.optimize import minimize

    xs, ys = designs(sur, EQS[:2])
    n = len(sur)
    result = oe.sureg(data=sur, equations=EQS[:2], iterate=True, tolerance=1e-12)

    def negative(beta):
        e = residuals(xs, ys, beta)
        return n / 2 * np.log(np.linalg.det(e.T @ e / n))

    b, _ = est(result)
    opt = minimize(negative, np.zeros(6), method="BFGS", options={"gtol": 1e-10})
    assert_allclose(b, opt.x, atol=1e-5)
    concentrated = -n / 2 * (2 * (1 + np.log(2 * np.pi))) - opt.fun
    assert_allclose(result.metrics["log_likelihood"], concentrated, rtol=1e-8)


def test_sureg_identical_regressors_equal_ols_and_mvreg(sur):
    eqs = [{"y": "y1", "x": ["x1", "x2"]}, {"y": "y2", "x": ["x1", "x2"]}]
    s = oe.sureg(data=sur, equations=eqs, dfk=True, small=True)
    m = oe.mvreg(data=sur, y=["y1", "y2"], x=["x1", "x2"], corr=True)
    assert_allclose(est(s)[0], est(m)[0], rtol=1e-10)
    assert_allclose(est(s)[1], est(m)[1], rtol=1e-9)
    for name in ("y1", "y2"):
        ols = sm.OLS(sur[name], sm.add_constant(sur[["x1", "x2"]])).fit()
        coefs = {c.term: c for c in m.coefficients}
        assert_allclose([coefs[f"{name}:{t}"].estimate for t in ("Intercept", "x1", "x2")],
                        ols.params.to_numpy(), rtol=1e-10)
        assert_allclose([coefs[f"{name}:{t}"].std_error for t in ("Intercept", "x1", "x2")],
                        ols.bse.to_numpy(), rtol=1e-9)
        assert_allclose(m.tests[f"{name}:model"]["statistic"], ols.fvalue, rtol=1e-8)
        assert_allclose(m.metrics[f"{name}:r_squared"], ols.rsquared, rtol=1e-10)
        assert_allclose(m.metrics[f"{name}:rmse"], np.sqrt(ols.mse_resid), rtol=1e-10)
    assert m.inference["df_inference"] == len(sur) - 3


def test_mvreg_cross_covariance_and_breusch_pagan(sur):
    x = np.column_stack([np.ones(len(sur)), sur[["x1", "x3"]].to_numpy()])
    y = sur[["y1", "y2", "y3"]].to_numpy()
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    n, k = x.shape
    sigma = e.T @ e / (n - k)
    v = np.kron(sigma, np.linalg.inv(x.T @ x))
    result = oe.mvreg(data=sur, y=["y1", "y2", "y3"], x=["x1", "x3"], corr=True)
    assert_allclose(est(result)[0], b.T.reshape(-1), rtol=1e-10)
    assert_allclose(np.array(result.covariance_matrix), v, rtol=1e-8, atol=1e-15)
    corr = np.corrcoef(e.T)
    lm = n * (corr[1, 0] ** 2 + corr[2, 0] ** 2 + corr[2, 1] ** 2)
    assert_allclose(result.tests["breusch_pagan"]["statistic"], lm, rtol=1e-9)
    assert_allclose(np.array(result.extra["correlation"]), corr, rtol=1e-9)
    plain = oe.mvreg(data=sur, y=["y1", "y2"], x=["x1"])
    assert "breusch_pagan" not in plain.tests


def test_sureg_constraints_match_restricted_gls(sur):
    xs, ys = designs(sur, EQS[:2])
    n = len(sur)
    sigma, _ = ols_sigma(xs, ys, n)
    big, y = stacked(xs), np.concatenate(ys)
    omega = np.kron(np.linalg.inv(sigma), np.eye(n))
    a = big.T @ omega @ big
    r = np.zeros((1, 6))
    r[0, 1], r[0, 4] = 1, -1                      # y1:x1 = y2:x1
    a_inv = np.linalg.inv(a)
    b_u = a_inv @ big.T @ omega @ y
    adj = a_inv @ r.T @ np.linalg.solve(r @ a_inv @ r.T, r @ b_u)
    b_r = b_u - adj
    v_r = a_inv - a_inv @ r.T @ np.linalg.solve(r @ a_inv @ r.T, r @ a_inv)
    result = oe.sureg(data=sur, equations=EQS[:2],
                      constraints=[{"terms": {"y1:x1": 1, "y2:x1": -1}, "value": 0}])
    b, se = est(result)
    assert_allclose(b, b_r, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v_r)), rtol=1e-8)
    assert_allclose(b[1], b[4], rtol=1e-12)
    assert result.extra["constraints"]["count"] == 1
    # A constraint fixing a coefficient moves it to extra['constrained_terms'].
    fixed = oe.sureg(data=sur, equations=EQS[:2],
                     constraints=[{"terms": {"y2:x3": 1}, "value": -1}])
    assert "y2:x3" not in [c.term for c in fixed.coefficients]
    assert fixed.extra["constrained_terms"] == {"y2:x3": -1.0}
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=EQS[:2], constraints=[{"terms": {"y9:x": 1}, "value": 0}])
    assert error.value.code == "invalid_constraint"


def test_sureg_weights(sur):
    xs, ys = designs(sur, EQS[:2])
    w = sur.w.to_numpy()
    wa = w * len(w) / w.sum()
    sigma, _ = ols_sigma(xs, ys, len(w), wa)
    beta, v = gls(xs, ys, sigma, w=wa)
    result = oe.sureg(data=sur, equations=EQS[:2], weights="w", weight_type="aweight")
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-8)
    expanded = sur.loc[sur.index.repeat(sur.w.astype(int))].reset_index(drop=True)
    f = oe.sureg(data=sur, equations=EQS[:2], weights="w", weight_type="fweight")
    d = oe.sureg(data=expanded, equations=EQS[:2])
    assert f.nobs == len(expanded)
    assert_allclose(est(f)[0], est(d)[0], rtol=1e-10)
    assert_allclose(est(f)[1], est(d)[1], rtol=1e-9)
    assert_allclose(f.tests["breusch_pagan"]["statistic"],
                    d.tests["breusch_pagan"]["statistic"], rtol=1e-9)


def test_large_offsets_keep_precision():
    frame = make_sur(seed=4, offset=1e7)
    eqs = EQS[:2]
    result = oe.sureg(data=frame, equations=eqs)
    shifted = frame.assign(x2=frame.x2 - 1e7)
    reference = oe.sureg(data=shifted, equations=eqs)
    b, se = est(result)
    b0, se0 = est(reference)
    slopes = [1, 2, 4, 5]
    assert_allclose(b[slopes], b0[slopes], rtol=1e-8)
    assert_allclose(se[slopes], se0[slopes], rtol=1e-8)
    assert_allclose(b[0], b0[0] - 1e7 * b[2], rtol=1e-7)


def test_collinearity_categorical_and_missing(sur):
    frame = sur.assign(dup=2 * sur.x1)
    eqs = [{"y": "y1", "x": ["x1", "dup", "g"]}, {"y": "y2", "x": ["x3"], "constant": False}]
    result = oe.sureg(data=frame, equations=eqs, categorical=["g"])
    terms = [c.term for c in result.coefficients]
    assert "y1:dup" not in terms and "y1:g[b]" in terms and "y2:Intercept" not in terms
    assert result.provenance["omitted_terms"] == ["y1:dup"]
    missing = sur.copy()
    missing.loc[3, "x3"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=missing, equations=EQS[:2])
    assert error.value.code == "missing_values"
    dropped = oe.sureg(data=missing, equations=EQS[:2], missing="drop")
    assert dropped.nobs == len(sur) - 1 and 3 not in dropped.sample_positions


def test_reg3_matches_kronecker_three_stage():
    rng = np.random.default_rng(5)
    n = 250
    z1, z2, z3, z4 = rng.normal(size=(4, n))
    e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1.5]], size=n)
    a = np.array([[1, -0.5], [0.4, 1]])
    rhs = np.column_stack([1 + z1 + 0.5 * z4 + e[:, 0], 2 + z2 + z3 + e[:, 1]])
    y = np.linalg.solve(a, rhs.T).T
    frame = pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1], "z1": z1, "z2": z2, "z3": z3, "z4": z4})
    eqs = [{"y": "y1", "x": ["y2", "z1"]}, {"y": "y2", "x": ["y1", "z2"]}]
    zmat = np.column_stack([np.ones(n), z1, z2, z3, z4])
    proj = zmat @ np.linalg.solve(zmat.T @ zmat, zmat.T)
    xs = [np.column_stack([np.ones(n), frame.y2, z1]), np.column_stack([np.ones(n), frame.y1, z2])]
    ys = [frame.y1.to_numpy(), frame.y2.to_numpy()]
    first = [np.linalg.solve((proj @ x).T @ x, (proj @ x).T @ yy) for x, yy in zip(xs, ys)]
    e2 = residuals(xs, ys, np.concatenate(first))
    sigma = e2.T @ e2 / n
    beta, v = gls(xs, ys, sigma, proj)
    result = oe.reg3(data=frame, equations=eqs, exogenous=["z3", "z4"])
    b, se = est(result)
    assert_allclose(b, beta, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-8)
    assert result.extra["endogenous"] == ["y2", "y1"]
    # instruments=... gives the same exogenous set explicitly.
    same = oe.reg3(data=frame, equations=eqs, instruments=["z1", "z2", "z3", "z4"])
    assert_allclose(est(same)[0], beta, rtol=1e-9)
    # Iterated 3SLS: fixed point of the NumPy recursion.
    for _ in range(500):
        beta_i, v_i = gls(xs, ys, sigma, proj)
        e3 = residuals(xs, ys, beta_i)
        new = e3.T @ e3 / n
        if np.abs(new - sigma).max() < 1e-14:
            break
        sigma = new
    iterated = oe.reg3(data=frame, equations=eqs, exogenous=["z3", "z4"], ireg3=True)
    assert_allclose(est(iterated)[0], beta_i, rtol=1e-7)
    assert_allclose(est(iterated)[1], np.sqrt(np.diag(v_i)), rtol=1e-6)
    # 2SLS (implies dfk, small, independent equations) equals ivregress 2sls small.
    tsls = oe.reg3(data=frame, equations=eqs, exogenous=["z3", "z4"], method="2sls")
    iv = oe.ivregress(data=frame, y="y1", x=["z1"], endog=["y2"], instruments=["z2", "z3", "z4"],
                      small=True)
    coefs = {c.term: c for c in tsls.coefficients}
    ivc = {c.term: c for c in iv.coefficients}
    for term in ("Intercept", "y2", "z1"):
        assert_allclose(coefs[f"y1:{term}"].estimate, ivc[term].estimate, rtol=1e-9)
        assert_allclose(coefs[f"y1:{term}"].std_error, ivc[term].std_error, rtol=1e-8)
    assert tsls.inference["use_t"] and tsls.extra["dfk"]
    cov = np.array(tsls.covariance_matrix)
    assert np.all(cov[:3, 3:] == 0)
    # ols: equation-by-equation OLS with N-k variances; sure equals sureg.
    ols = oe.reg3(data=frame, equations=eqs, method="ols")
    fit = sm.OLS(frame.y1, sm.add_constant(frame[["y2", "z1"]])).fit()
    assert_allclose(est(ols)[0][:3], fit.params.to_numpy(), rtol=1e-10)
    assert_allclose(est(ols)[1][:3], fit.bse.to_numpy(), rtol=1e-9)
    sure = oe.reg3(data=frame, equations=eqs, method="sure")
    assert_allclose(est(sure)[0], est(oe.sureg(data=frame, equations=eqs))[0], rtol=1e-10)


def test_reg3_errors():
    frame = make_sur(seed=6)
    eqs = [{"y": "y1", "x": ["y2", "x1"]}, {"y": "y2", "x": ["y1", "x1"]}]
    with pytest.raises(AnalysisError) as error:
        oe.reg3(data=frame, equations=eqs)
    assert error.value.code == "underidentified"
    with pytest.raises(AnalysisError) as error:
        oe.reg3(data=frame, equations=eqs, instruments=["x1"], endogenous=["x2"])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.reg3(data=frame, equations=eqs, exogenous=["y1"])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.reg3(data=frame, equations=eqs, method="2sls", ireg3=True, exogenous=["x2", "x3"])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.reg3(data=frame, equations=eqs, method="liml")
    assert error.value.code == "invalid_spec"


def test_failure_contract(sur):
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=[{"y": "y1", "x": ["x1"]}])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=[{"y": "y1", "x": "x1"}, {"y": "y2", "x": ["x1"]}])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=[{"y": "y1", "x": ["x1"]}, {"y": "y1", "x": ["x2"]}])
    assert error.value.code == "invalid_spec"
    twin = sur.assign(y4=sur.y1)
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=twin, equations=[{"y": "y1", "x": ["x1"]}, {"y": "y4", "x": ["x1"]}])
    assert error.value.code == "singular_sigma"
    exact = sur.assign(y5=1 + 2 * sur.x1)
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=exact, equations=[{"y": "y1", "x": ["x1"]}, {"y": "y5", "x": ["x1"]}])
    assert error.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=EQS, covariance="robust")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur.head(3), equations=EQS)
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=EQS, iterate=True, max_iterations=1)
    assert error.value.code == "nonconvergence"
    with pytest.raises(AnalysisError) as error:
        oe.mvreg(data=sur, y=["y1"], x=["x1"])
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.sureg(data=sur, equations=[{"y": "y1", "x": ["nope"]}, {"y": "y2", "x": ["x1"]}])
    assert error.value.code == "missing_columns"


def test_round_trip_rendering_and_registry(sur):
    result = oe.sureg(data=sur, equations=EQS, small=True)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert isinstance(ResultBundle.model_validate(json.loads(result.model_dump_json())),
                      ResultBundle)
    text = result.summary()
    assert "[y2]" in text and "Breusch-Pagan" in text
    assert "y1:x1" in result.to_latex() or "x1" in result.to_latex()
    assert result.spec.columns["system"] == ["y1", "x1", "x2", "y2", "x3", "y3", "x4"]
    assert result.spec.outcome == "y1"
    caps = oe.capabilities()
    names = caps if isinstance(caps, dict) else {}
    assert "sureg" in str(names) and "reg3" in str(names)
    for name in ("sureg", "mvreg", "reg3"):
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
    spec = result.spec.model_copy()
    refit = oe.fit(spec, data=sur)
    assert_allclose(est(refit)[0], est(result)[0], rtol=1e-12)


def _simultaneous(seed=7, n=220):
    rng = np.random.default_rng(seed)
    z1, z2, z3 = rng.normal(size=(3, n))
    e = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1.2]], size=n)
    a = np.array([[1, -0.4], [0.3, 1]])
    rhs = np.column_stack([1 + z1 + e[:, 0], -1 + z2 + 0.5 * z3 + e[:, 1]])
    y = np.linalg.solve(a, rhs.T).T
    frame = pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1], "z1": z1, "z2": z2, "z3": z3,
                          "w": rng.integers(1, 4, n).astype(float)})
    return frame, [{"y": "y1", "x": ["y2", "z1"]}, {"y": "y2", "x": ["y1", "z2", "z3"]}]


def test_reg3_weights_dfk_and_constraints():
    frame, eqs = _simultaneous()
    n = len(frame)
    w = frame.w.to_numpy() * n / frame.w.sum()
    zmat = np.column_stack([np.ones(n), frame.z1, frame.z2, frame.z3])
    zw = zmat * w[:, None]
    proj = zmat @ np.linalg.solve(zw.T @ zmat, zw.T)          # weighted projection P_Z
    xs = [np.column_stack([np.ones(n), frame.y2, frame.z1]),
          np.column_stack([np.ones(n), frame.y1, frame.z2, frame.z3])]
    ys = [frame.y1.to_numpy(), frame.y2.to_numpy()]

    def tsls(x, y):
        xh = proj @ x
        return np.linalg.solve((xh * w[:, None]).T @ x, (xh * w[:, None]).T @ y)

    e = residuals(xs, ys, np.concatenate([tsls(x, y) for x, y in zip(xs, ys)]))
    k = np.array([3, 4])
    sigma = (e * w[:, None]).T @ e / np.sqrt(np.outer(n - k, n - k))
    # GLS with Sigma^-1 kron (W^1/2 P_Z W^1/2) in the weighted metric.
    big, y = stacked(xs), np.concatenate(ys)
    middle = np.diag(w) @ proj
    omega = np.kron(np.linalg.inv(sigma), (middle + middle.T) / 2)
    v = np.linalg.inv(big.T @ omega @ big)
    beta = v @ big.T @ omega @ y
    result = oe.reg3(data=frame, equations=eqs, weights="w", weight_type="aweight", dfk=True)
    assert_allclose(est(result)[0], beta, rtol=1e-8)
    assert_allclose(est(result)[1], np.sqrt(np.diag(v)), rtol=1e-7)
    expanded = frame.loc[frame.index.repeat(frame.w.astype(int))].reset_index(drop=True)
    f = oe.reg3(data=frame, equations=eqs, weights="w", weight_type="fweight", small=True)
    d = oe.reg3(data=expanded, equations=eqs, small=True)
    assert_allclose(est(f)[0], est(d)[0], rtol=1e-9)
    assert_allclose(est(f)[1], est(d)[1], rtol=1e-8)
    assert f.inference["df_inference"] == len(expanded) - 3
    # A cross-equation constraint under 3SLS: restricted GLS by the Lagrangian.
    plain = oe.reg3(data=frame, equations=eqs)
    e = residuals(xs, ys, np.concatenate([np.linalg.solve((zmat @ np.linalg.solve(
        zmat.T @ zmat, zmat.T) @ x).T @ x, (zmat @ np.linalg.solve(zmat.T @ zmat, zmat.T) @ x).T
        @ yy) for x, yy in zip(xs, ys)]))
    sigma = e.T @ e / n
    p_z = zmat @ np.linalg.solve(zmat.T @ zmat, zmat.T)
    omega = np.kron(np.linalg.inv(sigma), p_z)
    a_inv = np.linalg.inv(big.T @ omega @ big)
    b_u = a_inv @ big.T @ omega @ y
    assert_allclose(est(plain)[0], b_u, rtol=1e-8)
    r = np.zeros((1, 7))
    r[0, 2], r[0, 5] = 1, 1                                   # y1:z1 + y2:z2 = 2
    b_r = b_u - a_inv @ r.T @ np.linalg.solve(r @ a_inv @ r.T, r @ b_u - 2)
    v_r = a_inv - a_inv @ r.T @ np.linalg.solve(r @ a_inv @ r.T, r @ a_inv)
    constrained = oe.reg3(data=frame, equations=eqs,
                          constraints=[{"terms": {"y1:z1": 1, "y2:z2": 1}, "value": 2}])
    assert_allclose(est(constrained)[0], b_r, rtol=1e-8)
    assert_allclose(est(constrained)[1], np.sqrt(np.diag(v_r)), rtol=1e-7)


def test_mvreg_frequency_weights_and_categorical(sur):
    expanded = sur.loc[sur.index.repeat(sur.w.astype(int))].reset_index(drop=True)
    f = oe.mvreg(data=sur, y=["y1", "y2"], x=["x1", "g"], categorical=["g"], weights="w",
                 weight_type="fweight")
    d = oe.mvreg(data=expanded, y=["y1", "y2"], x=["x1", "g"], categorical=["g"])
    assert_allclose(est(f)[0], est(d)[0], rtol=1e-10)
    assert_allclose(est(f)[1], est(d)[1], rtol=1e-9)
    assert "y2:g[c]" in [c.term for c in f.coefficients]
