"""Independent oracles for the linear mixed model (oe.mixed, oe.mixed_predict).

Log likelihoods, fixed effects, variance components and BLUPs are compared
with statsmodels MixedLM (ML and REML, random intercepts, unstructured random
slopes, two nested levels as variance components), every covariance structure
with a brute-force SciPy maximization of a likelihood written with explicit
dense V matrices, the variance-parameter standard errors with a numerical
Hessian of that dense likelihood, and the robust covariance with explicit
NumPy sandwich algebra.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixed.lmm_kernels import CovStructure, MixedLikelihood
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle


def make_data(seed=0, groups=40, size=6, slope_sd=0.6):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), size)
    n = groups * size
    frame = pd.DataFrame({"g": g, "x": rng.normal(size=n), "w": rng.normal(size=n) + 3.0})
    u0 = rng.normal(size=groups)
    u1 = slope_sd * rng.normal(size=groups)
    frame["y"] = (1 + 0.5 * frame.x - 0.3 * frame.w + u0[g] + u1[g] * frame.x
                  + rng.normal(size=n))
    frame["f"] = rng.integers(1, 3, size=n).astype(float)
    frame["cl"] = g // 4
    frame["cat"] = pd.Categorical(np.where(frame.x > 0.3, "b", np.where(frame.x < -0.3, "c", "a")))
    return frame


def make_nested(seed=3, schools=15, classes=4, size=5):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(schools), classes * size)
    k = np.tile(np.repeat(np.arange(classes), size), schools)
    n = schools * classes * size
    frame = pd.DataFrame({"s": s, "k": k, "x": rng.normal(size=n)})
    frame["y"] = (2 + frame.x + rng.normal(size=schools)[s]
                  + 0.7 * rng.normal(size=schools * classes)[s * classes + k] + rng.normal(size=n))
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def table(result):
    return {c.term: c for c in result.coefficients}


def dense_loglik(frame, y, xcols, zcols, groups, G, sigma2, reml=False, top=None, g_top=0.0):
    """Explicit-V (restricted) log likelihood with GLS-profiled fixed effects."""
    X = np.column_stack([np.ones(len(frame)), frame[xcols].to_numpy()])
    Z = np.column_stack([frame[c].to_numpy() if c != "_cons" else np.ones(len(frame))
                         for c in zcols])
    Y = frame[y].to_numpy()
    codes = frame[groups].to_numpy()
    V = sigma2 * np.eye(len(Y)) + (codes[:, None] == codes[None, :]) * (Z @ G @ Z.T)
    if top is not None:
        t = frame[top].to_numpy()
        V += g_top * (t[:, None] == t[None, :])
    Vi = np.linalg.inv(V)
    xvx = X.T @ Vi @ X
    b = np.linalg.solve(xvx, X.T @ Vi @ Y)
    r = Y - X @ b
    ll = -0.5 * (len(Y) * math.log(2 * math.pi) + np.linalg.slogdet(V)[1] + r @ Vi @ r)
    if reml:
        ll += -0.5 * np.linalg.slogdet(xvx)[1] + 0.5 * X.shape[1] * math.log(2 * math.pi)
    return ll, b, xvx


# ---- statsmodels MixedLM ---------------------------------------------------------------


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_random_intercept_matches_statsmodels(data, method):
    result = oe.mixed(data=data, y="y", x=["x", "w"], group="g", method=method)
    ref = smf.mixedlm("y ~ x + w", data, groups=data["g"]).fit(reml=method == "reml")
    terms = table(result)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-9)
    assert_allclose([terms[t].estimate for t in ("Intercept", "x", "w")], ref.fe_params,
                    rtol=1e-5)
    assert_allclose([terms[t].std_error for t in ("Intercept", "x", "w")], ref.bse_fe,
                    rtol=1e-4)
    assert_allclose(terms["/var(_cons[g])"].estimate, ref.cov_re.iloc[0, 0], rtol=1e-3)
    assert_allclose(terms["/var(Residual)"].estimate, ref.scale, rtol=1e-5)
    assert result.title == f"Mixed-effects {method.upper()} regression"
    var_u, var_e = terms["/var(_cons[g])"].estimate, terms["/var(Residual)"].estimate
    assert_allclose(result.metrics["icc"], var_u / (var_u + var_e), rtol=1e-12)
    assert result.metrics["n_groups"] == 40 and result.metrics["group_size_avg"] == 6


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_unstructured_slopes_match_statsmodels(data, method):
    result = oe.mixed(data=data, y="y", x=["x", "w"], group="g", random=["x"],
                      covstructure="unstructured", method=method)
    ref = smf.mixedlm("y ~ x + w", data, groups=data["g"], re_formula="~x").fit(
        reml=method == "reml")
    terms = table(result)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-9)
    assert_allclose([terms[t].estimate for t in ("Intercept", "x", "w")], ref.fe_params,
                    rtol=1e-5)
    cov = ref.cov_re
    assert_allclose(terms["/var(x[g])"].estimate, cov.loc["x", "x"], rtol=1e-4)
    assert_allclose(terms["/var(_cons[g])"].estimate, cov.loc["Group", "Group"], rtol=1e-4)
    assert_allclose(terms["/cov(x,_cons[g])"].estimate, cov.loc["x", "Group"], rtol=1e-3)
    assert [c.term for c in result.coefficients][3:] == [
        "/var(x[g])", "/var(_cons[g])", "/cov(x,_cons[g])", "/var(Residual)"]
    assert result.tests["lr_vs_linear"]["df"] == 3
    assert result.tests["lr_vs_linear"]["distribution"] == "chi2"


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_two_nested_levels_match_statsmodels(method):
    frame = make_nested()
    result = oe.mixed(data=frame, y="y", x=["x"], group=["s", "k"], method=method)
    ref = smf.mixedlm("y ~ x", frame, groups=frame["s"], re_formula="1",
                      vc_formula={"k": "0 + C(k)"}).fit(reml=method == "reml")
    terms = table(result)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-9)
    assert_allclose(terms["x"].estimate, ref.fe_params["x"], rtol=1e-6)
    assert_allclose(terms["/var(_cons[s])"].estimate, ref.cov_re.iloc[0, 0], rtol=1e-3)
    assert_allclose(terms["/var(_cons[k])"].estimate, ref.vcomp[0], rtol=1e-3)
    assert_allclose(terms["/var(Residual)"].estimate, ref.scale, rtol=1e-5)
    levels = result.extra["levels"]
    assert [level["n_groups"] for level in levels] == [15, 60]
    total = sum(terms[t].estimate for t in ("/var(_cons[s])", "/var(_cons[k])",
                                            "/var(Residual)"))
    assert_allclose(result.extra["icc"]["k|s"],
                    (terms["/var(_cons[s])"].estimate + terms["/var(_cons[k])"].estimate) / total)


def test_lower_level_labels_are_nested():
    """Class labels repeat across schools: classes are interpreted within schools."""
    frame = make_nested()
    frame["kk"] = frame.s.astype(str) + "-" + frame.k.astype(str)
    a = oe.mixed(data=frame, y="y", x=["x"], group=["s", "k"])
    b = oe.mixed(data=frame, y="y", x=["x"], group=["s", "kk"])
    assert_allclose(a.metrics["log_likelihood"], b.metrics["log_likelihood"], rtol=1e-12)


# ---- every covariance structure against a dense brute-force likelihood --------------------


def _structure_matrix(kind, t):
    if kind == "independent":
        return np.diag(np.exp(2 * t))
    if kind == "identity":
        return np.exp(2 * t[0]) * np.eye(2)
    if kind == "exchangeable":
        v, rho = np.exp(2 * t[0]), np.tanh(t[1])
        return v * np.array([[1, rho], [rho, 1]])
    lower = np.array([[np.exp(t[0]), 0], [t[2], np.exp(t[1])]])
    return lower @ lower.T


@pytest.mark.parametrize("kind,size", [("independent", 2), ("identity", 1),
                                       ("exchangeable", 2), ("unstructured", 3)])
@pytest.mark.parametrize("method", ["ml", "reml"])
def test_structures_match_dense_brute_force(kind, size, method):
    frame = make_data(seed=5, groups=25, size=5)
    reml = method == "reml"

    def negative(params):
        G = _structure_matrix(kind, params[:size])
        return -dense_loglik(frame, "y", ["x"], ["x", "_cons"], "g", G,
                             math.exp(2 * params[-1]), reml)[0]

    best = minimize(negative, np.zeros(size + 1), method="BFGS", options={"gtol": 1e-9})
    result = oe.mixed(data=frame, y="y", x=["x"], group="g", random=["x"], covstructure=kind,
                      method=method)
    assert_allclose(result.metrics["log_likelihood"], -best.fun, rtol=1e-8)
    G = _structure_matrix(kind, best.x[:size])
    terms = table(result)
    if kind in {"independent", "unstructured"}:
        assert_allclose(terms["/var(x[g])"].estimate, G[0, 0], rtol=2e-4)
        assert_allclose(terms["/var(_cons[g])"].estimate, G[1, 1], rtol=2e-4)
    else:
        assert_allclose(terms["/var(x _cons[g])"].estimate, G[0, 0], rtol=2e-4)
    if kind == "exchangeable":
        assert_allclose(terms["/cov(x,_cons[g])"].estimate, G[0, 1], rtol=2e-3, atol=1e-5)
    assert_allclose(result.extra["G"], G, rtol=2e-3, atol=1e-5)


def test_variance_standard_errors_from_dense_hessian():
    """Delta-method SEs from the observed information of the dense profiled likelihood."""
    frame = make_data(seed=11, groups=30, size=6)
    result = oe.mixed(data=frame, y="y", x=["x", "w"], group="g", random=["x"])
    terms = table(result)
    t = np.array([math.log(terms[name].estimate) / 2 for name in
                  ("/var(x[g])", "/var(_cons[g])", "/var(Residual)")])

    def loglik(params):
        return dense_loglik(frame, "y", ["x", "w"], ["x", "_cons"], "g",
                            np.diag(np.exp(2 * params[:2])), math.exp(2 * params[2]))[0]

    cov = np.linalg.inv(-approx_hess(t, loglik))
    jac = np.diag(2 * np.exp(2 * t))
    se = np.sqrt(np.diag(jac @ cov @ jac.T))
    got = [terms[name].std_error for name in ("/var(x[g])", "/var(_cons[g])", "/var(Residual)")]
    assert_allclose(got, se, rtol=2e-4)
    # Variance intervals are exp(ln v -/+ z se/v), covariances keep the Wald interval.
    term = terms["/var(_cons[g])"]
    spread = 1.959963984540054 * term.std_error / term.estimate
    assert_allclose([term.ci_low, term.ci_high],
                    [term.estimate * math.exp(-spread), term.estimate * math.exp(spread)])
    # Fixed effects: (X'V^-1 X)^-1 at the estimates.
    G = np.diag([terms["/var(x[g])"].estimate, terms["/var(_cons[g])"].estimate])
    _, b, xvx = dense_loglik(frame, "y", ["x", "w"], ["x", "_cons"], "g", G,
                             terms["/var(Residual)"].estimate)
    assert_allclose([terms[n].estimate for n in ("Intercept", "x", "w")], b, rtol=1e-9)
    assert_allclose([terms[n].std_error for n in ("Intercept", "x", "w")],
                    np.sqrt(np.diag(np.linalg.inv(xvx))), rtol=1e-9)


def test_lr_test_against_linear_regression(data):
    ml = oe.mixed(data=data, y="y", x=["x", "w"], group="g")
    ols = sm.OLS(data.y, sm.add_constant(data[["x", "w"]])).fit()
    lr = ml.tests["lr_vs_linear"]
    assert_allclose(lr["statistic"], 2 * (ml.metrics["log_likelihood"] - ols.llf), rtol=1e-9)
    assert lr["distribution"] == "chibar2" and lr["df"] == 1
    from scipy import stats

    assert_allclose(lr["p_value"], 0.5 * stats.chi2.sf(lr["statistic"], 1), rtol=1e-6)
    reml = oe.mixed(data=data, y="y", x=["x", "w"], group="g", method="reml")
    X = sm.add_constant(data[["x", "w"]]).to_numpy()
    n, p = X.shape
    ssr = ols.ssr
    restricted = (-(n - p) / 2 * (math.log(2 * math.pi) + math.log(ssr / (n - p)) + 1)
                  - 0.5 * np.linalg.slogdet(X.T @ X)[1])
    assert_allclose(reml.extra["linear_log_likelihood"], restricted, rtol=1e-10)
    wald = ml.tests["model"]
    terms = table(ml)
    v = np.array(ml.covariance_matrix)[1:3, 1:3]
    b = np.array([terms["x"].estimate, terms["w"].estimate])
    assert_allclose(wald["statistic"], b @ np.linalg.solve(v, b), rtol=1e-9)


def test_frequency_weights_equal_duplicated_rows(data):
    weighted = oe.mixed(data=data, y="y", x=["x"], group="g", random=["x"], weights="f",
                        weight_type="fweight")
    expanded = data.loc[data.index.repeat(data.f.astype(int))].reset_index(drop=True)
    plain = oe.mixed(data=expanded, y="y", x=["x"], group="g", random=["x"])
    assert weighted.nobs == plain.nobs == int(data.f.sum())
    assert_allclose([c.estimate for c in weighted.coefficients],
                    [c.estimate for c in plain.coefficients], rtol=1e-6)
    assert_allclose([c.std_error for c in weighted.coefficients],
                    [c.std_error for c in plain.coefficients], rtol=1e-5)
    assert_allclose(weighted.metrics["log_likelihood"], plain.metrics["log_likelihood"],
                    rtol=1e-10)


def test_robust_and_cluster_sandwich(data):
    result = oe.mixed(data=data, y="y", x=["x", "w"], group="g", covariance="robust")
    base = oe.mixed(data=data, y="y", x=["x", "w"], group="g")
    terms = table(result)
    G = np.array([[terms["/var(_cons[g])"].estimate]])
    s2 = terms["/var(Residual)"].estimate
    X = np.column_stack([np.ones(len(data)), data[["x", "w"]].to_numpy()])
    Y = data.y.to_numpy()
    b = np.array([terms[n].estimate for n in ("Intercept", "x", "w")])
    bread = np.zeros((3, 3))
    scores = []
    for _, block in data.groupby("g"):
        idx = block.index.to_numpy()
        Vg = s2 * np.eye(len(idx)) + G[0, 0]
        Vi = np.linalg.inv(Vg)
        bread += X[idx].T @ Vi @ X[idx]
        scores.append(X[idx].T @ Vi @ (Y[idx] - X[idx] @ b))
    scores = np.array(scores)
    bread = np.linalg.inv(bread)
    groups = len(scores)
    expected = groups / (groups - 1) * bread @ scores.T @ scores @ bread
    assert_allclose(np.array(result.covariance_matrix)[:3, :3], expected, rtol=1e-8)
    # Stata's mixed makes the variance parameters robust too (verified against numerical
    # group scores of a dense likelihood in test_econ_mixed_oracle.py) and prints no LR test.
    assert not np.allclose(np.array(result.covariance_matrix)[3:, 3:],
                           np.array(base.covariance_matrix)[3:, 3:])
    assert "lr_vs_linear" not in result.tests and "lr_vs_linear" in base.tests
    clustered = oe.mixed(data=data, y="y", x=["x", "w"], group="g", cluster="cl")
    assert clustered.inference["cluster_count"] == 10
    assert clustered.spec.covariance == "cluster"
    with pytest.raises(AnalysisError) as error:
        oe.mixed(data=data.assign(bad=np.arange(len(data)) % 7), y="y", x=["x"], group="g",
                 cluster="bad")
    assert error.value.code == "cluster_not_nested"


# ---- derivatives, collinearity, missing data, errors ----------------------------------------


@pytest.mark.parametrize("kind", ["independent", "unstructured", "exchangeable", "identity"])
@pytest.mark.parametrize("reml", [False, True])
def test_analytic_gradient_matches_numerical(kind, reml):
    rng = np.random.default_rng(1)
    frame = make_nested(seed=4, schools=6, classes=3, size=4)
    x = torch.tensor(np.column_stack([np.ones(len(frame)), frame.x]), dtype=torch.float64)
    z = torch.tensor(np.column_stack([frame.x, np.ones(len(frame))]), dtype=torch.float64)
    low = torch.tensor((frame.s * 3 + frame.k).to_numpy())
    structure = CovStructure(kind, 2)
    like = MixedLikelihood(x, torch.tensor(frame.y.to_numpy()), z, low, 18, structure,
                           top_of_low=torch.arange(18) // 3, n_top=6,
                           frequency=torch.tensor(rng.integers(1, 3, len(frame)) * 1.0),
                           reml=reml)
    theta = torch.tensor(np.r_[rng.normal(size=structure.size) * 0.3, -0.2, 0.1])
    report = check_derivatives(like, theta)
    assert report["gradient_max_rel_error"] < 1e-7


def test_collinear_regressor_is_omitted(data):
    frame = data.assign(x2=2 * data.x)
    result = oe.mixed(data=frame, y="y", x=["x", "x2", "w"], group="g")
    assert "x2" not in table(result)
    assert result.provenance["omitted_terms"] == ["x2"]
    assert any("collinearity" in warning for warning in result.warnings)


def test_categorical_regressor(data):
    result = oe.mixed(data=data, y="y", x=["x", "cat"], categorical=["cat"], group="g")
    assert {"cat[b]", "cat[c]"} <= set(table(result))


def test_missing_policy(data):
    frame = data.copy()
    frame.loc[[3, 17], "x"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.mixed(data=frame, y="y", x=["x"], group="g")
    assert error.value.code == "missing_values"
    dropped = oe.mixed(data=frame, y="y", x=["x"], group="g", missing="drop")
    clean = oe.mixed(data=frame.dropna().reset_index(drop=True), y="y", x=["x"], group="g")
    assert dropped.nobs == len(frame) - 2 and dropped.dropped_rows == 2
    assert_allclose(dropped.metrics["log_likelihood"], clean.metrics["log_likelihood"])


def test_error_contract(data):
    def code(**kwargs):
        options = {"data": data, "y": "y", "x": ["x"], "group": "g"}
        options.update(kwargs)
        with pytest.raises(AnalysisError) as error:
            oe.mixed(**options)
        return error.value.code

    assert code(random_intercept=False) == "invalid_spec"
    assert code(method="reml", covariance="robust") == "unsupported_covariance"
    assert code(covstructure="banded") == "invalid_spec"
    assert code(group=["g", "g", "x"]) == "invalid_spec"
    assert code(x=["g"]) == "invalid_spec"
    assert code(weights="f", weight_type="pweight") == "invalid_spec"
    single = data.assign(id=np.arange(len(data)))
    assert code(data=single, group="id") == "insufficient_group_size"
    rng = np.random.default_rng(2)
    flat = data.assign(y=data.x + rng.normal(size=len(data)))
    assert code(data=flat) == "boundary_solution"
    assert code(data=data.assign(one=1), group="one") == "insufficient_groups"


def test_predictions_and_blups(data):
    result = oe.mixed(data=data, y="y", x=["x", "w"], group="g", random=["x"],
                      covstructure="unstructured", method="reml")
    ref = smf.mixedlm("y ~ x + w", data, groups=data["g"], re_formula="~x").fit(reml=True)
    blups = oe.mixed_predict(result, data, kind="reffects")
    assert list(blups.columns) == ["level", "group", "effect", "blup"]
    cons = blups[blups.effect == "_cons"].set_index("group").blup
    slope = blups[blups.effect == "x"].set_index("group").blup
    effects = pd.DataFrame(ref.random_effects).T
    assert_allclose(cons.loc[effects.index].to_numpy(), effects["Group"], rtol=1e-3, atol=1e-5)
    assert_allclose(slope.loc[effects.index].to_numpy(), effects["x"], rtol=1e-3, atol=1e-5)
    fitted = oe.mixed_predict(result, data, kind="fitted")
    assert_allclose(fitted.fitted.to_numpy(), ref.fittedvalues.to_numpy(), atol=1e-4)
    xb = oe.mixed_predict(result, data, kind="xb")
    terms = table(result)
    expected = (terms["Intercept"].estimate + terms["x"].estimate * data.x
                + terms["w"].estimate * data.w)
    assert_allclose(xb.xb.to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert xb.row.tolist() == list(range(len(data)))
    nested = make_nested()
    two = oe.mixed(data=nested, y="y", x=["x"], group=["s", "k"])
    effects = oe.mixed_predict(two, nested, kind="reffects")
    assert set(effects.level) == {"s", "k"}
    assert (effects.level == "s").sum() == 15 and (effects.level == "k").sum() == 60
    with pytest.raises(AnalysisError):
        oe.mixed_predict(result, data, kind="residual")


def test_round_trip_and_rendering(data):
    result = oe.mixed(data=data, y="y", x=["x", "w"], group="g", random=["x"])
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert isinstance(result, ResultBundle)
    text = result.summary()
    assert "/var(Residual)" in text and "LR test vs. linear model" in text
    assert "tabular" in result.to_latex()
    assert result.provenance["stata_parity_validated"] is False
    assert oe.capabilities()["estimators"]["mixed"]["family"] == "mixed"
    spec = oe.ModelSpec(estimator="mixed", outcome="y", predictors=["x"],
                        columns={"group": ["g"]}, options={"method": "reml"})
    fitted = oe.fit(spec, data=data)
    assert_allclose(fitted.metrics["log_likelihood"],
                    oe.mixed(data=data, y="y", x=["x"], group="g",
                             method="reml").metrics["log_likelihood"])


def test_slope_without_random_intercept_and_no_constant(data):
    rng = np.random.default_rng(8)
    slope = data.assign(y=1 + 0.5 * data.x + rng.normal(size=40)[data.g] * data.x
                        + rng.normal(size=len(data)))
    result = oe.mixed(data=slope, y="y", x=["x", "w"], group="g", random=["x"],
                      random_intercept=False)
    ref = smf.mixedlm("y ~ x + w", slope, groups=slope["g"], re_formula="0 + x").fit(reml=False)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-9)
    assert_allclose(table(result)["/var(x[g])"].estimate, ref.cov_re.iloc[0, 0], rtol=1e-3)
    assert "icc" not in result.metrics
    # The data of the other tests have no slope-only variance: a boundary solution.
    with pytest.raises(AnalysisError) as error:
        oe.mixed(data=data, y="y", x=["x", "w"], group="g", random=["x"],
                 random_intercept=False)
    assert error.value.code == "boundary_solution"
    plain = oe.mixed(data=data, y="y", x=["x", "w"], group="g", intercept=False)
    ref = smf.mixedlm("y ~ 0 + x + w", data, groups=data["g"]).fit(reml=False)
    assert_allclose(plain.metrics["log_likelihood"], ref.llf, rtol=1e-9)
    assert "Intercept" not in table(plain)
