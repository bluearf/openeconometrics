"""Independent oracles for the random-effects GLMMs (oe.melogit, oe.meprobit, oe.mepoisson).

The reference likelihood integrates the random effects with a high-order
NON-adaptive Gauss-Hermite rule written in NumPy (and, for Poisson counts,
SciPy's adaptive ``quad``) and is maximized by brute force with
``scipy.optimize``; with many quadrature points OpenEcon's adaptive rule must
reproduce the exact integral, with Stata's 7 points it must be close.
Standard errors are checked against a numerical Hessian of the reference
likelihood, the robust covariance against an explicit sandwich of numerical
group scores, and the pooled comparison models against statsmodels.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermite
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixed.glmm_kernels import RandomEffectsGLMM
from openecon.engines.optimize import check_derivatives


def make_data(seed=0, groups=60, size=6, sd=1.0, slope_sd=0.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), size)
    n = groups * size
    frame = pd.DataFrame({"g": g, "x": rng.normal(size=n), "w": rng.normal(size=n) + 4.0})
    eta = (-0.3 + 0.8 * frame.x - 0.2 * (frame.w - 4) + sd * rng.normal(size=groups)[g]
           + slope_sd * rng.normal(size=groups)[g] * frame.x)
    frame["y"] = (rng.random(n) < expit(eta)).astype(float)
    frame["t"] = rng.uniform(0.5, 2.0, size=n)
    frame["c"] = rng.poisson(frame.t * np.exp(0.2 + 0.4 * frame.x
                                              + 0.6 * rng.normal(size=groups)[g]))
    frame["cl"] = g // 5
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def table(result):
    return {c.term: c for c in result.coefficients}


def log_f(family, y, eta):
    if family == "logit":
        return y * np.log(expit(eta)) + (1 - y) * np.log(expit(-eta))
    if family == "probit":
        return stats.norm.logcdf((2 * y - 1) * eta)
    y = np.asarray(y).reshape(-1)
    return (y[:, None] * eta - np.exp(eta)
            - np.array([math.lgamma(v + 1) for v in y])[:, None])


def reference_groups(frame, family, xcols, params, y="y", points=80, slope=None, offset=None):
    """Per-group log likelihoods with a non-adaptive product Gauss-Hermite rule."""
    X = np.column_stack([np.ones(len(frame)), frame[xcols].to_numpy()])
    Y = frame[y].to_numpy()[:, None]
    k = X.shape[1]
    xb = X @ params[:k] + (0 if offset is None else offset)
    nodes, weights = roots_hermite(points)
    if slope is None:
        sd = math.exp(params[k])
        eta = xb[:, None] + math.sqrt(2) * sd * nodes[None, :]
        log_w = np.log(weights / math.sqrt(math.pi))
    else:
        sd_s, sd_c = math.exp(params[k]), math.exp(params[k + 1])
        a = np.repeat(nodes, points)
        b = np.tile(nodes, points)
        z = frame[slope].to_numpy()[:, None]
        eta = xb[:, None] + math.sqrt(2) * (sd_s * z * a[None, :] + sd_c * b[None, :])
        log_w = np.log(np.repeat(weights, points) * np.tile(weights, points) / math.pi)
    contributions = log_f(family, Y if family != "poisson" else frame[y].to_numpy(), eta)
    sums = pd.DataFrame(contributions).groupby(frame.g.to_numpy()).sum().to_numpy()
    return logsumexp(sums + log_w[None, :], axis=1)


def reference_fit(frame, family, xcols, start, **options):
    def negative(params):
        return -reference_groups(frame, family, xcols, params, **options).sum()

    best = minimize(negative, start, method="BFGS", options={"gtol": 1e-8})
    return best


# ---- likelihood and estimates against brute force -------------------------------------


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_binary_models_match_brute_force(data, family):
    command = {"logit": oe.melogit, "probit": oe.meprobit}[family]
    exact = command(data=data, y="y", x=["x", "w"], group="g", intpoints=30)
    default = command(data=data, y="y", x=["x", "w"], group="g")
    start = np.r_[[c.estimate for c in default.coefficients[:3]],
                  0.5 * math.log(default.coefficients[3].estimate)]
    best = reference_fit(data, family, ["x", "w"], start)
    assert_allclose(exact.metrics["log_likelihood"], -best.fun, rtol=1e-8)
    terms = table(exact)
    estimates = [terms[t].estimate for t in ("Intercept", "x", "w")]
    assert_allclose(estimates, best.x[:3], rtol=1e-4, atol=1e-5)
    assert_allclose(terms["/var(_cons[g])"].estimate, math.exp(2 * best.x[3]), rtol=1e-4)
    # Stata's 7-point adaptive rule is close to the exact integral.
    assert_allclose(default.metrics["log_likelihood"], -best.fun, rtol=1e-5)
    assert_allclose([c.estimate for c in default.coefficients],
                    [c.estimate for c in exact.coefficients], rtol=2e-3, atol=2e-4)
    # Standard errors: inverse numerical Hessian of the reference, delta method to var.
    hessian = approx_hess(best.x, lambda p: reference_groups(data, family, ["x", "w"], p).sum())
    cov = np.linalg.inv(-hessian)
    jac = np.diag([1, 1, 1, 2 * math.exp(2 * best.x[3])])
    se = np.sqrt(np.diag(jac @ cov @ jac.T))
    assert_allclose([c.std_error for c in exact.coefficients], se, rtol=2e-4)
    latent = math.pi ** 2 / 3 if family == "logit" else 1.0
    v = terms["/var(_cons[g])"].estimate
    assert_allclose(exact.metrics["icc"], v / (v + latent), rtol=1e-12)
    term = terms["/var(_cons[g])"]
    spread = stats.norm.ppf(0.975) * term.std_error / term.estimate
    assert_allclose([term.ci_low, term.ci_high],
                    [v * math.exp(-spread), v * math.exp(spread)], rtol=1e-10)


def test_random_slope_matches_brute_force():
    frame = make_data(seed=4, groups=60, size=8, slope_sd=0.7)
    result = oe.melogit(data=frame, y="y", x=["x"], group="g", random=["x"], intpoints=30)
    params = np.r_[[c.estimate for c in result.coefficients[:2]],
                   [0.5 * math.log(c.estimate) for c in result.coefficients[2:]]]

    def reference(p):
        return reference_groups(frame, "logit", ["x"], p, slope="x", points=80).sum()

    # The 80 x 80 non-adaptive product rule is the exact integral to ~1e-6 here.
    assert_allclose(result.metrics["log_likelihood"], reference(params), rtol=1e-8)
    # First-order conditions of the independent likelihood hold at OpenEcon's estimates.
    assert np.abs(approx_fprime(params, reference, centered=True)).max() < 1e-3
    assert [c.term for c in result.coefficients][2:] == ["/var(x[g])", "/var(_cons[g])"]
    assert result.tests["lr_vs_pooled"]["df"] == 2
    assert result.tests["lr_vs_pooled"]["distribution"] == "chi2"


def test_poisson_matches_exact_integral(data):
    small = data[data.g < 30].reset_index(drop=True)
    result = oe.mepoisson(data=small, y="c", x=["x"], group="g", exposure="t", intpoints=25)
    terms = table(result)
    params = [terms["Intercept"].estimate, terms["x"].estimate,
              0.5 * math.log(terms["/var(_cons[g])"].estimate)]
    total = 0.0
    sd = math.exp(params[2])
    for _, block in small.groupby("g"):
        eta0 = params[0] + params[1] * block.x.to_numpy() + np.log(block.t.to_numpy())
        y = block.c.to_numpy()
        constant = -sum(math.lgamma(v + 1) for v in y)

        def density(v, eta0=eta0, y=y, constant=constant):
            eta = eta0 + sd * v
            return math.exp((y * eta - np.exp(eta)).sum() + constant - v * v / 2) / math.sqrt(
                2 * math.pi)

        value, _ = quad(density, -12, 12, epsabs=0, epsrel=1e-12, limit=200)
        total += math.log(value)
    assert_allclose(result.metrics["log_likelihood"], total, rtol=1e-10)
    # Exposure is an offset of ln(t); the score is zero at the estimates.
    offset = oe.mepoisson(data=small.assign(lt=np.log(small.t)), y="c", x=["x"], group="g",
                          offset="lt", intpoints=25)
    assert_allclose([c.estimate for c in offset.coefficients],
                    [c.estimate for c in result.coefficients], rtol=1e-10)
    gradient = approx_fprime(np.array(params), lambda p: reference_groups(
        small, "poisson", ["x"], p, y="c", points=150, offset=np.log(small.t.to_numpy())).sum())
    assert np.abs(gradient).max() < 1e-3


def test_pooled_comparison_models(data):
    X = sm.add_constant(data[["x", "w"]])
    for command, model, y in ((oe.melogit, sm.Logit, "y"), (oe.meprobit, sm.Probit, "y"),
                              (oe.mepoisson, sm.Poisson, "c")):
        result = command(data=data, y=y, x=["x", "w"], group="g")
        pooled = model(data[y], X).fit(disp=0)
        assert_allclose(result.extra["pooled_log_likelihood"], pooled.llf, rtol=1e-10)
        lr = result.tests["lr_vs_pooled"]
        assert_allclose(lr["statistic"], 2 * (result.metrics["log_likelihood"] - pooled.llf),
                        rtol=1e-10)
        assert_allclose(lr["p_value"], 0.5 * stats.chi2.sf(lr["statistic"], 1), rtol=1e-6)


# ---- covariance -----------------------------------------------------------------------------


def test_robust_sandwich_with_numerical_group_scores(data):
    result = oe.melogit(data=data, y="y", x=["x", "w"], group="g", covariance="robust",
                        intpoints=30)
    terms = table(result)
    params = np.array([terms["Intercept"].estimate, terms["x"].estimate, terms["w"].estimate,
                       0.5 * math.log(terms["/var(_cons[g])"].estimate)])
    scores = approx_fprime(params, lambda p: reference_groups(data, "logit", ["x", "w"], p),
                           centered=True)
    hessian = approx_hess(params, lambda p: reference_groups(data, "logit", ["x", "w"], p).sum())
    bread = np.linalg.inv(-hessian)
    groups = scores.shape[0]
    cov = groups / (groups - 1) * bread @ scores.T @ scores @ bread
    jac = np.diag([1, 1, 1, 2 * math.exp(2 * params[3])])
    expected = jac @ cov @ jac.T
    assert_allclose(np.array(result.covariance_matrix), expected, rtol=1e-3, atol=1e-8)
    clustered = oe.melogit(data=data, y="y", x=["x", "w"], group="g", cluster="cl")
    assert clustered.inference["cluster_count"] == 12
    with pytest.raises(AnalysisError) as error:
        oe.melogit(data=data.assign(bad=np.arange(len(data)) % 5), y="y", x=["x"], group="g",
                   cluster="bad")
    assert error.value.code == "cluster_not_nested"


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
@pytest.mark.parametrize("q", [1, 2])
def test_analytic_derivatives(data, family, q):
    x = torch.tensor(np.column_stack([np.ones(len(data)), data.x]), dtype=torch.float64)
    y = torch.tensor((data.y if family != "poisson" else data.c).to_numpy(), dtype=torch.float64)
    z = torch.tensor(np.column_stack([data.x, np.ones(len(data))])[:, 2 - q:],
                     dtype=torch.float64)
    objective = RandomEffectsGLMM(x, y, z, torch.tensor(data.g.to_numpy()), 60, family, 7)
    theta = torch.tensor([0.1, 0.5] + [-0.4, -0.2][:q], dtype=torch.float64)
    objective.adapt(theta)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-7
    assert report["hessian_max_rel_error"] < 1e-7


def test_integration_methods_agree_with_many_points(data):
    values = [oe.melogit(data=data, y="y", x=["x"], group="g", intmethod=method,
                         intpoints=points).metrics["log_likelihood"]
              for method, points in (("mvaghermite", 40), ("mcaghermite", 40),
                                     ("ghermite", 60))]
    assert_allclose(values, values[0], rtol=1e-8)


# ---- sample, errors, rendering ------------------------------------------------------------


def test_collinearity_and_missing(data):
    frame = data.assign(x2=3 * data.x)
    result = oe.melogit(data=frame, y="y", x=["x", "x2"], group="g")
    assert "x2" not in table(result) and result.provenance["omitted_terms"] == ["x2"]
    frame.loc[[0, 5], "x"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.melogit(data=frame, y="y", x=["x"], group="g")
    assert error.value.code == "missing_values"
    dropped = oe.melogit(data=frame, y="y", x=["x"], group="g", missing="drop")
    assert dropped.nobs == len(frame) - 2


def test_error_contract(data):
    def code(command=oe.melogit, **kwargs):
        options = {"data": data, "y": "y", "x": ["x"], "group": "g"}
        options.update(kwargs)
        with pytest.raises(AnalysisError) as error:
            command(**options)
        return error.value.code

    assert code(y="c") == "invalid_binary_outcome"
    assert code(command=oe.mepoisson, y="w") == "invalid_count_outcome"
    assert code(random=["x", "w"]) == "invalid_spec"
    assert code(intpoints=0) == "invalid_spec"
    assert code(intmethod="laplace") == "invalid_spec"
    assert code(group="x") == "invalid_spec"
    separated = data.assign(s=(data.y > 0).astype(float) * 2 - 1)
    assert code(data=separated, x=["x", "s"]) == "separation_detected"
    rng = np.random.default_rng(9)
    noise = data.assign(y=(rng.random(len(data)) < expit(0.5 * data.x)).astype(float))
    assert code(data=noise) == "boundary_solution"
    assert code(data=data.assign(one=1), group="one") == "insufficient_groups"


def test_round_trip_rendering_and_access(data):
    result = oe.mepoisson(data=data, y="c", x=["x"], group="g", exposure="t")
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "/var(_cons[g])" in text and "LR test vs. Poisson model" in text
    assert "tabular" in result.to_latex()
    assert result.provenance["stata_parity_validated"] is False
    assert result.extra["integration"] == {"method": "mvaghermite", "points": 7,
                                           "adaptations": result.extra["integration"][
                                               "adaptations"]}
    capability = oe.capabilities()["estimators"]["melogit"]
    assert capability["options"]["intpoints"]["default"] == 7
    spec = oe.ModelSpec(estimator="meprobit", outcome="y", predictors=["x"],
                        columns={"group": "g"})
    assert_allclose(oe.fit(spec, data=data).metrics["log_likelihood"],
                    oe.meprobit(data=data, y="y", x=["x"], group="g").metrics["log_likelihood"])
