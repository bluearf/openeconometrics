"""glm family: second independent verification pass.

Complements ``test_econ_glm_oracle.py`` (Stata fixtures on small data sets and
brute-force likelihoods) and ``test_econ_glm_adversarial.py`` with checks that share
no code and no assumption with either of them:

1. Stata 11 output for ``poisson``, ``nbreg`` and ``nbreg, dispersion(constant)`` on
   the 20,190-row RAND Health Insurance Experiment data (coefficients, standard
   errors, confidence limits, log likelihoods of the fitted and constant-only
   models, LR chi2, pseudo R2, AIC, BIC, alpha/delta with their intervals).
2. Closed-form constant-only models (every estimator now accepts ``x=[]``, as the
   Stata commands do).
3. Structural equivalences that hold for the statistical model whatever the code:
   grouped binomial = expanded Bernoulli rows, aggregated Poisson cells with
   exposure = individual rows, reflection ``y -> 1 - y`` (logit/probit symmetric,
   cloglog <-> loglog), ppmlhdfe = Poisson on explicit dummies up to the documented
   degrees-of-freedom factors.
4. Regression tests for the defects fixed in this pass (boundary solutions of the
   log and identity binomial links, the empty-design guard).
5. The documentation examples of ``docs/econometrics/glm.md`` are executed.
"""

import math
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import optimize, special, stats

import openecon as oe
from openecon.analysis import AnalysisError

X = ["x1", "x2"]


def coefs(result):
    return np.array([c.estimate for c in result.coefficients])


def ses(result):
    return np.array([c.std_error for c in result.coefficients])


def cov(result):
    return np.asarray(result.covariance_matrix)


def limits(result):
    return np.array([[c.ci_low, c.ci_high] for c in result.coefficients])


def quiet(function, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(**kwargs)


def expect(code, function, **kwargs):
    with pytest.raises(AnalysisError) as error:
        quiet(function, **kwargs)
    assert error.value.code == code, (error.value.code, str(error.value))
    return str(error.value)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261003)
    n = 420
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    frame["g"] = np.repeat(np.arange(30), 14)
    frame["t"] = np.tile(np.arange(14), 30)
    frame["c"] = rng.integers(0, 15, size=n)
    mu = np.exp(0.4 + 0.3 * frame.x1 - 0.25 * frame.x2)
    frame["cnt"] = rng.poisson(mu).astype(float)
    frame["over"] = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    frame["pos"] = rng.gamma(3.0, mu / 3.0)
    p = special.expit(0.2 + 0.6 * frame.x1 - 0.4 * frame.x2)
    frame["bin"] = rng.binomial(1, p).astype(float)
    frame["m"] = rng.integers(1, 6, size=n).astype(float)
    frame["succ"] = rng.binomial(frame.m.astype(int), p).astype(float)
    frame["beta"] = rng.beta(p * 7, (1 - p) * 7).clip(1e-6, 1 - 1e-6)
    frame["frac"] = np.where(rng.uniform(size=n) < 0.1, 1.0, frame.beta)
    frame["expo"] = rng.uniform(0.5, 3.0, size=n)
    frame["w"] = rng.uniform(0.5, 2.0, size=n)
    return frame


# ---- 1. Stata 11 on the RAND health insurance experiment ----------------------------------


@pytest.fixture(scope="module")
def randhie():
    fixtures = pytest.importorskip("statsmodels.discrete.tests.results.results_discrete")
    loaded = sm.datasets.randhie.load_pandas()
    frame = loaded.exog.copy()
    frame["mdvis"] = loaded.endog
    return frame, list(loaded.exog.columns), fixtures.RandHIE


STATA_ORDER = [*range(1, 10), 0]                    # Stata lists _cons after the regressors


def test_poisson_reproduces_stata_on_the_rand_health_data(randhie):
    frame, x, fixtures = randhie
    stata = fixtures.poisson
    result = oe.poisson(data=frame, y="mdvis", x=x)
    assert result.nobs == stata.nobs == 20190
    assert_allclose(coefs(result)[STATA_ORDER], stata.params, rtol=2e-6)
    assert_allclose(ses(result)[STATA_ORDER], stata.bse, rtol=2e-6)
    assert_allclose(limits(result)[STATA_ORDER], stata.conf_int, atol=6e-8)   # 7 decimals
    z = np.array([c.statistic for c in result.coefficients])[STATA_ORDER]
    assert_allclose(z, stata.z, rtol=2e-6)
    metrics, model = result.metrics, result.tests["model"]
    assert metrics["log_likelihood"] == pytest.approx(stata.llf, rel=1e-9)
    assert result.extra["null_log_likelihood"] == pytest.approx(stata.llnull, rel=1e-11)
    assert model["label"].startswith("LR") and model["df"] == stata.df_model == 9
    assert model["statistic"] == pytest.approx(stata.llr, rel=1e-7)
    assert metrics["pseudo_r_squared"] == pytest.approx(stata.prsquared, rel=1e-7)
    assert metrics["aic"] == pytest.approx(stata.aic, rel=1e-9)               # estat ic
    assert metrics["bic"] == pytest.approx(stata.bic, rel=1e-9)
    assert metrics["df_resid"] == stata.df_resid


@pytest.mark.parametrize("form,name,fixture", [
    ("mean", "alpha", "negativebinomial_nb2_bfgs"),
    ("constant", "delta", "negativebinomial_nb1_bfgs")])
def test_nbreg_reproduces_stata_on_the_rand_health_data(randhie, form, name, fixture):
    frame, x, fixtures = randhie
    stata = getattr(fixtures, fixture)
    result = oe.nbreg(data=frame, y="mdvis", x=x, dispersion=form)
    assert [c.term for c in result.coefficients][-1] == f"/ln{name}"
    # Coefficients (the NB2 fixture's standard errors are R's; Stata's limits pin ours down).
    assert_allclose(coefs(result)[STATA_ORDER], stata.params[:10], rtol=2e-6)
    assert_allclose(limits(result)[STATA_ORDER], stata.conf_int[:10], atol=6e-8)
    # /lnalpha, its standard error, and alpha = exp(/lnalpha) with Stata's interval.
    assert coefs(result)[10] == pytest.approx(stata.lnalpha, rel=1e-7)
    assert ses(result)[10] == pytest.approx(stata.lnalpha_std_err, rel=2e-6)
    record = result.extra["alpha"]
    assert record["parameter"] == name
    assert record["estimate"] == pytest.approx(stata.params[10], rel=1e-8)
    assert result.metrics[name] == pytest.approx(stata.params[10], rel=1e-8)
    assert_allclose([record["ci_low"], record["ci_high"]], stata.conf_int[10], atol=6e-7)
    if form == "mean":
        assert record["std_error"] == pytest.approx(stata.bse[10], rel=2e-5)   # delta method
    metrics, model = result.metrics, result.tests["model"]
    assert metrics["log_likelihood"] == pytest.approx(stata.llf, rel=1e-11)
    assert result.extra["null_log_likelihood"] == pytest.approx(stata.llnull, rel=1e-11)
    assert model["statistic"] == pytest.approx(stata.llr, rel=1e-9) and model["df"] == 9
    assert metrics["aic"] == pytest.approx(stata.aic, rel=1e-11)               # K + 1 parameters
    assert metrics["bic"] == pytest.approx(stata.bic, rel=1e-11)
    # The constant-only NB1 and NB2 models are the same model (delta = alpha * mu).
    assert fixtures.negativebinomial_nb1_bfgs.llnull == pytest.approx(
        fixtures.negativebinomial_nb2_bfgs.llnull, rel=1e-12)


# ---- 2. constant-only models -----------------------------------------------------------------


def test_constant_only_models_have_closed_forms(data):
    n = len(data)
    crit = stats.norm.isf(0.025)
    # Poisson with exposure: rate = sum y / sum exposure, Var(ln rate) = 1 / sum y.
    result = oe.poisson(data=data, y="cnt", x=[], exposure="expo")
    assert [c.term for c in result.coefficients] == ["Intercept"]
    assert coefs(result)[0] == pytest.approx(math.log(data.cnt.sum() / data.expo.sum()), rel=1e-10)
    assert ses(result)[0] == pytest.approx(data.cnt.sum() ** -0.5, rel=1e-9)
    assert limits(result)[0, 1] == pytest.approx(coefs(result)[0] + crit * ses(result)[0])
    model = result.tests["model"]                        # no slopes: nothing to test
    assert model["df"] == 0 and model["p_value"] is None and not model["statistic"]
    assert result.metrics["pseudo_r_squared"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["df_resid"] == n - 1
    # Robust: N/(N-1) sum (y - mu)^2 / (sum mu)^2.
    robust = oe.poisson(data=data, y="cnt", x=[], covariance="robust")
    mean = data.cnt.mean()
    assert ses(robust)[0] == pytest.approx(
        math.sqrt(n / (n - 1) * ((data.cnt - mean) ** 2).sum()) / (n * mean), rel=1e-9)
    assert robust.tests["model"]["statistic"] is None and robust.tests["model"]["df"] == 0
    # Gamma, log link: mu = ybar; phi = Pearson/(N-1); Var(b0) = phi / N.
    gamma = oe.glm(data=data, y="pos", x=[], family="gamma", link="log")
    mean = data.pos.mean()
    phi = ((data.pos - mean) ** 2 / mean ** 2).sum() / (n - 1)
    assert coefs(gamma)[0] == pytest.approx(math.log(mean), rel=1e-10)
    assert gamma.metrics["scale"] == pytest.approx(phi, rel=1e-9)
    assert ses(gamma)[0] == pytest.approx(math.sqrt(phi / n), rel=1e-9)
    # Binomial with trials, logit: p = sum y / sum m; Var(b0) = 1 / (M p (1 - p)).
    share = data.succ.sum() / data.m.sum()
    binomial = oe.glm(data=data, y="succ", x=[], family="binomial", trials="m")
    assert coefs(binomial)[0] == pytest.approx(special.logit(share), rel=1e-10)
    assert ses(binomial)[0] == pytest.approx((data.m.sum() * share * (1 - share)) ** -0.5,
                                             rel=1e-9)
    # cloglog: b0 = ln(-ln(1 - ybar)).
    share = data.bin.mean()
    cll = oe.cloglog(data=data, y="bin", x=[])
    assert coefs(cll)[0] == pytest.approx(math.log(-math.log1p(-share)), rel=1e-10)
    assert cll.metrics["log_likelihood"] == pytest.approx(
        n * (share * math.log(share) + (1 - share) * math.log1p(-share)), rel=1e-12)
    # fracreg: logit(ybar) with the sandwich N/(N-1) sum (y - ybar)^2 / (N ybar (1 - ybar))^2.
    share = data.frac.mean()
    fractional = oe.fracreg(data=data, y="frac", x=[])
    assert fractional.spec.covariance == "robust"
    assert coefs(fractional)[0] == pytest.approx(special.logit(share), rel=1e-10)
    variance = n / (n - 1) * ((data.frac - share) ** 2).sum() / (n * share * (1 - share)) ** 2
    assert ses(fractional)[0] == pytest.approx(math.sqrt(variance), rel=1e-9)
    # nbreg: the fitted model is its own null model; NB1 and NB2 coincide (delta = alpha mu).
    nb2 = oe.nbreg(data=data, y="over", x=[])
    nb1 = oe.nbreg(data=data, y="over", x=[], dispersion="constant")
    assert [c.term for c in nb2.coefficients] == ["Intercept", "/lnalpha"]
    assert nb2.tests["model"]["df"] == 0 and nb2.tests["model"]["statistic"] == 0.0
    assert nb1.metrics["log_likelihood"] == pytest.approx(nb2.metrics["log_likelihood"],
                                                          rel=1e-11)
    assert coefs(nb2)[0] == pytest.approx(math.log(data.over.mean()), rel=1e-9)
    assert nb1.metrics["delta"] == pytest.approx(nb2.metrics["alpha"] * data.over.mean(),
                                                 rel=1e-6)
    # betareg: the maximum likelihood fit of one beta distribution, mu = a/(a+b), phi = a+b.
    y = data.beta.to_numpy()
    fit = optimize.minimize(lambda t: -stats.beta.logpdf(y, *np.exp(t)).sum(), [0.0, 0.0],
                            method="BFGS", options={"gtol": 1e-10})
    a, b = np.exp(fit.x)
    beta = oe.betareg(data=data, y="beta", x=[])
    assert [c.term for c in beta.coefficients] == ["Intercept", "scale:Intercept"]
    assert_allclose(coefs(beta), [special.logit(a / (a + b)), math.log(a + b)], rtol=1e-6)
    assert beta.metrics["log_likelihood"] == pytest.approx(-fit.fun, rel=1e-10)
    assert beta.extra["precision"]["estimate"] == pytest.approx(a + b, rel=1e-6)


def test_a_model_without_any_term_is_refused(data):
    for function, kwargs in ((oe.glm, {"y": "pos"}), (oe.poisson, {"y": "cnt"}),
                             (oe.nbreg, {"y": "over"}), (oe.cloglog, {"y": "bin"}),
                             (oe.fracreg, {"y": "frac"}), (oe.betareg, {"y": "beta"})):
        message = expect("empty_design", function, data=data, x=[], intercept=False, **kwargs)
        assert "constant" in message
        # ... also when the only regressor is omitted (it is identically zero).
        expect("empty_design", function, data=data.assign(k=0.0), x=["k"], intercept=False,
               **kwargs)
    # ppmlhdfe absorbs the constant, so it needs a regressor.
    expect("invalid_spec", oe.ppmlhdfe, data=data, y="cnt", x=[], absorb=["g"])


# ---- 3. structural equivalences ---------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "loglog"])
def test_grouped_binomial_equals_the_expanded_bernoulli_rows(data, link):
    grouped = oe.glm(data=data, y="succ", x=X, family="binomial", trials="m", link=link)
    rows = []
    for x1, x2, m, s in data[["x1", "x2", "m", "succ"]].itertuples(index=False):
        rows.extend([(x1, x2, 1.0)] * int(s) + [(x1, x2, 0.0)] * int(m - s))
    expanded = pd.DataFrame(rows, columns=["x1", "x2", "y"])
    bernoulli = oe.glm(data=expanded, y="y", x=X, family="binomial", link=link)
    assert bernoulli.nobs == int(data.m.sum()) and grouped.nobs == len(data)
    assert_allclose(coefs(grouped), coefs(bernoulli), rtol=1e-8)
    assert_allclose(cov(grouped), cov(bernoulli), rtol=1e-7)            # OIM: the same likelihood
    # The likelihoods differ by the binomial coefficients only.
    combinations = (special.gammaln(data.m + 1) - special.gammaln(data.succ + 1)
                    - special.gammaln(data.m - data.succ + 1)).sum()
    assert grouped.metrics["log_likelihood"] == pytest.approx(
        bernoulli.metrics["log_likelihood"] + combinations, rel=1e-10)


def test_aggregated_poisson_cells_with_exposure_equal_the_individual_rows():
    rng = np.random.default_rng(31)
    n = 3000
    frame = pd.DataFrame({"a": rng.integers(0, 4, size=n), "b": rng.integers(0, 3, size=n)})
    frame["y"] = rng.poisson(np.exp(0.2 + 0.3 * frame.a - 0.4 * (frame.b == 2))).astype(float)
    cells = frame.groupby(["a", "b"], as_index=False).agg(y=("y", "sum"), size=("y", "size"))
    individual = oe.poisson(data=frame, y="y", x=["a", "b"], categorical=["b"])
    aggregated = oe.poisson(data=cells, y="y", x=["a", "b"], categorical=["b"],
                            exposure="size")
    assert aggregated.nobs == 12
    assert_allclose(coefs(aggregated), coefs(individual), rtol=1e-8)
    assert_allclose(cov(aggregated), cov(individual), rtol=1e-7)
    # The same through glm with the offset ln(size).
    offset = oe.glm(data=cells.assign(ln=np.log(cells["size"])), y="y", x=["a", "b"],
                    categorical=["b"], family="poisson", offset="ln")
    assert_allclose(coefs(offset), coefs(individual), rtol=1e-8)
    assert_allclose(cov(offset), cov(individual), rtol=1e-7)


def test_reflecting_the_outcome_flips_signs_and_swaps_cloglog_with_loglog(data):
    flipped = data.assign(bin=1 - data.bin, beta=1 - data.beta, frac=1 - data.frac)
    pairs = {"logit": "logit", "probit": "probit", "cloglog": "loglog", "loglog": "cloglog"}
    for link, mirror in pairs.items():
        for optimizer in ("ml", "irls"):
            first = oe.glm(data=data, y="bin", x=X, family="binomial", link=link,
                           optimizer=optimizer, covariance="robust")
            second = oe.glm(data=flipped, y="bin", x=X, family="binomial", link=mirror,
                            optimizer=optimizer, covariance="robust")
            assert_allclose(coefs(second), -coefs(first), rtol=1e-7, atol=1e-10)
            assert_allclose(cov(second), cov(first), rtol=1e-6)
            assert second.metrics["log_likelihood"] == pytest.approx(
                first.metrics["log_likelihood"], rel=1e-10)
        # Beta regression: the mean equation flips, the precision equation does not.
        first = oe.betareg(data=data, y="beta", x=X, scale=["x1"], link=link)
        second = oe.betareg(data=flipped, y="beta", x=X, scale=["x1"], link=mirror)
        sign = np.array([-1, -1, -1, 1, 1])
        assert_allclose(coefs(second), sign * coefs(first), rtol=1e-6, atol=1e-9)
        assert_allclose(cov(second), np.outer(sign, sign) * cov(first), rtol=1e-5, atol=1e-12)
        assert second.metrics["log_likelihood"] == pytest.approx(
            first.metrics["log_likelihood"], rel=1e-10)
        assert second.metrics["pseudo_r_squared"] == pytest.approx(
            first.metrics["pseudo_r_squared"], rel=1e-8)
    for link in ("logit", "probit"):
        first = oe.fracreg(data=data, y="frac", x=X, link=link, cluster="g")
        second = oe.fracreg(data=flipped, y="frac", x=X, link=link, cluster="g")
        assert_allclose(coefs(second), -coefs(first), rtol=1e-7, atol=1e-10)
        assert_allclose(cov(second), cov(first), rtol=1e-6)


def test_ppmlhdfe_is_poisson_on_dummies_up_to_the_documented_factors(data):
    """ppmlhdfe (reghdfe factors) versus poisson on explicit dummies (Stata -ml- factors)."""
    n = len(data)
    dummies = {"x": [*X, "g", "t"], "categorical": ["g", "t"]}
    for weights in ({}, {"weights": "w", "weight_type": "pweight"}):
        absorbed = oe.ppmlhdfe(data=data, y="cnt", x=X, absorb=["g", "t"], tolerance=1e-13,
                               **weights)
        explicit = oe.poisson(data=data, y="cnt", covariance="robust", **dummies, **weights)
        k_total = 2 + absorbed.metrics["df_absorbed"]
        assert k_total == len(explicit.coefficients) == 2 + 30 + 14 - 1
        assert_allclose(coefs(absorbed), coefs(explicit)[1:3], rtol=1e-8)
        # robust: N/(N-K) here, N/(N-1) for poisson.
        assert_allclose(cov(absorbed) * (n - k_total) / (n - 1), cov(explicit)[1:3, 1:3],
                        rtol=1e-7)
        absorbed = oe.ppmlhdfe(data=data, y="cnt", x=X, absorb=["g", "t"], tolerance=1e-13,
                               cluster="c", **weights)
        explicit = oe.poisson(data=data, y="cnt", cluster="c", **dummies, **weights)
        # cluster: G/(G-1) (N-1)/(N-K) here, G/(G-1) for poisson.
        assert_allclose(cov(absorbed) * (n - k_total) / (n - 1), cov(explicit)[1:3, 1:3],
                        rtol=1e-7)
        assert absorbed.inference["small_sample_correction"] == pytest.approx(
            15 / 14 * (n - 1) / (n - k_total))
    absorbed = oe.ppmlhdfe(data=data, y="cnt", x=X, absorb=["g", "t"], tolerance=1e-13,
                           covariance="nonrobust")
    explicit = oe.poisson(data=data, y="cnt", **dummies)
    assert_allclose(cov(absorbed), cov(explicit)[1:3, 1:3], rtol=1e-7)
    assert absorbed.metrics["log_likelihood"] == pytest.approx(
        explicit.metrics["log_likelihood"], rel=1e-11)
    assert absorbed.metrics["deviance"] == pytest.approx(explicit.metrics["deviance"], rel=1e-9)
    # Pseudo R2: against the constant-only Poisson model on the estimation sample.
    assert absorbed.metrics["pseudo_r_squared"] == pytest.approx(
        1 - explicit.metrics["log_likelihood"] / explicit.extra["null_log_likelihood"],
        rel=1e-9)


# ---- 4. regression tests for defects fixed in this pass ----------------------------------------


def test_log_and_identity_binomial_links_report_boundary_solutions_not_separation():
    """A log-binomial likelihood maximized where a fitted probability equals 1 has finite
    coefficients: it used to be reported as separation ("finite estimates do not exist")."""
    # Analytic case: P(y=1 | x=0) = 0.4 and every y = 1 at x = 1. The saturated fit is
    # mu(0) = 0.4, mu(1) = 1: finite coefficients (ln 0.4, -ln 0.4) on the edge mu = 1.
    two = pd.DataFrame({"x": [0.0] * 10 + [1.0] * 6, "y": [1.0] * 4 + [0.0] * 6 + [1.0] * 6})
    for optimizer in ("ml", "irls"):
        for link, where in (("log", "reach 1"), ("identity", "reach 0 or 1")):
            message = expect("boundary_solution", oe.glm, data=two, y="y", x=["x"],
                             family="binomial", link=link, optimizer=optimizer)
            assert "boundary" in message and where in message and "6 fitted" in message
            assert "separat" not in message
        # With a link onto (0, 1) the same data are quasi-completely separated.
        expect("separation_detected", oe.glm, data=two, y="y", x=["x"], family="binomial",
               optimizer=optimizer)
    # Continuous regressors: a constrained optimizer confirms that the maximum is on the edge.
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    p = special.expit(0.2 + 0.6 * frame.x1 - 0.4 * frame.x2)
    frame["y"] = rng.binomial(1, p).astype(float)
    design = np.column_stack([np.ones(n), frame.x1, frame.x2])
    y = frame.y.to_numpy()

    def negative(b):
        eta = np.minimum(design @ b, -1e-12)
        return -(y * eta + (1 - y) * np.log(-np.expm1(eta))).sum()

    constrained = optimize.minimize(
        negative, [-1.0, 0.0, 0.0], method="SLSQP", options={"maxiter": 500, "ftol": 1e-14},
        constraints=[{"type": "ineq", "fun": lambda b: -(design @ b)}])
    eta = design @ constrained.x
    assert constrained.success and abs(eta.max()) < 1e-8      # the maximum is ON the boundary
    assert np.abs(constrained.x).max() < 2                     # ... with finite coefficients
    assert y[eta.argmax()] == 1
    for optimizer in ("ml", "irls"):
        message = expect("boundary_solution", oe.glm, data=frame, y="y", x=X,
                         family="binomial", link="log", optimizer=optimizer)
        assert "oe.poisson" in message and "separat" not in message
    # The unconstrained log-linear mean is what the message recommends.
    assert quiet(oe.poisson, data=frame, y="y", x=X, covariance="robust").nobs == n
    # Links that map the whole line into (0, 1) still report separation as separation.
    separated = frame.assign(d=(frame.x1 > 0.2).astype(float))
    for link in ("logit", "probit", "cloglog", "loglog"):
        expect("separation_detected", oe.glm, data=separated, y="d", x=X, family="binomial",
               link=link)


def test_count_families_with_an_identity_link_refuse_fits_on_the_edge_mu_zero():
    """IRLS used to 'converge' ON the edge mu = 0 and report standard errors of 1e-7."""
    rng = np.random.default_rng(5)
    n = 300
    frame = pd.DataFrame({"x": rng.uniform(0, 1, size=n), "z": rng.normal(size=n)})
    frame["y"] = rng.poisson(3.0 * frame.x).astype(float)        # the mean passes through 0
    edge = pd.DataFrame({"x": [0.0] * 5, "z": rng.normal(size=5), "y": [0.0] * 5})
    frame = pd.concat([frame, edge], ignore_index=True)
    design = np.column_stack([np.ones(len(frame)), frame.x, frame.z])
    y = frame.y.to_numpy()
    constrained = optimize.minimize(
        lambda b: -(y * np.log(np.maximum(design @ b, 1e-300)) - design @ b).sum(),
        [0.5, 2.0, 0.0], method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000},
        constraints=[{"type": "ineq", "fun": lambda b: design @ b}])
    assert constrained.success and abs((design @ constrained.x).min()) < 1e-8
    for family in ("poisson", "nbinomial"):
        for optimizer in ("ml", "irls"):
            message = expect("boundary_solution", oe.glm, data=frame, y="y", x=["x", "z"],
                             family=family, link="identity", optimizer=optimizer)
            assert "fitted mean" in message and "link='log'" in message
    # Away from the edge the identity link is an ordinary fit (and ml agrees with irls).
    interior = frame.assign(y=rng.poisson(1.0 + 3.0 * frame.x).astype(float))
    first = oe.glm(data=interior, y="y", x=["x", "z"], family="poisson", link="identity")
    second = oe.glm(data=interior, y="y", x=["x", "z"], family="poisson", link="identity",
                    optimizer="irls")
    assert_allclose(coefs(first), coefs(second), rtol=1e-7, atol=1e-9)
    assert coefs(first)[0] == pytest.approx(1.0, abs=0.5)
    # A count model separated through the log link diverges: reported, never estimated.
    zeros = pd.DataFrame({"x": [0.0] * 10 + [1.0] * 6, "y": [0.0] * 10 + [1.0, 2, 3, 1, 0, 4]})
    for optimizer in ("ml", "irls"):
        message = expect("nonconvergence", oe.glm, data=zeros, y="y", x=["x"],
                         family="poisson", optimizer=optimizer)
        assert "separation" in message


def test_an_interior_log_binomial_fit_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    frame["y"] = rng.binomial(1, np.exp(-1.5 + 0.2 * frame.x1 - 0.2 * frame.x2)).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = sm.GLM(frame.y, sm.add_constant(frame[X]), family=sm.families.Binomial(
            sm.families.links.Log())).fit(tol=1e-14, maxiter=200)
    expected = oe.glm(data=frame, y="y", x=X, family="binomial", link="log", optimizer="irls")
    assert_allclose(coefs(expected), reference.params, rtol=1e-7)
    assert_allclose(ses(expected), reference.bse, rtol=1e-6)        # statsmodels reports the EIM
    observed = oe.glm(data=frame, y="y", x=X, family="binomial", link="log")
    assert_allclose(coefs(observed), reference.params, rtol=1e-7)
    assert np.abs(ses(observed) / ses(expected) - 1).max() > 1e-3   # OIM differs: non-canonical
    assert observed.metrics["deviance"] == pytest.approx(reference.deviance, rel=1e-9)
    assert observed.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-9)


def test_ppmlhdfe_does_not_lecture_about_non_count_outcomes(data):
    """PPML is meant for flows and amounts: only the count commands note a non-integer y."""
    flows = data.assign(flow=data.pos * 1000.0)
    absorbed = oe.ppmlhdfe(data=flows, y="flow", x=X, absorb=["g", "t"])
    assert absorbed.warnings == []
    noted = oe.poisson(data=flows, y="flow", x=X, covariance="robust")
    assert any("non-integer" in warning for warning in noted.warnings)
    # Both remain the same estimator: one absorbed dimension = explicit dummies.
    one = oe.ppmlhdfe(data=flows, y="flow", x=X, absorb=["g"], tolerance=1e-13)
    dummies = quiet(oe.poisson, data=flows, y="flow", x=[*X, "g"], categorical=["g"],
                    covariance="robust")
    assert_allclose(coefs(one), coefs(dummies)[1:3], rtol=1e-8)


# ---- 5. the documentation examples run ----------------------------------------------------------


_TOKEN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[^\s\d]+")


def _same_output(printed, documented):
    """Token-wise equality; numbers agree to 4 significant digits (platform rounding)."""
    first, second = _TOKEN.findall(printed), _TOKEN.findall(documented)
    if len(first) != len(second):
        return False
    for a, b in zip(first, second, strict=True):
        try:
            if not math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-12):
                return False
        except ValueError:
            if a != b:
                return False
    return True


def test_documentation_examples_run_and_report_what_the_page_says(capsys):
    page = pathlib.Path(__file__).resolve().parents[1] / "docs" / "econometrics" / "glm.md"
    text = page.read_text(encoding="utf-8")
    fences = re.findall(r"```(python|text)\n(.*?)```", text, flags=re.DOTALL)
    assert sum(kind == "python" for kind, _ in fences) >= 10
    namespace: dict = {}
    compared = 0
    capsys.readouterr()
    for position, (kind, block) in enumerate(fences):
        if kind != "python":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(block, str(page), "exec"), namespace)     # noqa: S102
        printed = capsys.readouterr().out
        following = fences[position + 1] if position + 1 < len(fences) else ("", "")
        if following[0] == "text":                 # the output shown on the page is the output
            assert _same_output(printed, following[1]), (printed, following[1])
            compared += 1
        else:
            assert printed == ""
    assert compared >= 9
    # Every estimator of the family is documented under its own heading.
    for name in ("glm", "poisson", "nbreg", "cloglog", "fracreg", "betareg", "ppmlhdfe"):
        assert f"## `oe.{name}`" in text, name
    for word in ("Uncertain conventions", "N/(N-1)", "G/(G-1)", "separation_detected",
                 "boundary_solution", "Deliberate differences from Stata"):
        assert word in text, word
