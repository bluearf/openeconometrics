"""Independent oracles for the discrete-choice family (verification stage).

Nothing here reuses the formulas of ``openecon.econometrics.discrete``. Every
model is re-derived from its definition:

* the per-observation log likelihood is written in NumPy / SciPy from the
  textbook probability (``scipy.stats`` distribution functions, an enumeration
  of the conditional-logit denominator, Owen's T function for the bivariate
  normal distribution function);
* it is maximized by SciPy's BFGS and polished by Newton steps whose gradient
  and Hessian are Richardson-extrapolated central differences;
* the covariance estimators are assembled from numerically differentiated
  per-observation scores and the numerical Hessian under Stata's -ml-
  conventions (OIM, OPG, N/(N-1) sandwich, G/(G-1) cluster sandwich);
* statsmodels (OrderedModel, MNLogit, ConditionalLogit, Probit) is used where
  it implements the identical model;
* invariances are checked exactly: frequency weights against duplicated rows,
  row permutations, rescaled regressors, relabelled categories, equivalent
  estimators (binary ologit = logit, mlogit with two categories = logit,
  clogit with one positive per group = McFadden's choice model, hetprobit and
  biprobit against probit).
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import optimize as sopt
from scipy import special, stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

import openecon as oe
from openecon.analysis import AnalysisError

KINDS = ("nonrobust", "opg", "robust", "cluster")


# ---- numerical calculus written for this file ----------------------------------------


def jacobian(function, theta, relative_step=2e-2):
    """d function / d theta by central differences extrapolated to O(h^6).

    ``function`` may return a scalar or an array; the derivative index is the
    last axis of the result.
    """
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        base = relative_step * max(1.0, abs(theta[j]))
        levels = []
        for halving in range(3):
            step = base / 2 ** halving
            shift = np.zeros_like(theta)
            shift[j] = step
            levels.append((np.asarray(function(theta + shift), dtype=float)
                           - np.asarray(function(theta - shift), dtype=float)) / (2 * step))
        first = (4 * levels[1] - levels[0]) / 3
        second = (4 * levels[2] - levels[1]) / 3
        columns.append((16 * second - first) / 15)
    return np.stack(columns, axis=-1)


def hessian_of(total, theta):
    matrix = jacobian(lambda point: jacobian(total, point), theta)
    return (matrix + matrix.T) / 2


def maximize(loglik_obs, start, weights=None):
    """Brute-force MLE: BFGS, then Newton steps with numerical derivatives."""
    def total(point):
        values = loglik_obs(point)
        return float((values if weights is None else weights * values).sum())

    def negative(point):
        with np.errstate(all="ignore"):
            value = -total(point)
        return value if np.isfinite(value) else 1e15

    theta = sopt.minimize(negative, np.asarray(start, dtype=float), method="BFGS",
                          options={"gtol": 1e-7, "maxiter": 2000}).x
    for _ in range(25):
        gradient = jacobian(total, theta)
        hessian = hessian_of(total, theta)
        step = np.linalg.solve(-hessian, gradient)
        theta = theta + step
        if np.max(np.abs(step) / np.maximum(1.0, np.abs(theta))) < 1e-11:
            break
    else:
        raise AssertionError("the oracle maximization did not converge")
    return theta, total(theta), hessian_of(total, theta)


def stata_covariance(kind, hessian, scores, *, nobs, frequency=None, clusters=None,
                     opg_weights=None):
    """Stata's -ml- covariance estimators from a Hessian and per-row scores.

    ``scores`` are already multiplied by analytic / importance / sampling
    weights; ``frequency`` replicates rows (N = sum of frequencies).
    """
    bread = np.linalg.inv(-hessian)
    if kind == "nonrobust":
        return bread
    frequency = np.ones(len(scores)) if frequency is None else np.asarray(frequency, dtype=float)
    outer = (scores * frequency[:, None]).T @ scores
    if kind == "opg":
        if opg_weights is not None:
            # The OPG estimates the information of sum_i w_i l_i, which is linear in
            # w_i: sum w s s' (scores arrive multiplied by w, so divide one factor out).
            scaled = scores * (frequency / np.asarray(opg_weights, dtype=float))[:, None]
            outer = scaled.T @ scores
        return np.linalg.inv(outer)
    if kind == "robust":
        return nobs / (nobs - 1) * bread @ outer @ bread
    labels, index = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.zeros((len(labels), scores.shape[1]))
    np.add.at(sums, index, scores * frequency[:, None])
    return len(labels) / (len(labels) - 1) * bread @ sums.T @ sums @ bread


def table(result):
    rows = result.coefficients
    return {name: np.array([getattr(row, name) for row in rows])
            for name in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")}


def terms(result):
    return [row.term for row in result.coefficients]


def check_table(result, theta, covariance, *, alpha=0.05, rtol=1e-7, cov_rtol=2e-6):
    """Coefficients, full covariance, z statistics, p-values and intervals."""
    got = table(result)
    scale = np.sqrt(np.diag(covariance))
    assert_allclose(got["estimate"], theta, rtol=rtol, atol=rtol * 1e-2)
    assert_allclose(np.array(result.covariance_matrix), covariance, rtol=cov_rtol,
                    atol=cov_rtol * np.outer(scale, scale).max() * 1e-3)
    se = np.sqrt(np.diag(np.array(result.covariance_matrix)))
    assert_allclose(got["std_error"], se, rtol=1e-12)
    assert_allclose(got["statistic"], got["estimate"] / se, rtol=1e-12)
    assert_allclose(got["p_value"], 2 * stats.norm.sf(np.abs(got["statistic"])), rtol=1e-9,
                    atol=1e-300)
    z = stats.norm.ppf(1 - alpha / 2)
    assert_allclose(got["ci_low"], got["estimate"] - z * se, rtol=1e-9, atol=1e-12)
    assert_allclose(got["ci_high"], got["estimate"] + z * se, rtol=1e-9, atol=1e-12)
    assert result.inference["distribution"] == "normal" and result.inference["use_t"] is False
    assert result.provenance["stata_parity_validated"] is False


def wald(theta, covariance, index):
    index = list(index)
    block = covariance[np.ix_(index, index)]
    return float(theta[index] @ np.linalg.solve(block, theta[index]))


def check_chi2(test, statistic, df, *, rtol=1e-6):
    assert test["distribution"] == "chi2" and test["df"] == df
    assert_allclose(test["statistic"], statistic, rtol=rtol, atol=1e-8)
    assert_allclose(test["p_value"], stats.chi2.sf(statistic, df), rtol=max(rtol, 1e-6) * 50,
                    atol=1e-300)


def expand(frame, weight):
    """The data set in which a frequency-weighted row appears ``weight`` times."""
    return frame.loc[frame.index.repeat(frame[weight].astype(int))].reset_index(drop=True)


def same_fit(left, right, *, rtol=1e-9, metrics=True):
    assert terms(left) == terms(right)
    assert_allclose(table(left)["estimate"], table(right)["estimate"], rtol=rtol, atol=1e-10)
    assert_allclose(np.array(left.covariance_matrix), np.array(right.covariance_matrix),
                    rtol=max(rtol, 1e-8), atol=1e-12)
    if metrics:
        assert left.nobs == right.nobs
        for name, value in left.metrics.items():
            assert_allclose(value, right.metrics[name], rtol=1e-9, atol=1e-9, err_msg=name)
        for name, value in left.tests.items():
            assert value["df"] == right.tests[name]["df"]
            assert_allclose(value["statistic"], right.tests[name]["statistic"], rtol=1e-7,
                            atol=1e-8, err_msg=name)


# ---- data ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def survey():
    """Unbalanced ordered / unordered outcomes, uneven clusters, weights of every kind."""
    rng = np.random.default_rng(9117)
    n = 480
    sizes = rng.integers(3, 22, size=200)
    cluster = np.repeat(np.arange(len(sizes)), sizes)[:n]
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": (rng.random(n) < 0.35).astype(float),
        "x3": rng.exponential(size=n),
        "region": rng.choice(["north", "south", "west"], size=n, p=[0.5, 0.3, 0.2]),
        "village": cluster, "fw": rng.integers(1, 5, size=n).astype(float),
        "aw": rng.uniform(0.2, 3.0, size=n), "pw": rng.uniform(10.0, 400.0, size=n),
        "exposure": rng.normal(scale=0.5, size=n),
    })
    effect = rng.normal(scale=0.6, size=cluster.max() + 1)[cluster]
    latent = 0.9 * frame.x1 - 0.7 * frame.x2 + 0.25 * frame.x3 + effect
    frame["health"] = np.digitize(latent + rng.logistic(size=n), [0.4, 1.6, 3.0]) * 10.0
    frame["grade"] = np.digitize(latent + rng.normal(size=n), [-0.2, 0.9, 2.2]) + 1.0
    utility = np.column_stack([
        1.1 + 0.0 * frame.x1, 0.6 * frame.x1 - 0.4 * frame.x2, -0.5 - 0.7 * frame.x1 + effect,
        -1.2 + 0.5 * frame.x3])
    frame["brand"] = np.array(["delta", "alpha", "gamma", "beta"])[
        (utility + rng.gumbel(size=(n, 4))).argmax(axis=1)]
    return frame


def design(frame, columns, *, constant=False, dummies=None):
    """Design matrix written by hand: constant, numeric columns, treatment dummies."""
    blocks = [np.ones((len(frame), 1))] if constant else []
    blocks.extend(frame[[name]].to_numpy(dtype=float) for name in columns)
    for name, levels in (dummies or {}).items():
        blocks.extend((frame[[name]].to_numpy() == level).astype(float) for level in levels[1:])
    return np.hstack(blocks)


def weight_setup(frame, weight_type):
    """Likelihood weights, score weights, frequencies and N under Stata's weight semantics."""
    n = len(frame)
    if weight_type is None:
        return None, np.ones(n), None, n
    if weight_type == "fweight":
        return frame.fw.to_numpy(), np.ones(n), frame.fw.to_numpy(), int(frame.fw.sum())
    if weight_type == "aweight":
        scaled = frame.aw.to_numpy() * n / frame.aw.sum()
        return scaled, scaled, None, n
    raw = frame[{"iweight": "aw", "pweight": "pw"}[weight_type]].to_numpy()
    return raw, raw, None, n


WEIGHT_COLUMN = {"fweight": "fw", "aweight": "aw", "iweight": "aw", "pweight": "pw"}


# ---- ordered logit / probit ------------------------------------------------------------


def ordered_loglik(theta, x, codes, link, offset=0.0):
    """ln[F(k_y - x'b) - F(k_{y-1} - x'b)] with scipy.stats distribution functions."""
    k = x.shape[1]
    edges = np.concatenate([[-np.inf], theta[k:], [np.inf]])
    index = x @ theta[:k] + offset
    cdf = stats.logistic.cdf if link == "logit" else stats.norm.cdf
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(cdf(edges[codes + 1] - index) - cdf(edges[codes] - index))


def ordered_oracle(frame, outcome, x, link, *, weight_type=None, offset=None):
    values, codes = np.unique(frame[outcome].to_numpy(), return_inverse=True)
    cuts = len(values) - 1
    weights, score_weights, frequency, nobs = weight_setup(frame, weight_type)
    shift = 0.0 if offset is None else frame[offset].to_numpy()

    def loglik(theta):
        return ordered_loglik(theta, x, codes, link, shift)

    start = np.concatenate([np.zeros(x.shape[1]), np.linspace(-1, 1, cuts)])
    theta, value, hessian = maximize(loglik, start, weights)
    scores = jacobian(loglik, theta) * score_weights[:, None]
    count = np.ones(len(frame)) if weights is None else weights
    totals = np.array([count[codes == j].sum() for j in range(len(values))])
    null = float((totals * np.log(totals / totals.sum())).sum())
    return {"theta": theta, "ll": value, "hessian": hessian, "scores": scores, "nobs": nobs,
            "frequency": frequency, "score_weights": score_weights, "null": null, "categories": values.tolist(),
            "counts": totals}


@pytest.mark.parametrize("command,outcome", [("ologit", "health"), ("oprobit", "grade")])
def test_ordered_models_against_brute_force_likelihood(survey, command, outcome):
    link = command[1:]
    names = ["x1", "x2", "x3"]
    x = design(survey, names, dummies={"region": ["north", "south", "west"]})
    oracle = ordered_oracle(survey, outcome, x, link)
    k, size = x.shape[1], len(oracle["theta"])
    for kind in KINDS:
        result = getattr(oe, command)(
            data=survey, y=outcome, x=[*names, "region"], categorical=["region"],
            covariance=None if kind == "cluster" else kind,
            cluster="village" if kind == "cluster" else None, alpha=0.1)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=oracle["nobs"], clusters=survey.village)
        check_table(result, oracle["theta"], covariance, alpha=0.1)
        assert terms(result) == ["x1", "x2", "x3", "region[south]", "region[west]", "/cut1",
                                 "/cut2", "/cut3"]
        assert [row.equation for row in result.coefficients] == [outcome] * k + [None] * 3
        assert result.nobs == len(survey) and result.extra["categories"] == oracle["categories"]
        assert_allclose(result.extra["cutpoints"], oracle["theta"][k:], rtol=1e-7)
        assert_allclose(result.extra["category_counts"], oracle["counts"])
        metrics = result.metrics
        assert_allclose(metrics["log_likelihood"], oracle["ll"], rtol=1e-11)
        assert_allclose(metrics["pseudo_r_squared"], 1 - oracle["ll"] / oracle["null"], rtol=1e-9)
        assert_allclose(metrics["aic"], -2 * oracle["ll"] + 2 * size, rtol=1e-11)
        assert_allclose(metrics["bic"], -2 * oracle["ll"] + size * math.log(len(survey)),
                        rtol=1e-11)
        assert metrics["n_categories"] == 4
        if kind == "nonrobust":
            check_chi2(result.tests["model"], 2 * (oracle["ll"] - oracle["null"]), k)
        else:
            check_chi2(result.tests["model"], wald(oracle["theta"], covariance, range(k)), k,
                       rtol=1e-5)
        if kind == "cluster":
            assert result.inference["cluster_count"] == survey.village.nunique()


@pytest.mark.parametrize("command,outcome", [("ologit", "health"), ("oprobit", "grade")])
def test_ordered_models_match_statsmodels(survey, command, outcome):
    names = ["x1", "x2", "x3"]
    model = OrderedModel(survey[outcome], survey[names],
                         distr="logit" if command == "ologit" else "probit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(method="newton", disp=False, maxiter=200, tol=1e-13)
    cuts = model.transform_threshold_params(fitted.params.to_numpy())[1:-1]
    result = getattr(oe, command)(data=survey, y=outcome, x=names)
    assert_allclose(table(result)["estimate"], np.concatenate([fitted.params[:3], cuts]),
                    rtol=1e-7)
    assert_allclose(result.metrics["log_likelihood"], fitted.llf, rtol=1e-11)
    assert_allclose(table(result)["std_error"][:3], fitted.bse[:3], rtol=1e-5)
    assert_allclose(result.metrics["pseudo_r_squared"], fitted.prsquared, rtol=1e-8)
    assert_allclose(result.tests["model"]["statistic"], fitted.llr, rtol=1e-8)


@pytest.mark.parametrize("weight_type", ["fweight", "aweight", "iweight", "pweight"])
@pytest.mark.parametrize("command,outcome", [("ologit", "health"), ("oprobit", "grade")])
def test_ordered_weights_against_brute_force(survey, command, outcome, weight_type):
    names = ["x1", "x2"]
    x = design(survey, names)
    oracle = ordered_oracle(survey, outcome, x, command[1:], weight_type=weight_type)
    kinds = ("robust", "cluster") if weight_type == "pweight" else KINDS
    size = len(oracle["theta"])
    for kind in kinds:
        result = getattr(oe, command)(
            data=survey, y=outcome, x=names, weights=WEIGHT_COLUMN[weight_type],
            weight_type=weight_type, covariance=None if kind == "cluster" else kind,
            cluster="village" if kind == "cluster" else None)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=oracle["nobs"], frequency=oracle["frequency"],
                                      opg_weights=oracle["score_weights"],
                                      clusters=survey.village)
        check_table(result, oracle["theta"], covariance)
        assert result.nobs == oracle["nobs"]
        assert_allclose(result.metrics["log_likelihood"], oracle["ll"], rtol=1e-11)
        assert_allclose(result.metrics["bic"],
                        -2 * oracle["ll"] + size * math.log(oracle["nobs"]), rtol=1e-11)
        assert_allclose(result.metrics["pseudo_r_squared"], 1 - oracle["ll"] / oracle["null"],
                        rtol=1e-9)
    if weight_type == "pweight":
        default = getattr(oe, command)(data=survey, y=outcome, x=names, weights="pw",
                                       weight_type="pweight")
        assert default.spec.covariance == "robust"
        scaled = survey.assign(pw=survey.pw * 37.0)
        other = getattr(oe, command)(data=scaled, y=outcome, x=names, weights="pw",
                                     weight_type="pweight")
        same_fit(default, other, rtol=1e-8, metrics=False)
        with pytest.raises(AnalysisError) as caught:
            getattr(oe, command)(data=survey, y=outcome, x=names, weights="pw",
                                 weight_type="pweight", covariance="nonrobust")
        assert caught.value.code == "unsupported_covariance"


@pytest.mark.parametrize("command", ["ologit", "oprobit"])
def test_ordered_frequency_weights_equal_duplicated_rows(survey, command):
    outcome = "health" if command == "ologit" else "grade"
    for kind in KINDS:
        options = {"covariance": None if kind == "cluster" else kind,
                   "cluster": "village" if kind == "cluster" else None}
        weighted = getattr(oe, command)(data=survey, y=outcome, x=["x1", "x2", "x3"],
                                        weights="fw", weight_type="fweight", **options)
        duplicated = getattr(oe, command)(data=expand(survey, "fw"), y=outcome,
                                          x=["x1", "x2", "x3"], **options)
        same_fit(weighted, duplicated)


@pytest.mark.parametrize("command", ["ologit", "oprobit"])
def test_ordered_offset_against_brute_force(survey, command):
    outcome = "health" if command == "ologit" else "grade"
    x = design(survey, ["x1", "x3"])
    oracle = ordered_oracle(survey, outcome, x, command[1:], offset="exposure")
    result = getattr(oe, command)(data=survey, y=outcome, x=["x1", "x3"], offset="exposure")
    covariance = stata_covariance("nonrobust", oracle["hessian"], oracle["scores"],
                                  nobs=len(survey))
    check_table(result, oracle["theta"], covariance)
    # Stata's constant-only model keeps the offset: maximize the cutpoints alone.
    codes = np.unique(survey[outcome].to_numpy(), return_inverse=True)[1]
    empty = np.empty((len(survey), 0))
    _, null, _ = maximize(lambda theta: ordered_loglik(theta, empty, codes, command[1:],
                                                       survey.exposure.to_numpy()),
                          np.linspace(-1, 1, 3))
    assert_allclose(result.extra["null_log_likelihood"], null, rtol=1e-10)
    check_chi2(result.tests["model"], 2 * (oracle["ll"] - null), 2)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - oracle["ll"] / null, rtol=1e-9)


def test_ordered_invariances_and_equivalences(survey):
    names = ["x1", "x2", "x3"]
    base = oe.ologit(data=survey, y="health", x=names, cluster="village")
    # Row order is irrelevant.
    shuffled = survey.sample(frac=1.0, random_state=5).reset_index(drop=True)
    same_fit(base, oe.ologit(data=shuffled, y="health", x=names, cluster="village"))
    # The category VALUES are irrelevant, only their order: 0/10/20/30 -> any increasing map.
    relabelled = survey.assign(health=survey.health.map({0.0: -3.5, 10.0: 0.0, 20.0: 1e-3,
                                                         30.0: 7e6}))
    other = oe.ologit(data=relabelled, y="health", x=names, cluster="village")
    same_fit(base, other)
    assert other.extra["categories"] == [-3.5, 0, 0.001, 7000000]
    # An ordered Categorical with text labels gives the same fit.
    labelled = survey.assign(health=pd.Categorical.from_codes(
        (survey.health / 10).astype(int), categories=["bad", "fair", "good", "great"],
        ordered=True))
    other = oe.ologit(data=labelled, y="health", x=names, cluster="village")
    same_fit(base, other)
    assert other.extra["categories"] == ["bad", "fair", "good", "great"]
    # Reversing the order flips the slopes and mirrors the cutpoints (logistic symmetry).
    reverse = oe.ologit(data=survey.assign(health=-survey.health), y="health", x=names,
                        cluster="village")
    assert_allclose(table(reverse)["estimate"][:3], -table(base)["estimate"][:3], rtol=1e-8)
    assert_allclose(table(reverse)["estimate"][3:], -table(base)["estimate"][3:][::-1],
                    rtol=1e-8)
    assert_allclose(table(reverse)["std_error"][3:], table(base)["std_error"][3:][::-1],
                    rtol=1e-7)
    # Rescaled regressors rescale coefficients and standard errors, nothing else.
    scaled = survey.assign(x1=survey.x1 * 1e8, x3=survey.x3 * 1e-8)
    other = oe.ologit(data=scaled, y="health", x=names, cluster="village")
    factor = np.array([1e-8, 1.0, 1e8])
    assert_allclose(table(other)["estimate"][:3], table(base)["estimate"][:3] * factor,
                    rtol=1e-6)
    assert_allclose(table(other)["std_error"][:3], table(base)["std_error"][:3] * factor,
                    rtol=1e-6)
    assert_allclose(table(other)["statistic"][:3], table(base)["statistic"][:3], rtol=1e-6)
    assert_allclose(other.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-11)
    assert_allclose(other.tests["model"]["statistic"], base.tests["model"]["statistic"],
                    rtol=1e-6)
    # A shifted regressor moves the cutpoints by shift * b and nothing else.
    shifted = oe.ologit(data=survey.assign(x3=survey.x3 + 1e4), y="health", x=names,
                        cluster="village")
    slopes = table(base)["estimate"][:3]
    assert_allclose(table(shifted)["estimate"][:3], slopes, rtol=1e-7)
    assert_allclose(table(shifted)["estimate"][3:], table(base)["estimate"][3:] + 1e4 * slopes[2],
                    rtol=1e-9)
    assert_allclose(table(shifted)["std_error"][:3], table(base)["std_error"][:3], rtol=1e-6)


@pytest.mark.parametrize("command", ["ologit", "oprobit"])
def test_binary_ordered_model_is_logit_or_probit(survey, command):
    binary = survey.assign(high=(survey.health >= 20).astype(float))
    names = ["x1", "x2", "x3"]
    reference = (sm.Logit if command == "ologit" else sm.Probit)(
        binary.high, sm.add_constant(binary[names])).fit(disp=False, tol=1e-13, maxiter=100)
    result = getattr(oe, command)(data=binary, y="high", x=names)
    # P(y = 1) = F(x'b - cut1): the cutpoint is minus the constant.
    assert_allclose(table(result)["estimate"], [*reference.params[1:], -reference.params.iloc[0]],
                    rtol=1e-7)
    assert_allclose(table(result)["std_error"], [*reference.bse[1:], reference.bse.iloc[0]],
                    rtol=1e-6)
    assert_allclose(result.metrics["log_likelihood"], reference.llf, rtol=1e-11)
    assert_allclose(result.metrics["pseudo_r_squared"], reference.prsquared, rtol=1e-9)
    assert_allclose(result.tests["model"]["statistic"], reference.llr, rtol=1e-8)
    assert_allclose(result.tests["model"]["p_value"], reference.llr_pvalue, rtol=1e-6)
    assert_allclose(result.metrics["aic"], reference.aic, rtol=1e-11)
    assert_allclose(result.metrics["bic"], reference.bic, rtol=1e-11)


def test_ordered_missing_collinear_and_tiny_samples(survey):
    rng = np.random.default_rng(3)
    holes = survey.copy()
    holes.loc[rng.choice(len(holes), 25, replace=False), "x1"] = np.nan
    holes.loc[rng.choice(len(holes), 15, replace=False), "health"] = np.nan
    holes.loc[rng.choice(len(holes), 10, replace=False), "village"] = np.nan
    complete = holes.dropna(subset=["x1", "x2", "health", "village"]).reset_index(drop=True)
    with pytest.raises(AnalysisError) as caught:
        oe.oprobit(data=holes, y="health", x=["x1", "x2"], cluster="village")
    assert caught.value.code == "missing_values"
    dropped = oe.oprobit(data=holes, y="health", x=["x1", "x2"], cluster="village",
                         missing="drop")
    same_fit(dropped, oe.oprobit(data=complete, y="health", x=["x1", "x2"], cluster="village"))
    assert dropped.nobs == len(complete) and dropped.dropped_rows == len(holes) - len(complete)
    assert dropped.sample_positions == holes.index[
        holes[["x1", "x2", "health", "village"]].notna().all(axis=1)].tolist()
    # Exact collinearity and a constant column: omitted, the fit is that of the reduced model.
    extended = survey.assign(combo=2 * survey.x1 - 3 * survey.x2, one=1.0)
    reduced = oe.ologit(data=survey, y="health", x=["x1", "x2"])
    full = oe.ologit(data=extended, y="health", x=["x1", "one", "x2", "combo"])
    same_fit(reduced, full)
    assert full.provenance["omitted_terms"] == ["one", "combo"]
    assert any("Omitted" in message for message in full.warnings)
    # Tiny sample: 14 rows, three categories, one regressor; still the brute-force maximum.
    tiny = pd.DataFrame({"x": [0.1, 0.5, -1.2, 2.0, 0.3, -0.7, 1.5, -2.2, 0.9, -0.1, 1.1, -1.6,
                               0.2, 2.4],
                         "y": [1, 2, 1, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3]})
    codes = tiny.y.to_numpy() - 1
    theta, value, hessian = maximize(
        lambda t: ordered_loglik(t, tiny[["x"]].to_numpy(), codes, "probit"), [0.0, -0.5, 0.5])
    result = oe.oprobit(data=tiny, y="y", x=["x"])
    check_table(result, theta, np.linalg.inv(-hessian))
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-11)


# ---- multinomial logit -----------------------------------------------------------------


def mlogit_loglik(theta, x, codes, categories, base):
    """x'b_y - ln sum_l exp(x'b_l) with the base category's coefficients fixed at zero."""
    coefficients = np.zeros((categories, x.shape[1]))
    coefficients[[j for j in range(categories) if j != base]] = theta.reshape(categories - 1, -1)
    index = x @ coefficients.T
    return index[np.arange(len(x)), codes] - special.logsumexp(index, axis=1)


def mlogit_oracle(frame, outcome, x, *, base=None, weight_type=None):
    labels, codes = np.unique(frame[outcome].to_numpy(), return_inverse=True)
    weights, score_weights, frequency, nobs = weight_setup(frame, weight_type)
    count = np.ones(len(frame)) if weights is None else weights
    totals = np.array([count[codes == j].sum() for j in range(len(labels))])
    position = int(np.argmax(totals)) if base is None else labels.tolist().index(base)

    def loglik(theta):
        return mlogit_loglik(theta, x, codes, len(labels), position)

    theta, value, hessian = maximize(loglik, np.zeros(x.shape[1] * (len(labels) - 1)), weights)
    scores = jacobian(loglik, theta) * score_weights[:, None]
    return {"theta": theta, "ll": value, "hessian": hessian, "scores": scores, "nobs": nobs,
            "frequency": frequency, "score_weights": score_weights, "labels": labels.tolist(), "base": labels[position],
            "null": float((totals * np.log(totals / totals.sum())).sum()), "counts": totals}


def test_mlogit_against_brute_force_likelihood(survey):
    names = ["x1", "x2", "x3"]
    x = design(survey, names, constant=True)
    oracle = mlogit_oracle(survey, "brand", x)
    assert oracle["base"] == survey.brand.value_counts().idxmax() == "delta"
    equations = ["alpha", "beta", "gamma"]
    expected = [f"{label}:{term}" for label in equations for term in ["Intercept", *names]]
    slopes = [i for i, term in enumerate(expected) if not term.endswith(":Intercept")]
    for kind in KINDS:
        result = oe.mlogit(data=survey, y="brand", x=names,
                           covariance=None if kind == "cluster" else kind,
                           cluster="village" if kind == "cluster" else None, alpha=0.01)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=len(survey), clusters=survey.village)
        check_table(result, oracle["theta"], covariance, alpha=0.01)
        assert terms(result) == expected
        assert [row.equation for row in result.coefficients] == [e for e in equations
                                                                  for _ in range(4)]
        assert result.extra["categories"] == ["alpha", "beta", "delta", "gamma"]
        assert result.extra["base"] == "delta"
        assert_allclose(result.extra["category_counts"], oracle["counts"])
        metrics = result.metrics
        assert_allclose(metrics["log_likelihood"], oracle["ll"], rtol=1e-11)
        assert_allclose(metrics["pseudo_r_squared"], 1 - oracle["ll"] / oracle["null"], rtol=1e-9)
        assert_allclose(metrics["aic"], -2 * oracle["ll"] + 2 * 12, rtol=1e-11)
        assert_allclose(metrics["bic"], -2 * oracle["ll"] + 12 * math.log(len(survey)),
                        rtol=1e-11)
        assert metrics["n_categories"] == 4
        if kind == "nonrobust":
            check_chi2(result.tests["model"], 2 * (oracle["ll"] - oracle["null"]), 9)
        else:
            check_chi2(result.tests["model"], wald(oracle["theta"], covariance, slopes), 9,
                       rtol=1e-5)


def test_mlogit_matches_statsmodels_and_base_reparameterization(survey):
    names = ["x1", "x2", "x3"]
    reference = sm.MNLogit(pd.Categorical(survey.brand).codes,
                           sm.add_constant(survey[names])).fit(disp=False, tol=1e-14,
                                                                 maxiter=200, method="newton")
    result = oe.mlogit(data=survey, y="brand", x=names, base="alpha")    # statsmodels' base
    assert_allclose(table(result)["estimate"], reference.params.to_numpy().T.ravel(), rtol=1e-8)
    assert_allclose(np.array(result.covariance_matrix), reference.cov_params().to_numpy(),
                    rtol=1e-7)
    assert_allclose(result.metrics["log_likelihood"], reference.llf, rtol=1e-12)
    assert_allclose(result.metrics["pseudo_r_squared"], reference.prsquared, rtol=1e-8)
    assert_allclose(result.tests["model"]["statistic"], reference.llr, rtol=1e-8)
    assert result.tests["model"]["df"] == reference.df_model == 9
    assert_allclose(result.tests["model"]["p_value"], reference.llr_pvalue, rtol=1e-6)
    assert_allclose(result.metrics["aic"], reference.aic, rtol=1e-12)
    assert_allclose(result.metrics["bic"], reference.bic, rtol=1e-12)
    # statsmodels' HC0 sandwich times Stata's N/(N-1).
    sandwich = sm.MNLogit(pd.Categorical(survey.brand).codes, sm.add_constant(survey[names])).fit(
        disp=False, tol=1e-14, maxiter=200, method="newton", cov_type="HC0")
    robust = oe.mlogit(data=survey, y="brand", x=names, base="alpha", covariance="robust")
    n = len(survey)
    assert_allclose(np.array(robust.covariance_matrix),
                    sandwich.cov_params().to_numpy() * n / (n - 1), rtol=1e-7)
    # Another base only reparameterizes: b_j(new base) = b_j - b_new, same likelihood.
    default = oe.mlogit(data=survey, y="brand", x=names)
    by_term = dict(zip(terms(result), table(result)["estimate"], strict=True))
    for label in ("alpha", "beta", "gamma"):
        for term in ("Intercept", *names):
            old = by_term.get(f"{label}:{term}", 0.0) - by_term[f"delta:{term}"]
            new = dict(zip(terms(default), table(default)["estimate"], strict=True))
            assert_allclose(new[f"{label}:{term}"], old, rtol=1e-7, atol=1e-10)
    assert_allclose(default.metrics["log_likelihood"], result.metrics["log_likelihood"],
                    rtol=1e-12)
    assert_allclose(default.tests["model"]["statistic"], result.tests["model"]["statistic"],
                    rtol=1e-9)


@pytest.mark.parametrize("weight_type", ["fweight", "aweight", "iweight", "pweight"])
def test_mlogit_weights_against_brute_force(survey, weight_type):
    names = ["x1", "x2"]
    x = design(survey, names, constant=True)
    oracle = mlogit_oracle(survey, "brand", x, weight_type=weight_type)
    kinds = ("robust", "cluster") if weight_type == "pweight" else KINDS
    for kind in kinds:
        result = oe.mlogit(data=survey, y="brand", x=names, weights=WEIGHT_COLUMN[weight_type],
                           weight_type=weight_type, covariance=None if kind == "cluster" else kind,
                           cluster="village" if kind == "cluster" else None)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=oracle["nobs"], frequency=oracle["frequency"],
                                      opg_weights=oracle["score_weights"],
                                      clusters=survey.village)
        # The base is the category with the largest WEIGHTED frequency.
        assert result.extra["base"] == oracle["base"]
        check_table(result, oracle["theta"], covariance)
        assert result.nobs == oracle["nobs"]
        assert_allclose(result.metrics["log_likelihood"], oracle["ll"], rtol=1e-11)
        assert_allclose(result.metrics["pseudo_r_squared"], 1 - oracle["ll"] / oracle["null"],
                        rtol=1e-9)
        assert_allclose(result.metrics["bic"], -2 * oracle["ll"] + 9 * math.log(oracle["nobs"]),
                        rtol=1e-11)
    if weight_type == "fweight":
        for kind in KINDS:
            options = {"covariance": None if kind == "cluster" else kind,
                       "cluster": "village" if kind == "cluster" else None}
            same_fit(oe.mlogit(data=survey, y="brand", x=names, weights="fw",
                               weight_type="fweight", **options),
                     oe.mlogit(data=expand(survey, "fw"), y="brand", x=names, **options))
    if weight_type == "pweight":
        default = oe.mlogit(data=survey, y="brand", x=names, weights="pw", weight_type="pweight")
        assert default.spec.covariance == "robust"
        other = oe.mlogit(data=survey.assign(pw=survey.pw / 123.0), y="brand", x=names,
                          weights="pw", weight_type="pweight")
        same_fit(default, other, rtol=1e-8, metrics=False)


def test_mlogit_weighted_base_differs_from_the_row_count_base():
    # 'a' has more rows, 'b' more weight: Stata's base is the largest weighted frequency.
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({"y": ["a"] * 60 + ["b"] * 40 + ["c"] * 30, "x": rng.normal(size=130)})
    frame["w"] = np.where(frame.y == "b", 3.0, 1.0)
    assert oe.mlogit(data=frame, y="y", x=["x"]).extra["base"] == "a"
    weighted = oe.mlogit(data=frame, y="y", x=["x"], weights="w", weight_type="fweight")
    assert weighted.extra["base"] == "b"
    assert terms(weighted) == ["a:Intercept", "a:x", "c:Intercept", "c:x"]
    # Ties go to the first category in sorted order.
    tie = pd.DataFrame({"y": [3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 2, 3], "x": rng.normal(size=12)})
    assert oe.mlogit(data=tie, y="y", x=["x"]).extra["base"] == 1


def test_mlogit_equivalences_and_invariances(survey):
    names = ["x1", "x2", "x3"]
    # Two categories: the binary logit of the non-base category.
    binary = survey.assign(high=(survey.health >= 20).astype(float))
    reference = sm.Logit(binary.high, sm.add_constant(binary[names])).fit(disp=False, tol=1e-13)
    result = oe.mlogit(data=binary, y="high", x=names, base=0)
    assert terms(result) == ["1:Intercept", "1:x1", "1:x2", "1:x3"]
    assert_allclose(table(result)["estimate"], reference.params, rtol=1e-8)
    assert_allclose(np.array(result.covariance_matrix), reference.cov_params(), rtol=1e-7)
    assert_allclose(result.metrics["log_likelihood"], reference.llf, rtol=1e-12)
    assert_allclose(result.tests["model"]["statistic"], reference.llr, rtol=1e-9)
    flipped = oe.mlogit(data=binary, y="high", x=names, base=1)
    assert_allclose(table(flipped)["estimate"], -reference.params, rtol=1e-8)
    assert terms(flipped)[0] == "0:Intercept"
    # Row order, label type and regressor scale do not matter.
    base = oe.mlogit(data=survey, y="brand", x=names, cluster="village")
    shuffled = survey.sample(frac=1.0, random_state=11).reset_index(drop=True)
    same_fit(base, oe.mlogit(data=shuffled, y="brand", x=names, cluster="village"))
    numbered = survey.assign(brand=survey.brand.map({"alpha": 1, "beta": 2, "delta": 3,
                                                     "gamma": 4}))
    other = oe.mlogit(data=numbered, y="brand", x=names, cluster="village")
    assert other.extra["base"] == 3 and terms(other)[:2] == ["1:Intercept", "1:x1"]
    assert_allclose(table(other)["estimate"], table(base)["estimate"], rtol=1e-9)
    assert_allclose(np.array(other.covariance_matrix), np.array(base.covariance_matrix),
                    rtol=1e-8)
    scaled = survey.assign(x1=survey.x1 * 1e8, x3=survey.x3 * 1e-8)
    other = oe.mlogit(data=scaled, y="brand", x=names, cluster="village")
    factor = np.tile([1.0, 1e-8, 1.0, 1e8], 3)
    assert_allclose(table(other)["estimate"], table(base)["estimate"] * factor, rtol=1e-6)
    assert_allclose(table(other)["std_error"], table(base)["std_error"] * factor, rtol=1e-6)
    assert_allclose(other.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-11)
    assert_allclose(other.tests["model"]["statistic"], base.tests["model"]["statistic"],
                    rtol=1e-6)


def test_mlogit_without_constant_categoricals_missing_and_collinearity(survey):
    names = ["x1", "x3"]
    x = design(survey, names)
    oracle = mlogit_oracle(survey, "brand", x)
    result = oe.mlogit(data=survey, y="brand", x=names, intercept=False)
    check_table(result, oracle["theta"],
                stata_covariance("nonrobust", oracle["hessian"], oracle["scores"],
                                 nobs=len(survey)))
    assert terms(result) == ["alpha:x1", "alpha:x3", "beta:x1", "beta:x3", "gamma:x1",
                             "gamma:x3"]
    # No constant: the null model is b = 0, every category equally likely, 6 restrictions.
    null = -len(survey) * math.log(4)
    assert_allclose(result.extra["null_log_likelihood"], null, rtol=1e-12)
    check_chi2(result.tests["model"], 2 * (oracle["ll"] - null), 6)
    # Categorical regressor = hand-made dummies.
    dummies = {"region": ["north", "south", "west"]}
    oracle = mlogit_oracle(survey, "brand", design(survey, ["x1"], constant=True,
                                                   dummies=dummies))
    result = oe.mlogit(data=survey, y="brand", x=["x1", "region"], categorical=["region"])
    assert terms(result)[:4] == ["alpha:Intercept", "alpha:x1", "alpha:region[south]",
                                 "alpha:region[west]"]
    check_table(result, oracle["theta"],
                stata_covariance("nonrobust", oracle["hessian"], oracle["scores"],
                                 nobs=len(survey)))
    check_chi2(result.tests["model"], 2 * (oracle["ll"] - oracle["null"]), 9)
    # Missing values and exact collinearity.
    rng = np.random.default_rng(12)
    holes = survey.copy()
    holes["brand"] = holes.brand.astype(object)
    holes.loc[rng.choice(len(holes), 20, replace=False), "brand"] = None
    holes.loc[rng.choice(len(holes), 20, replace=False), "x3"] = np.nan
    complete = holes.dropna(subset=["brand", "x1", "x3"]).reset_index(drop=True)
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=holes, y="brand", x=names)
    assert caught.value.code == "missing_values"
    dropped = oe.mlogit(data=holes, y="brand", x=names, missing="drop")
    same_fit(dropped, oe.mlogit(data=complete, y="brand", x=names))
    assert dropped.nobs == len(complete)
    extended = survey.assign(twice=2 * survey.x1, one=3.0)
    full = oe.mlogit(data=extended, y="brand", x=["x1", "twice", "one", "x3"])
    same_fit(full, oe.mlogit(data=survey, y="brand", x=names))
    assert full.provenance["omitted_terms"] == ["twice", "one"]


# ---- two-way clustering and a second look at derivatives --------------------------------


def test_two_way_clustering_is_inclusion_exclusion_of_cluster_sums(survey):
    names = ["x1", "x2", "x3"]
    data = survey.assign(month=np.arange(len(survey)) % 12)
    x = design(data, names)
    oracle = ordered_oracle(data, "health", x, "logit")
    bread = np.linalg.inv(-oracle["hessian"])

    def meat(labels):
        index = np.unique(labels, return_inverse=True)[1]
        sums = np.zeros((index.max() + 1, x.shape[1] + 3))
        np.add.at(sums, index, oracle["scores"])
        return sums.T @ sums

    both = data.village.to_numpy() * 100 + data.month.to_numpy()
    smallest = min(data.village.nunique(), 12)
    expected = smallest / (smallest - 1) * bread @ (
        meat(data.village.to_numpy()) + meat(data.month.to_numpy()) - meat(both)) @ bread
    result = oe.ologit(data=data, y="health", x=names, cluster=["village", "month"])
    assert result.inference["psd_adjusted"] is False
    check_table(result, oracle["theta"], expected)
    assert result.inference["cluster_columns"] == ["village", "month"]
    assert result.inference["cluster_count"] == smallest
    check_chi2(result.tests["model"], wald(oracle["theta"], expected, range(3)), 3, rtol=1e-5)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_ordered_kernel_derivatives_against_this_files_calculus(survey, link):
    import torch

    from openecon.econometrics.discrete.kernels import MultinomialObjective, OrderedObjective

    x = design(survey, ["x1", "x2", "x3"])
    codes = np.unique(survey.health.to_numpy(), return_inverse=True)[1]
    weights = survey.aw.to_numpy()
    offset = survey.exposure.to_numpy()
    objective = OrderedObjective(torch.from_numpy(x), torch.from_numpy(codes), 4,
                                 torch.from_numpy(weights), torch.from_numpy(offset), link)
    theta = np.array([0.4, -0.3, 0.2, -0.6, 0.5, 1.9])                 # not a stationary point

    def total(point):
        return float((weights * ordered_loglik(point, x, codes, link, offset)).sum())

    value, gradient, hessian = objective(torch.from_numpy(theta))
    assert_allclose(float(value), total(theta), rtol=1e-12)
    assert_allclose(gradient.numpy(), jacobian(total, theta), rtol=1e-8, atol=1e-8)
    assert_allclose(hessian.numpy(), hessian_of(total, theta), rtol=1e-6, atol=1e-6)
    rows = objective.score_rows(torch.from_numpy(theta)).numpy()
    assert_allclose(rows, jacobian(lambda p: ordered_loglik(p, x, codes, link, offset), theta),
                    rtol=1e-7, atol=1e-9)
    # Cutpoints out of order have no likelihood.
    assert float(objective.value(torch.tensor([0.4, -0.3, 0.2, 0.5, 0.5, 1.9],
                                              dtype=torch.float64))) == -math.inf
    if link == "probit":
        return
    labels, brands = np.unique(survey.brand.to_numpy(), return_inverse=True)
    xc = design(survey, ["x1", "x2"], constant=True)
    multinomial = MultinomialObjective(torch.from_numpy(xc), torch.from_numpy(brands), 4, 2,
                                       torch.from_numpy(weights))
    point = np.linspace(-0.5, 0.6, 9)

    def total_m(p):
        return float((weights * mlogit_loglik(p, xc, brands, 4, 2)).sum())

    value, gradient, hessian = multinomial(torch.from_numpy(point))
    assert_allclose(float(value), total_m(point), rtol=1e-12)
    assert_allclose(gradient.numpy(), jacobian(total_m, point), rtol=1e-8, atol=1e-8)
    assert_allclose(hessian.numpy(), hessian_of(total_m, point), rtol=1e-6, atol=1e-6)
    assert_allclose(multinomial.score_rows(torch.from_numpy(point)).numpy(),
                    jacobian(lambda p: mlogit_loglik(p, xc, brands, 4, 2), point), rtol=1e-7,
                    atol=1e-9)
