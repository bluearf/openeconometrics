"""glm family: invariances, equivalences between estimators and adversarial inputs.

Every estimator must give the same answer for the same statistical problem however it
is posed (row order, units of the regressors, frequency weights versus repeated rows,
one command versus its equivalent in another), and every invalid input must end in an
``AnalysisError`` with a code and a message, never in a traceback, a NaN or a silently
different model.
"""

import json
import math
import warnings
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle

X = ["x1", "x2"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(909)
    n = 300
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    frame["g"] = np.repeat(np.arange(25), 12)
    frame["t"] = rng.integers(0, 6, size=n)
    mu = np.exp(0.4 + 0.3 * frame.x1 - 0.25 * frame.x2 + 0.03 * frame.g - 0.05 * frame.t)
    frame["cnt"] = rng.poisson(mu).astype(float)
    frame["over"] = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    frame["pos"] = rng.gamma(3.0, mu / 3.0)
    p = 1 / (1 + np.exp(-(0.2 + 0.6 * frame.x1 - 0.4 * frame.x2)))
    frame["bin"] = rng.binomial(1, p).astype(float)
    frame["m"] = rng.integers(1, 6, size=n).astype(float)
    frame["succ"] = rng.binomial(frame.m.astype(int), p).astype(float)
    frame["beta"] = rng.beta(p * 7, (1 - p) * 7).clip(1e-6, 1 - 1e-6)
    frame["frac"] = np.where(rng.uniform(size=n) < 0.1, 1.0, frame.beta)
    frame["fw"] = rng.integers(1, 4, size=n).astype(float)
    frame["aw"] = rng.uniform(0.3, 3.0, size=n)
    frame["expo"] = rng.uniform(0.5, 3.0, size=n)
    frame["h"] = rng.integers(0, 20, size=n)            # a second, crossed cluster dimension
    return frame


# (name, call, keyword arguments, weight types that the estimator accepts)
ESTIMATORS = {
    "glm-gamma": (oe.glm, {"y": "pos", "x": X, "family": "gamma", "link": "log"},
                  ("fweight", "aweight", "iweight", "pweight")),
    "glm-binomial": (oe.glm, {"y": "succ", "x": X, "family": "binomial", "trials": "m",
                              "link": "probit"},
                     ("fweight", "aweight", "iweight", "pweight")),
    "poisson": (oe.poisson, {"y": "cnt", "x": X, "exposure": "expo"},
                ("fweight", "aweight", "iweight", "pweight")),
    "nbreg": (oe.nbreg, {"y": "over", "x": X}, ("fweight", "aweight", "iweight", "pweight")),
    "nbreg-constant": (oe.nbreg, {"y": "over", "x": X, "dispersion": "constant"},
                       ("fweight", "aweight", "iweight", "pweight")),
    "cloglog": (oe.cloglog, {"y": "bin", "x": X}, ("fweight", "iweight", "pweight")),
    "fracreg": (oe.fracreg, {"y": "frac", "x": X, "link": "probit"},
                ("fweight", "aweight", "iweight", "pweight")),
    "betareg": (oe.betareg, {"y": "beta", "x": X, "scale": ["x2"]},
                ("fweight", "iweight", "pweight")),
    "ppmlhdfe": (oe.ppmlhdfe, {"y": "cnt", "x": X, "absorb": ["g", "t"], "tolerance": 1e-12},
                 ("fweight", "aweight", "pweight")),
}
NAMES = list(ESTIMATORS)


def run(name, frame, **extra):
    function, kwargs, _ = ESTIMATORS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(data=frame, **{**kwargs, **extra})


def coefs(result):
    return np.array([c.estimate for c in result.coefficients])


def ses(result):
    return np.array([c.std_error for c in result.coefficients])


def cov(result):
    return np.asarray(result.covariance_matrix)


def covariances(name):
    return ("robust", "cluster", "nonrobust") if name == "ppmlhdfe" else (
        "nonrobust", "opg", "robust", "cluster")


def chosen(covariance):
    return {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}


def assert_same_fit(first, second, *, rtol=1e-7, metrics=True):
    assert [c.term for c in first.coefficients] == [c.term for c in second.coefficients]
    assert_allclose(coefs(first), coefs(second), rtol=rtol, atol=1e-9)
    assert_allclose(cov(first), cov(second), rtol=20 * rtol, atol=1e-11)
    assert first.nobs == second.nobs
    if metrics:
        for key, value in first.metrics.items():
            if key == "iterations" or value is None:
                continue
            assert value == pytest.approx(second.metrics[key], rel=20 * rtol, abs=1e-8), key
        for key, test in first.tests.items():
            assert test["df"] == second.tests[key]["df"]
            assert test["statistic"] == pytest.approx(second.tests[key]["statistic"],
                                                      rel=1e-5, abs=1e-8), key


# ---- invariances ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_row_order_does_not_matter(data, name):
    order = np.random.default_rng(1).permutation(len(data))
    shuffled = data.iloc[order].reset_index(drop=True)
    for covariance in covariances(name):
        first = run(name, data, **chosen(covariance))
        second = run(name, shuffled, **chosen(covariance))
        assert_same_fit(first, second)
        assert sorted(second.sample_positions) == list(range(len(data)))


@pytest.mark.parametrize("name", NAMES)
def test_units_of_the_regressors_do_not_matter(data, name):
    """x1 in units of 1e-8 and x2 in units of 1e8: coefficients and errors rescale exactly."""
    scaled = data.assign(x1=data.x1 * 1e8, x2=data.x2 * 1e-8)
    if name == "betareg":
        scaled["x2"] = data.x2                  # x2 is also the precision regressor
    factors = {"x1": 1e-8, "x2": 1.0 if name == "betareg" else 1e8}
    for covariance in covariances(name)[:2]:
        first = run(name, data, **chosen(covariance))
        second = run(name, scaled, **chosen(covariance))
        scale = np.array([factors.get(c.term, 1.0) for c in first.coefficients])
        assert_allclose(coefs(second), coefs(first) * scale, rtol=1e-6, atol=1e-12)
        assert_allclose(ses(second), ses(first) * scale, rtol=1e-5)
        assert_allclose([c.statistic for c in second.coefficients],
                        [c.statistic for c in first.coefficients], rtol=1e-5, atol=1e-7)
        if "log_likelihood" in first.metrics:
            assert second.metrics["log_likelihood"] == pytest.approx(
                first.metrics["log_likelihood"], rel=1e-9)
        assert second.tests["model"]["statistic"] == pytest.approx(
            first.tests["model"]["statistic"], rel=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_frequency_weights_equal_repeated_rows(data, name):
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for covariance in covariances(name):
        weighted = run(name, data, weights="fw", weight_type="fweight", **chosen(covariance))
        repeated = run(name, expanded, **chosen(covariance))
        assert weighted.nobs == int(data.fw.sum())
        assert_same_fit(weighted, repeated)
        assert_allclose([c.p_value for c in weighted.coefficients],
                        [c.p_value for c in repeated.coefficients], rtol=1e-4, atol=1e-300)


@pytest.mark.parametrize("name", NAMES)
def test_weight_scale_conventions(data, name):
    """aweights and pweights are scale free; iweights scale the information."""
    kinds = ESTIMATORS[name][2]
    rescaled = data.assign(aw=data.aw * 7.3)
    if "aweight" in kinds:
        for covariance in covariances(name):
            first = run(name, data, weights="aw", weight_type="aweight", **chosen(covariance))
            second = run(name, rescaled, weights="aw", weight_type="aweight",
                         **chosen(covariance))
            assert_same_fit(first, second)
            assert first.nobs == len(data)
    # pweights: robust is the default, the conventional covariances are refused.
    default = run(name, data, weights="aw", weight_type="pweight")
    assert default.spec.covariance == "robust"
    for covariance in ("robust", "cluster"):
        first = run(name, data, weights="aw", weight_type="pweight", **chosen(covariance))
        second = run(name, rescaled, weights="aw", weight_type="pweight", **chosen(covariance))
        assert_same_fit(first, second, metrics=False)
    for covariance in set(covariances(name)) - {"robust", "cluster"}:
        with pytest.raises(AnalysisError) as error:
            run(name, data, weights="aw", weight_type="pweight", covariance=covariance)
        assert error.value.code == "unsupported_covariance"
    if "iweight" in kinds:
        first = run(name, data, weights="aw", weight_type="iweight", covariance="nonrobust")
        second = run(name, rescaled, weights="aw", weight_type="iweight",
                     covariance="nonrobust")
        assert_allclose(coefs(second), coefs(first), rtol=1e-7, atol=1e-9)
        if name != "glm-gamma":        # (the gamma scale is Pearson/df, itself linear in w)
            assert_allclose(cov(second), cov(first) / 7.3, rtol=1e-6, atol=1e-12)
        integer = run(name, data, weights="fw", weight_type="iweight")
        frequency = run(name, data, weights="fw", weight_type="fweight")
        assert_allclose(coefs(integer), coefs(frequency), rtol=1e-8, atol=1e-10)
        assert integer.nobs == len(data) and frequency.nobs == int(data.fw.sum())
    else:
        with pytest.raises(AnalysisError) as error:
            run(name, data, weights="aw", weight_type="iweight")
        assert error.value.code == "invalid_spec"
    # Zero weights leave the estimation sample (Stata), like deleting the rows.
    holes = data.assign(fw=np.where(np.arange(len(data)) % 7 == 0, 0.0, data.fw))
    dropped = run(name, holes, weights="fw", weight_type="fweight")
    removed = run(name, holes[holes.fw > 0].reset_index(drop=True), weights="fw",
                  weight_type="fweight")
    assert_same_fit(dropped, removed)
    assert any("zero weight" in warning for warning in dropped.warnings)


@pytest.mark.parametrize("name", NAMES)
def test_missing_policy_and_two_way_clustering(data, name):
    holes = data.copy()
    holes.loc[[5, 17, 123], "x1"] = np.nan
    holes.loc[[40], ESTIMATORS[name][1]["y"]] = np.nan
    with pytest.raises(AnalysisError) as error:
        run(name, holes)
    assert error.value.code == "missing_values"
    dropped = run(name, holes, missing="drop")
    complete = run(name, holes.drop(index=[5, 17, 40, 123]).reset_index(drop=True))
    assert_same_fit(dropped, complete)
    assert dropped.dropped_rows == 4 and dropped.nobs_original == len(data)
    assert set(dropped.sample_positions).isdisjoint({5, 17, 40, 123})
    # Two cluster columns: Cameron-Gelbach-Miller V_g + V_h - V_(g x h), with G_min/(G_min-1)
    # for the likelihood estimators (the one-way pieces carry G/(G-1) each).
    both = data.assign(cell=data.g.astype(str) + ":" + data.h.astype(str))
    twoway = run(name, data, cluster=["g", "h"])
    pieces = {key: run(name, both, cluster=key) for key in ("g", "h", "cell")}
    counts = {"g": 25, "h": 20, "cell": both.cell.nunique()}
    assert twoway.inference["cluster_counts"][:2] == [25, 20]
    assert twoway.inference["cluster_count"] == 20 and twoway.inference["use_t"] is False
    if name == "ppmlhdfe":
        return          # reghdfe's factors depend on which effects each cluster nests
    combined = sum(sign * cov(pieces[key]) * (counts[key] - 1) / counts[key]
                   for key, sign in (("g", 1), ("h", 1), ("cell", -1))) * 20 / 19
    if twoway.inference["psd_adjusted"]:
        # Negative eigenvalues of the inclusion-exclusion matrix are set to zero (CGM 2011)
        # and the adjustment is reported.
        values, vectors = np.linalg.eigh(combined)
        assert values.min() < 0
        assert any("positive semidefinite" in warning for warning in twoway.warnings)
    else:
        assert_allclose(cov(twoway), combined, rtol=1e-8, atol=1e-13)


# ---- equivalences between estimators -----------------------------------------------------------


def test_commands_agree_with_their_glm_equivalents(data):
    for covariance in ("nonrobust", "opg", "robust", "cluster"):
        pick = chosen(covariance)
        poisson = oe.poisson(data=data, y="cnt", x=X, exposure="expo", **pick)
        general = oe.glm(data=data, y="cnt", x=X, family="poisson", exposure="expo", **pick)
        assert_allclose(coefs(poisson), coefs(general), rtol=1e-10)
        assert_allclose(cov(poisson), cov(general), rtol=1e-9)
        for key in ("log_likelihood", "deviance", "pearson", "aic", "bic", "df_resid"):
            assert poisson.metrics[key] == pytest.approx(general.metrics[key], rel=1e-12)
        cloglog = oe.cloglog(data=data, y="bin", x=X, **pick)
        general = oe.glm(data=data, y="bin", x=X, family="binomial", link="cloglog", **pick)
        assert_allclose(coefs(cloglog), coefs(general), rtol=1e-10)
        assert_allclose(cov(cloglog), cov(general), rtol=1e-9)
        assert cloglog.metrics["log_likelihood"] == pytest.approx(
            general.metrics["log_likelihood"], rel=1e-12)
        # A 0/1 outcome makes the fractional quasi-likelihood the Bernoulli likelihood.
        for link in ("logit", "probit"):
            fractional = oe.fracreg(data=data, y="bin", x=X, link=link, **pick)
            general = oe.glm(data=data, y="bin", x=X, family="binomial", link=link, **pick)
            assert_allclose(coefs(fractional), coefs(general), rtol=1e-10)
            assert_allclose(cov(fractional), cov(general), rtol=1e-9)
    # Binomial counts with trials are the Bernoulli model on one row per trial.
    rows = []
    for record in data[["x1", "x2", "succ", "m"]].itertuples(index=False):
        rows += [(record.x1, record.x2, 1.0)] * int(record.succ)
        rows += [(record.x1, record.x2, 0.0)] * int(record.m - record.succ)
    trials = pd.DataFrame(rows, columns=["x1", "x2", "d"])
    for link in ("logit", "cloglog"):
        grouped = oe.glm(data=data, y="succ", x=X, family="binomial", trials="m", link=link)
        single = oe.glm(data=trials, y="d", x=X, family="binomial", link=link)
        assert_allclose(coefs(grouped), coefs(single), rtol=1e-9)
        assert_allclose(cov(grouped), cov(single), rtol=1e-8)
        assert grouped.nobs == len(data) and single.nobs == int(data.m.sum())
        # ... the grouped likelihood adds the binomial coefficients ln C(m, y).
        binomial = sum(math.lgamma(m + 1) - math.lgamma(y + 1) - math.lgamma(m - y + 1)
                       for y, m in zip(data.succ, data.m))
        assert grouped.metrics["log_likelihood"] == pytest.approx(
            single.metrics["log_likelihood"] + binomial, rel=1e-11)


def test_gaussian_identity_is_least_squares_with_normal_inference(data):
    design = np.column_stack([np.ones(len(data)), data.x1, data.x2])
    y = data.pos.to_numpy()
    beta, rss = np.linalg.lstsq(design, y, rcond=None)[:2]
    n, k = design.shape
    variance = np.linalg.inv(design.T @ design) * float(rss[0]) / (n - k)
    for optimizer in ("ml", "irls"):
        result = oe.glm(data=data, y="pos", x=X, optimizer=optimizer)
        assert_allclose(coefs(result), beta, rtol=1e-10)
        assert_allclose(cov(result), variance, rtol=1e-9)
        assert result.metrics["scale"] == pytest.approx(float(rss[0]) / (n - k), rel=1e-11)
        assert result.metrics["log_likelihood"] == pytest.approx(
            -0.5 * n * (1 + np.log(2 * np.pi * float(rss[0]) / n)), rel=1e-12)
        # z, not t: Stata's glm uses the normal distribution.
        assert_allclose([c.p_value for c in result.coefficients],
                        2 * stats.norm.sf(np.abs(beta / np.sqrt(np.diag(variance)))),
                        rtol=1e-7, atol=1e-300)
    regress = oe.regress(data=data, y="pos", x=X)
    assert_allclose(ses(result), [c.std_error for c in regress.coefficients], rtol=1e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(regress.metrics["log_likelihood"],
                                                             rel=1e-11)


def test_ml_and_irls_agree_and_canonical_links_share_their_information(data):
    cases = [("poisson", "log", "cnt", True), ("binomial", "logit", "bin", True),
             ("gamma", "reciprocal", "pos", True), ("gaussian", "identity", "pos", True),
             ("gamma", "log", "pos", False), ("binomial", "probit", "bin", False),
             ("poisson", "identity", "cnt", False)]
    for family, link, outcome, canonical in cases:
        newton = oe.glm(data=data, y=outcome, x=X, family=family, link=link)
        scoring = oe.glm(data=data, y=outcome, x=X, family=family, link=link, optimizer="irls")
        assert_allclose(coefs(newton), coefs(scoring), rtol=1e-8)
        assert newton.metrics["deviance"] == pytest.approx(scoring.metrics["deviance"],
                                                           rel=1e-12)
        difference = np.abs(cov(newton) / cov(scoring) - 1).max()
        assert (difference < 1e-7) if canonical else (difference > 1e-4)


# ---- adversarial inputs -----------------------------------------------------------------------


def expect(code, function, **kwargs):
    with pytest.raises(AnalysisError) as error:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            function(**kwargs)
    assert error.value.code == code, (error.value.code, str(error.value))
    assert len(str(error.value)) > 20           # a sentence that says what to change
    return str(error.value)


@pytest.mark.parametrize("name", NAMES)
def test_degenerate_samples_and_columns_raise_analysis_errors(data, name):
    function, kwargs, kinds = ESTIMATORS[name]
    base = {"data": data, **kwargs}
    y = kwargs["y"]
    expect("empty_data", function, **{**base, "data": data.iloc[:0]})
    for rows in (1, 2, 3):
        with pytest.raises(AnalysisError) as error:
            run(name, data.iloc[:rows].reset_index(drop=True))
        assert error.value.code in {"insufficient_observations", "constant_outcome",
                                    "empty_sample", "separation_detected",
                                    "boundary_solution", "invalid_binomial_outcome"}
    expect("empty_sample", function,
           **{**base, "data": data.assign(x1=np.nan), "missing": "drop"})
    expect("missing_values", function, **{**base, "data": data.assign(x1=np.nan)})
    expect("missing_columns", function, **{**base, "x": ["x1", "nope"]})
    expect("non_numeric_column", function, **{**base, "data": data.assign(x1="text")})
    expect("non_numeric_column", function, **{**base, "data": data.assign(**{y: "text"})})
    expect("non_finite_values", function,
           **{**base, "data": data.assign(x1=np.where(np.arange(len(data)) == 3, np.inf,
                                                      data.x1))})
    expect("invalid_spec", function, **{**base, "x": "x1"})
    expect("invalid_spec", function, **{**base, "x": ["x1", "x1"]})
    expect("invalid_spec", function, **{**base, "x": ["x1", y]})
    expect("invalid_spec", function, **{**base, "covariance": "HC3"})
    expect("invalid_spec", function, **{**base, "alpha": 0.0})
    expect("invalid_spec", function, **{**base, "alpha": 1.0})
    expect("invalid_spec", function, **{**base, "missing": "ignore"})
    expect("invalid_spec", function, **{**base, "weights": "aw"})                 # no type
    expect("invalid_spec", function, **{**base, "weights": "aw", "weight_type": "bad"})
    expect("invalid_spec", function, **{**base, "cluster": ["g", "t", "m"]})
    expect("insufficient_clusters", function, **{**base, "cluster": "one",
                                                 "data": data.assign(one=1)})
    expect("negative_weights", function, **{**base, "weights": "neg", "weight_type": "fweight",
                                            "data": data.assign(neg=-1.0)})
    expect("noninteger_frequency_weights", function,
           **{**base, "weights": "aw", "weight_type": "fweight"})
    expect("empty_sample", function, **{**base, "weights": "zero", "weight_type": "fweight",
                                        "data": data.assign(zero=0.0)})
    expect("non_numeric_column", function, **{**base, "weights": "label",
                                              "weight_type": "fweight",
                                              "data": data.assign(label="w")})
    expect("constant_predictor", function,
           **{**base, "x": ["x1", "label"], "categorical": ["label"],
              "data": data.assign(label="only")})
    if "iweight" in kinds:
        expect("negative_weights", function, **{**base, "weights": "neg",
                                                "weight_type": "iweight",
                                                "data": data.assign(neg=-data.aw)})


@pytest.mark.parametrize("name", NAMES)
def test_collinear_and_constant_regressors_are_omitted_and_recorded(data, name):
    extra = data.assign(x3=2 * data.x1 - data.x2 + 5, k=3.0)
    base = run(name, data)
    result = run(name, extra, x=["x1", "x2", "x3", "k"])
    assert_same_fit(result, base)
    assert sorted(result.provenance["omitted_terms"]) == ["k", "x3"]
    assert sum("Omitted" in warning for warning in result.warnings) >= 1


def test_outcomes_outside_the_support_are_rejected(data):
    negative = data.assign(y=data.cnt - 1)
    for function, extra in ((oe.poisson, {}), (oe.nbreg, {}), (oe.ppmlhdfe, {"absorb": ["g"]}),
                            (oe.glm, {"family": "poisson"}), (oe.glm, {"family": "nbinomial"})):
        expect("invalid_count_outcome", function, data=negative, y="y", x=X, **extra)
    for family in ("gamma", "inverse_gaussian"):
        message = expect("invalid_positive_outcome", oe.glm, data=data, y="cnt", x=X,
                         family=family)
        assert "strictly positive" in message
    expect("invalid_binary_outcome", oe.glm, data=data, y="cnt", x=X, family="binomial")
    expect("invalid_binary_outcome", oe.cloglog, data=data, y="frac", x=X)
    expect("invalid_fractional_outcome", oe.fracreg, data=data, y="cnt", x=X)
    message = expect("invalid_fractional_outcome", oe.betareg, data=data, y="frac", x=X)
    assert "fracreg" in message                       # y == 1 occurs: points to fracreg
    expect("invalid_binomial_outcome", oe.glm, data=data.assign(y=data.m + 1), y="y", x=X,
           family="binomial", trials="m")
    expect("invalid_binomial_outcome", oe.glm, data=data.assign(y=data.succ + 0.5), y="y", x=X,
           family="binomial", trials="m")
    expect("invalid_trials", oe.glm, data=data.assign(m=data.m - 1), y="succ", x=X,
           family="binomial", trials="m")
    expect("invalid_trials", oe.glm, data=data.assign(m=data.m + 0.5), y="succ", x=X,
           family="binomial", trials="m")
    expect("invalid_spec", oe.glm, data=data, y="pos", x=X, trials="m")
    # Constant outcomes: nothing to estimate (or a degenerate likelihood).
    zero = data.assign(y=0.0)
    for function, extra in ((oe.poisson, {}), (oe.nbreg, {}), (oe.ppmlhdfe, {"absorb": ["g"]})):
        expect("constant_outcome", function, data=zero, y="y", x=X, **extra)
    for value in (0.0, 1.0):
        expect("constant_outcome", oe.cloglog, data=data.assign(y=value), y="y", x=X)
        expect("constant_outcome", oe.glm, data=data.assign(y=value), y="y", x=X,
               family="binomial")
        expect("constant_outcome", oe.fracreg, data=data.assign(y=value), y="y", x=X)
    expect("constant_outcome", oe.fracreg, data=data.assign(y=0.4), y="y", x=X)
    expect("constant_outcome", oe.betareg, data=data.assign(y=0.4), y="y", x=X)
    expect("perfect_fit", oe.glm, data=data.assign(y=3.0), y="y", x=X)
    expect("perfect_fit", oe.glm, data=data.assign(y=3.0), y="y", x=X, family="gamma")
    exact = data.assign(y=1 + 2 * data.x1 - data.x2)
    expect("perfect_fit", oe.glm, data=exact, y="y", x=X)
    expect("perfect_fit", oe.glm, data=exact, y="y", x=X, scale=1.0)   # unbounded likelihood
    # Not overdispersed: the negative binomial likelihood has no interior maximum.
    message = expect("boundary_solution", oe.nbreg, data=data.assign(y=2.0), y="y", x=X)
    assert "oe.poisson" in message
    # A constant count is a legitimate (if dull) Poisson sample: slope zero, deviance zero.
    flat = oe.poisson(data=data.assign(y=2.0), y="y", x=X)
    assert_allclose(coefs(flat), [math.log(2.0), 0.0, 0.0], atol=1e-10)
    # Stable deviance retains the tiny, mathematically positive difference
    # between the fitted float64 mean and exactly 2 instead of cancelling it.
    with localcontext() as context:
        context.prec = 70
        count = Decimal(2)
        exact_deviance = sum(
            2 * (count * (count / Decimal.from_float(row["fitted"])).ln()
                 - (count - Decimal.from_float(row["fitted"])))
            for row in flat.predictions
        )
    assert flat.metrics["deviance"] == pytest.approx(float(exact_deviance), rel=1e-13, abs=1e-40)
    assert math.copysign(1, flat.metrics["deviance"]) == 1
    assert flat.metrics["pseudo_r_squared"] == pytest.approx(0.0, abs=1e-12)


def test_option_bounds_and_inconsistent_options(data):
    base = {"data": data, "y": "pos", "x": X}
    for bad in (0.0, 1.0, -1e-3):
        expect("invalid_option", oe.glm, **base, family="gamma", link="log", tolerance=bad)
        expect("invalid_option", oe.ppmlhdfe, data=data, y="cnt", x=X, absorb=["g"],
               tolerance=bad)
    for bad in (0, -3):
        expect("invalid_spec", oe.glm, **base, max_iterations=bad)
        expect("invalid_spec", oe.ppmlhdfe, data=data, y="cnt", x=X, absorb=["g"],
               max_iterations=bad)
    for bad in (0, -1.0, "pearson", True, float("nan")):
        expect("invalid_option", oe.glm, **base, scale=bad)
    expect("invalid_option", oe.glm, **base, family="gamma", power=0.5)
    expect("invalid_option", oe.glm, **base, family="gamma", link="power")
    expect("invalid_spec", oe.glm, **base, family="gamma", link="power", power=float("inf"))
    expect("invalid_option", oe.glm, **base, family="gamma", link="logit")
    expect("invalid_option", oe.glm, **base, family="gamma", link="nbinomial")
    expect("invalid_option", oe.glm, **base, family="gamma", dispersion=2.0)
    for bad in (0.0, -1.0):
        expect("invalid_option", oe.glm, data=data, y="cnt", x=X, family="nbinomial",
               dispersion=bad)
    for name in ("family", "link", "optimizer"):
        expect("invalid_spec", oe.glm, **base, **{name: "nonsense"})
    expect("invalid_spec", oe.nbreg, data=data, y="over", x=X, dispersion="nb3")
    expect("invalid_spec", oe.fracreg, data=data, y="frac", x=X, link="cloglog")
    expect("invalid_spec", oe.betareg, data=data, y="beta", x=X, link="log")
    expect("invalid_spec", oe.betareg, data=data, y="beta", x=X, scale_link="logit")
    expect("invalid_spec", oe.betareg, data=data, y="beta", x=X, scale="x2")
    expect("invalid_spec", oe.cloglog, data=data, y="bin", x=X, weights="aw",
           weight_type="aweight")                         # Stata's cloglog has no aweights
    expect("invalid_spec", oe.betareg, data=data, y="beta", x=X, weights="aw",
           weight_type="aweight")
    expect("invalid_spec", oe.ppmlhdfe, data=data, y="cnt", x=X, absorb="g")
    expect("invalid_spec", oe.ppmlhdfe, data=data, y="cnt", x=X, absorb=[])
    expect("invalid_spec", oe.ppmlhdfe, data=data, y="cnt", x=X, absorb=["g"],
           covariance="opg")
    # Offsets and exposures.
    for function, extra in ((oe.poisson, {}), (oe.nbreg, {}), (oe.ppmlhdfe, {"absorb": ["g"]}),
                            (oe.glm, {"family": "poisson"})):
        outcome = "over" if function is oe.nbreg else "cnt"
        expect("invalid_spec", function, data=data, y=outcome, x=X, offset="x1",
               exposure="expo", **extra)
        expect("invalid_exposure", function, data=data.assign(e=data.expo - 1.0), y=outcome,
               x=X, exposure="e", **extra)
    # The power link at its named special cases, and the bounds of the iteration limit.
    log_link = oe.glm(**base, family="gamma", link="log")
    assert_allclose(coefs(oe.glm(**base, family="gamma", link="power", power=0)),
                    coefs(log_link), rtol=1e-12)
    identity = oe.glm(**base, family="gamma", link="identity")
    assert_allclose(coefs(oe.glm(**base, family="gamma", link="power", power=1)),
                    coefs(identity), rtol=1e-12)
    reciprocal = oe.glm(**base, family="gamma", link="reciprocal")
    assert_allclose(coefs(oe.glm(**base, family="gamma", link="power", power=-1)),
                    coefs(reciprocal), rtol=1e-12)
    for optimizer in ("ml", "irls"):
        message = expect("nonconvergence", oe.glm, **base, family="gamma", link="log",
                         optimizer=optimizer, max_iterations=1)
        assert "did not converge" in message
    expect("nonconvergence", oe.ppmlhdfe, data=data, y="cnt", x=X, absorb=["g"],
           max_iterations=1)


def test_separation_is_reported_not_estimated(data):
    separated = data.assign(d=(data.x1 > 0.2).astype(float))
    for optimizer in ("ml", "irls"):
        for link in ("logit", "probit", "cloglog", "loglog"):
            message = expect("separation_detected", oe.glm, data=separated, y="d", x=X,
                             family="binomial", link=link, optimizer=optimizer)
            assert "separat" in message
    expect("separation_detected", oe.cloglog, data=separated, y="d", x=X)
    expect("separation_detected", oe.fracreg, data=separated, y="d", x=X)
    # Quasi-complete separation: a dummy that is 1 only where the outcome is 1.
    quasi = data.assign(d=((data.bin == 1) & (data.x2 > 0.5)).astype(float))
    expect("separation_detected", oe.glm, data=quasi, y="bin", x=["x1", "d"],
           family="binomial")
    # Count models: a regressor that is 1 exactly where the outcome is zero.
    zeros = data.assign(d=(data.cnt == 0).astype(float))
    message = expect("nonconvergence", oe.poisson, data=zeros, y="cnt", x=["x1", "d"])
    assert "separation" in message
    message = expect("separation_detected", oe.ppmlhdfe, data=zeros, y="cnt", x=["x1", "d"],
                     absorb=["g"])
    assert "'d'" in message and "zero outcome" in message
    # ... while the same fit without the offending regressor is fine.
    assert oe.ppmlhdfe(data=zeros, y="cnt", x=["x1"], absorb=["g"]).nobs == len(data)


def test_ppmlhdfe_sample_and_design_degeneracies(data):
    base = {"data": data, "y": "cnt", "x": X}
    expect("missing_columns", oe.ppmlhdfe, **base, absorb=["nope"])
    expect("empty_sample", oe.ppmlhdfe, **base, absorb=["x1"])          # all singletons
    expect("empty_design", oe.ppmlhdfe, data=data.assign(level=data.g * 2.0), y="cnt",
           x=["level"], absorb=["g"])
    expect("insufficient_observations", oe.ppmlhdfe, data=data.iloc[:3], y="cnt", x=X,
           absorb=["g"])
    expect("missing_values", oe.ppmlhdfe,
           data=data.assign(g=np.where(data.g == 2, np.nan, data.g)), y="cnt", x=X, absorb=["g"])
    # A regressor that only varies between the absorbed levels is omitted, not estimated.
    result = oe.ppmlhdfe(data=data.assign(level=data.g * 2.0), y="cnt", x=["x1", "level"],
                         absorb=["g"])
    assert [c.term for c in result.coefficients] == ["x1"]
    assert result.provenance["omitted_terms"] == ["level"]
    # Text labels, one single level (an intercept) and three dimensions all work.
    labelled = data.assign(name="firm" + data.g.astype(str))
    assert_allclose(coefs(oe.ppmlhdfe(data=labelled, y="cnt", x=X, absorb=["name"])),
                    coefs(oe.ppmlhdfe(data=data, y="cnt", x=X, absorb=["g"])), rtol=1e-10)
    constant = oe.ppmlhdfe(data=data.assign(one=1), y="cnt", x=X, absorb=["one"],
                           tolerance=1e-12)
    assert_allclose(coefs(constant), coefs(oe.poisson(data=data, y="cnt", x=X))[1:], rtol=1e-8)
    assert constant.metrics["df_absorbed"] == 1
    three = oe.ppmlhdfe(data=data.assign(q=np.arange(len(data)) % 4), y="cnt", x=X,
                        absorb=["g", "t", "q"])
    assert three.metrics["df_absorbed"] == 25 + 6 + 4 - 2


@pytest.mark.parametrize("name", NAMES)
def test_extreme_magnitudes_of_the_outcome(data, name):
    """Outcomes in units of 1e8 or 1e-8 where the model is equivariant to the unit."""
    if name in {"glm-binomial", "cloglog", "fracreg", "betareg", "nbreg", "nbreg-constant"}:
        pytest.skip("the outcome has no free unit")
    y = ESTIMATORS[name][1]["y"]
    base = run(name, data)
    for factor in (1e8, 1e-8):
        scaled = run(name, data.assign(**{y: data[y] * factor}))
        slopes = [i for i, c in enumerate(base.coefficients) if c.term != "Intercept"]
        assert_allclose(coefs(scaled)[slopes], coefs(base)[slopes], rtol=1e-6, atol=1e-9)
        assert_allclose(ses(scaled)[slopes], ses(base)[slopes] * (
            1 if name != "poisson" or base.spec.covariance != "nonrobust" else factor ** -0.5),
            rtol=1e-5)
        if name != "ppmlhdfe":
            assert coefs(scaled)[0] == pytest.approx(coefs(base)[0] + math.log(factor), rel=1e-7)


# ---- results are finite, serializable and documented ------------------------------------------


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


@pytest.mark.parametrize("name", NAMES)
def test_results_are_finite_serializable_and_self_describing(data, name):
    result = run(name, data, cluster="g")
    payload = json.loads(result.model_dump_json())
    assert finite_json(payload)
    restored = ResultBundle.model_validate(payload)
    assert restored.coefficients == result.coefficients and restored.metrics == result.metrics
    assert restored.tests == result.tests and restored.extra == result.extra
    assert result.provenance["stata_parity_validated"] is False
    assert result.provenance["family"] == "glm" and result.provenance["precision"] == "float64"
    assert result.inference["covariance"] == "cluster" and result.inference["use_t"] is False
    assert result.inference["cluster_count"] == 25
    assert isinstance(result.inference["correction"], str) and result.inference["correction"]
    terms = [c.term for c in result.coefficients]
    assert len(set(terms)) == len(terms)
    assert ("Intercept" in terms) == (name != "ppmlhdfe")
    summary = result.summary()
    assert all(term in summary for term in terms)
    assert len(result.predictions) == min(result.nobs, 400)
    function = ESTIMATORS[name][0]
    doc = function.__doc__
    for word in ("Model", "Parameters", "Result", "Stata", "Example", "covariance", "weights"):
        assert word in doc, (function.__name__, word)
    assert getattr(oe, function.__name__) is function
