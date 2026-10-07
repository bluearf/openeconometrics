"""Adversarial inputs for the count family: a correct fit or a helpful AnalysisError.

Every call below either returns a result whose numbers are all finite (and
JSON-serializable) or raises ``AnalysisError`` with a snake_case code and a
message that says what to change - never a raw exception, never NaN/inf, never
a table of diverging coefficients presented as a converged fit.
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import special

import openecon as oe
from openecon.analysis import AnalysisError

OUTCOME = {"zip": "zip_y", "zinb": "zinb_y", "tpoisson": "trunc_p", "tnbreg": "trunc_nb",
           "churdle": "amount", "hurdle": "zinb_y", "gnbreg": "nb_y"}
NAMES = list(OUTCOME)


def make(n=700, seed=3):
    rng = np.random.default_rng(seed)
    x1, z1 = rng.normal(size=n), rng.normal(size=n)
    x2 = rng.integers(0, 3, size=n).astype(float)
    mu = np.exp(0.7 + 0.4 * x1 - 0.2 * x2)
    excess = rng.uniform(size=n) < special.expit(-0.3 + 0.9 * z1)
    negbin = rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * mu))

    def positive(draw):
        values = draw()
        while (values == 0).any():                           # exact zero-truncated draws
            values = np.where(values == 0, draw(), values)
        return values

    takes_part = 0.2 + 0.7 * z1 + rng.normal(size=n) > 0
    return pd.DataFrame({
        "x1": x1, "x2": x2, "z1": z1, "g": rng.integers(0, 40, size=n),
        "h": rng.integers(0, 35, size=n), "w": rng.uniform(0.5, 2.0, size=n),
        "f": rng.integers(1, 4, size=n),
        "zip_y": np.where(excess, 0, rng.poisson(mu)), "zinb_y": np.where(excess, 0, negbin),
        "nb_y": negbin, "trunc_p": positive(lambda: rng.poisson(mu)),
        "trunc_nb": positive(lambda: rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * mu))),
        "amount": np.where(takes_part, np.exp(1 + 0.5 * x1 + 0.6 * rng.normal(size=n)), 0.0),
    })


@pytest.fixture(scope="module")
def data():
    return make()


def call(name, frame, **options):
    base = {
        "zip": dict(x=["x1", "x2"], inflate=["z1"]),
        "zinb": dict(x=["x1", "x2"], inflate=["z1"]),
        "tpoisson": dict(x=["x1", "x2"]), "tnbreg": dict(x=["x1", "x2"]),
        "churdle": dict(x=["x1", "x2"], select_x=["z1"]),
        "hurdle": dict(x=["x1", "x2"], select_x=["z1"]),
        "gnbreg": dict(x=["x1", "x2"], lnalpha=["z1"]),
    }[name]
    arguments = {"data": frame, "y": OUTCOME[name], **base, **options}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(oe, name)(**arguments)


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, float):
        yield value


def valid(result):
    """A reportable fit: finite numbers everywhere, positive standard errors, JSON-safe."""
    for c in result.coefficients:
        numbers = (c.estimate, c.std_error, c.statistic, c.p_value, c.ci_low, c.ci_high)
        assert all(math.isfinite(v) for v in numbers), c
        assert c.std_error > 0 and 0 <= c.p_value <= 1 and c.ci_low <= c.estimate <= c.ci_high
    assert np.isfinite(np.array(result.covariance_matrix)).all()
    for block in (result.metrics, result.tests, result.extra, result.inference):
        assert all(math.isfinite(v) for v in _numbers(block)), block
    assert math.isfinite(result.metrics["log_likelihood"])
    json.loads(result.model_dump_json())
    return result


def fails(code, name, frame, *words, **options):
    with pytest.raises(AnalysisError) as caught:
        call(name, frame, **options)
    assert caught.value.code == code, (caught.value.code, str(caught.value))
    message = str(caught.value)
    assert len(message) > 25 and "Traceback" not in message
    for word in words:
        assert word in message, (word, message)
    return message


def slope(result, term="x1"):
    return next(c for c in result.coefficients if c.term == term)


# ---- samples --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_empty_tiny_and_missing_samples(data, name):
    fails("empty_data", name, data.iloc[:0])
    with pytest.raises(AnalysisError) as caught:
        call(name, data.iloc[:1])
    assert caught.value.code in {"insufficient_observations", "no_zero_outcomes",
                                 "no_selection_variation", "constant_outcome"}
    with pytest.raises(AnalysisError) as caught:                 # n <= k
        call(name, data.iloc[:3])
    assert caught.value.code in {"insufficient_observations", "separation_detected",
                                 "no_zero_outcomes", "no_selection_variation",
                                 "constant_outcome", "nonconvergence"}
    hollow = data.assign(x1=np.nan)
    fails("missing_values", name, hollow, "missing='drop'")
    fails("empty_sample", name, hollow, missing="drop")
    holes = data.copy()
    holes.loc[::9, "x1"] = np.nan
    holes.loc[5, OUTCOME[name]] = np.nan
    fails("missing_values", name, holes)
    kept = valid(call(name, holes, missing="drop"))
    assert kept.nobs == int(holes[["x1", OUTCOME[name]]].notna().all(axis=1).sum())
    assert any("missing model inputs" in text for text in kept.warnings)
    assert kept.dropped_rows == len(data) - kept.nobs


def test_degenerate_outcomes(data):
    zeros, threes, ones = (data.assign(**{column: value for column in set(OUTCOME.values())})
                           for value in (0, 3, 1))
    for name in ("zip", "zinb", "gnbreg"):
        fails("constant_outcome", name, zeros, "zero in every observation")
    for name in ("tpoisson", "tnbreg"):
        fails("outcome_not_truncated", name, zeros, "truncation point")
        fails("constant_outcome", name, ones, "ll + 1")
    for name in ("churdle", "hurdle"):
        fails("no_selection_variation", name, zeros)
        fails("no_selection_variation", name, threes)
    for name in ("zip", "zinb"):
        fails("no_zero_outcomes", name, threes, "oe.poisson" if name == "zip" else "oe.nbreg")
    # A constant outcome above the truncation point is a legitimate truncated Poisson sample.
    constant = valid(call("tpoisson", threes))
    assert abs(slope(constant).estimate) < 1e-8
    assert constant.tests["model"]["statistic"] == pytest.approx(0.0, abs=1e-8)
    fails("boundary_solution", "tnbreg", threes, "oe.tpoisson")
    fails("boundary_solution", "gnbreg", threes, "oe.poisson")
    fails("constant_outcome", "churdle",
          data.assign(amount=np.where(data.amount > 0, 2.0, 0.0)), "sigma")
    fails("constant_outcome", "hurdle", data.assign(zinb_y=(data.zinb_y > 0).astype(int)))
    for name in NAMES:
        negative = data.assign(**{OUTCOME[name]: data[OUTCOME[name]] - 50.0})
        if name == "churdle":
            fails("no_selection_variation", name, negative)
        else:
            fails("invalid_count_outcome", name, negative, "nonnegative")
    for name in ("tpoisson", "tnbreg", "hurdle"):
        fails("invalid_count_outcome", name,
              data.assign(**{OUTCOME[name]: data[OUTCOME[name]] * 1.5}), "integer")
    fractional = valid(call("gnbreg", data.assign(nb_y=data.nb_y * 1.5)))
    assert any("non-integer" in text for text in fractional.warnings)


@pytest.mark.parametrize("name", NAMES)
def test_collinear_and_constant_columns_are_omitted_and_recorded(data, name):
    frame = data.assign(twin=2 * data.x1 - data.x2, five=5.0, zz=3 * data.z1 + 1)
    reference = call(name, data)
    result = valid(call(name, frame, x=["x1", "x2", "twin", "five"]))
    assert result.provenance["omitted_terms"] == ["twin", "five"]
    assert any("Omitted because of collinearity: twin, five" in text for text in result.warnings)
    assert_allclose([c.estimate for c in result.coefficients],
                    [c.estimate for c in reference.coefficients], rtol=1e-8)
    role = {"zip": "inflate", "zinb": "inflate", "churdle": "select_x", "hurdle": "select_x",
            "gnbreg": "lnalpha"}.get(name)
    if role is not None:
        result = valid(call(name, frame, **{role: ["z1", "zz"]}))
        assert len(result.provenance["omitted_terms"]) == 1
        assert result.provenance["omitted_terms"][0].endswith(":zz")
        assert_allclose([c.estimate for c in result.coefficients],
                        [c.estimate for c in reference.coefficients], rtol=1e-8)
    fails("empty_design", name, data, "intercept=True", x=[], intercept=False)
    fails("constant_predictor", name, data.assign(one="a"), x=["x1", "one"],
          categorical=["one"])


# ---- weights and clusters --------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_bad_weights(data, name):
    index = np.arange(len(data))
    fails("negative_weights", name, data.assign(f=np.where(index == 0, -1, data.f)),
          weights="f", weight_type="fweight")
    fails("noninteger_frequency_weights", name, data, weights="w", weight_type="fweight")
    fails("negative_weights", name, data.assign(w=np.where(index == 3, -1.0, data.w)),
          weights="w", weight_type="iweight")
    fails("empty_sample", name, data.assign(w=0.0), weights="w", weight_type="aweight")
    fails("missing_values", name, data.assign(w=np.where(index % 3 == 0, np.nan, data.w)),
          weights="w", weight_type="aweight")
    fails("non_finite_values", name, data.assign(w=np.where(index == 3, np.inf, data.w)),
          weights="w", weight_type="aweight")
    fails("non_numeric_column", name, data.assign(w="heavy"), weights="w",
          weight_type="aweight")
    fails("unsupported_covariance", name, data, "covariance='robust'", weights="w",
          weight_type="pweight", covariance="nonrobust")
    fails("unsupported_covariance", name, data, weights="w", weight_type="pweight",
          covariance="opg")
    fails("invalid_spec", name, data, "weight_type", weights="w")
    fails("invalid_spec", name, data, weights="w", weight_type="xweight")
    # Zero weights leave the sample (Stata), with a recorded note.
    some = data.assign(w=np.where(index % 4 == 0, 0.0, data.w))
    result = valid(call(name, some, weights="w", weight_type="aweight"))
    assert result.nobs == int((some.w > 0).sum())
    assert any("zero weight" in text for text in result.warnings)
    reference = call(name, some[some.w > 0], weights="w", weight_type="aweight")
    assert_allclose([c.estimate for c in result.coefficients],
                    [c.estimate for c in reference.coefficients], rtol=1e-9)
    # The convenience default for sampling weights is the robust covariance.
    assert call(name, data, weights="w", weight_type="pweight").spec.covariance == "robust"


@pytest.mark.parametrize("name", NAMES)
def test_clusters(data, name):
    fails("insufficient_clusters", name, data.assign(g=1), "at least two", cluster="g")
    fails("missing_values", name,
          data.assign(g=np.where(np.arange(len(data)) < 5, np.nan, data.g)), cluster="g")
    fails("invalid_spec", name, data, "at most 2", cluster=["g", "h", "f"])
    fails("invalid_spec", name, data, "distinct", cluster=["g", "g"])
    fails("invalid_spec", name, data, "cluster column", covariance="cluster")
    fails("invalid_spec", name, data, covariance="robust", cluster="g")
    fails("invalid_spec", name, data, cluster=OUTCOME[name])
    labelled = valid(call(name, data.assign(g=data.g.map(lambda v: f"firm {v}")), cluster="g"))
    numeric = call(name, data, cluster="g")
    assert_allclose(labelled.covariance_matrix, numeric.covariance_matrix, rtol=1e-12)
    assert numeric.inference["cluster_count"] == 40
    few = valid(call(name, data.assign(g=np.arange(len(data)) % 2), cluster="g"))
    assert any("Only 2 clusters" in text for text in few.warnings)
    # Singleton clusters: the cluster sandwich is the robust one up to G/(G-1) = N/(N-1).
    single = valid(call(name, data.assign(g=np.arange(len(data))), cluster="g"))
    robust = call(name, data, covariance="robust")
    assert_allclose(single.covariance_matrix, robust.covariance_matrix, rtol=1e-9)
    valid(call(name, data, cluster=["g", "h"]))


# ---- wrong types, columns and options --------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_wrong_types_and_columns(data, name):
    outcome = OUTCOME[name]
    fails("non_numeric_column", name, data.assign(x1=data.x1.map(str)), "x1")
    fails("non_numeric_column", name, data.assign(**{outcome: "many"}), outcome)
    fails("non_numeric_column", name,
          data.assign(x1=pd.date_range("2020-01-01", periods=len(data))))
    fails("non_finite_values", name,
          data.assign(x1=np.where(np.arange(len(data)) == 3, np.inf, data.x1)))
    fails("missing_columns", name, data, "nope", x=["x1", "nope"])
    fails("missing_columns", name, data.drop(columns=[outcome]), outcome)
    fails("invalid_spec", name, data, "list of column names", x="x1")
    fails("invalid_spec", name, data, x=["x1", outcome])
    fails("invalid_spec", name, data, "duplicate", x=["x1", "x1"])
    fails("invalid_spec", name, data, categorical=["g"])
    fails("invalid_spec", name, data, alpha=0.0)
    fails("invalid_spec", name, data, alpha=1.0)
    fails("invalid_spec", name, data, missing="ignore")
    fails("invalid_spec", name, data, covariance="HC3")
    for bad in (None, "file.csv", 3):
        with pytest.raises(AnalysisError) as caught:
            call(name, bad)
        assert caught.value.code == "invalid_data"
    doubled = pd.concat([data, data[["x1"]]], axis=1)
    fails("duplicate_columns", name, doubled)
    # Other containers and dtypes give the same fit.
    reference = call(name, data)
    for frame in (data.to_dict("list"), data.to_dict("records"),
                  data.astype({"x1": "float32", "x2": "int32"})):
        result = call(name, frame)
        assert_allclose([c.estimate for c in result.coefficients],
                        [c.estimate for c in reference.coefficients], rtol=1e-5)
    wide = valid(call(name, data.assign(x2=data.x2 > 0)))            # a boolean regressor
    assert wide.nobs == reference.nobs


def test_options_at_their_bounds(data):
    for name in ("tpoisson", "tnbreg"):
        for bad in (-1, 1.5, True, None, float("nan"), float("inf"), [0]):
            fails("invalid_spec", name, data, "nonnegative integer", ll=bad)
        fails("outcome_not_truncated", name, data, ll=1)
        fails("outcome_not_truncated", name, data, ll=10**6)
        fails("missing_columns", name, data, ll="nope")
        fails("invalid_truncation", name, data.assign(cut=0.5), ll="cut")
        fails("invalid_truncation", name, data.assign(cut=-1), ll="cut")
        fails("non_numeric_column", name, data.assign(cut="low"), ll="cut")
        outcome = OUTCOME[name]
        fails("constant_outcome", name, data.assign(cut=data[outcome] - 1), ll="cut")
        assert valid(call(name, data, ll=0.0)).extra["truncation_point"] == 0
        fails("invalid_spec", name, data, "2^53", ll=10**30)        # regression: OverflowError
        fails("invalid_spec", name, data, "2^53", ll=1e300)
        fails("invalid_truncation", name, data.assign(cut=1e17), ll="cut")
    fails("invalid_spec", "tnbreg", data, "mean, constant", dispersion="nb3")
    for name in ("zip", "zinb"):
        fails("invalid_spec", name, data, "logit, probit", inflate_link="cloglog")
        constant = valid(call(name, data, inflate=None))             # inflate(_cons)
        assert [c.term for c in constant.coefficients][3] == "inflate:Intercept"
    fails("invalid_spec", "hurdle", data, dist="nb1")
    fails("invalid_spec", "hurdle", data, zero_link="cauchit")
    fails("invalid_spec", "churdle", data, "linear, exponential", model="tobit")
    fails("invalid_option", "churdle", data, "zero or positive", ll=-1)
    for bad in ("low", float("inf"), float("nan")):
        fails("invalid_spec", "churdle", data, ll=bad)
    fails("no_selection_variation", "churdle", data, "no outcome exceeds", ll=1e12)
    fails("no_selection_variation", "churdle", data, "oe.truncreg", ll=-5, model="linear")
    fails("invalid_spec", "churdle", data, "select_x", select_x=None)
    fails("invalid_spec", "churdle", data, "select_x", select_x=[])
    fails("invalid_spec", "churdle", data, "once", select=["z1"])
    alias = call("churdle", data, select_x=None, select=["z1"])
    assert_allclose(alias.metrics["log_likelihood"],
                    call("churdle", data).metrics["log_likelihood"], rtol=1e-14)
    below = valid(call("churdle", data.assign(amount=np.where(data.amount > 0, data.amount,
                                                              -2.0))))
    assert any("below the lower limit" in text for text in below.warnings)
    for name in ("zip", "zinb", "tpoisson", "tnbreg", "hurdle", "gnbreg"):
        frame = data.assign(e=1.5, lne=math.log(1.5))
        fails("invalid_spec", name, frame, "not both", offset="lne", exposure="e")
        fails("invalid_exposure", name, frame.assign(e=np.where(frame.index == 0, 0.0, 1.5)),
              "strictly positive", exposure="e")
        shifted = valid(call(name, frame, exposure="e"))
        assert_allclose(shifted.coefficients[0].estimate,
                        call(name, data).coefficients[0].estimate - math.log(1.5), rtol=1e-7)
    # Confidence levels close to the ends of (0, 1).
    wide = valid(call("zip", data, alpha=1e-12))
    narrow = valid(call("zip", data, alpha=0.999))
    assert slope(wide).ci_high - slope(wide).ci_low > 1000 * (
        slope(narrow).ci_high - slope(narrow).ci_low)
    # The registry rejects what the specification does not declare.
    from pydantic import ValidationError

    from openecon.models import ModelSpec

    with pytest.raises(ValidationError, match="no column role"):
        ModelSpec(estimator="zip", outcome="zip_y", predictors=["x1"], columns={"select_x": ["z1"]})
    with pytest.raises(ValidationError, match="at least 0"):
        ModelSpec(estimator="tpoisson", outcome="trunc_p", predictors=["x1"], options={"ll": -1})
    both = ModelSpec(estimator="tpoisson", outcome="trunc_p", predictors=["x1"],
                     columns={"truncation": "g"}, options={"ll": 1})
    with pytest.raises(AnalysisError) as caught:
        oe.fit(both, data=data)
    assert caught.value.code == "invalid_spec"


# ---- extreme magnitudes ------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_extreme_magnitudes_of_regressors_and_weights(data, name):
    reference = valid(call(name, data))
    base = slope(reference)
    for factor in (1e8, 1e-8):
        scaled = slope(valid(call(name, data.assign(x1=data.x1 * factor))))
        assert_allclose([scaled.estimate * factor, scaled.std_error * factor],
                        [base.estimate, base.std_error], rtol=1e-6)
    shifted = valid(call(name, data.assign(x1=data.x1 + 1e8)))           # a huge level
    assert_allclose([slope(shifted).estimate, slope(shifted).std_error],
                    [base.estimate, base.std_error], rtol=1e-5)
    assert_allclose(shifted.metrics["log_likelihood"], reference.metrics["log_likelihood"],
                    rtol=1e-9)
    role = {"zip": "inflate", "zinb": "inflate", "churdle": "select_x", "hurdle": "select_x",
            "gnbreg": "lnalpha"}.get(name)
    if role is not None:
        prefix = {"inflate": "inflate", "select_x": "select", "lnalpha": "lnalpha"}[role]
        for factor in (1e8, 1e-8):
            scaled = valid(call(name, data.assign(z1=data.z1 * factor)))
            assert_allclose(slope(scaled, f"{prefix}:z1").estimate * factor,
                            slope(reference, f"{prefix}:z1").estimate, rtol=1e-6)
    weighted = valid(call(name, data, weights="w", weight_type="iweight"))
    for factor in (1e8, 1e-8):
        heavy = valid(call(name, data.assign(w=data.w * factor), weights="w",
                           weight_type="iweight"))
        assert_allclose(slope(heavy).estimate, slope(weighted).estimate, rtol=1e-7)
        assert_allclose(slope(heavy).std_error * math.sqrt(factor), slope(weighted).std_error,
                        rtol=1e-6)
        sampling = valid(call(name, data.assign(w=data.w * factor), weights="w",
                              weight_type="pweight"))
        assert_allclose(slope(sampling).std_error,
                        slope(call(name, data, weights="w", weight_type="pweight")).std_error,
                        rtol=1e-6)


def test_extreme_magnitudes_of_outcomes_and_offsets(data):
    for name in ("zip", "zinb", "tpoisson", "tnbreg", "hurdle", "gnbreg"):
        reference = call(name, data)
        for level in (800.0, -800.0):                                    # exp(800) overflows
            moved = valid(call(name, data.assign(o=level), offset="o"))
            assert_allclose(moved.coefficients[0].estimate,
                            reference.coefficients[0].estimate - level, rtol=1e-9)
            assert_allclose(slope(moved).estimate, slope(reference).estimate, rtol=1e-7)
        scaled = data.assign(**{OUTCOME[name]: data[OUTCOME[name]] * 10**6})
        if name in {"zinb", "tnbreg", "gnbreg"}:
            # Counts exceed1e7: outside the verified NB gamma region, not a row limit.
            with pytest.raises(AnalysisError) as caught:
                call(name, scaled)
            assert caught.value.code == "precision_unsupported"
        else:
            millions = valid(call(name, scaled))
            assert 13 < millions.coefficients[0].estimate < 16
    for model, factor in (("exponential", 1e8), ("exponential", 1e-8), ("linear", 1e8),
                          ("linear", 1e-8)):
        frame = data.assign(amount=data.amount * factor)
        reference = call("churdle", data, model=model)
        result = valid(call("churdle", frame, model=model))
        if model == "exponential":                    # ln y shifts by ln(factor)
            assert_allclose(result.coefficients[0].estimate,
                            reference.coefficients[0].estimate + math.log(factor), rtol=1e-8)
            assert_allclose(slope(result).estimate, slope(reference).estimate, rtol=1e-7)
        else:                                         # y, b and sigma scale by the factor
            assert_allclose(slope(result).estimate, slope(reference).estimate * factor,
                            rtol=1e-6)
            assert_allclose(result.metrics["sigma"], reference.metrics["sigma"] * factor,
                            rtol=1e-6)


# ---- boundaries, separation and monotone likelihoods -----------------------------------------


def boundary_data(n=3000, seed=77):
    rng = np.random.default_rng(seed)
    x1, z1 = rng.normal(size=n), rng.normal(size=n)
    group = (rng.uniform(size=n) < 0.3).astype(float)
    counts = rng.poisson(np.exp(0.8 + 0.4 * x1))
    excess = rng.uniform(size=n) < special.expit(-0.5 + 0.8 * z1)
    overdispersed = rng.negative_binomial(2, 2 / (2 + np.exp(0.8 + 0.4 * x1)))
    takes_part = 0.2 + 0.7 * z1 + rng.normal(size=n) > 0
    return pd.DataFrame({
        "x1": x1, "z1": z1, "group": group, "poisson": counts,
        "zip_y": np.where(excess, 0, counts), "nb": overdispersed,
        "underdispersed": rng.binomial(8, np.minimum(np.exp(0.8 + 0.4 * x1) / 8, 0.95)),
        "takes_part": takes_part,
        "amount": np.where(takes_part, np.exp(1 + 0.5 * x1 + 0.6 * rng.normal(size=n)), 0.0),
        "piled": np.where(takes_part, rng.exponential(0.3, size=n), 0.0),
    })


def raises(code, function, *words, **arguments):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AnalysisError) as caught:
            function(**arguments)
    assert caught.value.code == code, (caught.value.code, str(caught.value))
    for word in words:
        assert word in str(caught.value), str(caught.value)


def test_boundary_solutions_name_the_simpler_model():
    frame = boundary_data()
    positive = frame[frame.poisson > 0]
    raises("boundary_solution", oe.zinb, "oe.zip", data=frame, y="zip_y", x=["x1"],
           inflate=["z1"])
    raises("boundary_solution", oe.tnbreg, "oe.tpoisson", data=positive, y="poisson", x=["x1"])
    raises("boundary_solution", oe.tnbreg, "oe.tpoisson", data=positive, y="poisson", x=["x1"],
           dispersion="constant")
    raises("boundary_solution", oe.hurdle, "dist='poisson'", data=frame, y="zip_y", x=["x1"],
           select_x=["z1"], dist="nbinomial")
    raises("boundary_solution", oe.gnbreg, "oe.poisson", data=frame, y="underdispersed",
           x=["x1"], lnalpha=["z1"])
    # Data with (far) fewer zeros than the count model predicts: no inflation to estimate.
    raises("boundary_solution", oe.zip, "oe.poisson", data=frame.assign(
        y=np.where(np.arange(len(frame)) == 0, 0, frame.poisson + 1)), y="y", x=["x1"],
        inflate=["z1"])
    fewer = np.where((frame.nb == 0) & (np.arange(len(frame)) % 4 > 0), 1, frame.nb)
    raises("boundary_solution", oe.zinb, "oe.nbreg", data=frame.assign(y=fewer), y="y", x=["x1"],
           inflate=[])


def test_gnbreg_dispersion_that_collapses_for_part_of_the_sample():
    """Regression: a group without overdispersion is a boundary solution, not noise.

    ``ln(alpha)`` of the underdispersed group runs to minus infinity; below
    alpha = 1e-9 the negative binomial density is a difference of huge
    numbers, and iterating there used to accept rounding noise as progress.
    """
    frame = boundary_data()
    frame["mixed"] = np.where(frame.group == 1, frame.underdispersed, frame.nb)
    for options in ({}, {"covariance": "robust"}, {"weights": "x_w", "weight_type": "iweight"}):
        raises("boundary_solution", oe.gnbreg, "runs to zero for", "lnalpha regressor",
               data=frame.assign(x_w=1.7), y="mixed", x=["x1"], lnalpha=["group"], **options)
    # The same regressor in a sample that is overdispersed everywhere is estimated.
    both = frame.assign(mixed=np.where(frame.group == 1, frame.nb, frame.nb * 2))
    result = valid(oe.gnbreg(data=both, y="mixed", x=["x1"], lnalpha=["group"]))
    assert result.extra["alpha"]["min"] > 1e-3
    # The guard rejects unverified gamma shapes before density evaluation.
    import torch

    from openecon.econometrics.count import gnbreg, kernels

    pieces = gnbreg._Guarded(kernels.CountPieces(kernels.NegBinDensity("mean"),
                                                 torch.tensor([0.0, 2.0, 5.0])))
    eta = torch.zeros(3, dtype=torch.float64)
    from openecon.engines.contracts import KernelError
    assert pieces([eta, torch.full((3,), math.log(1e-6))], False) is not None
    for dispersion in (torch.full((3,), math.log(1e-8)),
                       torch.tensor([0.0, math.log(1e-10), 0.0])):
        with pytest.raises(KernelError) as caught:
            pieces([eta, dispersion], False)
        assert caught.value.code == "precision_unsupported"


def test_separation_in_the_binary_equations():
    frame = boundary_data()
    never_zero = np.where(frame.group == 1, np.maximum(frame.poisson, 1), frame.zip_y)
    always_zero = np.where(frame.group == 1, 0, frame.zip_y)
    for y in (never_zero, always_zero):
        raises("separation_detected", oe.zip, "inflation equation", "remove or combine",
               data=frame.assign(y=y), y="y", x=["x1"], inflate=["z1", "group"])
        raises("separation_detected", oe.hurdle, "selection equation", data=frame.assign(y=y),
               y="y", x=["x1"], select_x=["z1", "group"])
    raises("separation_detected", oe.churdle, "selection equation", data=frame.assign(
        y=np.where(frame.group == 1, np.maximum(frame.amount, 0.5), frame.amount)), y="y",
        x=["x1"], select_x=["z1", "group"])
    # A regressor that is constant among the positive outcomes is omitted from the count part
    # of a hurdle model (its zeros belong to the participation equation) ...
    result = valid(oe.hurdle(data=frame.assign(y=always_zero), y="y", x=["x1", "group"],
                             select_x=["z1"]))
    assert result.provenance["omitted_terms"] == ["group"]
    # ... but leaves no finite estimate in a count equation that also has to explain zeros.
    # Regression: this used to surface as the Poisson starting fit's "nonconvergence".
    zeros_in_group = frame.assign(y=always_zero, nb_y=np.where(frame.group == 1, 0, frame.nb),
                                  other=1 - frame.group)
    for function, outcome, extra in ((oe.zip, "y", {"inflate": ["z1"]}),
                                     (oe.zinb, "nb_y", {"inflate": ["z1"]}),
                                     (oe.gnbreg, "nb_y", {"lnalpha": ["z1"]})):
        for dummy in ("group", "other"):                   # either coding of the indicator
            raises("separation_detected", function, "count equation", dummy,
                   "positive outcome", data=zeros_in_group, y=outcome, x=["x1", dummy],
                   **extra)


def test_truncated_likelihood_without_a_maximum_is_reported_as_separation():
    """Regression: a group whose outcomes all equal ll + 1 has a coefficient at -infinity.

    Gradient and curvature vanish together along that direction, so
    Newton-Raphson used to stop "converged" with a coefficient near -37 and a
    standard error near 1e6.
    """
    frame = boundary_data()
    index = np.arange(len(frame))
    lowest = frame.assign(
        y=np.where(frame.group == 1, 1, frame.poisson + 1),
        nb_y=np.where(frame.group == 1, 1, frame.nb + 1),
        y4=np.where(frame.group == 1, 4, frame.poisson + 4),
        single=(index == 7).astype(float), w=1e-8 * (1 + index % 3), f=1 + index % 2)
    lowest["y_single"] = np.where(lowest.single == 1, 1, frame.poisson + 1)
    words = ("runs to zero", "a coefficient diverges", "Remove that regressor")
    raises("separation_detected", oe.tpoisson, *words, data=lowest, y="y", x=["x1", "group"])
    raises("separation_detected", oe.tpoisson, *words, data=lowest, y="y", x=["x1", "group"],
           covariance="robust")
    raises("separation_detected", oe.tpoisson, *words, data=lowest, y="y", x=["x1", "group"],
           weights="w", weight_type="iweight")
    raises("separation_detected", oe.tpoisson, *words, data=lowest, y="y", x=["x1", "group"],
           weights="f", weight_type="fweight")
    raises("separation_detected", oe.tpoisson, "1 observation(s)", data=lowest, y="y_single",
           x=["x1", "single"])
    raises("separation_detected", oe.tpoisson, "ll + 1", data=lowest, y="y4", x=["x1", "group"],
           ll=3)
    raises("separation_detected", oe.tnbreg, *words, data=lowest, y="nb_y", x=["x1", "group"])
    hurdle = lowest.assign(h=np.where(frame.z1 > 0.5, 0, lowest.nb_y))
    for dist in ("poisson", "nbinomial"):
        raises("separation_detected", oe.hurdle, "always equals 1", data=hurdle, y="h",
               x=["x1", "group"], select_x=["x1"], dist=dist)
    # Not flagged: identified models with tiny fitted means or very little information.
    rng = np.random.default_rng(1)
    outlier = frame.x1.to_numpy().copy()
    outlier[:3] = -60.0                                     # fitted means near exp(-23)
    counts = rng.poisson(np.exp(0.8 + 0.4 * outlier))
    counts[:3] = 1
    kept = counts > 0
    result = valid(oe.tpoisson(data=pd.DataFrame({"y": counts[kept], "x": outlier[kept]}),
                               y="y", x=["x"]))
    assert abs(slope(result, "x").estimate - 0.4) < 0.05
    rare = pd.DataFrame({"y": 1 + (rng.uniform(size=3000) < 0.006).astype(int),
                         "x1": frame.x1})
    assert valid(oe.tpoisson(data=rare, y="y", x=["x1"])).coefficients[0].estimate < -3
    small = lowest.assign(tiny=(index < 5).astype(float))
    small["y5"] = np.where(small.tiny == 1, 1, frame.poisson + 1)
    small.loc[0, "y5"] = 2                                  # one event identifies the group
    result = valid(oe.tpoisson(data=small, y="y5", x=["x1", "tiny"]))
    assert -4 < slope(result, "tiny").estimate < -1


def test_cragg_hurdle_degenerate_outcome_equations():
    frame = boundary_data()
    raises("perfect_fit", oe.churdle, "sigma is zero", data=frame.assign(
        y=np.where(frame.takes_part, np.exp(1 + 0.5 * frame.x1), 0.0)), y="y", x=["x1"],
        select_x=["z1"])
    raises("perfect_fit", oe.churdle, data=frame.assign(
        y=np.where(frame.takes_part, 5 + frame.x1, 0.0)), y="y", x=["x1"], select_x=["z1"],
        model="linear")
    # Outcomes piled up next to the limit: the truncated normal has no interior maximum.
    raises("nonconvergence", oe.churdle, "model='exponential'", data=frame, y="piled",
           x=["x1"], select_x=["z1"], model="linear")
    assert valid(oe.churdle(data=frame, y="piled", x=["x1"], select_x=["z1"]))


def test_two_part_models_need_enough_observations_above_the_limit(data):
    """Regression: the message names the equation and the subsample it is estimated from."""
    index = np.arange(len(data))
    raises("insufficient_observations", oe.hurdle, "count equation", "above zero",
           data=data.assign(y=np.where(index == 0, 3, 0)), y="y", x=["x1"], select_x=["z1"])
    raises("insufficient_observations", oe.churdle, "outcome equation", "above the limit 0",
           data=data.assign(y=np.where(index < 2, 1.0 + index, 0.0)), y="y", x=["x1"],
           select_x=["z1"])
    one_zero = data.assign(y=np.where(index == 0, 0, data.trunc_p))
    result = valid(oe.hurdle(data=one_zero, y="y", x=["x1"], select_x=["z1"]))
    assert result.metrics["n_zero_observations"] == 1


def test_small_but_identified_parameters_are_estimated():
    """Boundary detection must not swallow small inflation or dispersion that is in the data."""
    rng = np.random.default_rng(15)
    n = 120_000
    x1, z1 = rng.normal(size=n), rng.normal(size=n)
    mu = np.exp(0.2 + 0.4 * x1)
    frame = pd.DataFrame({"x1": x1, "z1": z1})
    frame["rare"] = np.where(rng.uniform(size=n) < 0.01, 0, rng.poisson(mu))
    result = valid(oe.zip(data=frame, y="rare", x=["x1"], inflate=[]))
    assert abs(special.expit(result.coefficients[-1].estimate) - 0.01) < 0.004
    frame["tight"] = rng.negative_binomial(100, 100 / (100 + mu))           # alpha = 0.01
    result = valid(oe.gnbreg(data=frame, y="tight", x=["x1"], lnalpha=["z1"]))
    assert 0.003 < result.extra["alpha"]["mean"] < 0.03
    result = valid(oe.tnbreg(data=frame[frame.tight > 0], y="tight", x=["x1"]))
    assert 0.003 < result.metrics["alpha"] < 0.03
    frame["wild"] = rng.negative_binomial(0.05, 0.05 / (0.05 + mu))         # alpha = 20
    result = valid(oe.hurdle(data=frame, y="wild", x=["x1"], dist="nbinomial"))
    assert 8 < result.metrics["alpha"] < 50
    result = valid(oe.tnbreg(data=frame[frame.wild > 0], y="wild", x=["x1"]))
    assert 8 < result.metrics["alpha"] < 50


def test_high_truncation_points_agree_with_the_likelihood_written_out():
    rng = np.random.default_rng(21)
    n = 2500
    x1 = rng.normal(size=n)
    counts = rng.negative_binomial(1 / 0.3, 1 / (1 + 0.3 * np.exp(5.0 + 0.2 * x1)))
    for limit in (40, 110):
        kept = counts > limit
        y, x = counts[kept].astype(float), x1[kept]
        result = valid(oe.tnbreg(data=pd.DataFrame({"y": y, "x1": x}), y="y", x=["x1"],
                                 ll=limit))

        def loglik(theta, y=y, x=x, limit=limit):
            mu, m = np.exp(theta[0] + theta[1] * x), math.exp(-theta[2])
            j = np.arange(limit + 1)[:, None]
            below = np.exp(special.gammaln(j + m) - special.gammaln(m) - special.gammaln(j + 1)
                           + m * np.log(m / (m + mu)) + j * np.log(mu / (m + mu))).sum(axis=0)
            density = (special.gammaln(y + m) - special.gammaln(m) - special.gammaln(y + 1)
                       + m * np.log(m / (m + mu)) + y * np.log(mu / (m + mu)))
            return float((density - np.log1p(-below)).sum())

        theta = np.array([c.estimate for c in result.coefficients])
        assert_allclose(result.metrics["log_likelihood"], loglik(theta), rtol=1e-10)
        for j in range(3):                                   # the score vanishes at the estimates
            step = np.zeros(3)
            step[j] = 1e-5
            gradient = (loglik(theta + step) - loglik(theta - step)) / 2e-5
            assert abs(gradient) * result.coefficients[j].std_error < 1e-4
        poisson = valid(oe.tpoisson(data=pd.DataFrame({"y": y, "x1": x}), y="y", x=["x1"],
                                    ll=limit))
        assert poisson.metrics["log_likelihood"] < result.metrics["log_likelihood"]
    # Regression: a truncation point whose lower sum cannot be evaluated in reasonable time
    # is refused with advice instead of hanging (the Poisson tail has no such limit).
    huge = pd.DataFrame({"x1": x1, "y": 520_000 + rng.negative_binomial(5, 0.001, size=n)})
    raises("truncation_too_large", oe.tnbreg, "oe.tpoisson", data=huge, y="y", x=["x1"],
           ll=500_000)
    assert valid(oe.tpoisson(data=huge, y="y", x=["x1"], ll=500_000)).nobs == n


def test_constant_only_negative_binomial_without_a_maximum_is_not_the_poisson_model():
    """Regression: the null model of the LR test when its dispersion diverges.

    For these truncated counts the constant-only negative binomial likelihood
    increases with alpha all the way to its logarithmic-series limit. The
    implementation used to substitute the constant-only truncated *Poisson*
    log likelihood (the alpha = 0 boundary, which is the opposite end) and
    reported an LR chi2 four times too large and a pseudo R-squared of 0.096.
    """
    import torch

    from openecon.econometrics.count import common, kernels, truncated
    from openecon.models import ModelSpec

    rng = np.random.default_rng(21)
    n = 4000
    x1, x2 = rng.normal(size=n), rng.integers(0, 3, size=n).astype(float)
    counts = rng.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * np.exp(0.6 + 0.4 * x1 - 0.2 * x2)))
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": counts})
    frame = frame[frame.y > 3].reset_index(drop=True)
    y = frame.y.to_numpy(dtype=float)
    support = np.arange(4, 4000.0)

    def constant_only(log_mean, log_alpha):
        m, mean = math.exp(-log_alpha), math.exp(log_mean)

        def log_pmf(k):
            return (special.gammaln(k + m) - special.gammaln(m) - special.gammaln(k + 1)
                    + m * math.log(m / (m + mean)) + k * math.log(mean / (m + mean)))
        return float(log_pmf(y).sum() - len(y) * special.logsumexp(log_pmf(support)))

    from scipy import optimize

    profile = [-optimize.minimize_scalar(lambda b, a=a: -constant_only(b, a), bounds=(-30, 6),
                                         method="bounded", options={"xatol": 1e-10}).fun
               for a in (-2.0, 0.0, 2.0, 4.0, 8.0, 12.0)]
    assert all(later > earlier for earlier, later in zip(profile, profile[1:], strict=False))

    def log_series(logit):                                     # the limit alpha -> infinity
        p = special.expit(logit)
        return float((y * math.log(p) - np.log(y)).sum() - len(y) * special.logsumexp(
            support * math.log(p) - np.log(support)))

    supremum = -optimize.minimize_scalar(lambda q: -log_series(q), bounds=(-10, 10),
                                         method="bounded", options={"xatol": 1e-12}).fun
    assert profile[-1] < supremum < profile[-1] + 1e-4

    for dispersion in ("mean", "constant"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = valid(oe.tnbreg(data=frame, y="y", x=["x1", "x2"], ll=3,
                                     dispersion=dispersion))
        assert any("has no maximum" in text for text in result.warnings)
        assert result.tests["model"]["label"].startswith("Wald chi2")
        assert result.metrics["pseudo_r_squared"] is None
        assert result.extra["null_log_likelihood"] is None
        # The honest LR statistic against the supremum is far below the old Poisson-based one.
        honest = 2 * (result.metrics["log_likelihood"] - supremum)
        assert 0 < honest < 60 and result.tests["model"]["statistic"] < 60
        assert "alpha" in result.tests                  # the test against tpoisson is unaffected

    # The classification itself: a dispersion that grows is not "at zero"; one that collapses is.
    spec = ModelSpec(estimator="tnbreg", outcome="y", predictors=["x1", "x2"],
                     options={"ll": 3})
    sample, _ = truncated._prepare(spec, frame, "tnbreg")
    pieces = kernels.TruncatedPieces(kernels.NegBinDensity("mean"), sample.y, sample.limit)
    start = torch.tensor([1.5, 0.0], dtype=torch.float64)
    fitted, at_zero = truncated._constant_only(sample, pieces, start,
                                               [common.scalar_block("/lnalpha")])
    assert fitted is None and at_zero is False
    tight = pd.DataFrame({"x1": x1[:800], "x2": x2[:800], "y": 1 + rng.binomial(6, 0.5, 800)})
    spec = ModelSpec(estimator="tnbreg", outcome="y", predictors=["x1", "x2"])
    sample, _ = truncated._prepare(spec, tight, "tnbreg")
    pieces = kernels.TruncatedPieces(kernels.NegBinDensity("mean"), sample.y, sample.limit)
    fitted, at_zero = truncated._constant_only(sample, pieces, start,
                                               [common.scalar_block("/lnalpha")])
    assert fitted is None and at_zero is True


def small_design(seed):
    """Small random samples (40-160 rows) on which several likelihoods have no maximum."""
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(40, 160))
    x1 = rng.normal(size=n) * rng.choice([0.2, 1, 3])
    d1 = (rng.uniform(size=n) < rng.uniform(0.1, 0.9)).astype(float)
    z1 = rng.normal(size=n)
    mu = np.exp(rng.uniform(-1, 2) + 0.3 * x1 + rng.uniform(-1, 1) * d1)
    alpha = float(rng.choice([0.05, 0.3, 1.0, 3.0]))
    excess = rng.uniform(size=n) < special.expit(rng.uniform(-2, 1) + rng.uniform(-1.5, 1.5) * z1)
    negbin = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    poisson = rng.poisson(mu)
    return pd.DataFrame({"x1": x1, "d1": d1, "z1": z1, "zip_y": np.where(excess, 0, poisson),
                         "zinb_y": np.where(excess, 0, negbin), "nb_y": negbin,
                         "pois_y": poisson})


def test_flat_likelihoods_are_never_reported_as_converged_fits():
    """Regression: no table with a parameter parked on the flat part of the likelihood.

    Next to alpha = 0, and where fitted probabilities or means reach their
    limits, gradient and curvature vanish together and Newton-Raphson can meet
    its convergence rules anywhere. On the design of seed 11 the negative
    binomial hurdle used to return ``/lnalpha = -19.2`` with a standard error
    of 1458 as a converged fit.
    """
    frame = small_design(11)
    raises("boundary_solution", oe.hurdle, "dist='poisson'", data=frame, y="zinb_y",
           x=["x1", "d1"], select_x=["z1"], dist="nbinomial")
    # ln(delta) = -7.3 with a standard error of 330: flat, although delta is above 1e-4.
    frame = small_design(159)
    raises("boundary_solution", oe.tnbreg, "delta lies at zero", data=frame[frame.nb_y > 1],
           y="nb_y", x=["x1", "d1"], ll=1, dispersion="constant")
    # An intercept of -5.8 with a standard error of 20: fitted means on their way to zero.
    frame = small_design(62)
    raises("boundary_solution", oe.tnbreg, "logarithmic-series", "no interior maximum",
           data=frame[frame.nb_y > 1], y="nb_y", x=["x1", "d1"], ll=1, dispersion="constant")
    allowed = {"boundary_solution", "separation_detected", "nonconvergence"}
    outcomes = {"fit": 0, "error": 0}
    for seed in range(24):
        frame = small_design(seed)
        calls = [
            lambda: oe.zip(data=frame, y="zip_y", x=["x1", "d1"], inflate=["z1"]),
            lambda: oe.zinb(data=frame, y="zinb_y", x=["x1", "d1"], inflate=["z1"]),
            lambda: oe.gnbreg(data=frame, y="nb_y", x=["x1", "d1"], lnalpha=["z1"]),
            lambda: oe.hurdle(data=frame, y="zinb_y", x=["x1", "d1"], select_x=["z1"],
                              dist="nbinomial"),
            lambda: oe.tpoisson(data=frame[frame.pois_y > 1], y="pois_y", x=["x1", "d1"], ll=1),
            lambda: oe.tnbreg(data=frame[frame.nb_y > 0], y="nb_y", x=["x1", "d1"]),
            lambda: oe.tnbreg(data=frame[frame.nb_y > 1], y="nb_y", x=["x1", "d1"], ll=1,
                              dispersion="constant"),
        ]
        for fit in calls:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = valid(fit())
            except AnalysisError as error:
                assert error.code in allowed, (seed, error.code, str(error))
                outcomes["error"] += 1
                continue
            outcomes["fit"] += 1
            for c in result.coefficients:
                # The regressors are of order one: a finite maximum has moderate numbers.
                assert abs(c.estimate) < 25 and c.std_error < 500, (seed, c)
            for name in ("alpha", "delta"):
                if name in result.metrics:
                    assert result.metrics[name] > 1e-4, (seed, result.metrics)
    assert outcomes["fit"] > 100 and outcomes["error"] > 15, outcomes


def test_truncated_negative_binomial_at_its_logarithmic_series_limit():
    """The mean runs to zero with a positive dispersion: explained, not just "no convergence".

    As its mean goes to zero (and, in the mean-dispersion form, alpha to
    infinity) the zero-truncated negative binomial becomes the logarithmic-
    series distribution. Counts drawn from that distribution may or may not
    leave an interior maximum; both outcomes are checked against the
    supremum of the logarithmic-series likelihood, computed independently.
    """
    from scipy import optimize, stats

    frame = small_design(1)
    raises("boundary_solution", oe.tnbreg, "logarithmic-series", "run to zero",
           data=frame[frame.nb_y > 1], y="nb_y", x=["x1", "d1"], ll=1, dispersion="constant")
    frame = small_design(8)
    raises("boundary_solution", oe.tnbreg, "alpha grows without bound",
           data=frame[frame.nb_y > 0], y="nb_y", x=["x1", "d1"])

    support = np.arange(1, 6000.0)

    def series_supremum(y):
        def loglik(logit):
            p = special.expit(logit)
            return float((y * math.log(p) - np.log(y)).sum() - len(y) * special.logsumexp(
                support * math.log(p) - np.log(support)))
        return -optimize.minimize_scalar(lambda q: -loglik(q), bounds=(-10, 10),
                                         method="bounded", options={"xatol": 1e-12}).fun

    rng = np.random.default_rng(3003)
    outcomes = {}
    for n in (300, 1500, 20000):
        x1 = rng.normal(size=n)
        d1 = (rng.uniform(size=n) < 0.4).astype(float)
        y = stats.logser.rvs(0.75, size=n, random_state=rng)
        frame = pd.DataFrame({"x1": x1, "d1": d1, "y": y})
        supremum = series_supremum(y.astype(float))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = valid(oe.tnbreg(data=frame, y="y", x=[]))
        except AnalysisError as error:
            assert error.code == "boundary_solution", (n, error.code, str(error))
            assert "alpha grows without bound" in str(error)
            outcomes[n] = "boundary"
            continue
        # A reported maximum must beat the limiting model and be reasonably determined.
        assert result.metrics["log_likelihood"] > supremum + 1e-3, n
        assert result.coefficients[-1].std_error < 10
        outcomes[n] = "maximum"
    assert outcomes == {300: "maximum", 1500: "maximum", 20000: "boundary"}
    raises("boundary_solution", oe.tnbreg, "logarithmic-series", "run to zero", data=frame,
           y="y", x=["x1", "d1"], dispersion="constant")
    assert valid(oe.tpoisson(data=frame, y="y", x=["x1", "d1"], covariance="robust"))
