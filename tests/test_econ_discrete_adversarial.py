"""Adversarial inputs for the discrete-choice family (verification stage).

Every invalid or degenerate input must either be estimated correctly or raise
``AnalysisError`` with a snake_case code and a message that says what to
change: never a raw exception, never NaN / inf in a result. The second half
holds one regression test per defect fixed during verification.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics import registry
from openecon.models import ModelSpec

N = 300


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(1)
    data = pd.DataFrame({
        "x1": rng.normal(size=N), "x2": rng.normal(size=N), "g": np.repeat(np.arange(60), 5),
        "w": rng.uniform(0.5, 2, size=N), "f": rng.integers(1, 4, size=N).astype(float),
        "s": rng.choice(["a", "b", "c"], size=N),
    })
    latent = 0.8 * data.x1 - 0.5 * data.x2
    data["yo"] = np.digitize(latent + rng.logistic(size=N), [-1, 0.3, 1.5])
    utility = np.column_stack([np.zeros(N), 0.4 + 0.8 * data.x1, -0.2 - 0.5 * data.x1])
    data["ym"] = (utility + rng.gumbel(size=(N, 3))).argmax(axis=1)
    data["yb"] = (latent + rng.normal(size=N) > 0) * 1.0
    data["yb2"] = (0.3 * data.x1 + rng.normal(size=N) > 0) * 1.0
    effect = rng.normal(size=60)[data.g]
    data["yc"] = (rng.random(N) < 1 / (1 + np.exp(-(effect + data.x1)))) * 1.0
    return data


CALLS = {
    "ologit": lambda d, **k: oe.ologit(data=d, y="yo", x=["x1", "x2"], **k),
    "oprobit": lambda d, **k: oe.oprobit(data=d, y="yo", x=["x1", "x2"], **k),
    "mlogit": lambda d, **k: oe.mlogit(data=d, y="ym", x=["x1", "x2"], **k),
    "clogit": lambda d, **k: oe.clogit(data=d, y="yc", x=["x1", "x2"], group="g", **k),
    "hetprobit": lambda d, **k: oe.hetprobit(data=d, y="yb", x=["x1"], het=["x2"], **k),
    "biprobit": lambda d, **k: oe.biprobit(data=d, y1="yb", y2="yb2", x=["x1", "x2"], **k),
}
OUTCOME = {"ologit": "yo", "oprobit": "yo", "mlogit": "ym", "clogit": "yc", "hetprobit": "yb",
           "biprobit": "yb"}
COMMANDS = list(CALLS)


def code(call):
    """The AnalysisError code of a call that must fail in a controlled way."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AnalysisError) as caught:
            call()
    message = str(caught.value)
    assert caught.value.code == caught.value.code.lower() and " " not in caught.value.code
    assert len(message) > 20 and "Traceback" not in message
    return caught.value.code


def finite(result):
    """No NaN / inf anywhere in a result (JSON round trip included)."""
    for row in result.coefficients:
        values = (row.estimate, row.std_error, row.statistic, row.p_value, row.ci_low,
                  row.ci_high)
        assert all(math.isfinite(value) for value in values), row
        assert row.std_error > 0 and 0 <= row.p_value <= 1 and row.ci_low <= row.ci_high
    assert np.isfinite(np.array(result.covariance_matrix)).all()
    for name, value in result.metrics.items():
        assert value is None or math.isfinite(value), name
    text = result.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    assert type(result).model_validate_json(text).nobs == result.nobs
    return result


@pytest.mark.parametrize("command", COMMANDS)
def test_degenerate_samples_raise_helpful_errors(frame, command):
    call = CALLS[command]
    assert code(lambda: call(frame.iloc[:0])) == "empty_data"
    assert code(lambda: call(frame.iloc[:1])) in {"constant_outcome", "no_outcome_variation"}
    # Fewer observations (or informative groups) than parameters, or an outcome that cannot
    # vary in so small a sample.
    few = code(lambda: call(frame.iloc[:4]))
    assert few in {"insufficient_observations", "separation_detected", "constant_outcome",
                   "no_outcome_variation"}
    assert code(lambda: call(frame.assign(x1=np.nan))) == "missing_values"
    assert code(lambda: call(frame.assign(x1=np.nan), missing="drop")) == "empty_sample"
    constant = frame.assign(**{OUTCOME[command]: 1.0})
    assert code(lambda: call(constant)) in {"constant_outcome", "no_outcome_variation"}
    assert code(lambda: call(frame.assign(x1="text"))) == "non_numeric_column"
    infinite = frame.assign(x1=np.where(np.arange(N) == 3, np.inf, frame.x1))
    assert code(lambda: call(infinite)) == "non_finite_values"
    assert code(lambda: call(frame.drop(columns="x2"))) == "missing_columns"
    assert code(lambda: call(pd.concat([frame, frame[["x1"]]], axis=1))) == "duplicate_columns"


@pytest.mark.parametrize("command", COMMANDS)
def test_invalid_weights_clusters_and_options(frame, command):
    call = CALLS[command]
    kind = "iweight" if command == "clogit" else "aweight"
    weights = frame.assign(w=np.where(frame.g % 2 == 0, 1.0, 2.0))       # constant in groups
    assert code(lambda: call(weights.assign(w=-weights.w), weights="w", weight_type=kind)) \
        == "negative_weights"
    mixed = weights.assign(w=np.where(weights.g < 3, -1.0, 1.0))
    assert code(lambda: call(mixed, weights="w", weight_type="iweight")) == "negative_weights"
    assert code(lambda: call(weights.assign(w=0.0), weights="w", weight_type="iweight")) \
        == "empty_sample"
    assert code(lambda: call(weights.assign(w=weights.w + 0.5), weights="w",
                             weight_type="fweight")) == "noninteger_frequency_weights"
    assert code(lambda: call(weights.assign(w="heavy"), weights="w", weight_type="iweight")) \
        == "non_numeric_column"
    assert code(lambda: call(weights.assign(w=np.where(weights.g < 2, np.nan, weights.w)),
                             weights="w", weight_type="iweight")) == "missing_values"
    assert code(lambda: call(weights, weights="w")) == "invalid_spec"     # no weight_type
    assert code(lambda: call(weights, weights="w", weight_type="pweight",
                             covariance="nonrobust")) == "unsupported_covariance"
    assert code(lambda: call(weights, weights="w", weight_type="pweight", covariance="opg")) \
        == "unsupported_covariance"
    if command == "clogit":
        assert code(lambda: call(frame, weights="w", weight_type="aweight")) == "invalid_spec"
    # Extreme weight magnitudes: estimates do not move, the covariance scales with 1 / w.
    unit = finite(call(weights.assign(w=1.0), weights="w", weight_type="iweight"))
    for scale in (1e-12, 1e12):
        scaled = finite(call(weights.assign(w=scale), weights="w", weight_type="iweight"))
        assert_allclose([row.estimate for row in scaled.coefficients],
                        [row.estimate for row in unit.coefficients], rtol=1e-7)
        assert_allclose(np.array(scaled.covariance_matrix) * scale,
                        np.array(unit.covariance_matrix), rtol=1e-6)
    # Clusters.
    assert code(lambda: call(frame.assign(c=1), cluster="c")) == "insufficient_clusters"
    assert code(lambda: call(frame, covariance="cluster")) == "invalid_spec"
    assert code(lambda: call(frame, covariance="robust", cluster="g")) == "invalid_spec"
    three = frame.assign(c=np.arange(N) % 7, d=np.arange(N) % 5, e=np.arange(N) % 3)
    assert code(lambda: call(three, cluster=["c", "d", "e"])) == "invalid_spec"
    two = finite(call(frame.assign(c=frame.g % 2), cluster="c"))
    assert two.inference["cluster_count"] == 2 and any("Only 2 clusters" in w
                                                        for w in two.warnings)
    twoway = finite(call(frame.assign(c=frame.g // 3, d=frame.g % 13), cluster=["c", "d"]))
    assert twoway.inference["cluster_columns"] == ["c", "d"]
    # Options at and beyond their bounds.
    for alpha in (0.0, 1.0, -0.1, 2.0):
        assert code(lambda alpha=alpha: call(frame, alpha=alpha)) == "invalid_spec"
    wide = finite(call(frame, alpha=1e-12))
    narrow = finite(call(frame, alpha=1 - 1e-9))
    assert wide.coefficients[0].ci_low < narrow.coefficients[0].ci_low
    for covariance in ("HC1", "hac", "bootstrap", "oim"):
        assert code(lambda covariance=covariance: call(frame, covariance=covariance)) \
            == "invalid_spec"
    assert code(lambda: call(frame, missing="skip")) == "invalid_spec"


@pytest.mark.parametrize("command", COMMANDS)
def test_collinear_constant_and_badly_scaled_regressors(frame, command):
    call = CALLS[command]
    reference = finite(call(frame))
    names = [row.term for row in reference.coefficients]
    values = np.array([row.estimate for row in reference.coefficients])
    errors = np.array([row.std_error for row in reference.coefficients])
    position = next(i for i, name in enumerate(names) if name.endswith("x1"))
    for scale in (1e-8, 1e8, 1e-150, 1e150):
        scaled = finite(call(frame.assign(x1=frame.x1 * scale)))
        factor = np.array([1 / scale if name.endswith("x1") else 1.0 for name in names])
        assert_allclose([row.estimate for row in scaled.coefficients], values * factor, rtol=1e-6)
        assert_allclose([row.std_error for row in scaled.coefficients], errors * factor,
                        rtol=1e-5)
        assert_allclose(scaled.coefficients[position].statistic,
                        reference.coefficients[position].statistic, rtol=1e-5)
        assert_allclose(scaled.metrics["log_likelihood"], reference.metrics["log_likelihood"],
                        rtol=1e-10)
    if command == "hetprobit":
        return           # x2 is its variance regressor; its own collinearity test is below
    duplicate = finite(call(frame.assign(x2=-3 * frame.x1)))
    assert any(term.endswith("x2") for term in duplicate.provenance["omitted_terms"])
    assert any("Omitted" in message for message in duplicate.warnings)
    assert not any(row.term.endswith("x2") for row in duplicate.coefficients)
    constant = finite(call(frame.assign(x2=7.0)))
    assert any(term.endswith("x2") for term in constant.provenance["omitted_terms"])


def test_spec_path_enforces_the_same_contract(frame):
    # The constant is not optional in the models whose likelihood has none.
    for estimator in ("ologit", "oprobit", "clogit"):
        with pytest.raises(ValidationError):
            ModelSpec(estimator=estimator, outcome="yo", predictors=["x1"], intercept=True,
                      columns={"group": "g"} if estimator == "clogit" else {})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="clogit", outcome="yc", predictors=["x1"], intercept=False)
    with pytest.raises(ValidationError):
        ModelSpec(estimator="hetprobit", outcome="yb", predictors=["x1"])
    with pytest.raises(ValidationError):
        ModelSpec(estimator="biprobit", outcome="yb", predictors=["x1"])
    with pytest.raises(ValidationError):
        ModelSpec(estimator="mlogit", outcome="ym", predictors=["x1"], options={"link": "x"})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="mlogit", outcome="ym", predictors=["x1"], columns={"offset": "x2"})
    # pweights through oe.fit keep the default covariance (nonrobust) and are refused.
    spec = ModelSpec(estimator="oprobit", outcome="yo", predictors=["x1"], intercept=False,
                     weights="w", weight_type="pweight")
    with pytest.raises(AnalysisError) as caught:
        fit(spec, data=frame)
    assert caught.value.code == "unsupported_covariance"
    direct = fit(ModelSpec(estimator="biprobit", outcome="yb", predictors=["x1"],
                           columns={"outcome2": "yb2"}, covariance="robust"), data=frame)
    named = oe.biprobit(data=frame, y1="yb", y2="yb2", x=["x1"], covariance="robust")
    assert_allclose([row.estimate for row in direct.coefficients],
                    [row.estimate for row in named.coefficients], rtol=1e-12)
    # Every declared covariance and weight type is implemented (none is a manifest stub).
    weights = frame.assign(w=np.where(frame.g % 2 == 0, 1.0, 2.0))
    for command, call in CALLS.items():
        info = registry.get(command)
        assert info.inference == "z" and info.default_covariance == "nonrobust"
        for covariance in info.covariances:
            options = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
            finite(call(frame, **options))
        for weight_type in info.weights:
            finite(call(weights, weights="w", weight_type=weight_type,
                        covariance="robust" if weight_type == "pweight" else None))


def test_outcome_coding_errors(frame):
    assert code(lambda: oe.ologit(data=frame.assign(yo=frame.s), y="yo", x=["x1"])) \
        == "invalid_ordered_outcome"
    unordered = frame.assign(yo=pd.Categorical(frame.s))
    assert code(lambda: oe.oprobit(data=unordered, y="yo", x=["x1"])) == "invalid_ordered_outcome"
    many = pd.DataFrame({"y": np.arange(1200) % 600, "x": np.linspace(0, 1, 1200)})
    assert code(lambda: oe.ologit(data=many, y="y", x=["x"])) == "too_many_categories"
    assert code(lambda: oe.mlogit(data=many.assign(y=many.y % 400), y="y", x=["x"])) \
        == "too_many_categories"
    for base in (7, "1", True, [1], float("nan")):
        assert code(lambda base=base: oe.mlogit(data=frame, y="ym", x=["x1"], base=base)) \
            == "invalid_base_category"
    clash = frame.assign(ym=[1 if v == 0 else "1" if v == 1 else 2.5 for v in frame.ym])
    assert code(lambda: oe.mlogit(data=clash, y="ym", x=["x1"])) == "ambiguous_categories"
    for command in ("clogit", "hetprobit", "biprobit"):
        doubled = frame.assign(**{OUTCOME[command]: frame[OUTCOME[command]] * 2})
        assert code(lambda command=command, doubled=doubled: CALLS[command](doubled)) \
            == "invalid_binary_outcome"
    assert code(lambda: oe.biprobit(data=frame.assign(yb2=frame.yo), y1="yb", y2="yb2",
                                    x=["x1"])) == "invalid_binary_outcome"
    assert code(lambda: oe.biprobit(data=frame.assign(yb2=1.0), y1="yb", y2="yb2", x=["x1"])) \
        == "constant_outcome"
    # Structural mistakes in the convenience functions are invalid_spec, never TypeError.
    for call in (
        lambda: oe.ologit(data=frame, y="yo", x="x1"),
        lambda: oe.ologit(data=frame, y="yo", x=[]),
        lambda: oe.ologit(data=frame, y="yo", x=["x1", "x1"]),
        lambda: oe.ologit(data=frame, y="yo", x=["yo"]),
        lambda: oe.ologit(data=frame, y="yo", x=["x1"], offset=["x2"]),
        lambda: oe.ologit(data=frame, y="yo", x=["x1"], categorical=["s"]),
        lambda: oe.ologit(data=frame, y="yo", x=["x1"], cluster="yo"),
        lambda: oe.clogit(data=frame, y="yc", x=["x1"], group=None),
        lambda: oe.clogit(data=frame, y="yc", x=["x1"], group=["g"]),
        lambda: oe.clogit(data=frame, y="yc", x=["x1"], group="x1"),
        lambda: oe.clogit(data=frame, y="yc", x=["x1"], group="yc"),
        lambda: oe.hetprobit(data=frame, y="yb", x=["x1"], het=[]),
        lambda: oe.hetprobit(data=frame, y="yb", x=["x1"], het="x2"),
        lambda: oe.hetprobit(data=frame, y="yb", x=["x1"], het=["yb"]),
        lambda: oe.hetprobit(data=frame, y="yb", x=["x1"], het=["x2", "x2"]),
        lambda: oe.biprobit(data=frame, y1="yb", y2="yb", x=["x1"]),
        lambda: oe.biprobit(data=frame, y1="yb", y2=None, x=["x1"]),
        lambda: oe.biprobit(data=frame, y1="yb", y2="yb2", x=["x1"], x2=[]),
        lambda: oe.biprobit(data=frame, y1="yb", y2="yb2", x=["x1"], x2=["yb2"]),
        lambda: oe.biprobit(data=frame, y1="yb", y2="yb2", x=["x1", "yb2"], x2=["x1", "yb"]),
    ):
        assert code(call) == "invalid_spec"


def test_clogit_group_structure_errors(frame):
    singletons = frame.assign(g=np.arange(N))
    assert code(lambda: oe.clogit(data=singletons, y="yc", x=["x1"], group="g")) \
        == "no_outcome_variation"
    assert code(lambda: oe.clogit(data=frame.assign(g=1), y="yc", x=["x1", "x2"], group="g")) \
        == "insufficient_observations"
    level = frame.assign(x1=frame.g * 2.0, x2=1.0)
    assert code(lambda: oe.clogit(data=level, y="yc", x=["x1", "x2"], group="g")) \
        == "no_within_group_variation"
    assert code(lambda: oe.clogit(data=frame, y="yc", x=["x1"], group="g", weights="f",
                                  weight_type="fweight")) == "weights_not_constant_within_group"
    holes = frame.assign(g=np.where(np.arange(N) < 5, np.nan, frame.g))
    assert code(lambda: oe.clogit(data=holes, y="yc", x=["x1"], group="g")) == "missing_values"
    dropped = finite(oe.clogit(data=holes, y="yc", x=["x1"], group="g", missing="drop"))
    assert dropped.dropped_rows >= 5
    # An offset that is constant within groups cancels from the conditional likelihood.
    plain = finite(oe.clogit(data=frame, y="yc", x=["x1", "x2"], group="g"))
    shifted = finite(oe.clogit(data=frame.assign(o=frame.g * 3.0), y="yc", x=["x1", "x2"],
                               group="g", offset="o"))
    assert_allclose([row.estimate for row in shifted.coefficients],
                    [row.estimate for row in plain.coefficients], rtol=1e-8)
    # A few very large groups with hundreds of positives: the recursion neither overflows
    # nor underflows.
    rng = np.random.default_rng(3)
    large = pd.DataFrame({"g": np.repeat(np.arange(3), 1500), "x": rng.normal(size=4500)})
    large["y"] = (rng.random(4500) < 1 / (1 + np.exp(-0.5 * large.x))) * 1.0
    result = finite(oe.clogit(data=large, y="y", x=["x"], group="g"))
    assert abs(result.coefficients[0].estimate - 0.5) < 0.15


# ---- regression tests of the defects fixed during verification --------------------------


def test_clogit_is_invariant_to_the_level_of_a_regressor(frame):
    """Defect: the Hessian lost (level / spread)^2 digits; SEs were off by 6% at a level of 1e7."""
    for options in ({}, {"covariance": "robust"}, {"covariance": "opg"},
                    {"cluster": "town"}, {"cluster": "wave"}):
        data = frame.assign(town=frame.g // 5, wave=np.arange(N) % 5)
        reference = oe.clogit(data=data, y="yc", x=["x1", "x2"], group="g", **options)
        moved = oe.clogit(data=data.assign(x1=data.x1 + 1e7, x2=data.x2 - 3e6), y="yc",
                          x=["x1", "x2"], group="g", **options)
        assert_allclose([row.estimate for row in moved.coefficients],
                        [row.estimate for row in reference.coefficients], rtol=1e-7)
        assert_allclose(np.array(moved.covariance_matrix),
                        np.array(reference.covariance_matrix), rtol=1e-6)
    # Groups split across clusters (Stata's nonest) are accepted with a warning.
    crossed = oe.clogit(data=data, y="yc", x=["x1"], group="g", cluster="wave")
    assert any("split across the clusters of 'wave'" in message for message in crossed.warnings)
    nested = oe.clogit(data=data, y="yc", x=["x1"], group="g", cluster="town")
    assert not any("split across" in message for message in nested.warnings)


def test_clogit_frequency_weights_count_dropped_groups_as_replicated(frame):
    """Defect: with fweights the dropped groups and observations were counted unweighted."""
    weights = frame.assign(f=(frame.g % 3 + 1).astype(float))
    result = oe.clogit(data=weights, y="yc", x=["x1", "x2"], group="g", weights="f",
                       weight_type="fweight")
    share = weights.groupby("g").yc.transform("mean")
    lost = weights[(share == 0) | (share == 1)]
    groups = lost.groupby("g").f.first().sum()
    assert groups > lost.g.nunique() > 0                       # the weights matter here
    assert result.metrics["n_groups_dropped"] == groups
    assert result.extra["observations_dropped"] == lost.f.sum()
    assert f"note: {int(groups)} group(s) ({int(lost.f.sum())} obs) dropped" in result.warnings[0]
    kept = weights[(share > 0) & (share < 1)]
    assert result.metrics["n_groups"] == kept.groupby("g").f.first().sum()
    assert result.nobs == kept.f.sum()
    sizes = kept.groupby("g").agg(size=("f", "size"), f=("f", "first"))
    assert_allclose(result.extra["group_sizes"]["mean"],
                    (sizes["size"] * sizes.f).sum() / sizes.f.sum())


@pytest.mark.parametrize("command", ["ologit", "oprobit"])
def test_ordered_separation_of_a_cutpoint_is_reported_as_separation(frame, command):
    """Defect: a regressor separating the two sides of ONE cutpoint ended as 'nonconvergence'."""
    split = frame.assign(d=(frame.yo >= 2) * 1.0)
    with pytest.raises(AnalysisError) as caught:
        getattr(oe, command)(data=split, y="yo", x=["x1", "d"])
    assert caught.value.code == "separation_detected"
    assert "separation" in str(caught.value) and "Remove or combine" in str(caught.value)
    # A strong but imperfect predictor is still estimated.
    rng = np.random.default_rng(2)
    strong = frame.assign(z=frame.yo + rng.normal(scale=0.45, size=N))
    result = finite(getattr(oe, command)(data=strong, y="yo", x=["z"]))
    assert result.coefficients[0].estimate > 2
    # Quasi-separation that only bounds a slope from one side has a finite maximum.
    partial = frame.assign(d=((frame.yo >= 2) & (rng.random(N) < 0.5)) * 1.0)
    finite(getattr(oe, command)(data=partial, y="yo", x=["x1", "d"]))


def test_mlogit_quasi_separation_is_detected_early(frame):
    """Defect: a category ruled out for part of the sample ran 200 iterations to 'nonconvergence'."""
    rng = np.random.default_rng(4)
    never_base = frame.assign(d=((frame.ym > 0) & (rng.random(N) < 0.5)) * 1.0)
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=never_base, y="ym", x=["x1", "d"], base=0)
    assert caught.value.code == "separation_detected"
    only_one = frame.assign(d=((frame.ym == 2) & (rng.random(N) < 0.5)) * 1.0)
    assert code(lambda: oe.mlogit(data=only_one, y="ym", x=["x1", "d"])) == "separation_detected"
    complete = frame.assign(z=frame.ym + rng.uniform(-0.1, 0.1, size=N))
    assert code(lambda: oe.mlogit(data=complete, y="ym", x=["z"])) == "separation_detected"


def test_biprobit_empty_outcome_cell_is_a_boundary_solution(frame):
    """Defect: an empty cell of the 2x2 table crept for 200 iterations to 'nonconvergence'."""
    empty = frame.assign(yb2=np.where(frame.yb == 1, 1.0, frame.yb2))
    assert ((empty.yb == 1) & (empty.yb2 == 0)).sum() == 0
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=empty, y1="yb", y2="yb2", x=["x1", "x2"])
    assert caught.value.code == "boundary_solution"
    assert "2x2 table" in str(caught.value) and "+1" in str(caught.value)
    assert code(lambda: oe.biprobit(data=frame.assign(yb2=frame.yb), y1="yb", y2="yb2",
                                    x=["x1"])) == "boundary_solution"
    mirror = code(lambda: oe.biprobit(data=frame.assign(yb2=1 - frame.yb), y1="yb", y2="yb2",
                                      x=["x1"]))
    assert mirror == "boundary_solution"
    # A high but interior correlation is estimated, not refused.
    rng = np.random.default_rng(6)
    shock = rng.multivariate_normal([0, 0], [[1, 0.97], [0.97, 1]], size=4000)
    close = pd.DataFrame({"x": rng.normal(size=4000)})
    close["a"] = (0.2 + 0.5 * close.x + shock[:, 0] > 0) * 1.0
    close["b"] = (-0.4 + 0.3 * close.x + shock[:, 1] > 0) * 1.0
    result = finite(oe.biprobit(data=close, y1="a", y2="b", x=["x"]))
    assert 0.94 < result.metrics["rho"] < 0.995


def test_hetprobit_scale_outside_float64_is_reported_as_such(frame):
    """Defect: exp(mean(z)'g) between e^300 and e^700 ended as 'invalid_covariance'."""
    reference = finite(oe.hetprobit(data=frame, y="yb", x=["x1"], het=["x2"]))
    gamma = reference.coefficients[-1].estimate
    assert reference.coefficients[-1].term == "lnsigma:x2" and abs(gamma) > 1e-3
    for target in (400.0, -400.0, 5000.0):
        moved = frame.assign(x2=frame.x2 + target / gamma)
        with pytest.raises(AnalysisError) as caught:
            oe.hetprobit(data=moved, y="yb", x=["x1"], het=["x2"])
        assert caught.value.code == "variance_scale_overflow"
        assert "Centre the het columns" in str(caught.value)
    # Inside the range the same model is returned on its own scale, with a warning.
    moved = finite(oe.hetprobit(data=frame.assign(x2=frame.x2 + 100.0 / gamma), y="yb",
                                x=["x1"], het=["x2"]))
    assert_allclose(moved.coefficients[-1].estimate, gamma, rtol=1e-6)
    assert_allclose(moved.coefficients[1].estimate,
                    reference.coefficients[1].estimate * math.exp(100.0), rtol=1e-5)
    assert_allclose(moved.metrics["log_likelihood"], reference.metrics["log_likelihood"],
                    rtol=1e-9)
    assert any("large means" in message for message in moved.warnings)
    # Collinear and constant variance regressors.
    extended = frame.assign(z=2 * frame.x2 + 1, one=1.0)
    result = finite(oe.hetprobit(data=extended, y="yb", x=["x1"], het=["x2", "z", "one"]))
    assert result.provenance["omitted_terms"] == ["lnsigma:z", "lnsigma:one"]
    assert code(lambda: oe.hetprobit(data=extended, y="yb", x=["x1"], het=["one"])) \
        == "no_variance_regressors"


@pytest.mark.parametrize("command", COMMANDS)
def test_estimates_do_not_depend_on_the_level_of_a_regressor(frame, command):
    """Defect: standard errors lost (level / spread)^2 digits (0.4% at a level of 1e6) and
    the iteration failed at 1e8; equations with a constant are now fitted on centred data."""
    call = CALLS[command]
    reference = finite(call(frame))
    names = [row.term for row in reference.coefficients]
    values = np.array([row.estimate for row in reference.coefficients])
    covariance = np.array(reference.covariance_matrix)
    slopes = [i for i, name in enumerate(names)
              if not name.endswith("Intercept") and not name.startswith("/cut")]
    for level in (1e6, -1e8):
        moved = finite(call(frame.assign(x1=frame.x1 + level)))
        got = np.array([row.estimate for row in moved.coefficients])
        assert_allclose(got[slopes], values[slopes], rtol=1e-6 if abs(level) < 1e7 else 1e-5)
        assert_allclose(np.array(moved.covariance_matrix)[np.ix_(slopes, slopes)],
                        covariance[np.ix_(slopes, slopes)], rtol=1e-5)
        assert_allclose(moved.metrics["log_likelihood"], reference.metrics["log_likelihood"],
                        rtol=1e-9)
        assert moved.provenance["optimizer"]["iterations"] \
            == reference.provenance["optimizer"]["iterations"]
    # The constants move exactly as the algebra says: with x1 -> x1 + c the index is
    # unchanged when a -> a - c b1 (cut -> cut + c b1), and the covariance follows J V J'.
    shift = 1000.0
    moved = finite(call(frame.assign(x1=frame.x1 + shift)))
    jacobian = np.eye(len(names))
    for row, name in enumerate(names):
        if name.endswith("Intercept"):
            partner = names.index(name.replace("Intercept", "x1"))
            jacobian[row, partner] = -shift
        elif name.startswith("/cut"):
            jacobian[row, names.index("x1")] = shift
    assert_allclose([row.estimate for row in moved.coefficients], jacobian @ values,
                    rtol=1e-8, atol=1e-9)
    assert_allclose(np.array(moved.covariance_matrix), jacobian @ covariance @ jacobian.T,
                    rtol=1e-7, atol=1e-10)
    for name in reference.tests:
        assert_allclose(moved.tests[name]["statistic"], reference.tests[name]["statistic"],
                        rtol=1e-7)
