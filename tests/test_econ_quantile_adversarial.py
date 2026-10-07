"""Adversarial inputs and regression tests for the quantile family.

Every invalid input or numerical dead end must raise ``AnalysisError`` with a
snake_case code and a message that says what to change; every result must be
finite and JSON-serializable. The second half pins the defects found by the
verification pass (see the test names starting with ``test_fix_``).
"""

import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.quantile import formula as formulas
from openecon.econometrics.quantile import kernels
from openecon.models import ModelSpec

X = ["x1", "x2"]
DECAY = "{b0} + {b1} * exp(-{b2} * x)"
START = {"b0": 1, "b1": 1, "b2": 0.5}


def make_data(n=160, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "x": rng.uniform(0, 5, size=n)})
    frame["y"] = 1 + frame.x1 - frame.x2 + rng.standard_t(3, size=n)
    frame["z"] = 1.5 + 2.5 * np.exp(-0.8 * frame.x) + rng.normal(0, 0.2, size=n)
    frame["g"] = rng.integers(0, 12, size=n)
    frame["w"] = rng.uniform(0.5, 2, size=n)
    frame["label"] = "a"
    frame["rare"] = (np.arange(n) < 3).astype(float)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def code_of(function, **arguments):
    with pytest.raises(AnalysisError) as caught:
        function(**arguments)
    message = str(caught.value)
    assert caught.value.code == caught.value.code.lower() and " " not in caught.value.code
    assert len(message) > 20 and "Traceback" not in message
    return caught.value.code


def assert_sound(result):
    """Finite, JSON-safe output with positive standard errors."""
    payload = json.loads(result.model_dump_json())
    assert type(result).model_validate(payload) == result

    def walk(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    assert all(c.std_error > 0 for c in result.coefficients)
    assert result.provenance["stata_parity_validated"] is False
    assert len({c.term for c in result.coefficients}) == len(result.coefficients)


FITTERS = {
    "qreg": lambda frame, **kw: oe.qreg(data=frame, y="y", x=X, **kw),
    "bsqreg": lambda frame, **kw: oe.bsqreg(data=frame, y="y", x=X, seed=3, **kw),
    "sqreg": lambda frame, **kw: oe.sqreg(data=frame, y="y", x=X, seed=3, **kw),
    "iqreg": lambda frame, **kw: oe.iqreg(data=frame, y="y", x=X, seed=3, **kw),
    "rreg": lambda frame, **kw: oe.rreg(data=frame, y="y", x=X, **kw),
    "nl": lambda frame, **kw: oe.nl(data=frame, y="z", formula=DECAY, start=START, **kw),
}


@pytest.mark.parametrize("name", list(FITTERS))
def test_every_estimator_returns_sound_results_and_handles_bad_samples(data, name):
    fit = FITTERS[name]
    result = fit(data)
    assert_sound(result)
    assert result.spec.estimator == name and result.provenance["family"] == "quantile"
    assert result.inference["use_t"] is True
    assert callable(getattr(oe, name)) and len(getattr(oe, name).__doc__) > 800
    assert "Stata" in getattr(oe, name).__doc__ and "Example" in getattr(oe, name).__doc__
    # The sample: empty, too small, missing values, non-numeric and non-finite columns.
    assert code_of(fit, frame=data.iloc[:0]) == "empty_data"
    assert code_of(fit, frame=data.iloc[:1]) == "insufficient_observations"
    assert code_of(fit, frame=data.iloc[:3]) == "insufficient_observations"
    outcome, regressor = ("z", "x") if name == "nl" else ("y", "x1")
    holes = data.copy()
    holes.loc[[4, 9], regressor] = np.nan
    assert code_of(fit, frame=holes) == "missing_values"
    dropped = fit(holes, missing="drop")
    assert dropped.dropped_rows >= 2 and any("missing" in w for w in dropped.warnings)
    assert 4 not in dropped.sample_positions and 9 not in dropped.sample_positions
    assert code_of(fit, frame=data.assign(**{regressor: np.nan}), missing="drop") == "empty_sample"
    assert code_of(fit, frame=data.assign(**{regressor: data.label})) == "non_numeric_column"
    assert code_of(fit, frame=data.assign(**{outcome: data.label})) == "non_numeric_column"
    spoiled = data.copy()
    spoiled.loc[7, outcome] = np.inf
    assert code_of(fit, frame=spoiled) == "non_finite_values"
    assert code_of(fit, frame=data.drop(columns=[regressor])) == "missing_columns"
    assert code_of(fit, frame=data, alpha=0) == "invalid_spec"
    assert code_of(fit, frame=data, missing="ignore") == "invalid_spec"
    # A constant outcome has nothing to estimate a variance from.
    constant = code_of(fit, frame=data.assign(**{outcome: 2.0}))
    assert constant in {"perfect_fit", "zero_scale"}


@pytest.mark.parametrize("name", list(FITTERS))
def test_extreme_magnitudes_are_equivariant(data, name):
    fit = FITTERS[name]
    base = fit(data)
    outcome = "z" if name == "nl" else "y"
    for factor in (1e8, 1e-8):
        scaled = data.assign(**{outcome: data[outcome] * factor})
        if name == "nl":
            scaled_fit = oe.nl(data=scaled, y="z", formula=DECAY,
                               start={"b0": factor, "b1": factor, "b2": 0.5})
            expected = np.array([c.estimate for c in base.coefficients]) * [factor, factor, 1]
            errors = np.array([c.std_error for c in base.coefficients]) * [factor, factor, 1]
        else:
            scaled_fit = fit(scaled)
            expected = np.array([c.estimate for c in base.coefficients]) * factor
            errors = np.array([c.std_error for c in base.coefficients]) * factor
        assert_sound(scaled_fit)
        assert_allclose([c.estimate for c in scaled_fit.coefficients], expected, rtol=1e-6)
        assert_allclose([c.std_error for c in scaled_fit.coefficients], errors, rtol=1e-5)
    if name != "nl":
        # A regressor in tiny or huge units, and an outcome with a huge level.
        for factor in (1e8, 1e-8):
            rescaled = fit(data.assign(x1=data.x1 * factor))
            assert_allclose(rescaled.coefficients[1].estimate * factor,
                            base.coefficients[1].estimate, rtol=1e-6)
            assert_allclose(rescaled.coefficients[1].statistic, base.coefficients[1].statistic,
                            rtol=1e-5)
        shifted = fit(data.assign(y=data.y + 1e9))
        assert_allclose(shifted.coefficients[1].estimate, base.coefficients[1].estimate,
                        rtol=1e-5)


# ---- qreg --------------------------------------------------------------------------------


def test_qreg_option_and_specification_errors(data):
    def run(**options):
        return oe.qreg(**{"data": data, "y": "y", "x": X, **options})

    for quantile in (0, 1, -0.1, 1.5, 50):
        assert code_of(run, quantile=quantile) == "invalid_quantile"
    for quantile in (True, "0.5", float("nan"), [0.5]):
        assert code_of(run, quantile=quantile) in {"invalid_spec", "invalid_quantile"}
    assert run(quantile=None).metrics["quantile"] == 0.5        # None means the default
    cases = [
        ({"kernel": "uniform"}, "invalid_spec"), ({"bandwidth": "silverman"}, "invalid_spec"),
        ({"density": "histogram"}, "invalid_spec"), ({"covariance": "HC1"}, "invalid_spec"),
        ({"covariance": "bootstrap"}, "invalid_spec"),
        ({"covariance": "cluster"}, "invalid_spec"),
        ({"cluster": "g", "covariance": "robust"}, "invalid_spec"),
        ({"cluster": "y"}, "invalid_spec"),
        ({"density": "residual", "kernel": "gaussian"}, "invalid_spec"),
        ({"density": "fitted", "kernel": "gaussian"}, "invalid_spec"),
        ({"covariance": "robust", "density": "residual"}, "invalid_spec"),
        ({"cluster": "g", "density": "residual"}, "invalid_spec"),
        ({"x": "x1"}, "invalid_spec"), ({"x": ["x1", "x1"]}, "invalid_spec"),
        ({"x": ["y", "x1"]}, "invalid_spec"), ({"x": [], "intercept": False}, "no_regressors"),
        ({"weights": "w"}, "invalid_spec"), ({"weight_type": "aweight"}, None),
        ({"weights": "w", "weight_type": "iweight"}, "invalid_spec"),
        ({"weights": "w", "weight_type": "pweight", "covariance": "nonrobust"},
         "unsupported_covariance"),
        ({"cluster": "label"}, "insufficient_clusters"),
    ]
    for options, code in cases:
        if code is None:
            assert_sound(run(**options))      # weight_type without weights is ignored
        else:
            assert code_of(run, **options) == code, options
    assert code_of(run, data=data.assign(w=-data.w), weights="w",
                   weight_type="aweight") == "negative_weights"
    assert code_of(run, data=data.assign(w=data.w + 0.25), weights="w",
                   weight_type="fweight") == "noninteger_frequency_weights"
    assert code_of(run, data=data.assign(w=0.0), weights="w",
                   weight_type="aweight") == "empty_sample"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="qreg", outcome="y", predictors=X, options={"quantile": "half"})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="qreg", outcome="y", predictors=X, panel="g")


def test_qreg_small_samples_extreme_quantiles_and_discrete_outcomes(data):
    # tau +- h must stay inside (0, 1): tiny samples and extreme quantiles say what to do.
    for frame, quantile in ((data.iloc[:6], 0.5), (data, 0.999), (data.iloc[:40], 0.03)):
        with pytest.raises(AnalysisError) as caught:
            oe.qreg(data=frame, y="y", x=X, quantile=quantile)
        assert caught.value.code == "bandwidth_out_of_range" and "bsqreg" in str(caught.value)
    assert_sound(oe.bsqreg(data=data, y="y", x=X, quantile=0.999, seed=1))
    assert_sound(oe.qreg(data=data.iloc[:14], y="y", x=X))
    assert_sound(oe.qreg(data=data.iloc[:14], y="y", x=X, covariance="robust"))
    # Discrete outcomes produce ties: the result is still an exact minimizer.
    binary = data.assign(y=(data.y > 1).astype(float))
    for options in ({}, {"covariance": "robust"}, {"density": "kernel"}):
        try:
            assert_sound(oe.qreg(data=binary, y="y", x=X, **options))
        except AnalysisError as error:
            assert error.code in {"degenerate_sparsity", "singular_density_matrix"}
    rounded = oe.qreg(data=data.assign(y=data.y.round()), y="y", x=X)
    assert_sound(rounded)
    dummy = oe.qreg(data=data.assign(d=(data.x1 > 0).astype(float)), y="y", x=["d"])
    assert dummy.extra["unique_solution"] in (True, False)
    # Exact linear outcome, collinear and constant regressors, a boolean regressor.
    assert code_of(oe.qreg, data=data.assign(y=1 + 2 * data.x1), y="y", x=X) == "perfect_fit"
    collinear = oe.qreg(data=data.assign(twice=2 * data.x1, one=1.0), y="y",
                        x=["x1", "twice", "one", "x2"])
    assert [c.term for c in collinear.coefficients] == ["Intercept", "x1", "x2"]
    assert collinear.provenance["omitted_terms"] == ["twice", "one"]
    assert any("collinearity" in warning for warning in collinear.warnings)
    assert_allclose([c.estimate for c in collinear.coefficients],
                    [c.estimate for c in oe.qreg(data=data, y="y", x=X).coefficients], rtol=1e-9)
    assert_sound(oe.qreg(data=data.assign(b=data.x1 > 0), y="y", x=["b", "x2"]))
    two = oe.qreg(data=data.assign(g=np.arange(len(data)) % 2), y="y", x=X, cluster="g")
    assert two.inference["cluster_count"] == 2 and any("clusters" in w for w in two.warnings)
    # Weights of wildly different magnitude, zero weights, huge frequency weights.
    huge = oe.qreg(data=data.assign(w=data.w * 1e12), y="y", x=X, weights="w",
                   weight_type="aweight")
    plain = oe.qreg(data=data, y="y", x=X, weights="w", weight_type="aweight")
    assert_allclose([c.std_error for c in huge.coefficients],
                    [c.std_error for c in plain.coefficients], rtol=1e-7)
    zeroed = data.assign(w=np.where(np.arange(len(data)) % 4 == 0, 0.0, data.w))
    some = oe.qreg(data=zeroed, y="y", x=X, weights="w", weight_type="aweight")
    assert some.nobs == int((zeroed.w > 0).sum()) and any("zero weight" in w for w in some.warnings)
    big = oe.qreg(data=data.assign(f=1000.0), y="y", x=X, weights="f", weight_type="fweight")
    assert big.nobs == 1000 * len(data)
    assert_allclose(big.coefficients[1].estimate, oe.qreg(data=data, y="y", x=X)
                    .coefficients[1].estimate, rtol=1e-9)


def test_quantile_solver_rejects_invalid_kernel_input():
    from openecon.engines.contracts import KernelError

    x = torch.ones((5, 1), dtype=torch.float64)
    y = torch.arange(5, dtype=torch.float64)
    cases = [
        (lambda: kernels.prepare(x.float(), y), "invalid_design"),
        (lambda: kernels.prepare(x, y[:4]), "invalid_design"),
        (lambda: kernels.prepare(x[:1], y[:1]), "insufficient_observations"),
        (lambda: kernels.prepare(x, y, torch.zeros(5, dtype=torch.float64)), "invalid_weights"),
        (lambda: kernels.prepare(torch.zeros((5, 1), dtype=torch.float64), y), "singular_design"),
        (lambda: kernels.prepare(torch.ones((5, 2), dtype=torch.float64), y), "singular_design"),
        (lambda: kernels.solve(kernels.prepare(x, y), 1.0), "invalid_quantile"),
        (lambda: kernels.bandwidth(0.5, 0), "insufficient_observations"),
        (lambda: kernels.bandwidth(0.5, 10, "plug-in"), "unsupported_bandwidth"),
        (lambda: kernels.kernel_values(y, "uniform"), "unsupported_kernel"),
    ]
    for call, code in cases:
        with pytest.raises(KernelError) as caught:
            call()
        assert caught.value.code == code
    nan = y.clone()
    nan[2] = float("nan")
    with pytest.raises(KernelError):
        kernels.prepare(x, nan)


# ---- bootstrap commands ----------------------------------------------------------------------


def test_bootstrap_option_errors_and_fragile_resamples(data):
    for name in ("bsqreg", "sqreg", "iqreg"):
        function = getattr(oe, name)
        assert code_of(function, data=data, y="y", x=X, reps=1) == "invalid_spec"
        assert code_of(function, data=data, y="y", x=X, reps=2.5) == "invalid_spec"
        assert code_of(function, data=data, y="y", x=X, seed=-1) == "invalid_spec"
        assert code_of(function, data=data, y="y", x=X, seed=1.5) == "invalid_spec"
        assert code_of(function, data=data, y="y", x=X, cluster="label",
                       seed=1) == "insufficient_clusters"
        assert_sound(function(data=data, y="y", x=X, reps=2, seed=0))
        for options in ({"weights": "w", "weight_type": "fweight"}, {"covariance": "nonrobust"}):
            with pytest.raises(ValidationError):
                ModelSpec(estimator=name, outcome="y", predictors=X, **options)
    cases = [
        (oe.bsqreg, {"quantile": 0}), (oe.bsqreg, {"quantile": 1}),
        (oe.sqreg, {"quantiles": []}), (oe.sqreg, {"quantiles": [0.5, 0.5]}),
        (oe.sqreg, {"quantiles": [0.5, 1.0]}), (oe.sqreg, {"quantiles": "0.5"}),
        (oe.sqreg, {"quantiles": 0.5}), (oe.sqreg, {"quantiles": [[0.2], [0.8]]}),
        (oe.sqreg, {"quantiles": [True]}), (oe.sqreg, {"quantiles": [0.5, 0.5 + 1e-13]}),
        (oe.iqreg, {"quantiles": [0.75, 0.25]}), (oe.iqreg, {"quantiles": [0.25, 0.5, 0.75]}),
        (oe.iqreg, {"quantiles": [0.5]}),
    ]
    for function, options in cases:
        assert code_of(function, data=data, y="y", x=X, seed=1, **options) == "invalid_quantile"
    # A rare indicator is collinear in some resamples: they are skipped and reported.
    fragile = oe.bsqreg(data=data, y="y", x=["x1", "rare"], reps=40, seed=1)
    assert_sound(fragile)
    used = fragile.metrics["reps"]
    assert used < 40 and any(f"{40 - int(used)} of 40" in w for w in fragile.warnings)
    assert fragile.extra["bootstrap"]["reps_used"] == used
    # Unsorted quantiles are legal in sqreg and keep their order.
    unsorted = oe.sqreg(data=data, y="y", x=X, quantiles=[0.75, 0.25], reps=4, seed=1)
    assert unsorted.extra["equations"] == ["q75", "q25"]
    odd = oe.sqreg(data=data, y="y", x=X, quantiles=[0.025], reps=4, seed=1)
    assert odd.extra["equations"] == ["q2_5"] and odd.coefficients[0].term == "q2_5:Intercept"


# ---- rreg -----------------------------------------------------------------------------------


def test_rreg_option_errors_and_degenerate_fits(data):
    def run(**options):
        return oe.rreg(**{"data": data, "y": "y", "x": X, **options})

    cases = [
        ({"tune": 0}, "invalid_option"), ({"tune": -1}, "invalid_option"),
        ({"tolerance": 0}, "invalid_option"), ({"tolerance": 1}, "invalid_option"),
        ({"tune": "7"}, "invalid_spec"), ({"tolerance": None}, None),
        ({"covariance": "robust"}, "invalid_spec"), ({"x": None}, "invalid_spec"),
        ({"x": []}, "invalid_spec"),
    ]
    for options, code in cases:
        if code is None:
            assert_sound(run(**options))
        else:
            assert code_of(run, **options) == code, options
    for options in ({"weights": "w", "weight_type": "aweight"}, {"cluster": "g"},
                    {"covariance": "cluster", "cluster": "g"}):
        with pytest.raises(ValidationError):
            ModelSpec(estimator="rreg", outcome="y", predictors=X, **options)
    for options in ({"tune": 0.5}, {"tune": 1e6}, {"tolerance": 0.999}, {"tolerance": 1e-10},
                    {"intercept": False}):
        assert_sound(run(**options))
    assert code_of(run, data=data.assign(y=1 + 2 * data.x1)) in {"perfect_fit", "zero_scale"}
    # An outcome that is exact for most rows: the MAD scale collapses.
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x1": rng.normal(size=60)})
    frame["y"] = 2 * frame.x1
    frame.loc[:9, "y"] += rng.normal(size=10) * 0.01
    for tolerance in (0.01, 1e-6):
        assert code_of(oe.rreg, data=frame, y="y", x=["x1"], tolerance=tolerance) == "zero_scale"
    # A gross outlier with leverage is screened out by Cook's distance and reported.
    outlier = data.copy()
    outlier.loc[0, ["x1", "y"]] = [25.0, -300.0]
    screened = oe.rreg(data=outlier, y="y", x=X)
    assert screened.metrics["n_dropped_cooks"] == 1 and screened.nobs == len(data) - 1
    assert 0 not in screened.sample_positions
    assert any("Cook's distance" in warning for warning in screened.warnings)
    clean = oe.rreg(data=data.iloc[1:].reset_index(drop=True), y="y", x=X)
    assert_allclose([c.estimate for c in screened.coefficients],
                    [c.estimate for c in clean.coefficients], rtol=1e-9)
    collinear = oe.rreg(data=data.assign(twice=2 * data.x1), y="y", x=["x1", "twice", "x2"])
    assert collinear.provenance["omitted_terms"] == ["twice"]


# ---- nl -------------------------------------------------------------------------------------


def test_nl_formula_is_never_executed_and_grammar_errors_are_explained(data):
    hostile = [
        "{a} * __import__('os').system('echo pwned')", "{a} + open('/etc/passwd').read()",
        "{a} * x.__class__", "(lambda: 1)() + {a}", "{a} * x[0]", "{a} * (x > 1)",
        "{a} if x else {b}", "[{a} for _ in x]", "{a} + 'text'", "{a}; import os",
        "{a} + exp(x, 2)", "{a} + exp(u=x)", "{a} + pow(x, 2)", "{a} + eval('1')",
        "{a} and x", "{a} @ x", "{a} // x", "{a} % x", "{a} + x := 3", "{a} + 1j",
        "{a} + 1e999", "{a} + f(x)", "{a} * (x", "{a} {b}", "{a} +", "",
        "   ", "y = {a} * x", "{a b} * x", "{xb: x1 x2}", "{a=one} * x", "{a=1} + {a=2}*x",
        "{a} * x^2^3", "{a} ^ -x ^ 2", "{} * x", "{a} * {", "x + 1",
        "{a} * __oe_parameter_b", "{a}" + " + 1" * 3000,
    ]
    for text in hostile:
        with pytest.raises(AnalysisError) as caught:
            oe.nl(data=data, y="z", formula=text)
        assert caught.value.code == "invalid_formula", text
        assert len(str(caught.value)) > 20
    for text in (None, 3, ["{a}"]):
        assert code_of(oe.nl, data=data, y="z", formula=text) == "invalid_formula"
    assert code_of(oe.nl, data=data, y="z", formula="{a} * absent") == "missing_columns"
    assert code_of(oe.nl, data=data, y="z", formula="{a} * label") == "non_numeric_column"
    assert code_of(oe.nl, data=data, y="z", formula="{a} * z") == "invalid_spec"
    # Legal spellings of powers and signs.
    for text in ("{a} * 2^-(x^2) + {b}", "{a} * 2^(-x^2) + {b}", "{a} * (x^2)^0.5 + {b}",
                 "{a} * x ** 2 + {b}", "-{a} * -x + +{b}", "{a} * (-x^2)^2 + {b}"):
        assert_sound(oe.nl(data=data, y="z", formula=text))
    square = oe.nl(data=data, y="z", formula="{a} * -x^2 + {b}")
    x = data.x.to_numpy()
    design = np.column_stack([-(x ** 2), np.ones(len(x))])
    assert_allclose([c.estimate for c in square.coefficients],
                    np.linalg.lstsq(design, data.z.to_numpy(), rcond=None)[0], rtol=1e-8)


def test_nl_start_values_convergence_and_identification_errors(data):
    def run(**options):
        return oe.nl(**{"data": data, "y": "z", "formula": DECAY, "start": START, **options})

    cases = [
        ({"start": {"zz": 1}}, "invalid_start"), ({"start": {"b0": "1"}}, "invalid_start"),
        ({"start": [1, 2, 3]}, "invalid_start"), ({"start": {"b0": float("nan")}}, "invalid_start"),
        ({"start": {"b0": True}}, "invalid_start"),
        ({"start": {"b0": 1, "b1": 1, "b2": -1000}}, "invalid_start"),
        ({"formula": "{a} + {b} * ln(x1)", "start": {"a": 1, "b": 1}}, "invalid_start"),
        ({"formula": "{a} / {b} * x", "start": None}, "invalid_start"),
        ({"max_iterations": 1}, "nonconvergence"), ({"max_iterations": 0}, "invalid_spec"),
        ({"tolerance": 0}, "invalid_option"), ({"tolerance": 1}, "invalid_option"),
        ({"formula": "{a} + {b} + {c} * x", "start": None}, "not_identified"),
        ({"formula": "{a} * {b} * x + {c}", "start": {"a": 1, "b": -0.5, "c": 1}},
         "not_identified"),
        ({"formula": "{a} * {b} * x + {c}", "start": {"a": 1, "b": 1, "c": 1}}, "nonconvergence"),
        ({"covariance": "HC1"}, "invalid_spec"), ({"covariance": "opg"}, "invalid_spec"),
        ({"cluster": "label"}, "insufficient_clusters"),
        ({"weights": "w", "weight_type": "iweight"}, "invalid_spec"),
        ({"weights": "w", "weight_type": "pweight", "covariance": "nonrobust"},
         "unsupported_covariance"),
        ({"data": data.assign(z=1.5 + 2.5 * np.exp(-0.8 * data.x))}, "perfect_fit"),
        ({"data": data.iloc[:3]}, "insufficient_observations"),
    ]
    for options, code in cases:
        assert code_of(run, **options) == code, options
    with pytest.raises(AnalysisError) as caught:
        run(formula="{a} + {b} + {c} * x", start=None)
    assert "b" in str(caught.value) and "cannot be estimated" in str(caught.value)
    # No starting values: zeros with a warning; far starting values still converge.
    blind = run(start=None)
    assert any("started at 0" in warning for warning in blind.warnings)
    far = run(start={"b0": 100, "b1": -50, "b2": 5})
    reference = run()
    for other in (blind, far):
        assert_allclose([c.estimate for c in other.coefficients],
                        [c.estimate for c in reference.coefficients], rtol=1e-6)
    assert reference.warnings == []
    tiny = run(data=data.iloc[:4])
    assert tiny.nobs == 4 and tiny.metrics["df_resid"] == 1
    # Specs built directly: the registry contract and the formula's columns.
    with pytest.raises(ValidationError):
        ModelSpec(estimator="nl", outcome="z", predictors=["x"], options={"formula": DECAY})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="nl", outcome="z", predictors=[], intercept=False)
    spec = ModelSpec(estimator="nl", outcome="z", predictors=[], intercept=False,
                     options={"formula": DECAY, "start": START})
    assert_allclose([c.estimate for c in oe.fit(spec, data=data).coefficients],
                    [c.estimate for c in reference.coefficients], rtol=1e-12)
    stray = ModelSpec(estimator="nl", outcome="z", predictors=["x", "x1"], intercept=False,
                      options={"formula": DECAY, "start": START})
    assert code_of(oe.fit, spec=stray, data=data) == "invalid_spec"


# ---- the manifest ------------------------------------------------------------------------------


def test_manifest_loads_without_the_tensor_runtime_and_declares_what_is_handled(data):
    program = ("import sys; import openecon.econometrics.quantile as m; "
               "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
               "print(len(m.ESTIMATORS))")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "6", done.stderr
    listing = oe.capabilities()["estimators"]
    expected = {
        "qreg": (["nonrobust", "robust", "cluster"], ["aweight", "fweight", "pweight"]),
        "bsqreg": (["bootstrap"], []), "sqreg": (["bootstrap"], []), "iqreg": (["bootstrap"], []),
        "rreg": (["nonrobust"], []),
        "nl": (["nonrobust", "robust", "HC2", "HC3", "cluster"],
               ["aweight", "fweight", "pweight"]),
    }
    for name, (covariances, weights) in expected.items():
        record = listing[name]
        assert record["family"] == "quantile" and record["inference"] == "Student t"
        assert record["covariances"] == covariances and record["weights"] == weights
        assert record["stata"] == [name] and record["function"] == name
    # Every declared covariance and weight type is really handled.
    integer = data.assign(w=np.ceil(data.w))
    for covariance in expected["qreg"][0]:
        cluster = "g" if covariance == "cluster" else None
        for weight in (None, *expected["qreg"][1]):
            if weight == "pweight" and covariance == "nonrobust":
                continue
            assert_sound(oe.qreg(data=integer, y="y", x=X, covariance=covariance, cluster=cluster,
                                 weights="w" if weight else None, weight_type=weight))
    for covariance in expected["nl"][0]:
        cluster = "g" if covariance == "cluster" else None
        for weight in (None, *expected["nl"][1]):
            if weight == "pweight" and covariance == "nonrobust":
                continue
            assert_sound(oe.nl(data=integer, y="z", formula=DECAY, start=START,
                               covariance=covariance, cluster=cluster,
                               weights="w" if weight else None, weight_type=weight))
    summary = oe.sqreg(data=data, y="y", x=X, reps=3, seed=1).summary()
    assert "q25" in summary and "q75" in summary
    assert "tabular" in oe.rreg(data=data, y="y", x=X).to_latex()


# ---- regression tests for the defects fixed by the verification pass ----------------------------


def test_fix_raw_quantile_is_the_order_statistic_int_tau_n_plus_1():
    # Stata prints "Raw sum of deviations 41912.75 (about 4187)" for the .25 quantile of the
    # 74 auto prices: the 18th order statistic, not the 19th that minimizes the check loss.
    values = np.arange(1.0, 75.0) ** 1.5
    tensor = torch.tensor(values)
    assert kernels.raw_quantile(tensor, 0.25) == values[17]
    assert kernels.raw_quantile(tensor, 0.75) == values[55]
    assert kernels.raw_quantile(tensor, 0.5) == values[36]
    assert kernels.raw_quantile(tensor, 0.001) == values[0]
    assert kernels.raw_quantile(tensor, 0.999) == values[73]
    frame = pd.DataFrame({"y": values, "x": np.cos(np.arange(74.0))})
    result = oe.qreg(data=frame, y="y", x=["x"], quantile=0.25)
    assert result.metrics["raw_quantile"] == values[17]
    loss = np.where(values < values[17], 0.75 * (values[17] - values), 0.25 * (values - values[17]))
    assert_allclose(result.metrics["sum_rdev"], loss.sum(), rtol=1e-12)
    # Frequency weights: the same order statistic of the expanded outcome.
    counts = torch.tensor([3.0, 1.0, 2.0, 5.0, 1.0])
    small = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])
    expanded = np.sort(np.repeat(small.numpy(), counts.numpy().astype(int)))
    for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
        position = min(max(int(np.floor(tau * 13 + 1e-9)), 1), 12)
        assert kernels.raw_quantile(small, tau, counts) == expanded[position - 1]


def test_fix_kernel_bandwidth_scale_is_iqr_over_1_34(data):
    # Stata's e(kbwidth): min(sd, IQR / 1.34) (Phi^-1(tau + h) - Phi^-1(tau - h)), not 1.349.
    result = oe.qreg(data=data, y="y", x=X, density="kernel")
    beta = np.array([c.estimate for c in result.coefficients])
    resid = data.y.to_numpy() - np.column_stack([np.ones(len(data)), data[X]]) @ beta
    iqr = (np.quantile(resid, 0.75, method="averaged_inverted_cdf")
           - np.quantile(resid, 0.25, method="averaged_inverted_cdf"))
    h = result.metrics["bandwidth"]
    expected = min(resid.std(ddof=1), iqr / 1.34) * (stats.norm.ppf(0.5 + h)
                                                    - stats.norm.ppf(0.5 - h))
    assert_allclose(result.metrics["kernel_bandwidth"], expected, rtol=1e-9)


def test_fix_robust_covariance_defaults_to_the_fitted_density(data):
    # Stata's vce(robust) uses denmethod(fitted) unless a kernel is requested.
    default = oe.qreg(data=data, y="y", x=X, covariance="robust")
    fitted = oe.qreg(data=data, y="y", x=X, covariance="robust", density="fitted")
    kernel = oe.qreg(data=data, y="y", x=X, covariance="robust", kernel="epanechnikov")
    assert default.extra["density_method"] == "fitted"
    assert default.covariance_matrix == fitted.covariance_matrix
    assert kernel.extra["density_method"] == "kernel"
    assert kernel.covariance_matrix != default.covariance_matrix
    assert "zero_density_observations" in default.extra
    assert "zero_density_observations" not in kernel.extra
    assert default.metrics["kernel_bandwidth"] is None
    clustered = oe.qreg(data=data, y="y", x=X, cluster="g")
    assert clustered.extra["density_method"] == "kernel"
    pweighted = oe.qreg(data=data, y="y", x=X, weights="w", weight_type="pweight")
    assert pweighted.spec.covariance == "robust"
    assert pweighted.extra["density_method"] == "fitted"


def test_fix_rreg_reports_the_irls_fit_with_the_stata_pseudovalue_covariance(data):
    result = oe.rreg(data=data, y="y", x=X, tolerance=1e-10)
    x = np.column_stack([np.ones(len(data)), data[X]])
    y = data.y.to_numpy()
    beta = np.array([c.estimate for c in result.coefficients])
    resid = y - x @ beta
    mad = np.median(np.abs(resid - np.median(resid)))
    u = resid / (mad / 0.6745) / 4.685
    weights = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    # The reported coefficients solve the weighted normal equations X'We = 0 ...
    assert np.abs(x.T @ (weights * resid)).max() < 1e-6
    # ... and the covariance uses lambda = 1 + K/(N - K) (1 - m)/m.
    n, k = x.shape
    m = np.where(np.abs(u) < 1, (1 - u ** 2) * (1 - 5 * u ** 2), 0.0).mean()
    lam = 1 + k / (n - k) * (1 - m) / m
    assert_allclose(result.extra["pseudovalues"]["lambda"], lam, rtol=1e-7)
    expected = (lam / m) ** 2 * ((weights * resid) ** 2).sum() / (n - k) * np.linalg.inv(x.T @ x)
    assert_allclose(np.array(result.covariance_matrix), expected, rtol=1e-6)
    assert result.extra["huber_cutoff_in_mad"] == 2.0
    assert_allclose(result.extra["huber_c"], 1.349, rtol=1e-12)
    first = result.extra["iteration_log"][0]
    ols = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    cutoff = 2 * np.median(np.abs(ols - np.median(ols)))
    assert_allclose(first["max_weight_change"], 1 - cutoff / np.abs(ols).max(), rtol=1e-9)


def test_fix_seed_beyond_the_generator_range_is_a_spec_error(data):
    assert code_of(oe.bsqreg, data=data, y="y", x=X, seed=2 ** 70) == "invalid_spec"
    assert_sound(oe.bsqreg(data=data, y="y", x=X, reps=3, seed=2 ** 63 - 1))


def test_fix_cooks_screening_that_leaves_too_few_rows_is_explained(data):
    with pytest.raises(AnalysisError) as caught:
        oe.rreg(data=data.iloc[:4], y="y", x=X)
    assert caught.value.code == "insufficient_observations"
    assert "Cook's distance" in str(caught.value) and "3 coefficients" in str(caught.value)


def test_fix_parenthesized_power_under_a_unary_minus_is_accepted():
    parsed = formulas.parse("{a} * 2^-(x^2)")
    x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    theta = torch.tensor([3.0], dtype=torch.float64)
    value, jacobian = formulas.evaluate(parsed, {"x": x}, theta, 3)
    assert_allclose(value.numpy(), 3.0 * 2.0 ** -(x.numpy() ** 2), rtol=1e-14)
    assert_allclose(jacobian[:, 0].numpy(), 2.0 ** -(x.numpy() ** 2), rtol=1e-14)
    for ambiguous in ("{a} * x^2^3", "{a} * 2^-x^2", "{a}^{a}^x"):
        with pytest.raises(AnalysisError):
            formulas.parse(ambiguous)


def test_fix_heavily_tied_problems_do_not_cycle():
    # A four-valued outcome on hundreds of indicator columns: thousands of tied residuals.
    # The degenerate pivoting used to cycle (and fail after its step limit); the vertex is
    # now certified by the interior dual point, and the simplex path keeps the vertex
    # exactly across degenerate pivots.
    from scipy import optimize, sparse

    rng = np.random.default_rng(1)
    n, levels = 5000, 300       # tau = 0.3 on this instance cycled before the fix
    frame = pd.DataFrame({"c": rng.integers(0, levels, size=n).astype(str),
                          "x": rng.integers(0, 3, size=n).astype(float)})
    frame["y"] = rng.integers(0, 4, size=n).astype(float) + frame.x
    dummies = pd.get_dummies(frame.c, drop_first=True).to_numpy(dtype=float)
    x = np.column_stack([np.ones(n), frame.x, dummies])
    y = frame.y.to_numpy()
    k = x.shape[1]
    for tau in (0.3, 0.5):
        cost = np.r_[np.zeros(k), np.full(n, tau), np.full(n, 1 - tau)]
        a_eq = sparse.hstack([sparse.csr_matrix(x), sparse.eye(n), -sparse.eye(n)]).tocsr()
        lp = optimize.linprog(cost, A_eq=a_eq, b_eq=y, method="highs",
                              bounds=[(None, None)] * k + [(0, None)] * (2 * n))
        r = y - x @ lp.x[:k]
        optimum = np.where(r < 0, (tau - 1) * r, tau * r).sum()
        result = oe.qreg(data=frame, y="y", x=["x", "c"], categorical=["c"], quantile=tau,
                         density="kernel")
        assert_allclose(result.metrics["sum_adev"], optimum, rtol=1e-10)
        assert result.provenance["solver_diagnostics"]["simplex_pivots"] <= 5
        assert_sound(result)
    # The simplex path alone (no interior-point iterations) on a smaller tied problem.
    small = slice(0, 500)
    columns = np.flatnonzero(np.abs(x[small]).sum(axis=0) > 0)
    xs = x[small][:, columns]
    xs = xs[:, [j for j in range(xs.shape[1])
                if np.linalg.matrix_rank(xs[:, :j + 1]) == j + 1]]
    problem = kernels.prepare(torch.tensor(xs), torch.tensor(y[small]))
    for tau in (0.3, 0.5):
        reference = kernels.solve(problem, tau)
        simplex = kernels.solve(problem, tau, max_iterations=0, max_pivots=100_000)
        assert simplex.pivots > 0
        assert_allclose(simplex.objective, reference.objective, rtol=1e-10)


def test_fix_quantile_lists_accept_tuples_and_arrays(data):
    reference = oe.iqreg(data=data, y="y", x=X, quantiles=[0.1, 0.9], reps=3, seed=2)
    for quantiles in ((0.1, 0.9), np.array([0.1, 0.9]), pd.Series([0.1, 0.9])):
        other = oe.iqreg(data=data, y="y", x=X, quantiles=quantiles, reps=3, seed=2)
        assert other.covariance_matrix == reference.covariance_matrix
