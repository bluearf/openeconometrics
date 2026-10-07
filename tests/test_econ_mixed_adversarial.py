"""Adversarial inputs for the mixed family: every case works or raises AnalysisError.

Each defect found in the verification pass has a regression test here (constant
outcomes, exact fits, collinear random effects, flat likelihoods reaching the
interval helpers, short GEE panels, quadrature memory, predictions).
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.glmm_kernels import NODE_LIMIT, RandomEffectsGLMM


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    groups, size = 30, 5
    g = np.repeat(np.arange(groups), size)
    n = len(g)
    frame = pd.DataFrame({"g": g, "t": np.tile(np.arange(size), groups),
                          "x": rng.normal(size=n), "w": rng.normal(size=n)})
    u = rng.normal(size=groups)[g]
    frame["y"] = (1 + frame.x + u + 0.7 * rng.normal(size=groups)[g] * frame.x
                  + rng.normal(size=n))
    frame["b"] = (rng.random(n) < 1 / (1 + np.exp(-(frame.x + u)))).astype(float)
    frame["c"] = rng.poisson(np.exp(0.3 + 0.5 * frame.x + 0.5 * u)).astype(float)
    frame["f"] = rng.integers(1, 3, n).astype(float)
    frame["e"] = rng.uniform(0.5, 2, n)
    return frame


COMMANDS = {
    "mixed": lambda d, **k: oe.mixed(data=d, y=k.pop("y", "y"), x=["x"], group="g", **k),
    "melogit": lambda d, **k: oe.melogit(data=d, y=k.pop("y", "b"), x=["x"], group="g", **k),
    "meprobit": lambda d, **k: oe.meprobit(data=d, y=k.pop("y", "b"), x=["x"], group="g", **k),
    "mepoisson": lambda d, **k: oe.mepoisson(data=d, y=k.pop("y", "c"), x=["x"], group="g", **k),
    "xtgee": lambda d, **k: oe.xtgee(data=d, y=k.pop("y", "y"), x=["x"], panel="g", time="t",
                                     **k),
    "xtlogit": lambda d, **k: oe.xtlogit(data=d, y=k.pop("y", "b"), x=["x"], panel="g", **k),
    "xtprobit": lambda d, **k: oe.xtprobit(data=d, y=k.pop("y", "b"), x=["x"], panel="g", **k),
    "xtpoisson": lambda d, **k: oe.xtpoisson(data=d, y=k.pop("y", "c"), x=["x"], panel="g",
                                             **k),
}
OUTCOME = {"mixed": "y", "xtgee": "y", "melogit": "b", "meprobit": "b", "xtlogit": "b",
           "xtprobit": "b", "mepoisson": "c", "xtpoisson": "c"}


def code(fn, *args, **kwargs):
    with pytest.raises(AnalysisError) as error:
        fn(*args, **kwargs)
    return error.value.code


@pytest.mark.parametrize("name", list(COMMANDS))
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_outcome_is_refused(data, name, value):
    """Regression: melogit raised a raw OverflowError, others misleading nonconvergence."""
    frame = data.assign(**{OUTCOME[name]: value})
    assert code(COMMANDS[name], frame) == "constant_outcome"


@pytest.mark.parametrize("name", list(COMMANDS))
def test_common_sample_failures(data, name):
    command = COMMANDS[name]
    assert code(command, data.iloc[:1]) in {"insufficient_groups", "constant_outcome"}
    assert code(command, data[data.g == 0]) in {"insufficient_groups", "constant_outcome"}
    assert code(command, data.assign(x="a")) == "non_numeric_column"
    assert code(command, data.assign(x=np.where(data.index == 3, np.inf, data.x))) \
        == "non_finite_values"
    assert code(command, data.assign(x=np.nan), missing="drop") == "empty_sample"
    missing = data.assign(x=np.where(data.index % 9 == 0, np.nan, data.x))
    assert code(command, missing) == "missing_values"
    dropped = command(missing, missing="drop")
    assert dropped.nobs == len(data) - int((data.index % 9 == 0).sum())
    assert dropped.dropped_rows == int((data.index % 9 == 0).sum())


@pytest.mark.parametrize("name", list(COMMANDS))
def test_extreme_regressor_magnitudes_are_harmless(data, name):
    base = COMMANDS[name](data)
    for factor in (1e-8, 1e8):
        scaled = COMMANDS[name](data.assign(x=data.x * factor))
        assert math.isclose(scaled.metrics.get("log_likelihood") or 0.0,
                            base.metrics.get("log_likelihood") or 0.0, rel_tol=1e-9)
        for term in ("Intercept", scaled.coefficients[-1].term):
            new = next(c for c in scaled.coefficients if c.term == term)
            old = next(c for c in base.coefficients if c.term == term)
            if term != "x":
                assert math.isclose(new.estimate, old.estimate, rel_tol=1e-5)
        slope = next(c for c in scaled.coefficients if c.term == "x")
        reference = next(c for c in base.coefficients if c.term == "x")
        assert math.isclose(slope.estimate * factor, reference.estimate, rel_tol=1e-5)
        assert all(math.isfinite(c.std_error) for c in scaled.coefficients)


def test_exact_fits_are_refused(data):
    exact = data.assign(y=2 * data.x + 1)
    assert code(oe.mixed, data=exact, y="y", x=["x"], group="g") == "perfect_fit"
    assert code(oe.xtgee, data=exact, y="y", x=["x"], panel="g") == "perfect_fit"


def test_collinear_random_effects_are_refused(data):
    """Regression: a constant random slope ended in an infinite interval."""
    assert code(oe.mixed, data=data.assign(k=1.0), y="y", x=["x"], group="g",
                random=["k"]) == "collinear_random_effects"
    assert code(oe.mixed, data=data.assign(k=2 * data.x), y="y", x=["x"], group="g",
                random=["x", "k"]) == "collinear_random_effects"
    assert code(oe.melogit, data=data.assign(k=3.0), y="b", x=["x"], group="g",
                random=["k"]) == "collinear_random_effects"
    # A random slope with a large level is not mistaken for the intercept by the screen.
    common.require_independent_effects(
        torch.tensor(np.column_stack([data.x + 1e7, np.ones(len(data))])), ["k", "_cons"],
        None, "mixed")


def test_flat_variance_interval_raises_instead_of_overflowing():
    class Term:
        term, estimate, std_error = "/var(_cons[g])", 1.0, 1e300

    class Bundle:
        coefficients = [Term()]

    with pytest.raises(AnalysisError) as error:
        common.variance_intervals(Bundle(), [0], 0.05)
    assert error.value.code == "boundary_solution"
    with pytest.raises(AnalysisError):
        common.log_intervals(Bundle(), [0], [1e300], 0.05)


def test_weights_contract(data):
    assert code(oe.mixed, data=data.assign(f=-data.f), y="y", x=["x"], group="g", weights="f",
                weight_type="fweight") == "negative_weights"
    assert code(oe.mixed, data=data.assign(f=data.f + 0.5), y="y", x=["x"], group="g",
                weights="f", weight_type="fweight") == "noninteger_frequency_weights"
    assert code(oe.mixed, data=data, y="y", x=["x"], group="g", weights="f",
                weight_type="aweight") == "invalid_spec"
    zero = oe.mixed(data=data.assign(f=np.where(data.g < 3, 0, data.f)), y="y", x=["x"],
                    group="g", weights="f", weight_type="fweight")
    assert any("zero weight" in w for w in zero.warnings)
    assert zero.metrics["n_groups"] == 27


def test_groups_singletons_and_clusters(data):
    singletons = data.assign(g=np.arange(len(data)))
    assert code(oe.mixed, data=singletons, y="y", x=["x"], group="g") \
        == "insufficient_group_size"
    assert code(oe.xtgee, data=singletons, y="y", x=["x"], panel="g") \
        == "insufficient_panel_length"
    assert code(oe.xtgee, data=singletons, y="y", x=["x"], panel="g", time="t",
                corr="ar1") == "insufficient_panel_length"
    assert code(oe.mixed, data=data.assign(cl=1), y="y", x=["x"], group="g",
                cluster="cl") == "insufficient_clusters"
    assert code(oe.mixed, data=data.assign(cl=data.index % 4), y="y", x=["x"], group="g",
                cluster="cl") == "cluster_not_nested"
    assert code(oe.xtpoisson, data=data.assign(cl=data.g // 3), y="c", x=["x"], panel="g",
                model="fe", cluster="cl") == "unsupported_covariance"
    labels = oe.mixed(data=data.assign(g="id" + data.g.astype(str)), y="y", x=["x"],
                      group="g")
    plain = oe.mixed(data=data, y="y", x=["x"], group="g")
    assert math.isclose(labels.metrics["log_likelihood"], plain.metrics["log_likelihood"],
                        rel_tol=1e-12)
    assert code(oe.mixed, data=data.assign(g=np.where(data.index == 2, np.nan, data.g)),
                y="y", x=["x"], group="g") == "missing_values"


def test_time_structure_errors(data):
    assert code(oe.xtgee, data=data.assign(t=0), y="y", x=["x"], panel="g",
                time="t") == "repeated_time_values"
    assert code(oe.xtgee, data=data.assign(t=data.t + 0.5), y="y", x=["x"], panel="g",
                time="t", corr="ar1") == "invalid_time"
    assert code(oe.xtgee, data=data, y="y", x=["x"], panel="g", time="t", corr="stationary",
                corr_order=9) == "insufficient_panel_length"
    gapped = data[~((data.t == 2) & (data.g % 2 == 0))]
    assert code(oe.xtgee, data=gapped, y="y", x=["x"], panel="g", time="t",
                corr="unstructured") == "unequal_spacing"
    dated = data.assign(t=pd.to_datetime("2000-01-01") + pd.to_timedelta(data.t, "D"))
    fit = oe.xtgee(data=dated, y="y", x=["x"], panel="g", time="t", corr="ar1")
    assert any("Datetime" in w for w in fit.warnings)


def test_option_bounds(data):
    assert code(oe.melogit, data=data, y="b", x=["x"], group="g", intpoints=0) == "invalid_spec"
    assert code(oe.melogit, data=data, y="b", x=["x"], group="g", intpoints=65) \
        == "invalid_spec"
    one = oe.melogit(data=data, y="b", x=["x"], group="g", intpoints=1)   # Laplace-like rule
    assert one.extra["integration"]["points"] == 1
    for scale in ("foo", True, -1.0, 0):
        assert code(oe.xtgee, data=data, y="y", x=["x"], panel="g", scale=scale) \
            == "invalid_spec"
    assert code(oe.xtgee, data=data, y="c", x=["x"], panel="g", family="poisson",
                nbk=2.0) == "invalid_spec"
    assert code(oe.xtgee, data=data, y="c", x=["x"], panel="g", family="poisson",
                link="logit") == "invalid_spec"
    assert code(oe.mepoisson, data=data.assign(e=data.e - 1), y="c", x=["x"], group="g",
                exposure="e") == "invalid_exposure"
    assert code(oe.mixed, data=data, y="y", x=["x"], group="g", alpha=1.5) == "invalid_spec"


def test_quadrature_memory_guard():
    n = 10_000
    x = torch.ones((n, 1), dtype=torch.float64)
    with pytest.raises(Exception) as error:
        RandomEffectsGLMM(x, torch.zeros(n, dtype=torch.float64),
                          torch.ones((n, 2), dtype=torch.float64), torch.zeros(n, dtype=torch.int64),
                          1, "logit", 64)
    assert getattr(error.value, "code", None) == "design_too_large"
    assert n * 64 ** 2 > NODE_LIMIT


def test_no_lr_comparison_under_robust(data):
    assert "lr_vs_pooled" not in oe.melogit(data=data, y="b", x=["x"], group="g",
                                            covariance="robust").tests
    assert "rho" not in oe.xtprobit(data=data, y="b", x=["x"], panel="g",
                                    covariance="robust").tests
    assert "alpha" not in oe.xtpoisson(data=data, y="c", x=["x"], panel="g",
                                       covariance="robust").tests
    assert "lr_vs_linear" not in oe.mixed(data=data, y="y", x=["x"], group="g",
                                          covariance="robust").tests
    assert code(oe.mixed, data=data, y="y", x=["x"], group="g", method="reml",
                covariance="robust") == "unsupported_covariance"


def test_predictions_edge_cases(data):
    fit = oe.mixed(data=data, y="y", x=["x"], group="g", random=["x"])
    # Regression: kind='xb' needed the outcome; BLUPs for a single group were refused.
    xb = oe.mixed_predict(fit, data.assign(y=np.nan), kind="xb")
    assert len(xb) == len(data)
    one = oe.mixed_predict(fit, data[data.g == 4], kind="reffects")
    full = oe.mixed_predict(fit, data, kind="reffects")
    expected = full[(full.group == 4)].set_index("effect").blup
    assert np.allclose(one.set_index("effect").blup.loc[expected.index], expected, rtol=1e-10)
    assert code(oe.mixed_predict, fit, data.assign(y=np.nan), kind="fitted") == "missing_values"
    assert code(oe.mixed_predict, fit, data, kind="residuals") == "invalid_option"
    gee = oe.xtgee(data=data, y="y", x=["x"], panel="g")
    assert code(oe.mixed_predict, gee, data) == "invalid_result"


def test_fixed_effects_edge_cases(data):
    singles = data.assign(g=np.where(data.index < 10, 1000 + data.index, data.g))
    fe = oe.xtpoisson(data=singles, y="c", x=["x"], panel="g", model="fe")
    assert all(math.isfinite(c.std_error) for c in fe.coefficients)
    assert code(oe.xtlogit, data=data, y="b", x=["x"], panel="g", model="fe",
                covariance="robust") == "unsupported_covariance"
    assert code(oe.xtprobit, data=data, y="b", x=["x"], panel="g", model="fe") == "invalid_spec"
    within = data.assign(z=data.g * 1.0)
    assert code(oe.xtlogit, data=within, y="b", x=["z"], panel="g", model="fe") \
        == "no_within_group_variation"
