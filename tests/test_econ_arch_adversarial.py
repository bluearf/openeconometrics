"""Adversarial inputs for ``oe.arch`` and ``oe.arch_forecast``.

Every case must either produce a complete, finite result or raise ``AnalysisError`` with
a code and a message that says what to change: never a raw exception, never NaN or inf
in a result. The file also checks the rules of the family package itself (torch-free
manifest, no forbidden imports, no autograd).
"""

import math
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from test_econ_arch_verify import draw, errors, reported

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.models import ModelSpec

PACKAGE = pathlib.Path(oe.__file__).parent / "econometrics" / "arch"


@pytest.fixture(scope="module")
def base():
    return draw(500, 1)


def outcome(**keywords):
    """``("ok", result)`` for a complete finite fit or ``("error", code)``; nothing else."""
    try:
        result = oe.arch(**keywords)
    except AnalysisError as error:
        assert error.code and error.code == error.code.lower() and " " not in error.code
        assert len(str(error)) > 20                      # a sentence, not a bare code
        return "error", error.code
    values = np.r_[reported(result), errors(result), np.array(result.covariance_matrix).ravel()]
    assert np.isfinite(values).all() and (errors(result) > 0).all()
    assert math.isfinite(result.metrics["log_likelihood"])
    assert all(math.isfinite(c.p_value) and 0 <= c.p_value <= 1 for c in result.coefficients)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    return "ok", result


def code(**keywords):
    kind, value = outcome(**keywords)
    assert kind == "error", "expected an AnalysisError"
    return value


# ---- sample size ------------------------------------------------------------------------------


def test_too_few_observations(base):
    for rows in (1, 2, 5, 9):
        with pytest.raises(AnalysisError) as error:
            oe.arch(data=base.iloc[:rows], y="y")
        assert error.value.code == "insufficient_observations"
        assert "needs at least 10 observations" in str(error.value)
    assert code(data=base.iloc[:0], y="y") == "empty_data"
    # More parameters or longer lags need more observations.
    assert code(data=base.iloc[:50], y="y", ar=[45]) == "insufficient_observations"
    # 8 parameters + the longest lag + 5: the GJR-t regression needs 14 observations.
    assert code(data=base.iloc[:13], y="y", x=["x1", "x2"], model="gjr", dist="t") \
        == "insufficient_observations"


@pytest.mark.parametrize("rows", [10, 12, 16, 25])
def test_tiny_samples_fit_or_explain(base, rows):
    kind, value = outcome(data=base.iloc[:rows], y="y")
    if kind == "error":
        assert value in {"nonconvergence", "singular_information"}
    else:
        assert value.nobs == rows


def test_all_missing_columns(base):
    assert code(data=base.assign(x1=np.nan), y="y", x=["x1"], missing="drop") == "empty_sample"
    assert code(data=base.assign(y=np.nan), y="y", missing="drop") == "empty_sample"
    assert code(data=base.assign(x1=np.nan), y="y", x=["x1"]) == "missing_values"
    assert code(data=base.assign(zv=np.nan), y="y", variance_x=["zv"], missing="drop") \
        == "empty_sample"


# ---- degenerate outcomes and designs ----------------------------------------------------------


def test_degenerate_outcomes(base):
    assert code(data=base.assign(y=3.0), y="y") == "constant_outcome"
    assert code(data=base.assign(y=0.0), y="y", constant=False) == "constant_outcome"
    assert code(data=base.assign(y=lambda d: 2 - 3 * d["x1"]), y="y", x=["x1"]) == "perfect_fit"
    # Squared residuals without any variation leave the variance equation unidentified.
    alternating = base.assign(y=np.where(np.arange(500) % 2 == 0, 1.0, -1.0))
    assert code(data=alternating, y="y") in {"nonconvergence", "singular_information"}
    # One enormous outlier: an explained failure or a finite fit.
    kind, value = outcome(data=base.assign(y=np.where(np.arange(500) == 250, 1e6, base["y"])),
                          y="y")
    assert kind == "ok" or value == "nonconvergence"


def test_series_without_arch_effects(base):
    noise = pd.DataFrame({"y": np.random.default_rng(0).normal(size=600)})
    for keywords in (dict(), dict(model="egarch"), dict(model="gjr"), dict(model="arch", arch=3)):
        kind, value = outcome(data=noise, y="y", **keywords)
        assert kind == "ok" or value in {"nonconvergence", "singular_information"}


def test_integer_boolean_and_text_columns(base):
    rng = np.random.default_rng(3)
    kind, result = outcome(data=base.assign(y=rng.integers(-5, 6, 500)), y="y")
    assert kind == "ok" and result.nobs == 500
    kind, _ = outcome(data=base.assign(y=rng.integers(0, 2, 500).astype(bool)), y="y")
    assert kind == "ok"
    assert code(data=base.assign(y=base["y"].astype(str)), y="y") == "non_numeric_column"
    assert code(data=base.assign(x1="a"), y="y", x=["x1"]) == "non_numeric_column"
    assert code(data=base.assign(zv="a"), y="y", variance_x=["zv"]) == "non_numeric_column"
    infinite = base.assign(y=np.where(np.arange(500) == 3, np.inf, base["y"]))
    assert code(data=infinite, y="y") == "non_finite_values"


def test_collinear_and_constant_regressors(base):
    reference = oe.arch(data=base, y="y", x=["x2"])
    kind, result = outcome(data=base.assign(x1=1.0), y="y", x=["x1", "x2"])
    assert kind == "ok" and result.provenance["omitted_terms"] == ["x1"]
    np.testing.assert_allclose(reported(result), reported(reference), rtol=1e-9)
    kind, result = outcome(data=base.assign(x1=2 * base["x2"] + 1), y="y", x=["x2", "x1"])
    assert kind == "ok" and result.provenance["omitted_terms"] == ["x1"]
    np.testing.assert_allclose(reported(result), reported(reference), rtol=1e-9)
    kind, result = outcome(data=base.assign(zc=2.0), y="y", x=["x2"], variance_x=["zc"])
    assert kind == "ok" and result.provenance["omitted_terms"] == ["HET:zc"]
    # Without a variance regressor left, the model is the plain GARCH with its constant.
    assert [c.term for c in result.coefficients][-1] == "ARCH:Intercept"
    np.testing.assert_allclose(reported(result), reported(reference), rtol=1e-9)
    assert any("collinearity" in warning for warning in result.warnings)
    assert code(data=base, y="y", x=["x1", "x1"]) == "invalid_spec"
    assert code(data=base, y="y", x=["y"]) == "invalid_spec"
    assert code(data=base, y="y", variance_x=["y"]) == "invalid_spec"


def test_categorical_edge_cases(base):
    assert code(data=base.assign(c="a"), y="y", x=["c"], categorical=["c"]) \
        == "constant_predictor"
    assert code(data=base.assign(c="a"), y="y", x=["x1"], categorical=["c"]) == "invalid_spec"
    # One category per observation: more parameters than the sample supports.
    assert code(data=base.assign(c=np.arange(500)), y="y", x=["c"], categorical=["c"]) \
        == "insufficient_observations"
    # The same regressor may enter both equations.
    kind, result = outcome(data=base, y="y", x=["x1"], variance_x=["x1"])
    assert kind == "ok" and "HET:x1" in [c.term for c in result.coefficients]


# ---- magnitudes ---------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1e-8, 1e8, 1e-50, 1e50])
def test_extreme_units_of_the_outcome(base, factor):
    reference = oe.arch(data=base, y="y", x=["x1"])
    kind, result = outcome(data=base.assign(y=base["y"] * factor), y="y", x=["x1"])
    assert kind == "ok"
    scale = np.array([factor, factor, 1.0, 1.0, factor ** 2])
    np.testing.assert_allclose(reported(result), reported(reference) * scale, rtol=1e-5)
    np.testing.assert_allclose(errors(result), errors(reference) * scale, rtol=1e-4)
    assert result.metrics["log_likelihood"] == pytest.approx(
        reference.metrics["log_likelihood"] - 500 * math.log(factor), rel=1e-9)


@pytest.mark.parametrize("factor", [1e-150, 1e-70, 1e70, 1e150])
def test_units_beyond_float64_ask_for_rescaling(base, factor):
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=base.assign(y=base["y"] * factor), y="y", x=["x1"])
    assert error.value.code == "numerical_failure" and "rescale the outcome" in str(error.value)


@pytest.mark.parametrize("factor,offset", [(1e8, 0.0), (1e-8, 0.0), (1.0, 1e8), (1e3, -1e10)])
def test_extreme_units_of_a_regressor(base, factor, offset):
    reference = oe.arch(data=base, y="y", x=["x1"], covariance="nonrobust")
    kind, result = outcome(data=base.assign(x1=base["x1"] * factor + offset), y="y", x=["x1"],
                           covariance="nonrobust")
    assert kind == "ok"
    assert result.coefficients[1].estimate == pytest.approx(
        reference.coefficients[1].estimate / factor, rel=1e-5)
    assert result.coefficients[1].std_error == pytest.approx(
        reference.coefficients[1].std_error / factor, rel=1e-4)
    np.testing.assert_allclose(reported(result)[2:], reported(reference)[2:], rtol=1e-4)
    np.testing.assert_allclose(errors(result)[2:], errors(reference)[2:], rtol=1e-4)


# ---- time ---------------------------------------------------------------------------------------


def test_time_column_problems(base):
    assert code(data=base.assign(t=base["t"].astype(str)), y="y", time="t") == "invalid_time"
    assert code(data=base.assign(t=base["t"] % 2 == 0), y="y", time="t") == "invalid_time"
    assert code(data=base.assign(t=base["t"] + 0.5), y="y", time="t") == "invalid_time"
    assert code(data=base.assign(t=base["t"] * 2), y="y", time="t") == "time_gaps"
    assert code(data=base.drop(index=[250]), y="y", time="t") == "time_gaps"
    assert code(data=base.assign(t=base["t"].where(base["t"] != 9, 8)), y="y", time="t") \
        == "repeated_time_values"
    assert code(data=base.assign(t=pd.Timestamp("2020-01-01")), y="y", time="t") \
        == "repeated_time_values"
    assert code(data=base, y="y", time="y") == "invalid_spec"
    assert code(data=base, y="y", time="t", variance_x=["t"]) == "invalid_spec"
    # A missing period inside the series is a gap; at the edge it only shortens the sample.
    hole = base.assign(t=base["t"].where(base["t"] != 7))
    assert code(data=hole, y="y", time="t") == "missing_values"
    assert code(data=hole, y="y", time="t", missing="drop") == "time_gaps"
    edge = base.assign(t=base["t"].where(base["t"] != 1))
    kind, result = outcome(data=edge, y="y", time="t", missing="drop")
    assert kind == "ok" and result.nobs == 499 and result.sample_positions[0] == 1
    interior = base.assign(y=base["y"].where(base["t"] != 40))
    assert code(data=interior, y="y", missing="drop") == "time_gaps"
    # Negative and float-valued integer periods are fine.
    reference = oe.arch(data=base, y="y", time="t")
    for column in (base["t"] - 1000, base["t"].astype(float)):
        kind, result = outcome(data=base.assign(t=column), y="y", time="t")
        assert kind == "ok"
        np.testing.assert_allclose(reported(result), reported(reference), rtol=1e-12)


# ---- options at and beyond their bounds ---------------------------------------------------------


def test_option_bounds(base):
    common = dict(data=base, y="y", x=["x1"])
    for tolerance in (float("inf"), float("nan"), "a"):
        assert code(**common, tolerance=tolerance) == "invalid_spec"
    for tolerance in (0, 0.0, -1e-8):
        assert code(**common, tolerance=tolerance) == "invalid_option"
    assert code(**common, alpha=0.0) == "invalid_spec"
    assert code(**common, alpha=1.0) == "invalid_spec"
    assert code(**common, max_iterations=0) == "invalid_spec"
    assert code(**common, max_iterations=2.5) == "invalid_spec"
    assert code(**common, test_lags=0) == "invalid_spec"
    assert code(**common, test_lags=2.5) == "invalid_spec"
    assert code(**common, covariance="HC1") == "invalid_spec"
    assert code(**common, missing="maybe") == "invalid_spec"
    for lags in (True, -1, 1.0, "1", [0], [1, 1], [1.5], 61, [61], {"a": 1}):
        assert code(**common, arch=lags) == "invalid_lags", lags
    assert code(**common, arch=0) == "invalid_spec"
    assert code(**common, arch=[]) == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.arch(**common, max_iterations=1)
    assert error.value.code == "nonconvergence" and "max_iterations" in str(error.value)
    # Loose and very tight tolerances both end at the same maximum (the convergence test
    # is the scaled gradient, not the gradient tolerance alone).
    loose = oe.arch(**common, tolerance=1e300)
    tight = oe.arch(**common, tolerance=1e-30)
    np.testing.assert_allclose(reported(loose), reported(tight), rtol=1e-6)
    # Diagnostics lags beyond the sample are capped or left out, never an error.
    kind, result = outcome(**common, test_lags=1000)
    assert kind == "ok" and result.tests["ljung_box"]["lags"] == 499
    assert "arch_lm_residuals" not in result.tests


def test_lag_arguments_of_other_types(base):
    reference = reported(oe.arch(data=base, y="y", arch=2))
    for lags in (np.int64(2), (1, 2), np.array([2, 1]), range(1, 3)):
        np.testing.assert_allclose(reported(oe.arch(data=base, y="y", arch=lags)), reference,
                                   rtol=1e-12)
    default = reported(oe.arch(data=base, y="y"))
    np.testing.assert_allclose(reported(oe.arch(data=base, y="y", arch=None, garch=None)),
                               default, rtol=1e-12)
    for model in ("egarch", "parch"):                       # no GARCH terms
        kind, result = outcome(data=base, y="y", model=model, garch=0)
        assert kind == "ok" and not any("garch" in c.term for c in result.coefficients)


def test_other_data_containers_and_bad_arguments(base):
    reference = reported(oe.arch(data=base, y="y"))
    for data in ({"y": base["y"].tolist()}, base[["y"]].to_dict("records")):
        np.testing.assert_allclose(reported(oe.arch(data=data, y="y")), reference, rtol=1e-12)
    assert code(data=None, y="y") == "invalid_data"
    assert code(data=base, y="nope") == "missing_columns"
    assert code(data=base, y=None) == "invalid_spec"
    assert code(data=base, y="y", x="x1") == "invalid_spec"
    assert code(data=base, y="y", constant=1) == "invalid_spec"
    # Weights and clusters are not part of the estimator's contract.
    for extra in (dict(weights="x1", weight_type="fweight"), dict(cluster="x1"),
                  dict(covariance="cluster", cluster="x1"), dict(panel="x1")):
        with pytest.raises(ValidationError):
            ModelSpec(estimator="arch", outcome="y", **extra)


# ---- forecasts ---------------------------------------------------------------------------------


def forecast_code(*arguments, **keywords):
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(*arguments, **keywords)
    assert len(str(error.value)) > 20
    return error.value.code


def test_forecast_arguments(base):
    fit = oe.arch(data=base, y="y", x=["x1"], time="t")
    future = {"x1": [0.1, 0.2]}
    for steps in (0, -1, 1.5, "3", True, None, 10001):
        assert forecast_code(fit, steps, exog=future) == "invalid_steps"
    assert forecast_code(fit, 2) == "missing_exog"
    assert forecast_code(fit, 3, exog=future) == "invalid_exog"
    assert forecast_code(fit, 2, exog={"x2": [0.1, 0.2]}) == "missing_columns"
    assert forecast_code(fit, 2, exog={"x1": [0.1, np.nan]}) == "missing_values"
    assert forecast_code(fit, 2, exog={"x1": ["a", "b"]}) == "non_numeric_column"
    assert forecast_code(fit, 2, exog={"x1": [np.inf, 1.0]}) == "non_finite_values"
    assert forecast_code(fit, 2, exog=5) == "invalid_data"
    for alpha in (0, 1, "a", True, None):
        assert forecast_code(fit, 2, exog=future, alpha=alpha) == "invalid_option"
    assert forecast_code(None, 2) == "invalid_result"
    assert forecast_code(fit, 2, exog=future, data=base.iloc[:1]) == "insufficient_observations"
    assert forecast_code(fit, 2, exog=future, data=base[["y"]]) == "missing_columns"
    assert forecast_code(fit, 2, exog=future, data=base.drop(index=[100])) == "time_gaps"
    assert forecast_code(oe.arch(data=base, y="y"), 2, exog=future) == "invalid_exog"
    table = oe.arch_forecast(fit, np.int64(2), exog=pd.DataFrame(future))
    assert list(table["period"]) == [501, 502] and np.isfinite(table.to_numpy(float)).all()
    # A result restored from JSON forecasts identically.
    restored = type(fit).model_validate_json(fit.model_dump_json())
    np.testing.assert_allclose(oe.forecast(restored, 2, exog=future).to_numpy(float),
                               table.to_numpy(float), rtol=1e-14)


def test_forecast_of_a_variance_equation_that_leaves_its_domain(base):
    small = oe.arch(data=base.iloc[:12], y="y")              # negative variance constant
    assert any("negative coefficients" in warning for warning in small.warnings)
    assert forecast_code(small, 50) == "non_finite_result"
    # Long horizons of well-behaved models stay finite.
    for model in ("igarch", "egarch", "garch"):
        table = oe.arch_forecast(oe.arch(data=base, y="y", model=model), 10000)
        assert len(table) == 10000 and np.isfinite(table.to_numpy(float)).all()
        assert (table["variance_forecast"] > 0).all()


# ---- rules of the package -----------------------------------------------------------------------


def test_manifest_imports_neither_torch_nor_pandas():
    script = ("import sys; import openecon.econometrics.arch as m; "
              "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
              "assert m.ESTIMATORS[0].name == 'arch' and 'arch_forecast' in m.EXPORTS "
              "and m.FORECAST['arch'].endswith(':arch_forecast')")
    subprocess.run([sys.executable, "-c", script], check=True)


def test_package_source_follows_the_numerical_rules():
    forbidden = ("import numpy", "from numpy", "import scipy", "from scipy", "statsmodels",
                 "linearmodels", "sklearn", "requires_grad", "torch.autograd", "torch.func",
                 ".backward(", "torch.linalg.inv(", ".inverse()")
    for path in sorted(PACKAGE.glob("*.py")):
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{path.name}: {needle}"
        assert max(len(line) for line in text.splitlines()) <= 100, path.name
    manifest = (PACKAGE / "__init__.py").read_text()
    assert "import torch" not in manifest and "import pandas" not in manifest


def test_every_declared_option_and_covariance_is_handled(base):
    info = registry.get("arch")
    assert info.weights == () and info.panel == "none" and info.time == "optional"
    assert {option.name for option in info.options} == {
        "model", "dist", "arch", "garch", "archm", "ar", "ma", "test_lags", "max_iterations",
        "tolerance"}
    for covariance in info.covariances:
        kind, result = outcome(data=base, y="y", x=["x1"], covariance=covariance)
        assert kind == "ok" and result.inference["covariance"] == covariance
    for model in info.option("model").choices:
        kind, value = outcome(data=base, y="y", model=model)
        assert kind == "ok" or value == "nonconvergence", model
    for archm in info.option("archm").choices:
        kind, value = outcome(data=base.iloc[:300], y="y", archm=archm)
        assert kind == "ok" or value == "nonconvergence", archm
    assert callable(oe.arch) and callable(oe.arch_forecast) and callable(oe.forecast)
    for function in (oe.arch, oe.arch_forecast):
        for word in ("Parameters", "Stata", "Example"):
            assert word in function.__doc__


def test_long_lags_of_the_scan_engine_are_refused_before_allocating():
    # EGARCH and ARCH-in-mean carry one transition map per observation in their derivative
    # recursion: 8 n (lags)^2 bytes. Beyond 1 GiB the fit is refused with an explanation
    # instead of exhausting the memory.
    frame = pd.DataFrame({"y": np.random.default_rng(0).normal(size=40_000)})
    for keywords in (dict(model="egarch", arch=40), dict(archm="sd", ar=[60])):
        with pytest.raises(AnalysisError) as error:
            oe.arch(data=frame, y="y", **keywords)
        assert error.value.code == "model_too_large" and "shorter lags" in str(error.value)
    kind, result = outcome(data=frame.iloc[:4000], y="y", model="garch", arch=8)
    assert kind == "ok" or result == "nonconvergence"          # the filter engine has no limit
