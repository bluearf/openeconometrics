"""Tail/product/degenerate-covariance oracles independent of native kernels."""

import math

import numpy as np
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from test_econ_saved_prediction_hetprobit import configured, fitted as fitted


def at_row(frame, **values):
    result = frame.iloc[:1].drop(columns="y").copy()
    for name in ["x", "u", "z", "zcopy"]:
        result[name] = values.get(name, 0.0)
    result.g = "A"
    result.w = 1
    return result


def covariance_for(model, term, variance=1.0):
    v = np.zeros((len(model.coefficients), len(model.coefficients)))
    terms = [c.term for c in model.coefficients]
    v[terms.index(term), terms.index(term)] = variance
    return v


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("kind", ["response", "derivative"])
def test_normal_tail_standard_error_does_not_square_to_zero(fitted, sign, kind):
    model, frame = fitted
    altered = configured(model, {"x": 1.0})
    new = at_row(frame, x=sign * 35.0)
    density = math.exp(-35 * 35 / 2) / math.sqrt(2 * math.pi)
    terms = [c.term for c in model.coefficients]
    gradient = np.zeros(len(terms))
    if kind == "response":
        gradient[terms.index("Intercept")] = density
        gradient[terms.index("x")] = sign * 35 * density
        gradient[terms.index("lnsigma:x")] = -35 * 35 * density
    else:
        gradient[terms.index("Intercept")] = -sign * 35 * density
        gradient[terms.index("x")] = (1 - 35 * 35) * density
        gradient[terms.index("lnsigma:x")] = sign * 35 * (35 * 35 - 2) * density
    expected_se = math.hypot(*gradient)
    output = oe.predict(
        altered, new, kind=kind, term="x" if kind == "derivative" else None, interval="mean"
    )
    assert expected_se > 0
    np.testing.assert_allclose(output.std_error, expected_se, rtol=3e-13, atol=0)
    if kind == "derivative":
        np.testing.assert_allclose(output["dydx[x]"], density, rtol=3e-13, atol=0)


def test_density_underflow_is_restored_by_large_slope_and_variance_design(fitted):
    model, frame = fitted
    altered = configured(model, {"x": 1e200}, covariance_for(model, "lnsigma:z"))
    new = at_row(frame, x=-40 / 1e200, z=1.0)
    effect = math.exp(-800 - math.log(2 * math.pi) / 2 + math.log(1e200))
    gradient = effect * (40 * 40 - 1)
    output = oe.predict(altered, new, kind="derivative", term="x", interval="mean")
    np.testing.assert_allclose(output["dydx[x]"], effect, rtol=3e-13, atol=0)
    np.testing.assert_allclose(output.std_error, gradient, rtol=3e-13, atol=0)
    margins = oe.margins(altered, "x", data=new)
    j = [c.term for c in model.coefficients].index("lnsigma:z")
    np.testing.assert_allclose(margins.attrs["delta_gradients"][0][j], gradient, rtol=3e-13, atol=0)


def test_underflowed_density_times_huge_mean_design_preserves_probability_se(fitted):
    model, frame = fitted
    altered = configured(model, {"x": 1e-200}, covariance_for(model, "x"))
    new = at_row(frame, x=-40 / 1e-200)
    expected = math.exp(-800 - math.log(2 * math.pi) / 2 + math.log(40 / 1e-200))
    output = oe.predict(altered, new, interval="mean")
    np.testing.assert_allclose(output.std_error, expected, rtol=3e-13, atol=0)


def test_mixed_scales_preserve_small_uncertain_gradient_beside_huge_fixed_gradient(fitted):
    model, frame = fitted
    altered = configured(model, {"x": 1e200}, covariance_for(model, "x"))
    new = at_row(frame, x=-30 / 1e200, u=1e200)
    expected_effect = math.exp(-450 - math.log(2 * math.pi) / 2 + math.log(1e200))
    expected_gradient = -899 * math.exp(-450) / math.sqrt(2 * math.pi)
    output = oe.margins(altered, "x", data=new)
    i = [c.term for c in model.coefficients].index("x")
    np.testing.assert_allclose(output.estimate, expected_effect, rtol=3e-13, atol=0)
    np.testing.assert_allclose(
        output.attrs["delta_gradients"][0][i], expected_gradient, rtol=3e-13, atol=0
    )
    np.testing.assert_allclose(output.std_error, abs(expected_gradient), rtol=3e-13, atol=0)


@pytest.mark.parametrize("log_scale,beta", [(800.0, 1e200), (-800.0, 1e-200)])
def test_finite_response_derivative_without_representable_sigma(fitted, log_scale, beta):
    model, frame = fitted
    altered = configured(model, {"x": beta, "lnsigma:z": log_scale})
    new = at_row(frame, z=1.0)
    expected = math.exp(-math.log(2 * math.pi) / 2 - log_scale + math.log(beta))
    output = oe.predict(altered, new, kind="derivative", term="x")
    np.testing.assert_allclose(output["dydx[x]"], expected, rtol=3e-13, atol=0)
    assert oe.predict(altered, new).response.iloc[0] == 0.5


def test_saturated_categorical_change_keeps_small_positive_tail_contrast(fitted):
    model, frame = fitted
    altered = configured(model, {"Intercept": 35.0, "g[B]": 0.25})
    new = at_row(frame)
    from scipy import special

    expected = special.ndtr(-35.0) - special.ndtr(-35.25)
    output = oe.margins(altered, "g", data=new)
    assert expected > 0
    np.testing.assert_allclose(output.estimate.iloc[0], expected, rtol=5e-13, atol=0)


def test_roundoff_psd_covariance_still_rejects_material_negative_projection(fitted):
    model, frame = fitted
    v = np.eye(len(model.coefficients))
    terms = [c.term for c in model.coefficients]
    i, j = terms.index("Intercept"), terms.index("x")
    v[i, i] = v[j, j] = 1
    v[i, j] = v[j, i] = -1 - 6e-14
    altered = configured(model, {}, v)
    with pytest.raises(AnalysisError) as exc:
        oe.predict(altered, at_row(frame, x=1.0), interval="mean")
    assert exc.value.code == "invalid_inference"


@pytest.mark.parametrize("kind", ["response", "derivative", "ame", "mem"])
def test_underflowed_jacobian_times_large_covariance_sd_is_recovered(fitted, kind):
    model, frame = fitted
    altered = configured(model, {"x": 1.0}, covariance_for(model, "Intercept", 1e300))
    new = at_row(frame, x=-40.0)
    expected = math.exp(-800 - math.log(2 * math.pi) / 2 + 0.5 * math.log(1e300))
    if kind != "response":
        expected *= 40
    if kind in {"ame", "mem"}:
        output = oe.margins(altered, "x", data=new, method=kind)
    else:
        output = oe.predict(
            altered, new, kind=kind, term="x" if kind == "derivative" else None, interval="mean"
        )
    assert expected > 0
    np.testing.assert_allclose(output.std_error, expected, rtol=5e-13, atol=0)
    np.testing.assert_allclose(output.ci_high, 1.959963984540054 * expected, rtol=5e-13, atol=0)


@pytest.mark.parametrize("x", [1.0, 1e150, 1e-150])
@pytest.mark.parametrize("beta_factor", [1.0, 3.0])
def test_exact_overlapping_effect_cancellation_stays_zero_under_large_covariance(
    fitted, x, beta_factor
):
    model, frame = fitted
    altered = configured(
        model, {"x": beta_factor / x, "lnsigma:x": 1 / x}, covariance_for(model, "x", 1e300)
    )
    new = at_row(frame, x=x)
    output = oe.predict(altered, new, kind="derivative", term="x", interval="mean")
    assert output["dydx[x]"].iloc[0] == output.std_error.iloc[0] == 0
    for method in ["ame", "mem"]:
        result = oe.margins(altered, "x", data=new, method=method)
        i = [c.term for c in model.coefficients].index("x")
        assert result.estimate.iloc[0] == result.std_error.iloc[0] == 0
        assert result.attrs["delta_gradients"][0][i] == 0


def test_cpu_execution_scope_preserves_callers_meta_context(fitted):
    model, frame = fitted
    new = at_row(frame, x=0.5, z=0.2)
    expected_p = oe.predict(model, new, interval="mean")
    expected_m = oe.margins(model, ["x", "z", "g"], data=new)
    with torch.device("meta"):
        output = oe.predict(model, new, interval="mean")
        margins = oe.margins(model, ["x", "z", "g"], data=new)
        assert torch.empty(0).device.type == "meta"
        with pytest.raises(AnalysisError):
            oe.predict(model, new, outcome=1)
        assert torch.empty(0).device.type == "meta"
    import pandas as pd

    pd.testing.assert_frame_equal(output, expected_p)
    pd.testing.assert_frame_equal(margins, expected_m)


@pytest.mark.parametrize("kind", ["xb", "stdp", "sigma"])
def test_output_specific_indexes_do_not_require_unrelated_equation_products(fitted, kind):
    model, frame = fitted
    if kind == "sigma":
        altered = configured(model, {"x": 1e308, "lnsigma:z": 0.2})
        new = at_row(frame, x=2.0, z=1.0)
        expected = math.exp(0.2)
    else:
        altered = configured(model, {"x": 0.2, "lnsigma:z": 1e308})
        new = at_row(frame, x=1.0, z=2.0)
        expected = 0.2 if kind == "xb" else math.sqrt(2)
    result = oe.predict(altered, new, kind=kind, interval="mean" if kind != "stdp" else None)
    np.testing.assert_allclose(result[kind], expected, rtol=5e-14, atol=0)


def test_covariance_scaled_gradient_underflow_is_explicitly_refused(fitted):
    model, frame = fitted
    altered = configured(model, {"x": 1.0})
    with pytest.raises(AnalysisError) as exc:
        oe.predict(altered, at_row(frame, x=-40.0), interval="mean")
    assert exc.value.code == "prediction_precision"
    assert "covariance-scaled" in str(exc.value)


@pytest.mark.parametrize(
    "m,g,log_sigma", [(1e-200, 1e-200, -math.log(1e200)), (1e200, 1e200, math.log(1e200))]
)
def test_uncertifiable_factored_products_refuse_finite_rescaled_effects(fitted, m, g, log_sigma):
    model, frame = fitted
    altered = configured(model, {"Intercept": m, "lnsigma:x": g, "lnsigma:z": log_sigma})
    # The standardized index is about one and the mathematical effect is
    # finite. Our declared precision domain refuses unrepresentable m*g,
    # rather than fabricating an endpoint or zero-effect result.
    with pytest.raises(AnalysisError) as exc:
        oe.predict(altered, at_row(frame, z=1.0), kind="derivative", term="x")
    assert exc.value.code == "prediction_precision"
