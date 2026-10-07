"""Saved category identity and tail inference, with independent closed-form oracles.

Float64 can round a dominant probability to one while its derivatives remain
representable. The oracle never obtains density from ``p * (1 - p)`` or takes
the square root of an underflowed quadratic form.
"""

from __future__ import annotations

import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


@pytest.fixture(scope="module", params=["ologit", "oprobit"])
def ordered_saved(request):
    rng = np.random.default_rng(53918)
    data = pd.DataFrame({"x": rng.normal(size=300), "y": rng.integers(0, 3, size=300)})
    result = getattr(oe, request.param)(data=data, y="y", x=["x"], covariance="robust",
                                      missing="drop")
    return ResultBundle.model_validate_json(result.model_dump_json())


@pytest.fixture(scope="module")
def multinomial_saved():
    rng = np.random.default_rng(61412)
    data = pd.DataFrame({"x": rng.normal(size=360), "y": rng.integers(0, 3, size=360)})
    result = oe.mlogit(data=data, y="y", x=["x"], base=1, covariance="robust", missing="drop")
    return ResultBundle.model_validate_json(result.model_dump_json())


def ordered_state(saved, slope=1.0, cuts=(0.0, 2.0), covariance=None):
    result = saved.model_copy(deep=True)
    estimates = {"x": slope, "/cut1": cuts[0], "/cut2": cuts[1]}
    for coefficient in result.coefficients:
        coefficient.estimate = estimates[coefficient.term]
    result.extra["cutpoints"] = list(cuts)
    matrix = np.eye(3) if covariance is None else np.asarray(covariance)
    terms = [c.term for c in result.coefficients]
    order = [["x", "/cut1", "/cut2"].index(t) for t in terms]
    result.covariance_matrix = matrix[np.ix_(order, order)].tolist()
    return result


def density(logistic, edge):
    if logistic:
        q = math.exp(-abs(edge))
        log_value = -abs(edge) - 2 * math.log1p(q)
        ratio = (q - 1) / (1 + q) if edge >= 0 else (1 - q) / (1 + q)
    else:
        log_value = -edge * edge / 2 - math.log(2 * math.pi) / 2
        ratio = -edge
    return log_value, math.exp(log_value), ratio


def ordered_oracle(result, x):
    slope = next(c.estimate for c in result.coefficients if c.term == "x")
    cuts = result.extra["cutpoints"]
    values = [density(result.spec.estimator == "ologit", cut - slope * x) for cut in cuts]
    (_, f1, ratio1), (_, f2, ratio2) = values
    # Signed slope-density products are formed before rounding a tiny density.
    sf1 = math.copysign(math.exp(math.log(abs(slope)) + values[0][0]), slope)
    sf2 = math.copysign(math.exp(math.log(abs(slope)) + values[1][0]), slope)
    effects = np.array([-sf1, sf1 - sf2, sf2])
    probability_gradient = np.array([
        [-x * f1, f1, 0],
        [x * (f1 - f2), -f1, f2],
        [x * f2, 0, -f2],
    ])
    first = [-f1 + slope * x * f1 * ratio1, -sf1 * ratio1, 0]
    last = [f2 - slope * x * f2 * ratio2, 0, sf2 * ratio2]
    effect_gradient = np.array([first, -np.array(first) - np.array(last), last])
    terms = [c.term for c in result.coefficients]
    order = [["x", "/cut1", "/cut2"].index(t) for t in terms]
    return effects, probability_gradient[:, order], effect_gradient[:, order]


def stable_standard_errors(gradient, covariance):
    # Covariance is positive definite in these numerical fixtures. Cholesky
    # gives a separate factor oracle; hypot rescales before squaring tails.
    transformed = np.asarray(gradient) @ np.linalg.cholesky(np.asarray(covariance))
    return np.array([math.hypot(*row) for row in transformed])


@pytest.mark.parametrize("x", [-50.0, 50.0])
def test_saturated_ordered_logistic_effects_and_full_cutpoint_covariance(ordered_saved, x):
    if ordered_saved.spec.estimator != "ologit":
        pytest.skip("Logistic sigmoid saturation is the regression being tested.")
    covariance = [[0.3, 0.02, -0.01], [0.02, 0.4, 0.05], [-0.01, 0.05, 0.6]]
    result = ordered_state(ordered_saved, covariance=covariance)
    data = pd.DataFrame({"x": [x]})
    effect, jac_probability, jac_effect = ordered_oracle(result, x)
    probabilities = oe.predict(result, data, interval="mean")
    derivatives = oe.predict(result, data, kind="derivative", term="x", interval="mean")
    margins = oe.margins(result, "x", data=data)
    for i, label in enumerate(result.extra["categories"]):
        suffix = json.dumps(label)
        assert_allclose(probabilities[f"std_error[{suffix}]"],
                        stable_standard_errors(jac_probability, result.covariance_matrix)[i],
                        rtol=3e-13, atol=0)
        assert_allclose(derivatives[f"dydx[x][{suffix}]"], effect[i], rtol=3e-13, atol=0)
        assert_allclose(derivatives[f"std_error[{suffix}]"],
                        stable_standard_errors(jac_effect, result.covariance_matrix)[i],
                        rtol=3e-13, atol=0)
    assert_allclose(margins.estimate, effect, rtol=3e-13, atol=0)
    assert_allclose(margins.std_error,
                    stable_standard_errors(jac_effect, result.covariance_matrix), rtol=3e-13, atol=0)
    assert_allclose(margins.attrs["delta_gradients"], jac_effect, rtol=3e-13, atol=0)
    assert math.isclose(math.fsum(margins.estimate), 0.0, abs_tol=1e-35)


def test_underflowed_variance_does_not_erase_representable_category_standard_errors(ordered_saved):
    result = ordered_state(ordered_saved)
    edge = 500.0 if result.spec.estimator == "ologit" else 35.0
    data = pd.DataFrame({"x": [-edge]})
    effect, jac_probability, jac_effect = ordered_oracle(result, -edge)
    outcome = result.extra["categories"][0]
    probability = oe.predict(result, data, outcome=outcome, interval="mean")
    derivative = oe.predict(result, data, kind="derivative", term="x", outcome=outcome,
                            interval="mean")
    margin = oe.margins(result, "x", data=data, outcome=outcome)
    probability_se = stable_standard_errors(jac_probability, result.covariance_matrix)[0]
    effect_se = stable_standard_errors(jac_effect, result.covariance_matrix)[0]
    assert probability_se > 0 and effect_se > 0
    assert_allclose(probability.std_error, probability_se, rtol=4e-13, atol=0)
    assert_allclose(derivative["dydx[x]"], effect[0], rtol=4e-13, atol=0)
    assert_allclose(derivative.std_error, effect_se, rtol=4e-13, atol=0)
    assert_allclose(margin.std_error, effect_se, rtol=4e-13, atol=0)


def test_density_underflow_before_large_slope_product_cannot_silently_zero_effect(ordered_saved):
    result = ordered_state(ordered_saved, slope=1e200)
    edge = 800.0 if result.spec.estimator == "ologit" else 40.0
    data = pd.DataFrame({"x": [-edge / 1e200]})
    effect, _, jac_effect = ordered_oracle(result, float(data.x.iloc[0]))
    selected = result.extra["categories"][0]
    derivative = oe.predict(result, data, kind="derivative", term="x", outcome=selected,
                            interval="mean")
    margin = oe.margins(result, "x", data=data, outcome=selected)
    expected_se = stable_standard_errors(jac_effect, result.covariance_matrix)[0]
    assert effect[0] != 0 and expected_se > 0
    assert_allclose(derivative["dydx[x]"], effect[0], rtol=5e-13, atol=0)
    assert_allclose(margin.estimate, effect[0], rtol=5e-13, atol=0)
    assert_allclose(derivative.std_error, expected_se, rtol=5e-13, atol=0)
    assert_allclose(margin.std_error, expected_se, rtol=5e-13, atol=0)


def test_density_underflow_before_large_design_product_preserves_probability_se(ordered_saved):
    result = ordered_state(ordered_saved, slope=1e-200)
    edge = 800.0 if result.spec.estimator == "ologit" else 40.0
    x = -edge / 1e-200
    actual_edge = -1e-200 * x
    log_density, raw_density, _ = density(result.spec.estimator == "ologit", actual_edge)
    assert raw_density == 0.0
    # Probability itself rounds to one. Its slope derivative is still finite:
    # dP(first)/d beta = -x*f(cut-beta*x), evaluated before density underflows.
    slope_gradient = math.exp(math.log(abs(x)) + log_density)
    expected_se = math.hypot(slope_gradient, raw_density)
    assert expected_se > 0
    prediction = oe.predict(result, pd.DataFrame({"x": [x]}), outcome=0, interval="mean")
    assert_allclose(prediction.std_error, expected_se, rtol=5e-13, atol=0)


def test_mixed_predictor_scales_preserve_each_effect_gradient_and_uncertain_coordinate(ordered_saved):
    rng = np.random.default_rng(57192)
    data = pd.DataFrame({"x": rng.normal(size=300), "z": rng.normal(size=300),
                         "y": rng.integers(0, 3, size=300)})
    fitted = getattr(oe, ordered_saved.spec.estimator)(data=data, y="y", x=["x", "z"],
                                                     covariance="robust")
    result = ResultBundle.model_validate_json(fitted.model_dump_json())
    estimates = {"x": 1e200, "z": 0.0, "/cut1": 0.0, "/cut2": 2.0}
    for coefficient in result.coefficients:
        coefficient.estimate = estimates[coefficient.term]
    result.extra["cutpoints"] = [0.0, 2.0]
    terms = [c.term for c in result.coefficients]
    position = terms.index("x")
    covariance = np.zeros((4, 4))
    covariance[position, position] = 1.0
    result.covariance_matrix = covariance.tolist()
    edge = 460.0 if result.spec.estimator == "ologit" else 30.0
    x = -edge / 1e200
    actual_edge = -1e200 * x
    _, f, ratio = density(result.spec.estimator == "ologit", actual_edge)
    expected_gradient = f * (-1 - actual_edge * ratio)
    assert expected_gradient > 0
    # Every coordinate is representable, although the predictor scales span
    # more than float64's exponent range. The huge z derivative has no variance
    # in this PSD covariance and must not suppress uncertainty in beta_x.
    new = pd.DataFrame({"x": [x], "z": [1e200]})
    margin = oe.margins(result, "x", data=new, outcome=0)
    prediction = oe.predict(result, new, kind="derivative", term="x", outcome=0,
                            interval="mean")
    assert_allclose(margin.attrs["delta_gradients"][0][position], expected_gradient,
                    rtol=5e-13, atol=0)
    assert_allclose(margin.std_error, expected_gradient, rtol=5e-13, atol=0)
    assert_allclose(prediction.std_error, expected_gradient, rtol=5e-13, atol=0)


@pytest.mark.parametrize("edge,slope", [(50.0, 1.0), (800.0, 1e200)])
def test_multinomial_saturated_dominant_effect_and_cross_equation_covariance(multinomial_saved, edge, slope):
    result = multinomial_saved.model_copy(deep=True)
    for coefficient in result.coefficients:
        coefficient.estimate = {"0:Intercept": edge, "0:x": slope}.get(coefficient.term, 0.0)
    terms = [c.term for c in result.coefficients]
    names = ["0:Intercept", "0:x", "2:Intercept", "2:x"]
    factor = np.array([[1, 0, 0, 0], [.2, 1, 0, 0], [-.1, .3, 1, 0], [.1, -.2, .4, 1]])
    covariance = factor @ factor.T
    order = [names.index(t) for t in terms]
    result.covariance_matrix = covariance[np.ix_(order, order)].tolist()
    q = math.exp(-edge)
    denominator = 1 + 2 * q
    pair = q / denominator**2
    scaled_pair = math.exp(math.log(slope) - edge) / denominator**2
    effect = 2 * scaled_pair
    jac_probability = np.array([2 * pair, 0, -pair, 0])[order]
    jac_effect = np.array([effect * (1 - 2 / denominator), 2 * pair,
                           scaled_pair * (1 - 4 * q / denominator), -pair])[order]
    probability_se = stable_standard_errors([jac_probability], result.covariance_matrix)[0]
    effect_se = stable_standard_errors([jac_effect], result.covariance_matrix)[0]
    data = pd.DataFrame({"x": [0.0]})
    probability = oe.predict(result, data, outcome=0, interval="mean")
    derivative = oe.predict(result, data, outcome=0, kind="derivative", term="x", interval="mean")
    margin = oe.margins(result, "x", data=data, outcome=0)
    assert_allclose(probability.std_error, probability_se, rtol=3e-13, atol=0)
    assert_allclose(derivative["dydx[x]"], effect, rtol=3e-13, atol=0)
    assert_allclose(derivative.std_error, effect_se, rtol=3e-13, atol=0)
    assert_allclose(margin.estimate, effect, rtol=3e-13, atol=0)
    assert_allclose(margin.std_error, effect_se, rtol=3e-13, atol=0)
    assert_allclose(margin.attrs["delta_gradients"], [jac_effect], rtol=3e-13, atol=0)


@pytest.mark.parametrize("intercept,parameter,factor", [
    (math.log(2), "2:Intercept", 2 / 27),
    (0.0, "0:Intercept", 1 / 8),
])
def test_multinomial_underflowed_third_category_mass_retains_pair_effect_gradient(
    multinomial_saved, intercept, parameter, factor,
):
    result = multinomial_saved.model_copy(deep=True)
    estimates = {"0:Intercept": intercept, "0:x": 1e200, "2:Intercept": -800.0,
                 "2:x": 0.0}
    for coefficient in result.coefficients:
        coefficient.estimate = estimates[coefficient.term]
    terms = [c.term for c in result.coefficients]
    position = terms.index(parameter)
    covariance = np.zeros((4, 4))
    # Keep the public z statistic finite despite an enormous effect and a tiny
    # gradient. This changes only the known SE scale, not the derivative oracle.
    covariance[position, position] = 1e200
    result.covariance_matrix = covariance.tolist()
    # Differentiate beta0*P0*(1-P0). At intercept log2, the eta2 derivative is
    # -beta0*P0*P2*(1-2*P0) = 2*beta0*exp(-800)/27. At intercept0, the eta0
    # derivative is beta0*P0*(1-P0)*(1-2*P0) = beta0*exp(-800)/8. Corrections
    # involve another underflowed rare factor. Both derivatives remain finite.
    expected = math.exp(math.log(1e200) - 800) * factor
    data = pd.DataFrame({"x": [0.0]})
    margin = oe.margins(result, "x", data=data, outcome=0)
    prediction = oe.predict(result, data, outcome=0, kind="derivative", term="x",
                            interval="mean")
    assert_allclose(margin.attrs["delta_gradients"][0][position], expected, rtol=5e-13, atol=0)
    assert_allclose(margin.std_error, expected * 1e100, rtol=5e-13, atol=0)
    assert_allclose(prediction.std_error, expected * 1e100, rtol=5e-13, atol=0)


@pytest.mark.parametrize("selector", [np.int64(0), np.int64(1), np.float32(2)])
def test_numpy_scalar_actual_outcome_labels_work_after_json(multinomial_saved, selector):
    data = pd.DataFrame({"x": [-.5, .5]})
    canonical = selector.item()
    expected = oe.predict(multinomial_saved, data, outcome=canonical, interval="mean")
    actual = oe.predict(multinomial_saved, data, outcome=selector, interval="mean")
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(oe.margins(multinomial_saved, "x", data=data, outcome=selector),
                                  oe.margins(multinomial_saved, "x", data=data, outcome=canonical))
    with pytest.raises(AnalysisError):
        oe.predict(multinomial_saved, data, outcome=np.bool_(True))


@pytest.mark.parametrize("record", [[], {}])
def test_corrupt_json_order_record_has_structured_error(ordered_saved, record):
    payload = json.loads(ordered_saved.model_dump_json())
    payload["extra"]["category_order"] = record
    result = ResultBundle.model_validate_json(json.dumps(payload))
    with pytest.raises(AnalysisError) as caught:
        oe.predict(result, pd.DataFrame({"x": [0.0]}))
    assert caught.value.code == "invalid_result"


def test_material_negative_prediction_variance_is_rejected_even_with_psd_roundoff_allowance(ordered_saved):
    covariance = [[1, 1 + 6e-14, 0], [1 + 6e-14, 1, 0], [0, 0, 1]]
    result = ordered_state(ordered_saved, covariance=covariance)
    with pytest.raises(AnalysisError) as caught:
        oe.predict(result, pd.DataFrame({"x": [1.0]}), outcome=0, interval="mean")
    assert caught.value.code == "invalid_inference"


def test_material_negative_marginal_effect_variance_is_rejected(ordered_saved):
    if ordered_saved.spec.estimator != "ologit":
        pytest.skip("The exact logistic derivative contrast identifies this fixture.")
    covariance = [[1, -1 - 6e-14, 0], [-1 - 6e-14, 1, 0], [0, 0, 1]]
    result = ordered_state(ordered_saved, slope=2.0, cuts=(-math.log(3), 2.0), covariance=covariance)
    with pytest.raises(AnalysisError) as caught:
        oe.margins(result, "x", data=pd.DataFrame({"x": [0.0]}), outcome=0)
    assert caught.value.code == "invalid_inference"
