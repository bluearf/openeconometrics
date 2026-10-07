"""Malformed saved results and restrictions must produce structured errors."""

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from test_econ_postest_inference import bundle


@pytest.mark.parametrize("restriction,null,code", [
    ("absent", 0, "unknown_term"), (["x", "x"], 0, "invalid_restrictions"),
    ({}, 0, "invalid_restrictions"), ([], 0, "invalid_restrictions"),
    (2, 0, "invalid_restrictions"), ([[1, 0]], 0, "invalid_restrictions"),
    ([[1, 0, 0], [2, 0, 0]], 0, "invalid_restrictions"),
    ([[0, 0, 0]], 0, "invalid_restrictions"),
    ([{"x": 1}, {"z": 1}], [0], "invalid_restrictions"),
    ({"x": [1]}, 0, "invalid_restrictions"),
    ({"x": float("nan")}, 0, "invalid_restrictions"),
    ({"x": complex(1, 2)}, 0, "invalid_restrictions"),
    ({"x": True}, 0, "invalid_restrictions"),
    ({"x": 1}, float("inf"), "invalid_restrictions"),
    ({"x": 1}, True, "invalid_restrictions"),
    ([[[1, 0, 0]]], 0, "invalid_restrictions"),
])
def test_invalid_restrictions(restriction, null, code):
    with pytest.raises(AnalysisError) as error:
        oe.test(bundle(), restriction, null)
    assert error.value.code == code


@pytest.mark.parametrize("mutate,code", [
    (lambda m: setattr(m, "covariance_matrix", [[1, 0], [0, 1]]), "invalid_result"),
    (lambda m: setattr(m, "covariance_matrix", [[1, 2, 0], [0, 1, 0], [0, 0, 1]]), "invalid_result"),
    (lambda m: setattr(m, "covariance_matrix", [[1, 2, 0], [2, 1, 0], [0, 0, 1]]), "invalid_result"),
    (lambda m: setattr(m, "covariance_matrix", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), "invalid_result"),
    (lambda m: setattr(m, "covariance_matrix", [[0, .1, 0], [.1, 1, 0], [0, 0, 1]]), "invalid_result"),
    (lambda m: setattr(m.coefficients[0], "estimate", float("nan")), "invalid_result"),
    (lambda m: setattr(m.coefficients[1], "term", "Intercept"), "invalid_result"),
    (lambda m: setattr(m.coefficients[0], "equation", ""), "invalid_result"),
    (lambda m: m.inference.update(use_t=True, df_inference=None), "invalid_inference"),
    (lambda m: m.inference.update(use_t=True, distribution="t", df_inference=0), "invalid_inference"),
    (lambda m: m.inference.update(use_t=True, distribution="t", df_inference=float("inf")), "invalid_inference"),
    (lambda m: m.inference.update(use_t=True, distribution="t", df_inference=True), "invalid_inference"),
    (lambda m: m.inference.update(use_t="true"), "invalid_inference"),
    (lambda m: m.inference.update(distribution="F"), "invalid_inference"),
    (lambda m: m.inference.update(alpha=True), "invalid_inference"),
    (lambda m: m.inference.update(alpha=0), "invalid_inference"),
    (lambda m: m.inference.update(hansen=True), "estimation_state_unavailable"),
])
def test_malformed_result_records(mutate, code):
    model = bundle()
    mutate(model)
    with pytest.raises(AnalysisError) as error:
        oe.lincom(model, {"x": 1})
    assert error.value.code == code


@pytest.mark.parametrize("function", [
    "x / z", lambda b: 1., lambda b: b["x"].detach(), lambda b: b["x"][None],
    lambda b: b["x"] / 0, lambda b: (b["x"] - b["x"]).sqrt(),
    lambda b: b["x"].to(torch.complex128),
])
def test_nonlinear_expression_has_finite_scalar_derivatives(function):
    with pytest.raises(AnalysisError) as error:
        oe.nlcom(bundle(), function)
    assert error.value.code == "invalid_nlcom"


def test_zero_gradient_and_zero_restriction_variance_rejected_but_other_singular_covariance_allowed():
    model = bundle()
    model.covariance_matrix = [[.2, 0, 0], [0, .04, 0], [0, 0, 0]]
    assert oe.test(model, "x")["statistic"] > 0
    for operation in (lambda: oe.test(model, "z"), lambda: oe.lincom(model, {"z": 1}),
                      lambda: oe.nlcom(model, lambda b: b["x"] * 0)):
        with pytest.raises(AnalysisError) as error:
            operation()
        assert error.value.code == "nonestimable_restriction"


def test_alias_duplicate_and_undefined_nonlinear_term_are_explicit():
    model = bundle(terms=["a:Intercept", "a:x", "b:x"], equations=["a", "a", "b"])
    with pytest.raises(AnalysisError) as error:
        oe.lincom(model, {"a:x": 1, "[a]x": -1})
    assert error.value.code == "invalid_restrictions"
    with pytest.raises(AnalysisError) as error:
        oe.nlcom(model, lambda b: b["absent"])
    assert error.value.code == "unknown_term"


@pytest.mark.parametrize("patterns", [[], "", ["x", ""], [1], None])
def test_bad_testparm_patterns(patterns):
    with pytest.raises(AnalysisError) as error:
        oe.testparm(bundle(), patterns)
    assert error.value.code == "invalid_restrictions"


def test_missing_result_and_inference_are_not_silently_assumed_normal():
    with pytest.raises(AnalysisError) as error:
        oe.test({}, "x")
    assert error.value.code == "invalid_result"
    model = bundle()
    model.inference.clear()
    with pytest.raises(AnalysisError) as error:
        oe.test(model, "x")
    assert error.value.code == "invalid_inference"
