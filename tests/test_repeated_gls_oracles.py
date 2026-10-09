"""Independent R/nlme optima and NumPy observed-information residual GLS oracles.

No production covariance constructor, objective, gradient or Hessian is used by
the oracle. External fixtures preserve nlme's separate ML fixed covariance
convention; OpenEcon's ML joint OIM is checked from an independently evaluated
joint Gaussian likelihood. REML's design normalization is an explicit constant.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import chi2, norm

from openecon.analysis_contracts import AnalysisError


FIXTURE = Path(__file__).parent / "fixtures/repeated_gls/nlme.json"
CASES = tuple(f"{structure}_{method}" for structure in ("cs", "ar1", "diagonal", "unstructured")
              for method in ("ml", "reml"))


@lru_cache
def references():
    return json.loads(FIXTURE.read_text())["cases"]


def frame_for(case):
    frame = pd.DataFrame(references()[case]["data"])
    frame.index = [f"observation-{j:03d}" for j in range(len(frame))]
    return frame


def covariance(theta, structure, levels):
    """Independent raw parameter to global occasion covariance construction."""
    theta = np.asarray(theta, dtype=float)
    q = len(levels)
    if structure == "cs":
        lower = -1/(q-1)
        rho = lower + (1-lower)*expit(theta[1])
        return np.exp(2*theta[0])*((1-rho)*np.eye(q) + rho*np.ones((q, q)))
    if structure == "ar1":
        gaps = np.abs(np.subtract.outer(levels, levels))
        return np.exp(2*theta[0])*np.power(np.tanh(theta[1]), gaps)
    if structure == "diagonal":
        return np.diag(np.exp(2*theta))
    lower = np.zeros((q, q))
    ii, jj = np.tril_indices(q)
    lower[ii, jj] = theta
    lower[np.diag_indices(q)] = np.exp(lower.diagonal())
    return lower @ lower.T


def natural_parameters(theta, structure, levels):
    matrix = covariance(theta, structure, levels)
    if structure in ("cs", "ar1"):
        return np.array([matrix[0, 0], matrix[0, 1]/matrix[0, 0]])
    if structure == "diagonal":
        return matrix.diagonal()
    ii, jj = np.tril_indices(len(levels), -1)
    return np.r_[matrix.diagonal(), matrix[ii, jj]]


def theta_from_covariance(matrix, structure, levels):
    matrix = np.asarray(matrix)
    if structure in ("cs", "ar1"):
        rho = matrix[0, 1]/matrix[0, 0]
        if structure == "cs":
            lower = -1/(len(levels)-1)
            eta = np.log((rho-lower)/(1-rho))
        else:
            eta = np.arctanh(rho)  # fixture's first two actual times have lag one
        return np.array([.5*np.log(matrix[0, 0]), eta])
    if structure == "diagonal":
        return .5*np.log(matrix.diagonal())
    lower = np.linalg.cholesky(matrix)
    lower[np.diag_indices(len(levels))] = np.log(lower.diagonal())
    return lower[np.tril_indices(len(levels))]


class DenseOracle:
    def __init__(self, frame, structure):
        self.frame, self.structure = frame, structure
        self.levels = sorted(frame.occasion.unique())
        self.x = np.column_stack([np.ones(len(frame)), frame[["x", "z"]]])
        self.y = frame.y.to_numpy()
        self.groups = [np.flatnonzero(frame.subject.to_numpy() == label) for label in pd.unique(frame.subject)]
        self.occasions = np.array([self.levels.index(value) for value in frame.occasion])
        self.n, self.p = self.x.shape
        self.design_logdet = np.linalg.slogdet(self.x.T @ self.x)[1]

    def moments(self, theta):
        base = covariance(theta, self.structure, self.levels)
        a, b, c, logdet = np.zeros((self.p, self.p)), np.zeros(self.p), 0., 0.
        dense = np.zeros((self.n, self.n))
        for rows in self.groups:
            positions = self.occasions[rows]
            block = base[np.ix_(positions, positions)]
            dense[np.ix_(rows, rows)] = block
            inverse = np.linalg.inv(block)
            x, y = self.x[rows], self.y[rows]
            a += x.T @ inverse @ x
            b += x.T @ inverse @ y
            c += y @ inverse @ y
            logdet += np.linalg.slogdet(block)[1]
        return a, b, c, logdet, dense

    def evaluate(self, theta, method, beta=None):
        a, b, c, logdet, dense = self.moments(theta)
        fitted = np.linalg.solve(a, b)
        beta = fitted if beta is None else beta
        quadratic = c - 2*beta @ b + beta @ a @ beta
        nll = .5*(self.n*np.log(2*np.pi) + logdet + quadratic)
        if method == "REML":
            nll += .5*(-self.p*np.log(2*np.pi) + np.linalg.slogdet(a)[1] - self.design_logdet)
        return float(nll), fitted, np.linalg.inv(a), dense

    def objective(self, params, method):
        if method == "ML":
            return self.evaluate(params[self.p:], method, beta=params[:self.p])[0]
        return self.evaluate(params, method)[0]


def hessian(function, point, multiplier=1.):
    """Independent central mixed/5-point diagonal numerical observed Hessian."""
    point = np.asarray(point, dtype=float)
    steps = 2e-4*multiplier*np.maximum(1, np.abs(point))
    value = function(point)
    matrix = np.empty((len(point), len(point)))
    for i, step in enumerate(steps):
        delta = np.eye(len(point))[i]*step
        matrix[i, i] = (-function(point+2*delta) + 16*function(point+delta) - 30*value
                        + 16*function(point-delta) - function(point-2*delta))/(12*step*step)
        for j in range(i):
            other = np.eye(len(point))[j]*steps[j]
            matrix[i, j] = matrix[j, i] = (function(point+delta+other) - function(point+delta-other)
                - function(point-delta+other) + function(point-delta-other))/(4*step*steps[j])
    return matrix


def jacobian(function, point):
    point = np.asarray(point, dtype=float)
    columns = []
    for i in range(len(point)):
        delta = np.eye(len(point))[i]*(1e-5*max(1., abs(point[i])))
        columns.append((function(point+delta)-function(point-delta))/(2*delta[i]))
    return np.column_stack(columns)


@lru_cache
def fitted_case(case):
    from openecon.econometrics.mixed.repeated import repeated_gls
    reference = references()[case]
    return repeated_gls(data=frame_for(case), y="y", x=["x", "z"], subject="subject", occasion="occasion",
                        structure=reference["structure"], method=reference["method"])


@pytest.mark.parametrize("case", CASES)
def test_all_eight_external_optima_residual_geometry_and_independent_observed_information(case):
    reference = references()[case]
    frame = frame_for(case)
    oracle = DenseOracle(frame, reference["structure"])
    result = fitted_case(case)
    state, fit = result.attrs["state"], result.attrs["state"]["fit"]
    theta, beta = np.asarray(fit["theta"]), np.asarray(fit["beta"])
    objective, dense_beta, bread, residual_covariance = oracle.evaluate(theta, reference["method"])
    expected_likelihood = reference["fit"]["log_likelihood"]
    if reference["method"] == "REML":
        expected_likelihood += .5*oracle.design_logdet
    assert_allclose(beta, reference["coefficients"], rtol=5e-5, atol=4e-6)
    assert_allclose(beta, dense_beta, rtol=2e-9, atol=2e-9)
    assert_allclose(-objective, expected_likelihood, atol=3e-6, rtol=2e-8)
    assert_allclose(fit[f"loglik_{reference['method'].lower()}"], -objective, atol=2e-8, rtol=2e-9)
    assert_allclose(covariance(theta, reference["structure"], oracle.levels), reference["occasion_covariance"], rtol=5e-4, atol=8e-6)
    assert_allclose(result["residual_covariance"], residual_covariance, rtol=2e-9, atol=2e-9)
    assert_allclose(fit["conditional_gls_covariance"], bread, rtol=2e-9, atol=2e-9)
    assert state["levels"] == [0, 1, 3, 4]
    assert state["row_labels"] == list(frame.index)
    assert state["subject_labels"] == frame.subject.tolist()
    assert state["occasion_labels"] == frame.occasion.tolist()
    # Off-block covariance must be exactly zero; within blocks missing occasions
    # select the common global covariance rather than each subject's row ranks.
    different = frame.subject.to_numpy()[:, None] != frame.subject.to_numpy()[None, :]
    assert np.array_equal(residual_covariance[different], np.zeros(different.sum()))
    natural = natural_parameters(theta, reference["structure"], oracle.levels)
    assert_allclose(fit["covariance_parameters"], natural, rtol=2e-9, atol=2e-9)
    if reference["structure"] in ("cs", "ar1"):
        assert natural[1] < -.15
    point = np.r_[beta, theta] if reference["method"] == "ML" else theta
    def function(value):
        return oracle.objective(value, reference["method"])
    observed = hessian(function, point)
    fine = hessian(function, point, multiplier=.5)
    assert_allclose(observed, fine, rtol=3e-4, atol=8e-5)
    raw_covariance = np.linalg.inv(fine)
    transform = jacobian(lambda value: natural_parameters(value, reference["structure"], oracle.levels), theta)
    if reference["method"] == "ML":
        full_transform = np.zeros_like(raw_covariance)
        full_transform[:oracle.p, :oracle.p] = np.eye(oracle.p)
        full_transform[oracle.p:, oracle.p:] = transform
        expected_joint = full_transform @ raw_covariance @ full_transform.T
        assert_allclose(fit["joint_raw_covariance"], raw_covariance, rtol=3e-4, atol=8e-6)
        assert_allclose(fit["beta_theta_covariance"], raw_covariance[:oracle.p, oracle.p:], rtol=3e-4, atol=8e-6)
        assert_allclose(result["joint_covariance"], expected_joint, rtol=3e-4, atol=8e-6)
        expected_fixed = raw_covariance[:oracle.p, :oracle.p]
        expected_theta = expected_joint[oracle.p:, oracle.p:]
        # Estimated residual nuisance uncertainty has a nonzero cross block.
        assert np.max(np.abs(expected_joint[:oracle.p, oracle.p:])) > 1e-5
    else:
        expected_fixed = bread
        expected_theta = transform @ raw_covariance @ transform.T
        assert "joint_covariance" not in result
        assert fit["joint_covariance"] is None
        assert fit["joint_raw_covariance"] is None
        assert fit["beta_theta_covariance"] is None
    assert_allclose(result["covariance"], expected_fixed, rtol=3e-4, atol=8e-6)
    assert_allclose(result["covariance_parameter_covariance"], expected_theta, rtol=3e-4, atol=1e-5)
    assert_allclose(result["covariance_parameters"].std_error, np.sqrt(np.diag(expected_theta)), rtol=2e-4, atol=3e-6)
    standard_error = np.sqrt(np.diag(expected_fixed))
    assert_allclose(result["coefficients"].std_error, standard_error, rtol=2e-4, atol=3e-6)
    assert_allclose(result["coefficients"].p_value, 2*norm.sf(np.abs(beta/standard_error)), rtol=8e-4, atol=1e-7)
    assert_allclose(result["coefficients"].ci_low, beta-norm.ppf(.975)*standard_error, rtol=2e-4, atol=1e-5)
    assert_allclose(result["coefficients"].ci_high, beta+norm.ppf(.975)*standard_error, rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("case", CASES)
def test_row_permutation_retains_subject_geometry_original_labels_and_inference(case):
    from openecon.econometrics.mixed.repeated import repeated_gls
    original = fitted_case(case)
    frame = frame_for(case)
    order = np.random.default_rng(5634).permutation(len(frame))
    permuted = repeated_gls(data=frame.iloc[order], y="y", x=["x", "z"], subject="subject", occasion="occasion",
                            structure=references()[case]["structure"], method=references()[case]["method"])
    assert_allclose(permuted["coefficients"].estimate, original["coefficients"].estimate, rtol=1e-6, atol=1e-6)
    assert_allclose(permuted["covariance"], original["covariance"], rtol=3e-5, atol=2e-7)
    assert_allclose(permuted["residual_covariance"], original["residual_covariance"].to_numpy()[np.ix_(order, order)], rtol=2e-5, atol=2e-6)
    assert permuted.attrs["state"]["row_labels"] == frame.index[order].tolist()


@pytest.mark.parametrize("case", CASES)
def test_response_and_predictor_scale_equivariance_of_complete_covariance(case):
    from openecon.econometrics.mixed.repeated import repeated_gls
    original = fitted_case(case)
    frame = frame_for(case)
    frame["y"] *= 3.5
    frame["x"] *= 7
    frame["z"] *= -.4
    scaled = repeated_gls(data=frame, y="y", x=["x", "z"], subject="subject", occasion="occasion",
                          structure=references()[case]["structure"], method=references()[case]["method"])
    transform = np.diag([3.5, 3.5/7, 3.5/-.4])
    assert_allclose(scaled["coefficients"].estimate, transform @ original["coefficients"].estimate, rtol=2e-5, atol=2e-6)
    assert_allclose(scaled["covariance"], transform @ original["covariance"].to_numpy() @ transform, rtol=6e-5, atol=3e-6)
    assert_allclose(scaled["residual_covariance"], 3.5**2*original["residual_covariance"], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("case", CASES)
def test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize(case, monkeypatch):
    from openecon.econometrics.mixed import repeated_kernels
    from openecon.econometrics.mixed.repeated import repeated_gls_contrast, repeated_gls_predict, restore_repeated_gls
    result = fitted_case(case)
    def forbidden(*args, **kwargs):
        pytest.fail("saved result inference attempted an optimizer/refit")
    monkeypatch.setattr(repeated_kernels, "fit", forbidden)
    state = json.loads(json.dumps(result.attrs["state"], sort_keys=True, allow_nan=False))
    restored = restore_repeated_gls(state)
    assert list(restored) == list(result)
    assert restored.attrs == result.attrs
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_exact=True)
    assert restored.to_latex() == result.to_latex()
    profiles = pd.DataFrame({"x": [-.3, .7, 1.4], "z": [.2, -.5, .4]}, index=["first", "second", "third"])
    x = np.column_stack([np.ones(len(profiles)), profiles])
    beta, cov = result["coefficients"].estimate.to_numpy(), result["covariance"].to_numpy()
    prediction = repeated_gls_predict(restored, profiles)
    assert_allclose(prediction["means"].estimate, x @ beta, rtol=1e-12, atol=1e-12)
    assert_allclose(prediction["covariance"], x @ cov @ x.T, rtol=1e-12, atol=1e-12)
    contrast = pd.DataFrame([[0, 1, -1], [1, .2, .4]], columns=["Intercept", "x", "z"], index=["difference", "profile"])
    null = pd.Series([.6, 1.4], index=contrast.index)
    tested = repeated_gls_contrast(restored, contrast, null=null)
    contrast_matrix = contrast.to_numpy()
    estimate, uncertainty = contrast_matrix @ beta, contrast_matrix @ cov @ contrast_matrix.T
    delta = estimate-null.to_numpy()
    wald = delta @ np.linalg.solve(uncertainty, delta)
    assert_allclose(tested["contrasts"].estimate, estimate, rtol=1e-12, atol=1e-12)
    assert_allclose(tested["covariance"], uncertainty, rtol=1e-12, atol=1e-12)
    assert_allclose(tested["joint_test"].statistic, [wald], rtol=1e-12, atol=1e-12)
    assert_allclose(tested["joint_test"].p_value, [chi2.sf(wald, 2)], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("field", ["beta", "theta", "covariance", "residual_covariance", "covariance_parameter_covariance", "loglik_ml"])
def test_resealed_scientific_tamper_is_refused(field):
    from openecon.econometrics.mixed.repeated import restore_repeated_gls
    state = copy.deepcopy(fitted_case("unstructured_ml").attrs["state"])
    value = state["fit"][field]
    if isinstance(value, list):
        if isinstance(value[0], list):
            value[0][0] += .05
        else:
            value[0] += .05
    else:
        state["fit"][field] += .05
    state.pop("checksum")
    state["checksum"] = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(AnalysisError) as error:
        restore_repeated_gls(state)
    assert error.value.code == "invalid_result_state"


def reseal(state):
    state = copy.deepcopy(state)
    state.pop("checksum", None)
    state["checksum"] = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return state


@pytest.mark.parametrize("case,expected", [
    ("duplicate_pair", "duplicate_observation"), ("fractional_occasion", "invalid_occasion"),
    ("boolean_outcome", "invalid_numeric"), ("complex_predictor", "invalid_numeric"),
    ("numeric_string_outcome", "invalid_numeric"), ("missing_outcome", "missing_values"),
    ("missing_occasion", "missing_values"), ("ambiguous_subject", "ambiguous_identifier"),
    ("boolean_subject", "invalid_identifier"), ("aliased_numeric_subject", "ambiguous_identifier"),
    ("singular_design", "singular_design"), ("huge_outcome", "numeric_domain"),
    ("coarse_calendar", "unsupported_occasion_grid"), ("few_subjects", "resource_limit"),
    ("too_many_rows", "resource_limit"), ("insufficient_occasion", "insufficient_occasion_support"),
])
def test_invalid_admission_is_explicit_and_never_reaches_optimizer(case, expected, monkeypatch):
    from openecon.econometrics.mixed import repeated_kernels
    from openecon.econometrics.mixed.repeated import repeated_gls
    frame = frame_for("ar1_ml")
    if case == "duplicate_pair":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif case == "fractional_occasion":
        frame["occasion"] = frame.occasion.astype(float)
        frame.iloc[0, frame.columns.get_loc("occasion")] = .25
    elif case == "boolean_outcome":
        frame["y"] = frame.y > 0
    elif case == "complex_predictor":
        frame["x"] = frame.x.astype(complex)
    elif case == "numeric_string_outcome":
        frame["y"] = frame.y.astype(str)
    elif case == "missing_outcome":
        frame.iloc[0, frame.columns.get_loc("y")] = np.nan
    elif case == "missing_occasion":
        frame["occasion"] = frame.occasion.astype(float)
        frame.iloc[0, frame.columns.get_loc("occasion")] = np.nan
    elif case in ("ambiguous_subject", "aliased_numeric_subject", "boolean_subject"):
        frame["subject"] = frame.subject.astype(object)
        first, second = ((1, "1") if case == "ambiguous_subject" else (1, 1.0))
        frame.loc[frame.subject == "s00", "subject"] = first
        frame.loc[frame.subject == "s01", "subject"] = second
        if case == "boolean_subject":
            frame.iloc[0, frame.columns.get_loc("subject")] = True
    elif case == "singular_design":
        frame["z"] = 2*frame.x + 1
    elif case == "huge_outcome":
        frame.iloc[0, frame.columns.get_loc("y")] = 1e7
    elif case == "coarse_calendar":
        frame["occasion"] *= 2
    elif case == "few_subjects":
        frame = frame[frame.subject.isin(["s00", "s01", "s02"])]
    elif case == "too_many_rows":
        frame = pd.concat([frame]*3)
    elif case == "insufficient_occasion":
        frame = frame[(frame.occasion != 4) | frame.subject.isin(["s00", "s01", "s02"])]
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached the optimizer")
    monkeypatch.setattr(repeated_kernels, "fit", forbidden)
    with pytest.raises(AnalysisError) as error:
        repeated_gls(data=frame, y="y", x=["x", "z"], subject="subject", occasion="occasion", structure="ar1", method="ML")
    assert error.value.code == expected


@pytest.mark.parametrize("kind", ["beta_display", "fixed_covariance_display", "residual_covariance_display",
                                   "fitted_display", "natural_parameter_display"])
def test_visible_result_tamper_cannot_anchor_saved_inference(kind):
    from openecon.econometrics.mixed.repeated import repeated_gls_predict
    result = copy.deepcopy(fitted_case("cs_ml"))
    if kind == "beta_display":
        result["coefficients"].iloc[0, result["coefficients"].columns.get_loc("estimate")] += .1
    elif kind == "fixed_covariance_display":
        result["covariance"].iloc[0, 0] += .1
    elif kind == "residual_covariance_display":
        result["residual_covariance"].iloc[0, 0] += .1
    elif kind == "fitted_display":
        result["fitted"].iloc[0, result["fitted"].columns.get_loc("observed")] += .1
    else:
        result["covariance_parameters"].iloc[0, result["covariance_parameters"].columns.get_loc("estimate")] += .1
    with pytest.raises(AnalysisError) as error:
        repeated_gls_predict(result, pd.DataFrame({"x": [.1], "z": [.2]}))
    assert error.value.code == "invalid_result_state"


@pytest.mark.parametrize("kind", ["boolean_intercept", "string_intercept", "boolean_beta", "string_beta",
                                  "boolean_row_label", "subject_identity", "occasion_identity", "structure"])
def test_resealed_state_types_and_identity_tamper_cannot_bypass_semantic_validation(kind):
    from openecon.econometrics.mixed.repeated import restore_repeated_gls
    state = copy.deepcopy(fitted_case("cs_ml").attrs["state"])
    if kind == "boolean_intercept":
        state["x"][0][0] = True
    elif kind == "string_intercept":
        state["x"][0][0] = "1.0"
    elif kind == "boolean_beta":
        state["fit"]["beta"][0] = True
    elif kind == "string_beta":
        state["fit"]["beta"][0] = str(state["fit"]["beta"][0])
    elif kind == "boolean_row_label":
        state["row_labels"][0] = True
    elif kind == "subject_identity":
        state["subject_labels"][0] = "different-subject"
    elif kind == "occasion_identity":
        state["occasion_labels"][0] = 100
    elif kind == "structure":
        state["structure"] = "ar1"
    with pytest.raises(AnalysisError) as error:
        restore_repeated_gls(reseal(state))
    assert error.value.code == "invalid_result_state"
