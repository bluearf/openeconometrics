"""Independent Gaussian-system likelihood, joint-inference and query proofs."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from scripts.nonlinear_sur_reference import Reference, complex_jacobian, fixture, jacobian


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fit_public(frame, reference, vce="oim"):
    return oe.nlsur(frame, equations=reference.definitions, start=reference.start,
                    covariance=vce, cluster="cluster" if vce == "cr0" else None)


def physical(result):
    return np.asarray(result.attrs["nonlinear_sur_state"]["fit"]["params"])


def matrix(result, name):
    return result[name].iloc[:, 1:].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def sample():
    return fixture(rows=256, dependence=True)[0]


@pytest.fixture(scope="module")
def reference(sample):
    return Reference(sample)


@pytest.fixture(scope="module")
def oracle(reference):
    theta, best, starts = reference.fit()
    assert max(abs(reference.profile(theta[:reference.q])[1])) < 1e-6
    assert max(abs(value.fun-best.fun) for value in starts) < 1e-7
    return theta, best


@pytest.fixture(scope="module", params=("oim", "hc0", "cr0"))
def fitted(request, sample, reference):
    return request.param, fit_public(sample, reference, request.param)


@pytest.mark.parametrize("mode,equations", [("quadratic", 2), ("quadratic", 3),
                                           ("nonlinear", 2), ("linear_shared", 2),
                                           ("linear_separate", 2)])
def test_oracle_analytic_scores_hessian_and_mean_derivatives(mode, equations):
    frame, theta = fixture(rows=96, mode=mode, equations=equations)
    ref = Reference(frame, mode=mode, equations=equations)
    moments = ref.moments(theta)
    numerical_score = jacobian(lambda value: ref.moments(value)["row_loglikelihood"], theta)
    numerical_information = -jacobian(lambda value: ref.moments(value)["row_scores"].sum(0), theta)
    np.testing.assert_allclose(moments["row_scores"], numerical_score, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(moments["information"], numerical_information, rtol=2e-7, atol=2e-6)
    np.testing.assert_allclose(ref.derivatives(theta[:ref.q])[0], complex_jacobian(ref.means, theta[:ref.q]),
                               rtol=3e-14, atol=3e-14)


def test_all_physical_parameters_match_multistart_scipy(fitted, oracle):
    _, result = fitted
    theta, best = oracle
    np.testing.assert_allclose(physical(result), theta, rtol=2e-6, atol=3e-7)
    assert result.attrs["nonlinear_sur_state"]["fit"]["log_likelihood"] == pytest.approx(-best.fun, abs=2e-8)


@pytest.mark.parametrize("name", ("information", "bread", "meat", "covariance", "case_scores"))
def test_every_full_physical_matrix_and_score_cell(fitted, reference, name):
    vce, result = fitted
    expected = reference.covariance(physical(result), vce)
    actual = result.attrs["nonlinear_sur_state"]["fit"][name]
    np.testing.assert_allclose(actual, expected["row_scores" if name == "case_scores" else name],
                               rtol=3e-7, atol=3e-8)


def test_covariance_divisor_whole_row_likelihood_and_cluster_aggregation(fitted, reference):
    vce, result = fitted
    actual = result.attrs["nonlinear_sur_state"]["fit"]
    expected = reference.moments(physical(result))
    residual = expected["residuals"]
    np.testing.assert_allclose(actual["sigma"], residual.T@residual/len(residual), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual["case_loglikelihood"], expected["row_loglikelihood"], rtol=2e-12, atol=2e-12)
    if vce == "cr0":
        np.testing.assert_allclose(actual["cluster_scores"], expected["cluster_scores"], rtol=3e-10, atol=3e-10)
    else:
        assert actual["cluster_scores"] == []
    # Two equation responses form one likelihood/score row; cross products of
    # their Gaussian contributions must not be discarded.
    assert len(actual["case_scores"]) == len(reference.frame)


def test_observed_information_retains_beta_sigma_crossblocks(fitted, reference):
    _, result = fitted
    theta = physical(result)
    expected = reference.moments(theta)
    information = np.asarray(result.attrs["nonlinear_sur_state"]["fit"]["information"])
    assert np.linalg.norm(information[:reference.q, reference.q:]) > 1e-3
    np.testing.assert_allclose(information, expected["information"], rtol=3e-7, atol=3e-7)


def test_full_parameter_inference_tables(fitted, reference):
    _, result = fitted
    frame = result["parameters"]
    theta, covariance = physical(result), matrix(result, "covariance")
    se = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(frame.estimate, theta, rtol=0, atol=0)
    np.testing.assert_allclose(frame.std_error, se, rtol=2e-12, atol=2e-12)
    critical = norm.ppf(.975)
    diagonal = {reference.q+k for k, (i, j) in enumerate(reference.pairs) if i == j}
    for j, row in enumerate(frame.itertuples()):
        if j in diagonal:
            lower, upper = theta[j]*np.exp(np.array([-1., 1.])*critical*se[j]/theta[j])
        else:
            lower, upper = theta[j]+np.array([-1., 1.])*critical*se[j]
        assert row.ci_lower == pytest.approx(lower, rel=3e-12, abs=3e-12)
        assert row.ci_upper == pytest.approx(upper, rel=3e-12, abs=3e-12)
        if j < reference.q:
            assert row.z == pytest.approx(theta[j]/se[j], rel=3e-12)
            assert row.p_value == pytest.approx(2*norm.sf(abs(theta[j]/se[j])), rel=3e-10, abs=1e-14)


@pytest.mark.parametrize("mode,equations", [("quadratic", 3), ("nonlinear", 2),
                                           ("linear_shared", 2), ("linear_separate", 2)])
def test_alternate_systems_match_independent_full_likelihood_and_covariance(mode, equations):
    frame, _ = fixture(rows=192, mode=mode, equations=equations)
    ref = Reference(frame, mode=mode, equations=equations)
    result = fit_public(frame, ref)
    theta, best, _ = ref.fit()
    np.testing.assert_allclose(physical(result), theta, rtol=4e-6, atol=6e-7)
    np.testing.assert_allclose(matrix(result, "information"), ref.moments(physical(result))["information"],
                               rtol=2e-7, atol=2e-7)
    assert result.attrs["nonlinear_sur_state"]["fit"]["log_likelihood"] == pytest.approx(-best.fun, abs=1e-7)
    if mode == "nonlinear":
        theta = physical(result)
        first, _ = ref.derivatives(theta[:ref.q])
        gauss_newton = np.einsum("nmp,mk,nkq->pq", first, np.linalg.inv(ref.sigma(theta)), first)
        assert np.linalg.norm(matrix(result, "information")[:ref.q, :ref.q]-gauss_newton) > 1e-3


def test_identical_linear_design_reduces_to_closed_ols_sur():
    frame, _ = fixture(rows=160, mode="linear_separate")
    ref = Reference(frame, mode="linear_separate")
    result = fit_public(frame, ref)
    x = np.column_stack([np.ones(len(frame)), frame.x])
    beta = np.linalg.lstsq(x, ref.y, rcond=None)[0]
    residual = ref.y-x@beta
    sigma = residual.T@residual/len(frame)
    np.testing.assert_allclose(physical(result)[:4], beta.T.ravel(), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(matrix(result, "covariance")[:4, :4], np.kron(sigma, np.linalg.inv(x.T@x)),
                               rtol=3e-7, atol=2e-8)


@pytest.fixture(scope="module")
def query(sample):
    frame = sample.iloc[:4].copy()
    frame["x"], frame["z"] = [.2, .5, .8, 1.1], [.3, .6, .9, 1.2]
    frame["fixed_weight"] = [0., 1., 2., 3.]
    frame.index = pd.Index(["a", "a", 3, None], name="query")
    return frame


@pytest.mark.parametrize("given", [[], ["eq1"], ["eq2"]])
def test_conditional_unconditional_means_residual_sigma_and_full_joint_delta(fitted, query, given):
    _, fit = fitted
    result = oe.nlsur_predict(fit, query, given=given)
    ref, theta = Reference(query), physical(fit)
    expected, means, residual = ref.predictions(theta, given)
    gradient = complex_jacobian(lambda value: ref.predictions(value, given)[0], theta)
    actual = np.r_[result["means"].estimate, result["residual_covariance"].estimate]
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(matrix(result, "parameter_jacobian"), gradient, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(matrix(result, "target_covariance"), gradient@matrix(fit, "covariance")@gradient.T,
                               rtol=5e-9, atol=3e-11)
    np.testing.assert_allclose(matrix(result, "residual_variation"), residual, rtol=3e-12, atol=3e-12)
    assert list(result["means"].index) == list(query.index.repeat(means.shape[1]))
    if given:
        assert np.linalg.norm(gradient[:len(means), ref.q:]) > .01


@pytest.mark.parametrize("given", [[], ["eq1"], ["eq2"]])
@pytest.mark.parametrize("scale", ["effect", "elasticity"])
def test_margins_mixed_derivatives_and_fixed_weighted_averages(fitted, query, given, scale):
    _, fit = fitted
    result = oe.nlsur_margins(fit, query, x=["x", "z"], given=given, scale=scale, weights="fixed_weight")
    ref, theta = Reference(query), physical(fit)
    weights = query.fixed_weight.to_numpy()
    expected, _, _ = ref.effects(theta, ["x", "z"], given=given, scale=scale, weights=weights)
    gradient = complex_jacobian(lambda value: ref.effects(value, ["x", "z"], given=given,
                                                          scale=scale, weights=weights)[0], theta)
    np.testing.assert_allclose(np.r_[result["effects" if scale == "effect" else "elasticities"].estimate,
                                     result["averages"].estimate], expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(matrix(result, "parameter_jacobian"), gradient, rtol=3e-9, atol=3e-10)
    np.testing.assert_allclose(matrix(result, "target_covariance"), gradient@matrix(fit, "covariance")@gradient.T,
                               rtol=3e-8, atol=3e-11)


@pytest.mark.parametrize("given", [[], ["eq1"], ["eq2"]])
def test_effects_are_independent_raw_data_derivatives(fitted, query, given):
    _, fit = fitted
    theta = physical(fit)
    expected = Reference(query).effects(theta, ["x", "z"], given=given)[1]
    for row in range(len(query)):
        for column, name in enumerate(("x", "z")):
            delta = 1e-5
            shifted = []
            for direction in (-1., 1.):
                frame = query.copy()
                frame.iloc[row, frame.columns.get_loc(name)] += direction*delta
                shifted.append(Reference(frame).predictions(theta, given)[1][row])
            np.testing.assert_allclose((shifted[1]-shifted[0])/(2*delta), expected[row, :, column],
                                       rtol=2e-9, atol=2e-10)


def test_nonlinear_mean_covariance_contrasts_and_joint_wald(fitted):
    _, fit = fitted
    expressions = {"curvature": "{amp}*exp({rate})", "correlation":
                   "{cov__eq2__eq1}/sqrt({cov__eq1__eq1}*{cov__eq2__eq2})", "gap": "{b0}-{b1}"}
    null = [.9, .1, 1.5]
    result = oe.nlsur_contrast(fit, expressions, null=null)
    theta = physical(fit)
    def target(value):
        return np.array([value[2]*np.exp(value[3]), value[5]/np.sqrt(value[4]*value[6]), value[0]-value[1]])
    gradient = complex_jacobian(target, theta)
    covariance = gradient@matrix(fit, "covariance")@gradient.T
    np.testing.assert_allclose(result["contrasts"].estimate, target(theta), rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(matrix(result, "parameter_jacobian"), gradient, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(matrix(result, "target_covariance"), covariance, rtol=3e-9, atol=3e-11)
    difference = target(theta)-null
    statistic = difference@np.linalg.solve(covariance, difference)
    assert result["wald"].iloc[0].statistic == pytest.approx(statistic, rel=3e-9)
    assert result["wald"].iloc[0].p_value == pytest.approx(chi2.sf(statistic, 3), rel=3e-9, abs=1e-14)


@pytest.fixture(scope="module")
def three_equation_fit():
    frame, _ = fixture(rows=192, equations=3, dependence=True)
    ref = Reference(frame, equations=3)
    return frame, fit_public(frame, ref, "cr0")


@pytest.mark.parametrize("given", [["eq1"], ["eq2"], ["eq1", "eq3"], ["eq3", "eq1"]])
def test_three_equation_schur_covariance_multiple_given_and_cross_deltas(three_equation_fit, given):
    frame, fit = three_equation_fit
    query = frame.iloc[:3].copy()
    for j, name in enumerate(("x", "z", "w")):
        query[name] = [.2+j*.1, .5+j*.1, .8+j*.1]
    ref, theta = Reference(query, equations=3), physical(fit)
    result = oe.nlsur_predict(fit, query, given=given)
    expected, _, _ = ref.predictions(theta, given)
    gradient = complex_jacobian(lambda value: ref.predictions(value, given)[0], theta)
    np.testing.assert_allclose(np.r_[result["means"].estimate, result["residual_covariance"].estimate],
                               expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(matrix(result, "parameter_jacobian"), gradient, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(matrix(result, "target_covariance"), gradient@matrix(fit, "covariance")@gradient.T,
                               rtol=4e-9, atol=4e-11)
    effects = oe.nlsur_margins(fit, query, x=["x", "z", "w"], given=given, weights=[0., 1., 2.])
    def function(value):
        return ref.effects(value, ["x", "z", "w"], given=given, weights=[0., 1., 2.])[0]
    effect_gradient = complex_jacobian(function, theta)
    np.testing.assert_allclose(np.r_[effects["effects"].estimate, effects["averages"].estimate], function(theta),
                               rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(matrix(effects, "parameter_jacobian"), effect_gradient, rtol=3e-9, atol=3e-10)
    np.testing.assert_allclose(matrix(effects, "target_covariance"), effect_gradient@matrix(fit, "covariance")@effect_gradient.T,
                               rtol=4e-8, atol=4e-11)


def test_row_and_equation_order_preserves_aligned_joint_fit(sample, reference):
    base = fit_public(sample, reference, "cr0")
    shuffled = sample.sample(frac=1, random_state=679)
    reordered = oe.nlsur(shuffled, equations=list(reversed(reference.definitions)), start=reference.start,
                         covariance="cr0", cluster="cluster")
    permutation = [0, 1, 2, 3, 6, 5, 4]
    np.testing.assert_allclose(physical(reordered), physical(base)[permutation], rtol=2e-7, atol=2e-8)
    for name in ("information", "bread", "meat", "covariance"):
        np.testing.assert_allclose(matrix(reordered, name), matrix(base, name)[np.ix_(permutation, permutation)],
                                   rtol=2e-6, atol=2e-7)


def test_saved_confidence_level_is_inherited_by_queries(sample, reference, query):
    fit = oe.nlsur(sample, equations=reference.definitions, start=reference.start, level=.9)
    result = oe.nlsur_predict(fit, query, given=["eq1"])
    row = result["means"].iloc[0]
    assert row.ci_low == pytest.approx(row.estimate-norm.ppf(.95)*row.std_error, rel=3e-12)
    assert row.ci_high == pytest.approx(row.estimate+norm.ppf(.95)*row.std_error, rel=3e-12)


def seal(attrs):
    state = attrs["nonlinear_sur_state"]
    state["checksum"] = hashlib.sha256(json.dumps({key: value for key, value in state.items() if key != "checksum"},
                                                 sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return attrs


def test_portable_fit_restoration_runs_no_optimizer(fitted, monkeypatch):
    import openecon.econometrics.systems.nonlinear_sur as native
    _, fit = fitted
    def prohibited(*args, **kwargs):
        raise AssertionError("restoration must not optimize")
    monkeypatch.setattr(native, "_fit", prohibited)
    restored = oe.nlsur_restore(json.loads(json.dumps(fit.attrs, allow_nan=False)))
    assert restored.attrs == fit.attrs
    assert list(restored) == list(fit)
    assert all(restored[key].equals(fit[key]) for key in fit)


@pytest.mark.parametrize("field", ["params", "sigma", "information", "bread", "meat", "covariance", "case_scores", "case_loglikelihood"])
def test_resealed_scientific_payload_tampering_rejected(fitted, field):
    _, fit = fitted
    attrs = copy.deepcopy(fit.attrs)
    value = attrs["nonlinear_sur_state"]["fit"][field]
    if isinstance(value[0], list):
        value[0][0] += .1
    else:
        value[0] += .1
    with pytest.raises(AnalysisError):
        oe.nlsur_restore(seal(attrs))
