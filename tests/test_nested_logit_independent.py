"""Independent physical-coordinate nested-choice likelihood/inference oracles.

NumPy/SciPy are development tools only. No production private mathematical
helper is called by the independent reference.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from scripts.nested_logit_reference import Reference, jacobian

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_choices(**kwargs):
    namespace = runpy.run_path(str(ROOT/"docs/examples/nested_logit.py"),
                              init_globals={"NESTED_LOGIT_LIBRARY_ONLY": True})
    return namespace["synthetic_choices"](**kwargs)


def fit_public(frame, vce="oim", **kwargs):
    return oe.nlogit(frame, "chosen", ["cost", "quality"], case="choice",
                     alternative="alternative", nest="nest", available="available", vce=vce,
                     cluster="respondent" if vce == "cr0" else None, **kwargs)


def physical(result):
    return np.asarray(result.attrs["nested_logit_state"]["fit"]["params"], dtype=float)


def matrix(result, key):
    return result[key].iloc[:, 1:].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def fixture():
    return synthetic_choices()


@pytest.fixture(scope="module")
def reference(fixture):
    return Reference(fixture)


@pytest.fixture(scope="module")
def oracle(reference):
    best, starts = reference.fit()
    assert np.max(np.abs(reference.objective(best.x)[1])) < 2e-6
    assert np.all(best.x[2:] > .01) and np.all(best.x[2:] < .99)
    assert np.max([abs(value.fun-best.fun) for value in starts]) < 1e-7
    return best


@pytest.fixture(scope="module", params=("oim", "hc0", "cr0"))
def fitted(request, fixture):
    return request.param, fit_public(fixture, request.param)


def test_all_physical_parameters_match_multistart_scipy(fitted, oracle):
    _, result = fitted
    np.testing.assert_allclose(physical(result), oracle.x, rtol=2e-5, atol=2e-6)
    assert abs(result.attrs["nested_logit_state"]["fit"]["log_likelihood"]+oracle.fun) < 1e-8


@pytest.mark.parametrize("key", ("information", "bread", "meat", "covariance", "case_scores"))
def test_every_full_likelihood_and_inference_cell(fitted, reference, key):
    vce, result = fitted
    expected = reference.covariance(physical(result), vce)[key]
    actual = np.asarray(result.attrs["nested_logit_state"]["fit"][key])
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-7)


def test_cluster_scores_aggregate_original_choice_cases(fitted, reference):
    vce, result = fitted
    actual = result.attrs["nested_logit_state"]["fit"]["cluster_scores"]
    expected = reference.covariance(physical(result), vce)["cluster_scores"] if vce == "cr0" else np.empty((0,4))
    if vce == "cr0":
        np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=2e-9)
    else:
        assert actual == []


def test_complete_case_likelihood_available_probabilities(fitted, reference):
    _, result = fitted
    actual = result.attrs["nested_logit_state"]["fit"]
    expected = reference.moments(physical(result))
    rows = actual["probability_rows"]
    np.testing.assert_allclose(actual["case_loglikelihood"], expected["case_loglikelihood"], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual["probabilities"], expected["probability"][rows], rtol=3e-10, atol=2e-12)
    np.testing.assert_allclose(actual["log_probabilities"], expected["log_probability"][rows], rtol=3e-10, atol=2e-10)
    for group in reference.groups:
        assert expected["probability"][group].sum() == pytest.approx(1, abs=1e-14)


def test_joint_inference_standard_errors_intervals_pvalues(fitted):
    _, result = fitted
    frame = result["parameters"]
    estimate, se = physical(result), np.sqrt(np.diag(matrix(result, "covariance")))
    np.testing.assert_allclose(frame["estimate"], estimate, rtol=0, atol=0)
    np.testing.assert_allclose(frame["std_error"], se, rtol=2e-12, atol=0)
    # Dissimilarity intervals use the declared transformed interval policy;
    # beta coefficients still use ordinary physical-scale normal inference.
    for j in range(2):
        z = estimate[j]/se[j]
        assert frame.iloc[j].z == pytest.approx(z, rel=2e-12)
        assert frame.iloc[j].p_value == pytest.approx(2*norm.sf(abs(z)), rel=2e-12)
        assert frame.iloc[j].ci_lower == pytest.approx(estimate[j]-norm.ppf(.975)*se[j], rel=2e-12)
        assert frame.iloc[j].ci_upper == pytest.approx(estimate[j]+norm.ppf(.975)*se[j], rel=2e-12)


def test_score_is_independent_five_point_likelihood_derivative(reference, oracle):
    expected = jacobian(lambda params: reference.moments(params)["case_loglikelihood"], oracle.x)
    actual = reference.moments(oracle.x)["case_scores"]
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-8)
    information = reference.information(oracle.x)
    np.testing.assert_allclose(information, information.T, rtol=2e-6, atol=2e-7)
    assert np.linalg.eigvalsh(information)[0] > 0


def test_lambda_one_equals_independent_mnl(fixture):
    result = fit_public(fixture, fixed_dissimilarity={"A": 1., "B": 1.})
    params = physical(result)
    reference = Reference(fixture, fixed={"A": 1., "B": 1.})
    oracle, _ = reference.fit()
    np.testing.assert_allclose(params, oracle.x, rtol=1e-6, atol=1e-7)
    expected = reference.moments(params)
    for group in reference.groups:
        rows = group[reference.available[group]]
        utility = reference.x[rows] @ params
        np.testing.assert_allclose(expected["probability"][rows], np.exp(utility-logsumexp(utility)), rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(result.attrs["nested_logit_state"]["fit"]["probabilities"], expected["probability"][reference.available], rtol=3e-10, atol=1e-12)


def test_shared_case_attribute_offsets_preserve_joint_fit(fixture, oracle):
    shifted = fixture.copy()
    for name, multiplier in (("cost", 2**20), ("quality", -2**18)):
        shifted[name] += shifted.choice.map(lambda value: multiplier*(1+value%3))
    result = fit_public(shifted)
    np.testing.assert_allclose(physical(result), oracle.x, rtol=3e-5, atol=3e-6)


def test_row_order_preserves_parameters_after_lambda_catalogue_alignment(fixture, oracle):
    shuffled = fixture.sample(frac=1, random_state=609).reset_index(drop=True)
    result = fit_public(shuffled)
    state = result.attrs["nested_logit_state"]
    labels = state["catalogue"]["nests"]
    estimate = np.r_[state["fit"]["beta"], [state["fit"]["lambdas"][labels.index(label)] for label in ("A", "B")]]
    np.testing.assert_allclose(estimate, oracle.x, rtol=2e-5, atol=2e-6)


@pytest.fixture(scope="module")
def query():
    return synthetic_choices(seed=609, cases=3, offset=1000).drop(columns=["chosen", "respondent"])


@pytest.fixture(scope="module")
def components():
    return [dict(case=1000, alternative="A"), dict(case=1000, alternative="B", kind="conditional"),
            dict(case=1000, alternative="D", kind="log_probability"),
            dict(case=1001, alternative="A", kind="log_conditional"),
            dict(case=1001, nest="A", kind="nest_probability"),
            dict(case=1002, nest="B", kind="log_nest_probability")]


def test_all_six_prediction_scales_complete_joint_covariance(fitted, query, components):
    _, fit = fitted
    result = oe.nlogit_predict(fit, data=query, targets=components)
    reference = Reference(query)
    params = physical(fit)
    expected = np.asarray([reference.target(params, target) for target in components])
    gradient = jacobian(lambda value: [reference.target(value, target) for target in components], params)
    np.testing.assert_allclose(result["predictions"].estimate, expected, rtol=3e-9, atol=2e-11)
    np.testing.assert_allclose(matrix(result, "jacobian"), gradient, rtol=3e-6, atol=2e-8)
    np.testing.assert_allclose(matrix(result, "covariance"), gradient @ matrix(fit, "covariance") @ gradient.T, rtol=4e-6, atol=2e-9)


@pytest.mark.parametrize("kind", ("effect", "elasticity"))
def test_own_within_cross_nest_and_unbalanced_weighted_ames(fitted, query, kind):
    _, fit = fitted
    targets = [dict(outcome="A", changed="A", attribute="cost"), dict(outcome="B", changed="A", attribute="cost"),
               dict(outcome="D", changed="A", attribute="cost"), dict(outcome="C", changed="B", attribute="quality")]
    weights = {1000: 1., 1001: 2., 1002: 3.}
    result = oe.nlogit_margins(fit, data=query, targets=targets, case_weights=weights, kind=kind)
    reference = Reference(query)
    params = physical(fit)
    def estimates(value):
        return np.asarray([reference.ame(value, target, weights, kind) for target in targets])
    gradient = jacobian(estimates, params)
    np.testing.assert_allclose(result["margins"].estimate, estimates(params), rtol=4e-9, atol=2e-10)
    np.testing.assert_allclose(matrix(result, "jacobian"), gradient, rtol=4e-6, atol=3e-8)
    np.testing.assert_allclose(matrix(result, "covariance"), gradient @ matrix(fit, "covariance") @ gradient.T, rtol=7e-6, atol=3e-9)
    for row in result["per_case"].itertuples():
        expected = reference.effect(params, row.case, row.outcome, row.changed, row.attribute, kind)
        assert row.estimate == pytest.approx(expected, rel=5e-9, abs=2e-11)
    assert len(result["support"]) == len(query.choice.unique())*len(targets)


@pytest.mark.parametrize("outcome,changed", (("A", "A"), ("B", "A"), ("D", "A")))
def test_analytical_effect_against_independent_raw_attribute_perturbation(fitted, query, outcome, changed):
    _, fit = fitted
    params, reference = physical(fit), Reference(query)
    row = query.index[(query.choice == 1001) & (query.alternative == changed)][0]
    step = 1e-4
    values = []
    for multiple in (-2, -1, 1, 2):
        changed_data = query.copy()
        changed_data.loc[row, "cost"] += multiple*step
        values.append(Reference(changed_data).target(params, dict(case=1001, alternative=outcome)))
    numeric = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
    assert reference.effect(params, 1001, outcome, changed, "cost") == pytest.approx(numeric, rel=2e-7, abs=2e-9)


def seal(attrs):
    state = attrs["nested_logit_state"]
    state["checksum"] = hashlib.sha256(json.dumps({key: value for key, value in state.items() if key != "checksum"},
                                                sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    return attrs


def test_restoration_and_queries_cannot_call_optimizer(fitted, query, components, monkeypatch):
    _, fit = fitted
    import openecon.econometrics.discrete.nested_logit as native
    def forbidden(*args, **kwargs):
        raise AssertionError("restoration attempted optimizer")
    monkeypatch.setattr(native, "_fit", forbidden)
    attrs = json.loads(json.dumps(fit.attrs, allow_nan=False))
    restored = oe.nlogit_restore(attrs)
    assert restored.attrs == fit.attrs
    assert all(restored[key].equals(fit[key]) for key in fit)
    oe.nlogit_predict(attrs, data=query, targets=components)
    oe.nlogit_margins(attrs, data=query, targets=[dict(outcome="A", changed="B", attribute="cost")])


@pytest.mark.parametrize("field", ("information", "bread", "meat", "covariance", "case_scores"))
def test_resealed_numerical_state_tampering_requires_replay(fitted, field):
    _, fit = fitted
    attrs = copy.deepcopy(fit.attrs)
    attrs["nested_logit_state"]["fit"][field][0][0] += .01
    with pytest.raises(AnalysisError):
        oe.nlogit_restore(seal(attrs))


def test_all_six_complete_fitted_probability_components(fitted, reference):
    _, result = fitted
    expected = reference.moments(physical(result))
    fit = result.attrs["nested_logit_state"]["fit"]
    physical_cases = {}
    for ci, group in enumerate(reference.groups):
        physical_cases.update({row: ci for row in group})
    for row in fit["probability_components"]:
        physical_row, *actual = row
        ci, ni = physical_cases[physical_row], reference.nest_indices[physical_row]
        values = [expected["probability"][physical_row], expected["log_probability"][physical_row],
                  expected["conditional"][physical_row], expected["log_conditional"][physical_row],
                  expected["nest_probability"][ci, ni], expected["log_nest_probability"][ci, ni]]
        np.testing.assert_allclose(actual, values, rtol=2e-10, atol=2e-11)


def test_dissimilarity_physical_se_transformed_intervals_have_no_boundary_null_test(fitted):
    from scipy.special import expit, logit
    _, result = fitted
    frame = result["dissimilarities"]
    params = physical(result)
    se = np.sqrt(np.diag(matrix(result, "covariance")))
    for m in range(2):
        row = frame.iloc[m]
        transformed_se = se[2+m]/(params[2+m]*(1-params[2+m]))
        expected = expit(logit(params[2+m])+np.array([-1., 1.])*norm.ppf(.975)*transformed_se)
        np.testing.assert_allclose([row.ci_lower, row.ci_upper], expected, rtol=3e-12, atol=2e-13)
        assert row.std_error == pytest.approx(se[2+m], rel=2e-12)
        assert result["parameters"].iloc[2+m].p_value is None or np.isnan(result["parameters"].iloc[2+m].p_value)


def test_all_prediction_companion_jacobians_and_complete_joint_covariance(fitted, query, components):
    _, fit = fitted
    result = oe.nlogit_predict(fit, data=query, targets=components)
    reference, params = Reference(query), physical(fit)
    def values(value):
        requested = [reference.target(value, target) for target in components]
        probability_indices = [i for i, target in enumerate(components) if not target.get("kind", "probability").startswith("log_")]
        log_values = [np.log(requested[i]) for i in probability_indices]
        log_odds = [np.log(requested[i])-np.log1p(-requested[i]) for i in probability_indices]
        return requested+log_values+log_odds
    gradient = jacobian(values, params)
    np.testing.assert_allclose(matrix(result, "joint_jacobian"), gradient, rtol=3e-6, atol=3e-8)
    expected = gradient @ matrix(fit, "covariance") @ gradient.T
    np.testing.assert_allclose(matrix(result, "joint_covariance"), expected, rtol=5e-6, atol=3e-9)
    actual = matrix(result, "joint_covariance")
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-14)
    assert np.linalg.eigvalsh(actual)[0] >= -1e-12


@pytest.mark.parametrize("kind", ("effect", "elasticity"))
def test_missing_pair_rows_use_explicit_conditional_weight_support(fitted, query, kind):
    _, fit = fitted
    query = query.loc[~((query.choice == 1001) & (query.alternative == "D"))].reset_index(drop=True)
    weights = {1000: 2., 1001: 11., 1002: 3.}
    targets = [dict(outcome="D", changed="A", attribute="cost"), dict(outcome="C", changed="B", attribute="quality")]
    result = oe.nlogit_margins(fit, data=query, targets=targets, case_weights=weights, kind=kind)
    reference, params = Reference(query), physical(fit)
    expected = np.array([reference.ame(params, target, weights, kind) for target in targets])
    gradient = jacobian(lambda value: [reference.ame(value, target, weights, kind) for target in targets], params)
    np.testing.assert_allclose(result["margins"].estimate, expected, rtol=3e-9, atol=2e-10)
    np.testing.assert_allclose(matrix(result, "covariance"), gradient @ matrix(fit, "covariance") @ gradient.T, rtol=5e-6, atol=2e-9)
    for target in result["margins"].target:
        selected = result["per_case"].loc[result["per_case"].target == target]
        assert selected.normalized_support_weight.sum() == pytest.approx(1., abs=1e-14)
    assert len(result["support"]) == 6


def test_empty_query_nest_and_only_active_nest_are_structural(fitted, query):
    _, fit = fitted
    query = query.loc[query.choice == 1001].reset_index(drop=True)
    query["available"] = query.alternative.isin(["A", "B"])
    result = oe.nlogit_predict(fit, data=query, targets=[dict(case=1001, nest="A", kind="nest_probability"),
                                                      dict(case=1001, nest="A", kind="log_nest_probability"),
                                                      dict(case=1001, nest="B", kind="nest_probability")])
    np.testing.assert_array_equal(result["predictions"].estimate, [1., 0., 0.])
    np.testing.assert_array_equal(matrix(result, "covariance"), np.zeros((3,3)))
    np.testing.assert_array_equal(matrix(result, "jacobian"), np.zeros((3,4)))


def test_singleton_available_nests_have_unit_conditional_probabilities(fitted, query):
    _, fit = fitted
    query = query.loc[query.choice == 1001].reset_index(drop=True)
    query["available"] = query.alternative.isin(["A", "D"])
    result = oe.nlogit_predict(fit, data=query, targets=[dict(case=1001, alternative="A", kind="conditional"),
                                                      dict(case=1001, alternative="D", kind="log_conditional")])
    np.testing.assert_array_equal(result["predictions"].estimate, [1., 0.])
    np.testing.assert_array_equal(matrix(result, "covariance"), np.zeros((2,2)))
    np.testing.assert_array_equal(matrix(result, "jacobian"), np.zeros((2,4)))


def single_active_nest_fixture(seed=610, cases=240):
    """A fixed-A scale anchor identifies beta and free B from separate cases."""
    generator = np.random.default_rng(seed)
    frame = synthetic_choices(seed=seed, cases=cases)
    active = np.where(frame.choice.to_numpy() % 2 == 0, "A", "B")
    frame["available"] = (frame.nest.to_numpy() == active) & frame.available.to_numpy()
    frame["chosen"] = 0
    for _, rows in frame.groupby("choice", sort=False):
        frame.loc[rows.index[rows.available][0], "chosen"] = 1
    reference = Reference(frame, fixed={"A": 1.})
    probabilities = reference.moments([-.65, .48, .62])["probability"]
    frame["chosen"] = 0
    for rows in reference.groups:
        selected = min(int(np.searchsorted(probabilities[rows].cumsum(), generator.random(), side="right")), len(rows)-1)
        frame.loc[rows[selected], "chosen"] = 1
    return frame


@pytest.mark.parametrize("vce", ("oim", "hc0", "cr0"))
def test_fixed_anchor_single_active_nests_match_independent_physical_ml_and_full_inference(vce):
    frame = single_active_nest_fixture()
    reference = Reference(frame, fixed={"A": 1.})
    oracle, starts = reference.fit()
    assert .01 < oracle.x[-1] < .99
    assert np.max([abs(value.fun-oracle.fun) for value in starts]) < 1e-7
    assert np.max(abs(reference.objective(oracle.x)[1])) < 2e-6
    result = fit_public(frame, vce, fixed_dissimilarity={"A": 1.})
    np.testing.assert_allclose(physical(result), oracle.x, rtol=2e-5, atol=2e-6)
    expected = reference.covariance(physical(result), vce)
    for key in ("information", "bread", "meat", "covariance", "case_scores"):
        np.testing.assert_allclose(result.attrs["nested_logit_state"]["fit"][key], expected[key], rtol=3e-6, atol=3e-7)
    assert np.linalg.eigvalsh(expected["information"])[0] > 0
    restored = oe.nlogit_restore(json.loads(json.dumps(result.attrs, allow_nan=False)))
    assert restored.attrs == result.attrs


def test_all_free_single_active_nests_have_exact_scale_confounding_and_no_certified_fit():
    frame = single_active_nest_fixture()
    reference = Reference(frame)
    params = np.array([-.65, .48, .5, .62])
    likelihood = reference.moments(params)["case_loglikelihood"]
    for factor in (.5, .8, 1.2):
        np.testing.assert_allclose(reference.moments(params*factor)["case_loglikelihood"], likelihood, rtol=0, atol=1e-14)
    assert reference.moments(params)["case_scores"].sum(axis=0) @ params == pytest.approx(0., abs=1e-12)
    with pytest.raises(AnalysisError):
        fit_public(frame)
