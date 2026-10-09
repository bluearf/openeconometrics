"""Independent adaptive-integral multinomial-probit scientific checks."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math

import numpy as np
import pytest
from scipy.special import expit, log_ndtr, logsumexp, ndtr
from scipy.stats import norm
import torch

from openecon.analysis_contracts import AnalysisError
from scripts.mprobit_reference import (
    Reference, bivariate_log_derivatives, bivariate_logcdf,
    choice_effect, choice_log_derivatives, fixture, jacobian,
)


def high_precision_logcdf_third(point, decimal_digits):
    """Positive conditional integral and independently differentiated boundaries.

    Differencing the float64 log Hessian loses small third derivatives when
    large density-ratio terms cancel. Here only analytic CDF boundary formulas
    are differentiated at arbitrary precision, before applying the log chain.
    No native probability, derivative or autograd helper enters this oracle.
    """
    import mpmath as mp

    with mp.workdps(decimal_digits):
        a, b, rho = [mp.mpf(str(value)) for value in point]

        def phi(value):
            return mp.exp(-value*value/2)/mp.sqrt(2*mp.pi)

        def normal_cdf(value):
            return mp.erfc(-value/mp.sqrt(2))/2

        sd = mp.sqrt((1-rho)*(1+rho))
        upper, other = min(a, b), max(a, b)
        width = 1/max(mp.mpf(1), abs(upper), abs((other-rho*upper)/sd))
        pivot = phi(upper)*normal_cdf((other-rho*upper)/sd)

        def positive_integrand(distance):
            value = upper-distance*width
            return phi(value)*normal_cdf((other-rho*value)/sd)/pivot*width

        probability = pivot*mp.quad(positive_integrand, [0, 1, 8, 64, mp.inf])

        def cdf_second(x, y, correlation):
            complement = (1-correlation)*(1+correlation)
            scale = mp.sqrt(complement)
            pa = phi(x)*normal_cdf((y-correlation*x)/scale)
            pb = phi(y)*normal_cdf((x-correlation*y)/scale)
            pr = phi(x)*phi((y-correlation*x)/scale)/scale
            density_rho = (correlation/complement
                           + (x*y*(1+correlation*correlation)
                              - correlation*(x*x+y*y))/complement**2)
            return mp.matrix([
                [-x*pa-correlation*pr, pr, pr*(correlation*y-x)/complement],
                [pr, -y*pb-correlation*pr, pr*(correlation*x-y)/complement],
                [pr*(correlation*y-x)/complement,
                 pr*(correlation*x-y)/complement, pr*density_rho],
            ])

        first = [phi(a)*normal_cdf((b-rho*a)/sd),
                 phi(b)*normal_cdf((a-rho*b)/sd),
                 phi(a)*phi((b-rho*a)/sd)/sd]
        second = cdf_second(a, b, rho)
        third = np.empty((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    def boundary(value):
                        arguments = [a, b, rho]
                        arguments[k] = value
                        return cdf_second(*arguments)[i, j]

                    cdf_third = mp.diff(boundary, [a, b, rho][k])
                    third[i, j, k] = float(
                        cdf_third/probability
                        - (second[i, j]*first[k] + second[i, k]*first[j]
                           + second[j, k]*first[i])/probability**2
                        + 2*first[i]*first[j]*first[k]/probability**3)
        return float(mp.log(probability)), third


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fit_public(frame, vce="oim", *, alternatives=("A", "B", "C"), fixed=None):
    from openecon.econometrics.discrete.mprobit import mprobit
    return mprobit(frame, "chosen", ["x", "z"], case="case", alternative="alternative",
                   alternatives=list(alternatives), available="available", vce=vce,
                   cluster="cluster" if vce == "cr0" else None, covariance=fixed)


def physical(result):
    return np.asarray(result.attrs["mprobit_state"]["fit"]["params"], dtype=float)


@pytest.mark.parametrize("point", [(-1., .5, .2), (1., 1., -.5), (-8., -7., .85),
                                  (-12., 2., -.95), (2., -12., .95),
                                  (-1., -1., .97), (.5, -.2, -.97)])
def test_positive_integral_boundary_derivatives_independently_differenced(point):
    _, gradient, hessian = bivariate_log_derivatives(*point)
    numerical_gradient = jacobian(lambda value: bivariate_logcdf(*value), point)
    numerical_hessian = jacobian(lambda value: bivariate_log_derivatives(*value)[1], point)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=3e-7, atol=2e-8)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=5e-7, atol=3e-7)


@pytest.mark.parametrize("a,b", [(-30., -31.), (-8., 2.), (0., 0.), (6., 6.)])
def test_independent_zero_correlation_product_exact(a, b):
    assert bivariate_logcdf(a, b, 0.) == pytest.approx(float(log_ndtr(a)+log_ndtr(b)), abs=1e-13)


@pytest.mark.parametrize("alternatives,fixed", [(2, None), (3, None),
                                              (3, [[1., .3], [.3, 1.8]])])
def test_independent_full_physical_chain_score_and_hessian(alternatives, fixed):
    frame, truth = fixture(cases=24, alternatives=alternatives)
    reference = Reference(frame, catalogue=tuple(("A", "B", "C")[:alternatives]), fixed=fixed)
    theta = truth[:reference.k]
    expected = reference.moments(theta)
    score = jacobian(lambda value: reference.moments(value)["case_loglikelihood"], theta)
    information = -jacobian(lambda value: reference.moments(value)["case_scores"].sum(0), theta)
    np.testing.assert_allclose(expected["case_scores"], score, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(expected["information"], information, rtol=2e-7, atol=2e-7)


@pytest.fixture(scope="module")
def sample():
    return fixture(cases=256, dependence=True)[0]


@pytest.fixture(scope="module")
def reference(sample):
    return Reference(sample)


@pytest.fixture(scope="module")
def oracle(reference):
    theta, best, starts = reference.fit()
    assert max(abs(reference.moments(theta)["case_scores"].sum(0))) < 2e-5
    assert max(abs(result.fun-best.fun) for result in starts) < 2e-7
    return theta, best


@pytest.fixture(scope="module", params=("oim", "hc0", "cr0"))
def fitted(request, sample):
    return request.param, fit_public(sample, request.param)


def test_every_physical_parameter_and_likelihood_matches_scipy(fitted, oracle):
    _, result = fitted
    theta, best = oracle
    np.testing.assert_allclose(physical(result), theta, rtol=5e-6, atol=2e-6)
    assert result.attrs["mprobit_state"]["fit"]["log_likelihood"] == pytest.approx(-best.fun, abs=2e-7)


@pytest.mark.parametrize("name", ["information", "bread", "meat", "covariance",
                                  "case_scores", "cluster_scores", "case_loglikelihood"])
def test_every_full_physical_matrix_and_score_cell(fitted, reference, name):
    vce, result = fitted
    expected = reference.covariance(physical(result), vce)
    actual = result.attrs["mprobit_state"]["fit"][name]
    if name == "cluster_scores" and vce != "cr0":
        assert actual == []
        return
    np.testing.assert_allclose(actual, expected[name], rtol=5e-7, atol=5e-8)


def test_complete_joint_curvature_includes_beta_covariance_cross_blocks(fitted, reference):
    _, result = fitted
    expected = reference.moments(physical(result))["information"]
    assert np.max(abs(expected[:reference.q, reference.q:])) > 1.
    assert abs(expected[-1, -2]) > .1
    assert np.linalg.eigvalsh(expected).min() > 0
    np.testing.assert_allclose(result.attrs["mprobit_state"]["fit"]["gradient"],
                               reference.moments(physical(result))["case_scores"].sum(0),
                               rtol=1e-5, atol=5e-8)


def test_reported_full_parameter_standard_errors(fitted, reference):
    vce, result = fitted
    expected = reference.covariance(physical(result), vce)["covariance"]
    parameters = result["parameters"]
    np.testing.assert_allclose(parameters["std_error"], np.sqrt(np.diag(expected)),
                               rtol=5e-7, atol=5e-8)
    zcrit = norm.ppf(.975)
    for j, row in parameters.iterrows():
        value, se = physical(result)[j], math.sqrt(expected[j, j])
        if j < reference.q:
            bounds = [value-zcrit*se, value+zcrit*se]
        elif j == reference.q:
            bounds = np.exp([math.log(value)-zcrit*se/value, math.log(value)+zcrit*se/value])
        else:
            bounds = np.tanh([np.arctanh(value)-zcrit*se/(1-value*value),
                              np.arctanh(value)+zcrit*se/(1-value*value)])
        np.testing.assert_allclose([row.ci_lower, row.ci_upper], bounds, rtol=5e-7, atol=5e-8)
        if j != reference.q:
            assert row.z == pytest.approx(value/se, rel=5e-7, abs=5e-8)
            assert row.p_value == pytest.approx(2*norm.sf(abs(value/se)), rel=5e-7, abs=5e-8)


@pytest.mark.parametrize("vce", ["oim", "hc0", "cr0"])
def test_binary_reduction_matches_exact_normal_probabilities_and_curvature(vce):
    frame, _ = fixture(cases=128, alternatives=2, dependence=True)
    result = fit_public(frame, vce, alternatives=("A", "B"))
    reference = Reference(frame, catalogue=("A", "B"))
    expected = reference.covariance(physical(result), vce)
    for name in ("information", "bread", "meat", "covariance", "case_scores"):
        np.testing.assert_allclose(result.attrs["mprobit_state"]["fit"][name], expected[name],
                                   rtol=3e-9, atol=3e-9)
    for case, riskset in enumerate(reference.cases):
        cutoff = (riskset[0, 1:]-riskset[1, 1:])@physical(result)
        assert math.exp(reference.choice(physical(result), case, 0)[0]) == pytest.approx(float(ndtr(cutoff)), abs=1e-14)


def test_fixed_normalized_covariance_uses_global_comparisons_without_base():
    frame, _ = fixture(cases=160)
    fixed = [[1., -.45], [-.45, 1.7]]
    result = fit_public(frame, fixed=fixed)
    reference = Reference(frame, fixed=fixed)
    expected = reference.covariance(physical(result))
    assert len(physical(result)) == 2
    np.testing.assert_allclose(result.attrs["mprobit_state"]["fit"]["information"],
                               expected["information"], rtol=3e-8, atol=3e-8)
    absent_base = next(i for i, case in enumerate(reference.cases) if np.array_equal(case[:, 0], [1, 2]))
    case = reference.cases[absent_base]
    standard_deviation = math.sqrt(fixed[0][0]+fixed[1][1]-2*fixed[0][1])
    probability = ndtr((case[0, 1:]-case[1, 1:])@physical(result)/standard_deviation)
    assert math.exp(reference.choice(physical(result), absent_base, 1)[0]) == pytest.approx(float(probability), abs=1e-14)


def test_independent_probability_normalization_every_available_geometry(reference):
    _, truth = fixture(cases=8)
    for case, riskset in enumerate(reference.cases):
        probabilities = [math.exp(reference.choice(truth, case, int(row[0]))[0]) for row in riskset]
        assert sum(probabilities) == pytest.approx(1., abs=3e-13)


@pytest.mark.parametrize("selected", [0, 1, 2])
def test_independent_common_case_offset_leaves_probability_derivatives_exact(selected):
    theta = np.array([.6, -.3, 1.3, -.4])
    riskset = np.array([[0., 1., 2.], [1., 2., -.5], [2., -1., 3.]])
    before = choice_log_derivatives(riskset, theta, selected)
    shifted = riskset.copy()
    shifted[:, 1:] += np.array([120., -300.])
    after = choice_log_derivatives(shifted, theta, selected)
    for actual, expected in zip(after, before):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("outcome,changed", [(i, j) for i in range(3) for j in range(3)])
@pytest.mark.parametrize("attribute", [0, 1])
def test_independent_boundary_effect_matches_raw_cell_finite_difference(outcome, changed, attribute):
    theta = np.array([.6, -.3, 1.3, -.4])
    riskset = np.array([[0., 1., 2.], [1., 2., .5], [2., 1.5, 3.]])

    def probability(value):
        copied = riskset.copy()
        copied[changed, attribute+1] = value[0]
        return math.exp(choice_log_derivatives(copied, theta, outcome)[0])

    numerical = jacobian(probability, [riskset[changed, attribute+1]])[0]
    assert choice_effect(riskset, theta, outcome, changed, attribute) == pytest.approx(numerical, rel=2e-8, abs=2e-9)


@pytest.fixture(scope="module")
def query():
    frame, _ = fixture(seed=706_699, cases=12, positive=True)
    return frame.drop(columns=["chosen", "cluster"])


def matrix(output, name):
    return output[name].iloc[:, 1:].to_numpy(dtype=float)


def prediction_reference(reference, theta, targets):
    values, gradients, logs, log_gradients, odds, odds_gradients = [], [], [], [], [], []
    structural = []
    for target in targets:
        case = reference.case_ids.index(target["case"])
        alternative = reference.catalogue.index(target["alternative"])
        kind = target.get("kind", "probability")
        riskset = reference.cases[case]
        if alternative not in riskset[:, 0]:
            values.append(0.)
            gradients.append(np.zeros(len(theta)))
            structural.append(True)
            continue
        logp, score, _ = reference.choice(theta, case, alternative)
        other = [reference.choice(theta, case, int(row[0])) for row in riskset if row[0] != alternative]
        logq = float(logsumexp([item[0] for item in other]))
        derivative_q = sum(math.exp(item[0]-logq)*item[1] for item in other)
        values.append(logp if kind == "log_probability" else math.exp(logp))
        gradients.append(score if kind == "log_probability" else math.exp(logp)*score)
        logs.append(logp)
        log_gradients.append(score)
        odds.append(logp-logq)
        odds_gradients.append(score-derivative_q)
        structural.append(False)
    return np.asarray(values), np.stack(gradients), np.stack(gradients+log_gradients+odds_gradients), logs, odds, structural


def test_prediction_all_requested_and_transformed_joint_delta_cells(fitted, query):
    from openecon.econometrics.discrete.mprobit_postestimation import mprobit_predict
    _, result = fitted
    targets = [dict(case=0, alternative="A"), dict(case=0, alternative="C", kind="log_probability"),
               dict(case=8, alternative="B"), dict(case=8, alternative="C"),
               dict(case=8, alternative="A")]
    output = mprobit_predict(result, query, targets=targets)
    ref = Reference(query)
    values, J, joint_J, logs, odds, structural = prediction_reference(ref, physical(result), targets)
    V = np.asarray(result.attrs["mprobit_state"]["fit"]["covariance"])
    np.testing.assert_allclose(output["predictions"].estimate, values, rtol=2e-9, atol=3e-11)
    np.testing.assert_allclose(matrix(output, "jacobian"), J, rtol=5e-7, atol=5e-9)
    np.testing.assert_allclose(matrix(output, "joint_jacobian"), joint_J, rtol=5e-7, atol=5e-9)
    np.testing.assert_allclose(matrix(output, "covariance"), J@V@J.T, rtol=7e-7, atol=5e-10)
    np.testing.assert_allclose(matrix(output, "joint_covariance"), joint_J@V@joint_J.T, rtol=7e-7, atol=5e-10)
    np.testing.assert_allclose(matrix(output, "coefficient_covariance"), V, rtol=1e-14, atol=1e-14)
    for i, row in output["predictions"].iterrows():
        assert row.standard_error == pytest.approx(math.sqrt((J@V@J.T)[i, i]), rel=5e-7, abs=1e-10)
        if structural[i]:
            assert row.ci_lower == row.ci_upper == 0
            continue
        if row.kind == "probability":
            selected = i-sum(structural[:i])
            index = len(J)+len(logs)+selected
            odds_se = math.sqrt((joint_J@V@joint_J.T)[index, index])
            bounds = expit([odds[selected]-norm.ppf(.975)*odds_se,
                            odds[selected]+norm.ppf(.975)*odds_se])
            np.testing.assert_allclose([row.ci_lower, row.ci_upper], bounds, rtol=5e-7, atol=5e-10)


@pytest.mark.parametrize("elasticity", [False, True])
def test_every_own_cross_mixed_derivative_and_weighted_average_delta(fitted, query, elasticity):
    from openecon.econometrics.discrete.mprobit_postestimation import mprobit_margins
    _, result = fitted
    targets = [dict(outcome=i, changed=j, attribute="x") for i in ("A", "B", "C")
               for j in ("A", "B", "C")]+[dict(outcome="C", changed="A", attribute="z")]
    weights = {case: float(case+1) for case in query.case.unique()}
    output = mprobit_margins(result, query, targets=targets, elasticity=elasticity,
                            average=True, weights=weights, max_work=1_000_000_000)
    ref = Reference(query)
    theta = physical(result)
    values, J, averages, average_J = [], [], [], []
    for target in targets:
        outcome, changed = [ref.catalogue.index(target[key]) for key in ("outcome", "changed")]
        eligible = [case for case, riskset in enumerate(ref.cases)
                    if outcome in riskset[:, 0] and changed in riskset[:, 0]]
        selected_weights = np.array([weights[ref.case_ids[case]] for case in eligible])
        normalized = selected_weights/selected_weights.sum()
        block_values, block_J = [], []
        for case in eligible:
            def function(value, case=case):
                return ref.effect(value, case, outcome, changed,
                                  target["attribute"], elasticity=elasticity)
            block_values.append(function(theta))
            block_J.append(jacobian(function, theta))
        values.extend(block_values)
        J.extend(block_J)
        averages.append(normalized@np.array(block_values))
        average_J.append(normalized@np.stack(block_J))
    J = np.stack(J+average_J)
    expected_values = np.r_[values, averages]
    V = np.asarray(result.attrs["mprobit_state"]["fit"]["covariance"])
    table = output["elasticities" if elasticity else "effects"]
    np.testing.assert_allclose(np.r_[table.estimate, output["averages"].estimate], expected_values,
                               rtol=3e-7, atol=3e-9)
    np.testing.assert_allclose(matrix(output, "jacobian"), J, rtol=2e-6, atol=3e-8)
    np.testing.assert_allclose(matrix(output, "covariance"), J@V@J.T, rtol=3e-6, atol=3e-9)
    np.testing.assert_allclose(np.r_[table.standard_error, output["averages"].standard_error],
                               np.sqrt(np.diag(J@V@J.T)), rtol=2e-6, atol=3e-9)
    for margin, subset in table.groupby("margin", sort=False):
        assert subset.normalized_weight.sum() == pytest.approx(1., abs=2e-15)
        support = output["support"].loc[output["support"].margin == margin]
        assert len(support) == len(ref.cases)
        assert support.included.sum() == len(subset)
        weights_subset = subset.case.map(weights).to_numpy(dtype=float)
        np.testing.assert_allclose(subset.normalized_weight, weights_subset/weights_subset.sum(), atol=2e-15)


@pytest.mark.parametrize("point", [(-1., .5, .2), (1., 1., -.5), (-8., -7., .85),
                                  (-12., 2., -.95), (2., -12., .95),
                                  (-1., -1., .97), (.5, -.2, -.97), (8., 8., .97)])
def test_native_logcdf_through_third_backward_against_independent_integral(point):
    from openecon.econometrics.discrete.mprobit import _log_cdf
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)

    def function(theta):
        return _log_cdf(*theta.unbind())

    logp, gradient, hessian = bivariate_log_derivatives(*point)
    actual_gradient = torch.autograd.functional.jacobian(function, value)
    actual_hessian = torch.autograd.functional.hessian(function, value)
    actual_third = torch.autograd.functional.jacobian(
        lambda theta: torch.autograd.functional.hessian(function, theta, create_graph=True), value)
    reference_logp, expected_third = high_precision_logcdf_third(point, 80)
    convergence_logp, convergence_third = high_precision_logcdf_third(point, 50)
    assert reference_logp == pytest.approx(convergence_logp, rel=5e-14, abs=5e-13)
    np.testing.assert_allclose(expected_third, convergence_third, rtol=5e-13, atol=1e-12)
    assert reference_logp == pytest.approx(logp, rel=3e-14, abs=5e-11)
    assert float(function(value).detach()) == pytest.approx(logp, rel=3e-8, abs=5e-11)
    np.testing.assert_allclose(actual_gradient.detach(), gradient, rtol=3e-7, atol=3e-9)
    np.testing.assert_allclose(actual_hessian.detach(), hessian, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(actual_third.detach(), expected_third, rtol=8e-6, atol=2e-5)


@pytest.mark.parametrize("point", [(12., 10., .94), (-30., -31., -.8)])
def test_extreme_tail_score_hessian_retained_with_relative_accuracy(point):
    from openecon.econometrics.discrete.mprobit import _log_cdf
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)

    def function(theta):
        return _log_cdf(*theta.unbind())

    logp, score, curvature = bivariate_log_derivatives(*point)
    actual = float(function(value).detach())
    assert math.isfinite(actual)
    assert actual == pytest.approx(logp, rel=3e-8, abs=5e-12)
    np.testing.assert_allclose(torch.autograd.functional.jacobian(function, value).detach(),
                               score, rtol=5e-7, atol=1e-300)
    np.testing.assert_allclose(torch.autograd.functional.hessian(function, value).detach(),
                               curvature, rtol=3e-6, atol=1e-300)


@pytest.mark.parametrize("vce", ["oim", "hc0", "cr0"])
def test_free_global_reference_and_scale_change_transforms_full_covariance(vce):
    frame, _ = fixture(cases=128, dependence=True)
    base = fit_public(frame, vce)
    changed = fit_public(frame, vce, alternatives=("B", "C", "A"))

    def transform(theta):
        sd, rho = theta[-2:]
        scale = math.sqrt(1+sd*sd-2*sd*rho)
        return np.r_[theta[:2]/scale, 1/scale, (1-sd*rho)/scale]

    expected = transform(physical(base))
    derivative = jacobian(transform, physical(base))
    np.testing.assert_allclose(physical(changed), expected, rtol=3e-6, atol=3e-7)
    old = base.attrs["mprobit_state"]["fit"]
    new = changed.attrs["mprobit_state"]["fit"]
    expected_covariance = derivative@np.asarray(old["covariance"])@derivative.T
    np.testing.assert_allclose(new["covariance"], expected_covariance, rtol=8e-6, atol=8e-8)
    inverse = np.linalg.inv(derivative)
    np.testing.assert_allclose(new["information"], inverse.T@np.asarray(old["information"])@inverse,
                               rtol=8e-6, atol=8e-6)
    assert new["log_likelihood"] == pytest.approx(old["log_likelihood"], abs=1e-8)
    old_probabilities = base["probabilities"].set_index(["case", "alternative"]).sort_index()
    new_probabilities = changed["probabilities"].set_index(["case", "alternative"]).sort_index()
    np.testing.assert_allclose(new_probabilities.probability, old_probabilities.probability,
                               rtol=3e-7, atol=3e-8)


def seal(attrs):
    state = attrs["mprobit_state"]
    state["checksum"] = hashlib.sha256(json.dumps(
        {key: value for key, value in state.items() if key != "checksum"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return attrs


def test_sorted_json_restoration_has_identical_attrs_tables_and_latex_without_optimizer(fitted, monkeypatch):
    native = importlib.import_module("openecon.econometrics.discrete.mprobit")
    _, result = fitted

    def prohibited(*args, **kwargs):
        raise AssertionError("Scientific restoration must not optimize.")

    monkeypatch.setattr(native, "_fit", prohibited)
    payload = json.dumps(result.attrs, sort_keys=True, allow_nan=False)
    restored = native.mprobit_restore(json.loads(payload))
    assert json.dumps(restored.attrs, sort_keys=True, allow_nan=False) == payload
    assert list(restored) == list(result)
    for key in result:
        assert restored[key].equals(result[key])
        assert restored[key].to_latex(index=True) == result[key].to_latex(index=True)


@pytest.mark.parametrize("field", ["params", "information", "bread", "meat", "covariance",
                                  "case_scores", "cluster_scores", "case_loglikelihood", "gradient"])
def test_resealed_full_scientific_payload_tampering_fails(fitted, field):
    from openecon.econometrics.discrete.mprobit import mprobit_restore
    _, result = fitted
    attrs = copy.deepcopy(result.attrs)
    value = attrs["mprobit_state"]["fit"][field]
    if not value:
        value.append([.1]*len(physical(result)))
    elif isinstance(value[0], list):
        value[0][0] += .1
    else:
        value[0] += .1
    with pytest.raises(AnalysisError):
        mprobit_restore(seal(attrs))
