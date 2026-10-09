"""Closed binary reductions and complete multinomial-probit query contracts."""
from __future__ import annotations

import copy
import json
import math
import random

import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete import mprobit_postestimation as post


@pytest.fixture(scope="module")
def fit():
    from openecon.econometrics.discrete.mprobit import mprobit

    generator = random.Random(706)
    data = []
    for case in range(128):
        x = [generator.uniform(1., 5.) for _ in range(3)]
        z1, z2 = generator.gauss(0., 1.), generator.gauss(0., 1.)
        errors = [0., z1, 1.2*(.15*z1+math.sqrt(1-.15**2)*z2)]
        chosen = max(range(3), key=lambda j: .65*x[j]+errors[j])
        for j, alternative in enumerate(("base", "second", "third")):
            data.append(dict(case=case, alternative=alternative, x=x[j], chosen=int(chosen == j),
                             available=True, cluster=case//4))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return mprobit(pd.DataFrame(data), "chosen", ["x"], case="case", alternative="alternative",
                       alternatives=["base", "second", "third"],
                       available="available", cluster="cluster", vce="cr0")
    finally:
        torch.set_num_threads(previous)


def query(pair=("second", "third")):
    return pd.DataFrame([dict(case=label, alternative=alternative, x=value, available=True)
                         for label, numbers in ((1, [2., 3.]), ("1", [2.5, 4.]))
                         for alternative, value in zip(pair, numbers)])


def physical(fit):
    state = fit.attrs["mprobit_state"]
    return torch.tensor(state["fit"]["params"], dtype=torch.float64), state


def array(result, name):
    return torch.tensor(result[name].iloc[:, 1:].values.tolist(), dtype=torch.float64)


def numerical_jacobian(function, theta):
    columns = []
    for j in range(len(theta)):
        step = 2e-5*max(1., abs(float(theta[j])))
        d = torch.zeros_like(theta)
        d[j] = step
        columns.append((-function(theta+2*d)+8*function(theta+d)
                        -8*function(theta-d)+function(theta-2*d))/(12*step))
    return torch.stack(columns, 1)


def binary_values(theta, *, effect=False, elasticity=False):
    beta, sd, rho = theta
    scale = torch.sqrt(1+sd*sd-2*sd*rho)
    values = []
    for first, second in ((2., 3.), (2.5, 4.)):
        z = beta*(second-first)/scale
        probability = .5*torch.erfc(-z/math.sqrt(2))
        if effect or elasticity:
            derivative = beta/scale*torch.exp(-.5*z*z)/math.sqrt(2*math.pi)
            values.append(second*derivative/probability if elasticity else derivative)
        else:
            values.extend([1-probability, probability])
    return torch.stack(values)


def test_pair_reduction_full_probability_jacobian_and_covariance(fit):
    theta, state = physical(fit)
    result = post.mprobit_predict(fit, query())
    expected = binary_values(theta)
    J = numerical_jacobian(binary_values, theta)
    covariance = torch.tensor(state["fit"]["covariance"], dtype=torch.float64)
    assert torch.allclose(torch.tensor(result["predictions"].estimate.values, dtype=torch.float64), expected,
                          rtol=2e-11, atol=2e-13)
    assert torch.allclose(array(result, "jacobian"), J, rtol=2e-8, atol=2e-10)
    assert torch.allclose(array(result, "covariance"), J@covariance@J.T, rtol=3e-8, atol=2e-11)
    assert bool((J[:, 1:].abs().sum(0) > 0).all())
    assert result["predictions"].case.tolist() == [1, 1, "1", "1"]
    assert result.attrs["settings"]["no_optimizer"] is True


@pytest.mark.parametrize("elasticity", [False, True])
def test_mixed_derivative_and_fixed_weight_average_full_joint(fit, elasticity):
    theta, state = physical(fit)
    target = [dict(outcome="third", changed="third", attribute="x")]
    result = post.mprobit_margins(fit, query(), targets=target, elasticity=elasticity,
                                 average=True, weights=[1., 3.])

    def values(t):
        v = binary_values(t, effect=not elasticity, elasticity=elasticity)
        return torch.cat((v, ((v[0]+3*v[1])/4).reshape(1)))

    expected, J = values(theta), numerical_jacobian(values, theta)
    key = "elasticities" if elasticity else "effects"
    actual = torch.tensor([*result[key].estimate, *result["averages"].estimate], dtype=torch.float64)
    assert torch.allclose(actual, expected, rtol=2e-11, atol=2e-13)
    assert torch.allclose(array(result, "jacobian"), J, rtol=3e-8, atol=3e-10)
    covariance = torch.tensor(state["fit"]["covariance"], dtype=torch.float64)
    assert torch.allclose(array(result, "covariance"), J@covariance@J.T, rtol=4e-8, atol=2e-11)
    assert result["averages"].weight_denominator.tolist() == [4.]
    assert result.attrs["settings"]["joint_estimand_order"] == ["c1", "c2", "m1"]
    assert abs(float(array(result, "covariance")[0, 2])) > 0


def test_own_cross_adding_up_and_common_utility_shift(fit):
    data = query()
    targets = [dict(outcome=o, changed=c, attribute="x")
               for o in ("second", "third") for c in ("second", "third")]
    result = post.mprobit_margins(fit, data, targets=targets)
    values = torch.tensor(result["effects"].estimate.values, dtype=torch.float64).reshape(2, 2, 2)
    J = array(result, "jacobian").reshape(2, 2, 2, -1)
    assert torch.allclose(values.sum(0), torch.zeros((2, 2), dtype=torch.float64), atol=2e-14, rtol=0)
    assert torch.allclose(values.sum(1), torch.zeros((2, 2), dtype=torch.float64), atol=2e-14, rtol=0)
    assert torch.allclose(J.sum(0), torch.zeros_like(J.sum(0)), atol=2e-12, rtol=0)
    shifted = data.copy()
    shifted["x"] += 1e4
    a, b = post.mprobit_predict(fit, data), post.mprobit_predict(fit, shifted)
    assert torch.allclose(array(a, "jacobian"), array(b, "jacobian"), rtol=2e-11, atol=2e-13)
    assert a["predictions"].estimate.tolist() == b["predictions"].estimate.tolist()


def test_availability_support_no_truncation_and_structural_logs(fit):
    data = pd.concat((query(), pd.DataFrame([dict(case="q", alternative="base", x=2., available=True),
                                          dict(case="q", alternative="third", x=3., available=False)])),
                     ignore_index=True)
    prediction = post.mprobit_predict(fit, data, targets=[dict(case="q", alternative="third"),
                                                         dict(case="q", alternative="base")])
    assert prediction["predictions"].estimate.tolist() == [0., 1.]
    assert bool((array(prediction, "jacobian") == 0).all())
    with pytest.raises(AnalysisError, match="undefined log"):
        post.mprobit_predict(fit, data, targets=[dict(case="q", alternative="third")],
                             kind="log_probability")
    margins = post.mprobit_margins(fit, data,
        targets=[dict(outcome="third", changed="third", attribute="x")], average=True,
        weights=[1., 3., 1e8])
    assert margins["support"].included.tolist() == [True, True, False]
    assert margins["averages"].weight_denominator.tolist() == [4.]
    assert margins["averages"].eligible_cases.tolist() == [2]
    assert margins["support"].structural_effect.iloc[2] == 0


def test_singleton_effect_known_zero(fit):
    data = pd.DataFrame([dict(case="q", alternative="base", x=2., available=True)])
    result = post.mprobit_margins(fit, data,
        targets=[dict(outcome="base", changed="base", attribute="x")], average=True)
    assert result["effects"].estimate.tolist() == [0.]
    assert bool((array(result, "jacobian") == 0).all())
    assert result["effects"].ci_lower.isna().all()


def test_strict_positive_elasticity_domain_before_evaluation(fit, monkeypatch):
    data = query()
    data.loc[3, "x"] = 0.
    monkeypatch.setattr(post, "_value_gradient", lambda *a: pytest.fail("evaluation preceded admission"))
    with pytest.raises(AnalysisError, match="strictly positive"):
        post.mprobit_margins(fit, data,
            targets=[dict(outcome="third", changed="third", attribute="x")], elasticity=True)


@pytest.mark.parametrize("weights", [[1., 0.], [1., -1.], [True, 1.], [float("nan"), 1.],
                                      {1: 1.}, [{"case": 1, "weight": 1.}, {"case": 1, "weight": 2.}]])
def test_bad_weights_refused(fit, weights):
    with pytest.raises(AnalysisError):
        post.mprobit_margins(fit, query(), attribute="x", average=True, weights=weights)


def test_weight_rescale_and_typed_case_records(fit):
    kwargs = dict(targets=[dict(outcome="third", changed="third", attribute="x")], average=True)
    one = post.mprobit_margins(fit, query(), weights=[1., 3.], **kwargs)
    two = post.mprobit_margins(fit, query(), weights=[{"case": "1", "weight": 3e200},
                                                   {"case": 1, "weight": 1e200}], **kwargs)
    assert one["averages"].estimate.tolist() == two["averages"].estimate.tolist()
    assert torch.equal(array(one, "jacobian"), array(two, "jacobian"))


@pytest.mark.parametrize("kwargs", [{"max_work": 1}, {"max_bytes": 1}, {"max_work": True},
                                     {"max_bytes": 0}])
def test_budget_before_query_evaluation(fit, monkeypatch, kwargs):
    monkeypatch.setattr(post, "_logs", lambda *a, **k: pytest.fail("quadrature preceded admission"))
    with pytest.raises(AnalysisError):
        post.mprobit_predict(fit, query(), **kwargs)


def test_full_json_saved_replay_no_optimizer(fit, monkeypatch):
    from openecon.econometrics.discrete import mprobit as core

    saved = json.loads(json.dumps(fit.attrs, sort_keys=True, allow_nan=False))
    monkeypatch.setattr(core, "mprobit", lambda *a, **k: pytest.fail("query refit"))
    kwargs = dict(targets=[dict(outcome="third", changed="third", attribute="x")], average=True)
    first = post.mprobit_margins(fit, query(), **kwargs)
    second = post.mprobit_margins(saved, query(), **kwargs)
    assert first.attrs == second.attrs
    assert first.to_latex() == second.to_latex()
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    changed = copy.deepcopy(saved)
    changed["mprobit_state"]["fit"]["params"][0] += .1
    with pytest.raises(AnalysisError):
        post.mprobit_predict(changed, query())


@pytest.mark.parametrize("kind", ["unknown", "logit", True])
def test_unknown_kinds(fit, kind):
    with pytest.raises(AnalysisError):
        post.mprobit_predict(fit, query(), kind=kind)


def test_duplicate_and_absent_targets_refused(fit):
    target = dict(case=1, alternative="third")
    with pytest.raises(AnalysisError, match="unique"):
        post.mprobit_predict(fit, query(), targets=[target, target])
    with pytest.raises(AnalysisError, match="absent"):
        post.mprobit_predict(fit, query(), targets=[dict(case=1, alternative="new")])
    with pytest.raises(AnalysisError, match="no simultaneous"):
        post.mprobit_margins(fit, query(), targets=[dict(outcome="base", changed="third", attribute="x")])


def test_selected_tail_own_cross_effect_and_mixed_derivative_retained(fit):
    theta, _ = physical(fit)
    beta = float(theta[0])
    z = 10.
    data = pd.DataFrame([dict(case="tail", alternative="base", x=2., available=True),
                         dict(case="tail", alternative="second", x=2.+z/beta, available=True)])
    result = post.mprobit_margins(fit, data, targets=[
        dict(outcome="second", changed="second", attribute="x"),
        dict(outcome="base", changed="second", attribute="x"),
    ])
    density = math.exp(-z*z/2)/math.sqrt(2*math.pi)
    expected = torch.tensor([beta*density, -beta*density], dtype=torch.float64)
    expected_gradient = torch.tensor([density*(1-z*z), -density*(1-z*z)], dtype=torch.float64)
    assert expected[0] > 0
    assert torch.allclose(torch.tensor(result["effects"].estimate.values), expected, rtol=2e-12, atol=0)
    assert torch.allclose(array(result, "jacobian")[:, 0], expected_gradient, rtol=2e-12, atol=0)
    assert bool((array(result, "jacobian")[:, 1:] == 0).all())
    prediction = post.mprobit_predict(fit, data)
    assert prediction["predictions"].estimate.iloc[1] == 1.
    assert array(prediction, "jacobian")[1, 0] > 0


def test_admitted_rare_probability_log_and_elasticity_without_division(fit):
    theta, _ = physical(fit)
    beta = float(theta[0])
    changed = 100.-15.75/beta
    assert changed > 0
    data = pd.DataFrame([dict(case="rare", alternative="base", x=100., available=True),
                         dict(case="rare", alternative="second", x=changed, available=True)])
    prediction = post.mprobit_predict(fit, data, targets=[dict(case="rare", alternative="second")])
    logarithm = post.mprobit_predict(fit, data, targets=[dict(case="rare", alternative="second")],
                                    kind="log_probability")
    margin = post.mprobit_margins(fit, data,
        targets=[dict(outcome="second", changed="second", attribute="x")], elasticity=True)
    assert 0 < prediction["predictions"].estimate.iloc[0] < 1e-50
    assert float(logarithm["predictions"].estimate.iloc[0]) < -120
    assert math.isfinite(float(logarithm["predictions"].standard_error.iloc[0]))
    assert margin["elasticities"].estimate.iloc[0] > 0
    assert torch.isfinite(array(margin, "jacobian")).all()
    assert math.isfinite(float(margin["elasticities"].standard_error.iloc[0]))


@pytest.mark.parametrize("threshold,base", [(-1e4, 2e4), (-16.1, 100.),
                                             (16.1, 2.), (38.7, 1e12-100.)])
@pytest.mark.parametrize("query_kind", ["probability", "log_probability", "effect", "elasticity"])
def test_public_query_refuses_outside_gaussian_derivative_accuracy_domain(fit, threshold, base, query_kind):
    theta, _ = physical(fit)
    beta = float(theta[0])
    changed = base+threshold/beta
    data = pd.DataFrame([dict(case="outside", alternative="base", x=base, available=True),
                         dict(case="outside", alternative="second", x=changed, available=True)])
    assert 0 < changed < 1e12
    with pytest.raises(AnalysisError, match="absolute 16 computational derivative-accuracy") as error:
        if query_kind in ("probability", "log_probability"):
            post.mprobit_predict(fit, data,
                targets=[dict(case="outside", alternative="second")], kind=query_kind)
        else:
            post.mprobit_margins(fit, data,
                targets=[dict(outcome="second", changed="second", attribute="x")],
                elasticity=query_kind == "elasticity")
    assert error.value.code == "numerical_domain"


def test_admitted_rare_tail_elasticity_full_mixed_derivative(fit):
    theta, _ = physical(fit)
    beta = float(theta[0])
    base, a = 100., 15.75
    changed = base-a/beta
    data = pd.DataFrame([dict(case="boundary", alternative="base", x=base, available=True),
                         dict(case="boundary", alternative="second", x=changed, available=True)])
    result = post.mprobit_margins(fit, data,
        targets=[dict(outcome="second", changed="second", attribute="x")], elasticity=True)
    # Independent scaled-complementary-error-function inverse Mills ratio;
    # neither the tiny CDF nor the production log-CDF derivative is used.
    a = beta*(base-changed)
    g = math.sqrt(2/math.pi)/float(torch.special.erfcx(
        torch.tensor(a/math.sqrt(2), dtype=torch.float64)))
    curvature = g*(a-g)
    expected = changed*beta*g
    expected_j = changed*(g-a*curvature)
    assert result["elasticities"].estimate.iloc[0] == pytest.approx(expected, rel=2e-11)
    assert array(result, "jacobian")[0, 0] == pytest.approx(expected_j, rel=3e-11)
    assert bool((array(result, "jacobian")[0, 1:] == 0).all())


def test_saved_query_cpu_and_inference_context_preserved(fit):
    kwargs = dict(targets=[dict(outcome="third", changed="third", attribute="x")], average=True)
    expected = post.mprobit_margins(fit, query(), **kwargs)
    with torch.inference_mode():
        actual = post.mprobit_margins(fit, query(), **kwargs)
        assert torch.is_inference_mode_enabled()
    pd.testing.assert_frame_equal(expected["effects"], actual["effects"])
    assert torch.equal(array(expected, "jacobian"), array(actual, "jacobian"))
    assert torch.get_default_dtype() == torch.float32


def test_requested_subset_uses_complete_choice_set_and_other_cases_do_not_change_it(fit):
    data = pd.DataFrame([dict(case="a", alternative=alternative, x=value, available=True)
                         for alternative, value in zip(("base", "second", "third"), (2., 3., 4.))])
    target = [dict(case="a", alternative="second")]
    subset = post.mprobit_predict(fit, data, targets=target)
    complete = post.mprobit_predict(fit, data)
    assert subset["predictions"].estimate.iloc[0] == complete["predictions"].estimate.iloc[1]
    extra = data.copy()
    extra["case"] = "b"
    extra["x"] += [100., 200., 300.]
    combined = post.mprobit_predict(fit, pd.concat((data, extra), ignore_index=True), targets=target)
    assert torch.equal(array(subset, "jacobian"), array(combined, "jacobian"))
    assert subset["predictions"].estimate.tolist() == combined["predictions"].estimate.tolist()


def test_default_target_overflow_refused_without_truncation(fit, monkeypatch):
    data = pd.DataFrame([dict(case=case, alternative=alternative, x=value, available=True)
                         for case in range(129)
                         for alternative, value in (("base", 2.), ("second", 3.))])
    monkeypatch.setattr(post, "_logs", lambda *a, **k: pytest.fail("target evaluation preceded limit"))
    with pytest.raises(AnalysisError, match="Default predictions exceed"):
        post.mprobit_predict(fit, data)


def test_query_pair_count_above_256_is_budgeted_not_implicitly_cut(fit):
    data = pd.DataFrame([dict(case=case, alternative=alternative, x=value, available=True)
                         for case in range(70)
                         for alternative, value in (("base", 2.), ("second", 3.))])
    result = post.mprobit_margins(fit, data, attribute="x")
    assert len(result["effects"]) == 280
    assert len(result["covariance"]) == 280
    assert len(result["support"]) == 280
    assert result.attrs["settings"]["max_declared_targets"] == 256
    assert result.attrs["settings"]["max_per_case_support_rows"] == 8192
