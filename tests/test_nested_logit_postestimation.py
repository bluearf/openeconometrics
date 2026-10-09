"""Independent NumPy formulas, physical-parameter deltas and support contracts."""

import itertools
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete import nested_logit_postestimation as npst
from openecon.resources import use_workspace_budget


@pytest.fixture
def query(monkeypatch):
    X = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
            [4.0, 3.0],
            [4.0, 1.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float64,
        device="cpu",
    )
    theta = torch.tensor([0.4, -0.25, 0.6, 0.8], dtype=torch.float64, device="cpu")
    factor = np.array(
        [[0.2, 0, 0, 0], [0.1, 0.25, 0, 0], [0.015, 0.03, 0.07, 0], [0.02, -0.015, 0.01, 0.06]]
    )
    covariance = torch.tensor(factor @ factor.T, dtype=torch.float64, device="cpu")
    p = SimpleNamespace(
        X=X,
        case_rows=[[0, 1, 2, 3], [4, 5, 6, 7]],
        case_labels=[1, "1"],
        alternative_labels=["a", "b", "c", "d"] * 2,
        available=[1] * 8,
        nest_labels=["left", "right"],
        nest_codes=[0, 0, 1, 1] * 2,
        free_nests=[0, 1],
        fixed_lambdas={},
        parameters=["x1", "x2", "lambda[n1]", "lambda[n2]"],
        columns={"x": ["x1", "x2"]},
        inputs={"x": X.tolist()},
    )
    state = {
        "checksum": "test",
        "options": {"vce": "cr0"},
        "fit": {"params": theta.tolist(), "covariance": covariance.tolist()},
    }
    monkeypatch.setattr(npst, "_query", lambda result, data: (state, p, theta, covariance))
    return p, theta, covariance


def oracle(theta, X, nests):
    """Direct moderate-utility formula, deliberately independent of production."""
    beta, lambdas = theta[:2], theta[2:]
    utility = X @ beta
    sums = np.array([np.exp(utility[nests == m] / lambdas[m]).sum() for m in range(2)])
    group = sums**lambdas
    nest_probability = group / group.sum()
    conditional = np.array([np.exp(u / lambdas[m]) / sums[m] for u, m in zip(utility, nests)])
    return conditional * nest_probability[nests], conditional, nest_probability


def finite_gradient(function, theta, step=2e-6):
    result = []
    for i in range(len(theta)):
        delta = np.eye(len(theta))[i] * step
        result.append((function(theta + delta) - function(theta - delta)) / (2 * step))
    return np.asarray(result)


@pytest.mark.parametrize(
    "kind",
    [
        "probability",
        "log_probability",
        "conditional",
        "log_conditional",
        "nest_probability",
        "log_nest_probability",
    ],
)
@pytest.mark.parametrize("index", range(2))
def test_all_probability_scales_full_physical_deltas(query, kind, index):
    p, theta, covariance = query
    target = dict(case=1, kind=kind)
    target["nest" if "nest" in kind else "alternative"] = (
        p.nest_labels[index] if "nest" in kind else p.alternative_labels[index]
    )
    got = npst.nlogit_predict(None, targets=[target])

    def f(t):
        prob, cond, nest = oracle(t, p.X[:4].numpy(), np.array(p.nest_codes[:4]))
        value = (
            nest[index] if "nest" in kind else cond[index] if "conditional" in kind else prob[index]
        )
        return np.log(value) if kind.startswith("log_") else value

    expected_j = finite_gradient(f, theta.numpy())
    row = got["predictions"].iloc[0]
    assert row.estimate == pytest.approx(f(theta.numpy()), rel=3e-14)
    np.testing.assert_allclose(
        got["jacobian"].iloc[0, 1:].to_numpy(float), expected_j, rtol=2e-8, atol=2e-10
    )
    assert row.standard_error**2 == pytest.approx(
        expected_j @ covariance.numpy() @ expected_j, rel=3e-8, abs=1e-14
    )


@pytest.mark.parametrize(
    "kind,j,k,r", list(itertools.product(("effect", "elasticity"), range(4), range(4), range(2)))
)
def test_analytical_own_cross_margins_match_raw_cell_numerical_derivative(query, kind, j, k, r):
    p, theta, _ = query
    X = p.X[:4].numpy().copy()
    nests = np.array(p.nest_codes[:4])
    raw_step = 2e-5

    def independently_derived(t):
        upper, lower = X.copy(), X.copy()
        upper[k, r] += raw_step
        lower[k, r] -= raw_step
        pa = oracle(t, upper, nests)[0][j]
        pb = oracle(t, lower, nests)[0][j]
        derivative = (pa - pb) / (2 * raw_step)
        return derivative if kind == "effect" else derivative * X[k, r] / oracle(t, X, nests)[0][j]

    value, gradient = npst._derivative(lambda t: npst._margin_value(t, p, 0, j, k, r, kind), theta)
    assert value == pytest.approx(independently_derived(theta.numpy()), rel=3e-9, abs=3e-11)
    expected_j = finite_gradient(independently_derived, theta.numpy(), step=2e-4)
    np.testing.assert_allclose(gradient, expected_j, rtol=2e-6, atol=2e-7)


def test_full_prediction_covariance_includes_probability_log_odds_and_lambda_crossblocks(query):
    _, _, covariance = query
    result = npst.nlogit_predict(
        None,
        targets=[
            dict(case=1, alternative="a"),
            dict(case="1", alternative="c"),
            dict(case=1, nest="left", kind="nest_probability"),
        ],
    )
    J = result["joint_jacobian"].iloc[:, 1:].to_numpy(float)
    V = result["joint_covariance"].iloc[:, 1:].to_numpy(float)
    np.testing.assert_allclose(V, J @ covariance.numpy() @ J.T, rtol=3e-13, atol=1e-16)
    assert np.abs(J[:, 2:]).max() > 0.02
    assert np.abs(V[:3, 3:]).max() > 1e-4
    assert result.attrs["settings"]["refit"] is False


def test_selected_prediction_keeps_unmaterialized_alternatives_and_nests(query):
    p, theta, _ = query
    got = npst.nlogit_predict(None, targets=[dict(case=1, alternative="b")])
    row = got["predictions"].iloc[0]
    assert row.estimate == pytest.approx(
        oracle(theta.numpy(), p.X[:4].numpy(), np.array(p.nest_codes[:4]))[0][1]
    )
    assert row.riskset_size == 4


def test_weighted_support_and_full_average_and_per_case_covariance(query):
    p, _, covariance = query
    p.available[6] = 0
    targets = [
        dict(outcome="a", changed="b", attribute="x1"),
        dict(outcome="a", changed="c", attribute="x2"),
    ]
    got = npst.nlogit_margins(
        None, targets=targets, case_weights=[dict(case=1, weight=2.0), dict(case="1", weight=7.0)]
    )
    per_case = got["per_case"]
    for row in got["margins"].itertuples():
        members = per_case[per_case.target == row.target]
        assert row.estimate == pytest.approx(
            (members.estimate * members.normalized_support_weight).sum()
        )
        assert row.support_weight == members.case_weight.sum()
    assert got["margins"].n_support_cases.tolist() == [2, 1]
    assert got["support"].included_in_average.tolist() == [True, True, True, False]
    J = got["jacobian"].iloc[:, 1:].to_numpy(float)
    CJ = got["per_case_jacobian"].iloc[:, 1:].to_numpy(float)
    np.testing.assert_allclose(
        got["covariance"].iloc[:, 1:], J @ covariance.numpy() @ J.T, rtol=3e-13, atol=1e-16
    )
    np.testing.assert_allclose(
        per_case.standard_error.to_numpy() ** 2, np.diag(CJ @ covariance.numpy() @ CJ.T), rtol=3e-13
    )
    assert "lossless" in got.attrs["settings"]["per_case_joint_covariance"]


def test_effects_sum_to_zero_across_all_outcomes(query):
    got = npst.nlogit_margins(
        None, targets=[dict(outcome=o, changed="b", attribute="x1") for o in ("a", "b", "c", "d")]
    )
    assert got["margins"].estimate.sum() == pytest.approx(0, abs=2e-16)
    np.testing.assert_allclose(got["jacobian"].iloc[:, 1:].sum(axis=0), 0, atol=2e-16)


def test_fixed_lambda_one_reduces_to_multinomial_logit(query):
    p, theta, covariance = query
    p.free_nests = []
    p.fixed_lambdas = {0: 1.0, 1: 1.0}
    theta.resize_(2)
    covariance.resize_(2, 2)
    p.parameters = p.parameters[:2]
    eta = p.X[:4] @ theta
    expected = torch.softmax(eta, 0)
    dist = npst._distribution(theta, p, 0)
    for row in range(4):
        assert float(dist["alternative"][row][0].exp()) == pytest.approx(
            float(expected[row]), rel=3e-14
        )
    got = npst._margin_value(theta, p, 0, 0, 1, 0, "effect")
    assert float(got) == pytest.approx(float(-theta[0] * expected[0] * expected[1]), rel=3e-14)


def test_unavailable_empty_nest_and_undefined_logs_are_explicit(query):
    p, _, _ = query
    p.available[2] = p.available[3] = 0
    got = npst.nlogit_predict(
        None,
        targets=[
            dict(case=1, alternative="c"),
            dict(case=1, nest="right", kind="nest_probability"),
            dict(case=1, nest="left", kind="nest_probability"),
        ],
    )
    assert got["predictions"].estimate.tolist() == [0.0, 0.0, 1.0]
    assert got["predictions"].standard_error.tolist() == [0.0, 0.0, 0.0]
    assert all("known structural" in v for v in got["predictions"].inference_status)
    for kind in ("log_probability", "conditional", "log_conditional"):
        with pytest.raises(AnalysisError, match="undefined"):
            npst.nlogit_predict(None, targets=[dict(case=1, alternative="c", kind=kind)])
    with pytest.raises(AnalysisError, match="undefined"):
        npst.nlogit_predict(None, targets=[dict(case=1, nest="right", kind="log_nest_probability")])


@pytest.mark.parametrize("beta", [-800.0, -700.0, 700.0, 800.0])
def test_extreme_probability_tails_keep_log_odds_uncertainty(query, beta):
    p, theta, _ = query
    p.X[:] = 0
    p.X[0, 0] = 1
    theta[0] = beta
    theta[1] = 0
    got = npst.nlogit_predict(None, targets=[dict(case=1, alternative="a")])
    row = got["predictions"].iloc[0]
    assert np.isfinite(row.log_odds)
    assert row.log_odds_standard_error > 0
    assert not (row.ci_lower == row.ci_upper == row.estimate)


def test_elasticity_avoids_division_by_underflowed_outcome_probability(query):
    p, theta, _ = query
    p.X[:] = 1
    p.X[0, 0] = 2
    theta[0] = -800.0
    theta[1] = 0
    got = npst.nlogit_margins(
        None, kind="elasticity", targets=[dict(outcome="a", changed="b", attribute="x1")]
    )
    assert np.isfinite(got["margins"].estimate.iloc[0])
    assert np.isfinite(got["jacobian"].iloc[:, 1:].to_numpy(float)).all()


@pytest.mark.parametrize("beta", [35.0, 50.0, 100.0])
@pytest.mark.parametrize("kind", ["effect", "elasticity"])
def test_own_selected_tail_keeps_representable_complement_and_physical_gradient(query, beta, kind):
    p, theta, _ = query
    p.X[:] = 0
    p.X[0, 0] = 1
    theta[0], theta[1] = beta, 0

    def independent_tail(t):
        b, l0, l1 = t[0], t[2], t[3]
        one, two = Decimal(1), Decimal(2)
        lower_mass = (b / l0).exp() + one
        # Compute losing masses directly, never subtract a rounded one.
        q_complement = one / lower_mass
        nest_complement = two**l1 / (lower_mass**l0 + two**l1)
        q = one / (one + (-b / l0).exp())
        prob = q / (one + two**l1 / lower_mass**l0)
        bracket = q_complement / l0 + q * nest_complement
        return b * bracket if kind == "elasticity" else b * prob * bracket

    value, gradient = npst._derivative(lambda t: npst._margin_value(t, p, 0, 0, 0, 0, kind), theta)
    assert value > 0
    with localcontext() as context:
        context.prec = 100
        point = [Decimal.from_float(float(v)) for v in theta]
        assert value == pytest.approx(float(independent_tail(point)), rel=3e-13)
        expected = []
        step = Decimal("1e-15")
        for i in (0, 2, 3):
            upper, lower = point.copy(), point.copy()
            upper[i] += step
            lower[i] -= step
            expected.append(float((independent_tail(upper) - independent_tail(lower)) / (2 * step)))
    np.testing.assert_allclose(gradient[[0, 2, 3]], expected, rtol=3e-12, atol=0)
    if beta >= 50:
        assert float(npst._distribution(theta, p, 0)["alternative"][0][0].exp()) == 1.0


@pytest.mark.parametrize("beta", [50.0, 100.0])
@pytest.mark.parametrize("changed", [1, 2])
@pytest.mark.parametrize("kind", ["effect", "elasticity"])
def test_same_and_cross_nest_tail_effects_and_elasticities_decimal_gradient(
    query, beta, changed, kind
):
    p, theta, _ = query
    p.X[:] = 0
    p.X[:, 0] = 1
    p.X[0, 0] = 2
    theta[0], theta[1] = beta, 0

    def oracle_tail(point):
        b, l0, l1 = point[0], point[2], point[3]
        one, two = Decimal(1), Decimal(2)
        lower = (b / l0).exp() + one
        qc = one / lower
        nc = two**l1 / (lower**l0 + two**l1)
        outcome = (one - qc) * (one - nc)
        bracket = -((one / l0 - one) * qc + qc * (one - nc)) if changed == 1 else -nc / two
        return b * bracket if kind == "elasticity" else b * outcome * bracket

    value, gradient = npst._derivative(
        lambda t: npst._margin_value(t, p, 0, 0, changed, 0, kind), theta
    )
    with localcontext() as context:
        context.prec = 150
        point = [Decimal.from_float(float(v)) for v in theta]
        assert value < 0
        assert value == pytest.approx(float(oracle_tail(point)), rel=3e-13)
        expected = []
        step = Decimal("1e-15")
        for i in (0, 2, 3):
            upper, lower = point.copy(), point.copy()
            upper[i] += step
            lower[i] -= step
            expected.append(float((oracle_tail(upper) - oracle_tail(lower)) / (2 * step)))
    np.testing.assert_allclose(gradient[[0, 2, 3]], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [0.0, 1.0],
        [True, 1.0],
        {1: 1.0},
        [dict(case=1, weight=1.0), dict(case=1, weight=2.0)],
        [float("nan"), 1.0],
    ],
)
def test_complete_positive_typed_case_weights(query, weights):
    with pytest.raises(AnalysisError):
        npst.nlogit_margins(
            None, targets=[dict(outcome="a", changed="b", attribute="x1")], case_weights=weights
        )


def test_nonpositive_elasticity_attribute_and_no_pair_support(query):
    p, _, _ = query
    p.X[1, 0] = 0
    with pytest.raises(AnalysisError, match="strictly positive"):
        npst.nlogit_margins(
            None, kind="elasticity", targets=[dict(outcome="a", changed="b", attribute="x1")]
        )
    p.available[0] = p.available[4] = 0
    with pytest.raises(AnalysisError, match="no simultaneous"):
        npst.nlogit_margins(None, targets=[dict(outcome="a", changed="b", attribute="x1")])


def test_workspace_preflight_before_covariance_factor(query, monkeypatch):
    p, _, _ = query
    p.case_rows *= 30
    p.case_labels = [f"case{i}" for i in range(60)]

    def forbidden(*args):
        raise AssertionError("factor allocated before workspace preflight")

    monkeypatch.setattr(npst, "_factor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        npst.nlogit_predict(
            None, targets=[dict(case=case, alternative="a") for case in p.case_labels]
        )
    assert error.value.code == "workspace_limit"


def test_duplicate_targets_and_invalid_kind(query):
    target = dict(case=1, alternative="a")
    with pytest.raises(AnalysisError, match="unique"):
        npst.nlogit_predict(None, targets=[target, target])
    with pytest.raises(AnalysisError, match="Unknown"):
        npst.nlogit_predict(None, targets=[{**target, "kind": "odds"}])


def test_device_defaults_do_not_change_cpu_query_or_json_serialization(query):
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            result = npst.nlogit_margins(
                None, targets=[dict(outcome="a", changed="b", attribute="x1")]
            )
        assert np.isfinite(result["margins"].estimate.iloc[0])
        assert result.attrs["settings"]["precision"] == "float64"
        result["margins"].to_json()
    finally:
        torch.set_default_dtype(before)


def test_structural_effect_zero_exclusion_keeps_elasticity_policy(query):
    p, _, _ = query
    p.available[6] = 0
    result = npst.nlogit_margins(None, targets=[dict(outcome="a", changed="c", attribute="x1")])
    row = result["support"].iloc[1]
    assert row.structural_effect == 0
    assert row.structural_elasticity == 0
    p.available[4] = 0
    result = npst.nlogit_margins(None, targets=[dict(outcome="a", changed="c", attribute="x1")])
    assert pd.isna(result["support"].iloc[1].structural_elasticity)
