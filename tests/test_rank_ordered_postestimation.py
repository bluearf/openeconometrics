"""Independent derivative and finite query-contract checks, without fitting."""

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from openecon.econometrics.discrete import rank_ordered_postestimation as rp


@pytest.fixture
def query(monkeypatch):
    X = torch.tensor(
        [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [4.0, 1.0], [1.0, 3.0], [2.0, 2.0]],
        dtype=torch.float64,
        device="cpu",
    )
    beta = torch.tensor([0.4, -0.25], dtype=torch.float64, device="cpu")
    covariance = torch.tensor([[0.04, 0.025], [0.025, 0.09]], dtype=torch.float64, device="cpu")
    p = SimpleNamespace(
        X=X,
        case_rows=[[0, 1, 2], [3, 4, 5]],
        case_labels=["c1", "c2"],
        alternative_labels=["a", "b", "c", "a", "b", "c"],
        available=[1] * 6,
        rank=[1, 2, 0, 2, 0, 1],
        columns={"x": ["x1", "x2"]},
        inputs={"x": X.tolist()},
    )
    state = {
        "checksum": "test",
        "options": {"vce": "hc0"},
        "fit": {"beta": beta.tolist(), "covariance": covariance.tolist()},
    }
    monkeypatch.setattr(rp, "_query", lambda result, data, mode: (state, p, beta, covariance))
    return p, beta, covariance


@pytest.mark.parametrize("j", range(4))
def test_probability_and_log_derivatives_match_independent_autograd(j):
    X = torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 5.0], [3.0, 2.0]], dtype=torch.float64)
    beta = torch.tensor([0.45, -0.3], dtype=torch.float64)
    got = rp._choice(beta, X, list(range(4)), j)

    def f(b):
        eta = X @ b
        probability = torch.softmax(eta, 0)[j]
        log_probability = eta[j] - torch.logsumexp(eta, 0)
        log_odds = eta[j] - torch.logsumexp(eta[[k for k in range(4) if k != j]], 0)
        return torch.stack([probability, log_probability, log_odds])

    jac = torch.autograd.functional.jacobian(f, beta)
    np.testing.assert_allclose(
        [got["value"], got["log_probability"], got["log_odds"]], f(beta), rtol=2e-14
    )
    np.testing.assert_allclose(
        torch.stack([got["jacobian"], got["log_jacobian"], got["odds_jacobian"]]),
        jac,
        rtol=2e-13,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "kind,j,k,r", list(itertools.product(("effect", "elasticity"), range(3), range(3), range(2)))
)
def test_own_cross_effect_and_elasticity_jacobians(kind, j, k, r):
    X = torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
    beta = torch.tensor([0.45, -0.3], dtype=torch.float64)
    value, gradient, _, _ = rp._margin(beta, X, [0, 1, 2], j, k, r, kind)

    def f(b):
        prob = torch.softmax(X @ b, 0)
        derivative = b[r] * prob[j] * (float(j == k) - prob[k])
        return derivative if kind == "effect" else derivative * X[k, r] / prob[j]

    expected = torch.autograd.functional.jacobian(f, beta)
    assert value == pytest.approx(float(f(beta)), rel=3e-14, abs=1e-16)
    np.testing.assert_allclose(gradient, expected, rtol=4e-13, atol=1e-15)


def test_prediction_joint_covariance_retains_probability_log_crossblocks(query):
    _, _, covariance = query
    result = rp.rologit_predict(None, targets=[("c1", "a"), ("c1", "b"), ("c2", "c")])
    J = result["joint_jacobian"].iloc[:, 1:].to_numpy(float)
    V = result["joint_covariance"].iloc[:, 1:].to_numpy(float)
    np.testing.assert_allclose(V, J @ covariance.numpy() @ J.T, rtol=2e-13, atol=1e-16)
    assert np.max(np.abs(V[:3, 3:6])) > 1e-5
    assert result.attrs["settings"]["refit"] is False
    assert "source_fit_state" in result.attrs and "attrs" not in result.attrs


def test_ranking_prefix_probability_equals_compatible_permutation_mass(query):
    p, beta, _ = query
    result = rp.rologit_predict(None, mode="stages", targets=[{"case": "c1", "kind": "ranking"}])
    worth = np.exp(p.X[:3].numpy() @ beta.numpy())
    expected = sum(
        np.prod([worth[order[s]] / worth[list(order[s:])].sum() for s in range(3)])
        for order in itertools.permutations(range(3))
        if order[:2] == (0, 1)
    )
    row = result["predictions"].iloc[0]
    assert row.probability == pytest.approx(expected, rel=1e-14)
    assert row.log_probability == pytest.approx(np.log(expected), rel=1e-14)
    assert result["joint_covariance"].shape == (2, 3)


def test_selected_subset_keeps_full_denominator(query):
    p, beta, _ = query
    result = rp.rologit_predict(None, targets=[("c1", "b")])
    expected = torch.softmax(p.X[:3] @ beta, 0)[1]
    assert result["predictions"].probability.iloc[0] == pytest.approx(float(expected), rel=1e-14)
    assert result["predictions"].riskset_size.iloc[0] == 3


def test_structural_removed_unavailable_zero_and_singleton(query):
    p, _, _ = query
    p.available[5] = 0
    p.rank = [1, 2, 3, 1, 2, 0]
    result = rp.rologit_predict(
        None, mode="stages", targets=[("c1", 2, "a"), ("c2", 1, "c"), ("c1", 3, "c")]
    )
    rows = result["predictions"]
    assert list(rows.probability) == [0.0, 0.0, 1.0]
    assert list(rows.standard_error) == [0.0, 0.0, 0.0]
    assert list(rows.ci_lower) == [0.0, 0.0, 1.0]
    assert rows.log_probability.isna().tolist() == [True, True, False]
    assert result["log_probability_covariance"].shape == (1, 2)
    assert all("known structural" in s for s in rows.inference_status)


@pytest.mark.parametrize(
    "beta_value,outcome", [(800.0, "a"), (-800.0, "a"), (700.0, "a"), (-700.0, "a")]
)
def test_extreme_tail_has_stable_log_uncertainty_without_point_certainty(
    query, beta_value, outcome
):
    p, beta, _ = query
    p.X[:] = 0
    p.X[0, 0] = 1
    beta[:] = torch.tensor([beta_value, 0.0], dtype=torch.float64)
    result = rp.rologit_predict(None, targets=[("c1", outcome)])
    row = result["predictions"].iloc[0]
    assert np.isfinite(row.log_odds_ci_lower) and np.isfinite(row.log_odds_ci_upper)
    assert row.log_odds_ci_lower < row.log_odds_ci_upper
    assert not (row.ci_lower == row.ci_upper == row.probability)
    if beta_value > 700 or beta_value < -700:
        assert pd.isna(row.ci_lower) and pd.isna(row.ci_upper)


def test_weighted_ames_full_covariance_and_per_case_factorization(query):
    _, _, covariance = query
    targets = [(1, "a", "a", "x1"), (1, "a", "b", "x2"), (2, "b", "c", "x1")]
    result = rp.rologit_margins(
        None, mode="stages", targets=targets, case_weights={"c1": 2.0, "c2": 5.0}
    )
    cases = result["per_case"]
    for row in result["margins"].itertuples():
        members = cases[cases.target == row.target]
        assert row.estimate == pytest.approx(
            float((members.estimate * members.normalized_support_weight).sum())
        )
        assert row.n_support_cases == len(members)
        assert row.support_weight == members.case_weight.sum()
    J = result["jacobian"].iloc[:, 1:].to_numpy(float)
    np.testing.assert_allclose(
        result["covariance"].iloc[:, 1:], J @ covariance.numpy() @ J.T, rtol=2e-13, atol=1e-16
    )
    CJ = result["per_case_jacobian"].iloc[:, 1:].to_numpy(float)
    per_case_cov = CJ @ covariance.numpy() @ CJ.T
    np.testing.assert_allclose(
        cases.standard_error.to_numpy() ** 2, np.diag(per_case_cov), rtol=3e-13
    )
    assert abs(per_case_cov[0, 1]) > 1e-8
    assert "lossless" in result.attrs["settings"]["per_case_joint_covariance"]


def test_pair_support_no_silent_impossible_pair_average(query):
    p, _, _ = query
    p.available[5] = 0
    result = rp.rologit_margins(
        None, targets=[(1, "a", "c", "x1")], case_weights={"c1": 3.0, "c2": 7.0}
    )
    row = result["margins"].iloc[0]
    assert row.n_support_cases == 1 and row.n_query_cases == 2 and row.support_weight == 3
    assert list(result["per_case"].case) == ["c1"]
    support = result["support"].set_index("case")
    assert bool(support.loc["c1", "included_in_average"])
    assert not bool(support.loc["c2", "included_in_average"])
    assert support.loc["c2", "structural_effect"] == 0
    assert support.loc["c2", "structural_elasticity"] == 0
    assert "pair-conditional" in result.attrs["settings"]["averaging_population"]
    with pytest.raises(AnalysisError, match="no simultaneous"):
        rp.rologit_margins(None, mode="stages", targets=[(2, "a", "c", "x1")])


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_elasticities_refuse_nonpositive_changed_attribute(query, value):
    p, _, _ = query
    p.X[1, 0] = value
    with pytest.raises(AnalysisError, match="strictly positive"):
        rp.rologit_margins(None, kind="elasticity", targets=[(1, "a", "b", "x1")])


@pytest.mark.parametrize(
    "weights", [{"c1": 1}, {"c1": 1, "c2": 0}, {"c1": True, "c2": 1}, {"c1": 1, "c2": float("nan")}]
)
def test_complete_positive_fixed_standardization_weights(query, weights):
    with pytest.raises(AnalysisError):
        rp.rologit_margins(None, targets=[(1, "a", "a", "x1")], case_weights=weights)


def test_current_workspace_override_refuses_before_joint_allocation(query, monkeypatch):
    p, _, _ = query
    p.case_rows *= 30
    p.case_labels = [f"c{i}" for i in range(60)]
    targets = [(f"c{i}", "a") for i in range(60)]

    def forbidden(*args):
        raise AssertionError("covariance allocated before workspace preflight")

    monkeypatch.setattr(rp, "_factor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        rp.rologit_predict(None, targets=targets)
    assert error.value.code == "workspace_limit"


def test_global_tensor_defaults_and_saved_json(query):
    before_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            result = rp.rologit_margins(None, targets=[(1, "a", "a", "x1")])
        assert np.isfinite(result["margins"].estimate.iloc[0])
        json.dumps(result.attrs, allow_nan=False)
        assert "\\begin{tabular}" in result.to_latex()
    finally:
        torch.set_default_dtype(before_dtype)


def test_removed_outcome_zero_effect_has_undefined_elasticity(query):
    result = rp.rologit_margins(None, mode="stages", targets=[(2, "a", "b", "x1")])
    support = result["support"].set_index("case")
    assert list(result["per_case"].case) == ["c2"]
    assert support.loc["c1", "structural_effect"] == 0
    assert pd.isna(support.loc["c1", "structural_elasticity"])
    assert "zero outcome probability" in support.loc["c1", "support_status"]


def test_absent_alternative_gets_no_invented_derivative(query):
    p, _, _ = query
    p.alternative_labels[5] = "d"
    result = rp.rologit_margins(None, targets=[(1, "a", "c", "x1")])
    row = result["support"].set_index("case").loc["c2"]
    assert not row.included_in_average
    assert pd.isna(row.structural_effect) and pd.isna(row.structural_elasticity)
    assert "absent" in row.support_status


def test_matrix_identifier_columns_do_not_collide_with_attribute_names(query):
    p, _, _ = query
    p.columns["x"] = ["target", "parameter"]
    result = rp.rologit_margins(None, targets=[(1, "a", "a", "target")])
    assert result["jacobian"].columns.is_unique
    assert list(result["jacobian"].columns) == ["_target", "target", "parameter"]
    assert list(result["coefficient_covariance"].columns) == ["_parameter", "target", "parameter"]


def test_margin_workspace_override_refuses_before_case_jacobian_allocation(query, monkeypatch):
    p, _, _ = query
    p.X = torch.ones((6, 8), dtype=torch.float64)
    p.columns["x"] = [f"x{i}" for i in range(8)]
    p.case_rows *= 64
    p.case_labels = [f"c{i}" for i in range(128)]
    beta = torch.ones(8, dtype=torch.float64)
    covariance = torch.eye(8, dtype=torch.float64)
    state = {"checksum": "test", "options": {"vce": "hc0"}}
    monkeypatch.setattr(rp, "_query", lambda result, data, mode: (state, p, beta, covariance))
    targets = list(itertools.product([1], ["a", "b", "c"], ["a", "b", "c"], p.columns["x"]))[:64]

    def forbidden(*args):
        raise AssertionError("coefficient factor allocated before workspace preflight")

    monkeypatch.setattr(rp, "_factor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        rp.rologit_margins(None, targets=targets)
    assert error.value.code == "workspace_limit"


def test_zero_covariance_does_not_invent_probability_or_margin_certainty(query):
    _, _, covariance = query
    covariance.zero_()
    prediction = rp.rologit_predict(None, targets=[("c1", "a")])["predictions"].iloc[0]
    margin = rp.rologit_margins(None, targets=[(1, "a", "b", "x1")])["margins"].iloc[0]
    assert prediction.standard_error == margin.standard_error == 0
    assert pd.isna(prediction.ci_lower) and pd.isna(margin.ci_lower)
    assert pd.isna(margin.z) and pd.isna(margin.p_value)
    assert "unavailable" in prediction.inference_status and "unavailable" in margin.inference_status
