"""Independent full-support convex oracles for iid Turnbull interval NPMLE."""
from __future__ import annotations

import hashlib
from itertools import combinations_with_replacement
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog, minimize
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext import turnbull as implementation
from openecon.econometrics.survival_ext.turnbull import turnbull


def primitive_oracle(lower, upper, times=()):
    """Enumerate every boundary atom/open gap, without maximal-intersection code.

    Midpoints are only numerical representatives of constant observation
    membership in this independent test oracle. Requested times split gaps
    so LP can put mass on either side; no fitted location is assumed.
    """
    lower = np.array(lower, dtype=float)
    upper = np.array([math.inf if v is None else v for v in upper], dtype=float)
    endpoints = sorted({0., *lower, *upper[np.isfinite(upper)], *times})
    candidates = []
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        if left < right:
            candidates.append(left+(right-left)/2)
        if right > 0:
            candidates.append(right)
    candidates.append(endpoints[-1]+max(1., abs(endpoints[-1])))
    candidates = np.array(candidates)
    exact = lower == upper
    A = ((lower[:, None] < candidates) & (candidates <= upper[:, None]))
    A[exact] = candidates == lower[exact, None]
    admitted = A.any(0)
    return A[:, admitted].astype(float), candidates[admitted]


def scipy_npmle(lower, upper, times=()):
    A, locations = primitive_oracle(lower, upper, times)
    k = A.shape[1]

    def objective(p):
        q = A@p
        return -np.log(np.maximum(q, 1e-300)).sum()

    def gradient(p):
        return -(A.T@(1./np.maximum(A@p, 1e-300)))

    result = minimize(objective, np.full(k, 1/k), jac=gradient,
                      method="SLSQP", bounds=[(0., 1.)]*k,
                      constraints=dict(type="eq", fun=lambda p: p.sum()-1., jac=lambda p: np.ones(k)),
                      options=dict(ftol=1e-12, maxiter=2000))
    assert result.success, result.message
    return dict(A=A, locations=locations, log_likelihood=-result.fun,
                probabilities=A@result.x, mass=result.x)


SMALL_TYPES = ((1., 1.), (2., 2.), (0., 1.), (0., 2.),
               (1., 2.), (1., 3.), (1., None), (2., None))
SMALL_SAMPLES = list(combinations_with_replacement(range(len(SMALL_TYPES)), 3))


@pytest.mark.parametrize("sample", SMALL_SAMPLES)
def test_exhaustive_three_observation_small_oracle(sample):
    lower = [SMALL_TYPES[i][0] for i in sample]
    upper = [SMALL_TYPES[i][1] for i in sample]
    result = turnbull(lower, upper, tol=1e-10)
    expected = scipy_npmle(lower, upper)
    q = result["observations"]["probability"].to_numpy()
    np.testing.assert_allclose(q, expected["probabilities"], rtol=2e-6, atol=2e-7)
    assert result.attrs["log_likelihood"] == pytest.approx(expected["log_likelihood"], abs=4e-8)
    # Independent KKT gate over all primitive support columns, including
    # every column discarded as dominated by the production support sweep.
    gradient = expected["A"].T@(1/q)/len(lower)
    assert gradient.max() <= 1.+1.1e-10
    assert result.attrs["normalized_dual_gap"] <= 1e-10
    assert result["convergence"]["log_likelihood"].diff().dropna().min() >= -1e-11 or len(result["convergence"]) == 1
    assert result["support"]["mass"].sum() == pytest.approx(1., abs=2e-15)
    assert bool((result["support"]["mass"] >= 0).all())
    assert result.attrs["support_rank"] == len(result["support"])


MIXED = (
    ([0., 1., 1., 0., 0., 2.], [1., 3., 3., 2., 2., 3.]),
    ([1., 1., 2., 3., 4., 4.], [1., None, 2., None, 4., None]),
    ([0., 0., 1., 2., 2., 3., 4.], [2., 3., 3., 4., 2., 5., None]),
    ([0., 1., 2., 3.], [1., 2., 4., None]),
)


@pytest.mark.parametrize("lower,upper", MIXED)
def test_sharp_location_envelope_independent_full_support_linear_programs(lower, upper):
    times = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 7.]
    result = turnbull(lower, upper, times=times, tol=1e-10)
    expected = scipy_npmle(lower, upper, times)
    q = result["observations"]["probability"].to_numpy()
    np.testing.assert_allclose(q, expected["probabilities"], rtol=3e-6, atol=3e-7)
    A, locations = expected["A"], expected["locations"]
    equality = np.vstack((np.ones(A.shape[1]), A))
    target = np.r_[1., q]
    for row in result["cdf_bounds"].itertuples():
        cdf = (locations <= row.time).astype(float)
        minimum = linprog(cdf, A_eq=equality, b_eq=target, bounds=(0., None), method="highs")
        maximum = linprog(-cdf, A_eq=equality, b_eq=target, bounds=(0., None), method="highs")
        assert minimum.success, minimum.message
        assert maximum.success, maximum.message
        assert row.cdf_lower == pytest.approx(minimum.fun, abs=2e-8)
        assert row.cdf_upper == pytest.approx(-maximum.fun, abs=2e-8)
        assert row.survival_lower == pytest.approx(1-row.cdf_upper, abs=2e-15)
        assert row.survival_upper == pytest.approx(1-row.cdf_lower, abs=2e-15)
        assert 0 <= row.cdf_lower <= row.cdf_upper <= 1


def test_published_gentleman_geyer_self_consistency_counterexample():
    # Section4: the (1/2,0,1/2) fixed point is not the likelihood maximum.
    result = turnbull([0., 1., 1., 0., 0., 2.], [1., 3., 3., 2., 2., 3.])
    np.testing.assert_allclose(result["support"]["mass"], np.repeat(1/3, 3), atol=1e-15)
    q_bad = np.array([.5, .5, .5, .5, .5, .5])
    assert result.attrs["log_likelihood"] > np.log(q_bad).sum()
    np.testing.assert_allclose(result["support"]["normalized_score"], np.ones(3), atol=1e-15)


def test_exact_events_right_censor_ties_match_hand_product_limit():
    result = turnbull([1., 1., 2., 3., 4., 4.], [1., None, 2., None, 4., None],
                      times=[0., 1., 1.5, 2., 3., 4., 5.], tol=1e-11)
    # Risk sets 6,4,2 at failure times1,2,4; censoring at the event time
    # remains in that risk set, while its observation is strictly T>time.
    expected_mass = np.array([1/6, 5/24, 5/16, 5/16])
    np.testing.assert_allclose(result["support"]["mass"], expected_mass, rtol=2e-10, atol=1e-11)
    assert list(result["support"]["kind"]) == ["exact_point"]*3+["right_tail_interval"]
    np.testing.assert_allclose(result["support"]["lower"], [1., 2., 4., 4.])
    point = result["cdf_bounds"].set_index("time")
    assert point.loc[1., "cdf_lower"] == pytest.approx(1/6, abs=1e-11)
    assert point.loc[2., "cdf_lower"] == pytest.approx(3/8, abs=1e-11)
    assert point.loc[4., "cdf_lower"] == pytest.approx(11/16, abs=1e-11)
    assert point.loc[5., "cdf_lower"] == pytest.approx(11/16, abs=1e-11)
    assert point.loc[5., "cdf_upper"] == pytest.approx(1., abs=1e-15)
    assert not bool(point.loc[5., "point_identified"])


def test_all_exact_tied_atoms_are_empirical_distribution():
    exact = [3., 1., 1., 5., 3., 1.]
    result = turnbull(exact, exact, times=[0., 1., 2., 3., 4., 5., 6.])
    np.testing.assert_allclose(result["support"]["mass"], [3/6, 2/6, 1/6], atol=1e-15)
    np.testing.assert_allclose(result["support"]["lower"], [1., 3., 5.])
    np.testing.assert_array_equal(result["support"]["lower"], result["support"]["upper"])
    np.testing.assert_array_equal(result["support"]["left_closed"], [True]*3)
    np.testing.assert_array_equal(result["support"]["right_closed"], [True]*3)
    expected = [(np.array(exact) <= t).mean() for t in result["cdf_bounds"]["time"]]
    np.testing.assert_allclose(result["cdf_bounds"]["cdf_lower"], expected, atol=1e-15)
    np.testing.assert_allclose(result["cdf_bounds"]["cdf_upper"], expected, atol=1e-15)
    assert bool(result["cdf_bounds"]["point_identified"].all())


def test_open_left_closed_right_exact_boundary_membership_and_no_fake_midpoints():
    result = turnbull([0., 1., 2.], [1., 1., 3.], times=[0., 1., 2., 2.5, 3.])
    support = result["support"]
    assert list(support["kind"]) == ["exact_point", "interval"]
    assert list(support["lower"]) == [1., 2.]
    assert list(support["upper"]) == [1., 3.]
    assert list(support["left_closed"]) == [True, False]
    assert list(support["right_closed"]) == [True, True]
    bounds = result["cdf_bounds"].set_index("time")
    assert bounds.loc[2., "cdf_lower"] == pytest.approx(2/3)
    assert bounds.loc[2., "cdf_upper"] == pytest.approx(2/3)
    assert bounds.loc[2.5, "cdf_lower"] == pytest.approx(2/3)
    assert bounds.loc[2.5, "cdf_upper"] == pytest.approx(1.)
    assert not bool(bounds.loc[2.5, "point_identified"])
    assert "cdf" not in bounds.columns
    assert "time" not in support.columns


def test_all_right_tail_mass_is_not_a_cure_atom_and_all_left_locations_remain_unknown():
    right = turnbull([1., 2., 3.], [None, np.inf, None], times=[0., 3., 3.1, 100.])
    assert right["support"].iloc[0]["kind"] == "right_tail_interval"
    assert right["support"].iloc[0]["lower"] == 3.
    assert right["support"].iloc[0]["upper"] is None
    assert right["support"].iloc[0]["mass"] == 1.
    np.testing.assert_array_equal(right["cdf_bounds"]["cdf_lower"], [0.]*4)
    np.testing.assert_array_equal(right["cdf_bounds"]["cdf_upper"], [0., 0., 1., 1.])
    assert "no atom at infinity" in right.attrs["right_tail"]
    assert right.attrs["upper"] == [None]*3
    assert turnbull([1., 2.], [None, None])["cdf_bounds"]["time"].tolist() == [0.]
    left = turnbull([0., 0., 0.], [2., 3., 4.], times=[0., 1., 2., 3.])
    assert len(left["support"]) == 1
    assert left["support"].iloc[0]["lower"] == 0.
    assert left["support"].iloc[0]["upper"] == 2.
    np.testing.assert_array_equal(left["cdf_bounds"]["cdf_lower"], [0., 0., 1., 1.])
    np.testing.assert_array_equal(left["cdf_bounds"]["cdf_upper"], [0., 1., 1., 1.])


def test_location_bounds_complete_input_state_checksum_json_replay_and_latex():
    lower, upper = MIXED[2]
    original = turnbull(lower, upper, times=[0., 1., 2., 2.5, 3., 4., 5., 6.])
    saved = json.loads(json.dumps(original.attrs, allow_nan=False))
    state = saved["npmle_state"].copy()
    checksum = state.pop("checksum")
    assert checksum == hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    replay = turnbull(saved["lower"], saved["upper"], times=saved["times"], maxiter=saved["maxiter"],
                      tol=saved["tol"], device=saved["device"], weights=saved["weights"])
    assert replay.attrs == original.attrs
    for name in original:
        pd.testing.assert_frame_equal(original[name], replay[name])
        latex = original[name].to_latex()
        assert any(f"\\begin{{{env}}}" in latex and f"\\end{{{env}}}" in latex for env in ("tabular", "longtable"))
    assert original.attrs["complete_inputs_saved"]
    assert "not reported" in original.attrs["uncertainty"]
    assert len(state["observation_cell_membership"]) == len(lower)
    A = np.zeros((len(lower), len(state["mass"])))
    for i, admitted in enumerate(state["observation_cell_membership"]):
        A[i, np.array(admitted)-1] = 1.
    np.testing.assert_allclose(A@state["mass"], state["observation_probabilities"], atol=1e-15)
    assert original["observations"]["log_likelihood"].sum() == pytest.approx(original.attrs["log_likelihood"], abs=1e-14)


@pytest.mark.parametrize("name,value", (("maxiter", True), ("maxiter", 0), ("maxiter", 1.5),
                                        ("maxiter", 10001), ("tol", True), ("tol", 0),
                                        ("tol", np.nan), ("tol", 1e-13), ("tol", .1)))
def test_solver_controls_are_bounded(name, value):
    with pytest.raises(AnalysisError):
        turnbull([0., 1.], [1., 2.], **{name: value})


def test_small_em_changes_do_not_replace_kkt_certification_and_iteration_failures_refuse_result():
    with pytest.raises(AnalysisError, match="likelihood/KKT dual-gap"):
        turnbull([0., 1., 0., 2.], [1., 3., 2., 3.], maxiter=1, tol=1e-12)
    result = turnbull([0., 1., 0., 2.], [1., 3., 2., 3.], tol=1e-10)
    assert result.attrs["iterations"] > 1
    assert result.attrs["normalized_dual_gap"] <= 1e-10


def test_support_and_output_limits_refuse_thinning_and_budget_checked(monkeypatch):
    exact = list(np.arange(1., 258.))
    with pytest.raises(AnalysisError, match="support exceeds 256"):
        turnbull(exact, exact)
    with pytest.raises(AnalysisError, match="times exceeds 256"):
        turnbull([1.], [1.], times=list(np.arange(257.)))
    with monkeypatch.context() as patch:
        patch.setattr("openecon.econometrics.survival_ext.common.workspace_budget_bytes", lambda: 1)
        with pytest.raises(AnalysisError, match="workspace exceeds"):
            turnbull([1.], [1.])
    # Repeated observations scale to n=4096 without an endpoint-sized matrix.
    repeated = turnbull([1.]*4096, [1.]*4096, times=[0., 1., 2.])
    assert len(repeated["observations"]) == 4096
    assert repeated.attrs["support_cells"] == 1
    assert repeated["support"]["mass"].iloc[0] == 1.


def test_declared_iteration_work_is_checked_before_dense_support_allocation(monkeypatch):
    exact = list(np.arange(1., 257.))

    def unexpected_dense_allocation(*args, **kwargs):
        raise AssertionError("over-budget input reached a dense support allocation")

    with monkeypatch.context() as patch:
        patch.setattr(implementation.torch, "zeros", unexpected_dense_allocation)
        with pytest.raises(AnalysisError, match="iteration/support work exceeds"):
            turnbull(exact, exact, maxiter=10000)
        # Repeated EM alone is 419 million units, but the once-only rank
        # SVD adds 268 million and must also be rejected before allocation.
        repeated = exact*16
        with pytest.raises(AnalysisError, match="iteration/support work exceeds"):
            turnbull(repeated, repeated, maxiter=400)
    small = turnbull([1., 2.], [1., 2.], maxiter=10000)
    assert small.attrs["work_estimate"] == 200008
    assert small.attrs["work_limit"] == 500000000
    assert small.attrs["work_estimate"] <= small.attrs["work_limit"]


def test_full_rank_mass_certificate_is_required_even_if_em_would_self_consist(monkeypatch):
    real_support = implementation._support

    def duplicate_support(data):
        cells = real_support(data)
        return [cells[0], cells[0].copy()]

    monkeypatch.setattr(implementation, "_support", duplicate_support)
    with pytest.raises(AnalysisError, match="full column rank"):
        turnbull([1., 1.], [1., 1.])


def test_native_cpu_float64_and_decomposition_errors_override_torch_defaults(monkeypatch):
    lower, upper = MIXED[1]
    base = turnbull(lower, upper, times=[0., 1., 2., 4., 5.])
    previous_dtype, previous_device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = turnbull(lower, upper, times=[0., 1., 2., 4., 5.])
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)
    for name in base:
        pd.testing.assert_frame_equal(base[name], result[name])
    assert result.attrs["device"] == "cpu"
    assert result.attrs["precision"] == "float64"

    def decomposition_failure(*args, **kwargs):
        raise torch.linalg.LinAlgError("injected decomposition failure")

    monkeypatch.setattr(torch.linalg, "svdvals", decomposition_failure)
    with pytest.raises(AnalysisError, match="matrix decomposition"):
        turnbull(lower, upper)
