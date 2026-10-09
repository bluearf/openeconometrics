"""Independent weighted recursions, autograd IJ and analytic survival identities."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext import competing


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture():
    return (np.array([1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7], float),
            np.array([1, 2, 0, 1, 0, 2, 1, 2, 0, 2, 0, 1]))


def weighted_numpy(time, event, causes, grid, weights, hazard=False):
    survival = 1.
    estimate = np.zeros(len(causes))
    result = []
    for horizon in grid:
        survival = 1.
        estimate[:] = 0
        for u in sorted(set(time[time <= horizon])):
            y = weights[time >= u].sum()
            jumps = np.array([weights[(time == u) & (event == cause)].sum()/y for cause in causes])
            estimate += jumps if hazard else survival*jumps
            survival *= 1-weights[(time == u) & (event > 0)].sum()/y
        result.extend(estimate)
    return np.array(result)


def finite_difference_oracle(time, event, causes, grid, hazard=False):
    weights = np.ones(len(time))
    point = weighted_numpy(time, event, causes, grid, weights, hazard)
    derivative = []
    step = 2e-5
    for i in range(len(time)):
        plus, minus = weights.copy(), weights.copy()
        plus[i] += step
        minus[i] -= step
        derivative.append((weighted_numpy(time, event, causes, grid, plus, hazard)
                           -weighted_numpy(time, event, causes, grid, minus, hazard))/(2*step))
    derivative = np.column_stack(derivative)
    return point, derivative@derivative.T


def autograd_oracle(time, event, causes, grid, hazard=False):
    t = torch.tensor(time, dtype=torch.float64, device="cpu")
    e = torch.tensor(event, dtype=torch.int64, device="cpu")
    def estimates(weights):
        result = []
        for horizon in grid:
            survival = torch.ones((), dtype=torch.float64, device="cpu")
            value = torch.zeros(len(causes), dtype=torch.float64, device="cpu")
            for u in sorted(set(time[time <= horizon])):
                at_risk = weights[t >= u].sum()
                jumps = torch.stack([weights[(t == u) & (e == cause)].sum()/at_risk for cause in causes])
                value = value+jumps if hazard else value+survival*jumps
                survival = survival*(1-weights[(t == u) & (e > 0)].sum()/at_risk)
            result.append(value)
        return torch.cat(result)
    w = torch.ones(len(time), dtype=torch.float64, device="cpu", requires_grad=True)
    derivative = torch.autograd.functional.jacobian(estimates, w)
    return estimates(w).detach().numpy(), (derivative@derivative.T).detach().numpy()


@pytest.mark.parametrize("name,hazard", [("cumulative_incidence", False), ("cause_specific_hazard", True)])
@pytest.mark.parametrize("causes", [[1, 2], [2, 1], [1], [3, 1]])
@pytest.mark.parametrize("level", [.9, .95, .99])
def test_full_tied_cross_time_cause_ij_covariance_matches_independent_oracles(name, hazard, causes, level):
    t, e = fixture()
    grid = [0., .5, 1., 2., 3., 5.5, 7.]
    result = getattr(competing, name)(t, e, causes=causes, times=grid, level=level)
    values, covariance = autograd_oracle(t, e, causes, grid, hazard)
    np.testing.assert_allclose(result["curve"].estimate, values, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(result["covariance"], covariance, atol=3e-14, rtol=3e-14)
    point_fd, cov_fd = finite_difference_oracle(t, e, causes, grid, hazard)
    np.testing.assert_allclose(values, point_fd, atol=2e-14)
    np.testing.assert_allclose(covariance, cov_fd, atol=2e-10, rtol=2e-8)
    se = np.sqrt(covariance.diagonal())
    critical = stats.norm.ppf((1+level)/2)
    np.testing.assert_allclose(result["curve"].standard_error, se, atol=2e-14)
    expected_lower = np.where(se > 0, np.maximum(values-critical*se, 0), np.nan)
    np.testing.assert_allclose(result["curve"].ci_lower.to_numpy(dtype=float), expected_lower, atol=2e-14)
    upper = values+critical*se if hazard else np.minimum(values+critical*se, 1)
    np.testing.assert_allclose(result["curve"].ci_upper.to_numpy(dtype=float), np.where(se > 0, upper, np.nan), atol=2e-14)
    assert result.attrs["causes"] == causes
    assert result.attrs["derivatives_zero_sum_error"] < 2e-14
    assert len(result["risksets"]) == len(set(t))


@pytest.mark.parametrize("grid", [[1., 2., 4.], [0., 1., 3., 6.]])
def test_no_censoring_exact_empirical_multinomial_cross_time_covariance(grid):
    t = np.array([1, 1, 2, 3, 3, 4, 5, 6], float)
    e = np.array([1, 2, 1, 1, 2, 2, 1, 2])
    result = competing.cumulative_incidence(t, e, times=grid, causes=[1, 2])
    indicators = np.array([((t <= horizon) & (e == cause)).astype(float)
                           for horizon in grid for cause in (1, 2)])
    p = indicators.mean(axis=1)
    expected = (indicators@indicators.T/len(t)-p[:, None]*p[None, :])/len(t)
    np.testing.assert_allclose(result["curve"].estimate, p, atol=2e-15)
    np.testing.assert_allclose(result["covariance"], expected, atol=2e-15)
    if grid[-1] == 6:
        terminal = result["covariance"].to_numpy()[-2:, -2:]
        np.testing.assert_allclose(terminal@np.ones(2), 0, atol=2e-15)


def test_one_cause_ij_equals_greenwood_and_other_failures_affect_survival():
    t, e = fixture()
    binary = (e > 0).astype(int)
    grid = [1, 2, 3, 4, 6]
    result = competing.cumulative_incidence(t, binary, times=grid)
    expected, variance = [], []
    for horizon in grid:
        survival, greenwood = 1., 0.
        for u in sorted(set(t[t <= horizon])):
            risk = np.sum(t >= u)
            deaths = np.sum((t == u) & (binary > 0))
            survival *= 1-deaths/risk
            greenwood += deaths/(risk*(risk-deaths)) if deaths else 0.
        expected.append(1-survival)
        variance.append(survival*survival*greenwood)
    np.testing.assert_allclose(result["curve"].estimate, expected, atol=2e-15)
    np.testing.assert_allclose(result["curve"].standard_error**2, variance, atol=2e-15)
    real = competing.cumulative_incidence(t, e, causes=[1], times=grid)
    censored_competitors = competing.cumulative_incidence(t, (e == 1).astype(int), causes=[1], times=grid)
    assert np.any(np.abs(real["curve"].estimate-censored_competitors["curve"].estimate) > .01)


@pytest.mark.parametrize("name", ["cumulative_incidence", "cause_specific_hazard"])
def test_tied_censor_is_in_common_risk_set_and_exact_failure_ties_are_order_invariant(name):
    t, e = fixture()
    result = getattr(competing, name)(t, e, times=[1, 2, 4, 7])
    np.testing.assert_allclose(result["curve"].estimate[:2], [1/12, 1/12], atol=1e-15)
    assert result["risksets"].iloc[0].n_risk == 12
    assert result["risksets"].iloc[0].censored == 1
    order = np.array([2, 1, 0, 4, 3, 5, 7, 6, 8, 10, 9, 11])
    permuted = getattr(competing, name)(t[order], e[order], times=[1, 2, 4, 7])
    np.testing.assert_allclose(result["curve"].estimate, permuted["curve"].estimate, atol=2e-15)
    np.testing.assert_allclose(result["covariance"], permuted["covariance"], atol=2e-15)


def test_cif_fixed_time_group_contrast_and_joint_wald_from_independent_group_oracles():
    t, e = fixture()
    tb, eb = t+np.array([0, 0, 0, 0, .1, .1, .1, .1, 0, 0, 0, 0]), np.array([2, 1, 0, 1, 0, 1, 2, 2, 0, 1, 0, 2])
    grid = [1, 2, 4, 6]
    result = competing.cif_compare(np.r_[t, tb], np.r_[e, eb], ["A"]*len(t)+["B"]*len(tb), cause=1, times=grid)
    first, cov_first = autograd_oracle(t, e, [1], grid)
    second, cov_second = autograd_oracle(tb, eb, [1], grid)
    difference, covariance = second-first, cov_first+cov_second
    se = np.sqrt(covariance.diagonal())
    np.testing.assert_allclose(result["contrast"].difference, difference, atol=2e-14)
    np.testing.assert_allclose(result["covariance"], covariance, atol=2e-14)
    np.testing.assert_allclose(result["contrast"].z, difference/se, atol=2e-14)
    np.testing.assert_allclose(result["contrast"].p_value, 2*stats.norm.sf(np.abs(difference/se)), atol=2e-14)
    statistic = difference@np.linalg.solve(covariance, difference)
    joint = result["joint"].iloc[0]
    assert joint.identified
    assert joint.df == len(grid)
    np.testing.assert_allclose(joint.statistic, statistic, atol=2e-13)
    np.testing.assert_allclose(joint.p_value, stats.chi2.sf(statistic, len(grid)), atol=2e-14)
    permutation = np.arange(len(t)*2)[::-1]
    again = competing.cif_compare(np.r_[t, tb][permutation], np.r_[e, eb][permutation],
                                  np.array(["A"]*len(t)+["B"]*len(tb))[permutation], cause=1, times=grid)
    np.testing.assert_allclose(again["contrast"].difference, difference, atol=2e-14)
    np.testing.assert_allclose(again["covariance"], covariance, atol=2e-14)


@pytest.mark.parametrize("times", [[0, .5, 1], [1, 1.5, 2]])
def test_singular_joint_covariance_has_no_generalized_inverse_test(times):
    t, e = fixture()
    result = competing.cif_compare(np.r_[t, t], np.r_[e, e], [0]*len(t)+[1]*len(t), cause=1, times=times)
    joint = result["joint"].iloc[0]
    assert not joint.identified
    assert joint.numerical_rank < len(times)
    assert joint.statistic is None and joint.p_value is None
    if times[0] == 0:
        assert result["contrast"].iloc[0].inference_status.startswith("unavailable")
        assert np.isnan(result["contrast"].iloc[0].p_value)


@pytest.mark.parametrize("name", ["cumulative_incidence", "cause_specific_hazard", "cif_compare"])
def test_json_complete_inputs_replay_latex_and_global_torch_defaults(name):
    t, e = fixture()
    args = (t, e) if name != "cif_compare" else (np.r_[t, t], np.r_[e, e], ["A"]*len(t)+["B"]*len(t))
    options = dict(times=[1, 2, 4, 6])
    if name == "cif_compare":
        options["cause"] = 1
    method = getattr(competing, name)
    result = method(*args, **options)
    saved = json.loads(json.dumps(result.attrs, allow_nan=False))
    replay_args = (saved["time"], saved["event"])
    replay_options = dict(times=saved["times"], level=saved["level"])
    if name == "cif_compare":
        replay_args += (saved["group"],)
        replay_options["cause"] = saved["cause"]
    else:
        replay_options["causes"] = saved["causes"]
    replay = method(*replay_args, **replay_options)
    np.testing.assert_array_equal(result["covariance"], replay["covariance"])
    assert "settings" in result.to_latex() and "covariance" in result.to_latex()
    old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
    torch.set_default_device("meta")
    torch.set_default_dtype(torch.float32)
    try:
        default = method(*args, **options)
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)
    np.testing.assert_array_equal(result["covariance"], default["covariance"])


@pytest.mark.parametrize("option", [dict(causes=[0]), dict(causes=[True]), dict(causes=[1, 1]), dict(causes=[1.5]),
                                    dict(times=[2, 1]), dict(times=[1, 1]), dict(times=[8]), dict(level=1),
                                    dict(weights=[1]*12), dict(device="cuda")])
def test_explicit_invalid_options_and_refusals(option):
    t, e = fixture()
    with pytest.raises(AnalysisError):
        competing.cumulative_incidence(t, e, **option)


@pytest.mark.parametrize("bad", [[0]+[1]*11, [float("nan")]+[1]*11, [True]+[1]*11, [[1]]*12])
def test_bad_complete_time_vector_has_no_silent_row_drop(bad):
    _, e = fixture()
    with pytest.raises(AnalysisError):
        competing.cumulative_incidence(bad, e)


def test_all_censored_absent_cause_and_empty_default_causes():
    with pytest.raises(AnalysisError, match="supply"):
        competing.cumulative_incidence([1, 2, 3], [0, 0, 0])
    for name in ("cumulative_incidence", "cause_specific_hazard"):
        result = getattr(competing, name)([1, 2, 3], [0, 0, 0], causes=[1], times=[0, 1, 3])
        np.testing.assert_array_equal(result["curve"].estimate, 0)
        np.testing.assert_array_equal(result["covariance"], 0)
        assert result["curve"].ci_lower.isna().all()
        assert result["curve"].ci_upper.isna().all()
        assert result["curve"].inference_status.str.startswith("unavailable").all()


def test_complete_curve_resource_limits_and_maximum_sample():
    t = np.arange(1, 4097, dtype=float)
    e = np.ones(4096, dtype=int)
    with pytest.raises(AnalysisError, match="256"):
        competing.cumulative_incidence(t, e)
    result = competing.cumulative_incidence(t, e, times=[100, 1000, 4000])
    np.testing.assert_allclose(result["curve"].estimate, np.array([100, 1000, 4000])/4096, atol=2e-13)
    assert len(result["risksets"]) == 4096
    with pytest.raises(AnalysisError, match="4096"):
        competing.cumulative_incidence(np.r_[t, 4097], np.r_[e, 1], times=[1])
    with pytest.raises(AnalysisError, match="256"):
        competing.cumulative_incidence(t, e, causes=[1, 2], times=np.arange(1, 130))


@pytest.mark.parametrize("group", [["A"]*12, ["A"]+ ["B"]*11, ["A"]*4+["B"]*4+["C"]*4, [None]*12])
def test_compare_rejects_unidentified_or_missing_groups(group):
    t, e = fixture()
    with pytest.raises(AnalysisError):
        competing.cif_compare(t, e, group, cause=1, times=[1])


def test_comparison_support_and_required_prespecified_horizons():
    t, e = fixture()
    with pytest.raises(AnalysisError, match="both"):
        competing.cif_compare(np.r_[t, t/2], np.r_[e, e], ["A"]*12+["B"]*12, cause=1, times=[5])
    with pytest.raises(AnalysisError, match="explicit"):
        competing.cif_compare(np.r_[t, t], np.r_[e, e], ["A"]*12+["B"]*12, cause=1, times=None)


def test_large_integer_group_identifiers_remain_distinct_and_exact():
    t, e = fixture()
    labels = [2**60]*len(t)+[2**60+1]*len(t)
    result = competing.cif_compare(np.r_[t, t], np.r_[e, e], labels, cause=1, times=[1, 2])
    assert result.attrs["group_labels"] == [2**60, 2**60+1]
    assert result.attrs["group"] == labels
    assert result.attrs["group_sizes"] == [12, 12]
    json.dumps(result.attrs, allow_nan=False)


def test_unbounded_integer_cause_is_structured_refusal():
    t, e = fixture()
    with pytest.raises(AnalysisError, match="cause"):
        competing.cumulative_incidence(t, e, causes=[10**10000])


@pytest.mark.parametrize("n", [3, 7, 31])
def test_single_cause_terminal_empirical_boundary_has_exact_zero_ij_and_no_ci(n):
    result = competing.cumulative_incidence(list(range(1, n+1)), [1]*n, times=[n])
    row = result["curve"].iloc[0]
    assert row.estimate == 1.
    assert row.standard_error == 0.
    assert row.ci_lower is None and row.ci_upper is None
    assert row.inference_status.startswith("unavailable")
