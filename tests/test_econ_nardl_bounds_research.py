"""Independent numerical oracles, NOT a proof of bootstrap inferential validity."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks.research import nardl_bounds as research
from benchmarks.research import nardl_bounds_grid as simulation


def sample(seed=8413, total=76):
    random = np.random.default_rng(seed)
    x = 13 + random.normal(size=total).cumsum()
    y = 4 + random.normal(size=total).cumsum()
    return y, x


def np_signs(raw, anchors=(0., 0.)):
    change = np.r_[0., np.diff(raw)]
    return (anchors[0] + np.maximum(change, 0).cumsum(),
            anchors[1] + np.minimum(change, 0).cumsum())


def np_design(y, positive, negative, p, qp, qn):
    hold, total = max(p, qp, qn), len(y)
    columns = [np.ones(total-hold)]
    columns += [y[hold-i:total-i] for i in range(1, p+1)]
    columns += [positive[hold-i:total-i] for i in range(qp+1)]
    columns += [negative[hold-i:total-i] for i in range(qn+1)]
    return np.column_stack(columns), y[hold:]


def np_restrictions(p, qp, qn, null):
    rows = np.zeros((3, p+qp+qn+3))
    rows[0, 1:p+1] = 1
    rows[1, p+1:p+qp+2] = 1
    rows[2, p+qp+2:] = 1
    index = {None: [], "joint_levels": [0, 1, 2], "adjustment": [0],
             "explanatory_levels": [1, 2]}[null]
    return rows[index], np.array([1., 0., 0.])[index]


def np_fit(matrix, target, p, qp, qn, null=None):
    """Full parameter-space affine SVD, independent of profiled/scaled Torch QR."""
    r, value = np_restrictions(p, qp, qn, null)
    if len(r):
        anchor = np.linalg.lstsq(r, value, rcond=None)[0]
        _, _, right = np.linalg.svd(r, full_matrices=True)
        basis = right[len(r):].T
    else:
        anchor, basis = np.zeros(matrix.shape[1]), np.eye(matrix.shape[1])
    free = matrix @ basis
    left, singular, right = np.linalg.svd(free, full_matrices=False)
    coordinates = right.T @ ((left.T @ (target - matrix @ anchor))/singular)
    coefficients = anchor + basis @ coordinates
    residuals = target - matrix @ coefficients
    df = len(target) - basis.shape[1]
    factor = basis @ (right.T / singular)
    covariance = (residuals @ residuals/df) * (factor @ factor.T)
    return coefficients, covariance, residuals, df


def np_statistics(fitted, p, qp, qn):
    coefficient, covariance, _, _ = fitted
    rows, target = np_restrictions(p, qp, qn, "joint_levels")
    delta, variance = rows @ coefficient-target, rows @ covariance @ rows.T
    return np.array([delta @ np.linalg.solve(variance, delta)/3,
                     delta[0]/np.sqrt(variance[0, 0]),
                     delta[1:] @ np.linalg.solve(variance[1:, 1:], delta[1:])/2])


@pytest.mark.parametrize("lags", [(1, 1, 1), (2, 2, 1), (3, 1, 2), (1, 3, 4)])
@pytest.mark.parametrize("null", [None, *research.NULLS])
def test_affine_fits_covariance_and_levels_statistics_match_independent_svd(lags, null):
    y, x = sample()
    xp, xn = np_signs(x)
    matrix, target = np_design(y, xp, xn, *lags)
    expected = np_fit(matrix, target, *lags, null)
    actual = research.fit_affine(torch.tensor(matrix)[None], torch.tensor(target)[None],
                                 *research.restrictions(*lags, null))
    np.testing.assert_allclose(actual.coefficients[0], expected[0], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(actual.covariance[0], expected[1], rtol=3e-10, atol=2e-11)
    np.testing.assert_allclose(actual.residuals[0], expected[2], rtol=1e-10, atol=2e-11)
    assert actual.df == expected[3]
    rows, target = np_restrictions(*lags, null)
    np.testing.assert_allclose(rows @ actual.coefficients[0].numpy(), target, atol=2e-12)
    if null is None:
        np.testing.assert_allclose(research.statistics(actual, *lags)[0],
                                   np_statistics(expected, *lags), rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("lags", [(1, 1, 1), (2, 2, 1), (3, 1, 2)])
@pytest.mark.parametrize("options", [
    {"initial": "fixed_prefix", "recenter": "pool", "residual_scale": "df"},
    {"initial": "original_block", "recenter": "draw", "residual_scale": "none"},
])
def test_joint_raw_path_null_recursion_and_every_refit_match_numpy(lags, options):
    y, x = sample(total=68)
    p, qp, qn = lags
    hold, rows = max(lags), len(y)-max(lags)
    original_positive, original_negative = np_signs(x)
    original_matrix, original_target = np_design(y, original_positive, original_negative, *lags)
    result = research.research_bootstrap(y, x, p=p, q_positive=qp, q_negative=qn,
                                         replications=7, batch_size=3, seed=718,
                                         retain_paths=True, **options)
    original = np_fit(original_matrix, original_target, *lags)
    np.testing.assert_allclose(result["observed_statistics"], np_statistics(original, *lags),
                               rtol=3e-10, atol=2e-11)
    increments = x[hold:] - x[hold-1:-1]
    drift = increments.mean()
    marginal = increments-drift
    streams = {}
    for null in research.NULLS:
        constrained = np_fit(original_matrix, original_target, *lags, null)
        np.testing.assert_allclose(result["nulls"][null]["restricted_coefficients"], constrained[0],
                                   rtol=3e-10, atol=2e-11)
        residual = constrained[2]-constrained[2].mean()
        yscale = np.sqrt(rows/constrained[3]) if options["residual_scale"] == "df" else 1.
        xscale = np.sqrt(rows/(rows-1)) if options["residual_scale"] == "df" else 1.
        pool = np.column_stack((residual*yscale, marginal*xscale))
        coefficients = constrained[0]
        expected_statistics, cursor, indices_all = [], 0, []
        for block in result["paths"][null]:
            for i, indices in enumerate(block["indices"]):
                paired = pool[indices].copy()
                if options["recenter"] == "draw":
                    paired -= paired.mean(0)
                start = result["initial_block_starts"][cursor]
                raw = np.r_[x[start:start+hold], x[start+hold-1] + np.cumsum(drift+paired[:, 1])]
                xp, xn = np_signs(raw, (original_positive[start], original_negative[start]))
                simulated = np.empty(len(y))
                simulated[:hold] = y[start:start+hold]
                for t in range(hold, len(y)):
                    value, position = coefficients[0]+paired[t-hold, 0], 1+p
                    for variable, q in ((xp, qp), (xn, qn)):
                        for lag in range(q+1):
                            value += coefficients[position]*variable[t-lag]
                            position += 1
                    for lag in range(1, p+1):
                        value += coefficients[lag]*simulated[t-lag]
                    simulated[t] = value
                np.testing.assert_allclose(block["paired_innovations"][i], paired, atol=2e-11)
                np.testing.assert_allclose(block["raw"][i], raw, rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(block["positive"][i], xp, rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(block["negative"][i], xn, rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(block["y"][i], simulated, rtol=2e-10, atol=5e-10)
                np.testing.assert_allclose(raw-raw[0], (xp-xp[0])+(xn-xn[0]), atol=2e-12)
                assert np.all(np.diff(xp) >= 0) and np.all(np.diff(xn) <= 0)
                expected_statistics.append(np_statistics(np_fit(*np_design(simulated, xp, xn, *lags), *lags), *lags))
                indices_all.append(indices)
                cursor += 1
        np.testing.assert_allclose(result["draw_statistics"][null], expected_statistics,
                                   rtol=1e-9, atol=2e-9)
        streams[null] = indices_all
    assert streams[research.NULLS[0]] == streams[research.NULLS[1]] == streams[research.NULLS[2]]
    assert result["inferential_validity_established"] is False
    assert result["private_research_only"] is True
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("initial", ["fixed_prefix", "original_block"])
def test_batch_invariance_local_rng_and_input_preservation(initial):
    y, x = sample()
    originals = y.copy(), x.copy()
    state = torch.random.get_rng_state().clone()
    options = dict(replications=11, seed=19, initial=initial, p=2, q_positive=2, q_negative=1)
    first = research.research_bootstrap(y, x, batch_size=1, **options)
    second = research.research_bootstrap(y, x, batch_size=4, **options)
    third = research.research_bootstrap(y, x, batch_size=11, **options)
    assert torch.equal(state, torch.random.get_rng_state())
    assert first["index_stream_sha256"] == second["index_stream_sha256"] == third["index_stream_sha256"]
    assert first["initial_block_starts"] == second["initial_block_starts"] == third["initial_block_starts"]
    for null in research.NULLS:
        np.testing.assert_allclose(first["draw_statistics"][null], second["draw_statistics"][null], rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(first["draw_statistics"][null], third["draw_statistics"][null], rtol=2e-11, atol=2e-11)
    np.testing.assert_array_equal(y, originals[0])
    np.testing.assert_array_equal(x, originals[1])


@pytest.mark.parametrize("change,code", [
    ({"p": 0}, "invalid_input"), ({"q_positive": 0}, "invalid_input"),
    ({"q_negative": True}, "invalid_input"), ({"replications": 2}, "invalid_input"),
    ({"seed": -1}, "invalid_input"), ({"alpha": .5}, "invalid_input"),
    ({"initial": []}, "invalid_input"), ({"recenter": "quiet"}, "invalid_input"),
    ({"time": [1]*76}, "invalid_time"), ({"time": [float(i) for i in range(76)]}, "invalid_time"),
    ({"time": "wrong"}, "invalid_time"), ({"batch_size": 0}, "invalid_input"),
])
def test_explicit_input_guards(change, code):
    y, x = sample()
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, **change)
    assert error.value.code == code


def test_rank_nan_and_time_guards_do_not_silently_sort_or_drop():
    y, x = sample()
    with pytest.raises(research.ResearchFailure, match="rank|constant"):
        research.research_bootstrap(y, np.arange(len(y)))
    x[7] = np.nan
    with pytest.raises(research.ResearchFailure, match="finite"):
        research.research_bootstrap(y, x)
    y, x = sample()
    valid = research.research_bootstrap(y, x, time=list(range(703, 703+len(y))), replications=3)
    assert valid["rows_original"] == len(y)


def test_failed_draw_invalidates_the_whole_call_without_replacement(monkeypatch):
    y, x = sample()
    calls = []

    def fail(prefix, *args):
        calls.append(len(prefix))
        raise research.ResearchFailure("nonfinite", "Injected numerical failure.", local_indices=[1])

    monkeypatch.setattr(research, "generate_y", fail)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, replications=7, batch_size=3)
    assert calls == [3]
    assert error.value.details == {"null": "joint_levels", "replicate_indices": [1],
                                  "completed_batches": 0, "invalid_draws_replaced": 0,
                                  "completed_all_null_replications": 0,
                                  "whole_call_invalid": True}


def test_tail_conventions_are_explicit_discrete_monte_carlo_not_claimed_validity():
    values = torch.tensor([1., 2., 2., 3., 5.], dtype=torch.float64)
    upper = research._tail_report(2., values, "upper", .1)
    lower = research._tail_report(2., values, "lower", .1)
    assert upper["extreme_draws"] == 4 and lower["extreme_draws"] == 3
    assert upper["research_tail_probability_plus_one"] == 5/6
    assert lower["research_tail_probability_plus_one"] == 4/6
    assert upper["zero_based_order_rank"] == math.ceil(.9*5)-1
    assert lower["zero_based_order_rank"] == 0


def test_budget_limits_prevent_unbounded_research_work():
    y, x = sample(total=500)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, p=4, q_positive=4, q_negative=4, replications=20000)
    assert error.value.code == "work_limit"
    y, x = sample(total=1500)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, replications=3, memory_mb=1)
    assert error.value.code == "memory_limit"


@pytest.mark.parametrize("lags", [(1, 1, 1), (2, 2, 1), (3, 1, 2)])
@pytest.mark.parametrize("residual_scale", ["none", "df"])
def test_conditional_f_benchmark_uses_original_paths_and_unrestricted_residuals(lags, residual_scale):
    y, x = sample(total=71)
    p, qp, qn = lags
    hold = max(lags)
    xp, xn = np_signs(x)
    matrix, target = np_design(y, xp, xn, *lags)
    unrestricted = np_fit(matrix, target, *lags)
    coefficients = np_fit(matrix, target, *lags, "joint_levels")[0]
    pool = unrestricted[2] - unrestricted[2].mean()
    if residual_scale == "df":
        pool *= np.sqrt(len(pool)/unrestricted[3])
    result = research.conditional_f_benchmark(y, x, p=p, q_positive=qp, q_negative=qn,
                                             replications=7, seed=4981, batch_size=3,
                                             residual_scale=residual_scale, retain_paths=True)
    expected = []
    for block in result["paths"]:
        for i, indices in enumerate(block["indices"]):
            errors = pool[indices]
            simulated = np.empty(len(y))
            simulated[:hold] = y[:hold]
            for t in range(hold, len(y)):
                value = coefficients[0]+errors[t-hold]
                position = 1+p
                for values, q in ((xp, qp), (xn, qn)):
                    for lag in range(q+1):
                        value += coefficients[position]*values[t-lag]
                        position += 1
                for lag in range(1, p+1):
                    value += coefficients[lag]*simulated[t-lag]
                simulated[t] = value
            np.testing.assert_allclose(block["errors"][i], errors, atol=2e-11)
            np.testing.assert_allclose(block["y"][i], simulated, rtol=1e-10, atol=2e-10)
            expected.append(np_statistics(np_fit(*np_design(simulated, xp, xn, *lags), *lags), *lags)[0])
    np.testing.assert_allclose(result["draw_statistics"], expected, rtol=1e-9, atol=2e-9)
    assert result["residual_pool"].startswith("UNRESTRICTED")
    assert result["exact_paper_simulation_reproduction"] is False
    assert "tests" not in result and result["inferential_validity_established"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("cell", simulation.GRID, ids=lambda cell: cell.name)
def test_predeclared_grid_dgp_levels_recur_against_independent_numpy(cell):
    seed, burn = 7439, 64
    y, raw, metadata = simulation.generate_sample(cell, seed, burn)
    p, qp, qn = cell.lags
    generator = torch.Generator().manual_seed(seed)
    count = cell.total+burn
    if cell.skew:
        uniform = torch.rand(count, generator=generator, dtype=torch.float64)
        u = -np.log(uniform.numpy())-1
    else:
        u = torch.randn(count, generator=generator, dtype=torch.float64).numpy()
    v = torch.randn(count, generator=generator, dtype=torch.float64).numpy()
    raw_expected = np.r_[0., np.cumsum(cell.drift+u[1:])]
    np.testing.assert_allclose(raw, raw_expected[burn:], rtol=1e-12, atol=2e-12)
    xp, xn = np_signs(raw_expected)
    coefficients = np.array(metadata["level_coefficients_full_dgp"])
    outcome = np.zeros(count)
    errors = np.sqrt(1-cell.structural_contemporaneous_correlation**2)*v
    for t in range(max(cell.lags), count):
        value, position = coefficients[0]+errors[t], 1+p
        for values, q in ((xp, qp), (xn, qn)):
            for lag in range(q+1):
                value += coefficients[position]*values[t-lag]
                position += 1
        for lag in range(1, p+1):
            value += coefficients[lag]*outcome[t-lag]
        outcome[t] = value
    np.testing.assert_allclose(y, outcome[burn:], rtol=2e-11, atol=3e-10)
    assert abs(sum(coefficients[1:1+p])-1-cell.rho) < 1e-15
    assert abs(sum(coefficients[1+p:2+p+qp])-cell.theta_positive) < 1e-15
    assert abs(sum(coefficients[2+p+qp:])-cell.theta_negative) < 1e-15
    anchors = metadata["signed_level_anchors_at_observed_start"]
    assert metadata["observed_levels_intercept"] == pytest.approx(coefficients[0]+cell.theta_positive*anchors[0]+cell.theta_negative*anchors[1])


def test_whole_call_failures_remain_in_the_predeclared_outer_denominator(monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs["seed"])
        raise research.ResearchFailure("rank", "Injected failed whole-call simulation.",
                                       invalid_draws_replaced=0, replicate_indices=[3])

    monkeypatch.setattr(simulation, "research_bootstrap", failed)
    result = simulation.run_smoke(outer=2, inner=19)
    assert result["planned_outer_calls"] == 2*len(simulation.GRID) == len(calls)
    assert result["failed_outer_calls"] == len(calls) and result["completed_outer_calls"] == 0
    for cell in result["cells"]:
        assert cell["failed_outer_calls"] == 2
        for rate in cell["rates"].values():
            assert rate["planned_outer_denominator"] == 2
            assert rate["rate_bounds_with_failed_calls"] == [0., 1.]
    assert result["inferential_validity_established"] is False
    assert result["acceptance_evaluated"] is False
    assert result["invalid_draws_replaced"] == 0


def test_grid_small_numerical_smoke_has_no_premature_validity_claim():
    result = simulation.run_smoke(outer=1, inner=19)
    assert result["planned_outer_calls"] == len(simulation.GRID)
    assert result["completed_outer_calls"] + result["failed_outer_calls"] == len(simulation.GRID)
    assert result["acceptance_evaluated"] is False
    assert result["inferential_validity_established"] is False
    outside = [cell for cell in result["cells"] if not cell["specification"]["within_maintained_class"]]
    assert len(outside) == 1 and "I(2)" in outside[0]["dgp"]["outcome_order_scope"]
    for cell in result["cells"]:
        assert set(cell["rates"]) == {*research.NULLS, "all_three"}
    json.dumps(result, allow_nan=False)


def test_time_integer_extrema_are_exact_not_wrapped_or_rejected_by_unsigned_diff():
    y, x = sample()
    largest = 2**63-1
    wrapped = torch.tensor([largest, *range(-2**63, -2**63+75)], dtype=torch.int64)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, time=wrapped, replications=3)
    assert error.value.code == "invalid_time"
    for axis in (np.arange(len(y), dtype=np.uint64)+np.uint64(2**63+7),
                 np.arange(len(y), dtype=np.int64)+np.int64(largest-len(y)+1)):
        result = research.research_bootstrap(y, x, time=axis, replications=3)
        assert result["rows_original"] == len(y)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, time=np.arange(len(y), dtype=np.complex128), replications=3)
    assert error.value.code == "invalid_time"


def test_native_original_null_factorization_failure_is_structured(monkeypatch):
    y, x = sample()
    original, calls = research._fit_affine_unchecked, []

    def failed(matrix, target, restriction, imposed):
        calls.append(len(restriction))
        if len(restriction):
            raise RuntimeError("Injected native SVD convergence failure.")
        return original(matrix, target, restriction, imposed)

    monkeypatch.setattr(research, "_fit_affine_unchecked", failed)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, replications=3)
    assert calls == [0, 3]
    assert error.value.code == "torch_numerical"
    assert error.value.details["null"] == "joint_levels"
    assert error.value.details["whole_call_invalid"] is True
    assert error.value.details["invalid_draws_replaced"] == 0


def test_native_bootstrap_factorization_failure_keeps_replicate_indices(monkeypatch):
    y, x = sample()
    original = research._fit_affine_unchecked

    def failed(matrix, target, restriction, imposed):
        if matrix.shape[0] > 1:
            raise RuntimeError("Injected batched QR convergence failure.")
        return original(matrix, target, restriction, imposed)

    monkeypatch.setattr(research, "_fit_affine_unchecked", failed)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, replications=7, batch_size=3)
    assert error.value.code == "torch_numerical"
    assert error.value.details["replicate_indices"] == [0, 1, 2]
    assert error.value.details["invalid_draws_replaced"] == 0
    assert error.value.details["whole_call_invalid"] is True


def test_native_runtime_failure_in_a_later_batch_records_completed_counts(monkeypatch):
    y, x = sample()
    original, calls = research.generate_y, []

    def failed(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("Injected forcing/native runtime failure.")
        return original(*args, **kwargs)

    monkeypatch.setattr(research, "generate_y", failed)
    with pytest.raises(research.ResearchFailure) as error:
        research.research_bootstrap(y, x, replications=7, batch_size=3)
    assert error.value.code == "torch_numerical"
    assert error.value.details["replicate_indices"] == [3, 4, 5]
    assert error.value.details["completed_batches"] == 1
    assert error.value.details["completed_all_null_replications"] == 3
    assert error.value.details["invalid_draws_replaced"] == 0


def test_outer_generation_runtime_failures_remain_in_all_denominators(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("Injected outer DGP failure.")

    monkeypatch.setattr(simulation, "generate_sample", failed)
    result = simulation.run_smoke(outer=1, inner=19)
    assert result["failed_outer_calls"] == result["planned_outer_calls"] == len(simulation.GRID)
    assert all(cell["failures"][0]["code"] == "torch_numerical" for cell in result["cells"])


@pytest.mark.parametrize("function,replications", [(research.research_bootstrap, 200),
                                                  (research.conditional_f_benchmark, 500)])
def test_retained_payload_cap_includes_innovations_and_sampled_indices(function, replications):
    y, x = sample()
    with pytest.raises(research.ResearchFailure) as error:
        function(y, x, replications=replications, memory_mb=1, retain_paths=True)
    assert error.value.code == "memory_limit"
    assert error.value.details["invalid_draws_replaced"] == 0


def test_required_high_order_unit_root_recursion_is_not_screened_or_power_squared():
    random = np.random.default_rng(6421)
    total, hold = 1025, 4
    raw = random.normal(size=total).cumsum()
    xp, xn = np_signs(raw)
    # Four repeated unit roots are a deliberate numerical stress, not an I(1)
    # calibration DGP. Direct native recursion must still obey its equation.
    coefficients = np.array([.3, 4., -6., 4., -1., .2, -.2, -.1, .1])
    errors = random.normal(size=(2, total-hold))
    initial = random.normal(size=(2, hold))
    expected = np.empty((2, total))
    expected[:, :hold] = initial
    forcing = np.repeat(coefficients[:1], total-hold)
    for c, values, lag in ((.2, xp, 0), (-.2, xp, 1), (-.1, xn, 0), (.1, xn, 1)):
        forcing += c*values[hold-lag:total-lag]
    for i in range(2):
        for t in range(hold, total):
            value = forcing[t-hold]+errors[i, t-hold]
            for lag in range(1, 5):
                value += coefficients[lag]*expected[i, t-lag]
            expected[i, t] = value
    actual = research.generate_y(torch.tensor(initial), torch.tensor(xp)[None].expand(2, -1),
                                 torch.tensor(xn)[None].expand(2, -1), torch.tensor(coefficients),
                                 4, 1, 1, torch.tensor(errors))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["torch", "numpy", "pandas", "pandas_object", "sequence", "numpy_scalar_sequence"])
@pytest.mark.parametrize("argument", ["y", "x"])
@pytest.mark.parametrize("imaginary", [0., 1.])
def test_complex_inputs_are_rejected_before_cast_and_fingerprint(kind, argument, imaginary, monkeypatch):
    y, x = sample()
    values = (y if argument == "y" else x).astype(np.complex128)+imaginary*1j
    # Reject complex dtype even with an all-zero imaginary component.
    converted = {"torch": lambda: torch.tensor(values), "numpy": lambda: values,
                 "pandas": lambda: pd.Series(values),
                 "pandas_object": lambda: pd.Series(list(values), dtype=object),
                 "sequence": lambda: values.tolist(),
                 "numpy_scalar_sequence": lambda: list(values)}[kind]()

    def fingerprint_should_never_run(*args):
        raise AssertionError("Rejected complex input reached fingerprinting.")

    monkeypatch.setattr(research, "_fingerprint", fingerprint_should_never_run)
    with pytest.raises(research.ResearchFailure, match="complex") as error:
        research.research_bootstrap(converted if argument == "y" else y,
                                    converted if argument == "x" else x, replications=3)
    assert error.value.code == "invalid_input"


@pytest.mark.parametrize("value", [[.12345678901234567, 1e308], [2**70, 2**70+2**30],
                                 np.array([.12345678901234567, 1e308]),
                                 pd.Series([.12345678901234567, 1e308])])
def test_real_vector_conversion_keeps_original_float64_values(value):
    expected = torch.as_tensor(value, dtype=torch.float64)
    actual = research._vector(value, "real")
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float64
