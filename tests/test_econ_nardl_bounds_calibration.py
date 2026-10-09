"""Scientific gate/denominator regressions, without running the full matrix."""

from dataclasses import asdict
import json

import pytest
import numpy as np
import torch

from benchmarks.research import nardl_bounds_calibration as calibration
from benchmarks.research import nardl_bounds_numpy_oracle as oracle
from benchmarks.research import nardl_bounds_independent_size as independent_size
from benchmarks.research import nardl_bounds_receipt as receipt


def records(hits, count=500):
    return [
        {
            "tail_probabilities": {
                null: (0.001 if index < hits else 1.0) for null in calibration.NULLS
            }
        }
        for index in range(count)
    ]


def result(cell, hits=20, failed=0):
    failures = [{}] * failed
    return {
        "specification": asdict(cell),
        "failed_outer_calls": failed,
        "rates": {"0.05": calibration.rates(records(hits, 500 - failed), failures, 500, 0.05)},
    }


def test_protocol_json_roundtrip_source_pin_and_prespecified_dimensions():
    original = calibration.protocol()
    restored = json.loads(calibration.encoded(original))
    assert calibration.digest(restored) == calibration.digest(original)
    assert restored["outer_per_cell"] >= 500
    assert restored["inner_per_null"] >= 999
    assert restored["alphas"] == [0.01, 0.05, 0.1]
    assert len(restored["grid"]) == 15
    assert len({run["seed"] for run in restored["runs"]}) == 2
    assert (
        restored["work_budget"]["per_run_refit_operation_proxy"]
        <= restored["work_budget"]["maximum_per_run"]
    )
    assert restored["inferential_validity_established"] is False
    assert restored["public_bounds_inference_enabled"] is False


def test_mutated_frozen_protocol_fails_before_any_dgp(monkeypatch):
    specification = json.loads(calibration.encoded(calibration.protocol()))
    specification["runs"][0]["seed"] += 1
    monkeypatch.setattr(
        calibration, "generate_sample", lambda *args: pytest.fail("DGP must not run")
    )
    with pytest.raises(ValueError, match="protocol or numeric sources changed"):
        calibration.run_calibration(specification, 0)


@pytest.mark.parametrize("run", [True, -1, 4, 0.5, "0"])
def test_run_selection_rejects_coercion_and_out_of_range(run):
    with pytest.raises(ValueError, match="Run index"):
        calibration.run_calibration(calibration.protocol(), run)


def test_failure_denominator_includes_all_failed_calls_and_no_retries():
    rates = calibration.rates(records(25, 497), [{}, {}, {}], 500, 0.05)
    for null in (*calibration.NULLS, "all_three"):
        item = rates[null]
        assert item["planned_outer_denominator"] == 500
        assert item["failed_outer_calls"] == 3
        assert item["rate_bounds_with_failed_calls"] == [25 / 500, 28 / 500]
        assert (
            item["wilson95_all_failed_calls_reject"][1] > item["wilson95_no_failed_call_rejects"][1]
        )
    total = calibration.rates([], [{}] * 500, 500, 0.05)
    assert total["joint_levels"]["rate_bounds_with_failed_calls"] == [0.0, 1.0]


def test_multiple_alpha_uses_same_discrete_tail_probabilities():
    sample = [{"tail_probabilities": dict(zip(calibration.NULLS, [0.01, 0.05, 0.1], strict=True))}]
    low, middle, high = [calibration.rates(sample, [], 1, alpha) for alpha in calibration.ALPHAS]
    assert [item["joint_levels"]["rejected_successful_calls"] for item in (low, middle, high)] == [
        1,
        1,
        1,
    ]
    assert [item["adjustment"]["rejected_successful_calls"] for item in (low, middle, high)] == [
        0,
        1,
        1,
    ]
    assert [item["all_three"]["rejected_successful_calls"] for item in (low, middle, high)] == [
        0,
        0,
        1,
    ]


def test_degenerate_stationary_outcome_only_explanatory_null_is_true():
    stationary = calibration.GRID[7]
    assert calibration.true_nulls(stationary) == ("explanatory_levels",)
    item = result(stationary, 450)
    for null in ("joint_levels", "adjustment", "all_three"):
        item["rates"]["0.05"][null]["wilson95_all_failed_calls_reject"] = [0.8, 0.99]
    item["rates"]["0.05"]["explanatory_levels"]["wilson95_all_failed_calls_reject"] = [0.02, 0.06]
    assert calibration.evaluate([item])["numerical_acceptance_passed"] is True


def test_i2_stress_is_explicitly_excluded_and_never_enables_inference():
    cell = calibration.GRID[-1]
    assert calibration.true_nulls(cell) == ()
    evaluated = calibration.evaluate([result(cell, 500)])
    assert evaluated["numerical_acceptance_passed"] is True
    assert evaluated["inferential_validity_established"] is False
    assert evaluated["acceptance_complete"] is False
    assert evaluated["public_bounds_inference_enabled"] is False


def test_predeclared_size_power_and_failure_gates_are_separate():
    acceptable = [
        result(cell, 450 if cell.name == "asymmetric_T160" else 20) for cell in calibration.GRID
    ]
    assert calibration.evaluate(acceptable)["numerical_acceptance_passed"] is True
    size = calibration.evaluate([result(calibration.GRID[0], 50)])
    assert {item["null"] for item in size["violations"]} == {*calibration.NULLS, "all_three"}
    power = calibration.evaluate([result(calibration.GRID[11], 200)])
    assert power["violations"][0]["gate"] == "power_at_05"
    failure = calibration.evaluate([result(calibration.GRID[-1], 0, 1)])
    assert failure["violations"][0]["gate"] == "whole_call_success"


@pytest.mark.parametrize("cell", calibration.GRID, ids=lambda cell: cell.name)
def test_independent_difference_ecm_outer_dgp_matches_every_predeclared_cell(cell):
    seed = 37691
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = cell.total + 64
    if cell.skew:
        uniform = torch.rand(total, generator=generator, dtype=torch.float64)
        raw_shock = -torch.log(uniform.clamp_min(torch.finfo(torch.float64).tiny)) - 1
    else:
        raw_shock = torch.randn(total, generator=generator, dtype=torch.float64)
    independent_shock = torch.randn(total, generator=generator, dtype=torch.float64)
    actual_y, actual_raw, _ = calibration.generate_sample(cell, seed)
    expected_y, expected_raw = oracle.sample_from_shocks(
        cell, raw_shock.numpy(), independent_shock.numpy()
    )
    np.testing.assert_allclose(actual_y.numpy(), expected_y, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual_raw.numpy(), expected_raw, rtol=2e-12, atol=2e-12)


def test_independent_full_affine_svd_replicates_all_three_null_tails_and_ecm():
    result = oracle.compare(replications=7)
    assert len(result["records"]) == 8
    assert all(row["all_tail_counts_identical"] for row in result["records"])
    assert all(row["independent_difference_ecm_dgp_matches"] for row in result["records"])
    assert result["inferential_validity_established"] is False
    assert result["not_a_full_independent_size_replication"] is True


def test_independent_size_diagnostic_keeps_all_failed_calls_in_denominator(monkeypatch):
    attempts = []
    monkeypatch.setattr(
        independent_size, "sample_from_shocks", lambda *args: (np.zeros(80), np.zeros(80))
    )

    def fail(*args, **kwargs):
        attempts.append(1)
        raise np.linalg.LinAlgError("Injected independent SVD failure")

    monkeypatch.setattr(independent_size, "bootstrap", fail)
    result = independent_size.run()
    assert len(attempts) == result["planned_outer_calls"] == 2000
    assert result["completed_outer_calls"] == 0
    assert result["failed_outer_calls"] == 2000
    assert result["invalid_draws_replaced"] == 0
    assert result["inferential_validity_established"] is False
    for run in result["runs"]:
        assert len(run["failures"]) == 500
        assert run["records"] == []
        for alpha in ("0.01", "0.05", "0.1"):
            assert run["rates"][alpha]["all_three"]["rate_bounds_with_failed_calls"] == [0.0, 1.0]


@pytest.mark.parametrize(
    "corruption", ["duplicate_call", "seed", "fake_probability", "wrong_rate", "omitted_call"]
)
def test_receipt_recomputes_every_count_seed_rate_and_denominator(corruption):
    run = calibration.RUNS[0]
    sample = records(20)
    for index, row in enumerate(sample):
        row.update(
            outer_index=index,
            outer_seed=run["seed"] + 1009 * index,
            inner_seed=run["seed"] + 1009 * index + 524287,
            extreme_draws={null: (0 if index < 20 else 999) for null in calibration.NULLS},
        )
    cell = {
        "specification": asdict(calibration.GRID[0]),
        "records": sample,
        "failures": [],
        "completed_outer_calls": 500,
        "failed_outer_calls": 0,
        "invalid_draws_replaced": 0,
        "rates": {
            str(alpha): calibration.rates(sample, [], 500, alpha) for alpha in calibration.ALPHAS
        },
    }
    receipt.validate_cell(cell, 0, run)
    if corruption == "duplicate_call":
        sample[0]["outer_index"] = 1
    elif corruption == "seed":
        sample[0]["inner_seed"] += 1
    elif corruption == "fake_probability":
        sample[0]["tail_probabilities"]["joint_levels"] = 0.005
    elif corruption == "wrong_rate":
        cell["rates"]["0.05"]["all_three"]["rejected_successful_calls"] += 1
    else:
        sample.pop()
    with pytest.raises(ValueError):
        receipt.validate_cell(cell, 0, run)
