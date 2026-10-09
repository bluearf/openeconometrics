"""Execution guarantees for complete empirical targets, separate from math oracles."""

from contextlib import nullcontext
import json
import math

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from openecon.econometrics.survey import regression_replication as implementation
from openecon.econometrics.survey import replicate_margins as target_implementation
from test_survey_probit_poisson_replication_guards import (
    failure_fixture, support_removal_weights,
)
from test_survey_replicate_margins_math import fixture, fit


FAMILIES = ("logit", "probit", "poisson")
METHODS = ("brr", "fay", "jackknife", "bootstrap")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", ("mean", "ame"))
def test_original_and_every_replica_fit_and_target_run_once_without_refit_on_restore(
    monkeypatch, family, method, target,
):
    frame, design, _ = fixture()
    fitting, evaluating = [], []
    original_fit = implementation.fit_sample
    original_evaluate = target_implementation._TargetTrace.evaluate

    def record_fit(sample, *args, **kwargs):
        fitting.append((sample.weights.tolist(), torch.where(sample.selected)[0].tolist()))
        return original_fit(sample, *args, **kwargs)

    def record_target(trace, sample, fitted, *, replicate_id, ledger):
        evaluating.append(replicate_id)
        return original_evaluate(trace, sample, fitted, replicate_id=replicate_id, ledger=ledger)

    monkeypatch.setattr(implementation, "fit_sample", record_fit)
    monkeypatch.setattr(target_implementation._TargetTrace, "evaluate", record_target)
    state = fit(frame, design, family, method, target)
    record = state.source_result.metadata["replication"]
    assert len(fitting) == record["replicate_count"] + 1
    assert evaluating == [None, *record["replicate_ids"]]
    primitive = state.metadata["primitive"]
    assert fitting[0][0] == primitive["base_weights"]
    assert [entry[0] for entry in fitting[1:]] == primitive["replicate_weights"]
    assert fitting[0][1] == primitive["sample_positions"]
    assert [entry[1] for entry in fitting[1:]] == [
        evidence["sample_positions"] for evidence in record["replicate_convergence"]
    ]

    def unexpected(*args, **kwargs):
        raise AssertionError("Saved empirical target inference started another coefficient fit.")

    monkeypatch.setattr(implementation, "fit_sample", unexpected)
    restored = oe.SurveyReplicateMarginsResult.from_state(state.to_state())
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())
    assert len(evaluating) == record["replicate_count"] + 1


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("rank", (False, True), ids=("separated-support", "rank-support"))
def test_complete_scientific_fit_failure_ids_remain_reported_in_empirical_api(family, rank):
    frame, design = failure_fixture("probit" if family == "logit" else family, rank=rank)
    if family == "logit":
        assert oe.survey_logit(frame, design, "b", ["x", "z"],
                               tolerance=1e-11).metadata["convergence"]["converged"] is True
    original = frame.copy(deep=True)
    weights = support_removal_weights(frame)
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, family, "bootstrap", "ame", supplied=weights, rscales=None,
            centering="original", scale=1/len(weights.columns))
    assert error.value.code == "survey_replicate_failure"
    failures = error.value.failed_replicates
    assert [entry["replicate_id"] for entry in failures] == ["remove-support-a", "remove-support-b"]
    code = "survey_regression_rank" if rank else "survey_regression_separation"
    assert all(entry["code"] == code and entry["stage"] == "fit" for entry in failures)
    assert not hasattr(error.value, "unattempted_replicate_ids")
    assert 0 < error.value.work_record["actual_work_units"] <= 50_000_000
    pd.testing.assert_frame_equal(frame, original)


def test_all_poisson_target_overflow_ids_are_reported_and_later_fits_continue(monkeypatch):
    frame, design, _ = fixture()
    fitted = []
    original_fit = implementation.fit_sample

    def record_fit(sample, *args, **kwargs):
        fitted.append(int(sample.selected.sum()))
        return original_fit(sample, *args, **kwargs)

    monkeypatch.setattr(implementation, "fit_sample", record_fit)
    # Base x coefficient ~.12 gives eta~600; the first two BRR x slopes
    # exceed .20, making this finite partial profile overflow only there.
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, "poisson", "brr", "mean", profiles={"stress": {"x": 5000.}})
    assert error.value.code == "survey_replicate_failure"
    failures = error.value.failed_replicates
    assert [entry["replicate_id"] for entry in failures] == ["sylvester-0", "sylvester-1"]
    assert all(entry["stage"] == "target" and entry["code"] == "survey_numerical_failure"
               for entry in failures)
    assert fitted == [48, 24, 24, 24, 24]
    assert not hasattr(error.value, "unattempted_replicate_ids")


def test_original_poisson_target_overflow_stops_before_any_replica_fit(monkeypatch):
    frame, design, _ = fixture()
    fitted = []
    original_fit = implementation.fit_sample

    def record_fit(sample, *args, **kwargs):
        fitted.append(int(sample.selected.sum()))
        return original_fit(sample, *args, **kwargs)

    monkeypatch.setattr(implementation, "fit_sample", record_fit)
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, "poisson", "brr", "mean", profiles={"stress": {"x": 10000.}})
    assert error.value.code == "survey_numerical_failure"
    assert fitted == [48]


def test_new_target_evaluation_and_restore_work_is_refused_before_weight_planner(monkeypatch):
    frame, _, _ = fixture()
    frame = pd.concat([frame, frame, frame], ignore_index=True)
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    sample = implementation.admit(frame, design, "c", ["x", "z"])
    # The coefficient-only request fits under its complete request budget.
    _, _, _, legacy_ledger = implementation._preflight(sample, "poisson", "brr", 100, 32, None)
    assert legacy_ledger.planned < 50_000_000

    def unexpected(*args, **kwargs):
        raise AssertionError("Target and saved replay work were admitted after weight allocation.")

    monkeypatch.setattr(implementation, "plan_brr", unexpected)
    profiles = {f"profile-{i}": {"x": i/32} for i in range(32)}
    with pytest.raises(AnalysisError) as error:
        fit(frame, design, "poisson", "brr", "mean", profiles=profiles, replicates=32)
    assert error.value.code == "survey_regression_budget"


def test_new_target_primitive_live_workspace_is_refused_before_weight_planner(monkeypatch):
    frame, design, _ = fixture()
    sample = implementation.admit(frame, design, "c", ["x", "z"])
    with use_workspace_budget(1):
        implementation._preflight(sample, "poisson", "brr", 100, 32, None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Exact target replay weights were allocated before combined workspace admission.")

    monkeypatch.setattr(implementation, "admit", lambda *args, **kwargs: sample)
    monkeypatch.setattr(implementation, "plan_brr", unexpected)
    profiles = {f"profile-{i}": {"x": i/32} for i in range(32)}
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit(frame, design, "poisson", "brr", "mean", profiles=profiles, replicates=32)
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("target", ("mean", "ame"))
@pytest.mark.parametrize("inference_mode", (False, True))
def test_fit_primitive_replay_and_inference_are_cpu_float64_under_ambient_defaults(
    family, target, inference_mode,
):
    frame, design, _ = fixture()
    expected = fit(frame, design, family, "fay", target)
    old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode() if inference_mode else nullcontext():
            actual = fit(frame, design, family, "fay", target)
            assert actual.to_state() == expected.to_state()
            restored = oe.SurveyReplicateMarginsResult.from_state(actual.to_state())
            pd.testing.assert_frame_equal(restored.to_frame(), expected.to_frame())
            assert restored.source_result.metadata["device"] == "cpu"
            assert restored.source_result.metadata["precision"] == "float64"
        assert torch.get_default_device().type == "meta"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("count_scale", (1_000_000, 1_000_000_000_000, 100_000_000_000_000))
@pytest.mark.parametrize("covariate_scale", (.017, 2.7, 7.3, 31.1))
def test_large_exact_integer_poisson_counts_and_rescaled_covariates_keep_valid_score_replay(
    count_scale, covariate_scale,
):
    frame, _, _ = fixture()
    original_x, original_z = frame.x.tolist(), frame.z.tolist()
    frame["c"] = [round(count_scale * math.exp(.2*x - .1*z))
                  for x, z in zip(original_x, original_z)]
    assert all(type(value) is int and 0 <= value <= 2**53 for value in frame.c.tolist())
    frame["x"] *= covariate_scale
    frame["z"] /= covariate_scale + 2.3
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    profiles = {"low": {"x": -.8*covariate_scale}, "high": {"x": 1.2*covariate_scale}}
    state = fit(frame, design, "poisson", "fay", "mean", profiles=profiles)
    # Rounding the generated means to exact admitted integer counts changes
    # the fitted target slightly. Its normalized value stays near the known
    # generating mean even when outcome and covariate units are very large.
    total_weight = math.fsum(frame.w)
    generating_means = [
        math.fsum(weight * math.exp(.2*at - .1*z) for weight, z in zip(frame.w, original_z))
        / total_weight for at in (-.8, 1.2)
    ]
    assert [value/count_scale for value in state.estimates] == pytest.approx(
        generating_means, rel=1e-7, abs=1e-10,
    )
    assert all(math.isfinite(value) for row in state.covariance for value in row)
    # A standard JSON encoder may sort all object keys; explicit saved target
    # labels, rather than object key order, preserve the vector and covariance.
    restored = oe.SurveyReplicateMarginsResult.from_state(json.dumps(state.to_state(), sort_keys=True))
    assert restored.to_state() == state.to_state()
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())
