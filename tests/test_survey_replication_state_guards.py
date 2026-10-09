"""Adversarial coefficient-replication restore and bounded execution guards."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from test_survey_regression_replication import fixture, fit


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def complete_states():
    frame, design, _ = fixture()
    return {
        f"{family}_{method}": fit(frame, design, family, method)
        for family in ("linear", "logit")
        for method in ("brr", "fay", "jackknife", "bootstrap")
    }


def rehash(payload):
    """A digest is not authentication: these tests deliberately recompute it."""
    record = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    payload["integrity_sha256"] = hashlib.sha256(json.dumps(
        record, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return payload


@pytest.mark.parametrize("family", ["linear", "logit", "probit", "poisson"])
def test_pre_replication_native_states_keep_exact_legacy_digest_and_inference(family):
    """Previously installed native artifacts are real legacy compatibility fixtures."""
    states = json.loads(
        (ROOT / "docs/evidence/survey-regression-2026-10-07/persisted-states.json").read_text()
    )
    expected = states[family]
    restored = oe.SurveyRegressionResult.model_validate_json(json.dumps(expected))
    assert "replication" not in restored.metadata
    assert restored.model_dump(mode="json") == expected
    assert restored.integrity_sha256 == expected["integrity_sha256"]
    table = restored.to_frame()
    assert table.attrs["method"] == "single-stage Taylor coefficient delta"
    assert table.attrs["survey_regression_state"] == expected
    assert table.attrs["covariance_matrix"] == expected["covariance"]
    second = oe.SurveyRegressionResult.model_validate_json(restored.model_dump_json())
    pd.testing.assert_frame_equal(table, second.to_frame())
    assert oe.to_latex(table)


@pytest.mark.parametrize("family", ["linear", "logit", "probit", "poisson"])
def test_legacy_metadata_mutation_is_detected_without_changing_original(family):
    states = json.loads(
        (ROOT / "docs/evidence/survey-regression-2026-10-07/persisted-states.json").read_text()
    )
    original = states[family]
    corrupt = deepcopy(original)
    corrupt["metadata"]["convergence"]["objective"] += 1
    with pytest.raises(ValueError, match="integrity"):
        oe.SurveyRegressionResult.model_validate(corrupt)
    assert oe.SurveyRegressionResult.model_validate(original).model_dump(mode="json") == original


CORRUPTIONS = [
    ("not_record", "linear_bootstrap"),
    ("method_list", "linear_bootstrap"),
    ("centering_mapping", "logit_fay"),
    ("unsupported_family", "linear_brr"),
    ("count_bool", "linear_fay"),
    ("count_above_limit", "logit_brr"),
    ("duplicate_ids", "linear_bootstrap"),
    ("wrong_convergence_id", "logit_brr"),
    ("missing_convergence_id", "linear_fay"),
    ("long_id", "linear_bootstrap"),
    ("short_replica_row", "linear_jackknife"),
    ("boolean_coefficient", "logit_fay"),
    ("string_coefficient", "linear_brr"),
    ("zero_multiplier", "logit_bootstrap"),
    ("negative_multiplier", "linear_jackknife"),
    ("wrong_multiplier", "linear_fay"),
    ("boolean_multiplier", "logit_brr"),
    ("failed_replica", "logit_jackknife"),
    ("not_converged", "logit_fay"),
    ("zero_iterations", "linear_bootstrap"),
    ("boolean_iterations", "logit_brr"),
    ("excess_iterations", "logit_jackknife"),
    ("zero_max_iter", "logit_bootstrap"),
    ("no_support", "linear_fay"),
    ("missing_support", "logit_brr"),
    ("duplicate_support", "linear_jackknife"),
    ("boolean_support", "linear_fay"),
    ("nested_support", "logit_bootstrap"),
    ("outside_support", "linear_brr"),
    ("zero_work", "logit_bootstrap"),
    ("cumulative_work", "linear_fay"),
    ("default_df", "linear_brr"),
    ("boolean_df", "logit_fay"),
    ("explicit_df_nonbootstrap", "linear_jackknife"),
    ("explicit_df_flag", "logit_bootstrap"),
    ("bootstrap_df_count", "linear_bootstrap"),
    ("negative_variance", "linear_brr"),
    ("asymmetric_covariance", "logit_fay"),
    ("different_covariance", "linear_bootstrap"),
    ("fay_rho", "logit_fay"),
    ("brr_rho", "linear_brr"),
    ("jk_boolean_strata", "linear_jackknife"),
    ("jk_missing_stratum", "logit_jackknife"),
    ("bootstrap_scale", "linear_bootstrap"),
    ("bootstrap_rscale", "logit_bootstrap"),
    ("bootstrap_justification", "linear_bootstrap"),
    ("missing_fingerprint", "linear_fay"),
    ("invalid_fingerprint", "logit_bootstrap"),
    ("missing_generation", "linear_jackknife"),
    ("missing_signs", "linear_brr"),
    ("unbalanced_signs", "logit_fay"),
    ("nonorthogonal_signs", "linear_brr"),
    ("generated_row_order", "linear_fay"),
]


def corrupt(payload, kind):
    record = payload["metadata"]["replication"]
    convergence = record["replicate_convergence"][0]
    if kind == "not_record":
        payload["metadata"]["replication"] = []
    elif kind == "method_list":
        record["method"] = []
    elif kind == "centering_mapping":
        record["centering"] = {}
    elif kind == "unsupported_family":
        payload["family"] = "negative_binomial"
    elif kind == "count_bool":
        record["replicate_count"] = True
    elif kind == "count_above_limit":
        record["replicate_count"] = 4097
    elif kind == "duplicate_ids":
        record["replicate_ids"][1] = record["replicate_ids"][0]
    elif kind == "wrong_convergence_id":
        convergence["replicate_id"] = record["replicate_ids"][1]
    elif kind == "missing_convergence_id":
        convergence.pop("replicate_id")
    elif kind == "long_id":
        record["replicate_ids"][0] = "x" * 201
    elif kind == "short_replica_row":
        record["replicate_estimates"][0].pop()
    elif kind == "boolean_coefficient":
        record["replicate_estimates"][0][0] = True
    elif kind == "string_coefficient":
        record["replicate_estimates"][0][0] = "0.2"
    elif kind in {"zero_multiplier", "negative_multiplier", "wrong_multiplier", "boolean_multiplier"}:
        record["variance_multipliers"][0] = {
            "zero_multiplier": 0, "negative_multiplier": -1,
            "wrong_multiplier": 1.5, "boolean_multiplier": True,
        }[kind]
    elif kind == "failed_replica":
        record["failed_replicates"] = [{"replicate_id": record["replicate_ids"][0]}]
    elif kind == "not_converged":
        convergence["converged"] = False
    elif kind == "zero_iterations":
        convergence["iterations"] = 0
    elif kind == "boolean_iterations":
        convergence["iterations"] = True
    elif kind == "excess_iterations":
        convergence["iterations"] = convergence["max_iter"] + 1
    elif kind == "zero_max_iter":
        convergence["max_iter"] = 0
    elif kind == "no_support":
        convergence["n_used"] = len(payload["labels"])
    elif kind == "missing_support":
        convergence.pop("sample_positions")
    elif kind == "duplicate_support":
        convergence["sample_positions"][1] = convergence["sample_positions"][0]
    elif kind == "boolean_support":
        convergence["sample_positions"][0] = True
    elif kind == "nested_support":
        convergence["sample_positions"][0] = {}
    elif kind == "outside_support":
        convergence["sample_positions"][-1] = payload["metadata"]["n_design"]
    elif kind == "zero_work":
        convergence["work_units"] = 0
    elif kind == "cumulative_work":
        convergence["work_units"] = 50_000_000
    elif kind == "default_df":
        payload["df"] = 1
    elif kind == "boolean_df":
        payload["df"] = True
    elif kind == "explicit_df_nonbootstrap":
        record["df_explicit"] = True
    elif kind == "explicit_df_flag":
        record["df_explicit"] = 1
    elif kind == "bootstrap_df_count":
        record["df_explicit"] = True
        payload["df"] = record["replicate_count"]
    elif kind == "negative_variance":
        payload["covariance"][0][0] = -1
    elif kind == "asymmetric_covariance":
        payload["covariance"][0][1] += 1
    elif kind == "different_covariance":
        payload["covariance"][0][0] += 1
    elif kind == "fay_rho":
        record["fay_rho"] = 1
    elif kind == "brr_rho":
        record["fay_rho"] = .5
    elif kind == "jk_boolean_strata":
        record["replicate_stratum_indices"][0] = False
    elif kind == "jk_missing_stratum":
        record["replicate_stratum_indices"][-1] = 0
    elif kind == "bootstrap_scale":
        record["scale"] = 0
    elif kind == "bootstrap_rscale":
        record["rscales"][0] = True
    elif kind == "bootstrap_justification":
        record["justification"] = " "
    elif kind == "missing_fingerprint":
        record.pop("replicate_weight_sha256")
    elif kind == "invalid_fingerprint":
        record["replicate_weight_sha256"] = "not a weight digest"
    elif kind == "missing_generation":
        record.pop("generation")
    elif kind == "missing_signs":
        record.pop("balanced_signs")
    elif kind in {"unbalanced_signs", "nonorthogonal_signs", "generated_row_order"}:
        signs = record["balanced_signs"]
        if kind == "unbalanced_signs":
            signs[0][0] *= -1
        elif kind == "nonorthogonal_signs":
            for row in signs:
                row[1] = row[0]
        else:
            signs.reverse()
        record["balanced_signs_sha256"] = hashlib.sha256(json.dumps(
            signs, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode()).hexdigest()
    else:
        raise AssertionError(kind)
    return rehash(payload)


@pytest.mark.parametrize("kind,case", CORRUPTIONS)
def test_recomputed_digest_does_not_admit_invalid_replication_contract(complete_states, kind, case):
    original = complete_states[case].model_dump(mode="json")
    changed = corrupt(deepcopy(original), kind)
    with pytest.raises(ValueError):
        oe.SurveyRegressionResult.model_validate(changed)
    assert complete_states[case].model_dump(mode="json") == original


@pytest.mark.parametrize("family", ["linear", "logit"])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_replication_and_restore_are_cpu_float64_under_ambient_defaults(
    complete_states, family, inference_mode,
):
    frame, design, _ = fixture()
    original = frame.copy(deep=True)
    previous_device, previous_dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode() if inference_mode else nullcontext():
            actual = fit(frame, design, family, "fay")
            restored = oe.SurveyRegressionResult.model_validate_json(actual.model_dump_json())
            pd.testing.assert_frame_equal(
                restored.to_frame(), complete_states[family + "_fay"].to_frame(),
            )
            assert restored.metadata["device"] == "cpu"
            assert restored.metadata["precision"] == "float64"
        assert torch.get_default_device().type == "meta"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
    assert torch.get_default_device() == previous_device
    assert torch.get_default_dtype() == previous_dtype
    pd.testing.assert_frame_equal(frame, original)


def test_cumulative_logit_work_is_refused_before_replica_weight_plan(monkeypatch):
    from openecon.econometrics.survey import regression_replication as implementation

    frame, design, _ = fixture()

    def unexpected(*args, **kwargs):
        raise AssertionError("An oversized complete fit plan reached replica-weight allocation.")

    monkeypatch.setattr(implementation, "plan_brr", unexpected)
    with pytest.raises(AnalysisError) as error:
        oe.survey_logit_replicate(
            frame, design, "b", ["x", "z"], method="brr", replicates=4096, max_iter=1000,
        )
    assert error.value.code == "survey_regression_budget"


def test_live_replica_memory_is_refused_before_weight_plan(monkeypatch):
    from openecon.econometrics.survey import regression_replication as implementation

    frame, design, _ = fixture()

    def unexpected(*args, **kwargs):
        raise AssertionError("Replica weights were created before the full live-memory guard.")

    monkeypatch.setattr(implementation, "plan_brr", unexpected)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.survey_regress_replicate(
            frame, design, "y", ["x", "z"], method="brr", replicates=4096,
        )
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("guard", ["count", "source_cells", "replay_work", "memory"])
def test_saved_replica_guard_precedes_tensor_conversion(complete_states, monkeypatch, guard):
    from openecon.econometrics.survey import regression_common as implementation

    state = complete_states["linear_bootstrap"]
    record = deepcopy(state.metadata["replication"])
    design = state.design
    if guard == "count":
        record["replicate_count"] = 4097
    elif guard == "source_cells":
        # Only the declared full design size changes; no giant table is constructed.
        validation = design.validation.model_copy(update={"nobs": 8_000_001})
        design = design.model_copy(update={"validation": validation})
    elif guard == "replay_work":
        monkeypatch.setattr(implementation, "MAX_WORK", 1)
    else:
        monkeypatch.setattr(implementation, "workspace_budget_bytes", lambda: 1)

    def unexpected(*args, **kwargs):
        raise AssertionError("Saved coefficient replicas reached tensor conversion before admission.")

    monkeypatch.setattr(implementation.torch, "tensor", unexpected)
    with pytest.raises((ValueError, AnalysisError)):
        implementation.covariance_from_replicates(
            state.coefficients, record, design, state.metadata["sample_positions"],
            original_work=state.metadata["convergence"]["work_units"],
        )
