"""Complete-domain four-stage persistence, classification and preallocation guards."""

import json

import numpy as np
from pydantic import ValidationError
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey.common import digest
from openecon.econometrics.survey.four_stage_common import prepare_four_stage, stage_covariances
from openecon.resources import use_workspace_budget
from openecon.survey_four_stage import FourStageValidation
from test_survey_four_stage_design import NumericProbe, declare, fixture as design_fixture

METHODS = ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson")
REGRESSION = ("regress", "logit", "probit", "poisson")


def fixture():
    frame = design_fixture()
    # Unequal category shares at all cluster levels make removal of each
    # stage component observable; balanced leaves would make V1–V3 zero.
    threshold = 1 + (frame.h + frame.p + frame.s + frame.t) % 3
    frame["cat"] = np.where(frame.u < threshold, "a", "b")
    return frame


def run(method, frame=None, design=None, **options):
    frame = fixture() if frame is None else frame
    design = declare(frame) if design is None else design
    fn = getattr(oe, "survey_four_stage_" + method)
    if method == "ratio":
        return fn(frame, design, ["y", "x"], ["den", "den2"], **options)
    if method == "proportion":
        return fn(frame, design, "cat", categories=["a", "b", "absent"], **options)
    if method in REGRESSION:
        outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[method]
        return fn(frame, design, outcome, ["x", "z"], tolerance=1e-11, **options)
    return fn(frame, design, ["y", "x"], **options)


def rehash(raw):
    raw["integrity_sha256"] = digest({k: v for k, v in raw.items() if k != "integrity_sha256"})
    return raw


@pytest.fixture(scope="module")
def states():
    return {method: run(method) for method in METHODS}


@pytest.mark.parametrize("method", METHODS)
def test_ordered_states_sorted_json_and_all_covariance_components_restore(method, states):
    state = states[method]
    raw = state.model_dump(mode="json")
    restored = type(state).model_validate_json(json.dumps(raw, sort_keys=True))
    assert restored.model_dump(mode="json") == raw
    frame = restored.to_frame()
    assert frame.attrs["survey_state"] == raw
    assert frame.attrs["covariance_matrix"] == raw["covariance"]
    assert frame.attrs["df"] == state.design.validation.design_df
    assert "all four" in frame.attrs["uncertainty"]
    expected = sum(np.asarray(raw["metadata"][f"stage{i}_covariance"]) for i in range(1, 5))
    np.testing.assert_allclose(expected, state.covariance, rtol=1e-12, atol=0)
    frame.attrs["survey_state"]["metadata"]["stages"] = 99
    frame.attrs["covariance_matrix"][0][0] = 99
    frame.iloc[0, 0] = 99
    assert state.model_dump(mode="json") == raw


@pytest.mark.parametrize("method", METHODS)
def test_domain_and_missing_keep_zero_fsu_rows_and_all_original_sampled_parents(method):
    frame = fixture()
    original = declare(frame)
    # An entire first-stage PSU and a different terminal TSU disappear from the
    # target, but remain sampled design nodes at every level of covariance.
    excluded = (frame.h.eq(1) & frame.p.eq(1)) | (
        frame.h.eq(0) & frame.p.eq(0) & frame.s.eq(0) & frame.t.eq(0)
    )
    frame["domain"] = (~excluded).astype(int)
    outcome = {"proportion": "cat", "logit": "b", "probit": "b", "poisson": "c"}.get(method, "y")
    frame[outcome] = frame[outcome].astype(object)
    missing_row = int(np.flatnonzero(~excluded)[0])
    frame.iat[missing_row, frame.columns.get_loc(outcome)] = np.nan
    state = run(method, frame, original, domain="domain", missing="drop")
    assert state.design == original
    assert state.metadata["n_design"] == 64
    assert state.metadata["n_used"] == int((~excluded).sum()) - 1
    assert state.metadata["outcome_exclusions"] == [missing_row]
    assert state.metadata["out_of_domain_positions"] == list(np.flatnonzero(excluded))
    rows = state.metadata["row_scores" if method in REGRESSION else "row_influences"]
    for i in list(np.flatnonzero(excluded)) + [missing_row]:
        assert rows[i] == [0.0] * len(state.labels)
    assert state.metadata["n_tsu"] == 16 and state.metadata["stages"] == 4
    assert type(state).model_validate_json(state.model_dump_json()).model_dump(
        mode="json"
    ) == state.model_dump(mode="json")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "field,value",
    [
        ("stages", 3),
        ("n_tsu", 15),
        ("n_tsu", True),
        ("lower_stage_stratification", "stage3"),
        ("weight_semantics", "ordinary WLS"),
        ("device", "cuda"),
        ("precision", "float32"),
        ("calibration_support", True),
    ],
)
def test_recomputed_digest_cannot_forge_scientific_classification(method, field, value, states):
    state = states[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][field] = value
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("stage", range(1, 5))
def test_recomputed_digest_cannot_remove_one_stage_covariance(method, stage, states):
    state = states[method]
    raw = state.model_dump(mode="json")
    key = f"stage{stage}_covariance"
    assert np.max(np.abs(raw["metadata"][key])) > 0
    raw["metadata"][key] = np.zeros_like(raw["covariance"]).tolist()
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


@pytest.mark.parametrize("method", METHODS)
def test_typed_design_model_copy_inside_rehashed_target_cannot_bypass_stage_count(method, states):
    state = states[method]
    unsafe = state.design.model_copy(update={"stages": 3})
    raw = state.model_dump(mode="json")
    raw["design"] = unsafe.model_dump(mode="json")
    rehash(raw)
    raw["design"] = unsafe
    with pytest.raises(ValidationError):
        type(state).model_validate(raw)


@pytest.mark.parametrize("method", METHODS)
def test_saved_budget_is_checked_before_derived_weights_or_numeric_conversion(
    method, states, monkeypatch
):
    state = states[method]
    raw = state.model_dump(mode="json")
    raw["design"]["validation"]["nobs"] = 1000
    raw["design"]["max_memory_mb"] = 1
    vector = "coefficients" if method in REGRESSION else "estimates"
    raw[vector][0] = NumericProbe()

    def forbidden(_self):
        raise AssertionError("budget must be checked before any derived-weight allocation")

    monkeypatch.setattr(FourStageValidation, "weights", property(forbidden))
    with pytest.raises(ValidationError, match="workspace bytes"):
        type(state).model_validate(raw)


@pytest.mark.parametrize("method", METHODS)
def test_covariance_shape_rejects_before_numeric_probe(method, states):
    state = states[method]
    raw = state.model_dump(mode="json")
    raw["covariance"] = [[NumericProbe()]]
    with pytest.raises(ValidationError, match="bounded"):
        type(state).model_validate(raw)


@pytest.mark.parametrize("method", METHODS)
def test_df_zero_positive_lower_stage_variance_has_no_reference_tests_or_interval(method):
    frame = fixture().query("p == 0").copy()
    frame["N"] = 1
    state = run(method, frame)
    assert state.df == 0
    table = state.to_frame()
    positive = np.diag(state.covariance) > 0
    assert positive.any()
    assert table.loc[positive, ["statistic", "p_value", "ci_low", "ci_high"]].isna().all().all()


@pytest.mark.parametrize("method", METHODS)
def test_all_four_stages_census_have_zero_covariance_and_point_intervals(method):
    frame = fixture()
    frame[["N", "M", "L", "K"]] = [2, 2, 2, 4]
    state = run(method, frame)
    np.testing.assert_array_equal(
        state.covariance, np.zeros((len(state.labels), len(state.labels)))
    )
    table = state.to_frame()
    assert table[["statistic", "p_value"]].isna().all().all()
    np.testing.assert_array_equal(table.ci_low, table.estimate)
    np.testing.assert_array_equal(table.ci_high, table.estimate)


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
def test_descriptive_target_domain_validation_applies_before_numeric_values(method):
    frame = fixture()
    frame["domain"] = np.full(len(frame), True, dtype=object)
    frame.iat[0, frame.columns.get_loc("domain")] = "unknown"
    with pytest.raises(AnalysisError) as error:
        run(method, frame, domain="domain")
    assert error.value.code == "invalid_survey_domain"


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("alpha", [True, 0, 1, np.nan, np.inf, 1e-12, "0.05"])
def test_target_confidence_level_matches_live_and_saved_admission(method, alpha):
    with pytest.raises(AnalysisError):
        run(method, alpha=alpha)


@pytest.mark.parametrize(
    "field,value",
    [
        ("variance_formula", "ordinary IID"),
        ("confidence_interval", "exact finite-sample"),
        ("design_effect", 1),
        ("calibration_support", 1),
    ],
)
def test_descriptive_rehashed_payload_cannot_claim_unimplemented_uncertainty(field, value, states):
    state = states["mean"]
    raw = state.model_dump(mode="json")
    raw["metadata"][field] = value
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


def test_representable_tiny_targets_and_variance_cannot_be_replaced_with_zero():
    frame = fixture()
    frame["y"] *= 1e-160
    state = oe.survey_four_stage_mean(frame, declare(frame), ["y"])
    assert 0 < state.estimates[0] < 1e-159
    assert 0 < state.covariance[0][0] < 1e-310
    raw = state.model_dump(mode="json")
    raw["estimates"] = [0.0]
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))
    raw = state.model_dump(mode="json")
    raw["covariance"] = [[0.0]]
    for stage in range(1, 5):
        raw["metadata"][f"stage{stage}_covariance"] = [[0.0]]
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


def test_global_budget_refuses_target_before_numeric_tensor_creation(monkeypatch):
    frame, design = fixture(), declare()

    def forbidden(*args, **kwargs):
        raise AssertionError("target tensor creation must follow admission")

    monkeypatch.setattr(torch, "tensor", forbidden)
    with use_workspace_budget(1):
        with pytest.raises((AnalysisError, ValidationError)):
            prepare_four_stage(frame, design, "mean", ["y"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex128, torch.int64])
def test_internal_covariance_requires_float64_real_scores(dtype):
    design = declare()
    with pytest.raises(AnalysisError):
        stage_covariances(torch.zeros((64, 2), dtype=dtype), design)


def test_internal_covariance_rejects_meta_input_without_device_fallback():
    with pytest.raises(AnalysisError):
        stage_covariances(torch.zeros((64, 2), dtype=torch.float64, device="meta"), declare())


@pytest.mark.parametrize("role", ["p", "s", "t", "u", "N", "M", "L", "K", "y"])
def test_tensor_column_mapping_is_rejected_without_scalar_or_device_conversion(role):
    frame = fixture()
    values = frame.to_dict("list")
    values[role] = torch.zeros(len(frame), device="meta")
    with pytest.raises(AnalysisError):
        run("mean", values, declare(frame))


def test_ambient_dtype_meta_device_and_inference_mode_do_not_change_public_results(states):
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        with torch.inference_mode():
            actual = run("mean")
        assert actual.model_dump(mode="json") == states["mean"].model_dump(mode="json")
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)


def test_prior_three_stage_payload_still_has_its_own_exact_schema_and_numerics():
    frame = fixture()
    # One physical FSU per census TSU reduces to old three-stage sampling;
    # this compatibility case is not one of the eight new acceptance gates.
    frame = frame.query("u == 0").copy()
    frame["K"] = 1
    old = oe.survey_three_stage_design(
        frame,
        psu="p",
        ssu="s",
        tsu="t",
        strata="h",
        population_psu="N",
        population_ssu="M",
        population_tsu="L",
    )
    old_state = oe.survey_three_stage_total(frame, old, ["y", "x"])
    before = old_state.model_dump_json()
    new = run("total", frame)
    np.testing.assert_array_equal(new.estimates, old_state.estimates)
    np.testing.assert_array_equal(new.covariance, old_state.covariance)
    assert type(old_state).model_validate_json(before).model_dump_json() == before
    assert old_state.schema_version == "survey-three-stage-result-v1"
