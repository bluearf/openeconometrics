"""Independent saved-state acceptance guards for both lower-stage strata frames."""

from copy import deepcopy

import numpy as np
from pydantic import ValidationError
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey.common import digest
from openecon.econometrics.survey.fully_stratified_three_stage_regression_state import (
    regression_workspace,
    replay_work,
)
from test_survey_fully_stratified_three_stage_regression_state import (
    FAMILIES,
    declaration,
    fixture,
    independent_stage_covariances,
)

DESCRIPTIVE = ("mean", "total", "ratio", "proportion")
METHODS = (*DESCRIPTIVE, *FAMILIES)


@pytest.fixture(scope="module")
def saved():
    frame, design = fixture()
    frame["den"] = 2.0 + frame.j / 5 + frame.q / 10
    frame["den2"] = 3.0 + frame.j / 10 + frame.q / 5
    states = {}
    for method in METHODS:
        function = getattr(oe, "survey_fully_stratified_three_stage_" + method)
        if method in ("mean", "total"):
            states[method] = function(frame, design, ["y", "z"])
        elif method == "ratio":
            states[method] = function(frame, design, ["y", "z"], ["den", "den2"])
        elif method == "proportion":
            states[method] = function(frame, design, "b", categories=[0, 1, 2])
        else:
            outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[method]
            states[method] = function(frame, design, outcome, ["x", "z"], tolerance=1e-11)
    return states


def rehash(raw):
    raw["integrity_sha256"] = digest({k: v for k, v in raw.items() if k != "integrity_sha256"})
    return raw


def typed_state(state, raw):
    """Build an unsafe instance so its before/after validators must run again."""
    return state.model_copy(update=raw)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("representation", ["json", "typed"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("tsu_frame_sha256", "0" * 64),
        ("n_terminal_strata", 0),
        ("n_terminal_strata", 32.0),
        ("n_terminal_strata", True),
        ("lower_stage_stratification", "stage2 within sampled PSU"),
    ],
)
def test_terminal_frame_classification_replayed_after_coherent_outer_hash(
    method, representation, field, value, saved
):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][field] = value
    rehash(raw)
    if representation == "typed":
        raw = typed_state(state, raw)
    with pytest.raises(ValidationError, match="lower-stratum"):
        type(state).model_validate(raw)


@pytest.mark.parametrize("method", DESCRIPTIVE)
@pytest.mark.parametrize("representation", ["json", "typed"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("calibration_support", True),
        ("weight_semantics", "arbitrary probability weights"),
        ("variance_formula", "pooled within-SSU terminal rows"),
        ("design_effect", 1.0),
        ("confidence_interval", "normal reference interval"),
    ],
)
def test_descriptive_scientific_classification_cannot_be_relabelled(
    method, representation, field, value, saved
):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][field] = value
    rehash(raw)
    if representation == "typed":
        raw = typed_state(state, raw)
    with pytest.raises(ValidationError, match="classification|lower-stratum"):
        type(state).model_validate(raw)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("defect", ["covariance-rows", "covariance-columns", "estimate", "null"])
def test_saved_numeric_shapes_refuse_before_pydantic_conversion_and_weights(
    method, defect, saved, monkeypatch
):
    state = saved[method]
    raw = state.model_dump(mode="python")
    converted = []

    class Probe:
        def __float__(self):
            converted.append(1)
            return 0.0

    if defect == "covariance-rows":
        raw["covariance"] = [*raw["covariance"], [Probe()] * len(state.labels)]
    elif defect == "covariance-columns":
        raw["covariance"] = [list(row) for row in raw["covariance"]]
        raw["covariance"][0].append(Probe())
    else:
        key = "null" if defect == "null" else "coefficients" if method in FAMILIES else "estimates"
        raw[key] = [*raw[key], Probe()]

    def forbidden_weights(self):
        raise AssertionError("Malformed dimensions reached derived weight allocation.")

    monkeypatch.setattr(type(state.design.validation), "weights", property(forbidden_weights))
    with pytest.raises(ValidationError) as error:
        type(state).model_validate(raw)
    assert converted == []
    assert len(error.value.errors()) == 1
    assert error.value.errors()[0]["loc"] == ()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("container", [None, True, 3, "validation", [], {}])
def test_unsafe_typed_design_validation_container_is_a_typed_error(method, container, saved):
    state = saved[method]
    design = state.design.model_copy(update={"validation": container})
    raw = state.model_dump(mode="python")
    raw["design"] = design
    with pytest.raises(ValidationError):
        type(state).model_validate(raw)
    poisoned = state.model_copy(update={"design": design})
    with pytest.raises(ValidationError):
        type(state).model_validate(poisoned)


@pytest.mark.parametrize("method", FAMILIES)
@pytest.mark.parametrize("container", [None, True, 3, "validation", [], {}])
def test_live_model_rejects_unsafe_typed_validation_before_preparation(
    method, container, monkeypatch
):
    from openecon.econometrics.survey import fully_stratified_three_stage_regression as fitting

    frame, design = fixture()
    poisoned = design.model_copy(update={"validation": container})

    def forbidden_preparation(*args, **kwargs):
        raise AssertionError("Unsafe live design reached target or numerical preparation.")

    monkeypatch.setattr(fitting, "prepare_fully_stratified_three_stage", forbidden_preparation)
    function = getattr(oe, "survey_fully_stratified_three_stage_" + method)
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[method]
    with pytest.raises(AnalysisError, match="revalidation"):
        function(frame, poisoned, outcome, ["x", "z"])


@pytest.mark.parametrize("method", FAMILIES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_memory_mb", None),
        ("max_memory_mb", True),
        ("max_rows", "96"),
        ("nobs", None),
        ("n_psu", True),
        ("n_ssu", 2.5),
        ("n_cells", 0),
        ("n_tsu_cells", "32"),
        ("n_tsu_cells", 10**100),
    ],
)
def test_live_model_dimensions_are_strict_before_workspace_or_preparation(
    method, field, value, monkeypatch
):
    from openecon.econometrics.survey import fully_stratified_three_stage_regression as fitting

    frame, design = fixture()
    if field in ("max_memory_mb", "max_rows"):
        design = design.model_copy(update={field: value})
    else:
        validation = design.validation.model_copy(update={field: value})
        design = design.model_copy(update={"validation": validation})

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("Malformed raw dimensions reached workspace or preparation.")

    monkeypatch.setattr(fitting, "regression_workspace", forbidden_allocation)
    monkeypatch.setattr(fitting, "prepare_fully_stratified_three_stage", forbidden_allocation)
    function = getattr(oe, "survey_fully_stratified_three_stage_" + method)
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[method]
    with pytest.raises(AnalysisError, match="revalidation"):
        function(frame, design, outcome, ["x", "z"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "defect",
    ["parent", "frame", "count-zero", "count-float", "population", "rows", "membership"],
)
def test_unsafe_typed_terminal_geometry_is_deeply_revalidated(method, defect, saved):
    state = saved[method]
    design = state.design
    validation = design.validation
    cells, ssus = list(validation.tsu_cells), list(validation.ssus)
    cell = cells[0]
    if defect == "parent":
        change = {"ssu_index": validation.n_ssu - 1}
    elif defect == "frame":
        change = {"frame_index": validation.n_tsu_cells - 1}
    elif defect.startswith("count"):
        change = {"n_tsu": 0 if defect == "count-zero" else float(cell.n_tsu)}
    elif defect == "population":
        change = {"population_tsu": cell.population_tsu + 1}
    elif defect == "rows":
        change = {"row_positions": cell.row_positions[:-1]}
    else:
        change = {}
        ssus[0] = ssus[0].model_copy(update={"tsu_cell_indices": ssus[0].tsu_cell_indices[:-1]})
    cells[0] = cell.model_copy(update=change)
    validation = validation.model_copy(update={"tsu_cells": tuple(cells), "ssus": tuple(ssus)})
    design = design.model_copy(update={"validation": validation})
    raw = state.model_dump(mode="json")
    raw["design"] = design.model_dump(mode="json", warnings=False)
    rehash(raw)
    # A valid checksum does not authorize malformed typed geometry.
    raw["design"] = design
    with pytest.raises(ValidationError):
        type(state).model_validate(raw)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("defect", ["population", "identity", "extra-role"])
def test_terminal_frame_rehashed_corruption_is_rejected(method, defect, saved):
    state = saved[method]
    raw = state.model_dump(mode="json")
    frame = raw["design"]["tsu_frame"]
    if defect == "population":
        frame[0]["population_tsu"] += 1
    elif defect == "identity":
        frame[0]["tsu_stratum"] = ["str", "unknown-terminal-cell"]
    else:
        frame[0]["population_ssu"] = 1
    frame_sha = digest(frame)
    raw["design"]["validation"]["tsu_frame_input_sha256"] = frame_sha
    raw["metadata"]["tsu_frame_sha256"] = frame_sha
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("old_schema", ["three-stage", "stratified-three-stage"])
def test_old_sampling_class_cannot_be_restored_under_new_result_contract(method, old_schema, saved):
    state = saved[method]
    raw = state.model_dump(mode="json")
    suffix = "regression-result" if method in FAMILIES else "result"
    raw["schema_version"] = f"survey-{old_schema}-{suffix}-v1"
    raw["method"] = f"{old_schema}-taylor"
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


@pytest.mark.parametrize("method", METHODS)
def test_terminal_component_and_full_covariance_are_independently_bound(method, saved):
    state = saved[method]
    raw = state.model_dump(mode="json")
    stage3 = np.asarray(raw["metadata"]["stage3_covariance"])
    assert np.trace(stage3) > 0
    # Preserve the sum of three saved components by moving all V3 into V2.
    raw["metadata"]["stage2_covariance"] = (
        np.asarray(raw["metadata"]["stage2_covariance"]) + stage3
    ).tolist()
    raw["metadata"]["stage3_covariance"] = np.zeros_like(stage3).tolist()
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


def test_terminal_stratum_centering_discriminator_for_coefficient_scores():
    frame, _ = fixture()
    frame["y"] = 1.0 + frame.p / 5 + frame.q * 0.6 + frame.g / 10 + frame.s * 0.3
    state = oe.survey_fully_stratified_three_stage_regress(frame, declaration(frame), "y", ["z"])
    bread = np.asarray(state.metadata["bread"])
    influences = np.asarray(state.metadata["row_scores"]) @ np.linalg.inv(bread)
    correct = independent_stage_covariances(frame, influences)[2]
    np.testing.assert_allclose(state.metadata["stage3_covariance"], correct, atol=1e-26, rtol=0)
    np.testing.assert_allclose(correct, np.zeros((2, 2)), atol=1e-26, rtol=0)
    pooled = np.zeros((2, 2))
    for _, ssu in frame.groupby(["h", "p", "g", "s"], sort=False):
        block = influences[ssu.index]
        centered = block - block.mean(axis=0)
        pooled += centered.T @ centered
    assert np.trace(pooled) > 1e-5, "Fixture must discriminate pooled SSU terminal centering."


def test_terminal_cell_totals_are_charged_in_memory_and_replay_work():
    # Geometry per physical row is unchanged; independently retained cell totals cost more.
    small = regression_workspace(100, 100, 3, 4, 20, 8, 20, 64)
    large = regression_workspace(100, 100, 3, 4, 20, 8, 60, 64)
    assert large.estimated_bytes == small.estimated_bytes + 40 * 3 * 160
    small_work = replay_work(100, 100, 3, 4, 20, 8, 20)
    large_work = replay_work(100, 100, 3, 4, 20, 8, 60)
    assert large_work == small_work + 40 * 3 * 3 * 12


@pytest.mark.parametrize("method", METHODS)
def test_valid_instance_roundtrip_is_lossless_and_detached_metadata_is_independent(method, saved):
    state = saved[method]
    raw = state.model_dump(mode="json")
    restored = type(state).model_validate(raw)
    assert restored.model_dump(mode="json") == raw
    assert type(state).model_validate(state).model_dump(mode="json") == raw
    detached = deepcopy(raw)
    detached["metadata"]["tsu_frame_sha256"] = "0" * 64
    assert state.metadata["tsu_frame_sha256"] != detached["metadata"]["tsu_frame_sha256"]
