"""Adversarial saved-target, complete hierarchy and original-input proof gates."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
from pydantic import ValidationError
import pytest

import openecon as oe
from openecon.econometrics.survey.common import digest

from test_survey_four_stage_regression_state import FAMILIES, declaration, fixture

DESCRIPTIVE = ("mean", "total", "ratio", "proportion")
CASES = DESCRIPTIVE + FAMILIES


def rehash(payload):
    payload["integrity_sha256"] = digest(
        {k: v for k, v in payload.items() if k != "integrity_sha256"}
    )
    return payload


def refresh_sample(payload):
    metadata = payload["metadata"]
    selected = [i in metadata["sample_positions"] for i in range(metadata["n_design"])]
    metadata["sample_input_sha256"] = digest(
        [
            payload["design"]["validation"]["design_input_sha256"],
            selected,
            metadata["primitive_values"],
            metadata["primitive_denominators"],
        ]
    )
    return rehash(payload)


@pytest.fixture(scope="module")
def states():
    frame, design = fixture()
    frame["den"] = 2.0 + frame.x.abs()
    frame["den2"] = 3.0 + frame.z.abs()
    frame["category"] = np.array([("a", 1, True)[i % 3] for i in range(len(frame))], dtype=object)
    frame["take"] = ~((frame.h == 0) & (frame.p == 0) & (frame.s == 0) & (frame.t == 0))
    result = {
        "mean": oe.survey_four_stage_mean(frame, design, ["y", "x"], domain="take"),
        "total": oe.survey_four_stage_total(frame, design, ["y", "x"], domain="take"),
        "ratio": oe.survey_four_stage_ratio(
            frame, design, ["y", "x"], ["den", "den2"], domain="take"
        ),
        "proportion": oe.survey_four_stage_proportion(
            frame, design, "category", categories=["a", 1, True, "absent"], domain="take"
        ),
    }
    for family in FAMILIES:
        outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]
        result[family] = getattr(oe, "survey_four_stage_" + family)(
            frame, design, outcome, ["x", "z"], domain="take", tolerance=1e-11
        )
    return result


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize("domain", [None, "", " " * 3, "x" * 201, True, [], {}])
def test_rehashed_domain_metadata_must_describe_a_possible_original_partition(kind, domain, states):
    state = states[kind]
    payload = state.model_dump(mode="json")
    payload["metadata"]["domain_column"] = domain
    with pytest.raises(ValidationError, match="sample/geometry/weight semantics"):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize(
    "key,value",
    [("stages", 4.0), ("n_tsu", 24.0), ("lower_stage_stratification", "stage3 within SSU")],
)
def test_new_exact_four_stage_classification_cannot_be_relabelled(kind, key, value, states):
    state = states[kind]
    payload = state.model_dump(mode="json")
    payload["metadata"][key] = value
    with pytest.raises(ValidationError, match="sample/geometry/weight semantics"):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("kind", DESCRIPTIVE)
@pytest.mark.parametrize(
    "component",
    ["stage1_covariance", "stage2_covariance", "stage3_covariance", "stage4_covariance"],
)
def test_rehashed_descriptive_component_changes_fail_full_primitive_replay(kind, component, states):
    state = states[kind]
    payload = state.model_dump(mode="json")
    payload["metadata"][component][0][0] += 0.001
    with pytest.raises(ValidationError, match="stage-specific covariance"):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("kind", DESCRIPTIVE)
@pytest.mark.parametrize("field", ["primitive_values", "primitive_denominators", "row_influences"])
@pytest.mark.parametrize("shape", ["container", "rows", "columns"])
def test_descriptive_primitive_shapes_fail_before_numeric_scalar_conversion(
    kind, field, shape, states
):
    state = states[kind]
    payload = state.model_dump(mode="json")
    converted = []

    class Probe:
        def __float__(self):
            converted.append(1)
            return 1.0

    if shape == "container":
        payload["metadata"][field] = {"rows": Probe()}
    elif shape == "rows":
        payload["metadata"][field].append([Probe()] * len(state.labels))
    else:
        payload["metadata"][field][0].append(Probe())
    # The invalid raw shape itself must be rejected even without an outer digest.
    with pytest.raises(ValidationError, match="primitive dimensions/values"):
        type(state).model_validate(payload)
    assert converted == []


@pytest.mark.parametrize("kind", DESCRIPTIVE)
def test_changed_primitive_rows_fail_even_when_numeric_sample_and_outer_hash_are_refreshed(
    kind, states
):
    state = states[kind]
    payload = state.model_dump(mode="json")
    first = payload["metadata"]["sample_positions"][0]
    if kind == "proportion":
        payload["metadata"]["primitive_values"][first] = [1.0] * len(state.labels)
    else:
        payload["metadata"]["primitive_values"][first][0] += 0.1
    with pytest.raises(ValidationError, match="one-hot|estimates/linearized rows"):
        type(state).model_validate(refresh_sample(payload))


@pytest.mark.parametrize("kind", ["mean", "total", "ratio"])
def test_nonproportion_saved_targets_cannot_declare_categories_after_coherent_rehash(kind, states):
    state = states[kind]
    payload = state.model_dump(mode="json")
    payload["metadata"]["categories"] = [["str", "wrong-target-role"]]
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(payload))


def test_invalid_saved_category_scalar_is_rejected_before_building_a_label(states):
    state = states["proportion"]
    converted = []

    class Probe:
        def __str__(self):
            converted.append(1)
            return "a"

    poisoned = state.model_copy(deep=True)
    poisoned.metadata["categories"][0][1] = Probe()
    with pytest.raises(ValidationError, match="category identity"):
        type(state).model_validate(poisoned)
    assert converted == []


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize("field", ["estimates", "null", "covariance_column", "covariance_row"])
def test_raw_target_numeric_shape_is_bounded_before_parsing_or_derived_weights(
    kind, field, states, monkeypatch
):
    state = states[kind]
    converted = []

    class Probe:
        def __float__(self):
            converted.append(1)
            return 1.0

    if field.startswith("covariance"):
        key = "covariance"
        value = (
            tuple(tuple(row) + (Probe(),) for row in state.covariance)
            if field.endswith("column")
            else state.covariance + ((Probe(),) * len(state.labels),)
        )
    else:
        key = "coefficients" if field == "estimates" and kind in FAMILIES else field
        value = getattr(state, key) + (Probe(),)
    poisoned = state.model_copy(update={key: value})

    def forbidden(self):
        raise AssertionError("malformed target vector preceded numeric shape admission")

    monkeypatch.setattr(type(state.design.validation), "weights", property(forbidden))
    with pytest.raises(ValidationError, match="bounded|before parsing"):
        type(state).model_validate(poisoned)
    assert converted == []


@pytest.mark.parametrize("kind", DESCRIPTIVE)
def test_direct_unsafe_target_instances_are_revalidated_by_inference(kind, states):
    state = states[kind]
    poisoned = state.model_copy(update={"estimates": tuple(v + 1.0 for v in state.estimates)})
    with pytest.raises(ValidationError, match="estimates/linearized rows"):
        type(state).model_validate(poisoned)
    with pytest.raises(ValidationError, match="estimates/linearized rows"):
        poisoned.to_frame()


@pytest.mark.parametrize("kind", CASES)
def test_model_construct_terminal_geometry_cannot_bypass_typed_revalidation(kind, states):
    state = states[kind]
    terminals = list(state.design.validation.tsus)
    terminals[0] = type(terminals[0]).model_construct(**{**vars(terminals[0]), "n_fsu": 0})
    validation = type(state.design.validation).model_construct(
        **{**vars(state.design.validation), "tsus": tuple(terminals)}
    )
    design = type(state.design).model_construct(**{**vars(state.design), "validation": validation})
    poisoned = state.model_copy(update={"design": design})
    with pytest.raises(ValidationError):
        type(state).model_validate(poisoned)


@pytest.mark.parametrize("kind", ["mean", "total", "ratio"])
def test_subnormal_scale_target_variance_cannot_be_erased_after_rehash(kind):
    frame, design = fixture()
    frame[["y", "x"]] *= 1e-140
    if kind == "ratio":
        frame["den"] = 2.0
        frame["den2"] = 2.0
        state = oe.survey_four_stage_ratio(frame, design, ["y", "x"], ["den", "den2"])
    else:
        state = getattr(oe, "survey_four_stage_" + kind)(frame, design, ["y", "x"])
    assert 0 < max(abs(v) for row in state.covariance for v in row) < 1e-270
    payload = state.model_dump(mode="json")
    payload["covariance"] = [[0.0] * 2 for _ in range(2)]
    with pytest.raises(ValidationError, match="four recursive sampling stages"):
        type(state).model_validate(rehash(payload))


def test_original_input_oracle_rebuilds_ordered_geometry_and_dtype_fingerprint(states):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_survey_four_stage_oracles.py"
    spec = importlib.util.spec_from_file_location("review_four_stage_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    frame, _ = fixture()
    # Physical row order differs from nested traversal; saved indices must retain
    # first occurrence, not sorted group identity or DataFrame index labels.
    frame = frame.iloc[np.random.default_rng(44).permutation(len(frame))].reset_index(drop=True)
    design = declaration(frame)
    roles = {
        "psu": "p",
        "ssu": "s",
        "tsu": "t",
        "fsu": "f",
        "strata": "h",
        "population_psu": "N",
        "population_ssu": "M",
        "population_tsu": "L",
        "population_fsu": "K",
    }
    expected = reference.retained_geometry(frame, roles)
    actual = design.validation.model_dump(mode="json")
    for key, value in expected.items():
        assert actual[key] == value
    changed = deepcopy(frame)
    changed["f"] = changed["f"].astype(float)
    assert (
        reference.retained_geometry(changed, roles)["design_input_sha256"]
        != expected["design_input_sha256"]
    )
