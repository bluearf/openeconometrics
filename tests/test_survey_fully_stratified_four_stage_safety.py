"""Four-level declarations, early admission and semantic saved-state replay."""

from copy import deepcopy
import json
import pandas as pd
from pydantic import ValidationError
import pytest
import torch
import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey.common import digest
from openecon.econometrics.survey.fully_stratified_four_stage_common import stage_covariances
from openecon.resources import use_workspace_budget
from test_survey_fully_stratified_four_stage import KINDS, call, example


@pytest.fixture(scope="module")
def states():
    frame, roles = example.fixture()
    return {kind: call(frame, roles, kind, domain="domain") for kind in KINDS}


def rehash(raw):
    raw["integrity_sha256"] = digest({k: v for k, v in raw.items() if k != "integrity_sha256"})
    return raw


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "field",
    [
        "stage1_covariance",
        "stage2_covariance",
        "stage3_covariance",
        "stage4_covariance",
        "lower_stage_stratification",
        "device",
        "precision",
    ],
)
def test_outer_digest_cannot_forge_covariance_or_classification(kind, field, states):
    state = states[kind]
    raw = state.model_dump(mode="json")
    if field.endswith("_covariance"):
        raw["metadata"][field][0][0] += 0.001
    else:
        raw["metadata"][field] = "forged"
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


@pytest.mark.parametrize("kind", KINDS)
def test_complete_saved_state_replays_with_all_fit_functions_disabled(kind, states, monkeypatch):
    from openecon.econometrics.survey import fully_stratified_four_stage_regression as regression

    def forbidden(*a, **kw):
        raise AssertionError("Saved-state replay invoked an estimator")

    monkeypatch.setattr(regression, "_fit", forbidden)
    for name in KINDS:
        monkeypatch.setattr(oe, "survey_fully_stratified_four_stage_" + name, forbidden)
    state = states[kind]
    raw = state.model_dump(mode="json")
    restored = type(state).model_validate_json(json.dumps(raw, sort_keys=True))
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())
    assert oe.to_latex(restored.to_frame()) == oe.to_latex(state.to_frame())
    if kind in KINDS[:4]:
        pd.testing.assert_frame_equal(
            restored.contrast([1.0] + [0.0] * (len(state.labels) - 1)),
            state.contrast([1.0] + [0.0] * (len(state.labels) - 1)),
        )
    else:
        pd.testing.assert_frame_equal(
            restored.lincom([0.0, 1.0, 0.0]), state.lincom([0.0, 1.0, 0.0])
        )
        pd.testing.assert_frame_equal(
            restored.test([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            state.test([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        query = pd.DataFrame({"x": [-0.5, 0.5], "z": [0.2, -0.2]})
        pd.testing.assert_frame_equal(restored.predict(query), state.predict(query))


@pytest.mark.parametrize("level", ["ssu_frame", "tsu_frame", "fsu_frame"])
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "positive-unobserved",
        "wrong-parent",
        "fractional",
        "noncensus-singleton",
    ],
)
def test_every_positive_declared_lower_cell_requires_lawful_complete_sample(level, change):
    frame, roles = example.fixture()
    roles = deepcopy(roles)
    records = roles[level]
    population = {
        "ssu_frame": "population_ssu",
        "tsu_frame": "population_tsu",
        "fsu_frame": "population_fsu",
    }[level]
    if change == "missing":
        records.pop()
    if change == "duplicate":
        records.append(deepcopy(records[0]))
    if change == "positive-unobserved":
        extra = deepcopy(records[0])
        extra[
            {"ssu_frame": "ssu_stratum", "tsu_frame": "tsu_stratum", "fsu_frame": "fsu_stratum"}[
                level
            ]
        ] = 99
        records.append(extra)
    if change == "wrong-parent":
        records[0]["psu"] = 99
    if change == "fractional":
        records[0][population] = 2.5
    if change == "noncensus-singleton":
        child = {"ssu_frame": "s", "tsu_frame": "t", "fsu_frame": "f"}[level]
        frame = frame[frame[child] == 0].copy()
        roles = example.roles(frame)
    with pytest.raises(AnalysisError):
        oe.survey_fully_stratified_four_stage_design(frame, **roles)


def test_typed_paths_repeated_labels_interleaving_and_read_only_geometry():
    frame, roles = example.fixture()
    for column in ("p", "g", "s", "q", "t", "r", "f"):
        frame[column] = frame[column].map({0: True, 1: 1}).astype(object)
    # Bool True and integer 1 are distinct sampling identities despite Python equality.
    records = []
    original, _ = example.fixture()
    for row in original.to_dict("records"):
        for column in ("p", "g", "s", "q", "t", "r", "f"):
            row[column] = True if row[column] == 0 else 1
        records.append(row)
    frame = pd.DataFrame(records, dtype=object).sample(frac=1, random_state=23)
    # Build declarations from original canonical roles, preserving exact scalar types.
    _, roles = example.fixture()
    for key in ("ssu_frame", "tsu_frame", "fsu_frame"):
        for record in roles[key]:
            for name in ("psu", "ssu_stratum", "ssu", "tsu_stratum", "tsu", "fsu_stratum"):
                if name in record:
                    record[name] = True if record[name] == 0 else 1
    before = frame.copy(deep=True)
    design = oe.survey_fully_stratified_four_stage_design(frame, **roles)
    assert design.validation.nobs == 256 and design.validation.n_psu == 4
    assert design.revalidate(frame) == design.validation
    assert type(design).model_validate_json(design.model_dump_json()) == design
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValidationError):
        design.stages = 3
    assert all(
        isinstance(p, tuple) for path in design.validation.row_paths for p in path if p is not None
    )


@pytest.mark.parametrize("change", ["duplicate", "population", "row-order", "category-dictionary"])
def test_original_physical_sample_changes_are_detected(change):
    frame, roles = example.fixture()
    if change == "category-dictionary":
        frame["g"] = pd.Categorical(frame.g, categories=[0, 1, 99])
    design = oe.survey_fully_stratified_four_stage_design(frame, **roles)
    if change == "duplicate":
        frame.iloc[-1] = frame.iloc[0]
    if change == "population":
        frame["N"] += 1
    if change == "row-order":
        frame = frame.iloc[::-1]
    if change == "category-dictionary":
        frame["g"] = frame.g.cat.remove_unused_categories()
    with pytest.raises(AnalysisError):
        design.revalidate(frame)


@pytest.mark.parametrize("kind", KINDS)
def test_corrupt_state_geometry_and_budget_fail_before_tensor_allocation(kind, states, monkeypatch):
    raw = states[kind].model_dump(mode="json")
    raw["design"]["max_memory_mb"] = 1

    def forbidden(*a, **kw):
        raise AssertionError("Allocation preceded saved geometry admission")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises((ValidationError, AnalysisError)):
        type(states[kind]).model_validate(raw)


@pytest.mark.parametrize("dtype,device", [(torch.float32, "cpu"), (torch.float64, "meta")])
def test_score_device_and_precision_refused_before_numerical_use(dtype, device):
    frame, roles = example.fixture()
    design = oe.survey_fully_stratified_four_stage_design(frame, **roles)
    with pytest.raises(AnalysisError):
        stage_covariances(torch.empty((256, 2), dtype=dtype, device=device), design)


def test_workspace_limit_admits_original_geometry_before_target_conversion(monkeypatch):
    frame, roles = example.fixture()
    with use_workspace_budget(memory_mb=1):
        with pytest.raises(AnalysisError):
            oe.survey_fully_stratified_four_stage_design(frame, **roles)


def test_optional_first_stage_strata_have_one_reference_cell_with_complete_lower_frames():
    from test_survey_fully_stratified_four_stage import reference, CASES, assert_expected

    frame, _ = example.fixture()
    frame["p"] = frame.p + 2 * frame.h
    frame["h"] = 0
    frame["N"] = 6
    roles = example.roles(frame)
    roles["strata"] = None
    for key in ("ssu_frame", "tsu_frame", "fsu_frame"):
        for record in roles[key]:
            record["stratum"] = None
    state = call(frame, roles, "total", domain="domain")
    assert state.design.validation.n_strata == 1 and state.df == 3
    assert_expected(
        state,
        reference.expected_case(
            frame,
            "total",
            roles=roles,
            spec=CASES["total"],
            options={"missing": "drop", "domain": "domain", "alpha": 0.05},
        ),
    )
