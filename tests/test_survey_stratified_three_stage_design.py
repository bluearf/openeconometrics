"""Admission, explicit frame and restoration guards for three real sampling stages."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from openecon.econometrics.survey.common import digest
from openecon.econometrics.survey.stratified_three_stage_regression import (
    survey_stratified_three_stage_regress,
)
from openecon.econometrics.survey.stratified_three_stage_regression_state import (
    SurveyStratifiedThreeStageRegressionResult,
)

from openecon.analysis_contracts import AnalysisError
from openecon.survey_stratified_three_stage import (
    SurveyStratifiedThreeStageDesign,
    StratifiedThreeStageFrameCell,
    survey_stratified_three_stage_design,
)
from openecon.econometrics.survey.stratified_three_stage_common import (
    prepare_stratified_three_stage,
    validate_sample_metadata,
)

ROLES = dict(
    psu="p",
    ssu_strata="g",
    ssu="s",
    tsu="t",
    population_psu="N",
    population_ssu="M",
    population_tsu="L",
    strata="h",
)


def fixture():
    rows = [
        {
            "h": h,
            "p": p,
            "g": g,
            "s": s,
            "t": t,
            "N": 5 + h,
            "M": 3 + 2 * g + p,
            "L": 4 + s,
            "y": 1.0 + h + p + g + s + t,
            "domain": int(not (h == 1 and p == 1)),
        }
        for h in range(2)
        for p in range(2)
        for g in range(2)
        for s in range(2)
        for t in range(3)
    ]
    frame = pd.DataFrame(rows)
    frame.index = np.resize([8, -1, 8], len(frame))
    lower = [
        {"h": h, "p": p, "g": g, "M": 3 + 2 * g + p}
        for h in range(2)
        for p in range(2)
        for g in range(2)
    ]
    return frame, lower


def declare(data, lower, **options):
    return survey_stratified_three_stage_design(data, ssu_frame=lower, **(ROLES | options))


def test_actual_unequal_cell_weights_and_frozen_frame_binding():
    frame, lower = fixture()
    design = declare(frame, lower)
    expected = frame.N / 2 * frame.M / 2 * frame.L / 3
    np.testing.assert_array_equal(design.validation.weights, expected)
    assert design.validation.nobs == 48 and design.validation.n_ssu == 16
    assert design.validation.n_cells == 8 and design.validation.n_psu == 4
    assert design.validation.design_df == 2
    for cell in design.validation.cells:
        assert design.ssu_frame[cell.frame_index].population_ssu == cell.population_ssu
        assert (
            tuple(
                sorted(i for j in cell.ssu_indices for i in design.validation.ssus[j].row_positions)
            )
            == cell.row_positions
        )
    assert design.revalidate(frame) == design.validation
    reordered_frame = declare(frame, list(reversed(lower)))
    assert reordered_frame == design
    assert (
        type(design).model_validate_json(json.dumps(design.model_dump(mode="json"), sort_keys=True))
        == design
    )


@pytest.mark.parametrize("representation", ["dataframe", "columns", "records"])
def test_supported_resident_inputs_use_identical_physical_geometry(representation):
    frame, lower = fixture()
    original = declare(frame, lower)
    data = (
        frame
        if representation == "dataframe"
        else frame.to_dict("list" if representation == "columns" else "records")
    )
    actual = declare(data, lower)
    assert actual == original


def test_no_first_stage_strata_retains_explicit_lower_frame():
    frame, lower = fixture()
    frame = frame[frame.h == 0].drop(columns="h")
    lower = [{k: v for k, v in row.items() if k != "h"} for row in lower if row["h"] == 0]
    design = declare(frame, lower, strata=None)
    assert all(record.stratum is None for record in design.ssu_frame)
    assert design.validation.n_strata == 1
    assert design.revalidate(frame) == design.validation


@pytest.mark.parametrize("identity_role", ["h", "p", "g", "s", "t"])
def test_typed_identity_labels_do_not_collapse_bool_int_float_or_string(identity_role):
    identities = [False, 0, 0.0, "0"]
    rows = []
    for h in range(2):
        for p in range(2):
            for g in range(2):
                for s in range(2):
                    for t in range(2):
                        for label in identities:
                            row = dict(h=h, p=p, g=g, s=s, t=t, N=8, M=8, L=8)
                            row[identity_role] = label
                            rows.append(row)
    # Each identity role must form unique physical leaves; deduplicate the other
    # repeated loop dimensions after substituting one typed role.
    unique = {}
    for row in rows:
        key = tuple((type(row[k]).__name__, repr(row[k])) for k in ("h", "p", "g", "s", "t"))
        unique[key] = row
    frame = pd.DataFrame(list(unique.values()), dtype=object)
    lower = []
    seen = set()
    for row in frame.to_dict("records"):
        key = tuple((type(row[k]).__name__, repr(row[k])) for k in ("h", "p", "g"))
        if key not in seen:
            seen.add(key)
            lower.append({k: row[k] for k in ("h", "p", "g", "M")})
    design = declare(frame, lower)
    for key in ("h", "p", "g", "s", "t"):
        expected = 4 if key == identity_role else 2
        if key == "h":
            assert design.validation.n_strata == expected
    assert design.validation.nobs == 64
    assert design.revalidate(frame) == design.validation
    assert SurveyStratifiedThreeStageDesign.model_validate_json(design.model_dump_json()) == design


@pytest.mark.parametrize(
    "failure",
    [
        "missing_cell",
        "extra_cell",
        "extra_psu",
        "duplicate",
        "wrong_population",
        "missing_role",
        "extra_role",
        "zero_population",
    ],
)
def test_explicit_positive_lower_frame_must_exactly_cover_observed_psus_and_cells(failure):
    frame, lower = fixture()
    lower = deepcopy(lower)
    if failure == "missing_cell":
        lower.pop()
    elif failure == "extra_cell":
        lower.append(lower[0] | {"g": "unobserved"})
    elif failure == "extra_psu":
        lower.append(lower[0] | {"p": "unobserved"})
    elif failure == "duplicate":
        lower.append(lower[0])
    elif failure == "wrong_population":
        lower[0]["M"] += 1
    elif failure == "missing_role":
        lower[0].pop("g")
    elif failure == "extra_role":
        lower[0]["extra"] = 1
    else:
        lower[0]["M"] = 0
    with pytest.raises(AnalysisError):
        declare(frame, lower)


@pytest.mark.parametrize(
    "value", [None, True, 0, -1, 0.5, 2**53 + 1, "5", float("nan"), float("inf"), 5 + 0j]
)
@pytest.mark.parametrize("role", ["N", "M", "L"])
def test_population_counts_are_positive_exact_bounded_numbers(role, value):
    frame, lower = fixture()
    frame[role] = frame[role].astype(object)
    frame.iloc[0, frame.columns.get_loc(role)] = value
    with pytest.raises(AnalysisError):
        declare(frame, lower)


@pytest.mark.parametrize("role", ["h", "p", "g", "s", "t"])
@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), complex(1), "", "a" * 257, 2**801]
)
def test_invalid_physical_identity_is_rejected(role, value):
    frame, lower = fixture()
    frame[role] = frame[role].astype(object)
    frame.iloc[0, frame.columns.get_loc(role)] = value
    with pytest.raises(AnalysisError):
        declare(frame, lower)


@pytest.mark.parametrize("role", ["N", "M", "L"])
def test_population_count_must_be_constant_within_its_own_sampling_cell(role):
    frame, lower = fixture()
    frame.iloc[0, frame.columns.get_loc(role)] += 1
    with pytest.raises(AnalysisError):
        declare(frame, lower)


def test_reused_ssu_and_tsu_labels_are_nested_inside_their_lower_strata():
    frame, lower = fixture()
    design = declare(frame, lower)
    assert design.validation.n_ssu == 16
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(AnalysisError, match="unique nested TSU"):
        declare(duplicated, lower)


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("census", [False, True])
def test_singleton_sampling_units_require_proven_census_at_their_actual_stage(stage, census):
    frame, lower = fixture()
    if stage == 1:
        frame = frame[frame.p == 0].copy()
        lower = [r for r in lower if r["p"] == 0]
        frame["N"] = 1 if census else 2
    elif stage == 2:
        frame = frame[frame.s == 0].copy()
        frame["M"] = 1 if census else 2
        lower = [r | {"M": 1 if census else 2} for r in lower]
    else:
        frame = frame[frame.t == 0].copy()
        frame["L"] = 1 if census else 2
    if census:
        design = declare(frame, lower)
        assert design.revalidate(frame) == design.validation
    else:
        with pytest.raises(AnalysisError, match="singleton requires census"):
            declare(frame, lower)


@pytest.mark.parametrize("role", ["h", "p", "g", "s", "t", "N", "M", "L"])
def test_design_revalidation_binds_all_original_physical_roles(role):
    frame, lower = fixture()
    design = declare(frame, lower)
    changed = frame.copy()
    changed[role] = changed[role].astype(object)
    changed.iloc[0, changed.columns.get_loc(role)] = (
        "changed" if role in ("h", "p", "g", "s", "t") else 100
    )
    with pytest.raises(AnalysisError):
        design.revalidate(changed)


def test_outcomes_and_duplicate_dataframe_index_do_not_redefine_original_geometry():
    frame, lower = fixture()
    design = declare(frame, lower)
    changed = frame.copy()
    changed["y"] = np.nan
    changed.index = range(len(changed))
    assert design.revalidate(changed) == design.validation
    reversed_rows = frame.iloc[::-1].copy()
    with pytest.raises(AnalysisError, match="ordered geometry changed"):
        design.revalidate(reversed_rows)


@pytest.mark.parametrize("field", ["nobs", "n_cells", "n_ssu", "n_psu", "n_strata", "design_df"])
@pytest.mark.parametrize("value", [True, 0.5, -1, "2"])
def test_restore_does_not_coerce_geometry_counts(field, value):
    frame, lower = fixture()
    state = declare(frame, lower).model_dump(mode="json")
    state["validation"][field] = value
    with pytest.raises(ValueError):
        SurveyStratifiedThreeStageDesign.model_validate(state)


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("cells", "psu_index", 3),
        ("cells", "frame_index", 7),
        ("cells", "population_ssu", 200),
        ("cells", "ssu_indices", [0, 4]),
        ("cells", "row_positions", [0, 1]),
        ("psus", "cell_indices", [1]),
        ("psus", "ssu_indices", [0, 1]),
        ("psus", "row_positions", [0, 1]),
        ("ssus", "cell_index", 1),
        ("ssus", "psu_index", 1),
        ("ssus", "row_positions", [0, 1, 3]),
        ("strata", "psu_indices", [0, 2]),
    ],
)
def test_restore_rejects_parent_and_physical_partition_corruption(part, field, value):
    frame, lower = fixture()
    state = declare(frame, lower).model_dump(mode="json")
    state["validation"][part][0][field] = value
    with pytest.raises(ValueError):
        SurveyStratifiedThreeStageDesign.model_validate(state)


@pytest.mark.parametrize(
    "failure",
    ["duplicate", "population", "parent_id", "stratum_id", "ordering", "digest", "missing"],
)
def test_saved_canonical_frame_is_bound_to_the_complete_geometry(failure):
    frame, lower = fixture()
    state = declare(frame, lower).model_dump(mode="json")
    records = state["ssu_frame"]
    if failure == "duplicate":
        records[1] = deepcopy(records[0])
    elif failure == "population":
        records[0]["population_ssu"] += 1
    elif failure == "parent_id":
        records[0]["psu"] = ["str", "different"]
    elif failure == "stratum_id":
        records[0]["stratum"] = ["str", "different"]
    elif failure == "ordering":
        records.reverse()
    elif failure == "digest":
        state["validation"]["frame_input_sha256"] = "0" * 64
    else:
        records.pop()
    with pytest.raises(ValueError):
        SurveyStratifiedThreeStageDesign.model_validate(state)


@pytest.mark.parametrize("field", ["psu", "stratum", "ssu_stratum"])
@pytest.mark.parametrize(
    "value",
    [
        ["bool", 1],
        ["int", False],
        ["int", 1.0],
        ["float", "0.0"],
        ["float", "0x1p0"],
        ["float", "a" * 10000],
        ["float", "0x1p+99999"],
        ["float", "inf"],
        ["str", ""],
        ["str", "a" * 257],
        ["str", 1],
        ["other", 1],
    ],
)
def test_saved_frame_identity_uses_exact_finite_canonical_encoding(field, value):
    state = dict(stratum=["int", 0], psu=["int", 0], ssu_stratum=["str", "a"], population_ssu=3)
    state[field] = value
    with pytest.raises(ValueError):
        StratifiedThreeStageFrameCell.model_validate(state)


@pytest.mark.parametrize("field", ["n_design", "n_domain", "n_used", "n_lower_strata"])
def test_one_row_census_metadata_rejects_bool_aliasing_of_integer_counts(field):
    frame = pd.DataFrame([dict(h=0, p=0, g=0, s=0, t=0, N=1, M=1, L=1, y=2.0)])
    design = declare(frame, [dict(h=0, p=0, g=0, M=1)])
    target = prepare_stratified_three_stage(frame, design, "mean", ["y"])
    metadata = deepcopy(target.metadata)
    metadata[field] = True
    with pytest.raises(ValueError):
        validate_sample_metadata(metadata, design)


def test_complete_cell_frame_and_counts_remain_when_domain_or_outcomes_exclude_whole_cell():
    frame, lower = fixture()
    design = declare(frame, lower)
    frame.loc[(frame.p == 0) & (frame.g == 0), "domain"] = 0
    frame.loc[(frame.p == 0) & (frame.g == 0), "y"] = np.nan
    frame.loc[(frame.p == 1) & (frame.g == 1) & (frame.h == 0), "y"] = np.nan
    target = prepare_stratified_three_stage(
        frame, design, "mean", ["y"], domain="domain", missing="drop"
    )
    assert target.design == design
    assert target.metadata["n_lower_strata"] == 8
    assert target.metadata["ssu_frame_sha256"] == design.validation.frame_input_sha256
    assert len(validate_sample_metadata(target.metadata, design)) == target.metadata["n_used"]


@pytest.mark.parametrize("budget", [dict(max_memory_mb=1), dict(max_rows=1)])
def test_complete_geometry_budget_is_checked_before_reading_lower_frame(budget):
    class UnreadableFrame(list):
        def __iter__(self):
            raise AssertionError("Lower-frame values inspected before complete geometry budget.")

    frame, _ = fixture()
    if "max_memory_mb" in budget:
        frame = pd.concat([frame] * 20, ignore_index=True)
    with pytest.raises(AnalysisError):
        declare(frame, UnreadableFrame([{}]), **budget)


def test_saved_design_budget_precedes_large_canonical_frame_or_nested_allocations():
    class UnreadableFrame(list):
        def __iter__(self):
            raise AssertionError("Saved frame copied before parent workspace admission.")

    frame, lower = fixture()
    state = declare(frame, lower).model_dump(mode="json")
    state["max_memory_mb"] = 1
    state["validation"]["nobs"] = 1000
    state["ssu_frame"] = UnreadableFrame([{}] * 1000)
    with pytest.raises(ValueError, match="workspace bytes"):
        SurveyStratifiedThreeStageDesign.model_validate(state)


@pytest.mark.parametrize("key", ["ssu_frame", "strata", "psus", "cells", "ssus"])
def test_restore_rejects_oversized_schema_before_item_admission(key):
    class UnreadableRecord(dict):
        def get(self, *args):
            raise AssertionError("Oversized records read before shape admission.")

    frame, lower = fixture()
    state = declare(frame, lower).model_dump(mode="json")
    target = state if key == "ssu_frame" else state["validation"]
    target[key] = [UnreadableRecord() for _ in range(49)]
    with pytest.raises(ValueError):
        SurveyStratifiedThreeStageDesign.model_validate(state)


TYPED_CORRUPTIONS = [
    ("design", "stages", 2),
    ("design", "stages", True),
    ("design", "sampling", "pps"),
    ("design", "max_rows", 1),
    ("validation", "n_cells", True),
    ("strata", "population_psu", 0),
    ("psus", "stratum_index", 99),
    ("cells", "psu_index", 99),
    ("cells", "population_ssu", 0),
    ("ssus", "cell_index", 99),
    ("ssus", "n_tsu", 0),
    ("ssus", "population_tsu", 0),
    ("ssu_frame", "psu", ("float", "a" * 10000)),
    ("ssu_frame", "population_ssu", 0),
]


def copied_invalid_design(design, part, field, value):
    if part == "design":
        return design.model_copy(update={field: value})
    if part == "validation":
        validation = design.validation.model_copy(update={field: value})
    elif part == "ssu_frame":
        records = list(design.ssu_frame)
        records[0] = records[0].model_copy(update={field: value})
        return design.model_copy(update={"ssu_frame": tuple(records)})
    else:
        records = list(getattr(design.validation, part))
        records[0] = records[0].model_copy(update={field: value})
        validation = design.validation.model_copy(update={part: tuple(records)})
    return design.model_copy(update={"validation": validation})


@pytest.mark.parametrize("part,field,value", TYPED_CORRUPTIONS)
def test_typed_model_copy_cannot_skip_nested_or_public_design_validation(part, field, value):
    frame, lower = fixture()
    original = declare(frame, lower)
    changed = copied_invalid_design(original, part, field, value)
    with pytest.raises(ValueError):
        type(changed).model_validate(changed)
    with pytest.raises(AnalysisError, match="Saved three-stage design is invalid"):
        changed.revalidate(frame)
    assert original.revalidate(frame) == original.validation


@pytest.fixture(scope="module")
def valid_linear_fit():
    frame, lower = fixture()
    design = declare(frame, lower)
    return survey_stratified_three_stage_regress(frame, design, "y", ["p"])


@pytest.mark.parametrize("part,field,value", TYPED_CORRUPTIONS)
def test_rehashed_regression_cannot_hide_invalid_typed_design_instances(
    valid_linear_fit, part, field, value
):
    changed = copied_invalid_design(valid_linear_fit.design, part, field, value)
    state = valid_linear_fit.model_dump(mode="json")
    state["design"] = changed.model_dump(mode="json")
    state["integrity_sha256"] = digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    state["design"] = changed
    with pytest.raises(ValueError):
        SurveyStratifiedThreeStageRegressionResult.model_validate(state)


@pytest.mark.parametrize("limit", ["memory", "rows"])
def test_typed_design_admits_parent_budget_before_touching_frame_children(limit):
    class UnreadableFrame(list):
        def __iter__(self):
            raise AssertionError("Typed lower frame parsed before parent budget admission.")

    frame, lower = fixture()
    design = declare(frame, lower)
    changes = {"ssu_frame": UnreadableFrame([{}] * 1000)}
    if limit == "memory":
        changes["validation"] = design.validation.model_copy(update={"nobs": 1000})
        changes["max_memory_mb"] = 1
    else:
        changes["max_rows"] = 1
    with pytest.raises(ValueError):
        type(design).model_validate(design.model_copy(update=changes))


@pytest.mark.parametrize("part", ["strata", "psus", "cells", "ssus"])
def test_typed_geometry_rejects_oversized_children_before_iterating_or_parsing(part):
    class UnreadableRecords(list):
        def __iter__(self):
            raise AssertionError("Typed child records parsed before shape admission.")

    frame, lower = fixture()
    design = declare(frame, lower)
    validation = design.validation.model_copy(update={part: UnreadableRecords([{}] * 49)})
    with pytest.raises(ValueError, match="Nested record dimensions"):
        type(validation).model_validate(validation)
    with pytest.raises(ValueError, match="Nested record dimensions"):
        type(design).model_validate(design.model_copy(update={"validation": validation}))


def test_typed_geometry_rejects_oversized_index_storage_before_numeric_parsing():
    class UnreadableIndices(list):
        def __iter__(self):
            raise AssertionError("Typed index scalars parsed before index shape admission.")

    frame, lower = fixture()
    design = declare(frame, lower)
    children = list(design.validation.ssus)
    children[0] = children[0].model_copy(update={"row_positions": UnreadableIndices([0] * 49)})
    validation = design.validation.model_copy(update={"ssus": tuple(children)})
    with pytest.raises(ValueError, match="Nested index lists"):
        type(validation).model_validate(validation)


@pytest.mark.parametrize(
    "semantics", [None, True, "", "PPS", "calibrated weights", "two-stage SRSWOR"]
)
def test_common_saved_sample_rejects_false_weight_semantics(semantics):
    frame, lower = fixture()
    design = declare(frame, lower)
    target = prepare_stratified_three_stage(frame, design, "mean", ["y"])
    metadata = deepcopy(target.metadata)
    metadata["weight_semantics"] = semantics
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        validate_sample_metadata(metadata, design)
