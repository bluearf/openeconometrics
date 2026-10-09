"""Independent frame, geometry, replay and early-admission acceptance guards."""

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from test_survey_stratified_three_stage_oracles import (
    DESCRIPTIVE,
    GATES,
    ROLES,
    call,
    declare,
    fixture,
    ssu_frame,
)


@pytest.fixture(scope="module")
def base():
    return fixture()


@pytest.fixture(scope="module")
def saved(base):
    frame, design = base
    return {method: call(frame, design, method) for method in GATES}


def rehash(raw):
    raw["integrity_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in raw.items() if key != "integrity_sha256"},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return raw


def frame_rehash(raw):
    raw["validation"]["frame_input_sha256"] = hashlib.sha256(
        json.dumps(
            raw["ssu_frame"],
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return raw


@pytest.mark.parametrize(
    "defect",
    ["missing", "extra", "duplicate", "mismatch", "wrong-parent", "wrong-role", "extra-role"],
)
def test_explicit_frame_requires_exact_complete_positive_cells(base, defect):
    frame, _ = base
    records = ssu_frame(frame)
    if defect == "missing":
        records.pop()
    elif defect == "extra":
        records.append({**records[0], "g": "unobserved-positive-cell"})
    elif defect == "duplicate":
        records.append(dict(records[0]))
    elif defect == "mismatch":
        records[0]["M"] += 1
    elif defect == "wrong-parent":
        records[0]["p"] = "unknown-PSU"
    elif defect == "wrong-role":
        records[0]["population_ssu"] = records[0].pop("M")
    else:
        records[0]["unused"] = 1
    with pytest.raises(AnalysisError):
        declare(frame, records)


@pytest.mark.parametrize(
    "value", [None, True, False, 0, -1, 2.5, float("inf"), float("nan"), 2**53 + 1, 10**1000, "5"]
)
@pytest.mark.parametrize("where", ["declared-M", "N", "M", "L"])
def test_counts_require_exact_positive_bounded_populations(base, where, value):
    frame, _ = base
    records = ssu_frame(frame)
    if where == "declared-M":
        records[0]["M"] = value
    else:
        frame = frame.copy()
        frame[where] = frame[where].astype(object)
        frame.iloc[0, frame.columns.get_loc(where)] = value
    with pytest.raises(AnalysisError):
        declare(frame, records)


@pytest.mark.parametrize("where", ["N", "M", "L"])
def test_population_counts_cannot_change_inside_their_declared_parent(base, where):
    frame, _ = base
    changed = frame.copy()
    changed.iloc[0, changed.columns.get_loc(where)] += 1
    with pytest.raises(AnalysisError):
        declare(changed, ssu_frame(frame))


def test_declared_frame_is_order_independent_but_physical_input_is_bound(base):
    frame, design = base
    reverse_frame = declare(frame, list(reversed(ssu_frame(frame))))
    assert reverse_frame == design
    assert design.revalidate(frame.rename_axis("new-display-index")) == design.validation
    with pytest.raises(AnalysisError):
        design.revalidate(frame.iloc[::-1])
    with pytest.raises(AnalysisError):
        design.revalidate(frame.assign(g="changed-cell"))


def test_complete_frame_cannot_authenticate_an_external_cell_omitted_from_both_inputs(base):
    frame, _ = base
    first = frame.iloc[0]
    keep = ~((frame.h == first.h) & (frame.p == first.p) & (frame.g == first.g))
    supplied = frame.loc[keep].copy()
    declaration = declare(supplied, ssu_frame(supplied))
    assert declaration.validation.n_cells == 17
    with pytest.raises(AnalysisError):
        declare(supplied, ssu_frame(frame))


def small_frame():
    rows = [
        dict(
            h=0,
            p=p,
            g=g,
            s=s,
            j=j,
            N=3,
            M=3 + g,
            L=3,
            x=float(j + s + p + g),
            z=float(j - s + p - g),
            y=float(1 + j * j + s + p + g),
        )
        for p in range(2)
        for g in range(2)
        for s in range(2)
        for j in range(2)
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("census", [False, True])
def test_singletons_are_admitted_only_as_census_at_their_own_stage_or_cell(stage, census):
    frame = small_frame()
    if stage == 1:
        frame = frame[frame.p == 0].copy()
        frame["N"] = 1 if census else 2
    elif stage == 2:
        frame = frame[(frame.g != 0) | (frame.s == 0)].copy()
        frame.loc[frame.g == 0, "M"] = 1 if census else 2
    else:
        frame = frame[(frame.s != 0) | (frame.j == 0)].copy()
        frame.loc[frame.s == 0, "L"] = 1 if census else 2
    if census:
        declaration = declare(frame)
        assert np.isfinite(declaration.validation.weights).all()
        assert len(oe.survey_stratified_three_stage_total(frame, declaration, "y").to_frame()) == 1
    else:
        with pytest.raises(AnalysisError):
            declare(frame)


@pytest.mark.parametrize("role", ["h", "p", "g", "s", "j"])
def test_typed_scalar_identity_keeps_boolean_integer_float_and_string_separate(role):
    # Four labels which compare equal under ordinary Python bool/int/float equality.
    frame = small_frame()
    values = [False, 0, 0.0, "0"]
    if role == "h":
        copies = []
        for value in values:
            copy = frame.copy()
            copy[role] = pd.Series([value] * len(copy), index=copy.index, dtype=object)
            copies.append(copy)
        frame = pd.concat(copies, ignore_index=True)
    elif role in ("p", "g", "s"):
        copies = []
        for value in values:
            copy = frame[frame[role] == 0].copy()
            copy[role] = pd.Series([value] * len(copy), index=copy.index, dtype=object)
            copies.append(copy)
        frame = pd.concat(copies, ignore_index=True)
        frame[{"p": "N", "g": "M", "s": "M"}[role]] = 4 if role != "g" else 3
    else:
        copies = []
        for value in values:
            copy = frame[frame.j == 0].copy()
            copy[role] = pd.Series([value] * len(copy), index=copy.index, dtype=object)
            copies.append(copy)
        frame = pd.concat(copies, ignore_index=True)
        frame["L"] = 4
    design = declare(frame)
    expected = {"h": 4, "p": 4, "g": 8, "s": 16, "j": 8}
    actual = {
        "h": design.validation.n_strata,
        "p": design.validation.n_psu,
        "g": design.validation.n_cells,
        "s": design.validation.n_ssu,
        "j": design.validation.n_ssu,
    }
    assert actual[role] == expected[role]
    assert design.revalidate(frame) == design.validation


def test_frame_typed_identity_does_not_match_equal_boolean_with_integer():
    frame = small_frame()
    frame["g"] = frame.g.astype(object)
    frame.loc[frame.g == 0, "g"] = False
    records = ssu_frame(frame)
    for record in records:
        if type(record["g"]) is bool:
            record["g"] = 0
    with pytest.raises(AnalysisError):
        declare(frame, records)


@pytest.mark.parametrize("column", ["h", "p", "g", "s", "j"])
@pytest.mark.parametrize("value", [None, "", float("nan"), float("inf"), [], "x" * 257, 10**1000])
def test_design_and_frame_identity_scalars_are_bounded(base, column, value):
    frame, _ = base
    frame = frame.copy()
    frame[column] = frame[column].astype(object)
    frame.iat[0, frame.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError):
        declare(frame, ssu_frame(base[0]))


class UntouchedSequence(Sequence):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        raise AssertionError("Oversized input was inspected before admission")


@pytest.mark.parametrize(
    "data",
    [UntouchedSequence(1_000_001), {key: UntouchedSequence(1_000_001) for key in ROLES.values()}],
)
def test_complete_data_row_admission_precedes_iteration_or_dataframe_conversion(data):
    with pytest.raises(AnalysisError):
        oe.survey_stratified_three_stage_design(data, **ROLES, ssu_frame=UntouchedSequence(1))


def test_frame_length_admission_precedes_record_conversion(base):
    frame, _ = base
    with pytest.raises(AnalysisError):
        declare(frame, UntouchedSequence(len(frame) + 1))


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_rows", True),
        ("max_rows", 0),
        ("max_rows", 1_000_001),
        ("max_memory_mb", True),
        ("max_memory_mb", 0),
        ("max_memory_mb", 513),
    ],
)
def test_resource_options_admit_before_input_conversion(option, value):
    with pytest.raises(AnalysisError):
        oe.survey_stratified_three_stage_design(
            UntouchedSequence(1), **ROLES, ssu_frame=UntouchedSequence(1), **{option: value}
        )


@pytest.mark.parametrize(
    "defect",
    [
        "unknown-parent",
        "wrong-cell",
        "wrong-frame",
        "duplicate-cell",
        "duplicate-rows",
        "missing-rows",
        "cell-count",
        "frame-count",
        "typed-index",
        "inconsistent-census",
        "frame-hash",
        "frame-population",
        "typed-frame",
    ],
)
def test_saved_geometry_rejects_broken_complete_cell_or_parent_partitions(base, defect):
    _, design = base
    raw = design.model_dump(mode="json")
    v = raw["validation"]
    if defect == "unknown-parent":
        v["cells"][0]["psu_index"] = v["n_psu"]
    elif defect == "wrong-cell":
        v["ssus"][0]["cell_index"] = (v["ssus"][0]["cell_index"] + 1) % v["n_cells"]
    elif defect == "wrong-frame":
        v["cells"][0]["frame_index"] = (v["cells"][0]["frame_index"] + 1) % v["n_cells"]
    elif defect == "duplicate-cell":
        v["psus"][0]["cell_indices"].append(v["psus"][0]["cell_indices"][0])
    elif defect == "duplicate-rows":
        v["cells"][0]["row_positions"].append(v["cells"][0]["row_positions"][0])
    elif defect == "missing-rows":
        v["cells"][0]["row_positions"].pop()
    elif defect == "cell-count":
        v["n_cells"] -= 1
    elif defect == "frame-count":
        raw["ssu_frame"].pop()
    elif defect == "typed-index":
        v["cells"][0]["psu_index"] = False
    elif defect == "inconsistent-census":
        v["cells"][0]["population_ssu"] = v["cells"][0]["n_ssu"] - 1
    elif defect == "frame-hash":
        v["frame_input_sha256"] = "0" * 64
    elif defect == "frame-population":
        raw["ssu_frame"][0]["population_ssu"] += 1
        frame_rehash(raw)
    elif defect == "typed-frame":
        raw["ssu_frame"][0]["psu"] = ["int", False]
        frame_rehash(raw)
    with pytest.raises(ValidationError):
        oe.SurveyStratifiedThreeStageDesign.model_validate_json(json.dumps(raw, allow_nan=False))


@pytest.mark.parametrize("method", GATES)
@pytest.mark.parametrize(
    "key,value",
    [
        ("n_lower_strata", False),
        ("n_lower_strata", 1),
        ("lower_stage_stratification", "stage2 pooled SSUs"),
        ("ssu_frame_sha256", "0" * 64),
    ],
)
def test_rehashed_saved_results_reject_frame_metadata_forgery(saved, method, key, value):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][key] = value
    with pytest.raises(ValidationError):
        type(state).model_validate_json(json.dumps(rehash(raw)))


@pytest.mark.parametrize("method", GATES)
@pytest.mark.parametrize("part", ["stage1_covariance", "stage2_covariance", "stage3_covariance"])
def test_rehashed_saved_results_replay_each_stage_not_just_sum(saved, method, part):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][part][0][0] += 0.125
    raw["covariance"][0][0] += 0.125
    with pytest.raises(ValidationError):
        type(state).model_validate_json(json.dumps(rehash(raw)))


@pytest.mark.parametrize("method", GATES)
@pytest.mark.parametrize("defect", ["covariance-rows", "covariance-columns", "estimate", "null"])
def test_saved_numeric_dimensions_are_rejected_before_scalar_protocol_callbacks(
    saved, method, defect
):
    state = saved[method]
    raw = state.model_dump(mode="python")
    raw["covariance"] = [list(row) for row in raw["covariance"]]
    conversions = []

    class NumericProbe:
        def __float__(self):
            conversions.append(True)
            return 0.0

    if defect == "covariance-rows":
        raw["covariance"].append([NumericProbe() for _ in state.labels])
    elif defect == "covariance-columns":
        raw["covariance"][0].append(NumericProbe())
    else:
        key = (
            "null" if defect == "null" else "estimates" if method in DESCRIPTIVE else "coefficients"
        )
        raw[key] = [*raw[key], NumericProbe()]
    with pytest.raises(ValidationError) as error:
        type(state).model_validate(raw)
    assert conversions == []
    assert len(error.value.errors()) == 1 and error.value.errors()[0]["loc"] == ()


@pytest.mark.parametrize("method", DESCRIPTIVE)
@pytest.mark.parametrize("key", ["primitive_values", "primitive_denominators", "row_influences"])
def test_saved_descriptive_oversized_numeric_primitives_reject_as_validation(saved, method, key):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][key][0][0] = 10**1000
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(raw))


@pytest.mark.parametrize("method", GATES)
def test_result_class_is_distinct_and_refuses_old_three_stage_schema(saved, method):
    state = saved[method]
    raw = state.model_dump(mode="json")
    expected = (
        oe.SurveyStratifiedThreeStageResult
        if method in DESCRIPTIVE
        else oe.SurveyStratifiedThreeStageRegressionResult
    )
    assert type(state) is expected
    raw["schema_version"] = raw["schema_version"].replace("stratified-", "")
    with pytest.raises(ValidationError):
        expected.model_validate(rehash(raw))


@pytest.mark.parametrize("method", DESCRIPTIVE)
@pytest.mark.parametrize(
    "key,value",
    [
        ("calibration_support", True),
        ("weight_semantics", "arbitrary caller weights"),
        ("variance_formula", "pooled second-stage centering"),
        ("design_effect", 1),
        ("confidence_interval", "exact population coverage"),
    ],
)
@pytest.mark.parametrize("typed_copy", [False, True])
def test_descriptive_classification_cannot_overclaim_saved_method(
    saved, method, key, value, typed_copy
):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"][key] = value
    rehash(raw)
    mutated = (
        state.model_copy(
            update={"metadata": raw["metadata"], "integrity_sha256": raw["integrity_sha256"]}
        )
        if typed_copy
        else raw
    )
    with pytest.raises(ValidationError):
        type(state).model_validate(mutated)


@pytest.mark.parametrize("method", ["regress", "logit", "probit", "poisson"])
@pytest.mark.parametrize("typed_copy", [False, True])
def test_model_classification_cannot_claim_arbitrary_weights(saved, method, typed_copy):
    state = saved[method]
    raw = state.model_dump(mode="json")
    raw["metadata"]["weight_semantics"] = "arbitrary caller weights"
    rehash(raw)
    mutated = (
        state.model_copy(
            update={"metadata": raw["metadata"], "integrity_sha256": raw["integrity_sha256"]}
        )
        if typed_copy
        else raw
    )
    with pytest.raises(ValidationError):
        type(state).model_validate(mutated)


def test_native_helper_retains_legacy_33_modules_and_adds_exact_five_new_modules():
    root = Path(__file__).resolve().parents[1]

    def modules(path):
        import ast

        tree = ast.parse(path.read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MODULES" for t in node.targets)
        )
        return ast.literal_eval(assignment.value)

    previous = modules(root / "scripts/verify_survey_three_stage_installed.py")
    current = modules(root / "scripts/verify_survey_stratified_three_stage_installed.py")
    assert len(previous) == 33 and current[:33] == previous
    assert len(current) == len(set(current)) == 38
    assert set(current[33:]) == {
        "openecon.survey_stratified_three_stage",
        *[
            "openecon.econometrics.survey.stratified_three_stage_" + name
            for name in ("common", "targets", "regression", "regression_state")
        ],
    }


def test_standalone_oracle_requires_no_production_or_test_imports():
    import ast

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/verify_survey_stratified_three_stage_oracles.py"
    ).read_text()
    imports = [
        node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        name and (name.startswith("openecon") or name.startswith("test_") or name == "pytest")
        for name in imports
    )
