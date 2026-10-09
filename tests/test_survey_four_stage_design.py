"""Four real selection levels, typed geometry, FPCs and early admission guards."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from openecon.survey_four_stage import (
    FourStageValidation,
    SurveyFourStageDesign,
    survey_four_stage_design,
)
from openecon.econometrics.survey.four_stage_common import (
    prepare_four_stage,
    stage_covariances,
    validate_identity,
    validate_sample_metadata,
)

ROLES = dict(
    psu="p",
    ssu="s",
    tsu="t",
    fsu="u",
    strata="h",
    population_psu="N",
    population_ssu="M",
    population_tsu="L",
    population_fsu="K",
)


def fixture():
    rows = []
    for h in range(2):
        for p in range(2):
            for s in range(2):
                for t in range(2):
                    for u in range(4):
                        x = (-1.5, -0.5, 0.5, 1.5)[u]
                        z = s + 0.2 * p + 0.1 * h + 0.02 * t
                        rows.append(
                            dict(
                                h=h,
                                p=p,
                                s=s,
                                t=t,
                                u=u,
                                N=4 + h,
                                M=3 + p,
                                L=4 + s,
                                K=6 + t,
                                x=x,
                                z=z,
                                y=1.5 + 0.6 * x - 0.3 * z + 0.1 * u * u + 0.15 * h * p,
                            den=3.0 + 0.1 * x + 0.2 * z,
                            den2=3.0 + 0.1 * x + 0.2 * z,
                                b=(0, 1, 1, 0)[(u + h + p + s + t) % 4],
                                c=(0, 2, 1, 3)[(u + h + p + s + t) % 4],
                                cat=("a", "b")[(u + h + p + s + t) % 2],
                                domain=1,
                            )
                        )
    return pd.DataFrame(rows, index=np.resize([9, -1, 9], len(rows)))


def declare(frame=None, **options):
    return survey_four_stage_design(fixture() if frame is None else frame, **(ROLES | options))


def test_complete_weights_and_all_parent_descendants_are_physical_and_immutable():
    frame = fixture()
    before = frame.copy(deep=True)
    design = declare(frame)
    v = design.validation
    assert (v.nobs, v.n_strata, v.n_psu, v.n_ssu, v.n_tsu, v.design_df) == (64, 2, 4, 8, 16, 2)
    np.testing.assert_array_equal(v.weights, frame.N / 2 * frame.M / 2 * frame.L / 2 * frame.K / 4)
    assert all(len(t.row_positions) == t.n_fsu == 4 for t in v.tsus)
    for q in v.ssus:
        assert q.row_positions == tuple(
            sorted(i for t in q.tsu_indices for i in v.tsus[t].row_positions)
        )
    for p in v.psus:
        assert p.row_positions == tuple(
            sorted(i for s in p.ssu_indices for i in v.ssus[s].row_positions)
        )
    assert design.revalidate(frame) == v
    assert (
        SurveyFourStageDesign.model_validate_json(
            json.dumps(design.model_dump(mode="json"), sort_keys=True)
        )
        == design
    )
    pd.testing.assert_frame_equal(before, frame)
    with pytest.raises(ValidationError):
        design.stages = 3


@pytest.mark.parametrize("kind", ["dataframe", "columns", "records"])
def test_resident_representations_retain_same_ordered_geometry(kind):
    frame = fixture()
    data = (
        frame if kind == "dataframe" else frame.to_dict("list" if kind == "columns" else "records")
    )
    assert declare(data) == declare(frame)


def test_noncontiguous_interleaved_parents_keep_exact_original_positions():
    frame = fixture().sample(frac=1, random_state=9)
    v = declare(frame).validation
    for t in v.tsus:
        rows = frame.iloc[list(t.row_positions)]
        assert len(rows[["h", "p", "s", "t"]].drop_duplicates()) == 1
        assert len(rows.u.unique()) == t.n_fsu
    assert any(
        t.row_positions != tuple(range(t.row_positions[0], t.row_positions[0] + t.n_fsu))
        for t in v.tsus
    )


def test_optional_first_stage_strata_still_has_four_real_selection_levels():
    frame = fixture().query("h == 0").drop(columns="h")
    design = declare(frame, strata=None)
    assert (
        design.validation.n_strata,
        design.validation.n_psu,
        design.validation.n_ssu,
        design.validation.n_tsu,
    ) == (1, 2, 4, 8)
    assert design.revalidate(frame) == design.validation


@pytest.mark.parametrize("role", ["h", "p", "s", "t", "u"])
def test_typed_bool_int_float_and_string_identifiers_never_collapse(role):
    frame = fixture().astype(object)
    identity = [False, 0, 0.0, "0"]
    # Expand one level to four typed labels, retaining all lower-level units.
    rows = []
    selected = frame[frame[role] == 0]
    for label in identity:
        for row in selected.to_dict("records"):
            rows.append(row | {role: label, "N": 8, "M": 8, "L": 8, "K": 8})
    v = declare(pd.DataFrame(rows, dtype=object)).validation
    expected = {"h": 4, "p": 8, "s": 16, "t": 32, "u": 16}
    actual = {"h": v.n_strata, "p": v.n_psu, "s": v.n_ssu, "t": v.n_tsu, "u": v.n_tsu}
    assert actual[role] == expected[role]


@pytest.mark.parametrize("role", ["N", "M", "L", "K"])
@pytest.mark.parametrize(
    "value", [None, True, 0, -1, 0.5, 2**53 + 1, "6", np.nan, np.inf, 6 + 0j, 10**500]
)
def test_populations_require_positive_exact_bounded_real_counts(role, value):
    frame = fixture().astype({role: object})
    frame.iloc[0, frame.columns.get_loc(role)] = value
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("role", ["h", "p", "s", "t", "u"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, "", "x" * 257, 1j, [], {}, 10**300])
def test_nonfinite_nonscalar_or_oversized_ids_are_rejected(role, value):
    frame = fixture().astype({role: object})
    frame.iat[0, frame.columns.get_loc(role)] = value
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("role", ["N", "M", "L", "K"])
def test_population_is_constant_only_within_its_actual_sampled_parent(role):
    frame = fixture()
    frame.iat[0, frame.columns.get_loc(role)] += 1
    with pytest.raises(AnalysisError, match="population"):
        declare(frame)


def test_identical_fsu_labels_across_tsus_are_valid_but_duplicate_within_tsu_is_not():
    frame = fixture()
    assert declare(frame).validation.n_tsu == 16
    frame.iat[1, frame.columns.get_loc("u")] = frame.iat[0, frame.columns.get_loc("u")]
    with pytest.raises(AnalysisError) as error:
        declare(frame)
    assert error.value.code == "invalid_survey_nesting"


@pytest.mark.parametrize("level, count", [("p", "N"), ("s", "M"), ("t", "L"), ("u", "K")])
@pytest.mark.parametrize("census", [False, True])
def test_singleton_only_when_its_own_stage_population_is_one(level, count, census):
    frame = fixture().query(f"{level} == 0").copy()
    frame[count] = 1 if census else 2
    if census:
        assert declare(frame).validation.nobs == len(frame)
    else:
        with pytest.raises(AnalysisError, match="singleton"):
            declare(frame)


@pytest.mark.parametrize("pattern", range(16))
def test_every_stage_census_pattern_zeroes_exactly_its_own_component(pattern):
    frame = fixture()
    counts = [2, 2, 2, 4]
    for stage, (role, count) in enumerate(zip(["N", "M", "L", "K"], counts)):
        frame[role] = count if pattern & (1 << stage) else count + 2
    design = declare(frame)
    row = torch.tensor(np.column_stack([frame.y, frame.y**2]), dtype=torch.float64)
    parts = stage_covariances(torch.tensor(design.validation.weights)[:, None] * row, design)
    for stage, covariance in enumerate(parts):
        if pattern & (1 << stage):
            assert torch.equal(covariance, torch.zeros_like(covariance))
        else:
            assert float(covariance.diag().max()) > 0


@pytest.mark.parametrize("field", ["fsu", "population_fsu", "strata"])
def test_every_role_must_be_distinct(field):
    with pytest.raises(AnalysisError):
        declare(**{field: "psu" if field == "strata" else "p"})


@pytest.mark.parametrize("count", [True, 0, -1, 0.5, "4", 5])
def test_saved_stages_is_exact_four_and_not_boolean(count):
    raw = declare().model_dump(mode="json") | {"stages": count}
    with pytest.raises(ValidationError):
        SurveyFourStageDesign.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["population", "fsu", "dtype", "row_order", "category_dictionary"]
)
def test_revalidate_binds_all_complete_design_rows_and_role_types(change):
    frame = fixture()
    if change == "category_dictionary":
        frame["u"] = pd.Categorical(frame.u, categories=[0, 1, 2, 3, 4])
    design = declare(frame)
    altered = frame.copy()
    if change == "population":
        altered["K"] += 1
    elif change == "fsu":
        altered["u"] += 10
    elif change == "dtype":
        altered["u"] = altered.u.astype(float)
    elif change == "row_order":
        altered = altered.iloc[::-1]
    else:
        altered["u"] = altered.u.cat.remove_unused_categories()
    with pytest.raises(AnalysisError) as error:
        design.revalidate(altered)
    assert error.value.code == "survey_design_changed"


@pytest.mark.parametrize(
    "record, field, value",
    [
        ("strata", "population_psu", 1),
        ("psus", "population_ssu", 1),
        ("ssus", "population_tsu", 1),
        ("tsus", "population_fsu", 1),
        ("psus", "stratum_index", 1),
        ("ssus", "psu_index", 1),
        ("tsus", "ssu_index", 1),
        ("tsus", "n_fsu", 0),
        ("ssus", "n_tsu", 0),
        ("psus", "row_positions", [0]),
        ("ssus", "tsu_indices", [0, 0]),
        ("strata", "psu_indices", [0, 0]),
        ("tsus", "row_positions", [0, 1, 2, 4]),
    ],
)
def test_nested_typed_model_copy_must_revalidate_before_use(record, field, value):
    design = declare()
    records = list(getattr(design.validation, record))
    records[0] = records[0].model_copy(update={field: value})
    validation = design.validation.model_copy(update={record: tuple(records)})
    corrupted = design.model_copy(update={"validation": validation})
    with pytest.raises((ValidationError, AnalysisError)):
        SurveyFourStageDesign.model_validate(corrupted)
    with pytest.raises(AnalysisError):
        corrupted.revalidate(fixture())


@pytest.mark.parametrize(
    "field,value", [("stages", 3), ("sampling", "pps"), ("max_memory_mb", "64"), ("max_rows", 1)]
)
def test_typed_design_copy_never_skips_literal_or_resource_contract(field, value):
    design = declare().model_copy(update={field: value})
    with pytest.raises((ValidationError, AnalysisError)):
        SurveyFourStageDesign.model_validate(design)
    with pytest.raises(AnalysisError):
        design.revalidate(fixture())


def test_saved_partition_mutation_fails_even_if_sum_weights_is_recomputed():
    raw = declare().model_dump(mode="json")
    raw["validation"]["tsus"][1]["row_positions"] = raw["validation"]["tsus"][0]["row_positions"]
    with pytest.raises(ValidationError):
        SurveyFourStageDesign.model_validate(raw)


class NumericProbe:
    def __float__(self):
        raise AssertionError("resource admission must precede numeric conversion")

    def __int__(self):
        raise AssertionError("resource admission must precede numeric conversion")


def test_large_saved_design_admission_precedes_nested_numeric_conversion():
    raw = declare().model_dump(mode="json")
    raw["max_memory_mb"] = 1
    raw["validation"]["nobs"] = 1000
    raw["validation"]["sum_weights"] = NumericProbe()
    with pytest.raises(ValidationError, match="workspace bytes"):
        SurveyFourStageDesign.model_validate(raw)


def test_nested_geometry_shape_admission_precedes_primitive_numeric_conversion():
    raw = declare().validation.model_dump(mode="json")
    raw["tsus"][0]["row_positions"] = [0] * 65
    raw["sum_weights"] = NumericProbe()
    with pytest.raises(ValidationError, match="index"):
        FourStageValidation.model_validate(raw)


def test_ambient_workspace_ceiling_binds_design_and_restoration():
    design = declare()
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            declare()
        assert error.value.code == "workspace_limit"
        with pytest.raises(ValidationError, match="workspace bytes"):
            SurveyFourStageDesign.model_validate(design)


@pytest.mark.parametrize("key", ["n_design", "n_domain", "n_used", "n_tsu", "stages"])
def test_saved_metadata_counts_are_exact_integers(key):
    design = declare()
    sample = prepare_four_stage(fixture(), design, "mean", ["y"])
    altered = deepcopy(sample.metadata)
    altered[key] = True
    with pytest.raises(ValueError):
        validate_sample_metadata(altered, design)


@pytest.mark.parametrize(
    "identity",
    [
        ["float", "0x" + "1" * 100000 + "p0"],
        ["str", "x" * 257],
        ["int", 10**300],
        ["float", "0x1p+999999999"],
        ["int", True],
    ],
)
def test_saved_category_identity_is_bounded_before_canonical_decode(identity):
    with pytest.raises(ValueError):
        validate_identity(identity)
