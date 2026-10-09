"""Independent design geometry and adversarial persistence checks (no estimation)."""

import json
import subprocess
import sys

import pandas as pd
import pytest
from pydantic import ValidationError

import openecon as oe
from openecon.survey import SurveyDesign


def fixture():
    return pd.DataFrame({"w": [2., 3., 4., 2., 2., 3.],
                         "psu": [1, 1, 2, 1, 2, 2],
                         "h": ["a", "a", "a", "b", "b", "b"],
                         "N": [8, 8, 8, 4, 4, 4], "outcome": [1, 9, 2, 8, 3, 7]})


def declare(data=None, **kwargs):
    return oe.survey_design(fixture() if data is None else data, weights="w", psu="psu",
                            strata="h", fpc="N", **kwargs)


def test_geometry_is_nested_and_source_is_unchanged():
    frame = fixture()
    original = frame.copy(deep=True)
    d = declare(frame)
    assert (d.validation.nobs, d.validation.n_strata, d.validation.n_psu,
            d.validation.design_df, d.validation.sum_weights) == (6, 2, 4, 2, 16)
    assert [(s.nobs, s.n_psu, s.population_psu, s.certainty) for s in d.validation.strata] == [
        (3, 2, 8, False), (3, 2, 4, False)]
    pd.testing.assert_frame_equal(frame, original)
    assert d.revalidate(frame) == d.validation


def test_typed_ids_do_not_merge_int_string_bool_or_float():
    frame = {"w": [1.] * 4, "p": [1, "1", True, 1.], "h": ["a"] * 4}
    d = oe.survey_design(frame, weights="w", psu="p", strata="h")
    assert d.validation.n_psu == 4
    assert d.validation.design_df == 3


def test_typed_strata_are_distinct_and_same_psu_label_is_nested():
    frame = {"w": [1.] * 6, "p": [1, 2] * 3, "h": [1, 1, "1", "1", True, True]}
    d = oe.survey_design(frame, weights="w", psu="p", strata="h")
    assert d.validation.n_psu == 6
    assert d.validation.n_strata == 3


def test_json_roundtrip_revalidates_and_changes_are_detected(tmp_path):
    d = declare()
    path = tmp_path / "survey-design.json"
    path.write_text(d.model_dump_json())
    saved = SurveyDesign.model_validate_json(path.read_text())
    assert saved == d and saved.revalidate(fixture()) == d.validation
    frame = fixture()
    frame.loc[0, "w"] += 1
    with pytest.raises(oe.AnalysisError, match="changed") as e:
        saved.revalidate(frame)
    assert e.value.code == "survey_design_changed"
    # Outcomes and index labels are outside the declaration input digest.
    frame = fixture().assign(outcome=100).set_axis(["z"] * 6)
    assert saved.revalidate(frame) == d.validation


def test_hash_retains_order_dtype_and_categorical_metadata():
    d = declare()
    for frame in [fixture().iloc[::-1], fixture().astype({"psu": "object"})]:
        with pytest.raises(oe.AnalysisError):
            d.revalidate(frame)
    frame = fixture().astype({"h": "category"})
    d = declare(frame)
    frame["h"] = frame["h"].cat.add_categories(["unused"])
    with pytest.raises(oe.AnalysisError):
        d.revalidate(frame)


def test_csv_parser_dtype_drift_requires_explicit_schema_restoration(tmp_path):
    frame = fixture()
    d = declare(frame)
    path = tmp_path / "survey.csv"
    frame.to_csv(path, index=False)
    loaded = oe.read(path)
    with pytest.raises(oe.AnalysisError) as e:
        d.revalidate(loaded)
    assert e.value.code == "survey_design_changed"
    d.revalidate(loaded.astype(frame.dtypes.to_dict()))


def test_valid_structure_is_not_trusted_after_json_tampering():
    d = declare()
    record = json.loads(d.model_dump_json())
    record["validation"]["sum_weights"] = 99
    restored = SurveyDesign.model_validate(record)
    with pytest.raises(oe.AnalysisError):
        restored.revalidate(fixture())
    record["validation"]["n_psu"] = 99
    with pytest.raises(ValidationError):
        SurveyDesign.model_validate(record)


@pytest.mark.parametrize("patch", [{"stages": 2}, {"stages": True}, {"stages": "1"},
                                   {"psu": "w"}, {"replicates": []}, {"max_rows": True},
                                   {"schema_version": "future"}, {"max_memory_mb": "64"}])
def test_restored_declaration_fails_closed(patch):
    record = declare().model_dump()
    record.update(patch)
    with pytest.raises(ValidationError):
        SurveyDesign.model_validate(record)


def test_singleton_requires_explicit_certainty_and_census_geometry():
    frame = {"w": [3., 2., 4.], "psu": [1, 2, 2], "h": ["a", "b", "b"], "N": [1, 1, 1]}
    with pytest.raises(oe.AnalysisError):
        declare(frame)
    d = declare(frame, singleton="certainty")
    assert d.validation.design_df == 0
    assert all(s.certainty for s in d.validation.strata)
    frame["N"] = [3, 3, 3]
    with pytest.raises(oe.AnalysisError):
        declare(frame, singleton="certainty")
    with pytest.raises(oe.AnalysisError):
        oe.survey_design(frame, weights="w", psu="psu", strata="h", singleton="certainty")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "2", None, 1j, 10**500])
def test_invalid_weights_are_never_dropped(value):
    frame = fixture().astype({"w": "object"})
    frame.loc[0, "w"] = value
    with pytest.raises(oe.AnalysisError) as e:
        declare(frame)
    assert e.value.code == "invalid_survey_weights"


@pytest.mark.parametrize("value", [0, -1, .5, 1.5, float("nan"), float("inf"), True, "8", 2**54])
def test_fpc_is_unambiguous_population_count(value):
    frame = fixture().astype({"N": "object"})
    frame.loc[frame.h == "a", "N"] = value
    with pytest.raises(oe.AnalysisError):
        declare(frame)


def test_fpc_must_be_constant_and_at_least_sampled_psus():
    for values in [[8, 9, 8, 4, 4, 4], [1, 1, 1, 4, 4, 4]]:
        with pytest.raises(oe.AnalysisError):
            declare(fixture().assign(N=values))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "", "x" * 257, [1, 2], 10**500])
def test_invalid_id_cannot_silently_remove_a_row(value):
    frame = fixture().astype({"psu": "object"})
    frame.at[0, "psu"] = value
    with pytest.raises(oe.AnalysisError):
        declare(frame)


def test_budget_and_unsupported_sources_fail_before_materialization():
    for source in [fixture(), fixture().to_dict("list"), fixture().to_dict("records")]:
        with pytest.raises(oe.AnalysisError) as e:
            declare(source, max_rows=3)
        assert e.value.code == "survey_budget"
    with pytest.raises(oe.AnalysisError):
        declare(max_rows=True)
    with pytest.raises(oe.AnalysisError):
        declare(max_memory_mb="64")
    with pytest.raises(oe.AnalysisError):
        declare(fixture().loc[fixture().index.repeat(200)], max_memory_mb=1)
    with pytest.raises(oe.AnalysisError) as e:
        declare(oe.Dataset.from_frame(fixture()))
    assert e.value.code == "unsupported_survey_input"


def test_duplicate_columns_missing_columns_invalid_roles_and_empty_data():
    for frame in [pd.concat([fixture(), fixture()[["w"]]], axis=1), fixture().drop(columns="w"), fixture().iloc[:0]]:
        with pytest.raises(oe.AnalysisError):
            declare(frame)
    for p in [[], "", True, "w"]:
        with pytest.raises(oe.AnalysisError):
            oe.survey_design(fixture(), weights="w", psu=p)
    with pytest.raises(oe.AnalysisError):
        declare(singleton=[])


def test_overflowing_weight_sum_fails_explicitly():
    with pytest.raises(oe.AnalysisError):
        declare(fixture().assign(w=1e308))


def test_tensor_columns_and_tensor_scalars_are_explicitly_unsupported():
    import torch
    with pytest.raises(oe.AnalysisError) as e:
        oe.survey_design({"w": torch.ones(2), "p": [1, 2]}, weights="w", psu="p")
    assert e.value.code == "unsupported_survey_input"
    frame = fixture().astype({"psu": "object"})
    frame.at[0, "psu"] = torch.tensor(1)
    with pytest.raises(oe.AnalysisError):
        declare(frame)


def test_lazy_declaration_import_does_not_start_tensor_or_estimation_runtime():
    command = "import sys,openecon as oe; assert oe.SurveyDesign; assert oe.survey_design; assert 'torch' not in sys.modules; assert 'openecon.analysis' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
