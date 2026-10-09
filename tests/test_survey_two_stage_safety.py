"""Two-stage admission, complete geometry and adversarial saved-state guards.

The independent finite-population and numerical covariance oracles live in the
separate mathematics tests. These cases exercise the method's input contract.
"""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


METHODS = ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson")
REGRESSION = ("regress", "logit", "probit", "poisson")


def fixture():
    """Repeated labels across parents, unequal stage weights and finite GLM fits."""
    rows = []
    for h in range(2):
        for p in range(3):
            for j in range(4):
                x = (-1.5, -.5, .5, 1.5)[j]
                z = (-.7, .2, .9)[p] + .15 * h
                rows.append({
                    "h": ("north", "south")[h], "p": p + 1, "s": j + 1,
                    "N": (6, 9)[h], "M": (8, 12, 16)[p], "x": x, "z": z,
                    "y": 1.1 + .6 * x - .3 * z + (j * p + h) / 7 + j**2 / 20,
                    "den": 2.5 + .2 * x + .1 * z,
                    "binary": (j + p + h) % 2,
                    "count": (0, 2, 1, 3)[(j + p + h) % 4],
                    "category": ("a", "b")[(j + p + h) % 2], "domain": 1,
                })
    return pd.DataFrame(rows, index=[9, -3, 9, 0] * 6)


def declare(frame=None, **options):
    return oe.survey_two_stage_design(
        fixture() if frame is None else frame, psu="p", ssu="s", strata="h",
        population_psu="N", population_ssu="M", **options,
    )


def run(frame, design, method, **options):
    fn = getattr(oe, "survey_two_stage_" + method)
    if method == "ratio":
        return fn(frame, design, ["y", "den"], ["den", "y"], **options)
    if method == "proportion":
        return fn(frame, design, "category", categories=["a", "b", "absent"], **options)
    if method in REGRESSION:
        outcome = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[method]
        return fn(frame, design, outcome, ["x", "z"], **options)
    return fn(frame, design, ["y", "den"], **options)


def rehash(payload):
    """A digest does not authenticate a result: deliberately recompute it."""
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    payload["integrity_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return payload


def refresh_target_fingerprint(payload):
    metadata = payload["metadata"]
    selected = [i in metadata["sample_positions"] for i in range(metadata["n_design"])]
    primitive = [payload["design"]["validation"]["design_input_sha256"], selected,
                 metadata["primitive_values"], metadata["primitive_denominators"]]
    metadata["sample_input_sha256"] = hashlib.sha256(json.dumps(
        primitive, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return rehash(payload)


def refresh_regression_fingerprint(payload):
    """Recompute the shared admitted numeric sample fingerprint after mutation."""
    metadata = payload["metadata"]
    n = metadata["n_design"]
    width = len(payload["regressors"]) + 1
    values = [[0.] * width for _ in range(n)]
    for i, physical in enumerate(metadata["sample_positions"]):
        values[physical] = [metadata["primitive_y"][i], *metadata["primitive_X"][i][int(payload["intercept"]):]]
    selected = [i in metadata["sample_positions"] for i in range(n)]
    primitive = [payload["design"]["validation"]["design_input_sha256"], selected,
                 values, [[1.] * width for _ in range(n)]]
    metadata["sample_input_sha256"] = hashlib.sha256(json.dumps(
        primitive, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return rehash(payload)


@pytest.fixture(scope="module")
def complete_states():
    frame = fixture()
    design = declare(frame)
    return {method: run(frame, design, method) for method in METHODS}


@pytest.mark.parametrize("method", METHODS)
def test_independently_rehashed_unmodified_primitives_still_restore(method, complete_states):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    refreshed = (refresh_regression_fingerprint(payload) if method in REGRESSION else
                 refresh_target_fingerprint(payload))
    assert refreshed["metadata"]["sample_input_sha256"] == state.metadata["sample_input_sha256"]
    assert refreshed["integrity_sha256"] == state.integrity_sha256
    assert type(state).model_validate(refreshed).model_dump(mode="json") == state.model_dump(mode="json")


def test_complete_nested_geometry_and_derived_probabilities_do_not_mutate_source():
    frame = fixture()
    before = frame.copy(deep=True)
    design = declare(frame)
    v = design.validation
    assert (v.nobs, v.n_strata, v.n_psu, v.design_df) == (24, 2, 6, 4)
    assert [(h.n_psu, h.population_psu) for h in v.strata] == [(3, 6), (3, 9)]
    assert [p.n_ssu for p in v.psus] == [4] * 6
    assert [p.population_ssu for p in v.psus] == [8, 12, 16] * 2
    assert [p.stratum_index for p in v.psus] == [0] * 3 + [1] * 3
    assert [p.row_positions for p in v.psus] == [tuple(range(i, i + 4)) for i in range(0, 24, 4)]
    weights = np.repeat([4., 6., 8., 6., 9., 12.], 4)
    np.testing.assert_array_equal(design.validation.weights, weights)
    assert v.sum_weights == weights.sum()
    assert design.revalidate(frame) == v
    pd.testing.assert_frame_equal(frame, before)


def test_typed_psu_and_ssu_labels_remain_distinct_inside_their_parents():
    frame = {
        "h": ["a"] * 8, "p": [1, 1, "1", "1", True, True, 1., 1.],
        "s": [1, "1"] * 4, "N": [8] * 8, "M": [4] * 8,
    }
    d = declare(frame)
    assert d.validation.n_psu == 4
    assert [p.n_ssu for p in d.validation.psus] == [2] * 4
    assert d.validation.design_df == 3


def test_typed_strata_and_same_psu_ssu_labels_across_parents_are_valid():
    frame = {
        "h": [1] * 4 + ["1"] * 4 + [True] * 4,
        "p": [1, 1, 2, 2] * 3, "s": [1, 2] * 6,
        "N": [4] * 12, "M": [4] * 12,
    }
    d = declare(frame)
    assert (d.validation.n_strata, d.validation.n_psu, d.validation.nobs) == (3, 6, 12)


def test_duplicate_ssu_inside_one_psu_is_rejected_even_with_identical_outcomes():
    frame = fixture()
    frame.iloc[1, frame.columns.get_loc("s")] = frame.iloc[0].s
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column", ["N", "M"])
@pytest.mark.parametrize("value", [0, -1, .5, 1.5, float("nan"), float("inf"), True,
                                   "8", None, 1j, 2**53 + 1, 10**500])
def test_stage_population_counts_are_exact_positive_bounded_scalars(column, value):
    frame = fixture().astype({column: "object"})
    selection = frame.h.eq("north") if column == "N" else (frame.h.eq("north") & frame.p.eq(1))
    for position in np.flatnonzero(selection.to_numpy()):
        frame.iat[position, frame.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column", ["N", "M"])
def test_population_count_must_be_constant_within_its_stage_group(column):
    frame = fixture()
    frame.iat[0, frame.columns.get_loc(column)] += 1
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column,value", [("N", 2), ("M", 3)])
def test_population_count_cannot_be_less_than_complete_sample_geometry(column, value):
    frame = fixture().assign(**{column: value})
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column", ["h", "p", "s"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "", "x" * 257,
                                   [1, 2], 10**500])
def test_invalid_stage_id_is_never_silently_removed(column, value):
    frame = fixture().astype({column: "object"})
    frame.iat[0, frame.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError):
        declare(frame)


def test_noncensus_singletons_raise_at_both_stages_and_census_singletons_are_admitted():
    for frame in [pd.DataFrame({"h": ["a", "a"], "p": [1, 1], "s": [1, 2], "N": [2, 2], "M": [4, 4]}),
                  pd.DataFrame({"h": ["a", "a"], "p": [1, 2], "s": [1, 1], "N": [4, 4], "M": [2, 2]})]:
        with pytest.raises(AnalysisError):
            declare(frame)
    for frame in [pd.DataFrame({"h": ["a", "a"], "p": [1, 1], "s": [1, 2], "N": [1, 1], "M": [4, 4]}),
                  pd.DataFrame({"h": ["a", "a"], "p": [1, 2], "s": [1, 1], "N": [4, 4], "M": [1, 1]})]:
        assert declare(frame).validation.nobs == 2


def test_declaration_json_restoration_preserves_exact_design_and_requires_revalidation():
    frame = fixture()
    design = declare(frame)
    restored = type(design).model_validate_json(design.model_dump_json())
    assert restored == design and restored.revalidate(frame) == design.validation
    assert restored.revalidate(frame.assign(y=100).set_axis(["duplicate"] * len(frame))) == design.validation
    for changed in [frame.iloc[::-1], frame.astype({"p": "object"}), frame.assign(M=frame.M + 1)]:
        with pytest.raises(AnalysisError):
            restored.revalidate(changed)
    categorical = frame.astype({"h": "category"})
    d = declare(categorical)
    categorical["h"] = categorical.h.cat.add_categories(["unused"])
    with pytest.raises(AnalysisError):
        d.revalidate(categorical)


@pytest.mark.parametrize("patch", [
    {"stages": 1}, {"stages": 3}, {"stages": True}, {"stages": "2"},
    {"schema_version": "future"}, {"psu": "s"}, {"population_psu": "M"},
    {"weights": "w"}, {"max_rows": True}, {"max_memory_mb": "64"},
])
def test_saved_declaration_rejects_unsupported_or_coerced_schema(patch):
    original = declare()
    payload = original.model_dump(mode="json")
    payload.update(patch)
    with pytest.raises(ValidationError):
        type(original).model_validate(payload)


@pytest.mark.parametrize("defect", ["wrong-nobs", "wrong-psu-count", "wrong-stratum", "duplicate-row",
                                    "missing-row", "boolean-row", "outside-row", "wrong-weight-sum"])
def test_saved_geometry_is_structurally_checked_and_not_trusted_on_revalidation(defect):
    frame = fixture()
    original = declare(frame)
    payload = original.model_dump(mode="json")
    v = payload["validation"]
    if defect == "wrong-nobs":
        v["nobs"] += 1
    elif defect == "wrong-psu-count":
        v["n_psu"] += 1
    elif defect == "wrong-stratum":
        v["psus"][0]["stratum_index"] = 1
    elif defect == "duplicate-row":
        v["psus"][0]["row_positions"][1] = 0
    elif defect == "missing-row":
        v["psus"][0]["row_positions"].pop()
    elif defect == "boolean-row":
        v["psus"][0]["row_positions"][0] = False
    elif defect == "outside-row":
        v["psus"][0]["row_positions"][0] = len(frame)
    else:
        v["sum_weights"] += 1
    with pytest.raises((ValueError, AnalysisError)):
        saved = type(original).model_validate(payload)
        saved.revalidate(frame)


@pytest.mark.parametrize("source_kind", ["frame", "columns", "rows"])
def test_complete_row_budget_is_enforced_before_estimation(source_kind):
    frame = fixture()
    source = {"frame": frame, "columns": frame.to_dict("list"), "rows": frame.to_dict("records")}[source_kind]
    with pytest.raises(AnalysisError) as error:
        declare(source, max_rows=3)
    assert error.value.code == "survey_budget"


@pytest.mark.parametrize("options", [{"max_rows": True}, {"max_rows": 0}, {"max_rows": 1_000_001},
                                     {"max_memory_mb": "64"}, {"max_memory_mb": 0}, {"max_memory_mb": 513}])
def test_declaration_scalar_resource_limits_fail_closed(options):
    with pytest.raises(AnalysisError):
        declare(**options)


def test_oversized_column_sources_are_refused_before_materializing_rows():
    class ForbiddenRows:
        def __len__(self):
            return 1_000_001

        def __iter__(self):
            raise AssertionError("Oversized source reached column materialization.")

    source = {name: ForbiddenRows() for name in ("p", "s", "h", "N", "M")}
    with pytest.raises(AnalysisError) as error:
        declare(source)
    assert error.value.code == "survey_budget"


def test_duplicate_missing_empty_and_unsupported_design_inputs():
    frame = fixture()
    for source in [pd.concat([frame, frame[["s"]]], axis=1), frame.drop(columns="M"), frame.iloc[:0]]:
        with pytest.raises(AnalysisError):
            declare(source)
    with pytest.raises(AnalysisError):
        declare(oe.Dataset.from_frame(frame))
    source = frame.to_dict("list")
    source["s"] = torch.arange(len(frame))
    with pytest.raises(AnalysisError):
        declare(source)
    frame = frame.astype({"s": "object"})
    frame.iat[0, frame.columns.get_loc("s")] = torch.tensor(1)
    with pytest.raises(AnalysisError):
        declare(frame)


def test_two_stage_declaration_stays_lazy_without_starting_tensor_runtime():
    command = ("import sys,openecon as oe; assert oe.SurveyTwoStageDesign; "
               "assert oe.survey_two_stage_design; assert 'torch' not in sys.modules; "
               "assert 'openecon.analysis' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("role,value", [("psu", ""), ("ssu", True), ("strata", []),
                                       ("population_psu", "p"), ("population_ssu", "absent")])
def test_invalid_design_roles_fail_explicitly(role, value):
    options = {"psu": "p", "ssu": "s", "strata": "h", "population_psu": "N", "population_ssu": "M"}
    options[role] = value
    with pytest.raises(AnalysisError):
        oe.survey_two_stage_design(fixture(), **options)


@pytest.mark.parametrize("method", METHODS)
def test_domain_and_missing_rows_retain_complete_stage_geometry(method):
    frame = fixture()
    design = declare(frame)
    frame.loc[frame.h.eq("south") & frame.p.eq(3), "domain"] = 0
    outcomes = ["y", "den", "binary", "count", "category", "x", "z"]
    outside = frame.domain.eq(0)
    frame.loc[outside, outcomes] = np.nan
    before = frame.copy(deep=True)
    state = run(frame, design, method, domain="domain")
    assert state.design == design and state.df == 4
    assert state.metadata["n_design"] == 24
    assert state.metadata["sample_positions"] == list(range(20))
    assert state.design.validation.psus[-1].row_positions == (20, 21, 22, 23)
    pd.testing.assert_frame_equal(frame, before)
    frame.iat[0, frame.columns.get_loc("y" if method in ("mean", "total", "ratio", "regress") else
                                          "category" if method == "proportion" else
                                          "count" if method == "poisson" else "binary")] = np.nan
    with pytest.raises(AnalysisError):
        run(frame, design, method, domain="domain")
    before_drop = frame.copy(deep=True)
    dropped = run(frame, design, method, domain="domain", missing="drop")
    assert dropped.metadata["sample_positions"] == list(range(1, 20))
    assert dropped.design.validation == design.validation and dropped.df == 4
    assert dropped.metadata["outcome_exclusions"] == [0]
    pd.testing.assert_frame_equal(frame, before_drop)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("domain", [0, .5, float("nan"), "yes"])
def test_empty_or_nonbinary_domains_are_rejected(method, domain):
    frame = fixture().assign(domain=domain)
    with pytest.raises(AnalysisError):
        run(frame, declare(frame), method, domain="domain")


@pytest.mark.parametrize("method", METHODS)
def test_estimation_revalidates_stage_design_before_use(method):
    frame = fixture()
    design = declare(frame)
    frame["M"] += 1
    with pytest.raises(AnalysisError) as error:
        run(frame, design, method)
    assert error.value.code == "survey_design_changed"


@pytest.mark.parametrize("method", METHODS)
def test_one_stage_declaration_cannot_be_used_as_a_two_stage_design(method):
    frame = fixture().assign(w=1.)
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N")
    with pytest.raises(AnalysisError) as error:
        run(frame, design, method)
    assert error.value.code == "invalid_survey_design"


@pytest.mark.parametrize("method", METHODS)
def test_first_stage_census_with_positive_within_variance_has_no_df_zero_intervals(method):
    frame = fixture().loc[lambda x: x.p.eq(1)].copy()
    frame["N"] = 1
    state = run(frame, declare(frame), method)
    table = state.to_frame()
    positive = table.std_error.to_numpy(dtype=float) > 1e-12
    assert positive.any()
    assert state.df == 0 and table.std_error.notna().all()
    assert table.loc[positive, ["statistic", "p_value", "ci_low", "ci_high"]].isna().all().all()


@pytest.mark.parametrize("method", METHODS)
def test_complete_census_has_zero_covariance_and_point_intervals(method):
    frame = fixture().loc[lambda x: x.p.eq(1)].copy()
    frame["N"], frame["M"] = 1, 4
    state = run(frame, declare(frame), method)
    table = state.to_frame()
    np.testing.assert_array_equal(state.covariance, np.zeros((len(state.labels), len(state.labels))))
    np.testing.assert_array_equal(table.ci_low, table.estimate)
    np.testing.assert_array_equal(table.ci_high, table.estimate)
    assert table[["statistic", "p_value"]].isna().all().all()


def test_ratio_zero_denominator_and_unmatched_joint_columns_fail():
    frame = fixture()
    design = declare(frame)
    for outcomes, den in [(["y"], ["zero"]), (["y", "den"], ["den"]), (["y", "y"], ["den", "den"])]:
        with pytest.raises(AnalysisError):
            oe.survey_two_stage_ratio(frame.assign(zero=0), design, outcomes, den)


@pytest.mark.parametrize("categories", [[], ["a", "a"], [None], [float("nan")], [["a"]]])
def test_category_targets_require_distinct_nonmissing_scalar_categories(categories):
    frame = fixture()
    with pytest.raises(AnalysisError):
        oe.survey_two_stage_proportion(frame, declare(frame), "category", categories=categories)


@pytest.mark.parametrize("family", REGRESSION)
def test_regression_duplicate_roles_rank_loss_and_nonfinite_regressors_fail(family):
    frame = fixture()
    design = declare(frame)
    outcome = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[family]
    fn = getattr(oe, "survey_two_stage_" + family)
    for regressors in [["x", "x"], ["absent"], ["constant", "x"], ["x", "duplicate"], [outcome]]:
        with pytest.raises(AnalysisError):
            fn(frame.assign(constant=1., duplicate=frame.x * 2), design, outcome, regressors)
    bad = frame.copy()
    bad.iat[0, bad.columns.get_loc("x")] = np.inf
    with pytest.raises(AnalysisError):
        fn(bad, design, outcome, ["x", "z"])


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_binary_response_and_separation_contract_is_retained(family):
    frame = fixture()
    design = declare(frame)
    fn = getattr(oe, "survey_two_stage_" + family)
    for values in [0, 2, (frame.x > 0).astype(int)]:
        with pytest.raises(AnalysisError):
            fn(frame.assign(binary=values), design, "binary", ["x", "z"])


@pytest.mark.parametrize("value", [-1, .5, float("inf"), 2**53 + 1, True, complex(1, 1)])
def test_poisson_invalid_original_scalar_is_rejected_before_float_conversion(value):
    frame = fixture().astype({"count": "object"})
    design = declare(frame)
    frame.iat[0, frame.columns.get_loc("count")] = value
    before = frame.copy(deep=True)
    with pytest.raises(AnalysisError):
        oe.survey_two_stage_poisson(frame, design, "count", ["x", "z"])
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("method", METHODS)
def test_json_roundtrip_and_unrehashed_corruption_are_not_trusted(method, complete_states):
    state = complete_states[method]
    restored = type(state).model_validate_json(state.model_dump_json())
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")
    pd.testing.assert_frame_equal(restored.to_frame(), state.to_frame())
    payload = state.model_dump(mode="json")
    payload["metadata"]["n_design"] += 1
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(payload)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("defect", ["df", "different-covariance", "asymmetric-covariance", "negative-variance",
                                    "missing-position", "duplicate-position", "boolean-position", "outside-position"])
def test_rehashed_corruption_still_requires_complete_geometry_and_covariance(method, defect, complete_states):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    if defect == "df":
        payload["df"] += 1
    elif defect == "different-covariance":
        payload["covariance"][0][0] += .2
    elif defect == "asymmetric-covariance":
        payload["covariance"][0][1] += .2
    elif defect == "negative-variance":
        payload["covariance"][0][0] = -1.
    elif defect == "missing-position":
        payload["metadata"]["sample_positions"].pop()
    elif defect == "duplicate-position":
        payload["metadata"]["sample_positions"][1] = 0
    elif defect == "boolean-position":
        payload["metadata"]["sample_positions"][0] = False
    else:
        payload["metadata"]["sample_positions"][-1] = 24
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("defect", ["primitive-value", "primitive-denominator", "weighted-influence",
                                    "stage1-covariance", "stage2-covariance", "short-primitive-row",
                                    "missing-primitive-row", "boolean-primitive"])
def test_rehashed_descriptive_primitives_replay_estimates_and_both_stages(method, defect, complete_states):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    m = payload["metadata"]
    if defect == "primitive-value":
        m["primitive_values"][0][0] += .25
    elif defect == "primitive-denominator":
        m["primitive_denominators"][0][0] += .25
    elif defect == "weighted-influence":
        m["row_influences"][0][0] += .25
    elif defect in ("stage1-covariance", "stage2-covariance"):
        m[defect.replace("-", "_")][0][0] += .25
    elif defect == "short-primitive-row":
        m["primitive_values"][0].pop()
    elif defect == "missing-primitive-row":
        m["primitive_values"].pop()
    else:
        m["primitive_values"][0][0] = True
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(refresh_target_fingerprint(payload))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
def test_excluded_rows_cannot_become_nonzero_primitives_after_rehashing(method):
    frame = fixture()
    frame.iloc[20:, frame.columns.get_loc("domain")] = 0
    state = run(frame, declare(frame), method, domain="domain")
    payload = state.model_dump(mode="json")
    payload["metadata"]["primitive_values"][20][0] = 1.
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(refresh_target_fingerprint(payload))


@pytest.mark.parametrize("identity", [["unknown", "a"], ["bool", 1], ["int", True],
                                      ["str", ["a"]], ["float", "not-hex"]])
def test_saved_category_identity_encoding_cannot_be_invented_after_rehashing(identity, complete_states):
    state = complete_states["proportion"]
    payload = state.model_dump(mode="json")
    payload["metadata"]["categories"][0] = identity
    payload["labels"][0] = f"category[0]={identity[0]}:{identity[1]}"
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", REGRESSION)
@pytest.mark.parametrize("defect", ["primitive-X", "primitive-y", "primitive-intercept", "boolean-X",
                                    "coefficients", "bread", "row-scores", "stage1-covariance", "stage2-covariance"])
def test_rehashed_regression_primitives_replay_fit_scores_and_both_stages(method, defect, complete_states):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    m = payload["metadata"]
    if defect == "primitive-X":
        m["primitive_X"][0][-1] += .25
    elif defect == "primitive-y":
        m["primitive_y"][0] += .25
    elif defect == "primitive-intercept":
        m["primitive_X"][0][0] = 2.
    elif defect == "boolean-X":
        m["primitive_X"][0][-1] = True
    elif defect == "coefficients":
        payload["coefficients"][0] += .25
    elif defect == "bread":
        m["bread"][0][0] += .25
    elif defect == "row-scores":
        m["row_scores"][0][0] += .25
    else:
        m[defect.replace("-", "_")][0][0] += .25
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(refresh_regression_fingerprint(payload))


@pytest.mark.parametrize("method", METHODS)
def test_cpu_float64_contract_ignores_ambient_default_device_and_dtype(method, complete_states):
    frame = fixture()
    design = declare(frame)
    expected = complete_states[method]
    previous_device, previous_dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode():
            actual = run(frame, design, method)
            restored = type(actual).model_validate_json(actual.model_dump_json())
            pd.testing.assert_frame_equal(restored.to_frame(), expected.to_frame())
        assert torch.get_default_device().type == "meta"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("method", METHODS)
def test_small_ambient_workspace_budget_refuses_complete_execution(method):
    frame = pd.concat([fixture()] * 200, ignore_index=True)
    frame["s"] = frame.groupby(["h", "p"], sort=False).cumcount() + 1
    frame["M"] *= 200
    design = declare(frame)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        run(frame, design, method)
    assert error.value.code == "workspace_limit"
