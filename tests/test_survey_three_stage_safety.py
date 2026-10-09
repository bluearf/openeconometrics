"""Three-stage admission, complete geometry and adversarial saved-state guards.

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
    """Three levels repeat labels across parents; all GLMs have finite mixed responses."""
    rows = []
    for h in range(3):
        for p in range(2):
            for q in range(2):
                for t in range(8):
                    x = (-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75)[t]
                    z = (-0.7, 0.2)[q] + 0.15 * h + 0.1 * p + 0.035 * (t % 3)
                    rows.append(
                        {
                            "h": ("north", "central", "south")[h],
                            "p": p + 1,
                            "s": q + 1,
                            "t": t + 1,
                            "N": (2, 5, 6)[h],
                            "M": 2 if h == p == 0 else 3 + h + p,
                            "L": 8 if h == p == q == 0 else 12 + h + q + 2 * p,
                            "x": x,
                            "z": z,
                            "y": 1.5 + 0.6 * x - 0.3 * z + (t * p + h + q) / 7 + t**2 / 80,
                            "den": 2.5 + 0.2 * x + 0.1 * z,
                            "binary": (t + p + h + q) % 2,
                            "count": (0, 2, 1, 3)[(t + p + h + q) % 4],
                            "category": ("a", "b")[(t + p + h + q) % 2],
                            "domain": 1,
                        }
                    )
    return pd.DataFrame(rows, index=[9, -3, 9, 0] * 24)


def declare(frame=None, **options):
    return oe.survey_three_stage_design(
        fixture() if frame is None else frame,
        psu="p",
        ssu="s",
        tsu="t",
        strata="h",
        population_psu="N",
        population_ssu="M",
        population_tsu="L",
        **options,
    )


def run(frame, design, method, **options):
    fn = getattr(oe, "survey_three_stage_" + method)
    if method == "ratio":
        return fn(frame, design, ["y", "den"], ["den", "y"], **options)
    if method == "proportion":
        return fn(frame, design, "category", categories=["a", "b", "absent"], **options)
    if method in REGRESSION:
        outcome = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[
            method
        ]
        return fn(frame, design, outcome, ["x", "z"], **options)
    return fn(frame, design, ["y", "den"], **options)


def rehash(payload):
    """A digest does not authenticate a result: deliberately recompute it."""
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    payload["integrity_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return payload


def refresh_target_fingerprint(payload):
    metadata = payload["metadata"]
    selected = [i in metadata["sample_positions"] for i in range(metadata["n_design"])]
    primitive = [
        payload["design"]["validation"]["design_input_sha256"],
        selected,
        metadata["primitive_values"],
        metadata["primitive_denominators"],
    ]
    metadata["sample_input_sha256"] = hashlib.sha256(
        json.dumps(
            primitive,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return rehash(payload)


def refresh_regression_fingerprint(payload):
    """Recompute the shared admitted numeric sample fingerprint after mutation."""
    metadata = payload["metadata"]
    n = metadata["n_design"]
    width = len(payload["regressors"]) + 1
    values = [[0.0] * width for _ in range(n)]
    for i, physical in enumerate(metadata["sample_positions"]):
        values[physical] = [
            metadata["primitive_y"][i],
            *metadata["primitive_X"][i][int(payload["intercept"]) :],
        ]
    selected = [i in metadata["sample_positions"] for i in range(n)]
    primitive = [
        payload["design"]["validation"]["design_input_sha256"],
        selected,
        values,
        [[1.0] * width for _ in range(n)],
    ]
    metadata["sample_input_sha256"] = hashlib.sha256(
        json.dumps(
            primitive,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
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
    refreshed = (
        refresh_regression_fingerprint(payload)
        if method in REGRESSION
        else refresh_target_fingerprint(payload)
    )
    assert refreshed["metadata"]["sample_input_sha256"] == state.metadata["sample_input_sha256"]
    assert refreshed["integrity_sha256"] == state.integrity_sha256
    assert type(state).model_validate(refreshed).model_dump(mode="json") == state.model_dump(
        mode="json"
    )


def test_complete_nested_geometry_and_derived_probabilities_do_not_mutate_source():
    frame = fixture()
    before = frame.copy(deep=True)
    design = declare(frame)
    v = design.validation
    assert (v.nobs, v.n_strata, v.n_psu, v.n_ssu, v.design_df) == (96, 3, 6, 12, 3)
    assert [(h.n_psu, h.population_psu) for h in v.strata] == [(2, 2), (2, 5), (2, 6)]
    assert [p.n_ssu for p in v.psus] == [2] * 6
    assert [q.n_tsu for q in v.ssus] == [8] * 12
    assert [p.stratum_index for p in v.psus] == [0, 0, 1, 1, 2, 2]
    assert [q.psu_index for q in v.ssus] == [i for i in range(6) for _ in range(2)]
    assert [p.row_positions for p in v.psus] == [tuple(range(i, i + 16)) for i in range(0, 96, 16)]
    assert [q.row_positions for q in v.ssus] == [tuple(range(i, i + 8)) for i in range(0, 96, 8)]
    weights = [r.N / 2 * r.M / 2 * r.L / 8 for r in frame.itertuples()]
    np.testing.assert_array_equal(design.validation.weights, weights)
    assert v.sum_weights == sum(weights)
    assert design.revalidate(frame) == v
    pd.testing.assert_frame_equal(frame, before)


def test_typed_psu_ssu_and_tsu_labels_remain_distinct_inside_their_parents():
    rows = []
    for p in (1, "1", True, 1.0):
        for q in (1, "1"):
            for t in (1, "1"):
                rows.append({"h": "a", "p": p, "s": q, "t": t, "N": 8, "M": 4, "L": 4})
    d = declare(pd.DataFrame(rows, dtype=object))
    assert (d.validation.n_psu, d.validation.n_ssu, d.validation.nobs) == (4, 8, 16)
    assert [p.n_ssu for p in d.validation.psus] == [2] * 4
    assert [q.n_tsu for q in d.validation.ssus] == [2] * 8
    assert d.validation.design_df == 3


def test_typed_strata_and_repeated_labels_across_all_parents_are_valid():
    rows = [
        dict(h=h, p=p, s=q, t=t, N=4, M=4, L=4)
        for h in (1, "1", True, 1.0)
        for p in (1, 2)
        for q in (1, 2)
        for t in (1, 2)
    ]
    d = declare(pd.DataFrame(rows, dtype=object))
    assert (d.validation.n_strata, d.validation.n_psu, d.validation.n_ssu, d.validation.nobs) == (
        4,
        8,
        16,
        32,
    )


def test_duplicate_tsu_inside_one_ssu_is_rejected_even_with_identical_outcomes():
    frame = fixture()
    frame.iloc[1, frame.columns.get_loc("t")] = frame.iloc[0].t
    with pytest.raises(AnalysisError) as error:
        declare(frame)
    assert error.value.code == "invalid_survey_nesting"


def test_repeated_ssu_across_rows_is_admitted_but_collapsing_it_changes_population_constraints():
    frame = fixture()
    assert declare(frame).validation.n_ssu == 12
    frame["s"] = 1
    # TSU labels now collide within the merged SSU: dropping a stage cannot be implicit.
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column", ["N", "M", "L"])
@pytest.mark.parametrize(
    "value", [0, -1, 0.5, 1.5, float("nan"), float("inf"), True, "8", None, 1j, 2**53 + 1, 10**500]
)
def test_stage_population_counts_are_exact_positive_bounded_scalars(column, value):
    frame = fixture().astype({column: "object"})
    selection = frame.h.eq("north")
    if column in ("M", "L"):
        selection &= frame.p.eq(1)
    if column == "L":
        selection &= frame.s.eq(1)
    for position in np.flatnonzero(selection.to_numpy()):
        frame.iat[position, frame.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column", ["N", "M", "L"])
def test_population_count_must_be_constant_within_its_stage_group(column):
    frame = fixture()
    frame.iat[0, frame.columns.get_loc(column)] += 1
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column,value", [("N", 1), ("M", 1), ("L", 7)])
def test_population_count_cannot_be_less_than_complete_sample_geometry(column, value):
    frame = fixture().assign(**{column: value})
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("column", ["h", "p", "s", "t"])
@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), "", "x" * 257, [1, 2], 10**500]
)
def test_invalid_stage_id_is_never_silently_removed(column, value):
    frame = fixture().astype({column: "object"})
    frame.iat[0, frame.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError):
        declare(frame)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_each_noncensus_singleton_refuses_and_each_census_singleton_is_admitted(stage):
    frame = fixture()
    id_column, population_column = {1: ("p", "N"), 2: ("s", "M"), 3: ("t", "L")}[stage]
    frame = frame.loc[frame[id_column].eq(1)].copy()
    frame[population_column] = 2
    with pytest.raises(AnalysisError):
        declare(frame)
    frame[population_column] = 1
    design = declare(frame)
    assert design.validation.nobs == len(frame)
    assert design.revalidate(frame) == design.validation


def test_declaration_json_restoration_preserves_exact_design_and_requires_revalidation():
    frame = fixture()
    design = declare(frame)
    restored = type(design).model_validate_json(design.model_dump_json())
    assert restored == design and restored.revalidate(frame) == design.validation
    assert (
        restored.revalidate(frame.assign(y=100).set_axis(["duplicate"] * len(frame)))
        == design.validation
    )
    for changed in [frame.iloc[::-1], frame.astype({"p": "object"}), frame.assign(M=frame.M + 1)]:
        with pytest.raises(AnalysisError):
            restored.revalidate(changed)
    categorical = frame.astype({"h": "category"})
    d = declare(categorical)
    categorical["h"] = categorical.h.cat.add_categories(["unused"])
    with pytest.raises(AnalysisError):
        d.revalidate(categorical)


@pytest.mark.parametrize(
    "patch",
    [
        {"stages": 1},
        {"stages": 2},
        {"stages": 4},
        {"stages": True},
        {"stages": "3"},
        {"schema_version": "future"},
        {"psu": "s"},
        {"population_psu": "M"},
        {"tsu": "s"},
        {"population_tsu": "N"},
        {"weights": "w"},
        {"max_rows": True},
        {"max_memory_mb": "64"},
    ],
)
def test_saved_declaration_rejects_unsupported_or_coerced_schema(patch):
    original = declare()
    payload = original.model_dump(mode="json")
    payload.update(patch)
    with pytest.raises(ValidationError):
        type(original).model_validate(payload)


@pytest.mark.parametrize(
    "defect",
    [
        "wrong-nobs",
        "wrong-psu-count",
        "wrong-ssu-count",
        "wrong-stratum",
        "wrong-psu-parent",
        "duplicate-psu",
        "missing-psu",
        "boolean-psu",
        "duplicate-ssu",
        "missing-ssu",
        "boolean-ssu",
        "duplicate-row",
        "missing-row",
        "boolean-row",
        "outside-row",
        "parent-row-mismatch",
        "wrong-weight-sum",
    ],
)
def test_saved_geometry_checks_every_parent_partition_before_revalidation(defect):
    frame = fixture()
    original = declare(frame)
    payload = original.model_dump(mode="json")
    v = payload["validation"]
    if defect == "wrong-nobs":
        v["nobs"] += 1
    elif defect == "wrong-psu-count":
        v["n_psu"] += 1
    elif defect == "wrong-ssu-count":
        v["n_ssu"] += 1
    elif defect == "wrong-stratum":
        v["psus"][0]["stratum_index"] = 1
    elif defect == "wrong-psu-parent":
        v["ssus"][0]["psu_index"] = 1
    elif defect.endswith("psu"):
        indices = v["strata"][0]["psu_indices"]
        if defect.startswith("duplicate"):
            indices[1] = indices[0]
        elif defect.startswith("missing"):
            indices.pop()
        else:
            indices[0] = False
    elif defect.endswith("ssu"):
        indices = v["psus"][0]["ssu_indices"]
        if defect.startswith("duplicate"):
            indices[1] = indices[0]
        elif defect.startswith("missing"):
            indices.pop()
        else:
            indices[0] = False
    elif defect == "parent-row-mismatch":
        v["psus"][0]["row_positions"].pop()
    elif defect == "wrong-weight-sum":
        v["sum_weights"] += 1
    else:
        rows = v["ssus"][0]["row_positions"]
        if defect == "duplicate-row":
            rows[1] = rows[0]
        elif defect == "missing-row":
            rows.pop()
        elif defect == "boolean-row":
            rows[0] = False
        else:
            rows[0] = len(frame)
    with pytest.raises((ValueError, AnalysisError)):
        saved = type(original).model_validate(payload)
        saved.revalidate(frame)


@pytest.mark.parametrize("source_kind", ["frame", "columns", "rows"])
def test_complete_row_budget_is_enforced_before_estimation(source_kind):
    frame = fixture()
    source = {"frame": frame, "columns": frame.to_dict("list"), "rows": frame.to_dict("records")}[
        source_kind
    ]
    with pytest.raises(AnalysisError) as error:
        declare(source, max_rows=3)
    assert error.value.code == "survey_budget"


@pytest.mark.parametrize(
    "options",
    [
        {"max_rows": True},
        {"max_rows": 0},
        {"max_rows": 1_000_001},
        {"max_memory_mb": "64"},
        {"max_memory_mb": 0},
        {"max_memory_mb": 513},
    ],
)
def test_declaration_scalar_resource_limits_fail_closed(options):
    with pytest.raises(AnalysisError):
        declare(**options)


def test_oversized_column_sources_are_refused_before_materializing_rows():
    class ForbiddenRows:
        def __len__(self):
            return 1_000_001

        def __iter__(self):
            raise AssertionError("Oversized source reached column materialization.")

    source = {name: ForbiddenRows() for name in ("p", "s", "t", "h", "N", "M", "L")}
    with pytest.raises(AnalysisError) as error:
        declare(source)
    assert error.value.code == "survey_budget"


def test_duplicate_missing_empty_and_unsupported_design_inputs():
    frame = fixture()
    for source in [
        pd.concat([frame, frame[["s"]]], axis=1),
        frame.drop(columns="M"),
        frame.iloc[:0],
    ]:
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


def test_three_stage_declaration_stays_lazy_without_starting_tensor_runtime():
    command = (
        "import sys,openecon as oe; assert oe.SurveyThreeStageDesign; "
        "assert oe.survey_three_stage_design; assert 'torch' not in sys.modules; "
        "assert 'openecon.analysis' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "role,value",
    [
        ("psu", ""),
        ("ssu", True),
        ("tsu", []),
        ("strata", []),
        ("population_psu", "p"),
        ("population_ssu", "absent"),
        ("population_tsu", "L" * 201),
    ],
)
def test_invalid_design_roles_fail_explicitly(role, value):
    options = {
        "psu": "p",
        "ssu": "s",
        "tsu": "t",
        "strata": "h",
        "population_psu": "N",
        "population_ssu": "M",
        "population_tsu": "L",
    }
    options[role] = value
    with pytest.raises(AnalysisError):
        oe.survey_three_stage_design(fixture(), **options)


@pytest.mark.parametrize("method", METHODS)
def test_domain_and_missing_rows_retain_complete_stage_geometry(method):
    frame = fixture()
    design = declare(frame)
    frame.loc[frame.h.eq("south") & frame.p.eq(2), "domain"] = 0
    outcomes = ["y", "den", "binary", "count", "category", "x", "z"]
    outside = frame.domain.eq(0)
    frame.loc[outside, outcomes] = np.nan
    before = frame.copy(deep=True)
    state = run(frame, design, method, domain="domain")
    assert state.design == design and state.df == 3
    assert state.metadata["n_design"] == 96
    assert state.metadata["sample_positions"] == list(range(80))
    assert state.design.validation.psus[-1].row_positions == tuple(range(80, 96))
    pd.testing.assert_frame_equal(frame, before)
    frame.iat[
        0,
        frame.columns.get_loc(
            "y"
            if method in ("mean", "total", "ratio", "regress")
            else "category"
            if method == "proportion"
            else "count"
            if method == "poisson"
            else "binary"
        ),
    ] = np.nan
    with pytest.raises(AnalysisError):
        run(frame, design, method, domain="domain")
    before_drop = frame.copy(deep=True)
    dropped = run(frame, design, method, domain="domain", missing="drop")
    assert dropped.metadata["sample_positions"] == list(range(1, 80))
    assert dropped.design.validation == design.validation and dropped.df == 3
    assert dropped.metadata["outcome_exclusions"] == [0]
    pd.testing.assert_frame_equal(frame, before_drop)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("domain", [0, 0.5, float("nan"), "yes"])
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
def test_one_stage_declaration_cannot_be_used_as_a_three_stage_design(method):
    frame = fixture().assign(w=1.0)
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
    frame["N"], frame["M"], frame["L"] = 1, 2, 8
    state = run(frame, declare(frame), method)
    table = state.to_frame()
    np.testing.assert_array_equal(
        state.covariance, np.zeros((len(state.labels), len(state.labels)))
    )
    np.testing.assert_array_equal(table.ci_low, table.estimate)
    np.testing.assert_array_equal(table.ci_high, table.estimate)
    assert table[["statistic", "p_value"]].isna().all().all()


def test_ratio_zero_denominator_and_unmatched_joint_columns_fail():
    frame = fixture()
    design = declare(frame)
    for outcomes, den in [(["y"], ["zero"]), (["y", "den"], ["den"]), (["y", "y"], ["den", "den"])]:
        with pytest.raises(AnalysisError):
            oe.survey_three_stage_ratio(frame.assign(zero=0), design, outcomes, den)


@pytest.mark.parametrize("categories", [[], ["a", "a"], [None], [float("nan")], [["a"]]])
def test_category_targets_require_distinct_nonmissing_scalar_categories(categories):
    frame = fixture()
    with pytest.raises(AnalysisError):
        oe.survey_three_stage_proportion(frame, declare(frame), "category", categories=categories)


@pytest.mark.parametrize("family", REGRESSION)
def test_regression_duplicate_roles_rank_loss_and_nonfinite_regressors_fail(family):
    frame = fixture()
    design = declare(frame)
    outcome = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[family]
    fn = getattr(oe, "survey_three_stage_" + family)
    for regressors in [["x", "x"], ["absent"], ["constant", "x"], ["x", "duplicate"], [outcome]]:
        with pytest.raises(AnalysisError):
            fn(frame.assign(constant=1.0, duplicate=frame.x * 2), design, outcome, regressors)
    bad = frame.copy()
    bad.iat[0, bad.columns.get_loc("x")] = np.inf
    with pytest.raises(AnalysisError):
        fn(bad, design, outcome, ["x", "z"])


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_binary_response_and_separation_contract_is_retained(family):
    frame = fixture()
    design = declare(frame)
    fn = getattr(oe, "survey_three_stage_" + family)
    for values in [0, 2, (frame.x > 0).astype(int)]:
        with pytest.raises(AnalysisError):
            fn(frame.assign(binary=values), design, "binary", ["x", "z"])


@pytest.mark.parametrize("value", [-1, 0.5, float("inf"), 2**53 + 1, True, complex(1, 1)])
def test_poisson_invalid_original_scalar_is_rejected_before_float_conversion(value):
    frame = fixture().astype({"count": "object"})
    design = declare(frame)
    frame.iat[0, frame.columns.get_loc("count")] = value
    before = frame.copy(deep=True)
    with pytest.raises(AnalysisError):
        oe.survey_three_stage_poisson(frame, design, "count", ["x", "z"])
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
@pytest.mark.parametrize(
    "defect",
    [
        "df",
        "different-covariance",
        "asymmetric-covariance",
        "negative-variance",
        "missing-position",
        "duplicate-position",
        "boolean-position",
        "outside-position",
    ],
)
def test_rehashed_corruption_still_requires_complete_geometry_and_covariance(
    method, defect, complete_states
):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    if defect == "df":
        payload["df"] += 1
    elif defect == "different-covariance":
        payload["covariance"][0][0] += 0.2
    elif defect == "asymmetric-covariance":
        payload["covariance"][0][1] += 0.2
    elif defect == "negative-variance":
        payload["covariance"][0][0] = -1.0
    elif defect == "missing-position":
        payload["metadata"]["sample_positions"].pop()
    elif defect == "duplicate-position":
        payload["metadata"]["sample_positions"][1] = 0
    elif defect == "boolean-position":
        payload["metadata"]["sample_positions"][0] = False
    else:
        payload["metadata"]["sample_positions"][-1] = 96
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize(
    "defect",
    [
        "primitive-value",
        "primitive-denominator",
        "weighted-influence",
        "stage1-covariance",
        "stage2-covariance",
        "stage3-covariance",
        "short-primitive-row",
        "missing-primitive-row",
        "boolean-primitive",
    ],
)
def test_rehashed_descriptive_primitives_replay_estimates_and_all_three_stages(
    method, defect, complete_states
):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    m = payload["metadata"]
    if defect == "primitive-value":
        m["primitive_values"][0][0] += 0.25
    elif defect == "primitive-denominator":
        m["primitive_denominators"][0][0] += 0.25
    elif defect == "weighted-influence":
        m["row_influences"][0][0] += 0.25
    elif defect in ("stage1-covariance", "stage2-covariance", "stage3-covariance"):
        m[defect.replace("-", "_")][0][0] += 0.25
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
    frame.iloc[80:, frame.columns.get_loc("domain")] = 0
    state = run(frame, declare(frame), method, domain="domain")
    payload = state.model_dump(mode="json")
    payload["metadata"]["primitive_values"][80][0] = 1.0
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(refresh_target_fingerprint(payload))


@pytest.mark.parametrize(
    "identity", [["unknown", "a"], ["bool", 1], ["int", True], ["str", ["a"]], ["float", "not-hex"]]
)
def test_saved_category_identity_encoding_cannot_be_invented_after_rehashing(
    identity, complete_states
):
    state = complete_states["proportion"]
    payload = state.model_dump(mode="json")
    payload["metadata"]["categories"][0] = identity
    payload["labels"][0] = f"category[0]={identity[0]}:{identity[1]}"
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", REGRESSION)
@pytest.mark.parametrize(
    "defect",
    [
        "primitive-X",
        "primitive-y",
        "primitive-intercept",
        "boolean-X",
        "coefficients",
        "bread",
        "row-scores",
        "stage1-covariance",
        "stage2-covariance",
        "stage3-covariance",
    ],
)
def test_rehashed_regression_primitives_replay_fit_scores_and_all_three_stages(
    method, defect, complete_states
):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    m = payload["metadata"]
    if defect == "primitive-X":
        m["primitive_X"][0][-1] += 0.25
    elif defect == "primitive-y":
        m["primitive_y"][0] += 0.25
    elif defect == "primitive-intercept":
        m["primitive_X"][0][0] = 2.0
    elif defect == "boolean-X":
        m["primitive_X"][0][-1] = True
    elif defect == "coefficients":
        payload["coefficients"][0] += 0.25
    elif defect == "bread":
        m["bread"][0][0] += 0.25
    elif defect == "row-scores":
        m["row_scores"][0][0] += 0.25
    else:
        m[defect.replace("-", "_")][0][0] += 0.25
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
    frame = pd.concat([fixture()] * 40, ignore_index=True)
    frame["t"] = frame.groupby(["h", "p", "s"], sort=False).cumcount() + 1
    frame["L"] *= 40
    design = declare(frame)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        run(frame, design, method)
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("method", METHODS)
def test_each_stage_census_removes_only_its_own_uncertainty_term(stage, method):
    frame = fixture()
    if method == "proportion":
        h_code = frame.h.map({"north": 0, "central": 1, "south": 2})
        frame["category"] = np.where(frame.t <= 1 + h_code + frame.p + frame.s, "a", "b")
    column, census = {1: ("N", 2), 2: ("M", 2), 3: ("L", 8)}[stage]
    frame[column] = census
    state = run(frame, declare(frame), method)
    matrices = [np.array(state.metadata[f"stage{i}_covariance"]) for i in (1, 2, 3)]
    np.testing.assert_array_equal(matrices[stage - 1], np.zeros_like(matrices[stage - 1]))
    assert all(np.any(matrices[i] > 0) for i in range(3) if i != stage - 1)
    np.testing.assert_allclose(state.covariance, sum(matrices), rtol=1e-13, atol=0)


@pytest.mark.parametrize("method", METHODS)
def test_two_stage_declaration_is_not_implicitly_promoted(method):
    frame = fixture()
    terminal_frame = frame.assign(two_s=range(len(frame)), two_M=40)
    design = oe.survey_two_stage_design(
        terminal_frame,
        psu="p",
        ssu="two_s",
        strata="h",
        population_psu="N",
        population_ssu="two_M",
    )
    with pytest.raises(AnalysisError) as error:
        run(terminal_frame, design, method)
    assert error.value.code == "invalid_survey_design"


def test_oversized_row_sequence_is_refused_before_reading_a_row():
    from collections.abc import Sequence

    class ForbiddenSequence(Sequence):
        def __len__(self):
            return 1_000_001

        def __getitem__(self, position):
            raise AssertionError("Row iteration preceded complete-dimension admission.")

    with pytest.raises(AnalysisError) as error:
        declare(ForbiddenSequence())
    assert error.value.code == "survey_budget"


@pytest.mark.parametrize(
    "patch",
    [
        {"nobs": True},
        {"nobs": 1_000_001},
        {"nobs": 100_000},
        {"ssus": []},
        {"psus": []},
        {"strata": []},
    ],
)
def test_saved_declaration_rejects_invalid_dimensions_before_weight_expansion(monkeypatch, patch):
    design = declare()
    payload = design.model_dump(mode="json")
    payload["validation"].update(patch)

    def forbidden_weights(_):
        raise AssertionError("Invalid saved dimensions expanded derived row weights.")

    monkeypatch.setattr(type(design.validation), "weights", property(forbidden_weights))
    with pytest.raises((ValueError, AnalysisError)):
        type(design).model_validate(payload)


@pytest.mark.parametrize("field", ["psu_indices", "ssu_indices", "row_positions"])
def test_saved_nested_index_storage_is_bounded_before_weight_expansion(monkeypatch, field):
    design = declare()
    payload = design.model_dump(mode="json")
    parent = {"psu_indices": "strata", "ssu_indices": "psus", "row_positions": "ssus"}[field]
    payload["validation"][parent][0][field] = [0] * (len(fixture()) + 1)

    def forbidden_weights(_):
        raise AssertionError("Oversized nested records expanded derived row weights.")

    monkeypatch.setattr(type(design.validation), "weights", property(forbidden_weights))
    with pytest.raises((ValueError, AnalysisError)):
        type(design).model_validate(payload)


@pytest.mark.parametrize("method", METHODS)
def test_full_covariance_and_saved_target_order_are_inseparable(method, complete_states):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    payload["labels"] = list(reversed(payload["labels"]))
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))
    payload = state.model_dump(mode="json")
    cov = np.array(payload["covariance"])
    payload["covariance"] = cov[::-1, ::-1].tolist()
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", METHODS)
def test_rehashed_stage_reallocation_cannot_keep_the_same_total_covariance(method, complete_states):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    payload["metadata"]["stage1_covariance"][0][0] += 0.001
    payload["metadata"]["stage2_covariance"][0][0] -= 0.001
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
def test_unsafe_descriptive_copy_or_nested_metadata_cannot_skip_replay(method, complete_states):
    state = complete_states[method]
    unsafe = state.model_copy(update={"df": state.df + 1})
    with pytest.raises((ValueError, AnalysisError)):
        unsafe.to_frame()
    unsafe = state.model_copy(deep=True)
    unsafe.metadata["stage3_covariance"][0][0] += 0.1
    with pytest.raises((ValueError, AnalysisError)):
        unsafe.contrast([1.0] * len(unsafe.labels))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("bad_name", ["", " ", "x" * 201, True, None])
def test_descriptive_saved_roles_require_the_original_admissible_name_contract(
    method, bad_name, complete_states
):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    payload["metadata"]["outcomes"][0] = bad_name
    if method == "proportion":
        categories = payload["metadata"]["categories"]
        payload["labels"] = [f"{bad_name}[{i}]={c[0]}:{c[1]}" for i, c in enumerate(categories)]
    elif method == "ratio":
        payload["labels"][0] = f"{bad_name}/{payload['metadata']['denominators'][0]}"
    else:
        payload["labels"][0] = bad_name
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize(
    "field", ["sample_positions", "outcome_exclusions", "out_of_domain_positions"]
)
def test_saved_physical_partitions_cannot_overlap_or_omit_complete_parents(
    method, field, complete_states
):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    if field == "sample_positions":
        payload["metadata"][field].pop()
    else:
        payload["metadata"][field] = [0]
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", METHODS)
def test_zero_domain_stratum_psu_and_ssu_remain_in_complete_recursive_geometry(method):
    frame = fixture()
    design = declare(frame)
    excluded = (
        frame.h.eq("south")
        | (frame.h.eq("central") & frame.p.eq(1))
        | (frame.h.eq("north") & frame.p.eq(1) & frame.s.eq(1))
    )
    frame.loc[excluded, "domain"] = 0
    for column in ("y", "den", "binary", "count", "category", "x", "z"):
        frame[column] = frame[column].astype(object)
        frame.loc[excluded, column] = "invalid outside declared domain"
    before = frame.copy(deep=True)
    state = run(frame, design, method, domain="domain")
    expected = np.flatnonzero(~excluded.to_numpy()).tolist()
    assert state.metadata["sample_positions"] == expected
    assert state.metadata["out_of_domain_positions"] == np.flatnonzero(excluded.to_numpy()).tolist()
    assert state.metadata["n_used"] == 40 and state.df == 3
    assert state.design.validation == design.validation
    assert (
        state.design.validation.n_strata,
        state.design.validation.n_psu,
        state.design.validation.n_ssu,
    ) == (3, 6, 12)
    key = "row_scores" if method in REGRESSION else "row_influences"
    rows = np.array(state.metadata[key])
    np.testing.assert_array_equal(
        rows[excluded.to_numpy()], np.zeros_like(rows[excluded.to_numpy()])
    )
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["N", "M", "L"])
def test_exact_population_upper_boundary_is_preserved_without_float_rounding(column):
    frame = fixture().assign(**{column: 2**53})
    design = declare(frame)
    records, field = {
        "N": (design.validation.strata, "population_psu"),
        "M": (design.validation.psus, "population_ssu"),
        "L": (design.validation.ssus, "population_tsu"),
    }[column]
    assert all(
        type(getattr(record, field)) is int and getattr(record, field) == 2**53
        for record in records
    )
    restored = type(design).model_validate_json(design.model_dump_json())
    assert restored == design and restored.revalidate(frame) == design.validation


@pytest.mark.parametrize("field", ["outcomes", "denominators"])
def test_saved_ratio_roles_cannot_gain_duplicates_after_rehash(field, complete_states):
    state = complete_states["ratio"]
    payload = state.model_dump(mode="json")
    payload["metadata"][field][1] = payload["metadata"][field][0]
    payload["labels"] = [
        f"{y}/{x}"
        for y, x in zip(payload["metadata"]["outcomes"], payload["metadata"]["denominators"])
    ]
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


def test_saved_proportion_cannot_claim_a_ratio_denominator(complete_states):
    state = complete_states["proportion"]
    payload = state.model_dump(mode="json")
    payload["metadata"]["denominators"] = ["phantom denominator"]
    with pytest.raises((ValueError, AnalysisError)):
        type(state).model_validate(rehash(payload))


@pytest.mark.parametrize("method", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("field", ["primitive_values", "sum_design_weights"])
def test_coherently_rehashed_oversized_numeric_state_is_a_typed_rejection(
    method, field, complete_states
):
    state = complete_states[method]
    payload = state.model_dump(mode="json")
    if field == "primitive_values":
        payload["metadata"][field][0][0] = 10**500
        refresh_target_fingerprint(payload)
    else:
        payload["metadata"][field] = 10**500
        rehash(payload)
    with pytest.raises((ValidationError, AnalysisError)):
        type(state).model_validate(payload)
