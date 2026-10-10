"""Public sequential result, semantic replay and portable-display contracts.

Scientific probability/calibration oracles live in test_sequential_oracles.py.
These tests check complete saved results and guards at public entry points.
"""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.stats import sequential as api
from openecon.econometrics.stats import sequential_kernels as kernels
from openecon.resources import use_workspace_budget
from scripts.torch_test_state import preserve_torch_default_device


CONFIGURATIONS = [
    (law, sides, param)
    for law, param in (("ldof", None), ("ldpocock", None), ("hsd", -4.0), ("power", 2.0))
    for sides in (1, 2)
]
TABLES = [
    "summary", "boundaries", "null_stages", "alternative_stages",
    "canonical_covariance", "canonical_means", "numerical_accuracy",
]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def designs():
    with use_workspace_budget(128):
        return {
            (law, sides): oe.sequential_design(
                fractions=[0.25, 0.5, 0.75, 1.0], spending=law,
                param=param, sides=sides, effect=0.3, information=100.0,
            )
            for law, sides, param in CONFIGURATIONS
        }


def complete_bytes(result):
    """Retain every table/value/type/axis/attribute and complete LaTeX."""
    payload = {
        "title": result.title,
        "table_order": list(result),
        "attrs": result.attrs,
        "tables": {name: frame.to_dict(orient="split") for name, frame in result.items()},
        "dtypes": {
            name: [str(dtype) for dtype in frame.dtypes]
            for name, frame in result.items()
        },
        "table_attrs": {name: frame.attrs for name, frame in result.items()},
        "axes": {
            name: {"index_names": list(frame.index.names), "column_names": list(frame.columns.names)}
            for name, frame in result.items()
        },
        "latex": result.to_latex(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def resign(state):
    """An adversary can recompute a checksum; semantics must still be checked."""
    state = copy.deepcopy(state)
    state.pop("checksum", None)
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    state["checksum"] = hashlib.sha256(encoded.encode()).hexdigest()
    return state


def forbid_search(*args, **kwargs):
    raise AssertionError("Saved restoration must not calibrate boundaries or invert information")


@pytest.mark.parametrize("law,sides,param", CONFIGURATIONS)
def test_all_eight_full_portable_restore_without_search(designs, monkeypatch, law, sides, param):
    result = designs[law, sides]
    assert isinstance(result, TableSet)
    assert list(result) == TABLES
    assert result.attrs["state"]["schema"] == "openecon.sequential.design.v1"
    assert set(result.attrs["state"]) == {
        "schema", "settings", "calibration", "alternative", "inverse", "checksum",
    }
    expected = complete_bytes(result)
    generic = oe.restore_summary(oe.summary_state(result))
    # The general summary codec sorts mapping keys. Semantic reconstruction
    # supplies the canonical order and natural column dtypes again.
    assert set(generic) == set(result)
    monkeypatch.setattr(kernels, "calibrate", forbid_search)
    monkeypatch.setattr(api, "sequential_information", forbid_search)
    with use_workspace_budget(512):
        for saved in (
            copy.deepcopy(result.attrs["state"]),
            json.dumps(result.attrs["state"], sort_keys=True, allow_nan=False),
            copy.deepcopy(result), generic,
        ):
            assert complete_bytes(oe.restore_sequential_design(saved)) == expected
    assert complete_bytes(result) == expected


def test_one_sided_absent_boundary_and_boolean_cells_remain_finite(designs):
    result = designs["ldof", 1]
    assert result["boundaries"].lower.tolist() == [None] * 4
    assert result["boundaries"].lower_present.tolist() == [False] * 4
    assert result["alternative_stages"].lower_first_crossing.tolist() == [0.0] * 4
    assert result["numerical_accuracy"].rigorous_quadrature_error_bound.tolist() == [False]
    complete_bytes(result)  # Strict finite JSON, including all absent lower cells.


def test_fixed_boundary_query_does_not_mutate_original_or_recalibrate(designs, monkeypatch):
    result = designs["hsd", 2]
    before = complete_bytes(result)
    monkeypatch.setattr(kernels, "calibrate", forbid_search)
    changed = oe.sequential_power(result, effect=-0.2, information=150.0)
    state = changed.attrs["state"]
    assert state["calibration"] == result.attrs["state"]["calibration"]
    assert state["settings"]["effect"] == -0.2
    assert state["settings"]["information"] == 150.0
    assert state["inverse"] is None
    assert changed["boundaries"].upper.tolist() == result["boundaries"].upper.tolist()
    assert complete_bytes(result) == before
    assert complete_bytes(oe.restore_sequential_design(state)) == complete_bytes(changed)


@pytest.mark.parametrize("sides,effect", [(1, 0.3), (2, -0.3)])
def test_complete_information_inverse_replays_without_search(designs, monkeypatch, sides, effect):
    result = designs["power", sides]
    before = complete_bytes(result)
    sized = oe.sequential_information(result, effect=effect, power=0.8)
    inverse = sized.attrs["state"]["inverse"]
    assert list(sized) == TABLES + ["information_inversion"]
    assert inverse["power_low"] < 0.8 <= inverse["power_high"]
    assert inverse["power_high"] - 0.8 <= 2e-6
    assert inverse["information"] == sized["summary"].maximum_information.iloc[0]
    assert inverse["information"] != round(inverse["information"])
    assert complete_bytes(result) == before
    expected = complete_bytes(sized)
    generic = oe.restore_summary(oe.summary_state(sized))
    monkeypatch.setattr(kernels, "calibrate", forbid_search)
    monkeypatch.setattr(api, "sequential_information", forbid_search)
    with use_workspace_budget(512):
        for saved in (
            sized, copy.deepcopy(sized.attrs["state"]),
            json.dumps(sized.attrs["state"], sort_keys=True), generic,
        ):
            assert complete_bytes(oe.restore_sequential_design(saved)) == expected
    queried = oe.sequential_power(sized, effect=effect)
    assert queried.attrs["state"]["inverse"] is None
    assert queried["summary"].rejection_probability.iloc[0] == pytest.approx(inverse["power_high"], abs=2e-12)


def altered_result(result, change):
    out = copy.deepcopy(result)
    if change == "value":
        out["summary"].loc[0, "rejection_probability"] += 0.01
    elif change == "bool_value":
        out["boundaries"]["lower_present"] = out["boundaries"].lower_present.astype("int64")
    elif change == "bool_row_label":
        out["summary"].index = pd.Index([False])
    elif change == "columns":
        out["boundaries"] = out["boundaries"].iloc[:, ::-1]
    elif change == "rows":
        out["boundaries"] = out["boundaries"].iloc[::-1]
    elif change == "row_label":
        out["boundaries"] = out["boundaries"].rename(index={"look_1": "look_changed"})
    elif change == "index_name":
        out["boundaries"].index.name = "subject"
    elif change == "column_name":
        out["boundaries"].columns.name = "coefficient"
    elif change == "frame_attrs":
        out["boundaries"].attrs["interpretation"] = "futility"
    elif change == "attrs":
        out.attrs["notes"] = "An altered stopping policy"
    elif change == "title":
        out.title = "Futility design"
    elif change == "missing_table":
        out.pop("canonical_covariance")
    elif change == "extra_table":
        out["extra"] = out["summary"].copy()
    else:
        raise AssertionError(change)
    return out


@pytest.mark.parametrize("change", [
    "value", "bool_value", "bool_row_label", "columns", "rows", "row_label",
    "index_name", "column_name", "frame_attrs", "attrs", "title", "missing_table", "extra_table",
])
@pytest.mark.parametrize("operation", ["restore", "power", "information"])
def test_every_public_consumer_rejects_altered_complete_results(designs, change, operation):
    result = altered_result(designs["hsd", 1], change)
    with pytest.raises(AnalysisError) as error:
        if operation == "restore":
            oe.restore_sequential_design(result)
        elif operation == "power":
            oe.sequential_power(result, effect=0.2)
        else:
            oe.sequential_information(result, effect=0.3)
    assert error.value.code == "invalid_state"


def corrupt_state(state, change):
    state = copy.deepcopy(state)
    if change == "alpha":
        state["settings"]["alpha"] = 0.06
    elif change == "fractions":
        state["settings"]["fractions"][1] = 0.55
    elif change == "effect":
        state["settings"]["effect"] = 0.1
    elif change == "information":
        state["settings"]["information"] = 150.0
    elif change == "boolean_effect":
        state["settings"]["effect"] = False
    elif change == "boundary":
        state["calibration"]["bounds"][0] += 0.1
    elif change == "coarse_boundary":
        state["calibration"]["bounds_coarse"][0] += 0.1
    elif change == "spending":
        state["calibration"]["spending"][0] += 0.001
    elif change == "null_probability":
        state["calibration"]["probabilities"]["power"] += 0.01
    elif change == "alternative_probability":
        state["alternative"]["fine"]["upper"][0] += 0.01
    elif change == "refinement":
        state["alternative"]["refinement_error"] = 1e-4
    elif change == "work_receipt":
        state["calibration"]["numerical_receipt"]["integration_work_bound"] += 1000
    elif change == "false_workspace_provenance":
        state["alternative"]["fine"]["workspace"]["budget_bytes"] = 1
    elif change == "workspace_boolean":
        state["alternative"]["fine"]["workspace"]["budget_bytes"] = True
    elif change == "workspace_buffer":
        state["alternative"]["fine"]["workspace"]["buffers"]["transition_and_density_temporaries"] += 1
    elif change == "extra_setting":
        state["settings"]["futility"] = True
    elif change == "extra_state_field":
        state["claim"] = "integer sample size"
    else:
        raise AssertionError(change)
    return resign(state)


@pytest.mark.parametrize("change", [
    "alpha", "fractions", "effect", "information", "boolean_effect", "boundary",
    "coarse_boundary", "spending", "null_probability", "alternative_probability",
    "refinement", "work_receipt", "false_workspace_provenance", "workspace_boolean",
    "workspace_buffer", "extra_setting", "extra_state_field",
])
def test_recomputed_checksum_cannot_hide_impossible_saved_state(designs, change):
    state = corrupt_state(designs["hsd", 2].attrs["state"], change)
    with pytest.raises(AnalysisError) as error:
        oe.restore_sequential_design(state)
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("change", [
    "float_order", "float_iterations", "fractional_calibration_work",
    "fractional_alternative_integration",
])
def test_integer_protocol_receipts_reject_resigned_float_substitution(designs, change):
    state = copy.deepcopy(designs["hsd", 2].attrs["state"])
    if change == "float_order":
        state["calibration"]["order"] = 128.0
    elif change == "float_iterations":
        state["calibration"]["numerical_receipt"]["bisection_iterations"]["fine"][0] = 60.0
    elif change == "fractional_calibration_work":
        state["calibration"]["numerical_receipt"]["work_bound"] += 0.01
    else:
        state["alternative"]["fine"]["integration_work_bound"] += 0.001
    with pytest.raises(AnalysisError) as error:
        oe.restore_sequential_design(resign(state))
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("entrypoint", ["public", "kernel"])
@pytest.mark.parametrize("field", ["effect", "alpha", "fractions"])
def test_huge_integer_inputs_refuse_before_quadrature(entrypoint, field, monkeypatch):
    huge = 10**1000
    fractions = [0.25, 0.5, 0.75, 1.0]
    if field == "fractions":
        fractions[1] = huge
    monkeypatch.setattr(kernels, "_nodes", forbid_search)
    monkeypatch.setattr(kernels, "_calibrate_at_order", forbid_search)
    with pytest.raises(AnalysisError) as error:
        if entrypoint == "public":
            options = {"fractions": fractions}
            if field in ("effect", "alpha"):
                options[field] = huge
            oe.sequential_design(**options)
        elif field == "effect":
            kernels.probabilities(fractions, [3.0, 2.8, 2.5, 2.2], drift=huge, sides=2)
        else:
            kernels.calibrate(fractions, huge if field == "alpha" else 0.05, "power", sides=2)
    assert error.value.code == "invalid_sequential_input"


@pytest.mark.parametrize("state", [None, [], '{"schema":NaN}', " " * (1024 * 1024 + 1)])
def test_malformed_or_oversized_state_has_structured_error(state):
    with pytest.raises(AnalysisError) as error:
        oe.restore_sequential_design(state)
    assert error.value.code == "invalid_state"


def test_checksum_and_nested_allocation_guards_precede_probability_work(designs, monkeypatch):
    monkeypatch.setattr(kernels, "probabilities", forbid_search)
    checksum = copy.deepcopy(designs["hsd", 2].attrs["state"])
    checksum["checksum"] = "0" * 64
    oversized = copy.deepcopy(designs["hsd", 2].attrs["state"])
    oversized["settings"]["fractions"] = [0.25] * 101
    deep = copy.deepcopy(designs["hsd", 2].attrs["state"])
    deep["extra"] = [{}]
    cursor = deep["extra"][0]
    for _ in range(16):
        cursor["child"] = {}
        cursor = cursor["child"]
    for state in (checksum, oversized, deep):
        with pytest.raises(AnalysisError) as error:
            oe.restore_sequential_design(state)
        assert error.value.code == "invalid_state"


@pytest.mark.parametrize("operation", ["design", "restore", "power", "information"])
def test_insufficient_current_work_and_workspace_refuse_all_public_operations(designs, operation):
    result = designs["ldof", 1]
    def run(max_work=2_000_000_000):
        if operation == "design":
            return oe.sequential_design(fractions=[0.25, 0.5, 0.75, 1.0], sides=1, max_work=max_work)
        if operation == "restore":
            return oe.restore_sequential_design(result, max_work=max_work)
        if operation == "power":
            return oe.sequential_power(result, effect=0.2, max_work=max_work)
        return oe.sequential_information(result, effect=0.3, max_work=max_work)
    with pytest.raises(AnalysisError) as error:
        run(max_work=1)
    assert error.value.code == "resource_limit"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        run()
    assert error.value.code == "workspace_limit"


def test_callers_inputs_and_ambient_runtime_are_preserved(designs):
    fractions = [0.25, 0.5, 0.75, 1.0]
    previous_dtype = torch.get_default_dtype()
    before = complete_bytes(designs["hsd", 2])
    with preserve_torch_default_device():
        try:
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("meta")
            result = oe.sequential_design(fractions=fractions, spending="hsd", sides=2)
            restored = oe.restore_sequential_design(result.attrs["state"])
            assert complete_bytes(restored) == complete_bytes(result)
            assert str(torch.get_default_device()) == "meta"
            assert torch.get_default_dtype() == torch.float32
            assert result["canonical_covariance"].to_numpy().dtype == np.dtype("float64")
        finally:
            torch.set_default_dtype(previous_dtype)
    assert fractions == [0.25, 0.5, 0.75, 1.0]
    assert complete_bytes(designs["hsd", 2]) == before

    # Preserve the real override identity, not just the device value. An indexed
    # CUDA label needs no CUDA hardware or allocation for this cleanup proof.
    from torch.overrides import TorchFunctionMode, _get_current_function_mode_stack

    class PassthroughMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with preserve_torch_default_device():
        for starting_device in (None, "cpu", "meta", "cuda:0"):
            torch.set_default_device(starting_device)
            original = torch._GLOBAL_DEVICE_CONTEXT.device_context
            for exceptional_exit in (False, True):
                with PassthroughMode():
                    original_modes = tuple(_get_current_function_mode_stack())
                    try:
                        with preserve_torch_default_device():
                            torch.set_default_device("meta")
                            if exceptional_exit:
                                raise RuntimeError("intentional cleanup probe")
                    except RuntimeError as error:
                        assert exceptional_exit and str(error) == "intentional cleanup probe"
                    assert torch._GLOBAL_DEVICE_CONTEXT.device_context is original
                    actual_modes = tuple(_get_current_function_mode_stack())
                    assert len(actual_modes) == len(original_modes)
                    assert all(a is b for a, b in zip(actual_modes, original_modes))
            assert str(torch.get_default_device()) == (starting_device or "cpu")
    assert torch.get_default_dtype() == previous_dtype
