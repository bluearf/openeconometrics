"""Scalar iid CLR inference with full real-line geometry and semantic replay."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import math

import torch
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace
from . import kernels
from .common import (
    SCHEMA,
    TEST_SCHEMA,
    MAX_JSON,
    _freeze,
    _thaw,
    close,
    controls,
    digest,
    fail,
    load,
    metadata,
    real,
    restore_source,
    seal,
    source,
    specification,
)

ASSUMPTIONS = (
    "One continuous endogenous regressor; fixed full-rank excluded instruments and included controls; "
    "valid instrument exclusion and iid homoskedastic reduced-form innovations with positive-definite "
    "joint covariance. Feasible estimated-covariance inference additionally requires the iid regularity "
    "and moment conditions for uniform weak-instrument asymptotics. These assumptions are declared, "
    "not established by observed sample rank."
)
NUMERICS = (
    "Successive Gauss-Legendre integration differences and monotone root/endpoint brackets are "
    "numerical convergence diagnostics, not rigorous interval-arithmetic probability bounds. "
    "Unresolved quadratic topology is refused; no finite coefficient window or grid tails are inferred."
)


def checked(function):
    @wraps(function)
    @resident_cpu
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except AnalysisError:
            raise
        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            OverflowError,
            RecursionError,
        ) as exc:
            fail(f"Complete scalar CLR admission/replay failed: {exc}")

    return wrapped


def _precision(dtype, device):
    if dtype not in ("float64", torch.float64) or device != "cpu":
        fail("Scalar CLR supports native Torch float64 on CPU only.", "unsupported_backend")


def _plan(actual, saved):
    keys = {"operation", "estimated_workspace_bytes", "budget_bytes", "buffers", "scope"}
    if (
        not isinstance(saved, dict)
        or set(saved) != keys
        or type(saved["budget_bytes"]) is not int
        or not actual["estimated_workspace_bytes"] <= saved["budget_bytes"] <= 10**15
    ):
        fail("Invalid saved CLR resource-plan envelope.")
    for key in keys - {"budget_bytes"}:
        if actual[key] != saved[key]:
            fail("Saved CLR resource accounting differs.")


def _array(value, shape, label):
    if not isinstance(value, list) or len(value) != shape[0]:
        fail(f"Invalid saved {label} dimensions.")
    if len(shape) > 1:
        for item in value:
            _array(item, shape[1:], label)
    elif any(type(v) is not float or not math.isfinite(v) for v in value):
        fail(f"Saved {label} must contain finite float primitives.")


def _envelope(state):
    fields = {
        "schema",
        "spec",
        "options",
        "source",
        "positions",
        "geometry",
        "root",
        "solution",
        "admitted_work",
        "resource_plan",
        "sha256",
    }
    if (
        set(state) != fields
        or state["schema"] != SCHEMA
        or not isinstance(state["spec"], dict)
        or set(state["spec"]) != {"y", "endog", "x", "instruments"}
    ):
        fail("Invalid complete CLR confidence-set schema.")
    spec = specification(**state["spec"])
    if not isinstance(state["options"], dict) or set(state["options"]) != {
        "confidence",
        "intercept",
        "missing",
        "omega",
        "probability_tolerance",
        "root_tolerance",
        "max_order",
        "max_iterations",
        "max_work",
    }:
        fail("Invalid saved CLR numerical options.")
    cfg = controls(**state["options"])
    if spec != state["spec"] or cfg != state["options"]:
        fail("Saved CLR specification/options are not canonical.")
    if (
        not isinstance(state["positions"], list)
        or len(state["positions"]) > 10000
        or any(type(v) is not int or not 0 <= v < 10000 for v in state["positions"])
    ):
        fail("Invalid saved CLR physical sample positions.")
    g = state["geometry"]
    matrices = {
        "normalized_covariance",
        "reduced_form_covariance",
        "normalized_projected_gram",
        "projected_gram",
        "whitened_gram",
        "eigen_directions",
    }
    other = {
        "n",
        "k",
        "controls",
        "df_reduced_form",
        "response_scales",
        "coefficient_unit",
        "maximum_eigenvalue",
        "eigen_gap",
        "omega_convention",
    }
    if not isinstance(g, dict) or set(g) != matrices | other:
        fail("Invalid complete CLR joint geometry schema.")
    for key in matrices:
        _array(g[key], (2, 2), key)
    _array(g["response_scales"], (2,), "response scales")
    for key in ("n", "k", "controls", "df_reduced_form"):
        if type(g[key]) is not int or not 0 <= g[key] <= 10000:
            fail("Invalid CLR sample/rank cache.")
    for key in ("coefficient_unit", "maximum_eigenvalue", "eigen_gap"):
        if type(g[key]) is not float or not math.isfinite(g[key]) or g[key] < 0:
            fail("Invalid CLR numeric geometry cache.")
    if (
        g["coefficient_unit"] == 0
        or any(v <= 0 for v in g["response_scales"])
        or g["omega_convention"] not in ("supplied_known", "residual_df")
    ):
        fail("Invalid CLR covariance/scaling convention cache.")
    r = state["root"]
    if (
        not isinstance(r, dict)
        or set(r)
        != {
            "branch",
            "lr",
            "lr_bracket",
            "conditioning_threshold",
            "iterations",
            "evaluations",
            "trace",
        }
        or not isinstance(r["trace"], list)
        or len(r["trace"]) > cfg["max_iterations"]
    ):
        fail("Invalid complete CLR numerical root cache.")
    for record in r["trace"]:
        if not isinstance(record, dict) or set(record) != {
            "lr",
            "value",
            "error_estimate",
            "order",
            "evaluations",
        }:
            fail("Invalid CLR quadrature trace schema.")
        if (
            any(
                type(record[v]) is not float or not math.isfinite(record[v]) or record[v] < 0
                for v in ("lr", "value", "error_estimate")
            )
            or record["value"] > 1
        ):
            fail("Invalid CLR quadrature trace primitive values.")
        if any(
            type(record[v]) is not int or not 0 <= record[v] <= 32 * cfg["max_order"]
            for v in ("order", "evaluations")
        ):
            fail("Invalid CLR quadrature trace primitive counts.")
    if (
        type(r["iterations"]) is not int
        or not 0 <= r["iterations"] <= cfg["max_iterations"]
        or type(r["evaluations"]) is not int
        or not 0 <= r["evaluations"] <= 32 * cfg["max_order"] * cfg["max_iterations"]
        or r["branch"]
        not in ("all_real_small_maximum", "analytic_one_instrument", "monotone_conditional_root")
    ):
        fail("Invalid CLR inversion branch/count cache.")
    for key in ("lr", "conditioning_threshold"):
        if r[key] is not None and (
            type(r[key]) is not float or not math.isfinite(r[key]) or r[key] < 0
        ):
            fail("Invalid CLR critical scalar cache.")
    if r["lr_bracket"] is not None:
        _array(r["lr_bracket"], (2,), "critical bracket")
    solution = state["solution"]
    if not isinstance(solution, dict) or set(solution) != {
        "topology",
        "intervals",
        "endpoint_brackets",
        "normalized_inequality",
    }:
        fail("Invalid complete CLR global solution schema.")
    for key in ("intervals", "endpoint_brackets"):
        if (
            not isinstance(solution[key], list)
            or not 1 <= len(solution[key]) <= 2
            or any(not isinstance(row, list) or len(row) != 2 for row in solution[key])
        ):
            fail("Invalid CLR interval cache dimensions.")
    if solution["topology"] not in ("all_real", "bounded_interval", "two_unbounded_rays"):
        fail("Invalid CLR interval topology cache.")
    for row in solution["intervals"]:
        if any(v is not None and (type(v) is not float or not math.isfinite(v)) for v in row):
            fail("Invalid CLR endpoint primitive cache.")
    for row in solution["endpoint_brackets"]:
        for pair in row:
            if pair is not None:
                _array(pair, (2,), "endpoint bracket")
    if solution["normalized_inequality"] is not None:
        _array(solution["normalized_inequality"], (2, 2), "normalized inequality")
    return spec, cfg


def _raw(value, cls):
    if isinstance(value, TableSet):
        value = value.attrs.get("state")
    if isinstance(value, cls):
        value = value.payload  # Never trust model_copy/model_construct or serialize first.
    value = load(value)
    if set(value) == {"payload"}:
        value = load(value["payload"])
    return value


@checked
def _replay(value):
    state = _raw(value, CLRConfidenceSet)
    spec, cfg = _envelope(state)
    if (
        not isinstance(state["sha256"], str)
        or len(state["sha256"]) != 64
        or digest({k: v for k, v in state.items() if k != "sha256"}) != state["sha256"]
    ):
        fail("Saved CLR confidence-set integrity check failed.")
    positions, rows, work, plan = restore_source(state["source"], spec, cfg)
    if (
        positions != state["positions"]
        or type(state["admitted_work"]) is not int
        or state["admitted_work"] != work
    ):
        fail("Saved CLR source sample/admission differs.")
    _plan(plan, state["resource_plan"])
    g = kernels.geometry(rows, spec, cfg)
    close(g, state["geometry"], "geometry")
    root = kernels.critical(g, cfg)
    close(root, state["root"], "root")
    solution = kernels.intervals(g, root)
    close(solution, state["solution"], "solution")
    return state


def _result(state):
    g, cfg, s = state["geometry"], state["options"], state["solution"]
    rows = []
    for i, (lower, upper) in enumerate(s["intervals"]):
        brackets = s["endpoint_brackets"][i]
        rows.append(
            dict(
                component=i + 1,
                lower=lower,
                upper=upper,
                lower_unbounded=lower is None,
                upper_unbounded=upper is None,
                lower_closed=lower is not None,
                upper_closed=upper is not None,
                lower_bracket_low=None if brackets[0] is None else brackets[0][0],
                lower_bracket_high=None if brackets[0] is None else brackets[0][1],
                upper_bracket_low=None if brackets[1] is None else brackets[1][0],
                upper_bracket_high=None if brackets[1] is None else brackets[1][1],
            )
        )
    names = [state["spec"]["y"], state["spec"]["endog"]]
    exact = cfg["omega"] is not None
    return TableSet(
        {
            "intervals": table(rows),
            "reduced_form_covariance": table(
                g["reduced_form_covariance"], columns=names, index=names
            ),
            "projected_gram": table(g["projected_gram"], columns=names, index=names),
            "inversion": table(
                [
                    dict(
                        nobs=g["n"],
                        excluded_rank=g["k"],
                        df_reduced_form=g["df_reduced_form"],
                        maximum_eigenvalue=g["maximum_eigenvalue"],
                        eigen_gap=g["eigen_gap"],
                        lr_critical=state["root"]["lr"],
                        conditioning_threshold=state["root"]["conditioning_threshold"],
                        root_iterations=state["root"]["iterations"],
                        quadrature_evaluations=state["root"]["evaluations"],
                    )
                ]
            ),
        },
        title="Scalar iid CLR confidence set over the real line",
        command="iv_clr_confidence_set",
        state=state,
        nobs=g["n"],
        sample_positions=state["positions"],
        confidence=cfg["confidence"],
        topology=s["topology"],
        dtype="float64",
        device="cpu",
        full_state=True,
        inference=(
            "finite Gaussian exact conditional CLR law with caller-supplied known joint covariance"
            if exact
            else "feasible estimated-covariance CLR; iid uniform weak-instrument asymptotic law"
        ),
        known_covariance=exact,
        assumptions=ASSUMPTIONS,
        numerics=NUMERICS,
        original_author_code_validated=False,
        licensed_vendor_validated=False,
        robust_subvector_supported=False,
        global_real_line=True,
    )


@checked
def iv_clr_confidence_set(
    data,
    y,
    endog,
    x=(),
    *,
    instruments,
    confidence=0.95,
    intercept=True,
    missing="raise",
    omega=None,
    probability_tolerance=1e-11,
    root_tolerance=1e-9,
    max_order=512,
    max_iterations=64,
    max_work=2_000_000_000,
    dtype="float64",
    device="cpu",
) -> TableSet:
    """Invert scalar iid Moreira CLR on the whole real line, without a grid.

    Full rank fixed instruments and homoskedastic iid innovations are required.
    Estimated Ω uses reduced-form residual df. Supplied known Ω gives the exact
    conditional law only under the finite Gaussian reduced-form assumptions.
    Bounded intervals, two unbounded rays and all-real sets retain their topology.
    Complete source/index/sample/Ω/root/quadratic state is saved and replayable.
    """
    _precision(dtype, device)
    spec = specification(y, endog, x, instruments)
    cfg = controls(
        confidence=confidence,
        intercept=intercept,
        missing=missing,
        omega=omega,
        probability_tolerance=probability_tolerance,
        root_tolerance=root_tolerance,
        max_order=max_order,
        max_iterations=max_iterations,
        max_work=max_work,
    )
    saved, positions, rows, work, plan = source(data, spec, cfg)
    g = kernels.geometry(rows, spec, cfg)
    root = kernels.critical(g, cfg)
    solution = kernels.intervals(g, root)
    state = seal(
        dict(
            schema=SCHEMA,
            spec=spec,
            options=cfg,
            source=saved,
            positions=positions,
            geometry=g,
            root=root,
            solution=solution,
            admitted_work=work,
            resource_plan=plan,
        )
    )
    return _result(state)


@checked
def iv_clr_restore(value) -> TableSet:
    """Semantically restore full source, joint covariance and real-line CLR geometry."""
    return _result(_replay(value))


@checked
def _test_replay(value):
    state = _raw(value, CLRTestState)
    if set(state) != {"schema", "target", "test", "sha256"} or state["schema"] != TEST_SCHEMA:
        fail("Invalid complete CLR null-test schema.")
    test = state["test"]
    if not isinstance(test, dict) or set(test) != {
        "null",
        "statistic",
        "conditioning_statistic",
        "conditional_p_value",
        "probability_error_estimate",
        "quadrature_order",
        "quadrature_evaluations",
        "accepted",
    }:
        fail("Invalid full CLR null-test cache.")
    for key in (
        "null",
        "statistic",
        "conditioning_statistic",
        "conditional_p_value",
        "probability_error_estimate",
    ):
        if type(test[key]) is not float or not math.isfinite(test[key]):
            fail("Invalid CLR null-test primitive cache.")
    if type(test["accepted"]) is not bool or any(
        type(test[v]) is not int or not 0 <= test[v] <= 16384
        for v in ("quadrature_order", "quadrature_evaluations")
    ):
        fail("Invalid CLR null-test decision/count cache.")
    null = real(test["null"], "saved null")
    if digest({k: v for k, v in state.items() if k != "sha256"}) != state["sha256"]:
        fail("Saved CLR null-test integrity check failed.")
    target = _replay(state["target"])
    actual = kernels.test(target["geometry"], null, target["options"])
    close(actual, test, "null test")
    return state


def _test_result(state):
    target = state["target"]
    return TableSet(
        {"clr_test": table([state["test"]])},
        title="Scalar iid conditional likelihood-ratio test",
        command="iv_clr_test",
        state=state,
        full_state=True,
        dtype="float64",
        device="cpu",
        confidence=target["options"]["confidence"],
        sample_positions=target["positions"],
        known_covariance=target["options"]["omega"] is not None,
        assumptions=ASSUMPTIONS,
        numerics=NUMERICS,
        original_author_code_validated=False,
        licensed_vendor_validated=False,
    )


@checked
def iv_clr_test(value, *, null=0.0) -> TableSet:
    """Test a finite scalar null from complete restored CLR target geometry."""
    target = _replay(value)
    null = real(null, "null")
    test = kernels.test(target["geometry"], null, target["options"])
    return _test_result(seal(dict(schema=TEST_SCHEMA, target=target, test=test)))


@checked
def iv_clr_test_restore(value) -> TableSet:
    """Restore a null test and its required full target, recomputing its probability."""
    return _test_result(_test_replay(value))


def _encoded_json(value):
    if isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_JSON:
            fail("CLR encoded JSON exceeds 32 MiB.", "state_limit")
        plan_workspace("CLR JSON text decoding", {"bounded Unicode decoding": 4 * len(value)})
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            fail("CLR JSON must be valid UTF-8.")
    if not isinstance(value, str):
        fail("CLR typed JSON requires text, bytes, or bytearray.")
    return load(value)


def _string_json_bytes(value, ensure_ascii):
    size = 2
    for character in value:
        ordinal = ord(character)
        if character in '\\"':
            size += 2
        elif ordinal < 32:
            size += 2 if character in "\b\f\n\r\t" else 6
        elif 0xD800 <= ordinal <= 0xDFFF:
            size += 6
        elif ensure_ascii and ordinal >= 128:
            size += 6 if ordinal <= 0xFFFF else 12
        else:
            size += 1 if ordinal < 128 else 2 if ordinal < 2048 else 3 if ordinal <= 65535 else 4
    return size


def _json_output_admission(envelope, indent, ensure_ascii):
    """Bound output and formatting before the allocating Pydantic serializer.

    Counts are conservative for omitted fields and whitespace, and use integer
    arithmetic rather than building indentation strings or encoded JSON.
    """
    if indent is not None and (type(indent) is not int or indent < 0):
        fail("CLR JSON indentation must be a nonnegative integer or None.", "invalid_argument")
    if type(ensure_ascii) is not bool:
        fail("CLR JSON ensure_ascii must be Boolean.", "invalid_argument")
    metadata(envelope)
    size, objects = 0, 0
    pending = [(iter((envelope,)), 0)]
    while pending:
        iterator, depth = pending[-1]
        try:
            item = next(iterator)
        except StopIteration:
            pending.pop()
            continue
        if isinstance(item, Mapping):
            size += 2 + 3 * len(item)
            objects += 256 + 96 * len(item)
            pending.append((iter(v for pair in item.items() for v in pair), depth + 1))
        elif isinstance(item, (tuple, list)):
            size += 2 + 2 * len(item)
            objects += 64 + 16 * len(item)
            pending.append((iter(item), depth + 1))
        elif isinstance(item, str):
            size += _string_json_bytes(item, ensure_ascii)
            objects += 64 + 4 * len(item)
        else:
            # All finite float64 JSON representations fit this bound; integer
            # digit estimates also cover the admitted typed-index primitives.
            size += max(32, item.bit_length() // 3 + 2) if type(item) is int else 32
            objects += 32
        if indent is not None and isinstance(item, (Mapping, tuple, list)):
            size += indent * (len(item) * (depth + 1) + depth) + len(item) + 2
        if size > MAX_JSON:
            fail("Formatted CLR JSON exceeds 32 MiB.", "state_limit")
    plan_workspace(
        "CLR JSON serialization",
        {
            "formatted JSON and encoding copies": 4 * size,
            "replayed serializer metadata": 3 * objects,
        },
    )


class _CLRTransport(BaseModel):
    @classmethod
    def model_validate_json(
        cls, json_data, *, strict=None, extra=None, context=None, by_alias=None, by_name=None
    ):
        # Preserve Pydantic's strict/context/extra/name/alias validation options
        # after bounded parsing, rather than entering its allocating JSON parser.
        return cls.model_validate(
            _encoded_json(json_data),
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    def model_copy(self, *, update=None, deep=False):
        # Pydantic deliberately leaves updates unchecked. These immutable
        # scientific states validate both the original and proposed state before
        # reconstruction; all admitted descendants are frozen afterwards.
        original = self._admitted_payload()
        if update is not None and not isinstance(update, Mapping):
            fail("CLR state-copy updates must be a mapping.")
        metadata(update if update is not None else {})
        candidate = {"payload": original}
        if update:
            candidate.update(update)
        return type(self).model_validate(candidate)

    def __deepcopy__(self, memo=None):
        copied = self.model_copy(deep=True)
        if memo is not None:
            memo[id(self)] = copied
        return copied

    @wraps(BaseModel.model_dump)
    def model_dump(self, *args, **kwargs):
        # Pydantic does not invoke a field serializer when selectors omit that
        # field. Admission therefore belongs to the complete export lifecycle,
        # even when its requested projection contains no scientific payload.
        self._admitted_payload()
        return super().model_dump(*args, **kwargs)

    @wraps(BaseModel.model_dump_json)
    def model_dump_json(self, *, indent=None, ensure_ascii=False, **kwargs):
        envelope = {"payload": self._admitted_payload()}
        _json_output_admission(envelope, indent, ensure_ascii)
        return super().model_dump_json(indent=indent, ensure_ascii=ensure_ascii, **kwargs)


class CLRConfidenceSet(_CLRTransport):
    """Immutable complete scalar iid CLR state; all readers repeat semantic replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    payload: object

    def _admitted_payload(self):
        return _replay(self.payload)

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value):
        return _freeze(_replay(value))

    @field_serializer("payload")
    def serialize_payload(self, value):
        return _thaw(_replay(value))

    def to_tables(self):
        return iv_clr_restore(self)


class CLRTestState(_CLRTransport):
    """Immutable full CLR null-test state, including its required target."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    payload: object

    def _admitted_payload(self):
        return _test_replay(self.payload)

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value):
        return _freeze(_test_replay(value))

    @field_serializer("payload")
    def serialize_payload(self, value):
        return _thaw(_test_replay(value))

    def to_tables(self):
        return iv_clr_test_restore(self)
