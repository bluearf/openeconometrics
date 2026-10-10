"""Bounded, lossless source and numerical admission for scalar iid CLR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import (
    _decode_index,
    _encode_index,
    _freeze,
    _index_envelope,
    _label,
    _thaw,
    _unlabel,
)
from openecon.resources import plan_workspace

SCHEMA = "openecon.clr_confidence_set.v1"
TEST_SCHEMA = "openecon.clr_test.v1"
MAX_ROWS, MAX_JSON = 10_000, 32 * 1024**2


def fail(message, code="invalid_state"):
    raise AnalysisError(code, message)


def real(value, label, *, low=None, high=None):
    if type(value) not in (int, float):
        fail(f"{label} must be a finite real scalar.", "invalid_argument")
    try:
        number = float(value)
    except OverflowError:
        fail(f"{label} exceeds float64.", "invalid_argument")
    if (
        not math.isfinite(number)
        or low is not None
        and number < low
        or high is not None
        and number > high
    ):
        fail(f"{label} is outside its admitted finite range.", "invalid_argument")
    return number


def integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        fail(f"{label} must be an integer in {low}..{high}.", "invalid_argument")
    return value


def name(value):
    if not isinstance(value, str) or not value or len(value.encode()) > 128 or value == "_cons":
        fail(
            "CLR column names must be nonempty, at most 128 UTF-8 bytes, and not _cons.",
            "invalid_argument",
        )
    return value


def specification(y, endog, x, instruments):
    if (
        not isinstance(x, Sequence)
        or isinstance(x, (str, bytes))
        or not isinstance(instruments, Sequence)
        or isinstance(instruments, (str, bytes))
    ):
        fail("x and instruments must be explicit column-name sequences.", "invalid_argument")
    if len(x) > 12 or not 1 <= len(instruments) <= 12:
        fail(
            "CLR admits at most 12 included predictors and 1..12 excluded instruments.",
            "shape_limit",
        )
    names = [name(y), name(endog), *map(name, x), *map(name, instruments)]
    if len(set(names)) != len(names):
        fail(
            "Outcome, endogenous regressor, controls and instruments must be distinct.",
            "invalid_argument",
        )
    return dict(y=y, endog=endog, x=list(x), instruments=list(instruments))


def controls(
    *,
    confidence=0.95,
    intercept=True,
    missing="raise",
    omega=None,
    probability_tolerance=1e-11,
    root_tolerance=1e-9,
    max_order=512,
    max_iterations=64,
    max_work=2_000_000_000,
):
    confidence = real(confidence, "confidence", low=0.5, high=1 - 1e-8)
    if type(intercept) is not bool or missing not in ("raise", "drop"):
        fail("intercept must be Boolean; missing must be raise or drop.", "invalid_argument")
    probability_tolerance = real(
        probability_tolerance, "probability_tolerance", low=1e-13, high=1e-7
    )
    root_tolerance = real(root_tolerance, "root_tolerance", low=1e-12, high=1e-5)
    if probability_tolerance > root_tolerance / 10:
        fail(
            "Probability tolerance must be at most one tenth of root tolerance.", "invalid_argument"
        )
    max_order = integer(max_order, "max_order", 64, 512)
    if max_order & (max_order - 1):
        fail("max_order must be a power of two.", "invalid_argument")
    max_iterations = integer(max_iterations, "max_iterations", 8, 128)
    max_work = integer(max_work, "max_work", 1, 10**12)
    if omega is not None:
        if (
            not isinstance(omega, (list, tuple))
            or len(omega) != 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in omega)
        ):
            fail(
                "Known omega must be a full symmetric positive-definite 2 by 2 matrix.",
                "invalid_covariance",
            )
        omega = [[real(v, "omega entry") for v in row] for row in omega]
        if omega[0][0] <= 0 or omega[1][1] <= 0 or omega[0][1] != omega[1][0]:
            fail("Known omega must be symmetric with positive diagonal.", "invalid_covariance")
        correlation = omega[0][1] / math.sqrt(omega[0][0]) / math.sqrt(omega[1][1])
        if not math.isfinite(correlation) or abs(correlation) >= 1 - 1e-10:
            fail("Known omega is singular or numerically unresolved.", "invalid_covariance")
    return dict(
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


def admission(n, width, cfg):
    integer(n, "original rows", 1, MAX_ROWS)
    work = (
        80 * n * width**2
        + 4 * cfg["max_order"] ** 3
        + 64 * cfg["max_iterations"] * 2 * cfg["max_order"] * 12
    )
    if work > cfg["max_work"]:
        fail(f"CLR conservative arithmetic plan {work} exceeds max_work.", "work_limit")
    plan = plan_workspace(
        "scalar iid whole-line CLR",
        {
            "source copies and typed index": 512 * n * width + 3 * 2 * 1024**2,
            "partial projections and QR/SVD buffers": 8 * n * (12 * width + 32),
            "bounded quadrature eigensystem": 40 * cfg["max_order"] ** 2,
            "inversion trace and replay copies": 1024 * cfg["max_iterations"],
        },
    ).record()
    return work, plan


def metadata(value):
    """Iterator DFS bounds children before copying, thawing, hashing or tensors."""
    pending, size, nodes = [(iter((value,)), 0)], 0, 0
    while pending:
        iterator, depth = pending[-1]
        try:
            item = next(iterator)
        except StopIteration:
            pending.pop()
            continue
        nodes += 1
        if depth > 64 or nodes > 2_000_000:
            fail("Complete CLR metadata exceeds supported traversal bounds.", "state_limit")
        if isinstance(item, Mapping):
            size += 256 + 96 * len(item)
            pending.append((iter(v for pair in item.items() for v in pair), depth + 1))
        elif isinstance(item, (list, tuple)):
            size += 64 + 16 * len(item)
            pending.append((iter(item), depth + 1))
        elif isinstance(item, str):
            size += 64 + 4 * len(item)
        elif type(item) is int:
            if item.bit_length() > 1024:
                fail("Saved integer exceeds its finite primitive envelope.", "invalid_state")
            size += 32 + item.bit_length() // 8
        elif item is None or type(item) is bool or type(item) is float and math.isfinite(item):
            size += 32
        else:
            fail("CLR state requires finite JSON primitives.", "invalid_state")
        if nodes % 1024 == 0 or isinstance(item, (Mapping, list, tuple, str)) and len(item) > 1024:
            plan_workspace(
                "complete CLR metadata", {"metadata and bounded replay copies": 3 * size}
            )
        if size > 128 * 1024**2:
            fail("Complete CLR metadata exceeds its object-size boundary.", "state_limit")
    plan_workspace("complete CLR metadata", {"metadata and bounded replay copies": 3 * size})


def digest(state):
    return hashlib.sha256(
        json.dumps(_thaw(state), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seal(state):
    metadata(state)
    if len(json.dumps(state, allow_nan=False).encode()) > MAX_JSON:
        fail("Complete CLR JSON exceeds 32 MiB.", "state_limit")
    return dict(state, sha256=digest(state))


def json_admission(value, encoded_size):
    """Scan JSON syntax before decoding; dense tiny containers need more than
    a fixed byte multiplier. Strings are scanned without copying their contents.
    The actual decoder still performs syntax validation after this upper plan.
    """
    size, depth, nodes, quoted, escaped, token = 0, 0, 0, False, False, False
    for position, character in enumerate(value):
        if quoted:
            size += 4
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted, token = True, False
            size += 64
            nodes += 1
        elif character in "[{":
            size += 80 if character == "[" else 256
            depth += 1
            nodes += 1
            token = False
        elif character in "]}":
            depth -= 1
            token = False
        elif character == ":":
            size += 96
            token = False
        elif character == ",":
            size += 16
            token = False
        elif character.isspace():
            token = False
        elif not token:
            size += 32
            nodes += 1
            token = True
        if depth > 64 or nodes > 2_000_000 or size > 128 * 1024**2:
            fail("Encoded CLR JSON exceeds metadata traversal bounds.", "state_limit")
        if position % 4096 == 0:
            plan_workspace(
                "CLR JSON admission",
                {"decoded metadata and replay copies": 4 * size + encoded_size},
            )
    plan_workspace(
        "CLR JSON admission", {"decoded metadata and replay copies": 4 * size + encoded_size}
    )


def load(value):
    if isinstance(value, str):
        if len(value) > MAX_JSON:
            fail("CLR JSON exceeds 32 MiB.", "state_limit")
        encoded_size = 0
        for start in range(0, len(value), 65536):
            encoded_size += len(value[start : start + 65536].encode())
            if encoded_size > MAX_JSON:
                fail("CLR encoded JSON exceeds 32 MiB.", "state_limit")
        json_admission(value, encoded_size)
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            fail("CLR JSON is malformed or too deeply nested.")
    metadata(value)
    if not isinstance(value, Mapping):
        fail("Supply a complete CLR state, saved table, typed state, or JSON.")
    value = _thaw(value)
    if len(json.dumps(value, allow_nan=False).encode()) > MAX_JSON:
        fail("CLR JSON exceeds 32 MiB.", "state_limit")
    return value


def encode_index(index):
    # Extend the shared lossless codec for PeriodIndex, including nested levels.
    if isinstance(index, pd.PeriodIndex):
        return dict(
            kind="period", name=_label(index.name), freq=index.freqstr, ordinals=index.asi8.tolist()
        )
    if isinstance(index, pd.MultiIndex):
        if index.nlevels > 16:
            fail("CLR admits at most 16 row-index levels.", "unsupported_index")
        return dict(
            kind="multi",
            names=[_label(v) for v in index.names],
            levels=[encode_index(v) for v in index.levels],
            codes=[v.tolist() for v in index.codes],
        )
    if isinstance(index, pd.CategoricalIndex):
        return dict(
            kind="categorical",
            name=_label(index.name),
            categories=encode_index(index.categories),
            ordered=index.ordered,
            codes=index.codes.tolist(),
        )
    return _encode_index(index)


def decode_index(state, n, depth=0):
    if not isinstance(state, Mapping) or depth > 16:
        fail("Invalid CLR typed index nesting.")
    kind = state.get("kind")
    if kind == "period":
        if (
            set(state) != {"kind", "name", "freq", "ordinals"}
            or not isinstance(state["ordinals"], list)
            or len(state["ordinals"]) > n
            or any(type(v) is not int or not -(2**63) <= v < 2**63 for v in state["ordinals"])
        ):
            fail("Invalid saved period index.")
        return pd.PeriodIndex.from_ordinals(
            state["ordinals"], freq=state["freq"], name=_unlabel(state["name"])
        )
    if kind == "multi":
        if (
            set(state) != {"kind", "names", "levels", "codes"}
            or not 1 <= len(state["levels"]) <= 16
            or len(state["levels"]) != len(state["codes"])
            or len(state["levels"]) != len(state["names"])
            or any(len(v) != n for v in state["codes"])
        ):
            fail("Invalid saved multi index dimensions.")
        return pd.MultiIndex(
            levels=[decode_index(v, MAX_ROWS, depth + 1) for v in state["levels"]],
            codes=state["codes"],
            names=[_unlabel(v) for v in state["names"]],
            verify_integrity=True,
        )
    if kind == "categorical":
        if (
            set(state) != {"kind", "name", "categories", "ordered", "codes"}
            or type(state["ordered"]) is not bool
            or len(state["codes"]) > n
        ):
            fail("Invalid saved categorical index.")
        return pd.CategoricalIndex(
            pd.Categorical.from_codes(
                state["codes"],
                categories=decode_index(state["categories"], MAX_ROWS, depth + 1),
                ordered=state["ordered"],
            ),
            name=_unlabel(state["name"]),
        )
    _index_envelope(state, n)
    return _decode_index(state)


def dtype_check(dtype):
    if is_bool_dtype(dtype) or not (is_float_dtype(dtype) or is_integer_dtype(dtype)):
        fail(
            "CLR source columns require real floating or integer dtypes; Boolean, object, complex and categorical columns are unsupported.",
            "unsupported_dtype",
        )
    if getattr(dtype, "itemsize", 8) > 8:
        fail("CLR source dtypes wider than float64/int64 are unsupported.", "unsupported_dtype")


def source(data, spec, cfg):
    if not isinstance(data, pd.DataFrame):
        fail("CLR requires a resident pandas-compatible DataFrame.", "unsupported_input")
    names = [spec["y"], spec["endog"], *spec["x"], *spec["instruments"]]
    if not data.columns.is_unique or any(v not in data.columns for v in names):
        fail("CLR source needs unique columns and every declared variable.", "invalid_argument")
    work, plan = admission(len(data), len(names), cfg)
    for v in names:
        dtype_check(data[v].dtype)
    if data.index.memory_usage(deep=True) > 2 * 1024**2:
        fail("CLR typed original index exceeds 2 MiB.", "state_limit")
    index = encode_index(data.index)
    metadata(index)
    if len(json.dumps(index, allow_nan=False).encode()) > 2 * 1024**2:
        fail("CLR typed original index exceeds 2 MiB.", "state_limit")
    values, positions, rows = [], [], []
    for i, row in enumerate(data[names].itertuples(index=False, name=None)):
        saved = []
        for j, value in enumerate(row):
            if pd.isna(value):
                saved.append(None)
                continue
            if hasattr(value, "item"):
                value = value.item()
            if is_integer_dtype(data[names[j]].dtype):
                if type(value) is not int or abs(value) > 10**12 or int(float(value)) != value:
                    fail(
                        "Integer source values must be exactly representable and within ±1e12.",
                        "invalid_argument",
                    )
            else:
                value = float(value)
                if not math.isfinite(value) or abs(value) > 1e12:
                    fail("Floating source values must be finite within ±1e12.", "invalid_argument")
            saved.append(value)
        values.append(saved)
        if all(v is not None for v in saved):
            positions.append(i)
            rows.append(saved)
        elif cfg["missing"] == "raise":
            fail(
                "CLR source has missing selected values; request missing='drop' explicitly.",
                "missing_values",
            )
    return (
        dict(
            names=names,
            dtypes=[str(data[v].dtype) for v in names],
            values=values,
            index=index,
            n_original=len(data),
        ),
        positions,
        rows,
        work,
        plan,
    )


def restore_source(saved, spec, cfg):
    if not isinstance(saved, Mapping) or set(saved) != {
        "names",
        "dtypes",
        "values",
        "index",
        "n_original",
    }:
        fail("Invalid complete CLR source schema.")
    n = integer(saved["n_original"], "saved original rows", 1, MAX_ROWS)
    names = [spec["y"], spec["endog"], *spec["x"], *spec["instruments"]]
    width = len(names)
    admission(n, width, cfg)
    if (
        saved["names"] != names
        or not isinstance(saved["dtypes"], list)
        or len(saved["dtypes"]) != width
        or not isinstance(saved["values"], list)
        or len(saved["values"]) != n
        or any(not isinstance(v, list) or len(v) != width for v in saved["values"])
    ):
        fail("Invalid source names/dtypes/row dimensions.")
    if len(json.dumps(saved["index"], allow_nan=False).encode()) > 2 * 1024**2:
        fail("Saved typed index exceeds 2 MiB.", "state_limit")
    index = decode_index(saved["index"], n)
    if len(index) != n or encode_index(index) != saved["index"]:
        fail("Saved index is not a lossless canonical typed descriptor.")
    columns = {}
    for j, (column, dtype) in enumerate(zip(names, saved["dtypes"])):
        if not isinstance(dtype, str) or len(dtype) > 64:
            fail("Invalid saved numeric dtype.")
        dtype_check(pd.api.types.pandas_dtype(dtype))
        vals = [v[j] for v in saved["values"]]
        typ = int if is_integer_dtype(pd.api.types.pandas_dtype(dtype)) else float
        if any(v is not None and type(v) is not typ for v in vals):
            fail("Saved dtype and primitive value types disagree.")
        columns[column] = pd.Series(vals, index=index, dtype=dtype)
    data = pd.DataFrame(columns, index=index)
    canonical, positions, rows, work, plan = source(data, spec, cfg)
    if canonical != saved:
        fail("Saved source cannot round-trip without dtype/value/sample coercion.")
    return positions, rows, work, plan


def close(actual, saved, label):
    """Relative, unit-normalized replay; no dimensionful absolute tolerance."""
    if isinstance(actual, dict):
        if not isinstance(saved, dict) or set(actual) != set(saved):
            fail(f"Saved {label} schema differs.")
        for key in actual:
            close(actual[key], saved[key], f"{label}.{key}")
    elif isinstance(actual, list):
        if not isinstance(saved, list) or len(actual) != len(saved):
            fail(f"Saved {label} dimensions differ.")
        # Full square matrices use each diagonal's reporting unit. Cross entries
        # near zero are compared in their covariance/Gram coordinate units.
        if (
            len(actual) == 2
            and all(isinstance(v, list) and len(v) == 2 for v in actual)
            and ("covariance" in label or "gram" in label or "inequality" in label)
        ):
            scale = [math.sqrt(abs(actual[i][i])) for i in range(2)]
            for i in range(2):
                for j in range(2):
                    a, b = actual[i][j], saved[i][j]
                    bound = max(abs(a), scale[i] * scale[j])
                    if (
                        type(b) not in (int, float)
                        or not math.isfinite(b)
                        or abs(a - b) > 2e-11 * bound
                    ):
                        fail(f"Saved {label} numeric value differs in normalized units.")
        else:
            for i, (a, b) in enumerate(zip(actual, saved)):
                close(a, b, f"{label}[{i}]")
    elif type(actual) is float:
        if (
            type(saved) is not float
            or not math.isfinite(saved)
            or abs(actual - saved) > 2e-11 * abs(actual)
        ):
            fail(f"Saved {label} numeric value differs.")
    elif type(actual) is not type(saved) or actual != saved:
        fail(f"Saved {label} differs.")


__all__ = [
    "SCHEMA",
    "TEST_SCHEMA",
    "controls",
    "specification",
    "source",
    "restore_source",
    "metadata",
    "digest",
    "seal",
    "load",
    "close",
    "fail",
    "real",
    "_freeze",
    "_thaw",
]
