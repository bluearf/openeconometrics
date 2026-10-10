"""Finite BMA input, source and typed-lifecycle admission before allocations."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from functools import wraps
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_unsigned_integer_dtype,
)
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

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
from openecon.econometrics.state_lifecycle import (
    _encoded_json_admission,
    _json_export_admission,
    _metadata_geometry,
    _state_copy_admission,
)
from openecon.resources import plan_workspace, workspace_budget_bytes

MAX_ROWS = 10000
MAX_OPTIONAL = MAX_FORCED = 8
JSON_LIMIT = 32 * 1024**2
MAX_ELEMENTS = 2_000_000
DEFAULT_WORK = 100_000_000
DEFAULT_BYTES = 256 * 1024**2


def fail(text, code="invalid_bma_state"):
    raise AnalysisError(code, text)


def cpu_call(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.device("cpu"):
            try:
                return fn(*args, **kwargs)
            except (OverflowError, ZeroDivisionError) as exc:
                fail(
                    f"BMA numerical operation exceeds finite float64 support: {exc}",
                    "numerical_failure",
                )

    return wrapped


def integer(value, name, low=0, high=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        fail(f"{name} must be an integer in [{low},{high}].", "invalid_option")
    return int(value)


def real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        fail(f"{name} must be a finite real number, excluding booleans.", "invalid_option")
    try:
        valid = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid or (positive and value <= 0):
        fail(f"{name} must be finite{' and positive' if positive else ''}.", "invalid_option")
    return float(value)


def names(value, name, maximum):
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        fail(
            f"{name} must be a resident ordered list/tuple of at most {maximum} names.",
            "invalid_spec",
        )
    if any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in value) or len(
        set(value)
    ) != len(value):
        fail(f"{name} needs distinct nonempty names of at most1000 characters.", "invalid_spec")
    return list(value)


def specification(y, forced, optional, missing, alpha, max_work, max_bytes):
    forced, optional = (
        names(forced, "forced", MAX_FORCED),
        names(optional, "optional", MAX_OPTIONAL),
    )
    if not isinstance(y, str) or not y.strip() or len(y) > 1000:
        fail("A nonempty outcome name is required.", "invalid_spec")
    columns = [y, *forced, *optional]
    if len(set(columns)) != len(columns) or any(
        v in {"Intercept", "sigma_squared"} for v in columns
    ):
        fail(
            "Outcome and all predictors must be distinct; Intercept/sigma_squared are reserved.",
            "invalid_spec",
        )
    if missing not in ("raise", "drop"):
        fail("missing must be raise or drop.", "invalid_spec")
    alpha = real(alpha, "alpha")
    if not 0 < alpha < 1 or not 0 < alpha / 2 < 1 - alpha / 2 < 1:
        fail("alpha must resolve two strict float64 interval probabilities.", "invalid_option")
    return dict(
        y=y,
        forced=forced,
        optional=optional,
        missing=missing,
        alpha=alpha,
        max_work=integer(max_work, "max_work", 1, 10**12),
        max_bytes=integer(max_bytes, "max_bytes", 1, 2**40),
    )


def indices(spec):
    f, p = len(spec["forced"]), len(spec["optional"])
    return [
        [*range(f + 1), *(f + 1 + j for j in range(p) if mask & (1 << j))] for mask in range(1 << p)
    ]


def plan(n, spec, *, queries=0, draws=0, index_bytes=0):
    n = integer(n, "source rows", 1, MAX_ROWS)
    queries = integer(queries, "query rows", 0, MAX_ROWS)
    draws = integer(draws, "draws", 0, 10000)
    index_bytes = integer(index_bytes, "index bytes")
    k = 1 + len(spec["forced"]) + len(spec["optional"])
    models = indices(spec)
    k2, k3 = sum(len(v) ** 2 for v in models), sum(len(v) ** 3 for v in models)
    # Retained per-model query covariances plus all four joint output covariances,
    # three model-vector fields, three parameter/query cross-covariances, query
    # positions/means/four quantile endpoints, and the live query design.
    query_elements = (
        (len(models) + 4) * queries**2
        + 3 * len(models) * queries
        + 3 * (k + 1) * queries
        + 6 * queries
        + queries * k
    )
    elements = len(models) * k**2 + query_elements + draws * (k + 2 * queries + 3)
    if elements > MAX_ELEMENTS:
        fail("Complete model/query/draw arrays exceed2000000 values.", "dimension_limit")
    work = 16 * n * k2 + 80 * k3 + 8 * 2048 * len(models) * (k + 1 + 2 * queries)
    work += (
        16 * queries * k2
        + 16 * len(models) * queries**2
        + 16 * draws * (k**2 + queries * k + len(models) + 1)
    )
    if work > spec["max_work"]:
        fail(f"BMA needs {work:,} bounded work units, exceeding max_work.", "work_limit")
    return plan_workspace(
        "complete_finite_bma",
        {
            "one complete source and descriptors": 512 * n * (k + 3) + 8 * index_bytes,
            "all priors posteriors and factors": 2048 * len(models) * k**2,
            "complete joint query and output": 512 * query_elements,
            "exact joint draws and replay": 128 * draws * (k + 2 * queries + 3),
        },
        budget_bytes=min(spec["max_bytes"], workspace_budget_bytes()),
    ).record() | {"estimated_work": work}


def keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        fail(f"{name} has an invalid complete schema.")


def array(
    value, shape, name, *, nullable=False, items_nullable=False, booleans=False, integers=False
):
    if value is None and nullable:
        return
    if not shape:
        if booleans:
            if type(value) is not bool:
                fail(f"{name} requires Boolean primitives.")
        elif integers:
            if type(value) is not int:
                fail(f"{name} requires integer primitives.")
        elif value is None and nullable:
            return
        else:
            real(value, name)
        return
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        fail(f"{name} dimensions disagree before numerical replay.")
    for row in value:
        array(
            row,
            shape[1:],
            name,
            nullable=items_nullable and len(shape) == 1,
            items_nullable=items_nullable,
            booleans=booleans,
            integers=integers,
        )


def same(saved, expected, name):
    """Schema/type-aware component-relative comparison; structural zeros remain exact."""
    if isinstance(expected, Mapping):
        keys(saved, expected, name)
        for key in expected:
            same(saved[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, (tuple, list)):
        if not isinstance(saved, (tuple, list)) or len(saved) != len(expected):
            fail(f"{name} shape disagrees.")
        for i, (a, b) in enumerate(zip(saved, expected, strict=True)):
            same(a, b, f"{name}[{i}]")
    elif type(expected) in (str, bool) or expected is None or type(expected) is int:
        if type(saved) is not type(expected) or saved != expected:
            fail(f"{name} primitive disagrees.")
    else:
        a, b = real(saved, name), float(expected)
        unit = max(abs(a), abs(b))
        if (b == 0 and a != 0) or (unit and abs(a / unit - b / unit) > 2e-10):
            fail(f"{name} differs from complete analytic replay.")


def digest(value):
    _json_export_admission(value, limit=JSON_LIMIT, operation="BMA complete digest encoding")
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


def load(value, schema=None):
    if isinstance(value, (str, bytes, bytearray)):
        _encoded_json_admission(value, limit=JSON_LIMIT, operation="BMA public JSON parser")
        try:
            value = json.loads(value)
        except (ValueError, UnicodeError) as exc:
            fail(f"Invalid BMA JSON: {exc}")
    _state_copy_admission(value, operation="BMA complete input metadata")
    if isinstance(value, BaseModel):
        value = value.__dict__
    if not isinstance(value, Mapping):
        fail("Supply a complete BMA typed state, mapping or JSON string.")
    if "payload" in value:
        keys(value, ("schema_version", "payload"), "typed BMA outer state")
        if schema is not None and value["schema_version"] != schema:
            fail("Typed BMA outer schema is unsupported.")
        value = value["payload"]
    if not isinstance(value, Mapping):
        fail("The BMA payload must be a complete mapping.")
    if schema is not None and value.get("schema") != schema:
        fail("Unsupported BMA payload schema.")
    return value


def dtype_admission(dtype):
    try:
        dt = pd.api.types.pandas_dtype(dtype)
    except (TypeError, ValueError) as exc:
        fail(f"Unsupported numeric dtype: {exc}", "unsupported_dtype")
    numpy_dtype = getattr(dt, "numpy_dtype", dt)
    if (
        not is_numeric_dtype(dt)
        or is_bool_dtype(dt)
        or is_complex_dtype(dt)
        or getattr(numpy_dtype, "itemsize", 9) > 8
    ):
        fail(
            "BMA requires real numeric float64-compatible dtypes, excluding bool/complex/extended floats.",
            "unsupported_dtype",
        )
    return dt


def resident(data, columns):
    if not isinstance(data, pd.DataFrame):
        fail("BMA requires a resident numeric DataFrame.", "unsupported_data")
    if data.columns.has_duplicates:
        fail("Input columns must be unique.", "duplicate_columns")
    n = integer(len(data), "rows", 1, MAX_ROWS)
    if any(v not in data for v in columns):
        fail("A declared BMA column is absent.", "missing_columns")
    for name in columns:
        dtype_admission(data[name].dtype)
    return n


def encode_index(index):
    result = _encode_index(index)
    if isinstance(index, pd.MultiIndex):
        result["sortorder"] = index.sortorder
        result["levels"] = [encode_index(level) for level in index.levels]
    elif isinstance(index, pd.CategoricalIndex):
        result["categories"] = encode_index(index.categories)
    elif isinstance(index.dtype, pd.StringDtype):
        result["string_storage"] = index.dtype.storage
        result["string_missing"] = "pd.NA" if index.dtype.na_value is pd.NA else "nan"
    return result


class _LazyIndexArray(list):
    """Expose the exact encoded array geometry without retaining encoded children."""

    def __init__(self, values, encode):
        self._values = values
        self._encode = encode

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        for value in self._values:
            yield self._encode(value)


def _lazy_label(value):
    # A shared tuple graph expands once per occurrence in the portable codec.
    # Do not deduplicate it or eagerly expand its children during admission.
    if isinstance(value, tuple):
        if len(value) > 16:
            fail("Tuple row labels are limited to16 components.", "unsupported_index")
        return {"type": "tuple", "value": _LazyIndexArray(value, _lazy_label)}
    return _label(value)


def _lazy_index(index):
    """Stream precisely the lossless descriptor, including unused labels and names."""
    if len(index) > MAX_ROWS:
        fail("Index levels/categories may contain at most10000 entries.", "unsupported_index")
    if isinstance(index, pd.MultiIndex):
        if index.nlevels > 16:
            fail("At most16 row-index levels are supported.", "unsupported_index")
        return {
            "kind": "multi",
            "names": _LazyIndexArray(index.names, _lazy_label),
            "levels": _LazyIndexArray(index.levels, _lazy_index),
            "codes": _LazyIndexArray(index.codes, lambda code: _LazyIndexArray(code, int)),
            "sortorder": index.sortorder,
        }
    result = {"name": _lazy_label(index.name)}
    if isinstance(index, pd.RangeIndex):
        return result | {
            "kind": "range",
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
        }
    if isinstance(index, pd.CategoricalIndex):
        return result | {
            "kind": "categorical",
            "categories": _lazy_index(index.categories),
            "ordered": index.ordered,
            "codes": _LazyIndexArray(index.codes, int),
        }
    if isinstance(index, pd.DatetimeIndex):
        kind = "datetime"
    elif isinstance(index, pd.TimedeltaIndex):
        kind = "timedelta"
    elif type(index) is pd.Index:
        kind = "index"
    else:
        fail(f"Unsupported row-index type {type(index).__name__}.", "unsupported_index")
    result |= {
        "kind": kind,
        "dtype": str(index.dtype),
        "values": _LazyIndexArray(index, _lazy_label),
    }
    if kind in ("datetime", "timedelta"):
        result["freq"] = index.freqstr
    elif isinstance(index.dtype, pd.StringDtype):
        result["string_storage"] = index.dtype.storage
        result["string_missing"] = "pd.NA" if index.dtype.na_value is pd.NA else "nan"
    return result


def index_buffer_bytes(index, *, budget_bytes=DEFAULT_BYTES):
    """Admit exact expanded label geometry with O(depth) temporary storage.

    Names, nested tuple labels and all unused levels/categories are part of
    lossless sample identity. RangeIndex values stay compact. The same metadata
    walker used for complete states counts the virtual descriptor before any
    eager codec, source copy or parent posterior replay can run.
    """
    size, _ = _metadata_geometry(_lazy_index(index))
    native = int(index.memory_usage(deep=True))
    units = size + 8 * native
    plan_workspace(
        "BMA raw row-index descriptor admission",
        {"complete expanded index and simultaneous copies": 8 * units},
        budget_bytes=min(budget_bytes, workspace_budget_bytes()),
    )
    return units


def decode_index(value, n):
    try:
        _index_envelope(value, n)
        kind = value.get("kind")
        if kind == "multi":
            order = value.get("sortorder")
            if order is not None and (
                type(order) is not int or not 0 <= order <= len(value["levels"])
            ):
                fail("Invalid MultiIndex sortorder.")
            index = pd.MultiIndex(
                levels=[decode_index(v, len(_decode_index(v))) for v in value["levels"]],
                codes=[list(v) for v in value["codes"]],
                names=[_unlabel(v) for v in value["names"]],
                sortorder=order,
                verify_integrity=True,
            )
        elif kind == "categorical":
            categories = decode_index(value["categories"], len(_decode_index(value["categories"])))
            index = pd.CategoricalIndex(
                pd.Categorical.from_codes(
                    list(value["codes"]), categories=categories, ordered=value["ordered"]
                ),
                name=_unlabel(value["name"]),
            )
        elif "string_storage" in value:
            if value.get("string_missing") not in ("pd.NA", "nan"):
                fail("Invalid string-index missing metadata.")
            dtype = pd.StringDtype(
                storage=value["string_storage"],
                na_value=pd.NA if value["string_missing"] == "pd.NA" else float("nan"),
            )
            index = pd.Index(
                [_unlabel(v) for v in value["values"]],
                dtype=dtype,
                name=_unlabel(value["name"]),
                tupleize_cols=False,
            )
        else:
            index = _decode_index(value)
        if len(index) != n:
            fail("Complete source index length disagrees.")
        same(value, encode_index(index), "typed source index")
    except AnalysisError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        fail(f"Unsupported typed index: {exc}")
    return index


def source_values(data, columns):
    values = []
    dtypes = []
    for col in columns:
        dt = dtype_admission(data[col].dtype)
        dtypes.append(str(dt))
        row = []
        for value in data[col]:
            if pd.isna(value):
                row.append(None)
                continue
            value = value.item() if hasattr(value, "item") else value
            real(value, "source cell")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                fail("Invalid float64 source scalar.", "unsupported_precision")
            if isinstance(value, int) and int(float(value)) != value:
                fail(
                    "Integer source is not exactly float64 representable.", "unsupported_precision"
                )
            row.append(value)
        values.append(row)
    return values, dtypes


def capture(data, columns, missing, spec, *, query=False):
    n = resident(data, columns)
    plan(
        n if not query else spec["source_n"],
        spec,
        queries=n if query else 0,
        index_bytes=index_buffer_bytes(data.index, budget_bytes=spec["max_bytes"]),
    )
    values, dtypes = source_values(data, columns)
    positions = [i for i in range(n) if all(v[i] is not None for v in values)]
    if not positions:
        fail("The common complete-case sample is empty.", "empty_sample")
    if missing == "raise" and len(positions) != n:
        fail("Missing union-model values require missing='drop'.", "missing_data")
    source = dict(
        n=n,
        columns=columns,
        dtypes=dtypes,
        values=values,
        index=encode_index(data.index),
        positions=positions,
        missing=missing,
    )
    _state_copy_admission(
        source,
        operation="BMA captured source",
        budget_bytes=min(spec["max_bytes"], workspace_budget_bytes()),
    )
    source["digest"] = digest(source)
    return source


def source_admission(source, columns, missing):
    keys(
        source,
        ("n", "columns", "dtypes", "values", "index", "positions", "missing", "digest"),
        "complete BMA source",
    )
    n = integer(source["n"], "saved rows", 1, MAX_ROWS)
    if missing not in ("raise", "drop"):
        fail("Unsupported saved source missing policy.")
    if (
        not isinstance(source["columns"], (list, tuple))
        or list(source["columns"]) != columns
        or source["missing"] != missing
    ):
        fail("Saved union-source names/missing policy disagree.")
    array(source["values"], (len(columns), n), "source cells", items_nullable=True)
    if not isinstance(source["dtypes"], (list, tuple)) or len(source["dtypes"]) != len(columns):
        fail("Saved source dtype shape disagrees.")
    dtypes = [dtype_admission(dt) for dt in source["dtypes"]]
    # Every source primitive/dtype is checked before constructing even the
    # first Series. A later column cannot trigger a partially allocated source.
    for dtype, values in zip(dtypes, source["values"], strict=True):
        integer_dtype = is_integer_dtype(dtype)
        bits = int(dtype.itemsize) * 8
        unsigned = is_unsigned_integer_dtype(dtype)
        lower = 0 if unsigned else -(1 << (bits - 1))
        upper = (1 << bits) - 1 if unsigned else (1 << (bits - 1)) - 1
        for original in values:
            if original is None:
                if integer_dtype and not isinstance(dtype, pd.api.extensions.ExtensionDtype):
                    fail("A nonnullable integer dtype cannot restore missing source cells.")
                continue
            if isinstance(original, int) and int(float(original)) != original:
                fail("Saved integer is not exactly float64 representable.")
            if integer_dtype:
                if type(original) is not int or not lower <= original <= upper:
                    fail("Saved integer type/range disagrees with its declared dtype.")
            elif bits < 64:
                try:
                    code = "e" if bits == 16 else "f"
                    rounded = struct.unpack(code, struct.pack(code, float(original)))[0]
                except (OverflowError, struct.error):
                    fail("Saved real cell is outside its declared dtype range.")
                if rounded != original:
                    fail("Saved dtype/source precision disagrees.")
    positions = [i for i in range(n) if all(v[i] is not None for v in source["values"])]
    array(source["positions"], (len(positions),), "sample positions", integers=True)
    if (
        list(source["positions"]) != positions
        or not positions
        or (missing == "raise" and len(positions) != n)
    ):
        fail("Common physical sample binding disagrees.")
    try:
        _index_envelope(source["index"], n)
    except AnalysisError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        fail(f"Unsupported typed index: {exc}")
    for dtype, values in zip(dtypes, source["values"], strict=True):
        try:
            restored = pd.Series(values, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            fail(f"Saved numeric dtype cannot restore its cells: {exc}")
        for original, value in zip(values, restored, strict=True):
            if original is None:
                if not pd.isna(value):
                    fail("Saved missing dtype binding disagrees.")
            else:
                actual = value.item() if hasattr(value, "item") else value
                if (
                    pd.isna(actual)
                    or actual != original
                    or (is_integer_dtype(dtype) and type(original) is not int)
                ):
                    fail("Saved dtype/source precision disagrees.")
    index = decode_index(source["index"], n)
    if source["digest"] != digest({k: v for k, v in source.items() if k != "digest"}):
        fail("Complete source digest disagrees.")
    return index


def source_matrices(source, *, outcome=True):
    positions = source["positions"]
    block = torch.tensor(
        [[v[i] for v in source["values"]] for i in positions], dtype=torch.float64, device="cpu"
    )
    y = block[:, 0] if outcome else None
    x = block[:, 1:] if outcome else block
    return y, torch.cat(
        (torch.ones((len(positions), 1), dtype=torch.float64, device="cpu"), x), dim=1
    )


class TypedState(BaseModel):
    """Private reusable lifecycle; mathematical/schema validation is subclass-owned."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    payload: Any

    @classmethod
    def _replay(cls, value):
        raise NotImplementedError

    @model_validator(mode="before")
    @classmethod
    def _admission(cls, value):
        _state_copy_admission(value, operation="BMA typed validation metadata")
        return value

    @model_validator(mode="after")
    def _semantics(self):
        object.__setattr__(self, "payload", _freeze(type(self)._replay(self)))
        return self

    @field_serializer("payload")
    def _portable(self, value):
        return _thaw(value)

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        _encoded_json_admission(json_data, limit=JSON_LIMIT, operation="BMA typed JSON parser")
        return super().model_validate_json(json_data, **kwargs)

    @wraps(BaseModel.model_copy)
    def model_copy(self, *, update=None, deep=False):
        if deep:
            _state_copy_admission((self, update), operation="BMA typed deep-copy/update admission")
        return super().model_copy(update=update, deep=deep)

    def __deepcopy__(self, memo=None):
        _state_copy_admission(self, operation="BMA typed deep-copy metadata")
        type(self)._replay(self)
        return super().__deepcopy__(memo)

    @wraps(BaseModel.model_dump)
    def model_dump(self, **kwargs):
        _state_copy_admission(self, operation="BMA complete portable copy")
        type(self)._replay(self)
        return _thaw(super().model_dump(**kwargs))

    @wraps(BaseModel.model_dump_json)
    def model_dump_json(self, **kwargs):
        _json_export_admission(
            self,
            indent=kwargs.get("indent"),
            limit=JSON_LIMIT,
            operation="BMA complete formatted output",
        )
        type(self)._replay(self)
        return super().model_dump_json(**kwargs)
