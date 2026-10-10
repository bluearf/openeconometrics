"""Shared bounded transport, typed row identity and resident admission.

The portable source is complete. Digests detect corruption, not authenticity;
semantic replay independently checks every derived numerical field.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping
from functools import wraps

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsworkflows.sspace import _cpu_call
from openecon.resources import plan_workspace

FLOAT = torch.float64
EPS = torch.finfo(FLOAT).eps
MAX_ROWS = 20_000
MAX_JSON = 64 * 1024**2
MAX_WORK = 200_000_000_000
DEFAULT_WORK = 500_000_000
cpu_call = _cpu_call


def fail(code, message):
    raise AnalysisError(code, message)


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        fail("invalid_option", f"{name} must be an integer in {low}..{high}.")
    return value


def real(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        fail("invalid_option", f"{name} must be finite in {low}..{high}.")
    return float(value)


def keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        fail("invalid_state", f"{name} has incomplete or unknown fields.")


def sequence(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        fail("invalid_state", f"{name} has an invalid shape.")


def _mapping_children(value):
    for key, item in value.items():
        if type(key) is not str:
            fail("metadata_limit", "Metadata requires bounded string-key dictionaries.")
        yield key
        yield item


def _metadata_geometry(value):
    """Walk with O(depth) iterators; never materialize a container's children."""
    stack, nodes, size = [(iter((value,)), 0)], 0, 0
    encoded, indentation, lines = 0, 0, 0
    while stack:
        children, depth = stack[-1]
        try:
            item = next(children)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if depth > 48 or nodes > 2_000_000:
            fail("metadata_limit", "Supervised state exceeds its nesting/node envelope.")
        if isinstance(item, Mapping):
            if len(item) > 4096:
                fail("metadata_limit", "Metadata requires bounded string-key dictionaries.")
            size += 256 + 128 * len(item)
            encoded += 2 + 4 * len(item)
            indentation += (depth + 1) * len(item) + depth
            lines += len(item) + 1
            stack.append((iter(_mapping_children(item)), depth + 1))
        elif isinstance(item, (list, tuple)):
            if len(item) > MAX_ROWS * 2:
                fail("metadata_limit", "Metadata sequence exceeds the resident envelope.")
            size += 64 + 24 * len(item)
            encoded += 2 + 2 * len(item)
            indentation += (depth + 1) * len(item) + depth
            lines += len(item) + 1
            stack.append((iter(item), depth + 1))
        elif type(item) is str:
            if len(item) > 16_384:
                fail("metadata_limit", "A metadata string exceeds 16384 characters.")
            size += 64 + 4 * len(item)
            # ensure_ascii emits two six-character surrogates for non-BMP code points.
            # Twelve bytes per character also bounds every BMP/control escape.
            encoded += 2 + 12 * len(item)
        elif item is None or type(item) in (bool, int):
            size += 48
            encoded += 48
            if type(item) is int and abs(item) > 2**127:
                fail("metadata_limit", "Metadata integer exceeds its portable envelope.")
        elif type(item) is float and math.isfinite(item):
            size += 48
            encoded += 48
        else:
            fail("invalid_state", "State must contain finite primitive JSON values only.")
        if size > MAX_JSON or encoded > MAX_JSON:
            fail("metadata_limit", "Complete metadata exceeds its 64 MiB envelope.")
    plan_workspace(
        "supervised portable state admission",
        {"containers and copies": size * 5, "complete escaped JSON": encoded * 2},
    )
    return size, encoded, indentation, lines


def metadata(value):
    """Bound containers and strings before digest, copy, index or numeric work."""
    size, _, _, _ = _metadata_geometry(value)
    return size


def digest(value):
    metadata(value)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seal(value):
    value["digest"] = digest(value)
    return value


def unseal(value):
    metadata(value)
    if not isinstance(value, Mapping) or type(value.get("digest")) is not str:
        fail("invalid_state", "A complete digest-bearing state is required.")
    bare = {k: v for k, v in value.items() if k != "digest"}
    if digest(bare) != value["digest"]:
        fail("state_integrity", "Saved state digest disagrees with its contents.")


def load(value):
    if isinstance(value, Transport):
        value = value.payload
    if isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_JSON:
            fail("metadata_limit", "Encoded state exceeds 64 MiB.")
        plan_workspace("encoded supervised JSON", {"bounded encoded input": 8 * len(value)})
        if not isinstance(value, str):
            try:
                value = value.decode("utf-8")
            except UnicodeError as exc:
                fail("invalid_state", f"Saved state is not UTF-8: {exc}.")
        # Count allocation-producing punctuation outside strings before parsing.
        quoted = escaped = False
        depth = tokens = run = 0
        for char in value:
            if quoted:
                run += 1
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                if run > 100_000:
                    fail("metadata_limit", "Encoded JSON string exceeds its envelope.")
            elif char == '"':
                quoted, run = True, 0
            elif char in "[{":
                depth += 1
                tokens += 1
            elif char in "]}":
                depth -= 1
            elif char in ",:":
                tokens += 1
            if depth > 48 or tokens > 2_000_000:
                fail("metadata_limit", "Encoded JSON exceeds its nesting/node envelope.")
        plan_workspace(
            "supervised JSON decoder", {"input, objects and copies": 8 * len(value) + 512 * tokens}
        )
        try:
            value = json.loads(value)
        except (ValueError, RecursionError) as exc:
            fail("invalid_state", f"Saved state is invalid JSON: {exc}.")
    metadata(value)
    return value


def json_output(value, *, indent=None):
    if indent is not None:
        integer(indent, 0, 64, "indent")
    size, encoded, indentation, lines = _metadata_geometry(value)
    # Conservative escaped strings and per-line indentation before allocation.
    estimate = encoded + (0 if indent is None else indent * indentation + lines)
    if estimate > MAX_JSON:
        fail("metadata_limit", "Requested portable JSON formatting exceeds 64 MiB.")
    plan_workspace(
        "supervised JSON export", {"state and complete encoded output": size * 5 + estimate * 2}
    )
    output = json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)
    if len(output.encode()) > MAX_JSON:
        fail("metadata_limit", "Complete JSON exceeds its planned envelope.")
    return output


class Transport(BaseModel):
    """Typed state; every load/copy/dump replays the full semantic contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    payload: dict

    @classmethod
    def replay(cls, payload):
        raise NotImplementedError

    @model_validator(mode="before")
    @classmethod
    def admit(cls, value):
        metadata(value)
        return value

    @model_validator(mode="after")
    def valid(self):
        type(self).replay(self.payload)
        return self

    @field_serializer("payload")
    def serialize(self, value):
        type(self).replay(value)
        return value

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        return cls.model_validate(load(json_data), **kwargs)

    def model_copy(self, *, update=None, deep=False):
        if type(deep) is not bool:
            fail("invalid_option", "deep must be a boolean.")
        type(self).replay(self.payload)
        candidate = {"payload": self.payload}
        if update:
            metadata(update)
            candidate.update(update)
        if deep:
            candidate = json.loads(json_output(candidate))
        return type(self).model_validate(candidate)

    def __deepcopy__(self, memo=None):
        return self.model_copy(deep=True)

    @wraps(BaseModel.model_dump)
    def model_dump(self, **kwargs):
        # Excluding the payload skips its field serializer. Validate the whole
        # transport first, even when the caller requests a partial view.
        type(self).replay(self.payload)
        return super().model_dump(**kwargs)

    def model_dump_json(self, *, indent=None, **kwargs):
        type(self).replay(self.payload)
        # Supported portable dump is complete; partial dumps are intentionally refused.
        if kwargs:
            fail("invalid_option", "Portable state JSON supports only the indent option.")
        return json_output({"payload": self.payload}, indent=indent)


def same(saved, actual, name):
    """Type-strict, dimensionless relative replay; structural zeros stay exact."""
    if type(actual) is float:
        if (
            type(saved) is not float
            or not math.isfinite(saved)
            or abs(saved - actual) > 4096 * EPS * max(abs(saved), abs(actual))
        ):
            fail("state_replay", f"Saved {name} disagrees with numerical replay.")
    elif isinstance(actual, Mapping):
        keys(saved, actual, name)
        for key in actual:
            same(saved[key], actual[key], f"{name}.{key}")
    elif isinstance(actual, (list, tuple)):
        sequence(saved, len(actual), name)
        for i, (s, a) in enumerate(zip(saved, actual)):
            same(s, a, f"{name}[{i}]")
    elif type(saved) is not type(actual) or saved != actual:
        fail("state_replay", f"Saved {name} disagrees with its source or declared type.")


def label(value):
    if isinstance(value, tuple):
        if len(value) > 16:
            fail("unsupported_index", "Tuple row labels support at most 16 entries.")
        return {"type": "tuple", "value": [label(v) for v in value]}
    if value is pd.NaT:
        return {"type": "nat", "value": None}
    if value is pd.NA:
        return {"type": "missing", "value": None}
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {"type": "timedelta", "value": pd.Timedelta(value).value}
    if isinstance(value, dt.date):
        return {"type": "date", "value": value.isoformat()}
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    if type(value) is float and math.isnan(value):
        return {"type": "nan", "value": None}
    if value is None or type(value) in (str, bool, int, float):
        record = {"type": "scalar", "value": value}
        metadata(record)
        return record
    fail("unsupported_index", "Row labels must be scalar, datetime, timedelta or tuple values.")


def unlabel(value):
    keys(value, ("type", "value"), "typed row label")
    kind, raw = value["type"], value["value"]
    if kind == "tuple":
        if not isinstance(raw, (list, tuple)) or len(raw) > 16:
            fail("unsupported_index", "Invalid tuple row label.")
        return tuple(unlabel(v) for v in raw)
    if kind == "scalar" and (raw is None or type(raw) in (str, bool, int, float)):
        return raw
    if kind in ("nan", "nat", "missing") and raw is None:
        return {"nan": float("nan"), "nat": pd.NaT, "missing": pd.NA}[kind]
    if kind in ("timestamp", "date") and type(raw) is str:
        return pd.Timestamp(raw) if kind == "timestamp" else dt.date.fromisoformat(raw)
    if kind == "timedelta" and type(raw) is int:
        return pd.Timedelta(raw, unit="ns")
    fail("unsupported_index", "Invalid typed row label.")


def index_record(index):
    integer(len(index), 0, MAX_ROWS, "index rows")
    if isinstance(index, pd.MultiIndex):
        integer(index.nlevels, 1, 16, "index levels")
        return {
            "kind": "multi",
            "levels": [index_record(v) for v in index.levels],
            "codes": [v.tolist() for v in index.codes],
            "names": [label(v) for v in index.names],
        }
    name = label(index.name)
    if isinstance(index, pd.RangeIndex):
        return {
            "kind": "range",
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
            "name": name,
        }
    if isinstance(index, pd.CategoricalIndex):
        return {
            "kind": "categorical",
            "categories": index_record(index.categories),
            "codes": index.codes.tolist(),
            "ordered": index.ordered,
            "name": name,
        }
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return {
            "kind": "datetime" if isinstance(index, pd.DatetimeIndex) else "timedelta",
            "values": [label(v) for v in index],
            "dtype": str(index.dtype),
            "freq": index.freqstr,
            "name": name,
        }
    if type(index) is not pd.Index:
        fail("unsupported_index", f"Unsupported row index {type(index).__name__}.")
    return {
        "kind": "index",
        "values": [label(v) for v in index],
        "dtype": str(index.dtype),
        "name": name,
    }


def restore_index(record, n):
    metadata(record)
    integer(n, 0, MAX_ROWS, "index rows")
    kind = record.get("kind") if isinstance(record, Mapping) else None
    try:
        if kind == "multi":
            keys(record, ("kind", "levels", "codes", "names"), "multi-index")
            integer(len(record["levels"]), 1, 16, "index levels")
            sequence(record["codes"], len(record["levels"]), "index codes")
            sequence(record["names"], len(record["levels"]), "index names")
            levels = []
            for raw, codes in zip(record["levels"], record["codes"]):
                sequence(codes, n, "index codes")
                if any(type(c) is not int or not -1 <= c < MAX_ROWS for c in codes):
                    fail("unsupported_index", "Invalid index codes.")
                levels.append(restore_index(raw, _index_length(raw)))
            index = pd.MultiIndex(
                levels=levels,
                codes=record["codes"],
                names=[unlabel(v) for v in record["names"]],
                verify_integrity=True,
            )
        elif kind == "range":
            keys(record, ("kind", "start", "stop", "step", "name"), "range index")
            for key in ("start", "stop", "step"):
                integer(record[key], -(2**63), 2**63 - 1, key)
            if (
                record["step"] == 0
                or len(range(record["start"], record["stop"], record["step"])) != n
            ):
                fail("unsupported_index", "Invalid range index geometry.")
            index = pd.RangeIndex(
                record["start"], record["stop"], record["step"], name=unlabel(record["name"])
            )
        elif kind == "categorical":
            keys(record, ("kind", "categories", "codes", "ordered", "name"), "categorical index")
            sequence(record["codes"], n, "category codes")
            if type(record["ordered"]) is not bool or any(
                type(c) is not int or not -1 <= c < MAX_ROWS for c in record["codes"]
            ):
                fail("unsupported_index", "Invalid categorical index codes.")
            categories = restore_index(record["categories"], _index_length(record["categories"]))
            index = pd.CategoricalIndex(
                pd.Categorical.from_codes(
                    record["codes"], categories=categories, ordered=record["ordered"]
                ),
                name=unlabel(record["name"]),
            )
        elif kind in ("index", "datetime", "timedelta"):
            keys(
                record,
                ("kind", "values", "dtype", "name", *(("freq",) if kind != "index" else ())),
                "row index",
            )
            sequence(record["values"], n, "row labels")
            if type(record["dtype"]) is not str or len(record["dtype"]) > 128:
                fail("unsupported_index", "Invalid row index dtype.")
            if (
                kind != "index"
                and record["freq"] is not None
                and (type(record["freq"]) is not str or len(record["freq"]) > 32)
            ):
                fail(
                    "unsupported_index",
                    "Calendar frequency metadata exceeds its bounded string envelope.",
                )
            values = [unlabel(v) for v in record["values"]]
            factory = {
                "index": pd.Index,
                "datetime": pd.DatetimeIndex,
                "timedelta": pd.TimedeltaIndex,
            }[kind]
            options = {"dtype": record["dtype"], "name": unlabel(record["name"])}
            options.update(
                {"tupleize_cols": False} if kind == "index" else {"freq": record["freq"]}
            )
            index = factory(values, **options)
        else:
            fail("unsupported_index", "Unknown row index kind.")
    except (ValueError, TypeError, OverflowError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        fail("unsupported_index", f"Invalid saved row index: {exc}.")
    if len(index) != n:
        fail("unsupported_index", "Saved row-index length disagrees with source.")
    same(record, index_record(index), "row index canonical encoding")
    return index


def _index_length(record):
    if record.get("kind") == "range":
        if (
            any(type(record.get(k)) is not int for k in ("start", "stop", "step"))
            or record["step"] == 0
        ):
            fail("unsupported_index", "Invalid saved range index.")
        return len(range(record["start"], record["stop"], record["step"]))
    if record.get("kind") in ("multi", "categorical"):
        codes = record.get("codes")
        return len(codes[0]) if record["kind"] == "multi" and codes else len(codes)
    return len(record.get("values", ()))


def columns(names, maximum=64):
    if (
        not isinstance(names, (list, tuple))
        or not 1 <= len(names) <= maximum
        or any(type(v) is not str or not v or len(v) > 256 for v in names)
        or len(set(names)) != len(names)
    ):
        fail("invalid_spec", f"Supply 1..{maximum} distinct bounded column names.")
    return list(names)


def probe(data, names):
    """Shape-only resident gate before any copy, collection or numerical scan."""
    if isinstance(data, pd.DataFrame):
        if data.columns.has_duplicates or any(v not in data.columns for v in names):
            fail("invalid_data", "Data has duplicate or absent selected columns.")
        return len(data)
    if isinstance(data, Mapping):
        if any(v not in data for v in names):
            fail("invalid_data", "Selected source columns are absent.")
        lengths = []
        for name in names:
            col = data[name]
            if not isinstance(col, (list, tuple, pd.Series)) and not (
                type(col).__module__ == "numpy" and getattr(col, "ndim", None) == 1
            ):
                fail(
                    "unsupported_data", "Mappings require resident one-dimensional arrays or lists."
                )
            lengths.append(len(col))
        if len(set(lengths)) != 1:
            fail("invalid_data", "Source columns have unequal lengths.")
        series = [data[v] for v in names if isinstance(data[v], pd.Series)]
        if series and any(not s.index.equals(series[0].index) for s in series):
            fail("invalid_data", "Source Series indexes must be identical.")
        return lengths[0]
    fail(
        "unsupported_data",
        "Use a resident DataFrame or sized column mapping; Dataset/generators are refused.",
    )


def resident(data, names, n):
    index = (
        data.index
        if isinstance(data, pd.DataFrame)
        else next(
            (data[v].index for v in names if isinstance(data[v], pd.Series)), pd.RangeIndex(n)
        )
    )
    identity = index_record(index)
    metadata(identity)
    frame = (
        data.loc[:, names].copy()
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame({v: data[v] for v in names})
    )
    if len(frame) != n or not frame.index.equals(index):
        fail("invalid_data", "Resident conversion changed physical row alignment.")
    return frame, identity


def numeric(series, *, missing=False):
    dtype = series.dtype
    if (
        not is_numeric_dtype(dtype)
        or is_bool_dtype(dtype)
        or is_complex_dtype(dtype)
        or getattr(dtype, "itemsize", 8) > 8
    ):
        fail("non_numeric_column", f"Column {series.name!r} requires real numeric values.")
    raw = series.tolist()
    out = []
    for value in raw:
        if value is pd.NA or value is None or isinstance(value, float) and math.isnan(value):
            if not missing:
                fail("missing_values", f"Column {series.name!r} requires complete values.")
            out.append(None)
            continue
        if hasattr(value, "item"):
            value = value.item()
        converted = float(value)
        if type(value) is int and int(converted) != value:
            fail(
                "numeric_representation",
                "Integer measurement cannot be represented exactly in float64.",
            )
        if not math.isfinite(converted) or abs(converted) > 1e100:
            fail(
                "numerical_domain",
                "Numeric values must be finite and bounded by 1e100; rescale explicitly.",
            )
        out.append(converted)
    return out


def numeric_cache(value, n, *, missing=False, bound=1e100):
    sequence(value, n, "saved numeric source")
    if any(
        not (missing and v is None)
        and (type(v) is not float or not math.isfinite(v) or abs(v) > bound)
        for v in value
    ):
        fail("invalid_state", "Saved numerical source has invalid primitive values.")


def tensor(value):
    return torch.tensor(value, dtype=FLOAT, device="cpu")
