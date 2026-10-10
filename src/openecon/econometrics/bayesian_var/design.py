"""Explicit integer calendar and exact, deliberately bounded source-index codec."""

from __future__ import annotations

import math
from numbers import Integral

import pandas as pd

from openecon.analysis_contracts import AnalysisError

from .admission import DEFAULT_MAX_BYTES, admit, digest, integer, metadata_admit, names

NUMERIC_DTYPES = frozenset(
    [f"{prefix}{bits}" for prefix in ("int", "uint", "Int", "UInt") for bits in (8, 16, 32, 64)]
    + ["float16", "float32", "float64", "Float32", "Float64"]
)
PERIOD_UNITS = frozenset(("unit", "daily", "monthly", "quarterly", "annual"))
INDEX_DTYPES = frozenset(
    f"{prefix}{bits}" for prefix in ("int", "uint") for bits in (8, 16, 32, 64)
) | {"object"}


def index_preflight(index):
    """Refuse unsupported index classes/dtypes before copying any values."""
    if type(index) is pd.RangeIndex:
        _name(index.name)
    elif type(index) is pd.DatetimeIndex:
        _name(index.name)
        if index.tz is not None or index.hasnans or index.unit not in {"s", "ms", "us", "ns"}:
            raise AnalysisError(
                "unsupported_index",
                "BVAR datetime row indices require naive complete declared units.",
            )
    elif type(index) is pd.Index and str(index.dtype) in INDEX_DTYPES:
        _name(index.name)
        for value in index:
            _label(value)
    else:
        raise AnalysisError(
            "unsupported_index", "This BVAR row-index class/dtype is outside the declared codec."
        )


def _name(value):
    if value is not None and (not isinstance(value, str) or len(value) > 1000):
        raise AnalysisError(
            "unsupported_index", "BVAR index names must be bounded strings or None."
        )
    return value


def _label(value):
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AnalysisError(
            "unsupported_index", "Plain BVAR index labels must be strings or integers."
        )
    if isinstance(value, str) and len(value) > 1000:
        raise AnalysisError("unsupported_index", "Index strings are limited to 1,000 characters.")
    if isinstance(value, int) and not -(2**127) <= value <= 2**127:
        raise AnalysisError(
            "unsupported_index", "Index integers exceed the supported bounded domain."
        )
    return value


def index_record(index):
    index_preflight(index)
    if type(index) is pd.RangeIndex:
        return {
            "kind": "range",
            "name": _name(index.name),
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
        }
    if type(index) is pd.DatetimeIndex:
        if index.tz is not None or index.hasnans:
            raise AnalysisError(
                "unsupported_index",
                "BVAR datetime row indices must be timezone-naive and complete.",
            )
        return {
            "kind": "datetime",
            "name": _name(index.name),
            "unit": index.unit,
            "freq": index.freqstr,
            "values": index.asi8.tolist(),
        }
    if type(index) is not pd.Index or str(index.dtype) not in INDEX_DTYPES:
        raise AnalysisError(
            "unsupported_index", "This BVAR row-index class/dtype is outside the declared codec."
        )
    return {
        "kind": "index",
        "name": _name(index.name),
        "dtype": str(index.dtype),
        "values": [_label(v) for v in index],
    }


def restore_index(record):
    if not isinstance(record, dict):
        raise AnalysisError("invalid_state", "Index must have its explicit descriptor.")
    kind = record.get("kind")
    expected = {
        "range": {"kind", "name", "start", "stop", "step"},
        "index": {"kind", "name", "dtype", "values"},
        "datetime": {"kind", "name", "unit", "freq", "values"},
    }
    if kind not in expected or set(record) != expected[kind]:
        raise AnalysisError(
            "invalid_state", "Index descriptor keys disagree with its declared kind."
        )
    _name(record["name"])
    if kind == "range":
        for key in ("start", "stop", "step"):
            integer(record[key], key, low=-(2**63), high=2**63 - 1)
        if not record["step"]:
            raise AnalysisError("invalid_state", "RangeIndex step cannot be zero.")
        result = pd.RangeIndex(record["start"], record["stop"], record["step"], name=record["name"])
    elif kind == "datetime":
        if record["unit"] not in {"s", "ms", "us", "ns"} or not isinstance(
            record["values"], (tuple, list)
        ):
            raise AnalysisError("invalid_state", "Datetime row-index unit/values are unsupported.")
        for v in record["values"]:
            integer(v, "datetime tick", low=-(2**63) + 1, high=2**63 - 1)
        freq = record["freq"]
        if freq is not None and (not isinstance(freq, str) or len(freq) > 100):
            raise AnalysisError("invalid_state", "Datetime frequency is invalid.")
        result = pd.DatetimeIndex(
            record["values"], dtype=f"datetime64[{record['unit']}]", freq=freq, name=record["name"]
        )
    else:
        if record["dtype"] not in INDEX_DTYPES:
            raise AnalysisError("invalid_state", "Plain row-index dtype is unsupported.")
        if not isinstance(record["values"], (tuple, list)):
            raise AnalysisError("invalid_state", "Plain row-index values must be bounded arrays.")
        result = pd.Index(
            [_label(v) for v in record["values"]], dtype=record["dtype"], name=record["name"]
        )
    if digest(index_record(result)) != digest(record):
        raise AnalysisError(
            "invalid_state", "Index constructor did not preserve its exact declared metadata."
        )
    return result


def validate_source(record, *, max_bytes=DEFAULT_MAX_BYTES):
    metadata_admit(record, max_bytes=max_bytes, operation="bvar_source_canonical_validation")
    columns = record["source_columns"]
    values = record["source_values"]
    dtypes = record["source_dtypes"]
    series, time = names(record["series"]), record["time"]
    if tuple(columns) != (time, *series) or time in series or not isinstance(time, str):
        raise AnalysisError(
            "invalid_state", "Source columns disagree with the declared series/calendar."
        )
    if len(values) != len(columns) or len(dtypes) != len(columns):
        raise AnalysisError("invalid_state", "Source dtype/value dimensions disagree.")
    n = len(values[0])
    if any(len(v) != n for v in values) or any(dt not in NUMERIC_DTYPES for dt in dtypes):
        raise AnalysisError("invalid_state", "Source column dimensions or dtypes are unsupported.")
    if record["period_unit"] not in PERIOD_UNITS:
        raise AnalysisError("invalid_state", "Calendar period unit is invalid.")
    for j, column in enumerate(values):
        for v in column:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise AnalysisError(
                    "invalid_state", "Model source contains missing/nonfinite/nonreal values."
                )
            if j == 0 and (type(v) is not int or not -(2**63) <= v < 2**63):
                raise AnalysisError(
                    "invalid_calendar", "Calendar periods must be exact signed 64-bit integers."
                )
            if j and isinstance(v, int) and int(float(v)) != v:
                raise AnalysisError(
                    "unsupported_precision",
                    "Model integers must be exactly representable in float64.",
                )
    order = tuple(sorted(range(n), key=values[0].__getitem__))
    if tuple(record["permutation"]) != order:
        raise AnalysisError(
            "invalid_state", "Saved time permutation disagrees with complete source periods."
        )
    if any(values[0][order[i]] - values[0][order[i - 1]] != 1 for i in range(1, n)):
        raise AnalysisError(
            "invalid_calendar", "Calendar periods must be distinct and consecutive."
        )
    index = restore_index(record["source_index"])
    if len(index) != n:
        raise AnalysisError("invalid_state", "Original row index and source lengths disagree.")
    # Constructor equality is an admission boundary before any numerical factorization.
    for dtype, column in zip(dtypes, values, strict=True):
        restored = pd.Series(column, dtype=dtype)
        actual = [v.item() if hasattr(v, "item") else v for v in restored]
        if digest(actual) != digest(column) or str(restored.dtype) != dtype:
            raise AnalysisError(
                "invalid_state", "Source dtype constructor changed a declared cell."
            )
    body = {k: v for k, v in record.items() if k != "source_sha256"}
    if record.get("source_sha256") != digest(body):
        raise AnalysisError("invalid_state", "Source integrity digest disagrees.")
    return index


def capture(data, series, time, period_unit, *, p, intercept, max_work, max_bytes):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported", "BVAR requires resident data; Dataset is not collected."
        )
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError(
            "unsupported_data", "Supply a resident pandas/OpenEconometrics DataFrame."
        )
    series = names(series)
    if not isinstance(time, str) or not time or len(time) > 1000 or time in series:
        raise AnalysisError("invalid_spec", "Declare a separate bounded integer time column.")
    if (
        period_unit not in PERIOD_UNITS
        or data.columns.has_duplicates
        or any(c not in data for c in (time, *series))
    ):
        raise AnalysisError(
            "invalid_spec", "Columns must be unique/present and period_unit declared."
        )
    admit(
        len(data),
        len(series),
        p,
        intercept,
        max_work=max_work,
        max_bytes=max_bytes,
    )
    if len(data) <= p:
        raise AnalysisError(
            "empty_sample", "At least one response after the initial lag rows is required."
        )
    dtypes = tuple(str(data[c].dtype) for c in (time, *series))
    if any(dt not in NUMERIC_DTYPES for dt in dtypes) or "int" not in dtypes[0].lower():
        raise AnalysisError(
            "unsupported_column", "Series must be real numeric; time must be integral."
        )
    index_preflight(data.index)
    admit(
        len(data),
        len(series),
        p,
        intercept,
        max_work=max_work,
        max_bytes=max_bytes,
        index_bytes=int(data.index.memory_usage(deep=True)),
    )
    values = tuple(
        tuple(v.item() if hasattr(v, "item") else v for v in data[c]) for c in (time, *series)
    )
    record = {
        "series": series,
        "time": time,
        "period_unit": period_unit,
        "source_columns": (time, *series),
        "source_dtypes": dtypes,
        "source_values": values,
        "source_index": index_record(data.index),
        "permutation": tuple(sorted(range(len(data)), key=values[0].__getitem__)),
    }
    record["source_sha256"] = digest(record)
    validate_source(record, max_bytes=max_bytes)
    return record


def matrices(record, p, intercept):
    import torch

    order = record["permutation"]
    levels = torch.tensor(
        [[column[i] for column in record["source_values"][1:]] for i in order],
        dtype=torch.float64,
        device="cpu",
    )
    n = len(order)
    blocks = [levels[p - lag : n - lag] for lag in range(1, p + 1)]
    if intercept:
        blocks.insert(0, torch.ones((n - p, 1), dtype=torch.float64, device="cpu"))
    return levels[p:], torch.cat(blocks, dim=1), levels[-p:]
