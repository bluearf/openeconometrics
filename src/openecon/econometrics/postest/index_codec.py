"""Lossless scalar codes for object row labels in owned prediction storage.

Primitive/extension indexes use native Arrow columns. Object indexes need
explicit scalar tags: null-first and mixed integer/string blocks cannot share
an inferred Arrow type, and converting all labels to strings changes identity.
No pickle, arbitrary object constructor, or executable payload is accepted.
"""
from __future__ import annotations

import base64
from datetime import date, datetime, time
from decimal import Decimal
import json
from uuid import UUID

import pandas as pd

from openecon.analysis_contracts import AnalysisError


def _code(value):
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["na"]
    if value is pd.NaT:
        return ["nat"]
    if type(value).__module__.split(".", 1)[0] == "numpy" and hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", value.isoformat()]
    if isinstance(value, pd.Timedelta):
        return ["timedelta", value.isoformat()]
    if isinstance(value, pd.Period):
        return ["period", str(value.ordinal), value.freqstr]
    if isinstance(value, pd.Interval):
        return ["interval", _code(value.left), _code(value.right), value.closed]
    if isinstance(value, (datetime, date, time)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, UUID):
        return ["uuid", str(value)]
    if isinstance(value, tuple):
        return ["tuple", [_code(item) for item in value]]
    raise AnalysisError("unsupported_prediction_index", "Object row labels must use supported scalar or tuple identities.")


def encode(value):
    return json.dumps(_code(value), ensure_ascii=True, separators=(",", ":"))


def _value(record):
    kind, *data = record
    if kind in {"none", "na", "nat"}:
        return {"none": None, "na": pd.NA, "nat": pd.NaT}[kind]
    if kind in {"bool", "str"}:
        return data[0]
    if kind == "int":
        return int(data[0])
    if kind == "float":
        return float.fromhex(data[0])
    if kind == "bytes":
        return base64.b64decode(data[0])
    if kind == "decimal":
        return Decimal(data[0])
    if kind == "timestamp":
        return pd.Timestamp(data[0])
    if kind == "timedelta":
        return pd.Timedelta(data[0])
    if kind == "period":
        return pd.Period(ordinal=int(data[0]), freq=data[1])
    if kind == "interval":
        return pd.Interval(_value(data[0]), _value(data[1]), closed=data[2])
    if kind in {"datetime", "date", "time"}:
        return {"datetime": datetime, "date": date, "time": time}[kind].fromisoformat(data[0])
    if kind == "uuid":
        return UUID(data[0])
    if kind == "tuple":
        return tuple(_value(item) for item in data[0])
    raise AnalysisError("invalid_prediction_output", "Owned prediction row-label metadata are invalid.")


def decode(value):
    return _value(json.loads(value))


class IndexStorage:
    """Keep row identities in private columns with a stable per-level schema."""

    def __init__(self, index, columns):
        import pyarrow as pa
        self.names = list(index.names)
        self.fields, self.types, self.dtypes, self.encoded = [], [], [], []
        occupied = set(columns)
        for level in range(index.nlevels):
            values = index.get_level_values(level)
            name = f"__openecon_index_{level}__"
            while name in occupied:
                name = "_" + name
            occupied.add(name)
            dtype = values.dtype
            tagged = (pd.api.types.is_object_dtype(dtype)
                      or isinstance(dtype, (pd.CategoricalDtype, pd.PeriodDtype, pd.IntervalDtype))
                      or (isinstance(dtype, pd.ArrowDtype)
                          and (pa.types.is_string(dtype.pyarrow_dtype)
                               or pa.types.is_large_string(dtype.pyarrow_dtype))))
            try:
                arrow_type = pa.large_string() if tagged else pa.array(pd.Series([], dtype=dtype)).type
            except (TypeError, ValueError, NotImplementedError):
                tagged, arrow_type = True, pa.large_string()
            self.fields.append(name)
            self.types.append(arrow_type)
            self.dtypes.append(dtype)
            self.encoded.append(tagged)

    def append(self, record, index):
        import pyarrow as pa
        if index.nlevels != len(self.fields) or list(index.names) != self.names:
            raise AnalysisError("unsupported_prediction_index", "Row-index levels or names changed between source blocks.")
        for level, name in enumerate(self.fields):
            values = index.get_level_values(level)
            if not self.encoded[level] and not pd.api.types.is_dtype_equal(values.dtype, self.dtypes[level]):
                raise AnalysisError("unsupported_prediction_index", "A primitive row-index type changed between source blocks.")
            content = [encode(item) for item in values] if self.encoded[level] else values
            try:
                array = pa.array(content, type=self.types[level], from_pandas=True)
            except (TypeError, ValueError, NotImplementedError) as exc:
                raise AnalysisError("unsupported_prediction_index", "Row labels cannot be stored with their declared index type.") from exc
            record = record.append_column(name, array)
        return record

    def restore(self, frame):
        levels = []
        for level, name in enumerate(self.fields):
            values = frame.pop(name)
            if self.encoded[level]:
                values = [decode(item) for item in values]
                dtype = self.dtypes[level]
                # These identities carry fitted category/frequency/boundary
                # semantics. Reuse the original immutable dtype, including
                # unused category levels/order, without discovering new levels.
                restored = pd.Index(values, dtype=dtype if isinstance(dtype, (
                    pd.CategoricalDtype, pd.PeriodDtype, pd.IntervalDtype)) else object,
                                    tupleize_cols=False)
            else:
                restored = pd.Index(values, dtype=self.dtypes[level])
            levels.append(restored)
        frame.index = (levels[0].rename(self.names[0]) if len(levels) == 1
                       else pd.MultiIndex.from_arrays(levels, names=self.names))
        return frame
