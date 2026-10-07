"""Replay resident tables in bounded positional blocks without a full copy.

Pandas and NumPy are optional. Already typed arrays preserve their dtypes;
Python sequences retain their original scalars rather than inferring float
columns that could round unsafe integers. One-shot, unsized inputs retain the
legacy materialization protocol; they are not resident bounded-memory inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import islice


def _guard(count, limit):
    if limit is not None and count > limit:
        raise ValueError("Line charts support at most 10,000 rows. Filter or aggregate explicitly; "
                         "gaps are never silently downsampled.")


class _Frame:
    def __init__(self, data, names, limit):
        self.data, self.columns, self.count = data, list(data.columns), len(data)
        for name in names:
            if name not in self.columns:
                raise ValueError(f"Column '{name}' was not found.")
            if self.columns.count(name) != 1:
                raise ValueError(f"Column '{name}' is duplicated. Requested chart columns must be unique.")
        _guard(self.count, limit)
        self.positions = {name: self.columns.index(name) for name in names}

    def assert_unchanged(self):
        if len(self.data) != self.count or list(self.data.columns) != self.columns:
            raise ValueError("The chart source changed between passes; reopen the source before plotting.")

    def iter_batches(self, *, columns, batch_rows):
        positions = [self.positions[name] for name in columns]
        for start in range(0, self.count, batch_rows):
            # Two explicit steps: pandas may select columns first for a tuple
            # iloc indexer. Slice rows first, then copy only this small block.
            yield self.data.iloc[start:start + batch_rows].iloc[:, positions]


class _Columns:
    def __init__(self, values, count):
        self.values, self.columns, self.count = values, list(values), count

    def assert_unchanged(self):
        if any(len(value) != self.count for value in self.values.values()):
            raise ValueError("The chart source changed between passes; reopen the source before plotting.")

    def iter_batches(self, *, columns, batch_rows):
        for start in range(0, self.count, batch_rows):
            block = {}
            for name in columns:
                value = self.values[name]
                block[name] = (value.iloc[start:start + batch_rows] if hasattr(value, "iloc")
                               else value[start:start + batch_rows])
            # Preserve typed array dtypes and positional rows. Never infer a
            # float dtype from Python [unsafe_integer, None, ...] sequences.
            if all(hasattr(value, "dtype") for value in block.values()):
                try:
                    import numpy as np
                    import pandas as pd
                except ImportError:
                    pass
                else:
                    if all(isinstance(value, (pd.Series, pd.Index)) or
                           (isinstance(value, np.ndarray) and value.ndim == 1)
                           for value in block.values()):
                        yield pd.DataFrame({name: value.array if isinstance(value, (pd.Series, pd.Index))
                                            else value for name, value in block.items()})
                        continue
            yield block


class _Records:
    def __init__(self, data, names, limit):
        self.data, self.columns, self.count = data, names, len(data)
        _guard(self.count, limit)
        found = set()
        for row in data:
            if not isinstance(row, Mapping):
                raise TypeError("Rows must be mappings from column names to values.")
            found.update(name for name in names if name in row)
        for name in names:
            if data and name not in found:
                raise ValueError(f"Column '{name}' was not found.")

    def assert_unchanged(self):
        if len(self.data) != self.count:
            raise ValueError("The chart source changed between passes; reopen the source before plotting.")

    def iter_batches(self, *, columns, batch_rows):
        for start in range(0, self.count, batch_rows):
            stop = min(self.count, start + batch_rows)
            yield {name: [self.data[index].get(name) for index in range(start, stop)] for name in columns}


def source(data, names, *, limit=None):
    """Adapt resident inputs, preserving the existing Dataset reduction code."""
    from . import streaming
    if streaming.is_source(data):
        return data
    names = list(dict.fromkeys(names))
    if hasattr(data, "iloc") and hasattr(data, "dtypes"):
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            if isinstance(data, pd.DataFrame):
                return _Frame(data, names, limit)
    if isinstance(data, Mapping):
        values = {}
        for name in names:
            if name not in data:
                raise ValueError(f"Column '{name}' was not found.")
            value = data[name]
            if isinstance(value, (str, bytes, Mapping)):
                raise TypeError(f"Column '{name}' must be a sequence of values.")
            values[name] = value
        if all(hasattr(value, "__len__") and hasattr(value, "__getitem__") for value in values.values()):
            counts = {len(value) for value in values.values()}
            if len(counts) != 1:
                raise ValueError("Chart columns must have equal lengths.")
            count = counts.pop()
            _guard(count, limit)
            return _Columns(values, count)
        if limit is not None:
            values = {name: list(islice(value, limit + 1)) for name, value in values.items()}
            _guard(max(map(len, values.values())), limit)
            data = values
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return _Records(data, names, limit)
    elif limit is not None and not isinstance(data, (str, bytes)) and not hasattr(data, "to_dict"):
        data = list(islice(data, limit + 1))
        _guard(len(data), limit)
    from .charts import _columns
    if limit is not None and hasattr(data, "to_dict") and hasattr(data, "__len__"):
        _guard(len(data), limit)
    values = _columns(data, names)
    count = len(values[names[0]])
    _guard(count, limit)
    return _Columns(values, count)
