"""Bounded chart reductions over a replayable table protocol.

No OpenEconometrics, pandas or Torch dependency is required. Torch, when installed,
counts histogram bins on CPU; the independent standard-library path is exact too.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
import hashlib
import json
import math
import sys


def is_source(data):
    return callable(getattr(data, "iter_batches", None)) and hasattr(data, "columns")


class _NumericBlock(dict):
    """Optional native CPU64 columns; all projected values are still hashed."""

    def __init__(self, values, torch):
        super().__init__(values)
        self.torch = torch


def _numeric_columns(frame, names, pd):
    # Only actual numeric pandas dtypes take this path. Objects, strings,
    # Decimal, dates and complex values retain _number's exact validation.
    if any(list(frame.columns).count(name) != 1 for name in names):
        return None
    dtypes = [frame[name].dtype for name in names]
    if not all(pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
               or pd.api.types.is_bool_dtype(dtype) for dtype in dtypes):
        return None
    try:
        import torch
    except ImportError:
        return None
    columns = {}
    with torch.device("cpu"), torch.inference_mode():
        for name, dtype in zip(names, dtypes):
            series = frame[name]
            if pd.api.types.is_integer_dtype(dtype):
                if bool(((series > 2**53 - 1) | (series < -(2**53 - 1))).any()):
                    raise ValueError(f"Column '{name}' contains integers outside the browser's exact numeric range. Rescale them or use category labels.")
            values = series.to_numpy(dtype="float64", na_value=math.nan, copy=False)
            # Arrow-backed arrays can be read-only; the copy is bounded and
            # counted conservatively below, and avoids unsafe shared storage.
            if not values.flags.writeable or any(stride < 0 for stride in values.strides):
                values = values.copy()
            columns[name] = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    return _NumericBlock(columns, torch)


class Replay:
    def __init__(self, source, names):
        self.source, self.names = source, list(dict.fromkeys(names))
        columns = list(source.columns)
        for name in self.names:
            if columns.count(name) != 1:
                raise ValueError(f"Column '{name}' must exist exactly once.")
        self.baseline = None
        self.passes = 0
        self.batch_rows = min(65_536, max(1, (8 * 1024 * 1024) // (64 * len(self.names))))

    def batches(self, *, numeric=False):
        from .charts import _columns
        check = getattr(self.source, "assert_unchanged", lambda: None)
        check()
        digest, count = hashlib.sha256(), 0
        iterator = iter(self.source.iter_batches(columns=self.names, batch_rows=self.batch_rows))
        try:
            for batch in iterator:
                if isinstance(batch, Mapping) and any(hasattr(batch.get(name), "__len__")
                        and len(batch[name]) > self.batch_rows for name in self.names):
                    raise ValueError("Chart sources must honor the requested bounded batch size.")
                raw_bytes = 0
                if hasattr(batch, "memory_usage"):
                    raw_bytes = int(batch.memory_usage(index=True, deep=True).sum())
                    if raw_bytes > 16 * 1024 * 1024:
                        raise ValueError("A projected chart batch exceeds the memory budget; shorten labels or request fewer columns.")
                pandas_batch = False
                if hasattr(batch, "dtypes"):
                    try:
                        import pandas as pd
                        pandas_batch = isinstance(batch, pd.DataFrame)
                    except ImportError:
                        pass
                if pandas_batch and len(batch) > self.batch_rows:
                    raise ValueError("Chart sources must honor the requested bounded batch size.")
                values = _numeric_columns(batch, self.names, pd) if numeric and pandas_batch else None
                if values is None:
                    values = _columns(batch, self.names)
                size = len(values[self.names[0]])
                if size > self.batch_rows:
                    raise ValueError("Chart sources must honor the requested bounded batch size.")
                if isinstance(values, _NumericBlock):
                    converted_bytes = sum(2 * items.numel() * items.element_size() + sys.getsizeof(items)
                                          for items in values.values())
                else:
                    converted_bytes = sum(sys.getsizeof(items) + sum(sys.getsizeof(value) for value in items)
                                          for items in values.values())
                if raw_bytes + converted_bytes > 32 * 1024 * 1024:
                    raise ValueError("A projected chart batch exceeds the memory budget; shorten labels or request fewer columns.")
                if pandas_batch:
                    # Optional pandas vector hashing binds every projected value,
                    # including values omitted from a numeric display.
                    row_hashes = pd.util.hash_pandas_object(batch[self.names], index=False, categorize=True)
                    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
                else:
                    for row in zip(*(values[name] for name in self.names)):
                        encoded = [(type(value).__name__, repr(value)) for value in row]
                        digest.update(json.dumps(encoded, ensure_ascii=True).encode() + b"\n")
                count += size
                yield values
            check()
            identity = (count, digest.hexdigest())
            if self.baseline is not None and identity != self.baseline:
                raise ValueError("The chart source changed between passes; reopen the source before plotting.")
            self.baseline = identity
            self.passes += 1
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def verify(self):
        for _ in self.batches():
            pass

    def metadata(self, **details):
        return {"mode": "streaming", "passes": self.passes,
                "source_rows": self.baseline[0], "projected_columns": self.names,
                "reader_batch_rows": self.batch_rows, "projected_working_limit_bytes": 32 * 1024 * 1024,
                "integrity": "SHA-256 of all projected row values on every pass",
                **details}


def bounded_columns(source, names, limit):
    replay = Replay(source, names)
    columns = {name: [] for name in replay.names}
    count = 0
    for batch in replay.batches():
        count += len(batch[replay.names[0]])
        if count > limit:
            raise ValueError(f"This chart supports at most {limit:,} rows. Filter or aggregate explicitly; values and gaps are never silently downsampled.")
        for name in replay.names:
            columns[name].extend(batch[name])
    replay.verify()
    return columns, replay.metadata()


def scatter(source, x, y):
    from .charts import _number, _check_extent
    replay = Replay(source, [x, y])
    count = 0
    low = [math.inf, math.inf]
    high = [-math.inf, -math.inf]
    for batch in replay.batches(numeric=True):
        if isinstance(batch, _NumericBlock):
            torch = batch.torch
            with torch.inference_mode():
                valid = torch.isfinite(batch[x]) & torch.isfinite(batch[y])
                local = int(valid.sum())
                if local:
                    count += local
                    for index, name in enumerate((x, y)):
                        values = batch[name][valid]
                        low[index] = min(low[index], float(values.min()))
                        high[index] = max(high[index], float(values.max()))
            continue
        for a, b in zip(batch[x], batch[y]):
            xx, yy = _number(a, x), _number(b, y)
            if xx is not None and yy is not None:
                count += 1
                for index, value in enumerate((xx, yy)):
                    low[index], high[index] = min(low[index], value), max(high[index], value)
    if not count:
        raise ValueError("The scatter plot needs at least one finite numeric pair.")
    _check_extent((low[0], high[0]), x)
    _check_extent((low[1], high[1]), y)
    indices = set(range(count)) if count <= 2000 else {
        (2 * i * (count - 1) + 1999) // 3998 for i in range(2000)}
    selected = sorted(indices)
    points, position = [], 0
    for batch in replay.batches(numeric=True):
        if isinstance(batch, _NumericBlock):
            torch = batch.torch
            with torch.inference_mode():
                valid = torch.isfinite(batch[x]) & torch.isfinite(batch[y])
                xx, yy = batch[x][valid], batch[y][valid]
                end = position + len(xx)
                # Only the at-most2000 output ranks become Python values.
                first, stop = bisect_right(selected, position - 1), bisect_right(selected, end - 1)
                ranks = torch.tensor([rank - position for rank in selected[first:stop]],
                                     dtype=torch.int64, device="cpu")
                points.extend({"x": a, "y": b} for a, b in zip(xx[ranks].tolist(), yy[ranks].tolist()))
                position = end
            continue
        for a, b in zip(batch[x], batch[y]):
            xx, yy = _number(a, x), _number(b, y)
            if xx is not None and yy is not None:
                if position in indices:
                    points.append({"x": xx, "y": yy})
                position += 1
    return points, count, replay.baseline[0] - count, replay.metadata(
        sampling="Uniform ranks among finite pairs; endpoints retained", sample_limit=2000,
        extents={"x": [low[0], high[0]], "y": [low[1], high[1]]})


def histogram(source, x, bins):
    from .charts import _number
    replay = Replay(source, [x])
    count, lower, upper = 0, math.inf, -math.inf
    for batch in replay.batches(numeric=True):
        if isinstance(batch, _NumericBlock):
            torch = batch.torch
            with torch.inference_mode():
                values = batch[x][torch.isfinite(batch[x])]
                if len(values):
                    count += len(values)
                    lower, upper = min(lower, float(values.min())), max(upper, float(values.max()))
            continue
        for raw in batch[x]:
            value = _number(raw, x)
            if value is not None:
                count += 1
                lower, upper = min(lower, value), max(upper, value)
    if not count:
        raise ValueError("The histogram needs at least one finite numeric value.")
    if lower == upper:
        padding = max(abs(lower) * .01, .5)
        lower, upper = lower - padding, upper + padding
    width = (upper - lower) / bins
    if not math.isfinite(width) or width <= 0 or not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Histogram values exceed the supported numeric range; rescale them.")
    edges = [lower + index * width for index in range(bins)] + [upper]
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("Histogram bin edges collapse at this numeric precision. Center or rescale the values, or use fewer bins.")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        with torch.device("cpu"), torch.inference_mode():
            boundaries = torch.tensor(edges[1:-1], dtype=torch.float64)
    counts = [0] * bins
    for batch in replay.batches(numeric=True):
        if isinstance(batch, _NumericBlock):
            with torch.inference_mode():
                values = batch[x][torch.isfinite(batch[x])]
        else:
            values = [value for raw in batch[x] if (value := _number(raw, x)) is not None]
        if torch is None:
            for value in values:
                counts[min(bins - 1, max(0, bisect_right(edges, value) - 1))] += 1
        else:
            with torch.device("cpu"), torch.inference_mode():
                tensor = values if isinstance(batch, _NumericBlock) else torch.tensor(values, dtype=torch.float64)
                local = torch.bincount(torch.bucketize(tensor, boundaries, right=True), minlength=bins).tolist()
                counts = [left + right for left, right in zip(counts, local)]
    points = [{"x0": edges[index], "x1": edges[index + 1], "count": value}
              for index, value in enumerate(counts)]
    return points, count, replay.baseline[0] - count, replay.metadata(
        aggregation="Exact finite-value bin counts; final edge inclusive",
        kernel="torch.bucketize/bincount CPU float64" if torch is not None else "stdlib bisect")


def categorical(source, x, names, aggregate):
    from .charts import _number, _categories
    if aggregate is None:
        columns, metadata = bounded_columns(source, [x, *names], min(1000, 10000 // len(names)))
        return columns, None, None, metadata
    if not isinstance(aggregate, str) or aggregate not in {"sum", "mean", "count"}:
        raise ValueError("aggregate must be None, 'sum', 'mean' or 'count'.")
    replay = Replay(source, [x, *names])
    groups, cells, finite, missing = {}, 0, 0, 0
    for batch in replay.batches():
        for row, raw in enumerate(batch[x]):
            label = _categories([raw])[0]
            if len(label) > 2048:
                raise ValueError("Category labels exceed the bounded metadata budget; shorten labels or filter categories.")
            # Typed labels must not collide after rendering (e.g. integer1/'1').
            identity = (type(raw).__name__, str(raw))
            if label not in groups:
                if len(groups) >= 1000 or cells + len(names) > 10000:
                    raise ValueError("Categorical aggregation exceeds 1,000 categories or 10,000 values. Filter categories explicitly before plotting.")
                if sum(len(key.encode()) for key in groups) + len(label.encode()) > 2 * 1024 * 1024:
                    raise ValueError("Category labels exceed the bounded metadata budget; shorten labels or filter categories.")
                groups[label] = (identity, [[0., 0., 0] for _ in names])
                cells += len(names)
            prior, accumulators = groups[label]
            if identity != prior:
                raise ValueError("Category labels collide after conversion to text; use unambiguous string labels.")
            for index, name in enumerate(names):
                value = _number(batch[name][row], name)
                if value is None:
                    missing += 1
                    continue
                finite += 1
                state = accumulators[index]
                state[2] += 1
                if aggregate != "count":
                    total = state[0] + value
                    correction = ((state[0] - total) + value if abs(state[0]) >= abs(value)
                                  else (value - total) + state[0])
                    if not math.isfinite(total) or not math.isfinite(state[1] + correction):
                        raise ValueError("Aggregated values exceed finite numeric precision; rescale explicitly.")
                    state[0], state[1] = total, state[1] + correction
    replay.verify()
    columns = {x: list(groups), **{name: [] for name in names}}
    for _, states in groups.values():
        for name, state in zip(names, states):
            value = None if state[2] == 0 else (state[2] if aggregate == "count" else
                    math.fsum(state[:2]) / (state[2] if aggregate == "mean" else 1))
            columns[name].append(value)
    return columns, finite, missing, replay.metadata(aggregation=aggregate,
        category_limit=1000, value_limit=10000, missing="Ignored within groups; all-missing groups remain gaps")
