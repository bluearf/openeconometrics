"""Time-aware lag histories evaluated before estimation-sample selection."""
from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.streaming_design import numeric_values

_HISTORY_BYTES = 24 * 1024 * 1024
_BATCH_BYTES = 8 * 1024 * 1024


def _periods(frame, time, *, allow_missing):
    if time not in frame:
        raise AnalysisError("missing_columns", f"Missing time column: {time}.")
    series = frame[time]
    valid = ~series.isna()
    if not allow_missing and not valid.all():
        raise AnalysisError("missing_values", "Time contains missing periods.")
    observed = series.loc[valid]
    if getattr(observed.dtype, "kind", None) in {"i", "u"} and not observed.between(-(2**53), 2**53).all():
        raise AnalysisError("invalid_time", "Time exceeds exact float64 integer precision.")
    periods = numeric_values(observed, time)
    if bool((periods != periods.round()).any()) or bool((periods.abs() > 2**53).any()):
        raise AnalysisError("invalid_time", "Time must contain exactly represented integer periods.")
    if observed.duplicated().any():
        raise AnalysisError("invalid_time", "Time periods must be unique for lag operators.")
    return valid, pd.Index(periods.to(torch.int64).tolist())


def _values(series, name):
    valid = ~series.isna()
    values = torch.full((len(series),), math.nan, dtype=torch.float64)
    values[torch.tensor(valid.tolist(), dtype=torch.bool)] = numeric_values(series.loc[valid], name)
    return values.numpy()


def lag_values(frame, time, column, lag, *, allow_missing=False):
    valid, periods = _periods(frame, time, allow_missing=allow_missing)
    values = pd.Series(_values(frame.loc[valid, column], column), index=periods)
    previous = values.reindex(periods - lag).to_numpy(dtype="float64")
    result = torch.full((len(frame),), math.nan, dtype=torch.float64)
    result[torch.tensor(valid.tolist(), dtype=torch.bool)] = torch.tensor(previous, dtype=torch.float64)
    return result


def lag_source(source: Dataset, design, *, batch_rows=65536, columns=None):
    """Return a fresh, bounded raw-history adapter for every source pass.

    Streamed lagged models require ascending unique integer time. Missing
    outcomes and zero weights still contribute predictor history, as required
    by time operators. Gaps produce missing lags rather than row offsets.
    """
    if not design.lag_specs:
        return source
    columns = list(dict.fromkeys(columns or source.columns))
    absent = set([*design.required, design.time]) - set(columns)
    if absent:
        raise AnalysisError("missing_columns", f"Lag source needs columns: {', '.join(sorted(absent))}.")
    if set(design.lag_columns).intersection(source.columns):
        raise AnalysisError("reserved_columns", "Data contains reserved lag-history columns.")
    names = list(dict.fromkeys(name for name, _ in design.lag_specs))
    maximum = max(lag for _, lag in design.lag_specs)
    if maximum > 2**53:
        raise AnalysisError("invalid_formula", "The lag exceeds exact integer time precision.")
    rows = max(1, min(batch_rows, _BATCH_BYTES // (8 * max(1, len(columns) + len(names) + len(design.lag_columns)) * 8)))

    def factory():
        source.assert_unchanged()
        history = pd.DataFrame(columns=names, dtype="float64")
        last = None
        iterator = source.iter_batches(columns, batch_rows=rows)
        try:
            for frame in iterator:
                valid, periods = _periods(frame, design.time, allow_missing=True)
                if len(periods) and (not periods.is_monotonic_increasing or last is not None and periods[0] <= last):
                    raise AnalysisError("invalid_time", "Streamed lag models need ascending unique time periods.")
                current = pd.DataFrame({name: _values(frame.loc[valid, name], name) for name in names}, index=periods)
                if (len(history) + len(current)) * 8 * (len(names) + 1) > _HISTORY_BYTES:
                    raise AnalysisError("lag_history_limit", "The lag window exceeds the bounded history budget; use a shorter lag or coarser time unit.")
                pool = pd.concat([history, current], axis=0)
                output = frame.copy(deep=False)
                for (name, lag), label in design.lag_specs.items():
                    values = pd.Series(math.nan, index=range(len(frame)), dtype="float64")
                    values.loc[valid.to_numpy()] = pool[name].reindex(periods - lag).to_numpy(dtype="float64")
                    output[label] = values.to_numpy()
                if len(periods):
                    last = int(periods[-1])
                    history = pool.loc[pool.index >= last - maximum].copy()
                yield output
            source.assert_unchanged()
        finally:
            iterator.close()

    design.streaming_history = True
    wrapped = Dataset.from_batches(factory, [*columns, *design.lag_columns],
                                   row_count=source.row_count, metadata=source.metadata)
    # Preserve the underlying source identity check as well as replay digests.
    wrapped.assert_unchanged = source.assert_unchanged
    return wrapped
