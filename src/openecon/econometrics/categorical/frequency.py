"""Bounded physical-row admission and integer-replication geometry.

Frequency counts describe repeated observations, never precision or survey
weights. All numerical buffers scale with physical rows, not their count sum.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from .optimal import _key, _normalize_categories, _pava, _seal
from .scaling import _atom as _scalar_atom, _save

DT = torch.float64
BYTES = 128 * 1024**2
WORK = 300_000_000
MAX_ROWS = 3000
MAX_TOTAL = 1_000_000_000


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _atom(value):
    if hasattr(value, "ndim") and value.ndim != 0:
        _error("Categories must be bounded finite scalar labels.", "invalid_data")
    try:
        return _scalar_atom(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_data", "Categories must be bounded finite scalar labels.") from exc


def _admit(operation, size, work, max_bytes, max_work):
    for value, name, ceiling in ((max_bytes, "max_bytes", BYTES), (max_work, "max_work", WORK)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= ceiling:
            _error(f"{name} must be an integer from 1 to {ceiling}.")
    if work > max_work:
        _error(f"Declared {operation} work {work} exceeds max_work.", "resource_limit")
    plan = plan_workspace(operation, {"physical_rows_and_numerical_workspace": int(size)},
                          budget_bytes=min(int(max_bytes), workspace_budget_bytes()))
    return plan.record()


def _sample(data, names, frequency, missing, max_bytes, max_work, *, operation, min_rows=4):
    """Validate every supplied count before dropping any feature-missing row.

    Zero rows are excluded before inspecting their feature values. Missing count
    rows are dropped only under missing='drop'. All remaining feature missingness
    follows the same policy. Duplicate/nonunique DataFrame indexes are harmless:
    sample positions are always physical integer offsets.
    """
    if not isinstance(data, pd.DataFrame):
        _error("Frequency categorical methods require a pandas DataFrame.")
    if (not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 13
            or any(not isinstance(name, str) or not 1 <= len(name) <= 128 for name in names)
            or len(set(names)) != len(names)):
        _error("Required variables must be 1–13 distinct bounded column names.")
    if (not isinstance(frequency, str) or not 1 <= len(frequency) <= 128
            or frequency in names):
        _error("frequency must name a distinct count column.")
    if missing not in ("drop", "raise"):
        _error("missing must be 'drop' or 'raise'.")
    n = len(data)
    if not min_rows <= n <= MAX_ROWS:
        _error(f"Physical input rows must be from {min_rows} to {MAX_ROWS}.", "resource_limit")
    columns = list(names) + [frequency]
    for name in columns:
        if list(data.columns).count(name) != 1:
            _error(f"Column {name!r} must occur exactly once.")
    plan = _admit(operation + " input", 256*n*(len(columns)+1), 8*n*len(columns), max_bytes, max_work)
    counts, count_missing, zero = [], [], []
    total = 0
    for position, value in enumerate(data[frequency].tolist()):
        # Scalar missingness only: a list/array/object must never be coerced into
        # an integer or trigger an ambiguous truth value during validation.
        if value is None or value is pd.NA or (isinstance(value, Real) and not isinstance(value, Integral) and math.isnan(value)):
            if missing == "raise":
                _error("Frequency column contains missing counts.", "missing_data")
            counts.append(None)
            count_missing.append(position)
            continue
        if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, Real):
            _error("Frequency counts must be finite nonnegative exact integers.", "invalid_weights")
        if isinstance(value, Integral):
            valid = 0 <= value <= MAX_TOTAL
        else:
            valid = math.isfinite(value) and 0 <= value <= MAX_TOTAL and value == math.floor(value)
        if not valid:
            _error(f"Frequency counts must be exact integers from 0 to {MAX_TOTAL}.", "invalid_weights")
        count = int(value)
        total += count
        if total > MAX_TOTAL:
            _error(f"The supplied frequency total exceeds {MAX_TOTAL}.", "resource_limit")
        counts.append(count)
        if count == 0:
            zero.append(position)
    positive = [i for i, count in enumerate(counts) if count is not None and count > 0]
    # Project columns first: row-taking a wide caller frame would allocate
    # unrelated columns outside the admitted required-variable workspace.
    selected = data.loc[:, list(names)]
    frame = selected.iloc[positive]
    missing_mask = frame.isna().any(axis=1).tolist()
    missing_positions = count_missing + [i for i, absent in zip(positive, missing_mask) if absent]
    if missing == "raise" and any(missing_mask):
        _error("Required variables contain missing values.", "missing_data")
    positions = [i for i, absent in zip(positive, missing_mask) if not absent]
    if len(positions) < min_rows:
        _error(f"At least {min_rows} positive-frequency complete physical rows are required.", "insufficient_sample")
    retained = [counts[i] for i in positions]
    return dict(frame=selected.iloc[positions].reset_index(drop=True),
                positions=positions, input_nobs=n, counts=retained,
                frequency_total=sum(retained), zero_positions=zero,
                missing_positions=sorted(missing_positions), workspace=plan)


def _weights(counts):
    return torch.tensor(counts, dtype=DT)


def _standardize(vector, w):
    mean = (vector*w).sum()/w.sum()
    sd = torch.sqrt(((vector-mean).square()*w).sum()/w.sum())
    if not torch.isfinite(vector).all() or float(sd) <= 1e-12*max(float(vector.abs().max()), 1.):
        _error("Weighted numeric transform is nonfinite or constant.", "degenerate_transform")
    return (vector-mean)/sd, float(mean), float(sd)


def _means(target, codes, levels, w):
    counts = torch.bincount(codes, weights=w, minlength=levels)
    if bool((counts <= 0).any()):
        _error("Every retained category must have positive frequency.", "degenerate_transform")
    shape = (levels,) + tuple(target.shape[1:])
    sums = torch.zeros(shape, dtype=DT)
    divisor = counts if target.ndim == 1 else counts[:, None]
    weighted = target*w if target.ndim == 1 else target*w[:, None]
    sums.index_add_(0, codes, weighted)
    return sums/divisor, counts


def _prepare(frame, variables, scales, orders, default, w):
    """Build weighted optimal-scaling descriptors after full caller admission."""
    scales = {name: default for name in variables} if scales is None else scales
    if (not isinstance(scales, dict) or set(scales) != set(variables)
            or any(value not in ("nominal", "ordinal", "numeric") for value in scales.values())):
        _error("scales must assign nominal, ordinal or numeric to every variable.")
    orders = {} if orders is None else orders
    if not isinstance(orders, dict) or set(orders) != {name for name in variables if scales[name] == "ordinal"}:
        _error("orders must explicitly name exactly the ordinal variables.")
    descriptors, prepared = [], []
    for name in variables:
        if scales[name] == "numeric":
            series = frame[name]
            if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
                _error("Numeric scaling requires real numeric columns.")
            raw = series.tolist()
            if any(not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or abs(value) > 1e100 for value in raw):
                _error("Numeric values must be finite reals with magnitude at most 1e100.")
            z, mean, sd = _standardize(torch.tensor(raw, dtype=DT), w)
            descriptors.append(dict(name=name, scale="numeric", mean=mean, sd=sd))
            prepared.append(z)
            continue
        observed = [_atom(value) for value in frame[name].tolist()]
        labels, seen = [], set()
        for value in observed:
            key = _key(value)
            if key not in seen:
                labels.append(value)
                seen.add(key)
                if len(labels) > 32:
                    _error("At most 32 retained levels per variable.", "resource_limit")
        if len(labels) < 2:
            _error("Categorical variables need at least two retained levels.", "degenerate_transform")
        if scales[name] == "ordinal":
            order = orders[name]
            if not isinstance(order, (list, tuple)) or len(order) != len(labels):
                _error("Ordinal order must contain exactly the retained levels.")
            labels = [_atom(value) for value in order]
            keys = [_key(value) for value in labels]
            if len(set(keys)) != len(keys) or set(keys) != seen:
                _error("Ordinal order must contain each typed retained level exactly once.")
        mapping = {_key(value): i for i, value in enumerate(labels)}
        codes = torch.tensor([mapping[_key(value)] for value in observed], dtype=torch.int64)
        category_counts = torch.bincount(codes, weights=w, minlength=len(labels)).tolist()
        descriptors.append(dict(name=name, scale=scales[name], levels=labels,
                                counts=[int(value) for value in category_counts]))
        prepared.append(codes)
    return descriptors, prepared


__all__ = ["DT", "BYTES", "WORK", "MAX_ROWS", "MAX_TOTAL", "_admit", "_atom", "_sample",
           "_weights", "_standardize", "_means", "_prepare", "_normalize_categories", "_pava",
           "_seal", "_save"]
