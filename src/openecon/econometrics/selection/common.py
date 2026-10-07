"""Shared helpers of the model-building family.

Argument checks, column access without copying the caller's table, listwise
missing-value masks, weights, sorted group codes and result tables. pandas is
used to pick columns and to factorize labels; every number is computed on
float64 tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame, _json_scalar
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.engines import distributions as dist

# Squares of larger values overflow float64 sums of squares; when every value of a
# column is below the lower bound, squared deviations underflow instead.
LARGEST = 1e150
SMALLEST = 1e-100
# Row blocks hold about this many float64 elements.
BLOCK_ELEMENTS = 1 << 22
WEIGHT_TYPES = ("aweight", "fweight")


# ---- argument checks ---------------------------------------------------------------


def check_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError("invalid_spec", f"{what} must be the name of one column.")
    return value


def name_list(value: Any, what: str, *, minimum: int = 0, noun: str = "column name",
              example: str = "income") -> list[str]:
    """A list of distinct names; a bare string is an error."""
    if value is None:
        value = []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise AnalysisError("invalid_spec", f"{what} must be a list of {noun}s, for example "
                            f"{what}=['{example}'].")
    names = list(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise AnalysisError("invalid_spec", f"{what} must contain {noun}s (strings).")
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise AnalysisError("invalid_spec", f"{what} lists a {noun} more than once: "
                            f"{', '.join(repeated)}.")
    if len(names) < minimum:
        raise AnalysisError("invalid_spec", f"{what} needs at least {minimum} {noun}(s).")
    return names


def check_choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise AnalysisError("invalid_option", f"{name} must be one of: {', '.join(choices)}.")
    return value


def check_flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError("invalid_option", f"{name} must be True or False.")
    return value


def check_probability(value: Any, name: str, hint: str = "") -> float:
    """A number strictly between 0 and 1 (significance levels, tolerances)."""
    if not isinstance(value, Real) or isinstance(value, bool) or not 0.0 < value < 1.0:
        raise AnalysisError("invalid_option", f"{name} must be a number strictly between 0 "
                            f"and 1{hint}.")
    return float(value)


def check_weights(weights: Any, weight_type: Any) -> tuple[str | None, str | None]:
    """(weights column, weight type); analytic weights are assumed when no type is given."""
    if weights is None:
        if weight_type is not None:
            raise AnalysisError("invalid_spec", "weight_type needs a weights column.")
        return None, None
    check_name(weights, "weights")
    if weight_type is None:
        return weights, "aweight"
    if weight_type not in WEIGHT_TYPES:
        raise AnalysisError("invalid_option", "weight_type must be 'aweight' (analytic, the "
                            "default) or 'fweight' (frequency).")
    return weights, weight_type


# ---- columns -----------------------------------------------------------------------


def source(data: Any, names: Sequence[str], *, numeric: Sequence[str] = ()) -> pd.DataFrame:
    """The caller's table after structural checks; nothing is copied.

    Every name must be present and used in one role only; ``numeric`` columns
    must have a real numeric dtype.
    """
    names = list(names)
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise AnalysisError("invalid_spec", "A column is used in more than one role: "
                            f"{', '.join(repeated)}.")
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    for name in numeric:
        dtype = frame[name].dtype
        if not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)) or is_complex_dtype(dtype):
            raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
    return frame


def as_tensor(block: Any) -> Tensor:
    """A float64 tensor of a pandas block (missing entries become NaN)."""
    buffer = block.to_numpy(dtype="float64", na_value=float("nan"))
    if not buffer.flags.writeable:
        buffer = buffer.copy()
    return torch.as_tensor(buffer, dtype=torch.float64)


def blocks(frame: pd.DataFrame, names: Sequence[str], width: int | None = None):
    """Yield ``(start, stop, float64 block)`` over row blocks of the numeric columns ``names``.

    A block holds about ``BLOCK_ELEMENTS`` numbers (``width`` overrides the
    column count used for that budget), so wide tables are never copied whole.
    Missing entries are NaN.
    """
    names = list(names)
    rows = max(1024, BLOCK_ELEMENTS // max(width or len(names), 1))
    for start in range(0, len(frame), rows):
        stop = min(len(frame), start + rows)
        yield start, stop, as_tensor(frame.iloc[start:stop][names])


def rough_means(frame: pd.DataFrame, names: Sequence[str]) -> Tensor:
    """Approximate column means from at most 4096 evenly spaced rows (missing ignored).

    Used as a fixed shift before sums of squares are accumulated: any value near
    the mean removes the cancellation of raw moments.
    """
    step = max(1, len(frame) // 4096)
    block = as_tensor(frame.iloc[::step][list(names)])
    block = torch.where(torch.isfinite(block), block, torch.full_like(block, float("nan")))
    return torch.nan_to_num(torch.nanmean(block, dim=0), nan=0.0)


def present(frame: pd.DataFrame, names: Sequence[str], numeric: Sequence[str] = ()) -> Tensor:
    """Boolean mask of the rows with no missing value in any of ``names``.

    ``numeric`` columns are scanned as float64 row blocks (one pass over the
    table, however many columns); the others through pandas.
    """
    keep = torch.ones(len(frame), dtype=torch.bool)
    fast = [name for name in names if name in set(numeric)]
    for name in names:
        if name not in set(fast):
            keep &= ~torch.as_tensor(frame[name].isna().to_numpy(dtype="bool", copy=True))
    if fast:
        for start, stop, block in blocks(frame, fast):
            keep[start:stop] &= ~torch.isnan(block).any(dim=1)
    return keep


def listwise(frame: pd.DataFrame, names: Sequence[str], missing: str,
             numeric: Sequence[str] = ()) -> tuple[Tensor, int]:
    """(mask of complete rows, number of incomplete rows) under the missing-data policy."""
    check_choice(missing, "missing", ("drop", "raise"))
    keep = present(frame, names, numeric)
    count = int((~keep).sum())
    if count and missing == "raise":
        raise AnalysisError("missing_values", f"{count} observation(s) have missing values in "
                            "the columns used. Choose missing='drop' to exclude them.")
    if count == len(frame):
        raise AnalysisError("empty_sample", "No complete observations remain after excluding "
                            "missing values.")
    return keep, count


def check_finite(values: Tensor, what: str) -> None:
    """Refuse infinities and magnitudes whose squares overflow (NaN marks a missing value)."""
    if not values.numel():
        return
    if bool(torch.isinf(values).any()):
        raise AnalysisError("non_finite_values", f"{what} contains non-finite values.")
    if float(torch.nan_to_num(values).abs().max()) > LARGEST:
        raise AnalysisError("non_finite_values", f"{what} has values too large for sums of "
                            "squares in double precision; rescale the column.")


def block_peak(block: Tensor, what: str) -> Tensor:
    """Largest magnitude of every column of a complete block, after the finiteness checks.

    One pass over the block: infinities and magnitudes above ``LARGEST`` are
    errors, exactly as in ``check_finite``.
    """
    if not block.shape[0]:
        return torch.zeros(block.shape[1], dtype=torch.float64)
    peak = block.abs().amax(dim=0)
    if not bool(torch.isfinite(peak).all()):
        raise AnalysisError("non_finite_values", f"{what} contains non-finite values.")
    if bool((peak > LARGEST).any()):
        raise AnalysisError("non_finite_values", f"{what} has values too large for sums of "
                            "squares in double precision; rescale the column.")
    return peak


def check_scale(peak: Tensor, names: Sequence[str]) -> None:
    """Refuse columns whose largest magnitude ``peak`` is nonzero but below ``SMALLEST``."""
    tiny = ((peak > 0) & (peak < SMALLEST)).nonzero().flatten().tolist()
    if tiny:
        raise AnalysisError("non_finite_values", "Every value of "
                            f"{', '.join(repr(names[i]) for i in tiny)} is below {SMALLEST:g} "
                            "in magnitude, too small for sums of squares in double precision; "
                            "rescale the column(s) (for example multiply by 1e100).")


def column(frame: pd.DataFrame, name: str, keep: Tensor | None = None, *,
           allow_missing: bool = False) -> Tensor:
    """One float64 column, optionally restricted to the rows of ``keep``."""
    values = as_tensor(frame[name])
    if keep is not None:
        values = values[keep]
    check_finite(values, f"Column '{name}'")
    if values.numel():
        check_scale(torch.nan_to_num(values).abs().max().reshape(1), [name])
    if not allow_missing and bool(torch.isnan(values).any()):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains missing values.")
    return values


def weight_column(frame: pd.DataFrame, name: str, weight_type: str, keep: Tensor) -> Tensor:
    """Validate a weight column on the rows of ``keep``; zero weights are switched off in keep.

    Returns the full-length weight vector (NaN where missing). Negative weights
    and non-integer frequency weights are errors; observations with a zero
    weight leave the sample, as in Stata.
    """
    weights = as_tensor(frame[name])
    used = weights[keep]
    check_finite(used, f"Weight column '{name}'")
    if bool((used < 0).any()):
        raise AnalysisError("negative_weights", f"{weight_type}s must be nonnegative.")
    if weight_type == "fweight" and bool((used != used.round()).any()):
        raise AnalysisError("noninteger_frequency_weights", "Frequency weights must be integers.")
    keep &= ~(weights == 0)
    if not bool(keep.any()):
        raise AnalysisError("empty_sample", "No observation has a positive weight.")
    return weights


def group_codes(series: pd.Series, keep: Tensor, name: str) -> tuple[Tensor, list[Any]]:
    """Dense int64 codes over the rows of ``keep`` (-1 elsewhere) and ascending labels.

    Only levels observed in the kept rows receive a code. Categorical columns
    are ordered by their categories, everything else by value.
    """
    try:
        codes, labels = pd.factorize(series, sort=True)
    except TypeError:
        try:
            codes, labels = pd.factorize(series.astype(str).where(series.notna()), sort=True)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_groups", f"Values of '{name}' must be scalar "
                                "labels.") from exc
    codes = torch.as_tensor(codes.astype("int64"))
    codes = torch.where(keep, codes, torch.full_like(codes, -1))
    counts = torch.bincount(codes[codes >= 0], minlength=len(labels))
    observed = counts > 0
    remap = torch.cumsum(observed.to(torch.int64), 0) - 1
    remap = torch.cat([torch.where(observed, remap, torch.full_like(remap, -1)),
                       torch.tensor([-1])])            # index -1 stays -1
    try:
        names = [_json_scalar(label) for label, used in zip(labels, observed.tolist(),
                                                           strict=True) if used]
    except AnalysisError as exc:
        raise AnalysisError("invalid_groups", f"Values of '{name}' must be finite scalar "
                            "labels.") from exc
    return remap[codes], names


# ---- results -----------------------------------------------------------------------


def finite(value: Any) -> float | None:
    """A finite float, or None for a statistic that is undefined for this sample."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def result_table(rows: Any, *, columns: Sequence[str], index: Sequence[Any] | None = None,
                 **attrs: Any) -> pd.DataFrame:
    """A result table whose numeric columns are float even when every cell is undefined."""
    repeated = sorted({str(name) for name in columns if list(columns).count(name) > 1})
    if repeated:
        raise AnalysisError("invalid_spec", f"The name(s) {', '.join(repeated)} would appear "
                            "twice in the result table; rename the column(s) in the data.")
    out = table(rows, columns=columns, index=index, **attrs)
    for name in out.columns:
        if out[name].dtype == object and all(
                value is None or (isinstance(value, Real) and not isinstance(value, bool))
                for value in out[name]):
            out[name] = out[name].astype("float64")
    return out


def f_upper(statistic: float | None, df1: float, df2: float) -> float | None:
    """P(F(df1, df2) > statistic); 0 for an infinite statistic, None when undefined."""
    if statistic is None or math.isnan(statistic) or not (df1 > 0 and df2 > 0):
        return None
    if math.isinf(statistic):
        return 0.0
    return dist.f_sf(max(statistic, 0.0), df1, df2)


def t_two_sided(statistic: float | None, df: float) -> float | None:
    if statistic is None or not (math.isfinite(statistic) and df > 0):
        return None
    return min(1.0, 2.0 * dist.t_sf(abs(statistic), df))
