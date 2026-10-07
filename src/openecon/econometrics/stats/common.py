"""Shared helpers of the classical-statistics family.

Sample selection (listwise deletion with a reported count), numeric and
grouping columns, O(n) group moments by ``index_add_`` and small scalar helpers
for p-values and confidence limits. pandas is used to pick columns and to
factorize group labels; every number is computed on float64 tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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

# Squares of values beyond this bound overflow float64 sums of squares; when every
# value is below the lower bound, squared deviations underflow instead.
_LARGEST = 1e150
_SMALLEST = 1e-100


def check_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError("invalid_spec", f"{what} must be the name of one column.")
    return value


def name_list(value: Any, what: str, *, minimum: int = 1, noun: str = "column name",
              example: str = "income") -> list[str]:
    """A list of distinct names (a bare string is an error).

    ``noun`` and ``example`` word the messages for lists that do not hold column
    names (post hoc methods, statistics).
    """
    if value is None:
        value = []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise AnalysisError("invalid_spec", f"{what} must be a list of {noun}s, for "
                            f"example {what}=['{example}'].")
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


def check_alpha(alpha: Any) -> float:
    if not isinstance(alpha, Real) or isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise AnalysisError("invalid_option", "alpha must be a number strictly between 0 and 1 "
                            "(0.05 gives 95% confidence intervals).")
    return float(alpha)


def check_choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise AnalysisError("invalid_option", f"{name} must be one of: {', '.join(choices)}.")
    return value


def check_flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError("invalid_option", f"{name} must be True or False.")
    return value


def check_number(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise AnalysisError("invalid_option", f"{name} must be a finite number.")
    return float(value)


def select(data: Any, names: Sequence[str], *, numeric: Sequence[str] = (),
           missing: str = "drop", listwise: bool = True) -> tuple[pd.DataFrame, int]:
    """The named columns of ``data`` and the number of rows excluded for missing values.

    ``numeric`` columns must have a numeric dtype (checked before any row is
    dropped). With ``listwise`` every row with a missing value in any named
    column is excluded (``missing='drop'``) or rejected (``missing='raise'``);
    without it the rows are returned untouched and the caller handles missing
    values variable by variable or pair by pair.
    """
    check_choice(missing, "missing", ("drop", "raise"))
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
    frame = frame.loc[:, names].reset_index(drop=True)
    for name in numeric:
        dtype = frame[name].dtype
        if not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)) or is_complex_dtype(dtype):
            raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
    absent_rows = frame.isna().any(axis=1)
    count = int(absent_rows.sum())
    if count and missing == "raise":
        raise AnalysisError("missing_values", f"{count} observation(s) have missing values in "
                            f"{', '.join(names)}. Choose missing='drop' to exclude them.")
    if not listwise:
        return frame, count
    if count:
        frame = frame.loc[~absent_rows].reset_index(drop=True)
    if not len(frame):
        raise AnalysisError("empty_sample", "No complete observations remain after excluding "
                            "missing values.")
    return frame, count


def values(frame: pd.DataFrame, name: str, *, allow_missing: bool = False, check_scale: bool = True) -> Tensor:
    """A float64 column; missing entries are NaN when ``allow_missing``, infinities are errors."""
    buffer = frame[name].to_numpy(dtype="float64", na_value=float("nan"))
    column = torch.as_tensor(buffer.copy() if not buffer.flags.writeable else buffer,
                             dtype=torch.float64)
    present = ~torch.isnan(column)
    if bool(torch.isinf(column).any()) or (not allow_missing and not bool(present.all())):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite values.")
    if bool(present.any()):
        largest = float(column[present].abs().max())
        if largest > _LARGEST:
            raise AnalysisError("non_finite_values", f"Column '{name}' has values too large for "
                                "sums of squares in double precision; rescale it.")
        if check_scale and 0.0 < largest < _SMALLEST:
            raise AnalysisError("non_finite_values", f"Every value of column '{name}' is below "
                                f"{_SMALLEST:g} in magnitude, too small for sums of squares in "
                                "double precision; rescale it (for example multiply by 1e100).")
    return column


def matrix(frame: pd.DataFrame, names: Sequence[str], *, check_scale: bool = True) -> Tensor:
    if not names:
        return torch.empty((len(frame), 0), dtype=torch.float64)
    return torch.stack([values(frame, name, check_scale=check_scale) for name in names], dim=1)


def group_codes(frame: pd.DataFrame, name: str) -> tuple[Tensor, list[Any]]:
    """Dense int64 codes 0..G-1 and the group labels in ascending order."""
    try:
        codes, labels = pd.factorize(frame[name], sort=True)
    except TypeError:
        try:
            codes, labels = pd.factorize(frame[name].astype(str), sort=True)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_groups", f"Group labels in '{name}' must be scalar "
                                "values.") from exc
    if (codes < 0).any():
        raise AnalysisError("invalid_groups", f"Group labels in '{name}' must not be missing.")
    return torch.from_numpy(codes.astype("int64")), [_json_scalar(label) for label in labels]


@dataclass
class GroupMoments:
    n: Tensor          # [G] counts (float64)
    mean: Tensor       # [G]
    ss: Tensor         # [G] within-group sums of squares about the group mean
    var: Tensor        # [G] ss / (n - 1); NaN for groups of one observation


def centre(y: Tensor) -> tuple[Tensor, float]:
    """(y - shift, shift) with shift the rounded overall mean.

    Location-invariant statistics are computed on the shifted values, so a large
    common level (1e9 plus unit noise) costs no digits; reported means add the
    shift back.
    """
    shift = float(y.mean()) if y.numel() else 0.0
    return y - shift, shift


def group_sum(x: Tensor, codes: Tensor, groups: int) -> Tensor:
    """Sums of ``x`` ([n] or [n, m]) within groups, in O(n)."""
    out = torch.zeros((groups, *x.shape[1:]), dtype=torch.float64)
    return out.index_add_(0, codes, x)


def group_moments(y: Tensor, codes: Tensor, groups: int) -> GroupMoments:
    n = group_sum(torch.ones_like(y), codes, groups)
    mean = group_sum(y, codes, groups) / n
    # A second pass on the deviations removes the rounding error of the first mean.
    dev = y - mean[codes]
    mean = mean + group_sum(dev, codes, groups) / n
    dev = y - mean[codes]
    ss = group_sum(dev.square(), codes, groups)
    var = torch.where(n > 1, ss / (n - 1).clamp_min(1.0), torch.full_like(ss, float("nan")))
    return GroupMoments(n, mean, ss, var)


def group_extremes(y: Tensor, codes: Tensor, groups: int) -> tuple[Tensor, Tensor]:
    """(minimum, maximum) of every group; +inf / -inf for an empty group."""
    low = torch.full((groups,), float("inf"), dtype=torch.float64)
    high = torch.full((groups,), float("-inf"), dtype=torch.float64)
    return (low.scatter_reduce(0, codes, y, reduce="amin"),
            high.scatter_reduce(0, codes, y, reduce="amax"))


def frame(rows: Any, *, columns: Sequence[str], index: Sequence[Any] | None = None,
          **attrs: Any) -> pd.DataFrame:
    """A result table whose numeric columns are float even when every cell is undefined.

    Undefined statistics are passed as None and stored as missing cells (blank in the
    console and in LaTeX), never as text.
    """
    repeated = sorted({str(name) for name in columns if list(columns).count(name) > 1})
    if repeated:
        raise AnalysisError("invalid_spec", f"The column name(s) {', '.join(repeated)} are also "
                            "the name of a column of the result table; rename the column(s) in "
                            "the data.")
    out = table(rows, columns=columns, index=index, **attrs)
    for name in out.columns:
        if out[name].dtype == object and all(
                value is None or (isinstance(value, Real) and not isinstance(value, bool))
                for value in out[name]):
            out[name] = out[name].astype("float64")
    return out


def finite(value: Any) -> float | None:
    """A finite float, or None for a statistic that is undefined for this sample."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def t_two_sided(statistic: float, df: float) -> float | None:
    if not (math.isfinite(statistic) and df > 0):
        return None
    return min(1.0, 2.0 * dist.t_sf(abs(statistic), df))


def t_upper(statistic: float, df: float) -> float | None:
    """P(T > statistic)."""
    if not (math.isfinite(statistic) and df > 0):
        return None
    return dist.t_sf(statistic, df)


def t_critical(alpha: float, df: float) -> float | None:
    """Two-sided critical value: the upper alpha/2 point of Student's t."""
    if not df > 0:
        return None
    return dist.t_isf(0.5 * alpha, df)


def f_upper(statistic: float | None, df1: float, df2: float) -> float | None:
    if statistic is None or not (math.isfinite(statistic) and df1 > 0 and df2 > 0):
        return None
    return dist.f_sf(max(statistic, 0.0), df1, df2)


def chi2_upper(statistic: float | None, df: float) -> float | None:
    if statistic is None or not (math.isfinite(statistic) and df > 0):
        return None
    return dist.chi2_sf(max(statistic, 0.0), df)


def negligible(ss: float, scale: float, tol: float = 1e-24) -> bool:
    """True when a sum of squares is at the rounding level of ``scale`` (a total SS).

    The default suits residuals of a least-squares fit (QR leaves about 1e-15 of
    the outcome's norm); sums of squares of plain deviations from a mean are
    exact to one unit in the last place and are tested with ``tol=1e-30``.
    """
    return not ss > tol * scale


def ratio(numerator: float, denominator: float) -> float | None:
    """numerator / denominator, or None when the denominator is zero or not finite."""
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator == 0.0:
        return None
    return numerator / denominator


def root_product(a: float, b: float) -> float:
    """sqrt(a b) for two sums of squares, safe when the product overflows or underflows.

    sqrt(a b) is exact for perfectly related integer data (r = 1 exactly), so it is
    kept whenever the product is in range.
    """
    product = a * b
    if 1e-290 < product < float("inf"):
        return math.sqrt(product)
    return math.sqrt(a) * math.sqrt(b)


def levene(y: Tensor, codes: Tensor, groups: int, centers: Tensor) -> dict[str, Any]:
    """Levene-type test: one-way ANOVA F on z = |y - center of the group|.

    W = [(N - k) / (k - 1)] * sum_i n_i (zbar_i - zbar)^2 / sum_ij (z_ij - zbar_i)^2,
    referred to F(k - 1, N - k). ``within_ss`` and ``group_df`` are kept for the
    Satterthwaite-adjusted denominator degrees of freedom of the median version.
    """
    z = (y - centers[codes]).abs()
    moments = group_moments(z, codes, groups)
    total = float(moments.n.sum())
    grand = float((moments.n * moments.mean).sum()) / total
    between = float((moments.n * (moments.mean - grand).square()).sum())
    within = float(moments.ss.sum())
    df1, df2 = groups - 1, int(round(total)) - groups
    statistic = ratio(between / df1, within / df2) if df1 > 0 and df2 > 0 else None
    adjusted = None
    if within > 0.0:
        adjusted = ratio(within ** 2, float((moments.ss.square() / (moments.n - 1)
                                             .clamp_min(1.0))[moments.n > 1].sum()))
    return {"statistic": statistic, "df1": df1, "df2": df2,
            "p_value": f_upper(statistic, df1, df2), "adjusted_df2": adjusted}


def group_medians(y: Tensor, codes: Tensor, groups: int) -> Tensor:
    """Median of every group from one sort by (group, value)."""
    order = torch.argsort(y, stable=True)
    order = order[torch.argsort(codes[order], stable=True)]
    ordered = y[order]
    counts = torch.bincount(codes, minlength=groups)
    start = torch.cumsum(counts, 0) - counts
    low = start + (counts - 1) // 2
    high = start + counts // 2
    return 0.5 * (ordered[low] + ordered[high])


def sorted_by_group(y: Tensor, codes: Tensor, groups: int) -> tuple[Tensor, Tensor, Tensor]:
    """(values sorted by group then value, group start offsets, group counts)."""
    order = torch.argsort(y, stable=True)
    order = order[torch.argsort(codes[order], stable=True)]
    counts = torch.bincount(codes, minlength=groups)
    return y[order], torch.cumsum(counts, 0) - counts, counts
