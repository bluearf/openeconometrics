"""Shared helpers of the tsmodels family: time ordering, lags, specs and small tests.

Every estimator of the family works on one time series (plus regressors) in
time order. ``ordered_frame`` builds the ``ModelFrame``, sorts it by the time
column (or keeps the row order when there is none) and refuses series that are
not regularly spaced, because every lag, transformation and recursion depends
on consecutive periods. ``lagged`` builds lag and difference columns on
tensors without loops over observations.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Any

import torch
from pandas.api.types import is_datetime64_any_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import require_time_column
from openecon.econometrics.core import ModelFrame, kernel_call, make_spec
from openecon.models import ModelSpec


def build_spec(estimator: str, **fields: Any) -> ModelSpec:
    """``core.make_spec`` for a convenience function: every rejection is an AnalysisError."""
    try:
        return make_spec(estimator, **fields)
    except AnalysisError:
        raise
    except ValueError as exc:
        errors = exc.errors() if hasattr(exc, "errors") else []
        message = "; ".join(str(error.get("msg", "")).removeprefix("Value error, ")
                            for error in errors) or str(exc)
        raise AnalysisError("invalid_spec", message) from exc


def as_int(value: Any) -> Any:
    """NumPy and other integral scalars as ``int``; anything else is left for the registry."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        return value


def as_int_list(value: Any, name: str) -> list[int] | None:
    """A list of integers from a list, tuple or integer array (None passes)."""
    if value is None:
        return None
    if isinstance(value, (bool, int)) or as_int(value) is not value:
        return [as_int(value)]
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise AnalysisError("invalid_spec", f"{name} must be a list of integers.")
    items = []
    for item in value:
        if isinstance(item, bool):
            raise AnalysisError("invalid_spec", f"{name} must be a list of integers.")
        try:
            items.append(operator.index(item))
        except TypeError:
            raise AnalysisError("invalid_spec", f"{name} must be a list of integers.") from None
    return items


def gap_error(detail: str, what: str) -> AnalysisError:
    return AnalysisError("time_gaps", f"{detail} {what} needs a regularly spaced series without "
                         "gaps: fill or interpolate the missing periods, or restrict the sample to "
                         "one uninterrupted stretch.")


def ordered_frame(spec: ModelSpec, data: Any, *, what: str, require_regular: bool = True,
                  extra_columns: Sequence[str] = ()) -> tuple[ModelFrame, int | None]:
    """The estimation frame in time order and the integer time value of its last row.

    With a time column the rows are sorted by it (repeated values are an error);
    without one the row order is the time order. ``require_regular`` refuses
    skipped integer periods and rows dropped for missing values inside the
    series (missing values at the start or end only shorten it). Datetime
    columns are taken as consecutive periods in sorted order.
    """
    if spec.time is not None and spec.time == spec.outcome:
        raise AnalysisError("invalid_spec", "The time column must differ from the outcome.")
    frame = ModelFrame(spec, data, extra_columns=extra_columns)
    last_period = None
    if spec.time is not None:
        require_time_column(frame.original[spec.time], spec.time)
        frame.sort_panel()
        column = frame.original[spec.time]
        if require_regular and frame.n < len(column):
            order = torch.as_tensor(column.sort_values(kind="stable", na_position="last")
                                    .index.to_numpy(copy=True))
            kept = torch.zeros(len(column), dtype=torch.bool)
            kept[torch.as_tensor(frame.positions)] = True
            where = kept[order].nonzero().flatten()
            if int(where[-1] - where[0]) + 1 != len(where):
                raise gap_error("Observations with missing values lie inside the series.", what)
        index = frame.time_index()
        if not is_datetime64_any_dtype(column.dtype):
            if require_regular and frame.n > 1 and bool((index[1:] - index[:-1] != 1).any()):
                raise gap_error(f"Time column '{spec.time}' skips periods.", what)
            last_period = int(index[-1])
    elif require_regular:
        positions = torch.as_tensor(frame.positions)
        if frame.n > 1 and bool((positions[1:] - positions[:-1] != 1).any()):
            raise gap_error("Observations with missing values lie inside the series.", what)
    return frame, last_period


def lag(x: Tensor, k: int) -> Tensor:
    """``L^k x`` along the first dimension with NaN before the sample (k >= 0)."""
    if k == 0:
        return x
    out = torch.full_like(x, math.nan)
    if k < x.shape[0]:
        out[k:] = x[:x.shape[0] - k]
    return out


def durbin_watson(resid: Tensor) -> float:
    """sum_t (e_t - e_(t-1))^2 / sum_t e_t^2."""
    total = float(resid.square().sum())
    if not total > 0.0:
        return math.nan
    return float((resid[1:] - resid[:-1]).square().sum()) / total


def linear_restriction_test(params: Tensor, covariance: Tensor, matrix: Tensor, value: Tensor, *,
                            df_resid: float | None, label: str) -> dict[str, Any]:
    """Wald test of R b = r: chi2(q), or F(q, df_resid) = W / q when df_resid is given."""
    from openecon.engines.distributions import chi2_sf, f_sf
    from openecon.engines.linalg import wald_statistic

    difference = matrix @ params - value
    block = matrix @ covariance @ matrix.T
    statistic, rank = kernel_call(wald_statistic, difference, (block + block.T) / 2)
    if df_resid is None:
        return {"statistic": statistic, "df": rank, "p_value": chi2_sf(statistic, rank),
                "distribution": "chi2", "label": label}
    return {"statistic": statistic / rank, "df": rank, "df2": df_resid,
            "p_value": f_sf(statistic / rank, rank, df_resid), "distribution": "F", "label": label}


def require_variation(y: Tensor, name: str) -> None:
    """Refuse a constant outcome (nothing to explain; rho, SSR ratios are undefined)."""
    if y.numel() > 1 and bool((y == y[0]).all()):
        raise AnalysisError("constant_outcome", f"The outcome '{name}' is constant in the estimation "
                            "sample; there is nothing to explain.")


def perfect_fit(ssr: float, y: Tensor) -> bool:
    """True when least squares reproduces y to rounding (SSR <= 1e-24 * sum y^2)."""
    return not ssr > 1e-24 * float(y.square().sum())


def check_positive(value: Any, name: str, *, integer: bool = False, minimum: float = 0.0,
                   strict: bool = True) -> Any:
    """Validate a numeric argument of a table-returning function."""
    ok_type = (isinstance(value, int) and not isinstance(value, bool)) if integer else \
        (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
    if not ok_type or (value <= minimum if strict else value < minimum):
        kind = "an integer" if integer else "a number"
        bound = f"greater than {minimum:g}" if strict else f"at least {minimum:g}"
        raise AnalysisError("invalid_option", f"{name} must be {kind} {bound}.")
    return value
