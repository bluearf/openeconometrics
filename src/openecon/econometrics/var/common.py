"""Sample handling shared by the VAR and VEC estimators.

A vector model reads K endogenous series in time order. ``ModelSpec`` has one
``outcome`` field, so the convention of this family is

    spec.outcome            = the FIRST endogenous variable,
    spec.columns["system"]  = ALL endogenous variables, in model order
                              (the first one repeats the outcome),
    spec.predictors         = exogenous regressors (``var`` only).

The variable order matters: it is the Cholesky ordering of the orthogonalized
impulse responses and the order of the Johansen normalization.

The series must be regularly spaced without gaps, because every statistic
depends on the lag structure. With a ``time`` column the rows are sorted by it
(integer periods must be consecutive; datetimes are taken as consecutive in
sorted order); without one the row order is the time order. Rows excluded by
``missing='drop'`` may only lie at the ends of the series.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from pandas.api.types import is_datetime64_any_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES
from openecon.econometrics.arima.diagnostics import require_time_column
from openecon.econometrics.core import ModelFrame, make_spec
from openecon.models import ModelSpec, ResultBundle

MAX_EQUATIONS = 50


@dataclass
class Sample:
    """The time-ordered levels of one vector specification."""

    frame: ModelFrame
    names: list[str]           # endogenous variables, in model order
    levels: Tensor             # [n, K] float64, time ordered
    last_period: int | None    # integer time value of the last observation, when available


def _gap_error(detail: str) -> AnalysisError:
    return AnalysisError("time_gaps", f"{detail} Vector autoregressions need regularly spaced "
                         "series without gaps: fill or interpolate the missing periods, or "
                         "restrict the sample to one uninterrupted stretch.")


def system_names(spec: ModelSpec) -> list[str]:
    """The endogenous variables of a spec (role ``system``), validated."""
    value = spec.columns.get("system")
    names = [] if value is None else [value] if isinstance(value, str) else list(value)
    if not names:
        raise AnalysisError("invalid_spec", "The model needs its endogenous variables in the "
                            "column role 'system' (y=[...] in the convenience function).")
    if names[0] != spec.outcome:
        raise AnalysisError("invalid_spec", "The outcome must be the first endogenous variable "
                            "of the column role 'system'.")
    if len(names) > MAX_EQUATIONS:
        raise AnalysisError("model_too_large", f"A vector model is limited to {MAX_EQUATIONS} "
                            f"endogenous variables; {len(names)} were given.")
    overlap = [name for name in names if name in set(spec.predictors)]
    if overlap:
        raise AnalysisError("invalid_spec", "A variable cannot be both endogenous and exogenous: "
                            f"{', '.join(overlap)}.")
    if spec.time is not None and spec.time in names:
        raise AnalysisError("invalid_spec", "The time column must differ from the endogenous "
                            "variables.")
    return names


def load_sample(spec: ModelSpec, data: Any) -> Sample:
    """Order the sample in time and refuse gaps; nothing is differenced or lagged here."""
    names = system_names(spec)
    frame = ModelFrame(spec, data)
    last_period = None
    if spec.time is not None:
        require_time_column(frame.original[spec.time], spec.time)
        frame.sort_panel()
        column = frame.original[spec.time]
        if frame.n < len(column):
            # Rows excluded for missing values must not lie inside the series.
            order = torch.as_tensor(column.sort_values(kind="stable", na_position="last")
                                    .index.to_numpy(copy=True))
            kept = torch.zeros(len(column), dtype=torch.bool)
            kept[torch.as_tensor(frame.positions)] = True
            where = kept[order].nonzero().flatten()
            if int(where[-1] - where[0]) + 1 != len(where):
                raise _gap_error("Observations with missing values lie inside the series.")
        index = frame.time_index()
        if not is_datetime64_any_dtype(column.dtype):
            if frame.n > 1 and bool((index[1:] - index[:-1] != 1).any()):
                raise _gap_error(f"Time column '{spec.time}' skips periods.")
            last_period = int(index[-1])
    else:
        positions = torch.as_tensor(frame.positions)
        if frame.n > 1 and bool((positions[1:] - positions[:-1] != 1).any()):
            raise _gap_error("Observations with missing values lie inside the series.")
    return Sample(frame, names, frame.matrix(names), last_period)


def check_design_size(rows: int, columns: int) -> None:
    if rows * columns * 8 > _MAX_DESIGN_BYTES:
        raise AnalysisError("design_too_large", "The lagged design matrix exceeds the 256 MiB "
                            "float64 limit. Reduce the number of lags or variables.")


def integer(value: Any) -> Any:
    """NumPy and other integral scalars as ``int``; anything else is left for the registry."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        return value


def variable_list(value: Any, name: str = "y") -> list[str]:
    """The endogenous variables of a convenience call: a non-empty list of distinct names."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise AnalysisError("invalid_spec", f"{name} must be a list of column names, for example "
                            f"{name}=['income', 'consumption'].")
    if isinstance(value, (set, frozenset, Mapping)):
        # The order is part of the model (Cholesky ordering, Johansen normalization); the
        # iteration order of a set is arbitrary and may change from run to run.
        raise AnalysisError("invalid_spec", f"{name} must be an ordered list of column names "
                            f"(the order of the variables matters), not a "
                            f"{type(value).__name__}: write {name}=['income', 'consumption'].")
    names = list(value)
    if not names:
        raise AnalysisError("invalid_spec", f"{name} must list at least one column name.")
    wrong = [item for item in names if not isinstance(item, str) or not item.strip()]
    if wrong:
        raise AnalysisError("invalid_spec", f"Every entry of {name} must be a column name given "
                            f"as non-empty text; got {', '.join(repr(item) for item in wrong[:3])}. "
                            "Rename columns whose names are not text.")
    if len(names) != len(set(names)):
        raise AnalysisError("invalid_spec", f"{name} must not repeat a column name.")
    return names


def build_spec(estimator: str, **fields: Any) -> ModelSpec:
    """``core.make_spec`` for the convenience functions: every rejection is an AnalysisError."""
    try:
        return make_spec(estimator, **fields)
    except AnalysisError:
        raise
    except ValueError as exc:
        errors = exc.errors() if hasattr(exc, "errors") else []
        message = "; ".join(
            (f"{'.'.join(str(part) for part in error.get('loc', ()))}: " if error.get("loc")
             else "") + str(error.get("msg", "")).removeprefix("Value error, ")
            for error in errors) or str(exc)
        raise AnalysisError("invalid_spec", message) from exc


def require_result(result: Any, estimators: tuple[str, ...], what: str) -> ResultBundle:
    """A fitted result of one of ``estimators`` (post-estimation functions start here)."""
    if not isinstance(result, ResultBundle) or result.spec.estimator not in estimators:
        names = " or ".join(f"oe.{name}" for name in estimators)
        raise AnalysisError("invalid_result", f"{what} needs the result returned by {names}.")
    return result


def check_steps(steps: Any, limit: int, what: str = "steps") -> int:
    steps = integer(steps)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise AnalysisError("invalid_steps", f"{what} must be a positive integer.")
    if steps > limit:
        raise AnalysisError("invalid_steps", f"{what} is limited to {limit}.")
    return steps


def check_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise AnalysisError("invalid_alpha", "alpha must lie strictly between 0 and 1.")
    return float(alpha)
