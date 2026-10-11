"""Shared checks, samples, windows and kernel weights of the rdlocrand family.

The procedures follow the community ``rdlocrand`` package (Cattaneo, Titiunik
and Vazquez-Bare; R/Python release 3.0). Treatment is ``D = 1(R >= cutoff)``
and ``Rc = R - cutoff`` is the centred running variable. Windows are closed
intervals; ``rdrandinf`` and ``rdsensitivity`` compare values rounded to eight
decimals (the package's guard against representation noise), ``rdwinselect``
compares the centred values directly, exactly as the package does.

Kernel weights (``rdrandinf``): treated units use ``u = Rc/(wr - c)``, control
units ``u = Rc/(wl - c)``; triangular ``(1 - |u|) 1(|u| < 1)``, Epanechnikov
``0.75 (1 - u^2) 1(|u| < 1)``. ``rdwinselect`` uses ``u = Rc/w_right`` on both
sides with ``|u| <= 1`` and, for p = 0, replaces zero weights by the machine
epsilon so that every unit keeps a positive weight under permutation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError

FLOAT = torch.float64
EPS = torch.finfo(FLOAT).eps
KERNELS = {"uniform": "uniform", "triangular": "triangular", "epan": "epanechnikov",
           "epanechnikov": "epanechnikov"}
VCE = ("HC1", "HC2", "HC3")
# Largest number of observations a procedure accepts (sorting, ranks and the
# per-window permutation buffers are all linear in it).
MAX_ROWS = 10_000_000


# ---- argument checks -------------------------------------------------------------------


def choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise AnalysisError("invalid_option", f"{name} must be one of: {', '.join(choices)}.")
    return value


def flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError("invalid_option", f"{name} must be True or False.")
    return value


def number(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise AnalysisError("invalid_option", f"{name} must be a finite number.")
    return float(value)


def positive(value: Any, name: str, *, optional: bool = False) -> float | None:
    result = number(value, name, optional=optional)
    if result is not None and not result > 0:
        raise AnalysisError("invalid_option", f"{name} must be positive.")
    return result


def count(value: Any, name: str, *, minimum: int = 1, maximum: int | None = None,
          optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum \
            or (maximum is not None and value > maximum):
        bound = f" and at most {maximum:,}" if maximum is not None else ""
        raise AnalysisError("invalid_option", f"{name} must be an integer of at least {minimum}{bound}.")
    return int(value)


def level(value: Any, name: str) -> float:
    result = number(value, name)
    if not 0.0 < result < 1.0:
        raise AnalysisError("invalid_option", f"{name} must lie strictly between 0 and 1.")
    return result


def kernel_name(value: Any) -> str:
    if not isinstance(value, str) or value not in KERNELS:
        raise AnalysisError("invalid_option", "kernel must be 'uniform', 'triangular' or "
                            "'epanechnikov' (the package's 'epan').")
    return KERNELS[value]


def vce_name(value: Any) -> str:
    if not isinstance(value, str) or value.upper() not in VCE:
        raise AnalysisError("invalid_option", "vce must be 'HC1', 'HC2' or 'HC3'.")
    return value.upper()


def statistic_name(value: Any, allowed: Sequence[str]) -> str:
    """Canonical statistic name; the package's 'ttest' is an alias of 'diffmeans'."""
    if value == "ttest":
        value = "diffmeans"
    return choice(value, "statistic", allowed)


def order_p(value: Any) -> int:
    return count(value, "p", minimum=0, maximum=10)


def method_name(value: Any) -> str:
    return choice(value, "method", ("auto", "exact", "monte_carlo"))


def grid(values: Any, name: str) -> list[float]:
    """A sorted, duplicate-free list of finite numbers (the package's tlist/ci grid)."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not len(values):
        raise AnalysisError("invalid_option", f"{name} must be a nonempty list of numbers.")
    numbers = [number(value, name) for value in values]
    if len(numbers) > 100_000:
        raise AnalysisError("invalid_option", f"{name} may hold at most 100,000 values.")
    return sorted(set(numbers))


def column(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise AnalysisError("invalid_spec", f"{name} must be a column name.")
    return value


def columns(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) \
            or not all(isinstance(item, str) and item for item in value):
        raise AnalysisError("invalid_spec", f"{name} must be a list of column names, for example "
                            f"{name}=['age', 'income'].")
    if len(set(value)) != len(value):
        raise AnalysisError("invalid_spec", f"{name} contains duplicate column names.")
    return list(value)


# ---- samples -----------------------------------------------------------------------------


def sample(data: Any, required: Sequence[str], optional: Sequence[str] = (), *,
           missing: str, optional_missing: bool = False) -> tuple[pd.DataFrame, int]:
    """Rows complete in ``required`` and the number of rows dropped.

    ``missing='raise'`` refuses a missing value in a required column (and in an
    ``optional`` one unless ``optional_missing``); ``missing='drop'`` deletes
    rows with a missing ``required`` value. Missing values of ``optional``
    columns (window-selection covariates) are kept as NaN when
    ``optional_missing``; the caller excludes them window by window.
    """
    choice(missing, "missing", ("raise", "drop"))
    names = [*required, *optional]
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
    if len(frame) > MAX_ROWS:
        raise AnalysisError("resource_limit", f"rdlocrand procedures accept at most {MAX_ROWS:,} rows.")
    frame = frame.loc[:, names].reset_index(drop=True)
    for name in names:
        series = frame[name]
        if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
            raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
    checked = list(required) if optional_missing else names
    incomplete = frame[checked].isna().any(axis=1)
    if bool(incomplete.any()) and missing == "raise":
        raise AnalysisError("missing_values", f"{int(incomplete.sum())} observation(s) have missing "
                            f"values in {', '.join(checked)}. Pass missing='drop' to exclude them.")
    drop = frame[list(required)].isna().any(axis=1)
    dropped = int(drop.sum())
    if dropped:
        frame = frame.loc[~drop].reset_index(drop=True)
    if not len(frame):
        raise AnalysisError("empty_sample", f"No complete observations remain for {', '.join(required)}.")
    return frame, dropped


def numeric(frame: pd.DataFrame, name: str, *, allow_missing: bool = False) -> Tensor:
    """A float64 column; missing entries become NaN only when ``allow_missing``."""
    values = torch.as_tensor(frame[name].to_numpy(dtype="float64", na_value=float("nan")).copy(),
                             dtype=FLOAT)
    finite = torch.isfinite(values)
    if allow_missing:
        finite |= torch.isnan(values)
    if not bool(finite.all()):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite values.")
    present = values[~torch.isnan(values)]
    if present.numel() and float(present.abs().max()) > 1e150:
        raise AnalysisError("numerical_failure", f"Column '{name}' has values beyond 1e150 in magnitude; "
                            "rescale it before testing.")
    return values


def lexsort(keys: Sequence[Tensor]) -> Tensor:
    """Stable order by the first key, ties broken by the following keys (NaN last)."""
    n = keys[0].numel()
    order = torch.arange(n, dtype=torch.int64)
    for key in reversed(keys):
        values = key[order]
        if values.dtype.is_floating_point:
            values = torch.where(torch.isnan(values), torch.full_like(values, math.inf), values)
        order = order[torch.argsort(values, stable=True)]
    return order


def round8(values: Tensor) -> Tensor:
    return torch.round(values, decimals=8)


def round8_scalar(value: float) -> float:
    return float(round8(torch.tensor(value, dtype=FLOAT)))


def check_cutoff(r: Tensor, cutoff: float, *, strict: bool) -> None:
    low, high = float(r.min()), float(r.max())
    outside = (cutoff <= low or cutoff >= high) if strict else (cutoff < low or cutoff > high)
    if outside:
        raise AnalysisError("invalid_cutoff", f"The cutoff {cutoff:g} must lie "
                            f"{'strictly ' if strict else ''}inside the range of the running variable "
                            f"[{low:g}, {high:g}].")


# ---- kernel weights and polynomial terms ---------------------------------------------------


def _profile(u: Tensor, kernel: str, inside: Tensor) -> Tensor:
    if kernel == "triangular":
        values = 1 - u.abs()
    else:
        values = 0.75 * (1 - u.square())
    return torch.where(inside, values, torch.zeros_like(values))


def side_weights(rc: Tensor, d: Tensor, wl: float, wr: float, kernel: str) -> Tensor | None:
    """rdrandinf kernel weights with side-specific half-widths ``wr - c`` and ``wl - c``."""
    if kernel == "uniform":
        return None
    if not (wl < 0 < wr):
        raise AnalysisError("invalid_window", "Kernel weighting needs a window with positive width "
                            "on both sides of the cutoff.")
    u = torch.where(d, rc / wr, rc / wl)
    return _profile(u, kernel, u.abs() < 1)


def window_weights(rc: Tensor, w_right: float, kernel: str, p: int) -> Tensor | None:
    """rdwinselect kernel weights: ``u = Rc / w_right`` on both sides, |u| <= 1."""
    if kernel == "uniform":
        return None
    u = rc / w_right
    weights = _profile(u, kernel, u.abs() <= 1)
    if p == 0:
        weights = torch.where(weights == 0, torch.full_like(weights, EPS), weights)
    return weights


def powers(r_adj: Tensor, p: int) -> Tensor:
    """[R, R^2, ..., R^p] (the package's ``poly(R.adj, p, raw = TRUE)``)."""
    return r_adj[:, None] ** torch.arange(1, p + 1, dtype=FLOAT)


def interval_rows(accepted: list[bool], values: list[float]) -> list[tuple[float, float]]:
    """Maximal runs of accepted grid points as [first, last] intervals (the package's find_CI)."""
    rows: list[tuple[float, float]] = []
    start = None
    for index, ok in enumerate(accepted):
        if ok and start is None:
            start = index
        if not ok and start is not None:
            rows.append((values[start], values[index - 1]))
            start = None
    if start is not None:
        rows.append((values[start], values[-1]))
    return rows


REFERENCES = [
    "Cattaneo, Frandsen and Titiunik (2015), Journal of Causal Inference 3(1): 1-24.",
    "Cattaneo, Titiunik and Vazquez-Bare (2016), Stata Journal 16(2): 331-367.",
    "Cattaneo, Titiunik and Vazquez-Bare (2017), Journal of Policy Analysis and Management 36(3): 643-681.",
]
