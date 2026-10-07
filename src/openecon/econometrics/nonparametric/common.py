"""Sample handling, ranking kernels and small helpers of the nonparametric family.

The procedures of this family are tests and tables, not model fits. Each public
function takes a table (DataFrame, mapping of columns or list of records) and
column names, and returns result tables built with ``core.table`` /
``core.TableSet`` whose ``attrs`` carry the scalar results.

Missing data: every procedure deletes rows listwise over the columns it uses
(``missing="drop"``, the SPSS / Stata behaviour, with the number of rows used
and dropped reported) or refuses them (``missing="raise"``).

Ranking: ``midranks`` sorts once (O(n log n)) and gives tied values the mean of
the ranks they span; the sizes of the tie groups come back with the ranks so
that tie corrections never need a second pass.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call
from openecon.engines.distributions import beta_inc, normal_sf

FLOAT = torch.float64


def procedure(function: Callable[..., Any]) -> Callable[..., Any]:
    """Public-procedure wrapper: kernel failures surface as ``AnalysisError``."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return kernel_call(function, *args, **kwargs)

    return wrapper


# ---- argument checks -------------------------------------------------------------------


def check_choice(value: Any, name: str, choices: Sequence[Any]) -> Any:
    if not isinstance(value, str) or value not in choices:
        raise AnalysisError("invalid_option", f"{name} must be one of: "
                            f"{', '.join(map(str, choices))}.")
    return value


def check_flag(value: Any, name: str, *, optional: bool = False) -> bool | None:
    if value is None and optional:
        return None
    if not isinstance(value, bool):
        raise AnalysisError("invalid_option", f"{name} must be True or False"
                            f"{' (or None for the default rule)' if optional else ''}.")
    return value


def check_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0.0 < float(alpha) < 1.0:
        raise AnalysisError("invalid_option", "alpha must be a number strictly between 0 and 1 "
                            "(0.05 gives 95% confidence intervals).")
    return float(alpha)


def check_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise AnalysisError("invalid_option", f"{name} must be a finite number.")
    return float(value)


def check_probability(value: Any, name: str) -> float:
    number = check_number(value, name)
    if not 0.0 < number < 1.0:
        raise AnalysisError("invalid_option", f"{name} must lie strictly between 0 and 1.")
    return number


def check_count(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise AnalysisError("invalid_option", f"{name} must be an integer of at least {minimum}.")
    return int(value)


def column_names(value: Any, name: str, *, minimum: int = 1) -> list[str]:
    """A list of distinct column names (a bare string is an error)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) \
            or not all(isinstance(item, str) for item in value):
        raise AnalysisError("invalid_spec", f"{name} must be a list of column names, for "
                            f"example {name}=['before', 'after'].")
    if len(set(value)) != len(value):
        raise AnalysisError("invalid_spec", f"{name} contains duplicate column names.")
    if len(value) < minimum:
        raise AnalysisError("invalid_spec", f"{name} must name at least {minimum} column(s).")
    return list(value)


# ---- samples ---------------------------------------------------------------------------


def select(data: Any, names: Sequence[Any], *, missing: str = "drop",
           roles: bool = True) -> tuple[pd.DataFrame, int]:
    """Rows complete in ``names`` (listwise deletion) and the number of rows dropped.

    ``missing="raise"`` refuses incomplete rows instead. A column may not be
    used in two roles unless ``roles`` is False.
    """
    check_choice(missing, "missing", ("drop", "raise"))
    for name in names:
        if not isinstance(name, str):
            raise AnalysisError("invalid_spec", "Column names must be strings.")
    names = list(names)
    if roles and len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise AnalysisError("invalid_spec", "A column is used in more than one role: "
                            f"{', '.join(repeated)}.")
    names = list(dict.fromkeys(names))
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    frame = frame.loc[:, names].reset_index(drop=True)
    incomplete = frame.isna().any(axis=1)
    dropped = int(incomplete.sum())
    if dropped and missing == "raise":
        raise AnalysisError("missing_values", f"{dropped} observation(s) have missing values in "
                            f"{', '.join(names)}. Pass missing='drop' to exclude them.")
    if dropped:
        frame = frame.loc[~incomplete].reset_index(drop=True)
    if not len(frame):
        raise AnalysisError("empty_sample", "No complete observations remain for "
                            f"{', '.join(names)}.")
    return frame, dropped


# Differences and squares of values beyond this magnitude overflow float64.
LARGEST = 1e150


def numeric(frame: pd.DataFrame, name: str) -> Tensor:
    """A numeric column as a float64 tensor (``non_numeric_column`` otherwise).

    Values beyond 1e150 in magnitude are refused: their differences and
    squares overflow, and every procedure here is invariant to the unit of
    measurement, so rescaling loses nothing.
    """
    series = frame[name]
    if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
        raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric for this "
                            f"procedure, but it holds {series.dtype} values. Convert it to "
                            "numbers, or use a procedure for categories (oe.crosstab, "
                            "oe.tabulate, oe.chi2gof).")
    values = _numeric(series, name).contiguous()
    if values.numel() and float(values.abs().max()) > LARGEST:
        raise AnalysisError("numerical_failure", f"Column '{name}' has values of magnitude "
                            f"{float(values.abs().max()):.1e}, too large for float64 "
                            "arithmetic. Rescale the column (the tests do not depend on its "
                            "unit).")
    return values


def label(value: Any) -> Any:
    """A JSON-safe category label: numbers stay numbers, everything else is text."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) and abs(value) < 2 ** 53 else value
    return str(value)


def group_codes(series: pd.Series) -> tuple[Tensor, list[Any]]:
    """int64 codes 0..k-1 and the sorted distinct levels of a grouping column.

    Categorical columns keep their declared category order (unused categories
    are dropped); other columns are sorted ascending, or by their text when the
    values are not mutually comparable.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.cat.remove_unused_categories()
        codes, levels = series.cat.codes.to_numpy(dtype="int64"), list(series.cat.categories)
    else:
        try:
            codes, uniques = pd.factorize(series, sort=True)
        except TypeError:
            codes, uniques = pd.factorize(series.astype(str), sort=True)
        levels = list(uniques)
    return torch.as_tensor(codes, dtype=torch.int64), [label(level) for level in levels]


def two_groups(frame: pd.DataFrame, by: str, what: str) -> tuple[Tensor, list[Any]]:
    codes, levels = group_codes(frame[by])
    if len(levels) != 2:
        raise AnalysisError("invalid_groups", f"{what} compares exactly two groups, but column "
                            f"'{by}' has {len(levels)} distinct value(s). Restrict the sample to "
                            "two groups or use a k-sample test.")
    return codes, levels


def binary_codes(frame: pd.DataFrame, name: str, positive: Any = None) -> tuple[Tensor, list[Any]]:
    """0/1 codes of a two-valued column; ``positive`` names the level coded 1.

    Without ``positive`` the larger of the two sorted levels is the event
    (1 for a 0/1 column, True for booleans). A column with one distinct value
    is accepted only when that value is 0 or 1 or ``positive`` is given.
    """
    codes, levels = group_codes(frame[name])
    if positive is not None:
        positive = label(positive)
        if positive not in levels:
            raise AnalysisError("invalid_option", f"positive={positive!r} does not occur in "
                                f"column '{name}' (values: {', '.join(map(str, levels[:10]))}).")
        if len(levels) > 2:
            return (codes == levels.index(positive)).to(torch.int64), ["other", positive]
        other = [level for level in levels if level != positive]
        return ((codes == levels.index(positive)).to(torch.int64),
                [other[0] if other else None, positive])
    if len(levels) == 2:
        return codes, levels
    if len(levels) == 1 and levels[0] in (0, 1, True, False):
        value = int(levels[0])
        return torch.full_like(codes, value), [0, 1]
    raise AnalysisError("not_binary", f"Column '{name}' must have exactly two distinct values "
                        f"(it has {len(levels)}). Recode it or name the event with positive=.")


# ---- ranking kernels -------------------------------------------------------------------


def midranks(x: Tensor) -> tuple[Tensor, Tensor]:
    """Midranks (1-based, ties share the mean rank) and the tie-group sizes of ``x``."""
    values, order = torch.sort(x, stable=True)
    _, inverse, counts = torch.unique_consecutive(values, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).to(FLOAT)
    average = ends - (counts.to(FLOAT) - 1.0) / 2.0
    ranks = torch.empty_like(x, dtype=FLOAT)
    ranks[order] = average[inverse]
    return ranks, counts


def tie_term(counts: Tensor) -> float:
    """sum(t^3 - t) over tie groups of sizes ``counts``."""
    t = counts.to(FLOAT)
    return float((t * t * t - t).sum())


def row_midranks(x: Tensor) -> tuple[Tensor, Tensor]:
    """Midranks within each row of an [n, k] matrix and sum(t^3 - t) per row.

    One sort per row (torch.sort along dim 1); tie runs are found on the
    flattened sorted values, so there is no loop over rows.
    """
    n, k = x.shape
    values, order = torch.sort(x, dim=1, stable=True)
    flat = values.reshape(-1)
    start = torch.ones(n * k, dtype=torch.bool)
    if n * k > 1:
        start[1:] = flat[1:] != flat[:-1]
    start[::k] = True
    run = start.to(torch.int64).cumsum(0) - 1
    runs = int(run[-1]) + 1 if n * k else 0
    position = torch.arange(n * k, dtype=FLOAT) % k + 1.0
    size = torch.zeros(runs, dtype=FLOAT).index_add_(0, run, torch.ones(n * k, dtype=FLOAT))
    total = torch.zeros(runs, dtype=FLOAT).index_add_(0, run, position)
    average = (total / size)[run].reshape(n, k)
    ranks = torch.empty_like(x, dtype=FLOAT)
    ranks.scatter_(1, order, average)
    row_of_run = torch.zeros(runs, dtype=torch.int64)
    row_of_run[run] = torch.arange(n * k) // k
    ties = torch.zeros(n, dtype=FLOAT).index_add_(0, row_of_run, size ** 3 - size)
    return ranks, ties


def group_sum(values: Tensor, codes: Tensor, groups: int) -> Tensor:
    return torch.zeros(groups, dtype=FLOAT).index_add_(0, codes, values.to(FLOAT))


def group_count(codes: Tensor, groups: int) -> Tensor:
    return torch.bincount(codes, minlength=groups).to(FLOAT)


# ---- probabilities ---------------------------------------------------------------------


def normal_two_sided(z: float) -> float:
    return min(1.0, 2.0 * normal_sf(abs(z)))


def binomial_upper(k: float, n: float, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), through the regularized incomplete beta."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return beta_inc(float(k), float(n - k + 1), p)


def binomial_lower(k: float, n: float, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return beta_inc(float(n - k), float(k + 1), 1.0 - p)


def binomial_log_pmf(n: int, p: float) -> Tensor:
    """log P(X = k), k = 0..n, for X ~ Binomial(n, p) with 0 < p < 1."""
    k = torch.arange(n + 1, dtype=FLOAT)
    return (math.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def adjust_p_values(p_values: Sequence[float], method: str) -> list[float]:
    """Bonferroni or Holm step-down adjustment of a family of p-values."""
    m = len(p_values)
    if method == "none" or m == 0:
        return [float(p) for p in p_values]
    if method == "bonferroni":
        return [min(1.0, m * float(p)) for p in p_values]
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted, running = [0.0] * m, 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


# Largest number of categories of a square (paired) table: K x K cells are held and tested.
MAX_SQUARE = 1000


def square_table(frame: pd.DataFrame, a: str, b: str) -> tuple[Tensor, list[Any]]:
    """[K, K] counts of two columns classified by the union of their levels (rows = a)."""
    stacked = pd.concat([frame[a], frame[b]], ignore_index=True)
    codes, levels = group_codes(stacked)
    k, n = len(levels), len(frame)
    if k > MAX_SQUARE:
        raise AnalysisError("too_many_categories", f"'{a}' and '{b}' take {k} distinct values; "
                            f"a paired table can have at most {MAX_SQUARE} categories. Group "
                            "the values into classes first.")
    cell = codes[:n] * k + codes[n:]
    return torch.bincount(cell, minlength=k * k).reshape(k, k).to(FLOAT), levels


def weight_column(frame: pd.DataFrame, weights: str | None) -> Tensor | None:
    """Frequency weights: a nonnegative numeric column (None when not weighted)."""
    if weights is None:
        return None
    w = numeric(frame, weights)
    if bool((w < 0).any()):
        raise AnalysisError("invalid_weights", f"Weights in '{weights}' must be nonnegative.")
    if float(w.sum()) <= 0.0:
        raise AnalysisError("invalid_weights", f"The weights in '{weights}' sum to zero.")
    return w


def weighted_count(codes: Tensor, groups: int, weights: Tensor | None) -> Tensor:
    if weights is None:
        return torch.bincount(codes, minlength=groups).to(FLOAT)
    return torch.zeros(groups, dtype=FLOAT).index_add_(0, codes, weights)
