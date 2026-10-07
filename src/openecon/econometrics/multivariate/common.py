"""Shared helpers of the multivariate family: samples, moment matrices, signs, tables.

The procedures of this family are descriptive multivariate analyses, not model
fits. Each public function takes a table (DataFrame, mapping of columns or list
of records) and column names and returns a ``core.TableSet`` of result tables
whose ``attrs`` hold the scalar results.

Missing data. Rows with a missing value in any analysed column are deleted
listwise (``missing="drop"``, what SPSS FACTOR / RELIABILITY / DISCRIMINANT do
by default and what Stata's multivariate commands do) or refused
(``missing="raise"``); the number of rows used and dropped is reported.

Moments. Means, the centred cross-product matrix X'X, covariances and
correlations come from ONE pass over the data: the columns are centred at
their means (a second pass over the centred values removes the rounding error
of the first mean) and X'X is a single matrix product. No n-by-n matrix is
formed anywhere except in hierarchical clustering and multidimensional scaling,
which need the distance matrix and guard its size.

Signs. Eigenvectors and singular vectors are determined only up to sign. Every
procedure fixes the sign of each vector so that its entry of largest absolute
value is positive (the first such entry on ties), which makes results
deterministic across platforms and LAPACK versions.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.engines import distributions as dist

FLOAT = torch.float64
# Squares of values beyond this magnitude overflow float64 sums of squares.
LARGEST = 1e150
# A variable whose centred sum of squares is below this fraction of its raw sum of
# squares is constant to working precision.
_CONSTANT = 1e-24


def procedure(function: Callable[..., Any]) -> Callable[..., Any]:
    """Public-procedure wrapper: kernel failures surface as ``AnalysisError``."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return kernel_call(function, *args, **kwargs)
        except torch.linalg.LinAlgError as exc:
            raise AnalysisError("numerical_failure", "A matrix decomposition failed to "
                                "converge; check the data for extreme or degenerate "
                                "values.") from exc

    return wrapper


# ---- argument checks -------------------------------------------------------------------


def check_choice(value: Any, name: str, choices: Sequence[Any]) -> Any:
    if not isinstance(value, str) or value not in choices:
        raise AnalysisError("invalid_option", f"{name} must be one of: "
                            f"{', '.join(map(str, choices))}.")
    return value


def check_flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError("invalid_option", f"{name} must be True or False.")
    return value


def check_number(value: Any, name: str, *, minimum: float | None = None,
                 maximum: float | None = None, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise AnalysisError("invalid_option", f"{name} must be a finite number.")
    number = float(value)
    if minimum is not None and (number < minimum or (exclusive and number == minimum)):
        raise AnalysisError("invalid_option", f"{name} must be "
                            f"{'greater than' if exclusive else 'at least'} {minimum:g}.")
    if maximum is not None and number > maximum:
        raise AnalysisError("invalid_option", f"{name} must be at most {maximum:g}.")
    return number


def check_count(value: Any, name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise AnalysisError("invalid_option", f"{name} must be an integer of at least {minimum}.")
    if maximum is not None and value > maximum:
        raise AnalysisError("invalid_option", f"{name} must be at most {maximum}.")
    return int(value)


def check_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError("invalid_spec", f"{what} must be the name of one column.")
    return value


def name_list(value: Any, what: str, *, minimum: int = 1) -> list[str]:
    """A list of distinct column names (a bare string is an error)."""
    if value is None:
        value = []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) \
            or not all(isinstance(item, str) and item for item in value):
        raise AnalysisError("invalid_spec", f"{what} must be a list of column names, for "
                            f"example {what}=['x1', 'x2'].")
    names = list(value)
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise AnalysisError("invalid_spec", f"{what} lists a column more than once: "
                            f"{', '.join(repeated)}.")
    if len(names) < minimum:
        raise AnalysisError("invalid_spec", f"{what} must name at least {minimum} column(s).")
    return names


def check_result(result: Any, procedure_name: str, function: str) -> Any:
    """A TableSet produced by ``oe.<procedure_name>`` (checked through its attrs)."""
    attrs = getattr(result, "attrs", None)
    if not isinstance(result, dict) or not isinstance(attrs, dict) \
            or attrs.get("procedure") != procedure_name:
        raise AnalysisError("invalid_result", f"{function} needs the result returned by "
                            f"oe.{procedure_name}(...).")
    return result


# ---- samples ---------------------------------------------------------------------------


def source(data: Any) -> pd.DataFrame:
    """The supplied data as a DataFrame with unique column names and at least one row."""
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    if not len(frame):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    return frame


def require_numeric(frame: pd.DataFrame, names: Sequence[str]) -> None:
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    for name in names:
        dtype = frame[name].dtype
        if not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)) or is_complex_dtype(dtype):
            raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric for this "
                                f"procedure, but it holds {dtype} values.")


def select(data: Any, names: Sequence[str], *, numeric: Sequence[str] | None = None,
           missing: str = "drop") -> tuple[pd.DataFrame, pd.Series, int]:
    """Rows complete in ``names`` (listwise deletion).

    Returns the complete rows (index reset), the boolean mask of the rows kept
    (indexed like the supplied data) and the number of rows dropped. ``numeric``
    lists the columns that must be numeric (default: all of them).
    """
    check_choice(missing, "missing", ("drop", "raise"))
    names = list(names)
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise AnalysisError("invalid_spec", "A column is used in more than one role: "
                            f"{', '.join(repeated)}.")
    frame = source(data)
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    require_numeric(frame, names if numeric is None else numeric)
    frame = frame.loc[:, names]
    keep = ~frame.isna().any(axis=1)
    dropped = int((~keep).sum())
    if dropped and missing == "raise":
        raise AnalysisError("missing_values", f"{dropped} observation(s) have missing values in "
                            f"{', '.join(names)}. Pass missing='drop' to exclude them.")
    if dropped:
        frame = frame.loc[keep]
    if not len(frame):
        raise AnalysisError("empty_sample", "No complete observations remain after excluding "
                            "missing values.")
    return frame.reset_index(drop=True), keep, dropped


def column(frame: pd.DataFrame, name: str, *, allow_missing: bool = False) -> Tensor:
    """A numeric column as a float64 tensor; infinities and huge values are errors."""
    buffer = frame[name].to_numpy(dtype="float64", na_value=float("nan"))
    values = torch.as_tensor(buffer if buffer.flags.writeable else buffer.copy(), dtype=FLOAT)
    present = ~torch.isnan(values)
    if bool(torch.isinf(values).any()) or (not allow_missing and not bool(present.all())):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite values.")
    if bool(present.any()) and float(values[present].abs().max()) > LARGEST:
        raise AnalysisError("non_finite_values", f"Column '{name}' has values too large for "
                            "sums of squares in double precision; rescale it.")
    return values


def matrix(frame: pd.DataFrame, names: Sequence[str], *, allow_missing: bool = False) -> Tensor:
    """The named numeric columns as an [n, p] float64 tensor."""
    if not names:
        return torch.empty((len(frame), 0), dtype=FLOAT)
    return torch.stack([column(frame, name, allow_missing=allow_missing) for name in names],
                       dim=1)


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


def group_codes(series: pd.Series, name: str) -> tuple[Tensor, list[Any]]:
    """int64 codes 0..G-1 and the distinct levels of a grouping column, sorted ascending.

    Categorical columns keep their declared category order; values that are not
    mutually comparable are ordered by their text.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.cat.remove_unused_categories()
        codes, levels = series.cat.codes.to_numpy(dtype="int64"), list(series.cat.categories)
    else:
        try:
            codes, uniques = pd.factorize(series, sort=True)
        except TypeError:
            try:
                codes, uniques = pd.factorize(series.astype(str), sort=True)
            except (TypeError, ValueError) as exc:
                raise AnalysisError("invalid_groups", f"Values of '{name}' must be scalar "
                                    "labels.") from exc
        levels = list(uniques)
    if (codes < 0).any():
        raise AnalysisError("invalid_groups", f"Labels in '{name}' must not be missing.")
    labels = [label(level) for level in levels]
    if len(set(map(str, labels))) != len(labels):
        raise AnalysisError("invalid_groups", f"Labels in '{name}' are not distinct once "
                            "written as text; recode the column.")
    return torch.as_tensor(codes, dtype=torch.int64), labels


# ---- moments ---------------------------------------------------------------------------


def centre(x: Tensor) -> tuple[Tensor, Tensor]:
    """(X - mean, mean); a second pass removes the rounding error of the first mean."""
    mean = x.mean(0)
    centred = x - mean
    shift = centred.mean(0)
    return centred - shift, mean + shift


def moments(x: Tensor, names: Sequence[str], *, need_variation: bool = True
            ) -> tuple[Tensor, Tensor, Tensor]:
    """(mean [p], centred SSCP [p, p], centred data [n, p]) from one pass.

    Raises ``constant_column`` when a variable has no variation and
    ``need_variation`` (correlations and standardized scores are undefined).
    """
    centred, mean = centre(x)
    sscp = centred.T @ centred
    sscp = (sscp + sscp.T) / 2
    if need_variation:
        raw = x.square().sum(0)
        flat = [name for name, ss, total in zip(names, sscp.diagonal().tolist(), raw.tolist(),
                                                strict=True)
                if not ss > _CONSTANT * max(total, 1e-300)]
        if flat:
            raise AnalysisError("constant_column", f"Column(s) {', '.join(flat)} do not vary in "
                                "the sample; remove them from the analysis.")
    return mean, sscp, centred


def correlation(sscp: Tensor) -> Tensor:
    """Correlation matrix of a centred SSCP (or covariance) matrix; unit diagonal exactly."""
    scale = sscp.diagonal().sqrt()
    r = sscp / torch.outer(scale, scale)
    r = ((r + r.T) / 2).clamp(-1.0, 1.0)
    r.diagonal().fill_(1.0)
    return r


def fix_signs(vectors: Tensor) -> Tensor:
    """Flip the columns of ``vectors`` so that each column's largest |entry| is positive."""
    if vectors.numel() == 0:
        return vectors
    pivot = vectors.abs().argmax(0)
    sign = torch.sign(vectors[pivot, torch.arange(vectors.shape[1])])
    return vectors * torch.where(sign == 0, torch.ones_like(sign), sign)


def descending_eigh(a: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues (descending) and sign-fixed eigenvectors of a symmetric matrix."""
    values, vectors = torch.linalg.eigh((a + a.T) / 2)
    return values.flip(0), fix_signs(vectors.flip(1))


def positive_definite(a: Tensor, what: str, hint: str) -> Tensor:
    """Lower Cholesky factor of a symmetric matrix, or ``singular_matrix``.

    A pivot at the rounding level (relative to the diagonal) counts as zero, so
    exactly collinear variables are detected instead of inverted to 1/eps.
    """
    a = (a + a.T) / 2
    factor, info = torch.linalg.cholesky_ex(a, check_errors=False)
    k = a.shape[0]
    lost = factor.diagonal().square() <= 8 * k * torch.finfo(FLOAT).eps * a.diagonal().abs()
    if int(info) != 0 or not bool(torch.isfinite(factor).all()) or bool(lost.any()):
        raise AnalysisError("singular_matrix", f"The {what} is singular: {hint}")
    return factor


# ---- tables ----------------------------------------------------------------------------


def frame(rows: Any, *, columns: Sequence[str], index: Sequence[Any] | None = None,
          **attrs: Any) -> pd.DataFrame:
    """A result table whose numeric columns are float even when cells are undefined (None)."""
    columns = [str(name) for name in columns]
    repeated = sorted({name for name in columns if columns.count(name) > 1})
    if repeated:
        raise AnalysisError("invalid_spec", f"The name(s) {', '.join(repeated)} would label "
                            "more than one column of a result table; rename the column(s) or "
                            "category label(s) in the data.")
    if isinstance(rows, Tensor):
        rows = rows.tolist()
    out = table(rows, columns=columns, index=index, **attrs)
    for name in out.columns:
        if out[name].dtype == object and all(
                value is None or (isinstance(value, Real) and not isinstance(value, bool))
                for value in out[name]):
            out[name] = out[name].astype("float64")
    return out


def numbered(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, count + 1)]


def finite(value: Any) -> float | None:
    """A finite float, or None for a statistic that is undefined for this sample."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def chi2_upper(statistic: float | None, df: float) -> float | None:
    if statistic is None or not (math.isfinite(statistic) and df > 0):
        return None
    return dist.chi2_sf(max(statistic, 0.0), df)


def f_upper(statistic: float | None, df1: float, df2: float) -> float | None:
    if statistic is None or not (math.isfinite(statistic) and df1 > 0 and df2 > 0):
        return None
    return dist.f_sf(max(statistic, 0.0), df1, df2)


def aligned(values: Tensor, keep: pd.Series, columns: Sequence[str],
            *, integer: Sequence[str] = ()) -> pd.DataFrame:
    """Per-observation output as a table indexed like the supplied data.

    Rows that were not used (missing inputs) hold missing values. ``integer``
    columns are stored as nullable integers.
    """
    out = pd.DataFrame(index=keep.index, columns=list(columns), dtype="float64")
    out.loc[keep.to_numpy(), :] = values.numpy()
    result = table(out)
    for name in integer:
        result[name] = result[name].astype("Int64")
    return result
