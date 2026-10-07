"""Sample handling and small regression helpers shared by the unit-root family.

The procedures of this family are tests, not model fits: they take a table
and column names and return result tables (``core.table`` / ``core.TableSet``).
Series are taken in row order, or sorted by ``time`` when a time column is
named. Missing values, repeated periods and gaps are errors, because every
statistic here depends on the lag structure and silently dropping a row would
change the model. Columns whose squares would overflow or underflow float64
are refused with the advice to rescale (see ``check_magnitude``).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import pandas as pd
import torch
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype,
)
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.engines import covariance as cov
from openecon.engines.distributions import t_isf, t_sf
from openecon.engines.linalg import LeastSquares, least_squares

KERNELS = ("bartlett", "parzen", "quadratic_spectral")


# ---- argument checks -----------------------------------------------------------------


def check_choice(value: Any, name: str, choices: tuple[Any, ...]) -> Any:
    if not isinstance(value, str) or value not in choices:
        raise AnalysisError("invalid_option", f"{name} must be one of: "
                            f"{', '.join(map(str, choices))}.")
    return value


def check_flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError("invalid_option", f"{name} must be True or False.")
    return value


def check_count(value: Any, name: str, *, minimum: int = 0) -> int:
    """An integer option (Python or NumPy integer, not a bool) of at least ``minimum``."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise AnalysisError("invalid_lags" if "lag" in name else "invalid_option",
                            f"{name} must be an integer of at least {minimum}.")
    return int(value)


def check_fraction(value: Any, name: str) -> float:
    """A trimming fraction in [0, 0.5)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value < 0.5:
        raise AnalysisError("invalid_option", f"{name} must be a fraction in [0, 0.5).")
    return float(value)


def column_names(value: Any, name: str) -> list[str]:
    """A list of distinct column names (a bare string is an error)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) \
            or not all(isinstance(item, str) for item in value):
        raise AnalysisError("invalid_spec", f"{name} must be a list of column names, for "
                            f"example {name}=['income'].")
    if len(set(value)) != len(value):
        raise AnalysisError("invalid_spec", f"{name} contains duplicate column names.")
    return list(value)


# ---- samples ---------------------------------------------------------------------------


def _require_time_column(column: pd.Series, name: str) -> None:
    dtype = column.dtype
    if is_bool_dtype(dtype) or not (is_numeric_dtype(dtype) or is_datetime64_any_dtype(dtype)):
        raise AnalysisError("invalid_time", f"Time column '{name}' must contain integer periods "
                            "or datetimes. Convert text dates with pandas.to_datetime, or "
                            "number the periods 1, 2, 3, ...")


def _select(data: Any, names: list[str]) -> pd.DataFrame:
    for name in names:
        if not isinstance(name, str):
            raise AnalysisError("invalid_spec", "Column names must be strings.")
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "A column is used in more than one role: "
                            f"{', '.join(sorted({n for n in names if names.count(n) > 1}))}.")
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    frame = frame.loc[:, names].reset_index(drop=True)
    missing = int(frame.isna().any(axis=1).sum())
    if missing:
        raise AnalysisError("missing_values", f"{missing} observation(s) have missing values in "
                            f"{', '.join(names)}. These tests need complete, regularly spaced "
                            "series: restrict the sample or fill the gaps first.")
    return frame


def _integer_periods(column: pd.Series, name: str) -> Tensor:
    if is_integer_dtype(column.dtype) and not column.hasnans:
        # Integer storage is read as int64: float64 cannot tell 2^53 from 2^53 + 1.
        return torch.as_tensor(column.to_numpy(dtype="int64"))
    periods = _numeric(column, name)
    if bool((periods != periods.round()).any()):
        raise AnalysisError("invalid_time", f"Time column '{name}' must contain integer periods "
                            "or datetimes.")
    return periods.to(torch.int64)


# Sums of squares of values beyond these bounds overflow or underflow float64.
_LARGEST = 1e140
_SMALLEST = 1e-140


def check_magnitude(values: Tensor, names: list[str]) -> None:
    """Refuse columns whose squares would overflow or underflow float64.

    Every statistic of this family is a ratio of sums of squares. With values
    around 1e200 those sums are infinite, and with a spread around 1e-200 they
    are zero, which would otherwise be reported as a constant series or an
    exact fit. The tests are scale invariant, so rescaling loses nothing.
    """
    if not values.numel():
        return
    largest = values.abs().amax(dim=0)
    spread = values.amax(dim=0) - values.amin(dim=0)
    for index, name in enumerate(names):
        size, width = float(largest[index]), float(spread[index])
        if size > _LARGEST:
            raise AnalysisError("numerical_failure", f"Column '{name}' has values of magnitude "
                                f"{size:.1e}: their squares overflow float64. Rescale the "
                                "column (the tests do not depend on its unit).")
        if 0.0 < width < _SMALLEST:
            raise AnalysisError("numerical_failure", f"Column '{name}' varies by only "
                                f"{width:.1e}: the squares of its deviations underflow float64. "
                                "Rescale the column (the tests do not depend on its unit).")


def load_series(data: Any, columns: list[str],
                time: str | None = None) -> tuple[Tensor, pd.Series]:
    """Complete, regularly spaced series as an [n, c] tensor, and their period labels.

    Rows are sorted by ``time`` when given: integer periods must be consecutive,
    datetimes are taken as consecutive in sorted order. Without a time column
    the row order is the time order and the labels are 0-based row positions.
    The labels come back as a pandas Series in time order (see ``period_label``).
    """
    if time is not None and not isinstance(time, str):
        raise AnalysisError("invalid_spec", "time must be the name of one time column.")
    frame = _select(data, [*columns, *([time] if time is not None else [])])
    if time is None:
        labels = pd.Series(range(len(frame)), name="period")
    else:
        _require_time_column(frame[time], time)
        if bool(frame[time].duplicated().any()):
            raise AnalysisError("repeated_time_values", "Time values are repeated.")
        frame = frame.sort_values(time, kind="stable").reset_index(drop=True)
        if not is_datetime64_any_dtype(frame[time].dtype):
            periods = _integer_periods(frame[time], time)
            if len(periods) > 1 and bool((periods[1:] - periods[:-1] != 1).any()):
                raise AnalysisError("time_gaps", f"Time column '{time}' has gaps. These tests "
                                    "need consecutive periods: fill the gaps or restrict the "
                                    "sample.")
        labels = frame[time].rename("period")
    values = torch.stack([_numeric(frame[name], name) for name in columns], dim=1)
    check_magnitude(values, columns)
    return values, labels


def json_label(value: Any) -> Any:
    """A JSON-safe period or panel label (dates as ISO strings)."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else value
    return str(value)


def period_label(labels: pd.Series, position: int) -> Any:
    """JSON-safe label of the observation at a 0-based position."""
    return json_label(labels.iloc[position])


@dataclass
class Panel:
    """A panel sorted by unit and period, without gaps inside a unit."""

    values: Tensor          # [n, c]
    codes: Tensor           # int64 [n], 0..N-1 in sorted order
    period: Tensor          # int64 [n] period index
    counts: Tensor          # int64 [N] observations per unit
    labels: list            # unit labels
    balanced: bool

    @property
    def n_panels(self) -> int:
        return int(self.counts.shape[0])

    def cube(self) -> Tensor:
        """[N, T, c] view of a balanced panel."""
        periods = int(self.counts[0])
        return self.values.reshape(self.n_panels, periods, self.values.shape[1])


def load_panel(data: Any, columns: list[str], panel: str, time: str) -> Panel:
    """Sort a long panel by unit and period and check that each unit is gap free."""
    if not isinstance(panel, str) or not isinstance(time, str):
        raise AnalysisError("invalid_spec", "panel and time must each name one column.")
    frame = _select(data, [*columns, panel, time])
    _require_time_column(frame[time], time)
    if bool(frame.duplicated([panel, time]).any()):
        raise AnalysisError("repeated_time_values", "Time values are repeated within a panel.")
    if is_datetime64_any_dtype(frame[time].dtype):
        period = torch.as_tensor(frame[time].rank(method="dense").to_numpy(dtype="int64"))
    else:
        period = _integer_periods(frame[time], time)
    try:
        unit, labels = pd.factorize(frame[panel], sort=True)
    except TypeError:
        unit, labels = pd.factorize(frame[panel], sort=False)
    keys = pd.DataFrame({"unit": unit, "period": period.numpy()})
    order = keys.sort_values(["unit", "period"], kind="stable").index.to_numpy()
    frame = frame.iloc[order].reset_index(drop=True)
    codes = torch.as_tensor(keys["unit"].to_numpy(dtype="int64")[order])
    period = period[torch.as_tensor(order)]
    counts = torch.bincount(codes, minlength=len(labels))
    same_unit = codes[1:] == codes[:-1]
    if bool((same_unit & (period[1:] - period[:-1] != 1)).any()):
        raise AnalysisError("time_gaps", "At least one panel has gaps in its time values. Panel "
                            "unit-root tests need consecutive periods within each panel.")
    values = torch.stack([_numeric(frame[name], name) for name in columns], dim=1)
    check_magnitude(values, columns)
    starts = period[torch.cat([torch.ones(1, dtype=torch.bool), ~same_unit])]
    balanced = bool((counts == counts[0]).all()) and bool((starts == starts[0]).all())
    return Panel(values=values, codes=codes, period=period, counts=counts,
                 labels=[json_label(value) for value in labels], balanced=balanced)


def require_variation(x: Tensor, minimum: int, what: str) -> None:
    if x.shape[0] < minimum:
        raise AnalysisError("insufficient_observations",
                            f"{what} needs at least {minimum} observations; {x.shape[0]} given.")
    if not bool((x != x[0]).any()):
        raise AnalysisError("constant_series", f"{what} needs a series that is not constant.")


# ---- least squares ---------------------------------------------------------------------


@dataclass
class Regression:
    beta: Tensor
    se: Tensor
    resid: Tensor
    ssr: float
    nobs: int
    df: int
    s2: float
    xtx_inv: Tensor
    exact: bool = False     # the residual is at rounding level relative to y


def regress(x: Tensor, y: Tensor, what: str) -> Regression:
    """OLS by QR for an auxiliary test regression; rank deficiency is an error.

    A test regression that loses a column (a constant series, an exact trend,
    too many lags for the sample) no longer estimates the stated model, so it
    raises ``collinear_regressors`` instead of silently omitting the column.
    """
    n, k = x.shape
    if n <= k:
        raise AnalysisError("insufficient_observations", f"{what} has {n} observation(s) for {k} "
                            "parameter(s); use fewer lags or a longer series.")
    fit: LeastSquares = kernel_call(least_squares, x, y)
    if fit.omitted:
        raise AnalysisError("collinear_regressors", f"{what} is rank deficient: the series does "
                            "not vary enough to identify every term (for example it is constant, "
                            "an exact trend, or has too many lags for the sample).")
    ssr = float(fit.ssr)
    df = n - k
    s2 = ssr / df
    exact = not ssr > 1e-22 * float(y.square().sum())
    return Regression(beta=fit.beta, se=(s2 * fit.xtx_inv.diagonal()).sqrt(), resid=fit.resid,
                      ssr=ssr, nobs=n, df=df, s2=s2, xtx_inv=fit.xtx_inv, exact=exact)


def uncentre(fit: Regression, intercept: int, shifts: dict[int, float],
             constant: float = 0.0) -> Regression:
    """Express a regression fitted on shifted variables for the original ones.

    The test regressions with a constant are fitted on series measured from
    their first observation, which leaves every slope, residual and test
    statistic unchanged and keeps the least-squares problem well conditioned
    when the level of a series is large relative to its variation. Only the
    intercept depends on the origin: with regressor j shifted by ``shifts[j]``
    (x_j - c_j was used) and the outcome by ``constant``,

        intercept = intercept* - sum_j c_j b_j + constant,

    and its variance follows from the same linear map applied to (X'X)^{-1}.
    """
    k = fit.beta.shape[0]
    transform = torch.eye(k, dtype=torch.float64)
    for column, shift in shifts.items():
        transform[intercept, column] -= shift
    beta = transform @ fit.beta
    beta[intercept] += constant
    xtx_inv = transform @ fit.xtx_inv @ transform.T
    return Regression(beta=beta, se=(fit.s2 * xtx_inv.diagonal()).clamp_min(0.0).sqrt(),
                      resid=fit.resid, ssr=fit.ssr, nobs=fit.nobs, df=fit.df, s2=fit.s2,
                      xtx_inv=xtx_inv, exact=fit.exact)


def regression_table(terms: list[str], fit: Regression, **attrs: Any) -> pd.DataFrame:
    """Coefficient table of an auxiliary regression with t(n - k) inference."""
    critical = t_isf(0.025, fit.df)
    rows = []
    for name, beta, se in zip(terms, fit.beta.tolist(), fit.se.tolist(), strict=True):
        statistic = beta / se if se > 0 else None
        rows.append([beta, se, statistic,
                     None if statistic is None else min(1.0, 2 * t_sf(abs(statistic), fit.df)),
                     beta - critical * se, beta + critical * se])
    return table(rows, columns=["coefficient", "std_error", "statistic", "p_value", "ci_low",
                                "ci_high"],
                 index=terms, nobs=fit.nobs, df_resid=fit.df, rmse=fit.s2 ** 0.5, **attrs)


def deterministic_columns(n: int, case: str, first: int = 1) -> tuple[list[Tensor], list[str]]:
    """Constant / trend / squared-trend columns of a test regression.

    The trend counts first, first + 1, ... over the n rows (1..n by default).
    """
    columns, names = [], []
    if case in {"c", "ct", "ctt"}:
        trend = torch.arange(first, first + n, dtype=torch.float64)
        if case in {"ct", "ctt"}:
            columns.append(trend)
            names.append("_trend")
        if case == "ctt":
            columns.append(trend.square())
            names.append("_trend2")
        columns.append(torch.ones(n, dtype=torch.float64))
        names.append("Intercept")
    return columns, names


def adf_design(y: Tensor, lags: int, case: str, *, name: str = "y",
               start: int | None = None) -> tuple[Tensor, Tensor, list[str]]:
    """Design of the augmented Dickey-Fuller regression of one series.

    Rows are t = start+2..T (``start`` defaults to ``lags``; a larger value
    gives lag-order comparisons a common sample). Columns: y_{t-1}, the lagged
    differences, then the deterministic terms of ``case``. The trend is t - 1
    (1 at the first difference of the series, as Stata's ``_trend``), so it
    starts at start + 1 in the first row. Returns the design, the dependent
    variable (Delta y_t) and the term names.
    """
    start = lags if start is None else start
    total = y.shape[0]
    rows = total - 1 - start
    parameters = 1 + lags + {"n": 0, "c": 1, "ct": 2, "ctt": 3}[case]
    if rows <= parameters:
        raise AnalysisError("insufficient_observations", f"The Dickey-Fuller regression has "
                            f"{max(rows, 0)} observation(s) for {parameters} parameter(s); use "
                            "fewer lags or a longer series.")
    dy = y[1:] - y[:-1]
    columns = [y[start:total - 1]]
    names = [f"L1.{name}"]
    for j in range(1, lags + 1):
        columns.append(dy[start - j:total - 1 - j])
        names.append(f"LD.{name}" if j == 1 else f"L{j}D.{name}")
    extra, extra_names = deterministic_columns(total - 1 - start, case, first=start + 1)
    return torch.stack([*columns, *extra], dim=1), dy[start:].contiguous(), [*names, *extra_names]


def long_run_variance(u: Tensor, lags: int, kernel: str = "bartlett") -> float:
    """gamma_0 + 2 sum_j w_j gamma_j with gamma_j = (1/n) sum_t u_t u_{t-j} (no centring)."""
    return float(kernel_call(cov.meat_hac, u[:, None].contiguous(), lags, kernel)[0, 0]) \
        / u.shape[0]


def autocovariances(u: Tensor, lags: int) -> Tensor:
    """gamma_0..gamma_lags with divisor n (one dot product per lag)."""
    n = u.shape[0]
    return torch.stack([torch.dot(u[j:], u[:n - j]) for j in range(lags + 1)]) / n


def critical_columns(values: dict[str, float] | None) -> list[float | None]:
    """[1%, 5%, 10%] entries for a result row."""
    if values is None:
        return [None, None, None]
    return [values.get("1%"), values.get("5%"), values.get("10%")]


CRITICAL_NAMES = ["critical_1pct", "critical_5pct", "critical_10pct"]
