"""Series diagnostics: correlogram, portmanteau, Jarque-Bera and ARCH-LM tests.

These work on one numeric column of a table (or on residuals an estimator of
this family passes in as a tensor). The observations are taken in row order,
or sorted by ``time`` when a time column is named; the series must be regularly
spaced without missing values, because every statistic here depends on the lag
structure.

Autocorrelations use the Box-Jenkins estimator with divisor n,

    r_k = sum_{t=k+1..n} (y_t - ybar)(y_(t-k) - ybar) / sum_t (y_t - ybar)^2,

partial autocorrelations come from the Durbin-Levinson recursion on r_1..r_k
(Stata's ``corrgram, yw``; EViews and SPSS do the same) or, on request, from
regressions of y_t on a constant and k lags (Stata's default for corrgram/pac),
and the portmanteau statistic is Ljung and Box's

    Q_k = n (n + 2) sum_{j<=k} r_j^2 / (n - j)  ~  chi2(k)   under white noise.

Every public function returns a result table (``core.table``): ``corrgram`` one
row per lag, the tests one row per test with the scalar results repeated in
``attrs`` (``statistic``, ``df``, ``p_value``, ``distribution``, ``label``, ...).
The ``*_test`` helpers return the same numbers as plain dicts in the shape of
``core.wald_test``; ``oe.arima`` stores them in ``result.tests``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.filters import levinson
from openecon.econometrics.core import kernel_call, table
from openecon.engines.distributions import chi2_sf, normal_isf
from openecon.engines.linalg import least_squares


def require_time_column(column: pd.Series, name: str) -> None:
    """A time column holds integer periods or datetimes; text and booleans are refused."""
    dtype = column.dtype
    if is_bool_dtype(dtype) or not (is_numeric_dtype(dtype) or is_datetime64_any_dtype(dtype)):
        raise AnalysisError("invalid_time", f"Time column '{name}' must contain integer periods or "
                            "datetimes. Convert text dates with pandas.to_datetime, or number the "
                            "periods 1, 2, 3, ...")


def load_series(data: Any, y: str, time: str | None = None) -> tuple[Tensor, pd.Series]:
    """One complete, regularly spaced numeric series and its period labels.

    Rows are sorted by ``time`` when given (integer periods must be consecutive;
    datetimes are taken as consecutive in sorted order). Missing values are an
    error: dropping interior observations would silently change every lag.
    """
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    if time is not None and (not isinstance(time, str) or time == y):
        raise AnalysisError("invalid_spec", "time must be the name of one time column, different "
                            "from y.")
    frame = _coerce_frame(data)
    absent = [name for name in (y, time) if name is not None and name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    frame = frame.loc[:, [name for name in (y, time) if name is not None]].reset_index(drop=True)
    if bool(frame.isna().any().any()):
        raise AnalysisError("missing_values", f"Column '{y}' (or its time column) contains missing "
                            "values. Series diagnostics need a complete, regularly spaced series: "
                            "restrict the sample or fill the gaps first.")
    if time is None:
        return _numeric(frame[y], y), pd.Series(range(len(frame)), name="period")
    require_time_column(frame[time], time)
    if bool(frame[time].duplicated().any()):
        raise AnalysisError("repeated_time_values", "Time values are repeated.")
    frame = frame.sort_values(time, kind="stable").reset_index(drop=True)
    if not is_datetime64_any_dtype(frame[time].dtype):
        periods = _numeric(frame[time], time)
        if bool((periods != periods.round()).any()):
            raise AnalysisError("invalid_time", f"Time column '{time}' must contain integer "
                                "periods or datetimes.")
        if len(periods) > 1 and bool((periods[1:] - periods[:-1] != 1).any()):
            raise AnalysisError("time_gaps", f"Time column '{time}' has gaps. Series diagnostics "
                                "need consecutive periods: fill the gaps or restrict the sample.")
    return _numeric(frame[y], y), frame[time].rename("period")


def _require_variation(x: Tensor, minimum: int, what: str) -> None:
    if x.shape[0] < minimum:
        raise AnalysisError("insufficient_observations",
                            f"{what} needs at least {minimum} observations.")
    if not bool((x != x[0]).any()):
        raise AnalysisError("constant_series", f"{what} needs a series that is not constant.")


def default_lags(n: int) -> int:
    """Stata's default for corrgram and wntestq: min(floor(n/2) - 2, 40)."""
    return max(1, min(n // 2 - 2, 40))


def _test_table(test: dict[str, Any], columns: tuple[str, ...]) -> pd.DataFrame:
    """One-row table of a test; all of its scalar results are repeated in ``attrs``."""
    return table([[test[name] for name in columns]], columns=list(columns), **test)


def _check_lags(lags: Any, n: int, what: str) -> int:
    if lags is None:
        return default_lags(n)
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise AnalysisError("invalid_lags", f"{what}: lags must be a positive integer.")
    if lags >= n:
        raise AnalysisError("invalid_lags", f"{what}: lags must be smaller than the number of "
                            f"observations ({n}).")
    return lags


@torch.no_grad()
def autocorrelations(x: Tensor, lags: int) -> Tensor:
    """r_1..r_lags with divisor n (one dot product per lag)."""
    n = x.shape[0]
    centered = x - x.mean()
    scale = torch.dot(centered, centered)
    sums = [torch.dot(centered[k:], centered[:n - k]) for k in range(1, lags + 1)]
    return torch.stack(sums) / scale


@torch.no_grad()
def ljung_box(x: Tensor, lags: int) -> tuple[float, Tensor, Tensor]:
    """(Q at ``lags``, cumulative Q_1..Q_lags, r_1..r_lags)."""
    n = x.shape[0]
    r = autocorrelations(x, lags)
    weights = torch.arange(n - 1, n - lags - 1, -1, dtype=torch.float64)
    q = n * (n + 2.0) * (r.square() / weights).cumsum(dim=0)
    return float(q[-1]), q, r


def ljung_box_test(x: Tensor, lags: int, *, label: str,
                   fitted_parameters: int = 0) -> dict[str, Any]:
    """Portmanteau test dict for an estimator's residuals.

    ``df = lags`` as in Stata's wntestq. When ARMA parameters were estimated the
    Box-Pierce correction ``df = lags - fitted_parameters`` (EViews' convention)
    is reported alongside as ``df_adjusted`` / ``p_value_adjusted``.
    """
    statistic, _, _ = ljung_box(x, lags)
    test = {"statistic": statistic, "df": lags, "p_value": chi2_sf(statistic, lags),
            "distribution": "chi2", "label": label, "lags": lags}
    if fitted_parameters and lags > fitted_parameters:
        adjusted = lags - fitted_parameters
        test.update({"df_adjusted": adjusted, "p_value_adjusted": chi2_sf(statistic, adjusted)})
    return test


def _regression_pacf(x: Tensor, lags: int) -> list[float]:
    """phi_kk as the last OLS coefficient of y_t on a constant and k lags, k = 1..lags."""
    n = x.shape[0]
    partial = []
    for k in range(1, lags + 1):
        columns = [torch.ones(n - k, dtype=torch.float64)]
        columns += [x[k - j:n - j] for j in range(1, k + 1)]
        fit = kernel_call(least_squares, torch.stack(columns, dim=1), x[k:].contiguous())
        if k not in fit.kept:
            raise AnalysisError("collinear_lags", f"The regression of the series on {k} lag(s) is "
                                "rank deficient (the series repeats exactly), so the regression "
                                "partial autocorrelation is undefined; use pacf='yw'.")
        partial.append(float(fit.beta[-1]))
    return partial


def corrgram(data: Any, y: str, *, lags: int | None = None, time: str | None = None,
             pacf: str = "yw") -> pd.DataFrame:
    """Correlogram of one series: autocorrelations, partial autocorrelations and Q tests.

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of records.
    y : name of the numeric series.
    lags : number of lags; default ``min(floor(n/2) - 2, 40)`` (Stata's default).
    time : optional time column; rows are sorted by it and integer periods must be
        consecutive. Without it the row order is the time order.
    pacf : ``"yw"`` (default) computes partial autocorrelations by the
        Durbin-Levinson recursion on the autocorrelations (Stata's
        ``corrgram, yw``; EViews; SPSS). ``"regression"`` regresses ``y_t`` on a
        constant and ``k`` lags and reports the last coefficient (Stata's default).

    Returns
    -------
    Table with one row per lag: ``lag``, ``acf`` (divisor n), ``pacf``, ``q``
    (Ljung-Box ``n(n+2) sum_{j<=k} r_j^2/(n-j)``) and ``p_value`` (upper tail of
    chi2 with ``lag`` degrees of freedom). ``attrs`` records ``nobs``, ``lags``,
    the conventions used and ``white_noise_band`` = 1.96/sqrt(n), the usual
    two-standard-error band of the autocorrelations of white noise.

    Stata: ``corrgram y, lags(20)`` (add ``yw`` for the same PACF). EViews:
    View / Correlogram. SPSS: ``ACF`` / ``PACF``.

    Example
    -------
    >>> table = oe.corrgram(df, "inflation", lags=12)
    >>> table[["lag", "acf", "pacf", "q", "p_value"]]
    """
    if pacf not in {"yw", "regression"}:
        raise AnalysisError("invalid_option", "pacf must be 'yw' or 'regression'.")
    x, _ = load_series(data, y, time)
    _require_variation(x, 4, "A correlogram")
    lags = _check_lags(lags, x.shape[0], "corrgram")
    _, q, r = ljung_box(x, lags)
    if pacf == "yw":
        partial = levinson([1.0, *r.tolist()], lags)[1]
    else:
        partial = _regression_pacf(x, lags)
    n = int(x.shape[0])
    return table(
        {"lag": list(range(1, lags + 1)), "acf": r.tolist(), "pacf": partial, "q": q.tolist(),
         "p_value": [chi2_sf(value, k) for k, value in enumerate(q.tolist(), start=1)]},
        title=f"Correlogram of {y}", series=y, nobs=n, lags=lags,
        acf_method="divisor n (Box-Jenkins)",
        pacf_method="Durbin-Levinson (Yule-Walker)" if pacf == "yw"
        else "OLS on a constant and k lags",
        q_method="Ljung-Box, chi2(lag)",
        white_noise_band=normal_isf(0.025) / math.sqrt(n))


def wntestq(data: Any, y: str, *, lags: int | None = None,
            time: str | None = None) -> pd.DataFrame:
    """Portmanteau (Ljung-Box Q) test for white noise (Stata's ``wntestq``).

    ``Q = n(n+2) sum_{j=1..m} r_j^2 / (n-j)`` with ``m = lags`` (default
    ``min(floor(n/2) - 2, 40)``) is chi2(m) under the null hypothesis that the
    series is white noise. Returns a one-row table (``lags``, ``statistic``,
    ``df``, ``p_value``); ``attrs`` repeats them with ``distribution``,
    ``label`` and ``nobs``.

    When the series are residuals of a fitted ARMA model the degrees of freedom
    should be reduced by the number of ARMA parameters; ``oe.arima`` reports
    that adjusted p-value in ``result.tests["ljung_box"]``.

    Stata: ``wntestq y, lags(12)``. EViews: the Q-statistics of a correlogram.

    Example
    -------
    >>> oe.wntestq(df, "returns", lags=10).attrs["p_value"]
    """
    x, _ = load_series(data, y, time)
    _require_variation(x, 4, "The portmanteau test")
    lags = _check_lags(lags, x.shape[0], "wntestq")
    test = ljung_box_test(x, lags, label="Portmanteau (Q) test for white noise")
    test.update({"nobs": int(x.shape[0]), "series": y})
    return _test_table(test, ("lags", "statistic", "df", "p_value"))


@torch.no_grad()
def jarque_bera_test(x: Tensor, *, label: str = "Jarque-Bera normality test") -> dict[str, Any]:
    n = x.shape[0]
    centered = x - x.mean()
    m2 = float(centered.square().mean())
    skewness = float(centered.pow(3).mean()) / m2 ** 1.5
    kurtosis = float(centered.pow(4).mean()) / (m2 * m2)
    statistic = n / 6.0 * (skewness ** 2 + (kurtosis - 3.0) ** 2 / 4.0)
    return {"statistic": statistic, "df": 2, "p_value": chi2_sf(statistic, 2),
            "distribution": "chi2", "label": label, "skewness": skewness, "kurtosis": kurtosis,
            "nobs": int(n)}


def jarque_bera(data: Any, y: str) -> pd.DataFrame:
    """Jarque-Bera test of normality for one series.

    ``JB = n/6 * (S^2 + (K - 3)^2 / 4)`` with the moment estimators
    ``S = m3 / m2^1.5`` and ``K = m4 / m2^2``, ``m_j = mean((y - ybar)^j)``
    (divisor n, as EViews' histogram view and the original Jarque-Bera paper
    use). Under normality JB is asymptotically chi2(2). Returns a one-row table
    (``statistic``, ``df``, ``p_value``, ``skewness``, ``kurtosis``); ``attrs``
    repeats them with ``distribution``, ``label`` and ``nobs``. Missing values
    are an error; the row order does not matter.

    Stata has no command for this exact statistic: ``sktest`` reports the
    D'Agostino-Belanger-D'Agostino adjusted test instead, which differs in small
    samples. EViews: View / Descriptive Statistics / Histogram and Stats.

    Example
    -------
    >>> oe.jarque_bera(df, "residual")
    """
    x, _ = load_series(data, y)
    _require_variation(x, 4, "The Jarque-Bera test")
    test = jarque_bera_test(x)
    test["series"] = y
    return _test_table(test, ("statistic", "df", "p_value", "skewness", "kurtosis"))


@torch.no_grad()
def archlm_test(x: Tensor, lags: int, *, label: str = "ARCH-LM test") -> dict[str, Any]:
    """Engle's LM statistic on a tensor: T' R^2 of u_t^2 on a constant and its lags."""
    n = x.shape[0]
    squares = x.square()
    rows = n - lags
    columns = [torch.ones(rows, dtype=torch.float64)] \
        + [squares[lags - j:n - j] for j in range(1, lags + 1)]
    target = squares[lags:].contiguous()
    fit = kernel_call(least_squares, torch.stack(columns, dim=1), target)
    total = float((target - target.mean()).square().sum())
    if not total > 0.0:
        raise AnalysisError("constant_series", "The squared series is constant; the ARCH-LM test "
                            "is undefined.")
    r_squared = max(0.0, 1.0 - float(fit.ssr) / total)
    statistic = rows * r_squared
    return {"statistic": statistic, "df": lags, "p_value": chi2_sf(statistic, lags),
            "distribution": "chi2", "label": label, "lags": lags, "r_squared": r_squared,
            "nobs": int(rows)}


def archlm(data: Any, y: str, *, lags: int | list[int] = 1, time: str | None = None,
           demean: bool = False) -> pd.DataFrame:
    """Engle's Lagrange-multiplier test for ARCH effects in one series.

    The squared series ``u_t^2`` is regressed on a constant and ``p`` lags of
    itself; ``LM = T' R^2`` (``T' = n - p`` observations in the auxiliary
    regression) is chi2(p) under the null of no ARCH. ``u`` is the column
    itself, as when it holds regression residuals; ``demean=True`` uses
    ``y - mean(y)`` (the residuals of a constant-only mean equation).

    ``lags`` is one lag order ``p`` or a list of orders (one test each, like
    Stata's ``estat archlm, lags(1 2 3)``). Returns a table with one row per
    order: ``lags``, ``statistic``, ``df``, ``p_value``, ``r_squared``, ``nobs``.
    ``attrs`` holds ``distribution``, ``label`` and, for a single order, the
    scalar results.

    Stata: ``estat archlm, lags(p)`` after ``regress``. EViews: View / Residual
    Diagnostics / Heteroskedasticity Tests / ARCH.

    Example
    -------
    >>> oe.archlm(df, "returns", lags=4, demean=True)
    """
    if not isinstance(demean, bool):
        raise AnalysisError("invalid_option", "demean must be True or False.")
    orders = lags if isinstance(lags, (list, tuple)) else [lags]
    if not orders or any(not isinstance(p, int) or isinstance(p, bool) or p < 1 for p in orders) \
            or len(set(orders)) != len(orders):
        raise AnalysisError("invalid_lags", "archlm: lags must be a positive integer or a list "
                            "of distinct positive integers.")
    x, _ = load_series(data, y, time)
    needed = 2 * max(orders) + 3
    if x.shape[0] < needed:
        raise AnalysisError("insufficient_observations",
                            f"The ARCH-LM test with {max(orders)} lag(s) needs at least {needed} "
                            "observations.")
    _require_variation(x, 4, "The ARCH-LM test")
    u = x - x.mean() if demean else x
    tests = [archlm_test(u, p) for p in orders]
    columns = ("lags", "statistic", "df", "p_value", "r_squared", "nobs")
    if len(tests) == 1:
        return _test_table({**tests[0], "series": y, "demeaned": demean}, columns)
    return table([[test[name] for name in columns] for test in tests], columns=list(columns),
                 distribution="chi2", label="ARCH-LM test", series=y, demeaned=demean)
