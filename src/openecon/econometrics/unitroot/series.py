"""Unit-root and stationarity tests for one series: dfuller, dfgls, pperron, kpss.

Each function takes a table and the name of one numeric column, runs the
auxiliary least-squares regressions on float64 tensors (QR, never a normal
matrix inverse) and returns a result table whose ``attrs`` carry the scalar
results: ``statistic``, ``p_value``, ``distribution``, ``critical_values``
(``{"1%": .., "5%": .., "10%": ..}``), ``lags``, ``trend``, ``nobs``, ``label``
and ``notes``. p-values and critical values come from the published tables in
``tables.py``; nothing is simulated at run time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot import critical
from openecon.econometrics.unitroot.common import (
    CRITICAL_NAMES, adf_design, autocovariances, check_choice, check_count, check_flag,
    critical_columns, deterministic_columns, load_series, long_run_variance, regression_table,
    require_variation, uncentre,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.engines.distributions import normal_isf, t_cdf, t_isf

_ADF_CASES = {"none": "n", "constant": "c", "trend": "ct", "drift": "c"}
_PP_CASES = {"none": "n", "constant": "c", "trend": "ct"}
# Rows of a nested factorization are processed in blocks of about this many elements.
_BLOCK_ELEMENTS = 1 << 22


def one_series(data: Any, y: Any, time: str | None, what: str,
               minimum: int = 6) -> tuple[Tensor, Any]:
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    values, labels = load_series(data, [y], time)
    x = values[:, 0].contiguous()
    require_variation(x, minimum, what)
    return x, labels


def from_origin(x: Tensor) -> tuple[Tensor, float]:
    """(x - x_1, x_1): the series measured from its first observation.

    Every regression of this module that contains a constant is invariant to
    the level of the series; working from the first observation keeps y_{t-1}
    and the constant numerically distinct when the level is large relative to
    the variation (a series around 1e9 that moves by a few units).
    """
    origin = float(x[0])
    return x - origin, origin


def _positive_variance(fit, what: str) -> None:
    if fit.exact or not fit.s2 > 0.0 or not bool((fit.se > 0).all()):
        raise AnalysisError("perfect_fit", f"{what} fits the series exactly, so its t statistic "
                            "is undefined. The series is deterministic.")


def adf_statistic(x: Tensor, lags: int, case: str, *, name: str = "y",
                  start: int | None = None):
    """(t statistic of y_{t-1}, regression, term names) of one ADF regression."""
    design, target, terms = adf_design(x, lags, case, name=name, start=start)
    fit = ols(design, target, "The Dickey-Fuller regression")
    _positive_variance(fit, "The Dickey-Fuller regression")
    return float(fit.beta[0] / fit.se[0]), fit, terms


@dataclass
class NestedFit:
    """One member of a family of nested ADF regressions (see ``nested_adf``)."""

    lags: int
    nobs: int
    parameters: int
    ssr: float
    coefficient: float          # of y_{t-1}
    statistic: float | None     # its t ratio (None when the fit is exact)
    last_t: float | None        # t ratio of the last lagged difference (None without lags)
    level_ss: float             # sum of squares of y_{t-1} over the sample
    exact: bool                 # residual at rounding level relative to Delta y


@torch.no_grad()
def nested_adf(x: Tensor, maxlag: int, case: str, what: str) -> list[NestedFit]:
    """ADF regressions with 0..maxlag lagged differences on the sample t = maxlag+2..T.

    Lag-order comparisons fit the same regression with 0, 1, ..., maxlag lags on
    one common sample. These regressions are nested, so all of them follow from
    ONE triangular factor. With

        A = [deterministic terms, y_{t-1}, Dy_{t-1}, ..., Dy_{t-maxlag}, Dy_t] = Q R,

    the regression with k lags uses the leading m = d + 1 + k columns:
    R_m = R[:m, :m], Q_m'Dy = R[:m, -1] and SSR = ||R[m:, -1]||^2. The factor is
    accumulated over blocks of rows (R <- qr([R; block])), so the whole family
    costs O(T maxlag^2) time and one block of memory, instead of maxlag + 1
    separate O(T maxlag^2) factorizations of a T-by-maxlag design.
    """
    total = x.shape[0]
    rows = total - 1 - maxlag
    terms = {"n": 0, "c": 1, "ct": 2, "ctt": 3}[case]
    width = terms + maxlag + 2
    if rows < width:
        raise AnalysisError("insufficient_observations", f"{what} has {max(rows, 0)} "
                            f"observation(s) for {width - 1} parameter(s); use fewer lags or a "
                            "longer series.")
    dy = x[1:] - x[:-1]
    factor = torch.empty((0, width), dtype=torch.float64)
    step = max(width, _BLOCK_ELEMENTS // width)
    for low in range(0, rows, step):
        high = min(low + step, rows)
        first, last = maxlag + low, maxlag + high
        columns = []
        if terms:
            trend = torch.arange(first + 1, last + 1, dtype=torch.float64)
            columns.append(torch.ones(high - low, dtype=torch.float64))
            if terms > 1:
                columns.append(trend)
            if terms > 2:
                columns.append(trend.square())
        columns.append(x[first:last])
        columns.extend(dy[first - j:last - j] for j in range(1, maxlag + 1))
        columns.append(dy[first:last])
        block = torch.stack(columns, dim=1)
        factor = torch.linalg.qr(torch.cat([factor, block]), mode="r").R
    # Stata-style screen on the unpivoted factor: |R_jj| is the norm of column j after
    # projecting out the columns to its left.
    pivots = factor.diagonal()[:width - 1].square()
    norms = factor[:, :width - 1].square().sum(dim=0)
    if bool((pivots <= 1e-13 * norms).any()):
        raise AnalysisError("collinear_regressors", f"{what} is rank deficient: the series does "
                            "not vary enough to identify every term (for example it is constant, "
                            "an exact trend, or has too many lags for the sample).")
    target = factor[:, -1]
    scale = float(target.square().sum())
    level_ss = float(factor[:, terms].square().sum())
    fits = []
    for order in range(maxlag + 1):
        m = terms + 1 + order
        triangle = factor[:m, :m]
        beta = torch.linalg.solve_triangular(triangle, target[:m, None], upper=True)[:, 0]
        inverse = torch.linalg.solve_triangular(
            triangle, torch.eye(m, dtype=torch.float64), upper=True)
        spread = inverse.square().sum(dim=1)                  # diagonal of (X'X)^{-1}
        ssr = float(target[m:].square().sum())
        exact = not ssr > 1e-22 * scale
        s2 = ssr / (rows - m)
        statistic = last_t = None
        if not exact and s2 > 0.0:
            statistic = float(beta[terms]) / math.sqrt(s2 * float(spread[terms]))
            if order:
                last_t = float(beta[m - 1]) / math.sqrt(s2 * float(spread[m - 1]))
        fits.append(NestedFit(lags=order, nobs=rows, parameters=m, ssr=ssr,
                              coefficient=float(beta[terms]), statistic=statistic,
                              last_t=last_t, level_ss=level_ss, exact=exact))
    return fits


def dfuller(data: Any, y: str, *, time: str | None = None, lags: int = 0,
            trend: str = "constant", regress: bool = False):
    """Augmented Dickey-Fuller unit-root test (Stata's ``dfuller``).

    Model and estimator
    -------------------
    OLS of the augmented Dickey-Fuller regression on t = lags+2, ..., T:

        Delta y_t = a + d t + b y_{t-1} + sum_{j=1..lags} c_j Delta y_{t-j} + e_t

    The test statistic ``Z(t)`` is the t ratio of ``b`` (classical OLS standard
    error with N - K degrees of freedom). H0: the series has a unit root
    (b = 0); the alternative is stationarity (b < 0), so small (very negative)
    values reject.

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of records.
    y : name of the numeric series.
    time : optional time column. Rows are sorted by it; integer periods must be
        consecutive. Without it the row order is the time order.
    lags : number of lagged differences (default 0, as in Stata).
    trend : deterministic terms, with Stata's option in brackets:
        ``"none"`` (noconstant), ``"constant"`` (default), ``"trend"``
        (constant and linear trend), ``"drift"`` (constant, and the null is a
        random walk WITH drift, under which Z(t) is Student t).
    regress : also return the regression table.

    Reference distribution
    ----------------------
    ``none`` / ``constant`` / ``trend``: the Dickey-Fuller distribution. The
    p-value is MacKinnon's (1994, Tables 3-4) approximate asymptotic p-value,
    as in Stata. ``critical_values`` are MacKinnon's (2010, Tables 1-3)
    finite-sample response surfaces evaluated at N, the number of observations
    in the regression. Stata prints critical values interpolated linearly in N
    in Fuller's table instead: those are reported as
    ``critical_values_fuller`` (they agree with MacKinnon's to about two
    decimals).
    ``drift``: Student t with N - K degrees of freedom (lower tail), as Stata.

    Returns
    -------
    A one-row table (``statistic``, ``p_value``, ``critical_1pct``,
    ``critical_5pct``, ``critical_10pct``) with ``attrs``: ``statistic``,
    ``p_value``, ``distribution``, ``critical_values``,
    ``critical_values_fuller`` (not with ``drift``), ``lags``, ``trend``,
    ``nobs``, ``coefficient`` (b-hat), ``label``, ``notes``. With
    ``regress=True`` a ``TableSet`` with the tables ``test`` and ``regression``
    (terms ``L1.y``, ``LD.y``, ``L2D.y``, ..., ``_trend``, ``Intercept``; the
    trend is t - 1, i.e. 1 at the first difference of the series, as Stata's
    ``_trend``, so its first value in the regression is lags + 1) and the same
    ``attrs``.

    Stata: ``dfuller y, lags(2) trend regress``. EViews: Unit Root Test /
    Augmented Dickey-Fuller.

    Example
    -------
    >>> result = oe.dfuller(df, "gdp", lags=4, trend="trend")
    >>> result.attrs["statistic"], result.attrs["p_value"]
    """
    check_choice(trend, "trend", tuple(_ADF_CASES))
    check_flag(regress, "regress")
    lags = check_count(lags, "lags")
    x, _ = one_series(data, y, time, "The Dickey-Fuller test")
    case = _ADF_CASES[trend]
    origin = 0.0
    if case != "n":
        x, origin = from_origin(x)
    statistic, fit, terms = adf_statistic(x, lags, case, name=y)
    if origin:
        fit = uncentre(fit, len(terms) - 1, {0: origin})
    notes = []
    if trend == "drift":
        p_value = t_cdf(statistic, fit.df)
        values = {level: -t_isf(alpha, fit.df)
                  for level, alpha in (("1%", 0.01), ("5%", 0.05), ("10%", 0.10))}
        distribution = "t"
        notes.append("Under the null of a random walk with drift Z(t) is Student t with N-K "
                     "degrees of freedom (lower tail).")
    else:
        p_value = critical.mackinnon_p(statistic, case, 1)
        values = critical.mackinnon_critical(case, 1, fit.nobs)
        distribution = "Dickey-Fuller"
        notes.append("MacKinnon (1994) approximate p-value; MacKinnon (2010) finite-sample "
                     "critical values. critical_values_fuller: interpolated in Fuller's "
                     "table, as Stata prints them.")
    attrs = {
        "title": f"Augmented Dickey-Fuller test for a unit root in {y}",
        "test": "dfuller", "series": y, "statistic": statistic, "p_value": p_value,
        "distribution": distribution, "critical_values": values, "lags": lags, "trend": trend,
        "nobs": fit.nobs, "coefficient": float(fit.beta[0]),
        "label": "Z(t): augmented Dickey-Fuller t statistic (H0: unit root)", "notes": notes,
    }
    if trend == "drift":
        attrs["df"] = fit.df
    else:
        attrs["critical_values_fuller"] = critical.fuller_tau_critical(case, fit.nobs)
    result = table([[statistic, p_value, *critical_columns(values)]],
                   columns=["statistic", "p_value", *CRITICAL_NAMES], index=["Z(t)"], **attrs)
    if not regress:
        return result
    return TableSet({"test": result, "regression": regression_table(terms, fit)}, **attrs)


def pperron(data: Any, y: str, *, time: str | None = None, lags: int | None = None,
            trend: str = "constant", regress: bool = False):
    """Phillips-Perron unit-root test (Stata's ``pperron``).

    Model and estimator
    -------------------
    OLS of ``y_t = a + d t + rho y_{t-1} + u_t`` on the n = T - 1 observations
    t = 2..T (no lagged differences). Serial correlation in ``u`` is handled by
    correcting the statistics with a Newey-West long-run variance:

        gamma_j  = (1/n) sum_{t>j} u_t u_{t-j}
        lambda^2 = gamma_0 + 2 sum_{j=1..q} (1 - j/(q+1)) gamma_j
        s^2      = SSR / (n - K),      sigma = OLS standard error of rho-hat
        Z(rho)   = n (rho-hat - 1) - (1/2) (n^2 sigma^2 / s^2) (lambda^2 - gamma_0)
        Z(t)     = sqrt(gamma_0 / lambda^2) (rho-hat - 1) / sigma
                   - (1/2) (lambda^2 - gamma_0) (n sigma) / (lambda s)

    (Stata's Methods and formulas; Hamilton 1994, eq. 17.6.8 and 17.6.12).
    H0: unit root; small values reject.

    Parameters
    ----------
    data, y, time : as in ``dfuller``.
    lags : Newey-West truncation ``q``; default ``int(4 (n/100)^(2/9))``.
    trend : ``"none"``, ``"constant"`` (default) or ``"trend"``.
    regress : also return the regression table.

    Reference distribution
    ----------------------
    Z(t) has the Dickey-Fuller distribution: MacKinnon (1994) approximate
    p-value and MacKinnon (2010) critical values at n; the critical values
    Stata prints for Z(t), interpolated linearly in n in Fuller's table, are
    in ``critical_values_fuller``. Z(rho) has the distribution of
    n(rho-hat - 1): its critical values are interpolated linearly in n in
    Fuller's (1976 Table 8.5.1; 1996 Table 10.A.1) table, as Stata prints
    them; no p-value is published for it.

    Returns
    -------
    A two-row table indexed ``Z(rho)``, ``Z(t)`` with ``statistic``,
    ``p_value`` (missing for Z(rho)) and the three critical values. ``attrs``:
    ``statistic`` and ``p_value`` (of Z(t)), ``z_rho``, ``z_t``,
    ``critical_values`` (Z(t)), ``critical_values_fuller`` (Z(t)),
    ``critical_values_rho``, ``lags``, ``trend``, ``nobs``, ``rho``,
    ``long_run_variance``, ``label``, ``notes``. With ``regress=True`` a
    ``TableSet`` with ``test`` and ``regression``.

    Stata: ``pperron y, lags(4) trend``. EViews: Unit Root Test /
    Phillips-Perron (Bartlett kernel, fixed bandwidth).

    Example
    -------
    >>> oe.pperron(df, "gdp", trend="trend").attrs["z_t"]
    """
    check_choice(trend, "trend", tuple(_PP_CASES))
    check_flag(regress, "regress")
    x, _ = one_series(data, y, time, "The Phillips-Perron test")
    case = _PP_CASES[trend]
    n = x.shape[0] - 1
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    lags = check_count(lags, "lags")
    if lags >= n:
        raise AnalysisError("invalid_lags", f"lags must be smaller than the number of "
                            f"observations in the regression ({n}).")
    origin = 0.0
    if case != "n":
        x, origin = from_origin(x)
    extra, extra_names = deterministic_columns(n, case)
    design = torch.stack([x[:-1], *extra], dim=1)
    fit = ols(design, x[1:].contiguous(), "The Phillips-Perron regression")
    _positive_variance(fit, "The Phillips-Perron regression")
    rho, sigma = float(fit.beta[0]), float(fit.se[0])
    gamma0 = fit.ssr / n
    lambda2 = long_run_variance(fit.resid, lags)
    if not lambda2 > 0.0:
        raise AnalysisError("numerical_failure", "The long-run variance estimate is not "
                            "positive; use fewer lags.")
    # Ratios only: a product of two variances would overflow or underflow for series
    # measured in very large or very small units.
    relative = (lambda2 - gamma0) / fit.s2
    z_rho = n * (rho - 1) - 0.5 * (n * sigma) ** 2 * relative
    z_t = math.sqrt(gamma0 / lambda2) * (rho - 1) / sigma \
        - 0.5 * relative * n * sigma * math.sqrt(fit.s2 / lambda2)
    p_value = critical.mackinnon_p(z_t, case, 1)
    values_t = critical.mackinnon_critical(case, 1, n)
    values_rho = critical.fuller_rho_critical(case, n)
    notes = ["Z(t): MacKinnon (1994) approximate p-value and MacKinnon (2010) critical values; "
             "critical_values_fuller are interpolated in Fuller's table, as Stata prints them.",
             "Z(rho): critical values interpolated linearly in n in Fuller's table; no p-value "
             "is tabulated."]
    attrs = {
        "title": f"Phillips-Perron test for a unit root in {y}",
        "test": "pperron", "series": y, "statistic": z_t, "p_value": p_value,
        "distribution": "Dickey-Fuller", "critical_values": values_t,
        "critical_values_fuller": critical.fuller_tau_critical(case, n),
        "critical_values_rho": values_rho, "z_rho": z_rho, "z_t": z_t, "lags": lags,
        "kernel": "bartlett", "trend": trend, "nobs": n, "rho": rho,
        "long_run_variance": lambda2,
        "label": "Phillips-Perron Z(rho) and Z(t) (H0: unit root)", "notes": notes,
    }
    result = table([[z_rho, None, *critical_columns(values_rho)],
                    [z_t, p_value, *critical_columns(values_t)]],
                   columns=["statistic", "p_value", *CRITICAL_NAMES],
                   index=["Z(rho)", "Z(t)"], **attrs)
    if not regress:
        return result
    if origin:
        # y_t - c = a* + rho (y_{t-1} - c) + ...  =>  a = a* + c - c rho.
        fit = uncentre(fit, len(extra_names), {0: origin}, constant=origin)
    return TableSet({"test": result,
                     "regression": regression_table([f"L1.{y}", *extra_names], fit)}, **attrs)


def gls_detrend(x: Tensor, trend: bool) -> tuple[Tensor, Tensor]:
    """Elliott-Rothenberg-Stock GLS detrending; returns (detrended series, coefficients).

    With a* = 1 + c/T (c = -13.5 with a trend, -7 without) the quasi-differenced
    series (y_1, y_2 - a* y_1, ...) is regressed on the quasi-differenced
    deterministic terms; the detrended series is y minus the fitted
    deterministic part in levels.
    """
    total = x.shape[0]
    alpha = 1.0 + (-13.5 if trend else -7.0) / total
    index = torch.arange(1, total + 1, dtype=torch.float64)
    levels = [torch.ones(total, dtype=torch.float64)] + ([index] if trend else [])
    z = torch.stack(levels, dim=1)
    quasi_y = x.clone()
    quasi_y[1:] -= alpha * x[:-1]
    quasi_z = z.clone()
    quasi_z[1:] -= alpha * z[:-1]
    fit = ols(quasi_z, quasi_y, "The GLS detrending regression")
    return x - z @ fit.beta, fit.beta


def dfgls(data: Any, y: str, *, time: str | None = None, maxlag: int | None = None,
          trend: bool = True):
    """DF-GLS unit-root test of Elliott, Rothenberg and Stock (Stata's ``dfgls``).

    Model and estimator
    -------------------
    1. GLS detrending. With ``a* = 1 + c/T`` (``c = -13.5`` with a linear
       trend, ``c = -7`` with a constant only) regress the quasi-differenced
       series ``(y_1, y_2 - a* y_1, ..., y_T - a* y_{T-1})`` on the
       quasi-differenced constant (and trend) and remove the fitted
       deterministic part from the levels: ``y*_t = y_t - d0 - d1 t``.
    2. For every lag order k = 1..maxlag run the Dickey-Fuller regression
       WITHOUT deterministic terms on the common sample t = maxlag+2..T:

           Delta y*_t = b y*_{t-1} + sum_{j=1..k} c_j Delta y*_{t-j} + e_t

       The DF-GLS statistic is the t ratio of ``b``.
    3. Lag choice on that common sample with N = T - maxlag - 1 observations
       and ``s2_k = SSR_k / N``:
       - Ng-Perron sequential t: the largest k whose last lag has |t| > 1.645
         (10% level), else 0;
       - SC:   ``ln s2_k + (k + 1) ln(N) / N``;
       - MAIC: ``ln s2_k + 2 (tau_k + k) / N``, ``tau_k = b_k^2 sum y*_{t-1}^2 / s2_k``
         (Ng and Perron 2001).

    Parameters
    ----------
    data, y, time : as in ``dfuller``.
    maxlag : largest lag order; default Schwert's ``floor(12 (T/100)^0.25)`` with
        T the number of observations of the series (Stata's default, written
        ``(T+1)/100`` in its manual because the sample there is y_0..y_T).
        ``maxlag=0`` reports the single statistic without lagged differences.
    trend : ``True`` (default) detrends with a constant and a linear trend,
        ``False`` with a constant only (Stata's ``notrend``).

    Critical values
    ---------------
    With a trend: Elliott, Rothenberg and Stock (1996, Table I) for T = 50,
    100, 200 and infinity, interpolated linearly at the number of observations
    of the series; the T = 50 row is used below 50 and the asymptotic row
    above 200 (Stata's ``ers`` option, which also gives Stata's 1% value;
    Stata's default 5% and 10% values come from the lag-dependent Cheung-Lai
    (1995) response surface, which is not reproduced here). Without a trend
    the statistic has the Dickey-Fuller distribution of the no-constant case:
    MacKinnon (2010) critical values at N and MacKinnon (1994) p-values. No
    p-value is tabulated for the trend case.

    Returns
    -------
    A table with one row per lag order: ``lags``, ``statistic``, the three
    critical values, ``rmse`` (sqrt(SSR/N)), ``sc``, ``maic``. ``attrs``:
    ``lags`` and ``statistic`` at the MAIC choice, ``lags_seq_t``, ``lags_sc``,
    ``lags_maic``, ``p_value`` (constant case only), ``critical_values``,
    ``maxlag``, ``trend``, ``nobs``, ``label``, ``notes``.

    Stata: ``dfgls y, maxlag(8) ers`` (``notrend`` for ``trend=False``).
    EViews: Unit Root Test / Dickey-Fuller GLS (ERS).

    Example
    -------
    >>> result = oe.dfgls(df, "gdp")
    >>> result.attrs["lags_maic"], result.attrs["statistic"]
    """
    check_flag(trend, "trend")
    x, _ = one_series(data, y, time, "The DF-GLS test", minimum=8)
    total = x.shape[0]
    if maxlag is None:
        maxlag = int(math.floor(12 * (total / 100) ** 0.25))
        maxlag = max(0, min(maxlag, (total - 3) // 2))
    maxlag = check_count(maxlag, "maxlag")
    nobs = total - maxlag - 1
    if nobs <= maxlag + 1:
        raise AnalysisError("invalid_lags", f"maxlag={maxlag} leaves {max(nobs, 0)} "
                            f"observation(s) for {maxlag + 1} parameter(s); use fewer lags or "
                            "a longer series.")
    x, _ = from_origin(x)
    detrended, _ = gls_detrend(x, trend)
    if not float(detrended.square().sum()) > 1e-20 * float((x - x.mean()).square().sum()):
        raise AnalysisError("perfect_fit", "The series is an exact linear trend, so nothing is "
                            "left after GLS detrending; the DF-GLS test is undefined.")
    if trend:
        values = critical.ers_trend_critical(total)
    else:
        values = critical.mackinnon_critical("n", 1, nobs)
    orders = list(range(1, maxlag + 1)) if maxlag else [0]
    rows, last_t, statistics = [], {}, {}
    fits = nested_adf(detrended, maxlag, "n", "The DF-GLS regression")
    for k in orders:
        fit = fits[k]
        if fit.statistic is None:
            raise AnalysisError("perfect_fit", "The DF-GLS regression fits the series exactly, "
                                "so its t statistic is undefined. The series is deterministic.")
        statistic = fit.statistic
        s2 = fit.ssr / nobs
        tau = fit.coefficient ** 2 * fit.level_ss / s2
        sc = math.log(s2) + (k + 1) * math.log(nobs) / nobs
        maic = math.log(s2) + 2 * (tau + k) / nobs
        statistics[k] = statistic
        last_t[k] = fit.last_t
        rows.append([k, statistic, *critical_columns(values), math.sqrt(s2), sc, maic])
    threshold = normal_isf(0.05)
    seq_t = next((k for k in reversed(orders) if k and abs(last_t[k]) > threshold), 0)
    lags_sc = min(rows, key=lambda row: row[6])[0]
    lags_maic = min(rows, key=lambda row: row[7])[0]
    statistic = statistics[lags_maic]
    notes = ["Lag orders are compared on the common sample t = maxlag+2..T."]
    if trend:
        p_value = None
        notes.append("Critical values: Elliott, Rothenberg and Stock (1996, Table I), "
                     f"interpolated at the {total} observations of the series; no p-value is "
                     "tabulated.")
    else:
        p_value = critical.mackinnon_p(statistic, "n", 1)
        notes.append("Constant-only DF-GLS has the Dickey-Fuller no-constant distribution: "
                     "MacKinnon (2010) critical values, MacKinnon (1994) p-value.")
    if seq_t == 0 and maxlag:
        notes.append("Ng-Perron sequential t: no lag is significant at the 10% level (0 lags).")
    return table(
        rows, columns=["lags", "statistic", *CRITICAL_NAMES, "rmse", "sc", "maic"],
        title=f"DF-GLS test for a unit root in {y}", test="dfgls", series=y,
        statistic=statistic, p_value=p_value,
        distribution="DF-GLS (ERS)" if trend else "Dickey-Fuller",
        critical_values=values, lags=lags_maic, lags_maic=lags_maic, lags_sc=lags_sc,
        lags_seq_t=seq_t, maxlag=maxlag, trend="trend" if trend else "constant", nobs=nobs,
        nobs_series=total,
        label="DF-GLS tau (H0: unit root); statistic at the MAIC lag", notes=notes)


def kpss_bandwidth(resid: Tensor) -> int:
    """Automatic Bartlett bandwidth of Newey and West (1994) as in Hobijn et al. (1998).

    n = int(4 (T/100)^(2/9)); s0 = g0 + 2 sum_{j<=n} g_j; s1 = 2 sum_{j<=n} j g_j;
    gamma = 1.1447 ((s1/s0)^2)^(1/3); bandwidth = min(T - 1, int(gamma T^(1/3))).
    """
    total = resid.shape[0]
    count = min(total - 1, int(4 * (total / 100) ** (2 / 9)))
    gamma = autocovariances(resid, count)
    index = torch.arange(1, count + 1, dtype=torch.float64)
    s0 = float(gamma[0] + 2 * gamma[1:].sum())
    s1 = float(2 * (index * gamma[1:]).sum())
    if not abs(s0) > 0.0:
        return 0
    scale = 1.1447 * ((s1 / s0) ** 2) ** (1 / 3)
    return min(total - 1, int(scale * total ** (1 / 3)))


def kpss(data: Any, y: str, *, time: str | None = None, lags: int | None = None,
         trend: bool = False, auto: bool = False):
    """KPSS stationarity test (Kwiatkowski, Phillips, Schmidt and Shin 1992).

    Model and estimator
    -------------------
    H0: the series is stationary around a level (``trend=False``) or around a
    linear trend (``trend=True``); the alternative is a unit root. With ``e_t``
    the OLS residuals of ``y`` on a constant (and trend) and ``S_t`` their
    partial sums,

        eta      = sum_t S_t^2 / T^2
        s^2(l)   = (1/T) sum e_t^2 + (2/T) sum_{j=1..l} (1 - j/(l+1)) sum_t e_t e_{t-j}
        statistic = eta / s^2(l)

    Large values reject stationarity.

    Parameters
    ----------
    data, y, time : as in ``dfuller``.
    lags : largest Bartlett truncation ``l``; default Schwert's
        ``int(12 (T/100)^(1/4))`` (the default of Baum's ``kpss`` for Stata).
        The table reports every truncation 0..lags, as that command does.
    trend : ``False`` tests level stationarity (``kpss y, notrend``); ``True``
        trend stationarity (the Stata command's default).
    auto : choose one bandwidth automatically (Newey and West 1994 as described
        by Hobijn, Franses and Ooms 1998) and report that single row; ``lags``
        must then be left unset.

    Reference distribution
    ----------------------
    Critical values from KPSS (1992, Table 1): level 0.347 / 0.463 / 0.574 /
    0.739 and trend 0.119 / 0.146 / 0.176 / 0.216 at 10% / 5% / 2.5% / 1%. The
    p-value is interpolated linearly between those four points and is therefore
    only known to lie in [0.01, 0.10]; outside it the bound is reported with a
    note.

    Returns
    -------
    A table with one row per truncation: ``lags``, ``statistic``, ``p_value``.
    ``attrs``: ``statistic`` and ``p_value`` at the largest (or automatic)
    truncation, ``lags``, ``critical_values`` (keys ``10%``, ``5%``, ``2.5%``,
    ``1%``), ``trend``, ``nobs``, ``bandwidth`` (``"schwert"``, ``"user"`` or
    ``"auto"``), ``label``, ``notes``.

    Stata: ``kpss y, maxlag(8) notrend`` (Baum's ``kpss``; ``auto`` for the
    automatic bandwidth). EViews: Unit Root Test / Kwiatkowski-Phillips-
    Schmidt-Shin.

    Example
    -------
    >>> oe.kpss(df, "inflation", lags=4).attrs["statistic"]
    """
    check_flag(trend, "trend")
    check_flag(auto, "auto")
    if auto and lags is not None:
        raise AnalysisError("invalid_option", "Give either lags or auto=True, not both.")
    x, _ = one_series(data, y, time, "The KPSS test")
    x, _ = from_origin(x)
    total = x.shape[0]
    case = "trend" if trend else "level"
    extra, _ = deterministic_columns(total, "ct" if trend else "c")
    fit = ols(torch.stack(extra, dim=1), x, "The KPSS regression")
    if fit.exact:
        raise AnalysisError("constant_series", "The series has no variation around its "
                            f"{'trend' if trend else 'mean'}; the KPSS test is undefined.")
    resid = fit.resid
    eta = float(resid.cumsum(dim=0).square().sum()) / total ** 2
    if auto:
        lags, bandwidth = kpss_bandwidth(resid), "auto"
    elif lags is None:
        lags, bandwidth = min(total - 1, int(12 * (total / 100) ** 0.25)), "schwert"
    else:
        bandwidth = "user"
    lags = check_count(lags, "lags")
    if lags >= total:
        raise AnalysisError("invalid_lags", f"lags must be smaller than the number of "
                            f"observations ({total}).")
    # s^2(l) = g0 + 2 sum_{j<=l} (1 - j/(l+1)) g_j for every l at once, from the running
    # sums of g_j and j g_j.
    gamma = autocovariances(resid, lags)
    index = torch.arange(1, lags + 1, dtype=torch.float64)
    partial = gamma[1:].cumsum(dim=0) - (index * gamma[1:]).cumsum(dim=0) / (index + 1)
    variances = torch.cat([gamma[:1], gamma[0] + 2 * partial]).tolist()
    rows, notes = [], []
    for order in ([lags] if auto else range(lags + 1)):
        variance = variances[order]
        if not variance > 0.0:
            raise AnalysisError("numerical_failure", "The long-run variance estimate is not "
                                "positive; use fewer lags.")
        statistic = eta / variance
        p_value, note = critical.kpss_p(statistic, case)
        rows.append([order, statistic, p_value])
    statistic, p_value = rows[-1][1], rows[-1][2]
    _, note = critical.kpss_p(statistic, case)
    notes.append("Critical values: Kwiatkowski et al. (1992, Table 1); the p-value is "
                 "interpolated between them and bounded to [0.01, 0.10].")
    if note:
        notes.append(note)
    return table(
        rows, columns=["lags", "statistic", "p_value"],
        title=f"KPSS test for {'trend' if trend else 'level'} stationarity of {y}",
        test="kpss", series=y, statistic=statistic, p_value=p_value,
        distribution="KPSS", critical_values=critical.kpss_critical(case), lags=lags,
        kernel="bartlett", bandwidth=bandwidth, trend="trend" if trend else "constant",
        nobs=total, label="KPSS statistic (H0: stationarity)", notes=notes)


def schwert_maxlag(total: int, parameters: int) -> int:
    """Schwert's rule int(12 (T/100)^(1/4)), capped so that the regression stays estimable."""
    return max(0, min(int(12 * (total / 100) ** 0.25), (total - parameters - 2) // 2))


def adf_lag_choice(x: Tensor, case: str, method: str, maxlag: int) -> int:
    """Lag order of an ADF regression by AIC, BIC or the sequential t rule.

    All candidate orders 0..maxlag are fitted on the common sample
    t = maxlag+2..T (one nested factorization, see ``nested_adf``). ``aic`` /
    ``bic`` minimize ``N ln(SSR/N) + K c`` with ``c = 2`` or ``ln N``; ``t``
    starts at maxlag and stops at the first order whose last lagged difference
    has |t| >= 1.645 (0 if none).
    """
    fits = nested_adf(x, maxlag, case, "The lag-selection regression")
    best, best_value = 0, math.inf
    threshold = normal_isf(0.05)
    for fit in reversed(fits):
        if method == "t":
            if fit.lags == 0 or (fit.last_t is not None and abs(fit.last_t) >= threshold):
                return fit.lags
            continue
        if fit.exact or not fit.ssr > 0.0:
            return fit.lags
        penalty = 2.0 if method == "aic" else math.log(fit.nobs)
        value = fit.nobs * math.log(fit.ssr / fit.nobs) + fit.parameters * penalty
        if value <= best_value:
            best, best_value = fit.lags, value
    return best
