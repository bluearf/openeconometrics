"""Structural-break tests on a linear regression: chow, sbsingle and cusum.

All three work on the regression ``y_t = x_t'b + u_t`` in time order. The
scans over break dates and the recursive residuals are computed from running
cross products of the ORTHONORMALIZED regressors (the Q factor of the full
design) and the full-sample residuals, in blocks of rows: O(T K^3) time,
bounded memory, no loop over observations and no refitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.unitroot import critical, tables
from openecon.econometrics.unitroot.common import (
    check_choice, check_flag, check_fraction, column_names, load_series, period_label,
)
from openecon.engines.distributions import chi2_sf, f_sf
from openecon.engines.linalg import collinear_columns, least_squares

_BLOCK_ELEMENTS = 1 << 22
# A residual sum of squares below this share of the outcome's sum of squares is rounding.
_EXACT = 1e-22


@dataclass
class _Model:
    y: Tensor
    x: Tensor               # kept columns
    terms: list[str]
    omitted: list[str]
    labels: pd.Series
    resid: Tensor           # full-sample OLS residuals
    ssr: float

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def k(self) -> int:
        return int(self.x.shape[1])


def _model(data: Any, y: Any, x: Any, time: str | None, intercept: bool, what: str) -> _Model:
    """Load the regression, omit collinear regressors Stata-style and fit it once."""
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    check_flag(intercept, "intercept")
    if not names and not intercept:
        raise AnalysisError("invalid_spec", "The regression needs at least one regressor or "
                            "an intercept.")
    values, labels = load_series(data, [y, *names], time)
    n = values.shape[0]
    if intercept:
        # With a constant every statistic here is invariant to the level of y and of the
        # regressors (the constant breaks too): centring keeps a regressor with a large
        # level and little variation distinct from the constant.
        values = values - values.mean(dim=0)
    target = values[:, 0].contiguous()
    columns = ([torch.ones((n, 1), dtype=torch.float64)] if intercept else []) + [values[:, 1:]]
    design = torch.cat(columns, dim=1)
    terms = (["Intercept"] if intercept else []) + names
    kept, dropped = kernel_call(collinear_columns, design)
    omitted = [terms[i] for i in dropped]
    design, terms = design[:, kept].contiguous(), [terms[i] for i in kept]
    if not terms:
        raise AnalysisError("collinear_regressors", "No regressor remains after omitting "
                            "collinear columns.")
    if n <= 2 * len(terms):
        raise AnalysisError("insufficient_observations", f"{what} needs more than "
                            f"{2 * len(terms)} observations for {len(terms)} parameter(s).")
    fit = kernel_call(least_squares, design, target, drop_collinear=False)
    ssr = float(fit.ssr)
    if not ssr > _EXACT * float(target.square().sum()):
        raise AnalysisError("perfect_fit", "The regression fits the data exactly; break tests "
                            "are undefined.")
    return _Model(target, design, terms, omitted, labels, fit.resid, ssr)


def _notes(model: _Model) -> list[str]:
    if not model.omitted:
        return []
    return [f"Omitted because of collinearity: {', '.join(model.omitted)}."]


def _subsample_ssr(model: _Model, rows: slice, what: str) -> float:
    x, y = model.x[rows], model.y[rows]
    fit = kernel_call(least_squares, x, y)
    if fit.omitted or x.shape[0] <= x.shape[1]:
        raise AnalysisError("singular_subsample", f"The regressors are collinear or too many in "
                            f"the {what} ({x.shape[0]} observation(s), {x.shape[1]} "
                            "parameter(s)); move the break date or drop a regressor.")
    return float(fit.ssr)


def _break_position(model: _Model, break_at: Any, time: str | None) -> int:
    """0-based position of the first observation of the second regime."""
    n = model.n
    if time is None:
        if not isinstance(break_at, Integral) or isinstance(break_at, bool):
            raise AnalysisError("invalid_break", "Without a time column break_at is the 0-based "
                                "row position of the first observation after the break.")
        position = int(break_at)
    else:
        labels = model.labels
        dated = pd.api.types.is_datetime64_any_dtype(labels.dtype)
        # A number is not a date (pandas would read it as nanoseconds since 1970), and a
        # date, text or a missing value is not a period number.
        number = isinstance(break_at, Real) and not isinstance(break_at, bool)
        if break_at is None or dated == number or (number and break_at != break_at):
            raise AnalysisError("invalid_break", "break_at must be a value of the time column "
                                f"'{time}': " + ("a date such as '1984-01-01'." if dated
                                                 else "a period number."))
        try:
            if dated:
                break_at = pd.Timestamp(break_at)
                if break_at is pd.NaT:
                    raise ValueError("not a date")
                zone = getattr(labels.dt, "tz", None)
                if zone is not None and break_at.tzinfo is None:
                    break_at = break_at.tz_localize(zone)
            position = int((labels < break_at).sum())
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_break", "break_at must be comparable with the values "
                                f"of the time column '{time}'.") from exc
    if not 1 <= position <= n - 1:
        raise AnalysisError("invalid_break", "The break must leave at least one observation "
                            "in each regime.")
    return position


def chow(data: Any, y: str, x: list[str], break_at: Any, *, time: str | None = None,
         intercept: bool = True):
    """Chow tests for a structural break at a known date.

    Model and tests
    ---------------
    ``y_t = x_t'b + u_t`` is fitted on the whole sample (SSR_p, T observations,
    K parameters), on the n1 observations before the break (SSR_1) and on the
    n2 observations from the break on (SSR_2).

    - Breakpoint test (Chow 1960): ``F = [(SSR_p - SSR_1 - SSR_2)/K] /
      [(SSR_1 + SSR_2)/(T - 2K)]`` ~ F(K, T - 2K) under H0 of equal coefficients
      and equal error variance in both regimes.
    - Wald: ``K F`` ~ chi2(K) (EViews' Wald statistic; Stata's
      ``estat sbknown`` reports this chi2 form computed with the fitted model's
      VCE). LR: ``T ln(SSR_p / (SSR_1 + SSR_2))`` ~ chi2(K) (EViews' log
      likelihood ratio).
    - Forecast test (Chow's predictive test): ``F = [(SSR_p - SSR_1)/n2] /
      [SSR_1/(n1 - K)]`` ~ F(n2, n1 - K); it is valid even when the second
      regime has too few observations to be fitted.

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of records.
    y, x : outcome column and list of regressor columns (``x=[]`` tests a shift
        in the mean).
    break_at : first observation of the second regime: a value of the time
        column when ``time`` is given (rows with ``time >= break_at`` form the
        second regime; a period number for numeric time, a date or a date
        string such as ``"1984-01-01"`` for datetimes), otherwise a 0-based
        integer row position.
    time : optional time column (rows are sorted by it; no gaps).
    intercept : include a constant (default True); it is allowed to break too.

    Collinear regressors are omitted left to right, Stata style, and listed in
    ``attrs["omitted"]`` and the notes. A regime in which the remaining
    regressors are collinear makes the breakpoint test undefined: its rows are
    then missing and the note says why; the forecast test only needs the first
    regime.

    Returns
    -------
    A table indexed ``chow_f``, ``wald``, ``lr``, ``forecast_f`` with
    ``statistic``, ``df``, ``df2``, ``p_value``, ``distribution``. ``attrs``:
    ``statistic``, ``df``, ``df2``, ``p_value`` (the breakpoint F test),
    ``break_period``, ``break_index``, ``nobs``, ``nobs_1``, ``nobs_2``,
    ``ssr``, ``ssr_1``, ``ssr_2``, ``terms``, ``omitted``, ``label``, ``notes``.

    Stata: ``regress y x`` then ``estat sbknown, break(tq(1984q1))``. EViews:
    View / Stability Diagnostics / Chow Breakpoint Test and Chow Forecast Test.

    Example
    -------
    >>> result = oe.chow(df, "consumption", ["income"], 1984, time="year")
    >>> result.loc["chow_f", ["statistic", "p_value"]]
    """
    model = _model(data, y, x, time, intercept, "The Chow test")
    position = _break_position(model, break_at, time)
    n, k = model.n, model.k
    n1, n2 = position, n - position
    notes = _notes(model)
    if n1 <= k:
        raise AnalysisError("singular_subsample", f"The first regime has {n1} observation(s) "
                            f"for {k} parameter(s); move the break date.")
    ssr1 = _subsample_ssr(model, slice(0, position), "first regime")
    rows: list[list[Any]] = []
    ssr2 = None
    try:
        ssr2 = _subsample_ssr(model, slice(position, n), "second regime")
    except AnalysisError as exc:
        if exc.code != "singular_subsample":
            raise
        notes.append("The breakpoint tests are undefined: " + str(exc))
    main = {"statistic": None, "df": k, "df2": n - 2 * k, "p_value": None}
    scale = _EXACT * float(model.y.square().sum())
    if ssr2 is not None:
        unrestricted = ssr1 + ssr2
        if not unrestricted > scale:
            raise AnalysisError("perfect_fit", "The regime regressions fit the data exactly; "
                                "the Chow test is undefined.")
        f_stat = ((model.ssr - unrestricted) / k) / (unrestricted / (n - 2 * k))
        f_stat = max(f_stat, 0.0)
        lr = max(n * math.log(model.ssr / unrestricted), 0.0)
        main.update(statistic=f_stat, p_value=f_sf(f_stat, k, n - 2 * k))
        rows.append([f_stat, k, n - 2 * k, main["p_value"], "F"])
        rows.append([k * f_stat, k, None, chi2_sf(k * f_stat, k), "chi2"])
        rows.append([lr, k, None, chi2_sf(lr, k), "chi2"])
    else:
        rows.extend([[None, k, n - 2 * k, None, "F"], [None, k, None, None, "chi2"],
                     [None, k, None, None, "chi2"]])
    if not ssr1 > scale:
        raise AnalysisError("perfect_fit", "The first-regime regression fits exactly; the "
                            "forecast test is undefined.")
    forecast = max(((model.ssr - ssr1) / n2) / (ssr1 / (n1 - k)), 0.0)
    forecast_p = f_sf(forecast, n2, n1 - k)
    rows.append([forecast, n2, n1 - k, forecast_p, "F"])
    return table(
        rows, columns=["statistic", "df", "df2", "p_value", "distribution"],
        index=["chow_f", "wald", "lr", "forecast_f"],
        title=f"Chow tests for a break in the regression of {y}", test="chow",
        statistic=main["statistic"], df=k, df2=n - 2 * k, p_value=main["p_value"],
        distribution="F", forecast_statistic=forecast, forecast_p_value=forecast_p,
        break_period=period_label(model.labels, position), break_index=position, nobs=n,
        nobs_1=n1, nobs_2=n2, ssr=model.ssr, ssr_1=ssr1, ssr_2=ssr2, terms=model.terms,
        omitted=model.omitted, label="Chow breakpoint F test (H0: no break)", notes=notes)


@torch.no_grad()
def _orthonormal(x: Tensor) -> Tensor:
    return torch.linalg.qr(x)[0]


@torch.no_grad()
def wald_scan(q: Tensor, resid: Tensor, ssr: float, first: int, last: int) -> Tensor:
    """Wald statistics of a break after n1 = first..last observations.

    With e the full-sample residuals, h = sum_{t<=n1} q_t e_t and
    G = sum_{t<=n1} q_t q_t' (Q'Q = I, Q'e = 0), the drop in the residual sum of
    squares from splitting the sample is  d = h'(G^{-1} + (I - G)^{-1})h  and

        W(n1) = d / ((SSR - d) / (T - 2K)).
    """
    n, k = q.shape
    block = max(1, _BLOCK_ELEMENTS // (k * k))
    eye = torch.eye(k, dtype=torch.float64)
    gram = torch.zeros((k, k), dtype=torch.float64)
    moment = torch.zeros(k, dtype=torch.float64)
    scores = q * resid[:, None]
    out = torch.empty(last - first + 1, dtype=torch.float64)
    for start in range(0, last, block):
        stop = min(start + block, last)
        rows = q[start:stop]
        grams = gram + torch.einsum("ti,tj->tij", rows, rows).cumsum(dim=0)
        moments = moment + scores[start:stop].cumsum(dim=0)
        gram, moment = grams[-1].clone(), moments[-1].clone()
        low = max(first, start + 1)                 # n1 = index + 1
        if low > stop:
            continue
        g, h = grams[low - start - 1:], moments[low - start - 1:, :, None]
        left, info_left = torch.linalg.cholesky_ex(g)
        right, info_right = torch.linalg.cholesky_ex(eye - g)
        pivots = torch.minimum(left.diagonal(dim1=1, dim2=2).min(dim=1).values,
                               right.diagonal(dim1=1, dim2=2).min(dim=1).values)
        if bool((info_left != 0).any()) or bool((info_right != 0).any()) \
                or bool((pivots < 1e-5).any()):
            raise AnalysisError("singular_subsample", "The regressors are collinear within a "
                                "subsample at some candidate break date (for example a dummy "
                                "that is constant there); increase trim or drop the regressor.")
        drop = (h * torch.cholesky_solve(h, left)).sum(dim=(1, 2)) \
            + (h * torch.cholesky_solve(h, right)).sum(dim=(1, 2))
        # SSR - drop is a difference of sums of squares: below 1e-11 SSR it is rounding,
        # i.e. the two regime regressions fit exactly.
        unrestricted = ssr - drop
        wald = drop / (unrestricted / (n - 2 * k))
        wald[unrestricted <= 1e-11 * ssr] = float("inf")
        out[low - first:stop - first + 1] = wald
    return out


def sbsingle(data: Any, y: str, x: list[str], *, time: str | None = None, trim: float = 0.15,
             test: str = "supwald", intercept: bool = True):
    """Test for one structural break at an unknown date (Stata's ``estat sbsingle``).

    Model and statistics
    --------------------
    For every candidate break (the first regime holding n1 observations) the
    Wald statistic of H0 "all K coefficients are equal in both regimes" is

        W(n1) = (SSR_p - SSR_1 - SSR_2) / ((SSR_1 + SSR_2) / (T - 2K))  =  K F(n1),

    the chi2 form of the Chow test with the classical error variance. The
    candidates are n1 = m, ..., T - m with m = max(K+1, ceil(trim T)): each
    regime keeps at least the trimmed share of the sample (in Stata's manual
    example, T = 222 and 15% trimming give first regimes of 34 to 188
    observations).

    - ``test="supwald"`` (default): ``max W(n1)`` (Quandt 1960; Andrews 1993).
    - ``test="avewald"``: the average of W over the candidates;
      ``test="expwald"``: ``ln( mean exp(W/2) )`` (Andrews and Ploberger 1994).

    Parameters
    ----------
    data, y, x, time, intercept : as in ``oe.chow``.
    trim : fraction of the sample excluded at each end (default 0.15).
    test : ``"supwald"``, ``"avewald"`` or ``"expwald"``.

    Reference distribution
    ----------------------
    sup-Wald: with 15% trimming and K <= 20 the 10%, 5% and 1% critical values
    are those of Andrews (1993) as corrected in Andrews (2003), in the form
    tabulated by Stock and Watson (this table was checked by simulation only,
    which the notes of the result repeat). The p-value is the upper-tail
    approximation of DeLong (1981) quoted in Andrews (1993), evaluated at the
    actual trimming; it is approximate: within about 0.02 of simulated
    p-values at 0.05 and 0.10, within 0.08 at 0.5, and too small above 0.5
    (a note says so). Stata reports Hansen's (1997) approximation instead,
    whose coefficient tables are not reproduced here. With a single candidate
    date the statistic is an ordinary Wald statistic and the p-value is the
    chi-squared(K) tail. Average and exponential Wald: the statistic only
    (p-value and critical values are missing, with a note).

    Returns
    -------
    A one-row table (``statistic``, ``p_value``, three critical values,
    ``break_period``). ``attrs``: ``statistic``, ``p_value``, ``df`` (K),
    ``critical_values``, ``break_period`` and ``break_index`` (first
    observation of the second regime at the largest W; reported for every
    test), ``sup_wald``, ``ave_wald``, ``exp_wald``, ``trim``, ``candidates``,
    ``nobs``, ``terms``, ``omitted``, ``label``, ``notes``.

    Stata: ``regress y x`` then ``estat sbsingle, trim(15) swald``
    (``awald`` / ``ewald``). EViews: Quandt-Andrews Breakpoint Test.

    Example
    -------
    >>> result = oe.sbsingle(df, "inflation", ["unemployment"], time="quarter")
    >>> result.attrs["break_period"], result.attrs["p_value"]
    """
    check_choice(test, "test", ("supwald", "avewald", "expwald"))
    trim = check_fraction(trim, "trim")
    model = _model(data, y, x, time, intercept, "The break test")
    n, k = model.n, model.k
    edge = max(k + 1, math.ceil(trim * n - 1e-9))
    first, last = edge, n - edge
    if first > last:
        raise AnalysisError("insufficient_observations", "No candidate break date remains "
                            "after trimming; lower trim or drop regressors.")
    wald = wald_scan(_orthonormal(model.x), model.resid, model.ssr, first, last)
    if not bool(torch.isfinite(wald).all()):
        raise AnalysisError("perfect_fit", "A regime regression fits exactly at some candidate "
                            "break date; the break test is undefined.")
    best = int(torch.argmax(wald))
    sup = float(wald[best])
    average = float(wald.mean())
    half = wald / 2
    exponential = float(half.max() + torch.log(torch.exp(half - half.max()).mean()))
    position = first + best
    notes = _notes(model)
    values = p_value = None
    if test == "supwald":
        statistic = sup
        p_value = critical.supwald_p(sup, k, first / n, last / n)
        values = critical.supwald_critical(k, float(trim))
        if first == last:
            notes.append("Only one candidate break date remains after trimming: the p-value "
                         f"is the chi-squared({k}) tail of an ordinary Wald test.")
        else:
            notes.append("p-value: tail approximation of DeLong (1981) / Andrews (1993), "
                         "approximate and slightly conservative.")
            if p_value > 0.5:
                notes.append("The tail approximation understates p-values above about 0.5: "
                             "the true p-value is larger than the one reported.")
        if values is None:
            notes.append("Critical values are tabulated for 15% trimming and at most "
                         f"{len(tables.QLR_F_15)} parameters only (Andrews 2003).")
        else:
            notes.append("Critical values: Andrews (2003) as tabulated by Stock and Watson; "
                         "this table was checked by simulation of the limiting process only "
                         "(critical values not independently verified against a second "
                         "published transcription).")
    else:
        statistic = average if test == "avewald" else exponential
        notes.append("No p-value or critical values: the Andrews-Ploberger (1994) and Hansen "
                     "(1997) tables are not reproduced in OpenEconometrics.")
    labels = {"supwald": "Supremum Wald", "avewald": "Average Wald",
              "expwald": "Exponential Wald"}
    period = period_label(model.labels, position)
    row = [statistic, p_value, *(values[level] if values else None
                                 for level in ("1%", "5%", "10%")), period]
    return table(
        [row], columns=["statistic", "p_value", "critical_1pct", "critical_5pct",
                        "critical_10pct", "break_period"], index=[test],
        title=f"Test for a structural break at an unknown date in the regression of {y}",
        test="sbsingle", statistic=statistic, p_value=p_value, df=k,
        distribution="sup-Wald (Andrews)" if test == "supwald" else labels[test],
        critical_values=values, break_period=period, break_index=position, sup_wald=sup,
        ave_wald=average, exp_wald=exponential, trim=float(trim), candidates=last - first + 1,
        first_candidate=period_label(model.labels, first),
        last_candidate=period_label(model.labels, last), nobs=n, terms=model.terms,
        omitted=model.omitted, label=f"{labels[test]} test (H0: no structural break)",
        notes=notes)


@torch.no_grad()
def recursive_residuals(q: Tensor, y: Tensor) -> tuple[Tensor, int]:
    """Standardized recursive residuals w_t, t = start+1..T, and ``start``.

    w_t = (y_t - z_t'b_{t-1}) / sqrt(1 + z_t'(Z_{t-1}'Z_{t-1})^{-1} z_t) with
    b_{t-1} from the first t-1 observations (Brown, Durbin and Evans 1975);
    ``start`` is the first sample size at which the regressors have full rank
    (K unless early observations are collinear).
    """
    n, k = q.shape
    block = max(1, _BLOCK_ELEMENTS // (k * k))
    gram = torch.zeros((k, k), dtype=torch.float64)
    moment = torch.zeros(k, dtype=torch.float64)
    pieces: list[Tensor] = []
    start = None
    for begin in range(0, n - 1, block):
        stop = min(begin + block, n - 1)            # rows used as "history" end at stop
        rows = q[begin:stop]
        grams = gram + torch.einsum("ti,tj->tij", rows, rows).cumsum(dim=0)
        moments = moment + (rows * y[begin:stop, None]).cumsum(dim=0)
        gram, moment = grams[-1].clone(), moments[-1].clone()
        low = max(begin, k - 1)                     # history of index+1 >= k observations
        if low >= stop:
            continue
        g, h = grams[low - begin:], moments[low - begin:, :, None]
        factor, info = torch.linalg.cholesky_ex(g)
        diagonal = factor.diagonal(dim1=1, dim2=2)
        good = (info == 0) & (diagonal.min(dim=1).values
                              > 1e-5 * diagonal.max(dim=1).values)
        if start is None:
            usable = good.nonzero().flatten()
            if not len(usable):
                continue
            offset = int(usable[0])
            start = low + offset + 1
            g, h, factor, good = g[offset:], h[offset:], factor[offset:], good[offset:]
            low += offset
        if not bool(good.all()):
            raise AnalysisError("numerical_failure", "The recursive regressions became "
                                "numerically singular; rescale or drop a regressor.")
        nxt = q[low + 1:stop + 1, :, None]                       # the next observation
        beta = torch.cholesky_solve(h, factor)
        leverage = (nxt * torch.cholesky_solve(nxt, factor)).sum(dim=(1, 2))
        error = y[low + 1:stop + 1] - (nxt * beta).sum(dim=(1, 2))
        pieces.append(error / (1 + leverage).sqrt())
    if start is None or not pieces:
        raise AnalysisError("singular_subsample", "The regressors are collinear in every "
                            "initial subsample; recursive residuals are undefined.")
    return torch.cat(pieces), start


def cusum(data: Any, y: str, x: list[str], *, time: str | None = None, intercept: bool = True,
          level: str = "5%"):
    """CUSUM and CUSUM-of-squares tests of parameter stability (Brown, Durbin, Evans 1975).

    Model and statistics
    --------------------
    The regression ``y_t = x_t'b + u_t`` is fitted recursively. The recursive
    residual of observation t is the standardized one-step prediction error

        w_t = (y_t - x_t'b_{t-1}) / sqrt(1 + x_t'(X_{t-1}'X_{t-1})^{-1} x_t),  t = K+1..T,

    with ``b_{t-1}`` estimated from the first t-1 observations. Under H0 of
    constant coefficients the w_t are independent N(0, sigma^2).

    - CUSUM: ``W_t = sum_{j<=t} w_j / s`` with
      ``s^2 = sum_j (w_j - mean(w))^2 / (T - K)``, the variance of the
      recursive residuals about their mean as in Stata's ``estat sbcusum``
      (Brown, Durbin and Evans divide the uncentred ``sum w_j^2 = SSR`` by
      T - K; ``attrs["sigma_ols"]`` holds that value). The bounds are the
      lines through ``(K, +-a sqrt(T-K))`` and ``(T, +-3a sqrt(T-K))`` where
      ``a`` solves ``Q(3a) + exp(-4a^2)(1 - Q(a)) = alpha/2``: 0.9479 (5%),
      1.1430 (1%), 0.8499 (10%). The test statistic is
      ``max_t |W_t| / (sqrt(T-K) (1 + 2 (t-K)/(T-K)))``, to be compared with
      ``a`` (Stata's ``estat sbcusum`` statistic).
    - CUSUM of squares: ``S_t = sum_{j<=t} w_j^2 / sum_j w_j^2`` with bounds
      ``(t-K)/(T-K) +- c0``; ``c0`` is the Edgerton-Wells (1994) approximation
      to Durbin's (1969) critical value with m = (T-K)/2 - 1.

    Parameters
    ----------
    data, y, x, time, intercept : as in ``oe.chow``.
    level : significance level of the bounds: ``"5%"`` (default), ``"1%"`` or
        ``"10%"``.

    Returns
    -------
    A table with one row per recursive residual: ``period`` (the value of
    the time column, or the 0-based row position without one),
    ``recursive_residual``, ``cusum``, ``cusum_lower``, ``cusum_upper``,
    ``cusumsq``, ``cusumsq_lower``, ``cusumsq_upper``. ``attrs``:
    ``statistic`` (the CUSUM statistic above), ``critical_values`` (a at 1%,
    5%, 10%), ``p_value`` (None), ``cusum_crosses`` and ``cusumsq_crosses``
    (whether the path leaves the bounds at ``level``), ``cusumsq_statistic``
    (max |S_t - (t-K)/(T-K)|), ``cusumsq_critical``, ``sigma``, ``sigma_ols``,
    ``nobs``, ``n_recursive``, ``start``, ``terms``, ``omitted``, ``level``,
    ``label``, ``notes``.

    Stata: ``estat sbcusum`` (recursive residuals) after ``regress``; the
    community command ``cusum6`` draws both paths. EViews: View / Stability
    Diagnostics / Recursive Estimates (CUSUM Test, CUSUM of Squares Test).

    Example
    -------
    >>> path = oe.cusum(df, "consumption", ["income"], time="year")
    >>> path.attrs["cusum_crosses"], path.attrs["cusumsq_crosses"]
    """
    check_choice(level, "level", ("1%", "5%", "10%"))
    model = _model(data, y, x, time, intercept, "The CUSUM test")
    n, k = model.n, model.k
    w, start = recursive_residuals(_orthonormal(model.x), model.y)
    count = int(w.shape[0])
    notes = _notes(model)
    if start > k:
        notes.append(f"The first {start} observations were needed to identify the {k} "
                     "coefficients (early observations are collinear).")
    if count < 3:
        raise AnalysisError("insufficient_observations", "Too few recursive residuals for the "
                            "CUSUM tests.")
    total = float(w.square().sum())
    sigma = math.sqrt(float((w - w.mean()).square().sum()) / count)
    if not (sigma > 0.0 and total > 1e-22 * float(model.y.square().sum())):
        raise AnalysisError("perfect_fit", "The recursive residuals do not vary; the CUSUM "
                            "tests are undefined.")
    step = torch.arange(1, count + 1, dtype=torch.float64)
    path = w.cumsum(dim=0) / sigma
    values = {name: critical.cusum_critical(alpha)
              for name, alpha in (("1%", 0.01), ("5%", 0.05), ("10%", 0.10))}
    a = values[level]
    bound = a * math.sqrt(count) + 2 * a * step / math.sqrt(count)
    statistic = float((path.abs() / (math.sqrt(count) * (1 + 2 * step / count))).max())
    squares = w.square().cumsum(dim=0) / total
    expected = step / count
    c0 = critical.cusumsq_critical(count, level) if count > 6 else None
    deviation = float((squares - expected).abs().max())
    if c0 is None or not c0 > 0:
        c0 = None
        notes.append("Too few recursive residuals for the CUSUM-of-squares critical value.")
    lower = expected - c0 if c0 is not None else torch.full_like(expected, float("nan"))
    upper = expected + c0 if c0 is not None else torch.full_like(expected, float("nan"))
    labels = model.labels.iloc[start:].reset_index(drop=True)
    frame = {
        "period": labels.to_numpy(),             # the time values themselves (or positions)
        "recursive_residual": w.numpy(), "cusum": path.numpy(),
        "cusum_lower": (-bound).numpy(), "cusum_upper": bound.numpy(),
        "cusumsq": squares.numpy(), "cusumsq_lower": lower.numpy(),
        "cusumsq_upper": upper.numpy(),
    }
    notes.append("CUSUM bounds: Brown, Durbin and Evans (1975); CUSUM-of-squares bounds: "
                 "Edgerton and Wells (1994) approximation.")
    return table(
        frame, title=f"CUSUM tests of parameter stability in the regression of {y}",
        test="cusum", statistic=statistic, p_value=None,
        distribution="CUSUM (Brown-Durbin-Evans)",
        critical_values=values, cusum_crosses=bool(statistic > a),
        cusumsq_statistic=deviation, cusumsq_critical=c0,
        cusumsq_crosses=None if c0 is None else bool(deviation > c0), sigma=sigma,
        sigma_ols=math.sqrt(total / count), nobs=n,
        n_recursive=count, start=start, level=level, terms=model.terms, omitted=model.omitted,
        label="Recursive CUSUM statistic (H0: constant coefficients)", notes=notes)
