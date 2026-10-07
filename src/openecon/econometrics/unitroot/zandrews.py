"""Zivot-Andrews unit-root test with one endogenous structural break.

The test runs one augmented Dickey-Fuller regression per candidate break date.
Refitting T regressions costs O(T^2 K^2); here the regressors that do not
depend on the break date are orthogonalized once (QR), and the cross products
of the break dummies with everything else are reverse cumulative sums, so the
whole scan costs O(T K) plus one 2-by-2 solve per date.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.unitroot import critical
from openecon.econometrics.unitroot.common import (
    CRITICAL_NAMES, adf_design, check_choice, check_count, check_fraction, critical_columns,
    period_label, regress,
)
from openecon.econometrics.unitroot.series import (
    adf_lag_choice, from_origin, one_series, schwert_maxlag,
)

_MODELS = ("intercept", "trend", "both")


@torch.no_grad()
def break_scan(x: Tensor, lags: int, model: str) -> tuple[Tensor, int]:
    """t statistics of y_{t-1} for every break position of the Zivot-Andrews regression.

    Rows are t = lags+2..T. Entry ``c`` (c = 1..n-1) is the statistic when the
    first ``c`` rows precede the break, i.e. the break date is observation
    TB = c + lags + 1 (1-based) with DU_t = 1(t > TB) and DT_t = (t - TB) DU_t.
    Entries whose break regressors are collinear with the rest are +inf, and
    entries whose regression fits exactly (residual sum of squares at the
    rounding level of the no-break one) are NaN. Returns the statistics (entry
    0 is +inf) and the regression's residual degrees of freedom.

    With X0 = [1, t, lagged differences, y_{t-1}] = Q R and D the break
    columns: A = Q'D, G = D'D - A'A, g = G^{-1} D'r0 (r0 the residual of the
    regression without a break),

        SSR = r0'r0 - (D'r0)'g,
        t   = sign(R_kk) (q_k'dy - a_k'g) / sqrt(s^2 (1 + a_k'G^{-1}a_k)),

    where q_k is the last column of Q (the one of y_{t-1}) and a_k = D'q_k.
    """
    design, target, _ = adf_design(x, lags, "ct")
    # y_{t-1} last: its coefficient is then theta_k / R_kk.
    design = torch.cat([design[:, 1:], design[:, :1]], dim=1)
    n, k = design.shape
    q, r = torch.linalg.qr(design)
    theta = q.T @ target
    base_resid = target - q @ theta
    position = torch.arange(1, n + 1, dtype=torch.float64)

    def tail_sums(values: Tensor) -> Tensor:
        return values.flip(0).cumsum(dim=0).flip(0)

    count = torch.arange(n, dtype=torch.float64)                 # c for entry c
    after = n - count                                            # rows on or after the break
    q0, r0 = tail_sums(q), tail_sums(base_resid)
    blocks, cross, gram = [], [], {}
    if model in {"intercept", "both"}:
        blocks.append(q0)
        cross.append(r0)
        gram[("du", "du")] = after
    if model in {"trend", "both"}:
        q1, r1 = tail_sums(q * position[:, None]), tail_sums(base_resid * position)
        blocks.append(q1 - count[:, None] * q0)
        cross.append(r1 - count * r0)
        gram[("dt", "dt")] = after * (after + 1) * (2 * after + 1) / 6
        gram[("du", "dt")] = after * (after + 1) / 2
    a = torch.stack(blocks, dim=2)                               # [n, k, m]
    b = torch.stack(cross, dim=1)                                # [n, m]
    m = a.shape[2]
    dd = torch.zeros((n, m, m), dtype=torch.float64)
    if model == "intercept":
        dd[:, 0, 0] = gram[("du", "du")]
    elif model == "trend":
        dd[:, 0, 0] = gram[("dt", "dt")]
    else:
        dd[:, 0, 0], dd[:, 1, 1] = gram[("du", "du")], gram[("dt", "dt")]
        dd[:, 0, 1] = dd[:, 1, 0] = gram[("du", "dt")]
    g = dd - a.transpose(1, 2) @ a
    scale = dd.diagonal(dim1=1, dim2=2).clamp_min(1e-300)
    if m == 1:
        determinant, relative = g[:, 0, 0], g[:, 0, 0] / scale[:, 0]
    else:
        determinant = g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] * g[:, 1, 0]
        relative = determinant / (scale[:, 0] * scale[:, 1])
    usable = relative > 1e-10
    usable[0] = False
    safe = torch.where(usable[:, None, None], g, torch.eye(m, dtype=torch.float64).expand_as(g))
    rhs = torch.cat([b[:, :, None], a[:, k - 1, :, None]], dim=2)   # D'r0 and a_k
    solved = torch.linalg.solve(safe, rhs)
    gamma, weight = solved[:, :, 0], solved[:, :, 1]
    a_k = a[:, k - 1, :]
    df = n - k - m
    base_ssr = base_resid.square().sum()
    ssr = base_ssr - (b * gamma).sum(dim=1)
    variance = (ssr / df).clamp_min(0.0) * (1 + (a_k * weight).sum(dim=1))
    numerator = torch.sign(r[k - 1, k - 1]) * (theta[k - 1] - (a_k * gamma).sum(dim=1))
    statistic = numerator / variance.sqrt()
    statistic = torch.where(usable & (variance > 0), statistic,
                            torch.full_like(statistic, float("inf")))
    # SSR is a difference of two sums of squares: below 1e-12 of the larger it is rounding.
    exact = usable & (ssr <= 1e-12 * base_ssr)
    statistic = torch.where(exact, torch.full_like(statistic, float("nan")), statistic)
    return statistic, df


def zandrews(data: Any, y: str, *, time: str | None = None, break_: str = "intercept",
             trim: float = 0.15, lags: int | str = 0, maxlag: int | None = None):
    """Zivot-Andrews unit-root test allowing one endogenous structural break.

    Model and estimator
    -------------------
    For every candidate break date TB the augmented Dickey-Fuller regression

        Delta y_t = mu + beta t + theta DU_t + gamma DT_t + alpha y_{t-1}
                    + sum_{j=1..k} c_j Delta y_{t-j} + e_t,
        DU_t = 1(t > TB),   DT_t = (t - TB) 1(t > TB),

    is fitted by OLS on t = k+2..T. ``break_="intercept"`` (Zivot and
    Andrews's model A) includes DU, ``"trend"`` (model B) includes DT,
    ``"both"`` (model C) includes both. The statistic is the smallest t ratio
    of ``alpha`` over the candidate dates; H0 is a unit root without a break,
    the alternative a trend-stationary series with one break at an unknown
    date. Small (very negative) values reject.

    Parameters
    ----------
    data, y, time : as in ``oe.dfuller``.
    break_ : ``"intercept"`` (default), ``"trend"`` or ``"both"``.
    trim : fraction of the sample excluded at each end (default 0.15):
        candidates are TB = int(trim T)+1, ..., T - int(trim T), further limited
        to dates with observations on both sides inside the regression sample.
    lags : number of lagged differences k (default 0), or ``"aic"``, ``"bic"``
        or ``"t"`` to choose k once from the ADF regression with constant and
        trend but no break (as Baum's ``zandrews`` and statsmodels do): the
        orders 0..maxlag are compared on a common sample; ``"t"`` takes the
        largest order whose last lag has |t| >= 1.645.
    maxlag : largest order for the automatic choice; default Schwert's
        ``int(12 (T/100)^(1/4))``.

    Critical values
    ---------------
    Asymptotic critical values of Zivot and Andrews (1992): model A (Table 2)
    -5.34 / -4.80 / -4.58, model B (Table 3) -4.93 / -4.42 / -4.11, model C
    (Table 4) -5.57 / -5.08 / -4.82 at 1% / 5% / 10%. No p-value is tabulated.

    Returns
    -------
    A one-row table (``statistic``, ``p_value`` (missing), the three critical
    values, ``break_period``, ``lags``). ``attrs``: ``statistic``,
    ``p_value`` (None), ``critical_values``, ``break_period`` (the label of
    observation TB, the last one before the break), ``break_index`` (its
    0-based position), ``lags``, ``break``, ``trim``, ``nobs``, ``candidates``,
    ``label``, ``notes``.

    Stata: ``zandrews y, break(intercept) trim(0.15) lagmethod(AIC)``
    (Baum's ``zandrews``). EViews: Breakpoint Unit Root Test (innovational
    outlier, Dickey-Fuller min-t).

    Example
    -------
    >>> result = oe.zandrews(df, "gnp", break_="both", lags="aic")
    >>> result.attrs["statistic"], result.attrs["break_period"]
    """
    check_choice(break_, "break_", _MODELS)
    trim = check_fraction(trim, "trim")
    x, labels = one_series(data, y, time, "The Zivot-Andrews test", minimum=12)
    x, _ = from_origin(x)
    total = x.shape[0]
    if isinstance(lags, str):
        check_choice(lags, "lags", ("aic", "bic", "t"))
        limit = schwert_maxlag(total, 6) if maxlag is None else check_count(maxlag, "maxlag")
        if total - limit - 1 <= limit + 5:
            raise AnalysisError("invalid_lags", f"maxlag={limit} is too large for {total} "
                                "observations.")
        method, lags = lags, adf_lag_choice(x, "ct", lags, limit)
    else:
        lags = check_count(lags, "lags")
        if maxlag is not None:
            raise AnalysisError("invalid_option", "maxlag applies only with lags='aic', 'bic' "
                                "or 't'.")
        method = "fixed"
    nobs = total - lags - 1
    width = 2 if break_ == "both" else 1
    if nobs <= lags + 3 + width + 1:
        raise AnalysisError("insufficient_observations", f"The Zivot-Andrews regression with "
                            f"{lags} lag(s) needs more than {2 * lags + 5 + width} observations.")
    # Rank and exact-fit problems of the regression without a break surface here.
    design, target, _ = adf_design(x, lags, "ct")
    base = regress(design, target, "The Dickey-Fuller regression")
    if base.exact:
        raise AnalysisError("perfect_fit", "The Dickey-Fuller regression fits the series "
                            "exactly; the series is deterministic.")
    statistics, df = break_scan(x, lags, break_)
    edge = int(trim * total)
    first = max(edge + 1, lags + 3)                 # TB, 1-based
    last = min(total - edge, total - 2)
    if first > last:
        raise AnalysisError("insufficient_observations", "No candidate break date remains "
                            "after trimming; lower trim or use fewer lags.")
    window = statistics[first - lags - 1:last - lags]
    if bool(torch.isnan(window).any()):
        raise AnalysisError("perfect_fit", "At a candidate break date the regression fits the "
                            "series exactly: the series is a deterministic break, and the "
                            "Zivot-Andrews statistic is undefined.")
    best = int(torch.argmin(window))
    statistic = float(window[best])
    if not torch.isfinite(window[best]):
        raise AnalysisError("collinear_regressors", "The break regressors are collinear with "
                            "the trend at every candidate date; the test is undefined.")
    break_index = first + best - 1                  # 0-based position of observation TB
    values = critical.zivot_andrews_critical(break_)
    period = period_label(labels, break_index)
    notes = ["Asymptotic critical values of Zivot and Andrews (1992); no p-value is tabulated.",
             "The lag order is the same at every candidate break date."]
    return table(
        [[statistic, None, *critical_columns(values), period, lags]],
        columns=["statistic", "p_value", *CRITICAL_NAMES, "break_period", "lags"],
        index=["min t"], title=f"Zivot-Andrews unit-root test for {y}", test="zandrews",
        series=y, statistic=statistic, p_value=None, distribution="Zivot-Andrews",
        critical_values=values, break_period=period, break_index=break_index, lags=lags,
        lag_method=method, trim=float(trim), nobs=nobs, df_resid=df,
        candidates=last - first + 1, trend="trend", **{"break": break_},
        label="Minimum Dickey-Fuller t over break dates (H0: unit root without break)",
        notes=notes)
