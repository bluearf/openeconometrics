"""Nested augmented regressions with arbitrary deterministic columns.

The Fourier unit-root tests, the ERS point-optimal test and the HEGY seasonal
test all compare lag orders of one auxiliary regression on a common sample,
exactly as ``series.nested_adf`` does for the ordinary Dickey-Fuller
regression, but with deterministic columns that are not a constant/trend
(Fourier terms, differenced Fourier terms, seasonal dummies) or with several
level regressors (the HEGY filters). This module provides the shared pieces:

``nested_fits``   one triangular factor of
                  ``[fixed columns, lag_1, ..., lag_maxlag, target]`` gives the
                  regression with 0, 1, ..., maxlag trailing lag columns
                  (``R_m``, ``Q_m' y`` and ``SSR = ||R[m:, -1]||^2``); rows are
                  accumulated in blocks, so the family costs O(T K^2) time and
                  one block of memory.
``choose_lags``   AIC / BIC / sequential-t choice over those fits, with the
                  rule and tie-breaking of ``series.adf_lag_choice``.
``df_columns``    the Dickey-Fuller style columns of a level series.

Everything is float64 Torch; no observation-level Python loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot.common import check_choice, check_count
from openecon.engines.distributions import normal_isf

# Rows of the accumulated factorization are processed in blocks of about this many elements.
_BLOCK_ELEMENTS = 1 << 22
LAG_METHODS = ("aic", "bic", "t")


@dataclass
class NestedFit:
    """One member of a nested family (``lags`` trailing lag columns)."""

    lags: int
    nobs: int
    parameters: int
    ssr: float
    beta: Tensor            # [parameters] in column order
    variance_factor: Tensor  # diagonal of (X'X)^{-1}
    exact: bool             # residual at rounding level relative to the target

    @property
    def s2(self) -> float:
        return self.ssr / (self.nobs - self.parameters)

    def t_ratio(self, column: int) -> float | None:
        """Classical OLS t ratio of one column (None when the fit is exact)."""
        if self.exact or not self.ssr > 0.0:
            return None
        return float(self.beta[column]) / math.sqrt(self.s2 * float(self.variance_factor[column]))


@torch.no_grad()
def nested_fits(rows: int, fixed: int, maxlag: int,
                block: Callable[[int, int], Tensor], what: str) -> list[NestedFit]:
    """Fit ``target`` on the leading ``fixed + k`` columns for k = 0..maxlag.

    ``block(low, high)`` returns the design rows ``low..high-1`` of the common
    sample as ``[high - low, fixed + maxlag + 1]``: the fixed columns, the
    ``maxlag`` lag columns in order, and the target last. A rank-deficient
    family (a column that is a combination of the columns to its left) raises
    ``collinear_regressors``; Stata's screen on the unpivoted factor is used.
    """
    width = fixed + maxlag + 1
    if rows < width:
        raise AnalysisError("insufficient_observations", f"{what} has {max(rows, 0)} "
                            f"observation(s) for {width - 1} parameter(s); use fewer lags or a "
                            "longer series.")
    factor = torch.empty((0, width), dtype=torch.float64)
    step = max(width, _BLOCK_ELEMENTS // width)
    for low in range(0, rows, step):
        high = min(low + step, rows)
        factor = torch.linalg.qr(torch.cat([factor, block(low, high)]), mode="r").R
    pivots = factor.diagonal()[:width - 1].square()
    norms = factor[:, :width - 1].square().sum(dim=0)
    if bool((pivots <= 1e-13 * norms).any()):
        raise AnalysisError("collinear_regressors", f"{what} is rank deficient: the series does "
                            "not vary enough to identify every term (for example it is constant, "
                            "an exact deterministic path, or has too many lags for the sample).")
    target = factor[:, -1]
    scale = float(target.square().sum())
    fits = []
    for order in range(maxlag + 1):
        m = fixed + order
        triangle = factor[:m, :m]
        beta = torch.linalg.solve_triangular(triangle, target[:m, None], upper=True)[:, 0]
        inverse = torch.linalg.solve_triangular(
            triangle, torch.eye(m, dtype=torch.float64), upper=True)
        ssr = float(target[m:].square().sum())
        fits.append(NestedFit(lags=order, nobs=rows, parameters=m, ssr=ssr, beta=beta,
                              variance_factor=inverse.square().sum(dim=1),
                              exact=not ssr > 1e-22 * scale))
    return fits


def choose_lags(fits: list[NestedFit], method: str, minlag: int = 0) -> int:
    """Lag order by AIC, BIC or the sequential t rule over ``minlag..maxlag``.

    ``aic`` / ``bic`` minimize ``N ln(SSR/N) + K c`` with ``c = 2`` or ``ln N`` on
    the common sample (ties go to the smaller order); ``t`` starts at maxlag and
    stops at the first order whose last lag column has |t| >= 1.645 (the 10%
    two-sided normal point), else ``minlag``. These are the rules of
    ``series.adf_lag_choice``. Exact candidate fits are an error: the series is
    deterministic and no lag order is identified.
    """
    candidates = [fit for fit in fits if fit.lags >= minlag]
    if any(fit.exact or not fit.ssr > 0.0 for fit in candidates):
        raise AnalysisError("perfect_fit", "A lag-selection regression fits the series exactly; "
                            "the series is deterministic and the test is undefined.")
    if method == "t":
        threshold = normal_isf(0.05)
        for fit in reversed(candidates):
            if fit.lags == minlag:
                return minlag
            statistic = fit.t_ratio(fit.parameters - 1)
            if statistic is not None and abs(statistic) >= threshold:
                return fit.lags
        return minlag
    best, best_value = minlag, math.inf
    for fit in candidates:
        penalty = 2.0 if method == "aic" else math.log(fit.nobs)
        value = fit.nobs * math.log(fit.ssr / fit.nobs) + fit.parameters * penalty
        if value < best_value:
            best, best_value = fit.lags, value
    return best


def df_block(level: Tensor, change: Tensor, deterministic: Tensor | None, maxlag: int,
             start: int) -> Callable[[int, int], Tensor]:
    """Block builder of a Dickey-Fuller style design on rows t = start+2..T (1-based).

    Columns: ``deterministic[t]`` (rows of a [T, d] tensor indexed by
    observation), ``level[t-1]``, ``change[t-1], ..., change[t-maxlag]`` and the
    target ``change[t]``. ``change`` is indexed by observation as well
    (``change[0]`` is unused); a common sample uses ``start = maxlag``.
    """

    def block(low: int, high: int) -> Tensor:
        first, last = start + 1 + low, start + 1 + high          # 0-based observation t
        columns = []
        if deterministic is not None and deterministic.shape[1]:
            columns.append(deterministic[first:last])
        columns.append(level[first - 1:last - 1, None])
        if maxlag:
            columns.append(torch.stack([change[first - j:last - j] for j in range(1, maxlag + 1)],
                                       dim=1))
        columns.append(change[first:last, None])
        return torch.cat(columns, dim=1)

    return block


def differences(x: Tensor) -> Tensor:
    """First differences indexed by observation (entry 0 is 0 and never used)."""
    return torch.cat([torch.zeros(1, dtype=torch.float64), x[1:] - x[:-1]])


def check_lag_option(lags, maxlag, total: int, parameters: int, what: str) -> tuple[str, int]:
    """Validate ``lags`` (an integer or 'aic'/'bic'/'t') and ``maxlag``.

    Returns (method, order) where method is 'fixed' or the selection rule and
    order is the fixed order or the largest candidate. The default maxlag is
    Schwert's int(12 (T/100)^(1/4)), capped so that the common-sample regression
    keeps at least as many observations as parameters plus lags.
    """
    if isinstance(lags, str):
        check_choice(lags, "lags", LAG_METHODS)
        if maxlag is None:
            limit = max(0, min(int(12 * (total / 100) ** 0.25),
                               (total - parameters - 2) // 2))
        else:
            limit = check_count(maxlag, "maxlag")
        if total - 1 - limit <= parameters + limit:
            raise AnalysisError("invalid_lags", f"maxlag={limit} is too large for {total} "
                                f"observations in {what}.")
        return lags, limit
    order = check_count(lags, "lags")
    if maxlag is not None:
        raise AnalysisError("invalid_option", "maxlag applies only with lags='aic', 'bic' or 't'.")
    return "fixed", order
