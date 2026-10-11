"""Elliott-Rothenberg-Stock point-optimal unit-root test P_T.

Source of the statistic, the spectral estimator and the critical values:
Elliott, G., Rothenberg, T. J. and Stock, J. H. (1992), "Efficient tests for an
autoregressive unit root", NBER Technical Working Paper 130
(https://www.nber.org/system/files/working_papers/t0130/t0130.pdf), published
as Econometrica 64 (1996), 813-836: equations (13)-(15) for P_T, (20)-(21) for
the autoregressive spectral estimator and Table 1, panels A (P_T^mu, cbar = -7)
and B (P_T^tau, cbar = -13.5), for T = 50, 100, 200, 500 and infinity at 1%,
2.5%, 5% and 10%. The T = 50, 100, 200 and infinity rows at 1%, 5% and 10% are
identical in the R package urca (``ur.ers``, ``cvals.ptest``, version 1.3-4),
the second transcription; the whole table is re-derived by the seeded
simulation ``scripts/generate_unitroot_fourier_oracles.py`` (see
``tests/test_econ_unitroot_ers.py``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.unitroot.augmented import (
    choose_lags, df_block, differences, nested_fits,
)
from openecon.econometrics.unitroot.common import (
    CRITICAL_NAMES, check_choice, check_count, critical_columns,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot.critical import _interpolate_rows
from openecon.econometrics.unitroot.series import from_origin, one_series

INF = float("inf")
SOURCE = ("Elliott, Rothenberg and Stock (1992), NBER Technical Working Paper 130, Table 1 "
          "(published in Econometrica 64, 1996)")
LEVELS = ("1%", "2.5%", "5%", "10%")
ERS_T = (50.0, 100.0, 200.0, 500.0, INF)
# Table 1: (1%, 2.5%, 5%, 10%) per T row; P_T rejects for small values.
PT_CRITICAL = {
    "constant": ((1.87, 2.39, 2.97, 3.91), (1.95, 2.47, 3.11, 4.17), (1.91, 2.47, 3.17, 4.33),
                 (1.95, 2.53, 3.22, 4.38), (1.99, 2.55, 3.26, 4.48)),
    "trend": ((4.22, 4.94, 5.72, 6.77), (4.26, 4.90, 5.64, 6.79), (4.05, 4.83, 5.66, 6.86),
              (4.05, 4.80, 5.62, 6.93), (3.96, 4.78, 5.62, 6.89)),
}
GLS_C = {"constant": -7.0, "trend": -13.5}
_METHODS = ("aic", "bic", "t")


def _gls_ssr(x: Tensor, z: Tensor, alpha: float, what: str) -> float:
    """SSR of the GLS regression of (y_1, y_t - a y_{t-1}) on (z_1, z_t - a z_{t-1})."""
    quasi_y = torch.cat([x[:1], x[1:] - alpha * x[:-1]])
    quasi_z = torch.cat([z[:1], z[1:] - alpha * z[:-1]])
    return ols(quasi_z, quasi_y, what).ssr


def ers_pt(data: Any, y: str, *, time: str | None = None, trend: str = "constant",
           lags: int | str = "bic", maxlag: int | None = None, minlag: int = 0):
    """Elliott-Rothenberg-Stock point-optimal unit-root test P_T (ERS 1996).

    Model and estimator
    -------------------
    With z_t = 1 (``trend="constant"``) or (1, t) (``"trend"``), cbar = -7 or
    -13.5 and abar = 1 + cbar/T (T the number of observations):

        S(a)  = SSR of the GLS regression of (y_1, y_2 - a y_1, ..., y_T - a y_{T-1})
                on (z_1, z_2 - a z_1, ..., z_T - a z_{T-1})        (ERS eqs. 13-14)
        P_T   = [S(abar) - abar S(1)] / omega^2                    (eq. 15)

    omega^2 is the autoregressive spectral density at frequency zero (eqs.
    20-21): OLS of  Delta y_t = d0 + a0 y_{t-1} + sum_{j=1..p} a_j Delta y_{t-j}
    on t = p+2..T, omega^2 = s^2 / (1 - sum a_j)^2 with s^2 = SSR/(N - K). As in
    ERS (and R's ``urca::ur.ers``) this regression has a constant only, also in
    the trend case; ERS note the trend-case statistic is then not strictly
    invariant to the trend slope under the alternative. H0: unit root; P_T
    rejects for SMALL values.

    Parameters
    ----------
    data, y, time : as in ``oe.dfuller``.
    trend : ``"constant"`` (default, P_T^mu) or ``"trend"`` (P_T^tau).
    lags : order p of the spectral regression, or ``"bic"`` (default, ERS's
        AR(BIC)), ``"aic"`` or ``"t"``: orders minlag..maxlag are compared on the
        common sample t = maxlag+2..T with the rules of ``oe.zandrews``
        (BIC = Schwarz criterion as in ``oe.dfgls``'s SC), then the chosen order
        is refitted on t = p+2..T. ``oe.dfgls``'s MAIC is not offered: it is
        defined for the GLS-detrended regression, not for regression (21).
    maxlag : largest order for automatic selection; default Schwert's
        int(12 (T/100)^(1/4)) capped by the sample (``oe.dfgls``'s default).
    minlag : smallest order for automatic selection (default 0). ERS's Monte
        Carlo used BIC with 3 <= p <= 8: ``minlag=3, maxlag=8``.

    Critical values
    ---------------
    ERS Table 1 (NBER TWP 130; Econometrica 1996): P_T^mu and P_T^tau for
    T = 50, 100, 200, 500, infinity, interpolated linearly in T (the T = 50 row
    below 50 and the asymptotic row above 500, with a note). They were
    simulated with omega^2 = sigma^2 for Gaussian random walks; the asymptotic
    row is the limit under the general I(1) null. ``critical_values`` has 1%,
    2.5%, 5% and 10%. ``p_value`` is None: ERS publish quantiles only.

    Returns
    -------
    A one-row table (index ``P_T``: statistic, p_value, critical_1pct,
    critical_5pct, critical_10pct). ``attrs``: ``statistic``, ``p_value``
    (None), ``p_value_reason``, ``critical_values``, ``reject``, ``trend``,
    ``gls_c``, ``gls_alpha``, ``ssr_alpha`` (S(abar)), ``ssr_one`` (S(1)),
    ``spectral_variance`` (omega^2), ``ar_lag_sum``, ``lags``, ``lag_method``,
    ``minlag``, ``maxlag``, ``lag_selection_results``, ``nobs`` (spectral
    regression), ``nobs_series``, ``critical_value_source``, ``label``,
    ``notes``.

    R: ``urca::ur.ers(y, type = "P-test", model = "constant", lag.max = 4)``
    (urca chooses 1..lag.max lags by BIC on unequal samples and, in the trend
    case, computes S(1) with the T-observation mean, which differs from the GLS
    residual sum by (y_T - y_1)^2 / (T (T - 1)); see the documentation page).

    Example
    -------
    >>> result = oe.ers_pt(df, "gdp", trend="trend", minlag=3, maxlag=8)
    >>> result.attrs["statistic"], result.attrs["critical_values"]["5%"]
    """
    check_choice(trend, "trend", tuple(PT_CRITICAL))
    minlag = check_count(minlag, "minlag")
    x, _ = one_series(data, y, time, "The ERS point-optimal test", minimum=10)
    x, _ = from_origin(x)
    total = x.shape[0]
    if isinstance(lags, str):
        check_choice(lags, "lags", _METHODS)
        method = lags
        if maxlag is None:
            maxlag = max(minlag, min(int(12 * (total / 100) ** 0.25), (total - 5) // 2))
        maxlag = check_count(maxlag, "maxlag")
        if maxlag < minlag:
            raise AnalysisError("invalid_lags", "maxlag must be at least minlag.")
        if total - 1 - maxlag <= maxlag + 2:
            raise AnalysisError("invalid_lags", f"maxlag={maxlag} is too large for {total} "
                                "observations.")
    else:
        method = "fixed"
        lags = check_count(lags, "lags")
        if maxlag is not None or minlag:
            raise AnalysisError("invalid_option", "maxlag and minlag apply only with lags='aic', "
                                "'bic' or 't'.")
    change = differences(x)
    constant = torch.ones((total, 1), dtype=torch.float64)
    selection = []
    if method != "fixed":
        rows = total - 1 - maxlag
        fits = nested_fits(rows, 2, maxlag, df_block(x, change, constant, maxlag, maxlag),
                           "The ERS spectral regression")
        lags = choose_lags(fits, method, minlag)
        selection = [{"lags": fit.lags, "ssr": fit.ssr, "nobs": fit.nobs} for fit in fits
                     if fit.lags >= minlag]
    rows = total - 1 - lags
    if rows <= lags + 2:
        raise AnalysisError("insufficient_observations", f"The ERS spectral regression has "
                            f"{max(rows, 0)} observation(s) for {lags + 2} parameter(s); use "
                            "fewer lags or a longer series.")
    design = df_block(x, change, constant, lags, start=lags)(0, rows)
    fit = ols(design[:, :-1].contiguous(), design[:, -1].contiguous(),
              "The ERS spectral regression")
    if fit.exact or not fit.s2 > 0.0:
        raise AnalysisError("perfect_fit", "The ERS spectral regression fits the series exactly; "
                            "the series is deterministic.")
    lag_sum = float(fit.beta[2:].sum()) if lags else 0.0
    if abs(1.0 - lag_sum) <= 1e-10 * max(1.0, float(fit.beta[2:].abs().sum()) if lags else 1.0):
        raise AnalysisError("invalid_covariance", "The autoregressive spectral denominator "
                            "1 - sum(a_j) is zero; reduce lags or inspect the series.")
    spectral = fit.s2 / (1.0 - lag_sum) ** 2
    c = GLS_C[trend]
    alpha = 1.0 + c / total
    columns = [torch.ones(total, dtype=torch.float64)]
    if trend == "trend":
        columns.append(torch.arange(1, total + 1, dtype=torch.float64) / total)
    z = torch.stack(columns, dim=1)
    ssr_alpha = _gls_ssr(x, z, alpha, "The GLS regression under the local alternative")
    ssr_one = _gls_ssr(x, z, 1.0, "The GLS regression under the null")
    statistic = (ssr_alpha - alpha * ssr_one) / spectral
    if not math.isfinite(statistic) or not spectral > 0.0:
        raise AnalysisError("numerical_failure", "The P_T statistic is numerically unresolved; "
                            "rescale the series.")
    row = _interpolate_rows(ERS_T, PT_CRITICAL[trend], float(total), reciprocal_tail=False)
    values = dict(zip(LEVELS, row, strict=True))
    notes = [f"Critical values: {SOURCE}, panel {'A' if trend == 'constant' else 'B'}, "
             "interpolated linearly in T; quantiles only, no p-value.",
             "P_T rejects the unit root for small values."]
    if total < ERS_T[0]:
        notes.append(f"Critical values are tabulated from T = 50; the T = 50 row is used for "
                     f"T = {total}.")
    elif total > ERS_T[-2]:
        notes.append(f"T = {total} exceeds the largest finite row (500); the asymptotic row is "
                     "used.")
    if method != "fixed":
        notes.append(f"Spectral lag order chosen by {method.upper()} over {minlag}..{maxlag} on "
                     "the common sample, then refitted on t = p+2..T.")
    attrs = {
        "title": f"Elliott-Rothenberg-Stock point-optimal test for {y}", "test": "ers_pt",
        "series": y, "statistic": statistic, "p_value": None,
        "p_value_reason": "ERS (1996) publish quantiles only; no p-value is reported.",
        "inference_status": "critical_values_only", "distribution": "ERS P_T",
        "critical_values": values,
        "reject": {level: statistic < value for level, value in values.items()},
        "trend": trend, "gls_c": c, "gls_alpha": alpha, "ssr_alpha": ssr_alpha,
        "ssr_one": ssr_one, "spectral_variance": spectral, "ar_lag_sum": lag_sum,
        "lags": lags, "lag_method": method, "minlag": minlag if method != "fixed" else None,
        "maxlag": maxlag if method != "fixed" else None, "lag_selection_results": selection,
        "nobs": fit.nobs, "nobs_series": total, "df_resid": fit.df,
        "critical_value_source": SOURCE, "null": "Unit root",
        "alternative": "Level stationarity" if trend == "constant" else "Trend stationarity",
        "label": "ERS point-optimal P_T (H0: unit root; small values reject)", "notes": notes,
    }
    return table([[statistic, None, *critical_columns(values)]],
                 columns=["statistic", "p_value", *CRITICAL_NAMES], index=["P_T"], **attrs)
