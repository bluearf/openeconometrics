"""Fourier unit-root and stationarity tests with one smooth-break frequency.

``fourier_adf``   Enders and Lee (2012, Economics Letters 117): Dickey-Fuller
                  regression with sin/cos(2 pi k t / T) deterministic terms.
``fourier_kpss``  Becker, Enders and Lee (2006, JTSA 27): KPSS statistic of the
                  residuals of a level (or trend) plus Fourier regression.
``fourier_lm``    Enders and Lee (2012, Oxford Bulletin 74): Schmidt-Phillips
                  LM test with Fourier terms estimated in first differences.

The frequency k is fixed (1..5) or chosen by minimum sum of squared residuals
over 1..max_frequency (Davies 1987, as in all three papers); each procedure
also reports the F statistic for the null that the Fourier coefficients are
zero. Critical values: ``fourier_tables.py`` (sources there and in
``docs/econometrics/advanced-unitroot-fourier.md``). No p-values: the papers
publish quantiles only, and interpolating them would invent a distribution.

Series are taken in row order or sorted by ``time`` (consecutive periods);
missing values, gaps and constant series are errors (``common.load_series``).
Everything is float64 Torch; the regressions are QR least squares.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot import fourier_tables as tables
from openecon.econometrics.unitroot.augmented import (
    check_lag_option, choose_lags, df_block, differences, nested_fits,
)
from openecon.econometrics.unitroot.common import (
    CRITICAL_NAMES, autocovariances, check_choice, check_count, check_flag, critical_columns,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot.critical import _interpolate_rows
from openecon.econometrics.unitroot.series import from_origin, kpss_bandwidth, one_series

_TRENDS = ("constant", "trend")
_LEVELS = ("1%", "5%", "10%")
MAX_FREQUENCY = 5
_P_VALUE_REASON = ("The source papers publish 1%, 5% and 10% quantiles only; no response "
                   "surface or distribution function is available, so no p-value is reported.")


# ---- shared pieces ---------------------------------------------------------------------


def _frequencies(frequency: Any, max_frequency: Any) -> tuple[list[int], str]:
    """Candidate frequencies and the selection mode ('fixed' or 'min_ssr')."""
    if frequency == "auto":
        top = MAX_FREQUENCY if max_frequency is None else max_frequency
        if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= MAX_FREQUENCY:
            raise AnalysisError("invalid_option", "max_frequency must be an integer from 1 to "
                                f"{MAX_FREQUENCY}.")
        return list(range(1, top + 1)), "min_ssr"
    if isinstance(frequency, bool) or not isinstance(frequency, int) \
            or not 1 <= frequency <= MAX_FREQUENCY:
        raise AnalysisError("invalid_option", "frequency must be 'auto' or an integer from 1 to "
                            f"{MAX_FREQUENCY} (critical values are tabulated for k = 1..5).")
    if max_frequency is not None:
        raise AnalysisError("invalid_option", "max_frequency applies only with frequency='auto'.")
    return [frequency], "fixed"


def _fourier(total: int, k: int) -> Tensor:
    """[T, 2] columns sin(2 pi k t / T), cos(2 pi k t / T) for t = 1..T."""
    angle = 2.0 * math.pi * k * torch.arange(1, total + 1, dtype=torch.float64) / total
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def _base(total: int, trend: bool) -> Tensor:
    """[T, d] constant (and trend t/T) columns indexed by observation."""
    columns = [torch.ones(total, dtype=torch.float64)]
    if trend:
        columns.append(torch.arange(1, total + 1, dtype=torch.float64) / total)
    return torch.stack(columns, dim=1)


def _critical(nodes: tuple[float, ...], rows, total: int) -> tuple[dict[str, float], str | None]:
    """Row interpolated linearly in T; the first/last tabulated row outside the table."""
    values = dict(zip(_LEVELS, _interpolate_rows(nodes, rows, float(total),
                                                 reciprocal_tail=False), strict=True))
    note = None
    if total < nodes[0]:
        note = (f"Critical values are tabulated from T = {nodes[0]:g}; the T = {nodes[0]:g} row "
                f"is used for T = {total}.")
    elif total > nodes[-1]:
        note = (f"Critical values are tabulated up to T = {nodes[-1]:g}; that row is used for "
                f"T = {total}.")
    return values, note


def _reject(statistic: float, values: dict[str, float] | None, lower: bool) -> dict | None:
    if values is None:
        return None
    return {level: (statistic < value if lower else statistic > value)
            for level, value in values.items()}


def _df_fit(level: Tensor, change: Tensor, deterministic: Tensor, lags: int, what: str):
    """Augmented regression on t = lags+2..T: [deterministic, level_{t-1}, lags] (QR)."""
    rows = level.shape[0] - 1 - lags
    block = df_block(level, change, deterministic, lags, start=lags)(0, rows)
    fit = ols(block[:, :-1].contiguous(), block[:, -1].contiguous(), what)
    if fit.exact or not fit.s2 > 0.0:
        raise AnalysisError("perfect_fit", f"{what} fits the series exactly, so its statistic "
                            "is undefined. The series is deterministic.")
    return fit


def _select_lags(level: Tensor, change: Tensor, deterministic: Tensor, method: str,
                 order: int, what: str) -> tuple[int, float | None]:
    """(lags, SSR of the chosen order on the common sample) for one frequency."""
    if method == "fixed":
        return order, None
    rows = level.shape[0] - 1 - order
    fixed = deterministic.shape[1] + 1
    fits = nested_fits(rows, fixed, order, df_block(level, change, deterministic, order, order),
                       what)
    chosen = choose_lags(fits, method)
    return chosen, fits[chosen].ssr


def _f_statistic(restricted_ssr: float, unrestricted, restrictions: int = 2) -> float:
    return ((restricted_ssr - unrestricted.ssr) / restrictions) / unrestricted.s2


# ---- Fourier ADF -----------------------------------------------------------------------


def fourier_adf(data: Any, y: str, *, time: str | None = None, trend: str = "constant",
                frequency: int | str = "auto", max_frequency: int | None = None,
                lags: int | str = 0, maxlag: int | None = None):
    """Fourier Dickey-Fuller unit-root test (Enders and Lee 2012, Economics Letters).

    Model and estimator
    -------------------
    OLS of the Dickey-Fuller regression with one Fourier frequency k on
    t = p+2..T (T the number of observations, t = 1..T):

        Delta y_t = rho y_{t-1} + c1 [+ c2 t] + c3 sin(2 pi k t/T) + c4 cos(2 pi k t/T)
                    + sum_{j=1..p} b_j Delta y_{t-j} + e_t

    The statistic tau is the classical OLS t ratio of rho. H0: unit root
    (rho = 0); small values reject. ``trend="constant"`` is the tau_DF_C test
    (Table 1b), ``"trend"`` the tau_DF_t test (Table 1a).

    The F statistic tests c3 = c4 = 0 in the same regression (same sample and
    lags), F = ((SSR_r - SSR_u)/2) / (SSR_u/(N - K)). Its distribution under the
    unit-root null is nonstandard; Enders and Lee tabulate only the supremum
    F(k^) = max_k F(k) over k = 1..5, so F critical values are reported only
    when the frequency is chosen over 1..5.

    Parameters
    ----------
    data, y, time : as in ``oe.dfuller``.
    trend : ``"constant"`` (default) or ``"trend"``.
    frequency : integer 1..5, or ``"auto"`` (default): the frequency with the
        smallest SSR among 1..max_frequency (Enders and Lee, step 1). Ties go to
        the lower frequency.
    max_frequency : largest candidate with ``frequency="auto"`` (default 5).
    lags : number of lagged differences p (default 0), or ``"aic"``, ``"bic"`` or
        ``"t"`` to choose p for each candidate frequency on the common sample
        t = maxlag+2..T with the rules of ``oe.zandrews`` (``"t"``: largest p
        whose last lag has |t| >= 1.645). The frequency is then chosen by the
        SSR of those common-sample regressions, and the chosen (k, p) is refitted
        on t = p+2..T.
    maxlag : largest candidate p for automatic lags; default Schwert's
        int(12 (T/100)^(1/4)) capped by the sample.

    Critical values
    ---------------
    Enders and Lee (2012), Tables 1a/1b for T = 100, 200, 500, 2500,
    interpolated linearly in T (the T = 100 row below 100, the T = 2500 row
    above 2500, with a note). Simulated under iid errors without augmentation;
    with lags they are asymptotic. ``p_value`` is None (quantiles only).

    Returns
    -------
    ``TableSet`` with ``test`` (rows ``tau`` and ``F``: statistic, p_value and
    the three critical values) and ``frequencies`` (one row per candidate:
    frequency, lags, selection_ssr, statistic, f_statistic). ``attrs``:
    ``statistic``, ``p_value`` (None), ``p_value_reason``, ``critical_values``,
    ``reject``, ``f_statistic``, ``f_critical_values``, ``f_reject``,
    ``frequency``, ``frequency_selection``, ``max_frequency``, ``lags``,
    ``lag_method``, ``maxlag``, ``trend``, ``nobs``, ``nobs_series``,
    ``df_resid``, ``critical_value_source``, ``label``, ``notes``.

    Example
    -------
    >>> result = oe.fourier_adf(df, "spread", trend="constant", lags="t")
    >>> result.attrs["statistic"], result.attrs["frequency"]
    """
    check_choice(trend, "trend", _TRENDS)
    candidates, mode = _frequencies(frequency, max_frequency)
    x, _ = one_series(data, y, time, "The Fourier ADF test", minimum=12)
    x, _ = from_origin(x)
    total = x.shape[0]
    base = _base(total, trend == "trend")
    method, order = check_lag_option(lags, maxlag, total, base.shape[1] + 3,
                                     "the Fourier ADF regression")
    change = differences(x)
    rows, best = [], None
    for k in candidates:
        deterministic = torch.cat([base, _fourier(total, k)], dim=1)
        what = f"The Fourier ADF regression (k = {k})"
        chosen, common_ssr = _select_lags(x, change, deterministic, method, order, what)
        unrestricted = _df_fit(x, change, deterministic, chosen, what)
        restricted = _df_fit(x, change, base, chosen, "The Fourier ADF regression without "
                             "Fourier terms")
        level = deterministic.shape[1]
        statistic = float(unrestricted.beta[level] / unrestricted.se[level])
        f_stat = _f_statistic(restricted.ssr, unrestricted)
        selection_ssr = unrestricted.ssr if common_ssr is None else common_ssr
        row = {"frequency": k, "lags": chosen, "selection_ssr": selection_ssr,
               "statistic": statistic, "f_statistic": f_stat, "nobs": unrestricted.nobs,
               "df_resid": unrestricted.df}
        rows.append(row)
        if best is None or selection_ssr < best["selection_ssr"]:
            best = row
    case = trend
    k = best["frequency"]
    values, note = _critical(tables.ADF_T, tables.ADF_TAU[case][k - 1], total)
    f_values, f_reason = None, None
    if mode == "min_ssr" and len(candidates) == MAX_FREQUENCY:
        f_values, _ = _critical(tables.ADF_T, tables.ADF_SUPF[case], total)
    elif mode == "min_ssr":
        f_reason = ("Enders and Lee tabulate the supremum F over k = 1..5 only; with "
                    f"max_frequency={len(candidates)} no critical value is available.")
    else:
        f_reason = ("Enders and Lee tabulate only the supremum F(k^) = max_k F(k) over k = "
                    "1..5; F for a fixed frequency has no tabulated distribution under the "
                    "unit-root null.")
    statistic, f_stat = best["statistic"], best["f_statistic"]
    notes = ["Critical values: Enders and Lee (2012, Economics Letters), Table "
             f"{'1a' if trend == 'trend' else '1b'}, interpolated linearly in T; quantiles "
             "only, no p-value.",
             "The critical values for an estimated frequency are those of the chosen k, as "
             "Enders and Lee recommend; they ignore the selection step."]
    if note:
        notes.append(note)
    if f_reason:
        notes.append(f_reason)
    if method != "fixed":
        notes.append("Lag orders were compared on the common sample t = maxlag+2..T; the "
                     "chosen order was refitted on t = p+2..T.")
    attrs = {
        "title": f"Fourier ADF unit-root test for {y}", "test": "fourier_adf", "series": y,
        "statistic": statistic, "p_value": None, "p_value_reason": _P_VALUE_REASON,
        "inference_status": "critical_values_only",
        "distribution": "Fourier Dickey-Fuller (Enders-Lee 2012)",
        "critical_values": values, "reject": _reject(statistic, values, lower=True),
        "f_statistic": f_stat, "f_critical_values": f_values,
        "f_reject": _reject(f_stat, f_values, lower=False), "f_inference_reason": f_reason,
        "frequency": k, "frequency_selection": mode, "max_frequency": max(candidates),
        "lags": best["lags"], "lag_method": method,
        "maxlag": order if method != "fixed" else None, "trend": trend,
        "nobs": best["nobs"], "nobs_series": total, "df_resid": best["df_resid"],
        "critical_value_source": tables.SOURCES["fourier_adf"],
        "null": "Unit root", "alternative": "Stationarity around a smooth Fourier deterministic "
        "component" + (" and linear trend" if trend == "trend" else ""),
        "label": "Fourier ADF tau (H0: unit root); F tests the Fourier terms", "notes": notes,
    }
    test = table([[statistic, None, *critical_columns(values)],
                  [f_stat, None, *critical_columns(f_values)]],
                 columns=["statistic", "p_value", *CRITICAL_NAMES], index=["tau", "F"], **attrs)
    frequencies = table([[row["frequency"], row["lags"], row["selection_ssr"], row["statistic"],
                          row["f_statistic"]] for row in rows],
                        columns=["frequency", "lags", "selection_ssr", "statistic", "f_statistic"])
    return TableSet({"test": test, "frequencies": frequencies}, **attrs)


# ---- Fourier KPSS ----------------------------------------------------------------------


def _kpss_statistic(resid: Tensor, lags: int) -> tuple[float, float]:
    total = resid.shape[0]
    eta = float(resid.cumsum(dim=0).square().sum()) / total ** 2
    gamma = autocovariances(resid, lags)
    weights = 1.0 - torch.arange(1, lags + 1, dtype=torch.float64) / (lags + 1)
    variance = float(gamma[0] + 2.0 * (weights * gamma[1:]).sum())
    if not variance > 0.0:
        raise AnalysisError("numerical_failure", "The long-run variance estimate is not "
                            "positive; use fewer lags.")
    return eta / variance, variance


def fourier_kpss(data: Any, y: str, *, time: str | None = None, trend: str = "constant",
                 frequency: int | str = "auto", max_frequency: int | None = None,
                 lags: int | None = None, auto: bool = False):
    """Fourier KPSS stationarity test (Becker, Enders and Lee 2006).

    Model and estimator
    -------------------
    OLS of y on a constant [and trend t] and sin/cos(2 pi k t/T), t = 1..T
    (regression 3a / 3b). With e~_t the residuals and S_t their partial sums,

        tau(k) = sum_t S_t^2 / (T^2 s^2(l)),
        s^2(l) = g_0 + 2 sum_{j<=l} (1 - j/(l+1)) g_j,   g_j = (1/T) sum e~_t e~_{t-j}

    (Bartlett weights, as ``oe.kpss``). H0: stationarity around the level
    (``trend="constant"``, tau_mu) or trend (``"trend"``, tau_tau) plus the
    Fourier component; large values reject.

    The F statistic of eq. (7) tests the Fourier coefficients:
    F = ((SSR_0 - SSR_1)/2) / (SSR_1/(T - q)), q the number of regressors.

    Parameters
    ----------
    data, y, time : as in ``oe.kpss``.
    trend : ``"constant"`` (default) or ``"trend"``.
    frequency : 1..5 or ``"auto"`` (default; smallest SSR over 1..max_frequency,
        ties to the lower frequency).
    max_frequency : largest candidate with ``"auto"`` (default 5).
    lags : Bartlett truncation l; default Schwert's int(12 (T/100)^(1/4)) (the
        default of ``oe.kpss``).
    auto : choose l by Newey and West (1994) as ``oe.kpss(auto=True)`` does.

    Critical values
    ---------------
    tau: Becker, Enders and Lee (2006, Table 1) for k = 1, 2, 3 (T = 100, 500),
    OpenEconometrics seeded simulation for k = 4, 5; linear interpolation in T,
    the T = 100 row below 100 and the T = 500 row above 500. Supremum F (only
    with ``frequency="auto"`` over 1..5): seeded simulation of eq. (7) under iid
    errors; the published supremum-F values did not replicate (see the module
    ``fourier_tables``). BEL note that F over-rejects with serially correlated
    errors; F for a fixed frequency has no tabulated distribution here.
    ``p_value`` is None (quantiles only).

    Returns
    -------
    ``TableSet`` with ``test`` (rows ``tau`` and ``F``) and ``frequencies``
    (frequency, ssr, statistic, f_statistic, lags). ``attrs`` as in
    ``oe.fourier_adf`` plus ``bandwidth`` (``"schwert"``, ``"user"`` or
    ``"auto"``), ``kernel`` and ``long_run_variance``.

    Example
    -------
    >>> oe.fourier_kpss(df, "real_rate", lags=8).attrs["statistic"]
    """
    check_choice(trend, "trend", _TRENDS)
    check_flag(auto, "auto")
    if auto and lags is not None:
        raise AnalysisError("invalid_option", "Give either lags or auto=True, not both.")
    candidates, mode = _frequencies(frequency, max_frequency)
    x, _ = one_series(data, y, time, "The Fourier KPSS test", minimum=12)
    x, _ = from_origin(x)
    total = x.shape[0]
    if lags is not None:
        lags = check_count(lags, "lags")
        if lags >= total:
            raise AnalysisError("invalid_lags", f"lags must be smaller than the number of "
                                f"observations ({total}).")
    base = _base(total, trend == "trend")
    restricted = ols(base, x, "The KPSS regression without Fourier terms")
    rows, best = [], None
    for k in candidates:
        design = torch.cat([base, _fourier(total, k)], dim=1)
        fit = ols(design, x, f"The Fourier KPSS regression (k = {k})")
        if fit.exact:
            raise AnalysisError("constant_series", "The series has no variation around the "
                                "Fourier deterministic component; the test is undefined.")
        if auto:
            order, bandwidth = kpss_bandwidth(fit.resid), "auto"
        elif lags is None:
            order, bandwidth = min(total - 1, int(12 * (total / 100) ** 0.25)), "schwert"
        else:
            order, bandwidth = lags, "user"
        statistic, variance = _kpss_statistic(fit.resid, order)
        f_stat = ((restricted.ssr - fit.ssr) / 2.0) / (fit.ssr / (total - design.shape[1]))
        row = {"frequency": k, "ssr": fit.ssr, "statistic": statistic, "f_statistic": f_stat,
               "lags": order, "bandwidth": bandwidth, "long_run_variance": variance}
        rows.append(row)
        if best is None or fit.ssr < best["ssr"]:
            best = row
    k = best["frequency"]
    values, note = _critical(tables.KPSS_T, tables.KPSS_TAU[trend][k - 1], total)
    simulated = k in tables.KPSS_SIMULATED_FREQUENCIES
    f_values, f_reason = None, None
    if mode == "min_ssr" and len(candidates) == MAX_FREQUENCY:
        f_values, _ = _critical(tables.KPSS_T, tables.KPSS_SUPF[trend], total)
    elif mode == "min_ssr":
        f_reason = ("The supremum F is calibrated over k = 1..5 only; with "
                    f"max_frequency={len(candidates)} no critical value is available.")
    else:
        f_reason = ("Becker, Enders and Lee calibrate only the supremum F(k^) over k = 1..5; "
                    "a fixed-frequency F is reported without critical values.")
    statistic, f_stat = best["statistic"], best["f_statistic"]
    source = tables.SOURCES["fourier_kpss_simulated" if simulated else "fourier_kpss"]
    notes = [f"Critical values for k = {k}: {source}; linear interpolation in T.",
             "Large values reject stationarity. The critical values of an estimated frequency "
             "are those of the chosen k, as Becker, Enders and Lee recommend."]
    if note:
        notes.append(note)
    if f_values is not None:
        notes.append("Supremum-F critical values: OpenEconometrics seeded simulation under iid "
                     "errors (the published values did not replicate); F over-rejects with "
                     "serially correlated errors.")
    if f_reason:
        notes.append(f_reason)
    attrs = {
        "title": f"Fourier KPSS stationarity test for {y}", "test": "fourier_kpss", "series": y,
        "statistic": statistic, "p_value": None, "p_value_reason": _P_VALUE_REASON,
        "inference_status": "critical_values_only",
        "distribution": "Fourier KPSS (Becker-Enders-Lee 2006)",
        "critical_values": values, "reject": _reject(statistic, values, lower=False),
        "critical_value_source": source,
        "f_statistic": f_stat, "f_critical_values": f_values,
        "f_reject": _reject(f_stat, f_values, lower=False), "f_inference_reason": f_reason,
        "f_critical_value_source": tables.SOURCES["fourier_kpss_simulated"]
        if f_values is not None else None,
        "frequency": k, "frequency_selection": mode, "max_frequency": max(candidates),
        "lags": best["lags"], "kernel": "bartlett", "bandwidth": best["bandwidth"],
        "long_run_variance": best["long_run_variance"], "trend": trend, "nobs": total,
        "null": "Stationarity around the " + ("trend" if trend == "trend" else "level")
        + " plus a Fourier component", "alternative": "Unit root",
        "label": "Fourier KPSS statistic (H0: stationarity); F tests the Fourier terms",
        "notes": notes,
    }
    test = table([[statistic, None, *critical_columns(values)],
                  [f_stat, None, *critical_columns(f_values)]],
                 columns=["statistic", "p_value", *CRITICAL_NAMES], index=["tau", "F"], **attrs)
    frequencies = table([[row["frequency"], row["ssr"], row["statistic"], row["f_statistic"],
                          row["lags"]] for row in rows],
                        columns=["frequency", "ssr", "statistic", "f_statistic", "lags"])
    return TableSet({"test": test, "frequencies": frequencies}, **attrs)


# ---- Fourier LM ------------------------------------------------------------------------


def _lm_detrend(x: Tensor, change: Tensor, columns: Tensor, levels: Tensor, what: str) -> Tensor:
    """S~_t = y_t - y_1 - (levels_t - levels_1) delta with delta from Delta y on columns.

    ``columns`` are the differenced deterministic terms (rows t = 2..T) and
    ``levels`` the same terms in levels (rows t = 1..T); S~_1 = 0, which is the
    paper's psi~ normalization.
    """
    fit = ols(columns[1:].contiguous(), change[1:].contiguous(), what)
    path = levels @ fit.beta
    return (x - x[0]) - (path - path[0])


def fourier_lm(data: Any, y: str, *, time: str | None = None, frequency: int | str = "auto",
               max_frequency: int | None = None, lags: int | str = 0, maxlag: int | None = None):
    """Fourier LM unit-root test (Enders and Lee 2012, Oxford Bulletin).

    Model and estimator
    -------------------
    1. First differences (t = 2..T) are regressed on the differenced
       deterministic terms:  Delta y_t = d0 + d1 Delta sin(2 pi k t/T)
       + d2 Delta cos(2 pi k t/T) + u_t   (eq. 8).
    2. Detrend:  S~_t = y_t - psi - d0 t - d1 sin(.) - d2 cos(.), with psi such
       that S~_1 = 0  (eq. 9).
    3. OLS of  Delta y_t = phi S~_{t-1} + c0 + c1 Delta sin + c2 Delta cos
       + sum_{j=1..p} b_j Delta S~_{t-j} + e_t  on t = p+2..T  (eq. 10). tau_LM is
       the t ratio of phi; H0: unit root, small values reject.

    The deterministic part always includes a linear trend (it is estimated in
    first differences, so the test is invariant to its slope).

    The F statistic tests the Fourier terms as in the authors' simulation
    program for Table 1 (time-series.net, table1critvalues.PRG): with S^L the
    linearly detrended LM series (step 1 with a constant only), F compares
    Delta y on [S^L_{t-1}, 1, Delta sin, Delta cos, lags] with
    Delta y on [S^L_{t-1}, 1, lags]: F = ((SSR_r - SSR_u)/2) / (SSR_u/(N - K)).
    This is the restricted/unrestricted pair of the paper's Lemma 2.

    Parameters
    ----------
    data, y, time : as in ``oe.dfuller``.
    frequency : 1..5 or ``"auto"`` (default; smallest SSR of regression (10)
        over 1..max_frequency, ties to the lower frequency).
    max_frequency : largest candidate with ``"auto"`` (default 5).
    lags, maxlag : lagged Delta S~ terms, fixed or chosen per frequency by
        ``"aic"``, ``"bic"`` or ``"t"`` on the common sample (as in
        ``oe.fourier_adf``; Enders and Lee used the sequential t rule from 8).

    Critical values
    ---------------
    Enders and Lee (2012, Table 1): panel a for tau_LM (k = 1..5), panel b for
    F with a given k, panel c for the supremum F over k = 1..5 (only with
    ``frequency="auto"`` over 1..5); T = 100, 200, 500, 2500 interpolated
    linearly. ``p_value`` is None.

    Returns
    -------
    ``TableSet`` with ``test`` (rows ``tau`` and ``F``) and ``frequencies``;
    ``attrs`` as in ``oe.fourier_adf`` (``trend`` is always ``"trend"``).

    Example
    -------
    >>> oe.fourier_lm(df, "spread", frequency=1, lags="t", maxlag=8).attrs["statistic"]
    """
    candidates, mode = _frequencies(frequency, max_frequency)
    x, _ = one_series(data, y, time, "The Fourier LM test", minimum=12)
    x, _ = from_origin(x)
    total = x.shape[0]
    method, order = check_lag_option(lags, maxlag, total, 4, "the Fourier LM regression")
    change = differences(x)
    ones = torch.ones((total, 1), dtype=torch.float64)
    # The level counterpart of the differenced constant is t itself (Delta t = 1).
    index = torch.arange(1, total + 1, dtype=torch.float64)[:, None]
    linear = _lm_detrend(x, change, ones, index, "The linear LM detrending regression")
    linear_change = differences(linear)
    rows, best = [], None
    for k in candidates:
        waves = _fourier(total, k)
        steps = torch.cat([torch.zeros((1, 2), dtype=torch.float64), waves[1:] - waves[:-1]])
        deterministic = torch.cat([ones, steps], dim=1)
        what = f"The Fourier LM regression (k = {k})"
        detrended = _lm_detrend(x, change, deterministic, torch.cat([index, waves], dim=1),
                                f"The Fourier LM detrending regression (k = {k})")
        detrended_change = differences(detrended)
        chosen, common_ssr = _select_lags(detrended, detrended_change, deterministic,
                                          method, order, what)
        fit = _df_fit(detrended, detrended_change, deterministic, chosen, what)
        statistic = float(fit.beta[3] / fit.se[3])
        unrestricted = _df_fit(linear, linear_change, deterministic, chosen,
                               "The LM regression with Fourier terms")
        restricted = _df_fit(linear, linear_change, ones, chosen, "The linear LM regression")
        f_stat = _f_statistic(restricted.ssr, unrestricted)
        selection_ssr = fit.ssr if common_ssr is None else common_ssr
        row = {"frequency": k, "lags": chosen, "selection_ssr": selection_ssr,
               "statistic": statistic, "f_statistic": f_stat, "nobs": fit.nobs, "df_resid": fit.df}
        rows.append(row)
        if best is None or selection_ssr < best["selection_ssr"]:
            best = row
    k = best["frequency"]
    values, note = _critical(tables.LM_T, tables.LM_TAU[k - 1], total)
    f_reason = None
    if mode == "fixed":
        f_values, _ = _critical(tables.LM_T, tables.LM_F[k - 1], total)
        f_table = "panel b (given k)"
    elif len(candidates) == MAX_FREQUENCY:
        f_values, _ = _critical(tables.LM_T, tables.LM_SUPF, total)
        f_table = "panel c (supremum over k = 1..5)"
    else:
        f_values, f_table = None, None
        f_reason = ("Enders and Lee tabulate the supremum F over k = 1..5 only; with "
                    f"max_frequency={len(candidates)} no critical value is available.")
    statistic, f_stat = best["statistic"], best["f_statistic"]
    notes = ["Critical values: Enders and Lee (2012, Oxford Bulletin), Table 1, panel a, "
             "interpolated linearly in T; quantiles only, no p-value."]
    if f_table:
        notes.append(f"F critical values: Table 1, {f_table}.")
    if note:
        notes.append(note)
    if f_reason:
        notes.append(f_reason)
    if method != "fixed":
        notes.append("Lag orders were compared on the common sample t = maxlag+2..T; the "
                     "chosen order was refitted on t = p+2..T.")
    attrs = {
        "title": f"Fourier LM unit-root test for {y}", "test": "fourier_lm", "series": y,
        "statistic": statistic, "p_value": None, "p_value_reason": _P_VALUE_REASON,
        "inference_status": "critical_values_only",
        "distribution": "Fourier LM (Enders-Lee 2012)",
        "critical_values": values, "reject": _reject(statistic, values, lower=True),
        "f_statistic": f_stat, "f_critical_values": f_values,
        "f_reject": _reject(f_stat, f_values, lower=False), "f_inference_reason": f_reason,
        "frequency": k, "frequency_selection": mode, "max_frequency": max(candidates),
        "lags": best["lags"], "lag_method": method,
        "maxlag": order if method != "fixed" else None, "trend": "trend",
        "nobs": best["nobs"], "nobs_series": total, "df_resid": best["df_resid"],
        "critical_value_source": tables.SOURCES["fourier_lm"],
        "null": "Unit root", "alternative": "Trend stationarity around a smooth Fourier component",
        "label": "Fourier LM tau (H0: unit root); F tests the Fourier terms", "notes": notes,
    }
    test = table([[statistic, None, *critical_columns(values)],
                  [f_stat, None, *critical_columns(f_values)]],
                 columns=["statistic", "p_value", *CRITICAL_NAMES], index=["tau", "F"], **attrs)
    frequencies = table([[row["frequency"], row["lags"], row["selection_ssr"], row["statistic"],
                          row["f_statistic"]] for row in rows],
                        columns=["frequency", "lags", "selection_ssr", "statistic", "f_statistic"])
    return TableSet({"test": test, "frequencies": frequencies}, **attrs)
