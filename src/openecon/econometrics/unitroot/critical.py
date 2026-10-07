"""p-values and critical values from the published tables in ``tables.py``.

All functions work on Python floats. Interpolation rules are stated per
function; outside the tabulated range the nearest tabulated row is used and the
caller is told through the returned note.
"""

from __future__ import annotations

import math

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot import tables
from openecon.engines.distributions import chi2_sf, normal_cdf, normal_sf

# Deterministic terms of a test regression, as MacKinnon labels them.
TREND_CODES = {"none": "n", "constant": "c", "trend": "ct", "quadratic": "ctt"}
MAX_SERIES = 6


def _polynomial(coefficients: tuple[float, ...], x: float) -> float:
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def mackinnon_p(statistic: float, case: str, n_series: int = 1) -> float:
    """MacKinnon (1994) approximate asymptotic p-value of a tau statistic.

    ``case`` is "n", "c", "ct" or "ctt"; ``n_series`` is 1 for the
    Dickey-Fuller test and 1 + (number of regressors) for the Engle-Granger
    residual test (at most 6).
    """
    row = n_series - 1
    if statistic > tables.TAU_MAX[case][row]:
        return 1.0
    if statistic < tables.TAU_MIN[case][row]:
        return 0.0
    if statistic <= tables.TAU_STAR[case][row]:
        raw, scale = tables.TAU_SMALL_P[case][row], tables.TAU_SMALL_SCALE
    else:
        raw, scale = tables.TAU_LARGE_P[case][row], tables.TAU_LARGE_SCALE
    coefficients = tuple(value * unit for value, unit in zip(raw, scale, strict=True))
    return normal_cdf(_polynomial(coefficients, statistic))


def mackinnon_critical(case: str, n_series: int, nobs: float) -> dict[str, float] | None:
    """MacKinnon (2010) finite-sample critical values ``b_inf + b1/T + b2/T^2 + b3/T^3``.

    Returns ``None`` when the case is not tabulated (no constant with more
    than one series).
    """
    rows = tables.TAU_CRIT_2010[case]
    if n_series > len(rows):
        return None
    inverse = 0.0 if math.isinf(nobs) else 1.0 / nobs
    return {level: _polynomial(coefficients, inverse)
            for level, coefficients in zip(tables.LEVELS, rows[n_series - 1], strict=True)}


def mackinnon_asymptotic_critical(case: str, n_series: int) -> dict[str, float]:
    """Asymptotic critical values implied by the MacKinnon (1994) distribution function.

    Solves ``mackinnon_p(c) = level`` by bisection; used where the 2010 response
    surface has no entry.
    """
    result = {}
    for level, target in zip(tables.LEVELS, (0.01, 0.05, 0.10), strict=True):
        low, high = -12.0, 0.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            if mackinnon_p(middle, case, n_series) < target:
                low = middle
            else:
                high = middle
        result[level] = 0.5 * (low + high)
    return result


def _interpolate_rows(nodes: tuple[float, ...], rows: tuple[tuple[float, ...], ...], n: float,
                      *, reciprocal_tail: bool) -> tuple[float, ...]:
    """Row at ``n`` by linear interpolation in n between tabulated sample sizes.

    Below the first node the first row is returned. Between the last finite
    node and infinity the interpolation is linear in 1/n when
    ``reciprocal_tail`` is true; otherwise the asymptotic row is used there.
    """
    if n <= nodes[0]:
        return rows[0]
    for index in range(1, len(nodes)):
        left, right = nodes[index - 1], nodes[index]
        if n <= right:
            if math.isinf(right):
                if not reciprocal_tail:
                    return rows[index]
                weight = 1.0 - left / n
            else:
                weight = (n - left) / (right - left)
            return tuple(a + weight * (b - a)
                         for a, b in zip(rows[index - 1], rows[index], strict=True))
    return rows[-1]


def fuller_rho_critical(case: str, nobs: int) -> dict[str, float]:
    """Critical values of n(rho-hat - 1), i.e. Z(rho), from Fuller's table.

    Linear interpolation in n between the tabulated sample sizes 25, 50, 100,
    250 and 500 (as Stata's pperron prints them); the n = 25 row below 25 and
    the asymptotic row above 500.
    """
    row = _interpolate_rows(tables.FULLER_RHO_N, tables.FULLER_RHO[case], float(nobs),
                            reciprocal_tail=False)
    return dict(zip(tables.LEVELS, row, strict=True))


def fuller_tau_critical(case: str, nobs: int) -> dict[str, float]:
    """Dickey-Fuller t critical values interpolated in Fuller's table, as Stata prints them.

    Linear interpolation in n between the tabulated sample sizes 25, 50, 100,
    250 and 500; the n = 25 row below 25 and the asymptotic row above 500.
    """
    row = _interpolate_rows(tables.FULLER_RHO_N, tables.FULLER_TAU[case], float(nobs),
                            reciprocal_tail=False)
    return dict(zip(tables.LEVELS, row, strict=True))


def ers_trend_critical(nobs: int) -> dict[str, float]:
    """DF-GLS critical values with a linear trend (Elliott-Rothenberg-Stock Table I).

    Linear interpolation in T between T = 50, 100 and 200; the T = 50 row below
    50 and the asymptotic row above 200, as Stata's ``dfgls, ers`` (which
    evaluates the table at the number of observations of the series).
    """
    row = _interpolate_rows(tables.ERS_T, tables.ERS_TREND, float(nobs), reciprocal_tail=False)
    return dict(zip(tables.LEVELS, row, strict=True))


def kpss_p(statistic: float, case: str) -> tuple[float, str | None]:
    """p-value interpolated linearly between the four KPSS (1992) critical values.

    The table covers 0.01 <= p <= 0.10 only: outside it the bound is returned
    with a note saying on which side the true p-value lies.
    """
    critical, levels = tables.KPSS_CRIT[case], tables.KPSS_LEVELS
    if statistic <= critical[0]:
        return levels[0], "The p-value is above 0.10 (the statistic is below the 10% point)."
    if statistic >= critical[-1]:
        return levels[-1], "The p-value is below 0.01 (the statistic is above the 1% point)."
    for index in range(1, len(critical)):
        if statistic <= critical[index]:
            weight = (statistic - critical[index - 1]) / (critical[index] - critical[index - 1])
            return levels[index - 1] + weight * (levels[index] - levels[index - 1]), None
    return levels[-1], None


def kpss_critical(case: str) -> dict[str, float]:
    return {"10%": tables.KPSS_CRIT[case][0], "5%": tables.KPSS_CRIT[case][1],
            "2.5%": tables.KPSS_CRIT[case][2], "1%": tables.KPSS_CRIT[case][3]}


def zivot_andrews_critical(model: str) -> dict[str, float]:
    return dict(zip(tables.LEVELS, tables.ZIVOT_ANDREWS[model], strict=True))


def llc_adjustment(case: str, t_tilde: float) -> tuple[float, float, str | None]:
    """(mu*, sigma*, note) of Levin-Lin-Chu Table 2 at the average usable T.

    Linear interpolation in T between tabulated rows, linear in 1/T beyond
    T = 250; the T = 25 row is used below 25 (with a note).
    """
    note = None
    if t_tilde < tables.LLC_T[0]:
        note = (f"The adjustment terms are tabulated from T = 25; the T = 25 row is used for "
                f"T = {t_tilde:g}.")
    mean, deviation = _interpolate_rows(tables.LLC_T, tables.LLC_ADJUST[case], t_tilde,
                                        reciprocal_tail=True)
    return mean, deviation, note


def ips_moments(case: str, lags: int, nobs: int) -> tuple[float, float]:
    """(E[t], Var[t]) of Im-Pesaran-Shin Table 3 for one panel.

    ``nobs`` is the number of observations in the panel's ADF regression.
    Linear interpolation in T between tabulated columns; the T = 100 column is
    used beyond 100. Lag orders above 8, T below 10 and the blank cells of the
    table raise ``ips_moments_unavailable``.
    """
    if lags >= len(tables.IPS_MEAN[case]):
        raise AnalysisError("ips_moments_unavailable", "Im-Pesaran-Shin tabulate the moments of "
                            f"the ADF t statistic for at most 8 lags; a panel uses {lags}. "
                            "Reduce lags.")
    nodes = tables.IPS_T
    means, variances = tables.IPS_MEAN[case][lags], tables.IPS_VARIANCE[case][lags]
    available = [index for index, value in enumerate(means) if value is not None]
    if nobs < nodes[available[0]]:
        raise AnalysisError("ips_moments_unavailable", "Im-Pesaran-Shin tabulate the moments of "
                            f"the ADF t statistic with {lags} lag(s) from "
                            f"{nodes[available[0]]:g} observations per panel; a panel has "
                            f"{nobs} after lags. Reduce lags or drop short panels.")
    if nobs >= nodes[-1]:
        return float(means[-1]), float(variances[-1])
    for index in available[1:]:
        if nobs <= nodes[index]:
            left = index - 1
            weight = (nobs - nodes[left]) / (nodes[index] - nodes[left])
            return (means[left] + weight * (means[index] - means[left]),
                    variances[left] + weight * (variances[index] - variances[left]))
    return float(means[-1]), float(variances[-1])


def supwald_p(statistic: float, restrictions: int, low: float, high: float) -> float:
    """Approximate asymptotic p-value of a sup-Wald statistic over [low, high].

    The tail approximation of DeLong (1981) quoted by Andrews (1993) for the
    supremum of a normalized squared tied-down Bessel process of order q:

        P(sup > c) ~ c^(q/2) e^(-c/2) / (2^(q/2) Gamma(q/2))
                     * [(1 - q/c) ln(lambda) + 2/c],
        lambda = high (1 - low) / (low (1 - high)),

    with ``low`` and ``high`` the first and last candidate break fractions. It
    is accurate in the upper tail (it reproduces the tabulated 10%, 5% and 1%
    points to within about a fifth of their level, and simulated quantiles
    down to a p-value of about 0.5); above 0.5 it understates the p-value, and
    it is capped at 1. With a single candidate date the statistic is an
    ordinary Wald statistic and the chi-squared(q) tail is returned.
    """
    q, c = float(restrictions), float(statistic)
    if low >= high:
        return chi2_sf(c, restrictions)
    if c <= q:
        return 1.0
    ratio = high * (1.0 - low) / (low * (1.0 - high))
    density = math.exp(0.5 * q * math.log(c) - 0.5 * c - 0.5 * q * math.log(2.0)
                       - math.lgamma(0.5 * q))
    value = density * ((1.0 - q / c) * math.log(ratio) + 2.0 / c)
    return min(1.0, max(0.0, value))


def supwald_critical(restrictions: int, trim: float) -> dict[str, float] | None:
    """Andrews (2003) sup-Wald critical values for 15% trimming and q <= 20, else None."""
    if abs(trim - 0.15) > 1e-12 or not 1 <= restrictions <= len(tables.QLR_F_15):
        return None
    ten, five, one = tables.QLR_F_15[restrictions - 1]
    return {"1%": one * restrictions, "5%": five * restrictions, "10%": ten * restrictions}


def cusum_critical(alpha: float) -> float:
    """Brown-Durbin-Evans boundary parameter ``a`` of the recursive CUSUM test.

    The probability that a Brownian motion on [0, 1] crosses the pair of lines
    +-a (1 + 2 r) is approximately alpha when ``a`` solves

        Q(3a) + exp(-4 a^2) (1 - Q(a)) = alpha / 2,     Q = upper normal tail

    (Brown, Durbin and Evans 1975, eq. 2.6). Solved by bisection; it gives
    1.1430, 0.9479 and 0.8499 at 1%, 5% and 10%, which the article rounds to
    1.143, 0.948 and 0.850.
    """
    low, high = 0.1, 5.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        crossing = normal_sf(3.0 * middle) + math.exp(-4.0 * middle * middle) \
            * (1.0 - normal_sf(middle))
        if crossing > 0.5 * alpha:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def cusumsq_critical(recursive_count: int, level: str = "5%") -> float:
    """Edgerton-Wells (1994) approximation to the CUSUM-of-squares critical value."""
    m = 0.5 * recursive_count - 1.0
    a1, a2, a3 = tables.CUSUMSQ_COEFFICIENTS[level]
    return a1 / math.sqrt(m) + a2 / m + a3 / m ** 1.5
