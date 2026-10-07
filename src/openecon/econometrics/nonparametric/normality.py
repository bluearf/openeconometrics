"""Normality tests: swilk (Shapiro-Wilk), sfrancia (Shapiro-Francia), sktest.

The Shapiro-Wilk coefficients and p-value follow Royston's algorithm AS R94
(Applied Statistics 44, 1995), valid for 3 <= n <= 5000: the weights are built
from the normal scores m_i = Phi^{-1}((i - 3/8) / (n + 1/4)) with polynomial
corrections of the two largest weights, and ln(1 - W) is transformed to
normality with the published polynomial mean and standard deviation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.nonparametric.common import FLOAT, numeric, procedure, select
from openecon.engines.distributions import chi2_sf, normal_cdf, normal_isf, normal_sf

# Royston (1992, 1995) polynomial coefficients, lowest order first.
_C1 = (0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)
_C2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
_C3 = (0.544, -0.39978, 0.025054, -6.714e-4)
_C4 = (1.3822, -0.77857, 0.062767, -0.0020322)
_C5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
_C6 = (-0.4803, -0.082676, 0.0030302)
_G = (-2.273, 0.459)


def _poly(coefficients: tuple[float, ...], x: float) -> float:
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _sample(data: Any, y: Any, missing: str, low: int, high: int | None,
            what: str) -> tuple[Tensor, int]:
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    frame, dropped = select(data, [y], missing=missing)
    x = numeric(frame, y)
    n = x.numel()
    if n < low or (high is not None and n > high):
        bounds = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise AnalysisError("sample_size", f"{what} needs {bounds} observations (got {n}).")
    ordered = torch.sort(x).values
    if not float(ordered[-1] - ordered[0]) > 0.0:
        raise AnalysisError("no_variation", f"All values of '{y}' are equal.")
    return ordered, dropped


def normal_scores(n: int) -> Tensor:
    """Blom scores m_i = Phi^{-1}((i - 3/8) / (n + 1/4)), i = 1..n."""
    position = torch.arange(1, n + 1, dtype=FLOAT)
    return torch.special.ndtri((position - 0.375) / (n + 0.25))


def shapiro_wilk_weights(n: int) -> Tensor:
    """The n antisymmetric Shapiro-Wilk weights a_i of Royston's approximation."""
    if n == 3:
        return torch.tensor([-math.sqrt(0.5), 0.0, math.sqrt(0.5)], dtype=FLOAT)
    m = normal_scores(n)
    total = float((m * m).sum())
    root = math.sqrt(total)
    u = 1.0 / math.sqrt(n)
    weights = torch.empty(n, dtype=FLOAT)
    largest = float(m[-1]) / root + _poly(_C1, u)
    if n > 5:
        second = float(m[-2]) / root + _poly(_C2, u)
        scale = math.sqrt((total - 2.0 * float(m[-1]) ** 2 - 2.0 * float(m[-2]) ** 2)
                          / (1.0 - 2.0 * largest ** 2 - 2.0 * second ** 2))
        weights[2:-2] = m[2:-2] / scale
        weights[-2], weights[1] = second, -second
    else:
        scale = math.sqrt((total - 2.0 * float(m[-1]) ** 2) / (1.0 - 2.0 * largest ** 2))
        weights[1:-1] = m[1:-1] / scale
    weights[-1], weights[0] = largest, -largest
    return weights


def _squared_correlation_complement(ordered: Tensor, weights: Tensor) -> float:
    """1 - corr(x, a)^2, computed without cancellation for W close to 1."""
    spread = float(ordered[-1] - ordered[0])
    x = ordered / spread
    x = x - x.mean()
    a = weights - weights.mean()
    ssa, ssx, sax = float((a * a).sum()), float((x * x).sum()), float((a * x).sum())
    root = math.sqrt(ssa * ssx)
    return (root - sax) * (root + sax) / (ssa * ssx)


@procedure
def swilk(data: Any, y: str, *, missing: str = "drop"):
    """Shapiro-Wilk W test for normality (Royston's AS R94, 3 <= n <= 5000).

    W = (sum_i a_i x_(i))^2 / sum_i (x_i - xbar)^2, where x_(i) are the ordered
    values and a_i the weights of Royston's (1992) approximation. Small W
    indicates non-normality. The p-value transforms 1 - W to an approximately
    normal variable:

    * n = 3: exact, p = (6 / pi) (asin(sqrt(W)) - asin(sqrt(3/4)));
    * 4 <= n <= 11: z = (-ln(g - ln(1 - W)) - m) / s with g, m, s polynomials in n;
    * n >= 12: z = (ln(1 - W) - m) / s with m, s polynomials in ln n;

    p = 1 - Phi(z). This is the algorithm of SPSS (EXAMINE), R and SciPy;
    Stata's ``swilk`` uses the same transformation (documented for
    4 <= n <= 2000).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric column with 3 to 5000 non-missing values.
    missing : "drop" (missing values are removed and counted in
        ``n_dropped``) or "raise".

    Returns
    -------
    One table (row = y): n, statistic (W), z, p_value; the same in ``attrs``.

    Stata: ``swilk y``. SPSS: ``EXAMINE VARIABLES=y /PLOT NPPLOT``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [2.1, 3.4, 1.9, 5.6, 4.4, 3.8, 2.7, 9.5, 3.1, 4.0]}
    >>> round(oe.swilk(data, "x").attrs["statistic"], 4)
    0.8214
    """
    ordered, dropped = _sample(data, y, missing, 3, 5000, "The Shapiro-Wilk test")
    n = ordered.numel()
    complement = _squared_correlation_complement(ordered, shapiro_wilk_weights(n))
    w = 1.0 - complement
    if n == 3:
        p = max(0.0, 6.0 / math.pi * (math.asin(math.sqrt(min(w, 1.0))) - math.pi / 3.0))
        z = normal_isf(p) if 0.0 < p < 1.0 else None
    else:
        log_complement = math.log(max(complement, 1e-300))
        if n <= 11:
            gamma = _poly(_G, float(n))
            if log_complement >= gamma:
                z, p = None, 0.0
            else:
                z = (-math.log(gamma - log_complement) - _poly(_C3, float(n))) \
                    / math.exp(_poly(_C4, float(n)))
                p = normal_sf(z)
        else:
            log_n = math.log(n)
            z = (log_complement - _poly(_C5, log_n)) / math.exp(_poly(_C6, log_n))
            p = normal_sf(z)
    attrs = {"n": n, "statistic": w, "z": z, "p_value": p, "n_dropped": dropped,
             "label": "Shapiro-Wilk W test for normal data"}
    return table([[n, w, z, p]], columns=["n", "statistic", "z", "p_value"], index=[y], **attrs)


@procedure
def sfrancia(data: Any, y: str, *, missing: str = "drop"):
    """Shapiro-Francia W' test for normality (5 <= n <= 5000).

    W' is the squared correlation between the ordered values and the normal
    scores m_i = Phi^{-1}((i - 3/8) / (n + 1/4)). Royston (1993) normalizes it
    with u = ln n, v = ln u:

        mu = -1.2725 + 1.0521 (v - u),   sigma = 1.0308 - 0.26758 (v + 2 / u),
        z = (ln(1 - W') - mu) / sigma,   p = 1 - Phi(z).

    This is the default of Stata's ``sfrancia`` (its ``boxcox`` option selects
    an older transformation that is not implemented here).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric column with 5 to 5000 non-missing values.
    missing : "drop" (missing values are removed and counted in
        ``n_dropped``) or "raise".

    Returns
    -------
    One table (row = y): n, statistic (W'), z, p_value; the same in ``attrs``.

    Stata: ``sfrancia y``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [2.1, 3.4, 1.9, 5.6, 4.4, 3.8, 2.7, 9.5, 3.1, 4.0]}
    >>> round(oe.sfrancia(data, "x").attrs["statistic"], 4)
    0.8045
    """
    ordered, dropped = _sample(data, y, missing, 5, 5000, "The Shapiro-Francia test")
    n = ordered.numel()
    complement = _squared_correlation_complement(ordered, normal_scores(n))
    u = math.log(n)
    v = math.log(u)
    mu = -1.2725 + 1.0521 * (v - u)
    sigma = 1.0308 - 0.26758 * (v + 2.0 / u)
    z = (math.log(max(complement, 1e-300)) - mu) / sigma
    attrs = {"n": n, "statistic": 1.0 - complement, "z": z, "p_value": normal_sf(z),
             "n_dropped": dropped, "label": "Shapiro-Francia W' test for normal data"}
    return table([[n, attrs["statistic"], z, attrs["p_value"]]],
                 columns=["n", "statistic", "z", "p_value"], index=[y], **attrs)


def skewness_z(skewness: float, n: int) -> float:
    """D'Agostino's (1970) normalizing transformation of the sample skewness."""
    y = skewness * math.sqrt((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0)))
    beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) \
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))
    w2 = -1.0 + math.sqrt(2.0 * (beta2 - 1.0))
    delta = 1.0 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1.0))
    return delta * math.asinh(y / alpha)


def kurtosis_z(kurtosis: float, n: int) -> float:
    """Anscombe and Glynn's (1983) normalizing transformation of the sample kurtosis b2."""
    mean = 3.0 * (n - 1.0) / (n + 1.0)
    variance = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
    x = (kurtosis - mean) / math.sqrt(variance)
    root_beta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) \
        * math.sqrt(6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0)))
    a = 6.0 + 8.0 / root_beta1 * (2.0 / root_beta1 + math.sqrt(1.0 + 4.0 / root_beta1 ** 2))
    denominator = 1.0 + x * math.sqrt(2.0 / (a - 4.0))
    if denominator == 0.0:
        return -38.0                      # the transformation diverges: p underflows to 0
    cube = math.copysign(((1.0 - 2.0 / a) / abs(denominator)) ** (1.0 / 3.0), denominator)
    return ((1.0 - 2.0 / (9.0 * a)) - cube) / math.sqrt(2.0 / (9.0 * a))


def royston_adjusted_chi2(k2: float, n: int) -> float:
    """Royston's (1991) small-sample adjustment of the joint skewness-kurtosis chi2.

    With Zc = -Phi^{-1}(exp(-K2 / 2)) (the unadjusted p as a normal deviate),
    Zt = 0.55 n^0.2 - 0.21, a1 = (-5 + 3.46 ln n) exp(-1.37 ln n),
    b1 = 1 + (0.854 - 0.148 ln n) exp(-0.55 ln n), c = 2.13 / (1 - 2.37 ln n),
    b2 = b1 + c and a2 = a1 - c Zt:

        Z = Zc                 if Zc < -1,
            a1 + b1 Zc         if -1 <= Zc < Zt,
            a2 + b2 Zc         otherwise;

    the adjusted statistic is -2 ln(1 - Phi(Z)), referred to chi2(2).
    """
    p = min(1.0, max(math.exp(-0.5 * k2), 1e-300))
    if p >= 1.0:
        return 0.0
    zc = normal_isf(p)
    log_n = math.log(n)
    zt = 0.55 * n ** 0.2 - 0.21
    a1 = (-5.0 + 3.46 * log_n) * math.exp(-1.37 * log_n)
    b1 = 1.0 + (0.854 - 0.148 * log_n) * math.exp(-0.55 * log_n)
    c = 2.13 / (1.0 - 2.37 * log_n)
    if zc < -1.0:
        z = zc
    elif zc < zt:
        z = a1 + b1 * zc
    else:
        z = (a1 - c * zt) + (b1 + c) * zc
    return -2.0 * math.log(max(normal_sf(z), 1e-300))


@procedure
def sktest(data: Any, y: str, *, missing: str = "drop"):
    """Skewness and kurtosis tests for normality (Stata's ``sktest``) and Jarque-Bera.

    With the moment coefficients sqrt(b1) = m3 / m2^(3/2) and b2 = m4 / m2^2
    (m_r the r-th central moment with divisor n):

    * skewness: D'Agostino's (1970) transformation Z1 of sqrt(b1);
    * kurtosis: Anscombe and Glynn's (1983) transformation Z2 of b2;
    * joint: K2 = Z1^2 + Z2^2 ~ chi2(2), the D'Agostino-Pearson omnibus test
      (row ``joint``; Stata's ``sktest, noadjust``, SciPy's ``normaltest``);
    * joint_adjusted: K2 with Royston's (1991) empirical adjustment, the
      default line of Stata's ``sktest``;
    * jarque_bera: n/6 [b1 + (b2 - 3)^2 / 4] ~ chi2(2), the asymptotic test
      without small-sample transformations.

    At least 8 observations are required (as in Stata).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric column with at least 8 non-missing values.
    missing : "drop" (missing values are removed and counted in
        ``n_dropped``) or "raise".

    Returns
    -------
    One table with rows skewness, kurtosis, joint_adjusted, joint,
    jarque_bera and columns statistic (the coefficient or chi2), z, df,
    p_value. ``attrs``: n, skewness, kurtosis, z_skewness, z_kurtosis,
    p_skewness, p_kurtosis, statistic and p_value (adjusted joint test),
    chi2_unadjusted, p_unadjusted, jarque_bera, p_jarque_bera.

    Stata: ``sktest y``. SPSS reports the coefficients in ``EXAMINE`` /
    ``FREQUENCIES`` (with the bias-corrected g1, g2, which are not used here).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [2.1, 3.4, 1.9, 5.6, 4.4, 3.8, 2.7, 9.5, 3.1, 4.0]}
    >>> round(oe.sktest(data, "x").attrs["skewness"], 4)
    1.5749
    """
    ordered, dropped = _sample(data, y, missing, 8, None, "sktest")
    n = ordered.numel()
    centred = (ordered - ordered.mean()) / float(ordered[-1] - ordered[0])
    m2 = float((centred ** 2).mean())
    skewness = float((centred ** 3).mean()) / m2 ** 1.5
    kurtosis = float((centred ** 4).mean()) / (m2 * m2)
    z1, z2 = skewness_z(skewness, n), kurtosis_z(kurtosis, n)
    p1, p2 = 2.0 * normal_cdf(-abs(z1)), 2.0 * normal_cdf(-abs(z2))
    k2 = z1 * z1 + z2 * z2
    adjusted = royston_adjusted_chi2(k2, n)
    jb = n / 6.0 * (skewness ** 2 + (kurtosis - 3.0) ** 2 / 4.0)
    attrs = {"n": n, "skewness": skewness, "kurtosis": kurtosis, "z_skewness": z1,
             "z_kurtosis": z2, "p_skewness": p1, "p_kurtosis": p2, "statistic": adjusted,
             "df": 2, "p_value": chi2_sf(adjusted, 2), "chi2_unadjusted": k2,
             "p_unadjusted": chi2_sf(k2, 2), "jarque_bera": jb, "p_jarque_bera": chi2_sf(jb, 2),
             "distribution": "chi2", "n_dropped": dropped,
             "label": "Skewness and kurtosis tests for normality"}
    rows = [[skewness, z1, None, p1], [kurtosis, z2, None, p2],
            [adjusted, None, 2, attrs["p_value"]], [k2, None, 2, attrs["p_unadjusted"]],
            [jb, None, 2, attrs["p_jarque_bera"]]]
    return table(rows, columns=["statistic", "z", "df", "p_value"],
                 index=["skewness", "kurtosis", "joint_adjusted", "joint", "jarque_bera"], **attrs)
