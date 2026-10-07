"""Scalar reference-distribution calculations for float64 statistical results.

Student-t tails are evaluated through the regularized incomplete beta using a
modified Lentz continued fraction. Quantiles use a bracketed monotone solve.
These small scalar calculations have independent SciPy-oracle tests; no SciPy
distribution or third-party estimator runs in production inference.

Large finite degrees of freedom use a transformed Student-t density integral
with its finite-df corrections retained. This avoids subtracting almost equal
log-gamma values and does not replace Student-t inference with normal inference.
"""

from __future__ import annotations

import math

import torch

from .contracts import KernelError


def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = math.copysign(tiny, d)
    d = 1 / d
    result = d
    for m in range(1, 10001):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + numerator * d
        if abs(d) < tiny:
            d = math.copysign(tiny, d)
        c = 1 + numerator / c
        if abs(c) < tiny:
            c = math.copysign(tiny, c)
        d = 1 / d
        result *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + numerator * d
        if abs(d) < tiny:
            d = math.copysign(tiny, d)
        c = 1 + numerator / c
        if abs(c) < tiny:
            c = math.copysign(tiny, c)
        d = 1 / d
        delta = d * c
        result *= delta
        if abs(delta - 1) <= 4e-15:
            return result
    raise KernelError("inference_nonconvergence", "The Student-t tail calculation did not converge.")


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1) / (a + b + 2):
        result = front * _beta_fraction(a, b, x) / a
    else:
        result = 1 - front * _beta_fraction(b, a, 1 - x) / b
    return min(1.0, max(0.0, result))


def _large_df_student_t_two_sided(value: float, df: int | float) -> float:
    """Stable finite-df tails for df >= 1e6, including very small tails.

    Set z = sqrt(df * log1p(t**2 / df)) in the Student-t density integral.
    Its ratio to the standard normal density becomes

        C(df) * sqrt(a / (1 - exp(-a))),  a = z**2 / df,

    where C(df) = sqrt(2/df) * Gamma((df+1)/2) / Gamma(df/2).
    Integrate the first seven Taylor terms of the square-root factor using
    normal tail moments. The log C expansion follows Stirling's series;
    see https://dlmf.nist.gov/5.11.E8 and the Student-t density definition at
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm.

    For df >= 1e6 and transformed z <= 39, a <= .004096 throughout the
    effective integration range z..64. A Cauchy bound on the unit circle
    bounds the omitted Taylor terms there below 5e-17; the integral above
    64 is below float64's subnormal range. Larger z already has a tail below
    float64's range. Finite-df terms remain measurable near df = 1e6.
    """
    scaled_value = value / math.sqrt(df)
    ratio = scaled_value * scaled_value
    if math.isinf(ratio):
        return 0.0
    # This form avoids df * log1p(ratio) overflow and preserves near-zero t.
    z = value if ratio == 0 else value * math.sqrt(math.log1p(ratio) / ratio)
    if z > 39:
        return 0.0
    inverse_df = 1 / df
    inverse_df_squared = inverse_df * inverse_df
    log_normalizer = (-inverse_df / 4 + inverse_df * inverse_df_squared / 24
                      - inverse_df * inverse_df_squared * inverse_df_squared / 20)
    moment = 0.5 * math.erfc(z / math.sqrt(2))
    density = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    terms = [moment]
    power = inverse_df
    z_power = z
    # Taylor coefficients of sqrt(a / (1 - exp(-a))), excluding c0 = 1.
    coefficients = (1 / 4, 1 / 96, -1 / 384, -1 / 10240,
                    19 / 368640, 79 / 61931520)
    for order, coefficient in enumerate(coefficients, start=1):
        # J_m(z) = integral_z^infinity w**m phi(w) dw.
        moment = z_power * density + (2 * order - 1) * moment
        terms.append(coefficient * power * moment)
        power *= inverse_df
        z_power *= z * z
    return min(1.0, max(0.0, 2 * math.exp(log_normalizer) * math.fsum(terms)))


def student_t_two_sided(statistic: float, df: int | float) -> float:
    """P(|T_df| >= |statistic|), including stable values near zero."""
    if df <= 0 or not math.isfinite(df):
        raise KernelError("invalid_inference", "Student-t degrees of freedom must be positive and finite.")
    value = abs(float(statistic))
    if math.isnan(value):
        raise KernelError("invalid_inference", "The test statistic must not be NaN.")
    if value == 0:
        return 1.0
    if math.isinf(value):
        return 0.0
    if df >= 1_000_000:
        return _large_df_student_t_two_sided(value, df)
    square = value * value
    # Subtracting a complementary CDF is safe only near the origin; doing so
    # for a small tail (e.g. t=8, df=117) destroys its significant digits.
    if square <= 1:
        complement = square / (df + square)
        return 1 - _regularized_beta(0.5, df / 2, complement)
    x = (df / value) / value if math.isinf(square) else df / (df + square)
    return _regularized_beta(df / 2, 0.5, x)


def critical_value(alpha: float, df: int | None = None) -> float:
    """Positive two-sided critical value; df=None selects the standard normal."""
    if not 0 < alpha < 1:
        raise KernelError("invalid_inference", "The significance level must lie between zero and one.")
    if df is None:
        critical = -float(torch.special.ndtri(torch.tensor(alpha / 2, dtype=torch.float64)))
        if not math.isfinite(critical):
            raise KernelError("invalid_inference", "The requested confidence level exceeds float64 precision.")
        return critical
    low, high = 0.0, 1.0
    while student_t_two_sided(high, df) > alpha:
        high *= 2
        if not math.isfinite(high):
            raise KernelError("invalid_inference", "The requested Student-t quantile exceeds float64 range.")
    for _ in range(200):
        midpoint = (low + high) / 2
        if student_t_two_sided(midpoint, df) > alpha:
            low = midpoint
        else:
            high = midpoint
        if high - low <= 2e-14 * max(1.0, midpoint):
            break
    return (low + high) / 2


def two_sided_p_values(statistics: torch.Tensor, df: int | None = None) -> torch.Tensor:
    """Return float64 p-values on the same device as the statistics."""
    if df is None:
        return torch.special.erfc(statistics.abs() / math.sqrt(2))
    return torch.tensor([student_t_two_sided(value, df) for value in statistics.tolist()],
                        dtype=torch.float64, device=statistics.device)
