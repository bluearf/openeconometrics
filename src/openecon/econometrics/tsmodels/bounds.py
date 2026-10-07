"""Asymptotic critical-value bounds of the Pesaran-Shin-Smith (2001) bounds test.

Source: Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing
approaches to the analysis of level relationships. Journal of Applied
Econometrics 16(3), 289-326: Table CI(i)-CI(v) (F statistic, cases I-V) and
Table CII(i), CII(iii), CII(v) (t statistic, cases I, III and V; the paper gives
no t bounds for cases II and IV). k is the number of regressors in levels
(the forcing variables x, excluding the lagged dependent variable).

Each entry is ``(I(0) bound, I(1) bound)``: the critical value when every
regressor is I(0), and when every regressor is I(1). The 10%, 5% and 1% levels
are tabulated here; the paper's 2.5% column is not included. The values were
typed in from the published tables and agree with the independent
transcription in the ``pss``/``pssbounds`` module of Jordan and Philips (2018;
CRAN package nardl); tests cross-check every F entry against a 32-million
replication simulation (statsmodels' ``pss_critical_values``), which agrees
within Monte Carlo error. Case III's 1% I(0) t bound is -3.43 for every k (one
transcription prints -3.42 for k = 9).

No p-values: the paper tabulates critical values only.
"""

from __future__ import annotations

LEVELS = (0.10, 0.05, 0.01)

CASES = {
    1: "Case I: no intercept, no trend",
    2: "Case II: restricted intercept, no trend",
    3: "Case III: unrestricted intercept, no trend",
    4: "Case IV: unrestricted intercept, restricted trend",
    5: "Case V: unrestricted intercept, unrestricted trend",
}

# F statistic. Rows k = 0..10; per row (I0, I1) at 10%, 5%, 1%.
F_BOUNDS: dict[int, tuple[tuple[float, ...], ...]] = {
    1: (  # Table CI(i)
        (3.00, 3.00, 4.20, 4.20, 7.17, 7.17),
        (2.44, 3.28, 3.15, 4.11, 4.81, 6.02),
        (2.17, 3.19, 2.72, 3.83, 3.88, 5.30),
        (2.01, 3.10, 2.45, 3.63, 3.42, 4.84),
        (1.90, 3.01, 2.26, 3.48, 3.07, 4.44),
        (1.81, 2.93, 2.14, 3.34, 2.82, 4.21),
        (1.75, 2.87, 2.04, 3.24, 2.66, 4.05),
        (1.70, 2.83, 1.97, 3.18, 2.54, 3.91),
        (1.66, 2.79, 1.91, 3.11, 2.45, 3.79),
        (1.63, 2.75, 1.86, 3.05, 2.34, 3.68),
        (1.60, 2.72, 1.82, 2.99, 2.26, 3.60),
    ),
    2: (  # Table CI(ii)
        (3.80, 3.80, 4.60, 4.60, 6.44, 6.44),
        (3.02, 3.51, 3.62, 4.16, 4.94, 5.58),
        (2.63, 3.35, 3.10, 3.87, 4.13, 5.00),
        (2.37, 3.20, 2.79, 3.67, 3.65, 4.66),
        (2.20, 3.09, 2.56, 3.49, 3.29, 4.37),
        (2.08, 3.00, 2.39, 3.38, 3.06, 4.15),
        (1.99, 2.94, 2.27, 3.28, 2.88, 3.99),
        (1.92, 2.89, 2.17, 3.21, 2.73, 3.90),
        (1.85, 2.85, 2.11, 3.15, 2.62, 3.77),
        (1.80, 2.80, 2.04, 3.08, 2.50, 3.68),
        (1.76, 2.77, 1.98, 3.04, 2.41, 3.61),
    ),
    3: (  # Table CI(iii)
        (6.58, 6.58, 8.21, 8.21, 11.79, 11.79),
        (4.04, 4.78, 4.94, 5.73, 6.84, 7.84),
        (3.17, 4.14, 3.79, 4.85, 5.15, 6.36),
        (2.72, 3.77, 3.23, 4.35, 4.29, 5.61),
        (2.45, 3.52, 2.86, 4.01, 3.74, 5.06),
        (2.26, 3.35, 2.62, 3.79, 3.41, 4.68),
        (2.12, 3.23, 2.45, 3.61, 3.15, 4.43),
        (2.03, 3.13, 2.32, 3.50, 2.96, 4.26),
        (1.95, 3.06, 2.22, 3.39, 2.79, 4.10),
        (1.88, 2.99, 2.14, 3.30, 2.65, 3.97),
        (1.83, 2.94, 2.06, 3.24, 2.54, 3.86),
    ),
    4: (  # Table CI(iv)
        (5.37, 5.37, 6.29, 6.29, 8.26, 8.26),
        (4.05, 4.49, 4.68, 5.15, 6.10, 6.73),
        (3.38, 4.02, 3.88, 4.61, 4.99, 5.85),
        (2.97, 3.74, 3.38, 4.23, 4.30, 5.23),
        (2.68, 3.53, 3.05, 3.97, 3.81, 4.92),
        (2.49, 3.38, 2.81, 3.76, 3.50, 4.63),
        (2.33, 3.25, 2.63, 3.62, 3.27, 4.39),
        (2.22, 3.17, 2.50, 3.50, 3.07, 4.23),
        (2.13, 3.09, 2.38, 3.41, 2.93, 4.06),
        (2.05, 3.02, 2.30, 3.33, 2.79, 3.93),
        (1.98, 2.97, 2.21, 3.25, 2.68, 3.84),
    ),
    5: (  # Table CI(v)
        (9.81, 9.81, 11.64, 11.64, 15.73, 15.73),
        (5.59, 6.26, 6.56, 7.30, 8.74, 9.63),
        (4.19, 5.06, 4.87, 5.85, 6.34, 7.52),
        (3.47, 4.45, 4.01, 5.07, 5.17, 6.36),
        (3.03, 4.06, 3.47, 4.57, 4.40, 5.72),
        (2.75, 3.79, 3.12, 4.25, 3.93, 5.23),
        (2.53, 3.59, 2.87, 4.00, 3.60, 4.90),
        (2.38, 3.45, 2.69, 3.83, 3.34, 4.63),
        (2.26, 3.34, 2.55, 3.68, 3.15, 4.43),
        (2.16, 3.24, 2.43, 3.56, 2.97, 4.24),
        (2.07, 3.16, 2.33, 3.46, 2.84, 4.10),
    ),
}

# t statistic on the lagged dependent variable (one-sided, reject for large negative values).
T_BOUNDS: dict[int, tuple[tuple[float, ...], ...]] = {
    1: (  # Table CII(i)
        (-1.62, -1.62, -1.95, -1.95, -2.58, -2.58),
        (-1.62, -2.28, -1.95, -2.60, -2.58, -3.22),
        (-1.62, -2.68, -1.95, -3.02, -2.58, -3.66),
        (-1.62, -3.00, -1.95, -3.33, -2.58, -3.97),
        (-1.62, -3.26, -1.95, -3.60, -2.58, -4.23),
        (-1.62, -3.49, -1.95, -3.83, -2.58, -4.44),
        (-1.62, -3.70, -1.95, -4.04, -2.58, -4.67),
        (-1.62, -3.90, -1.95, -4.23, -2.58, -4.88),
        (-1.62, -4.09, -1.95, -4.43, -2.58, -5.07),
        (-1.62, -4.26, -1.95, -4.61, -2.58, -5.25),
        (-1.62, -4.42, -1.95, -4.76, -2.58, -5.44),
    ),
    3: (  # Table CII(iii)
        (-2.57, -2.57, -2.86, -2.86, -3.43, -3.43),
        (-2.57, -2.91, -2.86, -3.22, -3.43, -3.82),
        (-2.57, -3.21, -2.86, -3.53, -3.43, -4.10),
        (-2.57, -3.46, -2.86, -3.78, -3.43, -4.37),
        (-2.57, -3.66, -2.86, -3.99, -3.43, -4.60),
        (-2.57, -3.86, -2.86, -4.19, -3.43, -4.79),
        (-2.57, -4.04, -2.86, -4.38, -3.43, -4.99),
        (-2.57, -4.23, -2.86, -4.57, -3.43, -5.19),
        (-2.57, -4.40, -2.86, -4.72, -3.43, -5.37),
        (-2.57, -4.56, -2.86, -4.88, -3.43, -5.54),
        (-2.57, -4.69, -2.86, -5.03, -3.43, -5.68),
    ),
    5: (  # Table CII(v)
        (-3.13, -3.13, -3.41, -3.41, -3.96, -3.97),
        (-3.13, -3.40, -3.41, -3.69, -3.96, -4.26),
        (-3.13, -3.63, -3.41, -3.95, -3.96, -4.53),
        (-3.13, -3.84, -3.41, -4.16, -3.96, -4.73),
        (-3.13, -4.04, -3.41, -4.36, -3.96, -4.96),
        (-3.13, -4.21, -3.41, -4.52, -3.96, -5.13),
        (-3.13, -4.37, -3.41, -4.69, -3.96, -5.31),
        (-3.13, -4.53, -3.41, -4.85, -3.96, -5.49),
        (-3.13, -4.68, -3.41, -5.01, -3.96, -5.65),
        (-3.13, -4.82, -3.41, -5.15, -3.96, -5.79),
        (-3.13, -4.96, -3.41, -5.29, -3.96, -5.94),
    ),
}


def critical_values(statistic: str, case: int, k: int) -> dict[str, dict[str, float]] | None:
    """``{"10%": {"I0": ..., "I1": ...}, "5%": ..., "1%": ...}`` or None when not tabulated."""
    table = (F_BOUNDS if statistic == "F" else T_BOUNDS).get(case)
    if table is None or not 0 <= k < len(table):
        return None
    row = table[k]
    return {f"{level:.0%}": {"I0": row[2 * i], "I1": row[2 * i + 1]}
            for i, level in enumerate(LEVELS)}


def decide(statistic: str, value: float, bounds: dict[str, dict[str, float]]) -> dict[str, str]:
    """Bounds-test decision at each tabulated level.

    F: reject 'no level relationship' above the I(1) bound, do not reject below
    the I(0) bound, inconclusive in between. t: the same with the inequalities
    reversed (large negative values reject).
    """
    decisions = {}
    for level, bound in bounds.items():
        low, high = bound["I0"], bound["I1"]
        if statistic == "F":
            reject, accept = value > high, value < low
        else:
            reject, accept = value < high, value > low
        decisions[level] = ("reject H0 (no level relationship)" if reject else
                            "do not reject H0" if accept else "inconclusive")
    return decisions
