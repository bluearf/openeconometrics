"""Critical values of Johansen's trace and maximum-eigenvalue statistics.

Source: M. Osterwald-Lenum (1992), "A note with quantiles of the asymptotic
distribution of the maximum likelihood cointegration rank test statistics",
Oxford Bulletin of Economics and Statistics 54, 461-472. These are the values
Stata's ``vecrank`` reports. The tables correspond to Stata's trend()
specifications as follows:

    trend        Osterwald-Lenum table   deterministic terms
    none         Table 0                 none
    rconstant    Table 1*                constant restricted to the cointegrating equations
    constant     Table 1                 unrestricted constant
    rtrend       Table 2*                unrestricted constant, restricted trend
    trend        Table 2                 unrestricted constant and trend

Each list is indexed by the number of common trends under the null,
K - r = 1, 2, ..., 11 (the dimension of the limiting functional).

Provenance of this transcription (the values are data, not computed here).
The numbers were not typed from the printed paper; every entry was compared
with independent published transcriptions of the paper's tables:

- All five tables, all 11 dimensions, 5% and 1%: identical to the matrices in
  ``johans.ado`` (K. Heinecke, C. Morris and P. Joly, Statistical Software
  Components S418701, Boston College; the 95% and 99% columns of its Case 0,
  Case 1/1* and Case 2/2* tables).
- Tables 1* and 2* (5% and 1%): additionally identical, entry by entry, to
  the arrays in the R package urca (function ``ca.jo``, ecdet "const" and
  "trend").
- Table 1, dimensions 1-3 (5% and 1%, trace and maximum): additionally
  identical to the output printed in the Stata manual ([TS] vecrank); the 5%
  trace value of dimension 4 (47.21) is printed in [TS] vec intro.

A second verification pass downloaded ``johans.ado`` and urca's ``ca-jo.R``
again and compared all 220 shipped numbers (and the 88 of Tables 1* and 2*)
programmatically: no difference.

``SINGLE_SOURCE`` lists the entries that rest on ONE transcription only
(``johans.ado``): Table 0 for K - r >= 7 and Table 2 for K - r >= 6. They are
shipped because that transcription agrees with the other sources wherever a
comparison is possible, and ``vecrank`` flags them in its result.

Known difference from Stata (not verified against a Stata run): output posted
by Stata users shows 9.42 as the 5% critical value of ``trend(rconstant)`` for
K - r = 1, where Osterwald-Lenum's Table 1* has 9.24 in both independent
transcriptions (and in EViews 4 output). This module ships the paper's 9.24; a
statistic between 9.24 and 9.42 would be judged differently by Stata.

Plausibility checks (they catch a value from the wrong table or dimension,
not a wrong second decimal): every value lies within 4% of the
MacKinnon-Haug-Michelis (1999) response-surface value for the same case where
one exists (7% for Table 0 with one common trend: 3.84 against 4.13), and
the 5% trace values lie within 1.5% of Johansen (1995, Tables 15.1-15.5).
The original tables were simulated with 6,000 replications of random walks
of length 400 and are therefore slightly below exact asymptotic quantiles.

No p-values are provided: they would need the response surfaces of MacKinnon,
Haug and Michelis (1999), which OpenEconometrics does not ship.
"""

from __future__ import annotations

SOURCE = "Osterwald-Lenum (1992), Oxford Bulletin of Economics and Statistics 54: 461-472"
P_VALUE_NOTE = ("p-values are not reported: the limiting distributions are non-standard and "
                "the MacKinnon-Haug-Michelis response surfaces are not included")

# {trend: {statistic: {level in percent: [value for K - r = 1, 2, ..., 11]}}}
_TABLES: dict[str, dict[str, dict[int, list[float]]]] = {
    "none": {       # Table 0
        "trace": {5: [3.84, 12.53, 24.31, 39.89, 59.46, 82.49, 109.99, 141.20, 175.77, 212.67,
                      255.27],
                  1: [6.51, 16.31, 29.75, 45.58, 66.52, 90.45, 119.80, 152.32, 187.31, 226.40,
                      269.81]},
        "max": {5: [3.84, 11.44, 17.89, 23.80, 30.04, 36.36, 41.51, 47.99, 53.69, 59.06, 65.30],
                1: [6.51, 15.69, 22.99, 28.82, 35.17, 41.00, 47.15, 53.90, 59.78, 65.21, 72.36]},
    },
    "rconstant": {  # Table 1*
        "trace": {5: [9.24, 19.96, 34.91, 53.12, 76.07, 102.14, 131.70, 165.58, 202.92, 244.15,
                      291.40],
                  1: [12.97, 24.60, 41.07, 60.16, 84.45, 111.01, 143.09, 177.20, 215.74, 257.68,
                      307.64]},
        "max": {5: [9.24, 15.67, 22.00, 28.14, 34.40, 40.30, 46.45, 52.00, 57.42, 63.57, 69.74],
                1: [12.97, 20.20, 26.81, 33.24, 39.79, 46.82, 51.91, 57.95, 63.71, 69.94, 76.63]},
    },
    "constant": {   # Table 1
        "trace": {5: [3.76, 15.41, 29.68, 47.21, 68.52, 94.15, 124.24, 156.00, 192.89, 233.13,
                      277.71],
                  1: [6.65, 20.04, 35.65, 54.46, 76.07, 103.18, 133.57, 168.36, 204.95, 247.18,
                      293.44]},
        "max": {5: [3.76, 14.07, 20.97, 27.07, 33.46, 39.37, 45.28, 51.42, 57.12, 62.81, 68.83],
                1: [6.65, 18.63, 25.52, 32.24, 38.77, 45.10, 51.57, 57.69, 62.80, 69.09, 75.95]},
    },
    "rtrend": {     # Table 2*
        "trace": {5: [12.25, 25.32, 42.44, 62.99, 87.31, 114.90, 146.76, 182.82, 222.21, 263.42,
                      310.81],
                  1: [16.26, 30.45, 48.45, 70.05, 96.58, 124.75, 158.49, 196.08, 234.41, 279.07,
                      327.45]},
        "max": {5: [12.25, 18.96, 25.54, 31.46, 37.52, 43.97, 49.42, 55.50, 61.29, 66.23, 72.72],
                1: [16.26, 23.65, 30.34, 36.65, 42.36, 49.51, 54.71, 62.46, 67.88, 73.73, 79.23]},
    },
    "trend": {      # Table 2
        "trace": {5: [3.74, 18.17, 34.55, 54.64, 77.74, 104.94, 136.61, 170.80, 208.97, 250.84,
                      295.99],
                  1: [6.40, 23.46, 40.49, 61.24, 85.78, 114.36, 146.99, 182.51, 222.46, 263.94,
                      312.58]},
        "max": {5: [3.74, 16.87, 23.78, 30.33, 36.41, 42.48, 48.45, 54.25, 60.29, 66.10, 71.68],
                1: [6.40, 21.47, 28.83, 35.68, 41.58, 48.17, 54.48, 60.81, 66.91, 72.96, 78.51]},
    },
}

MAX_DIMENSION = 11

# Smallest K - r whose entries were confirmed by one transcription only (see the docstring).
SINGLE_SOURCE: dict[str, int] = {"none": 7, "trend": 6}
SINGLE_SOURCE_NOTE = ("critical values for K - r >= {dimension} with trend='{trend}' were "
                      "confirmed against one published transcription of Osterwald-Lenum's "
                      "table only (johans.ado, SSC S418701); proof-read them against the paper "
                      "before relying on the second decimal")


def lookup(trend: str, dimension: int, statistic: str, level: int) -> float | None:
    """Critical value for K - r = ``dimension``, or ``None`` when it is not tabulated."""
    values = _TABLES[trend][statistic][level]
    return values[dimension - 1] if 1 <= dimension <= len(values) else None


def available(trend: str) -> int:
    """Largest K - r with tabulated critical values for this trend specification."""
    return len(_TABLES[trend]["trace"][5])


def single_source(trend: str, dimension: int) -> bool:
    """Whether the entries for K - r = ``dimension`` rest on one transcription only."""
    return trend in SINGLE_SOURCE and SINGLE_SOURCE[trend] <= dimension <= available(trend)


def provenance_note(trend: str, largest_dimension: int) -> str | None:
    """The flag ``vecrank`` attaches when a reported critical value is single-sourced."""
    if not single_source(trend, min(largest_dimension, available(trend))):
        return None
    return SINGLE_SOURCE_NOTE.format(dimension=SINGLE_SOURCE[trend], trend=trend)
