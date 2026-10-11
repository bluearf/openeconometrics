"""Published critical values of the break/regime cointegration tests.

Every table is transcribed from the source named next to it and duplicated,
for audit, in ``tests/fixtures/cointegration_breaks/critical_values.json``
(the tests check that both agree and that the fixture hashes to
``TRANSCRIPTION_SHA256``). No value is interpolated or extrapolated: a cell
the sources do not give, or give inconsistently, is ``None`` and the test
reports no critical value for it.

Gregory and Hansen (1996a), "Residual-based tests for cointegration in models
with regime shifts", Journal of Econometrics 70, 99-126, Table 1 ("Approximate
asymptotic critical values"; response-surface intercepts from 10,000
replications at n = 50..300), read from the authors' scan
https://users.ssc.wisc.edu/~bhansen/papers/joe_96.pdf (page 109). Models C,
C/T, C/S; m = 1..4 regressors; levels 0.01, 0.025, 0.05, 0.10, 0.975; ADF*
and Z_t* share one row.

Gregory and Hansen (1996b), "Tests for cointegration in models with regime
and trend shifts", Oxford Bulletin of Economics and Statistics 58, 555-560,
Table 1 ("Approximate asymptotic critical values for regime and trend
shift"), from https://users.ssc.wisc.edu/~bhansen/papers/OxBulletin_96.pdf
(page 559).

Hatemi-J (2008), "Tests for cointegration with two unknown regime shifts with
an application to financial market integration", Empirical Economics 35,
497-505, Table 1 (regime-shift model with two breaks, k = 1..4 regressors,
1/5/10%). The article itself could not be retrieved; the values are the ones
on which three independent transcriptions agree: tspdlib
(github.com/aptech/tspdlib, commit a3660eaa, src/coint_hatemiJ.src), the CRAN
package hatemicoint 1.1.0 (R/utils.R) and the SSC module hatemij
(2026-06-25, hatemij.ado). They disagree on the 1% Z_alpha* value for k = 1
(-90.704 in tspdlib, -90.794 in the other two), which is therefore withheld.
The anomalous 10% Z_alpha* value for k = 1 (-52.232) is printed identically in
all three and is kept as published.

Maki (2012), "Tests for cointegration allowing for an unknown number of
breaks", Economic Modelling 29, 2011-2015, Table 1 (models 0-3, k = 1..4
regressors, m = 1..5 breaks, 1/5/10%), from the author's GAUSS code
("received from Daiki Maki by email October 6th, 2019", tspdlib commit
a3660eaa, src/coint_maki.src, procedure cv_coint_maki); identical to CRAN
makicoint 2.0.0 (R/engine.R), whose NEWS states that the values match the
published Table 1. The article itself could not be retrieved.

Bayer and Hanck (2013), "Combining non-cointegration tests", Journal of Time
Series Analysis 34, 83-95: critical values of the Fisher combinations EG-J and
EG-J-Ba-Bo at 1/5/10% by number of regressors K = 1..11 and deterministic case
(none / constant / trend), from the authors' Stata implementation
``bayerhanck.ado`` version 0.9 (5 June 2009; SSC, distribution date
2024-10-06, http://fmwww.bc.edu/RePEc/bocode/b/bayerhanck.ado). The component
null distributions are in ``bayer_hanck_null.py``.
"""

from __future__ import annotations

LEVELS_GH = ("1%", "2.5%", "5%", "10%", "97.5%")
LEVELS = ("1%", "5%", "10%")

# Gregory-Hansen (1996a) Table 1: {model: {m: (ADF*/Zt* row, Z_alpha* row)}}.
GREGORY_HANSEN = {
    "level": {
        1: ((-5.13, -4.83, -4.61, -4.34, -2.25), (-50.07, -45.01, -40.48, -36.19, -10.63)),
        2: ((-5.44, -5.16, -4.92, -4.69, -2.61), (-57.01, -51.41, -46.98, -42.49, -14.27)),
        3: ((-5.77, -5.50, -5.28, -5.02, -2.96), (-63.64, -57.96, -53.58, -48.65, -18.20)),
        4: ((-6.05, -5.80, -5.56, -5.31, -3.26), (-70.18, -64.41, -59.40, -54.38, -22.04)),
    },
    "trend": {
        1: ((-5.45, -5.21, -4.99, -4.72, -2.72), (-57.28, -52.09, -47.96, -43.22, -15.90)),
        2: ((-5.80, -5.51, -5.29, -5.03, -3.01), (-64.77, -58.57, -53.92, -48.94, -19.19)),
        3: ((-6.05, -5.79, -5.57, -5.33, -3.33), (-70.27, -64.26, -59.76, -54.94, -22.72)),
        4: ((-6.36, -6.07, -5.83, -5.59, -3.59), (-76.95, -70.56, -65.44, -60.12, -26.46)),
    },
    "regime": {
        1: ((-5.47, -5.19, -4.95, -4.68, -2.55), (-57.17, -51.32, -47.04, -41.85, -13.15)),
        2: ((-5.97, -5.73, -5.50, -5.23, -3.12), (-68.21, -63.28, -58.33, -52.85, -19.72)),
        3: ((-6.51, -6.23, -6.00, -5.75, -3.65), (-80.15, -73.91, -68.94, -63.42, -26.64)),
        4: ((-6.92, -6.64, -6.41, -6.17, -4.12), (-90.35, -84.00, -78.52, -72.56, -33.69)),
    },
    # Gregory-Hansen (1996b) Table 1.
    "regime_trend": {
        1: ((-6.02, -5.72, -5.50, -5.24, -3.30), (-69.37, -63.23, -58.58, -53.31, -21.99)),
        2: ((-6.45, -6.17, -5.96, -5.72, -3.76), (-79.65, -73.26, -68.43, -63.10, -28.13)),
        3: ((-6.89, -6.65, -6.32, -6.16, -4.17), (-90.84, -84.33, -78.87, -72.75, -34.26)),
        4: ((-7.31, -7.06, -6.84, -6.58, -4.57), (-100.69, -94.00, -88.47, -82.30, -40.99)),
    },
}

# Hatemi-J (2008) Table 1, regime-shift model with two breaks: {k: (ADF*/Zt*, Z_alpha*)}.
HATEMI_J = {
    1: ((-6.503, -6.015, -5.653), (None, -76.003, -52.232)),
    2: ((-6.928, -6.458, -6.224), (-99.458, -83.644, -76.806)),
    3: ((-7.833, -7.352, -7.118), (-118.577, -104.860, -97.749)),
    4: ((-8.353, -7.903, -7.705), (-140.135, -123.870, -116.169)),
}

# Maki (2012) Table 1: {model: {k: rows for m = 1..5 breaks (1%, 5%, 10%)}}.
MAKI = {
    "level": {
        1: ((-5.709, -4.602, -4.354), (-5.416, -4.892, -4.610), (-5.563, -5.083, -4.784),
            (-5.776, -5.230, -4.982), (-5.959, -5.426, -5.131)),
        2: ((-5.541, -5.004, -4.733), (-5.717, -5.211, -4.957), (-5.943, -5.392, -5.125),
            (-6.075, -5.550, -5.297), (-6.296, -5.760, -5.491)),
        3: ((-5.820, -5.341, -5.101), (-5.984, -5.517, -5.272), (-6.229, -5.704, -5.427),
            (-6.406, -5.871, -5.603), (-6.555, -6.038, -5.773)),
        4: ((-6.139, -5.650, -5.386), (-6.303, -5.839, -5.575), (-6.501, -5.992, -5.714),
            (-6.640, -6.132, -5.892), (-6.856, -6.306, -6.039)),
    },
    "trend": {
        1: ((-5.524, -5.038, -4.784), (-5.708, -5.196, -4.938), (-5.833, -5.373, -5.106),
            (-6.059, -5.508, -5.245), (-6.193, -5.699, -5.449)),
        2: ((-5.840, -5.359, -5.117), (-6.011, -5.518, -5.247), (-6.169, -5.691, -5.408),
            (-6.329, -5.831, -5.558), (-6.530, -5.993, -5.722)),
        3: ((-6.144, -5.645, -5.398), (-6.271, -5.796, -5.538), (-6.472, -5.957, -5.682),
            (-6.575, -6.086, -5.820), (-6.784, -6.250, -5.976)),
        4: ((-6.361, -5.913, -5.686), (-6.556, -6.055, -5.805), (-6.741, -6.214, -5.974),
            (-6.845, -6.373, -6.096), (-7.053, -6.494, -6.220)),
    },
    "regime": {
        1: ((-5.457, -4.895, -4.626), (-5.863, -5.363, -5.070), (-6.251, -5.703, -5.402),
            (-6.596, -6.011, -5.723), (-6.915, -6.357, -6.057)),
        2: ((-6.020, -5.558, -5.287), (-6.628, -6.093, -5.833), (-7.031, -6.516, -6.210),
            (-7.470, -6.872, -6.563), (-7.839, -7.288, -6.976)),
        3: ((-6.565, -6.035, -5.773), (-7.232, -6.702, -6.411), (-7.767, -7.155, -6.868),
            (-8.236, -7.625, -7.329), (-8.673, -8.110, -7.796)),
        4: ((-7.021, -6.520, -6.242), (-7.756, -7.244, -6.964), (-8.336, -7.803, -7.481),
            (-8.895, -8.292, -8.004), (-9.441, -8.869, -8.541)),
    },
    "regime_trend": {
        1: ((-6.048, -5.541, -5.281), (-6.620, -6.100, -5.845), (-7.082, -6.524, -6.267),
            (-7.553, -7.009, -6.712), (-8.004, -7.414, -7.110)),
        2: ((-6.523, -6.055, -5.795), (-7.153, -6.657, -6.397), (-7.673, -7.145, -6.873),
            (-8.217, -7.636, -7.341), (-8.713, -8.129, -7.811)),
        3: ((-6.964, -6.464, -6.220), (-7.737, -7.201, -6.926), (-8.331, -7.743, -7.449),
            (-8.851, -8.269, -7.960), (-9.428, -8.800, -8.508)),
        4: ((-7.400, -6.911, -6.649), (-8.167, -7.638, -7.381), (-8.865, -8.254, -7.977),
            (-9.433, -8.871, -8.574), (-10.08, -9.482, -9.151)),
    },
}

# Bayer-Hanck Fisher critical values: {level: (EG-J rows, EG-J-Ba-Bo rows)}, each row
# K = 1..11 regressors with columns (none, constant, trend).
BAYER_HANCK = {
    "1%": (
        ((16.948, 17.304, 17.289), (16.651, 16.679, 16.720), (16.236, 16.259, 16.263),
         (15.871, 15.845, 15.973), (15.626, 15.701, 15.666), (15.412, 15.348, 15.467),
         (15.312, 15.313, 15.184), (15.183, 15.000, 15.016), (14.960, 15.007, 15.069),
         (14.893, 14.853, 14.788), (14.690, 14.826, 14.745)),
        ((32.713, 33.969, 34.334), (31.793, 32.077, 32.601), (30.651, 31.169, 31.742),
         (30.088, 30.774, 30.836), (29.800, 29.850, 30.113), (29.222, 29.544, 29.962),
         (28.974, 29.037, 29.440), (28.780, 28.999, 29.084), (28.326, 28.840, 28.875),
         (28.208, 28.575, 28.577), (27.945, 28.055, 28.518)),
    ),
    "5%": (
        ((11.071, 11.229, 11.269), (10.838, 10.895, 10.858), (10.640, 10.637, 10.711),
         (10.516, 10.576, 10.532), (10.406, 10.419, 10.448), (10.312, 10.352, 10.311),
         (10.218, 10.295, 10.222), (10.185, 10.181, 10.189), (10.162, 10.154, 10.164),
         (10.079, 10.109, 10.070), (10.057, 10.059, 10.134)),
        ((21.352, 21.931, 22.215), (20.776, 21.106, 21.342), (20.237, 20.486, 20.788),
         (19.951, 20.143, 20.440), (19.747, 19.888, 20.170), (19.564, 19.761, 19.934),
         (19.471, 19.688, 19.722), (19.471, 19.447, 19.678), (19.365, 19.492, 19.582),
         (19.268, 19.365, 19.398), (19.151, 19.345, 19.404)),
    ),
    "10%": (
        ((8.612, 8.678, 8.686), (8.457, 8.479, 8.451), (8.350, 8.363, 8.352),
         (8.290, 8.301, 8.272), (8.221, 8.242, 8.276), (8.165, 8.200, 8.199),
         (8.125, 8.169, 8.146), (8.106, 8.134, 8.146), (8.067, 8.108, 8.096),
         (8.081, 8.067, 8.095), (8.084, 8.053, 8.084)),
        ((16.593, 16.964, 17.187), (16.171, 16.444, 16.507), (15.920, 16.097, 16.239),
         (15.776, 15.938, 16.086), (15.681, 15.804, 15.989), (15.644, 15.746, 15.872),
         (15.611, 15.731, 15.706), (15.561, 15.591, 15.705), (15.507, 15.528, 15.647),
         (15.422, 15.476, 15.565), (15.406, 15.476, 15.564)),
    ),
}
BAYER_HANCK_TRENDS = ("none", "constant", "trend")

# sha256 of tests/fixtures/cointegration_breaks/critical_values.json.
TRANSCRIPTION_SHA256 = "07fe58a57c42ccb4374907ccdda91a00d15f34025593661da0a0ed6cdd95f71e"


def gregory_hansen_critical(model: str, regressors: int) -> dict[str, dict[str, float]] | None:
    """{"ADF*": {...}, "Zt*": {...}, "Za*": {...}} or None beyond m = 4."""
    rows = GREGORY_HANSEN[model].get(regressors)
    if rows is None:
        return None
    tau, alpha = (dict(zip(LEVELS_GH, row, strict=True)) for row in rows)
    return {"ADF*": tau, "Zt*": dict(tau), "Za*": alpha}


def hatemi_j_critical(regressors: int) -> dict[str, dict[str, float | None]] | None:
    rows = HATEMI_J.get(regressors)
    if rows is None:
        return None
    tau, alpha = (dict(zip(LEVELS, row, strict=True)) for row in rows)
    return {"ADF*": tau, "Zt*": dict(tau), "Za*": alpha}


def maki_critical(model: str, regressors: int, breaks: int) -> dict[str, float] | None:
    rows = MAKI[model].get(regressors)
    if rows is None or not 1 <= breaks <= len(rows):
        return None
    return dict(zip(LEVELS, rows[breaks - 1], strict=True))


def bayer_hanck_critical(trend: str, regressors: int) -> dict[str, dict[str, float]] | None:
    if not 1 <= regressors <= 11:
        return None
    column = BAYER_HANCK_TRENDS.index(trend)
    return {name: {level: BAYER_HANCK[level][which][regressors - 1][column] for level in LEVELS}
            for which, name in enumerate(("EG-J", "EG-J-Ba-Bo"))}
