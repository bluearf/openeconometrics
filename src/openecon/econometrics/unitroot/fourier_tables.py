"""Critical values of the Fourier unit-root and stationarity tests.

Every published number below is typed from an author-hosted copy of the paper
(Walter Enders's site, https://www.time-series.net/fourier_papers) and compared
entry by entry with a second transcription in the GAUSS library TSPDLIB
(https://github.com/aptech/tspdlib, src/fourier_adf.src, fourier_kpss.src,
fourier_lm.src). Each table was also re-derived by the seeded simulation
``scripts/generate_unitroot_fourier_oracles.py``; the committed quantiles are
``tests/fixtures/unitroot_fourier/simulated_quantiles.json`` and the tests
compare every entry within Monte Carlo tolerance. Entries that did not
replicate are not published here; the simulated values replace them and say so.

Fourier ADF (Enders, W. and Lee, J. 2012, "The flexible Fourier form and
    Dickey-Fuller type unit root tests", Economics Letters 117, 196-199; revised
    final version of 4 March 2012, https://www.time-series.net/assets/docs/
    Enders_Lee_DF_Fourier_tests_Final_version_March_5_2012.40131805.pdf):
    Table 1a (constant and trend, tau_DF_t) and Table 1b (constant, tau_DF_C)
    for k = 1..5 and T = 100, 200, 500, 2500; the bottom rows give the
    supremum F(k^) = max_k F(k) critical values. In both F rows the paper's
    header reads 1%, 5%, 10% but the entries are printed in the order 10%, 5%,
    1% (the text quotes 9.14 as the 5% value for T = 100); they are stored here
    as 1%, 5%, 10%.
Fourier KPSS (Becker, R., Enders, W. and Lee, J. 2006, "A stationarity test in
    the presence of an unknown number of smooth breaks", Journal of Time Series
    Analysis 27, 381-409): Table 1 for k = 1, 2, 3 and T = 100, 500, as
    reproduced from that paper in Table 6 of Jones, P. M. and Enders, W. (2014),
    "On the use of the flexible Fourier form in unit roots tests, endogenous
    breaks, and parameter instability" (author-hosted version of 17 June 2012,
    https://www.time-series.net/assets/docs/Flexible_Fourier_Form06172012.3985105.pdf);
    identical in TSPDLIB. The k = 4, 5 entries (only in TSPDLIB, no primary copy
    located) and the supremum-F linearity-pretest entries are OpenEconometrics
    simulations (200,000 replications per cell, seed 20261010): the published
    supremum-F values for T = 100 (4.929 / 4.972 at 5%) exceed the simulated
    quantiles of eq. (7) with SSR1/(T - q) by 11-18 Monte Carlo standard errors
    (they are consistent with a denominator of SSR1/T), so they are not used.
Fourier LM (Enders, W. and Lee, J. 2012, "A unit root test using a Fourier
    series to approximate smooth breaks", Oxford Bulletin of Economics and
    Statistics 74, 574-599; final version of 16 May 2011,
    https://www.time-series.net/assets/docs/Enders_Lee_Final_version_May_16_2011.39101351.pdf):
    Table 1, panel a (tau_LM, k = 1..5), panel b (F(k) for a given k) and panel
    c (F(k^) = max_k F(k)) for T = 100, 200, 500, 2500. Panels b and c print the
    10%, 5%, 1% entries left to right under a 1%, 5%, 10% header (the authors'
    RATS program, table1critvalues.PRG on the same site, prints fractiles 90,
    95, 99); they are stored as 1%, 5%, 10%.

Rows are ``(1%, 5%, 10%)``. Unit-root statistics reject below the critical
value; the KPSS-type statistic and every F statistic reject above it.
"""

from __future__ import annotations

SOURCES = {
    "fourier_adf": "Enders and Lee (2012), Economics Letters 117, 196-199, Tables 1a-1b "
                   "(author-hosted final version, time-series.net)",
    "fourier_kpss": "Becker, Enders and Lee (2006), JTSA 27, 381-409, Table 1 (k = 1-3; as "
                    "reproduced in Jones and Enders 2014, Table 6, author-hosted)",
    "fourier_kpss_simulated": "OpenEconometrics seeded simulation, scripts/"
                              "generate_unitroot_fourier_oracles.py (200,000 replications, "
                              "seed 20261010)",
    "fourier_lm": "Enders and Lee (2012), Oxford Bulletin of Economics and Statistics 74, "
                  "574-599, Table 1 (author-hosted final version, time-series.net)",
}

# ---- Fourier ADF: Enders and Lee (2012, Economics Letters), Tables 1a and 1b -----------
ADF_T = (100.0, 200.0, 500.0, 2500.0)
# ADF_TAU[case][k - 1][t-row] = (1%, 5%, 10%)
ADF_TAU = {
    "trend": (                                                   # Table 1a, tau_DF_t
        ((-4.95, -4.35, -4.05), (-4.87, -4.31, -4.02), (-4.81, -4.29, -4.01), (-4.80, -4.27, -4.00)),
        ((-4.69, -4.05, -3.71), (-4.62, -4.01, -3.69), (-4.57, -3.99, -3.67), (-4.58, -3.98, -3.67)),
        ((-4.45, -3.78, -3.44), (-4.38, -3.77, -3.43), (-4.38, -3.76, -3.43), (-4.38, -3.75, -3.43)),
        ((-4.29, -3.65, -3.29), (-4.27, -3.63, -3.31), (-4.25, -3.64, -3.31), (-4.24, -3.63, -3.30)),
        ((-4.20, -3.56, -3.22), (-4.18, -3.56, -3.24), (-4.18, -3.56, -3.25), (-4.16, -3.55, -3.24)),
    ),
    "constant": (                                                # Table 1b, tau_DF_C
        ((-4.42, -3.81, -3.49), (-4.37, -3.78, -3.47), (-4.35, -3.76, -3.46), (-4.31, -3.75, -3.45)),
        ((-3.97, -3.27, -2.91), (-3.93, -3.26, -2.92), (-3.91, -3.26, -2.91), (-3.89, -3.25, -2.90)),
        ((-3.77, -3.07, -2.71), (-3.74, -3.06, -2.72), (-3.70, -3.06, -2.72), (-3.69, -3.05, -2.71)),
        ((-3.64, -2.97, -2.64), (-3.62, -2.98, -2.65), (-3.62, -2.97, -2.66), (-3.61, -2.96, -2.64)),
        ((-3.58, -2.93, -2.60), (-3.55, -2.94, -2.62), (-3.56, -2.94, -2.62), (-3.53, -2.93, -2.61)),
    ),
}
# Supremum F(k^) over k = 1..5, per T row.
ADF_SUPF = {
    "trend": ((12.21, 9.14, 7.78), (11.70, 8.88, 7.62), (11.52, 8.76, 7.53), (11.35, 8.71, 7.50)),
    "constant": ((10.35, 7.58, 6.35), (10.02, 7.41, 6.25), (9.78, 7.29, 6.16), (9.72, 7.25, 6.11)),
}

# ---- Fourier KPSS: Becker, Enders and Lee (2006), Table 1 ------------------------------
KPSS_T = (100.0, 500.0)
# KPSS_TAU[case][k - 1][t-row] = (1%, 5%, 10%); k = 4, 5 are simulated (see docstring).
KPSS_TAU = {
    "constant": (
        ((0.2699, 0.1720, 0.1318), (0.2709, 0.1696, 0.1294)),
        ((0.6671, 0.4152, 0.3150), (0.6615, 0.4075, 0.3053)),
        ((0.7182, 0.4480, 0.3393), (0.7046, 0.4424, 0.3309)),
        ((0.7341, 0.4598, 0.3469), (0.7274, 0.4501, 0.3393)),        # simulated
        ((0.7415, 0.4648, 0.3510), (0.7314, 0.4552, 0.3437)),        # simulated
    ),
    "trend": (
        ((0.0716, 0.0546, 0.0471), (0.0720, 0.0539, 0.0463)),
        ((0.2022, 0.1321, 0.1034), (0.1968, 0.1278, 0.0995)),
        ((0.2103, 0.1423, 0.1141), (0.2091, 0.1404, 0.1123)),
        ((0.2167, 0.1470, 0.1186), (0.2134, 0.1447, 0.1161)),        # simulated
        ((0.2180, 0.1487, 0.1203), (0.2152, 0.1464, 0.1179)),        # simulated
    ),
}
KPSS_SIMULATED_FREQUENCIES = (4, 5)
# Simulated supremum F(k^) over k = 1..5 of eq. (7), iid errors (see docstring).
KPSS_SUPF = {
    "constant": ((6.582, 4.804, 4.032), (6.302, 4.653, 3.911)),
    "trend": ((6.633, 4.816, 4.022), (6.300, 4.614, 3.887)),
}
# Published Table 1b values, kept for the record and the provenance test only.
KPSS_SUPF_PUBLISHED = {
    "constant": ((6.730, 4.929, 4.133), (6.281, 4.651, 3.935)),
    "trend": ((6.873, 4.972, 4.162), (6.315, 4.669, 3.928)),
}

# ---- Fourier LM: Enders and Lee (2012, OBES), Table 1 ----------------------------------
LM_T = (100.0, 200.0, 500.0, 2500.0)
# LM_TAU[k - 1][t-row] = (1%, 5%, 10%)                               panel a
LM_TAU = (
    ((-4.69, -4.10, -3.82), (-4.61, -4.07, -3.79), (-4.57, -4.05, -3.78), (-4.56, -4.03, -3.77)),
    ((-4.25, -3.57, -3.23), (-4.18, -3.55, -3.23), (-4.13, -3.54, -3.22), (-4.15, -3.54, -3.22)),
    ((-3.98, -3.31, -2.96), (-3.94, -3.30, -2.98), (-3.94, -3.31, -2.98), (-3.94, -3.30, -2.98)),
    ((-3.85, -3.18, -2.86), (-3.80, -3.18, -2.88), (-3.81, -3.19, -2.88), (-3.80, -3.19, -2.88)),
    ((-3.75, -3.11, -2.81), (-3.73, -3.12, -2.83), (-3.75, -3.14, -2.83), (-3.74, -3.13, -2.83)),
)
# LM_F[k - 1][t-row] = (1%, 5%, 10%)                                 panel b, given k
LM_F = (
    ((11.52, 8.48, 7.08), (11.10, 8.28, 6.93), (10.81, 8.15, 6.85), (10.69, 8.07, 6.79)),
    ((7.72, 4.90, 3.71), (7.51, 4.86, 3.73), (7.45, 4.84, 3.74), (7.32, 4.83, 3.71)),
    ((5.99, 3.78, 2.87), (5.89, 3.80, 2.90), (5.97, 3.82, 2.93), (5.95, 3.83, 2.94)),
    ((5.36, 3.41, 2.61), (5.36, 3.44, 2.63), (5.36, 3.45, 2.65), (5.37, 3.46, 2.65)),
    ((5.16, 3.26, 2.47), (5.09, 3.25, 2.48), (5.09, 3.28, 2.53), (5.03, 3.29, 2.52)),
)
# panel c, F(k^) = max over k = 1..5
LM_SUPF = ((11.79, 8.80, 7.50), (11.32, 8.60, 7.34), (11.03, 8.45, 7.24), (10.90, 8.37, 7.18))
