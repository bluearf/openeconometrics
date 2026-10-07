"""Published tables behind the unit-root, cointegration and break tests.

Every number in this module is typed in from the cited source; nothing is
simulated or fitted by OpenEconometrics. Each table was compared entry by entry with a
second transcription (statsmodels for MacKinnon, KPSS and the CUSUM-of-squares
coefficients; the R packages fUnitRoots and tseries for Fuller, urca for
Elliott-Rothenberg-Stock and Zivot-Andrews, plm for Levin-Lin-Chu and
Im-Pesaran-Shin) and, where Stata's manuals print values, with those. The one
exception is the Andrews sup-Wald table, for which no second transcription
was available: it was checked by simulating the limiting process only.
``docs/econometrics/unitroot.md`` lists the tables with these checks, and the
tests repeat them.

Sources
-------
MacKinnon, J. G. (1994). Approximate asymptotic distribution functions for
    unit-root and cointegration tests. Journal of Business & Economic
    Statistics 12, 167-176.  Tables 3 and 4 (tau statistics).
MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's
    Economics Department Working Paper 1227.  Tables 1-4.
Fuller, W. A. (1976, 1996). Introduction to Statistical Time Series. Wiley.
    Table 8.5.1 (1976) = Table 10.A.1 (1996): n(rho-hat - 1); also Hamilton
    (1994), Table B.5. Table 8.5.2 (1976): the tau statistics; also Hamilton
    (1994), Table B.6.
Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996). Efficient tests for an
    autoregressive unit root. Econometrica 64, 813-836.  Table I, panel C.
Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. and Shin, Y. (1992). Testing
    the null hypothesis of stationarity against the alternative of a unit root.
    Journal of Econometrics 54, 159-178.  Table 1.
Zivot, E. and Andrews, D. W. K. (1992). Further evidence on the great crash,
    the oil-price shock, and the unit-root hypothesis. Journal of Business &
    Economic Statistics 10, 251-270.  Tables 2-4 (asymptotic critical values).
Levin, A., Lin, C.-F. and Chu, C.-S. J. (2002). Unit root tests in panel data:
    asymptotic and finite-sample properties. Journal of Econometrics 108,
    1-24.  Table 2.
Im, K. S., Pesaran, M. H. and Shin, Y. (2003). Testing for unit roots in
    heterogeneous panels. Journal of Econometrics 115, 53-74.  Table 3.
Andrews, D. W. K. (2003). Tests for parameter instability and structural
    change with unknown change point: a corrigendum. Econometrica 71, 395-397,
    as tabulated for 15% trimming by Stock and Watson, Introduction to
    Econometrics (QLR critical values, F form).
Brown, R. L., Durbin, J. and Evans, J. M. (1975). Techniques for testing the
    constancy of regression relationships over time. JRSS B 37, 149-192.
Edgerton, D. and Wells, C. (1994). Critical values for the CUSUMSQ statistic in
    medium and large sized samples. Oxford Bulletin of Economics and Statistics
    56, 355-365.
"""

from __future__ import annotations

INF = float("inf")

# Deterministic cases: "n" none, "c" constant, "ct" constant and trend,
# "ctt" constant, trend and squared trend.

# ---- MacKinnon (1994): approximate asymptotic distribution of tau -----------------
# Row N-1 holds the entry for N integrated series (N = 1 is the Dickey-Fuller
# test, N > 1 the Engle-Granger residual test). For tau <= TAU_STAR the p-value
# is Phi(b0 + b1 tau + b2 tau^2) with Table 3 coefficients; above it
# Phi(b0 + b1 tau + b2 tau^2 + b3 tau^3) with Table 4 coefficients. Below
# TAU_MIN the p-value is 0, above TAU_MAX it is 1.
TAU_STAR = {
    "n": (-1.04, -1.53, -2.68, -3.09, -3.07, -3.77),
    "c": (-1.61, -2.62, -3.13, -3.47, -3.78, -3.93),
    "ct": (-2.89, -3.19, -3.50, -3.65, -3.80, -4.36),
    "ctt": (-3.21, -3.51, -3.81, -3.83, -4.12, -4.63),
}
TAU_MIN = {
    "n": (-19.04, -19.62, -21.21, -23.25, -21.63, -25.74),
    "c": (-18.83, -18.86, -23.48, -28.07, -25.96, -23.27),
    "ct": (-16.18, -21.15, -25.37, -26.63, -26.53, -26.18),
    "ctt": (-17.17, -21.1, -24.33, -24.03, -24.33, -28.22),
}
TAU_MAX = {
    "n": (INF, 1.51, 0.86, 0.88, 1.05, 1.24),
    "c": (2.74, 0.92, 0.55, 0.61, 0.79, 1.0),
    "ct": (0.7, 0.63, 0.71, 0.93, 1.19, 1.42),
    "ctt": (0.54, 0.79, 1.08, 1.43, 3.49, 1.92),
}
# Table 3; the third coefficient is printed in units of 1e-2.
TAU_SMALL_P = {
    "n": ((0.6344, 1.2378, 3.2496), (1.9129, 1.3857, 3.5322), (2.7648, 1.4502, 3.4186),
          (3.4336, 1.4835, 3.19), (4.0999, 1.5533, 3.59), (4.5388, 1.5344, 2.9807)),
    "c": ((2.1659, 1.4412, 3.8269), (2.92, 1.5012, 3.9796), (3.4699, 1.4856, 3.164),
          (3.9673, 1.4777, 2.6315), (4.5509, 1.5338, 2.9545), (5.1399, 1.6036, 3.4445)),
    "ct": ((3.2512, 1.6047, 4.9588), (3.6646, 1.5419, 3.6448), (4.0983, 1.5173, 2.9898),
           (4.5844, 1.5338, 2.8796), (5.0722, 1.5634, 2.9472), (5.53, 1.5914, 3.0392)),
    "ctt": ((4.0003, 1.658, 4.8288), (4.3534, 1.6016, 3.7947), (4.7343, 1.5768, 3.2396),
            (5.214, 1.6077, 3.3449), (5.6481, 1.6274, 3.3455), (5.9296, 1.5929, 2.8223)),
}
TAU_SMALL_SCALE = (1.0, 1.0, 1e-2)
# Table 4; coefficients two and three are printed in units of 1e-1, the fourth in 1e-2.
TAU_LARGE_P = {
    "n": ((0.4797, 9.3557, -0.6999, 3.3066), (1.5578, 8.558, -2.083, -3.3549),
          (2.2268, 6.8093, -3.2362, -5.4448), (2.7654, 6.4502, -3.0811, -4.4946),
          (3.2684, 6.8051, -2.6778, -3.4972), (3.7268, 7.167, -2.3648, -2.8288)),
    "c": ((1.7339, 9.3202, -1.2745, -1.0368), (2.1945, 6.4695, -2.9198, -4.2377),
          (2.5893, 4.5168, -3.6529, -5.0074), (3.0387, 4.5452, -3.3666, -4.1921),
          (3.5049, 5.2098, -2.9158, -3.3468), (3.9489, 5.8933, -2.5359, -2.721)),
    "ct": ((2.5261, 6.1654, -3.7956, -6.0285), (2.85, 5.272, -3.6622, -5.1695),
           (3.221, 5.255, -3.2685, -4.1501), (3.652, 5.9758, -2.7483, -3.2081),
           (4.0712, 6.6428, -2.3464, -2.546), (4.4735, 7.1757, -2.0681, -2.1196)),
    "ctt": ((3.0778, 4.9529, -4.1477, -5.9359), (3.4713, 5.967, -3.2507, -4.2286),
            (3.8637, 6.7852, -2.6286, -3.1381), (4.2736, 7.6199, -2.1534, -2.4026),
            (4.6679, 8.2618, -1.822, -1.9147), (5.0009, 8.3735, -1.6994, -1.6928)),
}
TAU_LARGE_SCALE = (1.0, 1e-1, 1e-1, 1e-2)

# ---- MacKinnon (2010): response surfaces for critical values ------------------------
# cv(T) = b_inf + b1 / T + b2 / T^2 + b3 / T^3 at the 1%, 5% and 10% levels. Row N-1
# is for N integrated series. The no-constant case is tabulated for N = 1 only
# (Table 1); Tables 2-4 give constant, constant + trend and constant + trend +
# squared trend. Rows N = 1..6 are stored: the range for which MacKinnon (1994)
# also provides p-values.
TAU_CRIT_2010 = {
    "n": (
        ((-2.56574, -2.2358, -3.627, 0.0), (-1.94100, -0.2686, -3.365, 31.223),
         (-1.61682, 0.2656, -2.714, 25.364)),
    ),
    "c": (
        ((-3.43035, -6.5393, -16.786, -79.433), (-2.86154, -2.8903, -4.234, -40.040),
         (-2.56677, -1.5384, -2.809, 0.0)),
        ((-3.89644, -10.9519, -33.527, 0.0), (-3.33613, -6.1101, -6.823, 0.0),
         (-3.04445, -4.2412, -2.720, 0.0)),
        ((-4.29374, -14.4354, -33.195, 47.433), (-3.74066, -8.5632, -10.852, 27.982),
         (-3.45218, -6.2143, -3.718, 0.0)),
        ((-4.64332, -18.1031, -37.972, 0.0), (-4.09600, -11.2349, -11.175, 0.0),
         (-3.81020, -8.3931, -4.137, 0.0)),
        ((-4.95756, -21.8883, -45.142, 0.0), (-4.41519, -14.0405, -12.575, 0.0),
         (-4.13157, -10.7417, -3.784, 0.0)),
        ((-5.24568, -25.6688, -57.737, 88.639), (-4.70693, -16.9178, -17.492, 60.007),
         (-4.42501, -13.1875, -5.104, 27.877)),
    ),
    "ct": (
        ((-3.95877, -9.0531, -28.428, -134.155), (-3.41049, -4.3904, -9.036, -45.374),
         (-3.12705, -2.5856, -3.925, -22.380)),
        ((-4.32762, -15.4387, -35.679, 0.0), (-3.78057, -9.5106, -12.074, 0.0),
         (-3.49631, -7.0815, -7.538, 21.892)),
        ((-4.66305, -18.7688, -49.793, 104.244), (-4.11890, -11.8922, -19.031, 77.332),
         (-3.83511, -9.0723, -8.504, 35.403)),
        ((-4.96940, -22.4694, -52.599, 51.314), (-4.42871, -14.5876, -18.228, 39.647),
         (-4.14633, -11.2500, -9.873, 54.109)),
        ((-5.25276, -26.2183, -59.631, 50.646), (-4.71537, -17.3569, -22.660, 91.359),
         (-4.43422, -13.6078, -10.238, 76.781)),
        ((-5.51727, -29.9760, -75.222, 202.253), (-4.98228, -20.3050, -25.224, 132.03),
         (-4.70233, -16.1253, -9.836, 94.272)),
    ),
    "ctt": (
        ((-4.37113, -11.5882, -35.819, -334.047), (-3.83239, -5.9057, -12.490, -118.284),
         (-3.55326, -3.6596, -5.293, -63.559)),
        ((-4.69276, -20.2284, -64.919, 88.884), (-4.15387, -13.3114, -28.402, 72.741),
         (-3.87346, -10.4637, -17.408, 66.313)),
        ((-4.99071, -23.5873, -76.924, 184.782), (-4.45311, -15.7732, -32.316, 122.705),
         (-4.17280, -12.4909, -17.912, 83.285)),
        ((-5.26780, -27.2836, -78.971, 137.871), (-4.73244, -18.4833, -31.875, 111.817),
         (-4.45268, -14.7199, -17.969, 101.92)),
        ((-5.52826, -30.9051, -92.490, 248.096), (-4.99491, -21.2360, -37.685, 194.208),
         (-4.71587, -17.0820, -18.631, 136.672)),
        ((-5.77379, -34.7010, -105.937, 393.991), (-5.24217, -24.2177, -39.153, 232.528),
         (-4.96397, -19.6064, -18.858, 174.919)),
    ),
}
LEVELS = ("1%", "5%", "10%")

# ---- Fuller: n (rho-hat - 1), the normalized-bias statistic Z(rho) ------------------
# Rows are sample sizes n = 25, 50, 100, 250, 500, infinity; columns the 1%, 5% and
# 10% points. Stata's pperron interpolates this table linearly in n.
FULLER_RHO_N = (25.0, 50.0, 100.0, 250.0, 500.0, INF)
FULLER_RHO = {
    "n": ((-11.9, -7.3, -5.3), (-12.9, -7.7, -5.5), (-13.3, -7.9, -5.6),
          (-13.6, -8.0, -5.7), (-13.7, -8.0, -5.7), (-13.8, -8.1, -5.7)),
    "c": ((-17.2, -12.5, -10.2), (-18.9, -13.3, -10.7), (-19.8, -13.7, -11.0),
          (-20.3, -14.0, -11.2), (-20.5, -14.0, -11.2), (-20.7, -14.1, -11.3)),
    "ct": ((-22.5, -17.9, -15.6), (-25.7, -19.8, -16.8), (-27.4, -20.7, -17.5),
           (-28.4, -21.3, -18.0), (-28.9, -21.5, -18.1), (-29.5, -21.8, -18.3)),
}

# ---- Fuller: the Dickey-Fuller t statistic tau ---------------------------------------
# Same layout as FULLER_RHO (rows n = 25, 50, 100, 250, 500, infinity; columns 1%, 5%,
# 10%), as printed in Fuller (1976, Table 8.5.2) and Hamilton (1994, Table B.6). These
# are the values Stata's dfuller and pperron interpolate linearly in n ("interpolated
# Dickey-Fuller critical values"): they reproduce the critical values printed in the
# examples of [TS] dfuller (n = 140) and [TS] pperron (n = 143).
FULLER_TAU = {
    "n": ((-2.66, -1.95, -1.60), (-2.62, -1.95, -1.61), (-2.60, -1.95, -1.61),
          (-2.58, -1.95, -1.62), (-2.58, -1.95, -1.62), (-2.58, -1.95, -1.62)),
    "c": ((-3.75, -3.00, -2.63), (-3.58, -2.93, -2.60), (-3.51, -2.89, -2.58),
          (-3.46, -2.88, -2.57), (-3.44, -2.87, -2.57), (-3.43, -2.86, -2.57)),
    "ct": ((-4.38, -3.60, -3.24), (-4.15, -3.50, -3.18), (-4.04, -3.45, -3.15),
           (-3.99, -3.43, -3.13), (-3.98, -3.42, -3.13), (-3.96, -3.41, -3.12)),
}

# ---- Elliott, Rothenberg and Stock (1996), Table I: DF-GLS with a linear trend ------
# Rows T = 50, 100, 200, infinity; columns 1%, 5%, 10%.
ERS_T = (50.0, 100.0, 200.0, INF)
ERS_TREND = ((-3.77, -3.19, -2.89), (-3.58, -3.03, -2.74), (-3.46, -2.93, -2.64),
             (-3.48, -2.89, -2.57))

# ---- KPSS (1992), Table 1: upper-tail critical values --------------------------------
KPSS_LEVELS = (0.10, 0.05, 0.025, 0.01)
KPSS_CRIT = {"level": (0.347, 0.463, 0.574, 0.739), "trend": (0.119, 0.146, 0.176, 0.216)}

# ---- Zivot and Andrews (1992), Tables 2-4: asymptotic critical values ----------------
# Model A (break in the intercept), B (break in the trend), C (both): 1%, 5%, 10%.
ZIVOT_ANDREWS = {"intercept": (-5.34, -4.80, -4.58), "trend": (-4.93, -4.42, -4.11),
                 "both": (-5.57, -5.08, -4.82)}

# ---- Levin, Lin and Chu (2002), Table 2: mean and standard deviation adjustments ----
# Rows are the average number of observations per panel T~ (after lags); columns
# (mu*, sigma*) for the models without deterministics, with panel means, and with
# panel means and trends.
LLC_T = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 250.0, INF)
LLC_ADJUST = {
    "none": ((0.004, 1.049), (0.003, 1.035), (0.002, 1.027), (0.002, 1.021), (0.001, 1.017),
             (0.001, 1.014), (0.001, 1.011), (0.000, 1.008), (0.000, 1.007), (0.000, 1.006),
             (0.000, 1.005), (0.000, 1.001), (0.000, 1.000)),
    "constant": ((-0.554, 0.919), (-0.546, 0.889), (-0.541, 0.867), (-0.537, 0.850),
                 (-0.533, 0.837), (-0.531, 0.826), (-0.527, 0.810), (-0.524, 0.798),
                 (-0.521, 0.789), (-0.520, 0.782), (-0.518, 0.776), (-0.509, 0.742),
                 (-0.500, 0.707)),
    "trend": ((-0.703, 1.003), (-0.674, 0.949), (-0.653, 0.906), (-0.637, 0.871),
              (-0.624, 0.842), (-0.614, 0.818), (-0.598, 0.780), (-0.587, 0.751),
              (-0.578, 0.728), (-0.571, 0.710), (-0.566, 0.695), (-0.533, 0.603),
              (-0.500, 0.500)),
}

# ---- Im, Pesaran and Shin (2003), Table 3: E[t_T(p, 0)] and Var[t_T(p, 0)] ----------
# Rows are the ADF lag order p = 0..8, columns the number of observations in the
# ADF regression T = 10, 15, 20, 25, 30, 40, 50, 60, 70, 100. None: not tabulated.
IPS_T = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0)
_NA = None
IPS_MEAN = {
    "constant": (
        (-1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532),
        (-1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530),
        (-1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514),
        (-1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512),
        (-1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495),
        (_NA, _NA, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494),
        (_NA, _NA, _NA, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476),
        (_NA, _NA, _NA, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474),
        (_NA, _NA, _NA, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456),
    ),
    "trend": (
        (-2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177),
        (-2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179),
        (-1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158),
        (-1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158),
        (-1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135),
        (_NA, _NA, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135),
        (_NA, _NA, _NA, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113),
        (_NA, _NA, _NA, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112),
        (_NA, _NA, _NA, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088),
    ),
}
IPS_VARIANCE = {
    "constant": (
        (1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735),
        (1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745),
        (1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754),
        (1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761),
        (2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771),
        (_NA, _NA, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781),
        (_NA, _NA, _NA, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795),
        (_NA, _NA, _NA, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806),
        (_NA, _NA, _NA, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818),
    ),
    "trend": (
        (1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597),
        (1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605),
        (1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613),
        (2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625),
        (3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629),
        (_NA, _NA, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638),
        (_NA, _NA, _NA, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650),
        (_NA, _NA, _NA, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661),
        (_NA, _NA, _NA, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670),
    ),
}

# ---- Andrews (2003) / Stock and Watson: QLR (sup-F) with 15% trimming ----------------
# Row q-1 is for q restrictions: the 10%, 5% and 1% critical values of the sup-F
# statistic; the sup-Wald critical value is q times the entry. NOT confirmed against a
# second transcription: checked by simulation of the limiting process (10% and 5%
# points within 4.5%) and by DeLong's tail formula.
QLR_F_15 = (
    (7.12, 8.68, 12.16), (5.00, 5.86, 7.78), (4.09, 4.71, 6.02), (3.59, 4.09, 5.12),
    (3.26, 3.66, 4.53), (3.02, 3.37, 4.12), (2.84, 3.15, 3.82), (2.69, 2.98, 3.57),
    (2.58, 2.84, 3.38), (2.48, 2.71, 3.23), (2.40, 2.62, 3.09), (2.33, 2.54, 2.97),
    (2.27, 2.46, 2.87), (2.21, 2.40, 2.78), (2.16, 2.34, 2.71), (2.12, 2.29, 2.64),
    (2.08, 2.25, 2.58), (2.05, 2.20, 2.53), (2.01, 2.17, 2.48), (1.99, 2.13, 2.43),
)

# ---- Brown, Durbin and Evans (1975): CUSUM boundary parameter a ----------------------
CUSUM_A = {"1%": 1.143, "5%": 0.948, "10%": 0.850}

# ---- Edgerton and Wells (1994): CUSUM-of-squares critical value ----------------------
# c0 = a1 / sqrt(m) + a2 / m + a3 / m^1.5 with m = (T - k) / 2 - 1; keys are the
# two-sided significance levels.
CUSUMSQ_COEFFICIENTS = {
    "10%": (1.2238734, -0.6700069, -0.7351697),
    "5%": (1.3581015, -0.6701218, -0.8858694),
    "1%": (1.6276236, -0.6703724, -1.2365861),
}
