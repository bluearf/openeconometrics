# Native seasonal adjustment

OpenEconometrics decomposes seasonal series with native CPU float64 Torch code.
No X-13ARIMA-SEATS, TRAMO-SEATS or other external binary runs at any time; the
routines below are independent implementations of published algorithms, checked
against the development references named in each section. This is method
evidence, not vendor parity: no Census Bureau, Stata, EViews or JDemetra+ output
was reproduced, and `provenance`-style parity flags are not claimed.

All routines take one complete, regularly spaced numeric series:

- rows are sorted by `time` when given; integer periods must be consecutive and
  datetimes must have an inferable regular calendar frequency (`time_gaps`
  otherwise); missing values are refused (`missing_values`), never dropped;
- a monthly or quarterly datetime calendar fixes the seasonal period (12 or 4); a
  declared `period` that contradicts it is refused (`period_mismatch`), annual
  calendars are refused (`not_seasonal`); other calendars and integer/row-order
  time need an explicit `period` (`missing_period`), 2..366;
- at most 50,000 periods; a `Dataset` (streaming) source is refused
  (`streaming_unsupported`); there are no observation weights;
- results are `SeasonalResult` table sets whose `to_json()` returns a sealed saved
  state (input series, period labels and calendar, effective settings and
  components, SHA-256). `oe.seasonal_restore(state)` checks the digest, replays the
  deterministic decomposition from the stored input and settings and refuses
  (`invalid_state`) any record whose components do not reproduce.

The decompositions are descriptive: they report no sampling distribution,
standard errors or p-values.

## STL

```python
parts = oe.stl(df, "sales", time="month", seasonal=13, robust=True)
parts["components"]          # period, observed, trend, seasonal, remainder,
                             # seasonally_adjusted, robustness_weight
fc = oe.stl_forecast(parts, 24, method="ets", ets_model="AdN")
```

`oe.stl` implements Cleveland, Cleveland, McRae and Terpenning (1990), "STL: A
Seasonal-Trend Decomposition Procedure Based on Loess", *Journal of Official
Statistics* 6(1), 3-73, following the reference Fortran step by step: inner loop
of cycle-subseries loess (extended by one value at each end), the low-pass filter
(moving averages of length period, period and 3 followed by loess), seasonal =
cycle-subseries smooth minus low-pass, and trend loess of the deseasonalized
series; outer loop of bisquare robustness weights `B(|R| / (6 median |R|))`.
Loess uses tricube weights over the `window` nearest points (the bandwidth grows by
`(window - n) // 2` when the window exceeds the subseries), local degree 0 or 1,
and evaluation every `jump` points with linear interpolation, exactly as in the
reference code. Every loess call is a batched tensor gather; there is no loop over
observations.

Two default conventions are offered and recorded in `attrs["settings"]`:

| setting | `convention="r"` (default; R `stats::stl`) | `convention="statsmodels"` (statsmodels `STL`) |
| --- | --- | --- |
| seasonal window | required: odd integer or `"periodic"` | 7 |
| seasonal / trend / low-pass degree | 0 / 1 / trend degree | 1 / 1 / 1 |
| jumps | ceil(window / 10) | 1 |
| inner, outer (robust) | 1, 15 | 2, 15 |
| inner, outer (not robust) | 2, 0 | 5, 0 |
| trend window | odd ceil(1.5 period / (1 - 1.5 / seasonal)) | same |
| low-pass window | period rounded up to odd | same |

`seasonal="periodic"` uses window 10n+1 with degree 0 and replaces the seasonal
component by its mean at each cycle position (R's convention). Windows must be odd
integers >= 3: the reference code silently rounds even windows up, here they are
refused. As in both references, `robust` only chooses the default iteration
counts. `n > 2 * period` is required. `max_work` bounds the loess kernel
evaluations before any computation (`search_budget`).

The robustness weights use the exact median of the absolute remainders, as in
statsmodels (which corrected the netlib partial sort). R's `stats::stl` still uses
the netlib `psort`, which for some residual vectors returns a neighbouring order
statistic instead of the median; R fits that hit this defect differ.

Output attrs also report the strength of trend and seasonality,
`max(0, 1 - Var(R) / Var(T + R))` and `max(0, 1 - Var(R) / Var(S + R))` (Wang,
Smith and Hyndman 2006), and the reconstruction error of
`observed = trend + seasonal + remainder`. For a multiplicative decomposition,
decompose the logarithm of a positive series.

### STL forecasts

`oe.stl_forecast(result, steps, method=...)` follows `forecast::forecast.stl`
(Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, section on
forecasting with decomposition): the seasonally adjusted series is forecast by a
non-seasonal method and the seasonal naive continuation of the STL seasonal
component (the last cycle repeated) is added. Methods:

- `"naive"`: random walk, sigma^2 = mean squared first difference,
  se_h = sigma sqrt(h) (`forecast::rwf`);
- `"drift"`: random walk with drift, drift = mean first difference,
  se_h^2 = s^2 h + h^2 se(drift)^2 with s^2 on n - 2 degrees of freedom
  (`forecast::rwf(drift = TRUE)`);
- `"ets"`: the native additive innovations ETS (`oe.ets`) with a non-seasonal
  model `ANN`, `AAN` or `AdN`, fitted to the adjusted series by its own joint ML;
- `"arima"`: the native `oe.arima` with `arima_order` and `arima_constant`.

Intervals are those of the adjusted-series forecast: the seasonal continuation is
treated as known, and decomposition uncertainty (and, except for the drift term,
parameter uncertainty) is excluded. Future period labels continue the integer
time, datetime calendar or row numbers.

### STL evidence

`tests/test_econ_seasonal_adjustment.py` compares

- seven R 4.6.1 `stats::stl` decompositions (monthly s.window 7/13/periodic,
  robust and not, quarterly with degree-1 seasonal smoothing and custom windows and
  jumps, inner 3/outer 2, and a daily series whose seasonal window exceeds the
  subseries length) stored in `tests/fixtures/seasonal/r_stl.json` by
  `scripts/generate_seasonal_oracles.R`. R's effective windows, degrees, jumps and
  iteration counts are derived natively from the same explicit arguments; the
  components and weights agree within 1e-9 (measured about 2e-12). The one robust
  quarterly case affected by R's partial-sort defect differs by 0.064 and is
  reproduced to 2e-13 when a test-only port of R's `psort` replaces the exact
  median;
- statsmodels 0.14 `STL` (defaults, robust, custom windows, degrees and jumps)
  live, within 1e-9 (measured about 1e-12);
- `forecast::forecast.stl` with `rwf` naive and drift forecasts, 80% and 95%
  intervals, within 1e-9; ETS/ARIMA forecasts equal the native `oe.ets`/`oe.arima`
  forecasts of the adjusted series plus the seasonal naive component;
- interval coverage in a seeded Monte Carlo (120 series, random walk plus a fixed
  monthly pattern, horizons 1-6, `seasonal="periodic"`, naive method): with n = 480
  the mean coverage is 0.956 for 95% and 0.790 for 80% intervals (tested within
  [0.90, 0.99] and [0.72, 0.88]); with n = 144 it falls to 0.912 and 0.744, because the
  seasonal component is treated as known — the intervals understate the uncertainty of
  short samples;
- reconstruction, exact periodicity, calendar/period validation, row-order
  invariance, saved-state replay and tamper detection, and the failure contract.

## X-11 method (native)

```python
sa = oe.x11(df, "sales", time="month")                 # multiplicative, 3x3 then 3x5
sa["components"]        # seasonal (D10), seasonally_adjusted (D11), trend (D12),
                        # irregular (D13), extreme_weight (C17), extreme_factor (C20),
                        # modified_si_replacement (D9)
sa["tables"]            # every B, C and D table aligned to the sample (NaN outside its span)
sa.attrs["henderson_steps"]   # Henderson length and I/C ratio of B7, C7, D7 and D12
```

`oe.x11` is the classical X-11 *method* (Shiskin, Young and Musgrave 1967; Ladiray
and Quenneville 2001, *Seasonal Adjustment with the X-11 Method*, Springer LNS
158) implemented natively and labelled "X-11 method (native)". **It is not
X-13ARIMA-SEATS**: there is no regARIMA pre-adjustment, no forecast/backcast
extension of the series, no trading-day or holiday estimation inside the
iterations, no MSR seasonal-filter selection, no cross-validation, no stable
seasonal filter and no SEATS. Calendar and intervention effects can be estimated
separately with `oe.calendar_regressors`/`oe.intervention_regressors` and the
regression part of `oe.arima`; the adjusted series is not pre-corrected
automatically.

Domain: monthly (period 12) or quarterly (period 4) series; a datetime calendar must
be monthly or quarterly (`unsupported_calendar` otherwise). Multiplicative mode needs
strictly positive data (`nonpositive_values`). Every cycle position needs a full
symmetric seasonal window: 6 years with 3x3/3x3, 7 years with the default 3x3/3x5
and 8 years with an initial 3x5 filter (`insufficient_observations`; X-13 would
switch to stable filters for shorter series).

The iteration (operator `(-)` is division in multiplicative and subtraction in
additive mode):

- **B**: B2 = centred 2x12 (2x4) moving average; B3 = SI ratios; B4 replacement
  values for extreme SI ratios, screened on B3 (-) normalized 3x3 seasonal factors;
  B5 = normalized initial seasonal filter of the modified SI, extended to the ends by
  repeating the nearest value of the same cycle position; B6 = B1 (-) B5; B7 =
  Henderson trend of B6; B8 = SI; B9 replacements; B10 = normalized final seasonal
  filter; B11, B13 = irregular; B17 weights and B20 extreme-value factors.
- **C**: C1 = B1 (-) B20 and the same steps (no SI replacement), C17 and C20.
- **D**: D1 = B1 (-) C20; D5, D6, D7 as in C; D8 = B1 (-) D7; D9 = modified SI where it
  differs from D8; D10 final seasonal factors; D11 = B1 (-) D10; D12 = Henderson trend
  of D1 (-) D10; D13 = D11 (-) D12.

Filters:

- seasonal moving averages per cycle position: 3x3 (1,2,3,2,1)/9 with end weights
  (5,11,11)/27 and (3,7,10,7)/27; 3x5 (1,2,3,3,3,2,1)/15 with (9,17,17,17)/60,
  (4,11,15,15,15)/60 and (4,8,13,13,13,9)/60. Defaults: `initial_seasonal_filter="3x3"`
  (B4/B5, C5, D5) and `seasonal_filter="3x5"` (B9/B10, C10, D10), the classical X-11
  choice. Seasonal factors are normalized by their centred 2x12 (2x4) moving average,
  whose end values copy the nearest computed one;
- Henderson trend filters of length 9, 13 or 23 (monthly) and 5 or 7 (quarterly),
  fixed by `henderson=` or chosen automatically from the I/C ratio (mean absolute
  period-to-period change of the irregular divided by that of the trend-cycle after a
  H13/H5 pre-smoothing; quarterly ratios times 3): monthly 9 (I/C < 1), 13 (< 3.5),
  23; quarterly 5 (< 1 or 1..3.5), 7 (>= 3.5); in B7 a ratio >= 1 selects H13/H5.
  Henderson weights are the classical closed form; the end points use Musgrave
  asymmetric weights with the X-11 ratios R = 1 (H9), 3.5 (H13), 4.5 (H23), 0.001
  (quarterly H5) and 4.5 (quarterly H7), D = 4 / (pi R^2). Multiplicative trends
  replace nonpositive values by the mean of their nearest positive neighbours.

Extreme values: the irregular is screened with moving five-year root mean squared
deviations from its mean (0 or 1), computed by calendar years (the first and last
years use the first/last five complete years together with any incomplete year;
fewer than five complete years use one global sigma), recomputed once without the
zero-weight values. Weights are 1 below `sigma_limits[0]` (default 1.5), 0 above
`sigma_limits[1]` (2.5) and linear in between; a year whose sigma is below 1e-5 is
not screened. An SI value with weight w < 1 is replaced by
(w SI + four nearest full-weight values of the same cycle position) / (4 + w) (two on
each side where available, otherwise more from the other side; the cycle-position mean
if fewer than four exist). Extreme-value factors are I / (1 + w (I - 1)) or (1 - w) I.
The calendar year boundaries depend on the cycle position of the first observation:
taken from a datetime calendar, otherwise `first_season` (default 1).

The table logic follows the X-11 description of Ladiray and Quenneville as
implemented in JDemetra+ 3 (`jdplus-main`, package `jdplus.x13.base.core.x11`, EUPL),
which was read as the algorithm description; no JDemetra+ or X-13 code or output is
used at run time or in the tests.

### X-11 evidence

- `scripts/generate_seasonal_oracles.py` contains an independent plain-loop
  Python/NumPy transcription of the same tables (explicit neighbour searches, no
  tensors) and stores eight cases in `tests/fixtures/seasonal/numpy_x11.json`:
  monthly multiplicative with outliers (automatic H13), a series starting in April,
  additive with fixed H23 and 3x3 final filter, an initial 3x5 filter with sigma limits
  (1.8, 2.8) and automatic H23 in C/D, a low-noise series selecting H9, clustered
  extremes that need the cycle-position-mean fallback, and quarterly multiplicative
  (automatic H5) and additive (H7) series. B7, B13, B17, C7, C17, C20 and D10-D13,
  the Henderson length of every step and the final I/C ratio agree within 1e-9
  (measured below 4e-12).
- Analytical checks: the 13-term Henderson weights equal the published values
  (-0.01935, -0.02786, 0, 0.06549, 0.14736, 0.21434, 0.24006; e.g. the NI LabVIEW
  [TSA Moving Average](https://www.ni.com/docs/en-DE/bundle/labview-advanced-signal-processing-toolkit-api-ref/page/lvtimeseriestk/tsa_moving_average.html)
  documentation); all Henderson filters reproduce cubic polynomials; every Musgrave
  end filter equals the independently solved constrained least-squares solution of
  the linear-trend revision criterion; seasonal end weights sum to one; the 2x12
  average annihilates a period-12 pattern; a constant level with a fixed seasonal
  pattern is recovered exactly in both modes with no extremes.
- Saved-state replay and tamper detection, row-order invariance, calendar
  (first season from monthly/quarterly datetimes) and the failure contract.

No published Ladiray-Quenneville worked-example table and no X-13ARIMA-SEATS or
JDemetra+ execution were available as references; agreement with those programs is
not claimed.
