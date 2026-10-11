# Forecasting extensions: calendar regressors, interventions, intermittent demand, combinations

These routines extend the native time-series workflows ([tsworkflows.md](tsworkflows.md),
[seasonal adjustment](seasonal-adjustment.md)) with deterministic regressors for the
regression part of `oe.arima` and with forecasting methods that need no external
engine. Everything runs on CPU float64 Torch tensors over resident tables; no
Dataset (streaming) route, GPU path or vendor parity is claimed.

## Calendar regressors (regARIMA inputs)

```python
reg = oe.calendar_regressors(df["month"], horizon=12)          # sample + 12 future months
data = df.merge(reg[~reg["future"]], left_on="month", right_on="period")
td = [c for c in reg.columns if c.startswith("td_")] + ["leap_year"]
fit = oe.arima(data=data, y="sales", x=td, time="month",
               order=(0, 1, 1), seasonal=(0, 1, 1), period=12)
fc = oe.forecast(fit, 12, exog=reg[reg["future"]][td].reset_index(drop=True))
```

`oe.calendar_regressors(dates | start, periods, frequency, horizon=0, ...)` builds, for
a regular monthly or quarterly calendar (`unsupported_calendar` otherwise; gaps are
`time_gaps`), the regressors of the X-13ARIMA-SEATS `regression` spec definitions:

- `trading_day="six"`: `td_mon` .. `td_sat` = number of each weekday in the period
  minus the number of Sundays (X-13 `td`); `trading_day="one"`: `td_weekday` =
  weekdays - 5/2 weekend days (`td1coef`);
- `leap_year=True`: 0.75 in a period containing a leap-year February, -0.25 in one
  containing a non-leap February, 0 otherwise (`lpyear`);
- `length_of_period=True`: days in the period minus 365.25/12 (or /4) (`lom`/`loq`);
- `holidays={name: dates}`: `holiday_<name>` = number of listed dates in the period.
  No holiday calendar (Easter, national holidays) is built in; dates outside the
  calendar are reported in `attrs["holidays"]` and not used.

Month-end and quarter-end stamps (`ME`, `QE-DEC`) and fiscal quarters (`QS-FEB`) are
mapped to their calendar months. `horizon` appends future periods (`future=True`)
for `oe.forecast(..., exog=...)`. Weekday counts use closed-form integer day
arithmetic on tensors. The regressors are not estimated, mean-adjusted or tested
here: their coefficients, standard errors and the full covariance come from
`oe.arima` (exact Gaussian ML, OPG/observed-information/robust covariance), whose
forecasts add `x_future' b` and condition on the estimated coefficients. The X-11
method does not pre-adjust for these effects.

Evidence (`tests/test_econ_forecast_extensions.py`): every column equals a brute-force
day-by-day enumeration with Python's `datetime` for monthly, month-end (crossing the
non-leap year 2100), quarterly, quarter-end and fiscal-quarter calendars, including
future periods; holiday counts; the regARIMA path reproduces the statsmodels SARIMAX
exact log-likelihood at the native estimates (1e-6), reaches at least SARIMAX's own
optimum and gives the same conditional forecasts and standard errors with future
regressors (1e-6); failure contract.

## Intervention regressors (pulse, step, ramp)

```python
reg = oe.intervention_regressors(df["month"], [
    {"name": "strike", "type": "pulse", "at": "2019-05-01", "decay": 0.7},
    {"name": "tariff", "type": "step", "at": "2020-01-01"},
    {"name": "rollout", "type": "ramp", "at": "2021-01-01", "end": "2021-07-01"}], horizon=6)
```

`oe.intervention_regressors(periods, events, horizon=0)` takes the sample's period labels
(consecutive integers or a regular datetime calendar) or a row count, and one record per
event:

- `pulse`: 1 at `at` (X-13 additive outlier `ao`); with `decay` d: d^(t - at) from `at`
  on (temporary change `tc`);
- `step`: 1 from `at` on (level shift `ls`); with `decay` d: the step passed through the
  first-order transfer function 1/(1 - dL), (1 - d^(t - at + 1)) / (1 - d), long-run gain
  1/(1 - d);
- `ramp`: 0 up to `at`, (t - at)/(end - at) in between, 1 from `end` on (X-13 `rp` plus
  one, equivalent in a model with a constant or differencing); without `end` the open
  ramp max(0, t - at).

The dynamic (transfer) effect is offered only where it is bounded: `decay` must lie in
[0, 1) and ramps take none (`unbounded_transfer`). The decay is a declared constant, not
an estimated parameter, so the regressor stays linear in its coefficient omega, which
`oe.arima` estimates by exact ML with the full covariance; choosing d by comparing
likelihoods over a declared grid is a separate, user-run model comparison whose
selection uncertainty is not included. Events may lie in the requested future horizon.
Event names must be new identifiers; unknown fields, missing periods and repeated names
are refused.

Evidence: every regressor equals an explicit loop (datetime, integer and row-position
labels, future events); a level shift plus temporary change in `oe.arima` reproduces the
statsmodels SARIMAX exact log-likelihood at the native estimates (1e-6) and its
conditional forecasts with future regressors; failure contract.

## Intermittent demand: Croston, SBA, SBJ, TSB

```python
out = oe.croston(df, "units", time="week", steps=8, method="sba", alpha=0.1)
out["forecast"]            # flat point forecasts
out["fitted"]              # observed, size_level, interval_level / probability_level, fitted
oe.croston(df, "units", method="tsb", alpha="estimate", beta="estimate", loss="mae")
```

Methods (point forecasts only):

- `croston` (Croston 1972, *Operational Research Quarterly* 23): exponential smoothing
  of the nonzero demand sizes (`alpha`) and of the inter-demand intervals (`beta`),
  both updated only in demand periods; forecast = size / interval;
- `sba` (Syntetos and Boylan 2005, *International Journal of Forecasting* 21): Croston
  times (1 - beta / 2);
- `sbj` (Shale, Boylan and Johnston 2006): Croston times (1 - beta / (2 - beta));
- `tsb` (Teunter, Syntetos and Babai 2011, *European Journal of Operational Research*
  214): sizes smoothed in demand periods (`alpha`), the demand probability smoothed in
  every period towards the occurrence indicator (`beta`); forecast = probability x size.

All methods start at the first demand period f with size y_f and interval f (as R
`forecast::croston_model`; TSB probability 1/f). One-step fitted values exist for
t > f (NaN before). The h-step forecast is flat at the value after the last
observation and uses every demand, including one in the final period (R forecast 9.x
returns the previous ratio in that case; this difference is tested). `beta=None` ties
the second constant to `alpha` (R's single constant). The recursions are evaluated as
exponentially weighted FFT convolutions on tensors.

Estimation: `alpha="estimate"` and/or `beta="estimate"` minimize the declared in-sample
one-step `loss` — `"mse"` (mean squared error) or `"mae"` (mean absolute error) over
t > f — on a 0.02 grid followed by coordinatewise golden-section refinement in the
neighbouring cells (three sweeps, tolerance 1e-7), within [0.01, 1]. This is
deterministic but not a global-optimum guarantee; estimates at a bound are listed in
`attrs["at_bound"]` and the grid minimum is reported. The constants are treated as
tuning values: no standard errors are produced. No prediction intervals are reported,
because Croston-type methods do not correspond to a consistent stochastic model
(Shenstone and Hyndman 2005, *Journal of Forecasting* 24). attrs also give the average
inter-demand interval (ADI) and the squared coefficient of variation of the sizes.
Demand must be nonnegative with at least two nonzero values; missing values are refused.

Evidence: R 4.6.1 `forecast` 9.0.2 `croston_model` fixtures (Croston, SBA, SBJ; alpha
0.1-0.35; `tests/fixtures/seasonal/r_croston.json`) agree to 1e-12 in fitted values,
smoothed sizes and intervals and forecasts; explicit NumPy loops for all four methods
with separate constants agree to 1e-12; estimated constants reproduce their reported
loss and are at least as good as a brute-force grid (within the 1e-6 search tolerance).

## Forecast combination

```python
combo = oe.forecast_combination({"sarima": oe.forecast(fit_a, 12),
                                 "stl": oe.stl_forecast(fit_b, 12, method="drift")},
                                method="bates_granger", history=validation,
                                origins={"sarima": "2024-12", "stl": "2024-12"})
```

`oe.forecast_combination(forecasts, method=..., history=None, origins=None)` combines the
point forecasts of 2..64 models supplied as saved forecast tables with `period` and
`forecast` columns. Every table must list exactly the same periods in the same order
(`mismatched_forecasts` otherwise — e.g. step-numbered ETS forecasts and calendar-labelled
ARIMA forecasts are refused until relabelled). With `origins`, the declared forecast
origins must be identical (`mismatched_origins`). Weights (Bates and Granger 1969;
Timmermann 2006):

- `"equal"`: 1/K;
- `"inverse_mse"`: proportional to 1/MSE_k of each model's past out-of-sample errors;
- `"bates_granger"`: w = S^-1 1 / (1' S^-1 1), with S = E'E / T the uncentred
  second-moment matrix of the past errors (the minimum-MSE weights summing to one; they
  may be negative). Singular S (`singular_error_covariance`) and T <= K
  (`insufficient_history`) are refused.

`history` is a table with `actual` and one column per model holding that model's past
forecasts for the same targets; incomplete rows are refused, never dropped. The same
weights apply to every horizon. The output lists the combined forecast and every
component; attrs give the weights, past MSE per model and of the combination. Only point
forecasts are combined: the joint error distribution of saved forecasts is unknown, so no
combined interval is reported, and the estimation uncertainty of the weights is not
propagated.

Evidence: equal, inverse-MSE and Bates-Granger weights equal their NumPy closed forms
(1e-12), the Bates-Granger combination attains a historical MSE no larger than any
component's, native SARIMA and STL forecasts with the same calendar combine while
step-numbered forecasts are refused, and the failure contract is tested.

## Rolling-origin backtests

```python
bt = oe.forecast_backtest(df, "sales", time="month", steps=6, minimum=96, step=3, models={
    "stl_drift": {"model": "stl", "forecast": "drift", "seasonal": 13, "robust": True},
    "stl_ets": {"model": "stl", "forecast": "ets", "ets_model": "AdN", "seasonal": "periodic"},
    "tsb": {"model": "croston", "variant": "tsb", "alpha": "estimate", "beta": "estimate"}})
history = bt[bt["horizon"] == 1][["actual", "stl_drift", "stl_ets"]]   # for forecast_combination
```

`oe.forecast_backtest` refits every declared model at each origin o = minimum,
minimum + step, ... on the observations up to o only: the STL decomposition, the ETS or
ARIMA model of its adjusted series and any estimated Croston/TSB constants are all
re-estimated inside the origin, so no transformation or parameter uses data after it
(`tests` alter the observations after an origin and verify that its forecasts do not
change). Rows (origin, horizon, period, actual, one column per model) exist for targets
with an observed actual; attrs give per-horizon MSE/MAE and counts. A model that fails at
an origin (for example fewer than two demands in an early Croston window) is listed in
`attrs["failures"]` with its error code and its forecasts stay missing — no fallback model
or data is substituted. At most `max_origins` (default 200, up to 2000) origins and 16
models are admitted before fitting. Filtering one horizon gives the `history` argument of
`oe.forecast_combination`. ARIMA-only backtests with arbitrary ModelSpecs remain
available through `oe.rolling`/`oe.recursive`.
