# Instructor notes · Lab 19: Forecasting Inflation Without Looking Ahead

The [student handout](../labs/19-forecasting/index.md) and
[source](../labs/19-forecasting/lab.py) use an original synthetic AR(1)
inflation series. The full history has 180 observations, but estimation uses
only the first 160. All twenty predictions are dynamic forecasts from the
same origin. They are not rolling one-step forecasts.

## Teaching approach

Draw a vertical information boundary at quarter 160 before discussing the
model. Ask which values a forecaster could know at that point, including
transformations, model-selection decisions, and possible revised data. Then
contrast a fixed-origin prediction of quarter 180 with an updated one-step
prediction made at quarter 179.

Pay particular attention to parameterization. Native `oe.arima` reports the
long-run mean as `Intercept` for this undifferenced model. Students accustomed
to an OLS recursion constant can obtain a completely wrong path if they use
that number as the constant in $y_t=c+\phi y_{t-1}+e_t$.

Keep the benchmark conclusion modest. The AR model's loss improvement over
the training mean is extremely small in this window. Do not turn the known
AR generator into a claim that estimated AR forecasts must always beat the
mean in finite samples.

## Worked exercise answers

1. The information set contains quarters **1–160**, with origin value
   **2.5854734813%**. Using quarter 161's actual value while making a
   forecast dated 160, estimating the mean on all 180 quarters, selecting lag
   order from held-out loss, or using future revisions as historical inputs
   would alter that information boundary.

2. Estimated mean is **2.7280280477** and AR coefficient
   **0.7041082236**. The recursion constant is
   $(1-0.7041082236)\times2.7280280477$, approximately **0.80720**.
   Using 2.728 as the recursion constant would imply a much larger long-run
   mean and an incorrect forecast path.

3. The two-step point forecast is
   $\hat\mu+\hat\phi^2(\pi_{160}-\hat\mu)$ = **2.657354%**.
   Its variance is $\hat\sigma^2(1+\hat\phi^2)$; standard error is
   **0.513268**. For horizon $h$, use the sum through $h-1$. Since
   $|\hat\phi|<1$, the variance approaches the finite geometric sum
   $\hat\sigma^2/(1-\hat\phi^2)$. At twenty quarters, the point forecast
   is **2.7279001774%**, SE **0.5910122603**.

4. First interval **[1.805109, 3.450199]%** and last interval
   approximately **[1.569537, 3.886263]%** concern future outcome
   uncertainty. They include future innovations while treating fitted
   parameters as fixed. They exclude parameter, selection, structural-change,
   and broader model uncertainty. They are not confidence intervals for
   the mean parameter or simultaneous coverage bands for the whole path.

5. MSE is **0.2411218428** for AR(1), **0.2425533883** for the training
   mean, and **0.3188576320** for the last-value benchmark, all in squared
   percentage points. The mean and last-value inputs, **2.7479458490** and
   **2.5854734813**, come only from training data. Losses are based on one
   common origin with overlapping forecast innovations across horizons, so
   the twenty losses are not twenty independent model comparisons.

6. Ljung–Box Q(8) p is about **0.5469**, Jarque–Bera p **0.6678**, and
   ARCH-LM(1) p **0.01724**. The last is a warning about variance dynamics.
   The known generator has constant variance, so a chance diagnostic alarm
   is possible, especially when inspecting several tests. In actual data,
   investigate residual variance patterns, unusual observations, regime
   changes, and interval calibration rather than dismissing the warning.

7. A valid rolling design specifies origins in advance, fits or updates
   parameters using only data available at each origin, forecasts a declared
   horizon, and records the later actual outcome. Parameter selection and
   preprocessing must also remain within the origin's history. Comparisons
   should align the same horizons and origins and recognize dependence in
   losses. If tuning is performed, reserve a separate final evaluation period
   or use a nested time-ordered validation design.

**Prepared input:** [forecast_series.xlsx](../labs/19-forecasting/forecast_series.xlsx). Use the stored observations for the published analysis. The [original generator](generators/19-forecasting.py) supports new dataset editions; update results and answers when changing observations. Fit on quarters 1–160 and evaluate on 161–180. The holdout outcomes must not enter model fitting.

## Reproduction and verification evidence

The [full reference state](../labs/19-forecasting/reference.json) includes the
complete native training model, its saved final state, all twenty forecast
rows and forecast metadata, and the aligned benchmark evaluation table.
Explicit exports use `run_lab(output_dir="forecast-output")`.

The independent forecast calculation evaluates
$\hat\mu+\hat\phi^h(\pi_T-\hat\mu)$ and
$\hat\sigma\sqrt{\sum_{j=0}^{h-1}\hat\phi^{2j}}$ using the fitted
parameters. Point discrepancies are zero in the checked run; standard-error
differences are below $2\times10^{-16}$. A separate squared-error sum
matches the native-path evaluation MSE. Checks retain the exact 160-row
estimation sample, origin 160, test calendar 161–180, and ordered interval
bounds. None of these checks uses a held-out observation in the fit.

Fresh source execution and Ruff passed. Actual `ConsoleSession` execution
passed with **model, table, plot, table** outputs and no error. The unique
whole-file result is `forecast_lab_result`. The default source writes no
exports. Verification covers the current source runtime, not an installed
release, real-time macroeconomic dataset, or hosted forecasting service.
