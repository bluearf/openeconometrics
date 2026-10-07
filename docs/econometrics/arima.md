# ARIMA models, forecasts, series diagnostics and exponential smoothing

`oe.arima` fits ARIMA, seasonal ARIMA and ARMAX models (regression with ARMA
disturbances) by exact Gaussian maximum likelihood or by conditional sum of
squares. Around it the family provides dynamic forecasts (`oe.forecast`), the
correlogram and three series tests (`oe.corrgram`, `oe.wntestq`,
`oe.jarque_bera`, `oe.archlm`) and exponential smoothing (`oe.tssmooth`).

Everything is implemented in OpenEconometrics on float64 PyTorch tensors. No estimation
library runs at fit time and nothing loops over observations: the exact
likelihood, the same number a Kalman filter with the stationary initial state
produces, is evaluated in closed form from polynomial filters, so an
ARIMA(2,1,2) on one million observations fits in about two seconds (timings at
the end).

```python
import openecon as oe

fit = oe.arima(data=df, y="lsales", order=(0, 1, 1), seasonal=(0, 1, 1), period=12,
               time="month", constant=False)       # arima lsales, arima(0,1,1) sarima(0,1,1,12) noconstant
print(fit.summary())
oe.forecast(fit, steps=12)                         # tsappend, add(12) + predict, y dynamic(...)
```

| Stata / EViews / SPSS | OpenEconometrics |
| --- | --- |
| `arima y, arima(1,1,1)` | `oe.arima(data=df, y="y", order=(1, 1, 1))` |
| `arima y x1 x2, ar(1) ma(1)` | `oe.arima(data=df, y="y", x=["x1", "x2"], order=(1, 0, 1))` |
| `arima y, arima(0,1,1) sarima(0,1,1,12)` | `..., order=(0, 1, 1), seasonal=(0, 1, 1), period=12` |
| `arima ..., condition` | `..., method="css"` |
| `arima ..., noconstant` | `..., constant=False` |
| `arima ..., vce(oim)` / `vce(robust)` | `..., covariance="nonrobust"` / `covariance="robust"` |
| EViews `ls d(y) c ar(1) ma(1)`; SPSS `ARIMA` | `oe.arima(data=df, y="y", order=(1, 1, 1))` |
| `predict f, y dynamic(...)` + `predict v, mse`; EViews Forecast | `oe.forecast(fit, steps=h)` |
| `corrgram y, lags(20) yw`; EViews Correlogram; SPSS `ACF`/`PACF` | `oe.corrgram(df, "y", lags=20)` |
| `wntestq y, lags(12)` | `oe.wntestq(df, "y", lags=12)` |
| EViews Histogram and Stats (Jarque-Bera) | `oe.jarque_bera(df, "y")` |
| `estat archlm, lags(4)`; EViews ARCH test | `oe.archlm(df, "resid", lags=4)` |
| `tssmooth exponential / dexponential / hwinters / shwinters` | `oe.tssmooth(data=df, y="y", method=...)` |

## `oe.arima`: ARIMA, SARIMA and ARMAX

### The model

A regression whose disturbance follows a multiplicative seasonal ARIMA process
(the form of Stata's `arima`):

    y_t = x_t'b + mu_t
    phi(L) Phi(L^s) (1 - L)^d (1 - L^s)^D mu_t = theta(L) Theta(L^s) e_t,     e_t ~ N(0, sigma^2)

    phi(L)   = 1 - phi_1 L - ... - phi_p L^p          Phi(L^s)   = 1 - Phi_1 L^s - ... - Phi_P L^(Ps)
    theta(L) = 1 + theta_1 L + ... + theta_q L^q      Theta(L^s) = 1 + Theta_1 L^s + ... + Theta_Q L^(Qs)

The signs are Stata's: AR coefficients enter with a minus sign in the lag
polynomial, MA coefficients with a plus sign. As in Stata the outcome and
every regressor are differenced first,

    w_t = (1 - L)^d (1 - L^s)^D y_t = b_0 + [(1 - L)^d (1 - L^s)^D x_t]'b + u_t,
    phi(L) Phi(L^s) u_t = theta(L) Theta(L^s) e_t,

so the constant `b_0` (`Intercept`) is the mean of the differenced outcome net
of the regressors: the level mean when `d = D = 0`, the drift per period when
`d = 1`. The differenced sample has `N = T - d - D s` observations; the first
`d + D s` rows are starting values and are reported as excluded. The ARMA part
is an ARMA(p + P s, q + Q s) process with the expanded polynomials
`phi*(L) = phi(L) Phi(L^s)` and `theta*(L) = theta(L) Theta(L^s)`.

### Arguments

| Argument | Meaning |
| --- | --- |
| `data`, `y` | Table and outcome column. |
| `order` | `(p, d, q)`, required: three non-negative integers (tuple, list or integer array). `d <= 3`. |
| `x`, `categorical` | Optional regressors; categorical ones are treatment-coded (`group[b]`). |
| `seasonal`, `period` | `(P, D, Q)` and the season length `s >= 2`; both or neither. `D <= 2`. |
| `time` | Optional time column (integer periods or datetimes). Rows are sorted by it. Without it the row order is the time order. |
| `method` | `"ml"` (default, exact likelihood) or `"css"` (conditional). |
| `constant` | Include `b_0` (default `True`). |
| `covariance` | `"opg"` (default), `"nonrobust"` (observed information) or `"robust"`. |
| `ljung_lags` | Lags of the residual Ljung-Box test; default `min(floor(N/2) - 2, 40)`. |
| `max_iterations`, `tolerance` | BFGS iteration limit (200) and gradient tolerance (1e-8, on the standardized problem). |
| `missing` | `"raise"` (default) or `"drop"`. |
| `alpha` | Significance level of the confidence intervals. |

The same model through the generic entry point is `oe.fit(oe.ModelSpec(
estimator="arima", outcome="y", predictors=[...], time="t",
options={"order": [1, 1, 1], "seasonal": [0, 1, 1], "period": 12}), data=df)`;
`intercept=False` there is `constant=False`. A specification mistake raises
`AnalysisError("invalid_spec")` or `AnalysisError("invalid_order")` from
`oe.arima` (building a `ModelSpec` directly raises pydantic's
`ValidationError` with the same message).

### Time order, gaps and missing values

ARIMA needs one regularly spaced series without holes.

- With `time`, rows are sorted by it; integer periods must be consecutive
  (`time_gaps` otherwise) and distinct (`repeated_time_values`). Datetime
  columns are ranked and taken as consecutive periods, with a recorded warning,
  because a calendar gap cannot be told from the sampling frequency. Text or
  boolean time columns raise `invalid_time` (convert them with
  `pandas.to_datetime` or number the periods).
- Without `time`, the row order is the time order (recorded in
  `provenance["time_order"]`).
- `missing="raise"` refuses incomplete rows. `missing="drop"` removes them only
  when they sit at the start or the end of the series; an incomplete row inside
  the series raises `time_gaps`, because dropping it would silently shift every
  lag. Fill or interpolate such values yourself.
- Regressors that are collinear after differencing are omitted left to right
  and recorded in `warnings` and `provenance["omitted_terms"]`, as Stata does.
  A linear trend, for example, becomes a constant under `d = 1` and is omitted
  next to the drift.

### Estimator: exact maximum likelihood (`method="ml"`)

Stata's default. The likelihood is that of the stationary Gaussian ARMA process
`u_t` on the differenced data, i.e. the prediction-error decomposition of the
Kalman filter started from the unconditional state mean and covariance. In
Harvey's state-space form (state dimension `r = max(p*, q* + 1)`, transition
`T` with `phi*` in its first column and ones above the diagonal,
`R = (1, theta*_1, ..., theta*_(r-1))'`) OpenEconometrics does not run the filter. It
uses an algebraically identical closed form:

1. *Conditional residuals.* With presample values at zero,
   `v~ = phi*(L) u / theta*(L)` for all `N` observations at once (polynomial
   filters by cyclic reduction, no time loop).
2. *Presample-state correction.* Writing the state as `alpha_t = x_t + R e_t`,
   the true disturbances are `e = v~ - G x_1`, where `x_1 ~ N(0, sigma^2 S_x)`
   is the presample state, `S_x = P - R R'`, `P = T P T' + R R'` the stationary
   state covariance (doubling algorithm) and `G[t, j] = psi_(t-j)` the impulse
   response of `1/theta*(L)`. Integrating `x_1` out gives

       ll = -N/2 ln(2 pi sigma^2) - D/2 - S / (2 sigma^2)
       D  = ln det(I + S_x M),              M = G'G
       S  = v~'v~ - b'W b,                  b = G'v~,   W = (S_x^-1 + M)^-1.

   `G` has only as many non-negligible rows as the impulse response is long,
   so the cost is a few `O(N)` filter passes plus `r x r` algebra.

`sigma^2 = S/N` is concentrated out and the other parameters are found by BFGS
with the analytic gradient. The reported `/sigma` is `sqrt(S/N)`, the ML
estimate (no degrees-of-freedom correction, as in Stata).

The exact likelihood exists only for a stationary AR part; trial points outside
that region are rejected by the line search. The MA polynomial is kept in its
invertible representation (every non-invertible MA polynomial has an invertible
twin with the same likelihood).

### Estimator: conditional sum of squares (`method="css"`)

Stata's `condition` option: presample disturbances and presample `u` are set to
their expectation zero, and

    ll_c = -N/2 ln(2 pi sigma^2) - sum_(t=1..N) e_t^2 / (2 sigma^2),     e = phi*(L) u / theta*(L)

is maximized over all `N` observations; `sigma^2 = sum e_t^2 / N`. This is
EViews' and SPSS' "conditional least squares" criterion. It does not require a
stationary AR part (a warning is issued when the estimate is not stationary).

### Standardized estimation

The optimizer never sees the data in their original units. With a constant in
the model the outcome and the regressors are centered at their means; the
outcome is divided by its standard deviation `c_y` and regressor `j` by its
root mean square `c_j`. This is an exact reparameterization: the ARMA
parameters are unit free,

    b_j = b*_j c_y / c_j,     sigma = sigma* c_y,     b_0 = c_y b*_0 + ybar - sum_j b_j xbar_j,
    ll = ll* - N ln c_y,      V = A V* A'

with `A` the Jacobian of that linear map. Estimates, covariance and log
likelihood are reported in the original units
(`provenance["standardization"]`). The gradient tolerance and the numerical
Hessian therefore do not depend on the units or the level of the data: a
series measured in billions, a regressor in millionths or a level of 1e9 with
unit variation give the same ARMA estimates and exactly rescaled coefficients
(tested from 1e-8 to 1e8). Without a constant nothing is centered, because the
model through the origin is not invariant to a shift.

### Starting values and local maxima

Starting values are Hannan-Rissanen: OLS for `b`, a long Yule-Walker
autoregression of the OLS residuals to estimate the disturbances, then a
regression of the residuals on their own lags and the lagged disturbance
estimates; each polynomial is pulled inside the unit circle (inverse roots at
most 0.95). The likelihood of a mixed or over-fitted ARMA model can have
several local maxima. For samples up to 20,000 observations and models with
both AR and MA terms (or three or more ARMA terms) OpenEconometrics therefore screens
several starts (Hannan-Rissanen with long and with cautious first steps, OLS
with white-noise errors and, for exact ML, the conditional estimate) and runs
the full maximization from the best one; `provenance["optimizer"]["starts"]`
lists the log likelihood each start reached. Larger samples use the better of
the first two starts only. `metrics["iterations"]` counts the BFGS iterations
of the screening run that supplied the final starting point
(`provenance["optimizer"]["screening_iterations"]`) plus those of the final
run (`final_iterations`).

A global maximum is still not guaranteed (no package guarantees it). In a
verification sweep of 36 random ARMA(p, q) regressions (p, q <= 2, 30 to 250
observations, exact ML and CSS) against a six-start brute-force search,
OpenEconometrics reached the same or a higher likelihood in 34 cases. In the other two
(ARMA(2,2) on 30 and 60 observations) the brute force found a higher maximum
with an MA root exactly on the unit circle, where a nearly cyclical AR pair is
cancelled; OpenEconometrics reported the interior maximum. If `starts` shows clearly
different values, or the fit warns about the invertibility boundary, the model
is probably over-parameterized.

### Covariance and inference

Coefficient tests are z tests (Stata reports z for `arima`). The covariance is
that of the reported parameters `(b, phi, theta, Phi, Theta, sigma)`:

| `covariance` | Stata | Formula |
| --- | --- | --- |
| `"opg"` (default) | `vce(opg)`, Stata's default for `arima` | `(S'S)^-1`, `S` the `N x K` matrix of per-observation scores of the prediction-error decomposition |
| `"nonrobust"` | `vce(oim)` | `(-H)^-1`, `H` the Hessian of the log likelihood |
| `"robust"` | `vce(robust)` | `N/(N-1) (-H)^-1 S'S (-H)^-1` |

The gradient and the per-observation scores are analytic (verified against
numerical derivatives in the tests). `H` is the numerical derivative of the
analytic gradient (Ridders-extrapolated central differences; the `sigma` row
and column are analytic), recorded in `provenance["optimizer"]["hessian"]`.
The per-observation log likelihood is `ll_t = -1/2 ln(2 pi sigma^2 F_t) -
v_t^2 / (2 sigma^2 F_t)` with innovation `v_t` and variance ratio `F_t`; for
`method="css"`, `F_t = 1` and `v_t = e_t`.

`/sigma` follows Stata: its p-value is one-sided (half the two-sided normal
p-value) and its confidence interval is truncated at zero.

**MA unit roots.** When a series is over-differenced the MA estimate piles up
exactly on `theta = -1`. The likelihood is symmetric in `theta -> 1/theta`
(with `sigma -> sigma |theta|`), which makes the `theta` score of every
observation the multiple `-sigma/2` of its `sigma` score: `S'S` is singular
and the sandwich gives `theta` a zero variance, while the observed information
stays regular. In that one case `opg` and `robust` are replaced by the
observed information; the result carries a warning,
`inference["covariance"] == "nonrobust"` and
`inference["requested_covariance"]`. Standard errors at a unit root are not
reliable under any estimator; the warning says so and recommends less
differencing.

### What the result contains

**Coefficients.** The regression terms (equation named after the differenced
outcome in Stata's operator notation: `y`, `D.y`, `D2.y`, `DS12.y`), then
`ARMA:L1.ar`, ..., `ARMA:L1.ma`, ... (equation `ARMA`), the seasonal terms
`ARMA12:L1.ar`, `ARMA12:L1.ma` (equation `ARMA<s>`) and `/sigma`.

**`metrics`.** `log_likelihood`, `aic = -2 ll + 2K`, `bic = -2 ll + K ln N`
(`K` counts every parameter including `sigma`, `N` the observations after
differencing, as `estat ic`), `sigma`, `n_differenced` (= `N`), `iterations`.

**`tests`.**

- `model`: Wald chi2 test that every coefficient except the constant is zero
  (the "Wald chi2" of Stata's header). Absent when there is nothing to test.
- `ljung_box`: Ljung-Box Q of the residuals at `ljung_lags` lags with
  `df = lags` (Stata's `wntestq` on the residuals). `df_adjusted = lags -
  (p + q + P + Q)` and `p_value_adjusted` give EViews' convention of
  subtracting the ARMA parameters.
- `jarque_bera` and `arch_lm` (one lag) of the residuals, defined as below.

**`predictions`.** One-step-ahead predictions of the outcome in levels against
the observed values (at most 400 evenly spaced rows); the residual is the
innovation `v_t`.

**`extra`.** `model_order`, `method`, `ar`, `ma`, `seasonal`
(`{"ar", "ma", "period"}`), `roots` (inverse roots of each polynomial with
real part, imaginary part and modulus; stationary / invertible when every
modulus is below one) and `last_state`: the last `p*` values of `u`, the
estimates of the last `q*` disturbances with their covariance, and the last
`d + D s` levels. That is all `oe.forecast` needs; no series is stored.

Warnings are recorded when an AR inverse root is at or above 0.99 ("close to a
unit root; consider differencing") and when an MA inverse root is at or above
0.99 ("boundary of the invertibility region; the series may be
over-differenced").

### Worked example

Twelve years of simulated monthly log sales that follow the airline model
(every number below was produced by this code):

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(1949)
n = 144
e = rng.normal(scale=0.03, size=n + 13)
w = e[13:] - 0.4 * e[12:-1] - 0.6 * e[1:-12] + 0.24 * e[:-13]     # (1 - 0.4 L)(1 - 0.6 L^12) e_t
u = np.zeros(n + 13)
for t in range(13, n + 13):                                        # integrate (1 - L)(1 - L^12)
    u[t] = u[t - 1] + u[t - 12] - u[t - 13] + w[t - 13]
month = np.arange(1, n + 1)
df = pd.DataFrame({"month": month,
                   "lsales": 5 + 0.01 * month + 0.1 * np.sin(2 * np.pi * month / 12) + u[13:]})
df["dlsales"] = df["lsales"].diff().fillna(0.0)

fit = oe.arima(data=df, y="lsales", order=(0, 1, 1), seasonal=(0, 1, 1), period=12,
               time="month", constant=False)
print(fit.summary())
```

```
ARIMA(0,1,1)x(0,1,1)[12] regression — lsales
Observations: 131  |  Covariance: opg  |  Confidence: 95%

Term           Estimate  Std. error         z     P>|stat|   CI lower   CI upper
------------  ---------  ----------  --------  -----------  ---------  ---------
[ARMA]
ARMA:L1.ma    -0.339472   0.0886891  -3.82766  0.000129368  -0.513299  -0.165644
[ARMA12]
ARMA12:L1.ma  -0.645594   0.0808674  -7.98337  1.42394e-15  -0.804092  -0.487097
/sigma        0.0319697  0.00225822    14.157  8.44925e-46  0.0275437  0.0363958

Excluded observations: 13

log_likelihood: 261.851  |  aic: -517.703  |  bic: -509.077  |  sigma: 0.0319697
n_differenced: 131  |  iterations: 6
Wald chi2 test that all coefficients except the constant are zero: chi2(2) = 77.363, p = 1.588e-17
Ljung-Box Q(40) test of the residuals: chi2(40) = 29.8347, p = 0.8798
Jarque-Bera normality test of the residuals: chi2(2) = 1.55957, p = 0.4585
ARCH-LM(1) test of the residuals: chi2(1) = 1.29522, p = 0.2551
Warning: Differencing uses the first 13 observation(s) as starting values; the estimation sample has 131 observations.
```

```python
fit.extra["roots"]["ma"]            # [{'real': 0.3395, 'imag': 0.0, 'modulus': 0.3395}]
fit.tests["ljung_box"]["p_value_adjusted"]      # 0.825 with df = 40 - 2
```

The other covariance estimators give standard errors (0.0849, 0.0856, 0.0020)
for `covariance="nonrobust"` and (0.0822, 0.0923, 0.0018) for `"robust"`;
`method="css"` gives the estimates (-0.3313, -0.6107, 0.0328).

A regression with ARMA(1,1) errors:

```python
rng = np.random.default_rng(7)
m = 200
x = rng.normal(size=m)
eps = rng.normal(size=m + 1)
dist, prev = np.zeros(m), 0.0
for t in range(m):
    prev = 0.6 * prev + eps[t + 1] + 0.3 * eps[t]
    dist[t] = prev
armax = pd.DataFrame({"t": np.arange(m), "x": x, "y": 2.0 + 0.8 * x + dist})
fx = oe.arima(data=armax, y="y", x=["x"], order=(1, 0, 1), time="t")
# Intercept 1.766 (0.215), x 0.788 (0.049), ARMA:L1.ar 0.531 (0.077), ARMA:L1.ma 0.467 (0.081),
# /sigma 0.955 (0.052); log likelihood -275.028; Wald chi2(3) = 425.49
oe.forecast(fx, 2, exog={"x": [0.0, 1.0]})
#    period  forecast  std_error  ci_low  ci_high
# 0     200    1.3565     0.9547 -0.5147   3.2277
# 1     201    2.3360     1.3487 -0.3073   4.9794
```

### Error codes

Every failure is an `AnalysisError` with one of these codes.

| Code | When |
| --- | --- |
| `invalid_order` | `order` missing or not three non-negative integers, `seasonal` likewise, `d > 3`, `D > 2`, or `seasonal` and `period` not given together |
| `invalid_spec` | an unknown `method`, `covariance`, `missing` policy or `alpha` outside (0, 1), `period < 2`, `max_iterations < 1`, a non-boolean `constant`, duplicate regressors, `x` given as a string, the time column equal to the outcome |
| `invalid_option`, `invalid_lags` | `tolerance <= 0`; `ljung_lags` not below the number of observations |
| `model_too_large` | state dimension `max(p + P s, q + Q s + 1)` above 200 |
| `time_gaps`, `repeated_time_values`, `invalid_time` | the series is not regularly spaced, or the time column is not integer or datetime |
| `missing_values`, `empty_data`, `empty_sample`, `missing_columns`, `non_numeric_column`, `non_finite_values` | unusable input columns |
| `insufficient_observations` | too few observations after differencing |
| `constant_outcome`, `perfect_fit` | nothing is left to model |
| `nonconvergence`, `singular_information` | the likelihood has no well-defined maximum; reduce the orders |

## `oe.forecast`: dynamic forecasts

```python
oe.forecast(result, steps, *, data=None, exog=None, alpha=0.05)
```

Returns a table with one row per step: `period`, `forecast`, `std_error`,
`ci_low`, `ci_high`. `oe.forecast` is the shared dispatcher of every
time-series family; for `arima` results it calls this family's function, which
is also exported as `oe.arima_forecast`.

**Point forecasts.** From the last observation `n` the stationary part follows

    u^_(n+h) = sum_i phi*_i u^_(n+h-i) + sum_(j>=h) theta*_j e^_(n+h-j),

with `u^_t = u_t` in the sample and `e^_t = E[e_t | y_1..y_n]` for the last `q*`
periods; future disturbances are zero. The drift is added, the differences are
integrated with the last `d + D s` levels, and the regression part
`x_(n+h)'b` of the supplied future regressors is added. The forecast is for
the outcome in levels (Stata's `predict, y`).

**Standard errors.** `std_error_h = sigma sqrt(sum_(j<h) Psi_j^2 + c_h' Omega
c_h)`, where `Psi(L) = theta*(L) / (phi*(L) (1-L)^d (1-L^s)^D)` are the
MA(infinity) weights of the integrated process and the second term carries the
uncertainty `Omega` about the last `q*` disturbances under exact ML (zero for
`method="css"`, negligible for an invertible model on a long sample).
Parameter uncertainty is not included, as in Stata's `predict, mse` and
EViews. Intervals are normal: `forecast -/+ z_(1-alpha/2) std_error`.
Forecasts and standard errors equal the conditional mean and variance of the
dense multivariate normal distribution of the model to nine digits in the
tests, and statsmodels' Kalman-filter forecasts at the same parameters.

**Arguments.**

- `steps`: 1 to 10,000 (an integer; NumPy integers are accepted).
- `exog`: future values of the regressors, one row per step, when the model
  has regressors (a DataFrame, a mapping of columns or a list of records;
  categorical columns use the fitted categories). Passing `exog` to a model
  without regressors is an error.
- `data`: by default the forecast starts at the end of the estimation sample
  from `result.extra["last_state"]`. With `data` the fitted model, parameters
  fixed, is run over that table (same columns as the fit) and the forecast
  starts at its end. Use it to forecast from a sample updated with new
  observations, or from an earlier origin. A categorical regressor in `data`
  must show the fitted categories and no others.
- `period` in the output continues the integer time column; it is the step
  number `1..steps` when there is no time column or it is a datetime.

```python
oe.forecast(fit, steps=3)
#    period  forecast  std_error  ci_low  ci_high
# 0     145    6.1126     0.0320  6.0500   6.1753
# 1     146    6.1796     0.0383  6.1045   6.2547
# 2     147    6.1864     0.0437  6.1007   6.2722
```

Errors: `invalid_result` (not an `oe.arima` result), `invalid_steps`,
`invalid_option` (`alpha`), `missing_exog`, `invalid_exog`, `missing_columns`,
`missing_values`, `invalid_data` (a category in `data` the model never saw),
`insufficient_observations` (`data` too short), `time_gaps`,
`non_finite_result` (an explosive conditional estimate over a long horizon).

## Series diagnostics

All four functions take a table and the name of one numeric column, and return
a result table (`openecon.frame.DataFrame`, renders as a table, exports with
`.to_latex()`); scalar results are repeated in `.attrs`. The series must be
complete: missing values raise `missing_values`, since dropping interior
observations would change every lag. With `time=` the rows are sorted and
integer periods must be consecutive (`time_gaps`, `repeated_time_values`,
`invalid_time` otherwise).

### `oe.corrgram(data, y, *, lags=None, time=None, pacf="yw")`

One row per lag with `lag`, `acf`, `pacf`, `q`, `p_value`.

    r_k  = sum_(t=k+1..n) (y_t - ybar)(y_(t-k) - ybar) / sum_t (y_t - ybar)^2        (divisor n, Box-Jenkins)
    Q_k  = n (n + 2) sum_(j<=k) r_j^2 / (n - j)                                      (Ljung-Box)
    p_k  = Pr(chi2(k) > Q_k)

`pacf="yw"` (default) computes the partial autocorrelations by the
Durbin-Levinson recursion on `r_1..r_k` (Stata's `corrgram, yw`, EViews,
SPSS). `pacf="regression"` reports the coefficient on the `k`-th lag in an OLS
regression of `y_t` on a constant and `k` lags, which is Stata's default for
`corrgram` and `pac`. Default `lags = min(floor(n/2) - 2, 40)` (Stata).
`attrs["white_noise_band"] = 1.96 / sqrt(n)` is the usual band for the
autocorrelations of white noise.

### `oe.wntestq(data, y, *, lags=None, time=None)`

Portmanteau test for white noise: `Q` at `lags` (same default) with
`chi2(lags)`. One row (`lags`, `statistic`, `df`, `p_value`). For residuals of
a fitted ARMA model use `fit.tests["ljung_box"]`, which also gives the
p-value with the degrees of freedom reduced by the number of ARMA parameters.

### `oe.jarque_bera(data, y)`

`JB = n/6 (S^2 + (K - 3)^2 / 4)` with the moment estimators
`S = m_3 / m_2^1.5`, `K = m_4 / m_2^2`, `m_j = mean((y - ybar)^j)` (divisor
`n`, as EViews and the original paper); `chi2(2)`. One row (`statistic`, `df`,
`p_value`, `skewness`, `kurtosis`). Stata's `sktest` is a different,
small-sample adjusted test.

### `oe.archlm(data, y, *, lags=1, time=None, demean=False)`

Engle's LM test: regress `u_t^2` on a constant and `p` of its lags;
`LM = T' R^2` with `T' = n - p`, `chi2(p)`. `u` is the column as given (use it
on residuals); `demean=True` uses `y - mean(y)`. `lags` may be a list, giving
one row per order as `estat archlm, lags(1 2 4)` does. Columns: `lags`,
`statistic`, `df`, `p_value`, `r_squared`, `nobs`.

```python
series = df.iloc[1:]                          # the first difference is undefined in row 0
oe.corrgram(series, "dlsales", lags=4)
#    lag     acf    pacf        q  p_value
# 0    1  0.2423  0.2423   8.5699   0.0034
# 1    2  0.0573 -0.0015   9.0523   0.0108
# 2    3  0.0290  0.0164   9.1767   0.0270
# 3    4 -0.3163 -0.3474  24.0965   0.0001
oe.corrgram(series, "dlsales", lags=4, pacf="regression")["pacf"]   # 0.2478, 0.0032, 0.0125, -0.3655
oe.wntestq(series, "dlsales", lags=12).attrs["statistic"]           # 177.82, chi2(12)
oe.jarque_bera(series, "dlsales")       # statistic 1.5181, p 0.4681, skewness -0.0781, kurtosis 2.5200
oe.archlm(series, "dlsales", lags=[1, 4])
#    lags  statistic  df   p_value  r_squared  nobs
# 0     1   0.289345   1  0.590640   0.002038   142
# 1     4  15.121162   4  0.004456   0.108785   139
```

## `oe.tssmooth`: exponential smoothing

```python
oe.tssmooth(*, data, y, method="hwinters", alpha=None, beta=None, gamma=None,
            period=None, additive=True, time=None, forecast=0)
```

| `method` | Stata | Recursions (`f_t` is the forecast of `x_t` made at `t - 1`) |
| --- | --- | --- |
| `"exponential"` | `tssmooth exponential` | `S_t = alpha x_t + (1 - alpha) S_(t-1)`, `f_t = S_(t-1)` |
| `"dexponential"` | `tssmooth dexponential` | Brown: `S_t` as above, `S2_t = alpha S_t + (1 - alpha) S2_(t-1)`, `f_(t+1) = (2 + alpha/(1-alpha)) S_t - (1 + alpha/(1-alpha)) S2_t` |
| `"hwinters"` (default) | `tssmooth hwinters` | Holt: `a_t = alpha x_t + (1-alpha)(a_(t-1) + b_(t-1))`, `b_t = beta (a_t - a_(t-1)) + (1-beta) b_(t-1)`, `f_t = a_(t-1) + b_(t-1)` |
| `"shwinters"`, `additive=True` | `tssmooth shwinters, additive` | `a_t = alpha (x_t - s_(t-m)) + (1-alpha)(a_(t-1) + b_(t-1))`, `b_t` as Holt, `s_t = gamma (x_t - a_t) + (1-gamma) s_(t-m)`, `f_t = a_(t-1) + b_(t-1) + s_(t-m)` |
| `"shwinters"`, `additive=False` | `tssmooth shwinters` | `a_t = alpha x_t / s_(t-m) + (1-alpha)(a_(t-1) + b_(t-1))`, `s_t = gamma x_t / a_t + (1-gamma) s_(t-m)`, `f_t = (a_(t-1) + b_(t-1)) s_(t-m)` |

Note the defaults: OpenEconometrics's seasonal smoother is additive unless
`additive=False`; Stata's `shwinters` is multiplicative unless `additive` is
given.

**Parameters.** `alpha`, `beta`, `gamma` in `[0, 1]`. Any parameter the method
uses and that is left `None` is estimated by minimizing the in-sample sum of
squared one-step errors `sum (x_t - f_t)^2`, as Stata, SPSS and EViews do.
The criterion is not convex: it can have several local minima, often with one
of them in a corner of the unit cube (`alpha = 1, beta = 0` is the random walk
with drift). The search therefore evaluates a grid over
`{0, 0.1, 0.5, 0.9, 1}` for every free parameter, descends from its three best
points (BFGS with analytic derivatives in the parameterization
`parameter = sin(angle)^2`) and keeps the lowest minimum; a series longer than
5,000 observations is searched on its first 5,000 and then optimized once on
the full sample from the point found. An estimate that ends on 0 or 1 is fixed
there exactly (`attrs["at_bounds"]`) and the others are re-optimized. Two
combinations make a parameter irrelevant: with `alpha = 0` the level and trend
never update, so `beta` drops out; with `alpha = 1` the seasonal terms never
update, so `gamma` drops out. Such a parameter is reported as 0 and listed in
`attrs["not_identified"]`.

**Initial values.**

- `exponential`: the mean of the first half of the sample.
- `dexponential`, `hwinters`: intercept (at `t = 0`) and slope of an OLS line
  through the first half of the sample.
- `shwinters`: from the complete seasons in the first half of the sample (at
  least two). The trend is the change in season means per period,
  `b_0 = (xbar_last - xbar_first) / ((years - 1) m)`; the level is
  `a_0 = xbar_first - b_0 (m + 1) / 2`; the seasonal terms are the averages,
  by season, of the deviations from (ratios to) the line `a_0 + b_0 t`,
  normalized to sum to zero (to average one).

They are reported in `attrs["initial"]`.

**Result.** A table with `n + forecast` rows: `period`, `observed`,
`smoothed` (the value after `x_t` is seen: `S_t`, the level `a_t`, or level
plus / times the seasonal term) and `forecast` (in the sample the one-step
prediction `f_t`; after it the `h`-step forecasts `S_n`, `a_n + h b_n`,
`a_n + h b_n + s_(n+h-m)` or `(a_n + h b_n) s_(n+h-m)`). `observed` and
`smoothed` are missing in the forecast rows. `attrs`: `parameters`,
`estimated`, `at_bounds`, `not_identified`, `sse`, `rmse = sqrt(SSE/n)`,
`nobs`, `iterations` (over all descents), `initial`, `seasonal_period`,
`additive`.

No standard errors are computed for smoothing parameters, which is why
`tssmooth` is a table function and not a registered estimator.

```python
out = oe.tssmooth(data=df, y="lsales", method="shwinters", period=12, time="month", forecast=3)
out.attrs["parameters"]     # {'alpha': 0.5744, 'beta': 0.0, 'gamma': 0.4887}
out.attrs["at_bounds"]      # ['beta']
out.attrs["rmse"]           # 0.0308
out.tail(3)                 # periods 145-147: forecasts 6.1060, 6.1796, 6.1975
oe.tssmooth(data=df, y="lsales", method="hwinters").attrs["parameters"]
                            # {'alpha': 1.0, 'beta': 0.0}: a corner solution (random walk with drift)
```

Errors: `invalid_option` (unknown method, `period` missing or misplaced),
`invalid_parameter` (outside `[0, 1]`, or a parameter the method does not
use), `invalid_steps`, `insufficient_observations` (fewer than four
observations or fewer than two seasons), `nonpositive_series` (multiplicative
smoothing of a series that is not strictly positive), `constant_series`,
`perfect_fit` (the initial values reproduce the series exactly, so the
parameters are not identified), `missing_values`, `time_gaps`, `invalid_time`,
`nonconvergence`.

Additive Holt-Winters is not stable for every parameter in the unit cube when
the season is long: for some combinations the error recursion has a root
outside the unit circle and the one-step errors grow. The estimated
parameters avoid that region because the criterion explodes there; parameters
you fix are used as given.

## Numerical notes

- **Filters without a time loop.** `x / c(L)` is computed by cyclic reduction:
  `1/c(L) = c(-L) / d(L^2)` with `d(L^2) = c(L) c(-L)`, applied repeatedly; the
  roots of the denominator are squared at each stage, so its coefficients
  vanish doubly exponentially for a stable polynomial, and the iteration is
  exact for any polynomial (unit roots included) once the lag unit exceeds the
  sample. Cost: `q log2(min(n, decay length))` vector additions.
- **Accuracy safeguard.** The reduction loses digits when roots of `c` meet
  under squaring (roots `rho` and `-rho`, or the seasonal roots of a dense
  polynomial near `(1 - L)(1 - L^s)`). Exactly even polynomials are rewritten
  in the doubled lag unit; every result of order two or more is verified by
  its residual `x - c(L) out` and, if needed, refined (iterative refinement)
  or recomputed on the companion form by matrix doubling. Tests hold the
  filter to the accuracy of the plain recursion for such polynomials.
- **Lyapunov equations** `P = T P T' + Q` are solved by doubling, also for the
  derivative equations of the analytic gradient.
- **Standardization.** See "Standardized estimation": the optimizer and the
  numerical Hessian work on centered, unit-variance data.
- **Smoothing.** The linear smoothers are polynomial filters
  (`theta(L) e = Delta(L) x`, with the error polynomial `theta` of the
  equivalent ARIMA model), so they have no time loop either. Only the
  multiplicative seasonal smoother, which is nonlinear in the data, runs a
  scalar recursion over time, with forward derivative recursions alongside.

## Verification

`tests/test_econ_arima_oracle.py` checks the family against oracles that share
no code or algebra with it:

- an explicit Kalman filter in NumPy (Harvey's form, stationary initial
  covariance from the Kronecker solve of the Lyapunov equation), maximized by
  brute force from an OLS / white-noise start on twelve designs (ARMAX, pure
  AR and MA, drift and regressors under differencing, the airline model,
  multiplicative seasonal ARMA, 26 observations, badly scaled data, a
  categorical regressor; exact ML and CSS). Coefficients agree to 2e-6, the
  log likelihood to 1e-10, and all three covariance matrices with the
  numerical OIM, OPG and sandwich of the oracle likelihood to 2e-5;
- statsmodels' SARIMAX filter, its OPG covariance (1e-6) and its numerical
  Hessian and sandwich covariances at the same parameters;
- the conditional mean and variance of a dense multivariate normal for the
  forecasts (1e-9), from the estimation sample and from supplied data;
- textbook formulas for the correlogram, the portmanteau, Jarque-Bera and
  ARCH-LM statistics, and explicit recursions with a bounded brute-force
  search for the smoothers;
- invariances: rescaling and shifting the data, row order with a time column,
  differencing inside versus by hand, categorical versus explicit indicators,
  collinear and trailing-missing designs.

`tests/test_econ_arima_adversarial.py` covers the failure contract (empty and
tiny samples, gaps, duplicates, text where numbers are needed, options at
their bounds, magnitudes from 1e-8 to 1e8, unit roots, non-convergence).

## Performance

Apple-silicon laptop, CPU, float64, wall time of one complete call (data
preparation, estimation, covariance, residual tests), measured after a warm-up
call:

| Task | n | Time |
| --- | --- | --- |
| ARIMA(2,1,2), exact ML, `opg` / `nonrobust` / `robust` | 100,000 | 0.4 s |
| ARIMA(2,1,2), exact ML, any covariance | 1,000,000 | 2.1 s |
| ARIMA(2,1,2), CSS | 1,000,000 | 2.0 s |
| AR(3), exact ML | 1,000,000 | 0.9 s |
| ARMAX(1,0,1) with 10 regressors and a constant | 1,000,000 | 6.4 s |
| Airline model (0,1,1)(0,1,1)12 | 200,000 | 0.6 s |
| ARMA(1,1) / ARMA(2,2), screened starts | 300 | 0.2 s |
| ARMA(3,3), screened starts (over-fitted: 1 to 3 s) | 300 | 1 - 3 s |
| `oe.forecast`, 120 steps | - | under 0.02 s |
| `oe.corrgram`, 40 lags | 1,000,000 | 0.02 s |
| `oe.corrgram`, 20 lags, `pacf="regression"` | 1,000,000 | 4.4 s |
| `wntestq` + `jarque_bera` + `archlm(5)` | 1,000,000 | 0.1 s |
| `tssmooth` exponential / dexponential / hwinters | 1,000,000 | 1.0 / 1.3 / 2.3 s |
| `tssmooth` shwinters additive, m = 12 | 100,000 | 5 - 7 s |
| `tssmooth` shwinters multiplicative, m = 12 | 100,000 | 4.2 s |
| `tssmooth` shwinters, either form, m = 12 | 500 | 0.1 s |

Additive seasonal smoothing is the slow case: its error polynomial has 13
roots near the unit circle, so every evaluation needs the full `log2(n)`
reduction stages, and one million observations take about 40 seconds.

## Limitations

- One series per fit; no panel ARIMA, no weights.
- No missing observations inside the series (Stata's Kalman filter can skip
  them; OpenEconometrics asks you to fill them). Integer time values must increase by
  one (there is no `delta` option).
- AR and MA lags are consecutive `1..p`, `1..q`; Stata's subset lag lists
  (`ar(1 4)`) are not available. Multiplicative seasonal terms cover the usual
  use.
- The differenced data are modelled as a stationary process (Stata's
  approach). The exact diffuse likelihood of the undifferenced series, which
  statsmodels uses unless `simple_differencing=True`, is not implemented; the
  two differ only in how the first `d + D s` observations are treated.
- `method="ml"` requires a stationary AR part; use `method="css"` or more
  differencing for a series with a unit root.
- The search for the maximum is local with several starts; a maximum on the
  invertibility boundary of an over-parameterized model can be missed (see
  "Starting values and local maxima").
- The constant always enters as the mean of the (differenced) regression;
  there is no separate "ARMA constant" parameterization (`c = mu (1 - sum
  phi)`).
- Forecast standard errors ignore parameter uncertainty.
- `oe.forecast` continues integer time values; for datetime time columns the
  `period` column is the step number.
- State dimension `max(p + P s, q + Q s + 1)` is limited to 200.
- Exponential smoothing reports no standard errors or prediction intervals,
  and offers the classical recursions only (no damped trend, no ETS
  likelihood). Additive seasonal smoothing of more than a few hundred
  thousand observations takes tens of seconds.

## Conventions and how sure we are of them

`provenance["stata_parity_validated"]` is `False`: no number on this page has
been compared with Stata, EViews or SPSS output. The conventions were taken
from the packages' documentation as recalled, and from the textbooks.

Conventions we are confident about (documented in Stata's [TS] manual and
visible in its output):

- `arima` fits a regression with ARMA disturbances; `arima(p,d,q)` differences
  the outcome and the regressors; the constant is that of the differenced
  equation.
- The default VCE of `arima` is OPG (the output header reads "OPG std.
  err."); `vce(oim)` and `vce(robust)` are the alternatives.
- z statistics; a "Wald chi2" model test of the coefficients other than the
  constant; `/sigma` (not `sigma^2`) with a one-sided test and an interval
  truncated at zero.
- Term names `ARMA: L1.ar`, `ARMA12: L1.ma`; MA coefficients with a plus sign.
- `wntestq` and `corrgram`: Ljung-Box Q, `chi2(lags)`, default
  `min(floor(n/2) - 2, 40)` lags, autocorrelations with divisor `n`.
- `estat archlm`: `T' R^2` on the `n - p` observations of the auxiliary
  regression.

Conventions that follow the documented or textbook form but were not verified:

- **Stata's `condition` option.** OpenEconometrics sets presample disturbances and
  presample `u` to zero and sums over all `N` observations. Stata's manual
  describes the presample values as "taken to be their expected value of
  zero"; whether Stata additionally drops the first `max(p, q)` terms from
  the criterion was not verified.
- **`vce(robust)` factor.** OpenEconometrics applies `N/(N-1)`, the factor of Stata's
  `ml` sandwich; whether `arima` applies it was not verified.
- **Information criteria.** `K` counts `sigma` and `N` is the number of
  observations after differencing, as `estat ic` is understood to do.
- **Stata's default optimizer path.** Stata maximizes with a BHHH/BFGS switch
  and numerical derivatives from its own starting values; OpenEconometrics uses BFGS
  with analytic gradients and screens several starts. At a unique maximum the
  estimates coincide; with several local maxima the packages can stop at
  different ones.
- **Unit-root pile-up.** Stata prints whatever its numerical OPG gives at an
  MA estimate of -1 (often missing or enormous standard errors); OpenEconometrics
  substitutes the observed information and says so.
- **Wald chi2 of the model test.** Taken to be the joint test of every
  coefficient except the constant, as Stata's header reports for other ML
  commands.
- **`corrgram` default.** Stata's default partial autocorrelations are
  regression based; OpenEconometrics's default is Yule-Walker / Durbin-Levinson
  (`pacf="regression"` gives the other).
- **Ljung-Box degrees of freedom on residuals.** `df = lags` (Stata's
  `wntestq`); the ARMA-adjusted p-value is reported separately (EViews).
- **`tssmooth` initial values.** Mean of the first half (exponential) and a
  first-half regression line (dexponential, hwinters) follow the Stata manual.
  For `shwinters` the season-means construction above is the textbook method
  the manual describes; the exact constants Stata uses (for example
  `a_0 = xbar_1 - b_0 m/2` versus `(m + 1)/2`) and its handling of the number
  of seasons were not verified, so estimated parameters can differ slightly
  from Stata's.
- **`tssmooth` optimum.** Stata's optimizer is local; where the criterion has
  several minima OpenEconometrics reports the lowest one it finds from three starts,
  which need not be the one Stata stops at.
- **What `tssmooth` stores.** OpenEconometrics returns both the smoothed state and the
  one-step prediction; Stata's generated variable is believed to be the
  one-step prediction (our `forecast` column in the sample), which was not
  verified.
- **SPSS and EViews ARIMA.** Both default to conditional or backcast least
  squares with their own treatment of the presample; `method="css"` is the
  no-backcast variant. EViews parameterizes the constant the same way
  (regression mean); SPSS output was not compared.
