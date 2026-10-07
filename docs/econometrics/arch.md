# ARCH and GARCH models: conditional heteroskedasticity

`oe.arch` fits regressions whose error variance changes over time: ARCH, GARCH,
GJR/threshold GARCH, EGARCH, power ARCH and integrated GARCH variance
equations, with an optional ARCH-in-mean term, ARMA disturbances, regressors in
the variance equation, and normal, Student-t or generalized-error (GED)
innovations. It is the counterpart of Stata's `arch` and of EViews' ARCH
estimation. `oe.forecast` (alias `oe.arch_forecast`) continues a fitted model
beyond the sample.

Everything is implemented in OpenEconometrics on float64 PyTorch tensors: the
likelihood, its analytic gradient (the recursive derivatives of the conditional
variance), the optimizer and the covariance estimators. No estimation library
runs at fit time. GARCH-type models are evaluated without any loop over time,
so a GARCH(1,1) regression on one million observations fits in a few seconds
(timings at the end).

```python
import openecon as oe

fit = oe.arch(data=df, y="ret", time="day", arch=1, garch=1)      # arch ret, arch(1) garch(1)
print(fit.summary())
oe.forecast(fit, steps=10)                                        # mean and variance forecasts
```

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `arch y x, arch(1)` | `oe.arch(data=df, y="y", x=["x"], model="arch", arch=1)` |
| `arch y x, arch(1) garch(1)`; EViews `arch(1,1) y c x` | `oe.arch(data=df, y="y", x=["x"])` |
| `arch y, arch(1/2) garch(1)` | `..., arch=2, garch=1` (or `arch=[1, 2]`) |
| `arch y, arch(1) tarch(1) garch(1)`; EViews `arch(1,1,thrsh=1)` | `..., model="gjr"` (alias `"tarch"`; see the sign convention below) |
| `arch y, earch(1) egarch(1)`; EViews `arch(1,1,egarch)` | `..., model="egarch"` |
| `arch y, parch(1) pgarch(1)`; EViews `arch(1,1,parch)` | `..., model="parch"` |
| EViews GARCH with the IGARCH restriction | `..., model="igarch"` |
| `arch y x, arch(1) garch(1) archm` | `..., archm="variance"` |
| `arch ..., archm archmexp(sqrt(X))` | `..., archm="sd"` (`"log"` for ln h) |
| `arch y x, ar(1) ma(1) arch(1) garch(1)` | `..., ar=1, ma=1` |
| `arch ..., het(z)` | `..., variance_x=["z"]` |
| `arch ..., distribution(t)` / `distribution(ged)` | `..., dist="t"` / `dist="ged"` |
| `arch ...` (default `vce(opg)`) / `vce(oim)` / `vce(robust)` | `covariance="opg"` (default) / `"nonrobust"` / `"robust"` |
| `arch ..., noconstant` | `..., constant=False` |
| `predict v, variance dynamic(...)`; EViews Forecast | `oe.forecast(fit, steps=h)` |
| `estat archlm` on a series or residuals | `oe.archlm(df, "y", lags=4)` (arima family) |

## `oe.arch`

### The model

```
y_t = x_t'b + psi g(h_t) + u_t
u_t = sum_j rho_j u_(t-j) + sum_k theta_k e_(t-k) + e_t,        e_t = sqrt(h_t) z_t
```

`z_t` is independent with mean 0 and variance 1, so `h_t` is the variance of
the innovation `e_t` given the past. The ARCH-in-mean function `g` is `h`
(`archm="variance"`), `sqrt(h)` (`"sd"`) or `ln h` (`"log"`). The disturbance
`u_t` may follow an ARMA process with Stata's signs (`ar`, `ma`); note that the
autoregression is in the structural disturbance `u_t = y_t - x_t'b - psi g(h_t)`,
as in Stata's `arch` and `arima`, not in `y_t` itself.

The conditional variance follows one of (lags `i` from `arch`, `j` from `garch`):

| `model` | variance equation | terms |
| --- | --- | --- |
| `"arch"`, `"garch"` | `h_t = omega + sum a_i e_(t-i)^2 + sum b_j h_(t-j)` (Engle 1982; Bollerslev 1986) | `ARCH:L1.arch`, `ARCH:L1.garch`, `ARCH:Intercept` |
| `"gjr"` / `"tarch"` | `h_t = omega + sum a_i e_(t-i)^2 + sum g_i e_(t-i)^2 1(e_(t-i) < 0) + sum b_j h_(t-j)` (Glosten, Jagannathan and Runkle 1993) | `ARCH:L1.arch`, `ARCH:L1.tarch`, `ARCH:L1.garch` |
| `"egarch"` | `ln h_t = omega + sum a_i z_(t-i) + sum g_i (abs(z_(t-i)) - sqrt(2/pi)) + sum b_j ln h_(t-j)`, `z = e / sqrt(h)` (Nelson 1991) | `ARCH:L1.earch`, `ARCH:L1.earch_a`, `ARCH:L1.egarch` |
| `"parch"` | `h_t^(phi/2) = omega + sum a_i abs(e_(t-i))^phi + sum b_j h_(t-j)^(phi/2)`, power `phi` estimated (Ding, Granger and Engle 1993, symmetric form) | `ARCH:L1.parch`, `ARCH:L1.pgarch`, `POWER:power` |
| `"igarch"` | the GARCH equation with `sum a_i + sum b_j = 1` imposed (Engle and Bollerslev 1986); the constant is kept | as GARCH |

- **Threshold sign.** `ARCH:L1.tarch` multiplies `1(e < 0)`, the GJR and EViews
  convention: a positive coefficient means that bad news raises the variance
  more than good news. Stata's `tarch()` term multiplies `1(e > 0)` instead, so
  Stata's `arch` coefficient equals `a + g` here and Stata's `tarch`
  coefficient equals `-g`. The fitted variances and the likelihood are the same.
- **EGARCH.** `earch` is the signed (leverage) term and `earch_a` the
  magnitude term, as in Stata. The centring constant is `sqrt(2/pi)`, the
  normal value of `E|z|`, for every innovation distribution.
- **IGARCH.** The last GARCH coefficient is not a free parameter: it is one
  minus the other ARCH and GARCH coefficients. It is still reported, with the
  standard error implied by the others.
- **Variance regressors.** `variance_x=["z"]` adds regressors to the variance
  equation as Stata's `het()` does: multiplicative heteroskedasticity
  `exp(l_0 + z_t'l)` replaces `omega` (in EGARCH `l_0 + z_t'l` enters `ln h_t`
  directly). The terms are `HET:z` and `HET:Intercept`; there is then no
  `ARCH:Intercept`.

The innovation density is chosen by `dist`, always standardized to unit
variance:

| `dist` | density of `z` | reported parameter |
| --- | --- | --- |
| `"normal"` | standard normal | none |
| `"t"` | Student t with `nu > 2` degrees of freedom, scaled by `sqrt((nu-2)/nu)` | `/lndfm2 = ln(nu - 2)` |
| `"ged"` | generalized error distribution with shape `s` (`s = 2` normal, `s = 1` Laplace) | `/lnshape = ln(s)` |

`extra["distribution"]` reports `df` or `shape` on the natural scale with the
confidence interval obtained by transforming the end points of the interval of
the log parameter, as Stata prints them.

### Arguments

| Argument | Meaning |
| --- | --- |
| `data` | DataFrame, mapping of columns or list of records |
| `y` | outcome column |
| `x` | optional list of regressors of the mean equation |
| `time` | optional time column (integer periods or datetimes); without it the row order is the time order |
| `arch` | ARCH order `q` (lags `1..q`) or a list of lags; default `1` |
| `garch` | GARCH order `p` or a list of lags; default `1` (none for `model="arch"`) |
| `model` | `"arch"`, `"garch"` (default), `"gjr"` / `"tarch"`, `"egarch"`, `"parch"`, `"igarch"` |
| `dist` | `"normal"` (default), `"t"`, `"ged"` |
| `archm` | `None`, `"variance"`, `"sd"`, `"log"` |
| `ar`, `ma` | AR and MA orders or lists of lags of the disturbance, e.g. `ar=1`, `ma=[1, 4]` |
| `constant` | include the constant of the mean equation (default `True`) |
| `variance_x` | optional list of regressors of the variance equation |
| `covariance` | `"opg"` (default, as in Stata), `"nonrobust"` (OIM), `"robust"` |
| `categorical` | regressors (of either equation) to expand into treatment-coded indicators |
| `test_lags` | lags of the residual diagnostics (default 5 for ARCH-LM, `min(floor(N/2) - 2, 40)` for Ljung-Box) |
| `max_iterations`, `tolerance` | BFGS iteration limit (500) and gradient tolerance (1e-8) |
| `missing` | `"raise"` (default) or `"drop"` |
| `alpha` | significance level of the confidence intervals (0.05) |

A lag argument given as a number is an order (`arch=2` means lags 1 and 2); a
list names individual lags (`arch=[1, 3]`), like Stata's numlists. At least one
ARCH term is required (without it the GARCH coefficients are not identified),
no lag may exceed 60, and weights are not supported.

### Time order, gaps and missing values

With `time`, rows are sorted by that column; integer periods must be
consecutive and repeated periods are an error. A datetime column is ranked and
its observations are treated as consecutive (weekends and holidays of daily
data are simply skipped), which is recorded as a warning. Without `time` the
rows are used in their input order.

ARCH recursions need an uninterrupted series. `missing="drop"` may only remove
rows at the start or the end of the sample; a missing value inside the series
raises `time_gaps` (Stata restarts the recursion after a gap; OpenEconometrics asks you
to fill the gap or restrict the sample).

### Estimator

Full maximum likelihood on the raw parameters:

```
ln L = sum_t [ ln f(e_t / sqrt(h_t)) - ln(h_t) / 2 ]
```

- **No constraints.** As in Stata, neither positivity of the variance
  coefficients nor stationarity is imposed. A parameter point at which some
  `h_t` is not positive is infeasible (the optimizer backs away from it).
  Persistence of one or more is reported with a warning, and so are negative
  variance coefficients at the estimates (a negative `arch`, `garch` or
  variance constant, or `arch + tarch < 0` for GJR): the fitted variance is
  positive at every observation of the sample, but it is not guaranteed to stay
  positive in forecasts.
- **Centred regressors.** During estimation the regressors of the mean
  equation and of the variance equation are centred at their sample means.
  With a constant in the equation this is an exact reparameterization
  (`c = c* - mean(x)'b`); estimates, covariance and every reported quantity are
  mapped back to the raw regressors. It keeps the constant and the slope of a
  regressor with a large offset (a calendar year, a price level) from being
  numerically collinear: without it `x + 1e6` produced standard errors that
  were wrong by a factor of 50.
- **Presample values** follow Stata's defaults. ARMA disturbances before the
  sample are zero (`arma0(zero)`). Presample variances and squared innovations
  equal `h_0 = (1/T) sum_t e~_t^2`, the mean squared residual of the mean
  equation with its ARMA terms *at the current parameters* (`arch0(xb)`); the
  dependence of `h_0` on the mean parameters is part of the analytic gradient.
  Presample news take their expected values: `e^2 = h_0`, `e^2 1(e<0) = h_0/2`,
  `z = 0`, `|z| - sqrt(2/pi) = 0`, `|e|^phi = h_0^(phi/2)`.
- **Gradient.** Analytic: the derivative of the variance state obeys a linear
  recursion, `d h_t = G_t + sum a_i d(e_(t-i)^2) + sum b_j d h_(t-j)` and its
  analogues, which is solved for all parameters at once. The derivatives are
  verified against numerical differentiation of an independent implementation.
  For GARCH-type models the gradient is accumulated in reverse mode: the
  weights `dl_t/dh_t` are filtered backwards in time once, so the cost of a
  gradient does not grow with the number of regressors. Per-observation scores
  (OPG and robust covariances) use the forward recursions.
- **Optimizer.** BFGS with a strong-Wolfe line search
  (`engines.optimize.maximize_bfgs`) on parameters divided by their natural
  scale (an exact linear reparameterization, so results are equivariant to the
  units of the data). The initial curvature is the outer product of the scores
  at the starting values. Convergence requires a negative definite Hessian and
  `g'(-H)^-1 g <= 1e-10`; a few Newton steps polish the BFGS iterate when
  needed.
- **Hessian.** Numerical derivative (Ridders extrapolation of central
  differences) of the analytic gradient, for the convergence test and the
  observed information. The first difference step is small (`1e-6` in the
  scaled parameters): the gradients of the GJR, power-ARCH and GED likelihoods
  have kinks at zero residuals (`1(e<0) e`, `|e|^(phi-1)`, `|e|^(s-1)`), and a
  stencil that straddles the zero of one residual gives a wrong second
  derivative for that observation. With the optimizer's default first step the
  observed-information standard errors of GED models were off by up to 6% in
  samples of a few hundred observations; with the small step they agree with
  an independent complex-step oracle to five digits or better.
- **Starting values.** OLS for the mean equation, a Hannan-Rissanen regression
  for the ARMA terms and a short list of variance-equation candidates scaled to
  the residual variance (for example `a = 0.05, b = 0.90`,
  `omega = s^2 (1 - a - b)`). The candidates are ranked by their likelihood;
  BFGS runs from the best and tries the next ones only after a failure.
  `provenance["optimizer"]` records which candidate was used.

**How the recursions are computed.** GARCH, GJR, IGARCH and power-ARCH models
without ARCH-in-mean are linear filters of known inputs: the variance path and
every derivative path are obtained by polynomial filtering (cyclic reduction),
with no loop over time (`provenance["optimizer"]["engine"] == "filter"`).
EGARCH (which divides by the lagged variance) and ARCH-in-mean (where the
residual depends on the current variance) need one loop over time for the
state path `(e_t, h_t)`; that loop runs on plain floats, carries no
derivatives and does the minimum per period (for EGARCH(1,1): three
multiplications, one exponential and one comparison). All derivative paths are
then obtained without a loop from a parallel prefix scan of the time-varying
linear recursion (`"scan"`).

### EGARCH: kinks of the likelihood

The term `g (|z| - sqrt(2/pi))` makes the EGARCH likelihood only piecewise
smooth: it has a kink wherever a standardized residual is zero, and the
gradient jumps across it. OpenEconometrics handles this explicitly rather than with
loose convergence tolerances:

- Hessians are taken on the smooth piece of the likelihood through the point
  (the signs of the residuals are held fixed while differentiating), so that a
  finite-difference stencil never straddles a kink. The gradient jumps are
  multiplied by terms with conditional mean zero (the scores of later
  variances), so the expectation of the piece Hessian is the information
  matrix: it plays the role of the observed information.
- Often the maximum lies exactly *on* a kink (in our simulations in 10% to 40%
  of the samples, depending on the model): the likelihood rises towards
  `z_t = 0` from both sides for one observation, or for a few. A smooth
  optimizer cannot converge there. For EGARCH, BFGS is therefore only asked to
  get close, and an active-set Newton method finishes the fit by solving the
  first-order conditions of the non-smooth problem: `z_t = 0` for the active
  observations and zero in the convex hull of the one-sided gradients (an
  observation becomes active when its residual changes sign during a Newton
  step on the current piece and is released when its weight leaves [-1, 1]).
  `provenance["optimizer"]` then lists `kink_observations` (0-based positions
  in time order) and `kink_weights`, and the covariance is computed on the
  piece defined by those weights. A fit that ends on a kink costs about four
  times as many likelihood evaluations as one that does not.
- Kinks are entered one at a time, at the first one crossed on the way to the
  Newton target (the classical active-set step). The surfaces `z_t = 0` are
  hyperplanes in the mean parameters, so no more of them than there are mean
  parameters can be active together; the maximum can be a vertex of several.
- **GED errors.** At a zero residual the curvature of the GED density is
  unbounded when the shape is below 2. When an EGARCH maximum lies on a kink,
  the density kernel `|z|^s` of the observations on the kink (which is
  identically zero on the kink, with zero gradient for `s > 1`) is left out of
  the scores and of the Hessian, and the result carries a warning. Without
  this the Hessian is dominated by that single observation: such fits used to
  report observed-information standard errors of order 1e-6, or failed.

### Covariance and inference

All tests are z tests and Wald chi-squared tests, as in Stata.

| `covariance` | estimator | Stata |
| --- | --- | --- |
| `"opg"` (default) | inverse outer product of the per-observation analytic scores `(S'S)^-1` | `vce(opg)`, the default of `arch` (its table header reads "OPG std. err.") |
| `"nonrobust"` | inverse observed information `(-H)^-1`, `H` the numerical Hessian of the analytic gradient | `vce(oim)` |
| `"robust"` | quasi-ML sandwich `N/(N-1) (-H)^-1 S'S (-H)^-1` with the full Hessian (Bollerslev and Wooldridge 1992); valid when the innovation distribution is misspecified | `vce(robust)` |

No small-sample correction other than the `N/(N-1)` of the sandwich is
applied; `aic`, `bic` and the tests use the number of observations `N` and
the number of free parameters.

Standard errors of an IGARCH model follow from the free parameters by the
restriction (the covariance matrix is singular by construction).

### What the result contains

Coefficients, in Stata's order: the mean equation (equation named after the
outcome, `Intercept` first), `ARCHM:sigma2` | `ARCHM:sigma` | `ARCHM:lnsigma2`,
`ARMA:L<j>.ar`, `ARMA:L<k>.ma`, `HET:<z>` and `HET:Intercept`, the `ARCH:` terms
of the table above, `ARCH:Intercept`, `POWER:power`, and `/lndfm2` or
`/lnshape`.

`metrics`:

| name | meaning |
| --- | --- |
| `log_likelihood`, `aic`, `bic` | `aic = -2 lnL + 2k`, `bic = -2 lnL + k ln N`, `k` the number of free parameters |
| `persistence` | `sum a + sum b` (GARCH); `sum a + sum g / 2 + sum b` (GJR); `sum b` (EGARCH, the autoregressive coefficient of `ln h`); `E|z|^phi sum a + sum b` (power ARCH, with the moment of the fitted distribution); `1` (IGARCH) |
| `unconditional_variance` | `omega / (1 - persistence)` for covariance-stationary GARCH and GJR models without variance regressors; otherwise empty |
| `iterations` | BFGS (plus polishing) iterations |

`tests`:

| name | test |
| --- | --- |
| `model` | Wald chi2 that all mean-equation coefficients except the constant are zero (regressors, ARCH-in-mean and ARMA terms) |
| `ljung_box` | Ljung-Box Q of the standardized residuals `z_t = e_t / sqrt(h_t)`, chi2 with `lags` degrees of freedom (Stata's `wntestq`); with ARMA terms the Box-Pierce correction `lags - (p + q)` is reported alongside as `df_adjusted` / `p_value_adjusted` (EViews' convention) |
| `ljung_box_squared` | Ljung-Box Q of `z_t^2`: ARCH effects the variance equation missed |
| `arch_lm_residuals` | Engle's ARCH-LM test on `z_t` |
| `jarque_bera` | normality of `z_t` (expected to reject for `dist="t"` or `"ged"`) |

`predictions`: the one-step-ahead conditional mean against the outcome
(a bounded chart sample); residuals are the innovations `e_t`.

`extra`: `model` (the specification), `persistence`, `unconditional_variance`,
`unconditional_log_variance` (EGARCH: `omega / (1 - sum b)`), `distribution`
(`df` or `shape` with its interval), `state` (the last `max lag` innovations,
disturbances and variances, the presample variance and the last period: what
`oe.forecast` needs) and `conditional_variance_tail` (the last 400 `h_t`, for a
chart). No per-observation array of the full sample is stored.

### Worked example

```python
import math
import numpy as np
import pandas as pd
import openecon as oe

rng = np.random.default_rng(42)
n = 1500
z = rng.standard_t(8, size=n) * math.sqrt(6 / 8)       # unit-variance Student t shocks
e, h = np.zeros(n), np.ones(n)
for t in range(1, n):
    h[t] = 0.05 + 0.04 * e[t - 1] ** 2 + 0.10 * e[t - 1] ** 2 * (e[t - 1] < 0) + 0.85 * h[t - 1]
    e[t] = math.sqrt(h[t]) * z[t]
df = pd.DataFrame({"day": np.arange(1, n + 1), "ret": 0.05 + e})

fit = oe.arch(data=df, y="ret", time="day", model="gjr", dist="t")
print(fit.summary())
# GJR-GARCH(1,1) regression, Student t errors — ret
# Observations: 1500  |  Covariance: opg  |  Confidence: 95%
# Term             Estimate  Std. error         z     P>|stat|
# [ret]
# Intercept       0.0178464   0.0200403  0.890525     0.373184
# [ARCH]
# ARCH:L1.arch    0.0141693   0.0210117  0.674351     0.500088
# ARCH:L1.tarch     0.13987   0.0402576   3.47438  0.000512032
# ARCH:L1.garch    0.798823   0.0515043   15.5098  2.97668e-54
# ARCH:Intercept  0.0894689   0.0286576   3.12199    0.0017963
# /lndfm2           1.53764    0.289261   5.31576  1.06215e-07
# log_likelihood: -1836  |  aic: 3684.01  |  bic: 3715.89  |  persistence: 0.882928
# unconditional_variance: 0.764221  |  iterations: 9
# ARCH-LM(5) test of the standardized residuals: chi2(5) = 1.51125, p = 0.9118

fit.extra["distribution"]     # {'name': 't', 'df': 6.65, 'df_ci': [4.64, 10.20]}

# Standard errors of the same fit with covariance="nonrobust" (OIM) and "robust":
#   0.0200162, 0.0253154, 0.0378811, 0.0578064, 0.0305936, 0.2528986
#   0.0201207, 0.0310257, 0.0359294, 0.0674271, 0.0338499, 0.2239747

oe.forecast(fit, steps=5)
#    period  mean_forecast  variance_forecast  std_error    ci_low   ci_high
# 0    1501       0.017846           0.633987   0.796233 -1.573519  1.609211
# 1    1502       0.017846           0.649234   0.805750 -1.592540  1.628233
# 2    1503       0.017846           0.662695   0.814061 -1.609150  1.644843
# 3    1504       0.017846           0.674581   0.821329 -1.623676  1.659369
# 4    1505       0.017846           0.685076   0.827693 -1.636395  1.672088
```

The Stata equivalent is `arch ret, arch(1) tarch(1) garch(1) distribution(t)`
(with the threshold sign convention described above).

### Error codes

| code | when |
| --- | --- |
| `invalid_spec` | unknown model, distribution, covariance or option value; `model="arch"` with GARCH terms; no ARCH term; the time column used as a variable |
| `invalid_lags` | a lag list that is not made of distinct positive integers up to 60 |
| `invalid_option` | non-positive `tolerance` |
| `numerical_failure` | the residual variance of the outcome is outside 1e-120 to 1e120 (the variance constant and its sampling variance would leave float64): rescale the outcome |
| `missing_values`, `missing_columns`, `non_numeric_column`, `non_finite_values`, `empty_data` | data problems (shared with every estimator) |
| `time_gaps`, `repeated_time_values`, `invalid_time` | the series is not regularly spaced, or missing values lie inside it |
| `insufficient_observations` | fewer observations than parameters plus the longest lag plus 5 |
| `model_too_large` | an EGARCH or ARCH-in-mean model whose derivative recursion (one transition map per observation, `8 n (lags)^2` bytes) would need more than 1 GiB: use shorter lags or a shorter sample |
| `constant_outcome`, `perfect_fit` | no variance to model |
| `nonconvergence` | the likelihood has no regular maximum that the optimizer could reach; the message says why when it can tell (see Limitations) |
| `singular_information` | a parameter is not identified, so standard errors are undefined |

## `oe.forecast` / `oe.arch_forecast`: mean and variance forecasts

```python
oe.forecast(fit, steps=20)                    # dispatches to oe.arch_forecast
oe.arch_forecast(fit, 20, data=longer_df)     # start at the end of another sample
oe.arch_forecast(fit, 5, exog=future_x)       # models with regressors
```

Returns a table with one row per step: `period` (the integer time value
continued from the sample, or the step number without an integer time column),
`mean_forecast`, `variance_forecast`, `std_error`, `ci_low`, `ci_high`.

From the last observation `n` the fitted recursions are continued with future
innovations replaced by their expectations:

```
GARCH / GJR / IGARCH   h^_(n+k) = omega + sum a_i N_(n+k-i) + sum g_i M_(n+k-i) + sum b_j h^_(n+k-j)
                        N = e^2, M = e^2 1(e<0) in the sample;   N = h^, M = h^/2 beyond it
EGARCH                 ln h^ continued with z = 0 and |z| - sqrt(2/pi) = E|z| - sqrt(2/pi)
power ARCH             s^ = h^(phi/2) continued with |e|^phi = E|z|^phi s^
mean                   y^_(n+k) = x_(n+k)'b + psi g(h^_(n+k)) + u^_(n+k),   future e = 0 in the ARMA part
```

- For GARCH, GJR and IGARCH the variance forecast is the exact conditional
  expectation `E_n[h_(n+k)]`; it converges to the unconditional variance at the
  rate of the persistence (and grows linearly for IGARCH with a positive
  constant).
- For EGARCH and power ARCH the reported variance is the transformation of the
  forecast state, `exp(E_n[ln h])` and `(E_n[h^(phi/2)])^(2/phi)`: exact one
  step ahead and the usual plug-in forecast beyond that (it is not `E_n[h]`).
  `E|z|` and `E|z|^phi` use the fitted innovation distribution; with Student-t
  errors and `phi >= df` the moment does not exist and the forecast raises
  `non_finite_result`.
- `std_error` is the root mean squared error of the mean forecast,
  `sqrt(sum_(j<k) psi_j^2 h^_(n+k-j))` with the MA(infinity) weights of the
  ARMA part (`sqrt(h^_(n+k))` without ARMA terms). The interval uses the
  quantile of the fitted innovation distribution and is exact one step ahead.
  Parameter uncertainty is ignored, as in Stata's `predict` and in EViews.
- `data=` runs the fitted model (parameters fixed) over another table with the
  same columns and forecasts from its end; `exog=` supplies the future values
  of the regressors of either equation, one row per step.

Stata: `tsappend, add(k)`, then `predict v, variance dynamic(...)` and
`predict yhat, y dynamic(...)`.

Errors: `invalid_result` (not an `oe.arch` result, or a result without its
stored end-of-sample state), `invalid_steps` (1 to 10000), `invalid_option`,
`missing_exog`, `invalid_exog`, `missing_columns`, `missing_values`,
`non_finite_result` (the forecast variance leaves its domain, for example with
a negative variance constant).

## Verification

`tests/test_econ_arch*.py` compare the implementation with two independent
oracles that share no code with the tensor kernels or with each other:

- `test_econ_arch_oracle.py`: every model written as an explicit loop over time
  on Python floats with SciPy's densities (`norm`, `t`, `gennorm`).
- `test_econ_arch_verify.py` (the verify-and-repair pass): GARCH-type models
  through `scipy.signal.lfilter`, EGARCH and ARCH-in-mean through a loop on
  complex numbers, densities written out with `loggamma`. Every function
  accepts complex parameters, so gradients and per-observation scores are
  complex-step derivatives (exact to rounding) and the Hessian is a central
  difference of those gradients. The oracle is maximized on its own (SciPy
  BFGS from the data-generating values, then Newton steps to a gradient below
  1e-8).

What is checked:

- Log likelihood and per-observation contributions of the kernel equal the
  first oracle to 1e-10 for 23 specifications covering every variance model,
  distribution, ARMA structure, ARCH-in-mean form and variance regressors; the
  analytic gradient and scores equal numerical derivatives; the two evaluation
  engines agree, and the reverse-mode gradient equals the sum of the forward
  scores.
- For 15 models (GARCH with normal, Student-t and GED errors, ARCH, GARCH(2,1),
  GJR, GJR-t, power ARCH, ARMA-GARCH, variance regressors, EGARCH, EGARCH-t
  with variance regressors, three ARCH-in-mean forms) the estimates equal the
  second oracle's own maximum to 1e-6 relative and 1e-4 standard errors, and
  the full OPG, OIM and robust covariance matrices equal those of the oracle
  to 2e-5 (in units of the standard errors); z statistics, p-values,
  intervals, AIC/BIC, persistence, unconditional variance and the Wald model
  test are recomputed with NumPy and SciPy. IGARCH is checked against the
  oracle maximized over the free parameters with the delta-method covariance.
  Seven richer specifications (sparse lags, GJR(2,2)-GED, IGARCH with MA
  errors, power ARCH in mean, GJR-in-mean with ARMA errors, variance regressors
  and Student-t errors, EGARCH-in-mean, ARCH(3) without a constant) are checked
  for the likelihood value, the first-order conditions and the OPG covariance.
- With the ARCH coefficient at zero, the likelihood and the scores of the mean
  and ARMA parameters equal those of statsmodels' state-space ARMA model
  (`SARIMAX`) started from a known zero presample state.
- Invariances: affine changes of the units of the outcome and of regressors
  (including offsets of 1e8 and scales of 1e-50 to 1e50), row order with a time
  column, unused columns; GJR on the negated outcome reproduces Stata's
  `tarch` parameterization (`arch' = arch + tarch`, `tarch' = -tarch`) and
  EGARCH on the negated outcome flips the sign of `earch`.
- EGARCH maxima on a kink satisfy the non-smooth first-order conditions on both
  oracles (zero standardized residual at the kink, zero in the convex hull of
  the one-sided gradients, negative definite piece Hessian) and are not
  improved by a simplex search or by random perturbations.
- Residual diagnostics equal explicit NumPy algebra and statsmodels'
  `acorr_ljungbox`, `het_arch` and `jarque_bera`; ARMA residuals equal
  `scipy.signal.lfilter`.
- Forecasts of every model equal a hand recursion on the oracle's path and the
  closed forms where they exist (geometric reversion of GARCH, linear growth of
  IGARCH, the MA(infinity) weights of ARMA(1,1), Student-t quantiles and
  `E|z|`).
- `test_econ_arch_adversarial.py`: empty, tiny and all-missing samples,
  constant and two-valued outcomes, outliers, collinear and constant
  regressors, text and boolean columns, gaps, duplicates and missing values in
  time, units from 1e-150 to 1e150, options at and beyond their bounds, bad
  forecast arguments and tampered results: each case gives a complete finite
  result or an `AnalysisError`.

Convergence on simulated samples of 600 observations (40 seeds per model,
after the repairs of the verify pass): GARCH, GJR, GED-GARCH, ARMA-GARCH,
GARCH-in-mean, variance regressors and EGARCH (plain, with AR errors, in mean,
with variance regressors, with GED errors, without regressors) converged in
every sample; 8% to 25% of the EGARCH fits ended on a kink, and every such
solution was verified to be a local maximum of the oracle. EGARCH with
Student-t errors failed once (the degrees of freedom diverged), an
over-parameterized EGARCH(2,2) fitted to EGARCH(1,1) data twice, IGARCH once,
and power ARCH in 13 of 40 samples (an independent simplex search confirms that
the likelihood of those samples peaks at a power below 1; see Limitations).

Defects found and repaired by the verify pass (each has a regression test):

- the default covariance was the observed information; Stata's `arch` reports
  OPG standard errors by default;
- the numerical Hessian used the optimizer's default first step and was wrong
  by up to 12% of the covariance for GED, 0.2% for power ARCH and 0.01% for GJR
  models when a residual was close to zero;
- regressors were not centred: an offset of 1e6 gave standard errors wrong by
  a factor of 50, 1e7 a spurious maximum and 1e8 no convergence;
- the EGARCH active-set method cycled between two adjacent pieces when three
  or more residuals changed sign at once, and tried to make more kinks active
  than there are mean parameters (about 5% of EGARCH(1,1) samples failed);
- EGARCH with GED errors on a kink reported observed-information standard
  errors of order 1e-6 (or failed) because of the unbounded curvature of the
  GED density at zero;
- `oe.arch_forecast` raised a raw `KeyError` on a result without its stored
  state; outcomes scaled by 1e-150 or 1e150 failed with an unhelpful message;
  negative variance coefficients were reported without a warning.

Parity with Stata output has not been measured:
`provenance["stata_parity_validated"]` is `False`.

## Performance

Wall time of the full `oe.arch` call (estimation, numerical Hessian,
covariance, diagnostics) on an Apple M-series laptop, 6 threads, float64, best
of two runs while other jobs were running; three regressors in the mean
equation unless stated:

| model | observations | time |
| --- | --- | --- |
| GARCH(1,1), `opg` / `nonrobust` / `robust` | 100,000 | 0.12 - 0.14 s |
| GARCH(1,1), GED errors | 100,000 | 0.28 s |
| GARCH(2,2) | 100,000 | 0.23 s |
| GJR-GARCH(1,1) | 100,000 | 0.15 s |
| power ARCH(1,1) | 100,000 | 0.23 s |
| IGARCH(1,1) | 100,000 | 0.16 s |
| GARCH(1,1) with ARMA(1,1) disturbances | 100,000 | 0.34 s |
| GARCH(1,1)-in-mean | 100,000 | 1.9 s |
| EGARCH(1,1), smooth maximum | 100,000 | 1.8 s |
| EGARCH(1,1), maximum on a kink | 100,000 | 8.6 - 10.1 s |
| GARCH(1,1) | 1,000,000 | 0.8 s |
| GARCH(1,1), 10 regressors | 1,000,000 | 1.8 s |
| GARCH(1,1), 10 regressors, robust | 1,000,000 | 1.8 s |
| GJR-GARCH(1,1), 10 regressors | 1,000,000 | 2.2 s |
| EGARCH(1,1), smooth maximum | 1,000,000 | 19 s |
| GARCH(1,1)-in-mean | 1,000,000 | 19 s |

`oe.forecast` takes a few milliseconds (1000 steps: 6 ms). The time of the
GARCH-type models grows linearly with the number of observations. EGARCH and
ARCH-in-mean models are slower because their state path needs a loop over time
(about 0.25 microseconds per observation and evaluation for EGARCH(1,1)); most
of the remaining time of every model is the numerical Hessian (six or more
gradient evaluations per parameter). A fit whose EGARCH maximum lies on a kink
needs several Hessians (about five times the time of a smooth maximum); in
samples of 100,000 observations that happened in two of three simulated
samples, because the more observations there are, the more standardized
residuals are close to zero.

## Limitations

- One series per fit; no weights (Stata's `arch` accepts only `iweight`s); no
  gaps inside the series.
- No positivity or stationarity constraints, as in Stata. Negative variance
  coefficients are reported with a warning. Two consequences are reported as
  `nonconvergence` with an explanation instead of an estimate:
  - **Unbounded likelihood.** With a negative ARCH or GARCH coefficient the
    variance of a single observation can approach zero, where the likelihood
    has a spike rather than a maximum. This typically means a lag that the data
    do not support (for example ARCH(2) fitted to ARCH(1) data); drop the lag
    or use `model="egarch"`.
  - **Power ARCH with a power of 1 or less.** The news term `|e|^phi` then has
    a kink or cusp at a zero residual and the likelihood has a ridge of
    non-differentiable local maxima at every observation; the optimizer stops
    on one of them. The power is weakly identified in short samples: with a
    true power of 1.3 the likelihood peaked below 1 in about 30% of simulated
    samples of 600 observations and 10% of samples of 2000. There is no option
    to fix the power; `model="garch"` is the power-2 member of the family.
- Power ARCH is the symmetric Ding-Granger-Engle form; the asymmetric
  variants (Stata's `aparch`, `nparch`, `tparch`, `sdgarch`, `abarch`, ...) and
  component or fractionally integrated GARCH are not implemented.
- The ARCH-in-mean term uses the current conditional variance only (no
  `archmlags()`), and one of three fixed transformations.
- The Student-t likelihood is stopped at 1000 degrees of freedom: when the
  data are indistinguishable from normal the fit raises `nonconvergence` and
  recommends `dist="normal"` instead of reporting a point on the flat ridge.
- The observed information is a numerical derivative of the analytic gradient.
  For GED models with a small shape (and power ARCH with a small power) the
  exact Hessian itself contains terms `|e_t|^(s-2)` that are very large for the
  residuals nearest to zero, so OIM and robust standard errors of those models
  vary more from sample to sample than OPG ones.
- An EGARCH maximum on a kink is a non-standard estimate (one or two residuals
  are exactly zero). Its covariance is computed on the smooth piece selected by
  the kink weights; with GED errors the density kernel of the kink
  observations is left out. Stata has no comparable treatment (its optimizer
  stops near such a point under its tolerances).
- EGARCH and power-ARCH forecasts beyond one step are plug-in forecasts of the
  transformed state, not conditional expectations of the variance.
- Forecast standard errors ignore parameter uncertainty and, for ARCH-in-mean
  models, the feedback of variance uncertainty on the mean.
- Outcomes whose residual variance is outside 1e-120 to 1e120 must be rescaled
  (`numerical_failure`).
- EGARCH and ARCH-in-mean models with long lags on long samples are refused
  (`model_too_large`) when their derivative recursion would need more than
  1 GiB: for example an EGARCH(6,6) on 1,000,000 observations (EGARCH(5,5)
  still fits). GARCH, GJR, IGARCH and power-ARCH models without ARCH-in-mean
  have no such limit.

## Conventions and how sure we are of them

Certain (textbook definitions, verified numerically against two oracles): the
likelihoods of the three unit-variance densities, the variance recursions, the
OPG / OIM / sandwich covariance formulas, z inference, the information
criteria (`k` = free parameters, `N` = observations).

Taken from Stata's documentation of `arch`, not verified against Stata output:

| convention | what OpenEconometrics does | confidence |
| --- | --- | --- |
| default VCE | OPG (`vce(opg)`; Stata's coefficient table is headed "OPG std. err.") | high |
| reference distribution | z tests and a Wald chi2 model test | high |
| term names | `ARCH:L1.arch`, `L1.garch`, `L1.earch`, `L1.earch_a`, `L1.egarch`, `L1.parch`, `L1.pgarch`, `ARCHM:sigma2`, `ARMA:L1.ar`, `HET:`, `POWER:power`, `/lndfm2`, `/lnshape` | high (Stata prints `_cons` where we print `Intercept`; `ARCHM:sigma` and `ARCHM:lnsigma2` are our names for Stata's `sigma2ex` with `archmexp()`) |
| EGARCH equation | `earch` multiplies `z`, `earch_a` multiplies `abs(z) - sqrt(2/pi)`, for every distribution | high for the normal; medium for t and GED (the manual writes the normal constant) |
| variance regressors | `exp(l0 + z'l)` replaces the constant; additive in `ln h` for EGARCH (Stata's `het()`) | high |
| ARMA structure | AR terms act on the structural disturbance `y - x'b - psi g(h)`; presample disturbances are zero (`arma0(zero)`) | medium-high |
| presample variance | `arch0(xb)`: the mean squared residual at the current parameters, recomputed at every evaluation; it also primes `e^2` | medium (the manual's wording; with ARCH-in-mean we compute it without the in-mean term, and for EGARCH we prime `z = 0`, `abs(z) - sqrt(2/pi) = 0`, which the manual does not spell out) |
| robust covariance | full Hessian sandwich with `N/(N-1)` (Stata's `_robust`) | medium for the factor |
| Wald model test | all mean-equation coefficients except the constant, the ARCH-in-mean coefficient and the ARMA terms | medium (matches the degrees of freedom of the manual's examples) |
| observations with gaps | refused (`time_gaps`); Stata restarts the recursions after a gap | deliberate difference |

Deliberate differences from Stata, chosen by the specification:

- **Threshold term**: `1(e < 0)` (GJR, EViews) rather than Stata's `1(e > 0)`;
  the mapping is exact (`arch_Stata = arch + tarch`, `tarch_Stata = -tarch`)
  and is tested.
- **IGARCH** is not a Stata model. It keeps the variance constant (Engle and
  Bollerslev's integrated GARCH with drift); EViews' IGARCH restriction also
  removes the constant.
- **Infeasible points**: a parameter vector with a non-positive `h_t` is
  rejected by the line search. We do not know how Stata treats such points
  internally.
- **Diagnostics** (`ljung_box`, `ljung_box_squared`, `arch_lm_residuals`,
  `jarque_bera`) are not part of Stata's `arch` output; the default lag choices
  are `min(floor(N/2) - 2, 40)` (Stata's `wntestq`) and 5 lags for ARCH-LM
  (ours).
- **Persistence of power ARCH** uses `E|z|^phi` of the fitted distribution;
  EGARCH persistence is `sum b`.

Differences from EViews: EViews backcasts the presample variance with an
exponential smoother (parameter 0.7) instead of the sample mean of squared
residuals, estimates an uncentred EGARCH equation (`abs(z)` rather than
`abs(z) - sqrt(2/pi)`, which only shifts the constant for the normal
distribution), and reports the Student-t degrees of freedom and the GED
parameter on their natural scale (here: `extra["distribution"]`).
