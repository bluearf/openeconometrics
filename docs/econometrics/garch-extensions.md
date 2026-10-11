# GARCH extensions: skewed innovations, component, fractional, MIDAS and multivariate models

This page documents the volatility models added for MARKET-161 (GitHub #47) on top
of [`oe.arch`](arch.md) and the [multivariate GARCH family](mgarch.md). Each
section gives the recursion and its source, the parameterization, the presample
convention, the inference contract, the declared unsupported combinations, and
the independent reference it is validated against. The earlier Gaussian
CCC/DCC/full-BEKK work (MARKET-131) is not reopened.

Everything runs natively on float64 PyTorch tensors with analytic first
derivatives (numerical Hessians of the analytic gradient), as in the rest of the
`arch` family. The reference packages (Python `arch`, R `rugarch`, `fGarch`,
`rmgarch`) are development-only oracles; their output is committed as JSON
fixtures by `scripts/generate_garch_extension_oracles.py` and
`scripts/generate_garch_extension_oracles.R`, and the tests compare against the
fixtures. `provenance["stata_parity_validated"]` stays `False`; this is not a
vendor-parity claim.

## 1. Skewed innovation distributions (`dist="skewt"`, `dist="sged"`)

Both are available for every univariate variance model of `oe.arch` (ARCH,
GARCH, GJR/TARCH, EGARCH, power ARCH, IGARCH, with ARMA disturbances,
ARCH-in-mean and variance regressors). They use **different skewing
mechanisms**, each the one of the package it is validated against; the
parameters are not interchangeable.

### Hansen skewed t (`dist="skewt"`)

Hansen, B. E. (1994), "Autoregressive conditional density estimation",
*International Economic Review* 35(3), 705-730, eq. (10)-(11); the
parameterization of Python `arch`'s `SkewStudent`. With `eta > 2` degrees of
freedom and asymmetry `-1 < lambda < 1`:

```
c   = Gamma((eta+1)/2) / (sqrt(pi (eta-2)) Gamma(eta/2))
a   = 4 lambda c (eta-2)/(eta-1),        b^2 = 1 + 3 lambda^2 - a^2
g(z) = b c (1 + ((b z + a)/(1 + s lambda))^2 / (eta-2))^(-(eta+1)/2),   s = sign(b z + a)
```

`z` has mean 0 and variance 1. Reported parameters: `/lndfm2 = ln(eta - 2)` (as
for `dist="t"`) and `/atlambda = atanh(lambda)`; `lambda > 0` skews to the
right and `lambda = 0` is the unit-variance Student t. `extra["distribution"]`
reports `df`, `lambda` and their intervals (end points of the interval of the
reported parameter, transformed), and `negative_share = E[z^2 1(z<0)]`.

R's `rugarch`/`fGarch` `"sstd"` is the Fernandez-Steel skewed t, a different
parameterization (skewness `xi > 0`); it is not offered.

### Fernandez-Steel skewed GED (`dist="sged"`)

Fernandez, C. and Steel, M. F. J. (1998), "On Bayesian modeling of fat tails
and skewness", *JASA* 93(441), 359-371, standardized as in `fGarch`'s `dsged`
and `rugarch`'s `"sged"` (C function `dsgedstd`). With `f` the unit-variance
GED of shape `nu` and skewness `xi > 0`:

```
lam = (2^(-2/nu) Gamma(1/nu)/Gamma(3/nu))^(1/2),   m1 = 2^(1/nu) lam Gamma(2/nu)/Gamma(1/nu)
mu  = m1 (xi - 1/xi),   sigma^2 = (1 - m1^2)(xi^2 + 1/xi^2) + 2 m1^2 - 1
y   = z sigma + mu,     g(z) = sigma 2/(xi + 1/xi) f(y / xi^sign(y))
```

Reported parameters: `/lnshape = ln(nu)` (as for `dist="ged"`) and
`/lnxi = ln(xi)`; `xi > 1` skews to the right, `xi = 1` is the GED.

### Conventions with skewed errors

- **Analytic derivatives** of the log density with respect to `e_t`, `ln h_t`
  and both reported parameters (closed forms, including the dependence of
  `a, b, c` and of `m1, mu, sigma` on the shape parameters).
- **GJR persistence** is `sum a + s sum g + sum b` with
  `s = E[z^2 1(z<0)]` of the fitted density (1/2 for symmetric densities), and
  multi-step GJR forecasts replace `E[e^2 1(e<0)]` by `s h`. The *presample*
  threshold news keeps the distribution-free value `h_0 / 2` (as Python `arch`
  does with `0.5 * backcast`), so the presample convention does not depend on
  the distribution parameters.
- **EGARCH** keeps the centring constant `sqrt(2/pi)` for every distribution
  (as for t and GED); EGARCH forecasts use `E|z|` of the fitted skewed density.
- **Power ARCH** persistence and forecasts use `E|z|^phi` of the fitted density.
- `E|z|^p` and `E[z^2 1(z<0)]` have no convenient closed form for these
  densities; they are computed by double-exponential (tanh-sinh / exp-sinh)
  quadrature of the density split at its kinks (`z = 0` and the switch point of
  the two pieces). Tests compare them with `scipy.integrate.quad` to 1e-9.
- **Forecast intervals** use the lower and upper quantiles of the fitted
  innovation distribution (asymmetric): Hansen's quantile in closed form through
  the Student t quantile (as `SkewStudent.ppf`), the skewed-GED quantile through
  the chi-square quantile of the GED kernel. `table.attrs["interval_quantiles"]`
  records them; they are exact one step ahead.
- Student-t degrees of freedom that diverge raise `nonconvergence` as for
  `dist="t"`.

### Unsupported

- The bounded Dataset (streaming) route of `arch` implements only normal, t and
  GED errors: a `Dataset` with a skewed distribution raises
  `streaming_unsupported`, and resident tables above 100,000 rows with a skewed
  distribution stay on the in-memory route instead of the automatic replay
  adapter. Dynamic predictions from Dataset histories (which rebuild the state
  through the replay adapter) are refused the same way.

### Variance regressors

Variance regressors already existed before this work and were not
reimplemented: `variance_x=[...]` is Stata's `het()`, multiplicative
heteroskedasticity `exp(l_0 + z_t'l)` in place of the variance constant
(`l_0 + z_t'l` additive in `ln h` for EGARCH), for every univariate model
including GARCH, GJR and EGARCH. Positivity of that constant holds by
construction (an exponential); the remaining coefficients are unconstrained, as
for every `oe.arch` model, and a parameter point with a non-positive `h_t` is
infeasible (rejected by the line search), with a warning for negative
coefficients at the estimates. Additive (rugarch `vxreg`-style) regressors are
not offered. The skewed densities combine with `variance_x` (tested).

### Validation (tests/test_econ_arch_extensions.py)

| check | reference | measured |
| --- | --- | --- |
| Hansen skewed-t log density (4 parameter pairs x 49 points) | Python `arch` 8.0 `SkewStudent.loglikelihood`; explicit NumPy formula | max abs 9e-15 |
| skewed-GED log density (4 pairs x 49 points) | `fGarch::dsged(log=TRUE)`, `rugarch::ddist("sged")`; explicit NumPy formula | max abs 3e-13 (fGarch), rel 1e-13 (rugarch) |
| GARCH / GJR / EGARCH(1,1) skewed-t per-observation log likelihood at fixed parameters (with and without a mean) | `arch` `GARCH`/`EGARCH.compute_variance(backcast = h_0)` + `SkewStudent` | max abs 5e-15 per observation; total rel 8e-16 |
| sGARCH(1,1)-sged log likelihood at fixed parameters (with and without a mean) | `rugarch::ugarchfilter` with one prepended presample observation (see the R generator) | max abs 5e-14 per observation |
| every variance model x both skewed densities (20 specifications with ARMA, ARCH-in-mean, variance regressors) | explicit-loop recursion of `test_econ_arch_oracle` + NumPy densities; numerical derivatives | log likelihood 1e-10; gradient rel 1e-6; scores 2e-5 |
| zero-mean MLE, skewed t, GARCH / GJR / EGARCH | `arch_model(mean="Zero", dist="skewt").fit(backcast=mean(y^2), cov_type="classic")` | estimates rel <= 2e-5; SEs rel <= 5e-5 (GJR 1.5e-3, see below) |
| zero-mean MLE, sGARCH-sged | `stats::optim` maximum of the rugarch filter likelihood, `numDeriv::hessian` (Richardson) | estimates rel 3e-8; log likelihood equal; SEs rel 7e-4 (see below) |
| OPG and robust covariance | inverse outer product of numerical scores; sandwich with `N/(N-1)` | rel 1e-4 |
| moments, quantiles | `scipy.integrate.quad` | 1e-9 / 1e-10 |

Tolerances on standard errors are looser than on the likelihood for a stated
reason: they come from *numerical* Hessians. The GJR threshold term has a
second derivative that jumps at a zero residual and arch's finite-difference
step straddles it; a GED kernel with shape below 2 has an unbounded second
derivative at its mode, so every finite-difference Hessian depends on its step
([arch.md, Limitations](arch.md#limitations)). OPG covariances, which need only
first derivatives, agree to 1e-4.

## 2. Component GARCH (`model="cgarch"`)

Engle, R. F. and Lee, G. G. J. (1999), "A long-run and short-run component
model of stock return volatility", in Engle and White (eds.), *Cointegration,
Causality and Forecasting*, Oxford University Press, 475-497. Parameterization
of rugarch's `csGARCH` (C function `csgarchfilter` in `filters.c`):

```
q_t = omega + rho q_(t-1) + phi (e_(t-1)^2 - h_(t-1))                    permanent component
h_t = q_t + alpha (e_(t-1)^2 - q_(t-1)) + beta (h_(t-1) - q_(t-1))       conditional variance
```

Engle and Lee write the permanent equation as `q_t = w + rho (q_(t-1) - w) + ...`;
here `omega = w (1 - rho)` and the long-run variance is `w = omega / (1 - rho)`.

```python
fit = oe.arch(data=df, y="ret", time="day", model="cgarch")      # CGARCH(1,1)
```

Terms: `ARCH:L1.arch` (alpha), `ARCH:L1.garch` (beta), `COMPONENT:L1.rho`,
`COMPONENT:L1.phi`, `COMPONENT:Intercept` (omega). The mean equation accepts
regressors and ARMA disturbances; every innovation density (including the
skewed ones) is available.

- **Evaluation.** With `x_t = (h_t, q_t)` the model is
  `x_t = c + A x_(t-1) + B e_(t-1)^2`, `A = [[beta - phi, rho - alpha - beta], [-phi, rho]]`.
  Eliminating `q` gives the exact second-order recursion
  `h_t = omega (1 - alpha - beta) + (phi + alpha) s_(t-1) - (alpha rho + phi (alpha + beta)) s_(t-2) + (beta - phi + rho) h_(t-1) + (phi (alpha + beta) - beta rho) h_(t-2)`
  (`s = e^2`), evaluated with the family's polynomial filter (no loop over
  time); every derivative path uses the same filter, and `q` follows from its
  own first-order filter.
- **Presample** (the family's `arch0(xb)` convention): `h_0 = q_0 = e_0^2 = h0`,
  the mean squared residual at the current parameters, so `h_1 = q_1 = omega + rho h0`.
  rugarch instead starts at observation 1 with `q_1 = omega / (1 - rho)` and
  `h_1 = mean(e^2) + q_1`.
- **No constraints are imposed** (family convention); a point with a
  non-positive `h_t` is infeasible. The Engle-Lee conditions for a positive,
  identified decomposition (`omega > 0`, `alpha > 0`, `phi > 0`, `beta > phi`,
  `alpha + beta < rho < 1`; rugarch's `.csgarchcon` imposes `rho > alpha + beta`
  and `beta > phi`) are checked at the estimates and a violation is reported as
  a warning.
- **Persistence** is `max(rho, alpha + beta)`, the larger root of the forecast
  recursion; `unconditional_variance = omega / (1 - rho)` when both components
  are stationary.
- **Forecasts** are exact conditional expectations:
  `E_n[q_(n+k)] = w + rho^(k-1) (q_(n+1) - w)` and
  `E_n[h_(n+k) - q_(n+k)] = (alpha + beta)^(k-1) (h_(n+1) - q_(n+1))`.
  `extra["state"]["permanent"]` stores `q_n` next to the last innovation and
  variance; a result without it is refused (`invalid_result`).
- **Covariance**: OPG (default), observed information or the quasi-ML sandwich,
  as for every `oe.arch` model.

### Unsupported

`unsupported_combination` for orders other than (1,1) (`arch=1, garch=1`), an
ARCH-in-mean term, and variance regressors (rugarch adds `vxreg` terms to the
permanent equation; not offered). The Dataset route refuses `cgarch`
(`streaming_unsupported`).

### Validation

| check | reference | measured |
| --- | --- | --- |
| h, q paths and per-observation log likelihood from rugarch's start-up state (two fixed-parameter cases, with and without a mean) | `rugarch::ugarchfilter` csGARCH vs the explicit NumPy two-state recursion in the tests | h, q rel 1e-14; log likelihood 2e-14 |
| kernel log likelihood under the oe.arch presample convention | the same NumPy recursion | 3e-14 |
| analytic gradient and per-observation scores (normal, t, skewed t; with ARMA(1,1) disturbances) | numerical derivatives of the NumPy recursion | gradient rel 1e-6, scores 2e-5 |
| MLE (3000 observations) | Nelder-Mead + BFGS maximization of the NumPy recursion (SciPy) | estimates rel 5e-6 |
| observed-information covariance | inverse Hessian from central differences of complex-step gradients of the NumPy recursion | rel 1e-5 |
| OPG and robust covariance | numerical scores of the NumPy recursion | rel 1e-5 / 5e-4 |
| package fit | `rugarch::ugarchfit` (own start-up, solver "hybrid") | every estimate within 0.25 standard errors (different start-up likelihood); rugarch's standard errors use a Hessian step of 0.1 that crosses `rho = 1` and are not compared |
| forecasts | the two-state recursion with `E[e^2] = h` | rel 1e-10 |

## 3. FIGARCH (`model="figarch"`)

Baillie, R. T., Bollerslev, T. and Mikkelsen, H. O. (1996), "Fractionally
integrated generalized autoregressive conditional heteroskedasticity",
*Journal of Econometrics* 74, 3-30, eq. (8)-(9):
`(1 - phi L)(1 - L)^d e_t^2 = omega + (1 - beta L) nu_t`, i.e. the ARCH(infinity)
form

```
h_t = omega / (1 - beta) + lambda(L) e_t^2,
lambda(L) = 1 - (1 - phi L)(1 - L)^d / (1 - beta L) = sum_(i>=1) lambda_i L^i,
```

in the parameterization and with the weight recursion of Python `arch`'s
`FIGARCH` (`figarch_weights`):

```
delta_1 = d,         delta_i = (i - 1 - d)/i delta_(i-1)
lambda_1 = phi - beta + d,   lambda_i = beta lambda_(i-1) + delta_i - phi delta_(i-1)
```

```python
fit = oe.arch(data=df, y="ret", model="figarch")                  # FIGARCH(1,d,1)
fit = oe.arch(data=df, y="ret", model="figarch", arch=0)          # FIGARCH(0,d,1)
fit = oe.arch(data=df, y="ret", model="figarch", truncation=500)
```

Terms: `ARCH:L1.phi`, `ARCH:L1.beta`, `ARCH:d`, `ARCH:Intercept` (omega).
`arch=0` drops `phi` and `garch=0` drops `beta` (FIGARCH(p,d,q) with p, q in
{0, 1}); the mean equation accepts regressors and ARMA disturbances and every
innovation density is available.

- **Truncation.** The ARCH(infinity) weights are truncated at `truncation` lags
  (default 1000, Python `arch`'s default; 1 to 5000), recorded in
  `extra["model"]["truncation"]`. The weights decay hyperbolically
  (`lambda_i ~ i^(-1-d)`), so `1 - sum_(i<=K) lambda_i` is of order `K^(-d)`:
  the truncation is a material modelling choice, not a numerical detail.
- **Presample.** Squared innovations before the sample equal `h_0`, the mean
  squared residual at the current parameters (Python `arch`'s `backcast`):
  `h_t = omega/(1 - beta) + sum_(i<t) lambda_i e_(t-i)^2 + h_0 sum_(i=t..K) lambda_i`.
  Samples shorter than the truncation are allowed.
- **Evaluation.** The weights and their derivatives with respect to
  `(phi, d, beta)` come from a cumulative product and a first-order filter; the
  convolution with the squared innovations (and every derivative path) is a
  real FFT, so there is no loop over time and no `T x K` matrix.
- **Domain and positivity.** `0 < d < 1` and `beta < 1` are the parameter
  domain (other points are infeasible); a search that runs into `d -> 0` or
  `d -> 1` stops with `nonconvergence` and recommends `model="garch"` or
  `model="igarch"`. Otherwise no constraints are imposed (family convention);
  negative truncated weights or a non-positive constant at the estimates (the
  Baillie-Bollerslev-Mikkelsen positivity conditions; Python `arch` imposes the
  sufficient `0 <= phi <= (1 - d)/2`, `0 <= beta <= d + phi`) are reported as a
  warning.
- **Persistence** is `sum_(i<=K) lambda_i` (1 for the untruncated model with
  `d > 0`); `unconditional_variance` is not reported (it does not exist for the
  untruncated model).
- **Forecasts** continue the truncated recursion with `E_n[e^2] = E_n[h]`
  beyond the sample: exact conditional expectations of the truncated model.
  `extra["state"]["squared_residuals"]` stores the last `min(truncation, N)`
  squared innovations (bounded by the truncation); a result without them is
  refused (`invalid_result`).

### FIEGARCH: not implemented

Fractionally integrated EGARCH (Bollerslev and Mikkelsen 1996) is not
implemented: neither reference package available to this work (Python `arch`
8.0, R `rugarch` 1.5) implements it, so there is no reliable independent
reference for its recursion and presample conventions.

### Unsupported

`unsupported_combination` for lags other than 0/1 in `arch` and `garch`, an
ARCH-in-mean term and variance regressors; `truncation` with any other model is
`invalid_option`. The Dataset route refuses `figarch` (`streaming_unsupported`).

### Validation

| check | reference | measured |
| --- | --- | --- |
| per-observation log likelihood and variance path at fixed parameters, FIGARCH(1,d,1), (0,d,1), (1,d,0), with and without a mean, n = 500 < truncation | Python `arch` `FIGARCH(truncation=1000).compute_variance(backcast = h_0)` | log likelihood 1e-14 per observation; variance rel 1e-13 |
| weights and their derivatives | the published recursion in NumPy, complex-step derivatives | rel 1e-12 / 1e-10 |
| scores and gradient (ARMA(1,0) mean; normal and skewed GED) | explicit NumPy loop over time with the published weights; numerical derivatives | scores 2e-5, gradient rel 1e-6 |
| zero-mean MLE (a FIGARCH(1,0.4,1) series simulated by `arch` and a GARCH series) | `arch_model(vol="FIGARCH").fit(backcast=mean(y^2), cov_type="classic")` | estimates rel 2e-5; standard errors rel 1e-4 |
| forecasts, short samples (n < truncation) | the truncated recursion written out | rel 1e-11 |

## 4. GARCH-MIDAS (`oe.garch_midas`)

Engle, R. F., Ghysels, E. and Sohn, B. (2013), "Stock market volatility and
macroeconomic fundamentals", *Review of Economics and Statistics* 95(3),
776-797, with the conventions of the authors-adjacent R package **mfGARCH**
0.2.2 (O. Kleen; Conrad and Kleen 2020, *Journal of Applied Econometrics* 35,
19-45), whose source on [github.com/cran/mfGARCH](https://github.com/cran/mfGARCH)
(`R/helper_functions.R`: `calculate_phi`, `calculate_tau_mf`, `llh_mf`;
`src/sum_tau.cpp`; `src/calculate_g.cpp`) defines every recursion below. For
day `t` of low-frequency period `s`:

```
y_t   = mu + sqrt(tau_s g_t) Z_t                                     Gaussian quasi-likelihood
tau_s = exp(m + theta sum_(k=1..K) phi_k(w1, w2) X_(s-k))              long-run component
u_t   = (y_t - mu) / sqrt(tau_(s(t)))
g_t   = (1 - alpha - beta - gamma/2) + (alpha + gamma 1(u_(t-1) < 0)) u_(t-1)^2 + beta g_(t-1)
phi_k = c_k / sum_j c_j,     c_k = (k/(K+1))^(w1-1) (1 - k/(K+1))^(w2-1)       beta weights
```

```python
fit = oe.garch_midas(data=daily, y="ret", period="month", covariate="ip_growth", lags=24)
fit = oe.garch_midas(data=daily, y="ret", period="month", lags=12)          # realized variance
oe.forecast(fit, 20)                     # dispatches to oe.garch_midas_forecast
```

- **Data.** One row per high-frequency observation; `period` holds integer
  low-frequency periods (consecutive integers that never decrease in time
  order, e.g. `12 * year + month`; dates must be converted). `covariate` must
  be constant within each period (one value per period); without it the
  covariate is the realized variance `RV_s = sum_(t in s) y_t^2` of the
  period (Engle-Ghysels-Sohn's fixed-span RV, of the raw outcome). Rows are
  sorted by `time` when given; missing values inside the series are refused
  (`time_gaps`).
- **Burn-in.** The first `K` periods only supply covariate lags; the
  likelihood is that of the days of periods `K+1..S` (as mfGARCH). `nobs` is the
  number of likelihood days; `extra["model"]` records the burn-in periods and
  observations.
- **Short-run start.** `g = 1`, its unconditional mean, on the first likelihood
  day. **Deliberate difference:** mfGARCH starts at `g_zero = var(y)`, the sample
  variance of the returns (a quantity in the units of `y`, while `g` is
  unit-free).
- **The boundary between periods.** `u_(t-1)` is standardized by the long-run
  component of its own day's period (mfGARCH's `calculate_g` on
  `(ret - mu)/sqrt(tau)` aligned by day); Engle, Ghysels and Sohn's eq. (4)
  divides `r_(i-1,t)` by the current period's `tau_t`. The two differ on the
  first day of each period only; mfGARCH's convention is used.
- **Weights.** `weighting="beta_restricted"` fixes `w1 = 1` and estimates `w2`
  (mfGARCH's default; needs `K >= 2`), `"beta_unrestricted"` estimates both (needs
  `K >= 3`; with `K = 2` the two are not separately identified and the request is
  refused). With `K = 1` there is no weight parameter. The domain is
  `w1, w2 > 0` (mfGARCH constrains them to be at least 1).
- **Long-run specification.** Only the log form `ln tau = m + theta sum phi X`
  (mfGARCH; Engle-Ghysels-Sohn's macro specification) is offered;
  Engle-Ghysels-Sohn's level form for realized variance,
  `tau = m + theta sum phi RV`, is not implemented. One covariate (mfGARCH's
  optional second covariate is not offered).
- **Estimator and domain.** Gaussian quasi-ML by BFGS with analytic gradients
  (the short-run recursion and every derivative path are first-order filters;
  the long-run part a K-column product), numerical Hessian of the analytic
  gradient, from three short-run starting candidates. The domain is
  `alpha + beta + gamma/2 < 1` (positive short-run constant, as mfGARCH) and
  `|beta| < 1`; no sign constraints on `alpha, beta` (mfGARCH requires them
  non-negative), with a warning when they are negative.
- **Covariance.** `"robust"` (default; quasi-ML sandwich with `N/(N-1)`,
  mfGARCH's `rob.std.err` without that factor), `"opg"` or `"nonrobust"`; z
  inference. mfGARCH's `opg.std.err` additionally rescales the OPG errors by
  `sqrt((mean(z^4) - 1)/2)`; that rescaling is not applied.
- **Reported.** Terms `Intercept`, `GARCH:alpha`, `GARCH:gamma`, `GARCH:beta`,
  `MIDAS:m`, `MIDAS:theta`, `MIDAS:w1`, `MIDAS:w2`; metrics `persistence`
  (`alpha + beta + gamma/2`) and mfGARCH's `variance_ratio`
  (`100 var(ln mean_s tau) / var(ln mean_s (tau g))` over the likelihood
  periods); `extra`: the estimated weights, the last 400 long-run and short-run
  components and the forecast state (last `g`, residual, `tau` and the last `K`
  covariate values).
- **Forecasts** (`oe.forecast` / `oe.garch_midas_forecast`): the forecast days
  belong to the period after the sample (`new_period=True`, default; its `tau`
  uses the last `K` covariate values including the last sample period, which
  must then be complete) or to the last sample period (`new_period=False`).
  `E_n[g_(n+1)]` follows the recursion and `E_n[g_(n+k)] = 1 + rho^(k-1)
  (E_n[g_(n+1)] - 1)`, `rho = alpha + beta + gamma/2` (exact for symmetric
  innovations); the variance forecast is `tau` times that. mfGARCH's
  `forecast_garch` helper uses `gamma/2 1(ret < 0)` in the first step; the model
  recursion `gamma 1(u < 0)` is used here.

### Unsupported

Dataset route (`streaming_unsupported`), weights, regressors in the mean,
non-Gaussian innovations, a second covariate, the level RV specification,
rolling-window RV, `beta_unrestricted` with fewer than 3 lags
(`unsupported_combination`), and periods that are not consecutive integers
(`invalid_period`).

### Validation (tests/test_econ_arch_extensions_midas.py)

mfGARCH itself is not installed in the oracle environment (it was not
installed, per the work rules), and `midasr`'s beta polynomials (`nbeta`,
`nbetaMT`) use a different grid (`(j - 1)/(d - 1)` with end-point offsets), so
the reference is an explicit NumPy transcription of mfGARCH's `llh_mf`,
`calculate_tau_mf`, `sum_tau`, `calculate_phi` and `calculate_g` (loops over
days, the period merge and burn-in included), called with `g_zero = 1`.

| check | reference | measured |
| --- | --- | --- |
| per-day log likelihood: GJR + covariate (K = 12, restricted), GARCH + RV (K = 6, unrestricted), K = 1, GARCH + RV (K = 3) | the transcription | 1e-11 |
| per-observation scores | complex-step derivatives of the transcription | rel 1e-8 |
| beta weights and derivatives | `calculate_phi`, numerical derivatives | 1e-13 / 1e-6 |
| MLE (1890 days, 90 months) | Nelder-Mead + BFGS on the transcription from mfGARCH's starting values | estimates rel 2e-5 |
| observed-information, OPG and robust covariance | complex-step gradients (central differences for the Hessian) of the transcription | rel 1e-5 (entries below 1e-6 of the largest compared absolutely) |
| seeded simulation | the data-generating parameters | every estimate within 3 standard errors |
| variance ratio, forecasts | pandas group means; the recursion written out | rel 1e-10 |

## 5. Multivariate: diagonal VECH, diagonal BEKK and asymmetric DCC

`oe.mgarch_dvech`, `oe.mgarch_dbekk` and `oe.mgarch_adcc` extend the
[multivariate GARCH family](mgarch.md) and share its contract: joint Gaussian
conditional ML of every mean and (co)variance parameter, the fixed recorded
initial covariance (`initial_covariance`, default the covariance of the first
`min(75, N)` residuals), consecutive periods, `max_n` 512 (at most 2048),
dimension 2 or 3, constant or zero mean, full observed information in physical
reporting units by native float64 autodiff of the complete matrix recursion
(the family's convention), an interior-maximum and reporting-unit score check,
persisted state and forecast contracts, and no Dataset route.

```python
fit = oe.mgarch_dvech(df, ["r1", "r2"], time="t")
fit = oe.mgarch_dbekk(df, ["r1", "r2", "r3"])
fit = oe.mgarch_adcc(df, ["r1", "r2"])
oe.forecast(fit, 10)
```

### Diagonal VECH (Bollerslev, Engle and Wooldridge 1988)

`H_t = C + A o (e_(t-1) e_(t-1)') + B o H_(t-1)` (Hadamard products; each element
`h_ij` is its own GARCH(1,1) recursion). Terms `C:i:j`, `A:i:j`, `B:i:j` (lower
triangles). The first `H` is the initial covariance.

- **Domain.** C positive definite (Cholesky factor with a positive diagonal);
  variance coefficients `a_ii, b_ii > 0` with `a_ii + b_ii < 1 - 1e-6` (simplex
  map); off-diagonal `a_ij, b_ij` free. Positive definiteness of every fitted
  `H_t` is enforced on the sample path (a trial point with a non-positive-definite
  `H_t` is rejected by the line search, never projected), and each forecast step
  is checked. At the estimates the stationarity of every element
  (`|a_ij + b_ij| < 1`, `|b_ij| < 1`) is verified.
- **Ding and Engle's (2001) sufficient conditions** (C positive definite, A and B
  positive semidefinite) guarantee positive definiteness for all data. They are
  reported (`constraints["ding_engle_psd"]`, smallest eigenvalues of A and B) and a
  warning is issued when they fail, but they are **not imposed**: in development
  the maximum of simulated samples of a few hundred periods frequently lay on the
  boundary of that cone (A or B singular, where the score is not zero), which would
  make the estimator unusable; a rank-one boundary is exactly the diagonal BEKK
  model below, which guarantees positive definiteness by construction. Also
  reported: whether the unconditional covariance `C / (1 - A - B)` is positive
  definite (warning otherwise).
- **Forecasts** `E[H_(t+k)] = C + (A + B) o E[H_(t+k-1)]` after the first step:
  exact conditional expectations.

### Diagonal BEKK (Engle and Kroner 1995)

`H_t = C C' + A' e e' A + B' H_(t-1) B` with `A = diag(a)`, `B = diag(b)`, so
`h_ij = (CC')_ij + a_i a_j e_i e_j + b_i b_j h_ij`. Terms `C:i:j` (lower
triangular, positive diagonal), `A:i`, `B:i`.

- **Identification.** `a` and `-a` (and `b`, `-b`) give the same model: `a_1, b_1 > 0`.
- **Domain.** The spectral radius of `A (x) A + B (x) B` is
  `max_i (a_i^2 + b_i^2)`; a radial map enforces `a_i^2 + b_i^2 < 1 - 1e-6`.
  Positive definiteness holds by construction (`H_t` is `CC'` plus positive
  semidefinite terms).
- **Forecasts:** exact, as for the full BEKK.

### Asymmetric DCC (Cappiello, Engle and Sheppard 2006)

Univariate GARCH(1,1) variances as in `mgarch_dcc` and, for the standardized
residuals `z_t` and `n_t = z_t 1(z_t < 0)` (elementwise; Cappiello, Engle and
Sheppard, *Journal of Financial Econometrics* 4(4), 537-572, eq. (6)-(7), scalar
version):

```
Q_(t+1) = (1 - a - b) Qbar - g Nbar + a z_t z_t' + b Q_t + g n_t n_t',     Q_1 = Qbar
R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2,   H_t = D_t R_t D_t
```

- **Targets.** `Qbar` is the jointly estimated SPD unit-diagonal target of
  `mgarch_dcc` (not the empirical correlation; see mgarch.md); `Nbar` is the
  sample second moment `(1/T) sum n_t n_t'` at the current variance parameters
  (Cappiello-Engle-Sheppard's estimator of `E[n n']`), so its dependence on the
  GARCH parameters is part of the likelihood and of the observed information.
  With `g = 0` the model is exactly `mgarch_dcc` (tested).
- **Domain.** `a, b, g > 0` and `a + b + delta g < 1 - 1e-6` with
  `delta = lambda_max(Qbar^-1/2 Nbar Qbar^-1/2)`, which makes the intercept
  `(1 - a - b) Qbar - g Nbar` positive definite (Cappiello-Engle-Sheppard's
  condition; rmgarch's `.adcccon`). A simplex map on `(a, b, delta g)` enforces it;
  `g` is reported in physical units and `constraints["adcc_persistence"]` is
  `a + b + delta g`. `Nbar` is persisted with the state.
- **Forecasts.** The next covariance is exact; later `Q` follow the
  Engle-Sheppard (2001, section 7) approximation with `E[n n']` replaced by
  `Nbar`, `Q_(k+1) = (1 - a - b) Qbar + (a + b) Q_k`, labeled as approximate.
- **Differences from rmgarch** (`dccspec(model = "aDCC")`): rmgarch estimates
  in two steps (univariate rugarch fits, then the DCC parameters), uses the
  centred sample covariances `Qbar = cov(z)` and `Nbar = cov(n)` (denominator
  `n - 1`), starts the univariate recursions at `h_1 = mean(e^2)` and the `Q`
  recursion from a presample `Q_0 = Qbar` with presample `z_0 = n_0 = (1, ..., 1)`
  (`rdcc.cpp`, `dccnormC2`). Correlation is not a spillover measure: none of
  these models identifies the direction of volatility transmission from
  correlations alone.

### Unsupported

Dimensions above 3 (`mgarch_dimension`), orders other than (1,1),
non-Gaussian innovations, regressors in the mean, robust covariance, weights,
the full (non-diagonal) VECH, VCC and the Dataset route.

### Validation (tests/test_econ_mgarch_extensions.py)

| check | reference | measured |
| --- | --- | --- |
| aDCC per-period joint log likelihoods, correlations, variances, `Qbar`, `Nbar` | `rmgarch::dccfit` (aDCC, sGARCH margins) vs an explicit NumPy recursion with rmgarch's conventions | log likelihoods 1e-10; correlations 1e-10; `Qbar`, `Nbar` 1e-12 |
| aDCC kernel path and likelihood (oe conventions; also with an explicit `Nbar`) | the same NumPy recursion | covariances rel 1e-12; log likelihood 1e-9 |
| scalar limits | aDCC with `g = 0` vs the DCC kernel; every conditional variance of dvech/dbekk/aDCC vs `oe.arch`'s univariate GARCH(1,1) kernel with matched presample | 1e-10; rel 1e-12 |
| diagonal VECH / BEKK likelihood, autodiff gradient and Hessian | explicit NumPy recursions in physical units; finite differences | 1e-10; 3e-5; 2e-4 |
| joint fits (dvech, dbekk, aDCC) | inverse Richardson finite-difference Hessian of the NumPy recursions at the estimates; zero numerical gradient | covariance rel 3e-4 (aDCC 1e-3) |
| dvech / dbekk forecasts | the expectation recursion written out | rel 1e-12 |
| aDCC one-step forecast | recursion written out | rel 1e-10 |
| aDCC estimates vs rmgarch's two-step estimates | `rmgarch::dccfit` | within 2 standard errors (different estimator: two-step, centred targets, other start-up) |
