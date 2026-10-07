# Further time-series models: Prais-Winsten, ARDL, filters, unobserved components, Markov switching, thresholds

The `tsmodels` family adds the regression-type time-series tools of Stata's
[TS] manual and EViews that are not ARIMA, ARCH or VAR models: regression with
AR(1) errors (`oe.prais`), autoregressive distributed lags with the bounds test
(`oe.ardl`), trend-cycle filters (`oe.tsfilter`), unobserved-components models
(`oe.ucm`, `oe.ucm_components`, `oe.forecast`), Markov-switching regressions
(`oe.mswitch`, `oe.mswitch_probabilities`) and threshold regression
(`oe.threshold`).

The family also includes [unrestricted NARDL](nardl.md), `oe.nardl` and
`oe.nardl_multipliers`: positive/negative partial sums, symmetry tests and
pointwise multiplier inference. Specialized NARDL bounds calibration,
imposed symmetry constraints and bootstrap bands remain partial coverage.

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `prais y x1 x2` (Prais-Winsten) | `oe.prais(data=df, y="y", x=["x1", "x2"], time="t")` |
| `prais y x1 x2, corc` (Cochrane-Orcutt) | `oe.prais(..., method="corc")` |
| `prais ..., twostep rhotype(dw) vce(robust)` | `oe.prais(..., twostep=True, rhotype="dw", covariance="robust")` |
| EViews `ls y c x ar(1)` | `oe.prais(...)` (same model; EViews uses nonlinear least squares) |
| `ardl y x1 x2, lags(2 1 0)` | `oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[2, 1, 0])` |
| `ardl y x1 x2, maxlags(4) aic` / `bic`; EViews ARDL | `oe.ardl(..., maxlags=4, ic="aic")` / `ic="bic"` |
| `ardl y x1 x2, ec` + `estat ectest`; EViews Bounds Test | `oe.ardl(..., ec=True)`; `fit.tests["bounds_f"]`, `fit.tests["bounds_t"]` |
| `tsfilter hp c = y, smooth(1600) trend(t)`; EViews HP filter | `oe.tsfilter(df, "y", method="hp", smooth=1600)` |
| `tsfilter bk c = y, minperiod(6) maxperiod(32) smaorder(12)` | `oe.tsfilter(df, "y", method="bk", minperiod=6, maxperiod=32, smaorder=12)` |
| `tsfilter cf c = y, drift`; EViews CF filter | `oe.tsfilter(df, "y", method="cf", drift=True)` |
| Hamilton (2018) regression filter | `oe.tsfilter(df, "y", method="hamilton", hamilton_h=8, hamilton_p=4)` |
| `ucm y, model(llevel)` | `oe.ucm(data=df, y="y", model="llevel")` |
| `ucm y x, model(lltrend) seasonal(12) cycle(1)` | `oe.ucm(data=df, y="y", x=["x"], model="lltrend", seasonal=12, cycle=True)` |
| `predict ..., smethod(smooth)` after `ucm` | `oe.ucm_components(fit, df)` |
| `predict ..., dynamic()` after `ucm`; EViews sspace forecast | `oe.forecast(fit, steps=h)` (`oe.ucm_forecast`) |
| `mswitch dr y, switch(x) varswitch`; EViews Markov switching | `oe.mswitch(data=df, y="y", x=["x"], switch=["Intercept", "x"], varswitch=True)` |
| `mswitch ar y, ar(1/2) arswitch` | `oe.mswitch(data=df, y="y", ar=2, arswitch=True)` |
| `predict pr*, pr smethod(smooth)` after `mswitch` | `oe.mswitch_probabilities(fit, df)` |
| `threshold y x1, threshvar(q) regionvars(x2) nthresholds(2)`; EViews TAR | `oe.threshold(data=df, y="y", x=["x1", "x2"], threshold_var="q", regions=["Intercept", "x2"], nthresholds=2)` |

Everything is implemented in OpenEconometrics on float64 PyTorch tensors; no estimation
library runs at fit time. Every function validates its input and fails with an
`AnalysisError(code, message)` that says what to change.

Common rules for the whole family:

- **Time order.** Pass `time=<column>` (integer periods or datetimes); rows are
  sorted by it and repeated periods are an error. Without a time column the row
  order is the time order. Integer periods must be consecutive (`time_gaps`
  otherwise) and with `missing="drop"` only rows at the start or the end of the
  series may be dropped, because every lag and recursion needs consecutive
  periods. Datetime columns are taken as consecutive periods in sorted order.
- **Results** are `ResultBundle`s (`summary()`, `to_latex()`, JSON round trip).
  The constant is called `Intercept`. `provenance["stata_parity_validated"]` is
  `False`: the formulas follow Stata's Methods and formulas, but no comparison
  with Stata output has been recorded.

## `oe.prais`: Prais-Winsten and Cochrane-Orcutt regression

### Model and estimator

A linear regression whose errors follow a stationary AR(1) process,

    y_t = x_t'b + u_t,        u_t = rho u_(t-1) + e_t,        |rho| < 1.

For known rho, GLS is OLS on quasi-differenced data:

    y*_t = y_t - rho y_(t-1),   x*_t = x_t - rho x_(t-1)          t = 2..N
    y*_1 = sqrt(1 - rho^2) y_1, x*_1 = sqrt(1 - rho^2) x_1       (Prais-Winsten only)

Prais-Winsten (`method="prais"`, the default) keeps the transformed first
observation; Cochrane-Orcutt (`method="corc"`) drops it, so N* = N - 1. The
constant is transformed like every other regressor.

rho is estimated from the untransformed residuals `u_t = y_t - x_t'b` by the
formula chosen with `rhotype` (Stata's `rhotype()` option):

| `rhotype` | formula |
| --- | --- |
| `"regress"` (default) | `sum_(t>=2) u_t u_(t-1) / sum_(t>=2) u_(t-1)^2` (u_t on u_(t-1), no constant) |
| `"freg"` | `sum_(t>=2) u_t u_(t-1) / sum_(t>=2) u_t^2` (u_t on u_(t+1)) |
| `"tscorr"` | `sum_(t>=2) u_t u_(t-1) / sum_t u_t^2` |
| `"dw"` | `1 - DW/2` |
| `"theil"` | `rho_tscorr * (N - K) / N` |
| `"nagar"` | `(rho_dw N^2 + K^2) / (N^2 - K^2)` |

The iteration follows Stata: OLS (rho = 0) -> rho from the OLS residuals ->
transformed regression -> new rho from the untransformed residuals -> ...
until rho changes by less than `tolerance` (default 1e-6; at most
`max_iterations` = 100 rho updates). The coefficients are those of the
transformed regression at the final rho. `twostep=True` stops after the first
transformed regression (the two-step estimator).

### Covariance and inference

Inference is that of OLS on the transformed data, treating rho as known (as in
Stata), with Student t on N* - K degrees of freedom:

| `covariance` | formula |
| --- | --- |
| `"nonrobust"` (default) | `s^2 (X*'X*)^-1`, `s^2 = SSR*/(N* - K)` |
| `"robust"` / `"HC1"` | White sandwich on the transformed data times `N*/(N* - K)` (Stata's `vce(robust)`) |
| `"HC2"`, `"HC3"` | leverage-adjusted sandwiches (Stata's `vce(hc2)`, `vce(hc3)`) |

### Results

- coefficients `b`;
- `metrics`: `r_squared`, `adjusted_r_squared`, `rmse` (all of the transformed
  regression, as Stata reports them; the R-squared is centered at the mean of
  y* when the model has a constant and uncentered otherwise), `rho`,
  `durbin_watson_original` (OLS residuals), `durbin_watson_transformed`,
  `iterations` (number of rho estimates), `df_model`, `df_resid`, `ssr`;
- `tests["model"]`: with the default `nonrobust` covariance, Stata's ANOVA F of
  the transformed regression, `((TSS* - SSR*)/(K - 1)) / (SSR*/(N* - K))` with
  TSS* centered at the mean of y* (what `regress ..., hascons` reports); with a
  robust covariance the Wald F that all coefficients except the constant are
  zero. For Prais-Winsten the two differ because the transformed constant
  (`sqrt(1 - rho^2)`, `1 - rho`, ...) is not a constant column: in Stata's own
  example (`prais usr idle`) the reported F(1, 28) = 7.12 while the squared t
  statistic of `idle` is 8.25. Without a constant the test is the F of all
  coefficients (uncentered R-squared);
- `extra`: `rho_path` (0 for OLS, then every rho), `method`, `rhotype`,
  `twostep`, `converged`;
- predictions: the structural fit `x_t'b` against `y_t`.

Errors: `rho_out_of_range` (Prais-Winsten needs |rho| < 1; a rho outside the
interval signals nonstationary errors: use `method="corc"` or difference the
data), `nonconvergence`, `time_gaps`, `insufficient_observations`,
`singular_design` (regressors collinear after the transformation),
`constant_outcome`, `perfect_fit` (OLS reproduces y to rounding: SSR <= 1e-24
sum y^2, so rho is undefined).

### Example

```python
fit = oe.prais(data=df, y="usr", x=["idle"], time="t")            # prais usr idle
fit_co = oe.prais(data=df, y="usr", x=["idle"], time="t", method="corc")
print(fit.summary())
fit.metrics["rho"], fit.metrics["durbin_watson_transformed"]
```

### Notes and limitations

- Weights and clustered covariances are not offered.
- The nonrobust model F is the ANOVA F of the transformed regression (see
  Results); for Cochrane-Orcutt the transformed constant is a constant column
  and it equals the Wald F of the slopes.
- Non-convergence of the rho iteration is an error (`nonconvergence`), never a
  silently reported last iterate.
- EViews' `ls y c x ar(1)` estimates the same model by nonlinear least squares
  conditional on the first observation; the estimates are close to
  Cochrane-Orcutt but not identical.

## `oe.ardl`: autoregressive distributed lag models and the bounds test

### Model

    y_t = c_0 [+ c_1 t] + sum_(i=1..p) phi_i y_(t-i) + sum_j sum_(l=0..q_j) b_jl x_(j,t-l) + w_t'g + e_t

`x` are the k regressors that enter with lags; `exog=[...]` are regressors that
enter only contemporaneously and are not part of the long-run relation
(Stata's `exog()`). `trend` chooses the deterministic terms: `"constant"`
(default), `"trend"` (constant and linear trend t = 1, 2, ... counted from the
first row of the ordered series) or `"none"`. The model is estimated by OLS
(Householder QR) on the observations after the largest lag. `lags=[0, q_1,
...]` gives a finite distributed-lag model.

### Lag selection

With `lags=[p, q_1, ..., q_k]` the orders are fixed. Otherwise every
combination `p = 1..maxlags_y`, `q_j = 0..maxlags_j` (one `maxlags` value for
all variables, default 4, or k + 1 values) is fitted on the common sample that
holds back `max(maxlags)` observations, and the combination minimizing

    AIC = -2 ll + 2 K       (ic="aic", the default, as in EViews)
    BIC = -2 ll + K ln N    (ic="bic", the default of Kripfganz and Schneider's Stata ardl)

with `ll = -N/2 (1 + ln 2 pi + ln(SSR/N))` and K the number of coefficients is
reported, on that same sample (so the reported AIC/BIC are those of the
search). The search is exhaustive (the grid has `maxlags_y * prod(maxlags_j + 1)`
models; more than 50,000 give a warning, more than 2,000,000 are refused) but
cheap: the deterministic and exogenous columns are partialled out once, the
candidate lag columns are written in nested difference form (`y_(t-1)`,
`D.y_(t-1)`, ..., `x_t`, `D.x_t`, ...; each model uses a prefix of every block,
which spans the same space as its levels lags), reduced by one QR
factorization, and the SSR of every subset comes from a small batched QR. Five
hundred models on one million observations take about 1.4 seconds.
`extra["lag_selection"]` lists the criterion, the number of models, the
selected orders and the ten best models with their criterion values.

### Error-correction form

`ec=True` reports Stata's `ardl, ec` parameterization (long-run relation in
`x_t`):

    D.y_t = c - a (y_(t-1) - theta'x_t) + sum_(i=1..p-1) psi_i D.y_(t-i)
            + sum_j sum_(l=0..q_j-1) omega_jl D.x_(j,t-l) + w_t'g + e_t

| term | value |
| --- | --- |
| `ADJ:L.y` | `-a = sum phi_i - 1` (speed of adjustment) |
| `LR:x_j` | `theta_j = sum_l b_jl / a` (long-run coefficient) |
| `SR:LD.y`, `SR:L2D.y`, ... | `psi_i = -sum_(m>i) phi_m` |
| `SR:D.x`, `SR:LD.x`, ... | `omega_jl = -sum_(m>l) b_jm` |
| `SR:Intercept`, `SR:trend`, exog | unchanged |

In case II (`restricted=True` with a constant) the constant is part of the
long-run relation (`LR:Intercept = c/a`); in case IV the trend (`LR:trend`).
These coefficients are exact functions of the levels estimates; their
covariance is the delta method (exact for the linear short-run terms, first
order for theta), which reproduces OLS on the EC regression. The long-run
coefficients with delta-method standard errors are also in
`extra["long_run"]` for the levels form. `ec=True` needs p >= 1.

### Bounds test (Pesaran, Shin and Smith 2001)

The PSS case follows from `trend` and `restricted`:

| `trend` | `restricted` | case |
| --- | --- | --- |
| `"none"` | - | I: no intercept, no trend |
| `"constant"` | `True` | II: restricted intercept |
| `"constant"` | `False` | III: unrestricted intercept (default) |
| `"trend"` | `True` | IV: unrestricted intercept, restricted trend |
| `"trend"` | `False` | V: unrestricted intercept and trend |

`tests["bounds_f"]` is the F statistic that the level coefficients of the
conditional EC model are zero (`a = 0` and `a theta = 0`, plus the restricted
constant or trend in cases II and IV; df = k + 1 or k + 2, df2 = N - K);
`tests["bounds_t"]` (cases I, III and V) the t statistic of `a`. Both use the
classical OLS covariance, as in PSS. Each test carries the asymptotic I(0) and
I(1) critical-value bounds at 10%, 5% and 1% for its case and k (PSS Tables
CI(i)-(v) and CII; k = 0..10), and a decision per level: reject "no level
relationship" when the statistic is beyond the I(1) bound (above it for F,
below it for t), do not reject when it is within the I(0) bound, inconclusive
in between. `p_value` is `None`: the tables give critical values only.

### Covariance and results

`covariance`: `"nonrobust"` (default), `"robust"`/`"HC1"`, `"HC2"`, `"HC3"`
(the coefficient table and the long-run coefficients). t inference on N - K
degrees of freedom.

- `metrics`: `r_squared` (of y, or of D.y in EC form), `adjusted_r_squared`,
  `rmse`, `log_likelihood`, `aic`, `bic`, `hqic`, `durbin_watson`, `df_model`,
  `df_resid`, `ssr`;
- `tests`: `model` (F that every reported coefficient except the constant is
  zero: in levels the slopes; in EC form the EC coefficients, i.e. the null
  `D.y_t = c + e_t`, which in levels is phi_1 = 1 and every other non-constant
  coefficient 0 - the Wald F is invariant to the reparameterization and with the
  nonrobust covariance equals the ANOVA F of the reported R-squared of D.y;
  without a constant all coefficients are tested), `bounds_f`, `bounds_t`;
- `extra`: `ardl_order` (`"ARDL(2,1,0)"`), `lags`, `lag_selection`, `case`,
  `speed_of_adjustment`, `long_run`, `levels_coefficients` (EC form).

Errors: `invalid_lags`, `invalid_option` (`restricted=True` with
`trend="none"`, `ec=True` with p = 0), `lag_grid_too_large`,
`insufficient_observations`, `collinear_lags` (a lag column is collinear: the
ARDL structure needs every lag), `constant_outcome`, `perfect_fit`,
`time_gaps`, `invalid_spec` (exog repeating y or x).

### Example

```python
fit = oe.ardl(data=df, y="lnc", x=["lny", "lnw"], time="quarter", maxlags=4, ic="bic", ec=True)
print(fit.summary())
fit.extra["ardl_order"]                      # e.g. "ARDL(2,0,1)"
fit.tests["bounds_f"]["decision"]["5%"]      # "reject H0 (no level relationship)"
```

### Limitations and conventions

- The Narayan (2005) small-sample bounds and Kripfganz-Schneider (2020)
  response-surface p-values are not implemented; only the PSS asymptotic
  bounds at 10/5/1% (the 2.5% column is omitted).
- The lag search keeps p >= 1 (pure distributed-lag models are fitted only with
  explicit `lags`), and reports the selected model on the common search
  sample. Refit with `lags=` to use the observations that larger maxlags held
  back.
- Lags are built on consecutive periods: interior gaps are refused. HAC
  covariance is not offered (use `oe.newey` on hand-built lags).

## `oe.tsfilter`: Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald and Hamilton filters

`oe.tsfilter(data, y, method=..., time=None, ...)` splits one series into a
trend and a cycle, `y_t = trend_t + cycle_t`, and returns a table with the
columns `period` (time value, or row number 0..T-1), `observed`, `trend` and
`cycle`; periods where a filter is undefined hold NaN (Stata's missing). The
series must be complete and regularly spaced. `attrs` records the method and
its settings, `lost_start`/`lost_end` and, for the Hamilton filter, the
regression coefficients.

| method | definition | settings (defaults for quarterly data) |
| --- | --- | --- |
| `"hp"` | trend minimizes `sum (y_t - tau_t)^2 + lambda sum (tau_(t+1) - 2 tau_t + tau_(t-1))^2`, i.e. `(I + lambda K'K) tau = y` | `smooth` = lambda = 1600 (6.25 or 100 annual, 129600 monthly) |
| `"bk"` | symmetric MA of order K with weights `b_0 = (w2 - w1)/pi`, `b_j = (sin(j w2) - sin(j w1))/(pi j)`, `w1 = 2 pi/maxperiod`, `w2 = 2 pi/minperiod`, shifted to sum to zero; first and last K cycle values missing | `minperiod=6`, `maxperiod=32`, `smaorder=12` |
| `"cf"` | Christiano-Fitzgerald asymmetric full-sample filter for a random walk (every observation uses the whole sample, end weights make each row sum to zero); `drift=True` first removes the line through the first and last observations | `minperiod=6`, `maxperiod=32`, `drift=True` |
| `"hamilton"` | OLS of `y_(t+h)` on a constant and `y_t, ..., y_(t-p+1)`; cycle = residual, trend = fitted value (dated t + h); first h + p - 1 periods missing | `hamilton_h=8`, `hamilton_p=4` (24 and 12 monthly) |

Implementation: the HP system is pentadiagonal and solved exactly by a banded
LDL' factorization in O(T) (a loop over periods on plain floats, about one
second for a million observations; no T x T matrix); the BK cycle is a
strided moving average; the CF interior sums are one FFT convolution
(O(T log T)) plus closed-form end weights from cumulative sums; the Hamilton
regression uses QR least squares. Results agree with statsmodels' `hpfilter`,
`bkfilter` and `cffilter` to rounding.

```python
parts = oe.tsfilter(df, "lgdp", method="hp", time="quarter")        # tsfilter hp c = lgdp, trend(t)
hamilton = oe.tsfilter(df, "lgdp", method="hamilton", time="quarter")
hamilton.attrs["coefficients"]
```

Notes: Stata's `tsfilter bw` (Butterworth) and the symmetric `stationary`
variant of the CF filter are not implemented. Stata's `tsfilter cf` removes
drift only with its `drift` option; OpenEconometrics follows the requested default
`drift=True` (statsmodels' default) - pass `drift=False` for Stata's default.
Stata chooses the default HP `smooth()` from the tsset periodicity; OpenEconometrics
uses 1600 unless told otherwise.

## `oe.ucm`: unobserved-components (structural time-series) models

### Model

    y_t = mu_t + gamma_t + psi_t + x_t'b + e_t,                    e_t ~ N(0, var(e))
    mu_(t+1)    = mu_t + beta_t + eta_t                            eta_t ~ N(0, var(level))
    beta_(t+1)  = beta_t + zeta_t                                  zeta_t ~ N(0, var(slope))
    gamma_(t+1) = -(gamma_t + ... + gamma_(t-s+2)) + omega_t       omega_t ~ N(0, var(seasonal))
    [psi, psi*]_(t+1)' = rho [cos l, sin l; -sin l, cos l] [psi, psi*]_t' + kappa_t,   var(cycle)

`model` chooses the trend with Stata's names:

| `model` | level | slope | irregular e_t |
| --- | --- | --- | --- |
| `"rwalk"` (default) | random walk | - | no |
| `"llevel"` | random walk | - | yes |
| `"lltrend"` | random walk | random walk | yes |
| `"strend"` (`"smooth_trend"`) | integrated (no own disturbance) | random walk | yes |
| `"rtrend"` | integrated | random walk | no |
| `"rwdrift"` | random walk | constant drift | no |
| `"lldtrend"` | random walk | constant drift | yes |
| `"dtrend"` | deterministic | constant | yes |
| `"dconstant"` | constant | - | yes |
| `"ntrend"` | - | - | yes |
| `"none"` | - | - | no (needs a seasonal or cycle) |

`seasonal=s` adds a stochastic dummy seasonal of period s; `cycle=True` a
stochastic damped cycle with frequency l in (0, pi) (period 2 pi / l),
damping rho in (0, 1) and disturbance variance var(cycle). `x` are regressors
(no constant: the level plays that role).

### Estimator

Exact Gaussian maximum likelihood. The level, slope and seasonal states are
diffuse and handled by the exact diffuse Kalman filter of Durbin and Koopman
(2012, sections 5.2 and 7.2); the cycle is stationary and starts from its
unconditional distribution `var(cycle) / (1 - rho^2) I`. The log likelihood is
the exact diffuse one (DK eq. 7.4):

    ll = -1/2 sum_t [ log 2 pi + log F_inf,t ]                       (diffuse periods with F_inf > 0)
         -1/2 sum_t [ log 2 pi + log F_t + v_t^2 / F_t ]             (all other periods)

Implementation details that matter for speed and accuracy:

- The filter is batched over parameter vectors and data columns: the
  regression coefficients b are concentrated out by GLS from the innovations of
  y and of every regressor (the gains do not depend on the data), and all the
  perturbed parameter vectors of a numerical derivative are filtered in one
  pass.
- Deterministic states (a state no disturbance reaches, such as the drift of
  `rwdrift` or a seasonal whose variance is zero) are moved into the GLS part
  as diffuse regressors `W_t = (Z T^(t-1))` restricted to those states. The
  likelihood is then de Jong's diffuse likelihood of the reduced model with
  `-1/2 log det(S_WW)` added, which equals the exact diffuse likelihood
  (checked to rounding in the tests).
- Once the filter covariance stops changing (relative change below 1e-14) the
  remaining innovations follow a time-invariant recursion and are computed
  without a time loop: `v_(t0+s) = y_(t0+s) - (Z L^s) a_t0 - sum_(i<s) (Z L^(s-1-i) K)
  y_(t0+i)` with the impulse responses by repeated squaring and the sum by FFT
  convolution (the smoother uses the same responses backwards).

Variances are optimized as logs (relative to the variance of the differenced
series), the frequency and damping through logits, by BFGS with
central-difference gradients and Hessian of the exact log likelihood
(no analytic Kalman-derivative recursions; recorded in `optimizer`). A
variance that converges to its zero boundary is fixed at zero, the model is
re-maximized, a warning is recorded and the variance is listed in
`extra["fixed_zero"]` (it is not a free parameter and has no row in the
coefficient table). "Converges to zero" means below 1e-9 of the scale, or
below 1e-6 of the scale when BFGS stopped without certifying a maximum - the
log-variance direction is flat at the boundary, so the Hessian there is
singular and BFGS can stall just above 1e-9 (this made `y * 1e8` fail before
the verification pass); the restricted maximum must then not be lower than the
point where the search stopped, otherwise `nonconvergence` is raised. The
exact diffuse likelihood is not scale equivariant in the usual way: rescaling
y by c changes it by `-(N - d) ln c` (d diffuse periods), because the diffuse
periods contribute `log F_inf`, which does not depend on the data. A cycle that collapses into an undamped
deterministic sinusoid (damping -> 1 or zero variance; usually a seasonal
pattern) raises `degenerate_cycle`.

### Covariance

| `covariance` | definition |
| --- | --- |
| `"nonrobust"` (default; Stata `vce(oim)`) | inverse of the numerical Hessian of the full log likelihood in (log variances, logits, b) |
| `"robust"` (Stata `vce(robust)`) | Huber-White sandwich with N/(N-1); per-period scores numerical for the variance parameters, analytic for b |
| `"opg"` | inverse outer product of the per-period scores |

The variances, frequency and damping and their covariance are reported in
natural units by the delta method. z tests; as in Stata the tests of variances
are one-sided (halved p-value) and their intervals are truncated at zero.

### Results

- coefficients: regressors, then `/frequency`, `/damping`, `/var(level)`,
  `/var(slope)`, `/var(seasonal)`, `/var(cycle)`, `/var(e)` (those the model
  estimates);
- `metrics`: `log_likelihood`, `aic`, `bic` (k = estimated parameters, N =
  observations), `diffuse_periods`, `iterations`;
- `tests`: Ljung-Box and Jarque-Bera tests of the standardized one-step
  residuals `v_t / sqrt(F_t)`;
- `extra`: `variances` (zero-fixed ones included), `fixed_zero`,
  `cycle_parameters` (frequency, damping, period), `last_state` (filtered state
  and covariance after the last observation, for forecasts), `state_names`;
- predictions: the smoothed signal `Z alpha^_t + x_t'b` against y.

`oe.ucm_components(fit, data)` returns the smoothed components (exact diffuse
fixed-interval smoother): `period`, `observed`, `level`, `slope`, `seasonal`,
`cycle` (those present), `regression`, `fitted`, `irregular`.
`oe.forecast(fit, steps, exog=...)` (also `oe.ucm_forecast`) propagates the
filtered state: `yhat = Z T^(h-1) a_(n+1) + x'b`, mean squared error
`Z P Z' + var(e)` with `P <- T P T' + Q` (parameter uncertainty ignored);
`data=` restarts the filter on another sample.

### Example

```python
fit = oe.ucm(data=df, y="lunemp", model="llevel", cycle=True, time="quarter")
print(fit.summary())
fit.extra["cycle_parameters"]["period"]
parts = oe.ucm_components(fit, df)          # smoothed level and cycle
oe.forecast(fit, steps=8)
```

### Notes, limitations and conventions

- Missing values inside the series are refused (Stata's ucm can skip them in
  the filter).
- Only one first-order cycle (Stata allows up to three cycles of order 1-3).
- The log likelihood is the exact diffuse one used by statsmodels
  (`use_exact_diffuse=True`); Stata's De Jong diffuse filter gives the same
  maximizer, but the reported constant of Stata's log likelihood has not been
  compared. statsmodels initializes a stochastic cycle as diffuse, OpenEconometrics (like
  the Stata documentation of ucm) uses its stationary distribution.
- `var(cycle)` is the variance of the cycle disturbance kappa_t (the cycle's
  own variance is `var(cycle) / (1 - rho^2)`).
- Speed: each likelihood evaluation is O(N). Once the filter reaches its
  steady state the rest of the sample is vectorized, so a local level on
  100,000 observations takes well under a second. While a variance is heading
  to zero during the search the filter covariance converges too slowly to reach
  the steady state and every evaluation loops over all periods (about 25
  microseconds per period): a local level with a seasonal whose variance goes
  to zero takes 0.4 s on 1,000, 12 s on 10,000 and about 40 s on 100,000
  observations. See Performance.

## `oe.mswitch`: Markov-switching dynamic regression and autoregression

### Model

    y_t = mu_t(s_t) + sum_(i in ar) phi_i (y_(t-i) - mu_(t-i)(s_(t-i))) + e_t,   e_t ~ N(0, sigma_(s_t)^2)
    mu_t(s) = x_t'beta_s + w_t'alpha
    P(s_t = b | s_(t-1) = a) = p_ab                         (constant transition probabilities)

The terms named in `switch` (default `["Intercept"]`) have state-specific
coefficients beta_s, every other term a common coefficient alpha.
`varswitch=True` gives state-specific variances. Without `ar` this is Stata's
dynamic-regression model (`mswitch dr`); `ar=p` (lags 1..p) or `ar=[1, 4]`
gives Stata's `mswitch ar` (Hamilton 1989): the autoregression acts on the
deviations from the state-dependent means, so the density of y_t depends on
(s_t, s_(t-1), ..., s_(t-P)); `arswitch=True` makes phi state-specific.

### Estimator

Exact maximum likelihood through the Hamilton filter. The chain starts from
its ergodic distribution (Stata's default `p0(transition)`); autoregressions
condition on the first P observations (the estimation sample starts at P + 1).
With lags the filter runs on the K = k^(P+1) "expanded" states. The filter
has no per-period Python loop: periods are cut into blocks of about sqrt(N);
the products of the per-period matrices `diag(eta_t) A'` are formed for all
blocks at once, a short loop over blocks gives the exact filtered
probabilities at every block start, and all blocks are then filtered in
parallel (the Kim smoother runs the same scheme backwards). Densities are
scaled by their per-period maximum, so nothing underflows.

The gradient is analytic (Fisher's identity: the expected complete-data score
given the data, with the smoothed state and transition probabilities and the
derivative of the ergodic distribution `d pi' = pi' dP (I - P + 1 pi')^-1`);
BFGS maximizes from three starting points (OLS with the constant split by
residual quantiles, persistence 0.9, 0.7, 0.95) and keeps the highest maximum.
The states are then relabelled in ascending order of the first switching
coefficient (the constant by default).

The likelihood is maximized in standardized units - y divided by its standard
deviation s_y, each regressor by its root mean square s_j - and mapped back
exactly (`beta_j = s_y / s_j * beta*_j`, `lnsigma = lnsigma* + ln s_y`, AR
coefficients and logits unchanged; the covariance by the same diagonal map, the
log likelihood by `- N ln s_y`). BFGS judges convergence partly by absolute
gradient sizes, so without this an outcome measured in tiny or huge units
(y * 1e-8) did not converge; the scales are recorded in
`extra["estimation_units"]`.

Parameterization: transition probabilities by multinomial logits against the
last state, `lgt(p_ac) = log(p_ac / p_ak)` (for two states `lgt(p11) =
logit(p11)`, `lgt(p21) = logit(p21)`, the values Stata's `p11`, `p21` are
computed from), standard deviations as `lnsigma`.

### Covariance

| `covariance` | definition |
| --- | --- |
| `"nonrobust"` (default; Stata `vce(oim)`) | inverse of the Hessian, the Ridders-extrapolated numerical derivative of the analytic gradient |
| `"robust"` (Stata `vce(robust)`) | Huber-White sandwich with N/(N-1); per-period scores of log f(y_t given the past) by central differences |
| `"opg"` | inverse outer product of the same scores |

z tests. Transition probabilities, expected durations and sigmas with
delta-method standard errors are in `extra`.

### Results

- coefficients: `state1:Intercept`, `state1:x`, ..., `state2:...`, common terms,
  `L1.ar` (or `state1:L1.ar`), `/lnsigma` (or `/lnsigma1`, ...), `/lgt(p11)`,
  `/lgt(p21)`, ...;
- `metrics`: `log_likelihood`, `aic`, `bic` (k = all parameters, N = observations
  used), `states`, `duration_state1..k` = 1/(1 - p_jj);
- `extra`: `transition_matrix` (p_ab with standard errors), `ergodic_probabilities`,
  `sigma` (per state, with standard errors), `starts` (log likelihood reached from
  each start), `switching_terms`, `ar_lags`;
- predictions: one-step predictions E[y_t | y_1..y_(t-1)].

`oe.mswitch_probabilities(fit, data)` returns `period`, `observed`,
`filtered_state1..k`, `smoothed_state1..k` (Kim smoother) and
`predicted_state1..k`.

### Example

```python
fit = oe.mswitch(data=df, y="fedfunds", time="quarter", states=2, varswitch=True)
print(fit.summary())
fit.extra["transition_matrix"]           # [[{"probability": 0.98, "std_error": ...}, ...], ...]
probs = oe.mswitch_probabilities(fit, df)
```

### Notes

- statsmodels' `MarkovAutoregression` (0.14) indexes a switching variance by
  the regime of t-1 instead of t when the AR order is 2 or more; OpenEconometrics uses
  sigma_(s_t) as Stata documents (the tests compare with statsmodels only where
  its formula is correct).
- Time-varying transition probabilities and Stata's `p0(fixed|smoothed)` are not
  implemented. At most 512 expanded states (k^(P+1)).
- Markov-switching likelihoods can have several local maxima; check
  `extra["starts"]` and consider more states/terms only with long samples.

## `oe.threshold`: threshold regression

### Model and estimator

    y_t = w_t'a + x_t'b_r + e_t      when gamma_(r-1) < q_t <= gamma_r,   r = 1..m+1

(region 1: q <= gamma_1; the last region: q > gamma_m). `regions` lists the
terms with region-specific coefficients (default: all terms, including the
constant); the others are common. The thresholds minimize the sum of squared
residuals over the observed values of the threshold variable q, keeping at
least `trim` * N observations (and more than the number of region-specific
terms) in every region (`trim=0.1` is Stata's `trim(10)`).

The search is exhaustive and vectorized: the data are sorted by q once; for a
candidate split the regression's cross-product matrix is built from prefix
sums of x x', w x' and x y (computed chunk by chunk, so memory stays bounded)
and the SSR follows from batched Cholesky solves; the five best candidates are
refitted by QR and the smallest exact SSR wins. Several thresholds are found
sequentially (each new threshold given those already found), followed by Bai's
(1997) refinement: every threshold is re-estimated given the others until none
changes. A single threshold on one million observations takes about 0.4 s.

Given the thresholds the coefficients are OLS, with the thresholds treated as
known (Hansen 2000: their estimation does not affect the slope inference
asymptotically). `covariance`: `"nonrobust"` (default), `"robust"`/`"HC1"`,
`"HC2"`, `"HC3"`; t tests on N - K degrees of freedom.

### Results

- coefficients: common terms, then `region1:<term>`, `region2:<term>`, ...;
- `metrics`: `r_squared`, `adjusted_r_squared`, `rmse`, `ssr`, `aic`, `bic`, `hqic`
  (`N ln(SSR/N) + 2K`, `+ K ln N`, `+ 2K ln ln N`), `threshold1`, ..., `df_model`,
  `df_resid`;
- `tests`: `model` (F that all coefficients except the constants are zero);
  with `bootstrap=R`, `threshold_effect`: Hansen's (1996, 2000) fixed-regressor
  bootstrap test of no threshold, `F = N (S_0 - S_1) / S_1` with S_0 the linear
  model's SSR and S_1 that of the best ONE-threshold model (also when
  `nthresholds` > 1; before the verification pass the lower of the refined
  thresholds was used), `y*_t = e^_t u_t`, u_t ~ N(0, 1), e^ the residuals of
  that one-threshold model, the same grid in every replication and seed `seed`;
- `extra`: `thresholds`, `region_sizes`, `regions`, `common`, `sequence` (the
  thresholds and SSR after each step), `by_number_of_thresholds` (SSR, AIC, BIC,
  HQIC for 0..m thresholds, for choosing m), `ssr_path` (the SSR over at most 200
  candidates of the first search), and for one threshold
  `threshold_confidence_set`: Hansen's LR inversion `{gamma: N (S(gamma) -
  S(gamma^)) / S(gamma^) <= -2 ln(1 - sqrt(1 - alpha))}` (homoskedastic
  asymptotic distribution; 7.35 at 95%), reported as the hull of the accepted
  candidates.

### Example

```python
fit = oe.threshold(data=df, y="growth", x=["inflation", "invest"], threshold_var="debt",
                   regions=["Intercept", "inflation"], nthresholds=1, bootstrap=500)
fit.extra["thresholds"], fit.extra["threshold_confidence_set"], fit.tests["threshold_effect"]
```

A self-exciting threshold autoregression (SETAR) is a threshold regression of
y_t on its own lags with a lagged value as the threshold variable: create the
lag columns and pass e.g. `x=["y_l1", "y_l2"], threshold_var="y_l1"`.

Errors: `constant_outcome`, `constant_threshold_variable`, `perfect_fit`,
`insufficient_observations` (fewer than (m + 1) regions of the minimum size),
`no_admissible_threshold`, `invalid_option` (`trim` outside (0, 0.5), unknown
`regions` terms).

### Notes and conventions

- Stata's `threshold` output may label the coefficient tests z rather than t;
  OpenEconometrics uses t on N - K degrees of freedom as requested (not verified).
- The sequential estimation with Bai's refinement is the textbook procedure;
  whether Stata refines after the sequential steps has not been verified.
- The information criteria count the regression coefficients K (not the
  thresholds).
- The LR confidence set assumes homoskedastic errors; Hansen's
  heteroskedasticity-corrected version is not implemented.

## Performance

Measured on synthetic data during the verification pass (Apple silicon, one
CPU thread, float64; wall time of the whole call including sample handling):

| model | size | time |
| --- | --- | --- |
| `oe.prais` (3 regressors, iterated Prais-Winsten / Cochrane-Orcutt) | 1,000,000 obs | 0.9 / 0.2 s |
| `oe.ardl` fixed lags ARDL(2,1,1) | 1,000,000 obs | 0.2 s |
| `oe.ardl` lag search, 3 regressors, maxlags 4 (500 models), EC form | 1,000,000 obs | 0.8 s |
| `oe.tsfilter` hp / bk / cf / hamilton | 1,000,000 obs | 1.4 / 0.5 / 0.5 / 0.4 s |
| `oe.threshold` one / two thresholds, 3 terms | 1,000,000 obs | 0.5 / 1.6 s |
| `oe.ucm` local level | 100,000 obs | 0.2-0.8 s |
| `oe.ucm` local linear trend + regressor (slope variance -> 0) | 1,000 / 10,000 / 100,000 obs | 1.8 / 5.5 / 45 s |
| `oe.ucm` local level + quarterly seasonal (seasonal variance -> 0) | 1,000 / 10,000 / 100,000 obs | 0.4 / 12 / 41 s |
| `oe.mswitch` 2 states, switching constant, 1 common regressor | 100,000 obs | 3.2 s |
| `oe.mswitch` 2 states, AR(1) (4 expanded states) | 100,000 obs | 4.7 s |

Every estimator is linear in N. The regressions and filters are vectorized
(the HP banded factorization loops over periods on plain floats). The UCM
filter is a recursion over periods that switches to a loop-free steady-state
computation once the filter covariance settles; when a variance converges to
zero that never happens within the sample, which is the slow case above. The
Markov-switching filter and smoother loop over sqrt(N)-sized blocks only.

## Verification

Besides the implementation tests (`tests/test_econ_tsmodels*.py`), the
verification pass added independent oracles and adversarial tests:

- `tests/test_econ_tsmodels_oracle.py`: Prais-Winsten / Cochrane-Orcutt as GLS
  with the explicit inverse AR(1) correlation matrix and the rho fixed point
  found by a scalar root finder for every `rhotype`; Stata's ANOVA F; White
  covariances on the transformed data; permutation, rescaling, categorical and
  collinearity invariances. ARDL EC form as OLS of the EC regression (nonrobust,
  HC1, HC3), the bounds F from restricted and unrestricted SSR in all five PSS
  cases, distributed lags, datetime time columns and a brute-force lag search
  with trend and exog. The HP trend against a 40-digit solution of the normal
  equations; BK and CF against statsmodels; Hamilton against least squares.
  Threshold regression against a global brute-force search over all splits
  (and pairs of splits). The local level against a sequential NumPy exact
  diffuse filter, a dense GLS smoother and SciPy maximization with numerical
  Hessians; Markov switching against a sequential NumPy Hamilton filter and Kim
  smoother maximized by SciPy, for 2 and 3 states and Hamilton's AR(1).
- `tests/test_econ_tsmodels_adversarial.py`: empty, one-row and n <= k samples,
  constant outcomes, missing and non-numeric columns, infinite values,
  repeated, fractional and gapped time values, perfect fits, scale equivariance
  (1e-8 and 1e8), large offsets, option bounds and undeclared weights or
  covariances - each works or raises `AnalysisError` with a specific code.

## Conventions that could not be verified against Stata output

- `oe.prais`: the nonrobust model F is the ANOVA F of the transformed regression.
  This is inferred from the example output printed in Stata's [TS] prais manual
  entry (F differs from the squared t of the only slope), not from a Stata run.
  The R-squared is centered at the mean of y* when there is a constant. In the
  `theil` and `nagar` formulas N is the number of observations of the
  untransformed series (also for Cochrane-Orcutt) and K the number of
  coefficients including the constant.
- `oe.ardl`: the default criterion is AIC (EViews; Kripfganz-Schneider's Stata
  `ardl` defaults to BIC); the lag search keeps p >= 1 and reports the selected
  model on the common search sample; the bounds tests use the classical
  covariance whatever `covariance` is. In EC form the model F tests the EC
  coefficients. The PSS critical values were typed in from the published
  tables and cross-checked against an independent transcription and a large
  simulation; the 2.5% column, Narayan's small-sample values and
  response-surface p-values are not provided.
- `oe.tsfilter`: `drift=True` is the default of the CF filter (statsmodels;
  Stata removes drift only on request); `smaorder` (default 12, Stata's name and
  default for quarterly data) is the BK truncation order - the requested
  signature called it `lead` with default 24. The output column is `period`.
- `oe.ucm`: the log likelihood is the exact diffuse one (Durbin-Koopman; the
  constant of Stata's De Jong diffuse likelihood has not been compared);
  `var(cycle)` is the disturbance variance; zero-boundary variances are fixed at
  zero and dropped from the table.
- `oe.mswitch`: states are relabelled by their first switching coefficient;
  transition probabilities and sigmas are reported through log-odds and
  `lnsigma` in the coefficient table (Stata displays p11, p21 and sigma), with
  the natural values and delta-method standard errors in `extra`.
- `oe.threshold`: sequential estimation with Bai's refinement; information
  criteria count the regression coefficients only; t (not z) inference.

## Not implemented

Stata's `tsfilter bw` (Butterworth) and the stationary CF variant; Narayan
(2005) small-sample bounds; UCM cycles of order 2-3, several cycles and missing
values inside the series; time-varying transition probabilities in `mswitch`;
Hansen's heteroskedasticity-robust threshold confidence set.
