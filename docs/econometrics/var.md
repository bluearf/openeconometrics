# Vector autoregressions and cointegration

This family covers the multivariate time-series toolbox of Stata's `var` and
`vec` suites (EViews: VAR and VEC objects):

| Function | What it does |
| --- | --- |
| `oe.var` | VAR(p) by OLS equation by equation, with every post-estimation statistic of Stata's `var` suite stored in the result |
| `oe.varsoc` | lag-order selection table (pre-estimation) |
| `oe.irf` | impulse responses (simple, orthogonalized, generalized) and the forecast-error variance decomposition, with standard errors |
| `oe.vecrank` | Johansen trace and maximum-eigenvalue tests for the cointegrating rank |
| `oe.vec` | vector error-correction model by Johansen's maximum likelihood |
| `oe.forecast` (`oe.var_forecast`, `oe.vec_forecast`) | dynamic forecasts with standard errors |

Everything is implemented in OpenEconometrics on float64 PyTorch tensors; no estimation
library runs at fit time. A VAR is one Householder QR of the lagged design, the
Johansen procedure one QR plus two small Cholesky factorizations and an SVD.
Nothing loops over observations.

```python
import openecon as oe

oe.varsoc(data=df, y=["income", "consumption"], time="quarter", maxlag=4)   # varsoc ..., maxlag(4)
fit = oe.var(data=df, y=["income", "consumption"], time="quarter", lags=2)  # var income consumption, lags(1/2)
print(fit.summary())
oe.irf(fit, steps=8, kind="orthogonalized")                                 # irf create ... ; irf table oirf fevd
oe.forecast(fit, 8)                                                         # fcast compute f_, step(8)

oe.vecrank(data=df, y=["y1", "y2", "y3"], lags=2)                           # vecrank y1 y2 y3, lags(2) max ic levela
vec = oe.vec(data=df, y=["y1", "y2", "y3"], lags=2, rank=1)                 # vec y1 y2 y3, lags(2) rank(1)
```

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `var y1 y2, lags(1/2)` | `oe.var(data=df, y=["y1", "y2"], lags=2)` |
| `var y1 y2, lags(1/2) exog(x)` | `..., x=["x"]` |
| `var ..., noconstant` | `..., constant=False` |
| `var ..., dfk` / `small` | `..., dfk=True` / `small=True` |
| `var ..., vce(robust)`; EViews White covariance | `..., covariance="robust"` |
| `varsoc y1 y2, maxlag(4)`; EViews Lag Length Criteria | `oe.varsoc(data=df, y=[...], maxlag=4)` or `fit.extra["lag_order_selection"]` |
| `varstable`; EViews AR Roots | `fit.extra["stability"]` |
| `vargranger`; EViews Granger Causality/Block Exogeneity | `fit.extra["granger"]`, `fit.tests["granger_all_<eq>"]` |
| `varwle`; EViews Lag Exclusion | `fit.extra["lag_exclusion"]`, `fit.tests["lag_exclusion_L<j>"]` |
| `varnorm`; EViews Normality Test | `fit.extra["normality"]`, `fit.tests["normality"]` |
| `varlmar, mlag(2)`; EViews Autocorrelation LM Test | `fit.tests["lm_autocorrelation_L<s>"]` (`lm_lags=`) |
| `irf create` + `irf table irf / oirf / fevd`; EViews Impulse Responses, Variance Decomposition | `oe.irf(fit, steps=8, kind=...)`, `fit.extra["irf"]` |
| EViews Generalized Impulses | `oe.irf(fit, kind="generalized")` |
| `fcast compute` | `oe.forecast(fit, steps)` |
| `vecrank y1 y2 y3, lags(2) trend(constant) max ic levela`; EViews Johansen Cointegration Test | `oe.vecrank(data=df, y=[...], lags=2, trend="constant")` |
| `vec y1 y2 y3, lags(2) rank(1) trend(rconstant)`; EViews VEC | `oe.vec(data=df, y=[...], lags=2, rank=1, trend="rconstant")` |
| `vecstable`, `vecnorm`, `veclmar` | `vec.extra["stability"]`, `vec.tests["normality"]`, `vec.tests["lm_autocorrelation_L<s>"]` |
| EViews VAR (OLS standard errors, t statistics) | `oe.var(..., small=True, dfk=True)` |

## Variables, time order and missing values

`y` is the list of endogenous variables. **Their order matters**: it is the
Cholesky order of orthogonalized impulse responses and the order of Johansen's
normalization. `y` must therefore be an ordered collection (a list or tuple);
a set or a dict is refused (`invalid_spec`), because its order is not part of
what the caller wrote. In a `ModelSpec` the first variable is `outcome` and the
whole list is the column role `system`:

```python
oe.ModelSpec(estimator="var", outcome="income",
             columns={"system": ["income", "consumption"]}, options={"lags": 2})
```

With a `time` column the rows are sorted by it. Integer periods must be
consecutive (a skipped period is the error `time_gaps`), repeated values are
`repeated_time_values`, datetimes are taken as consecutive in sorted order
(a warning records that gaps cannot be detected). Without a time column the row
order is the time order.

`missing="raise"` (default) refuses missing values. `missing="drop"` excludes
incomplete rows, but only at the beginning or the end of the series: a hole in
the middle would silently change every lag, so it is the error `time_gaps`.
The first `lags` observations supply the lagged values; the result records
that the estimation sample has `T = n - lags` observations
(`sample_positions`, a warning line, `provenance["sample"]`).

## `oe.var`: vector autoregression

### The model

    y_t = A_1 y_(t-1) + ... + A_p y_(t-p) + B x_t + v + d t + u_t,      E[u_t u_t'] = Sigma

for K endogenous variables `y`, optional exogenous regressors `x` (Stata's
`exog()`; `categorical=` expands them into indicators), a constant
(`constant=True`) and an optional linear trend (`trend=True`, t = 1, 2, ...
counted from the first observation of the ordered sample).

### Arguments

| Argument | Meaning |
| --- | --- |
| `y` | endogenous variables (list; one variable gives an AR(p) by OLS) |
| `x`, `categorical` | exogenous regressors; collinear ones are omitted with a warning |
| `time` | time column (optional) |
| `lags` | p; lags 1..p enter every equation (default 2) |
| `constant`, `trend` | deterministic terms |
| `small` | t and F statistics with `T - m` degrees of freedom; standard errors use the equation's degrees of freedom |
| `dfk` | `Sigma` estimated with the divisor `T - m` instead of `T` |
| `covariance` | `"nonrobust"` (default) or `"robust"` |
| `maxlag` | highest lag of the stored lag-order table (default: `lags`) |
| `irf_steps`, `irf_kinds` | horizon (8) and kinds of the stored impulse responses |
| `lm_lags` | lags of the residual autocorrelation LM test (2) |
| `missing`, `alpha` | missing-data policy; level of intervals and of the LR lag rule |

### Estimator

Every equation has the same m regressors, so OLS equation by equation is the
Gaussian ML estimator (and equals SUR/GLS). With `Z` the `[T, m]` design,

    B^ = (Z'Z)^-1 Z'Y  (one QR, X'X is never inverted),   U^ = Y - Z B^',
    Sigma_ml = U^'U^ / T,       ln L = -(T/2) { ln|Sigma_ml| + K ln(2 pi) + K }.

When the model has a constant, the regression runs on mean-deviated regressors
and outcomes and the constant and `(Z'Z)^-1` are mapped back exactly. This is
the same estimator; it keeps full accuracy for series whose level is large
relative to their variation (adding 1e9 to every series leaves the slopes, the
likelihood and all test statistics unchanged).

Regressors are reported per equation in Stata's order: the lags of the first
variable, the lags of the second, ..., the exogenous regressors, `trend`,
`Intercept`. Terms are named `<equation>:L<j>.<variable>`.

If the lagged endogenous block is linearly dependent (a variable that is an
exact combination of the others, or a constant series) the fit is refused
(`collinear_system`); OpenEconometrics does not silently drop a lag of an endogenous
variable, because impulse responses of such a system are meaningless.

An endogenous variable that is a deterministic function of the regressors (a
time trend, `0.5^t`, an accounting identity in lagged variables) is fitted
without error. Its residual variance is rounding noise, so standard errors, the
likelihood and every test would be meaningless numbers; the fit is refused with
`perfect_fit` (residual sum of squares below `1e-20` of the outcome's sum of
squares) and the message says to move the variable to `x` or drop it.

### Covariance, `small` and `dfk`

These follow the Methods and formulas of Stata's `var`:

| Options | Covariance of the coefficients | `extra["sigma"]` | Reference distribution |
| --- | --- | --- | --- |
| default | `Sigma_ml (x) (Z'Z)^-1` | `U'U / T` | z, chi2 |
| `dfk=True` | `[U'U / (T-m)] (x) (Z'Z)^-1` | `U'U / (T-m)` | z, chi2 |
| `small=True` | `[U'U / (T-m)] (x) (Z'Z)^-1` | `U'U / T` | t(T-m), F(q, T-m) |
| `covariance="robust"` | `(I (x) A) [sum_t (u_t u_t') (x) (z_t z_t')] (I (x) A)` times `T/(T-1)` (`T/(T-m)` with `small` or `dfk`), `A = (Z'Z)^-1` | as above | as above |

`small=True, dfk=True` reproduces `regress` equation by equation (and EViews'
VAR output). The log likelihood, `det_sigma_ml`, the information criteria and
the LM autocorrelation test always use the ML `Sigma`. Orthogonalized impulse
responses, the normality test and forecast standard errors use
`extra["sigma"]`, i.e. they change with `dfk`, as in Stata ("dfk estimator
used in computations" under `varnorm`; "varlmar always uses the ML estimator").

Stata's manual states the two options as: `dfk` uses the divisor `T - m` for
`Sigma` (m = average number of parameters per equation); `small` makes "Wald
tests ... have F or t distributions" and "the standard errors from each
equation are computed using the degrees of freedom for the equation". Both are
confirmed by the published `var`, `vargranger` and `varwle` output (see
Verification).

### What the result contains

`metrics`

| Name | Definition |
| --- | --- |
| `log_likelihood` | `-(T/2){ln|Sigma_ml| + K ln(2 pi) + K}` (Stata's `e(ll)`) |
| `aic`, `bic`, `hqic` | `-2 ln L` plus `2`, `ln T`, `2 ln ln T` times the `K m` coefficients |
| `aic_per_obs`, `hqic_per_obs`, `sbic_per_obs` | the same divided by T: the numbers Stata prints (`e(aic)`, `e(hqic)`, `e(sbic)`) |
| `fpe` | `|Sigma_ml| ((T + m)/(T - m))^K` |
| `det_sigma_ml`, `n_equations`, `n_lags`, `df_eq` (m), `df_model` (K m), `df_resid` (T - m), `T` | |

statsmodels and Lutkepohl drop the constant `K (1 + ln 2 pi)` of the likelihood
from the criteria; add it to compare (the tests do).

`tests` (each `{statistic, df, [df2], p_value, distribution, label}`)

- `granger_all_<eq>`: Wald test that the lags of all other endogenous variables
  are jointly zero in equation `<eq>` (the `ALL` rows of `vargranger`).
- `lag_exclusion_L<j>`: all endogenous variables at lag j, in all equations
  (the `All` table of `varwle`), `chi2(K^2)`.
- `normality`: joint Jarque-Bera statistic, `chi2(2K)`.
- `lm_autocorrelation_L<s>`: LM test of no residual autocorrelation at lag s,
  `chi2(K^2)`.

Wald tests are chi-squared; with `small=True` they are `F(q, T - m)` with
`F = chi2 / q`.

`extra`

- `sigma`, `sigma_ml`: residual covariance matrices.
- `equations`: Stata's header table. Per equation `parms`, `rmse` (always
  `sqrt(SSR / (T - m))`), `r_squared` (centered with a constant, uncentered
  without) and the Wald test that all coefficients except the constant are zero
  (`chi2(m - 1) = T R2 / (1 - R2)` by default, F with `small`).
- `lag_order_selection`: the `varsoc` table (below) for lags 0..`maxlag`.
- `stability`: eigenvalues of the companion matrix (`real`, `imag`, `modulus`,
  largest first) and `stable` (all moduli below 1). An unstable VAR adds a warning.
- `granger`: one row per equation and excluded variable, plus the `ALL` rows.
- `lag_exclusion`: one row per lag and equation, plus the joint rows.
- `normality`: per equation and jointly the skewness and kurtosis statistics
  with their chi-squared tests and the Jarque-Bera statistic.
- `irf`: arrays `[step][response][impulse]` for `simple`, `orthogonalized`,
  `generalized`, the variance decomposition `fevd` and `se` (standard errors of
  all four). Stored only while `(irf_steps + 1) K^2 <= 5000`; `oe.irf` computes
  any horizon up to 500 from the result.
- `forecast`: the last p observations and the regressor moments that
  `oe.forecast` needs; `layout`: variable names, lags and regressor labels.

Per-observation series (residuals, fitted values) are not stored; `predictions`
holds the usual bounded chart sample for the first equation.

### Lag-order selection (`varsoc`)

`oe.varsoc(data=..., y=[...], maxlag=4)` fits VARs with 0, 1, ..., `maxlag` lags
on the **same sample** (`T = n - maxlag`) and returns a table with

    ll    = -(T/2) {ln|Sigma_ml| + K ln(2 pi) + K}
    lr    = 2 {ll(j) - ll(j-1)}  ~  chi2(K^2)                  (with df and p_value)
    fpe   = |Sigma_ml| ((T + m_j)/(T - m_j))^K
    aic   = -2 ll/T + 2 K m_j / T
    hqic  = -2 ll/T + 2 ln(ln T) K m_j / T
    sbic  = -2 ll/T + ln(T) K m_j / T

`attrs["selected"]` gives the lag that minimizes each criterion and, for `lr`,
the highest lag whose LR test rejects at level `alpha` (Stata's stars; verified
against the table printed in the Stata manual). All nested fits come from a
single QR of the design ordered lag by lag: with R the triangular factor of
`[Z, Y]`, the residual cross product of the model on the first c columns is
`R[c:, Y]'R[c:, Y]`. The same table for the fitted model's own sample is in
`fit.extra["lag_order_selection"]` (Stata's `varsoc` after `var`).

Exogenous regressors and the constant enter every lag order, including lag 0,
and count in `m_j` (Stata: "the average number of parameters over the K
equations"). Stata's `lutstats` option prints Lutkepohl's versions, which drop
the constant of the likelihood and the penalty of the `d0` exogenous terms:
`aic_lutkepohl = aic - K (1 + ln 2 pi) - 2 K d0 / T` (and the same with
`2 ln ln T` and `ln T` for HQIC and SBIC). OpenEconometrics reports the standard
versions; the tests convert them to check the `lutstats` tables of the manual.

### Granger causality, lag exclusion, normality, autocorrelation

- **Granger causality** (`vargranger`): for equation i and variable v, the Wald
  statistic `b'V^-1 b` of the p coefficients on the lags of v, `chi2(p)`; the
  `ALL` row tests all other variables jointly, `chi2(p (K - 1))`.
- **Lag exclusion** (`varwle`): for lag j, the K coefficients of an equation,
  `chi2(K)`, and all `K^2` coefficients jointly.
- **Normality** (`varnorm`): the residuals are orthogonalized with the Cholesky
  factor of `extra["sigma"]`, `w_t = P^-1 u_t`. With `b1_k = mean(w_k^3)` and
  `b2_k = mean(w_k^4)`: skewness `T b1_k^2 / 6 ~ chi2(1)`, kurtosis
  `T (b2_k - 3)^2 / 24 ~ chi2(1)`, Jarque-Bera their sum `~ chi2(2)`; the joint
  statistics sum over equations (`chi2(K)`, `chi2(K)`, `chi2(2K)`).
- **LM autocorrelation** (`varlmar`, Johansen 1995): the VAR is augmented with
  the residuals lagged s times (missing initial values set to zero) and
  `LM_s = (T - d - 0.5) ln(|Sigma_ml| / |Sigma~_s|) ~ chi2(K^2)`, where `d = m + K`
  is the number of coefficients per equation of the augmented model. This
  reproduces the values printed in the Stata manual.

### Impulse responses and variance decomposition (`oe.irf`)

    simple            Phi_i,      Phi_0 = I,  Phi_i = sum_(j<=min(i,p)) Phi_(i-j) A_j
    orthogonalized    Theta_i = Phi_i P,          Sigma = P P' (P lower triangular)
    generalized       Psi_i   = Phi_i Sigma diag(Sigma)^(-1/2)       (Pesaran and Shin 1998)
    FEVD              omega_(rc,h) = sum_(i<h) Theta_i[r,c]^2 / sum_(i<h) (Phi_i Sigma Phi_i')[r,r]

`Sigma` is `extra["sigma"]`. Orthogonalized responses depend on the order of
`y`; generalized responses do not (the generalized response to the first
variable equals the orthogonalized one). The decomposition is reported with
`fevd = 0` at step 0 and the h-step share at step h, like Stata's `irf table`.

Standard errors are asymptotic, by the delta method of Lutkepohl (2005, 3.7),
which is what Stata's `irf create` uses:

    G_i = d vec(Phi_i)/d alpha' = sum_(m<i) J (M')^(i-1-m) (x) Phi_m      (M: companion matrix)
    V(vec Phi_i)   = G_i V_alpha G_i'
    V(vec Theta_i) = C_i V_alpha C_i' + Cbar_i V_sigma Cbar_i'
    C_i = (P' (x) I) G_i,   Cbar_i = (I (x) Phi_i) H,   H = L'{L (I + K_KK)(P (x) I) L'}^-1
    V_sigma = (2/T) D+ (Sigma (x) Sigma) D+'

`V_alpha` is taken from the coefficient covariance of the fit, so robust,
`small` and `dfk` choices carry over. Generalized responses use the Jacobian of
`Sigma diag(Sigma)^(-1/2)` in place of `H`; the FEVD standard errors accumulate
the Jacobians of `Theta_0..Theta_(h-1)` by the chain rule (equivalent to
Lutkepohl's d vectors). Internally everything runs in units of the innovation
standard deviations, so variables on very different scales do not degrade the
Jacobians. Standard errors are skipped (and the table says so) when
`K^2 p > 1500`.

`oe.irf(result, steps=8, kind="orthogonalized", alpha=0.05)` returns a long
table: `impulse`, `response`, `step`, `irf`, `std_error`, `ci_low`, `ci_high`
and, for the orthogonalized kind, `fevd`, `fevd_std_error`.

### Forecasts (`oe.forecast` / `oe.var_forecast`)

    y_T(h) = v + d (T + h) + A_1 y_T(h-1) + ... + A_p y_T(h-p) + B x_(T+h)
    Sigma_y^(h) = sum_(i<h) Phi_i Sigma Phi_i'  +  (1/T) Omega(h)

The second term is Lutkepohl's (2005, 3.5) estimation-uncertainty correction,
averaged over the sample regressor vectors, which Stata's `fcast compute`
includes by default:

    (1/T) Omega(h) = (1/T) sum_t G_t V G_t',   G_t = sum_(i<h) Z_t'(B')^(h-1-i) (x) Phi_i.

OpenEconometrics evaluates it without a loop over observations (it depends on the data
only through `Z'Z/T`) and with the coefficient covariance actually used. With
exogenous regressors or a trend only the innovation part is reported
(`attrs["mse_formula"]` says which; Stata reports no standard errors in that
case). `exog=` must supply the future values of the exogenous regressors, one
row per step. `data=` moves the forecast origin to the last row of another
table with the same columns. The table has one row per variable and step:
`variable`, `step`, `period` (integer time columns), `forecast`, `std_error`,
`ci_low`, `ci_high`.

### Worked example

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(42)
n = 200
e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.5, 0.8]]).T
y = np.zeros((n, 2))
for t in range(2, n):
    y[t] = ([0.5, 1.0] + np.array([[0.5, 0.2], [0.1, 0.4]]) @ y[t - 1]
            + np.array([[-0.2, 0.0], [0.1, 0.1]]) @ y[t - 2] + e[t])
df = pd.DataFrame(y, columns=["income", "consumption"])
df["quarter"] = np.arange(1, n + 1)

oe.varsoc(data=df, y=["income", "consumption"], time="quarter", maxlag=4)
#    lag        ll       lr   df  p_value     fpe     aic    hqic    sbic
# 0    0 -547.5022      NaN  NaN      NaN  0.9337  5.6072  5.6207  5.6406
# 1    1 -504.3508  86.3027  4.0   0.0000  0.6262  5.2077  5.2483  5.3080
# 2    2 -487.3567  33.9882  4.0   0.0000  0.5484  5.0751  5.1428  5.2423
# 3    3 -485.7170   3.2793  4.0   0.5122  0.5618  5.0992  5.1939  5.3333
# 4    4 -484.5522   2.3296  4.0   0.6754  0.5784  5.1281  5.2500  5.4291
# attrs["selected"] = {'fpe': 2, 'aic': 2, 'hqic': 2, 'sbic': 2, 'lr': 2}

fit = oe.var(data=df, y=["income", "consumption"], time="quarter", lags=2)
print(fit.summary())
# income:L1.income        0.3697 (0.0816)   income:L1.consumption   0.3702 (0.0900)   ...
# log_likelihood: -493.829 | aic_per_obs: 5.08918 | fpe: 0.556241 | det_sigma_ml: 0.502789
# Granger causality: all other variables excluded from income: chi2(2) = 17.0672, p = 0.0001967
# Jarque-Bera test that the disturbances are jointly normal: chi2(4) = 1.27785, p = 0.8651
# LM test of no residual autocorrelation at lag 1: chi2(4) = 5.2971, p = 0.2581

fit.extra["stability"]["stable"]            # True; moduli 0.4596, 0.4596, 0.3769, 0.3769

oe.irf(fit, steps=3).head(4)
#   impulse response  step     irf  std_error  ci_low  ci_high    fevd  fevd_std_error
# 0  income   income     0  0.9738     0.0489  0.8779   1.0697  0.0000          0.0000
# 1  income   income     1  0.5547     0.0753  0.4072   0.7022  1.0000          0.0000
# 2  income   income     2  0.1039     0.0778 -0.0486   0.2565  0.9453          0.0262
# 3  income   income     3  0.0268     0.0696 -0.1096   0.1632  0.9382          0.0316

oe.forecast(fit, 3)
#       variable  step  period  forecast  std_error  ci_low  ci_high
# 0       income     1     201    1.6358     0.9860 -0.2967   3.5683
# 1       income     2     202    1.4671     1.1676 -0.8214   3.7556
# ...
```

## `oe.vecrank`: Johansen tests for the cointegrating rank

### The procedure

The VAR(p) in levels is written in error-correction form

    D y_t = alpha (beta' y_(t-1) + mu + rho t) + sum_(i<p) Gamma_i D y_(t-i) + gamma + tau t + e_t.

`lags` is p, the number of lags of the VAR **in levels**; the VEC model has
`lags - 1` lagged differences (Stata's convention; statsmodels' `k_ar_diff` is
`lags - 1`). Following the Methods and formulas of Stata's `vec`, the model is
`Z0_t = alpha beta~' Z1_t + Psi Z2_t + e_t` with

| `trend` | Z1 | Z2 | Meaning | Osterwald-Lenum table | EViews case | statsmodels |
| --- | --- | --- | --- | --- | --- | --- |
| `"none"` | `y_(t-1)` | `D y` lags | no deterministic terms | 0 | 1 | `det_order=-1`, VECM `"n"` |
| `"rconstant"` | `(y_(t-1), 1)` | `D y` lags | constant only inside the cointegrating equations | 1* | 2 | VECM `"ci"` |
| `"constant"` (default) | `y_(t-1)` | `D y` lags, 1 | unrestricted constant: linear trends in the levels | 1 | 3 | `det_order=0`, VECM `"co"` |
| `"rtrend"` | `(y_(t-1), t)` | `D y` lags, 1 | trend inside the cointegrating equations | 2* | 4 | VECM `"coli"` |
| `"trend"` | `y_(t-1)` | `D y` lags, t, 1 | unrestricted trend: quadratic trends in the levels | 2 | 5 | VECM `"colo"` |

`R0`, `R1` are the residuals of `Z0`, `Z1` on `Z2` (one QR), `S_ij = R_i'R_j/T`,
and the eigenvalues `1 > l_1 >= ... >= l_K >= 0` solve
`|l S11 - S10 S00^-1 S01| = 0`. They are computed as the squared singular values
of `L^-1 S01 C^-T` with the Cholesky factors `S00 = L L'`, `S11 = C C'` (the
canonical correlations; no matrix is inverted). Then, for r = 0..K,

    ln L(r)  = -(T/2) [K {ln(2 pi) + 1} + ln|S00| + sum_(i<=r) ln(1 - l_i)]
    trace(r) = -T sum_(i>r) ln(1 - l_i)        H0: rank <= r   against   rank > r
    max(r)   = -T ln(1 - l_(r+1))              H0: rank = r    against   rank = r + 1
    parms(r) = K m2 + (K + m1 - r) r           (m1, m2 = columns of Z1, Z2)

statsmodels' `coint_johansen(det_order=1)` detrends the levels before
differencing, which is not Johansen's unrestricted-trend model; for
`trend="trend"` compare with statsmodels' `VECM(..., deterministic="colo")`.
Two more things to know when comparing with statsmodels (0.14; both were
checked against explicit algebra during verification): `coint_johansen(...,
k_ar_diff=0)` pairs the differences with the contemporaneous instead of the
lagged level, so its statistics for `lags=1` differ from OpenEconometrics's (and from
its own `VECM(k_ar_diff=0)`, which agrees with OpenEconometrics); and `VECM.stderr_beta`
/ `stderr_coint` for `coint_rank > 1` hold the right numbers in a scrambled
order (a vector ordered row by row is reshaped column by column).

### The table

One row per rank: `rank`, `parms`, `ll`, `eigenvalue`, `trace`, `trace_cv5`,
`trace_cv1`, `max`, `max_cv5`, `max_cv1`, `sbic`, `hqic`, `aic` (the criteria
per observation, as Stata's `ic` option). `attrs`:

- `selected_rank`: Johansen's estimate, the first r whose trace statistic does
  not exceed its 5% critical value (`selected_rank_1pct` with the 1% values,
  `selected_rank_max` with the maximum-eigenvalue statistic at 5%);
- `eigenvalues`, `n_obs`, `lags`, `trend`, `variables`, `critical_values`
  (the citation), `p_values` (why there are none), `critical_values_flag`
  (`None`, or a note when a reported critical value is single-sourced; see
  below).

The parameter counts, log likelihoods, statistics and information criteria
follow the table printed in [TS] vecrank (`vecrank y i c, lags(5)`: 39, 44, 47,
48 parameters; SBIC -25.12401 at rank 0 from LL = 1231.1041 and T = 91), which
the tests recompute.

### Critical values

The critical values are those of Osterwald-Lenum (1992), which Stata's `vecrank`
reports. They are data typed into `var/critical_values.py`; **all five tables
are shipped for K - r = 1..11** (5% and 1%, trace and maximum eigenvalue).
Nothing was taken from memory: every number was compared with independent
published transcriptions of the paper's tables, and the tests repeat the
comparison (`tests/test_econ_var_oracle.py`):

| Table (`trend`) | Compared with | Status |
| --- | --- | --- |
| 1* (`rconstant`), 2* (`rtrend`), dimensions 1-11 | `johans.ado` (Heinecke, Morris and Joly, SSC S418701) **and** the arrays of the R package urca (`ca.jo`) | two independent transcriptions agree |
| 1 (`constant`), dimensions 1-3 | `johans.ado` **and** the output printed in the Stata manual ([TS] vecrank; the 5% trace value of dimension 4, 47.21, is printed in [TS] vec intro) | two sources agree |
| 1 (`constant`), dimensions 4-11 | `johans.ado` (the first transcription in this module was typed separately and agrees) | one machine-readable source |
| 0 (`none`), dimensions 1-6; 2 (`trend`), dimensions 1-5 | `johans.ado` (same remark) | one machine-readable source |
| 0 (`none`), dimensions 7-11; 2 (`trend`), dimensions 6-11 | `johans.ado` only: these entries were added from it | **single source, flagged** |

A second verification pass downloaded `johans.ado` and urca's `ca-jo.R` again
and compared all 220 shipped numbers programmatically (no difference).

**One known difference from Stata.** Output posted by Stata users shows 9.42 as
the 5% critical value of `trend(rconstant)` for K - r = 1; Osterwald-Lenum's
Table 1* has 9.24 in both independent transcriptions (and in EViews 4 output),
and OpenEconometrics ships 9.24. This was not checked against a Stata run. If it is
right, a trace or maximum-eigenvalue statistic between 9.24 and 9.42 in the
last row of a `rconstant` table is a rejection here and not in Stata.

When a reported critical value belongs to the last row of the table above, the
result says so:
`attrs["critical_values_flag"]` of `oe.vecrank` (and
`extra["critical_values_flag"]` of `oe.vec`) holds the note
"critical values for K - r >= 7 with trend='none' were confirmed against one
published transcription ... only"; it is `None` otherwise. Proof-read those
entries against the paper before relying on their second decimal.

Plausibility checks, also in the tests (they would reveal a value from the
wrong table or dimension, not a wrong second decimal):

- every value of Tables 0, 1 and 2 lies within 4% of the MacKinnon-Haug-Michelis
  (1999) value of the same case that ships with statsmodels (7% for Table 0 with
  one common trend: 3.84 against 4.13, a known weakness of the 1992 table);
- the 5% trace values of all five tables lie within 1.2% of Johansen's (1995)
  Tables 15.1-15.5 for three or more common trends (3% for one or two; 7.5% for
  Table 0);
- a simulation of the limiting distribution (random walks of length 400, as in
  the paper) reproduces the 5% trace values of several dimensions, including the
  single-source ones, within 2.5%.

The 1992 tables were simulated with 6,000 replications and lie 1-3% below more
accurate asymptotic quantiles for large K - r. **P-values are not reported**:
the limit distributions are non-standard, and OpenEconometrics does not ship the
MacKinnon-Haug-Michelis response surfaces that EViews uses. EViews 5+ therefore
prints slightly different critical values than OpenEconometrics and Stata.

Note for readers of other software: the R package urca's `ecdet = "none"` table
is Osterwald-Lenum's Table 1.1* (unrestricted constant in the model, no drift
in the data), which Stata does not use; statsmodels' `coint_johansen` uses
MacKinnon-Haug-Michelis values (`det_order = -1, 0, 1` for `none`, `constant`,
`trend`) and has no restricted cases.

## `oe.vec`: vector error-correction model

### Estimator

Johansen's maximum likelihood, as in Stata's `vec`:

1. `beta~` = the eigenvectors of the r largest eigenvalues, normalized so that
   the first r rows are the identity (`beta~' = (I_r, beta-breve')`, Johansen's
   normalization). **Order `y` so that the variables on which the cointegrating
   equations are normalized come first**; if the leading block is singular the
   fit is refused (`normalization_failed`).
2. `alpha = S01 beta~ (beta~'S11 beta~)^-1`, `Omega = S00 - alpha beta~'S10`
   (ML, divisor T).
3. `mu` and `rho` are estimated inside the cointegrating equations for
   `rconstant` and `rtrend`. Otherwise they are backed out of the unrestricted
   constant `v` and trend `tau`: `mu = (alpha'alpha)^-1 alpha'v`,
   `rho = (alpha'alpha)^-1 alpha'tau`; the short-run constant is
   `gamma = v - alpha mu` (orthogonal to alpha), the short-run trend
   `tau - alpha rho`.
4. The short-run parameters are the OLS coefficients of `D y_t` on the
   cointegrating equations `E_t = beta'y_(t-1) + mu + rho t`, the lagged
   differences and the unrestricted deterministic terms, conditional on the
   (superconsistent) `beta`.

The deterministic terms come from `trend=` alone. `oe.vec` has no `constant`
argument; in a hand-built `ModelSpec` the `intercept` field is not used, and
`intercept=False` together with a `trend` that keeps an unrestricted constant
(`constant`, `rtrend`, `trend`) adds a warning that says so.

Numerically, whenever the model has a constant the lagged levels enter the
reduced-rank regression in deviations from their sample means, and the
normalization is carried out in units of the root mean squares of the `Z1`
columns. Both are exact reparameterizations (the constants and the standard
error of a restricted constant are mapped back), so the estimates do not
depend on the level or the units of the series: adding 1e6 to every series
changes only `_cons` (by `-beta' 1e6`), and variables 16 orders of magnitude
apart are normalized without loss. `mu = (alpha'alpha)^-1 alpha'v` is solved by
QR; this formula of Stata's depends on the units of the variables, and when
alpha is numerically rank deficient in those units the fit is refused with
`singular_adjustment` (rescale the variables, or use `rconstant` / `rtrend`).

### Covariance

Stata's small-sample convention: the divisor is `T - d` instead of `T`, with

    d = floor( {K m2 + (K + m1 - r) r} / K )        (free parameters per equation)
    V(short-run) = T/(T - d) * Omega (x) (W'W)^-1,   W = [E, Z2]
    V(vec beta-breve) = (1/(T - d)) (alpha' Omega^-1 alpha)^-1 (x) (H'S11 H)^-1

(`H` selects the rows of `Z1` below the normalized block). Normalized elements
and backed-out `mu`, `rho` have no standard errors. Inference is z based.
statsmodels uses the divisor T: its standard errors are smaller by
`sqrt((T - d)/T)`.

### Result

Coefficients are grouped by equation `D_<variable>`: `D_<v>:L._ce<i>` (the
adjustment coefficients alpha), `D_<v>:LD.<w>`, `D_<v>:L2D.<w>`, ... (Gamma_i),
`D_<v>:trend`, `D_<v>:Intercept`.

- `metrics`: `log_likelihood`, `aic`, `bic`, `hqic` and the per-observation
  `aic_per_obs`, `hqic_per_obs`, `sbic_per_obs` (Stata's), `det_sigma_ml`,
  `rank`, `n_equations`, `n_lags`, `df_model` (free parameters), `df_eq` (d), `T`.
- `tests`: `lag_exclusion_L<j>D` (all coefficients of the j-th lagged
  difference), `normality`, `lm_autocorrelation_L<s>` (the VAR statistics
  applied to the VEC regression given beta; `d` counts its coefficients per
  equation plus K).
- `extra["beta"]`: every cointegrating equation with `estimate`, `std_error`,
  `statistic`, `p_value`, interval and `status` (`normalized`, `estimated`,
  `backed out ...`) per variable, `_trend` and `_cons`, plus Stata's
  "Cointegrating equations" test: `parms = K - r` and the Wald `chi2(K - r)`
  (`statistic`, `df`, `p_value`) that the free coefficients on the variables are
  jointly zero (a restricted constant or trend is not part of this test).
- `extra["equations"]`: Stata's header table. `parms` = coefficients of the
  equation, `rmse = sqrt(SSR / (T - d))`, `r_squared` is the **uncentered**
  `1 - SSR / sum (D y)^2` and `statistic` the Wald test that **all**
  coefficients of the equation, the constant included, are zero:
  `chi2(parms) = (T - d) R2 / (1 - R2)`. The numbers printed in [TS] vec obey
  this identity (0.9313 and 664.4668 for `D_ln_ne` with T - d = 49); it is why
  the R-squared of a VEC equation with drifting series looks high. The centered
  R-squared is given as `r_squared_centered` when the equation has a constant.
- `extra`: `beta_matrix` (K x r), `alpha`, `ce_constant` (mu), `ce_trend` (rho),
  `pi = alpha beta'`, `omega`, `eigenvalues`, `rank_test` (the `vecrank` rows),
  `selected_rank`, `critical_values_flag`, `stability`, `var_representation`,
  `normality`.

`extra["stability"]` lists the eigenvalues of the companion matrix of the VAR in
levels, `A_1 = alpha beta' + Gamma_1 + I`, `A_i = Gamma_i - Gamma_(i-1)`,
`A_p = -Gamma_(p-1)`. The model imposes `K - r` unit moduli
(`unit_moduli_imposed`); `stable` says whether the remaining ones lie inside
the unit circle (a warning is added otherwise, the sign of a rank that is too
high).

### Forecasts and impulse responses

`oe.forecast(vec_result, steps)` iterates the VAR representation from the last
p observations; the standard errors are Stata's
`T/(T - d) * sum_(i<h) Phi_i Omega Phi_i'` (no parameter uncertainty).
`oe.irf(vec_result, ...)` gives the responses of the VAR representation (they
do not die out: shocks have permanent effects) without standard errors, as in
Stata.

### Worked example

```python
rng = np.random.default_rng(3)
w = rng.normal(size=300).cumsum()                      # one common stochastic trend
d2 = pd.DataFrame({"y1": w + rng.normal(size=300), "y2": 2.0 + 0.5 * w + rng.normal(size=300),
                   "y3": rng.normal(size=300).cumsum()})

oe.vecrank(data=d2, y=["y1", "y2", "y3"], lags=2)
#    rank  parms         ll  eigenvalue     trace  trace_cv5  trace_cv1       max  max_cv5 ...
# 0     0     12 -1479.6261         NaN  121.0358      29.68      35.65  114.4757    20.97
# 1     1     17 -1422.3882      0.3190    6.5602      15.41      20.04    5.8901    14.07
# 2     2     20 -1419.4432      0.0196    0.6701       3.76       6.65    0.6701     3.76
# 3     3     21 -1419.1081      0.0022       NaN        NaN        NaN       NaN      NaN
# attrs["selected_rank"] = 1

vec = oe.vec(data=d2, y=["y1", "y2", "y3"], lags=2, rank=1)
pd.DataFrame(vec.extra["beta"][0]["coefficients"])
#   variable  estimate  std_error  statistic  p_value  ...  status
# 0       y1    1.0000        NaN        NaN      NaN       normalized
# 1       y2   -2.0175     0.0454   -44.4039   0.0000       estimated
# 2       y3   -0.0303     0.0332    -0.9102   0.3627       estimated
# 3    _cons    4.0915        NaN        NaN      NaN       backed out of the unrestricted estimates
# (the data were generated with y1 - 2 y2 + 4 stationary)
{k: v for k, v in vec.extra["beta"][0].items() if k != "coefficients"}
# {'equation': '_ce1', 'parms': 2, 'statistic': 2167.648, 'df': 2, 'p_value': 0.0, 'distribution': 'chi2'}

pd.DataFrame(vec.extra["equations"])
#   equation  parms    rmse  r_squared  statistic  df  p_value distribution  r_squared_centered
# 0     D_y1      5  1.6236     0.1501    51.7590   5   0.0000         chi2              0.1491
# 1     D_y2      5  1.2012     0.3612   165.6587   5   0.0000         chi2              0.3609
# 2     D_y3      5  0.9908     0.0071     2.0848   5   0.8373         chi2              0.0059

# adjustment coefficients: D_y1:L._ce1 -0.2246 (0.0585), D_y2:L._ce1 0.3507 (0.0433),
#                          D_y3:L._ce1 -0.0100 (0.0357)
vec.extra["stability"]        # moduli 1, 1, 0.338, 0.105, 0.062, 0.058; 2 unit moduli imposed
oe.forecast(vec, 2)
#   variable  step  forecast  std_error   ci_low  ci_high
# 0       y1     1   16.2462     1.6236  13.0641  19.4283
# 1       y1     2   16.5752     1.9416  12.7697  20.3808
# ...
```

## Error codes

| Code | When |
| --- | --- |
| `invalid_spec` | `y` is not a list of distinct columns, a variable is both endogenous and exogenous, an option has the wrong type or value, unknown covariance (`vec` accepts only `"nonrobust"`), `rank < 1`, fewer than two variables for `vec` |
| `invalid_rank` | `rank >= K` |
| `invalid_option`, `invalid_lags`, `invalid_steps`, `invalid_alpha` | `irf_kinds`/`kind`/`max_rank`, `maxlag`, `steps`, `alpha` out of range; a keyword that the model's forecast function does not have (`oe.forecast(vec_result, 5, exog=...)`) |
| `missing_columns`, `missing_values`, `non_numeric_column`, `non_finite_values` | data problems |
| `time_gaps`, `repeated_time_values`, `invalid_time` | the series is not regularly spaced |
| `insufficient_observations` | too few observations for the lags and variables |
| `collinear_system`, `singular_residual_covariance` | linearly dependent endogenous variables (or a constant series) |
| `perfect_fit` | an endogenous variable is a deterministic function of the regressors (time trend, identity): its equation has no error |
| `singular_adjustment` | `vec`: mu or rho cannot be backed out because alpha is numerically rank deficient in the units of the data |
| `duplicate_terms` | an exogenous column is called `trend` or `Intercept` while the model has that term |
| `model_too_large`, `design_too_large` | more than 1000 coefficients (their full covariance is stored), more than 50 variables, or a design above 256 MiB |
| `normalization_failed` | Johansen's normalization is impossible in the given variable order |
| `missing_exog`, `invalid_exog` | forecasts of a model with exogenous regressors |
| `invalid_result` | a post-estimation function received another estimator's result |
| `numerical_failure`, `non_finite_result` | explosive responses or forecasts over the requested horizon |

## Verification

Seven test files compare the family with independent oracles.

`tests/test_econ_var_verify.py` (second verification pass):

- the worked example of Lutkepohl (2005, sections 3.2.3 and 3.5.3; West German
  investment, income and consumption, T = 73, `dfk`): the coefficient matrices,
  `Sigma_u`, the point and interval forecasts and the forecast MSE **with the
  estimation-uncertainty term** (23.34, 1.505, 0.978 and 25.12, 1.581, 1.009
  times 1e-4 for steps 1 and 2), reproduced to every printed digit: this is the
  formula of Stata's `fcast compute`;
- the closed form of the one-step forecast MSE, `Sigma (1 + m/T)` (and its
  `small` / `dfk` variants);
- the two `vecrank` tables printed in [TS] vec intro: parameter counts, trace
  statistics from the printed log likelihoods, critical values;
- generalized impulse responses **and their standard errors** equal the
  orthogonalized ones of the model refitted with the impulse variable ordered
  first (Pesaran and Shin 1998), for the default, `dfk` and robust covariances;
- statsmodels `VAR` with a trend (`"ct"`) and without a constant (`"n"`):
  coefficients, standard errors, likelihood, criteria, stability, forecasts;
- invariance to the order of the variables: every reduced-form statistic of
  `oe.var` (the Cholesky-based joint normality test is the documented
  exception), the `oe.vecrank` table, and the likelihood, `Pi = alpha beta'`,
  `Omega` and forecasts of `oe.vec` for all five trend specifications;
- a shuffled frame with its own row labels, a time column and missing ends;
- the defects fixed in this pass: a keyword unknown to the model's forecast
  function, `irf_kinds=5`, `vecrank(lags=None)` and `vecrank(trend=None)` raised
  raw `TypeError` / `KeyError` (they now raise `invalid_option` /
  `invalid_spec`, and `None` means the documented default); `y` given as a set
  was accepted although its order is arbitrary; a hand-built `vec` spec with
  `intercept=False` kept its constant without a warning.

`tests/test_econ_var_oracle.py` (published numbers and second sources):

- the Johansen critical values against `johans.ado`, urca, the Stata manual,
  MacKinnon-Haug-Michelis (statsmodels), Johansen (1995) and a simulation (see
  Critical values);
- output printed in the Stata 19 manuals for the Lutkepohl data, reproduced to
  every printed digit: [TS] var example 2 (exogenous variable, `dfk`:
  coefficients, standard errors, intervals, log likelihood, AIC/HQIC/SBIC, FPE,
  equation table), [TS] varsoc example 2 (post-estimation with an exogenous
  variable, `lutstats`), [TS] vargranger (`dfk small`: F, df, df_r, p),
  [TS] varwle (F and chi2 versions), [TS] varnorm (`dfk`: skewness, kurtosis,
  Jarque-Bera per equation and jointly), [TS] varlmar (`mlag(5)`),
  [TS] varstable (eigenvalues and moduli) and [TS] irf table (orthogonalized
  responses, variance decompositions, their standard errors and intervals, for
  two Cholesky orderings);
- the arithmetic of the tables printed in [TS] vecrank and [TS] vec, whose data
  are not available offline: parameter counts, information criteria, the
  degrees of freedom `d`, `chi2 = (T - d) R2 / (1 - R2)` with the uncentered
  R-squared, and the cointegrating-equation chi2.

`tests/test_econ_var_oracle_algebra.py` (derivations that differ from the
implementation's):

- `oe.var` as generalized least squares on the stacked Kronecker system (full
  covariance for default, `dfk`, `small`), the robust covariance as an explicit
  sum over observations, a numerically differentiated Gaussian likelihood and
  its brute-force maximization;
- every Wald test (Granger, lag exclusion per equation and jointly, the
  equation tests) from restricted and unrestricted regressions,
  `W = T tr{Sigma_u^-1 (Sigma_r - Sigma_u)}`;
- the lag-order table against separately fitted models on the common sample;
- companion eigenvalues as reciprocal roots of `det(I - A_1 z - A_2 z^2)`;
- impulse responses and the variance decomposition by simulating the
  propagation of a shock through the system;
- forecast standard errors from the companion form and Lutkepohl's trace
  formula for `Omega(h)` (default, `dfk`, `small`);
- equivalent specifications (built-in trend or constant against exogenous
  columns, rescaled regressors and variables, shuffled rows with a time
  column, dropped missing ends, categorical against indicator columns) and a
  seven-observation sample;
- `oe.vecrank` from canonical correlations (QR and SVD) for all five trend
  specifications; the nesting of the VEC model: rank K is the VAR(p) in levels
  and rank 0 the VAR(p - 1) in differences fitted by `oe.var`;
- `oe.vec`: residuals, likelihood, forecasts and forecast standard errors
  rebuilt from the reported coefficients in the original levels (all five trend
  specifications, including series with a level of 5000); both covariance
  formulas as inverse Hessians of an independently written likelihood; a
  brute-force maximization of the likelihood; the header statistics; exact
  invariance to the level and equivariance to the units of the series.

`tests/test_econ_var_adversarial.py`: more than 140 hostile inputs (empty and tiny
samples, all-missing and constant columns, collinear and deterministic
variables, text, infinities, magnitudes from 1e-8 to 1e9, gaps and duplicates
in time, options at and beyond their bounds, wrong result types) each of which
must either produce a finite, JSON-clean result or raise the documented error
code, and one regression test per defect fixed during verification.

`tests/test_econ_var.py`, `tests/test_econ_var_irf.py` and
`tests/test_econ_var_vec.py` (written with the implementation): explicit NumPy
algebra, statsmodels `VAR` (coefficients, likelihood, criteria, Granger tests,
normality, roots, lag selection, impulse responses with standard errors, FEVD,
forecasts), statsmodels `VECM` (all five trend cases) and `coint_johansen`,
numerical Jacobians for all delta-method standard errors, and the [TS] var,
varbasic, varsoc and varlmar examples.

The estimators are closed-form, so there is no iterative likelihood whose
score would need `check_derivatives`. `provenance["stata_parity_validated"]` is
`False`: the comparison is with published examples, not with a Stata run.

## Performance

Timings on a laptop CPU (6 threads), best of three, including all stored
diagnostics. Every estimator is linear in the number of observations.

| Task | n = 100,000 | n = 1,000,000 |
| --- | --- | --- |
| `oe.var`, K = 5, p = 4 | 0.07 s | 0.70 s |
| `oe.var`, K = 5, p = 4, robust | 0.09 s | 0.92 s |
| `oe.var`, K = 5, p = 4, time column, exogenous regressor, trend | 0.07 s | 0.76 s |
| `oe.var`, K = 3, p = 2 | 0.04 s | 0.35 s |
| `oe.varsoc`, K = 5, maxlag = 8 | 0.05 s | 0.49 s |
| `oe.vecrank`, K = 5, lags = 4 | 0.02 s | 0.38 s |
| `oe.vec`, K = 5, lags = 4, rank 1 | 0.06 s | 0.74 s |
| `oe.vec`, K = 5, lags = 4, rank 2, `rconstant` | 0.06 s | 0.60 s |

`oe.irf` with standard errors (K = 5, p = 4): 0.015 s for 40 steps, 0.07 s for
200 steps. `oe.forecast`, 100 steps: 0.013 s (VAR, with the estimation
uncertainty term), 0.004 s (VEC).

## Limitations

- `lags` is a single integer: lags 1..p are all included (Stata's `lags(1 3)`
  with holes is not available). Exogenous regressors enter contemporaneously;
  lag them yourself if needed. A row with a missing exogenous value is excluded
  even when it would only supply lagged values of the endogenous variables
  (Stata keeps it): fill such leading values with any number to keep the row.
- No constraints on coefficients (Stata's `constraints()`), no structural VAR
  (`svar`), no bootstrap standard errors for responses or forecasts, no
  cumulative responses or dynamic multipliers. No weights.
- Lutkepohl's versions of the information criteria (`lutstats`) are not
  reported (see Lag-order selection for the conversion).
- `vec` has no seasonal indicators and no constraints on alpha or beta; only
  Johansen's normalization; only the conventional covariance.
- Series must be gap free; panels of series are not supported.
- Critical values for the rank tests stop at K - r = 11 and those of
  `trend="none"` for K - r >= 7 and `trend="trend"` for K - r >= 6 are
  single-sourced (flagged in the result); no p-values for the rank tests.
- At most 1000 coefficients and 50 variables per system.

## Conventions and how sure we are of them

Verified against output printed in the Stata manuals (see Verification): the
default, `dfk` and `small` covariances and reference distributions of `var`;
the log likelihood, FPE and information criteria; the equation table of `var`
(RMSE with `T - m` whatever the options, centered R-squared, chi2 or F over
all coefficients but the constant); `varsoc` including exogenous variables;
`vargranger`, `varwle`, `varnorm` (with the `dfk` Sigma), `varlmar` (always the
ML Sigma, `d = m + K`), `varstable`; orthogonalized responses and the variance
decomposition with their standard errors; the parameter counts and
per-observation criteria of `vecrank` and `vec`; `d = floor(parameters / K)`
and the divisor `T - d` of the `vec` equation chi2.

Inferred from one published table: that the R-squared of the `vec` equation
table is uncentered and its chi2 covers all coefficients. The printed pairs
satisfy `chi2 = (T - d) R2 / (1 - R2)`, an identity that holds both for the
uncentered R-squared with all coefficients and for the centered one with the
slopes only; an R-squared of 0.93 for the growth rate of income is plausible
only for the uncentered version (the series drift), and Stata's "Parms" column
counts the constant. OpenEconometrics reports the uncentered version as `r_squared`
with its chi2 and adds `r_squared_centered`.

Verified against a textbook example instead of Stata output: the forecast
standard errors of `var` with Lutkepohl's estimation term (Lutkepohl 2005,
section 3.5.3, every printed digit; [TS] fcast compute documents this formula).

Taken from Stata's Methods and formulas but not checked against output (no
published example could be recomputed; the data sets of the [TS] vec, veclmar
and vecnorm examples, `rdinc`, `urates`, `txhprice` and `balance2`, are not
available offline): the `vec` covariance of the short-run parameters and of
beta with the divisor `T - d`; `veclmar` and `vecnorm` (the manual says they
are `varlmar` / `varnorm` applied to the VAR in differences augmented with the
cointegrating equations; ML `Omega`); forecast standard errors of `vec`
(`T/(T - d)`, no estimation term); no standard errors for forecasts with
exogenous variables.

Uncertain (textbook-standard choice, stated here so that it can be corrected):

- The critical value of `trend="rconstant"` for K - r = 1: OpenEconometrics ships
  Osterwald-Lenum's 9.24; Stata appears to print 9.42 (see Critical values).
- `covariance="robust"`: Stata's manual says `var` supports `vce(robust)` and
  refers to [P] _robust without printing the small-sample factor; OpenEconometrics uses
  `T/(T-1)` and `T/(T-m)` with `small` or `dfk`.
- R-squared of a `var` without a constant is uncentered, and its equation test
  covers all coefficients.
- `vec`: the trend variable is t = 1, 2, ... counted from the first observation
  of the ordered sample (including the rows that only supply lags), in both the
  restricted and the unrestricted trend terms. Another origin shifts
  `_cons`/`mu` (and the short-run constant) but not beta, alpha, Gamma, the
  likelihood or the tests.
- `vec`, "Cointegrating equations" test: `parms = K - r` is confirmed for
  `trend(rconstant)` by the manual (the restricted constant is not counted);
  for `trend(rtrend)` OpenEconometrics likewise leaves the restricted trend out.
- The trend of `oe.var` (`trend=True`) is an OpenEconometrics option; in Stata a trend
  is passed as an exogenous variable.
- Generalized impulse responses (Pesaran and Shin) have no Stata counterpart;
  their standard errors use the same delta method and were checked only
  against numerical derivatives.
