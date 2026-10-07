# Instrumental variables: `ivregress`, `xtivreg`, `ivreghdfe`

Linear models with endogenous regressors,

    y = X1 b1 + X2 b2 + u,        E[Z'u] = 0,   Z = [X1 Z2],

where `X1` are the exogenous regressors (`x`, with the constant), `X2` the
endogenous regressors (`endog`, `q` columns) and `Z2` the *excluded*
instruments (`instruments`, `L2` columns). `K = K1 + q` coefficients,
`L = K1 + L2` instruments, order condition `L2 >= q`.

Everything on this page is implemented in OpenEconometrics on float64 PyTorch tensors.
No projection matrix is ever formed: two Householder QR factorizations (the
first stage `[X2 y]` on `Z`, and `[X2 y Z2]` on `X1`) carry every estimator
and every diagnostic; fixed effects are absorbed by O(n) group means; cluster,
HAC and multiway meats come from the shared sandwich kernels. No estimation
library runs at fit time. One million rows with 10 exogenous regressors, two
endogenous regressors and four instruments fit in about a second (timings at
the end).

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({"firm": np.repeat(np.arange(200), 10),
                   "year": np.tile(np.arange(2010, 2020), 200)})
a = np.repeat(rng.normal(size=200), 10)
df["x1"] = rng.normal(size=n) + 0.5 * a
df["z1"], df["z2"], df["z3"] = rng.normal(size=n) + 0.3 * a, rng.normal(size=n), rng.normal(size=n)
v = rng.normal(size=n)
df["p"] = 0.8 * df.z1 + 0.5 * df.z2 + 0.3 * df.z3 + 0.4 * df.x1 + a + v
df["y"] = 1 + 2 * df.p - df.x1 + a + 0.7 * v + rng.normal(size=n) * (1 + 0.5 * df.z1.abs())

args = dict(data=df, y="y", x=["x1"], endog=["p"], instruments=["z1", "z2", "z3"])
print(oe.ivregress(**args, covariance="robust").summary())
```

```
Instrumental-variables (2SLS) regression — y
Observations: 2000  |  Covariance: robust  |  Confidence: 95%

Term        Estimate  Std. error         z     P>|stat|   CI lower   CI upper
---------  ---------  ----------  --------  -----------  ---------  ---------
Intercept   0.938767   0.0393061   23.8835  4.5471e-126   0.861729    1.01581
x1         -0.829722   0.0469177  -17.6847  5.50619e-70  -0.921679  -0.737765
p             2.0899   0.0390947   53.4575            0    2.01328    2.16653

r_squared: 0.862524  |  adjusted_r_squared: 0.862387  |  rmse: 1.75289  |  df_model: 2
df_resid: 1997  |  n_instruments: 3  |  n_endogenous: 1
Wald chi2 test of the slopes: chi2(2) = 3417.93, p = 0
Wooldridge robust score test of overidentifying restrictions: chi2(2) = 2.93241, p = 0.2308
Wooldridge robust score chi2 test of exogeneity: chi2(1) = 199.11, p = 3.267e-45
Robust regression-based F test of exogeneity: F(1, 1996) = 286.403, p = 3.853e-60
Cragg-Donald Wald F (minimum eigenvalue statistic): F(3, 1995) = 555.505
Kleibergen-Paap rk Wald F: F(3, 1995) = 521.821
```

Further calls on the same data (all outputs on this page were produced by the
code shown):

```python
oe.ivregress(**args, method="liml")          # kappa 1.0015989, b_p 2.088471
oe.ivregress(**args, method="gmm")           # b_p 2.079819 (se 0.038627), Hansen J 2.9324
oe.xtivreg(**args, panel="firm", time="year")                 # fe: b_p 1.92178 (se 0.0393025)
oe.xtivreg(**args, panel="firm", time="year", model="re")     # G2SLS: b_p 2.07235, theta 0.0773665
oe.ivreghdfe(**args, absorb=["firm", "year"], cluster="firm") # b_p 1.92278 (se 0.0422137)
```

| Stata | OpenEconometrics |
| --- | --- |
| `ivregress 2sls y x1 (p = z1 z2)` | `oe.ivregress(data=df, y="y", x=["x1"], endog=["p"], instruments=["z1","z2"])` |
| `..., vce(robust)` / `vce(cluster firm)` / `small` | `covariance="robust"` / `cluster="firm"` / `small=True` |
| `..., vce(hac nwest 4)` (after `tsset t`) | `covariance="hac", lags=4, time="t"` (`kernel="bartlett"`) |
| `ivregress liml ...` | `method="liml"` |
| `ivregress gmm ..., wmatrix(robust)` / `igmm` / `center` | `method="gmm"` (`wmatrix="robust"`) / `igmm=True` / `center=True` |
| `estat firststage` | `result.extra["first_stage"]`, `result.tests["cragg_donald"]` |
| `estat overid` | `result.tests["overid_sargan"]`, `["overid_basmann"]`, `["overid_score"]`, `["anderson_rubin"]`, `["basmann_f"]`, `["hansen_j"]` |
| `estat endogenous` | `result.tests["endog_durbin"]`, `["endog_wu_hausman"]`, `["endog_robust_score"]`, `["endog_robust_regression"]`, `["endog_c"]` |
| `ivreg2 ..., robust` weak-identification statistics | `result.tests["cragg_donald"]`, `["kleibergen_paap_rk_f"]` |
| `xtset id year` + `xtivreg y x1 (p = z1 z2), fe` | `oe.xtivreg(data=df, y="y", x=["x1"], endog=["p"], instruments=["z1","z2"], panel="id", time="year")` |
| `xtivreg ..., re` / `re ec2sls` / `be` / `fd` / `small` | `model="re"` / `model="re", ec2sls=True` / `model="be"` / `model="fd"` / `small=True` |
| `ivreghdfe y x1 (p = z1 z2), absorb(firm year) cluster(firm)` | `oe.ivreghdfe(data=df, y="y", x=["x1"], endog=["p"], instruments=["z1","z2"], absorb=["firm","year"], cluster="firm")` |

`instruments` lists only the excluded instruments: the exogenous regressors
instrument themselves and must not be repeated (`invalid_spec`). `x` may be
omitted. `categorical=[...]` applies to exogenous regressors (treatment coding,
first level as reference, terms such as `sector[b]`). `missing="drop"`
excludes incomplete rows and records how many; the default `"raise"` refuses
them. Coefficients are ordered exogenous first (`Intercept` first), then
endogenous.

## Numerical approach

With `P_A` the projection on the columns of `A`, `M_A = I - P_A`, and weights
`w` entering every inner product as `<a, b> = sum_i w_i a_i b_i`:

    first    [X2 y] on Z     ->  Xhat2 = P_Z X2,  E = M_Z X2,  e_y = M_Z y
    partial  [X2 y Z2] on X1 ->  X2p = M_1 X2,  y_p = M_1 y,  Z2p = M_1 Z2

and `G = X2p - E = (P_Z - P_1) X2`, the part of the first-stage fit that the
excluded instruments add. By Frisch-Waugh-Lovell the k-class estimator
`b = {X'(I - k M_Z)X}^-1 X'(I - k M_Z) y` is

    S_k = G'WG - (k - 1) E'WE
    b2  = S_k^-1 {G'W y_p - (k - 1) E'W e_y}        b1 = c_y - C2 b2

(`C2`, `c_y`: coefficients of `X2` and `y` on `X1`), with the bread
`{X'(I - k M_Z)X}^-1` assembled from `(X1'WX1)^-1`, `C2` and `S_k^-1`. 2SLS
(`k = 1`) is a QR of `G`; nothing is inverted except `q`-by-`q` and
`L`-by-`L` Cholesky factors. Residuals are always `u = y - X b` with the
*observed* endogenous regressors.

**Collinearity (Stata style).** Columns are omitted left to right with a
warning: first within `X1` (centered under the constant), then endogenous
regressors collinear with `X1` or earlier endogenous ones (recorded in
`provenance["omitted_terms"]`), then instruments collinear with `X1` or
earlier instruments (`extra["omitted_instruments"]`). The order condition is
checked afterwards (`underidentified`); an instrument set that does not move
every endogenous regressor independently fails the rank condition
(`underidentified`).

**Weights** (`ivregress`, `ivreghdfe`). `aweight`s and `pweight`s are rescaled
to sum to the number of rows (`N` = rows); `fweight`s replicate observations
(`N = sum f_i`; every number equals the duplicated-row data set's, including
the diagnostics and, for `ivreghdfe`, the singleton rule). `pweight`s need
`robust`, `cluster` or `hac` (`unsupported_covariance` otherwise) and default
to `robust`. `xtivreg` takes no weights (Stata's does not either).

## `oe.ivregress` — Stata `ivregress 2sls | liml | gmm`

    oe.ivregress(*, data, y, x=None, endog, instruments, method="2sls", covariance=None,
                 cluster=None, small=False, weights=None, weight_type=None, wmatrix=None,
                 igmm=False, lags=None, kernel=None, time=None, panel=None, center=False,
                 categorical=None, intercept=True, missing="raise", alpha=0.05)

### Estimators

- **`2sls`**: `b = (X'P_Z X)^-1 X'P_Z y`.
- **`liml`**: k-class with `kappa` the smallest eigenvalue of
  `(Y'M_Z Y)^-1 Y'M_1 Y`, `Y = [y X2]`, from a Cholesky factor of `Y'M_Z Y` and a
  symmetric eigenproblem. `kappa >= 1`; in an exactly identified model
  `kappa = 1` identically and LIML is 2SLS (the eigenproblem is skipped).
  Reported as `metrics["kappa"]`.
- **`gmm`**: two-step efficient GMM
  `b = (X'Z S^-1 Z'X)^-1 X'Z S^-1 Z'y`. The first step is 2SLS; `S` is the
  covariance of the moments `z_i w_i u_i` from its residuals, of type
  `wmatrix`: `"robust"` (default) `sum u_i^2 z_i z_i'`, `"cluster"` (outer
  products of cluster sums; needs `cluster`), `"hac"` (kernel-weighted
  autocovariances; needs `lags`, optional `kernel`, `time`, `panel`) or
  `"unadjusted"` (`s^2 Z'Z`, which reproduces 2SLS). `igmm=True` re-estimates
  `S` from the latest residuals until the coefficients change by less than
  `1e-10` (relative; `extra["gmm"]["iterations"]`). `center=True` subtracts
  the mean moment before forming `S`. `S` must be positive definite: a cluster
  weight matrix with fewer clusters than instruments raises
  `singular_weight_matrix`. With a constant the regressors and instruments are
  centered for the GMM algebra (an exact reparameterization that keeps `S`
  well conditioned).

### Covariance and inference (Stata's `ivregress` conventions)

`Xk = (I - kappa M_Z) X` are the score regressors (`Xhat = P_Z X` for 2SLS) and
`B = {X'(I - kappa M_Z) X}^-1` the bread.

| `covariance` | `small=False` (default) | `small=True` |
| --- | --- | --- |
| `nonrobust` (alias `unadjusted`) | `RSS/N * B` | `RSS/(N-K) * B` |
| `robust` | `B (sum u_i^2 xk_i xk_i') B` | times `N/(N-K)` |
| `cluster` (one or two columns) | cluster sandwich times `G/(G-1)` | times `G/(G-1) (N-1)/(N-K)` |
| `hac` | Newey-West sandwich (`lags`, `kernel`) | times `N/(N-K)` |
| coefficient tests | z | t with `N-K` df (`G-1` with clusters) |
| model test `tests["model"]` | Wald chi2 of the slopes | F = Wald/df |
| `metrics["rmse"]` | `sqrt(RSS/N)` | `sqrt(RSS/(N-K))` |

Defaults: `nonrobust` for 2sls/liml; `cluster` when `cluster=` is given;
`robust` for pweights; for GMM the type of the weight matrix. Two cluster
columns use the Cameron-Gelbach-Miller inclusion-exclusion meat with the
smaller dimension's `G/(G-1)` and `G_min - 1` degrees of freedom; an
indefinite result has its negative eigenvalues set to zero and is flagged
(`inference["psd_adjusted"]`). HAC kernels: `bartlett` (Newey-West, weights
`1 - l/(lags+1)`), `parzen`, `quadratic_spectral`, `truncated`; with `time`
only pairs exactly `l` periods apart contribute (gaps respected), with `panel`
autocovariances are formed within panels.

GMM: when the covariance type equals `wmatrix` the efficient
`(X'Z S^-1 Z'X)^-1` is reported (with `S` from the weighting step, times the
factor of the table); otherwise the sandwich
`(A'S^-1 A)^-1 A'S^-1 S_v S^-1 A (A'S^-1 A)^-1`, `A = Z'X`, with `S_v` of the
covariance type from the final residuals (`nonrobust`: `s^2 Z'Z`).

`metrics`: `r_squared` (`1 - RSS/TSS` with the IV residuals; centered with a
constant, uncentered without; it can be negative and is reported as is —
Stata prints nothing in that case), `adjusted_r_squared`
(`1 - (1-R^2)(N-c)/(N-K)`, always reported), `rmse`, `kappa` (liml),
`df_model`, `df_resid`, `n_instruments`, `n_endogenous`, and `j`,
`gmm_iterations` (gmm).

### Diagnostics

`a` below is the number of absorbed degrees of freedom (zero for `ivregress`).

**First stage** (`estat firststage`), `extra["first_stage"]`, one record per
endogenous regressor: `r_squared = 1 - RSS_Z/TSS`, `adjusted_r_squared` with
`(N - c - a)/(N - L - a)`, `partial_r_squared = 1 - RSS_Z/RSS_1` (after `Z`
and after `X1` only), `shea_partial_r_squared =
[(X'X)^-1]_jj / [(Xhat'Xhat)^-1]_jj`, and the F test of the excluded
instruments: `F(L2, N-L-a) = {(RSS_1 - RSS_Z)/L2} / {RSS_Z/(N-L-a)}`, or with a
robust, cluster or HAC model covariance the Wald F under that covariance with
`regress`'s factors (`N/(N-L)`; `G/(G-1) (N-1)/(N-L)` and `G - 1` second
degrees of freedom when clustered), whatever `small` says.

**Weak identification.** `tests["cragg_donald"]`: the Cragg-Donald Wald F,
Stata's "minimum eigenvalue statistic",
`(N-L-a)/L2 * min eig{(X2'M_Z X2)^-1 X2'(P_Z - P_1) X2}`. With a robust,
cluster or HAC covariance also `tests["kleibergen_paap_rk_f"]`, the
Kleibergen-Paap (2006) rk Wald statistic for the rank of the reduced form
being `q - 1` (singular value decomposition of the standardized reduced form,
variance of the smallest singular value under the model covariance; stored as
`rk_wald_chi2` with `rk_df = L2 - q + 1`), scaled as `ivreg2` scales it:
`rk/N * (N-L-a)/L2`, or with clusters `rk/(N-1) * (N-L-a) * (G-1)/G / L2`. With
one endogenous regressor it equals the robust first-stage F. Neither entry has
a p-value. The separate `stock_yogo(result)` API supplies the single-endogenous,
unweighted iid 2SLS reference; it does not assign Stock–Yogo thresholds to robust
KP output. Structural-null AR/CLR tests, scalar AR confidence sets and
Fuller/k-class fits are described in [inference extensions](inference-extensions.md).

**Overidentification** (`estat overid`), `df = L2 - q`; in an exactly
identified model the entry has `statistic: None` and a note.

| fit | entries |
| --- | --- |
| 2sls, `nonrobust` | `overid_sargan = N u'P_Z u / u'u`; `overid_basmann = u'P_Z u / {u'M_Z u/(N-L-a)}`; chi2 |
| 2sls, `robust`/`cluster`/`hac` | `overid_score`: Wooldridge's robust score test `s'M^-1 s`, `s = sum_i w_i k_i u_i`, `k` the residuals of the excluded instruments on `Xhat`, `M` the covariance of those moments of the model's type (White: `N - RSS` of 1 on `u k`) |
| liml | `anderson_rubin = N ln(kappa)` (chi2) and `basmann_f = (kappa - 1)(N-L-a)/(L2-q)`, `F(L2-q, N-L-a)` |
| gmm | `hansen_j = g(b)'S^-1 g(b)`, `g = Z'W u` (with `S` in the 1/N scaling: `N gbar'W gbar`) |

**Endogeneity** (`estat endogenous`; H0: the endogenous regressors are
exogenous), `q` restrictions. With `u_e` the OLS residuals,
`Q = u_e'P_[Z X2] u_e - u'P_Z u` (equivalently the drop in RSS when the
first-stage residuals are added to the OLS regression):

| fit | entries |
| --- | --- |
| 2sls, `nonrobust` | `endog_durbin = Q / (u_e'u_e/N)`, chi2(q); `endog_wu_hausman = (Q/q) / {(u_e'u_e - Q)/(N-K-q-a)}`, `F(q, N-K-q-a)` |
| 2sls, robust types | `endog_robust_score`: Wooldridge's score test for the first-stage residuals `v` in the OLS equation (`s'M^-1 s` of `r_i u_e,i`, `r = M_X v`), chi2(q); `endog_robust_regression`: Wald F of the coefficients of `v` in the regression of `y` on `[X, v]` under the model covariance with `regress`'s factors |
| gmm | `endog_c`: `C = J_e - J_c`, the J of the model that treats `X2` as exogenous (instruments `[Z X2]`) minus the J of the estimated model, both from one GMM step with the moment covariance of the former at the OLS residuals, chi2(q) |
| liml | none (as Stata) |

When the first stage has fewer than `q` residual degrees of freedom the
diagnostics are skipped with a warning; when the instruments reproduce an
endogenous regressor exactly the affected entries carry `statistic: None`.

## `oe.xtivreg` — Stata `xtivreg`

    oe.xtivreg(*, data, y, x=None, endog, instruments, panel, time=None, model="fe",
               covariance=None, cluster=None, ec2sls=False, small=False,
               categorical=None, missing="raise", alpha=0.05)

`y_it = x_it'b1 + endog_it'b2 + u_i + e_it`; `N` observations in `n` panels of
`T_i` periods, `k` slopes, `K = k + 1`. The sample is sorted by panel and time;
repeated periods are an error. Each model is 2SLS on transformed data.

| `model` | transformation | error variance of the conventional VCE |
| --- | --- | --- |
| `fe` (default) | panel means removed, grand means added back (the reported `Intercept` is Stata's `_cons = ybar - xbar'b`) | `sigma_e^2 = RSS/(N - n - k)` |
| `fd` | first differences over consecutive periods (needs `time`), with a constant; rows without a predecessor are dropped and recorded | `RSS/(N_d - K)` |
| `be` | panel means, with a constant (each panel once) | `RSS/(n - K)` |
| `re` | quasi-demeaning `w - theta_i wbar_i` of outcome, regressors (constant: `1 - theta_i`) and instruments (G2SLS); `ec2sls=True`: instruments are the within-transformed instruments and their panel means (Baltagi's EC2SLS) | `RSS*/(N - K)` of the transformed regression |

**Random-effects variance components** (Swamy-Arora adapted to unbalanced
panels, Baltagi and Chang 2000):

    sigma_e^2 = RSS_w / (N - n - k_w)                       within 2SLS
    sigma_u^2 = max(0, {SSR*_b - (n - K_b) sigma_e^2} / (N - r))
    SSR*_b = sum_i T_i ubar_i^2,   r = trace{(sum_i T_i z_i z_i')^-1 sum_i T_i^2 z_i z_i'}
    theta_i = 1 - sqrt(sigma_e^2 / (T_i sigma_u^2 + sigma_e^2))

`ubar_i` are the residuals of the between 2SLS in which each panel mean
appears `T_i` times, `z_i` the panel means of its regressors; `k_w` and `K_b`
count the columns that vary within / between panels (time-invariant columns
leave the within regression, a trend or period dummies in a balanced panel
leave the between regression). In balanced panels this is
`sigma_u^2 = RSS_b/(n - K_b) - sigma_e^2/T`. `extra["variance_components"]`
records every ingredient; `extra["theta"]` summarizes `theta_i`
(`metrics["theta"]` is the common value in balanced panels). If `sigma_u^2`
is truncated at zero the estimator is pooled 2SLS and a warning says so.

**Covariance.** `nonrobust`: `s^2 (X~'P X~)^-1` on the transformed data with
the error variance of the table. `robust` *is* clustering on the panel
variable (for `be`: HC1 on the panel-mean regression). `cluster`: one column
(constant within panel for `be`). Sandwiches carry
`G/(G-1) * (N-1)/(N-K)` with `K = k + 1`, the panel effects not being counted
when they are nested in the clusters (the `xtreg, fe` convention); for `fe` a
cluster column that does not nest the panels counts the `n - 1` absorbed
effects in `K` and warns; for `re` it only warns. With a cluster covariance
the `fe` constant has variance `xbar'V xbar` (within residuals sum to zero in
every panel): with regressors of near-zero grand mean it is rounding noise and
is flagged; an exact zero raises `zero_variance_constant`.

**Statistics.** z statistics and a Wald chi2 test of the slopes for every
model, as Stata's `xtivreg` prints by default; `small=True` gives t and F with
the residual degrees of freedom (`G - 1` with a cluster covariance). The
covariance matrix does not depend on `small`.

**Metrics.** fe/be/re: `r_squared_within`, `r_squared_between`,
`r_squared_overall`. The model's own transformation reports the R-squared of
its 2SLS regression, `1 - RSS/TSS` (within for `fe`, between for `be`; it can
be negative, where Stata prints a missing value); the others are squared
correlations of `x'b` and `y` (within deviations, panel means, levels), as for
`xtreg`. `fe`: `sigma_u` (standard deviation of `u_i = ybar_i - xbar_i'b` over
panels, `n - 1` divisor), `sigma_e`, `rho`, `corr_u_xb` (correlation of `u_i`
with `x_it'b` over the `N` observations), `rmse`. `re`: `sigma_u`, `sigma_e`,
`rho`, `theta`, `rmse` (of the transformed regression). `fd`: `r_squared`,
`adjusted_r_squared`, `rmse` of the differenced equation. All: `n_groups`,
`t_min`, `t_avg`, `t_max`, `n_instruments`, `n_endogenous`.

**Tests.** `tests["model"]`; `fe` with `nonrobust` adds
`tests["fixed_effects"]`, the F(n-1, N-n-k) test that all `u_i = 0` from the
pooled and within 2SLS residual sums of squares.

With instruments that reproduce the "endogenous" regressors every model equals
the corresponding `oe.xtreg` fit (`re` with `sa=True`) to rounding, including
the reported statistics; this is asserted in the tests.

## `oe.ivreghdfe` — community `ivreghdfe` (`ivreg2` + `reghdfe`)

    oe.ivreghdfe(*, data, y, x=None, endog, instruments, absorb, method="2sls",
                 covariance=None, cluster=None, small=True, weights=None, weight_type=None,
                 drop_singletons=True, tolerance=1e-8, max_iterations=10000,
                 categorical=None, missing="raise", alpha=0.05)

`y = X1 b1 + X2 b2 + sum_d alpha_d[codes_d] + u`. The pipeline:

1. `drop_singletons` (default): observations alone in a level of an absorbed
   dimension are dropped iteratively (frequency-weighted rows with `f >= 2`
   are never singletons).
2. `[y X1 X2 Z2]` are demeaned jointly (`engines.absorb.demean`: exact
   one-pass within transform for one dimension, conjugate-gradient accelerated
   alternating projections otherwise, weighted when weights are given).
3. Regressors and instruments without variation left are omitted and recorded.
4. 2SLS, LIML or two-step GMM exactly as in `ivregress` on the residualized
   data (Frisch-Waugh-Lovell: the estimates equal those of the IV regression
   with explicit dummies). GMM uses the weight matrix of the covariance type
   (`ivreg2`'s `gmm2s`; with `nonrobust` it equals 2SLS).
5. Degrees of freedom as `reghdfe`: `df_absorbed` sums the observed levels
   minus the redundant coefficients (connected components for the second
   dimension, pairwise for later ones); a dimension nested in *any* cluster
   column is not counted, and when every dimension is nested one degree of
   freedom is added for the constant. `K = k + df_absorbed`.
6. Exact fits are rejected (`perfect_fit`), with the iterative-demeaning
   resolution `(32 tolerance)^2 TSS_within` for two or more dimensions.

`small=True` (ivreghdfe's default): t and F statistics, `RSS/(N-K)`, robust
`N/(N-K)`, cluster `G/(G-1) (N-1)/(N-K)` with `G - 1` (`G_min - 1`) inference
degrees of freedom. `small=False` (`ivreg2` without `small`): z and chi2,
`RSS/N` and no finite-sample factor at all, not even `G/(G-1)`. No constant is
reported. Covariances: `nonrobust`, `robust`, `cluster` (one or two columns).

`metrics`: `r_squared` (with the fixed effects), `adjusted_r_squared`,
`r_squared_within`, `adjusted_r_squared_within`, `rmse`, `kappa` (liml),
`df_model`, `df_resid`, `df_absorbed`, `n_instruments`, `n_endogenous`,
`n_singletons_dropped`, `j` (gmm). `tests` and `extra["first_stage"]` are the
`ivregress` diagnostics on the residualized data with `a = df_absorbed` in
every denominator (first-stage R-squared is therefore a within R-squared).
`extra["absorbed"]` lists, per dimension, `column`, `levels`, `redundant`,
`nested`; `extra` also records the demeaning iterations and tolerance.

One absorbed dimension reproduces `xtivreg, fe` with `small=True` (slopes,
conventional and panel-clustered standard errors); absorbing a constant column
reproduces `ivregress` with `small=True`; and with instruments that reproduce
the "endogenous" regressors the fit equals `oe.reghdfe` (coefficients,
covariance under every option and weight type, degrees of freedom, R-squared).
All three are asserted in the tests.

## Errors

Every failure is an `AnalysisError` with a code and a message that says what to
change (specification-level mistakes such as an unknown option value are
rejected earlier, when the `ModelSpec` is built, as a pydantic
`ValidationError`).

| code | when |
| --- | --- |
| `underidentified` | fewer usable instruments than endogenous regressors after the collinearity screen; rank condition fails; within/between auxiliary regression of `re` not identified |
| `no_endogenous_regressors` | every endogenous regressor was omitted (collinear or absorbed) |
| `insufficient_observations` | no residual degrees of freedom; first stage with `N <= L`; LIML with `N - L < q + 1`; every panel a singleton (`fe`); fixed effects use every degree of freedom |
| `constant_outcome`, `perfect_fit` | no variation in `y`; the regressors (and fixed effects) fit `y` exactly |
| `perfect_first_stage` | overidentified LIML whose instruments reproduce a regressor (the eigenproblem is undefined) |
| `singular_weight_matrix` | GMM moment covariance not positive definite (fewer clusters than instruments) |
| `nonconvergence` | iterated GMM did not converge in 1000 iterations |
| `unsupported_covariance` | pweights with the conventional covariance |
| `insufficient_clusters`, `cluster_varies_within_panel`, `zero_variance_constant` | cluster problems |
| `repeated_time_values`, `invalid_time` | duplicate or non-integer periods |
| `empty_sample` | no complete rows; no consecutive periods (`fd`); every row a singleton (`ivreghdfe`) |
| `absorption_nonconvergence`, `invalid_option` | demeaning did not converge; bad `tolerance` / `max_iterations` |
| `invalid_spec` | overlapping roles, GMM or HAC options without their method, `fd` without `time`, `ec2sls` without `re` |
| `missing_values`, `non_numeric_column`, `non_finite_values`, `negative_weights`, `noninteger_frequency_weights` | data problems |

## Performance

Apple M-series laptop, CPU, float64; 10 exogenous regressors, 2 endogenous
regressors, 4 instruments, all diagnostics included.

| model | 1e5 rows | 1e6 rows |
| --- | --- | --- |
| `ivregress` 2sls, robust | 0.06 s | 1.3 s |
| `ivregress` 2sls, cluster / two-way cluster | 0.06 / 0.10 s | 1.1 / 1.7 s |
| `ivregress` liml, robust | 0.06 s | 1.1 s |
| `ivregress` gmm, robust / cluster + igmm | 0.07 / 0.08 s | 1.6 / 1.3 s |
| `ivregress` 2sls, hac(4) | 0.08 s | 1.7 s |
| `ivreghdfe` firm (1e3 / 1e4 levels) + year, cluster firm | 0.11 s | 1.0 s |
| `xtivreg` fe / fd / be | 0.09 / 0.07 / 0.04 s | 1.0 / 0.7 / 0.4 s |
| `xtivreg` re / ec2sls robust | 0.12 / 0.18 s | 1.7 / 4.2 s |

Cost is linear in `n`: there is no loop over observations and no n-by-n
matrix.

## Deliberate differences from Stata

- `adjusted_r_squared` is always reported for `ivregress` (Stata prints it
  only with `small`); a negative `r_squared` is reported as the number it is.
- `ivregress` accepts one or two cluster columns; Stata's takes one.
- `tests` holds every post-estimation statistic at once (first stage,
  overidentification, endogeneity, weak identification); Stata computes them
  with `estat` on request. Fuller/k-class fits retain first-stage and weak-ID
  diagnostics; uncalibrated overidentification/endogeneity tests are not assigned.
- `fit(ModelSpec(estimator="ivregress", options={"method": "gmm"}))` without a
  covariance uses the registry default `nonrobust` (the GMM sandwich with an
  unadjusted moment covariance); `oe.ivregress(method="gmm")` applies Stata's
  default, the covariance of the weight-matrix type.
- `xtivreg`: `fe` accepts a cluster column that does not nest the panels
  (with the absorbed effects counted and a warning); no weights; no
  `nosa`, `vce(bootstrap)` or `vce(jackknife)`.
- `ivreghdfe`: no HAC covariances, no `first`/`ffirst` printing, no saved
  fixed effects; `method="gmm"` is two-step only.

## What has been checked, and how

`provenance["stata_parity_validated"]` is `False` for every result: nothing
here has been run side by side with Stata, SPSS or EViews.

**Checked against independent oracles written for this purpose**
(`tests/test_econ_iv_oracle.py`: dense NumPy algebra with explicit n-by-n
projection matrices on square-root-weighted data, SciPy generalized
eigenvalues, dummy-variable regressions; plus `tests/test_econ_iv.py` and
`tests/test_econ_iv_panel.py`, which also use statsmodels' `IV2SLS`/`IVGMM`
and OLS): coefficients, full covariance matrices, degrees of freedom,
p-values, confidence intervals, metrics and every test statistic of
2SLS / LIML / GMM under every covariance, `small` setting and weight type;
two-step, iterated, centered and mismatched-VCE GMM; all first-stage,
Cragg-Donald, Kleibergen-Paap (from the 2006 paper's formulas), Sargan,
Basmann, Wooldridge, Anderson-Rubin, Durbin and Wu-Hausman statistics (the
last two from Stata's quadratic-form definitions); frequency weights against
duplicated rows; `xtivreg` fe against dummy-variable 2SLS and all models
against explicit transformations and against `oe.xtreg`; `ivreghdfe` against
two-way dummy-variable IV; invariance to row order and to rescaling by
`1e-8 .. 1e8`; the failure contract.

**Conventions we are not certain of** (Stata's manuals were not available
while this was written; these follow the stated source and our recollection of
it):

1. `ivregress` cluster covariance without `small`: `G/(G-1)` is applied (and
   `(N-1)/(N-K)` only with `small`); `ivreg2` applies neither without `small`.
2. LIML with a robust, cluster or HAC covariance: the k-class sandwich with
   score regressors `(I - kappa M_Z) X`. Stata's exact robust LIML formula was
   not verified.
3. GMM: the efficient `(X'Z S^-1 Z'X)^-1` uses the weighting-step `S` (as
   `ivreg2 gmm2s`); Stata may re-estimate `S` at the final residuals for the
   VCE. The `G/(G-1)` factor on the cluster GMM covariance, the `N/(N-K)`
   factor with `small`, and centering under weights
   (`w_i (m_i - mbar_w)`) are our reading.
4. `estat endogenous` after GMM (`endog_c`): both J statistics use one GMM
   step with the moment covariance evaluated at the OLS residuals.
5. Robust first-stage F and regression-based endogeneity F use `regress`'s
   factors (`N/(N-L)`, cluster `G/(G-1)(N-1)/(N-L)`) regardless of `small`;
   robust score tests carry no finite-sample factor.
6. Kleibergen-Paap rk Wald F: the scaling is `ivreg2`'s as we recall it.
7. `xtivreg`: z/chi2 as the default for every model, and the `fe` within
   (`be` between) R-squared as that of the transformed 2SLS regression, follow
   our recollection of the manual's examples. Stata's `fe` header prints a
   Wald chi2 that is not the plain Wald test of the slopes reported here.
8. `xtivreg, fe` F test of `u_i = 0`: pooled-versus-within residual sums of
   squares (clipped at zero); Stata's IV formula was not verified.
9. `xtivreg, re`: the Swamy-Arora variance components use the between 2SLS in
   which each panel mean appears `T_i` times and the trace term built from the
   regressors' panel means; the conventional VCE is `RSS*/(N-K) (X*'PX*)^-1`
   (by analogy with `xtreg, re`; Baltagi's textbook form is
   `sigma_e^2 (X*'PX*)^-1`). Time-invariant regressors reduce `k_w`. Balanced
   panels are unaffected by the first point. `nosa` (Baltagi-Chang) is not
   offered. EC2SLS in unbalanced panels uses the instrument set
   `[1, Q Z, P Z]` as in balanced ones.
10. `xtivreg, be` is unweighted 2SLS on the `n` panel means with `n - K`
    degrees of freedom; the manual describes a regression "in which each
    average appears T_i times", which differs in unbalanced panels.
11. Cluster/robust factors for `xtivreg` (`G/(G-1)(N-1)/(N-K)`, `K = k + 1`)
    are `xtreg`'s; `fd` reports only the differenced equation's R-squared
    (Stata also prints between/overall R-squared, `sigma_u`, `sigma_e`).
12. `ivreghdfe`: `small=True` as the default, the absorbed degrees of freedom
    in the first-stage, Cragg-Donald, Basmann and Wu-Hausman denominators
    (but `N` in Sargan and Durbin), and nesting judged against any cluster
    column follow `reghdfe`/`ivreg2` as we understand them.
