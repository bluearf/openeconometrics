# Panel-data linear models: `xtreg`, `hausman`, `xtfmb`

Linear models for data observed on `n` panels (firms, people, countries) over
`T_i` periods, `N = sum_i T_i` observations in all. Everything on this page is
implemented in OpenEconometrics on float64 PyTorch tensors: group means are one
`index_add_` pass, every least-squares step is a Householder QR, the
random-effects likelihood is evaluated in closed form, and the Driscoll-Kraay
and cluster covariances reuse the shared sandwich kernels. No estimation
library runs at fit time. A model on one million rows with 100,000 panels and
eight regressors fits in a fraction of a second (timings at the end).

The model throughout is

    y_it = x_it' b + u_i + e_it,      i = 1..n,  t = 1..T_i

with a panel effect `u_i` and an idiosyncratic error `e_it`. `b` has `k`
slopes; `K = k + 1` counts the constant.

```python
import openecon as oe

x = ["exp", "exp2", "union"]
robust = oe.xtreg(data=df, y="lwage", x=x, panel="id", time="year",
                  covariance="robust")                   # xtreg lwage exp exp2 union, fe vce(robust)
print(robust.summary())

fe = oe.xtreg(data=df, y="lwage", x=x, panel="id")       # conventional VCEs for the Hausman test
re = oe.xtreg(data=df, y="lwage", x=x, panel="id", model="re")
print(oe.hausman(fe, re, sigmamore=True))
```

| Stata | OpenEconometrics |
| --- | --- |
| `xtset id year` + `xtreg y x1 x2, fe` | `oe.xtreg(data=df, y="y", x=["x1","x2"], panel="id", time="year")` |
| `xtreg y x1 x2, fe vce(robust)` | `..., covariance="robust"` (clusters on `id`) |
| `xtreg y x1 x2, fe vce(cluster state)` | `..., covariance="cluster", cluster="state"` (`state` must nest `id`) |
| `xtscc y x1 x2, fe lag(3)` | `..., covariance="driscoll_kraay", lags=3` |
| `xtreg y x1 x2, re` / `re sa` | `..., model="re"` / `model="re", sa=True` |
| `xttest0` (after `xtreg, re`) | `re.tests["breusch_pagan"]` |
| `xtreg y x1 x2, be` / `be wls` | `model="be"` / `model="be", wls=True` |
| `regress D.y D.x1 D.x2` | `model="fd"` (time required) |
| `xtreg y x1 x2, mle` | `model="mle"` |
| `regress y x1 x2, vce(cluster id)` / `xtscc y x1 x2` | `model="pooled", covariance="robust"` / `covariance="driscoll_kraay"` |
| `hausman fe re, sigmamore` | `oe.hausman(fe, re, sigmamore=True)` |
| `xtfmb y x1 x2, lag(2)` | `oe.xtfmb(data=df, y="y", x=[...], panel="id", time="t", covariance="hac", lags=2)` |

The sample is always sorted by `panel` then `time` (Stata's `xtset`); repeated
periods within a panel are an error (`repeated_time_values`). `missing="drop"`
excludes incomplete rows and records how many; the default `"raise"` refuses
them. Categorical regressors (`categorical=[...]`) are treatment-coded with
the first level as reference, producing terms such as `sector[b]`.

## What each model accepts

| `model` | weights | covariances | coefficient tests |
| --- | --- | --- | --- |
| `fe` | `aweight`, `fweight`, `pweight`, constant within panel | `nonrobust`, `robust`, `cluster` (nested), `driscoll_kraay` | t |
| `re` | none | `nonrobust`, `robust`, `cluster` (nested) | z |
| `be` | none | `nonrobust`, `robust` (HC1), `cluster` (constant within panel) | t |
| `fd` | `aweight`, `fweight`, `pweight` (may vary by row) | `nonrobust`, `robust`, `cluster` | t |
| `pooled` | `aweight`, `fweight`, `pweight` (may vary by row) | `nonrobust`, `robust`, `cluster` (one or two columns), `driscoll_kraay` | t |
| `mle` | none | `nonrobust` (observed information) | z |

Weights follow the Stata manual's statement for `xtreg`: "aweights, fweights,
and pweights are allowed for the fixed-effects model ... iweights are allowed
for the maximum-likelihood random-effects model ... Weights must be constant
within panel" ([XT] xtreg, Options). `xtreg, re` and `xtreg, be` take no
weights, so OpenEconometrics rejects them there (`unsupported_weights`, naming the
model); `mle` would take iweights only, which OpenEconometrics does not implement.
`fd` and `pooled` are `regress` commands and take any of the three types.
Semantics: `aweight` is rescaled to sum to `N`; `fweight` replicates rows
(`N = sum f`, `T_i = sum_t f_it`, results identical to the duplicated
dataset); `pweight` requires `robust`, `cluster` or `driscoll_kraay`
(`unsupported_covariance` with `nonrobust`).

`oe.capabilities()` lists one inference label per estimator (`Student t` for
`xtreg`); the `re` and `mle` models nevertheless report z statistics, as the
table says and as `result.inference["use_t"]` records.

## Numerical approach shared by all models

**Centered slopes.** Every regression with a constant is run on `[1, X - m]`
with `m` the (weighted) grand means of the regressors, and the estimates are
mapped back exactly: `Intercept = Intercept_c - m'b`, `V = S V_c S'` with
`S = [[1, -m'], [0, I]]`. This is what Stata's `regress` achieves by sweeping
the constant first. Slopes, residuals, fitted values and every test are
unchanged; what changes is that a regressor with a large offset (a price level
of `1e8` with unit variation) is no longer numerically collinear with the
constant, and the sandwich covariances do not lose `eps * (offset/sd)^2` to
cancellation. For `fe` the centered regression is `[1, x_it - xbar_i]`; Stata's
transformed design `[1, x_it - xbar_i + xbar]` is the same regression with the
grand means added back. `mle` additionally scales every column and the outcome
to unit root mean square.

**Degenerate fits are refused.** A sum of squares of deviations of `y` cannot
be resolved below about `eps^2 * sum w y^2`. When the outcome's variation, or
the residual sum of squares, is below `1e-28 * sum w y^2` (the criterion of the
linear family), OpenEconometrics raises an error instead of reporting standard errors
of order `1e-16` with `p = 0`: `no_within_variation` (fe, re, mle),
`constant_outcome` or `perfect_fit` (be, fd, pooled, xtfmb). The yardstick is
the uncentered level of `y`, so an exact fit of `y = 1e10 + 2x` is caught while
a tiny but real signal on a large level (`sd/level` down to about `1e-14`) is
estimated. Stata prints zero or missing standard errors in these cases.

## `model="fe"`: the within (fixed-effects) estimator — `xtreg, fe`

**Estimator.** The panel effects are removed by the within transformation.
Stata regresses `y_it - ybar_i + ybar` on `x_it - xbar_i + xbar` and a
constant (grand means added back); OpenEconometrics reports exactly that regression,
so the `Intercept` is Stata's `_cons = ybar - xbar'b` with Stata's standard
error. The slopes equal those of the dummy-variable regression (LSDV); the n
dummies are never formed. Regressors without within variation (time-invariant
columns, a categorical that is constant inside every panel) are omitted and
recorded in `warnings` and `provenance["omitted_terms"]`, as Stata's "omitted
because of collinearity". The screen compares the within sum of squares of a
column with its mean-deviated total (ratio `1e-13`), so a regressor with a
large level and small within variation is kept.

**Formulas** ([XT] xtreg, Methods and formulas and Stored results).

    sigma_e^2 = RSS / (N - n - k),         V_nonrobust = sigma_e^2 (X*'X*)^-1
    u_i = ybar_i - xbar_i' b,              sigma_u = sd(u_1..u_n)  (n - 1 divisor)
    rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)
    corr_u_xb = corr(u_i, x_it' b)   over the N observations   (Stata's e(corr))
    R2 within  = corr((x_it - xbar_i)'b, y_it - ybar_i)^2
    R2 between = corr(xbar_i'b, ybar_i)^2
    R2 overall = corr(x_it'b, y_it)^2

`sigma_u` is the standard deviation of the `n` panel effects (each panel
counted once, `n - 1` divisor). `corr_u_xb` is Stata's `e(corr)`, stored as
"corr(u_i, x_it b)": each observation carries its panel's `u_i` and its own
`x_it'b`. (Over the `n` panel means the number is different: -0.1698 instead of
Stata's -0.1517 on the Grunfeld data.) Under weights the correlations within,
overall and `corr_u_xb` are weighted by the estimation weight; between is over
the `n` panel means.

**Tests.** `tests["model"]` is the F(k, N-n-k) test that all slopes are zero
(Wald F with `G - 1` denominator df under cluster/robust, `T - 1` under
Driscoll-Kraay). `tests["fixed_effects"]` (nonrobust only, as in Stata) is the
F test that all `u_i = 0`:

    F(n-1, N-n-k) = [(RSS_pooled - RSS_fe) / (n-1)] / [RSS_fe / (N-n-k)]

with `RSS_pooled` from pooled OLS of `y` on the same regressors and a constant.

**Metrics** (in this order): `r_squared_within`, `r_squared_between`,
`r_squared_overall`, `sigma_u`, `sigma_e`, `rho`, `corr_u_xb`, `rmse`
(= sigma_e), `df_resid`, `n_groups`, `t_min`, `t_avg`, `t_max`. `extra` holds
the RSS, the number of absorbed effects and the `K` used in small-sample
factors. Inference is Student t with `N - n - k` df (nonrobust).

**Covariances.**

- `nonrobust`: `sigma_e^2 (X*'X*)^-1`.
- `robust`: for `xtreg`, "specifying vce(robust) is equivalent to specifying
  vce(cluster panelvar)" ([XT] xtreg). OpenEconometrics implements it exactly so: the
  CR1 sandwich of the transformed regression (constant included) clustered on
  the panel, `G/(G-1) * (N-1)/(N-K)` with `K = k + 1`, t tests and the model F
  with `G - 1 = n - 1` degrees of freedom. The `n` fixed effects are *not*
  counted in `K` because they are nested in the clusters. The within residuals
  sum to zero in every panel, so the constant's own cluster score vanishes and
  the sandwich variance of `_cons` is exactly `xbar' V_b xbar` (asserted in the
  tests). It therefore depends on the level of the regressors, and with
  standardized regressors it is rounding noise (a standard error near `1e-17`).
  OpenEconometrics reports the number the formula gives, with a warning when it is
  zero up to rounding, and raises `zero_variance_constant` if it is exactly
  zero. Slopes are unaffected; `nonrobust` and `driscoll_kraay` do not have
  this property.
- `cluster` with `cluster="col"`: the same sandwich on those clusters, with
  `K = k + 1`. **The panels must be nested within the cluster column**: "The
  panel variable must be nested within the cluster variable because of the
  within-panel correlation induced by the within transform. The panel-nesting
  restriction is also enforced for multiway clustering" ([XT] xtreg, Methods
  and formulas). A column that varies inside a panel (a time variable, for
  example) raises `cluster_not_nested`, as Stata's `r(498)` does; it does not
  fall back to the `areg` degrees of freedom. Two cluster columns are accepted
  when both nest the panels (Cameron-Gelbach-Miller inclusion-exclusion,
  `G_min/(G_min-1) * (N-1)/(N-K)`, `G_min - 1` df; an indefinite meat is
  repaired and flagged as described under `pooled`, on Stata's transformed
  design). For clustering on time or on (panel, time), use `model="pooled"`.
- `driscoll_kraay` (needs `time`): see the Driscoll-Kraay section.

**Weights.** `aweight`, `fweight` and `pweight`, constant within panel
(`weights_vary_within_panel` otherwise).

## `model="re"`: Swamy-Arora random effects — `xtreg, re`

**Estimator.** Feasible GLS with the Swamy-Arora variance components. First
the within regression of the regressors that vary within panels gives

    sigma_e^2 = RSS_within / (N - n - k_w)

(`k_w` = number of regressors with within variation, by the same mean-deviated
screen as `fe`; time-invariant columns stay in the model but drop out of this
step). Then the *unweighted* between regression of `ybar_i` on `xbar_i` and a
constant (one row per panel) gives

    sigma_u^2 = max(0, RSS_between / (n - K_b) - sigma_e^2 / T_bar),
    T_bar = n / sum_i (1/T_i)   (harmonic mean of the panel lengths)

with `K_b` the rank of the between regression. This is Stata's default
(`sigma_uT` in the manual; the exact Swamy-Arora estimator when balanced).
With `sa=True` the Baltagi-Chang (1994) unbalanced estimator, Stata's `sa`
option, is used:

    sigma_u^2 = max(0, [SSR*_b - (n - K_b) sigma_e^2] / [N - c_tr]),
    SSR*_b = sum_i T_i (ybar_i - a_b - xbar_i'b_b)^2,
    c_tr = trace{(X'PX)^-1 X'ZZ'X} = trace{(sum_i T_i zbar_i zbar_i')^-1 sum_i T_i^2 zbar_i zbar_i'}

with `(a_b, b_b)` from the between regression weighted by `T_i`, the
regression whose residual sum of squares has exactly the expectation
`(n-K) sigma_e^2 + (N - c_tr) sigma_u^2` (see the unverified list for the
reading of the manual's symbols). In a balanced panel both estimators
coincide. Finally

    theta_i = 1 - sqrt(sigma_e^2 / (T_i sigma_u^2 + sigma_e^2))

and `y_it - theta_i ybar_i` is regressed on `x_it - theta_i xbar_i` and
`1 - theta_i` by QR. If `sigma_u^2` is truncated at zero the GLS reduces to
pooled OLS and a warning says so.

**Inference.** z statistics and a Wald chi2(k) model test, as Stata reports
for `xtreg, re`. `nonrobust` follows the manual: "the coefficient estimates
and the conventional variance-covariance matrix come from an OLS regression
of y*_it on x*_it and the transformed constant 1 - theta_i", i.e.

    V = rmse^2 (X*'X*)^-1,      rmse^2 = RSS* / (N - K)

with `RSS*` the residual sum of squares of the transformed regression. `rmse`
is Stata's `e(rmse)`, "root mean squared error of GLS regression", reported as
`metrics["rmse"]`; it is *not* the within `sigma_e`. (On the Grunfeld data
this gives the standard errors 28.8989 / .0104927 / .0171805; the
`sigma_e^2 (X*'X*)^-1` alternative gives 28.8893 / .0104892 / .0171747.)
`robust` clusters on the panel and `cluster` on the given column, both as
the CR1 sandwich of the transformed regression with `K = k + 1` and
`G/(G-1) * (N-1)/(N-K)`; the cluster column must nest the panels
(`cluster_not_nested` otherwise), as the manual requires for `re` too ("The
panel variable must be nested within the cluster variable because of the
within-panel correlation that is generally induced by the random-effects
transform"). Stata's `xtreg, re` takes a single cluster variable; two columns
(both nesting the panels, inclusion-exclusion as for `fe`) are an OpenEconometrics
extension.

**Metrics:** `r_squared_within`, `r_squared_between`, `r_squared_overall`
(squared correlations on the untransformed data with the GLS slopes),
`sigma_u`, `sigma_e`, `rho`, `theta` (the common value in balanced panels,
`None` otherwise), `rmse`, `n_groups`, `t_min`, `t_avg`, `t_max`.
`extra["variance_components"]` records the method, both variances, the within
and between sums of squares and degrees of freedom and `T_bar`;
`extra["theta"]` the min / 5% / median / 95% / max of `theta_i`;
`extra["ssr_transformed"]` is `RSS*`.

**Tests.** `tests["model"]`: Wald chi2(k). `tests["breusch_pagan"]`: the
Breusch-Pagan Lagrange multiplier test for `sigma_u = 0` in the Baltagi-Li
(1990) unbalanced form used by Stata's `xttest0`, on the pooled OLS residuals
`e`:

    LM = N^2 / (2 (sum_i T_i^2 - N)) * (sum_i (sum_t e_it)^2 / sum e_it^2 - 1)^2

The null is on the boundary of the parameter space, so the reference
distribution is chibar2(01): the reported p-value is half the chi2(1) upper
tail, as Stata prints it (`distribution="chibar2"`, `df=1`).

**Weights.** None (`unsupported_weights`): Stata's `xtreg, re` does not accept
weights. To weight by frequency, expand the rows.

## `model="be"`: between regression — `xtreg, be`

OLS of the panel means `ybar_i` on `xbar_i` and a constant over the `n`
panels; `wls=True` weights each panel by `T_i` (Stata's `wls` option).
Inference is Student t with `n - K` df; `tests["model"]` is F(k, n-K).
Metrics: `r_squared_between` (the R-squared of this regression),
`r_squared_within`, `r_squared_overall`, `rmse`, `df_resid`, `n_groups`,
`t_min/t_avg/t_max`. Weights are not accepted (Stata allows none).
`robust` (HC1 on this `n`-row regression) and `cluster` (a column constant
within panel, `cluster_varies_within_panel` otherwise, `G - 1` df) are OpenEconometrics
extensions: Stata's `xtreg, be` offers only the conventional, bootstrap and
jackknife VCEs.

## `model="fd"`: first differences — `regress D.y D.x`

`y_it - y_i,t-1` on `x_it - x_i,t-1` and a constant (the drift Stata's
`regress D.y D.x` includes), over pairs of *consecutive* periods of the same
panel only. The first observation of every panel and every observation after
a gap are dropped; the count is recorded in `warnings`,
`extra["dropped_for_differencing"]` and `dropped_rows`. Time-invariant
regressors difference to zero and are omitted. Inference is Student t with
`N_d - K` df (`N_d` differenced observations). `robust` clusters on the panel
(xtreg semantics, not HC1), `cluster` on the given column (any column; this is
a `regress`). Weights are those of the current row. Metrics are those of
`regress` (`r_squared`, `adjusted_r_squared`, `rmse`, `df_model`, `df_resid`)
plus the panel structure of the differenced sample. The chart sample's
observed values are the differenced outcome. The drift is always included
(`regress D.y D.x, noconstant` is not available).

## `model="pooled"`: pooled OLS with panel covariances

OLS of `y` on `x` and a constant, `regress`'s metrics plus `n_groups` and
`t_min/t_avg/t_max`; t inference with `N - K` df. `robust` clusters on the
panel (`G - 1` df), `cluster` on the given column(s), `driscoll_kraay` as
below. Being a `regress`, it accepts any cluster column, including two-way
clustering on (panel, time): the Cameron-Gelbach-Miller inclusion-exclusion
meat `M_a + M_b - M_ab` with `G_min/(G_min-1) * (N-1)/(N-K)` and `G_min - 1`
df. That meat need not be positive semidefinite; when it is not, the shared
covariance layer sets its negative eigenvalues to zero (Cameron, Gelbach and
Miller 2011, eq. 2.13), records `inference["psd_adjusted"] = True` and adds a
warning. The fix is applied to the meat of the regression on `[1, X]` (the
parameterization `core.linear_covariance` gives every linear estimator); like
any eigenvalue repair it is not invariant to the units or the level of the
regressors.

## Driscoll-Kraay standard errors — `xtscc` (fe and pooled)

`covariance="driscoll_kraay"` (requires `time`) follows Hoechle's `xtscc`
(Stata Journal 2007; version 1.4 of the ado). With per-observation scores
`s_it = x*_it e_it` (weighted by the estimation weight), the cross-sectional
sums `h_t = sum_i s_it` form a time series whose Newey-West HAC is the meat:

    M = H_0 + sum_{l=1..L} w_l (H_l + H_l'),   H_l = sum_t h_t h_{t-l}',
    V = (X*'X*)^-1 M (X*'X*)^-1 * T/(T-1) * (N-1)/(N-K)

`w_l` are Bartlett weights `1 - l/(L+1)` by default (`kernel` may also be
`truncated`, `parzen` or `quadratic_spectral`); the default bandwidth is
`L = floor(4 (T/100)^(2/9))` with `T` the number of periods (xtscc's
`lag()` default). Periods with gaps are respected: only pairs exactly `l`
periods apart contribute. The small-sample factor `T/(T-1) (N-1)/(N-K)` and
`K = k + 1` (the rank of the transformed regression; the fixed effects are not
counted) are what xtscc applies unless its `ase` option is given; t tests and
the model F use `T - 1` degrees of freedom, as xtscc's `e(df_r)`. The
inference record stores `lags`, `kernel`, `periods` and the factor.

StataNow's own `xtreg, fe vce(dkraay kernel [#])` uses the same factor,
`(n-1)/(n-k) * T/(T-1)` in the manual's notation, but differs in three ways
that OpenEconometrics does not copy: its default is `T - 2` lags (pass `lags=T-2` to
reproduce it), it exists for `fe` only, and "fweights and pweights are not
allowed with vce(dkraay)". OpenEconometrics accepts all three weight types here with
the definitions above (frequency weights replicate rows in the period sums
and in `N`; sampling weights use the same scores as analytic weights), which
is an extension checked against the explicit formula in the tests.

## `model="mle"`: random-effects maximum likelihood — `xtreg, mle`

Gaussian random effects: per panel `y_i ~ N(X_i b, sigma_e^2 I + sigma_u^2 11')`.
With `a_i = sigma_e^2 + T_i sigma_u^2`, `S_i = sum_t r_it` and
`W_i = sum_t (r_it - rbar_i)^2` for `r = y - Xb`, the inverse and determinant
of the panel covariance are closed form and

    ll = sum_i [ -(T_i/2) ln 2pi - ((T_i-1)/2) ln sigma_e^2 - (1/2) ln a_i
                 - W_i / (2 sigma_e^2) - S_i^2 / (2 T_i a_i) ]

costs O(N). The likelihood is maximized over `(b, ln sigma_u, ln sigma_e)` by
Newton-Raphson with the analytic gradient and Hessian (verified against
numerical derivatives in the tests), starting from the Swamy-Arora GLS
estimates.

**Conditioning.** The maximization runs on centered, unit-scale data:
`(x - xbar)/s_x` and `(y - ybar)/s_y`. The estimates, their covariance
(`A V A'` for the linear map `A`) and the log likelihood (`- N ln s_y`) are
transformed back exactly, so the fit is invariant to the units and the level
of every variable and a regressor with a `1e5` (or `1e8`) offset converges in
the same few iterations as any other.

**Ancillary parameters.** `sigma_u` and `sigma_e` are reported as the terms
`/sigma_u` and `/sigma_e` (equation `None`) with delta-method standard errors
`sigma * se(ln sigma)`. Their confidence intervals are the interval of
`ln sigma` transformed back,

    [sigma * exp(-z se_ln), sigma * exp(+z se_ln)],

which is what Stata prints: the manual's example row
`/sigma_u .2485556 .0035017 .2417863 .2555144` is reproduced by this rule and
not by the symmetric interval. Stata shows no z or P>|z| for these rows; the
statistic OpenEconometrics stores is `sigma / se`. `extra["ln_sigma"]` holds the
ln-scale estimates and standard errors. Stata's table also prints `rho` with
a standard error; OpenEconometrics reports `rho` as a metric only.

The covariance is the observed information (`nonrobust`, Stata's default
`vce(oim)`); z inference. Stata 19 also offers `vce(robust)` and
`vce(cluster)` for this model, which OpenEconometrics does not.

Metrics: `sigma_u`, `sigma_e`, `rho`, `log_likelihood`, `aic`, `bic`
(`k = K + 2`), `n_groups`, `t_min/t_avg/t_max`. Tests: `tests["model"]` is
the LR chi2(k) test against the constant-only random-effects model;
`tests["sigma_u"]` the LR test of `sigma_u = 0` against pooled OLS,
chibar2(01) with the chi2(1) tail halved. When the maximum lies at
`sigma_u = 0` the likelihood is flat in `ln sigma_u`; OpenEconometrics then raises
`boundary_solution` and points to `model="re"` or `"pooled"` instead of
reporting a non-converged fit. Weights are not accepted (Stata allows
iweights only, which are not supported).

## `oe.hausman(consistent, efficient, *, sigmamore=False, sigmaless=False, alpha=0.05, force=False)`

Stata's `hausman`. Over the common non-constant terms (ancillary `/...`
terms excluded) of two fits, typically `fe` (consistent) and `re`
(efficient):

    d = b_c - b_e,   V = V_c - V_e,   H = d' V^- d ~ chi2(rank V)

`V_c - V_e` is the variance of `b_c - b_e` only when both are the
conventional (model-based) covariances, so both fits must use
`covariance="nonrobust"` and no sampling weights. Stata refuses otherwise
("hausman cannot be used with vce(robust), vce(cluster cvar), or p-weighted
data") and so does OpenEconometrics (`unsupported_covariance`); `force=True`, Stata's
`force` option, computes the statistic regardless and marks the result
(`forced=True`, with a caveat in `note`).

`V^-` is a generalized inverse from the eigen-decomposition of `V` (scaled to
unit diagonal so the cutoff `1e-12` of the largest eigenvalue is unit free),
`df` is the number of eigenvalues kept. Negative eigenvalues are *kept*, as
Stata's `invsym` keeps them, so `H` can be negative: the result then has
`p_value=None`, `negative_definite=True` and Stata's note (`chi2 < 0: the
model fitted on these data fails to meet the asymptotic assumptions`). With
the default options nothing is rescaled.

`sigmamore` and `sigmaless` follow `help hausman`: both covariance matrices
are based on one disturbance variance, "from the efficient estimator"
(`sigmamore`: `V_c` is multiplied by `s_e^2 / s_c^2`, `V_e` already rests on
`s_e^2`) or "from the consistent estimator" (`sigmaless`: `V_e` is multiplied
by `s_c^2 / s_e^2`). Which scalar is the disturbance variance is also taken
from the help file: "e(sigma_e) is stored after the xtreg command with the fe
or mle option. e(rmse) is stored after the xtreg command with the re option."
OpenEconometrics therefore reads `metrics["sigma_e"]` for `xtreg` fe/mle fits and
`metrics["rmse"]` for `xtreg, re` and for every other estimator. For fe
against re the two differ (`rmse^2 = RSS*/(N-K)` of the GLS regression is not
the within `sigma_e^2`), so `sigmamore` changes the statistic and makes
`V_c - V_e` positive definite: both then equal `rmse^2` times `(X_w'X_w)^-1`
and `(X*'X*)^-1`, whose difference is positive semidefinite. The returned
dict holds `statistic`, `df`, `p_value`, `distribution`, `terms`,
`difference`, `se_difference` (`sqrt(diag V)`, `None` where negative),
`negative_definite`, `note`, `sigma` (the option used), `sigma2` (the two
variances read), `forced`, `alpha`, `reject` and the two estimator labels.

## `oe.xtfmb(...)`: Fama-MacBeth — community `xtfmb`

For every period a cross-sectional OLS of `y` on `x` and a constant gives
`b_t`; the coefficients are `b = mean_t b_t` and the standard errors
`sd(b_t)/sqrt(T)` (t inference with `T - 1` df, `tests["model"]` the
corresponding F). `covariance="hac"` treats the `b_t` as a time series and
applies Newey-West (Bartlett, `lags`, default `floor(4 (T/100)^(2/9))`) times
`T/(T-1)`, which reduces to the plain estimator at `lags=0`; period gaps are
respected. Every period needs more observations than parameters and a
full-rank design (`insufficient_observations` / `singular_design` name the
period). `metrics["r_squared"]` is the average period R-squared; the period
estimates' standard deviations are in `extra`. It takes no weights and no
cluster column.

## Errors

Every invalid input raises `AnalysisError(code, message)`:

| code | when |
| --- | --- |
| `unsupported_covariance` | a covariance the model does not offer; `pweight` with `nonrobust`; `hausman` on a fit with a robust, cluster or Driscoll-Kraay covariance or sampling weights (without `force=True`) |
| `unsupported_weights` | weights with `re`, `be` or `mle` |
| `weights_vary_within_panel` | fe weights that change inside a panel |
| `cluster_not_nested` | fe/re: a cluster column that varies inside a panel |
| `cluster_varies_within_panel` | be: the same condition for the panel-mean regression |
| `insufficient_clusters` | fewer than two clusters |
| `no_within_variation` | fe/re/mle: the outcome is constant inside every panel, or the within regression fits exactly |
| `zero_variance_constant` | fe with `robust`/`cluster`: the constant's sandwich variance `xbar'V xbar` is exactly zero |
| `constant_outcome`, `perfect_fit` | be/fd/pooled/xtfmb: no variation in the outcome, or residuals that are rounding noise |
| `invalid_spec` | `lags`/`kernel` without Driscoll-Kraay, `fd` or Driscoll-Kraay without `time`, `sa` outside `re`, `wls` outside `be`, `x` not a list |
| `insufficient_observations`, `insufficient_periods`, `empty_sample` | too few rows, panels, periods or consecutive pairs |
| `missing_values`, `repeated_time_values` | data problems named in the message |
| `boundary_solution`, `nonconvergence` | mle: `sigma_u = 0`, or Newton-Raphson stopped elsewhere |
| `no_common_terms`, `missing_sigma`, `singular_covariance` | hausman |

Unknown options, an `intercept=False` request, a third cluster column, an
`iweight` or a missing panel column are rejected by the registry when the
specification is built (a pydantic `ValidationError`), before any data are
read.

## Performance

One million rows, 100,000 panels of 10 periods, eight regressors (Apple
M-series laptop, six Torch threads, steady state after a small warm-up fit;
minimum of five runs, measured while the machine was busy with other work):

| model / covariance | seconds |
| --- | --- |
| fe nonrobust / robust / cluster / Driscoll-Kraay / fweight | 0.27 / 0.22 / 0.24 / 0.34 / 0.41 |
| re nonrobust / sa / robust | 0.34 / 0.33 / 0.35 |
| pooled nonrobust / robust / two-way cluster / Driscoll-Kraay | 0.17 / 0.18 / 0.22 / 0.22 |
| be / be wls | 0.15 / 0.15 |
| fd nonrobust / robust | 0.25 / 0.26 |
| mle (Newton, analytic Hessian) | 0.51 |
| xtfmb / xtfmb hac (10 periods) | 0.25 / 0.25 |

Two million rows take 0.71 s (fe) and 0.63 s (re): the cost is linear in `N`.
The fe path keeps two `N x (k+1)` float64 blocks alive (the design and one
buffer that holds the within-transformed regressors and is reused for the
pooled F-test regression) plus the QR factor; peak resident memory of a fresh
process for the one-million-row fit is about 1 GB (0.85 to 1.1 GB across
runs), of which 0.3 GB is the pandas table itself.

## Deliberate differences from Stata

- Degenerate fits raise errors (see above) where Stata prints zero or missing
  standard errors.
- `be` offers `robust` (HC1) and `cluster`; `fd` treats `robust` as clustering
  on the panel (Stata's `regress D.y D.x, vce(robust)` is HC1); the family-wide
  rule is that `robust` means `vce(cluster panelvar)` wherever rows are
  panel observations.
- Driscoll-Kraay follows `xtscc` (default lags, pooled model, weights) rather
  than StataNow's `vce(dkraay)`; see that section.
- An indefinite two-way cluster covariance (`pooled`) is repaired by the
  eigenvalue fix and flagged, rather than reported with missing values.
- Not offered: `xtreg, fe absorb()`, `vce(hc2 clustvar)` / `vce(hc3 ...)`,
  bootstrap and jackknife VCEs, `vce(robust)`/`vce(cluster)` for `mle`,
  iweights for `mle`, `xtreg, pa` and `xtreg, cre`, `noconstant` for `fd`.

## What has been checked, and how

`provenance["stata_parity_validated"]` is `False` for every result: nothing
here has been run side by side with Stata.

**Checked against the text of the Stata 19 [XT] xtreg manual (Options, Stored
results, Methods and formulas) and `help hausman`:** the fe transformed
regression and `_cons`; `sigma_u` as the standard deviation of `u_i` over
panels; `e(corr)` as `corr(u_i, x_it b)`; the F test for `u_i = 0`; the three
R-squared definitions; `vce(robust)` = `vce(cluster panelvar)` for fe and re;
the panel-nesting rule for cluster VCEs (fe and re, including multiway); which
weights each model takes; the harmonic-mean Swamy-Arora formulas and
`theta_i`; the re conventional VCE as the OLS VCE of the transformed
regression and `e(rmse)`; `be wls`; the mle likelihood, LR tests and the
log-transformed `/sigma` intervals (reproduced on the manual's printed
example); the scalars `hausman` reads for `sigmamore`/`sigmaless`; StataNow's
Driscoll-Kraay factor and default lag.

**Digits reproduced** (values quoted in our review as Stata's output for the
10-firm Grunfeld data, `xtreg invest value capital`; we did not run Stata
ourselves; asserted in `tests/test_econ_panel_oracle.py`): fe `sigma_u`
85.732315, `sigma_e` 52.767964, `rho` .72525012, F(9, 188) = 49.18, R-squared
.7668 / .8194 / .8060, `corr(u_i, Xb)` -0.1517; re `sigma_u` 84.20095
(`sigma_u^2` = 7089.80 and `sigma_e^2` = 2784.46 are also the textbook values
for these data), theta .8612, standard errors 28.8989 / .0104927 / .0171805.
The `/sigma_u` and `/sigma_e` rows printed in the manual's `xtreg, mle`
example are reproduced by the interval rule (arithmetic only).

**Checked only against independent oracles written by us** (NumPy dummy-variable
least squares, explicit Swamy-Arora/Baltagi-Chang algebra, explicit CR1 /
Cameron-Gelbach-Miller / Driscoll-Kraay sums, statsmodels OLS/WLS with
cluster, HC1 and HAC covariances, a dense-matrix likelihood maximized by
SciPy with an analytic Hessian): every estimator, covariance and weight
option on this page.

**Not verified against Stata, in order of how much they could matter:**

1. `xtreg, re sa`: the manual writes `SSR*_b = sum_i T_i (ybar_i - a_b -
   xbar_i b_b)^2` with `(a_b, b_b)` "coefficient estimates from the between
   regression", the same symbols it uses for the unweighted between
   regression. OpenEconometrics uses the coefficients of the `T_i`-weighted between
   regression, for which the stated expectation `(n-K) sigma_e^2 + (N - c_tr)
   sigma_u^2` holds exactly (Baltagi and Chang 1994). Under the literal
   reading (unweighted coefficients, `T_i`-weighted residuals) `sigma_u^2`
   differs in unbalanced panels (0.3500 against 0.3482 on one test panel).
   Balanced panels are unaffected.
2. The small-sample factor of the cluster/robust VCE for `re`: OpenEconometrics uses
   `G/(G-1) * (N-1)/(N-K)` with `K = k + 1` on the transformed regression (the
   `regress` factor; the manual only says the sandwich is computed "for the
   coefficients estimated in this regression").
3. Two cluster columns for fe (both nesting the panels): OpenEconometrics applies
   one factor `G_min/(G_min-1) * (N-1)/(N-K)` to the inclusion-exclusion meat
   and uses `G_min - 1` df; the manual refers to [R] regress for Stata's
   multiway computation, which we have not read (Stata may scale each
   component separately and treats a non-positive-semidefinite result in its
   own way). For re, two columns are an extension with no Stata counterpart.
4. Driscoll-Kraay for fe: `K = k + 1` and `T - 1` inference df are xtscc's
   conventions (read from its ado file); StataNow's degrees of freedom for
   `vce(dkraay)` are not stated in the passages we read.
5. Weighted fe statistics: R-squared within/overall and `corr_u_xb` are
   weighted squared correlations, R-squared between and `sigma_u` are
   unweighted over the `n` panels. The manual defines them without weights.
6. `hausman`: the direction of the rescaling (`sigmamore` multiplies `V_c` by
   `s_e^2/s_c^2`) follows the help file's wording; the resulting statistics
   have not been compared with Stata's output.
7. `xttest0`: the Baltagi-Li unbalanced LM formula and the halved p-value are
   the textbook form; Stata's `xttest0` entry was not read.
8. `xtfmb` and `xtscc` are community commands; their conventions were taken
   from the ado files and papers, not from running them.
