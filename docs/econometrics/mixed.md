# Mixed-effects and GEE models: `mixed`, `melogit`, `meprobit`, `mepoisson`, `xtlogit`, `xtprobit`, `xtpoisson`, `xtgee`

Models for grouped data — pupils in classes in schools, repeated
measurements on patients, firms observed over years — where observations of
the same group are correlated. Two approaches are covered:

* **Subject-specific (mixed-effects) models** put random effects into the
  linear predictor: `mixed` (linear, ML or REML), the random-intercept
  generalized linear mixed models `melogit`, `meprobit`, `mepoisson`, and the
  panel commands `xtlogit`, `xtprobit`, `xtpoisson` with `model="re"`.
  Conditional fixed-effects versions (`model="fe"`) remove the group effects
  by conditioning instead.
* **Population-averaged models** (`xtgee`, and `model="pa"` of the panel
  commands) model the marginal mean and treat the within-group correlation
  as a nuisance described by a working correlation matrix.

Everything on this page is implemented in OpenEconometrics on float64 PyTorch
tensors; no estimation library runs at fit time. Group work is `index_add_`
on group codes, per-group linear algebra is batched over groups on q-by-q
matrices, and no N-by-N matrix is ever formed. Timings are at the end.

| Stata / SPSS | OpenEconometrics |
| --- | --- |
| `mixed y x || id:` | `oe.mixed(data=df, y="y", x=["x"], group="id")` |
| `mixed y x || id: x, covariance(unstructured)` | `..., random=["x"], covstructure="unstructured"` |
| `mixed y x || school: || class:` | `..., group=["school", "class"]` |
| `mixed y x || id:, reml` / SPSS `MIXED ... /METHOD=REML` | `..., method="reml"` |
| `mixed y x || id:, vce(robust)` | `..., covariance="robust"` |
| `predict u*, reffects` / `predict f, fitted` / `predict xb, xb` | `oe.mixed_predict(fit, df, kind="reffects" / "fitted" / "xb")` |

## `mixed`: linear mixed models

### Model

With one grouping level, group `j` with `n_j` observations follows

    y_j = X_j b + Z_j u_j + e_j,     u_j ~ N(0, G),   e_j ~ N(0, sigma_e^2 I),

so `Var(y_j) = V_j = Z_j G Z_j' + sigma_e^2 I`. `Z` holds the random slopes
(`random=[...]`, numeric columns) followed by the random intercept (`_cons`,
dropped with `random_intercept=False`). With two nested levels
(`group=["school", "class"]`, top level first) the top level adds a random
intercept `v_s ~ N(0, g)` shared by all classes of school `s`; a class label
is read within its school, so class 1 of school A and class 1 of school B are
different classes (Stata's `|| school: || class:`). Random slopes are
available at the lowest level only.

`covstructure` sets the covariance of the lowest-level random effects:

| `covstructure` | G | parameters |
| --- | --- | --- |
| `independent` (default, Stata's) | diagonal, distinct variances | q |
| `unstructured` | any positive definite matrix | q(q+1)/2 |
| `exchangeable` | common variance, common covariance | 2 |
| `identity` | common variance, no covariance | 1 |

### Estimator

For given variance parameters the fixed effects are the GLS estimate
`b = (X'V^-1 X)^-1 X'V^-1 y` and are profiled out. With `p_R = p` for REML
and 0 for ML the (restricted) log likelihood is

    l = -(N - p_R)/2 ln(2 pi) - 1/2 ln|V| - 1/2 r'V^-1 r  [- 1/2 ln|X'V^-1 X|]

(the REML form is Harville's restricted likelihood without the `ln|X'X|`
constant, the convention of Stata and statsmodels). It is maximized over
Stata's metric — log standard deviations (`independent`, `identity`), the
log-Cholesky factor (`unstructured`), log-sd and a transformed correlation
(`exchangeable`), the top-level log sd and `ln sigma_e` — by BFGS with the
**analytic gradient**, polished by Newton steps whose Hessian is the Ridders
numerical derivative of that gradient (recorded in `provenance`).

Efficiency comes from the Woodbury identity. With `D = G/sigma_e^2 = L L'`
and `M_j = I + L'Z_j'Z_j L` (q-by-q),

    ln|V_j| = n_j ln sigma_e^2 + ln|M_j|
    u'V_j^-1 w = (u'w - (L'Z_j'u)' M_j^-1 (L'Z_j'w)) / sigma_e^2

so the likelihood depends on the data only through `X'X`, `X'y`, `y'y` and the
per-group cross products `Z_j'Z_j`, `Z_j'[1 X y]`, `1'[1 X y]`, accumulated
once. Each evaluation is then one batched Cholesky of the `[groups, q, q]`
matrices `M_j`. The top level of a two-level model is a rank-one update per
school (`ln|A + g 11'| = ln|A| + ln(1 + g 1'A^-1 1)`).

### Inference

* Fixed effects: `nonrobust` (default) is the model-based `(X'V^-1 X)^-1`;
  coefficient tests are z tests (Stata's default; `dfmethod()` is not
  offered).
* Random-effects parameters are reported on the variance scale as ancillary
  terms `/var(_cons[school])`, `/var(x[class])`, `/cov(x,_cons[class])` and
  `/var(Residual)`. Their covariance is the inverse observed information of
  the (restricted) log likelihood with `b` profiled out, carried to the
  variance scale by the delta method. Confidence intervals of variances are
  `exp(ln v -/+ z se(ln v))` with `se(ln v) = se(v)/v` (asymmetric, positive),
  covariances get the symmetric Wald interval — the "Random-effects
  Parameters" table Stata prints. The z statistics of these rows are
  `estimate / se` (Stata prints none).
* The default covariance matrix is block diagonal between fixed effects and
  variance parameters: Stata's [ME] mixed Methods and formulas treat the
  covariance of `b` with the variance parameters as zero because the two are
  asymptotically uncorrelated.
* `robust` follows the same Methods and formulas (scores of every parameter,
  aggregated at the top-level clusters; [P] _robust): with the block-diagonal
  matrix `D`
  above and the per-top-level-group scores `s_j` of `(b, theta)`,
  `V = G/(G-1) D [sum_j s_j s_j'] D`. The b-scores are `X_j'V_j^-1 r_j`, the
  theta-scores the group terms of the analytic gradient at fixed `b` (they sum
  to the gradient by the envelope theorem). Every parameter, variance
  components included, gets a robust standard error and the matrix is no
  longer block diagonal (Stata prints "Robust" over the random-effects table
  as well). `cluster` uses a column that nests the top-level groups
  (`cluster_not_nested` otherwise). As in Stata, robust and cluster need
  `method="ml"`: the restricted likelihood does not separate by groups.
* `tests["model"]`: Wald chi2 of the fixed slopes. `tests["lr_vs_linear"]`:
  LR test against the linear regression on the same regressors with the same
  criterion (ML or REML): `chibar2(01)` (chi2(1) tail halved) with one
  variance component, the conservative chi2 with as many degrees of freedom
  as variance components otherwise ("LR test is conservative", as Stata
  notes). It is not reported under `robust`/`cluster`, where Stata prints a
  log pseudolikelihood and no LR test.

### Result

`metrics`: `log_likelihood` (the restricted log likelihood under REML), `aic`,
`bic` (all parameters, N observations), `n_groups`, `group_size_min/avg/max`
(top level) and, for random-intercept-only models, `icc` (Stata's `estat icc`:
`var(_cons)/(var(_cons) + var(Residual))`; with two levels the top-level ICC,
both levels in `extra["icc"]`, e.g. `{"school": ..., "class|school": ...}`).
`extra`: `method`, `covstructure`, `G` (the lowest-level covariance matrix),
`residual_variance`, `levels` (groups and sizes per level), `theta` and
`theta_std_error` (estimates in the optimization metric) and the linear-model
log likelihood.

Frequency weights (`weight_type="fweight"`) replicate rows within their
group; results equal those of the expanded data. Other weights are not
offered (Stata's level-specific sampling weights are out of scope).

### Predictions

`oe.mixed_predict(result, data, kind=...)` returns tables:

* `"xb"`: `X b` (Stata's `predict, xb`), columns `row`, `xb`; the outcome may
  be missing;
* `"reffects"`: the BLUPs `u_j = G Z_j'V_j^-1 (y_j - X_j b)` (and
  `v_s = g 1'V_s^-1 r_s`), long format with columns `level`, `group`, `effect`,
  `blup` (Stata's `predict, reffects`);
* `"fitted"`: `X b + Z u (+ v)` (Stata's `predict, fitted`).

BLUPs are computed from the outcomes of each group's rows in `data` at the
estimated `(b, theta)`; `data` may hold a single group.

### Example

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(0)
g = np.repeat(np.arange(48), 9)                       # 48 pigs, 9 weeks
week = np.tile(np.arange(1, 10), 48).astype(float)
df = pd.DataFrame({"id": g, "week": week})
df["weight"] = (19 + 6.2 * week + rng.normal(0, 2.5, 48)[g]
                + rng.normal(0, 0.6, 48)[g] * week + rng.normal(0, 1.2, len(df)))

fit = oe.mixed(data=df, y="weight", x=["week"], group="id", random=["week"],
               covstructure="unstructured")           # mixed weight week || id: week, cov(un)
print(fit.summary())
blups = oe.mixed_predict(fit, df, kind="reffects")
```

Output (estimate, standard error, 95% interval):

```
Intercept                 19.0674    0.3185 [  18.4431,   19.6917]
week                       6.3089    0.0894 [   6.1337,    6.4840]
/var(week[id])             0.3566    0.0783 [   0.2319,    0.5482]
/var(_cons[id])            4.0239    0.9962 [   2.4770,    6.5371]
/cov(week,_cons[id])       0.3397    0.1997 [  -0.0517,    0.7311]
/var(Residual)             1.6028    0.1237 [   1.3779,    1.8645]
log_likelihood -857.659, aic 1727.317, bic 1751.728, 48 groups of 9
LR test vs. linear model: chi2(3) = 811.54 (conservative)
```

With `covariance="robust"` the standard errors become 0.3219, 0.0903,
0.0570, 0.9867, 0.1880 and 0.1179 and `tests` holds only `model`.

### Limitations

Random slopes at the lowest level only; the top level of a two-level model has
a random intercept only; at most two levels; no residual-error structures
(`residuals()`), no `dfmethod()`, fweights only. A variance component whose
estimate is at zero (sd below 1e-4 times the residual sd), a singular
correlation matrix or a variance whose interval would not be finite is refused
as `boundary_solution` rather than reported with meaningless standard errors.
A constant outcome (`constant_outcome`), regressors that reproduce the outcome
exactly (`perfect_fit`, checked before the optimization) and random-effects
columns that are collinear — a constant random slope duplicates the random
intercept (`collinear_random_effects`; slopes are screened after centring, so
a slope on calendar time is not mistaken for the intercept) — are refused up
front. A random slope with a huge level under an independent covariance is a
legitimate but very ill-conditioned model; if it fails to converge the message
advises centring.

## `melogit`, `meprobit`, `mepoisson`: random-effects GLMMs

| Stata / SPSS | OpenEconometrics |
| --- | --- |
| `melogit y x || g:` | `oe.melogit(data=df, y="y", x=["x"], group="g")` |
| `meprobit y x || g:` | `oe.meprobit(data=df, y="y", x=["x"], group="g")` |
| `mepoisson y x, exposure(t) || g:` | `oe.mepoisson(data=df, y="y", x=["x"], group="g", exposure="t")` |
| `melogit y x || g: x` (independent slope) | `..., random=["x"]` |
| `..., intpoints(12)` / `intmethod(mcaghermite)` | `intpoints=12` / `intmethod="mcaghermite"` |
| `..., vce(robust)` / `vce(cluster c)` | `covariance="robust"` / `cluster="c"` |
| SPSS `GENLINMIXED` with a random intercept | the same calls |

### Model

    g(E[y_ij | u_j]) = x_ij'b + offset_ij + u_cons,j + u_slope,j z_ij,

with the logit, probit or log link and binary (coded 0/1) or count outcomes.
The random intercept `u_cons,j ~ N(0, var(_cons))` and, with `random=["z"]`,
an independent random slope `u_slope,j ~ N(0, var(z))` (Stata's default
independent covariance; at most one slope).

### Estimator

The group likelihood `L_j = int prod_i f(y_ij | u) phi(u) du` is evaluated by
Gauss-Hermite quadrature on the **standardized** effects `v = u / sd`
(product rule over two dimensions with a slope):

    L_j ~= sum_k 2^(q/2) |tau_j| W_k exp(a_k'a_k) phi(v_jk) prod_i f(y_ij | v_jk),
    v_jk = mu_j + sqrt(2) tau_j a_k.

`intmethod="mvaghermite"` (default, Stata's) takes `mu_j` and `tau_j tau_j'`
as the posterior mean and covariance of `v_j` (fixed-point iteration from the
posterior mode); `"mcaghermite"` uses the posterior mode and curvature;
`"ghermite"` is the non-adaptive rule. `intpoints` (default 7, Stata's) is the
number of nodes per dimension. The adaptive parameters are held fixed while
Newton-Raphson maximizes the quadrature sum with its **analytic gradient and
Hessian** (posterior-weighted sums of the per-node derivatives, all
`index_add_` over `[N, nodes]` arrays) and are recomputed at the new estimates
until those move by less than 1e-8, so the final adaptive parameters belong to
the final estimates (Stata also updates them during the iterations; its xt
commands freeze them once the log likelihood changes by less than 1e-6, which
explains 7th-digit differences, see "Stata parity"). The starting values are
the pooled model and the best of a small grid of standard deviations;
regressors are centred internally and the constant is mapped back exactly.

### Inference and output

`nonrobust` (default) is the inverse observed information; `robust` is the
sandwich of the group scores clustered on the groups with `G/(G-1)` (Stata's
`vce(robust)` for the me commands clusters on the highest level); `cluster`
needs a column that nests the groups. Fixed effects have z tests; the
variances are `/var(_cons[g])` and `/var(z[g])` with delta-method standard
errors from the log-sd metric and intervals `exp(ln v -/+ z se(ln v))`.

`metrics`: `log_likelihood`, `aic`, `bic`, `n_groups`,
`group_size_min/avg/max` and, for random-intercept logit/probit models,
`icc = var(_cons)/(var(_cons) + pi^2/3)` (logit) or `/(var(_cons) + 1)`
(probit), the latent-variable ICC of Stata's `estat icc`. `tests["model"]` is
the Wald chi2 of the slopes and, with the default covariance,
`tests["lr_vs_pooled"]` the LR test against the pooled logit/probit/Poisson
model: `chibar2(01)` for the random intercept, the conservative chi2(2) with a
slope (the layout of Stata's [ME] melogit output, "LR test vs. logistic
model"). Under `robust`/`cluster` the LR test is omitted, as for `mixed`.
`extra` holds the integration settings, the pooled log likelihood and the ln
sd estimates with their standard errors.

A constant outcome (all 0, all 1, all-zero counts) raises `constant_outcome`
and a constant random slope `collinear_random_effects` before estimation. A
regressor that separates a binary outcome raises `separation_detected`
(detected in the pooled starting model); a variance estimated at zero (sd
below 1e-4 on the linear-predictor scale) raises `boundary_solution` — fit the
pooled model instead. The quadrature arrays hold `N x intpoints^q` values;
more than 2.5e7 raises `design_too_large` (lower `intpoints`). Weights are not
supported (Stata's level weights are out of scope).

### Example

```python
rng = np.random.default_rng(0)
g = np.repeat(np.arange(100), 10)
df = pd.DataFrame({"g": g, "x": rng.normal(size=1000)})
eta = -0.5 + 0.8 * df.x + rng.normal(size=100)[g]
df["y"] = (rng.random(1000) < 1 / (1 + np.exp(-eta))) * 1.0
fit = oe.melogit(data=df, y="y", x=["x"], group="g")        # melogit y x || g:
```

gives `Intercept -0.3956 (0.1201)`, `x 0.6630 (0.0850)`,
`/var(_cons[g]) 0.9029 (0.2324)`, log likelihood -619.644, `icc` 0.215 and
`LR test vs. logistic model: chibar2(01) = 60.12`.

## `xtlogit`, `xtprobit`, `xtpoisson`: random, fixed and population-averaged effects

| Stata (after `xtset id year`) | OpenEconometrics |
| --- | --- |
| `xtlogit y x, re` | `oe.xtlogit(data=df, y="y", x=["x"], panel="id")` |
| `xtlogit y x, fe` | `oe.xtlogit(..., model="fe")` |
| `xtlogit y x, pa corr(ar 1) vce(robust)` | `oe.xtlogit(..., time="year", model="pa", corr="ar1", covariance="robust")` |
| `xtprobit y x, re intpoints(20)` | `oe.xtprobit(..., intpoints=20)` |
| `xtpoisson y x, re exposure(t)` | `oe.xtpoisson(data=df, y="y", x=["x"], panel="id", exposure="t")` |
| `xtpoisson y x, re normal` | `oe.xtpoisson(..., normal=True)` |
| `xtpoisson y x, fe vce(robust)` | `oe.xtpoisson(..., model="fe", covariance="robust")` |

### `model="re"` (default)

* **Logit and probit**: `Pr(y_it = 1 | u_i) = F(x_it'b + u_i)`, `u_i ~ N(0,
  sigma_u^2)`, integrated by the adaptive quadrature of the GLMM section with
  Stata's xt defaults (`intpoints=12`, `intmethod="mvaghermite"`). Reported as
  Stata does: the ancillary `/lnsig2u = ln sigma_u^2` (estimated parameter,
  observed-information standard error), `metrics["sigma_u"]` and
  `metrics["rho"] = sigma_u^2 / (sigma_u^2 + c)` with `c = pi^2/3` (logit) or
  1 (probit); `extra["sigma_u"]` and `extra["rho"]` hold their delta-method
  standard errors and the intervals obtained by transforming the endpoints of
  the `/lnsig2u` interval. `tests["rho"]` is the LR test of rho = 0 against
  the pooled model (`chibar2(01)`).
* **Poisson** (gamma random effect, Stata's default): `y_it | nu_i ~
  Poisson(nu_i lambda_it)` with `nu_i ~ Gamma(1/alpha, 1/alpha)` (mean 1,
  variance alpha). The panel likelihood is closed form (Hausman, Hall and
  Griliches 1984): with `a = 1/alpha`, `Y_i = sum_t y_it`, `L_i = sum_t lambda_it`,

      ln L_i = sum_t [y_it eta_it - ln y_it!] + ln Gamma(Y_i + a) - ln Gamma(a)
               - a ln(1 + L_i/a) - Y_i ln(L_i + a),

  maximized over `(b, ln alpha)` by Newton-Raphson with the analytic gradient
  and Hessian. Reported: `/lnalpha`, `metrics["alpha"]` (with se and interval
  in `extra`), `tests["alpha"]` = LR test of alpha = 0 against the pooled
  Poisson model (`chibar2(01)`). `normal=True` replaces the gamma by a normal
  random intercept (quadrature; reported as `/lnsig2u`, sigma_u).
* Covariance: `nonrobust` (OIM), `robust` (panel-clustered sandwich with
  `G/(G-1)`), `cluster` (a column nesting the panels). The model test is the
  Wald chi2 (Stata's default; `lrmodel` is not offered); the LR tests of
  rho, sigma_u and alpha are reported with the default covariance only.

### Stata parity: the ships example

Stata's [XT] xtpoisson manual fits the ships data of McCullagh and Nelder
(1989, Table 6.2; 34 observations with positive service, 5 ships):
`xtpoisson accident op_75_79 co_65_69 co_70_74 co_75_79, exp(service)` with
`re`, `re normal`, `fe` and `pa vce(robust)`. With those data reconstructed in
`tests/test_econ_mixed_oracle.py`, OpenEconometrics reproduces every printed number:
log likelihoods -74.811217 (gamma), -74.780982 (normal), -54.641859 (fe); the
IRRs and their standard errors; `/lnalpha` -2.368406 (.8474597) with interval
[-4.029397, -.7074155] and alpha .0936298; `/lnsig2u` -2.351868 (.8586262)
and sigma_u .3085306; Wald chi2(4) 50.90 / 50.95 / 48.44 / 252.94; LR tests
10.61 and 10.67. The normal-RE and pa results agree to within 2-4 units of the
7th significant digit — Stata freezes its adaptive quadrature parameters once
the log likelihood changes by less than 1e-6 and stops xtgee at a coefficient
change of 4.4e-7 (its printed iteration log), while OpenEconometrics iterates to
convergence; all other numbers agree to the printed rounding. The pa example
(exchangeable, unbalanced panels of 7, 7, 7, 7 and 6, five clusters) confirms
the pooled exchangeable estimator and the `G/(G-1)` factor of the semi-robust
covariance. `provenance["stata_parity_validated"]` nevertheless stays `False`
(one published example is not a validation of every option).

### `model="fe"`

* **Logit**: the conditional logit likelihood (conditioning on the number of
  positive outcomes per panel; the recursive kernel of the discrete family's
  `clogit`). Panels with all-positive or all-negative outcomes are dropped and
  counted (`metrics["n_groups_dropped"]`, a warning); regressors constant within
  panels are omitted; there is no constant. `nonrobust` only (Stata's
  `xtlogit, fe` offers OIM, bootstrap and jackknife).
* **Poisson**: the conditional (multinomial) likelihood given the panel totals,

      ln L_i = ln Y_i! - sum_t ln y_it! + sum_t y_it ln p_it,  p_it = lambda_it / L_i,

  identical in `b` to Poisson regression with panel dummies. Panels with
  all-zero outcomes are dropped and counted. `nonrobust` or `robust`
  (panel-clustered, `G/(G-1)`).
* `xtprobit` has no fixed-effects model (as in Stata): `invalid_spec`.
* `tests["model"]` as Stata reports it: `xtlogit, fe` the LR chi2 against
  `b = 0` (`e(chi2type) = LR`; the null is the conditional likelihood at
  `b = 0`, `-sum_i ln C(n_i, k_i)`, and `metrics["pseudo_r_squared"] =
  1 - ll/ll_0` is Stata's `e(r2_p)`); `xtpoisson, fe` the Wald chi2 of the
  coefficients (`e(chi2type) = Wald`).

### `model="pa"`

Stata's `xtlogit, pa` is `xtgee, family(binomial) link(logit)`; `xtprobit, pa`
uses the probit link and `xtpoisson, pa` the Poisson family with log link. The
working correlation is `corr` (default exchangeable; `corr_order`, `force` as
in `xtgee`), the covariance conventional (scale 1) or `robust`.

## `xtgee`: generalized estimating equations

| Stata / SPSS | OpenEconometrics |
| --- | --- |
| `xtgee y x` (gaussian, exchangeable) | `oe.xtgee(data=df, y="y", x=["x"], panel="id")` |
| `xtgee y x, family(binomial) link(logit) corr(exchangeable) vce(robust)` | `..., family="binomial", covariance="robust"` |
| `xtgee y x, family(poisson) corr(ar 1)` | `..., time="t", family="poisson", corr="ar1"` |
| `xtgee y x, corr(stationary 2)` / `corr(nonstationary 2)` | `corr="stationary", corr_order=2` / `corr="nonstationary", corr_order=2` |
| `xtgee y x, corr(unstructured)` | `corr="unstructured"` (needs `time`) |
| `xtgee ..., nmp` / `scale(x2)` / `scale(dev)` / `force` | `nmp=True` / `scale="x2"` / `scale="dev"` / `force=True` |
| SPSS `GENLIN ... /REPEATED SUBJECT=id CORRTYPE=EXCHANGEABLE` | `oe.xtgee(..., corr="exchangeable", covariance="robust")` |

### Model and estimator

    g(E[y_it]) = x_it'b + offset_it,   Var(y_it) = phi V(mu_it),   Corr(y_i) = R_i(alpha).

Families `gaussian`, `binomial` (0/1 outcomes or proportions), `poisson`,
`gamma`, `nbinomial` (variance `mu + k mu^2`, `nbk = k`, default 1) and
`igaussian` with their canonical links by default (identity, logit, log,
reciprocal, log, inverse squared); other links where valid. The estimating
equations `sum_i D_i'V_i^-1 (y_i - mu_i) = 0` with `V_i = phi A_i^1/2 R_i
A_i^1/2` are solved as in Stata: starting from the independence GLM, one
Fisher-scoring step for `b`,

    b <- b + (sum_i xt_i'R_i^-1 xt_i)^-1 sum_i xt_i'R_i^-1 r_i,
    xt_it = x_it (dmu/deta) / sqrt(V(mu_it)),   r_it = (y_it - mu_it) / sqrt(V(mu_it)),

alternates with moment estimates of the correlation parameters until the
largest relative change of `b` falls below 1e-10. The estimators are those
printed in Stata's [XT] xtgee Methods and formulas ("Correlation structures"),
with `G` panels of sizes `n_i`, `N = sum n_i` and positions `t = 1..n_i`
within a panel:

| `corr` | estimator |
| --- | --- |
| `exchangeable` | `a = [sum_i sum_{t != s} r_it r_is / sum_i n_i (n_i - 1)] / [sum_i sum_t r_it^2 / N]` |
| `ar1` | `a = [sum_i (1/n_i) sum_t r_it r_i,t+1] / [sum_i (1/n_i) sum_t r_it^2]`, `R_ts = a^|t-s|` |
| `stationary` (m) | `a_k = [sum_i (1/n_i) sum_t r_it r_i,t+k] / [sum_i (1/n_i) sum_t r_it^2]`, `k <= m`; zero beyond |
| `nonstationary` (m) | `a_ts = [sum_i r_it r_is / N_ts] / [(1/G) sum_i (1/n_i) sum_t r_it^2]`, `0 < |t-s| <= m`; zero beyond |
| `unstructured` | as nonstationary for every pair of positions |
| `independent` | `R = I` |

`N_ts` is the number of panels observing positions `t` and `s`. The ar and
stationary estimators are Yule-Walker moments: the lag-k sums are divided by
`n_i`, not `n_i - k` (so for balanced panels the AR(1) estimate is
`sum r_t r_t+1 / sum r_t^2`), and every panel gets the same total weight.
None of the estimators depends on `nmp`, which changes only the scale `phi`.
As in Stata, the working correlation of a panel with `n_i` observations is the
upper-left `n_i x n_i` block of one `max(n_i) x max(n_i)` matrix (`e(R)`), so
the correlation is indexed by position within the panel, not by calendar
period, and only panels with `n_i > g` enter a lag-dependent structure:
`ar1` (g = 1), `stationary` and `nonstationary` (g = `corr_order`) drop
shorter panels with a warning (`insufficient_panel_length` if none is left);
exchangeable and unstructured keep every panel.

`R_i^-1` is applied in closed form for exchangeable (`[v - c_i 1 1'v]/(1-a)`,
`c_i = a/(1 + (n_i - 1) a)`) and AR(1) (tridiagonal), and for the patterned
structures by the inverse of the leading `n_i x n_i` block of the Cholesky
factor of `R`, formed once per distinct panel size and applied to all panels of
that size in one batched product (no padding to `max(n_i)`). The timed
structures need `time` and equally spaced observations within panels
(`unequal_spacing` otherwise; `force=True` treats the observations of each
panel as consecutive, Stata's `force`). An estimated correlation matrix that
is not positive definite raises `working_correlation_not_pd`; a constant
outcome raises `constant_outcome` and an exact fit (all Pearson residuals
zero) `perfect_fit`.

### Inference and output

`nonrobust` (Stata's conventional) is `phi (sum_i xt_i'R_i^-1 xt_i)^-1` with
`phi = 1` for binomial, Poisson and negative binomial and the Pearson estimate
otherwise (`scale` overrides: `"x2"`, `"dev"` or a number; `nmp` divides by
`N - p`); `robust` is the semi-robust sandwich clustered on the panels,
`G/(G-1) B^-1 [sum_i s_i s_i'] B^-1` with `s_i = xt_i'R_i^-1 r_i` (Stata's
[P] _robust convention; confirmed by the ships example above). z tests.
`nbk` is accepted with `family="nbinomial"` only.
`metrics`: `n_groups`, `group_size_min/avg/max`, `scale`, `pearson_chi2`,
`deviance`; `tests["model"]`: Wald chi2 of the slopes; `extra`: family, link,
corr, `alpha` (exchangeable, ar1, stationary), the working correlation matrix
(up to 50 periods), the Pearson dispersion and the iteration count. Weights
are not supported.

## Performance

Synthetic data, Apple-silicon laptop, float64 CPU, including sample
preparation and result assembly (measured in the verification pass; panels of
random size around 10 unless noted):

| model | data | time |
| --- | --- | --- |
| `mixed`, random intercept, 10 regressors (ML / REML / robust) | 1,000,000 rows, 100,000 groups | 0.9 / 0.4 / 0.4 s |
| `mixed`, unstructured random slope | 1,000,000 rows, 100,000 groups | 1.5 s |
| `mixed`, two levels / two levels + slope, robust | 1,000,000 rows, 10,000 / 100,000 groups | 0.6 / 2.2 s |
| `melogit` / `meprobit` / `mepoisson`, 7 points, 5 regressors | 100,000 rows, 10,000 groups | 1.1 / 0.8 / 0.5 s |
| `melogit` with a random slope (49 nodes) / robust | 100,000 rows, 10,000 groups | 4.2 / 0.6 s |
| `xtgee` gaussian exchangeable / poisson exchangeable robust, 10 regressors | 1,000,000 rows, 100,000 panels | 1.6 / 1.7 s |
| `xtgee` binomial AR(1) robust | 1,000,000 rows, 100,000 panels | 0.8 s |
| `xtgee` unstructured / stationary(2) / nonstationary(3) robust | 1,000,000 rows, 100,000 balanced panels of 10 | 1.9 / 1.1 / 0.9 s |
| `xtlogit, re` (12 points) / `xtlogit, fe` / `xtprobit, pa` | 100,000 rows, 10,000 panels | 0.6 / 0.17 / 0.06 s |
| `xtpoisson, re` gamma / normal (12 points) / `fe` | 1,000,000 rows, 100,000 panels | 0.9 / 5.3 / 0.4 s |

Every per-observation quantity is one `index_add_`; the only Python loops run
over quadrature nodes, distinct panel sizes, correlation lags and optimizer
iterations. The quadrature arrays are `N x intpoints^q`, so a random slope
multiplies the GLMM cost by `intpoints`.

## Conventions and how sure we are

`provenance["stata_parity_validated"]` is `False` for every model on this
page. The conventions were re-derived from the Stata manuals' Methods and
formulas ([ME] mixed, [ME] melogit, [XT] xtgee, [XT] xtpoisson, [XT] xtlogit)
in the verification pass; the xtpoisson re/fe/pa examples were reproduced
digit for digit (see "Stata parity"). Confidence per point:

* Confirmed by the manual text and the ships example: the gamma and normal RE
  Poisson likelihoods and their `/lnalpha`, `/lnsig2u`, alpha, sigma_u rows;
  conditional Poisson; the pooled exchangeable estimator over `sum r^2 / N`;
  the `G/(G-1)` semi-robust xtgee covariance; Wald model tests for re, pa and
  `xtpoisson, fe`; LR for `xtlogit, fe`.
* Confirmed by the manual text (no reproducible example): the ar/stationary
  and nonstationary/unstructured xtgee estimators, upper-left blocks by
  position, dropping panels with `n_i <= g`; mixed's block-diagonal default
  covariance, Harville REML with `(N - p) ln 2 pi`, robust scores for all
  parameters with no LR test, no robust under REML; melogit's chibar2 / conservative
  chi2 LR tests and clustering at the highest level; xtlogit/xtprobit re's
  sigma_u and rho rows (checked against the printed union example's
  arithmetic: delta-method standard errors, transformed interval endpoints).
* Uncertain:
  - whether nmp should change the xtgee correlation estimates (the printed
    formulas use `sum n_i` and `n_i`, so we leave alpha unaffected);
  - the xtgee glyph read as `n_i` in the ar/stationary denominators (the
    manual's PDF uses a script-size glyph; `n_i` is the only reading under
    which the first element of the estimated vector is 1 and the
    nonstationary estimator is a correlation, as the manual states);
  - omitting the LR comparison tests of melogit/meprobit/mepoisson and the xt
    re models under robust/cluster (confirmed for mixed only; applied by
    analogy);
  - the adaptive quadrature schedule: OpenEconometrics iterates the posterior
    mean/variance adaptation to convergence at the final estimates, Stata
    freezes it once the log likelihood changes by less than 1e-6 — 7th-digit
    differences in derived quantities on the ships example;
  - `xtpoisson, fe` keeps single-observation panels (they do not change the
    estimates, only N and the group counts); the manual does not say whether
    Stata drops them;
  - the exchangeable `covstructure` of mixed: our parameterization and the
    term names `/var(x _cons[g])`, `/cov(x,_cons[g])` (estimates are invariant);
  - group-size summaries with frequency weights count replicated rows.

Additional crossed, multilevel, random-parameter choice/NB and panel frontier
likelihoods have their [own supported domains](extended-mixed.md). Dataset BLUP
and saved conditional/integrated targets are documented in [group prediction](group-prediction.md).
