# Count-data extensions: `zip`, `zinb`, `tpoisson`, `tnbreg`, `churdle`, `hurdle`, `gnbreg`

Seven estimators for outcomes that the plain Poisson and negative binomial
models of the [glm family](glm.md) do not describe well:

- **too many zeros** (people who never visit a doctor, firms that never
  patent): zero-inflated models `zip`, `zinb` and the count hurdle `hurdle`;
- **no zeros at all, by construction** (length of stay of admitted patients,
  items bought by customers who bought something): truncated models
  `tpoisson`, `tnbreg`;
- **a continuous outcome with a mass at a limit** (expenditure, hours):
  Cragg's hurdle model `churdle`;
- **overdispersion that differs between groups**: `gnbreg`.

Everything on this page is implemented in OpenEconometrics on float64 PyTorch tensors:
full maximum likelihood by Newton-Raphson with analytic scores and Hessians (no
autograd, no estimation library at fit time), O(n) memory per iteration. One
million rows with ten regressors fit in one to three seconds (timings at the
end).

| Stata | OpenEconometrics |
| --- | --- |
| `zip y x1 x2, inflate(z1 z2) vuong` | `oe.zip(data=df, y="y", x=["x1","x2"], inflate=["z1","z2"])` |
| `zip y x1, inflate(_cons) probit` | `oe.zip(..., inflate=[], inflate_link="probit")` |
| `zinb y x1 x2, inflate(z1) zip exposure(t)` | `oe.zinb(..., inflate=["z1"], exposure="t")` |
| `tpoisson y x1 x2` (`ztp`) | `oe.tpoisson(data=df, y="y", x=["x1","x2"])` |
| `tpoisson y x1, ll(2)` / `ll(minvar)` | `oe.tpoisson(..., ll=2)` / `oe.tpoisson(..., ll="minvar")` |
| `tnbreg y x1 x2, dispersion(constant)` (`ztnb`) | `oe.tnbreg(..., dispersion="constant")` |
| `churdle linear y x1, select(z1 z2) ll(0)` | `oe.churdle(..., x=["x1"], select_x=["z1","z2"], model="linear", ll=0)` |
| `churdle exponential y x1, select(z1) ll(0)` | `oe.churdle(..., select_x=["z1"], model="exponential", ll=0)` |
| `hplogit y x1 x2` / `hnblogit y x1 x2` (community) | `oe.hurdle(..., x=["x1","x2"])` / `oe.hurdle(..., dist="nbinomial")` |
| `gnbreg y x1 x2, lnalpha(z1)` | `oe.gnbreg(..., x=["x1","x2"], lnalpha=["z1"])` |

SPSS has no dedicated procedure for these models (GENLIN stops at the negative
binomial); EViews offers count models with Poisson / negative binomial
likelihoods and censored / truncated *normal* regressions, of which the
truncated normal part of `churdle` is the closest relative. The conventions
below are therefore Stata's.

All examples use one synthetic data set and were run as shown
(`tests/test_econ_count_docs.py` executes every `python` block of this page and
compares the printed output):

```python
import numpy as np
import pandas as pd
import openecon as oe

rng = np.random.default_rng(2026)
n = 4000
df = pd.DataFrame({
    "age": rng.uniform(20, 60, n).round(),
    "female": rng.integers(0, 2, n),
    "income": rng.lognormal(3.0, 0.5, n).round(1),
    "urban": rng.integers(0, 2, n),
    "clinic": rng.integers(0, 80, n),
    "months": rng.integers(6, 25, n),
})
mu = df.months / 12 * np.exp(-0.6 + 0.03 * df.age + 0.25 * df.female)
never = rng.uniform(size=n) < 1 / (1 + np.exp(-(0.4 - 0.9 * df.urban - 0.02 * (df.age - 40))))
alpha = np.exp(-0.9 + 0.8 * df.urban)
counts = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
df["claims"] = counts                                       # overdispersion differs by group
df["visits"] = np.where(never, 0, counts)                   # excess zeros and overdispersion
df["visits_p"] = np.where(never, 0, rng.poisson(mu))        # excess zeros, Poisson counts
buys = 0.2 + 0.015 * (df.income - 20) + 0.4 * df.urban + rng.normal(size=n) > 0
df["spend"] = np.where(buys, np.exp(2.0 + 0.02 * df.income + 0.1 * df.female
                                    + 0.7 * rng.normal(size=n)), 0.0).round(2)
latent = 4.0 + 0.08 * df.income + 1.0 * df.female + 3.0 * rng.normal(size=n)
df["hours"] = np.where(buys & (latent > 0), latent, 0.0).round(2)
```

## What each estimator accepts

| estimator | outcome | extra columns | options | default covariance |
| --- | --- | --- | --- | --- |
| `zip` | count >= 0 with zeros | `inflate` (list), `offset`, `exposure` | `inflate_link` (`logit`, `probit`) | `nonrobust` |
| `zinb` | count >= 0 with zeros | `inflate` (list), `offset`, `exposure` | `inflate_link` | `nonrobust` |
| `tpoisson` | integer count > `ll` | `truncation`, `offset`, `exposure` | `ll` (integer >= 0, default 0) | `nonrobust` |
| `tnbreg` | integer count > `ll` | `truncation`, `offset`, `exposure` | `ll`, `dispersion` (`mean`, `constant`) | `nonrobust` |
| `churdle` | continuous, mass at `ll` | `select_x` (list, required) | `model` (`exponential`, `linear`), `ll`, `select_link` (`probit`, `logit`) | `nonrobust` |
| `hurdle` | integer count >= 0 with zeros | `select_x` (list), `offset`, `exposure` | `dist` (`poisson`, `nbinomial`), `zero_link` (`logit`, `probit`, `cloglog`) | `nonrobust` |
| `gnbreg` | count >= 0 | `lnalpha` (list), `offset`, `exposure` | - | `nonrobust` |

`x=[]` fits a constant-only first equation in every estimator. Every
estimator accepts the covariances `nonrobust`, `opg`, `robust` and
`cluster` (one or two cluster columns), the weight types `fweight`, `aweight`,
`pweight` and `iweight`, categorical regressors in every equation
(`categorical=[...]`), and `missing="raise"` (default) or `"drop"`.

## Numerical precision at large counts

Poisson log masses and deviances use a stable deviance/Stirling evaluation,
including at large counts near the fitted mean. This avoids cancellation of
large log-factorial terms; it does not remove float64 rounding of predictors,
means or far-tail quantities. Zero-inflated comparisons and PPML likelihoods
use the same primitive.

Negative-binomial evaluations are restricted to `y + shape + 1 <= 1e7`, where
`shape = 1 / alpha` for NB2 or `mu / delta` for NB1. Fixed-dispersion GLM,
ordinary, truncated, zero-inflated, generalized and censored NB models enforce
this count/shape precision region. Values outside it raise
`AnalysisError("precision_unsupported", ...)`; no rounded likelihood is
reported as a certified fit. This limits the magnitude of an individual count
and a special-function shape, **not the number of observations**. Zero-weight
rows are removed before validation.

Unsafe optimizer trials are rejected and tracked; valid final fits survive
such trials. A Poisson-boundary error requires separate likelihood/dispersion
information, rather than the mere rejection of a trial. An unsupported optional
null comparison produces a warning and model Wald inference without discarding
an otherwise valid fitted model. Poisson truncated cumulative cutoffs above
`1e12` are also outside the verified region; see
[censored count precision](censored_count.md) for its cumulative-tail contract.

## Conventions shared by the family

**Inference.** z statistics and normal confidence intervals, as Stata reports
for these commands.

**Covariance** (`core.ml_covariance`, Stata's `ml` conventions):

| `covariance` | formula | Stata |
| --- | --- | --- |
| `nonrobust` | `(-H)^-1`, the inverse observed information of **all** parameters | `vce(oim)` |
| `opg` | `(sum_i w_i s_i s_i')^-1` | `vce(opg)` |
| `robust` | `N/(N-1) (-H)^-1 (sum_i s_i s_i') (-H)^-1` | `vce(robust)` |
| `cluster` | `G/(G-1) (-H)^-1 (sum_g s_g s_g') (-H)^-1`; two columns: Cameron-Gelbach-Miller inclusion-exclusion with the smaller `G` | `vce(cluster id)` |

`H` is the Hessian of the log likelihood and `s_i` the score of observation i,
both over the complete parameter vector (count / outcome equation, auxiliary
equation, ancillary parameter). For the two-part models (`churdle`, `hurdle`)
the scores of the two parts are stacked, so sandwich covariances include the
covariance *between* the equations.

**Weights** (Stata's semantics): `fweight` replicates rows (`N` = sum of the
weights; results equal those of the expanded data set); `aweight` is rescaled
to sum to the number of rows; `iweight` enters as given; `pweight` enters as
given and needs `robust` or `cluster` (the convenience functions choose
`robust`). Estimates do not depend on the unit of `iweight` / `pweight`.

**Offsets.** `offset="col"` enters the count equation with coefficient one;
`exposure="col"` enters as `ln(col)`; they are mutually exclusive.

**Terms and equations.** The constant is `Intercept`. Terms of an auxiliary
equation carry a prefix and an equation label: `inflate:<term>`,
`select:<term>`, `lnalpha:<term>`. Ancillary parameters use Stata's slash names
(`/lnalpha`, `/lndelta`, `/lnsigma`); the parameter itself (`alpha`, `sigma`)
is in `metrics`, and `extra["alpha"]` / `extra["sigma"]` hold its delta-method
standard error and the confidence interval obtained by exponentiating the
interval of the logarithm, as Stata prints.

**Model test.** Stata's convention for multi-equation `ml` commands:
`tests["model"]` tests the slopes of the *first* equation only.

- `nonrobust`: LR chi2(k) against the model whose first equation holds only its
  constant, with the other equations (inflation, selection, `lnalpha`) and
  the ancillary parameter kept and re-estimated. `extra["null_log_likelihood"]`
  is that model's log likelihood; `pseudo_r_squared = 1 - ll/ll_0` uses it
  where Stata reports a pseudo R-squared (`tpoisson`, `tnbreg`, `churdle`,
  `gnbreg`; also `hurdle`).
- any other covariance (including `opg`), or `intercept=False`: Wald chi2(k)
  of the same slopes (and no pseudo R-squared without a constant).
- a comparison model that has no maximum of its own (it does not converge) is
  never replaced by something else: the Wald test is reported, the pseudo
  R-squared is missing and a warning says why.

**Collinearity.** Each equation's design is screened separately, left to
right, as Stata does; omitted terms are listed in
`provenance["omitted_terms"]` and in a warning.

**Numerical safeguards.** Every equation with a constant is estimated on
regressors centered at their means and mapped back exactly, so large-level
regressors (years, incomes) do not degrade the Hessian. Newton-Raphson never
declares convergence in a non-concave region; a likelihood without an interior
maximum is reported as `boundary_solution` or `separation_detected` with the
model to use instead - never as a table with meaningless standard errors.

**Flat likelihoods.** The models of this family have limits outside their
parameter space: `alpha -> 0` (Poisson), an inflation or participation
probability at 0 or 1, a truncated mean at zero, the logarithmic-series limit
of the truncated negative binomial. When the data favour such a limit the
likelihood keeps increasing towards it while its gradient *and* its curvature
vanish, so any optimizer's convergence rules are eventually met at an
arbitrary point (a coefficient of -37 with a standard error of 1e6,
`ln(alpha) = -19` with a standard error of 1458). OpenEconometrics therefore checks
every converged fit with one certificate: for the parameter block concerned,
the smallest generalized eigenvalue of the information `X' diag(w c_i) X`
relative to `X' diag(w) X`, times N. This is the information that the least
informative direction of the linear index receives from the whole sample, in
observations' worth, whatever the units of the regressors. One informative
observation keeps it of order one; below 0.01 (a standard error above ten
units of the linear predictor) *together with* the symptom of the limit
(a dispersion below one, fitted probabilities at 0 or 1, fitted means below
1e-8 / 1e-6) the fit is refused with the code of that limit. Identified
models with a few extreme fitted values (an outlying regressor, rare events,
small groups with one event) pass.

## `oe.zip` - Stata `zip`

### Model

With probability `F(z'g)` an observation is a structural ("excess") zero;
otherwise it is Poisson with mean `mu = exp(x'b + offset)`:

```
Pr(y = 0) = F + (1 - F) exp(-mu)
Pr(y = k) = (1 - F) exp(-mu) mu^k / k!        k = 1, 2, ...
E[y]      = (1 - F) mu
```

`F` is the logistic cdf (`inflate_link="logit"`, default) or the standard
normal cdf (`"probit"`). A positive `inflate:` coefficient *raises* the
probability of an excess zero.

### Estimation

`(b, g)` jointly by Newton-Raphson. The zero branch is evaluated as
`logaddexp(ln F, ln(1 - F) - mu)`; with `pi_i` the posterior probability that
a zero is structural (0 for positive counts) the score and Hessian are, per
observation,

```
dl/dg = pi a' + (1 - pi) b'              a = ln F,  b = ln(1 - F)
dl/db = (1 - pi) (y - mu) x
d2l/db db' = [-(1 - pi) mu + pi (1 - pi) mu^2] x x'      (zeros;  -mu x x' otherwise)
```

Starting values: the Poisson estimates of `b` and a binary fit of `1[y = 0]`
on the inflation regressors (fallback: a constant at the share of excess
zeros left by the Poisson fit).

### Example

```python
result = oe.zip(data=df, y="visits_p", x=["age", "female"], inflate=["urban", "age"],
                exposure="months")
print(result.summary())
```

```text
Zero-inflated Poisson regression — visits_p
Observations: 4000  |  Covariance: nonrobust  |  Confidence: 95%

Term                Estimate  Std. error         z      P>|stat|    CI lower    CI upper
-----------------  ---------  ----------  --------  ------------  ----------  ----------
[visits_p]                                                                              
Intercept           -3.03249   0.0668154  -45.3861             0    -3.16345    -2.90154
age                0.0287152  0.00134681   21.3209  7.27224e-101   0.0260755   0.0313549
female              0.256617   0.0283369    9.0559   1.35443e-19    0.201077    0.312156
[inflate]                                                                               
inflate:Intercept    1.22848    0.148309   8.28325   1.19854e-16    0.937801     1.51916
inflate:urban      -0.917346   0.0736515  -12.4552   1.30981e-35     -1.0617   -0.772992
inflate:age        -0.020291  0.00333735  -6.07998   1.20194e-09  -0.0268321  -0.0137499

log_likelihood: -5699.29  |  aic: 11410.6  |  bic: 11448.3  |  n_zero_observations: 2194
LR chi2 test that the slopes of the count equation are zero: chi2(2) = 552.761, p = 9.321e-121
Vuong test of zip vs. standard Poisson (z > 0 favours zip; p-value Pr(Z > z)): normal = 28.0434, p = 2.407e-173
```

The count equation recovers the data-generating coefficients (0.03 and 0.25),
and the inflation equation says that urban and older people are less likely to
be "never visitors" (true values -0.9 and -0.02).

### What is reported

- `metrics`: `log_likelihood`, `aic`, `bic` (all parameters), `n_zero_observations`.
- `tests["model"]`: LR chi2 of the count equation's slopes (Stata's header
  line), or Wald.
- `tests["vuong"]`: the Vuong (1989) test of `zip` against the Poisson model,
  Stata's `vuong` option. With `m_i = ln f_zip(y_i) - ln f_poisson(y_i)` (each
  model at its own maximum likelihood estimates),

  ```
  V = sqrt(N) mean(m) / sd(m)          sd with divisor N - 1
  ```

  is asymptotically standard normal; large positive values favour `zip`. The
  reported p-value is the upper tail `Pr(Z > V)`, the "Pr > z" Stata printed.
  `extra["vuong"]` adds the AIC- and BIC-corrected statistics of Desmarais and
  Harden (2013), which replace `mean(m)` by `mean(m) - (k1 - k2)/N` and
  `mean(m) - (k1 - k2) ln(N)/(2N)` (`k1 - k2` = number of inflation
  parameters), plus the two-sided p-value. The test is reported for unweighted
  and frequency-weighted fits with a likelihood-based covariance (Stata refuses
  `vuong` with `vce(robust)`).
- `extra`: `poisson_log_likelihood`, `null_log_likelihood`,
  `mean_inflation_probability`, `inflate_link`.
- `predictions` (chart sample): `E[y] = (1 - F) mu`.

```python
vuong = result.extra["vuong"]
print(round(vuong["z"], 3), round(vuong["z_aic"], 3), round(vuong["z_bic"], 3))
print(round(result.extra["mean_inflation_probability"], 4))
```

```text
28.043 27.994 27.837
0.4874
```

> Stata 15 removed the `vuong` option because the zero-inflated and the plain
> model are nested on the boundary of the parameter space, where Vuong's
> normal approximation is not justified (Wilson 2015). The statistic is still
> widely reported; treat it as descriptive and prefer information criteria.

`inflate=[]` is Stata's `inflate(_cons)`: one inflation probability for
everybody.

## `oe.zinb` - Stata `zinb`

As `zip` with negative binomial (NB2) counts: mean `mu`, variance
`mu (1 + alpha mu)` and, with `m = 1/alpha`,

```
f(k) = G(k + m) / (G(m) k!) (1 + alpha mu)^-m (alpha mu / (1 + alpha mu))^k
Pr(y = 0) = F + (1 - F) (1 + alpha mu)^-m
```

`(b, g, ln alpha)` are estimated jointly from the `zip` estimates and a
dispersion taken from the plain negative binomial fit.

```python
result = oe.zinb(data=df, y="visits", x=["age", "female"], inflate=["urban", "age"],
                 exposure="months")
for c in result.coefficients:
    print(f"{c.term:18s} {c.estimate:9.4f} {c.std_error:8.4f}")
print({name: round(value, 4) for name, value in result.extra["alpha"].items()
       if name in ("estimate", "std_error", "ci_low", "ci_high")})
for name in ("model", "alpha", "vuong"):
    test = result.tests[name]
    print(name, test["distribution"], round(test["statistic"], 2), f"{test['p_value']:.3g}")
```

```text
Intercept            -3.0423   0.1228
age                   0.0271   0.0025
female                0.2984   0.0519
inflate:Intercept     1.2305   0.1829
inflate:urban        -0.6375   0.0865
inflate:age          -0.0237   0.0040
/lnalpha             -0.3534   0.0979
{'estimate': 0.7023, 'std_error': 0.0687, 'ci_low': 0.5797, 'ci_high': 0.8508}
model chi2 139.61 4.83e-31
alpha chibar2 974.26 3.53e-214
vuong normal 7.7 6.78e-15
```

- `metrics` adds `alpha`; `extra["alpha"]` its standard error and interval.
- `tests["alpha"]`: LR test of `alpha = 0`, i.e. `zinb` against `zip` (Stata's
  `zip` option). The null lies on the boundary of the parameter space, so the
  reference distribution is `chibar2(01)`: the p-value is half the `chi2(1)`
  tail. Reported with `nonrobust` and `opg` only.
- `tests["vuong"]`: against the negative binomial model without inflation.
- `boundary_solution` is raised when `alpha` is estimated at zero (use
  `oe.zip`) or the inflation probability is estimated at zero (use `oe.nbreg`).

## `oe.tpoisson` - Stata `tpoisson` (formerly `ztp`)

### Model

Only outcomes above a truncation point are observed, `y > ll` (default
`ll = 0`). With `mu = exp(x'b + offset)` the mean of the *untruncated* Poisson
count,

```
Pr(y | y > ll) = exp(-mu) mu^y / y! / Pr(Y > ll)
Pr(Y > ll)     = 1 - sum_{j <= ll} exp(-mu) mu^j / j! = P(ll + 1, mu)
```

where `P` is the regularized lower incomplete gamma function. `ll` is a
nonnegative integer or the name of a column of observation-specific truncation
points (Stata's `ll(#)` / `ll(varname)`).

### Estimation

The truncated Poisson distribution is an exponential family in `x'b`, so the
log likelihood is globally concave:

```
score   = X' w (y - E[y | y > ll])
Hessian = -X' diag(w Var[y | y > ll]) X
E[y | y > ll] = mu + r,    r = mu f(ll) / Pr(Y > ll)
Var[y | y > ll] = mu + r (1 + ll - mu) - r^2
```

`ln Pr(Y > ll)` is computed from the incomplete gamma function (as
`log1p(-Q)` when the complement is the small tail, and as `ln(1 - exp(-mu))`
for zero truncation), so it stays accurate when `mu` is far below - or far
above - the truncation point.

```python
positive = df[df.visits > 0]
result = oe.tpoisson(data=positive, y="visits", x=["age", "female"], exposure="months",
                     covariance="robust")
print(result.summary())
```

```text
Truncated Poisson regression — visits
Observations: 1510  |  Covariance: robust  |  Confidence: 95%

Term        Estimate  Std. error         z      P>|stat|   CI lower   CI upper
---------  ---------  ----------  --------  ------------  ---------  ---------
Intercept   -2.63827    0.100687  -26.2027  2.47647e-151   -2.83561   -2.44092
age        0.0236328  0.00222638   10.6149   2.54087e-26  0.0192692  0.0279964
female      0.207348   0.0508295   4.07928    4.5175e-05   0.107724   0.306972

log_likelihood: -3613.23  |  pseudo_r_squared: 0.0486358  |  aic: 7232.46  |  bic: 7248.42
Wald chi2 test that the slopes of the count equation are zero: chi2(2) = 133.303, p = 1.132e-29
```

- Coefficients describe the latent (untruncated) mean, as in Stata: `exp(b)`
  is an incidence-rate ratio.
- `metrics`: `log_likelihood`, `pseudo_r_squared`, `aic`, `bic`.
- `tests["model"]`: LR chi2 against the constant-only model (`nonrobust`) or
  the Wald chi2 of the slopes.
- `predictions` use the conditional mean `E[y | y > ll]`, which is what the
  observed outcomes average to.
- An outcome at or below the truncation point contradicts the model:
  `outcome_not_truncated` (the rows are never dropped silently). Non-integer
  outcomes raise `invalid_count_outcome`.
- `Pr(y = ll + 1 | y > ll)` tends to one as the mean goes to zero. A regressor
  that singles out observations whose outcomes *all* equal `ll + 1` (a dummy
  for a group of one-day stays in a zero-truncated model) therefore has no
  finite coefficient: the likelihood keeps increasing as it goes to minus
  infinity, the truncated-count analogue of separation in a logit. Gradient and
  curvature vanish together along that direction, so an optimizer can stop at
  an arbitrary coefficient such as -37 with a standard error of 1e6. OpenEconometrics
  certifies the situation - a fitted mean below 1e-8 at an outcome of `ll + 1`
  together with less than 1e-3 observations' worth of information along some
  direction of the parameter space - and raises `separation_detected` (also in
  `tnbreg` and in the count part of `hurdle`). An identified model with a few
  tiny fitted means (an extreme regressor value, rare events) is not affected.

A higher truncation point (only stays of three or more visits are recorded):

```python
long = df[df.visits > 2]
result = oe.tpoisson(data=long, y="visits", x=["age", "female"], ll=2)
print(result.nobs, round(result.metrics["log_likelihood"], 3), result.extra["truncation_point"])
```

```text
789 -1975.359 2
```

## `oe.tnbreg` - Stata `tnbreg` (formerly `ztnb`)

Truncated negative binomial regression: the same conditional likelihood with

- `dispersion="mean"` (NB2, default): `Var = mu (1 + alpha mu)`; for zero
  truncation the normalizer is `1 - (1 + alpha mu)^(-1/alpha)`;
- `dispersion="constant"` (NB1): `Var = mu (1 + delta)`.

For `ll > 0` the lower sum `F(ll) = sum_{j <= ll} f(j)` is accumulated term by
term together with its first and second derivatives in `(x'b, ln alpha)`, so
the cost of one likelihood evaluation grows linearly with the largest
truncation point. Starting values are the `tpoisson` estimates.

```python
result = oe.tnbreg(data=positive, y="visits", x=["age", "female"], exposure="months")
for c in result.coefficients:
    print(f"{c.term:10s} {c.estimate:9.4f} {c.std_error:8.4f}")
print(round(result.metrics["alpha"], 4), round(result.metrics["pseudo_r_squared"], 4))
test = result.tests["alpha"]
print(test["distribution"], round(test["statistic"], 2), f"{test['p_value']:.3g}")
```

```text
Intercept    -2.9936   0.1237
age           0.0267   0.0025
female        0.2577   0.0559
/lnalpha     -0.3705   0.0983
0.6904 0.0195
chibar2 962.31 1.4e-211
```

- Terms: the regressors and `/lnalpha` (`/lndelta` for NB1); `metrics` has
  `alpha` (or `delta`).
- `tests["alpha"]`: LR test of `alpha = 0` against `tpoisson`, `chibar2(01)`.
- `tests["model"]`: LR against the constant-only model with a free dispersion.
  If that model is itself at the Poisson boundary (its `alpha` runs to zero),
  its truncated Poisson likelihood - the supremum - is used and a warning says
  so. If instead its dispersion grows without bound (truncated counts that look
  like a logarithmic-series distribution once the regressors are removed; this
  is common for `ll > 0`), the constant-only likelihood has a supremum but no
  maximum: the Wald test is reported, `pseudo_r_squared` is missing, and a
  warning says so.
- Counts that are not overdispersed raise `boundary_solution` (use
  `oe.tpoisson`).
- The other limit of the truncated negative binomial is the
  logarithmic-series distribution, reached as fitted means go to zero (with
  `alpha` going to infinity in the NB2 form). Truncated counts that follow it
  more closely than any negative binomial with finite parameters - heavy
  right tails with a mode at `ll + 1`; with `dispersion="constant"` it is
  enough that one group of a regressor does - leave no interior maximum:
  `boundary_solution`, with that explanation. A sample of such counts can
  still have a genuine, weakly determined maximum (large standard errors of
  the constant and of `/lnalpha`); it is reported when it is one.
- The cost of one likelihood evaluation is about 40 ns per observation and
  lower-tail term (0.9 s for the complete fit of 72,000 rows with `ll = 50`). A
  fit that would need more than 3e8 terms per evaluation (observations times
  `ll + 1`) is refused with `truncation_too_large` rather than left running for
  hours; `tpoisson` has no such limit.

## `oe.churdle` - Stata `churdle linear` / `churdle exponential`

### Model

Cragg's (1971) hurdle model for a continuous outcome with a mass of
observations at a lower limit `ll` (usually zero). Participation and amount
are separate decisions with separate regressors and coefficients - the
restriction tobit imposes (one index for both) is relaxed:

```
selection:   Pr(y > ll | z) = Phi(z'g)
outcome:     model="linear":        y    = x'b + e,   e ~ N(0, sigma^2), truncated to y > ll
             model="exponential":   ln y = x'b + e,   e ~ N(0, sigma^2), truncated to y > ll
```

Log likelihood of one observation (`d = 1[y > ll]`):

```
(1 - d) ln{1 - Phi(z'g)} + d [ln Phi(z'g) + ln f(y)]
linear:       ln f = ln phi((y - x'b)/sigma) - ln sigma - ln Phi((x'b - ll)/sigma)
exponential:  ln f = ln phi((ln y - x'b)/sigma) - ln sigma - ln y - ln Phi((x'b - ln ll)/sigma)
```

For `ll = 0` the last term of the exponential model vanishes: the positive
outcomes are lognormal and the outcome equation is the regression of `ln y`
(with `sigma^2 = SSR/n`). `select_link="logit"` replaces the probit by a logit
(an OpenEconometrics extension).

### Estimation

The two parts share no parameter, so the log likelihood is the sum of a binary
likelihood (all observations) and a truncated-normal likelihood (observations
above the limit). OpenEconometrics maximizes each part by Newton-Raphson with analytic
derivatives - the outcome part in `(b, ln sigma)` from least-squares starting
values - which is exactly the joint maximum. The joint Hessian is block
diagonal; `robust` and `cluster` covariances stack the scores of both parts.

```python
result = oe.churdle(data=df, y="spend", x=["income", "female"],
                    select_x=["income", "urban"], model="exponential", ll=0)
print(result.summary())
```

```text
Cragg hurdle regression — spend
Observations: 4000  |  Covariance: nonrobust  |  Confidence: 95%

Term                Estimate  Std. error          z      P>|stat|    CI lower   CI upper
----------------  ----------  ----------  ---------  ------------  ----------  ---------
[spend]                                                                                 
Intercept            2.01066   0.0322165    62.4107             0     1.94751     2.0738
income              0.021129  0.00106558    19.8287   1.68229e-87   0.0190406  0.0232175
female               0.05837   0.0270386    2.15876     0.0308685  0.00537529   0.111365
[select]                                                                                
select:Intercept  -0.0356535   0.0502521  -0.709492      0.478019   -0.134146  0.0628388
select:income      0.0134109  0.00187132    7.16655   7.69091e-13  0.00974321  0.0170787
select:urban        0.395585   0.0417272    9.48028   2.53603e-21    0.313801   0.477369
/lnsigma             -0.3539   0.0136007   -26.0207  2.88857e-149   -0.380557  -0.327243

log_likelihood: -12194.6  |  pseudo_r_squared: 0.0149131  |  aic: 24403.2  |  bic: 24447.2
sigma: 0.701945
LR chi2 test that the slopes of the outcome equation are zero: chi2(2) = 369.225, p = 6.665e-81
```

Terms are in Stata's order: the outcome equation, the selection equation
(`select:`, Stata's `selection_ll`) and `/lnsigma`. The linear model on an
outcome that is normal above zero, with a cluster-robust covariance:

```python
result = oe.churdle(data=df, y="hours", x=["income", "female"],
                    select_x=["income", "urban"], model="linear", ll=0, cluster="clinic")
for c in result.coefficients:
    print(f"{c.term:18s} {c.estimate:9.4f} {c.std_error:8.4f}")
print({name: round(value, 4) for name, value in result.extra["sigma"].items()
       if name in ("estimate", "std_error", "ci_low", "ci_high")})
print(result.tests["model"]["label"])
```

```text
Intercept             4.0948   0.1410
income                0.0760   0.0046
female                1.1091   0.1189
select:Intercept     -0.0874   0.0466
select:income         0.0143   0.0018
select:urban          0.3736   0.0419
/lnsigma              1.0767   0.0172
{'estimate': 2.9351, 'std_error': 0.0506, 'ci_low': 2.8376, 'ci_high': 3.0359}
Wald chi2 test that the slopes of the outcome equation are zero
```

### What is reported

- `metrics`: `log_likelihood`, `pseudo_r_squared`, `aic`, `bic`, `sigma`.
- `tests["model"]`: LR chi2 that the slopes of the outcome equation are zero
  (`nonrobust`), else Wald - Stata's header line.
- `extra`: `sigma` (estimate, delta-method standard error, exponentiated
  interval), `selection_log_likelihood`, `outcome_log_likelihood`,
  `null_log_likelihood`, `n_bounded_observations`,
  `mean_selection_probability`.
- `predictions`: `E[y] = (1 - P) ll + P E[y | y > ll]` with `P = Pr(y > ll)`;
  linear: `E[y | y > ll] = x'b + sigma phi(A)/Phi(A)`, `A = (x'b - ll)/sigma`;
  exponential: `exp(x'b + sigma^2/2) Phi(A + sigma)/Phi(A)`,
  `A = (x'b - ln ll)/sigma` (`exp(x'b + sigma^2/2)` for `ll = 0`).
- Outcomes strictly below `ll` are treated as bounded observations, with a
  warning. `no_selection_variation` is raised when no outcome, or every
  outcome, exceeds the limit; `separation_detected` when a selection regressor
  predicts participation perfectly.

Limitations: only a lower limit (Stata's `ul()` is not implemented) and a
homoskedastic outcome equation (Stata's `het()` suboption is not implemented).

## `oe.hurdle` - hurdle model for counts (Mullahy 1986; `hplogit`, `hnblogit`)

### Model

Zeros and positive counts come from two separate processes:

```
participation:   Pr(y > 0 | z) = F(z'g)
positive count:  Pr(y = k | y > 0) = f(k) / (1 - f(0)),     k = 1, 2, ...
```

`F` is logistic (`zero_link="logit"`, default), standard normal (`"probit"`)
or `1 - exp(-exp(z'g))` (`"cloglog"`: Mullahy's original Poisson hurdle, where
the zero part is a Poisson model censored at one); `f` is Poisson
(`dist="poisson"`) or negative binomial NB2 (`dist="nbinomial"`, adds
`/lnalpha`). In contrast to `zip`, *every* zero comes from the participation
equation, so the model also fits data with fewer zeros than the count
distribution predicts. `select_x` defaults to the regressors `x`.

There is no official Stata command; the model is what the community commands
`hplogit` and `hnblogit` (Hilbe) fit, and it equals `logit` on `y > 0` plus
`tpoisson` / `tnbreg` on the positive counts. OpenEconometrics reports the count
equation first (as `churdle` does); `hplogit` prints the logit equation first.

```python
result = oe.hurdle(data=df, y="visits", x=["age", "female"], select_x=["urban", "age"],
                   dist="nbinomial", exposure="months")
print(result.summary())
```

```text
Hurdle count regression — visits
Observations: 4000  |  Covariance: nonrobust  |  Confidence: 95%

Term               Estimate  Std. error         z      P>|stat|   CI lower   CI upper
----------------  ---------  ----------  --------  ------------  ---------  ---------
[visits]                                                                             
Intercept          -2.99362    0.123704     -24.2  2.22713e-129   -3.23608   -2.75117
age               0.0266946  0.00250564   10.6538   1.67431e-26  0.0217836  0.0316055
female             0.257697   0.0558645   4.61289   3.97118e-06   0.148204   0.367189
[select]                                                                             
select:Intercept   -2.06094    0.131328   -15.693   1.68807e-55   -2.31834   -1.80354
select:urban       0.511407   0.0669886   7.63424   2.27151e-14   0.380112   0.642703
select:age         0.031816   0.0029523   10.7767   4.43599e-27  0.0260296  0.0376024
/lnalpha           -0.37052   0.0982762  -3.77019   0.000163122  -0.563138  -0.177902

log_likelihood: -5692.58  |  pseudo_r_squared: 0.0108435  |  aic: 11399.2  |  bic: 11443.2
n_zero_observations: 2490  |  alpha: 0.690375
LR chi2 test that the slopes of the count equation are zero: chi2(2) = 124.808, p = 7.912e-28
LR test of alpha = 0 against the Poisson hurdle model (zero-truncated Poisson counts) (chibar2(01): chi2(1) tail halved): chibar2(1) = 962.307, p = 1.401e-211
```

The count equation is identical to the `tnbreg` fit of the positive counts
above; the `select:` equation is the logit of `visits > 0`.

- `metrics`: `log_likelihood`, `pseudo_r_squared`, `aic`, `bic`,
  `n_zero_observations`, `alpha`.
- `tests["model"]`: LR chi2 that the slopes of the count equation are zero
  (the participation part is common to both models), else Wald.
- `tests["alpha"]` (`dist="nbinomial"`): LR test against the Poisson hurdle,
  `chibar2(01)`.
- `predictions`: `E[y] = Pr(y > 0) mu / (1 - f(0))`.
- `select_x` defaults to `x` (categorical regressors included). A count
  regressor that is constant among the positive counts is omitted from the
  count equation with a warning; one that singles out observations whose
  positive counts all equal 1 raises `separation_detected` (see `tpoisson`).

## `oe.gnbreg` - Stata `gnbreg`

### Model

Negative binomial (NB2) regression in which the overdispersion parameter is a
function of covariates:

```
E[y | x]      = mu = exp(x'b + offset)
Var(y | x, z) = mu (1 + alpha mu),      ln(alpha) = z'd
```

The `lnalpha` equation always contains a constant; without regressors the
model is `nbreg`. A positive `lnalpha:` coefficient means more overdispersion.
`(b, d)` are estimated jointly by Newton-Raphson with the analytic derivatives
of the NB2 density in the two indices `(x'b, z'd)`, starting from the
constant-alpha fit.

```python
result = oe.gnbreg(data=df, y="claims", x=["age", "female"], lnalpha=["urban"],
                   exposure="months")
print(result.summary())
print({name: round(value, 4) for name, value in result.extra["alpha"].items()
       if name != "definition"})
```

```text
Generalized negative binomial regression — claims
Observations: 4000  |  Covariance: nonrobust  |  Confidence: 95%

Term                Estimate  Std. error         z     P>|stat|   CI lower   CI upper
-----------------  ---------  ----------  --------  -----------  ---------  ---------
[claims]                                                                             
Intercept            -3.0736   0.0630804  -48.7251            0   -3.19723   -2.94996
age                0.0286171  0.00140627   20.3497  4.67236e-92  0.0258609  0.0313734
female               0.29826    0.031973   9.32849  1.07385e-20   0.235594   0.360926
[lnalpha]                                                                            
lnalpha:Intercept  -0.960457    0.069801  -13.7599  4.43967e-43   -1.09726   -0.82365
lnalpha:urban       0.851611   0.0863434   9.86308  6.01741e-23   0.682382    1.02084

log_likelihood: -8135.27  |  pseudo_r_squared: 0.028457  |  aic: 16280.5  |  bic: 16312
LR chi2 test that the slopes of the mean equation are zero: chi2(2) = 476.572, p = 3.263e-104
LR chi2 test that the slopes of the lnalpha equation are zero (constant alpha: nbreg): chi2(1) = 104.345, p = 1.7e-24
{'mean': 0.6454, 'min': 0.3827, 'max': 0.8969}
```

- `metrics`: `log_likelihood`, `pseudo_r_squared`, `aic`, `bic`.
- `tests["model"]`: LR chi2 that the slopes of the mean equation are zero (the
  comparison model keeps the `lnalpha` equation), else Wald.
- `tests["lnalpha"]` (an OpenEconometrics addition): tests that the slopes of the
  `lnalpha` equation are zero, i.e. `nbreg` against `gnbreg` - LR under
  `nonrobust`, Wald otherwise.
- `extra["alpha"]`: mean, minimum and maximum of the fitted `alpha_i`;
  `extra["nbreg_log_likelihood"]`, `extra["poisson_log_likelihood"]`.
- `boundary_solution` is raised when the data are not overdispersed - in the
  whole sample (use `oe.poisson`) or in the part of it that a regressor of the
  `lnalpha` equation singles out, whose `ln(alpha)` then runs to minus
  infinity (remove that regressor or merge the category). The density is not
  evaluated for `alpha_i < 1e-9`, where `lnG(y + 1/alpha) - lnG(1/alpha)` is a
  difference of numbers of order 1e10 and carries no usable digits: such a
  trial point is a rejected step.

## Errors

Every invalid input or numerical failure is an `oe.AnalysisError` with a
snake_case `code` and a message that says what to change:

| code | meaning |
| --- | --- |
| `invalid_spec` | undeclared option, role, weight type or covariance; a column list given as a bare string; offset and exposure together; `ll` not a nonnegative integer (at most 2^53); `select` and `select_x` together |
| `invalid_option` | `churdle` exponential with a negative limit |
| `missing_values`, `empty_data`, `empty_sample`, `missing_columns`, `non_numeric_column` | sample problems |
| `negative_weights`, `noninteger_frequency_weights`, `unsupported_covariance` | weights (the last: `pweight` with `nonrobust` / `opg`) |
| `invalid_count_outcome` | negative counts; non-integer counts in `tpoisson`, `tnbreg`, `hurdle` |
| `invalid_truncation` | the truncation column is not a nonnegative integer |
| `invalid_exposure` | exposure not strictly positive |
| `outcome_not_truncated` | `tpoisson` / `tnbreg`: an outcome at or below the truncation point |
| `no_zero_outcomes` | `zip` / `zinb` without any zero |
| `no_selection_variation` | `churdle` / `hurdle`: no outcome, or every outcome, above the limit |
| `constant_outcome` | all zeros; every truncated outcome equal to `ll + 1`; no variation above the limit |
| `perfect_fit` | `churdle`: the outcome equation fits the outcomes above the limit exactly (sigma = 0) |
| `boundary_solution` | inflation probability, `alpha` or `delta` estimated at zero - the message names the simpler model; `gnbreg`: also when `alpha` runs to zero for the part of the sample an `lnalpha` regressor singles out; `tnbreg` / negative binomial `hurdle`: also the logarithmic-series limit (fitted means at zero) |
| `separation_detected` | the inflation or selection equation predicts perfectly; `tpoisson` / `tnbreg` / `hurdle`: a count regressor singles out observations whose outcomes all equal `ll + 1`; `zip` / `zinb` / `gnbreg`: a count regressor singles out observations whose outcomes are all zero (the message names the term) |
| `truncation_too_large` | `tnbreg`: observations times `ll + 1` above 3e8 lower-tail terms per likelihood evaluation |
| `nonconvergence`, `invalid_start`, `singular_information`, `numerical_failure` | numerical failures, with the likely cause |
| `empty_design`, `insufficient_observations`, `insufficient_clusters` | nothing to estimate / too few observations (for `churdle` / `hurdle` also: too few observations above the limit for the second equation) or clusters |

```python
checks = {
    "zip without zeros": lambda: oe.zip(data=df[df.visits > 0], y="visits", x=["age"],
                                        inflate=["urban"]),
    "tpoisson with zeros": lambda: oe.tpoisson(data=df, y="visits", x=["age"]),
    "tnbreg on Poisson counts": lambda: oe.tnbreg(
        data=df.assign(k=1 + rng.binomial(5, 0.4, n)), y="k", x=["age"]),
    "tpoisson, a group with one visit each": lambda: oe.tpoisson(
        data=positive.assign(k=positive.visits.where(positive.urban == 0, 1)), y="k",
        x=["age", "urban"]),
    "gnbreg, a group without overdispersion": lambda: oe.gnbreg(
        data=df.assign(k=df.claims.where(df.urban == 0, rng.binomial(4, 0.5, n))), y="k",
        x=["age"], lnalpha=["urban"]),
    "churdle, all above the limit": lambda: oe.churdle(
        data=df[df.spend > 0], y="spend", x=["income"], select_x=["urban"]),
    "pweight with nonrobust": lambda: oe.gnbreg(
        data=df, y="claims", x=["age"], weights="months", weight_type="pweight",
        covariance="nonrobust"),
}
for name, call in checks.items():
    try:
        call()
    except oe.AnalysisError as error:
        print(f"{name}: {error.code}")
```

```text
zip without zeros: no_zero_outcomes
tpoisson with zeros: outcome_not_truncated
tnbreg on Poisson counts: boundary_solution
tpoisson, a group with one visit each: separation_detected
gnbreg, a group without overdispersion: boundary_solution
churdle, all above the limit: no_selection_variation
pweight with nonrobust: unsupported_covariance
```

## Performance

Every iteration is O(n) in time and memory: designs are n-by-k, Hessian blocks
are k-by-k cross-products, and score rows are materialized only when an OPG or
sandwich covariance asks for them. Wall-clock time on one laptop CPU (Apple
silicon), 1,000,000 rows, ten regressors in the count / outcome equation and
one in the auxiliary equation, including sample construction, starting-value
fits, the comparison models and hashing:

| model | time |
| --- | --- |
| `zip` (nonrobust / robust / cluster) | 1.3 s / 1.1 s / 1.2 s |
| `zinb` (includes the `zip` and `nbreg` fits) | 2.5 s |
| `tpoisson`, 847,000 positive rows (`ll=0`; nonrobust / robust) | 0.6 s / 0.6 s |
| `tpoisson`, 361,000 rows with `ll=2` | 0.4 s |
| `tnbreg`, 743,000 positive rows (NB2 / NB1) | 1.0 s / 1.4 s |
| `tnbreg`, 328,000 rows with `ll=2` | 0.8 s |
| `hurdle` Poisson / negative binomial | 0.5 s / 0.8 s |
| `churdle` exponential / linear | 0.4 s / 0.6 s |
| `gnbreg` | 1.4 s |

Time is linear in the number of rows (100,000 rows take about a tenth of these
figures). The one cost that grows with something else is the lower tail of
`tnbreg`, linear in the truncation point: 72,000 rows with `ll = 50` take
0.9 s, against 0.05 s for `tpoisson`, whose tail is one incomplete gamma
function whatever the truncation point.

## Deliberate differences from Stata

- Convergence is stricter (scaled gradient 1e-10 instead of 1e-5), so
  estimates agree with Stata's to about six significant digits on flat
  likelihoods rather than exactly to its printed precision.
- Boundary cases are errors: `zinb` / `tnbreg` / `gnbreg` on data without
  overdispersion and `zip` / `zinb` on data without excess zeros raise
  `boundary_solution` instead of reporting `alpha` near zero or an inflation
  constant near minus infinity with enormous standard errors. This includes
  runs that meet the convergence rules on the flat part of the likelihood
  (see "Flat likelihoods" above), which Stata reports as converged.
- `tpoisson` / `tnbreg` refuse outcomes at or below the truncation point
  (`outcome_not_truncated`) and non-integer outcomes; rows are never dropped
  silently.
- Monotone likelihoods are errors: a count regressor that singles out
  observations whose truncated outcomes all equal `ll + 1`, or whose outcomes
  are all zero in `zip` / `zinb` / `gnbreg` (`separation_detected`), and an
  `lnalpha` regressor that singles out observations without overdispersion
  (`boundary_solution`) are reported instead of a diverging coefficient with a
  meaningless standard error.
- When the constant-only `tnbreg` (or negative binomial `hurdle`) model has no
  maximum because its dispersion diverges, the Wald test replaces the LR test
  and no pseudo R-squared is reported; Stata would print whatever log
  likelihood its constant-only iterations stopped at.
- `aweight` is accepted by every estimator (Stata's commands take `fweight`,
  `iweight` and `pweight`); two cluster columns (multiway clustering) are
  accepted by every estimator.
- The Vuong test is always computed for `zip` / `zinb` when it is defined
  (Stata 15+ removed the option), with the AIC/BIC-corrected versions.
- `zinb` always reports the LR test against `zip` (Stata needs the `zip`
  option).
- `churdle` has `select_link="logit"` in addition to the probit, and treats
  outcomes below `ll` as bounded with a warning; `ul()` and `het()` are not
  implemented.
- `hurdle` is not an official Stata command; `gnbreg` adds the test of
  constant alpha (`tests["lnalpha"]`).
- With `intercept=False` there is no constant-only comparison model: the Wald
  test is reported and `pseudo_r_squared` is missing.
- The inflation equation has no offset (Stata's `inflate(varlist, offset())`).

## Uncertain conventions

These choices follow the textbook or the specification of this project rather
than Stata output we could check; `provenance["stata_parity_validated"]` is
`False` for every estimator.

1. **Comparison model of the LR test** (`zip`, `zinb`, `gnbreg`, `churdle`).
   We keep the auxiliary equation in full and reduce only the first equation
   to its constant, which is what the degrees of freedom of Stata's "LR
   chi2(k)" header imply (k = slopes of the first equation; the published
   `churdle` example reproduces `Pseudo R2 = 1 - ll/ll_0` with this `ll_0`).
   A null model with constants only in *both* equations would give a different
   statistic.
2. **`churdle` selection index.** The selection probability is `Phi(z'g)` for
   `y > ll`. If Stata's parameterization shifts the index by the limit
   (`Phi(z'g - ll)`), the selection *constant* differs by `ll` when
   `ll != 0`; slopes, likelihood and every other estimate are unaffected, and
   for the usual `ll = 0` nothing differs.
3. **`churdle` exponential log likelihood** includes the Jacobian `-ln y` of
   the lognormal density, so its value is comparable with the linear model's.
   If Stata omits it, log likelihoods (and AIC/BIC, pseudo R-squared) differ by
   the constant `sum ln y`; estimates and tests do not.
4. **Outcomes at or below the truncation point** in `tpoisson` / `tnbreg` are
   an error here; we have not verified whether Stata excludes them with a note
   or stops.
5. **Vuong statistic.** We use the standard deviation with divisor `N - 1`
   (as `summarize`) and the one-sided upper-tail p-value; the AIC/BIC
   corrections follow Desmarais and Harden (2013), which no Stata version
   reported.
6. **`pseudo_r_squared` of `hurdle`** and its model test follow the `churdle`
   convention (first equation's slopes); `hplogit` / `hnblogit` print a Wald
   test.
7. **`aweight`** is not allowed by Stata's `zip`, `zinb`, `tpoisson`, `tnbreg`,
   `churdle` and `gnbreg`; we rescale it to sum to N as Stata's `ml` does for
   commands that accept it.
8. **Two cluster columns.** The multiway meat is made positive semidefinite,
   when necessary, in the estimation coordinates (regressors centered at their
   means), like the glm family; a different parameterization would clip
   slightly different eigenvalues. The result carries a warning when the
   adjustment was needed.
9. **Model test under `opg`.** We report the Wald chi2 for every covariance
   other than `nonrobust`, as the glm family does. Stata's `ml` chooses the
   Wald test because the likelihood is a pseudolikelihood (robust, cluster,
   sampling weights); `vce(opg)` leaves a true likelihood, so Stata most likely
   still prints the LR chi2 there. The LR-based `tests["alpha"]` and the Vuong
   test are reported under `opg`.
10. **N with `iweight`.** N is the number of rows (as in the glm family); it
    enters `bic` and the `N/(N-1)` factor of `robust`. Stata's `ml` may count
    the sum of the importance weights instead, as it does for `fweight`.
11. **Lower limit of `churdle`.** Outcomes equal to *or below* `ll` are the
    bounded observations (with a warning when some lie strictly below); we
    have not verified how Stata treats values strictly below `ll()`.

## Limitations

- `tnbreg` with `ll > 0` evaluates `Pr(y > ll)` as the complement of a lower
  sum; a trial point where that probability falls below 1e-12 for some
  observation is rejected, and a fit whose solution lies there fails with
  `nonconvergence` (the Poisson model has no such limit: it uses the
  incomplete gamma function). Its cost is linear in `ll`; beyond 3e8 terms per
  evaluation the fit is refused (`truncation_too_large`).
- The negative binomial density is evaluated from log-gamma differences and is
  not used below `alpha = 1e-9`; a dispersion that small is reported as the
  Poisson boundary.
- A truncated negative binomial at its logarithmic-series limit has no
  maximum: the main model is refused with `boundary_solution`; a
  constant-only comparison model in that state gives the Wald test instead of
  the LR test. OpenEconometrics has no logarithmic-series regression to offer instead.
- The flat-likelihood certificate refuses fits whose least informative
  direction has a standard error above ten units of the linear predictor
  next to a limit of the model. In very small samples this also refuses a few
  genuine but practically uninformative maxima.
- No upper truncation (`ul()`), no heteroskedastic `churdle`, no offset in
  auxiliary equations, no zero-inflated NB1.
- Post-estimation (`predict`, `margins`) is not implemented for these models;
  `result.predictions` holds a bounded chart sample of fitted means.

## What has been checked, and how

`tests/test_econ_count*.py` compare every estimator with independent oracles.
Two layers were written by different people with different code.

`tests/test_econ_count_oracle.py` (independent verification): every likelihood
is written again from the density with explicit `loggamma` algebra;
per-observation scores come from complex-step differentiation, Hessians from
central differences of those scores, the maximum from a separate Newton
iteration. On a deliberately awkward design (unbalanced clusters, a regressor
in units of 1e4 around 1e6, a categorical regressor, an exposure, a collinear
column, missing rows) it checks, for every estimator, link, truncation point
and dispersion form: coefficients to 1e-7; the complete covariance matrix for
`nonrobust`, `opg`, `robust`, `cluster`, two-way cluster and for `fweight`,
`aweight`, `iweight`, `pweight` to 5e-6; z statistics, p-values and intervals
at a non-default level; log likelihood, AIC, BIC, degrees of freedom; LR,
Wald, `chibar2(01)` and Vuong statistics with their p-values; the
exponentiated intervals of `alpha` / `sigma`; and the chart sample of fitted
means (`churdle`: by numerical quadrature of the truncated density). It also
checks every likelihood kernel's value, score rows and Hessian at points away
from the maximum, the equivalences `gnbreg` (constant alpha) = `nbreg`,
`hurdle` = binary model + truncated count model, lognormal `churdle` = probit
+ least squares of `ln y`, and the invariances of the whole pipeline
(frequency weights = duplicated rows for every covariance, row order, affine
rescaling of a regressor, the unit of the weights, JSON round trip).

`tests/test_econ_count_adversarial.py`: empty, one-row and `n <= k` samples,
all-missing columns, constant and negative outcomes, collinear and constant
columns, bad weights, single / missing / singleton clusters, text where
numbers are needed, regressors, weights, offsets and outcomes at magnitudes
1e-8 and 1e8, options at their bounds, and every boundary or separation
described on this page. Each case returns a fit whose numbers are all finite
or raises `AnalysisError` with the code documented above.

> A note on one oracle: statsmodels 0.14's analytic Hessian of
> `ZeroInflatedPoisson` leaves the cross block between the inflation and the
> count parameters at zero, so its `cov_params()` is not the inverse
> information. The tests compare with the numerically differentiated
> statsmodels likelihood instead, which agrees with OpenEconometrics.

The implementer's files (`test_econ_count.py`, `_zeroinflated`, `_truncated`,
`_hurdle`) add:

- brute-force maximization (`scipy.optimize`) of likelihoods written
  separately with `scipy.stats` pmfs / pdfs, for every model and link;
- numerical Hessians and per-observation scores of those likelihoods for the
  `nonrobust`, `opg`, `robust`, `cluster` and two-way cluster covariances;
- statsmodels `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomialP`,
  `TruncatedLFPoisson`, `TruncatedLFNegativeBinomialP`, `HurdleCountModel`,
  `Probit`, `Logit`, `NegativeBinomial` and least squares of `ln y` (lognormal
  hurdle);
- the Vuong statistic coded from its definition;
- duplicated rows for frequency weights, weighted brute force for analytic
  weights, weighted sandwiches for sampling weights;
- `optimize.check_derivatives` (analytic against numerical gradient and
  Hessian) for every likelihood kernel, and mpmath for the truncation
  probability in both tails;
- collinearity omission, the missing-data policy, every error code above,
  JSON round trips, `summary()` and `to_latex()`.

## References

- Cameron, A. C. and P. K. Trivedi (2013). *Regression Analysis of Count Data*, 2nd ed.
- Cragg, J. G. (1971). Some statistical models for limited dependent variables
  with application to the demand for durable goods. *Econometrica* 39.
- Desmarais, B. A. and J. J. Harden (2013). Testing for zero inflation in count
  models: bias correction for the Vuong test. *Stata Journal* 13.
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to
  defects in manufacturing. *Technometrics* 34.
- Mullahy, J. (1986). Specification and testing of some modified count data
  models. *Journal of Econometrics* 33.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
  non-nested hypotheses. *Econometrica* 57.
- Wilson, P. (2015). The misuse of the Vuong test for non-nested models to test
  for zero-inflation. *Economics Letters* 127.
- StataCorp. *Stata Base Reference Manual*: zip, zinb, tpoisson, tnbreg,
  churdle, gnbreg (Methods and formulas).
