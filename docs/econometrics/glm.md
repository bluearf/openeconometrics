# Generalized linear models, counts and fractions: `glm`, `poisson`, `nbreg`, `cloglog`, `fracreg`, `betareg`, `ppmlhdfe`

Seven estimators for outcomes that are not well described by a linear model
with constant variance: counts, rates, binary and grouped-binomial outcomes,
positive skewed amounts, proportions, and counts or flows with
high-dimensional fixed effects. Everything on this page is implemented in
OpenEconometrics on float64 PyTorch tensors with analytic scores and Hessians (no
autograd, no estimation library at fit time): Newton-Raphson on the observed
Hessian, Fisher scoring by weighted QR least squares, and, for `ppmlhdfe`,
IRLS whose least-squares steps partial the fixed effects out. A model on one
million rows with ten regressors fits in about a second (timings at the end).

| Stata | OpenEconometrics |
| --- | --- |
| `glm y x1 x2, family(gamma) link(log)` | `oe.glm(data=df, y="y", x=["x1","x2"], family="gamma", link="log")` |
| `glm d x, family(binomial n) link(probit) irls vce(robust)` | `oe.glm(..., y="d", family="binomial", trials="n", link="probit", optimizer="irls", covariance="robust")` |
| `glm y x, family(nbinomial 1.5) scale(x2)` | `oe.glm(..., family="nbinomial", dispersion=1.5, scale="x2")` |
| `poisson y x i.g, exposure(t) vce(cluster id)` | `oe.poisson(..., x=["x","g"], categorical=["g"], exposure="t", cluster="id")` |
| `estat gof` (after `poisson`) | `result.tests["gof_deviance"]`, `result.tests["gof_pearson"]` |
| `nbreg y x` / `nbreg y x, dispersion(constant)` | `oe.nbreg(...)` / `oe.nbreg(..., dispersion="constant")` |
| `cloglog d x, offset(o)` | `oe.cloglog(..., y="d", offset="o")` |
| `fracreg logit y x` / `fracreg probit y x` | `oe.fracreg(...)` / `oe.fracreg(..., link="probit")` |
| `betareg y x, scale(z) link(probit)` | `oe.betareg(..., scale=["z"], link="probit")` |
| `ppmlhdfe y x, absorb(i t) cluster(i)` | `oe.ppmlhdfe(..., absorb=["i","t"], cluster="i")` |

Unweighted `oe.logit` and `oe.probit` retain their separation-certified core and
Dataset routes. Their [direct weighted API](weighted-binary-eight-2026-10-10.md)
accepts `weights=` and `weight_type=` for resident f/a/i/pweights, with explicit
ML covariance conventions and resource admission. Offsets, grouped data and
other links use `oe.glm(family="binomial", ...)`.

All examples on this page use one synthetic data set and were run as shown
(`tests/test_econ_glm_verify.py` executes every `python` block of this page):

```python
import numpy as np
import pandas as pd
import openecon as oe

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "age": rng.uniform(20, 60, n).round(),
    "female": rng.integers(0, 2, n),
    "region": rng.choice(["north", "south", "west"], n),
    "firm": rng.integers(0, 100, n),
    "year": rng.integers(2015, 2021, n),
    "pyears": rng.uniform(0.5, 3.0, n).round(2),
})
index = -1.0 + 0.03 * df.age - 0.3 * df.female + 0.2 * (df.region == "south")
df["visits"] = rng.negative_binomial(2, 2 / (2 + df.pyears * np.exp(index)))   # overdispersed
df["cost"] = rng.gamma(2.0, np.exp(4 + 0.02 * df.age - 0.2 * df.female) / 2.0)  # positive, skewed
p = 1 / (1 + np.exp(-index))
df["insured"] = rng.binomial(1, p)
df["trials"] = rng.integers(1, 11, n)
df["successes"] = rng.binomial(df.trials, p)
df["share"] = rng.beta(p * 8, (1 - p) * 8)                           # strictly inside (0, 1)
df["rate"] = np.where(rng.uniform(size=n) < 0.08, 0.0, df.share)     # with corner values
effect = rng.normal(0, 0.4, 100)[df.firm] + 0.05 * (df.year - 2015)
df["sales"] = rng.poisson(np.exp(0.5 + 0.02 * df.age - 0.2 * df.female + effect))
```

## What each estimator accepts

| estimator | outcome | weights | covariances (default first) | extra columns | options |
| --- | --- | --- | --- | --- | --- |
| `glm` | by family | `fweight`, `aweight`, `pweight`, `iweight` | `nonrobust`, `opg`, `robust`, `cluster` | `offset`, `exposure`, `trials` | `family`, `link`, `power`, `dispersion`, `scale`, `optimizer`, `max_iterations`, `tolerance` |
| `poisson` | count >= 0 | all four | `nonrobust`, `opg`, `robust`, `cluster` | `offset`, `exposure` | — |
| `nbreg` | count >= 0 | all four | `nonrobust`, `opg`, `robust`, `cluster` | `offset`, `exposure` | `dispersion` (`mean`, `constant`) |
| `cloglog` | 0/1 | `fweight`, `pweight`, `iweight` | `nonrobust`, `opg`, `robust`, `cluster` | `offset` | — |
| `fracreg` | in [0, 1] | all four | `robust`, `cluster`, `nonrobust`, `opg` | — | `link` (`logit`, `probit`) |
| `betareg` | in (0, 1) | `fweight`, `pweight`, `iweight` | `nonrobust`, `opg`, `robust`, `cluster` | `scale` (list) | `link`, `scale_link` |
| `ppmlhdfe` | >= 0 | `fweight`, `aweight`, `pweight` | `robust`, `cluster`, `nonrobust` | `absorb` (list, required), `offset`, `exposure` | `drop_singletons`, `tolerance`, `max_iterations` |

Coefficient tests are z tests everywhere (as in Stata). `cluster` takes one or
two columns. `x` may be empty for every estimator except `ppmlhdfe`: the
constant-only model (for example an overall rate, `oe.poisson(..., x=[],
exposure="pyears")`). Anything not listed is rejected with `invalid_spec`
before estimation starts.

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

**Sample.** `missing="raise"` (default) refuses rows with a missing value in any
model column (outcome, regressors, weights, cluster, offset, exposure, trials,
scale, absorb); `missing="drop"` excludes them and records the count. Rows with
zero weight are excluded. `categorical=[...]` expands regressors into
treatment-coded dummies named `name[level]` (first level omitted). Collinear
and constant regressors are omitted left to right, Stata style, and recorded in
`result.provenance["omitted_terms"]` and `result.warnings`. A model left
without any term raises `empty_design`.

**Weights** (Stata's semantics for likelihood estimators). Every likelihood,
score and Hessian sum is `sum_i w_i (.)`:

| type | `w_i` | N | sandwich meat |
| --- | --- | --- | --- |
| `fweight` | integer frequency | `sum w_i` | `sum f_i s_i s_i'` (replicated rows) |
| `aweight` | rescaled to sum to the number of rows | rows | `sum (w_i s_i)(w_i s_i)'` |
| `pweight` | as given | rows | `sum (w_i s_i)(w_i s_i)'` |
| `iweight` | as given (must be positive) | rows | `sum (w_i s_i)(w_i s_i)'` |

Frequency-weighted results equal those of the data set with rows repeated, for
every covariance. `pweight` requires `robust` or `cluster`: the convenience
functions select `robust` automatically, and an explicit `nonrobust` or `opg`
raises `unsupported_covariance`.

**Offset and exposure.** `offset` enters the linear predictor with coefficient
1; `exposure` enters as `ln(exposure)` and must be strictly positive
(`invalid_exposure`). They are mutually exclusive. Constant-only comparison
models keep the offset and the weights.

**Covariance** (`core.ml_covariance`, Stata's `ml` conventions), with `H` the
Hessian of the weighted log likelihood and `s_i` the score rows:

| name | formula | Stata |
| --- | --- | --- |
| `nonrobust` | `(-H)^-1` | `vce(oim)` |
| `opg` | `(sum_i w_i s_i s_i')^-1` | `vce(opg)` |
| `robust` | `N/(N-1) (-H)^-1 M (-H)^-1`, `M` as in the table above | `vce(robust)` |
| `cluster` | `G/(G-1) (-H)^-1 (sum_g s_g s_g') (-H)^-1` | `vce(cluster g)` |

With two cluster columns the meat is the Cameron-Gelbach-Miller
inclusion-exclusion sum with the factor `G_min/(G_min - 1)` (an extension; a
non-positive-semidefinite result is repaired and reported in the warnings).
Fewer than two clusters raise `insufficient_clusters`; fewer than 30 add a
warning. `ppmlhdfe` uses reghdfe's factors instead (see its section).
`result.inference["correction"]` always states the factor that was applied.

**Centering.** With a constant in the model every likelihood is maximized on
the design with its other columns centered at their weighted means; estimates
and covariance are mapped back exactly (`b0 = b0_c - m'b`, `V = T V_c T'`).
This changes nothing statistically and keeps the information matrix well
conditioned for regressors such as calendar years.

**Newton-Raphson** (`engines.optimize.maximize_newton`). Cholesky Newton steps
on the analytic observed Hessian, Marquardt steps where it is not negative
definite, step halving, and convergence only at a concave point with scaled
gradient `g'(-H)^-1 g <= 1e-10` (Stata's `nrtolerance`, 1e-5 there) and a
relative step below `1e-10`. A likelihood that keeps rising (separation, a
dispersion running to its boundary) is never reported as converged.

## `oe.glm` — Stata `glm` (SPSS GENLIN, EViews GLM)

`g(E[y_i]) = eta_i = x_i'b + offset_i` and `Var(y_i) = phi V(mu_i) / w_i`. The
coefficients minimize the deviance `D = sum_i w_i d(y_i, mu_i)` (maximum
quasi-likelihood); they do not depend on `phi`.

| `family` | `V(mu)` | unit deviance `d(y, mu)` | default link | outcome |
| --- | --- | --- | --- | --- |
| `gaussian` | 1 | `(y - mu)^2` | `identity` | any |
| `binomial` | `mu (1 - mu)` | `2 [y ln(y/mu) + (1-y) ln((1-y)/(1-mu))]` | `logit` | 0/1, or a count `0..n` with `trials="n"` |
| `poisson` | `mu` | `2 [y ln(y/mu) - (y - mu)]` | `log` | >= 0 |
| `gamma` | `mu^2` | `2 [-ln(y/mu) + (y - mu)/mu]` | `reciprocal` | > 0 |
| `inverse_gaussian` | `mu^3` | `(y - mu)^2 / (mu^2 y)` | `inverse_squared` | > 0 |
| `nbinomial` | `mu + k mu^2` | `2 [y ln(y/mu) - (y + 1/k) ln((1 + k y)/(1 + k mu))]` | `log` | >= 0 |

`k` is the FIXED overdispersion `dispersion` (default 1, Stata's
`family(nbinomial #)`); to estimate it use `oe.nbreg`. With `trials` the
binomial family works on the proportion `y/n` with prior weight `w n`, which
reproduces the deviance, Pearson statistic and likelihood of the count model.

| `link` | `eta = g(mu)` | `mu = g^-1(eta)` | `dmu/deta` |
| --- | --- | --- | --- |
| `identity` | `mu` | `eta` | 1 |
| `log` | `ln mu` | `e^eta` | `mu` |
| `logit` | `ln(mu/(1-mu))` | `1/(1 + e^-eta)` | `mu (1 - mu)` |
| `probit` | `Phi^-1(mu)` | `Phi(eta)` | `phi(eta)` |
| `cloglog` | `ln(-ln(1-mu))` | `1 - exp(-e^eta)` | `e^eta exp(-e^eta)` |
| `loglog` | `-ln(-ln mu)` | `exp(-e^-eta)` | `e^-eta exp(-e^-eta)` |
| `power` (`power=a`) | `mu^a` | `eta^(1/a)` | `mu / (a eta)` |
| `reciprocal` | `1/mu` (power -1) | | |
| `inverse_squared` | `1/mu^2` (power -2) | | |
| `nbinomial` | `ln(k mu/(1 + k mu))` | `e^eta / (k (1 - e^eta))` | `mu (1 + k mu)` |

`power=0` is `log` and `power=1` is `identity` (Stata). The binomial family
takes `logit`, `probit`, `cloglog`, `loglog`, `log`, `identity`; the positive
families take `log`, `identity`, `power`, `reciprocal`, `inverse_squared`
(`nbinomial` also its canonical link); gaussian takes every link except
`nbinomial`. Any other pair raises `invalid_option`. Binomial tail probabilities are computed without
cancellation (`1 - Phi(8)` as `Phi(-8)`, complements through `expm1`), so
scores stay finite where fitted probabilities round to 0 or 1.

### Estimation

With `r_i = (y_i - mu_i)/V(mu_i)`, `m1 = dmu/deta`, `q = (d2mu/deta2)/m1` and
`V' = dV/dmu`, per observation

    score factor      s_i = r_i m1                       gradient = X'(w s)
    observed weight   h_i = -m1^2/V + s_i (q - m1 V'/V)  Hessian  = X' diag(w h) X
    expected weight   f_i = m1^2 / V                     Fisher information = X' diag(w f) X

and `h = -f` for a canonical link.

- `optimizer="ml"` (default, Stata's default): Newton-Raphson on `-D/2` with
  the observed Hessian; the reported conventional covariance is the OIM.
- `optimizer="irls"` (Stata's `irls`): Fisher scoring. Each step is a weighted
  QR least squares of the working response `z = eta - offset + (y - mu)/m1` on
  `X` with weights `w f`, halved while the deviance rises. It stops when the
  relative change in the deviance is at most `tolerance` AND the remaining
  Fisher step is negligible (`g' I^-1 g <= tolerance` and a relative step below
  `tolerance`); the second condition matters for non-canonical links, where
  Fisher scoring converges only linearly. The reported conventional covariance
  is the EIM, and the sandwich estimators use the EIM as bread.

Both give the same coefficients; OIM and EIM coincide for canonical links.
Starting values are one weighted least-squares step from `mu_0 = y` (gaussian,
gamma, inverse Gaussian), `(y + 0.5)/(n + 1)` (binomial, `y` the count) or
`y + 0.1` (poisson, nbinomial), falling back to the link of the mean when that
step leaves the link's domain. `tolerance` defaults to `1e-10` and
`max_iterations` to 100; running out of iterations raises `nonconvergence`.

### Dispersion and covariance (Stata's conventions)

`scale` sets the dispersion `phi` that multiplies the conventional covariance:
`"x2"` = Pearson chi2 / `df_resid`, `"dev"` = deviance / `df_resid`, or a
positive number. The default is 1 for binomial, poisson and nbinomial and
`"x2"` for gaussian, gamma and inverse Gaussian (Stata). `df_resid = N - K`.
With `I` the unit-dispersion information (observed for `ml`, expected for
`irls`) and `S` the unit-dispersion score rows:

    nonrobust   phi I^-1
    opg         phi^2 (S'S)^-1          (the scores of the dispersion-phi likelihood are S/phi)
    robust      N/(N-1) I^-1 S'S I^-1   (not scaled by phi)
    cluster     G/(G-1) I^-1 (sum_g s_g s_g') I^-1

### What is reported

`metrics`, in this order: `deviance`, `pearson`, `dispersion_deviance`
(deviance/df), `dispersion_pearson` (pearson/df), `scale` (the `phi` used),
`log_likelihood`, `aic`, `bic`, `aic_glm`, `bic_glm`, `df_resid`, `iterations`.

- `log_likelihood` is the full family likelihood as Stata's `glm` reports it
  and never depends on `scale`. Gaussian: `phi` is concentrated out
  (`phi = deviance/N`), which equals `regress`'s log likelihood. Gamma:
  `sum w [-(y/mu + ln mu)]`; inverse Gaussian:
  `sum w [-(y - mu)^2/(2 y mu^2) - ln(2 pi y^3)/2]` (both at `phi = 1`, as
  Stata). Binomial includes `ln C(n, y)`; poisson and nbinomial their
  factorial and gamma-function terms.
- `aic`, `bic`: `estat ic` (`-2 ll + 2K`, `-2 ll + K ln N`). `aic_glm`,
  `bic_glm`: the two statistics in Stata's `glm` header, `(-2 ll + 2K)/N` and
  `deviance - df_resid ln N`.
- `tests["model"]`: Wald chi2 of the slopes (Stata's `glm` prints no model test).
- `extra`: `family`, `link`, `variance_function`, `link_function`,
  `scale_rule`, `optimizer`, `information` (`observed` / `expected`),
  `canonical_link`, the dispersion at which the likelihood was evaluated.

```python
gamma = oe.glm(data=df, y="cost", x=["age", "female"], family="gamma", link="log")
print(gamma.summary())
print(gamma.inference["correction"])
```

```text
Generalized linear model — cost
Observations: 2000  |  Covariance: nonrobust  |  Confidence: 95%

Term        Estimate  Std. error         z     P>|stat|   CI lower    CI upper
---------  ---------  ----------  --------  -----------  ---------  ----------
Intercept    3.92349   0.0591937   66.2822            0    3.80747      4.0395
age        0.0218975  0.00136142   16.0843  3.28845e-58  0.0192292   0.0245659
female     -0.135312   0.0316355  -4.27723  1.89233e-05  -0.197317  -0.0733078

deviance: 1122.26  |  pearson: 998.836  |  dispersion_deviance: 0.561975  |  dispersion_pearson: 0.500168
scale: 0.500168  |  log_likelihood: -11470.4  |  aic: 22946.7  |  bic: 22963.5
aic_glm: 11.4734  |  bic_glm: -14056.7  |  df_resid: 1997  |  iterations: 4
Wald chi2 test of the slopes: chi2(2) = 279.336, p = 2.203e-61
observed information times dispersion 0.500168
```

Grouped binomial data, a non-canonical link, Fisher scoring and a robust
covariance:

```python
grouped = oe.glm(data=df, y="successes", x=["age", "female"], family="binomial",
                 trials="trials", link="probit", optimizer="irls", covariance="robust")
print(grouped.inference["correction"])
print({name: round(grouped.metrics[name], 4) for name in ("deviance", "pearson", "scale")})
```

```text
Huber-White sandwich: N/(N-1); bread: expected information
{'deviance': 2332.2476, 'pearson': 1972.9701, 'scale': 1.0}
```

### Separation and boundary solutions

- **Separation.** A binomial fit that does not converge while every
  observation with a fitted probability within 1e-6 of 0 or 1 is predicted
  correctly has a monotone likelihood (complete or quasi-complete separation):
  `separation_detected`, never a table of huge coefficients. The same check
  protects `cloglog` and `fracreg`.
- **Boundary solutions.** A link that does not keep the mean inside the
  family's support can have its maximum ON the edge of the link's domain with
  finite coefficients: binomial with `log` (a fitted probability of 1) or
  `identity` (0 or 1), and poisson / nbinomial with `identity` or a power link
  with exponent >= 1 (a fitted mean of 0). The score is not zero there and the
  information weights of the edge observations are unbounded, so neither
  optimizer yields valid standard errors. Any fit that ends within 1e-6 of such
  an edge (for counts: a fitted mean below 1e-6 times the mean outcome),
  converged or not, raises `boundary_solution` with the advice to change the
  link (`oe.poisson(..., covariance="robust")` estimates risk ratios
  without the constraint).
- **Exact fits.** A dispersion estimated from an exact fit is zero:
  `perfect_fit`.

## `oe.poisson` — Stata `poisson`

`y ~ Poisson(mu)`, `mu = exp(x'b + offset)`,
`ll = sum w [y ln mu - mu - lnGamma(y + 1)]`: globally concave, maximized by
Newton-Raphson with score `X'w(y - mu)` and Hessian `-X' diag(w mu) X`. The
outcome must be nonnegative; non-integer values are allowed with a recorded
note (as in Stata), because the estimator only needs `E[y|x] = mu`.

Reported: `log_likelihood`, `pseudo_r_squared` (McFadden, `1 - ll/ll_0` with
`ll_0` the constant-only model with the same offset and weights), `aic`, `bic`
(`estat ic`), `deviance`, `pearson`, `df_resid`. `tests["model"]` is the LR
chi2 against the constant-only model under `nonrobust` and the Wald chi2 of the
slopes otherwise. `tests["gof_deviance"]` and `tests["gof_pearson"]` are
`estat gof`: both statistics against chi2(`N - K`). `exp(estimate)` is an
incidence-rate ratio.

```python
poisson = oe.poisson(data=df, y="visits", x=["age", "female", "region"],
                     categorical=["region"], exposure="pyears")
print(poisson.summary())
```

```text
Poisson regression — visits
Observations: 2000  |  Covariance: nonrobust  |  Confidence: 95%

Term             Estimate  Std. error         z      P>|stat|   CI lower   CI upper
-------------  ----------  ----------  --------  ------------  ---------  ---------
Intercept        -1.04758   0.0659127  -15.8935   7.02941e-57   -1.17677  -0.918397
age             0.0316849  0.00135275   23.4225  2.52141e-121  0.0290335  0.0343362
female          -0.244778   0.0305138  -8.02189   1.04131e-15  -0.304584  -0.184972
region[south]    0.178297   0.0355761   5.01172   5.39469e-07   0.108569   0.248025
region[west]   -0.0435612   0.0385895  -1.12883      0.258968  -0.119195  0.0320729

log_likelihood: -3969.58  |  pseudo_r_squared: 0.0791313  |  aic: 7949.16  |  bic: 7977.17
deviance: 3974.38  |  pearson: 3958.17  |  df_resid: 1995
LR chi2 test against the constant-only model: chi2(4) = 682.221, p = 2.465e-146
Deviance goodness of fit: chi2(1995) = 3974.38, p = 7.331e-134
Pearson goodness of fit: chi2(1995) = 3958.17, p = 4.154e-132
```

The goodness-of-fit tests reject (these data are overdispersed): use a robust
covariance, or `oe.nbreg`.

```python
robust = oe.poisson(data=df, y="visits", x=["age", "female", "region"],
                    categorical=["region"], exposure="pyears", covariance="robust")
assert robust.tests["model"]["label"].startswith("Wald")
overall = oe.poisson(data=df, y="visits", x=[], exposure="pyears")      # constant-only: a rate
rate = np.exp(overall.coefficients[0].estimate)
assert abs(rate - df.visits.sum() / df.pyears.sum()) < 1e-9
```

A regressor that predicts zero outcomes perfectly makes the fitted means run
to zero; the iteration is reported as `nonconvergence` with that diagnosis.

## `oe.nbreg` — Stata `nbreg` (SPSS GENLIN negative binomial, EViews negative binomial count)

`E[y|x] = mu = exp(x'b + offset)` with overdispersion estimated jointly with
the coefficients:

- `dispersion="mean"` (NB2, default): `Var = mu (1 + alpha mu)`; with
  `m = 1/alpha`,
  `l = lnG(y+m) - lnG(m) - lnG(y+1) - (y+m) ln(1 + alpha mu) + y ln(alpha mu)`.
- `dispersion="constant"` (NB1): `Var = mu (1 + delta)`; with `m = mu/delta`,
  `l = lnG(y+m) - lnG(m) - lnG(y+1) - m ln(1 + delta) + y ln(delta/(1 + delta))`.

The parameters are `(b, ln alpha)` (resp. `(b, ln delta)`), maximized by
Newton-Raphson with analytic derivatives (digamma `psi`, trigamma `psi'`):

    NB2   dl/deta = (y - mu)/(1 + alpha mu)
          dl/dln a = m [psi(m) - psi(y+m) + ln(1 + alpha mu)] + (y - mu)/(1 + alpha mu)
          d2l/deta2 = -mu (1 + alpha y)/(1 + alpha mu)^2
    NB1   D = psi(y+m) - psi(m) - ln(1 + delta)
          dl/deta = m D,   dl/dln d = -m D + (y - mu)/(1 + delta)

(the remaining second derivatives are in `glm/kernels.py`; all are verified
against numerical derivatives). Starting values: the Poisson estimates and a
method-of-moments dispersion kept inside [0.05, 50].

Reported as Stata does: the ancillary parameter is the term `/lnalpha`
(`/lndelta`); `metrics["alpha"]` (`"delta"`) and `extra["alpha"]` give the
dispersion itself with its delta-method standard error `alpha se(ln alpha)` and
the interval obtained by exponentiating the interval of `ln alpha`.
`tests["alpha"]` is the likelihood-ratio test of `alpha = 0` against Poisson;
the null is on the boundary, so the p-value is half the chi2(1) tail (Stata's
`chibar2(01)`); it is omitted under `robust` and `cluster`, as in Stata.
`tests["model"]` is the LR chi2 against the constant-only negative binomial
model (`nonrobust`) or the Wald chi2 of the slopes. `aic`/`bic` count `K + 1`
parameters; `df_resid = N - K - 1`.

```python
nb = oe.nbreg(data=df, y="visits", x=["age", "female", "region"], categorical=["region"],
              exposure="pyears")
print(nb.summary())
print({key: round(value, 4) for key, value in nb.extra["alpha"].items()
       if isinstance(value, float)})
```

```text
Negative binomial regression — visits
Observations: 2000  |  Covariance: nonrobust  |  Confidence: 95%

Term             Estimate  Std. error          z     P>|stat|   CI lower   CI upper
-------------  ----------  ----------  ---------  -----------  ---------  ---------
[visits]
Intercept        -0.99983   0.0917322   -10.8994  1.15973e-27   -1.17962  -0.820038
age             0.0310703   0.0019474    15.9548  2.63888e-57  0.0272535  0.0348871
female          -0.270964   0.0447798   -6.05102  1.43932e-09  -0.358731  -0.183197
region[south]    0.160119   0.0531082    3.01496   0.00257011   0.056029   0.264209
region[west]   -0.0506128   0.0557353  -0.908093     0.363829  -0.159852  0.0586263
/lnalpha         -0.78137    0.071742   -10.8914   1.2669e-27  -0.921982  -0.640758

log_likelihood: -3668.47  |  pseudo_r_squared: 0.0379613  |  aic: 7348.93  |  bic: 7382.54
alpha: 0.457778  |  df_resid: 1994
LR chi2 test against the constant-only model: chi2(4) = 289.509, p = 1.984e-61
LR test of alpha = 0 against Poisson (chibar2(01): chi2(1) tail halved): chibar2(1) = 602.23, p = 2.74e-133
{'estimate': 0.4578, 'std_error': 0.0328, 'ci_low': 0.3977, 'ci_high': 0.5269}
```

**Boundary.** For data that are not overdispersed the likelihood increases
monotonically as `alpha -> 0` and has no interior maximum. Stata then reports
an `alpha` numerically at zero with an enormous interval; OpenEconometrics raises
`boundary_solution` and recommends `oe.poisson`. To fix `alpha` instead of
estimating it use `oe.glm(family="nbinomial", dispersion=alpha)`.

## `oe.cloglog` — Stata `cloglog`

`Pr(y = 1 | x) = 1 - exp(-exp(x'b + offset))`, the binary model implied by a
proportional-hazards process for grouped durations; `exp(b)` is a hazard
ratio. Maximum likelihood by Newton-Raphson with the observed Hessian (the
link is not canonical). Reported: z statistics, `log_likelihood`,
`pseudo_r_squared`, `aic`, `bic`, `df_resid`; `tests["model"]` is the LR chi2
(`nonrobust`) or the Wald chi2; `extra["zero_outcomes"]` and
`extra["nonzero_outcomes"]` are the (weighted) counts in Stata's header.
Weights as Stata: `fweight`, `pweight`, `iweight`. Separation raises
`separation_detected`. It is the same fit as `oe.glm(family="binomial",
link="cloglog")`.

```python
cll = oe.cloglog(data=df, y="insured", x=["age", "female"], cluster="firm")
print(cll.summary())
print(cll.extra["zero_outcomes"], cll.extra["nonzero_outcomes"])
```

```text
Complementary log-log regression — insured
Observations: 2000  |  Covariance: cluster  |  Confidence: 95%

Term        Estimate  Std. error         z     P>|stat|   CI lower   CI upper
---------  ---------  ----------  --------  -----------  ---------  ---------
Intercept  -0.852141    0.125518  -6.78901  1.12907e-11   -1.09815  -0.606131
age        0.0196214  0.00272378   7.20374  5.85834e-13  0.0142829  0.0249599
female     -0.284171   0.0590439  -4.81289  1.48765e-06  -0.399895  -0.168448

log_likelihood: -1334.05  |  pseudo_r_squared: 0.0277279  |  aic: 2674.11  |  bic: 2690.91
df_resid: 1997
Wald chi2 test of the slopes: chi2(2) = 76.7481, p = 2.16e-17
881.0 1119.0
```

## `oe.fracreg` — Stata `fracreg logit` / `fracreg probit`

`E[y | x] = G(x'b)` for an outcome in `[0, 1]` (0 and 1 included), `G` the
logistic or standard normal cdf. The coefficients maximize the Bernoulli
quasi-log-likelihood `sum w [y ln G + (1 - y) ln(1 - G)]` (Papke and Wooldridge
1996), which is consistent for the conditional mean without distributional
assumptions; the information-matrix equality does not hold, so the default
covariance is `robust` (Stata's default). Reported: `log_likelihood` (a log
PSEUDOlikelihood; `extra["quasi_likelihood"]` is true), `pseudo_r_squared`
(`1 - ll/ll_0`), `aic`, `bic`, `df_resid`; `tests["model"]` is always the Wald
chi2 of the slopes. A 0/1 outcome that is separated raises
`separation_detected`. Stata's `fracreg` heteroskedastic probit (`het()`) is
not implemented.

```python
frac = oe.fracreg(data=df, y="rate", x=["age", "female"])
print(frac.summary())
```

```text
Fractional response regression — rate
Observations: 2000  |  Covariance: robust  |  Confidence: 95%

Term        Estimate  Std. error         z     P>|stat|   CI lower   CI upper
---------  ---------  ----------  --------  -----------  ---------  ---------
Intercept   -1.11412   0.0741441  -15.0263  4.93481e-51   -1.25944  -0.968796
age        0.0299797  0.00172453   17.3843  1.08526e-67  0.0265997  0.0333598
female     -0.270694   0.0391244   -6.9188  4.55491e-12  -0.347376  -0.194012

log_likelihood: -1351.79  |  pseudo_r_squared: 0.0245614  |  aic: 2709.58  |  bic: 2726.38
df_resid: 1997
Wald chi2 test of the slopes: chi2(2) = 355.899, p = 5.217e-78
```

## `oe.betareg` — Stata `betareg`

`y_i ~ Beta(mu_i phi_i, (1 - mu_i) phi_i)` for `0 < y < 1`, so `E[y] = mu` and
`Var(y) = mu (1 - mu)/(1 + phi)`: `phi` is a precision (Ferrari and
Cribari-Neto 2004; Smithson and Verkuilen 2006). Two equations are estimated
jointly by maximum likelihood:

    mean       g(mu_i)  = x_i'b     link: logit (default), probit, cloglog, loglog
    precision  s(phi_i) = z_i'c     scale_link: log (default), identity, sqrt (Stata's "root")

    l_i = lnG(phi) - lnG(mu phi) - lnG((1-mu) phi) + (mu phi - 1) ln y + ((1-mu) phi - 1) ln(1-y)

The precision equation always has a constant; `scale=[...]` adds regressors.
With `y* = ln(y/(1-y))` and `mu* = psi(mu phi) - psi((1-mu) phi)`:

    dl/dmu  = phi (y* - mu*)
    dl/dphi = mu (y* - mu*) + ln(1-y) - psi((1-mu) phi) + psi(phi)

and the chain rule through both links gives the analytic gradient and Hessian
(trigamma terms) used by Newton-Raphson. Starting values (Ferrari and
Cribari-Neto): least squares of `g(y)` on `X`, and
`phi_0 = mean(mu_0 (1 - mu_0)/(s^2 (dmu/deta)^2)) - 1` from its residual
variance.

Reported: terms of the mean equation (equation = the outcome), then
`scale:Intercept` and `scale:<column>` (equation `scale`). `metrics`:
`log_likelihood`, `aic`, `bic` (all `K + Q` parameters), `pseudo_r_squared`,
`df_resid`. The pseudo R-squared is Ferrari and Cribari-Neto's: the squared
(weighted) correlation between `x'b` and `g(y)`; Stata's `betareg` prints none.
`tests["model"]` is the Wald chi2 of the mean-equation slopes. With a constant
precision `extra["precision"]` gives `phi` with its delta-method standard error
and interval; `extra["precision_range"]` the smallest and largest fitted `phi`.
Weights as Stata: `fweight`, `pweight`, `iweight`. An outcome equal to 0 or 1
raises `invalid_fractional_outcome` and points to `oe.fracreg`.

```python
beta = oe.betareg(data=df, y="share", x=["age", "female"], scale=["female"])
print(beta.summary())
constant = oe.betareg(data=df, y="share", x=["age", "female"])
print({key: round(value, 4) for key, value in constant.extra["precision"].items()
       if isinstance(value, float)})
```

```text
Beta regression — share
Observations: 2000  |  Covariance: nonrobust  |  Confidence: 95%

Term               Estimate  Std. error          z      P>|stat|   CI lower   CI upper
---------------  ----------  ----------  ---------  ------------  ---------  ---------
[share]
Intercept          -1.01264   0.0582038   -17.3981   8.52293e-68   -1.12672  -0.898561
age               0.0318041  0.00135223    23.5198  2.55998e-122  0.0291538  0.0344544
female            -0.319373   0.0310066   -10.3002   7.03364e-25  -0.380145  -0.258602
[scale]
scale:Intercept     2.01485   0.0417929    48.2103             0    1.93293    2.09676
scale:female     0.00270039   0.0593691  0.0454847      0.963721  -0.113661   0.119062

log_likelihood: 778.813  |  aic: -1547.63  |  bic: -1519.62  |  pseudo_r_squared: 0.252457
df_resid: 1995
Wald chi2 test of the mean-equation slopes: chi2(2) = 656.471, p = 2.812e-143
{'estimate': 7.5096, 'std_error': 0.224, 'ci_low': 7.0831, 'ci_high': 7.9617}
```

## `oe.ppmlhdfe` — community `ppmlhdfe` (Correia, Guimaraes and Zylkin 2020)

`E[y | x, d] = exp(x'b + alpha_1[d1] + alpha_2[d2] + ... + offset)` for any
nonnegative outcome (zeros welcome; `y` need not be a count). Only the
conditional mean has to be right (Santos Silva and Tenreyro 2006), so the
default covariance is `robust`. No `Intercept` is reported: the constant is
absorbed.

**Algorithm.** IRLS on the Poisson likelihood that never forms a dummy
variable. With working weights `W = w mu` and working response
`z = eta - offset + (y - mu)/mu`:

1. partial the fixed effects out of `[z X]` in the `W`-weighted inner product
   (`engines.absorb.demean`: exact for one dimension, conjugate-gradient
   accelerated alternating projections for more);
2. weighted QR least squares of `z~` on `X~` gives `b`;
3. by Frisch-Waugh-Lovell the fitted value of the dummy regression is `z`
   minus the residual of step 2, so `eta = offset + z - (z~ - X~ b)` without
   estimating the effects;
4. halve the step while the deviance rises; stop when its relative change is
   at most `tolerance` (default `1e-8`) and no linear predictor is still moving.

The start is `mu_0 = (y + ybar)/2`. The slopes, deviance and log
pseudolikelihood equal those of the Poisson regression on explicit dummies.

**Sample.** Observations in a fixed-effect level whose outcomes are all zero
are dropped iteratively (their effect diverges to minus infinity: "separated
by a fixed effect") and counted in `n_separated_dropped`; singletons are
dropped iteratively when `drop_singletons=True` (`n_singletons_dropped`; with
frequency weights a level is a singleton only when its total frequency is 1).
Regressors without variation within the absorbed levels are omitted and
recorded.

**Degrees of freedom and covariance** (reghdfe's conventions). `K = k +
df_absorbed`, where `df_absorbed` counts observed levels minus redundant ones
(connected components) and does not count dimensions nested in the cluster
variable (one degree of freedom is added back for the constant when every
dimension is nested). With `u = y - mu`, scores `x~_i w_i u_i` and bread
`B = (X~' W X~)^-1` at the converged weights:

    robust (default)   N/(N-K)              B (sum s_i s_i') B
    cluster            G/(G-1) (N-1)/(N-K)  B (sum_g s_g s_g') B     (two columns: G_min)
    nonrobust          B                                             (inverse information)

To compare with Stata's `poisson y x i.d, vce(robust)` on explicit dummies
(factors `N/(N-1)` and `G/(G-1)`), multiply the covariance by
`(N-K)/(N-1)`; the conventional covariance is identical.

Reported: `pseudo_r_squared` (`1 - ll/ll_0`, `ll_0` the constant-only Poisson
model with the offset on the final estimation sample), `log_likelihood` (log
pseudolikelihood), `deviance`, `df_resid`, `df_absorbed`, `iterations`,
`n_separated_dropped`, `n_singletons_dropped`; `tests["model"]` is the Wald
chi2 of the slopes; `extra["absorbed"]` lists column, levels, redundant and
nested per dimension.

```python
ppml = oe.ppmlhdfe(data=df, y="sales", x=["age", "female"], absorb=["firm", "year"],
                   cluster="firm")
print(ppml.summary())
print(ppml.extra["absorbed"])
print(ppml.inference["correction"])
```

```text
Poisson pseudo-likelihood regression with fixed effects — sales
Observations: 2000  |  Covariance: cluster  |  Confidence: 95%

Term     Estimate   Std. error        z     P>|stat|   CI lower   CI upper
------  ---------  -----------  -------  -----------  ---------  ---------
age     0.0185698  0.000938664  19.7833   4.1489e-87  0.0167301  0.0204096
female  -0.222455    0.0227881  -9.7619  1.64058e-22  -0.267119  -0.177791

pseudo_r_squared: 0.212162  |  log_likelihood: -3993.59  |  deviance: 2086.2  |  df_resid: 1992
df_absorbed: 6  |  iterations: 4  |  n_separated_dropped: 0  |  n_singletons_dropped: 0
Wald chi2 test of the slopes: chi2(2) = 426.265, p = 2.74e-93
[{'column': 'firm', 'levels': 100, 'redundant': 100, 'nested': True}, {'column': 'year', 'levels': 6, 'redundant': 0, 'nested': False}]
CR1: G/(G-1) * (N-1)/(N-K)
```

**Limitation: separation by regressors.** ppmlhdfe's general separation search
(ReLU / simplex, which finds observations separated by regressors or by
combinations of regressors and fixed effects and drops them) is NOT
implemented. Such separation is detected during the iteration (zero-outcome
observations whose linear predictor keeps falling while the deviance is flat)
and raises `separation_detected`, naming the diverging regressor; drop that
regressor or the separated observations yourself.

## Errors

Every invalid input or numerical failure is an `oe.AnalysisError` with a
snake_case `code` and a message that says what to change:

| code | meaning |
| --- | --- |
| `invalid_spec` | undeclared option, role, weight type or covariance; `x` not a list; offset and exposure together |
| `invalid_option` | family/link pair not available, `power` without `link="power"`, bad `scale`, `dispersion`, `tolerance` |
| `missing_values`, `empty_sample`, `empty_data`, `missing_columns`, `non_numeric_column`, `non_finite_values` | sample problems |
| `negative_weights`, `noninteger_frequency_weights`, `unsupported_covariance` | weights (the last: `pweight` with `nonrobust`/`opg`) |
| `invalid_count_outcome`, `invalid_positive_outcome`, `invalid_binary_outcome`, `invalid_binomial_outcome`, `invalid_trials`, `invalid_fractional_outcome`, `invalid_exposure` | outcome outside the model's support |
| `constant_outcome` | all zeros (count models), all 0 or all 1 (binary), no variation (fractional) |
| `perfect_fit` | exact fit: the estimated dispersion and all standard errors would be zero |
| `empty_design` | no term left to estimate (or, `ppmlhdfe`, every regressor absorbed) |
| `insufficient_observations` | not more observations than parameters (`ppmlhdfe`: no residual degrees of freedom) |
| `separation_detected` | binomial separation; `ppmlhdfe` separation by a regressor |
| `boundary_solution` | `nbreg` without overdispersion; a `glm` maximum on the edge of the link's domain |
| `nonconvergence`, `invalid_start`, `singular_information`, `numerical_failure` | numerical failures, with the likely cause |
| `insufficient_clusters` | fewer than two clusters |

```python
try:
    oe.glm(data=df.assign(old=(df.age > 40).astype(float)), y="old", x=["age"],
           family="binomial")
except oe.AnalysisError as error:
    print(error.code)
try:
    oe.nbreg(data=df.assign(k=rng.binomial(4, 0.5, n)), y="k", x=["age"])   # underdispersed
except oe.AnalysisError as error:
    print(error.code)
try:
    oe.glm(data=df, y="insured", x=["age", "female"], family="binomial",
           weights="pyears", weight_type="pweight", covariance="nonrobust")
except oe.AnalysisError as error:
    print(error.code)
```

```text
separation_detected
boundary_solution
unsupported_covariance
```

## Performance

Everything is O(n) in time and memory per iteration: designs are n-by-k,
Hessians are k-by-k cross-products, cluster and fixed-effect sums are
`index_add_` passes, and score rows are materialized only when a sandwich or
OPG covariance asks for them. Wall-clock time on one laptop CPU (Apple
silicon), 10 regressors, including sample construction and hashing:

| model | 100,000 rows | 1,000,000 rows |
| --- | --- | --- |
| `poisson` (nonrobust / robust / cluster on 1,000 groups) | 0.1 s | 0.9 - 1.1 s |
| `nbreg` NB2 / NB1 | 0.2 s | 1.7 / 2.1 s |
| `glm` gamma-log, `ml` / `irls` | 0.1 s | 0.9 / 1.0 s |
| `glm` binomial-probit, nbinomial, inverse Gaussian (pweight, robust) | 0.1 s | 0.8 - 1.0 s |
| `cloglog`, `fracreg` | 0.1 s | 1.1 - 1.2 s |
| `betareg` (constant precision / two scale regressors) | 0.2 s | 1.2 / 1.8 s |
| `ppmlhdfe`, 1,000 absorbed levels | 0.14 s | 1.1 s |
| `ppmlhdfe`, 1,000 x 50 levels, clustered | 0.6 s | 4.3 s |

## Deliberate differences from Stata

- Convergence is stricter (scaled gradient 1e-10 instead of 1e-5), so
  estimates agree with Stata's to about six significant digits on flat
  likelihoods rather than exactly to its printed precision.
- `nbreg` on data without overdispersion raises `boundary_solution` instead of
  reporting `alpha` near zero.
- Binomial separation raises `separation_detected` instead of dropping
  perfectly predicted observations (as `logit` does) or reporting huge
  coefficients (as `glm` may do); boundary solutions of the `log` and `identity`
  links raise `boundary_solution` instead of a "convergence not achieved" table.
- With `intercept=False` the LR model test and the pseudo R-squared compare
  with the model whose coefficients are all zero (`eta = offset`); Stata prints
  a Wald test and no pseudo R-squared. `extra["null_model"]` says which
  comparison was used.
- `cloglog` reports McFadden's pseudo R-squared (Stata's header has none) and
  `betareg` reports Ferrari and Cribari-Neto's.
- Two cluster columns (multiway clustering) are accepted by every estimator.
- `fracreg` also accepts `aweight` (handled as in `glm`).
- `ppmlhdfe` does not run the ReLU / simplex separation search (see above) and
  takes interactions as columns you build (`absorb=["exp_year"]`), not as
  `exp#year`.
- `betareg`'s `scale_link="sqrt"` is Stata's `slink(root)`.

## Uncertain conventions

These choices follow the textbook or the specification of this project rather
than Stata output we could check; `provenance["stata_parity_validated"]` is
`False` for every estimator.

1. **`ppmlhdfe` finite-sample factors.** `N/(N-K)` (robust) and
   `G/(G-1) (N-1)/(N-K)` (cluster) are reghdfe's; the Stata package may apply
   different factors to its Poisson sandwich. The conversion to `poisson`'s
   `N/(N-1)` and `G/(G-1)` is `(N-K)/(N-1)` (see above).
2. **`ppmlhdfe` degrees of freedom when a dimension nested in the cluster is
   absorbed together with one that is not**: the non-nested dimension is
   counted in full (it carries the constant); reghdfe may count one less.
3. **`ppmlhdfe` pseudo R-squared**: against the constant-only Poisson model
   (with the offset) on the final estimation sample.
4. **`betareg` model test**: the Wald chi2 of the mean-equation slopes for
   every covariance; Stata's header prints an LR chi2 under its default VCE.
5. **Model test under `opg`**: Wald here; Stata probably keeps the LR test
   because OPG is a model-based variance.
6. **`iweight`**: N is the number of rows and the sandwich meat is
   `sum (w s)(w s)'`; Stata may count the sum of the weights and treat
   importance weights like frequency weights in the meat. OIM and OPG are not
   affected (they are linear in the weights).
7. **Gaussian `glm` log likelihood under `iweight`/`pweight`**: `phi` is
   concentrated with the number of rows.
8. **`glm` with `opg` and an estimated dispersion**: `phi^2 (S'S)^-1` (the
   outer product of the scores of the dispersion-`phi` likelihood).
9. **`glm, irls` with `robust`/`cluster`**: the bread is the expected
   information.
10. **`estat gof`** is reported for every covariance and weight type; Stata
    may not offer it after robust or sampling-weighted fits, where the chi2
    reference distribution is not justified.

## What has been checked, and how

No formula of the implementation is reused by the tests; numpy, scipy and
statsmodels appear only there, as independent oracles.

- **Real Stata output** (the `e()` results that statsmodels ships in its test
  suite): `glm, family(poisson)` for no weights, `fweight`, `aweight`,
  `pweight` crossed with `vce(oim)`, `vce(robust)`, `vce(cluster)`
  (coefficients, standard errors, log likelihood, deviance, Pearson,
  dispersions, AIC, BIC, model chi2, p-values, confidence limits); Stata's log
  likelihood, AIC, BIC, deviance and scale for twelve family/link pairs;
  `poisson` and `nbreg` with exposure and robust / cluster covariances on the
  ships data; `poisson`, `nbreg` and `nbreg, dispersion(constant)` on the
  20,190-row RAND health insurance data (Stata 11: coefficients, standard
  errors, confidence limits, `ll`, `ll_0`, LR chi2, pseudo R2, AIC, BIC,
  `alpha`/`delta` with their intervals).
- **Brute force**: per-observation log likelihoods written in NumPy, maximized
  with scipy and polished by Newton steps on Richardson-extrapolated numerical
  derivatives; Hessians, score rows and therefore every covariance (OIM, OPG,
  robust, cluster) for every estimator, 19 family/link pairs, every weight
  type, offsets, exposure, categorical regressors, missing values and
  collinear designs.
- **Explicit dummy-variable Poisson** for `ppmlhdfe` (one and two absorbed
  dimensions, weights, exposure, nested clusters, separated levels,
  singletons), and OpenEconometrics's own `poisson` on dummies up to the documented
  factors.
- **Structural equivalences**: frequency weights = repeated rows; grouped
  binomial = expanded Bernoulli rows; aggregated Poisson cells with exposure =
  individual rows; reflection `y -> 1 - y` (logit and probit symmetric,
  cloglog <-> loglog) for `glm`, `fracreg` and `betareg`; row order and units
  of the regressors; `ml` = `irls`; each command = its `glm` equivalent;
  closed-form constant-only models.
- **Analytic derivatives** against numerical ones
  (`engines.optimize.check_derivatives`) for every family/link pair, both
  negative binomial forms and all twelve beta link pairs.
- **Adversarial inputs**: empty, one-row and `n <= k` samples, all-missing and
  constant columns, outcomes outside the support, bad weights, single
  clusters, text where numbers are needed, extreme magnitudes (1e-8, 1e8),
  separation, boundary solutions, options at their bounds; each ends in a
  correct fit or an `AnalysisError`, never in a NaN.

## References

- Cameron, A. C. and Trivedi, P. K. (2013). *Regression Analysis of Count Data*, 2nd ed.
- Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2011). Robust inference with multiway clustering. *JBES* 29.
- Correia, S., Guimaraes, P. and Zylkin, T. (2020). Fast Poisson estimation with high-dimensional fixed effects. *Stata Journal* 20.
- Ferrari, S. and Cribari-Neto, F. (2004). Beta regression for modelling rates and proportions. *Journal of Applied Statistics* 31.
- Hardin, J. W. and Hilbe, J. M. (2018). *Generalized Linear Models and Extensions*, 4th ed.
- McCullagh, P. and Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
- Papke, L. E. and Wooldridge, J. M. (1996). Econometric methods for fractional response variables. *Journal of Applied Econometrics* 11.
- Santos Silva, J. M. C. and Tenreyro, S. (2006). The log of gravity. *Review of Economics and Statistics* 88.
- Smithson, M. and Verkuilen, J. (2006). A better lemon squeezer? *Psychological Methods* 11.
- StataCorp. *Stata Base Reference Manual*: `glm`, `poisson`, `nbreg`, `cloglog`, `fracreg`, `betareg`, `ml`, `_robust` (Methods and formulas).
