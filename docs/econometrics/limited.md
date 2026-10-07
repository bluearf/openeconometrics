# Limited dependent variables: `tobit`, `truncreg`, `intreg`, `heckman`, `heckprobit`, `ivprobit`, `ivtobit`

Models for outcomes that are censored, truncated, known only up to an
interval, observed for a selected subsample, or driven by endogenous
regressors. Everything on this page is implemented in OpenEconometrics on float64
PyTorch tensors with analytic scores and Hessians (no autograd, no estimation
library at fit time).

```python
import openecon as oe

oe.tobit(data=df, y="hours", x=["age", "educ"], ll=0)                    # tobit hours age educ, ll(0)
oe.truncreg(data=df, y="hours", x=["age", "educ"], ll=0)                 # truncreg hours age educ, ll(0)
oe.intreg(data=df, y_low="wage1", y_high="wage2", x=["age", "educ"])     # intreg wage1 wage2 age educ
oe.heckman(data=df, y="wage", x=["educ", "age"], select="works",
           select_x=["married", "children", "educ", "age"])              # heckman wage educ age, select(works = married children educ age)
oe.heckprobit(data=df, y="private", x=["years", "logptax"], select="vote",
              select_x=["years", "loginc", "logptax"])                   # heckprobit private years logptax, select(vote = years loginc logptax)
oe.ivprobit(data=df, y="fem_work", x=["fem_educ", "kids"],
            endog=["other_inc"], instruments=["male_educ"])              # ivprobit fem_work fem_educ kids (other_inc = male_educ)
oe.ivtobit(data=df, y="fem_inc", x=["fem_educ", "kids"], ll=10,
           endog=["other_inc"], instruments=["male_educ"])               # ivtobit fem_inc fem_educ kids (other_inc = male_educ), ll(10)
```

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `tobit y x1 x2, ll(0)` · EViews `censored(d=n) y c x1 x2` | `oe.tobit(data=df, y="y", x=["x1","x2"], ll=0)` |
| `tobit y x1 x2, ll ul(10)` | `oe.tobit(..., ll="min", ul=10)` |
| `truncreg y x1 x2, ll(0)` · EViews `censored(d=n, t)` | `oe.truncreg(data=df, y="y", x=["x1","x2"], ll=0)` |
| `intreg ylo yhi x1 x2` | `oe.intreg(data=df, y_low="ylo", y_high="yhi", x=["x1","x2"])` |
| `heckman y x1, select(s = z1 x1)` · EViews `heckit(ml)` | `oe.heckman(data=df, y="y", x=["x1"], select="s", select_x=["z1","x1"])` |
| `heckman y x1, select(s = z1 x1) twostep` · EViews `heckit` | `..., method="twostep"` |
| `heckprobit y x1, select(s = z1 x1)` | `oe.heckprobit(data=df, y="y", x=["x1"], select="s", select_x=["z1","x1"])` |
| `ivprobit y x1 (y2 = z1 z2)` | `oe.ivprobit(data=df, y="y", x=["x1"], endog=["y2"], instruments=["z1","z2"])` |
| `ivprobit y x1 (y2 = z1 z2), twostep` | `..., method="twostep"` |
| `ivtobit y x1 (y2 = z1 z2), ll(0)` | `oe.ivtobit(data=df, y="y", x=["x1"], endog=["y2"], instruments=["z1","z2"], ll=0)` |
| `..., vce(robust)` / `vce(opg)` / `vce(cluster g)` | `covariance="robust"` / `"opg"` / `cluster="g"` |
| `... [fweight=n]` | `weights="n", weight_type="fweight"` |
| `..., offset(e)` (tobit, truncreg, intreg) | `offset="e"` |

SPSS has no built-in procedure for these models. Every function is
keyword-only, returns a `ResultBundle` (`summary()`, `to_latex()`,
`model_dump_json()`), and can equally be run as
`oe.fit(ModelSpec(estimator="tobit", ...), data=df)`.

## Conventions shared by the family

**Estimation.** Each likelihood is maximized by Newton-Raphson
(`engines.optimize.maximize_newton`) with the analytic gradient and Hessian of
the model; the tests compare both with numerical derivatives. A step is halved
until the log likelihood rises; in a non-concave region the step is a Marquardt
step. Convergence requires a negative definite Hessian, `g'(-H)^-1 g <= 1e-10`
(Stata's `nrtolerance`, default `1e-5` there), a gradient below `1e-8` and a
relative step below `1e-10`. The record is in `provenance["optimizer"]`. A
likelihood without an interior maximum raises `nonconvergence` (or
`boundary_solution` when a correlation runs to +-1); estimates of a failed
iteration are never returned.

**Centring.** In an equation with a constant the index is
`a + x'b = (a + m'b) + (x - m)'b`. Every estimator of the family therefore
works on regressors centred at their (weighted) means `m` and maps the
constant back exactly, `a = a_c - m'b`, with covariance `J V J'` for the
Jacobian `J` of that linear map (scores and Hessians transform the same way,
so this is exact for every covariance estimator, and for the two-step
estimators). The reported model is the one specified; what changes is the
arithmetic: a regressor with a large level relative to its spread (a date, a
timestamp, an identifier-like number) no longer costs `(mean / spread)^2`
digits. `heckman` / `heckprobit` centre the outcome regressors at the means of
the selected rows; `ivprobit` / `ivtobit` centre the exogenous variables and
the endogenous regressors, so the constants of the reduced forms are mapped
back too (`pi_0 = pi_0c + m_y - m_z'pi`). Equations without a constant
(`intercept=False`) are not centred. Shifting a regressor with unit spread by
`1e8` changes the slopes and their standard errors by about `1e-7` or less in
relative terms (the resolution of the shifted data themselves).

**Scale and correlation parameters.** Standard deviations are estimated as
`ln sigma` and correlations as `atanh rho` (Stata's `/lnsigma`, `/athrho`), so
the iteration is unconstrained. Quantities reported on the natural scale
(`/sigma`, `rho`, `lambda`) use the delta method, which is exact for the point
estimate and reproduces what Stata prints.

**Covariance** (`core.ml_covariance`, Stata's `ml` conventions). `H` is the
Hessian of the weighted log likelihood `sum_i w_i l_i` and `s_i` the score of
observation i (before weights):

| `covariance` | formula | Stata |
| --- | --- | --- |
| `nonrobust` (default) | `(-H)^-1` | `vce(oim)` |
| `opg` | `(sum_i w_i s_i s_i')^-1` | `vce(opg)` |
| `robust` | `N/(N-1) (-H)^-1 M (-H)^-1`, `M = sum_i w_i^2 s_i s_i'` (`sum_i f_i s_i s_i'` under fweights) | `vce(robust)` |
| `cluster` | `G/(G-1) (-H)^-1 [sum_g s_g s_g'] (-H)^-1`, `s_g = sum_{i in g} w_i s_i` | `vce(cluster g)` |

The outer product of gradients estimates the information of `sum_i w_i l_i`,
which is linear in the weights; with integer importance weights `nonrobust`
and `opg` therefore equal the frequency-weighted results.

Giving `cluster=` selects `cluster`. Two cluster columns give two-way
clustering (Cameron-Gelbach-Miller inclusion-exclusion with the smaller
`G/(G-1)`), an OpenEconometrics extension. The convention used is recorded in
`result.inference["correction"]`. The two-step estimators have their own
covariance (described with each estimator) and accept only `nonrobust`.

**Weights.** Every likelihood, score and Hessian sum is `sum_i w_i (.)`:

| `weight_type` | `w_i` | `N` | notes |
| --- | --- | --- | --- |
| `fweight` | integer frequency | `sum w_i` | identical to the data set with rows duplicated, under every covariance; counts (censored, selected, ...) are frequency-weighted too |
| `aweight` | rescaled to sum to the number of rows | rows | |
| `iweight` | used as given (must be positive) | rows | `nonrobust` and `opg` scale with `1 / w` (integer iweights reproduce the fweight covariance) |
| `pweight` | used as given | rows | needs `robust` or `cluster` (`robust` is the default of the convenience functions); the covariance is invariant to the scale of the weights |

Rows with zero weight are excluded. The two-step estimators accept
`fweight` only.

**Sample.** `missing="raise"` (default) refuses incomplete rows;
`missing="drop"` excludes them and says so in `result.warnings`. Columns in
which a missing value is data (an open interval bound of `intreg`, the outcome
of a nonselected observation in `heckman` / `heckprobit`) do not exclude a
row. Collinear regressors are omitted left to right, as Stata does, and listed
in `provenance["omitted_terms"]`. The constant is the term `Intercept`.

**Failure contract.** Every invalid input or numerical failure raises
`AnalysisError(code, message)`; a result never contains NaN. Codes shared by
the family: `perfect_fit` (the regressors fit a continuous outcome exactly, so
`sigma` is zero and the likelihood has no maximum), `nonconvergence`,
`boundary_solution`, `separation_detected` (a probit part is separated),
`singular_information`, `insufficient_observations`, `missing_values`,
`unsupported_covariance` (pweights with `nonrobust` / `opg`; anything but
`nonrobust` for a two-step estimator), `unsupported_weights` (two-step
estimators accept fweights only) and the `invalid_spec` family for
specification mistakes.

**`ModelSpec` fields.** The convenience functions only build a specification;
the same model can be run with `oe.fit(ModelSpec(...), data=df)`:

| estimator | `columns` (roles) | `options` |
| --- | --- | --- |
| `tobit` | `offset` | `ll`, `ul` (numbers), `ll_at_min`, `ul_at_max` (booleans; `ll="min"` / `ul="max"` in `oe.tobit`) |
| `truncreg` | `offset` | `ll`, `ul` |
| `intreg` | `upper` (required; the outcome is the lower bound), `offset` | - |
| `heckman` | `select`, `select_x` (required) | `method`: `"ml"` or `"twostep"` |
| `heckprobit` | `select`, `select_x` (required) | - |
| `ivprobit` | `endogenous`, `instruments` (required) | `method` |
| `ivtobit` | `endogenous`, `instruments` (required) | `method`, `ll`, `ul`, `ll_at_min`, `ul_at_max` |

## `tobit`: censored normal regression

**Model.** `y* = x'b + e`, `e ~ N(0, sigma^2)`, and `y = max(ll, min(y*, ul))`:
the outcome is recorded at a limit whenever the latent value is at or beyond
it. Observations with `y <= ll` are left-censored, with `y >= ul`
right-censored. The coefficients are effects on the latent `y*`.

**Estimator.** Full maximum likelihood,

```
ln L = sum_{uncensored} w [-ln sigma + ln phi((y - x'b)/sigma)]
     + sum_{left}       w ln Phi((ll - x'b)/sigma)
     + sum_{right}      w ln Phi((x'b - ul)/sigma)
```

on `(b, ln sigma)`. `ln Phi` uses `log_ndtr` and the inverse Mills ratio the
scaled complementary error function, so observations far beyond a limit keep
accurate scores and Hessians. Starting values: least squares of `y` on `x` and
its ML residual scale.

**Options.** `ll`, `ul`: a number (Python or NumPy scalar), `"min"` / `"max"`
(the smallest / largest
outcome in the sample, Stata's `ll` / `ul` without a value) or `None`. With no
limit the fit is the normal regression by ML (OLS coefficients,
`sigma^2 = SSR/N`). `offset`: a column added to `x'b` with coefficient one.

**Reported (Stata's conventions).**

- Coefficient tests are **Student t with `N - df_model` degrees of freedom**,
  `df_model` = the number of slope coefficients, under every covariance
  (`metrics["df_resid"]`).
- The ancillary term `/sigma` with its delta-method standard error.
  `extra["variance"]` holds `var(e) = sigma^2` with standard error
  `2 sigma se(sigma)`, the quantity recent Stata versions print as
  `var(e.y)`; `extra["lnsigma"]` the estimated parameter.
- `tests["model"]`: LR chi2(`df_model`) against the constant-only tobit under
  `nonrobust` / `opg`; the Wald `F(df_model, N - df_model)` under `robust` /
  `cluster` (or without a constant).
- `metrics`: `log_likelihood`, `pseudo_r_squared` (McFadden,
  `1 - ll / ll_0`), `aic`, `bic` (`K + 1` parameters), `sigma`,
  `n_left_censored`, `n_uncensored`, `n_right_censored`, `df_model`,
  `df_resid`.
- The chart sample compares the outcome with the linear prediction `x'b`.

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=1000)})
df["y"] = np.maximum(0.0, 0.5 + df.x + rng.normal(size=1000))
result = oe.tobit(data=df, y="y", x=["x"], ll=0)
print(result.summary())
result.metrics["sigma"], result.metrics["n_left_censored"]
```

**Errors.** `invalid_limits` (`ll >= ul`), `no_uncensored_observations`
(every observation is at or beyond a limit; even with censoring on both sides
the likelihood then increases monotonically in `sigma`), `perfect_fit` (a
constant or exactly fitted outcome), `nonconvergence` (for example a regressor
that separates the censored from the uncensored observations).

## `truncreg`: truncated normal regression

**Model.** `y = x'b + e`, `e ~ N(0, sigma^2)`, but units enter the sample
only when `ll < y < ul`. Unlike censoring, nothing is known about the
excluded units, so the density of an observed outcome is divided by the
probability of inclusion:

```
ln L = sum w { -ln sigma + ln phi((y - x'b)/sigma)
               - ln[Phi((ul - x'b)/sigma) - Phi((ll - x'b)/sigma)] }
```

**Estimator.** Rows with `y <= ll` or `y >= ul` are excluded with a warning
and counted in `metrics["n_truncated"]` (Stata's "obs. truncated");
`result.nobs` is the retained sample. Newton-Raphson on `(b, ln sigma)` from
least squares on the retained sample. The likelihood is not globally concave.

**Reported.** z tests; the ancillary `/sigma` (delta method);
`tests["model"]` is the Wald chi2 of the slopes under every covariance (what
Stata prints); `metrics`: `log_likelihood`, `aic`, `bic`, `sigma`,
`n_truncated` (frequency-weighted under fweights), `df_model`. No pseudo
R-squared. `ll` and `ul` are numbers (Stata also accepts variables; see
Limitations). `empty_sample` when no outcome lies strictly inside the limits.

```python
sample = df[df.y > 0]                       # only positive outcomes are sampled
oe.truncreg(data=sample, y="y", x=["x"], ll=0).summary()
```

## `intreg`: interval regression

**Model.** `y* = x'b + e`, `e ~ N(0, sigma^2)`; `y*` is known through two
columns:

| `y_low` | `y_high` | observation | likelihood term |
| --- | --- | --- | --- |
| `a` | `a` | point | `-ln sigma + ln phi((a - x'b)/sigma)` |
| `a` | `b > a` | interval | `ln[Phi((b - x'b)/sigma) - Phi((a - x'b)/sigma)]` |
| missing | `b` | left-censored | `ln Phi((b - x'b)/sigma)` |
| `a` | missing | right-censored | `ln Phi((x'b - a)/sigma)` |

A row with both bounds missing carries no information: it raises
`missing_values`, or is excluded under `missing="drop"`. `y_low > y_high`
raises `invalid_interval` (Stata silently ignores such rows). A sample that
contains only left-censored or only right-censored observations, or only
censored observations at one common value (a probit), raises
`no_uncensored_observations`: the scale is not identified. The interval
probability is formed in the tail where both distribution functions are small.
`tobit` is the special case with a common limit; the two commands give
identical estimates on equivalent data.

**Estimator.** Newton-Raphson on `(b, ln sigma)` from least squares of the
interval midpoints (the finite bound of censored observations).

**Reported.** z tests; the ancillary `/lnsigma` (what Stata's `e(b)` holds);
`extra["sigma"]` = `exp(lnsigma)` with its delta-method standard error and the
interval obtained by exponentiating the `lnsigma` interval, as Stata prints;
`tests["model"]`: LR chi2 against the constant-only model under `nonrobust` /
`opg`, Wald chi2 otherwise; `metrics`: `log_likelihood`, `aic`, `bic`, `sigma`,
`n_left_censored`, `n_uncensored` (point data, Stata's "Uncensored"; repeated
as `n_point`), `n_right_censored`, `n_interval`, `df_model`. The chart sample
compares `x'b` with the interval midpoint (or the finite bound).

```python
latent = 1.0 + 0.8 * df.x + rng.normal(size=1000)
df["lo"], df["hi"] = np.floor(latent), np.floor(latent) + 1      # known up to its unit
oe.intreg(data=df, y_low="lo", y_high="hi", x=["x"]).summary()
```

## `heckman`: regression with sample selection

**Model.**

```
outcome:    y = x'b + u1                 (observed only when select = 1)
selection:  select = 1[z'g + u2 > 0]
u1 ~ N(0, sigma^2),  u2 ~ N(0, 1),  corr(u1, u2) = rho
```

With `rho != 0`, least squares on the selected sample omits
`E[u1 | selected] = rho sigma lambda(z'g)`, `lambda(c) = phi(c)/Phi(c)` (the
inverse Mills ratio). `select` is a 0/1 column, `select_x` the regressors of
the selection equation (a constant is always added). `y` may be missing where
`select = 0`, and values recorded there are ignored; `x` must be complete in
every row (see Limitations). The model is best identified when `select_x`
contains a regressor that is excluded from `x`.

**`method="ml"` (default).** Full maximum likelihood over
`(b, g, athrho, lnsigma)`, `rho = tanh(athrho)`, `sigma = exp(lnsigma)`:

```
ln L = sum_{selected} w { ln Phi[(z'g + rho e) / sqrt(1 - rho^2)] - e^2/2 - ln(sqrt(2 pi) sigma) }
     + sum_{nonselected} w ln Phi(-z'g),          e = (y - x'b) / sigma
```

Newton-Raphson with the analytic score and Hessian (only univariate normal
functions are needed), started at the two-step estimates (`rho` clipped to
+-0.9). Reported: the outcome equation (equation = the outcome),
`select:<term>` (equation `select`), `/athrho`, `/lnsigma`; `metrics`:
`log_likelihood`, `aic`, `bic`, `rho`, `sigma`, `lambda = rho sigma`,
`n_selected`, `n_censored` (the nonselected observations); `extra["rho"]`,
`extra["sigma"]`, `extra["lambda"]` with delta-method standard errors (and
`tanh` / `exp` transformed intervals for `rho` / `sigma`), as Stata prints.
`tests["model"]` is the Wald chi2 of the outcome slopes; `tests["rho"]` is the
LR test of independent equations, `2 (ll - ll_probit - ll_regression)` with
one degree of freedom, under `nonrobust` / `opg`, and the Wald test of
`athrho = 0` under `robust` / `cluster`. A correlation that runs to +-1
raises `boundary_solution`: in small samples the selection likelihood can
increase all the way to `|rho| = 1` (Stata then reports `rho = +-1` with
missing standard errors). The likelihood is not globally concave; the reported
maximum is the one reached from the two-step estimates. Selected outcomes
fitted exactly by `x` raise `perfect_fit`.

**`method="twostep"`** (Heckman 1979; Stata's `heckman, twostep`).

1. Probit of `select` on `z`: `g`, covariance `V_p` (observed information).
2. `lambda_i = phi(z_i'g)/Phi(z_i'g)`, `delta_i = lambda_i (lambda_i + z_i'g)`.
3. Least squares (QR) of `y` on `X* = [x, lambda]` over the selected rows:
   `b`, `b_lambda`, residuals `e`.
4. `sigma^2 = (e'e + b_lambda^2 sum delta_i) / N_1`, `rho = b_lambda / sigma`.
   When `|rho| > 1`, `rho` is truncated to +-1 and `sigma = |b_lambda|`
   (Stata's default `rhosigma` rule), with a warning and
   `extra["rho_truncated"] = True`.
5. `V = sigma^2 (X*'X*)^-1 [X*'(I - rho^2 D) X* + Q] (X*'X*)^-1`,
   `Q = rho^2 (X*'D Z) V_p (Z'D X*)`, `D = diag(delta_i)`.

Reported: the outcome equation, `select:<term>` with the probit covariance,
and `mills:lambda` (the coefficient of the inverse Mills ratio); `metrics`:
`rho`, `sigma`, `lambda`, `n_selected`, `n_censored`; `tests["model"]`. The
covariance between the outcome block and the selection block is
`b_lambda (X*'X*)^-1 (X*'D Z) V_p` (see Uncertain conventions). Only
`covariance="nonrobust"` and frequency weights are accepted.

```python
rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({"educ": rng.normal(size=n), "kids": rng.normal(size=n)})
u = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
df["works"] = (0.4 + 0.5 * df.educ - 0.8 * df.kids + u[:, 1] > 0) * 1.0
df["wage"] = np.where(df.works == 1, 1.0 + 0.7 * df.educ + u[:, 0], np.nan)
ml = oe.heckman(data=df, y="wage", x=["educ"], select="works", select_x=["educ", "kids"])
two = oe.heckman(data=df, y="wage", x=["educ"], select="works", select_x=["educ", "kids"],
                 method="twostep")
ml.metrics["rho"], ml.tests["rho"]["p_value"], two.metrics["lambda"]
```

The chart sample of `heckman` and `heckprobit` holds the fitted selection
probability `Phi(z'g)` against the selection indicator, because the outcome
does not exist for nonselected rows.

## `heckprobit`: probit with sample selection

**Model.** `y = 1[x'b + u1 > 0]` is observed only when
`select = 1[z'g + u2 > 0]` equals one; `(u1, u2)` are standard bivariate
normal with correlation `rho`:

```
ln L = sum_{s=1, y=1} w ln Phi2(x'b, z'g; rho)
     + sum_{s=1, y=0} w ln Phi2(-x'b, z'g; -rho)
     + sum_{s=0}      w ln Phi(-z'g)
```

**Estimator.** Newton-Raphson over `(b, g, athrho)` with the analytic score
and analytic Hessian. A selected row is a bivariate-probit observation whose
second outcome is one, so the kernel reuses the discrete family's bivariate
normal distribution function (Genz's algorithm, absolute error about 1e-15,
with a tail-accurate logarithm for outlying observations). Starting values:
the probit of `y` on the selected sample, the probit of the selection
equation and `rho = 0`.

**Reported.** z tests; the outcome equation, `select:<term>`, `/athrho`;
`metrics`: `log_likelihood`, `aic`, `bic`, `rho`, `n_selected`, `n_censored`;
`extra["rho"]` (delta-method standard error, `tanh` interval);
`tests["model"]`: Wald chi2 of the outcome slopes; `tests["rho"]`: LR test
against the two separate probits (`nonrobust`, `opg`) or the Wald test
(`robust`, `cluster`). `boundary_solution` when the correlation runs to +-1,
`separation_detected` when the selection (or outcome) probit is separated.

```python
df["hired"] = np.where(df.works == 1, (0.2 + 0.8 * df.educ + u[:, 0] > 0) * 1.0, np.nan)
oe.heckprobit(data=df, y="hired", x=["educ"], select="works",
              select_x=["educ", "kids"]).summary()
```

## `ivprobit`, `ivtobit`: endogenous continuous regressors

**Model.**

```
outcome:       y1* = x1'g + y2'b + u            d = (g, b),  w = [x1, y2]
reduced forms: y2  = Pi z + v                   z = [x1, x2]  (x2 = excluded instruments)
(u, v) ~ N(0, Sigma)
```

`ivprobit`: `y1 = 1[y1* > 0]` and `var(u) = 1`. `ivtobit`:
`y1 = max(ll, min(y1*, ul))` and `var(u) = sigma_1^2` is estimated. The
endogenous regressors `endog` are continuous; they are correlated with `u`
through `Cov(v, u)`, and the model is identified by the excluded
`instruments` (at least one per endogenous regressor). `x` (the exogenous
regressors) is optional. `intercept=False` removes the constant from the
outcome equation and from the reduced forms.

**`method="ml"` (default).** The joint density is
`f(y1 | y2, z) f(y2 | z)` with `u | v ~ N(a'v, omega^2)`,
`a = Var(v)^-1 Cov(v, u)`, `omega^2 = var(u) - Cov(u, v) a`:

```
ivprobit:  ln L = sum w { ln Phi[q (w'd + a'v) / omega] + ln phi_p(v; Var(v)) },   q = 2 y1 - 1
ivtobit:   ln L = sum w { censored-normal term with mean w'd + a'v and sd omega + ln phi_p(v; Var(v)) }
```

The iteration runs in a recursive working parameterization in which every
equation is a regression with an independent error
(`r_j = y2_j - [z, y2_1..y2_{j-1}]'f_j`, index `w'd~ + r'k`; see
`limited/iv_kernels.py`). It is unconstrained for any number of endogenous
regressors, has a simple analytic score and Hessian, and costs
`O(n (k_z + p)^2)` per iteration. Starting values are the recursive
first-stage least squares and the control-function probit / tobit; in a
just-identified model these already are the ML estimates. The estimates are
then mapped exactly to Stata's parameters and their covariance obtained as
`J V J'` with the analytic Jacobian `J` (the delta method is exact for a
reparameterized maximum, for every covariance estimator).

Reported terms (ML):

| block | terms | equation |
| --- | --- | --- |
| outcome equation | exogenous terms, then the endogenous regressors | the outcome |
| reduced forms | `<endog>:<term>` for every exogenous variable | `<endog>` |
| correlations | `/athrho{i}_{j}` = `atanh corr(e_i, e_j)`, `i > j` | ancillary |
| scales | `/lnsigma{i}` = `ln sd(e_i)` | ancillary |

Equation 1 is the outcome equation and equation `m + 1` the m-th endogenous
regressor, so with one endogenous regressor `ivprobit` reports `/athrho2_1`
and `/lnsigma2` and `ivtobit` additionally `/lnsigma1` (Stata 15+ names; the
older output calls them `/athrho` and `/lnsigma`, and `/alpha`, `/lns`, `/lnv`
for `ivtobit`). `extra["correlations"]` and `extra["standard_deviations"]`
give `corr(e.y2, e.y1)`, `sd(e.y2)`, ... on the natural scale with
delta-method standard errors, as Stata prints them.

- `tests["model"]`: Wald chi2 of all slopes of the outcome equation.
- `tests["exogeneity"]`: Wald chi2(p) that every `corr(v_j, u)` is zero, i.e.
  `/athrho{j+1}_1 = 0` for all j (Stata's "Wald test of exogeneity").
- `metrics`: `log_likelihood`, `aic`, `bic`; for `ivtobit` also
  `n_left_censored`, `n_uncensored`, `n_right_censored` and `sigma` (`sd(u)`).
- z statistics for both commands (Stata's `ivtobit`, unlike `tobit`, reports
  z).
- `extra`: `exogenous`, `endogenous`, `instruments` (after omitting collinear
  ones), `first_stage` and the censoring `limits` (`ivtobit`).

**First stage.** `extra["first_stage"][<endog>]` holds, from the least-squares
regression of the endogenous regressor on all exogenous variables:
`r_squared`, `adjusted_r_squared`, `partial_r_squared` (of the excluded
instruments), `f_statistic` with `f_df1` = number of instruments and
`f_df2 = N - kz`, `f_p_value` and `rmse`. A small F warns of weak instruments.

**`method="twostep"`** (Newey 1987, Stata's `twostep` option). With `V-hat`
the first-stage residuals and `Pi-hat` the first-stage coefficients:

1. Reduced-form probit (tobit) of `y1` on `[z, V-hat]`: `alpha` (coefficients
   of `z`, covariance block `J_aa^-1`) and `lambda`.
2. Two-stage conditional (Rivers-Vuong, 2SIV) probit (tobit) of `y1` on
   `[w, V-hat]`: a consistent `beta` for the endogenous regressors. The Wald
   test that its `V-hat` coefficients are zero is `tests["exogeneity"]`.
3. Least squares of `y2 (lambda - beta)` on `z`; its covariance
   `s^2 (z'z)^-1`, `s^2 = SSR/(N - kz)`, gives
   `Omega = J_aa^-1 + s^2 (z'z)^-1`.
4. `d = (D' Omega^-1 D)^-1 D' Omega^-1 alpha`, `Var(d) = (D' Omega^-1 D)^-1`,
   `D = [I_1, Pi-hat]`.

Only the outcome equation is reported. The `ivprobit` two-step coefficients
are those of the index conditional on the first-stage errors: they equal the
ML coefficients divided by `sd(u | v)`, so the two sets are not directly
comparable (ratios of coefficients are). Only `covariance="nonrobust"` and
frequency weights are accepted.

```python
rng = np.random.default_rng(0)
n = 3000
df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
df["y2"] = 0.5 * df.x + 0.8 * df.z + e[:, 1]
df["work"] = (0.2 + 0.6 * df.x - 0.7 * df.y2 + e[:, 0] > 0) * 1.0
df["hours"] = np.maximum(0.0, 0.5 + 0.6 * df.x - 0.7 * df.y2 + e[:, 0])

probit = oe.ivprobit(data=df, y="work", x=["x"], endog=["y2"], instruments=["z"])
probit.tests["exogeneity"], probit.extra["first_stage"]["y2"]["f_statistic"]
tobit = oe.ivtobit(data=df, y="hours", x=["x"], endog=["y2"], instruments=["z"], ll=0,
                   method="twostep")
```

**Errors.** `underidentified` (fewer usable instruments than endogenous
regressors, or first-stage residuals collinear with the regressors),
`collinear_endogenous` (an endogenous regressor is an exact combination of the
exogenous variables), `invalid_spec` (a column in two roles),
`nonconvergence`, `boundary_solution` (an error correlation of +-1),
`separation_detected` (the probit is separated).

## Limitations

- `tobit`, `truncreg`: the limits are numbers common to all observations.
  Stata's variable limits (`ll(varname)`) are not supported; observation-
  specific censoring can be written as `intreg` bounds.
- `intreg`: Stata's `het()` (multiplicative heteroskedasticity) and
  constraints are not implemented.
- `heckman`, `heckprobit`: the selection indicator must be given explicitly
  (Stata can infer it from a missing outcome), the selection equation always
  has a constant, and the outcome regressors `x` must be complete in every
  row, including nonselected ones. Offsets and `heckman`'s `first` /
  `mills()` / `nshazard()` options are not implemented.
- `heckman, twostep` and the two-step `ivprobit` / `ivtobit` accept frequency
  weights only and provide no robust or cluster covariance.
- `ivprobit` / `ivtobit`: endogenous regressors must be continuous; there is
  no overidentification test and no `first` display beyond
  `extra["first_stage"]`.
- Post-estimation (`predict`, `margins`, the censored and truncated means of
  `tobit`) is not part of this family. The chart sample holds the linear
  prediction (or the fitted selection probability / `Phi(w'd)`).
- Results have not been compared with Stata output:
  `provenance["stata_parity_validated"]` is `False`. Every estimator is tested
  against an independent NumPy/SciPy maximum likelihood or explicit-formula
  oracle.

## Uncertain conventions

None of the conventions below was checked against Stata output
(`provenance["stata_parity_validated"]` is `False`); the confidence is the
verifier's.

- **`tobit` degrees of freedom** (high confidence for `nonrobust` / `robust`,
  low for `cluster`). Coefficient tests use Student t with `N - df_model`
  degrees of freedom (`df_model` = number of slopes; the constant and `sigma`
  are not subtracted). This is the value implied by the confidence intervals
  of the published Stata example (`tobit mpg wgt, ll(17)`, N = 74: the
  interval of `wgt` is `-6.87305 +- 1.99300 * 0.7002559`, and `1.99300` is the
  97.5% point of `t(73)`, not of `t(72)`). The same degrees of freedom are
  used under `cluster`, where Stata may use `G - 1`. The robust model test is
  `F(df_model, N - df_model)`.
- **`tobit` reporting** (high). `/sigma` is reported (Stata 14 and earlier);
  Stata 15+ prints `var(e.y)`, available as `extra["variance"]` (its interval
  in Stata is log-transformed; only the estimate and the delta-method standard
  error are stored here).
- **LR versus Wald model test** (medium). `tobit` and `intreg` report the LR
  test under `nonrobust` and `opg` and the Wald test under `robust` /
  `cluster` and without a constant; whether Stata switches to Wald under
  `vce(opg)` was not verified. The same rule decides between the LR and Wald
  tests of `rho = 0` in `heckman` and `heckprobit`.
- **`opg` with analytic / importance weights** (medium-high). `opg` is
  `(sum_i w_i s_i s_i')^-1`: the information of `sum w_i l_i` is linear in the
  weights and Stata treats iweights like non-integer frequencies in
  likelihood-based variance estimates. (The sandwich meats of `robust` and
  `cluster` use `w_i^2`, as for sampling weights.)
- **`N` under iweights** (medium). `N` is the number of rows for aweights,
  iweights and pweights and the sum of the weights for fweights; it enters
  `bic`, `N/(N-1)` and the tobit degrees of freedom.
- **`truncreg` count of truncated observations** (medium). `n_truncated` is
  frequency-weighted under fweights, like `N`.
- **`intreg` with `y_low > y_high`** (high). OpenEconometrics raises
  `invalid_interval`; Stata ignores such observations.
- **`heckman, twostep` cross covariance** (medium). The covariance between the
  outcome block and the selection block is `b_lambda (X*'X*)^-1 (X*'D Z) V_p`,
  the term implied by Heckman's expansion; whether Stata's `e(V)` stores this
  block or zeros was not verified. The diagonal blocks follow Stata's Methods
  and formulas (high).
- **`heckman, twostep` model test** (medium). `tests["model"]` is the Wald
  chi2 of the outcome slopes (recent Stata output); older Stata versions also
  counted the selection equation.
- **Weights accepted** (medium). As far as the Stata documentation is
  remembered: `heckman` ML takes pweights, aweights, fweights and iweights and
  `heckman, twostep` no weights; `heckprobit`, `ivprobit` and `ivtobit` (ML)
  take fweights, iweights and pweights; the two-step IV estimators fweights.
  OpenEconometrics accepts all four types for every ML estimator (aweights rescaled to
  sum to N) and fweights for every two-step estimator (identical to replicated
  rows, which the tests verify).
- **`ivprobit` / `ivtobit` ancillary parameters** (high for one endogenous
  regressor, medium for several). The `/athrho{i}_{j}` / `/lnsigma{i}` layout
  for every pair of equations follows the Stata 15+ output for one endogenous
  regressor; for several endogenous regressors Stata's exact ancillary
  parameterization (and therefore the exact Wald statistic of exogeneity,
  which is not invariant to reparameterization) was not verified. Point
  estimates of coefficients and reduced forms are invariant.
- **Order of the outcome equation** (high). OpenEconometrics lists the exogenous terms
  (constant first) and then the endogenous regressors; Stata lists the
  endogenous regressors first and the constant last.
- **Two-step weights** (medium). The auxiliary regression of Newey's step 3
  uses `N - kz` degrees of freedom with `N` the sum of the frequencies.
- **Two-way clustering** is an OpenEconometrics extension (Stata's commands take one
  cluster variable).

## Verification

Besides the implementer's tests (brute-force likelihoods, derivative checks
with `engines.optimize.check_derivatives`), the family is checked by
independently written oracles in `tests/test_econ_limited_oracle*.py`:

- every likelihood is rewritten from the model definition in a different
  parameterization (tobit in Olsen's `(b/sigma, 1/sigma)`, truncreg in
  `(b, sigma^2)`, intreg in `(b, sigma)`, the selection models in
  `(b, g, rho, sigma)` with a Plackett-quadrature bivariate normal, the IV
  models in the free elements of `Sigma = Var(u, v)`), maximized by Newton
  steps on complex-step scores, and mapped to the reported parameters with the
  Jacobian of the reparameterization;
- coefficients (relative tolerance about `1e-6`), full covariance matrices,
  p-values, confidence intervals, model / `rho` / exogeneity tests, information
  criteria, counts and the natural-scale records are compared for every
  covariance estimator and weight type;
- the Heckman two-step and Newey estimators are recomputed from their formulas
  on statsmodels probit / OLS fits;
- invariances: frequency weights = replicated rows, integer iweights =
  fweights (`nonrobust`, `opg`), units and levels of regressors and outcomes,
  row order, tobit = intreg on equivalent bounds, no limits = least squares,
  just-identified two-step = control function, singleton clusters = robust,
  two-way clustering = inclusion-exclusion of one-way fits.

`tests/test_econ_limited_regressions.py` holds one test per defect fixed in
that stage (centring, `opg` weights, `perfect_fit`, samples without
identifying observations, NumPy limits) and the adversarial inputs.

## Performance

Synthetic data, 10 regressors (plus instruments / selection regressors),
six CPU threads, wall time of the whole call including sample construction
and result assembly:

| call | 100,000 rows | 1,000,000 rows | Newton iterations |
| --- | --- | --- | --- |
| `tobit` (30% censored), `nonrobust` / `robust` / `cluster` | 0.17 s / 0.18 s / 0.19 s | 1.3 s / 1.3 s / 1.4 s | 5 |
| `truncreg` | 0.17 s | 1.2 s | 5 |
| `intreg` (unit-wide intervals) | 0.20 s | 1.3 s | 3 |
| `heckman` ML, `nonrobust` / `robust` | 0.27 s / 0.29 s | 2.0 s / 2.0 s | 2-3 |
| `heckman` two-step | 0.19 s | 1.3 s | - |
| `heckprobit` | 0.37 s | 3.0 s | 5 |
| `ivprobit` ML, `nonrobust` / `cluster` | 0.26 s / 0.38 s | 2.3 s / 2.4 s | 2 |
| `ivprobit` two-step | 0.44 s | 2.1 s | - |
| `ivtobit` ML | 0.32 s | 2.5 s | 2 |
| `ivtobit` two-step (two tobit fits of 6 iterations) | 0.34 s | 2.7 s | - |

Timings vary by a factor of about two between runs on a busy machine.

Time grows linearly in the number of rows. Every likelihood evaluation is a
fixed number of passes over the rows (`O(n k^2)` for the Hessian cross
products, accumulated in blocks); there is no loop over observations and no
n-by-n matrix. Score rows (n-by-K) are formed only when a robust, cluster or
OPG covariance needs them.
