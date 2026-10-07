# Linear regression family: `areg`, `reghdfe`, `cnsreg` (and the `regress` / `newey` wrappers)

Least-squares estimators of a linear conditional mean, implemented in OpenEconometrics
on float64 PyTorch tensors (no statsmodels, linearmodels or SciPy at fit
time). Every estimator solves its least-squares problem by one Householder QR
of the weighted design (`engines.linalg.least_squares`): `X'X` is never formed
or inverted. The conventions below follow the *Methods and formulas* of the
named Stata command; `provenance["stata_parity_validated"]` stays `False`
because no result has yet been compared with real Stata output.

| OpenEconometrics | Stata | SPSS | EViews | implemented by |
| --- | --- | --- | --- | --- |
| `oe.areg` | `areg ..., absorb(var)` | — | — | this family |
| `oe.reghdfe` | `reghdfe ..., absorb(v1 v2 ...)` (Correia 2017) | — | — | this family |
| `oe.cnsreg` | `cnsreg ..., constraints(...)` | — | LS with coefficient restrictions | this family |
| `oe.regress` | `regress` | REGRESSION, WLS | LS | thin wrapper over [`oe.ols`](../ols.md) |
| `oe.newey` | `newey ..., lag(#)` | — | LS with HAC (Newey-West) | thin wrapper over [`oe.ols`](../ols.md) |

Plain OLS/WLS is **not** estimated by this family. It belongs to the `ols`
estimator of `openecon.linear_ols` (`oe.ols`), which owns the HC0–HC3,
cluster, HAC, bootstrap and jackknife covariances and the post-estimation
commands (`oe.test`, `oe.lincom`, `oe.nlcom`, `oe.predict`, `oe.margins`); see
[OLS and weighted least squares](../ols.md). `"regress"` is therefore not an
estimator name for `oe.fit` / `ModelSpec`: use `estimator="ols"`.

Every function returns a `ResultBundle`: `coefficients` (estimate, standard
error, t statistic, p-value, confidence interval), `covariance_matrix`,
`metrics`, `tests`, `extra`, `warnings`, `inference` (which covariance,
small-sample factor and degrees of freedom were used) and `provenance`
(omitted terms, solver, hashes). `result.summary()` prints the table;
`result.to_latex()` exports it.

## `oe.regress` and `oe.newey` — Stata-named wrappers over `oe.ols`

```python
oe.regress(*, data, y, x, covariance=None, cluster=None, weights=None, weight_type=None,
           lags=None, kernel=None, time=None, categorical=None, intercept=True,
           missing="drop", alpha=0.05)
oe.newey(*, data, y, x, lag, time=None, panel=None, kernel="bartlett", weights=None,
         weight_type=None, categorical=None, intercept=True, missing="drop", alpha=0.05)
```

Both functions perform **no estimation of their own**. They translate Stata's
vocabulary into `oe.ols` keyword arguments, call `openecon.analysis.ols(...)`
and return its result unchanged (`result.spec.estimator == "ols"`), so
`oe.regress(...)` and the equivalent `oe.ols(...)` call give identical
coefficients, covariance, metrics and tests. All statistical conventions are
those documented in [ols.md](../ols.md).

| argument | translation |
| --- | --- |
| `covariance="robust"` (Stata `vce(robust)`) | `covariance="HC1"` |
| any other `covariance`, `cluster`, `weights`, `weight_type`, `lags`, `kernel`, `time`, `categorical`, `intercept`, `missing`, `alpha` | passed through as given |
| `oe.newey(lag=L, kernel=..., time=...)` | `oe.ols(covariance="hac", lags=L, kernel=..., time=...)` |

What the wrappers reject before calling `oe.ols`:

| request | error | reason |
| --- | --- | --- |
| `newey(panel=...)` | `unsupported_option` | `oe.ols` has no panel option (Stata's `newey ..., force` on `xtset` data) |
| HAC without `time` (`newey(time=None)`, `regress(covariance="hac")` without `time`) | `unsupported_option` | `oe.ols` forms autocovariances on an explicit integer period index and never assumes that rows are consecutive periods; Stata's `newey` likewise requires `tsset` data |
| HAC with `fweight`, `pweight` or `iweight` | `unsupported_weights` | Stata's `newey` allows analytic weights only |
| `newey(lag=...)` not a nonnegative integer; `x` given as a bare string | `invalid_spec` | |

Everything else `oe.ols` rejects surfaces with its own error code (for
example `invalid_spec` for `lags` without `covariance="hac"`, `invalid_time`
for repeated periods). `missing` defaults to `"drop"`, as in `oe.ols`.

| Stata | OpenEconometrics |
| --- | --- |
| `regress y x1 x2` | `oe.regress(data=df, y="y", x=["x1", "x2"])` |
| `regress y x1 [aw=w], vce(cluster firm)` | `oe.regress(..., weights="w", weight_type="aweight", cluster="firm")` |
| `tsset t` + `newey y x, lag(3)` | `oe.newey(data=df, y="y", x=["x"], lag=3, time="t")` |

---

## Conventions shared by `areg`, `reghdfe` and `cnsreg`

**Collinearity.** Columns are screened left to right after sweeping the
constant (as Stata does) and omitted columns are listed in
`provenance["omitted_terms"]` with the warning *"Omitted because of
collinearity: ..."*. In `areg` and `reghdfe` a regressor without within
variation is omitted as *"collinearity with the absorbed fixed effects"*: its
within sum of squares is at most `1e-13` times its mean-deviated original sum
of squares (`ModelFrame.drop_absorbed`). A regressor with a large offset or
large between-level variation but real within variation
(`x = 1e4 * firm + noise`) is kept.

**Weights** (`weights=` column plus `weight_type=`):

| type | point estimates | N | notes |
| --- | --- | --- | --- |
| `aweight` | WLS with `w` rescaled to sum to N | rows | Stata's analytic weights |
| `fweight` | WLS with the integer `w` | `sum w` | identical to the duplicated-row data set for every covariance, including reghdfe's singleton rule |
| `pweight` | as aweight | rows | sampling weights: need `HC1` (`robust`) or `cluster`; the functions default to `HC1`; with `nonrobust` the fit raises `unsupported_covariance` |

`iweight` is not offered (see *Limitations*).

**Covariance estimators** (`core.linear_covariance`, with `K` the number of
parameters as the estimator counts them, `N` as above):

| name | formula | small-sample factor |
| --- | --- | --- |
| `nonrobust` | `s^2 (X'WX)^-1`, `s^2 = RSS / df_resid` | — |
| `HC1` = `robust` (Stata `vce(robust)`) | `(X'WX)^-1 sum (w_i u_i)^2 x_i x_i' (X'WX)^-1` | `N/(N-K)` |
| `cluster` (CR1) | `sum_g t_g t_g'`, `t_g` the cluster score sum | `G/(G-1) (N-1)/(N-K)` |
| two cluster columns | Cameron-Gelbach-Miller inclusion-exclusion `M_1 + M_2 - M_12` | `G_min/(G_min-1) (N-1)/(N-K)` |

`robust` is accepted both by the functions and in a directly built
`ModelSpec`. The functions write `HC1` into the spec; a spec that says
`robust` keeps `spec.covariance == "robust"` and reports
`inference["covariance"] == "HC1"`.

The two-way meat `M_1 + M_2 - M_12` need not be positive semidefinite with few
clusters. Its negative eigenvalues are then set to zero (Cameron, Gelbach and
Miller 2011, eq. 2.13 — what reghdfe does), `inference["psd_adjusted"]` is
`True` and a warning is recorded. The fit never fails for that reason.

Coefficient tests are Student t: `df_resid` as the estimator defines it,
`G - 1` with one cluster column and `G_min - 1` with two. Confidence intervals
use the same distribution.

**Model test.** `tests["model"]` is the classical F with the conventional
covariance and the Wald F built on the reported covariance otherwise, as
Stata prints it. A cluster covariance has rank at most `G - 1`; with fewer
clusters than tested coefficients the generalized Wald statistic uses fewer
restrictions. The reduced test is then kept but marked: `rank_deficient:
True`, `restrictions` (the number asked for), a label ending in
*"(rank-reduced to r of q restrictions)"* and a warning. Stata prints a
missing F in that situation.

**Exact fits.** An exact fit has zero error variance and no standard errors;
it raises `perfect_fit`. The residual sum of squares is judged against the
rounding floor of the computation: `RSS <= 1e-28 * sum w y^2`. Residuals
carry an absolute rounding error of about `eps * |y|`, so the RSS of an exact
fit floors near `eps^2 ~ 5e-32` times the **uncentered** sum of squares
whatever the level of `y`; the threshold sits about `2e3 eps^2` above that
floor. A large offset therefore does not trigger the guard
(`y = 1e9 + 2x + 1e-3 e` is fitted normally) unless the residual variation is
below the float64 resolution of the level itself (`sigma/level < 1e-14`). A
purely centered scale cannot serve as the criterion: a true exact fit of
`y = 1e10 + 2x` leaves `RSS/TSS_centered ~ 1e-11`. `reghdfe` with two or more
dimensions adds a second floor, see below.

**Too few observations** raise `insufficient_observations`: no residual
degrees of freedom (`df_resid <= 0`) in every estimator; in `cnsreg` no more
observations than design columns; in `areg` no more observations than
absorbed levels. The row count is compared (frequency weights add no rank).

**Missing values.** `missing="raise"` (default) stops with the code
`missing_values`; `missing="drop"` excludes incomplete rows and records how
many. **Errors** are `openecon.AnalysisError(code, message)` with a
snake_case code (`unsupported_covariance`, `invalid_constraint`,
`inconsistent_constraints`, `insufficient_observations`,
`insufficient_groups`, `insufficient_clusters`, `perfect_fit`,
`constant_outcome`, `empty_design`, `empty_sample`,
`absorption_nonconvergence`, `invalid_option`, `invalid_spec`, ...); the
message says what to change.

---

## `oe.areg` — one absorbed fixed effect (Stata `areg`)

```python
oe.areg(*, data, y, x, absorb, covariance=None, cluster=None, weights=None, weight_type=None,
        categorical=None, missing="raise", alpha=0.05)
```

**Model.** `y_ig = x_ig'b + alpha_g + u_ig` with `G` levels of `absorb`.
The effects are removed by the (weighted) within transform
`x - mean_g(x)`; the slopes equal those of the dummy-variable regression.
Stata's `areg` reports a constant: the grand means are added back and the
regression `(y~ + ybar) on [1, X~ + xbar]` is solved, so that
`Intercept = ybar - xbar'b` with its standard error from the same problem.
Because `X~` has mean zero the slope covariance is exactly the within one.

**Degrees of freedom — the areg convention.** The absorbed indicators count
as parameters: `K_total = k + G` (k slopes, the constant, `G - 1`
indicators), `df_resid = N - K_total`, `s^2 = RSS/df_resid`. `HC1` uses
`N/(N - K_total)`, `cluster` uses `G_c/(G_c - 1) (N - 1)/(N - K_total)` with
the **same** `K_total` even when the absorbed levels are nested inside the
clusters. This is the documented difference from `xtreg, fe` (and `reghdfe`),
which do not count nested effects: with `absorb(id) cluster(id)` areg's
clustered standard errors are larger by `sqrt((N - k - 1)/(N - k - G))`.
Coefficient tests use `df_resid` (`G_c - 1` with clusters).

**Reported.** `metrics`: `r_squared` (overall, including the fixed effects:
`1 - RSS/TSS` about the grand mean), `adjusted_r_squared` (with `K_total`),
`r_squared_within` (`1 - RSS/TSS_within`), `rmse`, `df_model`, `df_resid`,
`df_absorbed = G - 1`, `n_groups`. `tests["model"]`: F test of the slopes
(classical `(MSS_within/k)/(RSS/df_resid)` or Wald-robust).
`tests["absorbed"]`: `F(G - 1, df_resid) = ((RSS_pooled - RSS)/(G - 1)) /
(RSS/df_resid)` that all level effects are zero, conventional covariance only
(what Stata prints). `extra["absorbed"]` has reghdfe's shape, a list with one
record `{column, levels, redundant, nested}`: `redundant = 1` is the level
absorbed into the reported constant (`levels - redundant = df_absorbed`) and
`nested` says whether the levels nest within a cluster column (recorded for
information; areg counts them regardless).

**Covariances:** `nonrobust`, `HC1`/`robust`, `cluster` (one or two columns).
**Weights:** aweight, fweight, pweight.

| Stata | OpenEconometrics |
| --- | --- |
| `areg y x1 x2, absorb(firm)` | `oe.areg(data=df, y="y", x=["x1", "x2"], absorb="firm")` |
| `areg y x1, absorb(firm) vce(cluster firm)` | `oe.areg(..., absorb="firm", cluster="firm")` |

```python
import numpy as np, pandas as pd, openecon as oe
rng = np.random.default_rng(0)
df = pd.DataFrame({"firm": np.repeat(range(40), 5), "x": rng.normal(size=200)})
df["y"] = 0.1 * df.firm + 1.5 * df.x + rng.normal(size=200)
result = oe.areg(data=df, y="y", x=["x"], absorb="firm")
result.tests["absorbed"]          # F(39, 159) that the firm effects are zero
```

## `oe.reghdfe` — high-dimensional fixed effects (Correia's `reghdfe`)

```python
oe.reghdfe(*, data, y, x, absorb, covariance=None, cluster=None, weights=None, weight_type=None,
           categorical=None, drop_singletons=True, tolerance=1e-8, max_iterations=10000,
           missing="raise", alpha=0.05)
```

**Model.** `y_i = x_i'b + sum_d alpha_d[codes_d(i)] + u_i` for any number of
absorbed dimensions. **No `Intercept` is reported**: the constant is absorbed
by the fixed effects, as in reghdfe.

**Pipeline.**

1. *Singletons.* With `drop_singletons=True` (reghdfe's default) observations
   alone in a level of any dimension are removed iteratively
   (`engines.absorb.singleton_mask`); the count is in `warnings` and in
   `metrics["n_singletons_dropped"]`. They carry no information about the
   slopes and bias clustered standard errors downward. With **frequency
   weights** a row stands for `f_i` observations: a row with `f_i >= 2` is
   never a singleton and keeps its levels alive, exactly as in the
   duplicated-row data set (ftools' `drop_singletons` with fweights).
2. *Demeaning.* `[y X]` is projected off the joint dummy space by symmetric
   Kaczmarz alternating projections accelerated with conjugate gradients
   (`engines.absorb.demean`; the exact one-pass within transform for one
   dimension), weighted when weights are given, to `tolerance` within
   `max_iterations` sweeps (`extra` records iterations, method, final update).
3. *Omission.* Regressors without within variation and collinear regressors
   are omitted and recorded.
4. *Least squares* by QR on the demeaned block.
5. *Exact-fit check.* Besides the rounding floor above, with two or more
   dimensions a fit is `perfect_fit` when `RSS <= (32 tolerance)^2
   TSS_within`: iterative demeaning leaves an error of about ten times its
   tolerance in `y~` and in `X~ b`, so the residuals of an exact fit stop
   there instead of at rounding level (without this check such a fit would be
   returned with standard errors of order `1e-12`). At the default tolerance
   this means a within R-squared above `1 - 1e-13`.

**Degrees of freedom — the reghdfe convention.** `K = k + df_absorbed`,
`df_resid = N - K`, where `df_absorbed` sums over dimensions the observed
levels minus the redundant coefficients: `0` for the first dimension (it also
absorbs the constant), the number of connected components ("mobility groups")
of the bipartite level graph between dimensions one and two for the second
(exact, Abowd-Creecy-Kramarz 2002), and for later dimensions the largest
component count against any earlier dimension (reghdfe's
`dofadjustments(pairwise)`, a lower bound, so inference is conservative).
With a cluster covariance, a dimension whose levels nest inside **any**
cluster column is not counted at all; the check runs against every cluster
column, so the order in which they are listed cannot change a result. The
remaining dimensions are ranked among themselves. When **every** dimension is
nested, one degree of freedom is added back for the constant they absorb.
This makes `absorb(id) cluster(id)` agree with `xtreg, fe vce(cluster id)`
(`K = k + 1`); likewise `absorb(firm year) cluster(firm year)` gives
`K = k + 1`. `extra["absorbed"]` lists `{column, levels, redundant, nested}`
per dimension and `extra["constant_degree_of_freedom_added"]` says whether the
constant was added back.

**Covariances:** `nonrobust` (`RSS/df_resid (X~'WX~)^-1`), `HC1`/`robust`
(`N/(N-K)`), `cluster` (CR1 `G/(G-1)(N-1)/(N-K)`; two columns use
inclusion-exclusion with the smallest `G` and `G_min - 1` degrees of freedom,
reghdfe's convention, with the eigenvalue adjustment described above).
**Weights:** aweight, fweight, pweight.

**Reported.** `metrics`: `r_squared` (overall, with the fixed effects),
`adjusted_r_squared = 1 - (1-R^2)(N-1)/df_resid`, `r_squared_within`,
`adjusted_r_squared_within = 1 - (RSS/df_resid)/(TSS_within/(N - df_absorbed))`,
`rmse`, `df_model`, `df_resid`, `df_absorbed`, `n_singletons_dropped`.
`tests["model"]`: F test of the slopes.

| Stata | OpenEconometrics |
| --- | --- |
| `reghdfe y x1 x2, absorb(firm year)` | `oe.reghdfe(data=df, y="y", x=["x1", "x2"], absorb=["firm", "year"])` |
| `reghdfe y x, absorb(firm year) vce(cluster firm year)` | `oe.reghdfe(..., absorb=["firm", "year"], cluster=["firm", "year"])` |
| `reghdfe y x, absorb(firm) keepsingletons` | `oe.reghdfe(..., absorb=["firm"], drop_singletons=False)` |

```python
df = pd.DataFrame({"firm": np.repeat(range(50), 6), "year": np.tile(range(6), 50)})
df["x"] = rng.normal(size=300)
df["y"] = 0.2 * df.firm + 0.5 * df.year + 1.5 * df.x + rng.normal(size=300)
result = oe.reghdfe(data=df, y="y", x=["x"], absorb=["firm", "year"], cluster="firm")
result.extra["absorbed"]   # firm: nested in the cluster; year: 6 levels, 0 redundant
```

## `oe.cnsreg` — constrained linear regression (Stata `cnsreg`)

```python
oe.cnsreg(*, data, y, x, constraints, covariance=None, cluster=None, weights=None,
          weight_type=None, categorical=None, intercept=True, missing="raise", alpha=0.05)
```

**Model.** `y = x'b + u` subject to `R b = r` (`q` independent constraints on
the `K` design terms). `constraints` is a list of
`{"terms": {term: coefficient, ...}, "value": number}` over the design term
names (`"Intercept"`, a regressor name, `"sector[b]"` for a categorical level):

```python
constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1},     # x1 + x2 = 1
             {"terms": {"x3": 1, "Intercept": -2}, "value": 0}]  # x3 = 2 * Intercept
```

Every number must be finite. An unknown term, a term omitted for
collinearity, a non-numeric, NaN or infinite number, or values so large that
`y - X b_p` leaves the float64 range raise `invalid_constraint`.

**Estimator.** The constraints are eliminated exactly through the complete QR
decomposition of `R'`: `R' = [Q1 Q2][T; 0]`, `b = b_p + Q2 g` with
`b_p = Q1 T'^-1 r` a particular solution and `g` (`K - q` free parameters)
unrestricted. `g` is ordinary (weighted) least squares of `y - X b_p` on
`X Q2`; `V_b = Q2 V_g Q2'` has rank `K - q`. This equals the Lagrangian
solution `b_ols - (X'WX)^-1 R'[R (X'WX)^-1 R']^-1 (R b_ols - r)` without
forming `(X'WX)^-1`. Dependent constraints are screened left to right: a
dependent constraint consistent with the earlier ones is dropped with a
warning, an inconsistent one raises `inconsistent_constraints`.

**Conventions.** `df_resid = N - K + q`; `s^2 = RSS/df_resid`; `HC1` uses
`N/(N - K + q)`; `cluster` uses `G/(G-1)(N-1)/(N-K+q)` (the regress formulas
with `K - q` parameters). `metrics`: `rmse`, `df_model` (rank of the model
test), `df_resid`, `df_constraints`; **no R-squared** (cnsreg reports none).
`tests["model"]`: Wald F that the free, reported non-constant coefficients are
zero. Its degrees of freedom are the rank of their covariance block, which is
the rank of the corresponding rows of `Q2` (a structural property of the
constraints); only when a cluster covariance falls short of that rank is the
test marked `rank_deficient`.

**Coefficients fixed by the constraints.** Coefficient `j` is determined
completely by `R b = r` exactly when `e_j` lies in the row space of `R`,
i.e. when row `j` of the orthonormal `Q2` is zero (norm at most `64 K eps`).
The criterion is structural: it does not depend on the data or on the units
of the regressors, so an estimated coefficient with a tiny variance (a
regressor measured in units of `1e8`) stays in the table, and a coefficient
pinned only by a combination of constraints (`x1 + x2 = 1` and `x1 - x2 = 0`)
is recognised. A fixed coefficient has no standard error: it is **not in the
coefficient table** but in `extra["constrained_terms"]` as `term -> value`,
and a warning names it. `extra["constraint_matrix"]` records `R` and `r` after
reduction.

**Observations.** `cnsreg` needs more observations than design columns
(`n > K`), also when constraints leave fewer free parameters: the
collinearity screen runs on the unconstrained design.

| Stata | OpenEconometrics |
| --- | --- |
| `constraint 1 x1 = x2` + `cnsreg y x1 x2, constraints(1)` | `oe.cnsreg(data=df, y="y", x=["x1", "x2"], constraints=[{"terms": {"x1": 1, "x2": -1}, "value": 0}])` |
| `constraint 2 x1 = 0.5` + `cnsreg ..., constraints(2) vce(robust)` | `oe.cnsreg(..., constraints=[{"terms": {"x1": 1}, "value": 0.5}], covariance="robust")` |

```python
df = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
df["y"] = 0.5 * df.x1 + 0.5 * df.x2 + rng.normal(size=100)
result = oe.cnsreg(data=df, y="y", x=["x1", "x2"],
                   constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1}])
[c.estimate for c in result.coefficients]   # Intercept, x1, x2 with x1 + x2 == 1
```

## Building specs directly

`areg`, `reghdfe` and `cnsreg` only build a `ModelSpec` and call `oe.fit`:

```python
spec = oe.ModelSpec(estimator="reghdfe", outcome="y", predictors=["x"], intercept=False,
                    columns={"absorb": ["firm", "year"]}, covariance="robust",
                    options={"drop_singletons": True})
result = oe.fit(spec, data=df)
```

The registry rejects anything an estimator does not support before data are
read (`pydantic.ValidationError`): `HC2`/`HC3`/`hac` for these estimators, an
intercept for `reghdfe`, `iweight`s, three cluster columns, a cluster column
without the cluster covariance, `cnsreg` without constraints. A spec with
pweights must name `HC1`/`robust` or a cluster column (only the functions
choose `HC1` for you).

## Performance

Measured on a laptop CPU shared with other jobs, 1e6 rows and 10 regressors
(best and worst of repeated runs):

| model | seconds |
| --- | --- |
| `areg` absorb(firm, 1e5 levels), nonrobust / cluster(firm) / two-way | 0.3 – 1.1 |
| `reghdfe` absorb(firm 1e5 × year 1e3), nonrobust / cluster / two-way cluster | 0.7 – 2.9 |
| `reghdfe` same, fweights with the weighted singleton rule, cluster(firm) | 1.1 – 1.7 |
| `reghdfe` three dimensions | 1.3 |
| `cnsreg` nonrobust / robust / cluster(firm) | 0.24 – 0.5 |

Cost is linear in the number of rows: no loop over observations, no n-by-n
object, group work by `index_add_`. `oe.regress` / `oe.newey` have the
performance of `oe.ols`, which switches to its bounded streaming solver for
designs above 64 MiB (about 1.2 s nonrobust or robust at 1e6 rows, but about
27 s with a cluster covariance there; 0.5 s at 5e5 rows on its dense path).

## Limitations and conventions we are not certain of

- `iweight`s are not offered by `areg`/`reghdfe`/`cnsreg`: Stata counts
  `N = sum of weights` (possibly fractional) for them and `ResultBundle.nobs`
  is an integer.
- `reghdfe` nesting. A fixed effect is treated as nested when it nests within
  any cluster variable, which is how we understand reghdfe's degrees-of-freedom
  table ("FE nested within cluster"), and one degree of freedom is added back
  for the constant only when every dimension is nested. Neither the
  partially nested case (`absorb(firm year) cluster(firm)` gives
  `K = k + levels_year`) nor the two-way case
  (`absorb(firm year) cluster(firm year)` gives `K = k + 1`) has been checked
  against reghdfe output. The pairwise rule for a third or later dimension is
  a lower bound on the redundancy, as in reghdfe.
- `reghdfe`'s `adjusted_r_squared_within` uses `N - df_absorbed` as the degrees
  of freedom of `TSS_within`; reghdfe's exact denominator was not verified.
- `cnsreg`: the HC1/cluster small-sample factors use `K - q` parameters and
  the model F test is the rank-reduced Wald test of the free non-constant
  coefficients; Stata's exact choice of model degrees of freedom was not
  verified.
- `areg`'s F test of the absorbed effects is reported for the conventional
  covariance only; Stata's behaviour under `vce(robust)` was not verified.
- Model F test with fewer clusters than slopes: we report the rank-reduced
  test, flagged; we believe Stata prints a missing F.
- The exact-fit thresholds (`1e-28 sum w y^2`; `(32 tolerance)^2 TSS_within`
  for iterative demeaning) are numerical choices of this implementation.
  Stata does not raise an error for an exact fit; it prints degenerate output.
  A fit whose residual variation is below those resolutions is reported as
  `perfect_fit` here.
- `insufficient_observations` compares the number of distinct rows. With
  frequency weights a design with fewer rows than columns is rejected even
  though the duplicated-row data set would be fitted by Stata with columns
  omitted for collinearity.
- `areg`, `reghdfe` and `cnsreg` offer no HC0/HC2/HC3 and no HAC covariance:
  leverage corrections are not meaningful once fixed effects are absorbed.
- `oe.newey` requires `time` and accepts analytic weights only; it cannot
  form autocovariances within panels. These follow `oe.ols` and Stata's
  `newey`; whether Stata's `regress` permits `vce(hc2)`/`vce(hc3)` with
  aweights or pweights is a question for the `oe.ols` conventions.
- Designs are dense float64 and capped at 256 MiB (about 3e6 rows x 10
  columns); absorb high-dimensional categories with `reghdfe` instead of
  expanding them as dummies.
