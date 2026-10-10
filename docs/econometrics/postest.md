# Post-estimation: `lrtest`, `estat_ic`, `bootstrap`, `jackknife`, `suest`, `fcast_eval`, `dm_test`

These procedures work on fitted results from any OpenEconometrics estimator. They do
not fit a model of their own: they read what a `ResultBundle` records (its
coefficients, covariance, log likelihood, estimation rows and specification)
or refit `result.spec` on modified data. Like every estimator, they are
implemented in OpenEconometrics on float64 PyTorch tensors; no estimation library runs
at fit time.

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `lrtest full restricted` | `oe.lrtest(full, restricted)` |
| `lrtest full restricted, df(2) force` | `oe.lrtest(full, restricted, df=2, force=True)` |
| `estat ic` / `estimates stats m1 m2` | `oe.estat_ic(m1, m2, names=["m1", "m2"])` |
| `bootstrap, reps(500) seed(1): probit y x` | `oe.bootstrap(oe.probit(...), df, reps=500, seed=1)` |
| `bootstrap, cluster(id) idcluster(nid): ...` | `oe.bootstrap(fit, df, cluster="id")` (relabelling is automatic) |
| `bootstrap, strata(region) size(100): ...` | `oe.bootstrap(fit, df, strata="region", size=100)` |
| `estat bootstrap, percentile bc` | `ci="percentile"` / `ci="bc"`, read `extra["bootstrap_ci"]` |
| `jackknife: poisson y x` | `oe.jackknife(oe.poisson(...), df)` |
| `jackknife, cluster(id): ...` | `oe.jackknife(fit, df, cluster="id")` |
| `suest m1 m2` / `suest m1 m2, vce(cluster id)` | `oe.suest(m1, m2, data=df)` / `..., cluster="id"` |
| EViews forecast evaluation | `oe.fcast_eval(actual, forecast)` |
| `dmariano y f1 f2, max(4)` (SSC) | `oe.dm_test(actual, f1, f2, horizon=5)` |

Names in OpenEconometrics results: every procedure that returns a model keeps the
original term names; `suest` prefixes them with the model name (`m1:x1`).

**Running the examples.** Every code block on this page runs as written after
this setup (synthetic data; the blocks were executed in this order):

```python
import numpy as np
import pandas as pd
import openecon as oe

rng = np.random.default_rng(1)
n = 500
df = pd.DataFrame({"age": rng.normal(40, 10, n), "grade": rng.integers(8, 18, n),
                   "school": rng.integers(0, 40, n)})
df["union"] = (0.02 * (df.age - 40) + 0.15 * (df.grade - 12)
               + rng.logistic(size=n) > 0).astype(int)
df["visits"] = rng.poisson(np.exp(0.5 + 0.01 * (df.age - 40)) * rng.gamma(2, 0.5, n))
panel_df = pd.DataFrame({"id": np.repeat(np.arange(60), 5), "year": np.tile(np.arange(5), 60),
                         "exp": rng.normal(size=300)})
panel_df["lwage"] = 1 + 0.3 * panel_df.exp + rng.normal(size=60)[panel_df.id] \
    + rng.normal(size=300)
```

## `lrtest`: likelihood-ratio test of nested models

**Model.** Two maximum-likelihood fits of the same outcome on the same
observations; the restricted model is the full model with `df` parameters
constrained. Under the null hypothesis that the restrictions hold,

    LR = 2 (ll_full - ll_restricted)  ~  chi2(df).

**Degrees of freedom.** By default `df` is the difference in the number of
estimated parameters: the reported coefficients, which is Stata's `e(rank)`.
Ancillary parameters (`/lnalpha`, `/sigma`, cutpoints) count; terms omitted
for collinearity do not, because they are not reported. Give `df=` when the
restrictions are imposed without dropping parameters (constraints); a note
records a given `df` that differs from the parameter-count difference.

One case needs `df=`: `ols` and `glm` estimate their error variance without
reporting it as a coefficient, so a full model that does report one
(`mixed` with `/var(Residual)`, `xtreg, mle` with `/sigma_e`) has one more
counted parameter than it truly adds. Stata's `e(rank)` rule overstates df by
one in the same way; OpenEconometrics adds a note and leaves the residual-variance
term out of the boundary list. `oe.lrtest(mixed_fit, ols_fit, force=True,
df=1)` then reproduces the chibar2(01) test of the random intercept that
`mixed` reports as `tests['lr_vs_linear']`.

**What is checked (all raise `AnalysisError('incompatible_models')`).**

- both results report `metrics['log_likelihood']` (least-squares-only, GMM and
  nonparametric fits do not; a combined `suest` result is refused);
- the outcomes are the same: the outcome column and every column that defines
  it (second outcome of `biprobit`, selection indicator of `heckman`, upper
  bound of `intreg`, failure and entry of survival data, binomial trials, the
  dependent variable of every equation of `sureg`/`reg3`, the endogenous
  variables of a VAR). Regressors may differ: two `sureg` systems with the same
  dependent variables and nested regressors can be compared;
- the estimation samples are identical: same `nobs`, same number of rows in
  the data and the same set of estimation rows. Stata only compares `e(N)`;
  OpenEconometrics compares the rows themselves. This check cannot be forced;
- neither fit uses a robust, cluster, HAC or Driscoll-Kraay covariance or
  sampling weights: the objective is then a pseudo-likelihood and LR is not
  chi2. `force=True` (Stata's `force`) computes it anyway and says so;
- both fits come from the same estimator, unless `force=True` (Stata's lrtest
  also requires `force` for, e.g., `poisson` against `nbreg`);
- the full model has more parameters (otherwise swap the arguments or give
  `df=`);
- the restricted model does not have the larger log likelihood beyond a
  relative tolerance of 1e-7 (rounding below that is set to LR = 0);
- REML fits (`oe.mixed(..., method="reml")`): a REML fit cannot be compared
  with an ML fit (never, not even with `force`), and two REML fits must have
  the same fixed effects, because the restricted likelihood depends on the
  fixed-effects design (LR tests of fixed effects are not valid under REML;
  refit with `method="ml"`). `force=True` overrides the second rule only.
  Nested random-effects structures with identical fixed effects are tested
  normally (the variance terms are boundary parameters, see below).

**Boundary parameters.** When the full model has extra variance or
overdispersion parameters (`/lnalpha` after `nbreg`, `/sigma_u` ...), their
null value lies on the boundary of the parameter space. The chi2 reference is
then conservative; for a single such parameter the correct reference is the
50:50 mixture chibar2(01), whose p-value (half the chi2(1) tail) is returned
in `attrs['p_value_chibar2']`, as Stata prints it after `nbreg`.

**Returns** a one-row table with `statistic`, `df`, `p_value`, `ll_full`,
`ll_restricted`; `attrs` hold `distribution`, `k_full`, `k_restricted`,
`nobs`, `df_source`, `forced`, `boundary_terms`, `p_value_chibar2` and the
list `notes` (always starting with the nesting assumption).

```python
full = oe.logit(data=df, y="union", x=["age", "grade"])
small = oe.logit(data=df, y="union", x=["age"])
oe.lrtest(full, small)                    # lrtest full small: LR chi2(1) = 17.23

nb = oe.nbreg(data=df, y="visits", x=["age"])
po = oe.poisson(data=df, y="visits", x=["age"])
oe.lrtest(nb, po, force=True).attrs["p_value_chibar2"]   # test of alpha = 0
```

## `estat_ic`: information criteria

For each result, with `ll` the maximized log likelihood, `k` the number of
estimated parameters (as for `lrtest`) and `N = nobs` (Stata's `e(N)`; the
sum of frequency weights for fweighted fits):

    AIC = -2 ll + 2 k,          BIC = -2 ll + k ln N.

The table has one row per model with `model`, `nobs`, `ll_null` (the
constant-only log likelihood when the result reports it, Stata's `ll(null)`),
`ll`, `df` (= `k`), `aic` and `bic`. `ll_null` comes from the result's stored
null log likelihood, from the McFadden pseudo R-squared of the core
`logit`/`probit` fits (`ll_0 = ll / (1 - R2_p)`) and, for an unweighted or
frequency-weighted `ols` fit with a constant, from `ll_0 = ll + (N/2) ln(1 -
R^2)` (the Gaussian log likelihoods of the model and of the constant-only
model differ by `(N/2) ln(RSS/TSS)`). As in Stata, `k` for `ols`/`glm` counts
the coefficients only (not `sigma` or the scale). Results without a log
likelihood raise `missing_log_likelihood`. `attrs['notes']` warns when the
models use different estimation samples or outcomes, or include REML fits, in
which case their criteria are not comparable. EViews divides both criteria by
`N`; multiply by `N` to compare.

```python
oe.estat_ic(po, nb, names=["poisson", "nbreg"])   # estimates stats: k = 2 and 3
```

## `bootstrap`: resampling covariance for any estimator

`oe.bootstrap(result, data, reps=200, seed=None, cluster=None, strata=None,
size=None, alpha=None, ci="normal")` refits `result.spec` on `reps` samples
drawn with replacement and returns a new result whose covariance is

    V = 1/(R-1) sum_r (b_r - b_bar)(b_r - b_bar)',     b_bar = (1/R) sum_r b_r,

over the `R` successful replicates (Stata's divisor; the variance is taken
around the replicate mean, not the original estimate). Coefficient tests use
`z = b / se` against the standard normal and the table's confidence intervals
are normal based, `b -+ z_(1-alpha/2) se`, which is what Stata reports by
default. `tests['model']` is the Wald chi2 test of all coefficients other than
constants and ancillary (`/...`) parameters, across all equations, under the
bootstrap covariance (Stata's model test of some multi-equation commands
covers the first equation only; use the covariance for other hypotheses).

**The data.** `data` must be exactly the table the model was fitted on. Its
model columns are hashed and compared with `result.provenance['data_hash']`;
a filtered, re-sorted or edited table raises `data_mismatch`. Large OLS fits
that `oe.ols` streamed out of core do not store their estimation rows; the
rows are then recovered (complete cases with a positive weight) and accepted
only if they match the recorded position hash.

**Resampling population and units.**

| setting | unit drawn | population |
| --- | --- | --- |
| default | one observation | the estimation sample (`result.sample_positions`), as Stata drops rows outside `e(sample)` |
| `cluster="c"` | a whole cluster | the estimation sample |
| spec with a `panel` column (`xtreg`, `xtabond`, ...) | a whole panel (default `cluster`) | every data row of the estimation panels, so periods used only as lags travel with their panel |
| `strata="s"` | as above, drawn independently within each stratum | clusters must nest in strata |
| `fweight` fit | `N = sum f` replicated observations, drawn with probability `f_i / N`; the draw counts become the new frequency weights | the bootstrap of the expanded data set |

Every drawn copy of a cluster is relabelled as a distinct cluster (Stata's
`idcluster()`), and so are the panel variable and any `group`/`id` column
(`clogit`, `mixed`, `stcox`) nested in the clusters, so the copies get
separate fixed effects, their periods do not collide and conditional-logit
groups stay separate. When a model has a `group`/`id` role but observations
are resampled, a warning suggests `cluster=`. Analytic and sampling weights
travel with their rows. `size` (Stata's `size()`) is the number of units drawn
per stratum; by default all of them.

When the fit itself clusters (`cluster=` in its specification) but the
bootstrap resamples single observations, a warning says so and suggests
`cluster=`. A cluster variable with a single cluster in the population raises
`insufficient_clusters`.

**Same model in every replicate.** Each replicate refits the original
specification. Data-driven choices of the original fit are pinned so that
every replicate estimates the same parameters: `mlogit` without `base=` takes
the most frequent outcome as its base, and a resample can make another
category the most frequent; the refits use the base of the original fit
(`extra['base']`), as Stata advises `baseoutcome()` with resampling.

**Determinism.** All draws come from one `torch.Generator` seeded with `seed`
(an integer between 0 and 2**64 - 1); the same seed reproduces every replicate
exactly. Without a seed a random one is drawn and recorded in
`inference['seed']`, so any run can be replayed.

**Failed replicates.** A replicate fails when the estimator raises (a
resample with a constant regressor, separation, non-convergence), when a
numerical exception escapes it on a degenerate resample (recorded as
`numerical_failure (<exception type>)`), or when it reports different terms (a
regressor omitted for collinearity, or a category of a categorical regressor
absent, in that resample). The failures are counted by reason, reported in a
warning, in `inference['failed_replicates']` and in `extra['bootstrap']`, and
excluded. More than 10% failures raises `bootstrap_failed`. If an estimate is
the same in every replicate up to rounding (relative spread below 1e-12: the
model fits the data exactly), no resampling covariance exists and
`invalid_covariance` is raised instead of reporting rounding noise as a
standard error.

**Intervals.** `ci` selects the interval stored in `extra['bootstrap_ci']`:

- `normal`: `b -+ z_(1-alpha/2) se` (the same as the coefficient table);
- `percentile`: the `alpha/2` and `1 - alpha/2` percentiles of the replicates,
  with Stata's `_pctile` definition (for `P = R p`: the mean of the `P`-th and
  `(P+1)`-th ordered values when `P` is an integer, otherwise the
  `ceil(P)`-th);
- `bc` (bias-corrected): `z0 = Phi^-1(#{b_r <= b} / R)`, `p1 = Phi(2 z0 - z)`,
  `p2 = Phi(2 z0 + z)` with `z = Phi^-1(1 - alpha/2)`, and the interval is the
  `(p1, p2)` percentiles. It is undefined (`None`, with a warning) when all
  replicates fall on one side of the estimate.

Use at least 1000 replications for percentile and bc intervals.

**Result.** A new `ResultBundle` with the original point estimates and
`inference['covariance'] = 'bootstrap'`, `reps`, `reps_completed`,
`failed_replicates`, `seed`, `resampling_unit`, `strata`, `size` and the
cluster count. `extra['bootstrap']` has the per-term `bias` (replicate mean
minus estimate) and `replicate_mean`; `provenance['postestimation']` records
the method, seed, generator and the original result id. Metrics that do not
depend on the covariance (log likelihood, R-squared) are kept; the original
specification tests and `extra` output are dropped, because they may rest on
the original covariance. `result.spec` is unchanged, so it still describes the
point estimates (its `covariance` field names the original estimator; the
title and `inference` name the bootstrap).

**Refused.** Time-series models (`arima`, `arch`, `var`, `prais`, ... and any
spec with a time column but no panel): resampling single observations
destroys the serial dependence (`bootstrap_unsupported`); fits that already
carry a bootstrap or jackknife covariance; combined `suest` results.

```python
fit = oe.probit(data=df, y="union", x=["age", "grade"])
boot = oe.bootstrap(fit, df, reps=1000, seed=20240501, ci="bc")
print(boot.summary())
boot.extra["bootstrap_ci"]["intervals"]["grade"]

panel = oe.xtreg(data=panel_df, y="lwage", x=["exp"], panel="id", time="year")
oe.bootstrap(panel, panel_df, reps=400, seed=7)   # resamples whole panels, relabelled
```

Performance: the loop calls the estimator's entry function directly and
keeps only the parameter vectors; the cost is `reps` times one fit.

## `jackknife`: delete-one covariance

`oe.jackknife(result, data, cluster=None, max_refits=2000)` refits the model
once per unit with that unit removed:

    V = (N-1)/N sum_i (b_(i) - b_bar)(b_(i) - b_bar)',     b_bar = (1/N) sum_i b_(i),

with Student t inference on `N - 1` degrees of freedom (Stata). Units are
observations, clusters with `cluster=` (then `N = G` and df `= G - 1`), or
whole panels for panel estimators. With frequency weights one replicated
observation is removed at a time (its weight is lowered by one); all copies
of a row give the same replicate, so each row's replicate counts `f_i` times
and `N = sum f`: the result equals the delete-one jackknife of the expanded
data. `extra['jackknife']` holds the Quenouille bias estimate
`(N-1)(b_bar - b)` and the replicate mean; `tests['model']` is a Wald F test
with `N - 1` denominator degrees of freedom.

More than `max_refits` units raise `jackknife_too_large` (use `cluster=`,
`oe.bootstrap`, or raise the bound). Failed refits are excluded and counted
(more than 10% raise `jackknife_failed`); `N` and the degrees of freedom then
count the successful replicates. The refits pin the `mlogit` base as for the
bootstrap; a single cluster raises `insufficient_clusters`, estimates that do
not vary beyond rounding raise `invalid_covariance`, and deleting single
observations of a clustered fit adds a warning suggesting `cluster=`.

```python
jk = oe.jackknife(oe.poisson(data=df, y="visits", x=["age", "grade"]), df)   # 500 refits
jk_c = oe.jackknife(fit, df, cluster="school")     # 40 refits, t(39) inference
```

## Explicit dependence schemes and advanced intervals

`oe.bootstrap` additionally accepts `scheme`, `block_length`, `time`, `wild`,
`null` and `max_refits`. The advanced path uses unweighted, unstratified,
full-size samples and a local generator. It records the actual seed, unit,
restriction, discarded-draw reasons and refit budget. Above 10% failed draws
the operation fails; studentization discards draws with nonpositive/nonfinite
standard errors. BCa requires every delete-unit fit to succeed.

| Scheme | Validated domain | Draw and restriction |
| --- | --- | --- |
| `iid` with `ci='bca'` or `'studentized'` | registry fits already eligible for iid pairs | rows or whole relabelled clusters; BCa deletes one such unit |
| `moving_block` | explicit-column OLS, one regularly spaced series | overlapping non-circular pairs blocks; concatenate and crop the final block to N |
| `residual` | explicit-column OLS without declared time dependence | fixed X, fitted means plus iid draws of centered residuals |
| `residual_block` | explicit-column OLS, one regularly spaced series | fixed X plus overlapping blocks of centered residuals |
| `wild_cluster` | explicit-column OLS, independent clusters | one shared Rademacher or Webb multiplier per cluster |

Blocks require `time` (or the fit's time column) and `2 <= block_length < N`;
the retained time observations must be distinct and evenly spaced. A moving
block refit with HAC covariance receives a fresh regular synthetic timeline.
Formula/lag-generated predictors and panels are outside the dependent domain.
The general iid time-series guard remains in force.

`ci='studentized'` forms `(b_star-b)/se_star` from each refit's own covariance,
then inverts its tail quantiles using the original SE. `ci='bca'` uses the
delete-unit acceleration `sum u^3 / (6 (sum u^2)^1.5)`, where `u=mean(jack)-jack`.
BCa currently applies only to iid pairs. Advanced empirical quantiles retain
the Stata `_pctile` convention; they do not reproduce R boot's interpolation.
Normal coefficient-table intervals use the replicate covariance; requested
BCa/studentized/percentile intervals are stored separately in `bootstrap_ci`.

For `wild_cluster`, `null={'x': 0}` generates residuals from restricted least
squares and returns two-sided coefficient bootstrap-t p-values. This requires
the original cluster covariance on the same column and `ci='normal'`;
the reported confidence intervals are **not** null-test inversions. Without
`null`, draws use the unrestricted fitted model. `max_refits` (default 10,000)
counts draws plus BCa acceleration refits, and stored draws/multiplier matrices
are bounded by 2,000,000 entries. No draws are silently omitted to fit a budget.

```python
fit = oe.ols(data=df, y="y", x=["x"], covariance="cluster", cluster="firm")
wild = oe.bootstrap(fit, df, scheme="wild_cluster", cluster="firm",
                    null={"x": 0}, wild="webb", reps=399, seed=19)
wild.extra["bootstrap_test"]
```

References: [R boot intervals](https://stat.ethz.ch/R-manual/R-devel/library/boot/html/boot.ci.html),
[R time-series bootstrap](https://stat.ethz.ch/R-manual/R-devel/library/boot/html/tsboot.html),
[Stata wild bootstrap](https://www.stata.com/manuals/rwildbootstrap.pdf).
Independent tests cover OLS draw replay, restricted cluster-t statistics,
BCa acceleration, replicate SEs, serial-block coverage agreement and failure
policies. Coverage diagnostics compare fixed synthetic designs with a reference;
they do not guarantee nominal coverage on every data-generating process.

## `suest`: seemingly unrelated estimation

`oe.suest(*results, data, names=None, cluster=None)` combines separately
fitted models into one parameter vector with a joint robust covariance, so
hypotheses across models can be tested (equal coefficients in two equations,
a coefficient in a logit against the same coefficient in a probit scaled by
1.6, ...). With `s_mi` the score contribution of observation `i` to model `m`
and `H_m` the Hessian of model `m`,

    V = D (q sum_i s_i s_i') D,     D = blockdiag((-H_1)^-1, ..., (-H_M)^-1),

where `s_i` stacks observation `i`'s scores for every model (zero for a model
whose estimation sample does not contain `i`), the sum runs over the `N`
observations in the union of the samples and `q = N/(N-1)` (the `_robust`
factor Stata's suest uses). With `cluster="c"` the scores are first summed
within clusters and `q = G/(G-1)`. Each diagonal block is that model's
Huber-White sandwich; the off-diagonal blocks are the cross-model
covariances. Inference uses z statistics.

**Scores.** Results do not store score contributions, so suest rebuilds each
model's design from its specification and the data (same treatment coding,
same estimation rows, the reported terms) and evaluates the analytic score
and Hessian at the reported estimates. The score sum must vanish there: a
Newton decrement `g'(-H)^-1 g` above 1e-6 means the model could not be
reproduced and raises `suest_mismatch`. Supported providers:

| estimator | parameters and scores |
| --- | --- |
| `ols` | Stata's suest treats `regress` as the normal likelihood in `(b, ln sigma^2)` with `sigma^2 = SSR/N`: `s_b = x u / sigma^2`, `s_lnvar = (u^2/sigma^2 - 1)/2`; a second equation `<name>_lnvar` holds `ln(SSR/N)`. The slope block equals HC0 times N/(N-1). |
| `logit` | `(y - L(x'b)) x`, `H = -sum L(1-L) x x'` |
| `probit` | `q r x` with `q = 2y-1`, `r = phi(q x'b)/Phi(q x'b)`; `H = -sum r (r + q x'b) x x'` |
| `poisson` | `(y - mu) x`, `H = -sum mu x x'`, offset/exposure included |
| `ologit`, `oprobit` | the ordered-model kernel's analytic scores in `(b, cutpoints)` |
| `mlogit` | `(1{y=j} - p_j) x` per non-base equation; the family kernel's Hessian |
| `glm` | analytic family/link quasi-score and observed Hessian, including trials and offsets; unit dispersion cancels from the sandwich |
| `nbreg` | analytic count and dispersion scores for the fitted mean/constant parameterization |
| `tobit`, `intreg`, `truncreg` | analytic density, censoring, interval or truncation probabilities; Hessian transformed to the reported sigma or log-sigma parameter |
| `cloglog` | Bernoulli likelihood with `p = 1-exp(-exp(eta))`, `eta=x'b+offset`; score `x [y A-(1-y) B]`, `A=exp(eta)/expm1(exp(eta))`, `B=exp(eta)`; observed Hessian |
| `fracreg` logit/probit | Bernoulli quasi-likelihood `y log G(eta)+(1-y) log(1-G(eta))`, including 0, 1 and interior responses; logit score `x(y-p)`, probit score `x [y phi(eta)/Phi(eta)-(1-y) phi(eta)/Phi(-eta)]`; observed Hessian |

Frequency, analytic and probability weights are supported when all input
models use the same weight type and values on their union sample. Frequency
weights reproduce literal row duplication: the independent-row meat is
`sum f_i s_i s_i'`, while cluster scores sum `f_i s_i`. Effective N is `sum f`.
Analytic weights use each estimator's normalized power-likelihood convention;
probability weights use a scale-invariant sandwich. Both use physical N for
the independent-row correction. This is not survey-design variance estimation.
Other estimators and importance weights raise `suest_unsupported`.
The input fits may carry any covariance (including a bundle returned by
`oe.bootstrap`/`oe.jackknife`, whose estimates and specification are the
original fit's); suest recomputes everything from the scores and records each
input's covariance in `extra['models']`.

For `cloglog` and `fracreg`, reconstruction verifies the complete saved
coefficient design, treatment coding, omitted columns, link, physical sample
positions, effective/original observation counts, dropped rows, sample hash
and recorded convergence. A stationary subset is refused even when it reproduces
the saved estimates. JSON restoration needs no refit. `cloglog`
retains its offset and native frequency/probability-weight domain; `fracreg`
retains its native frequency/probability/normalized analytic-weight convention
and has no offset role in this API. Importance weights remain refused. Fractional
probit uses the fractional Bernoulli QML derivatives; the binary sign shortcut
`q=2y-1` is not valid for interior responses. These formulas follow the
[cloglog](https://www.stata.com/manuals/rcloglog.pdf) and
[fracreg](https://www.stata.com/manuals/rfracreg.pdf) likelihood definitions;
native analytic weights are an OpenEcon convention, not a claim of Stata parity.

These saved score calculations run on resident CPU float64 and preserve the
caller's default device and NumPy/Torch random states. Dense work is admitted up
to `N*P^2+P^3 = 100,000,000` for each new provider and
`sum(N_m)*P_joint^2+P_joint^3 = 500,000,000` for a joint system containing it.
Individual work admission checks the reported parameter width, the complete
encoded width before design allocation/rank screening, and the kept width after
screening. Omitted columns and unused category levels count in that pre-screen
width. Early named-buffer plans also obey `OPENECON_WORKSPACE_MB` (default 512 MiB),
before score reconstruction and dense joint allocation. The joint union estimate
uses the conservative sum of model sample sizes; guards are implementation
resource bounds, not statistical sample-size limits or process RSS limits.
The joint plan is saved in provenance. A reported-coordinate information matrix
singular at float64 precision raises `singular_information` with a request to
center/rescale and refit; stationarity tolerances are not relaxed. Dataset replay,
survey designs and other score providers remain outside this extension.

**Result.** Terms are `<name>:<term>` (`m1:x1`, `m1:/lnvar`, `m2:/cut1`)
grouped in equations `<name>` (or `<name>_<equation>`; OLS fits use
`<name>_mean` and `<name>_lnvar`). Names default to the estimator names,
numbered when repeated. `nobs` is the effective union size (sum of frequency
weights, otherwise physical rows); both counts are recorded in inference;
`spec` is the first model's specification (the combined result cannot be
refitted, bootstrapped or LR-tested); `extra['models']` lists each model's
estimator, outcome, sample size and parameter count. The data must be the
table all models were fitted on (`data_mismatch` otherwise).

```python
m1 = oe.logit(data=df, y="union", x=["age", "grade"])
m2 = oe.probit(data=df, y="union", x=["age", "grade"])
joint = oe.suest(m1, m2, data=df, names=["lgt", "prb"], cluster="school")

V = np.array(joint.covariance_matrix)      # test b_lgt(grade) = 1.6 b_prb(grade)
b = np.array([c.estimate for c in joint.coefficients])
r = np.zeros(len(b))
r[2], r[5] = 1.0, -1.6
wald = (r @ b) ** 2 / (r @ V @ r)          # chi2(1)
```

Survey settings remain unsupported. See the [Stata suest manual](https://www.stata.com/manuals/rsuest.pdf)
for the same-weight and different-sample contracts.

## `fcast_eval`: forecast evaluation statistics

`oe.fcast_eval(actual, forecast, data=None, naive=None, seasonal_period=None,
missing="raise")` reproduces EViews' forecast evaluation table. `actual` and
`forecast` are column names (with `data`), Series, tensors, arrays or
sequences, aligned over the `h` evaluation periods; values must be
one-dimensional (an `(h, 1)` column is accepted, an `(h, 2)` block or an
unordered set raises `invalid_data`). The error is `e_t = y_t - f_t` (actual
minus forecast).

| statistic | definition |
| --- | --- |
| `mean_error` | `(1/h) sum e_t` (positive: the forecast is too low) |
| `mae`, `mse`, `rmse` | `(1/h) sum abs(e_t)`, `(1/h) sum e_t^2`, `sqrt(mse)` |
| `mape` | `100 (1/h) sum abs(e_t / y_t)`; empty with a note when an actual value is zero |
| `smape` | `100 (1/h) sum abs(f_t - y_t) / ((abs(f_t) + abs(y_t))/2)` (EViews' symmetric MAPE) |
| `theil_u1` | `rmse / (sqrt((1/h) sum f_t^2) + sqrt((1/h) sum y_t^2))`, in [0, 1] |
| `theil_u2` | EViews: `sqrt(sum ((f_(t+1) - y_(t+1))/y_t)^2 / sum ((y_(t+1) - y_t)/y_t)^2)`, t = 1..h-1; below 1 beats the no-change forecast |
| `bias_proportion` | `(mean f - mean y)^2 / mse` |
| `variance_proportion` | `(s_f - s_y)^2 / mse` |
| `covariance_proportion` | `2 (1 - r) s_f s_y / mse` (the three sum to one; `s` uses divisor `h` as in EViews) |
| `mase` | `mae / scale` (Hyndman and Koehler 2006) |

The MASE scale is the MAE of the benchmark forecast `naive` when given (a
relative MAE), otherwise the mean absolute `m`-period change of the actual
series, `(1/(h-m)) sum_(t>m) abs(y_t - y_(t-m))`, with `m = seasonal_period`
(default 1, the random walk). Hyndman and Koehler compute that scale on the
training sample; pass the in-sample naive forecast errors' benchmark through
`naive` if you want that version. The result is a table indexed by statistic
with one column `value`; undefined entries are empty and explained in
`attrs['notes']`.

```python
y = 10 + np.cumsum(rng.normal(size=60))
holdout = pd.DataFrame({"y": y, "f_rw": np.r_[y[0], y[:-1]],
                        "f_ar": y + rng.normal(scale=0.7, size=60)})
oe.fcast_eval("y", "f_ar", data=holdout)
oe.fcast_eval(holdout["y"], holdout["f_ar"], naive=holdout["f_rw"])   # MASE vs random walk
```

## `dm_test`: Diebold-Mariano test of equal accuracy

`oe.dm_test(actual, forecast1, forecast2, data=None, horizon=1,
loss="squared", harvey=True, kernel="bartlett", missing="raise")` tests
H0: `E[d_t] = 0` for the loss differential `d_t = L(e_1t) - L(e_2t)`, with
`L(e) = e^2` or `abs(e)`:

    DM = d_bar / sqrt(LRV / T),     LRV = gamma_0 + 2 sum_(k=1..h-1) w_k gamma_k,

`gamma_k = (1/T) sum_(t>k) (d_t - d_bar)(d_(t-k) - d_bar)`. An `h`-step
forecast error is MA(h-1), so `h - 1` autocovariances enter. `kernel`
selects the weights: `bartlett` (default; Newey-West `w_k = 1 - k/h`, always a
nonnegative variance) or `uniform` (`w_k = 1`, the truncated estimator of
Diebold and Mariano 1995 and of R's `forecast::dm.test`; a negative variance
raises `nonpositive_variance`). With `harvey=True` the Harvey, Leybourne and
Newbold (1997) statistic

    DM* = DM sqrt((T + 1 - 2h + h(h-1)/T) / T)

is compared with Student t(T-1); otherwise DM is compared with N(0, 1). The
p-value is two-sided; a positive statistic means forecast 1 has the larger
loss. A loss differential that is constant up to rounding raises
`constant_loss_differential`. The `h - 1` autocovariances are summed directly
for up to 64 lags and computed by FFT beyond, so long horizons stay O(T log T).
The HLN factor `(T + 1 - 2h + h(h-1)/T)/T = ((T-h)^2 + (T-h))/T^2` is positive
for every `h < T`.

```python
oe.dm_test("y", "f_ar", "f_rw", data=holdout, horizon=4)          # HLN, Bartlett
oe.dm_test(holdout.y, holdout.f_ar, holdout.f_rw, kernel="uniform", harvey=False)
```

## Performance

Timings on this development machine (Apple silicon, CPU, float64), synthetic
data with 1,000,000 rows and 10 regressors, all data in memory, measured in
the verification pass:

| call | time |
| --- | --- |
| `poisson` fit / `ols` fit (reference) | 1.2 s / 0.7 s |
| `bootstrap(poisson, reps=5)` / with `cluster=` (200 clusters) | 2.6 s / 2.7 s |
| `bootstrap(ols, reps=5)` | 3.5 s |
| `jackknife(poisson, cluster=)` with 10 clusters (10 refits) | 4.3 s |
| `jackknife(poisson)` on 2,000 rows (2,000 delete-one refits) | 8.7 s |
| `suest(ols, poisson)` / `suest(ols, poisson, probit, cluster=)` | 1.0 s / 1.2 s |
| `fcast_eval` / `dm_test(horizon=12)` / `dm_test(horizon=500)` on 1e6 periods | 0.01 / 0.01 / 0.04 s |
| `lrtest`, `estat_ic` (read stored results) | well under 0.1 s (plus the fits) |

Resampling costs `reps` (or the number of units) times one fit plus the row
selection of each replicate; only the parameter vectors are kept, and the
draws are O(n) per replicate. Bootstrapping a core `logit`/`probit` fit is
dominated by that estimator's own separation check (about 6 s per fit at
this size). suest builds one `N x P` score matrix (P = total parameters) and
two `P x P` products; no `N x N` object is formed anywhere.

## Limitations

- The legacy iid bootstrap and jackknife refuse time-series models. Explicit
  moving/residual blocks support the OLS domain described below. BCa and
  studentized intervals are available through the bounded advanced path.
  Stata's `mse` option remains unsupported.
- `bootstrap` resamples the estimation sample; Stata's `nodrop` (resampling
  rows outside `e(sample)`) is available only implicitly for panel models,
  whose whole panels are resampled.
- `jackknife` has no `strata`/`mse` options and is bounded by `max_refits`.
- `suest` supports the analytic score providers listed above with f/a/p weights;
  survey settings are not available. It does not
  provide its own Wald-test function: the general `oe.test`/`oe.lincom`
  functions now use the stored joint reporting covariance, including
  cross-equation restrictions; see [coefficient inference](inference.md).
- The returned bootstrap/jackknife bundles keep `result.spec` unchanged
  (`ModelSpec` cannot record `covariance='bootstrap'` for estimators that do
  not declare it), so `summary()` prints the original covariance name in its
  header line; the title and `inference['covariance']` name the resampling
  covariance. A bootstrap/jackknife bundle cannot itself be resampled again.

## Verification

`tests/test_econ_postest_{lr,resampling,suest,fcast}.py` (implementation
tests) and the independent verification files:

- `tests/test_econ_postest_oracle.py`: bootstrap replicates regenerated from
  the documented draw contract with NumPy index arithmetic and re-estimated by
  statsmodels (probit, poisson) or NumPy (OLS, the within estimator on
  relabelled panels), for observation, cluster and stratified draws with
  `size`, percentile and bias-corrected intervals with an independently coded
  `_pctile` rule; frequency weights equal duplicated rows (bootstrap with and
  without strata, jackknife); rescaling a regressor rescales its bootstrap SE;
  `ols` and `glm(gaussian)` share replicates; delete-one and delete-cluster
  jackknife loops over statsmodels poisson/logit and a brute-force SciPy tobit
  likelihood, including t(N-1) p-values, intervals, the F model test and the
  Quenouille bias; `lrtest`/`estat_ic` against statsmodels (poisson with a
  categorical regressor, OLS, mlogit, logit null likelihood, fweights) and
  brute-force ordered-logit and NB2 likelihoods; suest against stacked
  statsmodels scores (logit + poisson + OLS on overlapping samples, robust and
  cluster) and own analytic ordered-model scores (ologit, oprobit) with
  statsmodels mlogit; fcast_eval against the EViews definitions; dm_test
  against the DM/HLN formulas for both losses and kernels and horizons 1, 3
  and 90 (the FFT path), with the sign flip when the forecasts are swapped.
- `tests/test_econ_postest_adversarial.py`: tiny and degenerate samples,
  option bounds, wrong types and columns, edited or re-ordered data, single
  clusters, exact fits, multidimensional inputs, missing and constant series,
  extreme magnitudes (regressors scaled by 1e8 and 1e-8), row permutations,
  collinear terms, refused combinations, and one regression test per defect
  fixed in the verification pass.

## Conventions that were not verified against Stata output

`provenance['stata_parity_validated']` is `False` for every result here. The
following conventions are the textbook or documented ones but have not been
compared with Stata output:

- the bootstrap percentile rule (Stata's `_pctile` definition is assumed for
  `estat bootstrap, percentile` and `bc`);
- failed replicates: OpenEconometrics drops a failed replicate entirely (all
  coefficients), and a replicate that omits a collinear term or lacks a
  category counts as failed; Stata marks only the affected statistics as
  missing. The 10% failure bound is a project rule;
- a frequency-weighted fit is bootstrapped as the expanded data (N = sum f
  draws with probability f_i / N) and jackknifed by deleting one replicated
  observation at a time; Stata's handling of fweights inside the prefixes may
  differ. With failed refits the jackknife uses the number of successful
  replicates in `(N-1)/N` and in the degrees of freedom;
- `suest` uses `q = N/(N-1)` over the union of the samples (or `G/(G-1)`), as
  `_robust` does by default; for `regress` it uses the ML variance `SSR/N` in
  the `lnvar` equation;
- `lrtest` treats `bootstrap`/`jackknife` results as likelihood fits (their
  log likelihood is the original fit's), compares the estimation rows
  (stricter than Stata's `e(N)` check) and lets `force` override neither
  different samples nor different outcomes; REML fits with different fixed
  effects are refused unless forced, REML against ML always;
- `estat_ic`'s `ll_null` after `ols` is the constant-only Gaussian log
  likelihood `ll + (N/2) ln(1 - R^2)` (Stata's `e(ll_0)`), given only for
  unweighted or frequency-weighted fits with a constant;
- refits of an `mlogit` fit without `base=` use the base of the original fit;
- `dm_test` defaults to the Bartlett (Newey-West) long-run variance with `h-1`
  lags; the HLN correction factor was derived for the truncated (uniform)
  estimator, which `kernel="uniform"` selects;
- MASE without a benchmark scales by the naive changes of the evaluated actual
  series rather than of a training sample.
