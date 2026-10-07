# Treatment effects and policy evaluation: `teffects`, `didregress`, `eventstudy`, `rdrobust`

This page covers the estimators of causal effects of a treatment or policy:

* `oe.teffects` — Stata's `teffects`: regression adjustment (`ra`), inverse-
  probability weighting (`ipw`), the doubly robust IPW regression adjustment
  (`ipwra`) and augmented IPW (`aipw`), nearest-neighbour covariate matching
  (`nnmatch`) and propensity-score matching (`psmatch`);
* `oe.didregress` and `oe.eventstudy` — two-way fixed-effects difference in
  differences (Stata's `didregress` / `xtdidregress`) and dynamic event studies;
* `oe.csdid` — heterogeneity-robust staggered difference in differences:
  Callaway-Sant'Anna group-time ATTs with their aggregations;
* `oe.rdrobust` and `oe.rdplot` — sharp and fuzzy regression discontinuity with
  robust bias-corrected inference (the community `rdrobust` package of
  Calonico, Cattaneo, Farrell and Titiunik).

Everything is implemented in OpenEconometrics on float64 PyTorch tensors: the nuisance
models are fitted by Newton-Raphson with analytic derivatives or by QR least
squares, the stacked estimating equations are differentiated analytically,
fixed effects are absorbed with the shared `engines.absorb` kernels, and
neighbour searches never build an n-by-n matrix. No estimation library runs
at fit time. `provenance["stata_parity_validated"]` is `False` for every
result: the conventions below follow the Stata (and `rdrobust`)
documentation and are verified against independent implementations, not
against Stata output.

## 1. `teffects`: treatment effects under unconfoundedness

### The model

Each observation has potential outcomes `y(l)` for the treatment levels
`l = 0 (control), 1, ..., L-1`; only `y(D)` is observed, `D` being the level
received. The estimands are

* the potential-outcome means `POM_l = E[y(l)]`,
* the average treatment effects `ATE_l = POM_l - POM_0`,
* effects in the chosen treated population:
  `ATET_l = E[y(l) - y(control) | D = tlevel]`, with the first non-control
  level the default target. Matching remains binary.

They are identified when treatment is unconfounded given covariates `x`
(`y(l)` independent of `D` given `x`) and when every observation has a positive
probability of every level (overlap).

```python
import openecon as oe

ate = oe.teffects(data=df, y="bweight", treatment="mbsmoke",
                  x=["mage", "prenatal1", "mmarried", "fbaby"], method="aipw")
print(ate.summary())
att = oe.teffects(data=df, y="bweight", treatment="mbsmoke",
                  x=["mage", "prenatal1"], method="ipwra", estimand="atet")
```

| Stata | OpenEconometrics |
| --- | --- |
| `teffects ra (y x1 x2) (t)` | `oe.teffects(data=df, y="y", treatment="t", x=["x1","x2"])` |
| `teffects ra (y x1 x2, logit) (t), atet` | `..., omodel="logit", estimand="atet"` |
| `teffects ipw (y) (t x1 x2, probit)` | `..., method="ipw", tx=["x1","x2"], tmodel="probit"` (`x` omitted) |
| `teffects ipwra (y x1 x2, poisson) (t z1 z2)` | `..., method="ipwra", omodel="poisson", tx=["z1","z2"]` |
| `teffects aipw (y x1 x2) (t x1 x2), pomeans` | `..., method="aipw", estimand="pomeans"` |
| `teffects ... , control(2)` | `control=2` |
| `teffects ... , pstolerance(1e-4)` | `pstolerance=1e-4` |
| `teffects ... , vce(cluster id)` | `covariance="cluster", cluster="id"` |
| `teffects ... [pweight=w]` / `[fweight=f]` | `weights="w", weight_type="pweight"` / `"fweight"` |
| `teffects nnmatch (y x1 x2) (t), nneighbor(2) biasadj(x1)` | `method="nnmatch", neighbors=2, biasadj=["x1"]` |
| `teffects psmatch (y) (t x1 x2), caliper(0.05)` | `method="psmatch", tx=["x1","x2"], caliper=0.05` |
| `teoverlap` | `result.extra["overlap"]` (propensity-score summaries by group) |

When `tx` is not given, the treatment model uses the outcome covariates `x`
(recorded in `extra["treatment_covariates"]`). The outcome covariates of `ra`,
`ipwra` and `aipw` are `x`; `ipw` and `psmatch` read only the treatment model.

### Estimators

Notation: `p_l(x)` is the treatment-model probability of level `l`,
`mu_l(x) = m(x'b_l)` the outcome model of level `l`, fitted on the
observations with `D = l`, and `w_i` the user weight.

| `method` | potential-outcome mean `POM_l` | outcome model fitted by |
| --- | --- | --- |
| `ra` | mean of `mu_l(x_i)` over all observations (`atet`: over the treated) | (quasi-)ML, unweighted |
| `ipw` | `sum 1{D=l} omega_l y / sum 1{D=l} omega_l` (normalized weights) | — |
| `ipwra` | as `ra` | (quasi-)ML weighted by `omega_l` |
| `aipw` | mean of `1{D=l} (y - mu_l(x)) / p_l(x) + mu_l(x)` | (quasi-)ML, unweighted |

The inverse-probability weights are `omega_l = 1 / p_l(x)` for `ate` and
`pomeans`; for `atet` they are `p_target(x) / p_l(x)` for each level,
including weight 1 in the target level. `ipwra` and `aipw` are doubly robust:
they are consistent when either the treatment model or the outcome model is
correctly specified. `aipw` is the efficient-influence-function estimator of
Robins, Rotnitzky and Zhao (1994); `ipwra` is Wooldridge's (2007) weighted
regression adjustment.

Outcome models (`omodel`): `linear` (least squares by QR), `logit` and
`probit` (the outcome must lie in [0, 1]; a fractional outcome gives the
Bernoulli quasi-likelihood, i.e. Stata's `flogit`/`fprobit` outcome models)
and `poisson` (nonnegative outcome; exponential mean). Treatment models
(`tmodel`): `logit` (default) or `probit` for a binary treatment; with more
than two levels the treatment model is the multinomial logit (as in Stata,
`tmodel="probit"` is then refused). Every likelihood is maximized by
Newton-Raphson with analytic derivatives on covariates centred at their
(weighted) means; reported auxiliary coefficients are mapped back exactly.

### Standard errors: stacked estimating equations

Each estimator solves `sum_i w_i psi_i(theta) = 0` for
`theta = (gamma, b_0, ..., b_{L-1}, POM_0, ..., POM_{L-1})`:

    treatment model   x_t u_s(gamma)                    (logit / probit / mlogit score)
    outcome model l   x_o 1{D=l} omega_l r_l(b_l)       (omega_l = 1 for ra and aipw)
    POM_l, ra/ipwra   mu_l(x) - POM_l                   (atet: 1{D=1} (mu_l(x) - POM_l))
    POM_l, ipw        1{D=l} omega_l (y - POM_l)
    POM_l, aipw       1{D=l} (y - mu_l(x)) / p_l(x) + mu_l(x) - POM_l

with `r_l` the ML residual of the outcome model (`y - mu` for linear, logit
and Poisson; the probit generalized residual) and `u_s` that of the treatment
model. The covariance of `theta` is the M-estimation sandwich

    V = A^-1 B A^-T,   A = sum_i w_i d psi_i / d theta',   B = sum_i (w_i psi_i)(w_i psi_i)'

(`B = sum_i f_i psi_i psi_i'` with frequency weights; with `covariance="cluster"`
`B = sum_g t_g t_g'` with `t_g` the cluster totals of the weighted scores). It
accounts for the estimation of the treatment and outcome models. `A` is
assembled analytically: every block is a weighted cross product of covariate
rows with per-observation derivative scalars. The meat is accumulated over
blocks of rows, so the n-by-p score matrix is never stored. The reported
effects `ATE_l = POM_l - POM_0` and `POmean = POM_0` are a linear
transformation `T theta` with covariance `T V T'`. No small-sample factor is
applied (Stata's `teffects` estimators are GMM estimators and Stata's `gmm`
applies none); coefficient tests are z tests.

`covariance` is `"robust"` (default, Stata's `vce(robust)`) or `"cluster"`
(`vce(cluster clustvar)`, one cluster column).

### Overlap

As Stata's `pstolerance(1e-5)` rule, an estimated probability of any level
below `pstolerance` for any observation raises `overlap_violation` with the
number of offending observations. A treatment model that predicts treatment
perfectly (separation) raises the same error. `extra["overlap"]` summarizes
the estimated propensity scores (`P(t=1)` for a binary treatment, every
`P(t=l)` otherwise) by observed treatment level: n, min, quartiles, mean, max
— the numbers behind Stata's `teoverlap` density plots. `metrics` hold
`propensity_min` and `propensity_max` for a binary treatment.

### Result

* terms (Stata's `e(b)` names): `ATE:r1vs0.t` (one per non-control level) and
  `POmean:0.t`; `ATET:r1vs0.t` and `POmean:0.t` for `atet` (the POmean is then
  the mean of the untreated potential outcome among the treated); `POmeans:l.t`
  for every level with `pomeans`. The equation of each term is the part before
  the colon.
* `metrics`: `n_levels`, `n_control`, `n_treated` (binary), the treatment
  model's `treatment_log_likelihood`, `propensity_min`, `propensity_max`.
* `extra`: `auxiliary_equations` — `TME1` (treatment model; `TME1`, `TME2`, ...
  for a multinomial model) and `OME0`, `OME1`, ... (outcome models), each a
  list of `{term, estimate, std_error}` from the stacked covariance;
  `potential_outcome_means`; `observations_by_level`; `overlap`; the method,
  estimand, models and levels.

Collinear covariates are omitted equation by equation, Stata style, and
recorded in `warnings` and `provenance["omitted_terms"]` with the equation
prefix (`TME:x3`, `OME1:z`): a covariate that is constant among the treated
leaves the treated outcome equation only.

### Weights and missing data

`fweight` (results equal those of the duplicated-row data set, `N = sum f`) and
`pweight` (sampling weights; rescaling them changes nothing) multiply every
estimating equation. `missing="raise"` (default) refuses incomplete rows;
`missing="drop"` excludes them and records how many.

### Matching: `nnmatch` and `psmatch`

Every observation is matched with replacement to the `neighbors` (`M`,
default 1) nearest observations of the other treatment group. Ties with the
M-th distance are all matched, as in Stata, so the matched set `J(i)` has
`#J(i) >= M` members (`extra["matches"]` reports the smallest and largest set
and how many units had ties). The imputed potential outcomes are
`Yhat_i(D_i) = Y_i` and `Yhat_i(1 - D_i) = mean of Y over J(i)`;

    ATE  = (1/N) sum_i (Yhat_i(1) - Yhat_i(0))
    ATET = (1/N_1) sum_{D_i=1} (Y_i - Yhat_i(0))

* `nnmatch` matches on the covariates `x` with the distance
  `((x_i - x_j)' S^-1 (x_i - x_j))^(1/2)`: `S` the sample covariance of the
  covariates (`metric="mahalanobis"`, default), its diagonal (`"ivariance"`)
  or the identity (`"euclidean"`). Collinear covariates are dropped first.
* `psmatch` matches on the propensity score `p(x)` estimated by a logit
  (default) or probit treatment model on `tx` (default `x`), distance
  `|p_i - p_j|`; the overlap check of `pstolerance` applies.
* `biasadj=[...]` applies the Abadie-Imbens (2011) bias correction:
  `mu_w(x) = x_B' b_w` is fitted by least squares on the observations of group
  `w`, weighted by the number of times each is used as a match (the matched
  sample, `K_j` below), and the imputation becomes
  `mean_{j in J(i)} (Y_j + mu(x_i) - mu(x_j))`. The coefficients are in
  `extra["bias_adjustment"]`. Stata offers `biasadj()` for `nnmatch` only;
  with `psmatch` OpenEconometrics accepts it but warns that the Abadie-Imbens (2016)
  correction below is derived for the unadjusted estimator.
* `caliper`: if any observation has fewer than `M` matches within the caliper
  distance, `caliper_violation` is raised (Stata exits with an error rather
  than dropping such observations). Matching distances are those of the
  metric (for psmatch: differences of propensity scores).

**Variance** (Abadie and Imbens 2006; Stata's `vce(robust, nn(#))`). With
`K_j = sum_i 1{j in J(i)} / #J(i)` (how often unit `j` is used, each use
weighted by the size of the matched set), `K'_j = sum_i 1{j in J(i)} / #J(i)^2`
and the conditional variances
`sigma2_i = #H/(#H+1) (Y_i - mean_{H(i)} Y)^2`, where `H(i)` are the
`vce_neighbors` (default 2) nearest observations of `i`'s own group (ties
included; on the covariates for nnmatch, on the propensity score for
psmatch):

    V(ATE)  = (1/N^2)   sum_i [(Yhat_i(1) - Yhat_i(0) - ATE)^2 + (K_i^2 + 2 K_i - K'_i) sigma2_i]
    V(ATET) = (1/N_1^2) [sum_{D=1} (Y_i - Yhat_i(0) - ATET)^2 + sum_{D=0} (K_i^2 - K'_i) sigma2_i]

Without ties `K'_i = K_i / M`, which gives the textbook
`K_i^2 + (2M - 1)/M K_i`. With bias adjustment the imputations in the first
sum are the adjusted ones.

**Estimated propensity score** (Abadie and Imbens 2016). psmatch subtracts the
effect of estimating the score: with `f_i = dF/d(x'g)` at the estimates
(`p(1-p)` for logit, the normal density for probit), `V_g` the inverse
information of the treatment model and `cov_w,i` the covariance of `(x, y)`
among the `vce_neighbors` nearest observations (by `p`) of group `w`
(excluding `i` itself in its own group),

    ATE:   V - c' V_g c,            c   = (1/N)   sum_i f_i (cov_1,i / p_i + cov_0,i / (1 - p_i))
    ATET:  V - c_t' V_g c_t + d' V_g d,
                                    c_t = (1/N_1) sum_i f_i (cov_1,i + p_i / (1 - p_i) cov_0,i)
                                    d   = (1/N_1) sum_i f_i x_i (Yhat_i(1) - Yhat_i(0) - ATET)

(for the ATET every observation is matched in both directions, so that
`Yhat_i(1) - Yhat_i(0)` is available for the controls). `extra["variance"]`
reports the uncorrected Abadie-Imbens variance and the correction; psmatch
needs `vce_neighbors >= 2`. A simulation in the tests confirms that the
corrected standard errors track the sampling spread of the estimators.

**Neighbour search without n-by-n matrices.** psmatch (and nnmatch with a single
covariate) sorts the pool once: the `M` nearest of a query form a contiguous
window of the sorted pool, found by `M` vectorized comparisons of the next
left and right candidates plus a vectorized binary search for ties; window
means come from prefix sums and the usage counts `K_j` from difference
arrays — O(n log n) in all. nnmatch with several covariates computes exact
distances blockwise (each block of queries against the whole pool, at most
about four million distances in memory at a time) from exact coordinate
differences (`torch.cdist` without the `|a|^2 + |b|^2 - 2ab` expansion, so
duplicated covariate rows are at distance exactly zero and ties are exact),
takes the M-th distance with `topk` and accumulates the matched pairs
sparsely: O(n_1 n_0 k) time, about 3e8 pairs per second, with bounded memory.
A search that would need more than 4e9 distance evaluations raises
`matching_too_large` (use psmatch for very large samples).

The result has one term, `ATE:r1vs0.t` or `ATET:r1vs0.t`, as Stata reports.
Matching accepts bounded fweight duplication, only `covariance="robust"`, a binary treatment and
the `ate`/`atet` estimands.

### Option-level treatment-effects domain

This matrix describes resident tables. Dataset/replay adapters keep their
existing bounded domain and reject `tlevel`, `ematch`, iid matching variance,
new csdid sampling/anticipation/multiplier options and AIPW ATET explicitly.

| Option | RA / IPW / IPWRA / AIPW | NN matching | Propensity matching |
| --- | --- | --- | --- |
| ATE / POM | supported | ATE only | ATE only |
| ATET | binary or multivalued, `tlevel` selects population | binary | binary |
| fweights | likelihood/estimating-equation weights | literal bounded duplication | literal bounded duplication |
| pweights | stacked sandwich | unsupported | unsupported |
| exact-match `ematch` columns | unsupported | exact cells | unsupported |
| robust matching variance | not applicable | Abadie–Imbens | Abadie–Imbens with estimated-PS correction |
| `matching_vce='iid'` | unsupported | unadjusted nnmatch only | unsupported |
| heteroskedastic-probit treatment / WNLS outcome | unsupported | not applicable | unsupported |

For an ATET target `a`, let `S_i=1{D_i=a}`. The AIPW potential-outcome mean
in that population is

    tau_l = sum_i w_i [S_i mu_l(x_i) + 1{D_i=l} p_a(x_i)/p_l(x_i) (y_i-mu_l(x_i))] / sum_i w_i S_i.

Reported effects subtract the control mean; every contrast conditions on the
same target population, including multivalued contrasts. The covariance uses
the full stacked treatment, outcome and mean equations, including estimated
propensity and outcome derivatives. `extra['target_population']` records the
chosen level and estimand. Existing AIPW ATE/POM formulas are unchanged.
The target definition follows the [Stata AIPW manual](https://www.stata.com/manuals/causalteffectsaipw.pdf);
the tests independently solve the stacked equations and differentiate their
Jacobians numerically for all four methods and both multivalued targets.

`ematch=['sector', 'region']` restricts both outcome and variance neighbours
to the exact observed cell. All equal-distance ties are retained and weighted
equally. Insufficient opposite-group or own-group support raises
`exact_match_support`, with no silently dropped observations. The result
records the exact columns, cell count, tie policy and unmatched policy.

Matching frequency weights expand the sample to literal observations, bounded
by 100,000 rows and 2,000,000 data entries before allocation. Result identity
and physical sample positions still refer to the original weighted table;
effective `nobs` counts duplicates. For `matching_vce='iid'`, the common
conditional variance is estimated from squared matched differences around
the estimated effect and substituted into the Abadie–Imbens variance. This
option rejects bias adjustment and applies only to nnmatch. See the
[Stata matching manual](https://www.stata.com/manuals/causalteffectsnnmatch.pdf).
Brute-force exact-cell/tie comparisons and literal row duplication validate
the new point estimates and full reporting covariance.

## 2. Difference in differences: `didregress`

### The model

For observations `i` of group `g` (a state, a firm, a school) in period `t`
— repeated cross-sections or a panel —

    y_it = a_g + c_t + delta D_it + x_it' b + e_it

with group effects `a_g`, time effects `c_t` and the treatment indicator
`D_it` (1 for treated groups in treated periods). Under parallel trends,
`delta` is the average treatment effect on the treated (ATET).

```python
did = oe.didregress(data=df, y="outcome", treatment="treated_post",
                    group="county", time="year", x=["income"])
print(did.summary())
did.tests["parallel_trends"], did.tests["granger"]
```

| Stata | OpenEconometrics |
| --- | --- |
| `didregress (y x1) (treated), group(state) time(year)` | `oe.didregress(data=df, y="y", treatment="treated", group="state", time="year", x=["x1"])` |
| `xtdidregress (y x1) (treated), group(state) time(year)` | the same call (panel or repeated cross-sections) |
| `..., vce(cluster region)` | `covariance="cluster", cluster="region"` |
| (default `vce(cluster group)`) | `covariance="robust"` (the default) |
| — (not clustered) | `covariance="HC1"` / `"nonrobust"` |
| `estat ptrends` | `result.tests["parallel_trends"]` |
| `estat granger` | `result.tests["granger"]` |

### Estimator

Both sets of effects are absorbed exactly by alternating projections
accelerated with conjugate gradients (the shared `engines.absorb` kernel, the
method of `reghdfe`); no dummy matrix is formed. By Frisch-Waugh-Lovell the
slopes are QR least squares of the demeaned outcome on the demeaned `D` and
covariates. A regressor without variation after absorbing the effects is
omitted; if that regressor is `D` itself (every group treated at the same
time, no controls) the error `no_within_variation` is raised.

### Covariance and degrees of freedom

* `covariance="robust"` (default): cluster-robust at the **group** level,
  Stata's default `vce(cluster groupvar)`;
* `"cluster"` with `cluster="..."`: clusters on another column (for example a
  region containing several groups);
* `"HC1"`: heteroskedasticity-robust, `N/(N-K)`;
* `"nonrobust"`: classical, `SSR/(N-K)`.

Cluster covariances use the factor `G/(G-1) (N-1)/(N-K)` and t inference with
`G - 1` degrees of freedom. `K` counts the slopes plus the absorbed effects
that are not nested in the clusters (the `reghdfe` rule computed by
`engines.absorb.absorbed_degrees_of_freedom`): with group clusters the group
effects are free and the `T` time effects (including the constant) count;
without clusters `K = k + G + T - 1`.

### Tests

When every treated group starts treatment in the same period, treatment is
absorbing and there are never-treated controls and at least two
pre-treatment periods:

* `parallel_trends` (Stata's `estat ptrends`): the model is augmented with
  `treated_g x t x 1{t before treatment}` (a linear trend specific to the
  treated groups in the pre-treatment periods, `t` the value of the time
  variable, so unequally spaced periods are respected; a non-numeric time
  column uses the period index); the F(1, df) test of its coefficient.
* `granger` (Stata's `estat granger`): the model is augmented with the leads
  `treated_g x 1{t = t0 - l}` for `l = 1 .. P-1` (`P` pre-treatment periods,
  the earliest being the reference); the joint F test of the leads, i.e. of
  anticipation effects.

Both use the covariance and degrees of freedom of the main fit. With
staggered adoption or a non-absorbing indicator the tests are omitted and
`extra["tests_note"]` says why (use `oe.eventstudy` for pre-trends).

### Result

Terms: `ATET:r1vs0.treated` (equation `ATET`) followed by the covariates
(equation `Controls`). `metrics`: `r_squared` (of the full model with the
effects), `r_squared_within`, `rmse`, `df_resid`, `n_groups`, `n_periods`,
`n_clusters`. `extra`: adoption pattern (`common`, `staggered` or
`not absorbing`), treated and control group counts, first treated periods
(values of the time variable), absorbed degrees of freedom.

Stata's `didregress` also offers `vce(hc2)` with the Bell-McCaffrey degrees
of freedom and the wild cluster bootstrap; neither is implemented.

## 3. Event studies: `eventstudy`

    y_it = a_g + c_t + sum_{e != ref} beta_e 1{t - T0_g = e} + x_it' b + e_it

`T0_g` (`treatment_time`) is the first treated period of group `g`, missing
for never-treated groups. Never-treated groups and the not-yet-treated
observations of later cohorts act as controls through the time effects.
`beta_e` is the effect `e` periods after treatment (`lagE`, `e >= 0`) or before
it (`leadE`, `e < 0`) relative to the `reference` period (default `-1`).

```python
es = oe.eventstudy(data=df, y="outcome", group="county", time="year",
                   treatment_time="first_year", leads=4, lags=5)
es.tests["pretrends"]
table = es.extra["event_table"]      # relative_time, estimate, std_error, ci_low, ci_high
```

* `leads`/`lags` bound the window; relative times beyond it are binned into
  the end points (`e <= -leads` into `lead<leads>`, `e >= lags` into
  `lag<lags>`; `extra["event_table"][...]["binned"]` flags them). Without them
  every observed relative time gets an indicator; if no group is
  never treated, two indicators are then not identified and the collinearity
  screen omits the last one (recorded in `warnings`).
* Relative times without treated observations are omitted with a warning.
* `time` and `treatment_time` must be integer periods in the same units (a
  text time column raises `invalid_time`). Several observations per group and
  period (repeated cross-sections) are allowed.
* Covariance as for `didregress` (default: clustered by group, t(G-1)).
* `tests["pretrends"]`: joint F test that all lead coefficients are zero.
* `extra["event_table"]`: one row per relative time (the reference row has
  estimate 0) with the confidence interval at level `1 - alpha` — the data of
  an event-study plot; `extra["average_post_effect"]`: the mean of the lag
  coefficients with its standard error and interval; `extra["cohorts"]`, the
  number of treated and never-treated groups.

### Limitations of two-way fixed effects

With staggered adoption and treatment effects that differ across cohorts or
evolve over time, the TWFE `delta` is a weighted average of group-time
effects with weights that can be negative (Goodman-Bacon 2021; de
Chaisemartin and D'Haultfoeuille 2020), and TWFE event-study coefficients mix
effects from other relative periods, so leads can be non-zero even under
parallel trends (Sun and Abraham 2021). Both functions add a warning when
more than one adoption cohort is present; with staggered adoption and
heterogeneous effects use `oe.csdid` (next section). Sun-Abraham interaction
weights, imputation estimators and wild cluster bootstrap inference
(Stata's `didregress ..., wildbootstrap`) are not implemented.

## 3b. Staggered adoption: `csdid` (Callaway and Sant'Anna 2021)

For a balanced panel of units observed in integer periods, with
`treatment_time` the first treated period of each unit (missing for
never-treated units), `csdid` estimates the group-time effects

    ATT(g, t) = E[Y_t - Y_b | G = g] - E[Y_t - Y_b | control]

for every cohort `g` and period `t`, comparing the cohort only with clean
controls: never-treated units (`control="never"`, default) or units not yet
treated at `max(t, b)` (`"notyet"`). The base period `b` is `g - 1` after
treatment; before treatment `base="varying"` (default, as R's `did`) uses
`t - 1` and `"universal"` uses `g - 1`. Already-treated units never serve as
controls, so heterogeneous effects across cohorts and time do not
contaminate the estimates.

```python
cs = oe.csdid(data=df, y="lemp", group="county", time="year",
              treatment_time="first_treat", x=["lpop"], method="dr")
cs.extra["simple"], cs.extra["dynamic"], cs.tests["pretrends"]
```

| Stata / R | OpenEconometrics |
| --- | --- |
| `csdid y, ivar(id) time(t) gvar(g)` / `did::att_gt(...)` | `oe.csdid(data=df, y="y", group="id", time="t", treatment_time="g")` |
| `csdid y x, ... method(dripw)` / `est_method="dr"` | `x=["x"], method="dr"` |
| `... method(stdipw)` / `"ipw"`; `method(reg)` / `"reg"` | `method="ipw"` / `"reg"` |
| `csdid ..., notyet` / `control_group="notyettreated"` | `control="notyet"` |
| `... long2` / `base_period="universal"` | `base="universal"` |
| `estat simple / event / group / calendar` / `aggte(type=...)` | `extra["simple"]`, `["dynamic"]`, `["group"]`, `["calendar"]` |

With covariates `x` (their base-period values), the control change is
estimated by outcome regression (`reg`: OLS of the change on `x` among the
controls, Heckman-Ichimura-Todd), normalized inverse-probability weighting
(`ipw`: odds `p/(1-p)` from a logit of cohort membership, Abadie 2005) or the
doubly robust combination (`dr`, default: Sant'Anna and Zhao 2020's
`drdid_panel`). Without covariates the three coincide with the difference in
mean changes. Covariates that are constant or collinear within a group-time
comparison are omitted from it (with a warning), as R's `lm`/`glm` do; a
covariate that is constant among the controls but varies within the cohort
makes the propensity model separate and raises `overlap_violation` for `ipw`
and `dr`.

**Inference.** Each ATT(g, t) solves stacked estimating equations (logit
score, control-group OLS, treated and weighted-control means); the per-unit
influence contributions `phi_i = -e'A^-1 psi_i` (analytic `A`) give the joint
covariance `Phi'Phi` of all ATT(g, t), clustered by unit, with z inference.
`tests["pretrends"]` is the Wald chi2 test that every pre-treatment ATT(g, t)
is zero. The aggregations of `did::aggte` weight the ATT(g, t) by estimated
cohort shares `P(G = g)` and include the influence of the estimated weights:

* `simple`: post-treatment ATT(g, t) weighted by cohort size;
* `dynamic`: event-time effects `theta(e)` (cohort-size weights among cohorts
  observed at `e`) and `dynamic_overall`, their mean over `e >= 0`;
* `group`: cohort averages over post-treatment periods and `group_overall`,
  their cohort-size-weighted mean;
* `calendar`: period effects and `calendar_overall`, their mean.

Each row has `estimate`, `std_error`, `ci_low`, `ci_high`. Differences from
`did`/`csdid`: coefficient-table standard errors are analytic; optional shared
multiplier bands are described below. Propensity scores are not trimmed (a control propensity
within 1e-5 of one raises `overlap_violation`), units first treated in or
before the first period are removed from the estimation sample (so `nobs`
and `metrics["n_groups"]` count the remaining units, as `did` does) and units
treated after the last period count as never treated (both with a warning),
and the default panel must be balanced (`unbalanced_panel` otherwise).

### Additional csdid inference and sample domains

`bootstrap_reps=399, seed=19, uniform=True` uses shared standard-normal
multipliers on the ATT(g,t) influence contributions. Aggregated influences
also include uncertainty in estimated cohort shares. The coefficient table
retains analytic normal pointwise intervals; the `att_gt_bands`, `dynamic`,
`group` and `calendar` extra tables report separate multiplier pointwise
quantiles and optional simultaneous intervals. Each simultaneous family gets
its own `quantile(max(abs(multiplier_noise / analytic_se)), 1-alpha)` critical
value, terms and seed in inference. A simultaneous band covers the chosen
family, not every possible aggregation. This normalization intentionally uses
analytic SEs, rather than R did's bootstrap IQR scale estimate. Both the
multiplier matrix and output draws are bounded by 2,000,000 entries.

| `sample` | Retained comparison | Covariates and sampling contract |
| --- | --- | --- |
| `balanced` (default) | all observed periods per unit | existing reg/IPW/DR unit-cluster influence path |
| `unbalanced` | units observed in both base and target period, separately for each comparison | reg/IPW/DR with baseline covariates; pair-complete estimand, no attrition adjustment |
| `repeated_cross_section` | independent rows in four cohort/control × base/target cells | currently no `group` or covariates; difference of four means and independent-row influence |

The pair-complete unbalanced contract differs from R did's
`allow_unbalanced_panel` route, which uses repeated-cross-section comparisons.
It must not be interpreted as an attrition-corrected population effect.
Repeated cross-sections require at least two observations per retained cell.
`sample_decisions` records each base, target, treated/control support and
selection rule. Panel reconstruction is bounded by 50,000,000 unit-period
entries before allocation; oversized inputs fail explicitly.

`anticipation=A` moves the post-treatment baseline to the observed period
before `g-A`; pre-periods before that cutoff use the selected base convention.
Not-yet controls must remain untreated beyond `max(target, base)+A`. Early
cohorts without a valid unaffected baseline are excluded and recorded.
Future adoption beyond the observed domain with nonzero anticipation is
explicitly unsupported. See the authors' [did att_gt specification](https://bcallaway11.github.io/did/reference/att_gt.html).

`tests/test_econ_csdid_extended.py` independently reconstructs four-cell
influences, pair-complete stacked equations, estimated-share derivatives,
full covariance/pretrend statistics and all simultaneous families. The
default balanced analytic results remain covered by their original tests.

## 4. Regression discontinuity: `rdrobust`

### The design

Observations with running variable (score) `x >= c` are treated. In a
**sharp** design treatment switches on at the cutoff and the estimand is the
jump `tau = lim_{x↓c} E[y|x] - lim_{x↑c} E[y|x]`. In a **fuzzy** design
(`fuzzy="t"`, the treatment received) only the probability of treatment jumps
and the estimand is the ratio of the outcome and treatment jumps, a local
average treatment effect for compliers at the cutoff.

```python
rd = oe.rdrobust(data=df, y="vote", running="margin", cutoff=0.0)
print(rd.summary())        # Conventional / Bias-corrected / Robust
rd.metrics["h_left"], rd.metrics["n_h_left"]
fz = oe.rdrobust(data=df, y="earnings", running="score", cutoff=50, fuzzy="enrolled")
```

| rdrobust (Stata / R) | OpenEconometrics |
| --- | --- |
| `rdrobust y x, c(0)` | `oe.rdrobust(data=df, y="y", running="x", cutoff=0)` |
| `rdrobust y x, p(2) q(3) kernel(uniform)` | `p=2, q=3, kernel="uniform"` |
| `rdrobust y x, h(10) b(20)` / `h(8 12)` | `h=10, b=20` / `h=[8, 12]` |
| `rdrobust y x, h(10) rho(0.5)` | `h=10, rho=0.5` (b = h / rho) |
| `rdrobust y x, bwselect(msetwo)` / `bwselect(cerrd)` | `bwselect="msetwo"` / `"cerrd"` |
| `rdrobust y x, vce(hc3)` / `vce(nn 5)` | `vce="hc3"` / `vce="nn", nnmatch=5` |
| `rdrobust y x, vce(nncluster id)` | `covariance="cluster", cluster="id"` (`vce="nn"`) |
| `rdrobust y x, vce(cluster id)` | `covariance="cluster", cluster="id", vce="hc0"` |
| `rdrobust y x, fuzzy(t)` | `fuzzy="t"` |
| `rdrobust y x, scaleregul(0)` | `scaleregul=0` |
| `rdrobust y x, fuzzy(t) sharpbw` | `fuzzy="t", sharpbw=True` |
| `rdbwselect y x` | the bandwidths in `metrics` / `extra["bandwidth_selection"]` |

### Estimator

On each side of the cutoff a local polynomial of order `p` (default 1: local
linear) is fitted by weighted least squares (QR) with kernel weights
`K((x - c)/h)/h` — triangular `1 - |u|` (default), Epanechnikov
`0.75 (1 - u^2)` or uniform `0.5` on `|u| <= 1` — and the intercepts are
differenced. Following Calonico, Cattaneo and Titiunik (2014), the leading
bias of the order-`p` fit is estimated with an order-`q` polynomial (default
`q = p + 1`) at the bias bandwidth `b` and subtracted:

    beta_p  = G_p^-1 R_p' W_h y,            G_p = R_p' W_h R_p
    Q       = R_p W_h - h^(p+1) (W_b R_q G_q^-1 e_{p+1}) L',  L = R_p' W_h ((x-c)/h)^(p+1)
    beta_bc = G_p^-1 Q' y

The three result rows are rdrobust's:

| term | estimate | standard error |
| --- | --- | --- |
| `Conventional` | `tau` from `beta_p` | conventional `se` from `G_p^-1 M(R_p W_h) G_p^-1` |
| `Bias-corrected` | `tau_bc` from `beta_bc` | conventional `se` |
| `Robust` | `tau_bc` | robust `se` from `G_p^-1 M(Q) G_p^-1` (robust bias-corrected inference) |

The `Robust` row's confidence interval is the recommended one: it accounts
for the variability of the bias estimate, so it has correct coverage at the
MSE-optimal bandwidth. In a fuzzy design the estimates are ratios of the
outcome and treatment jumps, the bias correction and the variances use the
linearization `s = (1/tau_T, -tau_Y/tau_T^2)` of the ratio, and
`extra["first_stage"]` / `extra["reduced_form"]` hold the two jumps. Inference
is z (normal); the covariance matrix of the three rows is diagonal (they are
alternative estimates of one parameter; their covariances are not reported).

### Variance

The meat is `M(Z) = sum_i (s'e_i)^2 z_i z_i'` with residual vectors `e_i`:

* `vce="nn"` (default, `nnmatch=3`): nearest-neighbour residuals
  `sqrt(J/(J+1)) (y_i - mean of the J matched y)`, matching within the same
  side and bandwidth on the running variable. The matched set starts with all
  other observations at the same running value and grows by whole blocks of
  tied values, nearest first (both when equidistant), until it has at least
  `min(nnmatch, n - 1)` members — the loop of `rdrobust_res`, vectorized over
  the distinct values.
* `vce="hc0"`..`"hc3"`: fitted residuals times 1, `sqrt(n/(n-d))`,
  `1/sqrt(1 - h_ii)` or `1/(1 - h_ii)` (`n` the observations within the
  bandwidth on that side, `d` the number of polynomial coefficients, `h_ii`
  the weighted leverage of the order-`p` fit, also used for the bias stage).
* `covariance="cluster"`: the scores are summed by cluster on each side and
  multiplied by `((n-1)/(n-k)) (G/(G-1))`, `n` the observations within the
  bandwidths on that side and `k = p + 1` (also for the robust variance, as
  `rdrobust_vce` counts the columns of `Q`).

### Bandwidth selection

Without `h`, the bandwidths come from the rdbwselect plug-in rules. A pilot
`c_bw = C_K min(sd(x), IQR(x)/1.349) N^(-1/5)` (`C_K` = 2.576 triangular, 2.34
Epanechnikov, 1.843 uniform; IQR from R's type-2 quantiles) is used for every
variance constant. For a polynomial of order `o` estimating derivative `nu`,
each side yields

    V = (2nu + 1) h^(2nu+1) [G^-1 M G^-1]_{nu,nu}
    B = sqrt(2(o + 1 - nu)) BConst beta_{o+1},  BConst = h^nu [G^-1 R'W ((x-c)/h)^(o+1)]_nu
    R = 2(o + 1 - nu) 3 BConst^2 Var(beta_{o+1})      (regularization)

where `beta_{o+1}` comes from an order-`o_B` fit at bandwidth `h_B`. `mserd`
then computes, with rate `1/(2o + 3)`:

1. `d = ((V_l + V_r) / (B_r - B_l)^2)^rate` with `o = nu = q + 1`, `o_B = q + 2`
   and `h_B` the whole support of each side;
2. `b = ((V_l + V_r) / ((B_r - B_l)^2 + scaleregul (R_l + R_r)))^rate` with
   `o = q`, `nu = p + 1`, `o_B = q + 1`, `h_B = d`;
3. `h` the same with `o = p`, `nu = 0`, `o_B = q`, `h_B = b`.

`msetwo` applies the rules side by side, `msesum` uses the sum of the biases,
`msecomb1` the smaller of mserd and msesum, `msecomb2` the median of mserd,
msesum and msetwo (per side). The `cer*` versions multiply `h` by
`N^(-p/((3+p)(3+2p)))` (coverage-error-rate optimal for the robust interval;
`N` is the number of clusters on the left plus on the right when the variance
is clustered). With `bwrestrict=True` (default) every bandwidth is capped at
the distance from the cutoff to the farthest observation. As in rdbwselect,
step 1's bias bandwidth is each side's range plus `1e-8` in the units of the
running variable, so selected bandwidths are not exactly equivariant to a
rescaling of a running variable whose range is of order 1e-6 or smaller.

Fuzzy designs select the bandwidth as rdrobust does by default: every
constant is computed for the combination `s = (1/tau_T, -tau_Y/tau_T^2)` of the
outcome and treatment columns, `tau_Y` and `tau_T` being the order-`nu`
coefficients of that side's pilot fit (`rdrobust_bw`). `sharpbw=True` uses the
outcome equation alone, and so does a treatment that is constant on one side
(perfect compliance, rdbwselect's rule); `extra["bandwidth_equation"]` says
which was used. `h=` and `b=` (a number or `[left, right]`) override the
selection; `rho` sets `b = h / rho` (default `b = h` when only `h` is given).

### Result

* terms `Conventional`, `Bias-corrected`, `Robust` (z inference, confidence
  intervals at level `1 - alpha`);
* `metrics`: `h_left`, `h_right`, `b_left`, `b_right`, `n_left`, `n_right` (all
  observations on each side), `n_h_left`, `n_h_right`, `n_b_left`, `n_b_right`
  (observations with positive weight at `h` and `b`), `p`, `q`, `cutoff`;
* `extra`: `side_estimates` (conventional and bias-corrected limits on each
  side), `bandwidth_selection` (method, pilot), kernel, vce, nnmatch, design,
  and for fuzzy designs `first_stage` (jump in treatment with conventional and
  robust standard errors), `reduced_form` and, when the bandwidths were
  selected, `bandwidth_equation` (`fuzzy` or `outcome (sharp)`).

### Performance

Each side is sorted once; every kernel fit works on the contiguous slice of
observations within the bandwidth; nearest-neighbour residuals cost one
vectorized step per added block of tied values; least squares are QR. On one
million observations the complete procedure (bandwidth selection and
estimation, `vce="nn"`) takes about half a second.

## 5. RD plots: `rdplot`

`oe.rdplot` returns the data of a regression-discontinuity plot as a
`TableSet` with two tables (draw them with any plotting library):

* `bins`: one row per bin — `side`, `bin` (negative on the left), `x_low`,
  `x_high`, `x_mid`, `x_mean`, `y_mean`, `n`, `y_se` (standard deviation over
  `sqrt(n)`), `ci_low`, `ci_high` at level `1 - alpha`;
* `poly`: the global polynomial of order `p` (default 4) fitted separately on
  each side, evaluated on `grid` (default 200) points per side.

```python
plot = oe.rdplot(data=df, y="vote", running="margin", cutoff=0)
plot["bins"], plot["poly"], plot.attrs["bins_left"]
```

The running variable on each side is split into `J` bins, evenly spaced
(`es`, equal widths) or quantile spaced (`qs`, equal counts). The number of
bins follows Calonico, Cattaneo and Titiunik (2015) with constants estimated
from each side's data (`n_s` observations, support length `l`, consecutive
differences `dx_i`, `dy_i` of the data sorted by `x`, `mu'` the derivative of a
global quartic fit):

| | integrated variance `V` | integrated squared bias `B` |
| --- | --- | --- |
| evenly spaced | `(1/(2l)) sum dx_i dy_i^2` | `(l^2/(12 n_s)) sum mu'(x_i)^2` |
| quantile spaced | `(1/(2 n_s)) sum dy_i^2` | `(n_s/12) sum dx_i^2 mu'(xbar_i)^2` |

* `binselect="es"` / `"qs"`: IMSE-optimal `J = ceil((2 B n_s / V)^(1/3))`, the
  minimizer of `B/J^2 + V J/n_s`;
* `binselect="esmv"` (default) / `"qsmv"`: mimicking variance
  `J = ceil((var(y_s)/V) n_s / log(n_s)^2)`, which gives many more bins whose
  scatter resembles the raw data;
* `nbins=J` or `nbins=[J_left, J_right]` fixes the numbers.

| rdplot | OpenEconometrics |
| --- | --- |
| `rdplot y x, c(0)` | `oe.rdplot(data=df, y="y", running="x", cutoff=0)` |
| `rdplot y x, binselect(qs) p(3)` | `binselect="qs", p=3` |
| `rdplot y x, nbins(20 20)` | `nbins=[20, 20]` |

## Performance

Timings measured in the verification pass on an Apple-silicon laptop CPU
(float64, PyTorch's default thread pool), synthetic data held in memory:

| model | data | time |
| --- | --- | --- |
| `teffects` ra / ipw / ipwra / aipw, linear outcome | 1,000,000 rows, 10 covariates | 0.5 – 1.5 s |
| `teffects` aipw, logit outcome | 1,000,000 rows, 10 covariates | 1.0 s |
| `teffects` psmatch (sorted search, AI 2016 correction) | 1,000,000 rows, 3 covariates | 1.1 s |
| `teffects` nnmatch (blockwise exact distances), 3 covariates | 20,000 / 60,000 / 100,000 rows | 1.0 / 2.6 / 8.0 s |
| `didregress` | 1,000,000 rows, 20,000 groups x 50 periods | 0.2 s |
| `eventstudy` (leads = lags = 5) | the same panel | 0.5 s |
| `csdid` unconditional / dr with one covariate | 400,000 rows (50,000 units x 8 periods) | 0.1 / 0.15 s |
| `rdrobust` (mserd selection + estimation, `vce="nn"`) | 1,000,000 rows | 0.45 s |
| `rdplot` | 1,000,000 rows | 0.3 s |

nnmatch with several covariates is the only quadratic-time procedure: its
cost grows with `n_treated x n_control` (bounded memory); use psmatch, or
nnmatch on a single covariate (sorted search), for very large samples.

## Verification

Every number is checked against independent implementations written in the
tests (NumPy, SciPy and statsmodels appear only there):

* `tests/test_econ_teffects_oracle.py` — ra / ipw / ipwra / aipw for every
  method x estimand x outcome model x treatment model: the complete stacked
  moment system written from the formulas above and solved jointly by
  damped Newton / `scipy.optimize.root` from naive starting values, with a
  Richardson-extrapolated numerical Jacobian for the sandwich; the full
  covariance of the effects and of the auxiliary equations is compared (rtol
  1e-6), as are pweights, clusters, fweights (= duplicated rows), multivalued
  treatments with a non-default control and categorical covariates, a
  closed-form influence function of the RA-linear ATE, and invariance to
  affine rescaling, row order and the scale of pweights. Matching estimators
  against brute-force O(n^2) matched sets, with the Abadie-Imbens variance
  assembled as `V^E + V^tau(X)` from the matched sets (not from `K'`),
  including ties on an integer grid, bias adjustment, the sorted 1-D search
  and the Abadie-Imbens (2016) correction from the paper's formulas. DiD and
  event studies against explicit dummy-variable least squares with a manual
  cluster sandwich (all four covariances, aweights normalized to sum N,
  pweights, fweights, the ptrends/granger augmented regressions with equally
  and unequally spaced periods). csdid against a NumPy re-implementation of
  R `did::att_gt` with the `DRDID` influence functions (`drdid_panel`,
  `std_ipw_did_panel`, `reg_did_panel`) and `did::aggte` with its weight
  influence function. rdrobust against statsmodels WLS (with `b = h` the
  bias-corrected estimate is the order-`p+1` local polynomial and its HC0/HC1
  robust variance that of the order-`p+1` fit), a set-based definition of
  the nearest-neighbour residuals, and an independent NumPy port of
  `rdrobust_bw` / `rdbwselect` for all ten selectors, fuzzy designs and
  clusters.
* `tests/test_econ_teffects_adversarial.py` — empty, one-row and tiny
  samples, all-missing columns, constant outcomes, separation and overlap
  failures, collinearity, single clusters, bad weights, text where numbers
  are needed, gaps and labels in time, 1e-8 / 1e8 magnitudes, options at their
  bounds: each gives a finite result or an `AnalysisError` with a helpful
  code.

## Uncertain conventions

These follow the published formulas and the documented behaviour of the
Stata / rdrobust / did commands as closely as we could establish, but have
not been checked against output of those programs (none is available to the
test suite):

* `teffects` ra/ipw/ipwra/aipw: no small-sample factor in the robust and
  cluster sandwiches (the `gmm` convention); in particular `vce(cluster)`
  applies no `G/(G-1)`, which we could not confirm for Stata's `teffects`.
* `pstolerance` is applied to every level's probability for every
  observation, also for the ATET.
* nnmatch/psmatch: with ties, the variance uses `K^2 + 2K - K'` (ATE) and
  `K^2 - K'` (ATET), the exact generalization of the textbook
  `K^2 + (2M-1)/M K`; ties are exact equalities of the computed distances
  (Stata may use a tolerance); the Mahalanobis matrix is the full-sample
  covariance of the covariates; the bias-adjustment regressions are weighted
  by `K_j` (the matched sample, Abadie, Drukker, Herr and Imbens 2004);
  conditional variances use the `vce_neighbors` nearest observations of the
  own group *including ties*, on the propensity score for psmatch; a caliper
  raises an error when any observation has fewer than `M` matches within it.
* psmatch correction (Abadie-Imbens 2016): the conditional covariances
  exclude the unit itself when it belongs to the matched group, use divisor
  `L - 1`, and `V_g` is the inverse observed information of the treatment
  model (for probit this differs from the expected information of the
  paper in finite samples).
* `didregress`: the cluster small-sample factor counts the absorbed effects not
  nested in the clusters (reghdfe's rule, equal to `xtreg, fe` for a panel);
  Stata's `didregress` on repeated cross-sections may count the group effects
  as `areg` does. `estat ptrends` / `estat granger` are reproduced as the
  augmented-regression tests described above, for common adoption only.
* `eventstudy`: end-point binning when `leads`/`lags` are given.
* `rdrobust`: cluster factor `((n-1)/(n-k)) (G/(G-1))` with `k = p + 1` for
  both the conventional and the robust variance; HC2/HC3 bias-stage residuals
  reuse the order-`p` leverages; the CER rules scale `h` only; the fuzzy
  bandwidth selector (side-specific `s` from the pilot fit, perfect-compliance
  fallback) follows the official fixture. Mass-point
  adjustments (`masspoints`, `bwcheck`) use unique-value pilots and preliminary
  bandwidth floors; the pilot bandwidth
  is capped at the largest distance to the cutoff only when `bwrestrict`.
* `csdid`: analytic coefficient-table standard errors; optional normal IF multiplier bands use analytic scaling rather than did's IQR scaling. Propensity scores
  not trimmed (`did`'s `drdid_panel` trims controls with scores above
  0.995); covariates taken at the base period of each comparison.
* `rdplot`: the bin-selection constants above are our reading of CCT (2015)
  (in particular the `n_s / log(n_s)^2` factor of the mimicking-variance rule
  and the use of the side's own sample size in the IMSE rule).

## Not implemented

* Sun-Abraham and imputation event-study estimators; wild cluster bootstrap
  for DiD (the generic fixed-design OLS wild contract is separate).
* Stata's `hetprobit` treatment model and `wnls` outcome option; sampling
  weights and bias-adjusted iid variance for matching; covariate-adjusted
  repeated-cross-section csdid and attrition corrections for unbalanced panels.
* `rdrobust` covariate adjustment, derivative estimation and weights on Dataset
  replay (resident DataFrames support these options); automatic density bandwidth
  selection, restricted fits and plug-in VCE in `rddensity`. See
  [the validated inference domains](inference-extensions.md#rd-and-manipulation).
* `didregress` `vce(hc2)` with Bell-McCaffrey degrees of freedom and the wild
  cluster bootstrap.
