# Regression discontinuity under local randomization: `rdrandinf`, `rdwinselect`, `rdsensitivity`, `rdrbounds`

These procedures implement the local-randomization approach to regression
discontinuity (RD) designs of the community `rdlocrand` package (Cattaneo,
Frandsen and Titiunik 2015; Cattaneo, Titiunik and Vazquez-Bare 2016, 2017),
following its R/Python release 3.0. They complement the continuity-based
`rdrobust`/`rdplot`/`rddensity` of [teffects.md](teffects.md). Everything runs
in OpenEconometrics on float64 PyTorch tensors (int64 assignment indices); no
statistics package is used at run time.

The procedures are tests, not model fits: they register no estimator and
return a `TableSet` of named tables whose `.attrs` hold the scalar settings and
results (JSON-safe; `stata_parity_validated` is always `False`).

```python
import openecon as oe

sel = oe.rdwinselect(data=df, running="margin", covariates=["pop", "income"],
                     wmin=0.5, wstep=0.5, nwindows=10, reps=1000)
wl, wr = sel.attrs["recommended_left"], sel.attrs["recommended_right"]

res = oe.rdrandinf(data=df, y="vote", running="margin", wl=wl, wr=wr,
                   statistic="all", reps=5000, seed=1)
res["tests"]           # statistic, randomization and large-sample p-values, power

ci = oe.rdrandinf(data=df, y="vote", running="margin", wl=wl, wr=wr, reps=5000,
                  ci=0.05, ci_grid=[x / 4 for x in range(61)])
ci["ci"], ci["ci_grid"]

sens = oe.rdsensitivity(data=df, y="vote", running="margin",
                        wlist=[1, 2, 3, 4], tlist=[0, 4, 8, 12], reps=2000)
rb = oe.rdrbounds(data=df, y="vote", running="margin", wlist=[1, 2],
                  expgamma=[1.5, 2, 3], reps=1000)
```

| Stata / R (`rdlocrand`) | OpenEconometrics |
| --- | --- |
| `rdrandinf Y R, wl(-1) wr(1) statistic(all) reps(1000) seed(666)` | `oe.rdrandinf(data=df, y="Y", running="R", wl=-1, wr=1, statistic="all")` |
| `rdrandinf Y R, ci(.05 3(.25)20)` | `ci=0.05, ci_grid=[3 + 0.25 * i for i in range(69)]` |
| `rdrandinf Y R, p(1) evall() evalr()` | `p=1, evall=..., evalr=..., vce="HC3"` |
| `rdrandinf Y R, fuzzy(T tsls)`; R `fuzzy = list(T, "tsls")` | `fuzzy="T", fuzzy_statistic="tsls"` (`"ar"`/`"itt"` default) |
| `rdrandinf Y R, bernoulli(prob)` | `bernoulli="prob"` (a column), a common probability, or `True` (window share) |
| `rdrandinf Y R, covariates(X1 X2) wmin(.5) wstep(.125)` | `covariates=["X1", "X2"], wmin=0.5, wstep=0.125` |
| `rdwinselect R X1 X2, wobs(5) nwindows(10) approximate` | `oe.rdwinselect(data=df, running="R", covariates=["X1", "X2"], approx=True)` |
| `rdsensitivity Y R, wlist(.75(.25)2) tlist(0(1)5) ci(-1 1)` | `oe.rdsensitivity(..., wlist=[...], tlist=[...], ci=[-1, 1])` |
| `rdrbounds Y R, expgamma(1.5 2 3) wlist(.5 .75 1) reps(1000)` | `oe.rdrbounds(..., expgamma=[1.5, 2, 3], wlist=[0.5, 0.75, 1])` |

## The design

Units with running variable R at or above the cutoff c are treated,
D = 1(R >= c); Rc = R - c. Local randomization assumes that inside a window
W = [wl, wr] around c the assignment behaves as in a randomized experiment and
the outcomes do not depend on R (or, with p > 0, depend on it through a
polynomial that is adjusted for). Two assignment laws are available:

* **fixed margins** (complete randomization, the default): every set of n1 of
  the n window units is treated with probability 1/C(n, n1);
* **Bernoulli trials**: unit i is treated with probability pi_i independently
  (`bernoulli=`). Assignments without a treated or without a control unit are
  inadmissible and are dropped (the package records them as NA).

The sharp null of a constant effect tau0 imputes the untreated outcomes
Y - tau0 D. In a fuzzy design with treatment received T the Anderson-Rubin /
intention-to-treat statistic uses Y - tau0 T, permuting D only.

## Statistics and p-values

| `statistic` | T(D) | large-sample companion |
| --- | --- | --- |
| `diffmeans` (`ttest`) | mean(Y \| D=1) - mean(Y \| D=0); with kernel weights the weighted-least-squares coefficient of D | z = T / sqrt(s1^2/n1 + s0^2/n0) (Neyman); weighted: HC2 standard error |
| `ksmirnov` | D = max_v \|F1(v) - F0(v)\| over the distinct values | R's `ks.test`: the exact tie-aware Smirnov law when n0 n1 < 10,000, else the Kolmogorov limit at sqrt(n0 n1/(n0+n1)) D |
| `ranksum` | (T0 - n0 (n+1)/2) / sqrt(n0 n1 s_r^2 / n), T0 the control rank sum with mid-ranks, s_r^2 the rank variance | normal |
| `all` | the three above on the same assignments | as above |
| fuzzy `ar` | `diffmeans` of Y - tau0 T | Neyman (weighted: HC2) |
| fuzzy `tsls` | IV coefficient of T instrumented by D | HC1 sandwich (p = 0), `vce` (p > 0); no randomization test |

The randomization p-value is the (probability-weighted) share of admissible
assignments with |T*| >= |T(observed)|, the package's
`mean(abs(stats.distr) >= abs(obs.stat))` without a plus-one correction.
`method="auto"` (default) enumerates every assignment when there are at most
`reps` of them (C(n, n1) under fixed margins, 2^f for f units with 0 < pi_i < 1
under Bernoulli trials) and otherwise draws `reps` assignments;
`method="exact"` always enumerates (up to 10 million assignments),
`method="monte_carlo"` always draws. Exact enumeration unranks the
combinatorial number system in bounded chunks; draws use a seeded CPU
`torch.Generator` (a fixed-margins draw treats the n1 units with the smallest
of n uniform keys; a Bernoulli draw treats U_i < pi_i). Both are vectorized as
assignment-by-unit matrices processed in chunks of about 2^20 elements.
`attrs["method"]`, `attrs["draws"]` and the Monte Carlo standard errors
sqrt(p(1 - p)/draws) record what was done.

Ties are decided exactly where the statistic allows it: the KS statistic is
compared through integer numerators max |c1 n0 - c0 n1| (cross-multiplied
under Bernoulli trials) and, under fixed margins, the rank sum through the
integer 2 T0 - n0 (n+1). Floating statistics (difference in means, Bernoulli
rank sums, Hotelling) count a draw as extreme when |T*| >= |T| - tol with tol
16 n eps times the statistic's scale, so assignments whose statistic equals the
observed one in exact arithmetic (for example the mirror assignment of a
balanced design) are counted.

Units are put in a canonical order (running variable, then outcome and the
other used columns) before any draw, so a Monte Carlo result depends on the
data and the seed but not on the row order of the input.

The reported power is the package's: 1 - Phi(1.96 - d/se) + Phi(-1.96 - d/se)
for mean contrasts and Phi(sqrt(3 n0 n1 / ((n0+n1+1) pi)) d / sigma - 1.96)
for the rank sum (sigma the window's outcome standard deviation), with
d = `d` or `dscale` times the control-side outcome standard deviation
(default 0.5 sd). The constant 1.96 is the package's.

## Polynomial adjustment (p > 0)

As in rdlocrand 3.0, p > 0 replaces randomization inference by large-sample
inference: the coefficient of D in the weighted least-squares regression of
Y - tau0 D (fuzzy: Y - tau0 T) on [1, D, P, D P], P = (R - e, ..., (R - e)^p)
with e = `evalr` for treated and `evall` for control units (default the
cutoff), using units with positive kernel weight, with an HC1, HC2 or HC3
(default) sandwich and a normal reference. TSLS replaces the second column by
T instrumented by D (projected regressors for the leverage, structural
residuals). No randomization p-value is reported. The randomization test on
polynomial-adjusted outcomes of the 2016 package was withdrawn in rdlocrand
3.0 and is not offered. p > 0 is available for mean contrasts and fuzzy
designs only (rdlocrand 3.0 silently falls back to p = 0 for `ksmirnov`,
`ranksum`, `all` and `interfci`; here these combinations are refused).

## Confidence sets

`ci=alpha` inverts the test of H0: tau = tau0 over the grid `ci_grid`
(default: the package's ten steps over the classical 95% interval of the
window, rounded to two decimals). Every grid value is tested on the same
assignments, and the accepted values (p >= alpha) are reported as maximal runs
[first, last] (the package's `find_CI`); a disconnected set gives several
`ci` rows and `ci_grid` keeps every p-value. Notes flag sets that reach the end
of the grid. With p > 0 the grid is inverted with the HC test (without a grid,
mean contrasts report estimate +/- z se; fuzzy Anderson-Rubin sets need a
grid); TSLS reports the normal interval. `interfci=alpha` reports Rosenbaum's
interval under arbitrary interference, T - (q_{1-alpha/2}, q_{alpha/2}) with q
type-7 quantiles of the randomization distribution (fixed margins, one sharp
statistic).

## Window selection: `rdwinselect`

Nested windows are built on the sorted Rc exactly as the package does:
without `wmin` the first window holds `obsmin` (default 10) units per side and
each next window adds `wobs` (default 5) units per side, growing by whole
blocks of tied values and then extending the shorter side so the window is
symmetric (`findwobs_sym`); `wasymmetric=True` lets the sides grow separately
(`findwobs`); `wmasspoints=True` builds one window per mass point; `wstep`
widens symmetric windows by a fixed step (R's `seq`). `wmin` is a half-width or
offsets [left, right] from the cutoff. In each window an exact two-sided
binomial test of P(treated) = 1/2 is reported, and each covariate is tested
with `statistic` (`diffmeans`, `ksmirnov`, `ranksum`; `hotelling` tests them
jointly with T^2 = (n0 n1/n) delta' S_pool^-1 delta and, approximately,
F = (n-k-1)/((n-2)k) T^2 ~ F(k, n-1-k)), by randomization (all covariates share
the assignments; each window uses `seed`) or, with `approx=True`, by the
large-sample companion. Kernel weights use |Rc|/w_right on both sides and, for
p = 0, replace zero weights by the machine epsilon (as the package). p > 0
forces large-sample HC tests of the polynomial jump at the cutoff or, with
`evalat="means"`, at the side means of Rc.

The recommended window is the largest one such that the minimum balance
p-value is at least `level` (default 0.15) in it and in every smaller window;
a window without a defined p-value counts as failing (for example the
large-sample difference in means with a single unit on a side, which the
package reports as NaN); without covariates or
when the smallest window fails, `recommended_left`/`recommended_right` are
`None`. The `windows` table holds the limits (units of R), the minimum p-value
and its covariate, the binomial p-value, units below/above the cutoff and the
method/draws; `balance` holds every covariate's statistic and p-value;
`sample` reproduces the package's counts within the 1st, 5th, 10th and 20th
percentiles of the distance to the cutoff (type-1 quantiles rounded to five
decimals). Missing covariates are excluded window by window
(`dropmissing=False`, with `missing="drop"`) or removed before the windows are
built (`dropmissing=True`). `rdrandinf` without `wl`/`wr` but with
`covariates` runs the same selection (options `obsmin`, `wmin`, `wobs`,
`wstep`, `wasymmetric`, `wmasspoints`, `nwindows`, `dropmissing`, `rdwstat`,
`approx`, `rdwreps`, `level`) and reports its `windows` table.

## Sensitivity: `rdsensitivity`

For every window in `wlist` (right limits; `wlist_left` or the mirror images)
and every effect in `tlist`, the p-value is computed as by `rdrandinf`; within
a window every tau0 reuses the same assignments. Defaults follow the package:
the ten windows of `rdwinselect` without covariates and the ten-step default
grid (p = 0); p > 0 requires `tlist`. `ci=[left, right]` names one of the
windows and reports its confidence set at `ci_alpha`. The `pvalues` table is
long (tau, window, limits, p-value); `windows` gives the units per side and the
method per window.

## Rosenbaum bounds: `rdrbounds`

In each symmetric window |R - c| <= w (`wlist`, default the first five
`rdwinselect` windows rounded to two decimals) the baseline p-value treats the
assignment as Bernoulli trials with the window's treated share (or `prob`);
`fmpval=True` adds the fixed-margins p-value. Hidden bias Gamma = exp(gamma)
(`expgamma`, default 1.5, 2, 2.5, 3, or `gamma`) lets every unit's treatment
probability lie in [1/(1+Gamma), Gamma/(1+Gamma)]. As in the package, for
u = 1..n the u units with the largest observed outcomes receive the high
probability and the others the low one; `upper_bound` is the maximum and
`lower_bound` the minimum of these n p-values. Every configuration is a
Bernoulli randomization p-value of the chosen statistic (`ranksum` default);
Monte Carlo configurations share one stream of uniforms (common random
numbers), so the maximum over u is not inflated by independent simulation
noise, and small windows are enumerated exactly.

Rosenbaum's (2002) one-sided construction would search the smallest-outcome
configurations for the lower bound, but here that visits the same n p-values:
every statistic offered satisfies |T(1 - Z)| = |T(Z)| and Bernoulli trials
with probabilities 1 - pi are the law of 1 - Z under pi, so giving the high
probability to the u smallest outcomes has the p-value of giving it to the
n - u largest (and u = n, all units high, equals all units low). Rosenbaum's
result that monotone configurations attain the extremes is for one-sided
tests; for the package's two-sided p-values it does not hold. Enumerating
every one of the 2^n high/low assignments on small windows (124 random
designs, n = 9, all four statistics, ties and kernel weights) found smaller
p-values than `lower_bound` in 123 designs, sometimes by a factor of five,
and larger ones than `upper_bound` in 2 (by about 0.01). Read both as the
package's summary over these n configurations, not as guaranteed bounds over
the sensitivity model (`lower_bound` in particular). Configurations are
ordered by the observed outcome Y, as in the package, also when `nulltau` is
not 0.

## Validation

`tests/test_rdlocrand_reference.py` reproduces R rdlocrand 3.0 (R 4.6.1;
script `tests/reference/rdlocrand_reference.R`, data regenerated bit for bit
from an integer MINSTD generator):

* deterministic output to rounding: all statistics, Neyman, HC2 (kernel), HC1 (TSLS) and HC1/HC2/HC3 (polynomial)
  large-sample p-values and standard errors, p = 1, 2 polynomial fits with
  `evall`/`evalr`, triangular and Epanechnikov kernels, fuzzy TSLS (p = 0, 1)
  and Anderson-Rubin tests, normal and grid-inverted confidence intervals,
  power, summary statistics, both KS branches;
* exact randomization p-values against brute-force enumeration of every
  assignment in R with the package's own statistic function (75,582 fixed-
  margins assignments; kernel-weighted; fuzzy; Bernoulli with common, window-
  share and unit probabilities), the KS law also against `ks.test`'s exact
  tie-aware Smirnov p-value, grid inversion and every Rosenbaum configuration;
* Monte Carlo p-values within simulation error of `rdrandinf` with 20,000 reps;
* `rdwinselect` window sequences (symmetric, asymmetric, `wstep`, `wmin`
  pairs, `obsmin`/`wobs`, mass points, non-zero cutoff), unit counts,
  binomial and approximate balance p-values for every statistic, kernel and
  p = 1 (`cutoff`/`means`), recommended windows and summary counts;
  `rdsensitivity` default windows and grid and its p > 0 grids.

`tests/test_rdlocrand_review.py` (independent review; script
`tests/reference/rdlocrand_review.R`) adds R comparisons on discrete and heaped
running variables (mass points in the symmetric, asymmetric, `wmin`, `wmin`
pair, `wstep` and non-zero-cutoff sequences), Hotelling with five and six
covariates (F approximation and exact enumeration), ranksum/KS/p = 2/
Epanechnikov balance tests with six covariates, asymmetric windows with
polynomial and kernel fits, fuzzy TSLS and Anderson-Rubin fits with
triangular/Epanechnikov kernels, window selection inside `rdrandinf`,
sensitivity grids with kernels and fuzzy designs, and exact laws enumerated in
R with the package's statistics: Bernoulli trials with logistic unit
probabilities, Bernoulli and kernel-weighted Anderson-Rubin, a tied integer
outcome on two mass points (with `ks.test`'s exact law) and every
configuration of the Rosenbaum search; plus the zero-kernel-weight sandwich and
a NumPy enumeration of all 2^n Rosenbaum configurations.

`tests/test_rdlocrand_kernels.py` checks the kernels against exact rational
enumeration (`fractions.Fraction`, ties included, balanced mirror assignments,
weighted statistics with undefined assignments, Bernoulli laws with degenerate
units), Hotelling against NumPy, SciPy (Welch t statistic, rank sums,
Mann-Whitney with tie correction, `ks_2samp` exact, `kstwobign`, `binomtest`,
F) and statsmodels WLS HC1/HC2/HC3, the TSLS sandwich against explicit NumPy
algebra, Monte Carlo seeding, convergence and uniformity.
`tests/test_rdlocrand_procedures.py` checks invariances (shifting R and the
cutoff together, input row order, reflecting the outcome), sensitivity grids,
confidence sets and the interference interval against NumPy enumeration,
Rosenbaum bounds at Gamma = 1, the window-recommendation rule, missing-data
policy, the failure contract and JSON/LaTeX rendering.

## Conventions that differ from the reference packages

* **KS comparisons.** R's rdlocrand compares an observed KS statistic from
  `ks.test` with permuted ones from a second floating-point routine, so
  mathematically tied draws can be miscounted; this implementation (like the
  Python port) compares exact integer numerators.
* **Rank-sum variance.** The tie-corrected permutation variance
  n0 n1 var(ranks)/n is used for `ranksum` and `all` (R uses it for `ranksum`
  only; the Python port never does); without ties all agree.
* **Large-sample KS.** The Kolmogorov limit is summed to double precision;
  R's series stops at tolerance 1e-6 (differences up to about 1e-5 just below
  sqrt(n0 n1/N) D = 1). SciPy's `ks_2samp(method="asymp")` uses a finite-n law
  instead and is not the package convention.
* **Monte Carlo.** Draws come from torch's generator, so they differ from R's
  `sample`/`runif` streams; enumeration is used whenever it is affordable
  (R instead draws C(n, n1) random permutations when `reps` exceeds that
  number). Each window and each Rosenbaum configuration restarts the generator
  at `seed` (common random numbers).
* **Refusals instead of silent fallbacks.** p > 0 with `ksmirnov`, `ranksum`,
  `all`, `hotelling` or `interfci`, Bernoulli assignment with large-sample
  inference, `covariates` together with `wl`/`wr`, and `ci_grid` with TSLS
  raise `AnalysisError` (the package ignores the option or falls back).
* **Fuzzy Anderson-Rubin with kernels** uses the kernel-weighted difference in
  means (R ignores the kernel weights for this statistic at p = 0 but uses them
  at p > 0 and for TSLS); with the uniform kernel it equals R.
* **Zero kernel weights.** Triangular and Epanechnikov weights vanish for units
  exactly at a window limit (common with discrete or heaped running variables,
  or limits set at data values). The kernel HC2 (difference in means) and HC1
  (TSLS, p = 0) sandwiches here use the units with positive weight, as
  rdlocrand's own `rdlocrand_hc_fit` does for p > 0. R's p = 0 path calls
  `sandwich::vcovHC` on `lm`/`AER::ivreg` fits with zero weights, which mixes
  the positive-weight count of the bread with all rows of the meat: in a
  measured case R's TSLS standard error was deflated by n_positive/n (0.294
  instead of 0.446; p 1.4e-7 instead of 5.1e-4) and its HC2 standard error
  recycled vectors of different lengths (0.246 instead of 0.369). Without zero
  weights both agree with R to rounding. When a side keeps fewer than two units
  with positive weight the HC2 companion is undefined and only the
  randomization p-value is reported (with a note).
* **Confidence sets under Bernoulli trials** invert the Bernoulli test (R's
  `ci` reverts to fixed margins).
* **Fuzzy designs under Bernoulli trials** (`rdrandinf`, `rdrbounds`) run the
  Anderson-Rubin statistic; rdlocrand 3.0 stops for them (it calls its model
  with an empty statistic), so these are checked against brute-force
  enumeration with the package's difference in means, not against the package.
* **Missing data** follow `missing=` (`"raise"` default, `"drop"`): R drops
  incomplete outcome rows silently and keeps them for window selection.
* `rdrbounds` takes no `p`/`evalat` (polynomial adjustment is unavailable for
  the bounds in 3.0); its default window list uses the windows' right limits
  (the Python port's choice; R reads the wrong column).

## Limitations and budgets

* No Stata, R or Python parity is claimed beyond the measured cases above;
  `stata_parity_validated` stays `False`.
* Resident tables only (no Dataset route), CPU float64. Up to 10 million rows;
  exact enumeration up to 10 million assignments; up to 10 million draws. The
  assignment-by-unit work (assignments x units x tested columns) is capped at
  4e9 for each randomization run (the tests of an `rdrandinf` window, its
  confidence grid, each Rosenbaum configuration set), for all windows of an
  `rdwinselect` or `rdsensitivity` call together and for all configurations of
  an `rdrbounds` call; buffers are checked against the workspace budget
  (`OPENECON_WORKSPACE_MB`); exceeding a budget raises `work_budget_exceeded` /
  `workspace_limit` before the first draw. On one CPU thread a fixed-margins
  difference in means costs roughly 10-15 ns per unit and draw (n = 200,000
  units, 10,000 draws: about 25 s). Bernoulli KS comparisons need windows of at
  most 100,000 units.
* The deprecated `obsstep` option of `rdwinselect`, plots and `firststage`
  printing are not provided; the per-draw randomization distribution is not
  returned (only its interference-interval quantiles).
* Windows contain at least two units per side for `rdrandinf` and one for
  `rdsensitivity`/`rdrbounds`; p > 0 needs more units than polynomial terms.

## References

Cattaneo, M. D., Frandsen, B. and Titiunik, R. (2015). Randomization inference
in the regression discontinuity design: an application to party advantages in
the U.S. Senate. *Journal of Causal Inference* 3(1): 1-24.

Cattaneo, M. D., Titiunik, R. and Vazquez-Bare, G. (2016). Inference in
regression discontinuity designs under local randomization. *Stata Journal*
16(2): 331-367.

Cattaneo, M. D., Titiunik, R. and Vazquez-Bare, G. (2017). Comparing inference
approaches for RD designs: a reexamination of the effect of Head Start on child
mortality. *Journal of Policy Analysis and Management* 36(3): 643-681.

Rosenbaum, P. R. (2002). *Observational Studies*, 2nd ed. Springer.
