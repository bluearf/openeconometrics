# Shared frailty, recurrent events and survival treatment effects

This page documents the shared-frailty and recurrent-event extensions of the
survival family. Every estimator runs in OpenEconometrics on float64 PyTorch
tensors with analytic first and second derivatives; R, SciPy and NumPy appear
only in the committed reference generators and tests.
`provenance["stata_parity_validated"]` is `False` for every result: Stata command
names identify the corresponding model, not a measured vendor comparison.

| OpenEconometrics | Stata | R reference |
| --- | --- | --- |
| `oe.stcox_frailty(..., shared="g")` | `stcox x, shared(g)` | `coxph(Surv(t, d) ~ x + frailty(g))`, `coxme(Surv(t, d) ~ x + (1 \| g))` |
| `oe.streg_frailty(..., shared="g", frailty="gamma")` | `streg x, distribution(weibull) frailty(gamma) shared(g)` | explicit NumPy/SciPy integration over the frailty density |
| `oe.stcox_recurrent(..., model="ag")` | `stcox x, vce(cluster id)` on `(start, stop]` records | `coxph(Surv(start, stop, event) ~ x + cluster(id))` |
| `oe.stcox_recurrent(..., model="pwp_total"/"pwp_gap")` | `stcox x, strata(enum) vce(cluster id)` | `coxph(... + strata(enum) + cluster(id))` on total or gap time |
| `oe.stcox_recurrent(..., model="wlw")` | `stcox x, strata(k) vce(cluster id)` on the expanded data | `coxph(... + strata(k) + cluster(id))` on the marginal data |
| `oe.stteffects(..., method="ra"/"ipw"/"ipwra")` | `stteffects ra/ipw/ipwra` | explicit NumPy stacked estimating equations |

## `oe.stcox_frailty`: Cox regression with a shared frailty

```python
fit = oe.stcox_frailty(data=df, time="t", failure="d", x=["x1", "x2"], shared="g")
fit.metrics["theta"], fit.tests["theta"]            # variance and its LR chibar2 test
fit.extra["frailty_estimates"][:3]                  # log frailty per group
ln = oe.stcox_frailty(data=df, time="t", failure="d", x=["x1"], shared="g",
                      frailty="lognormal", ties="efron")
```

**Model.** `h_ij(t) = z_i h0_s(t) exp(x_ij'b + offset)`. The records of group `i`
(`shared`) share the frailty `z_i`: gamma with mean 1 and variance theta
(`frailty="gamma"`, the default) or log-normal with `ln z_i ~ N(0, theta)`
(`frailty="lognormal"`). The baseline hazard is unspecified per stratum
(`strata`). `entry` gives `(start, stop]` records; the frailty then multiplies the
whole history of the group, which is R's convention for counting-process data and
the appropriate one for split records and recurrent events. It is not the
distribution of a frailty selected by a genuine delayed entry.

**Estimation.** For a fixed theta, `b` and the log frailties `w` maximize the
penalized partial likelihood (Therneau, Grambsch and Pankratz 2003)

    PPL(b, w) = PL(b, w) - P(w),   P = (1/theta) sum_g (exp(w_g) - w_g)  (gamma)
                                   P = sum_g w_g^2 / (2 theta)            (log-normal)

where `PL` is the Breslow or Efron partial likelihood of the design `[X, G]` (`G` the
group indicators, never formed: its blocks are group sums and two failure-time by
group risk-set tables). Newton-Raphson uses the analytic (k + q)-dimensional
Hessian. theta maximizes the integrated log partial likelihood of that solution:

- gamma: `L(theta) = PL(b, w) + nu sum_g w_g + sum_g [D_g + nu ln nu + ln Gamma(nu + D_g)
  - ln Gamma(nu) - (nu + D_g) ln(nu + D_g)]`, `nu = 1/theta`, `D_g` the failures of
  group `g`. This is R survival's I-likelihood (`frailty.gammacon`). With Breslow
  ties it is the exact gamma marginal likelihood at the nonparametric baseline
  hazard (the EM solution of Klein 1992 and Nielsen et al. 1992), shifted by the
  constant that makes `L(0)` the Cox partial likelihood; the test suite checks the
  EM fixed point `w_g = ln((nu + D_g)/(nu + A_g))` with explicit loops. The group
  terms are evaluated as `D_g - nu log1p(D_g/nu) + sum_{j<D_g} log1p((j - D_g)/(nu + D_g))`,
  free of cancellation for every theta.
- log-normal: the Laplace approximation `L(theta) = PPL - 1/2 ln det(I + theta J_ww)`,
  `J_ww` minus the `ww` block of the partial-likelihood Hessian (Ripatti and
  Palmgren 2000), the integrated partial likelihood of R `coxme`.

The search evaluates `L` on the grid theta = 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1,
2, 4, 8, 16 (doubling above while it still increases, up to 1024, otherwise
`boundary_solution`), brackets the best point and refines it: gamma by the
Brent-Dekker root of the analytic derivative
`dL/dln(theta) = -nu (sum_g w_g + sum_g [ln(nu/(nu + D_g)) + sum_{j<D_g} 1/(nu + j)])`
(envelope theorem; `sum_g exp(w_g) = q` at every PPL maximum), log-normal by
Brent's derivative-free maximization. When the integrated likelihood gain over the
Cox model is at most `1e-10 max(1, |L(0)|)` the estimate is the boundary
`theta = 0` and the result is the Cox model (`extra["frailty"]["boundary"]`).
`theta=` fixes the variance instead (no test of theta is then reported).

**Inference.** Coefficients are log hazard ratios with z tests. Their covariance is
the `b` block of the inverse negative Hessian of the PPL at `(b, w, theta_hat)`,
conditional on theta_hat: R `coxph`'s `var` (with `sparse=FALSE`) and coxme's
fixed-effect variance. `extra["se2"]` holds the standard errors of R's alternative
`var2 = H^-1 I H^-1` (`I` the unpenalized information). `tests["theta"]` is the
likelihood-ratio statistic `2 (L(theta_hat) - L(0))` referred to the boundary
mixture `chibar2(01)`: `p = 0.5 P(chi2(1) >= stat)` and `p = 1` at 0 (Self and
Liang 1987), as Stata reports for `shared()`. `tests["model"]` is the Wald chi2 of
all coefficients. No standard error of theta is reported (R reports none; Stata's
convention is not validated here). `metrics`: `log_likelihood` (integrated),
`log_likelihood_cox`, `penalized_log_likelihood` (R's convention: the gamma
penalty is `nu sum(exp(w) - w - 1)`), `partial_log_likelihood`, `theta`, `aic`/`bic`
(k + 1 parameters when theta is estimated, N = records), `n_groups`, `n_failures`,
`time_at_risk`. `extra`: hazard ratios, `frailty_estimates` (log frailty, frailty
and failures per group), `profile` (every evaluated `(theta, L)`).

**Domain and failures.** Breslow or Efron ties; no weights, no robust/cluster
covariance, no `tvc`. At least two groups (`invalid_groups`), at most 2,000 groups
and 10,000,000 failure-time by group cells (`resource_limit`). A non-converging
inner fit raises `nonconvergence`; a variance that keeps increasing raises
`boundary_solution`; a non-positive fixed theta raises `invalid_option`.

**Verification** (`tests/test_econ_survival_frailty.py`, fixtures from
`scripts/generate_survival_frailty_oracles.py` and `.R`): R survival 3.8.6
`coxph(... frailty(g, distribution="gamma", sparse=FALSE))` and coxme 2.2.22 with
Breslow and Efron ties, with and without `(start, stop]` records and strata, at
fixed variances (coefficients and frailties within 1e-8, covariances within 1e-7
relative, integrated likelihoods within 1e-8) and at the maximizer of R's own
integrated likelihood found by `optimize` (variance within 2e-5 relative; our
integrated likelihood is never below R's); a no-frailty data set whose R profile
lies below the Cox likelihood on its whole grid (boundary estimate, LR = 0, p = 1);
the gamma EM fixed point; analytic against numerical derivatives; missing-row and
row-order invariance; JSON round trip and rendering. Measured on the committed
fixtures: maximal absolute coefficient difference 1.6e-8 at the estimated variance
and 1e-13 at fixed variances; maximal relative variance difference 3.8e-7.

## `oe.streg_frailty`: parametric regression with a shared frailty

```python
fit = oe.streg_frailty(data=df, time="t", failure="d", x=["x1", "x2"], shared="g",
                       distribution="weibull", frailty="invgaussian")
fit.extra["theta"]                 # theta with the exp-transformed /ln_the interval
```

**Model.** Conditional on the frailty `z_i` of group `i`, the hazard is
`z_i h(t | x)` with the exponential or Weibull `h` of `oe.streg` (PH, the default,
or AFT via `metric="aft"`; the same parameterizations). `z_i` is gamma
(`frailty="gamma"`) or inverse Gaussian (`frailty="invgaussian"`), both with mean 1
and variance theta.

**Marginal likelihood (closed form).** With `D_i` failures and cumulative hazard
`A_i = sum_j H(t_ij)` in group `i`, the frailty integrates out exactly:

    l_i = sum_j delta_ij ln h(t_ij) + ln M_{D_i}(A_i),   M_m(s) = E[Z^m exp(-s Z)]
    gamma:   ln M_m(s) = nu ln nu + ln Gamma(nu + m) - ln Gamma(nu) - (nu + m) ln(nu + s),  nu = 1/theta
    inverse Gaussian, u = sqrt(1 + 2 theta s):
             ln M_m(s) = (1 - u)/theta - m ln u + ln sum_{k=0}^{n} (n+k)!/(k!(n-k)!) (theta/(2u))^k,
             n = m - 1 (m >= 1), n = 0 (m = 0)

The inverse-Gaussian form follows from `int z^(m-3/2) exp(-a z - b/z) dz =
2 (b/a)^((m-1/2)/2) K_(m-1/2)(2 sqrt(ab))` and the finite sum of the half-integer
modified Bessel function `K_(n+1/2)` (DLMF 10.49.12); `(1 - u)/theta` is evaluated
as `-2 s/(1 + u)`. The first and second derivatives in `(s, ln theta)` are written
out (no numerical differentiation) and combined with the per-record derivatives of
`ln h` and `H` in the linear index and `ln p`, giving the analytic score and Hessian
of `(b, /ln_p, /ln_the)`. Newton-Raphson starts from the model without frailty
(theta = 0, which is `oe.streg`) with theta = 0.5, 0.05 and 2 and keeps the best
converged interior maximum. When none exceeds the log likelihood without frailty by
more than `1e-10 max(1, |ll|)` (runs whose `ln theta` falls below -18 count as
reaching the boundary) the estimate is `theta = 0` and the result is the `oe.streg`
model; `extra["frailty"]["score_at_zero"]` is the score of theta at 0,
`sum_i ((D_i - A_i)^2 - D_i)/2`, which is negative in that case. A non-converging fit
with a positive score at 0 raises `nonconvergence`.

**Inference.** Observed-information covariance of all parameters, including
`/ln_p` and `/ln_the` (Stata's names); `extra["theta"]` and `extra["p"]` carry the
delta-method standard errors and exp-transformed intervals. `tests["theta"]` is the
LR chibar2(01) test of theta = 0 against the model without frailty,
`tests["model"]` the Wald chi2 of the main-equation covariates. The log likelihood
is reported on Stata's log-time scale (`+ sum delta ln t`, as `oe.streg`).

**Domain.** Exponential and Weibull, PH and AFT; no delayed entry (the `entry` role
is not declared), weights, robust or cluster covariance, ancillary covariates or
strata. Inverse-Gaussian groups with more than 2,000 failures raise
`resource_limit`.

**Verification.** `scripts/generate_survival_frailty_oracles.py numpy` integrates
the frailty numerically for every group (`scipy.integrate.quad` over `ln z`,
relative tolerance 1e-13), maximizes with BFGS on central-difference gradients and
differentiates the maximum numerically for the covariance, for gamma and
inverse-Gaussian data with Weibull and exponential models. Measured: maximal
absolute parameter difference 4.0e-9, log likelihoods equal within 1e-13, standard
errors within 4.4e-7 relative (the tests allow 2e-4, the numerical Hessian's
accuracy); likelihood
values at fixed parameters equal the quadrature within 1e-9; analytic derivatives in
both metrics with offsets; AFT/PH invariance; equality with `oe.streg` at theta = 0
and on a data set whose score at 0 is negative.

## `oe.stcox_recurrent`: recurrent and multiple events

```python
ag = oe.stcox_recurrent(data=df, time="stop", entry="start", id="id", failure="event",
                        x=["treat", "age", "z"])                     # Andersen-Gill
gap = oe.stcox_recurrent(..., model="pwp_gap", ties="efron")
wlw = oe.stcox_recurrent(..., model="wlw", max_events=3)
```

**Data.** Counting-process records, one row per `(entry, time]` interval of a
subject `id` (both required), `failure = 1` when the interval ends in an event.
Covariates may change between records. Records of one subject must not overlap
(`overlapping_records`); a record ending on or before its entry is excluded with a
note, as `oe.stcox` does. With the subject's records ordered by entry, `prior` is
the number of the subject's events in earlier records and `last` the time of the
most recent one (0 if none), computed by sorted cumulative sums and cumulative
maxima of record positions (exact, no loop over observations).

**Models.**

- `ag` (Andersen and Gill 1982): one risk set on the total-time scale.
- `pwp_total` (Prentice, Williams and Peterson 1981): strata by event number
  `prior + 1` on the total-time scale; records after a subject's last event form the
  next (event-free) stratum, as R's `strata(enum)` does.
- `pwp_gap`: the same strata on the time since the previous event,
  `(entry - last, time - last]`. Because the subtraction leaves floating-point
  differences between equal decimal gaps (`0.3 - 0.1 != 0.2`), the derived gap times
  go through R survival's `aeqSurv` tie rule (the default `coxph` timefix): sorted
  distinct values closer than `sqrt(eps)` absolutely or relative to the mean absolute
  time form one run that takes its smallest value. A record that collapses to length
  zero raises `invalid_survival_time`. Total-time models use the times as given.
- `wlw` (Wei, Lin and Weissfeld 1989): strata `k = 1..K`, `K` the largest number of
  events of a subject. Stratum `k` holds every record of a subject that starts before
  its `k`-th event, on the total-time scale, with an event only on the record ending
  at the `k`-th event; subjects with fewer events are censored at the end of
  follow-up. The data are expanded to one row per record and stratum (at most
  5,000,000 rows, `resource_limit` otherwise).

`max_events` caps the strata: PWP pools event numbers above the cap into the last
stratum, WLW models the first `max_events` events only.

**Estimation and inference.** Cox partial likelihood (Breslow, the default, or Efron
ties) by Newton-Raphson with the analytic derivatives of `oe.stcox`. The default
`covariance="robust"` is the Lin-Wei sandwich of the efficient score residuals of
every (expanded) row, summed first within the original record and then within the
subject, times `G/(G-1)` (Stata's `vce(cluster id)`; R's `cluster(id)` applies no
`G/(G-1)`). `covariance="nonrobust"` is the inverse information, which ignores the
dependence of a subject's events and is offered only for comparison. `tests["model"]`
is the Wald chi2 (robust) or LR chi2 (nonrobust) of all coefficients. `metrics`
include `n_subjects`, `n_records`, `n_rows`, `n_failures`, `n_strata` and
`max_events`; `extra["strata"]` lists the rows and events of every stratum.
A shared frailty for recurrent events is `oe.stcox_frailty(..., entry="start",
shared="id")` (the gamma AG frailty model of R's `coxph(... + frailty(id))`).

**Domain.** No weights, no extra strata, offsets or `tvc`; at least two subjects
(`insufficient_clusters`).

**Verification** (`tests/test_econ_survival_recurrent.py`). R survival 3.8.6
`coxph` with `Surv(start, stop, event)`, `cluster(id)` and `strata()` on event
numbers, gap times and WLW data that the R script derives independently, for AG, PWP
total and gap time, WLW, and the capped PWP/WLW variants with Breslow and Efron ties
(100 subjects, up to 12 events, 356 records, 3,612 WLW rows): coefficients within
4e-15, robust covariances (times G/(G-1)) within 3e-14 relative (largest element
difference over the largest element), model-based covariances within the tested
1e-9 relative, partial log likelihoods within 5e-12. The event history is checked against
explicit loops, AG against `oe.stcox(..., id=, covariance="robust")`, and the AG gamma
frailty against R `coxph(... + frailty(id, theta = 0.5))`.

## `oe.stteffects`: treatment effects on mean survival time

```python
ra = oe.stteffects(data=df, time="t", failure="d", treatment="treat", x=["x1", "x2"])
ipw = oe.stteffects(..., method="ipw", cx=["x2"])                 # censoring covariates
ipwra = oe.stteffects(..., method="ipwra", estimand="atet")
ra.extra["potential_outcome_means"], ra.extra["auxiliary_equations"]["OME1"]
```

**Target.** For a binary treatment `A` (control: the lowest level or `control=`),
the potential-outcome means `POM_j = E[T(j)]` of the survival time, the ATE
`POM_1 - POM_0`, the ATET (both means over the treated) or both POMs
(`estimand="pomeans"`). The mean integrates the fitted Weibull survivor function to
infinity: it extrapolates beyond the follow-up and relies on the Weibull form.

**Estimators** (Weibull PH `S(t | x) = exp(-exp(x'b) t^p)`, mean
`m(x) = Gamma(1 + 1/p) exp(-x'b/p)`):

- `ra`: one Weibull model per arm fitted by maximum likelihood (censoring in the
  likelihood); `POM_j` is the mean of `m_j(x)` over the sample (ATE, pomeans) or the
  treated (ATET).
- `ipw`: a logit treatment model on `tx` (default `x`) and one Weibull censoring
  model per arm on `cx` (default `x`; censoring is the event). `POM_j` is the
  normalized weighted mean of the observed failure times of arm `j` with weights
  `delta / (P(A = j | z) G_j(t | c))`, `G_j` the fitted censoring survivor function
  (ATET: treated weight `delta / G_1`, control weight
  `delta P(A=1|z) / (P(A=0|z) G_0)`). Fitting the censoring model separately per arm
  is the declared convention of this implementation; it is not validated against
  Stata output.
- `ipwra`: the logit treatment model and per-arm Weibull outcome models fitted with
  inverse-probability weights (ATE `1/P(A=j|z)`; ATET 1 for the treated and
  `P(A=1|z)/P(A=0|z)` for controls); POMs as in `ra`.

**Inference.** The logit, Weibull outcome or censoring scores and the POM moments
form one system `sum_i psi_i(theta) = 0`, solved sequentially (each nuisance model is
an exact root of its own equations; the POM moments are linear). The covariance is
the M-estimation sandwich `A^-1 B A^-T` with the analytic Jacobian `A` (including the
derivatives of the weights in the treatment and censoring parameters) and
`B = sum psi psi'` (`covariance="robust"`, the default) or `sum_g (sum_{i in g} psi_i)(...)'`
(`covariance="cluster"`), without a small-sample factor - the convention of
`oe.teffects` (Stata's gmm-based `teffects`). The reported ATE/ATET and control POM
are the linear transform `(POM_1 - POM_0, POM_0)` of the POMs and their covariance.
`extra["auxiliary_equations"]` reports the nuisance equations (`TME1`, `OME0`/`OME1`
or `CME0`/`CME1`, constants restored after internal centring) with their stacked
standard errors; `extra["overlap"]` the propensity summaries. Estimated treatment
probabilities below `pstolerance` (default 1e-5) raise `overlap_violation`.

**Domain.** Binary treatment only (`invalid_treatment` otherwise); Weibull outcome and
censoring models only; no weights, delayed entry or restricted means; an arm without
censoring cannot fit the `ipw` censoring model (`insufficient_censoring`).

**Verification.** `scripts/generate_survival_frailty_oracles.py numpy` fits the
logit with statsmodels and the Weibull models with SciPy BFGS, writes every
estimating function from its definition in NumPy, differentiates the summed moments
by central differences and forms `A^-1 B A^-T` (600 observations; RA/IPW/IPWRA for
ATE and ATET, a 40-cluster IPWRA case and an IPW case with its own censoring
covariates). Measured: POMs within 6e-11, POM covariances within 3e-10 relative. The
RA outcome equations equal per-arm `oe.streg` fits, `pomeans` equals the POMs of the
ATE fit and switching the control level flips the effect.

## Performance

Apple silicon CPU, one thread, synthetic data with two covariates and about 60%
failures (measured in the implementation pass): `oe.stcox_frailty` with 4,000 records
in 400 groups 1.8 s (gamma, 17 profile evaluations) and 0.7 s (log-normal); 5,000
records in 1,000 groups 3.4 s and 3.8 s; `oe.streg_frailty` on the same data below
0.05 s. The Cox frailty cost is dominated by the dense (k + q)-dimensional Newton
system of every profile evaluation; the parametric frailty and `oe.stteffects` are
O(n k^2) per Newton step.
