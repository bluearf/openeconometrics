# Eight additional prospective design contracts

These APIs extend the bounded planning family with prespecified population
assumptions. They consume no observed sample, fitted model or Dataset. All
numerical work is native CPU float64; development reference libraries are not
shipped. A supported probability law does not imply blanket software parity.

| Linear stage | Public API | Scope |
|---|---|---|
| MARKET-529 | `power_welch` | Fixed-design Satterthwaite/noncentral-t approximation |
| MARKET-530 | `power_unbalanced_anova` | Fixed unequal-allocation Gaussian F law |
| MARKET-531 | `power_unequal_cluster_mean` | Known Gaussian exchangeable cluster covariance |
| MARKET-532 | `power_mcnemar_unconditional` | Random-discordance power of conditional exact test |
| MARKET-533 | `precision_twomeans_unknown` | Pooled normal two-mean width assurance |
| MARKET-534 | `precision_binomial` | Full Clopper-Pearson width distribution |
| MARKET-535 | `precision_poisson` | Garwood rate width with explicit omitted-tail bound |
| MARKET-536 | `survival_accrual` | Expected events under uniform accrual/exponential clocks |

The runnable [example](../examples/prospective_extensions.py) executes all eight
methods and displays 16 complete plan/scenario tables within the console output
limit. Full auxiliary tables and assumptions are retained by `summary_state`;
`restore_summary` reproduces their JSON and LaTeX. The verification script also
replays every input and reads the complete result files back from owned storage.

## Heterogeneous fixed allocations

`power_welch(effect, *, sd1, sd2, n1, n2, alpha=.05, alternative="two-sided")`
plans a two-sample Welch test at explicit integer group sizes. The signed effect
is `(mean2-mean1)-null_difference`. With `v_j=sd_j²/n_j`, the design uses
`SE=sqrt(v1+v2)`, `nu=(v1+v2)²/[v1²/(n1-1)+v2²/(n2-1)]` and noncentrality
`effect/SE`. Rejection probability uses a noncentral t with this population-based
Satterthwaite df. **This is approximate Welch power**: the actual analysis
estimates both variances and a random df from its samples. Group independence
and normal observations are assumed; SDs are planning assumptions, not known
analysis variances. Both sizes must be 2..20000, alpha in `[1e-8,.5)`, and
absolute noncentrality at most64. Small fractional df use a smoothed native
chi-radius integral with a `2e-11` refinement gate. Maximum quadrature order128
and a conservative2MiB law workspace are declared. Failure raises; there is no
normal replacement. [Stata's two-means method](https://www.stata.com/manuals13/psspowertwomeans.pdf)
provides the comparison convention; no licensed vendor execution is claimed.

`power_unbalanced_anova(means, *, sizes, sd, alpha=.05)` accepts explicit
integer allocations and group means for the fixed one-way equality-of-means
F test. `mu_bar=sum(n_j*mu_j)/N` and
`lambda=sum[n_j*(mu_j-mu_bar)²]/sd²`; numerator and denominator df are `K-1`
and `N-K`. Noncentral-F power is exact under independent homoskedastic normal
errors; the common SD is specified for design and estimated by the pooled
within-group residual variance during analysis. Every group contribution is
returned. It accepts2..30 groups, positive integer sizes, `N<=20000`, `N>K`
and `lambda<=4096`. This is not Welch/repeated/random-effects ANOVA. No generic
total-N inversion is claimed. [Stata's one-way method](https://www.stata.com/manuals13/psspoweroneway.pdf)
describes fixed group means and unequal allocations.

`power_unequal_cluster_mean(effect, *, sd1, sd2, sizes1, sizes2, icc1, icc2,
weighting="participant", alpha=.05, alternative="two-sided")` accepts exact
size vectors for two independent arms. A cluster mean of size `m` has variance
`sd²*[ICC+(1-ICC)/m]`. Participant weighting uses `a_j=m_j/sum(m)`;
cluster weighting uses `a_j=1/J`. The arm variance is the exact sum of
`a_j²*Var(cluster_mean_j)`. The complete arm-mean covariance is diagonal;
their difference variance is the sum. Gaussian power is exact **only when
these SD/ICC covariances are known at analysis**, with fixed sizes and common
arm means. Estimated ICC, cluster-t, optimal mixed-model weights and random
size/allocation uncertainty are outside this contract. The Gaussian random
intercept model is documented by [Lauer et al.(2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4382318/);
the implementation directly evaluates its known-covariance weighted mean,
rather than substituting a mean-size/CV approximation. ICC must be in `[0,1]`,
each arm has1..10000 clusters and at most10000000 participants. Every cluster's
weight, mean variance, variance contribution and ICC=0 reference contribution
is saved. Design effects refer to the same declared weights at ICC=0, not an
optimal mixed-model relative efficiency.

All three return complete `plan`, `allocation`, `scenarios` and `settings`
tables. The three scenarios retain fixed allocation and multiply the signed
effect or centered ANOVA means by0/.5/1. Complete summary JSON preserves every
allocation/settings row; JSON readback reproduces the tables and LaTeX exactly.
These are prespecified designs without data, fitted coefficients, observed
p-values, missing-data treatment or a fitted-model covariance. No sample-size
minimum, MDE inversion, Dataset/device acceleration or blanket vendor parity is
inferred. Runtime laws use native float64 CPU Torch with early live-buffer
budgets and no SciPy dependency.

Independent tests check full SciPy noncentral distributions, all allocation and
covariance entries, signed-tail/translation/scaling/permutation limits, invalid
numeric/design/resource domains, foreign default devices and complete owned-file
JSON readback. A protocol declared before execution also runs320000 actual
Welch tests on raw independent normal observations:8 cells,40000 trials each,
sizes8/24,24/8,40/60,80/80 with SD ratios and null/design shifts0/2.
Every trial estimates its own variance/df; all failures stay in denominators.
The absolute approximation tolerance is.025. This is a bounded diagnostic,
not a universal Welch calibration guarantee. Source/frozen/native/vendor
evidence remain separate.

# Prospective confidence-interval width assurance

These three methods plan the random total width of a future two-sided confidence interval. `confidence` controls the interval's coverage level; `assurance` controls the unconditional probability that its future total width is at most the requested width. Population parameters are prespecified design assumptions. They are not observed estimates. All methods require exactly one of `n` and `width`:

- Given `n`, return the assurance quantile of future width, together with its actual attained width-event probability.
- Given `width`, exhaustively inspect integer sample sizes and return the first design that qualifies within the explicit search and work budgets. No monotonicity shortcut or ambiguous integer is skipped.

The width comparison is inclusive (`future_width <= evaluated_width`). These are exact sampling-law constructions evaluated with native float64 numerical kernels, not exact arithmetic or a claim of unrestricted parity with other software. Inputs/results, full assumptions, selected design, neighbouring scenarios, probability/tail accounting, resource limits and references survive saved-summary restoration and JSON/LaTeX rendering. No random simulation, external estimator library or GPU is used at runtime.

## Two independent means with unknown common variance

```python
oe.precision_twomeans_unknown(
    sd=1.2, n=20, ratio=0.6, confidence=0.95, assurance=0.9,
)
oe.precision_twomeans_unknown(sd=1.2, width=1.5, ratio=0.6, max_n=100)
```

Signature: `precision_twomeans_unknown(*, sd, n=None, width=None, ratio=1.0, confidence=.95, assurance=.9, max_n=10000)`.

The two groups are independent iid normal samples with a common prespecified population SD `sd`. At analysis, their common variance is estimated by the pooled sample variance. `n` means `n1`; `n2 = ceil(ratio*n1)`, using the exact shortest decimal representation of `ratio` for integer allocation and per-arm budget boundaries. Thus `ratio=.14, n1=50` gives `n2=7`; binary multiplication roundoff cannot add a participant. Both groups require at least two observations and each must fit `max_n`. With `df = n1+n2-2`, the future total pooled-t interval width is

`W = 2*t_(1-alpha/2,df)*sd*sqrt(1/n1+1/n2)*sqrt(ChiSquare(df)/df)`.

The returned width is its continuous `assurance` quantile. The plan also gives the width-event CDF, actual group sizes and degrees of freedom. Width and SD use the mean-difference measurement units. Ratios lie in `[.01,100]`; at most 10,000 first-group candidate designs are admitted. This contract covers the unconditional width probability and independent common-variance normal model. Unequal population variances, Welch intervals, dependence, weights and confidence-conditional width probabilities require different laws.

SAS documents independent-sample CI precision, allocation, common SD and the distinction between conditional and unconditional width probabilities. This method explicitly uses the unconditional version; the formula above is the direct pooled-variance chi-square derivation. [SAS TWOSAMPLEMEANS statement](https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_power_syntax84.htm).

## Binomial proportion

```python
oe.precision_binomial(p=0.2, n=20, confidence=0.95, assurance=0.9)
oe.precision_binomial(p=0.35, width=0.6, assurance=0.8, max_n=30)
```

Signature: `precision_binomial(*, p, n=None, width=None, confidence=.95, assurance=.9, max_n=500)`.

For independent Bernoulli trials with prespecified `p` in `[0,1]`, enumerate every count `k=0..n` with its full Binomial probability and two-sided central Clopper-Pearson interval. Width is the upper minus lower endpoint, in proportion units. Endpoints have lower limit zero at count zero and upper limit one at count `n`. Complementary counts share one canonical width; all equal-width masses are grouped before the inclusive width quantile is assessed. The saved `width_distribution` retains every count, interval, width, probability and qualification flag.

The quantile is the smallest enumerated width whose represented CDF minus the stated float64 summation allowance reaches assurance. The plan reports actual attained mass and lower/upper probability bounds. Endpoint planning proportions have exact point mass. For `p=.5` and `n<=53`, combination masses and their dyadic cumulative sums are exactly representable, preserving equality at the assurance boundary. In general, a width search whose probability bounds straddle assurance refuses with `unresolved_assurance`.

`max_n` is bounded by 2,000 trials. The entire call, including adjacent scenarios, admits at most 20,000 CP count-width evaluations and 1,000,000 count cells. Exhaustive search can hit this work budget before `max_n`; it refuses without returning an unverified partial design. CP is a conservative exact-coverage interval; its coverage and width assurance are distinct probabilities. [statsmodels proportion_confint](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html).

## Poisson event rate with fixed exposure

```python
oe.precision_poisson(rate=1.2, exposure_per_unit=0.5, n=10)
oe.precision_poisson(
    rate=1.2, exposure_per_unit=0.5, width=3.0, assurance=0.8, max_n=20,
)
```

Signature: `precision_poisson(*, rate, exposure_per_unit=1.0, n=None, width=None, confidence=.95, assurance=.9, max_n=1000, tail_tolerance=1e-12, max_count=10000)`.

Independent homogeneous Poisson units have fixed exposure `exposure_per_unit`, so total exposure is `T=n*exposure_per_unit` and total count has mean `rate*T`. Rate and width use events per exposure unit. Garwood limits for count `k` are `ChiSquare(alpha/2,2k)/(2T)` and `ChiSquare(1-alpha/2,2(k+1))/(2T)`; the lower endpoint at count zero is exactly zero. Zero planning rate is supported and gives deterministic count zero. Random exposure, dependence and overdispersion are outside this model. [statsmodels confint_poisson](https://www.statsmodels.org/dev/generated/statsmodels.stats.rates.confint_poisson.html).

Every count from zero through the smallest admitted cutoff is retained. The native incomplete-gamma tail `gamma_p(cutoff+1, rate*T)` bounds omitted Poisson mass by `tail_tolerance`; probabilities are never renormalized after truncation. Width-event lower/upper bounds include represented mass, the explicit float64 summation allowance and all omitted mass, without assuming that omitted widths qualify or fail. Mass conservation is checked. These numerical allowances and independent oracle tests are not a formal interval-arithmetic proof of every kernel operation.

The fixed-n width is the smallest retained width whose represented CDF minus allowance reaches assurance, conservative within the omitted-mass budget. Actual attained represented mass, lower/upper probability bounds, cutoff, omitted mass and total exposure are reported. A first-n search refuses an ambiguous probability crossing. `tail_tolerance` lies in `[1e-14,1e-6]`, `max_n<=10000` and `max_count<=100000`; at most 20,000 interval-width evaluations and 1,000,000 count cells apply to the complete call. Insufficient cutoff, work/memory budget, unattainable search target and underflowed positive mean fail explicitly.

All three methods admit `confidence` in `(.5,1-1e-8]` and `assurance` in `[1e-6,1-1e-6]`. Their assurance is a probability statement under the prespecified model, rather than a guaranteed width for every future sample.

## Validation delivered with the module

The 62-case independent test file uses SciPy only as a development oracle for Student-t/chi-square, beta CP and Garwood/Poisson formulas. It checks finite support, width ties, first-integer searches, decimal allocation and per-arm budget boundaries, unequal allocation, zero/one proportions, zero counts/rate, extreme confidence, exposure-unit scaling, adversarial inputs, preflight memory/work limits and ambiguity refusal. Saved summaries reproduce all tables/metadata, JSON and LaTeX; computation preserves global RNG state and float32 caller defaults. Prespecified Monte Carlo checks independently construct future pooled-t, CP and Garwood interval widths rather than only comparing coefficients or formulas.

## Random discordance in paired binary designs

`power_mcnemar_unconditional(p10, p01, *, n=None, power=None, alpha=.05,
alternative="two-sided", direction=None, max_n=1000)` uses total independent
pairs. `p10=P(first=1,second=0)` and `p01=P(first=0,second=1)` must be
prespecified, nonnegative and sum to at most one. Exactly one of total `n`
and target `power` is provided; there is no detectable-effect inversion.

The conditional binomial rejection rule is unchanged. With
`q=p10+p01`, `D~Binomial(N,q)` and `B|D~Binomial(D,p10/q)`; power mixes
the inclusive, nonrandomized conditional rejection probability over every
possible D. D=0 never rejects. The upper alternative means p10>p01.
Achieved null size uses p10=p01=q/2. Zero discordance is evaluable with zero
power; its sample-size inversion is refused. This is unconditional **power
of the conditional test**, not a new unconditional rejection procedure.
[SAS POWER's primary PAIREDFREQ derivation](https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/statug_power_a0000000995.htm)
uses this mixture and an analogous mixture for achieved significance.

Minimum N scans every preceding integer, because discrete power can decrease.
The full evaluated search and conditional critical-value table survive summary
JSON. `N<=1000`, alpha in [1e-8,.5), and explicit tensor/work plans bound the
native CPU float64 computation; no pairs, draws or failures are dropped.
The existing fixed-discordance `power_mcnemar` remains a distinct contract.

## Enrollment, accrual and expected survival events

`survival_accrual(*, event_rate1, event_rate2, accrual, followup,
dropout_rate1=0., dropout_rate2=0., allocation=.5, n=None,
target_events=None, max_n=10_000_000)` declares a common time unit,
uniform entry over [0,A], administrative end A+F, and independent
per-arm exponential event and dropout clocks. Rates and times are nonnegative.

For q=lambda+kappa, each arm's observed-event probability is
`lambda/q * [1-exp(-q*F)*(-expm1(-q*A))/(q*A)]`. Continuous A=0 and q=0
limits are explicit; a Taylor expression preserves tiny positive probabilities.
The expectation follows the uniform-entry/loss calculation in the
[primary SAS SEQDESIGN reference](https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_seqdesign_details42.htm).

Allocation is the nominal arm-2 enrollment share, in [.01,.99]. Total N is
integer; the shortest decimal string of `allocation` defines an exact rational
share. Integer arithmetic sets `n2=floor(allocation*N)`, `n1=N-n2`, both
nonempty, so `.99,N=100` gives `1/99` and `.58,N=100` gives `42/58` without a
binary floating-point extra subject. The saved rational numerator/denominator,
achieved
allocation and both arm probabilities/counts are saved. Exactly one of n and
target expected events is supplied. The minimum enrollment search checks the
qualifying and immediately preceding admissible totals. N<=10 million.

This output describes an expectation. It supplies no logrank/Cox power,
probability of attaining the event target, or inferred risk-set allocation.
The Schoenfeld `power_logrank` API retains its separate assumptions. Numeric
arrays, full settings, scenarios and arm tables persist through summary JSON;
source oracles use nested numerical integration over entry and event time.
Neither method accepts fitted data, weights or GPU/Dataset input, and neither
claims licensed vendor execution, a public desktop release or blanket parity.
