# Prospective power and precision planning

`oe.power_mean`, `oe.power_proportion`, `oe.power_correlation` and
`oe.precision_mean` take **prespecified design assumptions**. They never fit a
dataset or compute retrospective power from an observed effect. Report input
effects, SDs and correlations as assumptions when presenting the calculations.

Each returns a `TableSet` with ordinary `plan`, `scenarios` and `settings`
tables. `scenarios` evaluates adjacent integer sample sizes, showing achieved
power or width. `settings` stores the complete method contract as JSON cells;
it survives console history and exports without relying on pandas attributes.
The plan includes achieved power/width, the requested target, group sizes and
the solved parameter. Display each table to save it as a structured result.

| Function | Supported design | Inversion |
| --- | --- | --- |
| `power_mean(effect, sd=..., n=..., power=...)` | One normal mean; known population SD | Exactly two of effect, n, power |
| `power_mean(..., sd2=..., ratio=...)` | Two independent normal means; both population SDs known | Same; n is group 1 |
| `power_proportion(p0, p1, n=..., power=...)` | iid Bernoulli; exact nonrandomized binomial rejection rule | Exactly two of p1, n, power |
| `power_correlation(rho0, rho, n=..., power=...)` | iid bivariate normal; approximate Fisher-z test | Exactly two of rho, n, power |
| `precision_mean(sd=..., n=..., width=...)` | Normal mean or independent difference with known SDs | Exactly one of n, total width |

## Normal means and precision

For one mean, standard error is `sd/sqrt(n)`. For independent groups, it is
`hypot(sd/sqrt(n), sd2/sqrt(n2))`. Allocation is explicitly
`n2=ceil(ratio*n)`, and the signed effect is `(mean2-mean1)-null_difference`
(or `mean-null` for one sample). These formulas assume the SDs are known
population parameters, independent normal observations and a fixed analysis.
A pilot estimate does not make an unknown population SD known.

The normal test rejects above `z_(1-alpha)` for an upper alternative, below
its negative for a lower alternative, and outside `+/-z_(1-alpha/2)` for a
two-sided alternative. Power integrates the shifted normal distribution over
that rejection region. `direction="lower"` selects the lower detectable-effect
branch of a two-sided test. One-sided direction must match its alternative.
Power for a supplied effect in the wrong direction is still reported honestly;
sample-size inversion for such an effect is refused.

Two-sided CI **total** width is `2*z_(1-alpha/2)*standard_error`, with
`alpha=1-confidence`. With known SD it is deterministic, so there is no
sampling-width assurance probability. This does not return an observed CI or
an interval centered on an invented sample mean.

Integer n inversion checks the chosen integer and the previous admissible
integer. It does not round a fractional allocation and then leave a target
unmet. Each group is limited by `max_n` (default and hard limit 10,000,000);
ratio is in [0.01,100]. There is no allocation of synthetic observations.
The search takes at most 50 scalar evaluations; effect inversion at most 66.
Input SD/effect/width magnitudes lie in [1e-100,1e100], with zero allowed for a
supplied effect. `alpha` is in [1e-8,0.5); target power must exceed alpha and
be at most 1-1e-12. Confidence is in (0.5,1-1e-8].
Detectable mean effects outside the reusable effect-input range are refused.

Examples from the official manuals: effect 25 with known SD 40 at two-sided
alpha .05 and target power .8 requires 21 observations; known SD .8 and total
95% CI width .5 requires 40 observations. SDs 5.5/5 and width 6 require 24
observations per independent group. These are printed-reference checks, not
execution of licensed Stata.

## Exact binomial proportions

For X~Binomial(n,p0), the lower critical count is the largest k for which
P0(X<=k)<=alpha/2, and the upper is the smallest k for which
P0(X>=k)<=alpha/2. A one-sided test assigns alpha to its active tail and
disables the other. Reject X<=lower or X>=upper; missing tails use -1/n+1.
This is the equal-tail rule based on twice the smaller tail, **not** the
probability-ordering rule used by some exact binomial test APIs.

The plan records these inclusive critical counts and actual null rejection
probability, which can be below alpha. Power is the sum of binomial masses
in this **fixed null rejection set**, evaluated at p1. Native CPU float64
Torch log-PMF tensors are normalized by log-sum-exp. No normal approximation,
randomized boundary, continuity correction or Monte Carlo is used.

Exact power can decrease when n increases. The n solver enumerates **every**
integer from 1 to the first n achieving the target. It guarantees the first
qualifying n, not that all larger n qualify. `max_n` defaults to 1000 and cannot
exceed 1000. The bounded search visits at most 1,003,000 PMF cells; workspaces
hold O(max_n) float64 values (a conservative 256 KiB bound). Requested n/budget
is checked before tensor allocation. No within-budget solution raises an error.

Detectable p1 holds the rejection region fixed and brackets the selected
branch from p0 to 0 or 1. A contiguous binomial acceptance interval has a
unimodal acceptance probability: since null power<=alpha<target, each branch
has at most one target crossing. Empty or unreachable rejection tails raise an
error. Inputs and returned p0/p1 must be strictly between 0 and 1. Endpoints
are used only to establish attainable limits during inversion.

## Correlation approximation

The assumed transformed statistic is approximately normal with mean
`atanh(rho)` and variance `1/(n-3)`. Its mean shift under the alternative is
`(atanh(rho)-atanh(rho0))*sqrt(n-3)`, used with the same normal rejection rule.
The method is explicitly labeled an approximation, including at its minimum
n=4; small samples, nonnormal pairs and correlations near +/-1 can depart
substantially from it. There is no exact finite-sample or bias-adjusted claim.

The n solver checks adjacent integer n>=4. Detectable rho inverts on the
Fisher scale, then transforms back. `abs(rho)<=1-1e-12` and n<=max_n<=10,000,000;
an unrepresentable or unattainable detectable alternative raises an error.

## Saved results, resources and boundaries

The public APIs do not accept observations, Dataset inputs, missing policies,
weights, clusters or a device override. There is no empirical covariance,
observed p-value, likelihood, fitted state or degrees of freedom to report.
Those fields are explicitly marked inapplicable in saved settings. Computation
uses native scalar distribution kernels and CPU float64 Torch binomial kernels,
without third-party runtime estimation. RNG state is untouched. Unsupported
arguments fail rather than invoking a different procedure.

The [example](../examples/prospective_planning.py) covers all inversion modes,
four complete saved contracts, adjacent-n scenarios and a user-selected
mean-power curve. Tables can be passed to existing chart tools; rendering a
scenario curve does not change its scientific assumptions.

The original four methods did not close MARKET-175. Additional bounded
designs and the remaining scope are described below; vendor parity remains
unclaimed. Source probability-law tests, frozen execution, actual
native Run/restart and public signed release are separate evidence layers.

## Primary references and validation

* [Stata power onemean](https://www.stata.com/manuals14/psspoweronemean.pdf),
  known-SD example 3 and normal power formulas.
* [Stata power twomeans](https://www.stata.com/manuals14/psspowertwomeans.pdf),
  known-SD difference variance and one/two-sided rejection regions.
* [Stata power oneproportion](https://www.stata.com/manuals13/psspoweroneproportion.pdf),
  final-page exact binomial rejection probability. Its exact option does not
  solve n or effects; our bounded integer/effect inversion is separately tested.
* [Stata power onecorrelation](https://www.stata.com/manuals15/psspoweronecorrelation.pdf),
  Fisher-z variance and prospective power formulas.
* [Stata ciwidth onemean](https://www.stata.com/manuals/pss-3ciwidthonemean.pdf)
  and [ciwidth twomeans](https://www.stata.com/manuals/pss-3ciwidthtwomeans.pdf),
  deterministic known-SD widths.

`tests/test_econ_planning.py` independently integrates a sample-mean density,
uses SciPy's normal/binomial laws, builds complete rejection sets, exhaustively
checks all preceding binomial sizes, verifies one-/two-sided and lower branches,
integer allocation, scale invariance, refusal/budget cases and saved metadata.
SciPy appears only in validation, never in these runtime procedures.

## Eight additional method contracts (MARKET-248–255)

These procedures also return three ordinary tables. Six power methods require
exactly two of the alternative effect, count and target power. One-sided
`alternative="upper"/"lower"` and two-sided detectable-effect `direction` use
the same normal or exact rejection rules above. Adjacent-count scenarios retain
the actual allocation and supported domain. Incorrect-direction count inversions
are refused; supplied-effect power remains valid even in the wrong direction.

| API | Count and prespecified assumptions | Law |
| --- | --- | --- |
| `oe.power_paired_mean(effect, sd_before=..., sd_after=..., correlation=..., n=..., power=...)` | n independent pairs, 2n measurements; effect=(after-before)-null | Exact known-covariance normal difference |
| `oe.power_two_proportions(p1, p2, n=..., power=..., ratio=...)` | Independent Bernoulli groups; n=group 1; n2=ceil(ratio*n) | Cohen arcsine-h **normal approximation** |
| `oe.power_two_correlations(rho1, rho2, n=..., power=..., ratio=...)` | Independent iid bivariate-normal groups, both n>=4 | Independent Fisher-z **normal approximation** |
| `oe.power_slope(effect, error_sd=..., design_variance=..., n=..., power=...)` | Fixed full-rank X, intercept+slope, iid normal errors, known error SD, n>=2 | Exact conditional normal slope |
| `oe.power_logrank(hazard_ratio, events=..., power=..., information_fraction=...)` | Constant prespecified group-2 fraction q in risk sets | Schoenfeld PH/local-alternative **normal approximation** |
| `oe.power_mcnemar(probability, discordant_pairs=..., power=...)` | Fixed discordant count m; conditional directional probability r | Exact conditional binomial McNemar |
| `oe.precision_mean_unknown(sd=..., width=..., n=..., confidence=..., assurance=...)` | Prespecified true population SD; sample SD estimated at analysis | Exact normal-sample Student-t CI width law |
| `oe.precision_variance(variance=..., width=..., n=..., confidence=..., assurance=...)` | Prespecified population variance, variance units | Exact normal-sample equal-tail chi-square CI width law |

Paired differences use the stable population SD
`hypot(sd_before-sd_after, sqrt(sd_before)*sqrt(sd_after)*sqrt(2*(1-rho)))`.
The SE divides this by sqrt(n). Pair dependence is accounted for; independent
groups or estimated-SD t tests have different laws. Perfect correlations are
refused; abs(rho)<=1-1e-12.

For proportions, `h=2*(asin(sqrt(p2))-asin(sqrt(p1)))` and SE=sqrt(1/n+1/n2).
This is a named variance-stabilized planning approximation, **not** Pearson,
score, Wald or exact two-sample test power. Both probabilities are in [.01,.99]
and each group must have at least ten expected successes and ten failures.
The returned n is the first qualifying integer **in that domain**. Detectable
p2 requires the null p1 to satisfy that domain in both groups too; the selected
branch stops at the expected-count/probability boundary. A boundary tolerance
of 1e-12 expected counts handles floating representation of decimal assumptions.

For correlations the standardized shift is
`(atanh(rho2)-atanh(rho1))/sqrt(1/(n-3)+1/(n2-3))`. The groups must not share
observations. Small samples or near-boundary correlations can make the
approximation poor. Inversion verifies power again after transforming to rho;
loss of representability is refused.

Slope SE is `error_sd/sqrt(n*design_variance)`, where
`design_variance=sum((x-xbar)^2)/n`. The entire prospective sequence must
maintain this prespecified centered Gram per observation. It is not the
unbiased sample variance with divisor n-1, population R-squared, random-X
regression planning or unknown-error-SD t/F power.

Log-rank standardized shift is `log(HR)*sqrt(events*q*(1-q))`. The risk-set
fraction q in [.05,.95] is assumed approximately constant and events distinct;
proportional hazards/local-alternative information is an approximation.
At least ten expected information events per group are required. The method
integrates both normal tails, rather than replacing two-sided power with a
single-tail quantile shortcut. `event_fraction` optionally returns
`ceil(events/event_fraction)` **expected** enrollment. No censoring, accrual,
follow-up or guaranteed event count is inferred. Fractions may be in (0,1];
projected counts above one billion are refused before results are returned.

For conditional McNemar, given fixed m=b+c, b~Binomial(m,r), with null r=.5.
The exact equal-tail critical counts, actual null size and alternative mass
are stored. Every smaller discordant count is examined because exact power
need not be monotone. max_discordant_pairs<=1000. Optional
`discordance_fraction` returns `ceil(m/fraction)` expected total pairs only.
It does not integrate over a random discordant count or guarantee that many
pairs will be available; **unconditional McNemar power is outside this contract**.

### Random CI widths and assurance

The two new precision APIs require exactly one of n or total width. Confidence
is interval coverage; assurance is the separate probability that a future
interval is no wider than the requested width. Neither API creates an observed
CI or treats an estimated SD as known. Under iid normal sampling,
`U=(n-1)*S²/sigma² ~ chi²(n-1)`.

For a mean, random total width is `2*tcrit*S/sqrt(n)`. Its assurance quantile
is `2*tcrit*sigma*sqrt(chi²_quantile(assurance,df)/(n*df))`, with df=n-1.
For a variance the actual interval is `[df*S²/q_high, df*S²/q_low]`, hence
width=`sigma²*U*(1/q_low-1/q_high)` with a positive reciprocal difference.
Its assurance quantile is `sigma²*chi²_quantile(assurance,df)*(1/q_low-1/q_high)`.
These probability laws are derived from the displayed interval bounds;
PDF text extraction can reverse reciprocal signs or misread inequalities.

Both solvers exhaustively inspect n=2 through the first satisfying integer;
no monotonicity assumption is needed. max_n<=10,000 bounds native quantile
work and the invocation-local cache to 9,999 scalar pairs. Confidence is in
(.5,1-1e-8], assurance in [1e-6,1-1e-6]. A count solve reports probability at
the requested width; a width solve reports probability at the returned width.
Stable log-scale CDF arguments avoid squaring extreme measurement scales.

### Sources and validation for the additional methods

* [Paired means, known SD](https://www.stata.com/manuals15/psspowerpairedmeans.pdf).
* [Cohen h and unequal two-proportion normal designs, author's pwr manual](https://www.stat.ethz.ch/CRAN/web/packages/pwr/refman/pwr.html).
* [Independent Fisher correlations](https://www.stata.com/manuals14/psspowertwocorrelations.pdf).
* Fixed-X slope follows the conditional Gaussian OLS law; validation generates
  actual regression errors and independently computes slopes from the design.
* [Schoenfeld log-rank information](https://www.stata.com/manuals14/psspowerlogrank.pdf).
* [Conditional exact McNemar binomial rule](https://www.stata.com/manuals/repitab.pdf), Methods and formulas, matched data.
* [Unknown-SD mean width](https://www.stata.com/manuals/pss-3ciwidthonemean.pdf).
* [Variance interval and width law](https://www.stata.com/manuals/pss-3ciwidthonevariance.pdf).

`tests/test_econ_planning_extended.py` uses independent SciPy distribution
oracles, complete binomial rejection sets, exhaustive preceding integers,
normal-density integration, fixed-X regression sampling and actual normal
interval coverage/width simulation. No runtime SciPy dependency is added.
The two runnable examples `eight_prospective_designs_1.py` and
`eight_prospective_designs_2.py` each display four complete contracts (12 tables),
within the console's 20-output limit; both exercise every solve mode.

The independent t/F, balanced ANOVA, fixed regression F, equal-cluster ICC and
Pearson planning stages are documented separately in [power-designs.md](power-designs.md).
Additional Welch, unequal-allocation/cluster, unconditional paired-proportion,
confidence-width and accrual stages are documented in
[prospective-extensions.md](prospective-extensions.md). MARKET-175 remains open
for sequential stopping, estimated covariance and broader unimplemented options. These eight bounded child issues do not establish whole-family vendor
parity or deliver a public signed release.
