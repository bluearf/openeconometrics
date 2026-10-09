# Eight prospective power designs

`oe.power_tmean`, `power_ttwomeans`, `power_tpaired`, `power_anova`,
`power_regression`, `power_cluster_mean`, `power_gof` and `power_independence`
accept prespecified population/design assumptions. They consume no observed
sample and compute no retrospective observed power. The original four APIs
in [planning.md](planning.md) remain available.

Specify exactly two of `effect`, `n`, `power` for the first six methods. For the
Pearson methods the trio is `strength`, `n`, `power`; default strength is 1,
and `strength=None` with n/power finds the detectable convex departure.
The result is a TableSet containing `plan`, adjacent integer `scenarios` and
complete JSON `settings`; Pearson methods also contain `cells`. `to_latex()`
exports every table. Saved inputs/settings can reproduce the whole calculation.

| API | Design and n unit | Probability law |
| --- | --- | --- |
| `power_tmean(effect, sd=...)` | iid normal observations, n observations | Student t, df=n-1, delta=effect/(sd/sqrt(n)) |
| `power_ttwomeans(effect, sd=..., ratio=...)` | independent equal-variance normal groups, n group-1 observations, n2=ceil(ratio*n) | pooled t, df=n+n2-2, delta=effect/(sd*sqrt(1/n+1/n2)) |
| `power_tpaired(effect, sd1=..., sd2=..., correlation=...)` | independent complete normal pairs, n pairs | t on differences, df=n-1; difference variance=sd1²+sd2²-2*rho*sd1*sd2 |
| `power_anova(effect, groups=...)` | balanced fixed normal means, common variance; n per group | noncentral F(groups-1, groups*(n-1)), lambda=groups*n*f²; effect is Cohen f |
| `power_regression(effect, predictors=..., tested=...)` | full-rank fixed Gaussian design with intercept; n total, p full-model slopes, q tested restrictions | noncentral F(q,n-p-1), lambda=n*f2; effect is prespecified conditional signal/noise normalized per observation |
| `power_cluster_mean(effect, sd=..., cluster_size=..., icc=..., ratio=...)` | two arms, equal cluster sizes m, common marginal SD/ICC, independent clusters; n arm-1 clusters | **normal design-effect approximation**, D=1+(m-1)*ICC, SE=sd*sqrt(D/m*(1/n+1/n2)); exact only with known-covariance Gaussian cluster means |
| `power_gof(null_probabilities, proposed_probabilities, ...)` | iid multinomial, fixed null probabilities; n total | **asymptotic** noncentral chi-square, df=cells-1, lambda=n*sum((planned-null)²/null) |
| `power_independence(joint_probabilities, ...)` | iid multinomial joint matrix; implied null is row margin × column margin; n total | **asymptotic** noncentral chi-square, df=(rows-1)*(columns-1), lambda as above |

The normal/t alternatives are `two-sided`, `upper`, `lower`; signed effect means
alternative-minus-null (two-group effect is mean2-mean1 minus null difference).
`direction` selects the two-sided MDE branch and must agree with a one-sided
alternative. ANOVA, regression and Pearson tests always reject in the upper
F/chi-square tail; their nonnegative effect has no signed branch. Paired t
uses a stable difference-SD calculation and rejects zero variance.

Regression's exact statement is conditional on the fixed design and specified
noncentrality. Planning over n assumes its signal norm scales as n*f2; random-X
R-squared distributions, robust/cluster errors and arbitrary changing designs
are outside this contract. Prespecified SD in t planning does **not** mean the
analysis treats SD as known: the probability integrates the independent sample
variance, unlike the existing normal z planning.

Pearson `planned=null+strength*(proposed-null)` uses 0<=strength<=1 without
extrapolation. In an independence table this path preserves both margins.
Null probabilities must be positive; joint matrices also reject structural
zeros. Probabilities must sum to one within 1e-12 and are never normalized or
merged. Every null expected count must reach 5. This heuristic gate does not
make the noncentral chi-square approximation finite-sample exact.

## Numerical and resource contract

Only bounded CPU float64 resident scalar/short probability-list calculations
are supported. DataFrames, Dataset, missing values, strings, boolean counts,
weights and device options are refused. There is no GPU or streaming route.
Alpha is [1e-8,.5), target power exceeds alpha and is <=1-1e-12. Each group/n
budget is <=20,000; ANOVA's total budget is 20,000, groups 2..30; predictors
1..100; ratio .05..20, at least two observations/clusters in either arm.
Cluster size is 1..10,000, ICC 0..1. Probability vectors/matrices have <=100
cells. Scalars have magnitude [1e-100,1e100] or zero where meaningful.

Noncentral t uses conditional-normal integration over the chi density, with
stable centered log density, 0.5-wide panels and explicit transition splits.
16/32 Gauss-Legendre rules must agree within 2e-11 absolute error. Independent
tail calls avoid subtracting a near-one CDF. F and chi-square use centered
Poisson weights and central beta/gamma tails; omitted Poisson mass is checked
<=2e-14 before normalization. Maximum noncentrality is 4096 (|t delta|<=64)
and df<=40,000. This gives fixed small work buffers, rather than an n-sized
sample or model matrix. Unsupported probabilities fail explicitly; no normal
fallback replaces an exact t/F result.

Integer n solves use an exponential bracket then bisection and check both n
and n-1. The laws are monotone along these declared allocation paths. MDE
uses a bracketed upper crossing and verifies achieved power within 1e-9.
Nonattainable targets and exceeded work budgets have explicit errors. An
adjacent scenario that exceeds the probability budget is omitted **with its
n/error recorded in `omitted_adjacent_scenarios`**, while the valid plan is
preserved. The example displays 16 tables and saves all 26 tables/settings.
No fitted covariance/SE/p-value/likelihood or convergence claim applies to a
prospective calculation; reported design SE, df and noncentrality are inputs
to the probability law, not estimates from observations.

## References and evidence

The [Stata onemean manual](https://www.stata.com/manuals15/psspoweronemean.pdf)
gives the t planning formula and examples reproduced here: effect 25, SD 40,
80% power gives n=23, and n=30 gives displayed power .9112. The
[oneway manual](https://www.stata.com/manuals15/psspoweroneway.pdf) example
means 260/289/295 with variance 4900 gives 69 per group. The
[rsquared manual](https://www.stata.com/manuals15/psspowerrsquared.pdf)
example f2=.1/.9 with five tested predictors gives n=122. These are numerical
agreements with published displayed facts, **not licensed Stata execution**.
The [SAS Pearson independence note](https://support.sas.com/kb/25/013.html)
describes the noncentral chi-square approximation; the
[J-PAL power guide](https://www.povertyactionlab.org/resource/power-calculations)
describes equal-cluster ICC design effects. Source implementation, independent
SciPy test oracles, simulation checks, frozen runtime and installed native
Run/restart receipts are recorded separately in the validation evidence.

Welch/unequal-variance and unequal-cluster designs have separate explicit
contracts in [prospective-extensions.md](prospective-extensions.md), which also
adds survival accrual expectations. Sequential/stopping rules, estimated ICC
and broader design options remain outside these methods. The broad MARKET-175/GitHub #62 parent stays open.
