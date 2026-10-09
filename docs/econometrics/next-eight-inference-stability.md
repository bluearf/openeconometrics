# Saved inference and conditional OLS stability

Three native CPU float64 helpers complete bounded inference/stability stages
of #36, #38, #48 and #53. They use different reference laws. Implementation,
independent numerical verification, licensed vendor equivalence and a new
desktop release are separate evidence claims.

## Scalar mixed Satterthwaite

`oe.mixed_satterthwaite(saved, data=original, contrast={"x": 1})` accepts
numeric unweighted Gaussian `mixed` ML/REML results with model-based
`nonrobust` covariance. Original full physical model-input rows and their
sample positions/hash must agree. An arbitrary named scalar fixed-effect
contrast is supported, including a prespecified linear mean. No optimizer or
coefficient refit runs.

At the saved variance coordinates theta, set
`C(theta) = (X' V(theta)^-1 X)^-1`, `v = c' C c`, and `g = d v/d theta`.
The inverse negative Hessian of the profiled ML/REML likelihood gives the
**full** variance-coordinate covariance `A`; fixed coefficients are profiled
out. The reported approximation is
`df = 2 v^2 / (g' A g)`, `t = (c' beta - null)/sqrt(v)` with two-sided
Student t p-values and intervals. Native derivative extrapolation and the
full off-diagonal information are retained. Stored complete fixed covariance,
fixed estimates/SE and variance-coordinate SE are checked against the saved
likelihood before output.

This is the scalar Satterthwaite construction described by
[Kuznetsova et al. (2017)](https://www.jstatsoft.org/article/view/v082i13)
and the [author implementation](https://github.com/runehaubo/lmerTestR/blob/master/R/contest.R).
It assumes Gaussian independent top-level groups, correctly specified random
covariance and interior information. It does not establish general few-cluster
coverage, Kenward–Roger covariance adjustment or joint multi-df inference.
It excludes `mixedflex`, robust/cluster covariance, weights and categories.
Those remain follow-up stages alongside repeated residual structures and
broader GLMM/nonlinear mixed families.

Admission precedes design/information allocations: <=8192 original/effective
rows, <=32 fixed and variance parameters, <=8 random columns, <=64 reported
parameters. Default scalar-work ceiling is 200 million; a named live-buffer
plan also honors the global/task workspace budget. The plan excludes caller
objects and private allocator workspace, so it is not a process RSS limit.
Complete output includes the contrast, theta gradient, full theta covariance,
original sample identity and assumptions.

Independent tests recover `G-1` between-group mean df and
`G(T-1)-1` within-slope df in balanced orthogonal REML examples. These
closed-form laws apply only to those geometries. A separate unbalanced
correlated-slope dense NumPy GLS/profile-Hessian oracle checks the general
ML/REML gradient, information, variance and approximate inference.

## Saved joint event/horizon bands

`oe.trajectory_bands(saved, assumptions="...", terms=[...])` supports saved
`lp`, `lpiv`, `panel_lp`, `heterodid`, `eventstudy` and `csdid` results.
It validates the complete saved finite symmetric PSD covariance and SE
diagonal **before** selecting targets. Explicit row maps instead use
`contrasts=A, labels=[...]`, reporting `A beta` and `A V A'`. A family must
be chosen before examining effects; supply the CLT and identification
assumptions appropriate to that source estimator.

The shared `simultaneous_ci` engine simulates a joint mean-zero normal
vector with the full reported correlation and chooses the empirical higher
`1-alpha` quantile of `max |z|`. Each interval is
`estimate +/- critical * SE`. Unique labels canonically order the local seeded
simulation. This is **normal-limit** inference, not a known common-scale
joint t pivot. Marginal cluster t reporting does not establish finite joint
coverage. Strong-IV/valid-instrument assumptions are necessary for LP-IV;
few-cluster, weak-IV, post-selection and pretrend-sensitive procedures are
outside this helper. No estimator refits and no duplicated bootstrap engine.

Bounds: <=384 source/reporting targets, 1000..200000 draws, <=8 million draw
cells, full source and live simulation workspace plans. Complete JSON retains
source spec/id/hash/inference, map, full cross-target covariance, critical
value, draws, seed, Monte Carlo CDF error and assumptions. This supplements
the existing [modern DID methods](modern-did.md), without treating switching,
multivalued, repeated-cross-section or DDD designs as absorbing binary panels.

## OLS residual CUSUM and CUSUMSQ

`oe.ols_cusum(data, "y", ["x"], time="t", replications=999, seed=1729)`
conditions on a **prespecified fixed exogenous design** and iid Gaussian
errors of common variance. With thin QR `X=QR` and OLS residuals
`e=(I-QQ')y`, the two paths are
`sum_{j<=t} e_j / sqrt(sum e_j^2)` and
`sum_{j<=t} e_j^2 / sum e_j^2 - t/N`.
Each test statistic is its own whole-path absolute maximum.

Every Gaussian draw is projected on exactly the complete design, uses its
own RSS and enters the denominator. No failed draws are discarded.
For B draws, `p=(1+count(T_draw>=T_observed))/(B+1)`; the higher empirical
`1-alpha` quantile is also reported. Conditional on fixed X, these residual
paths have a parameter-free Gaussian null law. Monte Carlo resolution is
explicit. The paths are controlled separately; the two tests are not a
single simultaneous family.

[Stata's OLS CUSUM manual](https://www.stata.com/manuals/tsestatsbcusum.pdf)
and the [statsmodels OLS-residual reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html)
document related OLS-residual diagnostics. The finite conditional simulation
here is a deliberately declared calibration; it is not a claim of matching
vendor asymptotic Brownian-bridge critical values or recursive-residual
CUSUM/CUSUMSQ bounds. Existing `oe.cusum` retains its recursive convention.
HAC/robust, sequential/subset break inference and remaining sourced critical
value conventions stay in follow-up work. Existing bounded Bai–Perron and
advanced unit-root stages keep their own documented reference domains.

<=2048 supplied rows and <=32 requested regressors including the intercept
are admitted before numeric model allocations; 99..9999 draws run in batches
of 32. Default work is bounded to 100 million, with a named workspace plan.
Missing/nonfinite values, time gaps/repeated periods and perfect fits raise
explicit errors. Collinear terms are omitted with notes. Complete sorted
period labels, input hash, every null maximum and settings survive full file
JSON restoration without fitting or simulation. The NumPy oracle checks
every projected draw, both whole paths, cutoffs and complete-denominator p.

## Reproducible example and proof boundaries

Run `docs/examples/next_eight_inference_stability.py OUTPUT_DIRECTORY` for
actual fitted-source summaries and complete file readback. Tests are
`test_next_eight_mixed_satterthwaite.py`, `test_next_eight_trajectory_bands.py`
and `test_next_eight_ols_stability.py`. Dated receipts distinguish source
execution with external estimator imports blocked from installed-wheel
execution. Historical frozen artifacts prove only their pinned unchanged
method bytes; these additions do not create a new native installer or
validate licensed vendor parity.
