# Single-stage coefficient replication

`survey_regress_replicate`, `survey_logit_replicate`, `survey_probit_replicate`
and `survey_poisson_replicate` provide sixteen family/method gates: numeric
linear/logit/probit/Poisson coefficient fits crossed with BRR, Fay, stratified
PSU jackknife and supplied design-bootstrap variance. They reuse the existing
weighted score/observed-curvature solvers and whole-PSU replicate planner.
They return `SurveyRegressionResult`, including complete coefficient covariance
and method-aware saved replay. Existing descriptive replicas and Taylor
regression are separate accepted procedures.

Pass the original complete resident table, unchanged `SurveyDesign`, outcome,
numeric regressors and explicit `method`. Domain and declared listwise missing
exclusions retain original physical positions and the full design universe.
Original and every replica use the same coefficient order. Zero replica weights
exclude those rows from that fit, while its original PSU/stratum declaration
remains in the plan. Full-rank finite converged estimates are required in every
fit; separation, unidentified coefficients and nonconvergence fail the request
with all failed replica IDs. No replica is deleted, penalized or repaired.

For estimates `b_r` and declared centers `c_r`, full covariance is
`V = sum_r multiplier_r (b_r - c_r)(b_r - c_r)'`. Every term, including
off-diagonal covariance, is retained. Centers are the original estimate,
replicate mean, or the jackknife mean within each deleted-PSU stratum.

| Method | Geometry and multiplier |
| --- | --- |
| BRR | Two noncensus PSUs per stratum; generated Sylvester signs or supplied verified balanced signs; `1/R` |
| Fay | Same balanced pairs; weight factors `rho` and `2-rho`; `1/[R(1-rho)^2]`; default `rho=0.5` |
| Jackknife | Delete each noncensus PSU once, expand surviving weights in its stratum by `m_h/(m_h-1)`; multiplier `(1-f_h)(m_h-1)/m_h` |
| Bootstrap | Supplied nonnegative PSU-constant factors with explicit design provenance and `scale*rscales[r]`; no row resampling |

BRR/Fay reject noncensus partial FPC. Census PSU weights remain unchanged;
all-census BRR/Fay/jackknife requests are refused; callers may use the
existing zero-variance Taylor API instead. Jackknife supports declared first-stage population counts.
Bootstrap provenance is caller supplied and is not authenticated. Calibrated
unit-varying replica factors and multistage plans remain unsupported.

Coefficient inference uses the stored full covariance and design-t degrees of
freedom `min(complete_design_df, R-1)`. Only supplied bootstrap may declare an
explicit positive df below `R`. `survey_predict`, `survey_margins`,
`survey_lincom` and `survey_test` restore complete saved state; they propagate
coefficient covariance through their existing fixed-covariate Jacobians.
These are conditional coefficient-delta targets, not full nonlinear replication
of margins, empirical-distribution uncertainty or future-outcome intervals.

JSON state retains original coefficients, sensitivity/PSU scores, admitted
sample/design, every replica ID/estimate/convergence, weight-plan fingerprint,
scale/rscales/rho, centering and df convention. Restore checks complete geometry
and independently replays the full covariance from saved coefficient replicas.
Legacy Taylor states keep their original schema and digest semantics. Digests
detect accidental changes; they do not authenticate supplied provenance.

Admission bounds are resident CPU float64, at most 32 coefficients, 4096
replicas, eight million full replica-weight cells and 50 million cumulative
fit/separation/replay work units, with declared memory/workspace limits. Full
refit planning precedes replica-weight allocation. Dataset/device input,
multistage/calibration, general design effects and
licensed vendor execution remain open.

[Stata variance-estimation formulas](https://www.stata.com/manuals/svyvarianceestimation.pdf)
describe coefficient estimating equations, BRR/Fay and stratified jackknife;
[Statistics Canada's supplied survey bootstrap workflow](https://www150.statcan.gc.ca/n1/pub/12-002-x/2014001/article/11901-eng.htm)
documents explicit bootstrap-weight scaling. These primary references guide
the formulas and scope; reading them is not licensed Stata execution evidence.
Independent weighted solves, binary logit/probit likelihood scores, Poisson
weighted exponential scores and enumerated coefficient replica covariance are
development-only checks. The
[runnable eight-stage example](../examples/survey_regression_replication_eight.py)
and the [probit/Poisson eight-stage example](../examples/survey_probit_poisson_replication_eight.py)
also produce saved conditional postestimation. Frozen/native Run and full
quit/reopen persistence are recorded separately in the acceptance evidence.

`stata_parity_validated` remains false. MARKET-449–456 are bounded implementation
gates under [MARKET-212](https://linear.app/bluearf/issue/MARKET-212);
MARKET-497–504 add the eight probit/Poisson gates without recounting prior work.

Probit admits exact binary 0/1 outcomes with observed likelihood curvature.
Poisson admits finite nonnegative integer counts at most `2**53`, including
zero counts, with a finite identified mean. Every original/replicate fit must
pass the native separation certificate and strict rank/convergence gates.
Removing the only opposite binary response or positive-count support may
therefore refuse a valid original model at the replica stage. Fay positive
factors preserve support but do not waive numerical convergence.

[Full empirical target replication](survey-replicate-margins.md) adds joint partial-profile predictive means and continuous AMEs for logit/probit/Poisson, with refitting and empirical reweighting inside every replica.
