# Single-stage survey regression and saved inference

Eight procedures extend the declared single-stage survey design:
`survey_regress`, `survey_logit`, `survey_probit`, `survey_poisson`,
`survey_predict`, `survey_margins`, `survey_lincom` and `survey_test`.
They use resident CPU float64 and explicit numeric columns. The
[runnable example](../examples/survey_regression_eight.py) fits all four models,
restores their complete states and produces four saved-result procedures.

## Model and design contract

Pass the original complete resident data and unchanged `SurveyDesign`; its
weights, PSU, strata and optional population-PSU FPC counts are revalidated.
Every equation uses one common listwise outcome/regressor sample in its domain.
Out-of-domain and incomplete records have zero scores, while every original PSU
and stratum remains in the covariance geometry and `df = PSUs - strata`.
There is no ordinary weighted-regression covariance substitution or HC1 factor.

Linear estimates solve sampling-weighted least squares. Logit/probit require
exact binary responses; Poisson requires exact nonnegative integer counts at
most `2**53`, checked before integer-to-float rounding. The latter is a count
pseudo-likelihood, without a broader real-response PPML claim. Weights are
normalized to mean one over fitting records. A common positive weight scaling
leaves coefficients and complete design covariance unchanged.

Write `s_i` for an unweighted row score and `A` for the negative weighted score
Jacobian. Complete PSU totals are `S_hp = sum_i w_i I_i s_i`. The covariance is

```text
V = A^-1 [sum_h (1-f_h) m_h/(m_h-1)
                 sum_p (S_hp-Sbar_h)(S_hp-Sbar_h)'] A^-T.
```

Logit and Poisson use their exact canonical sensitivity. Probit uses the
**observed** negative Hessian, rather than expected Fisher information.
Reference formulas: [survey variance manual](https://www.stata.com/manuals/svyvarianceestimation.pdf)
and [survey estimation manual](https://www.stata.com/manuals/svysvyestimation.pdf).
Manual/formula agreement is separate from running a licensed vendor executable.
`stata_parity_validated` remains false.

The weighted design and sensitivity must be identified and full rank. Native
primal/dual replay checks reject complete/quasi binary separation and Poisson
zero-count separating directions. Unsupported precision, overflow, an exhausted
diagnostic budget or failed Newton/step-halving convergence returns an error.
No ridge, pseudoinverse or incomplete fit conceals an identification failure.
Initial optima may require zero Newton updates; score evaluations and actual
updates are recorded separately.

## Saved state and postestimation

`SurveyRegressionResult` saves exact parameter order, coefficients, full
covariance, design, physical sample positions, exclusions, weight metadata,
convergence and complete PSU score/sensitivity records. JSON restoration replays
the covariance and checks an integrity digest. A digest is not authentication of
sampling or a substitute for the original design declaration.
`result.to_frame()` restores coefficient design-t inference and exportable LaTeX
without original observations or refitting. Census covariance may be zero;
zero-SE tests and zero-design-df tests remain undefined, with point intervals.

`survey_predict(result, data, kind="response")` returns conditional fitted means
with their complete joint coefficient-delta covariance. `kind="linear"` returns
the link index. New data needs the fitted numeric regressor columns, without an
outcome or original design. Missing exclusions retain original physical positions
and the input index. These are mean intervals, without future-outcome noise.

`survey_margins` averages response predictions if `variables=None`, or returns
continuous average marginal effects for the named fitted numeric regressors.
Optional `at` values fix covariates and an explicit `weights` column supplies
positive fixed standardization weights. The full delta covariance uses analytic
coefficient Jacobians. Evaluation covariates/distribution are fixed; empirical
sample-distribution uncertainty, unconditional margins and categorical/discrete
contrasts are separate unsupported targets. Wald probability intervals are
unclipped. See [margins delta/unconditional distinctions](https://www.stata.com/manuals/rmargins.pdf).

`survey_lincom` accepts one ordered coefficient vector or named mapping and
uses the complete saved covariance for design-t/CI inference.
`survey_test` accepts independent full-rank restrictions `R beta = c`. With
`q` restrictions and `nu` design df, its adjusted Wald statistic is
`F = (nu-q+1) W/(q nu)`, compared with `F(q, nu-q+1)`.
One restriction equals squared design t; `q > nu` and singular requested
covariance are refused. Requested restrictions are normalized by their row
scales for stable evaluation; the equivalent scale and covariance convention
are recorded. See the [official joint-test formula](https://www.stata.com/manuals/rtest.pdf).

## Explicit limits

At most 31 regressors plus an intercept, 32 coefficients and more complete
records than coefficients. Fits and covariance replay have a 50-million-unit
work budget and conservative resident allocation plans. The reserved 64-pass
separation diagnostic is included in the same cumulative work bound;
all four families admit at most **32 coefficients** subject to that bound.
Declared Newton iterations must be in `1..1000`; tolerance in `1e-14..1e-3`.
Every likelihood/Hessian evaluation, including rejected line-search proposals,
consumes the work budget. Full prediction covariance is limited to 256 input
rows. Margins has the declared resident row/memory and `N*K*K` work limits.

Only a validated single-stage design is supported. Multistage nesting and stage
weights/FPC, calibration/poststratification,
general regression design effects, additional GLMs, Dataset/device routes and
licensed vendor acceptance remain separate open MARKET-211 gates.
Probit/Poisson coefficient replication is covered by MARKET-497–504 under
MARKET-212; see [replication contracts](survey-regression-replication.md).

Full empirical-distribution uncertainty for bounded nonlinear partial-profile means and continuous AMEs is provided separately by [survey_margins_replicate](survey-replicate-margins.md).
