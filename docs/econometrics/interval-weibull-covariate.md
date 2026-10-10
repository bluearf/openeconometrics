# Covariate interval-censored Weibull regression

MARKET-689 adds one bounded iid Weibull regression family. It preserves the
existing intercept-only `stinterval_weibull`, ordinary exact/right `streg`, Cox,
and competing-risk methods. MARKET-163 remains a broader survival program.

`stinterval_weibull_regression(data, lower="lower", upper="upper", x=["x1","x2"],
parameterization="aft")` fits the original-time likelihood. The alternative
`parameterization="ph"` selects the table from a fixed transformation of the
same fit. It never selects another objective or discards scale uncertainty.
The result is a typed `WeibullIntervalFit`, with `.to_tables()`, `.table()`,
`.latex()`, `.dataset()`, `.model_dump()`, and `.model_dump_json()`.

## Model and likelihood

In the AFT chart, `log T = X beta + sigma epsilon`, where the minimum extreme
value error has survival `exp(-exp(epsilon))`. Include an intercept in `X`;
the last raw coordinate is `ell=log(sigma)`. Set `p=exp(-ell)` and
`z(t,x)=p(log(t)-X beta)`. Then cumulative hazard is `H=exp(z)` and survival
is `S=exp(-H)`. An exact observation contributes
`-ell-log(t)+z-exp(z)`; left censoring contributes `log(1-exp(-H(upper)))`;
right censoring contributes `-H(lower)`; and an interval contributes
`-H(lower)+log(1-exp(-(H(upper)-H(lower))))`.

Use equal positive endpoints for an exact density, zero lower for left
censoring, finite increasing endpoints for `(lower,upper]`, and explicit
`None` or positive infinity upper for right censoring. Original null/infinity
coding is retained separately, and `.dataset()` reconstructs it. Upper `NaN`
or `pd.NA` is ambiguous missing data and is refused. Zero exact times, negative
or inverted endpoints, and `(0,infinity)` are refused. No midpoint conversion,
row deletion, or censoring recode occurs. The exact density includes the
original-time Jacobian; likelihood values from different time units therefore
change by the exact-observation count times the log unit conversion.

The PH chart is `gamma=-p beta`, `log_shape=-ell`, so
`H(t,x)=exp(X gamma) t^p`. Its full Jacobian includes every coefficient–scale
cross derivative. Inference tables deliberately name the final coordinate
`log_sigma` or `log_shape`; displayed natural `sigma`/`shape` are transformed
summaries, rather than additional estimated parameters. All coefficient
tables retain the complete covariance, including shape coupling.

## Information, optimization and queries

The calculation centers/scales every covariate and centers log time. This is
a reversible computational chart. The saved state retains its exact source,
chart, source positions, original index, normalized coefficients, original
AFT score/full observed information, inverse information, and full AFT/PH
covariance. Covariance and queries use the normalized chart to avoid squaring
unit scales or cancelling large original-unit cross terms.

Three deterministic starting points use the representative-log-time QR fit
and scale multipliers 0.7,1,1.5. BFGS with an exact observed Hessian and checked
Newton polishing selects the largest accepted stationary likelihood. The
method does not certify a global optimum for every censoring design. A saved
endpoint must lie strictly inside `|normalized beta_j|<40` and
`|log_sigma|<log(1000)`, have positive definite full OIM with equilibrated
eigenvalue ratio above `1e-8`, and satisfy
`score' covariance score <=1e-12` and `max|score| <=1e-7*n`.
Design singular-value ratio must be at least `1e-6` after normalization.
Identification or convergence failures yield typed refusal, without ridge,
pseudoinverse, dropping terms, or substituting a boundary fit.

`interval_weibull_regression_predict(fit, profiles, times=[0,.5,1.5,3])`
returns a typed `WeibullIntervalPrediction`. It retains the complete fit and
complete query source, values, Jacobians, every cross-profile/time survival
and cumulative-hazard covariance, all survival–hazard cross blocks, and the
original parameter–query cross covariance. The ordering is all survival
row/time components, followed by all cumulative-hazard row/time components.
At time zero, survival 1 and hazard 0 have exactly zero uncertainty. Positive
times use full log-cumulative-hazard delta variance for pointwise positive
hazard and `[0,1]` survival confidence limits. These are asymptotic pointwise
intervals, without simultaneous-band or predictive-event claims. Finite-time
parametric extrapolation is conditional on the specified Weibull model;
infinite times and nonrepresentable values/limits are refused.

## Source, state and resource domain

The source is a resident DataFrame or aligned Series/list/tuple mapping,
with at most 4096 complete iid single-event rows and eight numeric covariates.
All source rows, numeric widths, endpoint coding, typed integer positions,
duplicate labels, and supported complete indices remain aligned. Range,
ordinary scalar/string, datetime/timezone, timedelta, categorical, and bounded
multi-indices are reconstructed and canonically re-encoded before tensors.
Boolean, complex, object covariates, missing numeric cells, integers outside
exact float64 representation, and numeric widths above 64 bits are refused.
Finite covariates are bounded by `1e140`; positive times lie in
`[1e-12,1e12]`. Small declared float widths must contain exactly representable
saved values. AFT/PH raw coordinates and all cache cells are checked in their
declared units, with no dimensionful absolute comparison floor; structural
zeros cannot absorb subnormal forgeries.

`maxiter` is 1..2000 per start. The explicit complete estimated work cap uses
`3*maxiter*n*q^2 <= max_work`, default 500 million, where `q=len(x)+2`.
Dense source/chart/information/query buffers, resident metadata copies,
escaped JSON and parser envelopes are charged before their allocations.
Queries retain at most 256 total survival/hazard components, with no thinning.
Encoded state is bounded by 32 MiB and index metadata by 1 MiB. Pretty-print
indent is 0..16 and is admitted before serialization. These are implementation
capacity boundaries, not statistical sample-size claims.

Restore, tables, partial/excluded exports, ordinary/deep copy and typed JSON
all replay complete semantics. They reconstruct source geometry and evaluate
likelihood/score/OIM at the saved endpoint; they never run the fit entrypoint,
BFGS, Newton iteration, substitute another estimate, or consume random draws.
Complete cached shapes, source/query primitives, and budgets are admitted
before expensive target/information work. Bounded optimizer counters are
recorded provenance; replay does not certify an unrecorded historical path.
The checksum detects corruption and does not authenticate a user's source.
Model constructors/copies cannot bypass these checks at public readers.

This family assumes correctly specified Weibull conditional failure times,
independent iid subjects, and noninformative observation intervals/censoring
conditional on fixed covariates. It supports no frailty, delayed entry,
clusters/robust sandwich, weights/survey, recurrent events, time-varying
covariates, interval Cox, Dataset collection, CUDA or MPS. The native numerical
runtime uses Torch CPU float64, without SciPy and without ambient RNG changes.

## Independent and original-author acceptance

The standalone `verify_interval_weibull_regression_oracles.py` imports neither
OpenEcon nor Torch. Independent NumPy analytic likelihood/score/full Hessian,
SciPy optimization, PH Jacobian, and cross-query covariance are checked. It
also actually executes the installed open-source `survival::survreg` 3.8-6
with `Surv(...,type="interval2")`, comparing all AFT/log-scale parameters,
original likelihood and full covariance across mixed-four-censor, exact,
inspection-only, and explicitly scaled fixtures. Raw original R endpoints
and matrices, version, invocation hash and errors are retained in the audit.
Portable source tests also compare the complete pinned numeric input and
full raw R output fixtures; their receipt explicitly says no new R run
occurred. A fresh R test can skip when the pinned reference is unavailable,
while the standalone default author audit requires an actual R execution.
This is open-source original-author reference evidence, without licensed
Stata/SAS/vendor parity or a finite-sample guarantee.

The 128-case calibration plan is timestamped/hashed before execution. Its
complete fixed seed denominator includes admission, numerical and optimizer
failures. It checks all AFT/PH marginal coverages, the full AFT joint
ellipsoid, fixed-profile survival/hazard coverage, bias relative to Monte
Carlo uncertainty, and the full whitened empirical covariance. A deliberately
underscaled covariance and a diagonal-only shape transformation are negative
controls. A passing simulation is bounded evidence for this stated correctly
specified model, rather than a universal coverage certificate.

Primary references:

- [survreg model and Weibull conventions](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html)
- [Surv interval encoding](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html)
- [Full log-scale covariance in survreg objects](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.object.html)
- [Pinned 3.8-6 original likelihood and joint derivatives](https://raw.githubusercontent.com/cran/survival/3.8-6/src/survregc1.c)
- [Pinned 3.8-6 LGPL package provenance](https://raw.githubusercontent.com/cran/survival/3.8-6/DESCRIPTION)
