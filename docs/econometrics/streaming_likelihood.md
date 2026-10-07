# Replayed native likelihood estimation

The Dataset route for the models listed below evaluates one global likelihood.
Each iteration reconstructs the existing native Torch analytic row objective
on bounded input blocks and sums its log likelihood, gradient and Hessian with
compensated float64 totals. It never averages separately fitted batch models or
fits the 400-row chart sample. The numerical iteration is currently CPU based;
the shared design QR phase can use the checked execution context.

## Coverage

| Models | Supported replay domain |
| --- | --- |
| `glm`, `poisson`, `cloglog`, `fracreg` | Existing family/link, trials, offset/exposure and dispersion contracts; ML/Newton optimizer. Fractional logit/probit uses its existing quasi likelihood. |
| `nbreg`, `betareg` | Native count shape and beta mean/precision equations. |
| `hetprobit`, `biprobit`, `ologit`, `oprobit`, `mlogit` | Native row likelihoods, global outcome categories and joint equation covariance. Saved categorical/heteroskedastic prediction metadata is retained. |
| `tobit`, `intreg`, `truncreg` | Normal likelihood; explicit scalar censoring/truncation limits, or interval endpoint columns with missing open sides. Tobit coefficient inference uses Student t with N minus the number of outcome slopes, as in the dense implementation. |
| `heckman`, `heckprobit` | ML selection likelihood, including missing outcomes for unselected observations. Two-step Heckman is not this route. |
| `ivprobit`, `ivtobit` | ML with one or multiple continuous endogenous regressors and full recursive normal reduced forms. Endogenous variables and instruments are centred when there is an intercept, scaled for numerical work, then reported in original units. The likelihood includes the exact density Jacobian for response scaling. Two-step estimators are not this route. |
| `cpoisson`, `cnbreg`, `tpoisson`, `tnbreg` | Native censored/truncated count likelihoods. Censored count row-specific bounds and event/interval indicators keep the dense contract. |
| `zip`, `zinb`, `gnbreg`, `hurdle`, `churdle` | Inflation, dispersion and participation equations; their global cross-equation score covariance is preserved. |
| `frontier` | Half-normal, exponential and truncated-normal stochastic frontier, production or cost sign, with delta-method ancillary output. |
| `streg` | Exponential, Weibull, Gompertz, log-normal, log-logistic and generalized-gamma parametric survival, supported PH/AFT metric, entry time, failure, offset, strata and ancillary equation. Independent records without subject `id`. |

Options outside these domains raise a specific error. In particular this route
refuses normal outcome-derived min/max limits, two-step endogenous/selection
methods, non-ML GLM optimization and survival subject IDs rather than ignoring
those requests. Existing registry validation still governs link/family,
weights, covariance and role options. No entire Stata command parity is claimed.

## Sample and covariance

`ReplaySample` discovers categories globally before constructing the design;
retains the exact missing/zero-weight sample; computes global weight
normalization, centring/scales and TSQR rank factors; and verifies projected raw
row and retained physical-position fingerprints on every completed pass.
Declared unused predictor categories remain in encoding metadata and collinear
terms are omitted deterministically. Ordered and multinomial outcome labels are
chosen from positive-weight retained observations.

Frequency weights represent repeated observations. Analytic weights are
normalized over the full retained sample. Importance and probability weights
keep the existing likelihood conventions; registry restrictions such as
probability weights with conventional covariance remain enforced.

Observed information, OPG, robust and one- or two-way clustered covariance are
supported. Cluster score sums use bounded SQLite-backed accumulation. Two-way
inclusion-exclusion and any PSD repair use the dense kernel's original-unit
centred working parameter coordinates before transforming to reported units.
The whole sample, including every cluster, enters the covariance. Robust and
cluster finite-sample factors match the existing native likelihood kernels.

Restricted and comparison likelihoods replay the same complete retained sample.
Affine parameter restrictions reuse the original native row objective, preserving
nuisance equations: ordered cutpoints, inflation/selection, dispersion and
survival ancillary equations remain free where the dense command retains them.
The resulting LR/Wald model tests, reference likelihoods and pseudo R-squared
follow the implemented dense conventions, including no-intercept references.
A comparison that cannot converge in its validated precision domain is recorded
as unavailable; it is never replaced by a sampled estimate. The model test then
retains its Wald fallback where that convention permits it.

Additional diagnostics include heteroskedastic-probit homoskedasticity,
bivariate/selection independence, IV exogeneity and global first-stage tests,
count zero-dispersion comparisons, constant-dispersion GNB comparisons,
unweighted/frequency-weighted likelihood-based ZIP/ZINB Vuong comparisons,
Poisson goodness of fit, beta's weighted correlation-based pseudo R-squared and
stochastic-frontier comparison against the full-sample normal regression.
Biprobit's joint slope test includes both outcome equations. Some secondary
command-specific report fields remain narrower than the dense report; this is
not a claim of complete option or Stata output parity.

## Resources and numerical guards

Input blocks and numeric work are planned against the workspace budget captured
before design allocation, with an internal working-budget ceiling of 128 MiB.
Global native designs have at most 384 columns/parameters, and category metadata
has a separate bounded budget. Kernel information, gradient/score work and
bivariate integration buffers are reserved before iteration; blocks shrink if
required. Cluster accumulation may use temporary disk. These are live-buffer
estimates rather than a process-RSS guarantee or a total-data-row limit.

There is no full N-by-k design, N-vector retained sample mask, or all-group map
in RAM. The source must be replayable, unchanged, and have enough positive-weight
observations to identify the fitted parameters. A wide model can be infeasible
even with few rows. Repeated scans make speed depend on storage and the number
of optimizer iterations.

Global binary separation uses the native Torch certification procedure.
Nonidentified scale, boundary correlation/dispersion/frontier solutions,
precision domains and nonconvergence fail explicitly. Starts use full-sample
moments and TSQR, not a sampled fit. Persisted results include source/sample
fingerprints, passes, resource estimates and exact original/used row counts.
Charts retain at most the first 400 retained observations with their physical
row positions. Saved results support JSON and LaTeX export; post-estimation
support remains model-specific.

## Validation

The dedicated replay tests compare coefficients, covariance and actual density
log likelihood against the existing dense native implementation across all 29
adapter names, categorical/missing/frequency-weight samples and likelihood
covariance choices. They separately check analytic gradient/Hessian and row-score
identities, two-way cluster repair, parameter unit changes, saved new-data
prediction/margins, all supported frontier/survival distributions, count event
domains, source mutation, workspace failure and temporary-file cleanup. NumPy
is used only by development assertions; no external estimator or numerical
optimizer is used by this runtime route.
