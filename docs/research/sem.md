# MARKET-146: latent SEM scope decision

Status: planning deliverable; all implementation phases deferred. Preserve the
earlier `Hayır` decision for CFA/SEM/GSEM. Reconsider only after a concrete
latent measurement use case and capacity for the identification/inference gates
below. Existing observed mediation and PCA/EFA remain available on their own
contracts; they do not fit a general latent measurement model.

**Name collision resolved in the plan:** current `oe.sem` is a **spatial error
model**, declared in [the spatial manifest](../../src/openecon/econometrics/spatial/__init__.py).
Keep its meaning. Proposed latent names `oe.cfa`, `oe.latent_sem`, `oe.gsem`
are draft names only and must pass registry collision checks before activation.
[ModelSpec/ResultBundle](../../src/openecon/models.py) need a versioned multiple
outcome/latent-equation contract; a single-outcome spec cannot silently encode it.

## Deferred phase matrix

| Phase | Domain and next dependency | Acceptance gate |
| --- | --- | --- |
| SEM-1 | Continuous single-group CFA; complete raw data; explicit marker-loading or unit-factor-variance scaling; covariance structure | Independent implied covariance and log likelihood; free-parameter Jacobian rank; identified one/two-factor fixtures; Heywood/singular/underidentified refusals |
| SEM-2 | Acyclic continuous latent/observed path model, means/intercepts, selected correlated errors; depends on SEM-1 | Full joint parameter covariance, direct/indirect/total effects and their delta covariance; mean/covariance likelihood and fitted moment reproduction |
| SEM-3 | Standardized solution, fit indices, factor-score predictions and summary-statistic input; depends on SEM-2 | Raw vs covariance input agrees using declared n versus n−1 conventions; null-model definition, CFI/TLI/RMSEA/SRMR edge cases; restored score equality |
| SEM-4 | Multigroup configural/metric/scalar restrictions, MAR FIML and robust uncertainty as separate option domains; depends on SEM-3 | Group-specific scales/means, constrained likelihood comparisons, missing-pattern likelihood, full sandwich/robust test reference; MAR is an assumption, not a detected property |
| SEM-5 | Generalized binary/ordinal/count outcomes, then multilevel latent models; depends on completed continuous gates and a separate integration design | Threshold/link identification, quadrature accuracy, cluster likelihood, full information, marginal vs conditional predictions; no inherited continuous fit-index interpretation |

## Method and option contract to implement

Start with iid unweighted Gaussian continuous outcomes; reject survey/pweights,
cluster/HAC covariance, ordinal variables, missing data and cyclic equations
until their named phase is accepted. The model syntax must compile into named
loading/path/mean/covariance matrices, a unique free-parameter map and explicit
equality constraints. Fixed versus free latent scaling cannot be inferred from
data or changed to rescue a failed optimizer.

For covariance ML, use Cholesky log determinants and solves for the implied
positive definite covariance. Preserve likelihood constants and divisor/nobs
conventions when comparing raw and sufficient-statistic input. A nonnegative
model df alone does not establish identification: check constraint Jacobian
rank and information rank, then test known nonidentification examples. Record
gradient/step/likelihood convergence and the actual constrained solution. A
boundary variance or inadmissible covariance blocks ordinary interior Wald
inference rather than producing invented SEs.

Fit indices require a saved baseline model, sample definition and statistic
convention. Saturated/zero-df models and degenerate baseline tests have explicit
unavailable values, not generic divisions by zero. Standardization transforms
both estimates and full covariance. Indirect-effect uncertainty uses joint
path covariance; bootstrap, robust corrections and multigroup LR calibration
must be separately identified before they are enabled.

Persist parameter/constraint order, latent scale and sign convention, means,
covariances, full covariance of free parameters, missing patterns, grouping,
baseline fit and the factor-score method. Score predictions must use saved
loadings and covariance, preserve row alignment, and distinguish estimates of
latent scores from observed-variable forecasts. Summary input without a mean
vector cannot support mean-structure claims; it cannot reconstruct individual
scores.

## Numerical and delivery plan

Use small generated moment matrices with hand-computed implied covariance,
then independently fit the same complete Gaussian CFA/path specification in
lavaan. Its CFA example supplies an explicit measurement model and fit output;
its estimator documentation distinguishes complete-data and missing-data ML.
[lavaan CFA](https://lavaan.ugent.be/tutorial/cfa.html),
[estimator/missing-data conventions](https://lavaan.ugent.be/tutorial/est.html).

Before runs, freeze dataset hashes, free/fixed mapping, residual correlations,
likelihood convention, reference version and full-precision output. Compare
all estimates, off-diagonal covariance, gradients, likelihood, fitted moments,
indices and scores. Declare scale-aware tolerances based on conditioning before
inspection; a loose coefficient match cannot excuse information failure.
Adversarial fixtures include duplicate indicators, too few indicators,
unidentified means, zero residual variance, inconsistent summary n/covariance,
unknown categories, missing-pattern extremes and invalid equality constraints.

Pilot resource admission accounts for resident data `O(n p)`, covariance and
factor work, parameter information `O(q²)`, bootstrap storage and missing-pattern
count; expose bounded iterations/evaluations. Reject over-budget models before
optimization. CPU-only source/frozen/native round trips are the first platform
gates. No scale, fit, vendor or GPU results exist yet for these deferred phases.

Open implementation: [GitHub #32](https://github.com/bluearf/openecon/issues/32).
Completion of this plan preserves MARKET-73 and that implementation work as open.
