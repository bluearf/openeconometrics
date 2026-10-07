# Replayed treatment effects

The Dataset route supports regression adjustment (`ra`), inverse-probability
weighting (`ipw`), IPW regression adjustment (`ipwra`) and augmented IPW (`aipw`).
It estimates all nuisance equations from every retained observation, evaluates
the potential-outcome means globally and accumulates the analytic stacked
estimating-equation Jacobian and score meat in bounded blocks. Standard errors
include fitted treatment and outcome nuisance uncertainty; they are not computed
as if estimated propensity scores were known constants.

ATE and potential-outcome means support binary or multivalued treatment. ATET
supports binary `ra`, `ipw` and `ipwra`; `aipw` ATET raises the same unsupported
estimand error as the dense implementation. Outcome models are linear, logit,
probit (including fractional outcomes) and Poisson. Treatment models are binary
logit/probit or multinomial logit. Declared categorical treatment order is
preserved, unused treatment levels are excluded and a requested control is moved
to the first position. Covariate categories are discovered globally and each
outcome equation has its own full-sample rank screen within its treatment level.

Frequency weights reproduce repeated observations. Probability weights are
normalized globally to mean one. The covariance is the native stacked
M-estimation sandwich with robust or one-way clustered score meat, without
small-sample correction, following the existing dense convention. Cluster sums
spill to bounded temporary SQLite storage. No full observation design,
propensity array, score array or group map is retained in RAM.

The source uses the shared ReplaySample missing/zero-weight policy, exact retained
physical-row fingerprints and per-pass source checks. Its snapshotted workspace
budget reserves global factors, treatment/covariate labels, nuisance and stacked
matrix work, row derivatives and any cluster cache before allocation. There is
an internal 128 MiB work ceiling and a 384-parameter limit for the stacked system;
these describe algorithm buffers, not a universal process-RSS or row-count promise.
The numerical iterations run on CPU in native float64 Torch.

Results preserve coefficient/covariance, auxiliary equation estimates and
standard errors, all-row potential-outcome means, original/retained counts,
JSON and LaTeX output. Overlap validation examines every fitted propensity.
Its summary records global minima, maxima and weighted means by probability
column; it does not pretend to supply the dense report's exact quantiles or
per-treatment quantile distributions. There is no chart subsample fitted model.

Nearest-neighbour and propensity-score matching (`nnmatch`, `psmatch`) raise an
explicit streaming-options error: their global neighbour-search and
Abadie-Imbens variance require a separate adapter. Source mutation, exhausted
workspace, nonidentified nuisance outcomes, separation, failed overlap,
nonconvergence and singular stacked derivatives fail explicitly.

Validation compares all four methods with the dense native implementation for
categorical/missing/frequency-weight clustered samples, and with independent
NumPy estimating equations whose Jacobian is obtained by numerical
differentiation and whose nuisance fits use development-only reference tools.
It also checks probability-weight normalization, multivalued controls,
ATET/POM estimands, analytic stacked derivatives, parameter-unit/permutation
invariance, bounded replayable readers, an external-estimator import block,
caller meta-device restoration, separation, collinearity, source mutation and
workspace failure. Runtime estimation uses no SciPy or external estimator.
