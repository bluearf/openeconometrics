# Categorical responses and full-refit CATREG resampling

`oe.catreg_nominal_response` and `oe.catreg_ordinal_response` learn a scalar
response quantification together with predictor quantifications. They extend
the numeric-response procedures; existing `catreg_nominal`, `catreg_ordinal`
and `catreg_predict` retain their original behavior.

## Response scaling

Supply one categorical outcome, distinct predictors, and explicit `scales`
for mixed nominal, ordinal and numeric predictors. Predictor scaling defaults
to nominal. Every ordinal predictor requires an `orders` entry. An ordinal
outcome separately requires `outcome_order`, containing exactly its observed
complete-sample categories in the intended order. Category identity is typed:
strings, finite numbers and booleans are distinct. Numeric predictors must be
explicitly declared; there is no automatic discretization.

The response map has frequency-weighted mean zero and population variance
one. Given current fitted scores, the nominal response update computes
category-conditional fitted means, centers them and divides by a positive
normalization factor. The ordinal update first projects these means onto the
declared increasing cone by weighted PAVA. Adjacent outcome levels may pool
to equal quantifications. Predictor partial-residual category updates and
joint least squares follow the response update. Loss is the actual mean
squared difference between quantified response and fitted scores.

The nominal response contrast is arbitrary up to global sign. After fitting,
the largest absolute response-map entry is made positive; response maps,
observations, regression effects and fitted values change sign together.
The ordinal response remains increasing in its declared order. Signed
predictor effects can describe decreasing associations.

All declared starts retain full objective/change traces and failure reasons.
Acceptance requires loss convergence, small fitted/response changes and
response-block stationarity. The chosen result is the best accepted start;
there is no general global-optimum certificate. A separate conditional
category-contrast eigenvalue audit checks the nominal response against the
current fitted predictor space. Tied leading nominal roots are refused as an
unidentified contrast; zero association, collapsed maps, rank-deficient
designs and exhausted iteration budgets are refused. The ordinal audit does
not equate an unconstrained leading eigenvector with an ordered optimum.

Results show original outcome labels separately from quantified observations,
fitted scores and residuals. Coefficients, SSE and R² refer to the learned
unit-variance response scale. No ordinary adaptive coefficient covariance,
SE, Wald test or confidence interval is supplied.

## Saved response prediction

`oe.catreg_outcome_predict(result, query)` applies saved predictor maps and
coefficients after complete structural and numerical state verification. It
accepts fitted results and results restored with `oe.restore_summary`.
Unknown predictor categories are refused. `missing='raise'` is the default;
explicit `missing='drop'` preserves original query positions.

The output is a quantified-response score. It is not a calibrated class
probability or an inverse category label. A nominal ordering is learned;
ordinal pooling can make inversion nonunique. Complete saved state retains
maps, typed categories, counts, original sample positions, raw training
predictors, transformed observations, conditional response roots and
normalization/convergence settings. Prediction verifies mapping agreement,
full-design least squares, response stationarity and the saved objective;
it does not refit. Exact versioned field sets, bounded typed primitive labels,
finite scalar matrices and a named JSON serialization envelope are checked
and admitted before hashing or tensor allocation.

## Fixed-query full-refit bootstrap

`oe.catreg_bootstrap(data, outcome, predictors, queries=...)` restricts its
estimand to fitted mean predictions at explicit fixed query rows in the
original units of a **numeric outcome**. It supports existing nominal or
ordinal predictor CATREG via `predictor_scale`, `scales` and `orders`.
It does not bootstrap categorical-response fits.

Each seeded paired iid row resample refits all category transformations,
numeric standardizations and coefficients, with declared multiple starts.
The same fixed raw query rows are evaluated in every accepted fit. This
functional is invariant to compensating predictor-map sign/scale choices;
comparing unaligned adaptive coefficients would not have that property.
Every draw retains selected starts, complete objective histories, learned
maps, coefficients and original sampled row positions.

When **all** declared draws succeed, tables contain the full joint empirical
prediction covariance (`reps-1` denominator) and percentile confidence
intervals, using linear empirical quantile interpolation. These quantify
variation of the complete fitted procedure under paired iid resampling.
They are not ordinary conditional OLS intervals, new-observation prediction
intervals, simultaneous bands or a general finite-sample coverage guarantee.
Small resample counts have coarse Monte Carlo resolution. No adaptive
coefficient p-values are returned.

Every resample must retain the original categorical schema. If a category
disappears, or a fit loses rank, degenerates or fails convergence, the draw
is a recorded estimator-domain failure. Default `failure='raise'` refuses
the result and attaches the encountered failure and original sampled
positions to the error. Explicit `failure='record'` completes the draw audit
and retains all failures and successful predictions, but withholds **all**
covariance and interval tables if any draw failed. Failed draws are never
silently discarded or replaced; surviving-draw inference is not claimed.

## Resident input and admission limits

All APIs use native Torch CPU float64, with resident DataFrames, selected
column mappings or records. Dataset collection, GPU, case/survey weights,
cluster resampling and streaming are unsupported. Caller default devices do
not alter the declared CPU domain.

Limits are 3,000 input rows and 32 observed levels per categorical variable.
Response fits allow 11 predictors plus the response; numeric-response
bootstrap inherits the existing 12-predictor limit. Column names and
categorical labels have at most 256 characters; each declared order has
2–32 levels. Starts are 1–12, maximum iterations 1–1,000 and tolerance
1e-12–1e-3. Response defaults are four starts and 500 iterations.

Bootstrap queries contain 1–64 complete rows; resamples are 2–499, with
defaults 39 resamples, two starts and 100 iterations. These maxima do not
promise that every combination is admitted. Aggregate worst-case fitting,
query and resampling work is admitted against `max_work` (default 300 million)
before draw fits. A conservative complete maps/indices/traces artifact
estimate must fit the 32 MiB summary domain before resampling.

Named input-selection, row-mask, numerical fit, query and draw/covariance
buffers are admitted before their major allocations. `max_bytes` defaults
to 128 MiB and is intersected with the current global workspace budget.
Mapping/record inputs project required columns before coercion. Caller
input, Python result objects, private BLAS/allocator workspace and process
RSS are outside this named-buffer contract.

Use `oe.summary_state(result)` and `oe.restore_summary(state)` for complete
JSON persistence. Console table previews are not complete state. All tables,
attributes, numerical payloads and standalone LaTeX round-trip exactly.

## Sources and independent validation

IBM's original [CATREG algorithm, response update and normalization](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catreg.pdf)
and [dependent-variable scaling options](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=catreg-analysis-subcommand-command)
support categorical responses. [Scale definitions](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=catreg-define-scale-in-categorical-regression)
distinguish nominal and order-preserving quantification. These references
support the algorithmic domains; they do not constitute a licensed IBM run.

Independent checks solve the nominal generalized category-contrast
eigenproblem, enumerate ordered pooling faces and isotonic partitions,
verify binary standardized OLS limits and a mixed additive dummy-space
limit, and compare every bootstrap query draw with separate NumPy dummy
OLS refits. Tests cover joint covariance/percentiles, exact state/LaTeX,
missing alignment, counted failed resamples, unknown levels, resealed
semantic corruption, tied roots, rank/convergence and early resource/device
refusals. No blanket SPSS/Stata parity or unrestricted inference is claimed.
