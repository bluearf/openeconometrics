# Declared-cell Poisson loglinear models

`oe.loglinear_ipf` fits a hierarchical cell model from generating margins;
`oe.loglinear_ml` fits a caller-declared general cell-design matrix with Newton
likelihood steps. These procedures use resident native CPU float64 Torch.
They do not substitute a row-level count regression for a declared cell grid.

## Sampling and support

Supply a DataFrame or existing supported table mapping, the dimension names,
integer `count` column and explicit typed `levels` for every dimension. Every
Cartesian cell must appear exactly once, including observed sampling zeros.
A separate boolean `structural` column removes impossible cells from the
sampling support; their counts must be zero. No missing cell is filled, no
row is dropped, and no pseudocount is added. Cell levels use typed finite
non-boolean strings/numbers. A fixed `offset` column represents log exposure.

Counts are independent Poisson observations. Multinomial, product-multinomial,
fixed margins, exposure estimation, sampling weights and clustered counts
need different inference and are outside this implementation. The full
Poisson likelihood includes the count-factorial constants. The fit retains
all active and excluded cells, canonical declared ordering, original positions,
counts, offsets and support.

## Hierarchical fitting

Generating `margins=[['a','b'],['b','c']]` includes every lower-order term.
Redundant nonmaximal margins are removed. IPF repeatedly scales supported
cell means to the observed sufficient margins, starting from fixed exposures.
It stops only when every generating margin attains the recorded tolerance,
then certifies the full likelihood score and linear log-mean representation.
An empty observed supported margin is a boundary MLE and is refused.

Parameters use first-declared-category treatment contrasts. A complete
Gram-Schmidt rank audit on active structural support records and removes
aliased contrast columns; degrees of freedom equal active cells minus this
identified rank. This differs from subtracting a raw parameter count without
accounting for structural-zero aliases. The full positive-definite observed
information is `X' diag(mu) X`; its inverse yields the joint covariance.

## General likelihood and comparison

`design` rows align exactly to the supplied cell rows, including structural
rows. The procedure reorders them together with the saved grid. Supply
identifiable contrasts and distinct `terms`; a rank-deficient caller design
is refused. Newton iterations use the score and full observed information,
with likelihood line search. A small relative score alone does not certify an
interior optimum: the scale-invariant log-mean displacement of the Newton step
must also be stationary. Saved states are checked by the same criterion before
likelihood-ratio comparison. Active means near
zero, singular information, failed line search or exhausted iteration budget
produce errors rather than successful partial fits.

`oe.loglinear_compare(restricted, full)` verifies identical counts, typed
levels, support, sampling and offsets. It checks that the restricted design
space is contained in the full design space and that the full model adds
identified parameters. The LR statistic is twice the likelihood difference,
with a rank-difference chi-square reference. Equal dimension names or merely
a larger coefficient count do not establish nestedness. Restored states are
checked for checksum, dimensions, means/information/covariance agreement,
likelihood and support before comparison.

## Results and bounded domain

Tables include all parameters with SE/z/p/CI, full covariance/information,
every cell mean and delta-method SE with log-link intervals, residuals,
rank-based deviance/Pearson GOF and complete convergence trace. Wald and
chi-square approximations are asymptotic; no small-count finite-sample
calibration is asserted. A saturated model has zero residual df and no GOF
p-value. Structural cells have fitted count zero and no artificial sampling
uncertainty.

Limits: 2–6 dimensions, 2–16 declared levels each, complete grid <=4096
cells, projected contrast/caller design <=128 parameters; count <=10 million
per cell and total <=1 billion. Log offsets lie in [-20,20], design entries
have absolute value <=100, and iterations lie in [1,2000] (IPF default1000,
ML default200). Default `max_work=300_000_000` admits a conservative count of
cell/design/iteration operations before tensor allocations; maximum user
work budget is2 billion. Default `max_bytes=128 MiB`, maximum512 MiB, is
intersected with the global workspace budget. Plans cover named selection,
index, design, information, solver and trace buffers; caller DataFrames,
Python result objects, private BLAS and process RSS are outside that bound.
No Dataset/GPU/automatic interaction-selection route is claimed.

Complete `oe.summary_state(result)` includes tables and full numerical state;
`oe.restore_summary(state)` does not refit. Canonical numeric table types and
table ordering retain exact standalone `to_latex()` across restoration.

## Independent sources and checks

IBM's original [HILOGLINEAR algorithm](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/hiloglinear.pdf)
states the generating-margin/IPF recurrence; [GENLOG Poisson algorithm](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/genlog_poisson.pdf)
states independent-cell likelihood and information. This implementation's
rank audit, first-category coding and convergence tolerance are explicit;
IBM's default iteration, coding and sampling choices are not inferred.

Independent acceptance compares closed-form independence/conditional
independence means, separate SciPy likelihood fits, every entry of the
finite-difference observed Hessian and transformed inference, known-offset
and structural-zero fits, and nested LR. Negative tests cover missing support,
boundary fits, nonnested designs, changed counts/state and early resource
refusal. This is source/formula/numerical validation, not a licensed IBM run.
