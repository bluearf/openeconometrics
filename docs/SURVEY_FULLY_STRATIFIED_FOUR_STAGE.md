# Four selections with explicit strata at every stage

`survey_fully_stratified_four_stage_design` declares exactly four successive
SRSWOR selections. It retains optional first-stage strata and separate SSU,
TSU and FSU strata inside each actual sampled parent. It adds eight methods:
mean, total, ratio, proportion, linear regression, logit, probit and Poisson.
See the executable [eight-method example](examples/survey_fully_stratified_four_stage_eight.py).

The caller supplies complete positive lower-stratum frames: `ssu_frame`,
`tsu_frame`, `fsu_frame`. Each record names its entire parent path, its stratum,
and its exact positive integer population. Every declared stratum needs sampled
coverage; a sampled singleton is accepted only when that stratum is a census.
Declarations record caller provenance; the program cannot authenticate a real
sampling operation. Repeated labels are local to their full typed parent paths.
Boolean, integer, floating-point and string identities remain distinct.

Each physical row is one uniquely identified sampled FSU. Its derived weight is
the product of the four cell-specific inverse sampling fractions. For a matrix
of full-weighted linearized row contributions, each stage centers totals for
its sampled units separately inside every stratum. The stratum's covariance
contribution is `ancestor_fraction_product * (1-f) * n/(n-1) * centered.T@centered`.
The ancestor products are 1, f1, f1*f2 and f1*f2*f3, taken from the actual path.
All four stage matrices are retained and sum to the full covariance matrix.

Domain and missing-value exclusions contribute zero rows to this original
physical hierarchy. They never redefine the sample or its FPCs. Reference df
are sampled PSUs minus first-stage strata. Positive-variance inference uses
Student t; zero df retains available estimates and standard errors while
withholding unsupported p-values and intervals. A zero covariance from a full
census gives point intervals. Mean, ratio, proportion and nonlinear-regression
uncertainty is Taylor linearization, with the stated design assumptions.

Saved results retain full geometry, complete primitive values or native scores
and sensitivity matrices, all covariance components, sample positions and
integrity hashes. Reload recomputes the statistical implications without an
optimizer. Tables, LaTeX, contrasts, linear combinations, Wald tests and saved
prediction remain available. CPU float64 and row, geometry, target, work and
memory limits are checked before numerical allocation or replay. The geometry
allowance is 32768 bytes per physical row, plus target/covariance workspace;
this is a conservative implementation resource bound.

Validation includes independent NumPy/SciPy estimation and nested-stratum
inference; exact finite-population enumeration for all 16 stage census masks;
full reduction to the existing four-stage API when every lower parent has one
stratum; typed identities, incomplete frames, changed samples, forged states,
allocation guards and optimizer-disabled postestimation. Installed acceptance
requires a source-pinned frozen worker, an actual native Run, eight complete
tables and LaTeX outputs, saved-file readback, normal app Quit and cold reopen.
Source tests, installed QA, hosted merge gates and public release are separate
claims. This implementation does not establish blanket Stata or R parity.

Five or more stages, PPS, arbitrary weights, calibration, generalized design
effects, noncensus singletons, additional model families, Dataset fitting,
streaming and GPU execution remain outside this bounded API. MARKET-211 stays
open for those continuations.

The recursive multistage variance principle is documented in the
[Stata survey variance manual](https://www.stata.com/manuals/svyvarianceestimation.pdf)
and its [stratified multistage design guidance](https://www.stata.com/support/faqs/statistics/stratified-multiple-stage-designs/).

Current-main integration preserved every incoming editor entry and all existing test selectors. The [fresh native receipt](econometrics/receipts/survey-fully-stratified-four-stage-native-main84ed-2026-10-10.json) binds a new frozen build to the integrated source, actual native runs, complete process exit, cold restoration and identical prior mathematical state/postestimation hashes. The separate [additive gate registration](econometrics/fully-stratified-four-stage-gate-extension-2026-10-10.json) preserves all 142 current-main selectors before the two survey files; the historical 138-selector plan and weighted extension remain byte-for-byte historical records.
