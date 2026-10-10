# Survey design and design-based inference

Planning: MARKET-144. Original audit: [GitHub #30](https://github.com/bluearf/openecon/issues/30).
Remaining advanced design acceptance is tracked in open
[GitHub #151](https://github.com/bluearf/openecon/issues/151).
Historical planning baseline `40add89` predated the checked survey declaration,
descriptive/Taylor regression and complete numeric linear/logit/probit/Poisson
coefficient replication now implemented below. Those accepted bounded methods
are baseline support; further stages reuse the score, distribution, tables and
persistence infrastructure only after auditing their own conventions.

## S0 — single-stage declaration (MARKET-202)

`oe.survey_design` now validates a resident single-stage design and returns a
JSON `SurveyDesign`; see [the implemented guide](../survey-design.md). It computes
geometry and a design-input digest, not estimates or uncertainty. Typed IDs are
nested within strata. FPC means integer population PSU counts only, so ambiguous
sampling fractions are rejected. Singleton certainty requires an explicit census
FPC. This is a deliberately narrower admission policy than the vendor declaration
syntax; it is not full `svyset` coverage. Subsequent kernels must call `revalidate`.

## S1 — descriptive Taylor targets ([MARKET-210](https://linear.app/bluearf/issue/MARKET-210))

Implemented bounded resident targets (MARKET-219 through MARKET-222): totals, Hájek means, category proportions and
ratios on a common declared sample. Domain membership is an indicator in the
estimand/influence vector; a PSU with zero domain observations remains part of
the design. Do not subset the frame before variance calculation. For stratum h
with m_h sampled PSUs, aggregate row influences u_hj and compute

`V = sum_h (1-f_h) * m_h/(m_h-1) * sum_j (u_hj-mean_h(u)) (u_hj-mean_h(u))'`.

Each target supplies its own linearized influence and denominator: a total is
not a weighted mean with renamed units. Confidence limits use the explicitly
reported survey df, initially PSUs minus strata. Initial singleton policies
remain raise/census-certainty, with census terms contributing zero. Return full
joint target covariance, weights/population estimates, PSU/stratum counts,
design hash, domain definition and outcome-specific exclusions. Zero ratio
denominators/empty domains and unavailable df are explicit failures or documented
noninferential targets, never ordinary-regression defaults.

Independent gate: handwritten two-strata/two-PSU influence sums, no-clustering
SRS reduction, FPC census, rescaled weights, domain with absent PSUs and official
survey manual data/output. Snapshot source/license before fixture storage. Test
all category levels, missing outcome policy, full covariance and t/CI, then the
frozen/display/restart layers described in the shared promotion gates.

## S2 — Taylor regression delivered; advanced designs remaining ([MARKET-211](https://linear.app/bluearf/issue/MARKET-211))

Numeric linear, logit, probit and Poisson regression with complete single-stage
PSU Taylor covariance, full score sensitivity, design df and saved conditional
prediction/margins/contrast/adjusted-Wald inference is delivered; see
[the regression contract](../survey-regression.md) and its
retained acceptance (internal evidence excluded from this public snapshot).
It preserves complete original domain/missing PSU geometry and typed saved replay.

Exactly two sequential SRSWOR stages are implemented by the
[two-stage methods](../survey-two-stage.md) (MARKET-569–576).
Eight new [three-stage methods](../survey-three-stage.md) add nested PSU/SSU/TSU
sampling, mandatory population counts and derived weights, all three recursive
FPC covariance components and distinct saved primitive replay (MARKET-577–583
and MARKET-585). Each physical row is one terminal sampled unit; domain and
missing exclusions preserve the complete hierarchy. The first-stage df is an
explicit reference convention, without exact-coverage claims.

Eight [stage-two-stratified three-stage methods](../survey-stratified-three-stage.md)
add bounded mean, total, ratio, proportion, WLS, logit, probit and Poisson
inference (MARKET-609–616). These are three real sequential SRSWOR selections,
with SSUs sampled separately within declared positive cells of each observed
PSU. The mandatory complete supplied cell frame, cell-specific population
counts and derived weights are retained in new typed records. Stage-two
covariance centers SSU totals independently within each cell; the third-stage
prefix uses that cell's sampling fraction. Domain and missing-value selection
retain the complete hierarchy. The supplied inputs cannot authenticate a cell
omitted from both the table and supplied frame, and the reference first-stage
df does not imply exact finite-sample coverage.

Eight [fully stratified three-stage methods](../survey-fully-stratified-three-stage.md)
add independent terminal TSU strata within every sampled SSU, with complete
positive-cell frames at both lower stages (MARKET-638–645). These remain three
real selections and preserve each sampling cell's own fraction.

Eight [four-stage methods](../survey-four-stage.md) add real PSU → SSU → TSU → FSU
selection (MARKET-707–714), optional first-stage strata and unstratified lower
stages. Sampled TSUs are parent clusters containing physical FSU rows; a new
fourth covariance contribution retains prefix f1*f2*f3. Separate typed state
replays all four stage contributions. Exhaustive finite-population HT checks
cover all16 census-stage combinations; nonlinear inference retains the explicit
reference first-stage df convention.

Eight [fully stratified four-stage methods](../../SURVEY_FULLY_STRATIFIED_FOUR_STAGE.md)
add separate SSU/TSU/FSU strata with complete positive lower-cell frames
(MARKET-744–751). Each real selection retains its own stratum-specific FPC and
actual ancestor sampling fractions. Typed saved state preserves all four
covariance components, complete physical sample geometry and optimizer-free
primitive replay. The earlier four-stage API remains unchanged.

Remaining subgates are five or more stages, PPS, arbitrary supplied weights,
additional singleton policies, calibration/poststratification, general DEFF and
broader regression/GLM, Dataset, streaming and GPU domains. MARKET-211 remains
open for these broader gates.
Any future ModelSpec design route must reference
the design record without treating a sampling weight as an aweight/fweight. A final product
of weights does not preserve the stage-specific variance structure. Label any
ultimate-cluster approximation explicitly. Regression influence needs the full
bread and scores, not cluster covariance pasted onto a survey label.

Persist every stage, categorical map, parameter order, design and actual sample
positions. Test finite-population two-stage enumeration and an independent
score/information oracle; include rank/separation, missing-in-domain geometry,
incorrect nesting/weight reuse and FPC inconsistencies. Publish design effects
only against a named SRS-with/without-replacement variance reference.

## S3 — replicate inference ([MARKET-212](https://linear.app/bluearf/issue/MARKET-212))

BRR/Fay, stratum-aware delete-one jackknife and design-consistent replicate
bootstrap each need separate method gates. Supplied replicate weights come with
explicit IDs, scales/rscales, centering/MSE rule, Fay convention, replicate count
and df. Generated replicates add design balance and local seed metadata. Audit
balanced two-PSU Hadamard geometry for BRR. Ordinary row bootstrap and generic
jackknife cannot stand in for design replication. Report all failed replicates
and fail a request whose complete covariance cannot be produced.

Independent enumeration of each variance multiplier/centering rule precedes
vendor comparison. Test singleton/census/domain behavior and incompatible
replicate weights. Budget full refits, output and covariance storage before
work begins. Bounded first-stage descriptive replication is implemented in MARKET-223 through MARKET-226.
MARKET-449–456 extend the same geometry to complete numeric linear/logit coefficient
refits with BRR/Fay/stratified PSU jackknife/supplied bootstrap and method-aware
saved covariance replay. See [coefficient replication contracts](../survey-regression-replication.md).
MARKET-497–504 add probit/Poisson coefficient refits across those same four
methods, retaining observed-curvature and integer-count/separation admission.
The separate linear/logit replication acceptance (internal evidence excluded from this public snapshot)
and probit/Poisson replication acceptance (internal evidence excluded from this public snapshot)
retain all sixteen bounded family/method gates, source-independent oracles,
actual native Run and complete Quit/reopen persistence at their own source pins.
Multistage, calibrated unit-varying factors, expanded supplied jackknife designs,
further model replication, count offset/exposure or real-response PPML contracts,
full nonlinear/unconditional margins and general replicate DEFF remain open in
[#151](https://github.com/bluearf/openecon/issues/151). Licensed vendor execution
remains separately tracked under [#28](https://github.com/bluearf/openecon/issues/28).

## Primary reference gates

- [Stata survey declaration manual](https://www.stata.com/manuals/svysvyset.pdf): stage/FPC/replicate conventions and singleton policy comparison target.
- [Stata survey estimation manual](https://www.stata.com/manuals/svysvy.pdf): domain and estimation conventions.
- [Stata survey introduction and worked outputs](https://www.stata.com/manuals/svy.pdf): future fixture target; observed manual output is not a local Stata run.

The policies above are OpenEconometrics proposals where stated; differing vendor
defaults must be recorded rather than silently adopted. The eight descriptive/first-stage method acceptance records are linked in the evidence (internal evidence excluded from this public snapshot).
Licensed vendor execution, expanded DEFF, multistage and remaining regression
families remain separate gates; S1/S3 parent records retain those boundaries.
