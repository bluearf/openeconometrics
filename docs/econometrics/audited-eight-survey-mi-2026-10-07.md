# Survey and missing-data capability milestone audit, 7 October 2026

The review starts from main `74d060d` and incorporates the merged missing-data
PR [#140](https://github.com/bluearf/openecon/pull/140), merge revision
`17760939fa6ae0812185b1a8ae85a2e19652d125`, published head
`04187d5aad3d0ebf3684796de83e789d6691bd7f`. Survey regression from
PR [#129](https://github.com/bluearf/openecon/pull/129) is already present.
The later merged PR [#165](https://github.com/bluearf/openecon/pull/165), merge
revision `f1dc50b3dd3893323354bb3a8ff06b0657ea535c`, adds complete coefficient
replication for numeric linear and logit models. This guide distinguishes that
new acceptance from the earlier Taylor/MI source review and native receipts.
The subsequent MI PR [#163](https://github.com/bluearf/openecon/pull/163), merge
revision `63f1924d379f75ac3a8a934bf02e2d54acffdb1f`, supplies separate bounded
discrete, passive, fixed-delta and pooled-contrast extensions.
PR [#169](https://github.com/bluearf/openecon/pull/169), merge revision
`2416f54d9b26f215f04453f066ed2d8b8ecd731b`, adds the corresponding eight
probit/Poisson coefficient-replication gates. All four numeric regression
families now support the four declared whole-PSU variance methods.
The original GitHub #30/#31 acceptance bodies retain their broader obligations.
Historical `research_notes` competitor inventories and `reports` plans are
comparison and traceability records, not current numerical or package receipts.

## Completion against the original issues

| Original requirement | Completed milestone | Remaining obligation |
| --- | --- | --- |
| #30 design declaration | Typed nested-within-stratum PSU identity, positive sampling weights, integer first-stage population-PSU FPC, explicit certainty singleton policy, immutable checked design | Multistage nesting, stage weights/FPC and multistage independent reference cases |
| #30 descriptive and domain inference | Means, totals, ratios, fixed-category proportions; joint missing/domain sample and complete Taylor covariance with all original PSUs | Broader unequal-weight design effects and additional reference conventions |
| #30 replication | Descriptive targets plus complete numeric linear/logit/probit/Poisson coefficient refits and full covariance for BRR, Fay, whole-PSU stratified jackknife and supplied justified first-stage bootstrap | Calibrated unit-varying plans, expanded supplied jackknife designs, further model replication, full nonlinear/unconditional margin replication |
| #30 regression | Four numeric single-stage fits and four saved-result procedures, complete PSU Taylor covariance, design df, convergence/identification guards and persistence | Calibration/poststratification, broader GLMs and unconditional/categorical margins |
| #31 missingness diagnostics | Pattern summaries, observed Gaussian ML EM, common-covariance Little MCAR | Robust/categorical/unequal-pattern-covariance variants and missingness-model sensitivity |
| #31 imputation workflows | Proper-prior NIW MVN, verified monotone Gaussian regression, Gaussian FCS, type-1 PMM, proper-prior binary logistic FCS; later proper-prior Poisson/ordered-logit/multinomial FCS | Cross-model controlled coverage, finite-chain precision/mixing and substantive compatibility; further conditional likelihood/support domains |
| #31 passive and sensitivity extensions | Dependency-ordered affine/product/integer-power derivations after MI; fixed single-incomplete-target normal/logit/Poisson delta scenarios | Within-sweep passive feedback, substantive compatibility and general/multitarget MNAR sensitivity with declared identification and coverage contracts |
| #31 saved state and pooled inference | Original missing/observed cells, physical positions/index, IDs/seeds/prior/model/sampler records; full Rubin covariance, marginal finite df/FMI and multivariate D1; later full-covariance pooled linear contrasts with nonzero nulls | Additional nonlinear/predictive pooled targets, D2/D3, multiplicity and the remaining sampling/imputation-model contracts |
| #31 controlled coverage | Predeclared MCAR and observed-x MAR experiment for congenial monotone Gaussian outcome MI plus iid OLS/Rubin | Coverage/convergence/sensitivity evidence for other sampling schemes and model combinations |

These are completed bounded milestones. Remaining original survey requirements
are preserved in open continuation [#151](https://github.com/bluearf/openecon/issues/151);
remaining original missing-data requirements are preserved in open continuation
[#152](https://github.com/bluearf/openecon/issues/152). Closing the core milestone
does not establish family-wide acceptance or erase the original broad scope.
Vendor execution, optional Dataset/device routes and public release acceptance
retain their separate evidence gates.

## Single-stage survey inference law and replay

[Design](survey-design.md), [descriptive/replicate inference](survey-inference.md)
and [regression](survey-regression.md) have distinct contracts. The current
regression APIs are `survey_regress`, `survey_logit`, `survey_probit`,
`survey_poisson`, `survey_predict`, `survey_margins`, `survey_lincom` and
`survey_test`, with `survey_regress_replicate`, `survey_logit_replicate`,
`survey_probit_replicate` and `survey_poisson_replicate` providing the later
coefficient-replication stage. Earlier declaration/descriptive-stage statements about missing
regression do not describe the current combined API.

Complete weighted row scores are summed within every original PSU. Domain and
listwise-excluded records have zero scores; their PSUs remain in each stratum's
centering and first-stage FPC correction. Regression covariance sandwiches the
stratified PSU score covariance with the full score sensitivity. Probit uses
the observed Hessian. The named design df is all PSUs minus all strata.
Linear contrasts use Student t and multiple restrictions use the declared
adjusted Wald F. Census/zero-df/zero-SE cases retain their explicit undefined
test conventions. Replication retains all draws, factors and centering; its df
conventions are separately declared rather than inferred from ordinary OLS.

Predictions and continuous average marginal effects propagate the complete
coefficient covariance. Evaluation covariates and standardization weights are
fixed; there is no empirical-distribution uncertainty, unconditional margin,
categorical/discrete contrast or future-outcome noise claim. This matches the
[official conditional/unconditional margins distinction](https://www.stata.com/manuals/rmargins.pdf).

Saved survey fits retain design/sample identity, coefficients, full covariance,
convergence and complete PSU scores/sensitivity. Restore checks the full state
and replays covariance without optimizing parameters. New independent checks
use an unequal-weight two-stratum fixture with one entirely excluded PSU and a
missing in-domain outcome; NumPy reconstructs the complete Taylor sandwich and
SciPy supplies the design-t reference. Conditional predictions retain duplicate
labels and original physical exclusions, including full off-diagonal covariance.

The [coefficient-replication contract](survey-regression-replication.md) retains
every complete linear/logit/probit/Poisson refit and computes
`V = sum_r multiplier[r] * (beta[r]-center[r])(beta[r]-center[r])'`.
BRR requires balanced paired noncensus PSUs, with multiplier `1/R`; Fay uses
`1/[R(1-rho)^2]`. These methods reject noncensus partial FPC. Generated
stratified delete-one PSU jackknife uses each stratum's
`(1-f_h)(m_h-1)/m_h`; supplied PSU-constant bootstrap factors require explicit
design justification and `scale*rscales[r]`. Centers are the original
coefficient vector, replicate mean, or eligible within-stratum jackknife mean.
Census weights are held fixed. All-census BRR/Fay/jackknife coefficient
replication is rejected with the separate Taylor route named explicitly;
there is no silent estimator substitution.

Every original/replicate fit must retain the same identified coefficient
order and converge. Zero replicate weights remove only that fit's active
rows; the original design universe and admitted physical sample remain fixed.
Any failed replica rejects the whole request. Its declared design-t convention
is `min(complete_design_df, R-1)`; only supplied bootstrap may declare a positive
df below `R`. Saved prediction, margins, contrasts and adjusted Wald inference
propagate the complete coefficient covariance through the existing fixed
covariate Jacobians. This does not replicate a nonlinear margin in every draw,
add empirical-distribution uncertainty or produce future-outcome intervals.
The [official variance formulas](https://www.stata.com/manuals/svyvarianceestimation.pdf)
and [Statistics Canada bootstrap workflow](https://www150.statcan.gc.ca/n1/pub/12-002-x/2014001/article/11901-eng.htm)
provide method references; the finite-df contract and independent validation
do not establish licensed-vendor equivalence.

Probit retains exact binary 0/1 admission and observed likelihood curvature.
Poisson requires finite exact nonnegative integer responses at most `2**53`;
offsets/exposures and real-response PPML are outside this bounded API. Every
original/replicate fit must pass strict rank, native separation and convergence
checks. A replica that loses the only opposite binary response or positive-count
support refuses the request rather than receiving a repaired estimate.

Saved replay validates complete replica IDs, coefficients, active physical
positions, convergence, balanced signs, scales/centering, method-specific df
and work records, then reconstructs full covariance without refitting. Its
weights/provenance digest is integrity evidence, not authenticated sampling
design. Resident CPU float64 admission bounds 32 coefficients, 4,096 replicas,
eight million complete replica-weight cells and 50 million cumulative work
units. Workspace and the complete fit plan are checked before replica allocation;
all actual solver and step-halving evaluations reserve cumulative work.

## Missing-data science and inference boundaries

The method contracts are [diagnostics](mi-diagnostics.md),
[Gaussian generation](mi-generation.md), [FCS](mi-chained.md) and
[pooling/D1](mi-pooling.md). Runtime kernels use resident CPU float64 Torch.
Development NumPy/SciPy oracles are independent checks, not runtime estimators.

EM maximizes observed marginal Gaussian likelihoods and includes conditional
covariance in expected second moments. It returns only after both standardized
parameter and relative likelihood changes meet the declared tolerance. Its
covariance is the fitted population covariance, not parameter uncertainty.
Little compares observed pattern means with the common fitted mean, uses
informative `n/(n-1)` times the ML covariance and an asymptotic chi-square
reference. Entirely missing rows retain identity but contribute no observed
information. A nonrejection does not establish MCAR or MAR.

MVN data augmentation uses an explicit proper NIW prior in original units,
conditional Gaussian predictive noise and a Bartlett inverse-Wishart parameter
draw. Monotone regression checks actual nesting and uses full-rank Gaussian
posterior coefficient and residual draws under the stated improper prior.
These conjugate laws follow
[Murphy's Gaussian analysis](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
and the [MICE Gaussian model](https://amices.org/mice/reference/mice.impute.norm.html).
Gaussian FCS includes parameter and residual uncertainty. PMM uses observed
fitted predictions versus missing posterior predictions, with declared donor
count and tie policy; it preserves observed donor support. This is the
[type-1 PMM convention](https://amices.org/mice/reference/mice.impute.pmm.html).
Binary FCS uses a proper Gaussian coefficient prior and a fixed, symmetric
Metropolis proposal targeting the exact logistic posterior. Finite Gibbs,
FCS and MH schedules provide diagnostics; no convergence, stationarity or
independence certificate follows from their requested iteration count.

Pooling preserves every per-imputation coefficient vector and full within
covariance. Rubin total covariance is `Ubar + (1+1/m)*B`; marginal Student-t
calibration uses Rubin or finite-complete-df Barnard–Rubin, with FMI retained.
D1 uses the proportional between/within covariance approximation, separate
Li finite-imputation branches and Reiter's finite-complete-df formula. Reiter's
positive-moment domain is checked explicitly; an unsupported domain does not
silently obtain another df. The independent restriction/unit-transformation
check preserves the whole statistic, df and p-value. Primary formula sources:
[author multiparameter text](https://stefvanbuuren.name/fimd/sec-multiparameter.html)
and [Reiter manuscript](https://www2.stat.duke.edu/~jerry/Papers/Bmtka07.pdf).

Later [discrete FCS](mi-discrete.md) implements declared Poisson, ordered-logit
and baseline multinomial conditionals. Ordinal/multinomial models require an
explicit numeric category declaration with three to eight observed categories;
proper Gaussian parameter priors and finite symmetric MH schedules retain
posterior/predictive state. These add appropriate bounded ordinal/count routes,
without proving FCS compatibility, stationarity, mixing or confidence-interval
coverage across arbitrary chained model combinations. The original Gaussian
MCAR/MAR experiment below does not validate those new laws automatically.

[Passive derivation](mi-passive.md) now applies a dependency-ordered declared
affine/product/integer-power program after completed MI. It preserves source
type and checked observed/missing propagation, rejects dependency/support
errors and reconstructs the full derived state. This is distinct from feeding
passive variables back into each FCS sweep. [Fixed delta sensitivity](mi-sensitivity.md)
fits one incomplete normal, binary or Poisson target and applies the caller's
declared offset in its mean, logit or log-rate domain. It provides conditional
scenario draws, not empirical MNAR identification, general multitarget NARFCS
or calibrated sensitivity coverage. [Pooled linear contrasts](mi-lincom.md)
project every coefficient vector and full within covariance before Rubin or
Barnard–Rubin marginal inference, including nonzero nulls. They do not add
joint/multiplicity adjustment, nonlinear or pooled predictive inference.
Each new typed state checks retained inputs, priors/parameters/derived outputs
or covariance, and admits work/workspace explicitly. Resident CPU float64
remains the implemented route; weights, Dataset and GPU need separate contracts.

## Predeclared controlled missingness experiment

Each mechanism uses 160 iid samples, `n=120`, `m=20`,
`y=.7+1.2*x+Normal(0,1.3²)` and observed `x~Normal(0,1)`. Outcome missingness is
independent Bernoulli `.30` for MCAR or `logistic(-1+.7*x)` for MAR conditional
on observed x. NumPy PCG64 seeds are `302310` and `302311`; every imputation
seed and all 320 replicate estimates/df/intervals are saved. Full iid OLS
covariance is computed independently for each completion, then native
Rubin/Barnard–Rubin uses complete-data df 118 and a 95% slope interval.

| Mechanism | Covered intervals | Observed coverage | Wilson 95% Monte Carlo interval | Exact 99% Monte Carlo interval |
| --- | --- | --- | --- | --- |
| MCAR | 153/160 | 95.625% | 91.246%–97.865% | 89.627%–98.711% |
| MAR given observed x | 154/160 | 96.250% | 92.061%–98.270% | 90.505%–99.029% |

The predeclared two-sided binomial check of nominal `.95` at `.01` gives
`p=.8568` and `.5872`. Its Monte Carlo uncertainty is reported, rather than
replacing inference with an arbitrary coverage cutoff. This checks one
congenial Gaussian geometry and one-pass monotone imputation. It supplies no
coverage claim for arbitrary FCS, PMM, binary chains, ordinal/count models,
MNAR or survey-weighted pooling. Receipts:
MCAR (internal evidence excluded from this public snapshot),
MAR (internal evidence excluded from this public snapshot).

## Evidence identity and complete artifact persistence

The retained survey acceptance (internal evidence excluded from this public snapshot)
and missing-data acceptance (internal evidence excluded from this public snapshot)
belong to their recorded source/frozen/application revisions. Survey's earlier
descriptive manifest does not include the later regression exports. The earlier
survey identity review, including its 17-of-18 selected-body match, describes
the source before PR #165. The new coefficient-replication stage changes called
`regression.py`, `regression_common.py` and the descriptive replica planner in
`replication.py`: the coefficient solver is extracted into `fit_sample`, while
the default Taylor build/restore branch still uses complete PSU scores,
sensitivity, first-stage FPC and design df with the original schema/digest.
That is an actual scientific dependency integration, not merely additive help
metadata. Its changed bodies must not be called byte-identical to the older
Taylor frozen application or covered only by the older 187-case run.

The separate replication acceptance (internal evidence excluded from this public snapshot)
records pin `361afa63a61462ab3ebf963d4751622b32fb1ede`, one actual native Run and
full Quit/relaunch with eight coefficient tables, all saved replica states and
40 complete postestimation tables. Rereading its retained files verifies 86,492
canonical state bytes, SHA256
`9888c928aed554688d947fb1e8888077078e7e238cbcfd408670cc3f6f7ec5fc`,
483,046 postestimation bytes, SHA256
`bcf52923d57d12ef1027b902156233be71618e964a015f140f688b5fb3a51c80`,
and all eight exact LaTeX exports. This is the historical linear/logit acceptance
pin; PR #169 subsequently expands the shared typed replica-family admission
and appends the two new wrappers. Its earlier whole-file identities must not be
relabeled as current identities. The retained
science bridge (internal evidence excluded from this public snapshot)
compares the eight complete source/native cases independently of the UI record.
This audit does not repeat or relabel those native actions.

The later probit/Poisson acceptance (internal evidence excluded from this public snapshot)
records frozen pin `a83981eb5cb8458e6cb5b34959ff959ac4168871`, one actual native
Run with eight tables/LaTeX exports, and full Quit/relaunch persistence of all
eight fits and forty helpers. Its
science bridge (internal evidence excluded from this public snapshot)
retains exact complete source/native replay with the selected source identity.
Independent rereading verifies 108,086 canonical fit-state bytes, SHA256
`0ebc5ab0cd28cb933c9302159b9db7336d585784d1f7a168ccd1c60f079e24f7`,
591,611 complete helper-state bytes, SHA256
`88b60e51351dd98e69d05790630deaef27238e99828c765321931dad4f282c31`,
all eight exact native LaTeX exports and the complete 33-file checksum roster.
The bridge's nineteen named numerical/acceptance modules match the integrated
source; the compiled selection's top-level exports and capability module keep
separate identities.
Independent review of the three scientific-file changes against the preceding
`f714b9d` integration finds two added exports, the replica-family admission
extension and two thin family wrappers. The common solver, likelihood/score,
observed sensitivity, variance planner and default Taylor computation are
unchanged by this extension. The source/native acceptance and rendered restart
records retain their own pins; they do not prove whole-current-SDK frozen identity.

The fresh survey/MI scientific rerun (internal evidence excluded from this public snapshot)
checks the original 187 cases against the later integrated source. It includes
the independent Taylor sandwich, complete conditional covariance, immutable
saved state and MCAR/MAR experiment. These are the same cases as the earlier
187-case receipt, not 187 additional unique tests; replication's separate
acceptance suite keeps its own scope. Full current source/installed SDK and
native frozen identity are separate proof layers.

For the original MI stage, the reviewed 15-file
scientific/execution/example/verifier set was byte-identical to its accepted
pin at that review. Its six numerical/typed-state module bodies remain
unchanged; PR #163 adds four modules and expands `mi/__init__.py` bindings.
The older 15-file identity must therefore retain its historical scope. That
set also differs from the old frozen receipt's 15 compiled modules, which
include the historical top-level binding, registry and contract modules.
Shared API/metadata integrations must not be disguised as identical compiled
whole-SDK state. The final published PR tree and current scientific files are
checked independently in the
source-identity receipt (internal evidence excluded from this public snapshot).

The later MI extension acceptance (internal evidence excluded from this public snapshot)
records native build `18d242916dece0deb6d6e0156e85d3d2fecb79e6` and accepted
frozen comparison pin `ca04908c2d1e57efa5cbd80e017ecb3da3e74708`.
Its merged scientific bridge (internal evidence excluded from this public snapshot)
names 28 scientific/execution/example/verifier files that still match the
integrated source. Its 26-module compiled manifest has separate registry and
capability identities; it is not whole-current-SDK frozen identity. Rereading
all eight complete saved extension states verifies 302,532 canonical bytes,
SHA256 `4a6e50a5a1b29cb1a46642770ec055617fb2de86b0d68803114d0be470a1e3f2`,
plus all eight exact retained LaTeX exports. Native Run performed typed
restoration; the recorded Quit/process exit/reopen establishes saved history,
files/states and rendered output persistence, without claiming retained Python
variables or a second native hydration. This is a separate local debug,
ad-hoc-signed QA application, not a public package or release acceptance.

The retained MI native execution JSON, 300,660 canonical full-state bytes and all
eight exact LaTeX files agree with the before/restart receipts. Full state SHA256
is `bca35c401324db8d1f1e7c405982457c185e82a51e9cc3f2c02fd6a79cc61b7f`.
The recorded native Quit proof identifies the three terminated prior processes.
This audit rereads the retained artifacts; it does not rerun or relabel their
actual native Run/restart observation. The survey QA bundle's macOS minimum is
15.0; the separate MI QA bundle's audited minimum is 26.0. Both are local
ad-hoc-signed acceptance applications, with no public release/notarization claim.

The new [example](../examples/audited_eight_survey_mi.py) exposes
`run(destination)`. It saves six exact OLS ResultBundle JSON files before pool/D1,
a standalone complete pool and all its tables, every fitted/imputed state,
complete conditional survey outputs and 16 LaTeX exports. Checksums cover actual
saved bytes; fresh run UUIDs retain their real bundle/provenance associations.
Source replay with actual external-estimator imports blocked is separate from
any later wheel or installed-runtime run. Immutable typed restoration checks
observed cells, missing geometry and physical identities without refitting.
An integrity digest is not an authenticated sampling or imputation proof.

## Focused continuation acceptance

Survey continuation [#151](https://github.com/bluearf/openecon/issues/151) preserves
the original multistage obligation as a stage with explicit
nested stage identities, stage weights and stage population-size/FPC semantics.
Its acceptance must include an independent two-stage reference covariance,
singleton/certainty and domain/missing fixtures, complete metadata persistence,
resource admission and native replay. Keep calibration/poststratification and
general DEFF/DEFT in a separate stage. Numeric linear/logit/probit/Poisson BRR, Fay, generated
stratified jackknife and supplied design-bootstrap coefficient covariance are
already delivered by PR #165/#169 and must not remain generic unfinished replication
items. Keep calibrated unit-varying replication, expanded supplied jackknife
designs, further model replication, count offset/exposure or real-response PPML
contracts, and full nonlinear or
unconditional margin replication as explicit remaining stages, each with its
own estimand, variance law, independent full covariance and failed-replica tests.

Missing-data continuation [#152](https://github.com/bluearf/openecon/issues/152)
retains cross-model controlled coverage, sampler precision/mixing and
congeniality as scientific validation stages with predeclared replicated
coverage and diagnostic criteria. Poisson/ordered-logit/multinomial FCS,
post-MI affine/product/integer-power passive derivations, single-target fixed
normal/logit/Poisson delta scenarios and full-covariance marginal linear
contrasts are already delivered by PR #163 and must not remain generic absent
methods. Within-sweep passive feedback and substantive compatibility need their
own dependency/observed-derived-cell contracts; broader MNAR or multitarget
sensitivity needs explicit identifying assumptions and calibrated validation.
Further likelihood/support domains, D2/D3, nonlinear/pooled prediction targets,
multiplicity, survey weights and Dataset/device routes keep their own bounded
contracts. The original controlled Gaussian experiment is retained without
extending its coverage conclusion to the new samplers or delta scenarios.

Each continuation should link the original acceptance and the completed core
receipts, without renaming unsupported methods as already validated. Licensed
vendor comparison and deployment acceptance remain distinct from source,
independent-reference, wheel and retained native evidence.
