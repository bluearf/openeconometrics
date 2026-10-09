# Categorical, conjoint and causal-design milestone audit — 2026-10-07

This audit reconciles GitHub #58, #60 and #75 with the original `3de1154`
requests, referenced research notes/reports, and merged PR #145, #142 and #141.
The categorical scope below also includes the bounded options merged in
[PR #162](https://github.com/bluearf/openecon/pull/162).
The causal scope includes the eight sensitivity/assignment methods merged in
[PR #150](https://github.com/bluearf/openecon/pull/150).
Each family now has a substantive native core with saved scientific state.
The broader original obligations remain implementation work in focused successor
stages: [categorical scaling, geometry and loglinear sampling #155](https://github.com/bluearf/openecon/issues/155),
[conjoint input, design and population inference #156](https://github.com/bluearf/openecon/issues/156),
and [causal sensitivity, assignment and observational targets #157](https://github.com/bluearf/openecon/issues/157).
A parent milestone closure does not assert every competitor option.

## Categorical scaling and loglinear core (#58)

| Delivered target | Numerical/sample/inference contract | Remaining domain |
| --- | --- | --- |
| `catreg_nominal`, `catreg_ordinal` | Numeric outcome; explicit nominal/ordinal/numeric predictor scales and exact ordinal orders; centered unit-variance maps, signed effects, multiple seeded starts and monotone ALS loss. Missing rows are explicitly excluded and stored. Saved `catreg_predict` applies trained maps in original outcome units. | Spline transformations, regularized scaling with valid tuning/inference and broader weighting. Adaptive coefficients have no naive OLS SE/CI. |
| `catreg_nominal_response`, `catreg_ordinal_response`, `catreg_outcome_predict` | Joint scalar response/predictor quantification; response mean zero and variance one under observed category counts, explicit ordinal outcome order and possible pooled levels. Multiple-start loss and block stationarity persist; tied leading nominal contrasts are refused. Saved prediction returns a quantified-response score without fitting again. | Calibrated class probabilities, justified original-category recovery where inversion is not unique, and uncertainty for learned categorical-response maps/effects. No ordinary adaptive coefficient SE/p/CI is supplied. |
| `catreg_bootstrap` | Numeric-outcome fitted means at explicit fixed raw query rows; seeded paired iid resamples refit every transformation and coefficient. Full joint empirical prediction covariance and marginal percentile intervals appear only when every draw succeeds; record-policy failures retain the audit and withhold all inference. | Categorical-response resampling, adaptive coefficient/score/rotation inference, simultaneous or outcome-noise intervals, cluster/survey resampling and supported weighted domains. This empirical functional is not a general selection-inference guarantee. |
| `catpca` | One scalar quantification per variable shared across retained dimensions; explicit numeric/nominal/ordinal scales, normalized object scores and full loadings/eigenvalues; saved-map projection. | Multiple-nominal CATPCA, additional passive-variable/spline domains and uncertainty after learned transformations. |
| `catpca_varimax`, `catpca_promax`, `catpca_rotated_predict` | Saved single-vector CATPCA component rotation: orthogonal varimax or powered-target least-squares promax, coherent pattern/structure/component correlation and inverse-transpose score transformation. Scalar maps and reconstruction persist unchanged; projection verifies source and rotation semantics. | Additional rotation families, original-category person centroids unavailable from tied scalar maps, multiple-nominal CATPCA and adaptive sampling uncertainty. No global rotation optimum or vendor equivalence is claimed. |
| `mca`, `overals` | Unweighted disjunctive MCA with raw inertia and saved supplementary projection; actual multiset homogeneity ALS with multiple-nominal, scalar nominal, ordinal and numeric variable blocks. Complete typed category maps, person/set scores and traces persist. | Adjusted MCA inertia, wider weighting/passive/missing-row domains, supplementary OVERALS targets and calibrated sampling inference. OVERALS already supports multidimensional multiple-nominal blocks; do not list them as absent. |
| `mds_nonmetric` | Complete labeled dissimilarities, primary/secondary ties, explicit include/exclude zero-pair rule, isotonic disparities and SMACOF; all observed pairs, stress denominators and converged-start traces persist. | Broader missing/weighted/asymmetric/proximity/individual-difference/unfolding targets and method-specific uncertainty. Multiple starts do not certify a global optimum. |
| `loglinear_ipf`, `loglinear_ml`, `loglinear_compare` | Declared complete Cartesian count grid, explicit structural zeros and fixed log offsets; hierarchical generating margins or full-rank general cell design. Independent-Poisson likelihood, full observed information/covariance and rank-based structural-zero df; nested LR verifies identical counts/support/offsets and actual design containment. | Additional fixed-margin/clustered sampling, exposure uncertainty, small-count exact/resampling inference and wider prediction contrasts. A row-level Poisson regression is not this workflow. |
| `loglinear_multinomial`, `loglinear_product_multinomial`, `loglinear_sampling_compare` | Direct fixed-grand-total or explicitly fixed-stratum conditional likelihood, full identified conditional information/covariance, Normal Wald and asymptotic rank-based GOF/LR. Complete count/support/offset/conditioning equality and actual centered-design nesting are verified; stratum-constant columns and boundary MLEs fail. | Additional fixed margins beyond the declared totals, clustered/survey or weighted laws, sparse/boundary finite-sample calibration and selection-aware inference. Independent-Poisson covariance is not reused. |
| `loglinear_select` | Bounded deterministic backward AIC/BIC path on hierarchical independent-Poisson IPF models, deleting maximal interactions and retaining all main effects. Every eligible candidate, complete fit and decision persists; BIC declares total-person-count N. Candidate failure refuses the search. | Selection-calibrated covariance/tests/intervals and further justified conditional-sampling, global or significance-cutoff search laws. Greedy backward AIC/BIC is already delivered and does not claim exhaustive global selection. |

CATREG/CATPCA admit at most 3,000 supplied rows, 12 analysis variables, 32 observed
levels per categorical variable, 12 starts and 1,000 iterations. MCA/OVERALS use
4–3,000 rows and 2–12 variables; MCA has 1–12 nonzero axes, OVERALS 1–6 dimensions
within its set/scaling rank restrictions. NMDS has 4–150 objects and 1–6 dimensions.
Default planned work is 300 million and the named numerical/input buffer cap is
128 MiB intersected with the global budget. Dimension-valid combinations may
still fail these plans. Loglinear admits 2–6 dimensions, 2–16 levels each, at
most 4,096 cells/128 projected parameters, 10 million counts per cell and one
billion total counts. It has separate bounded work/iteration/offset/design limits.
The added response-scaling route allows 11 predictors plus its categorical
response. Fixed-query bootstrap has 1–64 query rows and 2–499 declared draws;
whole-fit/query/retained-state work and serialization limits apply before
resampling. Saved rotations require 2–min(6,p−1) retained components and keep
their complete source summary. Conditional loglinear fits retain the bounded
complete-cell/parameter domain; backward selection separately allows 2–4
dimensions and a worst-case candidate cap before fitting. These are distinct
admission plans, not permission to combine every maximum.

These are resident native Torch CPU float64 methods; Dataset collection and GPU
are unsupported. Exploratory geometries and learned categorical-response scores
have no ordinary adaptive coefficient covariance/p/CI. The bounded full-refit
numeric-response fixed-query bootstrap has its empirical law above. Loglinear
SE/z/CI and GOF/LR chi-square laws are asymptotic under the declared Poisson or
conditional multinomial sampling scheme; selected-model intervals remain
unadjusted for selection. Saturation has df zero and no GOF p-value. Structural
cells have fitted zero and undefined sampling SE/CI, rather than invented precision.
`summary_state` retains every table, dtype, map, sample and trace; postestimation
checks canonical map/state integrity and semantic geometry before replay. A digest
detects alteration; it does not authenticate a fit or establish causal identity.
Detailed contracts: [optimal scaling](categorical-optimal.md),
[categorical geometry](categorical-scaling.md), [loglinear](categorical-loglinear.md),
[response scaling/bootstrap](categorical-outcomes.md),
[saved rotation](categorical-rotation.md) and
[conditional sampling/selection](categorical-sampling-selection.md).

## Scored full-profile conjoint core (#60)

The eight delivered procedures are `conjoint_plan`, `conjoint_orthogonal`,
`conjoint_diagnostics`, `conjoint_fit`, `conjoint_predict`, `conjoint_holdout`,
`conjoint_importance` and `conjoint_simulate`. Complete declared Cartesian plans
and regular two-level generator fractions have explicit order, rank and aliases;
they do not implement arbitrary mixed-level orthogonal-array search.

Each retained respondent scores every training profile once. Scores join by
profile IDs; duplicate/unknown pairs fail, and `drop_subjects` records whole
incomplete respondents. Discrete sum-zero coding and centered/scaled linear or
ideal-point terms export full raw-scale individual coefficient/utility covariance.
Direct QR uses df = training profiles minus free coefficients. Individual t
inference is conditional iid Gaussian OLS; saved predictions are fitted-mean
intervals, not noisy future responses. Group utilities and importance are
descriptive equal-respondent summaries, without population sampling covariance.
Importance averages individually normalized ranges; normalization of mean
utilities is explicitly a different quantity.

Holdouts refuse training IDs and renamed training factor combinations and cannot
refit utilities. Pearson/tie-adjusted Kendall/MAE/RMSE are descriptive. First-choice,
positive-score BTL and temperature-declared stable logit shares are conditional
preference calculations; they do not estimate a CBC choice model or population
market shares. Unknown numeric levels still fail, and quadratic curvature is not
forced to be an ideal maximum.

The resident CPU float64 domain has 4,096 profiles, 16 attributes, 128 free
coefficients, 128 respondents and 100,000 response cells, with work/workspace
admission and separate quadratic holdout checks. Weights, robust/clustered or
group sampling covariance, Dataset, GPU and rank/sequence input fail explicitly.
`conjoint_save/load` retains every table, sample label/position, coding, anchors,
options and full covariance in an integrity-checked artifact capped at 32 MiB.
Remaining implementation is ranked/sequence and CBC input, automatic arbitrary
mixed-level plan search, broader weighting/robust covariance and population
utility/importance/share inference. See [conjoint.md](conjoint.md).

## Balance, paired inference and randomized marginal targets (#75)

| Delivered procedure | Identified target and uncertainty |
| --- | --- |
| `ebalance` | KL donor weights targeting treated numeric moments relative to optional positive base weights; full-rank strictly interior primal/dual feasibility with a corrected equality certificate. No trimming, effect covariance or causal claim from achieved balance. |
| `cem` | Prespecified bins/typed categorical cells; both-arm strata only. Treated weights 1, controls `n_t/n_c`, excluded rows 0; target is retained treated people. |
| `balance` | Full fixed-weight independent-row HC1 arm-mean covariance, original pooled-SD SMD, ECDF ties/ESS and explicit zero-weight exclusions. This does not propagate estimated-weight uncertainty. |
| `rosenbaum_bounds` | Exact one-sided paired sign-test probability bounds at declared Gamma; zero contrasts conditioned out. The separate signed-rank route is below; neither route substitutes for general confounding or pretrend sensitivity. |
| `paired_randomization` | Independent equiprobable within-pair assignment and a sharp constant additive null must be declared. All `2**pairs` assignments or private-seed MC assignments/statistics persist; exact denominator or MC plus-one p and simulation precision are explicit. Pair labels do not prove randomization. |
| `treatment_cdf` | Randomized marginal `F1-F0` on a prespecified grid; both arm CDFs, full joint indicator covariance and pointwise normal intervals conditional on arm counts. |
| `treatment_quantile` | Marginal inverse-ECDF quantile difference, `ceil(n*q)-1`; independent-arm seeded bootstrap retains every replicate, full arm/effect joint covariance, SE and marginal percentile intervals. No invented p-values. |
| `treatment_rmst` | Prespecified common tau, randomized right-censored KM arm curves, complete risk/event/censor tables and integrated Greenwood joint covariance/pointwise normal intervals. Common observed support and independent within-arm censoring are required. |
| `ovb_sensitivity` | One omitted scalar regressor in full-rank classical homoskedastic OLS with an intercept; all partial-R2 grid cells, original full coefficient covariance, adjusted target t intervals with one df consumed, and equal-strength point-estimate zero tipping. Exact t interpretation requires independent Gaussian homoskedastic linear errors. No significance tipping or augmented/cross-scenario joint covariance is supplied. |
| `evalue` | Deterministic hidden-confounding bound for supplied positive risk ratios and optional closest-to-null paired confidence limits, including protective associations. No OR/HR/log conversion, newly estimated SE or CI. |
| `manski_ate` | Sharp empirical consistency-only ATE identification interval under declared common bounded potential-outcome support; unequal arm proportions and all endpoint-compatible potential-outcome completions persist. No sampling SE/covariance/CI. |
| `lee_bounds` | Declared randomized assignment and increasing/decreasing unit-level monotone selection; fractional empirical tail trimming with shared boundary ties identifies always-selected ATE bounds. All original treatment/selection rows determine rates; missing-row deletion is refused. No sampling SE/covariance/CI or covariate-specific Lee law. |
| `randomization_test` | Declared complete fixed-treated-count equiprobable assignment and a constant additive Fisher sharp null. Full exact universe or private uniform Monte Carlo draws/statistics, plus-one MC p and simulation precision persist. No ATE interval or effect-set inversion. |
| `stratified_randomization` | Independent declared strata with fixed treated counts and both arms per stratum; sample-size weighted within-stratum contrast, complete Cartesian exact universe or private uniform Monte Carlo. Sharp-null inference, without effect covariance/CI. |
| `rosenbaum_rank_bounds` | Exact one-sided paired signed-rank Gamma bounds: averaged tied absolute ranks, zero null-adjusted contrasts conditioned out, full dynamic-programming masses/tails. The independent-pair bounded-odds law is assumed; no Hodges–Lehmann/ATE interval or general two-sided sensitivity claim. |
| `bias_sensitivity` | Externally fixed coordinatewise bias box and fixed contrast of a supplied estimate vector, with the complete joint covariance. Exact support function and conservative pointwise Gaussian interval union; no restrictions estimated from noisy pretrends, HonestDiD or simultaneous inference. |

Population sampling, consistency/SUTVA and the stated assignment law are
assumptions, not facts inferred from the input. Quantile interval validity also
needs continuous outcomes and positive density at targets; ties/zero empirical
variance are reported. Distributional contrasts do not identify individual
counterfactual treatment-effect distributions. CDF/RMST intervals are untruncated;
degenerate variance has undefined z/p. Complete pairs retain assignment identities;
outcome exclusions require the documented assignment-independent availability law.

Inputs are resident CPU float64 with at most 100,000 rows and 64 selected columns;
generic weights, Dataset and GPU are unsupported. Specific entropy base weights
and fixed diagnostic weights have the distinct roles above. Explicit work and
workspace plans include full samples, all assignments/bootstrap draws, covariance
and saved state. Defaults are 100 million work units; resource refusal does not
truncate targets, draws or samples. Entropy tolerance/iterations, Gamma grids,
CEM cuts and MC draws have additional option bounds. Quantile replicates and
target grids are limited by their complete work/buffer plans, not a claimed
universal admissible size. Every typed table, sample label/position, assignment,
assumption and setting persists through `causal_design_save/load` with complete
state/table/artifact checksums and no numerical refit.

PR150's OVB role sample uses explicit retained-row identity and at least two
residual degrees of freedom, with 1–64 strictly increasing strengths per R2
axis in [0,1). Fixed external boxes admit at most 64 terms and 256 increasing
nonnegative scales and require a complete symmetric numerically PSD covariance
without projection. The new complete and stratified assignment procedures refuse
missing-row deletion, preserve exact typed strata and retain the declared
complete assignment law;
resource failure does not silently switch exact enumeration to Monte Carlo.
Lee preserves the full selection denominator. Manski's explicit row deletion
changes its empirical target. These sample rules are distinct from generic
complete-case dropping. The new methods retain every original/supplied covariance,
extremizer, trimming weight, assignment/PMF and declared restriction through the
same typed artifact contract. Identification bounds and E-values do not acquire
sampling intervals merely by being saved.

Remaining sensitivity work includes multiple-confounder and
heteroskedastic/clustered/weighted OVB, significance/benchmark and justified joint
scenario targets; data-adaptive/noisy pretrend restrictions, relative-magnitude
or smoothness restrictions, conditional/hybrid or optimal fixed-length intervals
and simultaneous targets. Sampling uncertainty for Manski/Lee endpoints/effect
sets and covariate-specific Lee are open. Assignment laws beyond independent
paired, complete fixed-count and independent stratified fixed-count designs,
weak-null/design-based average-effect intervals and effect-set inversion remain
open. Observational survival/CDF/quantile adjustment, broader survival targets
and treatment/IV/LATE variants remain substantive stages. Existing cross-fitted
binary-treatment `dmlirm` ATE/ATET and binary-IV `dmliivm` LATE are retained
separately under their own nuisance, overlap, identification and strength guards;
do not list all IRM/LATE as absent. Neither the
new randomized marginal targets nor these existing methods complete the broader LATE options
in the historical catalog. See [causal-design.md](causal-design.md) and
[causal-learning.md](causal-learning.md), [causal-sensitivity.md](causal-sensitivity.md),
[causal-identification-bounds.md](causal-identification-bounds.md) and
[causal-fixed-bias.md](causal-fixed-bias.md). Successor #157 preserves these
remaining domains and does not relist the delivered eight bounded methods.

## Traceability and independent verification

The original notes list CATREG/CATPCA/OVERALS/nonmetric MDS, independent-Poisson
and additional GENLOG sampling, CONJOINT/ORTHOPLAN, community CEM/entropy and
broader sensitivity/LATE/survival/distributional targets. The 2026-10-03/05 report
statuses predate these PRs. They are comparison obligations, not current absence
or evidence of licensed numerical equivalence. The advanced domains above
preserve those original obligations as implementation work.

At its recorded pre-PR162/pre-PR150 snapshot, the original source identity receipt checked
all 4 categorical modules against
`c9e09323cc74e39824fad603966a0fb24b841f6d`, all 5 conjoint modules against
`072e51e42ae140b4e041695a78a31106236d8f8f`, and all 6 causal-design modules
against `24a9865e1a654849f68475189790e43513bfac7e`; every file was byte-identical
at that snapshot. PR150 later extends shared causal sampling/serialization and
assignment code and adds its sensitivity kernels; the earlier six-file identity
claim is historical rather than a current whole-family identity claim.
The 9 accepted causal numerical/resource hashes also matched that snapshot.
PR162 extends the categorical dispatch manifest and adds four numerical modules;
the original `optimal.py`, `scaling.py` and `loglinear.py` bodies are unchanged.
That extension does not turn the earlier four-module identity claim into a
current eight-module or whole-SDK proof. Retained categorical
verification hashes and conjoint full native-artifact archive hash were checked.
Their earlier categorical (internal evidence excluded from this public snapshot),
conjoint (internal evidence excluded from this public snapshot) and
causal (internal evidence excluded from this public snapshot) frozen/installed
native Run/restart receipts keep their own pins and scopes. This audit does not
claim a fresh desktop Run of all current main APIs or a public release.

The distinct PR162 categorical-options evidence (internal evidence excluded from this public snapshot)
retains its accepted source/native pin, independent response/rotation/conditional
likelihood/selection checks, 15 complete JSON/LaTeX pairs, eight observed installed
app Run actions and restart/export evidence. Its exact pin and later source
bridges define that support. It does not certify a new desktop run of the
complete current integration, calibrated post-selection inference, licensed
vendor execution or a public release.

The separate PR150 sensitivity/assignment evidence (internal evidence excluded from this public snapshot)
pins accepted scientific source `af893f68acf0f715fe526d303ed88fdadc4ecd46` and
its later integration bridge. It retains 762 passing scoped acceptance cases,
independent OVB/FWL, potential-outcome completion, fractional trimming,
assignment enumeration, rank-PMF and correlated-box fixtures, eight full
artifacts with 24 tables, isolated-wheel/frozen receipts and observed installed
app Run/quit/relaunch evidence. The dedicated local macOS 26.6.2 ARM64 QA app
does not certify macOS 15, Windows, other devices or a public release. The
documented Torch-version provenance difference in one wheel artifact is retained;
whole-artifact byte identity across different Torch versions is not asserted.
Those source pins and packaged scopes remain distinct from this audit's final
merged source/installed-SDK comparison.

The [new replay example](../examples/audited_eight_categorical_causal.py) exposes
`run(destination)`: all 24 core procedures plus 5 saved projection/comparison
outputs write **29 complete typed result JSONs, 29 full LaTeX files and inputs**.
Every state/table/dtype/order/sample and exact LaTeX reopens without fitting again.
Independent new checks cover complete covariance/geometry, original-row alignment,
held-out no-refit/leakage rules, all assignments/quantile draws and finite-law
boundaries. NumPy/SciPy are development references only, not runtime estimators.

Fresh receipts in
the audit evidence directory (internal evidence excluded from this public snapshot)
record **397 existing family cases plus 23 distinct new audit cases**, all with
zero failures/errors/skips. Case identities are disjoint; historical integration
counts overlap and are not added. This is scoped method validation, not the
full repository, unconditional inference, GPU or licensed Stata/SPSS/EViews parity.
The 420 scoped cases above validate the original categorical/conjoint/causal
core and its audit targets; they are not relabelled as coverage of PR162's
11 additional public helpers or PR150's eight new sensitivity/assignment methods.
Those helpers retain their separate stage
receipts, and current integration/runtime checks identify their own source.
