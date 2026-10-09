# Survey and measurement acceptance audit, 2026-10-07

The audit starts from main `45d6b64f75b6d43a83cfa6bdc84be6fbffaafa8e` and the
original GitHub #30 and #57 acceptance bodies. Historical research plans are
comparison inventories, not current implementation receipts. In particular,
`research_notes/Ekonometrik modeller Stata EViews SPSS kataloğu/spss_official_procedures.md`
distinguishes SPSS Complex Samples declaration/descriptive/regression procedures;
the sibling `spss_extensions_amos_gaps.md` distinguishes IRT and ordinal
correlation extensions from built-in commands. The 2026-10-03/05
`reports/OpenEcon ekonometri tam ekleme planı.md` deferrals have been checked
against current source rather than reused as present-day missing-method claims.
The survey section is reconciled with the single-stage regression delivery in
PR #129, incorporated into main `c041f3f`; the audit's original source receipt
retains its earlier snapshot and scope.

## Retained scientific and native evidence

The module receipt (internal evidence excluded from this public snapshot)
checks source equality at the audit's starting revision against the retained
scientific frozen/native receipts: ten survey/dependency modules, eight
IRT/dependency modules and twelve reliability/dependency modules match. These
lists overlap and their counts are not an invented unique-module total.
The current survey manifest adds eight regression/postestimation exports and
the current IRT manifest adds three diagnostic exports. Those changed manifests
do not match the old pins, and new modules are outside those historical
receipts. The retained survey numerical/dependency modules and IRT
fitting/scoring sources remain unchanged. Survey regression has its own
source, frozen and installed native evidence (internal evidence excluded from this public snapshot)
at its declared revisions. The new IRT diagnostic module has its own
source/package acceptance and is outside the historical installed IRT QA
application.

`tests/test_next_eight_survey_measurement.py` replays complete retained artifacts
through current source with estimator calls disabled: eight survey states,
six IRT model states and eight reliability states. It verifies saved joint
contrasts including off-diagonal covariance and Student t inference, exact IRT
tables/LaTeX/person-score alignment, and every retained reliability subject
jackknife draw/covariance. No old receipt is relabelled as a newly executed
whole-main desktop build. The source, licensed vendor, wheel, frozen-runtime,
actual native Run/export/restart and public release proof layers stay separate.

## Survey #30

Current support includes a complete resident single-stage declaration, eight
descriptive/replicate helpers and eight regression/saved-result procedures.
The [declaration](survey-design.md) and
[inference contracts](survey-inference.md) define typed PSU/stratum identity,
exact weights/row revalidation, integer first-stage FPC, explicit certainty
singleton policy, joint missing/domain geometry, complete covariance, survey
degrees of freedom and checked persistence. Taylor means/totals/ratios/fixed
categorical proportions, two-PSU BRR/Fay, whole-PSU stratified jackknife and
supplied justified first-stage bootstrap weights have independent oracles and
retained native proof. Restricted equal-weight SRSWR DEFF is explicitly named.

The [single-stage regression contract](survey-regression.md) now provides
`survey_regress`, `survey_logit`, `survey_probit` and `survey_poisson`, plus
`survey_predict`, `survey_margins`, `survey_lincom` and `survey_test`. These
numeric resident CPU float64 models use full stratified-PSU Taylor score
covariance with the declared first-stage FPC and design df `PSUs - strata`.
Every original PSU remains in the covariance geometry after domain/listwise
exclusions, whose row scores are zero. Probit uses observed score sensitivity;
Poisson admits exact nonnegative integer counts. Complete saved fits retain
coefficients, full covariance, design/sample identity, convergence and PSU
score/sensitivity records for checked replay without refitting. Coefficient
contrasts use design t; joint restrictions use the declared adjusted Wald F.

Prediction and continuous average marginal effects use full coefficient-delta
covariance with fixed evaluation covariates and standardization weights.
Unconditional margins, uncertainty in the empirical evaluation distribution,
categorical/discrete contrasts and future-outcome prediction noise remain
outside these procedures. The linked contract defines identification,
separation, convergence, coefficient-count and work/memory limits; successful
single-stage estimation does not imply support outside those bounds.

The original #30 still requires multistage declaration, stage weights/FPC and
multistage reference cases. Calibration/poststratification, general
unequal-weight DEFF/DEFT, calibrated unit-varying replicates and
replicate-regression covariance also remain separate stages. Additional GLMs,
Dataset/device routes and licensed vendor executable acceptance retain their
own gates. #30 stays open for these concrete remaining requirements; its
single-stage regression Taylor stage is now implemented.

## Measurement #57: delivered core and new diagnostics

The [IRT contract](irt.md) covers six standard-normal unidimensional MML
families: fixed-slope Rasch, positive-slope 2PL, supplied-fixed-guessing 3PL,
graded response, fixed-slope partial credit and fixed-slope rating scale.
They retain full observed information/covariance, category/person/sample
alignment, EAP/posterior SD and complete portable curves. The
[measurement contract](measurement-reliability.md) separately covers two-step
latent-normal polychoric/polyserial correlation, one-factor ML omega total,
six balanced ICC variants and Cohen/Fleiss/Krippendorff/Gwet agreement. These
are distinct from CFA/SEM and from latent-group invariance.

This batch adds three lazy public helpers. See the
[runnable example](../examples/next_eight_irt_diagnostics.py).

| Helper | Accepted inference and sample |
| --- | --- |
| `irt_fit_diagnostics(result, data=None)` | All six saved IRT families; original complete fitted people only; EAP expected score/variance, item/test descriptive mean squares, item-pair residual Q3 and adjusted Q3 |
| `irt_mh_dif(data, items, group, reference, focal, match)` | Binary items; explicitly named two-group universe and prespecified observed integer matching score; common reference/focal correct-response odds, CMH chi-square(1) approximation and RBG log-odds intervals |
| `irt_diagnostics_restore(result)` | Full checked state replay and regenerated complete tables without item/person/model estimation |

### Fit diagnostics

For each fitted person, expected category score and variance are evaluated at
the saved EAP. Residual is observed minus expected score. Infit is
`sum(residual**2)/sum(conditional_variance)`; outfit is the mean of
`residual**2/conditional_variance`. Test-score outfit uses summed item residuals
and summed conditional item variances under local independence. Q3 is the
Pearson correlation of person residuals; adjusted Q3 subtracts the average
off-diagonal pair correlation. All residual rows retain original person
position/index and the complete sample table retains excluded people/missing
counts. Supplied source data must match the exact original selected-item hash,
indices, row positions and responses. A different source is rejected.

These are **descriptive EAP conventions**. Item/person parameters use the same
responses; residual correlations have estimation bias. No chi-square p,
reference degrees of freedom, universal Q3 threshold, parameter uncertainty
or bootstrap calibration is fabricated. The
[Christensen, Makransky and Horton study](https://doi.org/10.1177/0146621616677520)
shows why Q3 calibration depends on the actual data/model configuration.
Refit-parametric-bootstrap fit calibration remains a focused followup.
Conditional variance or residual variance below the explicit numerical floor
fails rather than producing unstable standardized values.

### Observed-score Mantel-Haenszel DIF

The caller supplies the prespecified matching-score column. The helper never
selects, purifies or adjusts the matching test, and never estimates latent
group means or item parameters. Exactly the declared reference/focal labels
are admitted. Selected items must be numeric 0/1, scores integer 0..256,
and missing handling is explicit joint `raise` or `drop`. Resident 3..100000
independent unweighted people, 1..16 items and up to 257 matching strata are
admitted under the global workspace budget. Dataset/device/weighted/clustered
people and polytomous DIF are outside the route. Typed group labels remain
distinct after the caller's DataFrame conversion; earlier dtype coercion
cannot be reversed.

Each saved 2x2 table is ordered as reference-correct, reference-incorrect,
focal-correct, focal-incorrect. The pooled odds ratio is
`sum(a*d/n)/sum(b*c/n)`, with `delta_MH=-2.35*log(odds)`. The conditional score
uses the fixed-margin hypergeometric expectation and variance. `continuity`
explicitly chooses the half-unit correction `max(0, abs(sum(a-E(a)))-.5)`;
the uncorrected option is also available. P-values use the chi-square(1)
approximation. Log-odds standard errors/normal Wald intervals reuse the
already validated native Robins-Breslow-Greenland kernel. No pseudo-counts are
added. Zero group/response margins remain in the complete strata table with
zero information and an explicit reason. No finite pooled odds or no
conditional variance produces an error. Sparse-cell asymptotic accuracy is
not asserted by successful arithmetic.

The common-odds summary assumes its declared common conditional association;
this route does not test nonuniform DIF, prove latent invariance or infer
causal item bias. Multiple items receive unadjusted tests with no automatic
multiplicity classification. The method distinction follows the primary
[Holland–Thayer report](https://www.ets.org/research/policy_research_reports/publications/report/1986/hwnk.html).
The RBG derivation is separately documented by
[Silcocks](https://doi.org/10.1186/1742-5573-2-9).

### Complete portable state

`IRTDiagnostics` extends the ordinary TableSet with `.to_json()`.
`openecon.irt.diagnostics.v1` retains the complete saved fitted IRT state for
fit diagnostics, or every item/stratum count, source hash, named groups,
matching levels, options and physical sample/exclusion identities for DIF.
The checksum and complete output-table digest reject accidental drift.
`irt_diagnostics_restore` validates the envelope, sample/count/group geometry,
and replays every fit quantity or conditional test/pooled uncertainty. It
performs no estimation. JSON is limited to 16 MiB; workspace plans account for
full residual/count/sample output before generation. All kernels are locally
CPU float64 even under another default Torch device. The checksum is an
integrity check, not authenticated sampling provenance.

New independent tests compare all six fit diagnostics to separately written
NumPy natural-category formulas, Q3 and mean-square calculations. DIF checks
use SciPy fixed-margin hypergeometric moments and statsmodels stratified RBG
uncertainty only as development oracles. Group/item-complement symmetry,
original source drift, exact persistence, duplicate indices, joint missing
alignment, zero-margin strata, unsupported input, portable tampering and
workspace/default-device guards are exercised. No external runtime estimator,
licensed vendor executable or desktop UI run is used for these new helpers.

Free common-discrimination 1PL, free-guessing 3PL, generalized partial credit,
multi-group/multidimensional models, parameter-uncertainty-aware scoring,
polytomous/nonuniform DIF, formal fit calibration, hierarchical/multifactor
omega and broad invariance remain explicit advanced stages. Existing fixed
identification is not renamed as those broader models.

## Mixed-model #38 audit

At the audit's starting revision, the newer `mixedflex` implementation admits
up to eight nested/crossed grouping levels and multiple correlated lowest-level
slopes. Its residual covariance is still a scalar variance times identity, and
the registry then provides no Satterthwaite/Kenward–Roger df/covariance method.

This batch separately adds `mixed_satterthwaite` for scalar model-based contrasts
of saved numeric unweighted base `mixed` ML/REML fits. Independent tests in
`tests/test_next_eight_mixed_satterthwaite.py` check balanced REML intercept
df=`G-1`, centered within-group slope df=`G*(T-1)-1`, and unbalanced correlated
random-slope GLS/profile-information matrices constructed separately in NumPy.
The balanced exact-law statements apply only to their specified orthogonal
Gaussian geometry; general df and t inference remain approximations. The
routine checks saved estimates/uncertainty and never refits the model. These
new source checks do not add a licensed vendor run or native desktop receipt.

Repeated residual structures, Kenward–Roger covariance/df, joint multi-df
contrasts and small-cluster coverage calibration remain separate requirements.
The new scalar helper is not a generic `mixedflex`/GLMM df method. Remaining
broader nonlinear/GLMM/prediction domains retain their own acceptance stages.
