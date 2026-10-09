# Three-stage SRSWOR with SSU strata inside each PSU

This API performs exactly three successive selections without replacement:
PSUs inside optional first-stage strata, SSUs separately inside each declared
positive SSU stratum of each sampled PSU, and TSUs inside each sampled SSU.
One physical row represents one unique terminal TSU. The second-stage strata
are deterministic cells, rather than an additional selection stage.

The caller must supply a complete `ssu_frame` of positive cells for the sampled
PSUs. Each record uses the actual column names of the PSU, SSU-stratum and SSU
population roles, plus the first-stage stratum role when declared. Every record
must have sampled SSUs; omitted, duplicate, extra or inconsistent cells fail.
The table and frame cannot authenticate a cell absent from both supplied inputs.
The declaration certifies their internal agreement, not an external population
frame's completeness or actual sampling history.

```python
import openecon as oe

# Independently known positive cells for the sampled PSUs; continue for every
# sampled PSU and its positive second-stage stratum.
ssu_frame = [
    {"region": "north", "county": "A", "block_type": "urban", "blocks_in_cell": 12},
    {"region": "north", "county": "A", "block_type": "rural", "blocks_in_cell": 8},
    # ... all remaining positive cells for all sampled PSUs ...
]
design = oe.survey_stratified_three_stage_design(
    data, psu="county", ssu_strata="block_type", ssu_frame=ssu_frame,
    ssu="block", tsu="person", strata="region",
    population_psu="counties_in_region", population_ssu="blocks_in_cell",
    population_tsu="people_in_block",
)
result = oe.survey_stratified_three_stage_mean(data, design, ["income", "hours"])
table = result.to_frame()
restored = oe.SurveyStratifiedThreeStageResult.model_validate_json(result.model_dump_json())
```

| Procedure | Target and retained inference |
| --- | --- |
| `survey_stratified_three_stage_mean` | Joint Hájek means and full covariance contrasts |
| `survey_stratified_three_stage_total` | Joint Horvitz–Thompson totals and full covariance contrasts |
| `survey_stratified_three_stage_ratio` | Joint ratios with nonzero weighted denominator totals |
| `survey_stratified_three_stage_proportion` | Declared typed category shares, including absent categories |
| `survey_stratified_three_stage_regress` | Analytic sampling-weighted WLS; prediction, lincom and joint tests |
| `survey_stratified_three_stage_logit` | Binary pseudo-score logit with observed sensitivity |
| `survey_stratified_three_stage_probit` | Binary pseudo-score probit with observed sensitivity |
| `survey_stratified_three_stage_poisson` | Exactly representable nonnegative integer counts |

## Counts, identities and covariance

Population counts are mandatory positive exact integers at most `2**53`.
`N_h` is constant within a first-stage stratum; `M_hig` is constant within an SSU
stratum of a PSU and equals its frame record; `L_higs` is constant within an SSU.
Each population dominates its sampled count. Singletons require census at their
own selection stage or SSU stratum. Counts are population sizes, not fractions.
Weights are derived as `N_h/n_h * M_hig/m_hig * L_higs/l_higs`.

Identities retain the scalar type, including distinct boolean, integer, float
and string labels. SSU and TSU labels may repeat in different declared parent
cells. Every terminal identity `(stratum, PSU, SSU stratum, SSU, TSU)` is unique.
Physical row positions, not display-index labels, determine sample alignment.

Let `u_higsk` be a full-row weighted linearized vector, `U_hi` its sampled PSU
sum, and `U_higs` its sampled SSU sum. With `f1=n/N`, `f2_g=m_g/M_g` and
`f3=l/L`, the three retained matrices are:

```text
V1 = Σ_h      (1-f1)             n/(n-1) Σ_i (U_hi - mean_i U_hi)(...)'
V2 = Σ_hig    f1*(1-f2_g)      m_g/(m_g-1) Σ_s (U_higs - mean_s U_higs)(...)'
V3 = Σ_higs   f1*f2_g*(1-f3)     l/(l-1) Σ_k (u_higsk - mean_k u_higsk)(...)'
V  = V1 + V2 + V3
```

Stage two centers SSU totals separately inside each cell. Stage three uses that
cell's sampling fraction in its prefix. Pooling SSU strata or using a PSU-wide
second-stage fraction produces a different covariance. A census term is
exactly zero and retains any lower noncensus term. One SSU stratum per PSU
reduces to the existing unstratified three-stage equations.

This follows the recursive between/within decomposition described by
[R survey's recursive variance documentation](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyrecvar.html)
and its [multistage implementation](https://raw.githubusercontent.com/cran/survey/master/R/multistage.R),
which centers within the current strata and carries each upper sampling
fraction into the lower recursion. These references describe the calculation;
this scope does not claim an executed vendor or R-package comparison.

Independent NumPy/SciPy tests enumerate complete finite populations and sample
worlds with unequal cell sampling fractions, including all eight census
combinations. For HT totals they compare the expected full covariance estimator
with the actual randomization covariance. Separate fixtures detect pooled-cell
centering and incorrect third-stage prefixes. Means, ratios and proportions
use joint primitive linearization. Models use native scores and observed
sensitivity, including probit's observed curvature. Nonlinear Taylor variance
and reference inference do not imply exact finite-sample coverage.

## Samples, saved states and reference inference

Domain selection and common listwise missing-value selection retain the full
design. Excluded rows contribute zero influences or scores at their original
physical positions, including an entirely excluded PSU or cell. Categories and
coefficient order remain explicit. Poisson counts are admitted before float64
conversion. Rank loss, separation, nonconvergence and invalid complete geometry
fail explicitly.

Reference degrees of freedom are complete first-stage PSUs minus first-stage
strata. This convention is not guaranteed exact or conservative. With zero
degrees of freedom and positive covariance, estimates, standard errors and full
covariance remain available; tests and confidence limits are unavailable.
Zero variance permits point intervals.

`SurveyStratifiedThreeStageResult` and
`SurveyStratifiedThreeStageRegressionResult` have separate new schemas.
Restoration admits dimensions before nested numeric conversion, validates the
canonical typed frame and complete parent/row partitions, and replays saved
primitives, sample selection, rank/convergence evidence, scores/sensitivity and
all three covariance components. It performs no fit optimization. Provenance
hashes bind accepted supplied content and do not authenticate external data or
sampling.

Saved contrast and `lincom` use full joint covariance. Joint coefficient
restrictions use the admitted adjusted design F reference. Prediction is
conditional on explicit fixed numeric covariates, retaining full coefficient
uncertainty and physical missing-row alignment; it excludes future-response
and random evaluation-distribution uncertainty. Tables retain full covariance
attributes and can be exported to LaTeX.

## Admission and remaining scope

This is a resident scalar DataFrame or row/column mapping route with Torch CPU
float64. `max_rows` is at most one million. `max_memory_mb` is 1–512, default 64,
and the global workspace ceiling also applies. Complete frame/identity,
hierarchy, target, score and serialized covariance storage are admitted before
large allocation. These are implementation/resource limits, not statistical
sample-size requirements. Joint descriptive and linear dimensions are at most
32; nonlinear dimensions are at most seven, subject to a complete
50-million-unit fit/replay budget. Prediction admits at most 256 rows.

Four or more selection stages, third-stage stratification, PPS, arbitrary
supplied weights, calibration/poststratification, general design effects,
replication in this API, additional model families, Dataset/streaming and
GPU/MPS remain unsupported. Broader parent survey scope remains open.
Licensed vendor execution, signing/notarization and public release are separate
gates. No blanket software parity is claimed.

The [eight-method synthetic example](../examples/survey_stratified_three_stage_eight.py)
uses 48 rows, six PSUs, twelve SSU strata and twenty-four SSUs. Source tests,
source/native equality, an actual native Run, full restart persistence, required
hosted CI, protected merge and tracker readback are separate acceptance layers.
