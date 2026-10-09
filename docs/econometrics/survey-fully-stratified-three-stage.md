# Three-stage SRSWOR with strata at both lower stages

This API performs exactly three successive selections without replacement:
PSUs inside optional first-stage strata, SSUs separately inside each declared
positive SSU stratum of each sampled PSU, and TSUs separately inside each
positive terminal stratum of each sampled SSU. One physical row represents one
unique terminal TSU. The two lower sets of strata are deterministic cells and
do not add a fourth selection stage.

The caller must supply complete positive-cell frames for observed parents:
`ssu_frame` for every sampled PSU and `tsu_frame` for every sampled SSU. Records
use the actual role-column names. Stage-two records contain PSU, SSU stratum
and SSU population, plus the first-stage stratum when declared. Terminal records
also contain SSU and TSU stratum, with that terminal cell's TSU population.
Every declared positive cell must have sampled units. Omitted observed cells,
duplicates, extra unobserved parents, inconsistent populations and cells with
no sampled units fail.

The table and frames cannot authenticate a real cell absent from all supplied
inputs. The declaration certifies their internal agreement and typed parent
partitions, not an external population frame's completeness or actual sampling
history.

```python
import openecon as oe

# Supply every positive lower cell for all observed parents from independently
# known population frames; these records show one parent of each kind.
ssu_frame = [
    {"region": "north", "county": "A", "block_type": "urban", "blocks_in_cell": 12},
    # ... all positive SSU cells for every sampled county ...
]
tsu_frame = [
    {"region": "north", "county": "A", "block_type": "urban", "block": "B1",
     "age_group": "adult", "people_in_cell": 40},
    {"region": "north", "county": "A", "block_type": "urban", "block": "B1",
     "age_group": "child", "people_in_cell": 15},
    # ... all positive TSU cells for every sampled block ...
]
design = oe.survey_fully_stratified_three_stage_design(
    data, psu="county", ssu_strata="block_type", ssu_frame=ssu_frame,
    ssu="block", tsu_strata="age_group", tsu_frame=tsu_frame,
    tsu="person", strata="region", population_psu="counties_in_region",
    population_ssu="blocks_in_cell", population_tsu="people_in_cell",
)
result = oe.survey_fully_stratified_three_stage_mean(data, design, ["income", "hours"])
table = result.to_frame()
restored = oe.SurveyFullyStratifiedThreeStageResult.model_validate_json(result.model_dump_json())
```

| Procedure | Target and retained inference |
| --- | --- |
| `survey_fully_stratified_three_stage_mean` | Joint Hájek means and full covariance contrasts |
| `survey_fully_stratified_three_stage_total` | Joint Horvitz–Thompson totals and full covariance contrasts |
| `survey_fully_stratified_three_stage_ratio` | Joint ratios with nonzero weighted denominator totals |
| `survey_fully_stratified_three_stage_proportion` | Declared typed category shares, including absent categories |
| `survey_fully_stratified_three_stage_regress` | Analytic sampling-weighted WLS; prediction, lincom and joint tests |
| `survey_fully_stratified_three_stage_logit` | Binary pseudo-score logit with observed sensitivity |
| `survey_fully_stratified_three_stage_probit` | Binary pseudo-score probit with observed sensitivity |
| `survey_fully_stratified_three_stage_poisson` | Exactly representable nonnegative integer counts |

## Counts, identities and covariance

Population counts are mandatory positive exact integers at most `2**53`.
`N_h` is constant within a first-stage stratum; `M_hig` is constant within an SSU
stratum of a PSU and equals its frame record; `L_higsq` is constant within a
TSU stratum of an SSU and equals its terminal frame record. There is no pooled SSU-level TSU population.
Each population dominates its sampled count. Singletons require census at their
own selection stage or lower-stage stratum. Counts are population sizes,
not fractions.
Weights are derived as `N_h/n_h * M_hig/m_hig * L_higsq/l_higsq`.

Identities retain the scalar type, including distinct boolean, integer, float
and string labels. SSU and TSU labels may repeat in different declared parent
cells. Every terminal identity
`(stratum, PSU, SSU stratum, SSU, TSU stratum, TSU)` is unique.
Physical row positions, not display-index labels, determine sample alignment.

Let `u_higsqk` be a full-row weighted linearized vector, `U_hi` its sampled PSU
sum, and `U_higs` its sampled SSU sum. With `f1=n/N`, `f2_g=m_g/M_g` and
`f3_q=l_q/L_q`, the three retained matrices are:

```text
V1 = Σ_h      (1-f1)             n/(n-1) Σ_i (U_hi - mean_i U_hi)(...)'
V2 = Σ_hig    f1*(1-f2_g)      m_g/(m_g-1) Σ_s (U_higs - mean_s U_higs)(...)'
V3 = Σ_higsq  f1*f2_g*(1-f3_q) l_q/(l_q-1) Σ_k (u_higsqk - mean_k u_higsqk)(...)'
V  = V1 + V2 + V3
```

Stage two centers SSU totals separately inside each stage-two cell. Each SSU
total aggregates all its terminal cells. Stage three centers physical rows
separately inside each terminal cell and carries its actual parent cell's
second-stage sampling fraction. Pooling terminal strata invents between-cell
variation; a pooled SSU-level terminal fraction also changes covariance.
Treating terminal strata as extra independently selected SSUs changes stage-two
covariance and is not a general reduction to the previous API.

A census term is exactly zero and retains any lower noncensus term. One terminal
stratum per SSU reduces exactly to the existing second-stage-stratified family.
One cell at both lower stages reduces to the unstratified three-stage equations.

This follows the recursive between/within decomposition described by
[R survey's recursive variance documentation](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyrecvar.html)
and its [multistage implementation](https://raw.githubusercontent.com/cran/survey/master/R/multistage.R),
which centers within the current strata and carries each upper sampling
fraction into the lower recursion. These references describe the calculation;
this scope does not claim an executed vendor or R-package comparison.

Independent NumPy/SciPy tests enumerate complete finite populations and sample
worlds with unequal cell sampling fractions, including all eight census
combinations. The completely noncensus finite population has 243 sample worlds,
with unequal second-stage and terminal-cell fractions. For HT totals they
compare the expected full covariance estimator
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

`SurveyFullyStratifiedThreeStageResult` and
`SurveyFullyStratifiedThreeStageRegressionResult` have separate new schemas.
Restoration admits dimensions before nested numeric conversion, validates
both canonical typed frames and complete parent/row partitions, and replays saved
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

Four or more selection stages, PPS, arbitrary supplied weights, calibration/poststratification, general design effects,
replication in this API, additional model families, Dataset/streaming and
GPU/MPS remain unsupported. Broader parent survey scope remains open.
Licensed vendor execution, signing/notarization and public release are separate
gates. No blanket software parity is claimed.

The [eight-method synthetic example](../examples/survey_fully_stratified_three_stage_eight.py)
uses 96 rows, six PSUs, twelve SSU strata, twenty-four SSUs and forty-eight
terminal strata. Its domain excludes an entire PSU and a terminal cell; missing
outcomes retain original physical design geometry. Source tests,
source/native equality, an actual native Run, full restart persistence, required
hosted CI, protected merge and tracker readback are separate acceptance layers.
