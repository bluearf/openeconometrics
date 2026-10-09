# Three-stage SRSWOR inference

This API adds eight distinct methods under exactly three successive simple random
selections without replacement. First-stage PSUs are selected within optional
strata, SSUs within each selected PSU, and TSUs within each selected SSU. One
physical row represents one unique terminal TSU. Existing single-stage and
two-stage declarations and saved states retain their own contracts.

```python
import openecon as oe

design = oe.survey_three_stage_design(
    data, psu="county", ssu="block", tsu="person", strata="region",
    population_psu="counties_in_region",
    population_ssu="blocks_in_county",
    population_tsu="people_in_block",
)
result = oe.survey_three_stage_mean(data, design, ["income", "hours"])
table = result.to_frame()
saved = result.model_dump_json()
restored = oe.SurveyThreeStageResult.model_validate_json(saved)
```

| Procedure | Target and saved inference |
| --- | --- |
| `survey_three_stage_mean` | Joint Hájek means; full covariance and contrasts |
| `survey_three_stage_total` | Joint Horvitz–Thompson totals; full covariance and contrasts |
| `survey_three_stage_ratio` | Joint ratios; nonzero denominators, full covariance and contrasts |
| `survey_three_stage_proportion` | Declared typed category shares, including absent categories |
| `survey_three_stage_regress` | Analytic sampling-weighted WLS; prediction, lincom and joint tests |
| `survey_three_stage_logit` | Binary pseudo-score logit with observed sensitivity |
| `survey_three_stage_probit` | Binary pseudo-score probit with observed sensitivity |
| `survey_three_stage_poisson` | Exactly representable nonnegative integer counts |

## Declaration and variance

Population counts are mandatory positive exact integers no greater than `2**53`,
constant within the corresponding parent, and at least the number of sampled
children. A sampled singleton must be a declared census at that stage. Counts
are not fractions or independently supplied weights. Row weights are derived as
`N_h/n_h * M_hi/m_hi * L_hij/l_hij`.

For full-row weighted linearized vectors `u_hijk`, let `U_hi` and `U_hij` be
their complete sampled PSU and SSU sums. With `f1=n/N`, `f2=m/M`, `f3=l/L`, the
three retained covariance components are:

```text
V1 = Σ_h    (1-f1)       n/(n-1) Σ_i (U_hi - mean_i U_hi)(...)'
V2 = Σ_hi   f1*(1-f2)    m/(m-1) Σ_j (U_hij - mean_j U_hij)(...)'
V3 = Σ_hij  f1*f2*(1-f3) l/(l-1) Σ_k (u_hijk - mean_k u_hijk)(...)'
V  = V1 + V2 + V3
```

Census terms are exactly zero; their lower noncensus terms remain. The third
term's prefix is `f1*f2`, because upper-level variability already contains
lower-level sampling noise. This is the three-level extension of the recursive
between/within decomposition documented by [R survey](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyrecvar.html)
and the [Stata variance manual, equation 2](https://www.stata.com/manuals/svyvarianceestimation.pdf).
The independent finite-population tests enumerate complete three-stage sample
worlds and check the total variance estimator's expectation against actual
randomization variance, including each stage's census combination.

Means, ratios and proportions use joint primitive linearization. Regressions
use full coefficient influences from the native score and observed sensitivity
at the returned coefficients. Probit retains observed curvature. Poisson count
admission occurs before conversion to float64. Rank loss, separation,
nonconvergence and invalid complete design geometry fail explicitly.

## Samples, inference and persistence

Common listwise/domain selection retains every original PSU, SSU and TSU. An
out-of-domain or missing row supplies zero target/score influence, retaining its
original physical place in all three variance components. Explicit typed
category dictionaries and coefficient order are retained.

The reference degrees of freedom are complete first-stage PSUs minus strata.
This is a declared reference convention, neither exact nor guaranteed
conservative. Positive covariance with df zero retains estimates, standard
errors and full covariance, while test statistics, p-values and interval
endpoints are unavailable. Zero variance permits point intervals. Nonlinear
Taylor covariance is not an exact finite-sample coverage result.

`SurveyThreeStageResult` and `SurveyThreeStageRegressionResult` are separate
typed schemas. Restoration validates dimensions before numeric parsing, then
replays complete hierarchy, physical sample, primitive target values or selected
X/y, fit rank/convergence, scores/sensitivity, all three stage matrices and full
covariance. It performs no model optimization. Content digests record
provenance and integrity, rather than authenticating sampling or data origin.

Saved prediction is conditional on explicit fixed numeric covariates, with full
joint coefficient-delta uncertainty and physical missing-row alignment. It
does not include future response or random evaluation-distribution uncertainty.
`lincom` uses the full covariance; joint restrictions use the declared adjusted
design F reference, admitting its required rank and degrees of freedom.

## Admission and remaining scope

Only resident scalar DataFrames or row/column mappings and native Torch CPU
float64 are supported. The global workspace limit also applies to the explicit
`max_memory_mb` (1–512); `max_rows` is at most one million. Full hierarchy and
serialized score/covariance storage are admitted before large arrays are built.
Joint descriptive/linear dimensions are at most 32; nonlinear dimensions are
at most seven, subject to a complete 50-million-unit fit and replay budget.
Prediction admits at most 256 rows.

Four or more stages, second/third-stage stratification, PPS, arbitrary supplied
weights, calibration/poststratification, general design effects, replication in
this API, additional models, Dataset/streaming and GPU/MPS routes remain
unsupported. Licensed vendor execution, signing/notarization and public release
are separate acceptance gates. No blanket software parity is claimed.

The eight-method synthetic example is
[`survey_three_stage_eight.py`](../examples/survey_three_stage_eight.py).
