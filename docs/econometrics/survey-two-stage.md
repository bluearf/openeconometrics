# Two-stage SRSWOR survey inference

`survey_two_stage_design` declares exactly two sequential sampling stages. Stage one selects `n_h` PSUs by SRS without replacement from `N_h` within each declared stratum. Stage two independently selects `m_hi` SSUs by SRS without replacement from `M_hi` within each selected PSU. One physical row represents one distinct sampled SSU. Typed PSU IDs nest within strata; typed SSU IDs nest within PSU/stratum. Labels may repeat across parents. Population-count columns are mandatory, exact positive integers at most 2^53, constant within their corresponding parent. Each stage admits a singleton only when its population count proves census. Stage-two stratification and three or more stages are unsupported.

The derived sampling weight is `(N_h/n_h)*(M_hi/m_hi)`. No arbitrary sampling-weight column, calibration, nonresponse adjustment or global scale is accepted. Totals retain the Horvitz–Thompson interpretation under this declared design. Means are Hájek ratios. Joint ratios and declared-category shares use Taylor linearization; absent category levels remain in the dictionary and full covariance can be singular.

```python
import openecon as oe

design = oe.survey_two_stage_design(data, psu="psu", ssu="ssu", strata="stratum",
                                    population_psu="population_psus",
                                    population_ssu="population_ssus")
means = oe.survey_two_stage_mean(data, design, ["income", "spend"],
                                  domain="eligible", missing="drop")
print(means.to_frame())
restored = oe.SurveyTwoStageResult.model_validate_json(means.model_dump_json())
print(restored.contrast([1., -1.]))
```

The eight estimation APIs are `survey_two_stage_mean`, `survey_two_stage_total`, `survey_two_stage_ratio`, `survey_two_stage_proportion`, `survey_two_stage_regress`, `survey_two_stage_logit`, `survey_two_stage_probit` and `survey_two_stage_poisson`. The four model APIs fit sampling-weighted pseudo-scores using the existing native CPU float64 solver; WLS uses analytic weighted least squares, while nonlinear models use observed-curvature Newton steps with certified separation/rank refusal. Probit uses observed sensitivity rather than an expected-information substitution. Poisson requires exact nonnegative integer responses at most 2^53.

## Both-stage covariance

For a full-row weighted linearized vector `u_hij`, let `U_hi=sum_j u_hij`, `f1_h=n_h/N_h` and `f2_hi=m_hi/M_hi`. The full joint covariance is

```
V1 = sum_h (1-f1_h)*n_h/(n_h-1) * sum_i (U_hi-mean_i U_hi)(...)'
V2 = sum_hi f1_h*(1-f2_hi)*m_hi/(m_hi-1) * sum_j (u_hij-mean_j u_hij)(...)'
V  = V1 + V2
```

Census terms are zero and skip division by a zero singleton denominator. The within-PSU multiplier is `f1_h`: observed between-PSU totals already include second-stage sampling noise. A stage-one census removes `V1` while retaining noncensus `V2`. Only census at both stages guarantees the overall census reduction. For regression, `u` is the full-row normalized weighted pseudo-score and coefficient covariance is `A^-1 V A^-T`, where `A` is the observed weighted sensitivity. Normalizing weights to mean one within the fit sample leaves this sandwich unchanged.

This formula follows equation (2), pages 4–5, of the primary [Stata variance-estimation manual](https://www.stata.com/manuals/svyvarianceestimation.pdf), and the recursive SRSWOR design described by the primary [R survey svyrecvar documentation](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyrecvar.html) and the stage-specific discussion in the [Stata multiple-stage design FAQ](https://www.stata.com/support/faqs/statistics/stratified-multiple-stage-designs/). It does not substitute an ultimate-cluster approximation. Exhaustive finite-population total enumeration verifies the variance estimator's expectation against actual randomization variance; nonlinear Taylor covariance does not claim finite-sample exact coverage.

## Samples, reference inference and saved state

Declaration and revalidation bind all physical rows, nested identities, population counts, dtypes, category ordering and row ordering. Domain indicators must be complete Boolean/0–1 values. Target/model missing values either raise or undergo explicit common listwise exclusion. Their full-row scores become zero; all PSUs and SSUs remain in covariance geometry. Out-of-domain outcomes need not be valid observed values. Complete-case estimation does not infer a missing-data mechanism.

The reference degrees of freedom are complete first-stage PSUs minus strata. This is a declared compatibility convention, neither exact nor guaranteed conservative, particularly with first-stage census and sparse second-stage uncertainty. Positive variance with df0 retains estimates, standard errors and full covariance but leaves statistic, p-value and CI endpoints unavailable. Zero variance produces a point interval and undefined tests. Nonzero-df confidence intervals use unclipped reference Student-t Wald inference.

`SurveyTwoStageResult` and `SurveyTwoStageRegressionResult` use distinct frozen extra-forbid schemas. JSON restore validates the complete sample partition, derived weights, ordered roles, dimensions, primitive estimates/weighted equations, linearized row scores, observed sensitivity and both covariance contributions. Checksums detect changed saved records; they are not authentication of the sampling design. Revalidate declarations against original data before new estimation. Saved descriptive contrasts and model coefficient `lincom`/adjusted joint `test` use full covariance; `predict` gives fixed-covariate fitted means, not future-outcome intervals.

Resident DataFrames and column/row mappings are supported. Dataset, CUDA/MPS, tensor columns, PPS, calibration/poststratification, replication and general DEFF are refused or outside this API. Limits are 32 joint targets or linear parameters, up to one million physical rows, explicit 1–512 MiB declaration budget and the ambient workspace budget, plus 50 million cumulative model work units. Nonlinear fits reserve three bounded separation certificates (fit, canonical score evaluation and saved-state replay); this permits at most seven parameters, including the intercept, subject to sample size and iteration work. Larger declared widths fail the complete-work gate before fitting. Nested indexing, tensors and replay serialization are admitted before allocation. Existing single-stage and replica result schemas remain separate.

The [eight-case executable example](../examples/survey_two_stage_eight.py) saves all states and exports all tables. Source tests, independent oracle, frozen source parity, installed native Run/full restart, licensed-vendor execution and public release are separate evidence levels. These APIs do not assert blanket Stata parity.
