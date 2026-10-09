# Exactly four-stage SRSWOR survey inference

MARKET-707–714 add joint mean, total, ratio, typed proportion, sampling-weighted
WLS, logit, observed-Hessian probit and Poisson. The public declaration is
`oe.survey_four_stage_design`; see the [complete executable example](../examples/survey_four_stage_eight.py).

Four real selections are PSU → SSU → TSU → FSU. First-stage strata are optional;
the three lower stages are unstratified within each sampled parent. A physical
row represents one unique sampled FSU, while a sampled TSU can contain multiple
FSUs. Typed identities distinguish Boolean, integer, float and string labels
and are nested within their actual parents. Labels may repeat across parents.
Mandatory population columns declare exact positive integers at most 2**53,
constant inside the relevant parent, and at least the sampled number of children.
A singleton is admitted only as an explicitly declared census at that stage.
All stages and ordered physical positions persist in a separate immutable
`SurveyFourStageDesign`; supplied population declarations are caller inputs,
and a checksum cannot authenticate their external sampling provenance.

## Estimation and covariance

Let the four actual sampling fractions be `f1=n/N`, `f2=m/M`, `f3=l/L`,
`f4=k/K`, with the appropriate parent-specific counts. Row weights are
`N/n * M/m * L/l * K/k`. The covariance uses the complete weighted linearized
row vectors `u`, including zeros for excluded outcomes and domain nonmembers.
For a block of `r` sampled children with fraction `f`, define
`C(block)=(1-f)*r/(r-1)*sum (u_i-mean(u)) (u_i-mean(u))'`, interpreted as zero
for a census, including a census singleton. The four contributions are:

- V1: centered PSU totals within each first-stage stratum, prefix 1.
- V2: centered SSU totals within each sampled PSU, prefix f1.
- V3: centered TSU totals within each sampled SSU, prefix f1*f2.
- V4: centered physical FSU rows within each sampled TSU, prefix f1*f2*f3.

Every parent uses its own fractions; an earlier census zeroes that stage's
component while retaining later noncensus uncertainty. This is the four-stage
extension of conditional recursive SRSWOR variance decomposition described in
[Stata's variance estimation manual, equation 2](https://www.stata.com/manuals/svyvarianceestimation.pdf).
The implementation is separately checked by exhaustive randomization of a
31-FSU population across all 16 stage-census combinations, comparing expected
reported full covariance with actual HT-total randomization covariance. That
exact HT check does not imply exact nonlinear Taylor variance or interval coverage.

Totals use Horvitz–Thompson influences; means use Hájek normalization; paired
ratios retain primitive numerator/denominator vectors and fail for zero weighted
denominators. Proportions retain ordered typed categories, one-hot primitives,
absent levels and full joint covariance. Models use native float64 likelihood
scores and full observed sensitivity, including observed Hessian for probit.
WLS is a sampling-weighted model, not analytic or frequency weight inference.
Binary separation, rank loss, failed finite convergence and invalid Poisson
counts fail explicitly. No SciPy or statsmodels numerical solver ships in this
runtime; NumPy/SciPy comparisons are development-only independent evidence.

The reference df is complete first-stage sampled PSUs minus strata. This is an
explicit reference convention; it does not guarantee exact or conservative
coverage. With df zero and positive variance, standard errors remain available
but tests and confidence intervals do not. Zero variance yields a point interval.
Student-t intervals are unclipped. Domain/listwise selection retains every
original PSU, SSU, TSU and FSU position, even an entirely zero-influence parent.
Never subset the design before computing covariance.

## Saved state, resources and boundaries

`SurveyFourStageResult` and `SurveyFourStageRegressionResult` have separate v1
schemas and method `four-stage-taylor`. Restore validates geometry, dimensions,
scientific metadata and primitive target/native-score replay, then recomputes
all four stage contributions and full covariance. A coherently rehashed altered
coefficient, bread, score, target or sampling component is still rejected.
Descriptive contrasts and model `lincom`, adjusted joint design-F tests and fixed
numeric conditional `predict` use full saved covariance. Prediction preserves
physical missing-row alignment and does not include new response noise or
sampling-design estimation uncertainty for the new rows.

CPU float64 resident admission charges 24,576 bytes per complete geometry row,
plus bounded target/model buffers and native work. The default memory budget is
64 MiB; caller budgets are 1–512 MiB and remain subject to the global resource
ceiling. `max_rows` cannot exceed one million, but the memory/work budget often
admits substantially fewer resident rows. These are implementation resource
guards, not statistical or algorithmic limits. Joint targets and linear
parameters are capped at 32; nonlinear parameters at seven including an
intercept. Native fitting and replay have a 50-million-operation guard, and
fixed conditional prediction admits at most 256 rows. Guards run before numeric
allocation, including raw saved-state dimensions and hostile meta-tensor inputs.

When every sampled TSU contains one FSU with K=k=1, V4 vanishes and the same
estimands reduce to the existing three-stage methods. A fourth-stage census
with several FSUs per TSU only removes V4; aggregating those rows does not in
general preserve mean/model estimands.

Five or more selections, lower-stage strata, PPS, arbitrary supplied weights,
additional singleton policies, calibration/poststratification, general DEFF,
replicate covariance, broader model families, Dataset/streaming, CUDA/MPS and
licensed vendor parity remain outside this bounded scope. MARKET-211 stays open
for those remaining acceptances. Installed local QA is separate from public
release, Developer ID signing or notarization.
