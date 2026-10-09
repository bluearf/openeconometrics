# Full empirical survey target replication

`survey_margins_replicate` fits a numeric logit, probit or Poisson model and
replicates the complete empirical target. It accepts the same admitted
single-stage PSU design and BRR, Fay, generated stratified delete-one PSU
jackknife, or supplied whole-PSU bootstrap plans as coefficient replication.
The eight acceptance gates are four methods crossed with two joint targets:
partial-profile standardized predictive means and continuous average marginal
effects (AMEs). All three nonlinear families belong to each gate.

## Targets and variance

For a declared profile `a`, replace a nonempty proper subset of fitted numeric
regressors with its fixed values; leave the remaining covariates at their
observed admitted values. The standardized mean is

`theta[a] = sum_i w[i] h(X[i; a] beta) / sum_i w[i]`.

For a continuous regressor `j`, the empirical AME is

`theta[j] = beta[j] sum_i w[i] h'(X[i] beta) / sum_i w[i]`.

Every replica first fits `beta[r]` using its actual whole-PSU weights and then
uses those same weights to average its target. Thus both the estimated model
and the empirical covariate distribution vary. The original domain/listwise
sample is admitted once; zero replica weights remove fitting and averaging
support, while the complete original PSU/stratum universe remains intact.
No replica is discarded after a fit or target failure.

The saved joint covariance is `sum_r multiplier[r] d[r] d[r]'`, including
cross-profile or cross-variable entries. Differences use the original target,
the joint replica mean, or the corresponding jackknife stratum mean as
explicitly declared. Method scales, Fay rho, jackknife FPC and replica df
retain the coefficient-plan contract. Output contains marginal design-t tests
and intervals with the declared scalar or ordered vector null. Intervals are
not simultaneous and are not clipped to response bounds.

The existing `survey_margins` remains a separate saved-coefficient delta
calculation conditional on fixed evaluation covariates. It is not used to
approximate the new target replication. Ordinary unmodified fitted means with
an intercept can equal existing descriptive weighted means, and linear AMEs
are coefficients; these identities are not counted as new methods here.

## API

```python
import openecon as oe

result = oe.survey_margins_replicate(
    data, design, "response", ["exposure", "age"],
    family="logit", method="fay", rho=0.5, target="mean",
    profiles={"lower": {"exposure": 0.0}, "higher": {"exposure": 1.0}},
)
console.display(result.to_frame())
restored = oe.SurveyReplicateMarginsResult.model_validate_json(result.model_dump_json())

# For target="ame", specify variables=["exposure", "age"] and omit profiles.
```

Profile order and AME variable order determine covariance order. Profiles must
leave at least one empirical regressor unfixed. Names are distinct and bounded;
at most 32 targets are admitted. AMEs are continuous derivatives of the
explicit numeric design columns; no categorical changes, formula expansion,
interaction chain rules or automatic causal interpretation are supplied.
Bootstrap requires the caller's explicit design justification and scale,
with optional rscales and df. The justification is recorded, not authenticated.

## Saved state and bounds

`SurveyReplicateMarginsResult` embeds the coefficient result and preserves the
admitted design matrix/physical sample, full original and actual replica
weights, whole-PSU factors, original and replica targets, complete covariance,
and inference options. Restoration checks the geometry, supports and hashes,
re-evaluates targets from stored coefficients and primitive inputs, and
replays the complete covariance without refitting. Keeping actual weights is
necessary because accepted supplied factors may differ within numerical
whole-PSU tolerance. Content hashes detect inconsistent changes; they do not
authenticate the original data or establish independent fitting provenance.
The original design declaration SHA is retained provenance: raw design-role
labels, dtypes and categorical dictionaries are not saved, so that exact
declaration hash is not reconstructed on restore. Restoration does bind
actual complete weights, numerical PSU/stratum/FPC geometry, physical support,
weighted score solutions, original sensitivity/PSU scores, targets and covariance.
The independent fixture oracle additionally compares the declaration SHA with
the original supplied rows.

Complete fit, target evaluation, serialization and restoration work and
resident workspace are bounded before replica allocation. The 50 million
cumulative work limit includes actual Newton/step-halving operations and
empirical target work. CPU float64 is explicit, including restoration under
ambient alternative Torch defaults. Unsupported designs and nonfinite links,
empty target support, failed fits or exceeded resource limits fail explicitly.

The scope excludes multistage designs, calibrated unit-varying replica factors,
partial-FPC BRR/Fay, supplied jackknife extensions, general design effects,
linear target replication, discrete AMEs, new evaluation populations,
future-outcome prediction intervals, streaming Dataset and GPU execution.
These are associational standardized model targets, without a blanket Stata
parity or public-release claim.

## Method references

The [Stata survey postestimation manual, example 4](https://www.stata.com/manuals/svysvypostestimation.pdf)
places model fitting and margins together inside the jackknife procedure.
The [survey package's `withReplicates` contract](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/withReplicates.html)
evaluates the entire statistic under the original and every replicate weight
vector. These establish the replication construction, not licensed vendor
execution of this implementation. Acceptance uses independent numerical
refits and complete target/covariance enumeration, with source, frozen and
installed native persistence evidence recorded separately.
