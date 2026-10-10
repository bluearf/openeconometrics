# Single-stage survey inference

Eight public procedures return a typed `oe.SurveyResult`: `survey_mean`,
`survey_total`, `survey_ratio`, `survey_proportion`, `survey_brr`, `survey_fay`,
`survey_jackknife` and `survey_bootstrap`. See the [runnable example](../examples/survey_inference.py).
Use a complete resident table and the existing [SurveyDesign](survey-design.md).
Every request revalidates its exact weights, typed PSU/stratum identities and row order.
The numerical kernels use CPU float64 Torch. These procedures are separate from `ModelSpec` regression estimators.

## Targets and complete design geometry

Means are Hájek ratios `sum(w*y)/sum(w)`; totals are unnormalized `sum(w*y)`;
ratios use one explicit denominator column per numerator. Categorical proportions
require a fixed typed category dictionary (or a pandas categorical dictionary),
including absent levels; all levels are retained in the singular joint covariance.
Zero or near-cancelled ratio denominators fail explicitly. Numerical overflows fail.

`domain` names a complete Boolean/0-1 column. `missing='raise'` is the default;
`missing='drop'` excludes joint incomplete in-domain targets while preserving every
design row and PSU, including those with zero eligible domain observations.
Outside-domain outcomes are irrelevant. Input rows must never be subset before declaring
the full design. The result retains exact original positions, exclusions, weight sums,
domain and target roles, design fingerprint and target-input fingerprint.

For totals, row influence is `I(domain)*y`. For ratios it is
`I(domain)*(y-theta*x)/sum(w*I(domain)*x)`; means and proportions use `x=1`.
Sum weighted influences within each PSU and retain zero-domain PSUs. Taylor covariance is
`sum_h (1-f_h)*m_h/(m_h-1) * sum_j centered(u_hj)*centered(u_hj)'`.
Integer first-stage population PSU counts provide `f_h=m_h/N_h`. Census strata contribute
zero; singleton certainty requires explicit census FPC. Other singleton strata fail
at declaration. Optional `deff=True` retains the accepted equal-weight SRSWR
independent-row calculation and supports positive unequal sampling weights with
a named weighted-population SRSWR plug-in reference. Both retain all original
design rows. For the existing linearized row influence vector `u_i`, define
`W=sum(w_i)`, `u_bar=sum(w_i*u_i)/W`, and
`V_srswr=W/(n-1)*sum(w_i*(u_i-u_bar)*(u_i-u_bar)')`, where `n` is the full
physical design-row count. Report each diagonal ratio `V_design/V_srswr` in
`metadata['design_effect']`; a zero reference variance has an undefined (`None`)
ratio. A census design can have zero design variance with a positive reference.

The unequal-weight reference treats the declared domain and joint missing-data
eligibility indicators as fixed, including zero influences outside them. It is
a linearized with-replacement empirical-population comparator, not the exact
finite-sample variance of a random ratio estimator. It differs from resampling
rows while retaining their unequal weights, and from dropping out-of-domain rows.
The saved reference records its complete joint covariance, full-design weight sum,
sample count, weighted influence mean and centered crossproducts. Restore checks
their dimensions, geometry, positive semidefiniteness, covariance replay and
diagonal ratios, even when a modified payload has a recomputed integrity digest.
Equal-weight results keep their previous serialized output unchanged.

The variance ratio above is named relative to SRSWR. It is not Stata's
FPC-adjusted SRSWOR `DEFF`, its default subpopulation comparator or `srssubpop`.
Without-replacement, calibrated/multistage/regression/replicate design effects
remain unsupported. Unequal-weight moment construction is bounded to 50 million
`n*k*(k+2)` work units alongside the existing resident-memory and target limits.

## Replicate conventions

Replicate methods use `target='mean'|'total'|'ratio'|'proportion'` and the same target
options. All replicas are evaluated; any failed estimate raises with every failed ID.
No replica is silently omitted. IDs, full estimates, weight digest, scales, centering,
variance multipliers, count and generation/provenance are saved.

- BRR/Fay require exactly two PSUs in each noncensus stratum. Generated signs are
  balanced orthogonal nonconstant Sylvester columns; the power-of-two replicate count
  strictly exceeds the number of stochastic strata. Only required columns are stored,
  without a replicate-count square. Supplied weights must verify the same PSU pair
  factors and balanced orthogonal selections. Census weights remain unchanged.
  Partial noncensus FPC is rejected. Fay supports `0 <= rho <= 1-1e-6`; PSU factors
  are `rho` and `2-rho`, with multiplier `1/[R*(1-rho)^2]`. `rho=0` is the BRR limit.
- Generated stratified jackknife deletes an entire PSU, inflates other PSUs in that
  stratum by `m_h/(m_h-1)` and leaves other strata unchanged. Each covariance term
  uses `(1-f_h)*(m_h-1)/m_h`. Census PSUs are not deleted. `centering` is `original`
  or `stratum_mean` (the average of replicas belonging to the deleted PSU's stratum).
- Bootstrap consumes an explicit resident replicate-weight DataFrame, a nonempty
  sampling justification and positive `scale`; optional `rscales` supply one positive
  factor per replica. Factors must be constant within first-stage PSUs and leave
  census PSUs unchanged. This is a bounded first-stage supplied design-weight route,
  not a bootstrap generator. The caller's sampling provenance is not authenticated.
  Calibrated unit-varying weights and ordinary row bootstrap remain unsupported.
  Numeric linear/logit/probit/Poisson coefficient replication is implemented separately for all
  four methods; see [coefficient replication](survey-regression-replication.md).

BRR/Fay/bootstrap support `original` or `replicate_mean` centering. Covariance is
`sum_r multiplier_r*(theta_r-center)*(theta_r-center)'`, with per-stratum centers
for the jackknife option. Replicate tables follow physical input-row order; their index
is not automatically aligned. All-census BRR/Fay/jackknife requests explicitly direct
the caller to zero-variance Taylor inference.

## Inference, persistence and budgets

`to_frame()` restores estimates, SE, statistic, two-sided p and unclipped Wald confidence
limits using Student t. Taylor df is all declared PSUs minus all declared strata,
including certainty and zero-domain strata. Default replicate df is `min(design_df,R-1)`;
bootstrap can explicitly declare an integer df in `1..R-1`. These are named OpenEconometrics
conventions, not universal vendor defaults. Zero SE/zero df gives undefined tests and
point intervals. `alpha` is explicit and `null` can be scalar or one per target.

JSON state retains the complete joint covariance and the full design/target/replica
record. Restore with `oe.SurveyResult.model_validate_json(text)` then `.to_frame()` or
`.contrast([coefficients], null=...)` without original data or refitting. Dimensions,
finite symmetric positive semidefinite covariance, geometry and integrity digest are
validated. Saved replicate covariance is also replayed from every stored replica.
The digest detects accidental modification; it is not an authenticated sampling proof.
LaTeX uses `oe.to_latex(result.to_frame())` and the ordinary native table export path.

At most 32 targets, 4,096 replicas, 8 million row-replicate cells and 50 million
row-replicate-target operations are admitted. Supplied BRR also requires `R > H`
and at most 50 million `R*H^2` balance-validation operations; its column dot
products never allocate an H-by-H matrix. These apply alongside declaration and global memory
budgets. Allocation plans include tensors, rows, serialization, full covariance and
replica output before copying/creating weights. PSU-factor validation is linear in
row-replicate cells. There is no dense observation-square allocation.
Multistage designs, calibrated unit-varying plans, further model replication,
Dataset estimation and device inputs remain unsupported by these procedures.
Single-stage Taylor regression and saved inference are implemented separately;
see [survey-regression.md](survey-regression.md). Complete numeric linear/logit/probit/Poisson
coefficient refits using BRR, Fay, generated stratified PSU jackknife and supplied
design-bootstrap weights return `oe.SurveyRegressionResult`; their full covariance,
method-aware df, failed-replica rejection and conditional saved postestimation are
documented in [survey-regression-replication.md](survey-regression-replication.md).
Full nonlinear/unconditional margins, multistage/calibration, general design
effects and broader regression families retain separate acceptance. Advanced work is
tracked in [GitHub #151](https://github.com/bluearf/openecon/issues/151).

## Evidence and primary references

Independent NumPy/SciPy tests cover full covariance/inference, fixed categories,
FPC/census/SRS limits, missing/domain geometry, weight scaling, all-replica enumeration,
invalid weights and saved contrasts. The official NHANES2 example in the
[Stata survey postestimation manual, Example 1](https://www.stata.com/manuals/svyestat.pdf)
is compared to the [official reference dataset](https://www.stata-press.com/data/r19/nhanes2.dta).
The external dataset is downloaded for validation and is not redistributed here.
[Stata variance estimation methods](https://www.stata.com/manuals/svyvarianceestimation.pdf)
provides the primary Taylor, ratio, BRR/Fay, stratum jackknife and bootstrap formulas.
Manual-output comparison is distinct from a licensed Stata execution;
`stata_parity_validated` remains false. Acceptance evidence (internal evidence excluded from this public snapshot)
records source, frozen and installed native execution/persistence separately.
