# Frequency, summary and saved repeated-measures options

MARKET-369–376 extend the existing reliability, adequacy, discriminant, CCA
and complete-subject RM kernels. Each is an option/input/state contract;
these additions do not certify universal SPSS/Stata multivariate parity.

| Scope | API and supported behavior | Explicit boundary |
| --- | --- | --- |
| Frequency reliability | `alpha(data, cols, weights="w", weight_type="fweight", model="alpha"/"split"/"guttman", reverse=..., standardized=...)`. Full item/scale tables use anchored integer-count moments and covariance divisor `W-1`, without expanding rows. Resident and bounded Dataset inputs. | Descriptive coefficients only; no sampling covariance/SE/CI, analytic/probability weights, automatic reversal or pairwise deletion. |
| Frequency adequacy | `factortest(..., weights="w", weight_type="fweight")`. KMO/item MSA and SMC use weighted correlation; Bartlett chi-square uses `W-1-(2p+5)/6`. All statistic/df/p and sample accounting persist. | Positive-definite correlation required. Gaussian Bartlett inference assumes the counts encode original independent observations; arbitrary duplicates or survey counts do not create valid inferential sample size. |
| Frequency discriminant | `discrim(..., weights="w", weight_type="fweight", method="lda"/"qda", loo=True/False)`. Stable grouped moments, count-weighted confusion and complete saved posterior state. LOO removes **one expanded frequency copy**; full-fit priors stay fixed. | Resident input; no Dataset, analytic/probability/survey weights or weighted-LOO vendor parity. Repeated identical rows do not cure covariance rank. Group/pooled and actual leave-out rank are checked. |
| Group-summary discriminant | `discrim_summary(group_means, group_covariances, counts, method=..., priors=...)`. Exactly ordered declared group means/covariances/counts reconstruct within SSCP `W=sum((n_g-1)S_g)` and between SSCP. Existing prediction/canonical/Gaussian tests reuse full saved state. | Counts `n_g>=2`; sample covariance convention `n_g-1`; each covariance PSD and rank compatible with its count. LDA needs pooled PD, QDA each group PD. No synthetic observations, inferred training confusion/accuracy or LOO. |
| Frequency CCA | `canon(data, x, y, weights="w", weight_type="fweight")`. Joint listwise count moments and block-Cholesky whitening/SVD; full coefficients, training moments and score state. Existing raw unweighted QR remains available. | Gaussian canonical tests require original independent frequency observations. Within-set standardized covariance must be full rank and sufficiently conditioned; joint singular PSD may contain a perfect canonical correlation. |
| Joint-summary CCA | `canon_matrix(joint, n=..., x=..., y=..., matrix="covariance"/"correlation", means=..., sds=...)`. Declares one common centered unbiased covariance/correlation sample with ordered disjoint sets, explicit count and full PSD joint matrix. `canon_scores` reuses full saved raw coefficients/moments. | No PSD repair or invented observations/moments. Raw centered score projection requires actual training means and correlation-input raw scales. Zero/repeated canonical roots have nonunique axes; a serialized valid basis is retained, not re-estimated or claimed unique. Perfect-root test values retain existing undefined conventions. |
| Wide RM input/state | `rm_anova_wide(data, columns, subject=None, missing="raise"/"drop_subject", sampling_model="iid_multivariate_normal")`. Ordered measurement columns adapt actual complete subject vectors to the existing long RM kernel. Full sample, subject/index labels/order/positions/hash, means, sample covariance and covariance of means persist; `rm_restore` revalidates state. | One within factor, resident CPU float64 and independent subject vectors. Incomplete subjects are dropped only by explicit policy; no mixed/between/exogenous/weighted/dependent-subject extension. Existing Mauchly/GG/HF approximations are preserved; no new ANOVA core or vendor correction parity. |
| Saved RM contrasts | `rm_contrasts(saved_result, C, null=..., alpha=.05, sampling_model="iid_multivariate_normal")`. Predeclared zero-sum full-row-rank contrasts give estimates `C mu`, full mean covariance `V=C S C'/n`, marginal t/CI (`df=n-1`), complete subject contrast vectors and an exact Gaussian joint Hotelling F. | `q<=k-1`, `n>q`, PD contrast covariance. No sphericity assumption for the multivariate contrast test, pseudoinverse, selected/between/weighted/dependent contrasts or familywise/non-Gaussian exact coverage. |

All frequency routes require finite nonnegative **exact integer** counts and
count total `W<=2**53`; zero counts are excluded. Listwise missing policies
include the count column. `n` denotes W while `physical_rows`, `n_missing`,
`n_zero_weight`, `weight_sum` and `covariance_divisor` are distinct. Full
coefficients/moments and prediction/score rows preserve source order and
missing alignment. Canonical Dataset projection is lazy and verifies a full
source digest on replay. Discriminant and RM inputs are resident.
New LDA scores compare centered pairwise log odds, including one-copy deletion,
instead of subtracting shared large squared distances. QDA refuses unresolved
float64 density comparisons rather than returning a rounded probability tie.

For RM contrasts, `T²=(C mu-null)'V^-1(C mu-null)` and
`F=(n-q)/(q*(n-1))*T²`, with degrees `(q,n-q)`. A one-row contrast therefore
has `F=t²`; substituting generic Wald `T²/q` and denominator `n-1` would be
incorrect for q>1. Marginal intervals use t with n-1 degrees. The declared
subject-level multivariate Gaussian sampling assumptions are recorded, not
empirically verified from observed finite values. Contrasts must be specified
independently of the observed outcome; the API cannot verify that intent.
Stored binary64 contrast rows must sum exactly to zero; near-zero sums are
refused instead of evaluating coefficients different from the declared C.
Within covariance is evaluated from subject differences. The adapter refuses
scales where the unchanged ANOVA core cannot resolve within sums of squares;
center each subject's measurement vector before requesting that analysis.

## Admission limits

Every new route is CPU float64 and admits workspace/work before its planned
numerical allocations. Caller-resident memory and total process RSS are not
claimed as bounded by the workspace estimate.

- Reliability/adequacy reuse at most 256 variables and matrix work at most
  2,000,000,000 through existing anchored frequency moment plans.
- Discriminant: at most 1,000,000 physical rows, 128 groups and 256 variables;
  work `(G+2)*p³ + rows*(3*p²+(4+4*int(loo))*G*p²) <=2,000,000,000`.
  Summary count totals remain bounded by exact integer precision; no row
  replication occurs. QDA needs enough observations and positive rank in
  each group and each requested leave-out fit.
- Frequency/summary CCA: at most 64 combined variables, 1,000,000 source rows,
  estimated work 500,000,000 and within-set standardized condition at most
  1e10. Unknown Dataset sizes are checked while batches are consumed.
- Wide RM: at most 10,000 subjects, 32 measurements and planned work
  100,000,000. Long adaptation, wide/sample copies, covariance/contrast and
  result-state memory are part of its workspace plan.

The tests independently compare complete numerical tables, statistics,
degrees/p-values, covariance and every score/prediction/contrast vector on
at least two domains. Literal row expansion and per-copy leave-out refits
check frequency semantics; summary/raw equality, covariance eigenproblems,
normal-density posteriors, direct item/KMO formulas, existing long RM and
independent sums-of-squares/paired-t/Hotelling calculations check the new
routes. Saved-state and invalid-label/rank/missing/resource refusal tests
are separate from numerical references. Source, frozen compiled identity,
actual native Run and restart readback remain separate proof layers.

Primary definitions: [Stata reliability](https://www.stata.com/manuals/mvalpha.pdf),
[Stata factor postestimation](https://www.stata.com/manuals/mvfactorpostestimation.pdf),
[Stata LDA](https://www.stata.com/manuals/mvdiscrimlda.pdf),
[Stata QDA](https://www.stata.com/manuals/mvdiscrimqda.pdf),
[Stata canonical correlation](https://www.stata.com/manuals/mvcanon.pdf),
[SAS frequency/summary canonical analysis](https://support.sas.com/documentation/onlinedoc/stat/141/cancorr.pdf),
[NIST Hotelling one-sample distribution](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/1samphot.htm),
and [Stata response transformations](https://www.stata.com/manuals/mvmanovapostestimation.pdf).

Broader MARKET-185 remains open for other weighting/design/stepwise and
Kaiser generalized-image domains, broader selected-model inference and
licensed vendor comparisons. Previous MARKET-256–263 and MARKET-311/313–319
are preserved and are not counted in this wave.
