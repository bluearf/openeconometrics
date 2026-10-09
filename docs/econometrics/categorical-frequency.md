# Frequency-weighted categorical methods

These eight residual MARKET-171 stages use integer replication counts directly
on the retained physical rows. They are separate APIs from the existing
unweighted categorical methods; none routes through `oe.fit` or expands rows.

| Issue | API | Statistical target |
| --- | --- | --- |
| MARKET-664 | `catreg_nominal_fweight` | Numeric-response CATREG with nominal/numeric predictors |
| MARKET-665 | `catreg_ordinal_fweight` | Numeric-response CATREG with explicitly ordered monotone predictors |
| MARKET-666 | `catreg_nominal_response_fweight` | Joint nominal response/predictor quantification |
| MARKET-667 | `catreg_ordinal_response_fweight` | Joint monotone ordered response/predictor quantification |
| MARKET-668 | `catpca_fweight` | Single-vector nominal/ordinal/numeric categorical PCA |
| MARKET-669 | `mca_fweight`, `mca_fweight_project` | Disjunctive MCA with weighted masses and fixed supplementary projection |
| MARKET-670 | `overals_fweight` | Frequency-weighted multiset optimal-scaling compromise |
| MARKET-671 | `catpca_category_centroids` | Original-category score means from retained membership, with declared basis transport |

For example, a count column can calibrate an ordered predictor while keeping
a numeric predictor on its weighted standardized scale:

```python
import openecon as oe

fit = oe.catreg_ordinal_fweight(
    data, "spending", ["rating", "income"], frequency="count",
    scales={"rating": "ordinal", "income": "numeric"},
    orders={"rating": ["low", "mid", "high"]},
)
restored = oe.restore_summary(oe.summary_state(fit))
predictions = oe.catreg_fweight_predict(restored, query_rows)
```

## Counts, sample and resources

Supply a pandas DataFrame and a required `frequency="count_column"`. Counts
must be finite, nonnegative exact integers; integral floating values are
accepted, booleans are refused. Each count and the total of **all supplied
nonmissing counts** must be at most 1,000,000,000. Counts represent repeated
observations, not inverse variances or survey inclusion probabilities.

All counts are validated before feature-based deletion. Under `missing="drop"`,
missing counts and positive-count rows with missing required variables are
removed. Under `missing="raise"`, those missing values are errors. Zero-count
rows are excluded before their feature values are inspected. Results retain
original integer row positions, counts, zero positions and missing positions;
duplicate DataFrame indexes do not realign observations. Category universes
and explicit ordinal orders refer to the retained positive-frequency sample.
MCA and OVERALS also preserve bounded finite scalar index labels; convert
unsupported structured/date labels explicitly before those calls.

Fit input is bounded at 3,000 physical rows, 12 variables/predictors and 32
observed levels per categorical variable. At least four complete positive
physical rows are required; rank and dimension checks may require more.
Names and category strings are bounded; typed integer, float, boolean and
string labels stay distinct. Numerical columns must be finite, nonconstant
real values of magnitude at most 1e100. Declared worst-case work and live
buffers must fit `max_work` (at most 300 million) and `max_bytes` (at most
128 MiB), also subject to the active workspace budget. Multistarts and nested
ordinal iterations consume that work budget. These are implementation/resource
boundaries, not a statistical requirement that large datasets be truncated.
Only resident CPU float64 is supported; there is no Dataset, GPU or streaming
fallback.

## Numerical contracts

Write D=diag(f) and F=sum(f). Numeric transformations subtract the weighted
mean and divide by the weighted population standard deviation. Categorical
updates use group totals and means under f; ordinal updates use weighted
isotonic regression. CATREG solves weighted least squares for each fixed set
of transformations. Numeric-response predictions return original response
units. Categorical-response predictions return the learned response score,
with no calibrated class probability or unique inverse category.

CATPCA decomposes sqrt(D/F) Q. Its person scores X satisfy X'DX/F=I, its
loadings A reconstruct Q approximately as XA', and its objective is the
weighted squared reconstruction residual divided by F. MCA uses f/F as
row masses and frequency-adjusted disjunctive category masses. OVERALS
minimizes the weighted sum of set reconstruction losses under weighted
centering and compromise-score normalization. Multiple starts report
converged local solutions and do not certify a global optimum.

These are descriptive adaptive methods. Coefficient covariance, SE, test df,
p-values and confidence intervals are not supplied by pretending that the
learned transformations were fixed in advance. Changing all counts by the
same positive integer leaves the descriptive geometry unchanged (within
roundoff); it changes recorded frequency totals. Verification uses independent
small literal-replication/formula references, never production row expansion.

## Saved predictions and original-category centroids

`catreg_fweight_predict`, `catpca_fweight_predict` and `mca_fweight_project`
take fitted or restored results and query DataFrames. Query rows do not need
a frequency column and do not change the fitted masses. Unknown typed
categories are refused; missingness follows the declared query policy.
Complete result tables, maps, sample metadata and guarded numerical state
persist through `summary_state`/`restore_summary`. Specialized reuse checks
the saved shapes, counts, mappings and method-specific numerical identities;
the checksum is only an integrity check, not scientific evidence by itself.

For category k of variable j, the centroid is the actual group mean
sum(f_i X_i : g_ij=k)/sum(f_i : g_ij=k). Two ordinal categories can share a
scalar quantification while their full score centroids differ. New frequency
CATPCA results retain the original category membership, so the helper can
distinguish them. Legacy map-only results are refused.

An optional square, finite, nonsingular `transform=T` declares a new basis:
loadings become AT, scores and centroids become X(T^-1)', and the score
covariance becomes T^-1(T^-1)'. Reconstruction is unchanged. This is explicit
basis transport; it does not implement automatic weighted varimax/promax.
Ill-conditioned or excessively scaled transforms are refused. Original and
transported centroids, scores, loadings, transform and covariance are saved.

## References and evidence scope

Weighted category updates and normalized alternating least squares follow
the published [IBM CATREG algorithms](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catreg.pdf)
and [IBM CATPCA algorithms](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catpca.pdf).
The multiset/rank-restriction interpretation is described by
[de Leeuw and Mair, 2009](https://www.jstatsoft.org/article/view/v031i04).
The implementation has its own bounded input, missingness, initialization and
saved-state contracts. Independent numerical checks, frozen runtime, native
application execution and public shipment are separate evidence layers.
Licensed-vendor equivalence, broader splines/inference options and blanket
Stata/SPSS parity are not asserted; parent MARKET-171 remains open.
