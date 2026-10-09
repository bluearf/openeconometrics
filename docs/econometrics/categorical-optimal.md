# Bounded single-vector optimal scaling

`catreg_nominal`, `catreg_ordinal` and `catpca` fit explicitly declared
unweighted resident CPU float64 analyses. They expose learned category maps,
all retained sample positions, transformed observations, objective histories
for every start, accepted/failed-start records and the chosen solution. These
are descriptive optimization procedures. They supply no ordinary regression
standard errors, covariance, p-values or confidence intervals after learning
category transformations.

```python
import openecon as oe

nominal = oe.catreg_nominal(
    data, "score", ["occupation", "income"],
    scales={"occupation": "nominal", "income": "numeric"},
)
ordinal = oe.catreg_ordinal(
    data, "score", ["satisfaction", "income"],
    scales={"satisfaction": "ordinal", "income": "numeric"},
    orders={"satisfaction": ["low", "medium", "high"]},
)
pca = oe.catpca(
    data, ["occupation", "satisfaction", "income"], components=1,
    scales={"occupation": "nominal", "satisfaction": "ordinal", "income": "numeric"},
    orders={"satisfaction": ["low", "medium", "high"]},
)
state = oe.summary_state(ordinal)  # complete JSON, including maps and tables
restored = oe.restore_summary(state)
prediction = oe.catreg_predict(restored, new_data)
projection = oe.catpca_predict(oe.restore_summary(oe.summary_state(pca)), new_data)
```

The regression outcome is numeric and is returned in its original units. It
is centered and rescaled internally for stable optimization; its final mean
is the intercept, and each transformed predictor has weighted mean zero and
population variance one. Explicit numeric predictors keep their original
spacing under a recorded affine standardization. The default predictor scale
is nominal for `catreg_nominal`, ordinal for `catreg_ordinal`, and single-vector
nominal for `catpca`. A supplied `scales` dictionary must name every predictor;
`orders` must name exactly the ordinal variables, including every observed
complete-sample category once. No ordinal order or numeric scaling is guessed.

For CATREG, each predictor's partial residual is aggregated into category
means. A nominal block takes those means. An ordinal block compares the
frequency-weighted increasing and decreasing isotonic projections, computed
by pool-adjacent-violators. The returned ordinal map is always nondecreasing;
a negative coefficient represents a decreasing effect. Category contributions
are centered, factorized into a normalized map and signed coefficient, and
the complete transformed design is solved jointly by least squares after each
sweep. This is a numeric-outcome subset of optimal-scaling regression. With
all nominal and numeric predictors, the fitted additive response agrees with
intercept-and-dummy least squares. Ordinal restrictions can create tied pooled
categories, and multiple starts do not establish a global optimum.

Single-vector CATPCA minimizes the reconstruction loss
`sum((Q - X @ A.T)**2) / n`, with centered, variance-one transformed columns
`Q`, normalized object scores `X.T @ X / n = I`, and loadings `A`. Every
categorical variable has one scalar category map shared across all retained
dimensions. Given scores, the nominal block uses the leading singular vector
of frequency-weighted category centroids, an exact small rank-one block solve.
The ordinal block alternates a rank-one loading with weighted isotonic
quantification for at most 100 inner steps. A truncated SVD of all transformed
columns jointly updates the object scores and loadings. All-numeric scaling
reduces to population-standardized correlation PCA. Component axes are signed
by their largest absolute loading; equal eigenvalues remain rotationally
unidentified. The saved projection uses transformed data times stored axes,
divided by square roots of stored eigenvalues, matching the training score
normalization.

The default starts are one systematic category-spacing configuration and two
seeded random configurations (`n_starts=3`, `seed=0`). Ordinal initial scores
are sorted; nominal scores are centered and normalized. Every completed sweep
records the actual loss and improvement. An increase beyond roundoff is
rejected. A start is accepted only after loss change falls below `tol` (default
`1e-8`), with nondegenerate maps and identified required component/design
spaces. Failed starts are reported; the best converged start is selected. If
none converge, the call fails instead of returning a partial fit. This is a
declared local numerical solution, not a certificate of global optimality.

Input limits are checked before coercing a resident mapping or record list:
at most 3,000 supplied rows, 12 analysis variables, 32 observed levels per
categorical variable, 12 starts and 1,000 outer iterations. Default outer
`maxiter=500`. CATREG requires more complete rows than predictors plus two;
CATPCA requires more complete rows than dimensions plus two and retains
between one and `variables - 1` dimensions. Strings, booleans and finite
integer/float categories have typed identities. Numeric data must be finite
and have magnitude at most `1e100`; constant/unstable scales are refused.
`missing='drop'` performs recorded listwise deletion; `missing='raise'` fails.
Duplicate names, missing columns, invalid orders, unsupported weights,
Dataset/lazy inputs and non-CPU devices fail explicitly.

The planned work charge covers all declared starts and maximum outer
iterations, group aggregation/coordinate operations, small matrix solves and
the complete 100-step ordinal CATPCA inner allowance. Default `max_work` is
300 million declared scalar-equivalent units; an otherwise dimension-valid
combination may exceed it. The named live input/score/solve/map/trace buffers
are checked before tensors are built against the minimum of `max_bytes`
(default 128 MiB) and the current global workspace budget. These estimates
exclude caller inputs, Python result objects, private BLAS workspace and
allocator overhead; they are not process-RSS limits. Saved-map prediction
checks its own work/buffer plan and never retrains an unknown level.
Selected input copies and row/index masks have an earlier named admission
before coercion or selection. Mapping and record inputs are projected to the
required named columns first; unused wide columns are not materialized. Both
preparation and numerical buffer records are retained in the result metadata.

`settings` is a display preview. `summary_state` is the complete persisted
artifact, retaining every map, numerical table, normalization, sample position
and objective history. The stored prediction state has a canonical integrity
digest and semantic structure checks. It is not a security signature. Restored
results project without refitting. Unknown categories fail; missing prediction
rows can be explicitly dropped while retaining source positions. Table order
is normalized so complete restoration preserves concatenated LaTeX output.

Independent tests compare nominal fits with NumPy dummy OLS, ordinal fits
with bounded quadratic least squares and all contiguous isotonic partitions,
negative coefficients and tied categories, numeric/binary CATPCA with NumPy
SVD, and the fixed-score nominal block with a separate weighted-centroid SVD.
They also check centering, normalization, loss monotonicity, reproducibility,
saved-map predictions, complete restore/LaTeX, missing/category failures,
nonconvergence and resource/default-device guards. These tests are separate
from frozen and installed-native delivery receipts. No licensed SPSS parity
or publication/release/platform validation is implied.

Source definitions and algorithm references:

- [IBM CATREG algorithm manual](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catreg.pdf), the optimal-scaling least-squares objective and category normalization.
- [IBM CATREG ordinal level](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=command-level-keyword-catreg), nondecreasing quantifications and numeric spacing.
- [IBM CATREG initial configurations](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=catreg-initial-subcommand-command), ordinal sign patterns and multiple starts.
- [IBM CATPCA algorithm manual](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catpca.pdf), single-vector `NOMI`/`ORDI` restrictions and alternating least squares, pp. 5–9. Multiple-nominal `MNOM` maps are outside this API.
- [Meulman and van der Kooij, ROS Regression, section 3](https://arxiv.org/pdf/1611.05433), partial-residual optimal-scaling updates.
