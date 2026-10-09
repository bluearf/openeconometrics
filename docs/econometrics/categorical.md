# Categorical scaling and loglinear workflows

Eight bounded native CPU float64 method stages extend the existing linear
PCA/CCA, simple correspondence, metric MDS and row-level Poisson procedures.
The existing methods remain available separately.

| Issue | Delivered method and identification |
| --- | --- |
| MARKET-296 | Numeric-outcome nominal optimal-scaling regression; centered unit-variance scalar category maps |
| MARKET-297 | Ordinal optimal-scaling regression; explicitly ordered monotone maps, signed regression effects |
| MARKET-298 | Single-vector nominal/ordinal/numeric categorical PCA; centered transformed columns and orthonormal person scores |
| MARKET-299 | Unweighted disjunctive MCA; raw inertia and mass-standardized coordinates, saved supplementary projection |
| MARKET-300 | Multiset nonlinear canonical/homogeneity ALS; equal-person/equal-set loss with identified compromise scores |
| MARKET-301 | Nonmetric MDS; monotone disparities, primary/secondary ties, declared zero-pair handling and stress majorization |
| MARKET-302 | Hierarchical independent-Poisson IPF on complete declared structural-zero support |
| MARKET-303 | General independent-Poisson cell-design ML, full observed information and verified nested LR |

Read the [optimal-scaling guide](categorical-optimal.md),
[categorical-geometries guide](categorical-scaling.md) and
[loglinear guide](categorical-loglinear.md) for exact input, normalization,
missingness, convergence, sample and work/buffer limits. All support is
resident CPU float64. No automatic Dataset collection or GPU route is used.

Learning category transformations makes these first six procedures
exploratory/descriptive. Optimized regression coefficients do not receive
naive OLS standard errors after adaptive quantification; multiple starts
compare converged local solutions and do not certify a global optimum.
Poisson cell models separately provide full covariance and explicitly
asymptotic inference under independent-cell sampling.

Complete results, transformation maps, original sample positions, numerical
objectives, convergence and cell design/support persist with `summary_state`.
Saved regression/PCA/MCA projections use the trained mappings and refuse
unknown categories; nested loglinear comparison checks full support/design.
Table previews are not complete saved state.

Eight further residual option stages extend this core:

| Issue | Delivered option and bounded interpretation |
| --- | --- |
| MARKET-417 | Joint nominal-response CATREG; normalized learned response scores and saved score prediction |
| MARKET-418 | Explicitly ordered monotone ordinal-response CATREG; pooled ties and saved response-score prediction |
| MARKET-419 | Saved CATPCA orthogonal varimax; coherent rotation and supplementary person scores |
| MARKET-420 | Saved CATPCA oblique promax; pattern, structure, unit-diagonal component covariance and reconstruction |
| MARKET-421 | IID pairs full-refit numeric-response CATREG bootstrap for fixed-query mean predictions; empirical joint covariance/percentile intervals only when every draw succeeds |
| MARKET-422 | Direct fixed-grand-total multinomial cell ML; conditional joint information, mean delta uncertainty and nested LR |
| MARKET-423 | Direct fixed-stratum product-multinomial cell ML; explicit conditioning and conditional nestedness |
| MARKET-424 | Hierarchy-preserving backward Poisson AIC/BIC; every candidate and decision plus complete selected fit |

Read [response scaling and bootstrap](categorical-outcomes.md),
[saved rotations](categorical-rotation.md) and
[sampling and selection](categorical-sampling-selection.md) for their exact
contracts. Response-score prediction does not imply calibrated class
probabilities or invertible labels. Rotation leaves scalar quantification maps
unchanged and cannot recover original category centroids when tied maps lost
membership. Bootstrap refits transformations in every draw; recorded failures
withhold uncertainty. Model selection is a greedy path and its selected-model
intervals are not adjusted for selection.

Eight frequency and retained-membership stages are documented in the
[frequency guide](categorical-frequency.md): numeric-response nominal/ordinal
CATREG (MARKET-664/665), nominal/ordinal-response CATREG (666/667), CATPCA (668),
MCA/projection (669), OVERALS (670), and original-category CATPCA centroids (671).
These separate `*_fweight` APIs use integer counts directly on physical rows.
The new CATPCA state retains actual membership, so tied scalar maps no longer
prevent its category-centroid calculation. Legacy states lack that information.

Eight further [declared-spline and weighted-option stages](categorical-spline-weighted-options.md)
cover MARKET-691–698: count-weighted varimax/promax, two numeric-response
count-multinomial full-refit CATREG bootstraps and four declared-knot degree-one
nonmonotone/monotone CATREG/CATPCA procedures. These are bounded new fitting
domains; earlier accepted stages are not counted again.

Broader MARKET-171 / GitHub#58 remain open for higher-degree/automatic spline options, additional
sampling and inference domains and licensed vendor validation. No blanket
parity or public release is implied.
