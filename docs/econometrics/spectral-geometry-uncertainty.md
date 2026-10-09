# Fixed spectral geometry and query uncertainty

MARKET-584, 586, 587, 588, 589, 591, 593 and 594 are eight new bounded
contracts following PR193. The broad MARKET-185 remains open for its other
method options and independent vendor acceptance.

## Invariant PCA subspaces

`pca_subspace_bootstrap(data, columns, components=m, matrix="covariance")`
resamples complete IID rows and refits unbiased centred covariance. The
`matrix="correlation"` route refits each draw's own means and sample SDs.
Both infer the fixed leading rank-m projector P=V_m V_m', captured trace
tr(PA), residual trace tr((I-P)A) and captured trace share. The component
count is fixed and strictly smaller than the variable count.

Only the retained/discarded boundary needs separation. Repeated roots
inside either block are allowed because the projector is invariant to
sign changes and rotations within that block. Upper-triangular projector
entries have a declared variable order; the full symmetric projector is
also saved. Individual tied axes or eigenvalues get no interval. Projector
covariance is structurally singular and is retained without inversion.
Draw distances/angles are diagnostics, not simultaneous confidence balls.

[Spectral projector functionals](https://arxiv.org/abs/1408.4643) motivate
the invariant target. The specialized
[bootstrap confidence-set construction](https://arxiv.org/abs/1703.00871)
is a different procedure; its coverage results are not claimed for this
ordinary IID resampling implementation.

## Canonical correlation

`canon_bootstrap(data, x, y, components=m, target="correlations")` returns
joint uncertainty for a caller-fixed count of ordered canonical correlations.
`target="coefficients"` additionally retains raw and standardized X/Y
coefficients and within/cross loadings. Raw canonical variates have unit
sample variance and positive paired correlation. Every draw recomputes the
centred moments, scales and conditioned whitening geometry.

All retained roots must be positive, simple, below one and separated from
adjacent roots, including the omitted boundary. Coefficient signs are set
by ordered X anchors chosen once at the point fit or declared by the caller;
every draw stays in the same local signed axis chart. There is no per-draw
argmax, relabelling, permutation or Procrustes alignment. Zero, tied and
perfect roots are outside this ordinary bootstrap contract. Root intervals
are not calibrated tests of zero canonical correlation. The underlying
functional follows the [canonical correlation definition](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cancor.html).

## Correspondence sampling models

`ca_bootstrap(counts, dimensions=m, sampling="multinomial")` accepts a
labelled exact integer contingency table under caller-declared independent
fixed-total multinomial sampling. The `sampling="row_multinomial"` route
instead fixes every observed row total and independently draws the cells
within each row. These are different sampling experiments. Fixed row masses
retain their explicit zero variance in the latter route; column masses vary.
The [multinomial and product-multinomial definitions](https://www.stat.ethz.ch/CRAN/web/packages/rTableICC/refman/rTableICC.html)
support this distinction.

Each draw refits masses and the SVD of standardized residuals. Full total
inertia, retained singular values/inertias, row/column masses and standard
and principal coordinates enter one joint parameter vector. Category order,
fixed row sign anchors and simple positive axes remain stable; zero margins,
unresolved gaps or a draw leaving the signed chart refuse the whole result.
No category is silently removed and no requested dimension is clamped. Every
complete drawn count table is retained. This is sampling uncertainty for
the observed CA functional, not simulation under the independence null.
Fractional, survey, clustered, Poisson or both-fixed-margin sampling is
outside these two routes.

## Fixed query PCA scores

`pca_bootstrap_scores(saved_pca, queries)` reuses a complete identified
`pca_bootstrap` result after reconstructing the kept source sample and
replaying the point fit and every seeded draw. Generic JSON restoration
alone does not admit altered numerical state.

For covariance PCA the score in draw b is `(query-mean_b) @ V_b`.
For correlation PCA it is `((query-mean_b)/sd_b) @ V_b`. Keeping point-fit
means/scales fixed would estimate a different functional. The query is
caller-fixed independently of the training data, so these intervals describe
training-estimator uncertainty at that query. They are not future-observation
or residual/noise prediction intervals. The
[PCA projection definition](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html)
and [bootstrap PCA score algorithm](https://arxiv.org/abs/1405.0922) provide
the underlying construction.

Numerically, centring uses an actual source-row origin and the twice-centred
offset from that origin, rather than subtracting a rounded large-level mean.
Point and every draw's origins/offsets persist with the complete fit state.

All complete query-by-axis scores share a full joint covariance. Repeated
queries are allowed and retain singular covariance without inversion.
Physical query row positions, typed row identities and missing alignment
persist; missing queries have no inferred target. The complete original
fit is retained in prefixed `fit__` tables and `fit_attrs`.

## Shared inference and saved-output contract

Every result retains complete point estimates, all planned replicate vectors,
full joint covariance with divisor B-1, standard errors, bootstrap bias and
linear-interpolated marginal percentile endpoints. No planned draw is dropped.
If any draw fails the geometry or precision requirements, the entire result
is refused. P-values and inferential degrees of freedom are unavailable.

For PCA/CCA, first-order IID bootstrap regularity assumes fixed dimensions,
finite fourth moments, interior conditioned moments and the declared separated
population geometry. CA assumes independent observations under its declared
multinomial experiment and regular positive population masses/simple axes.
Finite sample guards do not establish these population assumptions. A smooth
coordinate with a vanishing first derivative can still have nonstandard
coverage; numerical intervals are not a universal coverage certificate.
No exact, familywise, selected-count, high-dimensional, dependent-row or
weighted-bootstrap coverage is claimed.

Resident CPU float64 tensors are the numerical implementation. Work, named
memory buffers, metadata/identity traversal and complete portable output are
admitted before allocations, with a 32 MiB JSON domain and 250 million planned
operations. Source scientific references, frozen execution, actual installed
native Run and full Quit/reopen are separately evidenced; a source method is
not by itself public shipment or licensed vendor parity.
