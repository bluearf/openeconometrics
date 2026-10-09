# Spectral uncertainty and factorial moment inputs

MARKET-561–568 are eight new bounded option contracts following the completed
selection/MANOVA wave. They extend the existing methods; the broad MARKET-185
remains open for further options and independent licensed-vendor comparisons.

## PCA uncertainty

`pca_bootstrap(data, columns, components=..., matrix=...)` has two distinct
targets. Covariance PCA refits the centered unbiased sample covariance on
every IID row draw and preserves measurement units. Correlation PCA estimates
each draw's own means, sample standard deviations and correlation matrix.
Reusing the point fit's scales would estimate a different functional.

The component count is fixed by the caller. Ordered eigenvalues, signed
eigenvectors and loadings are retained together, with the complete joint
bootstrap covariance, standard errors, marginal percentile intervals and
every replicate vector. The covariance divisor is B−1 and percentile endpoints
use linear interpolation. Components need positive simple retained roots,
separation from their neighboring roots including the unretained boundary,
and a fixed sign anchor away from zero in every replicate. No permutation or
Procrustes alignment hides an unresolved axis. A failure refuses the complete
uncertainty result; failed draws are not silently removed.

PCA draws must also stay within a 45-degree chart of each fixed point axis.
Nonconstant variances below normal float64 resolution and overflowing moments
are refused. These finite numerical checks do not establish population regularity.

## Principal-factor uncertainty

`factor_multifactor_bootstrap(..., factors=..., target=None)` extends the
fixed one-factor functional to a caller-fixed count of two through four
principal factors. Each draw recomputes correlation, squared multiple
correlations and the reduced-matrix spectral decomposition. The unrotated
target requires separated retained axes, fixed nonzero signs and positive
interior uniquenesses in every fit.

With a complete caller-declared `target`, every draw uses the same orthogonal
Procrustes target. If A is the unrotated loading matrix and T the target,
the SVD of A′T gives the polar rotation Q=UV′. Full rank of A′T gives a unique
polar factor. The target is not re-estimated from the point fit or a draw.
The output retains full rotated loading/uniqueness covariance and all draws;
it does not treat a local varimax/geomin solution as an identified target.

These are first-order IID bootstrap procedures for fixed-dimensional
estimator functionals under finite fourth moments and regular interior
population covariance/spectral geometry. Finite sample diagnostics do not
establish these population assumptions. Intervals are marginal; no exact
finite-sample, simultaneous, high-dimensional, selected-count, dependent-row,
weighted-bootstrap or latent-parameter ML inference is claimed. P-values and
inferential degrees of freedom are unavailable. Full sample, row identity,
settings, anchors/target, every vector and numerical diagnostics persist.

## Factorial MANOVA moments

`manova_factorial(data, y, factors, weights=...)` admits a complete crossed
categorical design with exact nonnegative integer frequencies.
`manova_factorial_summary(cell_means, cell_covariances, counts, ...)` admits
the same declared design from explicitly ordered cell means, unbiased
within-cell covariances and integer counts. No observations are synthesized.

The model uses sum-to-zero factor coding and explicit factorial terms.
For cell design x_g, cell mean μ_g and count n_g,

```
X′X = Σ n_g x_g x_g′
X′Y = Σ n_g x_g μ_g′
E = Σ (n_g−1) S_g + Σ n_g (μ_g−x_g′B)(μ_g−x_g′B)′
```

Type III term hypotheses use the full fitted coefficient/bread geometry.
Complete H/E, univariate tables and four multivariate tests preserve the
existing exact/approximate/Roy-upper-bound conventions. Full saved
coefficients, bread and residual SSCP support `manova_contrast` after JSON
restoration. Continuous covariates and incomplete crossed cells are outside
these adapters.

## Repeated-measures moments

`rm_anova_fweight(data, y, subject, within, weights=..., between=...)` applies
one exact integer count to a whole complete subject profile. A count must be
consistent across that subject's measurement rows. Copies would represent
new independent subject vectors, rather than extra observations of the same
subject. Whole-subject missing policy, physical/missing/zero-count subjects
and total frequency are recorded separately.

`rm_anova_summary(cell_means, cell_covariances, counts, within=..., between=...)`
uses declared group moments over ordered original within cells. Existing
orthonormal within-factor transforms and Mauchly/GG/HF conventions are
retained. No individual subject vectors are invented from summaries. Original
cell coefficients/error geometry supports saved general `rm_mtest` hypotheses.

Gaussian common unrestricted response/cell covariance and independent original
observations or subject vectors underlie the classical inference. Integer
counts are literal frequencies with exact totals at most 2^53; arbitrary
duplicates do not establish the sampling assumptions. Each covariance must
be PSD and have rank compatible with count−1; fitted designs and requested
error projections must have resolved adequate rank. Labels, order, finite
precision, saved-state integrity and workspace/work are validated explicitly.
No incomplete-subject, dependent-subject, analytic/probability/survey weighting
or universal MANOVA/GLM/vendor parity is asserted.

For rank-one effects and saved hypotheses, the new moment adapters compute the exact F law directly
from the non-cancelling Hotelling root. This preserves the statistic convention
when a Pillai value approaches one at a large intercept. Existing adapters retain
their original behavior.

Spectral inputs are bounded to 10,000 resident rows, 16 variables, 19–1,999
draws and 250 million planned operations, with a 32 MiB portable-output bound.
Factorial inputs admit up to 100,000 resident rows, 64 complete crossed cells,
four factors, 32 outcomes and 256 coefficient targets. Repeated inputs admit
10,000 physical complete profiles and 32 within cells. Moment work is bounded
to 250 million planned operations; encoded labels and aggregate metadata have
explicit byte limits.

The acceptance example retains explicit table order and dtypes alongside the complete
JSON state and original LaTeX. JSON alone carries no numeric dtype for an
all-null p-value/df column; restoring the saved rendering dtypes recovers the
same LaTeX without changing a numeric cell or setting.

Primary definitions and boundaries: [R PCA documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html),
[bootstrap PCA algorithm](https://arxiv.org/abs/1405.0922),
[high-dimensional bootstrap limitations](https://arxiv.org/abs/1608.00948),
[SAS frequency statement](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_syntax08.htm),
and [SAS repeated-measures hypotheses](https://support.sas.com/documentation/cdl/en/statug/63962/HTML/default/statug_glm_sect037.htm).

Source numerical validation, frozen compiled identity, actual installed native
Run, saved reconstruction and full application Quit/reopen are distinct proof
layers recorded in the delivery evidence. A source method does not by itself
establish public shipment or licensed-vendor parity.
