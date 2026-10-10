# Fixed-query multivariate score uncertainty: eight acceptance domains

Preregistered on 10 October 2026 before implementation. Starting source:
`2921a8ea53dd7c6441cbd5b749d4acaedac864a9`. Parent: MARKET-185.
Owner: chat `01a11344-6de7-7b92-b2f6-56e95ef62894`, branch
`codex/multivariate-score-uncertainty-eight`.

## Scope

Existing point fits, unweighted PCA score intervals and bootstrap estimation
are preserved. Three new saved-result APIs will expose eight distinct domains:

1. Covariance PCA fixed-query scores from a literal-frequency bootstrap.
2. Correlation PCA fixed-query scores from a literal-frequency bootstrap;
   every draw uses its own training standard deviations.
3. Joint paired X/Y CCA fixed-query scores from an identified IID coefficient
   bootstrap, including covariance between both blocks and query rows.
4. The same joint paired CCA scores under literal-frequency IID resampling.
5. Regression factor scores from fixed unrotated multifactor PF IID bootstrap.
6. Regression factor scores from fixed full-target orthogonal PF IID bootstrap.
7. Fixed unrotated multifactor PF regression scores under literal frequencies.
8. Fixed full-target orthogonal PF regression scores under literal frequencies.

Public APIs: `pca_fweight_bootstrap_scores`, `canon_bootstrap_scores`,
`factor_bootstrap_scores`. A complete saved fit is the training input; the
query contains all declared measurement columns in the training order.
CCA requires the coefficient target, and PF requires the existing fixed
two-to-four-factor principal-factor functional. No other scoring/extraction
method is substituted.

## Statistical contract

Queries, component counts and full targets must be fixed independently of
training data. The saved identified sign chart remains fixed in all draws;
queries never select or change it. Results propagate training-estimator
uncertainty only: they are neither future-observation prediction intervals
nor latent individual posterior distributions. IID observed units, finite
fourth moments, regular identified population geometry and nonzero first-order
coordinate variance are assumptions. Numeric sample guards do not establish
these assumptions. Frequencies count actual independent units; analytic,
survey or correlated-replication interpretations remain unsupported.

Each score uses the point/draw's training centering and, where required,
scales and coefficient matrices. PF regression score coefficients are
`R^-1 L` for the accepted orthogonal loading orientation. CCA scores use
the accepted raw canonical coefficients and retain X/Y cross-covariance.
Stable origin-plus-offset centering avoids subtraction of rounded large means.

Every original point fit and seeded training draw must be numerically replayed
from the complete saved source before use; a checksum alone is insufficient.
No failed draw is dropped, replaced, reordered or repaired. Results retain
point scores, every planned joint draw, complete B-1 covariance, SE, bias and
marginal linear percentile intervals. P-values, inferential df and simultaneous
coverage are unavailable. The complete original fit, query values, physical
positions, duplicate/MultiIndex typed identities and missing-row policy persist.
The saved `query` table contains the complete-case scoring measurements;
excluded physical rows have an all-missing sentinel and retain their identities.

## Admission and evidence

Resident CPU float64 only. Bound source metadata/table geometry before
serialization, decoding, tensor copies or refitting. Inherit source limits
(10,000 physical rows, 16 variables, 1,999 draws, 100,000 frequency units where
applicable); add at most 128 physical query rows and 128 joint query-score
coordinates, 250 million combined work units and a 32 MiB complete export.
The helper also admits at most 500,000 cells in the complete saved training
tables. Combined replay/output bounds can be reached before individual source
limits; these are implementation/resource limits, not statistical limits.
Check live workspace admission before replay and query allocations.

Acceptance per domain requires independent NumPy algebra/refits for complete
draw vectors and inference, literal-copy frequency references, shifted/scaled
data, missing/zero-frequency rows and typed identities. Test forged/resealed
source geometry, invalid queries, budgets and complete JSON restore/replay.
Keep legacy results and catalogues intact. Source, frozen/installed/native,
current-head/current-base hosted checks, merge identity and fresh tracker
readback are distinct gates. The broad parent remains open. No licensed vendor
execution, CUDA proof or public-release claim is required or inferred.

## Usage

```python
fit = oe.canon_fweight_bootstrap(
    training, ["x1", "x2", "x3"], ["y1", "y2", "y3"],
    weights="frequency", components=2, target="coefficients",
    anchors=["x1", "x2"], replications=199, seed=19,
)
scores = oe.canon_bootstrap_scores(oe.restore_summary(oe.summary_state(fit)), query)
display(scores["estimates"])
```

The [eight-domain example](../examples/multivariate_score_eight.py) saves all
training and query results, including complete joint inference, typed identities,
table order/dtypes and LaTeX. Console previews remain separate from those full
portable results. See the acceptance receipts (internal evidence excluded from this public snapshot).
