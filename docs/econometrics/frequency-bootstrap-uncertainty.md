# Frequency bootstrap uncertainty for fixed spectral functionals

MARKET-617–624 define eight bounded uncertainty contracts under MARKET-185.
They extend existing point-frequency estimators and fixed unweighted bootstrap
functionals. They do not close the broader multivariate parent or establish
licensed vendor equivalence.

| Issue | API and option | Fixed inferential target |
| --- | --- | --- |
| MARKET-617 | `pca_fweight_bootstrap(..., matrix="covariance")` | Ordered covariance eigenvalues, signed eigenvectors and loadings |
| MARKET-618 | `pca_fweight_bootstrap(..., matrix="correlation")` | Ordered correlation eigenvalues, signed eigenvectors and loadings, with each draw restandardized |
| MARKET-619 | `pca_subspace_fweight_bootstrap(..., matrix="covariance")` | Leading covariance projector, captured/residual trace and captured trace share |
| MARKET-620 | `pca_subspace_fweight_bootstrap(..., matrix="correlation")` | Leading correlation projector and trace functionals, with each draw restandardized |
| MARKET-621 | `canon_fweight_bootstrap(..., target="correlations")` | Ordered positive simple interior canonical correlations |
| MARKET-622 | `canon_fweight_bootstrap(..., target="coefficients")` | Those correlations and all identified raw/standardized coefficients and within/cross loadings |
| MARKET-623 | `factor_multifactor_fweight_bootstrap(..., target=None)` | Fixed ordered unrotated principal-factor loadings and uniquenesses |
| MARKET-624 | `factor_multifactor_fweight_bootstrap(..., target=full_target)` | Fixed full-target orthogonal principal-factor loadings and uniquenesses |

All four APIs require a weight column through `weights="column_name"` and
accept only `weight_type="fweight"`. Component or factor count, variables,
identification settings, confidence, replication count and seed are declared
for the complete request. The routines use resident CPU float64 calculations;
Dataset, summary-matrix, non-CPU and dependent-sampling input are outside
these contracts.

## What a frequency means

A physical row contains one joint measurement vector and a nonnegative
integer count of actual observed units with that vector. The represented
units must be independent and identically distributed for this IID bootstrap.
Equal recorded measurements can occur across independent units; equality of
measurements does not establish their independence.

Assigning a larger count to a single observed person, cloned record, cluster
or time-series observation does not create independent evidence. The methods
exclude correlated copies and repeated measurements from the same unit. They
do not interpret frequencies as arbitrary importance, analytic, probability
or survey weights. They do not resample physical support rows as equally
likely observations, resample the supplied weights, or use a Kish effective
sample size. The observed unit count is exactly `N = sum(frequencies)` after
the declared common complete-case rule.

## Sampling and moment refitting

Let `z_i` be a complete physical measurement row and `w_i` its original
integer frequency. A bootstrap draw has complete physical-row count vector

```text
c_b ~ Multinomial(N; w_1/N, ..., w_r/N),     sum_i c_bi = N.
```

The empirical bootstrap can be represented by multinomial counts and weighted
sample moments instead of repeated measurement rows, as described by
[Chaibub Neto (2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131333).
Aggregating the unit-level categories belonging to the same physical row
gives the compressed law above. This aggregation is an algebraic consequence
of the count representation; that paper does not certify the CCA/PF geometry
or interval coverage of these APIs.

The implementation draws `N` uniform integer copy ranks in one CPU Torch
`randint` call per replication. Cumulative original integer frequencies and
a right-sided search map those ranks to physical source rows. A bincount
then produces the complete integer count vector. No rounded probability
vector is sampled, and no `N × p` expanded measurement matrix is allocated.
The temporary rank vector still has length `N`, so sampling is included in
the work and memory admission. The complete `B × r` count matrix persists.
This is fixed-size multinomial resampling, not a Poisson, Bayesian, balanced,
stratified or cluster bootstrap.

For the point fit use `a_i = w_i`; for a draw use `a_i = c_bi`. In both cases,

```text
mean_a = sum_i a_i z_i / N
S_a    = sum_i a_i (z_i - mean_a)(z_i - mean_a)' / (N - 1).
```

Computation first selects strictly positive counts, chooses an actual
positive-count row as origin, and calculates the weighted offset with a
second centering correction. Zero counts are removed before subtraction or
weighted products. The origin and offset persist separately so replay can
retain a small centered signal around a large measurement level. Every draw
re-estimates means, unbiased covariance, standard deviations and its family
geometry. Correlation PCA, CCA and PF standardize using that draw's own SDs.
Raw covariance is not replaced by point-fit moments, repaired, or inflated
by an effective-sample-size adjustment.

## Identification of the eight functionals

For PCA eigenpairs, each retained eigenvalue must be positive and separated
from adjacent roots, including the retained/discarded boundary when present.
The count is fixed; the routine does not select components. A caller-supplied
anchor per axis, or the strongest absolute point eigenvector entry recorded
once, fixes signs. Every draw must retain a nonzero anchor and remain inside
the signed 45-degree point-axis chart. There is no permutation, per-draw sign
selection or Procrustes alignment. The reported loading is the eigenvector
multiplied by the square root of its eigenvalue, in the chosen matrix units.

For PCA subspaces, the fixed rank is strictly smaller than the variable
count. Only the retained/discarded boundary requires a gap; roots inside
either block may repeat. The projector `P = V_m V_m'` is invariant to signs
and internal basis rotations. The joint vector contains its declared-order
upper-triangular entries, captured trace `tr(P A)`, residual trace
`tr((I-P) A)` and captured trace share, for covariance or correlation `A`.
These scopes do not give intervals for individual tied axes or tied roots.
Projector identities and distance diagnostics do not constitute a confidence
ball or simultaneous region.

For CCA, both within-block correlation matrices must have conditioned full
rank. Frequency-scaled standardized rows enter block QR/SVD without expanded
measurements. Every retained canonical correlation must be positive, simple
and below one, with the first omitted root separated. Root intervals are
not tests of zero correlation. The coefficient target additionally reports
raw and standardized X/Y coefficients and within/cross loadings. Canonical
variates have unit frequency sample variance and positive paired correlation.
Ordered X-variable anchors are caller declared or selected once from unique
strongest standardized point coefficients. Both signed coefficient directions
must remain in their fixed 45-degree point-covariance charts. Axes are never
relabelled or aligned. Zero, tied and perfect retained roots are refused.

Principal factors means one spectral decomposition of the correlation matrix
whose diagonal is replaced by re-estimated squared multiple correlations.
It is not ML latent-factor parameter estimation or iterated principal
factoring. The unrotated route requires positive simple retained reduced
roots, the retained/discarded gap, and fixed loading sign anchors away from
zero. Loadings and uniquenesses form one joint parameter vector.

The full-target PF route uses a finite complete `p × factors` target in the
declared variable and axis order. The caller must fix it independently of
the analyzed sample; it is unchanged in every draw. Orthogonal Procrustes
rotation has no Kaiser normalization and permits reflections. The loading
cross product with the target must identify a unique conditioned polar
factor. Internal retained roots may repeat because the target-oriented
loading functional is invariant to their arbitrary extraction basis; the
retained/discarded boundary still requires separation. The extraction-basis
rotation matrix is a nuisance mapping, not an inferred parameter. Both PF
routes refuse boundary or nonpositive uniquenesses.

## Uncertainty and its statistical limits

Every admitted result includes the complete point vector, all `B` replicate
vectors, their full joint covariance with divisor `B - 1`, standard errors,
bootstrap bias and linear-interpolated marginal percentile endpoints at the
declared confidence level. The moment divisor `N - 1` and uncertainty divisor
`B - 1` describe different calculations. P-values and inference degrees of
freedom are undefined. Joint covariance may be structurally singular because
of normalization or redundant functional coordinates; it is preserved without
an inverse, Wald test or simultaneous-region construction.

The statistical assumptions are IID complete-case units, fixed finite
dimensions/counts and identification settings, finite fourth moments, and
interior population moments with the stated population spectral/anchor/target
conditions. Numerical sample checks do not prove these population assumptions.
With missingness, the ordinary resampling target is the retained complete-case
population. The procedures provide no missingness model or correction that
would identify a different full-population target.

First-order marginal percentile coverage of a coordinate additionally needs
nonzero first-order derivative variance. A smooth coefficient, loading or
projector entry can violate that condition even when its axes are identified.
The software does not certify influence variance or empirical coverage from
one data set. Fixed-rank projectors and target-oriented PF do not inherit
specialized confidence-set theorems simply because their functionals are
invariant. The default `B=199` is a finite simulation setting, not a guarantee
of accurate tail quantiles; endpoints retain Monte Carlo uncertainty.

No exact, familywise, selected-count/selected-target, high-dimensional,
dependent-unit, survey, future-observation-noise or licensed-vendor coverage
claim is made. Every requested draw is attempted. If any refit fails a rank,
identification, uniqueness or numerical requirement, the whole uncertainty
request is refused. No failed draw is replaced, repaired or omitted in a
successful-subset covariance.

## Missing values, zeros and finite precision

Measurements and frequencies use one listwise `missing="drop"` or
`missing="raise"` policy fixed before bootstrap sampling. A missing frequency
is not interpreted as zero. Every nonmissing frequency is validated globally,
including rows whose measurement missingness would otherwise omit them.
Negative, fractional, nonfinite, Boolean, complex and string frequencies
are refused. Exact integer validation precedes float64 count conversion;
individual counts and their total may not exceed `2^53`, and the tighter
bootstrap unit limit below also applies.

Complete zero-frequency rows remain in the source sample, original-frequency
table and every draw-count table. Their draw counts are exactly zero and they
do not enter moment arithmetic. A missing measurement on such a row still
participates in the common drop/raise rule. The original source ledger retains
physical positions, typed row identities, completeness, original frequency
values and zero-frequency accounting, including omitted rows.

All admitted measurement cells must be finite real numeric values; complete
zero-frequency rows do not relax this input requirement. Standardized families
refuse zero variance or raw variance below the normal float64 range before
correlation construction. Nonconstant bootstrap variances that underflow,
nonfinite arithmetic and unresolved decompositions also refuse the result.
No clipping, ridge, variable deletion or rank reduction substitutes for the
declared functional.

## Resource and portable-state boundaries

These are implementation/resource limits, not statistical sample-size laws:

| Admission | Bound |
| --- | --- |
| Original resident physical rows | 10,000 |
| Complete literal IID units `N` | 100,000 |
| Replications `B` | 19–1,999; each percentile tail must have at least one expected order statistic |
| PCA eigenpair variables / joint parameters | 16 / 128, with the chosen count fitting the parameter bound |
| PCA subspace variables / joint parameters | 15 / 128 |
| CCA total variables / retained roots / joint parameters | 16 / 4 / 260 |
| PF variables / retained factors | 16 / 2–4, with fewer factors than variables |
| Planned cumulative work | 250 million units, including copy-rank sampling, all fits and full covariance |
| Complete portable result | 32 MiB, subject to conservative admission and actual serialized-output checks |
| Source identity domain | 4,096 bytes per encoded identity, 2 MiB aggregate, tuple depth 32 and bounded traversal |

Physical dimensions, declared roles, named workspace buffers and conservative
portable-output size are checked before selected-source copies and numerical
allocation. Global count validation and the actual complete-case `N`, identity
admission and cumulative work checks precede the selected float64/int64 tensor
copies and draws. Full count/moment/parameter/covariance buffers are included.
The public resource plan can refuse a request below an individual maximum
because combined work, caller memory budget or full output exceeds its limit.
Some point-frequency estimators admit larger integer totals; their admission
does not make this bootstrap unlimited.

## What the saved state and example prove

The complete result preserves source measurements and original frequencies,
source-position/typed-identity ledgers, every draw count, all parameters,
point/draw means, SDs, centering origins/offsets, family diagnostics, inference
tables and all fixed/RNG/resource settings. CCA/PF additionally retain full
replicate covariance matrices; PCA retains the source/count/moment state from
which its draw matrices can be refitted. Generic JSON restoration preserves
this declared state, not a shortened console preview.

Three validation layers have distinct purposes:

1. Full-state integrity and count support checks bind the saved content and
   check integer shapes, sums and zero-frequency support. A checksum does not
   certify the statistical model or that saved parameter tables were computed
   correctly.
2. Seeded semantic replay regenerates the declared integer copy ranks and
   counts, then refits the original frequencies and every saved draw count.
   Comparison covers complete vectors, moments and inference tables. A rerun
   of the production algorithm tests persistence and reconstruction consistency.
3. Independent source oracles literally expand measurement rows in tests and
   use independent NumPy/SciPy covariance eigensystems, generalized-eigen CCA,
   PF inverse/SMC decompositions and orthogonal Procrustes calculations. They
   compare full output vectors and joint inference across declared distribution
   domains. Oracle expansion is test-only; production does not expand rows.

The [eight-case source example](../examples/frequency_uncertainty_eight.py)
declares finite support atoms, independently draws unit categories to obtain
integer frequencies, and applies all eight fixed functionals with `B=199`.
It includes common missingness, complete zeros, typed/duplicate source identities,
full summary restoration and LaTeX output. Its assertions and marker prove
only what that execution checks. The example itself does not independently
certify coverage or vendor equivalence. Source oracles, frozen execution,
actual installed native Run, saved-history readback, numerical restart replay,
fresh hosted merge acceptance and tracker readback remain separate evidence
stages. This guide does not assert that any pending stage has completed or that
a package has been publicly released.

## References and attribution

- Chaibub Neto, E. (2015). [Speeding Up Non-Parametric Bootstrap Computations
  for Statistics Based on Sample Moments in Small/Moderate Sample Size
  Applications](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131333).
  *PLOS ONE* 10(6), e0131333. The article's Methods explain the empirical
  multinomial count/moment equivalence; its performance experiments are not
  OpenEconometrics performance or coverage results.
- Efron, B. (1979). [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552).
  *The Annals of Statistics* 7(1), 1–26. Original bootstrap reference, cited
  bibliographically; this guide does not claim a full-paper review or transfer
  an unspecified theorem to these eight estimator contracts.

The family-specific numerical contracts above follow the implemented fixed
PCA, CCA and principal-factor definitions. Their finite-sample guards and
independent numerical comparisons establish the bounded calculations, while
population assumptions and interval-coverage limitations remain explicit.
