# Gaussian finite-sample joint inference

`ols_stepdown`, `simultaneous_t_ci` and `hotelling_region` use CPU float64
Torch. Existing normal-limit `simultaneous_ci` and supplied-null `stepdown`
retain their contracts. Neither a Student-t marginal df nor an arbitrary
estimated covariance establishes a finite-sample joint t law.

## Fixed-design OLS and known-precision WLS

```python
fit = oe.ols(data=df, y="outcome", x=["income", "education"], covariance="nonrobust")
tests = oe.ols_stepdown(fit, terms=["income", "education"],
                       null_values=[0, 0], error_model="iid_gaussian")
```

Model and family must be specified independently of outcomes. Design and
sample selection are fixed or independent of errors. Independent Gaussian
errors have one common variance. OLS covariance is `s²*(X'X)^-1`, and
`s²/σ²` is an independent `chi2(n-k)/(n-k)`. Thus coefficient pivots are
`N(0,R)/sqrt(chi2(df)/df)`: **one denominator shared across the family**.
R is known from the fixed design. True-null subsets retain this law regardless
of false-null coefficients (subset pivotality). The existing monotone maxT
stepdown kernel then supplies strong FWER control with Monte Carlo plus-one
p-values. This is parametric pivotal simulation without nuisance refits.

`error_model="known_precision_gaussian"` accepts `aweight` WLS only when
weights are known relative inverse error variances. Estimated, survey,
frequency or importance weights do not automatically acquire that meaning.
Robust/cluster/HAC, absorbed FE, model mixtures, outcome-selected fits and
arbitrary non-Gaussian errors are unsupported. The caller declares the
Gaussian law; covariance type, coefficient/sample identities, full covariance
versus SE and residual df are checked from the restored `ResultBundle`.

Rows retain estimates/nulls/t statistics, analytic marginal p, simulated
marginal p, adjusted Monte Carlo p and rejection. `adjusted_exceedances` is
the running maximum of stage counts: adjusted p is `(count+1)/(draws+1)`.
Monte Carlo noise may place adjusted p slightly below analytic marginal p.
Attrs retain full selected covariance/correlation, df, seed, zero failures,
source model ID, sample/data hashes, sample-position checksum, missing policy
and weight semantics. The original fit and sample are preserved.

## Known joint shape and one common independent scale

```python
intervals = oe.simultaneous_t_ci(
    [0.4, -0.2], [[0.04, 0.01], [0.01, 0.09]], df=7,
    labels=["income", "education"],
    family_description="Prespecified Gaussian regression coefficients",
    pivot_description="Known design correlation and independent common chi-square scale",
)
```

`simultaneous_t_ci` requires this same joint pivot, not separate coefficient
dfs or a robust covariance. One finite df in [1,1,000,000] is accepted.
Intervals use `estimate ± max-|t| critical*SE`; higher empirical-CDF
interpolation and CDF Monte Carlo SE `sqrt(alpha*(1-alpha)/draws)` are recorded.
The reference law is finite-sample under its assumptions, with Monte Carlo
quantile error. Unique labels determine canonical RNG order. Singular PSD
shapes are allowed when each marginal variance is positive; correlation
eigenvalues below -1e-12 fail, tiny negative roundoff is clipped and recorded.
Missing/nonfinite values, incompatible shape and unresolved endpoints fail.

A local seeded Marsaglia-Tsang gamma sampler supplies the common chi-square.
Shape augmentation handles df=1; rejection is bounded to 128 iterations.
No global RNG, private Torch sampling API, replacement or dropped draw is used.

## Unknown multivariate covariance: Hotelling mean region

```python
region = oe.hotelling_region(df, ["income", "education"],
    sampling_model="iid_multivariate_normal", missing="raise")
```

One iid multivariate-normal sample has unknown positive-definite covariance.
Complete rows are shared across variables; n>p is required. `missing="drop"`
explicitly removes whole incomplete rows and records original positions; the
iid normal assumption must still hold in the retained sample. No weights,
clusters, dependent observations or `Dataset` route are accepted.

With sample covariance S, the region is
`n*(mean-theta)' S^-1 (mean-theta) <= p*(n-1)/(n-p)*F_upper(alpha,p,n-p)`.
The native F quantile is deterministic. The ellipsoid has exact 1-alpha
coverage under this model. Coordinate projections have **at least** 1-alpha
joint coverage, not exact rectangular coverage. p=1 reduces to the usual
Student-t mean interval. This unknown-covariance law is different from the
known-shape joint t law.

Attrs persist center, full mean covariance/correlation, normalization scales,
normalized precision, squared radius and F dfs. Their recorded quadratic
definition supports region membership after restoration without original data.
Correlation conditioning at 1e-12 rejects singular/unresolved covariance.
Translation/scale normalization stabilizes calculation; unrepresentable
original-unit covariance or endpoints fail. Inputs are preserved.

## Resources and evidence

Admission precedes numeric buffers and uses the global workspace budget.
Limits: 384 dimensions; 8,000,000 draw/sample cells; 100,000,000 planned
matrix-product work units; 1,000–200,000 joint-t draws; 1,000,000 stored OLS
sample identities. Joint-t work is `draws*p²+p³`; Hotelling work is
`original_n*p²+p³`. Plans exclude caller data, Python result objects, private
BLAS/allocator overhead and process RSS. No truncation or fallback occurs.

Numeric LaTeX tables and complete JSON attrs are distinct; assumptions are
not automatic table footnotes. Source/frozen evidence does not establish
vendor parity, GPU, installed user application or a public release. These
completed stages are MARKET-207/208/209. Cluster/model-specific resampling
remains in MARKET-191; non-Gaussian/weak-identification/bootstrap/inversion
regions remain in MARKET-192.

- [NIST Hotelling T²](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm): normal-mean pivot and F scaling.
- [Romano and Wolf (2005)](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf): finite-sample monotone intersection tests; our declared Gaussian pivot supplies subset pivotality.
- [Marsaglia and Tsang (2000)](https://doi.org/10.1145/358407.358414): gamma rejection sampler.
- [Independent tests](../../tests/test_finite_sample_joint.py), scientific protocol (internal evidence excluded from this public snapshot), [validator](../../scripts/validate_finite_sample_joint.py), [console example](../examples/finite_sample_joint.py).
