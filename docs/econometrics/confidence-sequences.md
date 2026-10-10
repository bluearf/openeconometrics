# Conservative confidence sequences for ordered iid samples

Eight public helpers compute a confidence interval at **every loaded prefix**.
The reported family may be inspected repeatedly and the reporting time may
depend on the observations. The targets, columns, known scales and population
supports must be fixed before observing the data.

```python
import openecon as oe

data = oe.DataFrame({"converted": [0, 1, 0, 0, 1, 1, 0, 1]})
result = oe.bernoulli_confidence_sequence(
    data, ["converted"], sampling_model="iid_bernoulli", alpha=0.05,
)
display(result["intervals"])
saved = oe.summary_state(result)
restored = oe.restore_summary(saved)
```

## Construction and guarantee

For a **fixed** family of `m` marginal targets, prefix `n` receives
`a_n = alpha / (m*n*(n+1))`. Its two-sided interval has fixed-time error at most
`a_n`. Since `1/(n*(n+1)) = 1/n - 1/(n+1)`, summing over all positive integers
and all `m` targets is `alpha`. The countable union bound therefore gives

`P(every target is covered at every integer time) >= 1-alpha`.

This implies coverage at any data-dependent reporting time under the declared
sampling assumptions. Independence between columns is unnecessary. Each
column's observations must be iid over rows. The guarantee does not allow
choosing new targets, changing supports, dropping inconvenient rows, resetting
the error budget or repeatedly restarting an experiment at the same alpha.

This elementary summable-error construction is conservative. Its intervals
need not be nested and do not claim mixture-martingale or iterated-logarithm
widths. The infinite-horizon argument is mathematical; this implementation
accepts a finite resident prefix and recomputes its intervals. It does not
implement a streaming continuation object.

## Eight model contracts

Let `q = a_n/2`, `S` be the prefix sum, `M` the prefix maximum, and `s²` the
unbiased sample variance. Quantile subscripts denote lower-tail probabilities.

| Public helper | Explicit sampling model and target | Fixed-time interval before numerical widening |
| --- | --- | --- |
| `bernoulli_confidence_sequence` | `iid_bernoulli`; numeric 0/1 probability | Equal-tail Clopper–Pearson beta limits; 0 or 1 at the corresponding boundary |
| `poisson_confidence_sequence` | `iid_poisson`; mean count per equal-exposure row | `[chi²(2S)_q/(2n), chi²(2S+2)_(1-q)/(2n)]`; lower zero if S=0 |
| `normal_mean_confidence_sequence` | `iid_normal`; mean with supplied known positive population SD | `mean ± z_(1-q)*sd/sqrt(n)` |
| `student_mean_confidence_sequence` | `iid_normal`; mean with unknown positive population variance | `mean ± t(n-1)_(1-q)*s/sqrt(n)`, n>=2 |
| `normal_variance_confidence_sequence` | `iid_normal`; unknown-mean population **variance** | `[(n-1)s²/chi²(n-1)_(1-q), (n-1)s²/chi²(n-1)_q]`, n>=2 |
| `exponential_mean_confidence_sequence` | `iid_exponential`; mean of uncensored, known-zero-origin exponential observations | `[2S/chi²(2n)_(1-q), 2S/chi²(2n)_q]` |
| `uniform_endpoint_confidence_sequence` | `iid_uniform_zero`; theta in Uniform(0, theta), known zero origin | `[M/(1-q)^(1/n), M/q^(1/n)]` |
| `hoeffding_confidence_sequence` | `iid_bounded`; mean with supplied population support [A,B] | `mean ± (B-A)*sqrt(log(2/a_n)/(2n))`, intersected with [A,B] |

The exponential pivot follows because `S/mean ~ Gamma(n,1)`. The uniform pivot
follows because `P(M/theta <= u) = u^n` for 0<=u<=1. These are exact model
pivots, not asymptotic normal approximations. Binomial/Poisson discreteness and
the union bound generally give coverage above the nominal minimum.

Student mean and normal variance have no informative interval at n=1. Exactly
constant observed Gaussian prefixes also return the full parameter space,
with `status='zero_observed_variation'`; rounded data cannot manufacture a
zero-width inferential result. Unbounded endpoints are JSON null with explicit
flags, never finite placeholders. The unused n=1 error allowance is not
redistributed to later times.

## Inputs, persistence and resource limits

- Select 1..8 distinct numeric columns and 1..512 ordered rows. Numeric 0/1 is
  required for Bernoulli. Poisson counts are nonnegative integers, with a
  cumulative maximum of 1,000,000 per marginal. Nothing is silently truncated.
- Declare `sampling_model` explicitly. Missing/infinite/boolean/complex/text
  observations, duplicate column names, wider-than-float64 values and integers
  outside the exact float64 domain are refused. Row indexes may be typed,
  duplicated or unsorted; the sample retains their encoded identities and
  original positions. Time means **row position**, not the index label.
- `sd` is a list aligned to columns or an exact-name mapping of known SDs.
  `bounds` similarly contains one prespecified (lower, upper) support per
  column. Observed ranges or estimated SDs cannot establish these assumptions.
- Use `0.001 <= alpha < 0.5`. Absolute observations and known scales/support
  endpoints are bounded by 1e100. Positive continuous observations and known
  SDs must be at least 1e-100. These are implementation/resource limits, not
  statistical limits. A named workspace plan is admitted before copying the
  numeric sample; no GPU or Dataset route is advertised.
- Native CPU float64 kernels compute both tails directly. Each finite endpoint
  is widened by 128 ulps of **that endpoint**, followed by outward rounding and
  intersection with known parameter support. Numerical overflow is refused.
- `summary_state` retains every interval, complete ordered numeric sample,
  typed identities, sample hashes, known SD/support declarations, assumptions,
  method name, error schedule and allocated/spent/remaining budgets. Restore
  requires no refitting. The sample and declarations also suffice for an
  explicit fresh call to the same helper.

Weights, unequal Poisson exposure, overdispersion, serial dependence, censoring,
unknown/estimated distribution origins, adaptive target families and online
continuation state are unsupported. Ordinary fixed-sample helpers remain
separate; their intervals are not automatically optional-stopping valid.

## References and verification

[Howard et al. (2021)](https://arxiv.org/abs/1810.08240) defines time-uniform
coverage and its relationship to arbitrary stopping. The implementation here
uses the elementary union-bound proof above rather than that paper's sharper
mixture/stitching algorithms. Fixed-time normal mean, variance and exact
binomial pivots are documented by [NIST](https://itl.nist.gov/div898/handbook/prc/section1/prc14.htm),
[NIST variance limits](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/sdconfli.htm)
and [NIST exact binomial limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm).

Independent tests compare every reported prefix with SciPy beta, chi-square,
gamma, normal and Student-t quantiles; enumerate complete Bernoulli paths to
check **any-time first crossing**, rather than only final-time coverage; and
verify exact prefix-extension invariance, full saved replay, boundary paths,
model refusals and resource admission. Source, packaged/frozen and actual
installed-app acceptance are recorded separately in the dated evidence.
