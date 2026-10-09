# Cluster and unequal Bernoulli assignment tests

`oe.cluster_randomization` and `oe.bernoulli_randomization` test Fisher's
constant additive sharp null under two explicitly declared assignment laws.
They preserve the complete original analysis population. These are separate
designs from the existing complete, stratified and paired randomization APIs.
Each returns an editable `TableSet` with `test`, `design` and `assignments`
tables and a complete checksummed scientific state.

```python
cluster_test = oe.cluster_randomization(
    data, "outcome", "treated", "school",
    design="cluster_randomized", null_effect=0.0, method="exact",
)
bernoulli_test = oe.bernoulli_randomization(
    data, "outcome", "treated", "true_assignment_probability",
    design="bernoulli_randomized", null_effect=0.0, method="exact",
)
simulation = oe.bernoulli_randomization(
    data, "outcome", "treated", "true_assignment_probability",
    design="bernoulli_randomized", method="monte_carlo", draws=301, seed=1729,
)
oe.causal_design_save(simulation, "assignment-test.json")
restored = oe.causal_design_load("assignment-test.json")
```

Both signatures take `data, y, treatment` and the named `cluster` or
`probability` column. Keyword options are required `design`, `null_effect=0.0`,
`alternative="two-sided"`, `method="exact"`, `draws=9999`, `seed=1729`,
`missing="raise"`, `device="cpu"`, `weights=None`, and
`max_work=100_000_000`. Alternatives are `greater`, `less`, and `two-sided`;
methods are `exact` and `monte_carlo`. Seeds are integers in `[0,2**63-1]`;
draws are integers in `[1,1_000_000]`. Exact runs consume neither seed nor draws,
but still validate their declared types and bounds.

Assignment design and predeclaration are caller assumptions. Treatment labels
alone cannot establish an experiment. Bernoulli probabilities must be the true
externally fixed assignment probabilities, independent of these outcomes;
fitting observational propensities and supplying them here does not establish
this randomization law.

## Fixed-count assignment of whole clusters

Every original cluster has one treatment value. Cluster labels are nonmissing
finite typed numeric, Boolean or text scalars; text and numeric labels, Boolean
and integer labels, and large integer identities retain distinct meanings.
First appearance determines cluster order. Both cluster arms must occur;
cluster sizes may differ.

Let `G` be the number of clusters, `K` the observed number of treated clusters,
and `N` the fixed total number of original units. The sharp null is
`Y_i(1)=Y_i(0)+tau` for every unit, with supplied `tau=null_effect`.
First impute `r_i=Y_i-tau*D_i` at the unit level. For a candidate assignment
`a_g` with exactly `K` treated clusters, the statistic is

```
T(a) = (G/N) * (mean_treated(sum_{i in g} r_i)
                        - mean_control(sum_{i in g} r_i))
```

This uses scaled cluster totals and fixed `N`; it is the unit-weighted
Horvitz–Thompson contrast. Means of cluster means instead weight each cluster
equally. Dividing by the received-arm unit counts introduces random
denominators. Neither replaces this statistic when sizes differ. Imputation
must precede aggregation: subtracting `tau` from the original HT contrast is
generally wrong because realized treated unit counts vary across assignments.
The statistical comparison uses the equivalent sum of unit contributions,
retaining cancellation remainders rather than first discarding them in rounded
cluster totals.

Exact inference enumerates every one of the `choose(G,K)` equiprobable cluster
assignments. MC uses an independent uniform private permutation for each draw,
selecting its first `K` clusters. Complete cluster and expanded unit assignment
strings are retained. Every assignment carries its original uniform mass
`1/choose(G,K)`. The design table retains cluster type, label, size, treatment
and imputed total; state retains every original member position.

The design requires no interference across clusters and well-defined
cluster-assignment potential outcomes for every unit. Within-cluster exposure
is the declared entire-cluster treatment. No paired/stratified cluster law,
unequal cluster-assignment probability or estimated compliance correction is
implicitly supplied.

## Independent unequal-probability Bernoulli assignment

Each original unit receives treatment independently with its supplied numeric
`p_i` strictly inside `(0,1)`. Treatment is numeric 0/1, never Boolean. An
observed all-zero or all-one assignment is valid; all such candidate assignments
also remain in the unconditional universe. The HT statistic is

```
T(a) = mean_i[r_i * (a_i/p_i - (1-a_i)/(1-p_i))]
```

There is no arm-mean denominator, so an empty received arm is well-defined.
Exact inference enumerates all `2**N` binary vectors with masses
`product_i[p_i**a_i * (1-p_i)**(1-a_i)]`. It sums the original masses of
assignments at least as extreme as observed. Unweighted fractions of extreme
assignments are wrong for unequal probabilities. Neither the observed treated
count nor selected covariate-balance events condition this law.

MC compares independent private CPU uniform draws with each `p_i`, sampling
from the same unequal Bernoulli law. MC tail counts are therefore unweighted:
sampling from the law already accounts for assignment probabilities. Every
draw's binary vector, original joint probability, statistic and extremeness
flag persist. The design table retains outcomes, null-imputed outcomes,
received assignments and both unit assignment probabilities.

## Inference and numerical admission

The one-sided tail uses the specified sign; two-sided extremeness uses
`abs(T(a)) >= abs(T(observed))`. Comparisons retain a recorded roundoff tolerance
in a common normalized statistic scale. A distinct, nonzero comparison gap
within that bound invalidates the run instead of converting a small genuine
difference into a tie. Exact probabilities retain a full-law
normalization certificate. A full-support tail is exactly one; only verified
roundoff outside the probability endpoints is clipped. Small positive
assignment masses and tails are retained, not thresholded to zero.

For `M` MC draws and `B` extreme draws, `p=(B+1)/(M+1)` includes the observed
assignment. Reported MC SE is the plug-in simulation diagnostic
`sqrt(M*p*(1-p))/(M+1)`; its maximum is `sqrt(M)/(2*(M+1))`.
This describes simulation error, not sampling uncertainty of a treatment
effect. These procedures report no effect covariance, SE, df or confidence
interval. They do not test a weak-null average effect or invert a p-value grid
into an effect confidence set.

Native Torch CPU float64 performs probability products and statistics. A
private generator preserves global Torch RNG and device/dtype settings.
Signed sums use a bounded 64-part floating expansion. All inputs must fit the
resident common domain of at most 100,000 rows and 64 selected columns, with
finite raw numeric magnitudes at most `1e150`.
Original index labels, column names and cluster labels are each limited to
256 bytes after canonical ASCII JSON escaping, before sample copies or saved
digests. This includes six-byte NUL escapes and Unicode surrogate pairs;
typed integer labels retain their exact values within the bound. Work and
conservative workspace plans cover the complete assignment tensors, all expansion parts, probability
calculations and persisted strings/state before substantial assignment
allocation. The common resource contract excludes caller inputs and private
allocator/BLAS overhead; this is not an RSS guarantee. A default MC request can
exceed `max_work`; reducing explicitly requested draws or raising the declared
budget is required. Exact support failure never triggers an automatic MC run.

`Dataset`, generic weights and non-CPU devices fail explicitly. Outcomes and
all original topology/probability fields must be complete; `missing="drop"`
is refused. Positive assignment masses that underflow, a probability absorbed
in its complement, null shifts absorbed at outcome scale, lost nonzero
statistic contributions, overflow and expansion-capacity failure invalidate
the entire run. No tiny probability clipping, missing-row deletion or failed
draw skipping repairs the calculation. Large Bernoulli MC supports retain the
exact symbolic `2**N` support specification when a materialized cardinality
would exceed the recorded scalar domain.

Complete artifact persistence retains table order/dtypes, physical sample
positions and typed row labels, input schema/hash, design topology,
probabilities, null imputation, every assignment/statistic/mass/flag, work and
workspace certificates and RNG provenance. Load performs no refitting or
redrawing. Independent tests enumerate cluster and unequal Bernoulli laws,
check every statistic and probability, replay every MC draw, verify unequal
cluster-size targets and nonzero sharp-null imputation, and check typed full
artifacts, resource refusals and numerical boundaries. Method validation,
source/wheel/frozen/native evidence and licensed vendor parity remain distinct.

Primary references:

- [Branson and Bind, Bernoulli-trial randomization inference](https://arxiv.org/abs/1707.04136).
- [Su and Ding, cluster totals and unit-average targets](https://arxiv.org/abs/2104.04647).
- [Li and Ding, finite-population randomization and its inferential limits](https://arxiv.org/abs/1610.04821).
