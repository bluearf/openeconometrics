# Known Bernoulli and multi-arm design inference

These three procedures estimate arm means and average-effect contrasts for the
original fixed population supplied by the caller. The experiment's actual known
assignment law must be declared. Their covariance table is an **estimated
conservative design bound**, whose expectation dominates the unidentified true
randomization covariance in positive-semidefinite order. It is not the actual
covariance or a guarantee that each observed bound exceeds that covariance.
Normal intervals and tests are pointwise and asymptotic, subject to the stated
design CLT and variance-bound consistency conditions.

```python
import openecon as oe

b = oe.bernoulli_neyman_ate(
    data, "outcome", "treated", "assignment_probability",
    design="bernoulli_randomized",
)
m = oe.multiarm_neyman_ate(
    data, "outcome", "arm", arms=["control", "low", "high"],
    design="complete_randomized",
    contrasts={"high_low": [0, -1, 1], "dose_average": [-1, .5, .5]},
    null_values={"high_low": 0, "dose_average": 0},
)
s = oe.stratified_multiarm_neyman_ate(
    data, "outcome", "arm", "stratum", arms=["control", "low", "high"],
    design="stratified_randomized",
)
oe.causal_design_save(s, "multiarm.json")
restored = oe.causal_design_load("multiarm.json")
```

The examples require the corresponding columns and original design topology.
They do not infer an assignment law from data.

## Independent known Bernoulli assignment

`bernoulli_neyman_ate(data, y, treatment, probability, *, design,
null_effect=0., alternative="two-sided", level=.95, missing="raise",
device="cpu", weights=None, max_work=100_000_000)` requires
`design="bernoulli_randomized"`. `treatment` contains numeric 0/1, and
`probability` contains the **true prespecified** independent treatment assignment
probability πᵢ, strictly between zero and one, for every original individual.
Fitted observational propensity scores are unsupported. Supplying a column is a
caller declaration, not authentication of its provenance. No clipping is used.

For fixed potential outcomes and N original individuals, the raw
Horvitz–Thompson arm means are

\[
\widehat\mu_0=N^{-1}\sum_i (1-A_i)Y_i/(1-\pi_i),\qquad
\widehat\mu_1=N^{-1}\sum_i A_iY_i/\pi_i.
\]

The ATE estimate is their difference, computed directly from original-row
contributions. These estimates use fixed N, without normalization by observed
arm counts or weights. They are not Hájek means and need not lie within the
observed outcome range. All-zero and all-one assignments are admitted; the
unobserved arm has zero observed HT sum and zero estimated bound, which does not
identify its finite-population mean or establish certainty. Constant observed
outcomes can have a genuine nonzero HT contrast under a realized assignment.

With L mapping the two means to `(mean_control, mean_treated, difference)`, the
full three-by-three bound is

\[
\widehat B=L\,\operatorname{diag}\left(
N^{-2}\sum_i (1-A_i)Y_i^2/(1-\pi_i)^2,
N^{-2}\sum_i A_iY_i^2/\pi_i^2\right)L'.
\]

It uses uncentered second moments with no HC1 correction. Independent Bernoulli
assignment gives the exact expectation identity

\[
E(\widehat B)-\operatorname{Cov}(L\widehat\mu)
=N^{-2}L\sum_i
\begin{bmatrix}Y_i(0)\\Y_i(1)\end{bmatrix}
\begin{bmatrix}Y_i(0)&Y_i(1)\end{bmatrix}L'\succeq0.
\]

This identity is an explicit derivation for the implemented bound. A usual
independent-assignment asymptotic interpretation additionally requires
probabilities bounded away from zero and one along a population sequence,
Lindeberg/no-dominating-score conditions, second-moment-bound consistency and a
nondegenerate contrast CLT. Admission of a finite data set does not verify those
asymptotic conditions. `alternative` can be `"two-sided"`, `"greater"`, or
`"less"`; `null_effect` is the weak-null ATE value.

## Fixed-count multi-arm and stratified multi-arm assignment

`multiarm_neyman_ate(data, y, treatment, *, arms, design, contrasts=None,
null_values=None, level=.95, missing="raise", device="cpu", weights=None,
max_work=100_000_000)` requires `design="complete_randomized"`.
`stratified_multiarm_neyman_ate(data, y, treatment, strata, *, ...)` requires
`design="stratified_randomized"` and otherwise has the same keywords.

`arms` explicitly declares 3–8 ordered distinct **typed** labels. Boolean,
integer, float and string labels remain distinct: `True`, `1`, `1.0` and `"1"`
can designate different arms. Each declared arm must occur at least twice in
the complete population, and at least twice **in each original stratum** for the
stratified design. Strata also use typed identities. The experiment is uniform
over assignments having the original arm counts; assignments in different
strata are independent. Counts, groups and the population cannot depend on
outcomes or be reconstructed after deleting missing outcomes.

`contrasts` is an ordered mapping of 1–32 names to K fixed finite coefficients,
with a nonzero row and **exactly zero represented coefficient sum**. Exact
dyadic integer certificates are retained. An approximately zero sum is refused;
for example `[.1, .2, -.3]` does not sum exactly to zero as binary64 inputs. The
default is every pair difference `difference[a,b] = mean[b] - mean[a]` in declared
arm order. `null_values`, when supplied, must name every contrast exactly once.
Otherwise every contrast null is zero. Tests are two-sided.

Let L stack the K arm-mean identity rows and the declared contrast matrix. With
unbiased arm sample variances sₐ² and fixed counts nₐ, complete-design estimates
are the observed arm means and their contrasts, with

\[
\widehat B=L\operatorname{diag}(s_a^2/n_a)L',\qquad
E(\widehat B)-\operatorname{Cov}(L\widehat\mu)
=L S_{\mathrm{potential}}L'/N\succeq0.
\]

S is the finite-population sample covariance matrix of all K potential outcomes,
using denominator N−1. It is unidentified because each individual reveals one
potential outcome. The retained bound includes shared covariance for overlapping
contrasts and can be singular.

For strata of fixed size Nₕ, let wₕ=Nₕ/N. The original-population target and
estimator use size weights, not equal-stratum or inverse-variance weights:

\[
\widehat\mu=\sum_h w_h\widehat\mu_h,\qquad
\widehat B=\sum_h w_h^2 L\operatorname{diag}(s_{ha}^2/n_{ha})L',
\]
\[
E(\widehat B)-\operatorname{Cov}(L\widehat\mu)
=\sum_h w_h^2 L S_{\mathrm{potential},h}L'/N_h\succeq0.
\]

The normal interpretation assumes a fixed number of arms/strata, positive
limiting arm fractions in nonnegligible strata, no dominating centered potential
outcome, bound consistency and a nondegenerate contrast CLT. No cluster,
post-treatment stratum, Bernoulli-conditioned count or unequal-probability
complete-design interpretation is supplied by these two methods.

## Numerical, result and persistence contract

All three methods require resident data, `missing="raise"`, `device="cpu"` and
`weights=None`. Dataset replay and generic sampling weights are unsupported.
Roles must be distinct. Original topology is validated before outcome sampling.
Original numeric outcomes, contrast coefficients and null values must fit
binary64 **exactly**; the procedures refuse silent changes such as `2**53+1` or
an exact `Fraction(1,3)` control. Floating input dtypes wider than binary64 are
explicitly unsupported.

Finite Torch float64 operations retain original-row contributions using bounded
64-part FastTwoSum expansions. Multi-arm point calculations use a common outcome
anchor, exact two-part products and integer-count numerators, with one common
denominator and an expanded division-remainder correction. Rounded arm/group
means are not aggregated. Anchor subtraction must have a zero exact TwoSum
remainder; otherwise original uncentered outcomes are retained. Constant
outcomes have exactly zero multi-arm
contrasts, including every saved group estimate. Bernoulli points restore the
actual observed inverse-probability anchor coefficients; those coefficients are
not replaced by the location-invariant multi-arm coefficients.

This is a strict numerical domain, not arbitrary-precision arithmetic. Native
expansions require gradual binary64 underflow; an active caller FTZ/DAZ mode is
refused by a read-only arithmetic probe and is never changed. Multi-arm
N times the least common multiple of all original within-stratum arm counts must
not exceed 2⁵³, so all numerator count factors and the denominator are represented
integers. Two-part product scaling must be exact and finite. Nonzero underflow,
overflow, expansion-capacity overflow, absorbed scientific location contributions,
nonrepresentable nonzero confidence shifts and materially indefinite unrepaired
covariance bounds are refused. No epsilon variance, clipping, PSD repair or
independence substitution is used.

`effects` retains each coordinate's point, standard error, null, normal statistic,
p-value, interval, undefined df and degenerate-variance flag. Arm means have no
weak-null test. Zero estimated standard error retains the point and covariance
bound, but leaves the statistic, p-value and **both interval endpoints undefined**.
It does not fabricate a point interval. Intervals are marginal: no simultaneous
coverage, omnibus test, exact Student-t, Fisher weak-null or finite-sample normal
coverage is asserted.

`covariance_bound` retains every coordinate pair. `subjects` retains original
positions, assignments and outcomes (also probabilities for Bernoulli).
Multi-arm `arms`, `groups` and `contrast_matrix` retain typed arm identities,
counts, weights, every group's scientific estimates/bound and declared contrasts.
Checksummed scientific state retains selected source identity and indices,
original inputs, design topology, point expansions/anchors/count certificates,
full covariance bounds, all group summaries, settings, assumptions and resource
plans. `causal_design_save/load` preserves every table's order, dtype and state,
and refuses checksum or semantic metadata tampering. Inputs and global RNG,
Torch dtype/device and thread settings are preserved.

Before selected data/numeric copies, complete work and workspace plans cover
original topology, worst-case 64-part sums, products, covariance/group tables,
scientific state and escaped labels. Admission is limited to 100,000 resident
rows and finite scalar design identifiers with bounded text. Original index
labels must be bounded numeric, text or calendar scalars; compound/custom index
objects are refused before generic string conversion. String identifiers/index
labels have at most 1,024 UTF-8 bytes. Escaped JSON bytes and repeated retained
copies are charged. These are implementation resource guards, not statistical
sample-size recommendations.

## Independent acceptance evidence and references

Owned acceptance tests enumerate all 90 three-arm assignments with counts
`[2,2,2]`, all 210 assignments with `[2,2,3]`, and the entire 8,100-assignment
Cartesian law for two three-arm strata. Independent original-outcome formulas
check each point and every covariance entry, unbiasedness and the **whole-matrix**
expectation gap against the theoretical potential-outcome PSD correction.
Heterogeneous known Bernoulli probabilities use all 32 assignments with their
independent product probabilities and verify the raw HT matrix identity, including
empty arms. Additional oracles cover overlapping contrasts, unequal stratum
sizes, large-location cancellation, exact constant-null group persistence, typed
identities, underflow/overflow refusals, resource refusal before copies, unchanged
global state and full checksum-protected round trips.

[Li and Ding (2017), *General forms of finite population central limit theorems
with applications to causal inference*](https://arxiv.org/abs/1610.04821) develops
vector randomization covariance and finite-population CLTs; its general covariance
formula gives the multi-arm unidentified PSD correction.
[Aronow and Samii (2013), *Conservative variance estimation for sampling designs
with zero pairwise inclusion probabilities*](https://www150.statcan.gc.ca/n1/pub/12-001-x/2013001/article/11831/section2-eng.htm)
provides the design-based Horvitz–Thompson and nonmeasurable-variance context.
The Bernoulli and stratified full-matrix expectation identities above are stated
explicitly for the implemented estimators rather than claiming a fitted-nuisance
or finite-sample normal guarantee.
