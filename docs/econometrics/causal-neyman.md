# Design-based finite-population average effects

These four procedures estimate the average treatment effect for **all original
supplied individuals**, treating their potential outcomes, membership and design
groups as fixed. Only the declared randomized assignment is random. Independent
population sampling, observational exchangeability and a constant individual
treatment effect are not imposed. Assignment labels cannot establish that the
experiment actually followed the declared design.

Each returns `effects`, the full three-by-three `covariance_bound`, original
`subjects` and a design summary (`groups`, `clusters` or `pairs`). The three
coordinates are `mean_control`, `mean_treated` and `difference`. The covariance
table estimates a **conservative bound**, not the unidentified exact
randomization covariance. Its expectation dominates the true covariance in
positive-semidefinite order. A particular realized bound need not exceed the
true variance. This distinction also applies when a diagonal entry happens to
be zero.

## Complete fixed-count assignment

```python
result = oe.neyman_ate(
    data, "outcome", "treated", design="complete_randomized",
)
```

Every subset with the observed treated count is equiprobable. Both arms need
at least two original units. Let `s_a²` be the unbiased observed arm outcome
variance and `n_a` the fixed count. Arm means and their difference estimate the
corresponding fixed-population means and average effect. Define

```text
L = [[1,0], [0,1], [-1,1]]
B_hat = L diag(s_0²/n_0, s_1²/n_1) L'
```

For the average-effect coordinate the exact variance is
`S_0²/n_0 + S_1²/n_1 - S_tau²/N`, where uppercase variances refer to the full
fixed potential-outcome population. The last term is unidentified. Omitting it
gives the standard Neyman bound. For the full matrix,
`E(B_hat) - Cov(mu_hat) = L Cov_pop(Y(0),Y(1)) L'/N`.
Zero off-diagonal entries between the two arm means belong to this bound;
they are not a claim that the actual finite-population arm estimators are
independent. [Li and Ding (2017)](https://arxiv.org/abs/1610.04821).

## Independent fixed-count strata

```python
result = oe.stratified_neyman_ate(
    data, "outcome", "treated", "stratum",
    design="stratified_randomized",
)
```

Each stratum is fixed before treatment and independently follows uniform
assignment with its observed treated count. Every original stratum needs at
least two treated and two control units. Overall arm means weight stratum arm
means by `w_h=N_h/N`; the full bound is `sum_h w_h² B_hat_h`. The variance
correction omitted from the effect bound is
`sum_h w_h² S_tau,h²/N_h`. No stratum is deleted or reweighted according to
observed precision. Normal inference needs an appropriate stratified design
CLT: for example, large within-stratum populations for a fixed number of strata,
or many independent strata with no dominant contribution. The observed minima
alone do not prove these asymptotic conditions.

## Fixed-count cluster assignment and individual-average HT target

```python
result = oe.cluster_neyman_ate(
    data, "outcome", "treated", "cluster",
    design="cluster_randomized",
)
```

Every original individual is observed; there is no second-stage sampling.
Treatment must be constant within each cluster. Exactly `G_1` of the `G`
fixed clusters are selected uniformly, with at least two clusters per arm.
Interference between clusters is excluded. Outcomes may depend on the common
assignment of the individual's own cluster.

For cluster `g`, form `W_g=(G/N) sum_{i in g} Y_i`. Apply complete-randomization
means and the full bound to these scaled totals. The resulting contrast is the
Horvitz–Thompson **individual-average** fixed-population effect even with unequal
cluster sizes. It is neither the raw difference between observed individual arm
means nor the unweighted average of cluster mean effects. The cluster table saves
every size, original member position, raw total, scaled total and assignment.
Point estimates aggregate original-row contributions before rounding any
cluster total. Covariance uses original-row total differences against a smallest
same-arm reference cluster, so a small between-cluster difference survives
rounding of two large totals. Reference totals and centered quantities persist;
rounded raw totals are display summaries.
Asymptotic inference requires sufficiently many randomized clusters and
appropriate control of dominant cluster totals; a large number of individuals
inside a few clusters does not supply a design CLT.
[Su and Ding (2021)](https://arxiv.org/abs/2104.04647).

## Independent fair two-unit pairs

```python
result = oe.paired_neyman_ate(
    data, "outcome", "treated", "pair", design="paired_randomized",
)
```

Each original pair contains exactly two individuals, one treated and one
control. Assignments are independent across pairs and each orientation has
probability one half; at least two pairs are required. Pairing is fixed before
treatment. Let the observed pair vector be
`X_m=[Y_control,m,Y_treated,m,Y_treated,m-Y_control,m]`. Its mean estimates the
full-population vector, and `B_hat=sample_cov(X_m)/M`. All signed cross-arm
entries are retained. With `eta_m=E_assignment(X_m)`, direct algebra gives
`E(B_hat)-Cov(mean(X_m))=sample_cov(eta_m)/M`, a PSD matrix. For the effect
coordinate this is the across-pair variance of pair-average treatment effects
divided by `M`. Thus the familiar paired-difference variance is conservative
in expectation for fixed heterogeneous pairs. It is not an exact paired t law.
[Imai (2008)](https://imai.fas.harvard.edu/research/files/matched-pair.pdf).

## Weak-null inference and admission

The API options are `null_effect=0`, `alternative="two-sided"` (also `greater`
or `less`) and `level=.95`. Only the `difference` row has z/p values, testing the
weak null that the finite-population average effect equals `null_effect`.
Heterogeneous individual effects are allowed. Existing Fisher randomization
procedures retain their separate sharp-null assignment laws and exact/Monte
Carlo p-values; these functions do not invoke them or impute missing potential
outcomes under a constant-effect null.

Standard errors are square roots of bound diagonals. Confidence intervals are
two-sided pointwise asymptotic normal intervals even when the p-value uses a
one-sided alternative. `df=None`; no exact t, finite-sample coverage,
simultaneous region or conditional randomization p-value is claimed. The
caller must supply a design sequence with suitable nondegenerate CLT and
variance consistency. A zero empirical variance gives a point interval,
`degenerate_variance=True` and undefined z/p, rather than an epsilon-adjusted
rejection or proof of certainty.

Only resident tables, numeric finite outcomes, both numeric0/1 treatment arms,
native Torch CPU float64 and `missing="raise"` are admitted. Outcome,
treatment and identifier roles must be distinct. Typed finite scalar identifiers
keep Boolean, integer, floating and text groups separate, including large
integers. Textual identifier, index and selected-column labels are limited to
1024 UTF-8 bytes; numeric labels have a bounded serialized textual domain.
Actual escaped JSON label sizes, including repeated metadata/table copies,
enter both preflight budgets. The original assignment/group topology is checked before selecting
outcomes. Missing identities/assignments/outcomes and unsupported generic
weights, Dataset replay, GPU, unequal-probability assignments and broken designs
are refused; no group or original individual is silently deleted.

Complete work and conservative workspace plans are checked using the original
row count before any selected topology or outcome copy. The resident row limit
is100000; `max_work=100000000` is an arithmetic budget, so the full conservative
plan can refuse smaller samples. Buffer plans are admission estimates, not RSS
limits. Means and signed covariance products use native FastTwoSum expansions
with hard64-part capacity and final float64 rounding. Overflow, positive/nonzero
underflow, absorbed pair/null or variance contributions, unrepresentable
confidence shifts and matrices outside the declared PSD roundoff domain fail
explicitly. No covariance clipping, positive epsilon, dropped expansion terms
or silent replacement of the inferential law is used.

Common-location centering preserves constant outcomes exactly despite unequal
arm counts. Aggregate points sum all original-row centered weighted contributions
and restore the anchor using the design's analytic coefficient sums; cluster
HT coefficient differences use integer population/cluster counts. Separately
rounded group means, pair contrasts and raw totals are never the aggregate
point-estimate path. Anchors, coefficients and every original-row contribution
persist. An absorbed nonzero pair contrast or restored location is refused.
Complete and stratum group point summaries use the same anchored original-row
calculation and save their own anchor contributions. Their differences are
computed directly, rather than subtracting two separately rounded arm means.
Constant outcomes have exactly zero complete/stratum differences; unequal-size
cluster HT estimates can retain a real nonzero observed difference because
assignment need not balance the observed individual counts.

Complete `causal_design_save/load` artifacts retain original physical positions
and labels, typed group identities/members, every selected outcome/assignment,
fixed counts, group/cluster/pair summaries, centered quantities, complete bound,
target, inference assumptions, settings and resource plans. No replay or refit
is required. Global Torch default device and random state remain unchanged.

Independent tests enumerate complete, stratified Cartesian, cluster and paired
assignment universes from fixed heterogeneous potential outcomes. They check
unbiased means, every covariance-bound cell, exact design variance and the full
matrix expectation gap. Unequal clusters distinguish HT individual targets from
biased realized-arm means. Additional tests verify normal calculations, no
samplewise-bound claim, typed identities, original-topology refusals, full
persistence/tampering, nested cancellation, numerical gates and preallocation
work/workspace refusal. These establish the declared method scopes, not broad
vendor or whole-product parity.
