# Finite-grid exact Fisher confidence sets

`randomization_confidence_set`, `paired_randomization_confidence_set` and
`cluster_randomization_confidence_set` invert exact two-sided Fisher tests on
a **caller-predeclared finite grid**. They extend existing single-null tests
with a new finite-parameter-space target. They do not estimate a heterogeneous
ATE or construct a confidence interval over the real line.

```python
complete = oe.randomization_confidence_set(
    data, "outcome", "treated", candidates=[-1., 0., 1., 2.],
    design="complete_randomized", level=.95,
)
paired = oe.paired_randomization_confidence_set(
    data, "outcome", "treated", "pair", candidates=[0., 1., 2.],
    design="paired_randomized",
)
clustered = oe.cluster_randomization_confidence_set(
    data, "outcome", "treated", "cluster", candidates=[0., 1., 2.],
    design="cluster_randomized",
)
```

Each signature requires `candidates` and `design`. Common optional arguments are
`level=.95`, `missing="raise"`, `device="cpu"`, `weights=None` and
`max_work=100_000_000`. The only computation mode is exact enumeration and the
only alternative is two-sided; there are no Monte Carlo or seed settings.

## Parameter space and coverage

Supply 1 through 64 distinct finite numeric effect candidates. Their original
order is retained. The caller must fix this grid before inspecting outcomes.
Integer candidates must be represented exactly in float64. Boolean values,
non-finite values, duplicates, strings, empty grids and larger grids are refused.
Every original numeric candidate must equal its float64 conversion. Exact dyadic
fractions remain admissible; a nonrepresentable fraction such as `Fraction(1, 3)`
or an extended floating value that changes during conversion is refused.

For each candidate `theta`, the sharp null is
`Y_i(1)=Y_i(0)+theta` for **every original unit**. The null-imputed outcome is
`Y_i-theta*D_i`. No interference, a correct declared assignment law and complete
outcomes for the fixed original assignment population are required.

For the supplied grid `G`, the returned set is

`C_G = {theta in G: p(theta) > 1-level}`.

The exact support numerator/denominator and the exact binary-float `level`
fraction determine acceptance; a rounded displayed p-value cannot change it.
Under the stated sharp-effect model and assignment law, this set has design
coverage at least `level` **if the true constant effect belongs to G**.
Coverage is one event for the true candidate, so no Bonferroni correction across
grid candidates is needed. This does not establish simultaneous inference for
multiple heterogeneous treatment effects.

An empty set means every supplied candidate was rejected. It does not reject
all effects outside the grid. An all-grid set means no supplied candidate was
rejected. Accepted candidates can be disconnected on the sorted grid; gaps,
isolated candidates and empty sets are preserved. No convex hull, interpolation,
range endpoints or whole-real-line interval is produced. Off-grid values receive
no confidence-set coverage claim.

## Three original assignment laws

Complete randomization requires both numeric 0/1 arms and fixes the original
treated-row count. Its support is all `choose(N,N_t)` unit assignments. The
statistic is the null-imputed treated mean minus control mean.

Paired randomization requires exactly two original rows per typed pair, with
one treated and one control. Assignments are independent fair choices within
each pair. All `2**P` choices are retained. The statistic is the mean of the
treated-minus-control null-imputed pair differences. Pair support bits select
the originally treated member when 1 and the originally controlled member
when 0; every result also retains full original-row assignment strings.

Cluster randomization requires a common assignment within each original
typed cluster, with both treatment and control clusters. It fixes the original
number of treated clusters. Every `choose(K,K_t)` whole-cluster assignment is
retained. Its unit-weighted fixed-`N` HT statistic is

`(K/N) * [sum(treated-cluster outcomes)/K_t - sum(control-cluster outcomes)/K_c]`.

Original cluster sizes and original-row outcomes enter the sums. Neither
cluster-average effects nor received-arm row means replace this target.
Unequal cluster sizes can give a nonzero realized HT statistic even for constant
outcomes; the entire randomization distribution accounts for this.

## Certified arithmetic and admission

Numerical kernels use native Torch CPU float64. Every candidate null imputation
is checked before any support-sized assignment allocation. A nonzero candidate
shift absorbed at the outcome scale, or a nonzero original outcome absorbed by
the shift, is refused. The exact low part of each admitted subtraction is saved.

Original-row high/low components receive reversible power-of-two scaling.
Integer count coefficients are applied through exact power-of-two terms;
the procedure never sums rounded cluster totals or rounded pair differences.
Bounded 64-part FastTwoSum expansions retain nested cancellation remainders.
Complete and cluster tests use integer-scaled numerator statistics; their
positive common reporting multiplier does not change the assignment p-value.

Two-sided comparisons use full absolute-numerator expansions. Exact-zero
comparison certificates give genuine ties. Nonzero comparisons are not rounded
into ties by an epsilon threshold. Overflow, irreversible normalization,
expansion-capacity failure, a nonrepresentable reported nonzero statistic or an
uncertifiable nonzero comparison returns `numerical_failure` for the entire grid.
An arithmetic probe requires gradual float64 underflow. Caller FTZ/DAZ mode is
refused without changing that mode. Default device, default dtype and global RNG
are also preserved.

All selected columns and original rows remain in the assignment population.
Only resident data, complete numeric outcomes and exact numeric 0/1 treatment
are supported. Boolean outcomes are refused. Original integer outcomes must be
represented exactly in float64; their admission precedes numeric tensors and
assignment support allocation. Exactly represented large integers remain
supported within the shared magnitude limit. Floating outcome dtypes wider than
8 bytes are refused before conversion. Selected roles must be distinct. `missing="drop"`, generic
weights, Dataset replay and GPU devices are unsupported. Typed finite scalar
identities distinguish Boolean, integer, floating and text labels; large integer
labels and repeated original indices persist. Index, identity and selected-name
labels must fit the existing 256-byte escaped-label domain. Outcomes inherit
the shared finite float64 input domain, including magnitude at most `1e150`.

The complete work/workspace plan multiplies grid size by assignment support
and includes original sample copies, null high/low parts, every assignment,
both expansion certificates, full result tables and serialized scientific state.
The common row limit is 100,000. Budget refusal occurs before constructing the
support; no grid truncation or approximate inference follows. These are explicit
buffer plans rather than process-RSS guarantees.

## Complete tables and persistence

The five tables are `profile`, `accepted_candidates`, `design`, `assignments`
and `tests`. The last table contains every candidate-by-assignment statistic,
normalized numerator, comparison sign, extreme flag and exact-tie flag.
Each support vector has a full original-row bit string and uniform PMF.

State retains the original outcome/treatment vectors, typed topology, original
sample labels/positions, candidate order, every null high/low part, every
statistic/comparison expansion, p-value fractions, exact acceptance threshold,
accepted-grid components and complete resource plans. Covariance, standard
errors, df and continuous confidence intervals are explicitly unavailable.
All tables, dtypes and scientific state round-trip through
`causal_design_save/load` with integrity checks.

`tests/test_causal_confidence_sets.py` independently enumerates every support
cell with `itertools`, exact `Fraction` arithmetic and `math.fsum` checks.
It exhausts realized randomized datasets under true grid candidates to check
coverage, and tests disconnected/empty/full sets, non-dyadic imputation
remainders, severe cancellation, large common locations, supported subnormal
rescaling, FTZ refusal, labels, topology, resource refusal and exact artifact
restoration. These are new inversion/state/coverage checks; passing the previous
single-null tests alone would not validate this target.

Primary references:

- [Cattaneo, Idrobo and Titiunik (2024)](https://mdcattaneo.github.io/books/Cattaneo-Idrobo-Titiunik_2024_CUP.pdf), Section 2.2.1: sharp-null adjustment and test inversion, including finite candidate grids.
- [Luo, Dasgupta, Xie and Liu (2021)](https://arxiv.org/abs/2004.08472): discrete Fisher p-value functions and restrictions needed when constructing ordinary intervals.
- [Su and Ding (2021)](https://arxiv.org/abs/2104.04647): unit-average targets from scaled cluster totals, distinct from cluster-average targets.
