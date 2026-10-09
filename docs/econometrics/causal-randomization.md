# Declared assignment tests and paired signed-rank sensitivity

`randomization_test`, `stratified_randomization` and `rosenbaum_rank_bounds`
return complete `TableSet` results. `causal_design_save` and
`causal_design_load` retain their original sample positions, typed identities,
settings, assumptions, all tables and scientific state with verified checksums.
The kernels use native CPU float64 Torch; pandas handles table presentation.
SciPy and NumPy are independent development oracles, not estimation engines.

## Complete and stratified assignment designs

```python
result = oe.randomization_test(
    data, "outcome", "treated", design="complete_randomized",
    null_effect=0.0, method="exact", alternative="two-sided",
)
blocked = oe.stratified_randomization(
    data, "outcome", "treated", "stratum", design="stratified_randomized",
    null_effect=0.0, method="monte_carlo", draws=9999, seed=1729,
)
```

The caller must declare the assignment design explicitly. A treatment label,
pairing, balance table or observed association does not establish randomization.
Treatment is numeric 0/1; Boolean treatment labels are refused.

The complete design conditions on the original observed treated count and gives
every subset of that size equal probability. The stratified design conditions
on the original treated count in each stratum, assumes independent assignment
across strata, and requires both arms in every stratum. Stratum identifiers are
finite typed numeric, Boolean or text scalars: `True`, integer `1` and float
`1.0` identify different strata. Large integer identities remain exact.

Both procedures test Fisher's **constant additive sharp null**:
`Y_i(1) - Y_i(0) = null_effect` for every unit. Imputing untreated outcomes as
`observed_y - null_effect * observed_treatment` gives the outcomes used under
every candidate assignment. The complete statistic is the treated-minus-control
mean; the stratified statistic is the sum of within-stratum contrasts weighted
by `stratum_n / total_n`. No estimated effect covariance, effect standard error,
average-effect confidence interval or rejection rule is invented.

Original assignment/stratum topology is checked before outcomes. These two
procedures accept only `missing='raise'`; deleting outcome rows would change the
declared assignment universe. Construct a substantively justified analysis
population before calling the procedure, rather than selecting it through an
outcome-missing option.

`method='exact'` enumerates all `choose(n, n_treated)` assignments, or the full
Cartesian product of within-stratum combinations. `method='monte_carlo'` draws
uniform fixed-count subsets using a private CPU Torch generator and independent
within-stratum permutations. It retains every assignment and statistic and uses
`p = (extreme_draws + 1) / (draws + 1)`. The reported plug-in binomial MCSE and
worst-case MCSE bound describe simulation uncertainty; they are not uncertainty
for an estimated causal effect. Exact enumeration draws no random numbers.

Alternatives are `greater`, `less` and `two-sided`; the last compares absolute
statistics. A positive common numerical scaling avoids overflowing the statistic
calculation. The result saves both normalized and original-unit statistics,
the scaling constant and the recorded roundoff tolerance used to include ties.
Underflow that loses a nonzero input/contribution or reported statistic fails
explicitly; no positive epsilon creates precision or repairs the data.

## Paired signed-rank hidden-bias bounds

```python
bounds = oe.rosenbaum_rank_bounds(
    matched, "outcome", "treated", "pair_id",
    gammas=[1.0, 1.5, 2.0], alternative="greater", null_effect=0.0,
)
```

Each original pair must contain exactly one treated and one control row. Missing
assignment/identity and broken topology are refused before outcome deletion.
Explicit `missing='drop'` may remove a whole outcome-missing pair; deleting just
one member cannot repair the design. Whole-pair availability must be fixed
independently of treatment assignment. Pair identities retain scalar types.

The statistic ranks the absolute null-adjusted treated-minus-control contrasts.
Exact zeros are conditioned out and reported. Tied magnitudes receive average
ranks; doubled ranks are integers, so native dynamic programming gives their
complete finite distribution without rounding fractional ranks away. For a
`greater` alternative, sum the ranks of positive contrasts; for `less`, sum
the ranks of negative contrasts. Only one-sided sensitivity bounds are supported.

Under independent assignments across pairs, Gamma bounds the within-pair odds
ratio of treatment. Each favourable-sign probability lies between
`1/(1+Gamma)` and `Gamma/(1+Gamma)`. A positive-rank upper tail is monotone in
each probability, so its exact lower/upper bounds come from the corresponding
endpoint laws. At Gamma=1, both bounds equal the conditional signed-rank
assignment p-value. These are sharp-null p-value bounds, not causal estimates,
Hodges–Lehmann intervals, ATE intervals or evidence that matching removed bias.

The result retains ranks, zero counts, every probability mass and upper tail
at every doubled score for every Gamma, complete pair data, normalization
diagnostics and assumptions. Probability branches that underflow fail closed;
the complete law is never silently truncated or replaced by a normal
approximation. Only verified float64 roundoff in tail probabilities may be
clipped to the mathematical endpoints 0/1. No distribution renormalization or
vendor exact-ties parity is claimed.

## Resource and validation scope

All three procedures require resident tables and explicitly refuse Dataset
replay, generic observation weights, and CUDA/MPS. The shared input domain is
100,000 rows and 64 selected columns. `max_work` and the workspace budget are
checked before constructing assignment matrices or rank-DP vectors; a large
support returns an error rather than a partial exact law or an automatic
Monte Carlo fallback. Complete assignments and distributions also receive
conservative output/state buffer plans. The budget estimates are not process
RSS limits. `draws` and `seed` are recorded Monte Carlo settings; exact results
do not depend on their random values.

`tests/test_causal_sensitivity_randomization.py` independently enumerates all
fixed-count/stratified assignments, sharp-null imputations and rank-sign laws.
It also enumerates every heterogeneous endpoint-odds vector in small matched
fixtures to verify the sensitivity extrema, including tied ranks. Tests cover
private RNG, typed strata, complete persistence, missing topology, degeneracy,
extreme numeric ranges and resource refusal. This is method-specific validation;
it does not enable a Stata/SPSS/EViews parity or universal causal-validity flag.

Primary references:

- [Cattaneo, Idrobo and Titiunik (2024), sharp-null assignment inference](https://rdpackages.github.io/references/Cattaneo-Idrobo-Titiunik_2024_CUP.pdf).
- [Rosenbaum (1987), sensitivity analysis for matched permutation inference](https://doi.org/10.1093/biomet/74.1.13).
- [Rosenbaum's exact signed-rank implementation and references](https://search.r-project.org/CRAN/refmans/DOS/html/senWilcoxExact.html); its documented no-ties domain is distinct from this bounded doubled-score extension.
