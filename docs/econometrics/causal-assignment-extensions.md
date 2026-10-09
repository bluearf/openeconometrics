# Assignment laws, average effects and population bounds

MARKET-656–663 extend the open MARKET-187 / GitHub157 scope with eight
separately defined methods. Earlier completed causal procedures remain supported.

| Issue | API | Target |
|---|---|---|
| MARKET-656 | `cluster_randomization` | Fixed-count whole-cluster Fisher sharp-null test, using unit-weighted cluster totals. |
| MARKET-657 | `bernoulli_randomization` | Independent known, possibly unequal Bernoulli probabilities and unconditional Horvitz–Thompson Fisher statistic. |
| MARKET-658 | `neyman_ate` | Complete-randomized finite-population average effect and conservative Neyman variance. |
| MARKET-659 | `stratified_neyman_ate` | Prespecified independent fixed-count strata and sample-size weighted average effect. |
| MARKET-660 | `cluster_neyman_ate` | Individual-average effect from intact cluster randomization and scaled cluster totals. |
| MARKET-661 | `paired_neyman_ate` | Independent fair two-unit pair assignments and conservative average-effect variance. |
| MARKET-662 | `manski_ate_inference` | IID population endpoint covariance and conservative outer confidence region for the whole identified interval. |
| MARKET-663 | `stratified_lee_bounds` | Prespecified discrete-covariate Lee bounds with always-selected population weights. |

The two Fisher procedures test a constant additive effect for every original
unit. Exact support and explicitly requested private Monte Carlo are distinct
execution modes. Bernoulli assignment includes all-zero and all-one vectors;
conditioning on the observed treatment count would change the declared design.

The four Neyman procedures permit heterogeneous effects. They estimate a
conservative covariance bound; the true finite-population covariance is generally
unidentified. Normal intervals and weak-null p values require the stated
large-sample randomization conditions. They have no exact finite-sample normal
or Student-t coverage guarantee. Unequal cluster sizes use a fixed original
unit denominator and scaled cluster totals.

Manski population endpoint scores share the same random component under fixed
common outcome support, hence their joint covariance is singular. Its outer
confidence region has asymptotic whole-set coverage. Conditional Lee aggregation
weights cells by covariate mass times their lower selection probability; sample
selection counts alone are wrong when treatment fractions differ across cells.
Lee returns identification endpoints without population confidence intervals.

See the complete method assumptions, formulas and independent references:

- [Cluster and Bernoulli assignment laws](causal-assignment-laws.md)
- [Design-based Neyman average-effect inference](causal-neyman.md)
- [Population endpoint inference and conditional Lee bounds](causal-population-bounds.md)

Every method saves all tables, typed sample positions, design membership,
scientific state and assumptions through `causal_design_save/load`. Native
float64 Torch CPU kernels reject unsupported Dataset, device, generic weights,
missingness and resource requests explicitly. Work and workspace limits never
select a different inference method or silently truncate the target.

The [executable example](../examples/causal_assignment_eight.py) and separate
source, wheel, frozen and installed-app receipts establish the accepted method
contracts. They do not establish licensed-vendor parity, a public binary release,
advanced noisy-pretrend sensitivity or the remaining broader nuisance models.
