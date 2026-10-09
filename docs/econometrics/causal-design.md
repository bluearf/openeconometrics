# Causal balance, sensitivity and treatment targets

These eight procedures extend the existing causal-learning and treatment-effects
families. Each returns a `TableSet`: named editable tables plus complete sample,
settings, assumptions and scientific state in `result.attrs['state']`.
`causal_design_save(result, path)` and `causal_design_load(path)` retain every
table, dtype and state field with SHA-256 checksums. No refitting occurs on load.

All kernels use native CPU float64 Torch; resident DataFrames are required.
`missing='raise'` is the default; explicit `drop` records original zero-based
positions and original row labels. Numeric treatment is exactly 0/1. Generic
weights, Dataset replay, GPU and unspecified assignment designs fail explicitly.
Full workspace and structural work are checked before substantial allocations;
`max_work` never causes a partial computation or a silently shortened sample.

| Procedure | Accepted target and uncertainty |
|---|---|
| `ebalance(data, treatment, x, ...)` | KL donor weights targeting treated numeric moments, relative to optional positive `base_weights`; strictly positive full-rank interior solution and declared residual tolerance. Preprocessing has no effect covariance. |
| `cem(data, treatment, x, cutpoints=..., categorical=..., ...)` | Exact matching of prespecified numeric bins and categorical cells, both-arm strata only; treated weights 1, controls `n_t/n_c`, unmatched 0. Target is the retained treated population. |
| `balance(data, treatment, x, balance_weights=..., ...)` | Before/after means, fixed original pooled-SD standardized differences, variance ratio, weighted ECDF distance and ESS. Full independent-row fixed-weight HC1 covariance concerns arm means, not estimated-weight causal inference. |
| `rosenbaum_bounds(data, y, treatment, pair, ...)` | Exact one-sided paired sign-test p bounds at declared Gamma odds ratios; zero contrasts conditioned out. Does not implement signed-rank or general pretrend sensitivity. |
| `paired_randomization(data, y, treatment, pair, design='paired_randomized', ...)` | Equiprobable independent treatment assignment within complete pairs and a sharp constant additive null. Complete sign enumeration or private-seed MC with plus-one p, precision and all assignments/statistics. |
| `treatment_cdf(data, y, treatment, design='randomized', thresholds=..., ...)` | Marginal `F1-F0` on a fixed grid, both arm CDFs, full joint sample covariance and pointwise normal intervals conditional on arm counts. |
| `treatment_quantile(data, y, treatment, design='randomized', quantiles=..., ...)` | Marginal inverse-ECDF quantile difference (`ceil(n*q)-1`), independent-arm seeded bootstrap, all replicates, complete covariance/SE and percentile intervals. P-values are not invented. |
| `treatment_rmst(data, time, event, treatment, design='randomized', tau=..., ...)` | Common prespecified restricted mean survival contrast from right-censored KM curves; complete risk/event/censor tables and integrated Greenwood covariance/normal intervals. |

Randomization and predeclaration are user declarations; the program cannot verify
them from a data table. Distribution, quantile and survival targets additionally
require independent population sampling and consistency/SUTVA. Survival requires
independent censoring within each arm and common observed support through `tau`;
there is no extrapolation. Quantile bootstrap intervals need continuous outcomes
and positive density at the chosen quantiles; ties/zero empirical variance are
reported, without claiming exact or universal small-sample coverage.

Paired inputs must have exactly one treated and one control observation per
original pair. Missing assignment/identity and broken pairs are refused; explicitly
dropping two missing outcomes may remove a complete pair under the separately
declared assignment-independent availability assumption.

Balanced covariates do not prove unconfoundedness. Marginal quantile differences
do not identify the distribution of individual unobserved treatment effects.
Zero empirical variance produces an explicitly degenerate interval and undefined
z/p, rather than artificial positive precision. CDF/RMST normal intervals remain
untruncated and pointwise. No Stata/SPSS/EViews parity flag is enabled.

Entropy acceptance additionally retains a corrected primal equality certificate
whose probabilities exceed `128*float64_epsilon*(number_of_moments+1)`. Boundary
and numerically weak-support targets fail even when the approximate dual score is
small. Positive base-weight normalization that underflows to zero also fails.
The supported interior domain is numerical, rather than every mathematically
feasible arbitrarily small probability. CEM categorical keys retain scalar types,
so Boolean and numeric labels are distinct.

[Runnable eight-method synthetic example](../examples/causal_design_eight.py).
[Pre-implementation stages](roadmaps/causal-design-eight.md).

Primary references:

- [Hainmueller entropy balancing](https://web.mit.edu/~jhainm/www/Paper/eb.pdf).
- [Iacus, King and Porro CEM software and author references](https://gking.harvard.edu/cem/).
- [Austin balance diagnostics](https://doi.org/10.1002/sim.3697).
- [Rosenbaum paired sensitivity model](https://doi.org/10.1093/biomet/74.1.13).
- [Cattaneo, Idrobo and Titiunik sharp-null randomization](https://rdpackages.github.io/references/Cattaneo-Idrobo-Titiunik_2024_CUP.pdf).
- [Firpo marginal quantile treatment targets](https://doi.org/10.1111/j.1468-0262.2007.00738.x).
- [Royston and Parmar restricted mean survival](https://doi.org/10.1186/1471-2288-13-152).


Eight additional sensitivity/assignment methods are described in [causal-sensitivity.md](causal-sensitivity.md), including OLS partial-R2, RR E-values, Manski/Lee identification bounds, complete/stratified randomization, exact signed-rank Gamma bounds and fixed external bias-box intervals. These extend the original eight procedures without changing their contracts.

Eight further [observational targets and OVB options](causal-observational-targets.md) add estimated-propensity CDF/quantile targets, cross-fitted AIPW CDFs, explicitly known-nuisance survival/RMST scores, and formal OVB benchmarks and robustness thresholds. Each retains its own identification, inference and numerical admission conditions.

Eight [confidence, multiarm and individual-effect extensions](causal-confidence-extensions.md) add finite-grid Fisher acceptance sets, known-Bernoulli and complete/stratified multiarm Neyman inference, and pointwise sharp discrete individual-effect CDF/quantile bounds. Their exact, asymptotic and identification-only interpretations are preserved separately.
