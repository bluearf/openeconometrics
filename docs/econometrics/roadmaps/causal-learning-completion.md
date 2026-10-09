# Causal learning completion (MARKET-157)

The requested additions extend existing PLR, teffects, DiD, RD and mediation
without replacing their delivered methods. Runtime estimation remains native
CPU float64 Torch; the catalogue remains Torch-free.

## Implementation stages

1. C1: cross-fitted binary IRM/AIPW ATE and ATET, deterministic row or
   whole-cluster folds, train-only linear/lasso or forest nuisance models,
   explicit overlap admission, IID and one-way cluster influence covariance.
2. C2: binary-instrument/binary-treatment IIVM LATE, separate outcome and
   treatment regressions by instrument, orthogonal reduced-form/first-stage
   ratio, first-stage relevance checks and full influence inference.
3. C3: native CART regression/probability forest nuisances; a three-way
   honest doubly robust CATE forest (nuisance training, split selection and
   leaf estimation are disjoint, including clusters). Persist all partitions,
   trees, leaf score state, training support and prediction integrity.
   This is an honest DR forest, not a claimed reproduction of GRF's
   instrumental-variable splits or little-bag variance.
4. C4: declared-group GATEs, predeclared finite-basis CATE projections,
   independent learned-ranking GATES, full joint covariance and heterogeneity
   Wald tests. Per-row IATE output means predicted CATE at observed X; the
   individual counterfactual effect is not identified.
5. C5: outcome-independent declared policy value, held-out learned-policy
   value, paired policy contrasts, overlap/balance diagnostics, robustness
   checks, runnable example, source/installed-wheel/native-runtime evidence.

## Contracts and acceptance

- Outcomes and treatment roles are distinct; treatment/instrument are binary.
  Numeric controls/bases are initially supported. Unsupported categories,
  complex survey/frequency weights, Dataset collection, multiway clusters and
  non-CPU requests fail explicitly. Optional positive importance weights define
  a declared weighted population; they do not imply survey-design inference.
- No clipping of propensities, trimming rows, hidden ridge, fallback models,
  dropped failed folds/trees or outcome-driven policy selection.
- Scores, sample positions, folds, fitted nuisance specifications, target,
  assumptions, covariance/SE/df/p/CI, convergence and resource plans persist.
- Query intervals for honest DR forests describe conditional local forest
  averages and sampling uncertainty; approximation bias and a universal
  pointwise CATE confidence guarantee are not asserted.
- Independent NumPy/SciPy score/covariance/solver fixtures; leakage and cluster
  partition tests; constant/heterogeneous/IV DGP recovery and coverage;
  negative overlap/weak-IV/budget/sample/state cases; JSON roundtrip.
- Existing regularized and teffects suites protect current support.
- Registry/capabilities and generated editor documentation include all APIs.
  Source tests, clean installed-wheel execution, frozen/native acceptance and
  any vendor comparisons are separate receipts. Blanket parity remains false.

## Primary references

- Chernozhukov et al. (2018), orthogonal IRM/IIVM scores:
  https://docs.doubleml.org/stable/guide/scores.html
- DoubleML group/basis targets:
  https://docs.doubleml.org/stable/guide/heterogeneity.html
- Kennedy (2023), doubly robust heterogeneous effects:
  https://arxiv.org/abs/2004.14497
- Wager and Athey (2018), honesty and forest inference:
  https://arxiv.org/abs/1510.04342
- GRF algorithm/causal forest reference:
  https://grf-labs.github.io/grf/reference/causal_forest.html

These references justify score/target choices, not vendor parity.

