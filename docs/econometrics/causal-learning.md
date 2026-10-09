# Orthogonal and honest causal learning

The existing teffects, DiD, RD, mediation and PLR methods are preserved.
These additions estimate declared causal targets using native CPU float64
Torch kernels. No sklearn, DoubleML, statsmodels or SciPy estimator runs at fit
time. Numerical reference packages are development-only.

| API | Target and supported inference |
| --- | --- |
| `oe.dmlirm(..., estimand="ate"/"atet")` | Binary cross-fitted IRM/AIPW ATE or ATET |
| `oe.dmliivm(..., instrument="z")` | Binary interactive-IV LATE |
| `oe.dmlcate(..., basis=["income"])` | Best linear CATE projection on a predeclared numeric basis |
| `oe.dmlgate(..., group="region")` | Predeclared group average effects |
| `oe.dmlgate(..., groups=4)` | Independently learned sorted GATES |
| `oe.causalforest(..., query=[[...]])` | Honest doubly robust forest conditional local averages |
| `oe.policyvalue(..., policy="rule", cost=.25)` | Predeclared probabilistic policy value and paired benchmarks |
| `oe.policyvalue(..., learned=True)` | Independent held-out learned-policy value |
| `oe.causal_predict(result, new_data)` | Saved basis/forest CATE prediction with complete query covariance |

All fitted APIs accept `data`, `y`, `treatment` and numeric controls `x`.
Outcome/treatment/instrument roles are distinct. Treatment and instrument use
exactly 0/1. Numeric controls and fixed numeric CATE bases are supported;
categorical encoding must be prepared explicitly as pre-treatment features.
Group labels may be scalar text or numeric values.
Declared basis columns, GATE group indicators and nonconstant policy
probabilities enter the nuisance conditioning design as well as the target.
This prevents using an outcome/treatment model that omits the target's
conditioning variable. The complete numeric/indicator design is persisted;
group indicators replace a redundant raw group-code feature.

## Scores and targets

IRM uses the cross-fitted AIPW potential-outcome scores
`phi_d = mu_d + 1{D=d}(Y-mu_d)/P(D=d|X)`. ATE is their weighted contrast.
ATET solves the treated-share ratio score, accounting for its estimated
denominator rather than treating the treated fraction as fixed.

IIVM fits instrument propensity, outcome regressions and treatment regressions
separately by instrument level. LATE is the orthogonal reduced-form contrast
divided by the orthogonal first stage. The first-stage lower confidence bound
must exceed the declared `min_first_stage`. This guard does not certify
strong-instrument asymptotics; conditional independence, exclusion, monotone
increasing compliance and relevance bounded away from zero remain assumptions.
Reorient a decreasing instrument explicitly.

GATE and fixed-basis CATE use full joint influence covariance, with joint
heterogeneity Wald tests. CATE projection intervals concern the declared best
linear projection. Pointwise equality with the true CATE additionally needs a
correct/sufficient basis. Singular bases and unidentified contrasts fail
explicitly; no columns are silently removed.

## Nuisance learners and honest forests

`learner="linear"` uses unpenalized QR outcome models and logistic treatment
models. `"lasso"` and `"ridge"` use a fixed declared `penalty`, training-only
scaling and native penalized logistic probability models with checked KKT
convergence. `"forest"` uses training-only, without-replacement CART subsamples.
`trees`, `max_depth`, `min_leaf`, `mtry`, `split_candidates` and the explicit
Bernoulli `leaf_prior` determine the forest. Options that do not apply fail;
no hidden tuning, propensity clipping or model fallback occurs.

IRM/IIVM/basis/declared-group/predeclared-policy procedures use deterministic
`folds` with all predictions outside the nuisance training sample. A supplied
cluster stays wholly in one fold.

Honest forests, learned GATES and learned policies use three disjoint samples:
A fits nuisances, B chooses splits/rankings/policies, C supplies effect scores.
Cluster assignment keeps whole clusters together. C's features constrain leaf
support, but C's outcomes never select splits, nuisance models, rankings or
policies. Empty/undersized leaf support is not repaired by skipping trees.

This is an honest **DR forest**, supported by DR-learner and honesty principles.
It is not an asserted reproduction of GRF's local-moment splitting or little-bag
variance. Query intervals concern conditional local forest averages and score
sampling variability; forest approximation bias and universal pointwise CATE
coverage are excluded. Per-row output predicts CATE at that person's features;
it does not identify that person's unobserved individual counterfactual effect.

## Covariance, diagnostics and persistence

`HC0` sums normalized influence outer products. `HC1` multiplies by N/(N-1).
One-way cluster covariance sums whole-cluster contributions, applies G/(G-1)
and uses t(G-1) coefficient inference. Cluster fold/partition and evaluation
counts are recorded. Joint heterogeneity tests use an asymptotic chi-square
reference and require adequate independent clusters.

Positive `weights` with `weight_type="iweight"` define a declared pre-treatment
importance-weighted population; normalization is scale-invariant. They do not
provide complex survey or frequency-replication inference. Nuisances estimate
conditional means in the observed sampling population; weight exogeneity is an
assumption. Zero/negative weights fail.

Propensity ranges, effective inverse-weight counts, standardized covariate
balance, and explicitly labeled RA/IPW/AIPW robustness comparisons are saved.
These are diagnostics, not tests establishing causal identification.
Policy value subtracts declared per-treatment `cost`; it saves full covariance
with treat-all, treat-none and paired gain. A deterministic gain is recorded
exactly without inventing a positive standard error.

ResultBundle persists positions, folds/partitions, nuisance models, scores,
targets, assumptions and full covariance/SE/df/p/CI. Forest/basis state carries
a version and SHA-256 integrity check; saved prediction does not refit.
Predictions preserve original index order and reject missing/out-of-support
queries.

## Admission and evidence

Resident inputs require at least 30 complete rows, at most 128 controls and
sufficient rows in every nuisance/leaf/group stratum. CATE bases have at most
32 columns; declared groups have at most 20 levels. At most 1000 simultaneous
CATE queries are supported because their complete covariance is retained;
larger inputs need explicit smaller query batches. Memory and `max_work`
budgets are checked. Work counts are conservative kernel operation accounting,
not wall-clock performance measurements.

Dataset collection, multiway clusters, non-CPU routes, survey/frequency
weights and unsupported categories are refused. `missing="drop"` is an
explicit complete-case policy; original positions remain auditable.
`nobs` describes the admitted input used across training/estimation, while
honest inference/evaluation positions identify the independent C sample.

Existing `dmlplr` additionally supports whole-cluster folds/covariance and
`nuisance="forest"` with a validated `forest_config` dictionary. Existing
lasso/ridge/elastic-net and inferential-lasso behavior is preserved.

Runnable example: [causal_learning.py](../examples/causal_learning.py).
Source tests, independent DoubleML/reference checks, bounded DGP coverage,
installed-wheel execution and frozen/native acceptance are distinct receipts.
No blanket Stata/vendor parity is claimed.

Primary sources:
[DoubleML orthogonal scores](https://docs.doubleml.org/stable/guide/scores.html),
[group/basis targets](https://docs.doubleml.org/stable/guide/heterogeneity.html),
[Kennedy DR learner](https://arxiv.org/abs/2004.14497),
[Wager–Athey honesty](https://arxiv.org/abs/1510.04342).
