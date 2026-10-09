# MARKET-156: supervised prediction scope decision

Status: planning deliverable; all training implementations deferred. Preserve
the historical `Hayır` decisions for trees, forests, boosting, neural networks,
SVM and k-nearest-neighbor prediction. Activate only after a held-out prediction
use case, defined training limits and a saved prediction contract are accepted.
Existing discriminant/clustering and nearest-neighbor **treatment matching**
have separate targets and remain unchanged. Existing `oe.network_gnn` has a
graph-specific representation/training contract; it is not a generic tabular
MLP/RBF family and is not reimplemented by this plan.

Proposed names (`cart`, `random_forest`, `gradient_boosting`, `mlp`, `rbf`,
`knn`, `svm`) are draft registry/API names, not available methods. The product
keeps Python code as its workflow; this plan does not introduce a training
wizard or select an external runtime estimation engine. H2O-based Stata
integration and IBM procedures are separate reference implementations, not a
native Stata estimator equivalence claim.

## Deferred phase matrix

| Phase | Model/option domain and dependency | Acceptance gate |
| --- | --- | --- |
| ML-1 | Versioned training/split/transformation/prediction state, regression and binary/multiclass metrics | Row/group/time split isolation; preprocessing fitted only on training; saved inference matches without fitting; no test labels used for tuning/calibration |
| ML-2 | Continuous-feature CART regression/classification, deterministic ties, case weights and cost-complexity pruning; depends on ML-1 | Exhaustive tiny split/pruning oracle, leaf probabilities/predictions and independent reference; separate validation for missing/surrogate/category routing |
| ML-3 | CHAID, exhaustive CHAID and QUEST as separate algorithms; depends on ML-2 state contract | Original algorithm statistical tests/merging/selection and independent vendor output, not renamed CART; calibrated multiple split selection and explicit category constraints |
| ML-4 | Random forest and gradient boosting as distinct training stages; depends on ML-2 | Bootstrap/subsample/feature seeds, independent tree-ensemble predictions, held-out metrics, out-of-bag rules, training-only early stopping and reproducible checkpoint restoration |
| ML-5 | Supervised kNN and SVM classifier/regressor; depends on ML-1 | Training-only scale/category maps, exact tiny distance/kernel/optimization oracle, unknown-category policy, held-out tuning; matching estimators are not reused as predictive classifiers |
| ML-6 | MLP then RBF networks; depends on ML-1 and reviewed optimizer/architecture | Independent forward/gradient/optimizer fixtures, bounded epochs, deterministic CPU training tolerance, immutable validation split, saved transform/weights and replayed predictions |

## Training and prediction contract

Keep ordered training/validation/test row identities and dataset fingerprints.
Allow ordinary random splits only for declared iid rows; grouped or chronological
splits are explicit choices. Reject overlapping identifiers, duplicate group
leakage and tuning on a locked test set. CV operates within training rows; a
calibration fold is distinct from final evaluation. Fit imputation/scaling,
category vocabulary, feature selection and class encoding in each training
fold, then freeze them for validation/new observations. Unseen categories need
an explicit unknown bucket or refusal; unseen outcome classes cannot silently
renumber predictions. Labelled prediction input must not be used to refit.
This isolation follows the primary scikit-learn guidance on preprocessing and
leakage; runtime kernels remain our own Torch implementation.
[Leakage and preprocessing references](https://scikit-learn.org/stable/common_pitfalls.html).

Phase ML-2 starts numeric, complete predictors, squared-error regression and
Gini classification. Case weights represent loss weights, not survey-design
inference; other weight semantics are refused until validated. Category splits,
surrogate missing routing, ordinal outcomes, censored losses and arbitrary
class costs each require their own option gate. Candidate splitting/pruning
needs deterministic tie ordering and a declared minimum leaf/weight policy.
Independent reference scope for CART must match those conventions.
[Decision tree algorithms](https://scikit-learn.org/stable/modules/tree.html).

Publish RMSE/MAE and held-out R² for regression; confusion matrix, classwise
precision/recall, log loss/Brier and calibrated probability diagnostics for
classification where defined. Record row counts, denominators, class order and
which split produced each metric. AUC is unavailable for a single-class test
sample; report that sample, not an invented score. Macro and weighted metrics
are different. Probabilities must sum to one. No coefficient SE/p-value is
manufactured for a prediction-only model.

Feature importance must name its estimand and evaluation sample. Impurity
importance is distinct from held-out permutation performance loss; dependence
can make permutation scores unstable. Neither is a causal effect. Boosting
learning-rate/loss and forest sampling/feature rules are separately specified.
[Ensemble reference definitions](https://scikit-learn.org/stable/modules/ensemble.html).

## State, budget and validation plan

Persist a versioned prediction-state adapter with feature names/dtypes/maps,
scaling/imputation parameters, split hashes, fitted nodes or support vectors or
network parameters, class order, seeds and training/stopping diagnostics.
Restoration cannot launch training. A predictive result may use the current
ResultBundle prediction rendering only through an explicit compatible adapter;
unavailable inference must remain unavailable and model JSON must retain the
complete fitted state outside any display truncation.

Before training, account for resident data, encoded features, split candidates,
all tree/node buffers, kernel/support-vector storage and optimizer state.
Expose node/tree/depth, epochs/evaluations and total work limits; reject a
declared request over budget, cancel explicitly and retain old saved models.
Kernel SVM cannot silently replace exact kernels with an approximation.
Resident CPU float64 is the initial accepted route; Dataset streaming and GPU
training each need algorithm-specific admission and hardware evidence.

Future protocol freezes independent reference versions, dataset hashes,
split IDs, preprocessing, hyperparameters, seeds and tolerances before fitting.
Use exhaustive tiny oracles, complete predictions/probabilities and held-out
metrics, then adversarial missing/imbalance/unseen-category/constant-feature
fixtures. Perturbing test labels must leave every fitted parameter unchanged.
Source, frozen and native saved-prediction/restart evidence are separate gates.
These tests are specified here, not executed for unimplemented models.

Open implementation: [GitHub #42](https://github.com/bluearf/openecon/issues/42).
