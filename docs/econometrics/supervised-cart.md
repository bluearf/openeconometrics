# Explicit supervised partitions and weighted CART

MARKET-683 implements the resident numeric CART and partition stage under
MARKET-363. Production numerical work uses Torch CPU float64. NumPy and native
R/rpart are development references. This stage does not complete the wider
supervised-learning parent.

`prediction_split` returns a typed `SplitState`. It accepts an explicit role per
physical row, private-seed iid random assignment, private-seed atomic-group
assignment, or chronological atomic-time-block assignment. Fractions refer to
atomic unit counts. Largest-remainder allocation reserves one unit for each
positive fraction, then distributes remaining units. Outcomes and case weights
do not enter partition assignment. Groups cannot cross roles. Chronological
roles must satisfy max(train time) < min(validation time) < min(test time),
omitting empty roles; ties stay in one atomic time block. Integer time comparisons
retain integer arithmetic above 2**53. Supplying group/time with explicit roles
checks these same leakage rules. No implicit outcome stratification is performed.

`cart(data, outcome=..., features=[...], split=..., task=...)` fits only the
declared training positions. The default split is private-seed iid 60/20/20.
The full original source, typed index, row positions, column dtypes, missing
entries, options, split and class map are saved. No observations are silently
dropped. Resident DataFrames and sized column mappings are supported;
Dataset, generators, GPU inputs and implicit collection are refused. Continuous
predictors require real numeric dtypes. Nonexact integer-to-float64 conversions,
complex/bool/wider-than-float64 numeric measurements and infinities are refused.
Classification labels are complete homogeneous integers or bounded strings;
the sorted training labels define the class order. Unknown validation classes
refuse validation selection. Unknown test classes preserve prediction and mark
test scoring unavailable. Explicit query scoring with an unknown class refuses.

## Training-only adapters

Complete numeric predictors are the default. `impute="mean"` explicitly uses
the case-weighted mean of observed training values. Each predictor must have at
least one observed training value. This is a mean adapter, not CART surrogate
split routing. `standardize="population"` uses weighted training moments after
the imputation step, with denominator sum(training case weights); constant
predictors receive scale one with an explicit flag. `"identity"` retains units.
The saved means/centers/scales are used unchanged for validation/test/query.
Test outcomes or features cannot change the maximal tree, transformations or
selected subtree. Validation outcomes can choose a subtree when explicitly
requested, and are not incorporated in leaf estimates afterward.

## Splitting, risk and pruning

Regression splits maximize weighted SSE reduction. Classification splits
maximize weighted Gini reduction. Every feature and every ordered distinct-value
boundary is examined, subject to physical `min_leaf`, parent `min_split`,
`min_weight_fraction` of total training weight, and declared `max_depth`.
Midpoints use half-low plus half-high; if adjacent representable values have no
representable midpoint, the lower value is the threshold. Routing uses <= on the
left. Within a 256*float64-epsilon objective-relative arithmetic envelope,
feature order then ascending threshold resolve ties. A gain within its zero
rounding envelope does not create a split. This numerical convention is recorded
by the versioned algorithm and is separate from the statistical tree definition.

Case weights are strictly positive **loss weights**. They are not survey,
frequency-replication or analytical-inference weights. Normalization must retain
every positive weight; excessive dynamic range is refused. Regression uses
centered outcome units for stable sufficient statistics. The saved normalization
converts node risk to outcome units. Constant-response nodes stop exactly.

Classification pruning risk is weighted misclassification, not Gini. Regression
pruning risk is weighted SSE. For an active subtree T below node t, the weakest
link is `g(t) = [R(t) - R(T)] / [number of leaves(T) - 1]`. All tied minimal links
are removed together. An ancestor covers a tied descendant. Saved edits retain
the complete path through the root without duplicating whole trees. Displayed
`cp = alpha / R(root)` is invariant to scaling all weights or regression outcome
units. `select="cp"` applies all pruning events with event_cp <= requested cp,
including zero-alpha events. `select="validation"` minimizes weighted validation
RMSE or misclassification; tied losses choose fewer leaves. No train+validation
refit is performed. Test labels never select a subtree. `max_nodes` is a resource
ceiling: if the required maximal tree exceeds it, fitting refuses instead of
returning a silently truncated tree.

## Losses and uncertainty

Regression reports weighted RMSE, MAE and R² with explicit weight denominators;
constant observed outcomes have unavailable R². Classification retains complete
weighted confusion matrices, class precision/recall/F1 and support, macro and
support-weighted averages with their defined-class denominators, multiclass
Brier (sum over classes), log loss, and binary AUC using weighted pairwise ties
at half credit. Binary AUC is unavailable if either class is absent. A true-class
probability zero produces `log_loss=None` and `log_loss_unbounded=True`, without
probability clipping. Ten top-label reliability bins are descriptive diagnostics,
not a fitted calibrator. Training impurity importance names its training loss
estimand; it is not held-out permutation importance.

Adaptive CART covariance, SE, df, p values and confidence/prediction intervals
are explicitly unavailable. Leaf summaries and probabilities are point estimates;
no conditional-leaf covariance is presented as adaptive-tree inference.

## Complete state and replay

`cart_restore` checks source types/dtypes/representability, row identity, split,
training moments, every cached node partition/statistic, exhaustive split
maximality, all weakest-link pruning stages, validation choice, predictions,
importance and full metrics. It never invokes `_grow` or an estimator to replace
cached values. Rehashed numerical forgeries still refuse. Digests detect
corruption; they do not authenticate a source supplied by its owner. Public
result JSON export performs the same semantic replay before serialization.

`cart_predict` returns a complete saved query result containing its frozen model,
original query source/dtypes/index, optional scoring outcomes/weights, leaf routes,
predictions/probabilities, full scores and unavailable uncertainty. Repeated and
unsorted rows remain in physical order. `cart_prediction_restore` replays every
query row using the saved model and transforms. `CartState`, `CartQueryState`
and `SplitState` provide typed transport. Typed JSON parsing, model copy/deepcopy,
result copies and dump formatting are admitted before their allocations and
revalidate semantics. Ambient default dtype/device and global random state are
preserved.

The implementation ceilings are 20,000 resident/query rows, 64 features, 32
classes, depth 20 and 2,047 nodes. They are resource bounds, not statistical
requirements. Worst-case sorting, all class prefix buffers, full node membership,
all pruning/path validation metrics, sorting for AUC, semantic replay, complete
source/query storage and JSON/copies are charged before materialization. The
default work budget is 500 million operations, configurable up to 200 billion;
the global workspace default is 512 MiB and the portable metadata envelope is
64 MiB. These plans may refuse substantially smaller geometries. No sampling,
candidate grid or output truncation is substituted after refusal.

## Primary and independent references

Breiman, Friedman, Olshen and Stone (1984), *Classification and Regression Trees*,
defines the CART/weakest-link method. The executed package-author reference is
Therneau and Atkinson's [rpart introduction](https://stat.ethz.ch/CRAN/web/packages/rpart/vignettes/longintro.pdf)
and [rpart 4.1.27](https://stat.ethz.ch/CRAN/web/packages/rpart/index.html).
Its GPL-2 | GPL-3 source/data are used externally for development only; no author
code or dataset is copied into production or the repository fixture. The pinned
tar SHA256 is `3183552d74f02749a70e2b989591c561ca0f7c146034f415748acc8480ca4050`;
the introduction PDF is `bfaf9d334344b5ffa5622f136acbabb6943fd78e8e03ad9d1daa57b2b7fbf5a6`.

For validation-selected models, native rpart selects the matching full-cptable
leaf-count stage at an interior cp from its own interval. Fixed-cp models use
the requested fixed cp. This avoids interpreting a saved stage as a critical
cp query in another engine whose event cp may differ by last-bit arithmetic.
Every cp/risk/count value still compares independently at the fixed tolerance.
Production event equality and adjacent representable cp/feature threshold
queries have explicit boundary regression tests.

Native R 4.6.1/rpart 4.1.27 compares complete maximal node memberships, weights,
physical risks, thresholds, class probabilities, every cost-complexity cp/risk/
leaf-count stage and selected predictions. Controls fix cp0 for maximal growth,
xval0, minbucket/minsplit/depth, no surrogate/competitor splits, and empirical
case-weighted class priors. The final prior is adjusted by the floating remainder
only to satisfy rpart's exact sum(prior)==1 input check. Original full 81-row
package-author kyphosis data and its default minsplit20/minbucket7/cp.01 example
are checked externally in addition to original repository synthetic fixtures.
Independent NumPy directly centered losses exhaust all candidate boundaries.
Before execution the fixed native tolerances are rtol2e-10, atol2e-12;
memberships/features/classes/path counts are exact. Package-author comparison
does not claim execution of proprietary Breiman CART, IBM or Stata.

Run `examples/supervised_cart.py`, then the development oracle with `--state`
pointing at the **actual saved result**. That mode forbids OpenEcon tree growth;
it compares saved cached values against independent NumPy and actual native
rpart. Frozen/installed application Run, persistence after a real quit/relaunch,
merge and tracker acceptance remain separate gates owned by integration.

Categorical predictors, CART surrogates, CHAID/QUEST, fitted calibration,
cross-validation, permutation importance, forests/boosting, KNN, SVM and neural
methods are separate remaining parent stages.
