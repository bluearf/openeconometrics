# Mixed-data TwoStep clustering

`oe.twostep` builds an actual CF tree from resident data, then repeatedly
merges the nearest pair of retained leaf summaries through the complete
hierarchy. Continuous columns and categorical columns are declared separately.
K-means and existing hierarchical clustering remain available.

```python
fit = oe.twostep(data=df, continuous=["income", "age"],
                 categorical=["region"], threshold=0.02,
                 criterion="bic", max_clusters=8)
profiles = oe.twostep_profiles(fit)
quality = oe.twostep_quality(fit)
saved = oe.twostep_save(fit)
restored = oe.twostep_load(saved)
prediction = oe.twostep_assign(restored, data=new_people)
three = oe.twostep_cut(restored, 3)
```

For a CF with count N, centered sum of squares M2 and category counts n_l,
the implemented score is
`xi = -N*(sum_cont .5*log(global_variance + M2/N) + sum_cat H)` with
`H = -sum_l (n_l/N)*log(n_l/N)` and `0*log(0)=0`.
Merge loss is `xi(A)+xi(B)-xi(A union B)`. Stable parallel Welford updates
avoid subtracting nearly equal raw second moments. Each continuous column is
centered and divided by its complete-training population standard deviation;
both variance terms use population divisors. Constant continuous columns are
rejected. Category maps distinguish bool, int, float and string identities;
single-level categorical columns contribute zero entropy and zero parameters.
Integer categories have magnitude at most `2**53`; use strings for larger
integer identifiers. Continuous input is interpreted in real float64.

The global variance addition regularizes the Gaussian term: this CF score is
not a normalized Gaussian mixture log likelihood. Conditional independence of
the variables within clusters is an assumption, rather than a verified data
property. BIC and AIC use `m_k = k*(2*p_cont + sum_cat(L-1))`,
`BIC = -2*sum(xi)+m_k*log(N)` and `AIC = -2*sum(xi)+2*m_k` on retained
non-noise observations. Training global variance/category universe remain fixed
when small leaves are excluded. Legacy `selection='global_min'` takes the global minimum across
`1..min(max_clusters, retained_leaves)`, with exact ties choosing the smaller
count. A fixed `n_clusters` can choose any feasible hierarchy cut, independently
of the automatic-search range.

The tree routes each row through its nearest internal summary, absorbs into
the nearest leaf when the absolute merge loss is at most `threshold`, and
splits overflowing nodes around their farthest pair. Equal distances use
deterministic physical-row/entry ordering. `threshold` is a CF loss, not a
geometric radius. `min_precluster_size` excludes undersized leaves before
agglomeration and records their original physical positions as noise (label0).
Missing rows with `missing='drop'` have missing labels and distinct status;
`missing='raise'` refuses the complete call. Assignments preserve every physical
input row, including duplicate DataFrame indices. New observations choose the
nearest retained saved CF; they are not automatically classified as noise.

## Two-stage selection and adaptive noise

`selection='two_stage'` uses the Statistics14 equations: initial criterion
change `dIC(J)=IC(J)-IC(J+1)`, relative change `dIC(J)/dIC(1)<0.04`, then
`R2(k)=d_min(C_k)/d_min(C_{k+1})`. A negative initial change chooses one
cluster. If the largest distance ratio exceeds 1.15 times the second-largest,
choose its count; otherwise choose the larger count among the top two.
The additional J+1 criterion is retained even at the declared search cap.
Fixed counts still override automatic selection.

The primary guide does not specify all degenerate/terminal cases. Recorded
local extensions choose one for zero initial change, two for an improving
two-cut hierarchy, the feasible capped initial count if no 0.04 crossing,
and the sole available jump if there is only one. Positive-over-zero ratios
rank above finite ratios; zero-over-zero is neutral1. Ratios that overflow
are represented as null plus a reason, preserving finite JSON. Tied jumps
choose the larger count. These policies are not licensed executable parity.

`rebuild=True` admits aggregate-CF rebuilding when leaves or nodes fill.
`max_rebuilds` is a strict integer0..16; exhaustion refuses the whole fit.
Prior immutable summaries plus the pending CF preserve rows even if a failed
split mutated internal nodes. Each retry raises the threshold to the next
finite float above `max(2*current_threshold, closest_retained_CF_loss)`.
Canonical physical-row order is used for summary reinsertion. Thresholds,
CFs, sparse/deferred/reinserted/noise rows and cumulative work are retained.
Threshold/tie/retry rules are explicit bounded implementation policies.

`noise='adaptive'` requires rebuild, positive `noise_fraction` (default0.25)
and `min_precluster_size=1`. Before rebuilding, summaries smaller than the
declared fraction of the largest summary are set aside. After rebuilding,
their individual physical records attempt absorption using the current tree
threshold without adding leaves/nodes. A final sparse-leaf pass repeats this
check. Records that still do not fit are retained in a separate complete noise
CF and excluded from the hierarchy. No noise row is silently discarded from
the result or its sample accounting.

For saved new-row assignment, adaptive noise uses Statistics14's strict
nearest-cluster rule `merge_loss < C`, with
`C=sum(log(training_standardized_continuous_ranges))+sum(log(category_levels))`.
Training scales, ranges and category universe remain fixed; query data cannot
change the cutoff or CFs. At equality, or for a nonpositive cutoff with
nonnegative loss, a complete query is noise (label0). Training hierarchy labels
remain CF membership; saved queries do not retrospectively relocate training
records. The cutoff and tree absorption threshold have different roles.

`order='input'` uses the supplied row order. `order='random', seed=...` uses a
private CPU generator and records the entire physical insertion order.
`oe.twostep_stability([fit1, fit2, ...])` compares label-invariant adjusted Rand
indices on common non-noise physical rows from the same fingerprinted corpus,
reporting noise exclusions. Order dependence is expected; a high ARI on one
fixture is not a general stability guarantee.

Profiles report cluster sizes, original-scale continuous population moments
and categorical counts/proportions. Exact silhouette uses singleton CF merge
loss as dissimilarity, excluding training noise/missing rows. Singleton clusters
have silhouette0; one-cluster quality is undefined and refused. Profiles,
silhouette and ARI are descriptive: selected-cluster covariance, SE, df, p,
CI and posterior membership probabilities are not supplied.

The admitted domain is unweighted CPU float64 resident DataFrames, column
mappings or record lists: at most2000 physical rows,8 continuous and8 categorical
columns,32 levels per categorical column,128 leaf summaries and512 tree nodes.
Exact quality admits at most500 retained observations. Conservative work and
combined workspace budgets are checked before selected input coercion and
before pair-distance buffers; a smaller declared tree may be needed with many
categorical features. Reaching a limit raises an error without returning a
partial fit, truncating rows or collecting a Dataset. Legacy insertion refuses
tree fullness; explicitly admitted rebuilds can raise the threshold within
their cumulative work/retry limits. Adaptive history has its own workspace
admission before input conversion; large retry/feature combinations may need
smaller bounds or an explicitly larger budget.
Weights, MPS/CUDA, formulas and undeclared options are refused.

`twostep_save/load` exports/restores all tables, typed maps, training moments,
physical samples, tree metadata, merge trace, complete hierarchy and integrity
digest. Console settings previews are abbreviated; they are not the full saved
model. Saved assignment/cut/profile/quality/stability helpers validate state
and displayed core tables. Fit, assignment and cut do not mutate caller data
or the saved training state.

Version1 results remain supported with their original semantics. Version2
retains the selection/noise witness. Validation authenticates adaptive tree
transitions against retained training data and controls within a separately
admitted work budget; this replay checks saved state and does not replace its
hierarchy or estimates. Assignment/cut budgets include both validation and the
requested operation. [Complete save/cold-replay example](../../examples/twostep_adaptive_acceptance.py).

Selection replay authenticates the decision from the saved criterion scores
after independently checking those scores against the retained CFs. It keeps
the original decision trace when another platform rounds logarithms differently.

The score, CF-tree and hierarchical construction follow the
[IBM TwoStep algorithm](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/twostep_cluster.pdf),
with AIC also described in the
[IBM Modeler algorithms guide](https://www.ibm.com/docs/en/SS3RA7_18.5.0/pdf/AlgorithmsGuide.pdf).
The original mixed-distance approach is described by
[Chiu et al. (2001)](https://doi.org/10.1145/502512.502549).
Licensed vendor execution has not been performed and was removed from issue
acceptance at the project owner's explicit9October2026 request. Independent
primary-equation references, source, frozen runtime and visible native
persistence evidence remain distinct. No whole-product Stata/SPSS parity or
physical CUDA acceptance is claimed.
