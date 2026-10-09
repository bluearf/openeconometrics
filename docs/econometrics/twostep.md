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
when small leaves are excluded. Selection takes the global minimum across
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
partial fit, raising the threshold, truncating rows or collecting a Dataset.
Weights, MPS/CUDA, formulas and undeclared options are refused.

`twostep_save/load` exports/restores all tables, typed maps, training moments,
physical samples, tree metadata, merge trace, complete hierarchy and integrity
digest. Console settings previews are abbreviated; they are not the full saved
model. Saved assignment/cut/profile/quality/stability helpers validate state
and displayed core tables. Fit, assignment and cut do not mutate caller data
or the saved training state.

The score, CF-tree and hierarchical construction follow the
[IBM TwoStep algorithm](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/twostep_cluster.pdf),
with AIC also described in the
[IBM Modeler algorithms guide](https://www.ibm.com/docs/en/SS3RA7_18.5.0/pdf/AlgorithmsGuide.pdf).
The original mixed-distance approach is described by
[Chiu et al. (2001)](https://doi.org/10.1145/502512.502549).
This bounded implementation differs from IBM's automatic two-stage
change/jump selection, adaptive threshold rebuild and adaptive noise
reinsertion. Licensed vendor execution has not been performed; no whole-product
Stata/SPSS parity or physical CUDA acceptance is claimed. MARKET-172 remains
open for those deferred comparisons and algorithms.
