# Additional native network methods

These CPU methods use Torch and OpenEconometrics' own algorithms. NetworkX,
SciPy and statsmodels are not execution dependencies. Each call admits explicit
owned-memory, structural-work and output limits; failure returns no partial
estimate. Planned workspace excludes caller input, Python/Torch startup,
allocator caches and whole-process RSS. Work counts structural operations,
not arbitrary-precision bit operations or wall-clock time.

[Run the synthetic example](examples/network_five_methods.py) in the Mac code
panel. It asserts hand-checkable results and displays eleven complete tables.
The independent tests and separately measured scale fixtures are described in
the acceptance evidence (internal evidence excluded from this public snapshot).

## Typed hypergraphs · MARKET-60

The concurrently merged implementation is retained without an API substitution.
See [sparse hyperedges and bounded paths](network-hypergraph-paths.md) for typed
record inputs, directed tail/head incidence, nonnegative/zero edge weights,
explicit projection reducers, lossless JSON persistence and admitted budgets.
Its independent and installed-native acceptance is pinned under
MARKET-60/67 evidence (internal evidence excluded from this public snapshot).

## DSP and joint MRQAP · MARKET-64

```python
r = response.qap_regression({'a': a, 'b': b, 'nuisance': z},
    method='dsp', joint={'a_and_b': ['a', 'b']}, adjustment='holm',
    permutations=999, seed=42, alternative='two-sided', max_work=50_000_000)
joint_table = oe.DataFrame(r.attrs['joint_tests'])
```

`method='freedman_lane'` remains the default, with unchanged coefficient tests.
DSP follows [Dekker, Krackhardt and Snijders, section 3.2.2](https://www.stats.ox.ac.uk/~snijders/DekkerKrackhardtSnijders.pdf):
residualize a focal X against nuisance Z, permute that residual by the same node
permutation at both dyad endpoints, then project against fixed Z again. Test its
partial correlation with the response residual. The default FL procedure instead
permutes the reduced response residual. Dyads include absent edges as zero;
direction, loops, binary/weight semantics and intercept remain explicit.

Joint hypotheses test all named coefficients zero conditional on the remaining
predictors. Their statistic is partial R-squared, monotone in F at fixed degrees
of freedom; their alternative is always upper-tail. DSP jointly permutes tested
X residuals; FL permutes reduced Y. All tests share each private CPU Torch draw.
Rank-deficient or undefined reduced/permuted designs fail without discarding
draws. Typed labels and predictor names use canonical ordering.

Raw plus-one p-values have resolution `1/(permutations+1)`. `none`, `bonferroni`,
`holm` and `bh` adjustments cover all reported slopes and joint hypotheses,
excluding the intercept. The selected family, actual permutations, resolution,
seed, null and exchangeability assumptions are recorded. BH needs independence
or PRDS; this option does not prove those assumptions for arbitrary dependent
dyads. Conditional node-exchangeability and an appropriate mean model are
required; these are approximate conditional tests, not causal guarantees.

## Induced four/five-node graphlets · MARKET-65

```python
g = graph.graphlets(size=4, connected=True, max_subgraphs=100_000,
                   max_output_rows=100_000, max_work=50_000_000)
g['classes']   # canonical class ID and count
g['orbits']    # sparse node/class/automorphism-orbit participation
```

Exact induced topology enumeration supports four and five vertices, directed
or undirected. `connected=True` uses weak connectivity for directed graphs;
False includes disconnected and empty classes. Stored edge strength is explicitly
reduced to presence. Loops and multigraphs are rejected. There is no sampled
estimate or non-induced counting branch.

Each class ID encodes the minimum off-diagonal adjacency bitmask over all vertex
permutations. Orbit IDs encode equivalence sets under adjacency-preserving
automorphisms; they are not ORCA/another package's orbit numbers. Missing sparse
orbit rows mean zero. Preflight caps all `choose(n,size)` subsets; canonicalization,
cache and complete class-plus-orbit output obey work/memory/row budgets. This
exhaustive backend is deliberately bounded; it is not a large-network sampling
or polynomial-time claim. See the [primary graphlet/orbit definition](https://www.nature.com/articles/srep37057).

## Simple paths and strong directed cuts · MARKET-67

The concurrently merged `k_shortest_paths` and strong-cut APIs are retained.
See [the existing contract](network-hypergraph-paths.md) for exact binary64
cost ordering, hop/frontier/output-node/work limits and SCC deletion semantics.
The combined example calls those APIs alongside the new methods below.

## Weighted assignment and general matching · MARKET-70

```python
bip.weighted_assignment(partition=node_to_0_or_1, objective='weight',
                        max_matrix_entries=100_000, max_work=50_000_000)
odd_graph.general_matching(objective='cardinality_weight',
    max_component_nodes=24, max_states=1_000_000, max_work=50_000_000)
```

Both return a NetworkMatchingResult with typed pair/edge-ID and unmatched tables,
objective value, metadata and certificate. `weight` maximizes algebraic reward
and may leave nonpositive edges unmatched. `cardinality` maximizes pair count;
`cardinality_weight` maximizes count first, then reward. Finite signed rewards
are accepted by SignedNetwork and MultiNetwork; simple Network retains its
existing zero-edge omission. Multigraph alternatives are never summed, and the
actual selected edge ID survives. Self-loops and directed graphs are rejected.
Ties are deterministic by typed endpoint/edge-ID order; general matching's
snapshot index ordering is recorded. No implicit bipartite projection occurs.

Assignment checks bipartiteness or a complete explicit 0/1 partition, then uses
native exact-rational rectangular Hungarian minimization with unmatched dummy
columns and forbidden missing edges. The dense augmented `left*(right+left)`
matrix is separately capped and charged. Returned row/column potentials,
primal/dual objectives and zero gap are exact rational strings and verified
before a certified result is returned.

General matching includes odd cycles through componentwise exact subset dynamic
programming. Its unmatched-or-pair recurrence exhausts all matchings and records
each component optimum/state count. It is exponential, default 24 vertices per
component (explicit maximum 32), with memo-state/work/memory limits. This is not
a polynomial blossom solver or a dual certificate. Both algorithms compare
binary64 input rewards as exact rationals; finite float64 output is required.
The existing unweighted bipartite `maximum_matching` behavior is unchanged.
