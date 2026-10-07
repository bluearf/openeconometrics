# Signed networks

`oe.signed_network` creates a separate CPU float64 signed adjacency snapshot
for attraction/repulsion weights or negative path costs. The existing
`oe.network` API keeps its nonnegative contract and algorithms.

```python
import openecon as oe

graph = oe.signed_network(edges, source="from", target="to", weight="effect",
                          directed=True, nodes=all_nodes, max_memory_mb=256)
strength = graph.signed_strength()
centrality = graph.signed_katz(alpha=.01)
partition = graph.signed_communities(max_work=50_000_000)
quality = graph.signed_modularity(partition)
distances = graph.signed_shortest_paths("origin", max_work=50_000_000)
```

Integer `1` and string `"1"` are different nodes. Weights must be finite real
numbers; booleans and infinities are errors. Missing endpoints/weights require
explicit `missing="drop"`; otherwise import fails. `weight=None` gives every
edge +1. Duplicate pairs sum **algebraically** before analysis: +3 and -1
become +2, while +3 and -3 cancel. Input zeros and cancelled pairs have no
adjacency entry, but their endpoints and explicit isolates remain. Reversed
pairs are duplicates in undirected graphs. No method silently converts weights
to magnitudes, samples the graph or selects another engine.

A diagonal self-loop counts **once** in signed adjacency strength/modularity;
an undirected off-diagonal edge represents two oppositely directed arcs.
Existing nonnegative `degree()` continues to count an undirected loop twice.

## Methods

`signed_strength()` reports positive strength, negative **magnitude** strength,
net strength (their difference), and absolute strength (their sum). Directed
graphs have outgoing/incoming columns. Every result table includes all typed
nodes in encounter order, including isolates.

`signed_katz(alpha, beta=1)` solves the incoming signed resolvent
`x = beta*1 + alpha*A.T*x`. Scores can be negative and have no probability or
normalization interpretation. The sufficient contraction condition is
`alpha * max_absolute_incoming_strength < 1`, with `alpha >= 0` and `beta > 0`.
This bound is deliberately conservative. The returned iterate has its own
fixed-point residual checked against `tol * (1 + max(abs(x)))`. Metadata
includes iterations, residual, contraction bound, residual-based infinity-norm
error bound and work. Nonfinite arithmetic or diagnostic bounds are errors. Exhausting `max_iter` raises an error without partial scores.

`signed_modularity(membership, resolution=1)` follows
[Gómez, Jensen and Arenas (2009), equations 18–20](https://arxiv.org/abs/0812.3030).
Let P/N be positive/negative-magnitude total **arc** weights, I+/I- their
within-community weights, and Pout/Pin/Nout/Nin each community's channel strengths:

```
Q = (I+ - I- - resolution * sum_c(Pout_c * Pin_c / P)
              + resolution * sum_c(Nout_c * Nin_c / N)) / (P + N)
```

Zero-mass channels contribute zero; empty adjacency has Q=0. All weights are
normalized by their largest magnitude to prevent mass overflow without changing
Q; erasing a nonzero edge causes a precision error. Membership must cover every
typed node exactly once, as a mapping, node/community table or graph-order vector.

`signed_communities()` uses deterministic single-node moves over **all** existing
communities plus one empty singleton. This matters for negative/null-model terms.
Nodes are visited in integer-then-string sorted order; only gains exceeding `tol`
are accepted. A final stable sweep has no accepted move. Metadata includes
objective trace, sweeps, work and stationarity. This is a greedy local solution;
it makes no global-optimum, multilevel Louvain or Leiden claim. `max_sweeps`
exhaustion raises an error without a partial partition.

`signed_shortest_paths(source)` uses synchronous sparse
[Bellman–Ford relaxation](https://algs4.cs.princeton.edu/44sp/index.php) for
minimum **walk** costs. Unreachable nodes have null distances and false
reachability. A source-reachable negative cycle raises `network_negative_cycle`;
an unreachable cycle has no effect. An undirected negative edge forms a
two-arc negative walk cycle and is refused when reachable. This API does not
solve the different simple-path problem. The final cycle-check pass consumes
work. Overflow or an addition that erases a nonzero path cost raises a precision
error instead of certifying a potentially wrong cycle result.

`pagerank`, unsigned centrality/community methods, capacity flow/cuts and
Bernoulli/Poisson/degree-corrected count-block models explicitly raise
`network_signed_method` on `SignedNetwork`. Other unsigned Network methods are
also unavailable; use the five named signed methods above.

## Resources and verification

DataFrames, column mappings, record iterators and `oe.Dataset` are read in
bounded batches. The complete sparse graph stays **resident**, without disk-native
algorithms. `max_memory_mb` guards estimated owned graph/import/algorithm buffers,
excluding caller input and total process RSS. `max_work` counts arc/node scans and
community candidates; it is a structural budget, not a wall-clock deadline. No
partial result is returned after resource or convergence refusal.

Storage/workspaces are O(V+E). Katz costs O(iterations*E); Bellman–Ford costs
O(VE) in the worst case. The all-community greedy search may cost O(sweeps*V²),
with sparse storage and an explicit work limit. Execution stays CPU float64
regardless of Torch's ambient default device. No SciPy, NetworkX or statsmodels
runtime solver is used.

Independent tests use exact Fraction modularity, rational Gaussian elimination
for Katz, min-plus all-pairs paths, and every single-node partition move on small
directed/undirected fixtures. They cover signs, cancellation, zero edges, loops,
typed labels, isolates, missing data, cycle reachability, precision and budgets.
The existing nonnegative network suite is run alongside them.

Physical Parquet scale receipts are in
[evidence/market-56-signed-networks](evidence/market-56-signed-networks/):

| Fixture | Full nodes/edges | Independent check |
| --- | ---: | --- |
| Strength/Katz | 100,000 / 1,000,000 | All strengths match analytic values; Katz equals 1.2 within recorded tolerance |
| Communities/modularity | 1,000 / 125,500 | Four planted groups recovered; objective recomputed on complete partition |
| Negative-cost directed DAG | 10,001 / 37,000 | Every distance equals its exact potential difference |

Each case uses a fresh Python process. Receipts contain input/source hashes,
counts, precision, threads, work, elapsed phases and lifetime process peak RSS.
These are local synthetic scale checks, without a universal performance claim.
Reproduce from the repository (choose a fresh output filename):

```sh
PYTHONPATH=src:packages/openecon-charts/src python benchmarks/network_signed.py \
  --case strength_katz --output /tmp/signed-strength-katz.json
# Repeat with --case communities or --case negative_dag_paths.
```

[The code-panel example](examples/network_signed.py) runs all five methods,
publishes four complete tables, and checks unsigned-method and cycle refusal.
The packager includes the lazy module; editor help includes exact signatures.
The frozen-runtime verifier runs the same example without injecting the checkout
into the worker and checks persistence after reset. Installed-app and release
proof are recorded separately from source and frozen-runtime checks.


The installed Mac receipt records the exact code-panel run, four five-row tables,
complete typed identities and an installed-bundle precision-guard check. Code,
outputs and events were unchanged after a complete application restart; the
four tables and saved script remained visible. The existing 20 human project
files were byte-identical. This local maintenance update retains the previously
working renderer and native code payload, with the previous complete app backup
kept and the bundle signature verified. It updates the local Python runtime;
it does not claim public installer delivery or release publication.
