# Sparse hyperedges and bounded structural paths

`oe.hypergraph(records, nodes=..., directed=False)` stores typed integer/string
hyperedge IDs and sparse binary incidence, not pairwise adjacency. Undirected
records have `id`, `members`, optional `weight`; directed records use nonempty
`tail` and `head`. Tail/head overlap and singleton edges are valid. Duplicate
membership within a role and duplicate hyperedge IDs are errors; distinct IDs
may describe parallel hyperedges. Missing/boolean/float IDs and negative or
nonfinite weights are rejected. Zero weights retain membership and degree but
contribute zero strength. No sample or dense V×H/V² matrix is substituted.

`incidence(role='members', weighted=False)` returns a detached CPU float64 sparse
copy. Directed callers must select `tail` or `head`. Weighted incidence multiplies
columns by edge weights. `incidence_matvec` takes an aligned finite Torch vector;
`transpose=True` multiplies the transpose. `degree`, `strength`, `memberships`,
and `summary` are direct hypergraph analyses, with complete typed output.

`clique_projection(reducer='sum'|'count'|'normalized')` explicitly creates a
Network. Directed tails connect to heads; self-pairs are excluded. Normalized
contribution is weight divided by the number of nonloop candidate pairs in that
hyperedge. `max_edges` admits the conservative candidate count before expansion,
including duplicate pairs and directed overlap self-pairs. `star_projection`
uses disjoint integer proxies; node attributes `kind` and `original_id` preserve
original typed identities. Directed spokes run tail→hyperedge→head. Each spoke
has the hyperedge weight: a graph path sums spoke costs. Network's zero-edge
policy applies to both projections. Projection algorithms are not direct
hypergraph community/spectral/path algorithms; those remain unsupported.

Membership, candidate-edge, structural-work and estimated owned-memory budgets
raise structured errors. Projected graph import is admitted against remaining
memory while the hypergraph and projection staging remain live. The estimate
does not bound process RSS, caller data or Torch allocator overhead. Hypergraph
work counts record/membership visits and admitted projection candidates; it is
not an elapsed-time limit or a count of Torch's internal sorting operations. Inert JSON
`write` creates a new file; `oe.read_hypergraph` checks file size before parsing
and validates all typed memberships again.

`Network.k_shortest_paths(source, target, k=5, ...)` returns rank, typed path
tuple, cost, hops and exact cost numerator/denominator. It finds distinct
loopless paths in a simple Network, using best-first prefix search. Costs are
exact rational sums of stored positive binary64 aggregate weights (unweighted
graphs use unique-edge hops), ordered by cost then typed-ID lexicographic order,
integers before strings. This is not Yen's algorithm: prefix enumeration has
exponential worst-case work and frontier size. `max_frontier`, `max_work`,
`max_output_nodes` and the input memory budget bound it. `max_path_length`
explicitly restricts the requested domain in hops; `None` allows V−1 hops.
Unreachable pairs return empty output, and equal endpoints return one trivial
path. A budget error returns no partial result. Cost overflow raises an error.

`strong_bridges()` and `strong_articulation_points()` require a directed Network.
They compare total SCC count before/after deleting each oriented aggregate arc
or vertex, using iterative sparse Kosaraju, O((V+E)²) worst-case work and O(V+E)
workspace. An increase identifies a strong cut. Loops cannot be strong bridges;
weak bridges/articulation APIs retain their prior conventions. Signed,
multigraph and disk-backed inputs have no implicit conversion to these APIs.

Run `docs/examples/network_hypergraph_and_paths.py` in the Mac code workspace.
Independent tiny incidence/path/reachability oracles and scale/native receipts
are recorded under `docs/evidence/market-60-67/`. These are bounded CPU method
checks; no CUDA, universal Stata parity or public release claim is made.
