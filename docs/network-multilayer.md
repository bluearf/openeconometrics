# Coupled multilayer networks

`oe.multilayer_network` is a resident, single-aspect coupled network. Every
state has an exact `(node, layer)` identity: integer `1` differs from string
`"1"` for nodes, layers and edges. Nodes may be absent from some layers;
the implementation never pads a Cartesian product. Explicit empty layers
and node-layer isolates survive interchange and edge edits.

```python
import openecon as oe

g = oe.multilayer_network([
    dict(edge_id="trade", source="A", source_layer="trade",
         target="B", target_layer="trade", weight=2.),
    dict(edge_id="coupling", source="A", source_layer="trade",
         target="A", target_layer="finance", weight=.5),
], weight="weight", directed=True,
   nodes=[("B", "finance")], layers=["trade", "finance", "empty"])

g.nodes()  # complete ordered supra_index -> (node, layer) map
g.edges()  # separate IDs, raw orientation, zero weights and scalar attributes
g.matvec({("B", "finance"): 0, ("A", "trade"): 1,
          ("B", "trade"): 2, ("A", "finance"): 3})
g.pagerank()  # rank per node-layer state, including explicit coupling
g.layer("trade")  # intra-layer MultiNetwork; excludes all inter-layer edges
g.project(reducer="sum", inter_layer="drop", zero="drop", attributes="drop")
```

The sparse supra-adjacency is `A[source state, target state]`. Parallel edge
weights sum for adjacency operations, although every original edge stays
separate in the data model. Undirected off-diagonal edges contribute both
orientations; undirected adjacency self-loops contribute once. `matvec`
computes `A @ x`, or `A.T @ x` with `transpose=True`. Its mapping must contain
every state; values may be finite negative reals. Overflow is an explicit
precision error.

`pagerank` is ordinary weighted PageRank on this supra-graph. Outgoing weights
set walk probabilities for both intra- and inter-layer edges. Teleportation is
uniform over states by default; personalization uses exact pair keys and
dangling mass follows it. Convergence uses the contraction error estimate and
failure raises an error. Zero-weight records remain in the source but contribute
no transition. This initial method is not a layer-balanced multiplex walk,
multislice community objective or automatically aggregated physical-node rank.
`supra_network()` explicitly returns the sum-weight analytical Network with
integer state IDs; `nodes()` provides the full identity map. It drops zero
pairs and edge attributes for analysis and records this choice.

`layer(layer_id)` retains intra-layer edge IDs, attributes and isolates.
`project` collapses replicas only by exact physical node ID. Its reducer is
required (`sum/min/max/mean/count/binary`); `inter_layer='drop'` excludes
inter-layer edges and `'include'` includes them, so diagonal coupling becomes
a physical-node loop. Conflicting replica/parallel attributes and zero pairs
raise unless explicitly dropped. Output metadata records the loss of layer
and edge identity. Neither operation modifies the source or treats layers as
ordered temporal snapshots.

`edit_edges(add=..., remove=..., weights=..., attributes=...)` returns a new
snapshot. Added edges include both endpoint layers; existing layers are an
allowlist and existing node-layer states stay, including isolates. Weight and
attribute updates use exact retained edge IDs. Node/layer renaming and
multi-aspect layers are not part of this initial API.

`g.write(path)` / `oe.read_multilayer_network(path)` use the lossless
`openecon.coupled-multilayer/1` NDJSON schema: one header, ordered layer
declarations, ordered node records, then ordered edge records. JSON values
preserve string/int64 identity, all scalar attributes, raw edge orientation,
parallel IDs, loops and zeros. Duplicate fields/declarations, undeclared
endpoints, unknown schema/fields, nonfinite values and missing trailing
newlines are rejected. Records are bounded at 16 MiB and reading charges
parse and resident-build workspace under `max_memory_mb`; `max_file_mb`
limits input bytes. Writes use an owned temporary file and atomic publication;
existing paths require `overwrite=True`. GraphML/GEXF export is unavailable
for this model; use an explicit projection first when loss is acceptable.

The numerical core is CPU Torch float64/int64, with sparse edge reductions and
no dense N² adjacency. This model does not claim disk-backed or GPU execution.
`max_memory_mb` covers conservative planned owned storage and transient import,
projection, edit and numerical buffers. It excludes caller input and total
process RSS. Iterative PageRank and projections enforce `max_work` before
numerical setup; no sampling, data truncation or alternative method is used on
budget failure. Returned derived snapshots may have a smaller remaining budget
because source/temporary storage was charged during their construction.

The small [Mac example](examples/network_multilayer.py) has nonuniform directed
weights, two distinct typed `1` identities, coupling, parallel edges, a loop,
a zero record and an isolate. Fraction Gaussian elimination independently
checks PageRank; scalar row equations check supra-matvec. The unit suite also
checks 48 seeded nonuniform small directed/undirected networks, projections,
lossless edits/interchange, resource refusal and sparse allocation.

The fresh-process measurement (internal evidence excluded from this public snapshot)
uses two undirected circulants plus diagonal coupling on macOS ARM64,
Python 3.13.5, Torch 2.14.0 and one CPU thread:

| Physical nodes | States | Separate edges | Layers | Import | Matvec | PageRank | Planned peak owned | Process peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 10,000 | 35,000 | 2 | 1.24 s | 0.048 s | 0.377 s | 145.2 MiB | 266.2 MiB |
| 20,000 | 40,000 | 140,000 | 2 | 3.38 s | 0.197 s | 1.35 s | 576.2 MiB | 341.8 MiB |

All result rows are returned. Regularity gives uniform PageRank in one
iteration; this workload does not measure convergence of arbitrary topology.
Independent expected adjacency row sum is 6.5. The 768 MiB planned budget is
separate from whole-process RSS, which includes Python/Torch and outputs.

The model and supra-graph interpretation follow the node-layer/supra-adjacency
definitions in [Kivelä et al., Multilayer Networks, sections 2.1 and 2.3](https://arxiv.org/abs/1309.7233).
The particular PageRank walk and projection policies above are explicit API
choices; they do not assert equivalence to every multilayer descriptor.

Installed native acceptance (internal evidence excluded from this public snapshot)
passed eight complete tables and a full application restart, with output/event
hashes unchanged and no recalculation. The existing 0.3.42 native shell was
preserved and its runtime was updated; six selected compiled SDK modules match
the recorded source revision. All 15 pre-existing source/data files retained
their hashes. Cloud team sharing showed the separately recorded MARKET-87
publication-protocol rejection; it is excluded from this local network
acceptance. Isolated frozen acceptance (internal evidence excluded from this public snapshot)
also passed and removed its owned process/data.
