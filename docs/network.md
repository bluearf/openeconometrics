# Sparse network analysis

The [explicit CPU/CUDA device matrix and validation status](network-devices.md)
cover degree/strength, PageRank, eigenvector, HITS, Katz and modularity scoring.

`oe.network` creates a graph snapshot from an edge dataframe, column mapping,
records iterator or projected `Dataset`. Import is batched. File-backed or other
`Dataset` input builds bounded disk indexes first, then returns a resident graph
when its conservative import bound fits the budget, or retains a `DiskNetwork`.
Passing `store="graph.sqlite"` requests a persistent disk snapshot. Degree,
strength, CPU PageRank and weak components scan disk edges with admitted O(V)
state. Storage and these selected methods do not certify out-of-core estimation
or all-algorithm scaling.
No dense V-by-V matrix, SciPy or NetworkX is used by the implementation.
Torch declares NetworkX as a transitive installation dependency, and its support
package is present in the Mac runtime. The native network algorithms below do
not call it; this is separate from claiming that dependency is absent.

The [support and independent validation matrix](network-reference-validation.md)
identifies method assumptions, small numerical oracles, pinned public-data
comparisons, installed Mac replay and unverified hardware/scale boundaries.

```python
import openecon as oe

graph = oe.network(data=oe.scan("edges.parquet"), source="from", target="to",
                   weight="amount", directed=True, nodes=["isolated"],
                   missing="drop", batch_rows=65536, max_memory_mb=256)
display(graph.summary())
display(graph.degree())
display(graph.pagerank())
display(graph.components())
display(graph.shortest_paths("origin"))
oe.plot.network(graph, title="Connections")
```

Import validates exact string/integer identities. Booleans and floats are not
node labels; integer `1` and string `"1"` are distinct nodes. Strings must be
nonempty, contain no C0/C1 control characters and fit 4,096 UTF-8 bytes. Integers
may have at most 256 bits. For integer columns with missing rows, use nullable
integer dtype rather than implicitly converting the entire column to floats.
Optional `nodes` retains isolates and collapses repeated node labels.

Weights must be finite, nonnegative real numbers. Missing endpoints or weights
raise by default; `missing='drop'` counts omitted physical rows. Zero-weight rows
are counted and dropped as edges, but their valid endpoints remain nodes.
Repeated positive edges add their weights; reverse pairs merge for undirected
graphs. COO coalescing and binary-level chunk merges perform this aggregation in
Torch float64, with ordinary floating-point rounding rather than exact decimal
arithmetic. Import metadata records physical/positive/missing/zero/duplicate
counts, consumed-input fingerprint and actual bounded-reader peak rows.

`degree()` counts unique aggregate connections, while strength sums aggregate
weights. A directed self-loop contributes once to each in/out degree and strength;
an undirected self-loop contributes twice to degree and strength. In PageRank it
is one outgoing transition. With no weight column, duplicate records still add
unit weights for strength/PageRank; unweighted shortest paths count unique links.

`pagerank(damping=.85, tol=1e-10, max_iter=200, personalization=None, device='cpu')`
uses weighted sparse power iteration. Personalization maps existing node labels
to nonnegative weights; omitted nodes get zero mass and total mass must be
positive. Dangling nodes redistribute through that same normalized distribution.
Per-source scaling avoids overflow in transition-weight normalization. Results
sum to one and include convergence iterations, L1 step and a contraction-based
stationary-error estimate. Convergence requires `damping/(1-damping) * L1_step <=
tol`; exhausted iterations raise `network_nonconvergence`. This bound is an
iteration estimate, not a rigorous floating-point certificate. Damping must be
in [0,1). Empty graphs return an empty converged table; isolate-only graphs return
the personalization distribution. CPU float64 is tested; CUDA float64 is optional
when available and memory is checked. Metal is refused because it lacks float64.

The Markov formulation follows Page et al., *The PageRank Citation Ranking:
Bringing Order to the Web* ([original Stanford report](http://ilpubs.stanford.edu:8090/422/));
the teleportation/iteration formulation is also explained in Stanford's
[PageRank computation](https://nlp.stanford.edu/IR-book/html/htmledition/the-pagerank-computation-1.html).
Duplicate addition follows [PyTorch's sparse COO contract](https://docs.pytorch.org/docs/stable/sparse.html#sparse-coo-tensors).
Implementation uses explicit checked sparse invariants, without disabling warnings.

`components()` returns weak components, ignoring direction, including isolates.
IDs are deterministic by each component's smallest typed label.
`components(connectivity='strong')` uses iterative Kosaraju, including isolates.
`shortest_paths(source)` runs
BFS without a weight column and Dijkstra with one. Directed paths obey directions;
undirected paths allow both directions. Weighted distance sums the **aggregate**
edge weights; if the weights represent interaction strength rather than cost,
transform them into a meaningful distance before building that graph. Disconnected
nodes have infinity. The method does not return all-pairs distances. Unrepresentable
aggregate weights, strengths or reachable path lengths fail explicitly.
Use `shortest_path(source, target)` for a route or the explicitly bounded
`distances(sources=..., targets=...)` query described below for pair output.

`summary()` and analytical outputs are ordinary OpenEconometrics DataFrames with
the existing LaTeX exporter. They are separate from graph rendering.
`to_plot_data(max_nodes=1000,max_edges=5000,seed=0)` chooses highest-degree nodes
and only their induced edges. The explicit display caps can be raised to 100,000
nodes and 1,000,000 edges, subject to the graph's memory allowance and the chart
payload budget. The defaults remain 1,000 nodes and 5,000 edges.
Ties use typed original labels; edge clipping follows that selected-node order.
Selection is deterministic, so a valid nonnegative integer seed does not alter
it. Payload IDs remain unique integers even when stringified labels look alike.
It records full and shown counts, `sampled`, and `selection='highest-degree induced
subgraph'`. Graph analyses always retain the full graph. The chart layer may impose
additional label/rendering budgets; a plot cap is not a statistical sample.

## Code-first network workbench

### Separate edge identities and parallel connections

`oe.multigraph` returns a resident CPU `MultiNetwork`. Each input row must carry
a unique string/integer edge ID under the same typed identity rules as nodes:
integer `1` and string `"1"` are separate edges. Duplicate IDs fail. Parallel
edges, their input order, supplied endpoint orientation, individual weights and
scalar attributes remain separate, including zero-weight edges and self-loops.
The existing `oe.network` aggregate-dyad behavior above remains unchanged.

```python
multi = oe.multigraph([
    {"edge_id": 1, "source": "a", "target": "b", "amount": 2},
    {"edge_id": "1", "source": "a", "target": "b", "amount": 3},
], weight="amount", nodes=["isolated"],
   edge_attributes={1: {"relation": "trade"}, "1": {"relation": "loan"}})
display(multi.edges())
display(multi.degree())
changed = multi.edit_edges(weights={1: 9}).filter(edge_ids=[1])
simple = multi.to_network(reducer="sum", attributes="drop")
display(simple.pagerank())
multi.write("connections.graphml")
restored = oe.read_multigraph("connections.graphml")
```

Column names `edge_id`, `source`, `target` and optional `weight`/`attributes` are
configurable for dataframe, mapping, record-iterator or projected `Dataset`
input. `attributes` names a column of bounded scalar dictionaries. Node and edge
attribute dictionaries use exact node/edge IDs; graph attributes are scalars.
Missing identities/endpoints/weights fail unless `missing="drop"` is explicit.
Invalid IDs, negative/nonfinite weights and duplicate IDs always fail.

| Multigraph method | Semantics |
| --- | --- |
| `edges(attributes=True)` / `nodes(attributes=True)` | Ordered exact identities and scalar `attr.NAME` columns; full materialized tables. |
| `degree(max_work=50_000_000)` | Each separate edge counts, including zero weights; strengths sum their weights. An undirected loop counts twice; a directed loop once in each direction. |
| `summary()` | Counts separate edges, loops and retained zero-weight edges. |
| `filter(nodes=None, edge_ids=None)` | Exact-ID selections combined with AND; retained edge order and selected isolates survive. Unknown IDs fail. |
| `edit_edges(add=(), remove=(), weights=None, attributes=None)` | Add records with independent IDs; remove/replace weights by edge ID. Attribute updates replace each selected edge's scalar dictionary. |
| `edit_nodes(add=(), remove=(), rename=None)` | Add isolates, remove nodes with their incident edges, or rename simultaneously without identity collisions; retained edge IDs/attributes survive. |
| `with_attributes(nodes=None, edges=None, graph_attributes=None)` | Replace selected scoped dictionaries in a new snapshot; omitted scopes survive. |
| `to_network(reducer=..., zero="raise", attributes="raise", max_work=50_000_000)` | Explicit aggregate-dyad projection with a mandatory reducer and recorded identity/loss policies. |

Projection reducers are `sum`, `min`, `max`, `mean`, `count` and `binary`.
Directed pairs retain direction; reverse undirected pairs share one dyad.
`count` counts every separate edge, including zero weights; `binary` gives each
observed dyad unit strength. Weight reducers that produce zero dyads fail unless
`zero="drop"` is explicit. Different parallel attribute dictionaries, including
equal numeric values with different scalar types, fail unless
`attributes="drop"` is explicit. Identical attributes survive. Every projection
records the reducer, source edge count, dropped zero pairs and edge-ID loss.
No analysis or plot method silently projects: multigraph-native analysis is
currently degree/strength and summary; other algorithms require this explicit
`Network` projection first. All edit/filter operations leave the source intact.

Static GraphML and GEXF read/write retain edge IDs, zero weights, loops,
orientation, ordering and supported node/edge scalar attributes. Native files
carry typed IDs in `openecon.edge_identity` metadata; generated XML wire IDs are
separate. This reader's identity-preserving mode requires a unique XML ID on
every edge, even where an external format allows missing IDs. An exported native
multigraph must be reopened with `oe.read_multigraph`; `oe.read_network` refuses
its identity metadata instead of silently coalescing. GraphML supports scalar
graph attributes; GEXF refuses them. Pajek multigraph read/write is explicitly
unsupported, and an attempted write preserves any existing destination. Choose
an explicit simple projection before using the ordinary Pajek exporter.
Dynamic, mixed, nested and hypergraph XML remain unsupported.

`max_memory_mb` admits owned resident state and planned buffers; it does not cap
caller input or whole-process RSS. Import rows, parser/retained-state ownership,
table output and transformation workspaces are bounded before large allocations;
degree/projection also admit `max_work`. This is resident storage, without an
out-of-core or GPU multigraph claim. Exact-ID operations can fail when their
source, new snapshot and admitted buffers do not fit the declared budget.

Paste the [runnable multigraph example](examples/network_multigraph.py) into the
code panel for five ordered tables and isolated temporary interchange files.
MARKET-57 scale evidence (internal evidence excluded from this public snapshot) checks
25,000 and 100,000 separate edges with analytic counts and exact ordered typed
record fingerprints through both XML formats. On the 100,000-edge local ARM64
CPU run, peak process RSS was approximately 337 MiB, separate from its explicit
512 MiB owned-workspace allowance. This is a three-node parallel-edge workload,
not an arbitrary-topology throughput guarantee.
The frozen worker receipt (internal evidence excluded from this public snapshot)
checks bundle-only execution, publication LaTeX and saved output/event readback
after worker reset. The final source passed 3,837 selected network, packaging and
editor-catalog cases (including 53 multigraph cases), plus 578 interface cases.
The isolated ARM64 frozen example completed in 8.09 seconds and produced five
ordered tables; packaged module code matched its recorded source. Installed app
delivery, browser pixels and public release delivery remain separate checks.

Graph edits, table operations, layouts, mappings, filters, labels, annotations
and timeline settings can all be defined in Python. There is no requirement to
configure them through a separate visual editor. The chart keeps a small set of
interaction controls for pan/zoom, node search/inspection, position dragging,
pinning, snapshot playback and export. These interactions change the view;
editing the analytical graph uses the immutable Python methods below.

The [advanced production-network laboratory](examples/network_advanced_lab.md)
provides a complete code-panel example: 2,400 firms, eight monthly directed
networks, matched capacity shocks, PageRank/Leiden, sampled brokerage, temporal
turnover, Poisson/DC block prediction, sampled MRQAP, held-out link ranking,
certified maximum flow and a firm/skill projection. Its 12 tables and six charts
display directly in the panel; finite budgets, exposure periods and inference
limits are explicit in the source and methodology notes.

The [runnable synthetic workbench example](examples/network_workbench.py) demonstrates the
complete workflow, including all nine layout specifications and saved-view
readback. Paste the entire Python file into the desktop code panel and press
**Run**. It displays two tables, all nine layouts and a snapshot timeline in the
results panel without writing any files. It uses no downloaded data, cloud
account or GUI configuration.

To save the HTML, JSON, CSV and LaTeX artifacts from a terminal:

```bash
python docs/examples/network_workbench.py --output /tmp/openeconometrics-workbench
```

### Editing the analytical graph

| Method | Behavior |
| --- | --- |
| `nodes(attributes=True)` | Exact typed node IDs, full-graph degree/strength columns and `attr.NAME` columns. |
| `edges(attributes=True)` | Canonical aggregate dyads, weights and `attr.NAME` columns. |
| `edit_nodes(add=(), remove=(), rename=None)` | A new graph with isolates retained and simultaneous exact-ID renaming; merging identities is refused. |
| `edit_edges(add=(), remove=(), weights=None)` | Add positive weights cumulatively, remove dyads or replace their aggregate weights; this remains a simple aggregate graph. |
| `update_attributes(nodes=None, edges=None, graph_attributes=None)` | Merge validated scalar values into a new snapshot. |
| `rename_attributes(mapping, scope='nodes')` / `drop_attributes(names, scope='nodes')` | Simultaneous column operations; scope can be nodes, edges or graph. |
| `filter(nodes=None, edges=None)` | An induced node selection, then an edge selection, evaluated against the complete analytical graph. |
| `with_positions(positions, fixed=True)` | Coordinate dictionaries keyed by exact typed labels; the default pins the supplied coordinates. |
| `Network.from_plot_data(data, max_memory_mb=256)` | Rebuild the explicitly saved display subgraph from typed identities; omitted nodes/edges are not recovered. |

Every edit leaves the source snapshot unchanged. The memory plan includes the
source, result and planned edit buffers within the source's `max_memory_mb`.
Table methods materialize their full requested tables; they are not streaming
tables. Graph tensors remain sparse and CPU float64. These helpers do not change
the resident-graph requirement or provide disk-backed analytical computation.

Core filters accept a Python callable returning a boolean, or up to 100
`{'field': ..., 'op': ..., 'value': ...}` predicates, combined with AND. Node
fields include `node`, `degree`, `strength` and `attrs.NAME`; edge fields include
`source`, `target`, `weight` and `attrs.NAME`. `attr.NAME` is an alias. Operators
are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in` and `not_in`; membership lists are
capped at 10,000 scalars. Missing values compare as `None`, and ordered filters
exclude missing/string/boolean values. Callable filters execute the caller's
Python; declarative chart predicates never evaluate code strings.

### Appearance and layouts

```python
chart = oe.plot.network(
    graph, groups=communities, title="Research network",
    max_nodes=10_000, max_edges=100_000,
    layout="forceatlas2", seed=7,
    layout_options={"iterations": 150, "work_limit": 50_000_000,
                    "time_limit_ms": 10_000, "linlog": True},
    node_size={"field": "degree", "scale": "sqrt", "range": [3, 14]},
    node_color={"field": "attrs.sector", "scale": "categorical",
                "range": ["#14263d", "#4c78a8"]},
    edge_width={"field": "weight", "range": [.5, 3]},
    labels={"show": True, "min_zoom": .5, "max_count": 80},
    annotations=[{"text": "Example", "x": 0, "y": -150}],
)
display(chart)
```

Node size/color/labels and edge width/color accept scalar settings or attribute
mappings. Mapping scales are linear, square root, logarithmic and categorical;
explicit domains, ranges and missing-value presentation are supported. Use
`attrs.NAME` to disambiguate an attribute from a structural field. Colors use
opaque hex values. Node-anchored annotations use display `node_id`, obtained
from `graph.to_plot_data()` by matching its explicit typed `identity`; they do
not interpret a stringified label as an ID. Coordinate annotations use `x/y`.

| `layout` | Meaning and selected `layout_options` |
| --- | --- |
| `d3-force` (`force` alias) | D3 force layout; charge, link distance, collision and Barnes–Hut theta. |
| `forceatlas2` | Native ForceAtlas2-family Barnes–Hut layout; scaling, gravity, linear/LinLog attraction, weight influence and adaptive speed controls. |
| `circular` | Deterministic circle; radius and angle. |
| `grid` | Deterministic grid; columns and spacing. |
| `radial` | Weak-adjacency BFS rings; root display ID, angle and spacing. |
| `hierarchical` | SCC condensation ranks for directed edges; spacing and TB/BT/LR/RL direction. |
| `geographic` | Longitude/latitude projection; equirectangular or Mercator, with scale. |
| `community` | Deterministic packing of explicit display groups; radius and spacing, not a multilevel optimizer. |
| `fixed` | Supplied coordinates or saved positions; no layout job. |

`groups=` accepts a complete node mapping or supported membership table for one
graph. Geographic layout requires coordinates on every saved node; fixed layout
requires positions for every node. Layout calculation runs in a browser worker,
independently from graph analysis. A saved view reopens its saved positions
without rerunning a force layout. Layout iteration/work/time budgets are explicit;
the default worker allowance is 200 million work units and 30 seconds. A stopped
job retains finite positions and reports the stop reason; it is not a convergence
certificate or a claim of identical coordinates to another application.

Chart `filters=` add `scope='nodes'` or `scope='edges'` to the declarative
predicates. They hide rows in the saved display only; they do not recompute
analytical results. Use `graph.filter(...)` when the selected graph should be
reanalyzed.

### Timelines and portable views

```python
layers = oe.network_snapshots({"before": before, "after": after}, ordered=True)
timeline = oe.plot.network(layers, timeline=True, frame_index=0,
                           layout="geographic", max_nodes=1000, max_edges=5000)
timeline.save_html("timeline.html")
timeline.save_view("timeline.json")
```

The timeline preserves explicit insertion order and stable union node IDs; it
does not infer timestamps, interpolate event durations or support continuous
dynamic GEXF spells directly. Use [explicit interval selection](network-dynamic.md)
and simple projection to construct those snapshots. Each frame preserves its own full/shown counts. At most
60 frames are saved, and the base graph plus all frames together must fit the
100,000-node/1,000,000-edge aggregate cap. `groups=` on a timeline is unsupported;
use scalar attributes for explicit per-frame color mappings.

Worker layouts start once for the selected initial frame. Changing frames,
restoring a saved view or closing a chart cancels its previous worker and
watchdog; retired messages cannot overwrite the current frame or its status.
Fixed layouts and initial saved views start no worker. The
[synthetic worker timeline example](examples/network_timeline_workers.py)
uses four ForceAtlas2 frames with `frame_index=1` to exercise this lifecycle
in the desktop panel without reading or writing user files.

`chart.save_view(path)` saves the complete validated PlotSpec; reopen it with
`PlotSpec.load_view(path)` from `openecon_charts`. Browser **Graph JSON** exports
the PlotSpec including the active frame, mappings, camera and positions;
**View JSON** contains view settings, not graph data. New exported view positions
include typed identity checks. Prefer the complete Graph JSON when transferring
a view, and do not reuse legacy numeric-ID-only views after editing node IDs.
`Network.from_plot_data(chart.config['network'])` reconstructs only the explicit
display subgraph; select a frame first if the payload is a timeline. A chart's
saved view is separate from the graph's `with_positions` attributes.

`save_html()` creates an offline document with bundled renderer, font and worker
code. Python JSON/HTML export does not execute a browser layout. The mounted
chart's export menu provides PNG, SVG, PDF, node/edge CSV, Graph JSON and View JSON.
There is currently no Python `to_pdf()` API: PDF export uses the mounted chart's
current view. SVG/PDF exports refuse more than 200,000 node-plus-edge primitives;
PNG remains available for larger views. PDF embeds its bundled font and reports
unsupported glyphs explicitly; SVG/PNG can be used for those labels. LaTeX
exports a network summary table, not an implicit graph drawing.

WebGL2 can display the admitted graph; when it is unavailable or its allocation
fails, the bounded Canvas fallback draws at most 10,000 nodes and 30,000 edges
and reports the drawn counts. Both paths retain honest full/shown/filter counts.
The chart payload budget is 128 MiB, with 16 MiB of aggregate node-label bytes;
display attributes have their own bounded scalar limits. Raising a display cap
does not guarantee frame rate or bypass any memory, work or vector-export limit.

In the local desktop, network plots above the small inline threshold are stored
as checked project-local artifacts and fetched when displayed, avoiding the
2 MiB worker-output envelope. Artifact storage has a 1 GiB project allowance.
On platforms supporting safe directory-relative cleanup, history expiry collects
eligible unreferenced artifacts, preserving plots created in the last five
minutes; unsupported platforms retain the cache and report that automatic
cleanup is unavailable. Large plots remain local during
team-result sharing: other supported outputs can be shared, with an explicit
partial-result notice for the omitted local graph. The cloud copy does not gain
access to a local filesystem path or silently present the full large graph as
shared. Export the intended JSON/HTML file explicitly when transferring that view.

`max_memory_mb` plans owned labels/hash maps, COO import/merges, sparse graph
storage and method buffers before large tensor allocations. Input rows adapt
downward to that budget. Label identity is retained in an O(V) Python dictionary;
edge storage and aggregation are Torch. Other topology operations use bounded
Python traversal/queues, and PageRank uses Torch reductions. The budget is an
estimate, **not total process RSS**: caller-resident inputs, reader/library caches,
Python/Torch runtime, device allocator behavior and output consumers are separate.
Large V or E can be refused regardless of the physical input's batch size.

`benchmarks/network.py` creates a seeded physical Parquet input and reports import,
degree, PageRank, weak-components and plot-selection times separately, with actual
process peak RSS and source fingerprints. It measures one topology/device; it is
not a speed guarantee. No benchmark result is asserted merely because the script
exists. Independent tests solve small stationary systems with plain Python Gaussian
elimination, verify known shortest paths/loops/isolates and physical-Parquet chunk
invariance, and exercise explicit label, precision, workspace and convergence guards.

The [network pipeline performance protocol](network-performance.md) extends this
with six topology families, separate result-table/JSON/artifact/browser phases,
fresh-process repetitions, profiled controls, and matched full/selected displays.
It includes a runnable installed-app example and scoped CPU/browser/native receipts.

The 2026-10-06 benchmark receipt (internal evidence excluded from this public snapshot) records a
controlled CPU run on an Apple M3 Pro with two Torch threads: 1,000,000 physical
Parquet rows, 100,000 nodes and 999,294 unique undirected weighted edges. Import
took 6.37 seconds, PageRank 0.24 seconds, degree 0.03 seconds and weak components
0.32 seconds. Process peak RSS was 536 MiB; it includes the Python/Torch runtime,
input creation and every phase, separately from the 256 MiB planned graph/buffer
budget. Library import and Parquet creation are excluded from operation timings.
These figures describe this input and machine, rather than every graph topology.

The Mac 0.3.36 delivery receipt (internal evidence excluded from this public snapshot) separates selected
source/UI tests, approved frozen code, packaged graph execution/reopen, installer
comparison, browser export checks and preservation of the existing project.
Physical CUDA and native desktop image downloads were not separately exercised.


## Advanced analysis (SDK 0.3.13a1 / Mac 0.3.37)

```python
communities = graph.communities(method="leiden", resolution=1, seed=42)
display(communities)
print(graph.modularity(communities))
display(graph.betweenness(samples=32, seed=42, max_work=500_000_000))
display(graph.closeness(direction="in"))
display(graph.harmonic(direction="out"))
display(graph.eigenvector())
display(graph.triangles())
display(graph.clustering())
display(graph.core_numbers())
display(graph.topology_summary())
core_graph = graph.k_core(k=3)
neighborhood = graph.subgraph(["origin", "partner", "isolated"])
oe.plot.network(graph, groups=communities, title="Leiden communities")
```

Community detection implements weighted Louvain and Leiden modularity heuristics.
Leiden performs fast local moves, singleton refinement, well-connected eligibility,
seeded randomized nonnegative refinement merges and refined aggregation retaining
its coarse initial partition. It is not a local-moving routine renamed Leiden.
Directed modularity uses the in/out configuration null model, and directed
refinement uses weak connectivity. No global or asymptotic optimality is claimed;
convergence means one unchanged original-graph partition. Results record the
objective, resolution, seed, iteration/hierarchy history and work consumed.
Undirected self-loop diagonal mass is doubled in modularity. Weight scaling rejects
underflow, rather than silently removing positive connections.

Betweenness implements [Brandes's original algorithm](https://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf)
with an indexed O(V) Dijkstra heap and log path counts. It is exact by default;
`samples=k` selects uniform sources and applies unbiased n/k rescaling. That is
an estimator, not an exact result or a guaranteed error bound. Reproducibility is
for the same snapshot node index order. Endpoint/normalization conventions are
reported. Closeness and harmonic are exact all-source calculations; directed
`direction='in'` reverses arcs, while `out` follows arcs. Closeness uses the
Wasserman–Faust disconnected correction by default. Weighted distance metrics use
aggregate edge costs; strength-based eigenvector centrality always uses aggregate
strengths. Its sparse shifted iteration checks both L2 change and a relative
eigen-residual, without assuming uniqueness on reducible graphs. Some directed
acyclic or nearly reducible graphs may legitimately fail its convergence limit.

Triangle counts and clustering use binary positive-edge topology, excluding loops.
Undirected C=2t/[k(k-1)]. Directed graphs use [Fagiolo's all-orientations convention](https://arxiv.org/abs/physics/0612169),
with reciprocal dyad multiplicity and its matching denominator. Core numbers use
[Batagelj–Zaversnik bin peeling](https://arxiv.org/abs/cs/0310049);
directed graphs explicitly use the weak simple projection, not in/out cores.
`density()`, `transitivity()` and `assortativity()` expose matching structural
statistics; constant endpoint-degree variance makes assortativity undefined (NaN).

All analyses operate on the full graph. CPU float64 Torch stores/reduces numerical
buffers; compact CSR memoryviews and bounded Python queues implement irregular
traversals without a graph-library dependency or dense matrix. New algorithms use
CPU; existing PageRank retains optional CUDA. All-source distances/Brandes remain
intrinsically expensive (roughly O(VE) unweighted). `max_work` rejects oversized
exact calculations; it does not silently choose approximations. Community work
is counted across all phases and levels. Triangle intersections have an exact
preflight probe count. Memory guards cover planned native/Python workspaces,
including triangle dictionaries, rather than promising arbitrary-size graphs or
bounding whole-process RSS.

## Static graph interchange

```python
graph = oe.read_network("connections.graphml", max_memory_mb=512)
annotated = graph.with_attributes(nodes={"origin": {"label": "Origin", "team": "A"}},
                                 edges={("origin", "partner"): {"note": "supplier"}})
annotated.to_graphml("export.graphml")
annotated.to_gexf("export.gexf")
annotated.to_pajek("export.net")
```

Import/export supports static [GraphML](https://graphml.ethz.ch/primer/graphml-primer.html),
[GEXF 1.2draft/1.3](https://gexf.net/schema.html) and Pajek Vertices plus Edges/Arcs,
including adjacency-list inputs. Records stream instead of building the whole
XML document or export string. XML is UTF-8; DTD/entities are refused. File,
attribute, nesting and graph-memory budgets are explicit. Export writes a full
graph atomically and refuses existing destinations unless `overwrite=True`.
Typed native identity metadata preserves integer 1 separately from string "1",
including isolates. Native exports also preserve the unweighted-distance flag
when duplicate unit edges carry aggregate strengths.

Node/edge attributes are bounded scalars: UTF-8 strings, bool, int64 and finite
float. Properties return independent copies. Attribute type must remain consistent
across records for XML export; duplicate aggregate edges cannot carry conflicting
attributes. Edge `weight` and native identity keys are reserved. GraphML preserves
scalar graph attributes; GEXF/Pajek exports refuse those rather than discarding
them. Dynamic, mixed, nested/hierarchical and hypergraphs need an explicit snapshot
or projection; this is a static scalar graph interchange implementation, not
blanket support for every feature of those specifications.

The chart accepts `groups=communities` or a complete node-to-group mapping.
Node label attributes control display text while exact node IDs remain distinct.
Group method and full/shown counts survive JSON, LaTeX and interactive HTML;
old stored charts remain compatible. Rendering stays capped for responsiveness;
`subgraph()` and `k_core(k=...)` provide induced graphs for focused exploration.
This does not make a clipped layout an analytical sample.

Independent small-graph tests cover complete partition objectives and aggregate
objective invariance, a Leiden refinement escape fixture, rational simple-path
Brandes oracles, all-source-subset sampling, directed triangle orientations,
reachability/core peeling/Pearson oracles, isolates/loops/duplicates, precision and
budget errors, and static-file round trips. These are method-level validation,
not a blanket claim of external network-application parity.


The advanced CPU benchmark receipt (internal evidence excluded from this public snapshot)
pins the measured source bytes and seeded physical Parquet inputs. On Apple M3
Pro / two Torch threads, 1,000,000 rows and 100,000 nodes produced 999,347 unique
directed edges: import 5.59 s, strong components 0.55 s, core numbers 0.88 s,
weighted incoming eigenvector 0.30 s and exact binary directed clustering 2.19 s.
Clustering performed 6,348,485 membership probes. On a separate unweighted
snapshot of the same file, 32-source sampled betweenness took 11.18 s; it measures
hop distances and is not exact all-source weighted betweenness. The planted
5,000-node / 40,000-row input took 0.39 s for Louvain and 0.59 s for Leiden; both
found 100 groups with modularity 0.906436. Entire-process peak RSS was 522 MiB;
the explicitly allowed owned working budget was 1 GiB. These are measurements
of the documented workloads, not a guarantee for every topology.

The Mac 0.3.37 delivery receipt (internal evidence excluded from this public snapshot) records the
advanced analysis release separately from the earlier 0.3.36 release: selected
source/UI tests, all approved frozen module fingerprints, 27 packaged network
operations and static formats, chart rendering/export, installer comparison,
installed static assets and preservation of the existing project on update.

## Paths, flow and two-mode analysis (SDK 0.3.14a1 / Mac 0.3.38)

This release adds the following 15 public methods. Numerical outputs use the
existing DataFrame/LaTeX API; derived graphs retain exact typed node identities.
`max_work` is a structural-operation budget, not a timeout. Exceeding a work,
pair-output, projected-edge or memory budget raises an explicit error and does
not return a clipped or silently sampled analytical result.

| Signature | Output and scope |
| --- | --- |
| `hits(*, max_iter=1000, tol=1e-10, normalization='l1', max_work=50_000_000)` | Per-node hub/authority scores from aggregate strengths; L1 or L2 normalization. |
| `katz(alpha=None, beta=1., *, direction='in', max_iter=1000, tol=1e-10, normalization='l2', max_work=50_000_000)` | Per-node Katz scores with a checked sufficient contraction bound; L1, L2 or `none` normalization. |
| `shortest_path(source, target, *, max_work=50_000_000)` | One deterministic route: `step`, exact `node` ID and cumulative `distance`. |
| `distances(sources=None, targets=None, *, direction='out', max_work=50_000_000, max_pairs=1_000_000)` | Selected ordered source/target distances in a long table, including diagonal zero entries. |
| `eccentricity(*, direction='out', disconnected='infinite', max_work=50_000_000)` | Per-node eccentricity and number of reachable other nodes. |
| `distance_summary(*, direction='out', disconnected='infinite', max_work=50_000_000)` | One-row average distance, diameter, radius, global efficiency and reachable/unreachable pair counts. |
| `bridges(*, max_work=50_000_000)` | Cut connections of the undirected simple graph or directed weak projection. |
| `articulation_points(*, max_work=50_000_000)` | Boolean cut-vertex flags for every node of that same projection. |
| `minimum_spanning_forest(*, max_work=50_000_000)` | Undirected minimum-cost `Network`, retaining all nodes and isolates. |
| `max_flow(source, target, *, max_work=50_000_000)` | Capacity flow, s-t cut, partitions and checked conservation/cut certificate. |
| `min_cut(source, target, *, max_work=50_000_000)` | The s-t minimum-capacity cut and its certified maximum flow. |
| `maximum_matching(partition=None, *, max_work=50_000_000)` | Maximum-cardinality bipartite matching plus a minimum-vertex-cover certificate. |
| `bipartite(partition=None)` | Infer or validate a complete 0/1 node partition, including isolates. |
| `bipartite_projection(partition=None, onto=0, weight='count', *, max_work=50_000_000, max_edges=1_000_000)` | One-mode `Network` with common-neighbor count, binary or aggregate-strength-product weights. |
| `link_prediction(pairs, method='jaccard', source='source', target='target', *, max_pairs=100_000, max_work=50_000_000, batch_rows=65536)` | Binary similarities for explicitly supplied undirected candidate pairs. |

### Exact routes and distance statistics

Routes share the centrality module's indexed Dijkstra/BFS implementation.
Targeted Dijkstra stops when the requested destination can be finalized; a
caller-owned O(V) predecessor vector reconstructs its route without building or
sorting a reversed O(E) graph. Tied routes choose the smallest snapshot-index
predecessor. Unreachable routes are empty tables with `reachable=False` and
`distance=inf` in their attributes. A source-to-itself route contains one row
with distance zero. Weighted costs are the same aggregate positive edge weights
used by `shortest_paths`; unweighted costs count unique connections.

Targeted float64 queries skip uncertifiable positive-cost additions and defer
their precision decision to one multi-source reachability pass: an ambiguous
branch below the final destination-distance bound is conservatively refused if
it can topologically reach the requested destination, rather than
building an approximate route through it. Irrelevant dead branches do not make
an otherwise valid route fail. Full-source distance calculations retain their
strict precision checks for every reachable node.

`sources` and `targets` are iterables of distinct exact node IDs, preserving user
order; `None` means every node. Directed `direction='in'` reverses arcs. With
both selections omitted, the output has V² rows and is refused above the explicit
`max_pairs` limit. Selecting sources does not reduce the resident graph; it limits
traversals/output. No dense distance matrix or implicit approximation is created.

For `disconnected='infinite'`, any unreachable ordered distinct pair makes the
average distance and diameter infinite. Radius is the minimum full-graph
eccentricity; in directed graphs a node may reach everyone even when others
cannot. For `reachable`, the mean uses only finite distinct pairs and extrema
use reachable-only eccentricities. An isolate then has eccentricity zero, so
it can determine radius. The empty/single-node graph has zero distance summary
values. Global efficiency always averages reciprocal finite distances over
**all** ordered distinct pairs, assigning zero to unreachable pairs; weighted
costs can make efficiency exceed one. These statistics are exact traversals,
not sampled diameter/radius estimates, and all-source work remains expensive.

```python
display(graph.shortest_path("origin", "partner"))
display(graph.distances(sources=["origin"], targets=["partner", "isolated"]))
display(graph.eccentricity(disconnected="reachable"))
display(graph.distance_summary(disconnected="reachable"))
display(graph.bridges())
display(graph.articulation_points())
```

Graph cuts use iterative Tarjan, so long chains do not hit Python's recursion
limit. They ignore positive self-loops and aggregate strengths. In a directed
graph a bridge denotes a **weak projected connection**: removing all directed
arcs/duplicate records on that connection raises the weak component count.
These are not strong directed articulation points or strong directed bridges.

Kruskal's spanning forest uses stable Torch sorting and O(V) union-find. It
rejects directed input, ignores loops and retains every isolate, node/graph
scalar attribute and selected original edge attribute. Weighted costs are
aggregate edge weights; unweighted costs are one per unique connection. The
returned graph preserves the original selected aggregate strengths and weighted
flag, and reports its component count and total optimization cost. Parent and
derived graph buffers are admitted together before allocation.

### Spectral scores and certified capacity flow

HITS implements weighted hub/authority power iteration with both step and
singular-equation residual checks. It follows the hub/authority formulation in
[Kleinberg's original paper](https://www.cs.cornell.edu/home/kleinber/auth.pdf),
using Torch sparse reductions without explicitly forming AᵀA. Edgeless graphs
have zero scores; uniqueness of a dominant vector is not assumed. Exhausted
iterations fail explicitly.

Katz computes `x = beta + alpha A.T x` for incoming scores; outgoing scores
reverse that orientation. `beta` is a finite nonnegative scalar or a complete
exact-node mapping, with at least one positive value for nonempty graphs.
`alpha=None` automatically chooses attenuation satisfying the conservative
maximum-strength contraction bound. Explicit nonnegative `alpha` is accepted
only when it meets that sufficient bound; some otherwise convergent choices may
be refused. `alpha=0` returns the baseline without edge iteration. Metadata
records attenuation, scaling, contraction and an iteration error estimate;
that estimate is not a rigorous floating-point certificate.

Capacity flow uses iterative Dinic, following the layered-network approach
described in [Dinitz's account](https://www.cs.bgu.ac.il/~dinitz/Papers/Dinitz_alg.pdf).
Aggregate weights are capacities even for duplicate unweighted input rows.
Directed flow is nonnegative; an undirected connection has one shared capacity
and a signed flow in its stored source/target orientation. Loops carry zero.
The source and target must be distinct. Power-of-two scaling, capacity bounds,
vertex conservation and flow/cut agreement are checked before a result is
returned. `exact` means no sampling, rather than arbitrary-precision arithmetic;
capacity ranges that lose positive residual updates are explicitly refused.

```python
flow = graph.max_flow("origin", "partner")
display(flow.summary())             # bounded publication table
display(flow["cut_edges"])          # full cut export
display(flow["flows"])              # every original aggregate edge
print(flow["metadata"]["certified"])
```

`min_cut(source, target)` is an s-t cut, not the global minimum over all source/
target pairs. Maximum matching uses iterative Hopcroft–Karp and verifies a
vertex cover of the same cardinality as its matching. Its table attributes
include the exact minimum-cover nodes and unmatched nodes. Directed graphs use
their weak binary projection; weights and duplicate records do not alter
matching cardinality. It solves bipartite cardinality, not weighted assignment
or general non-bipartite matching.

### Bipartite projection and explicit candidate links

`bipartite()` rejects positive self-loops and odd cycles. Inferred components
orient the smallest typed node label onto side zero; inferred isolates use
zero. For semantically named sides, supply a complete node-to-0/1 mapping or a
`node`/`partition` table; supplied orientation and isolate choices are preserved.
Directed validation uses weak topology.

Projection requires undirected input. `weight='count'` counts distinct common
opposite-side nodes, `binary` records connection presence, and `product` sums
products of the original aggregate strengths. The selected side's isolates,
node attributes and scalar graph attributes survive. Original edge attributes
are not transferred to newly derived edges. Projection can create a dense
one-mode graph; neighbor-pair work is admitted before enumeration and growth is
bounded by `max_edges` and the combined parent/derived memory plan.

```python
two_mode = oe.network([
    {"source": "A", "target": "job-1"},
    {"source": "A", "target": "job-2"},
    {"source": "B", "target": "job-1"},
], nodes=["isolated"])
partition = {"A": 0, "B": 0, "isolated": 0, "job-1": 1, "job-2": 1}
display(two_mode.bipartite(partition))
display(two_mode.maximum_matching(partition))
projected = two_mode.bipartite_projection(partition, onto=0, weight="count")
oe.plot.network(projected, title="Common jobs")
display(projected.link_prediction([("A", "B"), ("A", "isolated")]))
```

Link scores accept a pair iterable, endpoint columns, dataframe or batched
`Dataset`. Candidate order and duplicates are preserved; unknown endpoints and
self pairs are refused. Existing connections can be scored when explicitly
requested. No absent-pair list is generated. Methods are `common_neighbors`,
`jaccard`, `adamic_adar` (natural-log degree), `resource_allocation` and
`preferential_attachment`. They use loop-free binary topology and ignore
strengths. These are similarity scores, not calibrated link probabilities or
learned predictors. Directed link scoring is explicitly refused.

### Measured setup and graph-scale operations

The CSR comparison receipt (internal evidence excluded from this public snapshot) measures the
same canonical directed forward graph before/after eliminating redundant sorts
and copies. Forward CSR reuses immutable COO columns/weights and allocates
offsets. Its new numeric owned storage falls from 16,790,168 to 800,008 bytes
(about 16.79 to 0.80 MB); the original resident graph is separate. Reverse or
uncoalesced undirected traversal still needs two stable lexicographic sorts,
whose work and memory are charged. This setup optimization does not imply that
every network analysis becomes equally faster.

On 100,000 nodes / 1,000,000 physical directed rows (999,385 unique edges), median
forward-CSR setup fell from 21.11 ms to 2.56 ms, about 8.24 times faster on the
measured Apple M3 Pro / two-thread CPU run. This comparison excludes its first
warmup and concerns setup of that canonical directed graph only.

`benchmarks/network_paths_scale.py` creates seeded physical weighted undirected
Parquet data, releases creation tensors, imports the full graph in batches and
times exact cuts/forest operations. Its graph-scale receipt (internal evidence excluded from this public snapshot)
records a 100,000-node / 1,000,000-row run with two CPU threads, a 1 GiB allowed
owned budget and an explicitly raised `max_work=200_000_000`. The timings exclude
library import and Parquet creation; process lifetime RSS includes both. Source
fingerprints are checked before/after the measured run. No all-source distance
performance, universal speedup or out-of-core capability is inferred from it.
On the measured Apple M3 Pro run, 999,914 unique edges imported in 5.81 s;
exact bridges took 1.01 s, articulation flags 1.16 s and the minimum spanning
forest 0.69 s. The forest had 99,999 edges in one component and total cost
645,091. Whole-process lifetime peak RSS was about 434 MiB, separately from the
1 GiB planned graph/buffer allowance.

The workflow-scale receipt (internal evidence excluded from this public snapshot) separately
measures flow and matching on its documented physical fixtures. An input's
topology, degree distribution, convergence and output size affect work as much
as its row count. Independent tests use rational Floyd closure, cut deletion,
exhaustive spanning-forest subsets, small spectral systems and matching/flow
certificates; source tests, frozen execution and installed delivery are distinct
validation layers.

## Triad census, global cuts and network association (SDK 0.3.15a1 / Mac 0.3.39)

These three methods ship in SDK 0.3.15a1 and Mac 0.3.39. The
installed delivery receipt (internal evidence excluded from this public snapshot) separately records
source tests, frozen execution and installed application checks. The methods use
CPU execution, the complete resident sparse snapshot and explicit work/memory
budgets.

| Signature | Output and scope |
| --- | --- |
| `triad_census(*, max_work=50_000_000)` | Compact `triad` / `count` DataFrame: 16 directed or four undirected induced three-node classes. |
| `global_min_cut(*, max_work=50_000_000)` | `NetworkCutResult` for a directed outgoing or undirected global minimum-capacity cut, without selected endpoints. Directed support is pending release. |
| `qap_correlation(other, *, values='weight', include_loops=False, permutations=999, seed=0, alternative='two-sided', max_work=50_000_000)` | One-row Pearson dyad correlation and a Monte Carlo bivariate QAP p-value between two aligned snapshots. |

### Exact induced three-node census

`triad_census()` counts each unordered set of three distinct nodes once, including
isolates and disconnected triples. Positive unique edges define binary topology;
weights and self-loops do not affect motif counts. Duplicate input rows are
already aggregated, so they do not create extra motif occurrences.

Directed output contains all Davis–Leinhardt classes in this order:
`003`, `012`, `102`, `021D`, `021U`, `021C`, `111D`, `111U`, `030T`, `030C`,
`201`, `120D`, `120U`, `120C`, `210`, `300`. The three digits identify mutual,
asymmetric and null dyad counts; suffixes distinguish orientations. In particular,
`111D` points into its mutual pair, while `111U` points out of that pair. The
convention is illustrated in the published [sixteen-class triad table](https://www.journals.uchicago.edu/doi/10.1086/692757).
Undirected output contains `empty`, `one_edge`, `two_edge_path`, `triangle`.

The `count` column deliberately uses object dtype with arbitrary-precision
Python integers. Combinatorial counts never round through float64 or overflow
int64. Counts sum to V choose 3; attributes record that total, directedness,
induced/binary semantics, excluded loops, weak triangles and work accounting.
The existing DataFrame LaTeX exporter retains the integer counts.

```python
import openecon as oe

directed = oe.network([
    {"source": "a", "target": "b"},
    {"source": "b", "target": "c"},
    {"source": "a", "target": "c"},
], nodes=["a", "b", "c", "isolate"], directed=True)
census = directed.triad_census()
print(census)
print(census.to_latex(index=False))
```

The implementation combines algebraic wedge/dyadic/null counts with
degree-oriented forward intersections for closed weak triangles. A star's
quadratic number of open triples is counted by integer combinations rather than
enumeration. It constructs no dense adjacency or list of triples. Sparse census
background is given by [Batagelj and Mrvar](https://doi.org/10.1016/S0378-8733(01)00035-1)
and the [large-scale triadic analysis paper](https://arxiv.org/abs/1209.6308).

Workspace is O(V+E), including both directed CSR directions and forward
dictionaries. Conservative linear/sort setup and the exact forward membership
probe bound are admitted before those dictionaries are allocated. Metadata
`work_used` reports charged structural admission units, not measured CPU
operations or a time estimate. Triangle-heavy inputs can exceed the default
work budget even when they fit in memory; no sampled census is substituted.
This census counts a reciprocal directed triangle once as `300`. It differs
from `triangles()`/`clustering()`, whose Fagiolo convention includes reciprocal
orientation multiplicity. It is not motif enumeration beyond three nodes.

### Undirected global minimum-capacity cut

`global_min_cut()` minimizes total crossing capacity over all nonempty proper
partitions. It requires an undirected graph with at least two nodes. It has no
specified source or sink, whereas `min_cut(source, target)` constrains those two
nodes to opposite sides. A disconnected graph, including one with isolates,
has a deterministic zero global cut. Loops are ignored.

Stored aggregate weights are capacities, including aggregate unit weights from
duplicate records when the graph has no explicit weight column. The algorithm
implements [Stoer–Wagner maximum-adjacency contractions](https://doi.org/10.1145/263867.263872).
Each positive stored binary64 capacity is converted to exact common-power-of-two
integer units. All search comparisons and contraction sums are exact for those
stored values. This does not undo rounding during import/duplicate aggregation
or recover a caller's original decimal quantities. The returned `value` rounds
the exact optimum to binary64; an out-of-range result is explicitly refused.

`NetworkCutResult` is a mapping with `value`, `source_partition`,
`target_partition`, `cut_edges`, `metadata`. Both partitions contain exact typed
node labels; `source_partition` contains the smallest typed label by convention,
without implying selected terminals. Full cut rows have `source`, `target`,
`capacity` columns. Metadata includes the exact rational capacity
numerator/denominator and an original crossing-edge readback check, alongside
the Stoer–Wagner optimality guarantee. `summary()` and `to_latex()` export a
bounded publication table, leaving full cut edges separately accessible.

```python
import openecon as oe

undirected = oe.network([
    {"source": "a", "target": "b", "capacity": 2},
    {"source": "b", "target": "c", "capacity": 3},
    {"source": "a", "target": "c", "capacity": 4},
], weight="capacity")
cut = undirected.global_min_cut()
print(cut.summary())
print(cut["cut_edges"])
print(cut.to_latex())
```

Contraction adjacency remains sparse with an indexed O(V) heap. Dictionary
resizes/compaction, simultaneous result buffers and variable-size integer
storage are included in memory admission. Work charges include integer-word
comparisons/additions. Connected cuts still require up to V−1 phases; sparse
memory does not make their runtime linear. A remaining-phase lower bound can
reject a large connected request before contraction dictionaries exist.
Disconnected zero cuts avoid those phases. Exhausting a budget returns an
error, never a partial or approximate cut.

### Directed global minimum-capacity cut (pending release)

On a directed snapshot, `global_min_cut()` minimizes **outgoing** capacity from
any nonempty proper source partition to its complement. Incoming arcs do not
contribute. Reversing the two sides can change the value; the result preserves
the minimizing orientation even when its source side excludes the smallest
typed label. `min_cut(source, target)` remains a separate terminal-constrained
method, and undirected `global_min_cut()` retains its Stoer–Wagner behavior.

The method requires at least two nodes. Zero-capacity input rows do not create
arcs, parallel rows use their stored aggregate capacity, and self-loops never
cross a cut. A graph that is not strongly connected, including isolates,
has a zero outgoing cut. Original-arc reachability or the complement of
transpose reachability provides that cut before any terminal-flow search.
Integer `1` and string `"1"` remain different vertices throughout.

For a strongly connected graph, a fixed typed anchor separates every proper
cut from at least one vertex in one of two directions. Solving both directions
for each other vertex yields the global optimum in `2*(V-1)` terminal flows;
see [Goemans' MIT flow/cut notes, section 4.4](https://math.mit.edu/~goemans/18433S09/flowscuts.pdf).
The implementation uses iterative Dinic with reusable sparse Torch CPU
traversal indices and bounded Python integer residual capacities. Each terminal
flow checks exact edge feasibility, vertex conservation and flow/cut equality.
The final result independently rereads the original outgoing arcs.

As with the undirected method, search arithmetic is exact for stored binary64
capacities. It does not recover decimal quantities before import/coalescing.
`value` is the correctly rounded binary64 export; metadata also retains the
exact rational numerator and denominator. An overflowing export raises a
structured precision error. `cut_edges` contains only original outgoing arcs
in their original direction; both partitions are deterministically sorted by
typed identity. `summary()` reports the number of terminal flow problems.

Copy [this complete code-panel example](examples/network_directed_global_cut.py)
into the editor and Run to display the summary, outgoing edges and full graph.
It exercises unequal reverse capacity, coalesced parallel rows, a self-loop and
distinct integer/string identities, and verifies a global capacity of two.

Sparse workspace is O(V+E) plus the integer capacity bit width. Ordinary Dinic
has worst-case O(V²E) per terminal flow, so this exact rooted reduction has
worst-case O(V³E); sparse storage does not imply fast execution on every graph.
`max_work` is one shared structural/integer-word budget for all flows, including
certificates. A conservative remaining-flow lower bound can reject a strongly
connected request before flow search. Memory admission precedes residual
allocation; exhausting either budget returns an error without a partial,
sampled or approximate cut. CPU execution is explicit in result metadata.

The deterministic [directed benchmark](../benchmarks/network_directed_cut.py)
contains a bidirected ring and extra arcs confined to each half. The ring proves
every proper cut costs at least eight, and cutting the halves proves equality.
On one local ARM64 Mac run, all 500 vertices and 3,000 arcs were analyzed in
22.20 seconds across 998 terminal flows, using 112,218,402 charged work units
with an explicit 500,000,000 budget. Runtime packaging was running on the same
host during this measurement. Lifetime process RSS was 192.2 MB,
including interpreter, imports and input construction; it is separate from
the 376,096 graph-owned bytes and the graph memory budget. This is a full-graph
CPU measurement, with no sampling or out-of-core claim. The default 50,000,000
work budget is insufficient for this measured request. The
scale receipt (internal evidence excluded from this public snapshot) records
environment and source hashes; the
test receipt (internal evidence excluded from this public snapshot) records
the complete network regression run and independent partition oracles.
The frozen desktop receipt (internal evidence excluded from this public snapshot)
separately verifies the example against a freshly built ARM64 Mac runtime,
without injecting the source checkout or accessing human projects. It checks
both publication tables, the full graph payload, terminal/undirected cut
regressions, and identical stored outputs/events after a worker reset. Browser
rendering, replacement of the installed application and release delivery are
not asserted by that receipt; directed support remains pending review/release.

### Bivariate QAP correlation

`qap_correlation(other)` requires identical exact node-label sets, including
isolates, and equal directedness. Snapshot/import order may differ. All eligible
dyads enter Pearson correlation; an absent edge is zero, not a missing
observation. Directed off-diagonal pairs are ordered; undirected pairs count
once. Diagonals are excluded by default and included with `include_loops=True`.
`values='weight'` uses aggregate positive strengths; `binary` uses edge presence.

Uniform node permutations of `other` act on both endpoints of every edge,
preserving its relational structure. This is the
[bivariate QAP formulation](https://www.stats.ox.ac.uk/~snijders/DekkerKrackhardtSnijders.pdf)
in section 2.2, under node-label exchangeability. It does not provide MRQAP
regression, confounder adjustment, a causal conclusion, or a universal test of
zero correlation under arbitrary network dependence.

```python
import openecon as oe

first = oe.network([
    {"source": "a", "target": "b"},
    {"source": "b", "target": "c"},
], nodes=["a", "b", "c", "d"], directed=True)
second = oe.network([
    {"source": "a", "target": "b"},
    {"source": "c", "target": "d"},
], nodes=["d", "c", "b", "a"], directed=True)
association = first.qap_correlation(second, values="binary", permutations=999,
                                    seed=42, alternative="two-sided")
print(association)
print(association.to_latex(index=False))
```

The one-row output contains `correlation`, `pvalue`, `permutations`,
`extreme_permutations`, `dyads`, `nodes`. `two-sided` compares absolute
correlations; `greater` and `less` compare the signed statistic. Ties use a
documented conservative float64 tolerance. With B requested permutations,
`pvalue=(1+extreme_permutations)/(1+B)` and resolution is `1/(1+B)`. Permutations
are sampled uniformly with replacement, including possible repeats/identity;
this is a Monte Carlo test, not enumeration of every node permutation.

Uniform positive weights use an exact integer overlap numerator; general
weights use centered sparse-union products and power-of-two scaling. Empty or
constant dyad vectors have undefined correlation and fail explicitly. Weight
scaling that loses a positive edge, or an eligible dyad count above float64's
exact integer range, is refused. Sparse storage is O(V+E); no V² matrix or absent
dyad list is allocated. Each permutation still includes O(V) relabeling work.
A private CPU Torch generator preserves the caller's global RNG, using canonical
typed-label order; attributes record the seed and Torch version.

The complete requested permutation work is admitted before permutation tensors
are allocated. Both resident snapshots and their shared analytical workspace
count against both graphs' memory budgets; the same snapshot is counted once.
These estimates remain separate from whole-process RSS and caller/reader caches.
The physical-fixture benchmark `benchmarks/network_inference_scale.py` separates
creation, import and method timings, source hashes and process RSS. Row count
alone cannot predict motif, global-cut or permutation cost. Raising `max_work`
or `max_memory_mb` is an explicit allowance rather than a performance guarantee.

### Physical-fixture measurement and installed delivery

The 6 October 2026 measurement (internal evidence excluded from this public snapshot)
uses physical Parquet files on an Apple M3 Pro with two CPU threads. Each method
was measured once, without warmups; runtime startup and graph import are excluded
from the method timings below.

| Method | Retained graph and requested work | Method time |
| --- | --- | --- |
| `triad_census()` | 100,000 nodes, 1,000,000 directed edges; full census checked against analytical band counts | 6.59 s |
| `global_min_cut()` | 500 nodes, 2,000 undirected edges; known optimum of eight verified | 0.71 s |
| `qap_correlation()` | Two graphs of 10,000 nodes and 100,000 directed edges each; 39 permutations and all 99,990,000 eligible dyads | 0.55 s |

Triad graph import separately took 3.81 s. Whole-process peak RSS was about
501 MiB, including runtime libraries, fixture creation, imports and all three
methods. The measurement explicitly allowed 1,024 MiB per graph and 500 million
work units; these allowances are not measured allocations and exceed the default
budgets. The complete sparse graphs remained resident. This single generated
topology is not a universal throughput claim.

Mac 0.3.39 delivery passed 2,080 selected Python tests, 25 editor-catalog tests
and 471 interface tests as separate suites. Three physical-CUDA tests were
skipped because this Mac has no CUDA device. All 380 owned frozen modules match
approved source; the 304 econometrics modules are a subset. The frozen worker
executes 45 network operations/formats, including all three new methods, and
checks ordered outputs, LaTeX, saved exports, reopened history and token rotation.
The installed app matches the read-only installer inventory. The existing
account, open file and prior results were reopened; eleven protected files were
preserved. Only the normal `analysis.version` refresh was allowed and normalized
back to the original index byte hash. Verification ran no human analysis code.

### Remaining specialized scope

This is a broad sparse network-analysis library, not a claim that every feature
of every network application is present. The built-in module still does not
provide integral multicommodity flow or arbitrary model formulations beyond the
validated [native model and learning contracts](network-models.md).
Exact induced graphlets currently
cover four and five vertices; looped and parallel-edge classes are unsupported.
General matching and simple-path enumeration have explicit exponential-work
limits. See [additional native methods](network-extra-methods.md) for sparse
hypergraphs, DSP/joint MRQAP, graphlets, weighted matching and strong directed cuts.

The numerical methods below use a complete resident sparse graph in O(V+E)
memory. Disk snapshots provide bounded construction and adjacency readers;
existing methods can materialize a small snapshot only after memory admission.
Exact all-source
distance statistics can still be too expensive on a large graph even when the
graph itself fits. Candidate/output/projection limits refuse oversized results;
chart rendering retains its responsive display caps and full/shown counts. These
paths/flow/two-mode and census/global-cut/QAP methods run on CPU; existing
PageRank retains optional CUDA. Physical CUDA is not claimed tested, and Mac
Metal float64 remains unsupported. These
boundaries are explicit rather than silently replacing a requested analysis.

### Persistent and automatically selected disk snapshots

```python
graph = oe.network(oe.scan("edges.parquet"), source="from", target="to",
                   weight="amount", directed=True, nodes=["isolated"],
                   store="graph.sqlite", batch_rows=1024,
                   max_memory_mb=32, max_disk_mb=4096)
graph = oe.open_network("graph.sqlite", max_memory_mb=32)
display(graph.summary())
for block in graph.iter_edges(batch_rows=1024):
    consume(block)  # source, target, weight, scalar attributes
for block in graph.neighbors("origin", incoming=True):
    consume(block)  # indexed transpose; undirected neighbors include both ends
```

No engine option is required. Without `store`, an oversized Dataset snapshot
owns temporary storage until `close()` or garbage collection. Explicitly saved
snapshots survive `close()`. Construction never overwrites a destination,
including a path created concurrently by another process. A same-filesystem
exclusive publish makes only a committed, synced database visible at the final
path. Cancel with `cancelled=lambda: should_stop`; ordinary failure removes
staging files. A killed process may leave hidden `.building-*` files and journals,
which cannot be opened as completed snapshots; it does not modify the final path.

Typed string/integer identities, isolate order, directed edges, positive
float64 weight sums, zero-row endpoints and missing-row counters are preserved.
Duplicate weights sum in input order, so fractional sums may differ from a
resident parallel reduction by roundoff. `node_attributes`, `edge_attributes`
and `graph_attributes` accept the same bounded scalar dictionaries as
`with_attributes`; they are stored with their associated identities. Attributes
on missing nodes/edges and reserved weight keys are rejected.

Reopening checks storage integrity, manifest counts and original file identities
by default. If the source is changed or removed, it refuses to open; an explicit
`verify_source=False` opens the captured independent snapshot. This is a stat
identity check, not a cryptographic rehash of source files. A reader also rejects
changes to its graph file during use. `iter_nodes`, `iter_edges`, `neighbors` and
`summary` use bounded reads. `degree`, CPU float64 `pagerank` and weak `components`
use bounded numeric edge scans without constructing resident adjacency tensors.
Other numerical methods, strong components and chart exports raise an explicit
`network_capacity` error on a disk graph. Call `materialize()` explicitly for a
small snapshot that passes the resident import bound. Multi-graph helpers also
require explicitly materialized small snapshots. No sampled approximation,
algorithm or device fallback is substituted.

The quota reserves two-thirds of `max_disk_mb` for staging/journal overhead;
completed database pages use at most one-third. Input parser/runtime allocations
and caller-owned inputs are outside the graph workspace estimate. SQLite's page
cache is bounded and memory mapping is disabled. The measurement record (internal evidence excluded from this public snapshot)
reports source size, database bytes, batch size, indexing/reopen time and whole
process peak RSS separately. It covers synthetic chains, not every topology.
The [code-panel example](examples/network_disk_storage.py) exercises creation,
reopening and outputs without touching existing project files.

### Exact disk-scanned analysis and capacity

```python
display(graph.degree(batch_rows=8192))
display(graph.pagerank(tol=1e-10, max_iter=200, device="cpu",
                       max_work=50_000_000, max_scan_bytes=1024**3))
display(graph.components(connectivity="weak"))
```

The methods retain O(V) CPU Torch numerical state and return complete labelled
node tables. They admit label storage, working state, full outputs and bounded
SQLite/edge buffers against the graph's `max_memory_mb` before creating tensors.
If even O(V) state cannot fit, `network_memory_budget` is raised before setup;
edges being on disk does not make node state unlimited. The estimate covers
owned workspace, not total Python/Torch process RSS or caller-owned inputs.
Node and edge attributes stay on disk; numeric scans project only endpoints and
weights, while result metadata retains graph attributes.

Degree counts unique coalesced edges; undirected loops contribute twice to
degree and strength. Weak components use Torch int64 union-find and assign
deterministic IDs by the smallest typed node label in each component. PageRank
uses stable row-maximum scaling, then a row-total scan and one full edge scan
per power iteration. Undirected off-diagonal edges generate both arcs inside
each bounded batch; loops generate one. Dangling mass follows the normalized
personalization, and the resident contraction-based stopping rule is retained.
Zero damping returns personalization without edge scans. Nonconvergence returns
an explicit error rather than a partial ranking.

`max_work` charges 16 units per node at setup and each PageRank iteration, four
units per stored edge per scan, and one per union-find parent hop. At least one
complete iteration is admitted before setup; later iterations stop explicitly
if either work budget is exhausted. `max_scan_bytes` counts 24 logical numeric
bytes per stored edge per full pass. It excludes SQLite page/index overhead and
cache effects and is not a physical disk-byte counter. Required minimum scans
are admitted before setup. An exhausted budget raises `network_work_budget`
with no partial result. `batch_rows` is capped at 65,536 and reduced to fit the
remaining workspace. Results expose `exact=True`, `sampled=False`, `device`,
`graph_materialized=False`, scan counts, logical bytes, work used and planned
owned memory. Disk PageRank rejects CUDA/Metal requests explicitly.

The analysis measurement record (internal evidence excluded from this public snapshot)
records fresh-process peak RSS, database size, kernel disk-byte counters and
logical edge reads for 50,000/250,000/1,000,000 edges with a declared 8 MiB graph
workspace. All exceed resident admission and match closed-form degree, rank
and component results. Cached reads may contribute zero physical disk bytes;
the receipt keeps that observation separate from actual SQL row scans. These
balanced circulants converge in one PageRank iteration; independent small
asymmetric/dangling fixtures cover iterative numerical agreement.
The [runnable code-panel example](examples/network_disk_analysis.py) displays
four ordered tables with LaTeX. It computes complete 200-node results and explicitly
displays their first ten rows. The [storage example](examples/network_disk_storage.py)
builds and reopens a 30,000-edge chain with an isolate and an 8 MiB graph workspace.
Both examples work in source or installed Mac panels and report the actual runtime.
The isolated frozen ARM64 Mac receipt (internal evidence excluded from this public snapshot)
records the earlier bundled-kernel, worker-reset and temporary-data checks.

The installed Mac acceptance receipt (internal evidence excluded from this public snapshot)
completes the native-panel check: both exact saved examples produce six ordered
tables, and their output/event hashes survive a full application restart. A
separate owned fixture reopens its persistent 30,000-edge physical-CSV graph and
typed-identity graph in the new frozen process. It checks attribute/coalescing/
isolate preservation, source-change refusal, asymmetric iterative PageRank,
unsupported-method refusal, CPU-only behavior and memory/work/scan guards.
The installed compiled storage and kernel modules equal their recorded source.
This is a local Mac runtime maintenance update with the current renderer; the
native shell is retained and existing user files are unchanged. It does not
deliver a public release installer or establish CUDA support.

The Mac 0.3.38 delivery receipt (internal evidence excluded from this public snapshot) records 1,989
selected Python tests, 24 editor-catalog tests and 470 interface tests as
separate suites. All 377 owned frozen modules match their committed source;
the 304 econometrics modules are a subset, not additional coverage. The frozen
worker exercises 42 network operations and interchange formats, including all
15 new methods, ordered tables/LaTeX, history reopening and token rotation.
The finalized installer matches the installed application, whose existing
account, project, open file and prior history were reopened. Ten protected
project files were preserved; only the script-index version refresh was allowed
and normalized back to the original byte hash. No human analysis code was run.

### Multiple-network regression, fixed-K block models and snapshot workflows

`qap_regression(predictors)` fits native linear multiple-network regression over
all eligible dyads, including absent edges as zero. Predictors are a mapping
from distinct names to `Network` objects with the response's exact typed node
labels and directedness. Weighted strengths and binary presence are explicit
choices. Undirected pairs enter once; loops enter only with `include_loops=True`.
No dense dyad matrix is created.

```python
regression = response.qap_regression(
    {'trade': trade_network, 'distance': distance_network},
    permutations=999, seed=42)
print(regression.to_latex(index=False))
```

By default, each slope receives a coefficient-specific
[Freedman–Lane MRQAP test](https://www.stats.ox.ac.uk/~snijders/DekkerKrackhardtSnijders.pdf)
(section 3.2.1): fit the reduced model, permute its response residual by a shared
node permutation at both endpoints, and re-project on the fixed nuisance
predictors. With an intercept the statistic is partial correlation; without one it is
uncentered residual cosine. Neither fabricates independent-dyad standard errors. The intercept is estimated without a permutation
test. Plus-one p-values have resolution `1/(permutations+1)`; the default applies
no multiple-testing adjustment. Conditional residual node-exchangeability and an
appropriate linear mean model are assumptions. Explicit `method='dsp'`, `joint`
and `adjustment` options are described in [additional native methods](network-extra-methods.md).
These conditional permutation procedures do not establish causal effects or
valid inference under arbitrary network dependence.
Rank-deficient, severely ill-conditioned, undefined residual-statistic and
unrepresentable weight-scaling cases fail explicitly, without dropping draws.

Small normalized sufficient-product systems use Torch float64 Cholesky; sparse
residual unions and stable scalar reductions account for all absent dyads
algebraically. Canonical typed labels and sorted predictor names make a private
CPU Torch RNG reproducible without changing global RNG state. Work is admitted
for every requested draw before testing. Joint resident graphs and simultaneously
live sparse residual buffers count against every source graph's memory budget.

`block_model(groups)` is an ordinary hard-label
[Bernoulli stochastic block model](https://pmc.ncbi.nlm.nih.gov/articles/PMC3635708/),
with fixed nonempty K and off-diagonal binary dyads. Positive stored strengths
explicitly collapse to presence under `values='binary'`. For each partition,
block probabilities maximize its actual Bernoulli likelihood; node moves optimize
profile likelihood, rather than modularity. Native sparse structural initialization
and private seeded starts are followed by coordinate ascent. A supplied initial
partition can be a complete exact-node mapping or a `node,block` table.

```python
fit = graph.block_model(4, seed=42)
print(fit.summary())
print(fit['blocks'].to_latex(index=False))
oe.plot.network(graph, groups=fit['membership'], title='Estimated blocks')
```

`NetworkBlockResult` contains membership, block edge/dyad counts, probabilities
and metadata. Zero-dyad cells are unidentified with missing probabilities.
Its display and LaTeX use eleven bounded fit metrics, while full membership and
probability tables remain explicit. Completed no-move sweeps certify only a
single-node local optimum within tolerance. Exhausted iterations are marked
unconverged; exhausted work or memory returns an error without a partial fit.
K is chosen by the caller: this Bernoulli method has no automatic K selection,
degree correction, mixed membership or uncertainty inference. Sparse
CPU Torch count/CSR buffers use O(V+E+K²) memory and each sweep uses O(E+VK²) work.

`poisson_block_model(groups)` fits independent Poisson interaction counts over
off-diagonal dyads, including unobserved zeros. Its `mean_count` is the block
count total divided by eligible dyads. `degree_corrected_block_model(groups)`
fits the [Karrer–Newman count multigraph model](https://arxiv.org/abs/1008.3926),
including observed and zero self-loops. It estimates node-specific degrees
within blocks; directed input has separate outgoing and incoming parameters.
Undirected loop weights are actual loop multiplicities, contributing twice
to degree. Their raw expected count is `theta_i**2 * omega_rr / 2`;
off-diagonal means are `theta_i * theta_j * omega_rs`. Directed means are
`theta_out_i * theta_in_j * omega_rs` for all ordered pairs, including loops.
The directed extension follows the [directed count model](https://snap.stanford.edu/social2012/papers/zhu-yan-moore.pdf).

Both methods accept the same fixed-group, initial-partition, seed, starts,
iteration, tolerance and work options as the Bernoulli method. They return
`NetworkBlockResult`, with eleven bounded summary metrics, explicit membership
and block tables, and publication LaTeX. The reported likelihood is the complete
conditional Poisson log likelihood including factorial constants, rather than
a partition objective with dropped constants. Every completed no-move sweep
certifies only a local solution within the declared tolerance and rounding guard.
The degree-corrected model normalizes node parameters in positive-stub groups;
zero-stub groups export `theta=0` as an unidentified convention with explicit
flags. Node parameter uncertainty and automatic K selection are not estimated.

Each count-model restart uses its own private CPU Torch generator with
`(seed + start) % 2**63`. Changing an earlier start's iteration limit therefore
cannot change later random initial partitions. This revises the sequence used
by older versions for starts after the first; a fixed seed can consequently
select a different multi-start solution after upgrading. `start_fits` retains
each restart's seed, complete likelihood history, per-sweep likelihood gains,
and counts of changed hard memberships. Conditional node/block rates are
reprofiled exactly after membership changes. Trace storage is included in the
workspace budget.

`stopping_reason` distinguishes `no_admissible_move`, `max_iter`,
`fixed_partition` (zero requested sweeps), and `unique_partition` (K=1).
The selected restart's reason and seed are also exposed as top-level metadata.
An iteration limit never changes `converged` into a success flag.

The degree-corrected fitter now uses the same sparse binary adjacency-profile
initialization as the ordinary Poisson fitter for its first non-user start when
edges exist; additional starts remain balanced random. A supplied first
partition is preserved, and empty graphs use balanced starts. Initialization
uses only the fitted snapshot. Its extra O(K(V+E)) work is admitted before CSR
allocation. This changes the degree-corrected first-start partition relative
to older versions; use an explicit `initial` partition with `starts=1` to retain
a specific starting partition.

Nodes with zero incoming and outgoing stubs cannot improve the degree-corrected
profile likelihood through a membership move. They keep their membership and
identification conventions while skipping the K-1 candidate evaluations. Pure
sources and sinks in directed graphs remain eligible for fitting.

Weights must be **stored coalesced integer count multiplicities**, not arbitrary
continuous strengths. Validation occurs after network import, using stored
binary64 weights; original integers beyond binary64 precision or fractional
duplicate inputs can already have rounded or aggregated during import. No
original-input count provenance is claimed. Model count totals are capped at
`2**53`; undirected degree-corrected total stubs also obey that cap. Count weights
on excluded loops are validated too. The ordinary and degree-corrected methods
use different loop sample spaces, so their likelihoods must not be directly
compared as a model-selection statistic. No likelihood-ratio p-value is supplied.

```python
count_fit = graph.poisson_block_model(4, seed=42)
degree_fit = graph.degree_corrected_block_model(4, seed=42)
display(degree_fit)
display(oe.plot.network(graph, groups=degree_fit['membership']))
# Selected pairs only; no node-by-node prediction matrix is allocated.
predicted = degree_fit.expected_edges([('alice', 'bob'), ('alice', 'alice')])
display(predicted)
print(predicted.to_latex(index=False))
```

`expected_edges(pairs, max_pairs=100_000, max_memory_mb=256)` works across all
three block-model families. It returns `source`, `target`, `mean_count`,
`presence_probability` and `identified`. Bernoulli means equal presence
probabilities; Poisson presence is `1-exp(-mean_count)`, evaluated stably.
Only degree-corrected Poisson accepts loop pairs. Pair order, duplicates and
exact integer/string IDs are preserved; a `source,target` table is also accepted.
Unidentified zero-stub conventions remain visible, unknown nodes fail, and
oversized pair outputs are rejected without truncation. Prediction indexes only
the fitted node/block tables and requested pairs, using O(V+K²+P) memory/work.
Its memory budget includes fit tables and planned prediction buffers, excluding
caller input and process RSS. The fitted sparse graph still must fit memory;
these methods use CPU Torch buffers and do not promise graph streaming or GPU.

`network_snapshots(layers, ordered=False)` captures mapping insertion order and
exact string/integer layer IDs without guessing timestamps. Layers can have
changing node sets and isolates, but must share directedness. The mapping is
captured immutably; unique resident graphs are retained by reference and counted
against every graph's memory allowance.

```python
layers = oe.network_snapshots({'before': before, 'after': after}, ordered=True)
print(layers.snapshot_summary())
print(layers.transitions())
print(layers.edge_persistence(max_edges=100_000))
combined = layers.aggregate(reducer='sum')
path = layers.temporal_path('a', 'z')
print(path, path.attrs['reachable'])
```

Transitions report binary edge/node entry, exit and persistence. Edge persistence
has both all-snapshot and observed-endpoint denominators; eligibility uses
compressed node-presence intervals. Aggregation explicitly selects `sum`, `mean`,
`max` or `binary`; means divide by all snapshots, treating edge/node absence as
zero. Strength sums use exact integer units for stored binary64 values before
correctly rounded export. Output caps reject oversized union tables without
silently truncating them.

Strict [time-respecting paths](https://arxiv.org/abs/1108.1780) require
`ordered=True`: at most one edge per snapshot, increasing positions, with waiting
allowed only through continuously observed nodes. The result is an earliest-arrival
witness; `reachable` distinguishes an empty zero-hop witness from no path. There
is no inferred elapsed time, dynamic ERGM/SAOM or coupled multilayer model.

These three additions are native implementations and use no SciPy or external
network solver. Graph ingestion can read physical files in batches; the complete
sparse graph and selected snapshot collection still remain resident. This is not
an out-of-core graph-algorithm claim.

### Fourth-wave physical-fixture measurement

The 6 October 2026 model-workflow receipt (internal evidence excluded from this public snapshot)
uses physical Parquet files, an Apple M3 Pro and two CPU threads. Each method was
measured once with no warmups; Python/Torch startup, fixture creation and graph
imports are separate from the following method times.

| Operation | Retained graph/workflow | Time |
| --- | --- | --- |
| `qap_regression()` | 1,000 nodes, 31,000 total directed edges across response and two predictors; 39 permutations over 999,000 dyads | 2.78 s |
| `block_model()` | 4,000 nodes, 32,000 undirected edges, four supplied initial blocks, one start | 0.41 s |
| `transitions()` | Four snapshots, 25,000 nodes and 100,000 edges per layer | 0.32 s |
| `edge_persistence()` | Full 175,000-edge union with observed-endpoint eligibility | 0.87 s |
| `aggregate()` | Exact stored-strength sums over all four layers | 0.68 s |
| `temporal_path()` | Earliest-arrival strict two-edge witness | 0.07 s |

Whole-process peak RSS was 371 MiB including libraries, creation, imports and
all operations. The run explicitly allowed 1,024 MiB per graph and two billion
work units, exceeding default budgets. MRQAP coefficients matched the known
`Y=3X1+2X2`; block probabilities/profile likelihood were read back independently
for the returned partition. The SBM case started from supplied blocks and is
not a claim of recovering them from random initialization. Snapshot counts,
persistence and aggregate strengths matched analytical sparse-band expectations.
These fixtures are not a universal throughput or maximum-size guarantee.

### Previous Mac 0.3.40 verification

The fresh delivery receipt (internal evidence excluded from this public snapshot) records 2,575 selected
Python passes, 26 editor-catalog passes and 472 interface passes as separate
suites. Three physical CUDA checks were skipped because this Mac has no CUDA
device. All 383 owned frozen modules match committed source; the 304 econometrics
modules are a subset. The frozen worker exercises 55 cumulative network
operations/formats, including ten fourth-wave methods/summary formats. New
MRQAP results match an independently reconstructed dense SVD Freedman–Lane
oracle; tiny SBM likelihood and temporal witnesses receive independent readback.
Ordered outputs, publication LaTeX, block-coloured chart payloads, saved exports,
reopened history and rotated session tokens all passed.

The installed application matches the read-only installer. The existing account,
open `Network.py` file, prior charts and terminal history were reopened. Eleven
protected files were preserved; only the normal `analysis.version` refresh was
allowed, with the complete original index byte hash reproduced after normalization.
No human analysis code was executed. New model/snapshot methods run on CPU, and
this delivery does not claim GPU hardware validation or universal Stata parity.

### Count-model physical-fixture measurement

The 6 October 2026 count-model receipt (internal evidence excluded from this public snapshot)
uses physical Parquet and CSV files with 10,000 nodes, 80,176 positive stored
dyads, varying integer multiplicities and 176 observed loops. On an Apple M3 Pro
with two Torch CPU threads, each method was measured once, with no warmups;
startup, fixture creation, graph import and verification are separate phases.

| Model | Undirected fit | Directed fit |
| --- | --- | --- |
| Ordinary off-diagonal Poisson | 2.88 s | 5.40 s |
| Degree-corrected full-loop Poisson | 2.85 s | 3.62 s |

All fits began from four supplied groups and finished a no-move sweep; these
measurements do not test random-start group recovery. Independent full likelihood
and block rates matched, degree-corrected expected-degree identities were checked
for every node, and 1,000 selected pair predictions per fit matched independent
means and presence probabilities. Prediction took 0.05–0.33 s. Actual chart JSON
was generated with a declared 1,000-node/5,000-edge view cap; analysis used the full
network. Whole-process peak RSS was 217.58 MiB including libraries, fixtures and
imports. The run explicitly allowed 1,024 MiB per graph and two billion work units;
these generated fixtures are not a universal throughput or maximum-size guarantee.

### Installed Mac 0.3.41 verification

The fresh delivery receipt (internal evidence excluded from this public snapshot) records 3,401 selected
Python passes, 27 editor-catalog passes and 472 interface passes as separate
suites. Three CUDA checks were skipped because this Mac has no CUDA device.
All 386 owned frozen modules match approved source; 304 econometrics modules
are a subset. The frozen worker exercises 58 cumulative network operations
and formats, including both new count-model APIs and shared selected-pair
prediction. Independent full-dyad likelihood, observed-loop and zero-stub
oracles passed. Six saved models were reconstructed from CSV plus metadata,
and five prediction tables were recomputed after restart. Ordered output,
bounded fit summaries, publication LaTeX and block-coloured chart payloads
all passed. SciPy is absent from the installed distribution and frozen archive.

The installed app matches the read-only installer and retained the complete
previous .40 application for rollback. The existing account, project, open
`Network.py`, selected prior chart and terminal history were reopened. Eleven
protected files were preserved; only the normal `analysis.version` increment
was allowed, and normalization reproduced the original complete index hash.
No human analysis code was executed or edited. The general econometrics replay
also passed eight feature fits, twenty-two option fits and their saved-model
roundtrips. Its initial 120-second test allowance expired under heavy host load;
the retained failure was followed by a complete successful replay with a
600-second test allowance. This changes no application execution limit or model
option and is not a general performance guarantee. New block models remain
CPU and memory-resident; no universal network-suite or Stata parity is claimed.

## Signed weights

Use the separate [`oe.signed_network` API](network-signed.md) for signed strengths/Katz/modularity/communities and negative-cost paths with explicit cycle refusal. The existing nonnegative graph API is unchanged.

Coupled node-layer identities, explicit inter-layer weights and sparse supra-analysis: [Multilayer networks](network-multilayer.md).
