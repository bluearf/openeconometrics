# Continuous interval networks

[MARKET-58](https://linear.app/bluearf/issue/MARKET-58) adds resident continuous
interval data through `oe.dynamic_network()` and `oe.read_dynamic_network()`.
Spells, timed scalar attributes, parallel/zero/loop edges and distinct typed IDs
remain separate. Analysis begins with an explicit point or window selection;
each returns `MultiNetwork`. General simple-network methods then require the
[explicit projection reducer](network.md#separate-edge-identities-and-parallel-connections).

```python
import openecon as oe

g = oe.dynamic_network([
    {'node': 'a'}, {'node': 'b', 'start': 1}, {'node': 'isolated'},
], [
    {'edge_id': 1, 'source': 'a', 'target': 'b', 'weight': 2,
     'spells': [{'start': 0, 'end': 3}],
     'dynamic_attributes': {'status': [
         {'value': 'early', 'end': 1}, {'value': 'late', 'startopen': 1},
     ]}},
    {'edge_id': '1', 'source': 'a', 'target': 'b', 'weight': 0},
], directed=True, version='1.2draft')
point = g.at(1)
point.degree()              # Both edge IDs exist, including the zero edge.
window = g.window(1, 2, attributes='drop')
simple = point.to_network(reducer='count', attributes='drop')
simple.pagerank()
g.write('owned.gexf')        # Refuses an existing destination by default.
reopened = oe.read_dynamic_network('owned.gexf')
```

The [complete runnable example](examples/network_dynamic.py) creates only
synthetic data and its own temporary file. It displays five tables, checks exact
boundary behavior and typed identity, explicitly constructs ordered point
snapshots, and verifies full GEXF readback without changing the source.

## Records and time semantics

Input is an iterable of node dictionaries followed by an iterable of edge
dictionaries. Nodes require `node`; edges require `edge_id`, `source`, `target`
and declared endpoint nodes. Optional fields are `weight`, `attributes`,
`spells`, `dynamic_attributes` and direct time bounds. Node/edge IDs are unique
within their scope; integer `1` and string `"1"` differ. Boolean/float IDs are
rejected. Raw edge order and orientation are preserved, including for undirected
input. Weights must be finite nonnegative binary64 values; omitted weights are
one and zero-weight edges remain stored. The engine uses CPU Torch only when
the selected static graph is analyzed; no NetworkX/SciPy solver is introduced.

`spells` is a nonempty sequence of interval dictionaries. Omitting spells means
unbounded presence. A record may instead declare direct `start`/`end` bounds;
using both forms is rejected. Unordered/overlapping spells retain their original
order for interchange and use a separate normalized union for selection.
Graph-level `interval` intersects every node's presence. Edge activity is its
own presence intersected with the graph lifetime and both endpoint nodes.
Timed attributes do not independently create node/edge presence.

| Time format | Accepted representation and precision |
| --- | --- |
| `double` (default) | Finite binary64 numeric time; integer inputs that would lose precision are rejected. |
| `integer` | Exact signed integers of at most 256 bits, including bounded decimal XML tokens. |
| `date` | Valid ISO `YYYY-MM-DD`, or a Python `date`; integer day coordinate. |
| `dateTime` | ISO timestamp or Python `datetime`, normalized to UTC with exact microsecond coordinates. Missing timezone explicitly means UTC. Explicit offsets are honored; precision beyond six fractional digits and leap seconds are rejected. |

An omitted start/end is negative/positive infinity, respectively. Equal closed
bounds represent a point; reversed or empty open-point intervals are rejected.
GEXF 1.2draft supports `startopen`/`endopen`, whereas 1.3 intervals use inclusive
`start`/`end`. A bound cannot declare both variants. Choose `version='1.2draft'`
to preserve an open stored interval; 1.3 export never changes openness silently.
These conventions follow the [GEXF dynamics specification](https://gexf.net/dynamics.html)
and [version history](https://gexf.net/history.html). Query windows may use
`start_open=True`/`end_open=True` independently of stored-file version.

## Explicit selection and attributes

`g.at(time)` admits every entity active at that exact time. `g.window(start,end)`
uses `selection='overlap'` to admit any activity in the window, or
`selection='cover'` to require uninterrupted activity over the entire window.
Missing window endpoints mean unbounded queries. Selected node order and edge
IDs/order follow the source. No implicit sampling, time interpolation, duration
averaging or temporal statistical model is applied.

`attributes` contains bounded scalar string/bool/int/float fields.
`dynamic_attributes` maps a name to nonempty sequences of dictionaries containing
`value` and optional interval bounds. Constructor `node_defaults`/`edge_defaults`
are dynamic fallback values; a missing attribute outside its active value
interval remains absent when no fallback exists. Edge `weight` is a special
dynamic field, with the base weight as fallback unless a dynamic default is
declared. Static edge weight belongs in the record's `weight` field.

At a point, conflicting overlapping values raise `network_dynamic_ambiguity`.
Different scalar types are distinct values. In a window, varying, conflicting
or intermittent attributes also raise by default. Explicit `attributes='drop'`
drops only affected named fields and records the entity IDs/fields in selection
metadata. Constant fields remain. Varying weights require explicit
`weight='min'` or `'max'`; these choose extrema across active values/fallback,
without computing duration averages. Resolution considers times at which the
selected entity is present. Two differing inclusive values meeting at the same
endpoint are ambiguous at that endpoint; the reader does not choose XML order.

`nodes()`/`edges()` expose original spells and complete timed records rather than
resolved attributes. Their returned nested dictionaries are independent copies.
`metadata`, `defaults` and `interval` also return independent values.
`at()`/`window()` preserve the source. A window union does not establish a
continuous time-respecting route. Existing `oe.network_snapshots(...,ordered=True)`
retains its existing layer order and earliest-arrival behavior after explicit
point selection and projection; see the example's two-hop witness.

## GEXF interchange and refusal boundaries

`read_dynamic_network()` supports matching 1.2draft/1.3 namespaces, one dynamic
interval graph, node/edge IDs, graph/node/edge intervals, node/edge spells,
static/dynamic scalar declarations and defaults, scalar values, edge weights
and core labels/kinds. Forward declarations and edges preceding nodes are
resolved in bounded passes. Export uses native typed-ID scalar fields to retain
integer versus string IDs. Numerical/date times are canonicalized at input;
dateTime offsets normalize to equivalent UTC instants. Entity spell/value order,
scalar values, direction, raw endpoint orientation and edge identity survive
native read/write. Unused declaration schema and descriptive meta annotations
are outside the analytical-record preservation contract.

File mode requires explicit unique wire edge IDs. Mixed directions, timestamps,
compact `intervals` strings, nested/hierarchical nodes, visualization elements,
unknown XML fields/extensions, multiple graphs, non-UTC graph timezone defaults,
unsupported scalar types or conflicting declarations raise controlled errors.
UTF-8 only; DTD/entities are rejected. Identity/default conflicts and undeclared
endpoints are refused. Files are hashed across three passes; changed source
content fails. Static `read_network()`/`read_multigraph()` keep refusing dynamic
GEXF instead of flattening it. GraphML/Pajek export requires an explicit static
slice. Writes use an owned temporary file and atomic completion; failed writes
leave existing destinations intact and remove only their own temporary file.

## Resource limits and reproducible evidence

This representation remains resident. `max_memory_mb=256` governs planned owned
records, interval indexes, parser/wire-ID state and selected-output workspace;
caller-held input, additional retained result objects and total process RSS are
outside that bound. Large complete `nodes()`/`edges()` tables require admission
before materialization. Slice admission includes the resident source and planned
slice buffers. `max_events=1_000_000` counts lifetime spells and timed values
(maximum configurable value ten million), with at most 4096 events per entity.
Scopes admit 128 scalar fields; scalar strings are bounded. XML depth, token and
per-record children are separately bounded. Exceeding a budget fails rather than
truncating events or silently switching representation.

Construction and selections use `max_work=50_000_000` admitted interval work.
Reading additionally uses `max_file_mb=512`; three sequential file scans are
required. Actual disk capacity/I/O and caller-held graphs remain constraints.
No GPU, out-of-core dynamic graph, ERGM/SAOM, continuous-time path algorithm or
blanket GEXF application compatibility is claimed.

```sh
python -m pytest tests/test_network_dynamic.py
python docs/examples/network_dynamic.py
python scripts/benchmark_network_dynamic.py --output-dir owned-dynamic-benchmark
python scripts/verify_network_dynamic_desktop.py \
  --runtime /absolute/path/to/owned/openecon-runtime \
  --output new-frozen-dynamic-receipt.json
```

The [283 independent interval cases](../tests/test_network_dynamic.py) compare raw
rational membership and window coverage, not production interval helpers, and
cover typed identities, changing weights/attributes, XML defaults, resource
refusals and unchanged snapshot route order. After integrating GPU/sharing main,
4,179 Python cases and 630 web cases passed (internal evidence excluded from this public snapshot),
with seven optional physical CUDA checks skipped; the UI build, source lint and
393-entry committed-source catalog verification passed. These totals describe
the network, editor-catalog and packaging subset, not the entire Python project.
After the subsequent MARKET-110 merge, 344 dynamic/workflow/publication cases
and all 654 current web cases passed with a fresh UI build. Both revisions and
their distinct scopes are recorded in the validation receipt. The final MCP-job
integration additionally passed 352 dynamic/workflow/catalog/packaging/MCP cases;
its workspace changes leave the checked network/SDK modules unchanged.

The fresh-process scale receipt (internal evidence excluded from this public snapshot)
checks complete ordered typed-record fingerprints before and after native GEXF
read/write, with one spell and two timed attribute values per edge. Independent
scalar arithmetic checks point membership/degree and whole-window coverage.
Raw parallel edges, zero weights, loops, graph lifetime and source immutability
are retained. Declared planned owned workspace is 2048 MiB per operation;
observed process RSS is measured separately. Runs use three nodes and CPU with
one Torch thread, so they do not establish arbitrary-topology performance.

| Edges | Build | Point/degree | Cover window | Write | Three-pass read | Peak process RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 25,000 | 1.16 s | 0.67 s | 0.19 s | 0.64 s | 5.01 s | 295.6 MiB |
| 100,000 | 3.95 s | 1.58 s | 0.74 s | 2.29 s | 26.76 s | 501.5 MiB |

Installed application delivery, browser pixels and a public release require
separate verification. The isolated frozen ARM64 proof (internal evidence excluded from this public snapshot)
executed the runnable example from the bundled SDK/modules, generated five
ordered tables with publication LaTeX in 7.164 seconds, and read identical stored
outputs/events after resetting the worker. It checks compiled-code parity for
the seven listed network/SDK modules; it does not assert byte equality for every
application module. The verifier launched no source-path override, stopped its
own runtime and removed its temporary data. Human projects and the installed
application were untouched in that isolated check. It was packaged
computation/persistence evidence, with installed distribution and rendering
outside its scope.

The subsequent installed Mac acceptance (internal evidence excluded from this public snapshot)
used the frozen runtime already installed for MARKET-59 without another update.
Native Run generated five complete tables in 1.909 seconds; exact boundaries,
typed identity, explicit window loss, ordered temporal routes and full GEXF
readback passed. Seven selected compiled modules matched the recorded source
revision. Fully quitting/reopening preserved code/output/event hashes without
starting a Python worker, and 18 pre-existing source/data files stayed unchanged.
Native screenshots, exact synthetic QA code and scoped receipts are retained.
Local LaTeX/rendering/persistence passed; automatic cloud sharing encountered the
deployed MARKET-87 publication validator mismatch. Recipient delivery,
notarization and public release remain outside these checks.
