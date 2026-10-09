# Network support and independent numerical validation

This is the bounded validation record for [MARKET-54](https://linear.app/bluearf/issue/MARKET-54).
The machine-readable receipt (internal evidence excluded from this public snapshot)
records source fingerprints, test totals, full public-data checks and a separate
installed Mac runtime replay. Runtime estimators still use native algorithms and
Torch; this work adds no SciPy, NetworkX or other solver dependency.

## What was checked

The existing 30 network test files passed 3,353 cases on the pinned source;
the new public-reference file passed eight additional cases. Many existing
cases enumerate every small topology or permutation. Others test resource
guards, import/export, integration or output persistence; the count is not
3,361 independent real-world datasets. The matrix below identifies the actual
reference and relevant method assumptions.

| Family | Supported semantics on the tested source | Independent reference and evidence | Limits |
| --- | --- | --- | --- |
| Resident import, degree, strength, PageRank, components | Typed string/integer IDs; directed/undirected; nonnegative aggregate strengths; isolates, zero input weights, positive loops | [Scalar stationary linear system and analytic counts](../tests/test_network.py); [new public/perturbed topology checks](../tests/test_network_public_references.py) | Resident graph is O(V+E); CPU float64 measured. CUDA PageRank exists but physical CUDA remains unverified. |
| Disk snapshots; degree/strength, CPU PageRank, weak components | Same typed identities and nonnegative semantics; bounded edge scans, admitted O(V) state/output; explicit unsupported-method and resource errors | [Disk storage tests](../tests/test_network_store.py); [resident/independent Markov comparisons](../tests/test_network_disk_algorithms.py); 50k/250k/1m-edge RSS and kernel I/O (internal evidence excluded from this public snapshot) | Selected CPU methods verified in source and an isolated frozen ARM64 worker (internal evidence excluded from this public snapshot). Other disk methods, CUDA disk kernels and installed release delivery remain unverified or unsupported. |
| Separate edge-ID multigraphs | Typed unique edge IDs, raw orientation/order, parallel/zero/loop records, per-edge scalar attributes; immutable edits; mandatory explicit simple projection | [Independent scalar reductions, analytic counts, strict XML and persistence checks](../tests/test_network_multigraph.py); 25k/100k ordered roundtrip fingerprints and RSS (internal evidence excluded from this public snapshot); isolated frozen ARM64 replay (internal evidence excluded from this public snapshot) | Resident CPU only. GraphML/GEXF require edge IDs in this mode; Pajek is rejected. Native multigraph analysis is degree/strength/summary; other algorithms require an explicit reducer. Installed/release delivery and browser pixels remain unverified. |
| Continuous interval GEXF | Graph/node/edge lifetimes, spells and timed scalar values/defaults; exact typed node/edge IDs; explicit overlap/cover or point selection into multigraphs | [Independent Fraction membership/coverage and strict XML tests](../tests/test_network_dynamic.py); [support and evidence](network-dynamic.md) | Resident CPU only. Inclusive 1.3 or explicit open/closed 1.2draft intervals; dateTime to microseconds. Ambiguous values raise unless a documented window reduction is selected. No implicit continuous-time paths or mixed/nested/timestamp flattening; installed/release delivery and browser pixels remain unverified. |
| Betweenness, closeness, harmonic | Positive aggregate cost for weighted paths; unique-edge hops without a weight column; incoming/outgoing and unreachable conventions | [All simple paths with exact Fraction distances and hand formulas](../tests/test_network_centrality.py); public distance-sum identity | Exact all-source methods use `max_work`; betweenness sampling is an explicit estimator, not an exact result. |
| Eigenvector, HITS, Katz | Sparse weighted incoming eigenvector; HITS singular vectors; bounded Katz linear system | [Rank-one singular vectors and Fraction linear systems](../tests/test_network_spectral.py); [spectral centrality formulas](../tests/test_network_centrality.py) | Residual/iteration checks; nonconvergence is explicit. Reducible graphs do not imply a unique eigenvector. |
| Louvain, Leiden, modularity | Weighted configuration-model objective; directed out/in null model; seeded local solution | [All small partitions and brute-force move/objective checks](../tests/test_network_communities.py); exact rational objectives on Karate | Heuristic community discovery has no global optimality guarantee. Checking the objective validates the returned partition, not a unique true clustering. |
| Distances, routes, bridge/articulation, spanning forest | Directed paths; weak projection for structural cuts; aggregate cost versus strength distinguished | [Scalar closure, vertex/edge deletion, exhaustive subsets](../tests/test_network_paths.py); min-plus closure of all Karate pairs | Pair tables use `max_pairs`; traversal uses `max_work`. Negative costs and strong directed structural cuts remain unsupported. |
| s–t max-flow/min-cut, global cut | Nonnegative aggregate capacity; signed shared undirected flow; loops cannot carry net flow; undirected global cut | [Every small cut, exact Fraction capacity, primal conservation](../tests/test_network_flow.py); [all global partitions](../tests/test_network_global_cut.py); public Karate primal/dual certificate | Directed global cut, min-cost and multicommodity flows remain separate backlog items. |
| Bipartite, projection, matching, link scores | Typed partitions; weak directed bipartite/cardinality matching; explicit candidates for undirected link scores | [Complete color assignments, Fraction products and set scores](../tests/test_network_bipartite.py); [exhaustive cardinality](../tests/test_network_flow.py) | Projection has edge/work caps. Matching maximizes cardinality, not total weight; general non-bipartite matching is absent. |
| Triangles, clustering, cores, assortativity | Positive simple topology; loops/strength ignored where documented; directed Fagiolo clustering and weak core | [Dense cube/set counts, threshold pruning and scalar formulas](../tests/test_network_topology.py); published GR-QC statistics | Triangle work is bounded. Binary directed clustering does not implement arbitrary weighted clustering. |
| Directed triads | 16-class census on loopless simple directed topology | [Independent permutation/isomorphism classes](../tests/test_network_motifs.py) | No four-node motif/orbit support is claimed. |
| Bivariate QAP | Weight/binary dyads, exact typed alignment, optional loops, node permutations and chosen tail | [Dense Fraction-centered dyad vectors and every node permutation](../tests/test_network_qap.py) | Monte Carlo p-value uses the plus-one correction; assumes the documented label exchangeability. |
| MRQAP regression | Coefficient-specific Freedman–Lane residual permutation; fixed predictors and nuisance projection | [Explicit dense reconstructed response, SVD fits and rational coefficients](../tests/test_network_mrqap.py) | Approximate conditional inference; DSP, joint tests and multiple-test adjustment are not implemented on this source. |
| Bernoulli/Poisson/DC Poisson blocks | Fixed K hard labels; Bernoulli/Poisson exclude loops; DC Poisson includes its documented loop exposure | [Bernoulli dyad likelihood](../tests/test_network_sbm.py), [Poisson count likelihood](../tests/test_network_poisson.py), [DC full likelihood and exhaustive moves](../tests/test_network_dc_sbm.py) | Count data cannot be silently replaced with continuous weights. Local convergence is not a global optimum or predictive success. MARKET-55 tracks diagnostics/holdout improvements separately. |
| Block predictions and identity alignment | Explicit selected pairs, means/presence probability, identified flags and preserved pair order | [Independent one-group count/degree ratios, typed IDs, duplicate candidates](../tests/test_network_block_prediction.py); [persisted ordered results](../tests/test_network_count_models_integration.py) | Unidentified zero-stub groups remain flagged; no future holdout is used to fit these references. |
| Snapshots, turnover, aggregate, temporal paths | Ordered observed snapshots; exact typed node/layer identities; one hop per snapshot with waiting rules | [Raw sets/Fraction aggregates and explicit time-expanded states](../tests/test_network_temporal.py); [persistent workflow](../tests/test_network_workflows_integration.py) | Snapshots are not coupled multilayer graphs, continuous-time GEXF, SAOM or temporal statistical estimation. |

Disconnected graphs, isolates, positive self-loops, zero-weight input rows,
directed/undirected semantics, duplicate aggregation and integer `1` versus
string `"1"` are checked across the corresponding families. This does not imply
that every method supports every such combination; unsupported domains have
explicit error tests. Tests of a work/memory preflight check are distinct from
actual scale measurements.

## Public data and reproducible comparison

[Zachary's Karate network from Mark Newman](https://public.websites.umich.edu/~mejn/netdata/)
has 34 nodes and 78 unweighted edges in the pinned file. Original integer IDs
1 through 34 are retained. The independent validator checks raw degree and
reachability, neighbor-pair triangle counts, local clustering, scalar Gaussian
PageRank, all-pairs min-plus distances, the exact betweenness distance-sum
identity, rational modularity for two seeds of both community algorithms,
and a feasible primal flow with an equal independently summed cut. Additional
tests perturb this public topology with positive weights, direction, a duplicate,
a loop, zero rows, isolates and mixed typed identities.

The pinned [Stanford GR-QC archive](https://snap.stanford.edu/data/ca-GrQc.html)
has 28,980 physical records. Canonical deduplication before graph construction
produces 5,242 nodes and 14,496 undirected edges, including 12 loops. This is
an explicit unit-strength projection. The raw file's reciprocal records would
otherwise add strengths under OpenEconometrics' normal coalescing contract.
Loops remain for degree/import checks and are excluded from triangle/clustering
counts. The measured full graph matches 48,260 triangles, a largest component
of 4,158 nodes and mean local clustering 0.5296358110521362. The published
four-decimal clustering value 0.5296 is compared with absolute tolerance 0.00005.
Full-precision scalar comparisons use absolute tolerance 2e-11, PageRank 2e-12,
and integer path/topology comparisons are exact.

Source citations, archive/extracted hashes and transformations are in the
[fixture manifest](../tests/fixtures/network_reference/manifest.json) and
[fixture notes](../tests/fixtures/network_reference/README.md). Reference
arithmetic reads these raw files and never calls production CSR/likelihood/path
helpers to obtain expected values. The public graphs are empirical topologies;
weighted/directed perturbations are deliberately synthetic test cases.

```sh
python -m pytest tests/test_network*.py
python docs/examples/network_reference_validation.py --output reference-receipt.json
```

For the code panel, paste
[the validation example](examples/network_reference_validation.py) and append:

```python
receipt = validate("/absolute/path/to/openecon/tests/fixtures/network_reference")
display(oe.DataFrame([
    {"dataset": name, "nodes": row["nodes"], "edges": row["edges"],
     "triangles": row["triangles"]}
    for name, row in receipt["datasets"].items()
]))
```

The terminal runner does not replace existing receipts. Its elapsed time
includes file checks, graph import, reference arithmetic and analytical results;
it is not a pure estimator or renderer benchmark. Peak process RSS includes
Python/Torch imports and independent reference buffers, separately from the
graph's 256 MiB owned-buffer allowance. Full/shown graph counts and no analytical
sampling are recorded. Measurements are local CPU evidence on Apple M3 Pro;
they do not establish every algorithm's performance at GR-QC scale or beyond.

## Installed Mac replay and remaining limits

[The desktop verifier](../scripts/verify_network_reference_desktop.py) launches
the executable from the installed Mac application's bundle with a temporary
data root, ephemeral loopback port and disposable project. It executes the
same references in frozen Python, verifies the SDK comes from that bundle,
produces ordered table/chart outputs, resets the worker, and reads the saved
outputs/events back. The generated Karate chart contains all 34 nodes and 78
edges. It verifies runtime/fixture/example fingerprints, stops its own runtime
and removes only its temporary data. No human account/project/files or running
app installation is changed.

```sh
python scripts/verify_network_reference_desktop.py \
  --runtime /Applications/OpenEconometrics.app/Contents/Resources/runtime/openecon-runtime/openecon-runtime \
  --output frozen-reference-receipt.json
```

This is actual installed frozen computation and persisted panel-output evidence.
Native window/browser pixels, first paint, force-layout lifecycle and FPS were
not measured by this verifier; MARKET-49/50 cover those separately. Source
tests and installed runtime replay are separate evidence, with no blanket claim
that their entire source inventories are byte-identical. Windows/Linux frozen
packages, physical CUDA, Metal float64, every topology/scale combination,
unsupported disk algorithms, installed delivery of the new disk methods and the remaining advanced-method backlog remain
unverified or unsupported as stated in their own issues.
