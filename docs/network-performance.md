# Network pipeline performance

The MARKET-50 benchmark measures physical input, resident graph construction,
complete-graph algorithms, result tables, display selection, serialization,
production artifact storage, and headed-browser drawing separately. Its receipts
describe the measured workloads and machine. They do not establish universal
capacity, all-algorithm speed, cold application startup, or force-layout convergence.

Run from the repository with its Python environment:

```sh
python benchmarks/network_pipeline.py --output /tmp/my-network-run --repeats 3 --large --profile
python benchmarks/network_pipeline_server.py --input /tmp/my-network-run --port 8766
```

Open the printed loopback URL in headed Chrome and keep the benchmark tab visible.
The page runs automatically, verifies each physical artifact's SHA-256 and byte
count, measures three fresh mounts per case, and saves a uniquely named
`browser-*.json` receipt in the owned output directory. The server exposes only
that directory and bundled chart assets on `127.0.0.1`; no hosted data service is
involved. Python refuses an existing output directory. Invalid dimensions are
rejected before creating output. `--large` adds matched full/selected displays
of the same million-edge analytical fixture. `--profile` adds a separate control
run; its instrumented times are excluded from the ordinary medians.

## Workloads and protocol

Each Python repetition starts a fresh subprocess with two Torch threads, seed
716050, CPU float64 analytical buffers, positive float64 weights, and explicit
node identities. The receipts record hardware, Python/Torch versions, direction,
physical input SHA/counts, V/E, memory estimates, every phase, and measured-source
SHA-256 values. Sources must remain unchanged across repetitions.

| Fixture | Nodes/frame | Edges/frame | Frames | Direction | Structure |
| --- | ---: | ---: | ---: | --- | --- |
| Sparse | 10,000 | 40,000 | 1 | Undirected | Four-offset ring |
| Directed | 10,000 | 40,000 | 1 | Directed | Four-offset ring |
| Disconnected | 10,000 | 32,000 | 1 | Undirected | Eight blocks and 2,000 isolates |
| Hub | 10,000 | 39,996 | 1 | Undirected | Central star and three leaf-ring offsets |
| Dense | 600 | 179,700 | 1 | Undirected | Complete graph without loops |
| Temporal | 2,500 | 10,000 | 4 | Directed | Four explicit, changing edge sets |
| Large sparse, full | 100,000 | 1,000,000 | 1 | Undirected | Ten-offset ring; entire display |
| Large sparse, selected | 100,000 | 1,000,000 | 1 | Undirected | Same graph; 2,000 nodes / 10,000 edges displayed |

Degree, PageRank and weak components use every analytical node/edge in every
frame. Tables join on node identity with one-to-one validation. Sampling applies
only to the explicitly selected display. Temporal receipts count both the base
network and four saved frames and separately time collection and transitions.

Synthetic generation and physical Parquet writing have separate clocks. Input
reading uses the warm local file cache after writing. Plot validation and
normalization are separate. `plot_json` is a streaming serialization probe;
`production_artifact_store` uses the actual project store, including repeated
validation, streaming JSON, SHA computation, cleanup checks, fsync and atomic
rename. These overlapping probe/production clocks must not all be added into an
end-to-end latency. Integrity readback is separately timed. Child startup/imports
are outside operation clocks and inside the child RSS high-water mark and suite
wall clock.

Browser phases cover physical HTTP body transfer, SHA verification, UTF-8 decoding,
JSON parsing, production renderer validation/upload, first drawing with an explicit
`gl.finish()`, and two subsequent animation-frame opportunities. The latter is a
paint opportunity, not proof of compositor pixel presentation. The harness checks
the live, error-free WebGL context, drawing-buffer dimensions, uploaded GPU counts
and requested drawing counts. It uses deterministic unit-grid coordinates and
default drawing sizes; GPU coordinates are float32. Assets are warm. Mounts share
one tab without forced garbage collection. Pan measurements retain the individual
15 RAF intervals after five warm-up frames. React shell/history loading, native
WebView startup and force-worker convergence are outside these clocks.

## Recorded CPU results

The Python receipt (internal evidence excluded from this public snapshot) contains 24 ordinary
runs and one separate profiled control on Apple M3 Pro, 18 GiB RAM, macOS 26.6.2,
Python 3.13.5, Torch 2.14.0, two threads. Values below are seconds, medians of three
fresh processes. The local desktop had other applications running; background load
was not isolated. Raw repetitions and ranges are retained, without a percentile
claim from three samples.

| Fixture | Read | Build | Degree | PageRank | Components |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse | 0.028 | 0.182 | 0.0015 | 0.035 | 0.014 |
| Directed | 0.027 | 0.170 | 0.0013 | 0.018 | 0.013 |
| Disconnected | 0.025 | 0.141 | 0.0012 | 0.027 | 0.012 |
| Hub | 0.028 | 0.170 | 0.0013 | 0.021 | 0.013 |
| Dense | 0.028 | 0.724 | 0.0023 | 0.010 | 0.038 |
| Temporal, all four frames | 0.029 | 0.178 | 0.0022 | 0.035 | 0.014 |
| Large sparse, full display | 0.054 | 4.980 | 0.035 | 0.738 | 0.291 |
| Large sparse, selected display | 0.063 | 4.592 | 0.023 | 0.627 | 0.311 |

The last two rows perform identical complete-graph algorithms; they are fresh
measurements with timing variation, not an analytical speedup from sampling.

| Fixture | Select | Validate | Normalize | JSON probe | Production store | Child peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse | 0.042 | 0.279 | 0.121 | 0.210 | 0.668 | 312–340 |
| Directed | 0.035 | 0.279 | 0.127 | 0.211 | 0.672 | 336–337 |
| Disconnected | 0.035 | 0.254 | 0.112 | 0.186 | 0.630 | 296–325 |
| Hub | 0.043 | 0.289 | 0.130 | 0.215 | 0.690 | 291–338 |
| Dense | 0.131 | 0.677 | 0.286 | 0.601 | 1.800 | 370–474 |
| Temporal | 0.230 | 0.389 | 0.170 | 0.322 | 1.046 | 281–303 |
| Large sparse, full display | 1.281 | 6.841 | 4.204 | 5.736 | 17.810 | 752–900 |
| Large sparse, selected display | 0.344 | 0.056 | 0.031 | 0.053 | 0.224 | 377–518 |

The large analytical graph owns an estimated 125.89 MiB in both display modes.
Its full plot contains 71,390,316 JSON bytes; the selected plot contains 780,468
bytes. These distinct figures show why graph-buffer budgeting and Python/export
memory must be reported separately.

The profiled 10,000-node sparse control identifies the production store's repeated
validation and JSON encoder traversal as substantial costs. `cProfile` cumulative
times nest and cannot be summed. This issue adds measurement and profiling;
it changes no production algorithm and claims no optimization speedup. A later
optimization must retain these fixture/seed/thread/dtype/device settings, repeat
the baseline and candidate on the same hardware, preserve physical input hashes
and result/display coverage, and report both raw runs and changed-source hashes.

## Browser and installed-app verification

The browser receipt (internal evidence excluded from this public snapshot) records
24 actual headed Chrome measurements using the Python-produced artifacts and the
production chart renderer. Each case includes full/shown counts, exact byte hashes,
hardware metadata, Chrome/ANGLE renderer, viewport, phase clocks, allocated GPU
buffer/texture storage and separately scoped process/heap observations.

The recorded browser is Chrome 154 with ANGLE Metal / Apple M3 Pro, viewport
1512 × 746, device-pixel ratio 1, and a visible document in every case. Values
below are milliseconds, medians of three fresh mounts. The draw column times
the renderer call plus the explicit GL synchronization request; it excludes
mount/upload and does not represent total panel latency.

| Fixture | HTTP body | JSON parse | Mount/upload | First draw + GL finish | Mount to two RAFs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse | 15.9 | 23.3 | 126.3 | 0.8 | 165.5 |
| Directed | 9.2 | 24.8 | 150.9 | 0.5 | 160.1 |
| Disconnected | 14.6 | 23.5 | 176.8 | 0.3 | 191.5 |
| Hub | 7.3 | 22.8 | 120.8 | 0.3 | 134.9 |
| Dense | 14.7 | 77.0 | 271.6 | 0.3 | 286.7 |
| Temporal, initial frame | 6.3 | 27.5 | 92.5 | 0.4 | 97.1 |
| Large sparse, full display | 70.7 | 494.8 | 1629.1 | 0.3 | 1654.5 |
| Large sparse, selected display | 4.5 | 6.2 | 39.4 | 0.3 | 46.8 |

The full large graph reached 100,000 uploaded/drawn nodes and 1,000,000
uploaded/drawn edges with an error-free live context. Its mount times ranged
from 973.9 to 1958.4 ms; the selected display ranged from 35.3 to 39.6 ms.
Full drawing allocated 30,800,000 bytes in GPU buffers and 1,605,632 bytes in
the position texture. These allocations exclude driver/framebuffer memory.
UTF-8 decode, body SHA validation, raw RAF intervals and temporal frame-change
clocks are retained separately in the receipt.

The installed Mac receipt (internal evidence excluded from this public snapshot)
records the smaller [runnable companion](examples/network_performance.py) in the
installed 0.3.42 app: 2.65 seconds for code execution, one six-row result table,
six persisted charts, four temporal frames, and all charts observed as
`WebGL2 · Ready`. The script matches its saved history byte-for-byte; three
artifact-backed plots pass checksum readback and the other three are inline.
Graph pixels were visually inspected. These native timings are separate from
browser first-draw timings. The existing user project's 20 files remained
byte-identical. No new installer or public release was produced.

## Memory and admitted budgets

Python RSS is the OS lifetime high-water mark for the fresh child, including
imports, generated input, resident tables, validation copies and serialization.
Darwin reports bytes; Linux reports KiB and is converted to bytes. Phase RSS
marks are cumulative, not isolated phase peaks. Owned graph buffers and planned
import peaks are separate estimates, excluding caller-resident input and exported
Python objects. The benchmark's 2 GiB graph allowance is not a process-memory
guarantee; the small native companion uses 64 MiB per graph.

The browser's 200 ms RSS sampler covers **all Google Chrome processes on the
host**, including other tabs and shared GPU/browser processes. Summed RSS may
double-count shared pages and miss short peaks. It is an observed shared-process
peak, not private-tab memory or an exact OS lifetime peak. Missing process
observations are null. JavaScript heap and allocated GPU buffers/position textures
have separate scopes and are not substituted for process RSS or total GPU memory.

Displays remain subject to 100,000 aggregate nodes, 1,000,000 aggregate edges and
128 MiB serialized payload, including timeline base plus frames, and a 1 GiB
project artifact-store allowance. Timelines permit at most 60 frames. Canvas
fallback's 10,000-node / 30,000-edge drawing budget is not benchmarked as a complete
WebGL graph: this harness fails if requested coverage or WebGL allocation is not
reached. PageRank uses tolerance 1e-10 and at most 200 iterations, with explicit
convergence metadata. These tested sizes and guards do not establish an unlimited
network capacity or coverage for other algorithms/topologies/devices.
