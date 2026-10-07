# Historical performance measurements — OpenEcon 0.2

**Archive, not a measurement of the current release.** The 0.3 workbench has one
required PyTorch CPU core, its own inference routines and a changed public API.
Every timing and comparison below concerns the former 0.2 dual-engine
implementation. They establish no current speedup or GPU-performance claim.

The original benchmark source is preserved at
[benchmark_v0_2.py](../benchmarks/reference/benchmark_v0_2.py). Its commands and
extra dependencies below describe the historical environment; they are not
current setup instructions. To inspect the archived result without running a
benchmark:

```sh
uv run python scripts/benchmark.py --show-report
```

A bounded measurement of the current single-core workbench is documented in
[scaling.md](scaling.md). It covers local synthetic workloads and process RSS;
production large-data capacity and a matched estimator comparison remain
unmeasured. The measurements below still concern the former 0.2 implementation.

## Archived 0.2 native-engine comparison protocol

The measured 0.2 estimator runtime had native NumPy/SciPy and optional PyTorch engines.
`statsmodels` is a development-only numerical oracle and a preserved historical
adapter; it is not the native computation implementation. A GPU backend is not
assumed to be faster: transfer, validation and inference are included in the
comparison, and results are reported only for devices actually measured.

Reproduce the matched comparison from the repository root:

```sh
uv sync --locked --extra torch
uv run --extra torch python scripts/benchmark.py
```

The benchmark uses the same in-memory DataFrame, ModelSpec, covariance method and
result conversion for the former statsmodels adapter, native NumPy and native
PyTorch CPU. When CUDA is available it also measures PyTorch CUDA, synchronizing
before and after timing. CPU-to-device transfers are included. MPS is not measured;
the estimator contract uses float64, which is not an MPS inference option here.

The workloads comprise OLS/HC3 with 10,000 and 100,000 rows, each with five and
50 numeric predictors plus an intercept, and logit/probit with 10,000 rows and
five predictors. Binary fits include the same separation linear-program
diagnostic in every adapter. One warmup and five measured fits are made for each
workload/engine; engine order rotates between repetitions. NumPy uses the same
fixed seed as the historical measurement below. Every repeated fit includes
validation, design construction, estimation, inference, provenance hashes and
`ResultBundle.model_dump(mode="json")`. Preparation, process/import startup,
pre-run garbage collection, JSON text encoding, disk work, HTTP/MCP and UI work
are outside timing. Peak RAM is not measured; design size is theoretical only.

The script checks coefficients, covariance, p-values, chart predictions and
sample positions against the preserved adapter before accepting a timing. It
also records metric differences. UUIDs, timestamps, backend provenance and the
condition-number diagnostic are not compared: native design diagnostics use
different scaling. Full method equivalence is tested separately:

```sh
uv run --extra torch pytest tests/test_engine_equivalence.py
```

Those tests cover repeated random heteroskedastic OLS data, all four covariance
types, intercept/no-intercept models, logit/probit observed information and
cluster scores, category/reference and missing-row handling, three-cluster
finite-sample correction, binary tails and nearly collinear designs. CUDA tests
are explicitly skipped when unavailable. Coefficient and covariance comparisons
use absolute and relative tolerances; intrinsically unstable individual
coefficients are not tested as though all their digits were identifiable.

Thread environment variables request one CPU thread, `threadpoolctl` limits
discoverable pools, and PyTorch intra/inter-op threads are explicitly set to one.
Apple Accelerate runtime thread counts may not be discoverable, so this is not a
claim of universally verified single-thread execution. The emitted JSON records
actual available backend metadata, raw timings, source SHA-256 hashes and whether
the source remained unchanged. A changing checkout or numerical mismatch aborts
the measurement. Ratios are former-adapter time divided by native-engine time:
values above one mean a faster native engine on that measured workload only.

## Matched native measurement — 30 September 2026, 15:11 UTC

This final run compared the preserved adapter and both native CPU engines on the
same Apple M3 Pro (arm64, 12 logical CPUs), macOS 26.6.2 / Darwin 25.6.0, Python
3.13.5, NumPy 2.5.3, pandas 2.3.3, SciPy 1.18.1, statsmodels 0.14.6 and PyTorch
2.14.0. NumPy and SciPy used Apple Accelerate. PyTorch reported one intra-op and
one inter-op thread; the discovered OpenMP pool reported one thread.
Accelerate's actual runtime thread count remained unobservable. **CUDA was
unavailable and was not measured.** These are CPU measurements, including the
PyTorch column.

| Workload | Rows | Predictors | Former adapter median (ms) | NumPy median (ms) | PyTorch CPU median (ms) | Adapter / NumPy | Adapter / PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OLS / HC3 | 10,000 | 5 | 19.848 | 11.652 | 13.434 | 1.70× | 1.48× |
| OLS / HC3 | 100,000 | 5 | 142.617 | 63.011 | 122.232 | 2.26× | 1.17× |
| OLS / HC3 | 10,000 | 50 | 307.473 | 157.953 | 99.143 | 1.95× | 3.10× |
| OLS / HC3 | 100,000 | 50 | 5070.411 | 1418.355 | 2576.765 | 3.57× | 1.97× |
| Logit / nonrobust | 10,000 | 5 | 93.119 | 89.724 | 68.591 | 1.04× | 1.36× |
| Probit / nonrobust | 10,000 | 5 | 167.167 | 93.116 | 132.636 | 1.80× | 1.26× |

The native engines had lower measured median times than the former adapter on
these workloads. PyTorch CPU was slower than NumPy on four of the six workloads
and faster on 10,000-row/50-predictor OLS and logit. It is not a universal
acceleration switch. The ratio includes changes to preparation and provenance in
the new implementation as well as estimator kernels, so it must not be described
as a matrix-kernel-only speed comparison.

The machine was not a dedicated performance host, and this run had substantial
variance and different absolute timings from earlier runs. Project test/build
work was paused during measurement; other desktop activity and thermal effects
were not controlled. The final 100,000-row/five-predictor OLS ranges were
**131.627–324.092 ms** for the adapter, **42.324–93.560 ms** for NumPy and
**58.618–262.438 ms** for PyTorch; for 50 predictors the corresponding ranges were
**3,095.305–5,338.697 ms**, **1,231.346–5,084.218 ms** and
**1,497.009–5,109.463 ms**. The roughly 3.8% NumPy logit median difference is small
relative to the observed variation and does not establish a meaningful speedup.

This is an exploratory median comparison, not evidence of a stable latency
improvement at every percentile. Five measurements cannot establish tail
latency, production throughput or performance on another machine. The historical
adapter timings below are from an earlier, unmatched session and should not be
substituted into these ratios. Earlier exploratory native runs were also variable;
the final table uses the recorded run after making the reference adapter and
results reproducible from a fresh checkout, rather than selecting a faster run.

Every comparison passed the declared numerical checks. Across these workloads,
largest absolute differences versus the former adapter were:

| Quantity | Largest absolute difference |
| --- | ---: |
| Coefficients | 1.654e-14 |
| Covariance elements | 5.746e-18 |
| p-values | 5.673e-14 |
| Confidence interval endpoints | 1.688e-14 |
| Chart predictions | 2.043e-14 |

Exact sample positions also matched. These fixtures do not establish universal
estimator correctness or Stata parity. The independent full-pipeline suite
passed **80 tests** on NumPy and PyTorch CPU; **40 CUDA cases were skipped** because
CUDA was unavailable.

The complete raw timings, warmups, preparation timings, output differences,
device availability and source hashes are in
[2026-09-30-native-cpu.json](../benchmarks/results/2026-09-30-native-cpu.json).
The reference implementation is tracked at
[analysis_statsmodels.py](../benchmarks/reference/analysis_statsmodels.py), used
only by the development benchmark and excluded from the production wheel.
Source files stayed unchanged during the final measurement. Selected SHA-256
hashes:

| Source | SHA-256 |
| --- | --- |
| Preserved statsmodels adapter | `c9349d292d3666672200bdc6574550c2ecd6a2ff07454cafcc0dd6531a391874` |
| Benchmark script | `51ef8d65f154f06c944ea9301ac626f8c7d5227798e39f75c4d3e440831fbd09` |
| Native analysis pipeline | `60ef78329e338748926290541a48f8c3ef9c4fcc38e976eb6db31194d64cdc81` |
| NumPy engine | `0ba61c206764a6b8a687d8d3afcf268ec01ddb627dcc985e36b49df7e97de7b9` |
| PyTorch engine | `051e2f334599b28bae600a058c84b2423bf3401d4cee6969fce48a221d3f0de6` |
| Dependency lock | `9d17f73434f49cb834c73b51ddf4a9381b9fa9410a0e5515d6e98c30d8bb9e6c` |

## Historical adapter baseline — 30 September 2026, 14:41 UTC

The former statsmodels-backed OLS/HC3 engine completed this local synthetic workload in a median of
**8.397 ms for 10,000 rows** and **41.792 ms for 100,000 rows**.
These are measurements of this checkout on one machine, not a speed guarantee,
an end-to-end application timing, or a comparison with Stata.

Measured on **30 September 2026 at 14:41 UTC (17:41 Europe/Istanbul)**. The command
and script hash in this historical section describe that earlier implementation;
the archived 0.2 benchmark later ran the matched comparison documented above.

## Reproduce

From the repository root:

```sh
uv sync --locked
uv run python scripts/benchmark.py
```

The script emits JSON with the environment, backend details, every timing, source
hashes and theoretical design-matrix size. It omits personal hostnames, usernames
and absolute installation paths. Redirect stdout to a file if you want to retain
another run. Source files and the dependency lock are checked for changes during
execution; a changing checkout is rejected.

This is a small benchmark, not a pass/fail performance test. The script uses the
same `openecon.analysis.fit` and `ModelSpec` as the application. It neither changes
the estimator nor substitutes a stripped-down computation.

## Workload and timing boundary

- Two independent sizes: 10,000 and 100,000 rows.
- Five numeric predictors plus an intercept; all inputs are float64.
- Fixed NumPy seed `20260930`; normal predictors and a continuous outcome
  with heteroskedastic noise. These are synthetic observations.
- OLS with HC3 covariance, `missing="raise"` and `alpha=0.05`.
- One warmup followed by three measured fits for each size.
- `time.perf_counter` measures ModelSpec validation, all input/design checks,
  model fitting, HC3 inference, coefficient/result construction and
  `ResultBundle.model_dump(mode="json")`.
- NumPy data generation and pandas DataFrame construction are timed separately.
  Import/process startup, pre-run garbage collection, JSON text encoding, disk
  persistence, HTTP/MCP transport, chart rendering and the UI are outside fit timing.
- Normal garbage collection remains enabled while fitting. Each fit's result is
  released after its elapsed time and basic sample/design checks are recorded.

| Rows | Median fit (ms) | Minimum (ms) | Maximum (ms) | Three measured fits (ms) | Warmup (ms) |
| --- | ---: | ---: | ---: | --- | ---: |
| 10,000 | 8.397 | 8.033 | 9.934 | 8.033, 8.397, 9.934 | 11.307 |
| 100,000 | 41.792 | 41.588 | 42.154 | 42.154, 41.588, 41.792 | 65.712 |

Preparation measurements below are single observations for each size, not medians:

| Rows | Synthetic generation (ms) | DataFrame construction (ms) | Theoretical dense design (MiB) |
| --- | ---: | ---: | ---: |
| 10,000 | 0.595 | 0.703 | 0.458 |
| 100,000 | 5.848 | 0.892 | 4.578 |

The dense design has six columns including the intercept. Its theoretical size is
`rows × 6 × 8 / 1,048,576` MiB. This excludes the source frame, response vector,
copies, residuals, leverage/inference workspaces, Python objects and other runtime
allocations. **Peak RAM was not measured**; the table is not a process-memory claim.

## Environment and thread-control limits

| Item | Recorded value |
| --- | --- |
| CPU / architecture | Apple M3 Pro / arm64 |
| Logical CPUs | 12 |
| Operating system | macOS 26.6.2; Darwin 25.6.0 |
| Python | 3.13.5 |
| NumPy / pandas | 2.5.3 / 2.3.3 |
| SciPy / statsmodels | 1.18.1 / 0.14.6 |
| threadpoolctl | 3.7.0 |
| NumPy / SciPy BLAS | Apple Accelerate; backend version unreported |

The script requests one thread through `threadpool_limits(limits=1)`. Before
numerical imports it sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `BLIS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to `1`.

On this machine, `threadpool_info()` returned an empty list because the installed
Accelerate backend was not discoverable by that mechanism. The environment
request was applied, but its actual BLAS thread count was not observable.
**These results must not be described as a verified single-thread benchmark.**
The script records observed pools on environments where the backend is discoverable.

## Measured source snapshot

The following SHA-256 values identify the measured implementation and locked
dependencies. The script verified that these files remained unchanged during the
run. If they change, this table remains a historical result until a new
measurement is recorded.

| File | SHA-256 |
| --- | --- |
| `scripts/benchmark.py` | `83a98acee3ad53cf92f814b77e67f458a7eb173f80b26c2b017db89d6611a0ee` |
| `src/openecon/analysis.py` | `c9349d292d3666672200bdc6574550c2ecd6a2ff07454cafcc0dd6531a391874` |
| `src/openecon/models.py` | `3e131b38eb75b9329fb2726fee96cd5e01b49b8d6b7cea09206542e7b5279bd9` |
| `uv.lock` | `1b1481ebe129c41d77892a1daf399945bdf739481407518ff1bddc259ee98562` |

## What the result establishes

This run provides a repeatable baseline for a narrow numeric OLS/HC3 workload
within the alpha's current data-size range. It does not cover category expansion,
logit/probit, difficult conditioning, cluster covariance, panel estimators, file
loading or larger-than-memory data. Three warm measurements do not establish
tail latency or the effects of background work, memory pressure and thermal
conditions. Compare future changes using the same workload, environment and
timing boundary, and keep correctness/inference checks separate from speed.
