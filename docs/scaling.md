# Current scaling evidence — 2 October 2026

This page archives the dense2October audit. Current code automatically opens
large CSV/Parquet sources as Dataset, adds native replay kernels beyond the core
three, checks named model workspaces before allocation, and streams supported
chart reductions. Query `oe.capabilities()` for the exact current routes and
limits. SciPy has been removed from the runtime separation solver. Measurements
below must not be attributed to the current implementation.

OpenEcon 0.3.2 now adds [bounded numeric OLS and new measured workloads](streaming.md).
The dense 0.3.0 measurements below remain historical evidence for that path;
they are not measurements of the new streaming implementation.

**The current workbench is not validated for large-data production use.**
The native core can fit the measured in-memory workloads quickly, but normal
file imports remain limited to 100,000 rows and 100 columns. Web uploads are
limited to 24 MiB per file; `oe.read` separately permits files up to 32 MiB.
Those eager paths do not implement streaming or absorbed fixed effects. Use
`oe.scan()` for the new numeric OLS streaming path. PyTorch alone does not
establish production large-data capacity.

## Local measurements of the current 0.3 core

Measured on an Apple M3 Pro, macOS/Darwin 25.6.0, 12 logical CPUs and 18 GiB
physical RAM, Python 3.13.5, PyTorch 2.14.0, pandas 2.3.3. Each case uses a fresh
process, one warmup and three timed repetitions on deterministic synthetic,
well-conditioned numeric data. PyTorch intra-op/inter-op threads and the
discovered OpenMP pool were one. Apple Accelerate runtime threads may not be
observable. GPU was not used.

| Estimator / covariance | Rows | Numeric predictors, excluding intercept | Median fit + result conversion | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| OLS / HC3 | 10,000 | 5 | 5.30 ms | 266.9 MiB |
| OLS / HC3 | 100,000 | 5 | 17.73 ms | 311.1 MiB |
| OLS / HC3 | 1,000,000 | 5 | 153.40 ms | 740.9 MiB |
| OLS / HC3 | 100,000 | 50 | 174.35 ms | 638.2 MiB |
| OLS / cluster, 1,000 groups | 100,000 | 5 | 19.74 ms | 313.5 MiB |
| Logit / nonrobust | 100,000 | 5 | 317.04 ms | 551.6 MiB |

The public API, validation, design construction, estimation, inference,
provenance hashing and `model_dump(mode="json")` are timed. Binary timing also
includes the separation LP. Fixture creation, imports, garbage collection,
JSON text encoding, file loading, transfers, cloud startup, broker/sandbox and
UI are excluded. Peak RSS is the entire process high-water mark after fixture,
warmup, repeats and JSON encoding, including imported libraries; it is not a
kernel-only allocation or a Cloud Run container memory measurement.

The million-row case bypasses file-import limits by passing an already-created
DataFrame to the direct Python API. It does **not** demonstrate that a
million-row upload works. All cases bypass file-size constraints. Full result
JSON for the million-row case is 6.94 MB because it retains all sample positions;
console display strips those positions and covariance and limits plot samples.

Each case checked observation/sample counts, finite coefficients and positive
standard errors, and recovery of known simulated coefficients within 0.1.
These are sanity checks, not an independent numerical equivalence oracle or
proof of Stata parity. The existing `tests/test_analysis.py` suite also passed
during this audit. Three timings on one desktop do not establish tail latency,
throughput, hardware-independent speed, or behavior on difficult data.

Reproduce from the repository root:

```sh
.venv/bin/python benchmarks/scale_probe.py
```

The default output is `artifacts/verification/scale-local-current.json` with
raw repetitions, source hashes, measured memory, timing boundaries and package
versions. Source hashes were unchanged across the recorded run. This script
does not modify app configuration, cloud resources, or user datasets.

## Capacity limits and remaining work

The dense float64 design is capped at 256 MiB. That cap covers only `N*K*8`,
not total process memory: selected data columns, normalized design, reduced QR,
robust scores, Python sample positions and small parameter matrices also
consume memory. There is no N-by-N hat matrix. Fixed-column OLS scales roughly
linearly with row count, but widening the model adds QR and K-by-K covariance
work. Many category levels generate dense dummy columns rather than absorbed
fixed effects. Binary separation uses a full-data HiGHS linear program without
its own explicit time limit; well-conditioned numeric measurements do not
characterize its difficult cases.

An actual 100,001-row, 588,903-byte CSV was rejected by `oe.read` with
`DATA_LIMIT`; see `artifacts/verification/scale-data-limits.json`.
Removing these limits would not implement larger-data support.

The deployment contract is 2 CPU / 4 GiB per compute instance, one concurrent
request per instance and at most four compute instances. Python execution
defaults to 60 seconds and is capped at 120. Every run downloads all project
input files again, up to 64 MiB. Admission guards allow one active run per
project and four globally; they are not an expanding compute queue. Current
resource settings could not be refreshed because cloud credentials require
renewal. These resource figures are the source-enforced deployment contract,
not a new live configuration measurement.

Ordinary public GETs to `/` and `/api/auth/config` returned 200 during the audit.
The saved production workflow evidence uses only 480 observations, with
analysis and downloads taking 13 seconds. Neither establishes large-data
cloud capacity. See `artifacts/verification/scale-cloud-audit.json` for evidence
and verification gaps.

Before promising large-data production support, implement bounded chunked
data loading and estimation, a total working-memory budget, fixed-effect
absorption, and durable queued compute with suitable resource allocation.
Then measure actual supported model families and difficult category/missing/
cluster cases on production-equivalent resources, including file I/O, memory,
concurrency, cold starts and user-visible completion. Increase import limits
only with evidence that the full path fits those budgets.

The former 0.2 measurements in [performance.md](performance.md) remain an
archive and are not a current comparison or speedup claim.
