# Large local datasets

<!-- BEGIN source-generated capability scope -->
Current source: **157 registered fit names**, **92 Dataset fit routes**, **74 common saved predict/margins adapters**. The [generated method/option inventory](capabilities.md) states conditions, exclusions and devices.

Fit routes, common prediction adapters and family-specific helpers/forecasts have separate contracts. Source implementation does not establish independent scientific validation, installed-package verification or public shipment for a method/option. Those require their own dated, source-pinned evidence; historical measurements retain their original scope.
<!-- END source-generated capability scope -->

Current numerical execution uses OpenEconometrics's own float64 PyTorch kernels.
SciPy is absent from runtime dependencies and from the packaged application;
it remains an optional development dependency for independent test oracles.
`oe.capabilities()["streaming"]` is the authoritative current catalogue of
implemented Dataset adapters, algorithms and option conditions.

Large CSV/Parquet inputs opened with `oe.read()` automatically return a lazy
Dataset once the eager threshold is exceeded. The 100,000-row/32-MiB values
are thresholds for choosing a reader, not limits on these formats. Local
imports stream original files into owned snapshots. Large desktop files stay
local; project scripts and small supported cloud files still synchronize.
XLSX and DTA retain separate eager-format limits.

The Dataset subset includes the following source replay families. Consult the
generated inventory for current per-estimator options; this overview is not an
independent scientific, installed or large-physical-data receipt for every option:

| Family | Implemented native replay routes |
|---|---|
| Linear and fixed effects | OLS, areg, reghdfe, PPMLHDFE, cnsreg; xtreg FE/BE/RE/FD/pooled/MLE, native FE/pooled Driscoll-Kraay; xtfmb |
| Systems and moments | SUR, mvreg, reg3, GMM with all native HAC kernels; IV/absorbed IV 2SLS/LIML/GMM, centered/iterated GMM, panel FE/BE/FD IV, G2SLS/EC2SLS RE IV |
| Panel covariance and mixed models | xtgls/xtpcse; independent/exchangeable/AR1/stationary/nonstationary/unstructured GEE and XT PA; one/two nested Gaussian mixed levels with random slopes, optional lowest-level intercept and all native covariance structures; integrated GLMM and conditional panel logit/Poisson |
| Nonlinear and quantiles | nl, rreg, qreg, bsqreg, sqreg, iqreg |
| Generalized and discrete likelihoods | GLM ML/IRLS, Poisson, negative binomial, cloglog, fracreg, beta regression; ordered/multinomial logit/probit, hetprobit, biprobit |
| Limited and selection likelihoods | Tobit including global sample-extreme limits, interval/truncated regression; Heckman/IV probit/IV Tobit ML and native two-step methods; Heckprobit ML |
| Counts and frontier | Censored/truncated Poisson/NB, ZIP/ZINB, generalized NB, hurdle/churdle, frontier |
| Survival | streg with native subject IDs/interval-overlap checks and subject covariance; Cox Breslow/Efron/exactp and time-varying coefficients |
| Treatment effects | RA/IPW/IPWRA/AIPW, exact NN/propensity-score matching with Abadie-Imbens inference, DID, event study, group-time CSDID, RD bandwidth/bias inference |
| Time series | VAR, VEC/Johansen, ARDL, NARDL, Prais, ARIMA/SARIMA, ARCH/GARCH variants, UCM, Markov switching, threshold |
| Dynamic panel | Anderson-Hsiao; difference/system GMM, FD/FOD, one/two-step and Windmeijer inference |

The Dataset routes cover the declared estimation methods and option families
above;
they retain the same scientific restrictions as the native API. For example,
REML mixed inference is nonrobust, two-step selection/IV likelihoods use their
native nonrobust covariance, EC2SLS applies to RE panel IV, and Driscoll-Kraay
applies to FE/pooled panel regression. Fixed-effects panel probit and arbitrary
normal censoring-bound columns are not implemented in the native API.
Cox exactp preserves the unweighted/nonrobust/no-TVC restrictions. Its symmetric
recursion and TVC interval splitting have explicit bounds on live state and
actual computational work; these options are substantially more expensive than
an ordinary Breslow/Efron sweep.
The catalogue exposes conditions. Invalid options and excessive structural
workspaces fail explicitly and never collect the whole source into memory.
Model-name coverage and complete Stata option parity are distinct contracts.

CUDA float64 and checked Metal preconditioning are available for the native
QR factor operations that use the execution context, including replay rank
checks, OLS and joint system factors. Automatic helpers share the caller's
execution trace; the recorded devices describe actual operations. Metal
float32 supplies a preconditioner only: original CPU float64 observations
refine and certify its factor. Likelihood steps, disk group work and inference
remain CPU float64. Unsafe accelerations fall back to original CPU QR.
CUDA hardware was not available on the verification Mac.

```python
import openecon as oe

df = oe.scan("/path/to/data.parquet")  # also CSV or a Parquet directory
model = oe.logit(
    data=df, y="employed", x=["education", "experience", "region"],
    categorical=["region"], covariance="cluster", cluster="firm",
)
display(model)
model.to_latex("regression.tex")
```

`scan()` returns a replayable Dataset without loading the whole table. Readers
project only required model columns. `Dataset.from_frame` and
`Dataset.from_batches(factory, columns, row_count=...)` support integrations;
factories must return a fresh bounded iterator on every pass. Declared row counts
are verified by scanning actual observations. Metadata alone is not a benchmark.

## Algorithms and bounded state

OLS retains three passes: scaled merged moments, balanced augmented TSQR, and
residual/covariance accumulation. Coefficients are not solved from normal equations.
Logit and probit accumulate exact likelihood, gradient and observed information
at each Newton step, replaying the source for line search. Rank checks use TSQR.
Complete/quasi-complete separation checks use bounded native Torch primal-dual
constraint generation with verified dual bounds and full replayed witnesses.
Statistical fitting and inference use OpenEconometrics's own PyTorch implementation.

Categorical treatment coding matches the dense API, including declared pandas
category order and omitted first levels. Discovery adds one pass; category
metadata is limited to 2 MiB and expanded models to 384 parameters. Unsupported
ambiguous object-valued temporal cluster labels fail explicitly; ordinary typed
pandas datetime/timedelta columns are supported.

Cluster labels are encoded only on the final covariance pass. Earlier passes
still hash their original values and include their missingness in sample selection.
Cluster scores use compensated reductions, a fixed 4 MiB cache and a SQLite
spill database with a 2 MiB page cache. Group count can exceed the cache; disk
storage grows with group count and model width. Success and ordinary failures
remove the owned database. The desktop supervisor also removes its own nonce
scratch directory after Stop, timeout, reset, worker crash or exit.

The planner budgets 128 MiB for numerical working arrays, reduced factors and
cluster buffers. Batches shrink with model width, up to 65,536 rows. This is not
an OS-enforced RSS quota: Python, pandas, Arrow, reader buffers and allocator
caches use additional memory. Oversized CSV cells or Parquet row groups can
allocate reader buffers before batch checks; bounded row groups are preferable.
Source manifests are bounded to 100,000 files/10,000 columns and source blocks to
256 MiB/one million rows. These are metadata/per-block bounds, not total-row limits.

Dynamic-panel GMM processes one complete panel at a time and checks its values,
instrument block and time grid before fetching it. A single excessively long
panel can therefore be refused even when the raw reader has tiny batches.
Conditional logit similarly plans its complete-group workspace. Correlated
cross-panel GLS/PCSE requires a planned matrix quadratic in the number of panels.
These structural limits are separate from total observation counts.

General Gaussian mixed fitting stores low-level joint QR factors on disk and
merges whitened factors into top-level sufficient geometry; it does not collect
a long group. Patterned GEE correlations require a planned matrix quadratic in
the longest retained panel, even with tiny source batches. Ordered AR1 uses
bounded neighbour state. Matching uses disk coordinate prefixes and range
usage markers for one-dimensional searches, preserving all ties without
expanding matched edges in memory. Multidimensional matching and exact
quadratic-spectral HAC replay bounded candidate/pair tiles and retain explicit
computational-work budgets; they do not truncate matches or infinite-support
kernels to claim faster execution.

No replay estimator retains a full design/Q/observation covariance matrix or
full prediction/residual/position vector in RAM. UCM components and Markov
probabilities can produce complete disk-backed Dataset outputs. Ordinary model
results sample at most 400 observations with
original physical positions. Incremental projected-value and retained-position
hashes detect changes between passes, including changes with identical row counts.
Missing='drop' reports excluded rows; infinity and invalid outcomes remain errors.

## Practical limits

Binary likelihood currently runs on CPU. Binary likelihood and separation
checks can require many more source passes than OLS; memory safety does not imply
identical run time. Very wide models still require quadratic parameter storage.
Large category dictionaries must fit the declared metadata/parameter budgets.
Cluster inference requires writable temporary storage and at least two groups.
Singular designs, separation and nonconvergence return explicit errors.

Local desktop computation has no automatic 60/120-second deadline; an explicit
local deadline is optional and Stop remains available. Cloud computation,
permissions and cloud transfer limits are unchanged. `oe.read()` returns a small
DataFrame or a large CSV/Parquet Dataset automatically; `oe.scan()` always selects
the local bounded source path and never uploads data.

Completion time depends on repeated disk reads, decoding, model width and
conditioning. The practical target is large local datasets within the computer's
disk capacity, rather than a specified extreme row count. Current Dataset
coverage and conditions are listed above and in capabilities; full Stata parity
is a separate contract.

## Current verification

The installed macOS 0.3.33 package is documented in the
release evidence (internal evidence excluded from this public snapshot): all 356 owned frozen-code
fingerprints match the approved source; 30 actual Dataset fits, including 22
new option cases, match forced resident native references. Saved JSON and
LaTeX reopen in a new bundled worker, checked Metal factors match CPU float64,
and the reopened existing project's protected files retain their original
hashes. Cross-family source checks passed 1,133 tests with nine intended native
option-domain skips; 445 frontend and 27 native tests passed. These scoped
checks do not claim a fresh whole-repository suite, all option combinations or
new all-model throughput benchmarks. The
0.3.32 record (internal evidence excluded from this public snapshot) remains historical evidence.

Independent and dense/replay tests compare complete coefficient/covariance,
likelihood, model-test and forecasting contracts. Owned snapshots and numerical
blocks are tested separately: tiny raw-source batches alone do not prove
cross-block lag/risk-set logic after a source has been sorted. Current tests
force tiny sorted replay blocks as well. Full-sample restricted likelihood
references, pseudo-R² and family diagnostics replay the complete sample; an
unavailable secondary fit is recorded explicitly with a Wald fallback.

A fresh physical one-million-row Parquet probe with ten predictors and 100,003
dispersed fixed-effect groups took 35.999 seconds for areg (peak process RSS
470,663,168 bytes) and 11.703 seconds for constrained regression (peak RSS
449,003,520 bytes). Source reads, hashing and disk work are included in fit
time; setup/import time is recorded separately. These are individual runs on
one Mac, rather than a universal throughput guarantee.

Fresh physical ten-million-row Parquet measurements used four predictors and
65,536-row source groups, with bounded float64 generation and a 480,555,238-byte
original file. OLS/HC3 completed in 18.550 seconds (peak process RSS 604,372,992
bytes); Poisson/robust took 235.215 seconds (peak RSS 557,236,224 bytes). Fit time
includes decoding, integrity checks and every estimation/diagnostic pass;
imports and source metadata setup are recorded separately. The source hash,
actual counts, coefficient recovery, buffers and timings are retained in the
verification records. These single runs occurred under concurrent development
load on one Mac; they are not all-model latency guarantees.

The same actual ten-million-row source produced a 40-bin histogram in 0.682
seconds after vectorizing numeric chart blocks (cold total 2.096 seconds; peak
process RSS 378,912,768 bytes). Counts sum to all 10,000,000 observations. The
previous per-value Python reduction took 40.246 seconds in an earlier concurrent
run; these are single-run observations, not a controlled speedup benchmark.

Histograms count every finite source value in bounded passes. Scatter plots
retain at most 2,000 uniformly ranked finite pairs and record full extents;
categorical sum/mean/count reductions use at most 1,000 groups/10,000 values.
Unaggregated ordered line data has an explicit 10,000-row display guard so that
gaps and trajectories are not silently changed. Charts do not send millions of
Python values to the renderer.

Resident DataFrame and sized column-mapping inputs now use those existing
numeric histogram/scatter reductions, projecting only requested columns in
65,536-row blocks. Rows are sliced before DataFrame column selection; typed
columns retain positional order and dtype, while Python sequences retain their
original scalars. No N-sized Python point dictionaries or converted-value lists
are created before scatter sampling or histogram counting. Sized resident line
inputs are checked against 10,000 rows before values are copied; smaller lines
retain their irregular x spacing and gaps. Unsized histogram/scatter iterables
and custom `to_dict` inputs retain eager compatibility and are outside this
bounded resident-input claim.

Large resident inputs retain counts, pass information and full scatter extents
in `config.processing`, including after publication validation and saved-result
readback. Log axes validate the full source, including unsampled values. Small
resident inputs with at most 2,000 source rows keep their previous configuration
format. Existing `Dataset` reductions are unchanged.

The MARKET-99 receipts (internal evidence excluded from this public snapshot) measure
fresh owned physical Parquet sources loaded into resident DataFrames or NumPy
column mappings, with a new process for each one-/five-million-row case. At
five million rows, histogram took 0.415–0.421 seconds and scatter 0.560–0.606
seconds; additional traced Python peaks were below 8 MB and scatter JSON was
86,491 bytes. These instrumented warm-operation measurements exclude input
loading and dependency initialization; sampled RSS deltas, traced Python peaks
and lifetime RSS are recorded separately. They occurred during a concurrent
runtime build on one Mac and do not promise total process memory or latency.
The [runnable example](examples/resident_charts.py) separately verifies missing
counts, sampling, large-line refusal, gaps and persisted outputs in source and
freshly frozen desktop workers. The frozen check does not replace or release
the installed application.

## Historical OpenEcon 0.3.4 evidence

The suite counts, estimator table and release paths below are historical;
they are not the current build or a verification of this source revision.


Tests compare streamed and dense coefficients, covariance, inference, likelihood,
fit metrics, categorical metadata and positional samples for every supported
estimator/covariance combination. Independent oracles cover cluster CR1,
separation, precision, tiny batches, units, offsets and cancellation. File tests
cover CSV/Parquet projection, missing observations, changed sources and bounded
output. Spill tests cover cache eviction, many groups, storage failures and cleanup.

`benchmarks/stream_all_probe.py` measures real Parquet workloads for all eight
estimator/covariance combinations, with categorical predictors and 50,000 groups.
It records fixture preparation separately from complete fit time, actual rows,
source passes, peak process RSS, output size and disk-spill diagnostics. Each case
runs once in a fresh process on one busy Mac, with OS caches retained. These
synthetic fixtures are not Stata-equivalence or production-latency guarantees.

The final suite passed **2,152 Python tests** with three macOS platform skips.
The historical benchmark record (internal evidence excluded from this public snapshot) records all eight
actual one-million-row Parquet cases on an Apple M3 Pro with 18 GiB RAM.

| Estimator / covariance | Complete fit | Peak process RSS | Source passes |
|---|---:|---:|---:|
| ols / nonrobust | 2.637 s | 365.36 MiB | 4 |
| ols / HC1 | 2.550 s | 408.11 MiB | 4 |
| ols / HC3 | 2.597 s | 384.39 MiB | 4 |
| ols / cluster | 20.033 s | 425.02 MiB | 4 |
| logit / nonrobust | 5.352 s | 373.27 MiB | 18 |
| logit / cluster | 23.154 s | 352.44 MiB | 18 |
| probit / nonrobust | 5.309 s | 408.98 MiB | 18 |
| probit / cluster | 23.681 s | 355.89 MiB | 18 |

These figures include imports and fixture preparation in peak RSS, while fit
time includes all estimation passes. Category discovery adds one pass. Cluster
cases contain 50,000 observed groups and verify actual disk spill and cleanup.

Historical numeric OLS measurements in 0.3.2 fitted 10 million Parquet rows in
4.675 seconds and 100 million replayed generated rows in 60.563 seconds. The latter
includes generation on all three passes and is not an on-disk 100-million-row
measurement. See the historical measurement (internal evidence excluded from this public snapshot)
and historical installer verification (internal evidence excluded from this public snapshot).

The historical Apple Silicon installer was
`desktop/build/releases/0.3.4/target/release/bundle/dmg/OpenEcon_0.3.4_aarch64.dmg`.
Its verified identity and packaged-runtime checks are recorded in
the release evidence (internal evidence excluded from this public snapshot). The running user's application
and data are preserved; installing/reopening the new version activates these changes.

Algorithm reference: [Demmel et al., communication-optimal QR](https://arxiv.org/abs/0808.2664).
Reader reference: [Apache Arrow Scanner](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html).
