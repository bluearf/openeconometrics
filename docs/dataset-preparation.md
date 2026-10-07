# Local source readers and replayable preparation

Large XLSX/DTA sources now convert in a disposable parser to an owned Parquet
snapshot. XLSX uses two streamed passes (global dtype inference, then 8192-row
groups); heterogeneous non-lossless cell types are refused. DTA reads metadata
and bounded row ranges, preserving column/value labels, dates and a-z extended
missing tags. Numeric/date model values remain nullable; `iter_missing_codes`
replays tags separately at original indices. Hidden tag fields are never predictors.
`Dataset.close()` or its context manager releases the conversion snapshot.
Workspace import retains an exact original alongside the converted source and
records its fingerprint/counts; it never modifies the supplied file.

CSV fields/records are admitted before parsing (1/4 MiB). Parquet admits a 2 MiB
physical footer, 64 MiB total directory footers, bounded Thrift containers and
256 MiB uncompressed row groups. XLSX shared metadata/styles are bounded to
32 MiB. Disposable parser budgets are 512 MiB additional allocation, 1536 MiB
total RSS, 64 MiB transfer blocks, 120 seconds and 8 GiB conversion scratch.
Parent batches retain the separate 256 MiB guard. Linux also applies an address
space limit; macOS supervises RSS every 10 ms and checks true child peak RSS.
These are resource guards, not a security sandbox or a universal no-OOM proof.
Timeout, error and early iterator close stop the owned parser and remove scratch;
console Stop reclaims its owned process group/root. Physical full-source counts,
fingerprints, peak RSS and cleanup are recorded in the MARKET-98 evidence JSON.

```python
source = oe.scan("measurements.parquet")
prepared = source.project(["id", "x", "y"]).filter(lambda b: b["x"] > 0)
prepared = prepared.map(
    lambda b: b.astype({"id": "Int64", "x": "Float64", "y": "Float64"}),
    schema={"id": "Int64", "x": "Float64", "y": "Float64"},
)
joined = prepared.join(lookup, on="id", validate="m:1")
fit = oe.ols(data=joined, y="y", x=["x"], covariance="HC3")
```

Preparation reads fixed 8192-row blocks. A complete pass checks input/output row
counts and typed value/index/schema/missing-tag digests against subsequent
passes. Callable code changes, nondeterminism, one-shot sources and schema drift
produce controlled errors. A partial `head` is not a completed replay proof.
Map must return exact declared dtypes/categories and unchanged original row
indices; casts/derived columns are explicit. Filter accepts an aligned boolean
Series with missing decisions dropped/kept/refused according to policy.

Join uses an owned SQLite file with indexed typed keys. Defaults are `m:1`
validation and nonmatching nulls; `1:1`, `1:m`, `m:m` and `nulls="equal"` are
explicit alternatives. Boolean/string identities stay distinct from numeric
keys. Both key schemas/categories must agree. Pair counts and unmatched counts
are admitted before any output; index tuples retain both original row identities.
Output order is left order, matching right order, then unmatched right order.
Outer gaps promote integer/bool to nullable types and preserve categorical levels.

Long requires losslessly equal value-column dtypes and uses row-major order,
retaining `(original index, variable)` identities. Wide requires 1–1000 explicit
levels; duplicate cells error by default (`first`/`last` explicit), undeclared
levels error (`drop` explicit). There is no implicit aggregation. Wide output
index retains group keys and selected original indices for every level, including
missing cells. Missing tags follow their original cells; imputed nonmissing values
lose their tags. Tagged blocks require unique original indices to avoid ambiguity.
Empty disk/factory inputs whose pandas dtype cannot be inferred require a declared
typed map. These differences are also returned by `oe.capabilities()`.

Defaults: 64 MiB owned buffers per stage, at most 16 MiB SQLite cache, 1 GiB disk
per stage (hard SQLite page limit), 10 million output rows, 100 million checked
work units and 16 nested stages. Oversized keys/rows/blocks and expansion are
refused. These bounds do not include arbitrary caller function allocations or
total process RSS. Joins/wide remove scratch on success, error and iterator close;
console Stop cleans the owning root. The MARKET-108 evidence JSON checks an actual
100,001-row Parquet source, full join/wide replay, original indices/categories,
true parent peak RSS, disk bytes and independent scalar HC3/OLS equivalence.
