# Bounded shared datasets (MARKET-109)

This document describes the source implementation. Production cloud deployment
and an installed Mac end-to-end transfer run remain separate acceptance gates;
isolated storage, HTTP, Rust and browser tests do not satisfy those gates.

## Workflow

The native file chooser keeps computation on the user's computer. CSV and
Parquet files above the legacy 24 MiB upload limit use the private transfer
protocol. A local journal pins the selected name, physical byte count and
SHA-256 digest. It contains no Firebase token. Connection failures preserve
the local data and journal; the file sidebar exposes explicit resume/cancel
controls. Reopening or reconnecting discovers pending work without starting
an upload automatically. Other failures, including permission denials and
changed source bytes, do not become offline-success fallbacks.

The service reserves the filename and project budget before receiving parts.
Each part has a SHA-256 digest, exact byte count and stable immutable object key.
An uncertain storage/database acknowledgement is resolved against that key;
retrying does not create more copies of an unacknowledged part. Only a complete
file whose individual parts and whole-file SHA-256 have passed verification is
atomically published in the project catalogue. Partial transfers are not ready
Datasets and have no member-visible file manifest.

A member downloads a verified manifest and bounded parts. Private object keys,
storage capabilities and bearer tokens are not returned in the manifest or
sent to Python. Part and complete-file digests are checked before an assembled
file is installed atomically. Interrupted downloads retain verified chunks for
explicit retry; existing user files are preserved. A successful upload can
seed the same verified local cache rather than downloading its source again.
The Python runtime then imports the owned local file using its existing bounded
CSV/Parquet Dataset route. Run requests stay local: the cloud execution route
explicitly rejects projects containing chunked datasets.

## Authorization and cancellation

Every endpoint authenticates the current Firebase identity. Transfer creation,
parts and completion need an editor or owner. Journals additionally belong to
the initiating UID or the project owner; another editor cannot take over a
pending journal. Ready manifests and downloads allow current viewers, editors
and owners. Outsiders and revoked memberships are refused. Membership is
rechecked around reads, between whole-file verification steps, before part
buffering, and before publication or cleanup writes.

Cancellation is saved before cleanup. Known immutable generations are deleted
with generation pins. If an upload or finalization can still be in flight,
`CLEANUP_PENDING` preserves the cancellation and reservation until a retry can
finish cleanup. Upload leases last at most five minutes; finalization leases
last at most fifteen minutes and verification itself has a ten-minute deadline.
An already-published transfer returns `TRANSFER_COMPLETE`; this refusal never
deletes the ready dataset. A client must distinguish these responses from a
successful cancellation.

## Budgets and evidence boundaries

| Resource | Bound |
|---|---:|
| Shared CSV/Parquet file | 2 GiB |
| Project files, including pending reservations | 20 / 8 GiB |
| Pending upload journals per project | 4 |
| HTTP part / object read | 4 MiB |
| Parts per file | 512 |
| Manifest | 256 KiB |
| Transfer lifetime before expiry | 2 days |

The 8 GiB limit includes legacy small files; those retain their separate 64 MiB
aggregate limit. Expired transfers reserve their budget until cleanup, so expiry
cannot bypass storage limits. The limits bound selected bytes and owned transfer
buffers, not total process RSS, arbitrary user scripts or unlimited bandwidth.
Successful finalization rereads the uploaded bytes once to certify the complete
digest. Network retries and membership/manifest requests incur additional
operations. A download may need verified chunks plus an assembled copy and an
existing destination; allow up to roughly three file sizes of local disk during
replacement. Cancelled partial data stays private until cleanup completes.

`tests/test_team_transfer.py` exercises a physical 25,600,004-byte CSV with
6,400,000 observations: stop after two parts, inspect acknowledgements, resume,
download as a second member, compare whole-file SHA-256, and fit native Torch
Dataset OLS locally. Its storage adapter is an isolated physical-disk double,
not Google Cloud Storage. The native Rust tests use real loopback HTTP with a
physical 25 MiB plus 17-byte source, bounded part transport, lost acknowledgements,
source mutation, denial, interruption, cache integrity and cancellation. They
do not establish live picker, production Firebase/GCS or installed Mac behavior.

Live acceptance requires a disposable team project, two designated members and
physical CSV/Parquet files. Record the actual application/build identity and
source hashes; interrupt and resume the upload and download, confirm role
revocation/cancellation, then perform the local fit with the downloaded source.
Read back the catalogue and storage generations and report actual network,
storage and disk measurements. MARKET-109 remains open until those distinct
gates have passed.
