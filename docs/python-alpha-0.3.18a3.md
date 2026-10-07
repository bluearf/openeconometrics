# Python SDK alpha 0.3.18a3

This Python-only follow-up derives from SDK 0.3.18a2 public source
`0d8a8ea3ef12ddf8bdadfc8c64135052741f29c9`, merged as
`ff50b3f90634282208077cfe62e53919ebb09e82`. The underlying reviewed public
snapshot remains `7cb7517bf8bacf91f92c85f8df9a2904e76e8974`, originally exported
from private source `6cd9079a330b03a4f1a568e28aca04a23a8d897b`.

The SDK version is `0.3.18a3`, with the exact dependency
`openecon-charts==0.3.0a1`. Charts package files are reused byte for byte.
The existing Mac release `v0.3.43-alpha.1` still embeds SDK `0.3.18a1`; this
Python patch does not rebuild that application or establish acceptance of the
newer private SDK or desktop 0.3.44.

## Scoped repair

The parser now separates its startup handshake from active source parsing.
Before the child reports ready, a Linux child PID can briefly expose copied
parent memory before executing the disposable parser. That transient sample
must not become the new worker's retained peak. The ready handshake uses the
worker's own high-water measurement, establishes the post-exec current baseline,
and validates the existing total and growth budgets before parsing starts.

Startup still enforces its deadline, EOF/error handling and worker liveness.
Once ready, both supervisor current-RSS samples and the worker's reported
high-water RSS enforce the unchanged 1536 MiB total and 512 MiB growth budgets.
The Linux virtual-memory limit, 64 MiB batch/transfer limit, physical source
admission, owned scratch cleanup and parent-death termination remain intact.
Missing supervision and genuine sampled/transient allocation excess still fail.

Only `src/openecon/source_readers.py` and its scoped `tests/test_dataset.py`
regressions are imported from private commit
`c758b7cdd26e6b6548019c945735f17092730f77`. The earlier own-process Linux high-water and
streaming OLS repairs remain as documented in [the a2 scope](python-alpha-0.3.18a2.md).
SDK version, capability fingerprints, public manifest and package verification
metadata accompany this repair. No newer estimator, chart, UI or desktop
implementation is imported.

## Provenance and acceptance

`SOURCE-MANIFEST.json` keeps the original private/public bases, the prior public
SDK source, each scoped private patch and current per-file hashes. Its
`source_commit: null` explicitly identifies a derivative rather than claiming
that one private HEAD equals these public bytes. Public source/build commit and
fixed distribution hashes are recorded separately in release provenance.

The existing Linux `Verify Python SDK patch` workflow checks Python 3.11 and
3.13, strict metadata, isolated wheel/sdist installations and bounded parser/OLS
regressions. Actual hosted results, macOS installed-package receipts, public
Release access and PyPI registry installation each need successful readback.
The earlier a2 GitHub asset bytes remain immutable; the a3 package has a new
version and must receive fresh package/publication proof.

These checks do not establish the entire scientific suite, full Stata/vendor
parity, physical CUDA, Windows, Developer ID signing or notarization acceptance.
