# Python SDK 0.3.18a3 and charts 0.3.0a2 alpha

This Python-only follow-up derives from SDK 0.3.18a2 public source
`0d8a8ea3ef12ddf8bdadfc8c64135052741f29c9`, merged as
`ff50b3f90634282208077cfe62e53919ebb09e82`. The underlying reviewed public
snapshot remains `7cb7517bf8bacf91f92c85f8df9a2904e76e8974`, originally exported
from private source `6cd9079a330b03a4f1a568e28aca04a23a8d897b`.

The SDK version is `0.3.18a3`, with the exact dependency
`openecon-charts==0.3.0a2`. The charts patch repairs only disposable browser
selection and cleanup; the previous charts a1 distribution bytes remain immutable.
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
metadata accompany this repair. No newer estimator, renderer, UI or desktop
feature is imported.

## Scoped charts exporter repair

Charts `0.3.0a2` imports only the two-path browser patches
`efc9df77bca551a87403a2f867ca715ea6b86816` and
`a04b0466a94a7c1ea0b3cbfc7debc8cef3c242ab`, each restricted to
`packages/openecon-charts/src/openecon_charts/network_export.py` and its scoped
export tests. `OPENECON_BROWSER_EXECUTABLE` selects an installed real browser;
an explicit call argument retains precedence and an invalid configured path
fails without fallback. This avoids an Ubuntu Chromium launcher wrapper.

A disposable POSIX browser owns a dedicated process session. Cleanup stops its
owned process group, including profile writers after the launcher exits, before
deleting that profile. Only transient nonempty-directory races receive a fixed
five-attempt cleanup retry; persistent errors remain failures. Browser sandboxing,
loopback-only control, no browser download, atomic output and embedded fonts
remain unchanged. Windows keeps its existing direct process cleanup path; these
POSIX tests do not establish Windows descendant teardown acceptance.

## Provenance and acceptance

`SOURCE-MANIFEST.json` keeps the original private/public bases, the prior public
SDK source, each scoped private patch and current per-file hashes. Its
`source_commit: null` explicitly identifies a derivative rather than claiming
that one private HEAD equals these public bytes. Public source/build commit and
fixed distribution hashes are recorded separately in release provenance.

The Linux `Verify Python SDK patch` workflow builds both packages on Python
3.11 and 3.13 and checks strict metadata, isolated wheel/sdist installations and
bounded parser/OLS regressions. Separate charts-only environments verify offline
HTML/LaTeX, embedded WOFF2/TTF fonts and absence of SDK/Torch/pandas/NumPy/SciPy.
The provisioned Google Chrome keeps its sandbox enabled for actual PDF/SVG/PNG,
exclusive publication, error refusal and owned profile cleanup tests; a skipped
browser case fails this acceptance step. Actual hosted results, macOS installed-package receipts, public
Release access and PyPI registry installation each need successful readback.
The earlier a2 GitHub asset bytes remain immutable; the a3 package has a new
version and must receive fresh package/publication proof.

These checks do not establish the entire scientific suite, full Stata/vendor
parity, physical CUDA, Windows, Developer ID signing or notarization acceptance.
