# Python SDK alpha 0.3.18a2

The subsequent [SDK a3 patch](python-alpha-0.3.18a3.md) repairs a remaining
intermittent startup RSS refusal found in a longer hosted test run. Existing a2
GitHub release bytes remain historical and immutable; current PyPI publication
acceptance must use the a3 source and its fresh validation receipts.

This Python-only patch derives from the reviewed public source snapshot
`7cb7517bf8bacf91f92c85f8df9a2904e76e8974`, originally exported from private
development source `6cd9079a330b03a4f1a568e28aca04a23a8d897b`.

The SDK version is `0.3.18a2`. Its dependency remains exactly
`openecon-charts==0.3.0a1`; charts distribution bytes are unchanged. The existing
Mac desktop release `v0.3.43-alpha.1` remains pinned to SDK `0.3.18a1`. This patch
does not publish or claim acceptance of the newer private SDK or desktop 0.3.44.

## Scoped changes

- Linux file-parser supervision reads its own process high-water RSS from
  `/proc/self/status` instead of attributing an inherited parent peak to the
  parser. Unavailable or malformed measurements fail supervision. Total/growth
  memory limits, virtual-memory limits, and bounded batches remain enforced.
- Streaming OLS rejects numerical-zero residual variance relative to outcome
  scale and QR reduction depth. It distinguishes a perfect fit's rounding
  residual from small, real residual noise, including outcome rescaling and
  large intercept offsets.

The source fixes are imported from private commits
`b8a2b461702251412a151105be530de50e47b19e` and
`e4589e8aa12a4c8745d2ddcda235b0edd4f965bd`; the accompanying resource-test
portability adjustment is `8d29d23ce45e169a5896d9cc1469786412efc63f`.
Only the two repaired production modules and their scoped tests are imported.
Private Git history, new estimator/catalog/UI development, research fixture
bytes, and internal QA evidence remain excluded.

## Provenance and validation

`SOURCE-MANIFEST.json` identifies this as a derivative public patch, records the
base private/public ancestry, imported patch commits, and current file hashes.
It does not represent the patched bytes as an unchanged private HEAD. The
public patch commit and built distribution hashes are recorded separately in
the Python SDK release's provenance after review.

`Verify Python SDK patch` builds from this local public checkout on Linux with
Python 3.11 and 3.13. It checks strict package metadata, isolated wheel/sdist
installation, lazy registry module presence, OLS/Poisson, JSON restoration and
prediction alignment, LaTeX/chart output, Parquet parsing after Torch import,
and the exact-fit guard. Scoped parser/OLS regression suites are separate from
the installed-package smoke checks.

Actual hosted CI results, macOS installed-package receipts, secret scanning,
release access, and PyPI publication each require their own successful readback.
These bounded checks do not establish full vendor parity, the entire scientific
test suite, GPU execution, Windows acceptance, or desktop signing/notarization.
