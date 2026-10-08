# Public source snapshot

SDK `0.3.18a4` corrects package documentation and metadata on top of the reviewed
SDK `0.3.18a3` source `ca7854593e2591f56c380dc19f7abeba0b7b0117` and public repository
parent `19d7ae7801234aa1d1a55851c0de2fe131c70d37`. Runtime implementations remain
unchanged except the SDK version string. Charts `0.3.0a2` retains the exact
already-published distribution bytes. [PYPI-README.md](PYPI-README.md) is the
package description; the Mac desktop boundary below remains unchanged.

This is a derivative minimal Python SDK 0.3.18a3 and charts 0.3.0a2 patch, from the reviewed public snapshot `7cb7517bf8bacf91f92c85f8df9a2904e76e8974` originally exported from private source `6cd9079a330b03a4f1a568e28aca04a23a8d897b`. It retains the scoped Linux parser RSS and streaming OLS residual-variance fixes and adds the parser startup RSS handshake repair, with their tests. It follows public SDK a2 source `0d8a8ea3ef12ddf8bdadfc8c64135052741f29c9`, merged as `ff50b3f90634282208077cfe62e53919ebb09e82`. It also imports only the scoped installed-browser selection and POSIX descendant/profile cleanup repair for charts 0.3.0a2. Version metadata, delivery documentation and verification automation accompany those fixes. No single private HEAD is claimed to equal this derivative; `SOURCE-MANIFEST.json` uses `source_commit: null` and records the base ancestry and imported patch commits explicitly.

All other production Python, chart, web and desktop source remains from the reviewed public base. Charts 0.3.0a2 changes only that exporter and version metadata; the previous charts 0.3.0a1 release bytes remain immutable. The existing desktop release v0.3.43-alpha.1 still embeds SDK 0.3.18a1; no desktop artifact is rebuilt or retargeted by this Python patch. New private estimator/catalog/UI development and desktop 0.3.44 acceptance are separate.

This repository contains no private Git history, internal QA evidence/report files or external research fixture data. Original license and asset attribution files remain included. `SOURCE-MANIFEST.json` records source hashes and the explicit list of fixture-backed test modules omitted from default collection. Their source remains available for review; their external-data scientific checks are not claimed here. Synthetic tests and native kernels remain included. Repository `.github` delivery automation is maintained separately from the source-file manifest.

Install source with `uv sync --frozen --extra app`, `npm --prefix web ci` and `npm --prefix web run build`. Initial dependency installation requires internet. See [the Python patch scope](docs/python-alpha-0.3.18a3.md) for bounded package/parser/OLS validation and provenance. Preparation alone does not prove hosted CI, public Release access, PyPI publication, a fresh Mac download, Developer ID signing or notarization. Verify those delivery layers separately.

The fixed SDK 0.3.18a3 and charts 0.3.0a2 wheel/source archives were built from
public commit `ca7854593e2591f56c380dc19f7abeba0b7b0117`. Later publisher and
documentation commits retain those exact release assets. The current repository
manifest tracks current listed files; fixed release provenance retains the build
commit's manifest digest. Install the separate Python packages with
`python -m pip install 'openecon==0.3.18a3' 'openecon-charts==0.3.0a2'` after the
registry publication/readback described in [the publishing guide](docs/pypi-publishing.md).

SDK a4 release assets are built from public source
`03add0e10867885ed10cf426c6b6dccbd00dedee`. Its charts a2 assets retain the original
`ca7854593e2591f56c380dc19f7abeba0b7b0117` build and published hashes. Subsequent
publisher-only documentation commits update the current repository manifest;
they do not change either fixed package's bytes or its recorded build manifest.
