# Public source snapshot

This is a derivative Python SDK patch, version 0.3.18a2, from the reviewed public snapshot `7cb7517bf8bacf91f92c85f8df9a2904e76e8974` originally exported from private source `6cd9079a330b03a4f1a568e28aca04a23a8d897b`. It imports only the scoped Linux parser RSS and streaming OLS residual-variance fixes and their tests. Version metadata, delivery documentation and verification automation accompany those fixes. No single private HEAD is claimed to equal this derivative; `SOURCE-MANIFEST.json` uses `source_commit: null` and records the base ancestry and imported patch commits explicitly.

All other production Python, chart, web and desktop source remains from the reviewed public base. The charts package/distribution version remains 0.3.0a1. The existing desktop release v0.3.43-alpha.1 still embeds SDK 0.3.18a1; no desktop artifact is rebuilt or retargeted by this Python patch. New private estimator/catalog/UI development and desktop 0.3.44 acceptance are separate.

This repository contains no private Git history, internal QA evidence/report files or external research fixture data. Original license and asset attribution files remain included. `SOURCE-MANIFEST.json` records source hashes and the explicit list of fixture-backed test modules omitted from default collection. Their source remains available for review; their external-data scientific checks are not claimed here. Synthetic tests and native kernels remain included. Repository `.github` delivery automation is maintained separately from the source-file manifest.

Install source with `uv sync --frozen --extra app`, `npm --prefix web ci` and `npm --prefix web run build`. Initial dependency installation requires internet. See [the Python patch scope](docs/python-alpha-0.3.18a2.md) for bounded package/parser/OLS validation and provenance. Preparation alone does not prove hosted CI, public Release access, PyPI publication, a fresh Mac download, Developer ID signing or notarization. Verify those delivery layers separately.

The fixed SDK 0.3.18a2 wheel and source archive were built from public commit
`0d8a8ea3ef12ddf8bdadfc8c64135052741f29c9`. Later publisher/documentation commits
retain those exact release assets. The repository manifest tracks current listed
files; the fixed release provenance retains the build commit's manifest digest.
Install the separate Python packages with
`python -m pip install 'openecon==0.3.18a2' 'openecon-charts==0.3.0a1'` after the
registry publication/readback described in [the publishing guide](docs/pypi-publishing.md).
