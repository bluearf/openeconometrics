# Public source snapshot

This candidate retains all OpenEconometrics production Python, chart, web and desktop source modules from the recorded source commit. It contains no private Git history, internal QA evidence/report files or external research fixture data. Original license and asset attribution files remain included.

Install with `uv sync --frozen --extra app`, `npm --prefix web ci` and `npm --prefix web run build`. The initial installation requires internet. `SOURCE-MANIFEST.json` records every file and the explicit list of fixture-backed test modules omitted from default collection. Their source remains available for review; this does not claim those external-data scientific checks have run in this snapshot. Synthetic tests and native kernels remain included.

Links to internal evidence omitted by the publication policy are replaced with an explicit boundary, listed in `SOURCE-MANIFEST.json`. Dated measurement statements retain their original method, source and version scope. They are historical source documentation, not independently accessible evidence of this snapshot's installed or public-release behavior.

The public source build configuration omits force-include mappings for the internal replay fixtures that this snapshot excludes. The manifest records those exact identity mappings and original/transformed configuration hashes. The inherited build configuration is retained apart from the metadata-refresh version labels and exact charts dependency pin recorded below.

The archive records the prepared source snapshot. Its presence alone does not prove public GitHub access, PyPI publication, a public Mac download, Developer ID signing or notarization. Verify those delivery layers separately.

The package-description refresh uses explicit PyPI version pins for SDK `0.3.20a2` and charts `0.3.2a2`. `SOURCE-MANIFEST.json` records the documentation and package-metadata transformations from public baseline `f202f44f0f4d0c5539105c950e7e3646abc9713f`. The runtime identity guard compares SDK, charts and frontend source against the published `0.3.20a1` / `0.3.2a1` snapshot at `d11ed130feb4ad8e6e990e61bb0bd8a9101cb981`, allowing only the two literal `__version__` replacements. Statistical and chart algorithms, frontend source and desktop source remain unchanged. Fresh package installation checks and registry publication retain their own provenance; this refresh does not claim a new full scientific validation run.
