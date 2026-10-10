# Public source snapshot

This candidate retains all OpenEconometrics production Python, chart, web and desktop source modules from the recorded source commit. It contains no private Git history, internal QA evidence/report files or external research fixture data. Original license and asset attribution files remain included.

Install with `uv sync --frozen --extra app`, `npm --prefix web ci` and `npm --prefix web run build`. The initial installation requires internet. `SOURCE-MANIFEST.json` records every file and the explicit list of fixture-backed test modules omitted from default collection. Their source remains available for review; this does not claim those external-data scientific checks have run in this snapshot. Synthetic tests and native kernels remain included.

Links to internal evidence omitted by the publication policy are replaced with an explicit boundary, listed in `SOURCE-MANIFEST.json`. Dated measurement statements retain their original method, source and version scope. They are historical source documentation, not independently accessible evidence of this snapshot's installed or public-release behavior.

The public source build configuration omits force-include mappings for the internal replay fixtures that this snapshot excludes. The manifest records those exact identity mappings and original/transformed configuration hashes. Package metadata, dependencies and all other build settings are retained.

The archive records the prepared source snapshot. Its presence alone does not prove public GitHub access, PyPI publication, a public Mac download, Developer ID signing or notarization. Verify those delivery layers separately.

Publication documentation explicitly selects the new GitHub package assets and identifies the previous immutable PyPI pair. The manifest records these three documentation transformations against the original frozen source. Statistical, chart, web and desktop implementation bytes remain unchanged.
