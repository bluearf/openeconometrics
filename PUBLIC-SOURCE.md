# Public source snapshot

This candidate retains all OpenEconometrics production Python, chart, web and desktop source modules from the recorded source commit. It contains no private Git history, internal QA evidence/report files or external research fixture data. Original license and asset attribution files remain included.

Install with `uv sync --frozen --extra app`, `npm --prefix web ci` and `npm --prefix web run build`. The initial installation requires internet. `SOURCE-MANIFEST.json` records every file and the explicit list of fixture-backed test modules omitted from default collection. Their source remains available for review; this does not claim those external-data scientific checks have run in this snapshot. Synthetic tests and native kernels remain included.

The archive records the prepared source snapshot. Its presence alone does not prove public GitHub access, PyPI publication, a public Mac download, Developer ID signing or notarization. Verify those delivery layers separately.
