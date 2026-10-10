# OpenEconometrics Codex plugin 0.1.1 verification

Verified on macOS on 2026-10-09 against source base `3ad8b093eb8a7e45624bd79fd2b9623725a92898` and Codex CLI 0.151.0. This is local plugin integration evidence; it does not establish public listing, Windows execution or independent econometric parity.

- Portable manifests pass Agent Plugins 1.0 JSON schemas. Codex compatibility identity/interface agree; only the Codex MCP declaration forwards the optional explicit runtime-config environment selector.
- 26 launcher/MCP/desktop-connection tests pass, including actual STDIO discovery, synthetic fitting, saved-result readback, bounded configuration refusals, spaces in paths and protocol/exit-status preservation. Ruff and diff checks pass.
- All 753 SDK and 23 charts package files match source, built wheel and isolated installed runtime. Metadata, RECORD hashes and licenses pass; all 62 distributions in the application dependency closure are satisfied. Isolated imports use installed packages; SciPy/statsmodels are absent.
- The plugin was installed and enabled through the supported Codex plugin commands. A fresh app-server engine discovered its enabled skill and all ten MCP tools. The 23 previously installed plugins retained their state.
- Through Codex tool calls, an explicitly marked synthetic workspace ran a 480-observation OLS model (`wage ~ education + experience`, HC3, missing=raise). The completed job and persisted result agree. Closing and reopening the Codex engine returned the identical completed job and result. Raw observation arrays were omitted.
- A separate fresh Codex engine without the synthetic configuration override connected successfully to the persistent default workspace and read its empty dataset inventory. No synthetic data were inserted into that default workspace.

Receipts, JUnit output, runtime provenance and the verified archive are saved under `artifacts/codex-plugin/` in the development checkout. The archive SHA-256 is `44393996116fe9bfa756370ef1d3a3b2be6589b649f1953b8cc8c3cf35318958`.

The loaded desktop chat and its visual plugin panel were not verified; use a new Codex chat after installation. The existing full capability response is large; individual model execution remains validated by ModelSpec and the engine. An estimator/family capability filter would improve future catalog use.
