# OpenEconometrics Codex plugin

Run supported OpenEconometrics analyses from Codex against an explicitly selected local workspace. The plugin discovers ten structured tools for dataset inspection, synchronous or background fitting, cancellation and saved-result readback. It includes a workflow skill and uses the same engine and result records as the workbench.

## Install in Codex

From this repository, register its local marketplace and install the plugin:

```sh
codex plugin marketplace add /absolute/path/openecon --json
codex plugin add openeconometrics@openeconometrics --json
codex plugin list --json
```

Prepare the runtime configuration described in [workspace setup](skills/openeconometrics/references/setup.md) before connecting. Use an installed OpenEconometrics runtime with the agent dependencies, or the actual bundled executable from a desktop project's **Connect agent** panel. The launcher uses Python 3 and does not install or download packages.

The default configuration location is `~/.config/openeconometrics/codex-plugin.json`. It contains only an executable, a supported argument vector and an existing absolute workspace. A source environment uses `python -m openecon.cli mcp --workspace ...`; a bundled runtime uses `openecon-runtime mcp --workspace ...`. No research data or credentials belong in the plugin package.

Start a new Codex chat or reload MCP after installation/configuration. Enable the OpenEconometrics plugin and ask, for example:

> Inspect my datasets and available analysis methods.

> Run a synthetic OLS example and return a LaTeX regression table.

> Show my saved analyses and their assumptions.

## Behavior

Capabilities are read from the actual connected runtime. The MCP tools support imported dataset IDs and validated ModelSpec routes, rather than every Python-only helper. Analyses persist in the configured workspace and its Agent history. Connecting to a desktop project's exact workspace lets that project read the same records.

With a runtime exposing `get_result(include_latex=True)`, a saved result can return the existing publication formatter's full LaTeX table, math preview, inference notes and package requirements. This reads the saved numbers without rerunning the analysis. The skill prefers that presentation for regressions; copyable `.tex` output remains available when requested. Older runtimes continue to provide structured results until their server is updated.

Default responses omit raw rows and observation-level arrays. Explicit previews share at most ten rows. Names, labels, summary statistics and model output may still contain sensitive information and can be forwarded by Codex to its AI provider.

The server does not execute arbitrary Python/shell commands, import filesystem paths or install packages. Authorized imports use the workbench or the matching installed CLI. This is local integration, not cloud hosting or blanket Stata parity.

## Package and validation

The root `plugin.json`/`mcp.json` follow Agent Plugins 1.0. The synchronized `.codex-plugin/plugin.json`/`.mcp.json` provide Codex compatibility. The Codex declaration additionally forwards `OPENECON_PLUGIN_CONFIG` when it is explicitly set in the host environment; other hosts use the default configuration path. The single plugin directory is independently portable; the marketplace file is outside it.

The launcher tests start its actual configured process, discover the MCP tools, fit a clearly synthetic model and read its persisted result. They also verify setup refusals, paths containing spaces, protocol output and exit-code preservation. Native Windows execution requires a Windows verification run.

The application logo is reused under this repository's Apache-2.0 license.
