# Runtime and workspace setup

The launcher uses the existing OpenEconometrics MCP server. It requires Python 3 and an explicit local configuration at `~/.config/openeconometrics/codex-plugin.json`. `OPENECON_PLUGIN_CONFIG` or `--config` can select another absolute configuration path.

The configuration is a strict copy of the workbench's `mcp_server` entry, with optional `schema_version: 1`:

```json
{
  "schema_version": 1,
  "command": "/absolute/path/openecon-runtime",
  "args": ["mcp", "--workspace", "/absolute/path/existing-project"]
}
```

Both the executable and workspace must exist. Paths containing spaces are ordinary JSON strings. Do not include shell quoting inside a JSON argument. The launcher refuses a missing setup; it does not download a runtime or fall back to another workspace.

## Share an existing desktop project

Open the intended project in OpenEconometrics, then choose **Connect agent**. Copy the actual executable and workspace from that project's connection configuration. The bundled executable supports `mcp --workspace` directly. Use these exact values in the plugin configuration.

Changing the plugin configuration selects a different project on the next MCP connection. Reload the plugin's MCP connection or start a new Codex chat. Re-read capabilities and datasets before continuing; dataset IDs belong to the selected workspace.

## Use an installed Python package

An installed environment with `openecon[app]` can use:

```json
{
  "schema_version": 1,
  "command": "/absolute/path/environment/bin/python",
  "args": ["-m", "openecon.cli", "mcp", "--workspace", "/absolute/path/existing-workspace"]
}
```

The source-module forms `openecon.cli` and `openecon.desktop_entry` are supported. The latter needs the `agent` extra; the conventional CLI needs `cli` and `agent`. Use a maintained installed environment, not a temporary checkout's import path. Runtime dependencies are prepared during explicit setup, never downloaded during an MCP call.

## LaTeX result support

Plugin 0.1.2 prefers publication LaTeX for regression results when the selected runtime exposes `get_result(include_latex=True)`. The package does not change a separately installed or bundled server. Check the connected tool schema; an older server requires an explicit runtime update before it can return these fields. Preserve the selected workspace and its saved results when updating.

## Import a user-selected file

Prefer importing through the OpenEconometrics project UI. When the configured installed environment provides the CLI and the user has authorized the file, Codex can invoke that same environment's `openecon.cli import` with the exact selected workspace:

```text
/absolute/path/environment/bin/python -m openecon.cli import /absolute/path/research.csv --workspace /absolute/path/existing-workspace
```

Pass an argument list through the host's execution facility; avoid constructing a shell command from dataset names. Do not invoke an import mode on a frozen executable that only provides the desktop/MCP entrypoints. Read the returned dataset ID, then verify it with `list_datasets` and `inspect_dataset`. Importing stores a local snapshot. Ask for a file path only if it is missing from the user's request and available context.

## Connection failures

Configuration errors are written to stderr before starting the runtime. Check the explicit configuration, executable permissions, installed agent dependencies and workspace directory. Keep the configured selection intact while diagnosing a failure. Do not delete workspace files, install packages or change other plugins merely to make discovery pass.
