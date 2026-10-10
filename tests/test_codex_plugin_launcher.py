"""Exercise the actual plugin process and its existing OpenEconometrics server."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "plugins" / "openeconometrics" / "scripts" / "launch_mcp.py"
TOOLS = {
    "get_capabilities", "list_datasets", "inspect_dataset", "create_example_dataset",
    "run_analysis", "start_analysis", "get_analysis_job", "cancel_analysis_job",
    "get_result", "list_results",
}


def _source_environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), str(ROOT / "packages" / "openecon-charts" / "src")]
        ),
    }


def _config(tmp_path: Path, **changes) -> tuple[Path, Path]:
    workspace = tmp_path / "Project with spaces"
    workspace.mkdir(exist_ok=True)
    config = tmp_path / "Plugin config with spaces.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "command": sys.executable,
        "args": ["-m", "openecon.desktop_entry", "mcp", "--workspace", str(workspace)],
        **changes,
    }), encoding="utf-8")
    return config, workspace


def test_plugin_launches_real_server_and_persists_analysis(tmp_path):
    config, workspace = _config(tmp_path)
    # Preserve the selected virtual environment while testing an executable
    # path containing spaces. Moving its Python symlink would lose pyvenv.cfg.
    if os.name == "posix":
        directory = tmp_path / "Runtime with spaces"
        directory.mkdir()
        executable = directory / "python-runtime"
        executable.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        executable.chmod(0o700)
        contents = json.loads(config.read_text())
        contents["command"] = str(executable)
        config.write_text(json.dumps(contents))
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(LAUNCHER), "--config", str(config)],
        env=_source_environment(),
        cwd=str(tmp_path),
    )

    async def exercise():
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "OpenEconometrics"
                assert {tool.name for tool in (await session.list_tools()).tools} == TOOLS
                empty = await session.call_tool("list_datasets")
                assert empty.structuredContent == {"datasets": []}
                example = await session.call_tool("create_example_dataset")
                assert not example.isError
                dataset = example.structuredContent
                assert dataset["source"] == "example" and "preview" not in dataset
                result = await session.call_tool("run_analysis", {
                    "dataset_id": dataset["id"],
                    "spec": {"outcome": "wage", "predictors": ["education", "experience"],
                             "covariance": "HC3", "missing": "raise"},
                })
                assert not result.isError, result
                payload = result.structuredContent
                assert payload["nobs"] == 480
                assert payload["observation_data_included"] is False
                assert payload["provenance"]["synthetic_data"] is True
                assert not {"sample_positions", "fitted_values", "residuals"} & payload.keys()
                saved = await session.call_tool("get_result", {"result_id": payload["id"]})
                assert saved.structuredContent == payload
                return payload

    payload = asyncio.run(asyncio.wait_for(exercise(), timeout=45))
    persisted = json.loads((workspace / "results" / f"{payload['id']}.json").read_text())
    assert persisted["coefficients"] == payload["coefficients"]
    assert persisted["spec"]["covariance"] == "HC3"
    history = workspace / "console" / "mcp-results" / f"{payload['id']}.json"
    assert json.loads(history.read_text())["source"] == "mcp"
    assert not (tmp_path / ".openecon").exists()


@pytest.mark.parametrize("case,reason", [
    ("missing", "missing or unreadable"), ("malformed", "valid UTF-8 JSON"),
    ("duplicate", "Duplicate configuration keys"), ("nonfinite", "Nonfinite JSON"),
    ("oversized", "exceeds 64 KiB"), ("unknown_key", "Only schema_version, command and args"),
    ("schema_bool", "schema_version must be 1"), ("schema_version", "schema_version must be 1"),
    ("relative_command", "runtime executable must be an absolute path"),
    ("not_executable", "missing or is not executable"),
    ("unsupported_module", "launch arguments from Connect agent"),
    ("extra_args", "launch arguments from Connect agent"),
    ("relative_workspace", "workspace must be an absolute path"),
    ("missing_workspace", "workspace must be an existing directory"),
])
def test_invalid_configuration_refuses_before_runtime_or_workspace_creation(tmp_path, case, reason):
    uncreated = tmp_path / "Do not create this workspace"
    config, _ = _config(tmp_path)
    document = json.loads(config.read_text())
    document["args"][-1] = str(uncreated)
    if case == "missing":
        config.unlink()
    elif case == "malformed":
        config.write_text("{")
    elif case == "duplicate":
        config.write_text('{"command":"one","command":"two","args":[]}')
    elif case == "nonfinite":
        config.write_text('{"schema_version":NaN,"command":"ignored","args":[]}')
    elif case == "oversized":
        config.write_bytes(b" " * (64 * 1024 + 1))
    else:
        if case == "unknown_key":
            document["env"] = {"PYTHONPATH": "unexpected"}
        elif case == "schema_bool":
            document["schema_version"] = True
        elif case == "schema_version":
            document["schema_version"] = 2
        elif case == "relative_command":
            document["command"] = "python"
        elif case == "not_executable":
            target = tmp_path / "not-a-runtime"
            target.write_text("not an executable")
            target.chmod(0o600)
            document["command"] = str(target)
        elif case == "unsupported_module":
            document["args"][1] = "some_other_program"
        elif case == "extra_args":
            document["args"].append("--extra")
        elif case == "relative_workspace":
            document["args"][-1] = "workspace"
        config.write_text(json.dumps(document))
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--config", str(config)], cwd=tmp_path,
        env=_source_environment(), capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "OpenEconometrics plugin:" in result.stderr
    assert reason in result.stderr
    assert "Traceback" not in result.stderr
    assert not uncreated.exists()
    assert not (tmp_path / "workspace").exists()
    assert not (tmp_path / ".openecon").exists()


@pytest.mark.parametrize("explicit", ["relative.json", ""])
def test_config_environment_selector_and_explicit_argument_precedence(tmp_path, explicit):
    missing = tmp_path / "missing-config.json"
    environment = {**_source_environment(), "OPENECON_PLUGIN_CONFIG": str(missing)}
    # An explicit relative path must be refused even when an environment config
    # is present. Neither selector is allowed to guess or create a workspace.
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--config", explicit], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2 and "absolute path" in result.stderr
    result = subprocess.run(
        [sys.executable, str(LAUNCHER)], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2 and "missing or unreadable" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / ".openecon").exists()


@pytest.mark.skipif(os.name != "posix", reason="This passthrough fixture is a POSIX executable")
def test_frozen_runtime_preserves_pipes_diagnostics_and_exit_status(tmp_path):
    executable = tmp_path / "Configured runtime with spaces"
    executable.write_text(
        '#!/bin/sh\nIFS= read -r protocol\nprintf "%s\\n" "$protocol"\n'
        'printf "runtime diagnostic\\n" >&2\nexit 17\n'
    )
    executable.chmod(0o700)
    workspace = tmp_path / "Existing project"
    workspace.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "command": str(executable), "args": ["mcp", "--workspace", str(workspace)],
    }))
    result = subprocess.run(
        [sys.executable, str(LAUNCHER)], input='{"jsonrpc":"2.0","id":1}\n',
        env={**_source_environment(), "OPENECON_PLUGIN_CONFIG": str(config)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 17
    assert result.stdout == '{"jsonrpc":"2.0","id":1}\n'
    assert result.stderr == "runtime diagnostic\n"
