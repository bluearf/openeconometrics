"""The copyable launcher and desktop panel share an isolated project."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
import sys
from uuid import uuid4

from fastapi.testclient import TestClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest

from openecon.data import DataError
from openecon.desktop_runtime import create_desktop_app
from openecon.mcp_launcher import connection_config
from openecon.workspace import Workspace


def test_frozen_config_uses_bundle_with_spaces_and_no_sibling_cli(monkeypatch, tmp_path):
    launcher = tmp_path / "Application Support" / "openecon-runtime"
    workspace = tmp_path / "Project with spaces"
    monkeypatch.setattr(sys, "executable", str(launcher))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    config = connection_config(workspace)
    expected = [str(launcher), "mcp", "--workspace", str(workspace)]
    assert shlex.split(config["mcp_command"]) == expected
    assert shlex.split(config["codex_command"]) == ["codex", "mcp", "add", "openecon", "--", *expected]
    assert shlex.split(config["claude_command"]) == ["claude", "mcp", "add", "--transport", "stdio", "openecon", "--", *expected]
    assert config["mcp_server"] == {"command": str(launcher), "args": expected[1:]}


def test_desktop_mcp_results_reopen_in_same_project_without_history_overwrite(tmp_path):
    root = tmp_path / "Desktop data with spaces"
    first, second = "a" * 32, "b" * 32
    prefix = f"/api/desktop/projects/{first}/workspace"
    with TestClient(create_desktop_app(root)) as client:
        headers = {"X-OpenEcon-Token": client.get(prefix + "/session").json()["token"]}
        assert client.get(prefix + "/config").status_code == 401
        config = client.get(prefix + "/config", headers=headers).json()
        params = StdioServerParameters(**config["mcp_server"], env={
            **os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        })
        dataset = client.post(prefix + "/datasets/example", headers=headers).json()
        revision = client.get(prefix + "/results/revision", headers=headers).json()["revision"]
        project_store = Workspace(root / "projects" / first)
        console_record = {"id": "local-console-result", "created_at": "2026-10-06T00:00:00+00:00",
                          "code": "print('local')", "status": "ok", "stdout": "local\n",
                          "outputs": [], "variables": [], "error": None, "duration_ms": 1,
                          "session_generation": 1}

        async def exercise():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.call_tool("list_datasets")
                    assert listed.structuredContent["datasets"][0]["id"] == dataset["id"]
                    # A separate console writer may finish while the agent runs.
                    project_store.append_console_history(console_record)
                    result = await session.call_tool("run_analysis", {
                        "dataset_id": dataset["id"],
                        "spec": {"outcome": "wage", "predictors": ["education", "experience"],
                                 "covariance": "HC3"},
                    })
                    assert not result.isError, result
                    return result.structuredContent

        result = asyncio.run(asyncio.wait_for(exercise(), timeout=45))
        assert client.get(prefix + "/results/revision", headers=headers).json()["revision"] != revision
        history = client.get(prefix + "/console", headers=headers).json()["history"]
        assert {record["id"] for record in history} == {console_record["id"], result["id"]}
        agent = next(record for record in history if record.get("source") == "mcp")
        output = agent["outputs"][0]
        assert output["data"]["coefficients"] == result["coefficients"]
        assert "\\toprule" in output["latex"]
        assert "sample_positions" not in output["data"] and "covariance_matrix" not in output["data"]
        assert project_store.console_history() == [console_record]
        assert client.get(prefix + "/console", headers=headers).json()["status"]["pid"] is None

    with TestClient(create_desktop_app(root)) as client:
        headers = {"X-OpenEcon-Token": client.get(prefix + "/session").json()["token"]}
        assert len(client.get(prefix + "/console", headers=headers).json()["history"]) == 2
        other = f"/api/desktop/projects/{second}/workspace"
        other_headers = {"X-OpenEcon-Token": client.get(other + "/session").json()["token"]}
        assert client.get(other + "/console", headers=other_headers).json()["history"] == []
        assert client.get(other + "/datasets", headers=other_headers).json()["datasets"] == []
        assert client.get(other + "/results/revision", headers=other_headers).json() == {"revision": "0"}


def test_agent_history_budgets_bytes_before_json_parsing(tmp_path, monkeypatch):
    import openecon.workspace as workspace_module
    monkeypatch.setattr(workspace_module, "_MAX_HISTORY_BYTES", 1000)
    store = Workspace(tmp_path)
    directory = store.console_path / "mcp-results"
    directory.mkdir()
    for index in range(20):
        record = {"id": str(uuid4()), "created_at": str(index).zfill(3), "stdout": "x" * 250}
        path = directory / f"{record['id']}.json"
        path.write_text(json.dumps(record))
        os.utime(path, ns=(index + 1, index + 1))
    reads = []
    original = store._record

    def read(directory, record_id):
        reads.append(record_id)
        return original(directory, record_id)

    monkeypatch.setattr(store, "_record", read)
    history = store.display_history()
    assert len(reads) == len(history) == 1000 // path.stat().st_size
    assert history[-1]["created_at"] == "019"


def test_agent_display_limit_matches_actual_disk_serialization(tmp_path, monkeypatch):
    import openecon.workspace as workspace_module
    store = Workspace(tmp_path)
    data = store.create_example()
    result = store.run_analysis(data["id"], {"outcome": "wage", "predictors": ["education"]})
    store.save_agent_result(result)
    path = store.console_path / "mcp-results" / f"{result['id']}.json"
    compact_bytes = len(json.dumps(json.loads(path.read_text()), ensure_ascii=False).encode())
    actual_bytes = path.stat().st_size
    assert actual_bytes > compact_bytes
    monkeypatch.setattr(workspace_module, "_MAX_AGENT_RECORD_BYTES", (actual_bytes + compact_bytes) // 2)
    result["id"] = str(uuid4())
    with pytest.raises(DataError, match="history limit"):
        store.save_agent_result(result)
    assert not (path.parent / f"{result['id']}.json").exists()
