"""Directed cut orientation and publication outputs survive console restart."""
import json
from pathlib import Path

from openecon.console import ConsoleSession
from openecon.workspace import Workspace


def test_directed_cut_example_console_and_reopen(tmp_path):
    workspace = Workspace(tmp_path / "owned-directed-cut")
    code = (Path(__file__).resolve().parents[1] /
            "docs/examples/network_directed_global_cut.py").read_text()
    session = ConsoleSession(workspace)
    try:
        result = session.execute(code, timeout_seconds=60)
        assert result["status"] == "ok", result.get("error")
        assert [out["type"] for out in result["outputs"]] == ["table", "table", "plot"]
        summary, edges, plot = result["outputs"]
        assert summary["data"]["total_rows"] == 8
        assert "Flow problems" in json.dumps(summary)
        assert edges["data"]["columns"] == ["source", "target", "capacity"]
        assert edges["data"]["rows"] == [["1", 1, 2.0]]
        assert all("\\begin{tabular}" in out["latex"] for out in (summary, edges))
        chart = plot["data"]["config"]["network"]
        assert chart["node_count"] == 3 and chart["edge_count"] == 5
        assert chart["sampled"] is False
        assert "DIRECTED_GLOBAL_CUT_RECEIPT:" in result["stdout"]
        saved = Workspace(workspace.path).console_history()[-1]
        assert saved["outputs"] == result["outputs"]
        assert saved["events"] == result["events"]
        session.close()
        session = ConsoleSession(Workspace(workspace.path))
        reopened = session.snapshot()["history"][-1]
        assert reopened["outputs"] == result["outputs"]
        assert reopened["events"] == result["events"]
    finally:
        session.close()
