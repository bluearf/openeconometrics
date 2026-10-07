"""Network results use the same ordered, persisted console/chart protocol."""
import json
import subprocess
import sys

from openecon.console import ConsoleSession
from openecon.workspace import Workspace
from openecon.team_output import validate_plot
import pytest


def test_network_public_api_keeps_the_control_plane_lazy():
    result = subprocess.run([sys.executable, "-c", """
import sys
import openecon as oe
assert 'torch' not in sys.modules
assert {'Network', 'NetworkFlowResult', 'network'} <= set(dir(oe))
assert 'torch' not in sys.modules
assert oe.capabilities()['network']['out_of_core_graph'] is True
assert oe.capabilities()['network']['out_of_core_analysis'] is True
assert oe.capabilities()['network']['disk_native_algorithms'] == ['summary', 'degree', 'pagerank', 'components_weak']
assert 'torch' not in sys.modules
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_network_table_chart_and_latex_persist_in_execution_order(tmp_path):
    workspace = Workspace(tmp_path / "owned-network")
    session = ConsoleSession(workspace)
    try:
        run = session.execute("""
graph = oe.network(data={'source': ['A', 'B', 'C'], 'target': ['B', 'C', 'A']},
                   directed=True, nodes=['isolated'])
print('network-table')
display(graph.degree())
print('network-chart')
display(oe.plot.network(graph, title='Network <example>'))
print('network-complete')
""")
        assert run["status"] == "ok", run.get("error")
        assert [item["type"] for item in run["outputs"]] == ["table", "plot"]
        chart = run["outputs"][1]
        payload = chart["data"]["config"]["network"]
        assert validate_plot(chart["data"]) == chart["data"]
        malicious = json.loads(json.dumps(chart["data"]))
        malicious["config"]["javascript"] = "untrusted()"
        with pytest.raises(ValueError, match="Unsupported network"):
            validate_plot(malicious)
        assert payload["node_count"] == 4 and payload["edge_count"] == 3
        assert payload["sampled"] is False
        assert "\\begin{tabular}" in chart["latex"]
        assert run["events"] == [
            {"type": "stdout", "text": "network-table\n"},
            {"type": "output", "index": 0},
            {"type": "stdout", "text": "network-chart\n"},
            {"type": "output", "index": 1},
            {"type": "stdout", "text": "network-complete\n"},
        ]
        saved = json.loads(json.dumps(Workspace(workspace.path).console_history()[-1]))
        assert saved["outputs"] == run["outputs"]
        session.close()
        session = ConsoleSession(Workspace(workspace.path))
        assert session.snapshot()["history"][-1]["outputs"] == run["outputs"]
    finally:
        session.close()
