"""New network statistics share ordered display, LaTeX and durable history."""
import json
import subprocess
import sys

from openecon.console import ConsoleSession
from openecon.workspace import Workspace


CODE = '''
import openecon as oe
g = oe.network({'source':['a','a','b','c'], 'target':['b','c','c','d']})
d = oe.network({'source':['a','b','a'], 'target':['b','c','c']}, directed=True)
print('triads')
display(d.triad_census())
display(g.triad_census())
print('global-cut')
cut = g.global_min_cut()
assert isinstance(cut, oe.NetworkCutResult)
assert cut['value'] == 1
assert set(cut['source_partition']) | set(cut['target_partition']) == {'a','b','c','d'}
display(cut)
display(cut['cut_edges'])
print('qap')
comparison = g.qap_correlation(g, values='binary', permutations=39, seed=11)
assert abs(float(comparison.correlation.iloc[0]) - 1) < 1e-12
assert 0 < float(comparison.pvalue.iloc[0]) <= 1
display(comparison)
print('complete')
'''


def test_new_network_public_names_keep_control_plane_lazy():
    result = subprocess.run([sys.executable, "-c", '''
import sys
import openecon as oe
assert 'NetworkCutResult' in dir(oe)
assert 'torch' not in sys.modules
capability = oe.capabilities()['network']
assert {'global_min_cut','triad_census','qap_correlation'} <= set(capability['algorithms'])
assert capability['out_of_core_graph'] is True
assert capability['out_of_core_analysis'] is True
assert capability['devices']['disk_pagerank'] == ['cpu']
assert 'torch' not in sys.modules
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_network_statistics_console_latex_and_restart(tmp_path):
    workspace = Workspace(tmp_path / "owned-inference")
    session = ConsoleSession(workspace)
    try:
        result = session.execute(CODE, timeout_seconds=60)
        assert result["status"] == "ok", result.get("error")
        assert len(result["outputs"]) == 5
        assert all(out["type"] == "table" and "\\begin{tabular}" in out["latex"]
                   for out in result["outputs"])
        assert result["outputs"][0]["data"]["total_rows"] == 16
        assert result["outputs"][1]["data"]["total_rows"] == 4
        assert len(json.dumps(result["outputs"][2])) < 6000
        assert result["events"] == [
            {"type": "stdout", "text": "triads\n"},
            {"type": "output", "index": 0}, {"type": "output", "index": 1},
            {"type": "stdout", "text": "global-cut\n"},
            {"type": "output", "index": 2}, {"type": "output", "index": 3},
            {"type": "stdout", "text": "qap\n"}, {"type": "output", "index": 4},
            {"type": "stdout", "text": "complete\n"},
        ]
        saved = Workspace(workspace.path).console_history()[-1]
        assert saved["outputs"] == result["outputs"]
        session.close()
        session = ConsoleSession(Workspace(workspace.path))
        assert session.snapshot()["history"][-1]["outputs"] == result["outputs"]
    finally:
        session.close()
