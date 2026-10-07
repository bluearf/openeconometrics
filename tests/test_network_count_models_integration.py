"""Count models, predictions and charts retain ordered publication outputs."""
from openecon.console import ConsoleSession
from openecon.output_events import validate_output_events
from openecon.workspace import Workspace


def test_count_model_results_are_ordered_and_survive_reopening(tmp_path):
    workspace = Workspace(tmp_path / "owned")
    code = '''import openecon as oe
graph = oe.network({'source':[0,0,1,3,3,4], 'target':[1,2,2,4,5,5], 'w':[2,3,1,4,2,3]},
                   weight='w', nodes=range(6))
initial = {node:node//3 for node in range(6)}
ordinary = graph.poisson_block_model(2, initial=initial, starts=1)
corrected = graph.degree_corrected_block_model(2, initial=initial, starts=1)
print('counts')
display(ordinary)
display(oe.plot.network(graph, groups=ordinary['membership']))
print('degrees')
display(corrected)
display(oe.plot.network(graph, groups=corrected['membership']))
print('prediction')
predicted = corrected.expected_edges([(0,1),(0,0),(3,4)])
display(predicted)
predicted.to_csv('counts.csv', index=False)
with open('counts.tex', 'w') as stream:
    stream.write(corrected.to_latex())
'''
    session = ConsoleSession(workspace)
    try:
        run = session.execute(code)
        assert run["status"] == "ok", run["error"]
        assert [item["type"] for item in run["outputs"]] == ["table", "plot", "table", "plot", "table"]
        assert run["outputs"][0]["data"]["total_rows"] == run["outputs"][2]["data"]["total_rows"] == 11
        assert run["outputs"][4]["data"]["total_rows"] == 3
        assert run["outputs"][1]["data"]["config"]["network"]["grouping"] == "Poisson SBM blocks"
        assert run["outputs"][3]["data"]["config"]["network"]["grouping"] == "Degree-corrected Poisson SBM blocks"
        assert all("\\toprule" in item["latex"] for item in run["outputs"])
        validate_output_events(run["events"], run["stdout"], run["outputs"])
        assert run["events"] == [
            {"type": "stdout", "text": "counts\n"}, {"type": "output", "index": 0}, {"type": "output", "index": 1},
            {"type": "stdout", "text": "degrees\n"}, {"type": "output", "index": 2}, {"type": "output", "index": 3},
            {"type": "stdout", "text": "prediction\n"}, {"type": "output", "index": 4}]
        assert (workspace.path / "counts.csv").read_text().startswith("source,target,mean_count,presence_probability,identified")
        assert "\\begin{tabular}" in (workspace.path / "counts.tex").read_text()
        history = Workspace(workspace.path).console_history()
    finally:
        session.close()
    reopened = ConsoleSession(Workspace(workspace.path))
    try:
        assert reopened.snapshot()["history"] == history
        assert history[-1]["events"] == run["events"] and history[-1]["outputs"] == run["outputs"]
        readback = reopened.execute("import pandas as pd\nlen(pd.read_csv('counts.csv'))")
        assert readback["status"] == "ok" and readback["outputs"][0]["data"] == "3"
    finally:
        reopened.close()
