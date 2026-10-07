"""Public workflow methods render bounded tables and persist without graph dumps."""
import json

from openecon.console import ConsoleSession
from openecon.workspace import Workspace


CODE = '''
import openecon as oe
g = oe.network({'source':['a','a','b','b','c'], 'target':['b','c','c','d','d'],
                'weight':[3,2,1,2,3]}, weight='weight', directed=True, nodes=['iso'])
u = oe.network({'source':['a','b','c'], 'target':['b','c','d']}, nodes=['iso'])
bi = oe.network({'source':[1,1,2], 'target':['a','b','b']}, nodes=['iso'])
print('flow')
flow = g.max_flow('a','d')
assert isinstance(flow, oe.NetworkFlowResult)
assert flow['value'] == 5 and flow['metadata']['certified']
display(flow)
print('cut')
display(g.min_cut('a','d'))
print('workflows')
display(g.hits())
display(g.katz())
display(g.shortest_path('a','d'))
display(g.distances(sources=['a'], targets=['d','iso']))
display(u.eccentricity(disconnected='reachable'))
display(u.distance_summary(disconnected='reachable'))
display(u.bridges())
display(u.articulation_points())
display(u.minimum_spanning_forest())
display(bi.bipartite())
display(bi.maximum_matching())
display(bi.bipartite_projection())
display(u.link_prediction([('a','c'),('a','d')]))
print('complete')
'''


def test_new_workflows_console_latex_and_restart(tmp_path):
    workspace = Workspace(tmp_path / 'owned-workflows')
    session = ConsoleSession(workspace)
    try:
        result = session.execute(CODE, timeout_seconds=60)
        assert result['status'] == 'ok', result.get('error')
        assert len(result['outputs']) == 15
        assert all(out['type'] == 'table' and '\\begin{tabular}' in out['latex']
                   for out in result['outputs'])
        summary = result['outputs'][0]['data']
        assert summary['columns'] == ['Metric', 'Value'] and summary['total_rows'] == 8
        assert len(json.dumps(result['outputs'][0])) < 6000
        assert result['events'][:4] == [
            {'type':'stdout', 'text':'flow\n'}, {'type':'output', 'index':0},
            {'type':'stdout', 'text':'cut\n'}, {'type':'output', 'index':1}]
        assert result['events'][-1] == {'type':'stdout', 'text':'complete\n'}
        saved = Workspace(workspace.path).console_history()[-1]
        assert saved['outputs'] == result['outputs'] and saved['events'] == result['events']
        session.close()
        session = ConsoleSession(Workspace(workspace.path))
        assert session.snapshot()['history'][-1]['outputs'] == result['outputs']
    finally:
        session.close()
