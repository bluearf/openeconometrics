"""New network models use bounded displays and durable ordered outputs."""
import subprocess
import sys

from openecon.console import ConsoleSession
from openecon.output_events import validate_output_events
from openecon.workspace import Workspace


CODE = '''import openecon as oe
y = oe.network({'source':[0,0,1,3], 'target':[1,2,2,1], 'w':[2,1,4,3]},
               weight='w', nodes=range(4), directed=True)
a = oe.network({'source':[2,1,3,1], 'target':[3,0,1,2], 'w':[3,1,2,5]},
               weight='w', nodes=range(4), directed=True)
b = oe.network({'source':[2,3,1], 'target':[3,0,3], 'w':[1,4,5]},
               weight='w', nodes=range(4), directed=True)
mrqap = y.qap_regression({'a':a,'b':b}, permutations=19, seed=17)
print('regression')
display(mrqap)
graph = oe.network({'source':[0,0,1,3,3,4], 'target':[1,2,2,4,5,5]}, nodes=range(6))
fit = graph.block_model(2, initial={node:node//3 for node in range(6)}, starts=1)
print('blocks')
display(fit)
display(oe.plot.network(graph, groups=fit['membership']))
snapshots = oe.network_snapshots({'first':a,'next':b}, ordered=True)
print('snapshots')
display(snapshots)
display(snapshots.transitions())
mrqap.to_csv('mrqap.csv', index=False)
with open('blocks.tex', 'w') as stream:
    stream.write(fit.to_latex())
'''


def test_public_import_keeps_control_plane_lazy():
    result = subprocess.run([sys.executable, '-c', '''import sys
import openecon
assert all(name not in sys.modules for name in (
    'torch', 'pandas', 'openecon._network_sbm', 'openecon._network_temporal',
    'openecon._network_mrqap'))
assert {'network_snapshots','NetworkSnapshots','NetworkBlockResult'} <= set(openecon.__all__)
'''], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_models_display_publication_tables_chart_and_reopen_history(tmp_path):
    workspace = Workspace(tmp_path / 'owned')
    session = ConsoleSession(workspace)
    try:
        run = session.execute(CODE)
        assert run['status'] == 'ok', run['error']
        assert [item['type'] for item in run['outputs']] == ['table','table','plot','table','table']
        assert [item['data']['total_rows'] for item in run['outputs'] if item['type'] == 'table'] == [3,11,7,1]
        assert run['outputs'][2]['data']['config']['network']['grouping'] == 'Bernoulli SBM blocks'
        assert all('\\toprule' in item['latex'] for item in run['outputs'])
        validate_output_events(run['events'], run['stdout'], run['outputs'])
        assert run['events'] == [
            {'type':'stdout','text':'regression\n'}, {'type':'output','index':0},
            {'type':'stdout','text':'blocks\n'}, {'type':'output','index':1}, {'type':'output','index':2},
            {'type':'stdout','text':'snapshots\n'}, {'type':'output','index':3}, {'type':'output','index':4}]
        assert (workspace.path / 'mrqap.csv').read_text().startswith('term,coefficient,statistic,pvalue')
        assert '\\begin{tabular}' in (workspace.path / 'blocks.tex').read_text()
        history = Workspace(workspace.path).console_history()
        assert history[-1]['events'] == run['events'] and history[-1]['outputs'] == run['outputs']
    finally:
        session.close()
    reopened = ConsoleSession(Workspace(workspace.path))
    try:
        assert reopened.snapshot()['history'] == history
        readback = reopened.execute("import pandas as pd\nlen(pd.read_csv('mrqap.csv'))")
        assert readback['status'] == 'ok' and readback['outputs'][0]['data'] == '3'
    finally:
        reopened.close()
