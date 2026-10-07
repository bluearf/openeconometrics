"""Transport stays lossless across real console, LaTeX and shared admission."""
from copy import deepcopy

import pytest
import openecon as oe
from openecon.console_worker import _execute
from openecon.output_latex import add_output_latex
from openecon.team_output import validate_plot
from openecon_charts.timeline import unpack


def plot(nodes=2, frames=3, description='stable attributes ' * 20):
    graph = oe.network([dict(source=i, target=(i+1)%nodes) for i in range(nodes)],
                       nodes=range(nodes), node_attributes={i: {'description': description} for i in range(nodes)})
    snapshots = oe.network_snapshots({f'Frame {i}': graph for i in range(frames)}, ordered=True)
    return oe.plot.network(snapshots, layout='circular')


def test_console_retains_pooled_timeline_and_full_latex_without_mutation():
    timeline = plot()
    before = timeline.model_dump()
    result = _execute('display(timeline)', {'timeline': timeline}, 'pooled-output')
    assert result['status'] == 'ok' and not result['stdout']
    output, = result['outputs']
    assert output['type'] == 'plot' and output['latex']
    assert output['data']['config']['network']['encoding'] == 'timeline-pool-v1'
    assert unpack(output['data']) == before == timeline.model_dump()
    assert validate_plot(output['data']) == before
    saved = deepcopy(output['data'])
    enriched = add_output_latex({'type': 'plot', 'data': output['data']})
    assert enriched['latex'] == timeline.to_latex() and output['data'] == saved


def test_pooled_transport_cannot_bypass_expanded_shared_result_size_budget():
    transport = plot(nodes=20, frames=31, description='x'*3900).transport_dump()
    assert transport['config']['network']['encoding'] == 'timeline-pool-v1'
    with pytest.raises(ValueError, match='payload is too large'):
        validate_plot(transport)


def test_invalid_pooled_reference_refused_by_latex_and_shared_boundaries():
    transport = plot().transport_dump()
    transport['config']['network']['base']['nodes'][0] = -1
    with pytest.raises(ValueError, match='reference'):
        add_output_latex({'type':'plot','data':transport})
    with pytest.raises(ValueError, match='reference'):
        validate_plot(transport)
