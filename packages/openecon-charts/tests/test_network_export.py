"""Real headless exports: completed layout, typed view, fonts and atomic refusal."""
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

import pytest

from openecon_charts import network
from openecon_charts.network_export import NetworkExportError, _browser


def chart(**options):
    class Graph:
        def to_plot_data(self, **kwargs):
            return {'nodes': [
                {'id': 0, 'label': 'Ankara', 'degree': 2, 'group': 0,
                 'identity': {'type': 'integer', 'value': '1'}, 'x': -40, 'y': 0},
                {'id': 1, 'label': 'İstanbul', 'degree': 2, 'group': 1,
                 'identity': {'type': 'string', 'value': '1'}, 'x': 40, 'y': 20}],
                'edges': [{'source': 0, 'target': 1, 'weight': 1},
                          {'source': 1, 'target': 1, 'weight': .5}],
                'directed': True, 'node_count': 2, 'edge_count': 2,
                'shown_node_count': 2, 'shown_edge_count': 2,
                'sampled': False, 'selection': 'All nodes and edges'}
    return network(Graph(), title='Trade — Türkiye', seed=42, **options).annotate('Reference', x=0, y=-20)


@pytest.fixture
def browser():
    try:
        return _browser(None)
    except NetworkExportError:
        pytest.skip('Installed Chromium needed for genuine browser export validation')


@pytest.mark.parametrize('format', ['pdf', 'svg', 'png'])
def test_real_offline_exports_and_exclusive_publication(tmp_path, browser, format):
    plot = chart(layout='circular', layout_options={'radius': 70})
    original = deepcopy(plot.model_dump())
    before = set(Path(tempfile.gettempdir()).glob('openecon-network-export-*'))
    output = tmp_path / ('trade.' + format)
    getattr(plot, 'to_' + format)(output, browser_executable=browser, width=800, height=400)
    data = output.read_bytes()
    if format == 'pdf':
        assert data.startswith(b'%PDF-1.7') and b'/FontFile2' in data
        assert b'/Subtype /Image' not in data and b'/ToUnicode' in data
    elif format == 'svg':
        svg = ET.fromstring(data)
        ns = {'s': 'http://www.w3.org/2000/svg'}
        assert len(svg.findall('.//s:circle', ns)) == 2
        assert len(svg.findall('.//s:path', ns)) == 2
        assert svg.find('s:style', ns).text.startswith('@font-face')
        assert 'Reference' in ''.join(svg.itertext()) and 'İstanbul' in ''.join(svg.itertext())
        assert svg.attrib['width'] == '800'
    else:
        assert data.startswith(b'\x89PNG\r\n\x1a\n')
        assert struct.unpack('!II', data[16:24])[0] == 800
    fingerprint = hashlib.sha256(data).hexdigest()
    with pytest.raises(FileExistsError):
        getattr(plot, 'to_' + format)(output, browser_executable=browser)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == fingerprint
    assert plot.model_dump() == original
    assert set(Path(tempfile.gettempdir()).glob('openecon-network-export-*')) == before
    assert not list(tmp_path.glob('.openecon-export-*'))


def test_reproducible_seed_layout_and_saved_camera(tmp_path, browser):
    plot = chart(layout='circular')
    for name in ('a', 'b'):
        plot.to_svg(tmp_path / f'{name}.svg', browser_executable=browser)
    assert (tmp_path / 'a.svg').read_bytes() == (tmp_path / 'b.svg').read_bytes()
    saved = plot.with_options(view={'version': 1, 'positions': [
        {'id': 0, 'x': -30, 'y': 10, 'pinned': True,
         'identity': {'type': 'integer', 'value': '1'}},
        {'id': 1, 'x': 30, 'y': 20, 'pinned': True,
         'identity': {'type': 'string', 'value': '1'}}],
        'transform': {'k': 2, 'x': 300, 'y': 200}})
    path = saved.to_svg(tmp_path / 'view.svg', browser_executable=browser)
    assert 'translate(300 200) scale(2)' in path.read_text()
    assert 'cx="-30"' in path.read_text()


def test_font_failure_does_not_replace_existing_output(tmp_path, browser):
    path = tmp_path / 'trade.pdf'
    path.write_bytes(b'original')
    with pytest.raises(NetworkExportError, match='glyph|U\\+'):
        chart(layout='circular').with_options(title='Network 😀').to_pdf(
            path, browser_executable=browser, overwrite=True)
    assert path.read_bytes() == b'original'


def test_work_refusal_does_not_report_success(tmp_path, browser):
    with pytest.raises(NetworkExportError, match='work_limit'):
        chart(layout='d3-force', layout_options={'work_limit': 1000, 'iterations': 2000}).to_svg(
            tmp_path / 'limit.svg', browser_executable=browser)
    assert not (tmp_path / 'limit.svg').exists()


@pytest.mark.parametrize('options', [{'width': 319}, {'height': 1601}, {'scale': float('nan')},
    {'timeout': True}, {'overwrite': 1}, {'scale': 2}])
def test_export_options_rejected_before_browser_start(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        chart().to_pdf(tmp_path / 'bad.pdf', **options)
    assert not list(tmp_path.iterdir())
