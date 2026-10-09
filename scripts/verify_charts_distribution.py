"""Verify dependency-free installed charts in a fresh, isolated interpreter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

PROGRAM = r"""
import importlib.metadata as metadata
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import socket
import sys
import tempfile

for name in ('torch', 'pandas', 'numpy', 'scipy', 'openecon'):
    assert importlib.util.find_spec(name) is None, name
assert metadata.version('openecon-charts') == sys.argv[1]

def denied(*args, **kwargs):
    raise AssertionError('Standalone chart attempted network access')
class OfflineSocket(socket.socket):
    connect = denied
    connect_ex = denied
socket.socket = OfflineSocket
socket.create_connection = denied

import openecon_charts as charts
root = Path(charts.__file__).resolve().parent
assert root.is_relative_to(Path(sys.prefix).resolve())
assert (root / 'assets' / 'Barlow.woff2').read_bytes().startswith(b'wOF2')
assert (root / 'assets' / 'Barlow.ttf').read_bytes().startswith(b'\x00\x01\x00\x00')
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'offline.html'
    chart = charts.scatter(data={'x': [1, 2, 3], 'y': [2, 5, 4]}, x='x', y='y', title='Türkiye')
    chart.save_html(path)
    text = path.read_text()
    assert len(text) > 10000
    assert 'data:font/woff2;base64,' in text
    assert 'connect-src &#x27;none&#x27;' in text
    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'script':
                assert not attrs.get('src'), attrs
            if tag == 'link' and attrs.get('rel') in {'stylesheet', 'preload', 'modulepreload', 'prefetch'}:
                assert not attrs.get('href', '').startswith(('http:', 'https:', '//')), attrs
    Resources().feed(text)
    chart.to_latex(Path(directory) / 'offline.tex', standalone=True)
    assert (Path(directory) / 'offline.tex').stat().st_size > 100
assert not any(name.split('.')[0] in {'torch', 'pandas', 'numpy', 'scipy', 'openecon'} for name in sys.modules)
print(json.dumps({'status': 'passed', 'version': metadata.version('openecon-charts'),
                  'standalone_no_heavy_dependencies': True, 'offline_html_latex': True,
                  'embedded_woff2_and_ttf_fonts': True, 'source': str(root)}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--charts-version", default="0.3.1a1")
    args = parser.parse_args()
    # Keep the virtualenv launcher, including on Linux where it is a symlink.
    check = subprocess.run(
        [str(args.python.absolute()), "-I", "-c", PROGRAM, args.charts_version],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode:
        raise SystemExit(check.stderr)
    record = json.loads(check.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
