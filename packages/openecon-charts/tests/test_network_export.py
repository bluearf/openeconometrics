"""Real headless exports: completed layout, typed view, fonts and atomic refusal."""
from copy import deepcopy
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from openecon_charts import network
from openecon_charts.network_export import NetworkExportError, _browser
from openecon_charts import network_export


def test_linux_socket_environment_preserves_caller_and_other_settings(monkeypatch, tmp_path):
    long_root = str(tmp_path / ('nested-' + 'x' * 100))
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        monkeypatch.setenv(key, long_root)
    monkeypatch.setenv('OPENECON_SOCKET_TEST_SENTINEL', 'unchanged')
    monkeypatch.setattr(network_export.sys, 'platform', 'linux')
    environment = network_export._browser_environment()
    assert all(environment[key] == '/tmp' for key in ('TMPDIR', 'TMP', 'TEMP'))
    assert environment['OPENECON_SOCKET_TEST_SENTINEL'] == 'unchanged'
    assert all(os.environ[key] == long_root for key in ('TMPDIR', 'TMP', 'TEMP'))


@pytest.mark.parametrize('platform', ['darwin', 'win32'])
def test_other_platforms_preserve_native_browser_environment(monkeypatch, platform):
    monkeypatch.setattr(network_export.sys, 'platform', platform)
    assert network_export._browser_environment() is None


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX browser supervisor')
def test_supervisor_passes_short_socket_root_to_actual_native_child(monkeypatch, tmp_path):
    long_root = str(tmp_path / ('nested-' + 'x' * 100))
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        monkeypatch.setenv(key, long_root)
    monkeypatch.setenv('OPENECON_SOCKET_TEST_SENTINEL', 'unchanged')
    bootstrap = network_export._PYTHON_BROWSER_BOOTSTRAP
    marker = 'from openecon_charts.network_export import _supervise_browser; '
    assert marker in bootstrap
    monkeypatch.setattr(network_export, '_PYTHON_BROWSER_BOOTSTRAP',
                        bootstrap.replace(marker, marker + "sys.platform = 'linux'; "))
    output = tmp_path / 'native-environment.json'
    command = [sys.executable, '-I', '-c',
               'import json, os, sys; from pathlib import Path; '
               'Path(sys.argv[1]).write_text(json.dumps({key: os.environ[key] '
               'for key in ["TMPDIR", "TMP", "TEMP", "OPENECON_SOCKET_TEST_SENTINEL"]}))',
               str(output)]
    with (tmp_path / 'browser.log').open('wb') as log:
        process = network_export._start_browser(command, log, tmp_path / 'browser-state.json')
        try:
            deadline = time.monotonic() + 5
            while network_export._browser_exit_code(process) is None:
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert network_export._browser_exit_code(process) == 0
            assert json.loads(output.read_text()) == {
                'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp',
                'OPENECON_SOCKET_TEST_SENTINEL': 'unchanged'}
        finally:
            network_export._stop_browser(process)
    assert all(os.environ[key] == long_root for key in ('TMPDIR', 'TMP', 'TEMP'))


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
def browser(monkeypatch):
    try:
        executable = _browser(None)
    except NetworkExportError:
        pytest.skip('Installed Chromium needed for genuine browser export validation')
    original = network_export._wait_for_debugging_port

    def startup(descriptor, process, deadline):
        try:
            return original(descriptor, process, deadline)
        except NetworkExportError:
            log = descriptor.parent.parent / 'browser.log'
            if log.is_file():
                with log.open('rb') as handle:
                    handle.seek(0, os.SEEK_END)
                    handle.seek(max(0, handle.tell() - 4096))
                    print('Owned Chromium startup log:', handle.read(4096).decode(errors='replace'))
            raise

    monkeypatch.setattr(network_export, '_wait_for_debugging_port', startup)
    return executable


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


def test_configured_browser_avoids_automatic_wrapper_and_explicit_path_wins(tmp_path, monkeypatch):
    configured = tmp_path / 'installed-chrome'
    explicit = tmp_path / 'explicit-edge'
    for path in (configured, explicit):
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    monkeypatch.setenv('OPENECON_BROWSER_EXECUTABLE', str(configured))
    assert _browser(None) == str(configured)
    assert _browser(explicit) == str(explicit)


def test_invalid_configured_browser_fails_without_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENECON_BROWSER_EXECUTABLE', str(tmp_path / 'missing-chrome'))
    with pytest.raises(FileNotFoundError):
        _browser(None)


@pytest.fixture
def startup_clock(monkeypatch):
    clock = SimpleNamespace(now=0., sleeps=[])
    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds
    monkeypatch.setattr(network_export.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(network_export.time, 'sleep', sleep)
    return clock


@pytest.mark.parametrize('initial', [None, '', '\n', '42', '42\n',
    '42\n/devtools/brow', '42\n/devtools/browser/',
    'invalid\n/devtools/browser/owned', '0\n/devtools/browser/owned',
    '65536\n/devtools/browser/owned', '123456789\n/devtools/browser/owned',
    '42\n/devtools/browser/invalid path', '42\n/devtools/browser/invalid\0',
    '42\n/devtools/browser/invalid?query', '42\n/devtools/browser/invalid#fragment', b'\xff'])
def test_startup_waits_for_complete_valid_port_descriptor(tmp_path, startup_clock, initial):
    descriptor = tmp_path / 'DevToolsActivePort'
    if isinstance(initial, bytes):
        descriptor.write_bytes(initial)
    elif initial is not None:
        descriptor.write_text(initial, encoding='ascii')
    def poll():
        if startup_clock.sleeps:
            descriptor.write_text('43210\n/devtools/browser/owned', encoding='ascii')
        return None
    process = SimpleNamespace(poll=poll)
    assert network_export._wait_for_debugging_port(descriptor, process, 1.) == 43210
    assert startup_clock.sleeps == [.05]


@pytest.mark.parametrize('contents', [None, '', '42\n', 'bad\n/devtools/browser/owned'])
def test_startup_incomplete_descriptor_stops_at_original_deadline(tmp_path, startup_clock, contents):
    descriptor = tmp_path / 'DevToolsActivePort'
    if contents is not None:
        descriptor.write_text(contents, encoding='ascii')
    process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(NetworkExportError, match='startup exceeded the export timeout'):
        network_export._wait_for_debugging_port(descriptor, process, .12)
    assert startup_clock.now == .12
    assert startup_clock.sleeps == pytest.approx([.05, .05, .02])


@pytest.mark.parametrize('complete', [False, True])
def test_startup_browser_exit_rejects_even_present_descriptor(tmp_path, startup_clock, complete):
    descriptor = tmp_path / 'DevToolsActivePort'
    descriptor.write_text('43210\n/devtools/browser/owned' if complete else '', encoding='ascii')
    with pytest.raises(NetworkExportError, match='browser could not start'):
        network_export._wait_for_debugging_port(descriptor, SimpleNamespace(poll=lambda: 1), 1.)
    assert startup_clock.sleeps == []


def test_startup_browser_exit_during_partial_write_stops_waiting(tmp_path, startup_clock):
    descriptor = tmp_path / 'DevToolsActivePort'
    descriptor.write_text('432', encoding='ascii')
    process = SimpleNamespace(poll=lambda: 1 if startup_clock.sleeps else None)
    with pytest.raises(NetworkExportError, match='browser could not start'):
        network_export._wait_for_debugging_port(descriptor, process, 1.)
    assert startup_clock.sleeps == [.05]


def test_startup_descriptor_permission_error_is_not_retried(startup_clock):
    def read_text(**kwargs):
        raise PermissionError('cannot read owned descriptor')
    descriptor = SimpleNamespace(read_text=read_text)
    with pytest.raises(PermissionError, match='cannot read owned descriptor'):
        network_export._wait_for_debugging_port(descriptor, SimpleNamespace(poll=lambda: None), 1.)
    assert startup_clock.sleeps == []


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
def test_shutdown_stops_profile_writer_after_launcher_has_exited(tmp_path):
    profile = tmp_path / 'profile'
    writer = '''
import pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / 'ready').write_text('ready')
while True:
    (root / 'counter').write_text(str(time.monotonic_ns()))
    time.sleep(.001)
'''
    launcher = '''
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])
deadline = time.monotonic() + 5
while not (pathlib.Path(sys.argv[1]) / 'ready').exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('Writer did not start')
    time.sleep(.01)
'''
    with (tmp_path / 'browser.log').open('wb') as log:
        process = network_export._start_browser(
            [sys.executable, '-c', launcher, str(profile), writer],
            log, tmp_path / 'browser-state.json')
        try:
            deadline = time.monotonic() + 10
            while network_export._browser_exit_code(process) is None:
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert network_export._browser_exit_code(process) == 0
            assert process.returncode is None
            assert os.getpgid(process.pid) == process.pid
            assert (profile / 'ready').exists()
            network_export._stop_browser(process)
            # Deleting the profile must not race a surviving writer recreating files.
            network_export._remove_profile(profile)
            time.sleep(.05)
            assert not profile.exists()
        finally:
            if process.returncode is None:
                network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group ownership')
def test_shutdown_never_signals_an_unowned_or_reaped_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'killpg', lambda *args: calls.append(args))
    process = SimpleNamespace(pid=123, returncode=None, _openecon_group_anchor=False)
    with pytest.raises(NetworkExportError, match='not owned'):
        network_export._stop_browser(process)
    # Preserve an original startup failure after its dead supervisor was reaped.
    network_export._stop_browser(SimpleNamespace(pid=123, returncode=1, poll=lambda: 1, _openecon_group_anchor=True))
    assert not calls


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group ownership')
def test_shutdown_propagates_owned_group_permission_failure(monkeypatch):
    def denied(*args):
        raise PermissionError('owned group cannot be signalled')
    monkeypatch.setattr(os, 'killpg', denied)
    monkeypatch.setattr(os, 'getpgid', lambda pid: pid)
    process = SimpleNamespace(pid=123, returncode=None, poll=lambda: None, _openecon_group_anchor=True)
    with pytest.raises(PermissionError, match='cannot be signalled'):
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Frozen spawn early-bootstrap group ownership')
def test_shutdown_before_new_session_never_signals_inherited_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'getpgid', lambda pid: 999)
    monkeypatch.setattr(os, 'killpg', lambda *args: calls.append(('group', args)))
    process = SimpleNamespace(pid=123, returncode=None, _openecon_group_anchor=True,
                              poll=lambda: None,
                              kill=lambda: calls.append(('owned_child', 123)),
                              wait=lambda **kwargs: calls.append(('wait', kwargs)))
    network_export._stop_browser(process)
    assert calls == [('owned_child', 123), ('wait', {'timeout': 5})]


@pytest.mark.skipif(os.name != 'posix', reason='Frozen spawn reaped-group ownership')
def test_frozen_handle_syncs_child_reaped_by_multiprocessing_global_cleanup(monkeypatch):
    context = multiprocessing.get_context('spawn')
    worker = context.Process(target=time.sleep, args=(0.,))
    trigger = context.Process(target=time.sleep, args=(0.,))
    worker.start()
    handle = network_export._FrozenBrowserProcess(worker)
    handle._openecon_group_anchor = True
    try:
        time.sleep(.5)
        # Starting another Process invokes multiprocessing's global cleanup,
        # outside this handle. Observe its already-cached returncode directly.
        trigger.start()
        trigger.join(timeout=5)
        assert trigger.exitcode == 0
        assert worker._popen.returncode == 0
        assert handle.returncode is None
        monkeypatch.setattr(os, 'getpgid', lambda pid: pytest.fail('Reaped child cannot admit a group'))
        monkeypatch.setattr(os, 'killpg', lambda *args: pytest.fail('Reaped group cannot be signalled'))
        network_export._stop_browser(handle)
        assert handle.returncode == 0
    finally:
        if handle.returncode is None:
            worker.join(timeout=5)
            worker.close()
        if trigger.pid is not None:
            trigger.join(timeout=5)
            trigger.close()


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX browser supervisor')
def test_failed_supervisor_startup_preserves_original_failure_and_never_launches_browser(tmp_path):
    marker = tmp_path / 'browser-started'
    with (tmp_path / 'browser.log').open('wb') as log:
        process = network_export._start_browser(
            [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', str(marker)],
            log, tmp_path / 'missing-parent' / 'browser-state.json')
        try:
            with pytest.raises(NetworkExportError, match='browser could not start'):
                network_export._wait_for_debugging_port(tmp_path / 'missing-port', process, time.monotonic() + 3)
            assert process.returncode is not None
            network_export._stop_browser(process)
            assert not marker.exists()
        finally:
            if process.returncode is None:
                network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX browser supervisor')
def test_supervised_native_browser_failure_stops_startup_without_full_timeout(tmp_path):
    with (tmp_path / 'browser.log').open('wb') as log:
        process = network_export._start_browser(
            [sys.executable, '-c', 'raise SystemExit(7)'], log, tmp_path / 'browser-state.json')
        try:
            deadline = time.monotonic() + 3
            with pytest.raises(NetworkExportError, match='browser could not start'):
                network_export._wait_for_debugging_port(tmp_path / 'missing-port', process, deadline)
            assert time.monotonic() < deadline
            assert network_export._browser_exit_code(process) == 7
        finally:
            network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX browser supervisor')
def test_supervisor_removes_live_profile_writer_when_export_caller_dies(tmp_path):
    caller = r'''
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1])
from openecon_charts import network_export
root = Path(sys.argv[2])
writer = """
from pathlib import Path
import signal, sys, time
root = Path(sys.argv[1])
root.mkdir()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / 'counter').write_text('started')
(root / 'ready').write_text('ready')
while True:
    (root / 'counter.tmp').write_text(str(time.monotonic_ns()))
    (root / 'counter.tmp').replace(root / 'counter')
    time.sleep(.001)
"""
with (root / 'browser.log').open('wb') as log:
    process = network_export._start_browser([sys.executable, '-c', writer, str(root / 'profile')],
                                            log, root / 'browser-state.json')
    deadline = time.monotonic() + 5
    while not (root / 'profile' / 'ready').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned profile writer did not start')
        time.sleep(.01)
# Deliberately exit without calling cleanup: the supervisor must stop its group.
'''
    package_root = str(Path(network_export.__file__).resolve().parents[1])
    subprocess.run([sys.executable, '-I', '-c', caller, package_root, str(tmp_path)],
                   check=True, timeout=10)
    profile = tmp_path / 'profile'
    deadline = time.monotonic() + 3
    while True:
        before = (profile / 'counter').read_text()
        time.sleep(.05)
        if before and before == (profile / 'counter').read_text():
            break
        assert time.monotonic() < deadline, 'Orphaned browser profile writer survived'
    network_export._remove_profile(profile)
    time.sleep(.1)
    assert not profile.exists()


def test_profile_cleanup_retries_only_transient_nonempty_directory(tmp_path, monkeypatch):
    profile = tmp_path / 'profile'
    profile.mkdir()
    remove = network_export.shutil.rmtree
    attempts = []

    def transient(path):
        attempts.append(path)
        if len(attempts) < 3:
            raise OSError(errno.ENOTEMPTY, 'Profile write finishing', str(path))
        remove(path)

    monkeypatch.setattr(network_export.shutil, 'rmtree', transient)
    monkeypatch.setattr(network_export.time, 'sleep', lambda _: None)
    network_export._remove_profile(profile)
    assert len(attempts) == 3 and not profile.exists()


@pytest.mark.parametrize('error', [errno.ENOTEMPTY, errno.EACCES])
def test_profile_cleanup_never_hides_persistent_errors(tmp_path, monkeypatch, error):
    calls = []

    def denied(path):
        calls.append(path)
        raise OSError(error, 'Profile cannot be removed', str(path))

    monkeypatch.setattr(network_export.shutil, 'rmtree', denied)
    monkeypatch.setattr(network_export.time, 'sleep', lambda _: None)
    with pytest.raises(OSError) as exc:
        network_export._remove_profile(tmp_path / 'profile')
    assert exc.value.errno == error
    assert len(calls) == (5 if error == errno.ENOTEMPTY else 1)
