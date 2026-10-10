"""Real headless exports: completed layout, typed view, fonts and atomic refusal."""
from copy import deepcopy
import errno
import hashlib
import json
import multiprocessing
import subprocess
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from openecon_charts import network
from openecon_charts.network_export import NetworkExportError, _browser
from openecon_charts import network_export


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
def test_real_offline_exports_and_exclusive_publication(tmp_path, browser, format, monkeypatch):
    plot = chart(layout='circular', layout_options={'radius': 70})
    original = deepcopy(plot.model_dump())
    created_profiles = []
    temporary_directory = tempfile.TemporaryDirectory

    def owned_directory(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        created_profiles.append(Path(directory.name))
        return directory

    # Observe only this export's profiles; another shard can legitimately
    # create or remove its own profile in the shared system temporary folder.
    monkeypatch.setattr(network_export, 'tempfile', SimpleNamespace(
        TemporaryDirectory=owned_directory, mkstemp=tempfile.mkstemp))
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
    assert created_profiles and not any(path.exists() for path in created_profiles)
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
def test_shutdown_stops_profile_writer_after_launcher_has_exited(tmp_path, monkeypatch):
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
    process = network_export._launch_browser(
        [sys.executable, '-c', launcher, str(profile), writer])
    signals = []
    killpg = os.killpg

    def owned_signal(group, signum):
        assert group == process.pid and process.returncode is None
        assert network_export._browser_exit(process) == 0
        signals.append(signum)
        killpg(group, signum)

    try:
        deadline = time.monotonic() + 10
        while network_export._browser_exit(process) is None:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert network_export._browser_exit(process) == 0
        assert process.returncode is None  # The exited leader still anchors its group ID.
        assert (profile / 'ready').exists()
        with monkeypatch.context() as patch:
            patch.setattr(network_export.os, 'killpg', owned_signal)
            network_export._stop_browser(process)
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        assert process.returncode == 0
        # Deleting the profile must not race a surviving writer recreating files.
        network_export._remove_profile(profile)
        time.sleep(.05)
        assert not profile.exists()
    finally:
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
def test_exited_owned_launcher_is_reaped_without_signaling_empty_group(monkeypatch):
    process = network_export._launch_browser([sys.executable, '-c', 'pass'])
    try:
        deadline = time.monotonic() + 5
        while network_export._browser_exit(process) is None:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert process.returncode is None
        assert network_export._live_browser_group(process, deadline) == []

        def unsafe_signal(*args):
            pytest.fail('No live writer remains; group signaling is unnecessary')

        with monkeypatch.context() as patch:
            patch.setattr(network_export.os, 'killpg', unsafe_signal)
            network_export._stop_browser(process)
            assert process.returncode == 0
            # A later call cannot signal a group whose ID may have been reused.
            network_export._stop_browser(process)
    finally:
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
@pytest.mark.parametrize('error', [PermissionError, ProcessLookupError])
def test_signal_race_accepts_only_fresh_no_live_recipient_proof(monkeypatch, error):
    process = network_export._launch_browser(
        [sys.executable, '-c', 'import time; time.sleep(30)'])
    killpg = os.killpg
    calls = []

    def exited_recipient(group, signum):
        assert group == process.pid and process.returncode is None
        calls.append(signum)
        killpg(group, signum)
        deadline = time.monotonic() + 5
        while network_export._browser_exit(process) is None:
            assert time.monotonic() < deadline
            time.sleep(.01)
        raise error('Last recipient exited between inspection and signal')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(network_export.os, 'killpg', exited_recipient)
            network_export._stop_browser(process)
        assert calls == [signal.SIGTERM]
        assert process.returncode == -signal.SIGTERM
    finally:
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
@pytest.mark.parametrize('error', [PermissionError, ProcessLookupError])
def test_denied_signal_with_live_owned_recipient_is_never_hidden(monkeypatch, error):
    process = network_export._launch_browser(
        [sys.executable, '-c', 'import time; time.sleep(30)'])
    calls = []

    def denied(group, signum):
        calls.append((group, signum))
        raise error('Owned live recipient did not accept the signal')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(network_export.os, 'killpg', denied)
            with pytest.raises(error, match='did not accept'):
                network_export._stop_browser(process)
        assert calls == [(process.pid, signal.SIGTERM)]
        assert process.returncode is None and network_export._browser_exit(process) is None
    finally:
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
@pytest.mark.parametrize('external_reap', [False, True])
def test_reaped_launcher_refuses_group_signal_even_if_pid_were_reused(monkeypatch, external_reap):
    process = network_export._launch_browser([sys.executable, '-c', 'pass'])
    if external_reap:
        assert os.waitpid(process.pid, 0) == (process.pid, 0)
    else:
        assert process.wait(timeout=5) == 0

    def unsafe_signal(*args):
        pytest.fail('Reaped leader cannot certify ownership of a reused group ID')

    with monkeypatch.context() as patch:
        patch.setattr(network_export.os, 'killpg', unsafe_signal)
        with pytest.raises(NetworkExportError, match='reaped|ownership was lost'):
            network_export._stop_browser(process)
    process.wait(timeout=5)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
def test_unregistered_launcher_never_authorizes_group_signal(monkeypatch):
    def unsafe_signal(*args):
        pytest.fail('An arbitrary process handle cannot authorize a group signal')

    monkeypatch.setattr(network_export.os, 'killpg', unsafe_signal)
    with pytest.raises(NetworkExportError, match='owned dedicated session'):
        network_export._stop_browser(SimpleNamespace(pid=os.getpid(), returncode=None))


@pytest.mark.skipif(os.name != 'posix', reason='Owned POSIX exit observation')
@pytest.mark.parametrize('code', [0, 7, -signal.SIGTERM])
def test_native_exit_observation_matches_status_without_reaping(monkeypatch, code):
    program = (f'import os; os._exit({code})' if code >= 0 else
               f'import os, signal; os.kill(os.getpid(), {-code})')
    process = network_export._launch_browser([sys.executable, '-c', program])
    builtin = getattr(os, 'waitid', None)
    try:
        with monkeypatch.context() as patch:
            if sys.platform == 'darwin':
                patch.delattr(network_export.os, 'waitid', raising=False)
            deadline = time.monotonic() + 5
            while network_export._browser_exit(process) is None:
                assert time.monotonic() < deadline
                time.sleep(.01)
            observed = network_export._owned_exit_status(process.pid)
            assert network_export._browser_exit(process) == code
            assert process.returncode is None
        if builtin is not None:
            result = builtin(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            assert observed == (result.si_pid, result.si_code, result.si_status)
        network_export._stop_browser(process)
        assert process.returncode == code
    finally:
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
def test_missing_nonreaping_observation_refuses_before_process_launch(monkeypatch):
    def unsafe_launch(*args, **kwargs):
        pytest.fail('A browser cannot start without an ownership-preserving observer')

    monkeypatch.delattr(network_export.os, 'waitid', raising=False)
    monkeypatch.setattr(network_export.sys, 'platform', 'unavailable-posix')
    monkeypatch.setattr(network_export.subprocess, 'Popen', unsafe_launch)
    with pytest.raises(NetworkExportError, match='non-reaping exit observation'):
        network_export._launch_browser(['browser'])


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
@pytest.mark.parametrize('inspection', ['wrong_group', 'missing_live_leader', 'malformed'])
def test_invalid_group_inspection_refuses_before_signaling(monkeypatch, inspection):
    process = network_export._launch_browser(
        [sys.executable, '-c', 'import time; time.sleep(30)'])
    output = {'wrong_group': f'{process.pid} {os.getpgrp()} S\n',
              'missing_live_leader': '', 'malformed': 'not a process record\n'}[inspection]

    def unsafe_signal(*args):
        pytest.fail('Invalid process-group evidence cannot authorize a signal')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(network_export.subprocess, 'check_output', lambda *args, **kw: output)
            patch.setattr(network_export.os, 'killpg', unsafe_signal)
            if inspection == 'missing_live_leader':
                clock = iter(range(100))
                patch.setattr(network_export.time, 'monotonic', lambda: next(clock))
                patch.setattr(network_export.time, 'sleep', lambda _: None)
                with pytest.raises(network_export.subprocess.TimeoutExpired,
                                   match='owned browser group inspection'):
                    network_export._stop_browser(process)
            else:
                with pytest.raises(NetworkExportError, match='group|inspection'):
                    network_export._stop_browser(process)
    finally:
        network_export._stop_browser(process)


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


@pytest.mark.parametrize('exit_after_first_snapshot', [None, 0])
def test_empty_snapshot_waits_for_exit_then_inspects_children_again(monkeypatch, exit_after_first_snapshot):
    process = SimpleNamespace(pid=101, returncode=None)
    observations = iter([None, exit_after_first_snapshot, 0, 0])
    snapshots = iter(['101 101 Z\n', '101 101 Z\n102 101 S\n'])
    observed = []

    def exit_status(owned):
        assert owned is process and owned.returncode is None
        value = next(observations)
        observed.append(value)
        return value

    monkeypatch.setattr(network_export, '_browser_exit', exit_status)
    monkeypatch.setattr(network_export.subprocess, 'check_output',
                        lambda *args, **kwargs: next(snapshots))
    monkeypatch.setattr(network_export.time, 'sleep', lambda _: None)
    assert network_export._live_browser_group(process, time.monotonic() + 5) == [102]
    assert observed == [None, exit_after_first_snapshot, 0, 0]
    assert process.returncode is None


def test_profile_observation_preserves_another_export_owner(tmp_path, browser, monkeypatch):
    created = []
    factory = tempfile.TemporaryDirectory
    with factory(prefix='openecon-network-export-') as other_name:
        other = Path(other_name)
        sentinel = other / 'owned-by-another-export.txt'
        sentinel.write_bytes(b'another export remains active')

        def observed(*args, **kwargs):
            directory = factory(*args, **kwargs)
            created.append(Path(directory.name))
            return directory

        monkeypatch.setattr(network_export, 'tempfile', SimpleNamespace(
            TemporaryDirectory=observed, mkstemp=tempfile.mkstemp))
        chart(layout='circular').to_svg(tmp_path / 'owned.svg', browser_executable=browser)
        assert created and not any(path.exists() for path in created)
        assert sentinel.read_bytes() == b'another export remains active'


@pytest.mark.skipif(os.name != 'posix', reason='Ownership-preserving POSIX group signaling')
@pytest.mark.parametrize('boundary', ['during_ps', 'before_signal'])
@pytest.mark.parametrize('change', ['external_reap', 'spent_deadline'])
def test_changed_ownership_or_spent_snapshot_never_authorizes_signal(monkeypatch, boundary, change):
    process = SimpleNamespace(pid=101, returncode=None, _openecon_owned_session=True,
                              _openecon_session_closed=False)
    clock = {'now': 0.}
    monkeypatch.setattr(network_export, '_owned_exit_status', lambda _: None)
    monkeypatch.setattr(network_export.time, 'monotonic', lambda: clock['now'])

    def change_boundary():
        if change == 'external_reap':
            process.returncode = 0
        else:
            clock['now'] = 6.

    def snapshot(*args, **kwargs):
        change_boundary()
        return '101 101 S\n'

    def live_snapshot(*args):
        change_boundary()
        return [101]

    if boundary == 'during_ps':
        monkeypatch.setattr(network_export.subprocess, 'check_output', snapshot)
    else:
        monkeypatch.setattr(network_export, '_live_browser_group', live_snapshot)
    monkeypatch.setattr(network_export.os, 'killpg',
                        lambda *args: pytest.fail('Changed owner or spent snapshot cannot authorize a signal'))
    expected = (NetworkExportError if change == 'external_reap'
                else network_export.subprocess.TimeoutExpired)
    with pytest.raises(expected, match='reaped|owned browser group'):
        network_export._signal_browser_group(process, signal.SIGTERM, 5.)



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


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group teardown')
def test_supervised_shutdown_stops_profile_writer_after_launcher_has_exited(tmp_path):
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
    # A reaped supervisor cannot prove that SIGKILL ran its finally block.
    # No numeric PGID signal or successful profile-cleanup claim is allowed.
    with pytest.raises(NetworkExportError, match='reaped'):
        network_export._stop_browser(SimpleNamespace(pid=123, returncode=1, poll=lambda: 1, _openecon_group_anchor=True))
    assert not calls


@pytest.mark.skipif(os.name != 'posix', reason='Dedicated POSIX process-group ownership')
def test_shutdown_propagates_owned_group_permission_failure(monkeypatch):
    def denied(*args):
        raise PermissionError('owned group cannot be signalled')
    monkeypatch.setattr(os, 'killpg', denied)
    monkeypatch.setattr(os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(network_export, '_owned_exit_status', lambda pid: None)
    monkeypatch.setattr(network_export.subprocess, 'check_output', lambda *a, **k: '123 123 S\n')
    process = SimpleNamespace(pid=123, returncode=None, poll=lambda: None, _openecon_group_anchor=True)
    with pytest.raises(PermissionError, match='cannot be signalled'):
        network_export._stop_browser(process)


@pytest.mark.skipif(os.name != 'posix', reason='Frozen spawn early-bootstrap group ownership')
def test_shutdown_before_new_session_never_signals_inherited_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'getpgid', lambda pid: 999)
    monkeypatch.setattr(network_export, '_owned_exit_status', lambda pid: None)
    monkeypatch.setattr(network_export.time, 'monotonic', lambda: 0.)
    monkeypatch.setattr(os, 'kill', lambda pid, sig: calls.append(('owned_child', pid, sig)))
    monkeypatch.setattr(os, 'killpg', lambda *args: calls.append(('group', args)))
    process = SimpleNamespace(pid=123, returncode=None, _openecon_group_anchor=True,
                              poll=lambda: None,
                              kill=lambda: calls.append(('owned_child', 123)),
                              wait=lambda **kwargs: calls.append(('wait', kwargs)))
    network_export._stop_browser(process)
    assert calls == [('owned_child', 123, signal.SIGKILL), ('wait', {'timeout': 5})]


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
        with pytest.raises(NetworkExportError, match='reaped'):
            network_export._stop_browser(handle)
        assert handle.returncode == 0
    finally:
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
            assert network_export._browser_exit(process) is not None
            assert process.returncode is None
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


@pytest.fixture
def owned_supervisor_boundary(monkeypatch):
    clock = [0.]
    exited = [None]
    signals, waits = [], []
    process = SimpleNamespace(
        pid=123, returncode=None, _openecon_group_anchor=True,
        _openecon_owned_session=True, _openecon_session_closed=False,
        poll=lambda: pytest.fail('Supervisor shutdown cannot reap through poll'),
    )

    def wait(**kwargs):
        waits.append(kwargs)
        process.returncode = -signal.SIGKILL

    def observe(pid):
        assert pid == 123
        return None if exited[0] is None else (123, os.CLD_KILLED, exited[0])

    def inspect(*args, **kwargs):
        assert kwargs['timeout'] > 0
        return '123 123 S\n' if exited[0] is None else '123 123 Z\n'

    def send(pid, signum):
        assert pid == 123
        signals.append(signum)
        if signum == signal.SIGKILL:
            exited[0] = signum

    def sleep(seconds):
        assert seconds > 0
        clock[0] += seconds

    process.wait = wait
    monkeypatch.setattr(os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(os, 'killpg', send)
    monkeypatch.setattr(network_export, '_owned_exit_status', observe)
    monkeypatch.setattr(network_export.subprocess, 'check_output', inspect)
    monkeypatch.setattr(network_export.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(network_export.time, 'sleep', sleep)
    return SimpleNamespace(process=process, signals=signals, waits=waits,
                           clock=clock, send=send)


@pytest.mark.skipif(os.name != 'posix', reason='Owned supervisor error cleanup')
@pytest.mark.parametrize('status', [
    '{broken',
    '{"anchor_pid":true,"pgid":123,"browser_pid":124,"phase":"browser-started","returncode":null}',
    '{"anchor_pid":123,"pgid":123,"browser_pid":124,"phase":"unknown","returncode":null}',
    '{"anchor_pid":123,"pgid":123,"browser_pid":124,"phase":[],"returncode":null}',
    '{"anchor_pid":123,"pgid":123,"browser_pid":null,"phase":"native-exited","returncode":0}',
])
def test_invalid_supervisor_status_preserves_refusal_after_guarded_cleanup(
        tmp_path, monkeypatch, owned_supervisor_boundary, status):
    boundary = owned_supervisor_boundary
    state = tmp_path / 'browser-state.json'
    state.write_text(status)
    boundary.process._openecon_browser_state = state
    with pytest.raises(NetworkExportError, match='Invalid|Inconsistent|identity'):
        network_export._stop_browser(boundary.process)
    assert boundary.signals == [signal.SIGKILL]
    assert boundary.waits == [{'timeout': 5.}]
    assert boundary.process._openecon_session_closed
    network_export._stop_browser(boundary.process)
    assert boundary.signals == [signal.SIGKILL]


@pytest.mark.skipif(os.name != 'posix', reason='Owned supervisor TERM error cleanup')
def test_supervisor_term_permission_failure_keeps_error_and_stops_writers(
        monkeypatch, owned_supervisor_boundary):
    boundary = owned_supervisor_boundary
    failure = PermissionError('TERM permission was denied')

    def send(pid, signum):
        if signum == signal.SIGTERM:
            boundary.signals.append(signum)
            raise failure
        boundary.send(pid, signum)

    monkeypatch.setattr(os, 'killpg', send)
    monkeypatch.setattr(network_export, '_browser_state',
                        lambda process: {'phase': 'browser-started'})
    with pytest.raises(PermissionError) as caught:
        network_export._stop_browser(boundary.process)
    assert caught.value is failure
    assert boundary.signals == [signal.SIGTERM, signal.SIGKILL]
    assert boundary.waits == [{'timeout': 5.}]
    assert boundary.process._openecon_session_closed


@pytest.mark.skipif(os.name != 'posix', reason='Owned supervisor refused cleanup')
def test_supervisor_status_failure_and_external_reap_preserve_both_failures(
        monkeypatch, owned_supervisor_boundary):
    boundary = owned_supervisor_boundary
    failure = NetworkExportError('Native status read failed')

    def status(process):
        process.returncode = 0
        raise failure

    monkeypatch.setattr(network_export, '_browser_state', status)
    with pytest.raises(NetworkExportError) as caught:
        network_export._stop_browser(boundary.process)
    assert caught.value is failure
    assert isinstance(caught.value.__cause__, NetworkExportError)
    assert 'reaped' in str(caught.value.__cause__)
    assert not boundary.signals and not boundary.waits
    assert not boundary.process._openecon_session_closed


@pytest.mark.skipif(os.name != 'posix', reason='Owned supervisor native TERM completion')
@pytest.mark.parametrize('native_finishes', [False, True])
def test_supervisor_term_waits_for_native_status_with_original_two_deadlines(
        monkeypatch, owned_supervisor_boundary, native_finishes):
    boundary = owned_supervisor_boundary
    checks, deadlines = [], []
    actual_signal = network_export._signal_browser_group

    def native_status(process):
        checks.append(boundary.clock[0])
        return 0 if native_finishes and len(checks) >= 2 else None

    def signal_group(process, signum, deadline):
        deadlines.append((signum, deadline))
        actual_signal(process, signum, deadline)

    monkeypatch.setattr(network_export, '_browser_state',
                        lambda process: {'phase': 'browser-started'})
    monkeypatch.setattr(network_export, '_browser_exit_code', native_status)
    monkeypatch.setattr(network_export, '_signal_browser_group', signal_group)
    network_export._stop_browser(boundary.process)
    assert boundary.signals == [signal.SIGTERM, signal.SIGKILL]
    assert deadlines[0] == (signal.SIGTERM, 5.)
    assert deadlines[1] == (signal.SIGKILL, boundary.clock[0] + 5.)
    assert len(checks) >= 2
    if native_finishes:
        assert boundary.clock[0] == .01
    else:
        assert boundary.clock[0] == 5.
    assert boundary.waits == [{'timeout': 5.}]
    assert boundary.process._openecon_session_closed


@pytest.mark.skipif(os.name != 'posix', reason='Frozen cached exit ownership boundary')
@pytest.mark.parametrize('boundary', ['during_ps', 'before_signal'])
def test_frozen_cached_reap_cannot_authorize_supervisor_group_signal(monkeypatch, boundary):
    child = SimpleNamespace(returncode=None)

    class Worker:
        pid = 123
        _popen = child

        @property
        def exitcode(self):
            pytest.fail('Reading multiprocessing.exitcode would reap the group anchor')

    process = network_export._FrozenBrowserProcess(Worker())
    process.pid = 123
    process._openecon_group_anchor = True
    process._openecon_owned_session = True
    monkeypatch.setattr(network_export, '_owned_exit_status', lambda pid: None)
    monkeypatch.setattr(network_export.time, 'monotonic', lambda: 0.)
    monkeypatch.setattr(os, 'killpg', lambda *args: pytest.fail('Reaped group must not be signalled'))

    def inspect(*args, **kwargs):
        child.returncode = 0
        return '123 123 S\n'

    if boundary == 'during_ps':
        monkeypatch.setattr(network_export.subprocess, 'check_output', inspect)
    else:
        def live(*args):
            child.returncode = 0
            return [123]
        monkeypatch.setattr(network_export, '_live_browser_group', live)
    with pytest.raises(NetworkExportError, match='reaped'):
        network_export._signal_browser_group(process, signal.SIGKILL, 5.)
    assert process.returncode == 0


@pytest.mark.skipif(os.name != 'posix', reason='Frozen bootstrap external reap boundary')
def test_frozen_bootstrap_external_reap_never_signals_child_or_inherited_group(monkeypatch):
    child = SimpleNamespace(returncode=None)
    process = SimpleNamespace(pid=123, returncode=None, _openecon_group_anchor=True,
                              _process=SimpleNamespace(_popen=child))

    def group(pid):
        child.returncode = 0
        return 999

    monkeypatch.setattr(network_export, '_owned_exit_status', lambda pid: None)
    monkeypatch.setattr(network_export.time, 'monotonic', lambda: 0.)
    monkeypatch.setattr(os, 'getpgid', group)
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('Reaped bootstrap must not be signalled'))
    monkeypatch.setattr(os, 'killpg', lambda *args: pytest.fail('Inherited group must not be signalled'))
    with pytest.raises(NetworkExportError, match='reaped'):
        network_export._stop_browser(process)
