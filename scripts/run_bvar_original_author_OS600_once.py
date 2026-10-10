"""External hard600 watchdog using the reviewed nonreaping SDK group owner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / 'scripts/verify_bvar_chan2020_original_author.py'
LIFECYCLE = ROOT / 'scripts/run_parallel_sdk_groups.py'
LIFECYCLE_SHA256 = 'c4e4fae124766f30a316f69637d253d14e978d9cb4e60e7646dd470bebe07615'
OS_SECONDS = 600
REAP_SECONDS = 15


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def require(value, text):
    if not value:
        raise ValueError(text)


def ownership():
    require(pin(LIFECYCLE)['sha256'] == LIFECYCLE_SHA256,
            'Reviewed full nonreaping SDK lifecycle source differs')
    spec = importlib.util.spec_from_file_location('original_author_owned_lifecycle', LIFECYCLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(pid, timeout):
    raw = subprocess.check_output(['/bin/ps', '-A', '-o', 'pid=,ppid=,pgid=,stat='],
                                  text=True, timeout=timeout)
    rows = []
    for line in raw.splitlines():
        fields = line.split()
        require(len(fields) == 4 and all(value.isdecimal() for value in fields[:3]),
                'Invalid final complete process metadata')
        if int(fields[2]) == pid:
            rows.append(fields)
    return {'complete_actual_ps': raw, 'actual_original_group_rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--octave', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--ROOT-authorized-original-author-comparison', action='store_true', dest='authorized')
    args = parser.parse_args()
    require(args.authorized, 'Separate ROOT review and once execution authorization required')
    output = args.output.resolve()
    supervisor = output.with_name(output.name + '-OS-supervisor')
    require(not output.exists() and not supervisor.exists(), 'Both outputs must be fresh; no retry')
    # Keep the venv executable alias. Its resolved physical binary is pinned
    # separately; using that binary as argv[0] would discard venv sys.prefix.
    python = args.python.absolute()
    owned = ownership()
    supervisor.mkdir(parents=True)
    argv = [sys.executable, '-I', '-B', str(CHILD), '--output', str(output),
            '--python', str(python), '--octave', str(args.octave.resolve()),
            '--source-commit', args.source_commit, '--ROOT-authorized-original-author-comparison']
    started = time.monotonic()
    deadline = started + OS_SECONDS
    terminal = {'schema': 'openecon.original-author-Octave.external-OS-once.v2',
                'status': 'FAILED_OR_INCOMPLETE', 'argv': argv,
                'supervisor_source': pin(Path(__file__).resolve()), 'child_source': pin(CHILD),
                'reviewed_nonreaping_lifecycle_source': pin(LIFECYCLE),
                'python_executable_alias': str(python), 'resolved_python_binary': pin(python.resolve()),
                'started_monotonic': started, 'OS_seconds_cap': OS_SECONDS,
                'cleanup_reap_seconds_cap': REAP_SECONDS, 'no_retry': True}
    process = None
    try:
        with (supervisor / 'actual-worker.stdout').open('xb') as stdout, (supervisor / 'actual-worker.stderr').open('xb') as stderr:
            process = owned._launch_sdk(argv, cwd=ROOT, stdout=stdout, stderr=stderr)
            terminal['actual_child_PID'] = process.pid
            terminal['actual_owned_PGID'] = process.pid
            # Do not call Popen.wait/poll until the retained anchor has proved
            # every live group recipient stopped. The existing owner reaps.
            while (code := owned._sdk_exit(process)) is None:
                require(time.monotonic() < deadline, 'Original-author hard600 deadline exhausted; no retry')
                time.sleep(min(.05, max(0., deadline - time.monotonic())))
            terminal.update(observed_unreaped_child_exit=code, timed_out=False,
                            child_exit_observed_monotonic=time.monotonic())
            require(code == 0, 'Original-author process genuinely failed; no retry')
            live = owned._live_sdk_group(process, min(deadline, time.monotonic() + 2))
            terminal['actual_live_recipients_before_reap'] = live
            require(not live, 'Successful child left a live inherited process; no acceptance')
            owned._stop([process])
            terminal.update(OS_exit=process.returncode,
                            child_exited_and_reaped=owned._sdk_stopped(process))
        terminal['child_finished_monotonic'] = time.monotonic()
        terminal['child_seconds'] = terminal['child_finished_monotonic'] - started
        require(process.returncode == 0 and owned._sdk_stopped(process), 'Actual child was not successfully reaped')
        after = snapshot(process.pid, min(2., max(.001, deadline - time.monotonic())))
        terminal['complete_after_owned_group_readback'] = after
        require(not after['actual_original_group_rows'], 'Original owned PID/PGID not absent after reap')
        result_path = output / 'complete-once-reference-terminal.json'
        result = json.loads(result_path.read_bytes())
        require(result['status'] == 'PASS_ORIGINAL_AUTHOR_OCTAVE_DETERMINISTIC_POSTERIOR_THREE_CASES'
                and result['all_source_bytes_preserved'] is True,
                'Complete genuine child source/reference terminal required')
        terminal['actual_complete_child_terminal_pin'] = pin(result_path)
        require(time.monotonic() < deadline, 'Complete reference/group readback exceeded hard600 seconds')
        terminal['status'] = 'OS0_REAPED_GROUP_ABSENT_WITHIN600_COMPLETE_REFERENCE_PASS'
    except BaseException as exc:
        terminal['failure'] = {'type': type(exc).__name__, 'message': str(exc),
                               'complete_traceback': traceback.format_exc()}
        terminal['timed_out'] = time.monotonic() >= deadline
        terminal['status'] = 'FAILED_OR_INCOMPLETE'
        raise
    finally:
        cleanup_started = time.monotonic()
        try:
            if process is not None and not owned._sdk_stopped(process):
                owned._stop([process])
            if process is not None:
                terminal.update(OS_exit=process.returncode,
                                child_exited_and_reaped=owned._sdk_stopped(process))
                terminal['complete_final_owned_group_readback'] = snapshot(process.pid, 2)
                require(not terminal['complete_final_owned_group_readback']['actual_original_group_rows'],
                        'Original owned PID/PGID remains after bounded cleanup')
            require(time.monotonic() - cleanup_started <= REAP_SECONDS, 'Cleanup exceeded15second allowance')
            if terminal['status'] == 'OS0_REAPED_GROUP_ABSENT_WITHIN600_COMPLETE_REFERENCE_PASS':
                require(time.monotonic() < deadline, 'Final actual group readback exceeded hard600 seconds')
        except BaseException as exc:
            terminal['status'] = 'FAILED_OR_INCOMPLETE'
            terminal['cleanup_failure'] = {'type': type(exc).__name__, 'message': str(exc),
                                           'complete_traceback': traceback.format_exc()}
            raise
        finally:
            terminal['supervisor_finished_monotonic'] = time.monotonic()
            terminal['cleanup_seconds'] = time.monotonic() - cleanup_started
            terminal['actual_log_pins'] = [pin(path) for path in sorted(supervisor.glob('*.stdout'))] + [pin(path) for path in sorted(supervisor.glob('*.stderr'))]
            with (supervisor / 'complete-external-OS-terminal.json').open('x') as stream:
                json.dump(terminal, stream, allow_nan=False, indent=2)
                stream.write('\n')


if __name__ == '__main__':
    main()
