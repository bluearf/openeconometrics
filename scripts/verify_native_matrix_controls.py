"""Exercise timeout, storage, resource and unsupported-option refusal in an owned QA runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
from urllib.request import Request, urlopen
from uuid import uuid4

from verify_native_matrix_desktop import bundled_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    identity = bundled_identity(args.app)
    cases = {c["model"]: c for c in json.loads(args.manifest.read_text())["cases"]}
    records = []
    with tempfile.TemporaryDirectory(prefix="openecon-matrix-controls-") as temporary:
        runtime = args.app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        with (args.directory / "runtime.log").open("wb") as errors:
            process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", temporary],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                env=environment, start_new_session=True)
            try:
                assert select.select([process.stdout], [], [], 60)[0], "Runtime did not start"
                descriptor = json.loads(process.stdout.readline())
                assert descriptor["type"] == "ready" and descriptor["url"].startswith("http://127.0.0.1:")
                base = descriptor["url"] + f"/api/desktop/projects/{uuid4().hex}/workspace"
                token = json.load(urlopen(base + "/session"))["token"]

                def call(path, body=None):
                    return json.load(urlopen(Request(base + path,
                        headers={"Content-Type": "application/json", "X-OpenEcon-Token": token},
                        data=json.dumps(body).encode() if body is not None else None), timeout=120))

                probe = """
import sys, json, os
from pathlib import Path
import openecon as oe
assert getattr(sys, 'frozen', False)
assert str(oe.__file__).startswith(str(sys._MEIPASS))
assert not any(n == 'scipy' or n.startswith('scipy.') for n in sys.modules)
"""
                timeout = call("/console/execute", {"code": probe + "\nimport time\nprint('TIMEOUT_STARTED', flush=True)\ntime.sleep(20)", "timeout_seconds": .5})
                assert timeout["status"] == "timeout", timeout.get("error")
                assert not list(Path(temporary).rglob(".openecon-scratch-*"))
                records.append({"control": "console_timeout", "status": "passed", "run": timeout,
                                "worker_scratch_removed": True})
                guards = probe + """
import hashlib
import pandas as pd
from openecon.analysis_contracts import AnalysisError
from openecon.data import DataError
from openecon.resources import use_workspace_budget
controls = []
frame = pd.DataFrame({'x': [i / 500 for i in range(500)],
                      **{'y'+str(j): [i / 500 + j for i in range(500)] for j in range(3)}})
moments = ['y'+str(i)+' - '+ ' - '.join('{b'+str(i)+'_'+str(j)+'}*x' for j in range(100)) for i in range(3)]
try:
    with use_workspace_budget(1):
        oe.gmm(data=frame, moments=moments, instruments=['x'])
except AnalysisError as error:
    assert error.code == 'workspace_limit', error
    controls.append({'control': 'joint_gmm_resource_plan', 'code': error.code,
                     'resource_plan': error.resource_plan,
                     'scope': 'preallocation buffer plan; not a process RSS limit'})
else:
    raise AssertionError('Resource limit was ignored')
"""
                for name, control in (("rreg", "storage"), ("csdid", "unsupported")):
                    case = cases[name]
                    guards += f"\ncase = {case!r}\n" + """
source = Path(case['path'])
before = hashlib.sha256(source.read_bytes()).hexdigest()
spec = oe.ModelSpec.model_validate(case['spec'])
dataset = oe.scan(source)
"""
                    if control == "storage":
                        guards += """
blocker = Path(os.environ['OPENECON_SCRATCH_DIRECTORY']) / 'owned-blocker'
blocker.write_text('owned storage guard')
previous = os.environ['OPENECON_SCRATCH_DIRECTORY']
os.environ['OPENECON_SCRATCH_DIRECTORY'] = str(blocker)
try:
    try:
        oe.fit(spec, data=dataset)
    except (NotADirectoryError, FileExistsError, DataError) as error:
        assert 'NotADirectoryError' in str(error) or isinstance(error, (NotADirectoryError, FileExistsError)), error
        controls.append({'control': 'nonlinear_scratch_parent_is_file', 'type': type(error).__name__, 'message': str(error)})
    else:
        raise AssertionError('Blocked scratch was ignored')
finally:
    os.environ['OPENECON_SCRATCH_DIRECTORY'] = previous
assert blocker.read_text() == 'owned storage guard'
blocker.unlink()
"""
                    else:
                        guards += """
spec.options['bootstrap_reps'] = 9
try:
    oe.fit(spec, data=dataset)
except AnalysisError as error:
    assert error.code == 'streaming_options_unsupported', error
    controls.append({'control': 'csdid_bootstrap_refusal', 'code': error.code, 'message': str(error)})
else:
    raise AssertionError('Unsupported bootstrap was accepted')
"""
                    guards += "assert hashlib.sha256(source.read_bytes()).hexdigest() == before\n"
                guards += "print('NATIVE_MATRIX_CONTROLS:' + json.dumps(controls))\n"
                result = call("/console/execute", {"code": guards, "timeout_seconds": 90})
                assert result["status"] == "ok", result.get("error")
                measured = json.loads(result["stdout"].split("NATIVE_MATRIX_CONTROLS:", 1)[1])
                assert len(measured) == 3
                records.extend({**record, "status": "passed"} for record in measured)
                recovery = call("/console/execute", {"code": "print('RECOVERED_AFTER_CONTROLS')", "timeout_seconds": 10})
                assert recovery["status"] == "ok" and recovery["stdout"].strip() == "RECOVERED_AFTER_CONTROLS"
                call("/console/reset", {})
                history = call("/console")["history"]
                assert next(r for r in history if r["id"] == result["id"])["stdout"] == result["stdout"]
                assert not list(Path(temporary).rglob(".openecon-scratch-*"))
                evidence = {"identity": identity, "controls": records, "guard_run": result,
                            "recovery": recovery, "history_readback": True,
                            "owned_worker_scratch_removed": True, "human_project_access": False}
            finally:
                if process.poll() is None:
                    try:
                        process.stdin.write(b'{"type":"shutdown"}\n')
                        process.stdin.flush()
                        process.wait(timeout=12)
                    except (BrokenPipeError, subprocess.TimeoutExpired):
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
                process.stdin.close()
                process.stdout.close()
    evidence["owned_runtime_and_project_removed"] = not Path(temporary).exists()
    (args.directory / "controls.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"passed_controls": len(records), "recovery": "passed"}))


if __name__ == "__main__":
    main()
