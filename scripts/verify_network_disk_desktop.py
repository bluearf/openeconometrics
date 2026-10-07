"""Replay disk analysis in an isolated frozen Mac worker and read saved outputs.

No source path is injected into the worker; the installed application and human
projects are untouched. Browser rendering and release installation are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import subprocess
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify(runtime):
    runtime = Path(runtime).resolve(strict=True)
    example = ROOT / "docs/examples/network_disk_analysis.py"
    fingerprint = digest(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-disk-frozen-") as temporary:
        root = Path(temporary)
        with (root / "stderr.log").open("wb") as errors:
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                env=environment, start_new_session=True)
            descriptor, token = None, None
            prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"

            def call(path, body=None):
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["X-OpenEcon-Token"] = token
                request = Request(descriptor["url"] + prefix + path, headers=headers,
                    data=json.dumps(body).encode() if body is not None else None)
                with urlopen(request, timeout=180) as response:
                    return json.load(response)

            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if select.select([process.stdout], [], [], .25)[0]:
                        descriptor = json.loads(process.stdout.readline(8193))
                        break
                    if process.poll() is not None:
                        raise RuntimeError("Disposable frozen worker exited before ready")
                if not descriptor or descriptor.get("type") != "ready":
                    raise RuntimeError("Disposable frozen worker did not become ready")
                parsed = urlparse(descriptor["url"])
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != descriptor["port"]:
                    raise RuntimeError("Invalid frozen loopback descriptor")
                token = call("/session")["token"]
                code = '''
import json
import sys
from pathlib import Path
import openecon as oe
import openecon._network_disk_algorithms as kernels
assert getattr(sys, "frozen", False)
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
assert Path(kernels.__file__).is_relative_to(Path(sys._MEIPASS))
''' + example.read_text() + '''
disk_analysis_proof["source_runtime_only"] = False
disk_analysis_proof["frozen_execution"] = True
disk_analysis_proof["disk_kernels_from_bundle"] = True
disk_analysis_proof["sdk_from_bundle"] = True
print("FROZEN_DISK_ANALYSIS:" + json.dumps(disk_analysis_proof, allow_nan=False))
'''
                run = call("/console/execute", {"code": code, "timeout_seconds": 180})
                if run.get("status") != "ok":
                    raise RuntimeError(f"Frozen disk analysis failed: {run.get('error')}")
                markers = [json.loads(line.removeprefix("FROZEN_DISK_ANALYSIS:"))
                           for line in run["stdout"].splitlines() if line.startswith("FROZEN_DISK_ANALYSIS:")]
                if len(markers) != 1 or not markers[0]["resident_guarded"]:
                    raise RuntimeError("Missing unique frozen capacity/numerical proof")
                if [out["type"] for out in run["outputs"]] != ["table"] * 4:
                    raise RuntimeError("Missing four ordered frozen panel tables")
                if not all("\\begin{tabular}" in out["latex"] for out in run["outputs"]):
                    raise RuntimeError("Frozen panel tables lack publication LaTeX")
                call("/console/reset", {})
                saved = next(item for item in call("/console")["history"] if item["id"] == run["id"])
                if saved["outputs"] != run["outputs"] or saved["events"] != run["events"]:
                    raise RuntimeError("Frozen outputs changed after worker reset")
                if digest(runtime) != fingerprint:
                    raise RuntimeError("Frozen runtime changed during verification")
                record = dict(status="passed", captured_at_utc=datetime.now(timezone.utc).isoformat(),
                    platform=platform.platform(), machine=platform.machine(), proof=markers[0],
                    runtime_executable_sha256=fingerprint, example_sha256=digest(example),
                    verifier_sha256=digest(__file__), console_duration_ms=run["duration_ms"],
                    ordered_outputs=["table"] * 4, output_rows=[out["data"]["total_rows"] for out in run["outputs"]],
                    history_after_worker_reset=True, human_data_access=False,
                    installation_changed=False, browser_render_verified=False, release_delivered=False,
                    source_sha256={name: digest(ROOT / name) for name in (
                        "src/openecon/_network_store.py", "src/openecon/_network_disk_algorithms.py",
                        "desktop/scripts/package_runtime.py")})
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
            record["owned_runtime_stopped"] = process.poll() is not None
    record["owned_temporary_data_removed"] = not root.exists()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    record = verify(args.runtime)
    with args.output.open("x") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"status": record["status"], "frozen_execution": record["proof"]["frozen_execution"],
                      "history_after_worker_reset": record["history_after_worker_reset"]}))


if __name__ == "__main__":
    main()
