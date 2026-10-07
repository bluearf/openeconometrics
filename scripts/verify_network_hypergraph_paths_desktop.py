"""Run hypergraph incidence and bounded strong paths in a freshly frozen, disposable desktop console worker.

The host uses the standard library. No source path is injected into the frozen
worker, and no human application, installation or project store is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(runtime):
    runtime = Path(runtime).resolve(strict=True)
    example = ROOT / "docs/examples/network_hypergraph_and_paths.py"
    fingerprint = digest(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-directed-cut-desktop-") as temporary:
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
                        raise RuntimeError("Disposable desktop runtime exited before ready")
                if not descriptor or descriptor.get("type") != "ready":
                    raise RuntimeError("Disposable desktop runtime did not become ready")
                parsed = urlparse(descriptor["url"])
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or (
                        parsed.port != descriptor["port"]):
                    raise RuntimeError("Invalid loopback descriptor")
                token = call("/session")["token"]
                code = '''
import sys
from pathlib import Path
import openecon as oe
import openecon._network_hypergraph as directed_module
import openecon._network_bounded_paths as paths_module
assert getattr(sys, "frozen", False), "Execution must use frozen Python"
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
assert Path(directed_module.__file__).is_relative_to(Path(sys._MEIPASS))
''' + example.read_text() + '''
assert Path(paths_module.__file__).is_relative_to(Path(sys._MEIPASS))
print("FROZEN_HYPERGRAPH_PATHS:" + json.dumps({
    "frozen_execution": True, "sdk_from_bundle": True, "hypergraph_module_from_bundle": True,
    "paths_module_from_bundle": True, "platform": sys.platform,
}))
'''
                run = call("/console/execute", {"code": code, "timeout_seconds": 180})
                if run.get("status") != "ok":
                    raise RuntimeError(f"Frozen directed cut run failed: {run.get('error')}")
                markers = {}
                for marker in ("HYPERGRAPH_PATHS_RECEIPT:", "FROZEN_HYPERGRAPH_PATHS:"):
                    values = [json.loads(line[len(marker):]) for line in run["stdout"].splitlines()
                              if line.startswith(marker)]
                    if len(values) != 1:
                        raise RuntimeError("Missing unique frozen directed-cut receipt")
                    markers.update(values[0])
                if [out["type"] for out in run["outputs"]] != ["table"] * 5 + ["plot"]:
                    raise RuntimeError("Missing full direct/projection/path/cut output panels")
                if run["outputs"][1]["data"]["total_rows"] != 4:
                    raise RuntimeError("Direct hyperdegree output lost nodes")
                chart = run["outputs"][-1]["data"]["config"]["network"]
                if chart["node_count"] != 4 or chart["edge_count"] != 3 or chart["sampled"]:
                    raise RuntimeError("Projection chart lost full topology")
                call("/console/reset", {})
                saved = next(item for item in call("/console")["history"] if item["id"] == run["id"])
                if saved["outputs"] != run["outputs"] or saved["events"] != run["events"]:
                    raise RuntimeError("Frozen panel outputs changed after worker reset")
                if digest(runtime) != fingerprint:
                    raise RuntimeError("Frozen runtime changed during verification")
                record = dict(markers, status="passed", runtime_executable_sha256=fingerprint,
                    example_sha256=digest(example), verifier_sha256=digest(__file__),
                    console_status=run["status"], console_duration_ms=run["duration_ms"],
                    ordered_panel_outputs=["table"] * 5 + ["plot"], history_after_worker_reset=True,
                    displayed_nodes=4, displayed_edges=3, displayed_sampled=False,
                    human_data_access=False, installation_changed=False,
                    browser_render_verified=False, release_delivered=False)
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
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "frozen_execution", "history_after_worker_reset")}))


if __name__ == "__main__":
    main()
