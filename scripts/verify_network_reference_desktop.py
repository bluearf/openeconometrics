"""Replay public network references through a disposable installed desktop worker.

No human project, account, running app, installation or system Python is modified.
The host script uses only the standard library; numerical execution is frozen.
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
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(runtime, fixtures):
    runtime, fixtures = Path(runtime).resolve(strict=True), Path(fixtures).resolve(strict=True)
    example = ROOT / "docs/examples/network_reference_validation.py"
    source = example.read_text().split('\nif __name__ == "__main__"', 1)[0]
    fingerprint = digest(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-public-reference-desktop-") as temporary:
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
                code = source + f'''
check(getattr(sys, "frozen", False), "Reference execution did not use frozen Python")
check(Path(oe.__file__).is_relative_to(Path(sys._MEIPASS)), "SDK escaped the frozen runtime")
receipt = validate({str(fixtures)!r})
display(oe.DataFrame([{{"dataset": name, "nodes": item["nodes"], "edges": item["edges"],
    "triangles": item["triangles"]}} for name, item in receipt["datasets"].items()]))
nodes, edges, _, _ = load_fixture({str(fixtures)!r}, "karate")
graph = oe.network({{"source": [a for a, _ in edges], "target": [b for _, b in edges]}}, nodes=nodes)
display(oe.plot.network(graph, title="Pinned Zachary reference · full topology", layout="circular"))
receipt["frozen_execution"] = True
print("NETWORK_REFERENCE_RECEIPT:" + json.dumps(receipt, allow_nan=False))
'''
                run = call("/console/execute", {"code": code, "timeout_seconds": 180})
                if run.get("status") != "ok":
                    raise RuntimeError(f"Frozen reference run failed: {run.get('error')}")
                markers = [line.split(":", 1)[1] for line in run["stdout"].splitlines()
                           if line.startswith("NETWORK_REFERENCE_RECEIPT:")]
                if len(markers) != 1 or [item["type"] for item in run["outputs"]] != ["table", "plot"]:
                    raise RuntimeError("Missing frozen reference receipt or ordered panel outputs")
                record = json.loads(markers[0])
                chart = run["outputs"][1]["data"]["config"]["network"]
                if chart["node_count"] != 34 or chart["edge_count"] != 78 or chart["sampled"]:
                    raise RuntimeError("Frozen reference chart lost full topology")
                # Reset clears worker variables. History must reopen from the
                # actual workspace store and retain the same structured bytes.
                call("/console/reset", {})
                history = call("/console")["history"]
                saved = next(item for item in history if item["id"] == run["id"])
                if saved["outputs"] != run["outputs"] or saved["events"] != run["events"]:
                    raise RuntimeError("Frozen panel outputs changed after worker reset")
                if digest(runtime) != fingerprint:
                    raise RuntimeError("Installed runtime changed during verification")
                record.update(runtime_executable_sha256=fingerprint,
                    example_sha256=digest(example), verifier_sha256=digest(__file__),
                    console_status=run["status"], console_duration_ms=run["duration_ms"],
                    ordered_panel_outputs=["table", "plot"], history_after_worker_reset=True,
                    displayed_nodes=34, displayed_edges=78, displayed_sampled=False,
                    human_data_access=False, installation_changed=False,
                    browser_render_verified=False)
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
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests/fixtures/network_reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path; existing evidence is never replaced.")
    record = verify(args.runtime, args.fixtures)
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": record["status"], "frozen_execution": record["frozen_execution"],
                      "history_after_worker_reset": record["history_after_worker_reset"]}))


if __name__ == "__main__":
    main()
