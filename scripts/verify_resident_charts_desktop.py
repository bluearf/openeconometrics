"""Verify resident chart outputs using an owned frozen desktop worker and store."""
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
    example = ROOT / "docs/examples/resident_charts.py"
    fingerprint = digest(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-resident-desktop-") as temporary:
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
                        raise RuntimeError("Owned runtime exited before ready")
                if not descriptor or descriptor.get("type") != "ready":
                    raise RuntimeError("Owned runtime did not become ready")
                parsed = urlparse(descriptor["url"])
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != descriptor["port"]:
                    raise RuntimeError("Invalid loopback descriptor")
                token = call("/session")["token"]
                code = '''
import sys
from pathlib import Path
import openecon as oe
from openecon_charts import resident
assert getattr(sys, "frozen", False)
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
assert Path(resident.__file__).is_relative_to(Path(sys._MEIPASS))
''' + example.read_text() + '''
from openecon.team_output import validate_plot
assert validate_plot(scatter.with_options(color="#123456").model_dump())["config"]["processing"] == scatter.config["processing"]
print("FROZEN_RESIDENT_RECEIPT:" + json.dumps({"frozen_execution": True,
    "sdk_from_bundle": True, "resident_adapter_from_bundle": True,
    "publication_metadata_verified": True, "platform": sys.platform}))
'''
                run = call("/console/execute", {"code": code, "timeout_seconds": 180})
                if run.get("status") != "ok":
                    raise RuntimeError(f"Frozen resident chart run failed: {run.get('error')}")
                record = {}
                for marker in ("RESIDENT_CHARTS_RECEIPT:", "FROZEN_RESIDENT_RECEIPT:"):
                    values = [json.loads(line[len(marker):]) for line in run["stdout"].splitlines() if line.startswith(marker)]
                    if len(values) != 1:
                        raise RuntimeError("Missing unique chart receipt")
                    record.update(values[0])
                kinds = [out["type"] for out in run["outputs"]]
                if kinds != ["table", "plot", "plot", "plot"]:
                    raise RuntimeError("Missing ordered chart outputs")
                hist, scatter, line = [out["data"] for out in run["outputs"][1:]]
                if (sum(row["count"] for row in hist["data"]) != record["histogram_total_n"]
                        or scatter["sample_n"] != 2000 or len(line["data"]) != 6
                        or scatter["config"]["processing"]["source_rows"] != 1_000_003):
                    raise RuntimeError("Frozen chart counts, sampling or gaps changed")
                call("/console/reset", {})
                saved = next(item for item in call("/console")["history"] if item["id"] == run["id"])
                if saved["outputs"] != run["outputs"] or saved["events"] != run["events"]:
                    raise RuntimeError("Stored outputs changed after worker reset")
                if digest(runtime) != fingerprint:
                    raise RuntimeError("Frozen executable changed during verification")
                record.update(status="passed", runtime_executable_sha256=fingerprint,
                    example_sha256=digest(example), verifier_sha256=digest(__file__),
                    console_duration_ms=run["duration_ms"], ordered_panel_outputs=kinds,
                    history_after_worker_reset=True, human_data_access=False, installation_changed=False,
                    browser_render_verified=False, cloud_delivery_verified=False, release_delivered=False)
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
