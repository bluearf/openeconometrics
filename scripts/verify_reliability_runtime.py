"""Check compiled source identity, frozen execution and complete saved tables.

This verifier starts an owned temporary console, not a native window. Native
installed application evidence must be recorded separately.
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
from types import CodeType
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "__init__",
    "econometrics.measurement.__init__",
    "econometrics.measurement.common",
    "econometrics.measurement.agreement",
    "econometrics.measurement.scale",
    "econometrics.measurement.ordinal",
    "econometrics.multivariate.common",
    "econometrics.multivariate.extraction",
    "econometrics.discrete.bivariate",
    "engines.distributions",
    "engines.optimize",
    "econometrics.registry",
    "econometrics.core",
    "analysis_contracts",
    "resources",
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(code):
    return code.replace(
        co_filename="<bundled-source>",
        co_consts=tuple(
            normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts
        ),
    )


def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader

    runtime = Path(runtime).resolve(strict=True)
    fingerprint = digest(runtime)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        path = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        module = ("openecon." + name).removesuffix(".__init__")
        expected = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        if normalized(archive.extract(module)) != normalized(expected):
            raise RuntimeError(f"Compiled module differs from current source: {module}")
        sources[module] = digest(path)
    example = ROOT / "docs/examples/reliability_eight.py"
    with tempfile.TemporaryDirectory(prefix="openecon-reliability-") as temporary:
        root = Path(temporary)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        with (root / "stderr.log").open("wb") as errors:
            process = subprocess.Popen(
                [str(runtime), "--port", "0", "--data-root", str(root)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                env=environment,
                start_new_session=True,
            )
            descriptor, token = None, None
            prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"

            def call(path, body=None):
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["X-OpenEcon-Token"] = token
                request = Request(
                    descriptor["url"] + prefix + path,
                    headers=headers,
                    data=json.dumps(body).encode() if body is not None else None,
                )
                with urlopen(request, timeout=180) as response:
                    return json.load(response)

            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if select.select([process.stdout], [], [], 0.25)[0]:
                        descriptor = json.loads(process.stdout.readline(8193))
                        break
                    if process.poll() is not None:
                        raise RuntimeError("Owned runtime exited before readiness")
                if not descriptor or descriptor.get("type") != "ready":
                    raise RuntimeError("Owned runtime did not become ready")
                parsed = urlparse(descriptor["url"])
                if (
                    parsed.scheme != "http"
                    or parsed.hostname != "127.0.0.1"
                    or parsed.port != descriptor["port"]
                ):
                    raise RuntimeError("Invalid loopback descriptor")
                token = call("/session")["token"]
                code = (
                    """import sys
from pathlib import Path
import importlib
assert getattr(sys, "frozen", False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
for name in %r:
    module = importlib.import_module(name)
    assert Path(module.__file__).is_relative_to(Path(sys._MEIPASS))
"""
                    % (tuple(sources),)
                    + "ARTIFACT_DIRECTORY = "
                    + repr(str(root / "reliability-artifacts"))
                    + "\n"
                    + example.read_text()
                )
                run = call("/console/execute", {"code": code, "timeout_seconds": 180})
                if run.get("status") != "ok":
                    raise RuntimeError(f"Frozen example failed: {run.get('error')}")
                marker = "RELIABILITY_EIGHT_OK "
                records = [
                    json.loads(line[len(marker) :])
                    for line in run["stdout"].splitlines()
                    if line.startswith(marker)
                ]
                if len(records) != 1 or not records[0]["frozen"]:
                    raise RuntimeError("Missing unique frozen example receipt")
                proof = records[0]
                outputs = run["outputs"]
                expected_kinds = ["table"] * 8
                if [item["type"] for item in outputs] != expected_kinds:
                    raise RuntimeError(
                        f"Missing ordered model/table outputs: {[item['type'] for item in outputs]}"
                    )
                counts = [
                    len(
                        item["data"]["coefficients"]
                        if item["type"] == "model"
                        else item["data"]["rows"]
                    )
                    for item in outputs
                ]
                if (
                    proof["fits"] != [41, 41, 25, 7, 11, 11, 11, 11]
                    or not proof["complete_artifacts_equal"]
                ):
                    raise RuntimeError(
                        "Frozen complete fits or artifacts differ from synthetic example"
                    )
                if any("\\begin{tabular}" not in item.get("latex", "") for item in outputs):
                    raise RuntimeError("A frozen table lacks its complete LaTeX representation")
                call("/console/reset", {})
                restore_code = "import json, openecon as oe\nfrom pathlib import Path\nassert getattr(__import__('sys'),'frozen',False)\n"
                restore_code += "expected = " + repr(proof["artifact_sha256"]) + "\n"
                restore_code += "folder = Path(" + repr(str(root / "reliability-artifacts")) + ")\n"
                restore_code += "for name, sha in expected.items():\n    assert oe.reliability_save(oe.reliability_load(folder/(name+'.json')))['sha256'] == sha\n"
                restored = call("/console/execute", {"code": restore_code, "timeout_seconds": 30})
                if restored.get("status") != "ok":
                    raise RuntimeError(
                        "Full frozen artifacts failed restoration after worker reset"
                    )
                saved = next(
                    item for item in call("/console")["history"] if item["id"] == run["id"]
                )
                if (
                    saved["outputs"] != outputs
                    or saved["events"] != run["events"]
                    or saved["code"] != code
                ):
                    raise RuntimeError("Saved code/outputs/events changed after worker reset")
                if digest(runtime) != fingerprint:
                    raise RuntimeError("Runtime changed during verification")
                record = dict(
                    status="passed",
                    proof=proof,
                    compiled_modules_equal_source=sources,
                    runtime_sha256=fingerprint,
                    example_sha256=digest(example),
                    verifier_sha256=digest(__file__),
                    console_duration_ms=run["duration_ms"],
                    table_rows=counts,
                    ordered_output_types=expected_kinds,
                    outputs_sha256=hashlib.sha256(
                        json.dumps(outputs, sort_keys=True).encode()
                    ).hexdigest(),
                    saved_code_outputs_events_equal_after_worker_reset=True,
                    full_artifacts_restored_after_worker_reset=True,
                    frozen_execution=True,
                    source_path_injected=False,
                    human_data_access=False,
                    native_window_verified=False,
                    installation_changed=False,
                    public_release_delivered=False,
                    cuda_verified=False,
                )
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "frozen_execution", "table_rows")}))


if __name__ == "__main__":
    main()
