"""Verify scoped multiple-testing APIs in a disposable frozen loopback runtime.

Never uses an installed application, cloud account or human project. Persists
table plus attrs envelopes and verifies them after a full runtime restart.
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


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader

    runtime = runtime.resolve(strict=True)
    fingerprint = digest(runtime)
    example = ROOT / "docs/examples/multiple_testing.py"
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalized(code):
        return code.replace(
            co_filename="<scoped-bundle>",
            co_consts=tuple(normalized(v) if isinstance(v, CodeType) else v for v in code.co_consts),
        )

    hashes = {}
    for module in ["openecon.econometrics.postest.multiple", "openecon.econometrics.postest",
                   "openecon.econometrics.registry", "openecon.econometrics.core",
                   "openecon.analysis_contracts", "openecon.console_worker"]:
        source = ROOT / "src" / Path(*module.split("."))
        source = source / "__init__.py" if source.is_dir() else source.with_suffix(".py")
        assert normalized(archive.extract(module)) == normalized(
            compile(source.read_text(), str(source), "exec", dont_inherit=True)
        ), module
        hashes[module] = digest(source)

    with tempfile.TemporaryDirectory(prefix="openecon-multiple-frozen-") as temporary:
        root = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        environment = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT="1")
        environment.pop("PYTHONPATH", None)
        process = descriptor = token = None

        def call(path, body=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(descriptor["url"] + prefix + path, headers=headers,
                              data=json.dumps(body).encode() if body is not None else None)
            with urlopen(request, timeout=120) as response:
                return json.load(response)

        def start():
            nonlocal process, descriptor, token
            token = descriptor = None
            process = subprocess.Popen(
                [str(runtime), "--port", "0", "--data-root", str(root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, start_new_session=True,
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], .25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                if process.poll() is not None:
                    raise RuntimeError("Frozen runtime exited before ready")
            assert descriptor and descriptor.get("type") == "ready"
            parsed = urlparse(descriptor["url"])
            assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
            assert parsed.port == descriptor["port"]
            token = call("/session")["token"]

        def stop():
            if process and process.poll() is None:
                try:
                    process.stdin.write(b'{"type":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=12)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            if process:
                process.stdin.close()
                process.stdout.close()

        try:
            start()
            states_path = root / "multiple-state.json"
            code = """import sys
import importlib.util
from pathlib import Path
import openecon as oe
import openecon.econometrics.postest.multiple as multiple_module
assert getattr(sys, 'frozen', False)
assert Path(multiple_module.__file__).is_relative_to(Path(sys._MEIPASS))
assert importlib.util.find_spec('scipy') is None
assert importlib.util.find_spec('statsmodels') is None
assert oe.capabilities()['multiple_testing']['devices'] == ['cpu']
""" + example.read_text() + f"\n_ = Path({str(states_path)!r}).write_text(json.dumps(states, allow_nan=False))\n"
            run = call("/console/execute", {"code": code, "timeout_seconds": 120})
            assert run["status"] == "ok", run.get("error")
            marker = "MULTIPLE_TESTING_RECEIPT:"
            proof = [json.loads(line[len(marker):]) for line in run["stdout"].splitlines()
                     if line.startswith(marker)]
            assert len(proof) == 1
            assert [item["type"] for item in run["outputs"]] == ["table"] * 3
            assert all(item.get("latex") for item in run["outputs"])
            states_hash = digest(states_path)
            call("/console/reset", {})
            saved = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert saved["outputs"] == run["outputs"] and saved["events"] == run["events"]
            stop()
            start()
            reopened = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert reopened["outputs"] == run["outputs"] and reopened["events"] == run["events"]
            restore_code = f"""import json
from pathlib import Path
from io import StringIO
import pandas as pd
import openecon as oe
states = json.loads(Path({str(states_path)!r}).read_text())
restored = []
for state in states:
    frame = oe.DataFrame(pd.read_json(StringIO(json.dumps(state['table'])), orient='table'))
    frame.attrs = state['attrs']
    assert frame.to_latex()
    restored.append(frame)
assert restored[0]['adjusted_p_value'].tolist() == [.03, .2, .06]
assert restored[1].attrs['calibration'] == 'enumerated'
assert restored[2].attrs['joint_covariance'] == [[.04,.01,0],[.01,.09,.01],[0,.01,.01]]
assert restored[2].attrs['family_description']
assert all(frame.attrs['stata_parity_validated'] is False for frame in restored)
print('MULTIPLE_ENVELOPES_REOPENED:3')
"""
            restored = call("/console/execute", {"code": restore_code, "timeout_seconds": 120})
            assert restored["status"] == "ok", restored.get("error")
            assert "MULTIPLE_ENVELOPES_REOPENED:3" in restored["stdout"]
            assert digest(states_path) == states_hash and digest(runtime) == fingerprint
            receipt = dict(
                status="passed", issue="MARKET-182", proof=proof[0],
                runtime_executable_sha256=fingerprint, example_sha256=digest(example),
                verifier_sha256=digest(__file__), frozen_modules_match_source=hashes,
                console_duration_ms=run["duration_ms"], ordered_outputs=["table"] * 3,
                publication_latex=True, saved_attrs_and_table_envelopes=3,
                history_after_worker_reset=True, history_after_runtime_restart=True,
                full_metadata_after_restart=True, saved_envelopes_sha256=states_hash,
                external_oracle_packages_absent=True, human_data_access=False,
                cloud_endpoint_used=False, installed_ui_verified=False, release_delivered=False,
            )
        finally:
            stop()
        receipt["owned_runtime_stopped"] = process.poll() is not None
    receipt["owned_temporary_data_removed"] = not root.exists()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "tables": 3, "restart": True}))


if __name__ == "__main__":
    main()
