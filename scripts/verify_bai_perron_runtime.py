"""Frozen full-table execution and real server restart, in owned temporary data.

This does not claim native-window acceptance or a public release.
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
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["econometrics.unitroot.bai_perron", "econometrics.unitroot.__init__"]


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalized(code):
    return code.replace(co_filename="<bundled-source>", co_consts=tuple(
        normalized(item) if isinstance(item, CodeType) else item for item in code.co_consts))


class OwnedRuntime:
    def __init__(self, runtime, root, project):
        self.root, self.project = root, project
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        self.errors = (root / "verifier-stderr.log").open("ab")
        self.process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(root)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.errors, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if select.select([self.process.stdout], [], [], .25)[0]:
                    self.descriptor = json.loads(self.process.stdout.readline(8193))
                    break
                if self.process.poll() is not None:
                    raise RuntimeError("Owned runtime exited before readiness")
            else:
                raise RuntimeError("Owned runtime readiness timed out")
            d = self.descriptor
            if d.get("type") != "ready" or d["url"] != f"http://127.0.0.1:{d['port']}":
                raise RuntimeError("Invalid loopback descriptor")
            self.token = d["token"]
        except BaseException:
            self.close()
            raise

    def call(self, path, body=None, method=None):
        return self.request(f"/api/desktop/projects/{self.project}/workspace" + path, body, method)

    def request(self, path, body=None, method=None, *, desktop=False):
        d = self.descriptor
        request = Request(d["url"] + path,
                          headers={"Content-Type": "application/json", "X-OpenEcon-Token": d["token"] if desktop else self.token},
                          data=json.dumps(body).encode() if body is not None else None, method=method)
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    def open_project(self, *, create=False):
        if create:
            project = self.request("/api/desktop/local-projects", {"name": "Bai-Perron synthetic QA"}, desktop=True)
            self.project = project["id"]
        self.request(f"/api/desktop/local-projects/{self.project}/open", {}, desktop=True)
        self.token = self.call("/session")["token"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'{"type":"shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.errors.close()


def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    runtime = runtime.resolve(strict=True)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    hashes = {}
    for name in MODULES:
        source = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        module = ("openecon." + name).removesuffix(".__init__")
        if normalized(archive.extract(module)) != normalized(compile(source.read_text(), str(source), "exec", dont_inherit=True)):
            raise RuntimeError(f"Compiled module differs: {module}")
        hashes[module] = digest(source)
    example = ROOT / "docs/examples/bai_perron.py"
    code = """import sys
from pathlib import Path
import importlib
assert getattr(sys, 'frozen', False)
module = importlib.import_module('openecon.econometrics.unitroot.bai_perron')
assert Path(module.__file__).is_relative_to(Path(sys._MEIPASS))
""" + example.read_text()
    with tempfile.TemporaryDirectory(prefix="openecon-break-proof-") as temporary:
        root = Path(temporary)
        project = uuid4().hex
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project(create=True)
            project = owned.project
            owned.call("/console/script", {"code": code, "name": "analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
            if run.get("status") != "ok":
                raise RuntimeError(f"Frozen example failed: {run.get('error')}")
            marker = "BAI_PERRON_OK "
            proof = [json.loads(line[len(marker):]) for line in run["stdout"].splitlines() if line.startswith(marker)]
            if len(proof) != 1 or not proof[0]["frozen"]:
                raise RuntimeError("Missing frozen proof marker")
            proof = proof[0]
            outputs = run["outputs"]
            if len(outputs) != 7 or any(item["type"] != "table" for item in outputs):
                raise RuntimeError("Incomplete ordered table outputs")
            expected = [proof["table_rows"][name] for name in (
                "models", "tests", "segments", "coefficients", "covariance", "sample", "settings")]
            if [len(item["data"]["rows"]) for item in outputs] != expected:
                raise RuntimeError("Saved table rows were truncated")
            if any(item["data"]["total_rows"] != len(item["data"]["rows"]) for item in outputs):
                raise RuntimeError("Incomplete result preview")
            if any(not any(environment in item.get("latex", "")
                           for environment in ("\\begin{tabular}", "\\begin{longtable}"))
                   for item in outputs):
                raise RuntimeError("Missing persisted LaTeX: " + repr([
                    {"columns": item["data"]["columns"], "keys": list(item),
                     "latex_prefix": item.get("latex", "")[:200]} for item in outputs]))
            settings = {row[0]: json.loads(row[1]) for row in outputs[-1]["data"]["rows"]}
            if settings != proof["metadata"]:
                raise RuntimeError("Saved settings lost scientific metadata")
            document = owned.call("/console/script")
            saved = next(item for item in owned.call("/console")["history"] if item["id"] == run["id"])
            original = {key: saved[key] for key in ("code", "stdout", "outputs", "events")}
        finally:
            owned.close()
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project()
            history = owned.call("/console")["history"]
            after = next(item for item in history if item["id"] == run["id"])
            if original != {key: after[key] for key in original}:
                raise RuntimeError("Code/stdout/outputs/events changed across full server restart")
            if owned.call("/console/script") != document:
                raise RuntimeError("Saved script changed across full server restart")
        finally:
            owned.close()
        record = dict(status="passed", proof=proof, compiled_modules_equal_source=hashes,
                      runtime_sha256=digest(runtime), verifier_sha256=digest(__file__),
                      example_sha256=digest(example), execution_id=run["id"],
                      console_duration_ms=run["duration_ms"], ordered_table_rows=expected,
                      outputs_sha256=hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest(),
                      full_scientific_settings_saved=True, full_code_stdout_outputs_events_equal_after_server_restart=True,
                      saved_script_equal_after_server_restart=True, owned_runtime_stopped=True,
                      source_path_injected=False, human_data_access=False, native_window_verified=False,
                      installation_changed=False, public_release_delivered=False, cuda_verified=False)
    record["owned_temporary_data_removed"] = not root.exists()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "ordered_table_rows", "console_duration_ms")}))
