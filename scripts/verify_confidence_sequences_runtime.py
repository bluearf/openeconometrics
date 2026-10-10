"""Run and cold-restore eight complete confidence families in a frozen worker.

Only stdlib code executes in this controller. All openecon calculations occur
inside the supplied frozen runtime, with an entirely new disposable profile.
This is API/worker acceptance; actual native-window Run is a separate receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MARKER = "CONFIDENCE_SEQUENCES_ACCEPTANCE_OK "


class Runtime:
    def __init__(self, executable, profile):
        self.executable, self.profile = executable, profile
        self.http = build_opener(ProxyHandler({}))
        self.process = None
        self.token = None
        self.port = None

    def start(self):
        env = {key: value for key, value in os.environ.items()
               if key in {"SYSTEMROOT", "WINDIR", "COMSPEC", "HOME", "LANG"}}
        env.update(PATH=(str(Path(env["SYSTEMROOT"])/"System32") if os.name == "nt" else "/usr/bin:/bin"),
                   PYINSTALLER_RESET_ENVIRONMENT="1", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        ready = queue.Queue(maxsize=1)
        self.stderr = (self.profile/"runtime.stderr.log").open("ab")
        self.process = subprocess.Popen([str(self.executable), "--port", str(self.port or 0),
            "--data-root", str(self.profile)], cwd=self.profile, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr)
        def reader():
            try:
                raw = self.process.stdout.readline(16385)
                assert len(raw) <= 16384 and raw.endswith(b"\n")
                ready.put(json.loads(raw))
                while self.process.stdout.read(65536):
                    pass
            except BaseException as error:
                if ready.empty():
                    ready.put(error)
        threading.Thread(target=reader, daemon=True).start()
        value = ready.get(timeout=120)
        if isinstance(value, BaseException):
            raise value
        assert value["type"] == "ready" and value["pid"] == self.process.pid
        self.port = value["port"]
        self.origin = f"http://127.0.0.1:{self.port}"
        assert value["url"].rstrip("/") == self.origin
        self.token = self.call("/api/desktop/session")["token"]

    def call(self, path, body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token or self.token:
            headers["X-OpenEcon-Token"] = token or self.token
        request = Request(self.origin+path, headers=headers,
                          data=json.dumps(body).encode() if body is not None else None)
        with self.http.open(request, timeout=240) as response:
            raw = response.read(32*1024**2+1)
        assert len(raw) <= 32*1024**2
        return json.loads(raw)

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.stdin.write(b'{"type":"shutdown"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        assert self.process.returncode == 0
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
        self.process = None
        self.token = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory.exists():
        raise RuntimeError("An entirely new acceptance directory is required.")
    directory.mkdir(parents=True)
    profile = directory/"profile"
    profile.mkdir()
    executable = args.runtime.resolve(strict=True)
    worker = Runtime(executable, profile)
    code = (ROOT/"docs/examples/confidence_sequences_eight.py").read_text()
    receipt = {"status": "running", "runtime": str(executable),
        "runtime_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "example_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "native_window_verified": False, "source_path_injected": False}
    try:
        worker.start()
        project = worker.call("/api/desktop/local-projects", {"name": "Confidence sequences frozen QA", "description": "Synthetic offline acceptance."})
        worker.call(f"/api/desktop/local-projects/{project['id']}/open", {})
        prefix = f"/api/desktop/projects/{project['id']}/workspace"
        token = worker.call(prefix+"/session")["token"]
        run = worker.call(prefix+"/console/execute", {"code": code, "timeout_seconds": 180}, token)
        assert run["status"] == "ok", run.get("error")
        lines = [line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)]
        assert len(lines) == 1
        proof = json.loads(lines[0])
        assert proof["frozen"] and not proof["source_path_injected"] and not proof["scipy_imported"]
        assert proof["stages"] == proof["tables"] == proof["full_states"] == len(run["outputs"]) == 8
        assert all(item["type"] == "table" and item["data"]["total_rows"] == 24 for item in run["outputs"])
        project_path = profile/"projects"/project["id"]
        assert len(proof["files"]) == 16
        for name, item in proof["files"].items():
            raw = (project_path/"confidence_sequence_results"/name).read_bytes()
            assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
        history = worker.call(prefix+"/console", token=token)["history"]
        assert len(history) == 1
        before = json.dumps(history, sort_keys=True)
        worker.stop()
        worker.start()
        worker.call(f"/api/desktop/local-projects/{project['id']}/open", {})
        token = worker.call(prefix+"/session")["token"]
        cold = worker.call(prefix+"/console", token=token)
        assert cold["status"]["pid"] is None
        assert json.dumps(cold["history"], sort_keys=True) == before
        restore_code = '''
from pathlib import Path
import hashlib,json
import openecon as oe
def forbidden(*args, **kwargs):
    raise AssertionError("Cold restore must not recalculate confidence sequences")
for name in NAMES:
    setattr(oe, name, forbidden)
for name, item in EXPECTED.items():
    path=Path("confidence_sequence_results")/name
    raw=path.read_bytes()
    assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"]
    if name.endswith(".json"):
        state=raw.decode()
        result=oe.restore_summary(state)
        assert oe.summary_state(result)==state
        assert len(result["intervals"])==24 and len(result["sample"])==12
        tex=result["intervals"].to_latex()+"\\n"+result["sample"].to_latex()
        assert tex==(path.with_suffix(".tex")).read_text()
print("CONFIDENCE_SEQUENCES_COLD_RESTORE_OK")
'''.replace("NAMES", repr([name[:-5] for name in proof["files"] if name.endswith(".json")])).replace("EXPECTED", repr(proof["files"]))
        restored = worker.call(prefix+"/console/execute", {"code": restore_code, "timeout_seconds": 120}, token)
        assert restored["status"] == "ok", restored.get("error")
        assert "CONFIDENCE_SEQUENCES_COLD_RESTORE_OK" in restored["stdout"]
        worker.stop()
        receipt.update(status="passed", proof=proof, project_id=project["id"], run_id=run["id"],
                       duration_ms=run["duration_ms"], cold_history_equal=True, cold_worker_absent=True,
                       recalculation_disabled_full_restore=True, owned_runtime_stopped=True)
        (directory/"full-run.json").write_text(json.dumps(run, indent=2)+"\n")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        worker.stop()
        (directory/"acceptance.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("status", "duration_ms", "cold_history_equal", "owned_runtime_stopped")}))


if __name__ == "__main__":
    main()
