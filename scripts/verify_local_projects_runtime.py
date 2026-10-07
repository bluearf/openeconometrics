"""Verify fresh local projects and restart through an owned frozen runtime.

Uses synthetic data, a temporary root and no account. On macOS, --deny-network
also denies outbound connections except loopback in the runtime and its workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen


class Runtime:
    def __init__(self, executable: Path, root: Path, deny_network: bool):
        command = [str(executable), "--data-root", str(root)]
        if deny_network:
            command = ["/usr/bin/sandbox-exec", "-p", '(version 1) (allow default) (deny network-outbound) (allow network-outbound (remote ip "localhost:*"))', *command]
        self.command, self.root = command, root

    def __enter__(self):
        environment = {key: value for key, value in os.environ.items()
                       if key in {"HOME", "USER", "TMPDIR", "LANG"}}
        environment["PATH"] = "/usr/bin:/bin"
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(self.command, cwd=self.root, env=environment,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log.seek(0)
                raise RuntimeError(self.log.read().decode())
            if select.select([self.process.stdout], [], [], .2)[0]:
                descriptor = json.loads(self.process.stdout.readline())
                if descriptor.get("type") == "ready":
                    self.origin = descriptor["url"]
                    return self
        self.process.terminate()
        self.process.wait(timeout=10)
        raise TimeoutError("Frozen runtime did not become ready")

    def request(self, path, token=None, body=None, method=None, raw=None, content_type=None):
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": content_type or "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        with urlopen(Request(self.origin + path, headers=headers, data=data, method=method), timeout=90) as response:
            return json.load(response)

    def __exit__(self, *_):
        self.process.stdin.write(b'{"type":"shutdown"}\n')
        self.process.stdin.flush()
        self.process.wait(timeout=20)
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deny-network", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    code = '''import importlib.util, json, socket
import openecon as oe
model = oe.ols(data=oe.example(), y="wage", x=["education", "experience"], covariance="HC3")
assert model.nobs == 480
assert all(importlib.util.find_spec(name) is None for name in ["scipy", "statsmodels", "linearmodels"])
display(model)
print(json.dumps({"nobs": model.nobs, "runtime_exclusions": True}))
'''
    if args.deny_network:
        code += '''connection = socket.socket()
connection.settimeout(2)
assert connection.connect_ex(("1.1.1.1", 443)) == 1
connection.close()
print("outbound_network_denied")
'''
    receipt = {"runtime_sha256": hashlib.file_digest(runtime.open("rb"), "sha256").hexdigest(),
               "synthetic_only": True, "account_required": False, "system_python_required": False,
               "outbound_network_denied_except_loopback": args.deny_network}
    with tempfile.TemporaryDirectory(prefix="openecon-local-projects-") as temporary:
        root = Path(temporary)
        with Runtime(runtime, root, args.deny_network) as server:
            token = server.request("/api/desktop/session")["token"]
            assert server.request("/api/desktop/local-projects", token) == {"projects": []}
            project = server.request("/api/desktop/local-projects", token, {"name": "Offline synthetic research"})
            ident = project["id"]
            assert server.request(f"/api/desktop/local-projects/{ident}/open", token, {}) == project
            prefix = f"/api/desktop/projects/{ident}/workspace"
            local = server.request(prefix + "/session")["token"]
            saved = server.request(prefix + "/console/scripts/analysis", local, {"code": code, "version": 0}, "PUT")
            boundary = "openecon-synthetic-upload"
            upload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic.csv"\r\n'
                      f'Content-Type: text/csv\r\n\r\nx,y\n1,2\n2,3\n3,4\n\r\n--{boundary}--\r\n').encode()
            dataset = server.request(prefix + "/datasets/upload", local, method="POST", raw=upload,
                                     content_type=f"multipart/form-data; boundary={boundary}")
            environment = server.request(prefix + "/environment", local)
            run = server.request(prefix + "/console/execute", local, {"code": code})
            assert run["status"] == "ok", run
            assert '"nobs": 480' in run["stdout"]
            assert not args.deny_network or "outbound_network_denied" in run["stdout"]
            assert len(run["outputs"]) == 1 and run["outputs"][0]["type"] == "model"
            assert "\\begin{tabular}" in run["outputs"][0]["latex"]
            assert server.request(prefix + "/desktop-outbox", local) == {"items": []}
            assert not (root / "projects" / ident / "desktop-sync-state.json").exists()
        with Runtime(runtime, root, args.deny_network) as server:
            token = server.request("/api/desktop/session")["token"]
            assert server.request("/api/desktop/local-projects", token) == {"projects": [project]}
            assert server.request(f"/api/desktop/local-projects/{ident}/open", token, {}) == project
            local = server.request(prefix + "/session")["token"]
            assert server.request(prefix + "/console/scripts/analysis", local) == saved
            assert server.request(prefix + "/datasets", local)["datasets"][0]["data_hash"] == dataset["data_hash"]
            assert server.request(prefix + "/environment", local)["manifest"] == environment["manifest"]
            history = server.request(prefix + "/console", local)["history"]
            assert history[0]["id"] == run["id"] and history[0]["outputs"] == run["outputs"]
        receipt.update(status="passed", nobs=480, code_data_environment_history_survive_restart=True,
                       local_results_not_queued_for_sharing=True, runtime_exclusions_verified=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
