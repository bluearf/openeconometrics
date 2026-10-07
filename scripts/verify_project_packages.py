"""Exercise an owned desktop runtime with temporary projects only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def verify(runtime: Path | None, output: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="openecon-package-check-") as directory:
        root = Path(directory)
        command = ([str(runtime)] if runtime else [sys.executable, "-m", "openecon.desktop_entry"])
        process = None
        descriptor = None
        stderr_file = None
        stderr_path = root / "runtime-stderr.log"
        headers = {}
        report = {"status": "running", "human_data_access": False, "system_python_required": runtime is None,
                  "runtime": str(runtime) if runtime else "source"}

        def start():
            nonlocal process, descriptor, stderr_file
            env = os.environ.copy()
            if runtime:
                env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            stderr_file = stderr_path.open("ab")
            process = subprocess.Popen(command + ["--data-root", str(root)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr_file, env=env,
                                       start_new_session=os.name == "posix")
            started = time.monotonic()
            deadline = started + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], min(1, deadline - time.monotonic()))[0]:
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Owned runtime exited during startup: {process.returncode}")
            else:
                raise RuntimeError("Owned runtime startup timed out after 60 seconds")
            descriptor = json.loads(process.stdout.readline())
            assert descriptor["type"] == "ready"
            report.setdefault("startup_seconds", []).append(round(time.monotonic() - started, 3))

        def stop():
            nonlocal process, stderr_file
            if process is None:
                return
            if process.poll() is None:
                try:
                    process.stdin.write(b'{"type":"shutdown"}\n')
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=3)
            for pipe in [process.stdin, process.stdout]:
                pipe.close()
            stderr_file.close()
            stderr_file = None
            process = None

        def call(project, path, body=None, *, method=None):
            prefix = f"/api/desktop/projects/{project}/workspace"
            request = Request(descriptor["url"] + prefix + path,
                              data=json.dumps(body).encode() if body is not None else None,
                              headers={**headers.get(project, {}), "Content-Type": "application/json"},
                              method=method or ("POST" if body is not None else "GET"))
            try:
                with urlopen(request, timeout=90) as response:
                    return json.load(response)
            except HTTPError as error:
                raise RuntimeError(f"Owned API HTTP {error.code}: {error.read(2000).decode()}") from None

        def connect(project):
            headers[project] = {"X-OpenEcon-Token": call(project, "/session")["token"]}

        def execute(project, code):
            return call(project, "/console/execute", {"code": code})

        def operation(project, path, body, *, expected="complete"):
            call(project, path, body)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                snapshot = call(project, "/environment")
                if snapshot["job"]["state"] != "running":
                    assert snapshot["job"]["state"] == expected, snapshot["job"]
                    return snapshot
                time.sleep(.3)
            raise RuntimeError("Owned package operation timed out")

        first, second, third = (uuid4().hex for _ in range(3))
        try:
            start()
            connect(first)
            before = call(first, "/environment")["manifest"]["core"]
            initial = execute(first, "print(oe.ols(data=oe.example(), y='wage', x=['education', 'experience']).nobs)")
            assert initial["status"] == "ok" and "480" in initial["stdout"], initial
            if runtime:
                metadata = execute(first, "import importlib.metadata\nassert importlib.metadata.version('openecon') == oe.__version__")
                assert metadata["status"] == "ok", metadata
                report["distribution_metadata_matches_source"] = True
            installed = operation(first, "/environment/install", {"name": "humanize"})
            pure = execute(first, "import humanize\nprint(humanize.intcomma(12345))")
            assert pure["status"] == "ok" and "12,345" in pure["stdout"], pure
            installed = operation(first, "/environment/install", {"name": "polars"})
            native = execute(first, "import polars as pl\nprint(pl.DataFrame({'x':[1,2,3]}).select(pl.col('x').sum()).item())")
            assert native["status"] == "ok" and "6" in native["stdout"], native
            manifest = installed["manifest"]
            report["installed"] = manifest["locked"]
            failed = operation(first, "/environment/install", {"name": "openecon-package-qa-" + uuid4().hex}, expected="error")
            assert failed["manifest"] == manifest
            report["failed_install_preserves_environment"] = True
            connect(second)
            isolated = execute(second, "import humanize")
            assert isolated["status"] == "error" and isolated["error"]["type"] == "ModuleNotFoundError", isolated
            report["project_isolation"] = True
            stop()
            start()
            connect(first)
            retained = execute(first, "import humanize, polars\nprint(humanize.intcomma(12345))\nmodel=oe.ols(data=oe.example(), y='wage', x=['education', 'experience'])\ndisplay(model)")
            assert retained["status"] == "ok" and "12,345" in retained["stdout"], retained
            assert any("\\toprule" in item.get("latex", "") for item in retained["outputs"]), retained
            assert call(first, "/environment")["manifest"] == manifest
            assert manifest["core"] == before
            report["restart_persistence"] = report["core_unchanged"] = report["ols_and_latex"] = True
            connect(third)
            restored = operation(third, "/environment/restore", {"manifest": manifest})
            assert restored["manifest"] == manifest
            assert execute(third, "import polars, humanize\nprint(polars.__version__)")["status"] == "ok"
            report["exact_manifest_restore"] = True
            removed = operation(third, "/environment/remove", {"name": "humanize"})
            assert [item["name"] for item in removed["requirements"]] == ["polars"]
            assert execute(third, "import humanize")["error"]["type"] == "ModuleNotFoundError"
            assert execute(third, "import polars\nprint(polars.__version__)")["status"] == "ok"
            report["removal_and_dependencies"] = True
            report["status"] = "ok"
            return report
        except Exception as error:
            report["status"] = "error"
            report["error"] = str(error)[:4000]
            if process is not None:
                report["runtime_exit_code"] = process.poll()
            raise
        finally:
            stop()
            if stderr_path.exists():
                with stderr_path.open("rb") as stream:
                    stream.seek(max(0, stderr_path.stat().st_size - 32768))
                    report["runtime_stderr"] = stream.read().decode("utf-8", errors="replace")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.runtime, arguments.output)))
