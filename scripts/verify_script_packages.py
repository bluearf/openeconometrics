"""Verify code-driven installation against an owned source or frozen runtime."""
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
from urllib.request import Request, urlopen
from uuid import uuid4


def verify(runtime: Path | None, output: Path) -> dict:
    command = [str(runtime)] if runtime else [sys.executable, "-m", "openecon.desktop_entry"]
    report = {"human_data_access": False, "system_python_required": runtime is None,
              "runtime": str(runtime) if runtime else "source", "checks": {}}
    with tempfile.TemporaryDirectory(prefix="openecon-script-package-check-") as directory:
        root = Path(directory)
        process = None
        headers = {}
        descriptor = None
        errors = (root / "stderr.log").open("wb")

        def start():
            nonlocal process, descriptor
            environment = os.environ.copy()
            if runtime:
                environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            process = subprocess.Popen(command + ["--data-root", str(root)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=errors, env=environment,
                                       start_new_session=os.name == "posix")
            started = time.monotonic()
            if not select.select([process.stdout], [], [], 60)[0]:
                raise RuntimeError("Owned runtime did not become ready.")
            descriptor = json.loads(process.stdout.readline())
            report.setdefault("ready_seconds", []).append(round(time.monotonic() - started, 3))

        def stop():
            nonlocal process
            if process is None:
                return
            process.stdin.close()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=3)
            process.stdout.close()
            process = None

        def call(project, path, body=None):
            request = Request(descriptor["url"] + f"/api/desktop/projects/{project}/workspace" + path,
                              data=json.dumps(body).encode() if body is not None else None,
                              headers={**headers.get(project, {}), "Content-Type": "application/json"})
            with urlopen(request, timeout=180) as response:
                return json.load(response)

        def connect(project):
            headers[project] = {"X-OpenEcon-Token": call(project, "/session")["token"]}

        def execute(project, code):
            return call(project, "/console/execute", {"code": code})

        first, other = uuid4().hex, uuid4().hex
        try:
            start()
            connect(first)
            before = call(first, "/environment")["manifest"]["core"]
            code = ("saved = 37\nprint('before')\noe.install('humanize==4.16.0')\n"
                    "import humanize\nprint(humanize.intcomma(12345))\ndisplay(saved)")
            pure = execute(first, code)
            assert pure["status"] == "ok", pure
            assert pure["stdout"] == "before\nPackages ready: humanize==4.16.0\n12,345\n"
            assert pure["outputs"][0]["data"] == "37"
            assert [event["type"] for event in pure["events"]] == ["stdout", "output"]
            report["checks"]["pure_python_same_script_order"] = True
            installed = call(first, "/environment")
            pid = call(first, "/console")["status"]["pid"]
            started = time.monotonic()
            noop = execute(first, "oe.install('humanize')\nsaved")
            report["repeat_install_ms"] = round((time.monotonic() - started) * 1000, 2)
            assert noop["status"] == "ok" and noop["outputs"][0]["data"] == "37", noop
            assert call(first, "/environment")["job"]["id"] == installed["job"]["id"]
            assert call(first, "/console")["status"]["pid"] == pid
            report["checks"]["matching_install_no_job_or_worker_restart"] = True
            compiled = execute(first, "%pip install polars==1.44.2\nimport polars as pl\n"
                               "import humanize.number\nprint(pl.DataFrame({'x':[1,2,3]}).select(pl.col('x').sum()).item())\n"
                               "saved")
            assert compiled["status"] == "ok" and "6\n" in compiled["stdout"], compiled
            assert compiled["outputs"][0]["data"] == "37"
            report["checks"]["compiled_wheel_percent_pip_and_preserved_imports"] = True
            manifest = call(first, "/environment")["manifest"]
            rejected = execute(first, "oe.install('humanize==4.15.0')\nprint('must not run')")
            assert rejected["status"] == "error" and "Restart Python" in rejected["error"]["message"], rejected
            assert "must not run" not in rejected["stdout"]
            assert call(first, "/environment")["manifest"] == manifest
            assert execute(first, "saved")["outputs"][0]["data"] == "37"
            report["checks"]["imported_upgrade_rollback"] = True
            protected = execute(first, "%pip install torch==0.0.1")
            assert protected["status"] == "error" and "fixed versions" in protected["error"]["message"], protected
            report["checks"]["core_cannot_be_replaced"] = True
            regression = execute(first, "model = oe.ols(data=oe.example(), y='wage', x=['education','experience'])\n"
                                 "assert model.nobs == 480\nassert '\\\\toprule' in model.to_latex()\nprint(model.nobs)")
            assert regression["status"] == "ok", regression
            assert call(first, "/environment")["manifest"]["core"] == before
            report["checks"]["native_ols_and_latex_unchanged"] = True
            connect(other)
            isolated = execute(other, "import humanize")
            assert isolated["error"]["type"] == "ModuleNotFoundError", isolated
            report["checks"]["project_isolation"] = True
            stop()
            start()
            connect(first)
            resumed = execute(first, "import humanize, polars\nprint(humanize.intcomma(12345))")
            assert resumed["status"] == "ok" and "12,345" in resumed["stdout"], resumed
            assert call(first, "/environment")["manifest"] == manifest
            report["checks"]["restart_persistence"] = True
            report.update(status="passed", manifest=manifest)
        except BaseException as exc:
            report.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop()
            errors.close()
            report["owned_runtime_stopped"] = True
            if report.get("status") == "error":
                report["stderr_tail"] = (root / "stderr.log").read_text(errors="replace")[-4000:]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify(arguments.runtime, arguments.output)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))
