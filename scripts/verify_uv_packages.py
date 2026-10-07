"""Real code-driven uv proof, scoped to disposable local projects only."""
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
    report = {"status": "running", "human_data_access": False,
              "runtime": str(runtime) if runtime else "source", "checks": {}}
    with tempfile.TemporaryDirectory(prefix="openecon-uv-proof-") as directory:
        root = Path(directory)
        process, descriptor = None, None
        headers = {}
        stderr = (root / "stderr.log").open("ab")

        def start():
            nonlocal process, descriptor
            environment = os.environ.copy()
            if runtime:
                environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
                environment["PATH"] = "/usr/bin:/bin" if os.name == "posix" else environment.get("SYSTEMROOT", "")
                # No developer Python/package paths reach the shipped application.
                for key in list(environment):
                    if key.startswith(("PYTHON", "UV_", "PIP_")):
                        environment.pop(key)
                environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            started = time.monotonic()
            process = subprocess.Popen(command + ["--data-root", str(root)], env=environment,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                       start_new_session=os.name == "posix")
            if not select.select([process.stdout], [], [], 60)[0]:
                raise RuntimeError("Disposable runtime did not become ready.")
            descriptor = json.loads(process.stdout.readline())
            assert descriptor["type"] == "ready"
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
            with urlopen(request, timeout=240) as response:
                return json.load(response)

        def connect(project):
            headers[project] = {"X-OpenEcon-Token": call(project, "/session")["token"]}

        def execute(project, code):
            result = call(project, "/console/execute", {"code": code})
            assert result["status"] == "ok", result
            return result

        def install_snapshot(project):
            result = call(project, "/environment")
            assert result["job"]["state"] == "complete", result
            return result

        first, second = uuid4().hex, uuid4().hex
        try:
            start()
            connect(first)
            pure = execute(first, 'saved=37\nprint("before")\n%uv pip install "humanize>=4,<5"\nimport humanize\nprint(humanize.intcomma(12345))\nsaved')
            assert pure["stdout"].startswith("before\nPackages ready: ") and pure["stdout"].endswith("12,345\n"), pure
            assert pure["outputs"][0]["data"] == "37"
            initial = install_snapshot(first)
            assert initial["manifest"]["schema"] == 2 and initial["manifest"]["installer"] == "uv"
            assert initial["manifest"]["specifications"] == ["humanize<5,>=4"]
            report["checks"]["actual_uv_range_and_same_script_order"] = True
            old_pin = initial["manifest"]["requirements"][0]
            compiled = execute(first, '%uv add "polars>=1.44,<2"\nimport polars as pl\nprint(pl.DataFrame({"x":[1,2,3]}).select(pl.col("x").sum()).item())\nsaved')
            assert "6\n" in compiled["stdout"] and compiled["outputs"][0]["data"] == "37"
            native = install_snapshot(first)
            assert old_pin in native["manifest"]["requirements"]
            # Desktop workspace layout is intentionally discovered only below the owned root.
            downloads = list(root.rglob("generation-" + native["job"]["id"] + "/wheel-downloads.json"))
            assert len(downloads) == 1 and json.loads(downloads[0].read_text())["cache_hits"] >= 1
            report["checks"]["compiled_wheel_uv_add_saved_pin_and_verified_cache"] = True
            extra = execute(first, 'from pathlib import Path\nPath("requirements.txt").write_text("tabulate[widechars]>=0.9,<1\\n")\noe.install(installer="uv", requirements_file="requirements.txt")\nimport tabulate, wcwidth\nprint(wcwidth.wcswidth("表"))\nsaved')
            assert "2\n" in extra["stdout"] and extra["outputs"][0]["data"] == "37"
            manifest = install_snapshot(first)["manifest"]
            assert "tabulate[widechars]<1,>=0.9" in manifest["specifications"]
            assert {"tabulate", "wcwidth"} <= {row["name"] for row in manifest["locked"]}
            report["checks"]["requirements_file_real_extra_dependency"] = True
            pid = call(first, "/console")["status"]["pid"]
            job = call(first, "/environment")["job"]["id"]
            started = time.monotonic()
            execute(first, '!uv pip install "humanize>=4,<5"\nsaved')
            report["repeat_install_ms"] = round((time.monotonic() - started) * 1000, 2)
            assert call(first, "/environment")["job"]["id"] == job
            assert call(first, "/console")["status"]["pid"] == pid
            report["checks"]["matching_range_no_download_or_worker_restart"] = True
            execute(first, '%pip install "humanize>=4,<5"\nsaved')
            assert call(first, "/environment")["manifest"]["specifications"] == manifest["specifications"]
            report["checks"]["pip_compatibility_preserves_uv_declarations"] = True
            execute(first, '%uv pip install "humanize>=4,<5"')
            before = call(first, "/environment")["manifest"]
            assert before["installer"] == "uv"
            for code in ['%uv pip install torch==0.0.1', '%uv pip install "humanize>=99"']:
                failed = call(first, "/console/execute", {"code": code + '\nprint("must-not-run")'})
                assert failed["status"] == "error" and "must-not-run" not in failed["stdout"], failed
                assert call(first, "/environment")["manifest"] == before
            assert execute(first, "saved")["outputs"][0]["data"] == "37"
            report["checks"]["core_and_unsatisfiable_resolution_rollback"] = True
            model = execute(first, 'model=oe.ols(data=oe.example(),y="wage",x=["education","experience"],covariance="HC3")\nassert model.nobs==480\nassert "\\\\toprule" in model.to_latex()\nprint(model.nobs)')
            assert "480\n" in model["stdout"]
            report["checks"]["native_ols_latex_unchanged"] = True
            stop()
            start()
            connect(first)
            execute(first, "import humanize, polars, tabulate, wcwidth\nprint(polars.__version__)")
            assert call(first, "/environment")["manifest"] == before
            report["checks"]["restart_persistence"] = True
            connect(second)
            isolated = call(second, "/console/execute", {"code": "import humanize"})
            assert isolated["error"]["type"] == "ModuleNotFoundError", isolated
            call(second, "/environment/restore", {"manifest": before})
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                restored = call(second, "/environment")
                if restored["job"]["state"] != "running":
                    break
                time.sleep(.25)
            assert restored["job"]["state"] == "complete" and restored["manifest"] == before, restored
            execute(second, "import humanize, polars, tabulate, wcwidth")
            report["checks"]["project_isolation_and_exact_uv_manifest_restore"] = True
            report.update(status="passed", manifest=before, system_python_required=runtime is None)
        except BaseException as exc:
            report.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop()
            stderr.close()
            report["owned_runtime_stopped"] = True
            if report["status"] == "error":
                report["stderr_tail"] = (root / "stderr.log").read_text(errors="replace")[-4000:]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.runtime, args.output)
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))
