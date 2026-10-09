"""Exercise the NSIS-installed frozen runtime using a stdlib-only controller.

The controller's Python never executes application code. Every calculation and
package installation below runs in the installed, frozen console worker with a
sanitized environment and an owned profile. This is Windows x64 acceptance,
not ARM, SmartScreen, code-signing, cloud-login or licensed-vendor validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class InstalledRuntime:
    def __init__(self, executable: Path, root: Path, expected_version: str):
        self.executable, self.root = executable, root
        self.expected_version = expected_version
        self.process = None
        self.descriptor = None
        self.token = None
        self.port = 0
        self.stderr = (root / "runtime.stderr").open("ab")
        self.http = build_opener(ProxyHandler({}))
        self.startups = []

    def start(self):
        environment = {key: value for key, value in os.environ.items()
                       if key.upper() in {"SYSTEMROOT", "WINDIR", "COMSPEC", "LANG"}}
        environment.update({
            "PATH": os.path.join(os.environ["SYSTEMROOT"], "System32"),
            "USERPROFILE": str(self.root / "Kullanıcı profili"),
            "APPDATA": str(self.root / "Kullanıcı profili" / "Roaming"),
            "LOCALAPPDATA": str(self.root / "Kullanıcı profili" / "Local"),
            "TEMP": str(self.root / "Geçici dosyalar"),
            "TMP": str(self.root / "Geçici dosyalar"),
            "PYINSTALLER_RESET_ENVIRONMENT": "1", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        })
        for key in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP"):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        self.process = subprocess.Popen(
            [str(self.executable), "--port", str(self.port), "--data-root", str(self.root)],
            cwd=self.root, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, start_new_session=os.name == "posix",
        )
        ready = queue.Queue(maxsize=1)
        process = self.process

        def read_ready():
            try:
                line = process.stdout.readline(16385)
                if len(line) > 16384 or not line.endswith(b"\n"):
                    raise RuntimeError("Runtime readiness record is missing or oversized.")
                ready.put(json.loads(line))
                # Avoid a blocked worker without exposing ephemeral auth tokens.
                while process.stdout.read(65536):
                    pass
            except BaseException as error:
                if ready.empty():
                    ready.put(error)

        threading.Thread(target=read_ready, name="owned-readiness", daemon=True).start()
        try:
            descriptor = ready.get(timeout=120)
        except queue.Empty as error:
            raise RuntimeError("Installed runtime did not become ready in 120 seconds.") from error
        if isinstance(descriptor, BaseException):
            raise descriptor
        parsed = urlparse(descriptor.get("url", ""))
        if (descriptor.get("type") != "ready" or descriptor.get("protocol_version") != 1
                or parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.port != descriptor.get("port") or parsed.path not in {"", "/"}
                or descriptor.get("pid") != self.process.pid
                or (self.port and parsed.port != self.port)):
            raise RuntimeError("The installed child returned an untrusted readiness record.")
        self.descriptor = descriptor
        self.port = parsed.port
        session = self.call("/api/desktop/session")
        if (session.get("version") != self.expected_version
                or session.get("environment") != "desktop" or not session.get("persistent")):
            raise RuntimeError("Installed desktop session identity is incorrect.")
        self.token = session["token"]
        self.startups.append(round(time.monotonic() - started, 3))

    def call(self, path, body=None, *, token=None):
        headers = {"Content-Type": "application/json"}
        if token or self.token:
            headers["X-OpenEcon-Token"] = token or self.token
        request = Request(self.descriptor["url"] + path, headers=headers,
                          data=json.dumps(body).encode("utf-8") if body is not None else None)
        with self.http.open(request, timeout=300) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise RuntimeError("Runtime response exceeded the verification bound.")
        return json.loads(payload)

    def connect(self, project):
        prefix = f"/api/desktop/projects/{project}/workspace"
        return prefix, self.call(prefix + "/session")["token"]

    def execute(self, prefix, token, code, timeout=240):
        result = self.call(prefix + "/console/execute",
                           {"code": code, "timeout_seconds": timeout}, token=token)
        if result.get("status") != "ok":
            raise RuntimeError(f"Installed worker execution failed: {result.get('error')}")
        return result

    def stop(self):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write(b'{"type":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                if os.name == "nt":
                    subprocess.run([os.path.join(os.environ["SYSTEMROOT"], "System32", "taskkill.exe"),
                                    "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False, timeout=20)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if process.returncode != 0:
            raise RuntimeError(f"The installed runtime did not stop cleanly ({process.returncode}).")
        process.stdin.close()
        process.stdout.close()
        self.process = None
        self.token = None


def marker(result, name):
    values = [json.loads(line[len(name):]) for line in result["stdout"].splitlines()
              if line.startswith(name)]
    if len(values) != 1:
        raise RuntimeError(f"Exactly one {name} receipt is required.")
    return values[0]


def wheel_https_probe_code(expect_denied: bool) -> str:
    """A bounded read-only request in the actual frozen worker, never a mock download."""
    # The same immutable public wheel endpoint is queried on both sides of the
    # owned firewall transition. Actual package resolution/downloads still run
    # independently through the application's unchanged package manager.
    url = ("https://files.pythonhosted.org/packages/c3/5b/"
           "9512c5fb6c8218332b530f13500c6ff5f3ce3342f35e0dd7be9ac3856fd3/"
           "humanize-4.14.0-py3-none-any.whl")
    return f'''
import json, sys
from urllib.error import URLError
from urllib.request import Request, urlopen
assert getattr(sys, "frozen", False)
probe = {{"host":"files.pythonhosted.org", "timeout_seconds":5, "bytes_read":0}}
try:
    with urlopen(Request({url!r}, headers={{"User-Agent":"OpenEconometrics Windows acceptance"}}), timeout=5) as response:
        assert response.status == 200 and response.geturl() == {url!r}
        assert len(response.read(1)) == 1
        probe.update(status="available", http_status=200, bytes_read=1)
except (URLError, OSError) as error:
    reason = error.reason if isinstance(error, URLError) else error
    cause_type = type(reason).__name__
    error_number = getattr(reason, "winerror", None) or getattr(reason, "errno", None)
    # TLS, DNS, redirects, HTTP status and other failures are not evidence that
    # our Windows Internet-block rule denied the request. Never print raw errors.
    denied = isinstance(reason, TimeoutError) or (isinstance(reason, OSError) and error_number in {{13, 10013, 10060}})
    probe.update(status="denied" if denied else "unexpected_failure", cause_type=cause_type, error_number=error_number if isinstance(error_number,int) else None)
    if not {expect_denied!r} or not denied:
        raise RuntimeError("Actual frozen wheel HTTPS probe failed: "+json.dumps(probe)) from None
if {expect_denied!r} and probe["status"] != "denied":
    raise RuntimeError("Actual frozen wheel HTTPS was reachable while the owned Internet block was enabled.")
print("WINDOWS_WHEEL_HTTPS:"+json.dumps(probe))
'''


def migration(executable: Path, root: Path, output: Path, expected_version: str,
              expected_charts_version: str, source_sha: str, phase: str,
              snapshot: Path) -> dict:
    """Create a real old-version project or replay it after install replacement."""
    if os.name != "nt" or os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Migration acceptance requires disposable GitHub Windows.")
    executable = executable.resolve(strict=True)
    root = root.resolve()
    if phase == "prepare":
        if root.exists() or snapshot.exists():
            raise RuntimeError("Migration baseline requires entirely fresh owned paths.")
        root.mkdir(parents=True)
    elif phase != "verify" or not root.is_dir() or not snapshot.is_file():
        raise RuntimeError("Migration replay requires the existing owned baseline.")
    receipt = {"status": "running", "phase": phase, "migration_phase": phase, "source_sha": source_sha,
               "sdk_version": expected_version, "charts_version": expected_charts_version,
               "runtime_sha256": sha256(executable), "human_data_access": False,
               "checks": {}}
    runtime = InstalledRuntime(executable, root, expected_version)
    def history_digest(row):
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    try:
        runtime.start()
        if phase == "prepare":
            project = runtime.call("/api/desktop/local-projects", {
                "name": "Windows upgrade — kalıcı ölçüm", "description": "Owned synthetic migration."})
            runtime.call(f"/api/desktop/local-projects/{project['id']}/open", {})
            prefix, token = runtime.connect(project["id"])
            baseline = runtime.execute(prefix, token, '''
import hashlib, json, math
from pathlib import Path
import openecon as oe
import openecon_charts as charts
from openecon.models import ResultBundle
x = [float(i)/7 for i in range(60)]
y = [4 + 1.75*v + .2*math.sin(i) for i,v in enumerate(x)]
Path("migration.csv").write_text("x,y\\n" + "".join(f"{a},{b}\\n" for a,b in zip(x,y)), encoding="utf-8")
frame = oe.read("migration.csv")
model = oe.ols(data=frame, y="y", x=["x"], covariance="HC3")
Path("migration-model.json").write_text(model.model_dump_json(), encoding="utf-8")
Path("migration-model.tex").write_text(model.to_latex(), encoding="utf-8")
charts.scatter(data=frame, x="x", y="y", title="Migration — saved chart").save_html("migration-chart.html")
assert ResultBundle.model_validate_json(Path("migration-model.json").read_text()).model_dump(mode="json") == model.model_dump(mode="json")
globals()["display"](model)
files = {name: {"sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(), "bytes": Path(name).stat().st_size} for name in ["migration.csv", "migration-model.json", "migration-model.tex", "migration-chart.html"]}
print("WINDOWS_MIGRATION:" + json.dumps({"files": files, "model_sha256": files["migration-model.json"]["sha256"], "nobs": model.nobs}))
''')
            data = marker(baseline, "WINDOWS_MIGRATION:")
            script = runtime.call(prefix + "/console/scripts", {
                "name": "migration-check.py", "code": "print('saved-before-upgrade')"}, token=token)
            history = runtime.call(prefix + "/console", token=token)["history"]
            if not any(row.get("status") == "ok"
                       and "WINDOWS_MIGRATION:" in row.get("stdout", "")
                       and any(item.get("type") == "model" for item in row.get("outputs", []))
                       for row in history):
                raise RuntimeError("The baseline history must contain the real completed model result.")
            environment = runtime.call(prefix + "/environment", token=token)["manifest"]
            record = {"project_id": project["id"], "project_name": project["name"],
                      "files": data["files"], "script": {"id": script["id"], "code": script["code"]},
                      "history_ids": [row["id"] for row in history],
                      "history_sha256": [history_digest(row) for row in history],
                      "environment": environment,
                      "prepared_runtime_sha256": receipt["runtime_sha256"],
                      "prepared_source_sha": source_sha, "prepared_sdk_version": expected_version}
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            receipt["checks"]["real_old_version_project_model_chart_script_history_saved"] = True
        else:
            record = json.loads(snapshot.read_text(encoding="utf-8"))
            project_id = record["project_id"]
            if not any(row["id"] == project_id and row["name"] == record["project_name"]
                       for row in runtime.call("/api/desktop/local-projects")["projects"]):
                raise RuntimeError("The old-version project did not survive install replacement.")
            runtime.call(f"/api/desktop/local-projects/{project_id}/open", {})
            prefix, token = runtime.connect(project_id)
            project_root = root / "projects" / project_id
            for name, expected in record["files"].items():
                path = project_root / name
                if path.is_symlink() or sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                    raise RuntimeError("A persisted migration artifact changed during installation.")
            if runtime.call(prefix + f"/console/scripts/{record['script']['id']}", token=token)["code"] != record["script"]["code"]:
                raise RuntimeError("The old saved editor document did not survive.")
            history = runtime.call(prefix + "/console", token=token)["history"]
            if [row["id"] for row in history[:len(record["history_ids"])]] != record["history_ids"]:
                raise RuntimeError("The old execution history did not survive.")
            if [history_digest(row) for row in history[:len(record["history_sha256"])]] != record["history_sha256"]:
                raise RuntimeError("Original execution code, stdout, model outputs or metadata changed.")
            if runtime.call(prefix + "/environment", token=token)["manifest"] != record["environment"]:
                raise RuntimeError("The old project environment manifest changed.")
            replay = runtime.execute(prefix, token, '''
from pathlib import Path
from openecon.models import ResultBundle
saved = ResultBundle.model_validate_json(Path("migration-model.json").read_text(encoding="utf-8"))
assert saved.nobs == 60
assert saved.to_latex() == Path("migration-model.tex").read_text(encoding="utf-8")
assert "OpenEconCharts.mount" in Path("migration-chart.html").read_text(encoding="utf-8")
print("WINDOWS_MIGRATION_REPLAY_OK")
''')
            if "WINDOWS_MIGRATION_REPLAY_OK" not in replay["stdout"]:
                raise RuntimeError("The replaced frozen runtime could not replay old saved results.")
            receipt["checks"].update(project_and_four_artifact_hashes_preserved=True,
                                     saved_editor_history_and_environment_preserved=True,
                                     actual_replaced_frozen_runtime_replay_without_refit=True)
        runtime.stop()
        receipt.update(status="passed", owned_runtime_stopped=True,
                       snapshot_sha256=sha256(snapshot), ready_seconds=runtime.startups)
    except BaseException as error:
        receipt.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            runtime.stop()
        finally:
            runtime.stderr.close()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def verify(executable: Path, root: Path, output: Path, expected_version: str,
           expected_charts_version: str, source_sha: str, *, offline_only: bool = False) -> dict:
    if os.name != "nt":
        raise RuntimeError("Installed Windows acceptance must run on Windows.")
    executable = executable.resolve(strict=True)
    root = root.resolve()
    if root.exists():
        raise RuntimeError("An entirely new owned runtime profile is required.")
    root.mkdir(parents=True)
    runtime = InstalledRuntime(executable, root, expected_version)
    receipt = {"status": "running", "platform": "windows-x64", "source_sha": source_sha,
               "runtime": str(executable), "runtime_sha256": sha256(executable),
               "sdk_version": expected_version, "charts_version": expected_charts_version,
               "acceptance_mode": ("offline_numerical_only" if offline_only
                                   else "online_packages_and_persistence"),
               "human_data_access": False, "system_python_used_by_application": False,
               "checks": {}, "boundaries": ["unsigned binary / SmartScreen",
               "physical Windows machine", "Windows ARM64", "live cloud authentication",
               "firewall first-run interaction", "all-method or licensed-vendor parity"]}
    try:
        runtime.start()
        try:
            runtime.call("/api/desktop/status", token="invalid-owned-test-token")
        except HTTPError as error:
            if error.code != 401:
                raise
        else:
            raise RuntimeError("A wrong desktop session token was accepted.")
        receipt["checks"]["loopback_session_token_enforcement"] = True
        first = runtime.call("/api/desktop/local-projects", {
            "name": "Windows kurulum testi — ölçüm", "description": "Disposable synthetic acceptance."})
        runtime.call(f"/api/desktop/local-projects/{first['id']}/open", {})
        prefix, token = runtime.connect(first["id"])
        if not offline_only:
            probe = runtime.execute(prefix, token, wheel_https_probe_code(False))
            receipt["wheel_https"] = marker(probe, "WINDOWS_WHEEL_HTTPS:")
            receipt["checks"]["actual_installed_runtime_pypi_wheel_https_available"] = True
        code = f'''
import hashlib, importlib.util, json, math, os, socket, sys
from pathlib import Path
assert getattr(sys, "frozen", False)
bundle = Path(sys._MEIPASS).resolve()
assert Path(sys.executable).resolve() == Path({str(executable)!r})
allowed_roots = [Path(sys.executable).resolve().parent, Path({str(root)!r}).resolve()]
assert all(any(Path(value or Path.cwd()).resolve().is_relative_to(allowed) for allowed in allowed_roots) for value in sys.path)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
attempts = []
original_connect = socket.socket.connect
def deny_network(_socket, address):
    attempts.append(type(address).__name__)
    raise OSError("Synthetic offline computation guard.")
socket.socket.connect = deny_network
try:
    import torch, openecon as oe, openecon_charts as charts
    from openecon.models import ResultBundle
    assert oe.__version__ == {expected_version!r}
    assert charts.__version__ == {expected_charts_version!r}
    assert torch.version.cuda is None and not torch.cuda.is_available()
    assert all(Path(module.__file__).resolve().is_relative_to(bundle) for module in (torch, oe, charts))
    torch.set_num_threads(1)
    xs = [float(i) / 7 for i in range(60)]
    ys = [4.0 + 1.75 * x + 0.2 * math.sin(i) for i, x in enumerate(xs)]
    Path("ölçüm örneği.csv").write_text("x,y\\n" + "".join(f"{{x}},{{y}}\\n" for x, y in zip(xs, ys)), encoding="utf-8")
    frame = oe.read("ölçüm örneği.csv")
    model = oe.ols(data=frame, y="y", x=["x"], covariance="HC3")
    mx, my = sum(xs)/60, sum(ys)/60
    denominator = sum((x-mx)**2 for x in xs)
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / denominator
    intercept = my-slope*mx
    # Independent scalar HC3 oracle, including uncertainty rather than only coefficients.
    inverse = [[1/60+mx*mx/denominator, -mx/denominator], [-mx/denominator, 1/denominator]]
    meat = [[0.0,0.0],[0.0,0.0]]
    for x,y in zip(xs,ys):
        vector = [1.0,x]
        leverage = sum(vector[i]*inverse[i][j]*vector[j] for i in range(2) for j in range(2))
        adjusted = ((y-intercept-slope*x)/(1-leverage))**2
        for i in range(2):
            for j in range(2):
                meat[i][j] += adjusted*vector[i]*vector[j]
    covariance = [[sum(inverse[i][a]*meat[a][b]*inverse[b][j] for a in range(2) for b in range(2)) for j in range(2)] for i in range(2)]
    assert model.nobs == 60 and model.nobs_original == 60 and model.dropped_rows == 0
    assert model.sample_positions == list(range(60))
    assert math.isclose(model.coefficients[0].estimate,intercept,abs_tol=1e-10)
    assert math.isclose(model.coefficients[1].estimate,slope,abs_tol=1e-10)
    assert all(math.isclose(model.covariance_matrix[i][j],covariance[i][j],rel_tol=1e-9,abs_tol=1e-12) for i in range(2) for j in range(2))
    Path("model.json").write_text(model.model_dump_json(),encoding="utf-8")
    restored = ResultBundle.model_validate_json(Path("model.json").read_text(encoding="utf-8"))
    assert restored.model_dump(mode="json") == model.model_dump(mode="json")
    assert restored.to_latex() == model.to_latex() and "\\\\toprule" in model.to_latex()
    Path("model.tex").write_text(model.to_latex(),encoding="utf-8")
    plot = charts.scatter(data=frame,x="x",y="y",title="Windows — ölçüm")
    plot.save_html("grafik.html")
    assert plot.sample_n == 60
    html = Path("grafik.html").read_text(encoding="utf-8")
    assert "OpenEconCharts.mount" in html and "chart-data" in html
    assert "<script src=" not in html and "data:font/woff2;base64," in html
    globals()["display"](model)
    globals()["display"](plot)
    assert not attempts
    qa_value = 37
    files = {{name: {{"sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(),"bytes":Path(name).stat().st_size}} for name in ["ölçüm örneği.csv","model.json","model.tex","grafik.html"]}}
    print("WINDOWS_CPU:"+json.dumps({{"frozen":True,"bundle_imports":True,"python":sys.version.split()[0],"torch":torch.__version__,"cpu_only":True,"nobs":60,"scalar_ols_and_hc3_oracle":True,"json_latex_roundtrip":True,"offline_chart":True,"blocked_network_attempts":len(attempts),"files":files}}))
finally:
    socket.socket.connect = original_connect
'''
        result = runtime.execute(prefix, token, code)
        receipt["cpu"] = marker(result, "WINDOWS_CPU:")
        if not {"model", "plot"} <= {row["type"] for row in result["outputs"]}:
            raise RuntimeError("Actual worker did not produce both model and chart display outputs.")
        saved = runtime.call(prefix + "/console/scripts", {"name": "ölçüm testi.py",
                             "code": "print('persisted-windows-script')"}, token=token)
        receipt["checks"]["frozen_cpu_ols_hc3_json_latex_offline_chart"] = True
        receipt["checks"]["real_model_and_plot_console_outputs"] = True
        if offline_only:
            probe = runtime.execute(prefix, token, wheel_https_probe_code(True))
            receipt["wheel_https"] = marker(probe, "WINDOWS_WHEEL_HTTPS:")
            receipt["checks"]["actual_installed_runtime_pypi_wheel_https_denied"] = True
            runtime.stop()
            receipt.update(status="passed", ready_seconds=runtime.startups,
                           loopback_port=runtime.port, owned_runtime_stopped=True)
            return receipt
        install = runtime.execute(prefix, token,
            '%uv pip install "humanize==4.14.0" "polars==1.44.2"\n'
            'import humanize, polars\nfrom pathlib import Path\n'
            'assert humanize.__version__=="4.14.0"\nassert polars.__version__=="1.44.2"\n'
            'assert ".packages" in str(Path(humanize.__file__).resolve())\n'
            'assert ".packages" in str(Path(polars.__file__).resolve())\n'
            'assert humanize.intcomma(12345)=="12,345"\n'
            'assert polars.DataFrame({"x":[1,2,3]}).select(polars.col("x").sum()).item()==6\n'
            'assert qa_value==37\nprint("WINDOWS_PACKAGES_OK")')
        if "WINDOWS_PACKAGES_OK" not in install["stdout"]:
            raise RuntimeError("Installed pure/native packages did not execute.")
        environment = runtime.call(prefix + "/environment", token=token)
        manifest = environment["manifest"]
        if environment["job"]["state"] != "complete" or manifest["installer"] != "uv":
            raise RuntimeError("Real bundled uv installation did not complete.")
        status = runtime.call(prefix + "/console", token=token)["status"]
        worker_pid, generation = status["pid"], status["session_generation"]
        job_id = environment["job"]["id"]
        if (type(worker_pid) is not int or worker_pid <= 0
                or type(generation) is not int or generation <= 0 or not job_id):
            raise RuntimeError("The installed package no-op baseline has no live worker identity.")
        def manifest_hash(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()

        receipt["package_noop_baseline"] = {
            "worker_pid": worker_pid, "session_generation": generation,
            "package_job_id": job_id, "manifest_sha256": manifest_hash(manifest),
        }
        receipt["package_noop_transitions"] = []
        package_probe = '''
import humanize, polars, os, json
from pathlib import Path
assert humanize.__version__ == "4.14.0"
assert polars.__version__ == "1.44.2"
assert ".packages" in str(Path(humanize.__file__).resolve())
assert ".packages" in str(Path(polars.__file__).resolve())
assert humanize.intcomma(12345) == "12,345"
native_sum = polars.DataFrame({"x": [1, 2, 3]}).select(polars.col("x").sum()).item()
assert native_sum == 6 and qa_value == 37
print("WINDOWS_PACKAGE_NOOP:" + json.dumps({"pid": os.getpid(),
    "humanize_version": humanize.__version__, "polars_version": polars.__version__,
    "native_sum": native_sum, "qa_value": qa_value}))
'''
        for command, installer in [("%pip install", "pip"), ("%uv pip install", "uv")]:
            expected = {**manifest, "installer": installer}
            result = runtime.execute(prefix, token,
                                     command + ' "humanize==4.14.0"\n' + package_probe)
            probe = marker(result, "WINDOWS_PACKAGE_NOOP:")
            observed = runtime.call(prefix + "/environment", token=token)
            status = runtime.call(prefix + "/console", token=token)["status"]
            exact_manifest = observed["manifest"] == expected
            same_worker = (status["pid"] == worker_pid == probe["pid"]
                           and status["session_generation"] == generation
                           and result.get("session_generation") == generation)
            same_job = (observed["job"]["id"] == job_id
                        and observed["job"]["state"] == "complete")
            receipt["package_noop_transitions"].append({
                "requested_installer": installer, "manifest_exact_expected": exact_manifest,
                "expected_manifest_sha256": manifest_hash(expected),
                "observed_manifest_sha256": manifest_hash(observed["manifest"]),
                "worker_unchanged": same_worker, "package_job_unchanged": same_job,
                "worker_pid": status["pid"], "session_generation": status["session_generation"],
                "package_job_id": observed["job"]["id"], "python_probe": probe,
            })
            if not exact_manifest or not same_worker or not same_job:
                raise RuntimeError(f"Matching {installer} request changed the installed worker, job or exact expected manifest.")
        # Switching the installer label is declarative; restore the complete UV
        # manifest before checking protected-package rollback and project relaunch.
        if observed["manifest"] != manifest:
            raise RuntimeError("Matching uv request did not restore the exact saved UV manifest.")
        refused = runtime.call(prefix + "/console/execute", {
            "code": '%uv pip install torch==0.0.1\nprint("must-not-execute")'}, token=token)
        if refused.get("status") != "error" or "must-not-execute" in refused.get("stdout", ""):
            raise RuntimeError("A protected bundled dependency installation was accepted.")
        if runtime.call(prefix + "/environment", token=token)["manifest"] != manifest:
            raise RuntimeError("Rejected core installation changed the package manifest.")
        runtime.execute(prefix, token, "assert qa_value==37")
        receipt["checks"]["bundled_uv_pure_and_native_windows_wheels"] = True
        receipt["checks"]["pip_compatibility_and_same_script_state"] = True
        receipt["checks"]["protected_core_installation_rollback"] = True
        receipt["manifest"] = manifest
        runtime.stop()
        runtime.start()
        if first not in runtime.call("/api/desktop/local-projects")["projects"]:
            raise RuntimeError("The actual local project did not persist across relaunch.")
        runtime.call(f"/api/desktop/local-projects/{first['id']}/open", {})
        prefix, token = runtime.connect(first["id"])
        if runtime.call(prefix + f"/console/scripts/{saved['id']}", token=token)["code"] != saved["code"]:
            raise RuntimeError("The saved Python editor document did not persist.")
        if runtime.call(prefix + "/environment", token=token)["manifest"] != manifest:
            raise RuntimeError("The exact package manifest did not persist.")
        expected_files = receipt["cpu"]["files"]
        restore = runtime.execute(prefix, token, f'''
import hashlib,json
from pathlib import Path
import humanize,polars
from openecon.models import ResultBundle
expected = {expected_files!r}
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==row["sha256"] for name,row in expected.items())
saved_model=ResultBundle.model_validate_json(Path("model.json").read_text(encoding="utf-8"))
assert saved_model.to_latex()==Path("model.tex").read_text(encoding="utf-8")
assert humanize.intcomma(12345)=="12,345"
assert polars.DataFrame({{"x":[1,2,3]}}).select(polars.col("x").sum()).item()==6
print("WINDOWS_RELAUNCH_OK")
''')
        if "WINDOWS_RELAUNCH_OK" not in restore["stdout"]:
            raise RuntimeError("The relaunched installed worker did not read persisted artifacts.")
        history = runtime.call(prefix + "/console", token=token)
        receipt["persisted_console_history_entries"] = len(history.get("history", []))
        if receipt["persisted_console_history_entries"] < 3:
            raise RuntimeError("The saved console history did not persist across relaunch.")
        receipt["checks"]["stable_loopback_port_and_full_persisted_project_relaunch"] = True
        second = runtime.call("/api/desktop/local-projects", {"name": "Bağımsız ikinci proje", "description": "Isolation check."})
        runtime.call(f"/api/desktop/local-projects/{second['id']}/open", {})
        second_prefix, second_token = runtime.connect(second["id"])
        isolated = runtime.call(second_prefix + "/console/execute", {"code": "import humanize"}, token=second_token)
        if isolated.get("status") != "error" or isolated.get("error", {}).get("type") != "ModuleNotFoundError":
            raise RuntimeError("The package overlay leaked into a different project.")
        receipt["checks"]["per_project_package_isolation"] = True
        runtime.stop()
        receipt.update(status="passed", ready_seconds=runtime.startups,
                       loopback_port=runtime.port, owned_runtime_stopped=True)
        receipt["files"] = {name: {"sha256": sha256(root / "projects" / first["id"] / name),
                           "bytes": (root / "projects" / first["id"] / name).stat().st_size}
                            for name in expected_files}
    except BaseException as error:
        receipt.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            runtime.stop()
        finally:
            runtime.stderr.close()
            if receipt["status"] == "error":
                receipt["stderr_tail"] = (root / "runtime.stderr").read_text(errors="replace")[-4000:]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-version", required=True)
    parser.add_argument("--charts-version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--offline-only", action="store_true",
                        help="Prove numerical work and actual denied wheel HTTPS under the owned block; never install packages.")
    parser.add_argument("--migration-phase", choices=("prepare", "verify"))
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.migration_phase:
        if args.offline_only:
            parser.error("offline numerical acceptance is separate from migration")
        if args.snapshot is None:
            parser.error("--snapshot is required for migration acceptance")
        result = migration(args.runtime, args.profile, args.output, args.sdk_version,
                           args.charts_version, args.source_sha,
                           args.migration_phase, args.snapshot)
    else:
        result = verify(args.runtime, args.profile, args.output, args.sdk_version,
                        args.charts_version, args.source_sha, offline_only=args.offline_only)
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))
