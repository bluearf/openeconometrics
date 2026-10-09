"""Measure original Dataset cases in isolated workers of an installed QA app.

Numerical reference is the separately recorded physical resident run. Actual
native UI interactions are a separate receipt. No human project is accessed.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import plistlib
import select
import signal
import subprocess
import sys
import tempfile
import time
from types import CodeType
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from native_model_matrix_replay import compare, digest, write  # noqa: E402


def bundled_identity(app, expected_identifier="org.openecon.qa.networkmodels"):
    from PyInstaller.archive.readers import CArchiveReader
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    if expected_identifier not in {"org.openecon.qa.networkmodels", "org.openecon.qa.priorityseven"}:
        raise ValueError("Only explicitly owned QA identities are accepted")
    if info["CFBundleIdentifier"] != expected_identifier:
        raise ValueError("This verifier only owns the explicitly isolated QA application")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True,
                   capture_output=True)
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalize(code):
        return code.replace(co_filename="<verified>", co_consts=tuple(
            normalize(value) if isinstance(value, CodeType) else value for value in code.co_consts))

    hashes = {}
    # The local bundle deliberately excludes cloud/team server and CLI entry
    # points. They are outside this local analysis acceptance denominator.
    nonlocal_modules = {"team_job", "team_dispatch", "cloud", "team_history", "team_recovery",
        "team_sandbox_broker", "team_storage", "team_server", "desktop_cloud", "team_runner",
        "team_sandbox_worker", "team_cloud", "cli", "team_sandbox_runner", "team_auth",
        "sandbox_privileges", "entrypoints", "team_sandbox_auth", "team_store", "sandbox_rootfs",
        "team_transfer"}  # Server-side membership/storage routes, imported only by team_server.
    for source in sorted((ROOT / "src/openecon").rglob("*.py")):
        module = str(source.relative_to(ROOT / "src")).removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        if module.removeprefix("openecon.") in nonlocal_modules:
            continue
        if normalize(archive.extract(module)) != normalize(compile(
                source.read_text(), str(source), "exec", dont_inherit=True)):
            raise RuntimeError("Frozen module differs from source: " + module)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    return {"installed_app": str(app), "identifier": info["CFBundleIdentifier"],
            "shell_version": info["CFBundleShortVersionString"], "strict_ad_hoc_signature": True,
            "runtime_executable_sha256": digest(runtime), "modules_sha256": hashes,
            "all_local_analysis_sdk_embedded_code_matches_source": True,
            "unmeasured_nonlocal_modules": sorted(nonlocal_modules),
            "runtime_manifest": json.loads((app / "Contents/Resources/runtime/runtime-manifest.json").read_text()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def injected_worker(case_path, target):
    source = ROOT / "benchmarks/native_model_matrix_replay.py"
    tree = ast.parse(source.read_text())
    functions = {"digest", "write", "peak_rss_bytes", "postest", "worker", "compare"}
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) or
             isinstance(node, ast.FunctionDef) and node.name in functions]
    program = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    return program + f"""
from types import SimpleNamespace
_status = worker(SimpleNamespace(case=Path({str(case_path)!r}), directory=Path({str(target)!r}),
                                 mode='replay', skip_warm=False))
assert _status == 0, 'Physical model worker failed; inspect its receipt'
import openecon as oe
assert getattr(sys, 'frozen', False)
assert str(oe.__file__).startswith(str(sys._MEIPASS))
assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)
print('NATIVE_MATRIX_BUNDLE:' + json.dumps({{'frozen': True, 'sdk_from_bundle': True,
    'no_scipy_loaded': True}}))
display(oe.ResultBundle.model_validate_json((Path({str(target)!r})/'result.json').read_text()))
"""


def shard(runtime, cases, directory, references):
    records = []
    with tempfile.TemporaryDirectory(prefix="openecon-matrix-frozen-") as temporary:
        data_root = Path(temporary)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        with (directory / (uuid4().hex + "-runtime.log")).open("wb") as errors:
            process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(data_root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                env=environment, start_new_session=True)
            token = None
            prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
            try:
                deadline = time.monotonic() + 60
                descriptor = None
                while time.monotonic() < deadline:
                    if select.select([process.stdout], [], [], .25)[0]:
                        descriptor = json.loads(process.stdout.readline(8193))
                        break
                    if process.poll() is not None:
                        raise RuntimeError("Frozen worker exited before ready")
                if not descriptor or descriptor.get("type") != "ready":
                    raise RuntimeError("Frozen worker did not become ready")
                parsed = urlparse(descriptor["url"])
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
                    raise ValueError("Invalid frozen loopback descriptor")

                def call(path, body=None):
                    headers = {"Content-Type": "application/json"}
                    if token:
                        headers["X-OpenEcon-Token"] = token
                    request = Request(descriptor["url"] + prefix + path, headers=headers,
                        data=json.dumps(body).encode() if body is not None else None)
                    with urlopen(request, timeout=660) as response:
                        return json.load(response)

                token = call("/session")["token"]
                for case in cases:
                    target = directory / case["id"]
                    target.mkdir(exist_ok=False)
                    write(target / "case.json", case)
                    code = injected_worker(target / "case.json", target)
                    (target / "injected.py").write_text(code)
                    started = time.perf_counter()
                    future = None
                    scratch_peak = 0
                    with ThreadPoolExecutor(max_workers=1) as pending:
                        future = pending.submit(call, "/console/execute", {"code": code, "timeout_seconds": 600})
                        while not future.done():
                            try:
                                scratch_peak = max(scratch_peak, sum(p.stat().st_size for p in
                                    (target / "scratch").rglob("*") if p.is_file()))
                            except FileNotFoundError:
                                pass
                            time.sleep(.05)
                        run = future.result()
                    write(target / "console.json", run)
                    receipt_path = target / "receipt.json"
                    measured = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                    call("/console/reset", {})
                    saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
                    history_equal = saved["outputs"] == run["outputs"] and saved["events"] == run["events"]
                    # Source controllers can still be completing other cases.
                    # Wait for both committed worker receipts, then verify the
                    # same physical source identity before reading any numbers.
                    reference_started = time.monotonic()
                    one = None
                    while time.monotonic()-reference_started < 600:
                        one = next((root / case["id"] for root in references
                            if (root / case["id"] / "resident/receipt.json").is_file()
                            and (root / case["id"] / "replay/receipt.json").is_file()), None)
                        if one is not None:
                            break
                        time.sleep(.5)
                    if one is None:
                        raise RuntimeError("Missing measured source reference: " + case["model"])
                    assert json.loads((one / "case.json").read_text())["source_sha256"] == case["source_sha256"]
                    if run["status"] == "ok" and measured.get("status") == "passed":
                        expected = json.loads((one / "resident/result.json").read_text())
                        observed = json.loads((target / "result.json").read_text())
                        relative = 1e-3 if case["model"] in {"ucm", "mswitch", "arch", "arima"} else 5e-5
                        checked = compare(expected, observed, relative=relative, absolute=2e-6)
                        native_helper = measured["postest"]
                        source_helper = json.loads((one / "replay/receipt.json").read_text())["postest"]
                        if not native_helper["helper"].startswith("persisted physical"):
                            helper_checked = compare({**expected, "tests": source_helper},
                                {**expected, "tests": native_helper}, relative=relative, absolute=2e-6)
                            checked["postest_errors"] = helper_checked["errors"]
                            if helper_checked["status"] != "passed":
                                checked["status"] = "failed"
                    else:
                        checked = {"status": "not_run", "error": run.get("error")}
                    output_ok = bool(run["outputs"]) and all(output["type"] in {"model", "table"} and
                        "\\begin{tabular}" in output.get("latex", "") for output in run["outputs"])
                    bundle_ok = 'NATIVE_MATRIX_BUNDLE:' in run.get("stdout", "")
                    record = {"case": case, "measurement": measured, "comparison": checked,
                        "console_status": run["status"], "console_execution_id": run["id"],
                        "reference_wait_seconds": time.monotonic()-reference_started,
                        "console_wall_seconds": time.perf_counter()-started,
                        "output_tables": len(run["outputs"]), "output_table_latex_readback": output_ok,
                        "history_exact_after_worker_reset": history_equal,
                        "frozen_sdk_from_bundle_no_scipy": bundle_ok,
                        "sampled_scratch_peak_bytes": scratch_peak, "scratch_monitor_seconds": .05,
                        "injected_worker_sha256": hashlib.sha256(code.encode()).hexdigest()}
                    record["status"] = "passed" if checked["status"] == "passed" and history_equal and \
                        output_ok and bundle_ok and measured.get("source_unchanged") and \
                        not measured.get("scratch_remaining") else "failed"
                    write(target / "native-evidence.json", record)
                    records.append(record)
                    print(json.dumps({"native_model": case["model"], "status": record["status"]}), flush=True)
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
        for record in records:
            record["owned_runtime_stopped"] = process.poll() is not None
    for record in records:
        record["owned_temporary_project_removed"] = not data_root.exists()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--expected-identifier", default="org.openecon.qa.networkmodels")
    parser.add_argument("--jobs", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    # Workers execute inside isolated project directories. Resolve all host
    # paths before injecting them so relative CLI paths retain their meaning.
    args.app = args.app.resolve()
    args.manifest = args.manifest.resolve()
    args.directory = args.directory.resolve()
    args.source_directories = [path.resolve() for path in args.source_directories]
    args.directory.mkdir(parents=True, exist_ok=False)
    identity = bundled_identity(args.app, args.expected_identifier)
    identity["verifier_sha256"] = digest(__file__)
    references = args.source_directories
    cases = [case for case in json.loads(args.manifest.read_text())["cases"] if
             not args.models or case["model"] in args.models]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("A nonempty, unique physical case inventory is required")
    runtime = args.app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        work = [pool.submit(shard, runtime, cases[slot::args.jobs], args.directory, references)
                for slot in range(args.jobs)]
        for done in as_completed(work):
            try:
                records.extend(done.result())
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
            write(args.directory / "matrix.json", {"stage": "installed_frozen_runtime_workers",
                "identity": identity, "cases": sorted(records, key=lambda r: r["case"]["model"]),
                "passed": sum(r["status"] == "passed" for r in records), "measured": len(records),
                "concurrent_owned_runtime_processes": args.jobs, "worker_torch_threads_each": 2,
                "cache_boundary": "Fresh console worker per model; shared host OS pages; no cold-cache claim",
                "native_ui_verified": False, "human_project_access": False})
    # A transport/reference/reset exception may stop a shard after it saved some
    # cases. Recover those receipts and retain every remaining planned case.
    recorded = {record["case"]["id"]: record for record in records}
    for case in cases:
        if case["id"] not in recorded:
            receipt = args.directory / case["id"] / "native-evidence.json"
            recorded[case["id"]] = json.loads(receipt.read_text()) if receipt.is_file() else {
                "case": case, "status": "not_run", "scientific_result": "unverified",
                "controller_errors": errors,
                "attempt_started": (args.directory / case["id"] / "case.json").is_file()}
    records = [recorded[case["id"]] for case in cases]
    unchanged = digest(runtime) == identity["runtime_executable_sha256"]
    passed = unchanged and not errors and all(record["status"] == "passed" for record in records)
    write(args.directory / "matrix.json", {"stage": "installed_frozen_runtime_workers",
        "status": "passed" if passed else "failed", "identity": identity, "cases": records,
        "planned": len(cases), "passed": sum(row["status"] == "passed" for row in records),
        "not_run": sum(row["status"] == "not_run" for row in records),
        "controller_errors": errors, "runtime_unchanged": unchanged,
        "concurrent_owned_runtime_processes": args.jobs, "worker_torch_threads_each": 2,
        "cache_boundary": "Fresh console worker per model; shared host OS pages; no cold-cache claim",
        "native_ui_verified": False, "human_project_access": False})
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
