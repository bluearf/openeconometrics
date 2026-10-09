"""Full source/frozen/installed result readback for eight planning extensions.

The installed modes seed a private synthetic script or read saved results.
Only the native UI invokes Run there. Frozen mode owns its temporary server.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import plistlib
import runpy
import socket
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
import zipfile

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/prospective_extensions.py"
MARKER = "PROSPECTIVE_EXTENSIONS_OK "
METHODS = (
    "power_welch",
    "power_unbalanced_anova",
    "power_unequal_cluster_mean",
    "power_mcnemar_unconditional",
    "precision_twomeans_unknown",
    "precision_binomial",
    "precision_poisson",
    "survival_accrual",
)
MODULES = (
    "openecon.econometrics.stats",
    "openecon.econometrics.stats.planning",
    "openecon.econometrics.stats.planning_extended",
    "openecon.econometrics.stats.planning_heterogeneous",
    "openecon.econometrics.stats.planning_precision",
    "openecon.econometrics.stats.planning_discrete_survival",
    "openecon.econometrics.stats.power_distributions",
    "openecon.econometrics.stats.power_designs",
    "openecon.econometrics.nonparametric.distribution",
    "openecon.engines.distributions",
    "openecon.econometrics.summary_state",
    "openecon.analysis_contracts",
)
IDENTIFIER = "org.openecon.qa.prospective8"
PRODUCT = "OpenEconometrics Prospective QA"
APP = Path.home() / "Applications" / f"{PRODUCT}.app"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Prospective Extensions Eight QA"


def process_snapshot():
    listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,lstart=,command="], text=True)
    rows = []
    for line in listing.splitlines():
        fields = line.split(None, 7)
        if len(fields) == 8:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "parent": int(fields[1]),
                    "started_at": " ".join(fields[2:7]),
                    "owned_executable": fields[7].startswith(str(APP) + "/"),
                }
            )
    return rows


def owned_processes():
    rows = process_snapshot()
    owned = {row["pid"] for row in rows if row["owned_executable"]}
    while True:
        descendants = {row["pid"] for row in rows if row["parent"] in owned}
        if descendants <= owned:
            break
        owned |= descendants
    return [row for row in rows if row["pid"] in owned]


def stopped_check(receipt_path):
    before = json.loads(receipt_path.read_text())
    assert before["layer"] == "installed_native" and before["owned_processes"]
    assert owned_processes() == []
    previous = {(row["pid"], row["started_at"]) for row in before["owned_processes"]}
    assert not previous & {(row["pid"], row["started_at"]) for row in process_snapshot()}
    with socket.socket() as connection:
        connection.settimeout(1)
        assert connection.connect_ex(("127.0.0.1", before["port"])) != 0
    before["full_native_shutdown_verified"] = True
    return before


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_identity(runtime):
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    hashes = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        assert normalized(archive.extract(name)) == normalized(
            compile(path.read_text(), str(path), "exec", dont_inherit=True)
        ), name
        hashes[name] = digest(path)
    return hashes


def header(directory):
    return (
        "import sys, importlib, importlib.util\nfrom pathlib import Path\n"
        "assert getattr(sys, 'frozen', False)\n"
        "assert importlib.util.find_spec('scipy') is None\n"
        "assert importlib.util.find_spec('statsmodels') is None\n"
        f"for module_name in {MODULES!r}:\n"
        "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
        f"PROSPECTIVE_EXTENSION_RESULT_DIRECTORY = {str(directory)!r}\n"
    )


def proof_from(stdout):
    matches = [line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)]
    assert len(matches) == 1
    proof = json.loads(matches[0])
    assert set(proof["methods"]) == set(METHODS)
    assert proof["full_input_replay_verified"] and proof["full_summary_restore_verified"]
    assert proof["third_party_estimation_imports"] == []
    assert proof["displayed_tables"] == 16
    return proof


def verify_outputs(run):
    assert run["status"] == "ok", run.get("error")
    proof = proof_from(run["stdout"])
    assert proof["frozen"]
    assert [output["type"] for output in run["outputs"]] == ["table"] * 16
    for i, name in enumerate(METHODS):
        for offset, key in enumerate(("plan", "scenarios")):
            output = run["outputs"][2 * i + offset]
            assert len(output["data"]["rows"]) == output["data"]["total_rows"]
            assert len(output["data"]["rows"]) == proof["methods"][name]["table_rows"][key]
            assert any(
                "\\begin{" + env + "}" in output["latex"] for env in ("tabular", "longtable")
            )
    return proof


def verify_files(directory, proof):
    hashes = {}
    for name in METHODS:
        path = directory / (name + ".json")
        hashes[name] = digest(path)
        assert hashes[name] == proof["methods"][name]["sha256"]
        saved = json.loads(path.read_text())
        assert saved["summary"]["schema"] == "openecon.summary.v1"
        assert saved["summary"]["attrs"]["prospective"]
        assert saved["summary"]["attrs"]["observations_used"] is False
        assert {
            key: len(frame["data"]) for key, frame in saved["summary"]["tables"].items()
        } == proof["methods"][name]["table_rows"]
        assert "\\begin{tabular}" in saved["latex"]
    return hashes


def source_check():
    with tempfile.TemporaryDirectory(prefix="openecon-prospective-source-") as temporary:
        directory, stdout, displayed = Path(temporary), io.StringIO(), []
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(
                str(EXAMPLE),
                init_globals={
                    "PROSPECTIVE_EXTENSION_RESULT_DIRECTORY": str(directory),
                    "display": lambda frame: displayed.append(frame),
                },
            )
        proof = proof_from(stdout.getvalue())
        assert not proof["frozen"] and len(displayed) == 16
        hashes = verify_files(directory, proof)
    return dict(status="passed", layer="source", proof=proof, complete_result_hashes=hashes)


def frozen_check(runtime):
    runtime = runtime.resolve(strict=True)
    sources = source_identity(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-prospective-frozen-") as temporary:
        root = Path(temporary)
        directory = root / "complete-results"
        code = header(directory) + EXAMPLE.read_text()
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            project = owned.request("/api/desktop/local-projects", {"name": PROJECT}, desktop=True)[
                "id"
            ]
            owned.project = project
            owned.open_project()
            owned.call("/console/script", {"code": code}, method="PUT")
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = verify_outputs(run)
            hashes = verify_files(directory, proof)
            original = {key: run[key] for key in ("code", "stdout", "outputs", "events")}
            document = owned.call("/console/script")
        finally:
            owned.close()
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project()
            saved = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
            assert original == {key: saved[key] for key in original}
            assert owned.call("/console/script") == document
            assert verify_files(directory, proof) == hashes
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
        finally:
            owned.close()
    return dict(
        status="passed",
        layer="frozen",
        proof=proof,
        complete_result_hashes=hashes,
        compiled_modules_equal_source=sources,
        runtime_sha256=digest(runtime),
        full_history_and_document_equal_after_server_restart=True,
        execution_id=run["id"],
        duration_ms=run["duration_ms"],
        owned_runtime_stopped=True,
        temporary_data_removed=not root.exists(),
        native_window_verified=False,
    )


def wheel_check(wheel, charts_wheel):
    with tempfile.TemporaryDirectory(prefix="openecon-prospective-wheel-") as temporary:
        root = Path(temporary).resolve()
        site, directory = root / "site", root / "complete-results"
        site.mkdir()
        for artifact in (wheel, charts_wheel):
            with zipfile.ZipFile(artifact) as archive:
                assert all(
                    (site / name).resolve().is_relative_to(site) for name in archive.namelist()
                )
                archive.extractall(site)
        sources = {}
        for name in MODULES:
            relative = Path(*name.split("."))
            relative = (
                relative / "__init__.py"
                if (site / relative).is_dir()
                else relative.with_suffix(".py")
            )
            assert (site / relative).read_bytes() == (ROOT / "src" / relative).read_bytes(), name
            sources[name] = digest(site / relative)
        code = (
            f"import sys, importlib\nfrom pathlib import Path\nsys.path.insert(0, {str(site)!r})\n"
            f"for name in {(*MODULES, 'openecon_charts')!r}:\n"
            f"    assert Path(importlib.import_module(name).__file__).is_relative_to(Path({str(site)!r}))\n"
            "displayed=[]\ndef display(frame): displayed.append(frame)\n"
            f"PROSPECTIVE_EXTENSION_RESULT_DIRECTORY={str(directory)!r}\n"
            + EXAMPLE.read_text()
            + "\nassert len(displayed)==16\n"
        )
        process = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
            timeout=180,
        )
        proof = proof_from(process.stdout)
        assert not proof["frozen"]
        hashes = verify_files(directory, proof)
    return dict(
        status="passed",
        layer="wheel",
        proof=proof,
        complete_result_hashes=hashes,
        wheel_sha256=digest(wheel),
        charts_wheel_sha256=digest(charts_wheel),
        source_modules_equal=sources,
        imports_from_isolated_extracted_wheels=True,
        dependencies="existing development interpreter; no estimation-library imports",
    )


def installed_check(mode, receipt_path):
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == IDENTIFIER and not APP.is_symlink()
    assert not DATA.is_symlink() and DATA.resolve() == DATA.absolute()
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    sources = source_identity(runtime)
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    assert type(port) is int and 1024 <= port <= 65535
    desktop_token = workspace_token = None

    def call(path, body=None, *, workspace=False, method=None):
        token = workspace_token if workspace else desktop_token
        request = Request(
            f"http://127.0.0.1:{port}" + path,
            headers={
                "Content-Type": "application/json",
                **({"X-OpenEcon-Token": token} if token else {}),
            },
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    desktop_token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if mode == "seed" and not projects:
        call("/api/desktop/local-projects", {"name": PROJECT})
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects) == 1 and projects[0]["name"] == PROJECT
    project = projects[0]["id"]
    if mode == "seed":
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    workspace_token = call(prefix + "/session")["token"]
    directory = DATA / "complete-results"
    code = header(directory) + EXAMPLE.read_text()
    if mode == "seed":
        script = call(
            prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code}, workspace=True
        )
        return dict(status="seeded", project_id=project, script_id=script["id"])
    before = json.loads(receipt_path.read_text())
    assert before["project_id"] == project
    assert call(prefix + "/console/scripts/" + before["script_id"], workspace=True)["code"] == code
    history = call(prefix + "/console", workspace=True)["history"]
    matching = [run for run in history if MARKER in run.get("stdout", "") and run["status"] == "ok"]
    if before.get("execution_id"):
        run = next(run for run in matching if run["id"] == before["execution_id"])
    else:
        assert len(matching) == 1
        run = matching[0]
    proof = verify_outputs(run)
    assert run["code"] == code
    hashes = verify_files(directory, proof)
    history_hash = canonical({key: run[key] for key in ("code", "stdout", "outputs", "events")})
    if mode == "restart":
        assert before["full_native_shutdown_verified"]
        assert not {row["pid"] for row in before["owned_processes"]} & {
            row["pid"] for row in owned_processes()
        }
        assert before["history_sha256"] == history_hash
        assert before["complete_result_hashes"] == hashes
        assert call("/api/desktop/status")["console"]["pid"] is None
        before["full_native_restart_readback_unchanged"] = True
    before.update(
        status="passed",
        layer="installed_native",
        proof=proof,
        execution_id=run["id"],
        duration_ms=run["duration_ms"],
        history_sha256=history_hash,
        complete_result_hashes=hashes,
        compiled_modules_equal_source=sources,
        owned_qa_app=True,
        runtime_sha256=digest(runtime),
        human_data_access=False,
        public_release_delivered=False,
        owned_processes=owned_processes(),
        port=port,
    )
    return before


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("source", "wheel", "frozen", "seed", "capture", "stopped", "restart"),
        required=True,
    )
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--charts-wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "source":
        record = source_check()
    elif args.mode == "wheel":
        if args.wheel is None or args.charts_wheel is None:
            parser.error("--wheel and --charts-wheel are required for wheel mode")
        record = wheel_check(args.wheel, args.charts_wheel)
    elif args.mode == "frozen":
        if args.runtime is None:
            parser.error("--runtime is required for frozen mode")
        record = frozen_check(args.runtime)
    elif args.mode == "stopped":
        record = stopped_check(args.output)
    else:
        record = installed_check(args.mode, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: record[key] for key in ("status", "layer", "duration_ms") if key in record}
        )
    )
