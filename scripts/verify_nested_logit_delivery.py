"""Owned nested-choice source/frozen/native full-output and cold-relaunch proof.

Only seed creates synthetic QA data. Native Run/Quit/reopen are GUI actions;
the other native modes read saved data and passive status. No human projects
or other QA profiles are accessed. Receipts remain outside the owned profile.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import enum
import hashlib
import importlib
import io
import json
from pathlib import Path
import plistlib
import re
import runpy
import subprocess
import tempfile
from urllib.request import ProxyHandler, Request, build_opener

from verify_bai_perron_runtime import OwnedRuntime, normalized
import verify_native_relaunch as processes
from verify_rank_ordered_runtime import saved_frames, wire_scalar

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/nested_logit.py"
METHODS = ("oim", "hc0", "cr0", "probabilities", "components", "effects", "elasticities")
MARKER = "NESTED_LOGIT_ACCEPTANCE_OK "
IDENTIFIER = "org.openecon.qa.nestedchoice"
APP = Path.home() / "Applications/OpenEconometrics Nested Choice QA.app"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
NAME = "Nested Choice Eight QA"
MODULES = ("openecon.econometrics.discrete", "openecon.econometrics.discrete.nested_logit",
           "openecon.econometrics.discrete.nested_logit_postestimation",
           "openecon.econometrics.discrete.rank_ordered",
           "openecon.econometrics.discrete.rank_ordered_postestimation",
           "openecon.econometrics.registry", "openecon.analysis_contracts",
           "openecon.engines.optimize",
           "openecon.resources", "openecon.econometrics.core",
           "openecon.econometrics.nonparametric.common")


def digest(path):
    no_links(path)
    assert path.is_file()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_links(path):
    assert not any(p.is_symlink() for p in (path, *path.parents)), "Linked QA path"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    digest(path)
    return json.loads(path.read_text())


def checked_id(value):
    assert isinstance(value, str) and re.fullmatch(
        r"[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value
    ), "Invalid owned identifier"
    return value


def source_pin():
    head = identity()
    paths = {"example": EXAMPLE, "delivery_verifier": Path(__file__).resolve()}
    for name in ("openecon", *MODULES):
        path = ROOT / "src" / Path(*name.split("."))
        paths[name] = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    hashes = {}
    for name, path in paths.items():
        assert path.read_bytes() == subprocess.check_output(
            ["git", "show", head + ":" + str(path.relative_to(ROOT))], cwd=ROOT
        ), "Commit the scientific source before delivery verification"
        hashes[name] = digest(path)
    return dict(source_head=head, committed_file_hashes=hashes)


def identity():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def frozen_sources(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    hashes = {}
    enum_source = runtime.parent / "_internal" / "enum.py"
    assert enum_source.read_bytes() == Path(enum.__file__).read_bytes(), "Packaged Torch JIT stdlib source"
    hashes["stdlib.enum_source"] = digest(enum_source)
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        assert normalized(archive.extract(name)) == normalized(compile(
            path.read_text(), str(path), "exec", dont_inherit=True)), name
        hashes[name] = digest(path)
    return hashes


def header(directory):
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
            f"NESTED_LOGIT_RESULT_DIRECTORY = {str(directory)!r}\n")


def files(directory, proof):
    from openecon.econometrics.core import TableSet
    assert set(proof["methods"]) == set(METHODS)
    assert {p.name for p in directory.iterdir()} == {name + ".json" for name in METHODS}
    hashes, total = {}, 0
    for name in METHODS:
        path = directory / (name + ".json")
        hashes[name] = digest(path)
        meta = proof["methods"][name]
        assert hashes[name] == meta["sha256"]
        value = json.loads(path.read_text())
        assert set(value) == {"attrs", "tables", "latex"}
        assert len(meta["table_order"]) == len(set(meta["table_order"]))
        assert set(value["tables"]) == set(meta["table_order"]) == set(meta["table_rows"])
        frames = saved_frames(value)
        assert value["latex"] == TableSet({key: frames[key] for key in meta["table_order"]}).to_latex()
        assert {key: len(frame) for key, frame in frames.items()} == meta["table_rows"]
        assert value["attrs"]["settings"]["device"] == "cpu"
        json.dumps(value, allow_nan=False)
        total += len(frames)
    assert proof["saved_tables"] == total == 65
    return hashes


def outputs(run, directory):
    assert run["status"] == "ok", run.get("error")
    assert "Display limit reached" not in run["stdout"]
    markers = [line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(markers) == 1
    proof = json.loads(markers[0])
    assert proof["frozen"] and proof["all_eight_scopes_verified"] and proof["full_state_replay_verified"]
    assert proof["third_party_estimation_imports"] == []
    assert proof["displayed_tables"] == 7
    assert [item["type"] for item in run["outputs"]] == ["table"] * 7
    for name, output in zip(METHODS, run["outputs"]):
        value = json.loads((directory / (name + ".json")).read_text())
        keys = proof["methods"][name]["displayed_keys"]
        assert len(keys) == 1
        key = keys[0]
        frame, actual = value["tables"][key], output["data"]
        assert actual["columns"] == frame["columns"]
        assert actual["total_columns"] == len(frame["columns"])
        assert actual["total_rows"] == len(frame["data"]) == len(actual["rows"])
        assert actual["rows"] == [[wire_scalar(cell) for cell in row] for row in frame["data"]]
        assert actual["index"] == [[wire_scalar(v)] for v in frame["index"]]
        assert actual["index_names"] == [None]
        latex = str(saved_frames(value)[key].to_latex())
        assert output["latex"] == latex
        assert hashlib.sha256(latex.encode()).hexdigest() == proof["methods"][name]["displayed_latex_sha256"][key]
    return proof


def source(directory, baseline=None):
    before = source_pin()
    origins = {}
    for name in ("openecon", *MODULES):
        actual = Path(importlib.import_module(name).__file__).resolve()
        expected = ROOT / "src" / Path(*name.split("."))
        expected = expected / "__init__.py" if expected.is_dir() else expected.with_suffix(".py")
        assert actual == expected.resolve(), (name, actual)
        origins[name] = str(actual)
    stream = io.StringIO()
    with redirect_stdout(stream):
        runpy.run_path(str(EXAMPLE), init_globals={"NESTED_LOGIT_RESULT_DIRECTORY": str(directory)})
    markers = [line[len(MARKER):] for line in stream.getvalue().splitlines() if line.startswith(MARKER)]
    assert len(markers) == 1
    proof = json.loads(markers[0])
    assert proof["all_eight_scopes_verified"] and proof["full_state_replay_verified"]
    hashes = files(directory, proof)
    if baseline is not None:
        for name in METHODS:
            assert (directory / (name + ".json")).read_bytes() == (baseline / (name + ".json")).read_bytes(), name
    assert source_pin() == before, "Committed source changed during execution"
    return dict(status="passed", source_head=before["source_head"], source_pin_before=before,
                source_pin_after=source_pin(), executed_module_origins=origins,
                source_hashes={name: digest(Path(path)) for name, path in origins.items()},
                complete_result_hashes=hashes, proof=proof, all_bytes_equal_to_baseline=baseline is not None)


def frozen(runtime, baseline):
    before = source_pin()
    sources = frozen_sources(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-nested-choice-") as temporary:
        root = Path(temporary).resolve(strict=True)
        directory = root / "complete"
        code = header(directory) + EXAMPLE.read_text()
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            owned.open_project(create=True)
            project = owned.project
            script = owned.call("/console/scripts", {"code": code, "name": EXAMPLE.name})
            script_id = checked_id(script["id"])
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120,
                                                   "script_id": script_id})
            proof = outputs(run, directory)
            hashes = files(directory, proof)
            for name in METHODS:
                assert (directory / (name + ".json")).read_bytes() == (baseline / (name + ".json")).read_bytes(), name
            original = json_hash(run)
            document = owned.call("/console/scripts/" + script_id)
            owned.call("/console/reset", {})
            history = owned.call("/console")["history"]
            assert len(history) == 1 and history[0]["id"] == run["id"]
            assert original == json_hash(history[0])
        finally:
            owned.close()
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project()
            history = owned.call("/console")["history"]
            assert len(history) == 1 and history[0]["id"] == run["id"]
            assert original == json_hash(history[0])
            assert document == owned.call("/console/scripts/" + script_id)
            assert files(directory, proof) == hashes
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
        finally:
            owned.close()
    assert source_pin() == before and frozen_sources(runtime) == sources
    return dict(status="passed", source_head=before["source_head"], source_pin_before=before,
                source_pin_after=source_pin(), compiled_modules_equal_source=sources,
                runtime_sha256=digest(runtime), complete_result_hashes=hashes,
                proof=proof, duration_ms=run["duration_ms"], full_worker_reset_and_server_restart=True,
                all_bytes_equal_to_source=True, human_data_access=False, native_gui_observed=False)


def native_session():
    with (APP / "Contents/Info.plist").open("rb") as stream:
        assert plistlib.load(stream)["CFBundleIdentifier"] == IDENTIFIER
    original_app = processes.APP
    processes.APP = APP
    try:
        owned = processes.owned_processes(processes.process_table(), running=True)
    finally:
        processes.APP = original_app
    port = read_json(DATA / ".runtime-port.json")["port"]
    assert type(port) is int and 1024 <= port <= 65535
    listeners = subprocess.check_output(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], text=True)
    assert {int(pid) for pid in listeners.split()} <= {row["pid"] for row in owned}
    opener = build_opener(ProxyHandler({}))
    def request(path, body=None, *, token=None, method=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        req = Request(f"http://127.0.0.1:{port}" + path, headers=headers,
                      data=json.dumps(body).encode() if body is not None else None, method=method)
        with opener.open(req, timeout=30) as response:
            return json.load(response)
    token = request("/api/desktop/session")["token"]
    return port, request, token


def installed(mode, receipt, baseline):
    no_links(receipt)
    before = source_pin()
    _, request, token = native_session()
    projects = request("/api/desktop/local-projects", token=token)["projects"]
    if mode == "seed":
        assert not projects and not receipt.exists(), "Seeding requires a fresh owned profile"
        request("/api/desktop/local-projects", {"name": NAME, "description": "Owned MARKET-601..608 synthetic acceptance."}, token=token)
        projects = request("/api/desktop/local-projects", token=token)["projects"]
    assert len(projects) == 1 and projects[0]["name"] == NAME
    project = checked_id(projects[0]["id"])
    catalogue = read_json(DATA / "local-projects.json")
    assert len(catalogue["projects"]) == 1
    assert catalogue["projects"][0]["id"] == project and catalogue["projects"][0]["name"] == NAME
    if mode == "seed":
        request(f"/api/desktop/local-projects/{project}/open", {}, token=token)
    prefix = f"/api/desktop/projects/{project}/workspace"
    workspace = request(prefix + "/session")["token"]
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    sources = frozen_sources(runtime)
    directory = DATA / "complete-nested-choice"
    code = header(directory) + EXAMPLE.read_text()
    if mode == "seed":
        assert request(prefix + "/console", token=workspace)["history"] == []
        script = request(prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code}, token=workspace)
        assert source_pin() == before
        return dict(status="seeded", project_id=project, script_id=checked_id(script["id"]))
    record = read_json(receipt)
    checked_id(record["project_id"])
    checked_id(record["script_id"])
    assert record["project_id"] == project
    assert request(prefix + "/console/scripts/" + record["script_id"], token=workspace)["code"] == code
    history = request(prefix + "/console", token=workspace)["history"]
    matching = [row for row in history if MARKER in row.get("stdout", "")]
    assert len(history) == 1
    assert len(matching) == 1
    run = matching[0]
    checked_id(run["id"])
    assert run["code"] == code
    proof = outputs(run, directory)
    hashes = files(directory, proof)
    for name in METHODS:
        assert (directory / (name + ".json")).read_bytes() == (baseline / (name + ".json")).read_bytes(), name
    if record.get("execution_sha256"):
        assert record["execution_sha256"] == json_hash(run)
        assert record["complete_result_hashes"] == hashes
    record.update(status="passed", execution_id=run["id"], execution_sha256=json_hash(run),
                  proof=proof, complete_result_hashes=hashes, compiled_modules_equal_source=sources,
                  duration_ms=run["duration_ms"], all_bytes_equal_to_source=True, human_data_access=False,
                  native_gui_observed_by_helper=False, separate_gui_evidence_required=True,
                  runtime_sha256=digest(runtime), native_sha256=digest(APP / "Contents/MacOS/openecon-desktop"),
                  bundle_info_sha256=digest(APP / "Contents/Info.plist"),
                  source_pin_before=before, source_pin_after=source_pin())
    assert source_pin() == before and frozen_sources(runtime) == sources
    return record


def relaunch(phase, receipt, installed_receipt):
    no_links(receipt)
    previous = read_json(receipt) if receipt.exists() else {"phases": []}
    phases = ("baseline", "stopped", "after")
    assert [row["phase"] for row in previous["phases"]] == list(phases[:phases.index(phase)])
    accepted = read_json(installed_receipt)
    assert accepted["status"] == "passed"
    project, script, execution = (checked_id(accepted[key]) for key in ("project_id", "script_id", "execution_id"))
    console = DATA / "projects" / project / "console"
    catalogue = read_json(DATA / "local-projects.json")
    assert len(catalogue["projects"]) == 1
    assert catalogue["projects"][0]["id"] == project and catalogue["projects"][0]["name"] == NAME
    entries = list(console.iterdir())
    assert all(p.is_file() and not p.is_symlink() and p.suffix == ".json" for p in entries)
    assert {"history.json", "scripts.json", "script.json", script + ".json"} <= {p.name for p in entries}
    history = read_json(console / "history.json")["history"]
    assert len(history) == 1 and history[0]["id"] == execution
    assert json_hash(history[0]) == accepted["execution_sha256"]
    document = read_json(console / (script + ".json"))
    assert document["id"] == script and document["code"] == history[0]["code"]
    assert history[0]["status"] == "ok"
    state = dict(accepted_receipt_sha256=digest(installed_receipt),
                 console_files={p.name: digest(p) for p in console.iterdir()},
                 results={name: digest(DATA / "complete-nested-choice" / (name + ".json")) for name in METHODS},
                 catalogue=digest(DATA / "local-projects.json"),
                 runtime=digest(APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"),
                 native=digest(APP / "Contents/MacOS/openecon-desktop"))
    assert state["results"] == accepted["complete_result_hashes"]
    assert state["runtime"] == accepted["runtime_sha256"]
    assert state["native"] == accepted["native_sha256"]
    assert digest(APP / "Contents/Info.plist") == accepted["bundle_info_sha256"]
    with (APP / "Contents/Info.plist").open("rb") as stream:
        assert plistlib.load(stream)["CFBundleIdentifier"] == IDENTIFIER
    if phase == "baseline":
        previous["saved_snapshot"] = state
    else:
        assert previous["saved_snapshot"] == state
    row = dict(phase=phase, complete_saved_state_unchanged=True, native_gui_observed_by_helper=False)
    original_app = processes.APP
    processes.APP = APP
    try:
        if phase == "stopped":
            row.update(processes.stopped_state(previous["phases"][-1]))
        else:
            port, request, token = native_session()
            owned = processes.owned_processes(processes.process_table(), running=True)
            status = request("/api/desktop/status", token=token)
            assert status["project_id"] == accepted["project_id"]
            assert status["console"]["running"] is False and status["console"]["execution_id"] is None
            if phase == "after":
                assert status["console"]["pid"] is None
                old = {(p["pid"], p["started"]) for p in previous["phases"][0]["owned_processes"]}
                assert all((p["pid"], p["started"]) not in old for p in owned)
            row.update(port=port, owned_processes=owned, no_computation_worker=status["console"]["pid"] is None)
    finally:
        processes.APP = original_app
    previous["phases"].append(row)
    previous["status"] = "passed" if phase == "after" else "in_progress"
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("source", "frozen", "seed", "readback", "baseline", "stopped", "after"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--installed-receipt", type=Path)
    args = parser.parse_args()
    no_links(args.output)
    assert not args.output.resolve().is_relative_to(DATA.resolve())
    assert not args.output.resolve().is_relative_to(APP.resolve())
    if args.mode == "source":
        result = source(args.directory, args.baseline)
    elif args.mode == "frozen":
        result = frozen(args.runtime.resolve(), args.baseline)
    elif args.mode in ("seed", "readback"):
        result = installed(args.mode, args.output, args.baseline)
    else:
        result = relaunch(args.mode, args.output, args.installed_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "duration_ms", "source_head") if key in result}))


if __name__ == "__main__":
    main()
