"""Seed/read back one isolated native QA project; execution uses the real Run control."""

from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import plistlib
import subprocess
import tarfile
from types import CodeType
from urllib.request import Request, urlopen

import verify_multiple_testing_installed as native

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "OpenEconometrics Weighted Binary QA"
IDENTIFIER = "org.openecon.qa.weighted-binary-eight-20261010"
APP = Path.home() / "Applications" / (PRODUCT + ".app")
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Eight Direct Weighted Binary Domains"
MARKER = "WEIGHTED_BINARY_NATIVE_OK "
EXAMPLE = "docs/examples/weighted_binary_eight.py"


def digest(path):
    assert path.is_file() and not path.is_symlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(ref, path):
    return subprocess.check_output(["git", "show", ref + ":" + path], cwd=ROOT, text=True)


def source_parity(runtime, ref):
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    source = subprocess.check_output(["git", "archive", ref, "src/openecon"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(source)) as files:
        contents = {
            m.name: files.extractfile(m).read()
            for m in files
            if m.isfile() and m.name.endswith(".py")
        }

    def normalized(code):
        return code.replace(
            co_filename="<pinned>",
            co_consts=tuple(
                normalized(v) if isinstance(v, CodeType) else v for v in code.co_consts
            ),
        )

    result = {}
    for name in archive.toc:
        if name != "openecon" and not name.startswith("openecon."):
            continue
        path = "src/" + name.replace(".", "/")
        path = path + ".py" if path + ".py" in contents else path + "/__init__.py"
        assert path in contents, name
        assert normalized(archive.extract(name)) == normalized(
            compile(contents[path], path, "exec", dont_inherit=True)
        ), name
        result[name] = hashlib.sha256(contents[path]).hexdigest()
    assert "openecon.econometrics.weighted_binary" in result
    return result


def code_for(workspace, ref, modules):
    header = f"""import importlib, importlib.util, os, sys
from pathlib import Path
assert getattr(sys, "frozen", False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
assert Path.cwd().resolve() == Path({str(workspace)!r}).resolve()
_root = Path(sys._MEIPASS).resolve()
_modules = {{}}
for _name in {modules!r}:
    _file = Path(importlib.import_module(_name).__file__).resolve()
    assert _file.is_relative_to(_root), _name
    _modules[_name] = str(_file)
"""
    footer = f"""
_files = {{}}
for _name, _value in [("results",states),("queries",queries),("latex",latex),("inputs",oracle_inputs)]:
    _path = Path.cwd()/("weighted-binary-"+_name+".json")
    assert not _path.is_symlink()
    _bytes = json.dumps(_value,allow_nan=False,sort_keys=True,separators=(",",":")).encode()
    assert len(_bytes) < 2*1024*1024
    _path.write_bytes(_bytes)
    assert json.loads(_path.read_text()) == _value
    _files[_path.name] = __import__("hashlib").sha256(_bytes).hexdigest()
print({MARKER!r}+json.dumps({{"frozen":True,"root":str(_root),"worker_pid":os.getpid(),
    "source_ref":{ref!r},"modules":_modules,"files":_files,"cases":8}},sort_keys=True))
"""
    return header + pinned(ref, EXAMPLE) + footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--after-restart", action="store_true")
    args = parser.parse_args()
    ref = subprocess.check_output(
        ["git", "rev-parse", args.source_ref + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    assert pinned(ref, "scripts/verify_weighted_binary_installed.py") == Path(__file__).read_text()
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == IDENTIFIER
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    identity = {
        "app": str(APP),
        "source_ref": ref,
        "runtime_sha256": digest(runtime),
        "executable_sha256": digest(APP / "Contents/MacOS" / info["CFBundleExecutable"]),
        "manifest_sha256": digest(APP / "Contents/Resources/runtime/runtime-manifest.json"),
        "compiled_source": source_parity(runtime, ref),
        "example_sha256": hashlib.sha256(pinned(ref, EXAMPLE).encode()).hexdigest(),
    }
    pids = native.runtime_pids(str(runtime))
    app_pids = native.runtime_pids(str(APP / "Contents/MacOS" / info["CFBundleExecutable"]))
    native.require_unaliased_directory(DATA)
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    token = None

    def call(path, body=None):
        native.verify_listener(pids, port)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        request = Request(
            f"http://127.0.0.1:{port}" + path,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call(
            "/api/desktop/local-projects",
            {"name": PROJECT, "description": "Synthetic local-only weighted binary acceptance."},
        )
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects) == 1 and projects[0]["name"] == PROJECT
    project = projects[0]["id"]
    workspace = DATA / "projects" / project
    native.require_unaliased_directory(workspace)
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    # Import the numerical/reporting dependencies used by this example. Full
    # archive parity is separately checked above, without importing every module.
    modules = [
        "openecon",
        "openecon.analysis",
        "openecon.models",
        "openecon.analysis_contracts",
        "openecon.econometrics.weighted_binary",
        "openecon.econometrics.glm.glm",
        "openecon.econometrics.glm.commands",
        "openecon.econometrics.glm.kernels",
        "openecon.econometrics.core",
        "openecon.econometrics.registry",
        "openecon.econometrics.postest.prediction",
        "openecon.econometrics.postest.inference",
    ]
    code = code_for(workspace, ref, modules)
    if args.seed:
        assert not args.receipt.exists()
        script = call(
            prefix + "/console/scripts", {"name": "weighted_binary_eight.py", "code": code}
        )
        assert call(prefix + "/console/scripts/" + script["id"])["code"] == code
        record = {
            "status": "seeded",
            "identity": identity,
            "project_id": project,
            "script_id": script["id"],
            "runtime_pids": pids,
            "app_pids": app_pids,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        }
    else:
        record = json.loads(args.receipt.read_text())
        assert record["identity"] == identity
        assert call(prefix + "/console/scripts/" + record["script_id"])["code"] == code
        history = call(prefix + "/console")["history"]
        assert len(history) == 1 and history[0]["status"] == "ok", [
            (x["status"], x.get("error")) for x in history
        ]
        run = history[0]
        assert run["code"] == code and len(run["outputs"]) == 8
        assert all(x["type"] == "model" and x.get("latex") for x in run["outputs"])
        proof = json.loads(
            next(x[len(MARKER) :] for x in run["stdout"].splitlines() if x.startswith(MARKER))
        )
        assert proof["frozen"] and proof["cases"] == 8 and proof["source_ref"] == ref
        assert (
            set(proof["modules"]) == set(modules)
            and Path(proof["root"]) == runtime.parent / "_internal"
        )
        files = {p.name: digest(p) for p in workspace.glob("weighted-binary-*.json")}
        assert files == proof["files"] and len(files) == 4
        results = json.loads((workspace / "weighted-binary-results.json").read_text())
        assert len(results) == 8
        for output, state in zip(run["outputs"], results.values(), strict=True):
            # Serialized results use sorted keys; output order is the explicit
            # example loop. Match by estimator and weight domain instead.
            fitted = next(s for s in results.values() if s["id"] == output["data"]["id"])
            expected = {
                k: v
                for k, v in fitted.items()
                if k not in {"sample_positions", "covariance_matrix"}
            }
            expected["display_omitted"] = ["sample_positions", "covariance_matrix"]
            assert output["data"] == expected
        history_hash = digest(workspace / "console/history.json")
        execution_hash = hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if args.after_restart:
            native.require_pids_exited(record["runtime_pids"] + record["app_pids"])
            assert record["files"] == files and record["history_sha256"] == history_hash
            assert record["execution_sha256"] == execution_hash
        else:
            assert proof["worker_pid"] in pids
            record["runtime_pids"], record["app_pids"] = pids, app_pids
        record.update(
            status="restarted" if args.after_restart else "native-verified",
            proof=proof,
            files=files,
            history_sha256=history_hash,
            execution_sha256=execution_hash,
            execution_record=run,
            workspace=str(workspace),
            current_runtime_pids=pids,
            current_app_pids=app_pids,
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "project": project, "source": ref}))


if __name__ == "__main__":
    main()
