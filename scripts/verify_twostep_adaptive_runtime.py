"""Authenticate frozen TwoStep source and cold saved state; native UI is separate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import CodeType

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "examples/twostep_adaptive_acceptance.py"
MODULES = ("econometrics.twostep.__init__", "econometrics.twostep.kernel",
           "econometrics.twostep.adaptive", "econometrics.twostep.selection",
           "econometrics.twostep.public", "econometrics.twostep.helpers",
           "econometrics.summary_state", "econometrics.core", "econometrics.resident_cpu", "resources")
MARKER = "TWOSTEP_ADAPTIVE_ACCEPTANCE_OK "


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(code):
    return code.replace(co_filename="<bundled-source>", co_consts=tuple(
        normalized(item) if isinstance(item, CodeType) else item for item in code.co_consts))


def load_controller(path):
    spec = importlib.util.spec_from_file_location("_twostep_owned_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def committed(source_sha, relative):
    assert re.fullmatch(r"[0-9a-f]{40}", source_sha)
    value = subprocess.check_output(["git", "show", source_sha+":"+relative], cwd=ROOT)
    assert (ROOT/relative).read_bytes() == value, relative
    return value


def code(example, source_sha, replay):
    return "namespace={'__name__':'_twostep_adaptive_example'}\nexec("+repr(example)+",namespace)\nnamespace['twostep_adaptive_acceptance'](replay="+repr(replay)+", require_frozen=True, emit=display, source_sha="+repr(source_sha)+")\n"


def inspect(run, project_root, source_sha, replay):
    assert run["status"] == "ok", run.get("error")
    markers = [line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(markers) == 1
    marker = json.loads(markers[0])
    assert marker["source_sha"] == source_sha and marker["frozen"] is True
    assert marker["replay"] is replay and marker["fit_disabled_replay"] is replay
    assert len(marker["cases"]) == 4 and len(marker["files"]) == 9
    assert len(run["outputs"]) == 8 and all(item["type"] == "table" and item["latex"] for item in run["outputs"])
    for name, item in marker["files"].items():
        path = project_root/"twostep-adaptive-results"/name
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]
    return {key: marker[key] for key in ("cases", "files")}


def verify(runtime, version, source_sha):
    from PyInstaller.archive.readers import CArchiveReader
    if os.name == "nt" and (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("Windows verification requires disposable GitHub-hosted Windows.")
    runtime = runtime.resolve(strict=True)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in MODULES:
        relative = "src/openecon/"+name.replace(".", "/")+".py"
        source = committed(source_sha, relative)
        module = ("openecon."+name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(compile(source, relative, "exec", dont_inherit=True)), module
        modules[module] = hashlib.sha256(source).hexdigest()
    example = committed(source_sha, EXAMPLE).decode()
    controller = load_controller(ROOT/"scripts/verify_regularized_all_family_runtime.py")
    with tempfile.TemporaryDirectory(prefix="openecon-twostep-adaptive-") as temporary:
        root = Path(temporary)
        owned = controller.owned_runtime(runtime, root, "unused", version)
        try:
            owned.project = owned.request("/api/desktop/local-projects", {"name": "Adaptive TwoStep QA"}, desktop=True)["id"]
            owned.open_project()
            project = owned.project
            project_root = root/"projects"/project
            creation = code(example, source_sha, False)
            owned.call("/console/script", {"code": creation, "name": "analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code": creation, "timeout_seconds": 180})
            proof = inspect(run, project_root, source_sha, False)
            script = owned.call("/console/script")
            history = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
        finally:
            owned.close()
        owned = controller.owned_runtime(runtime, root, project, version)
        try:
            owned.open_project()
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
            reopened = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
            assert all(reopened[key] == history[key] for key in ("code", "stdout", "outputs", "events"))
            assert owned.call("/console/script") == script
            replay = owned.call("/console/execute", {"code": code(example, source_sha, True), "timeout_seconds": 180})
            assert inspect(replay, project_root, source_sha, True) == proof
        finally:
            owned.close()
    return {"status": "passed", "source_sha": source_sha, "runtime_sha256": digest(runtime),
            "example_sha256": hashlib.sha256(example.encode()).hexdigest(), "compiled_modules_equal_source": modules,
            "four_complete_saved_states": proof, "cold_history_without_worker": True,
            "fit_disabled_saved_replay": True, "temporary_profile_removed": not root.exists(),
            "owned_runtime_stopped": True, "source_path_injected": False,
            "native_window_verified": False, "licensed_vendor_execution": False, "public_release_delivered": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-version", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    result = verify(args.runtime, args.sdk_version, args.source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "source_sha": args.source_sha, "native_window_verified": False}))
