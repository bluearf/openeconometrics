"""Seed/read back the isolated Issues QA app; execute only with its native Run.

The fixed QA identifier, synthetic project and local-loopback APIs keep this
workflow separate from the primary app and all cloud identities. This script
never submits a console execution request. Use --after-restart only after
quitting and relaunching the same QA bundle through its native UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import subprocess
from types import CodeType
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "org.openecon.qa.nextissues-20261007"
PRODUCT = "OpenEconometrics Issues QA"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
APP = Path.home() / "Applications" / f"{PRODUCT}.app"
EXAMPLE = ROOT / "docs/examples/multiple_testing.py"
PROJECT = "Multiple Testing QA"
STATE_NAME = "qa-multiple-testing-states.json"
MARKER = "MULTIPLE_TESTING_INSTALLED:"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_identity(app: Path) -> dict:
    app = app.resolve(strict=True)
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    if info["CFBundleIdentifier"] != IDENTIFIER or app.name != f"{PRODUCT}.app":
        raise RuntimeError("Only the separately built, dedicated Issues QA bundle is allowed.")
    executable = app / "Contents/MacOS" / info["CFBundleExecutable"]
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    manifest = app / "Contents/Resources/runtime/runtime-manifest.json"
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    signature = subprocess.run(
        ["codesign", "-dv", str(app)], check=True, text=True, capture_output=True
    ).stderr
    if "Signature=adhoc" not in signature:
        raise RuntimeError("This acceptance script requires the separately ad-hoc signed QA app.")
    return {
        "app": str(app),
        "identifier": IDENTIFIER,
        "version": info["CFBundleShortVersionString"],
        "minimum_system_version": info["LSMinimumSystemVersion"],
        "signature_kind": "ad-hoc local QA; no Developer ID or notarization claim",
        "native_executable_sha256": digest(executable),
        "frozen_runtime_sha256": digest(runtime),
        "runtime_manifest_sha256": digest(manifest),
        "runtime_executable": str(runtime),
        "installed_pyz_source_hashes": verify_installed_pyz(runtime),
    }


def verify_installed_pyz(runtime: Path) -> dict[str, str]:
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalized(code: CodeType) -> CodeType:
        return code.replace(
            co_filename="<scoped-bundle>",
            co_consts=tuple(
                normalized(value) if isinstance(value, CodeType) else value
                for value in code.co_consts
            ),
        )

    hashes = {}
    for module in [
        "openecon.econometrics.postest.multiple",
        "openecon.econometrics.postest",
        "openecon.econometrics.registry",
        "openecon.econometrics.core",
        "openecon.analysis_contracts",
        "openecon.console_worker",
    ]:
        source = ROOT / "src" / Path(*module.split("."))
        source = source / "__init__.py" if source.is_dir() else source.with_suffix(".py")
        if normalized(archive.extract(module)) != normalized(
            compile(source.read_text(), str(source), "exec", dont_inherit=True)
        ):
            raise RuntimeError(f"The final installed PYZ differs from the pinned source: {module}")
        hashes[module] = digest(source)
    return hashes


def require_unaliased_directory(path: Path, *, may_be_missing: bool = False) -> None:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError(
            "The dedicated QA data/project directory must not contain symlink aliases."
        )
    if not path.is_dir() and not (may_be_missing and not path.exists()):
        raise RuntimeError("The dedicated QA data/project directory is invalid.")


def verify_listener(pids: list[int], port: int) -> list[int]:
    result = subprocess.run(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            "-p",
            ",".join(str(pid) for pid in pids),
            f"-iTCP:{port}",
            "-sTCP:LISTEN",
            "-Fpn",
        ],
        text=True,
        capture_output=True,
    )
    listeners = []
    current = None
    for line in result.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            current = int(line[1:])
        elif line == f"n127.0.0.1:{port}" and current in pids:
            listeners.append(current)
    if result.returncode or not listeners:
        raise RuntimeError("The loopback listener does not belong to the installed QA runtime.")
    return sorted(set(listeners))


def runtime_pids(runtime: str) -> list[int]:
    # comm contains the executable path only, never command-line credentials.
    lines = subprocess.check_output(["ps", "-ww", "-axo", "pid=,comm="], text=True).splitlines()
    result = []
    for line in lines:
        pieces = line.strip().split(maxsplit=1)
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1] == runtime:
            result.append(int(pieces[0]))
    if not result:
        raise RuntimeError("The dedicated installed QA runtime is not running.")
    return sorted(result)


def require_pids_exited(pids: list[int]) -> None:
    if not pids or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise RuntimeError("The prior QA runtime PID inventory is invalid.")
    result = subprocess.run(
        ["ps", "-p", ",".join(str(pid) for pid in pids), "-o", "pid="],
        text=True,
        capture_output=True,
    )
    if result.stdout.strip() or result.returncode not in (0, 1):
        raise RuntimeError("A previous QA runtime PID is still present after relaunch.")


def verification_code(project: str) -> str:
    workspace = DATA / "projects" / project
    header = f"""import importlib, importlib.util, os, sys
from pathlib import Path
assert getattr(sys, "frozen", False), "The console is not the frozen runtime"
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
_qa_frozen_root = Path(sys._MEIPASS).resolve()
_qa_workspace = Path({str(workspace)!r}).resolve()
assert Path.cwd().resolve() == _qa_workspace, "Not the dedicated synthetic QA project"
_qa_modules = {{}}
for _qa_name in ["openecon", "openecon.econometrics.postest.multiple", "torch", "numpy", "pandas"]:
    _qa_module = importlib.import_module(_qa_name)
    _qa_path = Path(_qa_module.__file__).resolve()
    assert _qa_path.is_relative_to(_qa_frozen_root), _qa_name
    _qa_modules[_qa_name] = {{"file": str(_qa_path), "version": getattr(_qa_module, "__version__", None)}}
"""
    footer = f"""
_qa_envelope = {{"schema": 1, "procedures": ["multipletests", "stepdown", "simultaneous_ci"], "states": states}}
_qa_bytes = json.dumps(_qa_envelope, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
assert len(_qa_bytes) < 256 * 1024
_qa_state = _qa_workspace / {STATE_NAME!r}
_qa_temp = _qa_workspace / ({STATE_NAME!r} + ".tmp")
_qa_temp.write_bytes(_qa_bytes)
_qa_temp.replace(_qa_state)
assert json.loads(_qa_state.read_text()) == _qa_envelope
print({MARKER!r} + json.dumps({{
    "frozen": True, "frozen_root": str(_qa_frozen_root),
    "worker_pid": os.getpid(), "modules": _qa_modules,
    "workspace": str(_qa_workspace), "state_file": _qa_state.name,
    "states_bytes": len(_qa_bytes), "states_sha256": __import__("hashlib").sha256(_qa_bytes).hexdigest(),
    "tables": len(states), "full_covariance": intervals.attrs["joint_covariance"],
    "stata_parity_validated": False, "scipy_available": False, "statsmodels_available": False,
}}, allow_nan=False, sort_keys=True))
"""
    return header + EXAMPLE.read_text() + footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--seed", action="store_true")
    actions.add_argument("--after-restart", action="store_true")
    parser.add_argument(
        "--accept-new-run",
        action="store_true",
        help="Pin a new observed native Run after an explicit rebuild",
    )
    parser.add_argument("--app", type=Path, default=APP)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    identity = native_identity(args.app)
    pids = runtime_pids(identity["runtime_executable"])
    require_unaliased_directory(DATA)
    require_unaliased_directory(DATA / "projects", may_be_missing=True)
    preference = DATA / ".runtime-port.json"
    if preference.is_symlink() or preference.stat().st_size > 256:
        raise RuntimeError("The dedicated runtime port file is unsafe.")
    port = json.loads(preference.read_text())["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise RuntimeError("The dedicated runtime port is invalid.")
    origin = f"http://127.0.0.1:{port}"
    listeners = verify_listener(pids, port)
    token = None

    def call(path: str, body=None, method=None):
        verify_listener(pids, port)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        request = Request(
            origin + path,
            headers=headers,
            data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
            method=method,
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call(
            "/api/desktop/local-projects",
            {
                "name": PROJECT,
                "description": "Synthetic local-only MARKET-182 installed acceptance; no cloud or personal data.",
            },
        )
        projects = call("/api/desktop/local-projects")["projects"]
    if (
        len(projects) != 1
        or projects[0]["name"] != PROJECT
        or not re.fullmatch(r"[0-9a-f]{32}", projects[0]["id"])
    ):
        raise RuntimeError(
            "Only the single designated synthetic Multiple Testing QA project is allowed."
        )
    project = projects[0]["id"]
    require_unaliased_directory(DATA / "projects" / project)
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    code = verification_code(project)
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    if args.seed:
        if args.receipt.exists():
            previous = json.loads(args.receipt.read_text())
            if previous.get("project_id") != project:
                raise RuntimeError("The seed receipt refers to a different project.")
            script = call(prefix + "/console/scripts/" + previous["script_id"])
            if script["code"] != code:
                script = call(
                    prefix + "/console/scripts/" + script["id"],
                    {
                        "name": "multiple_testing_installed.py",
                        "code": code,
                        "version": script["version"],
                    },
                    "PUT",
                )
        else:
            script = call(
                prefix + "/console/scripts", {"name": "multiple_testing_installed.py", "code": code}
            )
        saved = call(prefix + "/console/scripts/" + script["id"])
        if saved["code"] != code:
            raise RuntimeError("The saved QA source does not match the exact augmented example.")
        record = {
            "status": "seeded",
            "project_id": project,
            "script_id": script["id"],
            "source_sha256": code_hash,
            "example_sha256": digest(EXAMPLE),
            "app_identity": identity,
            "initial_runtime_pids": pids,
            "initial_listener_pids": listeners,
            "proof_layers": {
                "source_saved_exact": True,
                "native_run_readback": False,
                "states_file_readback": False,
                "restart_readback_unchanged": False,
            },
        }
    else:
        record = json.loads(args.receipt.read_text())
        if record["project_id"] != project or record["source_sha256"] != code_hash:
            raise RuntimeError("The project or expected source changed after seeding.")
        if record["app_identity"] != identity:
            raise RuntimeError(
                "The dedicated bundle changed; seed and observe a new native Run first."
            )
        script = call(prefix + "/console/scripts/" + record["script_id"])
        if script["code"] != code:
            raise RuntimeError("The saved native source changed.")
        history = call(prefix + "/console")["history"]
        completed = [
            item
            for item in history
            if item.get("status") == "ok"
            and item.get("code") == code
            and any(line.startswith(MARKER) for line in item.get("stdout", "").splitlines())
        ]
        if not completed:
            raise RuntimeError(
                "No completed native Run with the exact frozen/source markers exists."
            )
        if args.accept_new_run:
            execution = max(completed, key=lambda item: item["created_at"])
            record.pop("execution_sha256", None)
        elif record.get("execution_id"):
            execution = next(item for item in completed if item["id"] == record["execution_id"])
        elif len(completed) == 1:
            execution = completed[0]
        else:
            raise RuntimeError(
                "Several completed QA runs exist; explicitly choose --accept-new-run."
            )
        proof = json.loads(
            next(
                line[len(MARKER) :]
                for line in execution["stdout"].splitlines()
                if line.startswith(MARKER)
            )
        )
        if (
            not proof["frozen"]
            or proof["tables"] != 3
            or proof["stata_parity_validated"] is not False
        ):
            raise RuntimeError("The native frozen proof is invalid.")
        workspace = DATA / "projects" / project
        if Path(proof["workspace"]) != workspace or proof["state_file"] != STATE_NAME:
            raise RuntimeError("The QA persistence path changed.")
        frozen_root = args.app.resolve() / "Contents/Resources/runtime/openecon-runtime/_internal"
        if Path(proof["frozen_root"]) != frozen_root:
            raise RuntimeError("The console did not load the installed QA bundle's frozen runtime.")
        for module in proof["modules"].values():
            Path(module["file"]).relative_to(frozen_root)
        state_path = workspace / STATE_NAME
        if state_path.is_symlink() or not 1 <= state_path.stat().st_size < 256 * 1024:
            raise RuntimeError("The persisted QA state file is invalid.")
        if (
            digest(state_path) != proof["states_sha256"]
            or state_path.stat().st_size != proof["states_bytes"]
        ):
            raise RuntimeError("The persisted QA state differs from the native Run.")
        envelope = json.loads(state_path.read_text())
        states = envelope["states"]
        if (
            envelope["schema"] != 1
            or envelope["procedures"] != ["multipletests", "stepdown", "simultaneous_ci"]
            or len(states) != 3
        ):
            raise RuntimeError("The persisted state envelope is invalid.")
        for state in states:
            attrs = state["attrs"]
            if (
                attrs["device"] != "cpu"
                or attrs["precision"] != "float64"
                or attrs["dataset_support"] is not False
                or attrs["stata_parity_validated"] is not False
                or attrs["family_members"] != ["Employment", "Income", "Productivity"]
            ):
                raise RuntimeError(
                    "The declared family and bounded CPU metadata were not preserved."
                )
        if [row["adjusted_p_value"] for row in states[0]["table"]["data"]] != [0.03, 0.2, 0.06]:
            raise RuntimeError("The Holm family did not survive persistence.")
        if [row["adjusted_p_value"] for row in states[1]["table"]["data"]] != [0.5, 0.5, 0.5]:
            raise RuntimeError("The supplied joint-null family did not survive persistence.")
        if states[2]["attrs"]["joint_covariance"] != proof["full_covariance"]:
            raise RuntimeError("The full covariance did not survive persistence.")
        outputs = execution["outputs"]
        if len(outputs) != 3 or any(
            item.get("type") != "table" or not item.get("latex") for item in outputs
        ):
            raise RuntimeError("The native publication outputs are incomplete.")
        fingerprint = hashlib.sha256(
            json.dumps(execution, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if args.after_restart:
            if not record.get("execution_sha256") or record["execution_sha256"] != fingerprint:
                raise RuntimeError("The completed native result changed after restart.")
            if set(record["readback_runtime_pids"]) & set(pids):
                raise RuntimeError(
                    "The previous runtime is still active; a second readback is not a restart."
                )
            require_pids_exited(record["readback_runtime_pids"])
            record["restart"] = {
                "before_runtime_pids": record["readback_runtime_pids"],
                "after_runtime_pids": pids,
                "before_runtime_pids_exited": True,
                "same_execution_sha256": fingerprint,
                "same_states_sha256": digest(state_path),
            }
            record["proof_layers"]["restart_readback_unchanged"] = True
        elif record.get("execution_sha256") and record["execution_sha256"] != fingerprint:
            raise RuntimeError("The pinned native result changed.")
        record.update(
            status="passed",
            execution_id=execution["id"],
            execution_sha256=fingerprint,
            duration_ms=execution["duration_ms"],
            created_at=execution["created_at"],
            completed_at=execution["completed_at"],
            readback_runtime_pids=pids,
            readback_listener_pids=listeners,
            proof=proof,
            states_sha256=digest(state_path),
            ordered_outputs=[item["type"] for item in outputs],
            all_outputs_have_latex=True,
        )
        record["proof_layers"].update(
            source_saved_exact=True, native_run_readback=True, states_file_readback=True
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        (args.receipt.parent / "native-execution-record.json").write_text(
            json.dumps(execution, indent=2, allow_nan=False) + "\n"
        )
        (args.receipt.parent / "persisted-states.json").write_bytes(state_path.read_bytes())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    (args.receipt.parent / "saved-source.py").write_text(code)
    args.receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
