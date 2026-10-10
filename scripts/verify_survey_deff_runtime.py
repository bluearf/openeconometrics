"""Verify complete survey DEFF artifacts in a temporary owned frozen runtime.

The original execution, every result/contrast/reference table and exact JSON
bytes are retained. Worker reset and server restart replay saved states with
fitting disabled. Native-window acceptance is a separate external UI step.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from verify_bai_perron_runtime import OwnedRuntime
from verify_causal_targets_runtime import digest, normalized

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/survey_deff.py"
MARKER = "SURVEY_DEFF_RECEIPT:"
ARTIFACT = "survey-deff-artifacts.json"
SCHEMA = "survey-deff-complete-artifacts-v1"
NAMES = tuple(f"{scope}_{kind}" for scope in (
    "unequal_full", "unequal_fixed", "equal_legacy"
) for kind in ("mean", "total", "ratio", "proportion"))
MODULES = (
    "openecon", "openecon.survey", "openecon.econometrics",
    "openecon.econometrics.survey", "openecon.econometrics.survey.common",
    "openecon.econometrics.survey.targets", "openecon.econometrics.registry",
    "openecon.econometrics.core", "openecon.engines", "openecon.engines.inference",
    "openecon.engines.distributions", "openecon.analysis_contracts", "openecon.resources",
    "openecon.models", "openecon.frame", "openecon.latex", "openecon.output_latex",
    "openecon.console", "openecon.console_worker",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n")


def source_identity(runtime, source_ref):
    """Compare immutable source to complete PYZ code; normalize filenames only."""
    from PyInstaller.archive.readers import CArchiveReader

    if not re.fullmatch(r"[0-9a-f]{40}", source_ref):
        raise RuntimeError("Use a complete immutable scientific-source commit SHA")
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    hashes = {}
    for module in MODULES:
        path = ROOT / "src" / Path(*module.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(
            ["git", "show", f"{source_ref}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT,
        )
        expected = compile(source, str(path), "exec", dont_inherit=True)
        if normalized(archive.extract(module)) != normalized(expected):
            raise RuntimeError("Frozen source differs: " + module)
        hashes[module] = hashlib.sha256(source).hexdigest()
    return hashes


def header(directory):
    return (
        "import importlib, importlib.util, os, sys\nfrom pathlib import Path\n"
        "assert getattr(sys, 'frozen', False)\n"
        "assert importlib.util.find_spec('scipy') is None\n"
        "assert importlib.util.find_spec('statsmodels') is None\n"
        f"for _name in {MODULES!r}:\n"
        "    assert Path(importlib.import_module(_name).__file__).resolve().is_relative_to(Path(sys._MEIPASS).resolve()), _name\n"
        f"os.environ['OPENECON_ACCEPTANCE_DIR'] = {str(directory)!r}\n"
    )


def proof_from_execution(execution):
    if execution.get("status") != "ok":
        raise RuntimeError(f"Survey DEFF example failed: {execution.get('error')}")
    markers = [json.loads(line[len(MARKER):]) for line in execution.get("stdout", "").splitlines()
               if line.startswith(MARKER)]
    if len(markers) != 1:
        raise RuntimeError("Require one complete survey DEFF receipt")
    proof = markers[0]
    if (proof.get("schema") != SCHEMA or proof.get("status") != "passed"
            or proof.get("result_names") != list(NAMES)
            or type(proof.get("output_count")) is not int
            or proof["output_count"] != len(proof["result_names"])
            or proof.get("table_count") != 2 * proof["output_count"]
            or proof.get("artifact_filename") != ARTIFACT
            or proof.get("full_covariance_saved") is not True
            or proof.get("restored_equal") is not True
            or proof.get("stata_parity_validated") is not False):
        raise RuntimeError("Incomplete complete-state survey DEFF receipt")
    outputs = execution.get("outputs", [])
    if len(outputs) != proof["output_count"] or any(item.get("type") != "table" for item in outputs):
        raise RuntimeError("Actual native output count differs from the declared result inventory")
    for output in outputs:
        data = output["data"]
        if (data.get("total_rows") != len(data.get("rows", []))
                or data.get("total_columns") != len(data.get("columns", []))
                or not any("\\begin{" + kind + "}" in output.get("latex", "")
                           for kind in ("tabular", "longtable"))):
            raise RuntimeError("A saved representative table was truncated or lacks LaTeX")
    return proof


def artifact_files(directory, proof, execution=None):
    """Check every complete file, integrity state and representative output."""
    directory = Path(directory)
    aggregate = directory / ARTIFACT
    if aggregate.is_symlink() or not aggregate.is_file():
        raise RuntimeError("The aggregate artifact must be an owned regular file")
    if digest(aggregate) != proof["artifact_sha256"] or aggregate.stat().st_size != proof["artifact_bytes"]:
        raise RuntimeError("Aggregate artifact byte hash/size changed")
    packet = json.loads(aggregate.read_text())
    if aggregate.read_bytes() != canonical(packet).encode():
        raise RuntimeError("Aggregate artifact serialization differs from the exact example bytes")
    if (packet.get("schema") != SCHEMA or packet.get("status") != "passed"
            or packet.get("result_names") != list(NAMES)
            or set(packet.get("artifacts", {})) != set(NAMES)
            or packet.get("output_count") != proof["output_count"]
            or packet.get("table_count") != proof["table_count"]
            or packet.get("stata_parity_validated") is not False):
        raise RuntimeError("The aggregate artifact inventory differs from its proof")
    hashes = {ARTIFACT: digest(aggregate)}
    for index, name in enumerate(NAMES):
        path = directory / (name + ".json")
        record = packet["artifacts"][name]
        if path.is_symlink() or path.read_bytes() != canonical(record).encode():
            raise RuntimeError("An individual artifact differs from its complete aggregate: " + name)
        state, table = record["state"], record["table"]
        metadata = state["metadata"]
        if (record["name"] != name or record["covariance"] != state["covariance"]
                or record["srs_reference_covariance"] != metadata["srs_covariance"]
                or record["design_effect"] != metadata["design_effect"]
                or record["reference_state"] != metadata.get("srs_reference_state")
                or state["integrity_sha256"] != fingerprint({
                    key: value for key, value in state.items() if key != "integrity_sha256"
                }) or metadata.get("stata_parity_validated") is not False):
            raise RuntimeError("Complete covariance/reference/state integrity differs: " + name)
        for key in ("table", "contrast"):
            frame = record[key]
            if (set(frame) != {"index", "columns", "dtypes", "data", "attrs"}
                    or len(frame["index"]) != len(frame["data"])
                    or len(frame["dtypes"]) != len(frame["columns"])
                    or any(len(row) != len(frame["columns"]) for row in frame["data"])):
                raise RuntimeError("An artifact table lost values, dtypes, index or attrs: " + name)
        for key in ("table_latex", "contrast_latex"):
            if not any("\\begin{" + kind + "}" in record[key] for kind in ("tabular", "longtable")):
                raise RuntimeError("A complete artifact lacks its LaTeX table: " + name)
        if execution is not None:
            displayed = execution["outputs"][index]["data"]
            if (displayed["rows"] != table["data"] or displayed["columns"] != table["columns"]
                    or displayed["index"] != [[label] for label in table["index"]]):
                raise RuntimeError("A displayed table differs from its complete result artifact: " + name)
        hashes[path.name] = digest(path)
    return hashes


def restore_code(directory, proof):
    """Replay all saved models, references, tables and contrasts without fitting."""
    source = EXAMPLE.read_text()
    functions = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in {"plain", "frame_record"}}
    if set(functions) != {"plain", "frame_record"}:
        raise RuntimeError("The owned example lacks its complete frame serializer")
    return header(directory) + "import hashlib, json, math\nimport openecon as oe\n" + (
        functions["plain"] + "\n" + functions["frame_record"] + "\n"
    ) + f"""
import openecon.econometrics.survey.targets as _targets
def _forbidden_fit(*args, **kwargs):
    raise AssertionError("Saved DEFF restoration attempted a new fit")
_targets.taylor = _forbidden_fit
_targets.prepare = _forbidden_fit
for _name in ('survey_mean', 'survey_total', 'survey_ratio', 'survey_proportion'):
    setattr(oe, _name, _forbidden_fit)
_path = Path({str(directory)!r}) / {ARTIFACT!r}
assert hashlib.sha256(_path.read_bytes()).hexdigest() == {proof['artifact_sha256']!r}
_packet = json.loads(_path.read_text())
assert _packet['result_names'] == {list(NAMES)!r}
for _name in _packet['result_names']:
    _record = _packet['artifacts'][_name]
    _result = oe.SurveyResult.model_validate_json(json.dumps(_record['state'], allow_nan=False))
    assert _result.model_dump(mode='json') == _record['state']
    assert plain(_result.covariance) == _record['covariance']
    assert plain(_result.metadata['srs_covariance']) == _record['srs_reference_covariance']
    _table = _result.to_frame()
    _contrast = _result.contrast(_record['contrast_coefficients'])
    assert frame_record(_table) == _record['table']
    assert frame_record(_contrast) == _record['contrast']
    assert oe.to_latex(_table) == _record['table_latex']
    assert oe.to_latex(_contrast) == _record['contrast_latex']
print('SURVEY_DEFF_RESTORED:' + json.dumps({{'names': _packet['result_names'], 'fitting_disabled': True}}))
"""


def verify(runtime, directory, source_ref):
    runtime = Path(runtime).resolve(strict=True)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources, runtime_hash, example_hash = source_identity(runtime, source_ref), digest(runtime), digest(EXAMPLE)
    with tempfile.TemporaryDirectory(prefix="openecon-survey-deff-owned-") as temporary:
        data = Path(temporary)
        artifacts = data / "complete-artifacts"
        owned = None
        try:
            owned = OwnedRuntime(runtime, data, "unused")
            created = owned.request("/api/desktop/local-projects", {"name": "Survey DEFF Runtime QA"}, desktop=True)
            project = owned.project = created["id"]
            owned.open_project()
            code = header(artifacts) + EXAMPLE.read_text()
            document = owned.call("/console/script", {"name": "analysis.py", "code": code}, method="PUT")
            executed = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
            save_json(directory / "original-execution.json", executed)
            proof = proof_from_execution(executed)
            hashes = artifact_files(artifacts, proof, executed)
            original = next(row for row in owned.call("/console")["history"] if row["id"] == executed["id"])
            save_json(directory / "original-persisted-execution.json", original)
            if original != executed or original["code"] != code:
                raise RuntimeError("The original execution was not saved exactly")
            original_hash = fingerprint(original)
            document = owned.call("/console/script")
            save_json(directory / "saved-script.json", document)
            replay = restore_code(artifacts, proof)
            owned.call("/console/reset", {})
            if owned.call("/console")["status"]["pid"] is not None:
                raise RuntimeError("Worker reset did not stop the original worker")
            reset_run = owned.call("/console/execute", {"code": replay, "timeout_seconds": 120})
            save_json(directory / "worker-reset-restoration.json", reset_run)
            if reset_run.get("status") != "ok" or "SURVEY_DEFF_RESTORED:" not in reset_run.get("stdout", ""):
                raise RuntimeError(f"Fit-disabled restoration failed: {reset_run.get('error')}")
            saved = next(row for row in owned.call("/console")["history"] if row["id"] == original["id"])
            save_json(directory / "after-worker-reset-original.json", saved)
            if fingerprint(saved) != original_hash or artifact_files(artifacts, proof) != hashes:
                raise RuntimeError("Original execution or full artifacts changed after worker reset")
            first_pid = owned.process.pid
            owned.close()
            if owned.process.poll() is None:
                raise RuntimeError("The first owned runtime did not exit")
            owned = OwnedRuntime(runtime, data, project)
            second_pid = owned.process.pid
            owned.open_project()
            saved = next(row for row in owned.call("/console")["history"] if row["id"] == original["id"])
            save_json(directory / "after-runtime-restart-original.json", saved)
            if (first_pid == second_pid or fingerprint(saved) != original_hash
                    or owned.call("/console/script") != document):
                raise RuntimeError("Original execution/script changed or runtime was not restarted")
            restart_run = owned.call("/console/execute", {"code": replay, "timeout_seconds": 120})
            save_json(directory / "runtime-restart-restoration.json", restart_run)
            if restart_run.get("status") != "ok" or "SURVEY_DEFF_RESTORED:" not in restart_run.get("stdout", ""):
                raise RuntimeError(f"Post-restart saved restoration failed: {restart_run.get('error')}")
            if artifact_files(artifacts, proof) != hashes:
                raise RuntimeError("Full artifacts changed after runtime restart")
            if (digest(runtime) != runtime_hash or digest(EXAMPLE) != example_hash
                    or source_identity(runtime, source_ref) != sources):
                raise RuntimeError("Pinned runtime/example/source changed during acceptance")
            receipt = dict(
                status="passed", proof=proof, runtime_sha256=runtime_hash,
                compiled_modules_equal_source=sources, scientific_source_ref=source_ref,
                example_sha256=example_hash, verifier_sha256=digest(__file__),
                execution_id=original["id"], execution_sha256=original_hash,
                executed_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                artifact_files_sha256=hashes, original_outputs_complete=True,
                worker_reset_readback_unchanged=True, runtime_restart_readback_unchanged=True,
                saved_models_tables_contrasts_latex_restored_without_fitting=True,
                first_runtime_pid=first_pid, second_runtime_pid=second_pid,
                source_path_injected=False, external_oracle_packages_absent=True,
                native_window_verified=False, public_release_delivered=False,
            )
        finally:
            if owned is not None:
                owned.close()
            if artifacts.exists():
                shutil.copytree(artifacts, directory / "complete-artifacts")
            error_log = data / "verifier-stderr.log"
            if error_log.exists():
                shutil.copy2(error_log, directory / "runtime-stderr.log")
        receipt["owned_runtime_stopped"] = owned.process.poll() is not None
    receipt["owned_temporary_profile_removed"] = not data.exists()
    save_json(directory / "acceptance.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime, args.directory, args.source_ref)
    print(json.dumps({"status": receipt["status"], "directory": str(args.directory),
                      "outputs": receipt["proof"]["output_count"]}))


if __name__ == "__main__":
    main()
