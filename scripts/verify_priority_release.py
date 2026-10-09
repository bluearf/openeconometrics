"""Unified release runtime/module/UI-asset and upgrade/rollback acceptance.

All executions use a newly owned synthetic project. This verifier never changes
the user's application or project directories and never authenticates to cloud.
Native-window observations and public release readback are separate receipts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback

from verify_bai_perron_runtime import OwnedRuntime, digest
from verify_native_matrix_desktop import bundled_identity

ROOT = Path(__file__).resolve().parents[1]
MARKER = "PRIORITY_RELEASE_OK "


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def analysis_code():
    return """import sys, json, importlib
from pathlib import Path
import openecon as oe
import torch
from openecon.econometrics import registry
assert getattr(sys, "frozen", False)
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
torch.set_num_threads(2)
entries = registry.all_estimators()
for entry in entries:
    assert callable(registry.load_entry(entry)), entry.name
exports = registry.public_exports()
for name, (module, function) in exports.items():
    assert callable(getattr(importlib.import_module(module), function)), name
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
df = oe.example()
df.to_parquet("priority-release-data.parquet", index=False)
dataset = oe.scan("priority-release-data.parquet")
model = oe.ols(data=dataset, y="wage", x=["education", "experience"], covariance="HC3")
Path("priority-release-result.json").write_text(model.model_dump_json())
restored = oe.ResultBundle.model_validate_json(Path("priority-release-result.json").read_text())
assert restored.model_dump() == model.model_dump()
assert restored.covariance_matrix
predicted = oe.predict(restored, data=dataset, interval="mean")
pieces = list(predicted.iter_batches())
assert sum(map(len, pieces)) == len(df)
predicted.close()
margins = oe.margins(restored, data=dataset, variables=["education", "experience"])
display(df.head())
display(restored)
display(pieces[0].head())
display(margins)
display(oe.plot.scatter(data=dataset, x="education", y="wage", title="Education and wage"))
display(oe.plot.hist(data=dataset, x="wage", bins=20, title="Wage distribution"))
display(oe.Latex(restored.to_latex()))
print("PRIORITY_RELEASE_OK " + json.dumps({"frozen": bool(getattr(sys, "frozen", False)),
    "registered_entries_imported": sorted(x.name for x in entries),
    "exports_imported": sorted(exports), "actual_rows": len(df),
    "full_result_restored": True, "no_scipy_loaded": not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)}))
"""


def retained(run):
    # Failed/interrupted executions have no worker timeline. Preserve absence
    # rather than silently replacing either missing or lost events with [].
    result = {key: run[key] for key in ("code", "stdout", "outputs")}
    if "events" in run:
        result["events"] = run["events"]
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def capture_stage(owned, label, directory, *, execution=None):
    """Read the saved log independently of API LaTeX enrichment/projection."""
    path = owned.root / "projects" / owned.project / "console/history.json"
    if path.is_symlink() or not path.resolve().is_relative_to(owned.root.resolve()):
        raise AssertionError("History escaped the owned synthetic project")
    raw = json.loads(path.read_text())["history"]
    value = {
        "stage": label,
        "raw_history": raw,
        "raw_history_sha256": digest(path),
        "api_history": owned.call("/console")["history"],
        "document": owned.call("/console/script"),
        "console_status": owned.request("/api/desktop/status", desktop=True)["console"],
        "execution": execution,
    }
    write_json(directory / (label + ".json"), value)
    return value


def compare_history(before, after):
    """Never treat an API preview as proof of the complete persisted record."""
    if before["raw_history_sha256"] != after["raw_history_sha256"]:
        raise AssertionError("Raw saved history bytes changed across runtime lifecycle")
    if before["raw_history"] != after["raw_history"]:
        raise AssertionError("Raw saved history changed across runtime lifecycle")
    original, reopened = before["api_history"], after["api_history"]
    if [row["id"] for row in original] != [row["id"] for row in reopened]:
        raise AssertionError("API history identifiers/order changed")
    differences = []
    presentation = {"latex", "latex_math", "latex_style", "latex_notes"}

    def clean(row):
        return [
            {key: value for key, value in item.items() if key not in presentation}
            for item in row["outputs"]
        ]

    for first, second in zip(original, reopened, strict=True):
        if retained(first) == retained(second):
            continue
        if any(first[key] != second[key] for key in ("code", "stdout")):
            raise AssertionError("API code/stdout changed")
        # Only an absent or empty optional timeline can differ. In particular,
        # dropping a nonempty timeline still fails even with intact raw bytes.
        if first.get("events", []) != second.get("events", []):
            raise AssertionError("Nonempty API events changed or were lost")
        if clean(first) != clean(second):
            raise AssertionError("API output content changed beyond LaTeX presentation")
        differences.append(
            {
                "execution_id": first["id"],
                "optional_events_presence_changed": ("events" in first) != ("events" in second),
                "latex_presentation_changed": first["outputs"] != second["outputs"],
            }
        )
    return differences


def run_check(run):
    assert run["status"] == "ok", run.get("error")
    proofs = [
        json.loads(line[len(MARKER) :])
        for line in run["stdout"].splitlines()
        if line.startswith(MARKER)
    ]
    assert len(proofs) == 1 and proofs[0]["frozen"] and proofs[0]["no_scipy_loaded"]
    assert [item["type"] for item in run["outputs"]] == [
        "table",
        "model",
        "table",
        "table",
        "plot",
        "plot",
        "latex",
    ]
    # Console previews intentionally omit the quadratic V matrix. The complete
    # persisted ResultBundle was independently restored and compared in code.
    assert "covariance_matrix" in run["outputs"][1]["data"]["display_omitted"]
    assert run["outputs"][1]["data"]["coefficients"]
    assert all(
        any(token in item.get("latex", "") for token in ("\\begin{tabular}", "\\begin{longtable}"))
        for item in run["outputs"][:4]
    )
    return proofs[0]


def verify(app, previous_runtime, directory):
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "active_stage": "identity",
        "completed_stages": [],
        "verifier_sha256": digest(__file__),
        "human_project_access": False,
        "native_window_verified": False,
        "public_release_verified": False,
    }
    root = None

    def stage(label):
        receipt["active_stage"] = label
        write_json(directory / "acceptance.json", receipt)

    def saved(owned, label, *, execution=None):
        value = capture_stage(owned, label, directory, execution=execution)
        receipt["completed_stages"].append(label)
        write_json(directory / "acceptance.json", receipt)
        return value

    try:
        stage("identity")
        identity = bundled_identity(app, "org.openecon.qa.priorityseven")
        receipt["identity"] = identity
        runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
        bundled = app / "Contents/Resources/runtime/openecon-runtime/_internal/openecon/static"
        static_hashes = {}
        for source in sorted((ROOT / "src/openecon/static").rglob("*")):
            if source.is_file():
                relative = source.relative_to(ROOT / "src/openecon/static")
                assert digest(source) == digest(bundled / relative), relative
                static_hashes[str(relative)] = digest(source)
        assert static_hashes
        receipt["static_assets_sha256"] = static_hashes
        receipt["previous_runtime_sha256"] = digest(previous_runtime)
        history_hashes, startup = [], []
        with tempfile.TemporaryDirectory(prefix="openecon-priority-release-owned-") as temporary:
            root = Path(temporary)
            try:
                stage("old-seed")
                # Seed an actual old-runtime project, without copying human data.
                old = OwnedRuntime(previous_runtime, root, "unused")
                try:
                    old.open_project(create=True)
                    project = old.project
                    original = old.call(
                        "/console/execute",
                        {"code": "print('owned pre-upgrade project')", "timeout_seconds": 30},
                    )
                    seed = saved(old, "old-seed", execution=original)
                    assert original["status"] == "ok"
                finally:
                    old.close()
                stage("upgrade-open")
                started = time.monotonic()
                owned = OwnedRuntime(runtime, root, project)
                startup.append(time.monotonic() - started)
                try:
                    owned.open_project()
                    upgraded = saved(owned, "upgrade-open")
                    upgrade_differences = compare_history(seed, upgraded)
                    stage("analysis")
                    code = analysis_code()
                    owned.call(
                        "/console/script", {"code": code, "name": "analysis.py"}, method="PUT"
                    )
                    run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
                    analyzed = saved(owned, "analysis", execution=run)
                    proof = run_check(run)
                    document = analyzed["document"]
                    result_path = root / "projects" / project / "priority-release-result.json"
                    complete_result_sha256 = digest(result_path)
                    stage("stop")
                    # Actual interrupt, followed by recovery in a replacement worker.
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            owned.call,
                            "/console/execute",
                            {
                                "code": "import time; print('owned stop probe', flush=True); time.sleep(30)",
                                "timeout_seconds": 60,
                            },
                        )
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            status = owned.request("/api/desktop/status", desktop=True)
                            if status["console"].get("execution_id"):
                                break
                            time.sleep(0.1)
                        else:
                            raise AssertionError("Stop probe did not enter execution")
                        interrupt = owned.call("/console/interrupt", {})
                        write_json(directory / "interrupt-response.json", interrupt)
                        stopped = future.result(timeout=20)
                    saved(owned, "stop", execution=stopped)
                    assert stopped["status"] in {"interrupted", "cancelled"}, stopped.get("status")
                    stage("recovery")
                    recovery = owned.call(
                        "/console/execute",
                        {
                            "code": "assert 'model' not in globals(); print('recovered')",
                            "timeout_seconds": 30,
                        },
                    )
                    saved(owned, "recovery", execution=recovery)
                    assert recovery["status"] == "ok"
                    stage("reset")
                    write_json(directory / "reset-response.json", owned.call("/console/reset", {}))
                    before = saved(owned, "reset")
                    assert before["console_status"]["pid"] is None
                finally:
                    owned.close()
                # Same complete saved project opened by previous and then current runtime.
                for executable, label in (
                    (previous_runtime, "rollback"),
                    (runtime, "current-restart"),
                ):
                    stage(label)
                    started = time.monotonic()
                    reopened = OwnedRuntime(executable, root, project)
                    startup.append(time.monotonic() - started)
                    try:
                        reopened.open_project()
                        after = saved(reopened, label)
                        differences = compare_history(before, after)
                        assert after["document"] == document
                        assert digest(result_path) == complete_result_sha256
                        history_hashes.append(
                            {
                                "stage": label,
                                "raw_history_sha256": after["raw_history_sha256"],
                                "api_history_sha256": payload_hash(after["api_history"]),
                                "api_presentation_differences": differences,
                                "complete_result_sha256": complete_result_sha256,
                            }
                        )
                        assert after["console_status"]["pid"] is None
                    finally:
                        reopened.close()
            finally:
                log = root / "verifier-stderr.log"
                if log.exists():
                    (directory / "runtime-stderr.log").write_bytes(log.read_bytes())
        receipt.update(
            status="passed",
            active_stage="complete",
            proof=proof,
            run=run,
            stop_status=stopped["status"],
            recovery=recovery["status"],
            history=history_hashes,
            startup_seconds=startup,
            upgrade_api_presentation_differences=upgrade_differences,
            complete_code_outputs_events_equal_across_upgrade_rollback_restart=True,
            complete_saved_history_bytes_equal_across_upgrade_rollback_restart=True,
            complete_result_sha256=complete_result_sha256,
            persistence_evidence="Raw saved history bytes/JSON, including presence and complete contents of optional events; API LaTeX presentation differences are recorded separately.",
            cache_boundary="Fresh runtime processes; shared host OS cache, no cold-cache claim",
        )
    except BaseException as error:
        receipt.update(
            status="failed",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        write_json(directory / "failure.json", receipt["failure"])
        raise
    finally:
        receipt["temporary_project_removed"] = root is None or not root.exists()
        write_json(directory / "acceptance.json", receipt)
    print(
        json.dumps(
            {
                "status": "passed",
                "entry_count": len(proof["registered_entries_imported"]),
                "exports": len(proof["exports_imported"]),
                "startup_seconds": startup,
            }
        )
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--previous-runtime", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    verify(args.app, args.previous_runtime, args.directory)
