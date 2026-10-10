"""Verify all twenty teaching labs in fresh Python and real workbench workers.

Run from the repository root with ``uv run python scripts/verify_teaching_labs.py``.
No empirical data, network access, SciPy or additional test framework is needed.
The default receipt is written under ignored ``artifacts/teaching``. Temporary
workspaces are owned by this check and removed when their checks finish.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
LABS = {
    "01-economic-question": ["table", "table", "plot"],
    "02-wage-distributions": ["table", "plot", "plot"],
    "03-sampling": ["table", "plot", "plot", "plot"],
    "04-class-size": ["latex", "plot", "plot"],
    "05-inference": ["model", "table", "plot"],
    "06-controls": ["model", "model", "latex", "plot"],
    "07-interactions": ["model", "table", "plot"],
    "08-functional-form": ["model", "table", "plot"],
    "09-robust-uncertainty": ["model", "plot", "latex"],
    "10-joint-tests": ["model", "table", "latex"],
    "11-panel-fixed-effects": ["model", "model", "plot", "latex"],
    "12-binary-outcomes": ["model", "model", "table", "plot", "latex"],
    "13-instrumental-variables": ["model", "model", "model", "model", "latex"],
    "14-randomized-program": ["model", "model", "table", "latex"],
    "15-difference-in-differences": ["table", "model", "plot", "model"],
    "16-regression-discontinuity": ["plot", "model", "table"],
    "17-trending-series": ["model", "model", "table", "table", "table", "table", "plot"],
    "18-serial-correlation": ["model", "table", "plot"],
    "19-forecasting": ["model", "table", "plot", "table"],
    "20-research-report": ["model", "table", "table", "plot"],
}
TOLERANCE = 1e-8
SCIENTIFIC_FIELDS = (
    "spec",
    "nobs",
    "nobs_original",
    "dropped_rows",
    "coefficients",
    "covariance_matrix",
    "metrics",
    "warnings",
    "predictions",
    "sample_positions",
    "inference",
    "title",
    "tests",
    "extra",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected, path="result"):
    """Compare scientific values, preserving structure and nonnumeric meaning."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")
    elif isinstance(expected, (int, float)):
        require(
            not isinstance(actual, bool) and isinstance(actual, (int, float)),
            f"{path}: expected a number, got {type(actual).__name__}",
        )
        require(math.isfinite(actual) and math.isfinite(expected), f"{path}: nonfinite value")
        require(
            math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE),
            f"{path}: {actual!r} != {expected!r} within {TOLERANCE}",
        )
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"{path}: list structure differs",
        )
        for index, (value, reference) in enumerate(zip(actual, expected)):
            compare(value, reference, f"{path}[{index}]")
    elif isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            f"{path}: dictionary keys differ",
        )
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    else:
        raise AssertionError(f"{path}: unexpected reference type {type(expected).__name__}")


def verify_result(actual, reference):
    """Check saved numerical evidence and complete, restorable fitted models."""
    from openecon.models import ResultBundle

    require(actual.keys() == reference.keys(), "Full lab result sections differ")
    compare(actual["summary"], reference["summary"], "summary")
    compare(actual["chart_data"], reference["chart_data"], "chart_data")
    for section in reference.keys() - {"metadata", "summary", "models", "chart_data", "checks"}:
        compare(actual[section], reference[section], section)
    require(actual["models"].keys() == reference["models"].keys(), "Model names differ")
    for name, payload in actual["models"].items():
        require(
            "covariance_matrix" in payload and "sample_positions" in payload,
            f"{name}: a display preview was substituted for a complete result",
        )
        restored = ResultBundle.model_validate_json(json.dumps(payload, allow_nan=False))
        require(
            restored.model_dump(mode="json") == payload,
            f"{name}: full ResultBundle JSON roundtrip differs",
        )
        require(
            len(payload["sample_positions"]) == payload["nobs"],
            f"{name}: estimation sample positions are incomplete",
        )
        require(
            len(payload["covariance_matrix"]) == len(payload["coefficients"]),
            f"{name}: covariance dimensions differ from the coefficients",
        )
        # IDs, timestamps and runtime-version provenance are not numerical oracles.
        # They remain preserved in the exported payload, but may differ on rerun.
        for field in SCIENTIFIC_FIELDS:
            compare(payload[field], reference["models"][name][field], f"models.{name}.{field}")
    for name, value in actual["checks"].items():
        if isinstance(value, bool):
            require(value, f"Lab's independent calculation failed: {name}")
        else:
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and abs(value) < TOLERANCE,
                f"Lab calculation error exceeds tolerance: {name}",
            )
    if "seed" in actual["metadata"]:
        require(actual["metadata"]["seed"] == reference["metadata"]["seed"],
                "Original generation seed differs")
    else:
        require(bool(actual["metadata"].get("data_file") or actual["metadata"].get("data_source")),
                "Prepared-input provenance is missing")
    if "data_sha256" in actual["metadata"] and any(
        key in actual["chart_data"] for key in ["scatter", "workers"]
    ):
        # A byte hash identifies one realization, but tiny floating-point
        # differences between CPU architectures may change its JSON bytes.
        # Compare all rows numerically above, and independently check that the
        # current recorded hash identifies the current data rather than a stale
        # reference. Preserve both hashes in the receipt for provenance.
        rows_key = "scatter" if "scatter" in actual["chart_data"] else "workers"
        rows = actual["chart_data"][rows_key]
        digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        require(
            digest == actual["metadata"]["data_sha256"],
            "Recorded generated-data checksum does not identify the current rows",
        )


def result_from_namespace(namespace):
    candidates = [
        value
        for value in namespace.values()
        if isinstance(value, dict)
        and {"metadata", "summary", "models", "chart_data", "checks"} <= value.keys()
    ]
    require(len(candidates) == 1, "The whole-file entry did not leave one complete lab result")
    return candidates[0]


def standalone_worker(folder, receipt_path):
    """Run inside an independent OS process whose cwd starts empty."""
    import openecon as oe
    import torch

    started = time.perf_counter()
    source = (folder / "lab.py").read_text(encoding="utf-8")
    reference = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
    displayed = []
    namespace = {"__name__": "__main__", "display": displayed.append}
    require(not list(Path.cwd().iterdir()), "Standalone working directory was not empty")
    metadata_file = folder / "dataset.json"
    prepared_files = []
    if metadata_file.is_file():
        filename = json.loads(metadata_file.read_text())["filename"]
        shutil.copy2(folder / filename, filename)
        prepared_files = [filename]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(source, f"{folder.name}/lab.py", "exec"), namespace)
    require("__file__" not in namespace, "Editor simulation unexpectedly had __file__")
    require(sorted(path.name for path in Path.cwd().iterdir()) == prepared_files,
            "Default lab execution wrote files")
    actual = result_from_namespace(namespace)
    verify_result(actual, reference)
    require(len(displayed) == len(LABS[folder.name]), "Unexpected standalone display count")
    with contextlib.redirect_stdout(captured):
        exported = namespace["run_lab"](output_dir="explicit-export")
    saved_path = Path("explicit-export/reference.json")
    require(saved_path.is_file(), "Explicit full-result export was not written")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    require(saved == exported, "Explicit exported JSON differs from the returned result")
    verify_result(saved, reference)
    table = Path("explicit-export/table.tex").read_text(encoding="utf-8")
    require(
        r"\begin{table}" in table and r"\toprule" in table,
        "Explicit publication table is missing its table or booktabs structure",
    )
    receipt = {
        "status": "passed",
        "duration_seconds": round(time.perf_counter() - started, 4),
        "python": platform.python_version(),
        "openecon": oe.__version__,
        "torch": str(torch.__version__),
        "standalone_display_count": len(displayed),
        "full_models_verified": len(saved["models"]),
        "independent_lab_checks": len(saved["checks"]),
        "fresh_process": True,
        "no_file_global": True,
        "no_default_writes": True,
        "prepared_input_files": prepared_files,
        "explicit_export_readback": True,
        "full_result_json_roundtrip": True,
        "reference_scientific_fields_match": True,
        "reference_chart_data_match": True,
        "generated_data_sha256": saved["metadata"].get("data_sha256"),
        "reference_generated_data_sha256": reference["metadata"].get("data_sha256"),
        "stdout_characters": len(captured.getvalue()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def verify_console(folder, reference):
    """Execute the exact source through the application's native worker/history."""
    from openecon.console import ConsoleSession
    from openecon.workspace import Workspace

    started = time.perf_counter()
    source = (folder / "lab.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix=f"teaching-console-{folder.name}-") as temporary:
        workspace = Workspace(Path(temporary) / "workspace")
        metadata_file = folder / "dataset.json"
        imported = None
        if metadata_file.is_file():
            filename = json.loads(metadata_file.read_text())["filename"]
            imported = workspace.import_file(folder / filename)
        session = ConsoleSession(workspace)
        try:
            execution = session.execute(source, timeout_seconds=60)
            require(
                execution["status"] == "ok" and execution["error"] is None,
                f"Native console execution failed: {execution.get('error')}",
            )
            actual_types = [item["type"] for item in execution["outputs"]]
            require(
                actual_types == LABS[folder.name],
                f"Native output order/types differ: {actual_types}",
            )
            require(
                "Display limit reached" not in execution["stdout"],
                "Native console truncated the lab's outputs",
            )
            history = Workspace(workspace.path).console_history()
            require(
                len(history) == 1 and history[0]["code"] == source,
                "Saved native history does not contain the exact whole-file source",
            )
            require(
                history[0]["outputs"] == execution["outputs"]
                and history[0]["events"] == execution["events"],
                "Saved native output/event readback differs",
            )
            require(
                not list(workspace.path.rglob("reference.json"))
                and not list(workspace.path.rglob("table.tex")),
                "Default native lab execution exported result files",
            )
            plot_samples = []
            for output in execution["outputs"]:
                if output["type"] == "model":
                    payload = output["data"]
                    require(
                        payload["nobs"] > 0 and payload["coefficients"],
                        "Native model display lost its scientific result",
                    )
                    require(
                        "sample_positions" in payload.get("display_omitted", []),
                        "Native model display no longer declares preview omissions",
                    )
                if output["type"] == "plot":
                    plot = output["data"]
                    require(plot.get("sample_n", 0) > 0, "Native chart has no sample")
                    plot_samples.append(plot["sample_n"])
                if output["type"] in {"model", "table", "latex"}:
                    require(bool(output.get("latex")), "Native publication LaTeX is missing")
            # This is an explicitly requested export in a second command. The
            # first command above proves the default entry made no such export.
            exported = session.execute(
                'assert "__file__" not in globals()\n'
                'verification_result = run_lab(output_dir="verification-export")\n',
                timeout_seconds=60,
            )
            require(
                exported["status"] == "ok" and exported["error"] is None,
                f"Native explicit export failed: {exported.get('error')}",
            )
            saved = json.loads(
                (workspace.path / "verification-export/reference.json").read_text(encoding="utf-8")
            )
            verify_result(saved, reference)
            require(
                (workspace.path / "verification-export/table.tex").is_file(),
                "Native explicit export lost its publication table",
            )
            output_counts = dict(Counter(actual_types))
        finally:
            session.close()
        reopened = ConsoleSession(Workspace(workspace.path))
        try:
            restored_history = reopened.snapshot()["history"]
            require(
                len(restored_history) == 2
                and restored_history[0]["outputs"] == execution["outputs"],
                "Reopened application session did not preserve the lab's saved outputs",
            )
        finally:
            reopened.close()
    return {
        "status": "passed",
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_order": actual_types,
        "output_counts": output_counts,
        "plot_sample_counts": plot_samples,
        "exact_source_execution": True,
        "no_errors": True,
        "saved_history_readback": True,
        "cold_session_history_readback": True,
        "explicit_export_scientific_fields_match": True,
        "prepared_workbook_imported": imported is not None,
    }


class TeachingLabCase(unittest.TestCase):
    def __init__(self, lab_name, receipts):
        super().__init__("runTest")
        self.lab_name = lab_name
        self.receipts = receipts

    def __str__(self):
        return f"Teaching lab {self.lab_name}"

    def runTest(self):
        folder = ROOT / "docs/teaching/labs" / self.lab_name
        record = {
            "lab": self.lab_name,
            "status": "running",
            "source_sha256": hashlib.sha256((folder / "lab.py").read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(
                (folder / "reference.json").read_bytes()
            ).hexdigest(),
        }
        self.receipts.append(record)
        try:
            if (folder / "dataset.json").is_file():
                from prepare_teaching_datasets import verify_workbook
                record["prepared_dataset"] = verify_workbook(folder)
                source = (folder / "lab.py").read_text()
                require("def make_data(" not in source and "manual_seed(" not in source,
                        "Student analysis still regenerates its prepared observations")
            with tempfile.TemporaryDirectory(prefix="teaching-standalone-") as temporary:
                task_root = Path(temporary)
                working = task_root / "empty-working-directory"
                working.mkdir()
                receipt_path = task_root / "receipt.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(folder),
                        "--worker-receipt",
                        str(receipt_path),
                    ],
                    cwd=working,
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                    env=os.environ.copy(),
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"Fresh-process check failed:\n{completed.stdout}\n{completed.stderr}",
                )
                record["standalone"] = json.loads(receipt_path.read_text(encoding="utf-8"))
            reference = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
            record["console"] = verify_console(folder, reference)
            record["status"] = "passed"
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-report", type=Path, default=ROOT / "artifacts/teaching/verification.json"
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--labs", nargs="+", help="Verify selected slugs during editing; default: all twenty"
    )
    parser.add_argument("--worker-receipt", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        if not arguments.worker_receipt:
            parser.error("--worker-receipt is required with --worker")
        standalone_worker(arguments.worker.resolve(), arguments.worker_receipt.resolve())
        return 0

    started = time.perf_counter()
    receipts = []
    catalog = json.loads((ROOT / "docs/teaching/catalog.json").read_text())
    names = arguments.labs or [lab["slug"] for lab in catalog["labs"]]
    if not arguments.labs and len(names) != 20:
        raise ValueError("The complete course verifier requires exactly twenty lessons.")
    for name in names:
        if name not in LABS:
            raise ValueError(f"Missing independently reviewed console output contract: {name}")
        text = (ROOT / "docs/teaching/labs" / name / "README.md").read_text()
        require(
            (ROOT / "docs/teaching/instructors" / f"{name}.md").is_file(),
            f"Separate instructor guide missing: {name}",
        )
        for pattern in [
            r"(?im)^#{1,6} .*instructor",
            r"(?im)^#{1,6} .*worked answers",
            r"(?im)^#{1,6} .*reproduc",
            r"(?i)\b\d+(?:\s*[–-]\s*\d+)?\s+minutes\b",
            r"(?i)no previous .*experience is required",
            r"\]\([^)]*instructors/",
            r"\]\(reference\.json\)",
        ]:
            require(
                not re.search(pattern, text),
                f"Excluded instructor/timing material in student text: {name}",
            )
    suite = unittest.TestSuite(TeachingLabCase(name, receipts) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "numerical_relative_and_absolute_tolerance": TOLERANCE,
        "labs_run": result.testsRun,
        "labs_passed": sum(item["status"] == "passed" for item in receipts),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "labs": receipts,
    }
    arguments.output_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Verification receipt: {arguments.output_report}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
