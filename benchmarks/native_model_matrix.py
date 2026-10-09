"""Reproducible physical-source validation of the 80 Dataset model routes.

The capture stage reuses development fixtures and their assertions. It does not
replace sources while those tests execute. The replay stage opens the captured
Parquet/CSV in a new process and compares the whole reported result with the
explicitly bounded resident entry point. That reference is NOT an independent
statistical oracle. Neither stage claims cold OS caches or larger row coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages/openecon-charts/src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot():
    files = [*sorted((ROOT / "benchmarks").glob("native_model_matrix*.py")),
             ROOT / "benchmarks/native_model_matrix_tests.json",
             *sorted((ROOT / "src/openecon").rglob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    return {"git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": hashes,
        "aggregate_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Capture:
    """Observe successful fit returns; do not change numerical test execution."""

    def __init__(self, directory):
        import pandas as pd
        from openecon.dataset import Dataset
        from openecon.models import ResultBundle
        self.pd, self.Dataset, self.ResultBundle = pd, Dataset, ResultBundle
        self.directory = directory
        self.cases, self.reports, self.pending = {}, [], []
        self.item = None
        self.errors = []

    def pytest_collection_modifyitems(self, items):
        # Two geometries per function/model/family, including all model labels.
        counts, selected = {}, []
        for item in items:
            params = getattr(getattr(item, "callspec", None), "params", {})
            key = (item.originalname or item.name,
                   repr([(k, params[k]) for k in ("name", "estimator", "family", "model", "method") if k in params]))
            count = counts.get(key, 0)
            if count < 2:
                selected.append(item)
            counts[key] = count + 1
        deselected = [item for item in items if item not in selected]
        items[:] = selected
        if deselected:
            items[0].config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_setup(self, item):
        self.item, self.pending = item.nodeid, []

    def pytest_runtest_call(self, item):
        sys.setprofile(self.profile)

    def pytest_runtest_teardown(self, item):
        sys.setprofile(None)

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.reports.append({"nodeid": report.nodeid, "outcome": report.outcome,
                                 "seconds": report.duration})
            if report.passed:
                for case in self.pending:
                    self.cases.setdefault(case["id"], case)
            self.pending = []
        if report.failed:
            sys.setprofile(None)

    def frame(self, source):
        if source._kind == "frame":
            return source._frame
        if source._path and source._path.is_file():
            if source._kind == "parquet":
                return self.pd.read_parquet(source._path)
            if source._kind == "csv":
                return self.pd.read_csv(source._path)
        if source._kind == "factory":
            # Read a fixture closure, never replay a potentially mutable factory.
            for cell in getattr(source._factory, "__closure__", None) or ():
                value = cell.cell_contents
                if isinstance(value, self.pd.DataFrame):
                    return value
        return None

    def profile(self, frame, event, result):
        if event != "return" or not isinstance(result, self.ResultBundle):
            return
        if not frame.f_code.co_name.startswith("fit"):
            return
        source = next((v for v in frame.f_locals.values() if isinstance(v, self.Dataset)), None)
        if source is None:
            return
        try:
            data = self.frame(source)
            if data is None:
                return
            specification = result.spec.model_dump(mode="json")
            raw = self.pd.util.hash_pandas_object(data, index=False).values.tobytes()
            schema = [(str(c), str(data[c].dtype),
                       list(data[c].cat.categories) if isinstance(data[c].dtype, self.pd.CategoricalDtype) else None)
                      for c in data]
            identity = hashlib.sha256(json.dumps(
                {"spec": specification, "schema": schema, "metadata": source.metadata},
                sort_keys=True, default=str).encode() + raw).hexdigest()[:24]
            if identity in self.cases or any(c["id"] == identity for c in self.pending):
                return
            path = self.directory / (identity + ".parquet")
            if not path.exists():
                data.to_parquet(path, index=False, row_group_size=127)
            self.pending.append({"id": identity, "model": specification["estimator"],
                "spec": specification, "path": str(path), "rows": len(data),
                "source_sha256": digest(path), "source_bytes": path.stat().st_size,
                "metadata": source.metadata, "fixture_test": self.item,
                "fixture_reference_scope": "Original development assertions; not a physical performance run",
                "capture_adapter": str(Path(frame.f_code.co_filename).relative_to(ROOT)) + ":" + frame.f_code.co_name})
        except Exception as error:
            self.errors.append({"nodeid": self.item, "error": repr(error)})


def capture(args):
    import pytest
    import torch
    torch.set_num_threads(2)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "captured.json").exists():
        raise ValueError("Refusing to replace a previous capture manifest")
    plugin = Capture(directory)
    before = snapshot()
    selected = [*json.loads(args.selection.read_text())] if args.selection else []
    selected.extend(args.tests)
    if not selected:
        raise ValueError("Select the checked model fixtures or supply explicit test nodes")
    test_files = sorted({node.split("::")[0] for node in selected})
    test_hashes = {path: digest(ROOT / path) for path in test_files}
    code = pytest.main([*selected, "-q", "--tb=short"], plugins=[plugin])
    import openecon as oe
    capabilities = oe.capabilities()["streaming"]
    cases = sorted(plugin.cases.values(), key=lambda c: (c["model"], c["id"]))
    save(directory / "captured.json", {"schema": 1, "stage": "fixture_capture",
        "scope": "80 base Dataset routes; later eager models are outside this matrix",
        "code_before": before, "code_after": snapshot(), "cases": cases,
        "test_sources_sha256": test_hashes, "selected_tests": selected,
        "conditions": capabilities["conditions"],
        "models": {name: {"algorithm": capabilities["algorithms"][name],
            "conditions": capabilities["conditions"].get(name, {}),
            "supported_covariances": capabilities["covariances_by_estimator"].get(name),
            "case_ids": [c["id"] for c in cases if c["model"] == name]}
            for name in capabilities["estimators"]},
        "tests": plugin.reports, "capture_errors": plugin.errors,
        "pytest_exit_code": int(code)})
    print(json.dumps({"tests": len(plugin.reports), "cases": len(cases),
        "models": len({c["model"] for c in cases}), "exit_code": int(code),
        "errors": plugin.errors[:5]}))
    return int(code)


def generate(args):
    """Write seeded physical cases; development fixtures are never runtime deps."""
    import openecon as oe
    from native_model_matrix_fixtures import ROSTER, fixture
    import torch
    torch.set_num_threads(2)
    capabilities = oe.capabilities()["streaming"]
    assert len(ROSTER) == len(set(ROSTER)) == 80
    # The historical fixture/receipt roster proves the original80 routes.
    # Subsequent staged routes have independent physical-data receipts.
    assert set(ROSTER) <= set(capabilities["estimators"])
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    before = snapshot()
    cases = []
    selected = args.models or ROSTER
    for i, name in enumerate(selected):
        frame, specification = fixture(name, args.rows)
        assert specification.estimator == name
        extension = "csv" if i % 7 == 0 else "parquet"
        path = directory / (name + "." + extension)
        if extension == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False, row_group_size=8192, compression="zstd")
        case = {"id": name + "-" + digest(path)[:16], "model": name,
                "spec": specification.model_dump(mode="json"), "path": str(path),
                "rows": len(frame), "requested_rows": args.rows,
                "source_sha256": digest(path), "source_bytes": path.stat().st_size,
                "metadata": frame.attrs.get("metadata", {}),
                "fixture_generator": "benchmarks/native_model_matrix_fixtures.py:fixture",
                "fixture_reference_scope": "Seeded development geometry; resident entry on identical parsed rows is not an independent oracle"}
        cases.append(case)
        print(json.dumps({"model": name, "physical_rows": len(frame), "format": extension}), flush=True)
    save(directory / "captured.json", {"schema": 1, "stage": "seeded_physical_source_generation",
        "scope": "Actual physical files containing synthetic development data; no customer/source-data claim",
        "code_before": before, "code_after": snapshot(), "cases": cases,
        "conditions": capabilities["conditions"],
        "models": {name: {"algorithm": capabilities["algorithms"][name],
            "conditions": capabilities["conditions"].get(name, {}),
            "supported_covariances": capabilities["covariances_by_estimator"].get(name),
            "case_ids": [c["id"] for c in cases if c["model"] == name]} for name in ROSTER}})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("capture")
    collector.add_argument("--directory", type=Path, required=True)
    collector.add_argument("--selection", type=Path)
    collector.add_argument("tests", nargs="*")
    writer = sub.add_parser("generate")
    writer.add_argument("--directory", type=Path, required=True)
    writer.add_argument("--rows", type=int, default=1201)
    writer.add_argument("--models", nargs="*")
    args = parser.parse_args()
    return generate(args) if args.command == "generate" else capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
