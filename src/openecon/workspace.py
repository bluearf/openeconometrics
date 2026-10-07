"""Local immutable datasets and atomic, reproducible analysis records."""
from __future__ import annotations

import io
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from uuid import UUID, uuid4
import zipfile

import pandas as pd
from threadpoolctl import threadpool_limits

from openecon.data import (
    HASH_VERSION, STREAM_HASH_VERSION, MAX_FILE_BYTES, DataError, example_frame, frame_hash, metadata_hash,
    profile_frame, profile_dataset, read, validate_frame,
)
from openecon.models import ModelSpec
from openecon.frame import as_frame
from openecon import script_contracts as scripts
from openecon import file_layout

_FIT_LOCK = threading.Lock()
_CONSOLE_LOCKS: dict[Path, threading.RLock] = {}
_CONSOLE_LOCKS_GUARD = threading.Lock()
_MAX_HISTORY_BYTES = 8 * 1024 * 1024
_MAX_AGENT_RECORD_BYTES = 2 * 1024 * 1024


def _bounded_history(history: list[dict]) -> list[dict]:
    history = history[-500:]
    sizes = [len(json.dumps(item, ensure_ascii=False).encode("utf-8")) for item in history]
    total = sum(sizes)
    while total > _MAX_HISTORY_BYTES and len(history) > 1:
        total -= sizes.pop(0)
        history.pop(0)
    return history


def fit(spec: ModelSpec, *, data):
    """Resolve the tensor runtime on analysis; retain the patchable fit hook."""
    from openecon.analysis import fit as analyze
    return analyze(spec, data=data)


def _atomic_json(path: Path, data: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Workspace:
    def __init__(self, path: str | Path = ".openecon"):
        self.path = Path(path).expanduser().resolve()
        self.data_path = self.path / "datasets"
        self.result_path = self.path / "results"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.result_path.mkdir(parents=True, exist_ok=True)
        self.console_path = self.path / "console"
        self.console_path.mkdir(parents=True, exist_ok=True)
        with _CONSOLE_LOCKS_GUARD:
            self._console_lock = _CONSOLE_LOCKS.setdefault(self.path, threading.RLock())

    @staticmethod
    def _id(value: str) -> str:
        try:
            return str(UUID(value))
        except (ValueError, AttributeError) as exc:
            raise DataError("The record identifier is invalid.", "NOT_FOUND") from exc

    def _record(self, directory: Path, record_id: str) -> dict:
        path = directory / f"{self._id(record_id)}.json"
        if not path.is_file() or path.resolve().parent != directory.resolve():
            raise DataError("The requested record was not found.", "NOT_FOUND")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DataError("The saved record cannot be read.", "CORRUPT_RECORD") from exc

    def _save_frame(self, frame: pd.DataFrame, name: str, source: str, *, local_only=False) -> dict:
        frame = validate_frame(frame)
        record_id = str(uuid4())
        profile = profile_frame(frame, name=name, source=source, dataset_id=record_id)
        if local_only:
            profile["local_only"] = True
        path = self.data_path / f"{record_id}.parquet"
        temporary = self.data_path / f"{record_id}.parquet.tmp"
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
            with self._console_lock:
                _atomic_json(self.data_path / f"{record_id}.json", profile)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return profile

    def import_file(self, path: str | Path, *, name: str | None = None, local_only=False) -> dict:
        path = Path(path)
        from openecon.dataset import Dataset, scan
        data = read(path)
        if isinstance(data, Dataset):
            if not path.is_file():
                raise DataError("Import a single CSV/Parquet file; directory sources can be opened with oe.read or oe.scan.", "INVALID_SOURCE")
            record_id = str(uuid4())
            converted = "conversion" in data.provenance
            source_path = data._path if converted else path
            target = self.data_path / f"{record_id}.source{source_path.suffix.lower()}"
            original = self.data_path / f"{record_id}.original{path.suffix.lower()}"
            fd, temporary = tempfile.mkstemp(dir=self.data_path, suffix=source_path.suffix.lower())
            digest = hashlib.sha256()
            try:
                data.assert_unchanged()
                with os.fdopen(fd, "wb") as output, source_path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                    output.flush()
                    os.fsync(output.fileno())
                data.assert_unchanged()
                owned = scan(temporary)
                profile = profile_dataset(owned, name=name or path.name, dataset_id=record_id,
                                          file_digest=digest.hexdigest())
                profile["storage"] = {"kind": "dataset", "file": target.name,
                                      "format": source_path.suffix.lower().lstrip(".")}
                if converted:
                    profile.setdefault("provenance", {})["conversion"] = data.provenance["conversion"]
                    shutil.copyfile(path, original)
                    with original.open("rb") as file:
                        if hashlib.file_digest(file, "sha256").hexdigest() != data.provenance["conversion"]["source_sha256"]:
                            raise DataError("The original conversion source changed before snapshot.", "SOURCE_CHANGED")
                if local_only:
                    profile["local_only"] = True
                os.replace(temporary, target)
                with self._console_lock:
                    _atomic_json(self.data_path / f"{record_id}.json", profile)
                return profile
            except Exception:
                target.unlink(missing_ok=True)
                if converted:
                    original.unlink(missing_ok=True)
                raise
            finally:
                Path(temporary).unlink(missing_ok=True)
                data.close()
        profile = self._save_frame(data, name or path.name, "upload", local_only=local_only)
        # The unmodified source remains available locally, including DTA metadata.
        target = self.data_path / f"{profile['id']}.source{path.suffix.lower()}"
        shutil.copyfile(path, target)
        return profile

    def create_example(self) -> dict:
        for profile in self.list_datasets():
            if profile["source"] == "example":
                self.load_frame(profile["id"])
                return profile
        return self._save_frame(example_frame(), "Wages and education · example data", "example")

    def list_datasets(self) -> list[dict]:
        return sorted([self._record(self.data_path, p.stem) for p in self.data_path.glob("*.json")],
                      key=lambda x: x["created_at"], reverse=True)

    def get_dataset(self, dataset_id: str) -> dict:
        return self._record(self.data_path, dataset_id)

    def _snapshot_path(self, profile: dict) -> Path:
        storage = profile.get("storage", {})
        if not isinstance(storage, dict) or (storage.get("kind") is not None and storage.get("kind") != "dataset"):
            raise DataError("The saved Dataset descriptor is invalid.", "CORRUPT_RECORD")
        if storage.get("kind") == "dataset":
            suffix = storage.get("format")
            expected = f"{self._id(profile['id'])}.source.{suffix}"
            if not isinstance(suffix, str) or suffix not in {"csv", "parquet"} or storage.get("file") != expected:
                raise DataError("The saved Dataset descriptor is invalid.", "CORRUPT_RECORD")
            path = self.data_path / expected
        else:
            path = self.data_path / f"{self._id(profile['id'])}.parquet"
        if not path.is_file() or path.resolve().parent != self.data_path.resolve():
            raise DataError("The dataset snapshot was not found.", "NOT_FOUND")
        return path

    def load_frame(self, dataset_id: str):
        """Restore a small frame or a verified, replayable large Dataset."""
        profile = self.get_dataset(dataset_id)
        path = self._snapshot_path(profile)
        if profile.get("storage", {}).get("kind") == "dataset":
            from openecon.dataset import scan
            if profile.get("hash_version") != STREAM_HASH_VERSION:
                raise DataError("The Dataset snapshot uses a different integrity format.", "HASH_VERSION_MISMATCH")
            data = scan(path)
            if (data.file_hash() != profile.get("data_hash")
                    or metadata_hash(data.metadata) != metadata_hash(profile.get("metadata", {}))
                    or data.columns != [item.get("name") for item in profile.get("columns", [])]):
                raise DataError("The dataset snapshot or descriptor changed after import.", "DATA_INTEGRITY")
            row_count = profile.get("row_count")
            if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
                raise DataError("The Dataset row count is invalid.", "CORRUPT_RECORD")
            if data.row_count is not None and data.row_count != row_count:
                raise DataError("The dataset row count changed after import.", "DATA_INTEGRITY")
            data._row_count = row_count
            data._provenance["row_count"] = row_count
            return data
        if profile.get("hash_version") != HASH_VERSION:
            raise DataError("The dataset uses a different integrity format. Reimport the original source in this version.", "HASH_VERSION_MISMATCH")
        if profile.get("hash_pandas_version") != pd.__version__:
            raise DataError("The saved dataset hash depends on a different pandas version. Use the recorded environment or reimport the original source.", "HASH_VERSION_MISMATCH")
        frame = pd.read_parquet(path)
        if metadata_hash(frame.attrs.get("metadata", {})) != metadata_hash(profile.get("metadata", {})):
            raise DataError("The dataset metadata changed after import.", "DATA_INTEGRITY")
        if frame_hash(frame) != profile["data_hash"]:
            raise DataError("The dataset snapshot changed after import.", "DATA_INTEGRITY")
        frame.attrs["metadata"] = profile.get("metadata", {})
        return as_frame(frame)

    def compute_analysis(self, dataset_id: str, spec: ModelSpec | dict) -> dict:
        """Compute without publishing; interruptible jobs commit only after completion."""
        if not isinstance(spec, ModelSpec):
            spec = ModelSpec.model_validate(spec)
        profile = self.get_dataset(dataset_id)
        frame = self.load_frame(dataset_id)
        with _FIT_LOCK, threadpool_limits(limits=1):
            result = fit(spec, data=frame).model_dump(mode="json")
        result.update({"dataset_id": dataset_id, "dataset_name": profile["name"]})
        result["provenance"]["dataset_snapshot_hash"] = profile["data_hash"]
        result["provenance"]["dataset_hash_version"] = profile["hash_version"]
        result["provenance"]["dataset_hash_pandas_version"] = profile["hash_pandas_version"]
        result["provenance"]["source"] = profile["source"]
        if profile.get("metadata", {}).get("synthetic"):
            result["provenance"]["synthetic_data"] = True
            result["provenance"]["data_generation_seed"] = profile["metadata"]["seed"]
        return result

    def run_analysis(self, dataset_id: str, spec: ModelSpec | dict) -> dict:
        result = self.compute_analysis(dataset_id, spec)
        _atomic_json(self.result_path / f"{self._id(result['id'])}.json", result)
        return result

    def get_result(self, result_id: str) -> dict:
        return self._record(self.result_path, result_id)

    def list_results(self) -> list[dict]:
        return sorted([self._record(self.result_path, p.stem) for p in self.result_path.glob("*.json")],
                      key=lambda x: x["created_at"], reverse=True)

    def save_agent_result(self, result: dict, *, duration_ms: float = 0) -> None:
        """Publish one bounded UI view without writing the console's shared history.

        Each agent process owns its new UUID record, so simultaneous agent and
        console writes cannot overwrite each other's read-modify-write history.
        The full model remains in results/ for reproducible export.
        """
        record = self.agent_result_record(result, duration_ms=duration_ms)
        directory = self.console_path / "mcp-results"
        directory.mkdir(exist_ok=True)
        _atomic_json(directory / f"{record['id']}.json", record)

    def agent_result_record(self, result: dict, *, duration_ms: float = 0) -> dict:
        """Validate the bounded display before a background job publishes either file."""
        from openecon.mcp_server import compact_result
        result_id = self._id(result["id"])
        display = compact_result(result)
        for field in ("dataset_id", "dataset_name", "observation_data_included"):
            display.pop(field, None)
        display.update({field: result[field] for field in ("title", "tests") if field in result})
        display.update(predictions=[], display_omitted=["predictions", "sample_positions", "covariance_matrix"])
        spec = json.dumps(result["spec"], ensure_ascii=False)
        code = (
            "import json\nimport openecon as oe\n\n"
            f"data = oe.load_dataset({result['dataset_id']!r})\n"
            f"spec = oe.ModelSpec.model_validate(json.loads({spec!r}))\n"
            "model = oe.fit(spec, data=data)\ndisplay(model)\n"
        )
        record = {
            "id": result_id, "source": "mcp", "result_id": result_id,
            "created_at": result["created_at"], "code": code, "status": "ok",
            "stdout": "", "error": None, "outputs": [{"type": "model", "data": display}],
            "events": [{"type": "output", "index": 0}], "variables": [],
            "duration_ms": duration_ms, "session_generation": 0,
        }
        if len(json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")) > _MAX_AGENT_RECORD_BYTES:
            raise DataError("The agent result display exceeds the local history limit.", "DISPLAY_LIMIT")
        return record

    def agent_results_revision(self) -> str:
        """A cheap change token; no result files or model arrays are loaded."""
        directory = self.console_path / "mcp-results"
        try:
            status = directory.stat()
        except FileNotFoundError:
            return "0"
        return f"{status.st_ino}:{status.st_mtime_ns}"

    def display_history(self) -> list[dict]:
        """Merge independently persisted agent views with the local console log."""
        history = self.console_history()
        directory = self.console_path / "mcp-results"
        if not directory.is_dir():
            return history

        def candidates():
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        status = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if status.st_size <= _MAX_AGENT_RECORD_BYTES:
                        yield status.st_mtime_ns, entry.name, status.st_size

        # Budget file bytes before JSON parsing, rather than accumulating up to
        # 500 x 2 MiB and trimming after the allocations have already happened.
        remaining = _MAX_HISTORY_BYTES
        for _, name, size in heapq.nlargest(500, candidates()):
            if size > remaining:
                break
            history.append(self._record(directory, Path(name).stem))
            remaining -= size
        history.sort(key=lambda item: (item.get("created_at", ""), item["id"]))
        return _bounded_history(history)

    def python_script(self, result_id: str, *, portable: bool = False) -> str:
        result = self.get_result(result_id)
        profile = self.get_dataset(result["dataset_id"])
        data_path = self._snapshot_path(profile)
        streamed = profile.get("storage", {}).get("kind") == "dataset"
        spec_json = json.dumps(result["spec"], ensure_ascii=False, indent=2)
        expected_versions = json.dumps(result["provenance"]["versions"], sort_keys=True)
        expected_hash = result["provenance"]["dataset_snapshot_hash"]
        expected_hash_version = result["provenance"]["dataset_hash_version"]
        expected_pandas = result["provenance"]["dataset_hash_pandas_version"]
        expected_sample = result["provenance"]["sample_hash"]
        filename = "dataset" + data_path.suffix
        path_expr = f'Path(__file__).resolve().parent / {filename!r}' if portable else repr(str(data_path))
        version_constant = "STREAM_HASH_VERSION" if streamed else "HASH_VERSION"
        loader = "scan" if streamed else "read"
        return (
            '# Generated by OpenEconometrics. Data stays local; no external AI service is called.\n'
            'import json\nimport platform\nimport warnings\n'
            'from importlib.metadata import version\nfrom pathlib import Path\n'
            'import openecon as oe\nfrom openecon.data import HASH_VERSION, STREAM_HASH_VERSION, dataset_hash\n\n'
            f'expected_versions = json.loads({expected_versions!r})\n'
            'current_versions = {name: platform.python_version() if name == "python" else version(name)\n'
            '                    for name in expected_versions}\n'
            f'if {version_constant} != {expected_hash_version!r}:\n'
            '    raise RuntimeError("Dataset hash format differs. Use the recorded OpenEconometrics environment.")\n'
            f'if current_versions["pandas"] != {expected_pandas!r}:\n'
            '    raise RuntimeError("Pandas version differs from the saved hash environment. Row hashes depend on pandas; use the recorded lockfile.")\n'
            'for name, expected in expected_versions.items():\n'
            '    if current_versions[name] != expected:\n'
            '        warnings.warn(f"Replay environment differs: {name} {current_versions[name]} installed; {expected} recorded. Numerical equivalence is not guaranteed.", RuntimeWarning)\n\n'
            f'data = oe.{loader}({path_expr})\n'
            f'if dataset_hash(data) != {expected_hash!r}:\n'
            '    raise RuntimeError("Dataset integrity check failed: values, schema, category coding or metadata differ from the recorded snapshot.")\n'
            f'spec = oe.ModelSpec.model_validate(json.loads({spec_json!r}))\n'
            'result = oe.fit(spec, data=data)\n'
            f'if result.provenance["sample_hash"] != {expected_sample!r}:\n'
            '    raise RuntimeError("The estimation sample differs from the recorded analysis. Review the environment and model semantics.")\n'
            'print(result.model_dump_json(indent=2))\n'
        )

    def export_bundle(self, result_id: str, *, destination: str | Path | None = None):
        result = self.get_result(result_id)
        # Validate the saved source before claiming the export can reproduce it.
        data = self.load_frame(result["dataset_id"])
        data_path = self._snapshot_path(self.get_dataset(result["dataset_id"]))
        if destination is None and data_path.stat().st_size > MAX_FILE_BYTES:
            raise DataError("Write large bundles to a file with export_bundle(..., destination=path).", "STREAMING_EXPORT_REQUIRED")
        output = Path(destination).expanduser() if destination is not None else io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(data_path, "dataset" + data_path.suffix)
            archive.writestr("analysis.py", self.python_script(result_id, portable=True))
            archive.writestr("result.json", json.dumps(result, ensure_ascii=False, indent=2))
            archive.writestr("dataset.json", json.dumps(self.get_dataset(result["dataset_id"]), ensure_ascii=False, indent=2))
            archive.writestr("README.txt", (
                "OpenEconometrics reproducibility bundle\n\n"
                "This archive contains the dataset, including its original imported columns.\n"
                "Install OpenEconometrics from its source repository using the corresponding uv.lock.\n"
                "Install the recorded environment: uv sync --frozen. PyTorch is a required dependency.\n"
                "Current model execution uses the OpenEconometrics float64 CPU core.\n"
                "Run: python analysis.py\n"
                "See result.json for model, sample, inference and package versions.\n"
                "The script verifies dataset and estimation-sample hashes before returning results.\n"
                "Dataset hashes include category order and metadata and depend on the recorded pandas version.\n"
                "A pandas/hash-format mismatch stops replay; other version differences emit an explicit warning.\n"
                "New result IDs and timestamps are expected on rerun.\n"
                "Floating-point results may vary within numerical tolerance across platforms.\n"
            ))
            # Installed source checkouts can include their exact locked environment.
            root = Path(__file__).resolve().parents[2]
            for file in ["pyproject.toml", "uv.lock"]:
                if (root / file).is_file():
                    archive.write(root / file, "environment/" + file)
        if hasattr(data, "assert_unchanged"):
            data.assert_unchanged()
        return output if destination is not None else output.getvalue()


    def console_history(self) -> list[dict]:
        with self._console_lock:
            path = self.console_path / "history.json"
            if not path.is_file():
                return []
            try:
                return json.loads(path.read_text(encoding="utf-8")).get("history", [])
            except (OSError, json.JSONDecodeError) as exc:
                raise DataError("The saved Python history cannot be read.", "CORRUPT_RECORD") from exc

    def append_console_history(self, execution: dict) -> None:
        with self._console_lock:
            history = self.console_history()
            history.append(execution)
            # Bound disk/API size as well as count; oldest runs expire first.
            history = _bounded_history(history)
            _atomic_json(self.console_path / "history.json", {"history": history})
            if (self.console_path / "network-plots").exists():
                from .plot_artifacts import collect_plot_garbage
                try:
                    collect_plot_garbage(self.path, history=history)
                except (DataError, OSError):
                    # Cache cleanup is conservative. A refusal preserves files
                    # and must not turn an already persisted run into a failure.
                    pass

    def console_script(self) -> dict:
        with self._console_lock:
            self._recover_script_transaction()
            return self._legacy_script()

    def save_console_script(self, code: str, name: str = "analysis.py") -> dict:
        if len(code) > 64000 or name != "analysis.py":
            raise DataError("The editor stores analysis.py with at most 64,000 characters.", "INVALID_SCRIPT")
        with self._console_lock:
            self._recover_script_transaction()
            index, main = self._script_index()
            self._save_indexed_script(index, {**main, "code": self._script_value(scripts.script_code, code),
                                             "version": main["version"] + 1})
        return {"name": "analysis.py", "code": code}

    @staticmethod
    def _script_value(validate, *values, **options):
        try:
            return validate(*values, **options)
        except scripts.ScriptValidationError as exc:
            raise DataError(str(exc), exc.code) from exc

    def _script_path(self, name: str) -> Path:
        if (self.console_path.is_symlink() or not self.console_path.is_dir()
                or self.console_path.resolve().parent != self.path):
            raise DataError("The editor storage is unsafe.", "CORRUPT_RECORD")
        path = self.console_path / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise DataError("A linked editor file cannot be used.", "CORRUPT_RECORD")
        return path

    def _script_json(self, name: str, default):
        path = self._script_path(name)
        if not path.exists():
            return default
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("Editor record too large.")
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError("The saved source document cannot be read.", "CORRUPT_RECORD") from exc

    def _legacy_script(self) -> dict:
        document = self._script_json("script.json", {"name": "analysis.py", "code": None})
        if (not isinstance(document, dict) or set(document) != {"name", "code"}
                or document["name"] != "analysis.py"
                or (document["code"] is not None and not isinstance(document["code"], str))):
            raise DataError("The saved source document cannot be read.", "CORRUPT_RECORD")
        if document["code"] is not None:
            self._script_value(scripts.script_code, document["code"])
        return document

    @staticmethod
    def _script_hash(code: str | None) -> str:
        return hashlib.sha256(json.dumps(code, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _validate_script_index(self, index: object) -> dict:
        try:
            if (not isinstance(index, dict) or set(index) != {"schema", "analysis", "scripts"}
                    or type(index["schema"]) is not int or index["schema"] != 1):
                raise ValueError("Invalid index.")
            main = index["analysis"]
            if (not isinstance(main, dict) or set(main) != {"version", "hash", "size"}
                    or not isinstance(main["hash"], str) or len(main["hash"]) != 64
                    or any(c not in "0123456789abcdef" for c in main["hash"])
                    or type(main["size"]) is not int
                    or not 0 <= main["size"] <= scripts.MAX_SCRIPT_CHARACTERS * 4):
                raise ValueError("Invalid main metadata.")
            scripts.script_version(main["version"])
            rows = scripts.validate_script_rows(index["scripts"])
            if main["size"] + sum(row["size"] for row in rows) > scripts.MAX_SCRIPT_BYTES:
                raise ValueError("Editor index too large.")
            return {"schema": 1, "analysis": dict(main), "scripts": rows}
        except (ValueError, TypeError, KeyError) as exc:
            raise DataError("The saved source file list cannot be read.", "CORRUPT_RECORD") from exc

    def _script_index(self) -> tuple[dict, dict]:
        legacy = self._legacy_script()
        code_hash = self._script_hash(legacy["code"])
        size = len((legacy["code"] or "").encode("utf-8"))
        index = self._script_json("scripts.json", None)
        if index is None:
            index = {"schema": 1, "analysis": {"version": 0, "hash": code_hash, "size": size},
                     "scripts": []}
        else:
            index = self._validate_script_index(index)
            if index["analysis"]["hash"] != code_hash:
                # A pre-upgrade writer or interrupted legacy save may have changed
                # script.json. Keep that exact code and advance its local version.
                index["analysis"]["version"] += 1
                index["analysis"].update(hash=code_hash, size=size)
                self._script_value(scripts.script_version, index["analysis"]["version"])
        return index, {"id": "analysis", **legacy, "version": index["analysis"]["version"]}

    def _indexed_script(self, index: dict, identifier: str, main: dict) -> dict:
        if identifier == "analysis":
            return main
        row = next((row for row in index["scripts"] if row["id"] == identifier), None)
        if row is None:
            raise DataError("The source file was not found.", "NOT_FOUND")
        document = self._script_json(f"{identifier}.json", None)
        try:
            if (not isinstance(document, dict) or set(document) != {"id", "name", "code", "version"}
                    or scripts.metadata(document) != scripts.metadata(row)
                    or len(scripts.script_code(document["code"]).encode("utf-8")) != row["size"]):
                raise ValueError("Invalid editor record.")
        except (ValueError, TypeError, KeyError) as exc:
            raise DataError("The saved source document cannot be read.", "CORRUPT_RECORD") from exc
        return document

    def _write_script_transaction(self, document: dict, index: dict) -> None:
        name = "script.json" if document["id"] == "analysis" else f'{document["id"]}.json'
        target, index_path = self._script_path(name), self._script_path("scripts.json")
        stored = ({"name": "analysis.py", "code": document["code"]}
                  if document["id"] == "analysis" else document)
        _atomic_json(target, stored)
        _atomic_json(index_path, index)

    def _recover_script_transaction(self) -> None:
        pending = self._script_json("scripts-pending.json", None)
        if pending is None:
            return
        try:
            if not isinstance(pending, dict) or set(pending) != {"document", "index"}:
                raise ValueError("Invalid editor transaction.")
            document = pending["document"]
            if not isinstance(document, dict) or set(document) != {"id", "name", "code", "version"}:
                raise ValueError("Invalid editor document.")
            scripts.script_id(document["id"])
            scripts.script_name(document["name"])
            scripts.script_code(document["code"])
            scripts.script_version(document["version"])
            index = self._validate_script_index(pending["index"])
            if document["id"] == "analysis":
                if (document["name"] != "analysis.py"
                        or index["analysis"]["version"] != document["version"]
                        or index["analysis"]["hash"] != self._script_hash(document["code"])
                        or index["analysis"]["size"] != len(document["code"].encode("utf-8"))):
                    raise ValueError("Invalid main transaction.")
            else:
                row = next((row for row in index["scripts"] if row["id"] == document["id"]), None)
                if (row is None or scripts.metadata(row) != scripts.metadata(document)
                        or row["size"] != len(document["code"].encode("utf-8"))):
                    raise ValueError("Invalid file transaction.")
        except (ValueError, TypeError, KeyError) as exc:
            raise DataError("The pending source file save cannot be read.", "CORRUPT_RECORD") from exc
        self._write_script_transaction(document, index)
        self._script_path("scripts-pending.json").unlink()

    def _save_indexed_script(self, index: dict, document: dict) -> dict:
        self._script_value(scripts.script_version, document["version"])
        size = len(document["code"].encode("utf-8"))
        if document["id"] == "analysis":
            index["analysis"] = {"version": document["version"],
                                 "hash": self._script_hash(document["code"]), "size": size}
        else:
            row = {**scripts.metadata(document), "size": size}
            existing = next((i for i, item in enumerate(index["scripts"])
                             if item["id"] == document["id"]), None)
            if existing is None:
                index["scripts"].append(row)
            else:
                index["scripts"][existing] = row
        if (len(index["scripts"]) >= scripts.MAX_SCRIPTS
                or index["analysis"]["size"] + sum(row["size"] for row in index["scripts"])
                > scripts.MAX_SCRIPT_BYTES):
            raise DataError("The project's source file storage limit was reached.", "SCRIPT_LIMIT")
        # A small redo journal makes the document and metadata index recoverable
        # as one operation if the process stops between their atomic replacements.
        target = "script.json" if document["id"] == "analysis" else f'{document["id"]}.json'
        self._script_path(target)
        self._script_path("scripts.json")
        journal = self._script_path("scripts-pending.json")
        _atomic_json(journal, {"document": document, "index": index})
        self._write_script_transaction(document, index)
        journal.unlink()
        return dict(document)

    def console_scripts(self) -> dict:
        with self._console_lock:
            self._recover_script_transaction()
            index, main = self._script_index()
            return {"scripts": [scripts.metadata(main),
                                *(scripts.metadata(row) for row in index["scripts"])]}

    def _file_inventory(self, known_scripts: list[dict] | None = None) -> list[dict]:
        if known_scripts is not None and (not isinstance(known_scripts, list) or len(known_scripts) > scripts.MAX_SCRIPTS):
            raise DataError("The file catalog is invalid.", "INVALID_FILE_LAYOUT")
        current = self.console_scripts()["scripts"]
        present = {row["id"] for row in current}
        for row in known_scripts or []:
            if not isinstance(row, dict) or "id" not in row or "name" not in row:
                raise DataError("The file catalog is invalid.", "INVALID_FILE_LAYOUT")
            identifier = self._script_value(scripts.script_id, row["id"])
            name = self._script_value(scripts.script_name, row["name"])
            if identifier not in present:
                current.append({"id": identifier, "name": name})
                present.add(identifier)
        if len(current) > scripts.MAX_SCRIPTS:
            raise DataError("The project file limit has been reached.", "SCRIPT_LIMIT")
        return file_layout.inventory(current, self.list_datasets())

    @staticmethod
    def _layout_value(operation, *args):
        try:
            return operation(*args)
        except file_layout.FileLayoutError as exc:
            raise DataError(str(exc), exc.code) from exc

    def get_file_layout(self, *, known_scripts: list[dict] | None = None) -> dict:
        with self._console_lock:
            path = self._script_path("file-layout.json")
            with file_layout.storage_lock(self.console_path):
                stored = self._layout_value(file_layout.read, path)
                return self._layout_value(file_layout.reconcile, stored, self._file_inventory(known_scripts))

    def put_file_layout(self, document: dict, *, known_scripts: list[dict] | None = None) -> dict:
        """Compare-and-swap display metadata without renaming code or snapshots."""
        validated = self._layout_value(file_layout.validate, document)
        with self._console_lock:
            path = self._script_path("file-layout.json")
            with file_layout.storage_lock(self.console_path):
                stored = self._layout_value(file_layout.read, path)
                previous_version = stored["version"] if stored is not None else 0
                if validated["version"] != previous_version:
                    raise DataError("The file layout was changed elsewhere. Reload the current list.", "VERSION_CONFLICT")
                self._layout_value(file_layout.validate_inventory, validated, self._file_inventory(known_scripts))
                result = {"version": self._layout_value(file_layout.version, previous_version + 1),
                          "entries": validated["entries"]}
                _atomic_json(path, result)
                return result

    def named_console_script(self, script_id: str) -> dict:
        identifier = self._script_value(scripts.script_id, script_id)
        with self._console_lock:
            self._recover_script_transaction()
            index, main = self._script_index()
            return self._indexed_script(index, identifier, main)

    def create_console_script(self, name: str = "untitled.py", code: str = "", *,
                              script_id: str | None = None) -> dict:
        name = self._script_value(scripts.script_name, name)
        code = self._script_value(scripts.script_code, code)
        identifier = (uuid4().hex if script_id is None
                      else self._script_value(scripts.script_id, script_id, creating=True))
        with self._console_lock:
            self._recover_script_transaction()
            index, main = self._script_index()
            if any(row["id"] == identifier for row in index["scripts"]):
                existing = self._indexed_script(index, identifier, main)
                if existing["name"] == name and existing["code"] == code:
                    return existing
                raise DataError("The source file identifier already exists.", "SCRIPT_ID_CONFLICT")
            if self._script_path(f"{identifier}.json").exists():
                raise DataError("An unindexed source file already exists.", "CORRUPT_RECORD")
            name = self._script_value(scripts.unique_script_name, name,
                                      ["analysis.py", *(row["name"] for row in index["scripts"])])
            return self._save_indexed_script(index, {"id": identifier, "name": name,
                                                     "code": code, "version": 0})

    def save_named_console_script(self, script_id: str, code: str, version: int) -> dict:
        identifier = self._script_value(scripts.script_id, script_id)
        code, version = (self._script_value(scripts.script_code, code),
                         self._script_value(scripts.script_version, version))
        with self._console_lock:
            self._recover_script_transaction()
            index, main = self._script_index()
            previous = self._indexed_script(index, identifier, main)
            if previous["version"] != version:
                raise DataError("The source file was changed by another writer.", "VERSION_CONFLICT")
            return self._save_indexed_script(index, {**previous, "code": code, "version": version + 1})

    def cache_console_script(self, script_id: str, name: str, code: str, version: int) -> dict:
        """Desktop-only exact-name import; never execute or change cloud versions."""
        identifier = self._script_value(scripts.script_id, script_id)
        name, code, version = (self._script_value(scripts.script_name, name),
                               self._script_value(scripts.script_code, code),
                               self._script_value(scripts.script_version, version))
        with self._console_lock:
            self._recover_script_transaction()
            index, main = self._script_index()
            row = next((row for row in index["scripts"] if row["id"] == identifier), None)
            if identifier == "analysis" or row is not None:
                previous = self._indexed_script(index, identifier, main)
                if previous["version"] != version:
                    raise DataError("The source file was changed by another writer.", "VERSION_CONFLICT")
                if previous["name"] == name and previous["code"] == code:
                    return dict(previous)
                next_version = version + 1
            else:
                if version != 0 or self._script_path(f"{identifier}.json").exists():
                    raise DataError("The source file identifier already exists.", "SCRIPT_ID_CONFLICT")
                next_version = 0
            others = ["analysis.py", *(item["name"] for item in index["scripts"]
                                       if item["id"] != identifier)]
            if ((identifier == "analysis" and name != "analysis.py")
                    or (identifier != "analysis" and name.casefold() in
                        {item.casefold() for item in others})):
                raise DataError("Another source file uses this name.", "SCRIPT_NAME_CONFLICT")
            return self._save_indexed_script(index, {"id": identifier, "name": name,
                                                     "code": code, "version": next_version})
