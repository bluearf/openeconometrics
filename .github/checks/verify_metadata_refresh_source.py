"""Bind the documentation-only alpha refresh to its published runtime source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


BASELINE_COMMIT = "d11ed130feb4ad8e6e990e61bb0bd8a9101cb981"
SCOPES = ("src/openecon", "packages/openecon-charts/src/openecon_charts", "web")
VERSION_FILES = {
    "src/openecon/__init__.py": ("0.3.20a1", "0.3.20a2"),
    "packages/openecon-charts/src/openecon_charts/__init__.py": ("0.3.2a1", "0.3.2a2"),
}
# These are generated outputs, rather than baseline source. The package-pair
# verifier separately binds the built frontend and every shipped wheel file.
GENERATED_DIRECTORIES = ("src/openecon/static", "web/node_modules", "web/dist")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input_bytes, capture_output=True,
        check=True, timeout=60,
    ).stdout


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _read_baseline(root: Path, commit: str = BASELINE_COMMIT) -> dict:
    require(commit == BASELINE_COMMIT, "Published runtime baseline pin differs")
    resolved = _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    require(resolved == BASELINE_COMMIT, "Published runtime baseline could not be resolved exactly")
    tree = _git(root, "ls-tree", "-r", "-z", commit, "--", *SCOPES)
    records = []
    for record in tree.split(b"\0"):
        if not record:
            continue
        fields, raw_path = record.split(b"\t", 1)
        mode, kind, oid = fields.decode("ascii").split()
        path = raw_path.decode("utf-8")
        require(kind == "blob" and mode in {"100644", "100755"},
                f"Nonregular published runtime source: {path}")
        records.append((path, mode, oid))
    require(bool(records), "Published runtime baseline is empty")
    objects = io.BytesIO(_git(
        root, "cat-file", "--batch",
        input_bytes="".join(f"{oid}\n" for _, _, oid in records).encode("ascii"),
    ))
    result = {}
    for path, mode, oid in records:
        header = objects.readline().decode("ascii").rstrip("\n").split()
        require(len(header) == 3 and header[:2] == [oid, "blob"],
                f"Missing published runtime blob: {path}")
        data = objects.read(int(header[2]))
        require(objects.read(1) == b"\n" and _blob_id(data) == oid,
                f"Published runtime blob bytes differ: {path}")
        require(path not in result, f"Duplicate published runtime source: {path}")
        result[path] = (mode, data)
    require(not objects.read(), "Unexpected published runtime object data")
    return result


def _generated(path: str) -> bool:
    return "__pycache__" in PurePosixPath(path).parts or any(
        path == prefix or path.startswith(prefix + "/") for prefix in GENERATED_DIRECTORIES
    )


def _read_current(root: Path) -> dict:
    result = {}
    for scope in SCOPES:
        start = root / scope
        require(start.is_dir() and not start.is_symlink(), f"Missing/linked runtime scope: {scope}")
        for directory, names, files in os.walk(start, followlinks=False):
            parent = Path(directory)
            for name in names[:]:
                path = parent / name
                relative = path.relative_to(root).as_posix()
                if _generated(relative):
                    names.remove(name)
                else:
                    require(not path.is_symlink(), f"Linked runtime directory: {relative}")
            for name in files:
                path = parent / name
                relative = path.relative_to(root).as_posix()
                if _generated(relative):
                    continue
                require(path.is_file() and not path.is_symlink(),
                        f"Nonregular runtime source: {relative}")
                mode = "100755" if path.stat().st_mode & 0o111 else "100644"
                result[relative] = (mode, path.read_bytes())
    return result


def _inventory_digest(files: dict) -> str:
    rows = [{"path": path, "mode": mode, "sha256": hashlib.sha256(data).hexdigest()}
            for path, (mode, data) in sorted(files.items())]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def compare_sources(baseline: dict, current: dict) -> dict:
    require(set(VERSION_FILES) <= set(baseline), "Published package version files are missing")
    require(set(current) == set(baseline),
            "Runtime source paths differ: "
            f"added={sorted(set(current) - set(baseline))}, "
            f"removed={sorted(set(baseline) - set(current))}")
    normalized = {}
    versions = []
    for path, (mode, original) in sorted(baseline.items()):
        actual_mode, actual = current[path]
        require(mode == actual_mode, f"Runtime source file mode differs: {path}")
        expected = original
        if path in VERSION_FILES:
            before, after = VERSION_FILES[path]
            old = f'__version__ = "{before}"'.encode()
            new = f'__version__ = "{after}"'.encode()
            require(original.count(old) == 1, f"Published version literal differs: {path}")
            expected = original.replace(old, new, 1)
            versions.append({"path": path, "before": before, "after": after,
                             "before_sha256": hashlib.sha256(original).hexdigest(),
                             "after_sha256": hashlib.sha256(expected).hexdigest()})
        require(actual == expected, f"Runtime source bytes differ beyond version literal: {path}")
        normalized[path] = (actual_mode, original)
    baseline_digest = _inventory_digest(baseline)
    normalized_digest = _inventory_digest(normalized)
    require(baseline_digest == normalized_digest, "Normalized runtime source inventory differs")
    return {
        "runtime_source_files": len(baseline),
        "unchanged_runtime_source_files": len(baseline) - len(VERSION_FILES),
        "version_only_files": versions,
        "baseline_inventory_sha256": baseline_digest,
        "normalized_current_inventory_sha256": normalized_digest,
    }


def verify(root: Path) -> dict:
    result = compare_sources(_read_baseline(root), _read_current(root))
    checkout = _git(root, "rev-parse", "HEAD").decode().strip()
    ci = {"event": os.environ.get("GITHUB_EVENT_NAME"),
          "run_id": os.environ.get("GITHUB_RUN_ID"),
          "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}
    if any(value is not None for value in ci.values()):
        require(all(isinstance(value, str) and value for value in ci.values()),
                "Incomplete metadata-refresh CI identity")
        require(os.environ.get("GITHUB_SHA") == checkout, "CI checkout commit differs")
    return {
        "schema_version": 1,
        "status": "passed_metadata_only_runtime_identity",
        "baseline_repository": "bluearf/openeconometrics",
        "baseline_public_commit": BASELINE_COMMIT,
        "public_checkout_commit": checkout,
        **ci,
        "scopes": list(SCOPES),
        "excluded_generated_directories": list(GENERATED_DIRECTORIES),
        "generated_frontend": "separately bound by verify_public_package_pair.py",
        "new_full_scientific_gate_claimed": False,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
