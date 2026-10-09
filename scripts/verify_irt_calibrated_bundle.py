"""Read-only source and frontend identity receipt for the owned irt QA app.

This does not launch the app, open a profile, exercise methods, or prove native UI
execution. Python equality is limited to the explicit modules recorded below;
frontend equality covers the complete locally built static tree, not Git sources.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import sys
from types import CodeType

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDENTIFIER = "org.openecon.qa.irt-calibrated-20261007"
EXPECTED_MINIMUM_MACOS = "26.0.0"
IRT_PREFIX = "src/openecon/econometrics/irt/"

# Deliberately scoped to the irt interface, numerical/sample/table
# helpers, and console/desktop persistence path. This is not a transitive audit
# of every imported package, other econometric families, or third-party code.
SUPPORT_SOURCES = (
    "src/openecon/__init__.py",
    "src/openecon/analysis.py",
    "src/openecon/analysis_contracts.py",
    "src/openecon/resources.py",
    "src/openecon/data.py",
    "src/openecon/dataset.py",
    "src/openecon/frame.py",
    "src/openecon/models.py",
    "src/openecon/latex.py",
    "src/openecon/output_latex.py",
    "src/openecon/output_events.py",
    "src/openecon/engines/__init__.py",
    "src/openecon/engines/contracts.py",
    "src/openecon/engines/distributions.py",
    "src/openecon/engines/inference.py",
    "src/openecon/engines/torch_engine.py",
    "src/openecon/econometrics/__init__.py",
    "src/openecon/econometrics/registry.py",
    "src/openecon/econometrics/core.py",
    "src/openecon/econometrics/resident_cpu.py",
    "src/openecon/econometrics/summary_state.py",
    "src/openecon/optional_dependencies.py",
    "src/openecon/console.py",
    "src/openecon/console_worker.py",
    "src/openecon/script_packages.py",
    "src/openecon/project_packages.py",
    "src/openecon/script_contracts.py",
    "src/openecon/file_layout.py",
    "src/openecon/workspace.py",
    "src/openecon/server.py",
    "src/openecon/desktop_runtime.py",
    "src/openecon/desktop_sharing.py",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(code: CodeType) -> CodeType:
    """Ignore only build paths, including nested functions and comprehensions."""
    if not isinstance(code, CodeType):
        raise RuntimeError("PYZ entry is not a Python code object")
    return code.replace(
        co_filename="<pinned-bundled-source>",
        co_consts=tuple(
            normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts
        ),
    )


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def module_name(source: str) -> str:
    name = ".".join(Path(source).relative_to("src").with_suffix("").parts)
    return name.removesuffix(".__init__")


def regular_tree(root: Path) -> tuple[dict[str, Path], list[str]]:
    """Inventory every file and directory without silently following symlinks."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Expected a regular frontend directory: {root}")
    files, directories = {}, []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Frontend symlinks are outside this verifier's scope: {path}")
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files[relative] = path
        else:
            raise RuntimeError(f"Unexpected frontend entry: {path}")
    if "index.html" not in files:
        raise RuntimeError(f"Built frontend lacks index.html: {root}")
    return files, directories


def frontend_identity(reference: Path, installed: Path) -> dict:
    expected, expected_directories = regular_tree(reference)
    actual, actual_directories = regular_tree(installed)
    if expected.keys() != actual.keys() or expected_directories != actual_directories:
        raise RuntimeError(
            "Frontend trees differ: "
            + json.dumps(
                {
                    "missing_files": sorted(expected.keys() - actual.keys()),
                    "extra_files": sorted(actual.keys() - expected.keys()),
                    "missing_directories": sorted(
                        set(expected_directories) - set(actual_directories)
                    ),
                    "extra_directories": sorted(
                        set(actual_directories) - set(expected_directories)
                    ),
                }
            )
        )
    records = []
    for relative, path in expected.items():
        source_bytes, installed_bytes = path.read_bytes(), actual[relative].read_bytes()
        if source_bytes != installed_bytes:
            raise RuntimeError(f"Installed frontend bytes differ: {relative}")
        records.append(
            {
                "file": relative,
                "bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    manifest = {"directories": expected_directories, "files": records}
    return {
        "reference": str(reference),
        "installed": str(installed),
        "reference_kind": "locally built working-tree static artifacts, not pinned Git source",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **manifest,
    }


def verify(app: Path, source_ref: str) -> dict:
    from PyInstaller.archive.readers import CArchiveReader

    app = app.resolve(strict=True)
    if not app.is_dir() or app.suffix != ".app":
        raise RuntimeError("--app must identify an installed .app directory")
    commit = git("rev-parse", "--verify", "--end-of-options", f"{source_ref}^{{commit}}")
    commit = commit.decode().strip()
    irt_sources = git("ls-tree", "-r", "--name-only", commit, "--", IRT_PREFIX)
    irt_sources = sorted(
        path for path in irt_sources.decode().splitlines() if path.endswith(".py")
    )
    if IRT_PREFIX + "__init__.py" not in irt_sources:
        raise RuntimeError("Pinned source lacks the irt manifest")
    source_paths = sorted(set(irt_sources) | set(SUPPORT_SOURCES))
    plist_path = app / "Contents/Info.plist"
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    frontend = runtime.parent / "_internal/openecon/static"
    for path in (plist_path, runtime, frontend):
        if not path.resolve(strict=True).is_relative_to(app):
            raise RuntimeError(f"Installed bundle entry resolves outside the app: {path}")
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    if plist.get("CFBundleIdentifier") != EXPECTED_IDENTIFIER:
        raise RuntimeError(f"Unexpected QA bundle identifier: {plist.get('CFBundleIdentifier')!r}")
    if plist.get("LSMinimumSystemVersion") != EXPECTED_MINIMUM_MACOS:
        raise RuntimeError(f"Unexpected macOS minimum: {plist.get('LSMinimumSystemVersion')!r}")
    runtime_hash, plist_hash = digest(runtime), digest(plist_path)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    checked = []
    for source in source_paths:
        module = module_name(source)
        if module not in archive.toc:
            raise RuntimeError(f"Required scoped module is absent from installed PYZ: {module}")
        source_bytes = git("show", f"{commit}:{source}")
        expected = compile(source_bytes, source, "exec", dont_inherit=True, optimize=0)
        if normalized(archive.extract(module)) != normalized(expected):
            raise RuntimeError(f"Installed Python code differs from pinned source: {module}")
        checked.append(
            {
                "module": module,
                "source": source,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "normalized_code_equal": True,
            }
        )
    assets = frontend_identity(ROOT / "src/openecon/static", frontend)
    command = ["codesign", "--verify", "--deep", "--strict", str(app)]
    signature = subprocess.run(command, check=True, capture_output=True, text=True)
    # Reject a runtime, plist or frontend replacement during the read-only audit.
    if digest(runtime) != runtime_hash or digest(plist_path) != plist_hash:
        raise RuntimeError("Installed runtime or plist changed during verification")
    if assets != frontend_identity(ROOT / "src/openecon/static", frontend):
        raise RuntimeError("Frontend changed during verification")
    return {
        "schema": "openecon.irt.bundle-identity.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "app": str(app),
        "source_ref_requested": source_ref,
        "source_commit": commit,
        "python_verifier_version": sys.version,
        "verifier_sha256": digest(Path(__file__).resolve()),
        "bundle": {
            "identifier": plist["CFBundleIdentifier"],
            "minimum_macos": plist["LSMinimumSystemVersion"],
            "version": plist.get("CFBundleShortVersionString"),
            "info_plist_sha256": plist_hash,
            "runtime": str(runtime),
            "runtime_sha256": runtime_hash,
        },
        "scope": {
            "python": "all pinned irt .py modules plus the enumerated support modules",
            "code_comparison": "CodeType equality with recursively normalized co_filename only",
            "external_libraries_or_all_runtime_code_audited": False,
            "method_execution_verified": False,
            "native_ui_run_verified": False,
            "native_restart_or_saved_history_verified": False,
            "profile": (
                "Owned QA identity declared in Info.plist; no profile was opened. Actual compiled "
                "native data-root isolation must be verified separately during native delivery."
            ),
            "signing": "strict local bundle verification; no Developer ID or notarization claim",
        },
        "irt_sources": irt_sources,
        "checked_module_count": len(checked),
        "checked_modules": checked,
        "frontend": assets,
        "signature": {
            "command": command,
            "returncode": signature.returncode,
            "stdout": signature.stdout,
            "stderr": signature.stderr,
        },
        "checks": {
            "pinned_commit_resolved": True,
            "owned_bundle_identifier": True,
            "minimum_macos_26": True,
            "all_pinned_irt_modules_present_and_equal": True,
            "all_enumerated_support_modules_present_and_equal": True,
            "frontend_exact_tree_and_bytes": True,
            "codesign_deep_strict": True,
            "measured_inputs_unchanged_during_check": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    app = arguments.app.resolve(strict=True)
    output = arguments.output.resolve()
    if output.is_relative_to(app):
        parser.error("--output must be outside the app; this verifier never mutates the bundle")
    if output.exists():
        parser.error("--output already exists; use a fresh receipt path")
    receipt = verify(app, arguments.source_ref)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "receipt": str(output),
                "source_commit": receipt["source_commit"],
                "modules_matched": receipt["checked_module_count"],
                "frontend_files_matched": receipt["frontend"]["file_count"],
                "native_ui_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
