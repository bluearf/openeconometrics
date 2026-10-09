"""Bind a fresh installation to the already verified four-archive provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def regular_path(path: Path, prefix: Path) -> Path:
    """Refuse symlink files and parent directories before resolving an origin."""
    path = path.absolute()
    require(".." not in path.parts, f"Installed path contains parent traversal: {path}")
    canonical_prefix = prefix.resolve()
    anchor = prefix if path.is_relative_to(prefix) else canonical_prefix
    require(path.is_relative_to(anchor), f"Installed path is outside the venv: {path}")
    current = anchor
    for part in path.relative_to(anchor).parts:
        current /= part
        require(not current.is_symlink(), f"Installed path is a symlink: {current}")
    canonical = path.resolve()
    require(canonical.is_relative_to(canonical_prefix),
            f"Resolved installed path is outside the venv: {path}")
    return canonical


def check_paths(workspace: Path, prefix: Path) -> None:
    allowed = (prefix.resolve(), Path(sys.base_prefix).resolve(), Path(sys.base_exec_prefix).resolve())
    require(all(sys.path), "Isolated interpreter retained a current-directory path")
    for value in sys.path:
        resolved = Path(value).resolve()
        require(
            not resolved.is_relative_to(workspace),
            f"Source workspace entered the installed interpreter: {value}",
        )
        require(any(resolved.is_relative_to(root) for root in allowed),
                f"Foreign interpreter search path outside the venv/runtime: {value}")
    for name, module in list(sys.modules.items()):
        # The explicit verification script is executed, not imported as a package.
        if name == "__main__":
            continue
        require(not name.startswith("__editable__"), "Editable import hook is resident")
        origin = getattr(module, "__file__", None)
        if origin:
            resolved = Path(origin).resolve()
            require(
                not resolved.is_relative_to(workspace),
                f"Source workspace module entered the installed interpreter: {name}",
            )
            require(any(resolved.is_relative_to(root) for root in allowed),
                    f"Foreign module outside the venv/runtime: {name}")


def verify_child(request: dict) -> dict:
    require(
        sys.flags.isolated == 1 and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1,
        "Installation verification requires an isolated interpreter",
    )
    prefix = Path(sys.prefix).absolute()
    require(not prefix.is_symlink(), "Installed venv prefix is a symlink")
    require(
        prefix.resolve() == Path(request["prefix"]).resolve()
        and prefix.resolve() != Path(sys.base_prefix).resolve(),
        "Installed interpreter has a foreign prefix or is not a fresh venv",
    )
    workspace = Path(request["workspace"]).resolve()
    check_paths(workspace, prefix)
    binding = request["binding"]
    require(binding["status"] == "passed", "Archive provenance did not pass")
    require(
        re.fullmatch(r"[0-9a-f]{40}", binding["source_commit"]) is not None,
        "Archive provenance has no frozen source commit",
    )
    packages = {}
    for name in request["packages"]:
        module_name = name.replace("-", "_")
        expected = binding["production_files"][name]
        require(isinstance(expected, dict) and bool(expected), "Empty production inventory")
        for relative, value in expected.items():
            path = PurePosixPath(relative)
            require(
                path.as_posix() == relative and not path.is_absolute()
                and ".." not in path.parts and path.parts[0] == module_name
                and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                f"Invalid production inventory entry: {relative}",
            )
        distribution = metadata.distribution(name)
        require(
            distribution.version == binding["versions"][name],
            f"Installed distribution version differs: {name}",
        )
        files = list(distribution.files or ())
        require(bool(files), f"Installed distribution has no file record: {name}")
        require(
            not any(str(path).endswith(".pth") or "__editable__" in str(path) for path in files),
            f"Installed distribution contains an editable path hook: {name}",
        )
        metadata_files = [path for path in files if str(path).endswith(".dist-info/METADATA")]
        require(len(metadata_files) == 1, f"Installed metadata is not unique: {name}")
        require(
            regular_path(Path(distribution.locate_file(metadata_files[0])), prefix).is_file(),
            f"Installed metadata is missing: {name}",
        )
        direct = distribution.read_text("direct_url.json")
        if direct:
            require(
                not json.loads(direct).get("dir_info", {}).get("editable", False),
                f"Installed distribution is editable: {name}",
            )
        spec = importlib.util.find_spec(module_name)
        require(spec is not None and spec.origin is not None, f"Installed package is missing: {name}")
        origin = regular_path(Path(spec.origin), prefix)
        located = regular_path(Path(distribution.locate_file(module_name + "/__init__.py")), prefix)
        require(origin == located, f"Installed metadata and package origins differ: {name}")
        root = regular_path(origin.parent, prefix)
        require(root.is_dir(), f"Installed package directory is missing: {name}")
        paths = {}
        for directory, subdirectories, filenames in os.walk(root, followlinks=False):
            directory = regular_path(Path(directory), prefix)
            for child in subdirectories:
                regular_path(directory / child, prefix)
            for filename in filenames:
                path = regular_path(directory / filename, prefix)
                require(path.is_file(), f"Nonregular installed production file: {path}")
                relative = module_name + "/" + path.relative_to(root).as_posix()
                if path.suffix == ".pyc":
                    relative_path = PurePosixPath(relative)
                    if relative_path.parent.name == "__pycache__":
                        match = re.fullmatch(r"(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc", path.name)
                        require(match is not None, f"Unexpected installed bytecode cache: {relative}")
                        source = str(relative_path.parent.parent / (match[1] + ".py"))
                    else:
                        source = str(relative_path.with_suffix(".py"))
                    require(source in expected, f"Unbound installed bytecode cache: {relative}")
                    continue
                require("__pycache__" not in path.relative_to(root).parts,
                        f"Unexpected non-bytecode cache file: {relative}")
                paths[relative] = path
        require(
            set(paths) == set(expected),
            f"Installed production file set differs: {name}; "
            f"missing={sorted(set(expected) - set(paths))}; extra={sorted(set(paths) - set(expected))}",
        )
        actual = {relative: hashlib.sha256(path.read_bytes()).hexdigest()
                  for relative, path in sorted(paths.items())}
        require(actual == expected, f"Installed production bytes differ: {name}")
        packages[name] = {"version": distribution.version, "root": str(root),
                          "production_files": actual, "imported_modules": {}}

    # Inspect every module actually imported by the verified package roots. Lazy
    # scientific imports remain for the subsequent installed numerical checks.
    for name in packages:
        module = importlib.import_module(name.replace("-", "_"))
        require(
            module.__version__ == packages[name]["version"],
            f"Imported package version differs: {name}",
        )
    check_paths(workspace, prefix)
    for name, package in packages.items():
        module_name = name.replace("-", "_")
        root = Path(package["root"])
        for imported, module in list(sys.modules.items()):
            if imported != module_name and not imported.startswith(module_name + "."):
                continue
            origin = getattr(module, "__file__", None)
            spec = getattr(module, "__spec__", None)
            require(origin and spec and spec.origin, f"Unbound installed module origin: {imported}")
            path = regular_path(Path(origin), prefix)
            require(path.is_relative_to(root), f"Foreign installed module origin: {imported}")
            require(
                regular_path(Path(spec.origin), prefix) == path,
                f"Installed module origin and loader differ: {imported}",
            )
            relative = module_name + "/" + path.relative_to(root).as_posix()
            require(
                relative in package["production_files"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == package["production_files"][relative],
                f"Imported module was not bound to verified source bytes: {imported}",
            )
            for search in getattr(module, "__path__", ()):
                require(
                    regular_path(Path(search), prefix).is_relative_to(root),
                    f"Foreign installed package search path: {imported}",
                )
            package["imported_modules"][imported] = relative
    return {"status": "passed", "source_commit": binding["source_commit"],
            "source_files_sha256": binding["source_files_sha256"],
            "provenance_sha256": request["provenance_sha256"],
            "python": sys.version, "prefix": str(prefix), "isolated": True,
            "workspace_paths_absent": True, "editable_installations_absent": True,
            "packages": packages,
            "scope": "Installed inventory and imported origins; numerical acceptance runs separately."}


def verify(python: Path, provenance: Path, packages: list[str], workspace: Path) -> dict:
    raw = provenance.read_bytes()
    request = {"binding": json.loads(raw), "provenance_sha256": hashlib.sha256(raw).hexdigest(),
               "packages": packages, "prefix": str(python.absolute().parents[1]),
               "workspace": str(workspace.resolve())}
    with tempfile.TemporaryDirectory(prefix="openecon-installed-inventory-") as directory:
        result = subprocess.run(
            [str(python.absolute()), "-I", str(Path(__file__).absolute()), "--child"],
            input=json.dumps(request), cwd=directory, text=True, capture_output=True, timeout=60,
        )
    require(result.returncode == 0, f"Installed inventory verification failed: {result.stderr[-8000:]}")
    return json.loads(result.stdout)


def main() -> None:
    if sys.argv[1:] == ["--child"]:
        print(json.dumps(verify_child(json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--package", action="append", choices=("openecon", "openecon-charts"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(len(args.package) == len(set(args.package)), "Duplicate package verification request")
    record = verify(args.python, args.provenance, args.package, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "source_commit": record["source_commit"],
                      "packages": {name: len(value["production_files"])
                                   for name, value in record["packages"].items()}}))


if __name__ == "__main__":
    main()
