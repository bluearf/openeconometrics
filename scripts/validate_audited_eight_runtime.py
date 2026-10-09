"""Audited eight capability workflows, saved state and source/installed SDK proof.

Runtime execution forbids external estimator imports. Dependency discovery
with find_spec is permitted without executing those packages. Every currently
present OpenEcon source module is pinned, including untracked source files.
This script does not query or modify Git, trackers, apps or release state.
"""

from __future__ import annotations

import argparse
import builtins
from contextlib import redirect_stdout
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--receipt", required=True, type=Path)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--installed-root", type=Path)
parser.add_argument("--issues", type=int, nargs="+", default=None,
                    help="Explicit tracker identifiers supplied by the caller; no issue scope is inferred.")
args = parser.parse_args()
receipt_path = args.receipt.resolve()
installed_root = None if args.installed_root is None else args.installed_root.resolve()
package_root = ROOT/"src/openecon" if installed_root is None else installed_root/"openecon"
package_root = package_root.resolve()
blocked = {"scipy", "statsmodels", "sklearn", "linearmodels"}
assert not any(name.split(".")[0] in blocked for name in sys.modules), "Forbidden estimator was preloaded"
assert not any(name == "openecon" or name.startswith("openecon.") for name in sys.modules), "OpenEcon must be imported only after the guard"
attempts = []
discoveries = []
original_import = builtins.__import__
original_import_module = importlib.import_module
original_find_spec = importlib.util.find_spec


def deny(name, mechanism):
    attempts.append({"module": name, "mechanism": mechanism})
    raise AssertionError("External estimator runtime import forbidden: " + name)


def guarded_import(name, *a, **kw):
    if name.split(".")[0] in blocked:
        deny(name, "builtins.__import__")
    return original_import(name, *a, **kw)


def guarded_import_module(name, package=None):
    fullname = importlib.util.resolve_name(name, package) if name.startswith(".") else name
    if fullname.split(".")[0] in blocked:
        deny(fullname, "importlib.import_module")
    return original_import_module(name, package)


class DeniedLoader(importlib.abc.Loader):
    def __init__(self, fullname):
        self.fullname = fullname

    def create_module(self, spec):
        deny(self.fullname, "module loader create_module")

    def exec_module(self, module):
        deny(self.fullname, "module loader exec_module")


def protected_spec(spec):
    if spec is None:
        return None
    # Preserve availability, origin and package search paths for discovery.
    # Executing the discovered module through its loader remains forbidden.
    guarded = importlib.machinery.ModuleSpec(spec.name, DeniedLoader(spec.name),
        origin=spec.origin, is_package=spec.submodule_search_locations is not None)
    guarded.submodule_search_locations = spec.submodule_search_locations
    return guarded


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            return protected_spec(importlib.machinery.PathFinder.find_spec(fullname, path))
        return None


def discovery(name, package=None):
    fullname = importlib.util.resolve_name(name, package) if name.startswith(".") else name
    if fullname.split(".")[0] not in blocked:
        return original_find_spec(name, package)
    # Nested find_spec normally imports its parent. Walk the filesystem search
    # paths directly, so nested dependency discovery also executes no package.
    path, spec = None, None
    parts = fullname.split(".")
    for index in range(len(parts)):
        spec = importlib.machinery.PathFinder.find_spec(".".join(parts[:index+1]), path)
        if spec is None:
            break
        path = spec.submodule_search_locations
        if index < len(parts)-1 and path is None:
            spec = None
            break
    discoveries.append({"module": fullname, "available": spec is not None})
    return protected_spec(spec)


builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
importlib.util.find_spec = discovery
sys.meta_path.insert(0, Finder())
torch.set_num_threads(1)

# Positive discovery and negative import controls establish that the guard
# distinguishes inspection from execution. These deliberate denied controls
# are recorded separately from attempts during the analytical workflows.
guard_controls = []
for dependency in sorted(blocked):
    spec = importlib.util.find_spec(dependency)
    for operation in (lambda name=dependency: __import__(name),
                      lambda name=dependency: importlib.import_module(name)):
        try:
            operation()
        except AssertionError:
            pass
        else:
            raise AssertionError("Guard control did not refuse import: " + dependency)
    if spec is not None:
        try:
            spec.loader.create_module(spec)
        except AssertionError:
            pass
        else:
            raise AssertionError("Guard control did not refuse discovered loader: " + dependency)
    guard_controls.append({"dependency": dependency, "discovery_allowed": True,
                           "available": spec is not None, "actual_import_refused": True})
guard_control_denials = list(attempts)
attempts.clear()
assert not any(name.split(".")[0] in blocked for name in sys.modules)


def fingerprint_sources():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT/"src/openecon").rglob("*.py"))
        if "__pycache__" not in path.parts
    }


source_hashes = fingerprint_sources()
example_paths = [ROOT/"docs/examples"/name for name in (
    "audited_eight_timeseries.py", "audited_eight_survey_mi.py",
    "audited_eight_categorical_causal.py",
)]
example_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in example_paths}
validator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert source_hashes
if installed_root:
    for relative, expected in source_hashes.items():
        path = installed_root/relative.removeprefix("src/")
        assert path.is_file(), "Installed SDK is missing current source: " + relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Installed source differs: " + relative

import openecon as oe  # noqa: E402

assert Path(oe.__file__).resolve().is_relative_to(package_root), "OpenEcon import origin does not match requested source/installed SDK"


def strict_json(path):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(path.read_text(), parse_constant=nonfinite)


def run_example(filename, destination, callback=None):
    destination.mkdir(parents=True, exist_ok=True)
    prior = os.getcwd()
    stream = io.StringIO()
    try:
        os.chdir(destination)
        with redirect_stdout(stream):
            variables = runpy.run_path(str(ROOT/"docs/examples"/filename))
            returned = variables["run"](destination) if callback else None
    finally:
        os.chdir(prior)
    return variables, returned, stream.getvalue()


def restored_summary_file(path):
    raw = path.read_text()
    restored = oe.restore_summary(raw)
    assert oe.summary_state(restored) == raw, "Complete summary readback differs: " + str(path)
    assert len(restored.to_latex()) > 0
    return {name: len(frame) for name, frame in restored.items()}


def execute(destination):
    workflows = {}
    for filename in ("audited_eight_timeseries.py", "audited_eight_survey_mi.py",
                     "audited_eight_categorical_causal.py"):
        label = filename.removeprefix("audited_eight_").removesuffix(".py")
        _, result, stdout = run_example(filename, destination/label, callback=True)
        assert result, "Example must return its complete replay receipt: " + filename
        workflows[label] = {"receipt": result, "stdout": stdout}
    return workflows


temporary = None
if args.output_dir:
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
else:
    temporary = tempfile.TemporaryDirectory(prefix="openecon-audited-eight-native-")
    destination = Path(temporary.name)
try:
    workflows = execute(destination)
    assert fingerprint_sources() == source_hashes, "Source files changed during validation; rerun against one stable source snapshot"
    assert {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in example_paths} == example_hashes, "Example source changed during validation"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == validator_hash, "Validator changed during execution"
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name != "openecon" and not name.startswith("openecon."):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = Path(path).resolve()
        assert path.is_relative_to(package_root), "Mixed OpenEcon runtime origin: " + name
        relative = "src/openecon/"+str(path.relative_to(package_root))
        assert relative in source_hashes, "Loaded module is outside the pinned Python source set: " + name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hashes[relative], "Loaded module bytes differ: " + name
        origins[name] = {"relative_source": relative, "origin": str(path), "sha256": source_hashes[relative]}
    assert not attempts
    assert not any(name.split(".")[0] in blocked for name in sys.modules)
    files = {
        str(path.relative_to(destination)): {"bytes": path.stat().st_size,
                                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    receipt = {
        "schema": "openecon.audited-eight-runtime.v1",
        "issues": args.issues,
        "scope": "Native resident CPU workflows and complete saved-state replay from source or installed SDK; not licensed vendor execution, frozen binary or installed desktop validation",
        "execution": "source" if installed_root is None else "installed Python SDK",
        "package_origin": str(Path(oe.__file__).resolve()),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "pandas": pd.__version__, "openecon": getattr(oe, "__version__", None)},
        "all_current_source_modules_sha256": source_hashes,
        "example_sources_sha256": example_hashes, "validator_sha256": validator_hash,
        "source_module_count": len(source_hashes),
        "loaded_owned_module_origins": origins,
        "loaded_owned_module_count": len(origins),
        "installed_all_current_sources_equal": installed_root is not None,
        "guard": "builtins import plus importlib.import_module plus protected meta-path loaders; find_spec discovery allowed without executing package code",
        "blocked_libraries": sorted(blocked), "forbidden_import_attempts": attempts,
        "guard_controls": guard_controls, "deliberate_guard_control_denials": guard_control_denials,
        "allowed_dependency_discovery": discoveries,
        "workflows": workflows, "files": files,
        "output_directory": str(destination), "outputs_retained": args.output_dir is not None,
        "device": "cpu", "precision": "float64", "vendor_execution": False,
        "frozen_runtime_validation": False, "desktop_validation": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, allow_nan=False, sort_keys=True, indent=2)+"\n")
    print("AUDITED_EIGHT_RUNTIME_OK " + json.dumps({"execution": receipt["execution"],
        "workflows": len(workflows), "saved_files": len(files), "source_modules": len(source_hashes),
        "loaded_modules": len(origins), "forbidden_import_attempts": attempts,
        "complete_state_readback": True, "receipt": str(receipt_path)}, sort_keys=True))
finally:
    if temporary:
        temporary.cleanup()
