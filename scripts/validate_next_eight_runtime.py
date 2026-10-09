"""All next-eight native workflows, saved state and source/installed SDK proof.

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
    "next_eight_proxy_svar.py", "next_eight_irt_diagnostics.py",
    "smoothing_multivariate_targets.py", "next_eight_inference_stability.py",
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


def regularized_models(destination):
    destination.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(28391)
    x = torch.randn((150, 3), generator=generator, dtype=torch.float64)*torch.tensor([1, 20, .01], dtype=torch.float64)
    y = 2+x@torch.tensor([.8, -.03, 12], dtype=torch.float64)+.4*torch.randn(150, generator=generator, dtype=torch.float64)
    frame = pd.DataFrame(x.tolist(), columns=list("abc"))
    frame["y"] = y.tolist()
    training, holdout = frame.iloc[:120].copy(), frame.iloc[120:].copy()
    results = {}
    for name in ("ridge", "lasso", "elasticnet", "pls"):
        if name == "pls":
            options = {"selection": "cv", "component_path": [1, 2, 3], "folds": 4, "seed": 82}
        else:
            options = {"selection": "cv", "n_lambdas": 4, "lambda_ratio": .02,
                       "folds": 4, "seed": 82, "forced_controls": ["a"],
                       "penalty_factors": [1, .5, 2]}
            if name == "elasticnet":
                options["l1_ratio"] = .4
        model = getattr(oe, name)(data=training, y="y", x=list("abc"), **options)
        path = destination/(name+".json")
        path.write_text(model.model_dump_json())
        restored = oe.ResultBundle.model_validate_json(path.read_text())
        assert restored.model_dump(mode="json") == strict_json(path)
        before, after = oe.regularized_predict(model, holdout), oe.regularized_predict(restored, holdout)
        assert before.equals(after)
        changed_labels = holdout.assign(y=1e100)
        assert after.equals(oe.regularized_predict(restored, changed_labels))
        assert list(after.index) == list(holdout.index)
        assert model.coefficients == model.covariance_matrix == []
        assert model.inference["available"] is False
        state = model.extra["penalized_state"]
        if name != "pls":
            assert state["penalty_factors"] == [0, .5, 2]
            assert state["forced_controls"] == ["a"]
            assert len(state["fold_path_diagnostics"]) == 4
        (destination/(name+".tex")).write_text(str(oe.regularized_table(restored).to_latex()))
        predictions = {"indices": list(after.index), "values": after.tolist()}
        pred_path = destination/(name+"-heldout.json")
        pred_path.write_text(json.dumps(predictions, allow_nan=False, sort_keys=True))
        assert strict_json(pred_path) == predictions
        results[name] = {"training_rows": 120, "heldout_rows": 30,
                         "full_model_file_readback": True, "exact_restored_prediction": True,
                         "changed_holdout_labels_do_not_refit": True,
                         "selected_components": state.get("components"),
                         "selected_penalty": state.get("selected_penalty"),
                         "coefficients": state["coefficients"],
                         "constant": state["constant"],
                         "all_heldout_predictions": after.tolist(),
                         "selected_model_inference_available": False}
    return results


def execute(destination):
    workflows = {}
    variables, _, stdout = run_example("next_eight_proxy_svar.py", destination/"proxy")
    assert variables["proof"]["status"] == "passed"
    proxy = variables["identified"]
    assert oe.summary_state(variables["restored"]) == variables["state_file"].read_text()
    workflows["proxy_svar"] = {"example": variables["proof"], "stdout": stdout,
        "all_response_values": variables["replayed"].irf.tolist(),
        "moment_matrix": proxy.attrs["proxy_state"]["complete_joint_centered_moments"],
        "proxy_hash": proxy.attrs["proxy_state"]["proxy_data_hash"]}
    variables, pair, stdout = run_example("next_eight_irt_diagnostics.py", destination/"irt", callback=True)
    fit, dif = pair
    for name, output in (("fit", fit), ("dif", dif)):
        path = destination/"irt"/(name+".json")
        replay = oe.irt_diagnostics_restore(path.read_text())
        assert replay.to_json() == output.to_json()
        (destination/"irt"/(name+".tex")).write_text(replay.to_latex())
    workflows["irt_diagnostics"] = {"stdout": stdout, "complete_state_readback": True,
        "fit_table_rows": {name: len(fit[name]) for name in fit},
        "dif_table_rows": {name: len(dif[name]) for name in dif}}
    variables, _, stdout = run_example("smoothing_multivariate_targets.py", destination/"smoothing-rm")
    payload = strict_json(destination/"smoothing-rm/smoothing_multivariate_targets.json")
    summaries = {}
    for name in ("kernelreg", "localreg"):
        state = payload[name]
        restored = oe.ResultBundle.model_validate(state["result"])
        assert restored.model_dump(mode="json") == state["result"]
        targets = oe.restore_summary(json.dumps(state["targets"], allow_nan=False, sort_keys=True))
        assert json.loads(oe.summary_state(targets)) == state["targets"]
        summaries[name] = {table: len(frame) for table, frame in targets.items()}
    for name, state in payload["repeated_measures"].items():
        restored = oe.restore_summary(json.dumps(state, allow_nan=False, sort_keys=True))
        assert json.loads(oe.summary_state(restored)) == state
        summaries["repeated_measures_"+name] = {table: len(frame) for table, frame in restored.items()}
    workflows["smoothing_multivariate"] = {"stdout": stdout, "full_payload_readback": True, "tables": summaries}
    _, sizes, stdout = run_example("next_eight_inference_stability.py", destination/"inference", callback=True)
    for name in ("trajectory", "ols-cusum", "mixed-contrast"):
        assert restored_summary_file(destination/"inference"/(name+".json")) == sizes[name]
    for name in ("lp-result", "mixed-result"):
        path = destination/"inference"/(name+".json")
        restored = oe.ResultBundle.model_validate_json(path.read_text())
        assert restored.model_dump(mode="json") == strict_json(path)
    workflows["inference_stability"] = {"stdout": stdout, "full_state_readback": True, "tables": sizes}
    workflows["regularized_pls"] = regularized_models(destination/"regularized")
    return workflows


temporary = None
if args.output_dir:
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
else:
    temporary = tempfile.TemporaryDirectory(prefix="openecon-next-eight-native-")
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
        "schema": "openecon.next-eight-runtime.v1",
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
    print("NEXT_EIGHT_RUNTIME_OK " + json.dumps({"execution": receipt["execution"],
        "workflows": len(workflows), "saved_files": len(files), "source_modules": len(source_hashes),
        "loaded_modules": len(origins), "forbidden_import_attempts": attempts,
        "complete_state_readback": True, "receipt": str(receipt_path)}, sort_keys=True))
finally:
    if temporary:
        temporary.cleanup()
