"""Native proxy-SVAR source/installed SDK dependency and complete-state proof."""

import argparse
import builtins
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import pandas  # noqa: F401
import torch


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--receipt", type=Path)
parser.add_argument("--installed-root", type=Path)
args = parser.parse_args()
blocked = {"scipy", "statsmodels", "sklearn", "linearmodels"}
assert not any(name.split(".")[0] in blocked for name in sys.modules)
attempts = []
original_import = builtins.__import__


def guarded(name, *a, **kw):
    if name.split(".")[0] in blocked:
        attempts.append(name)
        raise AssertionError("External estimator imported: " + name)
    return original_import(name, *a, **kw)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            raise AssertionError("External estimator imported: " + fullname)
        return None


builtins.__import__ = guarded
sys.meta_path.insert(0, Finder())
torch.set_num_threads(1)
import openecon as oe  # noqa: E402

if args.installed_root:
    assert Path(oe.__file__).resolve().is_relative_to(args.installed_root.resolve())
receipt = {
    "schema": "openecon.proxy-svar-runtime.v1", "issue": 36,
    "guard": "builtins import plus MetaPathFinder before OpenEcon import; no forbidden library preloaded",
    "blocked_libraries": sorted(blocked),
    "scope": "Resident CPU native single-proxy point identification and full JSON replay; not licensed vendor or installed desktop validation",
    "sdk_origin": str(Path(oe.__file__).resolve()),
}
with tempfile.TemporaryDirectory(prefix="openecon-proxy-state-proof-") as directory:
    prior = os.getcwd()
    try:
        os.chdir(directory)
        outputs = runpy.run_path(str(ROOT/"docs/examples/next_eight_proxy_svar.py"))
        receipt["example"] = outputs["proof"]
        receipt["complete_state_file_sha256"] = hashlib.sha256(outputs["state_file"].read_bytes()).hexdigest()
        receipt["complete_var_file_sha256"] = hashlib.sha256(outputs["model_file"].read_bytes()).hexdigest()
        receipt["all_response_values"] = outputs["replayed"].irf.tolist()
        receipt["proxy_hash"] = outputs["restored"].attrs["proxy_state"]["proxy_data_hash"]
    finally:
        os.chdir(prior)
paths = [
    "src/openecon/econometrics/structural/proxy.py",
    "src/openecon/econometrics/structural/__init__.py",
    "src/openecon/econometrics/var/estimators.py",
    "src/openecon/econometrics/var/kernels.py",
    "src/openecon/econometrics/var/common.py",
    "src/openecon/econometrics/core.py",
    "src/openecon/econometrics/summary_state.py",
    "src/openecon/econometrics/resident_cpu.py",
]
receipt["source_modules_sha256"] = {
    path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths
}
if args.installed_root:
    for path, expected in receipt["source_modules_sha256"].items():
        installed = args.installed_root/path.removeprefix("src/")
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == expected, path
    receipt["installed_modules_equal_source"] = True
receipt["example_sha256"] = hashlib.sha256((ROOT/"docs/examples/next_eight_proxy_svar.py").read_bytes()).hexdigest()
receipt["forbidden_import_attempts"] = attempts
assert not attempts
encoded = json.dumps(receipt, allow_nan=False, sort_keys=True, indent=2)
if args.receipt:
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(encoded+"\n")
print("PROXY_RUNTIME_OK " + json.dumps({"response_rows": receipt["example"]["response_rows"],
      "json_restored": True, "forbidden_import_attempts": attempts, "installed_sdk": bool(args.installed_root)}, sort_keys=True))
