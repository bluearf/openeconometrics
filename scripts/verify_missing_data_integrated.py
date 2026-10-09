"""Verify that a concurrent main merge preserves accepted MI science and help."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"MIResult", "MIDiagnosticResult", "MIPoolResult", "MIJointResult"}
HELPERS = {"missing_patterns", "mvnorm_em", "little_mcar", "mi_mvn",
           "mi_monotone", "mi_chained", "mi_pool", "mi_test"}


def committed(pin, path):
    return subprocess.check_output(["git", "show", f"{pin}:{path}"], cwd=ROOT)


def contract(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "missing_data":
                    return ast.literal_eval(value)
    raise ValueError("Missing-data contract missing")


def bindings(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets
        ):
            return {key.value: ast.literal_eval(value)
                    for key, value in zip(node.value.keys, node.value.values)
                    if isinstance(key, ast.Constant) and key.value in TYPES}
    raise ValueError("Public bindings missing")


def help_entries(source):
    # Compare complete logical metadata across every published catalog codec.
    from verify_editor_api_preserved import logical

    return {name: item for name, item in logical(source).items()
            if name.startswith("openecon.") and name.split(".")[1] in TYPES | HELPERS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_ref + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    paths = sorted(str(path.relative_to(ROOT))
                   for path in (ROOT / "src/openecon/econometrics/mi").glob("*.py"))
    paths += ["src/openecon/models.py", "src/openecon/console_worker.py",
              "src/openecon/econometrics/core.py", "src/openecon/resources.py",
              "src/openecon/engines/distributions.py", "docs/examples/missing_data_eight.py",
              "scripts/verify_missing_data_native.py", "scripts/verify_missing_data_oracles.py"]
    hashes = {}
    for path in paths:
        current = (ROOT / path).read_bytes()
        if committed(pin, path) != current:
            raise ValueError(f"Accepted scientific/execution source changed: {path}")
        hashes[path] = hashlib.sha256(current).hexdigest()
    path = "src/openecon/analysis_contracts.py"
    assert contract(committed(pin, path)) == contract((ROOT / path).read_bytes())
    path = "src/openecon/__init__.py"
    assert bindings(committed(pin, path)) == bindings((ROOT / path).read_bytes())
    path = "web/src/editor-api.json"
    assert help_entries(committed(pin, path)) == help_entries((ROOT / path).read_bytes())
    from openecon.econometrics import registry
    from openecon.econometrics.mi import EXPORTS
    public = registry.public_exports()
    assert all(public[key] == tuple(value.split(":")) for key, value in EXPORTS.items())
    proof = {"status": "pass", "accepted_native_source_ref": pin,
             "combined_source_ref": subprocess.check_output(
                 ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "unchanged_scientific_and_execution_files": hashes,
             "missing_data_capability_contract_equal": True,
             "typed_public_bindings_equal": True, "all_mi_helper_targets_equal": True,
             "all_accepted_mi_help_equal_after_lossless_decoding": True,
             "unrelated_shared_changes": ["conjoint/categorical families", "SurveyRegressionResult binding",
                                           "survey/conjoint/simultaneous-inference metadata",
                                           "smoothing saved postestimation", "causal-design family"],
             "whole_sdk_frozen_parity_claim": False}
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"status": "pass", "unchanged_files": len(hashes)}))


if __name__ == "__main__":
    main()
