"""Preserve accepted CF science and every current-main logical help field."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from verify_editor_api_preserved import logical

ROOT = Path(__file__).resolve().parents[1]


def committed(pin, path):
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT)


def cf_contract(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "control_function":
                    return ast.literal_eval(value)
    raise ValueError("Control-function contract missing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    pin, baseline = [subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=ROOT, text=True).strip() for ref in (args.source_ref, args.baseline_ref)]
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", pin], cwd=ROOT, text=True).splitlines()
    accepted = [name for name in paths if name.startswith("src/openecon/econometrics/control_function/")
                or name.startswith("tests/test_control_function_")
                or name in {"scripts/verify_control_function_oracles.py", "scripts/verify_control_function_native.py", "examples/control_functions_eight.py", "docs/econometrics/control-functions.md"}]
    changed = [name for name in accepted if committed(pin, name) != (ROOT/name).read_bytes()]
    if changed:
        raise ValueError("Accepted CF science/helper changed: " + ", ".join(changed))
    if cf_contract(committed(pin, "src/openecon/analysis_contracts.py")) != cf_contract((ROOT/"src/openecon/analysis_contracts.py").read_bytes()):
        raise ValueError("Accepted CF contract changed.")
    before = logical(committed(baseline, "web/src/editor-api.json"))
    current = logical((ROOT/"web/src/editor-api.json").read_bytes())
    lost = [name for name, value in before.items() if current.get(name) != value]
    if lost:
        raise ValueError("Current-main logical help differs: " + ", ".join(lost))
    accepted_help = logical(committed(pin, "web/src/editor-api.json"))
    cf_help = {name: value for name, value in accepted_help.items() if name in {
        "openecon.cfregress", "openecon.cflogit", "openecon.cfprobit", "openecon.cfcloglog", "openecon.cfpoisson", "openecon.cfgamma", "openecon.cfinvgauss", "openecon.cffraclogit", "openecon.cf_predict"}}
    if len(cf_help) != 9 or any(current.get(name) != value for name, value in cf_help.items()):
        raise ValueError("Accepted CF help lost a public signature or metadata field.")
    report = dict(status="pass", source_ref=pin, baseline_ref=baseline,
                  accepted_files={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in accepted},
                  accepted_scientific_file_count=len(accepted), accepted_contract_equal=True,
                  accepted_cf_help_equal=True, current_main_help_count=len(before),
                  existing_current_main_metadata_equal=True, current_help_count=len(current),
                  current_catalog_bytes=(ROOT/"web/src/editor-api.json").stat().st_size)
    if report["current_catalog_bytes"] >= 500000:
        raise ValueError("Catalog exceeded its unchanged byte limit.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "accepted_scientific_file_count", "current_main_help_count", "current_catalog_bytes")}))


if __name__ == "__main__":
    main()
