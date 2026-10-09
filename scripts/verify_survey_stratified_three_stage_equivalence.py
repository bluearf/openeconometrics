"""Compare pinned source with actual native stratified-three-stage saved state and history.

The development interpreter executes a temporary Git archive of --source-ref.
No checkout, production file, native project, or retained input is modified.
This verifies equality/provenance, separately from the independent math oracle.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "src/openecon").is_dir()
)
CASES = ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson")
EXAMPLE = "docs/examples/survey_stratified_three_stage_eight.py"
HELPER = "scripts/verify_survey_stratified_three_stage_installed.py"
MARKER = "SURVEY_STRATIFIED_THREE_STAGE_INSTALLED:"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def adjacent(directory, name):
    return next((p for p in (directory / name, directory / (name + ".gz")) if p.is_file()), None)


def pinned_bytes(source_ref, path):
    return subprocess.check_output(["git", "show", f"{source_ref}:{path}"], cwd=ROOT)


def helper_contract(source_ref):
    """Read literal provenance constants; never import or operate the installed helper."""
    source = pinned_bytes(source_ref, HELPER)
    names = {"MODULES", "MARKER", "IDENTIFIER", "STATE", "POSTSTATE"}
    contract = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                contract[target.id] = ast.literal_eval(node.value)
    if set(contract) != names or contract["MARKER"] != MARKER:
        raise RuntimeError("Pinned installed-helper contract is incomplete.")
    modules = contract["MODULES"]
    if len(modules) != len(set(modules)) or len(modules) != 38:
        raise RuntimeError("Require the complete38-module pinned helper inventory.")
    contract["sha256"] = hashlib.sha256(source).hexdigest()
    return contract


def compare(source, native, path=""):
    record = {
        "numeric_leaves": 0,
        "max_abs_difference": 0.0,
        "max_relative_difference": 0.0,
        "non_numeric_differences": [],
        "numeric_differences": [],
    }

    def visit(a, b, at):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                record["non_numeric_differences"].append(
                    {"path": at, "source_keys": sorted(a), "native_keys": sorted(b)}
                )
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], at + "/" + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                record["non_numeric_differences"].append(
                    {"path": at, "source_length": len(a), "native_length": len(b)}
                )
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, at + "/" + str(i))
        elif type(a) in (int, float) and type(b) in (int, float):
            assert math.isfinite(a) and math.isfinite(b), at
            record["numeric_leaves"] += 1
            delta, scale = abs(a - b), max(abs(a), abs(b))
            relative = delta / scale if scale else 0.0
            record["max_abs_difference"] = max(record["max_abs_difference"], delta)
            record["max_relative_difference"] = max(record["max_relative_difference"], relative)
            if delta or type(a) is not type(b):
                record["numeric_differences"].append(
                    {
                        "path": at,
                        "source": a,
                        "native": b,
                        "source_type": type(a).__name__,
                        "native_type": type(b).__name__,
                    }
                )
        elif a != b or type(a) is not type(b):
            record["non_numeric_differences"].append({"path": at, "source": a, "native": b})

    visit(source, native, path)
    record["canonical_json_exact"] = canonical(source) == canonical(native)
    return record


def pinned_execution(source_ref):
    """Import only archived package source in an isolated development process."""
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", source_ref, "src/openecon", EXAMPLE], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(prefix="stratified-three-stage-pinned-source-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(root, filter="data")
        code = """
import contextlib, hashlib, io, json, pathlib, platform, runpy, sys
root=pathlib.Path(sys.argv[1])
sys.path.insert(0,str(root/'src'))
import openecon as oe
import pandas as pd
import pydantic
import torch
assert pathlib.Path(oe.__file__).resolve().is_relative_to(root.resolve()/'src')
displays=[]
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):
    source=runpy.run_path(str(root/'docs/examples/survey_stratified_three_stage_eight.py'),
                         init_globals={'display':displays.append})
assert len(displays)==8
assert len(source['latexstates'])==8
latex=[oe.to_latex(table) for table in displays]
assert latex==[source['latexstates'][case] for case in
               ('mean','total','ratio','proportion','regress','logit','probit','poisson')]
saved_postestimation={}
for case, state in source['restored'].items():
    if case in ('mean','total','ratio','proportion'):
        targets={'contrast':source['table_state'](state.contrast([1.]+[0.]*(len(state.labels)-1)))}
    else:
        targets={
            'lincom':source['table_state'](state.lincom([0.,1.,0.])),
            'test':source['table_state'](state.test([[0.,1.,0.],[0.,0.,1.]])),
            'predict':source['table_state'](state.predict(pd.DataFrame({'x':[-.5,.5],'z':[.2,-.2]}))),
        }
    saved_postestimation[case]={kind:hashlib.sha256(json.dumps(value,allow_nan=False,
        sort_keys=True,separators=(',',':')).encode()).hexdigest() for kind,value in targets.items()}
assert sum(len(value) for value in saved_postestimation.values())==16
record={'payload':{key:source[key] for key in ('states','poststates','oracle_inputs')},
        'latex':latex,'display_count':len(displays),
        'saved_postestimation':saved_postestimation,
        'source_stdout_receipt_present':'SURVEY_STRATIFIED_THREE_STAGE_RECEIPT:' in stdout.getvalue(),
        'source_example_sha256':hashlib.sha256((root/'docs/examples/survey_stratified_three_stage_eight.py').read_bytes()).hexdigest(),
        'environment':{'python_executable':sys.executable,'python_version':platform.python_version(),
                       'torch':torch.__version__,'pandas':pd.__version__,'pydantic':pydantic.__version__},
        'package_loaded_from_archived_source':True}
print(json.dumps(record,allow_nan=False))
"""
        environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-c", code, str(root)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--native-execution", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_ref + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    if not args.payload.is_file():
        raise RuntimeError(
            "Require the actual retained native payload; no native claim without it."
        )
    contract = helper_contract(source_ref)
    directory = args.native_receipt.resolve().parent
    execution_path = args.native_execution or adjacent(directory, "native-execution.json")
    if execution_path is None:
        raise RuntimeError(
            "Require the actual native execution history, including all eight outputs."
        )
    paths = [
        args.native_receipt,
        args.payload,
        execution_path,
        *(directory / f"output-{i}.tex" for i in range(1, 9)),
    ]
    state_path, table_path = (
        adjacent(directory, "persisted-states.json"),
        adjacent(directory, "postestimation-states.json"),
    )
    paths.extend(p for p in (state_path, table_path) if p is not None)
    if args.receipt.resolve() in {path.resolve() for path in paths}:
        raise RuntimeError("Comparison receipt must not overwrite any retained native input.")
    before = {str(path.resolve()): sha(path) for path in paths}
    native_receipt, native, execution = (
        read_json(path) for path in (args.native_receipt, args.payload, execution_path)
    )
    assert native_receipt["status"] in ("native-verified", "native-restarted")
    assert native_receipt["identity"]["source_ref"] == source_ref
    identity = native_receipt["identity"]
    assert identity["identifier"] == contract["IDENTIFIER"]
    assert identity["verifier_source_sha256"] == contract["sha256"]
    example_sha = hashlib.sha256(pinned_bytes(source_ref, EXAMPLE)).hexdigest()
    assert identity["example_source_sha256"] == example_sha
    assert set(identity["source_parity"]) == set(contract["MODULES"])
    for module in contract["MODULES"]:
        module_path = "src/" + module.replace(".", "/")
        file_path = module_path + ("/__init__.py" if (ROOT / module_path).is_dir() else ".py")
        assert (
            identity["source_parity"][module]
            == hashlib.sha256(pinned_bytes(source_ref, file_path)).hexdigest()
        ), module
    proof = native_receipt["proof"]
    assert proof["source_ref"] == source_ref and proof["frozen"] is True
    assert proof["sampling_stages"] == 3 and proof["stages"] == proof["latex_count"] == 8
    assert proof["all_three_fpc_terms"] is True and proof["restored_equal"] is True
    assert proof["stage2_stratification"] is True and proof["n_cells"] == 12
    assert proof["rows"] == 48 and proof["design_df"] == 3
    assert proof["source_oracles_available"] is False
    assert set(proof["modules"]) == set(contract["MODULES"])
    assert set(native) == {"states", "poststates", "oracle_inputs"}
    assert set(native["states"]) == set(native["poststates"]) == set(CASES)
    assert execution["status"] == "ok" and execution["id"] == native_receipt["execution_id"]
    assert object_sha(execution) == native_receipt["execution_sha256"]
    assert object_sha([execution]) == native_receipt["history_sha256"]
    assert hashlib.sha256(execution["code"].encode()).hexdigest() == native_receipt["source_sha256"]
    markers = [
        line[len(MARKER) :] for line in execution["stdout"].splitlines() if line.startswith(MARKER)
    ]
    assert len(markers) == 1 and json.loads(markers[0]) == proof
    if state_path is not None:
        assert canonical(read_json(state_path)) == canonical(native["states"])
    if table_path is not None:
        assert canonical(read_json(table_path)) == canonical(native["poststates"])
    for key, payload_key, filename in (
        ("states", "states", contract["STATE"]),
        ("postestimation", "poststates", contract["POSTSTATE"]),
        ("oracle_inputs", "oracle_inputs", "survey-stratified-three-stage-oracle-inputs.json"),
    ):
        # These hashes describe compact persisted worker files. Retained copies
        # may be pretty printed, so reconstruct the helper's exact byte encoding.
        raw = json.dumps(native[payload_key], allow_nan=False, separators=(",", ":")).encode()
        expected = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        assert native_receipt["files"][key] == expected
        assert proof["files"][key] == {"file": filename, **expected}
    outputs = execution["outputs"]
    assert len(outputs) == len(native_receipt["outputs"]) == 8
    source = pinned_execution(source_ref)
    assert source["source_example_sha256"] == example_sha
    cases = {}
    for index, case in enumerate(CASES, start=1):
        output, table = outputs[index - 1], native["poststates"][case]
        assert output["type"] == "table" and output["data"]["rows"]
        latex_path = directory / f"output-{index}.tex"
        native_latex_bytes = latex_path.read_bytes()
        native_latex = native_latex_bytes.decode("utf-8")
        assert native_latex == output["latex"]
        assert (
            hashlib.sha256(native_latex_bytes).hexdigest()
            == native_receipt["outputs"][index - 1]["latex_sha256"]
        )
        assert output["data"]["columns"] == table["columns"]
        assert output["data"]["rows"] == table["data"]
        assert output["data"]["index"] == [[label] for label in table["index"]]
        assert output["data"]["total_rows"] == len(table["index"])
        assert output["data"]["total_columns"] == len(table["columns"])
        cases[case] = {
            "state": compare(
                source["payload"]["states"][case], native["states"][case], case + "/state"
            ),
            "postestimation_table": compare(
                source["payload"]["poststates"][case], table, case + "/postestimation_table"
            ),
            "latex_exact": source["latex"][index - 1].encode("utf-8") == native_latex_bytes,
            "native_history_matches_saved_table": True,
        }
    after = {str(path.resolve()): sha(path) for path in paths}
    assert after == before, "Comparison changed retained native evidence."
    full = compare(source["payload"], native)
    footer_hashes = proof.get("saved_postestimation")
    footer_exact = (
        footer_hashes == source["saved_postestimation"] if footer_hashes is not None else None
    )
    exact = (
        full["canonical_json_exact"]
        and all(case["latex_exact"] for case in cases.values())
        and footer_exact is not False
    )
    receipt = {
        "status": "exact" if exact else "differences observed",
        "source_pin": source_ref,
        "current_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_execution": "temporary archive of explicit source commit in isolated development subprocess",
        "source_package_loaded_from_archive": source["package_loaded_from_archived_source"],
        "source_example_sha256": source["source_example_sha256"],
        "source_environment": source["environment"],
        "installed_helper_source_sha256": contract["sha256"],
        "native_module_source_hashes_match_pin": True,
        "native_module_count": len(contract["MODULES"]),
        "native_status": native_receipt["status"],
        "native_execution_sha256": object_sha(execution),
        "native_history_sha256": object_sha([execution]),
        "native_evidence_sha256": before,
        "native_evidence_unchanged": after == before,
        "case_count": 8,
        "display_table_count": source["display_count"],
        "latex_count": 8,
        "source_stdout_receipt_present": source["source_stdout_receipt_present"],
        "complete_payload": full,
        "cases": cases,
        "saved_postestimation_footer": {
            "available": footer_hashes is not None,
            "exact": footer_exact,
            "source_hashes": source["saved_postestimation"],
            "native_hashes": footer_hashes,
            "verified_hash_count": 16 if footer_exact else 0,
        },
        "source_canonical_payload_sha256": object_sha(source["payload"]),
        "native_canonical_payload_sha256": object_sha(native),
        "scope": "pinned source versus actual installed native saved eight states, full covariance, metadata, displayed tables and LaTeX exports",
        "independent_oracle_claim": False,
        "checksums_authenticate_source_data": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cases": 8,
                "numeric_leaves": full["numeric_leaves"],
                "max_abs_difference": full["max_abs_difference"],
                "receipt": str(args.receipt),
            }
        )
    )
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
