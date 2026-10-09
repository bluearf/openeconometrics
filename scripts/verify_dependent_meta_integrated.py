"""Prove scoped native/source identity after additive main integration.

The QA app remains pinned to its original complete compiled baseline. Its
scientific modules must still match current source, while independently added
metadata is checked separately. This is not a claim that the app ships every
new method subsequently added to main.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

from verify_conditional_integrated import metadata_difference as legacy_metadata_difference
from verify_dependent_meta_runtime import MARKER, MODULES, RESULT_NAMES, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = {"openecon.econometrics.meta.dependent", "openecon.econometrics.meta.dependent_kernels",
              "openecon.econometrics.meta.dependent_post"}
OBSERVED_ROOT_ADDITIONS = {
    "SurveyStratifiedThreeStageDesign": ("openecon.survey_stratified_three_stage", "SurveyStratifiedThreeStageDesign"),
    "survey_stratified_three_stage_design": ("openecon.survey_stratified_three_stage", "survey_stratified_three_stage_design"),
    "SurveyStratifiedThreeStageResult": ("openecon.econometrics.survey.stratified_three_stage_targets", "SurveyStratifiedThreeStageResult"),
    "SurveyStratifiedThreeStageRegressionResult": ("openecon.econometrics.survey.stratified_three_stage_regression_state", "SurveyStratifiedThreeStageRegressionResult"),
}


def shape(node):
    return ast.dump(node, include_attributes=False)


def contract_mapping(tree):
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "capabilities")
    result = function.body[-1]
    if not isinstance(result, ast.Return) or not isinstance(result.value, ast.Dict):
        raise RuntimeError("Unexpected static capability declaration")
    return result.value


def entries(mapping):
    if any(not isinstance(key, ast.Constant) or not isinstance(key.value, str) for key in mapping.keys):
        raise RuntimeError("Capability declarations must remain static named fields")
    return {key.value: value for key, value in zip(mapping.keys, mapping.values)}


def root_metadata_difference(before, after):
    """Admit only four inspected literal additions, retaining the legacy gate."""
    left, right = ast.parse(before), ast.parse(after)
    def exports(tree):
        return next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets))
    a, b = exports(left), exports(right)
    def literal_fields(mapping):
        return {ast.literal_eval(key): value for key, value in zip(mapping.keys, mapping.values) if key is not None}
    old, new = literal_fields(a.value), literal_fields(b.value)
    observed = (set(new)-set(old)) & set(OBSERVED_ROOT_ADDITIONS)
    if not observed:
        return legacy_metadata_difference("__init__", before, after)
    if observed != set(OBSERVED_ROOT_ADDITIONS):
        raise RuntimeError("Unexpected partial observed survey export addition")
    if not set(old) <= set(new) or any(shape(new[key]) != shape(value) for key, value in old.items()):
        raise RuntimeError("An original root public binding changed")
    for key, target in OBSERVED_ROOT_ADDITIONS.items():
        if ast.literal_eval(new[key]) != target or sum(node is not None and ast.literal_eval(node) == key for node in b.value.keys) != 1:
            raise RuntimeError("Observed survey export target or multiplicity changed")
    def public_names(tree):
        return [ast.literal_eval(value) for node in tree.body
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "__all__" and isinstance(node.value, ast.List)
                for value in node.value.elts]
    old_names, new_names = public_names(left), public_names(right)
    if any(new_names.count(key)-old_names.count(key) != 1 for key in observed):
        raise RuntimeError("Observed root exports require exactly matching public-name additions")
    kept = [(key, value) for key, value in zip(b.value.keys, b.value.values)
            if key is None or ast.literal_eval(key) not in observed]
    b.value.keys = [key for key, _ in kept]
    b.value.values = [value for _, value in kept]
    body = []
    for node in right.body:
        if (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "__all__" and isinstance(node.value, ast.List)):
            node.value.elts = [value for value in node.value.elts if ast.literal_eval(value) not in observed]
            if not node.value.elts:
                continue
        body.append(node)
    right.body = body
    if shape(left) == shape(right):
        legacy = {}
    else:
        # Any other additions must still satisfy the unmodified published
        # legacy rules; executable/import/getattr changes remain failures.
        legacy = legacy_metadata_difference("__init__", before, ast.unparse(right))
    return {"observed_literal_root_exports": OBSERVED_ROOT_ADDITIONS,
            "matching_public_name_additions": sorted(observed),
            "all_original_root_bindings_equal": True,
            "all_other_root_ast_equal": True, "additional_legacy_metadata": legacy}


def metadata_difference(name, before, after):
    if name == "openecon":
        return root_metadata_difference(before, after)
    if name == "openecon.econometrics.registry":
        return legacy_metadata_difference("econometrics.registry", before, after)
    if name == "openecon.econometrics.meta":
        left, right = ast.parse(before), ast.parse(after)
        def exports(tree):
            return next(node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "EXPORTS" for target in node.targets))
        a, b = exports(left), exports(right)
        old, new = ast.literal_eval(a.value), ast.literal_eval(b.value)
        if not set(old) <= set(new) or any(new[key] != value for key, value in old.items()):
            raise RuntimeError("An existing meta public binding changed")
        a.value = b.value = ast.Constant(value=None)
        if shape(left) != shape(right):
            raise RuntimeError("Meta-family executable catalogue logic changed")
        return {"added_public_bindings": {key: new[key] for key in sorted(set(new)-set(old))},
                "all_existing_meta_bindings_equal": True, "all_other_meta_catalogue_ast_equal": True}
    if name != "openecon.analysis_contracts":
        raise RuntimeError(f"Unclassified scientific/shared module difference: {name}")
    left, right = ast.parse(before), ast.parse(after)
    a, b = contract_mapping(left), contract_mapping(right)
    old, new = entries(a), entries(b)
    old_meta, new_meta = entries(old["meta_analysis"]), entries(new["meta_analysis"])
    if shape(old_meta["dependent_effects"]) != shape(new_meta["dependent_effects"]):
        raise RuntimeError("The accepted dependent-meta capability contract changed")
    changed = sorted(key for key in set(old)|set(new)
                     if key not in old or key not in new or shape(old[key]) != shape(new[key]))
    for mapping in (a, b):
        keep = [(key, value) for key, value in zip(mapping.keys, mapping.values) if key.value not in changed]
        mapping.keys = [key for key, _ in keep]
        mapping.values = [value for _, value in keep]
    if shape(left) != shape(right):
        raise RuntimeError("Capability/error executable logic changed outside static metadata")
    return {"changed_metadata_fields": changed, "dependent_meta_contract_ast_equal": True,
            "all_other_contract_ast_equal": True}


def relative_source(name):
    directory = "packages/openecon-charts/src" if name.startswith("openecon_charts") else "src"
    path = Path(directory)/Path(*name.split("."))
    return path/"__init__.py" if (ROOT/path).is_dir() else path.with_suffix(".py")


def verify(runtime, frozen_receipt, native_receipt, native_artifacts, baseline_head=None):
    from PyInstaller.archive.readers import CArchiveReader
    runtime = Path(runtime).resolve(strict=True)
    frozen = json.loads(Path(frozen_receipt).read_text())
    native = json.loads(Path(native_receipt).read_text())
    baseline = baseline_head or frozen["source_head"]
    if frozen["status"] != "passed" or not frozen["complete_source_frozen_artifacts_equal"]:
        raise RuntimeError("A successful full frozen/source artifact receipt is required")
    if native["status"] != "passed" or not native.get("native_run_and_rendered_tables_observed") or not native.get("app_restart_readback_unchanged"):
        raise RuntimeError("Observed native Run and actual app quit/relaunch readback are required")
    if digest(runtime) != frozen["runtime_sha256"] or digest(runtime) != native["runtime_sha256"]:
        raise RuntimeError("The selected runtime is not the native/frozen verified runtime")
    if set(frozen["compiled_modules_equal_source"]) != set(MODULES):
        raise RuntimeError("The frozen compiled baseline declaration is incomplete")
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    unchanged, metadata = {}, {}
    for name in MODULES:
        relative = relative_source(name)
        before = subprocess.check_output(["git", "show", baseline+":"+relative.as_posix()], cwd=ROOT)
        after = (ROOT/relative).read_bytes()
        expected_sha = hashlib.sha256(before).hexdigest()
        if frozen["compiled_modules_equal_source"][name] != expected_sha:
            raise RuntimeError(f"Frozen source receipt does not match baseline Git source: {name}")
        bundled = normalized(archive.extract(name))
        if bundled != normalized(compile(before.decode(), relative.as_posix(), "exec", dont_inherit=True)):
            raise RuntimeError(f"Whole compiled module differs from the pinned baseline: {name}")
        if before == after:
            if bundled != normalized(compile(after.decode(), str(ROOT/relative), "exec", dont_inherit=True)):
                raise RuntimeError(f"Unchanged current module differs from compiled baseline: {name}")
            unchanged[name] = expected_sha
        else:
            if name in SCIENTIFIC:
                raise RuntimeError(f"Accepted scientific source changed: {name}")
            metadata[name] = metadata_difference(name, before.decode(), after.decode())
    if not SCIENTIFIC <= set(unchanged):
        raise RuntimeError("The three accepted scientific modules were not preserved")
    expected = frozen["complete_result_hashes"]
    if expected != native["complete_result_hashes"] or set(expected) != set(RESULT_NAMES):
        raise RuntimeError("The full native and frozen artifact hashes differ")
    native_artifacts = Path(native_artifacts).resolve(strict=True)
    compared = []
    with tempfile.TemporaryDirectory(prefix="dependent-meta-integrated-source-") as temporary:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(ROOT/"docs/examples/dependent_meta_eight.py"), init_globals={
                "display": lambda _: None, "DEPENDENT_META_RESULT_DIRECTORY": temporary})
        proofs = [json.loads(line[len(MARKER):]) for line in output.getvalue().splitlines() if line.startswith(MARKER)]
        if len(proofs) != 1 or proofs[0]["frozen"] or not proofs[0]["complete_artifacts_equal"]:
            raise RuntimeError("Fresh integrated source example did not produce a complete receipt")
        if proofs[0]["artifact_sha256"] != expected:
            raise RuntimeError("Integrated current-source full artifacts differ from native/frozen execution")
        for name in RESULT_NAMES:
            current, installed = Path(temporary)/(name+".json"), native_artifacts/(name+".json")
            if digest(current) != expected[name] or digest(installed) != expected[name] or current.read_bytes() != installed.read_bytes():
                raise RuntimeError(f"The complete native/source result bytes differ: {name}")
            compared.append(name+".json")
    return {"status": "passed", "bundle_baseline_source": baseline,
            "current_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "whole_compiled_baseline_modules_verified": len(MODULES),
            "current_complete_compiled_modules_unchanged": unchanged,
            "metadata_only_differences": metadata,
            "accepted_scientific_modules_unchanged": sorted(SCIENTIFIC),
            "complete_current_source_native_artifacts_equal": compared,
            "complete_result_hashes": expected, "runtime_sha256": digest(runtime),
            "frozen_receipt_sha256": digest(frozen_receipt), "native_receipt_sha256": digest(native_receipt),
            "verifier_sha256": digest(__file__), "qa_bundle_contains_new_other_families": False,
            "qa_bundle_scientific_baseline_preserved": True,
            "human_data_access": False, "public_release_delivered": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--frozen-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-artifacts", type=Path, required=True)
    parser.add_argument("--baseline-head")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    record = verify(args.runtime, args.frozen_receipt, args.native_receipt, args.native_artifacts, args.baseline_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": record["status"], "compiled_baseline_modules": len(MODULES),
                      "metadata_modules": list(record["metadata_only_differences"]),
                      "complete_artifacts_equal": len(record["complete_current_source_native_artifacts_equal"])}))


if __name__ == "__main__":
    main()
