"""Bridge the pinned native CF computation to committed integrated source.

Only the registry family declaration and declarative capability metadata may
change. Explicit approved baselines also permit their exact additive auxiliary
export metadata helper/import/call and one literal survey result export. This
proves source parity for thirty dependencies;
GUI execution, restart receipts, final web assets and release identity remain
separate evidence. --working-tree produces an explicitly provisional bridge.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = "df539c628c2ab277e89cf82eb5c790196273b652"
APPROVED_METADATA_BASELINE = "337c0465478128848179146df14f0e9598efd6f6"
APPROVED_PUBLIC_EXPORT_BASELINE = "6a1ab5c91dd67bc7a0b958024a88ebfc7061a79a"
APPROVED_METADATA_BASELINES = (APPROVED_METADATA_BASELINE, APPROVED_PUBLIC_EXPORT_BASELINE)
PUBLIC_EXPORT_NAME = "SurveyReplicateMarginsResult"
PUBLIC_EXPORT_TARGET = ("openecon.econometrics.survey.replicate_margins_state", PUBLIC_EXPORT_NAME)
REPORT = ROOT / "docs/evidence/control-functions-eight-2026-10-07/native-source-bridge.json"
METADATA_MODULES = {"openecon", "openecon.econometrics.registry", "openecon.analysis_contracts"}
EXPECTED_CF = {
    "openecon.econometrics.control_function", "openecon.econometrics.control_function.commands",
    "openecon.econometrics.control_function.kernels", "openecon.econometrics.control_function.state",
    "openecon.econometrics.control_function.postest",
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def committed(pin, path):
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT)


def resolve(ref):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], cwd=ROOT, text=True,
    ).strip()


def dump(node):
    return ast.dump(node, include_attributes=False)


def parsed(raw, name):
    try:
        return ast.parse(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot parse {name}; finish source conflict resolution before bridging.") from exc


def native_modules(pin):
    raw = committed(pin, "scripts/verify_control_function_native.py")
    candidates = [n.value for n in parsed(raw, "native verifier").body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODULES" for t in n.targets)]
    if len(candidates) != 1:
        raise ValueError("Native pin must declare exactly one dependency manifest.")
    modules = ast.literal_eval(candidates[0])
    if (not isinstance(modules, list) or len(modules) != 30 or len(set(modules)) != 30
            or any(not isinstance(v, str) or not re.fullmatch(r"openecon(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*", v) for v in modules)
            or not EXPECTED_CF | METADATA_MODULES | {"openecon.engines.separation"} <= set(modules)):
        raise ValueError("Native pin must declare thirty unique bounded CF dependencies including separation.")
    return modules, sha(raw)


def source_path(pin, module):
    stem = "src/" + module.replace(".", "/")
    for path in (stem + ".py", stem + "/__init__.py"):
        test = subprocess.run(["git", "cat-file", "-e", pin + ":" + path], cwd=ROOT, capture_output=True)
        if test.returncode == 0:
            return path
    raise ValueError("Pinned native dependency is missing: " + module)


def auxiliary_helper(raw):
    helpers = [n for n in parsed(raw, "auxiliary registry metadata").body
               if isinstance(n, ast.FunctionDef) and n.name == "auxiliary_exports"]
    if len(helpers) != 1:
        raise ValueError("Approved baseline must contain exactly one auxiliary_exports metadata helper.")
    return helpers[0]


def auxiliary_import(raw):
    tree = parsed(raw, "auxiliary capability metadata")
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "capabilities"]
    if len(functions) != 1:
        raise ValueError("Approved baseline must contain one capabilities() definition.")
    imports = [n for n in functions[0].body if isinstance(n, ast.ImportFrom)
               and n.module == "openecon.econometrics.registry"]
    if (len(imports) != 1 or sum(alias.name == "auxiliary_exports" and alias.asname is None
                                for alias in imports[0].names) != 1):
        raise ValueError("Approved baseline must import the single unaliased auxiliary_exports metadata helper.")
    return imports[0]


def public_export_parts(raw):
    tree = parsed(raw, "public export metadata")
    lists = [n for n in tree.body if isinstance(n, ast.AugAssign)
             and isinstance(n.target, ast.Name) and n.target.id == "__all__"
             and any(isinstance(v, ast.Constant) and v.value == "SurveyRegressionResult"
                     for v in ast.walk(n.value))]
    exports = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in n.targets)]
    if (len(lists) != 1 or not isinstance(lists[0].op, ast.Add)
            or not isinstance(lists[0].value, ast.List)
            or any(not isinstance(v, ast.Constant) or not isinstance(v.value, str)
                   for v in lists[0].value.elts)
            or len(exports) != 1 or len(exports[0].targets) != 1
            or not isinstance(exports[0].value, ast.Dict)):
        raise ValueError("Public exports must retain one literal survey __all__ addition and one original _EXPORTS dictionary.")
    names = [v.value for v in lists[0].value.elts]
    keys = [k.value for k in exports[0].value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    if len(set(names)) != len(names) or len(set(keys)) != len(keys):
        raise ValueError("Public export metadata has duplicate survey names or literal export keys.")
    return tree, lists[0], exports[0], names


def public_export_proof(native, current, *, baseline=None):
    before, old_list, old_exports, old_names = public_export_parts(native)
    after, new_list, new_exports, new_names = public_export_parts(current)
    added = new_names != old_names or dump(new_exports) != dump(old_exports)
    if added:
        if baseline is None:
            raise ValueError("The survey result export requires the exact approved 6a1ab5c --baseline-ref.")
        _, baseline_list, baseline_exports, baseline_names = public_export_parts(baseline)
        if (old_names != ["SurveyRegressionResult"]
                or new_names != [*old_names, PUBLIC_EXPORT_NAME]
                or new_names != baseline_names or dump(new_list) != dump(baseline_list)):
            raise ValueError("Public __all__ changes must preserve the native survey list and add only its exact baseline survey result.")
        selected = [i for i, k in enumerate(new_exports.value.keys)
                    if isinstance(k, ast.Constant) and k.value == PUBLIC_EXPORT_NAME]
        approved = [i for i, k in enumerate(baseline_exports.value.keys)
                    if isinstance(k, ast.Constant) and k.value == PUBLIC_EXPORT_NAME]
        if len(selected) != 1 or len(approved) != 1:
            raise ValueError("Public _EXPORTS must add exactly one approved survey result binding.")
        if (selected[0] != approved[0]
                or dump(new_exports.value.keys[selected[0]]) != dump(baseline_exports.value.keys[approved[0]])):
            raise ValueError("The added survey export key and its insertion position must exactly match the approved baseline.")
        value, approved_value = new_exports.value.values[selected[0]], baseline_exports.value.values[approved[0]]
        if (not isinstance(value, ast.Tuple) or len(value.elts) != 2
                or any(not isinstance(v, ast.Constant) or not isinstance(v.value, str) for v in value.elts)
                or tuple(v.value for v in value.elts) != PUBLIC_EXPORT_TARGET
                or dump(value) != dump(approved_value)):
            raise ValueError("The added survey export must be the exact approved literal module/class tuple; calls are refused.")
        normalized = deepcopy(new_exports)
        del normalized.value.keys[selected[0]]
        del normalized.value.values[selected[0]]
        if dump(normalized) != dump(old_exports):
            raise ValueError("Public _EXPORTS changed an old binding or added another export.")
        after.body[after.body.index(new_list)] = deepcopy(old_list)
        after.body[after.body.index(new_exports)] = normalized
    if dump(before) != dump(after):
        raise ValueError("Public package executable AST changed outside the exact approved survey export metadata.")
    outside = dump(ast.Module(body=[n for n in before.body if n is not old_list and n is not old_exports], type_ignores=[]))
    return {"excluded_ast": "One exact approved survey __all__ addition and one literal _EXPORTS binding if present",
            "outside_metadata_logic_ast_equal": True,
            "outside_metadata_logic_ast_sha256": sha(outside.encode()),
            "normalized_full_module_ast_equal": True,
            "normalized_full_module_ast_sha256": sha(dump(before).encode()),
            "all_native_export_bindings_preserved_in_order": True,
            "native_survey_all": old_names, "integrated_survey_all": new_names,
            "survey_result_export_added": added,
            "added_export_name": PUBLIC_EXPORT_NAME if added else None,
            "added_export_target": list(PUBLIC_EXPORT_TARGET) if added else None,
            "added_export_is_exact_baseline_literal_metadata": added}


def registry_proof(native, current, *, baseline=None):
    trees = [parsed(value, "registry") for value in (native, current)]
    families, logic, helper_added = [], [], False
    helpers = [[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "auxiliary_exports"]
               for tree in trees]
    if any(len(found) > 1 for found in helpers):
        raise ValueError("Registry has duplicate auxiliary_exports definitions.")
    if helpers[1] and not helpers[0]:
        if baseline is None or dump(helpers[1][0]) != dump(auxiliary_helper(baseline)):
            raise ValueError("Additive auxiliary_exports helper requires its exact approved --baseline-ref AST.")
        helper_added = True
    for i, tree in enumerate(trees):
        assignments = [n for n in tree.body if isinstance(n, ast.AnnAssign)
                       and isinstance(n.target, ast.Name) and n.target.id == "FAMILIES"]
        if len(assignments) != 1:
            raise ValueError("Registry must retain one explicit annotated FAMILIES declaration.")
        values = ast.literal_eval(assignments[0].value)
        if (not isinstance(values, tuple) or not values or len(values) > 100
                or any(not isinstance(v, str) or not re.fullmatch(r"[a-z_][a-z_0-9]*", v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Registry families must remain a unique bounded literal tuple.")
        families.append(values)
        excluded = [assignments[0], *(helpers[i] if i == 1 and helper_added else [])]
        logic.append(dump(ast.Module(body=[n for n in tree.body if all(n is not part for part in excluded)], type_ignores=[])))
    if logic[0] != logic[1]:
        raise ValueError("Original registry executable logic changed outside explicitly approved additive metadata.")
    if "control_function" not in families[1] or tuple(v for v in families[1] if v in families[0]) != families[0]:
        raise ValueError("Registry dropped/reordered a native family or lost control_function.")
    return {"excluded_ast": "FAMILIES literal tuple; exact approved additive auxiliary_exports helper if present", "outside_metadata_logic_ast_equal": True,
            "outside_metadata_logic_ast_sha256": sha(logic[0].encode()),
            "native_families": list(families[0]), "integrated_families": list(families[1]),
            "added_families": [v for v in families[1] if v not in families[0]],
            "all_native_families_preserved_in_order": True, "control_function_family_preserved": True,
            "auxiliary_exports_helper_added": helper_added,
            "auxiliary_exports_helper_ast_sha256": sha(dump(helpers[1][0]).encode()) if helper_added else None,
            "auxiliary_exports_helper_is_additive_registry_metadata": helper_added}


def capability_parts(raw, *, approved_import=None):
    tree = parsed(raw, "analysis_contracts")
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "capabilities"]
    if len(functions) != 1:
        raise ValueError("Capability metadata needs one capabilities() definition.")
    function = functions[0]
    returns = [n for n in function.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    if len(returns) != 1:
        raise ValueError("Capability metadata must retain its explicit return dictionary.")
    values = {}
    for key, value in zip(returns[0].value.keys, returns[0].value.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in values:
            raise ValueError("Capability metadata must retain unique literal top-level keys.")
        values[key.value] = value
    outside = dump(ast.Module(body=[n for n in tree.body if n is not function], type_ignores=[]))
    prelude = deepcopy(function)
    next(n for n in prelude.body if isinstance(n, ast.Return)).value = ast.Constant(value=None)
    imports = [n for n in prelude.body if isinstance(n, ast.ImportFrom)
               and n.module == "openecon.econometrics.registry"]
    if len(imports) != 1:
        raise ValueError("Capabilities must retain one original registry import.")
    added_import = any(alias.name == "auxiliary_exports" for alias in imports[0].names)
    if added_import:
        if approved_import is None or dump(imports[0]) != dump(approved_import):
            raise ValueError("Auxiliary capability import requires its exact approved --baseline-ref AST.")
        imports[0].names = [alias for alias in imports[0].names if alias.name != "auxiliary_exports"]
    return outside, dump(prelude), values, added_import


def pure_metadata(node):
    allowed = (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Constant, ast.Load,
               ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.FloorDiv, ast.Pow, ast.Mod, ast.UAdd, ast.USub)
    for part in ast.walk(node):
        if not isinstance(part, allowed):
            return False
        if isinstance(part, ast.Constant):
            value = part.value
            if (value is not None and not isinstance(value, (str, bool, int, float))) or (
                    isinstance(value, float) and not math.isfinite(value)):
                return False
    return True


def zero_argument_auxiliary_call(value):
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "auxiliary_exports" and not value.args and not value.keywords)


def capability_proof(native, current, *, baseline=None):
    imported = None if baseline is None else auxiliary_import(baseline)
    left, right = capability_parts(native), capability_parts(current, approved_import=imported)
    if left[:2] != right[:2]:
        raise ValueError("Capability executable logic changed outside its declarative return values.")
    before, after = left[2], right[2]
    if not before.keys() <= after.keys() or "control_function" not in before or "control_function" not in after:
        raise ValueError("Integrated capabilities dropped a native declaration or CF contract.")
    if dump(before["control_function"]) != dump(after["control_function"]):
        raise ValueError("Control-function capability value AST changed from the native pin.")
    changed = [key for key in before if dump(before[key]) != dump(after[key])]
    added = [key for key in after if key not in before]
    auxiliary_call_added = False
    for key in [*changed, *added]:
        if pure_metadata(after[key]):
            continue
        if (key != "auxiliary_exports" or baseline is None or key not in added
                or not zero_argument_auxiliary_call(after[key])):
            raise ValueError("Nonliteral capability additions require the specifically approved zero-argument auxiliary_exports call.")
        baseline_values = capability_parts(baseline, approved_import=imported)[2]
        value = baseline_values.get(key)
        if value is None or not zero_argument_auxiliary_call(value) or dump(value) != dump(after[key]):
            raise ValueError("Auxiliary capability call AST differs from the approved baseline.")
        auxiliary_call_added = True
    if auxiliary_call_added and not right[3]:
        raise ValueError("Auxiliary capability call needs the approved additive registry import.")
    return {"excluded_ast": "declarative capabilities() return values only",
            "outside_metadata_logic_ast_equal": True, "capabilities_prelude_ast_equal": True,
            "outside_metadata_logic_ast_sha256": sha(left[0].encode()),
            "capabilities_prelude_ast_sha256": sha(left[1].encode()),
            "control_function_value_ast_equal": True,
            "control_function_value_ast_sha256": sha(dump(before["control_function"]).encode()),
            "all_native_top_level_keys_preserved": True,
            "unchanged_native_keys": [key for key in before if key not in changed],
            "changed_declarative_keys": changed, "added_declarative_keys": added,
            "changed_values_are_pure_metadata_or_exact_baseline_auxiliary_call": True,
            "auxiliary_exports_import_added": right[3], "auxiliary_exports_call_added": auxiliary_call_added,
            "auxiliary_exports_call_ast_sha256": sha(dump(after["auxiliary_exports"]).encode()) if auxiliary_call_added else None}


def bridge(native_ref, integrated_ref=None, *, working_tree=False, baseline_ref=None):
    if working_tree == (integrated_ref is not None):
        raise ValueError("Choose exactly one committed integrated_ref or provisional working_tree.")
    native = resolve(native_ref)
    integrated = None if working_tree else resolve(integrated_ref)
    baseline = None if baseline_ref is None else resolve(baseline_ref)
    if baseline is not None and baseline not in APPROVED_METADATA_BASELINES:
        raise ValueError("Metadata exceptions are pinned only to the explicitly approved baselines " + ", ".join(APPROVED_METADATA_BASELINES))
    registry_baseline = None if baseline is None else committed(baseline, "src/openecon/econometrics/registry.py")
    contracts_baseline = None if baseline is None else committed(baseline, "src/openecon/analysis_contracts.py")
    public_baseline = (committed(baseline, "src/openecon/__init__.py")
                       if baseline == APPROVED_PUBLIC_EXPORT_BASELINE else None)
    if baseline == APPROVED_PUBLIC_EXPORT_BASELINE:
        original_registry = committed(APPROVED_METADATA_BASELINE, "src/openecon/econometrics/registry.py")
        original_contracts = committed(APPROVED_METADATA_BASELINE, "src/openecon/analysis_contracts.py")
        if (dump(auxiliary_helper(registry_baseline)) != dump(auxiliary_helper(original_registry))
                or dump(auxiliary_import(contracts_baseline)) != dump(auxiliary_import(original_contracts))):
            raise ValueError("The newer baseline must preserve the original approved auxiliary helper/import AST.")
    modules, manifest_hash = native_modules(native)
    records, snapshot = [], {}
    for module in modules:
        path = source_path(native, module)
        accepted = committed(native, path)
        current = (ROOT/path).read_bytes() if working_tree else committed(integrated, path)
        snapshot[path] = current
        record = {"module": module, "path": path, "native_sha256": sha(accepted),
                  "integrated_sha256": sha(current), "source_bytes_equal": current == accepted}
        if module not in METADATA_MODULES:
            if current != accepted:
                raise ValueError("Native computation source bytes changed: " + path)
            record["comparison"] = "exact source bytes"
        else:
            record["comparison"] = "strict original executable AST and specifically approved declarative metadata"
            if module == "openecon":
                record["metadata_proof"] = public_export_proof(accepted, current, baseline=public_baseline)
            elif module == "openecon.econometrics.registry":
                record["metadata_proof"] = registry_proof(accepted, current, baseline=registry_baseline)
            else:
                record["metadata_proof"] = capability_proof(accepted, current, baseline=contracts_baseline)
        records.append(record)
    proofs = {record["module"]: record.get("metadata_proof") for record in records if record["module"] in METADATA_MODULES}
    registry = proofs["openecon.econometrics.registry"]
    contracts = proofs["openecon.analysis_contracts"]
    if (contracts["auxiliary_exports_import_added"] or contracts["auxiliary_exports_call_added"]) and not registry["auxiliary_exports_helper_added"]:
        raise ValueError("Additive auxiliary capability metadata requires the exact approved registry helper.")
    if working_tree and any((ROOT/path).read_bytes() != raw for path, raw in snapshot.items()):
        raise ValueError("Working-tree dependency changed during bridge verification; retry after freeze.")
    return {"schema": "openecon.control_function.native_source_bridge.v1", "status": "pass",
            "native_ref": native, "integrated_ref": integrated,
            "baseline_ref": baseline,
            "auxiliary_metadata_exception_used": registry["auxiliary_exports_helper_added"] or contracts["auxiliary_exports_import_added"],
            "auxiliary_metadata_exception_scope": "Only the exact additive auxiliary_exports registry helper, exact capability import and exact zero-argument return value from the approved baseline; original executable logic remains equal.",
            "public_export_metadata_exception_used": proofs["openecon"]["survey_result_export_added"],
            "public_export_metadata_exception_scope": "Only the exact 6a1ab5c SurveyReplicateMarginsResult survey __all__ addition and literal module/class _EXPORTS binding; all original bindings, order and executable AST remain equal.",
            "source_mode": "working_tree_provisional" if working_tree else "committed",
            "committed_source_verified": not working_tree,
            "working_tree_head": resolve("HEAD") if working_tree else None,
            "native_dependency_manifest_sha256": manifest_hash,
            "native_dependency_module_count": len(modules),
            "exact_computation_module_count": len(modules)-len(METADATA_MODULES),
            "metadata_exception_module_count": len(METADATA_MODULES), "modules": records,
            "verifier_sha256": sha(Path(__file__).read_bytes()),
            "control_function_method_source_parity": True,
            "whole_integrated_app_byte_identity_claimed": False,
            "whole_integrated_app_native_acceptance_claimed": False,
            "public_release_readiness_claimed": False,
            "scope": "Only MARKET-506 through MARKET-513 conditional means and their declared native computation dependencies; parent MARKET-168 remains broader open scope.",
            "limits": ["Source parity alone does not establish actual GUI Run, independent numeric or restart acceptance; retain those separate native receipts.",
                       "Editor help, frontend assets and newly added unrelated families are outside this native method source bridge.",
                       "A working-tree bridge is provisional; final closure evidence requires --integrated-ref naming the committed resolved source."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-ref", default=DEFAULT_NATIVE)
    parser.add_argument("--baseline-ref", help="Required for exact approved metadata additions; allowed pins: " + ", ".join(APPROVED_METADATA_BASELINES))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--integrated-ref")
    inputs.add_argument("--working-tree", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    try:
        report = bridge(args.native_ref, args.integrated_ref, working_tree=args.working_tree, baseline_ref=args.baseline_ref)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Native CF source bridge refused: " + str(exc)) from exc
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "source_mode", "native_ref", "integrated_ref", "baseline_ref",
                                                 "auxiliary_metadata_exception_used", "public_export_metadata_exception_used", "native_dependency_module_count",
                                                 "exact_computation_module_count", "metadata_exception_module_count")}))


if __name__ == "__main__":
    main()
