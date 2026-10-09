"""Pinned, exhaustive PR91 source/API/option reconciliation; no old code execution.

Git comparisons and AST metadata inspection are read-only. Dispositions describe
source preservation, not newly executed scientific or installer acceptance.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OLD = "830b9c387fe980ebae159a0babad45b8c0370948"
GROUPS = {
    "mgarch": ("MARKET-131", 47, "docs/econometrics/mgarch.md", "tests/test_mgarch"),
    "regularized": ("MARKET-127", 33, "docs/econometrics/regularized.md", "tests/test_regularized"),
    "spatial": ("MARKET-124", 52, "docs/econometrics/spatial.md", "tests/test_spatial"),
    "robust": ("MARKET-132", 63, "docs/econometrics/robust.md", "tests/test_robust"),
    "tsworkflows": ("MARKET-129", 72, "docs/econometrics/tsworkflows.md", "tests/test_tsworkflows"),
    "common_factors": (
        "MARKET-128",
        34,
        "docs/econometrics/common-factors.md",
        "tests/test_econ_common_factors.py",
    ),
    "mi": ("MARKET-145", 31, "docs/econometrics/mi-pooling.md", "tests/test_mi_joint.py"),
    "multiple": (
        "MARKET-182",
        70,
        "docs/econometrics/multiple-testing.md",
        "tests/test_multiple_testing.py",
    ),
    "metadata": ("MARKET-516", 29, "docs/capabilities.md", "tests/test_capability_docs.py"),
}
RELOCATIONS = {
    ("src/openecon/econometrics/mi/pooling.py", "mi_pool"): (
        "src/openecon/econometrics/mi/joint.py",
        "mi_pool",
    ),
    ("src/openecon/econometrics/tsworkflows/sspace.py", "System"): (
        "src/openecon/econometrics/tsworkflows/ssmodel.py",
        "System",
    ),
    ("src/openecon/econometrics/tsworkflows/sspace.py", "System.__init__"): (
        "src/openecon/econometrics/tsworkflows/ssmodel.py",
        "System.__init__",
    ),
    ("src/openecon/econometrics/tsworkflows/sspace.py", "System.physical"): (
        "src/openecon/econometrics/tsworkflows/ssmodel.py",
        "System.physical",
    ),
    ("src/openecon/econometrics/tsworkflows/sspace.py", "System.values"): (
        "src/openecon/econometrics/tsworkflows/ssmodel.py",
        "System.values",
    ),
    ("src/openecon/econometrics/tsworkflows/sspace.py", "kalman"): (
        "src/openecon/econometrics/tsworkflows/ssengine.py",
        "kalman",
    ),
}
UNIQUE = {
    ("scripts/generate_editor_api.py", "literal_bindings"),
    ("src/openecon/econometrics/registry.py", "auxiliary_exports"),
    (
        "tests/test_editor_api_catalog.py",
        "test_estimator_comprehensions_reject_executable_iterators",
    ),
    (
        "tests/test_editor_api_catalog.py",
        "test_bounded_manifest_comprehensions_bind_literal_pairs_and_string_entries",
    ),
}


def git(*args, binary=False):
    return (
        subprocess.check_output(["git", *args], cwd=ROOT, text=not binary).strip()
        if not binary
        else subprocess.check_output(["git", *args], cwd=ROOT)
    )


@lru_cache(maxsize=1024)
def blob(ref, path):
    result = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def definitions(data):
    if data is None:
        return {}
    tree = ast.parse(data)
    result = {}

    def walk(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                result[name] = node
                if isinstance(node, ast.ClassDef):
                    walk(node.body, name + ".")

    walk(tree.body)
    return result


def signature(node):
    return (
        ast.unparse(node.args)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )


def parameter_names(node):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    return [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]


def group(path):
    if (
        "common_factors" in path
        or path.endswith("/panel/cce.py")
        or path.endswith("common-factors.md")
    ):
        return "common_factors"
    if "/mi/" in path or path.endswith("/mi.md") or path.endswith("test_mi_pooling.py"):
        return "mi"
    if "multiple" in path or path.endswith("/postest/__init__.py"):
        return "multiple"
    for name in ("mgarch", "regularized", "spatial", "robust", "tsworkflows"):
        if (
            name in path
            or {"mgarch": "market-131", "spatial": "market-124", "robust": "market-132"}.get(
                name, "@none"
            )
            in path
        ):
            return name
    return "metadata"


def reason(path, relation, category):
    if relation == "exact_bytes":
        return "Old-head bytes are preserved exactly in pinned main, including their original limitations."
    if path.startswith(
        ("docs/evidence/capabilities-completion-", "docs/evidence/capability-source-integration-")
    ):
        return "Unique historical/interrupted-run receipt retained at immutable old-head link and in local binary patch; not promoted to current acceptance."
    if path.endswith(("capability-workstreams.json", "capability-implementation-status.md")):
        return "Original 51-workstream scope is preserved verbatim in the historical companion; current canonical GitHub issues remain authoritative."
    if path.endswith("/mi.md") or path.endswith("test_mi_pooling.py"):
        return "Superseded by typed persisted MI generation/pooling/joint state and current diagnostic/joint tests; old budgets are not promises of current support."
    if path.startswith(("docs/evidence/", "reports/")):
        return "Dated proof is compared by hash and retained at its original revision; newer receipt bytes do not retroactively verify the old branch."
    if path.startswith("tests/"):
        return "Existing checks preserved or expanded for current contracts; removed unique parser canaries are restored in dedicated reconciliation tests."
    if path.endswith("sspace.py"):
        return "New scheduled/masked filter, smoother and complete state replace old internals; System/kalman relocated. Old missing-measurement docstring alone was not implementation."
    if path.endswith(
        (
            "registry.py",
            "analysis_contracts.py",
            "generate_editor_api.py",
            "generate_capability_docs.py",
        )
    ):
        return "Current catalogue/state retained; unique Torch-free auxiliary inventory and bounded static manifest syntax are restored narrowly, without old family rollback."
    if path.endswith(("console_worker.py", "output_latex.py")):
        return "Forecast notes retained; newer compact plot transport/decoding replaces old serialization rather than being reverted."
    return f"Current {category} source/guide retains the older bounded domain with later extensions; exact symbol and option differences are recorded below."


def build(main_ref):
    current = git("rev-parse", main_ref)
    base = git("merge-base", OLD, current)
    changed = git("diff", "--name-only", base, OLD).splitlines()
    if len(changed) != 119:
        raise ValueError(f"Expected the complete pinned 119-path draft, got {len(changed)}")
    rows = []
    for path in changed:
        old, new, initial = blob(OLD, path), blob(current, path), blob(base, path)
        category = group(path)
        issue, github, guide, test = GROUPS[category]
        if "panel_mmqr" in path or "nlswork-mmqr" in path:
            github = 68
        elif category == "tsworkflows" and "midas" in path:
            github = 45
        elif category == "tsworkflows" and "sspace" in path:
            github = 76
        elif category == "tsworkflows" and "ets" in path:
            github = 49
        relation = (
            "exact_bytes" if old == new else "absent_in_main" if new is None else "changed_in_main"
        )
        symbols = []
        if path.endswith(".py"):
            before, after = definitions(old), definitions(new)
            for name, node in before.items():
                target_path, target_name = RELOCATIONS.get((path, name), (path, name))
                target = (
                    definitions(blob(current, target_path)).get(target_name)
                    if target_path != path
                    else after.get(name)
                )
                if (path, name) in UNIQUE:
                    disposition = "unique_preserved_in_reconciliation_patch"
                elif target is None:
                    disposition = (
                        "superseded_test_module"
                        if path.endswith("test_mi_pooling.py")
                        else "missing_requires_review"
                    )
                elif ast.dump(node, include_attributes=False) == ast.dump(
                    target, include_attributes=False
                ):
                    disposition = "exact_ast_preserved"
                else:
                    disposition = "relocated_or_newer_contract"
                symbols.append(
                    {
                        "name": name,
                        "old_line": node.lineno,
                        "old_signature": signature(node),
                        "old_ast_sha256": sha(ast.dump(node, include_attributes=False).encode()),
                        "main_target_path": target_path,
                        "main_target_name": target_name,
                        "main_line": target.lineno if target else None,
                        "main_signature": signature(target) if target else None,
                        "main_ast_sha256": sha(ast.dump(target, include_attributes=False).encode())
                        if target
                        else None,
                        "disposition": disposition,
                    }
                )
        disposition = (
            "already_integrated"
            if relation == "exact_bytes"
            else "unique_historical_artifact_preserved"
            if relation == "absent_in_main" and path.startswith("docs/evidence/")
            else "historical_scope_preserved"
            if path.endswith(("capability-workstreams.json", "capability-implementation-status.md"))
            else "superseded_by_current_implementation"
        )
        rows.append(
            {
                "path": path,
                "relation": relation,
                "disposition": disposition,
                "old_sha256": sha(old),
                "base_sha256": sha(initial),
                "main_sha256": sha(new),
                "old_url": f"https://github.com/bluearf/openecon/blob/{OLD}/{path}",
                "main_url": f"https://github.com/bluearf/openecon/blob/{current}/{path}"
                if new is not None
                else None,
                "canonical_linear": f"https://linear.app/bluearf/issue/{issue}",
                "canonical_github": f"https://github.com/bluearf/openecon/issues/{github}",
                "guide": guide,
                "existing_validation_source": test,
                "rationale": reason(path, relation, category),
                "symbols": symbols,
            }
        )
    parser = runpy.run_path(str(ROOT / "scripts/generate_editor_api.py"))
    old_exports = parser["registry_exports"](parser["SourceReader"](OLD))
    current_exports = parser["registry_exports"](parser["SourceReader"](current))
    apis = []
    for name, target in sorted(old_exports.items()):
        later = current_exports.get(name)
        if later is None:
            raise ValueError(f"Unpreserved public API: {name}")
        old_path, current_path = (
            target[0].replace(".", "/") + ".py",
            later[0].replace(".", "/") + ".py",
        )
        a = definitions(blob(OLD, "src/" + old_path)).get(target[1])
        b = definitions(blob(current, "src/" + current_path)).get(later[1])
        removed_parameters = (
            sorted(set(parameter_names(a)) - set(parameter_names(b))) if a and b else []
        )
        if removed_parameters:
            raise ValueError(
                f"Unclassified removed public parameters: {name}: {removed_parameters}"
            )
        apis.append(
            {
                "name": name,
                "old_target": ":".join(target[:2]),
                "current_target": ":".join(later[:2]),
                "old_signature": signature(a) if a else None,
                "current_signature": signature(b) if b else None,
                "old_return_annotation": ast.unparse(a.returns)
                if a and getattr(a, "returns", None)
                else None,
                "current_return_annotation": ast.unparse(b.returns)
                if b and getattr(b, "returns", None)
                else None,
                "added_parameters": sorted(set(parameter_names(b)) - set(parameter_names(a)))
                if a and b
                else [],
                "removed_parameters": removed_parameters,
                "disposition": "registration_preserved_same_target"
                if target[:2] == later[:2]
                else "registration_preserved_relocated_target",
            }
        )
    old_inventory = json.loads(blob(OLD, "docs/econometrics/capabilities.generated.json"))
    current_inventory = json.loads(blob(current, "docs/econometrics/capabilities.generated.json"))
    options = []
    for name, value in sorted(old_inventory["estimators"].items()):
        later = current_inventory["estimators"].get(name)
        if later is None:
            raise ValueError(f"Unpreserved estimator: {name}")
        for option, contract in value.get("options", {}).items():
            if option not in later.get("options", {}):
                raise ValueError(f"Unclassified removed option: {name}.{option}")
            removed_choices = set(map(str, contract.get("choices", []))) - set(
                map(str, later["options"][option].get("choices", []))
            )
            if removed_choices:
                raise ValueError(
                    f"Unclassified removed option choices: {name}.{option}: {removed_choices}"
                )
        if set(value.get("covariances", [])) - set(later.get("covariances", [])):
            raise ValueError(f"Unclassified removed covariance kind: {name}")
        options.append(
            {
                "estimator": name,
                "old_contract": value,
                "current_contract": later,
                "old_dataset": name in old_inventory["streaming"]["estimators"],
                "current_dataset": name in current_inventory["streaming"]["estimators"],
                "disposition": "same_source_option_contract"
                if value == later
                else "newer_source_option_contract_requires_guide_conditions",
            }
        )
    missing = [
        (r["path"], s["name"])
        for r in rows
        for s in r["symbols"]
        if s["disposition"] == "missing_requires_review"
    ]
    if missing:
        raise ValueError(f"Unclassified old symbols: {missing}")
    old_dataset = set(old_inventory["streaming"]["estimators"])
    old_prediction = set(old_inventory["postestimation"]["saved_prediction"]["estimators"])
    if old_dataset - set(current_inventory["streaming"]["estimators"]):
        raise ValueError("An old Dataset registration is missing")
    if old_prediction - set(current_inventory["postestimation"]["saved_prediction"]["estimators"]):
        raise ValueError("An old common saved-prediction registration is missing")
    legacy = json.loads(blob(OLD, "docs/econometrics/capability-workstreams.json"))
    return {
        "schema": 1,
        "issue": "MARKET-520",
        "old_pr": "https://github.com/bluearf/openecon/pull/91",
        "old_head": OLD,
        "comparison_main": current,
        "merge_base": base,
        "counts": {
            "paths": len(rows),
            "file_relations": dict(Counter(row["relation"] for row in rows)),
            "old_public_registrations": len(apis),
            "current_public_registrations": len(current_exports),
            "old_registered_estimator_option_contracts": len(options),
            "historical_workstreams": len(legacy["workstreams"]),
            "unclassified_symbols": 0,
        },
        "files": rows,
        "public_apis": apis,
        "estimator_options": options,
        "routing_contracts": {
            "old_dataset_conditions": old_inventory["streaming"]["conditions"],
            "current_dataset_conditions": current_inventory["streaming"]["conditions"],
            "old_common_prediction": old_inventory["postestimation"]["saved_prediction"],
            "current_common_prediction": current_inventory["postestimation"]["saved_prediction"],
            "registration_loss": False,
            "domain_boundary": "Registration preservation is not all-option or all-data-geometry coverage; current conditions remain authoritative.",
        },
        "historical_workstreams": legacy,
        "unique_patch": [
            "Torch-free auxiliary_exports and capability inventory",
            "bounded static tuple/comprehension bindings and strict EstimatorInfo field binding",
            "dedicated no-execution/over-budget/duplicate-constructor metadata regressions",
        ],
        "preservation": "Original draft commit/branch retained; complete binary Git diff also saved locally in ignored artifacts. Do not delete the original branch during retirement.",
        "retirement_gate": "After reviewed metadata patch and matrix merge, parent may close PR91 as superseded with replacement links, retaining its branch; no wholesale merge.",
        "evidence_boundary": "Source/AST/option registration reconciliation only; existing guide/test/receipt links retain their own numeric, installed and public evidence scope. No new scientific, vendor, native-app or public-release execution is claimed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.main_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps(value["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
