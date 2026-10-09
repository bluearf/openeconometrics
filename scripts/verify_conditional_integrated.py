"""Prove scoped conditional source/native identity after independent main additions.

The QA bundle remains pinned to its original whole-module baseline. Scientific
modules must be bytecode-identical to current source. Registry additions and
non-conditional capability metadata may differ, with no changed executable logic.
A current-source execution must reproduce every complete native artifact.
"""

from __future__ import annotations
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tarfile
import tempfile

from verify_conditional_runtime import MODULES, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/conditional-eight-2026-10-07"


def shape(tree):
    return ast.dump(tree, include_attributes=False)


def metadata_difference(name, baseline, current):
    left, right = ast.parse(baseline), ast.parse(current)
    if name == "econometrics.registry":

        def families(tree):
            return next(
                node
                for node in tree.body
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "FAMILIES"
            )

        a, b = families(left), families(right)
        old, new = ast.literal_eval(a.value), ast.literal_eval(b.value)
        assert len(new) == len(set(new)) and set(old) <= set(new)
        assert tuple(value for value in new if value in old) == old
        added = sorted(set(new) - set(old))
        assert added and "conditional" not in added
        a.value = b.value = ast.Constant(value=None)
        assert shape(left) == shape(right), "Registry executable logic changed"
        return {"added_families": added, "all_other_registry_ast_equal": True}
    if name == "__init__":

        def exports(tree):
            return next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in node.targets)
            )

        a, b = exports(left), exports(right)

        def literal_fields(mapping):
            return {
                ast.literal_eval(k): v
                for k, v in zip(mapping.keys, mapping.values)
                if k is not None
            }

        old, new = literal_fields(a.value), literal_fields(b.value)
        added = sorted(set(new) - set(old))
        assert (
            added
            and set(old) <= set(new)
            and all(shape(new[k]) == shape(v) for k, v in old.items())
        )
        targets = {k: ast.literal_eval(new[k]) for k in added}
        assert all(
            k[0].isupper()
            and len(targets[k]) == 2
            and targets[k][0].startswith("openecon.econometrics.")
            and targets[k][1] == k
            for k in added
        )

        def all_literals(tree):
            names = []
            for node in tree.body:
                if (
                    isinstance(node, ast.AugAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "__all__"
                ):
                    if isinstance(node.value, ast.List):
                        names.extend(ast.literal_eval(node.value))
            return names

        old_all, new_all = all_literals(left), all_literals(right)
        assert set(new_all) - set(old_all) == set(added) and set(old_all) <= set(new_all)
        # Strip ONLY added literal class exports/names, then require the entire
        # remaining module AST (including all import/getattr logic) to be equal.
        kept_fields = [
            (k, v)
            for k, v in zip(b.value.keys, b.value.values)
            if k is None or ast.literal_eval(k) not in added
        ]
        b.value.keys = [k for k, _ in kept_fields]
        b.value.values = [v for _, v in kept_fields]
        body = []
        for node in right.body:
            if (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
                and isinstance(node.value, ast.List)
            ):
                kept = [value for value in node.value.elts if ast.literal_eval(value) not in added]
                if not kept:
                    continue
                node.value.elts = kept
            body.append(node)
        right.body = body
        assert shape(left) == shape(right), "Root export executable logic changed"
        return {"added_literal_class_exports": added, "all_other_root_ast_equal": True}
    assert name == "analysis_contracts"

    def declaration(tree):
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "capabilities"
        )
        result = fn.body[-1]
        assert isinstance(result, ast.Return) and isinstance(result.value, ast.Dict)
        return result.value

    a, b = declaration(left), declaration(right)

    def values(mapping):
        assert all(
            isinstance(key, ast.Constant) and isinstance(key.value, str) for key in mapping.keys
        )
        return {key.value: value for key, value in zip(mapping.keys, mapping.values)}

    av, bv = values(a), values(b)
    changed = sorted(
        key
        for key in set(av) | set(bv)
        if key not in av or key not in bv or shape(av[key]) != shape(bv[key])
    )
    assert (
        changed
        and "conditional_regression" not in changed
        and shape(av["conditional_regression"]) == shape(bv["conditional_regression"])
    )
    for mapping in (a, b):
        keep = [
            (key, value)
            for key, value in zip(mapping.keys, mapping.values)
            if key.value not in changed
        ]
        mapping.keys, mapping.values = [key for key, _ in keep], [value for _, value in keep]
    assert shape(left) == shape(right), "Capability executable logic changed"
    return {
        "changed_non_conditional_metadata_fields": changed,
        "conditional_contract_ast_equal": True,
        "all_other_contract_ast_equal": True,
    }


def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader

    baseline = json.loads((EVIDENCE / "acceptance.json").read_text())["source_commit"]
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    unchanged, metadata = {}, {}
    for name in MODULES:
        relative = "src/openecon/" + name.replace(".", "/") + ".py"
        before = subprocess.check_output(
            ["git", "show", baseline + ":" + relative], cwd=ROOT, text=True
        )
        path = ROOT / relative
        after = path.read_text()
        module = ("openecon." + name).removesuffix(".__init__")
        bundled = normalized(archive.extract(module))
        assert bundled == normalized(compile(before, relative, "exec", dont_inherit=True)), module
        if before == after:
            assert bundled == normalized(compile(after, str(path), "exec", dont_inherit=True)), (
                module
            )
            unchanged[module] = digest(path)
        else:
            metadata[module] = metadata_difference(name, before, after)
    assert set(metadata) <= {
        "openecon",
        "openecon.analysis_contracts",
        "openecon.econometrics.registry",
    }
    native = json.loads((EVIDENCE / "conditional-native-run.json").read_text())
    assert digest(runtime) == native["runtime_sha256"]
    assert native["restart_readback_unchanged"]
    with tempfile.TemporaryDirectory(prefix="conditional-merged-source-") as temporary:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(
                str(ROOT / "docs/examples/conditional_eight.py"),
                init_globals={"display": lambda _: None, "ARTIFACT_DIRECTORY": temporary},
            )
        marker = "CONDITIONAL_EIGHT_OK "
        proof = json.loads(
            next(
                line[len(marker) :]
                for line in stdout.getvalue().splitlines()
                if line.startswith(marker)
            )
        )
        assert not proof["frozen"] and proof["complete_artifacts_equal"]
        assert proof["artifact_sha256"] == native["proof"]["artifact_sha256"]
        compared = []
        with tarfile.open(EVIDENCE / "complete-artifacts.tar.gz") as saved:
            members = saved.getmembers()
            assert len(members) == 8 and len({m.name for m in members}) == 8
            for member in members:
                assert member.isfile() and Path(member.name).name == member.name
                assert json.loads(saved.extractfile(member).read()) == json.loads(
                    (Path(temporary) / member.name).read_text()
                )
                compared.append(member.name)
    return {
        "status": "passed",
        "bundle_baseline_source": baseline,
        "current_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "whole_compiled_baseline_modules_verified": len(MODULES),
        "current_complete_compiled_modules_unchanged": unchanged,
        "metadata_only_differences": metadata,
        "complete_current_source_native_artifacts_equal": sorted(compared),
        "runtime_sha256": digest(runtime),
        "verifier_sha256": digest(__file__),
        "qa_bundle_contains_new_other_families": False,
        "human_data_access": False,
        "public_release_delivered": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Choose a new receipt"
    result = verify(args.runtime.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "current_complete_modules_unchanged": len(
                    result["current_complete_compiled_modules_unchanged"]
                ),
                "metadata_only_modules": list(result["metadata_only_differences"]),
                "complete_artifacts_equal": len(
                    result["complete_current_source_native_artifacts_equal"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
