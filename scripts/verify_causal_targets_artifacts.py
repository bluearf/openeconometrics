"""Compare all eight complete causal-target artifacts across four evidence layers.

This reads saved results and renders their restored tables. It does not fit a
model, regenerate an artifact, execute a native console or establish UI proof.
Only clearly named Torch-version metadata may be normalized for a separately
reported comparison. All numerical or other structural differences fail.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from openecon.econometrics.causal_design import common as codec  # noqa: E402
from openecon.econometrics.core import TableSet  # noqa: E402
from openecon.frame import DataFrame  # noqa: E402


NAMES = (
    "ovb_benchmark",
    "ovb_robustness",
    "treatment_cdf_ipw",
    "treatment_quantile_ipw",
    "treatment_cdf_aipw",
    "treatment_survival_ipcw",
    "treatment_rmst_ipcw",
    "treatment_rmst_aipw",
)
LAYERS = ("source", "wheel", "frozen", "native")
MAX_SUMMARY = 100
ABSENT = object()
TORCH_VERSION = re.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?[a-zA-Z0-9.+_-]*\Z")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=invalid_constant,
    )


def pointer(path):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


def preview(value):
    if value is ABSENT:
        return {"type": "absent"}
    serialized = canonical(value)
    return dict(
        type=type(value).__name__,
        canonical_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        preview=serialized[:240],
        preview_truncated=len(serialized) > 240,
    )


def differences(left, right, path=()):
    """Lossless difference paths; primitive comparison retains JSON types/-0.0."""
    if left is ABSENT or right is ABSENT:
        yield path, "presence", left, right
    elif type(left) is not type(right):
        yield path, "type", left, right
    elif isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            yield from differences(left.get(key, ABSENT), right.get(key, ABSENT), (*path, key))
    elif isinstance(left, list):
        if len(left) != len(right):
            yield (*path, "#length"), "length", len(left), len(right)
        for index in range(max(len(left), len(right))):
            yield from differences(
                left[index] if index < len(left) else ABSENT,
                right[index] if index < len(right) else ABSENT,
                (*path, index),
            )
    elif canonical(left) != canonical(right):
        yield path, "value", left, right


def version_metadata(path, left, right):
    # A source column named torch_version, scientific schema version or an
    # arbitrary string elsewhere is never a provenance exclusion.
    if path[:3] != ("payload", "attrs", "state") or len(path) < 5:
        return False
    permitted = (
        path[-1] in {"torch_version", "pytorch_version"}
        and path[-2] in {"rng", "versions", "provenance", "runtime"}
    ) or (path[-1] == "torch" and path[-2] == "versions")
    if not permitted:
        return False
    return all(
        value is ABSENT or (isinstance(value, str) and TORCH_VERSION.fullmatch(value))
        for value in (left, right)
    )


def remove_path(value, path):
    current = value
    for part in path[:-1]:
        if isinstance(current, dict):
            if part not in current:
                return
            current = current[part]
        elif isinstance(current, list) and isinstance(part, int) and part < len(current):
            current = current[part]
        else:
            return
    if isinstance(current, dict):
        current.pop(path[-1], None)


def normalize_version_only(artifact, paths):
    value = copy.deepcopy(artifact)
    for path in paths:
        remove_path(value, path)
    # These two integrity fields are derived from the fully validated input.
    # Recompute them after the single explicit metadata exclusion; do not erase
    # numerical values, table hashes, unknown nested digests or other metadata.
    if paths:
        attrs = value["payload"]["attrs"]
        attrs["state_sha256"] = fingerprint(attrs["state"])
        value["sha256"] = fingerprint(value["payload"])
    return value


def latex_check(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty restored-table LaTeX")
    environments = {kind: text.count("\\begin{" + kind + "}") for kind in ("tabular", "longtable")}
    if not sum(environments.values()):
        raise ValueError("Restored LaTeX has no complete table environment")
    for kind, count in environments.items():
        if count != text.count("\\end{" + kind + "}"):
            raise ValueError(f"Unbalanced LaTeX {kind} environment")
    return dict(
        bytes=len(text.encode()),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        table_environments=environments,
    )


def load_artifact(path, name):
    artifact = read_json(path)
    restored = codec.causal_design_load(artifact)
    if type(restored) is not TableSet or restored.attrs.get("procedure") != name:
        raise ValueError("Artifact filename/procedure/result type mismatch")
    if restored.attrs.get("stata_parity_validated") is not False:
        raise ValueError("Unexpected blanket parity flag")
    roundtrip = codec.causal_design_save(restored)
    if canonical(roundtrip) != canonical(artifact):
        raise ValueError("Complete typed save/load roundtrip differs")
    tables = {}
    for key, table in restored.items():
        if type(table) is not DataFrame:
            raise ValueError(f"{key} is not an OpenEcon DataFrame")
        tables[key] = dict(
            type=f"{type(table).__module__}.{type(table).__name__}",
            shape=list(table.shape),
            dtypes=[str(item) for item in table.dtypes],
            index_type=type(table.index).__name__,
            columns_type=type(table.columns).__name__,
            latex=latex_check(str(table.to_latex())),
        )
    latex = latex_check(restored.to_latex())
    if sum(latex["table_environments"].values()) != len(restored):
        raise ValueError("Complete result LaTeX does not contain every restored table")
    record = dict(
        path=str(path),
        file_sha256=file_hash(path),
        canonical_json_sha256=fingerprint(artifact),
        payload_sha256=artifact["sha256"],
        state_sha256=restored.attrs["state_sha256"],
        tables_sha256=restored.attrs["tables_sha256"],
        table_count=len(restored),
        table_order=list(restored),
        complete_typed_roundtrip_equal=True,
        all_tables_open_econ_dataframe=True,
        all_tables_have_complete_latex=True,
        full_result_latex=latex,
        table_schemas=tables,
    )
    return artifact, restored, record


def independent_tables(left, right):
    checks, errors = {}, {}
    if list(left) != list(right):
        return False, {}, {"table_order": "Restored table names/order differ"}
    for name in left:
        try:
            pd.testing.assert_frame_equal(
                left[name],
                right[name],
                check_exact=True,
                check_dtype=True,
                check_frame_type=True,
                check_index_type=True,
                check_column_type=True,
            )
            checks[name] = True
        except AssertionError as exc:
            checks[name] = False
            errors[name] = str(exc)[:800]
    return all(checks.values()), checks, errors


def compare(left, left_result, right, right_result, layer, name, difference_file):
    counts, summary, version_paths, version_records = {}, [], [], []
    total = 0
    for path, kind, before, after in differences(left, right):
        provenance = version_metadata(path, before, after)
        record = dict(
            layer=layer,
            procedure=name,
            path=pointer(path),
            kind=kind,
            demonstrable_torch_version_metadata=provenance,
            source=preview(before),
            candidate=preview(after),
        )
        difference_file.write(canonical(record) + "\n")
        total += 1
        counts[kind] = counts.get(kind, 0) + 1
        if len(summary) < MAX_SUMMARY:
            summary.append(record)
        if provenance:
            version_paths.append(path)
            version_records.append(record)
    left_normal = normalize_version_only(left, version_paths)
    right_normal = normalize_version_only(right, version_paths)
    state_equal = canonical(left_normal["payload"]["attrs"]["state"]) == canonical(
        right_normal["payload"]["attrs"]["state"]
    )
    complete_equal = canonical(left) == canonical(right)
    excluded_equal = canonical(left_normal) == canonical(right_normal)
    table_payload_equal = canonical(left["payload"]["tables"]) == canonical(
        right["payload"]["tables"]
    )
    typed_tables_equal, table_checks, table_errors = independent_tables(left_result, right_result)
    independent_equal = typed_tables_equal and table_payload_equal and state_equal
    provenance_only = (
        not complete_equal and bool(version_paths) and excluded_equal and independent_equal
    )
    status = (
        "exact"
        if complete_equal and independent_equal
        else "provenance_only"
        if provenance_only
        else "different"
    )
    return dict(
        status=status,
        complete_canonical_json_exact=complete_equal,
        all_fields_exact_after_torch_version_only=excluded_equal,
        independent_tables_and_state_exact_excluding_only_torch_version=independent_equal,
        complete_table_payload_exact=table_payload_equal,
        restored_table_types_dtypes_index_columns_values_exact=typed_tables_equal,
        restored_table_checks=table_checks,
        restored_table_errors=table_errors,
        complete_state_exact_excluding_only_torch_version=state_equal,
        torch_version_metadata_differences=version_records,
        normalized_derived_integrity_fields=["/sha256", "/payload/attrs/state_sha256"]
        if version_paths
        else [],
        source_normalized_canonical_sha256=fingerprint(left_normal),
        candidate_normalized_canonical_sha256=fingerprint(right_normal),
        difference_count=total,
        difference_kinds=counts,
        difference_summary=summary,
        difference_summary_limit=MAX_SUMMARY,
        difference_summary_truncated=total > MAX_SUMMARY,
        every_complete_difference_path_in_sidecar=True,
    )


def verify(directories, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(output.stem + "-differences.jsonl")
    report = dict(
        schema="openecon.causal_targets.comparison.v1",
        names=list(NAMES),
        evidence_scope="Saved artifact validation/comparison only; no scientific fits, artifact regeneration or native UI execution.",
        input_directory_count=len(set(directories.values())),
        four_distinct_input_directories=len(set(directories.values())) == len(LAYERS),
        execution_layer_identity_established_by_comparator=False,
        provenance_exclusion="Only named string Torch-version metadata inside scientific-state rng/versions/provenance/runtime contexts; derived validated top-level/state hashes recomputed afterward.",
        verification_environment=dict(
            python=sys.version.split()[0],
            torch=torch.__version__,
            codec_file=str(Path(codec.__file__).resolve()),
            codec_sha256=file_hash(Path(codec.__file__)),
            verifier_sha256=file_hash(Path(__file__)),
        ),
        layers={},
        comparisons={layer: {} for layer in LAYERS if layer != "source"},
        errors=[],
    )
    expected = {name + ".json" for name in NAMES}
    admitted = {}
    for layer, directory in directories.items():
        files = {path.name for path in directory.glob("*.json")} if directory.is_dir() else set()
        valid = files == expected and directory.is_dir()
        report["layers"][layer] = dict(
            directory=str(directory),
            exactly_eight_named_json_files=valid,
            missing_files=sorted(expected - files),
            unexpected_json_files=sorted(files - expected),
            files={},
        )
        admitted[layer] = valid
        if not valid:
            report["errors"].append(
                dict(
                    layer=layer,
                    error="Directory must contain exactly the eight named complete JSON artifacts",
                )
            )
    with sidecar.open("w", encoding="utf-8") as difference_file:
        for name in NAMES:
            loaded = {}
            for layer, directory in directories.items():
                if not admitted[layer]:
                    continue
                try:
                    artifact, restored, record = load_artifact(directory / (name + ".json"), name)
                    report["layers"][layer]["files"][name] = record
                    loaded[layer] = (artifact, restored)
                except Exception as exc:
                    report["errors"].append(
                        dict(
                            layer=layer, procedure=name, error=f"{type(exc).__name__}: {exc}"[:1000]
                        )
                    )
            if "source" in loaded:
                for layer in LAYERS[1:]:
                    if layer in loaded:
                        report["comparisons"][layer][name] = compare(
                            *loaded["source"], *loaded[layer], layer, name, difference_file
                        )
    comparisons = [value for files in report["comparisons"].values() for value in files.values()]
    complete = len(comparisons) == 3 * len(NAMES) and not report["errors"]
    exact = complete and all(item["status"] == "exact" for item in comparisons)
    scientific_equal = complete and all(
        item["status"] in {"exact", "provenance_only"} for item in comparisons
    )
    report.update(
        status="passed_exact"
        if exact
        else "passed_provenance_only"
        if scientific_equal
        else "failed",
        all_32_complete_artifacts_validated=complete,
        full_canonical_json_exact_all_layers=exact,
        independent_tables_state_and_structure_exact_excluding_only_torch_version=scientific_equal,
        numerical_or_structural_differences_automatically_excused=False,
        complete_difference_paths=dict(
            path=str(sidecar),
            sha256=file_hash(sidecar),
            count=sum(item["difference_count"] for item in comparisons),
        ),
    )
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for layer in LAYERS:
        parser.add_argument("--" + layer + "-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    directories = {layer: getattr(args, layer + "_dir").resolve() for layer in LAYERS}
    report = verify(directories, args.output.resolve())
    print(
        json.dumps(
            dict(
                status=report["status"],
                all_32_complete_artifacts_validated=report["all_32_complete_artifacts_validated"],
                full_canonical_json_exact_all_layers=report["full_canonical_json_exact_all_layers"],
                independent_tables_state_and_structure_exact_excluding_only_torch_version=report[
                    "independent_tables_state_and_structure_exact_excluding_only_torch_version"
                ],
                complete_difference_path_count=report["complete_difference_paths"]["count"],
            )
        )
    )
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
