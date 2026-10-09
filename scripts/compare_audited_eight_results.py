"""Compare complete runtime artifacts using validated, path-specific provenance."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
from uuid import UUID


class ComparisonError(ValueError):
    """An artifact, state or provenance reference could not be proved equal."""


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def decode_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ComparisonError(f"Nonfinite JSON constant: {value}")

    def finite_float(token):
        value = float(token)
        require(math.isfinite(value), f"Nonfinite JSON exponent: {token}")
        return value

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                      parse_float=finite_float)


def load_json(path):
    return decode_json(Path(path).read_text())


def valid_uuid(value):
    try:
        require(isinstance(value, str) and str(UUID(value)) == value,
                "Run IDs must be canonical UUID strings")
    except (ValueError, AttributeError) as exc:
        raise ComparisonError("Run IDs must be canonical UUID strings") from exc


def valid_time(value):
    try:
        require(isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None,
                "Run creation timestamps must have an ISO timezone")
    except ValueError as exc:
        raise ComparisonError("Invalid run creation timestamp") from exc


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                   separators=(",", ":")).encode()).hexdigest()


class Comparison:
    def __init__(self, source_receipt, installed_receipt):
        self.receipt_paths = tuple(Path(p) for p in (source_receipt, installed_receipt))
        self.receipts = tuple(load_json(p) for p in self.receipt_paths)
        self.folders = tuple(Path(r["output_directory"]).resolve() for r in self.receipts)
        self.files = sorted(self.receipts[0]["files"])
        require(set(self.files) == set(self.receipts[1]["files"]), "Artifact file rosters differ")
        self.allowed = {}
        self.references = []
        self.metadata = {}
        self.maps = {kind: {} for kind in ("UUID", "timestamp", "state digest")}
        self.reverse_maps = {kind: {} for kind in self.maps}
        self.uuid_anchors = {kind: ({}, {}) for kind in ("bundle", "origin")}
        self.bundle_count = self.origin_count = self.mi_count = 0
        self.json_pairs = {}
        self.artifact_references = []
        for folder, receipt in zip(self.folders, self.receipts, strict=True):
            present = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
            require(present == set(self.files), "Output directory and complete receipt file roster differ")
            for filename in self.files:
                path = self.file_path(folder, filename)
                record = receipt["files"][filename]
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == record["sha256"],
                        f"Actual artifact SHA differs: {filename}")
                if "bytes" in record:
                    require(len(raw) == record["bytes"], f"Actual artifact byte count differs: {filename}")
        for filename in self.files:
            if filename.endswith(".json"):
                pair = tuple(load_json(folder/filename) for folder in self.folders)
                self.json_pairs[filename] = pair
                self.scan(*pair, (filename,))
        for path, first, second, kind in self.references:
            require(self.maps[kind].get(second) == first,
                    f"Unrecognized {kind} reference at {self.location(path)}")
            self.allow(path, first, second, f"reference to validated {kind}")
        for filename, pair in self.json_pairs.items():
            if filename.endswith("receipt.json"):
                self.scan_artifact_references(*pair, (filename,))

    @staticmethod
    def location(path):
        return "/".join(map(str, path))

    @staticmethod
    def file_path(folder, filename):
        relative = PurePosixPath(filename)
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Unsafe artifact path: {filename}")
        path = (folder/filename).resolve()
        require(path.is_relative_to(folder), f"Artifact escapes output directory: {filename}")
        return path

    def allow(self, path, first, second, reason):
        record = (first, second, reason)
        require(path not in self.allowed or self.allowed[path] == record,
                f"Ambiguous provenance path at {self.location(path)}")
        self.allowed[path] = record

    def bind(self, first, second, kind):
        previous, reverse = self.maps[kind], self.reverse_maps[kind]
        require(second not in previous or previous[second] == first, f"Ambiguous {kind} mapping")
        require(first not in reverse or reverse[first] == second, f"Ambiguous reverse {kind} mapping")
        previous[second], reverse[first] = first, second

    def anchor(self, saved, side, kind, identity_key):
        identities = self.uuid_anchors[kind][side]
        identity, fingerprint = saved[identity_key], digest(saved)
        require(all(identity not in anchors[side] for other, anchors in self.uuid_anchors.items()
                    if other != kind),
                "Unsupported cross-kind UUID alias between full bundle and rolling origin")
        require(identity not in identities or identities[identity] == fingerprint,
                f"Ambiguous {kind} UUID anchor: repeated ID has a different complete payload")
        identities[identity] = fingerprint

    def scan(self, first, second, path):
        require(type(first) is type(second), f"JSON types differ at {self.location(path)}")
        if isinstance(first, dict):
            require(first.keys() == second.keys(), f"JSON keys differ at {self.location(path)}")
            if {"spec", "coefficients", "created_at", "id"} <= first.keys():
                from openecon.models import ResultBundle

                try:
                    for side, saved in enumerate((first, second)):
                        ResultBundle.model_validate(saved)
                        valid_uuid(saved["id"])
                        valid_time(saved["created_at"])
                        self.anchor(saved, side, "bundle", "id")
                except Exception as exc:
                    raise ComparisonError(f"Invalid typed ResultBundle at {self.location(path)}: {exc}") from exc
                for key, kind in (("id", "UUID"), ("created_at", "timestamp")):
                    self.bind(first[key], second[key], kind)
                    self.allow(path+(key,), first[key], second[key], f"validated ResultBundle {key}")
                self.bundle_count += 1
                if first["spec"]["estimator"] == "sspace":
                    for saved in (first, second):
                        payload = {**saved, "extra": dict(saved["extra"])}
                        declared = payload["extra"].pop("state_sha256", None)
                        require(declared == digest(payload), "State-space full saved-state integrity digest differs")
                    hashes = first["extra"]["state_sha256"], second["extra"]["state_sha256"]
                    self.bind(*hashes, "state digest")
                    self.allow(path+("extra", "state_sha256"), *hashes,
                               "independently reconstructed full state-space bundle digest")
            if first.get("schema_version") in {"mi-pool-v1", "mi-joint-v1"}:
                from openecon.econometrics.mi.joint import MIPoolResult, MIJointResult

                cls = MIPoolResult if first["schema_version"] == "mi-pool-v1" else MIJointResult
                try:
                    cls.model_validate(first)
                    cls.model_validate(second)
                except Exception as exc:
                    raise ComparisonError(f"Invalid typed MI state at {self.location(path)}: {exc}") from exc
                self.allow(path+("integrity_sha256",), first["integrity_sha256"], second["integrity_sha256"],
                           "typed MI validator reconstructs complete digest; all child science compared")
                self.mi_count += 1
                if cls is MIPoolResult:
                    require(len(first["imputation_ids"]) == len(second["imputation_ids"]),
                            "MI imputation identity counts differ")
                    for index, (one, two) in enumerate(zip(first["imputation_ids"], second["imputation_ids"], strict=True)):
                        self.references.append((path+("imputation_ids", index), one, two, "UUID"))
                    metadata = tuple(decode_json(saved["metadata_json"]) for saved in (first, second))
                    self.metadata[path+("metadata_json",)] = metadata
                    for saved, meta in zip((first, second), metadata, strict=True):
                        records = meta.get("source_results", [])
                        require(isinstance(records, list) and all(isinstance(r, dict) for r in records)
                                and [r.get("id") for r in records] == saved["imputation_ids"],
                                "MI metadata source IDs must match complete imputation IDs")
                    for index, (one, two) in enumerate(zip(metadata[0]["source_results"], metadata[1]["source_results"], strict=True)):
                        self.references.append((path+("metadata_json", "$json", "source_results", index, "id"),
                                                one["id"], two["id"], "UUID"))
                    self.scan(*metadata, path+("metadata_json", "$json"))
            if first.get("schema") == "openecon.summary.v1":
                for saved in (first, second):
                    require(isinstance(saved.get("attrs"), dict) and isinstance(saved.get("tables"), dict),
                            "Summary references require complete tables and attrs")
                    for values in saved["tables"].values():
                        require(isinstance(values, dict) and {"columns", "data", "index"} <= values.keys()
                                and len(values["data"]) == len(values["index"])
                                and all(len(row) == len(values["columns"]) for row in values["data"]),
                                "Summary table geometry is incomplete")
                for key, kind in (("source_result_id", "UUID"), ("state_sha256", "state digest")):
                    if key in first["attrs"] and first["attrs"][key] != second["attrs"][key]:
                        self.references.append((path+("attrs", key), first["attrs"][key], second["attrs"][key], kind))
                origins = first["attrs"].get("origins", []), second["attrs"].get("origins", [])
                require(len(origins[0]) == len(origins[1]), "Rolling origin counts differ")
                for index, (one, two) in enumerate(zip(*origins, strict=True)):
                    if "result_id" in one or "result_id" in two:
                        self.scan_origin(one, two, path+("attrs", "origins", index))
            for key in first:
                self.scan(first[key], second[key], path+(key,))
        elif isinstance(first, list):
            require(len(first) == len(second), f"JSON lengths differ at {self.location(path)}")
            for index, (one, two) in enumerate(zip(first, second, strict=True)):
                self.scan(one, two, path+(index,))

    def scan_origin(self, first, second, path):
        from openecon.models import ResultBundle

        required = {"result_id", "spec", "coefficients", "covariance_matrix", "inference", "metrics",
                    "nobs", "sample_positions", "input_positions", "origin", "status"}
        for side, saved in enumerate((first, second)):
            require(required <= saved.keys() and saved["status"] == "ok",
                    f"Incomplete scientific rolling origin at {self.location(path)}")
            valid_uuid(saved["result_id"])
            positions, inputs = saved["sample_positions"], saved["input_positions"]
            require(isinstance(positions, list) and isinstance(inputs, list)
                    and all(type(v) is int and v >= 0 for v in positions+inputs)
                    and len(set(inputs)) == len(inputs) and len(set(positions)) == len(positions)
                    and set(positions) <= set(inputs) and type(saved["nobs"]) is int
                    and saved["nobs"] == len(positions) > 0 and type(saved["origin"]) is int,
                    "Rolling origin sample geometry is invalid")
            try:
                ResultBundle.model_validate({"id": saved["result_id"], "created_at": "2000-01-01T00:00:00+00:00",
                    "spec": saved["spec"], "coefficients": saved["coefficients"], "covariance_matrix": saved["covariance_matrix"],
                    "metrics": saved["metrics"], "inference": saved["inference"], "nobs": saved["nobs"],
                    "nobs_original": len(inputs), "dropped_rows": len(inputs)-len(positions),
                    "sample_positions": positions, "warnings": [], "predictions": [], "provenance": {}})
            except Exception as exc:
                raise ComparisonError(f"Invalid typed rolling-origin science: {exc}") from exc
            k = len(saved["coefficients"])
            require(len(saved["covariance_matrix"]) == k and all(len(row) == k for row in saved["covariance_matrix"]),
                    "Rolling origin full covariance dimensions differ")
            self.anchor(saved, side, "origin", "result_id")
        self.bind(first["result_id"], second["result_id"], "UUID")
        self.allow(path+("result_id",), first["result_id"], second["result_id"], "complete validated scientific rolling origin ID")
        self.origin_count += 1

    def artifact_reference(self, first, second, path, filename):
        target = str(PurePosixPath(path[0]).parent / filename)
        require(target in self.files, f"Artifact reference absent from complete receipts: {target}")
        hashes = tuple(receipt["files"][target]["sha256"] for receipt in self.receipts)
        require((first, second) == hashes, f"Artifact reference SHA differs from actual file: {target}")
        self.allow(path, first, second, f"actual verified artifact {target}; complete content also compared")
        self.artifact_references.append({"path": self.location(path), "file": target})
        return target

    def scan_artifact_references(self, first, second, path):
        for name in first.get("files", {}):
            self.artifact_reference(first["files"][name], second["files"][name], path+("files", name), name)
        for group in ("sources", "outputs"):
            for name, record in first.get(group, {}).items():
                other = second[group][name]
                require(record["file"] == other["file"], "Artifact reference filenames differ")
                target = self.artifact_reference(record["sha256"], other["sha256"],
                                                 path+(group, name, "sha256"), record["file"])
                if "state_digest" in record:
                    pair = self.json_pairs[target]
                    for saved, declared in zip(pair, (record["state_digest"], other["state_digest"]), strict=True):
                        require(saved.get("spec", {}).get("estimator") == "sspace"
                                and saved["extra"]["state_sha256"] == declared,
                                "Digest reference differs from actual full state")
                    self.allow(path+(group, name, "state_digest"), record["state_digest"], other["state_digest"],
                               "digest of independently verified full referenced state-space artifact")

    def equal(self, first, second, path):
        if path in self.allowed:
            one, two, _ = self.allowed[path]
            require((first, second) == (one, two), f"Provenance value changed at {self.location(path)}")
            return
        if path in self.metadata:
            self.equal(*self.metadata[path], path+("$json",))
            return
        require(type(first) is type(second), f"Scientific JSON types differ at {self.location(path)}")
        if isinstance(first, dict):
            require(first.keys() == second.keys(), f"Scientific JSON keys differ at {self.location(path)}")
            for key in first:
                self.equal(first[key], second[key], path+(key,))
        elif isinstance(first, list):
            require(len(first) == len(second), f"Scientific JSON lengths differ at {self.location(path)}")
            for index, (one, two) in enumerate(zip(first, second, strict=True)):
                self.equal(one, two, path+(index,))
        else:
            require(first == second, f"Unrecognized or scientific difference at {self.location(path)}")

    def report(self):
        rows = {}
        for filename in self.files:
            if filename in self.json_pairs:
                self.equal(*self.json_pairs[filename], (filename,))
            else:
                require((self.folders[0]/filename).read_bytes() == (self.folders[1]/filename).read_bytes(),
                        f"Exact text/binary content differs: {filename}; no substring normalization allowed")
            rows[filename] = {"source_sha256": self.receipts[0]["files"][filename]["sha256"],
                              "installed_sha256": self.receipts[1]["files"][filename]["sha256"],
                              "raw_byte_equal": (self.folders[0]/filename).read_bytes() == (self.folders[1]/filename).read_bytes(),
                              "complete_content_equal_after_explicit_run_provenance_mapping": True}
        return {"complete_files": len(rows), "exact_scientific_content_equal": True,
                "raw_byte_equality_claimed": False, "validated_result_bundles": self.bundle_count,
                "validated_rolling_origins": self.origin_count, "validated_mi_states": self.mi_count,
                "provenance_mapping_only": "exact recorded paths; validated UUID/time and known full-state/artifact references",
                "latex_comparison": "exact bytes; no UUID, timestamp, digest or substring replacement",
                "embedded_mi_metadata_comparison": "complete parsed finite JSON with duplicate keys refused; only source_results ID paths may differ",
                "normalized_paths": [{"path": self.location(path), "source": one, "installed": two, "reason": reason}
                                     for path, (one, two, reason) in self.allowed.items() if one != two],
                "verified_saved_artifact_hash_mapping": self.artifact_references, "files": rows}


def compare(source_receipt, installed_receipt, output, archive):
    comparison = Comparison(source_receipt, installed_receipt)
    report = comparison.report()
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    with tarfile.open(archive, "w:gz") as pack:
        for label, folder, receipt in zip(("source", "installed"), comparison.folders, comparison.receipt_paths, strict=True):
            for filename in comparison.files:
                pack.add(folder/filename, arcname=label+"/"+filename)
            pack.add(receipt, arcname=label+"-receipt.json")
    print(f"{len(report['files'])} complete files match; both raw result sets and receipts are preserved.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-receipt", "installed-receipt", "output", "archive"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    compare(args.source_receipt, args.installed_receipt, args.output, args.archive)
