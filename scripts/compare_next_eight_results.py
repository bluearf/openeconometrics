"""Compare complete source/SDK files with explicit verified run provenance mapping."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def compare(source_receipt, installed_receipt, output, archive):
    a, b = (json.loads(Path(f).read_text()) for f in (source_receipt, installed_receipt))
    left, right = Path(a["output_directory"]), Path(b["output_directory"])
    assert set(a["files"]) == set(b["files"])
    mapping = {}

    def collect(x, y):
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            if {"spec", "coefficients", "created_at", "id"} <= x.keys():
                mapping[y["id"]], mapping[y["created_at"]] = x["id"], x["created_at"]
            for key in x:
                collect(x[key], y[key])
        elif isinstance(x, list):
            assert len(x) == len(y)
            for first, second in zip(x, y):
                collect(first, second)

    for file in a["files"]:
        for folder, receipt in ((left, a), (right, b)):
            assert hashlib.sha256((folder/file).read_bytes()).hexdigest() == receipt["files"][file]["sha256"]
        if file.endswith(".json"):
            collect(json.loads((left/file).read_text()), json.loads((right/file).read_text()))
    model, state = "proxy/next-eight-reduced-var.json", "proxy/next-eight-proxy-state.json"
    lh, rh = (hashlib.sha256((folder/model).read_bytes()).hexdigest() for folder in (left, right))
    for folder, expected in ((left, lh), (right, rh)):
        assert json.loads((folder/state).read_text())["attrs"]["proxy_state"]["source_result_json_hash"] == expected
    mapping[rh] = lh

    def normalized(v):
        if isinstance(v, dict):
            return {k: normalized(x) for k, x in v.items()}
        if isinstance(v, list):
            return [normalized(x) for x in v]
        if isinstance(v, str):
            if v in mapping:
                return mapping[v]
            for first, second in mapping.items():
                if v == json.dumps(first):
                    return json.dumps(second)
        return v

    rows = {}
    for file in a["files"]:
        source_file, installed_file = (folder/file for folder in (left, right))
        if file.endswith(".json"):
            assert json.loads(source_file.read_text()) == normalized(json.loads(installed_file.read_text())), file
        else:
            # The renderer inserts visual break markers inside long UUIDs.
            first, second = (f.read_text().replace("\\allowbreak{}", "") for f in (source_file, installed_file))
            for previous, new in mapping.items():
                second = second.replace(previous, new)
            assert first == second, file
        rows[file] = {"source_sha256": a["files"][file]["sha256"],
                      "installed_sha256": b["files"][file]["sha256"],
                      "raw_byte_equal": source_file.read_bytes() == installed_file.read_bytes(),
                      "complete_content_equal_after_explicit_run_provenance_mapping": True}
    report = {"complete_files": len(rows), "exact_scientific_content_equal": True,
              "raw_byte_equality_claimed": False,
              "provenance_mapping_only": ["model run UUID and its references", "model creation timestamp",
                  "hash of source VAR JSON verified against each actual file; UUID/timestamp differ"],
              "latex_comparison": "exact rendered text after removing visual allowbreak markers and mapping only verified run provenance",
              "files": rows}
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    with tarfile.open(archive, "w:gz") as pack:
        for label, folder in (("source", left), ("installed", right)):
            for file in sorted(a["files"]):
                pack.add(folder/file, arcname=label+"/"+file)
    print(f"{len(rows)} complete files match; both raw result sets are preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-receipt", "installed-receipt", "output", "archive"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    compare(args.source_receipt, args.installed_receipt, args.output, args.archive)
