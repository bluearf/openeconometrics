"""Compare full legacy native planning results with additive axis-name metadata."""

from __future__ import annotations

import argparse
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tarfile
import tempfile

from verify_prospective_extensions import EXAMPLE, METHODS, ROOT, proof_from, verify_files


def verify(native_archive: Path, output_directory: Path):
    import openecon as oe

    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prospective-axis-bridge-") as temporary:
        directory, stdout, displayed = Path(temporary), io.StringIO(), []
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(str(EXAMPLE), init_globals={
                "PROSPECTIVE_EXTENSION_RESULT_DIRECTORY": str(directory),
                "display": displayed.append,
            })
        proof = proof_from(stdout.getvalue())
        assert not proof["frozen"] and len(displayed) == 16
        current_hashes = verify_files(directory, proof)
        records = {}
        with tarfile.open(native_archive) as archive:
            members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
            assert set(members) == {name + ".json" for name in METHODS}
            for name in METHODS:
                old_bytes = archive.extractfile(members[name + ".json"]).read()
                old = json.loads(old_bytes)
                current = json.loads((directory / (name + ".json")).read_bytes())
                expected = deepcopy(old)
                for frame in expected["summary"]["tables"].values():
                    assert "index_names" not in frame and "column_names" not in frame
                    frame["index_names"] = [None]
                    frame["column_names"] = [None]
                # Only these exact two additive null-name fields are admitted.
                assert current == expected, name
                restored = oe.restore_summary(json.dumps(old["summary"], allow_nan=False))
                assert json.loads(oe.summary_state(restored)) == current["summary"], name
                assert restored.to_latex() == old["latex"] == current["latex"], name
                records[name] = {
                    "native_sha256": hashlib.sha256(old_bytes).hexdigest(),
                    "current_sha256": current_hashes[name],
                    "tables": len(current["summary"]["tables"]),
                    "only_added_fields": {"index_names": [None], "column_names": [None]},
                    "legacy_restore_matches_complete_current_summary": True,
                    "all_other_fields_and_latex_identical": True,
                }
        with tarfile.open(output_directory / "axis-current-complete-results.tar.gz", "w:gz") as archive:
            for name in METHODS:
                archive.add(directory / (name + ".json"), arcname=name + ".json")
    record = {
        "status": "passed", "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "native_archive_sha256": hashlib.sha256(native_archive.read_bytes()).hexdigest(),
        "methods": records, "saved_tables": sum(value["tables"] for value in records.values()),
        "current_complete_result_hashes": current_hashes,
        "historical_native_hashes_equal_current": False,
        "scope": "Current source adds only two null axis-name lists per table; historical native/frozen receipts remain pinned to their original revision.",
    }
    (output_directory / "axis-bridge.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "passed", "methods": len(records), "tables": record["saved_tables"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    verify(args.native_archive, args.output_directory)
