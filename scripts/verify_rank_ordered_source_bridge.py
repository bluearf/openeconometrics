"""Compare complete current-source results with this chat's installed QA files."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

from verify_rank_ordered_runtime import EXAMPLE, METHODS, ROOT


def source_origins():
    """Verify the executed imports, rather than merely hashing unused files."""
    origins = {}
    for name in ("openecon", "openecon.econometrics.discrete.rank_ordered",
                 "openecon.econometrics.discrete.rank_ordered_postestimation"):
        actual = Path(importlib.import_module(name).__file__).resolve(strict=True)
        path = ROOT / "src" / Path(*name.split("."))
        expected = (path / "__init__.py" if path.is_dir() else path.with_suffix(".py")).resolve(strict=True)
        assert actual == expected, (name, "source import origin", str(actual), str(expected))
        origins[name] = str(actual)
    return origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    installed = Path.home() / "Library/Application Support/org.openecon.qa.rankordered/complete-rank-ordered"
    scientific = [ROOT / "src/openecon/econometrics/discrete" / name for name in
                  ("rank_ordered.py", "rank_ordered_postestimation.py")]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    before = {p.name: digest(p) for p in scientific}
    origins = source_origins()
    receipts = {}
    with tempfile.TemporaryDirectory(prefix="rank-ordered-source-bridge-") as temporary:
        directory = Path(temporary)
        with redirect_stdout(io.StringIO()):
            runpy.run_path(str(EXAMPLE), init_globals={"RANK_ORDERED_RESULT_DIRECTORY": str(directory)})
        for name in METHODS:
            generated = directory / (name + ".json")
            baseline = installed / (name + ".json")
            assert not baseline.is_symlink()
            assert generated.read_bytes() == baseline.read_bytes(), name
            value = json.loads(generated.read_text())
            receipts[name] = dict(sha256=digest(baseline), tables=len(value["tables"]),
                                  complete_tables_attrs_latex_equal=True)
    assert before == {p.name: digest(p) for p in scientific}
    assert origins == source_origins()
    output = dict(status="passed", source_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        scientific_sources=before, results=receipts, all_bytes_equal=True,
        executed_source_module_origins=origins,
        saved_tables=sum(row["tables"] for row in receipts.values()),
        installed_read_only=True, native_gui_observed_by_this_helper=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert output["saved_tables"] == 76
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("status", "saved_tables", "all_bytes_equal")}))


if __name__ == "__main__":
    main()
