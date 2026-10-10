"""Regenerate instructor-owned synthetic input rows, without writing workbooks.

The committed Excel files are the student inputs. This utility exposes their
original generating mechanisms to instructors and supports independent checks.
Run with --output to write lossless JSON inputs for a workbook authoring tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def original_frame(slug):
    namespace = runpy.run_path(
        str(ROOT / "docs/teaching/instructors/generators" / f"{slug}.py")
    )
    frame = pd.DataFrame(namespace["make_data"]())
    if slug == "13-instrumental-variables":
        weak = namespace["make_data"](relevance=0.12)
        invalid = namespace["make_data"](direct_effect=0.15)
        for column in ("schooling", "log_earnings"):
            frame[column + "_weak"] = weak[column]
            frame[column + "_invalid"] = invalid[column]
    frame.attrs["generation_seed"] = namespace["SEED"]
    return frame


def verify_workbook(folder):
    """Check raw rows, column order, missing cells and Excel numeric precision."""
    metadata = json.loads((folder / "dataset.json").read_text())
    source = folder / metadata["filename"]
    with pd.ExcelFile(source, engine="openpyxl") as workbook:
        if workbook.sheet_names != ["Data", "Dictionary"]:
            raise AssertionError(f"Unexpected worksheets: {source}")
        actual = pd.read_excel(workbook, sheet_name="Data")
        dictionary = pd.read_excel(workbook, sheet_name="Dictionary")
    expected = original_frame(folder.name)
    reference = json.loads((folder / "reference.json").read_text())
    if expected.attrs["generation_seed"] != reference["metadata"]["seed"]:
        raise AssertionError(f"Original generation seed differs: {source}")
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-13
    )
    if list(dictionary.iloc[:len(expected.columns), 0]) != list(expected.columns):
        raise AssertionError(f"Dictionary does not describe every input column: {source}")
    return {
        "filename": metadata["filename"],
        "generation_seed": expected.attrs["generation_seed"],
        "rows": len(actual),
        "columns": list(actual.columns),
        "missing_cells": int(actual.isna().sum().sum()),
        "original_rows_and_order_match": True,
        "original_missing_mask_matches": actual.isna().equals(expected.isna()),
        "numeric_relative_and_absolute_tolerance": 1e-13,
        "dictionary_columns_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    catalog = json.loads((ROOT / "docs/teaching/catalog.json").read_text())
    for lab in catalog["labs"]:
        folder = ROOT / "docs/teaching/labs" / lab["slug"]
        if lab["number"] == 3:
            continue
        if arguments.output:
            frame = original_frame(lab["slug"])
            rows = [
                [None if pd.isna(value) else value for value in row]
                for row in frame.itertuples(index=False, name=None)
            ]
            arguments.output.mkdir(parents=True, exist_ok=True)
            (arguments.output / f"{lab['slug']}.json").write_text(json.dumps(
                {"slug": lab["slug"], "columns": list(frame.columns), "rows": rows},
                allow_nan=False,
            ) + "\n")
        else:
            receipt = verify_workbook(folder)
            print(f"{lab['slug']}: {receipt['rows']} rows, raw input verified")


if __name__ == "__main__":
    main()
