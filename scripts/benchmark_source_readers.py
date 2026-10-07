"""Physical XLSX/DTA conversion, peak parser RSS, bounded replay and native OLS oracle."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

import pandas as pd
import pyreadstat
from openpyxl import Workbook

import openecon as oe

ROOT = Path(__file__).resolve().parents[1]


def values(i):
    x = i % 101
    return x, 2 + 0.75 * x + math.cos(i) / 100


def run(suffix, rows):
    with tempfile.TemporaryDirectory(prefix="openecon-reader-scale-") as tmp:
        path = Path(tmp) / ("physical." + suffix)
        if suffix == "xlsx":
            workbook = Workbook(write_only=True)
            sheet = workbook.create_sheet()
            sheet.append(["x", "y", "label"])
            for i in range(rows):
                sheet.append([*values(i), "Türkiye " + str(i % 13)])
            workbook.save(path)
            workbook.close()
        else:
            frame = pd.DataFrame(
                {
                    "x": [values(i)[0] for i in range(rows)],
                    "y": [values(i)[1] for i in range(rows)],
                    "label": ["Türkiye " + "original evidence " * 8] * rows,
                }
            )
            pyreadstat.write_dta(
                frame,
                str(path),
                column_labels={"y": "Outcome"},
                variable_value_labels={"x": {0: "Baseline", 1: "One"}},
            )
            del frame
        with path.open("rb") as file:
            fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
        physical_bytes = path.stat().st_size
        source = oe.read(path)
        assert isinstance(source, oe.Dataset) and source.row_count == rows
        directory = source._path.parent
        receipt = source.provenance["conversion"]
        count = 0
        max_batch = 0
        for batch in source.iter_batches(columns=["x", "y"], batch_rows=4096):
            count += len(batch)
            max_batch = max(max_batch, len(batch))
        assert count == rows and max_batch <= 4096
        fit = oe.ols(data=source, y="y", x=["x"], device="cpu", covariance="HC3")
        sx = math.fsum(values(i)[0] for i in range(rows))
        sy = math.fsum(values(i)[1] for i in range(rows))
        sxx = math.fsum(values(i)[0] ** 2 for i in range(rows))
        sxy = math.fsum(values(i)[0] * values(i)[1] for i in range(rows))
        slope = (sxy - sx * sy / rows) / (sxx - sx * sx / rows)
        intercept = sy / rows - slope * sx / rows
        actual = {entry.term: entry.estimate for entry in fit.coefficients}
        assert abs(actual["x"] - slope) < 1e-10
        constant = next(value for name, value in actual.items() if name != "x")
        assert abs(constant - intercept) < 1e-10
        with path.open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == fingerprint
        parser_receipt = source.reader_receipt
        source.close()
        assert not directory.exists()
        return dict(
            format=suffix,
            physical_source_bytes=physical_bytes,
            physical_rows=rows,
            full_replay_rows=count,
            max_observed_batch_rows=max_batch,
            conversion=receipt,
            downstream_parser=parser_receipt,
            native_ols_rows=fit.nobs,
            independent_ols={"intercept": intercept, "slope": slope},
            actual_coefficients=actual,
            source_unchanged=True,
            owned_conversion_scratch_removed=True,
        )


def main():
    records = [run("xlsx", 120001), run("dta", 300001)]
    record = dict(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        status="passed",
        runs=records,
        limits={
            "parser_extra_bytes": 512 * 1024 * 1024,
            "parser_total_rss_bytes": 1536 * 1024 * 1024,
            "parser_buffer_bytes": 64 * 1024 * 1024,
            "retained_batch_bytes": 256 * 1024 * 1024,
            "conversion_disk_bytes": 8 * 1024 * 1024 * 1024,
        },
        limitations="Owned Mac CPU fixture. Parser peaks include runtime imports; child getrusage peak plus parent RSS sampling. macOS RSS supervision supplements physical preflight; RLIMIT_AS is used only on Linux. No Windows or universal file/schema acceptance claim.",
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/openecon/source_readers.py",
                "src/openecon/dataset.py",
                "src/openecon/data.py",
                "src/openecon/workspace.py",
                "scripts/benchmark_source_readers.py",
            )
        },
    )
    destination = ROOT / "docs/evidence/market-98-source-readers-2026-10-07.json"
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": [r["physical_rows"] for r in records],
                "peak_rss": [r["conversion"]["parser_peak_rss_bytes"] for r in records],
            }
        )
    )


if __name__ == "__main__":
    main()
