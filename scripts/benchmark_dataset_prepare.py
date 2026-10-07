"""Physical lazy preparation, disk cleanup, replay and independent HC3 oracle."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import openecon as oe

ROOT = Path(__file__).resolve().parents[1]


def values(i):
    x = float(i % 101)
    return x, 2 + 0.75 * x + math.cos(i) / 100


def typed(block):
    return block.astype({"row": "Int64", "id": "Int64", "x": "Float64", "y": "Float64"})


def run():
    started = time.monotonic()
    rows = 100001
    with tempfile.TemporaryDirectory(prefix="openecon-preparation-scale-") as tmp:
        root = Path(tmp)
        scratch = root / "scratch"
        scratch.mkdir()
        previous = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        os.environ["OPENECON_SCRATCH_DIRECTORY"] = str(scratch)
        try:
            path = root / "physical.parquet"
            writer = None
            for first in range(0, rows, 8192):
                ix = range(first, min(rows, first + 8192))
                frame = pd.DataFrame(
                    {
                        "row": list(ix),
                        "id": [i % 2000 for i in ix],
                        "x": [values(i)[0] for i in ix],
                        "y": [values(i)[1] for i in ix],
                    },
                    index=pd.Index([i + 500000 for i in ix], name="original"),
                )
                table = pa.Table.from_pandas(frame, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=8192)
            writer.close()
            del frame, table
            source = oe.scan(path)
            fingerprint = source.file_hash()
            assert source.row_count == rows
            prepared = source.map(
                typed, schema={"row": "Int64", "id": "Int64", "x": "Float64", "y": "Float64"}
            ).filter(lambda b: b["x"] >= 25)
            lookup_frame = pd.DataFrame(
                {
                    "id": pd.array(range(2000), dtype="Int64"),
                    "group": pd.Categorical(
                        ["A" if i % 2 else "B" for i in range(2000)],
                        categories=["A", "B", "unused"],
                        ordered=True,
                    ),
                },
                index=pd.Index(range(700000, 702000), name="lookup"),
            )
            joined = prepared.join(oe.Dataset.from_frame(lookup_frame), on="id", validate="m:1")
            maximum = count = 0
            for block in joined.iter_batches(batch_rows=4096):
                maximum = max(maximum, len(block))
                for identity, row in zip(block.index, block.itertuples(index=False), strict=True):
                    assert identity == (int(row.row) + 500000, int(row.id) + 700000)
                    assert row.x == values(int(row.row))[0] and row.y == values(int(row.row))[1]
                assert list(block["group"].cat.categories) == ["A", "B", "unused"]
                count += len(block)
            expected = sum(values(i)[0] >= 25 for i in range(rows))
            assert count == expected and maximum <= 4096
            digest = joined.preparation_receipt["output_digest"]
            fit = oe.ols(data=joined, y="y", x=["x"], covariance="HC3", device="cpu")
            assert fit.nobs == expected and joined.preparation_receipt["output_digest"] == digest
            # Constant-size sufficient statistics and HC3 meat from the original
            # scalar generator; no dense full fixture and no OpenEcon oracle.
            sx = math.fsum(values(i)[0] for i in range(rows) if values(i)[0] >= 25)
            sxx = math.fsum(values(i)[0] ** 2 for i in range(rows) if values(i)[0] >= 25)
            sy = math.fsum(values(i)[1] for i in range(rows) if values(i)[0] >= 25)
            sxy = math.fsum(values(i)[0] * values(i)[1] for i in range(rows) if values(i)[0] >= 25)
            inverse = np.linalg.inv(np.array([[expected, sx], [sx, sxx]], dtype=float))
            beta = inverse @ np.array([sy, sxy])
            meat = np.zeros((2, 2))
            for i in range(rows):
                x, y = values(i)
                if x >= 25:
                    v = np.array([1.0, x])
                    adjustment = ((y - v @ beta) / (1 - v @ inverse @ v)) ** 2
                    meat += adjustment * np.outer(v, v)
            errors = np.sqrt(np.diag(inverse @ meat @ inverse))
            for position, coefficient in enumerate(fit.coefficients):
                assert abs(coefficient.estimate - beta[position]) < 1e-10
                assert abs(coefficient.std_error - errors[position]) < 1e-10
            long = prepared.project(["row", "x", "y"]).reshape_long(
                id_vars=["row"], value_vars=["x", "y"]
            )
            wide = long.reshape_wide(
                keys="row", variable="variable", value="value", levels=["x", "y"]
            )
            wide_count = 0
            for block in wide.iter_batches(batch_rows=4096):
                for index, row in zip(block.index, block.itertuples(index=False), strict=True):
                    i = int(row.row)
                    assert (row.x, row.y) == values(i)
                    assert index == ((i,), (("x", (i + 500000, "x")), ("y", (i + 500000, "y"))))
                wide_count += len(block)
            assert wide_count == expected and source.file_hash() == fingerprint
            assert not list(scratch.iterdir())
            record = dict(
                status="passed",
                physical_rows=rows,
                physical_bytes=path.stat().st_size,
                source_sha256=fingerprint,
                full_join_rows=count,
                full_wide_rows=wide_count,
                max_observed_block_rows=maximum,
                native_ols_rows=fit.nobs,
                independent_coefficients=beta.tolist(),
                independent_HC3_errors=errors.tolist(),
                join_receipt=joined.preparation_receipt,
                wide_receipt=wide.preparation_receipt,
                parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                elapsed_seconds=time.monotonic() - started,
                source_unchanged=True,
                owned_scratch_removed=True,
            )
        finally:
            if previous is None:
                os.environ.pop("OPENECON_SCRATCH_DIRECTORY", None)
            else:
                os.environ["OPENECON_SCRATCH_DIRECTORY"] = previous
    return record


if __name__ == "__main__":
    record = dict(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        run=run(),
        scope="macOS CPU; true process peak includes imports/Torch, per-stage buffers and parser RSS are distinct budgets; no Windows claim",
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "src/openecon/dataset_prepare.py",
                "src/openecon/dataset.py",
                "scripts/benchmark_dataset_prepare.py",
            ]
        },
    )
    destination = ROOT / "docs/evidence/market-108-dataset-preparation-2026-10-07.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["run"], indent=2))
