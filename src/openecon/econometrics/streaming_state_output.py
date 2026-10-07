"""Owned disk-backed diagnostic outputs; no all-period DataFrame allocation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import scan


class StateOutput:
    def __init__(self, metadata):
        self.scratch, self.writer, self.dataset = None, None, None
        self.metadata = metadata
        try:
            self.scratch = tempfile.TemporaryDirectory(
                prefix="openecon-state-output-",
                dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None,
            )
            self.path = Path(self.scratch.name) / "output.parquet"
        except OSError as exc:
            raise AnalysisError(
                "state_space_spill_failed",
                "State diagnostic output needs writable local scratch storage.",
            ) from exc

    def __enter__(self):
        return self

    def write(self, frame):
        import pyarrow as pa
        import pyarrow.parquet as pq

        record = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, record.schema)
            self.path.chmod(0o600)
        self.writer.write_table(record)

    def finish(self):
        if self.writer is None:
            raise AnalysisError("empty_sample", "No usable state diagnostic rows remain.")
        self.writer.close()
        self.writer = None
        output = scan(self.path)
        output._metadata = {"analysis": self.metadata}
        # TemporaryDirectory stays owned by the returned Dataset. Deleting the
        # Dataset releases storage; callers can export it to a durable location.
        output._owned_state_output = self.scratch
        self.dataset = output
        return output

    def __exit__(self, typ, *_):
        if self.writer is not None:
            self.writer.close()
        if self.dataset is None and self.scratch is not None:
            self.scratch.cleanup()
