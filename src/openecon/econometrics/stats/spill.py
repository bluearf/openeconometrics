"""Owned, file-backed statistical ordering with a bounded SQLite page cache."""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile

from openecon.analysis_contracts import AnalysisError


class Spill:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="openecon-statistics-",
            dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None)
        try:
            self.db = sqlite3.connect(Path(self.temp.name) / "statistics.sqlite")
            self.db.execute("PRAGMA temp_store=FILE")
            self.db.execute("PRAGMA cache_size=-2048")
            self.db.execute("PRAGMA mmap_size=0")
            self.db.execute("PRAGMA journal_mode=OFF")
            return self.db
        except BaseException:
            self.temp.cleanup()
            raise

    def __exit__(self, kind, value, traceback):
        self.db.close()
        self.temp.cleanup()
        if kind is not None and issubclass(kind, (sqlite3.Error, OSError)):
            raise AnalysisError("statistics_spill_failed", "Statistical ordering needs writable local scratch storage.") from value


def blocks(cursor, rows=4096):
    while block := cursor.fetchmany(rows):
        yield block
