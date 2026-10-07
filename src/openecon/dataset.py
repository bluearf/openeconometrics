"""Replayable, projected local datasets without a total row or file-size limit.

This source only materializes one bounded batch at a time. Estimators that make
multiple passes must additionally compare their content digests between passes:
file identities cannot prove that an in-memory frame or a user factory is immutable.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pandas as pd

from openecon.data import DataError, metadata_hash
from openecon.frame import DataFrame, as_frame

DEFAULT_BATCH_ROWS = 65_536
MAX_BATCH_ROWS = 1_000_000
MAX_BATCH_BYTES = 256 * 1024 * 1024
MAX_SOURCE_FILES = 100_000
MAX_SOURCE_COLUMNS = 10_000


def _columns(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise DataError("Columns must be a sequence of names.")
    values = tuple(names)
    if not values or len(values) > MAX_SOURCE_COLUMNS:
        raise DataError(f"A source needs between 1 and {MAX_SOURCE_COLUMNS:,} columns.")
    if any(not isinstance(name, str) or not name.strip() or len(name) > 200
           for name in values) or len(set(values)) != len(values):
        raise DataError("Column names must be nonempty, unique strings of at most 200 characters.")
    return values


def _positive_int(value: int, name: str, *, maximum: int | None = None,
                  zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise DataError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer.")
    if maximum is not None and value > maximum:
        raise DataError(f"{name} cannot exceed {maximum:,}; request smaller batches.",
                        "BATCH_LIMIT")
    return value


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise DataError("Dataset sources must be regular local files.", "INVALID_SOURCE")
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    except OSError as exc:
        raise DataError("A dataset source disappeared or became unreadable.",
                        "SOURCE_CHANGED") from exc


def _discover_parquet(root: Path) -> tuple[Path, ...]:
    files = []
    try:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".parquet":
                files.append(path)
                if len(files) > MAX_SOURCE_FILES:
                    raise DataError(f"A dataset manifest supports up to {MAX_SOURCE_FILES:,} files.",
                                    "MANIFEST_LIMIT")
    except OSError as exc:
        raise DataError("The Parquet directory could not be inspected.", "INVALID_SOURCE") from exc
    if not files:
        raise DataError("The directory contains no Parquet files.", "FILE_NOT_FOUND")
    return tuple(sorted(files))


def _frame_schema(frame: pd.DataFrame) -> tuple:
    schema = []
    for name, dtype in frame.dtypes.items():
        category = None
        if isinstance(dtype, pd.CategoricalDtype):
            # A frame can contain huge category dictionaries in an unused
            # column. Do not copy all levels into a Python tuple merely to
            # record schema identity; hash the immutable Index in small blocks.
            categories = dtype.categories
            digest = hashlib.sha256(str(categories.dtype).encode("utf-8"))
            for start in range(0, len(categories), DEFAULT_BATCH_ROWS):
                hashes = pd.util.hash_pandas_object(
                    categories[start:start + DEFAULT_BATCH_ROWS], index=False, categorize=False,
                )
                digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
            category = (len(categories), digest.hexdigest(), dtype.ordered)
        schema.append((name, str(dtype), category))
    return tuple(schema)


class Dataset:
    """A local source that can be read again in the same row order.

    Construct file-backed sources with :func:`scan`. Frames and callable batch
    factories are supported for library integrations. ``row_count`` is an exact
    Python integer when known, and ``None`` for CSV sources; it is never a float.
    """

    def __init__(self) -> None:
        self._kind = "uninitialized"
        self._columns: tuple[str, ...] = ()
        self._row_count: int | None = None
        self._metadata: dict = {}
        self._provenance: dict = {}
        self._path: Path | None = None
        self._files: tuple[Path, ...] = ()
        self._manifest: tuple = ()
        self._arrow_dataset: Any = None
        self._frame: pd.DataFrame | None = None
        self._frame_schema: tuple = ()
        self._frame_metadata_hash: str | None = None
        self._factory: Callable[[], Iterator[pd.DataFrame]] | None = None
        self._previous_factory_iterator: Any = None

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def row_count(self) -> int | None:
        return self._row_count

    @property
    def nrows(self) -> int | None:
        return self._row_count

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    @property
    def provenance(self) -> dict:
        return deepcopy(self._provenance)

    def __repr__(self) -> str:
        rows = "unknown" if self.row_count is None else f"{self.row_count:,}"
        return f"Dataset({rows} rows, {len(self._columns)} columns; replayable batches)"

    def close(self) -> None:
        """Release owned conversion scratch; original source files remain intact."""
        cleanup = getattr(self, "_cleanup", None)
        if cleanup is not None:
            cleanup()

    def project(self, columns, **limits) -> Dataset:
        """Lazily project columns with checked input/output replay and original indices."""
        from openecon.dataset_prepare import project
        return project(self, columns, **limits)

    def filter(self, predicate, *, missing="drop", **limits) -> Dataset:
        """Lazily filter an aligned nullable boolean predicate; missing is drop/keep/error."""
        from openecon.dataset_prepare import filter_rows
        return filter_rows(self, predicate, missing=missing, **limits)

    def map(self, function, *, schema, metadata=None, **limits) -> Dataset:
        """Map bounded blocks at unchanged original indices with an explicit dtype schema.

        Caller functions own their allocations and side effects. Output must
        already have the declared dtype/category schema; no implicit casts.
        """
        from openecon.dataset_prepare import map_rows
        return map_rows(self, function, schema=schema, metadata=metadata, **limits)

    def join(self, right, *, on, how="inner", validate="m:1", nulls="never",
             suffixes=("_x", "_y"), **limits) -> Dataset:
        """Replayable keyed disk join; output indices retain (left index, right index).

        Explicit 1:1/m:1/1:m/m:m validation and preallocated expansion admission.
        Key dtypes/categories must agree. Null keys never match unless equal is
        explicit; booleans and numeric keys are distinct. Source files stay local.
        """
        from openecon.dataset_prepare import join
        return join(self, right, on=on, how=how, validate=validate, nulls=nulls,
                    suffixes=suffixes, **limits)

    def reshape_long(self, *, id_vars, value_vars, variable="variable", value="value", **limits) -> Dataset:
        """Bounded row-major long data, preserving (original index, original column).

        Value columns require equal dtype/category schema. Unlike pandas melt,
        source rows precede variable order; expansion is admitted before allocation.
        """
        from openecon.dataset_prepare import reshape_long
        return reshape_long(self, id_vars=id_vars, value_vars=value_vars, variable=variable, value=value, **limits)

    def reshape_wide(self, *, keys, variable, value, levels, duplicates="error", unknown="error", **limits) -> Dataset:
        """Disk wide data with explicit bounded levels and duplicate/unknown policies.

        Original source indices per level remain in each group-key index tuple.
        First-appearance group order is stable; missing cells remain nullable.
        """
        from openecon.dataset_prepare import reshape_wide
        return reshape_wide(self, keys=keys, variable=variable, value=value, levels=levels,
                            duplicates=duplicates, unknown=unknown, **limits)

    @property
    def preparation_receipt(self) -> dict:
        """Latest complete-pass digest, expansion and owned scratch cleanup evidence."""
        return deepcopy(getattr(self, "_preparation_receipt", {}))

    @property
    def reader_receipt(self) -> dict:
        """Latest parser RSS/buffer limits and owned-process cleanup evidence."""
        return deepcopy(getattr(self, "_reader_receipt", {}))

    def iter_missing_codes(self, column: str, *, batch_rows=DEFAULT_BATCH_ROWS):
        """Replay original DTA a-z tags as bounded code-to-original-row mappings.

        Estimation batches contain nullable numeric/date values. Tags remain in
        separate physical Parquet fields and are never model predictors.
        """
        if (column not in self._metadata.get("extended_missing_storage", {})
                and column not in self._metadata.get("extended_missing_columns", [])
                and column not in self._metadata.get("extended_missing_codes", {})):
            raise DataError("This column has no retained DTA missing-code storage.")
        iterator = self.iter_batches(columns=[column], batch_rows=batch_rows)
        try:
            for batch in iterator:
                yield deepcopy(batch.attrs.get("extended_missing_codes", {}).get(column, {}))
        finally:
            iterator.close()

    def __enter__(self):
        self.assert_unchanged()
        return self

    def __exit__(self, *exception):
        self.close()

    def file_hash(self) -> str:
        """Hash a single file in bounded blocks, including its physical metadata."""
        if len(self._files) != 1 or self._path is None or self._path.is_dir():
            raise DataError("A file hash requires a single file-backed dataset.", "INVALID_SOURCE")
        self.assert_unchanged()
        digest = hashlib.sha256()
        with self._files[0].open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        self.assert_unchanged()
        return digest.hexdigest()

    def content_hash(self, columns=None, *, batch_rows=DEFAULT_BATCH_ROWS) -> str:
        """Hash projected row values/metadata without retaining the complete table.

        Row hashes depend on pandas and reader dtypes. File snapshots use
        ``file_hash`` instead, binding all source bytes and physical metadata.
        """
        selected = self._projection(columns)
        digest = hashlib.sha256(json.dumps({"columns": selected,
            "metadata_hash": metadata_hash(self._metadata)}, sort_keys=True).encode())
        for batch in self.iter_batches(selected, batch_rows=batch_rows):
            hashes = pd.util.hash_pandas_object(batch, index=False, categorize=True)
            digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
        return digest.hexdigest()

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Dataset:
        """Wrap an existing frame without copying or imposing an overall row cap."""
        if not isinstance(frame, pd.DataFrame):
            raise DataError("from_frame requires a pandas DataFrame.")
        source = cls()
        source._kind = "frame"
        source._columns = _columns(frame.columns)
        source._frame = frame
        source._row_count = len(frame)
        source._frame_schema = _frame_schema(frame)
        source._metadata = deepcopy(frame.attrs.get("metadata", {}))
        source._frame_metadata_hash = metadata_hash(source._metadata)
        source._provenance = {"kind": "frame", "row_count": len(frame),
                              "integrity": "schema and engine pass-content digests"}
        return source

    @classmethod
    def from_batches(cls, factory: Callable[[], Iterator[pd.DataFrame]], columns: Sequence[str],
                     *, row_count: int | None = None, metadata: dict | None = None) -> Dataset:
        """Wrap a factory that returns a fresh batch iterator on every pass.

        An iterator alone is deliberately rejected: robust regression needs
        replayable data for later passes. A declared count is checked whenever
        a pass is consumed in full, and can exceed signed 32-bit integer limits.
        The factory must itself construct bounded batches. Splitting a supplied
        frame cannot undo memory already allocated by an external producer.
        """
        if not callable(factory):
            raise DataError("Provide a callable returning a fresh iterator, not a one-shot iterator.",
                            "NON_REPLAYABLE_SOURCE")
        if row_count is not None:
            _positive_int(row_count, "row_count", zero=True)
        source = cls()
        source._kind = "factory"
        source._columns = _columns(columns)
        source._factory = factory
        source._row_count = row_count
        source._metadata = deepcopy(metadata or {})
        metadata_hash(source._metadata)
        source._provenance = {"kind": "batch_factory", "row_count": row_count,
                              "integrity": "engine pass-content digests"}
        return source

    def assert_unchanged(self) -> None:
        """Fail if a source's file identity, schema, or frame metadata changed."""
        for parent in getattr(self, "_parents", ()):
            parent.assert_unchanged()
        if self._kind in {"parquet", "csv"}:
            assert self._path is not None
            if self._path.is_dir():
                try:
                    current = _discover_parquet(self._path)
                except DataError as exc:
                    raise DataError("The dataset file manifest changed.", "SOURCE_CHANGED") from exc
                if current != self._files:
                    raise DataError("The dataset file manifest changed.", "SOURCE_CHANGED")
            if tuple(_file_identity(path) for path in self._files) != self._manifest:
                raise DataError("A dataset file changed; create a new source before fitting.",
                                "SOURCE_CHANGED")
        elif self._kind == "frame":
            assert self._frame is not None
            if (len(self._frame) != self._row_count
                    or _frame_schema(self._frame) != self._frame_schema
                    or metadata_hash(self._frame.attrs.get("metadata", {}))
                    != self._frame_metadata_hash):
                raise DataError("The source frame schema, count or metadata changed.",
                                "SOURCE_CHANGED")
        elif self._kind != "factory":
            raise DataError("Create a source with scan, from_frame or from_batches.", "INVALID_SOURCE")

    def _projection(self, columns: Sequence[str] | None) -> tuple[str, ...]:
        selected = self._columns if columns is None else _columns(columns)
        missing = set(selected).difference(self._columns)
        if missing:
            raise DataError(f"Unknown source columns: {', '.join(sorted(missing))}.",
                            "MISSING_COLUMN")
        return selected

    def _raw_batches(self, selected: tuple[str, ...], batch_rows: int) -> Iterator[pd.DataFrame]:
        if self._kind in {"csv", "parquet"}:
            from openecon.source_readers import guarded_batches
            yield from guarded_batches(self, selected, batch_rows)
        else:
            yield from self._local_batches(selected, batch_rows)

    def _local_batches(self, selected: tuple[str, ...], batch_rows: int) -> Iterator[pd.DataFrame]:
        if self._kind == "parquet":
            import pyarrow as pa
            def pandas_dtype(dtype):
                # Preserve declared category levels/order; all other extension
                # columns retain Arrow's nullable integer/decimal precision.
                return None if pa.types.is_dictionary(dtype) else pd.ArrowDtype(dtype)
            # Arrow's pandas index columns are physical storage, not model
            # predictors. Read them even for a narrow projection so to_pandas
            # restores the original index instead of inventing one per batch.
            physical = list(dict.fromkeys([*selected, *getattr(self, "_index_columns", ())]))
            missing_storage = self._metadata.get("extended_missing_storage", {})
            physical.extend(name for column, name in missing_storage.items() if column in selected)
            scanner = self._arrow_dataset.scanner(
                columns=physical, batch_size=batch_rows, use_threads=False,
                batch_readahead=1, fragment_readahead=1,
            )
            offset = 0
            for batch in scanner.to_batches():
                # Arrow extension dtypes preserve nullable integers and their precision.
                frame = batch.to_pandas(types_mapper=pandas_dtype)
                if not getattr(self, "_index_columns", ()):
                    record = getattr(self, "_range_index", None)
                    start, step, name = (record["start"], record["step"], record.get("name")) if record else (0, 1, None)
                    frame.index = pd.RangeIndex(start + offset * step,
                                                start + (offset + len(frame)) * step,
                                                step, name=name)
                offset += len(frame)
                if missing_storage:
                    frame.attrs["extended_missing_codes"] = {
                        column: {str(code): [int(position) for position in frame.index[frame[name].eq(code).fillna(False)]]
                                 for code in frame[name].dropna().unique()}
                        for column, name in missing_storage.items() if column in selected}
                    frame = frame.drop(columns=[name for column, name in missing_storage.items() if column in selected])
                yield frame
        elif self._kind == "csv":
            with pd.read_csv(self._path, usecols=list(selected), chunksize=batch_rows,
                             encoding="utf-8-sig", dtype_backend="numpy_nullable") as reader:
                for batch in reader:
                    yield batch.loc[:, list(selected)]
        elif self._kind == "frame":
            assert self._frame is not None
            for start in range(0, len(self._frame), batch_rows):
                chunk = self._frame.iloc[start:start + batch_rows]
                yield chunk if selected == self._columns else chunk.loc[:, list(selected)]
        else:
            assert self._factory is not None
            iterator = iter(self._factory())
            if iterator is self._previous_factory_iterator:
                raise DataError("The batch factory reused a one-shot iterator.",
                                "NON_REPLAYABLE_SOURCE")
            self._previous_factory_iterator = iterator
            try:
                for frame in iterator:
                    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != self._columns:
                        raise DataError("Every factory batch must be a DataFrame with the declared columns.",
                                        "INVALID_BATCH")
                    for start in range(0, len(frame), batch_rows):
                        chunk = frame.iloc[start:start + batch_rows]
                        yield chunk if selected == self._columns else chunk.loc[:, list(selected)]
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    def iter_batches(self, columns: Sequence[str] | None = None,
                     batch_rows: int = DEFAULT_BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yield projected batches; full-pass consumption verifies known row counts.

        The batch byte ceiling is separate from total source size. Extremely
        wide or oversized individual rows require a narrower projection; their
        parser allocation itself cannot be bounded by a row-count argument.
        """
        _positive_int(batch_rows, "batch_rows", maximum=MAX_BATCH_ROWS)
        selected = self._projection(columns)
        self.assert_unchanged()
        rows = 0
        iterator = self._raw_batches(selected, batch_rows)
        try:
            for batch in iterator:
                if tuple(batch.columns) != selected or len(batch) > batch_rows:
                    raise DataError("The reader returned an invalid batch.", "INVALID_BATCH")
                if int(batch.memory_usage(index=True, deep=True).sum()) > MAX_BATCH_BYTES:
                    raise DataError("A batch exceeds 256 MiB; use fewer rows or columns.", "BATCH_LIMIT")
                if batch.empty:
                    continue
                if "extended_missing_codes" in batch.attrs:
                    def index_key(value):
                        return tuple(index_key(item) for item in value) if isinstance(value, (list, tuple)) else value
                    present = set(batch.index)
                    batch.attrs["extended_missing_codes"] = {
                        column: {code: [index_key(index) for index in positions if index_key(index) in present]
                                 for code, positions in mapping.items()}
                        for column, mapping in batch.attrs["extended_missing_codes"].items() if column in selected}
                rows += len(batch)
                yield batch
            if self._row_count is not None and rows != self._row_count:
                raise DataError("The replayed row count differs from the declared source count.",
                                "SOURCE_CHANGED")
        except DataError:
            raise
        except Exception as exc:
            from openecon.analysis_contracts import AnalysisError
            if isinstance(exc, AnalysisError):
                raise
            raise DataError(f"The dataset batch could not be read: {type(exc).__name__}.",
                            "INVALID_BATCH") from exc
        finally:
            iterator.close()
            self.assert_unchanged()

    def head(self, n: int = 5) -> DataFrame:
        """Read a bounded preview without loading or counting the whole source."""
        _positive_int(n, "n", maximum=MAX_BATCH_ROWS, zero=True)
        self.assert_unchanged()
        if n == 0:
            return as_frame(pd.DataFrame(columns=self._columns))
        parts = []
        remaining = n
        iterator = self.iter_batches(batch_rows=n)
        try:
            for batch in iterator:
                parts.append(batch.iloc[:remaining])
                remaining -= min(remaining, len(batch))
                if remaining == 0:
                    break
        finally:
            iterator.close()
        result = pd.concat(parts) if parts else pd.DataFrame(columns=self._columns)
        result.attrs["metadata"] = deepcopy(self._metadata)
        return as_frame(result)


def scan(path: str | Path) -> Dataset:
    """Open a replayable local CSV file or Parquet file/directory lazily.

    Parquet row counts come from file footers. CSV counts remain unknown until
    an estimator consumes a pass. Data values are never loaded by ``scan``.
    Directory sources may include Hive partition columns. Every file must have
    the same physical Arrow schema; heterogeneous schemas are rejected.
    """
    if not isinstance(path, (str, Path)) or (isinstance(path, str) and "://" in path):
        raise DataError("scan accepts local paths only; network URLs are not supported.",
                        "INVALID_SOURCE")
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise DataError("The dataset path does not exist.", "FILE_NOT_FOUND") from exc
    source = Dataset()
    source._path = root
    if root.is_dir():
        source._kind = "parquet"
        source._files = _discover_parquet(root)
    elif root.suffix.lower() in {".csv", ".parquet"}:
        source._kind = "csv" if root.suffix.lower() == ".csv" else "parquet"
        source._files = (root,)
    else:
        raise DataError("scan supports CSV files and Parquet files or directories.",
                        "UNSUPPORTED_FORMAT")
    source._manifest = tuple(_file_identity(file) for file in source._files)
    try:
        if source._kind == "csv":
            from openecon.source_readers import preflight_csv
            preflight_csv(root, header_only=True)
            with root.open(encoding="utf-8-sig", newline="") as stream:
                source._columns = _columns(next(csv.reader(stream), []))
        else:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
            physical_schema = None
            index_columns = None
            row_count = 0
            source_metadata = None
            footer_bytes = 0
            for file in source._files:
                from openecon.source_readers import preflight_parquet, MAX_ROW_GROUP_BYTES
                footer_bytes += preflight_parquet(file)
                if footer_bytes > 64 * 1024 * 1024:
                    raise DataError("Parquet directory exceeds its 64 MiB total footer budget.", "READER_LIMIT")
                with pq.ParquetFile(file, thrift_string_size_limit=2*1024*1024,
                                    thrift_container_size_limit=100000) as reader:
                    if any(reader.metadata.row_group(i).total_byte_size > MAX_ROW_GROUP_BYTES
                           for i in range(reader.metadata.num_row_groups)):
                        raise DataError("Parquet row group exceeds its 256 MiB uncompressed parser budget; rewrite with smaller row groups.", "READER_LIMIT")
                    schema = reader.schema_arrow
                    pandas_record = (schema.metadata or {}).get(b"pandas", b"{}")
                    if len(pandas_record) > 2 * 1024 * 1024:
                        raise DataError("Parquet index metadata exceeds the bounded metadata budget.", "METADATA_LIMIT")
                    parsed_pandas = json.loads(pandas_record)
                    range_indexes = [record for record in parsed_pandas.get("index_columns", [])
                                     if isinstance(record, dict) and record.get("kind") == "range"]
                    if len(source._files) == 1 and range_indexes:
                        if (len(range_indexes) != 1
                                or any(type(range_indexes[0].get(key)) is not int for key in ("start", "stop", "step"))
                                or range_indexes[0]["step"] == 0):
                            raise DataError("Parquet range index metadata is invalid.", "INVALID_SOURCE")
                        source._range_index = range_indexes[0]
                    current_indexes = tuple(name for name in parsed_pandas.get("index_columns", []) if isinstance(name, str))
                    if index_columns is not None and current_indexes != index_columns:
                        raise DataError("Parquet files must share physical index columns.", "SCHEMA_MISMATCH")
                    if set(current_indexes) - set(schema.names):
                        raise DataError("Parquet physical index metadata is invalid.", "INVALID_SOURCE")
                    index_columns = current_indexes
                    attrs = (schema.metadata or {}).get(b"PANDAS_ATTRS", b"{}")
                    if len(attrs) > 2 * 1024 * 1024:
                        raise DataError("Parquet attribute metadata exceeds the bounded metadata budget.", "METADATA_LIMIT")
                    parsed = json.loads(attrs)
                    metadata = parsed.get("metadata", {}) if isinstance(parsed, dict) else {}
                    if not isinstance(metadata, dict):
                        raise DataError("Parquet dataset metadata must be a dictionary.", "INVALID_SOURCE")
                    missing_fields = metadata.get("extended_missing_storage", {})
                    if (not isinstance(missing_fields, dict)
                        or any(not isinstance(key, str) or not isinstance(value, str)
                               or key not in schema.names or value not in schema.names
                               or key == value or value in current_indexes
                               or str(schema.field(value).type) not in {"string", "large_string"}
                               for key, value in missing_fields.items())
                        or len(set(missing_fields.values())) != len(missing_fields)
                        or set(missing_fields) & set(missing_fields.values())):
                        raise DataError("Parquet missing-code storage metadata is invalid.", "INVALID_SOURCE")
                    current_metadata = metadata_hash(metadata)
                    if source_metadata is not None and current_metadata != source_metadata:
                        raise DataError("Parquet files must share dataset attribute metadata.", "SCHEMA_MISMATCH")
                    source_metadata = current_metadata
                    source._metadata = metadata
                    if physical_schema is not None and not schema.equals(physical_schema,
                                                                         check_metadata=False):
                        raise DataError("Parquet files must share the same physical schema.",
                                        "SCHEMA_MISMATCH")
                    physical_schema = schema
                    row_count += reader.metadata.num_rows
            source._arrow_dataset = ds.dataset(
                [str(file) for file in source._files], format=ds.ParquetFileFormat(
                    default_fragment_scan_options=ds.ParquetFragmentScanOptions(
                        thrift_string_size_limit=2*1024*1024, thrift_container_size_limit=100000,
                        pre_buffer=False)),
                partitioning="hive" if root.is_dir() else None,
                partition_base_dir=str(root) if root.is_dir() else None,
            )
            source._index_columns = index_columns or ()
            source._columns = _columns([name for name in source._arrow_dataset.schema.names
                                       if name not in source._index_columns
                                       and name not in source._metadata.get("extended_missing_storage", {}).values()])
            source._row_count = int(row_count)
        source.assert_unchanged()
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"The dataset metadata could not be read: {type(exc).__name__}.",
                        "INVALID_SOURCE") from exc
    identity = hashlib.sha256(json.dumps(
        [(str(file), values) for file, values in zip(source._files, source._manifest, strict=True)],
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    source._provenance = {"kind": source._kind, "path": str(root),
                          "file_count": len(source._files), "row_count": source._row_count,
                          "manifest_identity": identity,
                          "integrity": "file stat manifest and engine pass-content digests"}
    return source
