"""Replay-checked block transforms and owned disk joins/reshapes for Dataset.

Caller functions own their allocations/side effects. This layer bounds the
library's blocks, SQLite state, work and admitted output expansion; it never
collects a complete source into pandas or invokes a cloud service.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import marshal
import math
import os
import pickle
import sqlite3
import struct
import tempfile

import pandas as pd

from openecon.data import DataError, metadata_hash
from openecon.dataset import Dataset, _columns, _frame_schema, _positive_int

BLOCK_ROWS = 8192
DEFAULT_MEMORY_BYTES = 64 * 1024 * 1024
DEFAULT_DISK_BYTES = 1024 * 1024 * 1024
MAX_ROW_BYTES = 4 * 1024 * 1024


def _charge(digest, value):
    raw = pickle.dumps(value, protocol=5)  # Generated values only; never load user pickle files.
    if len(raw) > MAX_ROW_BYTES:
        raise DataError("Preparation row/index exceeds 4 MiB.", "PREPARATION_LIMIT")
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)
    return raw


def _nullable(dtype):
    if isinstance(dtype, pd.CategoricalDtype) or isinstance(
        dtype, pd.api.extensions.ExtensionDtype
    ):
        return dtype
    if pd.api.types.is_integer_dtype(dtype):
        return (
            pd.UInt64Dtype() if pd.api.types.is_unsigned_integer_dtype(dtype) else pd.Int64Dtype()
        )
    if pd.api.types.is_bool_dtype(dtype):
        return pd.BooleanDtype()
    return dtype


def _schema(frame):
    return dict(frame.dtypes.items())


def _index_schema(index):
    levels = index.levels if isinstance(index, pd.MultiIndex) else [index]
    return (type(index).__name__, tuple(index.names), tuple(str(level.dtype) for level in levels))


def _empty_schema(source):
    if getattr(source, "_declared_schema", None) is not None:
        return source._declared_schema.copy()
    if source._kind == "frame":
        return _schema(source._frame)
    if source._kind == "parquet":
        import pyarrow as pa

        if any(
            pa.types.is_dictionary(source._arrow_dataset.schema.field(name).type)
            for name in source.columns
        ):
            raise DataError(
                "Declare category schema with map before preparing an empty Parquet source.",
                "SCHEMA_MISMATCH",
            )
        return {
            name: pd.ArrowDtype(source._arrow_dataset.schema.field(name).type)
            for name in source.columns
        }
    raise DataError(
        "Declare schema with map before preparing an empty factory/CSV source.", "SCHEMA_MISMATCH"
    )


def _metadata(source, columns, *, rename=None):
    value = source.metadata
    rename = rename or {name: name for name in columns}
    for field in ("column_labels", "value_labels", "original_types", "readstat_types"):
        if isinstance(value.get(field), dict):
            value[field] = {
                rename[name]: record for name, record in value[field].items() if name in rename
            }
    missing = (
        set(value.pop("extended_missing_storage", {}))
        | set(value.pop("extended_missing_columns", []))
        | set(value.get("extended_missing_codes", {}))
    )
    value.pop("extended_missing_codes", None)
    value["extended_missing_columns"] = [
        rename[name] for name in source.columns if name in missing and name in rename
    ]
    return value


def _tags(frame):
    codes = frame.attrs.get(
        "extended_missing_codes", frame.attrs.get("metadata", {}).get("extended_missing_codes", {})
    )
    if codes and not frame.index.is_unique:
        raise DataError(
            "Retained missing codes require unique original indices within each block.",
            "INVALID_TRANSFORM",
        )

    def identity(value):
        return (
            tuple(identity(item) for item in value) if isinstance(value, (list, tuple)) else value
        )

    result = {}
    for column, mapping in codes.items():
        for code, positions in mapping.items():
            for index in positions:
                index = identity(index)
                result.setdefault(index, {})[column] = code
    return result


def _attach_tags(frame, tags):
    frame.attrs.pop("extended_missing_codes", None)
    codes = {}
    for position, (index, mapping) in enumerate(zip(frame.index, tags, strict=True)):
        for column, code in mapping.items():
            if column in frame and bool(pd.isna(frame[column].iloc[position])):
                codes.setdefault(column, {}).setdefault(code, []).append(index)
    frame.attrs["extended_missing_codes"] = codes


class _Plan:
    def __init__(
        self,
        parents,
        operation,
        *,
        memory_bytes=DEFAULT_MEMORY_BYTES,
        disk_bytes=DEFAULT_DISK_BYTES,
        max_rows=10_000_000,
        max_work=100_000_000,
    ):
        for name, value in (
            ("memory_bytes", memory_bytes),
            ("disk_bytes", disk_bytes),
            ("max_rows", max_rows),
            ("max_work", max_work),
        ):
            _positive_int(value, name)
        self.parents, self.operation = tuple(parents), operation
        self.memory, self.disk, self.max_rows, self.max_work = (
            memory_bytes,
            disk_bytes,
            max_rows,
            max_work,
        )
        self.input_baseline = self.output_baseline = None
        self.active = False
        self.depth = 1 + max(
            (getattr(parent, "_preparation_depth", 0) for parent in self.parents), default=0
        )
        if self.depth > 16:
            raise DataError(
                "Preparation plan exceeds 16 bounded stages; persist a checked intermediate explicitly.",
                "PREPARATION_LIMIT",
            )
        self.owner = None
        self.receipt = {}

    def inputs(self, digests):
        value = tuple(digests)
        if self.input_baseline is not None and self.input_baseline != value:
            raise DataError(
                "Preparation source changed between complete passes.", "NON_REPLAYABLE_SOURCE"
            )
        self.input_baseline = value

    def make(self, columns, factory, *, metadata, row_count=None, schema=None):
        def replay():
            if self.active:
                raise DataError(
                    "Close the pending preparation iterator before starting another pass.",
                    "NON_REPLAYABLE_TRANSFORM",
                )
            self.active = True
            digest, rows = hashlib.sha256(), 0
            expected_schema = None
            self.steps = 0
            try:
                for frame in factory():
                    if tuple(frame.columns) != tuple(columns):
                        raise DataError(
                            "Preparation columns differ from the declared schema.",
                            "SCHEMA_MISMATCH",
                        )
                    current = (_frame_schema(frame), _index_schema(frame.index))
                    if expected_schema is None:
                        expected_schema = current
                        _charge(digest, (current, metadata_hash(metadata)))
                    elif expected_schema != current:
                        raise DataError(
                            "Preparation dtype/category schema changed within a pass.",
                            "SCHEMA_MISMATCH",
                        )
                    if int(frame.memory_usage(index=True, deep=True).sum()) > self.memory:
                        raise DataError(
                            "Preparation output exceeds its owned block budget.",
                            "PREPARATION_LIMIT",
                        )
                    tags = _tags(frame)
                    frame.attrs["metadata"] = deepcopy(metadata)
                    for index, values in zip(
                        frame.index, frame.itertuples(index=False, name=None), strict=True
                    ):
                        _charge(digest, (index, values, tags.get(index, {})))
                    rows += len(frame)
                    if rows > self.max_rows:
                        raise DataError("Preparation output exceeds max_rows.", "PREPARATION_LIMIT")
                    yield frame
                fingerprint = (rows, digest.hexdigest())
                if self.output_baseline is not None and self.output_baseline != fingerprint:
                    raise DataError(
                        "Preparation produced different output on replay.",
                        "NON_REPLAYABLE_TRANSFORM",
                    )
                self.output_baseline = fingerprint
                self.owner._row_count = rows
                self.owner._provenance["preparation"]["verified_rows"] = rows
                self.owner._preparation_receipt = {
                    **self.receipt,
                    "rows": rows,
                    "output_digest": digest.hexdigest(),
                    "complete_pass_verified": True,
                }
            finally:
                self.active = False
                for parent in self.parents:
                    parent.assert_unchanged()

        self.owner = Dataset.from_batches(
            replay, columns=columns, row_count=row_count, metadata=metadata
        )
        self.owner._parents = self.parents
        self.owner._preparation_depth = self.depth
        self.owner._declared_schema = schema
        self.owner._provenance["preparation"] = dict(
            operation=self.operation,
            max_rows=self.max_rows,
            max_work=self.max_work,
            memory_bytes=self.memory,
            disk_bytes=self.disk,
            replay="complete input and output digests include original typed indices and schemas",
            caller_function_allocation="caller-owned; cannot be undone by this layer",
        )
        return self.owner

    def work(self, amount=1):
        self.steps += amount
        if self.steps > self.max_work:
            raise DataError("Preparation exceeds its work budget.", "PREPARATION_LIMIT")


def _read(source, plan, digest, columns=None, *, allow_schema_variation=False):
    iterator = source.iter_batches(columns=columns, batch_rows=BLOCK_ROWS)
    schema = None
    try:
        for frame in iterator:
            plan.work(len(frame) * max(1, len(frame.columns)))
            if int(frame.memory_usage(index=True, deep=True).sum()) > plan.memory:
                raise DataError(
                    "Preparation input block exceeds its buffer budget; project columns first.",
                    "PREPARATION_LIMIT",
                )
            current = (_frame_schema(frame), _index_schema(frame.index))
            if schema is None:
                schema = current
                _charge(digest, (current, metadata_hash(source.metadata)))
            elif schema != current:
                if not allow_schema_variation:
                    raise DataError(
                        "Source dtype/category schema changed between blocks; declare a typed map first.",
                        "SCHEMA_MISMATCH",
                    )
                schema = current
                _charge(digest, current)
            tags = _tags(frame)
            for index, values in zip(
                frame.index, frame.itertuples(index=False, name=None), strict=True
            ):
                _charge(digest, (index, values, tags.get(index, {})))
            yield frame
    finally:
        iterator.close()


def _callable_identity(function):
    if not callable(function):
        raise DataError("A preparation transform must be callable.")
    code = getattr(
        function, "__code__", getattr(getattr(function, "__call__", None), "__code__", None)
    )
    return hashlib.sha256(
        marshal.dumps(code)
        if code is not None
        else (type(function).__module__ + ":" + type(function).__qualname__).encode()
    ).hexdigest()


def project(source, columns, **limits):
    names = source._projection(columns)
    plan = _Plan([source], "project", **limits)

    def factory():
        digest = hashlib.sha256()
        for frame in _read(source, plan, digest, names):
            tags = _tags(frame)
            _attach_tags(frame, [tags.get(index, {}) for index in frame.index])
            yield frame
        plan.inputs([digest.hexdigest()])

    declared = getattr(source, "_declared_schema", None)
    return plan.make(
        names,
        factory,
        metadata=_metadata(source, names),
        row_count=source.row_count,
        schema={name: declared[name] for name in names} if declared else None,
    )


def filter_rows(source, predicate, *, missing="drop", **limits):
    identity = _callable_identity(predicate)
    if missing not in {"drop", "keep", "error"}:
        raise DataError("Filter missing must be drop, keep or error.")
    plan = _Plan([source], "filter:" + identity, **limits)

    def factory():
        if _callable_identity(predicate) != identity:
            raise DataError("Filter code identity changed.", "NON_REPLAYABLE_TRANSFORM")
        digest = hashlib.sha256()
        for frame in _read(source, plan, digest):
            mask = predicate(frame.copy(deep=True))
            if (
                not isinstance(mask, pd.Series)
                or not mask.index.equals(frame.index)
                or not pd.api.types.is_bool_dtype(mask.dtype)
            ):
                raise DataError(
                    "Filter must return an aligned nullable boolean Series.", "INVALID_TRANSFORM"
                )
            if missing == "error" and mask.isna().any():
                raise DataError("Filter returned missing decisions.", "INVALID_TRANSFORM")
            output = frame.loc[mask.fillna(missing == "keep")].copy()
            tags = _tags(frame)
            _attach_tags(output, [tags.get(index, {}) for index in output.index])
            yield output
        if _callable_identity(predicate) != identity:
            raise DataError(
                "Filter code identity changed during its pass.", "NON_REPLAYABLE_TRANSFORM"
            )
        plan.inputs([digest.hexdigest()])

    return plan.make(
        source.columns,
        factory,
        metadata=_metadata(source, source.columns),
        schema=getattr(source, "_declared_schema", None),
    )


def map_rows(source, function, *, schema, metadata=None, **limits):
    identity = _callable_identity(function)
    if not isinstance(schema, dict):
        raise DataError("map requires a name-to-pandas-dtype schema.")
    names = _columns(list(schema))
    declared = {name: pd.api.types.pandas_dtype(dtype) for name, dtype in schema.items()}
    plan = _Plan([source], "map:" + identity, **limits)

    def factory():
        if _callable_identity(function) != identity:
            raise DataError("Map code identity changed.", "NON_REPLAYABLE_TRANSFORM")
        digest = hashlib.sha256()
        for frame in _read(source, plan, digest, allow_schema_variation=True):
            output = function(frame.copy(deep=True))
            if (
                not isinstance(output, pd.DataFrame)
                or tuple(output.columns) != names
                or not output.index.identical(frame.index)
            ):
                raise DataError(
                    "map must return the declared columns at unchanged original indices/row count.",
                    "INVALID_TRANSFORM",
                )
            if any(
                not pd.api.types.is_dtype_equal(output[name].dtype, dtype)
                for name, dtype in declared.items()
            ):
                raise DataError(
                    "map output differs from its declared dtype/category schema; cast explicitly in the function.",
                    "SCHEMA_MISMATCH",
                )
            tags = _tags(frame)
            _attach_tags(output, [tags.get(index, {}) for index in output.index])
            yield output
        if _callable_identity(function) != identity:
            raise DataError(
                "Map code identity changed during its pass.", "NON_REPLAYABLE_TRANSFORM"
            )
        plan.inputs([digest.hexdigest()])

    return plan.make(
        names,
        factory,
        metadata=deepcopy(metadata) if metadata is not None else _metadata(source, names),
        row_count=source.row_count,
        schema=declared,
    )


def _key(values, nulls):
    encoded = []
    for value in values:
        if type(value).__module__.split(".")[0] == "numpy":
            value = value.item()
        if (
            value is None
            or value is pd.NA
            or value is pd.NaT
            or (isinstance(value, float) and math.isnan(value))
            or (isinstance(value, Decimal) and value.is_nan())
        ):
            if nulls == "never":
                return None
            encoded.append(["null"])
            continue
        if isinstance(value, bool):
            encoded.append(["bool", value])
        elif isinstance(value, (int, float, Decimal)):
            if isinstance(value, float) and not math.isfinite(value):
                raise DataError("Join/reshape keys must be finite.", "INVALID_TRANSFORM")
            if isinstance(value, Decimal) and not value.is_finite():
                raise DataError("Join/reshape keys must be finite.", "INVALID_TRANSFORM")
            numerator, denominator = (
                (value, 1) if isinstance(value, int) else value.as_integer_ratio()
            )
            encoded.append(["number", str(numerator), str(denominator)])
        elif isinstance(value, str):
            encoded.append(["string", value])
        elif isinstance(value, (datetime, date)):
            encoded.append([type(value).__name__, value.isoformat()])
        else:
            raise DataError(
                "Join/reshape keys support nullable scalar numbers, strings, booleans and dates.",
                "INVALID_TRANSFORM",
            )
    result = json.dumps(encoded, allow_nan=False, separators=(",", ":")).encode()
    if len(result) > 16384:
        raise DataError("Join/reshape key exceeds 16 KiB.", "PREPARATION_LIMIT")
    return result


class _Disk:
    def __init__(self, plan):
        if plan.disk < 4096:
            raise DataError(
                "Preparation disk budget cannot admit one SQLite page.", "PREPARATION_LIMIT"
            )
        self.plan = plan
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".openecon-preparation-", dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        )
        self.connection = sqlite3.connect(os.path.join(self.temporary.name, "rows.sqlite"))
        self.connection.execute("PRAGMA page_size=4096")
        self.connection.execute(f"PRAGMA max_page_count={plan.disk // 4096}")
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute(f"PRAGMA cache_size=-{max(1, min(16384, plan.memory // 4096))}")
        self.peak = 0
        self.connection.set_progress_handler(self.progress, 2000)

    def progress(self):
        try:
            self.plan.work(2000)
            self.check()
            return 0
        except DataError:
            return 1

    def check(self):
        from pathlib import Path

        size = sum(
            path.stat().st_size for path in Path(self.temporary.name).iterdir() if path.is_file()
        )
        self.peak = max(self.peak, size)
        if size > self.plan.disk:
            raise DataError("Preparation scratch exceeds its disk budget.", "PREPARATION_LIMIT")

    def close(self):
        self.connection.close()
        self.temporary.cleanup()
        self.plan.receipt = {
            **self.plan.receipt,
            **dict(
                disk_peak_bytes=self.peak,
                disk_budget_bytes=self.plan.disk,
                scratch_removed=not os.path.exists(self.temporary.name),
                source_unchanged=True,
            ),
        }


def _stage(source, plan, disk, table, keys, nulls):
    connection = disk.connection
    connection.execute(f"CREATE TABLE {table}(seq INTEGER PRIMARY KEY, key BLOB, payload BLOB)")
    connection.execute(f"CREATE INDEX {table}_lookup ON {table}(key,seq)")
    digest, count, schema = hashlib.sha256(), 0, None
    for frame in _read(source, plan, digest):
        schema = _schema(frame)
        positions = [frame.columns.get_loc(key) for key in keys]
        tags = _tags(frame)
        for index, values in zip(
            frame.index, frame.itertuples(index=False, name=None), strict=True
        ):
            key = _key([values[position] for position in positions], nulls)
            payload = pickle.dumps((index, values, tags.get(index, {})), protocol=5)
            if len(payload) > MAX_ROW_BYTES:
                raise DataError("Preparation row exceeds 4 MiB.", "PREPARATION_LIMIT")
            connection.execute(f"INSERT INTO {table} VALUES(?,?,?)", (count, key, payload))
            count += 1
        disk.check()
    connection.commit()
    disk.check()
    return count, digest.hexdigest(), schema or _empty_schema(source)


def _frames(records, schema, plan):
    rows, indices, tags, size = [], [], [], 0

    def frame():
        result = pd.DataFrame.from_records(rows, columns=list(schema))
        result.index = pd.Index(indices, dtype=object, tupleize_cols=False, name="source_indices")
        for name, dtype in schema.items():
            result[name] = result[name].astype(dtype)
        _attach_tags(result, tags)
        return result

    for values, index, codes, cost in records:
        plan.work(len(schema))
        cost = max(cost * 4, len(schema) * 256)
        if cost > plan.memory:
            raise DataError(
                "Prepared row exceeds its planned owned buffer budget.", "PREPARATION_LIMIT"
            )
        if rows and (size + cost > plan.memory or len(rows) == BLOCK_ROWS):
            yield frame()
            rows, indices, tags, size = [], [], [], 0
        rows.append(values)
        indices.append(index)
        tags.append(codes)
        size += cost
    if rows:
        yield frame()


def join(
    left, right, *, on, how="inner", validate="m:1", nulls="never", suffixes=("_x", "_y"), **limits
):
    if not isinstance(right, Dataset):
        raise DataError("join requires a replayable Dataset on both sides.")
    keys = left._projection([on] if isinstance(on, str) else on)
    right._projection(keys)
    if (
        how not in {"inner", "left", "right", "outer"}
        or validate not in {"1:1", "m:1", "1:m", "m:m"}
        or nulls not in {"never", "equal"}
    ):
        raise DataError("Invalid join how/validate/nulls policy.")
    if len(suffixes) != 2 or any(not isinstance(value, str) for value in suffixes):
        raise DataError("Join suffixes must be two strings.")
    shared = set(left.columns) & set(right.columns) - set(keys)
    lnames = {name: name + suffixes[0] if name in shared else name for name in left.columns}
    rnames = {
        name: name + suffixes[1] if name in shared else name
        for name in right.columns
        if name not in keys
    }
    columns = _columns([*lnames.values(), *rnames.values()])
    plan = _Plan([left, right], "disk join", **limits)
    metadata = _metadata(left, columns, rename=lnames)
    other = _metadata(right, columns, rename=rnames)
    for field in ("column_labels", "value_labels", "original_types", "readstat_types"):
        metadata[field] = {**metadata.get(field, {}), **other.get(field, {})}
    metadata["extended_missing_columns"] = [
        *metadata.get("extended_missing_columns", []),
        *other.get("extended_missing_columns", []),
    ]

    def factory():
        disk = _Disk(plan)
        connection = disk.connection
        try:
            nl, dl, ls = _stage(left, plan, disk, "L", keys, nulls)
            nr, dr, rs = _stage(right, plan, disk, "R", keys, nulls)
            plan.inputs([dl, dr])
            if any(not pd.api.types.is_dtype_equal(ls[name], rs[name]) for name in keys):
                raise DataError(
                    "Join key dtypes/categories must agree; normalize explicitly with a declared map.",
                    "SCHEMA_MISMATCH",
                )
            for table, unique in (
                ("L", validate in {"1:1", "1:m"}),
                ("R", validate in {"1:1", "m:1"}),
            ):
                if (
                    unique
                    and connection.execute(
                        f"SELECT 1 FROM {table} WHERE key IS NOT NULL GROUP BY key HAVING COUNT(*)>1 LIMIT 1"
                    ).fetchone()
                ):
                    raise DataError("Join duplicate keys violate validate.", "DUPLICATE_KEY")
            pairs = 0
            for key, count in connection.execute(
                "SELECT key,COUNT(*) FROM L WHERE key IS NOT NULL GROUP BY key"
            ):
                pairs += (
                    count
                    * connection.execute("SELECT COUNT(*) FROM R WHERE key=?", (key,)).fetchone()[0]
                )
                if pairs > plan.max_rows:
                    raise DataError(
                        "Join expansion exceeds max_rows before output allocation.",
                        "PREPARATION_LIMIT",
                    )
            unmatched_l = connection.execute(
                "SELECT COUNT(*) FROM L WHERE NOT EXISTS(SELECT 1 FROM R WHERE R.key=L.key)"
            ).fetchone()[0]
            unmatched_r = connection.execute(
                "SELECT COUNT(*) FROM R WHERE NOT EXISTS(SELECT 1 FROM L WHERE L.key=R.key)"
            ).fetchone()[0]
            total = (
                pairs
                + (unmatched_l if how in {"left", "outer"} else 0)
                + (unmatched_r if how in {"right", "outer"} else 0)
            )
            if total > plan.max_rows:
                raise DataError(
                    "Join expansion exceeds max_rows before output allocation.", "PREPARATION_LIMIT"
                )
            schema = {
                **{
                    lnames[name]: _nullable(dtype) if how in {"right", "outer"} else dtype
                    for name, dtype in ls.items()
                },
                **{
                    rnames[name]: _nullable(dtype) if how in {"left", "outer"} else dtype
                    for name, dtype in rs.items()
                    if name in rnames
                },
            }
            plan.owner._declared_schema = schema
            right_positions = [right.columns.index(name) for name in rnames]
            key_positions = {name: right.columns.index(name) for name in keys}

            def record(lp, rp):
                li, lv, lt = pickle.loads(lp) if lp else (None, (pd.NA,) * len(left.columns), {})
                ri, rv, rt = pickle.loads(rp) if rp else (None, (pd.NA,) * len(right.columns), {})
                values = list(lv)
                if lp is None:
                    for name in keys:
                        values[left.columns.index(name)] = rv[key_positions[name]]
                values.extend(rv[position] for position in right_positions)
                tags = {
                    **{lnames[name]: code for name, code in lt.items() if name in lnames},
                    **{rnames[name]: code for name, code in rt.items() if name in rnames},
                }
                return values, (li, ri), tags, len(lp or b"") + len(rp or b"") + len(columns) * 128

            def records():
                for key, lp in connection.execute("SELECT key,payload FROM L ORDER BY seq"):
                    found = False
                    for (rp,) in connection.execute(
                        "SELECT payload FROM R WHERE key=? ORDER BY seq", (key,)
                    ):
                        found = True
                        yield record(lp, rp)
                    if not found and how in {"left", "outer"}:
                        yield record(lp, None)
                if how in {"right", "outer"}:
                    for (rp,) in connection.execute(
                        "SELECT payload FROM R WHERE NOT EXISTS(SELECT 1 FROM L WHERE L.key=R.key) ORDER BY seq"
                    ):
                        yield record(None, rp)

            yield from _frames(records(), schema, plan)
            plan.receipt = dict(input_left_rows=nl, input_right_rows=nr, admitted_output_rows=total)
        except sqlite3.Error as exc:
            raise DataError(
                "Disk preparation exceeded its work/disk budget or SQLite could not continue.",
                "PREPARATION_LIMIT",
            ) from exc
        finally:
            disk.close()

    return plan.make(columns, factory, metadata=metadata)


def reshape_long(source, *, id_vars, value_vars, variable="variable", value="value", **limits):
    identifiers = tuple(id_vars)
    if identifiers:
        source._projection(identifiers)
    values = source._projection(value_vars)
    if set(identifiers) & set(values):
        raise DataError("Long identifiers and value columns must be disjoint.")
    columns = _columns([*identifiers, variable, value])
    plan = _Plan([source], "reshape long", **limits)
    if source.row_count is not None and source.row_count * len(values) > plan.max_rows:
        raise DataError(
            "Long expansion exceeds max_rows before reading/allocating output.", "PREPARATION_LIMIT"
        )
    metadata = _metadata(source, identifiers)
    metadata["long_variables"] = {
        name: {
            field: source.metadata.get(field, {}).get(name)
            for field in ("column_labels", "value_labels", "original_types")
        }
        for name in values
    }
    if any(
        name in source.metadata.get("extended_missing_storage", {})
        or name in source.metadata.get("extended_missing_columns", [])
        or name in source.metadata.get("extended_missing_codes", {})
        for name in values
    ):
        metadata["extended_missing_columns"].append(value)

    def factory():
        digest = hashlib.sha256()
        count = 0
        for frame in _read(source, plan, digest):
            if any(
                not pd.api.types.is_dtype_equal(frame[name].dtype, frame[values[0]].dtype)
                for name in values
            ):
                raise DataError(
                    "Long value columns must share a lossless dtype/category schema; normalize explicitly.",
                    "SCHEMA_MISMATCH",
                )
            count += len(frame) * len(values)
            if count > plan.max_rows:
                raise DataError(
                    "Long expansion exceeds max_rows before allocating its block.",
                    "PREPARATION_LIMIT",
                )
            schema = {
                **{name: frame[name].dtype for name in identifiers},
                variable: pd.StringDtype(),
                value: frame[values[0]].dtype,
            }
            plan.owner._declared_schema = schema
            tags = _tags(frame)
            positions = {name: frame.columns.get_loc(name) for name in (*identifiers, *values)}

            def records():
                for index, row in zip(
                    frame.index, frame.itertuples(index=False, name=None), strict=True
                ):
                    for name in values:
                        selected = [
                            *[row[positions[key]] for key in identifiers],
                            name,
                            row[positions[name]],
                        ]
                        codes = {
                            key: code
                            for key, code in tags.get(index, {}).items()
                            if key in identifiers
                        }
                        if name in tags.get(index, {}):
                            codes[value] = tags[index][name]
                        cost = len(pickle.dumps((index, selected), protocol=5))
                        yield selected, (index, name), codes, cost

            yield from _frames(records(), schema, plan)
        plan.inputs([digest.hexdigest()])

    return plan.make(
        columns,
        factory,
        metadata=metadata,
        row_count=source.row_count * len(values) if source.row_count is not None else None,
    )


def reshape_wide(
    source, *, keys, variable, value, levels, duplicates="error", unknown="error", **limits
):
    keys = source._projection([keys] if isinstance(keys, str) else keys)
    source._projection([*keys, variable, value])
    if duplicates not in {"error", "first", "last"} or unknown not in {"error", "drop"}:
        raise DataError("Wide duplicates must be error/first/last and unknown levels error/drop.")
    if not isinstance(levels, (list, tuple)) or not 1 <= len(levels) <= 1000:
        raise DataError(
            "Wide requires 1 to 1000 explicit levels; implicit unbounded discovery is unsupported."
        )
    names = [str(level) for level in levels]
    encoded = [_key([level], "equal") for level in levels]
    if len(set(encoded)) != len(encoded):
        raise DataError("Wide levels must be exactly typed and unique.")
    positions = {key: index for index, key in enumerate(encoded)}
    columns = _columns([*keys, *names])
    plan = _Plan([source], "disk reshape wide", **limits)
    metadata = _metadata(source, keys)
    for field in ("column_labels", "value_labels", "original_types", "readstat_types"):
        original = source.metadata.get(field, {}).get(value)
        if original is not None:
            metadata.setdefault(field, {}).update({name: original for name in names})
    if (
        value in source.metadata.get("extended_missing_storage", {})
        or value in source.metadata.get("extended_missing_columns", [])
        or value in source.metadata.get("extended_missing_codes", {})
    ):
        metadata["extended_missing_columns"].extend(names)

    def factory():
        disk = _Disk(plan)
        connection = disk.connection
        try:
            connection.execute("CREATE TABLE groups(key BLOB PRIMARY KEY, seq INTEGER UNIQUE)")
            connection.execute(
                "CREATE TABLE cells(key BLOB, level INTEGER, payload BLOB, PRIMARY KEY(key,level))"
            )
            digest, count, schema = hashlib.sha256(), 0, None
            for frame in _read(source, plan, digest):
                schema = _schema(frame)
                indices = [frame.columns.get_loc(name) for name in keys]
                vp, xp = frame.columns.get_loc(variable), frame.columns.get_loc(value)
                tags = _tags(frame)
                for index, row in zip(
                    frame.index, frame.itertuples(index=False, name=None), strict=True
                ):
                    level = positions.get(_key([row[vp]], "equal"))
                    if level is None:
                        if unknown == "error":
                            raise DataError(
                                "Wide input contains an undeclared level.", "UNKNOWN_LEVEL"
                            )
                        continue
                    selected = tuple(row[position] for position in indices)
                    key = _key(selected, "equal")
                    payload = pickle.dumps(
                        (index, selected, row[xp], tags.get(index, {}).get(value)), protocol=5
                    )
                    if len(payload) > MAX_ROW_BYTES:
                        raise DataError("Wide cell/index exceeds 4 MiB.", "PREPARATION_LIMIT")
                    connection.execute("INSERT OR IGNORE INTO groups VALUES(?,?)", (key, count))
                    try:
                        if duplicates == "last":
                            connection.execute(
                                "INSERT INTO cells VALUES(?,?,?) ON CONFLICT(key,level) DO UPDATE SET payload=excluded.payload",
                                (key, level, payload),
                            )
                        else:
                            connection.execute(
                                "INSERT OR IGNORE INTO cells VALUES(?,?,?)"
                                if duplicates == "first"
                                else "INSERT INTO cells VALUES(?,?,?)",
                                (key, level, payload),
                            )
                    except sqlite3.IntegrityError as exc:
                        raise DataError(
                            "Wide contains duplicate key/level cells.", "DUPLICATE_CELL"
                        ) from exc
                    count += 1
                disk.check()
            plan.inputs([digest.hexdigest()])
            schema = schema or _empty_schema(source)
            groups = connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
            if groups > plan.max_rows or groups * len(levels) > plan.max_work:
                raise DataError(
                    "Wide row/cell expansion exceeds its output/work budget before allocation.",
                    "PREPARATION_LIMIT",
                )
            output_schema = {
                **{name: schema[name] for name in keys},
                **{name: _nullable(schema[value]) for name in names},
            }
            plan.owner._declared_schema = output_schema

            def records():
                for (key,) in connection.execute("SELECT key FROM groups ORDER BY seq"):
                    row = [pd.NA] * len(levels)
                    originals = [None] * len(levels)
                    codes, selected, cost = {}, None, len(columns) * 128
                    for level, payload in connection.execute(
                        "SELECT level,payload FROM cells WHERE key=? ORDER BY level", (key,)
                    ):
                        cost += len(payload)
                        if cost * 4 > plan.memory:
                            raise DataError(
                                "Wide group exceeds its planned owned buffer budget.",
                                "PREPARATION_LIMIT",
                            )
                        original, selected, cell, tag = pickle.loads(payload)
                        row[level], originals[level] = cell, original
                        if tag is not None:
                            codes[names[level]] = tag
                    yield (
                        [*selected, *row],
                        (selected, tuple(zip(levels, originals, strict=True))),
                        codes,
                        cost,
                    )

            yield from _frames(records(), output_schema, plan)
            plan.receipt = dict(
                admitted_output_rows=groups, admitted_output_cells=groups * len(levels)
            )
        except sqlite3.Error as exc:
            raise DataError(
                "Disk wide preparation exceeded its resource budget or SQLite could not continue.",
                "PREPARATION_LIMIT",
            ) from exc
        finally:
            disk.close()

    return plan.make(columns, factory, metadata=metadata)
