"""Persist public summary metadata in ordinary console-saveable table cells."""

import json
import math
from collections.abc import Mapping

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace
from .core import table, TableSet, _json_safe
from .state_lifecycle import _encoded_json_admission, _metadata_geometry


def _ivqr_export_admission(output):
    """Admit complete IVQR metadata/table buffers before portable conversion.

    Other procedures retain their existing permissive metadata conversion.
    IVQR's complete state is already finite JSON; table buffers are charged
    separately before their values/labels are copied or serialized.
    """
    state = output.attrs.get("state")
    if not isinstance(state, Mapping) or (
        state.get("schema") != "openecon.ivquantile.v1"
        and state.get("schema_version") != "openecon.ivquantile.v1"
    ):
        return
    metadata, _ = _metadata_geometry({"attrs": output.attrs, "title": output.title})
    for name, frame in output.items():
        for label in (name, *frame.index.names, *frame.columns.names):
            size, _ = _metadata_geometry(label)
            metadata += size
    cells = sum(frame.shape[0] * frame.shape[1] for frame in output.values())
    labels = sum(sum(frame.shape) for frame in output.values())
    buffers = 256 * (cells + labels + len(output))
    plan_workspace(
        "IVQR complete summary export",
        {"complete JSON metadata and copies": 8 * metadata, "table/axis copy buffers": buffers},
    )
    # After the dimension plan, bounded per-column bookkeeping can measure
    # resident object strings without creating to_numpy/tolist output buffers.
    # The multiplier covers escaped strings, labels and serializer copies.
    table_bytes = sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        + int(frame.columns.memory_usage(deep=True))
        + 256 * (len(frame.index.names) + len(frame.columns.names))
        for frame in output.values()
    )
    encoded_bound = metadata + buffers + 16 * table_bytes
    if encoded_bound > 32 * 1024**2:
        raise AnalysisError("resource_limit", "Summary JSON exceeds the 32 MiB export domain.")
    plan_workspace(
        "IVQR complete summary export",
        {
            "complete JSON metadata and copies": 8 * metadata,
            "table/axis copy buffers": buffers + 16 * table_bytes,
            "complete encoded export buffers": 4 * encoded_bound,
        },
    )


def saved_summary(output):
    """Save scalar settings in cells; large state uses complete JSON export.

    Console previews truncate strings at 500 chars and tables at 50 rows.
    Do not represent a preview as complete matrices or bootstrap persistence.
    """
    _ivqr_export_admission(output)
    rows = []
    for key, value in output.attrs.items():
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        if len(encoded) > 400:
            encoded = json.dumps(
                {"full_state": "oe.summary_state(output)", "characters": len(encoded)}
            )
        rows.append([key, encoded])
    output["settings"] = table(rows, columns=["setting", "json"])
    return output


def summary_state(output):
    """Complete finite versioned JSON for a TableSet; save to an owned file.

    Includes every table row, typed row labels and all metadata; independent
    of console preview limits. Numerical state is not refitted on restoration.
    """
    if not isinstance(output, TableSet):
        raise AnalysisError("invalid_result", "Pass a structured TableSet summary.")
    _ivqr_export_admission(output)
    payload = {
        "schema": "openecon.summary.v1",
        "title": output.title,
        "attrs": output.attrs,
        "tables": {
            name: {
                "columns": list(frame.columns),
                "index": list(frame.index),
                "data": frame.to_numpy().tolist(),
                "index_names": list(frame.index.names),
                "column_names": list(frame.columns.names),
            }
            for name, frame in output.items()
        },
    }
    encoded = json.dumps(_json_safe(payload), allow_nan=False, sort_keys=True)
    if len(encoded.encode()) > 32 * 1024**2:
        raise AnalysisError("resource_limit", "Summary JSON exceeds the 32 MiB export domain.")
    return encoded


def restore_summary(state):
    """Restore all tables/metadata from summary_state JSON without estimation."""
    if not isinstance(state, str):
        raise AnalysisError("invalid_state", "Supply summary JSON up to 32 MiB.")
    _encoded_json_admission(state, limit=32 * 1024**2, operation="complete summary JSON decoding")
    try:

        def reject(value):
            raise ValueError("Nonfinite JSON")

        value = json.loads(state, parse_constant=reject)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite JSON number")
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        if (
            value["schema"] != "openecon.summary.v1"
            or not isinstance(value["attrs"], dict)
            or not isinstance(value["tables"], dict)
        ):
            raise ValueError("Unknown schema")
        tables = {}
        for name, frame in value["tables"].items():
            if (
                not isinstance(name, str)
                or len(frame["index"]) != len(frame["data"])
                or any(len(row) != len(frame["columns"]) for row in frame["data"])
            ):
                raise ValueError("Invalid table dimensions")
            output = table(frame["data"], columns=frame["columns"], index=frame["index"])
            for field, axis in (("index_names", 0), ("column_names", 1)):
                # These optional fields retain the names of ordinary table
                # axes without changing label identities or the v1 schema.
                # Legacy payloads omitted them and continue to restore.
                if field not in frame:
                    continue
                names = frame[field]
                labels = output.index if axis == 0 else output.columns
                if not isinstance(names, list) or len(names) != labels.nlevels \
                        or any(item is not None and not isinstance(item, (str, bool, int, float))
                               for item in names):
                    raise ValueError("Invalid axis names")
                output = output.rename_axis(names, axis=axis)
            tables[name] = output
        return TableSet(tables, title=value["title"], **value["attrs"])
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise AnalysisError(
            "invalid_state", "Summary schema, finite JSON or table dimensions are invalid."
        ) from exc
