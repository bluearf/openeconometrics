"""Persist public summary metadata in ordinary console-saveable table cells."""

import json
import math

from openecon.analysis_contracts import AnalysisError
from .core import table, TableSet, _json_safe


def saved_summary(output):
    """Save scalar settings in cells; large state uses complete JSON export.

    Console previews truncate strings at 500 chars and tables at 50 rows.
    Do not represent a preview as complete matrices or bootstrap persistence.
    """
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
    if not isinstance(state, str) or len(state.encode()) > 32 * 1024**2:
        raise AnalysisError("invalid_state", "Supply summary JSON up to 32 MiB.")
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
