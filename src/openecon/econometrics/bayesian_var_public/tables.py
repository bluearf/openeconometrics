"""Whole dedicated BVAR tables and bounded complete-state JSON, never ResultBundle.

All array coordinates are retained. Positional axes are explicitly zero based;
the full typed source/terms/series/future-period descriptors travel in attrs.
Imports and numerical replay occur only after combined source/copy admission.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import json

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.econometrics.bayesian_var.admission import (
    MAX_JSON_BYTES, admit, integer, load_mapping, metadata_size)
from .transport import ROUTES, restore


def view(value):
    return value.__dict__ if hasattr(value, "__pydantic_fields__") else value


def unencoded(value):
    # The original 32 MiB UTF8/depth/value gate precedes JSON decoding.
    return load_mapping(value) if isinstance(value, (str, bytes, bytearray)) else view(value)


def geometry(value):
    """Rectangular primitive numeric tensor geometry; no tensor allocation."""
    if type(value) in (int, float, bool):
        return ()
    if isinstance(value, (tuple, list)) and value:
        shapes = [geometry(item) for item in value]
        if shapes[0] is not None and all(shape == shapes[0] for shape in shapes):
            return (len(value),) + shapes[0]
    return None


def admission(value, *, indent=None):
    if indent is not None:
        indent = integer(indent, "indent", high=4)
    raw = unencoded(value)
    size = metadata_size(raw)
    if not isinstance(raw, Mapping) or type(raw.get("schema_version")) is not str or raw["schema_version"] not in ROUTES:
        raise AnalysisError("invalid_state", "Supply one complete declared BVAR state.")
    root, state = raw, raw
    if "joint_draws" in root:
        root = view(root["joint_draws"])
    if "parent" in root:
        root = view(root["parent"])
    try:
        source = view(root["source"])
        source_values = source["source_values"]
        if (not isinstance(source_values, (list, tuple)) or not source_values
                or not isinstance(source_values[0], (list, tuple)) or not source_values[0]):
            raise AnalysisError("invalid_state", "Complete nonempty BVAR source columns are required.")
        n, m = len(source_values[0]), len(source["series"])
        max_work = integer(root["max_work"], "max_work", low=1)
        max_bytes = integer(root["max_bytes"], "max_bytes", low=1)
        joint = view(state.get("joint_draws", state))
        draws = joint.get("draws", 0)
        horizon = state.get("horizon", 0)
        irf = state["schema_version"] == "openecon.bayesian_var_impulse.v1"
        queries = len(state.get("design", ()))
        old = admit(n, m, root["lags"], root["intercept"], draws=draws,
                    horizon=horizon, queries=queries, irf=irf,
                    max_work=max_work, max_bytes=max_bytes,
                    index_bytes=metadata_size(source["source_index"]))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError("invalid_state", "Complete inherited BVAR geometry/budgets are required.") from exc
    # metadata_size charges >=32 bytes per node and all container edges. 64*size
    # bounds a full escaped JSON with depth32/indent<=4, decimal integers<=128bit,
    # finite float repr, keys<=1000, and strings<=10000, before output creation.
    escaped = 64*size
    if escaped > MAX_JSON_BYTES:
        raise AnalysisError("metadata_limit", "Conservative complete escaped BVAR state exceeds 32 MiB.")
    # The pinned source graph contains at most five original numerical replay
    # phases (forecast), each charged by the full original operation estimate.
    # Twelve estimates reserve nested phases plus metadata/copy work. This is a
    # bound relative to original declared work units, not CPU or process RSS.
    work = 12*old["estimated_work"] + 64*size
    if work > max_work:
        raise AnalysisError("work_limit", "Complete replay/tables/export copies exceed inherited max_work.")
    plan = plan_workspace("complete BVAR public replay/tables/export", {
        "original full replay numerical/source buffers": old["estimated_workspace_bytes"],
        "raw typed primitive snapshots and attrs copies": 12*size,
        "all table values coordinate labels frames and export snapshots": 16*size,
        "complete escaped JSON UTF8 and encoder buffers": 3*escaped,
    }, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    # Retain a top-level model until transport's complete primitive normalization:
    # __dict__ alone would hide forged extra/private typed metadata.
    admitted = value if hasattr(value, "__pydantic_fields__") else raw
    return admitted, plan.record() | {"estimated_work": work, "escaped_json_upper_bytes": escaped,
                                "scope_extension": "includes declared table/primitive/UTF8 copies; excludes process RSS"}


def primitive(value):
    from pydantic import BaseModel
    return BaseModel.model_dump(value, mode="python")


def leaves(value, path=()):
    value = view(value)
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif isinstance(value, (list, tuple)) and value and geometry(value) is not None:
        yield path, value, geometry(value)
    elif isinstance(value, (list, tuple)):
        for position, child in enumerate(value):
            yield from leaves(child, path + (str(position),))
    else:
        yield path, value, None


def array_rows(value, shape):
    if len(shape) == 1:
        return [[cell] for cell in value], list(range(shape[0])), ["value"]
    rows, positions = [], []
    for position in product(*(range(size) for size in shape[:-1])):
        node = value
        for coordinate in position:
            node = node[coordinate]
        rows.append(list(node))
        positions.append(position[0] if len(position) == 1 else position)
    return rows, positions, list(range(shape[-1]))


def bayes_var_tables(value):
    """Return every numeric field/whole covariance and explicit unavailable scalar.

    Full nested source, prior, posterior and cached draw primitives are retained
    both in tables and complete_state attrs; no preview slice is an export.
    Exact matrix-t/Student-t fields and finite-draw MC fields retain their names
    and original availability/interval/MC-error labels. No SE/t/p is invented.
    """
    raw, plan = admission(value)
    checked = restore(raw, _admitted=True)
    body = primitive(checked)
    import pandas as pd
    from openecon.econometrics.core import TableSet
    frames, scalar_rows = {}, []
    for path, item, shape in leaves(body):
        name = ".".join(path)
        if shape:
            rows, index, columns = array_rows(item, shape)
            if len(shape) > 2:
                index = pd.MultiIndex.from_tuples(index, names=["axis_"+str(i) for i in range(len(shape)-1)])
            # Mixed source arrays contain signed-int64 periods alongside floats.
            # Object cells preserve exact integers; float inference would round
            # valid periods above2**53 before the UI's existing safe-int formatter.
            frame = pd.DataFrame(rows, index=index, columns=columns, dtype=object)
            frame.attrs = {"complete_shape": shape, "zero_based_coordinate_axes": True,
                           "typed_field_path": path, "whole_array": True}
            frames[name] = frame
        else:
            scalar_rows.append((name, item, "unavailable" if item is None else "recorded"))
    frames["scalar_fields_and_availability"] = pd.DataFrame(scalar_rows, columns=["field", "value", "availability"], dtype=object)
    return TableSet(frames, title="Complete dedicated proper MNIW BVAR state", complete_state=body,
                    resource_plan=plan, numerical_contract="original unchanged exact posterior and declared finite-draw MC",
                    replay_contract="cached Bartlett/normal algebra; fit/RNG disabled; no seed-stream authentication",
                    display_contract="whole tables; native previews require explicit complete row/column batches")


def bayes_var_state_json(value, *, indent=None):
    """Export one complete checked BVAR state with bounded indent and UTF8 buffers."""
    raw, _ = admission(value, indent=indent)
    body = primitive(restore(raw, _admitted=True))
    payload = json.dumps(body, ensure_ascii=True, allow_nan=False, indent=indent,
                         separators=(",", ":") if indent is None else None)
    if len(payload.encode("utf-8")) > MAX_JSON_BYTES:
        raise AnalysisError("metadata_limit", "Complete BVAR JSON exceeds its fixed UTF8 limit.")
    return payload
