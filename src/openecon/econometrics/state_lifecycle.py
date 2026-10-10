"""Streaming capacity admission before saved-state parsers, copies and exports.

These are allocation bounds, not model validation or numerical tolerances.
Dense decoded objects can be much larger than their encoded JSON input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from pydantic import BaseModel

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace


def _error(message):
    raise AnalysisError("state_limit", message)


def _metadata_geometry(value):
    """Return conservative object bytes and indentation depth without a wide stack."""
    stack = [(iter((value,)), 0)]
    size = nodes = depth_sum = 0
    while stack:
        try:
            item = next(stack[-1][0])
        except StopIteration:
            stack.pop()
            continue
        depth = stack[-1][1]
        nodes += 1
        if depth > 64 or nodes > 2_000_000:
            _error("Saved state exceeds its bounded metadata nesting or item count.")
        depth_sum += depth + 1
        if isinstance(item, BaseModel):
            item = item.__dict__
        if isinstance(item, Mapping):
            if len(item) > 2_000_000 - nodes:
                _error("Saved state has too many mapping entries.")
            size += 256 + 96 * len(item)
            for key in item:
                if not isinstance(key, str):
                    _error("Saved state metadata requires string object keys.")
                size += 64 + 12 * len(key)
            stack.append((iter(item.values()), depth + 1))
        elif isinstance(item, (list, tuple)):
            if len(item) > 2_000_000 - nodes:
                _error("Saved state has too many array entries.")
            size += 64 + 16 * len(item)
            stack.append((iter(item), depth + 1))
        elif isinstance(item, str):
            # ASCII-escaped non-BMP characters need two six-byte surrogates.
            size += 64 + 12 * len(item)
        elif type(item) is int:
            bits = item.bit_length()
            size += 32 + (bits + 7) // 8 + 6 * (bits * 30103 // 100000 + 2)
        elif item is None or type(item) is bool:
            size += 32
        elif type(item) is float and math.isfinite(item):
            size += 32
        else:
            _error("Saved metadata requires finite JSON scalars and resident arrays/objects.")
    return size, depth_sum


def _encoded_json_admission(value, *, limit, operation):
    """Admit input/decoder buffers before UTF-8 encoding or either JSON parser."""
    if not isinstance(value, (str, bytes, bytearray)) or len(value) > limit:
        _error(f"{operation} exceeds its {limit:,}-byte encoded input envelope.")
    plan_workspace(operation, {"encoded input and dense decoded objects": 64 * len(value)})
    if isinstance(value, str):
        size = 0
        for character in value:
            code = ord(character)
            size += 1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4
            if size > limit:
                _error(f"{operation} exceeds its {limit:,}-byte UTF-8 input envelope.")
    else:
        size = len(value)
    plan_workspace(operation, {"encoded input and dense decoded objects": 64 * size})
    return size


def _state_copy_admission(value, *, operation, budget_bytes=None, extra_bytes=0):
    """Bound complete resident metadata and copy buffers before deep/Pydantic copies."""
    size, _ = _metadata_geometry(value)
    return plan_workspace(
        operation,
        {
            "complete resident metadata and copies": 8 * size,
            "additional table buffers": extra_bytes,
        },
        budget_bytes=budget_bytes,
    )


def _json_export_admission(value, *, indent=None, limit, operation, budget_bytes=None):
    """Bound complete output and format padding before native JSON serialization."""
    if indent is not None and (type(indent) is not int or indent < 0):
        _error("JSON indentation must be a nonnegative integer or None.")
    if indent is not None and indent > limit:
        _error("JSON indentation alone exceeds the complete encoded envelope.")
    size, depth_sum = _metadata_geometry(value)
    # Each container can contribute opening and closing indentation lines.
    encoded_bound = size + 2 * (indent or 0) * depth_sum
    if encoded_bound > limit:
        _error(f"Complete formatted JSON exceeds its {limit:,}-byte encoded envelope.")
    return plan_workspace(
        operation,
        {"metadata, output and serializer buffers": 4 * size + 4 * encoded_bound},
        budget_bytes=budget_bytes,
    )
