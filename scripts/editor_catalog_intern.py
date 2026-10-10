"""Prospective lossless editor-wire envelope; stdlib metadata only.

This is an off-tree source proposal, not an accepted browser/runtime codec.
It wraps the complete existing compact wire, whose token tables stay unchanged.
"""
from __future__ import annotations

from collections import Counter
import json
import math

SCHEMA = "openecon.editor-api.interned.v1"
MAX_WIRE_BYTES = 500_000
MAX_DECODED_BYTES = 8 * 1024 * 1024
MAX_ITEMS = 500_000
MAX_DEPTH = 32
MAX_STRINGS = 8192
MAX_ENTRIES = 10_000
MAX_STRING_BYTES = 1024 * 1024
MAX_SAFE_INTEGER = 2**53 - 1


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _utf8(text):
    if type(text) is not str:
        raise ValueError("Expected a string")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("Unpaired Unicode surrogate") from error
    if len(raw) > MAX_STRING_BYTES:
        raise ValueError("String byte budget exceeded")
    return raw


def _raw_admit(raw):
    if type(raw) is bytes:
        if len(raw) > MAX_WIRE_BYTES:
            raise ValueError("Wire byte budget exceeded")
        try:
            raw = raw.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("Wire must be UTF-8") from error
    if type(raw) is not str or len(raw.encode("utf-8", errors="strict")) > MAX_WIRE_BYTES:
        raise ValueError("Wire byte budget exceeded")
    # Bound JSON nesting before the JSON decoder allocates its object tree.
    depth, quoted, escaped = 0, False, False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise ValueError("Wire depth budget exceeded")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Invalid JSON nesting")
    if depth or quoted:
        raise ValueError("Incomplete JSON")
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _parse(raw):
    return json.loads(_raw_admit(raw), object_pairs_hook=_unique_object,
                      parse_int=lambda value: int(value) if value != "-0" else (_ for _ in ()).throw(ValueError("Negative-zero JSON number")),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def _walk_admit(value, *, strings=None, depth=0, state=None):
    if state is None:
        state = [0, 0]
    if depth > MAX_DEPTH:
        raise ValueError("Decoded depth budget exceeded")
    state[0] += 1
    if state[0] > MAX_ITEMS:
        raise ValueError("Decoded item budget exceeded")
    if type(value) is str:
        state[1] += len(_utf8(value)) * 6 + 2  # conservative JSON escaping reserve
        if strings is not None:
            strings[value] += 1
    elif value is None or type(value) is bool:
        state[1] += 5
    elif type(value) is int:
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("Unsafe JSON integer")
        state[1] += 32
    elif type(value) is float:
        if (not math.isfinite(value) or abs(value) > MAX_SAFE_INTEGER
                or value == 0 and math.copysign(1, value) < 0):
            raise ValueError("Nonfinite or negative-zero JSON number")
        state[1] += 32
    elif type(value) is list:
        state[1] += len(value) + 2
        for child in value:
            _walk_admit(child, strings=strings, depth=depth+1, state=state)
    elif type(value) is dict:
        state[1] += 2 * len(value) + 2
        for key, child in value.items():
            state[1] += len(_utf8(key)) * 6 + 2
            _walk_admit(child, strings=strings, depth=depth+1, state=state)
    else:
        raise ValueError("Only bounded JSON metadata is supported")
    if state[1] > MAX_DECODED_BYTES:
        raise ValueError("Decoded byte reserve exceeded")
    return state


def _dictionary(counts):
    candidates = []
    for value, count in counts.items():
        length = len(_json(value).encode("utf-8"))
        saving = count * (length - 12) - length - 1
        if count >= 2 and length >= 24 and saving > 0:
            candidates.append((saving, value))
    # Fixed tie-breaking, then UTF-8 order: independent of object traversal order.
    candidates.sort(key=lambda pair: (-pair[0], _utf8(pair[1])))
    return sorted((value for _, value in candidates[:MAX_STRINGS]), key=_utf8)


def _envelope(entries):
    counts = Counter()
    _walk_admit(entries, strings=counts)
    strings = _dictionary(counts)
    tokens = {value: index for index, value in enumerate(strings)}

    def pack(value):
        if type(value) is str and value in tokens:
            return {"$s": tokens[value]}
        if type(value) is list:
            return [pack(child) for child in value]
        if type(value) is dict:
            # Reserved-key objects are escaped rather than losing unknown fields.
            if any(key.startswith("$") for key in value):
                return {"$o": [[key, pack(child)] for key, child in sorted(value.items(), key=lambda pair: _utf8(pair[0]))]}
            return {key: pack(child) for key, child in value.items()}
        return value

    return {"schema": SCHEMA, "strings": strings, "entries": pack(entries)}


def encode_wire(entries):
    """Intern complete repeated values deterministically; retain every field."""
    if type(entries) is not list or len(entries) > MAX_ENTRIES:
        raise ValueError("Expected bounded catalogue entries")
    raw = (_json(_envelope(entries)) + "\n").encode("utf-8")
    _raw_admit(raw)
    return raw


def decode_wire(raw):
    """Admit old full/compact arrays and the exact new envelope, before copying."""
    value = _parse(raw)
    if type(value) is list:
        if len(value) > MAX_ENTRIES:
            raise ValueError("Entry count exceeded")
        _walk_admit(value)
        return value
    if type(value) is not dict or set(value) != {"schema", "strings", "entries"} or value["schema"] != SCHEMA:
        raise ValueError("Unknown or mixed editor wire schema")
    strings, entries = value["strings"], value["entries"]
    if (type(strings) is not list or len(strings) > MAX_STRINGS
            or type(entries) is not list or len(entries) > MAX_ENTRIES):
        raise ValueError("Envelope geometry exceeded")
    for text in strings:
        _utf8(text)
    if strings != sorted(set(strings), key=_utf8):
        raise ValueError("String dictionary must be unique and UTF-8 sorted")
    _walk_admit(value)
    uses = [0] * len(strings)
    state = [0, 0]

    def unpack(node, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("Decoded depth budget exceeded")
        if type(node) is dict and "$s" in node:
            token = node.get("$s")
            if (set(node) != {"$s"} or type(token) not in (int, float)
                    or not math.isfinite(token) or token != int(token)
                    or not 0 <= token < len(strings)):
                raise ValueError("Invalid string token")
            uses[int(token)] += 1
            result = strings[int(token)]
        elif type(node) is dict and "$o" in node:
            if set(node) != {"$o"} or type(node["$o"]) is not list:
                raise ValueError("Invalid escaped object")
            result = {}
            for pair in node["$o"]:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                    raise ValueError("Invalid escaped object pair")
                result[pair[0]] = unpack(pair[1], depth+1)
            if not any(key.startswith("$") for key in result):
                raise ValueError("Unnecessary escaped object")
        elif type(node) is dict:
            if any(key.startswith("$") for key in node):
                raise ValueError("Unknown reserved string-wire key")
            result = {key: unpack(child, depth+1) for key, child in node.items()}
        elif type(node) is list:
            result = [unpack(child, depth+1) for child in node]
        else:
            result = node
        # Charge each reconstructed scalar/container before exposing the tree.
        # Containers' children are already charged; avoid quadratic traversal.
        if type(result) in (list, dict):
            state[0] += 1
            state[1] += 2 + (len(result) if type(result) is list else sum(2 + len(_utf8(key))*6 + 2 for key in result))
        else:
            _walk_admit(result, state=state)
        if state[0] > MAX_ITEMS or state[1] > MAX_DECODED_BYTES:
            raise ValueError("Decoded metadata budget exceeded")
        return result

    result = unpack(entries)
    if any(count < 2 for count in uses):
        raise ValueError("Unused or singleton interned value")
    if value != _envelope(result):
        raise ValueError("Noncanonical string interning")
    return result


def admit_logical(entries):
    """Independent full logical limit after all legacy parameter-token expansion.

    The unchanged 500000-byte bound applies to encoded output; this separate
    8-MiB conservative decoded JSON reserve and 500000-node cap are not RSS.
    """
    if type(entries) is not list or len(entries) > MAX_ENTRIES:
        raise ValueError("Expected bounded logical catalogue entries")
    return _walk_admit(entries)


def decode_parsed_wire(value):
    """Bound and revalidate an already parsed trusted legacy/envelope value.

    This cannot recover duplicate raw keys or original number lexemes erased by
    an earlier JSON parser. Raw saved-wire validation must use decode_wire.
    No legacy token expansion/guard is changed by this outer transport adapter.
    """
    _walk_admit(value)
    return decode_wire((_json(value) + "\n").encode("utf-8"))
