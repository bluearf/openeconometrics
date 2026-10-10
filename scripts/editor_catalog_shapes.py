"""Source-only full-object shape wire. Old array/intern-v1 routes are exact.

Arrays of shape-index plus values represent objects; {$a: [...]} represents an
original array. Every original key (including reserved/prototype-looking keys)
is carried in the deterministic UTF-8 shape dictionary. No science is imported.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('_editor_wire_preserved_intern_v1',
                                              Path(__file__).with_name('editor_catalog_intern.py'))
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
SCHEMA = 'openecon.editor-api.shaped-arrays.v1'
MAX_SHAPES = 8192
MAX_WIRE_BYTES = _legacy.MAX_WIRE_BYTES
MAX_DECODED_BYTES = _legacy.MAX_DECODED_BYTES
MAX_ITEMS = _legacy.MAX_ITEMS
MAX_DEPTH = _legacy.MAX_DEPTH
MAX_ENTRIES = _legacy.MAX_ENTRIES
MAX_STRINGS = _legacy.MAX_STRINGS


def _shape_order(fields):
    return tuple(_legacy._utf8(field) for field in fields)


def _envelope(entries):
    if type(entries) is not list or len(entries) > MAX_ENTRIES:
        raise ValueError('Expected bounded complete catalogue entries')
    counts = Counter()
    _legacy._walk_admit(entries, strings=counts)
    strings = _legacy._dictionary(counts)
    string_tokens = {text: index for index, text in enumerate(strings)}
    shapes = set()

    def collect(node):
        if type(node) is list:
            for child in node:
                collect(child)
        elif type(node) is dict:
            shapes.add(tuple(sorted(node, key=_legacy._utf8)))
            if len(shapes) > MAX_SHAPES:
                raise ValueError('Object shape count exceeded')
            for child in node.values():
                collect(child)

    collect(entries)
    shapes = sorted(shapes, key=_shape_order)
    shape_tokens = {fields: index for index, fields in enumerate(shapes)}

    def pack(node):
        if type(node) is str and node in string_tokens:
            return {'$s': string_tokens[node]}
        if type(node) is list:
            return {'$a': [pack(child) for child in node]}
        if type(node) is dict:
            fields = tuple(sorted(node, key=_legacy._utf8))
            return [shape_tokens[fields], *[pack(node[field]) for field in fields]]
        return node

    return {'schema': SCHEMA, 'strings': strings, 'shapes': [list(fields) for fields in shapes],
            'entries': [pack(entry) for entry in entries]}


def encode_wire(entries):
    raw = (_legacy._json(_envelope(entries)) + '\n').encode('utf-8')
    _legacy._raw_admit(raw)
    return raw


def decode_wire(raw):
    """Decode exact old routes or complete shaped objects under original caps."""
    envelope = _legacy._parse(raw)
    if type(envelope) is not dict or envelope.get('schema') != SCHEMA:
        return _legacy.decode_wire(raw)
    if set(envelope) != {'schema', 'strings', 'shapes', 'entries'}:
        raise ValueError('Unknown or mixed shape-wire fields')
    strings, shapes, entries = envelope['strings'], envelope['shapes'], envelope['entries']
    if (type(strings) is not list or len(strings) > MAX_STRINGS
            or type(shapes) is not list or len(shapes) > MAX_SHAPES
            or type(entries) is not list or len(entries) > MAX_ENTRIES):
        raise ValueError('Shape-wire geometry exceeded')
    _legacy._walk_admit(envelope)
    for text in strings:
        _legacy._utf8(text)
    if strings != sorted(set(strings), key=_legacy._utf8):
        raise ValueError('String dictionary must be unique and UTF-8 sorted')
    typed_shapes = []
    for fields in shapes:
        if type(fields) is not list:
            raise ValueError('Object shape must be a complete key array')
        for field in fields:
            _legacy._utf8(field)
        if fields != sorted(set(fields), key=_legacy._utf8):
            raise ValueError('Object shape keys must be unique and UTF-8 sorted')
        typed_shapes.append(tuple(fields))
    if typed_shapes != sorted(set(typed_shapes), key=_shape_order):
        raise ValueError('Object shape dictionary must be unique and UTF-8 sorted')
    string_uses, shape_uses = [0] * len(strings), [0] * len(shapes)
    state = [0, 0]

    def index(token, limit):
        # Exact int includes JSON lexical 0 but rejects 0.0, 0e0, booleans and
        # negative zero. The new TS raw parser checks the same token lexemes.
        if type(token) is not int or not 0 <= token < limit:
            raise ValueError('Invalid canonical shape/string index')
        return token

    def unpack(node, depth):
        if depth > MAX_DEPTH:
            raise ValueError('Decoded depth budget exceeded')
        if type(node) is list:
            if not node:
                raise ValueError('Object shape token is missing')
            token = index(node[0], len(shapes))
            fields = shapes[token]
            if len(node) != len(fields) + 1:
                raise ValueError('Complete object shape/value count disagrees')
            shape_uses[token] += 1
            result = {field: unpack(child, depth + 1) for field, child in zip(fields, node[1:], strict=True)}
        elif type(node) is dict:
            if set(node) == {'$s'}:
                token = index(node['$s'], len(strings))
                string_uses[token] += 1
                result = strings[token]
            elif set(node) == {'$a'} and type(node['$a']) is list:
                result = [unpack(child, depth + 1) for child in node['$a']]
            else:
                raise ValueError('Unknown, mixed or unescaped shape-wire object')
        else:
            result = node
        if type(result) in (list, dict):
            state[0] += 1
            state[1] += 2 + (len(result) if type(result) is list
                            else sum(4 + 6 * len(_legacy._utf8(field)) for field in result))
        else:
            _legacy._walk_admit(result, state=state)
        if state[0] > MAX_ITEMS or state[1] > MAX_DECODED_BYTES:
            raise ValueError('Decoded metadata budget exceeded')
        return result

    # Charge the original complete catalogue array before returning any tree.
    state[0], state[1] = 1, 2 + len(entries)
    result = [unpack(entry, 1) for entry in entries]
    if any(count < 2 for count in string_uses) or any(count < 1 for count in shape_uses):
        raise ValueError('Unused/singleton string or unused object shape')
    if envelope != _envelope(result):
        raise ValueError('Noncanonical complete shape/string wire')
    return result


def admit_logical(entries):
    return _legacy.admit_logical(entries)


def decode_parsed_wire(value):
    """Trusted already-parsed adapter; cannot recover erased raw lexical facts.

    Browser activation uses a generated exact raw-string companion, never this
    adapter, so duplicate raw keys and index number spelling remain checkable.
    The unchanged inherited legacy float-token rules are not relaxed.
    """
    _legacy._walk_admit(value)
    return decode_wire((_legacy._json(value) + '\n').encode('utf-8'))


def raw_typescript_module(raw):
    """Synchronous exact raw wire module for Node and browser strict admission."""
    text = _legacy._raw_admit(raw)
    # JSON quoting is also a valid ES2019 string literal; no string interpolation
    # or execution of metadata occurs. Both files must be checked by generator.
    return ('// Generated complete raw editor wire. Do not edit.\n'
            'const wire: string = ' + _legacy._json(text) + ';\nexport default wire;\n')
