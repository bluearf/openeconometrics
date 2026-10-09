"""Catalog preservation compares compact legacy and current logical metadata."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_editor_api_preserved.py"
SPEC = importlib.util.spec_from_file_location("editor_api_preservation", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def catalog(parameter):
    return [{
        "n": "~Example.method", "kind": "m", "s": "method(value)",
        "d": "Retain complete help.", "R": "TableSet",
        "p": [{"n": "value", "a": "str", "v": "''", "d": "",
               "c": ["'Türkçe'", "'normal'"], **parameter}],
    }]


def test_pinned_parameter_layout_preserves_colliding_branch_tokens_without_mutation():
    parameters = [{
        "name": "archived_value", "kind": "keyword-only", "default": "'literal'",
        "annotation": "str", "description": "Complete archived Türkçe help.",
        "choices": ["'literal'", "'other'"],
    }]
    layout = [[] for _ in range(18)] + [deepcopy(parameters)]
    before = deepcopy(layout)
    wire = catalog({})
    wire[0].pop("p")
    wire[0]["P"] = 18
    expected = deepcopy(wire)
    expected[0].pop("P")
    expected[0]["parameters"] = deepcopy(parameters)
    baseline = verifier.logical(json.dumps(wire), parameter_lists=layout)
    assert baseline == verifier.logical(json.dumps(expected))
    assert baseline != verifier.logical(json.dumps(wire))
    baseline["openecon.Example.method"]["parameters"][0]["choices"].append("'edited'")
    assert layout == before
    with pytest.raises(ValueError, match="recognized integer token"):
        verifier.logical(json.dumps(wire), parameter_lists=())


def test_pinned_parameter_layout_is_read_from_literal_source_without_executing_it(monkeypatch):
    expected = ([{"name": "original", "kind": "positional-only", "default": "None"}],)
    source = "raise AssertionError('historical code must not execute')\nSTORED_PARAMETER_LISTS = " + repr(expected)
    calls = []

    def committed(command, **options):
        calls.append(command)
        return source.encode()

    monkeypatch.setattr(verifier.subprocess, "check_output", committed)
    assert verifier.parameter_lists_at("immutable-pin") == expected
    assert calls == [["git", "show", "immutable-pin:scripts/generate_editor_api.py"]]


def test_pinned_signature_layout_preserves_colliding_branch_tokens_without_mutation():
    suffix = "(archived: str, *, choice='Türkçe')"
    layout = ("()",) * 6 + (suffix,)
    wire = catalog({})
    wire[0].pop("s")
    wire[0]["I"] = 6
    original = deepcopy(wire)
    expected = deepcopy(wire)
    expected[0].pop("I")
    expected[0]["signature"] = "method" + suffix
    baseline = verifier.logical(json.dumps(wire), signature_suffixes=layout)
    assert baseline == verifier.logical(json.dumps(expected))
    assert baseline != verifier.logical(json.dumps(wire))
    baseline["openecon.Example.method"]["signature"] = "Changed locally."
    assert wire == original and layout[-1] == suffix
    assert verifier.logical(json.dumps(wire), signature_suffixes=layout) == verifier.logical(json.dumps(expected))
    with pytest.raises(ValueError, match="recognized integer token"):
        verifier.logical(json.dumps(wire), signature_suffixes=())


def test_pinned_signature_layout_is_read_from_literal_source_without_executing_it(monkeypatch):
    expected = ("(value: Any, *, option=None)",)
    source = "raise AssertionError('historical code must not execute')\nSTORED_SIGNATURE_SUFFIXES = " + repr(expected)
    calls = []

    def committed(command, **options):
        calls.append(command)
        return source.encode()

    monkeypatch.setattr(verifier.subprocess, "check_output", committed)
    assert verifier.signature_suffixes_at("immutable-pin") == expected
    assert calls == [["git", "show", "immutable-pin:scripts/generate_editor_api.py"]]


def test_pinned_parameter_names_preserve_colliding_branch_tokens_without_mutation():
    layout = ("unused",) * 18 + ("archived_name",)
    wire = catalog({"n": 18})
    original = deepcopy(wire)
    expected = catalog({"name": "archived_name"})
    baseline = verifier.logical(json.dumps(wire), parameter_names=layout)
    assert baseline == verifier.logical(json.dumps(expected))
    assert baseline != verifier.logical(json.dumps(wire))
    baseline["openecon.Example.method"]["parameters"][0]["choices"].append("'local'")
    assert wire == original and layout[-1] == "archived_name"
    with pytest.raises(ValueError, match="valid integer index"):
        verifier.logical(json.dumps(wire), parameter_names=())


def test_pinned_parameter_names_are_read_without_executing_historical_code(monkeypatch):
    expected = ("archived_name", "other")
    source = "raise AssertionError('historical code must not execute')\nSTORED_PARAMETER_NAMES = " + repr(expected)
    calls = []

    def committed(command, **options):
        calls.append(command)
        return source.encode()

    monkeypatch.setattr(verifier.subprocess, "check_output", committed)
    assert verifier.parameter_names_at("immutable-pin") == expected
    assert calls == [["git", "show", "immutable-pin:scripts/generate_editor_api.py"]]


@pytest.mark.parametrize("token,kind", [
    ("p", "positional-only"), ("k", "keyword-only"),
    ("*", "var-positional"), ("**", "var-keyword"),
])
def test_preservation_matches_legacy_and_current_kinds_with_complete_metadata(token, kind):
    legacy, current = catalog({"t": token}), catalog({"k": token})
    before = deepcopy(legacy)
    decoded = verifier.logical(json.dumps(legacy, ensure_ascii=False))
    assert decoded == verifier.logical(json.dumps(current, ensure_ascii=False))
    entry = decoded["openecon.Example.method"]
    assert entry["owner"] == "openecon.Example"
    assert entry["returns"] == "openecon.TableSet"
    assert entry["parameters"] == [{
        "name": "value", "annotation": "str", "default": "''",
        "description": "", "choices": ["'Türkçe'", "'normal'"], "kind": kind,
    }]
    assert legacy == before
    changed = deepcopy(current)
    changed[0]["p"][0]["v"] = "'changed'"
    assert decoded != verifier.logical(json.dumps(changed))


@pytest.mark.parametrize("fields,kind", [
    ({"kind": "positional-only", "t": "k", "k": "**"}, "positional-only"),
    ({"t": "k", "k": "**"}, "var-keyword"),
    ({"t": "**"}, "var-keyword"),
])
def test_parameter_kind_precedence_full_then_canonical_then_legacy(fields, kind):
    decoded = verifier.logical(json.dumps(catalog(fields)))
    parameter = decoded["openecon.Example.method"]["parameters"][0]
    assert parameter["kind"] == kind
    assert "t" not in parameter and "k" not in parameter


@pytest.mark.parametrize("fields,expected", [
    ({"N": 1}, "None"),
    ({"v": "", "N": 1}, ""),
    ({"v": "0", "N": 1}, "0"),
    ({"default": "False", "v": "0", "N": 1}, "False"),
    ({"default": "", "v": "None", "N": 1}, ""),
])
def test_none_default_alias_preserves_complete_metadata_and_default_precedence(fields, expected):
    stored = catalog({})
    parameter = stored[0]["p"][0]
    parameter.pop("v")
    parameter.update(fields)
    before = deepcopy(stored)
    decoded = verifier.logical(json.dumps(stored))
    canonical = deepcopy(stored)
    canonical[0]["p"][0] = {key: value for key, value in parameter.items()
                            if key not in ("default", "v", "N")}
    canonical[0]["p"][0]["default"] = expected
    assert decoded == verifier.logical(json.dumps(canonical))
    assert decoded["openecon.Example.method"]["parameters"][0]["default"] == expected
    assert "N" not in decoded["openecon.Example.method"]["parameters"][0]
    assert stored == before
