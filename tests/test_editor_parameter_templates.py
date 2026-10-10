"""Fixed complete parameter-list tokens preserve help, fallback and legacy data."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


generator = module("template_generator", "scripts/generate_editor_api.py")
independent = module("template_reader", "scripts/verify_editor_api_preserved.py")


def entry(parameters):
    return {"name": "openecon.saved.restore", "signature": "restore(payload)",
            "kind": "method", "owner": "openecon.saved", "description": "Complete replay.",
            "parameters": parameters, "returns": "openecon.saved"}


@pytest.mark.parametrize("index", range(len(generator.STORED_PARAMETER_LISTS)))
def test_parameter_templates_roundtrip_every_field_with_independent_reader(index):
    logical = [entry(deepcopy(generator.STORED_PARAMETER_LISTS[index]))]
    original = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert logical == original
    assert stored[0]["P"] == index and "p" not in stored[0]
    assert generator.decode_catalog(stored) == original
    assert list(independent.logical(json.dumps(stored)).values()) == original
    decoded = generator.decode_catalog(stored)
    decoded[0]["parameters"][0]["description"] = "Changed locally."
    assert generator.decode_catalog(stored) == original


@pytest.mark.parametrize("field,value", [("description", "Exact note"), ("annotation", "Any"),
                                         ("default", "None "), ("choices", ["None"]),
                                         ("kind", "positional-only")])
def test_near_template_keeps_complete_literal_metadata(field, value):
    parameters = deepcopy(generator.STORED_PARAMETER_LISTS[0])
    parameters[0][field] = value
    logical = [entry(parameters)]
    stored = generator.encode_catalog(logical)
    assert "P" not in stored[0]
    assert generator.decode_catalog(stored) == logical
    assert list(independent.logical(json.dumps(stored)).values()) == logical


@pytest.mark.parametrize("token", [None, True, False, -1, len(generator.STORED_PARAMETER_LISTS), 1.0, -0.0, "0", [], {}])
def test_unknown_or_noninteger_parameter_tokens_fail(token):
    stored = [{"n": "~saved.restore", "S": "(payload)", "d": "Replay", "P": token}]
    with pytest.raises(ValueError, match="parameters"):
        generator.decode_catalog(stored)
    with pytest.raises(ValueError, match="parameters"):
        independent.logical_parsed(stored)
    # The raw route preserves strict JSON admission before token semantics.
    raw_message = "Nonfinite or negative-zero JSON number" if isinstance(token, float) and token == 0 else "parameters"
    with pytest.raises(ValueError, match=raw_message):
        independent.logical(json.dumps(stored))


@pytest.mark.parametrize("field", ["parameters", "p"])
def test_explicit_parameter_metadata_has_legacy_precedence(field):
    original = entry([{"name": "payload", "kind": "positional-only", "default": "'literal'"}])
    stored = generator.encode_catalog([original])
    stored[0].pop("p")
    stored[0][field] = deepcopy(original["parameters"])
    stored[0]["P"] = True
    assert generator.decode_catalog(stored) == [original]
    assert list(independent.logical(json.dumps(stored)).values()) == [original]


def test_canonical_reserved_template_marker_is_refused():
    original = entry([])
    original["P"] = 0
    with pytest.raises(ValueError, match="reserved"):
        generator.encode_catalog([original])
