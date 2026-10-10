"""Future metadata-only tests; source review never collects or executes them."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/"tests/fixtures/editor-catalog-intern-v2-pair-order.json"


def codec():
    spec=importlib.util.spec_from_file_location("metadata_wire_v2",ROOT/"scripts/editor_catalog_intern.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_escaped_pair_order_and_whole_unknown_field_roundtrips():
    wire=codec()
    cases=json.loads(FIXTURES.read_bytes())
    for case in cases["positive"]:
        assert wire.decode_wire(case["raw"])==case["logical"]
        assert wire.encode_wire(case["logical"]).decode()==case["raw"]
        assert wire.decode_wire(wire.encode_wire(case["logical"]))==case["logical"]
    for case in cases["negative"]:
        with pytest.raises(ValueError):
            wire.decode_wire(case["raw"])


def test_all_original_v1_negative_wires_remain_refused_without_waivers():
    wire=codec()
    source=ROOT/"tests/fixtures/editor-catalog-intern-original-v1-negative-fixtures.json"
    for case in json.loads(source.read_bytes()):
        with pytest.raises(ValueError):
            wire.decode_wire(case["raw"])


def test_old_arrays_and_current_whole_catalog_bytes_are_independently_preserved():
    # Root must bind these exact frozen inputs before execution. No fabricated
    # successful browser/whole-future-union receipt is supplied by source fixtures.
    root=ROOT/"tests/fixtures/editor-catalog-intern-whole-source-bindings.json"
    import hashlib
    for record in json.loads(root.read_bytes())["files"]:
        body=(ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest()==record["sha256"]
    wire=codec()
    for record in json.loads(root.read_bytes())["legacy_and_interned_pairs"]:
        old_raw=(ROOT / record["legacy"]).read_bytes()
        if len(old_raw)>wire.MAX_WIRE_BYTES:
            # An over-budget original compact reference remains inadmissible.
            # Its immutable whole parsed source is an expected oracle, not a
            # accepted raw-wire path or a waiver of the old capacity failure.
            with pytest.raises(ValueError):
                wire.decode_wire(old_raw)
            old=json.loads(old_raw)
            wire.admit_logical(old)
        else:
            old=wire.decode_wire(old_raw)
        new=wire.decode_wire((ROOT / record["interned"]).read_bytes())
        assert json.dumps(old,sort_keys=True,ensure_ascii=False)==json.dumps(new,sort_keys=True,ensure_ascii=False)
