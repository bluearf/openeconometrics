"""Shipped editor/help contracts expose the bounded APIs without widening scope."""
import json
import importlib.util
from pathlib import Path

import openecon as oe
from openecon.econometrics.survival_ext import EXPORTS


def test_survival_scope_and_public_editor_contracts():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("survival_catalog_generator",root/"scripts/generate_editor_api.py")
    generator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    catalog={row["name"]:row for row in generator.decode_catalog(json.loads((root/"web/src/editor-api.json").read_text()))}
    contract=oe.capabilities()["survival_extensions"]
    assert len(contract["procedures"])==8
    assert set(contract["procedures"])|set(contract["saved_prediction"])==set(EXPORTS)
    assert contract["devices"]==["cpu"] and not contract["dataset_support"]
    assert not contract["stata_parity_validated"]
    assert contract["budgets"]["turnbull_estimated_work"]==500000000
    for name in EXPORTS:
        entry=catalog["openecon."+name]
        assert entry["returns"]=="openecon.TableSet"
        assert entry["description"]
        parameters={p["name"]:p for p in entry["parameters"]}
        assert parameters["device"]["default"]=="'cpu'"
        assert parameters["weights"]["default"]=="None"
    assert "group_2 minus group_1" in catalog["openecon.cif_compare"]["description"]
    assert "sampling covariance/CI" in catalog["openecon.turnbull"]["description"]
