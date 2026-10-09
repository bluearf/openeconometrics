"""Discoverability and explicit specialized rank-choice boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import openecon as oe


def test_rank_choice_public_metadata_and_saved_helpers_are_complete():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rank_choice_editor", root / "scripts/generate_editor_api.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    catalog = {row["name"]: row for row in generator.decode_catalog(json.loads((root / "web/src/editor-api.json").read_text()))}
    cap = oe.capabilities()["rank_ordered_choice"]
    assert cap["covariances"] == ["oim", "hc0", "cr0"]
    assert cap["devices"] == ["cpu"] and cap["precision"] == "float64"
    assert not cap["dataset_support"] and not cap["stata_parity_validated"] and cap["weights"] == []
    assert cap["budgets"]["joint_targets"] == 256
    for name in cap["procedures"]:
        assert callable(getattr(oe, name))
        entry = catalog["openecon." + name]
        assert entry["returns"] == "openecon.TableSet"
    parameters = {p["name"]: p for p in catalog["openecon.rologit"]["parameters"]}
    assert {"rank", "case", "alternative", "available", "vce", "cluster"} <= set(parameters)
    assert parameters["vce"]["default"] == "'oim'"
    assert "complete case/cluster" in catalog["openecon.rologit"]["description"]
    assert "without refitting" in catalog["openecon.rologit_restore"]["description"]
    assert "eligible pair support" in catalog["openecon.rologit_margins"]["description"]


def test_rank_capabilities_discovery_does_not_load_tensor_runtime():
    script = "import sys, openecon as oe; assert 'rank_ordered_choice' in oe.capabilities(); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
