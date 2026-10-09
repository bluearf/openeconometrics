"""Public metadata retains the eight bounded model domains and their targets."""
import importlib.util
import json
from pathlib import Path

import openecon as oe


def test_binary_model_domains_and_saved_target_help_are_complete():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("binary_mediation_catalog", root / "scripts/generate_editor_api.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    catalog = {entry["name"]:entry for entry in generator.decode_catalog(json.loads((root / "web/src/editor-api.json").read_text()))}
    cap = oe.capabilities()["binary_mediation"]
    assert cap["mediator_links"] == ["logit", "probit"]
    assert cap["outcome_models"] == ["gaussian", "logit", "probit", "poisson"]
    assert cap["model_domains"] == len(cap["mediator_links"]) * len(cap["outcome_models"]) == 8
    assert cap["covariances"] == ["OIM", "HC0"]
    assert cap["devices"] == ["cpu"] and not cap["dataset_support"]
    assert not cap["stata_parity_validated"]
    fit = catalog["openecon.mediation_binary"]
    saved = catalog["openecon.mediation_binary_restore"]
    assert fit["returns"] == saved["returns"] == "openecon.TableSet"
    assert "fixed retained controls" in fit["description"]
    assert "without refitting" in saved["description"]
    params = {p["name"]:p for p in fit["parameters"]}
    assert {"mediator_link", "outcome_model", "covariance", "interaction", "assumptions"} <= set(params)
    assert params["mediator_link"]["default"] == "'logit'"
    assert params["outcome_model"]["default"] == "'gaussian'"
    assert params["covariance"]["default"] == "'HC0'"
    assert params["mediator_link"]["choices"] == ["'logit'", "'probit'"]
    assert params["outcome_model"]["choices"] == ["'gaussian'", "'logit'", "'probit'", "'poisson'"]
    assert params["covariance"]["choices"] == ["'OIM'", "'HC0'"]
    assert callable(oe.mediation_binary) and callable(oe.mediation_binary_restore)
