"""Generated scope documents must fail closed on metadata/document drift."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "capability_docs", ROOT / "scripts/generate_capability_docs.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_committed_inventory_matches_registry_prediction_and_option_contracts():
    from openecon.econometrics import registry
    from openecon.prediction_capabilities import SAVED_ESTIMATORS
    saved = json.loads(generator.JSON_PATH.read_text())
    current = generator.snapshot(saved["source_ref"])
    assert saved == current
    from openecon.analysis_contracts import capabilities
    assert set(saved["estimators"]) == {item.name for item in registry.all_estimators()}
    assert set(saved["streaming"]["estimators"]) == set(saved["estimators"]) - set(capabilities()["eager_only_estimators"])
    prediction = saved["postestimation"]["saved_prediction"]
    assert prediction["estimator_count"] == len(SAVED_ESTIMATORS)
    specialized = saved['postestimation']['specialized_saved_targets']
    assert specialized['common_scalar_dispatch'] is False
    assert specialized['refits'] is False
    assert set(prediction["estimators"]).isdisjoint(prediction["remaining_estimators"])
    assert set(prediction["estimators"]) | set(prediction["remaining_estimators"]) == set(
        saved["estimators"]
    )
    assert saved["stata_parity_validated"] is False
    assert saved["estimators"]["xtreg"]["options"]["model"]["choices"]
    assert generator.MD_PATH.read_text() == generator.markdown(saved)


def test_check_detects_both_document_edit_and_changed_source_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "JSON_PATH", tmp_path / "inventory.json")
    monkeypatch.setattr(generator, "MD_PATH", tmp_path / "inventory.md")
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--source-ref", "test-pin"])
    assert generator.main() == 0
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--check"])
    assert generator.main() == 0
    generator.MD_PATH.write_text(generator.MD_PATH.read_text() + "\nmanual drift")
    assert generator.main() == 1
    saved = json.loads(generator.JSON_PATH.read_text())
    generator.MD_PATH.write_text(generator.markdown(saved))
    from openecon import analysis_contracts

    original = analysis_contracts.capabilities

    def changed():
        value = original()
        value["estimators"]["xtreg"]["options"]["model"]["choices"].append("future-option")
        return value

    monkeypatch.setattr(analysis_contracts, "capabilities", changed)
    assert generator.main() == 1
