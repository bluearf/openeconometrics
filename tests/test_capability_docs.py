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
    assert set(saved["streaming"]["estimators"]) == set(saved["estimators"]) - set(
        capabilities()["eager_only_estimators"]
    )
    prediction = saved["postestimation"]["saved_prediction"]
    assert prediction["estimator_count"] == len(SAVED_ESTIMATORS)
    specialized = saved["postestimation"]["specialized_saved_targets"]
    assert specialized["common_scalar_dispatch"] is False
    assert specialized["refits"] is False
    assert set(prediction["estimators"]).isdisjoint(prediction["remaining_estimators"])
    assert set(prediction["estimators"]) | set(prediction["remaining_estimators"]) == set(
        saved["estimators"]
    )
    assert saved["stata_parity_validated"] is False
    assert saved["estimators"]["xtreg"]["options"]["model"]["choices"]
    assert generator.MD_PATH.read_text() == generator.markdown(saved)


def test_check_detects_both_document_edit_and_changed_source_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "SCOPE_DOCUMENTS", ())
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


def test_entry_point_scope_blocks_match_source_and_reject_universal_route_claims():
    saved = json.loads(generator.JSON_PATH.read_text())
    for name in generator.SCOPE_DOCUMENTS:
        text = (ROOT / name).read_text()
        assert generator.scope_document(text, saved, name) == text
        assert not generator.unsupported_scope_claims(text), name
    assert generator.unsupported_scope_claims(
        "All registered estimator names have replayable CSV/Parquet Dataset fitting routes."
    )
    assert generator.unsupported_scope_claims(
        "All 80 currently registered model names have native Dataset routes."
    )
    # Preserve an actual historical installed count, without promoting its scope.
    assert not generator.unsupported_scope_claims(
        "The 0.3.27 runtime contained 230 modules and 80 registered estimators."
    )


def test_new_eager_only_estimator_and_removed_adapter_change_document_scope(tmp_path, monkeypatch):
    from dataclasses import replace
    from openecon.econometrics import registry
    from openecon import prediction_capabilities

    monkeypatch.setattr(generator, "JSON_PATH", tmp_path / "inventory.json")
    monkeypatch.setattr(generator, "MD_PATH", tmp_path / "inventory.md")
    guide = tmp_path / "guide.md"
    guide.write_text("# Scope\n\n" + generator.SCOPE_BEGIN + "\n" + generator.SCOPE_END + "\n")
    monkeypatch.setattr(generator, "SCOPE_DOCUMENTS", (str(guide),))
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--source-ref", "test-pin"])
    assert generator.main() == 0
    initial = json.loads(generator.JSON_PATH.read_text())
    original_guide = guide.read_text()
    guide.write_text(original_guide + "\nAll registered estimators support Dataset inputs.\n")
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--check"])
    assert generator.main() == 1
    guide.write_text(original_guide)
    assert generator.main() == 0
    original = registry.all_estimators
    extra = replace(original()[-1], name="future_eager_only", legacy=False)
    monkeypatch.setattr(registry, "all_estimators", lambda: (*original(), extra))
    # A genuinely new registry entry has no Dataset or common-prediction adapter.
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--check"])
    assert generator.main() == 1
    current = generator.snapshot("test-pin")
    assert len(current["estimators"]) == len(initial["estimators"]) + 1
    assert current["streaming"]["estimators"] == initial["streaming"]["estimators"]
    assert (
        current["evidence_scope"]["per_estimator"]["future_eager_only"]["source_dataset_fit"]
        is False
    )
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--source-ref", "test-pin"])
    assert generator.main() == 0
    assert str(len(current["estimators"])) in guide.read_text()
    monkeypatch.setattr(
        prediction_capabilities,
        "SAVED_ESTIMATORS",
        prediction_capabilities.SAVED_ESTIMATORS - {"ols"},
    )
    monkeypatch.setattr(generator.sys, "argv", ["generate", "--check"])
    assert generator.main() == 1
    removed = generator.snapshot("test-pin")
    assert (
        removed["postestimation"]["saved_prediction"]["estimator_count"]
        == initial["postestimation"]["saved_prediction"]["estimator_count"] - 1
    )
    assert removed["evidence_scope"]["per_estimator"]["ols"]["source_common_prediction"] is False
