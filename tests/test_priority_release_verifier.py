"""Release lifecycle evidence distinguishes saved state from API presentation."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "priority_release_verifier", ROOT / "scripts/verify_priority_release.py"
)
verifier = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec.loader.exec_module(verifier)
finally:
    sys.path.pop(0)


def record(*, events=True):
    value = {
        "id": "owned-run",
        "code": "display('synthetic')",
        "stdout": "",
        "outputs": [{"type": "text", "data": "synthetic", "latex": "old presentation"}],
    }
    if events:
        value["events"] = [{"type": "output", "index": 0}]
    return value


def snapshot(value):
    return {
        "raw_history_sha256": verifier.payload_hash(value),
        "raw_history": [deepcopy(value)],
        "api_history": [deepcopy(value)],
    }


def test_interrupted_optional_timeline_absence_is_preserved():
    interrupted = record(events=False)
    interrupted.update(status="interrupted", outputs=[])
    assert "events" not in verifier.retained(interrupted)
    assert verifier.retained(interrupted) != verifier.retained({**interrupted, "events": []})
    assert verifier.compare_history(snapshot(interrupted), snapshot(interrupted)) == []


def test_missing_nonempty_api_events_cannot_be_normalized_away():
    before = snapshot(record())
    after = deepcopy(before)
    del after["api_history"][0]["events"]
    with pytest.raises(AssertionError, match="Nonempty API events"):
        verifier.compare_history(before, after)


def test_raw_event_loss_fails_even_if_api_and_supplied_digest_look_unchanged():
    before = snapshot(record())
    after = deepcopy(before)
    del after["raw_history"][0]["events"]
    with pytest.raises(AssertionError, match="Raw saved history changed"):
        verifier.compare_history(before, after)


def test_raw_byte_change_is_not_reduced_to_projected_fields():
    before = snapshot(record())
    after = deepcopy(before)
    after["raw_history_sha256"] = "different bytes"
    with pytest.raises(AssertionError, match="Raw saved history bytes"):
        verifier.compare_history(before, after)


def test_legitimate_latex_projection_is_recorded_separately_from_identical_raw_history():
    before = snapshot(record())
    after = deepcopy(before)
    after["api_history"][0]["outputs"][0]["latex"] = "new presentation"
    assert verifier.compare_history(before, after) == [
        {
            "execution_id": "owned-run",
            "optional_events_presence_changed": False,
            "latex_presentation_changed": True,
        }
    ]


def test_lost_output_content_cannot_be_called_presentation():
    before = snapshot(record())
    after = deepcopy(before)
    after["api_history"][0]["outputs"][0]["data"] = "missing original output"
    with pytest.raises(AssertionError, match="output content changed"):
        verifier.compare_history(before, after)


def test_failure_is_persisted_before_any_runtime_can_start(monkeypatch, tmp_path):
    def invalid_identity(*args):
        raise ValueError("synthetic identity refusal")

    monkeypatch.setattr(verifier, "bundled_identity", invalid_identity)
    directory = tmp_path / "failure-evidence"
    with pytest.raises(ValueError, match="synthetic identity refusal"):
        verifier.verify(tmp_path / "unopened.app", tmp_path / "unopened-runtime", directory)
    import json

    receipt = json.loads((directory / "acceptance.json").read_text())
    assert receipt["status"] == "failed" and receipt["active_stage"] == "identity"
    assert receipt["completed_stages"] == [] and receipt["temporary_project_removed"]
    assert json.loads((directory / "failure.json").read_text())["type"] == "ValueError"
