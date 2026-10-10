"""Keep daily science verification separate from required quick PR checks."""
from pathlib import Path

from scripts.verify_parallel_merge_gate import CONTEXT


ROOT = Path(__file__).resolve().parents[1]


def triggers(workflow):
    return workflow.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]


def test_full_suite_is_daily_and_manual_with_all_original_shards():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    events = triggers(workflow)
    assert "  schedule:\n    - cron: '17 1 * * *'" in events
    assert "  workflow_dispatch:" in events
    assert all(f"  {event}:" not in events for event in ("pull_request", "push", "merge_group"))
    assert "sdk: ['3.11', '3.13']" in workflow
    assert "shard: [0, 1, 2, 3]" in workflow
    assert "scripts/verify_parallel_merge_gate.py" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "name: OpenEconometrics / daily full gate" in workflow
    assert f"name: {CONTEXT}\n" in workflow
    assert "name: OpenEconometrics / merge gate\n" not in workflow


def test_quick_gate_requires_successful_sdk_and_web_on_all_merge_events():
    workflow = (ROOT / ".github/workflows/ci-quick.yml").read_text()
    events = triggers(workflow)
    assert all(f"  {event}:" in events for event in ("pull_request", "push", "merge_group"))
    assert "branches: [main]" in events and "  schedule:" not in events
    assert "python: ['3.11', '3.13']" in workflow
    aggregate = workflow.split("\n  merge-gate:\n", 1)[1]
    assert "name: OpenEconometrics / merge gate\n" in aggregate
    assert "needs: [sdk-smoke, web]" in aggregate and "if: always()" in aggregate
    for job, variable in (("sdk-smoke", "SDK_RESULT"), ("web", "WEB_RESULT")):
        assert f"{variable}: ${{{{ needs.{job}.result }}}}" in aggregate
        assert f'test "${variable}" = success' in aggregate
    assert "continue-on-error:" not in workflow
    assert "scripts/run_parallel_sdk_groups.py" not in workflow
    assert "junit_counts(Path('artifacts/quick/pytest.xml'))" in workflow
    assert "'full_sdk_suite': False" in workflow
    assert "scripts/generate_capability_docs.py --check" in workflow
    assert "scripts/generate_editor_api.py --check" in workflow
    assert "scripts/verify_web_gate.mjs" in workflow
    assert "npm --prefix web run build" in workflow
    web = workflow.split("\n  web:\n", 1)[1].split("\n  merge-gate:\n", 1)[0]
    assert "fetch-depth: 0" in web
