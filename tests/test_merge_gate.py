"""Acceptance evidence must fail closed for empty, skipped and stale executions."""
import ast
import base64
import gzip
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("merge_gate", ROOT / "scripts/verify_merge_candidate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_weighted_binary_gate_extension_preserves_the_entire_base_scope():
    extension = json.loads((ROOT / "docs/econometrics/weighted-binary-gate-extension-2026-10-10.json").read_text())
    raw = subprocess.check_output(
        ["git", "show", extension["base_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == extension["base_producer_sha256"]
    original = ast.literal_eval(next(
        node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    assert len(original) == extension["base_selector_count"] == 138
    assert gate.TESTS[:142] == original + extension["appended_selectors"]
    assert extension["appended_selectors"] == [
        "tests/test_weighted_binary.py", "tests/test_econ_glm.py",
        "tests/test_econ_glm_oracle.py", "tests/test_econ_saved_prediction_categories.py"]
    assert len(gate.TESTS[:142]) == len(set(gate.TESTS[:142])) == extension["complete_selector_count"] == 142
    for selectors, key in ((original, "base_selectors_sha256"),
                           (gate.TESTS[:142], "complete_selectors_sha256")):
        assert hashlib.sha256(json.dumps(selectors, separators=(",", ":")).encode()).hexdigest() == extension[key]
    assert hashlib.sha256((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan.json").read_bytes()).hexdigest() == extension["unchanged_base_plan_sha256"]
    assert extension["timeout_seconds"] == 900
    assert extension["sdk_minors"] == ["3.11", "3.13"]
    assert extension["distributed_shards_per_minor"] == 4
    current = json.loads((ROOT / "docs/econometrics/fully-stratified-four-stage-gate-extension-2026-10-10.json").read_text())
    incoming_raw = subprocess.check_output(["git", "show", current["base_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    incoming = ast.literal_eval(next(node.value for node in ast.parse(incoming_raw).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    assert hashlib.sha256(incoming_raw).hexdigest() == current["base_producer_sha256"]
    assert incoming == gate.TESTS[:142]
    assert gate.TESTS[:144] == incoming + current["appended_selectors"]
    assert current["appended_selectors"] == ['tests/test_survey_fully_stratified_four_stage.py', 'tests/test_survey_fully_stratified_four_stage_safety.py']
    assert len(gate.TESTS[:144]) == len(set(gate.TESTS[:144])) == current["complete_selector_count"] == 144
    assert current["timeout_seconds"] == 900
    assert current["sdk_minors"] == ["3.11", "3.13"]
    assert current["all_current_main_selectors_in_original_order"] is True


def test_score_gate_extension_preserves_full_static_and_runtime_scope():
    extension = json.loads((ROOT / "docs/econometrics/multivariate-score-gate-extension-2026-10-10.json").read_text())
    raw = subprocess.check_output(["git", "show", extension["base_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == extension["base_producer_sha256"]
    def selectors(source):
        return ast.literal_eval(next(node.value for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    original = selectors(raw)
    assert len(original) == extension["base_selector_count"] == 145
    assert extension["appended_selectors"] == ["tests/test_multivariate_score_uncertainty.py"]
    assert selectors((ROOT / "scripts/verify_merge_candidate.py").read_text())[:146] == gate.TESTS[:146] == original + extension["appended_selectors"]
    assert len(gate.TESTS[:146]) == len(set(gate.TESTS[:146])) == extension["complete_selector_count"] == 146
    for values, key in ((original, "base_selectors_sha256"), (gate.TESTS[:146], "complete_selectors_sha256")):
        assert hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest() == extension[key]
    assert hashlib.sha256((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan.json").read_bytes()).hexdigest() == extension["unchanged_base_plan_sha256"]
    assert extension["timeout_seconds"] == 900 and extension["sdk_minors"] == ["3.11", "3.13"]
    assert extension["distributed_shards_per_minor"] == 4


def _node_gate_fixture(tmp_path, cloud="pass"):
    node = shutil.which("node")
    assert node is not None, "The source gate requires provisioned Node 24"
    script = tmp_path / "scripts/verify_web_gate.mjs"
    script.parent.mkdir()
    script.write_bytes((ROOT / "scripts/verify_web_gate.mjs").read_bytes())
    web = tmp_path / "web/tests"
    web.mkdir(parents=True)
    if cloud != "no_web":
        (web / "existing.test.ts").write_text(
            "import test from 'node:test'; test('existing web guard', () => {});\n")
    controller = tmp_path / "desktop/scripts/verify_windows_cloud_ui.test.mjs"
    controller.parent.mkdir(parents=True)
    if cloud != "missing":
        source = "import test from 'node:test';\n"
        if cloud != "empty":
            for index in range(20 if cloud == "short" else 21):
                name = json.dumps(f"cloud guard {index}")
                if index == 0 and cloud == "failed":
                    source += f"test({name}, () => {{ throw new Error('cloud failure sentinel'); }});\n"
                elif index == 0 and cloud in ("skipped", "todo"):
                    method = "skip" if cloud == "skipped" else "todo"
                    source += f"test.{method}({name}, () => {{}});\n"
                else:
                    source += f"test({name}, () => {{}});\n"
        controller.write_text(source)
    return node, script


@pytest.mark.parametrize("cloud", ["pass", "missing", "empty", "short", "failed", "skipped", "todo", "no_web"])
def test_cloud_node_guards_are_required_separately_from_passing_web_tests(tmp_path, cloud):
    node, script = _node_gate_fixture(tmp_path, cloud)
    result = subprocess.run([node, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is (cloud == "pass"), result.stdout + result.stderr
    if cloud == "no_web":
        assert "No web tests discovered" in result.stderr
    else:
        assert "existing web guard" in result.stdout and "# pass 1" in result.stdout
    if cloud == "pass":
        assert "cloud guard 20" in result.stdout and "# tests 21" in result.stdout
    elif cloud in ("empty", "short", "skipped", "todo"):
        assert "Incomplete test gate" in result.stderr


def test_web_and_cloud_node_suites_share_the_original_deadline(tmp_path):
    node, script = _node_gate_fixture(tmp_path)
    # Advance the parent clock after the actual web subprocess. No long wait or
    # replacement child result is needed to prove the second suite cannot reset it.
    launcher = (
        "let checks = 0; Object.defineProperty(globalThis, 'performance', {value: {"
        "now: () => ++checks <= 3 ? 0 : 600001}}); "
        f"await import({json.dumps(script.as_uri())});"
    )
    result = subprocess.run([node, "--input-type=module", "-e", launcher], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "existing web guard" in result.stdout and "# pass 1" in result.stdout
    assert "shared 600-second deadline" in result.stderr
    assert "cloud guard" not in result.stdout



@pytest.mark.parametrize("field,count", [("tests", 0), ("failures", 1), ("errors", 1), ("skipped", 1)])
def test_empty_failed_error_or_skipped_is_not_pass(tmp_path, field, count):
    attrs = dict(tests=4, failures=0, errors=0, skipped=0)
    attrs[field] = count
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite ' + ' '.join(f'{k}="{v}"' for k, v in attrs.items()) + '/></testsuites>')
    with pytest.raises(ValueError, match="nonempty, passing, unskipped"):
        gate.junit_counts(path)


def test_nonempty_success_is_counted(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="7" failures="0" errors="0" skipped="0"/></testsuites>')
    assert gate.junit_counts(path) == dict(tests=7, failures=0, errors=0, skipped=0)


def test_current_main_plan_retains_all_prior_selectors_and_one_thread_environment():
    newest = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan.json").read_text())
    archived = {}
    for label, commit in [("previous_current_main830_full126_plan", "83091ae7eda517c2954efb373f7e5d79e68dcc3f"),
                          ("previous_current_PR213_cdf620_full136_plan", "cdf620a5dea7da61df34a70fa1525ac7dabb4094")]:
        record = newest[label]
        compressed = base64.b64decode(record["compressed_archive"]["value"], validate=True)
        assert len(compressed) == record["compressed_archive"]["bytes"]
        assert hashlib.sha256(compressed).hexdigest() == record["compressed_archive"]["sha256"]
        prior_raw = gzip.decompress(compressed)
        assert record["source_commit"] == commit
        assert prior_raw == subprocess.check_output(["git", "show", commit + ":" + record["path"]], cwd=ROOT)
        assert len(prior_raw) == record["bytes"]
        assert hashlib.sha256(prior_raw).hexdigest() == record["sha256"]
        archived[label] = json.loads(prior_raw)
    def _check_complete_previous136(plan, scoped_tests):
        future_registered = plan
        current_main_history = future_registered["previous_current_main769_full124_plan"]
        current_main_compressed = base64.b64decode(current_main_history["compressed_archive"]["value"], validate=True)
        assert len(current_main_compressed) == current_main_history["compressed_archive"]["bytes"]
        assert hashlib.sha256(current_main_compressed).hexdigest() == current_main_history["compressed_archive"]["sha256"]
        current_main_raw = gzip.decompress(current_main_compressed)
        assert current_main_raw == subprocess.check_output(["git", "show", current_main_history["source_commit"] + ":" + current_main_history["path"]], cwd=ROOT)
        assert len(current_main_raw) == current_main_history["bytes"]
        assert hashlib.sha256(current_main_raw).hexdigest() == current_main_history["sha256"]
        combined_registered = json.loads(current_main_raw)
        main124_tests = combined_registered["selected_tests"]
        balanced_history = combined_registered["previous_current_pr218_262b_plan"]
        balanced_compressed = base64.b64decode(balanced_history["compressed_archive"]["value"], validate=True)
        assert len(balanced_compressed) == balanced_history["compressed_archive"]["bytes"]
        assert hashlib.sha256(balanced_compressed).hexdigest() == balanced_history["compressed_archive"]["sha256"]
        balanced_raw = gzip.decompress(balanced_compressed)
        assert balanced_raw == subprocess.check_output(["git", "show", balanced_history["source_commit"] + ":" + balanced_history["path"]], cwd=ROOT)
        assert len(balanced_raw) == balanced_history["bytes"]
        assert hashlib.sha256(balanced_raw).hexdigest() == balanced_history["sha256"]
        live_registered = json.loads(balanced_raw)
        history = live_registered["previous_current_pr218_9aa_plan"]
        history_compressed = base64.b64decode(history["compressed_archive"]["value"], validate=True)
        assert hashlib.sha256(history_compressed).hexdigest() == history["compressed_archive"]["sha256"]
        assert len(history_compressed) == history["compressed_archive"]["bytes"]
        original_raw = gzip.decompress(history_compressed)
        original_git = subprocess.check_output(["git", "show", history["source_commit"] + ":" + history["path"]], cwd=ROOT)
        assert original_raw == original_git
        assert len(original_raw) == history["bytes"]
        assert hashlib.sha256(original_raw).hexdigest() == history["sha256"]
        current_registered = json.loads(original_raw)
        registered_tests = current_registered["selected_tests"]
        plan = current_registered
        registered_plan = current_registered["previous_current_b634_plan"]["value"]
        b634_tests = registered_plan["selected_tests"]
        current_plan = registered_plan["previous_current_main99_plan"]
        prior_raw = subprocess.check_output(
            ["git", "show", "39f47872dcb46014d753c298c0d0cc4344748da4:docs/econometrics/merge-gate-sdk-groups-plan.json"],
            cwd=ROOT)
        assert hashlib.sha256(prior_raw).hexdigest() == registered_plan["previous_current_main99_plan_sha256"] == "ed7283aa1af316deb8d6af5ec274e9cffaa76f3e8aba5471fac07b11cbfa6764"
        assert json.loads(prior_raw) == current_plan
        prior_source = subprocess.check_output(
            ["git", "show", "39f47872dcb46014d753c298c0d0cc4344748da4:scripts/verify_merge_candidate.py"],
            cwd=ROOT, text=True)
        prior_tests = ast.literal_eval(next(
            node.value for node in ast.parse(prior_source).body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
        wrapped = current_plan["previous_current_categorical_plan"]
        assert wrapped["source_commit"] == "743b5ed7b6ed9ec1bec2ab54e09621eee7da3f24"
        assert wrapped["sha256"] == "505e02ef1e12209fa0cbb91007d6d68c4491384d6c55c613ebfe55715832e27c"
        assert wrapped["sha256"] == hashlib.sha256(
            (json.dumps(wrapped["value"], indent=2) + "\n").encode()).hexdigest()
        plan = wrapped["value"]
        historical_tests = plan["selected_tests"]
        historical = plan["previous_three_group_plan"]
        previous = historical["value"]
        assert historical["source_commit"] == "6ea254163149439de892c3dc64e092f220642ed9"
        assert historical["sha256"] == hashlib.sha256(
            (json.dumps(previous, indent=2) + "\n").encode()).hexdigest()
        assert len(previous["selected_tests"]) == 85
        assert previous["selected_test_scope_sha256"] == "ebbe449bc05acf1dd040768a078c7c42b25e7bd4a02944517e998cb23c266ab4"
        additions = ["tests/test_mprobit.py", "tests/test_mprobit_postestimation.py",
                     "tests/test_mprobit_independent.py"]
        assert plan["authoritative_main_commit"] == "3a8c51d0f942409e4cd3a1808332679bcef23eb5"
        assert plan["main_added_selectors"] == additions
        historical_main = plan["previous_current_main_plan"]
        old_main = historical_main["value"]
        assert historical_main["source_commit"] == "bc3a846b4a4893074334ef66f75a6fb801df7de1"
        assert historical_main["sha256"] == "250dd4ac8e016bf571228ecf64bec24d4c49a8aa589093f0fe92272232d633c4"
        assert historical_main["sha256"] == hashlib.sha256(
            (json.dumps(old_main, indent=2) + "\n").encode()).hexdigest()
        assert old_main["selected_tests"] == additions + previous["selected_tests"]
        assert len(old_main["selected_tests"]) == 88
        incoming = plan["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == 89
        assert [item for item in historical_tests if item in incoming] == incoming
        assert [item for item in historical_tests if item in old_main["selected_tests"]] == old_main["selected_tests"]
        assert [item for item in historical_tests if item not in old_main["selected_tests"]] == plan["incoming_main_added_selectors"]
        assert [item for item in historical_tests if item not in incoming] == plan["prior_branch_added_selectors"]
        assert len(plan["incoming_main_added_selectors"]) == 10
        assert len(plan["prior_branch_added_selectors"]) == 9
        historical94 = plan["previous_current_survey_plan"]
        prior94 = historical94["value"]["selected_tests"]
        assert len(prior94) == 94
        assert historical94["sha256"] == hashlib.sha256(
            (json.dumps(historical94["value"], indent=2) + "\n").encode()).hexdigest()
        assert [item for item in historical_tests if item in prior94] == prior94
        assert len(plan["latest_main_added_selectors"]) == 4
        assert plan["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert plan["serial_steps_plan"]["default_timeout_seconds"] == 1800
        assert historical_tests == plan["selected_tests"]
        assert len(historical_tests) == len(set(historical_tests)) == 98
        assert plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(historical_tests, separators=(",", ":")).encode()).hexdigest()
        assert [len(group) for group in plan["effective_groups"]] == [1, 5, 92]
        distributed = plan["hosted_distributed_plan"]
        assert distributed["job_count"] == 4 and distributed["children_per_job"] == 2
        assert distributed["global_child_indices_by_shard"] == [[0, 1], [2, 3]]
        assert distributed["cohort_deadline_seconds_including_collection_queue_cleanup_and_aggregate_reassembly"] == 900
        current = gate.test_environment()
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert current[variable] == plan["environment"][variable] == "1"
            assert plan["historical_measurements_environment"][variable] == previous["environment"][variable] == "2"
        assert plan["timeout_seconds"] == previous["timeout_seconds"] == 900

        assert current_plan["authoritative_main_commit"] == "99d530614aff4fc6abbbb2b2b52d94751630c1cf"
        assert prior_tests == current_plan["selected_tests"]
        assert len(prior_tests) == len(set(prior_tests)) == 111
        incoming = current_plan["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == 102
        assert [item for item in prior_tests if item in incoming] == incoming
        assert [item for item in prior_tests if item in historical_tests] == historical_tests
        assert [item for item in prior_tests if item not in historical_tests] == current_plan["incoming_main_added_selectors"]
        assert [item for item in prior_tests if item not in incoming] == current_plan["prior_branch_added_selectors"]
        assert len(current_plan["incoming_main_added_selectors"]) == 13
        assert len(current_plan["prior_branch_added_selectors"]) == 9
        assert current_plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(prior_tests, separators=(",", ":")).encode()).hexdigest()
        assert [len(group) for group in current_plan["effective_groups"]] == [1, 5, 105]
        assert current_plan["hosted_legacy_plan"]["group_count"] == 3
        assert current_plan["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert current_plan["hosted_legacy_plan"]["historical_two_child_receipts_eligible"] is False
        distributed = current_plan["hosted_distributed_plan"]
        assert distributed["job_count"] == 8 and distributed["children_per_job"] == 1
        assert distributed["shard_count_per_version"] == 4
        assert distributed["global_child_indices_by_shard"] == [[0], [1], [2], [3]]
        assert distributed["cohort_deadline_seconds_including_collection_queue_cleanup_and_aggregate_reassembly"] == 900
        assert current_plan["environment"] == plan["environment"]
        assert current_plan["timeout_seconds"] == plan["timeout_seconds"] == 900


        assert registered_plan["source_parent_commit"] == "39f47872dcb46014d753c298c0d0cc4344748da4"
        assert registered_plan["authoritative_main_commit"] == "b6347f31e2cd38a069ccd3599a5d147dd9266fae"
        sequential = ["tests/test_sequential_kernels.py", "tests/test_sequential_oracles.py",
                      "tests/test_sequential_contracts.py"]
        assert b634_tests == registered_plan["selected_tests"] == sequential + prior_tests
        assert len(b634_tests) == len(set(b634_tests)) == 114
        incoming = registered_plan["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == len(set(incoming)) == 105 and set(incoming) <= set(b634_tests)
        qualification = registered_plan["selector_order_qualification"]
        moved = qualification["incoming_moved_to_end_conflicting_selectors"]
        assert len(moved) == 8 and qualification["both_complete_orders_compatible"] is False
        assert [item for item in b634_tests if item in prior_tests] == prior_tests
        assert [item for item in b634_tests if item in incoming and item not in moved] == [
            item for item in incoming if item not in moved]
        assert [item for item in b634_tests if item not in incoming] == current_plan["prior_branch_added_selectors"]
        assert registered_plan["selected_test_scope_sha256"] == "5d689c78ec0a7df70e8b6c2dfeb5bee77c46727185c78e86c5587ffa8ab1004c"
        assert registered_plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(b634_tests, separators=(",", ":")).encode()).hexdigest()
        assert [len(group) for group in registered_plan["effective_groups"]] == [1, 5, 108]
        assert registered_plan["fixed_groups"] == current_plan["fixed_groups"]
        assert registered_plan["environment"] == current_plan["environment"]
        assert registered_plan["timeout_seconds"] == current_plan["timeout_seconds"] == 900
        assert registered_plan["explicit_local_cpu_policy"]["threads_per_child"] == 1
        assert registered_plan["hosted_legacy_plan"]["group_count"] == 3
        assert registered_plan["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert registered_plan["hosted_legacy_plan"]["historical_two_child_receipts_eligible"] is False
        distributed = registered_plan["hosted_distributed_plan"]
        assert distributed["job_count"] == 8 and distributed["children_per_job"] == 1
        assert distributed["shard_count_per_version"] == 4
        assert distributed["global_child_indices_by_shard"] == [[0], [1], [2], [3]]
        assert distributed["cohort_deadline_seconds_including_collection_queue_cleanup_and_aggregate_reassembly"] == 900

        assert current_registered["authoritative_main_commit"] == "9e511cb32c828a2cc3293053fb0a5b2248da4d9e"
        assert current_registered["previous_current_b634_plan"]["sha256"] == hashlib.sha256(
            (json.dumps(registered_plan, indent=2) + "\n").encode()).hexdigest()
        assert registered_tests == current_registered["selected_tests"]
        assert len(registered_tests) == len(set(registered_tests)) == 119
        assert [item for item in registered_tests if item in b634_tests] == b634_tests
        incoming = current_registered["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == 110 and set(incoming) <= set(registered_tests)
        assert [item for item in registered_tests if item not in b634_tests] == current_registered["latest_main_added_selectors"]
        assert [len(group) for group in current_registered["effective_groups"]] == [1, 5, 113]
        archived = current_registered["previous_current_5b_plan"]
        compressed = base64.b64decode(archived["compressed_archive"]["value"], validate=True)
        assert len(compressed) == archived["compressed_archive"]["bytes"]
        assert hashlib.sha256(compressed).hexdigest() == archived["compressed_archive"]["sha256"]
        archived_raw = gzip.decompress(compressed)
        frozen_raw = subprocess.check_output(
            ["git", "show", "5b9b5b04d62127654dd90d93811f54e7fab24117:docs/econometrics/merge-gate-sdk-groups-plan.json"],
            cwd=ROOT)
        assert archived_raw == frozen_raw
        assert len(archived_raw) == archived["bytes"] == 3793047
        assert hashlib.sha256(archived_raw).hexdigest() == archived["sha256"] == "ad5983269a77250c3d419c599d1acc15bcb0eb86a84c63666fe22da42649d517"
        frozen_plan = json.loads(archived_raw)
        assert frozen_plan["previous_current_b634_plan"] == current_registered["previous_current_b634_plan"]
        assert registered_tests[:-2] == frozen_plan["selected_tests"]
        assert registered_tests[-2:] == ["tests/test_verify_team_cleanup.py",
                                  "tests/test_verify_windows_cloud_ui_fixture.py"]
        assert current_registered["timeout_seconds"] == 900
        assert current_registered["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(registered_tests, separators=(",", ":")).encode()).hexdigest()

        assert history["source_commit"] == "9aa6fb18fb079b5f581009c73e8ef8ad1af59807"
        assert live_registered["source_parent_commit"] == history["source_commit"]
        assert live_registered["authoritative_main_commit"] == "3ad8b093eb8a7e45624bd79fd2b9623725a92898"
        assert live_registered["selected_tests"] == current_registered["selected_tests"] == registered_tests
        assert live_registered["selected_test_scope_sha256"] == current_registered["selected_test_scope_sha256"]
        assert live_registered["fixed_groups"][0] == current_registered["fixed_groups"][0]
        assert live_registered["fixed_groups"][1][:-1] == current_registered["fixed_groups"][1]
        assert live_registered["fixed_groups"][1][-1] == "tests/test_parallel_sdk_groups.py"
        assert [len(group) for group in live_registered["effective_groups"]] == [1, 6, 112]
        assert live_registered["environment"] == current_registered["environment"]
        assert live_registered["timeout_seconds"] == current_registered["timeout_seconds"] == 900
        for key in ("named_execution_budgets", "hosted_distributed_plan", "explicit_local_cpu_policy",
                    "previous_current_b634_plan", "previous_current_5b_plan", "measurements"):
            assert live_registered[key] == current_registered[key]
        assert live_registered["hosted_legacy_plan"]["groups"] == live_registered["effective_groups"]
        assert live_registered["hosted_legacy_plan"]["timeout_seconds"] == current_registered["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert live_registered["current_controller_balance"]["actual_acceptance_of_proposal"] is False

        incoming_history = combined_registered["previous_current_main_ba5_plan"]
        incoming_compressed = base64.b64decode(incoming_history["compressed_archive"]["value"], validate=True)
        assert len(incoming_compressed) == incoming_history["compressed_archive"]["bytes"]
        assert hashlib.sha256(incoming_compressed).hexdigest() == incoming_history["compressed_archive"]["sha256"]
        incoming_raw = gzip.decompress(incoming_compressed)
        assert incoming_raw == subprocess.check_output(["git", "show", incoming_history["source_commit"] + ":" + incoming_history["path"]], cwd=ROOT)
        assert len(incoming_raw) == incoming_history["bytes"]
        assert hashlib.sha256(incoming_raw).hexdigest() == incoming_history["sha256"]
        incoming_registered = json.loads(incoming_raw)
        assert balanced_history["source_commit"] == "262b7e9d8ac3d7e33ac123be866fbd48ca401912"
        assert incoming_history["source_commit"] == "ba5b7e5b147277e9d744e6a36b3a70ba6ac71a17"
        assert incoming_registered["authoritative_main_commit"] == "3ad8b093eb8a7e45624bd79fd2b9623725a92898"
        assert incoming_registered["selected_tests"][:119] == registered_tests
        assert incoming_registered["selected_tests"][119:] == incoming_registered["twostep_added_selectors"]
        assert len(incoming_registered["selected_tests"]) == len(set(incoming_registered["selected_tests"])) == 124
        assert [x for x in incoming_registered["selected_tests"] if x in b634_tests] == b634_tests
        assert [x for x in incoming_registered["selected_tests"] if x not in b634_tests] == incoming_registered["latest_main_added_selectors"]
        assert [len(group) for group in incoming_registered["effective_groups"]] == [1, 5, 118]
        assert incoming_registered["selected_test_scope_sha256"] == hashlib.sha256(json.dumps(incoming_registered["selected_tests"], separators=(",", ":")).encode()).hexdigest()
        assert incoming_registered["selected_tests"][:117] == frozen_plan["selected_tests"]
        assert incoming_registered["selected_tests"][117:119] == ["tests/test_verify_team_cleanup.py", "tests/test_verify_windows_cloud_ui_fixture.py"]
        assert incoming_registered["previous_current_b634_plan"] == current_registered["previous_current_b634_plan"]
        assert incoming_registered["previous_current_5b_plan"] == current_registered["previous_current_5b_plan"]
        assert combined_registered["source_parent_commit"] == balanced_history["source_commit"]
        assert combined_registered["authoritative_main_commit"] == incoming_history["source_commit"]
        assert combined_registered["selected_tests"] == incoming_registered["selected_tests"] == main124_tests
        assert main124_tests[:119] == live_registered["selected_tests"] == registered_tests
        assert main124_tests[119:] == combined_registered["twostep_added_selectors"] == incoming_registered["twostep_added_selectors"]
        assert len(main124_tests) == len(set(main124_tests)) == 124
        assert combined_registered["selected_test_scope_sha256"] == incoming_registered["selected_test_scope_sha256"]
        assert combined_registered["fixed_groups"] == live_registered["fixed_groups"]
        assert [len(group) for group in combined_registered["effective_groups"]] == [1, 6, 117]
        assert combined_registered["effective_groups"][:2] == live_registered["effective_groups"][:2]
        assert combined_registered["effective_groups"][2][:-5] == live_registered["effective_groups"][2]
        assert combined_registered["effective_groups"][2][-5:] == incoming_registered["twostep_added_selectors"]
        assert combined_registered["timeout_seconds"] == incoming_registered["timeout_seconds"] == live_registered["timeout_seconds"] == 900
        for key in ("environment", "named_execution_budgets", "hosted_distributed_plan", "explicit_local_cpu_policy",
                    "previous_current_b634_plan", "previous_current_5b_plan", "measurements", "scientific_source_change",
                    "source_science_bridge_required", "scientific_source_change_scope", "current_controller_balance"):
            assert combined_registered[key] == live_registered[key]
        assert combined_registered["hosted_legacy_plan"]["groups"] == combined_registered["effective_groups"]
        assert combined_registered["hosted_legacy_plan"]["timeout_seconds"] == incoming_registered["hosted_legacy_plan"]["timeout_seconds"] == 1800
        integration = combined_registered["current_main_integration"]
        assert integration["incoming_main_scientific_sources_preserved_exactly"] is True
        assert integration["old_historical_proofs_are_not_current_base_acceptance"] is True
        assert integration["all_inherited_source_science_bridge_requirements_preserved"] is True
        assert integration["actual_acceptance_of_proposal"] is False

        assert current_main_history["source_commit"] == "769f5b5618b6d03f88cebfd999286995373678b6"
        owned_history = future_registered["previous_current_PR213_2a_full131_plan"]
        owned_compressed = base64.b64decode(owned_history["compressed_archive"]["value"], validate=True)
        assert len(owned_compressed) == owned_history["compressed_archive"]["bytes"]
        assert hashlib.sha256(owned_compressed).hexdigest() == owned_history["compressed_archive"]["sha256"]
        owned_raw = gzip.decompress(owned_compressed)
        assert owned_raw == subprocess.check_output(["git", "show", owned_history["source_commit"] + ":" + owned_history["path"]], cwd=ROOT)
        assert len(owned_raw) == owned_history["bytes"]
        assert hashlib.sha256(owned_raw).hexdigest() == owned_history["sha256"]
        owned_registered = json.loads(owned_raw)
        assert owned_history["source_commit"] == "2a3228bec1ccc80a907cf3b46f5a9cf75e5e05a4"
        assert owned_registered["authoritative_main_commit"] == "3ad8b093eb8a7e45624bd79fd2b9623725a92898"
        assert len(owned_registered["selected_tests"]) == len(set(owned_registered["selected_tests"])) == 131
        assert owned_registered["selected_tests"][:119] == registered_tests
        assert [len(group) for group in owned_registered["effective_groups"]] == [1, 5, 125]
        assert owned_registered["selected_test_scope_sha256"] == hashlib.sha256(json.dumps(owned_registered["selected_tests"], separators=(",", ":")).encode()).hexdigest()
        assert owned_registered["previous_current_b634_plan"] == current_registered["previous_current_b634_plan"]
        assert owned_registered["previous_current_5b_plan"] == current_registered["previous_current_5b_plan"]
        assert owned_registered["timeout_seconds"] == 900
        owned_additions = owned_registered["selected_tests"][119:]
        assert len(owned_additions) == len(set(owned_additions)) == 12
        assert future_registered["selected_tests"] == scoped_tests == main124_tests + owned_additions
        assert len(scoped_tests) == len(set(scoped_tests)) == 136
        assert future_registered["authoritative_main_commit"] == current_main_history["source_commit"]
        assert future_registered["source_parent_commit"] == owned_history["source_commit"]
        assert future_registered["fixed_groups"][0] == combined_registered["fixed_groups"][0]
        assert future_registered["fixed_groups"][1] == combined_registered["fixed_groups"][1] + owned_additions
        assert future_registered["effective_groups"][:2] == [[item for item in scoped_tests if item in group] for group in future_registered["fixed_groups"]]
        assert future_registered["effective_groups"][2] == combined_registered["effective_groups"][2]
        assert [len(group) for group in future_registered["effective_groups"]] == [1, 18, 117]
        assert future_registered["selected_test_scope_sha256"] == hashlib.sha256(json.dumps(scoped_tests, separators=(",", ":")).encode()).hexdigest()
        assert future_registered["timeout_seconds"] == 900
        for key in ("environment", "named_execution_budgets", "hosted_distributed_plan", "explicit_local_cpu_policy",
                    "previous_current_b634_plan", "previous_current_5b_plan", "measurements", "scientific_source_change",
                    "source_science_bridge_required", "scientific_source_change_scope", "current_controller_balance"):
            assert future_registered[key] == combined_registered[key]
        integration213 = future_registered["PR213_current_main_integration"]
        assert integration213["complete136_order"] == scoped_tests
        assert integration213["all12_original_scientific_additions"] == owned_additions
        assert integration213["full900_required"] is True
        assert integration213["old_timings_are_not_capacity_proof"] is True
        assert integration213["actual_acceptance_of_proposal"] is False
        for key in ("scientific_source_change", "source_science_bridge_required", "scientific_source_change_scope"):
            assert integration213["all_original213_science_requirements_preserved"][key] == owned_registered[key]
        assert future_registered["hosted_legacy_plan"]["groups"] == future_registered["effective_groups"]
        assert future_registered["hosted_legacy_plan"]["timeout_seconds"] == 1800
    def _check_complete_incoming126(plan, scoped_tests):
        prospective_plan = plan
        extension = prospective_plan["survey_deff_scope_extension"]
        incoming124_raw = subprocess.check_output(
            ["git", "show", extension["prior_plan_source_commit"] + ":" + extension["prior_plan_path"]],
            cwd=ROOT)
        assert extension["prior_plan_source_commit"] == "769f5b5618b6d03f88cebfd999286995373678b6"
        assert len(incoming124_raw) == extension["prior_plan_bytes"] == 8902107
        assert hashlib.sha256(incoming124_raw).hexdigest() == extension["prior_plan_sha256"] == (
            "c45e85f6a376a44c45e92cf1e31ef3cb3b59c9f13bbfefb8ea3196362f044797")
        plan = json.loads(incoming124_raw)
        incoming124_producer = subprocess.check_output(
            ["git", "show", extension["prior_plan_source_commit"] + ":scripts/verify_merge_candidate.py"],
            cwd=ROOT, text=True)
        prior_gate_tests = ast.literal_eval(next(
            node.value for node in ast.parse(incoming124_producer).body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
        combined_registered = plan
        balanced_history = combined_registered["previous_current_pr218_262b_plan"]
        balanced_compressed = base64.b64decode(balanced_history["compressed_archive"]["value"], validate=True)
        assert len(balanced_compressed) == balanced_history["compressed_archive"]["bytes"]
        assert hashlib.sha256(balanced_compressed).hexdigest() == balanced_history["compressed_archive"]["sha256"]
        balanced_raw = gzip.decompress(balanced_compressed)
        assert balanced_raw == subprocess.check_output(["git", "show", balanced_history["source_commit"] + ":" + balanced_history["path"]], cwd=ROOT)
        assert len(balanced_raw) == balanced_history["bytes"]
        assert hashlib.sha256(balanced_raw).hexdigest() == balanced_history["sha256"]
        live_registered = json.loads(balanced_raw)
        history = live_registered["previous_current_pr218_9aa_plan"]
        history_compressed = base64.b64decode(history["compressed_archive"]["value"], validate=True)
        assert hashlib.sha256(history_compressed).hexdigest() == history["compressed_archive"]["sha256"]
        assert len(history_compressed) == history["compressed_archive"]["bytes"]
        original_raw = gzip.decompress(history_compressed)
        original_git = subprocess.check_output(["git", "show", history["source_commit"] + ":" + history["path"]], cwd=ROOT)
        assert original_raw == original_git
        assert len(original_raw) == history["bytes"]
        assert hashlib.sha256(original_raw).hexdigest() == history["sha256"]
        current_registered = json.loads(original_raw)
        registered_tests = current_registered["selected_tests"]
        plan = current_registered
        registered_plan = current_registered["previous_current_b634_plan"]["value"]
        b634_tests = registered_plan["selected_tests"]
        current_plan = registered_plan["previous_current_main99_plan"]
        prior_raw = subprocess.check_output(
            ["git", "show", "39f47872dcb46014d753c298c0d0cc4344748da4:docs/econometrics/merge-gate-sdk-groups-plan.json"],
            cwd=ROOT)
        assert hashlib.sha256(prior_raw).hexdigest() == registered_plan["previous_current_main99_plan_sha256"] == "ed7283aa1af316deb8d6af5ec274e9cffaa76f3e8aba5471fac07b11cbfa6764"
        assert json.loads(prior_raw) == current_plan
        prior_source = subprocess.check_output(
            ["git", "show", "39f47872dcb46014d753c298c0d0cc4344748da4:scripts/verify_merge_candidate.py"],
            cwd=ROOT, text=True)
        prior_tests = ast.literal_eval(next(
            node.value for node in ast.parse(prior_source).body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
        wrapped = current_plan["previous_current_categorical_plan"]
        assert wrapped["source_commit"] == "743b5ed7b6ed9ec1bec2ab54e09621eee7da3f24"
        assert wrapped["sha256"] == "505e02ef1e12209fa0cbb91007d6d68c4491384d6c55c613ebfe55715832e27c"
        assert wrapped["sha256"] == hashlib.sha256(
            (json.dumps(wrapped["value"], indent=2) + "\n").encode()).hexdigest()
        plan = wrapped["value"]
        historical_tests = plan["selected_tests"]
        historical = plan["previous_three_group_plan"]
        previous = historical["value"]
        assert historical["source_commit"] == "6ea254163149439de892c3dc64e092f220642ed9"
        assert historical["sha256"] == hashlib.sha256(
            (json.dumps(previous, indent=2) + "\n").encode()).hexdigest()
        assert len(previous["selected_tests"]) == 85
        assert previous["selected_test_scope_sha256"] == "ebbe449bc05acf1dd040768a078c7c42b25e7bd4a02944517e998cb23c266ab4"
        additions = ["tests/test_mprobit.py", "tests/test_mprobit_postestimation.py",
                     "tests/test_mprobit_independent.py"]
        assert plan["authoritative_main_commit"] == "3a8c51d0f942409e4cd3a1808332679bcef23eb5"
        assert plan["main_added_selectors"] == additions
        historical_main = plan["previous_current_main_plan"]
        old_main = historical_main["value"]
        assert historical_main["source_commit"] == "bc3a846b4a4893074334ef66f75a6fb801df7de1"
        assert historical_main["sha256"] == "250dd4ac8e016bf571228ecf64bec24d4c49a8aa589093f0fe92272232d633c4"
        assert historical_main["sha256"] == hashlib.sha256(
            (json.dumps(old_main, indent=2) + "\n").encode()).hexdigest()
        assert old_main["selected_tests"] == additions + previous["selected_tests"]
        assert len(old_main["selected_tests"]) == 88
        incoming = plan["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == 89
        assert [item for item in historical_tests if item in incoming] == incoming
        assert [item for item in historical_tests if item in old_main["selected_tests"]] == old_main["selected_tests"]
        assert [item for item in historical_tests if item not in old_main["selected_tests"]] == plan["incoming_main_added_selectors"]
        assert [item for item in historical_tests if item not in incoming] == plan["prior_branch_added_selectors"]
        assert len(plan["incoming_main_added_selectors"]) == 10
        assert len(plan["prior_branch_added_selectors"]) == 9
        historical94 = plan["previous_current_survey_plan"]
        prior94 = historical94["value"]["selected_tests"]
        assert len(prior94) == 94
        assert historical94["sha256"] == hashlib.sha256(
            (json.dumps(historical94["value"], indent=2) + "\n").encode()).hexdigest()
        assert [item for item in historical_tests if item in prior94] == prior94
        assert len(plan["latest_main_added_selectors"]) == 4
        assert plan["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert plan["serial_steps_plan"]["default_timeout_seconds"] == 1800
        assert historical_tests == plan["selected_tests"]
        assert len(historical_tests) == len(set(historical_tests)) == 98
        assert plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(historical_tests, separators=(",", ":")).encode()).hexdigest()
        assert [len(group) for group in plan["effective_groups"]] == [1, 5, 92]
        distributed = plan["hosted_distributed_plan"]
        assert distributed["job_count"] == 4 and distributed["children_per_job"] == 2
        assert distributed["global_child_indices_by_shard"] == [[0, 1], [2, 3]]
        assert distributed["cohort_deadline_seconds_including_collection_queue_cleanup_and_aggregate_reassembly"] == 900
        current = gate.test_environment()
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert current[variable] == plan["environment"][variable] == "1"
            assert plan["historical_measurements_environment"][variable] == previous["environment"][variable] == "2"
        assert plan["timeout_seconds"] == previous["timeout_seconds"] == 900

        assert current_plan["authoritative_main_commit"] == "99d530614aff4fc6abbbb2b2b52d94751630c1cf"
        assert prior_tests == current_plan["selected_tests"]
        assert len(prior_tests) == len(set(prior_tests)) == 111
        incoming = current_plan["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == 102
        assert [item for item in prior_tests if item in incoming] == incoming
        assert [item for item in prior_tests if item in historical_tests] == historical_tests
        assert [item for item in prior_tests if item not in historical_tests] == current_plan["incoming_main_added_selectors"]
        assert [item for item in prior_tests if item not in incoming] == current_plan["prior_branch_added_selectors"]
        assert len(current_plan["incoming_main_added_selectors"]) == 13
        assert len(current_plan["prior_branch_added_selectors"]) == 9
        assert current_plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(prior_tests, separators=(",", ":")).encode()).hexdigest()
        assert [len(group) for group in current_plan["effective_groups"]] == [1, 5, 105]
        assert current_plan["hosted_legacy_plan"]["group_count"] == 3
        assert current_plan["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert current_plan["hosted_legacy_plan"]["historical_two_child_receipts_eligible"] is False
        distributed = current_plan["hosted_distributed_plan"]
        assert distributed["job_count"] == 8 and distributed["children_per_job"] == 1
        assert distributed["shard_count_per_version"] == 4
        assert distributed["global_child_indices_by_shard"] == [[0], [1], [2], [3]]
        assert distributed["cohort_deadline_seconds_including_collection_queue_cleanup_and_aggregate_reassembly"] == 900
        assert current_plan["environment"] == plan["environment"]
        assert current_plan["timeout_seconds"] == plan["timeout_seconds"] == 900


        assert registered_plan["source_parent_commit"] == "39f47872dcb46014d753c298c0d0cc4344748da4"
        assert registered_plan["authoritative_main_commit"] == "b6347f31e2cd38a069ccd3599a5d147dd9266fae"
        sequential = ["tests/test_sequential_kernels.py", "tests/test_sequential_oracles.py",
                      "tests/test_sequential_contracts.py"]
        assert b634_tests == registered_plan["selected_tests"] == sequential + prior_tests
        assert len(b634_tests) == len(set(b634_tests)) == 114
        incoming = registered_plan["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == len(set(incoming)) == 105 and set(incoming) <= set(b634_tests)
        qualification = registered_plan["selector_order_qualification"]
        moved = qualification["incoming_moved_to_end_conflicting_selectors"]
        assert len(moved) == 8 and qualification["both_complete_orders_compatible"] is False
        assert [item for item in b634_tests if item in prior_tests] == prior_tests
        assert [item for item in b634_tests if item in incoming and item not in moved] == [
            item for item in incoming if item not in moved]
        assert [item for item in b634_tests if item not in incoming] == current_plan["prior_branch_added_selectors"]
        assert registered_plan["selected_test_scope_sha256"] == "5d689c78ec0a7df70e8b6c2dfeb5bee77c46727185c78e86c5587ffa8ab1004c"
        assert registered_plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(b634_tests, separators=(",", ":")).encode()).hexdigest()
        assert [len(group) for group in registered_plan["effective_groups"]] == [1, 5, 108]
        assert registered_plan["fixed_groups"] == current_plan["fixed_groups"]
        assert registered_plan["environment"] == current_plan["environment"]
        assert registered_plan["timeout_seconds"] == current_plan["timeout_seconds"] == 900
        assert registered_plan["explicit_local_cpu_policy"]["threads_per_child"] == 1
        assert registered_plan["hosted_legacy_plan"]["group_count"] == 3
        assert registered_plan["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert registered_plan["hosted_legacy_plan"]["historical_two_child_receipts_eligible"] is False
        distributed = registered_plan["hosted_distributed_plan"]
        assert distributed["job_count"] == 8 and distributed["children_per_job"] == 1
        assert distributed["shard_count_per_version"] == 4
        assert distributed["global_child_indices_by_shard"] == [[0], [1], [2], [3]]
        assert distributed["cohort_deadline_seconds_including_collection_queue_cleanup_and_aggregate_reassembly"] == 900

        assert current_registered["authoritative_main_commit"] == "9e511cb32c828a2cc3293053fb0a5b2248da4d9e"
        assert current_registered["previous_current_b634_plan"]["sha256"] == hashlib.sha256(
            (json.dumps(registered_plan, indent=2) + "\n").encode()).hexdigest()
        assert registered_tests == current_registered["selected_tests"]
        assert len(registered_tests) == len(set(registered_tests)) == 119
        assert [item for item in registered_tests if item in b634_tests] == b634_tests
        incoming = current_registered["authoritative_incoming_selector_scope"]["selected_tests"]
        assert len(incoming) == 110 and set(incoming) <= set(registered_tests)
        assert [item for item in registered_tests if item not in b634_tests] == current_registered["latest_main_added_selectors"]
        assert [len(group) for group in current_registered["effective_groups"]] == [1, 5, 113]
        archived = current_registered["previous_current_5b_plan"]
        compressed = base64.b64decode(archived["compressed_archive"]["value"], validate=True)
        assert len(compressed) == archived["compressed_archive"]["bytes"]
        assert hashlib.sha256(compressed).hexdigest() == archived["compressed_archive"]["sha256"]
        archived_raw = gzip.decompress(compressed)
        frozen_raw = subprocess.check_output(
            ["git", "show", "5b9b5b04d62127654dd90d93811f54e7fab24117:docs/econometrics/merge-gate-sdk-groups-plan.json"],
            cwd=ROOT)
        assert archived_raw == frozen_raw
        assert len(archived_raw) == archived["bytes"] == 3793047
        assert hashlib.sha256(archived_raw).hexdigest() == archived["sha256"] == "ad5983269a77250c3d419c599d1acc15bcb0eb86a84c63666fe22da42649d517"
        frozen_plan = json.loads(archived_raw)
        assert frozen_plan["previous_current_b634_plan"] == current_registered["previous_current_b634_plan"]
        assert registered_tests[:-2] == frozen_plan["selected_tests"]
        assert registered_tests[-2:] == ["tests/test_verify_team_cleanup.py",
                                  "tests/test_verify_windows_cloud_ui_fixture.py"]
        assert current_registered["timeout_seconds"] == 900
        assert current_registered["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(registered_tests, separators=(",", ":")).encode()).hexdigest()

        assert history["source_commit"] == "9aa6fb18fb079b5f581009c73e8ef8ad1af59807"
        assert live_registered["source_parent_commit"] == history["source_commit"]
        assert live_registered["authoritative_main_commit"] == "3ad8b093eb8a7e45624bd79fd2b9623725a92898"
        assert live_registered["selected_tests"] == current_registered["selected_tests"] == registered_tests
        assert live_registered["selected_test_scope_sha256"] == current_registered["selected_test_scope_sha256"]
        assert live_registered["fixed_groups"][0] == current_registered["fixed_groups"][0]
        assert live_registered["fixed_groups"][1][:-1] == current_registered["fixed_groups"][1]
        assert live_registered["fixed_groups"][1][-1] == "tests/test_parallel_sdk_groups.py"
        assert [len(group) for group in live_registered["effective_groups"]] == [1, 6, 112]
        assert live_registered["environment"] == current_registered["environment"]
        assert live_registered["timeout_seconds"] == current_registered["timeout_seconds"] == 900
        for key in ("named_execution_budgets", "hosted_distributed_plan", "explicit_local_cpu_policy",
                    "previous_current_b634_plan", "previous_current_5b_plan", "measurements"):
            assert live_registered[key] == current_registered[key]
        assert live_registered["hosted_legacy_plan"]["groups"] == live_registered["effective_groups"]
        assert live_registered["hosted_legacy_plan"]["timeout_seconds"] == current_registered["hosted_legacy_plan"]["timeout_seconds"] == 1800
        assert live_registered["current_controller_balance"]["actual_acceptance_of_proposal"] is False

        incoming_history = combined_registered["previous_current_main_ba5_plan"]
        incoming_compressed = base64.b64decode(incoming_history["compressed_archive"]["value"], validate=True)
        assert len(incoming_compressed) == incoming_history["compressed_archive"]["bytes"]
        assert hashlib.sha256(incoming_compressed).hexdigest() == incoming_history["compressed_archive"]["sha256"]
        incoming_raw = gzip.decompress(incoming_compressed)
        assert incoming_raw == subprocess.check_output(["git", "show", incoming_history["source_commit"] + ":" + incoming_history["path"]], cwd=ROOT)
        assert len(incoming_raw) == incoming_history["bytes"]
        assert hashlib.sha256(incoming_raw).hexdigest() == incoming_history["sha256"]
        incoming_registered = json.loads(incoming_raw)
        assert balanced_history["source_commit"] == "262b7e9d8ac3d7e33ac123be866fbd48ca401912"
        assert incoming_history["source_commit"] == "ba5b7e5b147277e9d744e6a36b3a70ba6ac71a17"
        assert incoming_registered["authoritative_main_commit"] == "3ad8b093eb8a7e45624bd79fd2b9623725a92898"
        assert incoming_registered["selected_tests"][:119] == registered_tests
        assert incoming_registered["selected_tests"][119:] == incoming_registered["twostep_added_selectors"]
        assert len(incoming_registered["selected_tests"]) == len(set(incoming_registered["selected_tests"])) == 124
        assert [x for x in incoming_registered["selected_tests"] if x in b634_tests] == b634_tests
        assert [x for x in incoming_registered["selected_tests"] if x not in b634_tests] == incoming_registered["latest_main_added_selectors"]
        assert [len(group) for group in incoming_registered["effective_groups"]] == [1, 5, 118]
        assert incoming_registered["selected_test_scope_sha256"] == hashlib.sha256(json.dumps(incoming_registered["selected_tests"], separators=(",", ":")).encode()).hexdigest()
        assert incoming_registered["selected_tests"][:117] == frozen_plan["selected_tests"]
        assert incoming_registered["selected_tests"][117:119] == ["tests/test_verify_team_cleanup.py", "tests/test_verify_windows_cloud_ui_fixture.py"]
        assert incoming_registered["previous_current_b634_plan"] == current_registered["previous_current_b634_plan"]
        assert incoming_registered["previous_current_5b_plan"] == current_registered["previous_current_5b_plan"]
        assert combined_registered["source_parent_commit"] == balanced_history["source_commit"]
        assert combined_registered["authoritative_main_commit"] == incoming_history["source_commit"]
        assert combined_registered["selected_tests"] == incoming_registered["selected_tests"] == prior_gate_tests
        assert prior_gate_tests[:119] == live_registered["selected_tests"] == registered_tests
        assert prior_gate_tests[119:] == combined_registered["twostep_added_selectors"] == incoming_registered["twostep_added_selectors"]
        assert len(prior_gate_tests) == len(set(prior_gate_tests)) == 124
        assert combined_registered["selected_test_scope_sha256"] == incoming_registered["selected_test_scope_sha256"]
        assert combined_registered["fixed_groups"] == live_registered["fixed_groups"]
        assert [len(group) for group in combined_registered["effective_groups"]] == [1, 6, 117]
        assert combined_registered["effective_groups"][:2] == live_registered["effective_groups"][:2]
        assert combined_registered["effective_groups"][2][:-5] == live_registered["effective_groups"][2]
        assert combined_registered["effective_groups"][2][-5:] == incoming_registered["twostep_added_selectors"]
        assert combined_registered["timeout_seconds"] == incoming_registered["timeout_seconds"] == live_registered["timeout_seconds"] == 900
        for key in ("environment", "named_execution_budgets", "hosted_distributed_plan", "explicit_local_cpu_policy",
                    "previous_current_b634_plan", "previous_current_5b_plan", "measurements", "scientific_source_change",
                    "source_science_bridge_required", "scientific_source_change_scope", "current_controller_balance"):
            assert combined_registered[key] == live_registered[key]
        assert combined_registered["hosted_legacy_plan"]["groups"] == combined_registered["effective_groups"]
        assert combined_registered["hosted_legacy_plan"]["timeout_seconds"] == incoming_registered["hosted_legacy_plan"]["timeout_seconds"] == 1800
        integration = combined_registered["current_main_integration"]
        assert integration["incoming_main_scientific_sources_preserved_exactly"] is True
        assert integration["old_historical_proofs_are_not_current_base_acceptance"] is True
        assert integration["all_inherited_source_science_bridge_requirements_preserved"] is True
        assert integration["actual_acceptance_of_proposal"] is False
        added = ["tests/test_survey_deff.py", "tests/test_survey_inference.py"]
        assert extension["kind"] == "append-only-mandatory-survey-selector-extension"
        assert extension["added_selectors"] == added
        assert extension["prior_selected_test_count"] == 124
        assert extension["prior_selected_test_scope_sha256"] == combined_registered["selected_test_scope_sha256"]
        assert extension["selected_test_count"] == 126
        assert extension["ordered_prior_prefix_exact"] is True
        assert extension["new_gate_source_registration_only"] is True
        assert extension["historical_measurements_requalified_as_current_acceptance"] is False
        assert extension["scientific_evidence_source_commit"] == "b97a503224c126f03b8238d87a8670cf9e9d112b"
        assert extension["packaging_evidence_source_commit"] == "204de04b78dcb1f49a71757f174960d704b997fa"
        assert extension["scientific_evidence_source_is_distinct_from_future_transport_gate_source"] is True
        superseded = extension["superseded_prior121_registration"]
        superseded_raw = subprocess.check_output(
            ["git", "show", superseded["source_commit"] + ":" + superseded["plan_path"]], cwd=ROOT)
        assert len(superseded_raw) == superseded["bytes"]
        assert hashlib.sha256(superseded_raw).hexdigest() == superseded["sha256"]
        superseded_plan = json.loads(superseded_raw)
        assert superseded_plan["selected_tests"][:119] == combined_registered["selected_tests"][:119]
        assert superseded_plan["selected_tests"][119:] == added
        assert len(superseded_plan["selected_tests"]) == superseded["selected_test_count"] == 121
        assert superseded_plan["selected_test_scope_sha256"] == superseded["selected_test_scope_sha256"]
        assert superseded["current_hosted_acceptance_claimed"] is False
        previous126 = extension["superseded_prior126_registration"]
        previous126_raw = subprocess.check_output(
            ["git", "show", previous126["source_commit"] + ":" + previous126["plan_path"]], cwd=ROOT)
        assert previous126["source_commit"] == "e1dc6d3e1c2ef119d510fa91768b2fa7685159f0"
        assert len(previous126_raw) == previous126["bytes"]
        assert hashlib.sha256(previous126_raw).hexdigest() == previous126["sha256"]
        previous126_plan = json.loads(previous126_raw)
        assert previous126_plan["selected_tests"] == combined_registered["selected_tests"] + added
        assert len(previous126_plan["selected_tests"]) == previous126["selected_test_count"] == 126
        assert previous126_plan["selected_test_scope_sha256"] == previous126["selected_test_scope_sha256"]
        assert previous126_plan["survey_deff_scope_extension"]["superseded_prior121_registration"] == superseded
        assert previous126["current_hosted_acceptance_claimed"] is False
        assert scoped_tests == prospective_plan["selected_tests"] == combined_registered["selected_tests"] + added
        assert len(scoped_tests) == len(set(scoped_tests)) == 126
        assert scoped_tests[:124] == combined_registered["selected_tests"]
        assert scoped_tests[119:124] == combined_registered["twostep_added_selectors"]
        assert prospective_plan["effective_groups"][:-1] == combined_registered["effective_groups"][:-1]
        assert prospective_plan["effective_groups"][-1] == combined_registered["effective_groups"][-1] + added
        assert [len(group) for group in prospective_plan["effective_groups"]] == [1, 6, 119]
        assert prospective_plan["hosted_legacy_plan"]["groups"] == prospective_plan["effective_groups"]
        assert extension["incoming_legacy_inventory_reconciled_to_complete_mandatory_source"] is True
        for group, prior_group in zip(prospective_plan["hosted_legacy_plan"]["groups"],
                                      combined_registered["hosted_legacy_plan"]["groups"]):
            assert group[:len(prior_group)] == prior_group
            assert len(group) >= len(prior_group)
        assert {key: value for key, value in prospective_plan["hosted_legacy_plan"].items()
                if key != "groups"} == {key: value for key, value in combined_registered["hosted_legacy_plan"].items()
                                        if key != "groups"}
        assert prospective_plan["selected_test_scope_sha256"] == hashlib.sha256(
            json.dumps(scoped_tests, separators=(",", ":")).encode()).hexdigest()
        changed_fields = {"stage", "selected_tests", "selected_test_scope_sha256", "effective_groups",
                          "hosted_legacy_plan", "registered_at_utc", "decision", "acceptance_statement"}
        assert set(prospective_plan) == set(combined_registered) | {"survey_deff_scope_extension"}
        assert {key: value for key, value in prospective_plan.items()
                if key not in changed_fields | {"survey_deff_scope_extension"}} == {
                    key: value for key, value in combined_registered.items() if key not in changed_fields}

    prior136 = archived["previous_current_PR213_cdf620_full136_plan"]
    incoming126 = archived["previous_current_main830_full126_plan"]
    _check_complete_previous136(prior136, prior136["selected_tests"])
    _check_complete_incoming126(incoming126, incoming126["selected_tests"])
    owned12 = prior136["selected_tests"][124:]
    assert len(owned12) == len(set(owned12)) == 12
    assert incoming126["selected_tests"][:124] == prior136["selected_tests"][:124]
    assert incoming126["selected_tests"][124:] == ["tests/test_survey_deff.py", "tests/test_survey_inference.py"]
    assert newest["selected_tests"] == gate.TESTS[:138] == incoming126["selected_tests"] + owned12
    assert len(gate.TESTS[:146]) == len(set(gate.TESTS[:146])) == 146
    assert newest["selected_test_scope_sha256"] == hashlib.sha256(json.dumps(gate.TESTS[:138], separators=(",", ":")).encode()).hexdigest()
    assert newest["authoritative_main_commit"] == "83091ae7eda517c2954efb373f7e5d79e68dcc3f"
    assert newest["source_parent_commit"] == "cdf620a5dea7da61df34a70fa1525ac7dabb4094"
    assert newest["fixed_groups"] == prior136["fixed_groups"]
    assert newest["fixed_groups"][0] == incoming126["fixed_groups"][0]
    assert newest["fixed_groups"][1] == incoming126["fixed_groups"][1] + owned12
    assert newest["effective_groups"][:2] == [[item for item in gate.TESTS if item in group] for group in newest["fixed_groups"]]
    assert newest["effective_groups"][2] == incoming126["effective_groups"][2]
    assert [len(group) for group in newest["effective_groups"]] == [1, 18, 119]
    assert newest["timeout_seconds"] == 900
    assert newest["hosted_legacy_plan"]["groups"] == newest["effective_groups"]
    assert newest["hosted_legacy_plan"]["timeout_seconds"] == 1800
    for key in ("environment", "named_execution_budgets", "hosted_distributed_plan", "explicit_local_cpu_policy",
                "previous_current_b634_plan", "previous_current_5b_plan", "measurements", "scientific_source_change",
                "source_science_bridge_required", "scientific_source_change_scope", "current_controller_balance", "survey_deff_scope_extension"):
        assert newest[key] == incoming126[key]
    integration = newest["PR213_current_main_integration"]
    assert integration["complete138_order"] == gate.TESTS[:138]
    assert integration["all12_original_scientific_additions"] == owned12
    assert integration["full900_required"] is True
    assert integration["old_timings_are_not_capacity_proof"] is True
    assert integration["actual_acceptance_of_proposal"] is False
    assert integration["all_previous136_integration_requirements_preserved_complete"] == prior136["PR213_current_main_integration"]
    for key in ("scientific_source_change", "source_science_bridge_required", "scientific_source_change_scope"):
        assert integration["all_previous136_top_level_science_requirements_preserved"][key] == prior136[key]
    assert newest["previous_current_main769_full124_plan"] == prior136["previous_current_main769_full124_plan"]
    assert newest["previous_current_PR213_2a_full131_plan"] == prior136["previous_current_PR213_2a_full131_plan"]




@pytest.mark.parametrize("shard", [0, 1])
def test_full_local_mode_cannot_claim_a_distributed_ci_shard(tmp_path, shard):
    args = _gate_args(tmp_path)
    args.local_sdk_groups = True
    args.timeout = 900
    args.ci_shard_index = shard
    with pytest.raises(ValueError, match="Local SDK groups cannot be a distributed CI shard"):
        gate.run(args)
    assert not args.directory.exists()


def test_nonzero_exit_and_timeout_are_failures(tmp_path):
    result = gate.run_step("bad", [sys.executable, "-c", "raise SystemExit(9)"], tmp_path, 5)
    assert result["status"] == "failed" and result["exit_code"] == 9
    result = gate.run_step("timeout", [sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, .1)
    assert result["status"] == "cancelled_or_timeout" and result["exit_code"] != 0


def test_stale_pr_head_rejected_before_running_code(monkeypatch):
    monkeypatch.setattr(gate, "command_json", lambda cmd: {"state": "open", "head": {"sha": "new"},
        "base": {"ref": "main"}} if 'pulls' in cmd[-1] else {"sha": "base"})
    with pytest.raises(ValueError, match="current open PR head"):
        gate.require_current_pr("owner/repo", 1, "old")


def test_inherited_selectors_and_plugins_cannot_shrink_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_one")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_inherited_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "77")
        assert gate.test_environment()[key] == "1"
    tests = tmp_path / "test_scope.py"
    tests.write_text("import os\ndef test_one():\n"
                     "    assert all(os.environ[name] == '1' for name in "
                     "('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))\n"
                     "def test_two(): pass\n")
    xml = tmp_path / "scope.xml"
    result = gate.run_step("scope", [sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
                           "--junitxml", str(xml), str(tests)], tmp_path, 30,
                           env=gate.test_environment())
    assert result["status"] == "passed"
    assert gate.junit_counts(xml)["tests"] == 2


def test_dirty_local_candidate_is_rejected_even_without_publishing(monkeypatch, tmp_path):
    def reject():
        raise ValueError("dirty checkout")
    monkeypatch.setattr(gate, "require_clean", reject)
    with pytest.raises(ValueError, match="dirty checkout"):
        gate.run(SimpleNamespace(directory=tmp_path / "receipt", publish_repo=None))


def _gate_args(tmp_path, component=None):
    return SimpleNamespace(directory=tmp_path / "receipt", publish_repo=None, pr=None,
                           timeout=1800, ci_sdk_version=component,
                           python=[tmp_path / "python311", tmp_path / "python313"])


def _distributed_args(tmp_path, component, shard):
    args = _gate_args(tmp_path, component)
    args.ci_shard_index = shard
    args.timeout = 900
    return args


def _mock_gate_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "require_clean", lambda: None)
    monkeypatch.setattr(gate, "identity", lambda: "a" * 40)
    monkeypatch.setattr(gate.subprocess, "check_output", lambda cmd, **kwargs:
                        "3.11\n" if cmd[0].endswith("python311") else "3.13\n")
    monkeypatch.setattr(gate, "junit_counts", lambda path: dict(tests=3, failures=0, errors=0, skipped=0))
    monkeypatch.setattr(gate, "package_manifest", lambda directory: [])

    def step(name, command, directory, timeout, *, env=None):
        calls.append((name, command, timeout, env))
        if name.startswith("sdk-"):
            Path(command[command.index("--junitxml") + 1]).write_text("mock junit bytes")
            Path(command[command.index("--gate-timings") + 1]).write_text("mock phase bytes")
            if "--directory" in command:
                groups = Path(command[command.index("--directory") + 1])
                groups.mkdir()
                fixed = [('tests/test_control_stream_acceptance.py',), ('tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_function_stream_state.py', 'tests/test_nested_logit_independent.py', 'tests/test_parallel_sdk_groups.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py')]
                members = {item for group in fixed for item in group}
                partition = [[item for item in gate.TESTS if item in group] for group in fixed]
                partition.append([item for item in gate.TESTS if item not in members])
                manifest = {"status": "passed", "seconds": .01, "timeout_seconds": timeout,
                            "tests": dict(tests=3, failures=0, errors=0, skipped=0),
                            "groups": [dict(index=index, selectors=selected, status="passed", exit_code=0,
                                            tests=dict(tests=1, failures=0, errors=0, skipped=0))
                                       for index, selected in enumerate(partition)]}
                if "--local" in command:
                    manifest.update(execution=None, execution_sha256=None, component_sdk_version=None,
                                    source_commit="a" * 40, selected_tests=gate.TESTS,
                                    checkout_root=str(gate.ROOT), python=command[0], timeout_seconds=900)
                if "--shard-index" in command:
                    manifest.update(schema=2, kind="distributed_sdk_shard",
                                   shard_index=int(command[command.index("--shard-index") + 1]),
                                   shard_count=2, group_count=4, partial_scope=True,
                                   source_commit="a" * 40, selected_tests=gate.TESTS,
                                   component_sdk_version=command[command.index("--component-sdk-version") + 1])
                (groups / "report.json").write_text(json.dumps(manifest))
        return dict(name=name, command=command, status="passed", exit_code=0,
                    seconds=.01, log=name + ".log", log_sha256="0" * 64)

    monkeypatch.setattr(gate, "run_step", step)
    return calls



@pytest.mark.parametrize("component", [None, "3.11", "3.13"])
def test_ci_component_preserves_complete_selected_suite_and_all_common_checks(monkeypatch, tmp_path, component):
    calls = _mock_gate_execution(monkeypatch)
    args = _gate_args(tmp_path, component)
    assert gate.run(args) == 0
    report = json.loads((args.directory / "report.json").read_text())
    versions = [component] if component else ["3.11", "3.13"]
    assert [v[0] for v in calls] == ["ruff", "capabilities", "editor", "web-install",
                                    "web-tests", "web-build", *("sdk-" + v for v in versions), "packages"]
    assert all(v[2] == 1800 for v in calls)
    for name, command, _, env in calls:
        if name.startswith("sdk-"):
            if component:
                assert command[1] == "scripts/run_parallel_sdk_groups.py"
                assert command[command.index("--component-sdk-version") + 1] == component
                assert command[command.index("--timeout") + 1] == "1800"
                assert command[command.index("--execution") + 1] == str(args.directory.parent / "execution.json")
            else:
                assert command[command.index("--junitxml") + 2:] == gate.TESTS
                assert command[command.index("-p") + 1] == "scripts.pytest_gate_timings"
            assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert report["python_versions_provisioned"] == ["3.11", "3.13"]
    assert report["sdk_versions_planned"] == versions
    assert report["sdk_versions_executed"] == versions
    assert report["selected_tests"] == gate.TESTS
    assert report["component_mode"] is (component is not None)
    assert report["component_sdk_version"] == component
    assert report["status_context"] == (None if component else gate.CONTEXT)
    for step in report["steps"]:
        if step["name"].startswith("sdk-"):
            version = step["name"][4:]
            assert step["junit"] == f"pytest-{version}.xml"
            assert step["phase_timings"] == f"pytest-{version}-timings.jsonl"
            assert len(step["junit_sha256"]) == len(step["phase_timings_sha256"]) == 64
            if component:
                assert step["sdk_groups"] == f"sdk-{version}-groups/report.json"
                assert len(step["sdk_groups_sha256"]) == 64
    if component:
        assert "Partial CI component" in report["scope"] and "requires the other" in report["scope"]
    else:
        assert "Python 3.11/3.13" in report["scope"]


def test_local_groups_keep_full_dual_python_scope_and_all_common_checks(monkeypatch, tmp_path):
    calls = _mock_gate_execution(monkeypatch)
    args = _gate_args(tmp_path)
    args.local_sdk_groups = True
    args.timeout = 900
    assert gate.run(args) == 0
    report = json.loads((args.directory / "report.json").read_text())
    assert [item[0] for item in calls] == ["ruff", "capabilities", "editor", "web-install",
                                          "web-tests", "web-build", "sdk-3.11", "sdk-3.13", "packages"]
    assert report["component_mode"] is False and report["component_sdk_version"] is None
    assert report["local_sdk_groups"] is True and report["status_context"] == gate.CONTEXT
    assert report["sdk_versions_planned"] == ["3.11", "3.13"]
    assert report["sdk_versions_executed"] == ["3.11", "3.13"]
    assert report["selected_tests"] == gate.TESTS
    for name, command, timeout, _ in calls:
        assert timeout == 900
        if name.startswith("sdk-"):
            assert command[1] == "scripts/run_parallel_sdk_groups.py" and command[-1] == "--local"
            assert "--execution" not in command and "--component-sdk-version" not in command
            assert command[command.index("--timeout") + 1] == "900"
            step = next(item for item in report["steps"] if item["name"] == name)
            assert len(step["sdk_groups_sha256"]) == 64


@pytest.mark.parametrize("failed_step", ["ruff", "sdk-3.11"])
def test_failed_step_receipt_retains_only_actual_sdk_versions(monkeypatch, tmp_path, failed_step):
    calls = _mock_gate_execution(monkeypatch)
    actual = gate.run_step

    def fail(name, command, directory, timeout, **kwargs):
        result = actual(name, command, directory, timeout, **kwargs)
        if name == failed_step:
            result.update(status="failed", exit_code=9)
        return result

    monkeypatch.setattr(gate, "run_step", fail)
    args = _gate_args(tmp_path)
    args.local_sdk_groups = True
    args.timeout = 900
    assert gate.run(args) == 1
    report = json.loads((args.directory / "report.json").read_text())
    expected = [] if failed_step == "ruff" else ["3.11"]
    assert report["status"] == "failed" and f"{failed_step} failed" in report["error"]
    assert report["python_versions_provisioned"] == ["3.11", "3.13"]
    assert report["sdk_versions_planned"] == ["3.11", "3.13"]
    assert report["sdk_versions_executed"] == expected
    assert [name[4:] for name, *_ in calls if name.startswith("sdk-")] == expected
    assert report["steps"][-1]["name"] == failed_step
    assert report["steps"][-1]["status"] == "failed" and report["steps"][-1]["exit_code"] == 9
    assert not any(item["name"] in ("sdk-3.13", "packages") for item in report["steps"])
    assert not (args.directory / "pytest-3.13.xml").exists()
    if failed_step == "ruff":
        assert not (args.directory / "pytest-3.11.xml").exists()


@pytest.mark.parametrize("timeout", [899, 901, 1800])
def test_local_groups_cannot_change_deadline(tmp_path, timeout):
    args = _gate_args(tmp_path)
    args.local_sdk_groups, args.timeout = True, timeout
    with pytest.raises(ValueError, match="900-second"):
        gate.run(args)
    assert not args.directory.exists()


def test_local_groups_cannot_be_partial_ci_before_any_work(tmp_path):
    args = _gate_args(tmp_path, "3.11")
    args.local_sdk_groups = True
    args.timeout = 900
    with pytest.raises(ValueError, match="cannot be a partial CI"):
        gate.run(args)
    assert not args.directory.exists()


def test_local_groups_still_require_both_validated_interpreters(monkeypatch, tmp_path):
    calls = _mock_gate_execution(monkeypatch)
    args = _gate_args(tmp_path)
    args.local_sdk_groups, args.python = True, args.python[:1]
    args.timeout = 900
    assert gate.run(args) == 1 and calls == []
    report = json.loads((args.directory / "report.json").read_text())
    assert "requires exactly Python versions" in report["error"]


@pytest.mark.parametrize('mode', ['local', '3.11'])
@pytest.mark.parametrize('attack', [
    'missing_third', 'failed_third', 'skipped_third', 'duplicate_index',
    'boolean_index', 'missing_third_scope', 'duplicate_third_scope',
    'third_count', 'combined_count',
])
def test_complete_grouped_dispatch_requires_all_three_children_in_both_modes(
    monkeypatch, tmp_path, mode, attack
):
    _mock_gate_execution(monkeypatch)
    actual = gate.run_step

    def altered(name, command, directory, timeout, **kwargs):
        result = actual(name, command, directory, timeout, **kwargs)
        if name.startswith('sdk-'):
            path = Path(command[command.index('--directory') + 1]) / 'report.json'
            manifest = json.loads(path.read_text())
            child = manifest['groups'][2]
            if attack == 'missing_third':
                manifest['groups'].pop()
            elif attack == 'failed_third':
                child.update(status='failed', exit_code=1)
            elif attack == 'skipped_third':
                child['tests']['skipped'] = 1
            elif attack == 'duplicate_index':
                child['index'] = 1
            elif attack == 'boolean_index':
                manifest['groups'][1]['index'] = True
            elif attack == 'missing_third_scope':
                child['selectors'].pop()
            elif attack == 'duplicate_third_scope':
                child['selectors'] = manifest['groups'][0]['selectors']
            elif attack == 'third_count':
                child['tests']['tests'] += 1
            else:
                manifest['tests']['tests'] += 1
            path.write_text(json.dumps(manifest))
        return result

    monkeypatch.setattr(gate, 'run_step', altered)
    args = _gate_args(tmp_path, None if mode == 'local' else mode)
    args.local_sdk_groups = mode == 'local'
    args.timeout = 900 if mode == 'local' else 1800
    assert gate.run(args) == 1
    report = json.loads((args.directory / 'report.json').read_text())
    expected = ('all three children' if attack in ('missing_third', 'duplicate_index', 'boolean_index')
                else 'passed complete source scope' if attack in ('failed_third', 'skipped_third')
                else 'complete original scope or JUnit counts')
    assert report['status'] == 'failed' and expected in report['error']
    assert not any(step['name'] == 'packages' for step in report['steps'])


@pytest.mark.parametrize("change", [
    {"execution": {}}, {"execution_sha256": "0" * 64}, {"component_sdk_version": "3.11"},
    {"source_commit": "b" * 40}, {"selected_tests": ["tests/test_one.py"]},
    {"checkout_root": "/another/source"}, {"python": "/another/python"}, {"timeout_seconds": 901},
    {"timeout_seconds": 1800},
])
def test_local_group_receipt_requires_exact_source_scope_and_null_ci_binding(monkeypatch, tmp_path, change):
    _mock_gate_execution(monkeypatch)
    actual = gate.run_step

    def altered(name, command, directory, timeout, **kwargs):
        result = actual(name, command, directory, timeout, **kwargs)
        if name.startswith("sdk-"):
            path = Path(command[command.index("--directory") + 1]) / "report.json"
            manifest = json.loads(path.read_text())
            manifest.update(change)
            path.write_text(json.dumps(manifest))
        return result

    monkeypatch.setattr(gate, "run_step", altered)
    args = _gate_args(tmp_path)
    args.local_sdk_groups = True
    args.timeout = 900
    assert gate.run(args) == 1
    report = json.loads((args.directory / "report.json").read_text())
    expected = "original deadline" if "timeout_seconds" in change else "Local SDK source/scope binding"
    assert expected in report["error"]
    assert report["sdk_versions_planned"] == ["3.11", "3.13"]
    assert report["sdk_versions_executed"] == ["3.11"]
    assert not any(item["name"] == "packages" for item in report["steps"])


def test_gate_cli_rejects_local_and_partial_ci_together(tmp_path):
    directory = tmp_path / "receipt"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_merge_candidate.py"),
                             "--python", sys.executable, "--directory", str(directory),
                             "--local-sdk-groups", "--ci-sdk-version", "3.11"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "not allowed with argument" in result.stderr
    assert not directory.exists()


def test_partial_ci_component_rejects_publication_before_any_work(monkeypatch, tmp_path):
    args = _gate_args(tmp_path, "3.11")
    args.publish_repo = "owner/repo"
    monkeypatch.setattr(gate, "publish", lambda *args: pytest.fail("partial publication attempted"))
    with pytest.raises(ValueError, match="cannot publish"):
        gate.run(args)
    assert not args.directory.exists()


@pytest.mark.parametrize("timeout", [899, 901, 1799, 1801])
def test_partial_ci_component_cannot_change_deadline(tmp_path, timeout):
    args = _gate_args(tmp_path, "3.13")
    args.timeout = timeout
    with pytest.raises(ValueError, match="1800-second"):
        gate.run(args)
    assert not args.directory.exists()


def test_partial_component_still_requires_both_validated_interpreters(monkeypatch, tmp_path):
    calls = _mock_gate_execution(monkeypatch)
    args = _gate_args(tmp_path, "3.11")
    args.python = args.python[:1]
    assert gate.run(args) == 1
    assert calls == []
    report = json.loads((args.directory / "report.json").read_text())
    assert report["status"] == "failed" and report["status_context"] is None
    assert "requires exactly Python versions" in report["error"]


def test_grouped_step_parent_clock_is_fresh_and_inside_original_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SDK_PARENT_STARTED_MONOTONIC", "-1")
    monkeypatch.setenv("OPENECON_SDK_PARENT_DEADLINE_MONOTONIC", "9999999999")
    script = tmp_path / "run_parallel_sdk_groups.py"
    script.write_text("import json,os,time\nprint(json.dumps({'started':float(os.environ['OPENECON_SDK_PARENT_STARTED_MONOTONIC']),'deadline':float(os.environ['OPENECON_SDK_PARENT_DEADLINE_MONOTONIC']),'now':time.monotonic()}))\n")
    result = gate.run_step("group-clock", [sys.executable, str(script)], tmp_path, 5)
    assert result["status"] == "passed" and result["seconds"] <= 5
    clock = json.loads((tmp_path / "group-clock.log").read_text())
    assert clock["started"] <= clock["now"] < clock["deadline"]
    assert clock["deadline"] == clock["started"] + 5



@pytest.mark.parametrize("seconds,status", [(901, "passed"), (1801, "passed"), (1, "failed"),
                                           (float("nan"), "passed"), (True, "passed"),
                                           (-1, "passed"), (None, "passed")])
def test_failed_or_overbudget_group_receipt_is_not_accepted(monkeypatch, tmp_path, seconds, status):
    _mock_gate_execution(monkeypatch)
    original = gate.run_step

    def alter(name, command, directory, timeout, *, env=None):
        result = original(name, command, directory, timeout, env=env)
        if name.startswith("sdk-"):
            group = Path(command[command.index("--directory") + 1]) / "report.json"
            group.write_text(json.dumps({"status": status, "seconds": seconds}))
        return result

    monkeypatch.setattr(gate, "run_step", alter)
    args = _gate_args(tmp_path, "3.13")
    if seconds == 901:
        args.ci_shard_index = 0
        args.timeout = 900
    assert gate.run(args) == 1
    report = json.loads((args.directory / "report.json").read_text())
    assert report["status"] == "failed" and "original deadline" in report["error"]
    assert report["steps"][-1]["sdk_groups"] == "sdk-3.13-groups/report.json"
    assert len(report["steps"][-1]["sdk_groups_sha256"]) == 64


def test_package_manifest_requires_actual_wheels_and_sdists_and_hashes_bytes(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in ("one.whl", "two.whl", "one.tar.gz"):
        (dist / name).write_bytes(b"actual package bytes")
    with pytest.raises(ValueError, match="actual wheel and sdist"):
        gate.package_manifest(tmp_path)
    (dist / "two.tar.gz").write_bytes(b"other package bytes")
    before = gate.package_manifest(tmp_path)
    assert len(before) == 4 and all(v["bytes"] > 0 for v in before)
    (dist / "one.whl").write_bytes(b"changed package bytes")
    after = gate.package_manifest(tmp_path)
    assert {v["path"]: v["sha256"] for v in before}["dist/one.whl"] != {
        v["path"]: v["sha256"] for v in after}["dist/one.whl"]


def _four_distribution_files(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in ("one.whl", "two.whl", "one.tar.gz", "two.tar.gz"):
        (dist / name).write_bytes(b"test distribution bytes")
    return dist


def test_package_manifest_accepts_only_the_observed_uv_marker(tmp_path):
    dist = _four_distribution_files(tmp_path)
    before = gate.package_manifest(tmp_path)
    (dist / ".gitignore").write_bytes(b"*")
    assert gate.package_manifest(tmp_path) == before
    assert len(before) == 4 and all(v["path"] != "dist/.gitignore" for v in before)


@pytest.mark.parametrize("attack", ["empty_marker", "wrong_byte", "extra_newline", "directory_marker",
                                    "symlink_marker", "other_extra", "missing_package", "empty_package",
                                    "symlink_package", "directory_package"])
def test_package_manifest_does_not_hide_changed_markers_or_invalid_outputs(tmp_path, attack):
    dist = _four_distribution_files(tmp_path)
    marker = dist / ".gitignore"
    marker.write_bytes(b"*")
    if attack == "empty_marker":
        marker.write_bytes(b"")
    elif attack == "wrong_byte":
        marker.write_bytes(b"x")
    elif attack == "extra_newline":
        marker.write_bytes(b"*\n")
    elif attack == "directory_marker":
        marker.unlink()
        marker.mkdir()
    elif attack == "symlink_marker":
        outside = tmp_path / "outside-marker"
        outside.write_bytes(b"*")
        marker.unlink()
        marker.symlink_to(outside)
    elif attack == "other_extra":
        (dist / ".DS_Store").write_bytes(b"extra")
    elif attack == "missing_package":
        (dist / "one.whl").unlink()
    elif attack == "empty_package":
        (dist / "one.whl").write_bytes(b"")
    elif attack == "symlink_package":
        (dist / "one.whl").unlink()
        (dist / "one.whl").symlink_to(dist / "two.whl")
    elif attack == "directory_package":
        (dist / "one.whl").unlink()
        (dist / "one.whl").mkdir()
    with pytest.raises(ValueError):
        gate.package_manifest(tmp_path)


def test_component_cli_refuses_local_publication_before_receipt_creation(tmp_path):
    directory = tmp_path / "not-created"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_merge_candidate.py"),
                             "--python", sys.executable, "--directory", str(directory),
                             "--ci-sdk-version", "3.11", "--publish-repo", "owner/repo"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Partial CI components cannot publish" in result.stderr
    assert not directory.exists()


def test_daily_parallel_workflow_keeps_all_eight_jobs_and_full_aggregate():
    # Each supported minor owns four real VM jobs, eight jobs in total.
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    component, aggregate = workflow.split("\n  merge-gate:\n")
    assert "name: OpenEconometrics / SDK Python ${{ matrix.sdk }} shard ${{ matrix.shard }}" in component
    assert "runs-on: ubuntu-latest" in component
    assert "fail-fast: false" in component and "sdk: ['3.11', '3.13']" in component
    assert "shard: [0, 1, 2, 3]" in component
    for version in ("3.11", "3.13"):
        assert f"uv sync --frozen --python {version} --extra cloud --extra desktop" in component
    assert "--python .venv311/bin/python --python .venv313/bin/python" in component
    assert "--ci-sdk-version '${{ matrix.sdk }}'" in component
    assert "--ci-shard-count 4" in component
    assert "--ci-shard-index '${{ matrix.shard }}'" in component
    assert "merge-gate-${{ github.run_id }}-${{ github.run_attempt }}-python-${{ matrix.sdk }}-shard-${{ matrix.shard }}" in component
    assert "node-version: '24'" in component and "Chrome with its sandbox enabled" in component
    assert "name: OpenEconometrics / daily full gate" in aggregate
    assert "needs: sdk-components" in aggregate and "if: always()" in aggregate
    assert "fetch-depth: 0" in component
    assert "fetch-depth: 2" in aggregate
    assert "sparse-checkout-cone-mode: false" in aggregate
    for source_path in ("/scripts/", "/src/", "/packages/openecon-charts/",
                        "/pyproject.toml", "/uv.lock", "/.github/workflows/ci.yml"):
        assert f"            {source_path}\n" in aggregate
    assert "scripts/verify_parallel_merge_gate.py" in aggregate
    assert "--shard-count 4" in aggregate
    assert "--github-jobs artifacts/ci/github-jobs.json" in aggregate
    assert "actions: read" in workflow
    assert aggregate.count("uses: actions/download-artifact@v4") == 1
    assert "pattern: merge-gate-${{ github.run_id }}-${{ github.run_attempt }}-python-{3.11,3.13}-shard-{0,1,2,3}" in aggregate
    assert "merge-multiple: false" in aggregate
    for version in ("3.11", "3.13"):
        for shard in range(4):
            assert f'--component "artifacts/ci/components/merge-gate-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-python-{version}-shard-{shard}"' in aggregate
            assert f'--job-result "{version}:{shard}=$SDK_COMPONENT_RESULT"' in aggregate
    assert "SDK_COMPONENT_RESULT: ${{ needs.sdk-components.result }}" in aggregate
    for expected in ("source", "head", "base", "event", "run-id", "run-attempt"):
        assert f"--expected-{expected}" in aggregate
    assert "--publish-repo" not in workflow and "--timeout" not in workflow
    assert gate.TESTS.count("tests/test_parallel_merge_gate.py") == 1


def test_sdk_upload_excludes_only_unused_pytest_scratch_and_keeps_complete_proof():
    import fnmatch

    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    component = workflow.split("\n  merge-gate:\n", 1)[0]
    upload = component.split("      - name: Retain gate evidence\n", 1)[1]
    path_block = upload.split("          path: |\n", 1)[1].split(
        "          if-no-files-found:", 1)[0]
    patterns = [line.strip() for line in path_block.splitlines() if line.strip()]
    assert patterns == ["artifacts/ci/", "!artifacts/ci/**/pytest-temp/**"]
    exclusion = patterns[1].removeprefix("!")

    producer_spec = importlib.util.spec_from_file_location(
        "sdk_transport_scope", ROOT / "scripts/run_parallel_sdk_groups.py")
    producer = importlib.util.module_from_spec(producer_spec)
    producer_spec.loader.exec_module(producer)
    protected = ["artifacts/ci/execution.json", "artifacts/ci/receipt/report.json"]
    protected += [f"artifacts/ci/receipt/{step}.log" for step in (
        "ruff", "capabilities", "editor", "web-install", "web-tests", "web-build", "packages")]
    protected += ["artifacts/ci/receipt/dist/" + name for name in (
        "openecon-py3-none-any.whl", "openecon.tar.gz",
        "openecon_charts-py3-none-any.whl", "openecon_charts.tar.gz")]
    for version in ("3.11", "3.13"):
        protected += [f"artifacts/ci/receipt/pytest-{version}{suffix}" for suffix in (
            ".xml", "-timings.jsonl")]
        protected.append(f"artifacts/ci/receipt/sdk-{version}.log")
        base = Path(f"artifacts/ci/receipt/sdk-{version}-groups")
        protected.append(str(base / "report.json"))
        for group in range(4):
            directory = base / f"group-{group}"
            command = producer.child_command(sys.executable, ROOT, directory,
                                             gate.TESTS, group_index=group)
            scratch = Path(command[command.index("--basetemp") + 1])
            assert scratch == directory / "pytest-temp"
            for name in ("test_actual_build0/dist/sdk.whl", "source/src/example.py",
                         "pytest-current/fixture-state.json", "nested/deep/output.tar.gz"):
                assert fnmatch.fnmatchcase(str(scratch / name), exclusion)
            protected += [str(directory / name) for name in (
                "pytest.log", "pytest.xml", "pytest-timings.jsonl", "pytest-collection.jsonl",
                "pytest-full-collection.jsonl", "partition-plan.json",
                "pytest-temp-notes.log", "pytest-temp.json")]
    assert all(not fnmatch.fnmatchcase(path, exclusion) for path in protected)


@pytest.mark.parametrize("component,shard", [(None, 0), (None, 1), ("3.13", -1), ("3.13", 2),
                                            ("3.13", True), ("3.13", 0.0), ("3.11", "0")])
def test_invalid_or_unscoped_ci_shard_is_refused_before_work(tmp_path, component, shard):
    args = _gate_args(tmp_path, component)
    args.ci_shard_index = shard
    with pytest.raises(ValueError):
        gate.run(args)
    assert not args.directory.exists()


@pytest.mark.parametrize("component,shard", [("3.11", 0), ("3.11", 1), ("3.13", 0), ("3.13", 1)])
def test_every_distributed_shard_keeps_all_common_checks_and_cannot_be_required_gate(monkeypatch, tmp_path, component, shard):
    calls = _mock_gate_execution(monkeypatch)
    args = _distributed_args(tmp_path, component, shard)
    assert gate.run(args) == 0
    report = json.loads((args.directory / "report.json").read_text())
    assert report["component_mode"] is True and report["status_context"] is None
    assert report["component_sdk_version"] == component and report["component_sdk_shard"] == shard
    assert report["python_versions_provisioned"] == ["3.11", "3.13"]
    assert report["selected_tests"] == gate.TESTS and len(gate.TESTS) == 160
    assert gate.TESTS[117:119] == ["tests/test_verify_team_cleanup.py",
                              "tests/test_verify_windows_cloud_ui_fixture.py"]
    assert gate.TESTS[119:124] == ["tests/test_twostep_kernel.py", "tests/test_twostep_reference.py",
                                  "tests/test_twostep_contract.py", "tests/test_twostep_scale.py",
                                  "tests/test_twostep_adaptive_reference.py"]
    assert gate.TESTS[124:126] == ["tests/test_survey_deff.py", "tests/test_survey_inference.py"]
    assert [call[0] for call in calls] == ["ruff", "capabilities", "editor", "web-install", "web-tests",
                                         "web-build", "sdk-" + component, "packages"]
    sdk = next(call for call in calls if call[0].startswith("sdk-"))
    command = sdk[1]
    assert command[command.index("--shard-index") + 1] == str(shard)
    assert command[command.index("--shard-count") + 1] == "2"
    assert sdk[2] == 900
    assert report["timeout_seconds"] == 900
    assert command[command.index("--timeout") + 1] == "900"
    assert "Partial" in report["scope"] and "requires" in report["scope"]
    assert report["status_context"] != gate.CONTEXT

    assert gate.TESTS[126:138] == ['tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py']



@pytest.mark.parametrize("shard", [0, 1])
def test_partial_sdk_shard_cannot_publish_success_or_pending_status(monkeypatch, tmp_path, shard):
    args = _gate_args(tmp_path, "3.13")
    args.ci_shard_index = shard
    args.publish_repo = "owner/repo"
    monkeypatch.setattr(gate, "publish", lambda *args: pytest.fail("partial shard published"))
    with pytest.raises(ValueError, match="cannot publish"):
        gate.run(args)
    assert not args.directory.exists()


@pytest.mark.parametrize("field,value", [
    ("shard_index", 1), ("shard_index", False), ("shard_count", 2.0), ("group_count", 3),
    ("partial_scope", False), ("source_commit", "b" * 40),
    ("component_sdk_version", "3.11"), ("selected_tests", ["tests/test_merge_gate.py"]),
    ("seconds", 900.000001), ("timeout_seconds", 1800),
])
def test_rehashed_shard_manifest_cannot_mix_identity_scope_or_capacity(monkeypatch, tmp_path, field, value):
    _mock_gate_execution(monkeypatch)
    original = gate.run_step

    def replace_manifest(name, command, directory, timeout, *, env=None):
        result = original(name, command, directory, timeout, env=env)
        if name.startswith("sdk-"):
            path = Path(command[command.index("--directory") + 1]) / "report.json"
            record = json.loads(path.read_text())
            record[field] = value
            path.write_text(json.dumps(record))
        return result

    monkeypatch.setattr(gate, "run_step", replace_manifest)
    args = _distributed_args(tmp_path, "3.13", 0)
    assert gate.run(args) == 1
    report = json.loads((args.directory / "report.json").read_text())
    assert report["status"] == "failed" and report["status_context"] is None
    step = next(item for item in report["steps"] if item["name"] == "sdk-3.13")
    assert len(step["sdk_groups_sha256"]) == 64


@pytest.mark.parametrize("shard,budget", [(None, 1800), (0, 900), (1, 900)])
def test_sdk_mode_default_allocation_is_explicit_and_keeps_original_deadline(monkeypatch, tmp_path, shard, budget):
    calls = _mock_gate_execution(monkeypatch)
    args = _gate_args(tmp_path, "3.13")
    args.ci_shard_index = shard
    args.timeout = None
    assert gate.run(args) == 0
    assert all(call[2] == budget for call in calls)
    report = json.loads((args.directory / "report.json").read_text())
    assert report["timeout_seconds"] == budget
    command = next(call[1] for call in calls if call[0].startswith("sdk-"))
    assert command[command.index("--timeout") + 1] == str(budget)


@pytest.mark.parametrize("shard,budget", [(None, 900), (0, 1800), (1, 1800), (0, 899), (1, 901)])
def test_sdk_modes_cannot_borrow_or_change_other_mode_deadline(tmp_path, shard, budget):
    args = _gate_args(tmp_path, "3.13")
    args.ci_shard_index = shard
    args.timeout = budget
    with pytest.raises(ValueError, match="step deadline"):
        gate.run(args)
    assert not args.directory.exists()


def test_distributed_parent_utc_and_pid_are_fresh_even_with_inherited_poison(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SDK_PARENT_STARTED_UTC", "-1")
    monkeypatch.setenv("OPENECON_SDK_PARENT_PID", "-1")
    script = tmp_path / "run_parallel_sdk_groups.py"
    script.write_text("import json,os,time\nprint(json.dumps({'utc':float(os.environ['OPENECON_SDK_PARENT_STARTED_UTC']),'pid':int(os.environ['OPENECON_SDK_PARENT_PID']),'monotonic':float(os.environ['OPENECON_SDK_PARENT_STARTED_MONOTONIC'])}))\n")
    before = time.time()
    result = gate.run_step("distributed-clock", [sys.executable, str(script), "--shard-index", "0"], tmp_path, 5)
    after = time.time()
    assert result["status"] == "passed"
    child = json.loads((tmp_path / "distributed-clock.log").read_text())
    clock = result["sdk_clock"]
    assert before <= child["utc"] == clock["started_utc"] <= clock["finished_utc"] <= after
    assert child["pid"] == clock["clock_identity"]["pid"] > 0
    assert child["monotonic"] == clock["monotonic_started"]
    assert abs(clock["finished_utc"] - clock["started_utc"] - clock["elapsed_seconds"]) <= 1
    assert gate.TESTS.count("tests/test_parallel_sdk_groups.py") == 1
    assert gate.TESTS.count("tests/test_pytest_gate_timings.py") == 1


def test_timing_diagnostic_preserves_pass_fail_skip_and_setup_error(tmp_path):
    tests = tmp_path / "test_outcomes.py"
    tests.write_text('''import pytest
@pytest.fixture
def broken():
    raise RuntimeError("PRIVATE_SETUP_PAYLOAD")
def test_pass():
    pass
def test_fail():
    assert False, "PRIVATE_FAILURE_PAYLOAD"
@pytest.mark.skip(reason="PRIVATE_SKIP_PAYLOAD")
def test_skip():
    pass
def test_setup_error(broken):
    pass
''')
    outcomes = []
    for instrumented in (False, True):
        xml = tmp_path / f"outcomes-{instrumented}.xml"
        timings = tmp_path / "timings.jsonl"
        command = [sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
                   "--junitxml", str(xml)]
        if instrumented:
            command += ["-p", "scripts.pytest_gate_timings", "--gate-timings", str(timings)]
        result = gate.run_step(f"outcomes-{instrumented}", [*command, str(tests)], tmp_path, 30,
                               env=gate.test_environment())
        assert result["exit_code"] == 1 and result["status"] == "failed"
        tree = ET.parse(xml).getroot()
        suite = next(tree.iter("testsuite"))
        assert {key: int(suite.get(key)) for key in ("tests", "failures", "errors", "skipped")} == {
            "tests": 4, "failures": 1, "errors": 1, "skipped": 1}
        outcomes.append([(case.get("name"), tuple(child.tag for child in case))
                         for case in tree.iter("testcase")])
    assert outcomes[0] == outcomes[1]
    payload = timings.read_text()
    assert "PRIVATE_" not in payload
    records = [json.loads(line) for line in payload.splitlines()]
    assert {(item["nodeid"].split("::")[-1], item["phase"]): item["outcome"] for item in records} == {
        ("test_pass", "setup"): "passed", ("test_pass", "call"): "passed",
        ("test_pass", "teardown"): "passed", ("test_fail", "setup"): "passed",
        ("test_fail", "call"): "failed", ("test_fail", "teardown"): "passed",
        ("test_skip", "setup"): "skipped", ("test_skip", "teardown"): "passed",
        ("test_setup_error", "setup"): "failed", ("test_setup_error", "teardown"): "passed"}
    for item in records:
        assert set(item) == {"nodeid", "phase", "outcome", "duration", "start", "stop"}
        assert all(math.isfinite(item[key]) for key in ("duration", "start", "stop"))
        assert item["duration"] >= 0 and item["stop"] >= item["start"]


def test_timing_diagnostic_is_flushed_before_unfinished_process_exits(tmp_path):
    tests = tmp_path / "test_unfinished.py"
    tests.write_text("import time\ndef test_complete(): pass\ndef test_unfinished(): time.sleep(20)\n")
    timings = tmp_path / "unfinished.jsonl"
    process = subprocess.Popen([sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
                                "-p", "scripts.pytest_gate_timings", "--gate-timings", str(timings),
                                str(tests)], cwd=ROOT, env=gate.test_environment(),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        records = []
        while time.monotonic() < deadline and process.poll() is None:
            if timings.exists():
                records = [json.loads(line) for line in timings.read_text().splitlines()]
                if any(item["nodeid"].endswith("::test_unfinished") and item["phase"] == "setup"
                       for item in records):
                    break
            time.sleep(.05)
        assert process.poll() is None
        assert any(item["nodeid"].endswith("::test_unfinished") and item["phase"] == "setup"
                   for item in records)
        assert any(item["nodeid"].endswith("::test_complete") and item["phase"] == "call"
                   and item["outcome"] == "passed" for item in records)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    assert process.returncode != 0
    assert [json.loads(line) for line in timings.read_text().splitlines()] == records
    assert not any(item["nodeid"].endswith("::test_unfinished") and item["phase"] == "call"
                   for item in records)


def test_complete_local_default_cannot_borrow_incoming_legacy_budget(monkeypatch, tmp_path):
    calls = _mock_gate_execution(monkeypatch)
    args = _gate_args(tmp_path)
    args.local_sdk_groups = True
    args.timeout = None
    assert gate.run(args) == 0
    report = json.loads((args.directory / "report.json").read_text())
    assert report["local_sdk_groups"] is True and report["timeout_seconds"] == 900
    assert all(call[2] == 900 for call in calls)
    assert all("--local" in call[1] and "--execution" not in call[1]
               for call in calls if call[0].startswith("sdk-"))


def test_confidence_sequence_extension_keeps_full_current_main_scope_and_history():
    extension = json.loads((ROOT / "docs/econometrics/confidence-sequences-gate-extension-2026-10-10.json").read_text())
    raw = subprocess.check_output(["git", "show", extension["base_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    assert extension["base_commit"] == "35f347fd175aba76a367220b4d6ba571376887c1"
    assert hashlib.sha256(raw).hexdigest() == extension["base_producer_sha256"]
    prior = ast.literal_eval(next(node.value for node in ast.parse(raw).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    assert len(prior) == extension["base_selector_count"] == 144
    assert extension["added_selectors"] == ["tests/test_confidence_sequences.py"]
    assert gate.TESTS[:145] == prior + extension["added_selectors"]
    assert len(gate.TESTS[:145]) == len(set(gate.TESTS[:145])) == extension["complete_selector_count"] == 145
    for selectors, key in ((prior, "base_selector_scope_sha256"), (gate.TESTS[:145], "complete_selector_scope_sha256")):
        assert hashlib.sha256(json.dumps(selectors, separators=(",", ":")).encode()).hexdigest() == extension[key]
    plan_path = "docs/econometrics/merge-gate-sdk-groups-plan.json"
    unchanged = subprocess.check_output(["git", "show", extension["base_commit"] + ":" + plan_path], cwd=ROOT)
    assert (ROOT / plan_path).read_bytes() == unchanged
    assert hashlib.sha256(unchanged).hexdigest() == extension["unchanged_complete_historical_plan_sha256"]
    assert extension["timeout_seconds"] == json.loads(unchanged)["timeout_seconds"] == 900
    assert extension["legacy_timeout_seconds"] == json.loads(unchanged)["hosted_legacy_plan"]["timeout_seconds"] == 1800


def test_current_BVAR151_plan_preserves_complete146_prefix_original_history_and_900_bound():
    historical_selectors = gate.TESTS[:151]
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan-bvar151-2026-10-10.json").read_text())
    def literal(source, name):
        return ast.literal_eval(next(node.value for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)))
    before = subprocess.check_output(["git", "show", plan["actual_source_parent_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    previous = literal(before, "TESTS")
    added = ["tests/test_bayesian_var_conjugate.py", "tests/test_bayesian_var_sbc_protocol.py", "tests/test_bayesian_var_public_integration.py", "tests/test_bayesian_var_public_admission_v2.py", "tests/test_editor_catalog_intern_v2.py"]
    assert len(previous) == len(set(previous)) == 146
    assert historical_selectors[:146] == plan["complete_previous146_selectors"] == previous
    assert historical_selectors[146:] == plan["appended_selectors"] == added
    assert historical_selectors == plan["selected_tests"] and len(historical_selectors) == len(set(historical_selectors)) == 151
    for values, key in ((previous, "complete_previous146_scope_sha256"), (historical_selectors, "selected_test_scope_sha256")):
        assert hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest() == plan[key]
    sdk = literal((ROOT / "scripts/run_parallel_sdk_groups.py").read_text(), "FIXED_GROUPS")
    aggregate = literal((ROOT / "scripts/verify_parallel_merge_gate.py").read_text(), "SDK_FIXED_GROUPS")
    prior_fixed = literal(subprocess.check_output(["git", "show", plan["actual_source_parent_commit"] + ":scripts/run_parallel_sdk_groups.py"], cwd=ROOT), "FIXED_GROUPS")
    assert sdk == aggregate == (prior_fixed[0], prior_fixed[1] + tuple(added))
    assert plan["fixed_groups"] == [list(group) for group in sdk]
    fixed = set(sum(sdk, ()))
    groups = [[item for item in historical_selectors if item in group] for group in sdk]
    groups.append([item for item in historical_selectors if item not in fixed])
    assert groups == plan["effective_groups"] and list(map(len, groups)) == [1, 23, 127]
    assert groups[1][:-5] == plan["complete_previous146_effective_groups"][1]
    assert groups[1][-5:] == added and groups[2] == plan["complete_previous146_effective_groups"][2]
    assert len(sum(groups, [])) == len(set(sum(groups, []))) == len(historical_selectors)
    assert set(sum(groups, [])) == set(historical_selectors)
    history = plan["whole_original_historical138_plan"]
    raw = (ROOT / history["path"]).read_bytes()
    assert raw == subprocess.check_output(["git", "show", history["source_commit"] + ":" + history["path"]], cwd=ROOT)
    assert len(raw) == history["bytes"] and hashlib.sha256(raw).hexdigest() == history["sha256"]
    old = json.loads(raw)
    for key in ("environment", "hosted_distributed_plan", "named_execution_budgets", "explicit_local_cpu_policy"):
        assert plan[key] == old[key]
    assert old["selected_tests"] == historical_selectors[:138]
    assert plan["timeout_seconds"] == old["timeout_seconds"] == 900
    assert plan["sdk_minors"] == ["3.11", "3.13"] and plan["group_count"] == 3
    assert plan["hosted_distributed_plan"]["shard_count_per_version"] == 4
    for relative, pin in plan["complete_added_source_pins"].items():
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]
    assert plan["acceptance"] is False


def _complete_protocol2_fixture_module():
    fixture_spec = importlib.util.spec_from_file_location(
        "synthetic_complete_report_fixture", ROOT / "tests/gate_report_protocol_fixture.py")
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    return fixture


def test_explicit_protocol2_step_validates_122_parents_147_native_units_and391_events(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, nodes = fixture.make(tmp_path / "step")
    result = {}
    actual = gate.validated_protocol2_step(xml, [sys.executable, "--gate-timings", str(raw)],
                                          None, result, fixture.SELECTORS)
    assert actual == {"tests": 122, "failures": 0, "errors": 0, "skipped": 0}
    assert result["phase_report_protocol"] == 2
    assert result["report_counts"]["native_JUnit_reported_tests"] == 147
    assert result["report_counts"]["all_phase_records"] == 391
    assert result["report_counts"]["collected_cases"] == len(nodes)


def test_explicit_protocol2_step_refuses_missing_source_terminal_instead_of_subtracting25(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, _ = fixture.make(tmp_path / "step-refused")
    fixture.reports.paths(raw)[1].unlink()
    with pytest.raises((ValueError, OSError)):
        gate.validated_protocol2_step(xml, [sys.executable, "--gate-timings", str(raw)],
                                     None, {}, fixture.SELECTORS)


def test_current_BMA153_scope_retains_full151_prefix_two_complete_modules_and_900():
    historical_selectors = gate.TESTS[:153]
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan-bma153-2026-10-10.json").read_text())
    before = subprocess.check_output(["git", "show", plan["actual_source_parent_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    old = ast.literal_eval(next(node.value for node in ast.parse(before).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    assert len(old) == len(set(old)) == 151
    assert historical_selectors[:151] == plan["complete_previous151_selectors"] == old
    assert historical_selectors[151:] == plan["appended_selectors"] == ["tests/test_bayesian_bma_oracles.py", "tests/test_bayesian_bma_state.py"]
    assert historical_selectors == plan["selected_tests"] and len(historical_selectors) == len(set(historical_selectors)) == 153
    assert hashlib.sha256(json.dumps(historical_selectors, separators=(",", ":")).encode()).hexdigest() == plan["selected_test_scope_sha256"]
    fixed = plan["fixed_groups"]
    sdk_source = (ROOT / "scripts/run_parallel_sdk_groups.py").read_text()
    validator_source = (ROOT / "scripts/verify_parallel_merge_gate.py").read_text()
    def literal(source, name):
        return ast.literal_eval(next(node.value for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)))
    assert literal(sdk_source, "FIXED_GROUPS") == literal(validator_source, "SDK_FIXED_GROUPS") == tuple(tuple(group) for group in fixed)
    prior_sdk = subprocess.check_output(["git", "show", plan["actual_source_parent_commit"] + ":scripts/run_parallel_sdk_groups.py"], cwd=ROOT)
    assert literal(prior_sdk, "FIXED_GROUPS") == tuple(tuple(group) for group in fixed)
    partition = [[item for item in historical_selectors if item in group] for group in fixed]
    partition.append([item for item in historical_selectors if item not in set(sum(fixed, []))])
    assert partition == plan["effective_groups"] and list(map(len, partition)) == [1, 23, 129]
    assert len(sum(partition, [])) == len(set(sum(partition, []))) == len(historical_selectors)
    assert set(sum(partition, [])) == set(historical_selectors)
    historical = plan["whole_original_BVAR151_plan"]
    raw = (ROOT / historical["path"]).read_bytes()
    assert raw == subprocess.check_output(["git", "show", plan["actual_source_parent_commit"] + ":" + historical["path"]], cwd=ROOT)
    assert len(raw) == historical["bytes"] and hashlib.sha256(raw).hexdigest() == historical["sha256"]
    oldplan = json.loads(raw)
    for key in ("environment", "hosted_distributed_plan", "named_execution_budgets", "explicit_local_cpu_policy"):
        assert plan[key] == oldplan[key]
    assert plan["timeout_seconds"] == oldplan["timeout_seconds"] == 900
    assert plan["phase_report_protocol"] == gate.PHASE_REPORT_PROTOCOL == 2
    assert plan["acceptance"] is False
    for name, pin in plan["complete_added_BMA_source_pins"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]


def test_conjoint_extension_preserves_complete153_prefix_and_adds_both_scientific_files():
    historical_selectors = gate.TESTS[:155]
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-conjoint155-2026-10-10.json").read_text())
    previous_source = subprocess.check_output(["git", "show", plan["source_parent_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    previous = ast.literal_eval(next(node.value for node in ast.parse(previous_source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    assert previous == plan["previous_selectors"] and len(previous) == 153
    assert plan["appended_selectors"] == ["tests/test_econ_conjoint.py", "tests/test_econ_conjoint_robust.py"]
    assert historical_selectors == previous + plan["appended_selectors"] == plan["complete_selectors"]
    assert len(historical_selectors) == len(set(historical_selectors)) == 155
    for selectors, key in [(previous, "previous_scope_sha256"), (historical_selectors, "complete_scope_sha256")]:
        assert hashlib.sha256(json.dumps(selectors, separators=(",", ":")).encode()).hexdigest() == plan[key]
    assert plan["timeout_seconds"] == 900 and plan["fixed_groups_unchanged"]
    for path, constant in [("scripts/run_parallel_sdk_groups.py", "FIXED_GROUPS"),
                           ("scripts/verify_parallel_merge_gate.py", "SDK_FIXED_GROUPS")]:
        before = subprocess.check_output(["git", "show", plan["source_parent_commit"] + ":" + path], cwd=ROOT)
        def literal(source):
            return ast.literal_eval(next(node.value for node in ast.parse(source).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == constant for target in node.targets)))
        assert literal(before) == literal((ROOT / path).read_text())


def test_conjoint_bootstrap_extension_preserves_every_current159_selector_and_fixed_group():
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-conjoint-bootstrap160-2026-10-10.json").read_text())
    before = subprocess.check_output(["git", "show", plan["source_parent_commit"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT)
    def literal(source, name):
        return ast.literal_eval(next(node.value for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)))
    old = literal(before, "TESTS")
    assert old == plan["previous_selectors"] and len(old) == len(set(old)) == 159
    assert gate.TESTS[:159] == old
    assert gate.TESTS == old + ["tests/test_econ_conjoint_bootstrap.py"] == plan["complete_selectors"]
    assert len(gate.TESTS) == len(set(gate.TESTS)) == 160
    for selectors, key in ((old, "previous_scope_sha256"), (gate.TESTS, "complete_scope_sha256")):
        assert hashlib.sha256(json.dumps(selectors, separators=(",", ":")).encode()).hexdigest() == plan[key]
    assert plan["timeout_seconds"] == 900 and plan["fixed_groups_unchanged"]
    for path, constant in (("scripts/run_parallel_sdk_groups.py", "FIXED_GROUPS"),
                           ("scripts/verify_parallel_merge_gate.py", "SDK_FIXED_GROUPS")):
        old_source = subprocess.check_output(["git", "show", plan["source_parent_commit"] + ":" + path], cwd=ROOT)
        assert literal(old_source, constant) == literal((ROOT / path).read_text(), constant)
