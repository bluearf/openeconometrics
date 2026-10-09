"""Acceptance evidence must fail closed for empty, skipped and stale executions."""
import importlib.util
import json
import math
from pathlib import Path
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
                grouped = {"status": "passed", "seconds": .01, "timeout_seconds": timeout}
                if "--shard-index" in command:
                    grouped.update(schema=2, kind="distributed_sdk_shard",
                                   shard_index=int(command[command.index("--shard-index") + 1]),
                                   shard_count=2, group_count=4, partial_scope=True,
                                   source_commit="a" * 40, selected_tests=gate.TESTS,
                                   component_sdk_version=command[command.index("--component-sdk-version") + 1])
                (groups / "report.json").write_text(json.dumps(grouped))
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


def test_partial_ci_component_rejects_publication_before_any_work(monkeypatch, tmp_path):
    args = _gate_args(tmp_path, "3.11")
    args.publish_repo = "owner/repo"
    monkeypatch.setattr(gate, "publish", lambda *args: pytest.fail("partial publication attempted"))
    with pytest.raises(ValueError, match="cannot publish"):
        gate.run(args)
    assert not args.directory.exists()


@pytest.mark.parametrize("timeout", [1799, 1801])
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
    assert clock["deadline"] - clock["started"] == 5


@pytest.mark.parametrize("seconds,status", [(1801, "passed"), (1, "failed"),
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


def test_parallel_workflow_keeps_two_real_jobs_and_required_aggregate():
    # Each supported minor now owns two real VM jobs, four jobs in total.
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    component, aggregate = workflow.split("\n  merge-gate:\n")
    assert "name: OpenEconometrics / SDK Python ${{ matrix.sdk }} shard ${{ matrix.shard }}" in component
    assert "runs-on: ubuntu-latest" in component
    assert "fail-fast: false" in component and "sdk: ['3.11', '3.13']" in component
    assert "shard: [0, 1]" in component
    for version in ("3.11", "3.13"):
        assert f"uv sync --frozen --python {version} --extra cloud --extra desktop" in component
    assert "--python .venv311/bin/python --python .venv313/bin/python" in component
    assert "--ci-sdk-version '${{ matrix.sdk }}'" in component
    assert "--ci-shard-index '${{ matrix.shard }}'" in component
    assert "merge-gate-${{ github.run_id }}-${{ github.run_attempt }}-python-${{ matrix.sdk }}-shard-${{ matrix.shard }}" in component
    assert "node-version: '24'" in component and "Chrome with its sandbox enabled" in component
    assert "name: OpenEconometrics / merge gate" in aggregate
    assert "needs: sdk-components" in aggregate and "if: always()" in aggregate
    assert "fetch-depth: 2" in aggregate
    assert "sparse-checkout-cone-mode: false" in aggregate
    for source_path in ("/scripts/", "/src/", "/packages/openecon-charts/",
                        "/pyproject.toml", "/uv.lock", "/.github/workflows/ci.yml"):
        assert f"            {source_path}\n" in aggregate
    assert "scripts/verify_parallel_merge_gate.py" in aggregate
    assert "--shard-count 2" in aggregate
    assert "--github-jobs artifacts/ci/github-jobs.json" in aggregate
    assert "actions: read" in workflow
    assert aggregate.count("uses: actions/download-artifact@v4") == 1
    assert "pattern: merge-gate-${{ github.run_id }}-${{ github.run_attempt }}-python-{3.11,3.13}-shard-{0,1}" in aggregate
    assert "merge-multiple: false" in aggregate
    for version in ("3.11", "3.13"):
        for shard in (0, 1):
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
    assert report["selected_tests"] == gate.TESTS and len(gate.TESTS) == 102
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
