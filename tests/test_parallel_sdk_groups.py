"""Real subprocess and tamper checks for complete two-group SDK execution."""

import importlib.util
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sdk_groups", ROOT / "scripts/run_parallel_sdk_groups.py"
)
groups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groups)
gate_spec = importlib.util.spec_from_file_location(
    "sdk_groups_producer", ROOT / "scripts/verify_merge_candidate.py"
)
gate = importlib.util.module_from_spec(gate_spec)
gate_spec.loader.exec_module(gate)
# These small real-subprocess fixtures retain their declared toy partition.
# Production scope/order is checked separately against the actual gate source.
TOY_HEAVY = ("tests/test_streaming_control_function_engine.py",
             "tests/test_control_stream_acceptance.py")
SELECTORS = ["tests/test_before.py", *TOY_HEAVY, "tests/test_after.py"]


def repository(path, bodies=None):
    (path / "tests").mkdir(parents=True)
    (path / "scripts").mkdir()
    shutil.copyfile(
        ROOT / "scripts/pytest_gate_timings.py", path / "scripts/pytest_gate_timings.py"
    )
    shutil.copyfile(
        ROOT / "scripts/run_parallel_sdk_groups.py", path / "scripts/run_parallel_sdk_groups.py"
    )
    (path / "scripts/verify_merge_candidate.py").write_text("TESTS = " + repr(SELECTORS) + "\n")
    (path / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts="-q"\n')
    (path / ".gitignore").write_text("receipts/\n.pytest_cache/\n__pycache__/\n")
    body = """import os
import pytest
@pytest.mark.parametrize("value", [3, 1, 2])
def test_values(value):
    assert value in [3, 1, 2]
    assert os.environ["OMP_NUM_THREADS"] in ("1", "2")
    assert all(os.environ[key] == os.environ["OMP_NUM_THREADS"] for key in (
        "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS"))
class TestExact:
    def test_class(self):
        assert True
"""
    for selector in SELECTORS:
        (path / selector).write_text((bodies or {}).get(selector, body))
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "config", "user.name", "Test Fixture"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "isolated grouping fixture"],
    ):
        subprocess.run(
            command, cwd=path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return path


def execute(root, *, timeout=30, receipt="receipts/run", **kwargs):
    base = root / receipt
    base.mkdir(parents=True)
    report = groups.run_groups(
        root=root,
        python=Path(sys.executable),
        selectors=SELECTORS,
        directory=base / "sdk-groups",
        junit=base / "combined.xml",
        timings=base / "combined.jsonl",
        timeout=timeout,
        expensive=TOY_HEAVY,
        **kwargs,
    )
    return base, report


@pytest.fixture(scope="module")
def actual_pass(tmp_path_factory):
    root = repository(tmp_path_factory.mktemp("sdk-real"))
    base, report = execute(root)
    assert report["status"] == "passed", report
    return root, base, report


def copied_children(actual_pass, tmp_path):
    _, base, _ = actual_pass
    children = [tmp_path / f"group-{index}" for index in range(2)]
    for index, target in enumerate(children):
        shutil.copytree(base / "sdk-groups" / f"group-{index}", target)
    return children


def test_real_parallel_pass_has_raw_hashes_and_exact_source_scope(actual_pass):
    root, base, report = actual_pass
    assert report["selected_tests"] == SELECTORS
    assert report["source_commit"] == groups.source_identity(root)["source_commit"]
    assert report["source_tree"] == groups.source_identity(root)["source_tree"]
    assert report["seconds"] <= 30 and report["tests"] == {
        "tests": 16,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert [item["selectors"] for item in report["groups"]] == groups.split_groups(SELECTORS, TOY_HEAVY)
    for item in report["groups"]:
        assert item["status"] == "passed" and item["exit_code"] == 0 and item["pid"] > 0
        assert report["environment"] == groups.child_environment_contract(
            report["cpu_budget"]["threads_per_child"])
        temporary = base / "sdk-groups" / f"group-{item['index']}/runtime-temp"
        assert item["environment"] == {**report["environment"], **{
            key: str(temporary) for key in groups.TEMPORARY_ENVIRONMENT_KEYS}}
        record = groups.child_temporary_receipt(temporary.parent / "runtime-environment.json",
                                                temporary, item["pid"], removed=True)
        assert item["temporary_directory_removed_after_stop"] is True
        assert not temporary.exists()
        assert record["tempfile_directory"] == str(temporary)
        assert groups.digest(temporary.parent / "runtime-environment.json") == item["runtime_environment_sha256"]
        assert 0 <= item["start_seconds"] <= item["stop_seconds"] <= report["seconds"]
        for key in ("log", "junit", "phase_timings", "collection"):
            assert groups.digest(base / "sdk-groups" / item[key]) == item[key + "_sha256"]
        assert item["collected_tests"] == item["tests"]["tests"]
    assert groups.digest(base / "combined.xml") == report["combined"]["junit_sha256"]
    assert groups.digest(base / "combined.jsonl") == report["combined"]["phase_timings_sha256"]


def cpu_hierarchy(tmp_path, *, version=2, member="/tenant/job", mount_root="/",
                  quota="max", period="100000", parent_quota="max"):
    proc, mount = tmp_path / "proc", tmp_path / "cpu"
    proc.mkdir()
    mount.mkdir()
    relative = Path(member).relative_to(mount_root)
    leaf = mount / relative
    leaf.mkdir(parents=True, exist_ok=True)
    controller = "" if version == 2 else "cpu,cpuacct"
    (proc / "cgroup").write_text(f"0:{controller}:{member}\n")
    kind = "cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,cpu,cpuacct"
    (proc / "mountinfo").write_text(f"1 0 0:1 {mount_root} {mount} rw - {kind}\n")
    for current, value in ((leaf, quota), (mount, parent_quota)):
        if version == 2:
            (current / "cpu.max").write_text(f"{value} {period}\n")
        else:
            (current / "cpu.cfs_quota_us").write_text("-1" if value == "max" else value)
            (current / "cpu.cfs_period_us").write_text(period)
    return proc, mount, leaf


@pytest.mark.parametrize("quota,effective,threads", [
    ("50000", 1.0, 1), ("150000", 1.5, 1), ("200000", 2.0, 1),
    ("390000", 3.9, 1), ("400000", 4.0, 2), ("800000", 8.0, 2),
])
@pytest.mark.parametrize("version", [1, 2])
def test_current_cgroup_quota_and_fractional_budget_set_two_child_threads(
        tmp_path, monkeypatch, quota, effective, threads, version):
    proc, _, leaf = cpu_hierarchy(tmp_path, version=version, quota=quota)
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["effective_cpus"] == effective and budget["threads_per_child"] == threads
    assert budget["groups"] == 2 and budget["cgroup"]["version"] == version
    assert budget["max_concurrent_children"] == min(2, max(1, math.floor(effective)))
    assert budget["cgroup"]["limits"][0]["cpus"] == int(quota) / 100000
    assert Path(budget["cgroup"]["limits"][0]["path"]).parent == leaf
    name = "cpu.max" if version == 2 else "cpu.cfs_quota_us"
    assert budget["cgroup"]["probes"] == [
        {"view": 0, "path": str(current / name), "status": status}
        for current, status in ((leaf, "present"), (leaf.parent, "missing"),
                                (leaf.parent.parent, "present"))]
    env = groups.environment(ROOT, {key: "99" for key in groups.NUMERICAL_THREAD_KEYS},
                             threads=threads)
    assert {env[key] for key in groups.NUMERICAL_THREAD_KEYS} == {str(threads)}


@pytest.mark.parametrize("version", [1, 2])
def test_ancestor_quota_is_not_lost_when_current_group_is_unlimited(tmp_path, monkeypatch, version):
    proc, mount, _ = cpu_hierarchy(tmp_path, version=version, mount_root="/tenant",
                                  parent_quota="200000")
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["effective_cpus"] == 2 and budget["threads_per_child"] == 1
    assert budget["cgroup"]["status"] == "limited"
    assert budget["cgroup"]["views"] == [{"membership": "/tenant/job", "mount_root": "/tenant",
                                          "mountpoint": str(mount)}]
    assert [item["cpus"] for item in budget["cgroup"]["limits"]] == [None, 2]
    assert [item["status"] for item in budget["cgroup"]["probes"]] == ["present", "present"]


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("subtree_first", [True, False])
def test_all_mount_views_preserve_a_visible_tighter_nonroot_ancestor(
        tmp_path, monkeypatch, version, subtree_first):
    proc, mount, _ = cpu_hierarchy(tmp_path, version=version, member="/tenant/work")
    subtree = tmp_path / "subtree"
    subtree.mkdir()
    if version == 2:
        (subtree / "cpu.max").write_text("max 100000\n")
        (mount / "tenant/cpu.max").write_text("200000 100000\n")
    else:
        (subtree / "cpu.cfs_quota_us").write_text("-1\n")
        (subtree / "cpu.cfs_period_us").write_text("100000\n")
        (mount / "tenant/cpu.cfs_quota_us").write_text("200000\n")
        (mount / "tenant/cpu.cfs_period_us").write_text("100000\n")
    kind = "cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,cpu,cpuacct"
    extra = f"2 0 0:2 /tenant/work {subtree} rw - {kind}\n"
    original = (proc / "mountinfo").read_text()
    (proc / "mountinfo").write_text(extra + original if subtree_first else original + extra)
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["effective_cpus"] == 2 and budget["threads_per_child"] == 1
    assert len(budget["cgroup"]["views"]) == 2
    assert any(item["cpus"] == 2 for item in budget["cgroup"]["limits"])
    assert len(budget["cgroup"]["probes"]) == 4
    assert all(item["status"] == "present" for item in budget["cgroup"]["probes"])


@pytest.mark.parametrize("affinity,threads", [(2, 1), (4, 2)])
def test_affinity_is_a_limit_even_with_more_host_cpus_and_unlimited_quota(
        tmp_path, monkeypatch, affinity, threads):
    proc, _, _ = cpu_hierarchy(tmp_path)
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(affinity)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["host_cpus"] == 64 and budget["affinity_cpus"] == affinity
    assert budget["cgroup"]["status"] == "unlimited"
    assert budget["effective_cpus"] == affinity and budget["threads_per_child"] == threads


@pytest.mark.parametrize("quota,period", [
    ("oops", "100000"), ("0", "100000"), ("-2", "100000"),
    ("200000", "0"), ("200000", "bad"), ("max extra", "100000"), ("9" * 400, "1"),
])
def test_malformed_quota_is_recorded_and_conservatively_allocated(tmp_path, monkeypatch, quota, period):
    proc, _, leaf = cpu_hierarchy(tmp_path, quota=quota, period=period)
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["effective_cpus"] == budget["threads_per_child"] == 1
    assert budget["cgroup"]["status"] == "malformed" and budget["cgroup"]["error"]
    assert budget["cgroup"]["probes"] == [
        {"view": 0, "path": str(leaf / "cpu.max"), "status": "present"}]
    assert budget["cgroup"]["limits"] == []


@pytest.mark.parametrize("version", [1, 2])
def test_unreadable_ancestor_preserves_probes_before_read_and_uses_one_thread(
        tmp_path, monkeypatch, version):
    proc, mount, _ = cpu_hierarchy(tmp_path, version=version, quota="400000")
    name = "cpu.max" if version == 2 else "cpu.cfs_quota_us"
    original = Path.read_text

    def unavailable(path, *args, **kwargs):
        if path == mount / name:
            raise PermissionError("fixture quota is unreadable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unavailable)
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["effective_cpus"] == budget["threads_per_child"] == 1
    assert budget["cgroup"]["status"] == "unavailable" and budget["cgroup"]["error"]
    assert [item["status"] for item in budget["cgroup"]["probes"]] == [
        "present", "missing", "present"]
    assert [item["cpus"] for item in budget["cgroup"]["limits"]] == [4]


def test_unavailable_quota_and_cpu_probes_use_one_thread_without_changing_parent_env(tmp_path, monkeypatch):
    monkeypatch.setattr(groups.os, "cpu_count", lambda: None)
    monkeypatch.delattr(groups.os, "sched_getaffinity", raising=False)
    before = dict(os.environ)
    budget = groups.detect_cpu_budget(proc=tmp_path / "absent", system="Linux")
    assert budget["host_cpus"] is budget["affinity_cpus"] is None
    assert budget["effective_cpus"] == budget["threads_per_child"] == 1
    assert budget["cgroup"]["status"] == "unavailable" and budget["cgroup"]["error"]
    assert budget["cgroup"]["probes"] == []
    assert dict(os.environ) == before


def test_failed_host_and_affinity_probes_are_conservative(monkeypatch):
    def unavailable(*args):
        raise OSError("unavailable probe")
    monkeypatch.setattr(groups.os, "cpu_count", unavailable)
    monkeypatch.setattr(groups.os, "sched_getaffinity", unavailable, raising=False)
    budget = groups.detect_cpu_budget(system="Darwin")
    assert budget["host_cpus"] is budget["affinity_cpus"] is None
    assert budget["effective_cpus"] == budget["threads_per_child"] == 1
    assert budget["cgroup"]["probes"] == []


def test_missing_v2_cpu_controller_can_use_the_actual_v1_cpu_mount(tmp_path, monkeypatch):
    proc, mount, _ = cpu_hierarchy(tmp_path, version=1, quota="200000")
    unified = tmp_path / "unified"
    unified.mkdir()
    with (proc / "cgroup").open("a") as stream:
        stream.write("0::/\n")
    with (proc / "mountinfo").open("a") as stream:
        stream.write(f"2 0 0:2 / {unified} rw - cgroup2 cgroup rw\n")
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["cgroup"]["version"] == 1 and budget["cgroup"]["views"][0]["mountpoint"] == str(mount)
    assert budget["effective_cpus"] == 2 and budget["threads_per_child"] == 1


def test_host_count_limits_an_unavailable_affinity_probe(tmp_path, monkeypatch):
    proc, _, _ = cpu_hierarchy(tmp_path)
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 2)
    monkeypatch.delattr(groups.os, "sched_getaffinity", raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    assert budget["effective_cpus"] == 2 and budget["threads_per_child"] == 1


def test_real_children_use_cpu_allocation_and_failure_preserves_inherited_env(tmp_path, monkeypatch):
    body = """import os
def test_threads():
    for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ[key] == '1'
    assert 'PYTEST_ADDOPTS' not in os.environ
    assert 'PYTEST_PLUGINS' not in os.environ
    assert False, 'intentional child failure after checking real environment'
"""
    root = repository(tmp_path / "repo", {SELECTORS[-1]: body})
    # Non-Linux fixture CPU count isolates allocation from this developer machine's hierarchy.
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    budget = groups.detect_cpu_budget(system="Darwin")
    monkeypatch.setattr(groups, "detect_cpu_budget", lambda: budget)
    inherited = {**os.environ, "PYTEST_ADDOPTS": "-k nonexistent", "PYTEST_PLUGINS": "bad",
                 **{key: "99" for key in groups.NUMERICAL_THREAD_KEYS}}
    original, before = dict(inherited), dict(os.environ)
    base, report = execute(root, env=inherited)
    assert report["status"] == "failed" and report["cpu_budget"] == budget
    assert report["selected_tests"] == SELECTORS and report["timeout_seconds"] == 30
    assert [item["selectors"] for item in report["groups"]] == groups.split_groups(SELECTORS, TOY_HEAVY)
    assert all(item["environment"] == {**groups.child_environment_contract(1), **{
        key: item["temporary_directory"] for key in groups.TEMPORARY_ENVIRONMENT_KEYS}}
        for item in report["groups"])
    assert "intentional child failure" in (base / "sdk-groups/group-1/pytest.log").read_text()
    assert inherited == original and dict(os.environ) == before


def test_actual_children_override_inherited_threads_and_pytest_filters(tmp_path, monkeypatch):
    body = '''import os
import torch, pyarrow

def test_actual_environment():
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[key] == "1"
    assert torch.get_num_threads() == 1
    assert pyarrow.cpu_count() == pyarrow.io_thread_count() == 1
    assert os.environ["ARROW_IO_THREADS"] == "1"
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert all(key not in os.environ for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"))
    assert "foreign-test" not in os.environ.get("PYTEST_CURRENT_TEST", "")
    assert os.environ["SDK_ENVIRONMENT_SENTINEL"] == "retained"
'''
    root = repository(tmp_path, {selector: body for selector in SELECTORS})
    # Pin the fixture allocation while each real subprocess checks Torch's runtime threads.
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    budget = groups.detect_cpu_budget(system="Darwin")
    monkeypatch.setattr(groups, "detect_cpu_budget", lambda: budget)
    inherited = {**os.environ, "OMP_NUM_THREADS": "8", "OMP_THREAD_LIMIT": "8", "ARROW_IO_THREADS": "8", "MKL_NUM_THREADS": "7",
                 "OPENBLAS_NUM_THREADS": "6", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0",
                 "PYTEST_ADDOPTS": "-k nonexistent_case", "PYTEST_PLUGINS": "nonexistent_plugin",
                 "PYTEST_CURRENT_TEST": "foreign-test", "SDK_ENVIRONMENT_SENTINEL": "retained"}
    original = inherited.copy()
    base, report = execute(root, env=inherited)
    assert report["status"] == "passed", report
    assert report["cpu_budget"] == budget and budget["effective_cpus"] == 2
    assert budget["threads_per_child"] == 1
    assert inherited == original
    assert report["tests"] == {"tests": 4, "failures": 0, "errors": 0, "skipped": 0}
    for child in report["groups"]:
        rows, _ = groups.read_child(base / "sdk-groups" / child["junit"],
                                   base / "sdk-groups" / child["phase_timings"], child["selectors"])
        assert len(rows) == 2


def test_canonical_output_uses_actual_elements_and_records_in_original_order(actual_pass):
    _, base, report = actual_pass
    expected = []
    for item in report["groups"]:
        actual, _ = groups.read_child(
            base / "sdk-groups" / item["junit"],
            base / "sdk-groups" / item["phase_timings"],
            item["selectors"],
        )
        expected.extend(actual)
    expected.sort(key=lambda row: groups.selector_index(row[0].split("::", 1)[0], SELECTORS))
    merged = list(ET.parse(base / "combined.xml").getroot().iter("testcase"))
    assert [ET.tostring(item) for item in merged] == [ET.tostring(row[1]) for row in expected]
    assert (base / "combined.jsonl").read_bytes() == b"".join(
        record for row in expected for record in row[2]
    )
    assert expected[0][0].startswith("tests/test_before.py::")
    assert expected[-1][0].startswith("tests/test_after.py::")


def test_group_processes_really_overlap(actual_pass):
    _, _, report = actual_pass
    a, b = report["groups"]
    assert max(a["start_seconds"], b["start_seconds"]) < min(a["stop_seconds"], b["stop_seconds"])
    assert (
        a["command"][a["command"].index("--basetemp") + 1]
        != b["command"][b["command"].index("--basetemp") + 1]
    )
    assert a["command"][a["command"].index("-o") + 1] != b["command"][b["command"].index("-o") + 1]


@pytest.mark.parametrize(
    "body",
    [
        "def test_bad(): assert False\n",
        "import pytest\n@pytest.mark.skip(reason='fixture')\ndef test_bad(): pass\n",
        "import pytest\n@pytest.fixture\ndef bad(): raise RuntimeError('fixture')\ndef test_bad(bad): pass\n",
    ],
)
def test_real_fail_skip_or_setup_error_cannot_pass(tmp_path, body):
    root = repository(tmp_path / "repo", {SELECTORS[-1]: body})
    base, report = execute(root)
    assert report["status"] == "failed" and "error" in report
    assert (base / "sdk-groups/report.json").exists()
    assert any((base / "sdk-groups" / f"group-{i}/pytest.log").stat().st_size for i in range(2))


def test_future_selector_is_automatically_in_complete_complement():
    selectors = ["tests/test_future.py", *SELECTORS]
    assert groups.split_groups(selectors, TOY_HEAVY)[0] == list(TOY_HEAVY)
    assert groups.split_groups(selectors, TOY_HEAVY)[1] == [selectors[0], SELECTORS[0], SELECTORS[-1]]


def test_production_four_file_partition_retains_actual_source_scope_and_order():
    selectors = groups.selector_contract(ROOT)
    expected_heavy = (
        "tests/test_econ_saved_prediction_linear.py",
        "tests/test_control_function_common_prediction.py",
        "tests/test_streaming_control_function_engine.py",
        "tests/test_control_stream_acceptance.py",
    )
    assert groups.EXPENSIVE == expected_heavy
    assert selectors == gate.TESTS and len(selectors) >= 76
    heavy, other = groups.split_groups(selectors)
    assert heavy == [item for item in selectors if item in expected_heavy]
    assert other == [item for item in selectors if item not in expected_heavy]
    assert len(heavy) == 4 and len(other) == len(selectors) - 4
    assert set(heavy).isdisjoint(other) and set(heavy + other) == set(selectors)
    # Tuple order controls membership only, never the original file order.
    assert groups.split_groups(selectors, tuple(reversed(expected_heavy))) == [heavy, other]


def test_mprobit_survey_categorical_causal_and_packaging_merge_preserves_complete_batches():
    selectors = groups.selector_contract(ROOT)
    merged_batches = {
        "tests/test_mprobit.py",
        "tests/test_mprobit_postestimation.py",
        "tests/test_mprobit_independent.py",
        "tests/test_categorical_spline_regression.py",
        "tests/test_categorical_spline_pca.py",
        "tests/test_categorical_frequency_rotation.py",
        "tests/test_categorical_frequency_bootstrap.py",
        "tests/test_survey_four_stage_oracles.py",
        "tests/test_survey_four_stage_design.py",
        "tests/test_survey_four_stage_review.py",
        "tests/test_survey_four_stage_safety.py",
        "tests/test_survey_four_stage_regression.py",
        "tests/test_survey_four_stage_regression_state.py",
        "tests/test_causal_confidence_sets.py",
        "tests/test_causal_multiarm_neyman.py",
        "tests/test_causal_effect_distribution.py",
        "tests/test_causal_confidence_delivery.py",
        "tests/test_sdist_packaging.py",
    }
    assert merged_batches <= set(selectors)
    assert len(selectors) == len(set(selectors)) == 102
    heavy, other = groups.split_groups(selectors)
    assert len(heavy) == 4 and len(other) == 98
    assert merged_batches <= set(other)
    assert set(heavy).isdisjoint(other)
    assert set(heavy + other) == set(selectors)


def test_production_future_selectors_remain_complete_and_in_original_complement_order():
    selectors = groups.selector_contract(ROOT)
    midpoint = len(selectors) // 2
    future = [
        "tests/test_future_before.py", *selectors[:midpoint],
        "tests/test_future_middle.py", *selectors[midpoint:], "tests/test_future_after.py",
    ]
    heavy, other = groups.split_groups(future)
    assert heavy == groups.split_groups(selectors)[0]
    assert other == [item for item in future if item not in groups.EXPENSIVE]
    assert len(heavy) + len(other) == len(future) == len(selectors) + 3
    assert set(heavy).isdisjoint(other) and set(heavy + other) == set(future)


@pytest.mark.parametrize("missing", groups.EXPENSIVE)
def test_production_missing_any_heavy_source_file_is_rejected(missing):
    selectors = [item for item in groups.selector_contract(ROOT) if item != missing]
    with pytest.raises(ValueError, match="Heavy SDK selectors missing"):
        groups.split_groups(selectors)


@pytest.mark.parametrize(
    "selectors",
    [
        [],
        ["one"],
        [*SELECTORS, SELECTORS[0]],
        [*SELECTORS, "../escape.py"],
        [*SELECTORS, "/absolute.py"],
        [*SELECTORS, "tests"],
        [*SELECTORS, "tests//bad.py"],
        [*SELECTORS, "tests/test_x.py::test_one"],
    ],
)
def test_invalid_overlapping_or_duplicate_selector_scope_rejected(selectors):
    with pytest.raises(ValueError):
        groups.split_groups(selectors, TOY_HEAVY)


def test_missing_heavy_selector_cannot_silently_change_partition():
    with pytest.raises(ValueError, match="Heavy SDK selectors missing"):
        groups.split_groups([SELECTORS[0], SELECTORS[-1]], TOY_HEAVY)


@pytest.mark.parametrize(
    "attack",
    [
        "duplicate_case",
        "missing_case",
        "count_only",
        "failure",
        "skip",
        "empty",
        "outside_source",
        "reordered_phases",
        "duplicate_phase",
        "missing_phase",
        "nonfinite",
        "duplicate_json_key",
    ],
)
def test_real_raw_evidence_tamper_rejected(actual_pass, tmp_path, attack):
    children = copied_children(actual_pass, tmp_path)
    child = children[0]
    xml = child / "pytest.xml"
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    cases = suite.findall("testcase")
    phase = child / "pytest-timings.jsonl"
    lines = phase.read_bytes().splitlines(keepends=True)
    if attack == "duplicate_case":
        suite.append(ET.fromstring(ET.tostring(cases[0])))
        suite.set("tests", str(len(cases) + 1))
    elif attack == "missing_case":
        suite.remove(cases[0])
        suite.set("tests", str(len(cases) - 1))
    elif attack == "count_only":
        suite.set("tests", "999")
    elif attack in ("failure", "skip"):
        ET.SubElement(cases[0], "skipped" if attack == "skip" else "failure")
    elif attack == "empty":
        for case in cases:
            suite.remove(case)
        suite.set("tests", "0")
    elif attack == "outside_source":
        record = json.loads(lines[0])
        record["nodeid"] = "tests/test_other.py::test_x"
        lines[0] = (json.dumps(record) + "\n").encode()
    elif attack == "reordered_phases":
        lines[0], lines[1] = lines[1], lines[0]
    elif attack == "duplicate_phase":
        lines += lines[:3]
    elif attack == "missing_phase":
        lines.pop()
    elif attack == "nonfinite":
        record = json.loads(lines[0])
        record["duration"] = math.inf
        lines[0] = (json.dumps(record) + "\n").encode()
    elif attack == "duplicate_json_key":
        lines[0] = lines[0].replace(b"{", b'{"outcome":"passed",', 1)
    tree.write(xml, encoding="utf-8")
    phase.write_bytes(b"".join(lines))
    with pytest.raises(ValueError):
        groups.merge_evidence(
            children,
            SELECTORS,
            groups.split_groups(SELECTORS, TOY_HEAVY),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_selector_missing_all_real_cases_is_rejected(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    child = children[1]
    xml = child / "pytest.xml"
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    for case in list(suite):
        if case.get("classname") == "tests.test_before" or case.get("classname", "").startswith(
            "tests.test_before."
        ):
            suite.remove(case)
    suite.set("tests", str(len(suite.findall("testcase"))))
    tree.write(xml, encoding="utf-8")
    phase = child / "pytest-timings.jsonl"
    phase.write_bytes(
        b"".join(
            line
            for line in phase.read_bytes().splitlines(keepends=True)
            if not json.loads(line)["nodeid"].startswith("tests/test_before.py::")
        )
    )
    with pytest.raises(ValueError, match="omit|coverage missing"):
        groups.merge_evidence(
            children,
            SELECTORS,
            groups.split_groups(SELECTORS, TOY_HEAVY),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_cross_file_reordering_rejected_even_if_xml_and_phases_agree(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    child = children[0]
    xml = child / "pytest.xml"
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    cases = suite.findall("testcase")
    suite[:] = cases[4:] + cases[:4]
    tree.write(xml, encoding="utf-8")
    phase = child / "pytest-timings.jsonl"
    lines = phase.read_bytes().splitlines(keepends=True)
    phase.write_bytes(b"".join(lines[12:] + lines[:12]))
    with pytest.raises(ValueError, match="reorder"):
        groups.merge_evidence(
            children,
            SELECTORS,
            groups.split_groups(SELECTORS, TOY_HEAVY),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_partition_cannot_omit_repeat_or_move_a_selector(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    partition = groups.split_groups(SELECTORS, TOY_HEAVY)
    partition[1].pop()
    with pytest.raises(ValueError, match="partition"):
        groups.merge_evidence(
            children, SELECTORS, partition, tmp_path / "x.xml", tmp_path / "x.jsonl"
        )


def test_original_parent_clock_includes_import_startup_and_merge(tmp_path):
    root = repository(tmp_path / "repo")
    started = time.monotonic() - 31
    _, report = execute(root, timeout=30, parent_started=started, parent_deadline=started + 30)
    assert report["status"] == "failed" and "deadline" in report["error"]
    assert report["groups"] == [] and report["seconds"] >= 31


def test_real_total_timeout_preserves_partial_phases_and_kills_children(tmp_path):
    body = "import time\ndef test_complete(): pass\ndef test_sleep(): time.sleep(30)\n"
    root = repository(tmp_path / "repo", {item: body for item in SELECTORS})
    base, report = execute(root, timeout=1)
    assert report["status"] == "failed" and "deadline" in report["error"]
    assert all(item["exit_code"] != 0 for item in report["groups"])
    assert any(
        (base / "sdk-groups" / item["phase_timings"]).stat().st_size > 0
        for item in report["groups"]
    )


@pytest.mark.skipif(os.name == "nt", reason="Process-group regression requires POSIX")
def test_outer_wrapper_timeout_stops_every_child_and_grandchild(tmp_path, monkeypatch):
    # A smaller test watchdog deliberately fires before the helper's own clock.
    body = """import pathlib,subprocess,sys,time
def test_worker():
    marker = pathlib.Path(__file__).with_suffix('.orphan')
    child = subprocess.Popen([sys.executable, '-c', "import pathlib,time; time.sleep(4); pathlib.Path(" + repr(str(marker)) + ").write_text('orphan')"])
    pathlib.Path(__file__).with_suffix('.pid').write_text(str(child.pid))
    time.sleep(30)
"""
    root = repository(tmp_path / "repo", {item: body for item in SELECTORS})
    wrapper = root / "scripts/run_parallel_sdk_groups.py"
    wrapper.write_text(f"""import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('actual_helper',{str(ROOT / "scripts/run_parallel_sdk_groups.py")!r})
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
root=Path({str(root)!r});base=root/'receipts/outer';base.mkdir(parents=True)
report=helper.run_groups(root=root,python=Path(sys.executable),selectors={SELECTORS!r},directory=base/'sdk-groups',junit=base/'all.xml',timings=base/'all.jsonl',timeout=60,expensive={TOY_HEAVY!r})
raise SystemExit(0 if report['status']=='passed' else 1)
""")
    monkeypatch.setattr(gate, "ROOT", root)
    receipt = root / "receipts"
    receipt.mkdir()
    result = gate.run_step(
        "outer", [sys.executable, str(wrapper)], receipt, 2, env=groups.environment(root)
    )
    assert result["status"] == "cancelled_or_timeout" and result["exit_code"] != 0
    report = json.loads((receipt / "outer/sdk-groups/report.json").read_text())
    assert report["status"] == "failed" and "signal" in report["error"]
    pid_files = list((root / "tests").glob("*.pid"))
    assert len(pid_files) == 2, "Both real child processes must have spawned a grandchild"
    time.sleep(2.5)
    assert not list((root / "tests").glob("*.orphan")), (
        "A grandchild survived process-group cleanup"
    )


@pytest.mark.parametrize("shard,timeouts", [(None, (900, 1799, 1801)), (0, (899, 901, 1800)),
                                           (1, (899, 901, 1800))])
def test_production_cli_cannot_extend_shrink_or_borrow_mode_deadline(tmp_path, shard, timeouts):
    for timeout in timeouts:
        shard_arguments = [] if shard is None else ["--shard-index", str(shard)]
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_parallel_sdk_groups.py"),
                "--python",
                sys.executable,
                "--directory",
                str(tmp_path / "groups"),
                "--junitxml",
                str(tmp_path / "out.xml"),
                "--gate-timings",
                str(tmp_path / "out.jsonl"),
                "--execution",
                str(tmp_path / "execution.json"),
                "--component-sdk-version",
                "3.13",
                "--timeout",
                str(timeout),
                *shard_arguments,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        expected = (f"This SDK mode retains its {1800 if shard is None else 900}-second deadline"
                    if timeout in (900, 1800) else "invalid choice")
        assert result.returncode == 2 and expected in result.stderr
    assert not (tmp_path / "groups").exists()


def test_source_selector_contract_reads_future_additions_without_importing_runtime(tmp_path):
    root = repository(tmp_path / "repo")
    values = [*SELECTORS, "tests/test_future.py"]
    (root / "scripts/verify_merge_candidate.py").write_text(
        "TESTS=" + repr(values) + "\nraise RuntimeError('must not import')\n"
    )
    assert groups.selector_contract(root) == values
    assert groups.split_groups(values, TOY_HEAVY)[1][-1] == "tests/test_future.py"


def test_inherited_selectors_cannot_shrink_either_real_child(tmp_path, monkeypatch):
    root = repository(tmp_path / "repo")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nonexistent")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_plugin")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "77")
        assert groups.environment(root, threads=2)[key] == groups.child_environment_contract(2)[key]
    _, report = execute(root)
    assert report["status"] == "passed" and report["tests"]["tests"] == 16
    for child in report["groups"]:
        assert all(child["environment"][key] == str(report["cpu_budget"]["threads_per_child"])
                   for key in groups.NUMERICAL_THREAD_KEYS)


@pytest.mark.parametrize("inherited_threads", ["99", "2"])
def test_real_children_override_inherited_threads_and_record_actual_environment(
    tmp_path, inherited_threads
):
    body = """import json
import os
from pathlib import Path

def test_child_environment():
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_THREAD_LIMIT', 'ARROW_IO_THREADS')
    actual = {name: os.environ[name] for name in names}
    assert len(set(actual.values())) == 1 and actual[names[0]] in ('1', '2')
    assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert not any(name in os.environ for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'))
    assert os.environ['PYTEST_CURRENT_TEST'] == (
        'tests/' + Path(__file__).name + '::test_child_environment (call)')
    target = Path(os.environ['SDK_ENV_TEST_DIRECTORY']) / (Path(__file__).stem + '.json')
    target.write_text(json.dumps(actual))
"""
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    actual_directory = root / "receipts/actual-environment"
    actual_directory.mkdir(parents=True)
    inherited = {
        **os.environ,
        "OMP_NUM_THREADS": inherited_threads,
        "MKL_NUM_THREADS": inherited_threads,
        "OPENBLAS_NUM_THREADS": inherited_threads,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0",
        "PYTEST_ADDOPTS": "-k nonexistent",
        "PYTEST_PLUGINS": "nonexistent_plugin",
        "PYTEST_CURRENT_TEST": "inherited test identity",
        "SDK_ENV_TEST_DIRECTORY": str(actual_directory),
    }
    original = inherited.copy()
    _, report = execute(root, env=inherited)
    assert report["status"] == "passed" and report["tests"]["tests"] == len(SELECTORS)
    assert inherited == original
    assert report["environment"] == groups.child_environment_contract(report["cpu_budget"]["threads_per_child"])
    observed = list(actual_directory.glob("*.json"))
    assert {path.stem for path in observed} == {Path(selector).stem for selector in SELECTORS}
    for child in report["groups"]:
        assert child["environment"] == {**report["environment"], **{key: child["temporary_directory"] for key in groups.TEMPORARY_ENVIRONMENT_KEYS}}
        for selector in child["selectors"]:
            actual = json.loads((actual_directory / (Path(selector).stem + ".json")).read_text())
            assert actual == {name: child["environment"][name] for name in actual}


@pytest.mark.parametrize(
    "target,kind",
    [
        ("combined.xml", "file"),
        ("combined.jsonl", "file"),
        ("combined.xml", "symlink"),
        ("sdk-groups", "directory"),
    ],
)
def test_stale_or_reused_output_is_rejected_before_any_child(tmp_path, target, kind):
    root = repository(tmp_path / "repo")
    base = root / "receipts/run"
    base.mkdir(parents=True)
    path = base / target
    if kind == "directory":
        path.mkdir()
        (path / "report.json").write_text('{"status":"passed"}')
    elif kind == "symlink":
        path.symlink_to(base / "missing-target")
    else:
        path.write_text("old result")
    with pytest.raises((ValueError, FileExistsError)):
        groups.run_groups(
            root=root,
            python=Path(sys.executable),
            selectors=SELECTORS,
            directory=base / "sdk-groups",
            junit=base / "combined.xml",
            timings=base / "combined.jsonl",
            timeout=30,
            expensive=TOY_HEAVY,
        )
    assert not list((base / "sdk-groups").glob("group-*"))


def test_real_collection_error_never_creates_accepted_merge(tmp_path):
    root = repository(
        tmp_path / "repo", {SELECTORS[-1]: "raise RuntimeError('collection fixture error')\n"}
    )
    base, report = execute(root)
    assert report["status"] == "failed"
    assert not (base / "combined.xml").exists() and not (base / "combined.jsonl").exists()


def test_execution_source_or_component_mismatch_is_rejected_before_children(tmp_path):
    root = repository(tmp_path / "repo")
    with pytest.raises(ValueError, match="execution/source binding"):
        execute(
            root,
            execution={"source_commit": "0" * 40, "component_sdk_version": "3.13"},
            component="3.13",
        )


def test_parent_clock_cannot_extend_original_budget(tmp_path):
    root = repository(tmp_path / "repo")
    started = time.monotonic()
    with pytest.raises(ValueError, match="differs from original"):
        execute(root, timeout=30, parent_started=started, parent_deadline=started + 31)


def test_coherent_case_and_phase_omission_cannot_hide_collected_test(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    child = children[0]
    xml = child / "pytest.xml"
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    suite.remove(suite.findall("testcase")[0])
    suite.set("tests", str(len(suite.findall("testcase"))))
    tree.write(xml, encoding="utf-8")
    phase = child / "pytest-timings.jsonl"
    phase.write_bytes(b"".join(phase.read_bytes().splitlines(keepends=True)[3:]))
    with pytest.raises(ValueError, match="same-process collected tests"):
        groups.merge_evidence(
            children,
            SELECTORS,
            groups.split_groups(SELECTORS, TOY_HEAVY),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


@pytest.mark.parametrize("attack", ["missing", "duplicate", "reorder", "invalid_fields"])
def test_same_process_collection_tamper_rejected(actual_pass, tmp_path, attack):
    children = copied_children(actual_pass, tmp_path)
    collection = children[0] / "pytest-collection.jsonl"
    lines = collection.read_bytes().splitlines(keepends=True)
    if attack == "missing":
        lines.pop()
    elif attack == "duplicate":
        lines.append(lines[0])
    elif attack == "reorder":
        lines[0], lines[1] = lines[1], lines[0]
    else:
        lines[0] = b'{"nodeid":"tests/test_x.py::test_one","other":1}\n'
    collection.write_bytes(b"".join(lines))
    with pytest.raises(ValueError):
        groups.merge_evidence(
            children,
            SELECTORS,
            groups.split_groups(SELECTORS, TOY_HEAVY),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_unknown_current_and_future_nodes_are_assigned_once_with_stable_ties():
    nodes = [f"tests/test_new_scope.py::test_case[{i}]" for i in range(13)]
    plan = groups.make_partition_plan(nodes)
    assert plan["assignments"] == [i % 4 for i in range(13)]
    partition = groups.partition_nodes(nodes)
    assert Counter(node for child in partition for node in child) == Counter(nodes)
    assert plan["group_counts"] == [4, 3, 3, 3]
    expected_raw = b"".join(
        (json.dumps({"nodeid": node}, separators=(",", ":")) + "\n").encode()
        for node in nodes
    )
    assert plan["full_collection_sha256"] == hashlib.sha256(expected_raw).hexdigest()
    assert groups.make_partition_plan(nodes) == plan
    future = nodes + ["tests/test_future_scope.py::TestFuture::test_added"]
    future_partition = groups.partition_nodes(future)
    assert Counter(node for child in future_partition for node in child) == Counter(future)
    assert all(child == [node for node in future if node in child] for child in future_partition)


def test_exhaustive_causal_law_weight_is_advisory_and_keeps_complete_future_scope():
    exhaustive = (
        "tests/test_causal_multiarm_neyman.py::"
        "test_two_strata_full_cartesian_8100_law_full_covariance_and_population_weights"
    )
    assert groups.ADVISORY_NODE_SECONDS[exhaustive] == 809.884951728
    known = sorted(node for node in groups.ADVISORY_NODE_SECONDS if node != exhaustive)
    future = [f"tests/test_new_scope.py::test_future[{i}]" for i in range(4000)]
    nodes = [exhaustive, *known, *future]
    original = nodes.copy()
    plan = groups.make_partition_plan(nodes)
    partition = groups.partition_nodes(nodes)
    assert nodes == original
    assert partition[plan["assignments"][0]] == [exhaustive]
    assert Counter(node for child in partition for node in child) == Counter(nodes)
    assert sum(plan["group_counts"]) == len(nodes)
    assert all(child == [node for node in nodes if node in child] for child in partition)
    assert all(seconds > 0 for seconds in plan["estimated_group_seconds"])
    # These are scheduling estimates; passing them does not certify the deadline.
    assert plan["kind"] == "source_bound_node_partition" and "status" not in plan


@pytest.mark.parametrize("count", [True, False, 0, 1, 2, 3, 5, 4.0, "4", None])
def test_distributed_partition_requires_exact_four_integer_groups(count):
    nodes = [f"tests/test_scope.py::test_{i}" for i in range(4)]
    with pytest.raises(ValueError):
        groups.make_partition_plan(nodes, group_count=count)


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        ["tests/test_one.py::test_a"] * 4,
        [[], "tests/test_one.py::test_a", "tests/test_one.py::test_b", "tests/test_one.py::test_c"],
        [None, "tests/test_one.py::test_a", "tests/test_one.py::test_b", "tests/test_one.py::test_c"],
        ["../escape.py::test_a", "tests/test_one.py::test_a", "tests/test_one.py::test_b", "tests/test_one.py::test_c"],
        ["/absolute.py::test_a", "tests/test_one.py::test_a", "tests/test_one.py::test_b", "tests/test_one.py::test_c"],
        ["tests//bad.py::test_a", "tests/test_one.py::test_a", "tests/test_one.py::test_b", "tests/test_one.py::test_c"],
        ["tests/test_one.py", "tests/test_one.py::test_a", "tests/test_one.py::test_b", "tests/test_one.py::test_c"],
    ],
)
def test_invalid_complete_node_scope_cannot_create_a_plan(nodes):
    with pytest.raises(ValueError):
        groups.make_partition_plan(nodes)


def test_actual_current_selector_collection_has_exact_four_way_node_union(tmp_path):
    selectors = groups.selector_contract(ROOT)
    assert selectors == gate.TESTS and len(selectors) == 102
    assert selectors[:3] == ["tests/test_mprobit.py", "tests/test_mprobit_postestimation.py",
                             "tests/test_mprobit_independent.py"]
    assert {
        "tests/test_causal_confidence_sets.py", "tests/test_causal_multiarm_neyman.py",
        "tests/test_causal_effect_distribution.py", "tests/test_causal_confidence_delivery.py",
        "tests/test_sdist_packaging.py",
    } <= set(selectors)
    collection = tmp_path / "complete.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
         "-p", "scripts.pytest_gate_timings", "--gate-collection", str(collection),
         "--collect-only", *selectors],
        cwd=ROOT, env=groups.environment(ROOT, distributed=True), capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = groups.collected_nodes(collection)
    # Preserve the complete previous 9,921-case scope plus all new causal and packaging cases.
    assert len(nodes) >= 9921
    assert sum(node.startswith("tests/test_sdist_packaging.py::") for node in nodes) == 3
    assert {groups.selector_index(node.split("::", 1)[0], selectors) for node in nodes} == set(range(len(selectors)))
    partition = groups.partition_nodes(nodes)
    assert len(partition) == 4 and all(partition)
    assert Counter(node for child in partition for node in child) == Counter(nodes)
    assert all(child == [node for node in nodes if node in child] for child in partition)
    exhaustive = (
        "tests/test_causal_multiarm_neyman.py::"
        "test_two_strata_full_cartesian_8100_law_full_covariance_and_population_weights"
    )
    assert sum(node == exhaustive for node in nodes) == 1
    assert [child for child in partition if exhaustive in child] == [[exhaustive]]


@pytest.fixture(scope="module")
def actual_shards(tmp_path_factory):
    body = """import pytest,time
@pytest.mark.parametrize("value", [3, 1, 2])
def test_values(value):
    time.sleep(.1)
    assert value in [3, 1, 2]
class TestExact:
    def test_class(self):
        time.sleep(.1)
        assert True
"""
    root = repository(tmp_path_factory.mktemp("sdk-shards"), {selector: body for selector in SELECTORS})
    driver = root / "shard_driver.py"
    driver.write_text(f"""import json,sys
from pathlib import Path
from scripts.run_parallel_sdk_groups import run_groups
root=Path({str(root)!r});shard=int(sys.argv[1]);base=root/'receipts'/('shard-'+str(shard));base.mkdir(parents=True)
report=run_groups(root=root,python=Path(sys.executable),selectors={SELECTORS!r},directory=base/'sdk-groups',junit=base/'combined.xml',timings=base/'combined.jsonl',timeout=30,expensive={TOY_HEAVY!r},shard_index=shard)
print(json.dumps(report));raise SystemExit(0 if report['status']=='passed' else 1)
""")
    processes = [subprocess.Popen([sys.executable, str(driver), str(shard)], cwd=root,
                                  env=groups.environment(root), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
                 for shard in (0, 1)]
    reports = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=45)
        assert process.returncode == 0, stdout + stderr
        reports.append(json.loads(stdout))
    return root, [root / "receipts" / f"shard-{i}" for i in range(2)], reports


def test_two_real_shards_keep_full_collection_and_exact_actual_case_union(actual_shards):
    _, bases, reports = actual_shards
    full, plans, selected, actual = [], [], [], []
    for shard, (base, report) in enumerate(zip(bases, reports, strict=True)):
        assert report["schema"] == 2 and report["partial_scope"] is True
        assert report["shard_index"] == shard and report["shard_count"] == 2
        assert [item["index"] for item in report["groups"]] == [2 * shard, 2 * shard + 1]
        assert report["selected_tests"] == SELECTORS
        assert report["environment"] == groups.child_environment_contract(1)
        for item in report["groups"]:
            assert item["environment"] == {
                **report["environment"],
                **{key: item["temporary_directory"] for key in groups.TEMPORARY_ENVIRONMENT_KEYS},
            }
            directory = base / "sdk-groups" / f"group-{item['index']}"
            full.append(groups.collected_nodes(directory / "pytest-full-collection.jsonl"))
            plans.append(json.loads((directory / "partition-plan.json").read_text()))
            nodes = groups.collected_nodes(directory / "pytest-collection.jsonl")
            selected.extend(nodes)
            phase = [json.loads(line) for line in (directory / "pytest-timings.jsonl").read_bytes().splitlines()]
            assert [row["nodeid"] for row in phase if row["phase"] == "call"] == nodes
            assert all(row["outcome"] == "passed" for row in phase)
            actual.extend(nodes)
            assert item["command"][-len(SELECTORS):] == SELECTORS
            for filename in ("pytest-full-collection.jsonl", "pytest-collection.jsonl", "partition-plan.json"):
                assert (directory / filename).is_file()
        assert len(list(ET.parse(base / "combined.xml").getroot().iter("testcase"))) == 8
    assert len(full) == 4 and all(nodes == full[0] for nodes in full)
    assert len(full[0]) == 16 and all(plan == plans[0] for plan in plans)
    assert Counter(actual) == Counter(full[0]) == Counter(selected)
    assert len(actual) == len(set(actual)) == 16


def test_real_distributed_children_use_four_isolated_temp_and_cache_locations(actual_shards):
    _, _, reports = actual_shards
    commands = [item["command"] for report in reports for item in report["groups"]]
    assert len({cmd[cmd.index("--basetemp") + 1] for cmd in commands}) == 4
    assert len({cmd[cmd.index("-o") + 1] for cmd in commands}) == 4
    assert {int(cmd[cmd.index("--gate-group-index") + 1]) for cmd in commands} == set(range(4))


def test_four_real_children_overlap_in_actual_phase_wall_intervals(actual_shards):
    _, bases, reports = actual_shards
    intervals = []
    for base, report in zip(bases, reports, strict=True):
        for item in report["groups"]:
            path = base / "sdk-groups" / f"group-{item['index']}" / "pytest-timings.jsonl"
            phases = [json.loads(line) for line in path.read_bytes().splitlines()]
            intervals.append((min(row["start"] for row in phases), max(row["stop"] for row in phases)))
    assert len(intervals) == 4
    assert max(start for start, _ in intervals) < min(stop for _, stop in intervals)


def test_partial_canonical_merge_preserves_actual_xml_and_untouched_phase_bytes(actual_shards):
    _, bases, reports = actual_shards
    for base, report in zip(bases, reports, strict=True):
        expected = []
        for item in report["groups"]:
            child = base / "sdk-groups" / f"group-{item['index']}"
            selected = groups.collected_nodes(child / "pytest-collection.jsonl")
            rows, _ = groups.read_child(child / "pytest.xml", child / "pytest-timings.jsonl",
                                        SELECTORS, expected_nodes=selected)
            expected.extend(rows)
        full = groups.collected_nodes(base / "sdk-groups" / f"group-{report['groups'][0]['index']}" / "pytest-full-collection.jsonl")
        expected.sort(key=lambda row: full.index(row[0]))
        merged = list(ET.parse(base / "combined.xml").getroot().iter("testcase"))
        assert [ET.tostring(case) for case in merged] == [ET.tostring(row[1]) for row in expected]
        assert (base / "combined.jsonl").read_bytes() == b"".join(raw for row in expected for raw in row[2])


def test_real_shard_clock_contains_collection_execution_cleanup_and_merge(actual_shards):
    _, _, reports = actual_shards
    for report in reports:
        clock = report["sdk_clock"]
        assert 0 <= clock["elapsed_seconds"] <= 30
        assert abs(clock["elapsed_seconds"] - report["seconds"]) <= 5.1e-7
        assert abs(clock["monotonic_finished"] - clock["monotonic_started"] - clock["elapsed_seconds"]) <= 1e-6
        assert abs(clock["finished_utc"] - clock["started_utc"] - clock["elapsed_seconds"]) <= 1
        assert clock["clock_identity"]["pid"] > 0 and clock["clock_identity"]["hostname"]
        for item in report["groups"]:
            assert clock["started_utc"] <= item["start_utc"] <= item["stop_utc"] <= clock["finished_utc"]
            assert 0 <= item["start_seconds"] <= item["stop_seconds"] <= report["seconds"]


@pytest.mark.parametrize("value", [False, -1, 0, math.nan, math.inf, "now"])
def test_invalid_distributed_utc_clock_is_rejected_before_children(tmp_path, value):
    root = repository(tmp_path / "repo")
    with pytest.raises(ValueError):
        execute(root, shard_index=0, parent_started_utc=value)
    assert not list((root / "receipts").rglob("group-*"))


def test_distributed_utc_and_monotonic_start_cannot_describe_different_intervals(tmp_path):
    root = repository(tmp_path / "repo")
    with pytest.raises(ValueError, match="UTC/monotonic"):
        execute(root, shard_index=0, parent_started_utc=time.time() - 20)
    assert not list((root / "receipts").rglob("group-*"))


def test_distributed_expired_parent_budget_refuses_collection(tmp_path):
    root = repository(tmp_path / "repo")
    started = time.monotonic() - 31
    _, report = execute(root, timeout=30, shard_index=0, parent_started=started,
                        parent_deadline=started + 30, parent_started_utc=time.time() - 31)
    assert report["status"] == "failed" and "deadline" in report["error"]
    assert not report["groups"] and report["seconds"] >= 31


def test_distributed_evidence_merge_is_inside_original_budget(tmp_path, monkeypatch):
    root = repository(tmp_path / "repo")
    original_merge = groups.merge_partition_evidence
    monotonic, utc = time.monotonic, time.time

    def expensive_merge(*args, **kwargs):
        result = original_merge(*args, **kwargs)
        # Model elapsed validation work after real pytest/collection have finished.
        monkeypatch.setattr(groups.time, "monotonic", lambda: monotonic() + 31)
        monkeypatch.setattr(groups.time, "time", lambda: utc() + 31)
        return result

    monkeypatch.setattr(groups, "merge_partition_evidence", expensive_merge)
    _, report = execute(root, timeout=30, shard_index=0)
    assert report["status"] == "failed" and "deadline" in report["error"]
    assert report["seconds"] >= 31 and all(item["exit_code"] == 0 for item in report["groups"])


@pytest.mark.parametrize("body", [
    "def test_bad(): assert False\n",
    "import pytest\n@pytest.mark.skip(reason='fixture')\ndef test_bad(): pass\n",
    "import pytest\n@pytest.fixture\ndef bad(): raise RuntimeError('fixture')\ndef test_bad(bad): pass\n",
])
def test_distributed_real_fail_skip_setup_error_never_passes(tmp_path, body):
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    _, report = execute(root, shard_index=0)
    assert report["status"] == "failed" and "error" in report


def test_distributed_thread_and_pytest_poison_cannot_change_actual_children(tmp_path):
    body = """import os, torch, pyarrow
def test_environment():
    assert all(os.environ[name]=='1' for name in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','ARROW_IO_THREADS'])
    assert torch.get_num_threads() == 1
    assert pyarrow.cpu_count() == pyarrow.io_thread_count() == 1
"""
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    poison = dict(os.environ, OMP_NUM_THREADS="999", OMP_THREAD_LIMIT="999", ARROW_IO_THREADS="999", MKL_NUM_THREADS="999", OPENBLAS_NUM_THREADS="999",
                  PYTEST_ADDOPTS="-k missing", PYTEST_PLUGINS="nonexistent_plugin")
    _, report = execute(root, shard_index=0, env=poison)
    assert report["status"] == "passed"
    for item in report["groups"]:
        assert all(item["environment"][key] == "1" for key in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))


@pytest.mark.parametrize("shard", [None, 0])
def test_one_cpu_quota_serializes_real_children_and_preserves_actual_scope(tmp_path, monkeypatch, shard):
    (tmp_path / "hierarchy").mkdir()
    proc, _, _ = cpu_hierarchy(tmp_path / "hierarchy", quota="50000")
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    budget = groups.detect_cpu_budget(proc=proc, system="Linux")
    monkeypatch.setattr(groups, "detect_cpu_budget", lambda: budget)
    body = '''import time, torch, pyarrow
from pathlib import Path
def test_capacity():
    assert torch.get_num_threads() == 1
    assert pyarrow.cpu_count() == pyarrow.io_thread_count() == 1
    lock = Path("receipts/worker.lock")
    lock.mkdir()
    try:
        time.sleep(.05)
    finally:
        lock.rmdir()
'''
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    base, report = execute(root, shard_index=shard)
    assert report["status"] == "passed", report
    assert report["cpu_budget"]["max_concurrent_children"] == 1
    first, second = report["groups"]
    assert first["pid"] != second["pid"] and first["stop_seconds"] <= second["start_seconds"]
    assert report["tests"]["tests"] == (4 if shard is None else 2)
    assert report["selected_tests"] == SELECTORS
    assert groups.digest(base / "combined.xml") == report["combined"]["junit_sha256"]


@pytest.mark.parametrize("shard", [True, False, -1, 2, 0.0, "0"])
def test_invalid_shard_identity_rejected_before_children(tmp_path, shard):
    root = repository(tmp_path / "repo")
    with pytest.raises(ValueError):
        execute(root, shard_index=shard)
    assert not list((root / "receipts").rglob("group-*"))


@pytest.mark.parametrize("attack", ["wrong_group", "duplicate_group", "missing_group", "plan_assignment", "full_omission", "selected_omission", "coherent_executed_omission"])
def test_distributed_raw_plan_and_scope_tamper_cannot_merge(actual_shards, tmp_path, attack):
    _, bases, _ = actual_shards
    children = [tmp_path / f"group-{i}" for i in (0, 1)]
    for index, child in enumerate(children):
        shutil.copytree(bases[0] / "sdk-groups" / f"group-{index}", child)
    indices = [0, 1]
    if attack == "wrong_group":
        indices = [0, 2]
    elif attack == "duplicate_group":
        indices = [0, 0]
    elif attack == "missing_group":
        children.pop()
        indices.pop()
    elif attack == "plan_assignment":
        path = children[0] / "partition-plan.json"
        plan = json.loads(path.read_text())
        plan["assignments"][0] = (plan["assignments"][0] + 1) % 4
        path.write_text(json.dumps(plan))
    elif attack == "full_omission":
        path = children[0] / "pytest-full-collection.jsonl"
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[1:]))
    elif attack == "selected_omission":
        path = children[0] / "pytest-collection.jsonl"
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[1:]))
    else:
        child = children[0]
        xml = child / "pytest.xml"
        tree = ET.parse(xml)
        suite = next(tree.getroot().iter("testsuite"))
        suite.remove(suite.findall("testcase")[0])
        suite.set("tests", str(len(suite.findall("testcase"))))
        tree.write(xml, encoding="utf-8")
        phases = child / "pytest-timings.jsonl"
        phases.write_bytes(b"".join(phases.read_bytes().splitlines(keepends=True)[3:]))
        collection = child / "pytest-collection.jsonl"
        collection.write_bytes(b"".join(collection.read_bytes().splitlines(keepends=True)[1:]))
        # All edited files have valid new hashes; complete collection still proves loss.
        assert all(len(groups.digest(child / name)) == 64 for name in
                   ("pytest.xml", "pytest-timings.jsonl", "pytest-collection.jsonl"))
    with pytest.raises(ValueError):
        groups.merge_partition_evidence(children, SELECTORS, indices,
                                        tmp_path / "merged.xml", tmp_path / "merged.jsonl")


@pytest.mark.parametrize("filtering", [
    ["-k", "test_class"], ["-m", "selected"],
    ["--deselect", "tests/test_before.py::TestExact::test_class"],
    ["--ignore", "tests/test_before.py"], ["--ignore-glob", "*test_before.py"],
])
def test_direct_distributed_plugin_cannot_filter_original_collection(tmp_path, filtering):
    root = repository(tmp_path / "repo")
    child = root / "receipts/direct"
    child.mkdir(parents=True)
    temporary = child / "runtime-temp"
    temporary.mkdir()
    command = groups.child_command(Path(sys.executable), root, child, SELECTORS, group_index=0)
    result = subprocess.run(command + filtering, cwd=root,
                            env=groups.environment(root, distributed=True, temporary=temporary),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "forbids caller test filtering" in result.stdout + result.stderr
    assert not (child / "pytest-full-collection.jsonl").exists()
    assert not (child / "partition-plan.json").exists()


@pytest.mark.parametrize("shard", [None, 0, 1])
def test_concurrent_live_export_profiles_keep_strict_local_cleanup(tmp_path, monkeypatch, shard):
    # Reproduce the real PDF failure: both children hold a browser-style profile
    # open at once, while retaining the original before/after cleanup assertion.
    body = '''import json, tempfile, time
from pathlib import Path
def test_live_export_profile():
    temporary = Path(tempfile.gettempdir())
    child, gate = temporary.parent, temporary.parent.parent
    count = child / "barrier-round"
    round_number = int(count.read_text()) + 1 if count.exists() else 1
    count.write_text(str(round_number))
    before = sorted(temporary.glob("openecon-network-export-*"))
    with tempfile.TemporaryDirectory(prefix="openecon-network-export-") as name:
        profile = Path(name)
        marker = gate / f"live-{round_number}-{child.name}.json"
        pending = marker.with_suffix(".pending")
        pending.write_text(json.dumps({"profile": name}))
        pending.replace(marker)
        deadline = time.monotonic() + 8
        while True:
            records = [json.loads(path.read_text()) for path in gate.glob(f"live-{round_number}-*.json")]
            if len(records) == 2 and all(Path(row["profile"]).is_dir() for row in records):
                break
            assert time.monotonic() < deadline, "Both actual children must hold live profiles"
            time.sleep(.01)
        assert sorted(temporary.glob("openecon-network-export-*")) == [*before, profile]
        assert len({str(Path(row["profile"]).parent) for row in records}) == 2
        (gate / f"checked-{round_number}-{child.name}").write_text("checked")
        while len(list(gate.glob(f"checked-{round_number}-*"))) != 2:
            assert time.monotonic() < deadline
            time.sleep(.01)
    assert sorted(temporary.glob("openecon-network-export-*")) == before
'''
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    monkeypatch.setattr(groups.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    detect = groups.detect_cpu_budget
    monkeypatch.setattr(groups, "detect_cpu_budget", lambda *a, **kw: detect(*a, **kw, system="Darwin"))
    foreign = root / "receipts/foreign"
    foreign.mkdir(parents=True)
    sentinel = foreign / "keep"
    sentinel.write_text("preserve")
    for key in groups.TEMPORARY_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, str(foreign))
    inherited = dict(os.environ)
    base, report = execute(root, shard_index=shard)
    assert report["status"] == "passed", report
    assert report["tests"]["tests"] == (4 if shard is None else 2)
    assert len({item["temporary_directory"] for item in report["groups"]}) == 2
    for item in report["groups"]:
        temporary = Path(item["temporary_directory"])
        record = groups.child_temporary_receipt(temporary.parent / "runtime-environment.json",
                                                temporary, item["pid"], removed=True)
        assert record["temporary_environment"] == {key: str(temporary) for key in groups.TEMPORARY_ENVIRONMENT_KEYS}
        assert item["temporary_directory_removed_after_stop"] is True
        assert item["command"][-len(item["selectors"]):] == item["selectors"]
    assert max(item["start_seconds"] for item in report["groups"]) < min(item["stop_seconds"] for item in report["groups"])
    assert dict(os.environ) == inherited
    assert sentinel.read_text() == "preserve"
    assert (base / "combined.xml").is_file()


@pytest.mark.parametrize("attack", ["relative", "traversal", "missing", "file", "link", "linked-parent"])
def test_temporary_root_refuses_foreign_or_ambiguous_paths(tmp_path, attack):
    real = tmp_path / "real"
    real.mkdir()
    sentinel = real / "foreign-data"
    sentinel.write_text("preserve")
    path = tmp_path / "runtime-temp"
    path.mkdir()
    if attack == "relative":
        path = Path("runtime-temp")
    elif attack == "traversal":
        path = tmp_path / "runtime-temp/../real"
    elif attack == "missing":
        path.rmdir()
    elif attack == "file":
        path.rmdir()
        path.write_text("file")
    elif attack == "link":
        path.rmdir()
        path.symlink_to(real, target_is_directory=True)
    else:
        linked = tmp_path / "linked"
        linked.symlink_to(tmp_path, target_is_directory=True)
        path = linked / "runtime-temp"
    with pytest.raises(ValueError, match="SDK temporary"):
        groups.environment(tmp_path, temporary=path)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("optimize", ["0", "1"])
@pytest.mark.parametrize("attack", ["TMPDIR", "TMP", "TEMP", "missing-env", "nonempty", "reused-receipt", "linked-root"])
def test_actual_bootstrap_refuses_invalid_temp_before_pytest_even_when_optimized(tmp_path, optimize, attack):
    child = tmp_path / "group-0"
    child.mkdir()
    temporary = child / "runtime-temp"
    temporary.mkdir()
    receipt = child / "runtime-environment.json"
    environment = {**os.environ, "PYTHONOPTIMIZE": optimize,
                   **{key: str(temporary) for key in groups.TEMPORARY_ENVIRONMENT_KEYS}}
    if attack in groups.TEMPORARY_ENVIRONMENT_KEYS:
        environment[attack] = str(tmp_path)
    elif attack == "missing-env":
        environment.pop("TMPDIR")
    elif attack == "nonempty":
        (temporary / "reused").write_text("preserve")
    elif attack == "reused-receipt":
        receipt.write_text("preserve")
    else:
        temporary.rmdir()
        temporary.symlink_to(tmp_path, target_is_directory=True)
    result = subprocess.run([sys.executable, "-c", groups.SDK_CHILD_BOOTSTRAP, str(receipt), "--help"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "pytest" not in result.stdout
    if attack == "reused-receipt":
        assert receipt.read_text() == "preserve"
    else:
        assert not receipt.exists()
    assert not list(child.glob("pytest*"))


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("pid", True), ("pid", 124), ("tempfile_directory", "/tmp"),
    ("owned_directory", "/tmp"), ("canonical_directory", "/tmp"),
    ("no_symlink_ancestors", 1), ("empty_at_bootstrap", 1),
])
def test_actual_temp_receipt_rejects_rehashed_false_runtime_facts(tmp_path, field, value):
    temporary = tmp_path / "runtime-temp"
    temporary.mkdir()
    record = {"schema": 1, "pid": 123, "temporary_environment": {
        key: str(temporary) for key in groups.TEMPORARY_ENVIRONMENT_KEYS},
        "owned_directory": str(temporary), "tempfile_directory": str(temporary),
        "canonical_directory": str(temporary), "no_symlink_ancestors": True,
        "empty_at_bootstrap": True}
    record[field] = value
    path = tmp_path / "runtime-environment.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Actual SDK child temporary"):
        groups.child_temporary_receipt(path, temporary, 123)


def test_owned_cleanup_removes_only_its_original_root_and_does_not_follow_nested_links(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "keep"
    sentinel.write_text("preserve")
    temporary = tmp_path / "runtime-temp"
    temporary.mkdir()
    metadata = temporary.stat()
    (temporary / "fixture").write_text("owned")
    (temporary / "foreign-link").symlink_to(foreign, target_is_directory=True)
    groups.remove_child_temporary_directory(temporary, (metadata.st_dev, metadata.st_ino))
    assert not temporary.exists() and sentinel.read_text() == "preserve"


@pytest.mark.parametrize("attack", ["replaced", "link", "unsafe-remover"])
def test_owned_cleanup_refuses_replaced_or_foreign_directory(tmp_path, monkeypatch, attack):
    temporary = tmp_path / "runtime-temp"
    temporary.mkdir()
    metadata = temporary.stat()
    original = tmp_path / "original"
    temporary.rename(original)
    if attack == "link":
        temporary.symlink_to(original, target_is_directory=True)
    else:
        temporary.mkdir()
        (temporary / "foreign").write_text("preserve")
    if attack == "unsafe-remover":
        monkeypatch.setattr(groups.shutil.rmtree, "avoids_symlink_attacks", False)
    with pytest.raises(ValueError, match="SDK temporary"):
        groups.remove_child_temporary_directory(temporary, (metadata.st_dev, metadata.st_ino))
    assert original.is_dir() and temporary.exists()
    if attack != "link":
        assert (temporary / "foreign").read_text() == "preserve"


def test_failed_actual_children_cleanup_owned_temp_and_keep_failure_evidence(tmp_path):
    body = '''import tempfile
from pathlib import Path
def test_failure():
    (Path(tempfile.gettempdir()) / "large-owned-fixture").write_bytes(b"x" * 65536)
    assert False, "intentional failure after creating an owned fixture"
'''
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    base, report = execute(root)
    assert report["status"] == "failed" and report["groups"]
    for item in report["groups"]:
        assert item["temporary_directory_removed_after_stop"] is True
        assert not Path(item["temporary_directory"]).exists()
        assert (base / "sdk-groups" / item["log"]).is_file()
    assert "intentional failure" in "".join((base / "sdk-groups" / item["log"]).read_text() for item in report["groups"])


def test_cleanup_error_fails_gate_and_still_cleans_the_other_owned_child(tmp_path, monkeypatch):
    root = repository(tmp_path / "repo")
    remove = groups.remove_child_temporary_directory
    calls = []

    def failed_first(path, identity):
        calls.append(path)
        if path.parent.name == "group-0":
            raise OSError("intentional owned cleanup refusal")
        remove(path, identity)

    monkeypatch.setattr(groups, "remove_child_temporary_directory", failed_first)
    base, report = execute(root)
    assert report["status"] == "failed" and "owned temporary cleanup failed" in report["error"]
    assert len(calls) == 2
    first, second = report["groups"]
    assert first["temporary_directory_removed_after_stop"] is False
    assert first["temporary_cleanup_error"] == "OSError: intentional owned cleanup refusal"
    assert Path(first["temporary_directory"]).is_dir()
    assert second["temporary_directory_removed_after_stop"] is True
    assert not Path(second["temporary_directory"]).exists()
    for item in report["groups"]:
        assert (base / "sdk-groups" / item["log"]).is_file()
        assert (base / "sdk-groups" / item["junit"]).is_file()
