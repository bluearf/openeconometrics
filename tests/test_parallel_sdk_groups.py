"""Real subprocess and tamper checks for complete three-group SDK execution."""

import importlib.util
import builtins
from collections import Counter
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
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
timing_spec = importlib.util.spec_from_file_location("sdk_timing_plugin", ROOT / "scripts/pytest_gate_timings.py")
timing_plugin = importlib.util.module_from_spec(timing_spec)
timing_spec.loader.exec_module(timing_plugin)
# These small real-subprocess fixtures retain their declared toy partition.
# Production scope/order is checked separately against the actual gate source.
TOY_FIXED = (("tests/test_control_stream_acceptance.py",),
             ("tests/test_streaming_control_function_engine.py",))
FIXED_SELECTORS = tuple(item for group in groups.FIXED_GROUPS for item in group)
SELECTORS = ["tests/test_before.py", TOY_FIXED[1][0], TOY_FIXED[0][0], "tests/test_after.py"]
# Incoming two-child probe labels retain their exact selector identities.
TOY_HEAVY = (TOY_FIXED[1][0], TOY_FIXED[0][0])


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
        fixed_groups=TOY_FIXED,
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
    children = [tmp_path / f"group-{index}" for index in range(3)]
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
    assert [item["selectors"] for item in report["groups"]] == groups.split_groups(SELECTORS, TOY_FIXED)
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
    # Keep the incoming historical parameter IDs while checking the current three-child allocation.
    assert threads == min(2, max(1, math.floor(effective / 2)))
    assert budget["effective_cpus"] == effective
    assert budget["threads_per_child"] == min(2, max(1, math.floor(effective / 3)))
    assert budget["groups"] == 3 and budget["cgroup"]["version"] == version
    assert budget["max_concurrent_children"] == min(3, max(1, math.floor(effective)))
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
    assert threads == min(2, max(1, math.floor(affinity / 2)))
    assert budget["effective_cpus"] == affinity
    assert budget["threads_per_child"] == min(2, max(1, math.floor(affinity / 3)))


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
    assert [item["selectors"] for item in report["groups"]] == groups.split_groups(SELECTORS, TOY_FIXED)
    assert all(item["environment"] == {**groups.child_environment_contract(1), **{
        key: item["temporary_directory"] for key in groups.TEMPORARY_ENVIRONMENT_KEYS}}
        for item in report["groups"])
    assert "intentional child failure" in (base / "sdk-groups/group-2/pytest.log").read_text()
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
    assert [child["selectors"] for child in report["groups"]] == groups.split_groups(SELECTORS, TOY_FIXED)
    assert [child["tests"]["tests"] for child in report["groups"]] == [1, 1, 2]
    for child in report["groups"]:
        rows, _ = groups.read_child(base / "sdk-groups" / child["junit"],
                                   base / "sdk-groups" / child["phase_timings"], child["selectors"])
        assert len(rows) == len(child["selectors"])


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
    children = report["groups"]
    assert len(children) == 3
    workers = report["cpu_budget"]["max_concurrent_children"]
    events = sorted((value, delta) for child in children for value, delta in (
        (child["start_seconds"], 1), (child["stop_seconds"], -1)))
    active, maximum = 0, 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
        assert 0 <= active <= workers
    assert active == 0 and maximum == workers
    if workers == 3:
        assert max(item["start_seconds"] for item in children) < min(item["stop_seconds"] for item in children)
    assert len({item["command"][item["command"].index("--basetemp") + 1] for item in children}) == 3
    assert len({item["command"][item["command"].index("-o") + 1] for item in children}) == 3


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
    assert any((base / "sdk-groups" / f"group-{i}/pytest.log").stat().st_size for i in range(3))


def test_future_selector_is_automatically_in_complete_complement():
    selectors = ["tests/test_future.py", *SELECTORS]
    assert groups.split_groups(selectors, TOY_FIXED)[0] == list(TOY_FIXED[0])
    assert groups.split_groups(selectors, TOY_FIXED)[1] == list(TOY_FIXED[1])
    assert groups.split_groups(selectors, TOY_FIXED)[2] == [selectors[0], SELECTORS[0], SELECTORS[-1]]


def test_preregistered_three_groups_retain_actual_source_scope_and_order():
    selectors = groups.selector_contract(ROOT)
    expected_fixed = (('tests/test_control_stream_acceptance.py',), ('tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_function_stream_state.py', 'tests/test_nested_logit_independent.py', 'tests/test_parallel_sdk_groups.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py'))
    current_fixed = (expected_fixed[0], expected_fixed[1] + ('tests/test_bayesian_var_conjugate.py', 'tests/test_bayesian_var_sbc_protocol.py', 'tests/test_bayesian_var_public_integration.py', 'tests/test_bayesian_var_public_admission_v2.py', 'tests/test_editor_catalog_intern_v2.py'))
    assert groups.FIXED_GROUPS == current_fixed
    assert expected_fixed[1][:5] == (
        "tests/test_econ_saved_prediction_linear.py",
        "tests/test_control_function_common_prediction.py",
        "tests/test_streaming_control_function_engine.py",
        "tests/test_control_function_stream_state.py",
        "tests/test_nested_logit_independent.py",
    )
    assert expected_fixed[1][5] == "tests/test_parallel_sdk_groups.py"
    assert selectors == gate.TESTS and len(selectors) == 160
    partition = groups.split_groups(selectors)
    assert partition[:2] == [[item for item in selectors if item in group] for group in current_fixed]
    assert partition[2] == [item for item in selectors if item not in FIXED_SELECTORS]
    assert [len(partition[0]), len(partition[1])] == [1, 23]
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan.json").read_text())
    assert plan["group_count"] == 3 and plan["fixed_groups"] == [list(group) for group in expected_fixed]
    baseline = plan["selected_tests"]
    assert selectors[:138] == baseline
    assert selectors[145:146] == ["tests/test_multivariate_score_uncertainty.py"]
    assert selectors[138:142] == ["tests/test_weighted_binary.py", "tests/test_econ_glm.py", "tests/test_econ_glm_oracle.py", "tests/test_econ_saved_prediction_categories.py"]
    assert plan["effective_groups"] == groups.split_groups(baseline, expected_fixed)
    assert list(map(len, partition)) == [1, 23, 136]
    assert plan["environment"] == groups.GATE_ENVIRONMENT
    assert plan["timeout_seconds"] == 900
    assert plan["selected_test_scope_sha256"] == hashlib.sha256(
        json.dumps(baseline, separators=(",", ":")).encode()).hexdigest()
    assert set(sum(partition, [])) == set(selectors)
    assert sum(map(len, partition)) == len(selectors)
    # Within a fixed group, membership never replaces original source order.
    assert groups.split_groups(selectors, tuple(tuple(reversed(group)) for group in current_fixed)) == partition

    assert expected_fixed[1][6:] == tuple(gate.TESTS[126:138])



def test_production_four_file_partition_retains_actual_source_scope_and_order():
    # Retain the incoming case name and its complete heavy-file scope. Legacy
    # execution now uses three groups; distributed shards use four node groups.
    selectors = groups.selector_contract(ROOT)
    expected_heavy = (
        "tests/test_econ_saved_prediction_linear.py",
        "tests/test_control_function_common_prediction.py",
        "tests/test_streaming_control_function_engine.py",
        "tests/test_control_stream_acceptance.py",
    )
    assert groups.EXPENSIVE == expected_heavy
    assert selectors == gate.TESTS and len(selectors) == 160
    partition = groups.split_groups(selectors)
    assert list(map(len, partition)) == [1, 23, 136]
    assert Counter(item for group in partition for item in group) == Counter(selectors)
    assert set(expected_heavy) <= set(sum(partition[:2], []))
    original_order = [item for item in selectors if item in expected_heavy]
    actual = [item for group in partition for item in group if item in expected_heavy]
    assert sorted(actual, key=selectors.index) == original_order


def test_mprobit_survey_and_categorical_merge_preserves_all_complete_batches():
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
    }
    assert merged_batches <= set(selectors)
    assert len(selectors) == len(set(selectors)) == 160
    first, second, other = groups.split_groups(selectors)
    assert [len(first), len(second), len(other)] == [1, 23, 136]
    assert merged_batches <= set(other)
    assert set(first).isdisjoint(second)
    assert set(first + second).isdisjoint(other)
    assert set(first + second + other) == set(selectors)


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
        "tests/test_econ_postest_suest.py",
        "tests/test_econ_suest_extended.py",
        "tests/test_econ_suest_binomial_commands.py",
        "tests/test_verify_team_cleanup.py",
        "tests/test_verify_windows_cloud_ui_fixture.py",
    }
    assert merged_batches <= set(selectors)
    assert len(selectors) == len(set(selectors)) == 160
    first, second, other = groups.split_groups(selectors)
    assert [len(first), len(second), len(other)] == [1, 23, 136]
    assert merged_batches <= set(other)
    assert set(first).isdisjoint(second)
    assert set(first + second).isdisjoint(other)
    assert set(first + second + other) == set(selectors)


def test_production_future_selectors_remain_complete_and_in_original_complement_order():
    selectors = groups.selector_contract(ROOT)
    midpoint = len(selectors) // 2
    future = [
        "tests/test_future_before.py", *selectors[:midpoint],
        "tests/test_future_middle.py", *selectors[midpoint:], "tests/test_future_after.py",
    ]
    partition = groups.split_groups(future)
    assert partition[:2] == groups.split_groups(selectors)[:2]
    assert partition[2] == [item for item in future if item not in FIXED_SELECTORS]
    assert sum(map(len, partition)) == len(future) == len(selectors) + 3
    assert set(sum(partition, [])) == set(future)


@pytest.mark.parametrize("missing", FIXED_SELECTORS)
def test_production_missing_any_heavy_source_file_is_rejected(missing):
    selectors = [item for item in groups.selector_contract(ROOT) if item != missing]
    with pytest.raises(ValueError, match="Fixed SDK selectors missing"):
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
        groups.split_groups(selectors, TOY_FIXED)


def test_missing_heavy_selector_cannot_silently_change_partition():
    with pytest.raises(ValueError, match="Fixed SDK selectors missing"):
        groups.split_groups([SELECTORS[0], SELECTORS[-1]], TOY_FIXED)


@pytest.mark.parametrize('fixed', [
    (), (TOY_FIXED[0],), (*TOY_FIXED, ('tests/test_before.py',)),
    ((), TOY_FIXED[1]), (TOY_FIXED[0], TOY_FIXED[0]),
    (TOY_FIXED[0], ('tests/test_missing.py',)),
])
def test_fixed_groups_cannot_drop_duplicate_or_add_a_worker_scope(fixed):
    with pytest.raises(ValueError):
        groups.split_groups(SELECTORS, fixed)


@pytest.mark.parametrize('attack', ['missing_third', 'two_group_plan', 'duplicate_third'])
def test_actual_three_child_merge_cannot_accept_an_incomplete_or_replayed_child(
    actual_pass, tmp_path, attack
):
    children = copied_children(actual_pass, tmp_path)
    partition = groups.split_groups(SELECTORS, TOY_FIXED)
    if attack == 'missing_third':
        children.pop()
    elif attack == 'two_group_plan':
        partition = [partition[0] + partition[1], partition[2]]
    else:
        children[2] = children[1]
    with pytest.raises(ValueError):
        groups.merge_evidence(children, SELECTORS, partition,
                              tmp_path / 'merged.xml', tmp_path / 'merged.jsonl')
    assert not (tmp_path / 'merged.xml').exists()


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
            groups.split_groups(SELECTORS, TOY_FIXED),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_selector_missing_all_real_cases_is_rejected(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    child = children[2]
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
            groups.split_groups(SELECTORS, TOY_FIXED),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_cross_file_reordering_rejected_even_if_xml_and_phases_agree(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    child = children[2]
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
            groups.split_groups(SELECTORS, TOY_FIXED),
            tmp_path / "merged.xml",
            tmp_path / "merged.jsonl",
        )


def test_partition_cannot_omit_repeat_or_move_a_selector(actual_pass, tmp_path):
    children = copied_children(actual_pass, tmp_path)
    partition = groups.split_groups(SELECTORS, TOY_FIXED)
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


def test_stop_reaps_exited_leader_before_signalling_its_group(tmp_path, monkeypatch):
    # Keep the incoming historical ID: an empty group is reaped after proof;
    # no signal occurs after its original session anchor is released.
    ready = tmp_path / "ready"
    process = groups._launch_sdk(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(ready)!r}).touch()"])
    try:
        if os.name == "posix":
            assert _sdk_lifecycle_exited(process) == 0
            assert process.returncode is None
            with monkeypatch.context() as patch:
                patch.setattr(os, "killpg", lambda *args: pytest.fail("Proved empty group must not receive a signal"))
                groups._stop([process])
            assert process._openecon_sdk_session_closed is True
        else:
            import _winapi
            assert _winapi.WaitForSingleObject(process._handle, 3000) == _winapi.WAIT_OBJECT_0
            assert ready.exists() and process.returncode is None
            groups._stop([process])
        assert process.returncode == 0
    finally:
        groups._stop([process])



@pytest.mark.parametrize("stubborn", [False, True])
def test_stop_cleans_live_worker_after_leader_exit(tmp_path, stubborn):
    ready, survived = tmp_path / "ready", tmp_path / "survived"
    worker = ("import os,pathlib,signal,time; from pathlib import Path; "
              + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if stubborn else "")
              + _atomic_sdk_pid_marker(ready)
              + f"time.sleep(6); Path({str(survived)!r}).touch(); time.sleep(30)")
    command = ([sys.executable, "-c", worker] if os.name == "nt" else
               [sys.executable, "-c", f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{worker!r}])"])
    process = groups._launch_sdk(command)
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        worker_pid = int(ready.read_text())
        if os.name == "posix":
            assert _sdk_lifecycle_exited(process) == 0 and process.returncode is None
        groups._stop([process])
        if os.name == "posix":
            assert worker_pid != process.pid
            actual = subprocess.run(["/bin/ps", "-p", str(worker_pid), "-o", "stat="],
                                    capture_output=True, text=True, timeout=3)
            assert actual.returncode in (0, 1)
            assert not any(state and state[0] not in "ZX" for state in actual.stdout.split()), (
                "Actual descendant writer survived completed group cleanup")
        time.sleep(1.1)
        assert not survived.exists(), "A worker survived owned process cleanup"
        assert process.returncode is not None
    finally:
        groups._stop([process])



def test_group_permission_retry_requires_success_or_absence(tmp_path, monkeypatch):
    process = groups._launch_sdk([sys.executable, "-c", "import time; time.sleep(30)"])
    calls = []
    try:
        if os.name == "posix":
            actual = os.killpg

            def actual_last_recipient_exit(pgid, signum):
                calls.append((pgid, signum))
                assert pgid == process.pid and process.returncode is None
                actual(pgid, signum)
                assert _sdk_lifecycle_exited(process) == -signum
                raise PermissionError(errno.EPERM, "actual last-recipient zombie transition")

            with monkeypatch.context() as patch:
                patch.setattr(os, "killpg", actual_last_recipient_exit)
                groups._stop([process])
            # Retain the original success-or-absence role, with real absence.
            # A still-live recipient denial must not be silently retried into PASS.
            assert calls == [(process.pid, groups.signal.SIGTERM)]
            assert process._openecon_sdk_session_closed is True
        else:
            groups._stop([process])
        assert process.returncode is not None
    finally:
        groups._stop([process])



def test_persistent_group_denial_fails_and_still_stops_other_owned_child(monkeypatch):
    processes = [groups._launch_sdk([sys.executable, "-c", "import time; time.sleep(30)"]) for _ in range(2)]
    try:
        with monkeypatch.context() as patch:
            if os.name == "posix":
                actual = groups._signal_sdk_group

                def denied_first(process, signum, deadline):
                    if process is processes[0]:
                        raise PermissionError(errno.EPERM, "persistent owned group denial")
                    return actual(process, signum, deadline)

                patch.setattr(groups, "_signal_sdk_group", denied_first)
            else:
                def denied_handle():
                    raise PermissionError(errno.EPERM, "persistent owned handle denial")

                patch.setattr(processes[0], "terminate", denied_handle)
                patch.setattr(processes[0], "kill", denied_handle)
            with pytest.raises(PermissionError, match="persistent owned (group|handle) denial"):
                groups._stop(processes)
            if os.name == "posix":
                assert processes[0]._openecon_sdk_session_closed is False
                assert processes[0].returncode is None and groups._sdk_exit(processes[0]) is None
                assert processes[1]._openecon_sdk_session_closed is True
            else:
                assert processes[0].poll() is None
            assert processes[1].returncode is not None
    finally:
        groups._stop(processes)



@pytest.mark.parametrize("unsafe_temporary_remover,fd_remover_absent", [(False, False), (True, False), (False, True)])
def test_process_cleanup_denial_preserves_failed_report_and_child_logs(tmp_path, monkeypatch,
                                                                     unsafe_temporary_remover, fd_remover_absent):
    path = tmp_path / "repo"
    ready = path / "persistent-denial-ready"
    # Keep the first child actually alive on both platforms. Reaping an already
    # exited Windows handle cannot meaningfully exercise native signal denial.
    other = ("import pathlib,time\n"
             "def test_bad():\n"
             f"    marker=pathlib.Path({str(ready)!r})\n"
             "    deadline=time.monotonic()+5\n"
             "    while not marker.exists():\n"
             "        assert time.monotonic()<deadline\n"
             "        time.sleep(.01)\n"
             "    assert False, 'retained failure'\n")
    root = repository(path, {item: other for item in SELECTORS})
    (root / TOY_FIXED[0][0]).write_text(
        "import pathlib,time,pytest\n"
        "@pytest.fixture(autouse=True)\n"
        "def actual_live_writer():\n"
        f"    pathlib.Path({str(ready)!r}).touch()\n"
        "    yield\n"
        "    time.sleep(30)\n"
        "def test_bad(): assert False, 'retained failure'\n")
    owned, actual_launch = [], groups._launch_sdk
    try:
        with monkeypatch.context() as patch:
            def retained_launch(*args, **kwargs):
                process = actual_launch(*args, **kwargs)
                owned.append(process)
                if os.name == "nt" and len(owned) == 1:
                    def denied_handle():
                        raise PermissionError(errno.EPERM, "persistent receipt cleanup denial")

                    patch.setattr(process, "terminate", denied_handle)
                    patch.setattr(process, "kill", denied_handle)
                return process

            patch.setattr(groups, "_launch_sdk", retained_launch)
            if os.name == "posix":
                actual_signal = groups._signal_sdk_group

                def denied_first(process, signum, deadline):
                    if process is owned[0]:
                        raise PermissionError(errno.EPERM, "persistent receipt cleanup denial")
                    return actual_signal(process, signum, deadline)

                patch.setattr(groups, "_signal_sdk_group", denied_first)
            if fd_remover_absent:
                patch.setattr(groups.shutil.rmtree, "avoids_symlink_attacks", False)
            if unsafe_temporary_remover:
                def refused_temporary_cleanup(path, identity):
                    raise ValueError("SDK temporary cleanup refused a replaced directory or unsafe remover")

                patch.setattr(groups, "remove_child_temporary_directory", refused_temporary_cleanup)
            base, report = execute(root)
            assert report["status"] == "failed" and "SDK child process failed" in report["error"]
            assert report["execution_error"] == report["error"]
            assert "PermissionError" in report["process_cleanup_error"]
            persisted = json.loads((base / "sdk-groups/report.json").read_text())
            for key in ("status", "error", "execution_error", "process_cleanup_error", "temporary_cleanup_errors"):
                assert persisted[key] == report[key]
            assert "retained failure" in "".join((base / "sdk-groups" / item["log"]).read_text()
                                                for item in report["groups"])
            first = next(item for item in report["groups"] if item["pid"] == owned[0].pid)
            assert first["temporary_directory_removed_after_stop"] is False
            assert "verified owned process-group shutdown" in first["temporary_cleanup_error"]
            assert Path(first["temporary_directory"]).is_dir()
            failures = report["temporary_cleanup_errors"]
            removal_refused = unsafe_temporary_remover or not groups.shutil.rmtree.avoids_symlink_attacks
            assert len(failures) == (len(report["groups"]) if removal_refused else 1)
            for item in report["groups"]:
                assert (base / "sdk-groups" / item["log"]).is_file()
                if item is first or removal_refused:
                    assert item["temporary_directory_removed_after_stop"] is False
                    assert Path(item["temporary_directory"]).is_dir()
                    assert {"group_index": item["index"], "directory": item["temporary_directory"],
                            "error": item["temporary_cleanup_error"]} in failures
                    if item is not first:
                        assert "unsafe remover" in item["temporary_cleanup_error"]
                else:
                    assert item["temporary_directory_removed_after_stop"] is True
    finally:
        groups._stop(owned)



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
# TOY three-worker scheduling must launch every real child before the watchdog.
helper.detect_cpu_budget=lambda: {{"system": "Darwin", "host_cpus": 3,
    "affinity_cpus": 3, "effective_cpus": 3., "groups": 3,
    "max_concurrent_children": 3, "threads_per_child": 1, "cgroup": {{
        "status": "not_applicable", "version": None, "views": [],
        "probes": [], "limits": [], "error": None}}}}
root=Path({str(root)!r});base=root/'receipts/outer';base.mkdir(parents=True)
report=helper.run_groups(root=root,python=Path(sys.executable),selectors={SELECTORS!r},directory=base/'sdk-groups',junit=base/'all.xml',timings=base/'all.jsonl',timeout=60,fixed_groups={TOY_FIXED!r})
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
    assert len(pid_files) == 3, "All three real child processes must have spawned a grandchild"
    time.sleep(2.5)
    assert not list((root / "tests").glob("*.orphan")), (
        "A grandchild survived process-group cleanup"
    )


def test_production_cli_cannot_extend_or_shrink_900_deadline(tmp_path):
    modes = (["--local"], ["--execution", str(tmp_path / "execution.json"),
                           "--component-sdk-version", "3.13", "--shard-index", "0"])
    for mode in modes:
        for timeout in (899, 901):
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
                    *mode,
                    "--timeout",
                    str(timeout),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode != 0 and "invalid choice" in result.stderr
        assert not (tmp_path / "groups").exists()

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


def local_cli_repository(path):
    """Run the production main and partition against a small committed scope."""
    root = repository(path)
    with (root / ".gitignore").open("a") as stream:
        stream.write(".venv*\n")
    selectors = [*SELECTORS, *(item for item in FIXED_SELECTORS if item not in SELECTORS)]
    for selector in selectors:
        if not (root / selector).exists():
            shutil.copyfile(root / SELECTORS[0], root / selector)
    (root / "scripts/verify_merge_candidate.py").write_text("TESTS=" + repr(selectors) + "\n")
    wrapper = root / "scripts/run_parallel_sdk_groups.py"
    wrapper.write_text(f"""import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('actual_helper',{str(ROOT / 'scripts/run_parallel_sdk_groups.py')!r})
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.ROOT=Path({str(root)!r})
raise SystemExit(helper.main())
""")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "production local CLI fixture"], cwd=root, check=True)
    return root, selectors


def local_cli(root, base, *, started=None, extra=(), env=None, clock_offset=0):
    base.mkdir(parents=True)
    started = time.monotonic() + clock_offset if started is None else started
    options = groups.environment(root) if env is None else env.copy()
    options.update(OPENECON_SDK_PARENT_STARTED_MONOTONIC=str(started),
                   OPENECON_SDK_PARENT_DEADLINE_MONOTONIC=str(started + 900))
    command = [sys.executable, "scripts/run_parallel_sdk_groups.py", "--python", sys.executable,
               "--directory", str(base / "sdk-groups"), "--junitxml", str(base / "combined.xml"),
               "--gate-timings", str(base / "combined.jsonl"), "--local", *extra]
    if clock_offset:
        # A fresh hosted machine can have uptime below the 900-second budget.
        # Shift only the clock origin in the real CLI process: elapsed time,
        # the production admission guard, and the fixed deadline are unchanged.
        command = [sys.executable, "-c",
                   "import runpy,sys,time; actual=time.monotonic; "
                   f"time.monotonic=lambda:actual()+{clock_offset!r}; "
                   "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                   *command[1:]]
    return subprocess.run(command, cwd=root, env=options, capture_output=True, text=True, timeout=15)


def test_real_local_cli_has_full_source_scope_and_null_hosted_identity(tmp_path, monkeypatch):
    root, selectors = local_cli_repository(tmp_path / "repo")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nonexistent")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_plugin")
    base = root / "receipts/local"
    result = local_cli(root, base, env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((base / "sdk-groups/report.json").read_text())
    assert manifest["source_commit"] == groups.source_identity(root)["source_commit"]
    assert manifest["source_tree"] == groups.source_identity(root)["source_tree"]
    assert manifest["execution"] is manifest["execution_sha256"] is manifest["component_sdk_version"] is None
    assert manifest["timeout_seconds"] == 900 and 0 < manifest["seconds"] < 900
    assert manifest["selected_tests"] == selectors and manifest["tests"]["tests"] == 4 * len(selectors)
    assert [item["selectors"] for item in manifest["groups"]] == groups.split_groups(selectors)
    assert len((base / "combined.jsonl").read_text().splitlines()) == 12 * len(selectors)
    for item in manifest["groups"]:
        assert item["execution"] is None and item["exit_code"] == 0
        for key in ("log", "junit", "phase_timings", "collection"):
            path = base / "sdk-groups" / item[key]
            assert path.is_file() and item[key + "_sha256"] == groups.digest(path)


def test_complete_local_gate_runs_real_group_cli_for_both_bound_interpreters(tmp_path, monkeypatch):
    # Both toy paths use this test's interpreter. This proves CLI wiring and
    # full versus partial scope; the fixture is not dual-SDK scientific proof.
    root, selectors = local_cli_repository(tmp_path / "repo")
    paths = []
    for version in ("311", "313"):
        environment = root / (".venv" + version)
        environment.symlink_to(sys.prefix, target_is_directory=True)
        paths.append(environment / "bin/python")
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "TESTS", selectors)
    actual_output = subprocess.check_output

    def version_output(command, **kwargs):
        if len(command) == 3 and command[1] == "-c" and "sys.version_info" in command[2]:
            return "3.11\n" if "/.venv311/" in command[0] else "3.13\n"
        return actual_output(command, **kwargs)

    monkeypatch.setattr(gate.subprocess, "check_output", version_output)
    actual_step = gate.run_step

    def common_fixture(name, command, directory, timeout, **kwargs):
        if name.startswith("sdk-"):
            return actual_step(name, command, directory, timeout, **kwargs)
        return dict(name=name, command=command, status="passed", exit_code=0,
                    seconds=.001, log=name + ".log", log_sha256="0" * 64)

    monkeypatch.setattr(gate, "run_step", common_fixture)
    monkeypatch.setattr(gate, "package_manifest", lambda directory: [])
    args = SimpleNamespace(directory=root / "receipts/full-local", python=paths,
                           timeout=900, local_sdk_groups=True, ci_sdk_version=None,
                           publish_repo=None, pr=None)
    assert gate.run(args) == 0
    report = json.loads((args.directory / "report.json").read_text())
    assert report["source_commit"] == groups.source_identity(root)["source_commit"]
    assert report["component_mode"] is False and report["component_sdk_version"] is None
    assert report["sdk_versions_executed"] == ["3.11", "3.13"]
    assert report["selected_tests"] == selectors
    for version in report["sdk_versions_executed"]:
        manifest = json.loads((args.directory / f"sdk-{version}-groups/report.json").read_text())
        assert manifest["python"] == str(paths[0 if version == "3.11" else 1])
        assert manifest["tests"] == dict(tests=4 * len(selectors), failures=0, errors=0, skipped=0)
        assert manifest["execution"] is manifest["component_sdk_version"] is None
        assert groups.collected_nodes(args.directory / f"sdk-{version}-groups/group-0/pytest-collection.jsonl")
        assert len((args.directory / f"pytest-{version}-timings.jsonl").read_text().splitlines()) == 12 * len(selectors)


@pytest.mark.parametrize("age", [901, 899.98])
def test_real_local_cli_cannot_restart_expired_or_nearly_spent_900_budget(tmp_path, age):
    root, _ = local_cli_repository(tmp_path / "repo")
    base = root / "receipts/deadline"
    origin = 1_000_000
    result = local_cli(root, base, started=time.monotonic() + origin - age,
                       clock_offset=origin)
    assert result.returncode == 1
    manifest = json.loads((base / "sdk-groups/report.json").read_text())
    assert manifest["status"] == "failed" and "deadline" in manifest["error"]
    assert manifest["timeout_seconds"] == 900 and manifest["seconds"] >= 900
    assert "tests" not in manifest and not (base / "combined.xml").exists()


@pytest.mark.parametrize("extra,reason", [
    (["--execution", "missing.json"], "not allowed with argument"),
    (["--component-sdk-version", "3.11"], "cannot claim"),
    (["--timeout", "899"], "invalid choice"),
    (["--timeout", "901"], "invalid choice"),
    (["--shard-index", "0"], "cannot claim"),
    (["--shard-index", "1"], "cannot claim"),
    (["--timeout", "1800"], "900-second deadline"),
])
def test_local_cli_rejects_hosted_identity_or_changed_budget_before_children(tmp_path, extra, reason):
    root, _ = local_cli_repository(tmp_path / "repo")
    base = root / "receipts/rejected"
    result = local_cli(root, base, extra=extra)
    assert result.returncode != 0 and reason in result.stderr
    assert not (base / "sdk-groups").exists()


def test_hosted_cli_still_requires_component_before_reading_execution(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_parallel_sdk_groups.py"),
                             "--python", sys.executable, "--directory", str(tmp_path / "groups"),
                             "--junitxml", str(tmp_path / "out.xml"), "--gate-timings", str(tmp_path / "out.jsonl"),
                             "--execution", str(tmp_path / "missing.json")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "requires --component-sdk-version" in result.stderr
    assert not (tmp_path / "groups").exists()


@pytest.mark.parametrize("claim", [{"component": "3.11"}, {"execution_sha256": "0" * 64}])
def test_internal_local_runner_cannot_claim_hosted_identity(tmp_path, claim):
    root = repository(tmp_path / "repo")
    with pytest.raises(ValueError, match="cannot claim a hosted"):
        execute(root, **claim)
    assert not list((root / "receipts/run/sdk-groups").glob("group-*"))


def test_source_selector_contract_reads_future_additions_without_importing_runtime(tmp_path):
    root = repository(tmp_path / "repo")
    values = [*SELECTORS, "tests/test_future.py"]
    (root / "scripts/verify_merge_candidate.py").write_text(
        "TESTS=" + repr(values) + "\nraise RuntimeError('must not import')\n"
    )
    assert groups.selector_contract(root) == values
    assert groups.split_groups(values, TOY_FIXED)[2][-1] == "tests/test_future.py"


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
            fixed_groups=TOY_FIXED,
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
            groups.split_groups(SELECTORS, TOY_FIXED),
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
            groups.split_groups(SELECTORS, TOY_FIXED),
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
    assert groups.ADVISORY_NODE_SECONDS[exhaustive] > 0
    known = sorted(node for node in groups.ADVISORY_NODE_SECONDS if node != exhaustive)
    future = [f"tests/test_new_scope.py::test_future[{i}]" for i in range(4000)]
    nodes = [exhaustive, *known, *future]
    original = nodes.copy()
    plan = groups.make_partition_plan(nodes)
    partition = groups.partition_nodes(nodes)
    assert nodes == original
    assert partition[plan["assignments"][0]].count(exhaustive) == 1
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
    assert selectors == gate.TESTS and len(selectors) == 160
    assert selectors[:3] == ["tests/test_sequential_kernels.py", "tests/test_sequential_oracles.py",
                             "tests/test_sequential_contracts.py"]
    assert selectors[3:6] == ["tests/test_mprobit.py", "tests/test_mprobit_postestimation.py",
                             "tests/test_mprobit_independent.py"]
    assert {
        "tests/test_causal_confidence_sets.py", "tests/test_causal_multiarm_neyman.py",
        "tests/test_causal_effect_distribution.py", "tests/test_causal_confidence_delivery.py",
        "tests/test_sdist_packaging.py",
        "tests/test_verify_team_cleanup.py",
        "tests/test_verify_windows_cloud_ui_fixture.py",
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
    assert selectors[119:124] == ["tests/test_twostep_kernel.py", "tests/test_twostep_reference.py",
                                  "tests/test_twostep_contract.py", "tests/test_twostep_scale.py",
                                  "tests/test_twostep_adaptive_reference.py"]
    assert selectors[124:126] == ["tests/test_survey_deff.py", "tests/test_survey_inference.py"]
    assert sum(node.startswith("tests/test_survey_deff.py::") for node in nodes) == 84
    assert sum(node.startswith("tests/test_survey_inference.py::") for node in nodes) == 108
    assert all(any(node.startswith(selector + "::") for node in nodes) for selector in selectors[119:124])
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
    assert sum(child.count(exhaustive) for child in partition) == 1

    assert selectors[126:138] == ['tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py']
    assert selectors[142:144] == ['tests/test_survey_fully_stratified_four_stage.py', 'tests/test_survey_fully_stratified_four_stage_safety.py']
    assert selectors[144:145] == ["tests/test_confidence_sequences.py"]
    assert sum(node.startswith(selectors[142] + "::") for node in nodes) == 57
    assert sum(node.startswith(selectors[143] + "::") for node in nodes) == 99
    assert all(any(node.startswith(selector + "::") for node in nodes) for selector in selectors[126:])
    assert selectors[145:146] == ["tests/test_multivariate_score_uncertainty.py"]
    assert selectors[138:142] == ["tests/test_weighted_binary.py", "tests/test_econ_glm.py", "tests/test_econ_glm_oracle.py", "tests/test_econ_saved_prediction_categories.py"]
    assert sum(node.startswith("tests/test_weighted_binary.py::") for node in nodes) == 111
    assert sum(node.startswith("tests/test_confidence_sequences.py::") for node in nodes) == 66



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
report=run_groups(root=root,python=Path(sys.executable),selectors={SELECTORS!r},directory=base/'sdk-groups',junit=base/'combined.xml',timings=base/'combined.jsonl',timeout=30,shard_index=shard)
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
    children = report["groups"]
    assert len(children) == (3 if shard is None else 2)
    assert len({child["pid"] for child in children}) == len(children)
    assert all(first["stop_seconds"] <= second["start_seconds"]
               for first, second in zip(children, children[1:]))
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
    # Every actual child holds a browser-style profile during its first case.
    # Unequal later file counts retain strict per-child before/during/after cleanup.
    participants = 3 if shard is None else 2
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
        if round_number == 1:
            marker = gate / f"live-{round_number}-{child.name}.json"
            pending = marker.with_suffix(".pending")
            pending.write_text(json.dumps({"profile": name}))
            pending.replace(marker)
            deadline = time.monotonic() + 8
            while True:
                records = [json.loads(path.read_text()) for path in gate.glob(f"live-{round_number}-*.json")]
                if len(records) == EXPECTED_PARTICIPANTS and all(Path(row["profile"]).is_dir() for row in records):
                    break
                assert time.monotonic() < deadline, "Every actual child must hold a live profile"
                time.sleep(.01)
            assert len({str(Path(row["profile"]).parent) for row in records}) == EXPECTED_PARTICIPANTS
        assert sorted(temporary.glob("openecon-network-export-*")) == [*before, profile]
        if round_number == 1:
            (gate / f"checked-{round_number}-{child.name}").write_text("checked")
            while len(list(gate.glob(f"checked-{round_number}-*"))) != EXPECTED_PARTICIPANTS:
                assert time.monotonic() < deadline
                time.sleep(.01)
    assert sorted(temporary.glob("openecon-network-export-*")) == before
'''.replace("EXPECTED_PARTICIPANTS", str(participants))
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    monkeypatch.setattr(groups.os, "cpu_count", lambda: participants)
    monkeypatch.setattr(groups.os, "sched_getaffinity", lambda _: set(range(participants)), raising=False)
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
    assert report["tests"]["tests"] == (len(SELECTORS) if shard is None else len(SELECTORS) // 2)
    assert len(report["groups"]) == participants
    assert len({item["temporary_directory"] for item in report["groups"]}) == participants
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
    assert len(calls) == 3
    first, *others = report["groups"]
    assert first["temporary_directory_removed_after_stop"] is False
    assert first["temporary_cleanup_error"] == "OSError: intentional owned cleanup refusal"
    assert Path(first["temporary_directory"]).is_dir()
    for child in others:
        assert child["temporary_directory_removed_after_stop"] is True
        assert not Path(child["temporary_directory"]).exists()
    for item in report["groups"]:
        assert (base / "sdk-groups" / item["log"]).is_file()
        assert (base / "sdk-groups" / item["junit"]).is_file()


def _add_third_child_native_probe(root, body):
    """Keep all incoming toy cases and add an actual probe for the third child."""
    selector = TOY_FIXED[0][0]
    source = root / selector
    original = source.read_text()
    assert "def test_values(value):" in original and "class TestExact:" in original
    source.write_text(original + "\n" + body)
    for command in (["git", "add", selector], ["git", "commit", "-qm", "retain cases and probe third child"]):
        subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _prove_native_thread_recipe(tmp_path, monkeypatch, component, threads):
    # Both recipes execute real Torch children using the available interpreter;
    # the toy component labels do not establish hosted Python-version proof.
    # Current main allocates one worker per concurrent SDK child. Retain a
    # real source-hook check of both admitted opt-in counts, including prior2.
    if component is not None:
        probe = "import importlib.util; spec=importlib.util.spec_from_file_location('actual_hook', " + repr(str(ROOT / "scripts/pytest_gate_timings.py")) + "); hook=importlib.util.module_from_spec(spec); spec.loader.exec_module(hook); hook._configure_native_threads(); import torch; assert torch.get_num_threads()==" + str(threads)
        probe_env = {**os.environ, **{name: str(threads) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENECON_GATE_TORCH_THREADS")}}
        subprocess.run([sys.executable, "-c", probe], env=probe_env, check=True, timeout=30)
    if component is not None:
        # TOY CPU observations exercise both exact three-child budget recipes.
        cpus = 3 * threads
        monkeypatch.setattr(groups, "detect_cpu_budget", lambda: {
            "system": "Darwin", "host_cpus": cpus, "affinity_cpus": cpus,
            "effective_cpus": float(cpus), "groups": 3, "max_concurrent_children": 3,
            "threads_per_child": min(2, max(1, cpus // 3)), "cgroup": {
                "status": "not_applicable", "version": None, "views": [],
                "probes": [], "limits": [], "error": None}})
    child_threads = 1 if component is None else groups.detect_cpu_budget()["threads_per_child"]
    body = '''import os
import torch

def test_native_threads():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[name] == "__EXPECTED__"
    assert torch.get_num_threads() == __EXPECTED__
'''.replace("__EXPECTED__", str(child_threads))
    root = repository(tmp_path / "repo", {SELECTORS[0]: body, TOY_HEAVY[0]: body})
    _add_third_child_native_probe(root, body)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "64")
    expected = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    assert {name: gate.test_environment(component=component)[name] for name in expected} == expected
    assert {name: groups.environment(root, component=component, threads=1)[name] for name in expected} == expected
    execution = _native_fixture_execution(root, component)
    timeout = 900 if component is None else 1800
    # Opt into the actual copied timing hook's declared recipe in every child.
    # Torch's initial count alone can differ between platform/builds.
    monkeypatch.setenv("OPENECON_GATE_TORCH_THREADS", str(child_threads))
    base, report = execute(root, timeout=timeout, component=component, **execution)
    assert report["timeout_seconds"] == timeout
    assert report["execution"] == execution.get("execution")
    assert report["execution_sha256"] == execution.get("execution_sha256")
    assert report["status"] == "passed" and report["tests"]["tests"] == 11, json.dumps({
        "report": report, "child_logs": {
            item["index"]: (base / "sdk-groups" / f"group-{item['index']}" / "pytest.log").read_text()
            for item in report["groups"]}}, indent=2)
    assert report["selected_tests"] == SELECTORS
    assert report["component_sdk_version"] == component
    assert {name: report["environment"][name] for name in expected} == {name: str(child_threads) for name in expected}
    for item in report["groups"]:
        assert {name: item["environment"][name] for name in expected} == {name: str(child_threads) for name in expected}


def test_parallel_children_override_inherited_native_threads_and_use_one_torch_thread(tmp_path, monkeypatch):
    _prove_native_thread_recipe(tmp_path, monkeypatch, None, 1)


def test_parallel_children_own_complete_temporary_namespaces(tmp_path):
    body = '''import json
import os
import tempfile
from pathlib import Path

def test_native_temp():
    directory = Path(tempfile.gettempdir())
    assert directory.name == 'runtime-temp'
    assert all(os.environ[name] == str(directory) for name in ('TMPDIR', 'TMP', 'TEMP'))
    before = set(directory.glob('openecon-network-export-*'))
    with tempfile.TemporaryDirectory(prefix='openecon-network-export-') as profile:
        assert Path(profile).parent == directory
        assert set(directory.glob('openecon-network-export-*')) == before | {Path(profile)}
    assert set(directory.glob('openecon-network-export-*')) == before
    (Path('receipts') / f'observed-{os.getpid()}.json').write_text(json.dumps({'pid': os.getpid(), 'temp': str(directory)}))
'''
    root = repository(tmp_path / 'repo', {SELECTORS[0]: body, TOY_HEAVY[0]: body})
    _add_third_child_native_probe(root, body)
    _, report = execute(root)
    assert report['status'] == 'passed' and report['tests']['tests'] == 11
    temporary = []
    for child in report['groups']:
        directory = Path(child['temporary_directory'])
        assert {key: child['environment'][key] for key in ('TMPDIR', 'TMP', 'TEMP')} == {name: str(directory) for name in ('TMPDIR', 'TMP', 'TEMP')}
        observed = json.loads((root / 'receipts' / f"observed-{child['pid']}.json").read_text())
        assert observed == {'pid': child['pid'], 'temp': str(directory)}
        temporary.append(directory)
    assert len(set(temporary)) == 3


@pytest.mark.parametrize("component,threads", [("3.11", 1), ("3.13", 2)])
def test_parallel_children_apply_component_native_recipe(tmp_path, monkeypatch, component, threads):
    _prove_native_thread_recipe(tmp_path, monkeypatch, component, threads)


def _timing_configuration(path=None):
    writers = []
    options = {"--gate-timings": str(path) if path else None, "--gate-collection": None,
               "--gate-full-collection": None, "--gate-partition-plan": None,
               "--gate-group-index": None, "--gate-group-count": None,
               "--gate-report-protocol": 1, "--gate-report-ledger": None,
               "--gate-report-terminal": None}
    return SimpleNamespace(getoption=options.__getitem__, option=SimpleNamespace(),
                           pluginmanager=SimpleNamespace(register=lambda writer, name: writers.append(writer))), writers


def _forbid_torch_import(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert name != "torch", "Torch must not import before a complete enabled recipe is admitted"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


@pytest.mark.parametrize("caps", [("1", None, None), ("1", "2", "1"), ("01", "01", "01"),
                                  ("64", "64", "64"), ("1.0", "1.0", "1.0"), (" 1", " 1", " 1")])
def test_timing_plugin_refuses_malformed_native_recipe_before_import_and_receipt(tmp_path, monkeypatch, caps):
    monkeypatch.setenv("OPENECON_GATE_TORCH_THREADS", "1")
    for name, value in zip(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), caps, strict=True):
        monkeypatch.delenv(name, raising=False) if value is None else monkeypatch.setenv(name, value)
    _forbid_torch_import(monkeypatch)
    destination = tmp_path / "timings.jsonl"
    config, writers = _timing_configuration(destination)
    with pytest.raises(ValueError, match="same canonical 1 or 2"):
        timing_plugin.pytest_configure(config)
    assert not destination.exists() and not writers


@pytest.mark.parametrize("enabled", [False, True])
def test_timing_plugin_without_enabled_declared_recipe_never_imports_torch(tmp_path, monkeypatch, enabled):
    monkeypatch.delenv("OPENECON_GATE_TORCH_THREADS", raising=False) if enabled else monkeypatch.setenv("OPENECON_GATE_TORCH_THREADS", "2")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False) if enabled else monkeypatch.setenv(name, "64")
    _forbid_torch_import(monkeypatch)
    destination = tmp_path / "timings.jsonl"
    config, writers = _timing_configuration(destination if enabled else None)
    timing_plugin.pytest_configure(config)
    assert destination.exists() is enabled and len(writers) == int(enabled)
    for writer in writers:
        writer.pytest_unconfigure()


def test_timing_plugin_refuses_unrealized_exact_recipe_before_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_GATE_TORCH_THREADS", "2")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "2")
    requested = []
    fake = SimpleNamespace(set_num_threads=requested.append, get_num_threads=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", fake)
    destination = tmp_path / "timings.jsonl"
    config, writers = _timing_configuration(destination)
    with pytest.raises(RuntimeError, match="exact declared"):
        timing_plugin.pytest_configure(config)
    assert requested == [2] and not destination.exists() and not writers


@pytest.mark.parametrize("component", ["3.10", "3.12", "unknown", True])
def test_unknown_component_recipe_is_rejected_before_children(tmp_path, component):
    with pytest.raises(ValueError, match="Unknown SDK component"):
        gate.test_environment(component=component)
    with pytest.raises(ValueError, match="Unknown SDK component"):
        groups.environment(tmp_path, component=component)
    with pytest.raises(ValueError, match="Unknown SDK component"):
        gate.test_environment(component=component, distributed=True)
    with pytest.raises(ValueError, match="Unknown SDK component"):
        groups.environment(tmp_path, component=component, distributed=True)
    root = repository(tmp_path / "repo")
    with pytest.raises(ValueError, match="Unknown SDK component"):
        execute(root, component=component)
    with pytest.raises(ValueError, match="Unknown SDK component"):
        execute(root, component=component, shard_index=0, receipt="receipts/distributed")
    assert not list((root / "receipts/run").glob("sdk-groups/group-*"))
    assert not list((root / "receipts/distributed").glob("sdk-groups/group-*"))


@pytest.mark.parametrize("attack", ["component", "interpreter"])
def test_production_cli_binds_component_to_actual_interpreter(tmp_path, attack):
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert version in ("3.11", "3.13")
    component = ("3.11" if version == "3.13" else "3.13") if attack == "component" else version
    executable = sys.executable if attack == "component" else str(tmp_path / "different-python")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_parallel_sdk_groups.py"),
         "--python", executable, "--directory", str(tmp_path / "groups"),
         "--junitxml", str(tmp_path / "out.xml"),
         "--gate-timings", str(tmp_path / "out.jsonl"),
         "--execution", str(tmp_path / "execution.json"),
         "--component-sdk-version", component, "--timeout", "1800"],
        capture_output=True, text=True, timeout=10,
    )
    message = ("grouping interpreter differs from its recorded component version"
               if attack == "component" else "child interpreter differs from its grouping interpreter")
    assert result.returncode != 0 and message in result.stderr
    assert not (tmp_path / "groups").exists()


@pytest.mark.parametrize("component", ["3.11", "3.13"])
def test_distributed_component_recipe_uses_one_actual_torch_thread(tmp_path, component):
    # Real children execute both recipes with the available interpreter;
    # actual CLI version binding has its separate production refusal proof.
    body = '''import os
import torch

def test_actual_recipe():
    assert all(os.environ[name] == "1" for name in
               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert torch.get_num_threads() == 1
'''
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    inherited = {**os.environ, "OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "64",
                 "OPENBLAS_NUM_THREADS": "64"}
    expected = {name: "1" for name in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    assert {name: gate.test_environment(component=component, distributed=True)[name]
            for name in expected} == expected
    execution = _native_fixture_execution(root, component, shard_index=0, shard_count=2)
    base, report = execute(root, timeout=900, component=component, shard_index=0, shard_count=2, env=inherited, **execution)
    assert report["timeout_seconds"] == 900
    assert report["execution"] == execution["execution"]
    assert report["execution_sha256"] == execution["execution_sha256"]
    assert report["status"] == "passed", report
    assert report["partial_scope"] is True and report["component_sdk_version"] == component
    assert report["selected_tests"] == SELECTORS and report["tests"]["tests"] == 2
    assert {name: report["environment"][name] for name in expected} == expected
    for item in report["groups"]:
        assert {name: item["environment"][name] for name in expected} == expected
        assert item["collected_tests"] == 1 and item["tests"]["tests"] == 1
        full = base / "sdk-groups" / item["full_collection"]
        assert len(full.read_text().splitlines()) == 4


def _native_fixture_execution(root, component, shard_index=None, shard_count=None):
    """Bind toy native probes explicitly; these records do not claim hosted version proof."""
    if component is None:
        return {}
    execution = {"source_commit": groups.source_identity(root)["source_commit"],
                 "component_sdk_version": component}
    if shard_index is not None:
        execution["component_sdk_shard"] = shard_index
        execution["component_sdk_shard_count"] = shard_count
    directory = root / "receipts"
    directory.mkdir(exist_ok=True)
    receipt = directory / "native-execution.json"
    with receipt.open("x") as stream:
        stream.write(json.dumps(execution, sort_keys=True) + "\n")
    return {"execution": execution, "execution_sha256": groups.digest(receipt)}


# POSIX ownership regression checks use stdlib subprocesses only. These are
# lifecycle fixtures, not scientific or hosted SDK acceptance.
def _atomic_sdk_pid_marker(path):
    # Existence is the readiness signal: the public name must never expose the
    # interval between opening an empty file and flushing its complete PID.
    return (f"_ready = pathlib.Path({str(path)!r}); "
            "_pending = _ready.with_name(_ready.name + '.pending'); "
            "_pending.write_text(str(os.getpid())); _pending.replace(_ready); ")


def _sdk_lifecycle_exited(process, seconds=5):
    deadline = time.monotonic() + seconds
    while (code := groups._sdk_exit(process)) is None:
        assert time.monotonic() < deadline, "Owned leader did not exit"
        time.sleep(.01)
    return code


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_pid_readiness_never_exposes_an_empty_marker(tmp_path):
    ready, writing, release = (tmp_path / name for name in ("ready.pid", "writing", "release"))
    body = f'''import os,pathlib,time
writing = pathlib.Path({str(writing)!r})
release = pathlib.Path({str(release)!r})
original_write = pathlib.Path.write_text
def paused_write(path, text):
    path.touch()
    writing.touch()
    deadline = time.monotonic() + 5
    while not release.exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    return original_write(path, text)
pathlib.Path.write_text = paused_write
{_atomic_sdk_pid_marker(ready)}
time.sleep(30)
'''
    process = groups._launch_sdk([sys.executable, "-c", body])
    try:
        deadline = time.monotonic() + 5
        while not writing.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        pending = ready.with_name(ready.name + ".pending")
        assert pending.is_file() and pending.read_text() == ""
        assert not ready.exists(), "An incomplete PID cannot claim that the worker is ready"
        release.touch()
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert int(ready.read_text()) == process.pid and not pending.exists()
        assert groups._sdk_exit(process) is None
    finally:
        groups._stop([process])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_exited_leader_stops_its_live_grandchild(tmp_path, monkeypatch):
    marker = tmp_path / "grandchild.pid"
    body = ("import pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))")
    process = groups._launch_sdk([sys.executable, "-c", body])
    try:
        assert _sdk_lifecycle_exited(process) == 0 and process.returncode is None
        grandchild = int(marker.read_text())
        assert grandchild in groups._live_sdk_group(process, time.monotonic() + 2)
        calls, original = [], groups.os.killpg

        def owned_signal(group, signum):
            assert group == process.pid and process.returncode is None
            calls.append(signum)
            original(group, signum)

        with monkeypatch.context() as patch:
            patch.setattr(groups.os, "killpg", owned_signal)
            groups._stop([process])
        assert calls == [groups.signal.SIGTERM]
        assert process.returncode == 0 and process._openecon_sdk_session_closed is True
        rows = subprocess.check_output(["/bin/ps", "-A", "-o", "pid=,pgid=,stat="], text=True)
        assert not [line for line in rows.splitlines()
                    if int(line.split()[1]) == process.pid and not line.split()[2].startswith("Z")]
    finally:
        groups._stop([process])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_exited_empty_group_is_reaped_once_without_signals(monkeypatch):
    process = groups._launch_sdk([sys.executable, "-c", "pass"])
    try:
        assert _sdk_lifecycle_exited(process) == 0 and process.returncode is None
        assert groups._live_sdk_group(process, time.monotonic() + 2) == []
        with monkeypatch.context() as patch:
            patch.setattr(groups.os, "killpg", lambda *args: pytest.fail("Empty/reaped group cannot be signalled"))
            groups._stop([process])
            assert process.returncode == 0 and process._openecon_sdk_session_closed is True
            groups._stop([process])
    finally:
        groups._stop([process])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
@pytest.mark.parametrize("error", [PermissionError, ProcessLookupError])
def test_sdk_owned_last_recipient_signal_race_requires_fresh_empty_proof(monkeypatch, error):
    process = groups._launch_sdk([sys.executable, "-c", "import time; time.sleep(30)"])
    calls, original = [], groups.os.killpg

    def exited_recipient(group, signum):
        assert group == process.pid and process.returncode is None
        calls.append(signum)
        original(group, signum)
        assert _sdk_lifecycle_exited(process) == -signum
        raise error("Last live recipient exited after its owned snapshot")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(groups.os, "killpg", exited_recipient)
            groups._stop([process])
        assert calls == [groups.signal.SIGTERM]
        assert process.returncode == -groups.signal.SIGTERM
        assert process._openecon_sdk_session_closed is True
    finally:
        groups._stop([process])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
@pytest.mark.parametrize("error", [PermissionError, ProcessLookupError])
def test_sdk_owned_denied_live_recipient_is_never_hidden(monkeypatch, error):
    process = groups._launch_sdk([sys.executable, "-c", "import time; time.sleep(30)"])
    calls = []

    def denied(group, signum):
        assert group == process.pid and process.returncode is None
        calls.append(signum)
        raise error("Owned live recipient denied its signal")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(groups.os, "killpg", denied)
            with pytest.raises(error, match="Owned live recipient denied"):
                groups._stop([process])
        assert calls == [groups.signal.SIGTERM, groups.signal.SIGKILL]
        assert process.returncode is None and groups._sdk_exit(process) is None
        assert process._openecon_sdk_session_closed is False
    finally:
        groups._stop([process])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
@pytest.mark.parametrize("external_reap", [False, True])
def test_sdk_owned_cached_or_external_reap_never_signals_a_reused_group(monkeypatch, external_reap):
    process = groups._launch_sdk([sys.executable, "-c", "pass"])
    if external_reap:
        assert os.waitpid(process.pid, 0) == (process.pid, 0)
    else:
        assert process.wait(timeout=5) == 0
    with monkeypatch.context() as patch:
        patch.setattr(groups.os, "killpg", lambda *args: pytest.fail("Reaped PID cannot authorize a group signal"))
        with pytest.raises(ValueError, match="reaped|ownership was lost"):
            groups._stop([process])
    process.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_arbitrary_unregistered_handle_never_authorizes_signal(monkeypatch):
    monkeypatch.setattr(groups.os, "killpg", lambda *args: pytest.fail("Unowned handle cannot authorize a group signal"))
    with pytest.raises(ValueError, match="owned dedicated session"):
        groups._stop([SimpleNamespace(pid=os.getpid(), returncode=None)])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
@pytest.mark.parametrize("code", [0, 7, -15])
def test_sdk_owned_darwin_311_observer_keeps_exact_native_status(monkeypatch, code):
    # Darwin executes its real public ABI fallback. Other supported POSIX hosts
    # proxy their native waitid result through that exact ctypes Siginfo layout.
    builtin = getattr(os, "waitid", None)
    if sys.platform != "darwin" and builtin is None:
        with monkeypatch.context() as patch:
            patch.setattr(groups.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Unsupported observer launched a child"))
            with pytest.raises(ValueError, match="non-reaping exit observation"):
                groups._launch_sdk([sys.executable, "-c", "pass"])
        return
    body = (f"import sys; sys.exit({code})" if code >= 0 else
            f"import os,signal; os.kill(os.getpid(),{-code})")
    process = groups._launch_sdk([sys.executable, "-c", body])
    try:
        with monkeypatch.context() as patch:
            if sys.platform != "darwin":
                import ctypes

                class NativeWaitidProxy:
                    def __call__(self, kind, pid, target, flags):
                        observed = builtin(kind, pid, flags)
                        if observed is not None:
                            target._obj.si_pid = observed.si_pid
                            target._obj.si_code = observed.si_code
                            target._obj.si_status = observed.si_status
                        return 0

                proxy = NativeWaitidProxy()
                patch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(waitid=proxy))
                patch.setattr(groups.sys, "platform", "darwin")
            patch.delattr(groups.os, "waitid", raising=False)
            assert _sdk_lifecycle_exited(process) == code and process.returncode is None
            assert groups._live_sdk_group(process, time.monotonic() + 2) == []
            groups._stop([process])
        assert process.returncode == code and process._openecon_sdk_session_closed is True
    finally:
        groups._stop([process])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_missing_observer_refuses_before_actual_launch(monkeypatch):
    calls = []
    monkeypatch.delattr(groups.os, "waitid", raising=False)
    monkeypatch.setattr(groups.sys, "platform", "linux")
    monkeypatch.setattr(groups.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match="non-reaping exit observation"):
        groups._launch_sdk([sys.executable, "-c", "pass"])
    assert calls == []


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_cleanup_failure_still_stops_other_independent_groups(monkeypatch):
    first = groups._launch_sdk([sys.executable, "-c", "import time; time.sleep(30)"])
    second = groups._launch_sdk([sys.executable, "-c", "import time; time.sleep(30)"])
    original, denied_calls = groups.os.killpg, []

    def denied_first(group, signum):
        if group == first.pid:
            denied_calls.append(signum)
            raise PermissionError("First owned group denied cleanup")
        original(group, signum)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(groups.os, "killpg", denied_first)
            with pytest.raises(PermissionError, match="First owned group denied cleanup"):
                groups._stop([first, second])
        assert denied_calls == [groups.signal.SIGTERM, groups.signal.SIGKILL]
        assert first.returncode is None and first._openecon_sdk_session_closed is False
        assert second.returncode == -groups.signal.SIGTERM and second._openecon_sdk_session_closed is True
    finally:
        groups._stop([first, second])


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_unproved_group_keeps_its_temp_and_other_children_keep_raw_evidence(tmp_path, monkeypatch):
    root = repository(tmp_path / "repo")
    stopped, original = [], groups._stop

    def stop_only_independent_children(processes):
        stopped.extend(processes)
        original(processes[1:])
        raise PermissionError("First owned group has no shutdown proof")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(groups, "_stop", stop_only_independent_children)
            base, report = execute(root)
        assert report["status"] == "failed"
        assert "First owned group has no shutdown proof" in report["error"]
        assert "First owned group has no shutdown proof" in report["process_cleanup_error"]
        assert len(report["groups"]) == 3
        first, *others = report["groups"]
        assert first["temporary_directory_removed_after_stop"] is False
        assert "verified owned process-group shutdown" in first["temporary_cleanup_error"]
        assert Path(first["temporary_directory"]).is_dir()
        assert all(group["temporary_directory_removed_after_stop"] is True for group in others)
        for group in report["groups"]:
            assert (base / "sdk-groups" / group["log"]).is_file()
            assert (base / "sdk-groups" / group["junit"]).is_file()
        assert json.loads((base / "sdk-groups/report.json").read_text())["error"] == report["error"]
    finally:
        original(stopped)


@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_kill_signals_every_group_before_a_wait_can_spend_the_fresh_deadline(tmp_path, monkeypatch):
    processes = []
    try:
        for index in range(2):
            ready = tmp_path / f"term-resistant-{index}.pid"
            body = ("import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    + _atomic_sdk_pid_marker(ready) + "time.sleep(30)")
            process = groups._launch_sdk([sys.executable, "-c", body])
            processes.append(process)
            deadline = time.monotonic() + 5
            while not ready.exists():
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert int(ready.read_text()) == process.pid and groups._sdk_exit(process) is None
    except BaseException:
        groups._stop(processes)
        raise
    calls, snapshots = [], []
    first, second = processes
    original_signal, original_wait = groups.os.killpg, groups._wait_sdk_group
    original_snapshot = groups._live_sdk_group

    def owned_signal(group, signum):
        assert group in {first.pid, second.pid}
        assert all(process.returncode is None for process in processes)
        calls.append((group, signum))
        original_signal(group, signum)

    def fresh_snapshot(process, deadline):
        snapshot = original_snapshot(process, deadline)
        snapshots.append((process.pid, snapshot))
        return snapshot

    try:
        with monkeypatch.context() as patch:
            def controlled_wait(process, deadline):
                kills = {pid for pid, signum in calls if signum == groups.signal.SIGKILL}
                if not kills:
                    # Model the spent TERM wait; both actual children ignore TERM.
                    return False
                if process is first:
                    assert kills == {first.pid, second.pid}, (
                        "Every independently owned group must receive KILL before waiting")
                    assert {pid for pid, live in snapshots if pid in live} == {first.pid, second.pid}
                    patch.setattr(groups.time, "monotonic", lambda: deadline + .01)
                    raise subprocess.TimeoutExpired("controlled first SDK group wait", 2)
                return original_wait(process, deadline)

            patch.setattr(groups.os, "killpg", owned_signal)
            patch.setattr(groups, "_live_sdk_group", fresh_snapshot)
            patch.setattr(groups, "_wait_sdk_group", controlled_wait)
            with pytest.raises(subprocess.TimeoutExpired, match="controlled first SDK group wait"):
                groups._stop(processes)
        assert calls == [(first.pid, groups.signal.SIGTERM), (second.pid, groups.signal.SIGTERM),
                         (first.pid, groups.signal.SIGKILL), (second.pid, groups.signal.SIGKILL)]
        assert all(process.returncode is None and process._openecon_sdk_session_closed is False
                   for process in processes), "A spent proof clock cannot authorize an unproved reap"
    finally:
        groups._stop(processes)


def test_sdk_owned_windows_process_disappearance_retains_original_signal_race_guards(monkeypatch):
    calls, stage = [], [0]

    class DisappearingChild:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("TERM")
            raise ProcessLookupError("Child exited at TERM")

        def kill(self):
            calls.append("KILL")
            stage[0] = 1
            raise ProcessLookupError("Child exited at KILL")

        def wait(self, timeout=None):
            assert timeout is not None and 0 <= timeout <= 2
            calls.append(("wait", timeout))
            if not stage[0]:
                raise subprocess.TimeoutExpired("controlled Windows TERM wait", timeout)
            self.returncode = 0
            return 0

    process = DisappearingChild()
    with monkeypatch.context() as patch:
        patch.setattr(groups, "os", SimpleNamespace(name="nt"))
        groups._stop([process])
    assert calls[0] == "TERM" and calls[2] == "KILL"
    assert calls[1][0] == calls[3][0] == "wait" and process.returncode == 0



@pytest.mark.skipif(os.name != "posix", reason="Owned POSIX SDK process groups")
def test_sdk_owned_original_failure_and_all_independent_cleanup_causes_survive_receipt_and_log(
        tmp_path, monkeypatch, capsys):
    body = 'def test_primary_failure(): assert False, "original primary SDK test failure"\n'
    root = repository(tmp_path / "repo", {selector: body for selector in SELECTORS})
    retained, original = [], groups._stop

    def independent_cleanup_failures(processes):
        retained.extend(processes)
        original(processes[2:])
        # Any child may fail first on a loaded runner. Await this case's real
        # primary failure and flushed log before injecting independent cleanup
        # causes; an empty still-running sibling is not failure evidence.
        assert _sdk_lifecycle_exited(processes[0]) == 1
        assert processes[0].returncode is None  # Retain the owned session anchor.
        try:
            raise PermissionError("First independent SDK cleanup denial")
        except PermissionError as first:
            try:
                raise OSError("Second independent SDK cleanup refusal")
            except OSError as second:
                raise first from BaseExceptionGroup("Independent owned SDK failures", [second])

    try:
        with monkeypatch.context() as patch:
            patch.setattr(groups, "_stop", independent_cleanup_failures)
            base, report = execute(root)
        assert report["status"] == "failed" and "SDK child process failed" in report["error"]
        trace = report["process_cleanup_traceback"]
        assert "First independent SDK cleanup denial" in trace
        assert "Second independent SDK cleanup refusal" in trace
        assert "Independent owned SDK failures" in trace
        assert len(trace.encode()) < 2 * 1024 * 1024
        parent_log = capsys.readouterr().err
        assert trace in parent_log
        saved = json.loads((base / "sdk-groups/report.json").read_text())
        assert saved["error"] == report["error"] and saved["process_cleanup_traceback"] == trace
        assert all(group["temporary_directory_removed_after_stop"] is False
                   for group in report["groups"][:2])
        assert all(Path(group["temporary_directory"]).is_dir() for group in report["groups"][:2])
        assert "original primary SDK test failure" in (base / "sdk-groups" / report["groups"][0]["log"]).read_text()
        for group in report["groups"]:
            assert (base / "sdk-groups" / group["log"]).is_file()
    finally:
        original(retained)


@pytest.mark.parametrize("denied_phase", ["terminate", "kill", "wait"])
def test_sdk_windows_denial_retains_all_causes_and_stops_independent_handle(monkeypatch, denied_phase):
    # Explicit fake native-handle protocol on every host; no Windows-installation claim.
    calls = []

    class NativeHandle:
        def __init__(self, pid):
            self.pid, self.returncode, self.stage = pid, None, "TERM"

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append((self.pid, "TERM"))
            if self.pid == 1 and denied_phase == "terminate":
                raise PermissionError("first native terminate denial")

        def kill(self):
            self.stage = "KILL"
            calls.append((self.pid, "KILL"))
            if self.pid == 1 and denied_phase == "kill":
                raise PermissionError("first native kill denial")

        def wait(self, timeout=None):
            assert timeout is not None and 0 <= timeout <= 2
            calls.append((self.pid, "wait", self.stage, timeout))
            if self.pid == 1 and denied_phase == "wait":
                raise PermissionError("first native wait denial")
            if self.stage == "TERM":
                raise subprocess.TimeoutExpired("controlled pending TERM handle", timeout)
            if self.pid == 1 and denied_phase == "kill":
                raise subprocess.TimeoutExpired("denied KILL retains live handle", timeout)
            self.returncode = 0
            return 0

    first, second = NativeHandle(1), NativeHandle(2)
    with monkeypatch.context() as patch:
        patch.setattr(groups, "os", SimpleNamespace(name="nt"))
        with pytest.raises(PermissionError, match="first native") as caught:
            groups._stop([first, second])
    assert second.returncode == 0
    assert calls[:2] == [(1, "TERM"), (2, "TERM")]
    kills = [i for i, call in enumerate(calls) if len(call) == 2 and call[1] == "KILL"]
    kill_waits = [i for i, call in enumerate(calls) if len(call) == 4 and call[2] == "KILL"]
    assert max(kills) < min(kill_waits)
    if denied_phase in ("kill", "wait"):
        assert first.returncode is None
        assert isinstance(caught.value.__cause__, BaseExceptionGroup)


def _complete_protocol2_fixture_module():
    fixture_spec = importlib.util.spec_from_file_location(
        "synthetic_complete_report_fixture", ROOT / "tests/gate_report_protocol_fixture.py")
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    return fixture


def test_protocol2_SDK_reads_all_nested_events_without_relabeling_parent_count(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, nodes = fixture.make(tmp_path / "sdk")
    actual, _ = groups.read_child(xml, raw, fixture.SELECTORS, report_protocol=2, root=ROOT)
    assert [row[0] for row in actual] == nodes and len(actual) == 122
    assert sum(len(row[2]) for row in actual) == 391
    assert b"".join(line for row in actual for line in row[2]) == raw.read_bytes()
    with pytest.raises(ValueError):
        groups.read_child(xml, raw, fixture.SELECTORS)


@pytest.mark.parametrize("damage", ["drop_nested", "duplicate_context", "nested_nonzero", "wrong_native_class",
                                     "after_outer_call", "nested_ordinary_parent", "missing_terminal", "xml146"])
def test_protocol2_SDK_refuses_resealed_missing_reordered_or_forged_nested_controls(tmp_path, damage):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, _ = fixture.make(tmp_path / damage)
    records, events = fixture.rows(raw)
    nested = next(i for i, e in enumerate(events) if e["report_kind"] == "unittest_subreport")
    if damage == "drop_nested":
        records.pop(nested)
        events.pop(nested)
    elif damage == "duplicate_context":
        events[nested+1]["native_context"] = events[nested]["native_context"]
    elif damage == "nested_nonzero":
        records[nested]["start"] = records[nested]["stop"] = 1000
    elif damage == "wrong_native_class":
        events[nested]["native_report_class"] = "_pytest.reports.TestReport"
    elif damage == "after_outer_call":
        records[nested], records[nested+2] = records[nested+2], records[nested]
        events[nested], events[nested+2] = events[nested+2], events[nested]
    elif damage == "nested_ordinary_parent":
        records[nested]["nodeid"] = "tests/test_protocol2_synthetic.py::test_parent_0"
        events[nested]["parent_nodeid"] = records[nested]["nodeid"]
        events[nested]["native_location"][0] = "tests/test_protocol2_synthetic.py"
    elif damage == "xml146":
        tree = ET.parse(xml)
        tree.getroot().set("tests", "146")
        tree.write(xml, encoding="utf-8", xml_declaration=True)
    fixture.reseal(raw, records, events)
    if damage == "missing_terminal":
        fixture.reports.paths(raw)[1].unlink()
    with pytest.raises((ValueError, OSError)):
        groups.read_child(xml, raw, fixture.SELECTORS, report_protocol=2, root=ROOT)


def test_protocol2_three_child_merge_keeps_nested_blocks_and_native_XML_counter(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    nodes = fixture.complete_nodes()
    nodes[-1] = "tests/test_protocol2_other.py::test_last"
    selectors = [*fixture.SELECTORS, "tests/test_protocol2_other.py"]
    partition = [[selectors[0]], [selectors[1]], [selectors[2]]]
    children = []
    for index, selected in enumerate(partition):
        path = tmp_path / f"group-{index}"
        fixture.make(path, [node for node in nodes if node.split("::", 1)[0] in selected])
        children.append(path)
    xml, raw = tmp_path / "merged.xml", tmp_path / "merged.jsonl"
    result = groups.merge_evidence(children, selectors, partition, xml, raw,
                                   report_protocol=2, root=ROOT)
    assert result == {"tests": 122, "failures": 0, "errors": 0, "skipped": 0}
    assert ET.parse(xml).getroot().find("testsuite").get("tests") == "147"
    assert len(list(ET.parse(xml).getroot().iter("testcase"))) == 122
    assert len(raw.read_bytes().splitlines()) == 391
    terminal = fixture.reports.terminal(raw, nodes, root=ROOT, derived=True)
    assert terminal["report_counts"]["nested_reports"] == 25
