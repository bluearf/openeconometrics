"""Two passing jobs need complete, current, independently checked artifacts."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parallel_merge_gate", ROOT / "scripts/verify_parallel_merge_gate.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
SELECTORS = ['tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_stream_acceptance.py', 'tests/test_nested_logit_independent.py', 'tests/test_control_function_stream_state.py', 'packages/openecon-charts/tests', 'tests/test_parallel_sdk_groups.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py', 'tests/test_bayesian_var_conjugate.py', 'tests/test_bayesian_var_sbc_protocol.py', 'tests/test_bayesian_var_public_integration.py', 'tests/test_bayesian_var_public_admission_v2.py', 'tests/test_editor_catalog_intern_v2.py']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def distributions(directory, root):
    result = []
    for name, version, package, source in (
        ("openecon", "0.0.1", "openecon", root / "src/openecon/__init__.py"),
        (
            "openecon-charts",
            "0.0.2",
            "openecon_charts",
            root / "packages/openecon-charts/src/openecon_charts/__init__.py",
        ),
    ):
        distribution = name.replace("-", "_") + "-" + version
        metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n".encode()
        wheel = directory / "dist" / (distribution + "-py3-none-any.whl")
        wheel.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(package + "/__init__.py", source.read_bytes())
            archive.writestr(distribution + ".dist-info/METADATA", metadata)
            archive.writestr(distribution + ".dist-info/WHEEL", "Wheel-Version: 1.0\n")
        sdist = directory / "dist" / (distribution + ".tar.gz")
        with tarfile.open(sdist, "w:gz") as archive:
            for relative, raw in (
                ("PKG-INFO", metadata),
                ("src/" + package + "/__init__.py", source.read_bytes()),
            ):
                info = tarfile.TarInfo(distribution + "/" + relative)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
        for path in (wheel, sdist):
            result.append(
                {"path": "dist/" + path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            )
    return result


@pytest.fixture(scope="module")
def genuine_artifacts(tmp_path_factory):
    """Use real pytest XML/phases, real archives and a real temporary PR merge.

    The common-step receipts are fixture metadata, not hosted acceptance. The
    test deliberately uses a small separate Git repository instead of running
    the scientific gate or altering this checkout.
    """
    directory = tmp_path_factory.mktemp("parallel-gate-input")
    root = directory / "project"
    for relative in (
        "scripts",
        "src/openecon",
        "tests",
        "packages/openecon-charts/src/openecon_charts",
        "packages/openecon-charts/tests",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("artifacts/\n.venv*/\n.pytest_cache/\n__pycache__/\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname="openecon"\nversion="0.0.1"\n[tool.pytest.ini_options]\naddopts="-q"\n'
    )
    (root / "packages/openecon-charts/pyproject.toml").write_text(
        '[project]\nname="openecon-charts"\nversion="0.0.2"\n'
    )
    (root / "src/openecon/__init__.py").write_text("VALUE = 'checked SDK source'\n")
    (root / "packages/openecon-charts/src/openecon_charts/__init__.py").write_text(
        "VALUE = 'checked chart source'\n"
    )
    (root / SELECTORS[0]).write_text("def test_linear(): pass\n")
    (root / SELECTORS[1]).write_text("def test_common(): pass\n")
    (root / SELECTORS[2]).write_text(
        "class TestGroup:\n    def test_alpha(self): pass\ndef test_beta(): pass\n"
    )
    (root / SELECTORS[3]).write_text("def test_control(): pass\n")
    (root / SELECTORS[4]).write_text("def test_nested(): pass\n")
    (root / SELECTORS[5]).write_text("def test_stream_state(): pass\n")
    (root / SELECTORS[7]).write_text("def test_controller_fixture(): pass\n")
    for selector in SELECTORS[8:]:
        (root / selector).write_text("def test_incoming_scientific_selector_fixture(): pass\n")
    (root / "packages/openecon-charts/tests/test_chart.py").write_text("def test_chart(): pass\n")
    (root / "scripts/verify_merge_candidate.py").write_text("TESTS = " + repr(SELECTORS) + "\n")
    shutil.copyfile(
        ROOT / "scripts/pytest_gate_timings.py", root / "scripts/pytest_gate_timings.py"
    )
    shutil.copyfile(
        ROOT / "scripts/verify_parallel_merge_gate.py",
        root / "scripts/verify_parallel_merge_gate.py",
    )
    shutil.copyfile(ROOT / "scripts/run_parallel_sdk_groups.py",
                    root / "scripts/run_parallel_sdk_groups.py")
    # This isolated artifact-schema fixture has one available interpreter,
    # intentionally aliased as both toy virtualenvs. Mock only its version
    # observation; real production CLI version refusal is tested separately.
    helper = root / "scripts/run_parallel_sdk_groups.py"
    source = helper.read_text()
    observation = 'f"{sys.version_info.major}.{sys.version_info.minor}" == args.component_sdk_version'
    assert source.count(observation) == 1
    source = source.replace(observation, 'args.component_sdk_version in ("3.11", "3.13")')
    # These TOY legacy artifacts deliberately prove all three-child adversaries
    # with three workers, independently of the machine running this fixture.
    entry_point = 'if __name__ == "__main__":'
    assert source.count(entry_point) == 1
    toy_budget = 'def detect_cpu_budget(*, proc=Path("/proc/self"), system=None):\n    # This copied schema fixture declares three TOY workers; real children,\n    # collection, phases, PID and source Git binding remain independently read.\n    return {"system": "Darwin", "host_cpus": 3, "affinity_cpus": 3,\n            "effective_cpus": 3., "groups": 3, "max_concurrent_children": 3,\n            "threads_per_child": 1, "cgroup": {"status": "not_applicable",\n            "version": None, "views": [], "probes": [], "limits": [], "error": None}}\n\n\n'
    helper.write_text(source.replace(entry_point, toy_budget + entry_point))
    for version in ("311", "313"):
        (root / f".venv{version}").symlink_to(Path(sys.prefix), target_is_directory=True)
    git(root, "init", "-q")
    git(root, "add", ".")
    author = ("-c", "user.name=Gate fixture", "-c", "user.email=gate@example.test")
    git(root, *author, "commit", "-qm", "baseline")
    base = git(root, "rev-parse", "HEAD")
    git(root, *author, "commit", "--allow-empty", "-qm", "PR head")
    head = git(root, "rev-parse", "HEAD")
    source = git(
        root, *author, "commit-tree", "HEAD^{tree}", "-p", base, "-p", head, "-m", "Tested PR merge"
    )
    git(root, "checkout", "-q", "--detach", source)
    git(root, "branch", "candidate", source)
    expected = {
        "source_commit": source,
        "pull_request_head": head,
        "pull_request_base": base,
        "event": "pull_request",
        "run_id": "123456789",
        "run_attempt": "2",
    }
    components = []
    for version in ("3.11", "3.13"):
        component = root / "artifacts" / ("component-" + version)
        receipt = component / "receipt"
        receipt.mkdir(parents=True)
        xml = receipt / f"pytest-{version}.xml"
        timings = receipt / f"pytest-{version}-timings.jsonl"
        checkout, output = str(root), str(receipt)
        python311, python313 = checkout + "/.venv311/bin/python", checkout + "/.venv313/bin/python"
        executable = python311 if version == "3.11" else python313
        # Both tiny fixture environments intentionally point to this test
        # interpreter; this is real grouped execution, not hosted SDK proof.
        execution = {**expected, "browser_executable": "/usr/bin/google-chrome",
                     "component_sdk_version": version}
        write_json(component / "execution.json", execution)
        actual_command = [
            executable, "scripts/run_parallel_sdk_groups.py", "--python", executable,
            "--directory", output + f"/sdk-{version}-groups",
            "--junitxml", str(xml), "--gate-timings", str(timings), "--timeout", "1800",
            "--execution", str(component / "execution.json"),
            "--component-sdk-version", version,
        ]
        env = os.environ.copy()
        started = time.monotonic()
        env.update(OPENECON_SDK_PARENT_STARTED_MONOTONIC=str(started),
                   OPENECON_SDK_PARENT_DEADLINE_MONOTONIC=str(started + 1800))
        sdk_log = receipt / f"sdk-{version}.log"
        with sdk_log.open("w") as stream:
            completed = subprocess.run(actual_command, cwd=root, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=30, check=False)
        assert completed.returncode == 0, sdk_log.read_text()
        sdk_seconds = time.monotonic() - started
        commands = [
            (
                "ruff",
                [
                    python313,
                    "-m",
                    "ruff",
                    "check",
                    "src",
                    "tests",
                    "scripts",
                    "packages/openecon-charts",
                ],
            ),
            ("capabilities", [python313, "scripts/generate_capability_docs.py", "--check"]),
            ("editor", [python313, "scripts/generate_editor_api.py", "--check"]),
            ("web-install", ["npm", "ci", "--prefix", "web", "--ignore-scripts"]),
            ("web-tests", ["node", "scripts/verify_web_gate.mjs"]),
            ("web-build", ["npm", "--prefix", "web", "run", "build"]),
            (f"sdk-{version}", actual_command),
            ("packages", ["uv", "build", "--all-packages", "--out-dir", output + "/dist"]),
        ]
        steps = []
        for name, command in commands:
            log = receipt / (name + ".log")
            if name != f"sdk-{version}":
                log.write_text("Fixture common-step log\n")
            steps.append(
                {
                    "name": name,
                    "command": command,
                    "status": "passed",
                    "exit_code": 0,
                    "seconds": sdk_seconds if name == f"sdk-{version}" else 0.1,
                    "log": log.name,
                    "log_sha256": digest(log),
                }
            )
        steps[-2].update(
            tests={"tests": 26, "failures": 0, "errors": 0, "skipped": 0},
            junit=xml.name,
            junit_sha256=digest(xml),
            phase_timings=timings.name,
            phase_timings_sha256=digest(timings),
            sdk_groups=f"sdk-{version}-groups/report.json",
            sdk_groups_sha256=digest(receipt / f"sdk-{version}-groups/report.json"),
        )
        report = {
            "schema": 1,
            "source_commit": source,
            "base_commit": None,
            "component_mode": True,
            "component_sdk_version": version,
            "status_context": None,
            "status": "passed",
            "timeout_seconds": 1800,
            "python_versions_provisioned": ["3.11", "3.13"],
            "python_interpreters": {"3.11": python311, "3.13": python313},
            "sdk_versions_executed": [version],
            "selected_tests": SELECTORS,
            "checkout_root": checkout,
            "receipt_directory": output,
            "gate_environment": {
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "inherited_pytest_options_removed": [
                    "PYTEST_ADDOPTS",
                    "PYTEST_PLUGINS",
                    "PYTEST_CURRENT_TEST",
                ],
                "explicit_plugin": "scripts.pytest_gate_timings",
            },
            "steps": steps,
            "packages": distributions(receipt, root),
        }
        write_json(
            component / "execution.json",
            {
                **expected,
                "browser_executable": "/usr/bin/google-chrome",
                "component_sdk_version": version,
            },
        )
        write_json(receipt / "report.json", report)
        components.append(component)
    return root, components, expected


@pytest.fixture
def artifact_copy(genuine_artifacts, tmp_path):
    root, templates, expected = genuine_artifacts
    components = [tmp_path / path.name for path in templates]
    for template, path in zip(templates, components):
        shutil.copytree(template, path)
    return root, components, copy.deepcopy(expected)


def validate(inputs, results=None):
    root, components, expected = inputs
    return validator.validate_components(
        components, expected, results or {"3.11": "success", "3.13": "success"}, root=root
    )


def edit_report(component, change):
    path = component / "receipt/report.json"
    report = json.loads(path.read_text())
    change(report)
    write_json(path, report)


def edit_groups(component, change):
    version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
    relative = f"sdk-{version}-groups/report.json"
    path = component / "receipt" / relative
    manifest = json.loads(path.read_text())
    change(manifest)
    write_json(path, manifest)
    edit_report(component, lambda report: report["steps"][-2].update(sdk_groups_sha256=digest(path)))


def rebind_outputs(component):
    version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
    xml = component / f"receipt/pytest-{version}.xml"
    phases = component / f"receipt/pytest-{version}-timings.jsonl"
    edit_report(component, lambda report: report["steps"][-2].update(
        junit_sha256=digest(xml), phase_timings_sha256=digest(phases)))
    sdk = json.loads((component / "receipt/report.json").read_text())["steps"][-2]

    def bind(manifest):
        manifest.update(junit_sha256=sdk["junit_sha256"],
                        phase_timings_sha256=sdk["phase_timings_sha256"])
        manifest["combined"].update(junit_sha256=sdk["junit_sha256"],
                                    phase_timings_sha256=sdk["phase_timings_sha256"])
        for group in manifest["groups"]:
            for key in ("log", "junit", "phase_timings", "collection"):
                path = component / f"receipt/sdk-{version}-groups" / group[key]
                group[key + "_sha256"] = digest(path)

    edit_groups(component, bind)


def test_two_current_artifacts_bind_real_cases_phases_and_package_sources(artifact_copy):
    result = validate(artifact_copy)
    assert result["status"] == "passed"
    assert result["selected_tests"] == SELECTORS
    assert [item["version"] for item in result["components"]] == ["3.11", "3.13"]
    for component in result["components"]:
        assert component["JUnit"] == {"tests": 26, "failures": 0, "errors": 0, "skipped": 0}
        assert component["phase_records"] == 78
        assert component["sdk_groups"]["kind"] == "parallel_sdk_groups"
        cpu_budget = component["sdk_groups"]["cpu_budget"]
        assert cpu_budget["groups"] == 3 and cpu_budget["threads_per_child"] in (1, 2)
        assert component["sdk_groups"]["environment"]["OMP_THREAD_LIMIT"] == str(
            cpu_budget["threads_per_child"])
        groups = component["sdk_groups"]["groups"]
        assert [group["selectors"] for group in groups] == [[SELECTORS[3]], SELECTORS[:3] + SELECTORS[4:6] + SELECTORS[7:], SELECTORS[6:7]]
        assert [group["tests"]["tests"] for group in groups] == [1, 24, 1]
        assert [group["collected_tests"] for group in groups] == [1, 24, 1]
        assert len({group["pid"] for group in groups}) == 3
        assert len(component["packages"]) == 4
        assert all(
            package["verified_python_source_files"] == 1 for package in component["packages"]
        )
    for component in artifact_copy[1]:
        report = json.loads((component / "receipt/report.json").read_text())
        assert report["gate_environment"] == validator.GATE_ENVIRONMENT
        version = report["component_sdk_version"]
        manifest = json.loads(
            (component / f"receipt/sdk-{version}-groups/report.json").read_text()
        )
        expected = validator.sdk_child_environment(manifest)
        assert manifest["environment"] == expected
        assert all(group["environment"] == {**expected, **{key: group["temporary_directory"]
                          for key in ("TMPDIR", "TMP", "TEMP")}}
                   for group in manifest["groups"])
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert report["gate_environment"][name] == "1"
            assert manifest["environment"][name] == str(manifest["cpu_budget"]["threads_per_child"])


def test_preregistered_three_groups_are_complete_disjoint_and_in_source_order():
    assert validator.SDK_FIXED_GROUPS == (('tests/test_control_stream_acceptance.py',), ('tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_function_stream_state.py', 'tests/test_nested_logit_independent.py', 'tests/test_parallel_sdk_groups.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py', 'tests/test_bayesian_var_conjugate.py', 'tests/test_bayesian_var_sbc_protocol.py', 'tests/test_bayesian_var_public_integration.py', 'tests/test_bayesian_var_public_admission_v2.py', 'tests/test_editor_catalog_intern_v2.py'))
    historical_fixed = (('tests/test_control_stream_acceptance.py',), ('tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_function_stream_state.py', 'tests/test_nested_logit_independent.py', 'tests/test_parallel_sdk_groups.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py'))
    source = validator.selector_contract(ROOT)
    assert source[155] == "tests/test_multivariate_score_contrasts.py"
    fixed = [item for group in validator.SDK_FIXED_GROUPS for item in group]
    partition = [[item for item in source if item in group] for group in validator.SDK_FIXED_GROUPS]
    partition.append([item for item in source if item not in fixed])
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan.json").read_text())
    assert plan["group_count"] == 3
    assert plan["fixed_groups"] == [list(group) for group in historical_fixed]
    baseline = plan["selected_tests"]
    assert source[:138] == baseline
    assert source[145:146] == ["tests/test_multivariate_score_uncertainty.py"]
    assert source[138:142] == ["tests/test_weighted_binary.py", "tests/test_econ_glm.py", "tests/test_econ_glm_oracle.py", "tests/test_econ_saved_prediction_categories.py"]
    assert plan["effective_groups"] == [[item for item in baseline if item in group] for group in partition]
    assert list(map(len, partition)) == [1, 23, 136]
    assert plan["environment"] == validator.GATE_ENVIRONMENT
    assert plan["selected_test_scope_sha256"] == hashlib.sha256(
        json.dumps(baseline, separators=(",", ":")).encode()).hexdigest()
    assert plan["timeout_seconds"] == 900
    assert len(partition) == 3 and set(sum(partition, [])) == set(source)
    assert sum(map(len, partition)) == len(source)



def test_production_four_file_partition_is_complete_disjoint_and_in_source_order():
    # This incoming case retains its heavy-file and complete-scope assertions
    # while the current legacy receipt requires three independent children.
    source = validator.selector_contract(ROOT)
    assert source[155] == "tests/test_multivariate_score_contrasts.py"
    original_heavy = {
        "tests/test_econ_saved_prediction_linear.py",
        "tests/test_control_function_common_prediction.py",
        "tests/test_streaming_control_function_engine.py",
        "tests/test_control_stream_acceptance.py",
    }
    fixed = [item for group in validator.SDK_FIXED_GROUPS for item in group]
    partition = [[item for item in source if item in group] for group in validator.SDK_FIXED_GROUPS]
    partition.append([item for item in source if item not in fixed])
    assert len(source) == 160 and list(map(len, partition)) == [1, 23, 136]
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan.json").read_text())
    assert source[:138] == plan["selected_tests"]
    assert source[145:146] == ["tests/test_multivariate_score_uncertainty.py"]
    assert source[138:142] == ["tests/test_weighted_binary.py", "tests/test_econ_glm.py", "tests/test_econ_glm_oracle.py", "tests/test_econ_saved_prediction_categories.py"]
    assert source[:119] == plan["selector_order_qualification"]["current_order_preserving_proposed_union"]
    assert source[119:124] == plan["twostep_added_selectors"]
    assert source[124:126] == plan["survey_deff_scope_extension"]["added_selectors"] == ["tests/test_survey_deff.py", "tests/test_survey_inference.py"]
    assert source[126:138] == plan["PR213_current_main_integration"]["all12_original_scientific_additions"]
    assert original_heavy <= set(sum(partition[:2], []))
    assert len(sum(partition, [])) == len(set(sum(partition, []))) == len(source)
    assert set(sum(partition, [])) == set(source)
    assert all(group == [item for item in source if item in group] for group in partition)


@pytest.mark.parametrize("attack", [
    "missing_budget", "effective", "threads", "groups", "environment", "child_environment",
    "quota_ratio", "missing_quota", "sibling_quota", "malformed_count", "foreign_platform", "workers",
])
def test_cpu_budget_and_actual_child_environment_tampering_is_rejected(artifact_copy, attack):
    def change(manifest):
        budget = manifest["cpu_budget"]
        if attack == "missing_budget":
            manifest.pop("cpu_budget")
        elif attack == "effective":
            budget["effective_cpus"] += 1
        elif attack == "threads":
            budget["threads_per_child"] = 3
        elif attack == "workers":
            budget["max_concurrent_children"] = 4
        elif attack == "groups":
            budget["groups"] = 2
        elif attack == "environment":
            manifest["environment"]["OMP_THREAD_LIMIT"] = "99"
        elif attack == "child_environment":
            manifest["groups"][0]["environment"]["MKL_NUM_THREADS"] = "99"
        elif attack in ("quota_ratio", "missing_quota", "sibling_quota"):
            budget["system"] = "Linux"
            budget["cgroup"] = {"status": "limited", "version": 2, "views": [
                {"membership": "/", "mount_root": "/", "mountpoint": "/sys/fs/cgroup"}],
                "probes": [{"view": 0, "path": "/sys/fs/cgroup/cpu.max", "status": "present"}],
                "error": None, "limits": []}
            if attack in ("quota_ratio", "sibling_quota"):
                budget["cgroup"]["limits"] = [{"view": 0, "path": "/sys/fs/cgroup/cpu.max", "quota_us": 200000,
                                               "period_us": 100000, "cpus": 99}]
            if attack == "sibling_quota":
                budget.update(host_cpus=8, affinity_cpus=8, effective_cpus=2.0, threads_per_child=1)
                budget["cgroup"]["views"][0]["membership"] = "/tenant/job"
                budget["cgroup"]["limits"][0].update(path="/sys/fs/cgroup/unrelated/cpu.max", cpus=2.0)
                budget["cgroup"]["probes"] = [
                    {"view": 0, "path": path, "status": status} for path, status in (
                        ("/sys/fs/cgroup/tenant/job/cpu.max", "missing"),
                        ("/sys/fs/cgroup/tenant/cpu.max", "missing"),
                        ("/sys/fs/cgroup/cpu.max", "present"))]
                environment = {**validator.GATE_ENVIRONMENT, **{key: "1" for key in (
                    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
                manifest["environment"] = environment
                for group in manifest["groups"]:
                    group["environment"] = {**environment, **{key: group["temporary_directory"]
                        for key in ("TMPDIR", "TMP", "TEMP")}}
        elif attack == "malformed_count":
            budget["affinity_cpus"] = True
        elif attack == "foreign_platform":
            budget["system"] = "Linux"
            budget["cgroup"]["status"] = "not_applicable"
    edit_groups(artifact_copy[1][0], change)
    with pytest.raises(ValueError, match="SDK|CPU"):
        validate(artifact_copy)


def test_receipt_quota_chain_honors_nonroot_mount_mapping_and_rejects_reordering():
    budget = {"system": "Linux", "host_cpus": 8, "affinity_cpus": 8,
              "effective_cpus": 2.0, "groups": 3, "max_concurrent_children": 2, "threads_per_child": 1,
              "cgroup": {"status": "limited", "version": 2, "error": None,
                         "views": [{"membership": "/tenant/job", "mount_root": "/tenant",
                                    "mountpoint": "/sys/fs/cgroup"}],
                         "probes": [
                             {"view": 0, "path": "/sys/fs/cgroup/job/cpu.max", "status": "present"},
                             {"view": 0, "path": "/sys/fs/cgroup/cpu.max", "status": "present"}],
                         "limits": [
                             {"view": 0, "path": "/sys/fs/cgroup/job/cpu.max", "quota_us": None,
                              "period_us": 100000, "cpus": None},
                             {"view": 0, "path": "/sys/fs/cgroup/cpu.max", "quota_us": 200000,
                              "period_us": 100000, "cpus": 2.0}]}}
    environment = {**validator.GATE_ENVIRONMENT, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
    manifest = {"cpu_budget": budget, "environment": environment}
    assert validator.sdk_child_environment(manifest) == environment
    budget["cgroup"]["limits"].reverse()
    with pytest.raises(ValueError, match="ordered ancestor chain"):
        validator.sdk_child_environment(manifest)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("attack", ["limit", "probe", "both", "reordered", "sibling"])
def test_limiting_ancestor_or_probe_cannot_be_omitted_from_rebound_receipt(version, attack):
    name = "cpu.max" if version == 2 else "cpu.cfs_quota_us"
    leaf = "/sys/fs/cgroup/tenant/job/" + name
    parent = "/sys/fs/cgroup/tenant/" + name
    mount = "/sys/fs/cgroup/" + name
    budget = {"system": "Linux", "host_cpus": 8, "affinity_cpus": 8,
              "effective_cpus": 2.0, "groups": 3, "max_concurrent_children": 2, "threads_per_child": 1,
              "cgroup": {"status": "limited", "version": version, "error": None,
                         "views": [{"membership": "/tenant/job", "mount_root": "/",
                                    "mountpoint": "/sys/fs/cgroup"}],
                         "probes": [
                             {"view": 0, "path": path, "status": status}
                             for path, status in ((leaf, "present"), (parent, "present"),
                                                  (mount, "missing"))],
                         "limits": [
                             {"view": 0, "path": path, "quota_us": quota,
                              "period_us": 100000, "cpus": quota / 100000}
                             for path, quota in ((leaf, 400000), (parent, 200000))]}}
    environment = {**validator.GATE_ENVIRONMENT, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
    manifest = {"cpu_budget": budget, "environment": environment}
    assert validator.sdk_child_environment(manifest) == environment
    if attack in ("limit", "both"):
        budget["cgroup"]["limits"].pop()
    if attack in ("probe", "both"):
        budget["cgroup"]["probes"].pop(1)
    if attack == "reordered":
        budget["cgroup"]["probes"].reverse()
    if attack == "sibling":
        budget["cgroup"]["probes"][1]["path"] = "/sys/fs/cgroup/unrelated/" + name
    budget.update(effective_cpus=4.0, threads_per_child=2)
    manifest["environment"] = {**environment, **{key: "2" for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
    with pytest.raises(ValueError, match="ancestor probe"):
        validator.sdk_child_environment(manifest)


@pytest.mark.parametrize("status", ["unavailable", "malformed"])
def test_failed_cpu_probe_receipt_may_be_partial_only_with_conservative_allocation(status):
    budget = {"system": "Linux", "host_cpus": 8, "affinity_cpus": 8,
              "effective_cpus": 1.0, "groups": 3, "max_concurrent_children": 1, "threads_per_child": 1,
              "cgroup": {"status": status, "version": 2, "error": "fixture quota read failed",
                         "views": [{"membership": "/tenant/job", "mount_root": "/",
                                    "mountpoint": "/sys/fs/cgroup"}],
                         "probes": [
                             {"view": 0, "path": "/sys/fs/cgroup/tenant/job/cpu.max", "status": "present"},
                             {"view": 0, "path": "/sys/fs/cgroup/tenant/cpu.max", "status": "present"}],
                         "limits": [{"view": 0, "path": "/sys/fs/cgroup/tenant/job/cpu.max",
                                     "quota_us": 400000, "period_us": 100000, "cpus": 4.0}]}}
    environment = {**validator.GATE_ENVIRONMENT, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
    manifest = {"cpu_budget": budget, "environment": environment}
    assert validator.sdk_child_environment(manifest) == environment
    budget.update(effective_cpus=4.0, threads_per_child=2)
    with pytest.raises(ValueError, match="CPU budget or child allocation"):
        validator.sdk_child_environment(manifest)


def test_previous_two_file_partition_cannot_be_replayed_as_the_rebalanced_source(artifact_copy):
    component = artifact_copy[1][0]

    def old_partition(manifest):
        manifest["groups"].pop()
        manifest["groups"][0]["selectors"] = SELECTORS[2:4]
        manifest["groups"][1]["selectors"] = SELECTORS[:2] + SELECTORS[4:]

    edit_groups(component, old_partition)
    with pytest.raises(ValueError, match="Three distinct SDK child groups"):
        validate(artifact_copy)


def test_previous_four_file_partition_cannot_be_replayed_as_preregistered_five(artifact_copy):
    component = artifact_copy[1][0]
    report_path = component / "receipt/report.json"
    report_before = json.loads(report_path.read_text())
    version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
    manifest_path = component / f"receipt/sdk-{version}-groups/report.json"
    manifest_before = json.loads(manifest_path.read_text())

    def old_partition(manifest):
        manifest["groups"].pop()
        manifest["groups"][0]["selectors"] = SELECTORS[:4]
        manifest["groups"][1]["selectors"] = SELECTORS[4:]

    edit_groups(component, old_partition)
    with pytest.raises(ValueError, match="Three distinct SDK child groups"):
        validate(artifact_copy)


    write_json(report_path, report_before)
    write_json(manifest_path, manifest_before)
    def previous_five_membership(manifest):
        manifest["groups"][1]["selectors"] = SELECTORS[:3] + SELECTORS[4:6]
        manifest["groups"][2]["selectors"] = SELECTORS[6:]
    edit_groups(component, previous_five_membership)
    with pytest.raises(ValueError, match="SDK child command, scope, execution or outcome differs"):
        validate(artifact_copy)

def test_third_child_pid_cannot_replay_either_prior_child(artifact_copy):
    component = artifact_copy[1][0]
    def coherently_duplicate_third_pid(manifest):
        third = manifest["groups"][2]
        third["pid"] = manifest["groups"][1]["pid"]
        version = manifest["component_sdk_version"]
        runtime = component / f"receipt/sdk-{version}-groups" / third["runtime_environment"]
        record = json.loads(runtime.read_text())
        record["pid"] = third["pid"]
        write_json(runtime, record)
        third["runtime_environment_sha256"] = digest(runtime)

    edit_groups(component, coherently_duplicate_third_pid)
    with pytest.raises(ValueError, match='process identity is duplicated'):
        validate(artifact_copy)


@pytest.mark.parametrize('attack', ['no_third_overlap', 'third_start_reordered'])
def test_concurrency_and_start_order_apply_to_all_three_actual_children(artifact_copy, attack):
    component = artifact_copy[1][0]

    def forged_clock(manifest):
        starts = [0., .01, 5.] if attack == 'no_third_overlap' else [0., .2, .1]
        for child, start in zip(manifest['groups'], starts, strict=True):
            child.update(start_seconds=start, stop_seconds=start + 3., seconds=3.)
        manifest['seconds'] = 9.

    edit_groups(component, forged_clock)
    edit_report(component, lambda report: report['steps'][-2].update(seconds=9.))
    with pytest.raises(ValueError, match='did not execute concurrently'):
        validate(artifact_copy)


def test_hosted_aggregate_refuses_actual_passing_local_group_receipt(artifact_copy, tmp_path):
    root, components, _ = artifact_copy
    base = root / "artifacts/local-proof" / tmp_path.name
    base.mkdir(parents=True)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(OPENECON_SDK_PARENT_STARTED_MONOTONIC=str(started),
               OPENECON_SDK_PARENT_DEADLINE_MONOTONIC=str(started + 900))
    result = subprocess.run([str(root / ".venv311/bin/python"), "scripts/run_parallel_sdk_groups.py",
                             "--python", str(root / ".venv311/bin/python"), "--local",
                             "--directory", str(base / "groups"), "--junitxml", str(base / "all.xml"),
                             "--gate-timings", str(base / "all.jsonl")], cwd=root, env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    local = json.loads((base / "groups/report.json").read_text())
    assert local["status"] == "passed" and local["tests"]["tests"] == 26
    assert local["execution"] is local["execution_sha256"] is local["component_sdk_version"] is None

    def substitute(manifest):
        manifest.clear()
        manifest.update(local)

    edit_groups(components[0], substitute)
    with pytest.raises(ValueError, match="Grouped SDK source, execution or scope differs"):
        validate(artifact_copy)


def test_exact_regular_uv_marker_keeps_four_real_distributions(artifact_copy):
    for component in artifact_copy[1]:
        (component / "receipt/dist/.gitignore").write_bytes(b"*")
    result = validate(artifact_copy)
    assert result["status"] == "passed"
    assert all(len(component["packages"]) == 4 for component in result["components"])


@pytest.mark.parametrize("attack", ["wrong_byte", "newline", "empty", "symlink", "directory"])
def test_uv_marker_exception_does_not_admit_changed_or_aliased_extras(
    artifact_copy, tmp_path, attack
):
    marker = artifact_copy[1][0] / "receipt/dist/.gitignore"
    if attack == "wrong_byte":
        marker.write_bytes(b"x")
    elif attack == "newline":
        marker.write_bytes(b"*\n")
    elif attack == "empty":
        marker.write_bytes(b"")
    elif attack == "symlink":
        outside = tmp_path / "uv-marker"
        outside.write_bytes(b"*")
        marker.symlink_to(outside)
    else:
        marker.mkdir()
    with pytest.raises(ValueError, match="distribution-directory marker"):
        validate(artifact_copy)


@pytest.mark.parametrize("event", ["push", "merge_group", "workflow_dispatch", "schedule"])
def test_non_pr_execution_preserves_null_parent_binding(artifact_copy, event):
    root, components, expected = artifact_copy
    expected.update(event=event, pull_request_head=None, pull_request_base=None)
    for component in components:
        path = component / "execution.json"
        execution = json.loads(path.read_text())
        execution.update(expected)
        write_json(path, execution)
        def bind(manifest):
            manifest.update(execution=execution, execution_sha256=digest(path))
            for group in manifest["groups"]:
                group["execution"] = execution
        edit_groups(component, bind)
    result = validate((root, components, expected))
    assert result["status"] == "passed" and result["verified_git"]["PR_head_tree"] is None


def test_dirty_tracked_source_cannot_be_attested_as_the_commit(artifact_copy):
    root = artifact_copy[0]
    source = root / "src/openecon/__init__.py"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"CHANGED = True\n")
        with pytest.raises(ValueError, match="tracked source differs"):
            validate(artifact_copy)
    finally:
        source.write_bytes(original)


def test_complete_components_must_have_the_same_test_order(artifact_copy):
    component = artifact_copy[1][1]
    path = component / "receipt/pytest-3.13-timings.jsonl"
    records = path.read_text().splitlines()
    path.write_text("\n".join(records[:6] + records[9:12] + records[6:9] + records[12:]) + "\n")
    raw = component / "receipt/sdk-3.13-groups/group-1/pytest-timings.jsonl"
    rows = path.read_bytes().splitlines(keepends=True)
    fixed = set(SELECTORS[:3] + SELECTORS[4:6] + SELECTORS[7:])
    raw.write_bytes(b"".join(row for row in rows if json.loads(row)["nodeid"].split("::", 1)[0] in fixed))
    collection = component / "receipt/sdk-3.13-groups/group-1/pytest-collection.jsonl"
    rows = collection.read_bytes().splitlines(keepends=True)
    collection.write_bytes(b"".join(rows[:2]) + rows[3] + rows[2] + b"".join(rows[4:]))
    for xml in (component / "receipt/pytest-3.13.xml",
                component / "receipt/sdk-3.13-groups/group-1/pytest.xml"):
        tree = ET.parse(xml)
        suite = next(tree.getroot().iter("testsuite"))
        cases = suite.findall("testcase")
        suite.remove(cases[2])
        suite.remove(cases[3])
        suite.insert(2, cases[2])
        suite.insert(2, cases[3])
        tree.write(xml)
    rebind_outputs(component)
    with pytest.raises(ValueError, match="same ordered tests"):
        validate(artifact_copy)


@pytest.mark.parametrize("attack", [
    "missing_group", "duplicate_group", "reverse_groups", "duplicate_pid",
    "missing_selector", "reversed_heavy", "duplicate_selector", "wrong_complement",
    "filtered_command", "shared_basetemp", "failed_child", "skipped_child",
    "wrong_environment", "wrong_execution", "wrong_source", "wrong_tree", "wrong_version",
    "stale_execution_hash", "relaxed_timeout", "whole_wall_exceeded", "sequential_children",
    "stop_after_parent", "negative_start", "wrong_duration", "nonfinite_duration",
    "counts_only", "hidden_error", "child_path_traversal", "wrong_kind", "serial_order_claim",
    "collection_count", "collection_hash", "collection_path",
    "wrong_plugin_environment", "borrow_local_timeout", "old900_relaxed_timeout",
    "wrong_environment_zero",
])
def test_grouped_manifest_cannot_omit_relabel_or_relax_actual_children(artifact_copy, attack):
    component = artifact_copy[1][0]

    def mutate(manifest):
        groups = manifest["groups"]
        group = groups[0]
        if attack == "missing_group":
            groups.pop()
        elif attack == "duplicate_group":
            groups[1] = copy.deepcopy(group)
        elif attack == "reverse_groups":
            groups.reverse()
        elif attack == "duplicate_pid":
            groups[1]["pid"] = group["pid"]
        elif attack == "missing_selector":
            group["selectors"].pop()
        elif attack == "reversed_heavy":
            groups[1]["selectors"].reverse()
        elif attack == "duplicate_selector":
            group["selectors"].append(group["selectors"][0])
        elif attack == "wrong_complement":
            groups[1]["selectors"] = group["selectors"]
        elif attack == "filtered_command":
            group["command"] += ["-k", "test_alpha"]
        elif attack == "shared_basetemp":
            index = groups[1]["command"].index("--basetemp") + 1
            groups[1]["command"][index] = group["command"][group["command"].index("--basetemp") + 1]
        elif attack == "failed_child":
            group.update(status="failed", exit_code=1)
        elif attack == "skipped_child":
            group["tests"]["skipped"] = 1
        elif attack == "wrong_environment":
            force_one_thread_legacy_fixture(manifest)
            group["environment"]["OMP_NUM_THREADS"] = "2"
        elif attack == "wrong_environment_zero":
            group["environment"]["OMP_NUM_THREADS"] = "0"
        elif attack == "wrong_plugin_environment":
            group["environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
        elif attack == "borrow_local_timeout":
            manifest["timeout_seconds"] = 900
        elif attack == "old900_relaxed_timeout":
            manifest["timeout_seconds"] = 901
        elif attack == "wrong_execution":
            group["execution"]["run_attempt"] = "3"
        elif attack == "wrong_source":
            manifest["source_commit"] = "9" * 40
        elif attack == "wrong_tree":
            manifest["source_tree"] = "9" * 40
        elif attack == "wrong_version":
            manifest["component_sdk_version"] = "3.13"
        elif attack == "stale_execution_hash":
            manifest["execution_sha256"] = "0" * 64
        elif attack == "relaxed_timeout":
            manifest["timeout_seconds"] = 1801
        elif attack == "whole_wall_exceeded":
            manifest["seconds"] = 1800.000001
        elif attack == "sequential_children":
            groups[1]["start_seconds"] = group["stop_seconds"]
            groups[1]["seconds"] = groups[1]["stop_seconds"] - groups[1]["start_seconds"]
        elif attack == "stop_after_parent":
            group["stop_seconds"] = manifest["seconds"] + 1
            group["seconds"] = group["stop_seconds"] - group["start_seconds"]
        elif attack == "negative_start":
            group["start_seconds"] = -1
        elif attack == "wrong_duration":
            group["seconds"] += 1
        elif attack == "nonfinite_duration":
            manifest["seconds"] = float("nan")
        elif attack == "counts_only":
            group["tests"]["tests"] += 1
        elif attack == "hidden_error":
            manifest["error"] = "A child failed"
        elif attack == "child_path_traversal":
            group["junit"] = "../pytest-3.11.xml"
        elif attack == "wrong_kind":
            manifest["kind"] = "serial"
        elif attack == "serial_order_claim":
            manifest["ordering"] = "Serial original chronology"
        elif attack == "collection_count":
            group["collected_tests"] -= 1
        elif attack == "collection_hash":
            group["collection_sha256"] = "0" * 64
        elif attack == "collection_path":
            group["collection"] = "../pytest-3.11-timings.jsonl"

    edit_groups(component, mutate)
    with pytest.raises(ValueError):
        validate(artifact_copy)


@pytest.mark.parametrize("variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
@pytest.mark.parametrize("target", ["manifest", "group-0", "group-1", "group-2"])
def test_child_receipt_cannot_exceed_recorded_thread_allocation(
    artifact_copy, variable, target
):
    def mutate(manifest):
        record = manifest if target == "manifest" else manifest["groups"][int(target[-1])]
        record["environment"][variable] = "3"

    edit_groups(artifact_copy[1][0], mutate)
    message = ("SDK numerical thread environment differs" if target == "manifest"
               else "SDK child temporary environment is not fresh and bound to its exact group")
    with pytest.raises(ValueError, match=message):
        validate(artifact_copy)


@pytest.mark.parametrize("variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
def test_parent_component_environment_must_retain_one_thread(artifact_copy, variable):
    edit_report(
        artifact_copy[1][0], lambda report: report["gate_environment"].update({variable: "2"})
    )
    with pytest.raises(ValueError, match="test environment differ"):
        validate(artifact_copy)


@pytest.mark.parametrize("attack", [
    "missing_junit", "empty_log", "changed_log", "symlink_junit", "failed_junit",
    "skipped_junit", "duplicate_junit", "case_duration", "case_output",
    "raw_phase_duration", "raw_phase_failed", "raw_phase_reordered", "raw_phase_truncated",
])
def test_raw_child_evidence_is_independently_validated_against_canonical_records(
    artifact_copy, tmp_path, attack
):
    component = artifact_copy[1][0]
    child = component / "receipt/sdk-3.11-groups/group-1"
    xml, phases, log = child / "pytest.xml", child / "pytest-timings.jsonl", child / "pytest.log"
    if attack == "missing_junit":
        xml.unlink()
    elif attack == "empty_log":
        log.write_bytes(b"")
    elif attack == "changed_log":
        log.write_bytes(log.read_bytes() + b"changed")
    elif attack == "symlink_junit":
        outside = tmp_path / "outside.xml"
        outside.write_bytes(xml.read_bytes())
        xml.unlink()
        xml.symlink_to(outside)
    elif "junit" in attack or attack in ("case_duration", "case_output"):
        tree = ET.parse(xml)
        suite = next(tree.getroot().iter("testsuite"))
        first = suite.find("testcase")
        if attack == "failed_junit":
            ET.SubElement(first, "failure")
        elif attack == "skipped_junit":
            ET.SubElement(first, "skipped")
        elif attack == "duplicate_junit":
            suite.append(copy.deepcopy(first))
            suite.set("tests", "6")
        elif attack == "case_duration":
            first.set("time", "99")
        else:
            ET.SubElement(first, "system-out").text = "Changed raw output\n"
        tree.write(xml)
    else:
        rows = phases.read_bytes().splitlines(keepends=True)
        if attack == "raw_phase_duration":
            record = json.loads(rows[1])
            record["duration"] += 1
            rows[1] = (json.dumps(record) + "\n").encode()
        elif attack == "raw_phase_failed":
            record = json.loads(rows[1])
            record["outcome"] = "failed"
            rows[1] = (json.dumps(record) + "\n").encode()
        elif attack == "raw_phase_reordered":
            rows = rows[3:6] + rows[:3] + rows[6:]
        else:
            rows.pop()
        phases.write_bytes(b"".join(rows))
    if attack not in ("missing_junit", "changed_log"):
        rebind_outputs(component)
    with pytest.raises((ValueError, OSError)):
        validate(artifact_copy)


@pytest.mark.parametrize("attack", ["case_duration", "case_output", "phase_duration", "suite_duration"])
def test_rehashed_combined_outputs_cannot_fabricate_raw_testcase_or_phase_contents(
    artifact_copy, attack
):
    component = artifact_copy[1][0]
    xml = component / "receipt/pytest-3.11.xml"
    phases = component / "receipt/pytest-3.11-timings.jsonl"
    if attack == "phase_duration":
        rows = phases.read_bytes().splitlines(keepends=True)
        record = json.loads(rows[1])
        record["duration"] += 1
        rows[1] = (json.dumps(record) + "\n").encode()
        phases.write_bytes(b"".join(rows))
    else:
        tree = ET.parse(xml)
        suite = next(tree.getroot().iter("testsuite"))
        first = suite.find("testcase")
        if attack == "suite_duration":
            suite.set("time", str(float(suite.get("time")) + 1))
        elif attack == "case_duration":
            first.set("time", "99")
        else:
            ET.SubElement(first, "system-out").text = "Fabricated combined output\n"
        tree.write(xml)
    rebind_outputs(component)
    with pytest.raises(ValueError, match="actual raw children"):
        validate(artifact_copy)


def test_missing_or_changed_group_manifest_cannot_be_replaced_by_pass_flags(artifact_copy):
    component = artifact_copy[1][0]
    edit_report(component, lambda report: report["steps"][-2].pop("sdk_groups"))
    with pytest.raises(ValueError, match="grouped SDK evidence"):
        validate(artifact_copy)


def test_parent_sdk_step_cannot_reset_the_total_deadline(artifact_copy):
    component = artifact_copy[1][0]
    edit_report(component, lambda report: report["steps"][-2].update(seconds=1800.001))
    with pytest.raises(ValueError, match="single total wall deadline"):
        validate(artifact_copy)


def test_coherent_testcase_omission_in_both_versions_cannot_pass(artifact_copy):
    for component in artifact_copy[1]:
        version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
        for xml in (component / f"receipt/pytest-{version}.xml",
                    component / f"receipt/sdk-{version}-groups/group-1/pytest.xml"):
            tree = ET.parse(xml)
            suite = next(tree.getroot().iter("testsuite"))
            suite.remove(suite.findall("testcase")[2])
            suite.set("tests", str(int(suite.get("tests")) - 1))
            tree.write(xml)
        for phase in (component / f"receipt/pytest-{version}-timings.jsonl",
                      component / f"receipt/sdk-{version}-groups/group-1/pytest-timings.jsonl"):
            records = phase.read_bytes().splitlines(keepends=True)
            phase.write_bytes(b"".join(records[:6] + records[9:]))
        edit_report(component, lambda report: report["steps"][-2]["tests"].update(
            tests=report["steps"][-2]["tests"]["tests"] - 1))

        def shrink(manifest):
            manifest["tests"]["tests"] -= 1
            manifest["combined"]["tests"]["tests"] -= 1
            manifest["groups"][1]["tests"]["tests"] -= 1
            manifest["groups"][1]["collected_tests"] -= 1

        edit_groups(component, shrink)
        rebind_outputs(component)
    with pytest.raises(ValueError, match="collection"):
        validate(artifact_copy)


@pytest.mark.parametrize("attack", [
    "missing", "empty", "reordered", "omitted", "extra", "duplicate",
    "identity", "fields", "duplicate_key", "symlink",
])
def test_same_process_collection_receipt_requires_every_collected_node_in_order(
    artifact_copy, tmp_path, attack
):
    component = artifact_copy[1][0]
    path = component / "receipt/sdk-3.11-groups/group-1/pytest-collection.jsonl"
    rows = path.read_bytes().splitlines(keepends=True)
    if attack == "missing":
        path.unlink()
    elif attack == "symlink":
        outside = tmp_path / "collection.jsonl"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        if attack == "empty":
            rows = []
        elif attack == "reordered":
            rows[0], rows[1] = rows[1], rows[0]
        elif attack == "omitted":
            rows.pop(0)
        elif attack == "extra":
            rows.append(rows[0])
        elif attack == "duplicate":
            rows[1] = rows[0]
        elif attack == "identity":
            rows[0] = (json.dumps({"nodeid": "tests/test_fake.py::test_fake"}) + "\n").encode()
        elif attack == "fields":
            record = json.loads(rows[0])
            record["count"] = 3
            rows[0] = (json.dumps(record) + "\n").encode()
        elif attack == "duplicate_key":
            rows[0] = b'{"nodeid":"a","nodeid":"b"}\n'
        path.write_bytes(b"".join(rows))
    if attack != "missing":
        rebind_outputs(component)
    with pytest.raises((ValueError, OSError)):
        validate(artifact_copy)


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "timed_out", ""])
def test_a_failed_cancelled_skipped_or_missing_job_cannot_pass(artifact_copy, result):
    with pytest.raises(ValueError, match="jobs must finish"):
        validate(artifact_copy, {"3.11": "success", "3.13": result})


@pytest.mark.parametrize("attack", ["one", "same_directory", "same_version"])
def test_one_replayed_or_duplicate_version_component_is_rejected(artifact_copy, attack):
    root, components, expected = artifact_copy
    if attack == "one":
        components.pop()
    elif attack == "same_directory":
        components[1] = components[0]
    else:
        execution = components[1] / "execution.json"
        data = json.loads(execution.read_text())
        data["component_sdk_version"] = "3.11"
        write_json(execution, data)
        shutil.rmtree(components[1])
        shutil.copytree(components[0], components[1])
    with pytest.raises(ValueError):
        validate((root, components, expected))


@pytest.mark.parametrize(
    "field",
    ["source_commit", "pull_request_head", "pull_request_base", "event", "run_id", "run_attempt"],
)
def test_stale_execution_binding_is_rejected_even_with_passing_receipts(artifact_copy, field):
    path = artifact_copy[1][1] / "execution.json"
    data = json.loads(path.read_text())
    data[field] = "push" if field == "event" else "9" * 40
    write_json(path, data)
    with pytest.raises(ValueError, match="stale"):
        validate(artifact_copy)


@pytest.mark.parametrize("thread_variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
def test_two_thread_component_cannot_replay_as_one_thread_gate(artifact_copy, thread_variable):
    component = artifact_copy[1][0]

    def mutate(report):
        report["gate_environment"][thread_variable] = "2"

    edit_report(component, mutate)
    with pytest.raises(ValueError, match="environment"):
        validate(artifact_copy)


def force_one_thread_legacy_fixture(manifest):
    """Coherently rebind this tiny receipt's CPU metadata before an adversarial child edit."""
    manifest["cpu_budget"] = {
        "system": "Darwin", "host_cpus": 3, "affinity_cpus": 3,
        "effective_cpus": 3.0, "groups": 3, "max_concurrent_children": 3,
        "threads_per_child": 1, "cgroup": {
            "status": "not_applicable", "version": None, "views": [],
            "probes": [], "limits": [], "error": None}}
    manifest["environment"] = {**validator.GATE_ENVIRONMENT, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
    for group in manifest["groups"]:
        group["environment"] = {**manifest["environment"], **{
            key: group["temporary_directory"] for key in ("TMPDIR", "TMP", "TEMP")}}


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("thread_variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
def test_two_thread_child_cannot_hide_under_one_thread_parent(artifact_copy, index, thread_variable):
    component = artifact_copy[1][0]
    def mutate(manifest):
        force_one_thread_legacy_fixture(manifest)
        manifest["groups"][index]["environment"][thread_variable] = "2"

    edit_groups(component, mutate)
    with pytest.raises(ValueError, match="SDK child"):
        validate(artifact_copy)


@pytest.mark.parametrize(
    "attack",
    [
        "full_mode",
        "full_context",
        "failed",
        "missing_step",
        "reordered_step",
        "timeout",
        "selector_removed",
        "selector_reordered",
        "filter_added",
        "plugin_removed",
        "provisioned_version",
        "executed_version",
        "inherited_filter",
        "log_changed",
    ],
)
def test_report_cannot_shrink_or_relax_the_complete_gate(artifact_copy, attack):
    component = artifact_copy[1][0]

    def mutate(report):
        if attack == "full_mode":
            report["component_mode"] = False
        elif attack == "full_context":
            report["status_context"] = "OpenEconometrics / local merge gate"
        elif attack == "failed":
            report["steps"][0]["status"] = "cancelled_or_timeout"
        elif attack == "missing_step":
            report["steps"].pop(4)
        elif attack == "reordered_step":
            report["steps"][0], report["steps"][1] = report["steps"][1], report["steps"][0]
        elif attack == "timeout":
            report["timeout_seconds"] = 1801
        elif attack == "selector_removed":
            report["selected_tests"].pop()
        elif attack == "selector_reordered":
            report["selected_tests"].reverse()
        elif attack == "filter_added":
            report["steps"][-2]["command"] += ["-k", "test_alpha"]
        elif attack == "plugin_removed":
            report["steps"][-2]["command"] += ["-p", "scripts.pytest_gate_timings"]
        elif attack == "provisioned_version":
            report["python_versions_provisioned"] = ["3.11"]
        elif attack == "executed_version":
            report["sdk_versions_executed"] = ["3.11", "3.13"]
        elif attack == "inherited_filter":
            report["gate_environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
        elif attack == "log_changed":
            (component / "receipt/ruff.log").write_text("Changed log\n")

    edit_report(component, mutate)
    with pytest.raises(ValueError):
        validate(artifact_copy)


@pytest.mark.parametrize(
    "attack",
    [
        "counts_only",
        "hidden_failure",
        "duplicate_case",
        "phase_truncated",
        "phase_failed",
        "phase_duplicate",
        "phase_identity",
        "missing_selector",
    ],
)
def test_real_testcase_and_phase_evidence_must_be_complete(artifact_copy, attack):
    component = artifact_copy[1][0]
    xml = component / "receipt/pytest-3.11.xml"
    phases = component / "receipt/pytest-3.11-timings.jsonl"
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    records = [json.loads(line) for line in phases.read_text().splitlines()]
    if attack == "counts_only":
        for case in suite.findall("testcase"):
            suite.remove(case)
    elif attack == "hidden_failure":
        ET.SubElement(suite.find("testcase"), "failure")
    elif attack == "duplicate_case":
        suite.append(copy.deepcopy(suite.find("testcase")))
        suite.set("tests", "7")
    elif attack == "phase_truncated":
        records.pop()
    elif attack == "phase_failed":
        records[1]["outcome"] = "failed"
    elif attack == "phase_duplicate":
        records[-3:] = copy.deepcopy(records[:3])
    elif attack == "phase_identity":
        records[1]["nodeid"] += "changed"
    elif attack == "missing_selector":
        suite.remove(suite.findall("testcase")[-1])
        suite.set("tests", "5")
        records = records[:-3]
    tree.write(xml)
    phases.write_text("".join(json.dumps(record) + "\n" for record in records))

    def bind_modified_evidence(report):
        sdk = report["steps"][-2]
        sdk["junit_sha256"] = digest(xml)
        sdk["phase_timings_sha256"] = digest(phases)
        if attack == "missing_selector":
            sdk["tests"]["tests"] = 5

    edit_report(component, bind_modified_evidence)
    with pytest.raises(ValueError):
        validate(artifact_copy)


@pytest.mark.parametrize(
    "attack", ["missing", "source_changed", "metadata_changed", "hash_changed", "unbound_file"]
)
def test_real_packages_are_required_and_replayed_against_checkout(artifact_copy, attack):
    component = artifact_copy[1][0]
    package = component / "receipt/dist/openecon-0.0.1-py3-none-any.whl"
    if attack == "missing":
        package.unlink()
    elif attack in ("source_changed", "metadata_changed"):
        with zipfile.ZipFile(package) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        target = (
            "openecon/__init__.py"
            if attack == "source_changed"
            else "openecon-0.0.1.dist-info/METADATA"
        )
        contents[target] = (
            b"VALUE = 'different source'\n"
            if attack == "source_changed"
            else b"Name: openecon\nVersion: 99\n"
        )
        with zipfile.ZipFile(package, "w") as archive:
            for name, raw in contents.items():
                archive.writestr(name, raw)

        def rebind(report):
            entry = next(item for item in report["packages"] if item["path"].endswith(package.name))
            entry.update(bytes=package.stat().st_size, sha256=digest(package))

        edit_report(component, rebind)
    elif attack == "hash_changed":
        package.write_bytes(package.read_bytes() + b"changed")
    else:
        (package.parent / "unbound.whl").write_bytes(b"Unbound output")
    with pytest.raises(ValueError):
        validate(artifact_copy)


@pytest.mark.parametrize("attack", ["traversal", "symlink", "duplicate_json"])
def test_artifacts_cannot_escape_or_ambiguously_replace_receipt_metadata(
    artifact_copy, tmp_path, attack
):
    component = artifact_copy[1][0]
    if attack == "traversal":
        edit_report(
            component, lambda report: report["packages"][0].update(path="../../outside.whl")
        )
    elif attack == "symlink":
        log = component / "receipt/ruff.log"
        outside = tmp_path / "outside.log"
        outside.write_bytes(log.read_bytes())
        log.unlink()
        log.symlink_to(outside)
    else:
        path = component / "receipt/report.json"
        text = path.read_text()
        path.write_text(
            text.replace('"status": "passed"', '"status": "passed", "status": "failed"', 1)
        )
    with pytest.raises(ValueError):
        validate(artifact_copy)


def test_shallow_pr_checkout_needs_both_parent_objects(genuine_artifacts, tmp_path):
    root, _, expected = genuine_artifacts
    for depth in (1, 2):
        clone = tmp_path / f"depth-{depth}"
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "--depth",
                str(depth),
                "--branch",
                "candidate",
                root.as_uri(),
                str(clone),
            ],
            check=True,
        )
        if depth == 1:
            with pytest.raises(subprocess.CalledProcessError):
                validator.expected_binding(expected, clone)
        else:
            validator.expected_binding(expected, clone)


def test_existing_large_parameter_id_is_preserved_without_truncation(tmp_path):
    test = tmp_path / "test_large_id.py"
    test.write_text(
        "import pytest\n@pytest.mark.parametrize('value', [0], ids=['x' * 3700000])\ndef test_large(value): pass\n"
    )
    xml = tmp_path / "large.xml"
    phases = tmp_path / "large.jsonl"
    collection = tmp_path / "large-collection.jsonl"
    env = os.environ.copy()
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(name, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pyproject.toml"),
            "--rootdir",
            str(tmp_path),
            "-p",
            "scripts.pytest_gate_timings",
            "--gate-timings",
            str(phases),
            "--gate-collection",
            str(collection),
            "--junitxml",
            str(xml),
            str(test),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=True,
    )
    cases = validator.junit_cases(xml)
    nodes = validator.phase_cases(phases, cases, [test.name])
    assert len(nodes) == 1 and len(nodes[0]) > 3700000
    assert nodes[0].endswith("x" * 3700000 + "]")
    validator.collection_cases(collection, nodes)


def test_malformed_step_produces_an_explicit_failed_cli_receipt(artifact_copy, tmp_path):
    root, components, expected = artifact_copy
    edit_report(components[0], lambda report: report["steps"].append(None))
    output = tmp_path / "malformed.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/verify_parallel_merge_gate.py"),
            "--component",
            str(components[0]),
            "--component",
            str(components[1]),
            "--expected-source",
            expected["source_commit"],
            "--expected-head",
            expected["pull_request_head"],
            "--expected-base",
            expected["pull_request_base"],
            "--expected-event",
            "pull_request",
            "--expected-run-id",
            expected["run_id"],
            "--expected-run-attempt",
            expected["run_attempt"],
            "--job-result",
            "3.11=success",
            "--job-result",
            "3.13=success",
            "--report",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"
    assert "omitted, repeated or reordered" in json.loads(output.read_text())["error"]


def test_cli_success_and_nonzero_failed_receipt_are_source_bound(artifact_copy, tmp_path):
    root, components, expected = artifact_copy
    output = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        str(root / "scripts/verify_parallel_merge_gate.py"),
        "--component",
        str(components[0]),
        "--component",
        str(components[1]),
        "--expected-source",
        expected["source_commit"],
        "--expected-head",
        expected["pull_request_head"],
        "--expected-base",
        expected["pull_request_base"],
        "--expected-event",
        "pull_request",
        "--expected-run-id",
        expected["run_id"],
        "--expected-run-attempt",
        expected["run_attempt"],
        "--job-result",
        "3.11=success",
        "--job-result",
        "3.13=success",
        "--report",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["status"] == "passed"
    command[command.index("3.13=success")] = "3.13=cancelled"
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "jobs must finish" in report["error"]


def utc_text(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_distributed_artifacts(genuine_artifacts, shard_count):
    """Four real small partial coordinators, synthetic authenticated API fixture.

    This proves the verifier's counterexamples, not hosted acceptance. Each
    child really collects the entire separate Git repository and runs its
    source-derived selected node partition. Packages remain actual archives.
    """
    root, legacy, expected = genuine_artifacts
    components, jobs = [], []
    for version in validator.VERSIONS:
        template = json.loads((legacy[validator.VERSIONS.index(version)] / "receipt/report.json").read_text())
        for shard in range(shard_count):
            component = root / "artifacts" / f"distributed-{shard_count}-{version}-{shard}"
            receipt = component / "receipt"
            receipt.mkdir(parents=True)
            execution = {**expected, "browser_executable": "/usr/bin/google-chrome",
                         "component_sdk_version": version, "component_sdk_shard": shard,
                         "host_cpu": {"logical": 2, "available": 2},
                         "workflow_job_name": f"OpenEconometrics / SDK Python {version} shard {shard}"}
            write_json(component / "execution.json", execution)
            report = copy.deepcopy(template)
            report.update(schema=2, component_sdk_shard=shard, sdk_shard_count=shard_count,
                          timeout_seconds=900,
                          sdk_group_count=4, partial_scope=True,
                          job_name=execution["workflow_job_name"], receipt_directory=str(receipt),
                          gate_environment=copy.deepcopy(validator.DISTRIBUTED_GATE_ENVIRONMENT))
            commands = validator.commands(report, version, SELECTORS)
            env = os.environ.copy()
            started, started_utc = time.monotonic(), time.time()
            env.update(OPENECON_SDK_PARENT_STARTED_MONOTONIC=str(started),
                       OPENECON_SDK_PARENT_DEADLINE_MONOTONIC=str(started + 900),
                       OPENECON_SDK_PARENT_STARTED_UTC=str(started_utc),
                       OPENECON_SDK_PARENT_PID=str(os.getpid()))
            log = receipt / f"sdk-{version}.log"
            with log.open("w") as output:
                result = subprocess.run(commands[f"sdk-{version}"], cwd=root, env=env,
                                        stdout=output, stderr=subprocess.STDOUT,
                                        timeout=30, check=False)
            assert result.returncode == 0, log.read_text()
            finished, finished_utc = time.monotonic(), time.time()
            manifest_path = receipt / f"sdk-{version}-groups/report.json"
            manifest = json.loads(manifest_path.read_text())
            parent_clock = {**manifest["sdk_clock"], "finished_utc": finished_utc,
                            "monotonic_finished": finished, "elapsed_seconds": finished - started}
            for step in report["steps"]:
                step["command"] = commands[step["name"]]
                path = receipt / step["log"]
                if step["name"] != f"sdk-{version}":
                    path.write_text("Fixture common-step log; not hosted evidence\n")
                step["log_sha256"] = digest(path)
            xml = receipt / f"pytest-{version}.xml"
            phases = receipt / f"pytest-{version}-timings.jsonl"
            report["steps"][-2].update(seconds=round(finished - started, 6), sdk_clock=parent_clock,
                tests=manifest["tests"], junit_sha256=digest(xml), phase_timings_sha256=digest(phases),
                sdk_groups_sha256=digest(manifest_path))
            report["packages"] = distributions(receipt, root)
            write_json(receipt / "report.json", report)
            job_id = 100 + len(components)
            jobs.append({"id": job_id, "name": execution["workflow_job_name"],
                         "run_id": int(expected["run_id"]), "run_attempt": int(expected["run_attempt"]),
                         "head_sha": expected["pull_request_head"], "status": "completed", "conclusion": "success",
                         "started_at": utc_text(started_utc - 1), "completed_at": utc_text(finished_utc + 1),
                         "runner_id": 1000 + job_id, "runner_name": f"GitHub Actions {job_id}",
                         "labels": ["ubuntu-latest"],
                         "html_url": f"https://github.com/fixture/openecon/actions/runs/{expected['run_id']}/job/{job_id}"})
            components.append(component)
    latest = max(json.loads((component / "receipt/report.json").read_text())["steps"][-2]["sdk_clock"]["finished_utc"]
                 for component in components)
    clock = {"started_utc": latest + 0.01, "finished_utc": latest + 0.1,
             "elapsed_seconds": 0.09, "monotonic_started": 100.0, "monotonic_finished": 100.09,
             "clock_identity": {"hostname": "aggregate-fixture", "pid": 500, "boot_id": None}}
    jobs.append({**jobs[0], "id": 200, "name": validator.CONTEXT,
                 "started_at": utc_text(latest - 1), "completed_at": utc_text(latest + 1),
                 "runner_id": 2000, "runner_name": "GitHub Actions 200",
                 "html_url": f"https://github.com/fixture/openecon/actions/runs/{expected['run_id']}/job/200"})
    api = root / f"artifacts/distributed-{shard_count}-jobs.json"
    write_json(api, {"unused": True})
    api.write_text(json.dumps([{"total_count": len(jobs), "jobs": jobs}]) + "\n")
    return root, components, expected, api, clock


@pytest.fixture(scope="module")
def genuine_distributed_artifacts(genuine_artifacts):
    return build_distributed_artifacts(genuine_artifacts, 2)


@pytest.fixture(scope="module")
def genuine_eight_distributed_artifacts(genuine_artifacts):
    return build_distributed_artifacts(genuine_artifacts, 4)


@pytest.fixture
def distributed_copy(genuine_distributed_artifacts, tmp_path):
    root, templates, expected, api, clock = genuine_distributed_artifacts
    components = [tmp_path / path.name for path in templates]
    for source, target in zip(templates, components, strict=True):
        shutil.copytree(source, target)
    jobs = tmp_path / "github-jobs.json"
    shutil.copyfile(api, jobs)
    return root, components, copy.deepcopy(expected), jobs, copy.deepcopy(clock)


def validate_distributed(inputs, **overrides):
    root, components, expected, jobs, clock = inputs
    options = {"root": root, "shard_count": 2, "github_jobs": jobs,
               "recorded_aggregate_clock": clock}
    options.update(overrides)
    return validator.validate_components(components, expected,
        {f"{version}:{shard}": "success" for version in validator.VERSIONS
         for shard in range(options["shard_count"])},
        **options)


def edit_jobs(inputs, change):
    path = inputs[3]
    pages = json.loads(path.read_text())
    change(pages[0]["jobs"])
    path.write_text(json.dumps(pages) + "\n")


def test_four_real_partial_shards_complete_actual_union_once_and_reassemble(distributed_copy, tmp_path):
    output = tmp_path / "complete"
    result = validate_distributed(distributed_copy, output_directory=output)
    assert result["status"] == "passed" and result["schema"] == 2
    assert result["status_context"] == validator.CONTEXT
    assert result["selected_tests"] == SELECTORS
    assert [(item["version"], item["shard"]) for item in result["components"]] == [
        ("3.11", 0), ("3.11", 1), ("3.13", 0), ("3.13", 1)]
    assert all(item["step_count"] == 8 and len(item["packages"]) == 4 for item in result["components"])
    assert all(0 < item["JUnit"]["tests"] < 26 for item in result["components"])
    for union in result["complete_sdk_unions"]:
        assert union["JUnit"] == {"tests": 26, "failures": 0, "errors": 0, "skipped": 0}
        assert union["phase_records"] == 78
        xml = output / f"pytest-{union['version']}.xml"
        phases = output / f"pytest-{union['version']}-timings.jsonl"
        assert digest(xml) == union["JUnit_sha256"] and digest(phases) == union["phase_sha256"]
        assert len(validator.phase_cases(phases, validator.junit_cases(xml), SELECTORS, ordered=True)) == 26
    assert all(0 < cohort["seconds"] <= 900 for cohort in result["sdk_cohorts"])
    assert result["aggregate_clock_mode"] == "recorded-completed-job-replay"


def test_daily_aggregate_cannot_use_the_quick_merge_check_name(distributed_copy):
    assert validator.CONTEXT == "OpenEconometrics / daily full gate"
    assert validate_distributed(distributed_copy)["status_context"] == validator.CONTEXT
    edit_jobs(distributed_copy, lambda jobs: jobs[-1].update(name="OpenEconometrics / merge gate"))
    with pytest.raises(ValueError, match="Missing or duplicate current-attempt"):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["missing", "duplicate", "legacy-mode", "no-job-bounds"])
def test_distributed_requires_all_four_unique_current_shards(distributed_copy, attack):
    if attack == "missing":
        distributed_copy[1].pop()
    elif attack == "duplicate":
        distributed_copy[1][-1] = distributed_copy[1][0]
    options = {"shard_count": 1} if attack == "legacy-mode" else {"github_jobs": None} if attack == "no-job-bounds" else {}
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy, **options)


@pytest.mark.parametrize("field,value", [
    ("run_id", 123456788), ("run_attempt", 1), ("head_sha", "1" * 40),
    ("name", "OpenEconometrics / SDK Python 3.11"), ("status", "in_progress"),
    ("conclusion", "failure"), ("runner_id", 0), ("labels", ["self-hosted"]),
    ("html_url", "https://github.com/fixture/openecon/actions/runs/1/job/100"),
    ("started_at", "2099-01-01T00:00:00Z"), ("completed_at", "2000-01-01T00:00:00Z"),
])
def test_distributed_rejects_foreign_or_unsuccessful_authenticated_job(distributed_copy, field, value):
    edit_jobs(distributed_copy, lambda jobs: jobs[0].update({field: value}))
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["duplicate-id", "duplicate-name", "missing-shard", "missing-aggregate", "active-replay"])
def test_distributed_authentication_requires_unique_sdk_and_completed_replay_job(distributed_copy, attack):
    def mutate(jobs):
        if attack == "duplicate-id":
            jobs[1]["id"] = jobs[0]["id"]
        elif attack == "duplicate-name":
            jobs.append({**jobs[0], "id": 999})
        elif attack == "missing-shard":
            jobs.pop(0)
        elif attack == "missing-aggregate":
            jobs.pop()
        else:
            jobs[-1].update(status="in_progress", conclusion=None, completed_at=None)
    edit_jobs(distributed_copy, mutate)
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("field,value", [
    ("component_sdk_shard", True), ("component_sdk_shard", 1),
    ("source_commit", "1" * 40), ("pull_request_head", "2" * 40),
    ("pull_request_base", "3" * 40), ("run_attempt", "1"),
    ("workflow_job_name", "OpenEconometrics / SDK Python 3.11 shard 1"),
])
def test_distributed_execution_is_strict_and_bound_to_current_source(distributed_copy, field, value):
    path = distributed_copy[1][0] / "execution.json"
    execution = json.loads(path.read_text())
    execution[field] = value
    write_json(path, execution)
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("cpu", [None, {}, {"logical": 2}, {"logical": 2, "available": True},
    {"logical": True, "available": 1}, {"logical": 0, "available": 1},
    {"logical": 2, "available": 0}, {"logical": 2, "available": 3},
    {"logical": 65537, "available": 2}, {"logical": 2, "available": 2, "unbound": 1}])
def test_distributed_host_capacity_is_recorded_with_strict_positive_bounds(distributed_copy, cpu):
    path = distributed_copy[1][0] / "execution.json"
    execution = json.loads(path.read_text())
    execution["host_cpu"] = cpu
    write_json(path, execution)
    with pytest.raises(ValueError, match="host CPU"):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("field,value", [
    ("schema", 1), ("sdk_shard_count", True), ("sdk_group_count", 2),
    ("partial_scope", False), ("status_context", validator.CONTEXT),
    ("job_name", "OpenEconometrics / SDK Python 3.11 shard 1"),
    ("timeout_seconds", 1800),
])
def test_distributed_partial_report_cannot_claim_complete_or_different_scope(distributed_copy, field, value):
    edit_report(distributed_copy[1][0], lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("thread", ["2", "4", "0", None])
@pytest.mark.parametrize("key", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
def test_distributed_thread_policy_is_exact_one(distributed_copy, key, thread):
    edit_report(distributed_copy[1][0], lambda report: report["gate_environment"].update({key: thread}))
    with pytest.raises(ValueError, match="environment"):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["missing_budget", "worker_count", "missing_probe", "missing_limit", "overlap"])
def test_distributed_receipt_binds_actual_cpu_probes_and_concurrency(distributed_copy, attack):
    def mutate(manifest):
        budget = {"system": "Linux", "host_cpus": 8, "affinity_cpus": 8,
                  "effective_cpus": 2.0, "groups": 2, "max_concurrent_children": 2,
                  "threads_per_child": 1, "cgroup": {
                      "status": "limited", "version": 2, "error": None,
                      "views": [{"membership": "/tenant/job", "mount_root": "/tenant",
                                 "mountpoint": "/sys/fs/cgroup"}],
                      "probes": [{"view": 0, "path": path, "status": "present"} for path in (
                          "/sys/fs/cgroup/job/cpu.max", "/sys/fs/cgroup/cpu.max")],
                      "limits": [{"view": 0, "path": path, "quota_us": quota,
                                  "period_us": 100000, "cpus": quota / 100000}
                                 for path, quota in (("/sys/fs/cgroup/job/cpu.max", 400000),
                                                      ("/sys/fs/cgroup/cpu.max", 200000))]}}
        manifest["cpu_budget"] = budget
        if attack == "missing_budget":
            manifest.pop("cpu_budget")
        elif attack == "worker_count":
            budget["max_concurrent_children"] = 3
        elif attack == "missing_probe":
            budget["cgroup"]["probes"].pop()
        elif attack == "missing_limit":
            budget["cgroup"]["limits"].pop()
            budget["effective_cpus"] = 4.0
        elif attack == "overlap":
            budget.update(effective_cpus=1.0, max_concurrent_children=1)
            budget["cgroup"]["limits"][-1].update(quota_us=100000, cpus=1.0)
    edit_groups(distributed_copy[1][0], mutate)
    with pytest.raises(ValueError, match="CPU|ancestor|concurrent"):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("distributed", [False, True])
def test_serial_metadata_cannot_hide_overlapping_raw_phases(
        artifact_copy, distributed_copy, distributed):
    inputs = distributed_copy if distributed else artifact_copy
    component = inputs[1][0]
    version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
    base = component / f"receipt/sdk-{version}-groups"
    manifest = json.loads((base / "report.json").read_text())
    records = [
        [json.loads(line) for line in (base / group["phase_timings"]).read_text().splitlines()]
        for group in manifest["groups"]
    ]
    # Shift each real child's complete phase timeline to one common start.
    # This makes the overlap counterexample deterministic without changing
    # phase order, duration, testcase identities, JUnit or collection evidence.
    anchor = min(record["start"] for child in records for record in child)
    phases = {}
    for group, child in zip(manifest["groups"], records, strict=True):
        shift = anchor - min(record["start"] for record in child)
        for record in child:
            record["start"] += shift
            record["stop"] += shift
            phases[record["nodeid"], record["phase"]] = record
        (base / group["phase_timings"]).write_text(
            "".join(json.dumps(record) + "\n" for record in child))
    combined = component / f"receipt/pytest-{version}-timings.jsonl"
    original_order = [json.loads(line) for line in combined.read_text().splitlines()]
    combined.write_text("".join(
        json.dumps(phases[record["nodeid"], record["phase"]]) + "\n"
        for record in original_order))
    rebind_outputs(component)

    def two_workers(value):
        workers = 2 if distributed else 3
        value["cpu_budget"] = {
            "system": "Linux", "host_cpus": workers, "affinity_cpus": workers,
            "effective_cpus": float(workers), "groups": workers, "max_concurrent_children": workers,
            "threads_per_child": 1, "cgroup": {
                "status": "limited", "version": 2, "error": None,
                "views": [{"membership": "/", "mount_root": "/",
                           "mountpoint": "/sys/fs/cgroup"}],
                "probes": [{"view": 0, "path": "/sys/fs/cgroup/cpu.max", "status": "present"}],
                "limits": [{"view": 0, "path": "/sys/fs/cgroup/cpu.max", "quota_us": workers * 100000,
                            "period_us": 100000, "cpus": float(workers)}]}}
        value["environment"] = {**validator.GATE_ENVIRONMENT, **{key: "1" for key in (
            "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")}}
        for group in value["groups"]:
            group["environment"] = {**copy.deepcopy(value["environment"]), **{
                key: group["temporary_directory"] for key in ("TMPDIR", "TMP", "TEMP")}}

    edit_groups(component, two_workers)
    check = validate_distributed if distributed else validate
    assert check(inputs)["status"] == "passed"
    untouched = {path: path.read_bytes() for path in base.rglob("*")
                 if path.is_file() and path.name != "report.json"}
    untouched[combined] = combined.read_bytes()

    def serial_claim(value):
        budget = value["cpu_budget"]
        budget.update(effective_cpus=1.0, max_concurrent_children=1)
        budget["cgroup"]["limits"][0].update(quota_us=100000, cpus=1.0)
        cursor = 0.0
        for group in value["groups"]:
            xml = ET.parse(base / group["junit"]).getroot()
            duration = sum(float(suite.get("time")) for suite in xml.iter("testsuite")
                           if not suite.findall("testsuite")) + 0.002
            group.update(start_seconds=cursor, stop_seconds=cursor + duration, seconds=duration)
            if distributed:
                group.update(start_utc=value["sdk_clock"]["started_utc"] + cursor,
                             stop_utc=value["sdk_clock"]["started_utc"] + cursor + duration)
            cursor += duration + 0.002
        assert cursor <= value["seconds"]

    edit_groups(component, serial_claim)
    with pytest.raises(ValueError, match="Actual SDK phases overlap"):
        check(inputs)
    assert all(path.read_bytes() == raw for path, raw in untouched.items())


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("shard_index", True), ("shard_count", 1), ("group_count", 2),
    ("source_tree", "0" * 40), ("partial_scope", False),
    ("job_name", "OpenEconometrics / SDK Python 3.13 shard 0"),
    ("timeout_seconds", 901),
])
def test_distributed_manifest_cannot_rehash_foreign_source_or_scope(distributed_copy, field, value):
    edit_groups(distributed_copy[1][0], lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


def edit_partition(component, change):
    version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
    def mutate(manifest):
        group = manifest["groups"][0]
        path = component / f"receipt/sdk-{version}-groups" / group["partition_plan"]
        plan = json.loads(path.read_text())
        change(plan)
        write_json(path, plan)
        group["partition_plan_sha256"] = digest(path)
        group["partition_plan_digest"] = hashlib.sha256(validator.canonical(plan)).hexdigest()
    edit_groups(component, mutate)


@pytest.mark.parametrize("attack", ["assignment", "counts", "weights", "algorithm", "collection", "bool-index", "load"])
def test_coherently_rehashed_plan_must_match_independent_source_partition(distributed_copy, attack):
    def mutate(plan):
        if attack == "assignment":
            plan["assignments"][0] = (plan["assignments"][0] + 1) % 4
        elif attack == "counts":
            plan["group_counts"][0] += 1
        elif attack == "weights":
            plan["weights_sha256"] = "0" * 64
        elif attack == "algorithm":
            plan["algorithm"] = "file-only"
        elif attack == "collection":
            plan["full_collection_sha256"] = "0" * 64
        elif attack == "bool-index":
            plan["assignments"][0] = False
        else:
            plan["estimated_group_seconds"][0] += 1
    edit_partition(distributed_copy[1][0], mutate)
    with pytest.raises(ValueError, match="independently derived"):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["utc-reset", "elapsed", "identity", "start-reset", "phase-outside", "child-index", "same-pid"])
def test_distributed_rejects_coherently_rehashed_clock_or_child_forgeries(distributed_copy, attack):
    component = distributed_copy[1][0]
    def mutate(manifest):
        clock = manifest["sdk_clock"]
        group = manifest["groups"][0]
        if attack == "utc-reset":
            clock["finished_utc"] = clock["started_utc"] - 1
        elif attack == "elapsed":
            clock["elapsed_seconds"] += 0.01
        elif attack == "identity":
            clock["clock_identity"]["pid"] += 1
        elif attack == "start-reset":
            clock["monotonic_started"] += 0.001
            clock["elapsed_seconds"] -= 0.001
            clock["started_utc"] += 0.001
            manifest["seconds"] = round(clock["elapsed_seconds"], 6)
        elif attack == "child-index":
            group["index"] = 2
        elif attack == "same-pid":
            manifest["groups"][1]["pid"] = group["pid"]
        else:
            version = "3.11"
            path = component / f"receipt/sdk-{version}-groups" / group["phase_timings"]
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record["start"] -= 100
                record["stop"] -= 100
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            group["phase_timings_sha256"] = digest(path)
    edit_groups(component, mutate)
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["deadline", "backwards", "monotonic", "outside-job", "before-sdk"])
def test_complete_sdk_deadline_counts_queue_gap_and_actual_merge(distributed_copy, attack):
    clock = distributed_copy[4]
    if attack == "deadline":
        earliest = min(json.loads((component / "receipt/report.json").read_text())["steps"][-2]["sdk_clock"]["started_utc"]
                       for component in distributed_copy[1][:2])
        elapsed = earliest + 900.001 - clock["started_utc"]
        clock.update(finished_utc=earliest + 900.001, elapsed_seconds=elapsed,
                     monotonic_finished=clock["monotonic_started"] + elapsed)
        edit_jobs(distributed_copy, lambda jobs: jobs[-1].update(completed_at=utc_text(clock["finished_utc"] + 1)))
    elif attack == "backwards":
        clock["finished_utc"] = clock["started_utc"] - 1
    elif attack == "monotonic":
        clock["monotonic_finished"] += 1
    elif attack == "outside-job":
        clock["started_utc"] -= 100
        clock["finished_utc"] -= 100
    else:
        clock["started_utc"] -= 0.2
        clock["finished_utc"] -= 0.2
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


def test_cross_host_child_pid_reuse_is_scoped_by_authenticated_job(distributed_copy):
    first = json.loads((distributed_copy[1][0] / "receipt/sdk-3.11-groups/report.json").read_text())
    pids = [group["pid"] for group in first["groups"]]
    def reuse(manifest):
        for group, pid in zip(manifest["groups"], pids, strict=True):
            group["pid"] = pid
            path = distributed_copy[1][1] / "receipt/sdk-3.11-groups" / group["runtime_environment"]
            record = json.loads(path.read_text())
            record["pid"] = pid
            write_json(path, record)
            group["runtime_environment_sha256"] = digest(path)
    edit_groups(distributed_copy[1][1], reuse)
    assert validate_distributed(distributed_copy)["status"] == "passed"


def test_distributed_live_clock_covers_real_reassembly_and_keeps_current_job_bounds(distributed_copy, tmp_path):
    now = time.time()
    edit_jobs(distributed_copy, lambda jobs: jobs[-1].update(status="in_progress", conclusion=None,
                                                           started_at=utc_text(now - 1), completed_at=None))
    result = validate_distributed(distributed_copy, recorded_aggregate_clock=None,
                                  output_directory=tmp_path / "live-union")
    assert result["aggregate_clock_mode"] == "live-source-validation"
    assert result["aggregate_clock"]["started_utc"] <= now + 1
    assert result["aggregate_clock"]["finished_utc"] >= now
    assert all(cohort["finished_utc"] == result["aggregate_clock"]["finished_utc"]
               for cohort in result["sdk_cohorts"])


def test_coherently_rehashed_partial_omission_cannot_replace_complete_partition(distributed_copy):
    component = distributed_copy[1][0]
    version = "3.11"
    directory = component / f"receipt/sdk-{version}-groups"
    manifest = json.loads((directory / "report.json").read_text())
    group = manifest["groups"][0]
    xml, phases, collection = (directory / group[key] for key in ("junit", "phase_timings", "collection"))
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    suite.remove(suite.findall("testcase")[0])
    suite.set("tests", str(int(suite.get("tests")) - 1))
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    phases.write_bytes(b"".join(phases.read_bytes().splitlines(keepends=True)[3:]))
    collection.write_bytes(b"".join(collection.read_bytes().splitlines(keepends=True)[1:]))
    group["tests"]["tests"] -= 1
    group["collected_tests"] -= 1
    for key in ("junit", "phase_timings", "collection"):
        group[key + "_sha256"] = digest(directory / group[key])
    # Rebuild the partial aggregate from the coherently changed actual children;
    # leave their original full collection and complete source plan intact.
    children = [directory / f"group-{index}" for index in (0, 1)]
    combined_xml = component / f"receipt/pytest-{version}.xml"
    combined_phases = component / f"receipt/pytest-{version}-timings.jsonl"
    full = validator.full_collection(children[0] / "pytest-full-collection.jsonl", SELECTORS)
    order = {node: index for index, node in enumerate(full)}
    rows, duration = [], 0.0
    for child in children:
        raw = ET.parse(child / "pytest.xml").getroot()
        duration += sum(float(s.get("time")) for s in raw.iter("testsuite"))
        data = (child / "pytest-timings.jsonl").read_bytes().splitlines(keepends=True)
        elements = list(raw.iter("testcase"))
        for index, element in enumerate(elements):
            triple = b"".join(data[index * 3:index * 3 + 3])
            rows.append((json.loads(data[index * 3])["nodeid"], element, triple))
    rows.sort(key=lambda row: order[row[0]])
    counts = {"tests": len(rows), "failures": 0, "errors": 0, "skipped": 0}
    output = ET.Element("testsuites")
    suite = ET.SubElement(output, "testsuite", name="pytest-sdk-shard", time=str(duration),
                          **{key: str(value) for key, value in counts.items()})
    for _, element, _ in rows:
        suite.append(element)
    ET.ElementTree(output).write(combined_xml, encoding="utf-8", xml_declaration=True)
    combined_phases.write_bytes(b"".join(row[2] for row in rows))
    manifest.update(tests=counts, junit_sha256=digest(combined_xml), phase_timings_sha256=digest(combined_phases))
    manifest["combined"].update(tests=counts, junit_sha256=digest(combined_xml), phase_timings_sha256=digest(combined_phases))
    write_json(directory / "report.json", manifest)
    edit_report(component, lambda report: report["steps"][-2].update(
        tests=counts, junit_sha256=digest(combined_xml), phase_timings_sha256=digest(combined_phases),
        sdk_groups_sha256=digest(directory / "report.json")))
    with pytest.raises(ValueError, match="complete partition"):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["missing", "duplicate", "reordered", "foreign", "noncanonical"])
def test_rehashed_full_collection_must_cover_source_and_bind_all_actual_nodes(distributed_copy, attack):
    component = distributed_copy[1][0]
    def mutate(manifest):
        group = manifest["groups"][0]
        directory = component / "receipt/sdk-3.11-groups"
        path = directory / group["full_collection"]
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if attack == "missing":
            records.pop(-1)  # removes a complete required selector
        elif attack == "duplicate":
            records.append(records[0])
        elif attack == "reordered":
            records.reverse()
        elif attack == "foreign":
            records[0]["nodeid"] = "tests/foreign.py::test_foreign"
        text = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
        if attack == "noncanonical":
            text = " " + text
        path.write_text(text)
        group["full_collection_sha256"] = digest(path)
        group["full_collected_tests"] = len(records)
    edit_groups(component, mutate)
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("attack", ["command-selectors", "command-k", "group-env", "raw-contents", "raw-log", "raw-path"])
def test_distributed_cannot_rehash_narrowed_or_changed_raw_child_evidence(distributed_copy, attack):
    component = distributed_copy[1][0]
    def mutate(manifest):
        group = manifest["groups"][0]
        if attack == "command-selectors":
            group["command"].pop()
        elif attack == "command-k":
            group["command"] += ["-k", "test_linear"]
        elif attack == "group-env":
            group["environment"]["OMP_NUM_THREADS"] = "2"
        elif attack == "raw-path":
            group["junit"] = "../pytest-3.11.xml"
        elif attack == "raw-log":
            path = component / "receipt/sdk-3.11-groups" / group["log"]
            path.write_text("changed raw pytest output")  # hash intentionally stale
        else:
            path = component / "receipt/sdk-3.11-groups" / group["junit"]
            tree = ET.parse(path)
            next(tree.getroot().iter("testcase")).set("time", "0.123")
            tree.write(path, encoding="utf-8", xml_declaration=True)
            group["junit_sha256"] = digest(path)
    edit_groups(component, mutate)
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


def test_distributed_cli_runs_live_bound_union_and_failure_writes_nonzero_receipt(distributed_copy, tmp_path):
    root, components, expected, jobs, _ = distributed_copy
    edit_jobs(distributed_copy, lambda records: records[-1].update(status="in_progress", conclusion=None,
        started_at=utc_text(time.time() - 1), completed_at=None))
    report = tmp_path / "cli/report.json"
    command = [sys.executable, str(root / "scripts/verify_parallel_merge_gate.py")]
    for component in components:
        command += ["--component", str(component)]
    command += ["--shard-count", "2", "--github-jobs", str(jobs)]
    for name, value in (("source", expected["source_commit"]), ("head", expected["pull_request_head"]),
                        ("base", expected["pull_request_base"]), ("event", expected["event"]),
                        ("run-id", expected["run_id"]), ("run-attempt", expected["run_attempt"])):
        command += ["--expected-" + name, value]
    for version in validator.VERSIONS:
        for shard in (0, 1):
            command += ["--job-result", f"{version}:{shard}=success"]
    command += ["--report", str(report)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(report.read_text())
    assert proof["schema"] == 2 and proof["status"] == "passed"
    assert proof["aggregate_clock_mode"] == "live-source-validation"
    for union in proof["complete_sdk_unions"]:
        assert digest(report.parent / f"complete-sdk-unions/pytest-{union['version']}.xml") == union["JUnit_sha256"]
    command[command.index("3.13:1=success")] = "3.13:1=cancelled"
    failed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert failed.returncode == 1
    assert json.loads(report.read_text())["status"] == "failed"


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("attack", [
    "global-temp", "source-temp", "sibling-temp", "missing-env", "divergent-env",
    "fresh-false", "cleanup-missing", "cleanup-false", "cleanup-integer", "cleanup-error",
    "actual-tempfile", "actual-env", "actual-pid", "schema-boolean", "symlink-integer",
    "empty-integer", "canonical-path", "missing-receipt", "changed-hash",
    "receipt-traversal", "receipt-link", "retained-root", "retained-link", "retained-file",
    "different-bootstrap",
])
def test_owned_temp_cannot_be_replayed_or_coherently_rehashed_as_foreign_runtime(
        artifact_copy, distributed_copy, tmp_path, distributed, attack):
    inputs = distributed_copy if distributed else artifact_copy
    component = inputs[1][0]
    version = json.loads((component / "execution.json").read_text())["component_sdk_version"]
    directory = component / f"receipt/sdk-{version}-groups"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "keep"
    sentinel.write_text("preserve")

    def mutate(manifest):
        group = manifest["groups"][0]
        path = directory / group["runtime_environment"]
        record = json.loads(path.read_text())
        if attack in ("global-temp", "source-temp", "sibling-temp"):
            value = {"global-temp": "/tmp", "source-temp": manifest["checkout_root"],
                     "sibling-temp": manifest["groups"][1]["temporary_directory"]}[attack]
            group["temporary_directory"] = value
            for key in ("TMPDIR", "TMP", "TEMP"):
                group["environment"][key] = value
                record["temporary_environment"][key] = value
            for key in ("owned_directory", "tempfile_directory", "canonical_directory"):
                record[key] = value
        elif attack == "missing-env":
            group["environment"].pop("TMPDIR")
        elif attack == "divergent-env":
            group["environment"]["TEMP"] = "/tmp"
        elif attack == "fresh-false":
            group["temporary_directory_fresh_before_launch"] = False
        elif attack == "cleanup-missing":
            group.pop("temporary_directory_removed_after_stop")
        elif attack in ("cleanup-false", "cleanup-integer"):
            group["temporary_directory_removed_after_stop"] = False if attack == "cleanup-false" else 1
        elif attack == "cleanup-error":
            group["temporary_cleanup_error"] = "forged cleanup failure"
        elif attack == "actual-tempfile":
            record["tempfile_directory"] = "/tmp"
        elif attack == "actual-env":
            record["temporary_environment"]["TMP"] = "/tmp"
        elif attack == "actual-pid":
            record["pid"] += 1
        elif attack == "schema-boolean":
            record["schema"] = True
        elif attack == "symlink-integer":
            record["no_symlink_ancestors"] = 1
        elif attack == "empty-integer":
            record["empty_at_bootstrap"] = 1
        elif attack == "canonical-path":
            record["canonical_directory"] += "/../runtime-temp"
        elif attack == "missing-receipt":
            path.unlink()
            return
        elif attack == "changed-hash":
            group["runtime_environment_sha256"] = "f" * 64
            return
        elif attack == "receipt-traversal":
            group["runtime_environment"] = f"group-{group['index']}/../group-{group['index']}/runtime-environment.json"
        elif attack == "receipt-link":
            target = foreign / "runtime-environment.json"
            shutil.copyfile(path, target)
            path.unlink()
            path.symlink_to(target)
            return
        elif attack.startswith("retained-"):
            temporary = directory / f"group-{group['index']}/runtime-temp"
            if attack == "retained-root":
                temporary.mkdir()
                (temporary / "unremoved-fixture").write_text("owned")
            elif attack == "retained-link":
                temporary.symlink_to(foreign, target_is_directory=True)
            else:
                temporary.write_text("retained")
        elif attack == "different-bootstrap":
            group["command"][2] += "\n"
        write_json(path, record)
        group["runtime_environment_sha256"] = digest(path)

    edit_groups(component, mutate)
    with pytest.raises(ValueError, match="SDK|child|group|runtime|temporary|[Aa]rtifact"):
        (validate_distributed if distributed else validate)(inputs)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("distributed", [False, True])
def test_all_actual_child_temp_receipts_are_distinct_same_pid_and_cleaned(
        artifact_copy, distributed_copy, distributed):
    result = (validate_distributed if distributed else validate)(distributed_copy if distributed else artifact_copy)
    children = [group for component in result["components"] for group in component["sdk_groups"]["groups"]]
    assert len(children) == (8 if distributed else 6)
    assert len({group["temporary_directory"] for group in children}) == len(children)
    for group in children:
        record = group["actual_runtime_environment"]
        assert record["pid"] == group["pid"]
        assert record["tempfile_directory"] == group["temporary_directory"]
        assert group["temporary_directory_fresh_before_launch"] is True
        assert group["temporary_directory_removed_after_stop"] is True


def test_eight_real_shards_keep_all_cases_clocks_and_reject_missing_or_duplicate_evidence(
        genuine_eight_distributed_artifacts, tmp_path):
    root, components, expected, jobs, clock = genuine_eight_distributed_artifacts
    inputs = root, components, expected, jobs, clock
    result = validate_distributed(inputs, shard_count=4, output_directory=tmp_path / "complete")
    assert result["status"] == "passed" and result["timeout_seconds"] == 900
    assert result["shard_count"] == result["group_count"] == 4
    assert [(row["version"], row["shard"]) for row in result["components"]] == [
        (version, shard) for version in validator.VERSIONS for shard in range(4)]
    assert all(len(row["sdk_groups"]["groups"]) == 1 for row in result["components"])
    for row in result["complete_sdk_unions"]:
        assert row["JUnit"] == {"tests": 26, "failures": 0, "errors": 0, "skipped": 0}
        assert row["phase_records"] == 78
        assert digest(tmp_path / f"complete/pytest-{row['version']}.xml") == row["JUnit_sha256"]
        assert digest(tmp_path / f"complete/pytest-{row['version']}-timings.jsonl") == row["phase_sha256"]
    for selected in (components[:-1], components[:4], [*components[:-1], components[0]]):
        with pytest.raises(ValueError):
            validate_distributed((root, selected, expected, jobs, clock), shard_count=4)
    with pytest.raises(ValueError):
        validate_distributed(inputs, shard_count=2)
    copied = tmp_path / "dishonest-child-count"
    shutil.copytree(components[0], copied)
    def duplicate_child(manifest):
        manifest["groups"].append(copy.deepcopy(manifest["groups"][0]))
    edit_groups(copied, duplicate_child)
    with pytest.raises(ValueError):
        validate_distributed((root, [copied, *components[1:]], expected, jobs, clock), shard_count=4)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("location", ["parent", "manifest", "child", "all"])
def test_other_component_native_recipe_cannot_be_rebound_as_current(artifact_copy, index, location):
    component = artifact_copy[1][index]
    wrong_threads = "3"
    wrong = {name: wrong_threads for name in
             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if location in ("parent", "all"):
        edit_report(component, lambda report: report["gate_environment"].update(wrong))
    if location != "parent":
        def mutate(manifest):
            if location in ("manifest", "all"):
                manifest["environment"].update(wrong)
            if location in ("child", "all"):
                for group in manifest["groups"]:
                    group["environment"].update(wrong)
        edit_groups(component, mutate)
    with pytest.raises(ValueError):
        validate(artifact_copy)


@pytest.mark.parametrize("variable", ["TMPDIR", "TMP", "TEMP"])
@pytest.mark.parametrize("attack", ["missing", "shared", "other-child"])
def test_actual_child_temporary_namespace_cannot_be_rebound(artifact_copy, variable, attack):
    component = artifact_copy[1][0]

    def mutate(manifest):
        environment = manifest["groups"][0]["environment"]
        if attack == "missing":
            environment.pop(variable)
        elif attack == "shared":
            environment[variable] = "/tmp"
        else:
            environment[variable] = manifest["groups"][1]["environment"][variable]

    edit_groups(component, mutate)
    with pytest.raises(ValueError, match="SDK child temporary environment"):
        validate(artifact_copy)


@pytest.mark.parametrize("version", [None, "3.10", "3.12", True])
def test_independent_validator_rejects_unknown_component_native_recipe(artifact_copy, version):
    component = artifact_copy[1][0]
    edit_report(component, lambda report: report.update(component_sdk_version=version))
    with pytest.raises(ValueError):
        validate(artifact_copy)


@pytest.mark.parametrize("variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
@pytest.mark.parametrize("target", ["manifest", "group-0", "group-1"])
def test_two_thread_child_receipt_cannot_replace_single_thread_contract(
    artifact_copy, variable, target
):
    def mutate(manifest):
        record = manifest if target == "manifest" else manifest["groups"][int(target[-1])]
        record["environment"][variable] = "2" if record["environment"][variable] == "1" else "1"

    edit_groups(artifact_copy[1][0], mutate)
    with pytest.raises(ValueError, match="numerical thread environment|SDK child temporary environment"):
        validate(artifact_copy)


@pytest.mark.parametrize("key", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
@pytest.mark.parametrize("target", ["manifest", "group0", "group1"])
def test_distributed_rehashed_child_environment_cannot_claim_two_threads(distributed_copy, key, target):
    def mutate(manifest):
        record = manifest if target == "manifest" else manifest["groups"][int(target[-1])]
        record["environment"][key] = "2"

    edit_groups(distributed_copy[1][0], mutate)
    with pytest.raises(ValueError):
        validate_distributed(distributed_copy)


@pytest.mark.parametrize("variable", ["TMPDIR", "TMP", "TEMP"])
@pytest.mark.parametrize("attack", ["missing", "shared", "other-child"])
def test_distributed_child_temporary_namespace_cannot_be_rebound(distributed_copy, variable, attack):
    component = distributed_copy[1][0]

    def mutate(manifest):
        environment = manifest["groups"][0]["environment"]
        if attack == "missing":
            environment.pop(variable)
        elif attack == "shared":
            environment[variable] = "/tmp"
        else:
            environment[variable] = manifest["groups"][1]["environment"][variable]

    edit_groups(component, mutate)
    with pytest.raises(ValueError, match="SDK child temporary environment"):
        validate_distributed(distributed_copy)


def _complete_protocol2_fixture_module():
    fixture_spec = importlib.util.spec_from_file_location(
        "synthetic_complete_report_fixture", ROOT / "tests/gate_report_protocol_fixture.py")
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    return fixture


def test_protocol2_independent_aggregate_counts_native_zero_times_without_clock_reset(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, expected = fixture.make(tmp_path / "independent")
    cases = validator.junit_cases(xml, report_protocol=2)
    bounds = [float("inf"), 0.0]
    nodes = validator.phase_cases(raw, cases, fixture.SELECTORS, ordered=True,
                                  offsets=True, bounds=bounds, report_protocol=2, root=ROOT)
    assert [node for node, _ in nodes] == expected
    assert len(nodes) == 122 and sum(length for _, (_, length) in nodes) == raw.stat().st_size
    assert 1000 <= bounds[0] <= bounds[1] and bounds[0] != 0
    with pytest.raises(ValueError):
        validator.junit_cases(xml)


@pytest.mark.parametrize("damage", ["wrong_context", "wrong_ordinal", "extra_nested", "wrong_framework",
                                     "wrong_source", "missing_ledger_line", "truncated_outer"])
def test_protocol2_independent_aggregate_refuses_resealed_incomplete_contract(tmp_path, damage):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, _ = fixture.make(tmp_path / damage)
    records, events = fixture.rows(raw)
    i = next(i for i, event in enumerate(events) if event["report_kind"] == "unittest_subreport")
    if damage == "wrong_context":
        events[i]["native_context"]["kwargs_in_native_order"][0][1] = "'not-the-source-value'"
    elif damage == "wrong_ordinal":
        events[i]["parent_nested_ordinal"] = 999
    elif damage == "extra_nested":
        records.insert(i, dict(records[i]))
        events.insert(i, dict(events[i]))
    elif damage == "truncated_outer":
        records.pop()
        events.pop()
    fixture.reseal(raw, records, events)
    ledger, final, _ = fixture.reports.paths(raw)
    if damage in ("wrong_framework", "wrong_source"):
        value = fixture.reports.strict_json(final.read_bytes())
        if damage == "wrong_framework":
            value["framework"]["pytest_version"] = "9.1.0"
        else:
            value["source_identity"]["parent_sources"] = {}
        final.write_bytes(fixture.reports.encode(value))
    if damage == "missing_ledger_line":
        ledger.write_bytes(b"".join(ledger.read_bytes().splitlines(True)[:-1]))
    with pytest.raises((ValueError, OSError)):
        validator.phase_cases(raw, validator.junit_cases(xml, report_protocol=2), fixture.SELECTORS,
                               ordered=True, report_protocol=2, root=ROOT)


def test_protocol2_whole_derived_origin_union_preserves_every_native_report_byte(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, nodes = fixture.make(tmp_path / "child")
    merged = tmp_path / "complete-union.jsonl"
    merged.write_bytes(raw.read_bytes())
    fixture.reports.derived_evidence(merged, nodes, [raw], ROOT)
    actual = validator.verify_derived_report_origins(merged, nodes, [raw], ROOT)
    assert actual["collected_cases"] == 122 and actual["all_phase_records"] == 391
    assert actual["native_JUnit_reported_tests"] == 147 and actual["nested_reports"] == 25
    ledger, final, _ = fixture.reports.paths(merged)
    records, rows = fixture.rows(merged)
    rows[1]["origin"]["source_raw_report_ordinal"] += 1
    ledger.write_bytes(b"".join(fixture.reports.encode_event(row, record)
                                for row, record in zip(rows, records, strict=True)))
    value = fixture.reports.strict_json(final.read_bytes())
    value["ledger"] = fixture.reports.pin(ledger, fixture.reports.MAX_LEDGER)
    final.write_bytes(fixture.reports.encode(value))
    with pytest.raises(ValueError, match="origin"):
        validator.verify_derived_report_origins(merged, nodes, [raw], ROOT)


def test_protocol2_complete_four_shard_union_preserves_all391_events_and122_elements(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    full = fixture.complete_nodes()
    rows = []
    for index in range(4):
        xml, raw, _ = fixture.make(tmp_path / f"group-{index}", full[index::4])
        cases = validator.junit_cases(xml, report_protocol=2)
        actual = validator.phase_cases(raw, cases, fixture.SELECTORS, ordered=True,
                                       offsets=True, require_all_selectors=False,
                                       report_protocol=2, root=ROOT)
        rows.append([(node, element, raw, byte_range) for (node, byte_range), element in
                     zip(actual, ET.parse(xml).getroot().iter("testcase"), strict=True)])
    results = {f"{version}:{shard}": {
        "phase_report_protocol": 2, "sdk_groups": {"partition_plan_digest": "synthetic-plan",
        "full_collection_sha256": "synthetic-collection"},
        "_raw": {"full": full, "rows": rows[shard], "duration": 1.0}}
        for version in validator.VERSIONS for shard in range(4)}
    output = tmp_path / "aggregate"
    unions = validator.merge_distributed_rows(results, fixture.SELECTORS, output, 4, root=ROOT)
    assert len(unions) == 2
    for union in unions:
        assert union["JUnit"]["tests"] == 122
        assert union["phase_records"] == 391
        assert union["report_counts"]["native_JUnit_reported_tests"] == 147
        xml = output / ("pytest-" + union["version"] + ".xml")
        raw = output / ("pytest-" + union["version"] + "-timings.jsonl")
        cases = validator.junit_cases(xml, report_protocol=2)
        assert len(cases) == 122
        assert validator.phase_cases(raw, cases, fixture.SELECTORS, ordered=True,
                                     report_protocol=2, root=ROOT, derived=True) == full
        assert len(raw.read_bytes().splitlines()) == 391


@pytest.mark.parametrize("source,expected", [
    ("TESTS = ['tests/test_bayesian_var_sbc_protocol.py']\n", 1),
    ("PHASE_REPORT_PROTOCOL = 2\nTESTS = []\n", 2),
])
def test_report_protocol_is_bound_to_actual_producer_source_without_filename_bypass(tmp_path, source, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify_merge_candidate.py").write_text(source)
    assert validator.required_report_protocol(tmp_path) == expected


@pytest.mark.parametrize("declaration", [
    "PHASE_REPORT_PROTOCOL = 1", "PHASE_REPORT_PROTOCOL = 3", "PHASE_REPORT_PROTOCOL = True",
    "PHASE_REPORT_PROTOCOL = '2'", "PHASE_REPORT_PROTOCOL = int(2)",
    "PHASE_REPORT_PROTOCOL = 2\nPHASE_REPORT_PROTOCOL = 2",
    "PHASE_REPORT_PROTOCOL: int = 2", "PHASE_REPORT_PROTOCOL = 2\nPHASE_REPORT_PROTOCOL += 0",
    "PHASE_REPORT_PROTOCOL = 2\ndef mutate():\n    global PHASE_REPORT_PROTOCOL\n    PHASE_REPORT_PROTOCOL = 1",
    "other = PHASE_REPORT_PROTOCOL = 2", "print(PHASE_REPORT_PROTOCOL)",
])
def test_unknown_duplicate_or_nonliteral_report_protocol_source_refuses(tmp_path, declaration):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify_merge_candidate.py").write_text(declaration + "\nTESTS = []\n")
    with pytest.raises(ValueError, match="explicit source report protocol"):
        validator.required_report_protocol(tmp_path)
