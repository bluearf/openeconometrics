"""Run the bounded merge gate locally and optionally publish a SHA-bound status.

This does not enable branch protection, start a paid runner, or replace the
separate scientific-scale and installed-Mac acceptance. Publishing requires a
clean, current PR head containing the current base. No existing receipt can be
replayed as a successful status.

Hosted CI may run one explicitly partial SDK shard. Both locked Python
environments and all common checks remain required; the separate aggregate
validates every declared current-run shard artifact and the single 900-second SDK
cohort before the hosted gate passes. Legacy component mode remains available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

def _report_metadata():
    try:
        from scripts import pytest_gate_report_protocol as reports
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        import pytest_gate_report_protocol as reports
    return reports


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = "OpenEconometrics / local merge gate"
PYTHONS = {"3.11", "3.13"}
PHASE_REPORT_PROTOCOL = 2
TESTS = ['tests/test_sequential_kernels.py', 'tests/test_sequential_oracles.py', 'tests/test_sequential_contracts.py', 'tests/test_mprobit.py', 'tests/test_mprobit_postestimation.py', 'tests/test_mprobit_independent.py', 'tests/test_survey_four_stage_oracles.py', 'tests/test_survey_four_stage_design.py', 'tests/test_survey_four_stage_review.py', 'tests/test_survey_four_stage_safety.py', 'tests/test_survey_four_stage_regression.py', 'tests/test_survey_four_stage_regression_state.py', 'tests/test_categorical_spline_regression.py', 'tests/test_categorical_spline_pca.py', 'tests/test_categorical_frequency_rotation.py', 'tests/test_categorical_frequency_bootstrap.py', 'tests/test_causal_assignment_extensions.py', 'tests/test_causal_neyman.py', 'tests/test_causal_identification_extensions.py', 'tests/test_causal_assignment_delivery.py', 'tests/test_causal_confidence_sets.py', 'tests/test_causal_multiarm_neyman.py', 'tests/test_causal_effect_distribution.py', 'tests/test_causal_confidence_delivery.py', 'tests/test_repeated_gls_kernels.py', 'tests/test_repeated_gls_oracles.py', 'tests/test_categorical_frequency_admission.py', 'tests/test_categorical_frequency_regression.py', 'tests/test_categorical_frequency_pca.py', 'tests/test_categorical_frequency_scaling.py', 'tests/test_nonlinear_sur.py', 'tests/test_nonlinear_sur_postestimation.py', 'tests/test_nonlinear_sur_independent.py', 'tests/test_dependent_meta.py', 'tests/test_dependent_meta_oracles.py', 'tests/test_public_package_pair_metadata.py', 'tests/test_deploy_build_metadata.py', 'tests/test_regularized_extended.py', 'tests/test_regularized_contracts.py', 'tests/test_regularized_runtime_acceptance.py', 'tests/test_regularized_glm_contracts.py', 'tests/test_regularized_glm_design.py', 'tests/test_regularized_glm_oracles.py', 'tests/test_merge_gate.py', 'tests/test_priority_release_verifier.py', 'tests/test_physical_acceptance_guards.py', 'tests/test_analysis.py', 'tests/test_dataset.py', 'tests/test_streaming_analysis.py', 'tests/test_streaming_ols_engine.py', 'tests/test_streaming_binary.py', 'tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_control_function_state.py', 'tests/test_control_function_api.py', 'tests/test_control_function_kernels.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_function_stream_state.py', 'tests/test_control_stream_acceptance.py', 'tests/test_multivariate_pca_uncertainty.py', 'tests/test_multivariate_factor_uncertainty.py', 'tests/test_manova_factorial_moments.py', 'tests/test_rm_moment_methods.py', 'tests/test_moment_admission.py', 'tests/test_survey_fully_stratified_three_stage_oracles.py', 'tests/test_survey_fully_stratified_three_stage_design.py', 'tests/test_survey_fully_stratified_three_stage_regression_state.py', 'tests/test_survey_fully_stratified_three_stage_review.py', 'tests/test_nested_logit.py', 'tests/test_nested_logit_postestimation.py', 'tests/test_nested_logit_independent.py', 'tests/test_nested_logit_delivery.py', 'tests/test_desktop_econometrics_packaging.py', 'tests/test_desktop_runtime.py', 'tests/test_survey_stratified_three_stage_oracles.py', 'tests/test_survey_stratified_three_stage_design.py', 'tests/test_survey_stratified_three_stage_regression_state.py', 'tests/test_survey_stratified_three_stage_review.py', 'tests/test_pca_subspace_uncertainty.py', 'tests/test_canon_uncertainty.py', 'tests/test_ca_uncertainty.py', 'tests/test_pca_score_uncertainty.py', 'tests/test_frequency_bootstrap_admission.py', 'tests/test_pca_frequency_uncertainty.py', 'tests/test_canon_frequency_uncertainty.py', 'tests/test_factor_frequency_uncertainty.py', 'tests/test_survey_three_stage_oracles.py', 'tests/test_survey_three_stage_safety.py', 'tests/test_survey_three_stage_regression_state.py', 'tests/test_survey_three_stage_review.py', 'tests/test_bayesian_conjugate_oracles.py', 'tests/test_bayesian_conjugate_state.py', 'tests/test_foundations_state_lifecycle.py', 'tests/test_cfa.py', 'tests/test_ivquantile.py', 'tests/test_ivquantile_oracles.py', 'tests/test_sspace_diffuse.py', 'tests/test_sspace_diffuse_kfas.py', 'tests/test_lazy_console_display.py', 'tests/test_editor_parameter_templates.py', 'tests/test_econ_streaming_prediction.py', 'tests/test_prediction_capabilities.py', 'tests/test_capability_docs.py', 'tests/test_econ_postest_suest.py', 'tests/test_econ_suest_extended.py', 'tests/test_econ_suest_binomial_commands.py', 'tests/test_editor_api_catalog.py', 'tests/test_editor_api_preservation.py', 'tests/test_latex.py', 'tests/test_output_latex.py', 'tests/test_script_packages.py', 'tests/test_uv_script_packages.py', 'packages/openecon-charts/tests', 'tests/test_parallel_merge_gate.py', 'tests/test_parallel_sdk_groups.py', 'tests/test_pytest_gate_timings.py', 'tests/test_sdist_packaging.py', 'tests/test_verify_team_cleanup.py', 'tests/test_verify_windows_cloud_ui_fixture.py', 'tests/test_twostep_kernel.py', 'tests/test_twostep_reference.py', 'tests/test_twostep_contract.py', 'tests/test_twostep_scale.py', 'tests/test_twostep_adaptive_reference.py', 'tests/test_survey_deff.py', 'tests/test_survey_inference.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py', 'tests/test_weighted_binary.py', 'tests/test_econ_glm.py', 'tests/test_econ_glm_oracle.py', 'tests/test_econ_saved_prediction_categories.py', 'tests/test_survey_fully_stratified_four_stage.py', 'tests/test_survey_fully_stratified_four_stage_safety.py', 'tests/test_confidence_sequences.py', 'tests/test_multivariate_score_uncertainty.py', 'tests/test_bayesian_var_conjugate.py', 'tests/test_bayesian_var_sbc_protocol.py', 'tests/test_bayesian_var_public_integration.py', 'tests/test_bayesian_var_public_admission_v2.py', 'tests/test_editor_catalog_intern_v2.py', 'tests/test_bayesian_bma_oracles.py', 'tests/test_bayesian_bma_state.py', 'tests/test_econ_conjoint.py', 'tests/test_econ_conjoint_robust.py', 'tests/test_multivariate_score_contrasts.py', 'tests/test_interval_weibull_regression_oracles.py', 'tests/test_interval_weibull_regression_state.py', 'tests/test_interval_weibull_publication.py', 'tests/test_econ_conjoint_bootstrap.py']


def command_json(command):
    return json.loads(subprocess.check_output(command, cwd=ROOT, text=True))


def identity():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean():
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                     cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("Commit changes and keep generated evidence in ignored artifacts/ before publishing")


def require_current_pr(repo, number, head):
    pr = command_json(["gh", "api", f"repos/{repo}/pulls/{number}"])
    base = command_json(["gh", "api", f"repos/{repo}/commits/{pr['base']['ref']}"])["sha"]
    if pr["state"] != "open" or pr["head"]["sha"] != head:
        raise ValueError("This checkout is not the current open PR head")
    if pr["head"]["repo"]["full_name"].lower() != repo.lower():
        raise ValueError("Local credential publishing is restricted to reviewed same-repository branches")
    # Fetch only the base; never checkout or execute a different PR's source.
    subprocess.run(["git", "fetch", "origin", base], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT).returncode:
        raise ValueError("Integrate the current PR base before validating this candidate")
    return base


def publish(repo, head, state, description):
    subprocess.run(["gh", "api", "--method", "POST", f"repos/{repo}/statuses/{head}",
                    "-f", f"state={state}", "-f", f"context={CONTEXT}",
                    "-f", f"description={description[:140]}"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    # pytest emits leaf suites; avoid double counting nested aggregate suites.
    suites = [s for s in suites if not s.findall("testsuite")]
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] <= 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise ValueError(f"Gate requires nonempty, passing, unskipped tests: {counts}")
    return counts


def run_step(name, command, directory, timeout, *, env=None):
    started = time.monotonic()
    started_utc = time.time()
    grouped = len(command) > 1 and Path(command[1]).name == "run_parallel_sdk_groups.py"
    distributed = grouped and "--shard-index" in command
    if grouped:
        env = dict(os.environ if env is None else env)
        env["OPENECON_SDK_PARENT_STARTED_MONOTONIC"] = repr(started)
        env["OPENECON_SDK_PARENT_DEADLINE_MONOTONIC"] = repr(started + timeout)
        env["OPENECON_SDK_PARENT_STARTED_UTC"] = repr(started_utc)
        env["OPENECON_SDK_PARENT_PID"] = str(os.getpid())
    log = directory / f"{name}.log"
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=os.name != "nt")
        try:
            code = process.wait(timeout=timeout)
            status = "passed" if code == 0 else "failed"
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            code, status = process.returncode, "cancelled_or_timeout"
    finished = time.monotonic()
    finished_utc = time.time()
    seconds = finished - started
    if grouped and seconds > timeout:
        status = "cancelled_or_timeout"
    result = {"name": name, "command": command, "status": status, "exit_code": code,
            "seconds": round(seconds, 6 if grouped else 3), "log": log.name,
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    if distributed:
        boot = Path("/proc/sys/kernel/random/boot_id")
        result["sdk_clock"] = {"started_utc": started_utc, "finished_utc": finished_utc,
                               "elapsed_seconds": seconds, "monotonic_started": started,
                               "monotonic_finished": finished,
                               "clock_identity": {"hostname": socket.gethostname(), "pid": os.getpid(),
                                                  "boot_id": boot.read_text().strip() if boot.is_file() else None}}
        if abs((finished_utc - started_utc) - seconds) > 1:
            result["status"] = "cancelled_or_timeout"
    return result


def test_environment(*, distributed=False, component=None):
    """Inherited pytest selectors/plugins must not silently shrink the gate."""
    if component is not None and component not in ("3.11", "3.13"):
        raise ValueError("Unknown SDK component native-thread recipe")
    env = os.environ.copy()
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(name, None)
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def package_manifest(directory):
    files = sorted((directory / "dist").iterdir())
    marker = directory / "dist" / ".gitignore"
    if marker in files:
        if (not marker.is_file() or marker.is_symlink() or marker.stat().st_size != 1
                or marker.read_bytes() != b"*"):
            raise ValueError("UV .gitignore sidecar must be a regular one-byte '*' file")
        files.remove(marker)
    if (len(files) != 4 or sum(path.name.endswith(".whl") for path in files) != 2
            or sum(path.name.endswith(".tar.gz") for path in files) != 2
            or any(not path.is_file() or path.is_symlink() or path.stat().st_size <= 0 for path in files)):
        raise ValueError("Gate requires both packages' actual wheel and sdist outputs")
    result = []
    for path in files:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result.append({"path": str(path.relative_to(directory)),
                       "bytes": path.stat().st_size, "sha256": digest})
    return result


def validated_protocol2_step(junit, command, sdk_groups, result, selectors):
    # Validate the actual complete stream; never turn XML147 into122 by
    # subtracting a hard-coded25 or trust an unvalidated producer counter.
    try:
        from scripts import verify_parallel_merge_gate as independent
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        import verify_parallel_merge_gate as independent
    timings = Path(command[command.index("--gate-timings") + 1])
    cases = independent.junit_cases(junit, report_protocol=2)
    nodes = independent.phase_cases(timings, cases, selectors, ordered=True,
                                    require_all_selectors="--shard-index" not in command,
                                    report_protocol=2, root=ROOT, derived=sdk_groups is not None)
    info = _report_metadata().evidence_fields(timings, nodes, root=ROOT, derived=sdk_groups is not None)
    if sdk_groups is not None:
        manifest = json.loads(sdk_groups.read_bytes())
        if manifest.get("phase_report_protocol") != 2 or not _report_metadata().same_counts(manifest.get("report_counts"), info["report_counts"]):
            raise ValueError("Grouped parent/subreport counts or protocol differ")
        children = [sdk_groups.parent / f"group-{child['index']}" / "pytest-timings.jsonl"
                    for child in manifest["groups"]]
        independent.verify_derived_report_origins(timings, nodes, children, ROOT)
        if manifest.get("kind") == "parallel_sdk_groups":
            _report_metadata().verify_capacity_prepass(sdk_groups.parent, manifest, ROOT)
    result.update(info)
    return {"tests": len(nodes), "failures": 0, "errors": 0, "skipped": 0}


def run(args):
    protocol = getattr(args, "phase_report_protocol", 1)
    if type(protocol) is not int or protocol not in (1, 2):
        raise ValueError("Unknown explicit report protocol")
    component = getattr(args, "ci_sdk_version", None)
    local_groups = getattr(args, "local_sdk_groups", False)
    shard = getattr(args, "ci_shard_index", None)
    shard_count = getattr(args, "ci_shard_count", 2)
    # These are execution budgets; scientific admission is unchanged.
    required_timeout = 900 if local_groups or shard is not None else 1800
    if getattr(args, "timeout", None) is None:
        args.timeout = required_timeout
    if local_groups:
        if component is not None:
            raise ValueError("Local SDK groups cannot be a partial CI component")
        if shard is not None:
            raise ValueError("Local SDK groups cannot be a distributed CI shard")
        if args.timeout != 900:
            raise ValueError("Local SDK groups retain the 900-second step deadline")
    if shard is not None and (component is None or type(shard) is not int
                             or type(shard_count) is not int or shard_count not in (2, 4)
                             or not 0 <= shard < shard_count):
        raise ValueError("CI shard requires an explicit SDK minor and valid index/count")
    if component is not None:
        if component not in PYTHONS:
            raise ValueError("CI SDK component requires Python 3.11 or 3.13")
        if args.publish_repo:
            raise ValueError("Partial CI components cannot publish a merge status")
        # CI wall allocation only; scientific sample/work/memory admission is unchanged.
        if args.timeout != required_timeout:
            raise ValueError(f"CI SDK components retain the {required_timeout}-second step deadline")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    require_clean()
    head = identity()
    base = None
    if args.publish_repo:
        if not args.pr:
            raise ValueError("--pr is required with --publish-repo")
        base = require_current_pr(args.publish_repo, args.pr, head)
        publish(args.publish_repo, head, "pending", "Running complete local gate on current PR head")
    report = {"schema": 1, "source_commit": head, "base_commit": base,
              "status_context": None if component else CONTEXT, "status": "running", "steps": [],
              "component_mode": component is not None, "component_sdk_version": component,
              "timeout_seconds": args.timeout, "checkout_root": str(ROOT),
              "receipt_directory": str(directory), "selected_tests": TESTS,
              "enforcement": "Not established by this script; requires GitHub branch protection",
              "scope": (f"Partial CI component: Python {component} complete selected SDK suite, Ruff, "
                        "catalogues, web and wheel/sdist build; requires the other SDK component"
                        if component else
                        "Python 3.11/3.13 bounded SDK gate, Ruff, catalogues, web and wheel/sdist build"),
              "excluded_acceptance": ["full scientific suite", "physical scale", "Mac installed acceptance",
                                      "CUDA hardware", "live cloud/team authentication"]}
    if protocol == 2:
        report["phase_report_protocol"] = 2
    if local_groups:
        report["local_sdk_groups"] = True
    if shard is not None:
        report.update(schema=2, component_sdk_shard=shard, sdk_shard_count=shard_count, sdk_group_count=4,
                      partial_scope=True,
                      job_name=f"OpenEconometrics / SDK Python {component} shard {shard}",
                      scope="Partial distributed SDK shard: all common checks/packages; requires every declared shard of both minors and actual900-second cohort aggregate")
    try:
        interpreters = {}
        for executable in args.python:
            executable = str(executable.absolute())
            version = subprocess.check_output([executable, "-c",
                "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
            if version in interpreters:
                raise ValueError("Duplicate Python version")
            interpreters[version] = executable
        if set(interpreters) != PYTHONS:
            raise ValueError(f"Gate requires exactly Python versions {sorted(PYTHONS)}")
        env = test_environment(distributed=shard is not None)
        report["python_versions_provisioned"] = sorted(interpreters)
        report["python_interpreters"] = interpreters
        report["sdk_versions_planned"] = [component] if component else sorted(interpreters)
        report["sdk_versions_executed"] = []
        report["gate_environment"] = {name: env[name] for name in (
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        report["gate_environment"].update(
            inherited_pytest_options_removed=["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"],
            explicit_plugin="scripts.pytest_gate_timings")

        def step(name, command, *, junit=None, sdk_groups=None):
            result = run_step(name, command, directory, args.timeout, env=env)
            report["steps"].append(result)
            if name.startswith("sdk-"):
                report["sdk_versions_executed"].append(name[4:])
            if sdk_groups and sdk_groups.is_file():
                result.update(sdk_groups=str(sdk_groups.relative_to(directory)),
                              sdk_groups_sha256=hashlib.sha256(sdk_groups.read_bytes()).hexdigest())
            if result["status"] != "passed":
                raise RuntimeError(f"{name} {result['status']}; inspect {directory / result['log']}")
            if junit:
                if protocol == 1:
                    result["tests"] = junit_counts(junit)
                else:
                    result["tests"] = validated_protocol2_step(junit, command, sdk_groups, result, TESTS)
                timings = Path(command[command.index("--gate-timings") + 1])
                result.update(junit=junit.name,
                              junit_sha256=hashlib.sha256(junit.read_bytes()).hexdigest(),
                              phase_timings=timings.name,
                              phase_timings_sha256=hashlib.sha256(timings.read_bytes()).hexdigest())
                if sdk_groups:
                    grouped = json.loads(sdk_groups.read_text())
                    seconds = grouped.get("seconds")
                    if (grouped.get("status") != "passed"
                            or type(seconds) not in (int, float) or not math.isfinite(seconds)
                            or not 0 <= seconds <= required_timeout
                            or type(grouped.get("timeout_seconds")) not in (int, float)
                            or grouped["timeout_seconds"] != required_timeout):
                        raise ValueError("A complete grouped SDK execution inside the original deadline is required")
                    if shard is not None:
                        if (type(grouped.get("schema")) is not int or grouped["schema"] != 2
                                or grouped.get("kind") != "distributed_sdk_shard"
                                or type(grouped.get("shard_index")) is not int or grouped["shard_index"] != shard
                                or type(grouped.get("shard_count")) is not int or grouped["shard_count"] != shard_count
                                or type(grouped.get("group_count")) is not int or grouped["group_count"] != 4
                                or grouped.get("partial_scope") is not True
                                or type(grouped.get("timeout_seconds")) not in (int, float)
                                or grouped["timeout_seconds"] != required_timeout
                                or grouped.get("source_commit") != head
                                or grouped.get("component_sdk_version") != component
                                or grouped.get("selected_tests") != TESTS):
                            raise ValueError("Distributed shard result requires genuine partial shard evidence")
                        return
                    if local_groups and (
                            any(key not in grouped or grouped[key] is not None for key in
                                ("execution", "execution_sha256", "component_sdk_version"))
                            or grouped.get("source_commit") != head
                            or grouped.get("selected_tests") != TESTS
                            or grouped.get("checkout_root") != str(ROOT)
                            or grouped.get("python") != command[0]
                            or grouped.get("timeout_seconds") != 900):
                        raise ValueError("Local SDK source/scope binding requires null hosted execution and component")
                    children = grouped.get("groups")
                    if (not isinstance(children, list) or len(children) != 3
                            or not all(isinstance(child, dict) for child in children)
                            or [child.get("index") for child in children] != [0, 1, 2]
                            or any(type(child["index"]) is not int for child in children)):
                        raise ValueError("A complete grouped SDK receipt requires all three children")
                    child_scope = []
                    total = 0
                    for child in children:
                        selected, counts = child.get("selectors"), child.get("tests")
                        if (child.get("status") != "passed" or type(child.get("exit_code")) is not int
                                or child["exit_code"] != 0 or not isinstance(selected, list) or not selected
                                or not all(isinstance(item, str) for item in selected)
                                or selected != [item for item in TESTS if item in selected]
                                or not isinstance(counts, dict)
                                or set(counts) != {"tests", "failures", "errors", "skipped"}
                                or any(type(value) is not int for value in counts.values())
                                or counts["tests"] <= 0
                                or any(counts[key] != 0 for key in ("failures", "errors", "skipped"))):
                            raise ValueError("Every grouped SDK child requires passed complete source scope")
                        child_scope.extend(selected)
                        total += counts["tests"]
                    if (len(child_scope) != len(TESTS) or set(child_scope) != set(TESTS)
                            or total != result["tests"]["tests"] or grouped.get("tests") != result["tests"]):
                        raise ValueError("Grouped SDK children differ from the complete original scope or JUnit counts")

        python = interpreters["3.13"]  # The checked-in editor catalogue uses packaging Python.
        step("ruff", [python, "-m", "ruff", "check", "src", "tests", "scripts", "packages/openecon-charts"])
        step("capabilities", [python, "scripts/generate_capability_docs.py", "--check"])
        step("editor", [python, "scripts/generate_editor_api.py", "--check"])
        step("web-install", ["npm", "ci", "--prefix", "web", "--ignore-scripts"])
        step("web-tests", ["node", "scripts/verify_web_gate.mjs"])
        step("web-build", ["npm", "--prefix", "web", "run", "build"])
        for version, executable in sorted(interpreters.items()):
            if component and version != component:
                continue
            xml = directory / f"pytest-{version}.xml"
            timings = directory / f"pytest-{version}-timings.jsonl"
            if component or local_groups:
                groups = directory / f"sdk-{version}-groups"
                command = [executable, "scripts/run_parallel_sdk_groups.py",
                     "--python", executable, "--directory", str(groups),
                     "--junitxml", str(xml), "--gate-timings", str(timings), "--timeout", str(required_timeout)]
                command += (["--local"] if local_groups else
                            ["--execution", str(directory.parent / "execution.json"),
                             "--component-sdk-version", version])
                if shard is not None:
                    command += ["--shard-index", str(shard), "--shard-count", str(shard_count)]
                if protocol == 2:
                    command += ["--phase-report-protocol", "2"]
                step(f"sdk-{version}", command, junit=xml, sdk_groups=groups / "report.json")
            else:
                command = [executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
                           "-p", "scripts.pytest_gate_timings", "--gate-timings", str(timings)]
                if protocol == 2:
                    command += ["--gate-collection", str(directory / f"pytest-{version}-collection.jsonl"),
                                *_report_metadata().options(timings)]
                step(f"sdk-{version}", [*command, "--junitxml", str(xml), *TESTS], junit=xml)
        step("packages", ["uv", "build", "--all-packages", "--out-dir", str(directory / "dist")])
        report["packages"] = package_manifest(directory)
        if identity() != head:
            raise ValueError("Source commit changed during validation")
        require_clean()
        if args.publish_repo:
            if require_current_pr(args.publish_repo, args.pr, head) != base:
                raise ValueError("Base changed during validation; rerun on the integrated candidate")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.publish_repo:
            publish(args.publish_repo, head, "success" if report["status"] == "passed" else "failure",
                    f"{report['status']}: Python 3.11/3.13 SDK, web, catalogues and packages")
    print(json.dumps({"status": report["status"], "source": head,
                      "report": str(directory / "report.json"), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, action="append", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=None,
                        help="900 seconds for explicit full local groups or distributed shards; 1800 for incoming serial/hosted legacy steps.")
    sdk_mode = parser.add_mutually_exclusive_group()
    sdk_mode.add_argument("--ci-sdk-version", choices=sorted(PYTHONS),
                          help="Run one explicitly partial hosted CI SDK component; both interpreters remain required.")
    sdk_mode.add_argument("--local-sdk-groups", action="store_true",
                          help="Run the complete selected SDK suite in three fixed parallel groups for both Python versions, within each original 900-second deadline.")
    parser.add_argument("--ci-shard-index", type=int, choices=[0, 1, 2, 3],
                        help="Run one partial distributed shard; every declared shard of both minors remains mandatory.")
    parser.add_argument("--ci-shard-count", type=int, choices=[2, 4], default=2)
    parser.add_argument("--phase-report-protocol", type=int, choices=[PHASE_REPORT_PROTOCOL], default=PHASE_REPORT_PROTOCOL,
                        help="Explicit complete parent/subreport evidence; legacy1 remains strict.")
    parser.add_argument("--publish-repo")
    parser.add_argument("--pr", type=int)
    raise SystemExit(run(parser.parse_args()))
