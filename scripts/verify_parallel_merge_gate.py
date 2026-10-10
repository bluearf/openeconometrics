"""Validate both current-run Python gate artifacts before the required CI passes.

This consumes actual downloaded component receipts. It does not execute tests,
publish statuses, or treat a partial/local receipt as a complete hosted gate.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import tarfile
import time
import tomllib
import xml.etree.ElementTree as ET
import zipfile

def _report_metadata():
    try:
        from scripts import pytest_gate_report_protocol as reports
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        import pytest_gate_report_protocol as reports
    return reports


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("3.11", "3.13")
CONTEXT = "OpenEconometrics / daily full gate"
COMMON_STEPS = ("ruff", "capabilities", "editor", "web-install", "web-tests", "web-build")
GATE_ENVIRONMENT = {
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "inherited_pytest_options_removed": ["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"],
    "explicit_plugin": "scripts.pytest_gate_timings",
}
DISTRIBUTED_GATE_ENVIRONMENT = {
    **GATE_ENVIRONMENT,
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
}
MAX_PHASE_LINE = 8 * 1024 * 1024
MAX_PHASE_BYTES = 256 * 1024 * 1024
SDK_FIXED_GROUPS = (('tests/test_control_stream_acceptance.py',), ('tests/test_econ_saved_prediction_linear.py', 'tests/test_control_function_common_prediction.py', 'tests/test_streaming_control_function_engine.py', 'tests/test_control_function_stream_state.py', 'tests/test_nested_logit_independent.py', 'tests/test_parallel_sdk_groups.py', 'tests/test_bayesian_hypothesis_oracles.py', 'tests/test_bayesian_hypothesis_state.py', 'tests/test_latent_sem_lifecycle.py', 'tests/test_latent_sem_math.py', 'tests/test_latent_sem_state.py', 'tests/test_dynamic_factor.py', 'tests/test_finite_mixture.py', 'tests/test_weakiv_clr_math.py', 'tests/test_weakiv_clr_state.py', 'tests/test_supervised.py', 'tests/test_supervised_integration_lifecycle.py', 'tests/test_five_model_public_integration.py', 'tests/test_bayesian_var_conjugate.py', 'tests/test_bayesian_var_sbc_protocol.py', 'tests/test_bayesian_var_public_integration.py', 'tests/test_bayesian_var_public_admission_v2.py', 'tests/test_editor_catalog_intern_v2.py'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_file(directory, relative):
    require(
        isinstance(relative, str) and relative and "\\" not in relative,
        "Artifact path must be a relative POSIX file",
    )
    parts = relative.split("/")
    require(
        not PurePosixPath(relative).is_absolute()
        and all(part not in ("", ".", "..") for part in parts),
        "Artifact path escapes its component",
    )
    path = directory
    for part in parts:
        path /= part
        require(not path.is_symlink(), "Artifact symlink is not accepted")
    require(
        path.is_file() and path.resolve().is_relative_to(directory.resolve()),
        "Required artifact file is missing or escapes its component",
    )
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON metadata key")
        result[key] = value
    return result


def json_value(raw):
    def reject_constant(_):
        raise ValueError("Nonfinite JSON metadata")

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(path):
    require(path.stat().st_size <= 2 * 1024 * 1024, "Receipt metadata exceeds its bound")
    value = json_value(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Receipt must be a JSON object")
    return value


def selector_contract(root):
    tree = ast.parse((root / "scripts/verify_merge_candidate.py").read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)
    ]
    require(
        len(values) == 1 and isinstance(values[0], list) and values[0],
        "Current checkout lacks its complete selector contract",
    )
    selectors = values[0]
    require(
        all(isinstance(item, str) and item for item in selectors)
        and len(set(selectors)) == len(selectors),
        "Invalid current selector contract",
    )
    return selectors


def expected_binding(values, root):
    require(
        set(values)
        == {
            "source_commit",
            "pull_request_head",
            "pull_request_base",
            "event",
            "run_id",
            "run_attempt",
        },
        "Incomplete independent execution binding",
    )
    for key in ("source_commit", "pull_request_head", "pull_request_base"):
        value = values[key]
        require(
            (value is None and key != "source_commit")
            or (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)),
            "Invalid expected commit binding",
        )
    require(
        values["event"] in ("pull_request", "push", "merge_group", "workflow_dispatch", "schedule"),
        "Unsupported expected workflow event",
    )
    if values["event"] == "pull_request":
        require(
            values["pull_request_head"] is not None and values["pull_request_base"] is not None,
            "Pull-request execution needs both independent parent bindings",
        )
    else:
        require(
            values["pull_request_head"] is None and values["pull_request_base"] is None,
            "Non-PR execution must not claim PR parents",
        )
    require(
        all(
            isinstance(values[key], str) and re.fullmatch(r"[1-9][0-9]*", values[key])
            for key in ("run_id", "run_attempt")
        ),
        "Invalid independent workflow run binding",
    )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(
        source == values["source_commit"], "Aggregate checkout differs from expected tested source"
    )
    require(
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root).returncode == 0,
        "Aggregate tracked source differs from its tested commit",
    )
    commit = subprocess.check_output(["git", "cat-file", "-p", "HEAD"], cwd=root, text=True)
    headers = commit.split("\n\n", 1)[0].splitlines()
    parents = [line.removeprefix("parent ") for line in headers if line.startswith("parent ")]
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    head_tree = None
    if values["event"] == "pull_request":
        require(
            parents == [values["pull_request_base"], values["pull_request_head"]],
            "Tested PR merge does not have the expected base/head parents",
        )
        head_tree = subprocess.check_output(
            ["git", "rev-parse", values["pull_request_head"] + "^{tree}"], cwd=root, text=True
        ).strip()
        require(
            source_tree == head_tree,
            "Tested merge tree differs from the PR head; integrate its base",
        )
    return {
        "source_commit": source,
        "source_tree": source_tree,
        "parents": parents,
        "PR_head_tree": head_tree,
    }


def commands(report, version, selectors):
    checkout = report.get("checkout_root")
    receipt = report.get("receipt_directory")
    require(
        isinstance(checkout, str)
        and PurePosixPath(checkout).is_absolute()
        and str(PurePosixPath(checkout)) == checkout
        and ".." not in PurePosixPath(checkout).parts,
        "Invalid recorded checkout root",
    )
    require(
        isinstance(receipt, str)
        and PurePosixPath(receipt).is_absolute()
        and str(PurePosixPath(receipt)) == receipt
        and PurePosixPath(receipt).is_relative_to(PurePosixPath(checkout))
        and ".." not in PurePosixPath(receipt).parts,
        "Invalid recorded receipt directory",
    )
    interpreters = report.get("python_interpreters")
    require(
        isinstance(interpreters, dict) and set(interpreters) == set(VERSIONS),
        "Both locked Python interpreters must be recorded",
    )
    for key, executable in interpreters.items():
        require(
            executable == f"{checkout}/.venv{key.replace('.', '')}/bin/python",
            "Recorded interpreter differs from its provisioned environment",
        )
    python = interpreters["3.13"]
    result = {
        "ruff": [
            python,
            "-m",
            "ruff",
            "check",
            "src",
            "tests",
            "scripts",
            "packages/openecon-charts",
        ],
        "capabilities": [python, "scripts/generate_capability_docs.py", "--check"],
        "editor": [python, "scripts/generate_editor_api.py", "--check"],
        "web-install": ["npm", "ci", "--prefix", "web", "--ignore-scripts"],
        "web-tests": ["node", "scripts/verify_web_gate.mjs"],
        "web-build": ["npm", "--prefix", "web", "run", "build"],
        f"sdk-{version}": [
            interpreters[version],
            "scripts/run_parallel_sdk_groups.py",
            "--python",
            interpreters[version],
            "--directory",
            f"{receipt}/sdk-{version}-groups",
            "--junitxml",
            f"{receipt}/pytest-{version}.xml",
            "--gate-timings",
            f"{receipt}/pytest-{version}-timings.jsonl",
            "--timeout",
            "900" if report.get("schema") == 2 else "1800",
            "--execution",
            str(PurePosixPath(receipt).parent / "execution.json"),
            "--component-sdk-version",
            version,
        ],
        "packages": ["uv", "build", "--all-packages", "--out-dir", f"{receipt}/dist"],
    }
    if report.get("schema") == 2:
        result[f"sdk-{version}"] += ["--shard-index", str(report["component_sdk_shard"]),
                                     "--shard-count", str(report["sdk_shard_count"])]
    if report.get("phase_report_protocol") == 2:
        result[f"sdk-{version}"] += ["--phase-report-protocol", "2"]
    return result


def _junit_cases_legacy(path):
    require(path.stat().st_size <= 32 * 1024 * 1024, "JUnit metadata exceeds its bound")
    raw = path.read_bytes()
    require(
        b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "JUnit declarations are not accepted"
    )
    root = ET.fromstring(raw)
    require(root.tag in ("testsuite", "testsuites"), "Invalid JUnit document")
    require(
        not any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")),
        "JUnit contains an unsuccessful outcome",
    )
    suites = [suite for suite in root.iter("testsuite") if not suite.findall("testsuite")]
    require(suites, "JUnit has no leaf test suite")
    result = []
    for suite in suites:
        counts = {}
        for key in ("tests", "failures", "errors", "skipped"):
            value = suite.get(key)
            require(
                isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "Invalid JUnit counts"
            )
            counts[key] = int(value)
        cases = suite.findall("testcase")
        require(
            counts["tests"] == len(cases)
            and len(cases) > 0
            and not any(counts[key] for key in ("failures", "errors", "skipped")),
            "JUnit must contain actual nonempty passing unskipped cases",
        )
        for case in cases:
            require(
                not any(case.findall(tag) for tag in ("failure", "error", "skipped")),
                "JUnit case contains an unsuccessful outcome",
            )
            classname, name = case.get("classname"), case.get("name")
            require(
                isinstance(classname, str) and classname and isinstance(name, str) and name,
                "JUnit case lacks its identity",
            )
            result.append((classname, name))
    require(len(result) == len(set(result)), "Duplicate JUnit testcase identity")
    require(len(list(root.iter("testcase"))) == len(result), "Uncounted JUnit testcase")
    return result


def _junit_cases_protocol2(path):
    reports = _report_metadata()
    require(path.stat().st_size <= 32 * 1024 * 1024, "JUnit metadata exceeds its bound")
    raw = path.read_bytes()
    require(
        b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "JUnit declarations are not accepted"
    )
    root = ET.fromstring(raw)
    require(root.tag in ("testsuite", "testsuites"), "Invalid JUnit document")
    require(
        not any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")),
        "JUnit contains an unsuccessful outcome",
    )
    suites = [suite for suite in root.iter("testsuite") if not suite.findall("testsuite")]
    require(suites, "JUnit has no leaf test suite")
    result = []
    for suite in suites:
        counts = {}
        for key in ("tests", "failures", "errors", "skipped"):
            value = suite.get(key)
            require(
                isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "Invalid JUnit counts"
            )
            counts[key] = int(value)
        cases = suite.findall("testcase")
        require(
            counts["tests"] == len(cases) + sum(reports.nested_identity(c.get("classname"), c.get("name")) for c in cases)
            and len(cases) > 0
            and not any(counts[key] for key in ("failures", "errors", "skipped")),
            "JUnit must contain actual nonempty passing unskipped cases",
        )
        for case in cases:
            require(
                not any(case.findall(tag) for tag in ("failure", "error", "skipped")),
                "JUnit case contains an unsuccessful outcome",
            )
            classname, name = case.get("classname"), case.get("name")
            require(
                isinstance(classname, str) and classname and isinstance(name, str) and name,
                "JUnit case lacks its identity",
            )
            result.append((classname, name))
    require(len(result) == len(set(result)), "Duplicate JUnit testcase identity")
    require(len(list(root.iter("testcase"))) == len(result), "Uncounted JUnit testcase")
    return result


def junit_cases(path, *, report_protocol=1):
    require(type(report_protocol) is int and report_protocol in (1, 2), "Unknown report protocol")
    return _junit_cases_protocol2(path) if report_protocol == 2 else _junit_cases_legacy(path)


def _phase_cases_legacy(path, cases, selectors, *, ordered=False, offsets=False,
                require_all_selectors=True, bounds=None):
    require(path.stat().st_size <= MAX_PHASE_BYTES, "Phase evidence exceeds its byte bound")
    groups = []
    pending = []
    seen = set()
    ranges = []
    with path.open("rb") as stream:
        position = 0
        for line in iter(lambda: stream.readline(MAX_PHASE_LINE + 1), b""):
            start = position
            position += len(line)
            require(len(line) <= MAX_PHASE_LINE and line.strip(), "Invalid phase record")
            record = json_value(line)
            require(
                isinstance(record, dict)
                and set(record) == {"nodeid", "phase", "outcome", "duration", "start", "stop"},
                "Invalid phase record fields",
            )
            require(record["outcome"] == "passed", "Phase report is not passed")
            require(
                all(
                    type(record[key]) in (int, float)
                    and math.isfinite(record[key])
                    and record[key] >= 0
                    for key in ("duration", "start", "stop")
                )
                and record["stop"] >= record["start"],
                "Invalid phase timings",
            )
            if bounds is not None:
                bounds[0] = min(bounds[0], record["start"])
                bounds[1] = max(bounds[1], record["stop"])
            nodeid = record["nodeid"]
            require(isinstance(nodeid, str) and "::" in nodeid, "Invalid phase node identity")
            phase = ("setup", "call", "teardown")[len(pending)]
            require(
                record["phase"] == phase and (not pending or pending[0] == nodeid),
                "Phase records do not contain complete ordered test triples",
            )
            pending.append(nodeid)
            if len(pending) == 1:
                triple_start = start
            if len(pending) == 3:
                require(nodeid not in seen, "Duplicate completed phase identity")
                seen.add(nodeid)
                groups.append(nodeid)
                ranges.append((triple_start, position - triple_start))
                pending = []
    require(
        not pending and groups and len(groups) == len(cases),
        "Incomplete or truncated phase evidence",
    )
    files = {nodeid.split("::", 1)[0] for nodeid in groups}
    require(
        not require_all_selectors or all(
            any(file == selector or file.startswith(selector.rstrip("/") + "/") for file in files)
            for selector in selectors
        ),
        "A required selector has no completed testcase",
    )
    require(
        all(
            any(
                file == selector or file.startswith(selector.rstrip("/") + "/")
                for selector in selectors
            )
            for file in files
        ),
        "Phase evidence contains tests outside the selectors",
    )
    # Validate each source once, then bind each distinct JUnit class once.
    # Keep every matching module: ambiguous dotted paths or module/class
    # prefixes must still refuse, rather than being overwritten in a map.
    modules = []
    for file in files:
        require(
            file.endswith(".py")
            and not PurePosixPath(file).is_absolute()
            and ".." not in PurePosixPath(file).parts,
            "Invalid phase source path",
        )
        modules.append((file, file[:-3].replace("/", ".")))
    class_prefixes = {}
    expected_nodes = []
    for classname, name in cases:
        if classname not in class_prefixes:
            matches = []
            for file, module in modules:
                if classname == module:
                    matches.append(file + "::")
                elif classname.startswith(module + "."):
                    classes = classname[len(module) + 1 :].replace(".", "::")
                    matches.append(file + "::" + classes + "::")
            require(len(matches) == 1, "JUnit testcase cannot be bound to one phase source")
            class_prefixes[classname] = matches[0]
        expected_nodes.append(class_prefixes[classname] + name)
    require(
        Counter(expected_nodes) == Counter(groups), "JUnit and phase testcase identities differ"
    )
    if ordered:
        require(expected_nodes == groups, "JUnit and phase testcase order differs")
        file_ranks = {file: selector_index(file, selectors) for file in files}
        ranks = [file_ranks[node.split("::", 1)[0]] for node in groups]
        require(ranks == sorted(ranks), "Completed testcase selector order differs")
    return list(zip(groups, ranges, strict=True)) if offsets else groups


def _phase_cases_protocol2(path, cases, selectors, *, ordered=False, offsets=False,
                require_all_selectors=True, bounds=None, root=None, derived=False):
    reports = _report_metadata()
    root = ROOT if root is None else Path(root)
    require(path.stat().st_size <= MAX_PHASE_BYTES, "Phase evidence exceeds its byte bound")
    groups, pending, ranges, contexts = [], [], [], []
    seen = set()
    position = 0
    expected_phase, parent, native_location = "setup", None, None
    for record, event, line, _ in reports.bound_reports(path, derived=derived):
        start = position
        position += len(line)
        nodeid = record["nodeid"]
        if event["report_kind"] == "unittest_subreport":
            require(expected_phase == "call" and parent == nodeid
                    and nodeid in reports.CONTEXTS
                    and event["native_location"] == native_location
                    and event["parent_nested_ordinal"] == len(contexts),
                    "Subreport is outside the exact declared parent/call context")
            contexts.append(event["native_context"])
            pending.append(nodeid)
            continue
        require(record["phase"] == expected_phase and (parent is None or parent == nodeid),
                "Protocol2 outer reports are incomplete or reordered")
        if expected_phase == "setup":
            require(nodeid not in seen, "Repeated protocol2 parent")
            parent, native_location, triple_start = nodeid, event["native_location"], start
        else:
            require(event["native_location"] == native_location, "Native outer location differs")
        # unittest.addSubTest has genuine zero CallInfo times. Only the three
        # original outer phases contribute real authenticated UTC bounds.
        if bounds is not None:
            bounds[0] = min(bounds[0], record["start"])
            bounds[1] = max(bounds[1], record["stop"])
        if expected_phase == "call":
            reports.check_contexts(nodeid, contexts)
        pending.append(nodeid)
        if expected_phase == "teardown":
            seen.add(nodeid)
            groups.append(nodeid)
            ranges.append((triple_start, position - triple_start))
            pending, contexts, parent, native_location, expected_phase = [], [], None, None, "setup"
        else:
            expected_phase = "call" if expected_phase == "setup" else "teardown"
    require(
        not pending and groups and len(groups) == len(cases),
        "Incomplete or truncated phase evidence",
    )
    files = {nodeid.split("::", 1)[0] for nodeid in groups}
    require(
        not require_all_selectors or all(
            any(file == selector or file.startswith(selector.rstrip("/") + "/") for file in files)
            for selector in selectors
        ),
        "A required selector has no completed testcase",
    )
    require(
        all(
            any(
                file == selector or file.startswith(selector.rstrip("/") + "/")
                for selector in selectors
            )
            for file in files
        ),
        "Phase evidence contains tests outside the selectors",
    )
    # Validate each source once, then bind each distinct JUnit class once.
    # Keep every matching module: ambiguous dotted paths or module/class
    # prefixes must still refuse, rather than being overwritten in a map.
    modules = []
    for file in files:
        require(
            file.endswith(".py")
            and not PurePosixPath(file).is_absolute()
            and ".." not in PurePosixPath(file).parts,
            "Invalid phase source path",
        )
        modules.append((file, file[:-3].replace("/", ".")))
    class_prefixes = {}
    expected_nodes = []
    for classname, name in cases:
        if classname not in class_prefixes:
            matches = []
            for file, module in modules:
                if classname == module:
                    matches.append(file + "::")
                elif classname.startswith(module + "."):
                    classes = classname[len(module) + 1 :].replace(".", "::")
                    matches.append(file + "::" + classes + "::")
            require(len(matches) == 1, "JUnit testcase cannot be bound to one phase source")
            class_prefixes[classname] = matches[0]
        expected_nodes.append(class_prefixes[classname] + name)
    require(
        Counter(expected_nodes) == Counter(groups), "JUnit and phase testcase identities differ"
    )
    if ordered:
        require(expected_nodes == groups, "JUnit and phase testcase order differs")
        file_ranks = {file: selector_index(file, selectors) for file in files}
        ranks = [file_ranks[node.split("::", 1)[0]] for node in groups]
        require(ranks == sorted(ranks), "Completed testcase selector order differs")
    reports.terminal(path, groups, root=root, derived=derived)
    return list(zip(groups, ranges, strict=True)) if offsets else groups


def phase_cases(path, cases, selectors, *, ordered=False, offsets=False,
                require_all_selectors=True, bounds=None, report_protocol=1, root=None, derived=False):
    require(type(report_protocol) is int and report_protocol in (1, 2), "Unknown report protocol")
    if report_protocol == 1:
        require(not derived, "Legacy reports cannot claim a protocol2 derived ledger")
        return _phase_cases_legacy(path, cases, selectors, ordered=ordered, offsets=offsets,
                                  require_all_selectors=require_all_selectors, bounds=bounds)
    return _phase_cases_protocol2(path, cases, selectors, ordered=ordered, offsets=offsets,
                                 require_all_selectors=require_all_selectors, bounds=bounds,
                                 root=root, derived=derived)


def selector_index(file, selectors):
    matches = [
        index
        for index, selector in enumerate(selectors)
        if file == selector or file.startswith(selector.rstrip("/") + "/")
    ]
    require(len(matches) == 1, "Testcase source does not bind to exactly one selector")
    return matches[0]


def element_value(element):
    """Preserve each raw testcase's contents while ignoring XML indentation."""
    return (
        element.tag,
        sorted(element.attrib.items()),
        element.text if not len(element) or (element.text and element.text.strip()) else None,
        [(element_value(child), child.tail if child.tail and child.tail.strip() else None)
         for child in element],
    )


def collection_cases(path, nodes):
    require(path.stat().st_size <= MAX_PHASE_BYTES, "Collection evidence exceeds its byte bound")
    with path.open("rb") as stream:
        for node in nodes:
            line = stream.readline(MAX_PHASE_LINE + 1)
            require(line and len(line) <= MAX_PHASE_LINE and line.strip(),
                    "Incomplete or excessive collection evidence")
            record = json_value(line)
            require(
                isinstance(record, dict) and set(record) == {"nodeid"}
                and record["nodeid"] == node,
                "Same-process collection testcase identities or order differ from raw JUnit and phases",
            )
        require(not stream.read(1), "Collection evidence contains extra testcase identities")


def sdk_child_environment(manifest):
    """Validate recorded resource evidence and derive the only permitted allocation."""
    budget = manifest.get("cpu_budget")
    require(isinstance(budget, dict) and set(budget) == {
        "system", "host_cpus", "affinity_cpus", "cgroup", "effective_cpus", "groups",
        "threads_per_child", "max_concurrent_children"}, "Missing or invalid SDK CPU budget")
    require(isinstance(budget["system"], str) and budget["system"]
            and all(value is None or (type(value) is int and value > 0)
                    for value in (budget["host_cpus"], budget["affinity_cpus"])),
            "Invalid SDK detected CPU counts")
    cgroup = budget["cgroup"]
    require(isinstance(cgroup, dict) and set(cgroup) == {
        "status", "version", "views", "probes", "limits", "error"}
        and cgroup["status"] in ("limited", "unlimited", "unavailable", "malformed", "not_applicable")
        and (cgroup["version"] is None or (type(cgroup["version"]) is int
                                          and cgroup["version"] in (1, 2)))
        and all(isinstance(cgroup[key], list) for key in ("views", "probes", "limits"))
        and (not (cgroup["views"] or cgroup["probes"] or cgroup["limits"])
             or cgroup["version"] in (1, 2)),
        "Invalid SDK cgroup evidence")
    available = cgroup["status"] in ("limited", "unlimited")
    require((cgroup["status"] == "not_applicable") == (budget["system"] != "Linux")
            and (isinstance(cgroup["error"], str) and bool(cgroup["error"])
                 if cgroup["status"] in ("unavailable", "malformed") else cgroup["error"] is None)
            and (not available or (cgroup["version"] in (1, 2) and cgroup["limits"]
                                   and cgroup["views"])),
            "SDK cgroup availability differs from its evidence")
    chains = []
    for view in cgroup["views"]:
        require(isinstance(view, dict) and set(view) == {"membership", "mount_root", "mountpoint"}
                and all(isinstance(value, str) and PurePosixPath(value).is_absolute()
                        and ".." not in PurePosixPath(value).parts for value in view.values()),
                "Invalid SDK cgroup mount view")
        member, mount_root, mount = (PurePosixPath(view[key])
                                     for key in ("membership", "mount_root", "mountpoint"))
        require(member.is_relative_to(mount_root), "SDK membership lies outside its cgroup mount root")
        current = mount.joinpath(*member.relative_to(mount_root).parts)
        chain = []
        while True:
            chain.append(current)
            if current == mount:
                break
            current = current.parent
        chains.append(chain)
    quota_name = "cpu.max" if cgroup["version"] == 2 else "cpu.cfs_quota_us"
    expected_probes = [(view, str(current / quota_name))
                       for view, chain in enumerate(chains) for current in chain]
    probed, present = [], []
    for probe in cgroup["probes"]:
        require(isinstance(probe, dict) and set(probe) == {"view", "path", "status"}
                and type(probe["view"]) is int and 0 <= probe["view"] < len(chains)
                and isinstance(probe["path"], str)
                and probe["status"] in ("present", "missing"), "Invalid SDK CPU ancestor probe")
        identity = (probe["view"], probe["path"])
        probed.append(identity)
        if probe["status"] == "present":
            present.append(identity)
    require(probed == expected_probes if available else
            probed == expected_probes[:len(probed)],
            "SDK CPU ancestor probe chain is incomplete or reordered")
    quotas, paths = [], []
    previous = {}
    for item in cgroup["limits"]:
        require(isinstance(item, dict) and set(item) == {"view", "path", "quota_us", "period_us", "cpus"}
                and type(item["view"]) is int and 0 <= item["view"] < len(chains)
                and isinstance(item["path"], str)
                and PurePosixPath(item["path"]).parent in chains[item["view"]]
                and ".." not in PurePosixPath(item["path"]).parts
                and PurePosixPath(item["path"]).name == (
                    "cpu.max" if cgroup["version"] == 2 else "cpu.cfs_quota_us")
                and type(item["period_us"]) is int and item["period_us"] > 0
                and (item["quota_us"] is None or (type(item["quota_us"]) is int
                                                  and item["quota_us"] > 0)),
                "Invalid SDK CPU quota")
        rank = chains[item["view"]].index(PurePosixPath(item["path"]).parent)
        require(rank > previous.get(item["view"], -1), "SDK quotas are outside their ordered ancestor chain")
        previous[item["view"]] = rank
        try:
            expected = None if item["quota_us"] is None else item["quota_us"] / item["period_us"]
        except OverflowError as error:
            raise ValueError("Nonfinite SDK CPU quota") from error
        require(item["cpus"] is None if expected is None else
                type(item["cpus"]) in (int, float) and math.isfinite(item["cpus"])
                and item["cpus"] > 0 and item["cpus"] == expected, "SDK quota CPU ratio differs")
        paths.append((item["view"], item["path"]))
        if expected is not None:
            quotas.append(expected)
    require(paths == present if available else paths == present[:len(paths)],
            "SDK CPU quotas differ from present ancestor probes")
    require(len(paths) == len(set(paths))
            and (not available or (bool(quotas) == (cgroup["status"] == "limited")))
            and (cgroup["status"] != "not_applicable" or
                 (not paths and cgroup["version"] is None and not cgroup["views"]
                  and not cgroup["probes"])),
            "SDK CPU quota evidence is inconsistent")
    limits = [value for value in (budget["host_cpus"], budget["affinity_cpus"])
              if value is not None] + quotas
    if cgroup["status"] in ("unavailable", "malformed"):
        limits.append(1)
    effective = max(1.0, min(limits, default=1.0))
    distributed = manifest.get("kind") == "distributed_sdk_shard"
    if distributed:
        shard_count = manifest.get("shard_count")
        require(type(shard_count) is int and shard_count in (2, 4),
                "Invalid SDK CPU topology shard count")
        group_count = 4 // shard_count
    else:
        group_count = 3
    local = (manifest.get("kind") == "parallel_sdk_groups"
             and "execution" in manifest and manifest["execution"] is None)
    threads = 1 if distributed or local else min(2, max(1, math.floor(effective / 3)))
    workers = min(group_count, max(1, math.floor(effective)))
    require(type(budget["effective_cpus"]) in (int, float)
            and math.isfinite(budget["effective_cpus"]) and budget["effective_cpus"] == effective
            and type(budget["groups"]) is int and budget["groups"] == group_count
            and type(budget["max_concurrent_children"]) is int
            and budget["max_concurrent_children"] == workers
            and type(budget["threads_per_child"]) is int and budget["threads_per_child"] == threads,
            "SDK CPU budget or child allocation differs from detected limits")
    expected = {**GATE_ENVIRONMENT, **{key: str(threads) for key in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "ARROW_IO_THREADS")}}
    require(manifest.get("environment") == expected, "SDK numerical thread environment differs")
    return expected


def child_concurrency(manifest, phase_bounds):
    groups = manifest["groups"]
    require(len(groups) == len(phase_bounds)
            and len({group["pid"] for group in groups}) == len(groups),
            "SDK child process identity is duplicated")
    starts = [group["start_seconds"] for group in groups]
    require(starts == sorted(starts), "SDK children did not execute concurrently in source start order")
    if manifest.get("kind") == "distributed_sdk_shard" and manifest.get("shard_count") == 4:
        require(len(groups) == 1,
                "A four-shard job must contain exactly one verified child")
        return
    workers = manifest["cpu_budget"]["max_concurrent_children"]
    events = sorted((value, change) for group in groups
                    for value, change in ((group["start_seconds"], 1), (group["stop_seconds"], -1)))
    active, maximum = 0, 0
    for _, change in events:
        active += change
        maximum = max(maximum, active)
        require(0 <= active <= workers,
                "SDK children did not execute concurrently within their CPU allocation")
    require(active == 0 and (workers == 1 or maximum == workers),
            "SDK children did not execute concurrently within their CPU allocation")
    if workers == len(groups):
        require(max(starts) < min(group["stop_seconds"] for group in groups),
                "SDK children did not execute concurrently across all actual children")
    phase_events = sorted((value, change) for start, stop in phase_bounds
                          for value, change in ((start, 1), (stop, -1)))
    active = 0
    for _, change in phase_events:
        active += change
        require(0 <= active <= workers,
                "Actual SDK phases overlap the recorded CPU allocation")
    require(active == 0, "Actual SDK phases overlap the recorded CPU allocation")


def source_child_bootstrap(root):
    tree = ast.parse((Path(root) / "scripts/run_parallel_sdk_groups.py").read_text())
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "SDK_CHILD_BOOTSTRAP"
                      for target in node.targets)]
    require(len(values) == 1 and isinstance(values[0], str) and 0 < len(values[0]) <= 4096,
            "Missing exact source-pinned SDK child bootstrap")
    return values[0]


def child_temporary_environment(directory, group, child, environment):
    temporary = child + "/runtime-temp"
    require(temporary.startswith("/") and not temporary.startswith("//")
            and "\\" not in temporary and "\0" not in temporary
            and all(part not in ("", ".", "..") for part in temporary.split("/")[1:])
            and str(PurePosixPath(temporary)) == temporary,
            "SDK temporary directory is not an absolute canonical owned path")
    expected = {**environment, **{key: temporary for key in ("TMPDIR", "TMP", "TEMP")}}
    require(group.get("environment") == expected
            and group.get("temporary_directory") == temporary
            and group.get("temporary_directory_fresh_before_launch") is True,
            "SDK child temporary environment is not fresh and bound to its exact group")
    relative = f"group-{group['index']}/runtime-environment.json"
    require(group.get("runtime_environment") == relative,
            "Missing exact SDK child temporary runtime receipt path")
    path = artifact_file(directory, relative)
    require(0 < path.stat().st_size <= 4096
            and sha256(path) == group.get("runtime_environment_sha256"),
            "Actual SDK child temporary runtime receipt is missing or changed")
    record = read_json(path)
    require(record == {"schema": 1, "pid": group["pid"],
                      "temporary_environment": {key: temporary for key in ("TMPDIR", "TMP", "TEMP")},
                      "owned_directory": temporary, "tempfile_directory": temporary,
                      "canonical_directory": temporary, "no_symlink_ancestors": True,
                      "empty_at_bootstrap": True}
            and type(record.get("schema")) is int and type(record.get("pid")) is int
            and record.get("no_symlink_ancestors") is True
            and record.get("empty_at_bootstrap") is True,
            "Actual SDK child tempfile path/PID/environment differs from its owned group")
    archived_temporary = directory / f"group-{group['index']}/runtime-temp"
    require(group.get("temporary_directory_removed_after_stop") is True
            and "temporary_cleanup_error" not in group
            and not archived_temporary.exists() and not archived_temporary.is_symlink(),
            "SDK owned temporary directory was not cleaned after its child stopped")
    return record


def unique_child_temporary_directories(components):
    paths = [group["temporary_directory"] for component in components
             for group in component["sdk_groups"]["groups"]]
    require(len(paths) == len(set(paths)),
            "SDK children share a temporary directory instead of owning distinct group paths")


def grouped_sdk(directory, report, sdk, execution, version, selectors, cases, nodes):
    protocol = report.get("phase_report_protocol", 1)
    require(type(protocol) is int and protocol in (1, 2), "Unknown component report protocol")
    reports = _report_metadata() if protocol == 2 else None
    relative = f"sdk-{version}-groups/report.json"
    require(sdk.get("sdk_groups") == relative, "Missing grouped SDK evidence binding")
    path = artifact_file(directory, "receipt/" + relative)
    require(sha256(path) == sdk.get("sdk_groups_sha256"), "Grouped SDK manifest hash differs")
    manifest = read_json(path)
    require(manifest.get("phase_report_protocol", 1) == protocol, "Child manifest protocol differs")
    child_environment = sdk_child_environment(manifest)
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=report["checkout_root"], text=True
    ).strip()
    require(
        type(manifest.get("schema")) is int
        and manifest["schema"] == 1
        and manifest.get("kind") == "parallel_sdk_groups"
        and manifest.get("status") == "passed"
        and "error" not in manifest
        and manifest.get("source_commit") == report["source_commit"]
        and manifest.get("source_tree") == tree
        and manifest.get("execution") == execution
        and manifest.get("execution_sha256") == sha256(directory / "execution.json")
        and manifest.get("component_sdk_version") == version
        and manifest.get("checkout_root") == report["checkout_root"]
        and manifest.get("python") == report["python_interpreters"][version]
        and manifest.get("selected_tests") == selectors
        and manifest.get("environment") == child_environment
        and manifest.get("ordering")
        == "Raw children preserve execution order; merged actual cases follow original selectors",
        "Grouped SDK source, execution or scope differs",
    )
    require(
        type(manifest.get("timeout_seconds")) in (int, float)
        and manifest["timeout_seconds"] == 1800
        and type(manifest.get("seconds")) in (int, float)
        and math.isfinite(manifest["seconds"])
        and 0 < manifest["seconds"] <= 1800
        and manifest["seconds"] <= sdk["seconds"] + 0.0000011
        and sdk["seconds"] <= 1800
        and manifest.get("duration_precision")
        == "seconds rounded to six decimal places; one parent monotonic clock",
        "Grouped SDK did not retain its single total wall deadline",
    )
    require(
        len(set(selectors)) == len(selectors)
        and set(item for group in SDK_FIXED_GROUPS for item in group) <= set(selectors)
        and not any(
            a.startswith(b.rstrip("/") + "/") for a in selectors for b in selectors if a != b
        ),
        "Complete SDK selectors cannot be partitioned unambiguously",
    )
    fixed = [item for group in SDK_FIXED_GROUPS for item in group]
    partition = [[item for item in selectors if item in group] for group in SDK_FIXED_GROUPS]
    partition.append([item for item in selectors if item not in fixed])
    groups = manifest.get("groups")
    require(
        isinstance(groups, list)
        and len(groups) == 3
        and all(isinstance(group, dict) for group in groups)
        and [group.get("index") for group in groups] == [0, 1, 2]
        and all(type(group["index"]) is int for group in groups)
        and all(partition),
        "Three distinct SDK child groups are required",
    )
    if protocol == 2:
        reports.verify_capacity_prepass(path.parent, manifest, ROOT)
    actual = []
    checked = []
    raw_duration = 0.0
    phase_bounds = []
    bootstrap = source_child_bootstrap(report["checkout_root"])
    for index, (group, selected) in enumerate(zip(groups, partition, strict=True)):
        child = f"{report['receipt_directory']}/sdk-{version}-groups/group-{index}"
        expected_command = [
            report["python_interpreters"][version], "-c", bootstrap,
            child + "/runtime-environment.json", "-c",
            f"{report['checkout_root']}/pyproject.toml", "-p", "scripts.pytest_gate_timings",
            "--gate-timings", child + "/pytest-timings.jsonl",
            "--gate-collection", child + "/pytest-collection.jsonl",
            "--junitxml", child + "/pytest.xml", "--basetemp", child + "/pytest-temp",
            "-o", "cache_dir=" + child + "/pytest-cache",
            *selected,
        ]
        if protocol == 2:
            at = expected_command.index("--gate-full-collection") if "--gate-full-collection" in expected_command else expected_command.index("--junitxml")
            expected_command[at:at] = reports.options(Path(child) / "pytest-timings.jsonl")
            at = expected_command.index("--junitxml")
            expected_command[at:at] = ["--gate-report-capacity-source",
                f"{report['receipt_directory']}/sdk-{version}-groups/capacity-admission/native-location-collection.jsonl"]
        require(
            group.get("selectors") == selected
            and group.get("command") == expected_command
            and group.get("status") == "passed"
            and type(group.get("exit_code")) is int and group["exit_code"] == 0
            and type(group.get("pid")) is int and group["pid"] > 0
            and group.get("execution") == execution,
            "SDK child command, scope, execution or outcome differs",
        )
        runtime_environment = child_temporary_environment(path.parent, group, child,
                                                           child_environment)
        require(
            all(
                type(group.get(key)) in (int, float)
                and math.isfinite(group[key]) and group[key] >= 0
                for key in ("start_seconds", "stop_seconds", "seconds")
            )
            and group["start_seconds"] <= group["stop_seconds"] <= manifest["seconds"]
            and abs(group["seconds"] - (group["stop_seconds"] - group["start_seconds"])) <= 0.0000011,
            "SDK child timing differs from the single parent wall clock",
        )
        files = {}
        for key, filename in (
            ("log", "pytest.log"), ("junit", "pytest.xml"),
            ("phase_timings", "pytest-timings.jsonl"),
            ("collection", "pytest-collection.jsonl"),
        ):
            require(group.get(key) == f"group-{index}/{filename}", "Unexpected SDK child path")
            files[key] = artifact_file(path.parent, group[key])
            require(
                files[key].stat().st_size > 0
                and sha256(files[key]) == group.get(key + "_sha256"),
                "SDK child evidence is missing, empty or changed",
            )
        if protocol == 2:
            for key, evidence in zip(("report_ledger", "report_terminal"), reports.paths(files["phase_timings"])[:2], strict=True):
                require(group.get(key) == f"group-{index}/{evidence.name}"
                        and group.get(key + "_sha256") == sha256(evidence),
                        "Native ledger/terminal path or hash differs")
        raw_cases = junit_cases(files["junit"], report_protocol=protocol)
        require(
            group.get("tests")
            == {"tests": len(raw_cases), "failures": 0, "errors": 0, "skipped": 0},
            "SDK child counts differ from actual raw cases",
        )
        bounds = [math.inf, 0.0]
        raw_nodes = phase_cases(
            files["phase_timings"], raw_cases, selected, ordered=True, offsets=True,
            bounds=bounds, report_protocol=protocol, root=ROOT,
        )
        phase_bounds.append(bounds)
        require(
            type(group.get("collected_tests")) is int
            and group["collected_tests"] == len(raw_nodes),
            "SDK child collection count differs from actual completed cases",
        )
        collection_cases(files["collection"], [node for node, _ in raw_nodes])
        raw_xml = ET.parse(files["junit"]).getroot()
        elements = list(raw_xml.iter("testcase"))
        child_duration = 0.0
        for suite in raw_xml.iter("testsuite"):
            if not suite.findall("testsuite"):
                value = float(suite.get("time", "0"))
                require(math.isfinite(value) and value >= 0, "Invalid raw SDK JUnit duration")
                child_duration += value
        require(
            math.isfinite(child_duration) and child_duration <= group["seconds"] + 0.001001,
            "Raw SDK JUnit duration exceeds its actual child wall time",
        )
        raw_duration += child_duration
        for (node, byte_range), element in zip(raw_nodes, elements, strict=True):
            actual.append((node, element_value(element), files["phase_timings"], byte_range))
        checked.append({
            key: group[key] for key in (
                "index", "selectors", "command", "pid", "status", "exit_code", "tests",
                "start_seconds", "stop_seconds", "seconds", "log", "log_sha256",
                "junit", "junit_sha256", "phase_timings", "phase_timings_sha256",
                "collection", "collection_sha256", "collected_tests",
                "environment", "temporary_directory", "temporary_directory_fresh_before_launch",
                "temporary_directory_removed_after_stop",
                "runtime_environment", "runtime_environment_sha256",
            )
        })
        if protocol == 2:
            report_counts = reports.counts([node for node, _ in raw_nodes], reports.expected_nested([node for node, _ in raw_nodes]))
            require(group.get("phase_report_protocol") == 2 and reports.same_counts(group.get("report_counts"), report_counts),
                    "Native child parent/subreport/JUnit counters differ")
            checked[-1].update(phase_report_protocol=2, report_counts=report_counts,
                              report_ledger_sha256=group["report_ledger_sha256"],
                              report_terminal_sha256=group["report_terminal_sha256"])
        checked[-1]["actual_runtime_environment"] = runtime_environment
    child_concurrency(manifest, phase_bounds)
    actual.sort(key=lambda row: selector_index(row[0].split("::", 1)[0], selectors))
    require(
        [row[0] for row in actual] == nodes and len(actual) == len(cases)
        and len({row[0] for row in actual}) == len(actual),
        "Combined SDK identities or ordering differ from the complete raw children",
    )
    expected_extra = {}
    if protocol == 2:
        merged_raw = artifact_file(directory, "receipt/" + sdk["phase_timings"])
        children_raw = [path.parent / f"group-{index}" / "pytest-timings.jsonl" for index in [0, 1, 2]]
        info = reports.evidence_fields(merged_raw, nodes, root=ROOT, derived=True)
        verify_derived_report_origins(merged_raw, nodes, children_raw, ROOT)
        expected_extra = info
        require(all((reports.same_counts(sdk.get(k), v) and reports.same_counts(manifest.get(k), v))
                    if k == "report_counts" else sdk.get(k) == v and manifest.get(k) == v
                    for k, v in info.items()),
                "Complete combined report metadata/counters differ")
    combined = manifest.get("combined")
    require(
        combined == {
            "junit": sdk["junit"], "junit_sha256": sdk["junit_sha256"],
            "phase_timings": sdk["phase_timings"],
            "phase_timings_sha256": sdk["phase_timings_sha256"], "tests": sdk["tests"], **expected_extra,
        }
        and manifest.get("tests") == sdk["tests"]
        and manifest.get("junit_sha256") == sdk["junit_sha256"]
        and manifest.get("phase_timings_sha256") == sdk["phase_timings_sha256"],
        "Grouped SDK combined-output binding differs",
    )
    merged_xml = artifact_file(directory, "receipt/" + sdk["junit"])
    merged_root = ET.parse(merged_xml).getroot()
    merged_elements = list(merged_root.iter("testcase"))
    merged_suites = list(merged_root.iter("testsuite"))
    require(
        len(merged_suites) == 1
        and merged_suites[0].get("name") == "pytest-sdk-groups"
        and float(merged_suites[0].get("time", "nan")) == raw_duration,
        "Combined JUnit duration differs from actual raw children",
    )
    require(
        [element_value(element) for element in merged_elements] == [row[1] for row in actual],
        "Combined JUnit testcase contents differ from the actual raw children",
    )
    merged_phases = artifact_file(directory, "receipt/" + sdk["phase_timings"])
    raw_streams = {}
    try:
        with merged_phases.open("rb") as combined_stream:
            for _, _, raw_path, (offset, length) in actual:
                if raw_path not in raw_streams:
                    raw_streams[raw_path] = raw_path.open("rb")
                raw_stream = raw_streams[raw_path]
                raw_stream.seek(offset)
                while length:
                    block = raw_stream.read(min(length, 1024 * 1024))
                    require(
                        block and combined_stream.read(len(block)) == block,
                        "Combined phase records differ from the actual raw children",
                    )
                    length -= len(block)
            require(not combined_stream.read(1), "Combined phase evidence has extra records")
    finally:
        for stream in raw_streams.values():
            stream.close()
    return {"manifest": relative, "manifest_sha256": sha256(path),
            "kind": manifest["kind"], "seconds": manifest["seconds"], "groups": checked,
            "cpu_budget": manifest["cpu_budget"], "environment": child_environment}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def clock_value(value):
    require(isinstance(value, dict) and set(value) == {
        "started_utc", "finished_utc", "elapsed_seconds", "monotonic_started",
        "monotonic_finished", "clock_identity"}, "Invalid SDK/aggregate clock fields")
    require(all(type(value[key]) in (int, float) and math.isfinite(value[key])
                and value[key] > 0 for key in value if key != "clock_identity"),
            "Invalid SDK/aggregate clock values")
    elapsed = value["monotonic_finished"] - value["monotonic_started"]
    require(elapsed > 0 and abs(elapsed - value["elapsed_seconds"]) <= 0.0000011
            and abs(value["finished_utc"] - value["started_utc"] - elapsed) <= 1,
            "UTC/monotonic clock rollback or elapsed mismatch")
    identity = value["clock_identity"]
    require(isinstance(identity, dict) and set(identity) == {"hostname", "pid", "boot_id"}
            and isinstance(identity["hostname"], str) and identity["hostname"]
            and type(identity["pid"]) is int and identity["pid"] > 0
            and (identity["boot_id"] is None or isinstance(identity["boot_id"], str)
                 and re.fullmatch(r"[0-9a-f-]{36}", identity["boot_id"])),
            "Invalid host-scoped process clock identity")
    return value


def aggregate_clock(started):
    monotonic, utc = started
    finished, finished_utc = time.monotonic(), time.time()
    boot = Path("/proc/sys/kernel/random/boot_id")
    return clock_value({"started_utc": utc, "finished_utc": finished_utc,
                        "elapsed_seconds": finished - monotonic,
                        "monotonic_started": monotonic, "monotonic_finished": finished,
                        "clock_identity": {"hostname": socket.gethostname(), "pid": __import__("os").getpid(),
                                           "boot_id": boot.read_text().strip() if boot.is_file() else None}})


def independent_plan(root, nodes):
    """Derive all assignments independently from the current tracked cost constants."""
    tree = ast.parse((root / "scripts/run_parallel_sdk_groups.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "DEFAULT_NODE_SECONDS", "ADVISORY_NODE_SECONDS"):
                    require(target.id not in values, "Repeated scheduling source constant")
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {"DEFAULT_NODE_SECONDS", "ADVISORY_NODE_SECONDS"}
            and isinstance(values["ADVISORY_NODE_SECONDS"], dict),
            "Missing source scheduling constants")
    default, advisory = values["DEFAULT_NODE_SECONDS"], values["ADVISORY_NODE_SECONDS"]
    require(all(isinstance(key, str) and "::" in key for key in advisory)
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in [default, *advisory.values()]), "Invalid scheduling source weights")
    weights = [advisory.get(node, default) for node in nodes]
    loads, counts, assignments = [0.0] * 4, [0] * 4, [-1] * len(nodes)
    for index in sorted(range(len(nodes)), key=lambda i: (-weights[i], i)):
        group = min(range(4), key=lambda i: (loads[i], i))
        assignments[index] = group
        loads[group] += weights[index]
        counts[group] += 1
    require(all(counts), "All four complete collection partitions must be nonempty")
    raw = b"".join((json.dumps({"nodeid": node}, separators=(",", ":")) + "\n").encode()
                   for node in nodes)
    return {"schema": 1, "kind": "source_bound_node_partition", "group_count": 4,
            "algorithm": "stable-longest-processing-time-v1",
            "weights_sha256": hashlib.sha256(canonical({"default_seconds": default,
                                         "advisory_seconds": advisory})).hexdigest(),
            "full_collection_sha256": hashlib.sha256(raw).hexdigest(),
            "assignments": assignments, "group_counts": counts, "estimated_group_seconds": loads}


def full_collection(path, selectors):
    require(path.stat().st_size <= MAX_PHASE_BYTES, "Full collection exceeds its byte bound")
    nodes = []
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(MAX_PHASE_LINE + 1), b""):
            require(len(line) <= MAX_PHASE_LINE and line.strip(), "Invalid full collection line")
            record = json_value(line)
            require(isinstance(record, dict) and set(record) == {"nodeid"}
                    and isinstance(record["nodeid"], str) and "::" in record["nodeid"],
                    "Invalid full collection identity")
            node = record["nodeid"]
            file = node.split("::", 1)[0]
            require(file.endswith(".py") and "\\" not in file
                    and not PurePosixPath(file).is_absolute()
                    and all(part not in ("", ".", "..") for part in file.split("/")),
                    "Invalid full collection source path")
            require(line == (json.dumps({"nodeid": node}, separators=(",", ":")) + "\n").encode(),
                    "Full collection is not canonical actual collection bytes")
            nodes.append(node)
    require(nodes and len(set(nodes)) == len(nodes), "Empty or duplicate full collection")
    file_ranks = {file: selector_index(file, selectors)
                  for file in {node.split("::", 1)[0] for node in nodes}}
    ranks = [file_ranks[node.split("::", 1)[0]] for node in nodes]
    require(ranks == sorted(ranks) and set(ranks) == set(range(len(selectors))),
            "Full collection lacks or reorders a mandatory selector")
    return nodes


def distributed_sdk(directory, report, sdk, execution, version, selectors, cases, nodes, root):
    protocol = report.get("phase_report_protocol", 1)
    require(type(protocol) is int and protocol in (1, 2), "Unknown component report protocol")
    reports = _report_metadata() if protocol == 2 else None
    shard = report["component_sdk_shard"]
    shard_count = report["sdk_shard_count"]
    relative = f"sdk-{version}-groups/report.json"
    require(sdk.get("sdk_groups") == relative, "Missing distributed SDK manifest binding")
    path = artifact_file(directory, "receipt/" + relative)
    require(sha256(path) == sdk.get("sdk_groups_sha256"), "Distributed SDK manifest hash differs")
    manifest = read_json(path)
    require(manifest.get("phase_report_protocol", 1) == protocol, "Child manifest protocol differs")
    child_environment = sdk_child_environment(manifest)
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    require(type(manifest.get("schema")) is int and manifest["schema"] == 2
            and manifest.get("kind") == "distributed_sdk_shard"
            and manifest.get("partial_scope") is True
            and type(manifest.get("shard_index")) is int and manifest["shard_index"] == shard
            and type(manifest.get("shard_count")) is int and manifest["shard_count"] == shard_count
            and type(manifest.get("group_count")) is int and manifest["group_count"] == 4
            and manifest.get("job_name") == execution["workflow_job_name"]
            and manifest.get("status") == "passed" and "error" not in manifest
            and manifest.get("source_commit") == report["source_commit"]
            and manifest.get("source_tree") == tree and manifest.get("execution") == execution
            and manifest.get("execution_sha256") == sha256(directory / "execution.json")
            and manifest.get("component_sdk_version") == version
            and manifest.get("checkout_root") == report["checkout_root"]
            and manifest.get("python") == report["python_interpreters"][version]
            and manifest.get("selected_tests") == selectors
            and manifest.get("environment") == child_environment
            and manifest.get("ordering") == "Raw child selected IDs preserve full collection order; partial merged cases follow complete source collection"
            and manifest.get("duration_precision") == "seconds rounded to six decimal places; one parent monotonic clock",
            "Distributed SDK source, host, shard, execution or scope differs")
    parent, clock = clock_value(sdk.get("sdk_clock")), clock_value(manifest.get("sdk_clock"))
    require(clock["clock_identity"] == parent["clock_identity"]
            and clock["started_utc"] == parent["started_utc"]
            and clock["monotonic_started"] == parent["monotonic_started"]
            and clock["finished_utc"] <= parent["finished_utc"] + 0.0000011
            and clock["monotonic_finished"] <= parent["monotonic_finished"]
            and type(manifest.get("timeout_seconds")) in (int, float)
            and manifest["timeout_seconds"] == 900
            and type(manifest.get("seconds")) in (int, float)
            and abs(manifest["seconds"] - clock["elapsed_seconds"]) <= 0.0000011
            and abs(sdk["seconds"] - parent["elapsed_seconds"]) <= 0.0000011
            and 0 < parent["elapsed_seconds"] <= 900,
            "SDK parent/import clock reset or local deadline mismatch")
    groups = manifest.get("groups")
    width = 4 // shard_count
    indices = list(range(shard * width, (shard + 1) * width))
    require(isinstance(groups, list) and len(groups) == width
            and all(isinstance(group, dict) and type(group.get("index")) is int for group in groups)
            and [group["index"] for group in groups] == indices,
            "Distributed SDK requires its declared distinct global child indexes")
    actual, checked, full, plan, raw_duration = [], [], None, None, 0.0
    phase_bounds = []
    bootstrap = source_child_bootstrap(root)
    for group, index in zip(groups, indices, strict=True):
        child = f"{report['receipt_directory']}/sdk-{version}-groups/group-{index}"
        command = [report["python_interpreters"][version], "-c", bootstrap,
                   child + "/runtime-environment.json", "-c",
                   f"{report['checkout_root']}/pyproject.toml", "-p", "scripts.pytest_gate_timings",
                   "--gate-timings", child + "/pytest-timings.jsonl",
                   "--gate-collection", child + "/pytest-collection.jsonl",
                   "--gate-full-collection", child + "/pytest-full-collection.jsonl",
                   "--gate-partition-plan", child + "/partition-plan.json",
                   "--gate-group-index", str(index), "--gate-group-count", "4",
                   "--junitxml", child + "/pytest.xml", "--basetemp", child + "/pytest-temp",
                   "-o", "cache_dir=" + child + "/pytest-cache", *selectors]
        if protocol == 2:
            at = command.index("--gate-full-collection")
            command[at:at] = reports.options(Path(child) / "pytest-timings.jsonl")
        require(group.get("selectors") == selectors and group.get("command") == command
                and group.get("status") == "passed" and type(group.get("exit_code")) is int
                and group["exit_code"] == 0 and type(group.get("pid")) is int and group["pid"] > 0
                and group.get("execution") == execution,
                "Distributed child command, environment, scope or outcome differs")
        runtime_environment = child_temporary_environment(path.parent, group, child,
                                                           child_environment)
        require(all(type(group.get(key)) in (int, float) and math.isfinite(group[key])
                    and group[key] >= 0 for key in ("start_seconds", "stop_seconds", "seconds",
                                                   "start_utc", "stop_utc"))
                and group["start_seconds"] <= group["stop_seconds"] <= clock["elapsed_seconds"]
                and abs(group["seconds"] - group["stop_seconds"] + group["start_seconds"]) <= 0.0000011
                and abs(group["start_utc"] - clock["started_utc"] - group["start_seconds"]) <= 1
                and abs(group["stop_utc"] - clock["started_utc"] - group["stop_seconds"]) <= 1,
                "Distributed child relative/UTC clock differs")
        files = {}
        for key, filename in (("log", "pytest.log"), ("junit", "pytest.xml"),
                              ("phase_timings", "pytest-timings.jsonl"),
                              ("collection", "pytest-collection.jsonl"),
                              ("full_collection", "pytest-full-collection.jsonl"),
                              ("partition_plan", "partition-plan.json")):
            require(group.get(key) == f"group-{index}/{filename}", "Unexpected distributed child path")
            files[key] = artifact_file(path.parent, group[key])
            require(files[key].stat().st_size > 0 and sha256(files[key]) == group.get(key + "_sha256"),
                    "Distributed child evidence is missing, empty or changed")
        collected = full_collection(files["full_collection"], selectors)
        expected_plan = independent_plan(root, collected)
        require(full is None or full == collected, "Shard children collected different full scope")
        full, plan = collected, expected_plan
        require(canonical(read_json(files["partition_plan"])) == canonical(plan)
                and group.get("partition_plan_digest") == hashlib.sha256(canonical(plan)).hexdigest()
                and type(group.get("full_collected_tests")) is int
                and group["full_collected_tests"] == len(full),
                "Partition plan differs from independently derived source assignments")
        selected = [node for node, assigned in zip(full, plan["assignments"], strict=True)
                    if assigned == index]
        if protocol == 2:
            for key, evidence in zip(("report_ledger", "report_terminal"), reports.paths(files["phase_timings"])[:2], strict=True):
                require(group.get(key) == f"group-{index}/{evidence.name}"
                        and group.get(key + "_sha256") == sha256(evidence),
                        "Native ledger/terminal path or hash differs")
        raw_cases = junit_cases(files["junit"], report_protocol=protocol)
        bounds = [math.inf, 0.0]
        raw_nodes = phase_cases(files["phase_timings"], raw_cases, selectors, ordered=True,
                                offsets=True, require_all_selectors=False, bounds=bounds, report_protocol=protocol, root=root)
        phase_bounds.append(bounds)
        require([node for node, _ in raw_nodes] == selected
                and group.get("tests") == {"tests": len(selected), "failures": 0, "errors": 0, "skipped": 0}
                and type(group.get("collected_tests")) is int and group["collected_tests"] == len(selected),
                "Raw child execution omits, duplicates or substitutes its complete partition")
        collection_cases(files["collection"], selected)
        require(group["start_utc"] - 1 <= bounds[0] <= bounds[1] <= group["stop_utc"] + 1,
                "Actual phase timestamps escape the authenticated child clock")
        raw_xml = ET.parse(files["junit"]).getroot()
        duration = sum(float(suite.get("time", "nan")) for suite in raw_xml.iter("testsuite")
                       if not suite.findall("testsuite"))
        require(math.isfinite(duration) and 0 <= duration <= group["seconds"] + 0.001001,
                "Raw JUnit duration exceeds the child wall time")
        raw_duration += duration
        for (node, byte_range), element in zip(raw_nodes, raw_xml.iter("testcase"), strict=True):
            actual.append((node, element, files["phase_timings"], byte_range))
        checked.append({key: group[key] for key in (
            "index", "pid", "command", "tests", "start_seconds", "stop_seconds", "seconds",
            "start_utc", "stop_utc", "collected_tests", "full_collected_tests", "partition_plan_digest",
            "log_sha256", "junit_sha256", "phase_timings_sha256", "collection_sha256",
            "full_collection_sha256", "partition_plan_sha256", "environment",
            "temporary_directory", "temporary_directory_fresh_before_launch",
            "temporary_directory_removed_after_stop",
            "runtime_environment", "runtime_environment_sha256")})
        if protocol == 2:
            report_counts = reports.counts([node for node, _ in raw_nodes], reports.expected_nested([node for node, _ in raw_nodes]))
            require(group.get("phase_report_protocol") == 2 and reports.same_counts(group.get("report_counts"), report_counts),
                    "Native child parent/subreport/JUnit counters differ")
            checked[-1].update(phase_report_protocol=2, report_counts=report_counts,
                              report_ledger_sha256=group["report_ledger_sha256"],
                              report_terminal_sha256=group["report_terminal_sha256"])
        checked[-1]["actual_runtime_environment"] = runtime_environment
    child_concurrency(manifest, phase_bounds)
    order = {node: index for index, node in enumerate(full)}
    actual.sort(key=lambda row: order[row[0]])
    require([row[0] for row in actual] == nodes and len(set(nodes)) == len(nodes),
            "Partial merged IDs differ from the actual two child partitions")
    expected_extra = {}
    if protocol == 2:
        merged_raw = artifact_file(directory, "receipt/" + sdk["phase_timings"])
        children_raw = [path.parent / f"group-{index}" / "pytest-timings.jsonl" for index in indices]
        info = reports.evidence_fields(merged_raw, nodes, root=root, derived=True)
        verify_derived_report_origins(merged_raw, nodes, children_raw, root)
        expected_extra = info
        require(all((reports.same_counts(sdk.get(k), v) and reports.same_counts(manifest.get(k), v))
                    if k == "report_counts" else sdk.get(k) == v and manifest.get(k) == v
                    for k, v in info.items()),
                "Complete combined report metadata/counters differ")
    require(manifest.get("combined") == {
        "junit": sdk["junit"], "junit_sha256": sdk["junit_sha256"],
        "phase_timings": sdk["phase_timings"], "phase_timings_sha256": sdk["phase_timings_sha256"],
        "tests": sdk["tests"], **expected_extra} and manifest.get("tests") == sdk["tests"]
        and manifest.get("junit_sha256") == sdk["junit_sha256"]
        and manifest.get("phase_timings_sha256") == sdk["phase_timings_sha256"],
        "Partial combined-output binding differs")
    combined_root = ET.parse(artifact_file(directory, "receipt/" + sdk["junit"])).getroot()
    suites = list(combined_root.iter("testsuite"))
    require(len(suites) == 1 and suites[0].get("name") == "pytest-sdk-shard"
            and float(suites[0].get("time", "nan")) == raw_duration
            and [element_value(element) for element in combined_root.iter("testcase")]
            == [element_value(row[1]) for row in actual],
            "Partial merged JUnit differs from actual raw child contents")
    with artifact_file(directory, "receipt/" + sdk["phase_timings"]).open("rb") as output:
        for _, _, raw_path, (offset, length) in actual:
            with raw_path.open("rb") as stream:
                stream.seek(offset)
                while length:
                    block = stream.read(min(length, 1024 * 1024))
                    require(block and output.read(len(block)) == block,
                            "Partial merged phase bytes differ from actual raw child records")
                    length -= len(block)
        require(not output.read(1), "Partial merged phase output has extra records")
    return {"manifest": relative, "manifest_sha256": sha256(path), "kind": manifest["kind"],
            "seconds": manifest["seconds"], "sdk_clock": parent, "groups": checked,
            "cpu_budget": manifest["cpu_budget"], "environment": child_environment,
            "full_collection_sha256": plan["full_collection_sha256"],
            "partition_plan_digest": hashlib.sha256(canonical(plan)).hexdigest()}, {
                "full": full, "rows": actual, "duration": raw_duration}


def package_sources(root, relative):
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", relative], cwd=root)
    paths = [Path(item.decode()) for item in tracked.split(b"\0") if item and item.endswith(b".py")]
    require(paths, "Checkout has no tracked package Python source")
    return {str(path.relative_to(relative)): (root / path).read_bytes() for path in paths}


def metadata_matches(raw, project):
    metadata = BytesParser().parsebytes(raw)
    require(
        len(metadata.get_all("Name", [])) == 1 and len(metadata.get_all("Version", [])) == 1,
        "Package metadata lacks unique name/version",
    )

    def normalize(value):
        return re.sub(r"[-_.]+", "-", value).lower()

    require(
        normalize(metadata["Name"]) == normalize(project["name"])
        and metadata["Version"] == project["version"],
        "Package metadata differs from current project",
    )


def check_package(path, project, package, sources, wheel):
    distribution = project["name"].replace("-", "_") + "-" + project["version"]
    if wheel:
        require(path.name == distribution + "-py3-none-any.whl", "Unexpected wheel filename")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "Duplicate wheel member")
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                require(
                    not member.is_absolute()
                    and ".." not in member.parts
                    and stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK,
                    "Unsafe wheel member",
                )
            metadata_matches(archive.read(distribution + ".dist-info/METADATA"), project)
            require(archive.read(distribution + ".dist-info/WHEEL"), "Empty wheel format metadata")
            python_members = {
                name for name in names if name.startswith(package + "/") and name.endswith(".py")
            }
            require(
                python_members == {package + "/" + name for name in sources},
                "Wheel Python source inventory differs from checkout",
            )
            for name, raw in sources.items():
                require(
                    archive.getinfo(package + "/" + name).file_size == len(raw),
                    "Wheel source size differs from tested checkout",
                )
                require(
                    archive.read(package + "/" + name) == raw,
                    "Wheel source differs from tested checkout",
                )
    else:
        require(path.name == distribution + ".tar.gz", "Unexpected sdist filename")
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            require(len(names) == len(set(names)), "Duplicate sdist member")
            require(
                all(
                    not PurePosixPath(member.name).is_absolute()
                    and ".." not in PurePosixPath(member.name).parts
                    and (member.isfile() or member.isdir())
                    for member in members
                ),
                "Unsafe sdist member",
            )
            metadata = archive.extractfile(distribution + "/PKG-INFO")
            require(metadata is not None, "Missing sdist metadata")
            metadata_matches(metadata.read(), project)
            prefix = distribution + "/src/" + package + "/"
            python_members = {
                member.name
                for member in members
                if member.isfile()
                and member.name.startswith(prefix)
                and member.name.endswith(".py")
            }
            require(
                python_members == {prefix + name for name in sources},
                "Sdist Python source inventory differs from checkout",
            )
            for name, raw in sources.items():
                require(
                    archive.getmember(distribution + "/src/" + package + "/" + name).size
                    == len(raw),
                    "Sdist source size differs from tested checkout",
                )
                member = archive.extractfile(distribution + "/src/" + package + "/" + name)
                require(
                    member is not None and member.read() == raw,
                    "Sdist source differs from tested checkout",
                )


def packages(directory, report, root):
    manifest = report.get("packages")
    require(
        isinstance(manifest, list) and len(manifest) == 4,
        "Four actual wheel/sdist outputs are required",
    )
    entries = {}
    for entry in manifest:
        require(
            isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"},
            "Invalid package manifest",
        )
        path = artifact_file(directory, "receipt/" + entry["path"])
        require(
            entry["path"].startswith("dist/") and len(PurePosixPath(entry["path"]).parts) == 2,
            "Package output must be inside receipt/dist",
        )
        require(
            path.name not in entries
            and type(entry["bytes"]) is int
            and entry["bytes"] > 0
            and path.stat().st_size == entry["bytes"]
            and sha256(path) == entry["sha256"],
            "Package hash, size or inventory mismatch",
        )
        entries[path.name] = path
    actual = []
    for path in (directory / "receipt/dist").iterdir():
        if path.name == ".gitignore":
            # uv 0.9.26 creates this exact one-byte output-directory marker.
            require(
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == 1
                and path.read_bytes() == b"*",
                "Unexpected UV distribution-directory marker",
            )
        else:
            actual.append(path)
    require({path.name for path in actual} == set(entries), "Unbound package output")
    checked = []
    for project_path, package, relative in (
        ("pyproject.toml", "openecon", "src/openecon"),
        (
            "packages/openecon-charts/pyproject.toml",
            "openecon_charts",
            "packages/openecon-charts/src/openecon_charts",
        ),
    ):
        with (root / project_path).open("rb") as stream:
            project = tomllib.load(stream)["project"]
        distribution = project["name"].replace("-", "_") + "-" + project["version"]
        sources = package_sources(root, relative)
        for wheel, suffix in ((True, "-py3-none-any.whl"), (False, ".tar.gz")):
            filename = distribution + suffix
            require(filename in entries, "A current SDK/charts distribution is missing")
            check_package(entries[filename], project, package, sources, wheel)
            checked.append(
                {
                    "path": "dist/" + filename,
                    "sha256": sha256(entries[filename]),
                    "bytes": entries[filename].stat().st_size,
                    "verified_python_source_files": len(sources),
                }
            )
    return checked


def required_report_protocol(root):
    # Source-pinned historical producers have no separate report protocol.
    # Their old exact-triple schema remains1. Current source declares2; this
    # cannot be downgraded by artifact fields, filename tricks or toy content.
    tree = ast.parse((Path(root) / "scripts/verify_merge_candidate.py").read_bytes())
    mentions = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                and node.id == "PHASE_REPORT_PROTOCOL"]
    if not mentions:
        return 1
    declarations = [node for node in tree.body if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "PHASE_REPORT_PROTOCOL"]
    stores = [node for node in mentions if isinstance(node.ctx, (ast.Store, ast.Del))]
    require(len(stores) == len(declarations) == 1
            and isinstance(declarations[0].value, ast.Constant)
            and type(declarations[0].value.value) is int and declarations[0].value.value == 2,
            "Invalid explicit source report protocol")
    return 2


def validate_component(directory, expected, selectors, root, *, distributed=False, shard_count=2):
    require(
        directory.is_dir() and not directory.is_symlink(),
        "Component artifact directory is missing or aliased",
    )
    execution_path = artifact_file(directory, "execution.json")
    report_path = artifact_file(directory, "receipt/report.json")
    execution, report = read_json(execution_path), read_json(report_path)
    protocol = report.get("phase_report_protocol", 1)
    require(type(protocol) is int and protocol in (1, 2), "Unknown report protocol")
    reports = _report_metadata() if protocol == 2 else None
    require(protocol == required_report_protocol(root),
            "Report protocol differs from the actual source-pinned producer version")
    version = execution.get("component_sdk_version")
    shard = execution.get("component_sdk_shard")
    if distributed or "host_cpu" in execution:
        cpu = execution.get("host_cpu")
        require(isinstance(cpu, dict) and set(cpu) == {"logical", "available"}
                and all(type(cpu[key]) is int for key in cpu)
                and 1 <= cpu["available"] <= cpu["logical"] <= 65536,
                "Invalid recorded host CPU capacity bounds")
    if distributed:
        require(type(shard_count) is int and shard_count in (2, 4)
                and type(shard) is int and 0 <= shard < shard_count
                and execution.get("workflow_job_name") == f"OpenEconometrics / SDK Python {version} shard {shard}",
                "Invalid distributed execution shard/job binding")
    require(
        version in VERSIONS and all(execution.get(key) == value for key, value in expected.items()),
        "Component execution is stale or belongs to another source/event/run",
    )
    require(
        isinstance(execution.get("browser_executable"), str)
        and PurePosixPath(execution["browser_executable"]).is_absolute(),
        "Missing provisioned browser binding",
    )
    require(
        type(report.get("schema")) is int
        and report["schema"] == (2 if distributed else 1)
        and report.get("source_commit") == expected["source_commit"]
        and report.get("component_mode") is True
        and report.get("component_sdk_version") == version
        and report.get("status_context") is None
        and report.get("status") == "passed"
        and "error" not in report,
        "A complete passed CI component receipt is required",
    )
    # This total CI wall budget is a resource policy, not a scientific-domain limit.
    if distributed:
        require(type(report.get("component_sdk_shard")) is int and report["component_sdk_shard"] == shard
                and type(report.get("sdk_shard_count")) is int and report["sdk_shard_count"] == shard_count
                and type(report.get("sdk_group_count")) is int and report["sdk_group_count"] == 4
                and report.get("partial_scope") is True
                and report.get("job_name") == execution["workflow_job_name"],
                "Partial distributed report shard contract differs")
    required_timeout = 900 if distributed else 1800
    require(
        type(report.get("timeout_seconds")) in (int, float) and report["timeout_seconds"] == required_timeout,
        f"The component must retain its {required_timeout}-second deadline",
    )
    require(
        report.get("python_versions_provisioned") == list(VERSIONS)
        and report.get("sdk_versions_executed") == [version],
        "Incorrect provisioned/executed Python scope",
    )
    require(
        report.get("selected_tests") == selectors
        and report.get("gate_environment") == (DISTRIBUTED_GATE_ENVIRONMENT if distributed else GATE_ENVIRONMENT),
        "Component selectors or test environment differ from the complete gate",
    )
    require(
        distributed or report.get("checkout_root") == str(root),
        "Component checkout root differs from aggregate checkout",
    )
    expected_commands = commands(report, version, selectors)
    steps = report.get("steps")
    names = [*COMMON_STEPS, f"sdk-{version}", "packages"]
    require(
        isinstance(steps, list)
        and all(isinstance(step, dict) for step in steps)
        and [step.get("name") for step in steps] == names,
        "A component step was omitted, repeated or reordered",
    )
    for step in steps:
        require(
            step.get("command") == expected_commands[step["name"]]
            and step.get("status") == "passed"
            and type(step.get("exit_code")) is int
            and step["exit_code"] == 0,
            "A component command or outcome differs from the complete gate",
        )
        require(
            type(step.get("seconds")) in (int, float)
            and math.isfinite(step["seconds"])
            and step["seconds"] >= 0,
            "Invalid component duration",
        )
        require(step.get("log") == step["name"] + ".log", "Unexpected component log path")
        log = artifact_file(directory, "receipt/" + step["log"])
        require(
            log.stat().st_size > 0 and sha256(log) == step.get("log_sha256"),
            "Component log is missing, empty or changed",
        )
    xml = artifact_file(directory, f"receipt/pytest-{version}.xml")
    cases = junit_cases(xml, report_protocol=protocol)
    sdk = steps[-2]
    require(
        sdk.get("tests") == {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0},
        "Reported JUnit counts differ from actual cases",
    )
    timings = artifact_file(directory, f"receipt/pytest-{version}-timings.jsonl")
    require(
        sdk.get("junit") == xml.name
        and sdk.get("junit_sha256") == sha256(xml)
        and sdk.get("phase_timings") == timings.name
        and sdk.get("phase_timings_sha256") == sha256(timings),
        "JUnit or phase evidence differs from its producer hash binding",
    )
    nodes = phase_cases(timings, cases, selectors, ordered=True, require_all_selectors=not distributed,
                        report_protocol=protocol, root=root, derived=protocol == 2)
    raw = None
    if distributed:
        sdk_groups, raw = distributed_sdk(directory, report, sdk, execution, version, selectors,
                                          cases, nodes, root)
    else:
        sdk_groups = grouped_sdk(directory, report, sdk, execution, version, selectors, cases, nodes)
    package_outputs = packages(directory, report, root)
    result = {
        "version": version,
        "execution_sha256": sha256(execution_path),
        "report_sha256": sha256(report_path),
        "JUnit": sdk["tests"],
        "JUnit_sha256": sha256(xml),
        "phase_records": 3 * len(nodes) + (reports.expected_nested(nodes) if protocol == 2 else 0),
        "phase_sha256": sha256(timings),
        "sdk_groups": sdk_groups,
        "step_count": len(steps),
        "steps": [
            {
                key: step[key]
                for key in ("name", "status", "exit_code", "seconds", "log", "log_sha256")
            }
            for step in steps
        ],
        "packages": package_outputs,
    }
    if protocol == 2:
        info = reports.evidence_fields(timings, nodes, root=root, derived=True)
        result.update(info)
    if distributed:
        result.update(shard=shard, job_name=execution["workflow_job_name"],
                      host_cpu=execution["host_cpu"],
                      checkout_root=report["checkout_root"], receipt_directory=report["receipt_directory"],
                      _raw=raw)
    return result, nodes


def github_time(value):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value),
            "Invalid authenticated GitHub UTC job timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def authenticated_jobs(path, expected, shard_count=2):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024 * 1024,
            "Missing or excessive authenticated GitHub jobs response")
    pages = json_value(path.read_bytes())
    require(isinstance(pages, list) and pages and all(isinstance(page, dict)
            and isinstance(page.get("jobs"), list) for page in pages),
            "GitHub jobs response must retain paginated authenticated API objects")
    jobs = [job for page in pages for job in page["jobs"]]
    require(all(isinstance(job, dict) for job in jobs)
            and all(type(job.get("id")) is int and job["id"] > 0 for job in jobs)
            and len({job["id"] for job in jobs}) == len(jobs),
            "Duplicate or invalid authenticated GitHub job ID")
    names = [f"OpenEconometrics / SDK Python {version} shard {shard}"
             for version in VERSIONS for shard in range(shard_count)] + [CONTEXT]
    selected = {}
    for name in names:
        matches = [job for job in jobs if job.get("name") == name]
        require(len(matches) == 1, "Missing or duplicate current-attempt SDK/aggregate job")
        job = matches[0]
        require(type(job.get("run_id")) is int and str(job["run_id"]) == expected["run_id"]
                and type(job.get("run_attempt")) is int and str(job["run_attempt"]) == expected["run_attempt"]
                and job.get("head_sha") == (expected["pull_request_head"] or expected["source_commit"]),
                "Authenticated GitHub job belongs to another source/run/attempt")
        require(type(job.get("runner_id")) is int and job["runner_id"] > 0
                and isinstance(job.get("runner_name"), str) and job["runner_name"]
                and isinstance(job.get("labels"), list)
                and set(job["labels"]) == {"ubuntu-latest"},
                "Job does not use its declared standard Ubuntu runner")
        require(isinstance(job.get("html_url"), str) and re.fullmatch(
            rf"https://github\.com/[^/]+/[^/]+/actions/runs/{expected['run_id']}/job/{job['id']}",
            job["html_url"]), "Authenticated job URL/run binding differs")
        start = github_time(job.get("started_at"))
        end = github_time(job["completed_at"]) if job.get("completed_at") is not None else None
        if name != CONTEXT:
            require(job.get("status") == "completed" and job.get("conclusion") == "success"
                    and end is not None and end >= start,
                    "A genuine SDK shard job did not complete successfully")
        else:
            require((job.get("status") == "in_progress" and job.get("conclusion") is None and end is None)
                    or (job.get("status") == "completed" and job.get("conclusion") == "success"
                        and end is not None and end >= start),
                    "Authenticated aggregate job is not active or successful")
        selected[name] = {"id": job["id"], "runner_id": job["runner_id"],
                          "runner_name": job["runner_name"], "started_utc": start,
                          "finished_utc": end, "html_url": job["html_url"]}
    return selected


def clock_inside_job(clock, job, *, replay=False):
    require(clock["started_utc"] >= job["started_utc"] - 1
            and (job["finished_utc"] is None or clock["finished_utc"] <= job["finished_utc"] + 1),
            "Source-generated clock escapes authenticated GitHub job bounds")
    require(not replay or job["finished_utc"] is not None,
            "Recorded aggregate-clock replay needs completed authenticated job bounds")


def complete_union_report_metadata(rows, raw_sha256, raw_name, root, output_directory):
    """Independently rebase ALL native events for the actual aggregate union."""
    reports = _report_metadata()
    children = sorted({row[2] for row in rows}, key=lambda p: int(p.parent.name.split("-")[-1]))
    index, terminals, framework = {}, [], None
    for child in children:
        selected = reports.read_collection(reports.collection_path(child))
        t = reports.terminal(child, selected, root=root)
        require(framework is None or framework == t["framework"], "Union framework identity differs")
        framework = t["framework"]
        terminals.append(reports.pin(reports.paths(child)[1], reports.MAX_META))
        ledger_sha, collection_sha = sha256(reports.paths(child)[0]), sha256(reports.collection_path(child))
        for record, event, raw, companion in reports.bound_reports(child):
            index.setdefault(record["nodeid"], []).append((event, raw, {
                "source_ledger_sha256": ledger_sha,
                "source_ledger_line_sha256": hashlib.sha256(companion).hexdigest(),
                "source_raw_report_ordinal": event["raw_report_ordinal"],
                "source_collection_sha256": collection_sha}))
    nodes = [row[0] for row in rows]
    require(set(index) == set(nodes) and len(set(nodes)) == len(nodes), "Aggregate parent origin union differs")
    ledger_digest, raw_digest = hashlib.sha256(), hashlib.sha256()
    ordinal = offset = ledger_bytes = 0
    target = None
    raw_path = Path(raw_name) if output_directory is None else output_directory / raw_name
    ledger_path, terminal_path, collection_path = reports.paths(raw_path)
    if output_directory is not None:
        require(not any(p.exists() or p.is_symlink() for p in (ledger_path, terminal_path, collection_path)),
                "Aggregate report metadata outputs must be fresh")
        target = ledger_path.open("xb")
    try:
        for node in nodes:
            for event, raw, origin in index[node]:
                updated = dict(event)
                updated["raw_report_ordinal"], updated["raw_line_offset"] = ordinal, offset
                updated["origin"] = origin
                encoded = reports.encode_event(updated, reports.strict_json(raw))
                require(len(encoded) <= reports.MAX_LINE and ledger_bytes + len(encoded) <= reports.MAX_LEDGER,
                        "Aggregate all-event ledger exceeds unchanged admission")
                raw_digest.update(raw)
                ledger_digest.update(encoded)
                if target is not None:
                    target.write(encoded)
                offset += len(raw)
                ledger_bytes += len(encoded)
                ordinal += 1
    finally:
        if target is not None:
            target.close()
    require(raw_digest.hexdigest() == raw_sha256 and offset + ledger_bytes <= MAX_PHASE_BYTES,
            "Aggregate complete event union differs or exceeds unchanged256MiB bound")
    collection_raw = b"".join(reports.encode({"nodeid": n}) for n in nodes)
    metadata = {"schema": "openecon.pytest.complete-report-terminal.v2", "phase_report_protocol": 2,
                "kind": "derived_union", "session_finished": True, "session_exitstatus": 0,
                "raw": {"bytes": offset, "sha256": raw_sha256},
                "ledger": {"bytes": ledger_bytes, "sha256": ledger_digest.hexdigest()},
                "collection": {"bytes": len(collection_raw), "sha256": hashlib.sha256(collection_raw).hexdigest()},
                "collection_name": collection_path.name,
                "report_counts": reports.counts(nodes, reports.expected_nested(nodes)),
                "source_identity": reports.source_identity(root), "framework": framework,
                "original_child_terminals": terminals}
    terminal_raw = reports.encode(metadata)
    require(len(terminal_raw) <= reports.MAX_META, "Aggregate report terminal excessive")
    if output_directory is not None:
        collection_path.write_bytes(collection_raw)
        terminal_path.write_bytes(terminal_raw)
        verify_derived_report_origins(raw_path, nodes, children, root)
    return {"phase_report_protocol": 2, "report_counts": metadata["report_counts"],
            "report_ledger": ledger_path.name, "report_ledger_sha256": ledger_digest.hexdigest(),
            "report_terminal": terminal_path.name, "report_terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
            "report_collection": collection_path.name,
            "report_collection_sha256": hashlib.sha256(collection_raw).hexdigest()}


def merge_distributed_rows(results, selectors, output_directory, shard_count=2, *, root=None):
    """Reassemble actual complete case elements and raw phase bytes before clock finish."""
    root = ROOT if root is None else root
    unions, reference = [], None
    for version in VERSIONS:
        pair = [results[f"{version}:{shard}"] for shard in range(shard_count)]
        protocol = pair[0].get("phase_report_protocol", 1)
        require(type(protocol) is int and protocol in (1, 2)
                and all(result.get("phase_report_protocol", 1) == protocol for result in pair),
                "Complete union mixes report protocols")
        full = pair[0]["_raw"]["full"]
        require(all(full == result["_raw"]["full"] for result in pair)
                and (reference is None or full == reference),
                "The four Python shard jobs did not collect the same complete ordered tests")
        reference = full
        require(all(pair[0]["sdk_groups"]["partition_plan_digest"]
                    == result["sdk_groups"]["partition_plan_digest"] for result in pair),
                "Python shards derived different source-bound partition plans")
        order = {node: index for index, node in enumerate(full)}
        rows = [row for result in pair for row in result["_raw"]["rows"]]
        rows.sort(key=lambda row: order[row[0]])
        require([row[0] for row in rows] == full and len(set(row[0] for row in rows)) == len(full),
                "Distributed complete union omits or repeats a collected testcase")
        # Every selector is checked against the actual executed union, separately
        # from the full collections, so a planned-but-unexecuted file cannot pass.
        require(set(selector_index(row[0].split("::", 1)[0], selectors) for row in rows)
                == set(range(len(selectors))), "Complete executed union lacks a mandatory selector")
        counts = {"tests": len(rows), "failures": 0, "errors": 0, "skipped": 0}
        xml_counts = dict(counts)
        if protocol == 2:
            xml_counts["tests"] += _report_metadata().expected_nested(full)
        suite = ET.Element("testsuite", name="pytest-sdk-distributed",
                           time=str(sum(result["_raw"]["duration"] for result in pair)),
                           **{key: str(value) for key, value in xml_counts.items()})
        for _, element, _, _ in rows:
            suite.append(element)
        tree = ET.Element("testsuites")
        tree.append(suite)
        xml = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
        phases = hashlib.sha256()
        target = None
        if output_directory is not None:
            output_directory.mkdir(parents=True, exist_ok=True)
            xml_path, phase_path = (output_directory / f"pytest-{version}.xml",
                                    output_directory / f"pytest-{version}-timings.jsonl")
            require(not any(path.exists() or path.is_symlink() for path in (xml_path, phase_path)),
                    "Complete union outputs must be fresh")
            xml_path.write_bytes(xml)
            target = phase_path.open("wb")
        try:
            for _, _, path, (offset, length) in rows:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while length:
                        block = stream.read(min(length, 1024 * 1024))
                        require(block, "Truncated raw phase evidence during complete reassembly")
                        phases.update(block)
                        if target is not None:
                            target.write(block)
                        length -= len(block)
        finally:
            if target is not None:
                target.close()
        unions.append({"version": version, "JUnit": counts,
                       "JUnit_sha256": hashlib.sha256(xml).hexdigest(),
                       "phase_records": 3 * len(rows) + (_report_metadata().expected_nested(full) if protocol == 2 else 0), "phase_sha256": phases.hexdigest(),
                       "full_collection_sha256": pair[0]["sdk_groups"]["full_collection_sha256"],
                       "partition_plan_digest": pair[0]["sdk_groups"]["partition_plan_digest"]})
        if protocol == 2:
            unions[-1].update(complete_union_report_metadata(rows, phases.hexdigest(),
                              f"pytest-{version}-timings.jsonl", root, output_directory))
    return unions


def validate_components(component_paths, expected, job_results, *, root=None, shard_count=1,
                        github_jobs=None, recorded_aggregate_clock=None, output_directory=None):
    started = time.monotonic(), time.time()
    root = ROOT if root is None else root
    source_binding = expected_binding(expected, root)
    require(type(shard_count) is int and shard_count in (1, 2, 4), "Invalid aggregate shard count")
    if shard_count in (2, 4):
        require(job_results == {f"{version}:{shard}": "success" for version in VERSIONS for shard in range(shard_count)},
                "All declared genuine SDK shard jobs must finish successfully")
        require(len(component_paths) == 2 * shard_count
                and len({path.resolve() for path in component_paths}) == 2 * shard_count,
                "Exactly all declared distinct partial shard artifacts are required")
        require(github_jobs is not None, "Distributed gate requires authenticated current-attempt job bounds")
        jobs = authenticated_jobs(github_jobs, expected, shard_count)
        selectors = selector_contract(root)
        components = {}
        for directory in component_paths:
            result, _ = validate_component(directory, expected, selectors, root,
                                           distributed=True, shard_count=shard_count)
            key = f"{result['version']}:{result['shard']}"
            require(key not in components, "Duplicate component Python/shard identity")
            job = jobs[result["job_name"]]
            clock_inside_job(result["sdk_groups"]["sdk_clock"], job)
            result["authenticated_job"] = job
            components[key] = result
        require(set(components) == set(job_results), "Missing SDK Python/shard component")
        unique_child_temporary_directories(components.values())
        unions = merge_distributed_rows(components, selectors, output_directory, shard_count, root=root)
        clock = aggregate_clock(started) if recorded_aggregate_clock is None else clock_value(recorded_aggregate_clock)
        clock_inside_job(clock, jobs[CONTEXT], replay=recorded_aggregate_clock is not None)
        cohorts = []
        for version in VERSIONS:
            pair = [components[f"{version}:{shard}"] for shard in range(shard_count)]
            earliest = min(result["sdk_groups"]["sdk_clock"]["started_utc"] for result in pair)
            latest = max(result["sdk_groups"]["sdk_clock"]["finished_utc"] for result in pair)
            require(clock["started_utc"] >= latest - 1 and clock["finished_utc"] >= latest,
                    "Aggregate clock precedes the completed SDK shard work")
            seconds = clock["finished_utc"] - earliest
            require(0 < seconds <= 900, "Complete SDK cohort including queue/collection/reassembly exceeded 900 seconds")
            cohorts.append({"version": version, "started_utc": earliest,
                            "finished_utc": clock["finished_utc"], "seconds": seconds})
        for result in components.values():
            del result["_raw"]
        return {"schema": 2, "kind": "current-run distributed hosted merge gate aggregate",
                "status": "passed", "status_context": CONTEXT, "execution": expected,
                "job_results": job_results, "verified_git": source_binding, "timeout_seconds": 900,
                "selected_tests": selectors, "shard_count": shard_count, "group_count": 4,
                "components": [components[f"{version}:{shard}"] for version in VERSIONS for shard in range(shard_count)],
                "complete_sdk_unions": unions, "sdk_cohorts": cohorts,
                "aggregate_clock": clock, "aggregate_clock_mode": "recorded-completed-job-replay" if recorded_aggregate_clock is not None else "live-source-validation",
                "authenticated_aggregate_job": jobs[CONTEXT], "github_jobs_sha256": sha256(github_jobs),
                "excluded_acceptance": ["full scientific suite", "physical scale", "Mac installed acceptance",
                                        "CUDA hardware", "live cloud/team authentication"]}
    require(github_jobs is None and recorded_aggregate_clock is None and output_directory is None,
            "Legacy aggregate cannot claim distributed hosted cohort evidence")
    require(
        job_results == {version: "success" for version in VERSIONS},
        "Both genuine component jobs must finish successfully",
    )
    require(
        len(component_paths) == 2 and len({path.resolve() for path in component_paths}) == 2,
        "Exactly two distinct component artifacts are required",
    )
    selectors = selector_contract(root)
    components = {}
    node_lists = []
    for directory in component_paths:
        result, nodes = validate_component(directory, expected, selectors, root)
        require(result["version"] not in components, "Duplicate component Python version")
        components[result["version"]] = result
        node_lists.append(nodes)
    require(set(components) == set(VERSIONS), "Both component Python versions are required")
    unique_child_temporary_directories(components.values())
    require(
        node_lists[0] == node_lists[1],
        "The two Python components did not complete the same ordered tests",
    )
    return {
        "schema": 1,
        "kind": "current-run parallel hosted merge gate aggregate",
        "status": "passed",
        "status_context": CONTEXT,
        "execution": expected,
        "job_results": job_results,
        "verified_git": source_binding,
        "timeout_seconds": 1800,
        "selected_tests": selectors,
        "components": [components[version] for version in VERSIONS],
        "excluded_acceptance": [
            "full scientific suite",
            "physical scale",
            "Mac installed acceptance",
            "CUDA hardware",
            "live cloud/team authentication",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, action="append", required=True)
    for name in ("source", "head", "base", "event", "run-id", "run-attempt"):
        parser.add_argument("--expected-" + name, required=True)
    parser.add_argument("--job-result", action="append", required=True, metavar="VERSION=RESULT")
    parser.add_argument("--shard-count", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--github-jobs", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    expected = {
        "source_commit": args.expected_source,
        "pull_request_head": args.expected_head or None,
        "pull_request_base": args.expected_base or None,
        "event": args.expected_event,
        "run_id": args.expected_run_id,
        "run_attempt": args.expected_run_attempt,
    }
    result = {"schema": 1, "status": "failed", "execution": expected}
    try:
        pairs = [value.split("=", 1) for value in args.job_result]
        require(
            len(pairs) == 2 * args.shard_count
            and all(len(pair) == 2 for pair in pairs)
            and len({pair[0] for pair in pairs}) == 2 * args.shard_count,
            "Exactly the declared distinct component job results are required",
        )
        result = validate_components(args.component, expected, dict(pairs), shard_count=args.shard_count,
                                     github_jobs=args.github_jobs,
                                     output_directory=args.report.parent / "complete-sdk-unions" if args.shard_count in (2, 4) else None)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        ET.ParseError,
        zipfile.BadZipFile,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "report": str(args.report), "error": result.get("error")}
        )
    )
    return 0 if result["status"] == "passed" else 1




def verify_derived_report_origins(raw, nodes, children, root):
    reports = _report_metadata()
    actual_terminal = reports.terminal(raw, nodes, root=root, derived=True)
    expected, terminal_pins = {}, []
    for child in children:
        selected = reports.read_collection(reports.collection_path(child))
        reports.terminal(child, selected, root=root)
        ledger = reports.paths(child)[0]
        ledger_sha = sha256(ledger)
        collection_sha = sha256(reports.collection_path(child))
        terminal_pins.append(reports.pin(reports.paths(child)[1], reports.MAX_META))
        for original, event, line, companion in reports.bound_reports(child):
            expected.setdefault(original["nodeid"], []).append((event, line, {
                "source_ledger_sha256": ledger_sha,
                "source_ledger_line_sha256": hashlib.sha256(companion).hexdigest(),
                "source_raw_report_ordinal": event["raw_report_ordinal"],
                "source_collection_sha256": collection_sha}))
    require(set(expected) == set(nodes) and actual_terminal["original_child_terminals"] == terminal_pins,
            "Derived union native parent/terminal origins differ")
    cursor = iter(reports.bound_reports(raw, derived=True))
    for node in nodes:
        for event, line, origin in expected[node]:
            item = next(cursor, None)
            require(item is not None, "Missing derived report origin")
            _, derived, actual_line, _ = item
            require(actual_line == line and derived["origin"] == origin
                    and {k: v for k, v in derived.items() if k not in
                         ("raw_report_ordinal", "raw_line_offset", "origin")}
                    == {k: v for k, v in event.items() if k not in
                        ("raw_report_ordinal", "raw_line_offset")},
                    "Complete derived raw/class/context/ordinal origin differs from actual child")
    require(next(cursor, None) is None, "Extra derived report origin")
    return actual_terminal["report_counts"]


if __name__ == "__main__":
    raise SystemExit(main())
