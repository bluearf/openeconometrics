"""The fixed fork proof accepts one known warning and rejects extra output."""
import importlib.util
from pathlib import Path
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sandbox_fork_qa", Path(__file__).parents[1] / "scripts/verify_sandbox_broker_live.py")
proof = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = proof
_SPEC.loader.exec_module(proof)


def fixed_case(name="detached_fork"):
    return next(case for case in proof.CASES if case.name == name)


def warning(line=2):
    return (f"<openecon:12345678-1234-1234-1234-123456789abc>:{line}: DeprecationWarning: "
            "This process (pid=7) is multi-threaded, use of fork() may lead to deadlocks in the child.\n")


@pytest.mark.parametrize("name,marker,line", [
    ("detached_fork", "fixed-orphan-created", 2),
    ("beacon_child_positive_control", "fixed-beacon-child-created", 6),
    ("deleted_sandbox_orphan_beacon", "fixed-beacon-child-created", 6),
])
@pytest.mark.parametrize("with_warning", [False, True])
def test_exact_marker_or_known_warning_at_exact_fixed_source_line(name, marker, line, with_warning):
    case = fixed_case(name)
    stdout = (warning(line) if with_warning else "") + marker + "\n"
    assert proof.fork_stdout_classification(case, stdout) == (True, with_warning)
    checks = proof.fixed_output_checks(case, {"status": "ok", "stdout": stdout}, [])
    assert all(checks.values())
    # Warning presence is an observation, not a required passing check.
    assert "known_fork_warning_observed" not in checks


@pytest.mark.parametrize("stdout", [
    "", "fixed-orphan-created\nextra", "prefix\nfixed-orphan-created\n",
    "fixed-orphan-created\nfixed-orphan-created\n",
    warning(6) + "fixed-orphan-created\n",
    warning().replace("12345678-1234-1234-1234-123456789abc", "invalid") + "fixed-orphan-created\n",
    warning().replace("pid=7", "pid=-7") + "fixed-orphan-created\n",
    warning().replace("DeprecationWarning", "RuntimeWarning") + "fixed-orphan-created\n",
    warning().replace("may lead to deadlocks", "failed") + "fixed-orphan-created\n",
    warning() + warning() + "fixed-orphan-created\n",
    warning() + "fixed-orphan-created\nTraceback: failure",
])
def test_unrelated_or_malformed_output_never_satisfies_fork_proof(stdout):
    case = fixed_case()
    assert proof.fork_stdout_classification(case, stdout) == (False, False)
    assert not all(proof.fixed_output_checks(case, {"status": "ok", "stdout": stdout}, []).values())


def test_known_warning_does_not_override_execution_error():
    checks = proof.fixed_output_checks(
        fixed_case(), {"status": "error", "stdout": warning() + "fixed-orphan-created\n"}, [])
    assert checks["fork_code_completed"] is True
    assert checks["record_status_matches"] is False
