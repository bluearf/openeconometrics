"""The BVAR original-author runner pins lifecycle code, not refreshed timing hints."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "bvar_original_author_once", ROOT / "scripts/run_bvar_original_author_OS600_once.py")
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)
SOURCE = runner.LIFECYCLE.read_text()
OWNED = ("_launch_sdk", "_sdk_exit", "_live_sdk_group", "_stop", "_sdk_stopped")
REFUSED = "Reviewed nonreaping SDK lifecycle code differs"


def with_table(table, comment=""):
    node = next(statement for statement in ast.parse(SOURCE).body
                if isinstance(statement, ast.Assign)
                and getattr(statement.targets[0], "id", None) == runner.LIFECYCLE_TIMING_TABLE)
    lines = SOURCE.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + comment + table + "".join(lines[node.end_lineno:])


def lifecycle(tmp_path, source):
    path = tmp_path / "scripts/run_parallel_sdk_groups.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def test_current_lifecycle_matches_the_reviewed_code_pin():
    raw = runner.LIFECYCLE.read_bytes()
    assert runner.lifecycle_code_sha256(raw) == runner.LIFECYCLE_CODE_SHA256
    owned = runner.ownership()
    assert all(callable(getattr(owned, name)) for name in OWNED)
    assert owned.__file__ == str(runner.LIFECYCLE)


def test_timing_hint_refresh_keeps_the_pin_and_loads_the_new_table(tmp_path):
    refreshed = with_table(
        "ADVISORY_NODE_SECONDS = {\n    'tests/test_new.py::test_case[a]': 3.5,\n"
        "    'tests/test_other.py::test_case': 2,\n}\n",
        comment="# Refreshed from a later passed full-gate run.\n\n")
    refreshed = refreshed.replace("\n\ndef make_partition_plan(", "\n\n\ndef make_partition_plan(")
    assert hashlib.sha256(refreshed.encode()).hexdigest() != hashlib.sha256(SOURCE.encode()).hexdigest()
    owned = runner.ownership(lifecycle(tmp_path, refreshed))
    assert owned.ADVISORY_NODE_SECONDS == {"tests/test_new.py::test_case[a]": 3.5,
                                           "tests/test_other.py::test_case": 2}
    assert all(callable(getattr(owned, name)) for name in OWNED)


@pytest.mark.parametrize("old,new", [
    ("DEFAULT_NODE_SECONDS = 0.02\n", "DEFAULT_NODE_SECONDS = 0.03\n"),
    ('"OMP_NUM_THREADS": "1",', '"OMP_NUM_THREADS": "2",'),
    ("def make_partition_plan(nodes, group_count=4):",
     "def make_partition_plan(nodes, group_count=5):"),
])
def test_changed_lifecycle_code_is_refused(tmp_path, old, new):
    assert SOURCE.count(old) == 1
    with pytest.raises(ValueError, match=REFUSED):
        runner.ownership(lifecycle(tmp_path, SOURCE.replace(old, new)))


def test_added_lifecycle_statement_is_refused_before_execution(tmp_path):
    marker = tmp_path / "executed"
    path = lifecycle(tmp_path, SOURCE + f"\nPath({str(marker)!r}).write_text('ran')\n")
    with pytest.raises(ValueError, match=REFUSED):
        runner.ownership(path)
    assert not marker.exists()


@pytest.mark.parametrize("table,message", [
    ("ADVISORY_NODE_SECONDS = {'tests/a.py::t': __import__('pathlib').Path(MARKER).write_text('ran')}\n",
     "inert str-to-number literal"),
    ("ADVISORY_NODE_SECONDS = {'tests/a.py::t': float('1.5')}\n", "inert str-to-number literal"),
    ("ADVISORY_NODE_SECONDS = {**{'tests/a.py::t': 1.5}}\n", "inert str-to-number literal"),
    ("ADVISORY_NODE_SECONDS = {1: 1.5}\n", "inert str-to-number literal"),
    ("ADVISORY_NODE_SECONDS = {'tests/a.py::t': True}\n", "inert str-to-number literal"),
    ("ADVISORY_NODE_SECONDS = {'tests/a.py::t': -(1.5)}\n", "inert str-to-number literal"),
    ("ADVISORY_NODE_SECONDS = {key: 1.5 for key in ()}\n", "exactly one advisory timing table"),
    ("ADVISORY_NODE_SECONDS = dict(a=1.5)\n", "exactly one advisory timing table"),
    ("ADVISORY = ADVISORY_NODE_SECONDS = {'tests/a.py::t': 1.5}\n", "exactly one advisory timing table"),
    ("ADVISORY_NODE_SECONDS = {}\nADVISORY_NODE_SECONDS = {}\n", "exactly one advisory timing table"),
    ("TIMINGS = {'tests/a.py::t': 1.5}\n", "exactly one advisory timing table"),
])
def test_excluded_timing_table_must_stay_an_inert_literal(tmp_path, table, message):
    marker = tmp_path / "executed"
    source = with_table(table.replace("MARKER", repr(str(marker))))
    with pytest.raises(ValueError, match=message):
        runner.ownership(lifecycle(tmp_path, source))
    assert not marker.exists()
