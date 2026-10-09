"""The opt-in collection receipt comes from the same actual pytest process."""

import json
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def command(test, directory, *, phases=True, collection=True):
    result = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(ROOT / "pyproject.toml"),
        "-p",
        "scripts.pytest_gate_timings",
        "--junitxml",
        str(directory / "actual.xml"),
        "--basetemp",
        str(directory / "temp"),
        "-o",
        "cache_dir=" + str(directory / "cache"),
    ]
    if phases:
        result += ["--gate-timings", str(directory / "phases.jsonl")]
    if collection:
        result += ["--gate-collection", str(directory / "collection.jsonl")]
    return [*result, str(test)]


def env():
    values = os.environ.copy()
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        values.pop(key, None)
    values["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return values


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped", "setup_error"])
def test_real_process_records_all_collected_nodes_for_each_outcome(tmp_path, outcome):
    test = tmp_path / "test_actual.py"
    bodies = {
        "passed": "def test_last(): pass\n",
        "failed": "def test_last(): assert False, 'PRIVATE_FAILURE_PAYLOAD'\n",
        "skipped": "import pytest\n@pytest.mark.skip(reason='PRIVATE_SKIP_PAYLOAD')\ndef test_last(): pass\n",
        "setup_error": "import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError('PRIVATE_SETUP_PAYLOAD')\ndef test_last(broken): pass\n",
    }
    test.write_text(
        "import pytest\n@pytest.mark.parametrize('value',[3,1,2])\ndef test_values(value): assert value\n"
        + bodies[outcome]
    )
    result = subprocess.run(
        command(test, tmp_path),
        cwd=ROOT,
        env=env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert result.returncode == (1 if outcome in ("failed", "setup_error") else 0)
    collected = [
        json.loads(line) for line in (tmp_path / "collection.jsonl").read_text().splitlines()
    ]
    assert all(set(item) == {"nodeid"} for item in collected)
    assert [item["nodeid"].split("::")[-1] for item in collected] == [
        "test_values[3]",
        "test_values[1]",
        "test_values[2]",
        "test_last",
    ]
    phases = [json.loads(line) for line in (tmp_path / "phases.jsonl").read_text().splitlines()]
    actual_order = list(dict.fromkeys(item["nodeid"] for item in phases))
    assert actual_order == [item["nodeid"] for item in collected]
    assert len(list(ET.parse(tmp_path / "actual.xml").getroot().iter("testcase"))) == 4
    assert "PRIVATE_" not in (tmp_path / "collection.jsonl").read_text()


def test_legacy_timing_only_behavior_is_unchanged(tmp_path):
    test = tmp_path / "test_legacy.py"
    test.write_text("def test_pass(): pass\n")
    result = subprocess.run(
        command(test, tmp_path, collection=False),
        cwd=ROOT,
        env=env(),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0 and not (tmp_path / "collection.jsonl").exists()
    phases = [json.loads(line) for line in (tmp_path / "phases.jsonl").read_text().splitlines()]
    assert [item["phase"] for item in phases] == ["setup", "call", "teardown"]
    assert all(item["outcome"] == "passed" for item in phases)


def test_collection_only_option_does_not_need_phase_writer(tmp_path):
    test = tmp_path / "test_collection_only.py"
    test.write_text("def test_pass(): pass\n")
    result = subprocess.run(
        command(test, tmp_path, phases=False), cwd=ROOT, env=env(), capture_output=True, timeout=30
    )
    assert result.returncode == 0 and not (tmp_path / "phases.jsonl").exists()
    assert len((tmp_path / "collection.jsonl").read_text().splitlines()) == 1


def test_collection_is_flushed_before_an_unfinished_test_process_is_terminated(tmp_path):
    test = tmp_path / "test_unfinished.py"
    test.write_text(
        "import time\ndef test_complete(): pass\ndef test_unfinished(): time.sleep(30)\ndef test_not_started(): pass\n"
    )
    process = subprocess.Popen(
        command(test, tmp_path),
        cwd=ROOT,
        env=env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        phase = tmp_path / "phases.jsonl"
        while time.monotonic() < deadline and process.poll() is None:
            if phase.exists():
                records = [json.loads(line) for line in phase.read_text().splitlines()]
                if any(
                    item["nodeid"].endswith("::test_unfinished") and item["phase"] == "setup"
                    for item in records
                ):
                    break
            time.sleep(0.05)
        assert process.poll() is None
        collected_bytes = (tmp_path / "collection.jsonl").read_bytes()
        nodes = [
            json.loads(line)["nodeid"].split("::")[-1] for line in collected_bytes.splitlines()
        ]
        assert nodes == ["test_complete", "test_unfinished", "test_not_started"]
        assert not any(item["nodeid"].endswith("::test_not_started") for item in records)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    assert process.returncode != 0
    assert (tmp_path / "collection.jsonl").read_bytes() == collected_bytes


@pytest.mark.parametrize("limit", ["line", "total"])
def test_collection_writer_fails_receipt_instead_of_dropping_overbudget_nodes(monkeypatch, limit):
    spec = importlib.util.spec_from_file_location(
        "bounded_timing", ROOT / "scripts/pytest_gate_timings.py"
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    assert plugin.MAX_COLLECTION_LINE == 8 * 1024 * 1024
    assert plugin.MAX_COLLECTION_BYTES == 256 * 1024 * 1024
    monkeypatch.setattr(
        plugin, "MAX_COLLECTION_LINE" if limit == "line" else "MAX_COLLECTION_BYTES", 64
    )
    receipt = io.StringIO()
    writer = plugin._TimingWriter(None, receipt)
    nodes = (
        ["tests/test_source.py::" + "x" * 100]
        if limit == "line"
        else ["tests/test_source.py::test_one", "tests/test_source.py::test_two"]
    )
    with pytest.raises(RuntimeError, match="Complete collection receipt exceeds"):
        writer.pytest_collection_finish(
            SimpleNamespace(items=[SimpleNamespace(nodeid=node) for node in nodes])
        )
    assert len(receipt.getvalue().encode()) <= 64


def test_actual_large_collected_identifier_is_preserved_within_existing_bound(tmp_path):
    test = tmp_path / "test_large_identifier.py"
    size = 3_700_000
    test.write_text(
        f"import pytest\n@pytest.mark.parametrize('value',[1],ids=['v'*{size}])\ndef test_large(value): assert value == 1\n"
    )
    result = subprocess.run(
        command(test, tmp_path), cwd=ROOT, env=env(), capture_output=True, timeout=30
    )
    assert result.returncode == 0
    collected = json.loads((tmp_path / "collection.jsonl").read_bytes())["nodeid"]
    phases = [
        json.loads(line)["nodeid"] for line in (tmp_path / "phases.jsonl").read_bytes().splitlines()
    ]
    assert phases == [collected] * 3
    assert collected.endswith("[" + "v" * size + "]")
