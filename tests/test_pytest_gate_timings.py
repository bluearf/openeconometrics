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


@pytest.mark.parametrize("outcome", ["passed", "failed"])
def test_real_call_phase_json_patch_preserves_metadata_and_actual_outcome(tmp_path, outcome):
    test = tmp_path / "test_shared_json_patch.py"
    encoder_calls = tmp_path / "application-encoder-called.txt"
    test.write_text(
        "import json\nfrom pathlib import Path\n"
        "def test_patched_encoder(monkeypatch):\n"
        "    def forbidden(*args, **kwargs):\n"
        f"        Path({str(encoder_calls)!r}).write_text('called')\n"
        "        raise AssertionError('APPLICATION_ENCODER_CALLED_BY_HARNESS')\n"
        "    monkeypatch.setattr(json, 'dumps', forbidden)\n"
        f"    assert {outcome == 'passed'}, 'ACTUAL_FUNCTIONAL_FAILURE'\n"
    )
    result = subprocess.run(
        command(test, tmp_path), cwd=ROOT, env=env(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=30,
    )
    (tmp_path / "subprocess.log").write_bytes(result.stdout)
    assert result.returncode == (0 if outcome == "passed" else 1)
    assert not encoder_calls.exists()
    assert b"INTERNALERROR" not in result.stdout
    collected = [
        json.loads(line) for line in (tmp_path / "collection.jsonl").read_text().splitlines()
    ]
    assert len(collected) == 1 and set(collected[0]) == {"nodeid"}
    assert collected[0]["nodeid"].endswith("::test_patched_encoder")
    phases = [json.loads(line) for line in (tmp_path / "phases.jsonl").read_text().splitlines()]
    assert [item["phase"] for item in phases] == ["setup", "call", "teardown"]
    assert [item["outcome"] for item in phases] == ["passed", outcome, "passed"]
    assert all(item["nodeid"] == collected[0]["nodeid"] for item in phases)
    assert all(set(item) == {"nodeid", "phase", "outcome", "duration", "start", "stop"}
               and item["duration"] >= 0 and item["start"] <= item["stop"] for item in phases)
    case, = ET.parse(tmp_path / "actual.xml").getroot().iter("testcase")
    assert case.find("error") is None
    if outcome == "failed":
        assert "ACTUAL_FUNCTIONAL_FAILURE" in case.find("failure").get("message")
    else:
        assert case.find("failure") is None


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


def _complete_protocol2_fixture_module():
    fixture_spec = importlib.util.spec_from_file_location(
        "synthetic_complete_report_fixture", ROOT / "tests/gate_report_protocol_fixture.py")
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    return fixture


def test_protocol2_writer_retains_all391_events_and_147_native_JUnit_units(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    xml, raw, nodes = fixture.make(tmp_path / "complete")
    records, events = fixture.rows(raw)
    terminal = fixture.reports.terminal(raw, nodes, root=ROOT)
    assert len(nodes) == 122
    assert len(records) == len(events) == 391
    assert terminal["report_counts"] == {
        "collected_cases": 122, "outer_phase_records": 366, "nested_reports": 25,
        "all_phase_records": 391, "native_JUnit_reported_tests": 147, "XML_testcase_elements": 122}
    assert len(list(ET.parse(xml).getroot().iter("testcase"))) == 122
    assert ET.parse(xml).getroot().get("tests") == "147"
    nested = [e for e in events if e["report_kind"] == "unittest_subreport"]
    assert len(nested) == 25
    assert all(e["native_report_class"] == "_pytest.subtests.SubtestReport" for e in nested)
    assert all(r["start"] == r["stop"] == r["duration"] == 0
               for r, e in zip(records, events, strict=True) if e["report_kind"] == "unittest_subreport")


@pytest.mark.parametrize("function", ["loads", "dumps"])
def test_protocol2_json_instrumentation_isolated_from_application_patches(
    tmp_path, monkeypatch, function
):
    fixture = _complete_protocol2_fixture_module()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("APPLICATION_JSON_FUNCTION_CALLED_BY_PROTOCOL2")

    with monkeypatch.context() as application_patch:
        application_patch.setattr(json, function, forbidden)
        xml, raw, nodes = fixture.make(
            tmp_path / "patched-json", ["tests/test_protocol2_synthetic.py::test_parent_0"]
        )
        records, events = fixture.rows(raw)
        terminal = fixture.reports.terminal(raw, nodes, root=ROOT)
        with pytest.raises(ValueError, match="Duplicate protocol metadata key"):
            fixture.reports.strict_json('{"key":1,"key":2}')
        with pytest.raises(ValueError, match="Nonfinite protocol metadata"):
            fixture.reports.strict_json('{"key":NaN}')

    assert calls == []
    assert [row["phase"] for row in records] == ["setup", "call", "teardown"]
    assert [row["outcome"] for row in records] == ["passed"] * 3
    assert len(events) == terminal["report_counts"]["all_phase_records"] == 3
    assert terminal["report_counts"]["collected_cases"] == 1
    assert len(list(ET.parse(xml).getroot().iter("testcase"))) == 1


def test_protocol2_failed_session_never_emits_successful_terminal(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    _, raw, nodes = fixture.make(tmp_path / "failed")
    final = fixture.reports.paths(raw)[1]
    value = fixture.reports.strict_json(final.read_bytes())
    value["session_exitstatus"] = 1
    final.write_bytes(fixture.reports.encode(value))
    with pytest.raises(ValueError, match="successful.*terminal"):
        fixture.reports.terminal(raw, nodes, root=ROOT)


def test_protocol2_terminal_refuses_resealed_different_collection_order(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    _, raw, nodes = fixture.make(tmp_path / "reordered-collection")
    collection = fixture.reports.collection_path(raw)
    collection.write_bytes(b"".join(fixture.reports.encode({"nodeid": node}) for node in reversed(nodes)))
    final = fixture.reports.paths(raw)[1]
    value = fixture.reports.strict_json(final.read_bytes())
    value["collection"] = fixture.reports.pin(collection)
    final.write_bytes(fixture.reports.encode(value))
    with pytest.raises(ValueError, match="collection differs"):
        fixture.reports.terminal(raw, nodes, root=ROOT)


def test_protocol2_zero_nested_counter_cannot_be_boolean_false(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    _, raw, nodes = fixture.make(tmp_path / "boolean-zero", ["tests/test_protocol2_synthetic.py::test_parent_0"])
    final = fixture.reports.paths(raw)[1]
    value = fixture.reports.strict_json(final.read_bytes())
    value["report_counts"]["nested_reports"] = False
    final.write_bytes(fixture.reports.encode(value))
    with pytest.raises(ValueError, match="report counters"):
        fixture.reports.terminal(raw, nodes, root=ROOT)


@pytest.mark.parametrize("name", ["plain", "class", "unicode"], ids=["plain", "class", "unicode"])
@pytest.mark.parametrize("derived", [False, True], ids=["native", "derived"])
def test_reference_wire_preserves_every_logical_field_and_exact_raw_bytes(name, derived):
    from scripts import pytest_gate_report_protocol as reports
    nodes = {"plain": "tests/test_x.py::test_x[raw::parameter]",
             "class": "tests/test_x.py::ExactClass::test_x[raw::parameter]",
             "unicode": "tests/test_x.py::test_x[İı漢字😀::Ω]"}
    node = nodes[name]
    location = "ExactClass.test_x[raw::parameter]" if name == "class" else node.split("::", 1)[1]
    record = {"nodeid": node, "phase": "call", "outcome": "passed", "duration": .1, "start": 1., "stop": 1.1}
    raw = reports.encode(record)
    event = {"schema": "openecon.pytest.complete-report-ledger.v2", "raw_report_ordinal": 7,
             "raw_line_offset": 100, "raw_line_bytes": len(raw), "raw_line_sha256": reports.sha_bytes(raw),
             "parent_nodeid": node, "native_report_class": reports.OUTER, "report_kind": "outer_phase",
             "native_location": ["tests/test_x.py", 17, location]}
    if derived:
        event["origin"] = {"source_ledger_sha256": "1" * 64, "source_ledger_line_sha256": "2" * 64,
                           "source_raw_report_ordinal": 3, "source_collection_sha256": "3" * 64}
    wire = reports.encode_event(event, record)
    assert reports.strict_json(wire)[:2] == [reports.WIRE_TAG, 1]
    assert reports.decode_event(wire, record) == event
    assert reports.encode(record) == raw
    assert reports.decode_event(reports.encode(event), record) == event
    assert reports.encode_event(reports.decode_event(wire, record), record) == wire


@pytest.mark.parametrize("damage", ["version", "bool-version", "truncated", "unknown-tag", "bad-ref", "bool-ref",
                                    "negative-suffix", "oversized-suffix", "noncanonical-literal", "extra-origin", "bad-location"],
                         ids=["version", "bool-version", "truncated", "unknown-tag", "bad-ref", "bool-ref",
                              "negative-suffix", "oversized-suffix", "noncanonical-literal", "extra-origin", "bad-location"])
def test_reference_wire_refuses_malformed_or_noncanonical_physical_rows(damage):
    from scripts import pytest_gate_report_protocol as reports
    node = "tests/test_x.py::test_x"
    record = {"nodeid": node}
    event = {"schema": "openecon.pytest.complete-report-ledger.v2", "raw_report_ordinal": 0,
             "raw_line_offset": 0, "raw_line_bytes": 1, "raw_line_sha256": "f" * 64,
             "parent_nodeid": node, "native_report_class": reports.OUTER, "report_kind": "outer_phase",
             "native_location": ["tests/test_x.py", 1, "test_x"]}
    value = reports.strict_json(reports.encode_event(event, record))
    if damage == "version":
        value[1] = 2
    elif damage == "bool-version":
        value[1] = True
    elif damage == "truncated":
        value.pop()
    elif damage == "unknown-tag":
        value[0] = "unknown-wire"
    elif damage == "bad-ref":
        value[6] = [9]
    elif damage == "bool-ref":
        value[6] = [True]
    elif damage == "negative-suffix":
        value[6] = [2, "", -1]
    elif damage == "oversized-suffix":
        value[6] = [2, "", len(node) + 1]
    elif damage == "noncanonical-literal":
        value[6] = [3, node]
    elif damage == "extra-origin":
        value[11] = ["1" * 64] * 5
    else:
        value[9] = []
    with pytest.raises(ValueError):
        reports.decode_event(reports.encode(value), record)


def test_reference_wire_retains_complete_nested_context_and_rejects_duplicate_keys():
    from scripts import pytest_gate_report_protocol as reports
    node = next(iter(reports.CONTEXTS))
    record = {"nodeid": node}
    event = {"schema": "openecon.pytest.complete-report-ledger.v2", "raw_report_ordinal": 0,
             "raw_line_offset": 0, "raw_line_bytes": 1, "raw_line_sha256": "f" * 64,
             "parent_nodeid": node, "native_report_class": reports.NESTED, "report_kind": "unittest_subreport",
             "native_location": [node.split("::", 1)[0], 1, node.split("::", 1)[1]],
             "parent_nested_ordinal": 0, "native_context": reports.CONTEXTS[node]["contexts"][0]}
    wire = reports.encode_event(event, record)
    assert reports.decode_event(wire, record) == event
    corrupted = wire.replace(b'"msg":null', b'"msg":null,"msg":null')
    assert corrupted != wire
    with pytest.raises(ValueError, match="Duplicate"):
        reports.decode_event(corrupted, record)


def test_reference_wire_does_not_repair_an_explicit_null_origin(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    _, raw, _ = fixture.make(tmp_path / "invalid-null-origin")
    records, events = fixture.rows(raw)
    events[0]["origin"] = None
    encoded = fixture.reports.encode_event(events[0], records[0])
    assert fixture.reports.strict_json(encoded) == events[0]
    assert "origin" in fixture.reports.decode_event(encoded, records[0])
    fixture.reseal(raw, records, events)
    with pytest.raises(ValueError, match="raw/ledger/class binding"):
        list(fixture.reports.bound_reports(raw))


def test_reference_wire_full_raw_and_physical_origin_binding_remains_mandatory(tmp_path):
    fixture = _complete_protocol2_fixture_module()
    _, raw, _ = fixture.make(tmp_path / "full-wire")
    original = raw.read_bytes()
    assert len(list(fixture.reports.bound_reports(raw))) == 391
    for record, event, line, companion in fixture.reports.bound_reports(raw):
        assert fixture.reports.decode_event(companion, record) == event
        assert event["raw_line_sha256"] == fixture.reports.sha_bytes(line)
        assert fixture.reports.strict_json(companion)[:2] == [fixture.reports.WIRE_TAG, 1]
    assert raw.read_bytes() == original
    records, events = fixture.rows(raw)
    events[0]["parent_nodeid"] += "-forged"
    fixture.reseal(raw, records, events)
    with pytest.raises(ValueError, match="raw/ledger/class binding"):
        list(fixture.reports.bound_reports(raw))


def test_complete_capacity_counts_all122_parents25_nested_and_derived_origins_before_run(monkeypatch):
    fixture = _complete_protocol2_fixture_module()
    reports = fixture.reports
    rows = [{"nodeid": n, "native_location": [n.split("::", 1)[0], 1, n.split("::", 1)[1]]}
            for n in fixture.complete_nodes()]
    value = reports.collection_capacity(rows)
    assert value["report_counts"]["collected_cases"] == 122
    assert value["report_counts"]["all_phase_records"] == 391
    assert value["report_counts"]["nested_reports"] == 25
    assert value["upper_bounds"]["derived_ledger_bytes"] > value["upper_bounds"]["native_ledger_bytes"]
    assert value["limits"] == {"line_bytes": 8388608, "ledger_bytes": 33554432,
                               "combined_raw_and_ledger_bytes": 268435456, "timing_token_bytes": 32}
    with monkeypatch.context() as isolated_limit:
        isolated_limit.setattr(reports, "MAX_LEDGER", value["upper_bounds"]["native_ledger_bytes"])
        with pytest.raises(ValueError, match="Full native/derived report union"):
            reports.collection_capacity(rows)


def test_capacity_source_matches_full_native_locations_and_order_not_display_guesses(tmp_path):
    from scripts import pytest_gate_report_protocol as reports
    rows = [{"nodeid": "tests/test_x.py::ExactClass::test_x[a::b]",
             "native_location": ["tests/test_x.py", 12, "ExactClass.test_x[a::b]"]},
            {"nodeid": "tests/test_x.py::test_y", "native_location": ["tests/test_x.py", 14, "test_y"]}]
    path = tmp_path / "locations.jsonl"
    reports.write_capacity_collection(path, rows, ROOT)
    loaded, _ = reports.read_capacity_collection(path, ROOT)
    assert loaded == rows
    items = [SimpleNamespace(nodeid=r["nodeid"], location=tuple(r["native_location"])) for r in rows]
    assert reports.match_capacity_collection(rows, items) == {r["nodeid"]: r["native_location"] for r in rows}
    with pytest.raises(ValueError, match="order differs"):
        reports.match_capacity_collection(rows, list(reversed(items)))
    items[0].location = ("tests/test_x.py", 12, "ExactClass.test_x[a.b]")
    with pytest.raises(ValueError, match="identity/location differs"):
        reports.match_capacity_collection(rows, items)


def test_real_capacity_refusal_precedes_any_scientific_runtest_side_effect(tmp_path):
    marker = tmp_path / "runtest-marker"
    source = tmp_path / "test_capacity_before_run.py"
    source.write_text("from pathlib import Path\ndef test_never_started():\n"
                      f"    Path({str(marker)!r}).write_text('must not run')\n")
    collection = tmp_path / "capacity-collection.jsonl"
    capacity = tmp_path / "capacity-locations.jsonl"
    # Deliberately small metadata bound is isolated to this negative process.
    bootstrap = ("import pytest; from scripts import pytest_gate_report_protocol as r; "
                 "r.MAX_LEDGER=1; raise SystemExit(pytest.main(__import__('sys').argv[1:]))")
    result = subprocess.run([sys.executable, "-c", bootstrap, "-c", str(ROOT / "pyproject.toml"),
                             "-p", "scripts.pytest_gate_timings", "--collect-only", "--rootdir", str(tmp_path),
                             "--gate-collection", str(collection), "--gate-report-capacity-only", str(capacity),
                             str(source)], cwd=ROOT, env=env(), capture_output=True, timeout=30)
    assert result.returncode != 0
    assert b"Full native/derived report union cannot fit unchanged admission" in result.stdout + result.stderr
    assert not marker.exists()
    assert not capacity.exists()
