"""Synthetic report integrity controls, never scientific or acceptance receipts."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET

from _pytest.reports import TestReport
from _pytest.subtests import SubtestContext, SubtestReport
from scripts import pytest_gate_report_protocol as reports

ROOT = Path(__file__).resolve().parents[1]
SELECTORS = ["tests/test_bayesian_var_sbc_protocol.py", "tests/test_protocol2_synthetic.py"]


def complete_nodes():
    return [*reports.CONTEXTS, *["tests/test_protocol2_synthetic.py::test_parent_" + str(i)
                                for i in range(116)]]


def make(directory, nodes=None):
    directory = Path(directory)
    directory.mkdir(parents=True)
    nodes = complete_nodes() if nodes is None else nodes
    raw, collection = directory / "pytest-timings.jsonl", directory / "pytest-collection.jsonl"
    collection.write_bytes(b"".join(reports.encode({"nodeid": n}) for n in nodes))
    ledger, final, _ = reports.paths(raw)
    writer = reports.NativeLedger(raw, ledger, final, collection, ROOT)
    writer.nodes = list(nodes)
    suite = ET.Element("testsuite", name="synthetic-metadata-only", tests=str(len(nodes) + reports.expected_nested(nodes)),
                       failures="0", errors="0", skipped="0", time="1")
    stamp = 1000.0
    with raw.open("wb") as output:
        for node in nodes:
            parts = node.split("::")
            location = (parts[0], 10, ".".join(parts[1:]))
            items = [TestReport(node, location, {}, "passed", None, "setup", duration=.1, start=stamp, stop=stamp+.1)]
            for context in reports.CONTEXTS.get(node, {}).get("contexts", []):
                native = SubtestContext(msg=None, kwargs={k: ast.literal_eval(v) for k, v in context["kwargs_in_native_order"]})
                base = TestReport(node, location, {}, "passed", None, "call", duration=0, start=0, stop=0)
                items.append(SubtestReport._new(base, native, None, None))
            items.extend([TestReport(node, location, {}, "passed", None, "call", duration=.1, start=stamp+.1, stop=stamp+.2),
                          TestReport(node, location, {}, "passed", None, "teardown", duration=.1, start=stamp+.2, stop=stamp+.3)])
            for item in items:
                line = reports.encode({"nodeid": item.nodeid, "phase": item.when, "outcome": item.outcome,
                                       "duration": item.duration, "start": item.start, "stop": item.stop})
                output.write(line)
                output.flush()
                writer.record(item, line)
            classname, name = reports.native_identity(node)
            case = ET.SubElement(suite, "testcase", classname=classname, name=name, time=".3")
            properties = ET.SubElement(case, "properties")
            ET.SubElement(properties, "property", name="synthetic-control", value=node)
            stamp += 1
    writer.exitstatus = 0
    writer.finish()
    ET.ElementTree(suite).write(directory / "pytest.xml", encoding="utf-8", xml_declaration=True)
    return directory / "pytest.xml", raw, nodes


def rows(raw):
    ledger = reports.paths(raw)[0]
    records = [reports.strict_json(x) for x in raw.read_bytes().splitlines()]
    events = [reports.decode_event(line + b"\n", record)
              for line, record in zip(ledger.read_bytes().splitlines(), records, strict=True)]
    return records, events


def reseal(raw, records, events):
    """Re-sign declared bytes to test semantic refusal beyond hash-only checks."""
    ledger, final, _ = reports.paths(raw)
    offset = 0
    raw_lines, ledger_lines = [], []
    for i, (record, event) in enumerate(zip(records, events, strict=True)):
        line = reports.encode(record)
        event.update(raw_report_ordinal=i, raw_line_offset=offset, raw_line_bytes=len(line),
                     raw_line_sha256=reports.sha_bytes(line))
        raw_lines.append(line)
        ledger_lines.append(reports.encode_event(event, record))
        offset += len(line)
    raw.write_bytes(b"".join(raw_lines))
    ledger.write_bytes(b"".join(ledger_lines))
    terminal = reports.strict_json(final.read_bytes())
    terminal.update(raw=reports.pin(raw), ledger=reports.pin(ledger, reports.MAX_LEDGER))
    final.write_bytes(reports.encode(terminal))
