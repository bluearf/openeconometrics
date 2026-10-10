"""Opt-in phase timings that survive an unfinished merge-gate pytest process.

Load explicitly with ``-p scripts.pytest_gate_timings --gate-timings PATH``.
An explicitly opted-in component realizes its declared native-thread recipe
before collection. Generic timing-only callers do not import or configure Torch.
Only timing/collection metadata is recorded; captured output and failure details
are omitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

def _report_metadata():
    try:
        from scripts import pytest_gate_report_protocol as reports
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        import pytest_gate_report_protocol as reports
    return reports


MAX_COLLECTION_LINE = 8 * 1024 * 1024
MAX_COLLECTION_BYTES = 256 * 1024 * 1024
# Application tests may patch the shared json module through another import.
# Receipt writing must retain its own encoder until all test phases finish.
_METADATA_DUMPS = json.dumps


def _configure_native_threads():
    """Realize a complete declared gate recipe before collection or receipts."""
    declared = os.environ.get("OPENECON_GATE_TORCH_THREADS")
    if declared is None:
        return
    caps = [os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")]
    if declared not in ("1", "2") or any(value != declared for value in caps):
        raise ValueError("Gate native-thread caps must all declare the same canonical 1 or 2 recipe")
    # Environment declarations alone can produce a different initial Torch
    # count on another build. Use its public setter before test collection.
    import torch
    threads = int(declared)
    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError("Torch did not realize the exact declared gate native-thread recipe")


def pytest_addoption(parser):
    parser.addoption("--gate-timings", metavar="PATH", default=None,
                     help="Flush test-phase timing metadata to this JSONL file.")
    parser.addoption("--gate-collection", metavar="PATH", default=None,
                     help="Record ordered collected node IDs in this same pytest process.")
    parser.addoption("--gate-full-collection", metavar="PATH", default=None)
    parser.addoption("--gate-partition-plan", metavar="PATH", default=None)
    parser.addoption("--gate-group-index", type=int, default=None)
    parser.addoption("--gate-group-count", type=int, default=None)
    parser.addoption("--gate-report-protocol", type=int, choices=[1, 2], default=1)
    parser.addoption("--gate-report-ledger", metavar="PATH", default=None)
    parser.addoption("--gate-report-terminal", metavar="PATH", default=None)
    parser.addoption("--gate-report-capacity-source", metavar="PATH", default=None)
    parser.addoption("--gate-report-capacity-only", metavar="PATH", default=None)


def pytest_configure(config):
    path = config.getoption("--gate-timings")
    collection = config.getoption("--gate-collection")
    full = config.getoption("--gate-full-collection")
    plan = config.getoption("--gate-partition-plan")
    index = config.getoption("--gate-group-index")
    count = config.getoption("--gate-group-count")
    protocol = config.getoption("--gate-report-protocol")
    ledger = config.getoption("--gate-report-ledger")
    terminal = config.getoption("--gate-report-terminal")
    capacity_only = getattr(config.option, "gate_report_capacity_only", None)
    capacity_source = getattr(config.option, "gate_report_capacity_source", None)
    if capacity_only is not None:
        if not (getattr(config.option, "collectonly", False) and collection
                and protocol == 1 and path is None and ledger is None
                and terminal is None and capacity_source is None
                and all(value is None for value in (full, plan, index, count))):
            raise ValueError("Capacity-only admission requires exact collect-only metadata mode")
        _configure_native_threads()
        writer = _CapacityOnlyTimingWriter(
            Path(collection).open("x", encoding="utf-8", newline="\n"), Path(capacity_only))
        config.pluginmanager.register(writer, "gate-phase-timing-writer")
        return
    if capacity_source is not None and protocol != 2:
        raise ValueError("A capacity source requires actual protocol2 collection")
    if protocol == 2 and not all((path, collection, ledger, terminal)):
        raise ValueError("Protocol2 requires all actual report/collection/ledger/terminal paths")
    if protocol == 1 and (ledger is not None or terminal is not None):
        raise ValueError("Legacy reports cannot claim protocol2 metadata")
    distributed = any(value is not None for value in (full, plan, index, count))
    if distributed and not (full and plan and path and collection and type(index) is int
                            and 0 <= index < 4 and count == 4):
        raise ValueError("Distributed gate requires complete collection/plan and exact group0..3/4")
    if distributed and any(getattr(config.option, key, None) for key in
                           ("keyword", "markexpr", "deselect", "ignore", "ignore_glob")):
        raise ValueError("Distributed gate forbids caller test filtering before full collection")
    if path is not None or collection is not None:
        _configure_native_threads()
        writer_class = _Protocol2TimingWriter if protocol == 2 else _TimingWriter
        extra = {"ledger": ledger, "terminal": terminal, "capacity_source": capacity_source} if protocol == 2 else {}
        writer = writer_class(
            Path(path).open("w", encoding="utf-8", newline="\n") if path else None,
            Path(collection).open("w", encoding="utf-8", newline="\n") if collection else None,
            full=Path(full) if full else None, plan=Path(plan) if plan else None,
            group_index=index, group_count=count, **extra)
        config.pluginmanager.register(writer, "gate-phase-timing-writer")


class _TimingWriter:
    def __init__(self, output, collection=None, *, full=None, plan=None,
                 group_index=None, group_count=None):
        self.output = output
        self.collection = collection
        self.collection_bytes = 0
        self.full = full
        self.plan = plan
        self.group_index = group_index
        self.group_count = group_count

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        # Run after the source's other collection hooks; no -k/environment
        # filtering is accepted by the surrounding producer command contract.
        yield
        if self.group_index is None:
            return
        from scripts.run_parallel_sdk_groups import collection_bytes, make_partition_plan
        nodes = [item.nodeid for item in items]
        raw = collection_bytes(nodes)
        plan = make_partition_plan(nodes, self.group_count)
        encoded = (_METADATA_DUMPS(plan, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False) + "\n").encode()
        if len(encoded) > 2 * 1024 * 1024:
            raise RuntimeError("Partition plan exceeds its fixed metadata bound")
        self.full.write_bytes(raw)
        self.plan.write_bytes(encoded)
        selected, deselected = [], []
        for item, group in zip(items, plan["assignments"], strict=True):
            (selected if group == self.group_index else deselected).append(item)
        if not selected:
            raise RuntimeError("Distributed child received no collected tests")
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

    def pytest_collection_finish(self, session):
        if self.collection is not None:
            for item in session.items:
                line = _METADATA_DUMPS({"nodeid": item.nodeid}, allow_nan=False,
                                       separators=(",", ":")) + "\n"
                size = len(line.encode("utf-8"))
                if (size > MAX_COLLECTION_LINE
                        or self.collection_bytes + size > MAX_COLLECTION_BYTES):
                    raise RuntimeError("Complete collection receipt exceeds its fixed metadata bound")
                self.collection.write(line)
                self.collection_bytes += size
            self.collection.flush()

    def pytest_runtest_logreport(self, report):
        if self.output is None:
            return
        record = {"nodeid": report.nodeid, "phase": report.when,
                  "outcome": report.outcome, "duration": report.duration,
                  "start": report.start, "stop": report.stop}
        self.output.write(_METADATA_DUMPS(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.output.flush()

    def pytest_unconfigure(self):
        for output in (self.output, self.collection):
            if output is not None:
                output.close()


class _Protocol2TimingWriter(_TimingWriter):
    def __init__(self, output, collection, *, ledger, terminal, capacity_source=None, **kwargs):
        super().__init__(output, collection, **kwargs)
        self.capacity_source = Path(capacity_source) if capacity_source else None
        self.capacity_rows = None
        NativeLedger = _report_metadata().NativeLedger
        self.report_ledger = NativeLedger(output.name, ledger, terminal, collection.name,
                                         Path(__file__).resolve().parents[1])

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        inherited = super().pytest_collection_modifyitems(session, config, items)
        next(inherited)
        yield
        reports = _report_metadata()
        if self.capacity_source is not None:
            rows, _ = reports.read_capacity_collection(self.capacity_source, self.report_ledger.root)
            reports.match_capacity_collection(rows, items)
        else:
            rows = [{"nodeid": item.nodeid, "native_location": list(item.location)} for item in items]
            target = self.report_ledger.raw.with_name(self.report_ledger.raw.name + ".capacity-collection.jsonl")
            reports.write_capacity_collection(target, rows, self.report_ledger.root)
        self.capacity_rows = rows
        try:
            next(inherited)
        except StopIteration:
            pass

    def pytest_collection_finish(self, session):
        super().pytest_collection_finish(session)
        self.report_ledger.nodes = [item.nodeid for item in session.items]
        self.report_ledger.admitted_locations = _report_metadata().match_capacity_collection(
            self.capacity_rows, session.items)

    def pytest_runtest_logreport(self, report):
        record = {"nodeid": report.nodeid, "phase": report.when,
                  "outcome": report.outcome, "duration": report.duration,
                  "start": report.start, "stop": report.stop}
        # This is the same original six-field serialization, including failed
        # and zero-time subreports. The ledger never filters raw reports.
        raw = (_METADATA_DUMPS(record, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.output.buffer.write(raw)
        self.output.flush()
        self.report_ledger.record(report, raw)

    def pytest_sessionfinish(self, session, exitstatus):
        self.report_ledger.exitstatus = int(exitstatus)

    def pytest_unconfigure(self):
        super().pytest_unconfigure()
        self.report_ledger.finish()


class _CapacityOnlyTimingWriter(_TimingWriter):
    """One full source collection, no scientific runtest or report receipt."""
    def __init__(self, collection, capacity_path):
        super().__init__(None, collection)
        self.capacity_path = capacity_path

    def pytest_collection_finish(self, session):
        super().pytest_collection_finish(session)
        reports = _report_metadata()
        reports.write_capacity_collection(
            self.capacity_path,
            [{"nodeid": item.nodeid, "native_location": list(item.location)} for item in session.items],
            Path(__file__).resolve().parents[1])
