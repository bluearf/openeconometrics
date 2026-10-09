"""Opt-in phase timings that survive an unfinished merge-gate pytest process.

Load explicitly with ``-p scripts.pytest_gate_timings --gate-timings PATH``.
Only report metadata is recorded; captured output and failure details are omitted.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

MAX_COLLECTION_LINE = 8 * 1024 * 1024
MAX_COLLECTION_BYTES = 256 * 1024 * 1024


def pytest_addoption(parser):
    parser.addoption("--gate-timings", metavar="PATH", default=None,
                     help="Flush test-phase timing metadata to this JSONL file.")
    parser.addoption("--gate-collection", metavar="PATH", default=None,
                     help="Record ordered collected node IDs in this same pytest process.")
    parser.addoption("--gate-full-collection", metavar="PATH", default=None)
    parser.addoption("--gate-partition-plan", metavar="PATH", default=None)
    parser.addoption("--gate-group-index", type=int, default=None)
    parser.addoption("--gate-group-count", type=int, default=None)


def pytest_configure(config):
    path = config.getoption("--gate-timings")
    collection = config.getoption("--gate-collection")
    full = config.getoption("--gate-full-collection")
    plan = config.getoption("--gate-partition-plan")
    index = config.getoption("--gate-group-index")
    count = config.getoption("--gate-group-count")
    distributed = any(value is not None for value in (full, plan, index, count))
    if distributed and not (full and plan and path and collection and type(index) is int
                            and 0 <= index < 4 and count == 4):
        raise ValueError("Distributed gate requires complete collection/plan and exact group0..3/4")
    if distributed and any(getattr(config.option, key, None) for key in
                           ("keyword", "markexpr", "deselect", "ignore", "ignore_glob")):
        raise ValueError("Distributed gate forbids caller test filtering before full collection")
    if path is not None or collection is not None:
        writer = _TimingWriter(
            Path(path).open("w", encoding="utf-8", newline="\n") if path else None,
            Path(collection).open("w", encoding="utf-8", newline="\n") if collection else None,
            full=Path(full) if full else None, plan=Path(plan) if plan else None,
            group_index=index, group_count=count)
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
        encoded = (json.dumps(plan, sort_keys=True, separators=(",", ":"),
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
                line = json.dumps({"nodeid": item.nodeid}, allow_nan=False,
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
        self.output.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.output.flush()

    def pytest_unconfigure(self):
        for output in (self.output, self.collection):
            if output is not None:
                output.close()
