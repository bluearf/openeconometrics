"""Explicit protocol2 metadata for complete pytest parent/subreport evidence.

No application or numerical module is imported. Raw six-field report bytes keep
their original meaning; the companion ledger classifies every actual event.
"""
from __future__ import annotations

from collections import Counter
import ast
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

MAX_LINE = 8 * 1024**2
MAX_RAW = 256 * 1024**2
MAX_LEDGER = 32 * 1024**2
MAX_META = 2 * 1024**2
PROTOCOL = 2
OUTER = "_pytest.reports.TestReport"
NESTED = "_pytest.subtests.SubtestReport"
RAW_KEYS = {"nodeid", "phase", "outcome", "duration", "start", "stop"}
LEDGER_KEYS = {"schema", "raw_report_ordinal", "raw_line_offset", "raw_line_bytes",
               "raw_line_sha256", "parent_nodeid", "native_report_class",
               "report_kind", "native_location"}
ORIGIN_KEYS = {"source_ledger_sha256", "source_ledger_line_sha256",
               "source_raw_report_ordinal", "source_collection_sha256"}
_DUMPS = json.dumps
_LOADS = json.loads

# Filled by the source-only author from the immutable full six-parent contract.
CONTEXTS = {'tests/test_bayesian_var_sbc_protocol.py::ResumeTests::test_committed_target_keyboard_interrupt_and_system_exit_propagate_without_refit': {'mode': 'ordered',
                                                                                                                                            'contexts': [{'msg': None,
                                                                                                                                                          'kwargs_in_native_order': [['interruption',
                                                                                                                                                                                      "'KeyboardInterrupt'"]]},
                                                                                                                                                         {'msg': None,
                                                                                                                                                          'kwargs_in_native_order': [['interruption',
                                                                                                                                                                                      "'SystemExit'"]]}]},
 'tests/test_bayesian_var_sbc_protocol.py::ResumeTests::test_every_irreversible_uncommitted_phase_is_terminal_without_redraw': {'mode': 'multiset',
                                                                                                                                'contexts': [{'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['phase',
                                                                                                                                                                          "'random'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['phase',
                                                                                                                                                                          "'fit'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['phase',
                                                                                                                                                                          "'draw'"]]}]},
 'tests/test_bayesian_var_sbc_protocol.py::GuardTests::test_isolated_bytecode_and_all_preimport_thread_declarations_required': {'mode': 'ordered',
                                                                                                                                'contexts': [{'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['flag',
                                                                                                                                                                          "'isolated'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['flag',
                                                                                                                                                                          "'ignore_PYTHON_environment'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['flag',
                                                                                                                                                                          "'dont_write_bytecode'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['variable',
                                                                                                                                                                          "'OMP_NUM_THREADS'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['variable',
                                                                                                                                                                          "'MKL_NUM_THREADS'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['variable',
                                                                                                                                                                          "'OPENBLAS_NUM_THREADS'"]]},
                                                                                                                                             {'msg': None,
                                                                                                                                              'kwargs_in_native_order': [['variable',
                                                                                                                                                                          "'VECLIB_MAXIMUM_THREADS'"]]}]},
 'tests/test_bayesian_var_sbc_protocol.py::GuardTests::test_first_native_state_binds_defaults_before_effect_and_restart_drift_refused': {'mode': 'ordered',
                                                                                                                                         'contexts': [{'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'default_dtype'"]]},
                                                                                                                                                      {'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'default_dtype'"]]},
                                                                                                                                                      {'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'native_threads'"]]},
                                                                                                                                                      {'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'native_threads'"]]},
                                                                                                                                                      {'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'interop_threads'"]]},
                                                                                                                                                      {'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'default_device'"]]},
                                                                                                                                                      {'msg': None,
                                                                                                                                                       'kwargs_in_native_order': [['field',
                                                                                                                                                                                   "'torch_version'"]]}]},
 'tests/test_bayesian_var_sbc_protocol.py::AggregationTests::test_bool_bin_numeric_coverage_bad_control_and_unknown_status_refused': {'mode': 'ordered',
                                                                                                                                      'contexts': [{'msg': None,
                                                                                                                                                    'kwargs_in_native_order': [['key',
                                                                                                                                                                                "'rank_bins'"]]},
                                                                                                                                                   {'msg': None,
                                                                                                                                                    'kwargs_in_native_order': [['key',
                                                                                                                                                                                "'coverage'"]]},
                                                                                                                                                   {'msg': None,
                                                                                                                                                    'kwargs_in_native_order': [['key',
                                                                                                                                                                                "'point_mass_rank_bins'"]]}]},
 'tests/test_bayesian_var_sbc_protocol.py::AggregationTests::test_fixed_registration_family_and_pause_cannot_change': {'mode': 'ordered',
                                                                                                                       'contexts': [{'msg': None,
                                                                                                                                     'kwargs_in_native_order': [['key',
                                                                                                                                                                 "'total_cases'"]]},
                                                                                                                                    {'msg': None,
                                                                                                                                     'kwargs_in_native_order': [['key',
                                                                                                                                                                 "'posterior_draws'"]]},
                                                                                                                                    {'msg': None,
                                                                                                                                     'kwargs_in_native_order': [['key',
                                                                                                                                                                 "'pause_after_committed'"]]}]}}
SOURCE_PINS = {'tests/test_bayesian_var_sbc_protocol.py': {'bytes': 31299, 'sha256': '63d82b219559d87750f89f899f44429d70836885fa6ce7fe226879dd814e2796'}}
FRAMEWORK_PINS = {'_pytest.junitxml': {'bytes': 25961,
                      'sha256': '557b8978c66b8010fafc50ada4bf56333f9927f31b2b5909fe4c1a9df1a3c0f0'},
 '_pytest.reports': {'bytes': 23230,
                     'sha256': '2e9272a67cde38b925eba24a288b2ef77f741c0daa22da6fbce53ee6e58fa41d'},
 '_pytest.runner': {'bytes': 20346,
                    'sha256': 'c8d7fd553c0516f7884502ee0e7954eb5a5b1542f21fcb5c4357500ed2fe1570'},
 '_pytest.subtests': {'bytes': 13507,
                      'sha256': 'f514883bad06a84cec0d7ddc1dd8ce8c16e5680597f9fc5a9bd8836b878f92a4'},
 '_pytest.unittest': {'bytes': 25176,
                      'sha256': '24657b6c497eacbf50a978713598b3baed4c05c29f1c01891563cfefed119224'}}


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate protocol metadata key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite protocol metadata")
    return _LOADS(raw, object_pairs_hook=pairs, parse_constant=constant)


def encode(value):
    return (_DUMPS(value, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


# Physical encoding only. The six raw fields and expanded ledger.v2 objects
# keep their original meanings and are still checked by every state consumer.
WIRE_TAG = "openecon.pytest.report-wire"
WIRE_VERSION = 1


def _text_reference(value, node):
    if value == node:
        return [0]
    if value == node.split("::", 1)[0]:
        return [1]
    # A suffix reference also covers unittest's Class.method location spelling
    # without assuming that parameter values cannot themselves contain '::'.
    size = min(len(value), len(node))
    if node.endswith(value):
        common = len(value)
    else:
        common = 0
        while common < size and value[-common - 1] == node[-common - 1]:
            common += 1
    literal = [3, value]
    reference = [2, value[:len(value) - common], common]
    return reference if common and len(encode(reference)) < len(encode(literal)) else literal


def _read_text_reference(value, node):
    require(type(value) is list and value and type(value[0]) is int,
            "Invalid report-wire text reference")
    if value == [0]:
        text = node
    elif value == [1]:
        text = node.split("::", 1)[0]
    elif value[0] == 2:
        require(len(value) == 3 and type(value[1]) is str
                and type(value[2]) is int and 0 < value[2] <= len(node),
                "Invalid report-wire suffix reference")
        text = value[1] + node[-value[2]:]
    else:
        require(value[0] == 3 and len(value) == 2 and type(value[1]) is str,
                "Unknown report-wire text reference")
        text = value[1]
    require(len(text.encode("utf-8")) <= MAX_LINE
            and value == _text_reference(text, node),
            "Noncanonical or excessive report-wire text reference")
    return text


def encode_event(event, record):
    """Losslessly encode a logical event; legacy/negative controls stay readable.

    Invalid logical dictionaries deliberately keep their original expanded form.
    Re-sealed adversarial controls must reach the same semantic refusal checks,
    rather than being silently repaired by a compressor.
    """
    if type(event) is not dict or type(record) is not dict or type(record.get("nodeid")) is not str:
        return encode(event)
    nested = event.get("report_kind") == "unittest_subreport"
    extra = {"parent_nested_ordinal", "native_context"} if nested else set()
    if "origin" in event:
        extra |= {"origin"}
    location = event.get("native_location")
    origin = event.get("origin")
    if (set(event) != LEDGER_KEYS | extra
            or event.get("schema") != "openecon.pytest.complete-report-ledger.v2"
            or type(event.get("parent_nodeid")) is not str
            or type(location) is not list or len(location) != 3
            or type(location[0]) is not str or type(location[2]) is not str
            or ("origin" in event and (type(origin) is not dict or set(origin) != ORIGIN_KEYS))):
        return encode(event)
    node = record["nodeid"]
    wire = [WIRE_TAG, WIRE_VERSION, event["raw_report_ordinal"], event["raw_line_offset"],
            event["raw_line_bytes"], event["raw_line_sha256"],
            _text_reference(event["parent_nodeid"], node), event["native_report_class"],
            event["report_kind"], [_text_reference(location[0], node), location[1],
                                   _text_reference(location[2], node)],
            [event["parent_nested_ordinal"], event["native_context"]] if nested else None,
            [origin["source_ledger_sha256"], origin["source_ledger_line_sha256"],
             origin["source_raw_report_ordinal"], origin["source_collection_sha256"]]
            if origin is not None else None]
    return encode(wire)


def decode_event(companion, record):
    """Expand one bounded physical row against its immutable bound raw row.

    No cross-row dictionary, implicit template, forward reference, or whole-file
    logical allocation is used. Original expanded v2 evidence is cold readable.
    """
    require(len(companion) <= MAX_LINE, "Excessive report-wire row")
    value = strict_json(companion)
    if type(value) is dict:
        return value
    require(type(value) is list and len(value) == 12
            and value[0] == WIRE_TAG and type(value[1]) is int and value[1] == WIRE_VERSION
            and type(record) is dict and type(record.get("nodeid")) is str,
            "Unknown or incomplete report-wire version")
    node = record["nodeid"]
    location, nested, origin = value[9], value[10], value[11]
    require(type(location) is list and len(location) == 3,
            "Invalid report-wire native location")
    event = {"schema": "openecon.pytest.complete-report-ledger.v2",
             "raw_report_ordinal": value[2], "raw_line_offset": value[3],
             "raw_line_bytes": value[4], "raw_line_sha256": value[5],
             "parent_nodeid": _read_text_reference(value[6], node),
             "native_report_class": value[7], "report_kind": value[8],
             "native_location": [_read_text_reference(location[0], node), location[1],
                                 _read_text_reference(location[2], node)]}
    if nested is not None:
        require(type(nested) is list and len(nested) == 2,
                "Invalid report-wire complete native context")
        event.update(parent_nested_ordinal=nested[0], native_context=nested[1])
    if origin is not None:
        require(type(origin) is list and len(origin) == 4, "Invalid report-wire origin")
        event["origin"] = dict(zip(("source_ledger_sha256", "source_ledger_line_sha256",
                                    "source_raw_report_ordinal", "source_collection_sha256"),
                                   origin, strict=True))
    require(len(encode(event)) <= MAX_LINE and encode_event(event, record) == companion,
            "Noncanonical or excessive expanded report-wire row")
    return event


# This is a representation/work admission bound, never a statistical tolerance.
# Pytest's finite float timings fit 32 JSON bytes; current native production
# checks that bound before writing. The original six raw fields stay unchanged.
MAX_TIMING_TOKEN = 32


def collection_capacity(rows):
    """Bound every future native and derived row before any runtest setup.

    Actual collected native locations are supplied, not guessed from display
    node IDs. Maximal ordinals/offsets/lengths and complete three-origin digests
    make the bound independent of durations, partitioning, and test outcomes.
    """
    require(type(rows) is list and rows, "Empty complete report-capacity collection")
    nodes, collection_bytes = [], 0
    native = derived = raw_bytes = records = nested_count = 0
    native_line = derived_line = raw_line = logical_line = 0
    for row in rows:
        require(type(row) is dict and set(row) == {"nodeid", "native_location"},
                "Invalid complete report-capacity collection row")
        node, location = row["nodeid"], row["native_location"]
        file = safe_node(node)
        require(type(location) is list and len(location) == 3 and location[0] == file
                and type(location[1]) is int and location[1] >= 0
                and type(location[2]) is str and location[2],
                "Invalid collected native report location")
        row_size = len(encode(row))
        collection_bytes += row_size
        require(row_size <= MAX_LINE and collection_bytes <= MAX_RAW,
                "Complete native-location collection exceeds unchanged admission")
        nodes.append(node)
        contexts = CONTEXTS.get(node, {}).get("contexts", [])
        for phase, context, nested_ordinal in (
                [("setup", None, None), ("call", None, None), ("teardown", None, None)]
                + [("call", context, i) for i, context in enumerate(contexts)]):
            nested = context is not None
            raw = {"nodeid": node, "phase": phase, "outcome": "skipped",
                   "duration": 0, "start": 0, "stop": 0}
            # Each zero placeholder is one byte. All three actual finite
            # nonnegative numeric tokens are explicitly bounded at production.
            size = len(encode(raw)) + (0 if nested else 3 * (MAX_TIMING_TOKEN - 1))
            event = {"schema": "openecon.pytest.complete-report-ledger.v2",
                     "raw_report_ordinal": MAX_RAW, "raw_line_offset": MAX_RAW,
                     "raw_line_bytes": MAX_LINE, "raw_line_sha256": "f" * 64,
                     "parent_nodeid": node, "native_report_class": NESTED if nested else OUTER,
                     "report_kind": "unittest_subreport" if nested else "outer_phase",
                     "native_location": location}
            if nested:
                event.update(parent_nested_ordinal=nested_ordinal, native_context=context)
            native_size = len(encode_event(event, raw))
            event["origin"] = {"source_ledger_sha256": "f" * 64,
                               "source_ledger_line_sha256": "f" * 64,
                               "source_raw_report_ordinal": MAX_RAW,
                               "source_collection_sha256": "f" * 64}
            derived_size = len(encode_event(event, raw))
            raw_bytes += size
            native += native_size
            derived += derived_size
            records += 1
            nested_count += int(nested)
            raw_line = max(raw_line, size)
            native_line = max(native_line, native_size)
            derived_line = max(derived_line, derived_size)
            logical_line = max(logical_line, len(encode(event)))
            require(raw_line <= MAX_LINE and native_line <= MAX_LINE
                    and derived_line <= MAX_LINE and logical_line <= MAX_LINE
                    and native <= MAX_LEDGER and derived <= MAX_LEDGER
                    and raw_bytes + derived <= MAX_RAW,
                    "Full native/derived report union cannot fit unchanged admission")
    require(len(nodes) == len(set(nodes)), "Duplicate full report-capacity parent")
    return {"schema": "openecon.pytest.complete-report-capacity.v1",
            "wire_version": WIRE_VERSION, "report_counts": counts(nodes, nested_count),
            "limits": {"line_bytes": MAX_LINE, "ledger_bytes": MAX_LEDGER,
                       "combined_raw_and_ledger_bytes": MAX_RAW,
                       "timing_token_bytes": MAX_TIMING_TOKEN},
            "upper_bounds": {"raw_bytes": raw_bytes, "native_ledger_bytes": native,
                             "derived_ledger_bytes": derived,
                             "combined_raw_and_derived_bytes": raw_bytes + derived,
                             "raw_line_bytes": raw_line, "native_line_bytes": native_line,
                             "derived_line_bytes": derived_line,
                             "expanded_logical_line_bytes": logical_line,
                             "native_location_collection_bytes": collection_bytes}}


def write_capacity_collection(path, rows, root):
    path = Path(path)
    metadata = path.with_name(path.name + ".capacity.json")
    require(not any(p.exists() or p.is_symlink() for p in (path, metadata)),
            "Report-capacity evidence must be fresh")
    value = collection_capacity(rows)
    with path.open("xb") as output:
        for row in rows:
            output.write(encode(row))
    value.update(collection=pin(path), source_identity=source_identity(root))
    data = encode(value)
    require(len(data) <= MAX_META, "Report-capacity terminal exceeds unchanged admission")
    with metadata.open("xb") as output:
        output.write(data)
    return value


def read_capacity_collection(path, root):
    path = Path(path)
    metadata = path.with_name(path.name + ".capacity.json")
    pin(path)
    pin(metadata, MAX_META)
    rows = []
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(MAX_LINE + 1), b""):
            require(len(line) <= MAX_LINE and line.endswith(b"\n") and line.strip(),
                    "Truncated native-location capacity collection")
            row = strict_json(line)
            require(encode(row) == line, "Noncanonical native-location capacity collection")
            rows.append(row)
    expected = collection_capacity(rows)
    expected.update(collection=pin(path), source_identity=source_identity(root))
    actual = strict_json(metadata.read_bytes())
    require(actual == expected, "Complete report-capacity/source proof differs")
    return rows, actual


def match_capacity_collection(rows, items):
    """Match a selected actual collection against the full pre-admitted order."""
    index = {row["nodeid"]: row["native_location"] for row in rows}
    actual = [{"nodeid": item.nodeid, "native_location": list(item.location)} for item in items]
    require(actual and len(actual) == len({row["nodeid"] for row in actual})
            and all(index.get(row["nodeid"]) == row["native_location"] for row in actual),
            "Current actual item identity/location differs from pre-admitted full collection")
    selected = {row["nodeid"] for row in actual}
    require([row for row in rows if row["nodeid"] in selected] == actual,
            "Actual collected parent order differs from full capacity admission")
    return {row["nodeid"]: row["native_location"] for row in actual}


def verify_capacity_prepass(directory, manifest, root):
    """Recompute complete pre-runtest capacity and match all three child unions."""
    directory = Path(directory)
    item = manifest.get("capacity_admission")
    require(type(item) is dict and item.get("schema") == "openecon.sdk.complete-report-capacity-prepass.v1"
            and item.get("status") == "passed" and type(item.get("exit_code")) is int
            and item["exit_code"] == 0 and type(item.get("pid")) is int and item["pid"] > 0
            and item.get("scope") == "Complete collected node/native-location metadata; zero runtest phases"
            and item.get("temporary_directory_removed_after_stop") is True,
            "No successful actual full report-capacity prepass")
    for key, name in (("log", "pytest.log"), ("collection", "pytest-collection.jsonl"),
                      ("native_locations", "native-location-collection.jsonl"),
                      ("capacity", "native-location-collection.jsonl.capacity.json"),
                      ("runtime_environment", "runtime-environment.json")):
        require(item.get(key) == "capacity-admission/" + name,
                "Unexpected prepass evidence path")
        require(pin(directory / item[key])["sha256"] == item.get(key + "_sha256"),
                "Actual prepass evidence changed")
    origin = Path(manifest["checkout_root"])
    receipt = Path(manifest["groups"][0]["command"][3]).parent.parent
    child = receipt / "capacity-admission"
    expected = [manifest["python"], "-c", manifest["groups"][0]["command"][2],
                str(child / "runtime-environment.json"), "-c", str(origin / "pyproject.toml"),
                "-p", "scripts.pytest_gate_timings", "--gate-collection",
                str(child / "pytest-collection.jsonl"), "--gate-report-capacity-only",
                str(child / "native-location-collection.jsonl"), "--collect-only",
                "--basetemp", str(child / "pytest-temp"), "-o", "cache_dir=" + str(child / "pytest-cache"),
                *manifest["selected_tests"]]
    require(item.get("command") == expected, "Prepass command was not exact collect-only whole scope")
    runtime = strict_json((directory / item["runtime_environment"]).read_bytes())
    temporary = str(child / "runtime-temp")
    require(runtime == {"schema": 1, "pid": item["pid"],
                        "temporary_environment": {key: temporary for key in ("TMPDIR", "TMP", "TEMP")},
                        "tempfile_directory": temporary, "owned_directory": temporary,
                        "canonical_directory": temporary, "no_symlink_ancestors": True,
                        "empty_at_bootstrap": True}
            and not (directory / "capacity-admission/runtime-temp").exists()
            and not (directory / "capacity-admission/runtime-temp").is_symlink(),
            "Prepass did not preserve its exact owned tempfile bootstrap and cleanup")
    require(all(type(item.get(k)) in (int, float) and math.isfinite(item[k]) and item[k] >= 0
                for k in ("start_seconds", "stop_seconds", "finished_seconds"))
            and item["start_seconds"] <= item["stop_seconds"] <= item["finished_seconds"]
            and item["finished_seconds"] <= min(g["start_seconds"] for g in manifest["groups"])
            and item["finished_seconds"] <= manifest["seconds"] <= manifest["timeout_seconds"],
            "Scientific child started before complete capacity admission or deadline differs")
    rows, capacity = read_capacity_collection(directory / item["native_locations"], root)
    require([row["nodeid"] for row in rows] == read_collection(directory / item["collection"])
            and list(dict.fromkeys(row["nodeid"].split("::", 1)[0] for row in rows)) == manifest["selected_tests"]
            and item["collected_tests"] == len(rows)
            and item["report_counts"] == capacity["report_counts"]
            and item["upper_bounds"] == capacity["upper_bounds"]
            and item["source_identity"] == capacity["source_identity"],
            "Full prepass collection/count/resource/source proof differs")
    index = {row["nodeid"]: row["native_location"] for row in rows}
    selected = []
    for group in manifest["groups"]:
        raw = directory / ("group-" + str(group["index"])) / "pytest-timings.jsonl"
        nodes = read_collection(collection_path(raw))
        selected.extend(nodes)
        require(all(index.get(record["nodeid"]) == event["native_location"]
                    for record, event, _, _ in bound_reports(raw)),
                "Executed native locations differ from actual prepass")
    require(len(selected) == len(index) and len(set(selected)) == len(selected)
            and set(selected) == set(index), "Executed children do not cover full pre-admitted parent union")
    return capacity


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path, limit=MAX_RAW):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "Missing, symlinked or excessive protocol evidence")
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
            require(size <= limit, "Protocol evidence grew beyond its bound")
    return {"bytes": size, "sha256": h.hexdigest()}


def paths(raw):
    raw = Path(raw)
    return (raw.with_name(raw.name + ".report-ledger.jsonl"),
            raw.with_name(raw.name + ".report-terminal.json"),
            raw.with_name(raw.name + ".report-collection.jsonl"))


def options(raw):
    ledger, terminal, _ = paths(raw)
    return ["--gate-report-protocol", "2", "--gate-report-ledger", str(ledger),
            "--gate-report-terminal", str(terminal)]


def safe_node(node):
    require(type(node) is str and "::" in node, "Invalid report parent identity")
    file = node.split("::", 1)[0]
    require(file.endswith(".py") and not PurePosixPath(file).is_absolute()
            and ".." not in PurePosixPath(file).parts and "\\" not in file,
            "Unsafe report parent source")
    return file


def expected_nested(nodes):
    return sum(len(CONTEXTS[node]["contexts"]) for node in nodes if node in CONTEXTS)


def counts(nodes, nested):
    require(type(nested) is int and nested >= 0, "Invalid nested counter")
    return {"collected_cases": len(nodes), "outer_phase_records": 3 * len(nodes),
            "nested_reports": nested, "all_phase_records": 3 * len(nodes) + nested,
            "native_JUnit_reported_tests": len(nodes) + nested,
            "XML_testcase_elements": len(nodes)}


def same_counts(actual, expected):
    return (type(actual) is dict and set(actual) == set(expected)
            and all(type(value) is int and value >= 0 for value in actual.values())
            and actual == expected)


def check_contexts(node, actual):
    expected = CONTEXTS.get(node, {"mode": "ordered", "contexts": []})
    for context in actual:
        require(type(context) is dict and set(context) == {"msg", "kwargs_in_native_order"}
                and context["msg"] is None and type(context["kwargs_in_native_order"]) is list
                and len(context["kwargs_in_native_order"]) == 1,
                "Invalid complete native subtest context")
        pair = context["kwargs_in_native_order"][0]
        require(type(pair) is list and len(pair) == 2
                and all(type(value) is str and value and len(value) <= 1000 for value in pair),
                "Native saferepr context must remain complete strings")
    if expected["mode"] == "multiset":
        require(Counter(encode(c) for c in actual) == Counter(encode(c) for c in expected["contexts"]),
                "Complete source-declared frozenset context multiplicities differ")
    else:
        require(actual == expected["contexts"], "Source-declared context order or duplicates differ")


def source_identity(root):
    root = Path(root)
    for relative, expected in SOURCE_PINS.items():
        require(pin(root / relative, MAX_META) == expected, "Nested parent full source pin changed")
    own = Path(__file__).resolve()
    return {"contract_module": pin(own, MAX_META), "parent_sources": SOURCE_PINS,
            "plugin": pin(root / "scripts/pytest_gate_timings.py", MAX_META)}


def framework_identity():
    import pytest
    require(pytest.__version__ == "9.1.1", "Protocol2 requires the reviewed pytest9.1.1 framework")
    sources = {}
    for name, expected in FRAMEWORK_PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        raw = path.read_bytes()
        actual = {"bytes": len(raw), "sha256": sha_bytes(raw),
                  "whole_ast_sha256": sha_bytes(ast.dump(ast.parse(raw), include_attributes=True).encode())}
        require({key: actual[key] for key in expected} == expected,
                "Actual pytest framework source differs from reviewed protocol")
        sources[name] = {"path": str(path), **actual}
    return {"pytest_version": pytest.__version__, "python": sys.version, "sources": sources}


def check_framework(value):
    require(type(value) is dict and set(value) == {"pytest_version", "python", "sources"}
            and value["pytest_version"] == "9.1.1" and type(value["python"]) is str
            and value["python"] and type(value["sources"]) is dict
            and set(value["sources"]) == set(FRAMEWORK_PINS), "Incomplete framework provenance")
    for name, expected in FRAMEWORK_PINS.items():
        item = value["sources"][name]
        require(type(item) is dict and set(item) == {"path", "whole_ast_sha256", *expected}
                and type(item["path"]) is str and PurePosixPath(item["path"]).is_absolute()
                and re.fullmatch(r"[0-9a-f]{64}", item["whole_ast_sha256"] or "")
                and {key: item[key] for key in expected} == expected,
                "Unreviewed framework source/AST provenance")


def terminal(raw, nodes, *, root, derived=False):
    ledger, metadata, merged_collection = paths(raw)
    pin(metadata, MAX_META)
    value = strict_json(metadata.read_bytes())
    keys = {"schema", "phase_report_protocol", "kind", "session_finished", "session_exitstatus",
            "raw", "ledger", "collection", "collection_name", "report_counts", "source_identity", "framework"}
    if derived:
        keys |= {"original_child_terminals"}
    require(metadata.stat().st_size <= MAX_META and type(value) is dict and set(value) == keys
            and value["schema"] == "openecon.pytest.complete-report-terminal.v2"
            and type(value["phase_report_protocol"]) is int and value["phase_report_protocol"] == 2
            and value["kind"] == ("derived_union" if derived else "native_child")
            and value["session_finished"] is True and type(value["session_exitstatus"]) is int
            and value["session_exitstatus"] == 0, "No genuine successful protocol2 terminal")
    collection = collection_path(raw)
    require(not derived or collection == merged_collection, "Noncanonical derived collection")
    require(read_collection(collection) == nodes, "Complete terminal collection differs from executed parent order")
    require(value["raw"] == pin(raw) and value["ledger"] == pin(ledger, MAX_LEDGER)
            and value["collection"] == pin(collection)
            and value["source_identity"] == source_identity(root), "Complete protocol evidence/source pin differs")
    require(Path(raw).stat().st_size + ledger.stat().st_size <= MAX_RAW,
            "Combined report metadata exceeds unchanged256MiB bound")
    check_framework(value["framework"])
    require(same_counts(value["report_counts"], counts(nodes, expected_nested(nodes))), "Protocol report counters differ")
    if derived:
        require(type(value["original_child_terminals"]) is list and value["original_child_terminals"]
                and all(type(p) is dict and set(p) == {"bytes", "sha256"}
                        and type(p["bytes"]) is int and 0 < p["bytes"] <= MAX_META
                        and re.fullmatch(r"[0-9a-f]{64}", p["sha256"] or "")
                        for p in value["original_child_terminals"]), "Missing actual child terminal origins")
    return value


def bound_reports(raw, *, derived=False):
    """Validate byte/class/context metadata only; each consumer owns its state parser."""
    ledger, _, _ = paths(raw)
    require(pin(raw)["bytes"] + pin(ledger, MAX_LEDGER)["bytes"] <= MAX_RAW,
            "Excessive combined report evidence")
    ordinal = offset = 0
    with Path(raw).open("rb") as source, ledger.open("rb") as proof:
        while True:
            line, companion = source.readline(MAX_LINE + 1), proof.readline(MAX_LINE + 1)
            if not line or not companion:
                require(line == companion == b"", "Raw/ledger EOF differs")
                break
            require(len(line) <= MAX_LINE and len(companion) <= MAX_LINE
                    and line.endswith(b"\n") and companion.endswith(b"\n")
                    and line.strip() and companion.strip(), "Truncated or oversized report row")
            record = strict_json(line)
            event = decode_event(companion, record)
            require(type(record) is dict and set(record) == RAW_KEYS and record["outcome"] == "passed",
                    "Raw event is missing, failed, errored or skipped")
            require(all(type(record[k]) in (int, float) and math.isfinite(record[k]) and record[k] >= 0
                        for k in ("duration", "start", "stop")) and record["stop"] >= record["start"],
                    "Invalid report timing")
            file = safe_node(record["nodeid"])
            nested = type(event) is dict and event.get("report_kind") == "unittest_subreport"
            extra = {"parent_nested_ordinal", "native_context"} if nested else set()
            if derived:
                extra |= {"origin"}
            require(type(event) is dict and set(event) == LEDGER_KEYS | extra
                    and event["schema"] == "openecon.pytest.complete-report-ledger.v2"
                    and type(event["raw_report_ordinal"]) is int and event["raw_report_ordinal"] == ordinal
                    and type(event["raw_line_offset"]) is int and event["raw_line_offset"] == offset
                    and type(event["raw_line_bytes"]) is int and event["raw_line_bytes"] == len(line)
                    and event["raw_line_sha256"] == sha_bytes(line)
                    and event["parent_nodeid"] == record["nodeid"]
                    and event["native_report_class"] == (NESTED if nested else OUTER)
                    and event["report_kind"] == ("unittest_subreport" if nested else "outer_phase"),
                    "Incomplete all-event raw/ledger/class binding")
            location = event["native_location"]
            require(type(location) is list and len(location) == 3 and location[0] == file
                    and type(location[1]) is int and location[1] >= 0
                    and type(location[2]) is str and location[2], "Invalid full native report location")
            if nested:
                require(record["phase"] == "call" and record["duration"] == record["start"] == record["stop"] == 0
                        and type(event["parent_nested_ordinal"]) is int and event["parent_nested_ordinal"] >= 0,
                        "Native unittest subreport must retain exact zero CallInfo timing")
            if derived:
                origin = event["origin"]
                require(type(origin) is dict and set(origin) == ORIGIN_KEYS
                        and type(origin["source_raw_report_ordinal"]) is int and origin["source_raw_report_ordinal"] >= 0
                        and all(type(origin[k]) is str and re.fullmatch(r"[0-9a-f]{64}", origin[k])
                                for k in ORIGIN_KEYS - {"source_raw_report_ordinal"}), "Invalid derived ledger origin")
            yield record, event, line, companion
            ordinal += 1
            offset += len(line)


class NativeLedger:
    def __init__(self, raw, ledger, terminal_path, collection, root):
        self.raw, self.ledger_path, self.terminal_path = Path(raw), Path(ledger), Path(terminal_path)
        require((self.ledger_path, self.terminal_path) == paths(self.raw)[:2], "Noncanonical protocol paths")
        require(not self.ledger_path.exists() and not self.terminal_path.exists(), "Protocol evidence must be fresh")
        self.collection, self.root = Path(collection), Path(root)
        self.sources, self.framework = source_identity(root), framework_identity()
        self.stream = self.ledger_path.open("xb")
        self.ordinal = self.offset = self.ledger_bytes = self.nested = self.outer = 0
        self.parent_ordinals, self.nodes, self.exitstatus = {}, None, None
        self.admitted_locations = None

    def record(self, report, raw_line):
        cls = type(report).__module__ + "." + type(report).__qualname__
        require(cls in (OUTER, NESTED), "Unknown actual pytest report class")
        event = {"schema": "openecon.pytest.complete-report-ledger.v2",
                 "raw_report_ordinal": self.ordinal, "raw_line_offset": self.offset,
                 "raw_line_bytes": len(raw_line), "raw_line_sha256": sha_bytes(raw_line),
                 "parent_nodeid": report.nodeid, "native_report_class": cls,
                 "report_kind": "unittest_subreport" if cls == NESTED else "outer_phase",
                 "native_location": list(report.location)}
        if cls == NESTED:
            ordinal = self.parent_ordinals.get(report.nodeid, 0)
            event.update(parent_nested_ordinal=ordinal,
                         native_context={"msg": report.context.msg,
                                         "kwargs_in_native_order": [[k, v] for k, v in report.context.kwargs.items()]})
            self.parent_ordinals[report.nodeid] = ordinal + 1
            self.nested += 1
        else:
            self.outer += 1
        record = strict_json(raw_line)
        require(record["phase"] in ("setup", "call", "teardown")
                and record["outcome"] in ("passed", "failed", "skipped"),
                "Native phase/outcome exceeds pre-admitted representation")
        require(all(type(record[k]) in (int, float) and math.isfinite(record[k])
                    and record[k] >= 0 and len(encode(record[k]).strip()) <= MAX_TIMING_TOKEN
                    for k in ("duration", "start", "stop")),
                "Native timing token exceeds pre-admitted representation")
        if cls == NESTED:
            declared = CONTEXTS.get(report.nodeid, {}).get("contexts", [])
            require(event["parent_nested_ordinal"] < len(declared)
                    and event["native_context"] in declared
                    and record["phase"] == "call"
                    and record["duration"] == record["start"] == record["stop"] == 0,
                    "Native nested report exceeds complete source-declared admission")
        if self.admitted_locations is not None:
            require(self.admitted_locations.get(report.nodeid) == event["native_location"],
                    "Native report location differs from pre-runtest admission")
        require(len(encode(event)) <= MAX_LINE, "Expanded logical ledger row exceeds unchanged bound")
        companion = encode_event(event, record)
        require(len(raw_line) <= MAX_LINE and len(companion) <= MAX_LINE
                and self.ledger_bytes + len(companion) <= MAX_LEDGER
                and self.offset + len(raw_line) + self.ledger_bytes + len(companion) <= MAX_RAW,
                "Complete report metadata exceeds fixed admission")
        self.stream.write(companion)
        self.stream.flush()
        self.ordinal += 1
        self.offset += len(raw_line)
        self.ledger_bytes += len(companion)

    def finish(self):
        self.stream.close()
        require(self.nodes is not None and self.exitstatus is not None, "No genuine sessionfinish/collection marker")
        value = {"schema": "openecon.pytest.complete-report-terminal.v2", "phase_report_protocol": 2,
                 "kind": "native_child", "session_finished": True, "session_exitstatus": self.exitstatus,
                 "raw": pin(self.raw), "ledger": pin(self.ledger_path, MAX_LEDGER),
                 "collection": pin(self.collection), "collection_name": self.collection.name,
                 "report_counts": counts(self.nodes, self.nested),
                 "source_identity": self.sources, "framework": self.framework}
        require(self.outer == 3 * len(self.nodes) and self.ordinal == self.outer + self.nested,
                "Native terminal cannot conceal incomplete outer reports")
        raw = encode(value)
        require(len(raw) <= MAX_META and source_identity(self.root) == self.sources,
                "Terminal source changed or metadata excessive")
        with self.terminal_path.open("xb") as stream:
            stream.write(raw)


def native_identity(node):
    parts = node.split("::")
    require(len(parts) >= 2, "Incomplete native case identity")
    return (parts[0][:-3].replace("/", ".") +
            ("." + ".".join(parts[1:-1]) if len(parts) > 2 else ""), parts[-1])


def nested_identity(classname, name):
    matches = [node for node in CONTEXTS if native_identity(node) == (classname, name)]
    require(len(matches) <= 1, "Ambiguous source-bound nested identity")
    return len(CONTEXTS[matches[0]]["contexts"]) if matches else 0


def read_collection(path):
    pin(path)
    result = []
    with Path(path).open("rb") as stream:
        for line in iter(lambda: stream.readline(MAX_LINE + 1), b""):
            require(len(line) <= MAX_LINE and line.endswith(b"\n") and line.strip(),
                    "Incomplete collection row")
            value = strict_json(line)
            require(type(value) is dict and set(value) == {"nodeid"}, "Invalid collection row")
            safe_node(value["nodeid"])
            result.append(value["nodeid"])
    require(result and len(result) == len(set(result)), "Empty or duplicate collection")
    return result


def derived_evidence(raw, nodes, children, root):
    """Write explicit rebased origins; original raw report bytes remain untouched."""
    ledger, terminal_path, collection = paths(raw)
    require(not any(p.exists() or p.is_symlink() for p in (ledger, terminal_path, collection)),
            "Derived evidence must be fresh")
    index, terminals, framework = {}, [], None
    for child in children:
        child = Path(child)
        selected = read_collection(collection_path(child))
        t = terminal(child, selected, root=root)
        require(framework is None or framework == t["framework"], "Child framework identity differs")
        framework = t["framework"]
        terminals.append(pin(paths(child)[1], MAX_META))
        ledger_pin = pin(paths(child)[0], MAX_LEDGER)
        collection_pin = pin(collection_path(child))
        for record, event, line, companion in bound_reports(child):
            node = record["nodeid"]
            index.setdefault(node, []).append((event, line, companion, ledger_pin, collection_pin))
    require(set(index) == set(nodes) and len(nodes) == len(set(nodes)),
            "Derived parent union differs from native child sources")
    collection_raw = b"".join(encode({"nodeid": node}) for node in nodes)
    require(len(collection_raw) <= MAX_RAW, "Derived collection too large")
    collection.write_bytes(collection_raw)
    ordinal = offset = ledger_size = 0
    with ledger.open("xb") as output, Path(raw).open("rb") as actual:
        for node in nodes:
            for event, line, companion, child_ledger, child_collection in index[node]:
                require(actual.read(len(line)) == line, "Derived raw report union differs from original child")
                updated = {**event, "raw_report_ordinal": ordinal, "raw_line_offset": offset,
                           "origin": {"source_ledger_sha256": child_ledger["sha256"],
                                      "source_ledger_line_sha256": sha_bytes(companion),
                                      "source_raw_report_ordinal": event["raw_report_ordinal"],
                                      "source_collection_sha256": child_collection["sha256"]}}
                encoded = encode_event(updated, strict_json(line))
                require(len(encoded) <= MAX_LINE and ledger_size + len(encoded) <= MAX_LEDGER,
                        "Derived ledger exceeds unchanged bound")
                output.write(encoded)
                ordinal += 1
                offset += len(line)
                ledger_size += len(encoded)
        require(not actual.read(1), "Derived raw stream has extra reports")
    require(offset + ledger_size <= MAX_RAW, "Combined derived evidence exceeds unchanged bound")
    value = {"schema": "openecon.pytest.complete-report-terminal.v2", "phase_report_protocol": 2,
             "kind": "derived_union", "session_finished": True, "session_exitstatus": 0,
             "raw": pin(raw), "ledger": pin(ledger, MAX_LEDGER), "collection": pin(collection),
             "collection_name": collection.name,
             "report_counts": counts(nodes, expected_nested(nodes)), "source_identity": source_identity(root),
             "framework": framework, "original_child_terminals": terminals}
    encoded = encode(value)
    require(len(encoded) <= MAX_META, "Derived terminal too large")
    terminal_path.write_bytes(encoded)
    return value["report_counts"]


def evidence_fields(raw, nodes, *, root, derived=False):
    value = terminal(raw, nodes, root=root, derived=derived)
    ledger, final, collection = paths(raw)
    result = {"phase_report_protocol": 2, "report_counts": value["report_counts"],
              "report_ledger": ledger.name, "report_ledger_sha256": pin(ledger, MAX_LEDGER)["sha256"],
              "report_terminal": final.name, "report_terminal_sha256": pin(final, MAX_META)["sha256"]}
    if derived:
        result.update(report_collection=collection.name,
                      report_collection_sha256=pin(collection)["sha256"])
    return result


def collection_path(raw):
    metadata = paths(raw)[1]
    pin(metadata, MAX_META)
    value = strict_json(metadata.read_bytes())
    name = value.get("collection_name") if type(value) is dict else None
    require(type(name) is str and name and name not in (".", "..")
            and "/" not in name and "\\" not in name, "Unsafe terminal collection path")
    return Path(raw).parent / name
