"""Resident continuous interval networks with explicit identity-preserving slices.

GEXF 1.2draft admits open endpoints; 1.3 uses inclusive endpoints. Missing
endpoints are unbounded. No interval window is silently a temporal path model.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import re

import pandas as pd

from openecon.frame import as_frame
from openecon.networks import Network, _Budget, _error, _integer, _label, _weight
from openecon._network_multi import multigraph, _identity_bytes
from openecon._network_io import _attribute_bytes, _attributes, _scalar
from openecon._network_temporal import _Work

_BOUNDS = {"start", "end", "startopen", "endopen"}
_ABSENT = object()


def _time(value, timeformat):
    if value is None:
        return None
    if isinstance(value, bool):
        _error("dynamic_time", "Boolean values are not times.")
    if timeformat == "integer":
        if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]{1,78}", value):
            value = int(value)
        if not isinstance(value, int) or value.bit_length() > 256:
            _error("dynamic_time", "Integer times require bounded exact integers.")
        return value
    if timeformat == "double":
        if not isinstance(value, (str, int, float)):
            _error("dynamic_time", "Double times require finite real numbers.")
        if isinstance(value, str) and len(value) > 128:
            _error("dynamic_time", "Numeric time tokens are bounded to 128 characters.")
        try:
            converted = float(value)
        except (ValueError, OverflowError):
            _error("dynamic_time", "Double times require finite real numbers.")
        if not math.isfinite(converted) or isinstance(value, int) and converted != value:
            _error("dynamic_time", "Double times must be finite; integer inputs must survive binary64 conversion exactly.")
        return converted
    if timeformat == "date":
        if isinstance(value, date) and not isinstance(value, datetime):
            value = value.isoformat()
        try:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                raise ValueError()
            return date.fromisoformat(value).isoformat()
        except ValueError:
            _error("dynamic_time", "Date times require valid ISO YYYY-MM-DD values.")
    if isinstance(value, datetime):
        value = value.isoformat()
    try:
        if not isinstance(value, str) or len(value) > 64 or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})?", value):
            raise ValueError()
        if re.search(r"\.[0-9]{7,}", value):
            _error("dynamic_time", "dateTime precision is limited to microseconds; no truncation is performed.")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        return parsed.isoformat()
    except (ValueError, OverflowError):
        _error("dynamic_time", "dateTime values require ISO timestamps; timezone-free values explicitly mean UTC.")


def _coordinate(value, timeformat):
    if timeformat == "date":
        return date.fromisoformat(value).toordinal()
    if timeformat == "dateTime":
        delta = datetime.fromisoformat(value) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    return value


@dataclass(frozen=True)
class _Interval:
    start: object
    end: object
    left_open: bool
    right_open: bool
    low: object
    high: object

    def record(self):
        result = {}
        if self.start is not None:
            result["startopen" if self.left_open else "start"] = self.start
        if self.end is not None:
            result["endopen" if self.right_open else "end"] = self.end
        return result


def _interval(raw, timeformat, version, *, query=False):
    if not isinstance(raw, Mapping) or set(raw) - _BOUNDS:
        _error("dynamic_time", "An interval contains only start/end or startopen/endopen bounds.")
    if "start" in raw and "startopen" in raw or "end" in raw and "endopen" in raw:
        _error("dynamic_time", "A boundary cannot be both open and closed.")
    left_open, right_open = "startopen" in raw, "endopen" in raw
    if version == "1.3" and not query and (left_open or right_open):
        _error("file_feature", "GEXF 1.3 supports inclusive intervals only; choose version='1.2draft' to retain open bounds.")
    start, end = _time(raw.get("startopen" if left_open else "start"), timeformat), _time(raw.get("endopen" if right_open else "end"), timeformat)
    if left_open and start is None or right_open and end is None:
        _error("dynamic_time", "An open boundary needs an explicit time.")
    low = -math.inf if start is None else _coordinate(start, timeformat)
    high = math.inf if end is None else _coordinate(end, timeformat)
    if low > high or low == high and (left_open or right_open):
        _error("dynamic_time", "An interval must contain at least one time; equal inclusive bounds represent a point.")
    return _Interval(start, end, left_open, right_open, low, high)


def _meet(left, right):
    if left.low > right.low:
        start, low, lo = left.start, left.low, left.left_open
    elif left.low < right.low:
        start, low, lo = right.start, right.low, right.left_open
    else:
        start, low, lo = left.start, left.low, left.left_open or right.left_open
    if left.high < right.high:
        end, high, ro = left.end, left.high, left.right_open
    elif left.high > right.high:
        end, high, ro = right.end, right.high, right.right_open
    else:
        end, high, ro = left.end, left.high, left.right_open or right.right_open
    return None if low > high or low == high and (lo or ro) else _Interval(start, end, lo, ro, low, high)


def _union(intervals, work):
    work.add(8 * len(intervals) * max(1, len(intervals).bit_length()))
    result = []
    for current in sorted(intervals, key=lambda item: (item.low, item.left_open, item.high, item.right_open)):
        if result and (current.low < result[-1].high or current.low == result[-1].high
                       and not (current.left_open and result[-1].right_open)):
            previous = result[-1]
            if current.high > previous.high:
                end, high, ro = current.end, current.high, current.right_open
            elif current.high < previous.high:
                end, high, ro = previous.end, previous.high, previous.right_open
            else:
                end, high, ro = previous.end, previous.high, previous.right_open and current.right_open
            result[-1] = _Interval(previous.start, end, previous.left_open, ro, previous.low, high)
        else:
            result.append(current)
    return tuple(result)


def _intersection(left, right, work):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        work.add(4)
        overlap = _meet(left[i], right[j])
        if overlap is not None:
            result.append(overlap)
        if left[i].high < right[j].high:
            i += 1
        elif left[i].high > right[j].high:
            j += 1
        else:
            # Equal endpoints can still meet the next interval at a point.
            if left[i].right_open == right[j].right_open:
                i, j = i + 1, j + 1
            elif left[i].right_open:
                i += 1
            else:
                j += 1
    return _union(result, work)


def _same_interval(left, right):
    return (left.low, left.high, left.left_open, left.right_open) == (right.low, right.high, right.left_open, right.right_open)


def _covers(intervals, periods, work):
    intersected = _intersection(intervals, periods, work)
    return len(intersected) == len(periods) and all(_same_interval(a, b) for a, b in zip(intersected, periods))


def dynamic_network(nodes, edges, *, directed=False, timeformat="double", version="1.3",
                    interval=None, node_defaults=None, edge_defaults=None,
                    max_memory_mb=256, max_events=1_000_000, max_work=50_000_000):
    """Capture typed node/edge records, spells and timed scalar attributes.

    Nodes use node/spells/attributes/dynamic_attributes; edges additionally use
    edge_id/source/target/weight. Dynamic values contain value and interval bounds.
    Missing bounds mean unbounded time; topology is resident, never coalesced.
    """
    return DynamicNetwork(nodes, edges, directed=directed, timeformat=timeformat, version=version,
        interval=interval, node_defaults=node_defaults, edge_defaults=edge_defaults,
        max_memory_mb=max_memory_mb, max_events=max_events, max_work=max_work)


class DynamicNetwork:
    """An interval snapshot with typed identities and explicit static conversion."""
    def __init__(self, nodes, edges, *, directed=False, timeformat="double", version="1.3",
                 interval=None, node_defaults=None, edge_defaults=None, max_memory_mb=256,
                 max_events=1_000_000, max_work=50_000_000, _live_bytes=None, _external_guard=None):
        if (not isinstance(directed, bool) or not isinstance(timeformat, str) or not isinstance(version, str)
                or timeformat not in {"double", "integer", "date", "dateTime"} or version not in {"1.2draft", "1.3"}):
            _error("invalid_option", "Use boolean directed, GEXF 1.2draft/1.3 and double/integer/date/dateTime.")
        self.directed, self.timeformat, self.version = directed, timeformat, version
        self._budget, self._work = _Budget(max_memory_mb), _Work(max_work)
        self._event_limit, self._events = _integer(max_events, "max_events", 10_000_000), 0
        self._record_events = 0
        self._owned, self._live = 1_048_576, _live_bytes
        self._external_guard = _external_guard
        self._nodes, self._edges, self._node_index = [], [], {}
        self._edge_index, self._presence, self._edge_presence = {}, [], []
        self._modes = {"node": {}, "edge": {}}
        self._interval = _interval({} if interval is None else interval, timeformat, version)
        self._defaults = {"node": self._copy_attrs({} if node_defaults is None else node_defaults),
                          "edge": self._copy_attrs({} if edge_defaults is None else edge_defaults)}
        for scope, defaults in self._defaults.items():
            self._modes[scope].update({name: "dynamic" for name in defaults})
        if "weight" in self._defaults["edge"]:
            self._defaults["edge"]["weight"] = _weight(self._defaults["edge"]["weight"])
        self.weighted = "weight" in self._defaults["edge"]
        self._consume(nodes, "node")
        self._consume(edges, "edge")
        self._metadata = dict(representation="continuous interval network", storage="resident", device="cpu",
            timeformat=timeformat, version=version, node_count=len(self._nodes), edge_count=len(self._edges),
            events=self._events, directed=directed, weighted=self.weighted, exact=True, sampled=False,
            max_memory_bytes=self._budget.limit, estimated_owned_bytes=self._owned,
            estimated_import_peak_bytes=self._budget.peak, import_work=self._work.used,
            edge_identity_semantics="unique typed edge IDs; parallel/zero/loop records retained",
            missing_time_bounds="unbounded; activity intersects graph, node and edge intervals",
            snapshot_semantics="explicit MultiNetwork point/window; no implicit aggregate or temporal path model",
            memory_scope="owned resident records, interval indexes and planned buffers; excludes caller input and total process RSS")

    def _guard(self, workspace=0):
        planned = self._owned + int(workspace) + 262144
        self._budget.check(planned)
        if self._external_guard is not None:
            self._external_guard(planned)
        if self._live is not None:
            self._live[0] = planned

    def _charge(self, size):
        self._guard(size)
        self._owned += size
        self._guard()

    def _copy_attrs(self, values):
        size = _attribute_bytes(values)
        self._charge(2 * size)
        return _attributes(values)

    def _spells(self, raw):
        if raw is None:
            raw = [{}]
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Iterable):
            _error("dynamic_time", "Spells must be an iterable of interval dictionaries.")
        intervals = []
        iterator = iter(raw)
        try:
            for bounds in iterator:
                self._event()
                self._charge(1024)
                intervals.append(_interval(bounds, self.timeformat, self.version))
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
        if not intervals:
            _error("dynamic_time", "An explicit spell list must be nonempty; omit spells for unbounded activity.")
        return tuple(intervals)

    def _event(self):
        self._events += 1
        self._record_events += 1
        self._work.add(8)
        if self._events > self._event_limit:
            _error("event_budget", "Dynamic records exceed max_events; no events are truncated.")
        if self._record_events > 4096:
            _error("event_budget", "One resident record supports at most 4096 spell/value events.")

    def _dynamic(self, raw, scope):
        if not isinstance(raw, dict) or len(raw) > 128:
            _error("file_attribute", "Timed attributes require at most 128 named sequences.")
        result = {}
        for name, values in raw.items():
            _attributes({name: 0})
            if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
                _error("file_attribute", "Each dynamic attribute contains timed value dictionaries.")
            self._charge(512)
            events, iterator = [], iter(values)
            try:
                for value in iterator:
                    if not isinstance(value, Mapping) or "value" not in value or set(value) - (_BOUNDS | {"value"}):
                        _error("file_attribute", "Timed values contain value and interval bounds only.")
                    self._event()
                    scalar = _scalar(value["value"])
                    if scope == "edge" and name == "weight":
                        scalar = _weight(scalar)
                    self._charge(1024 + 2 * _attribute_bytes({name: scalar}))
                    events.append((_interval({k: v for k, v in value.items() if k != "value"}, self.timeformat, self.version), scalar))
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
            if not events:
                _error("file_attribute", "Dynamic attribute sequences must be nonempty.")
            result[name] = tuple(events)
        return result

    def _consume(self, records, scope):
        if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
            _error("invalid_data", "Dynamic graph inputs must be iterables of node/edge records.")
        identity = "node" if scope == "node" else "edge_id"
        iterator = iter(records)
        try:
            for raw in iterator:
                self._work.add(16)
                self._record_events = 0
                required = {identity} if scope == "node" else {identity, "source", "target"}
                fields = required | {"spells", "attributes", "dynamic_attributes"} | _BOUNDS | ({"weight"} if scope == "edge" else set())
                if not isinstance(raw, Mapping) or len(raw) > 16 or not required <= set(raw) or set(raw) - fields:
                    _error("invalid_data", "Dynamic records require their exact identities and only documented scalar/spell fields.")
                identifier = _label(raw[identity])
                index = self._node_index if scope == "node" else self._edge_index
                if identifier in index:
                    _error("duplicate_edge_id" if scope == "edge" else "invalid_label", "Dynamic graph identities must be unique within their scope.")
                self._charge(2048 + _identity_bytes(identifier))
                bounds = {k: raw[k] for k in _BOUNDS if k in raw}
                if "spells" in raw and bounds:
                    _error("dynamic_time", "Use direct lifetime bounds or spells, not both on one record.")
                spells = self._spells(raw.get("spells", [bounds] if bounds else None))
                attrs = self._copy_attrs(raw.get("attributes", {}))
                dynamic = self._dynamic(raw.get("dynamic_attributes", {}), scope)
                if scope == "edge" and "weight" in attrs:
                    _error("file_attribute", "Static edge weight belongs in weight, not attributes.")
                if len(set(attrs) | set(dynamic) | set(self._defaults[scope])) > 128:
                    _error("file_attribute", "Each item accepts at most 128 scalar attribute fields.")
                for name in attrs:
                    if self._modes[scope].get(name) == "dynamic":
                        _error("file_attribute", "A named attribute cannot be both static and dynamic in one scope.")
                    self._modes[scope][name] = "static"
                for name in dynamic:
                    if self._modes[scope].get(name) == "static":
                        _error("file_attribute", "A named attribute cannot be both static and dynamic in one scope.")
                    self._modes[scope][name] = "dynamic"
                if len(self._modes[scope]) > 128:
                    _error("file_attribute", "Each scope supports at most 128 named scalar fields.")
                presence = _intersection(_union(spells, self._work), (self._interval,), self._work)
                record = {identity: identifier, "spells": spells, "attributes": attrs, "dynamic_attributes": dynamic}
                if scope == "node":
                    index[identifier] = len(self._nodes)
                    self._nodes.append(record)
                    self._presence.append(presence)
                else:
                    u, v = _label(raw["source"]), _label(raw["target"])
                    if u not in self._node_index or v not in self._node_index:
                        _error("unknown_node", "Dynamic edges must name declared nodes; no endpoints are invented.")
                    presence = _intersection(presence, self._presence[self._node_index[u]], self._work)
                    presence = _intersection(presence, self._presence[self._node_index[v]], self._work)
                    record.update(source=u, target=v, weight=_weight(raw.get("weight", 1.)))
                    self.weighted |= "weight" in raw or "weight" in dynamic
                    index[identifier] = len(self._edges)
                    self._edges.append(record)
                    self._edge_presence.append(presence)
                self._charge(1024 * len(presence))
        finally:
            if hasattr(iterator, "close"):
                iterator.close()

    @property
    def node_count(self):
        return len(self._nodes)

    @property
    def edge_count(self):
        return len(self._edges)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def defaults(self):
        self._guard(2 * sum(_attribute_bytes(v) for v in self._defaults.values()))
        return deepcopy(self._defaults)

    @property
    def interval(self):
        return self._interval.record()

    def _records(self, scope):
        for record in self._nodes if scope == "node" else self._edges:
            yield {**record, "spells": [s.record() for s in record["spells"]],
                "attributes": dict(record["attributes"]),
                "dynamic_attributes": {name: [{"value": value, **period.record()} for period, value in events]
                                       for name, events in record["dynamic_attributes"].items()}}

    def _table(self, scope):
        self._guard(self._owned + 1024 * (self.node_count if scope == "node" else self.edge_count))
        columns = ["node"] if scope == "node" else ["edge_id", "source", "target", "weight"]
        columns += ["spells", "attributes", "dynamic_attributes"]
        records = list(self._records(scope))
        result = as_frame(pd.DataFrame({name: pd.Series([record[name] for record in records], dtype=object)
                                       for name in columns}))
        result.attrs.update(network=self.metadata, exact=True, sampled=False)
        return result

    def nodes(self):
        """Materialize typed node records, original spells and static/timed attributes."""
        return self._table("node")

    def edges(self):
        """Materialize separate typed edge IDs, weights, original spells and attributes."""
        return self._table("edge")

    def summary(self):
        """Tabulate resident topology, time format and admitted interval-event counts."""
        self._guard(16384)
        return as_frame(pd.DataFrame([("Nodes", self.node_count), ("Separate edges", self.edge_count),
            ("Interval/value events", self._events), ("Time format", self.timeformat), ("GEXF version", self.version),
            ("Directed", self.directed)], columns=["Metric", "Value"]))

    def _resolve(self, record, periods, scope, work, *, attributes, weight, dropped):
        result = dict(record["attributes"])
        resolved_weight = record.get("weight", 1.)
        names = set(record["dynamic_attributes"]) | set(self._defaults[scope])
        for name in sorted(names):
            values, active, distinct = [], [], set()
            for period, value in record["dynamic_attributes"].get(name, ()):
                overlap = _intersection((period,), periods, work)
                if overlap:
                    active.extend(overlap)
                    work.add(8 + (len(value) // 8 if isinstance(value, str) else 0))
                    key = (type(value), value)
                    if key not in distinct:
                        distinct.add(key)
                        values.append(value)
                self._guard(1024 * (len(active) + len(values)))
            covered = _covers(_union(active, work), periods, work)
            if not covered:
                fallback = self._defaults[scope].get(name, resolved_weight if scope == "edge" and name == "weight" else _ABSENT)
                if (type(fallback), fallback) not in distinct:
                    values.append(fallback)
            if scope == "edge" and name == "weight":
                if len(values) > 1 and weight == "raise":
                    _error("dynamic_ambiguity", "Edge weight varies or conflicts in this selection; explicitly choose window weight='min' or 'max'.")
                resolved_weight = min(values) if weight == "min" else max(values) if weight == "max" else values[0]
            elif len(values) > 1:
                if attributes == "raise":
                    _error("dynamic_ambiguity", "An attribute varies, is intermittent or conflicts in this selection; choose window attributes='drop' explicitly.")
                self._guard(1024 * (len(dropped) + 1))
                dropped.append({"scope": scope, "id": record["node" if scope == "node" else "edge_id"], "attribute": name})
            elif values and values[0] is not _ABSENT:
                result[name] = values[0]
        return result, resolved_weight

    def _slice(self, query, *, selection, attributes, weight, maximum, kind):
        work, dropped = _Work(maximum), []
        workspace = 2048 * (self.node_count + self.edge_count + self._events) + 262144
        self._guard(workspace)
        available = (self._budget.limit - self._owned - workspace - 262144) / 1024**2
        if available <= 0:
            _error("memory_budget", "The dynamic source and planned slice cannot fit max_memory_mb.")
        nodes, node_attrs = [], {}
        for record, presence in zip(self._nodes, self._presence):
            periods = _intersection(presence, (query,), work)
            if not periods or selection == "cover" and not _covers(presence, (query,), work):
                continue
            attrs, _ = self._resolve(record, periods, "node", work, attributes=attributes, weight=weight, dropped=dropped)
            nodes.append(record["node"])
            if attrs:
                node_attrs[record["node"]] = attrs

        def edges():
            for record, presence in zip(self._edges, self._edge_presence):
                periods = _intersection(presence, (query,), work)
                if not periods or selection == "cover" and not _covers(presence, (query,), work):
                    continue
                attrs, scalar = self._resolve(record, periods, "edge", work, attributes=attributes, weight=weight, dropped=dropped)
                yield dict(edge_id=record["edge_id"], source=record["source"], target=record["target"], weight=scalar, attributes=attrs)

        graph = multigraph(edges(), weight="weight", attributes="attributes", nodes=nodes, node_attributes=node_attrs,
            directed=self.directed, max_memory_mb=available)
        graph._budget.peak += self._owned + workspace + 262144
        graph._budget.limit = self._budget.limit
        graph.weighted = self.weighted
        graph._metadata["dynamic_selection"] = dict(kind=kind, interval=query.record(), timeformat=self.timeformat,
            selection=selection, attributes_policy=attributes, weight_policy=weight, dropped_dynamic_attributes=dropped,
            work_used=work.used, max_work=work.maximum, edge_ids_preserved=True,
            temporal_path_model=False, activity="intersection of graph/node/edge presence")
        graph._refresh()
        return graph

    def at(self, time, *, max_work=50_000_000):
        """Return a point MultiNetwork; ambiguous timed values fail instead of choosing one."""
        if time is None:
            _error("dynamic_time", "A point query needs an explicit finite time.")
        query = _interval({"start": time, "end": time}, self.timeformat, self.version, query=True)
        return self._slice(query, selection="overlap", attributes="raise", weight="raise", maximum=max_work, kind="point")

    def window(self, start=None, end=None, *, start_open=False, end_open=False, selection="overlap",
               attributes="raise", weight="raise", max_work=50_000_000):
        """Select any-overlap or whole-window activity; attribute/weight reduction is explicit.

        Missing bounds are unbounded. Variable/intermittent attributes raise or
        explicitly drop; weight varies only with explicit min/max selection.
        """
        if (not isinstance(start_open, bool) or not isinstance(end_open, bool)
                or not isinstance(selection, str) or selection not in {"overlap", "cover"}
                or not isinstance(attributes, str) or attributes not in {"raise", "drop"}
                or not isinstance(weight, str) or weight not in {"raise", "min", "max"}):
            _error("invalid_option", "Use boolean endpoint openness, overlap/cover, raise/drop attributes and raise/min/max weights.")
        query = _interval({"startopen" if start_open else "start": start,
                           "endopen" if end_open else "end": end}, self.timeformat, self.version, query=True)
        return self._slice(query, selection=selection, attributes=attributes, weight=weight, maximum=max_work, kind="window")

    def write(self, path, *, overwrite=False):
        """Atomically export interval GEXF with typed identities; no static flattening."""
        from openecon._network_dynamic_io import write_dynamic_network
        return write_dynamic_network(self, path, overwrite=overwrite)

    def __getattr__(self, name):
        if not name.startswith("_") and callable(getattr(Network, name, None)):
            def unavailable(*args, **kwargs):
                _error("dynamic_capacity", f"Dynamic {name} needs an explicit at(time) or window(...) and simple projection first.")
            return unavailable
        raise AttributeError(name)


def read_dynamic_network(path, *, max_memory_mb=256, max_file_mb=512,
                         max_events=1_000_000, max_work=50_000_000):
    """Read interval GEXF 1.2draft/1.3 without losing edge IDs, spells or timed values."""
    from openecon._network_dynamic_io import read_dynamic_network as read
    return read(path, max_memory_mb=max_memory_mb, max_file_mb=max_file_mb, max_events=max_events, max_work=max_work)
