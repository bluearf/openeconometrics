"""Bounded, atomically published graph snapshots with disk B-tree adjacency.

SQLite stores identity/coalescing indexes and both adjacency directions. It is
storage, not an analytical solver; numerical methods still use native Torch.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import weakref

from openecon.dataset import Dataset, _file_identity
from openecon.analysis_contracts import AnalysisError
from openecon.data import DataError
from openecon.networks import Network, _Budget, _error, _integer, _key, _label, _missing, _rows, _weight

_FORMAT = "openecon.disk-network.v1"


def _identity(raw):
    value = _label(raw)
    return ("i:" + str(value) if isinstance(value, int) else "s:" + value), value


def _decode(token):
    return int(token[2:]) if token.startswith("i:") else token[2:]


def _connect(path, limit, *, readonly=False):
    uri = Path(path).resolve().as_uri() + "?mode=ro" if readonly else str(path)
    connection = sqlite3.connect(uri, uri=readonly)
    try:
        connection.execute("PRAGMA mmap_size=0")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA cache_size=-{max(16, min(1024, limit // 4096))}")
    except BaseException:
        connection.close()
        raise
    return connection


def _cancel(callback):
    if callback is not None and callback():
        _error("cancelled", "Disk graph construction was cancelled; no completed snapshot was published.")


def build_store(data, path=None, *, source="source", target="target", weight=None,
                directed=False, nodes=None, missing="raise", batch_rows=65536,
                max_memory_mb=256, max_disk_mb=4096, cancelled=None,
                node_attributes=None, edge_attributes=None, graph_attributes=None):
    from openecon._network_io import _attribute_bytes, _attributes

    if not isinstance(directed, bool) or not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("invalid_option", "directed must be boolean; missing must be 'raise' or 'drop'.")
    if cancelled is not None and not callable(cancelled):
        _error("invalid_option", "cancelled must be a callable returning a cancellation flag.")
    columns = [source, target] + ([weight] if weight is not None else [])
    if (any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in columns)
            or len(set(columns)) != len(columns)):
        _error("invalid_columns", "Source, target and optional weight must be distinct valid column names.")
    requested = _integer(batch_rows, "batch_rows", 1_000_000)
    budget = _Budget(max_memory_mb)
    disk = _Budget(max_disk_mb)
    # Reserve SQLite's bounded page cache and fixed setup before input batches.
    fixed = min(1024 * 1024, budget.limit // 4) + 262144
    budget.check(fixed + 32768)
    rows = min(requested, max(1, (budget.limit - fixed) // 32768))
    budget.check(fixed + rows * 32768)
    for value in (node_attributes, edge_attributes, graph_attributes):
        if value is not None and not isinstance(value, Mapping):
            _error("file_attribute", "Node, edge and graph attributes must be mappings.")
    attribute_bytes = _attribute_bytes(graph_attributes or {})
    budget.check(fixed + attribute_bytes + 32768)
    graph_attrs = _attributes(dict(graph_attributes or {}))
    # Scalar attribute validation may need a single larger bounded workspace.
    budget.check(fixed + len(json.dumps(graph_attrs, ensure_ascii=False).encode("utf-8")) * 16 + 32768)
    if nodes is not None and (not isinstance(nodes, Iterable) or isinstance(nodes, (str, bytes, Mapping))):
        _error("invalid_label", "nodes must be an iterable of string/integer isolate labels.")
    temporary = tempfile.mkdtemp(prefix="openecon-graph-") if path is None else None
    destination = Path(temporary, "graph.sqlite") if temporary else Path(path).expanduser().resolve()
    if destination.exists():
        if temporary:
            shutil.rmtree(temporary)
        _error("store_exists", "The graph destination already exists; open it or choose a new path.")
    connection = None
    staging = None
    returned = False
    try:
        fd, staging_name = tempfile.mkstemp(prefix="." + destination.name + ".building-", dir=destination.parent)
        os.close(fd)
        staging = Path(staging_name)
        connection = _connect(staging, budget.limit)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        # Reserve space for journal and staging overhead as well as DB pages.
        connection.execute(f"PRAGMA max_page_count={max(1, disk.limit // (3 * 4096))}")
        connection.executescript("""
            CREATE TABLE nodes(id INTEGER PRIMARY KEY, token TEXT UNIQUE NOT NULL, attrs TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE edges(src INTEGER NOT NULL REFERENCES nodes(id), dst INTEGER NOT NULL REFERENCES nodes(id), w REAL NOT NULL, attrs TEXT NOT NULL DEFAULT '{}', PRIMARY KEY(src,dst)) WITHOUT ROWID;
            CREATE INDEX incoming ON edges(dst,src);
            CREATE TABLE sources(path TEXT PRIMARY KEY, identity TEXT NOT NULL);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        if isinstance(data, Dataset):
            data.assert_unchanged()
            for file in data._files:
                connection.execute("INSERT INTO sources VALUES (?,?)", (str(file), json.dumps(_file_identity(file))))
        node_count = label_bytes = 0
        max_record_workspace = 65536
        input_rows = missing_rows = zero_rows = positive_rows = peak_rows = 0
        digest = hashlib.sha256()

        def intern(raw):
            nonlocal node_count, label_bytes
            token, value = _identity(raw)
            found = connection.execute("SELECT id FROM nodes WHERE token=?", (token,)).fetchone()
            if found is not None:
                return found[0], value
            identifier = node_count
            connection.execute("INSERT INTO nodes(id,token) VALUES (?,?)", (identifier, token))
            node_count += 1
            label_bytes += 512 + 2 * sys.getsizeof(value) + (len(value.encode("utf-8")) if isinstance(value, str) else 32)
            return identifier, value

        for label in nodes if nodes is not None else ():
            _cancel(cancelled)
            intern(label)
        iterator = _rows(data, columns, rows)
        try:
            for block, size in iterator:
                _cancel(cancelled)
                if isinstance(data, Dataset):
                    data.assert_unchanged()
                peak_rows = max(peak_rows, size)
                for record in block:
                    record = tuple(record)
                    input_rows += 1
                    if any(_missing(value) for value in record):
                        if missing == "raise":
                            _error("missing_data", "Edge endpoints/weights contain missing values; choose missing='drop' explicitly.")
                        missing_rows += 1
                        continue
                    i, u = intern(record[0])
                    j, v = intern(record[1])
                    value = _weight(record[2]) if weight is not None else 1.0
                    digest.update(repr((_key(u), _key(v), value)).encode("utf-8") + b"\n")
                    if value == 0:
                        zero_rows += 1
                        continue
                    positive_rows += 1
                    if not directed and i > j:
                        i, j = j, i
                    connection.execute("INSERT INTO edges(src,dst,w) VALUES (?,?,?) ON CONFLICT(src,dst) DO UPDATE SET w=edges.w+excluded.w", (i, j, value))
                    aggregate = connection.execute("SELECT w FROM edges WHERE src=? AND dst=?", (i, j)).fetchone()[0]
                    if not math.isfinite(aggregate):
                        _error("precision", "Aggregated duplicate weights overflow float64.")
                    if input_rows % 256 == 0:
                        _cancel(cancelled)
        finally:
            iterator.close()

        def lookup(raw):
            token, _ = _identity(raw)
            found = connection.execute("SELECT id FROM nodes WHERE token=?", (token,)).fetchone()
            if found is None:
                _error("unknown_node", "Attributes refer to a node outside the graph.")
            return found[0]

        for label, attrs in (node_attributes or {}).items():
            _cancel(cancelled)
            addition = _attribute_bytes(attrs)
            budget.check(fixed + addition + 32768)
            attribute_bytes += addition
            max_record_workspace = max(max_record_workspace, addition + 32768)
            encoded = json.dumps(_attributes(attrs), allow_nan=False, ensure_ascii=False)
            budget.check(fixed + len(encoded.encode("utf-8")) * 16 + 32768)
            max_record_workspace = max(max_record_workspace, len(encoded.encode("utf-8")) * 16 + 32768)
            connection.execute("UPDATE nodes SET attrs=? WHERE id=?", (encoded, lookup(label)))
        for pair, attrs in (edge_attributes or {}).items():
            _cancel(cancelled)
            if not isinstance(pair, tuple) or len(pair) != 2:
                _error("file_attribute", "Edge attributes use (source, target) tuple keys.")
            i, j = map(lookup, pair)
            if not directed and i > j:
                i, j = j, i
            addition = _attribute_bytes(attrs)
            budget.check(fixed + addition + 32768)
            attribute_bytes += addition
            max_record_workspace = max(max_record_workspace, addition + 32768)
            attrs = _attributes(attrs)
            if "weight" in attrs:
                _error("file_attribute", "Edge attribute 'weight' is reserved for the aggregate edge weight.")
            encoded = json.dumps(attrs, allow_nan=False, ensure_ascii=False)
            budget.check(fixed + len(encoded.encode("utf-8")) * 16 + 32768)
            max_record_workspace = max(max_record_workspace, len(encoded.encode("utf-8")) * 16 + 32768)
            if connection.execute("UPDATE edges SET attrs=? WHERE src=? AND dst=?", (encoded, i, j)).rowcount != 1:
                _error("file_attribute", "Attributes refer to an edge outside the graph.")
        edge_count = connection.execute("SELECT count(*) FROM edges").fetchone()[0]
        metadata = dict(format=_FORMAT, complete=True, node_count=node_count, edge_count=edge_count,
            directed=directed, weighted=weight is not None, input_rows=input_rows,
            missing_rows_dropped=missing_rows, zero_weight_rows_dropped=zero_rows,
            positive_edge_rows=positive_rows, duplicate_edge_rows_aggregated=positive_rows-edge_count,
            requested_batch_rows=requested, effective_batch_rows=rows, actual_peak_batch_rows=peak_rows,
            input_sha256=digest.hexdigest(), label_storage_bound=label_bytes,
            attribute_storage_bound=attribute_bytes, max_record_workspace=max_record_workspace,
            max_memory_bytes=budget.limit, estimated_import_peak_bytes=budget.peak,
            max_disk_bytes=disk.limit, storage="disk indexed adjacency and transpose",
            coalescing="float64 sums in input order; undirected endpoint canonicalization",
            memory_scope="bounded import buffers and SQLite page cache; excludes caller input, runtime and total process RSS",
            graph_attributes=graph_attrs, analytical_device="cpu")
        connection.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(metadata, allow_nan=False, ensure_ascii=False),))
        _cancel(cancelled)
        if isinstance(data, Dataset):
            data.assert_unchanged()
        connection.commit()
        connection.close()
        connection = None
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        _cancel(cancelled)
        # Same-filesystem exclusive link atomically publishes one complete DB.
        # It cannot replace a destination created concurrently by another job.
        os.link(staging, destination)
        staging.unlink()
        if os.name == "posix":
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        result = open_store(destination, max_memory_mb=max_memory_mb)
        if temporary:
            result._cleanup = weakref.finalize(result, shutil.rmtree, temporary, True)
        returned = True
        return result
    except FileExistsError:
        _error("store_exists", "The graph destination was created concurrently; no existing file was replaced.")
    except sqlite3.OperationalError as exc:
        if "full" in str(exc).lower():
            _error("disk_budget", "Disk graph construction exceeded available disk space or max_disk_mb.")
        _error("store_io", "Disk graph construction failed; no usable snapshot was returned.")
    finally:
        if connection is not None:
            connection.close()
        if staging is not None:
            for name in (str(staging), str(staging) + "-journal"):
                Path(name).unlink(missing_ok=True)
        if temporary and not returned:
            shutil.rmtree(temporary, ignore_errors=True)


class DiskNetwork:
    """Reopenable snapshot with bounded identity, edge and adjacency readers.

    Degree, CPU PageRank and weak components scan disk edges with admitted O(V)
    state. Other numerical methods require explicit guarded materialization.
    """
    def __init__(self, path, metadata, limit):
        self.path = Path(path)
        self._metadata, self._limit = metadata, limit
        self._identity = _file_identity(self.path)
        self.node_count, self.edge_count = metadata["node_count"], metadata["edge_count"]
        self.directed, self.weighted = metadata["directed"], metadata["weighted"]
        self._cleanup = None
        self._closed = False

    def __repr__(self):
        return f"DiskNetwork(nodes={self.node_count}, edges={self.edge_count}, directed={self.directed})"

    @property
    def metadata(self):
        return deepcopy(dict(self._metadata, disk_bytes=self.path.stat().st_size))

    def _reader(self):
        if self._closed:
            _error("store_closed", "The disk graph is closed.")
        if _file_identity(self.path) != self._identity:
            _error("store_changed", "The disk graph changed since it was opened; reopen and verify it.")
        return _connect(self.path, self._limit, readonly=True)

    def _batches(self, sql, parameters=(), *, batch_rows=65536):
        requested = _integer(batch_rows, "batch_rows", 1_000_000)
        fixed = min(1024 * 1024, self._limit // 4) + 262144
        workspace = self._metadata["max_record_workspace"]
        _Budget(self._limit / 1024**2).check(fixed + workspace)
        rows = min(requested, max(1, (self._limit - fixed) // workspace))
        connection = self._reader()
        try:
            cursor = connection.execute(sql, parameters)
            while block := cursor.fetchmany(rows):
                yield block
            if _file_identity(self.path) != self._identity:
                _error("store_changed", "The disk graph changed during reading.")
        finally:
            connection.close()

    def iter_nodes(self, *, batch_rows=65536):
        for block in self._batches("SELECT token,attrs FROM nodes ORDER BY id", batch_rows=batch_rows):
            yield [{"node": _decode(token), "attributes": json.loads(attrs)} for token, attrs in block]

    def iter_edges(self, *, batch_rows=65536):
        sql = "SELECT a.token,b.token,e.w,e.attrs FROM edges e JOIN nodes a ON a.id=e.src JOIN nodes b ON b.id=e.dst ORDER BY e.src,e.dst"
        for block in self._batches(sql, batch_rows=batch_rows):
            yield [{"source": _decode(u), "target": _decode(v), "weight": w,
                    "attributes": json.loads(attrs)} for u, v, w, attrs in block]

    def neighbors(self, node, *, incoming=False, batch_rows=65536):
        if not isinstance(incoming, bool):
            _error("invalid_option", "incoming must be boolean.")
        token, _ = _identity(node)
        connection = self._reader()
        try:
            found = connection.execute("SELECT id FROM nodes WHERE token=?", (token,)).fetchone()
        finally:
            connection.close()
        if found is None:
            _error("unknown_node", "The requested node is outside the graph.")
        identifier = found[0]
        left, right = ("dst", "src") if incoming else ("src", "dst")
        sql = f"SELECT n.token,e.w FROM edges e JOIN nodes n ON n.id=e.{right} WHERE e.{left}=?"
        parameters = (identifier,)
        if not self.directed:
            sql += f" UNION ALL SELECT n.token,e.w FROM edges e JOIN nodes n ON n.id=e.{left} WHERE e.{right}=? AND e.src!=e.dst"
            parameters += (identifier,)
        for block in self._batches(sql, parameters, batch_rows=batch_rows):
            yield [{"node": _decode(token), "weight": w} for token, w in block]

    def summary(self):
        from openecon.frame import as_frame
        import pandas as pd

        connection = self._reader()
        try:
            loops = connection.execute("SELECT count(*) FROM edges WHERE src=dst").fetchone()[0]
            if _file_identity(self.path) != self._identity:
                _error("store_changed", "The disk graph changed during its summary scan.")
        finally:
            connection.close()
        records = [("Nodes", self.node_count), ("Edges", self.edge_count), ("Directed", self.directed),
            ("Weighted input", self.weighted), ("Self-loops", loops), ("Input rows", self._metadata["input_rows"]),
            ("Missing rows dropped", self._metadata["missing_rows_dropped"]),
            ("Zero-weight rows dropped", self._metadata["zero_weight_rows_dropped"]),
            ("Duplicate rows aggregated", self._metadata["duplicate_edge_rows_aggregated"]), ("Storage", "disk")]
        result = as_frame(pd.DataFrame(records, columns=["Metric", "Value"]))
        result.attrs.update(network=self.metadata, kind="network_summary", exact=True,
                            sampled=False, device="cpu", storage="disk", graph_materialized=False)
        return result

    def degree(self, *, batch_rows=65536, max_work=50_000_000, max_scan_bytes=1024**3):
        from openecon._network_disk_algorithms import degree
        return degree(self, batch_rows=batch_rows, max_work=max_work, max_scan_bytes=max_scan_bytes)

    def pagerank(self, damping=.85, tol=1e-10, max_iter=200, personalization=None, device="cpu", *,
                 batch_rows=65536, max_work=50_000_000, max_scan_bytes=1024**3):
        from openecon._network_disk_algorithms import pagerank
        return pagerank(self, damping, tol, max_iter, personalization, device,
                        batch_rows=batch_rows, max_work=max_work, max_scan_bytes=max_scan_bytes)

    def components(self, connectivity="weak", *, batch_rows=65536, max_work=50_000_000,
                   max_scan_bytes=1024**3):
        from openecon._network_disk_algorithms import components
        return components(self, connectivity, batch_rows=batch_rows,
                          max_work=max_work, max_scan_bytes=max_scan_bytes)

    def materialize(self):
        from openecon._network_io import attach_attributes
        from openecon.networks import network

        _Budget(self._limit / 1024**2).check(self.resident_import_bound)
        nodes = (row["node"] for block in self.iter_nodes() for row in block)
        edges = (row for block in self.iter_edges() for row in block)
        graph = network(edges, weight="weight", directed=self.directed, nodes=nodes,
                        max_memory_mb=self._limit / 1024**2, batch_rows=self._metadata["effective_batch_rows"])
        node_attrs = {row["node"]: row["attributes"] for block in self.iter_nodes() for row in block if row["attributes"]}
        edge_attrs = {(row["source"], row["target"]): row["attributes"] for block in self.iter_edges() for row in block if row["attributes"]}
        graph.weighted = self.weighted
        for key in ("input_rows", "missing_rows_dropped", "zero_weight_rows_dropped", "positive_edge_rows",
                    "duplicate_edge_rows_aggregated", "input_sha256"):
            graph._metadata[key] = self._metadata[key]
        graph._metadata.update(storage="resident", weighted=self.weighted, materialized_from=str(self.path))
        return attach_attributes(graph, nodes=node_attrs, edges=edge_attrs,
                                 graph_attributes=self._metadata["graph_attributes"])

    @property
    def resident_import_bound(self):
        return (self._metadata["label_storage_bound"] + 1024 * self.edge_count +
                512 * self.node_count + 2 * self._metadata["attribute_storage_bound"] + 262144)

    def __getattr__(self, name):
        if not name.startswith("_") and callable(getattr(Network, name, None)):
            def delegated(*args, **kwargs):
                _error("capacity", f"Disk-native {name} is unavailable; explicitly materialize an admitted small graph.")
            return delegated
        raise AttributeError(name)

    def close(self):
        self._closed = True
        if self._cleanup is not None:
            self._cleanup()


def open_store(path, *, max_memory_mb=256, verify_source=True):
    """Open a completed immutable snapshot; verify its original sources by default."""
    if not isinstance(verify_source, bool):
        _error("invalid_option", "verify_source must be boolean.")
    budget = _Budget(max_memory_mb)
    budget.check(min(1024 * 1024, budget.limit // 4) + 262144 + 32768)
    path = Path(path).expanduser().resolve()
    before = _file_identity(path)
    connection = None
    try:
        connection = _connect(path, budget.limit, readonly=True)
        size = connection.execute("SELECT length(CAST(value AS BLOB)) FROM metadata WHERE key='manifest'").fetchone()
        if size is None or size[0] > 3_000_000:
            _error("invalid_store", "The file is not a completed disk graph snapshot.")
        budget.check(size[0] * 16 + 262144)
        row = connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        metadata = json.loads(row[0])
        if not isinstance(metadata, dict) or metadata.get("format") != _FORMAT or metadata.get("complete") is not True:
            _error("invalid_store", "The graph format is unsupported or incomplete.")
        if verify_source:
            for source, identity in connection.execute("SELECT path,identity FROM sources"):
                if list(_file_identity(Path(source))) != json.loads(identity):
                    _error("source_changed", "An original source changed since graph construction.")
        for table, key in (("nodes", "node_count"), ("edges", "edge_count")):
            if connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != metadata[key]:
                _error("invalid_store", "Disk graph counts do not match its completed manifest.")
        if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
            _error("invalid_store", "The disk graph failed its storage integrity check.")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            _error("invalid_store", "Disk adjacency refers to a missing node identity.")
        if _file_identity(path) != before:
            _error("store_changed", "The graph changed while opening it.")
        metadata["source_verification"] = "file identities verified" if verify_source else "explicit independent snapshot"
        return DiskNetwork(path, metadata, budget.limit)
    except (AnalysisError, DataError):
        raise
    except (sqlite3.DatabaseError, ValueError, KeyError, TypeError):
        _error("invalid_store", "The file is not a valid completed disk graph snapshot.")
    finally:
        if connection is not None:
            connection.close()
