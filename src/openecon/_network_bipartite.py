"""Bounded bipartite projections and explicit-pair binary link scores.

No V-by-V matrix or implicit all-pair candidate enumeration is constructed.
Projections count each unique positive connection once (or multiply aggregate
strengths explicitly); link scores use loop-free simple undirected topology.
CPU Torch owns sparse numeric storage; traversal uses the shared compact CSR.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
import math
from numbers import Integral

import pandas as pd
import torch

from openecon._network_sparse import csr
from openecon.dataset import Dataset
from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key, _label, _rows, network


_METHODS = {"common_neighbors", "jaccard", "adamic_adar", "resource_allocation",
            "preferential_attachment"}


def _side(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        _error("invalid_partition", "Each bipartite partition must be integer 0 or 1.")
    return int(value)


def _partition_ids(graph, partition=None, *, _topology=None):
    """Return a full validated 0/1 list in original typed node order.

    Directed input is checked on its weak binary projection. Inferred connected
    components put their smallest exact typed label on side 0; isolates use 0.
    Caller-supplied orientation is preserved, including isolated node choices.
    """
    n, arcs = graph.node_count, graph._arcs._nnz()
    graph._guard(512 * n + 128 * arcs)
    with torch.device("cpu"), torch.no_grad():
        pairs = graph._edges.indices()
        if bool((pairs[0] == pairs[1]).any()):
            _error("not_bipartite", "A graph with a positive self-loop is not bipartite.")
    ids = [-1] * n
    if partition is not None:
        if isinstance(partition, pd.DataFrame):
            if (len(partition) != n
                    or list(partition.columns).count("node") != 1
                    or list(partition.columns).count("partition") != 1):
                _error("invalid_partition", "Use a complete node/partition table with each node exactly once.")
            records = zip(partition["node"], partition["partition"])
        elif isinstance(partition, Mapping):
            if len(partition) != n:
                _error("invalid_partition", "Partition mapping must cover every graph node exactly once.")
            records = partition.items()
        else:
            _error("invalid_partition", "Use a node-to-0/1 mapping or a complete node/partition table.")
        for raw, value in records:
            label = _label(raw)
            if label not in graph._index:
                _error("unknown_node", "Bipartite partition contains an unknown node.")
            node = graph._index[label]
            if ids[node] != -1:
                _error("invalid_partition", "Partition table repeats a node.")
            ids[node] = _side(value)
        if any(side == -1 for side in ids):
            _error("invalid_partition", "Partition must cover every graph node exactly once.")
    view = _topology if _topology is not None else csr(graph, loops=False, undirected=graph.directed)
    if partition is not None:
        for node in range(n):
            if any(ids[other] == ids[node] for other in view.neighbors(node)):
                _error("not_bipartite", "An edge joins nodes on the same supplied bipartite side.")
        return ids
    for root in range(n):
        if ids[root] != -1:
            continue
        ids[root] = 0
        queue, head, minimum = [root], 0, root
        while head < len(queue):
            node = queue[head]
            head += 1
            if _key(graph._labels[node]) < _key(graph._labels[minimum]):
                minimum = node
            following = 1 - ids[node]
            for other in view.neighbors(node):
                if ids[other] == -1:
                    ids[other] = following
                    queue.append(other)
                elif ids[other] != following:
                    _error("not_bipartite", "The graph contains an odd cycle and is not bipartite.")
        if ids[minimum] == 1:
            for node in queue:
                ids[node] = 1 - ids[node]
    return ids


def bipartite(graph, partition=None):
    """Infer or validate both sides, including disconnected components/isolates."""
    ids = _partition_ids(graph, partition)
    return graph._frame({"partition": ids}, kind="network_bipartite",
                        directed_semantics="weak binary projection" if graph.directed else "undirected",
                        inferred=partition is None,
                        isolate_side="0 when inferred; caller choice when supplied",
                        side_zero=sum(side == 0 for side in ids),
                        side_one=sum(side == 1 for side in ids))


def bipartite_projection(graph, partition=None, onto=0, weight="count", *,
                         max_work=50_000_000, max_edges=1_000_000):
    """Project one side with count, binary, or aggregate-strength product weights.

    ``product`` means sum(w[u,z] * w[v,z]) over common opposite-side nodes.
    Directed graphs are rejected; direction/strength cannot be silently lost.
    Selected node and graph scalar attributes survive. Original edge attributes
    have no corresponding projected edge and are explicitly not transferred.
    """
    if graph.directed:
        _error("directed_unsupported", "Bipartite projection requires an undirected graph; choose a weak projection explicitly.")
    onto = _side(onto)
    if not isinstance(weight, str) or weight not in {"count", "binary", "product"}:
        _error("invalid_option", "Projection weight must be 'count', 'binary' or 'product'.")
    maximum = _integer(max_work, "max_work", 2**63 - 1)
    edge_limit = _integer(max_edges, "max_edges", 2**63 - 1)
    n, arcs = graph.node_count, graph._arcs._nnz()
    # The structural lower bound is known before copying partitions or
    # constructing CSR. Reject an impossible request before those allocations.
    if n + arcs > maximum:
        _error("work_budget", f"Bipartite projection setup exceeds max_work={maximum:,}; no partial graph is returned.")
    workspace = 512 * n + 128 * arcs
    graph._guard(workspace)
    view = csr(graph, loops=False)
    # Reuse this operation-owned CSR for validation instead of constructing a
    # second sorted sparse copy. It is released after projected edges are ready.
    ids = _partition_ids(graph, partition, _topology=view)
    planned = n + view.arc_count
    for center in range(n):
        if ids[center] != onto:
            degree = view.offsets[center + 1] - view.offsets[center]
            planned += degree * (degree - 1) // 2
    if planned > maximum:
        _error("work_budget", f"Bipartite projection requires {planned:,} planned pair/traversal units, above max_work={maximum:,}; no partial graph is returned.")
    chosen = [label for label, side in zip(graph._labels, ids) if side == onto]
    # Covers the projected graph's own labels/attributes plus existing parent,
    # projection map resize, bounded import blocks and constructor temporaries.
    attributes = graph._metadata.get("attribute_storage_bytes", 0)
    result_base = 2 * graph._label_bytes + 2048 * len(chosen) + 3 * attributes
    graph._guard(workspace + result_base)
    projected, corrections = {}, {}
    for center in range(n):
        if ids[center] == onto:
            continue
        neighbors, strengths = view.neighbors(center), view.weights_for(center)
        for at in range(len(neighbors)):
            left = neighbors[at]
            for following in range(at + 1, len(neighbors)):
                right = neighbors[following]
                pair = (min(left, right), max(left, right))
                if pair not in projected:
                    size = len(projected) + 1
                    if size > edge_limit:
                        _error("edge_budget", f"Bipartite projection exceeds max_edges={edge_limit:,}; no partial graph is returned.")
                    graph._guard(workspace + result_base + 1024 * size
                                 + 1024 * min(size, 4096))
                    projected[pair] = 0.0
                    if weight == "product":
                        corrections[pair] = 0.0
                if weight == "binary":
                    projected[pair] = 1.0
                elif weight == "count":
                    value = projected[pair] + 1.0
                    if value == projected[pair]:
                        _error("precision", "A shared-neighbor count cannot be represented exactly in float64.")
                    projected[pair] = value
                else:
                    term = strengths[at] * strengths[following]
                    value = projected[pair] + term
                    if term == 0 or not math.isfinite(term) or not math.isfinite(value):
                        _error("precision", "Projected aggregate-strength products overflow or underflow float64.")
                    previous = projected[pair]
                    corrections[pair] += ((previous - value) + term
                                          if abs(previous) >= abs(term) else (term - value) + previous)
                    projected[pair] = value
    del view

    if weight == "product":
        for pair, correction in corrections.items():
            value = projected[pair] + correction
            if not math.isfinite(value):
                _error("precision", "Projected aggregate-strength products overflow float64.")
            projected[pair] = value
        del corrections

    def records():
        for (left, right), value in projected.items():
            yield {"source": graph._labels[left], "target": graph._labels[right], "weight": value}

    size = len(projected)
    graph._guard(result_base + 1024 * size + 1024 * min(max(1, size), 4096))
    result = network(records(), nodes=chosen, weight="weight", directed=False,
                     batch_rows=min(max(1, size), 4096),
                     max_memory_mb=graph._budget.limit / 1024**2)
    result.weighted = weight != "binary"
    result._metadata.update(weighted=result.weighted, derived_from="bipartite projection",
                            parent_node_count=n, parent_edge_count=graph.edge_count,
                            partition_side=onto, projection_weight=weight,
                            weight_semantics={"count": "number of distinct common opposite-side nodes",
                                              "binary": "one for each positive shared-neighbor connection",
                                              "product": "sum of products of aggregate input strengths"}[weight],
                            edge_attributes="not transferred to derived edges",
                            planned_work=planned, max_work=maximum, max_edges=edge_limit,
                            memory_scope="parent graph plus planned projection/derived buffers; excludes caller input and total process RSS")
    selected = set(chosen)
    return result.with_attributes(nodes={label: attrs for label, attrs in graph._node_attributes.items() if label in selected},
                                  graph_attributes=graph._graph_attributes)


def _candidate_size(pairs, source, target):
    if isinstance(pairs, Dataset):
        return pairs.row_count
    if isinstance(pairs, pd.DataFrame):
        return len(pairs)
    if isinstance(pairs, Mapping):
        if source not in pairs or target not in pairs:
            _error("invalid_columns", "Both candidate endpoint columns are required.")
        try:
            left, right = len(pairs[source]), len(pairs[target])
        except TypeError:
            _error("invalid_data", "Candidate endpoint columns must be sized sequences.")
        if left != right:
            _error("invalid_data", "Candidate endpoint columns must have equal length.")
        return left
    if isinstance(pairs, Sized) and not isinstance(pairs, (str, bytes)):
        return len(pairs)
    return None


def _candidate_rows(pairs, source, target, rows):
    if isinstance(pairs, (pd.DataFrame, Mapping, Dataset)):
        yield from _rows(pairs, [source, target], rows)
        return
    if not isinstance(pairs, Iterable) or isinstance(pairs, (str, bytes)):
        _error("invalid_data", "Use candidate endpoint columns, a table, Dataset, pair tuples or records.")
    iterator = iter(pairs)

    def records():
        for record in iterator:
            if isinstance(record, Mapping):
                yield record
            elif isinstance(record, (tuple, list)) and len(record) == 2:
                yield {source: record[0], target: record[1]}
            else:
                _error("invalid_data", "Each candidate must be an endpoint pair or a source/target record.")

    try:
        yield from _rows(records(), [source, target], rows)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def link_prediction(graph, pairs, method="jaccard", source="source", target="target", *,
                    max_pairs=100_000, max_work=50_000_000, batch_rows=65536):
    """Score explicit distinct-node pairs; preserve input order and duplicates.

    Scores use positive-edge binary undirected topology, ignore loops/strengths,
    and also permit explicitly requested already-connected candidate pairs.
    No absent pair is generated and no candidate is silently filtered/sampled.
    """
    if graph.directed:
        _error("directed_unsupported", "Binary link prediction requires an undirected graph.")
    if not isinstance(method, str) or method not in _METHODS:
        _error("invalid_option", "Choose common_neighbors, jaccard, adamic_adar, resource_allocation or preferential_attachment.")
    if (not isinstance(source, str) or not source.strip() or len(source) > 200
            or not isinstance(target, str) or not target.strip() or len(target) > 200 or source == target):
        _error("invalid_columns", "Candidate endpoint columns must be distinct nonempty bounded names.")
    pair_limit = _integer(max_pairs, "max_pairs", 2**63 - 1)
    maximum = _integer(max_work, "max_work", 2**63 - 1)
    rows = _integer(batch_rows, "batch_rows", 1_000_000)
    count = _candidate_size(pairs, source, target)
    if count is not None and count > pair_limit:
        _error("pair_budget", f"Candidate table exceeds max_pairs={pair_limit:,}; no partial result is returned.")
    rows = min(rows, pair_limit + 1, max(1, count) if count is not None else rows,
               max(1, (graph._budget.limit - graph._base_bytes - 4096) // 2048))
    n, arcs = graph.node_count, graph._arcs._nnz()
    workspace = 512 * n + 128 * arcs + 1024 * rows
    graph._guard(workspace + 512 * (count if count is not None else 0))
    planned = n + arcs
    if count is not None:
        # Every candidate needs at least one scoring unit. Reject a known-size
        # impossible request before constructing CSR or copying any pair rows.
        if planned + count > maximum:
            _error("work_budget", f"Link prediction setup/candidates exceed max_work={maximum:,}.")
    if planned > maximum:
        _error("work_budget", f"Link prediction setup exceeds max_work={maximum:,}.")
    view = csr(graph, loops=False)
    degrees = [view.offsets[node + 1] - view.offsets[node] for node in range(n)]
    left_labels, right_labels, scores = [], [], []
    iterator = _candidate_rows(pairs, source, target, rows)
    try:
        for block, size in iterator:
            if len(scores) + size > pair_limit:
                _error("pair_budget", f"Candidate iterator exceeds max_pairs={pair_limit:,}; no partial result is returned.")
            graph._guard(workspace + 512 * (len(scores) + size))
            for raw_left, raw_right in block:
                left_label, right_label = _label(raw_left), _label(raw_right)
                if left_label not in graph._index or right_label not in graph._index:
                    _error("unknown_node", "Candidate endpoints must be known graph nodes.")
                left, right = graph._index[left_label], graph._index[right_label]
                if left == right:
                    _error("invalid_pairs", "Link prediction candidates must have distinct endpoints.")
                planned += 1 if method == "preferential_attachment" else degrees[left] + degrees[right] + 1
                if planned > maximum:
                    _error("work_budget", f"Candidate scoring exceeds max_work={maximum:,}; no partial result is returned.")
                if method == "preferential_attachment":
                    score = float(degrees[left] * degrees[right])
                else:
                    first, second = view.neighbors(left), view.neighbors(right)
                    at = following = common = 0
                    score = correction = 0.0
                    while at < len(first) and following < len(second):
                        a, b = first[at], second[following]
                        if a < b:
                            at += 1
                        elif b < a:
                            following += 1
                        else:
                            common += 1
                            if method in {"adamic_adar", "resource_allocation"}:
                                term = 1.0 / (math.log(degrees[a]) if method == "adamic_adar" else degrees[a])
                                # Neumaier accumulation for unequal common-node contributions.
                                value = score + term
                                correction += (score - value) + term if abs(score) >= abs(term) else (term - value) + score
                                score = value
                            at += 1
                            following += 1
                    if method == "common_neighbors":
                        score = float(common)
                    elif method == "jaccard":
                        union = degrees[left] + degrees[right] - common
                        score = common / union if union else 0.0
                    else:
                        score += correction
                left_labels.append(graph._labels[left])
                right_labels.append(graph._labels[right])
                scores.append(score)
    finally:
        iterator.close()
    result = as_frame(pd.DataFrame({"source": pd.Series(left_labels, dtype=object),
                                   "target": pd.Series(right_labels, dtype=object),
                                   "score": pd.Series(scores, dtype="float64")}))
    result.attrs.update(network=graph.metadata, kind="network_link_prediction", method=method,
                        candidate_count=len(scores), input_source=source, input_target=target,
                        weight_semantics="positive-edge binary undirected topology; strengths ignored",
                        self_loops="excluded", candidate_semantics="only explicit distinct-node pairs, including existing edges",
                        planned_work=planned, max_work=maximum, max_pairs=pair_limit,
                        batch_rows=rows, sampled=False)
    return result
