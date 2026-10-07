"""Bounded selected-pair prediction from native sparse block-model results."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import pandas as pd

from openecon.frame import as_frame
from openecon.networks import _Budget, _error, _integer, _label, _real


def _flag(value):
    if not pd.api.types.is_bool(value):
        _error("invalid_result", "Block-model identification flags must be booleans.")
    return bool(value)


def _columns(frame, columns):
    if not isinstance(frame, pd.DataFrame) or any(list(frame.columns).count(c) != 1 for c in columns):
        _error("invalid_result", "Block-model tables require unique model-specific columns.")


def expected_edges(result, pairs, *, max_pairs=100_000, max_memory_mb=256):
    """Validate a saved fit and predict only explicitly supplied exact node pairs.

    Bounds cover resident fit tables and planned indexing/output buffers, not
    process RSS or the caller's input. Iterables consume at most max_pairs+1
    records, and oversized output is rejected without silently sampling it.
    """
    maximum = _integer(max_pairs, "max_pairs", 2**63 - 1)
    budget = _Budget(max_memory_mb)
    if not isinstance(result, Mapping) or not isinstance(result.get("metadata"), Mapping):
        _error("invalid_result", "Expected-edge prediction needs a native block-model result.")
    meta = result["metadata"]
    model, directed = meta.get("model"), meta.get("directed")
    if model not in {"bernoulli", "poisson", "degree_corrected_poisson"} or not isinstance(directed, bool):
        _error("invalid_result", "Unsupported block-model family or directedness.")
    nodes = _integer(meta.get("nodes"), "nodes", 2**63 - 1)
    groups = _integer(meta.get("groups"), "groups", nodes)
    corrected = model == "degree_corrected_poisson"
    if meta.get("self_loops", False) is not corrected:
        _error("invalid_result", "Loop sample space disagrees with the model family.")
    parameters = (["theta_out", "theta_in", "theta_out_identified", "theta_in_identified"]
                  if directed else ["theta", "theta_identified"]) if corrected else []
    rate = "omega" if corrected else "probability" if model == "bernoulli" else "mean_count"
    membership, blocks = result.get("membership"), result.get("blocks")
    _columns(membership, ["node", "block", *parameters])
    _columns(blocks, ["source_block", "target_block", rate, "identified"])
    expected_cells = groups**2 if directed else groups * (groups + 1) // 2
    if len(membership) != nodes or len(blocks) != expected_cells:
        _error("invalid_result", "Membership or complete block-table size disagrees with fit metadata.")
    # Check shape-based buffers before deep label accounting or building maps.
    live = 8192 + 1024 * nodes + 640 * expected_cells
    budget.check(live)
    live += int(membership.memory_usage(index=True, deep=True).sum())
    live += int(blocks.memory_usage(index=True, deep=True).sum())
    budget.check(live)
    member = {}
    present = set()
    norms_left, norms_right = [[] for _ in range(groups)], [[] for _ in range(groups)]
    group_flags = {}
    for row in membership.itertuples(index=False, name=None):
        # Columns can be reordered without changing their meaning.
        values = dict(zip(membership.columns, row))
        node = _label(values["node"])
        block = _integer(values["block"], "block", groups - 1, zero=True)
        if node in member:
            _error("invalid_result", "Membership contains duplicate exact node IDs.")
        if corrected:
            if directed:
                left, right = values["theta_out"], values["theta_in"]
                left_id, right_id = _flag(values["theta_out_identified"]), _flag(values["theta_in_identified"])
            else:
                left = right = values["theta"]
                left_id = right_id = _flag(values["theta_identified"])
            left = _real(left, "invalid_result", "Node parameters must be finite probabilities.")
            right = _real(right, "invalid_result", "Node parameters must be finite probabilities.")
            if (not (0 <= left <= 1 and 0 <= right <= 1)
                    or (0 < left < 2**-53) or (0 < right < 2**-53)
                    or (not left_id and left != 0) or (not right_id and right != 0)):
                _error("invalid_result", "Node parameters violate nonnegative normalized or unidentified-zero conventions.")
            if block in group_flags and group_flags[block] != (left_id, right_id):
                _error("invalid_result", "Node identification must be consistent within each group.")
            group_flags[block] = left_id, right_id
            norms_left[block].append(left)
            norms_right[block].append(right)
        else:
            left = right = 1.
            left_id = right_id = True
        member[node] = (block, left, right, left_id, right_id)
        present.add(block)
    if len(present) != groups:
        _error("invalid_result", "Every fitted group must contain a node.")
    if corrected:
        for block in range(groups):
            for values, identified in zip((norms_left[block], norms_right[block]), group_flags[block]):
                if not math.isclose(math.fsum(values), float(identified), rel_tol=0, abs_tol=1e-12):
                    _error("invalid_result", "Node parameters must normalize in each positive-stub group.")
    cells = {}
    for row in blocks.itertuples(index=False, name=None):
        values = dict(zip(blocks.columns, row))
        a = _integer(values["source_block"], "source_block", groups - 1, zero=True)
        b = _integer(values["target_block"], "target_block", groups - 1, zero=True)
        if (not directed and a > b) or (a, b) in cells:
            _error("invalid_result", "Block pairs must be unique and canonically ordered.")
        identified = _flag(values["identified"])
        if corrected and identified != (group_flags[a][0] and group_flags[b][1]):
            _error("invalid_result", "Block identification disagrees with its relevant node groups.")
        raw = values[rate]
        if not corrected and not identified:
            if not pd.api.types.is_number(raw) or not math.isnan(float(raw)):
                _error("invalid_result", "An unidentified zero-dyad rate must be missing.")
            mean = math.nan
        else:
            mean = _real(raw, "invalid_result", "Block rates must be finite and nonnegative.")
            if (not 0 <= mean <= 2**53 or (model == "bernoulli" and mean > 1)
                    or (corrected and not identified and mean != 0)):
                _error("invalid_result", "Block rates violate their model or unidentified-zero conventions.")
        cells[a, b] = mean, identified
    if isinstance(pairs, pd.DataFrame):
        if any(list(pairs.columns).count(c) != 1 for c in ("source", "target")):
            _error("invalid_pairs", "Pair tables need unique source and target columns.")
        if len(pairs) > maximum:
            _error("output_budget", "Selected-pair prediction exceeds max_pairs; no output is truncated.")
        source_index, target_index = pairs.columns.get_loc("source"), pairs.columns.get_loc("target")
        iterator = ((row[source_index], row[target_index]) for row in pairs.itertuples(index=False, name=None))
    elif isinstance(pairs, Iterable) and not isinstance(pairs, (str, bytes, Mapping)):
        if hasattr(pairs, "__len__") and len(pairs) > maximum:
            _error("output_budget", "Selected-pair prediction exceeds max_pairs; no output is truncated.")
        iterator = iter(pairs)
    else:
        _error("invalid_pairs", "Supply source/target rows or an iterable of two-element node pairs.")
    sources, targets, means, probabilities, flags = [], [], [], [], []
    for index, pair in enumerate(iterator):
        if index >= maximum:
            _error("output_budget", "Selected-pair prediction exceeds max_pairs; no output is truncated.")
        budget.check(live + 1024 * (index + 1))
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            _error("invalid_pairs", "Each selected pair must contain exactly two node IDs.")
        source, target = _label(pair[0]), _label(pair[1])
        if source not in member or target not in member:
            _error("unknown_node", "Selected pairs must reference exact fitted node IDs.")
        if source == target and not corrected:
            _error("invalid_pairs", "This block model excludes self-loop dyads.")
        a, left, _, left_id, _ = member[source]
        b, _, right, _, right_id = member[target]
        cell = (a, b) if directed or a <= b else (b, a)
        mean, identified = cells[cell]
        if not identified and not corrected:
            _error("invalid_result", "An eligible selected pair refers to an unidentified zero-dyad block.")
        if corrected:
            # Multiply the large count by each small factor before the factors
            # can underflow; graph count limits bound omega to exact binary64.
            mean = mean * left * right
            if not directed and source == target:
                mean /= 2
            identified = identified and left_id and right_id
        probability = mean if model == "bernoulli" else -math.expm1(-mean)
        sources.append(source)
        targets.append(target)
        means.append(mean)
        probabilities.append(probability)
        flags.append(identified)
    frame = as_frame(pd.DataFrame({"source": pd.Series(sources, dtype=object),
        "target": pd.Series(targets, dtype=object), "mean_count": pd.Series(means, dtype=float),
        "presence_probability": pd.Series(probabilities, dtype=float), "identified": pd.Series(flags, dtype=bool)}))
    frame.attrs.update(kind="network_expected_edges", model=model, directed=directed,
        self_loops=corrected, sampled=False, parameter_uncertainty="not estimated",
        identification_scope="block and relevant node parameters; zero-stub means use an unidentified zero convention",
        memory_scope="resident fit tables plus planned indexing/output; excludes caller input and process RSS",
        estimated_workspace_bytes=budget.peak, max_pairs=maximum)
    return frame
