"""Explicit resource admission and typed tables for native network models."""

from __future__ import annotations

import time

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import Network, _error, _integer, _real


class Work:
    def __init__(self, graph, *, workspace, max_work, timeout, cancelled=None):
        self.limit = _integer(max_work, "max_work", 2**63 - 1)
        self.timeout = _real(timeout, "invalid_option", "timeout must be positive and finite.")
        if self.timeout <= 0 or (cancelled is not None and not callable(cancelled)):
            _error("invalid_option", "timeout must be positive; cancelled must be a callable.")
        graph._guard(workspace)
        self.workspace, self.used = workspace, 0
        self.started, self.cancelled = time.monotonic(), cancelled

    def add(self, amount):
        if self.cancelled is not None and self.cancelled():
            _error("cancelled", "Network model cancelled; no partial fit is returned.")
        if time.monotonic() - self.started > self.timeout:
            _error("time_budget", "Network model exceeded its explicit timeout.")
        if self.used + amount > self.limit:
            _error("work_budget", "Network model exceeds max_work; no partial fit is returned.")
        self.used += amount

    def metadata(self):
        return {
            "work_used": self.used,
            "max_work": self.limit,
            "timeout_seconds": self.timeout,
            "estimated_workspace_bytes": self.workspace,
            "memory_scope": "owned model workspace plus source graph; excludes process RSS",
            "device": "cpu",
            "dtype": "float64",
        }


def binary_graph(graph, *, directed=None):
    if not isinstance(graph, Network):
        _error("unsupported_graph", "This model requires a resident simple Network.")
    if directed is not None and graph.directed != directed:
        _error(
            "unsupported_graph",
            "This model requires " + ("a directed" if directed else "an undirected") + " Network.",
        )
    indices = graph._edges.indices()
    if bool((indices[0] == indices[1]).any()):
        _error(
            "unsupported_graph", "Model sample space excludes self-loops; remove them explicitly."
        )
    if graph.node_count < 2:
        _error("invalid_model", "The model requires at least two nodes.")


def adjacency(graph):
    """Call only after admitting the deliberately dense bounded model geometry."""
    n = graph.node_count
    result = torch.zeros((n, n), dtype=torch.float64, device="cpu")
    indices = graph._edges.indices()
    result[indices[0], indices[1]] = 1
    if not graph.directed:
        result[indices[1], indices[0]] = 1
    return result


class ModelResult(dict):
    def summary(self):
        meta = self["metadata"]
        selected = [
            "model",
            "method",
            "nodes",
            "observations",
            "converged",
            "objective",
            "iterations",
            "seed",
            "uncertainty",
            "device",
            "dtype",
        ]
        return as_frame(
            pd.DataFrame(
                [(name, meta[name]) for name in selected if name in meta],
                columns=["Metric", "Value"],
            )
        )

    def to_latex(self, buf=None, **kwargs):
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)


def coefficients(terms, theta, covariance=None):
    rows = []
    for i, name in enumerate(terms):
        se = None if covariance is None else float(covariance[i, i].clamp_min(0).sqrt())
        rows.append((name, float(theta[i]), se))
    return as_frame(pd.DataFrame(rows, columns=["term", "coefficient", "std_error"]))
