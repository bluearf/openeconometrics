"""Compact CPU CSR traversal for native network algorithms.

Numeric storage and row construction use Torch. Python memoryviews provide
zero-copy scalar traversal; ``numpy()`` is only the buffer bridge, not a solver.
No cached copy is retained on a graph, so workspaces die after each operation.
"""
from __future__ import annotations

import torch


class CSR:
    def __init__(self, offsets, columns, weights):
        self._tensors = offsets, columns, weights
        self.offsets = memoryview(offsets.numpy())
        self.columns = memoryview(columns.numpy())
        self.weights = memoryview(weights.numpy())
        self.node_count = len(self.offsets) - 1
        self.arc_count = len(self.columns)
        self.storage_bytes = sum(t.numel() * t.element_size() for t in self._tensors)

    def neighbors(self, node):
        return self.columns[self.offsets[node]:self.offsets[node + 1]]

    def weights_for(self, node):
        return self.weights[self.offsets[node]:self.offsets[node + 1]]

    def row(self, node):
        return zip(self.neighbors(node), self.weights_for(node))


def csr(graph, *, reverse=False, loops=True, undirected=False):
    """Return compact row-sorted arcs, optionally taking a weak simple projection.

    ``undirected=True`` is a topological weak projection: values count reciprocal
    arc presence, without adding potentially huge unused strengths. Loop removal
    affects topology only. Original graph tensors remain untouched.
    """
    n = graph.node_count
    arcs = graph._arcs
    extra = arcs._nnz() if undirected and graph.directed else 0
    a = arcs._nnz() + extra
    # Include COO projection sort/reduction and optional filtered tensor copies.
    needs_sort = reverse or not arcs.is_coalesced()
    factor = 128 if extra else 96 if needs_sort else 64 if not loops else 32
    graph._guard(factor * a + 32 * n)
    with torch.device('cpu'), torch.no_grad():
        pairs, values = arcs._indices(), arcs._values()
        if extra:
            pairs = torch.cat((pairs, pairs.flip(0)), dim=1)
            values = torch.ones(pairs.shape[1], dtype=torch.float64)
            weak = torch.sparse_coo_tensor(pairs, values, (n, n), dtype=torch.float64,
                                          device='cpu', check_invariants=True).coalesce()
            pairs, values = weak.indices(), weak.values()
            needs_sort = reverse  # projection coalesces into canonical row/column order
        if not loops:
            keep = pairs[0] != pairs[1]
            pairs, values = pairs[:, keep], values[keep]
        source, target = pairs[1] if reverse else pairs[0], pairs[0] if reverse else pairs[1]
        # Stable row order avoids a source*n+target key whose integer can overflow.
        if needs_sort:
            # Lexicographic rows are required by linear neighbor intersections.
            # Sorting source alone leaves undirected rows in two unordered blocks.
            # Stable column-then-row sorts avoid an overflowing source*n+target key.
            by_column = torch.argsort(target, stable=True)
            order = by_column[torch.argsort(source[by_column], stable=True)]
            columns = target[order].contiguous()
            weights = values[order].contiguous()
        else:
            # Forward canonical COO already has sorted rows and columns. Keep
            # zero-copy views instead of re-sorting and copying every edge.
            columns, weights = target, values
        counts = torch.bincount(source, minlength=n)
        offsets = torch.empty(n + 1, dtype=torch.int64)
        offsets[0] = 0
        torch.cumsum(counts, dim=0, out=offsets[1:])
    return CSR(offsets, columns, weights)
