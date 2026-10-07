"""Traversal-order guarantees and verified elimination of redundant sorting."""
import pytest
import torch

import openecon as oe
from openecon._network_sparse import csr


@pytest.mark.parametrize('directed', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('loops', [False, True])
@pytest.mark.parametrize('weak', [False, True])
def test_csr_rows_match_independent_sorted_arc_oracle(directed, reverse, loops, weak):
    graph = oe.network({'source': [4, 1, 3, 2, 4, 3],
                        'target': [1, 3, 2, 4, 4, 1],
                        'weight': [1., 2., 3., 4., 5., 6.]},
                       directed=directed, weight='weight', nodes=['iso'])
    expected = {}
    edges = zip(graph._edges.indices()[0].tolist(), graph._edges.indices()[1].tolist(),
                graph._edges.values().tolist())
    for u, v, value in edges:
        arcs = [(u, v, value)]
        if not directed and u != v:
            arcs.append((v, u, value))
        if directed and weak:
            arcs = [(u, v, 1.), (v, u, 1.)]
        for u, v, value in arcs:
            if not loops and u == v:
                continue
            if reverse:
                u, v = v, u
            expected[u, v] = expected.get((u, v), 0.) + value
    view = csr(graph, reverse=reverse, loops=loops, undirected=weak)
    for u in range(graph.node_count):
        assert list(view.row(u)) == sorted((v, value) for (source, v), value in expected.items()
                                           if source == u)


def test_forward_directed_csr_skips_sorting_and_shares_original_buffers(monkeypatch):
    graph = oe.network({'source': [3, 1, 2], 'target': [1, 2, 3]}, directed=True)

    def unnecessary(*args, **kwargs):
        pytest.fail('Canonical forward COO must not be sorted again.')

    monkeypatch.setattr(torch, 'argsort', unnecessary)
    view = csr(graph)
    assert view._tensors[1].data_ptr() == graph._arcs._indices()[1].data_ptr()
    assert view._tensors[2].data_ptr() == graph._arcs._values().data_ptr()
    assert view._tensors[0].numel() == graph.node_count + 1
