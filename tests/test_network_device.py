"""Independent equations, device refusal and allocation-failure contracts."""
from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_communities import _modularity_reduction
from openecon._network_device import execution, resolve


METHODS = ('degree', 'pagerank', 'eigenvector', 'hits', 'katz', 'modularity')


def graph():
    return oe.network([{'source': u, 'target': v, 'weight': w} for u, v, w in
                       [(0, 1, 2), (1, 0, 1), (1, 2, 3), (2, 0, 2), (2, 2, 1)]],
                      nodes=range(4), directed=True, weight='weight')


def call(g, method, device):
    return getattr(g, method)(*[list(range(g.node_count))] if method == 'modularity' else [], device=device)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('device', ['mps', 'meta', 'cpu:0', 'cuda:invalid', 'cuda:-1', 1, None])
def test_invalid_or_unsupported_device_never_returns_cpu_result(method, device):
    with pytest.raises(AnalysisError) as error:
        call(graph(), method, device)
    assert error.value.code == 'network_device'


@pytest.mark.parametrize('method', METHODS)
def test_absent_cuda_is_an_error_even_for_an_empty_graph(monkeypatch, method):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(AnalysisError) as error:
        call(oe.network([], nodes=[]), method, 'cuda')
    assert error.value.code == 'network_device'


def test_device_index_and_preflight_memory_refusal_before_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    assert str(resolve('cuda')) == 'cuda:1'
    with pytest.raises(AnalysisError, match='index'):
        resolve('cuda:2')
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (1, 100))
    with pytest.raises(AnalysisError) as error:
        with execution(graph(), 'cuda:0', 2):
            pytest.fail('Preflight must precede device allocation')
    assert error.value.code == 'network_memory_budget'


@pytest.mark.parametrize('method', METHODS)
def test_allocation_oom_has_no_partial_result_or_fallback(monkeypatch, method):
    g = graph()
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError('injected allocator failure')
    monkeypatch.setattr(torch, 'zeros', fail)
    with pytest.raises(AnalysisError) as error:
        call(g, method, 'cpu')
    assert error.value.code == 'network_memory_budget'
    assert isinstance(error.value.__cause__, torch.OutOfMemoryError)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires a real CUDA device; no emulated pass'))])
def test_independent_dense_equations_and_identity_alignment(device):
    g = graph()
    matrix = np.array([[0., 2., 0., 0.], [1., 0., 3., 0.], [2., 0., 1., 0.], [0., 0., 0., 0.]])
    d = g.degree(device=device)
    assert d.node.tolist() == list(range(4))
    np.testing.assert_array_equal(d.in_degree, [2, 1, 2, 0])
    np.testing.assert_array_equal(d.out_strength, [2, 4, 3, 0])
    transition = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    transition[-1] = .25
    expected = np.linalg.solve(np.eye(4) - .85 * transition.T, np.full(4, .15 / 4))
    np.testing.assert_allclose(g.pagerank(device=device).pagerank, expected, atol=1e-10, rtol=0)
    values, vectors = np.linalg.eig(matrix.T)
    eigen = np.abs(vectors[:, np.argmax(values.real)].real)
    eigen /= np.linalg.norm(eigen)
    np.testing.assert_allclose(g.eigenvector(device=device).eigenvector, eigen, atol=2e-10, rtol=0)
    u, _, vt = np.linalg.svd(matrix)
    result = g.hits(device=device)
    np.testing.assert_allclose(result.hub, np.abs(u[:, 0]) / np.abs(u[:, 0]).sum(), atol=2e-10, rtol=0)
    np.testing.assert_allclose(result.authority, np.abs(vt[0]) / np.abs(vt[0]).sum(), atol=2e-10, rtol=0)
    expected = np.linalg.solve(np.eye(4) - .1 * matrix.T, np.ones(4))
    np.testing.assert_allclose(g.katz(.1, normalization='none', device=device).katz, expected,
                               atol=2e-10, rtol=0)
    assert g.modularity([0, 0, 1, 2], device=device) == pytest.approx(float(Fraction(-2, 27)), abs=1e-14)
    for method in METHODS[:-1]:
        result = call(g, method, device)
        assert result.attrs['device'] == str(resolve(device))
        assert result.attrs['dtype'] == 'float64' and not result.attrs['device_fallback']
        assert g._edges.device.type == 'cpu'


@pytest.mark.parametrize('directed', [False, True])
def test_modularity_reduction_matches_exact_fraction_objective_with_self_loops(directed):
    records = [(0, 0, 2), (0, 1, 3), (1, 2, 5), (2, 0, 7), (2, 2, 1)]
    g = oe.network([dict(source=u, target=v, weight=w) for u, v, w in records],
                   nodes=range(4), directed=directed, weight='weight')
    arcs = records if directed else records + [(v, u, w) for u, v, w in records]
    total = sum(w for _, _, w in arcs)
    for labels in product(range(2), repeat=4):
        within = sum(w for u, v, w in arcs if labels[u] == labels[v])
        null = sum(sum(w for u, _, w in arcs if labels[u] == group) *
                   sum(w for _, v, w in arcs if labels[v] == group) for group in range(2))
        expected = Fraction(within, total) - Fraction(null, total**2)
        actual = _modularity_reduction(g, list(labels), 1., torch.device('cpu'))
        assert math.isclose(actual, float(expected), rel_tol=0, abs_tol=3e-16)
        assert actual == pytest.approx(g.modularity(labels), abs=3e-16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires real CUDA hardware')
@pytest.mark.parametrize('method', METHODS)
def test_real_cuda_cpu_parity_weighted_loops_isolates_and_duplicates(method):
    for directed in (False, True):
        g = oe.network([dict(source=u, target=v, weight=w) for u, v, w in
                        [(1, '1', 3), (1, '1', 2), ('1', 1, 4), ('1', '1', 7)]],
                       nodes=['iso'], directed=directed, weight='weight')
        cpu, gpu = call(g, method, 'cpu'), call(g, method, 'cuda')
        if method == 'modularity':
            assert gpu == pytest.approx(cpu, abs=2e-14)
        else:
            assert gpu.node.tolist() == cpu.node.tolist()
            np.testing.assert_allclose(gpu.iloc[:, 1:].to_numpy(), cpu.iloc[:, 1:].to_numpy(), atol=2e-10, rtol=0)
            assert gpu.attrs['device'].startswith('cuda:')
