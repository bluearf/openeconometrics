"""Independent rank-one singular vectors and Katz linear-system oracles."""
from fractions import Fraction
import math

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.mark.parametrize('scale', [1., 1e300, 1e-300])
@pytest.mark.parametrize('normalization', ['l1', 'l2'])
def test_hits_matches_known_rank_one_singular_vectors(scale, normalization):
    graph = oe.network({'source': [1, 1, '1', '1'], 'target': ['a', 'b', 'a', 'b'],
                        'weight': [3*scale, 4*scale, 6*scale, 8*scale]},
                       weight='weight', directed=True, nodes=['iso'])
    result = graph.hits(normalization=normalization)
    hub_divisor = 3 if normalization == 'l1' else math.sqrt(5)
    auth_divisor = 7 if normalization == 'l1' else 5
    lookup = result.set_index('node')
    assert lookup.loc[1, 'hub'] == pytest.approx(1/hub_divisor, abs=1e-11)
    assert lookup.loc['1', 'hub'] == pytest.approx(2/hub_divisor, abs=1e-11)
    assert lookup.loc['a', 'authority'] == pytest.approx(3/auth_divisor, abs=1e-11)
    assert lookup.loc['b', 'authority'] == pytest.approx(4/auth_divisor, abs=1e-11)
    assert lookup.loc['iso', 'hub'] == lookup.loc['iso', 'authority'] == 0
    assert result.attrs['relative_singular_residual'] <= 1e-10
    assert '\\begin{tabular}' in result.to_latex()


def solve_fraction(matrix, target):
    augmented = [[Fraction(x) for x in row] + [Fraction(value)]
                 for row, value in zip(matrix, target)]
    for column in range(len(target)):
        pivot = next(row for row in range(column, len(target)) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        denominator = augmented[column][column]
        augmented[column] = [x/denominator for x in augmented[column]]
        for row in range(len(target)):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [x-factor*y for x, y in zip(augmented[row], augmented[column])]
    return [float(row[-1]) for row in augmented]


@pytest.mark.parametrize('directed', [False, True])
@pytest.mark.parametrize('direction', ['in', 'out'])
def test_katz_matches_independent_exact_fraction_linear_system(directed, direction):
    records = [(0, 1, 2), (1, 2, 3), (2, 0, 1), (2, 2, 2)]
    graph = oe.network([{'source': u, 'target': v, 'weight': w} for u,v,w in records],
                       nodes=range(4), directed=directed, weight='weight')
    alpha = Fraction(1, 10)
    matrix = [[Fraction(i == j) for j in range(4)] for i in range(4)]
    for u,v,w in records:
        arcs = [(u,v,w)] + ([(v,u,w)] if not directed and u != v else [])
        for u,v,w in arcs:
            if direction == 'out':
                u,v = v,u
            matrix[v][u] -= alpha*w
    baseline = {i: i+1 for i in range(4)}
    expected = solve_fraction(matrix, [baseline[i] for i in range(4)])
    result = graph.katz(alpha=float(alpha), beta=baseline, direction=direction, normalization='none')
    assert dict(zip(result.node, result.katz)) == pytest.approx(dict(enumerate(expected)), abs=1e-8)
    assert result.attrs['relative_error_bound_estimate'] <= 1e-10


def test_katz_safe_automatic_scaling_and_score_normalization():
    graph = oe.network({'source':[1,'1'], 'target':['1',1], 'weight':[1e308,1e308]},
                       directed=True, weight='weight')
    result = graph.katz()
    assert result.attrs['alpha_selected_automatically']
    assert result.attrs['contraction_bound'] == pytest.approx(.85)
    assert result.katz.tolist() == pytest.approx([1/math.sqrt(2)]*2)


@pytest.mark.parametrize('method', ['hits','katz'])
def test_sparse_spectral_empty_isolates_and_cpu_tensor_context(method):
    empty = oe.network({'source':[], 'target':[]})
    graph = oe.network({'source':[], 'target':[]}, nodes=['a','b'])
    with torch.device('meta'):
        assert len(getattr(empty, method)()) == 0
        result = getattr(graph, method)()
    assert result.attrs['device'] == 'cpu' and result.attrs['iterations'] == 0
    if method == 'hits':
        assert result.hub.sum() == result.authority.sum() == 0
    else:
        assert result.katz.tolist() == pytest.approx([1/math.sqrt(2)]*2)


@pytest.mark.parametrize('method', ['hits','katz'])
def test_spectral_work_and_memory_guard_before_tensor_allocation(method, monkeypatch):
    graph = oe.network({'source':['a'], 'target':['b']}, directed=True)
    def unexpected(*args, **kwargs):
        pytest.fail('A failing preflight must precede tensor allocation.')
    monkeypatch.setattr(torch, 'zeros', unexpected)
    monkeypatch.setattr(torch, 'empty', unexpected)
    with pytest.raises(AnalysisError) as error:
        getattr(graph, method)(max_work=1)
    assert error.value.code == 'network_work_budget'
    graph._budget.limit = graph._base_bytes + 4096
    with pytest.raises(AnalysisError) as error:
        getattr(graph, method)()
    assert error.value.code == 'network_memory_budget'


@pytest.mark.parametrize('options', [{'alpha':-1}, {'alpha':1}, {'beta':0}, {'beta':{'a':1}},
                                    {'beta':{'a':1,'unknown':1}}, {'normalization':'bad'}, {'tol':0}])
def test_invalid_katz_options(options):
    graph = oe.network({'source':['a'], 'target':['b']}, directed=True)
    with pytest.raises(AnalysisError):
        graph.katz(**options)


@pytest.mark.parametrize('method', ['hits','katz'])
def test_nonconvergence_is_explicit_and_graph_is_unchanged(method):
    graph = oe.network({'source':['a','b'], 'target':['b','c']}, directed=True)
    before = graph._edges.clone()
    with pytest.raises(AnalysisError) as error:
        getattr(graph, method)(max_iter=1, tol=1e-14)
    assert error.value.code == 'network_nonconvergence'
    assert torch.equal(before.indices(), graph._edges.indices())
    assert torch.equal(before.values(), graph._edges.values())


@pytest.mark.parametrize('scale', [1e-300, 1., 1e300])
@pytest.mark.parametrize('normalization', ['none', 'l1', 'l2'])
@pytest.mark.parametrize('direction', ['in', 'out'])
def test_katz_chain_scale_invariant_iteration_against_exact_fraction_oracle(scale, normalization, direction):
    graph = oe.network({'source': [1, '1'], 'target': ['1', 'c']}, directed=True)
    base = [float(Fraction(1)), float(Fraction(37, 20)), float(Fraction(1029, 400))]
    if direction == 'out':
        base.reverse()
    divisor = sum(base) if normalization == 'l1' else math.hypot(*base) if normalization == 'l2' else 1.
    expected = [value / divisor * (scale if normalization == 'none' else 1.) for value in base]
    result = graph.katz(beta=scale, direction=direction, normalization=normalization)
    assert result.node.tolist() == [1, '1', 'c']
    assert result.katz.tolist() == pytest.approx(expected, rel=1e-12, abs=0.)
    assert result.attrs['iterations'] == 3
    assert result.attrs['relative_error_bound_estimate'] == 0
    assert result.attrs['baseline_scale'] == scale
    assert result.attrs['iteration_scale'] == 'beta/max(beta)'


@pytest.mark.parametrize('scale', [1e-300, 1e300, 1e308])
@pytest.mark.parametrize('normalization', ['l1', 'l2'])
def test_zero_alpha_safe_normalization_of_extreme_baseline(scale, normalization):
    graph = oe.network({'source': ['a'], 'target': ['b']}, directed=True)
    result = graph.katz(alpha=0, beta=scale, normalization=normalization)
    expected = .5 if normalization == 'l1' else 1 / math.sqrt(2)
    assert result.katz.tolist() == pytest.approx([expected, expected], abs=1e-15)
    assert result.attrs['iterations'] == 0
    assert result.attrs['alpha'] == 0


@pytest.mark.parametrize('normalization', ['none', 'l1', 'l2'])
def test_zero_alpha_bypasses_irrelevant_mixed_edge_scaling_and_edge_work(normalization, monkeypatch):
    import openecon._network_spectral as module
    graph = oe.network({'source': [1, '1'], 'target': ['1', 'c'], 'weight': [1e308, 1e-300]},
                       weight='weight', directed=True)
    beta = {1: 1., '1': 2., 'c': 3.}

    def forbidden(*args, **kwargs):
        pytest.fail('Exact alpha=0 must not read or scale irrelevant edge strengths.')

    monkeypatch.setattr(module, '_scaled', forbidden)
    result = graph.katz(alpha=0, beta=beta, normalization=normalization, max_work=graph.node_count)
    divisor = 6 if normalization == 'l1' else math.sqrt(14) if normalization == 'l2' else 1.
    assert result.katz.tolist() == pytest.approx([value / divisor for value in beta.values()])
    assert result.attrs['work_used'] == graph.node_count
    assert result.attrs['weight_scale'] is None
    assert result.attrs['relative_error_bound_estimate'] == 0


def test_zero_alpha_unnormalized_preserves_extreme_beta_dynamic_range():
    graph = oe.network({'source': ['a'], 'target': ['b']}, directed=True)
    result = graph.katz(alpha=0, beta={'a': 1e308, 'b': 1e-300}, normalization='none')
    assert result.katz.tolist() == [1e308, 1e-300]


@pytest.mark.parametrize('normalization', ['l1', 'l2'])
def test_normalized_katz_rejects_positive_beta_underflow_but_accepts_genuine_zeros(normalization):
    graph = oe.network({'source': ['a'], 'target': ['b']}, directed=True)
    beta = {'a': 1e308, 'b': 1e-300}
    with pytest.raises(AnalysisError) as error:
        graph.katz(alpha=0, beta=beta, normalization=normalization)
    assert error.value.code == 'network_precision'
    assert graph.katz(alpha=0, beta=beta, normalization='none').katz.tolist() == [1e308, 1e-300]
    assert graph.katz(alpha=0, beta={'a': 1e308, 'b': 0.}, normalization=normalization).katz.tolist() == [1., 0.]


@pytest.mark.parametrize('normalization', ['l1', 'l2'])
def test_normalization_final_division_underflow_is_explicit(normalization):
    graph = oe.network({'source': [], 'target': []}, nodes=['a', 'b', 'c', 'd', 'tiny'])
    beta = {'a': 1., 'b': 1., 'c': 1., 'd': 1., 'tiny': 5e-324}
    with pytest.raises(AnalysisError) as error:
        graph.katz(alpha=0, beta=beta, normalization=normalization)
    assert error.value.code == 'network_precision'


def test_normalized_katz_can_represent_scores_when_raw_scores_overflow():
    graph = oe.network({'source': ['a', 'b'], 'target': ['b', 'a']}, directed=True)
    result = graph.katz(beta=1e308, normalization='l2')
    assert result.katz.tolist() == pytest.approx([1 / math.sqrt(2)] * 2, abs=1e-14)
    with pytest.raises(AnalysisError) as error:
        graph.katz(beta=1e308, normalization='none')
    assert error.value.code == 'network_precision'


def test_nonzero_katz_rejects_baseline_and_coefficient_underflow():
    graph = oe.network({'source': ['a'], 'target': ['b']}, directed=True)
    with pytest.raises(AnalysisError) as error:
        graph.katz(beta={'a': 1e308, 'b': 1e-300})
    assert error.value.code == 'network_precision'
    tiny = oe.network({'source': ['a'], 'target': ['b'], 'weight': [1e-300]},
                      weight='weight', directed=True)
    with pytest.raises(AnalysisError) as error:
        tiny.katz(alpha=1e-300)
    assert error.value.code == 'network_precision'


def test_unscaled_katz_output_underflow_is_explicit_while_normalized_scores_are_valid():
    graph = oe.network({'source': ['a'], 'target': ['b']}, directed=True)
    beta = {'a': 5e-324, 'b': 0.}
    normalized = graph.katz(alpha=.1, beta=beta, normalization='l2')
    assert normalized.katz.tolist() == pytest.approx([1 / math.sqrt(1.01), .1 / math.sqrt(1.01)], rel=1e-12)
    with pytest.raises(AnalysisError) as error:
        graph.katz(alpha=.1, beta=beta, normalization='none')
    assert error.value.code == 'network_precision'


def test_automatic_katz_alpha_metadata_handles_original_coefficient_overflow():
    graph = oe.network({'source': ['a'], 'target': ['b'], 'weight': [5e-324]},
                      weight='weight', directed=True)
    result = graph.katz(normalization='none')
    assert result.katz.tolist() == pytest.approx([1., 1.85], rel=1e-12)
    assert result.attrs['alpha'] is None
    assert result.attrs['alpha_representable'] is False
    assert result.attrs['alpha_scaled'] == pytest.approx(.85)


@pytest.mark.parametrize('normalization', ['l1', 'l2'])
def test_edgeless_katz_baseline_extreme_values_have_valid_normalization(normalization):
    graph = oe.network({'source': [], 'target': []}, nodes=[1, '1'])
    for scale in (1e-300, 1e308):
        result = graph.katz(beta=scale, normalization=normalization)
        expected = .5 if normalization == 'l1' else 1 / math.sqrt(2)
        assert result.katz.tolist() == pytest.approx([expected] * 2, abs=1e-15)
