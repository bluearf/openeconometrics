"""Independent oracles for the fixed-effect absorption kernel.

Demeaning is checked against explicit dummy-variable least squares (numpy lstsq),
graph routines against scipy.sparse.csgraph and brute-force loops, and the degrees
of freedom against the numerical rank of the dummy matrix.
"""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as scipy_components

from openecon.engines import absorb
from openecon.engines.contracts import KernelError


def _dummies(codes, levels):
    matrix = np.zeros((len(codes), levels))
    matrix[np.arange(len(codes)), codes] = 1.0
    return matrix


def _dummy_residual(x, dims, weights=None):
    """Residuals of weighted least squares on the explicit dummy matrix."""
    design = np.hstack([_dummies(codes, levels) for codes, levels in dims])
    root = np.ones(len(x)) if weights is None else np.sqrt(weights)
    target = x * (root[:, None] if x.ndim == 2 else root)
    coefficients = np.linalg.lstsq(design * root[:, None], target, rcond=None)[0]
    return x - design @ coefficients


def _relative_error(actual, expected):
    actual, expected = np.atleast_2d(actual.T).T, np.atleast_2d(expected.T).T
    error = np.linalg.norm(actual - expected, axis=0) / np.linalg.norm(expected, axis=0)
    return float(error.max())


def _tensors(dims):
    return [(torch.as_tensor(codes, dtype=torch.int64), levels) for codes, levels in dims]


def _design(seed, count, balanced, weighted):
    rng = np.random.default_rng(seed)
    if balanced:
        shape = (7, 5, 4)[:count]
        grid = np.indices(shape).reshape(count, -1)
        dims = [(np.tile(grid[d], 3), shape[d]) for d in range(count)]
        n = len(dims[0][0])
    else:
        n = 1400
        first = np.minimum(rng.geometric(0.03, n) - 1, 119)
        dims = [(first, 120)]
        if count > 1:
            # Banded assignment: level of the second dimension follows the first one,
            # so the two-way graph is sparse and alternating projections are slow.
            dims.append(((first // 4 + rng.integers(0, 3, n)) % 30, 30))
        if count > 2:
            dims.append((rng.integers(0, 6, n), 6))
    x = rng.normal(size=(n, 4)) + 3.0
    x[:, 1] += 0.4 * dims[0][0]
    x[:, 2] *= 1 + dims[-1][0]
    weights = rng.uniform(0.2, 3.0, n) if weighted else None
    return x, dims, weights


def _ladder(rungs, repeats=3, seed=5):
    """Chain a_0 - b_0 - a_1 - b_1 - ...: a path graph, the worst case for plain sweeps."""
    first = np.repeat(np.concatenate([np.arange(rungs), np.arange(rungs - 1)]), repeats)
    second = np.repeat(np.concatenate([np.arange(rungs), np.arange(1, rungs)]), repeats)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(first), 2))
    x[:, 1] += np.linspace(0.0, 20.0, rungs)[first]
    return x, [(first, rungs), (second, rungs)]


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_demean_matches_explicit_dummy_projection(count, weighted, balanced):
    x, dims, weights = _design(100 + count, count, balanced, weighted)
    expected = _dummy_residual(x, dims, weights)
    tensor_weights = None if weights is None else torch.as_tensor(weights)
    default = absorb.demean(torch.as_tensor(x), _tensors(dims), tensor_weights)
    assert default.converged and default.values.shape == x.shape
    assert default.values.dtype == torch.float64
    assert _relative_error(default.values.numpy(), expected) < 1e-8
    tight = absorb.demean(torch.as_tensor(x), _tensors(dims), tensor_weights, tol=1e-13)
    assert _relative_error(tight.values.numpy(), expected) < 1e-11
    if count == 1:
        assert (default.iterations, default.method, default.max_update) == (1, "within", 0.0)
    else:
        assert default.method == "symmetric_kaczmarz_cg"
        assert default.max_update <= 1e-10 and tight.max_update <= 1e-13
        assert 1 <= default.iterations <= tight.iterations


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_plain_alternating_projections_reach_the_same_limit(count, weighted):
    x, dims, weights = _design(7, count, False, weighted)
    expected = _dummy_residual(x, dims, weights)
    tensor_weights = None if weights is None else torch.as_tensor(weights)
    plain = absorb.demean(torch.as_tensor(x), _tensors(dims), tensor_weights, accelerate=False)
    fast = absorb.demean(torch.as_tensor(x), _tensors(dims), tensor_weights)
    assert plain.method == "symmetric_kaczmarz" and plain.max_update <= 1e-10
    assert _relative_error(plain.values.numpy(), expected) < 1e-7
    assert fast.iterations < plain.iterations


def test_vector_input_keeps_its_shape_and_equals_the_matrix_column():
    x, dims, weights = _design(11, 2, False, True)
    block = absorb.demean(torch.as_tensor(x), _tensors(dims), torch.as_tensor(weights), tol=1e-13)
    vector = absorb.demean(torch.as_tensor(x[:, 1].copy()), _tensors(dims),
                           torch.as_tensor(weights), tol=1e-13)
    assert vector.values.shape == (len(x),)
    assert_allclose(vector.values.numpy(), block.values[:, 1].numpy(), rtol=0, atol=1e-10)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_demean_is_idempotent_and_orthogonal_to_every_dummy(count):
    x, dims, weights = _design(21, count, False, True)
    tensors, tensor_weights = _tensors(dims), torch.as_tensor(weights)
    once = absorb.demean(torch.as_tensor(x), tensors, tensor_weights, tol=1e-13).values
    twice = absorb.demean(once, tensors, tensor_weights, tol=1e-13).values
    scale = float(once.abs().max())
    assert_allclose(twice.numpy(), once.numpy(), rtol=0, atol=1e-11 * scale)
    for codes, levels in tensors:
        means = absorb.group_means(once, codes, levels, tensor_weights)
        assert float(means.abs().max()) < 1e-11 * scale


def test_inputs_are_not_modified():
    x, dims, weights = _design(31, 3, False, True)
    values, tensor_weights = torch.as_tensor(x).clone(), torch.as_tensor(weights)
    tensors = _tensors(dims)
    before = values.clone(), tensor_weights.clone(), [codes.clone() for codes, _ in tensors]
    for accelerate in (True, False):
        result = absorb.demean(values, tensors, tensor_weights, accelerate=accelerate)
        assert result.values is not values
    absorb.fixed_effect_estimates(values[:, 0], tensors, tensor_weights)
    absorb.group_means(values, tensors[0][0], tensors[0][1], tensor_weights)
    assert torch.equal(values, before[0]) and torch.equal(tensor_weights, before[1])
    assert all(torch.equal(codes, saved) for (codes, _), saved in zip(tensors, before[2]))
    assert not result.values.requires_grad and not result.values.is_inference()


def test_strided_inputs_give_the_same_result_as_contiguous_ones():
    x, dims, weights = _design(33, 2, False, True)
    tensors, tensor_weights = _tensors(dims), torch.as_tensor(weights)
    expected = absorb.demean(torch.as_tensor(x), tensors, tensor_weights).values
    column_major = torch.as_tensor(np.asfortranarray(x))
    assert not column_major.is_contiguous()
    assert torch.equal(absorb.demean(column_major, tensors, tensor_weights).values, expected)
    wide = torch.as_tensor(np.repeat(x, 2, axis=1))[:, ::2]
    assert torch.equal(absorb.demean(wide, tensors, tensor_weights).values, expected)
    strided_codes = [(torch.stack([codes, codes], dim=1)[:, 1], levels)
                     for codes, levels in tensors]
    single = absorb.demean(wide[:, 2], strided_codes, tensor_weights).values
    assert_allclose(single.numpy(), expected[:, 2].numpy(), rtol=0, atol=1e-8)


def test_zero_dimensions_return_an_independent_copy():
    values = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    result = absorb.demean(values, [])
    assert (result.iterations, result.converged, result.method) == (0, True, "none")
    assert torch.equal(result.values, values)
    result.values.zero_()
    assert float(values.sum()) == 15.0
    assert absorb.fixed_effect_estimates(values, []) == []


def test_group_means_weighted_unweighted_and_empty_levels():
    rng = np.random.default_rng(3)
    codes = np.array([0, 0, 2, 2, 2, 5, 5])
    x, weights = rng.normal(size=(7, 3)), rng.uniform(0.5, 2.0, 7)
    tensor_codes = torch.as_tensor(codes)
    plain = absorb.group_means(torch.as_tensor(x), tensor_codes, 7)
    weighted = absorb.group_means(torch.as_tensor(x), tensor_codes, 7, torch.as_tensor(weights))
    assert plain.shape == (7, 3)
    for level in range(7):
        rows = codes == level
        if rows.any():
            assert_allclose(plain[level].numpy(), x[rows].mean(axis=0), rtol=1e-14)
            assert_allclose(weighted[level].numpy(),
                            np.average(x[rows], axis=0, weights=weights[rows]), rtol=1e-14)
        else:
            assert float(plain[level].abs().max()) == 0.0 == float(weighted[level].abs().max())
    vector = absorb.group_means(torch.as_tensor(x[:, 0].copy()), tensor_codes, 7)
    assert vector.shape == (7,)
    assert_allclose(vector.numpy(), plain[:, 0].numpy(), rtol=0, atol=0)
    # A level whose total weight is zero is treated as empty.
    zeroed = weights.copy()
    zeroed[codes == 2] = 0.0
    assert float(absorb.group_means(torch.as_tensor(x), tensor_codes, 7,
                                    torch.as_tensor(zeroed))[2].abs().max()) == 0.0


def test_zero_weight_rows_and_empty_levels_do_not_disturb_the_projection():
    x, dims, weights = _design(41, 2, False, True)
    weights[dims[0][0] == 2] = 0.0
    weights[::9] = 0.0
    positive = weights > 0
    padded = [(dims[0][0], dims[0][1] + 5), (dims[1][0], dims[1][1] + 3)]
    result = absorb.demean(torch.as_tensor(x), _tensors(padded), torch.as_tensor(weights),
                           tol=1e-13)
    expected = _dummy_residual(x[positive], [(c[positive], levels) for c, levels in dims],
                               weights[positive])
    assert _relative_error(result.values.numpy()[positive], expected) < 1e-11


def test_columns_converge_independently_and_degenerate_columns_stop():
    rng = np.random.default_rng(51)
    x, dims = _ladder(40)
    first, second = dims[0][0], dims[1][0]
    n = len(first)
    block = np.column_stack([
        x[:, 1],                                  # slow: smooth trend along the chain
        np.zeros(n),                              # exactly zero
        np.full(n, 7.0),                          # constant
        2.0 * first + np.sin(second),             # inside the fixed-effect span
        1e6 + rng.normal(size=n),                 # huge mean, unit within variation
        rng.normal(size=n),                       # ordinary
    ])
    expected = _dummy_residual(block, dims)
    result = absorb.demean(torch.as_tensor(block), _tensors(dims))
    actual = result.values.numpy()
    assert result.converged and result.max_update <= 1e-10
    assert _relative_error(actual[:, [0, 5]], expected[:, [0, 5]]) < 1e-8
    assert_allclose(actual[:, 4], expected[:, 4], rtol=0, atol=1e-7)
    assert float(np.abs(actual[:, 1]).max()) == 0.0
    assert float(np.abs(actual[:, 2]).max()) < 1e-13
    assert float(np.abs(actual[:, 3]).max()) < 1e-11 * float(np.abs(block[:, 3]).max())
    for column in (0, 4, 5):
        alone = absorb.demean(torch.as_tensor(block[:, column].copy()), _tensors(dims)).values
        assert_allclose(actual[:, column], alone.numpy(), rtol=0, atol=1e-8)


def test_conjugate_gradient_acceleration_on_a_poorly_connected_ladder():
    rungs = 40
    x, dims = _ladder(rungs)
    expected = _dummy_residual(x, dims)
    fast = absorb.demean(torch.as_tensor(x), _tensors(dims))
    slow = absorb.demean(torch.as_tensor(x), _tensors(dims), accelerate=False, max_iter=100_000)
    # A path graph with 2 * rungs nodes: conjugate gradients terminate in about `rungs`
    # sweeps, plain alternating projections need O(rungs**2 * log(1 / tol)).
    assert fast.iterations <= rungs + 5
    assert slow.iterations > 50 * fast.iterations
    assert _relative_error(fast.values.numpy(), expected) < 1e-9
    assert _relative_error(slow.values.numpy(), expected) < 1e-5


def test_long_runs_keep_full_accuracy_on_an_ill_conditioned_ladder():
    # 300 sweeps with step lengths up to ~1e5: drift of the conjugate-gradient vectors out
    # of range(M_1) costs two to three digits here unless they are projected again.
    x, dims = _ladder(300)
    expected = _dummy_residual(x, dims)
    result = absorb.demean(torch.as_tensor(x), _tensors(dims), tol=1e-13)
    assert result.iterations <= 320
    assert _relative_error(result.values.numpy(), expected) < 5e-12


@pytest.mark.parametrize("seed", [2, 5, 7])
def test_saturated_three_way_design_converges_to_zero_instead_of_diverging(seed):
    # More dummy columns than rows and full rank: every residual is exactly zero, the
    # sweep operator has eigenvalues down to ~1e-7, and the columns can only stop at
    # machine precision. Unprotected conjugate gradients blow up on these seeds.
    rng = np.random.default_rng(seed)
    n = 240
    dims = [(rng.integers(0, 45, n), 45), (rng.integers(0, 95, n), 95),
            (rng.integers(0, 150, n), 150)]
    weights = rng.uniform(0.01, 5.0, n)
    x = rng.normal(size=(n, 2)) * 1e3 + 50
    assert np.linalg.matrix_rank(np.hstack([_dummies(c, levels) for c, levels in dims])) == n
    result = absorb.demean(torch.as_tensor(x), _tensors(dims), torch.as_tensor(weights))
    assert result.converged and result.iterations < 2_000
    assert float(result.values.abs().max()) < 1e-8 * np.abs(x).max()
    effects = absorb.fixed_effect_estimates(torch.as_tensor(x), _tensors(dims),
                                            torch.as_tensor(weights))
    fitted = sum(effect.numpy()[codes] for effect, (codes, _) in zip(effects, dims))
    assert_allclose(fitted, x, rtol=0, atol=1e-8 * np.abs(x).max())


@pytest.mark.parametrize("accelerate", [True, False])
def test_nonconvergence_is_reported_as_a_kernel_error(accelerate):
    x, dims = _ladder(40)
    with pytest.raises(KernelError) as caught:
        absorb.demean(torch.as_tensor(x), _tensors(dims), max_iter=1, accelerate=accelerate)
    assert caught.value.code == "absorption_nonconvergence"
    with pytest.raises(KernelError) as caught:
        absorb.fixed_effect_estimates(torch.as_tensor(x[:, 0].copy()), _tensors(dims), max_iter=3)
    assert caught.value.code == "absorption_nonconvergence"


def test_balanced_two_way_design_is_solved_in_two_sweeps():
    x, dims, _ = _design(61, 2, True, False)
    for accelerate in (True, False):
        result = absorb.demean(torch.as_tensor(x), _tensors(dims), accelerate=accelerate, tol=1e-15)
        assert result.iterations == 2
        assert _relative_error(result.values.numpy(), _dummy_residual(x, dims)) < 1e-12


def test_invalid_inputs_raise_kernel_errors():
    values = torch.zeros(4, dtype=torch.float64)
    codes = torch.tensor([0, 1, 0, 1])
    cases = [
        (lambda: absorb.demean(values.float(), [(codes, 2)]), "invalid_values"),
        (lambda: absorb.demean(values.reshape(2, 2, 1), [(codes, 2)]), "invalid_values"),
        (lambda: absorb.demean(values, [(codes.int(), 2)]), "invalid_codes"),
        (lambda: absorb.demean(values, [(codes[:3], 2)]), "invalid_codes"),
        (lambda: absorb.demean(values, [(codes, 1)]), "invalid_codes"),
        (lambda: absorb.demean(values, [(codes - 1, 2)]), "invalid_codes"),
        (lambda: absorb.demean(values, [(codes, 2.5)]), "invalid_codes"),
        (lambda: absorb.demean(values, [(codes, 2)], torch.ones(3, dtype=torch.float64)),
         "invalid_weights"),
        (lambda: absorb.demean(values, [(codes, 2)], -torch.ones(4, dtype=torch.float64)),
         "invalid_weights"),
        (lambda: absorb.demean(values, [(codes, 2)], torch.zeros(4, dtype=torch.float64)),
         "invalid_weights"),
        (lambda: absorb.demean(values, [(codes, 2)], tol=0.0), "invalid_solver_options"),
        (lambda: absorb.demean(values, [(codes, 2)], max_iter=0), "invalid_solver_options"),
        (lambda: absorb.demean(torch.tensor([1.0, float("nan"), 0.0, 2.0], dtype=torch.float64),
                               [(codes, 2)]), "non_finite_values"),
        (lambda: absorb.demean(torch.tensor([1.0, float("inf"), 0.0, 2.0], dtype=torch.float64),
                               [(codes, 2), (codes, 2)]), "non_finite_values"),
        (lambda: absorb.singleton_mask([]), "invalid_dimensions"),
        (lambda: absorb.relabel(codes.double()), "invalid_codes"),
        (lambda: absorb.connected_components(codes, 2, codes[:2], 2), "invalid_codes"),
        (lambda: absorb.absorbed_degrees_of_freedom([(codes, 2)], (codes[:2], 2)), "invalid_codes"),
    ]
    for call, code in cases:
        with pytest.raises(KernelError) as caught:
            call()
        assert caught.value.code == code


def _backfit(x, dims, weights, sweeps):
    """Gauss-Seidel on the dummy normal equations from zero, symmetric sweep 1..D..1."""
    weights = np.ones(len(x)) if weights is None else weights
    effects = [np.zeros(levels) for _, levels in dims]
    residual = x.copy()
    order = list(range(len(dims))) + list(range(len(dims) - 2, -1, -1))
    for _ in range(sweeps):
        for d in order:
            codes, levels = dims[d]
            total = np.bincount(codes, weights, levels)
            mean = np.divide(np.bincount(codes, weights * residual, levels), total,
                             out=np.zeros(levels), where=total > 0)
            effects[d] += mean
            residual -= mean[codes]
    return effects


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_fixed_effect_estimates_reproduce_the_projection(count, weighted):
    x, dims, weights = _design(71, count, False, weighted)
    target = x[:, 1].copy()
    tensors = _tensors(dims)
    tensor_weights = None if weights is None else torch.as_tensor(weights)
    effects = absorb.fixed_effect_estimates(torch.as_tensor(target), tensors, tensor_weights,
                                            tol=1e-13)
    assert [tuple(effect.shape) for effect in effects] == [(levels,) for _, levels in dims]
    fitted = sum(effect.numpy()[codes] for effect, (codes, _) in zip(effects, dims))
    projection = target - _dummy_residual(target, dims, weights)
    assert_allclose(fitted, projection, rtol=0, atol=1e-10 * np.abs(target).max())
    # The normalization is the limit of backfitting from zero with symmetric sweeps.
    # (Plain backfitting needs thousands of sweeps on this banded design.)
    reference = _backfit(target, dims, weights, 1 if count == 1 else 8000)
    for effect, expected in zip(effects, reference):
        assert_allclose(effect.numpy(), expected, rtol=0, atol=1e-9 * np.abs(target).max())
    if count == 1:
        means = absorb.group_means(torch.as_tensor(target), tensors[0][0], dims[0][1],
                                   tensor_weights)
        assert torch.equal(effects[0], means)


def test_fixed_effect_estimates_normalize_the_second_dimension_within_components():
    rng = np.random.default_rng(81)
    # Two disconnected blocks: levels 0-5 x 0-3 and 6-9 x 4-8.
    first = np.concatenate([rng.integers(0, 6, 150), rng.integers(6, 10, 120)])
    second = np.concatenate([rng.integers(0, 4, 150), rng.integers(4, 9, 120)])
    weights = rng.uniform(0.5, 2.0, 270)
    truth = rng.normal(size=10)[first] + rng.normal(size=9)[second] + 4.0
    tensors = _tensors([(first, 10), (second, 9)])
    one, two = absorb.fixed_effect_estimates(torch.as_tensor(truth), tensors,
                                             torch.as_tensor(weights))
    assert_allclose(one.numpy()[first] + two.numpy()[second], truth, rtol=0, atol=1e-9)
    spread = two.numpy()[second] * weights
    assert abs(spread[:150].sum()) < 1e-9 and abs(spread[150:].sum()) < 1e-9
    block = torch.as_tensor(np.column_stack([truth, rng.normal(size=270)]))
    blocks = absorb.fixed_effect_estimates(block, tensors, torch.as_tensor(weights))
    assert [tuple(effect.shape) for effect in blocks] == [(10, 2), (9, 2)]
    assert_allclose(blocks[0][:, 0].numpy(), one.numpy(), rtol=0, atol=1e-9)
    assert_allclose(blocks[1][:, 0].numpy(), two.numpy(), rtol=0, atol=1e-9)


def _brute_force_singletons(dims):
    keep = np.ones(len(dims[0][0]), dtype=bool)
    while True:
        drop = np.zeros_like(keep)
        for codes, levels in dims:
            counts = np.bincount(codes[keep], minlength=levels)
            drop |= keep & (counts[codes] == 1)
        if not drop.any():
            return keep
        keep &= ~drop


def test_singleton_mask_cascades_on_a_hand_built_example():
    person = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    firm = torch.tensor([0, 1, 1, 2, 2, 2, 2, 2])
    # Row 0 is alone in firm 0; without it row 1 is alone in person 0; then row 2 is
    # alone in firm 1; then row 3 is alone in person 1. A single pass finds only row 0.
    keep = absorb.singleton_mask([(person, 4), (firm, 3)])
    assert keep.dtype == torch.bool
    assert keep.tolist() == [False, False, False, False, True, True, True, True]
    assert absorb.singleton_mask([(firm, 3)]).tolist() == [False] + [True] * 7
    assert absorb.singleton_mask([(person, 4)]).all()
    everything = absorb.singleton_mask([(torch.arange(5), 5),
                                        (torch.zeros(5, dtype=torch.int64), 1)])
    assert not everything.any()


@pytest.mark.parametrize("seed", range(6))
def test_singleton_mask_matches_brute_force_iteration(seed):
    rng = np.random.default_rng(seed)
    n = 400
    dims = [(rng.integers(0, 260, n), 260), (rng.integers(0, 150, n), 150)]
    if seed % 2:
        dims.append((rng.integers(0, 90, n), 95))
    expected = _brute_force_singletons(dims)
    actual = absorb.singleton_mask(_tensors(dims))
    assert_array_equal(actual.numpy(), expected)
    assert 0 < expected.sum() < n


def test_singleton_mask_long_cascade_on_a_chain():
    rows = np.arange(2001)
    dims = [(rows // 2, 1001), ((rows + 1) // 2, 1001)]
    # Both ends are singletons and each removal exposes the next link: 1000 rounds.
    assert not absorb.singleton_mask(_tensors(dims)).any()
    closed = [(np.append(rows // 2, 1000), 1001), (np.append((rows + 1) // 2, 0), 1001)]
    assert absorb.singleton_mask(_tensors(closed)).all()


def test_relabel_is_dense_and_order_preserving():
    codes = torch.tensor([40, -3, 40, 7, 10**15, -3, 7])
    dense, levels = absorb.relabel(codes)
    assert levels == 4 and dense.tolist() == [2, 0, 2, 1, 3, 0, 1]
    assert dense.dtype == torch.int64
    rng = np.random.default_rng(9)
    for values in (rng.integers(-50, 400, 300), rng.integers(-10**12, 10**12, 300)):
        dense, levels = absorb.relabel(torch.as_tensor(values))
        unique, inverse = np.unique(values, return_inverse=True)
        assert levels == len(unique)
        assert_array_equal(dense.numpy(), inverse)
    empty, levels = absorb.relabel(torch.zeros(0, dtype=torch.int64))
    assert levels == 0 and len(empty) == 0
    assert codes.tolist() == [40, -3, 40, 7, 10**15, -3, 7]


def _scipy_components(first, n_first, second, n_second):
    """Components among observed levels: isolated (unobserved) nodes are removed."""
    graph = coo_matrix((np.ones(len(first)), (first, n_first + second)),
                       shape=(n_first + n_second, n_first + n_second))
    count, _ = scipy_components(graph, directed=False)
    empty = (n_first - len(np.unique(first))) + (n_second - len(np.unique(second)))
    return count - empty


def test_connected_components_on_hand_built_graphs():
    def count(first, n_first, second, n_second):
        return absorb.connected_components(torch.tensor(first), n_first, torch.tensor(second),
                                           n_second)

    # a0-b0, a0-b1, a1-b1 | a2-b2 | a3-b3, a4-b3
    assert count([0, 0, 1, 2, 3, 4], 5, [0, 1, 1, 2, 3, 3], 4) == 3
    assert count([0, 0, 1, 2, 3, 4], 5, [0, 1, 1, 2, 3, 3], 4) == count(
        [0, 1, 1, 2, 3, 3], 4, [0, 0, 1, 2, 3, 4], 5)
    # Linking a2 to b1 merges the first two groups; duplicate rows change nothing.
    assert count([0, 0, 1, 2, 2, 3, 4, 4], 5, [0, 1, 1, 2, 1, 3, 3, 3], 4) == 2
    # Levels without observations are not components.
    assert count([0, 0, 1, 2, 3, 4], 9, [0, 1, 1, 2, 3, 3], 12) == 3
    assert count([3], 5, [2], 4) == 1
    assert count([0, 1, 2], 3, [0, 0, 0], 1) == 1
    assert count([0, 1, 2], 3, [0, 1, 2], 3) == 3
    empty = torch.zeros(0, dtype=torch.int64)
    assert absorb.connected_components(empty, 3, empty, 2) == 0


@pytest.mark.parametrize("seed", range(8))
def test_connected_components_match_scipy(seed):
    rng = np.random.default_rng(seed)
    n_first, n_second = [(300, 200), (150, 400), (2000, 30), (40, 1500)][seed % 4]
    n = int((n_first + n_second) * (0.45 + 0.25 * (seed // 4)))
    first, second = rng.integers(0, n_first, n), rng.integers(0, n_second, n)
    expected = _scipy_components(first, n_first, second, n_second)
    actual = absorb.connected_components(torch.as_tensor(first), n_first,
                                         torch.as_tensor(second), n_second)
    assert actual == expected
    assert expected > 1 or seed % 4 >= 2


def test_connected_components_on_long_shuffled_chains_and_large_graphs():
    rng = np.random.default_rng(13)
    rows = np.arange(40_001)
    # One path through 40 002 nodes with shuffled labels: needs many hooking rounds.
    first, second = rng.permutation(20_001)[rows // 2], rng.permutation(20_001)[(rows + 1) // 2]
    assert absorb.connected_components(torch.as_tensor(first), 20_001,
                                       torch.as_tensor(second), 20_001) == 1
    cut = np.delete(rows, [9_000, 25_000])
    assert absorb.connected_components(torch.as_tensor(first[cut]), 20_001,
                                       torch.as_tensor(second[cut]), 20_001) == 3
    n = 300_000
    first, second = rng.integers(0, 180_000, n), rng.integers(0, 150_000, n)
    assert absorb.connected_components(
        torch.as_tensor(first), 180_000, torch.as_tensor(second), 150_000
    ) == _scipy_components(first, 180_000, second, 150_000)


@pytest.mark.parametrize("seed", range(5))
def test_two_way_degrees_of_freedom_equal_the_rank_of_the_dummy_matrix(seed):
    rng = np.random.default_rng(200 + seed)
    n = 110
    dims = [(rng.integers(0, 70, n), 75), (rng.integers(0, 50, n), 50)]
    rank = np.linalg.matrix_rank(np.hstack([_dummies(codes, levels) for codes, levels in dims]))
    result = absorb.absorbed_degrees_of_freedom(_tensors(dims))
    observed = [len(np.unique(codes)) for codes, _ in dims]
    components = absorb.connected_components(*_tensors(dims)[0], *_tensors(dims)[1])
    assert components > 1
    assert result.levels == observed and result.nested == [False, False]
    assert result.redundant == [0, components]
    assert result.total == observed[0] + observed[1] - components == rank
    single = absorb.absorbed_degrees_of_freedom(_tensors(dims[:1]))
    assert (single.levels, single.redundant, single.total) == ([observed[0]], [0], observed[0])
    none = absorb.absorbed_degrees_of_freedom([])
    assert (none.levels, none.redundant, none.nested, none.total) == ([], [], [], 0)


def test_three_way_degrees_of_freedom_are_a_conservative_bound():
    rng = np.random.default_rng(301)
    n = 500
    connected = [(rng.integers(0, 40, n), 40), (rng.integers(0, 12, n), 12),
                 (rng.integers(0, 5, n), 5)]
    rank = np.linalg.matrix_rank(np.hstack([_dummies(c, levels) for c, levels in connected]))
    for pairwise in (True, False):
        result = absorb.absorbed_degrees_of_freedom(_tensors(connected), pairwise=pairwise)
        assert result.redundant == [0, 1, 1] and result.total == rank == 40 + 12 + 5 - 2
    # Two blocks that share no level in any dimension: every pair has two components.
    half = n // 2
    blocks = [(np.concatenate([rng.integers(0, 20, half), rng.integers(20, 40, half)]), 40),
              (np.concatenate([rng.integers(0, 6, half), rng.integers(6, 12, half)]), 12),
              (np.concatenate([rng.integers(0, 2, half), rng.integers(2, 5, half)]), 5)]
    rank = np.linalg.matrix_rank(np.hstack([_dummies(c, levels) for c, levels in blocks]))
    pairwise = absorb.absorbed_degrees_of_freedom(_tensors(blocks))
    firstpair = absorb.absorbed_degrees_of_freedom(_tensors(blocks), pairwise=False)
    assert pairwise.redundant == [0, 2, 2] and pairwise.total == rank
    assert firstpair.redundant == [0, 2, 1] and firstpair.total == rank + 1
    # A third dimension tied to the second but not to the first: pairwise takes the maximum.
    tied = [connected[0], blocks[1], blocks[2]]
    rank = np.linalg.matrix_rank(np.hstack([_dummies(c, levels) for c, levels in tied]))
    result = absorb.absorbed_degrees_of_freedom(_tensors(tied))
    assert result.redundant == [0, 1, 2] and result.total >= rank


def test_dimensions_nested_in_the_cluster_cost_no_degrees_of_freedom():
    rng = np.random.default_rng(401)
    person = np.repeat(np.arange(30), 6)
    year = np.tile(np.arange(6), 30)
    keep = rng.random(180) < 0.8
    person, year = person[keep], year[keep]
    people = len(np.unique(person))
    dims = _tensors([(person, 30), (year, 6)])
    by_person = absorb.absorbed_degrees_of_freedom(dims, (torch.as_tensor(person), 30))
    assert by_person.nested == [True, False]
    assert by_person.levels == [people, 6] and by_person.redundant == [people, 0]
    assert by_person.total == 6
    by_group = absorb.absorbed_degrees_of_freedom(dims, (torch.as_tensor(person // 3), 10))
    assert by_group.nested == [True, False] and by_group.total == 6
    # The year effects of the reversed order are second among the non-nested ones.
    reverse = absorb.absorbed_degrees_of_freedom(dims[::-1], (torch.as_tensor(person), 30))
    assert reverse.nested == [False, True] and reverse.redundant == [0, people]
    # Clusters that cut through persons and years: nothing is nested, the count is the rank.
    crossing = absorb.absorbed_degrees_of_freedom(dims, (torch.as_tensor((person + year) % 4), 4))
    assert crossing.nested == [False, False] and crossing.redundant == [0, 1]
    assert crossing.total == people + 6 - 1
    only = absorb.absorbed_degrees_of_freedom(dims[:1], (torch.as_tensor(person), 30))
    assert only.nested == [True] and only.total == 0
    unclustered = absorb.absorbed_degrees_of_freedom(dims)
    assert unclustered.nested == [False, False] and unclustered.total == people + 5


def test_moderately_large_two_way_problem_is_orthogonal_to_both_dummy_sets():
    generator = torch.Generator().manual_seed(17)
    n = 200_000
    first = torch.randint(0, 20_000, (n,), generator=generator)
    second = torch.randint(0, 300, (n,), generator=generator)
    values = torch.randn(n, 3, dtype=torch.float64, generator=generator) + first[:, None] * 1e-3
    weights = torch.rand(n, dtype=torch.float64, generator=generator) + 0.1
    result = absorb.demean(values, [(first, 20_000), (second, 300)], weights)
    assert result.converged and result.iterations < 30
    for codes, levels in ((first, 20_000), (second, 300)):
        assert float(absorb.group_means(result.values, codes, levels, weights).abs().max()) < 1e-9
