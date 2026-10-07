"""Adversarial review of the fixed-effect absorption kernel.

Independent oracles: explicit dummy-variable least squares (numpy lstsq), brute-force
loops for singletons and backfitting, scipy.sparse.csgraph for mobility groups, and the
numerical rank of the dummy matrix for degrees of freedom. Attacks: extreme and mixed
magnitudes, in-span columns on poorly connected designs, rank-deficient (duplicate,
nested) dimensions, gaps and empty levels, single-element groups, k = 1, m = 1,
n barely above the number of dummies, shapes and dtypes, aliasing of the inputs,
determinism, adversarial graph labelings, and a million-row timing with a memory bound.
"""

import ast
import inspect
import resource
import sys
import time

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


def _design_matrix(dims):
    return np.hstack([_dummies(codes, levels) for codes, levels in dims])


def _lstsq_residual(x, dims, weights=None):
    design = _design_matrix(dims)
    root = np.ones(len(x)) if weights is None else np.sqrt(weights)
    target = x * (root[:, None] if x.ndim == 2 else root)
    coefficients = np.linalg.lstsq(design * root[:, None], target, rcond=None)[0]
    return x - design @ coefficients


def _relative_error(actual, expected):
    actual, expected = np.atleast_2d(actual.T).T, np.atleast_2d(expected.T).T
    scale = np.maximum(np.linalg.norm(expected, axis=0), np.finfo(float).tiny)
    return float((np.linalg.norm(actual - expected, axis=0) / scale).max())


def _tensors(dims):
    return [
        (torch.as_tensor(np.asarray(codes), dtype=torch.int64), int(levels))
        for codes, levels in dims
    ]


def _ladder(rungs, repeats=3):
    first = np.repeat(np.concatenate([np.arange(rungs), np.arange(rungs - 1)]), repeats)
    second = np.repeat(np.concatenate([np.arange(rungs), np.arange(1, rungs)]), repeats)
    return first, second


def _scipy_components(first, n_first, second, n_second):
    graph = coo_matrix(
        (np.ones(len(first)), (first, n_first + second)),
        shape=(n_first + n_second, n_first + n_second),
    )
    count, _ = scipy_components(graph, directed=False)
    empty = (n_first - len(np.unique(first))) + (n_second - len(np.unique(second)))
    return count - empty


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
            mean = np.divide(
                np.bincount(codes, weights * residual, levels),
                total,
                out=np.zeros(levels),
                where=total > 0,
            )
            effects[d] += mean
            residual -= mean[codes]
    return effects


# --------------------------------------------------------------------------- rules


def test_runtime_imports_and_no_autograd():
    source = inspect.getsource(absorb)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0] if node.level == 0 else "openecon")
    assert imported <= {
        "torch",
        "math",
        "numbers",
        "operator",
        "dataclasses",
        "__future__",
        "openecon",
    }, imported
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {
        "requires_grad",
        "requires_grad_",
        "autograd",
        "func",
        "backward",
        "grad",
        "enable_grad",
    }, attributes
    assert max(len(line) for line in source.splitlines()) <= 100


# --------------------------------------------------------------------------- magnitudes


@pytest.mark.parametrize("scale", [1e-170, 1e-120, 1e-40, 1e40, 1e120, 1e200])
def test_extreme_column_magnitudes_are_fully_absorbed(scale):
    # Squares of columns below ~1e-154 underflow to zero and above ~1e154 overflow: the
    # recursion must not mistake either for convergence (the original code returned the
    # one-way within transform after a single sweep at 1e-170).
    rng = np.random.default_rng(1)
    n = 600
    first = rng.integers(0, 80, n)
    second = (first // 3 + rng.integers(0, 2, n)) % 30
    dims = [(first, 80), (second, 30)]
    x = rng.normal(size=(n, 3))
    expected = _lstsq_residual(x, dims)
    result = absorb.demean(torch.as_tensor(x * scale), _tensors(dims))
    assert result.iterations > 1 and np.isfinite(result.values.numpy()).all()
    assert _relative_error(result.values.numpy() / scale, expected) < 1e-10
    effects = absorb.fixed_effect_estimates(torch.as_tensor(x[:, 0] * scale), _tensors(dims))
    fitted = sum(effect.numpy()[codes] for effect, (codes, _) in zip(effects, dims))
    assert_allclose(fitted / scale, x[:, 0] - expected[:, 0], rtol=0, atol=1e-10)


def test_mixed_magnitude_columns_in_one_block_converge_independently():
    rng = np.random.default_rng(2)
    n = 500
    first, second = rng.integers(0, 60, n), rng.integers(0, 25, n)
    dims = [(first, 60), (second, 25)]
    base = rng.normal(size=(n, 4))
    scales = np.array([1e-165, 1.0, 1e165, 1e-3])
    expected = _lstsq_residual(base, dims)
    result = absorb.demean(torch.as_tensor(base * scales), _tensors(dims), tol=1e-13)
    assert _relative_error(result.values.numpy() / scales, expected) < 1e-11
    assert result.max_update <= 1e-13


def test_power_of_two_rescaling_is_bit_exact():
    rng = np.random.default_rng(3)
    n = 400
    first, second = rng.integers(0, 50, n), rng.integers(0, 20, n)
    dims = _tensors([(first, 50), (second, 20)])
    x = torch.as_tensor(rng.normal(size=(n, 2)))
    weights = torch.as_tensor(rng.uniform(0.5, 2.0, n))
    reference = absorb.demean(x, dims, weights)
    for power in (-700, -300, -17, 9, 300, 700):
        factor = 2.0**power
        scaled = absorb.demean(x * factor, dims, weights)
        assert scaled.iterations == reference.iterations
        assert torch.equal(scaled.values, reference.values * factor)


def test_huge_fixed_effects_with_small_within_variation_reach_the_rounding_floor():
    # The within part is 1e-8 of the level effects; the attainable accuracy is bounded by
    # eps * ||M_1 x||, so the column must stop at the machine-precision exit rather than
    # spin until max_iter, and the result must be as accurate as the explicit regression.
    rng = np.random.default_rng(4)
    n = 800
    first, second = rng.integers(0, 100, n), rng.integers(0, 40, n)
    dims = [(first, 100), (second, 40)]
    effect, noise = 1e8 * rng.normal(size=40)[second], rng.normal(size=n)
    x = effect + noise
    # The exact residual of the floating-point x: the level effects project to nothing,
    # and the rounding of x itself (~1e-8) is a well-scaled problem of its own.
    truth = _lstsq_residual(noise, dims) + _lstsq_residual(x - effect - noise, dims)
    result = absorb.demean(torch.as_tensor(x), _tensors(dims))
    assert result.iterations < 200
    error = np.abs(result.values.numpy() - truth).max()
    assert error < 100 * np.finfo(float).eps * np.abs(x).max()
    assert error < 2 * np.abs(_lstsq_residual(x, dims) - truth).max()


# --------------------------------------------------------------------------- designs


@pytest.mark.parametrize("seed", range(12))
def test_random_unbalanced_designs_with_gaps_match_lstsq(seed):
    rng = np.random.default_rng(100 + seed)
    count = 2 + seed % 3
    n = int(rng.integers(60, 400))
    dims = []
    for d in range(count):
        levels = int(rng.integers(2, 10 + 40 * (d == 0)))
        used = rng.choice(levels, size=max(1, levels // 2), replace=False)
        codes = used[rng.integers(0, len(used), n)]
        # Gaps: the declared level count exceeds the largest code.
        dims.append((codes, levels + int(rng.integers(0, 5))))
    rank = np.linalg.matrix_rank(_design_matrix(dims))
    if rank >= n:
        pytest.skip("saturated draw")
    weights = rng.uniform(0.05, 4.0, n) if seed % 2 else None
    x = (
        rng.normal(size=(n, 2)) * np.array([1.0, 50.0])
        + rng.normal(size=dims[0][1])[dims[0][0]][:, None]
    )
    expected = _lstsq_residual(x, dims, weights)
    tensor_weights = None if weights is None else torch.as_tensor(weights)
    result = absorb.demean(torch.as_tensor(x), _tensors(dims), tensor_weights, tol=1e-13)
    assert _relative_error(result.values.numpy(), expected) < 1e-10
    plain = absorb.demean(
        torch.as_tensor(x), _tensors(dims), tensor_weights, accelerate=False, max_iter=200_000
    )
    assert _relative_error(plain.values.numpy(), expected) < 1e-7


def test_n_barely_above_the_number_of_dummies():
    # 10 + 6 - 1 = 15 identified coefficients with 17 rows: two residual degrees of freedom.
    rng = np.random.default_rng(5)
    first = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 9, 0])
    second = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 2, 3, 4, 5])
    dims = [(first, 10), (second, 6)]
    assert np.linalg.matrix_rank(_design_matrix(dims)) == 15
    x = rng.normal(size=(17, 3))
    expected = _lstsq_residual(x, dims)
    result = absorb.demean(torch.as_tensor(x), _tensors(dims), tol=1e-13)
    assert _relative_error(result.values.numpy(), expected) < 1e-9
    assert absorb.absorbed_degrees_of_freedom(_tensors(dims)).total == 15


def test_single_level_dimensions_and_single_element_groups():
    rng = np.random.default_rng(6)
    n = 50
    x = rng.normal(size=(n, 2))
    ones = np.zeros(n, dtype=np.int64)
    # k = 1: one level is the grand mean; twice is still the grand mean.
    one = absorb.demean(torch.as_tensor(x), _tensors([(ones, 1)]))
    assert_allclose(one.values.numpy(), x - x.mean(axis=0), rtol=0, atol=1e-14)
    two = absorb.demean(torch.as_tensor(x), _tensors([(ones, 1), (ones, 1)]))
    assert two.iterations == 1
    assert_allclose(two.values.numpy(), x - x.mean(axis=0), rtol=0, atol=1e-14)
    # Every observation its own level: the dummies fit everything, residuals vanish.
    each = absorb.demean(torch.as_tensor(x), _tensors([(np.arange(n), n), (ones, 1)]))
    assert float(each.values.abs().max()) == 0.0 and each.iterations == 1
    # Mixed: half the levels are single observations.
    first = np.concatenate([np.arange(25), np.repeat(np.arange(25, 30), 5)])
    second = rng.integers(0, 4, n)
    dims = [(first, 30), (second, 4)]
    expected = _lstsq_residual(x, dims)
    result = absorb.demean(torch.as_tensor(x), _tensors(dims), tol=1e-13)
    assert _relative_error(result.values.numpy(), expected) < 1e-10
    assert float(np.abs(result.values.numpy()[:25]).max()) < 1e-13


def test_duplicate_and_nested_dimensions_are_rank_deficient_but_exact():
    rng = np.random.default_rng(7)
    n = 500
    firm = rng.integers(0, 80, n)
    industry = firm // 10
    x = rng.normal(size=(n, 2))
    expected = _lstsq_residual(x, [(firm, 80)])
    for dims in (
        [(firm, 80), (firm, 80)],
        [(firm, 80), (industry, 8)],
        [(industry, 8), (firm, 80)],
        [(firm, 80), (industry, 8), (firm, 80)],
    ):
        result = absorb.demean(torch.as_tensor(x), _tensors(dims), tol=1e-13)
        assert result.iterations <= 3
        assert _relative_error(result.values.numpy(), expected) < 1e-13
    # Degrees of freedom: the nested dimension is fully redundant (one component).
    nested = absorb.absorbed_degrees_of_freedom(_tensors([(firm, 80), (industry, 8)]))
    assert nested.redundant == [0, 8] and nested.total == 80
    duplicate = absorb.absorbed_degrees_of_freedom(_tensors([(firm, 80), (firm, 80)]))
    assert duplicate.redundant == [0, 80] and duplicate.total == 80


def test_zero_weight_levels_in_later_dimensions_do_not_disturb_positive_rows():
    rng = np.random.default_rng(8)
    n = 700
    dims = [(rng.integers(0, 50, n), 50), (rng.integers(0, 20, n), 22), (rng.integers(0, 6, n), 6)]
    weights = rng.uniform(0.2, 2.0, n)
    weights[dims[1][0] == 3] = 0.0
    weights[dims[2][0] == 0] = 0.0
    positive = weights > 0
    x = rng.normal(size=(n, 2))
    expected = _lstsq_residual(
        x[positive], [(c[positive], levels) for c, levels in dims], weights[positive]
    )
    result = absorb.demean(torch.as_tensor(x), _tensors(dims), torch.as_tensor(weights), tol=1e-13)
    assert _relative_error(result.values.numpy()[positive], expected) < 1e-11
    assert np.isfinite(result.values.numpy()).all()


def test_in_span_columns_on_poorly_connected_designs_stop_at_machine_precision():
    rng = np.random.default_rng(9)
    for rungs in (300, 1000):
        first, second = _ladder(rungs)
        n = len(first)
        span = rng.normal(size=rungs)[first] * 10 + rng.normal(size=rungs)[second]
        block = np.column_stack([span, span * 1e6, rng.normal(size=n)])
        weights = torch.as_tensor(rng.uniform(1e-3, 1e3, n))
        result = absorb.demean(
            torch.as_tensor(block), _tensors([(first, rungs), (second, rungs)]), weights, tol=1e-13
        )
        assert result.iterations <= rungs + 100
        residual = result.values.numpy()
        assert np.abs(residual[:, 0]).max() < 1e-11 * np.abs(span).max()
        assert np.abs(residual[:, 1]).max() < 1e-11 * np.abs(span).max() * 1e6
        expected = _lstsq_residual(block[:, 2], [(first, rungs), (second, rungs)], weights.numpy())
        assert _relative_error(residual[:, 2], expected) < 1e-10


def test_error_tracks_tol_on_a_sparse_three_way_design():
    rng = np.random.default_rng(10)
    n = 3000
    first = rng.integers(0, 500, n)
    dims = [
        (first, 500),
        ((first // 3 + rng.integers(0, 2, n)) % 170, 170),
        ((first // 7 + rng.integers(0, 2, n)) % 72, 72),
    ]
    x = rng.normal(size=(n, 2))
    expected = _lstsq_residual(x, dims)
    previous = np.inf
    for tol in (1e-6, 1e-9, 1e-12):
        result = absorb.demean(torch.as_tensor(x), _tensors(dims), tol=tol)
        error = _relative_error(result.values.numpy(), expected)
        assert error < 100 * tol and error <= previous
        previous = max(error, 1e-14)


# --------------------------------------------------------------------------- API


def test_shapes_dtypes_and_option_types():
    rng = np.random.default_rng(11)
    n = 120
    dims = _tensors([(rng.integers(0, 12, n), 12), (rng.integers(0, 5, n), 5)])
    x = rng.normal(size=(n, 1))
    column = absorb.demean(torch.as_tensor(x), dims)
    vector = absorb.demean(torch.as_tensor(x[:, 0].copy()), dims)
    assert column.values.shape == (n, 1) and vector.values.shape == (n,)
    assert torch.equal(column.values[:, 0], vector.values)
    effects = absorb.fixed_effect_estimates(torch.as_tensor(x), dims)
    assert [tuple(effect.shape) for effect in effects] == [(12, 1), (5, 1)]
    means = absorb.group_means(torch.as_tensor(x), dims[0][0], 12)
    assert means.shape == (12, 1) and means.dtype == torch.float64
    # Accepted option scalars.
    accepted = absorb.demean(torch.as_tensor(x), dims, tol=np.float32(1e-8), max_iter=np.int64(50))
    assert accepted.converged
    rejected = [
        (lambda: absorb.demean(torch.as_tensor(x), dims, max_iter=True), "invalid_solver_options"),
        (lambda: absorb.demean(torch.as_tensor(x), dims, max_iter=2.0), "invalid_solver_options"),
        (
            lambda: absorb.demean(torch.as_tensor(x), dims, tol=float("nan")),
            "invalid_solver_options",
        ),
        (lambda: absorb.demean(torch.as_tensor(x), dims, tol="1e-8"), "invalid_solver_options"),
        (lambda: absorb.demean(torch.as_tensor(x), [(dims[0][0], True)]), "invalid_codes"),
        (lambda: absorb.demean(torch.as_tensor(x), [(dims[0][0][:, None], 12)]), "invalid_codes"),
        (
            lambda: absorb.demean(torch.as_tensor(x), dims, torch.ones(n, 1, dtype=torch.float64)),
            "invalid_weights",
        ),
        (
            lambda: absorb.demean(
                torch.as_tensor(x),
                dims,
                torch.tensor([float("inf")] + [1.0] * (n - 1), dtype=torch.float64),
            ),
            "invalid_weights",
        ),
        (lambda: absorb.demean(torch.as_tensor(x).to(torch.int64), dims), "invalid_values"),
        (lambda: absorb.demean(x, dims), "invalid_values"),
        (lambda: absorb.group_means(torch.as_tensor(x), dims[0][0], 11), "invalid_codes"),
        (
            lambda: absorb.fixed_effect_estimates(torch.as_tensor(x), dims, max_iter=0),
            "invalid_solver_options",
        ),
    ]
    for call, code in rejected:
        with pytest.raises(KernelError) as caught:
            call()
        assert caught.value.code == code


def test_empty_inputs():
    empty = torch.zeros(0, dtype=torch.int64)
    values = torch.zeros((0, 2), dtype=torch.float64)
    result = absorb.demean(values, [(empty, 3), (empty, 2)], torch.zeros(0, dtype=torch.float64))
    assert result.values.shape == (0, 2) and result.converged
    effects = absorb.fixed_effect_estimates(values, [(empty, 3), (empty, 2)])
    assert [tuple(effect.shape) for effect in effects] == [(3, 2), (2, 2)]
    assert float(absorb.group_means(values, empty, 3).abs().sum()) == 0.0
    assert absorb.singleton_mask([(empty, 3)]).shape == (0,)
    assert absorb.connected_components(empty, 0, empty, 0) == 0
    none = absorb.absorbed_degrees_of_freedom([(empty, 3), (empty, 2)], (empty, 1))
    assert none.levels == [0, 0] and none.total == 0
    wide = absorb.demean(
        torch.zeros((5, 0), dtype=torch.float64), [(torch.zeros(5, dtype=torch.int64), 1)]
    )
    assert wide.values.shape == (5, 0)


def test_views_and_strided_inputs_are_read_only_and_equivalent():
    rng = np.random.default_rng(12)
    n = 300
    first, second = rng.integers(0, 30, n), rng.integers(0, 9, n)
    dims = _tensors([(first, 30), (second, 9)])
    full = torch.as_tensor(rng.normal(size=(2 * n, 3)))
    weights_full = torch.as_tensor(rng.uniform(0.5, 1.5, 2 * n))
    values, weights = full[::2], weights_full[::2]
    assert not values.is_contiguous() and not weights.is_contiguous()
    saved = full.clone(), weights_full.clone()
    reference = absorb.demean(values.contiguous(), dims, weights.contiguous(), tol=1e-13)
    strided = absorb.demean(values, dims, weights, tol=1e-13)
    assert torch.equal(strided.values, reference.values)
    column = absorb.demean(values[:, 1], dims, weights, tol=1e-13)
    # Alone, the column stops after its own number of sweeps: equal to tolerance, not bits.
    assert_allclose(column.values.numpy(), reference.values[:, 1].numpy(), rtol=0, atol=1e-11)
    absorb.fixed_effect_estimates(values, dims, weights)
    absorb.group_means(values, dims[0][0], 30, weights)
    assert torch.equal(full, saved[0]) and torch.equal(weights_full, saved[1])
    tracked = values.clone().requires_grad_(True)
    result = absorb.demean(tracked, dims, weights)
    assert not result.values.requires_grad and result.values.grad_fn is None
    assert not tracked.grad_fn and tracked.grad is None
    assert_allclose(result.values.numpy(), reference.values.numpy(), rtol=0, atol=1e-9)


def test_results_are_deterministic_across_calls_and_thread_counts():
    rng = np.random.default_rng(13)
    n = 20_000
    dims = _tensors([(rng.integers(0, 2000, n), 2000), (rng.integers(0, 40, n), 40)])
    x = torch.as_tensor(rng.normal(size=(n, 3)))
    weights = torch.as_tensor(rng.uniform(0.1, 3.0, n))
    first = absorb.demean(x, dims, weights)
    second = absorb.demean(x, dims, weights)
    assert torch.equal(first.values, second.values) and first.iterations == second.iterations
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        single = absorb.demean(x, dims, weights)
    finally:
        torch.set_num_threads(threads)
    assert _relative_error(single.values.numpy(), first.values.numpy()) < 1e-13
    effects = [absorb.fixed_effect_estimates(x, dims, weights) for _ in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(*effects))


# --------------------------------------------------------------------------- fixed effects


def test_fixed_effect_estimates_identities_on_disconnected_weighted_designs():
    rng = np.random.default_rng(14)
    # Three components of different sizes, empty levels in both dimensions, weights 1e-3..1e3.
    first = np.concatenate(
        [rng.integers(0, 8, 200), rng.integers(10, 13, 60), rng.integers(14, 20, 90)]
    )
    second = np.concatenate(
        [rng.integers(0, 5, 200), rng.integers(6, 8, 60), rng.integers(9, 15, 90)]
    )
    n = len(first)
    weights = np.exp(rng.uniform(-7, 7, n))
    dims = [(first, 21), (second, 16)]
    x = rng.normal(size=20)[np.minimum(first, 19)] + rng.normal(size=n) + 3.0
    tensors, tensor_weights = _tensors(dims), torch.as_tensor(weights)
    one, two = absorb.fixed_effect_estimates(torch.as_tensor(x), tensors, tensor_weights, tol=1e-13)
    projection = x - _lstsq_residual(x, dims, weights)
    assert_allclose(one.numpy()[first] + two.numpy()[second], projection, rtol=0, atol=1e-10)
    # Empty levels carry zero.
    assert (
        one.numpy()[[8, 9, 13, 20]].tolist() == [0.0] * 4
        and two.numpy()[[5, 8, 15]].tolist() == [0.0] * 3
    )
    # The second dimension is centred (weighted) within every component.
    for rows in (slice(0, 200), slice(200, 260), slice(260, n)):
        assert abs((two.numpy()[second][rows] * weights[rows]).sum()) < 1e-9 * weights[rows].sum()
    # The limit of backfitting is the same decomposition.
    reference = _backfit(x, dims, weights, 400)
    assert_allclose(one.numpy(), reference[0], rtol=0, atol=1e-9)
    assert_allclose(two.numpy(), reference[1], rtol=0, atol=1e-9)


def test_three_way_fixed_effects_match_backfitting_limit():
    rng = np.random.default_rng(15)
    n = 600
    dims = [(rng.integers(0, 12, n), 12), (rng.integers(0, 7, n), 7), (rng.integers(0, 4, n), 4)]
    x = rng.normal(size=(n, 2)) + np.array([1.0, -4.0])
    weights = rng.uniform(0.3, 3.0, n)
    effects = absorb.fixed_effect_estimates(
        torch.as_tensor(x), _tensors(dims), torch.as_tensor(weights), tol=1e-13
    )
    for column in range(2):
        reference = _backfit(x[:, column], dims, weights, 3000)
        for effect, expected in zip(effects, reference):
            assert_allclose(effect[:, column].numpy(), expected, rtol=0, atol=1e-9)
    fitted = sum(effect.numpy()[codes] for effect, (codes, _) in zip(effects, dims))
    assert_allclose(fitted, x - _lstsq_residual(x, dims, weights), rtol=0, atol=1e-10)
    zero = absorb.fixed_effect_estimates(torch.zeros(n, dtype=torch.float64), _tensors(dims))
    assert all(float(effect.abs().max()) == 0.0 for effect in zero)


# --------------------------------------------------------------------------- singletons


@pytest.mark.parametrize("seed", range(10))
def test_singleton_mask_matches_brute_force_on_sparse_many_dimension_designs(seed):
    rng = np.random.default_rng(300 + seed)
    n = int(rng.integers(5, 260))
    count = 1 + seed % 4
    dims = []
    for d in range(count):
        levels = int(rng.integers(1, n + 3))
        dims.append((rng.integers(0, levels, n), levels + int(rng.integers(0, 3))))
    expected = _brute_force_singletons(dims)
    actual = absorb.singleton_mask(_tensors(dims))
    assert actual.dtype == torch.bool and actual.shape == (n,)
    assert_array_equal(actual.numpy(), expected)


def test_singleton_mask_edge_shapes():
    assert absorb.singleton_mask([(torch.tensor([0]), 1)]).tolist() == [False]
    assert absorb.singleton_mask([(torch.tensor([0, 0]), 1)]).tolist() == [True, True]
    assert absorb.singleton_mask(
        [(torch.tensor([0, 0]), 1), (torch.tensor([0, 1]), 2)]
    ).tolist() == [False, False]
    # A long two-dimensional cascade from both ends of a chain with a knot in the middle.
    rows = np.arange(4001)
    first, second = rows // 2, (rows + 1) // 2
    knot = [(np.append(first, [1000, 1000]), 2001), (np.append(second, [1000, 1001]), 2001)]
    expected = _brute_force_singletons(knot)
    assert_array_equal(absorb.singleton_mask(_tensors(knot)).numpy(), expected)
    assert expected.sum() == 4


# --------------------------------------------------------------------------- graphs


def _edge_design(edges, nodes):
    """Bipartite design whose b-side graph is the given edge list: one a-level per edge."""
    count = len(edges)
    first = np.repeat(np.arange(count), 2)
    second = np.asarray(edges).reshape(-1)
    return first, count, second, nodes


@pytest.mark.parametrize(
    "structure",
    [
        "path_decreasing",
        "path_increasing",
        "path_shuffled",
        "path_alternating",
        "binary_tree",
        "binary_tree_reversed",
        "grid",
        "star_max_centre",
        "sparse_random",
        "two_cliques",
    ],
)
def test_connected_components_on_adversarial_labelings(structure):
    rng = np.random.default_rng(16)
    nodes = 60_000
    if structure.startswith("path"):
        chain = np.column_stack([np.arange(nodes - 1), np.arange(1, nodes)])
        label = {
            "path_decreasing": nodes - 1 - np.arange(nodes),
            "path_increasing": np.arange(nodes),
            "path_shuffled": rng.permutation(nodes),
        }.get(structure)
        if label is None:
            label = np.empty(nodes, dtype=np.int64)
            label[0::2] = np.arange((nodes + 1) // 2)
            label[1::2] = np.arange(nodes - 1, (nodes + 1) // 2 - 1, -1)[: nodes // 2]
        edges = label[chain]
        expected = 1
    elif structure.startswith("binary_tree"):
        child = np.arange(1, nodes)
        edges = np.column_stack([child, (child - 1) // 2])
        if structure.endswith("reversed"):
            edges = nodes - 1 - edges
        expected = 1
    elif structure == "grid":
        side = int(np.sqrt(nodes))
        grid = np.arange(side * side).reshape(side, side)
        edges = np.vstack(
            [
                np.column_stack([grid[:, :-1].ravel(), grid[:, 1:].ravel()]),
                np.column_stack([grid[:-1].ravel(), grid[1:].ravel()]),
            ]
        )
        expected = 1
    elif structure == "star_max_centre":
        edges = np.column_stack([np.full(nodes - 1, nodes - 1), np.arange(nodes - 1)])
        expected = 1
    elif structure == "two_cliques":
        half = 300
        pairs = np.array([(i, j) for i in range(half) for j in range(i + 1, half)])
        edges = np.vstack([pairs, pairs + half])
        nodes = 2 * half
        expected = 2
    else:
        edges = rng.integers(0, nodes, (nodes, 2))
        expected = None
    first, n_first, second, n_second = _edge_design(edges, nodes)
    oracle = _scipy_components(first, n_first, second, n_second)
    assert expected is None or oracle == expected
    start = time.perf_counter()
    actual = absorb.connected_components(
        torch.as_tensor(first), n_first, torch.as_tensor(second), n_second
    )
    elapsed = time.perf_counter() - start
    assert actual == oracle
    assert elapsed < 2.0
    # The same answer with the sides swapped and with the codes permuted.
    assert (
        absorb.connected_components(
            torch.as_tensor(second), n_second, torch.as_tensor(first), n_first
        )
        == oracle
    )
    order = rng.permutation(len(first))
    assert (
        absorb.connected_components(
            torch.as_tensor(first[order]), n_first, torch.as_tensor(second[order]), n_second
        )
        == oracle
    )


def test_relabel_branches_order_independence_and_int64_extremes():
    rng = np.random.default_rng(17)
    extremes = torch.tensor(
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min, 0, np.iinfo(np.int64).min]
    )
    dense, levels = absorb.relabel(extremes)
    assert levels == 3 and dense.tolist() == [2, 0, 1, 0]
    for values in (
        rng.integers(0, 3, 1000),
        rng.integers(-2000, 2000, 1000) * 3,
        rng.integers(-(10**17), 10**17, 1000),
        np.full(50, 7),
    ):
        dense, levels = absorb.relabel(torch.as_tensor(values))
        unique, inverse = np.unique(values, return_inverse=True)
        assert levels == len(unique) and dense.dtype == torch.int64
        assert_array_equal(dense.numpy(), inverse)
        order = rng.permutation(len(values))
        permuted, _ = absorb.relabel(torch.as_tensor(values[order]))
        assert_array_equal(permuted.numpy(), dense.numpy()[order])
    single, levels = absorb.relabel(torch.tensor([-5]))
    assert levels == 1 and single.tolist() == [0]


@pytest.mark.parametrize("seed", range(6))
def test_degrees_of_freedom_equal_the_rank_with_gaps_and_cluster_nesting(seed):
    rng = np.random.default_rng(500 + seed)
    n = 90
    first = rng.choice(np.arange(0, 70, 2), size=n)
    second = rng.integers(0, 30, n)
    dims = [(first, 71), (second, 33)]
    rank = np.linalg.matrix_rank(_design_matrix(dims))
    result = absorb.absorbed_degrees_of_freedom(_tensors(dims))
    assert result.levels == [len(np.unique(first)), len(np.unique(second))]
    assert result.total == rank and result.nested == [False, False]
    # Cluster = a coarsening of the first dimension nests it; totals follow the convention
    # that the first non-nested dimension carries the constant.
    nested = absorb.absorbed_degrees_of_freedom(_tensors(dims), (torch.as_tensor(first // 4), 18))
    assert nested.nested == [True, False]
    assert nested.redundant == [result.levels[0], 0] and nested.total == result.levels[1]
    both = absorb.absorbed_degrees_of_freedom(
        _tensors(dims), (torch.zeros(n, dtype=torch.int64), 1)
    )
    assert both.nested == [True, True] and both.total == 0
    # A cluster finer than the first dimension does not nest it.
    finer = absorb.absorbed_degrees_of_freedom(_tensors(dims), (torch.arange(n), n))
    assert finer.nested == [False, False] and finer.total == rank


# --------------------------------------------------------------------------- scale


def _max_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def test_million_rows_two_way_five_columns_in_seconds_with_linear_memory():
    generator = torch.Generator().manual_seed(23)
    n = 1_000_000
    first = torch.randint(0, 100_000, (n,), generator=generator)
    second = torch.randint(0, 1_000, (n,), generator=generator)
    values = torch.randn(n, 5, dtype=torch.float64, generator=generator)
    values += first[:, None].double() * 1e-3 + second[:, None].double() * 1e-2
    weights = torch.rand(n, dtype=torch.float64, generator=generator) + 0.1
    before = _max_rss_bytes()
    start = time.perf_counter()
    result = absorb.demean(values, [(first, 100_000), (second, 1_000)], weights)
    elapsed = time.perf_counter() - start
    grown = _max_rss_bytes() - before
    assert elapsed < 5.0, elapsed
    # Six [n, 5] float64 working blocks are 240 MB; an n-by-n matrix would be 8 TB.
    assert grown < 1_500_000_000, grown
    assert result.converged and result.iterations < 40 and result.values.shape == (n, 5)
    for codes, levels in ((first, 100_000), (second, 1_000)):
        means = absorb.group_means(result.values, codes, levels, weights)
        assert float(means.abs().max()) < 1e-9
    start = time.perf_counter()
    effects = absorb.fixed_effect_estimates(
        values[:, 0], [(first, 100_000), (second, 1_000)], weights
    )
    fitted = effects[0][first] + effects[1][second]
    assert float((fitted - (values[:, 0] - result.values[:, 0])).abs().max()) < 1e-8
    keep = absorb.singleton_mask([(first, 100_000), (second, 1_000)])
    assert int(keep.sum()) > n - 200
    assert absorb.connected_components(first, 100_000, second, 1_000) == 1
    dof = absorb.absorbed_degrees_of_freedom([(first, 100_000), (second, 1_000)], (second, 1_000))
    assert dof.nested == [False, True]
    assert time.perf_counter() - start < 5.0
