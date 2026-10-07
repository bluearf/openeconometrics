"""Adversarial stress tests for engines.linalg and engines.covariance.

Written by an independent reviewer.  Oracles are deliberately different from the
implementation: exact rational arithmetic (fractions.Fraction) for least squares,
explicit double loops over observation pairs and dense n-by-n kernel matrices for the
meats, statsmodels for weighted robust covariances, and a full-data sequential
orthogonalization for the collinearity screen.
"""

import ast
import itertools
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose

from openecon.engines import covariance as cv
from openecon.engines import linalg
from openecon.engines.contracts import KernelError

EPS = np.finfo(float).eps


def t(values):
    return torch.tensor(np.asarray(values), dtype=torch.float64)


def codes(values):
    return torch.tensor(np.asarray(values), dtype=torch.int64)


def raises(code, call):
    with pytest.raises(KernelError) as error:
        call()
    assert error.value.code == code, (error.value.code, str(error.value))


# --------------------------------------------------------------------------- oracles

def exact_inverse(matrix):
    """Gauss-Jordan inverse of a square matrix of Fractions."""
    k = len(matrix)
    work = [list(row) + [Fraction(int(i == j)) for j in range(k)] for i, row in enumerate(matrix)]
    for column in range(k):
        pivot = next(row for row in range(column, k) if work[row][column] != 0)
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(k):
            if row != column and work[row][column] != 0:
                factor = work[row][column]
                work[row] = [a - factor * b for a, b in zip(work[row], work[column])]
    return [row[k:] for row in work]


def exact_wls(x, y, w=None):
    """Exact (rational) beta, (X'WX)^{-1}, leverages and weighted SSR for float inputs."""
    n, k = x.shape
    rows = [[Fraction(float(value)) for value in row] for row in x]
    outcome = [Fraction(float(value)) for value in y]
    weight = [Fraction(1) if w is None else Fraction(float(value)) for value in
              (range(n) if w is None else w)]
    gram = [[sum(weight[i] * rows[i][p] * rows[i][q] for i in range(n)) for q in range(k)]
            for p in range(k)]
    moment = [sum(weight[i] * rows[i][p] * outcome[i] for i in range(n)) for p in range(k)]
    inverse = exact_inverse(gram)
    beta = [sum(inverse[p][q] * moment[q] for q in range(k)) for p in range(k)]
    leverage, ssr = [], Fraction(0)
    for i in range(n):
        half = [sum(inverse[p][q] * rows[i][q] for q in range(k)) for p in range(k)]
        leverage.append(float(weight[i] * sum(rows[i][p] * half[p] for p in range(k))))
        ssr += weight[i] * (outcome[i] - sum(rows[i][p] * beta[p] for p in range(k))) ** 2
    return (np.array([float(value) for value in beta]),
            np.array([[float(value) for value in row] for row in inverse]),
            np.array(leverage), float(ssr))


def sequential_screen(x, w=None, tol=1e-13):
    """Left-to-right screen on the full n-row design (unit-norm columns, numpy QR)."""
    a = x if w is None else x * np.sqrt(w)[:, None]
    norm = np.linalg.norm(a, axis=0)
    a = a / np.where(norm > 0, norm, 1.0)
    kept, omitted = [], []
    for j in range(a.shape[1]):
        residual = a[:, j]
        if kept:
            basis = np.linalg.qr(a[:, kept])[0]
            residual = residual - basis @ (basis.T @ residual)
            residual = residual - basis @ (basis.T @ residual)
        (omitted if residual @ residual <= tol * (a[:, j] @ a[:, j]) else kept).append(j)
    return kept, omitted


def kernel(name, distance, lags):
    if distance == 0:
        return 1.0
    z = distance / (lags + 1)
    if name == "bartlett":
        return max(0.0, 1 - z)
    if name == "truncated":
        return 1.0 if distance <= lags else 0.0
    if name == "parzen":
        if z <= 0.5:
            return 1 - 6 * z ** 2 + 6 * z ** 3
        return 2 * (1 - z) ** 3 if z < 1 else 0.0
    a = 6 * math.pi * z / 5
    return 25 / (12 * math.pi ** 2 * z ** 2) * (math.sin(a) / a - math.cos(a))


def brute_hac(scores, period, unit, name, lags):
    """Double loop over all observation pairs; lags = 0 is the White meat by convention."""
    n, k = scores.shape
    total = np.zeros((k, k))
    for i in range(n):
        for j in range(n):
            if unit is not None and unit[i] != unit[j]:
                continue
            distance = abs(int(period[i]) - int(period[j]))
            weight = (1.0 if distance == 0 else 0.0) if lags == 0 else kernel(name, distance, lags)
            if weight:
                total += weight * np.outer(scores[i], scores[j])
    return total


def brute_shared(scores, groups):
    """Pairs of observations sharing a cluster in ANY dimension enter exactly once."""
    n, k = scores.shape
    total = np.zeros((k, k))
    for i in range(n):
        for j in range(n):
            if any(group[i] == group[j] for group in groups):
                total += np.outer(scores[i], scores[j])
    return total


KERNELS = ["bartlett", "truncated", "parzen", "quadratic_spectral"]

LONGLEY = np.array([
    [60323, 83.0, 234289, 2356, 1590, 107608, 1947],
    [61122, 88.5, 259426, 2325, 1456, 108632, 1948],
    [60171, 88.2, 258054, 3682, 1616, 109773, 1949],
    [61187, 89.5, 284599, 3351, 1650, 110929, 1950],
    [63221, 96.2, 328975, 2099, 3099, 112075, 1951],
    [63639, 98.1, 346999, 1932, 3594, 113270, 1952],
    [64989, 99.0, 365385, 1870, 3547, 115094, 1953],
    [63761, 100.0, 363112, 3578, 3350, 116219, 1954],
    [66019, 101.2, 397469, 2904, 3048, 117388, 1955],
    [67857, 104.6, 419180, 2822, 2857, 118734, 1956],
    [68169, 108.4, 442769, 2936, 2798, 120445, 1957],
    [66513, 110.8, 444546, 4681, 2637, 121950, 1958],
    [68655, 112.6, 482704, 3813, 2552, 123366, 1959],
    [69564, 114.2, 502601, 3931, 2514, 125368, 1960],
    [69331, 115.7, 518173, 4806, 2572, 127852, 1961],
    [70551, 116.9, 554894, 4007, 2827, 130081, 1962],
])


# ------------------------------------------------------------------ runtime contract

def test_runtime_modules_import_only_torch_and_stdlib_and_never_use_autograd():
    for module in (linalg, cv):
        source = pathlib.Path(module.__file__).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:                       # relative: must stay inside engines
                    assert node.level == 1 and node.module in ("contracts", "linalg")
                    continue
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                assert name == "torch" or name in sys.stdlib_module_names, name
        for banned in ("autograd", "requires_grad", "torch.func", "numpy", "scipy", "pandas"):
            assert banned not in source, banned
        assert max(len(line) for line in source.splitlines()) <= 100


def test_kernels_work_inside_inference_mode_and_ignore_the_default_dtype():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(40, 3)), rng.normal(size=40)
    period = np.arange(40)
    expected = linalg.least_squares(t(x), t(y), need_leverage=True)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)      # the kernels must not lean on the default
    try:
        with torch.inference_mode():
            fit = linalg.least_squares(t(x), t(y), need_leverage=True)
            wide = linalg.least_squares(t(np.column_stack([x, x[:, 0]])), t(y))
            meat = cv.meat_hac(t(x), 3, "parzen", time=codes(period[::-1].copy()))
            spectral = cv.meat_hac(t(x), 3, "quadratic_spectral")
            multi = cv.meat_multiway(t(x), [(codes(period % 5), 5), (codes(period % 3), 3)])
            statistic, rank = linalg.wald_statistic(fit.beta, fit.xtx_inv)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(fit.beta, expected.beta) and wide.omitted == [3] and rank == 3
    for value in (fit.beta, fit.leverage, meat, spectral, multi.meat, cv.kernel_weights(4)):
        assert value.dtype == torch.float64


# -------------------------------------------------------------------- least squares

def test_longley_matches_the_exact_rational_solution():
    y, x = LONGLEY[:, 0], np.column_stack([np.ones(16), LONGLEY[:, 1:]])
    beta, inverse, leverage, ssr = exact_wls(x, y)
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    assert fit.omitted == [] and fit.rank == 7
    # NIST StRD certified leading digits (higher difficulty reference data set).
    assert_allclose(beta[[0, 1, 6]], [-3482258.63459582, 15.0618722713733, 1829.15146461355],
                    rtol=1e-13)
    assert_allclose(fit.beta.numpy(), beta, rtol=2e-9)
    assert_allclose(fit.xtx_inv.numpy(), inverse, rtol=2e-8)
    assert_allclose(fit.leverage.numpy(), leverage, rtol=1e-8)
    assert_allclose(fit.ssr.item(), ssr, rtol=1e-6)
    # The normal equations are far worse on this design (condition 5e9 before scaling).
    normal = np.linalg.solve(x.T @ x, x.T @ y)
    assert np.abs(fit.beta.numpy() / beta - 1).max() < 0.01 * np.abs(normal / beta - 1).max()


@pytest.mark.parametrize("pattern", ["unit", "1e-6..1e6", "1e-12..1e12", "all 1e-150",
                                     "all 1e150", "two tiers 240 orders apart"])
def test_wls_with_extreme_weight_ranges_matches_exact_arithmetic(pattern):
    rng = np.random.default_rng(len(pattern))
    n, k = 40, 4
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = x @ [1.0, 2.0, -1.0, 0.5] + rng.normal(size=n)
    w = {
        "unit": np.ones(n),
        "1e-6..1e6": 10.0 ** rng.uniform(-6, 6, size=n),
        "1e-12..1e12": 10.0 ** rng.uniform(-12, 12, size=n),
        "all 1e-150": 1e-150 * rng.uniform(0.5, 2, size=n),
        "all 1e150": 1e150 * rng.uniform(0.5, 2, size=n),
        "two tiers 240 orders apart": np.where(np.arange(n) % 2, 1e120, 1e-120)
        * rng.uniform(0.5, 2, size=n),
    }[pattern]
    beta, inverse, leverage, ssr = exact_wls(x, y, w)
    fit = linalg.least_squares(t(x), t(y), t(w), need_leverage=True)
    assert fit.kept == [0, 1, 2, 3]
    assert_allclose(fit.beta.numpy(), beta, rtol=1e-11)
    assert_allclose(fit.xtx_inv.numpy(), inverse, rtol=1e-11)
    assert_allclose(fit.leverage.numpy(), leverage, rtol=1e-9, atol=1e-13)
    assert_allclose(fit.ssr.item(), ssr, rtol=1e-9)
    assert_allclose(fit.leverage.sum().item(), k, rtol=1e-12)
    assert float(fit.leverage.min()) >= 0 and float(fit.leverage.max()) <= 1 + 1e-12


def test_columns_spanning_two_hundred_orders_of_magnitude():
    rng = np.random.default_rng(7)
    n, k = 30, 5
    z = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = z @ rng.normal(size=k) + rng.normal(size=n)
    units = 10.0 ** np.array([-100.0, 97.0, 0.0, -8.0, 8.0])        # not powers of two
    x = z * units
    beta, inverse, leverage, ssr = exact_wls(x, y)
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    assert fit.omitted == [] and fit.condition_number < 50
    assert_allclose(fit.beta.numpy(), beta, rtol=1e-12)
    assert_allclose(fit.xtx_inv.numpy(), inverse, rtol=1e-12)
    assert_allclose(fit.leverage.numpy(), leverage, rtol=1e-11)
    assert_allclose(fit.ssr.item(), ssr, rtol=1e-11)
    # Outcomes of extreme magnitude scale beta exactly (power of two).
    big = linalg.least_squares(t(x), t(y * 2.0 ** 300))
    assert torch.equal(big.beta, fit.beta * 2.0 ** 300)


@pytest.mark.parametrize("n,k", [(1, 1), (2, 1), (5, 4), (4, 4), (3, 7), (6, 6)])
def test_tiny_and_saturated_shapes(n, k):
    rng = np.random.default_rng(100 * n + k)
    x, y = rng.normal(size=(n, k)), rng.normal(size=n)
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    rank = min(n, k)
    assert fit.rank == rank and fit.kept == list(range(rank))
    assert fit.omitted == list(range(rank, k))
    assert fit.beta.shape == (rank,) and fit.xtx_inv.shape == (rank, rank)
    assert fit.fitted.shape == (n,) and fit.resid.shape == (n,) and fit.ssr.shape == ()
    assert fit.leverage.shape == (n,)
    reduced = x[:, :rank]
    assert_allclose(fit.beta.numpy(), np.linalg.lstsq(reduced, y, rcond=None)[0], rtol=1e-7,
                    atol=1e-9)
    assert_allclose(fit.leverage.sum().item(), rank, rtol=1e-10)
    if n <= k:                                      # saturated: exact fit, unit leverage
        assert_allclose(fit.resid.numpy(), 0, atol=1e-9)
        assert_allclose(fit.leverage.numpy(), 1, rtol=1e-10)
        raises("undefined_leverage_correction",
               lambda: cv.hc_residuals(fit.resid, fit.leverage, "HC3"))


def test_vector_and_single_column_outcomes_agree_exactly():
    rng = np.random.default_rng(3)
    x, y, w = rng.normal(size=(25, 3)), rng.normal(size=25), rng.uniform(0.5, 2, size=25)
    flat = linalg.least_squares(t(x), t(y), t(w), need_leverage=True)
    wide = linalg.least_squares(t(x), t(y[:, None]), t(w), need_leverage=True)
    assert wide.beta.shape == (3, 1) and wide.fitted.shape == (25, 1)
    assert wide.resid.shape == (25, 1) and wide.ssr.shape == (1,)
    assert torch.equal(wide.beta[:, 0], flat.beta) and torch.equal(wide.resid[:, 0], flat.resid)
    assert torch.equal(wide.ssr[0], flat.ssr) and torch.equal(wide.xtx_inv, flat.xtx_inv)
    assert torch.equal(wide.leverage, flat.leverage)
    # One regressor: closed forms.
    single = linalg.least_squares(t(x[:, :1]), t(y), t(w))
    denominator = (w * x[:, 0] ** 2).sum()
    assert_allclose(single.beta.item(), (w * x[:, 0] * y).sum() / denominator, rtol=1e-13)
    assert_allclose(single.xtx_inv.item(), 1 / denominator, rtol=1e-13)
    assert single.condition_number == 1.0
    # An empty set of right-hand sides is legal.
    none = linalg.least_squares(t(x), torch.zeros((25, 0), dtype=torch.float64))
    assert none.beta.shape == (3, 0) and none.ssr.shape == (0,)


def test_inputs_are_never_modified_or_aliased_and_results_are_deterministic():
    rng = np.random.default_rng(4)
    x, y, w = rng.normal(size=(60, 5)), rng.normal(size=(60, 2)), rng.uniform(0, 2, size=60)
    x[:, 4] = x[:, 0] - x[:, 1]
    tx, ty, tw = t(x), t(y), t(w)
    first = linalg.least_squares(tx, ty, tw, need_leverage=True)
    second = linalg.least_squares(tx, ty, tw, need_leverage=True)
    plain = linalg.least_squares(tx, ty)
    linalg.collinear_columns(tx, tw)
    assert np.array_equal(tx.numpy(), x) and np.array_equal(ty.numpy(), y)
    assert np.array_equal(tw.numpy(), w)
    for name in ("beta", "xtx_inv", "fitted", "resid", "ssr", "leverage"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    pointers = {tx.data_ptr(), ty.data_ptr(), tw.data_ptr()}
    for fit in (first, plain, linalg.least_squares(tx[:, :0], ty)):
        assert not pointers & {fit.fitted.data_ptr(), fit.resid.data_ptr()}
    # Non-contiguous views (transposes, strides, negative steps are not allowed in torch).
    base = t(rng.normal(size=(5, 120)))
    view, target = base.T[::2], t(rng.normal(size=120))[::2]
    assert not view.is_contiguous()
    strided = linalg.least_squares(view, target, need_leverage=True)
    dense = linalg.least_squares(view.contiguous(), target.contiguous(), need_leverage=True)
    assert torch.equal(strided.beta, dense.beta) and torch.equal(strided.leverage, dense.leverage)


def test_frequency_weights_equal_replicated_rows_and_zero_weights_equal_deletion():
    rng = np.random.default_rng(5)
    n = 30
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = rng.normal(size=n)
    count = rng.integers(0, 4, size=n)
    weighted = linalg.least_squares(t(x), t(y), t(count.astype(float)), need_leverage=True)
    repeated = linalg.least_squares(t(np.repeat(x, count, axis=0)), t(np.repeat(y, count)),
                                    need_leverage=True)
    assert_allclose(weighted.beta.numpy(), repeated.beta.numpy(), rtol=1e-12)
    assert_allclose(weighted.xtx_inv.numpy(), repeated.xtx_inv.numpy(), rtol=1e-12)
    assert_allclose(weighted.ssr.item(), repeated.ssr.item(), rtol=1e-12)
    # h_i of a weighted row is the sum of the leverages of its replicas.
    replica = np.repeat(np.arange(n), count)
    summed = np.bincount(replica, weights=repeated.leverage.numpy(), minlength=n)
    assert_allclose(weighted.leverage.numpy(), summed, rtol=1e-11, atol=1e-13)
    # Zero-weight rows do not affect the fit but still get fitted values and residuals.
    assert weighted.fitted.shape == (n,)
    assert_allclose(weighted.fitted.numpy(), x @ repeated.beta.numpy(), rtol=1e-11)
    assert_allclose((weighted.resid + weighted.fitted).numpy(), y, rtol=1e-13)
    # Weighted normal equations hold: X'W e = 0.
    score = x.T @ (count * weighted.resid.numpy())
    assert np.abs(score).max() < 1e-11 * np.abs(x.T @ (count * y)).max() + 1e-12
    # A non-finite design value is rejected even in a zero-weight row (pitfall: callers
    # must drop or clean such rows themselves).
    dirty = x.copy()
    dirty[int(np.flatnonzero(count == 0)[0]), 1] = np.nan
    raises("non_finite_values", lambda: linalg.least_squares(
        t(dirty), t(y), t(count.astype(float))))


def test_collinearity_screen_matches_a_full_data_oracle_on_random_designs():
    rng = np.random.default_rng(123)
    checked = 0
    for trial in range(500):
        n, k = int(rng.integers(1, 25)), int(rng.integers(1, 12))
        rank = int(rng.integers(0, k + 1))
        basis = rng.normal(size=(n, max(rank, 1)))
        mixing = rng.integers(-2, 3, size=(max(rank, 1), k)).astype(float) * (rank > 0)
        x = basis @ mixing * 10.0 ** rng.integers(-8, 9, size=k)      # badly scaled columns
        w = None
        if trial % 2:
            w = rng.uniform(0.1, 5, size=n)
            w[rng.uniform(size=n) < 0.3] = 0.0                         # zero weights kill rows
        weights = None if w is None else t(w)
        expected = sequential_screen(x, w)
        assert linalg.collinear_columns(t(x), weights) == expected, trial
        fit = linalg.least_squares(t(x), t(rng.normal(size=n)), weights, need_leverage=True)
        assert (fit.kept, fit.omitted) == expected and fit.rank == len(expected[0])
        assert fit.kept == sorted(fit.kept) and fit.omitted == sorted(fit.omitted)
        assert_allclose(fit.leverage.sum().item(), fit.rank, rtol=1e-8, atol=1e-8)
        if expected[1]:
            raises("singular_design", lambda: linalg.least_squares(
                t(x), t(np.zeros(n)), weights, drop_collinear=False))
            checked += 1
    assert checked > 250


def test_dummy_variable_trap_and_empty_levels_follow_stata_ordering():
    rng = np.random.default_rng(6)
    n = 90
    level = rng.integers(0, 4, size=n)
    level[level == 2] = 0                                   # level 2 is an empty category
    dummies = (level[:, None] == np.arange(4)).astype(float)
    z = rng.normal(size=n)
    # Intercept first: the empty level and the LAST nonempty level are omitted.
    x = np.column_stack([np.ones(n), dummies, z])
    assert linalg.collinear_columns(t(x)) == ([0, 1, 2, 5], [3, 4])
    # Dummies first: the intercept is the one that goes.
    x = np.column_stack([dummies, np.ones(n), z])
    assert linalg.collinear_columns(t(x)) == ([0, 1, 3, 5], [2, 4])
    # Two full sets of dummies plus an intercept: one from each later block is dropped.
    other = (rng.integers(0, 3, size=n)[:, None] == np.arange(3)).astype(float)
    x = np.column_stack([np.ones(n), dummies[:, [0, 1, 3]], other])
    assert linalg.collinear_columns(t(x)) == ([0, 1, 2, 4, 5], [3, 6])
    fit = linalg.least_squares(t(x), t(z), need_leverage=True)
    kept = x[:, fit.kept]
    hat = kept @ np.linalg.inv(kept.T @ kept) @ kept.T
    assert_allclose(fit.leverage.numpy(), np.diag(hat), rtol=1e-10)
    assert_allclose(fit.fitted.numpy(), hat @ z, rtol=1e-9, atol=1e-11)
    # A single-observation category is fitted exactly: unit leverage, HC3 undefined.
    lone = np.zeros(n)
    lone[17] = 1.0
    fit = linalg.least_squares(t(np.column_stack([np.ones(n), z, lone])), t(z ** 2),
                               need_leverage=True)
    assert_allclose(fit.leverage[17].item(), 1.0, rtol=1e-13)
    assert abs(fit.resid[17].item()) < 1e-12
    raises("undefined_leverage_correction",
           lambda: cv.hc_residuals(fit.resid, fit.leverage, "HC2"))


def test_tolerance_semantics_explicit_spec_value_and_default():
    rng = np.random.default_rng(9)
    n = 200
    a, noise = rng.normal(size=(2, n))
    near = np.column_stack([np.ones(n), a, a + 1e-6 * noise])
    # Default: a relative independent length of 1e-6 is genuine variation and is kept.
    assert linalg.collinear_columns(t(near)) == ([0, 1, 2], [])
    fit = linalg.least_squares(t(near), t(a + noise))
    assert fit.omitted == [] and 1e5 < fit.condition_number < 1e8
    exact = exact_wls(near, a + noise)[0]
    # beta is about (0, -1e6, 1e6): the intercept is a difference of two 1e6 terms, so
    # its error is bounded relative to ||beta|| (and by condition * eps), not to itself.
    assert np.abs(fit.beta.numpy() - exact).max() < 1e-12 * np.abs(exact).max()
    # Fitted values are differences of 1e6-sized products: absolute error ~1e6 * eps.
    assert_allclose(fit.fitted.numpy(), near @ exact, rtol=0, atol=1e-8)
    # The literal 1e-10 sum-of-squares rule drops the same column (ratio ~ 1e-12).
    assert linalg.collinear_columns(t(near), tol=1e-10) == ([0, 1], [2])
    assert linalg.least_squares(t(near), t(a), tol=1e-10).omitted == [2]
    # tol = 0 only drops columns whose residual is exactly zero.
    assert linalg.collinear_columns(t(near[:, [0, 1, 0]]), tol=0.0)[1] in ([2], [])
    assert linalg.collinear_columns(t(np.zeros((4, 2))), tol=0.0) == ([], [0, 1])
    for bad in (-1e-3, 1.0, float("nan"), float("inf")):
        raises("invalid_solver_options", lambda: linalg.collinear_columns(t(near), tol=bad))


def test_least_squares_fuzz_stays_within_conditioning_bounds_of_the_svd_solution():
    """Random shapes, units over 13 orders of magnitude, near-collinear pairs, zero weights."""
    rng = np.random.default_rng(2024)
    checked = 0
    for trial in range(150):
        n, k = int(rng.integers(2, 60)), int(rng.integers(1, 8))
        if n < k:
            continue
        x = rng.normal(size=(n, k)) * 10.0 ** rng.integers(-6, 7, size=k)
        if k > 1 and trial % 3 == 0:
            x[:, -1] = 1e3 * x[:, 0] + 1e-5 * rng.normal(size=n)
        y = rng.normal(size=n)
        w = None
        if trial % 2:
            w = rng.uniform(0.1, 3, size=n)
            w[rng.uniform(size=n) < 0.1] = 0.0
        fit = linalg.least_squares(t(x), t(y), None if w is None else t(w), need_leverage=True)
        if fit.omitted:
            continue
        a = x if w is None else x * np.sqrt(w)[:, None]
        b = y if w is None else y * np.sqrt(w)
        scale = np.linalg.norm(a, axis=0)
        beta = np.linalg.lstsq(a / scale, b, rcond=None)[0] / scale
        condition = np.linalg.cond(a / scale)
        assert_allclose(fit.condition_number, condition, rtol=1e-8)
        # Backward-stable solvers agree to about condition * eps relative to ||beta||.
        assert np.abs(fit.beta.numpy() - beta).max() <= 2000 * condition * EPS * np.abs(beta).max()
        inverse = np.linalg.inv((a / scale).T @ (a / scale)) / np.outer(scale, scale)
        assert np.abs(fit.xtx_inv.numpy() - inverse).max() <= \
            (2000 * condition ** 2 * EPS + 1e-13) * np.abs(inverse).max()
        assert_allclose(fit.leverage.sum().item(), k, rtol=1e-9)
        assert fit.leverage.min().item() >= -1e-15 and fit.leverage.max().item() <= 1 + 1e-12
        checked += 1
    assert checked > 100


def test_least_squares_error_codes():
    rng = np.random.default_rng(10)
    x, y = rng.normal(size=(8, 2)), rng.normal(size=8)
    cases = [
        ("invalid_design", lambda: linalg.least_squares(x, t(y))),
        ("invalid_design", lambda: linalg.least_squares(t(x), y)),
        ("invalid_design", lambda: linalg.least_squares(t(x).to(torch.float32), t(y))),
        ("invalid_design", lambda: linalg.least_squares(t(x), codes(np.arange(8)))),
        ("invalid_design", lambda: linalg.least_squares(t(x), t(y)[:, None, None])),
        ("invalid_design", lambda: linalg.least_squares(t(x[:0]), t(y[:0]))),
        ("invalid_weights", lambda: linalg.least_squares(t(x), t(y), t(np.full(8, np.inf)))),
        ("invalid_weights", lambda: linalg.least_squares(t(x), t(y), t(np.ones((8, 1))))),
        ("invalid_weights", lambda: linalg.least_squares(t(x), t(y), np.ones(8))),
        ("non_finite_values", lambda: linalg.least_squares(t(x * np.inf), t(y))),
        ("non_finite_values", lambda: linalg.collinear_columns(t(np.where(x > 9, 0, np.nan)))),
        ("singular_design", lambda: linalg.least_squares(
            t(np.zeros((8, 1))), t(y), drop_collinear=False)),
    ]
    for code, call in cases:
        raises(code, call)
    # All weights zero: nothing is estimable, every column is omitted.
    empty = linalg.least_squares(t(x), t(y), torch.zeros(8, dtype=torch.float64))
    assert empty.rank == 0 and empty.omitted == [0, 1] and torch.equal(empty.resid, t(y))


# ------------------------------------------------------------- symmetric k-by-k helpers

def test_cholesky_helpers_never_invert_a_singular_gram_matrix():
    rng = np.random.default_rng(0)
    for trial in range(300):
        k = int(rng.integers(2, 12))
        n = 3 * k + 20
        x = rng.normal(size=(n, k)) * 10.0 ** rng.integers(-3, 4, size=k)
        dependent = int(rng.integers(1, k))
        x[:, dependent] = x[:, :dependent] @ rng.integers(1, 4, size=dependent)
        w = rng.uniform(0.1, 3, size=n)
        gram = x.T @ (x * w[:, None]) if trial % 2 else x.T @ x
        # LAPACK alone accepts about half of these (positive rounding pivot).
        raises("singular_information", lambda: linalg.cholesky_inverse(
            t(gram), code="singular_information", what="information matrix"))
        raises("singular_matrix", lambda: linalg.cholesky_solve(t(gram), t(np.ones(k))))
        raises("singular_matrix", lambda: linalg.quadratic_form(
            t(np.ones(k)), t(gram), inverse=True))
    # A cluster meat with fewer clusters than parameters is singular as well.
    scores = rng.normal(size=(50, 6))
    meat = cv.meat_cluster(t(scores), codes(np.arange(50) % 4), 4)
    raises("singular_matrix", lambda: linalg.cholesky_inverse(meat))


def test_cholesky_helpers_accept_ill_conditioned_and_badly_scaled_spd_matrices():
    rng = np.random.default_rng(1)
    for condition in (1e3, 1e8, 1e12):
        for _ in range(20):
            k = 7
            rotation = np.linalg.qr(rng.normal(size=(k, k)))[0]
            a = (rotation * np.geomspace(1, 1 / condition, k)) @ rotation.T
            a = (a + a.T) / 2
            inverse = linalg.cholesky_inverse(t(a))
            assert torch.equal(inverse, inverse.T)
            assert np.abs(inverse.numpy() @ a - np.eye(k)).max() < 100 * condition * EPS
            b = rng.normal(size=(k, 2))
            assert_allclose(a @ linalg.cholesky_solve(t(a), t(b)).numpy(), b,
                            atol=100 * condition * EPS * np.abs(b).max())
    # Units from 1e-70 to 1e70: the factorization is invariant to diagonal scaling.
    root = rng.normal(size=(4, 4))
    core = root @ root.T + np.eye(4)
    units = 10.0 ** np.array([-70.0, 70.0, 0.0, 30.0])
    a = core * np.outer(units, units)
    matrix = [[Fraction(float(value)) for value in row] for row in a]
    exact = np.array([[float(value) for value in row] for row in exact_inverse(matrix)])
    assert_allclose(linalg.cholesky_inverse(t(a)).numpy(), exact, rtol=1e-12)
    v = rng.normal(size=4) * units
    assert_allclose(linalg.quadratic_form(t(v), t(a), inverse=True), v @ exact @ v, rtol=1e-12)
    statistic, rank = linalg.wald_statistic(t(v), t(a))
    assert rank == 4
    assert_allclose(statistic, v @ exact @ v, rtol=1e-11)


def test_cholesky_helpers_never_reject_spd_matrices_up_to_condition_1e12():
    """Rotated spectra with condition 1..1e12, k up to 60, units from 1e-50 to 1e50."""
    rng = np.random.default_rng(14)
    for trial in range(120):
        k = int(rng.integers(1, 61))
        condition = 10.0 ** rng.uniform(0, 12)
        rotation = np.linalg.qr(rng.normal(size=(k, k)))[0]
        core = (rotation * np.geomspace(1, 1 / condition, k)) @ rotation.T
        units = 10.0 ** rng.uniform(-50, 50, size=k)
        a = (core + core.T) / 2 * np.outer(units, units)
        inverse = linalg.cholesky_inverse(t(a))          # must not raise
        # In the units of core (D a^{-1} D = core^{-1}) the residual is condition * eps.
        rescaled = inverse.numpy() * np.outer(units, units)
        assert np.abs(rescaled @ core - np.eye(k)).max() < 1e3 * k * condition * EPS
        v = rng.normal(size=k) * units
        # a = D core D with D = diag(units):  v' a^{-1} v = (v / units)' core^{-1} (v / units).
        expected = (v / units) @ np.linalg.solve(core, v / units)
        assert_allclose(linalg.quadratic_form(t(v), t(a), inverse=True), expected,
                        rtol=1e3 * condition * EPS + 1e-12)


def test_wald_statistic_on_singular_badly_scaled_covariances_matches_the_factor_solution():
    """V = R R' (rank r <= k) and b = R z give b' V^- b = z'z exactly, rank r."""
    rng = np.random.default_rng(31)
    for trial in range(150):
        k = int(rng.integers(1, 7))
        r = int(rng.integers(1, k + 1))
        root = rng.normal(size=(k, r)) * 10.0 ** rng.integers(-2, 3, size=r)
        rows = 10.0 ** rng.integers(-6, 7, size=k)                  # parameter units
        root = root * rows[:, None]
        z = rng.normal(size=r)
        statistic, rank = linalg.wald_statistic(t(root @ z), t(root @ root.T))
        # Directions of the correlation matrix below 1e-12 of its largest eigenvalue are
        # treated as redundant constraints; this draw must not sit near that edge.
        correlation = (root / rows[:, None]) @ (root / rows[:, None]).T
        spread = np.sqrt(np.diag(correlation))
        spectrum = np.linalg.eigvalsh(correlation / np.outer(spread, spread))[::-1][:r]
        if spectrum[-1] < 1e-8 * spectrum[0]:
            continue
        assert rank == r, (trial, rank, r)
        assert_allclose(statistic, z @ z, rtol=1e-6)


def test_cholesky_edge_cases():
    empty = torch.zeros((0, 0), dtype=torch.float64)
    assert linalg.cholesky_inverse(empty).shape == (0, 0)
    assert_allclose(linalg.cholesky_inverse(t([[4.0]])).numpy(), [[0.25]])
    a = t([[4.0, 1.0], [1.0, 3.0]])
    saved = a.clone()
    assert linalg.cholesky_solve(a, t([1.0, 2.0])).shape == (2,)
    assert linalg.cholesky_solve(a, t([[1.0], [2.0]])).shape == (2, 1)
    assert torch.equal(a, saved)
    for matrix in ([[np.nan, 0.0], [0.0, 1.0]], [[np.inf, 0.0], [0.0, 1.0]],
                   [[0.0, 0.0], [0.0, 1.0]], [[1.0, 2.0], [2.0, 1.0]], [[-1.0]],
                   [[1e-320, 0.0], [0.0, 1.0]]):
        raises("singular_matrix", lambda: linalg.cholesky_inverse(t(matrix)))
    raises("singular_matrix", lambda: linalg.cholesky_inverse(t(np.ones(3))))
    raises("singular_matrix", lambda: linalg.cholesky_inverse(torch.eye(2)))      # float32
    raises("invalid_input", lambda: linalg.cholesky_solve(a, torch.ones(2)))
    raises("invalid_input", lambda: linalg.cholesky_solve(a, [1.0, 2.0]))


def test_wald_statistic_degenerate_covariances():
    rng = np.random.default_rng(2)
    root = rng.normal(size=(3, 3))
    sigma = root @ root.T + np.eye(3)
    b = rng.normal(size=3)
    full = b @ np.linalg.solve(sigma, b)
    # Every constraint repeated three times: rank and statistic are unchanged.
    repeat = np.vstack([np.eye(3)] * 3)
    statistic, rank = linalg.wald_statistic(t(repeat @ b), t(repeat @ sigma @ repeat.T))
    assert rank == 3
    assert_allclose(statistic, full, rtol=1e-9)
    # Zero-variance coordinates (omitted coefficients) are dropped, wherever they sit.
    padded = np.zeros((5, 5))
    index = [0, 2, 4]
    padded[np.ix_(index, index)] = sigma
    estimate = np.zeros(5)
    estimate[index] = b
    statistic, rank = linalg.wald_statistic(t(estimate), t(padded))
    assert rank == 3
    assert_allclose(statistic, full, rtol=1e-11)
    # One coordinate: (b / se)^2, in any units.
    for scale in (1e-150, 1.0, 1e150):
        statistic, rank = linalg.wald_statistic(t([3.0 * scale]), t([[4.0 * scale ** 2]]))
        assert rank == 1
        assert_allclose(statistic, 2.25, rtol=1e-12)
    # A genuinely indefinite covariance contributes only its positive directions.
    indefinite = np.diag([2.0, -1.0, 0.5])
    indefinite[0, 0] = 2.0
    raises("invalid_covariance", lambda: linalg.wald_statistic(t([1.0, 1.0, 1.0]),
                                                                 t(indefinite)))
    twisted = np.array([[1.0, 2.0], [2.0, 1.0]])                  # eigenvalues 3 and -1
    statistic, rank = linalg.wald_statistic(t([1.0, 1.0]), t(twisted))
    assert rank == 1
    assert_allclose(statistic, 2.0 / 3.0, rtol=1e-12)
    raises("invalid_covariance", lambda: linalg.wald_statistic(t([1.0, np.nan]), t(np.eye(2))))
    raises("invalid_covariance", lambda: linalg.wald_statistic(t([1.0, 1.0]),
                                                                 t([[1.0, np.inf], [np.inf, 1]])))
    raises("singular_covariance", lambda: linalg.wald_statistic(
        torch.zeros(0, dtype=torch.float64), torch.zeros((0, 0), dtype=torch.float64)))


def test_weighted_crossprod_blocks_shapes_and_validation(monkeypatch):
    rng = np.random.default_rng(3)
    n = 257
    x, z, w = rng.normal(size=(n, 3)), rng.normal(size=(n, 2)), rng.normal(size=n)
    tx, tz, tw = t(x), t(z), t(w)
    whole = [linalg.weighted_crossprod(tx, tw), linalg.weighted_crossprod(tx, tw, tz),
             linalg.weighted_crossprod(tx, tw, tz[:, 0]), linalg.weighted_crossprod(tx, None)]
    for rows_per_block in (1, 7, 256, 257):
        monkeypatch.setattr(linalg, "_BLOCK_ELEMENTS", 3 * rows_per_block)
        blocks = [linalg.weighted_crossprod(tx, tw), linalg.weighted_crossprod(tx, tw, tz),
                  linalg.weighted_crossprod(tx, tw, tz[:, 0]), linalg.weighted_crossprod(tx, None)]
        for left, right in zip(whole, blocks):
            assert left.shape == right.shape
            assert_allclose(left.numpy(), right.numpy(), rtol=1e-12, atol=1e-12)
    assert whole[2].shape == (3,)
    assert_allclose(whole[1].numpy(), x.T @ (z * w[:, None]), rtol=1e-12, atol=1e-13)
    assert np.array_equal(tx.numpy(), x) and np.array_equal(tw.numpy(), w)
    assert linalg.weighted_crossprod(tx[:0], tw[:0]).tolist() == [[0.0] * 3] * 3
    assert linalg.weighted_crossprod(tx[:, :0], tw).shape == (0, 0)
    for call in (lambda: linalg.weighted_crossprod(tx, tw, z),
                 lambda: linalg.weighted_crossprod(tx, w),
                 lambda: linalg.weighted_crossprod(tx, tw, tz.float()),
                 lambda: linalg.weighted_crossprod(tx[:, 0], tw),
                 lambda: linalg.weighted_crossprod(tx, tw[:, None])):
        raises("invalid_input", call)


# ------------------------------------------------------------------ group operations

def test_group_sums_with_empty_singleton_and_unbalanced_groups():
    rng = np.random.default_rng(4)
    sizes = [0, 1, 0, 500, 1, 2, 0, 37]
    group = np.repeat(np.arange(len(sizes)), sizes)
    rng.shuffle(group)
    values = rng.normal(size=(len(group), 3))
    tv, tg = t(values), codes(group)
    sums = cv.group_sums(tv, tg, len(sizes))
    expected = np.stack([values[group == g].sum(axis=0) for g in range(len(sizes))])
    assert sums.shape == (8, 3)
    assert_allclose(sums.numpy(), expected, rtol=1e-12, atol=1e-13)
    assert torch.equal(sums[[0, 2, 6]], torch.zeros((3, 3), dtype=torch.float64))
    assert torch.equal(sums[1], tv[group == 1][0])                 # a singleton is exact
    assert cv.group_counts(tg, len(sizes)).tolist() == sizes
    assert cv.group_sums(tv[:, 0], tg, len(sizes)).shape == (8,)
    assert np.array_equal(tv.numpy(), values) and np.array_equal(tg.numpy(), group)
    # Integer payloads keep their dtype; no rows at all gives zeros.
    assert cv.group_sums(tg, tg, len(sizes)).dtype == torch.int64
    assert cv.group_sums(tv[:0], tg[:0], 4).tolist() == [[0.0] * 3] * 4
    assert cv.group_counts(tg[:0], 4).tolist() == [0, 0, 0, 0]
    for call in (lambda: cv.group_sums(tv, tg, 7), lambda: cv.group_sums(tv, tg.to(torch.int32), 8),
                 lambda: cv.group_sums(tv, tg, 0), lambda: cv.group_sums(tv, tg, True),
                 lambda: cv.group_sums(tv, tg, 8.0), lambda: cv.group_counts(group, 8)):
        raises("invalid_clusters", call)


def test_cluster_meat_degenerate_groupings_and_sparse_identifiers():
    rng = np.random.default_rng(5)
    n = 60
    scores = rng.normal(size=(n, 3))
    ts = t(scores)
    total = scores.sum(axis=0)
    one = cv.meat_cluster(ts, codes(np.zeros(n)), 1)
    assert_allclose(one.numpy(), np.outer(total, total), rtol=1e-12, atol=1e-12)
    own = cv.meat_cluster(ts, codes(rng.permutation(n)), n)
    assert_allclose(own.numpy(), scores.T @ scores, rtol=1e-13)
    # Raw identifiers used as codes (declared range 2^40) must neither allocate the
    # declared range nor change the answer.
    group = rng.integers(0, 9, size=n)
    sparse = group * 123_456_789_012 + 5
    compact = cv.meat_cluster(ts, codes(group), 9)
    assert_allclose(cv.meat_cluster(ts, codes(sparse), 1 << 40).numpy(), compact.numpy(),
                    rtol=1e-13)
    multi = cv.meat_multiway(ts, [(codes(sparse), 1 << 40), (codes(group % 2 * 10 ** 9), 1 << 31)],
                             adjust="none")
    assert multi.group_counts == [9, 2]
    # group is nested in group % 2: pairs sharing a cluster in either dimension are the
    # pairs sharing the COARSER one, so the two-way meat is the one-way meat on group % 2.
    coarse = cv.meat_cluster(ts, codes(group % 2), 2)
    assert_allclose(multi.meat.numpy(), coarse.numpy(), rtol=1e-11, atol=1e-11)
    assert not np.allclose(multi.meat.numpy(), compact.numpy())
    # k = 1 scores and the scalar factor.
    column = cv.meat_cluster(ts[:, :1], codes(group), 9)
    sums = np.bincount(group, weights=scores[:, 0], minlength=9)
    assert column.shape == (1, 1)
    assert_allclose(column.item(), (sums ** 2).sum(), rtol=1e-13)
    assert cv.cluster_factor(n, 3, 9) == 9 / 8 * (n - 1) / (n - 3)


def test_multiway_meat_four_dimensions_nested_duplicate_and_conventions():
    rng = np.random.default_rng(6)
    n = 45
    scores = rng.normal(size=(n, 2))
    ts = t(scores)
    groups = [rng.integers(0, size, size=n) for size in (5, 3, 7, 2)]
    groups[2][groups[2] == 4] = 0                            # an empty level
    dimensions = [(codes(group), int(group.max()) + 3) for group in groups]
    result = cv.meat_multiway(ts, dimensions, adjust="none")
    assert_allclose(result.meat.numpy(), brute_shared(scores, groups), rtol=1e-10, atol=1e-10)
    assert len(result.terms) == 15
    assert result.group_counts == [len(np.unique(group)) for group in groups]
    assert result.min_groups == 2
    assert [term["sign"] for term in result.terms] == \
        [1 if len(term["dimensions"]) % 2 else -1 for term in result.terms]
    assert sorted(tuple(term["dimensions"]) for term in result.terms) == sorted(
        subset for size in range(1, 5) for subset in itertools.combinations(range(4), size))

    def rebuilt(outcome):
        """Reassemble the meat from the reported terms alone."""
        total = np.zeros((2, 2))
        for term in outcome.terms:
            labels = np.stack([groups[d] for d in term["dimensions"]], axis=1)
            cell = np.unique(labels, axis=0, return_inverse=True)[1].ravel()
            assert term["groups"] == cell.max() + 1
            sums = np.stack([scores[cell == g].sum(axis=0) for g in range(cell.max() + 1)])
            total += term["sign"] * term["factor"] * sums.T @ sums
        return total

    for adjust in ("none", "min", "each"):
        for extra in ({}, {"n": n, "k": 4}):
            outcome = cv.meat_multiway(ts, dimensions, adjust=adjust, **extra)
            assert_allclose(outcome.meat.numpy(), rebuilt(outcome), rtol=1e-9, atol=1e-9)
            degrees = (n - 1) / (n - 4) if extra else 1.0
            for term in outcome.terms:
                cells = term["groups"]
                expected = {"none": 1.0, "min": 2.0, "each": cells / (cells - 1)}[adjust]
                assert_allclose(term["factor"], expected * degrees, rtol=1e-14)
            assert torch.equal(outcome.meat, outcome.meat.T)
    # The same dimension twice, and a dimension nested in another, collapse to one-way.
    firm = groups[0]
    state = firm // 2
    one_way = cv.meat_cluster(ts, codes(state), 3).numpy()
    for pair in ([(codes(state), 3), (codes(state), 3)], [(codes(firm), 5), (codes(state), 3)]):
        nested = cv.meat_multiway(ts, pair, adjust="none")
        assert_allclose(nested.meat.numpy(), one_way, rtol=1e-11, atol=1e-11)
    # force_psd output is positive semidefinite and flagged only when it had to act.
    forced = cv.meat_multiway(ts, dimensions, adjust="each", force_psd=True)
    assert np.linalg.eigvalsh(forced.meat.numpy())[0] > -1e-10
    raw = cv.meat_multiway(ts, dimensions, adjust="each")
    assert forced.psd_adjusted == bool(np.linalg.eigvalsh(raw.meat.numpy())[0] < -1e-10)
    for call in (lambda: cv.meat_multiway(ts, [codes(firm)]),
                 lambda: cv.meat_multiway(ts, (codes(firm), 5)),
                 lambda: cv.meat_multiway(ts, [(codes(firm), 5, 1)])):
        raises("invalid_clusters", call)
    raises("invalid_cluster_adjustment", lambda: cv.meat_multiway(ts, dimensions, n=4, k=4))
    raises("invalid_cluster_adjustment", lambda: cv.meat_multiway(ts, dimensions, k=4))


def test_weighted_robust_and_cluster_covariances_match_statsmodels_wls():
    rng = np.random.default_rng(7)
    n, k, clusters = 180, 4, 14
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    group = rng.integers(0, clusters, size=n)
    w = rng.uniform(0.2, 4.0, size=n)
    y = x @ [1.0, -1.0, 0.5, 2.0] + rng.normal(size=clusters)[group] \
        + rng.normal(size=n) * (1 + np.abs(x[:, 1]))
    fit = linalg.least_squares(t(x), t(y), t(w), need_leverage=True)
    oracle = sm.WLS(y, x, weights=w).fit()
    # Scores are x_i * w_i * u_i ("already multiplied by any weight").
    for kind, expected in [("HC0", oracle.cov_HC0), ("HC1", oracle.cov_HC1),
                           ("HC2", oracle.cov_HC2), ("HC3", oracle.cov_HC3)]:
        scores = t(x) * (t(w) * cv.hc_residuals(fit.resid, fit.leverage, kind))[:, None]
        covariance = cv.sandwich(fit.xtx_inv, cv.meat_white(scores))
        if kind == "HC1":
            covariance = covariance * n / (n - k)
        assert_allclose(covariance.numpy(), expected, rtol=1e-9, atol=1e-15)
    scores = t(x) * (t(w) * fit.resid)[:, None]
    clustered = cv.sandwich(fit.xtx_inv, cv.meat_cluster(scores, codes(group), clusters)) \
        * cv.cluster_factor(n, k, clusters)
    expected = sm.WLS(y, x, weights=w).fit(cov_type="cluster",
                                           cov_kwds={"groups": group}).cov_params()
    assert_allclose(clustered.numpy(), expected, rtol=1e-9, atol=1e-15)
    hac = cv.sandwich(fit.xtx_inv, cv.meat_hac(scores, 4)) * n / (n - k)
    expected = sm.WLS(y, x, weights=w).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}).cov_params()
    assert_allclose(hac.numpy(), expected, rtol=1e-9, atol=1e-15)


def test_hc_residuals_boundaries_and_validation():
    resid = t([1.0, -2.0, 3.0])
    fine = t([0.0, 0.5, 1 - 1e-10])
    assert_allclose(cv.hc_residuals(resid, fine, "HC3").numpy(), resid.numpy() / (1 - fine.numpy()),
                    rtol=1e-15)
    assert cv.hc_residuals(resid[:, None], fine, "HC2").shape == (3, 1)
    saved = resid.clone()
    cv.hc_residuals(resid, fine, "HC2")
    assert torch.equal(resid, saved)
    for leverage in ([0.0, 0.5, 1 - 50 * EPS], [0.0, 0.5, 1.0], [0.0, 0.5, 1.5],
                     [0.0, np.nan, 0.5]):
        for kind in ("HC2", "HC3"):
            raises("undefined_leverage_correction",
                   lambda: cv.hc_residuals(resid, t(leverage), kind))
    raises("missing_leverage", lambda: cv.hc_residuals(resid, t([0.1, 0.2]), "HC2"))
    raises("missing_leverage", lambda: cv.hc_residuals(resid, t([[0.1], [0.2], [0.3]]), "HC3"))
    raises("missing_leverage", lambda: cv.hc_residuals([1.0, 2.0, 3.0], fine, "HC3"))
    raises("unsupported_covariance", lambda: cv.hc_residuals(resid, fine, "hc3"))


# ---------------------------------------------------------------------------- HAC

@pytest.mark.parametrize("force_fft", [False, True])
def test_hac_fuzz_against_pair_loops(monkeypatch, force_fft):
    """Gaps, shuffled rows, unbalanced panels, singleton units, far-away time origins."""
    if force_fft:
        monkeypatch.setattr(cv, "_FFT_OFFSETS", 0)
        monkeypatch.setattr(cv, "_FFT_SPARSITY", 10 ** 12)
    rng = np.random.default_rng(99)
    for trial in range(160):
        n, k = int(rng.integers(1, 30)), int(rng.integers(1, 4))
        scores = rng.normal(size=(n, k))
        name = KERNELS[trial % 4]
        lags = int(rng.integers(0, 8))
        origin = int(rng.choice([0, -50, 10 ** 9, -10 ** 12]))
        if trial % 3 == 0:
            period = origin + np.cumsum(rng.choice([1, 1, 2, 5, 40], size=n))
            unit = None
        else:
            units, length = int(rng.integers(1, 6)), int(rng.integers(1, 12))
            cells = [(u, p) for u in range(units) for p in range(length)]
            chosen = rng.choice(len(cells), size=min(n, len(cells)), replace=False)
            unit = np.array([cells[i][0] for i in chosen]) * int(rng.choice([1, 7, 1000])) \
                + int(rng.choice([0, -3, 10 ** 6]))
            period = np.array([cells[i][1] for i in chosen]) * int(rng.choice([1, 3])) + origin
            scores = scores[: len(chosen)]
            if trial % 2:                               # already sorted: the no-sort path
                order = np.lexsort((period, unit))
                unit, period, scores = unit[order], period[order], scores[order]
        expected = brute_hac(scores, period, unit, name, lags)
        meat = cv.meat_hac(t(scores), lags, name, time=codes(period),
                           panel=None if unit is None else codes(unit))
        bound = 1e-10 * max(1.0, np.abs(expected).max())
        assert np.abs(meat.numpy() - expected).max() < bound, (trial, name, lags)
        assert torch.equal(meat, meat.T)
        if unit is None:
            # Driscoll-Kraay with several rows per period, and the consecutive-row path.
            copies = rng.integers(1, 4, size=len(scores))
            spread = np.repeat(scores, copies, axis=0) * rng.normal(size=(copies.sum(), 1))
            when = np.repeat(period, copies)
            shuffle = rng.permutation(len(when))
            pooled = cv.meat_driscoll_kraay(t(spread[shuffle]), codes(when[shuffle]), lags, name)
            expected = brute_hac(spread, when, None, name, lags)
            assert np.abs(pooled.numpy() - expected).max() < 1e-10 * max(1, np.abs(expected).max())
            ordered = scores[np.argsort(period)]
            plain = cv.meat_hac(t(ordered), lags, name)
            expected = brute_hac(ordered, np.arange(len(ordered)), None, name, lags)
            assert np.abs(plain.numpy() - expected).max() < 1e-10 * max(1, np.abs(expected).max())


@pytest.mark.parametrize("name", KERNELS)
def test_long_bandwidths_use_the_fft_path_and_match_a_dense_kernel_matrix(name, monkeypatch):
    rng = np.random.default_rng(11)
    n, k, lags = 700, 3, 150
    scores = rng.normal(size=(n, k)) * [1.0, 1e-6, 1e6]
    scores[1:] += 0.7 * scores[:-1]
    scores[13] *= 1e4                                          # an outlying period
    calls = []
    original = cv._lagged_sums_fft
    monkeypatch.setattr(cv, "_lagged_sums_fft",
                        lambda *args: calls.append(1) or original(*args))

    def dense(period, unit=None):
        gap = np.abs(period[:, None] - period[None, :])
        weight = np.vectorize(lambda d: kernel(name, d, lags))(gap)
        if unit is not None:
            weight = weight * (unit[:, None] == unit[None, :])
        return scores.T @ weight @ scores

    scale = np.sqrt(np.diag(scores.T @ scores))
    tolerance = 1e-10 * np.outer(scale, scale)

    def close(meat, expected):
        return bool((np.abs(meat.numpy() - expected) <= tolerance).all())

    index = np.arange(n)
    assert close(cv.meat_hac(t(scores), lags, name), dense(index))
    assert len(calls) == 1
    # Shuffled rows with an explicit period index containing gaps.
    period = np.cumsum(rng.choice([1, 1, 1, 2, 3], size=n)) - 10 ** 6
    order = rng.permutation(n)
    shuffled = cv.meat_hac(t(scores[order]), lags, name, time=codes(period[order]))
    assert close(shuffled, dense(period)) and len(calls) == 2
    # A long panel: two units of unequal length, lags beyond the FFT threshold.
    unit = (index >= 400).astype(np.int64) * 9
    within = np.where(unit == 0, index, 2 * (index - 400))
    panel = cv.meat_hac(t(scores[order]), lags, name, time=codes(within[order]),
                        panel=codes(unit[order]))
    assert close(panel, dense(within, unit)) and len(calls) == 3
    # The loop and FFT paths agree with each other to rounding.
    monkeypatch.setattr(cv, "_FFT_OFFSETS", 10 ** 9)
    looped = cv.meat_hac(t(scores), lags, name)
    assert len(calls) == 3 and close(looped, dense(index))
    # Sparse periods (grid over four times the rows) stay on the offset scan.
    sparse = period * 50
    slow = cv.meat_hac(t(scores[:120]), lags, name, time=codes(sparse[:120]))
    monkeypatch.setattr(cv, "_FFT_OFFSETS", 64)
    fast = cv.meat_hac(t(scores[:120]), lags, name, time=codes(sparse[:120]))
    assert len(calls) == 3 and torch.equal(slow, fast)


@pytest.mark.parametrize("name", KERNELS)
def test_panel_hac_fft_path_with_raw_unit_ids_column_blocks_and_any_row_order(name, monkeypatch):
    """70 lags on an unbalanced 7-unit panel: dense grid -> FFT, whatever the unit codes."""
    rng = np.random.default_rng(17)
    units, periods, lags = 7, 120, 70
    cells = [(u, p) for u in range(units) for p in range(periods) if rng.uniform() > 0.2]
    unit = np.array([cell[0] for cell in cells])
    period = np.array([cell[1] for cell in cells])
    n = len(cells)
    scores = rng.normal(size=(n, 5)) * [1.0, 1e-5, 1e5, 3.0, 0.1]
    scores[1:] += 0.8 * scores[:-1]
    raw_unit = unit * 1_000_003 - 7_000_000               # non-compact, negative ids
    raw_period = period + 2_000_000_000                   # far-away origin
    calls = []
    original = cv._lagged_sums_fft
    monkeypatch.setattr(cv, "_lagged_sums_fft", lambda *args: calls.append(1) or original(*args))
    gap = np.abs(period[:, None] - period[None, :])
    weight = np.vectorize(lambda d: kernel(name, d, lags))(gap) * (unit[:, None] == unit[None, :])
    expected = scores.T @ weight @ scores
    scale = np.sqrt(np.diag(scores.T @ scores))
    tolerance = 1e-11 * np.outer(scale, scale)
    ts, tu, tp = t(scores), codes(raw_unit), codes(raw_period)
    saved = (ts.clone(), tu.clone(), tp.clone())
    sorted_rows = cv.meat_hac(ts, lags, name, time=tp, panel=tu)
    assert len(calls) == 1 and (np.abs(sorted_rows.numpy() - expected) <= tolerance).all()
    order = rng.permutation(n)
    shuffled = cv.meat_hac(ts[order], lags, name, time=tp[order], panel=tu[order])
    assert len(calls) == 2 and (np.abs(shuffled.numpy() - expected) <= tolerance).all()
    assert all(torch.equal(a, b) for a, b in zip((ts, tu, tp), saved))
    assert torch.equal(sorted_rows, cv.meat_hac(ts, lags, name, time=tp, panel=tu))
    # One column per FFT block, and the offset scan, give the same meat.
    monkeypatch.setattr(cv, "_FFT_ELEMENTS", 1)
    blocked = cv.meat_hac(ts[order], lags, name, time=tp[order], panel=tu[order])
    assert len(calls) == 4 and (np.abs(blocked.numpy() - expected) <= tolerance).all()
    monkeypatch.setattr(cv, "_FFT_OFFSETS", 10 ** 9)
    looped = cv.meat_hac(ts[order], lags, name, time=tp[order], panel=tu[order])
    assert len(calls) == 4 and (np.abs(looped.numpy() - expected) <= tolerance).all()
    # Unit ids too wide to combine with the periods in one int64 key are renumbered.
    huge = cv.meat_hac(ts, lags, name, time=tp, panel=tu * (1 << 40))
    assert (np.abs(huge.numpy() - expected) <= tolerance).all()


def test_fft_and_offset_scan_agree_on_a_hundred_thousand_row_panel(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    n, k = 100_000, 6
    scores = torch.randn((n, k), dtype=torch.float64, generator=generator)
    scores[:, 2] *= 1e4
    index = torch.arange(n)
    unit, period = index // 200, index % 200                      # 500 units x 200 periods
    for name, lags in (("bartlett", 100), ("parzen", 90), ("quadratic_spectral", 8)):
        fast = cv.meat_hac(scores, lags, name, time=period, panel=unit)
        monkeypatch.setattr(cv, "_FFT_OFFSETS", 10 ** 9)
        slow = cv.meat_hac(scores, lags, name, time=period, panel=unit)
        monkeypatch.setattr(cv, "_FFT_OFFSETS", 64)
        scale = scores.square().sum(dim=0).sqrt()
        assert bool(((fast - slow).abs() <= 1e-11 * scale[:, None] * scale[None, :]).all()), name


def test_unbalanced_driscoll_kraay_matches_statsmodels_groupsum():
    import statsmodels.stats.sandwich_covariance as sw

    rng = np.random.default_rng(23)
    units, periods, lags = 13, 25, 4
    keep = rng.uniform(size=units * periods) > 0.3
    period = np.tile(np.arange(periods), units)[keep]
    assert len(np.unique(period)) == periods                        # every period observed
    scores = rng.normal(size=(len(period), 3)) + rng.normal(size=(periods, 3))[period]
    expected = sw.S_hac_groupsum(scores, period, nlags=lags)
    order = rng.permutation(len(period))
    pooled = cv.meat_driscoll_kraay(t(scores[order]), codes(period[order] + 1900), lags)
    assert_allclose(pooled.numpy(), expected, rtol=1e-11, atol=1e-11)


def test_hac_conventions_and_degenerate_inputs():
    rng = np.random.default_rng(12)
    scores = rng.normal(size=(9, 2))
    ts = t(scores)
    white = cv.meat_white(ts)
    total = scores.sum(axis=0)
    for name in KERNELS:
        # lags = 0 is the White meat for EVERY kernel (also the quadratic spectral one).
        assert torch.equal(cv.meat_hac(ts, 0, name), white)
        assert torch.equal(cv.meat_hac(ts, 0, name, time=codes(np.arange(9))), white)
        assert torch.equal(cv.meat_driscoll_kraay(ts, codes(np.arange(9) * 3), 0, name), white)
        # One row, one period, one row per unit: no autocovariance terms.
        assert_allclose(cv.meat_hac(ts[:1], 5, name, time=codes([7])).numpy(),
                        np.outer(scores[0], scores[0]), rtol=1e-14)
        lonely = cv.meat_hac(ts, 5, name, time=codes(np.zeros(9)), panel=codes(np.arange(9)))
        assert_allclose(lonely.numpy(), white.numpy(), rtol=1e-14)
        pooled = cv.meat_driscoll_kraay(ts, codes(np.full(9, -4)), 5, name)
        assert_allclose(pooled.numpy(), np.outer(total, total), rtol=1e-12, atol=1e-13)
        assert cv.meat_driscoll_kraay(ts[:0], codes([]), 3, name).tolist() == [[0.0] * 2] * 2
    # More lags than periods: Bartlett still uses bandwidth L + 1, truncated links all.
    many = cv.meat_hac(ts, 500, "truncated")
    assert_allclose(many.numpy(), np.outer(total, total), rtol=1e-12, atol=1e-12)
    wide = cv.meat_hac(ts, 500)
    assert_allclose(wide.numpy(), brute_hac(scores, np.arange(9), None, "bartlett", 500),
                    rtol=1e-12, atol=1e-13)
    # Scaling a column by a power of two scales the meat exactly; inputs are untouched.
    scaled = ts * t([2.0 ** 200, 2.0 ** -300])
    exact = cv.meat_hac(ts, 3, "parzen") * t(np.outer([2.0 ** 200, 2.0 ** -300],
                                                     [2.0 ** 200, 2.0 ** -300]))
    assert torch.equal(cv.meat_hac(scaled, 3, "parzen"), exact)
    assert np.array_equal(ts.numpy(), scores)
    extreme = codes([-2 ** 63, 2 ** 63 - 1, 0, 1, 2, 3, 4, 5, 6])
    raises("invalid_time", lambda: cv.meat_hac(ts, 2, time=extreme))
    raises("repeated_time_values",
           lambda: cv.meat_hac(ts, 2, time=codes([0, 1, 2, 3, 4, 5, 6, 7, 7])))
    raises("invalid_lags", lambda: cv.meat_hac(ts, True))
    raises("invalid_lags", lambda: cv.meat_driscoll_kraay(ts, codes(np.arange(9)), -1))
    raises("unsupported_kernel", lambda: cv.meat_driscoll_kraay(ts, codes(np.arange(9)), 1, "qs"))
    raises("invalid_scores", lambda: cv.meat_hac(scores, 1))


def test_quadratic_spectral_kernel_is_exact_to_rounding_at_every_angle():
    """Oracle: 40-term Taylor series of sin(a)/a - cos(a) in exact rational arithmetic."""
    def exact(z):
        a = Fraction(6) * Fraction(math.pi) * Fraction(z) / 5
        total = sum(Fraction((-1) ** (m + 1) * 2 * m, math.factorial(2 * m + 1)) * a ** (2 * m)
                    for m in range(1, 41))
        return float(3 * total / a ** 2)

    for lags, count in ((1, 4), (3, 9), (10, 30), (999, 300), (10 ** 6, 5), (10 ** 9, 3)):
        weights = cv.kernel_weights(lags, "quadratic_spectral", count=count).numpy()
        expected = [exact(d / (lags + 1)) for d in range(1, count + 1)]
        assert_allclose(weights, expected, rtol=1e-14)
    # The series / closed-form switch at a = 1 (lag 265 of bandwidth 1000) is seamless.
    weights = cv.kernel_weights(999, "quadratic_spectral", count=300).numpy()
    assert np.abs(np.diff(weights, 2)[250:280]).max() < 1e-5
    assert_allclose(cv.kernel_weights(10 ** 9, "quadratic_spectral", count=1).item(), 1.0,
                    rtol=1e-15)


def test_kernel_weights_shapes_limits_and_stata_bartlett():
    # Newey-West weights for L = 4: 1 - l/5 (Stata newey, lag(4)).
    assert_allclose(cv.kernel_weights(4).numpy(), [0.8, 0.6, 0.4, 0.2], rtol=1e-15)
    for name in KERNELS:
        for lags in (1, 3, 10):
            weights = cv.kernel_weights(lags, name, count=3 * lags).numpy()
            expected = [kernel(name, d, lags) for d in range(1, 3 * lags + 1)]
            assert_allclose(weights, expected, rtol=1e-12, atol=1e-15)
            assert np.abs(weights).max() <= 1.0
    # Parzen is continuous at z = 1/2 and the quadratic spectral kernel tends to 1 at 0.
    assert_allclose(cv.kernel_weights(1, "parzen").item(), 0.25, rtol=1e-15)
    assert_allclose(cv.kernel_weights(10 ** 6, "quadratic_spectral", count=1).item(), 1.0,
                    rtol=1e-9)
    assert cv.kernel_weights(5, "bartlett", count=0).shape == (0,)
    raises("invalid_lags", lambda: cv.kernel_weights(3, count=-1))
    assert [cv.newey_west_lags(n) for n in (1, 99, 100, 316, 1000, 10 ** 6)] == \
        [1, 3, 4, 5, 6, 30]


def test_sandwich_and_nearest_psd_validation():
    rng = np.random.default_rng(13)
    bread, root = rng.normal(size=(2, 3)), rng.normal(size=(3, 3))
    meat = root @ root.T
    result = cv.sandwich(t(bread), t(meat))
    assert result.shape == (2, 2) and torch.equal(result, result.T)
    assert_allclose(result.numpy(), bread @ meat @ bread.T, rtol=1e-12)
    raises("invalid_covariance", lambda: cv.sandwich(t(np.eye(3)), t(np.eye(2))))
    raises("invalid_covariance", lambda: cv.sandwich(torch.eye(2), t(np.eye(2))))
    raises("invalid_covariance", lambda: cv.sandwich(t(np.eye(2)), np.eye(2)))
    raises("invalid_covariance", lambda: cv.sandwich(t(np.ones(2)), t(np.eye(2))))
    raises("invalid_covariance", lambda: cv.nearest_psd(t([[np.nan, 0.0], [0.0, 1.0]])))
    fixed, adjusted = cv.nearest_psd(t([[1.0, 2.0], [2.0, 1.0]]))
    assert adjusted is True
    assert_allclose(fixed.numpy(), [[1.5, 1.5], [1.5, 1.5]], rtol=1e-13)
    zero, adjusted = cv.nearest_psd(torch.zeros((2, 2), dtype=torch.float64))
    assert adjusted is False and torch.equal(zero, torch.zeros((2, 2), dtype=torch.float64))
    raises("insufficient_clusters", lambda: cv.cluster_factor(10, 2, 1))
    raises("insufficient_observations", lambda: cv.cluster_factor(2, 2, 5))


# ------------------------------------------------------------------- large problems

def test_million_row_problems_stay_linear_in_time_and_memory():
    """n = 1e6: any n-by-n object (8e12 bytes) or per-row Python loop could not finish."""
    generator = torch.Generator().manual_seed(2026)
    n, k = 1_000_000, 12
    x = torch.randn((n, k), dtype=torch.float64, generator=generator)
    x[:, 0] = 1.0
    truth = torch.arange(1, k + 1, dtype=torch.float64) / k
    y = x @ truth + torch.randn(n, dtype=torch.float64, generator=generator)
    w = torch.rand(n, dtype=torch.float64, generator=generator) + 0.5
    firm = torch.randint(0, 20_000, (n,), generator=generator)
    year = torch.randint(0, 30, (n,), generator=generator)
    timings = {}

    def clock(label, call):
        start = time.perf_counter()
        value = call()
        timings[label] = time.perf_counter() - start
        return value

    fit = clock("least_squares (weights, leverage)",
                lambda: linalg.least_squares(x, y, w, need_leverage=True))
    gram = clock("weighted_crossprod", lambda: linalg.weighted_crossprod(x, w))
    moment = linalg.weighted_crossprod(x, w, y)
    normal = np.linalg.solve(gram.numpy(), moment.numpy())
    assert_allclose(fit.beta.numpy(), normal, rtol=1e-9)
    assert_allclose(fit.xtx_inv.numpy(), np.linalg.inv(gram.numpy()), rtol=1e-8, atol=1e-16)
    assert_allclose(fit.leverage.sum().item(), k, rtol=1e-10)
    assert abs(fit.beta.numpy() - truth.numpy()).max() < 0.02

    collinear = torch.cat([x, x[:, 1:3].sum(dim=1, keepdim=True)], dim=1)
    dropped = clock("least_squares (collinear column)",
                    lambda: linalg.least_squares(collinear, y, w))
    assert dropped.omitted == [k]
    assert_allclose(dropped.beta.numpy(), fit.beta.numpy(), rtol=1e-9)

    scores = x * (w * fit.resid)[:, None]
    white = clock("meat_white", lambda: cv.meat_white(scores))
    cluster = clock("meat_cluster", lambda: cv.meat_cluster(scores, firm, 20_000))
    two_way = clock("meat_multiway (2 dims)",
                    lambda: cv.meat_multiway(scores, [(firm, 20_000), (year, 30)], adjust="none"))
    by_year = cv.meat_cluster(scores, year, 30)
    both = cv.meat_cluster(scores, firm * 30 + year, 600_000)
    assert_allclose(two_way.meat.numpy(), (cluster + by_year - both).numpy(), rtol=1e-9, atol=1e-6)

    newey = clock("meat_hac (8 lags)", lambda: cv.meat_hac(scores, 8))
    period = torch.arange(n)
    order = torch.randperm(n, generator=generator)
    shuffled = clock("meat_hac (8 lags, shuffled rows + time)",
                     lambda: cv.meat_hac(scores[order], 8, time=period[order]))
    assert_allclose(shuffled.numpy(), newey.numpy(), rtol=1e-9, atol=1e-6)
    spectral = clock("meat_hac (quadratic spectral, all 1e6 lags)",
                     lambda: cv.meat_hac(scores, 8, "quadratic_spectral"))
    assert torch.isfinite(spectral).all() and float(spectral.diagonal().min()) > 0
    unit, within = period // 25, period % 25
    panel = clock("meat_hac (panel, 4 lags)",
                  lambda: cv.meat_hac(scores, 4, time=within, panel=unit))
    long_panel = clock("meat_hac (panel 1000x1000, 100 lags, FFT)",
                       lambda: cv.meat_hac(scores, 100, time=period % 1000, panel=period // 1000))
    assert torch.isfinite(long_panel).all() and float(long_panel.diagonal().min()) > 0
    kraay = clock("meat_driscoll_kraay (4 lags)",
                  lambda: cv.meat_driscoll_kraay(scores, year, 4))
    for meat in (white, cluster, newey, panel, kraay):
        assert meat.shape == (k, k) and torch.isfinite(meat).all()
    print("\n" + "\n".join(f"  {label:45s} {seconds * 1000:8.1f} ms"
                           for label, seconds in timings.items()))
    # Generous ceilings (typically 5-20x the measured times on a laptop).
    assert max(timings.values()) < 6.0, timings
    assert sum(timings.values()) < 12.0, timings
