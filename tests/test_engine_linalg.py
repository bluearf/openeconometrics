"""Independent numerical expectations for the shared least-squares kernel.

Oracles: numpy.linalg (lstsq, inv, explicit hat matrices), statsmodels WLS, and a
brute-force sequential collinearity screen that works on the full n-row design.
"""

import numpy as np
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose

from openecon.engines import linalg
from openecon.engines.contracts import KernelError


def t(values):
    return torch.tensor(np.asarray(values), dtype=torch.float64)


def sequential_screen(x, w=None, tol=1e-13):
    """Oracle: regress each column on the previously kept ones with numpy's QR."""
    a = x if w is None else x * np.sqrt(w)[:, None]
    kept, omitted = [], []
    for j in range(a.shape[1]):
        column = a[:, j]
        residual = column
        if kept:
            basis = np.linalg.qr(a[:, kept])[0]
            residual = column - basis @ (basis.T @ column)
            residual = residual - basis @ (basis.T @ residual)
        if residual @ residual <= tol * (column @ column):
            omitted.append(j)
        else:
            kept.append(j)
    return kept, omitted


@pytest.fixture
def data():
    rng = np.random.default_rng(20261002)
    n = 150
    x = np.column_stack(
        [np.ones(n), rng.normal(size=(n, 4)) * [1.0, 3.0, 0.2, 10.0] + [0, 2, -1, 5]])
    y = x @ np.array([1.0, 0.5, -2.0, 4.0, 0.03]) + rng.normal(size=n) * (1 + np.abs(x[:, 1]))
    w = rng.uniform(0.2, 3.0, size=n)
    w[[3, 17, 90]] = 0.0
    return x, y, w


def test_ols_matches_lstsq_normal_equations_and_hat_matrix(data):
    x, y, _ = data
    n, k = x.shape
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    bread = np.linalg.inv(x.T @ x)
    resid = y - x @ beta
    assert fit.kept == list(range(k)) and fit.omitted == [] and fit.rank == k
    assert_allclose(fit.beta.numpy(), beta, rtol=1e-11, atol=1e-12)
    assert_allclose(fit.xtx_inv.numpy(), bread, rtol=1e-10, atol=1e-15)
    assert_allclose(fit.fitted.numpy(), x @ beta, rtol=1e-12, atol=1e-11)
    assert_allclose(fit.resid.numpy(), resid, rtol=1e-9, atol=1e-11)
    assert_allclose(fit.ssr.item(), resid @ resid, rtol=1e-12)
    assert fit.ssr.ndim == 0 and fit.beta.shape == (k,)
    assert_allclose(fit.leverage.numpy(), np.diag(x @ bread @ x.T), rtol=1e-11, atol=1e-14)
    assert_allclose(fit.leverage.sum().item(), k, rtol=1e-12)
    assert_allclose(fit.condition_number, np.linalg.cond(x / np.linalg.norm(x, axis=0)), rtol=1e-10)
    assert torch.equal(fit.xtx_inv, fit.xtx_inv.T)
    assert linalg.least_squares(t(x), t(y)).leverage is None


def test_wls_matches_statsmodels_and_weighted_hat_matrix(data):
    x, y, w = data
    fit = linalg.least_squares(t(x), t(y), t(w), need_leverage=True)
    oracle = sm.WLS(y, x, weights=w).fit()
    assert_allclose(fit.beta.numpy(), oracle.params, rtol=1e-11, atol=1e-12)
    assert_allclose(fit.xtx_inv.numpy(), oracle.normalized_cov_params, rtol=1e-10, atol=1e-15)
    assert_allclose(fit.xtx_inv.numpy(), np.linalg.inv(x.T @ (x * w[:, None])), rtol=1e-10)
    # resid is NOT weighted, ssr is the weighted sum of squares.
    assert_allclose(fit.resid.numpy(), y - x @ oracle.params, rtol=1e-9, atol=1e-11)
    assert_allclose(fit.fitted.numpy(), x @ oracle.params, rtol=1e-12, atol=1e-11)
    assert_allclose(fit.ssr.item(), oracle.ssr, rtol=1e-12)
    hat = w * np.einsum("ij,jk,ik->i", x, np.linalg.inv(x.T @ (x * w[:, None])), x)
    assert_allclose(fit.leverage.numpy(), hat, rtol=1e-11, atol=1e-14)
    assert (fit.leverage.numpy()[w == 0] < 1e-25).all()
    scaled = x * np.sqrt(w)[:, None]
    assert_allclose(fit.condition_number,
                    np.linalg.cond(scaled / np.linalg.norm(scaled, axis=0)), rtol=1e-10)


def test_multiple_right_hand_sides_share_one_factorization(data):
    x, y, w = data
    rng = np.random.default_rng(5)
    ys = np.column_stack([y, rng.normal(size=len(y)), x[:, 2] ** 2])
    fit = linalg.least_squares(t(x), t(ys), t(w), need_leverage=True)
    assert fit.beta.shape == (5, 3) and fit.fitted.shape == ys.shape and fit.ssr.shape == (3,)
    for column in range(3):
        single = linalg.least_squares(t(x), t(ys[:, column]), t(w))
        oracle = sm.WLS(ys[:, column], x, weights=w).fit()
        assert_allclose(fit.beta[:, column].numpy(), oracle.params, rtol=1e-11, atol=1e-12)
        assert_allclose(fit.ssr[column].item(), oracle.ssr, rtol=1e-12)
        assert_allclose(fit.beta[:, column].numpy(), single.beta.numpy(), rtol=1e-13, atol=1e-14)
        assert_allclose(fit.resid[:, column].numpy(), single.resid.numpy(), rtol=1e-10, atol=1e-11)


def test_badly_scaled_columns_are_solved_to_full_precision():
    rng = np.random.default_rng(11)
    n = 200
    z = np.column_stack([np.ones(n), rng.normal(size=(n, 5))])
    y = z @ np.array([1.0, -2.0, 0.5, 3.0, 0.25, -1.5]) + rng.normal(size=n)
    units = np.array([1e-8, 1e8, 1.0, 1e-4, 1e6, 1e3])
    x = z * units
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    bread = np.linalg.inv(z.T @ z)
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    assert fit.omitted == []
    assert_allclose(fit.beta.numpy(), beta / units, rtol=1e-11)
    assert_allclose(fit.xtx_inv.numpy(), bread / np.outer(units, units), rtol=1e-10)
    assert_allclose(fit.leverage.numpy(), np.diag(z @ bread @ z.T), rtol=1e-11)
    assert_allclose(fit.resid.numpy(), y - z @ beta, rtol=1e-8, atol=1e-10)
    # The condition number is that of the unit-norm design, whatever the units.
    assert_allclose(fit.condition_number, np.linalg.cond(z / np.linalg.norm(z, axis=0)), rtol=1e-9)
    assert fit.condition_number < 10


def test_power_of_two_units_are_unscaled_exactly(data):
    x, y, w = data
    units = np.array([2.0 ** -40, 2.0 ** 33, 1.0, 2.0 ** -7, 2.0 ** 60])
    base = linalg.least_squares(t(x), t(y), t(w))
    scaled = linalg.least_squares(t(x * units), t(y), t(w))
    assert torch.equal(scaled.beta * t(units), base.beta)
    assert torch.equal(scaled.xtx_inv * t(np.outer(units, units)), base.xtx_inv)
    assert scaled.condition_number == base.condition_number


def test_extreme_magnitudes_do_not_overflow():
    rng = np.random.default_rng(3)
    z = np.column_stack([np.ones(60), rng.normal(size=(60, 2))])
    y = z @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=60)
    units = np.array([1e-120, 1e150, 1.0])      # squares of the entries leave float64 range
    fit = linalg.least_squares(t(z * units), t(y))
    assert fit.omitted == []
    assert_allclose(fit.beta.numpy() * units, np.linalg.lstsq(z, y, rcond=None)[0], rtol=1e-11)
    assert_allclose(fit.xtx_inv.numpy() * np.outer(units, units), np.linalg.inv(z.T @ z),
                    rtol=1e-10)
    # (X'X)^{-1} itself is not representable once a unit is below ~1e-154.
    with pytest.raises(KernelError) as error:
        linalg.least_squares(t(z * np.array([1e-170, 1.0, 1.0])), t(y))
    assert error.value.code == "numerical_failure"


def _columns():
    rng = np.random.default_rng(77)
    n = 80
    a, b, c, noise = rng.normal(size=(4, n))
    return n, a, b, c, noise, np.ones(n), np.zeros(n)


def _collinear_cases():
    n, a, b, c, noise, one, zero = _columns()
    return {
        "duplicate_drops_later_copy": ([one, a, b, a], [3]),
        "combination_of_three": ([one, a, b, c, 2 * a - 3 * b + 0.5 * c], [4]),
        "constant_after_intercept": ([one, a, 5 * one], [2]),
        "intercept_after_constant": ([5 * one, a, one], [2]),
        "all_zero_column": ([one, zero, a], [1]),
        "leading_zero_column": ([zero, one, a], [0]),
        "several_dependent_at_once": (
            [one, a, b, a + b, 3 * one, c, 2 * c, a + c, zero, a - b + c + one],
            [3, 4, 6, 7, 8, 9]),
        "dependence_on_later_columns_keeps_earlier": ([a + b, one, a, b, c], [3]),
        "badly_scaled_dependence": ([one, 1e7 * a, 1e-6 * b, 3e7 * a - 2e-6 * b + one, c], [3]),
        "near_collinear_is_kept": ([one, a, a + 1e-6 * noise], []),
        "near_collinear_then_exact": ([one, a, a + 1e-6 * noise, 2 * a + one, b], [3]),
    }


@pytest.mark.parametrize("name", list(_collinear_cases()))
def test_collinearity_is_resolved_left_to_right(name):
    columns, expected = _collinear_cases()[name]
    x = np.column_stack(columns)
    k = x.shape[1]
    kept, omitted = linalg.collinear_columns(t(x))
    assert omitted == expected
    assert kept == [j for j in range(k) if j not in expected]
    assert (kept, omitted) == sequential_screen(x)

    rng = np.random.default_rng(1)
    y = rng.normal(size=len(x))
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    assert fit.kept == kept and fit.omitted == omitted and fit.rank == len(kept)
    assert fit.beta.shape == (len(kept),) and fit.xtx_inv.shape == (len(kept), len(kept))
    reduced = x[:, kept]
    scale = np.linalg.norm(reduced, axis=0)
    beta = np.linalg.lstsq(reduced / scale, y, rcond=None)[0] / scale
    loose = 1e-4 if "near" in name else 1e-9      # near-collinear: condition ~ 1e6
    assert_allclose(fit.beta.numpy(), beta, rtol=loose, atol=loose)
    assert_allclose(fit.fitted.numpy(), reduced @ beta, rtol=1e-8, atol=1e-8)
    assert_allclose(fit.leverage.sum().item(), len(kept), rtol=1e-10)
    projector = np.linalg.qr(reduced / scale)[0]
    assert_allclose(fit.leverage.numpy(), (projector ** 2).sum(axis=1), rtol=1e-7, atol=1e-9)
    if "near" not in name:
        gram = (reduced / scale).T @ (reduced / scale)
        assert_allclose(fit.xtx_inv.numpy(), np.linalg.inv(gram) / np.outer(scale, scale),
                        rtol=1e-8)


@pytest.mark.parametrize("seed", range(12))
def test_random_rank_deficient_designs_match_the_sequential_oracle(seed):
    rng = np.random.default_rng(seed)
    n, k = 60, 14
    rank = int(rng.integers(3, 9))
    basis = rng.normal(size=(n, rank)) * 10.0 ** rng.integers(-3, 4, size=rank)
    mixing = rng.integers(-2, 3, size=(rank, k)).astype(float)
    mixing[:, rng.integers(0, k)] = 0.0
    x = basis @ mixing
    w = rng.uniform(0.5, 2.0, size=n) if seed % 2 else None
    expected = sequential_screen(x, w)
    assert len(expected[0]) == np.linalg.matrix_rank(mixing)
    weights = None if w is None else t(w)
    assert linalg.collinear_columns(t(x), weights) == expected
    fit = linalg.least_squares(t(x), t(rng.normal(size=n)), weights)
    assert (fit.kept, fit.omitted) == expected


def test_weights_can_create_collinearity():
    rng = np.random.default_rng(8)
    n = 40
    a = rng.normal(size=n)
    twin = a.copy()
    twin[:5] += rng.normal(size=5)
    x = np.column_stack([np.ones(n), a, twin])
    w = np.ones(n)
    w[:5] = 0.0
    assert linalg.collinear_columns(t(x)) == ([0, 1, 2], [])
    assert linalg.collinear_columns(t(x), t(w)) == ([0, 1], [2])
    fit = linalg.least_squares(t(x), t(rng.normal(size=n)), t(w))
    assert fit.omitted == [2] and fit.fitted.shape == (n,)


def test_tolerance_is_a_ratio_of_sums_of_squares():
    rng = np.random.default_rng(21)
    n = 50
    a = rng.normal(size=n)
    a /= np.linalg.norm(a)
    e = rng.normal(size=n)
    e -= a * (a @ e)
    e /= np.linalg.norm(e)
    for delta, tol, dropped in [(2e-3, 1e-6, False), (5e-4, 1e-6, True),
                                (1e-6, 1e-13, False), (1e-7, 1e-13, True),
                                (1e-9, 1e-20, False), (1e-11, 1e-20, True)]:
        x = t(np.column_stack([a, a + delta * e]))     # ratio = delta^2 / (1 + delta^2)
        assert linalg.collinear_columns(x, tol=tol)[1] == ([1] if dropped else [])
    assert linalg.collinear_columns(t(np.column_stack([a, a + 1e-6 * e])))[1] == []
    assert linalg.collinear_columns(t(np.column_stack([a, a + 1e-7 * e])))[1] == [1]


def test_raw_year_quadratic_is_kept_and_estimated_accurately():
    # year and year^2 for 1995..2015 have an uncentered ratio near 7e-11: a common
    # specification that must survive the default screen (the cubic does not).
    rng = np.random.default_rng(40)
    year = rng.integers(1995, 2016, size=2000).astype(float)
    centered = year - 2005
    y = 1 + 0.5 * centered - 0.02 * centered ** 2 + rng.normal(size=2000)
    x = np.column_stack([np.ones(2000), year, year ** 2, year ** 3])
    assert linalg.collinear_columns(t(x)) == ([0, 1, 2], [3])
    fit = linalg.least_squares(t(x[:, :3]), t(y), drop_collinear=False)
    design = np.column_stack([np.ones(2000), centered, centered ** 2])
    oracle = np.linalg.lstsq(design, y, rcond=None)[0]
    assert_allclose(fit.beta[2].item(), oracle[2], rtol=1e-8)
    assert_allclose(fit.fitted.numpy(), design @ oracle, rtol=1e-8, atol=1e-8)
    standard_error = np.sqrt(np.linalg.inv(design.T @ design)[2, 2])
    assert_allclose(np.sqrt(fit.xtx_inv[2, 2].item()), standard_error, rtol=1e-7)


def test_more_columns_than_rows_keeps_the_first_independent_ones():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 6))
    y = rng.normal(size=4)
    assert linalg.collinear_columns(t(x)) == ([0, 1, 2, 3], [4, 5])
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    assert fit.kept == [0, 1, 2, 3] and fit.rank == 4
    assert_allclose(fit.beta.numpy(), np.linalg.solve(x[:, :4], y), rtol=1e-9)
    assert_allclose(fit.resid.numpy(), 0, atol=1e-11)
    assert_allclose(fit.leverage.numpy(), 1, rtol=1e-12)


def test_drop_collinear_false_raises_singular_design(data):
    x, y, _ = data
    singular = np.column_stack([x, x[:, 1] - x[:, 2]])
    with pytest.raises(KernelError) as error:
        linalg.least_squares(t(singular), t(y), drop_collinear=False)
    assert error.value.code == "singular_design"
    assert linalg.least_squares(t(x), t(y), drop_collinear=False).rank == x.shape[1]
    assert linalg.least_squares(t(singular), t(y)).omitted == [5]


def test_empty_fits_return_the_outcome_as_residual():
    y = t([1.0, -2.0, 3.0])
    for x in (torch.zeros((3, 2), dtype=torch.float64), torch.zeros((3, 0), dtype=torch.float64)):
        fit = linalg.least_squares(x, y, t([1.0, 2.0, 0.5]), need_leverage=True)
        assert fit.rank == 0 and fit.kept == [] and fit.omitted == list(range(x.shape[1]))
        assert fit.beta.shape == (0,) and fit.xtx_inv.shape == (0, 0)
        assert torch.equal(fit.resid, y)
        assert torch.equal(fit.leverage, torch.zeros(3, dtype=torch.float64))
        assert_allclose(fit.ssr.item(), 1 + 8 + 4.5)
    assert linalg.collinear_columns(torch.zeros((3, 0), dtype=torch.float64)) == ([], [])


def test_invalid_inputs_raise_kernel_errors(data):
    x, y, w = data
    cases = [
        ("invalid_design", lambda: linalg.least_squares(t(x).float(), t(y))),
        ("invalid_design", lambda: linalg.least_squares(t(x), t(y[:-1]))),
        ("invalid_design", lambda: linalg.least_squares(t(x[:, 0]), t(y))),
        ("invalid_weights", lambda: linalg.least_squares(t(x), t(y), t(-w))),
        ("invalid_weights", lambda: linalg.least_squares(t(x), t(y), t(w[:-1]))),
        ("invalid_solver_options", lambda: linalg.collinear_columns(t(x), tol=1.5)),
    ]
    bad = x.copy()
    bad[7, 2] = np.nan
    cases.append(("non_finite_values", lambda: linalg.least_squares(t(bad), t(y))))
    bad_y = y.copy()
    bad_y[0] = np.inf
    cases.append(("non_finite_values", lambda: linalg.least_squares(t(x), t(bad_y))))
    for code, call in cases:
        with pytest.raises(KernelError) as error:
            call()
        assert error.value.code == code


def test_results_are_ordinary_float64_tensors_without_autograd(data):
    x, y, w = data
    fit = linalg.least_squares(t(x), t(y), t(w), need_leverage=True)
    for value in (fit.beta, fit.xtx_inv, fit.fitted, fit.resid, fit.ssr, fit.leverage):
        assert value.dtype == torch.float64 and not value.requires_grad
        assert not value.is_inference()
    fit.beta.add_(1.0)      # callers may update results in place


def test_cholesky_inverse_and_solve_match_numpy():
    rng = np.random.default_rng(31)
    root = rng.normal(size=(6, 6))
    a = root @ root.T + 0.5 * np.eye(6)
    b = rng.normal(size=(6, 3))
    inverse = linalg.cholesky_inverse(t(a))
    assert_allclose(inverse.numpy(), np.linalg.inv(a), rtol=1e-10)
    assert torch.equal(inverse, inverse.T)
    assert_allclose(linalg.cholesky_solve(t(a), t(b)).numpy(), np.linalg.solve(a, b), rtol=1e-10)
    vector = linalg.cholesky_solve(t(a), t(b[:, 0]))
    assert vector.shape == (6,)
    assert_allclose(vector.numpy(), np.linalg.solve(a, b[:, 0]), rtol=1e-10)

    indefinite = a - 50 * np.eye(6)
    singular = np.ones((3, 3))
    for matrix in (indefinite, singular):
        with pytest.raises(KernelError) as error:
            linalg.cholesky_inverse(t(matrix), code="singular_information", what="information")
        assert error.value.code == "singular_information" and "information" in str(error.value)
        with pytest.raises(KernelError) as error:
            linalg.cholesky_solve(t(matrix), t(np.ones(len(matrix))))
        assert error.value.code == "singular_matrix"
    with pytest.raises(KernelError):
        linalg.cholesky_inverse(t(np.ones((2, 3))))
    with pytest.raises(KernelError):
        linalg.cholesky_solve(t(a), t(np.ones(5)))


def test_symmetrize_and_weighted_crossprod(monkeypatch):
    rng = np.random.default_rng(13)
    n = 500
    x = rng.normal(size=(n, 4))
    z = rng.normal(size=(n, 3))
    w = rng.normal(size=n)          # signed weights are allowed
    a = rng.normal(size=(4, 4))
    assert_allclose(linalg.symmetrize(t(a)).numpy(), (a + a.T) / 2, rtol=0, atol=0)

    def check():
        gram = linalg.weighted_crossprod(t(x), t(w))
        assert_allclose(gram.numpy(), x.T @ np.diag(w) @ x, rtol=1e-11, atol=1e-11)
        assert torch.equal(gram, gram.T)
        assert_allclose(linalg.weighted_crossprod(t(x), t(w), t(z)).numpy(),
                        x.T @ np.diag(w) @ z, rtol=1e-11, atol=1e-11)
        moment = linalg.weighted_crossprod(t(x), t(w), t(z[:, 0]))
        assert moment.shape == (4,)
        assert_allclose(moment.numpy(), x.T @ (w * z[:, 0]), rtol=1e-11, atol=1e-11)
        assert_allclose(linalg.weighted_crossprod(t(x), None).numpy(), x.T @ x, rtol=1e-12)
        assert_allclose(linalg.weighted_crossprod(t(x), None, t(z)).numpy(), x.T @ z, rtol=1e-12,
                        atol=1e-12)

    check()
    monkeypatch.setattr(linalg, "_BLOCK_ELEMENTS", 4 * 37)    # force the row-block path
    check()
    with pytest.raises(KernelError):
        linalg.weighted_crossprod(t(x), t(w[:-1]))
    with pytest.raises(KernelError):
        linalg.weighted_crossprod(t(x), t(w), t(z[:-1]))


def test_quadratic_form_both_conventions():
    rng = np.random.default_rng(17)
    root = rng.normal(size=(5, 5))
    a = root @ root.T + np.eye(5)
    v = rng.normal(size=5)
    assert_allclose(linalg.quadratic_form(t(v), t(a), inverse=True), v @ np.linalg.solve(a, v),
                    rtol=1e-12)
    assert_allclose(linalg.quadratic_form(t(v), t(np.linalg.inv(a)), inverse=False),
                    v @ np.linalg.solve(a, v), rtol=1e-11)
    assert isinstance(linalg.quadratic_form(t(v), t(a), inverse=True), float)
    with pytest.raises(KernelError) as error:
        linalg.quadratic_form(t(v), t(a - 100 * np.eye(5)), inverse=True)
    assert error.value.code == "singular_matrix"
    with pytest.raises(KernelError):
        linalg.quadratic_form(t(v[:-1]), t(a), inverse=False)


def test_wald_statistic_matches_explicit_inverse():
    rng = np.random.default_rng(19)
    root = rng.normal(size=(5, 5))
    v = root @ root.T + 0.1 * np.eye(5)
    b = rng.normal(size=5)
    statistic, rank = linalg.wald_statistic(t(b), t(v))
    assert rank == 5 and isinstance(statistic, float) and isinstance(rank, int)
    assert_allclose(statistic, b @ np.linalg.solve(v, b), rtol=1e-11)
    # Parameters in wildly different units leave the statistic unchanged.
    units = np.array([1e-9, 1e7, 1.0, 1e3, 1e-4])
    scaled, rank = linalg.wald_statistic(t(b * units), t(v * np.outer(units, units)))
    assert rank == 5
    assert_allclose(scaled, statistic, rtol=1e-10)


def test_wald_statistic_with_singular_covariance_drops_redundant_constraints():
    rng = np.random.default_rng(23)
    root = rng.normal(size=(4, 4))
    sigma = root @ root.T + 0.2 * np.eye(4)
    beta = rng.normal(size=4)
    independent = rng.normal(size=(2, 4))
    # Rows 3 and 4 are combinations of rows 1 and 2: two redundant constraints.
    constraints = np.vstack([independent, independent[0] - 2 * independent[1], 3 * independent[1]])
    statistic, rank = linalg.wald_statistic(t(constraints @ beta),
                                            t(constraints @ sigma @ constraints.T))
    reduced = independent @ beta
    expected = reduced @ np.linalg.solve(independent @ sigma @ independent.T, reduced)
    assert rank == 2
    assert_allclose(statistic, expected, rtol=1e-9)

    # A zero-variance coordinate (e.g. an omitted coefficient) is dropped as well.
    padded = np.zeros((3, 3))
    padded[:2, :2] = independent @ sigma @ independent.T
    statistic, rank = linalg.wald_statistic(t(np.append(reduced, 0.0)), t(padded))
    assert rank == 2
    assert_allclose(statistic, expected, rtol=1e-10)

    for code, estimate, covariance in [
        ("singular_covariance", np.ones(2), np.zeros((2, 2))),
        ("invalid_covariance", np.ones(2), np.array([[1.0, 0.0], [0.0, -1.0]])),
        ("invalid_covariance", np.ones(3), np.eye(2)),
    ]:
        with pytest.raises(KernelError) as error:
            linalg.wald_statistic(t(estimate), t(covariance))
        assert error.value.code == code
