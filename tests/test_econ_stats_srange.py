"""Studentized range and Dunnett distributions against SciPy and exact identities."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from openecon.econometrics.stats import srange
from openecon.engines.contracts import KernelError


def test_ptukey_matches_scipy_over_groups_df_and_q():
    worst = 0.0
    for k in (2, 3, 6, 20, 100):
        for df in (2, 5, 30, 2000):
            q = np.array([0.5, 2.0, 3.5, 5.0, 7.0, 12.0])
            mine = srange.ptukey_sf(torch.tensor(q), k, float(df)).numpy()
            reference = stats.studentized_range.sf(q, k, df)
            worst = max(worst, float(np.abs(mine - reference).max()))
            cdf = srange.ptukey(torch.tensor(q), k, float(df)).numpy()
            worst = max(worst, float(np.abs(cdf - stats.studentized_range.cdf(q, k, df)).max()))
    # SciPy's own quadrature is accurate to about 1e-10.
    assert worst < 5e-10


@pytest.mark.parametrize("df", [1.0, 1.5, 2.0, 2.5, 7.3, 50.0, 1e4, 1e6])
def test_two_groups_reduce_to_student_t_for_any_fractional_df(df):
    q = np.array([0.3, 2.0, 5.0, 20.0])
    mine = srange.ptukey_sf(torch.tensor(q), 2, df).numpy()
    exact = 2.0 * stats.t.sf(q / math.sqrt(2.0), df)
    np.testing.assert_allclose(mine, exact, rtol=2e-11)


def test_infinite_df_is_the_range_of_normals():
    q = np.array([0.5, 2.0, 4.0, 7.0])
    mine = srange.ptukey_sf(torch.tensor(q), 2, math.inf).numpy()
    np.testing.assert_allclose(mine, 2.0 * stats.norm.sf(q / math.sqrt(2.0)), rtol=1e-12)

    # k = 3 by direct quadrature of k * phi(z) * [Phi(z) - Phi(z - w)]^(k-1).
    def tail(w: float, k: int) -> float:
        value = integrate.quad(lambda z: k * stats.norm.pdf(z)
                               * (stats.norm.cdf(z) - stats.norm.cdf(z - w)) ** (k - 1),
                               -12, 12, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
        return 1.0 - value

    for k in (3, 12):
        mine = srange.ptukey_sf(torch.tensor([1.0, 3.0, 5.0]), k, math.inf).numpy()
        np.testing.assert_allclose(mine, [tail(w, k) for w in (1.0, 3.0, 5.0)], atol=2e-12)


def test_small_tails_keep_relative_accuracy():
    # k = 2 has the closed form 2 * t.sf; far tails must not be lost to cancellation.
    mine = float(srange.ptukey_sf(20.0, 2, 1e6)[0])
    exact = 2.0 * stats.t.sf(20.0 / math.sqrt(2.0), 1e6)
    assert exact < 1e-40 and abs(mine - exact) / exact < 1e-10
    assert float(srange.ptukey_sf(0.0, 5, 10.0)[0]) == 1.0
    assert float(srange.ptukey_sf(500.0, 5, 1000.0)[0]) == 0.0
    # Every regime of (df, q): tiny df with huge q, huge df, and tails down to 1e-250.
    q = np.array([1e-6, 1e-3, 0.1, 1, 5, 10, 20, 30, 40, 48, 100, 1e3, 1e4])
    for df in (1, 2, 3, 10, 30, 100, 1e3, 1e4, 1e5, 1e7, 1e9):
        exact = 2.0 * stats.t.sf(q / math.sqrt(2.0), df)
        keep = exact > 1e-250
        mine = srange.ptukey_sf(torch.tensor(q), 2, float(df)).numpy()
        np.testing.assert_allclose(mine[keep], exact[keep], rtol=1e-10)
        assert (mine >= 0).all() and (mine <= 1).all()
    d = np.array([1e-3, 0.1, 1, 3, 6, 10, 20, 30, 36, 100, 1e3])
    for df in (1, 2, 10, 100, 1e3, 1e4, 1e6):
        exact = 2.0 * stats.t.sf(d, df)
        keep = exact > 1e-250
        mine = srange.pdunnett_sf(torch.tensor(d), [0.6], float(df)).numpy()
        np.testing.assert_allclose(mine[keep], exact[keep], rtol=1e-10)


@pytest.mark.parametrize(("k", "df", "p"), [
    (3, 10, 0.95), (5, 20, 0.99), (10, 2, 0.95), (100, 1000, 0.95), (4, 3.7, 0.9),
    (3, 2, 0.999), (2, 1, 0.5), (20, 60, 0.05)])
def test_qtukey_matches_scipy_and_inverts(k, df, p):
    q = float(srange.qtukey(p, k, float(df))[0])
    assert q == pytest.approx(stats.studentized_range.ppf(p, k, df), rel=2e-9)
    assert float(srange.ptukey(q, k, float(df))[0]) == pytest.approx(p, abs=1e-11)


def test_vectorized_over_q_and_df():
    q = torch.linspace(0.2, 8.0, 300, dtype=torch.float64)
    df = torch.linspace(3.0, 90.0, 300, dtype=torch.float64)
    batch = srange.ptukey_sf(q, 6, df)
    assert batch.shape == (300,) and batch.dtype == torch.float64
    single = torch.stack([srange.ptukey_sf(float(a), 6, float(b))[0]
                          for a, b in zip(q[::37], df[::37], strict=True)])
    np.testing.assert_allclose(batch[::37].numpy(), single.numpy(), rtol=1e-13)
    critical = srange.qtukey(0.95, 6, df)
    np.testing.assert_allclose(srange.ptukey_sf(critical, 6, df).numpy(), 0.05, rtol=1e-9)
    assert bool((critical[1:] <= critical[:-1]).all())      # decreasing in df


@pytest.mark.parametrize(("call", "args"), [
    (srange.ptukey_sf, (1.0, 1, 10.0)), (srange.ptukey_sf, (1.0, 3, 0.5)),
    (srange.ptukey_sf, (float("nan"), 3, 10.0)), (srange.ptukey_sf, (float("inf"), 3, 10.0)),
    (srange.qtukey, (1.0, 3, 10.0)), (srange.qtukey, (0.0, 3, 10.0)),
    (srange.ptukey_sf, (1.0, 2.5, 10.0)), (srange.pdunnett_sf, (1.0, [1.0], 10.0)),
    (srange.pdunnett_sf, (1.0, [], 10.0)), (srange.qdunnett, (1.5, [0.5], 10.0))])
def test_invalid_arguments_raise_kernel_errors(call, args):
    with pytest.raises(KernelError) as error:
        call(*args)
    assert error.value.code == "invalid_distribution_argument"


def _dunnett_oracle(d: float, lambdas: list[float], df: float) -> float:
    lam = np.asarray(lambdas)
    scale = np.sqrt(1.0 - lam ** 2)

    def inner(s: float) -> float:
        def integrand(u: float) -> float:
            return stats.norm.pdf(u) * np.prod(stats.norm.cdf((lam * u + d * s) / scale)
                                               - stats.norm.cdf((lam * u - d * s) / scale))
        return integrate.quad(integrand, -9, 9, epsabs=1e-11, epsrel=1e-11, limit=200)[0]

    def outer(s: float) -> float:
        return stats.chi.pdf(s * math.sqrt(df), df) * math.sqrt(df) * inner(s)

    upper = 1.0 + 9.0 / math.sqrt(df)
    return 1.0 - integrate.quad(outer, 0, upper, epsabs=1e-11, epsrel=1e-10, limit=200)[0]


@pytest.mark.parametrize(("lambdas", "df", "d"), [
    ([math.sqrt(0.5)] * 2, 10, 2.57), ([0.5, 0.8, 0.6], 7.5, 2.0),
    ([0.9, 0.3, 0.7, 0.7], 30, 3.1), ([0.99, 0.2], 4, 1.2)])
def test_dunnett_matches_an_independent_double_quadrature(lambdas, df, d):
    mine = float(srange.pdunnett_sf(d, lambdas, float(df))[0])
    assert mine == pytest.approx(_dunnett_oracle(d, lambdas, df), abs=1e-9)


def test_dunnett_with_one_comparison_is_student_t_and_quantile_inverts():
    d = np.array([0.5, 2.5, 6.0])
    mine = srange.pdunnett_sf(torch.tensor(d), [0.6], 5.0).numpy()
    np.testing.assert_allclose(mine, 2.0 * stats.t.sf(d, 5), rtol=1e-11)
    lambdas = [math.sqrt(0.5)] * 3
    critical = float(srange.qdunnett(0.95, lambdas, 20.0)[0])
    # Dunnett's (1955) two-sided table: 2.54 for three treatments and 20 error df.
    assert round(critical, 2) == 2.54
    assert float(srange.pdunnett_sf(critical, lambdas, 20.0)[0]) == pytest.approx(0.05, abs=1e-11)


def test_dunnett_refuses_extreme_imbalance():
    with pytest.raises(KernelError) as error:
        srange.pdunnett_sf(2.0, [math.sqrt(100000 / 100002)], 30.0)
    assert error.value.code == "dunnett_unbalanced"
