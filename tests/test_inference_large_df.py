"""Independent large-sample Student-t checks; no large data allocations."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value, student_t_two_sided, two_sided_p_values


@pytest.mark.parametrize("df", [10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12])
def test_large_finite_df_tails_match_independent_oracle(df):
    values = np.array([0., 1e-12, 1e-8, .01, .1, .9, 1., 1.01,
                       1.96, 2., 5., 8., 12., 20., 30., 37.])
    actual = np.array([student_t_two_sided(value, df) for value in values])
    expected = 2 * stats.t.sf(values, df)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-314)
    assert (np.diff(actual) <= 0).all()
    np.testing.assert_equal([student_t_two_sided(-value, df) for value in values], actual)


@pytest.mark.parametrize("df", [10**6, 10**8, 10**10, 10**11, 10**12])
@pytest.mark.parametrize("alpha", [.5, .1, .05, .01, 1e-6, 1e-12, 1e-100, 1e-300])
def test_large_finite_df_quantiles_match_oracle_and_invert_tail(df, alpha):
    actual = critical_value(alpha, df)
    assert actual == pytest.approx(stats.t.isf(alpha / 2, df), rel=1e-12)
    assert student_t_two_sided(actual, df) == pytest.approx(alpha, rel=2e-11, abs=0)


def test_finite_df_corrections_are_not_silently_replaced_by_normal():
    df = 10**6
    for value in [1., 1.96, 5., 8.]:
        actual = student_t_two_sided(value, df)
        normal_tail = 2 * stats.norm.sf(value)
        assert actual > normal_tail
        assert actual - normal_tail == pytest.approx(
            2 * stats.t.sf(value, df) - normal_tail, rel=1e-8, abs=1e-15)
    assert critical_value(.05, df) > critical_value(.05)


def test_one_hundred_billion_df_regression_and_tensor_contract():
    df = 100_000_000_000
    values = torch.tensor([-8., -1., 0., 1., 8.], dtype=torch.float64)
    actual = two_sided_p_values(values, df)
    assert actual.dtype == torch.float64
    assert actual.device == values.device
    np.testing.assert_allclose(actual.numpy(), 2 * stats.t.sf(values.abs().numpy(), df),
                               rtol=2e-13, atol=0)
    assert actual[1].item() == pytest.approx(.31731050786533377, rel=2e-15)


def test_large_df_extreme_tails_stay_finite_and_preserve_subnormals():
    # SciPy's Student-t survival function underflows at t=38 in this version.
    # The finite-df Student-t tail is positive and exceeds the normal tail.
    assert 0 < student_t_two_sided(38., 10**12) < 1e-310
    assert student_t_two_sided(38., 10**12) >= math.erfc(38. / math.sqrt(2))
    for value in [40., 100., 1e100, 1e308, math.inf]:
        assert student_t_two_sided(value, 10**11) == 0
    assert student_t_two_sided(1e-300, 10**11) == 1
    with pytest.raises(KernelError, match="NaN"):
        student_t_two_sided(math.nan, 10**11)


def test_large_df_branch_boundary_matches_continued_fraction():
    for value in [.1, 1., 1.96, 5., 8.]:
        before = student_t_two_sided(value, 999_999)
        after = student_t_two_sided(value, 1_000_000)
        assert before == pytest.approx(2 * stats.t.sf(value, 999_999), rel=1e-9)
        assert after == pytest.approx(2 * stats.t.sf(value, 1_000_000), rel=2e-13)
