"""Independent distribution oracles across tails and small/large sample sizes."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from openecon.engines.inference import critical_value, student_t_two_sided, two_sided_p_values


@pytest.mark.parametrize("df", [1, 2, 3, 5, 11, 30, 117, 1000, 100000])
def test_student_t_tails_match_independent_distribution(df):
    values = np.array([0., 1e-8, 1e-4, .1, .9, 1., 1.01, 2., 5., 8., 20., 40., 100.])
    actual = np.array([student_t_two_sided(v, df) for v in values])
    expected = 2 * stats.t.sf(values, df)
    if df == 1:
        # The exact Cauchy identity avoids SciPy 1.18's loss of accuracy for
        # t(1) extremely close to zero (its 1e-8 result differs by ~3e-9).
        expected = np.array([1 - 2 * math.atan(v) / math.pi if v <= 1
                             else 2 * math.atan(1 / v) / math.pi for v in values])
    np.testing.assert_allclose(actual, expected, rtol=8e-10, atol=1e-300)
    assert (np.diff(actual) <= 0).all()
    np.testing.assert_allclose([student_t_two_sided(-v, df) for v in values], actual, rtol=0, atol=0)


@pytest.mark.parametrize("df", [1, 2, 5, 30, 117, 1000, 100000])
@pytest.mark.parametrize("alpha", [.5, .10, .05, .01, 1e-4, 1e-12])
def test_student_t_quantiles_match_oracle_and_invert_tail(df, alpha):
    actual = critical_value(alpha, df)
    assert actual == pytest.approx(stats.t.isf(alpha / 2, df), rel=1e-9)
    assert student_t_two_sided(actual, df) == pytest.approx(alpha, rel=1e-9)


def test_tiny_tail_does_not_cancel_to_zero():
    expected = 2 * stats.t.sf(8.3, 117)
    actual = student_t_two_sided(8.3, 117)
    assert 0 < actual < 1e-12
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)


def test_normal_tails_and_quantiles_match_oracle():
    values = torch.tensor([-38., -8., -2., 0., 2., 8., 38.], dtype=torch.float64)
    np.testing.assert_allclose(two_sided_p_values(values), 2 * stats.norm.sf(values.abs()), rtol=5e-13, atol=1e-315)
    for alpha in [0.5, .1, .05, 1e-8, 1e-100]:
        assert critical_value(alpha) == pytest.approx(stats.norm.isf(alpha / 2), rel=1e-13)


def test_boundary_statistics_and_exact_cauchy_tail():
    assert student_t_two_sided(math.inf, 1) == 0
    for value in [.1, 1., 10., 1e6]:
        assert student_t_two_sided(value, 1) == pytest.approx(2 * math.atan(1 / value) / math.pi, rel=1e-13)
