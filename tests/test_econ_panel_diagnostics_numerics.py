"""Correlation unit invariance, independently checked on unscaled data."""
import numpy as np
import pytest
import torch

from openecon.econometrics.panel.kernels import correlation


@pytest.mark.parametrize("units", [(1e-150, 1e-150), (1e150, 1e150),
                                   (1e-150, 1e150), (1e150, 1e-150),
                                   (1e-300, 1e300), (1e300, 1e-300)])
@pytest.mark.parametrize("weighted", [False, True])
def test_pearson_correlation_matches_numpy_despite_independent_unit_changes(units, weighted):
    rng = np.random.default_rng(473)
    a = rng.normal(size=60)
    b = .35 * a + rng.normal(size=len(a))
    weights = rng.uniform(.2, 4, len(a)) if weighted else None
    if weights is None:
        expected = np.corrcoef(a, b)[0, 1]
    else:
        centered = np.column_stack([a - np.average(a, weights=weights),
                                    b - np.average(b, weights=weights)])
        covariance = np.cov(centered.T, aweights=weights, ddof=0)
        expected = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    arrays = [torch.tensor(a * units[0]), torch.tensor(b * units[1])]
    weight_tensor = None if weights is None else torch.tensor(weights)
    actual = correlation(*arrays, weight_tensor)
    assert actual == pytest.approx(expected, rel=2e-13, abs=1e-14)
    negative = correlation(-arrays[0], arrays[1], weight_tensor)
    assert negative == pytest.approx(-expected, rel=2e-13, abs=1e-14)


@pytest.mark.parametrize("unit", [1e-300, 1e300])
def test_pearson_weights_can_change_units_without_changing_the_result(unit):
    a = torch.tensor([.2, 1., -3., 7.], dtype=torch.float64)
    b = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    weights = torch.tensor([.3, .8, 2., 1.2], dtype=torch.float64)
    assert correlation(a, b, weights * unit) == pytest.approx(correlation(a, b, weights), rel=1e-13)


def test_constant_or_single_positive_weight_support_has_no_correlation():
    values = torch.tensor([.1, .4, .8], dtype=torch.float64)
    assert correlation(torch.ones_like(values) * 1e-200, values) is None
    assert correlation(values, torch.zeros_like(values)) is None
    # The differing first observation carries no weight.
    a = torch.tensor([100., .1, .1], dtype=torch.float64)
    weights = torch.tensor([0., .3, .8], dtype=torch.float64)
    assert correlation(a, values, weights) is None
    assert correlation(a, values, torch.tensor([0., 1., 0.], dtype=torch.float64)) is None


def test_large_finite_means_do_not_overflow_before_correlation():
    # Raw sums overflow; all original values and the reference coordinates are finite.
    a = np.array([.6, .7, .8, .9])
    b = np.array([.9, .8, .7, .6])
    actual = correlation(torch.tensor(a * 1e308), torch.tensor(b * 1e308))
    assert actual == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-14)
