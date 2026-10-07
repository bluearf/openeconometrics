import pytest
import torch

from openecon.econometrics.arima.filters import apply_polynomial, inverse_filter
from openecon.engines.streaming_filter import PolynomialState


@pytest.mark.parametrize("block", [1, 3, 17, 701])
@pytest.mark.parametrize("unit", [1, 12])
@pytest.mark.parametrize("inverse", [False, True])
def test_state_matches_full_native_filter(block, unit, inverse):
    values = torch.randn((5, 997), generator=torch.Generator().manual_seed(193), dtype=torch.float64)
    coefficients = [.31, -.19, .07]
    state = PolynomialState(coefficients if inverse else [1., *[0. if i % unit else coefficients[i//unit-1] for i in range(1, len(coefficients)*unit+1)]], inverse=inverse, unit=unit if inverse else 1)
    actual = torch.cat([state(values[:, start:start+block]) for start in range(0, values.shape[1], block)], dim=1)
    expected = inverse_filter(values, coefficients, unit) if inverse else apply_polynomial(values, state.coefficients)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)
