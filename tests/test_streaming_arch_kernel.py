import math

import pytest
import torch

from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.arch.maximize import _variance_candidates, starting_values
from openecon.econometrics.streaming_arch_kernel import StreamingArchLikelihood


@pytest.mark.parametrize("block", [3, 71])
@pytest.mark.parametrize("kind,dist,in_mean", [
    ("garch", "normal", None), ("gjr", "normal", "variance"), ("egarch", "normal", None),
    ("parch", "t", "sd"), ("garch", "ged", "log"), ("igarch", "normal", None)])
def test_complete_arch_recursion_and_derivatives(kind, dist, in_mean, block):
    generator = torch.Generator().manual_seed(421)
    n = 251
    x = torch.cat((torch.ones((n, 1), dtype=torch.float64), torch.randn((n, 1), generator=generator, dtype=torch.float64)), 1)
    y, z = torch.randn(n, generator=generator, dtype=torch.float64), torch.randn((n, 1), generator=generator, dtype=torch.float64)
    layout = Layout(kind, dist, (1, 3), (1, 2), (1, 3), (2,), in_mean, 2, 1)
    ix = layout.index
    theta = torch.zeros(ix.k, dtype=torch.float64)
    theta[ix.x], theta[ix.z] = torch.tensor([.13, -.21], dtype=torch.float64), .12
    theta[ix.ar], theta[ix.ma] = torch.tensor([.15, -.07], dtype=torch.float64), .11
    seed = _variance_candidates(layout, 1.)[0]
    theta[ix.a], theta[ix.b] = torch.tensor(seed["a"], dtype=torch.float64), torch.tensor(seed["b"], dtype=torch.float64)
    theta[ix.c] = seed["omega"] if kind == "egarch" else math.log(seed["omega"])
    if ix.g:
        theta[ix.g] = torch.tensor(seed["g"], dtype=torch.float64)
    if ix.m:
        theta[ix.m] = .08
    if ix.p:
        theta[ix.p] = 1.7
    if ix.d:
        theta[ix.d] = math.log(5.) if dist == "t" else math.log(1.6)
    expected = ArchLikelihood(y, x, z, layout).evaluate(theta, scores=True)
    replay = StreamingArchLikelihood(lambda: ((x[j:j+block], y[j:j+block], z[j:j+block]) for j in range(0, n, block)), n, layout, budget_bytes=128*1024**2)
    actual = replay.evaluate(theta, scores=True)
    assert expected is not None and actual is not None
    assert actual.value == pytest.approx(expected.value, abs=1e-10)
    assert actual.presample_variance == pytest.approx(expected.presample_variance, rel=2e-13)
    torch.testing.assert_close(actual.gradient, expected.gradient, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(actual.scores.T@actual.scores, expected.scores.T@expected.scores, rtol=2e-11, atol=3e-11)
    torch.testing.assert_close(actual.residual, expected.residual, rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(actual.variance, expected.variance, rtol=3e-13, atol=3e-13)
    full_starts, full_scale = starting_values(ArchLikelihood(y, x, z, layout))
    replay_starts, replay_scale = replay.replay_starting_values()
    torch.testing.assert_close(replay_scale, full_scale, rtol=3e-12, atol=3e-12)
    for actual_start, expected_start in zip(replay_starts, full_starts, strict=True):
        torch.testing.assert_close(actual_start, expected_start, rtol=3e-12, atol=3e-12)
