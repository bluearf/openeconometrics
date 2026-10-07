import pytest
import torch

from openecon.econometrics.arima.kernels import ArmaLikelihood, starting_values
from openecon.econometrics.streaming_arma_kernel import StreamingArmaLikelihood


def test_presample_and_new_head_share_the_existing_workspace_budget():
    """Independent plans must not each consume the same remaining allowance."""
    from openecon.analysis_contracts import AnalysisError
    from openecon.resources import plan_workspace

    plans = []
    budget = 2 * 1024**2
    touched = False

    def reserve(operation, buffers):
        plan = plan_workspace(operation, {"live_source_and_design": 1024**2, **buffers}, budget_bytes=budget)
        plans.append(plan)
        return plan

    def source():
        nonlocal touched
        touched = True
        raise AssertionError("The combined resource refusal must precede reading model rows")
        yield

    replay = StreamingArmaLikelihood(source, 401, 2, q=1, exact=True,
                                     budget_bytes=budget, workspace_reserver=reserve)
    replay._impulse([.2], [])
    budget = plans[-1].estimated_bytes + 1
    with pytest.raises(AnalysisError) as error:
        replay.evaluate(torch.tensor([.1, .2, .2], dtype=torch.float64))
    assert error.value.code == "workspace_limit"
    record = error.value.resource_plan
    assert record["buffers"]["live_source_and_design"] == 1024**2
    assert record["buffers"]["finite_head_and_derivatives"] > 0
    assert record["buffers"]["presample_head"] > 0
    assert not touched


@pytest.mark.parametrize("block", [3, 73])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("orders,arma", [({}, []), ({"p": 2}, [.31, -.09]),
    ({"q": 2}, [.29, -.07]), ({"p": 1, "q": 1}, [.31, .24]),
    ({"p": 1, "q": 1, "seasonal_p": 1, "seasonal_q": 1, "period": 12}, [.21, .13, .17, .11])])
def test_entire_likelihood_derivatives_and_real_scores(block, exact, orders, arma):
    generator = torch.Generator().manual_seed(121)
    x = torch.cat((torch.ones((431, 1), dtype=torch.float64), torch.randn((431, 2), generator=generator, dtype=torch.float64)), dim=1)
    y = torch.randn(431, generator=generator, dtype=torch.float64)
    dense = ArmaLikelihood(y, x, exact=exact, **orders)
    replay = StreamingArmaLikelihood(lambda: ((x[start:start+block], y[start:start+block]) for start in range(0, len(y), block)), len(y), x.shape[1], budget_bytes=128*1024**2, exact=exact, **orders)
    psi = torch.tensor([.12, .29, -.11, *arma], dtype=torch.float64)
    expected, actual = dense.evaluate(psi), replay.evaluate(psi)
    assert actual.ss == pytest.approx(expected.ss, rel=2e-13)
    assert actual.logdet == pytest.approx(expected.logdet, abs=2e-13)
    torch.testing.assert_close(actual.d_ss, expected.d_ss, rtol=5e-13, atol=5e-13)
    torch.testing.assert_close(actual.d_logdet, expected.d_logdet, rtol=5e-13, atol=5e-13)
    torch.testing.assert_close(replay.gauss_newton(psi), dense.gauss_newton(psi), rtol=5e-13, atol=5e-13)
    theta = torch.cat((psi, torch.tensor([1.3], dtype=torch.float64)))
    torch.testing.assert_close(replay.replay_starting_values(), starting_values(y, x, replay.sizes, replay.period), rtol=5e-12, atol=5e-12)
    full = dense.decompose(theta)
    blocks = list(replay.decomposition_blocks(theta))
    torch.testing.assert_close(torch.cat([row[0] for row in blocks]), full.innovations, rtol=5e-13, atol=5e-13)
    torch.testing.assert_close(torch.cat([row[1] for row in blocks]), full.variance_ratio, rtol=5e-13, atol=5e-13)
    torch.testing.assert_close(torch.cat([row[2] for row in blocks]), full.scores, rtol=5e-13, atol=5e-13)
    torch.testing.assert_close(replay.end_disturbances, full.disturbances, rtol=5e-13, atol=5e-13)
    torch.testing.assert_close(replay.end_disturbance_covariance, full.disturbance_covariance, rtol=5e-13, atol=5e-13)
