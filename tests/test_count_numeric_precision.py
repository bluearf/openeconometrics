"""Shared count likelihood precision, independent mass and truncation oracles."""
import math

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammainc
import torch

import openecon as oe
from openecon.econometrics.count import common
from openecon.econometrics.count.kernels import (
    CountPieces, IndexObjective, NegBinDensity, PoissonDensity, TruncatedPieces,
)
from openecon.econometrics.glm.ppmlhdfe import _poisson_log_likelihood
from openecon.engines.contracts import KernelError
from openecon.engines.count_numeric import poisson_deviance, poisson_logmass


@pytest.mark.parametrize("count", [0., .3, 1., 14., 15., 15.5, 1e6, 1e9, 1e12, float(2**53)])
@pytest.mark.parametrize("shift", [-2, 0, 2])
def test_shared_poisson_mass_and_deviance_match_independent_high_precision(count, shift):
    mu = max(.01, count + shift * math.sqrt(max(1., count)))
    y, mean = torch.tensor([count], dtype=torch.float64), torch.tensor([mu], dtype=torch.float64)
    eta = torch.log(mean)
    with mp.workdps(85):
        c, m = mp.mpf(count), mp.mpf(mu)
        expected = c * mp.log(m) - m - mp.loggamma(c + 1)
        expected_deviance = 2 * (c * mp.log(c / m) - (c - m)) if count else 2 * m
    actual = poisson_logmass(y, mean)
    assert actual.item() == pytest.approx(float(expected), abs=5e-12)
    assert poisson_deviance(y, mean).item() == pytest.approx(float(expected_deviance), rel=2e-13, abs=1e-12)
    cached = torch.lgamma(y + 1)
    value, score, curvature = PoissonDensity().terms(y, cached, [eta])
    with mp.workdps(85):
        native_mu = mp.mpf(torch.exp(eta).item())
        density_expected = mp.mpf(count) * mp.log(native_mu) - native_mu - mp.loggamma(mp.mpf(count) + 1)
    assert value.item() == pytest.approx(float(density_expected), abs=5e-12)
    assert score[0].item() == count - torch.exp(eta).item()
    assert curvature[0].item() == -torch.exp(eta).item()
    assert _poisson_log_likelihood(y, mean, torch.ones_like(y)) == pytest.approx(float(expected), abs=5e-12)


@pytest.mark.parametrize("form", ["mean", "constant"])
@pytest.mark.parametrize("count", [1e9, 1e12, float(2**53)])
def test_shared_nb_densities_refuse_unverified_counts_instead_of_positive_log_probability(form, count):
    y = torch.tensor([count], dtype=torch.float64)
    with pytest.raises(KernelError) as error:
        NegBinDensity(form).terms(y, torch.lgamma(y + 1), [torch.log(y), torch.tensor(0., dtype=torch.float64)])
    assert error.value.code == "precision_unsupported"


def test_index_objective_precision_mode_tracks_rejected_trial_then_permits_valid_evaluation():
    y = torch.full((10,), 2., dtype=torch.float64)
    objective = IndexObjective(CountPieces(NegBinDensity(), y),
                               [torch.ones((10, 1), dtype=torch.float64), None], torch.ones_like(y))
    bad = torch.tensor([math.log(2.), -25.], dtype=torch.float64)
    with pytest.raises(KernelError) as error:
        objective(bad)
    assert error.value.code == "precision_unsupported"
    objective.reject_precision_trials = True
    assert objective(bad)[0].item() == -math.inf and objective.precision_rejected
    good = torch.tensor([math.log(2.), math.log(.8)], dtype=torch.float64)
    assert torch.isfinite(objective(good)[0])
    objective.reject_precision_trials = False
    objective.check_precision(good)
    with pytest.raises(KernelError) as error:
        objective.check_precision(bad)
    assert error.value.code == "precision_unsupported"


def test_count_optimizer_final_guard_and_trial_mode_restore(monkeypatch):
    from openecon.engines.optimize import OptimResult
    y = torch.full((10,), 2., dtype=torch.float64)
    model = common.Model(CountPieces(NegBinDensity(), y),
                         [common.constant_block(10, "y", None), common.scalar_block("/lnalpha")], torch.ones_like(y))
    start = torch.tensor([math.log(2.), math.log(.8)], dtype=torch.float64)
    def unverified_result(*args, **kwargs):
        return OptimResult(theta=torch.tensor([math.log(2.), -25.], dtype=torch.float64),
                           value=-10., gradient=torch.zeros(2, dtype=torch.float64),
                           hessian=-torch.eye(2, dtype=torch.float64), iterations=1,
                           converged=True, method="newton", diagnostics={})
    monkeypatch.setattr(common.optimize, "maximize_newton", unverified_result)
    with pytest.raises(oe.AnalysisError) as error:
        common.maximize(model.objective, start, what="count precision regression")
    assert error.value.code == "precision_unsupported"
    assert model.objective.reject_precision_trials is False


def test_large_poisson_truncation_hazard_uses_stable_boundary_mass():
    cutoff, mu = 10**9, 10**9 + 3.
    y = torch.tensor([cutoff + 5.], dtype=torch.float64)
    eta = torch.tensor([math.log(mu)], dtype=torch.float64)
    actual_mu = torch.exp(eta).item()
    value, gradient, curvature = TruncatedPieces(PoissonDensity(), y, cutoff)([eta])
    survival = gammainc(cutoff + 1, actual_mu)
    with mp.workdps(85):
        m = mp.mpf(actual_mu)
        boundary_mass = cutoff * mp.log(m) - m - mp.loggamma(cutoff + 1)
        observed_mass = (cutoff + 5) * mp.log(m) - m - mp.loggamma(cutoff + 6)
    hazard = actual_mu * math.exp(float(boundary_mass)) / survival
    expected_hazard_hessian = hazard * ((cutoff - actual_mu) + 1) - hazard**2
    assert value.item() == pytest.approx(float(observed_mass) - math.log(survival), abs=1e-10)
    assert gradient[0].item() == pytest.approx(cutoff + 5 - actual_mu - hazard, rel=1e-10)
    assert curvature[0].item() == pytest.approx(-actual_mu - expected_hazard_hessian, rel=1e-9)


@pytest.mark.parametrize("name", ["nbreg", "tnbreg", "gnbreg", "zinb"])
def test_public_nb_models_have_specific_static_precision_errors(name):
    data = pd.DataFrame({"y": np.tile([10**9 - 1, 10**9], 30), "x": np.tile([-1., 1.], 30)})
    if name == "zinb":
        data.loc[::5, "y"] = 0
    options = {"lnalpha": ["x"]} if name == "gnbreg" else {"inflate": ["x"]} if name == "zinb" else {}
    with pytest.raises(oe.AnalysisError) as error:
        getattr(oe, name)(data=data, y="y", x=["x"], **options)
    assert error.value.code == "precision_unsupported"


def test_zero_weight_oversized_nb_row_is_screened_before_precision_validation():
    rng = np.random.default_rng(784)
    data = pd.DataFrame({"y": rng.negative_binomial(2., .4, 250), "x": rng.normal(size=250), "w": 1.})
    expected = oe.nbreg(data=data, y="y", x=["x"])
    ignored = pd.DataFrame({"y": [10**12], "x": [3.], "w": [0.]})
    actual = oe.nbreg(data=pd.concat([data, ignored], ignore_index=True), y="y", x=["x"], weights="w", weight_type="iweight")
    assert actual.nobs == expected.nobs
    assert [c.estimate for c in actual.coefficients] == pytest.approx([c.estimate for c in expected.coefficients], rel=1e-12)


@pytest.mark.parametrize("outcomes, curvature, expected", [
    ([0., 2., 2., 2.], -1e-8, True),
    ([0., 20., 0., 20.], -1e-8, False),
    ([0., 2., 2., 2.], .1, False),
])
def test_zinb_precision_rejection_requires_independent_limit_score_and_small_absolute_curvature(outcomes, curvature, expected):
    from types import SimpleNamespace
    from openecon.econometrics.count import zeroinflated_fit as zi
    from openecon.econometrics.glm.common import Weights
    y = torch.tensor(outcomes, dtype=torch.float64)
    weights = Weights(torch.ones_like(y), None, len(y), False)
    indices = [torch.full_like(y, math.log(2.)), torch.full_like(y, -1.), torch.tensor(-16., dtype=torch.float64)]
    limit = zi.ZeroInflatedPieces(PoissonDensity(), y, "logit")(indices[:2], False)[0].sum()
    class Objective:
        def indices(self, theta):
            return indices
        def __call__(self, theta):
            hessian = -torch.eye(3, dtype=torch.float64)
            hessian[-1, -1] = curvature
            return limit, torch.zeros(3, dtype=torch.float64), hessian
    sample = SimpleNamespace(y=y, zero=y == 0, link="logit", weights=weights)
    model = SimpleNamespace(objective=Objective())
    theta = torch.tensor([math.log(2.), -1., -16.], dtype=torch.float64)
    assert zi._poisson_dispersion_boundary(sample, model, theta, 2) is expected
