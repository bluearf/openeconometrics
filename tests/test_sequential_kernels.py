"""Independent adaptive-integral oracles for canonical first crossings."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats import sequential_kernels as kernels
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("sides", [1, 2])
@pytest.mark.parametrize("drift", [-8.0, -1.3, 0.0, 2.1, 8.0])
def test_two_look_first_crossings_adaptive_brownian_oracle(sides, drift):
    # Independent integration in Brownian score B(t)+drift*t, not Z grids,
    # kernel transition densities or production quadrature constructors.
    times, bounds = [0.4, 1.0], [2.7, 1.9]
    first_sd = math.sqrt(times[0])
    first_mean = drift * times[0]
    first_bound = bounds[0] * first_sd
    lower = -first_bound if sides == 2 else -np.inf
    upper0 = norm.sf(first_bound, loc=first_mean, scale=first_sd)
    lower0 = norm.cdf(-first_bound, loc=first_mean, scale=first_sd) if sides == 2 else 0.0
    delta = times[1] - times[0]
    next_sd = math.sqrt(delta)

    def integral(kind):
        def density(score):
            mean = score + drift * delta
            if kind == "upper":
                probability = norm.sf(bounds[1], loc=mean, scale=next_sd)
            elif kind == "lower":
                probability = norm.cdf(-bounds[1], loc=mean, scale=next_sd)
            else:
                probability = (norm.cdf(bounds[1], loc=mean, scale=next_sd)
                               - (norm.cdf(-bounds[1], loc=mean, scale=next_sd) if sides == 2 else 0.0))
            return norm.pdf(score, loc=first_mean, scale=first_sd) * probability
        return quad(density, lower, first_bound, epsabs=1e-12, epsrel=1e-12, limit=150)[0]

    result = kernels.probabilities(times, bounds, drift, sides, 256)
    assert result["upper"] == pytest.approx([upper0, integral("upper")], abs=2e-12)
    assert result["lower"] == pytest.approx([lower0, integral("lower") if sides == 2 else 0.0], abs=2e-12)
    assert result["continuation"][-1] == pytest.approx(integral("continue"), abs=2e-12)
    assert result["reach"] == pytest.approx([1.0, 1.0-upper0-lower0], abs=2e-12)
    assert result["expectedfraction"] == pytest.approx(times[0]+delta*(1-upper0-lower0), abs=2e-12)


@pytest.mark.parametrize("law,param", [("ldof", None), ("ldpocock", None),
                                      ("hsd", -4.0), ("hsd", 0.0),
                                      ("hsd", 8.0), ("power", 0.25), ("power", 4.0)])
@pytest.mark.parametrize("sides", [1, 2])
def test_spending_primary_formula_and_total_alpha_tail_convention(law, param, sides):
    t = np.array([0.2, 0.45, 0.8, 1.0])
    alpha = 0.01
    if law == "ldof":
        expected = sides * 2 * norm.sf(norm.isf(alpha/(2*sides)) / np.sqrt(t))
    elif law == "ldpocock":
        expected = alpha * np.log1p(np.expm1(1.0) * t)
    elif law == "hsd":
        expected = alpha * t if param == 0 else alpha * np.expm1(-param*t) / np.expm1(-param)
    else:
        expected = alpha * t**param
    assert kernels.spend(t.tolist(), alpha, law, param, sides).tolist() == pytest.approx(expected, abs=2e-16)


def test_six_look_directional_symmetry_monotonicity_and_conservation():
    t = [.2, .4, .6, .8, .95, 1.]
    for sides in (1, 2):
        fitted = kernels.calibrate(t, .01, "ldof", sides=sides)
        assert fitted["probabilities"]["power"] == pytest.approx(.01, abs=2e-12)
        assert np.cumsum(np.array(fitted["probabilities"]["upper"])+np.array(fitted["probabilities"]["lower"])) == pytest.approx(fitted["spending"], abs=2e-12)
        a = kernels.probabilities(t, fitted["bounds"], -3.0, sides, 256)
        b = kernels.probabilities(t, fitted["bounds"], 3.0, sides, 256)
        assert a["power"] + a["continuation"][-1] == pytest.approx(1.0, abs=2e-12)
        if sides == 2:
            assert a["upper"] == pytest.approx(b["lower"], abs=2e-12)
            assert a["lower"] == pytest.approx(b["upper"], abs=2e-12)
            assert a["power"] == pytest.approx(b["power"], abs=2e-12)
        else:
            assert a["lower"] == [0.0]*6
            assert a["power"] < .01 < b["power"]


def test_resource_refusal_precedes_quadrature_allocation(monkeypatch):
    def forbidden(*args):
        raise AssertionError("No quadrature allocation is allowed after failed admission.")
    monkeypatch.setattr(kernels, "_nodes", forbidden)
    with pytest.raises(AnalysisError, match="work bound") as error:
        kernels.calibrate([.3, .7, 1.], .05, "power", max_work=1)
    assert error.value.code == "resource_limit"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        kernels.probabilities([.3, .7, 1.], [3., 2.4, 2.], order=256)
    assert error.value.code == "workspace_limit"


def test_saved_calibration_plan_rederived_without_boundary_search(monkeypatch):
    fractions = [.2, .5, .8, 1.]
    fit = kernels.calibrate(fractions, .05, "ldpocock", sides=2)

    def forbidden(*args):
        raise AssertionError("Reconstructing a numerical plan must not allocate quadrature or calibrate.")
    monkeypatch.setattr(kernels, "_nodes", forbidden)
    monkeypatch.setattr(kernels, "_calibrate_at_order", forbidden)
    assert kernels.calibration_work_bound(fractions, sides=2) == fit["numerical_receipt"]


@pytest.mark.parametrize("fractions", [[.1, 1.], [.2, .96, 1.], [.2, .5, .9],
                                        [.2, True, 1.], [.2, .5, float("nan"), 1.]])
def test_unsupported_calendar_is_an_explicit_admission_boundary(fractions):
    with pytest.raises(AnalysisError):
        kernels.calibrate(fractions, .05, "hsd")


def test_explicit_cpu_float64_under_changed_torch_defaults():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        spending = kernels.spend([.3, .7, 1.], .05, "power")
        assert spending.dtype == torch.float64 and spending.device.type == "cpu"
        assert kernels.probabilities([.3, .7, 1.], [3., 2.4, 2.])["power"] > 0
    finally:
        torch.set_default_dtype(previous)
