"""Large count precision, independent Decimal oracles and optimizer failure contracts."""

import importlib
import math
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.glm.common import maximize
from openecon.econometrics.glm.families import Log, NegativeBinomial, Poisson
from openecon.econometrics.glm.kernels import (
    GlmObjective, NegativeBinomialObjective, _poisson_deviance,
)
from openecon.engines.contracts import KernelError
from openecon.engines.optimize import OptimResult

nb_module = importlib.import_module("openecon.econometrics.glm.nbreg")
optimizer_module = importlib.import_module("openecon.engines.optimize")


def decimal_poisson(y, mu):
    """Independent 70-digit log likelihood/deviance, not float64 log subtraction.

    The Stirling error after the fifth term is negligible for these y >= 1e6.
    ln(2*pi) is stored at substantially greater precision than float64.
    """
    with localcontext() as context:
        context.prec = 70
        count, mean = Decimal.from_float(float(y)), Decimal.from_float(float(mu))
        log_2pi = Decimal(
            "1.837877066409345483560659472811235279722794947275566825634303081"
        )
        correction = (1 / (12 * count) - 1 / (360 * count**3)
                      + 1 / (1260 * count**5) - 1 / (1680 * count**7)
                      + 1 / (1188 * count**9))
        log_factorial = (count + Decimal("0.5")) * count.ln() - count
        log_factorial += log_2pi / 2 + correction
        mass = count * mean.ln() - mean - log_factorial
        deviance = 2 * (count * (count / mean).ln() - (count - mean))
        return float(mass), float(deviance)


@pytest.mark.parametrize("level", [1e6, 1e9, 1e12, 1e15])
def test_poisson_family_large_near_mean_against_decimal(level):
    y = torch.tensor([level - 2, level, level + 2], dtype=torch.float64)
    mu = torch.full_like(y, math.exp(math.log(level)))
    expected = np.array([decimal_poisson(count, mean)
                         for count, mean in zip(y.tolist(), mu.tolist(), strict=True)])
    assert_allclose(Poisson().log_likelihood(y, mu, 1), expected[:, 0], atol=1e-13, rtol=0)
    assert_allclose(Poisson().unit_deviance(y, mu), expected[:, 1], rtol=2e-15, atol=1e-28)


@pytest.mark.parametrize("level", [1e9, 1e12, 1e15])
@pytest.mark.parametrize("optimizer", ["ml", "irls"])
def test_public_poisson_fit_large_counts_matches_independent_intercept_mle(level, optimizer):
    frame = pd.DataFrame({"y": [level - 2, level - 1, level, level + 1, level + 2]})
    result = oe.glm(data=frame, y="y", x=[], family="poisson", optimizer=optimizer)
    beta = result.coefficients[0]
    mean = result.predictions[0]["fitted"]
    log_likelihood = sum(decimal_poisson(count, mean)[0] for count in frame.y)
    deviance = sum(decimal_poisson(count, mean)[1] for count in frame.y)
    assert beta.estimate == pytest.approx(math.log(level), abs=1e-13)
    assert beta.std_error == pytest.approx(1 / math.sqrt(len(frame) * mean), rel=1e-13)
    assert result.metrics["log_likelihood"] == pytest.approx(log_likelihood, abs=1e-12)
    assert result.metrics["deviance"] == pytest.approx(deviance, rel=2e-13, abs=1e-27)
    assert result.metrics["aic"] == pytest.approx(-2 * log_likelihood + 2, abs=2e-12)


@pytest.mark.parametrize("level", [1e9, 1e12, 1e15])
def test_ppml_weighted_iteration_deviance_matches_decimal_near_large_mean(level):
    y = torch.tensor([level - 2, level, level + 2], dtype=torch.float64)
    mu = torch.full_like(y, math.exp(math.log(level)))
    weights = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64)
    expected = sum(weight * decimal_poisson(count, mean)[1]
                   for count, mean, weight in zip(y.tolist(), mu.tolist(), weights.tolist(),
                                                  strict=True))
    assert _poisson_deviance(y, mu, weights) == pytest.approx(expected, rel=2e-15, abs=1e-28)


@pytest.mark.parametrize("level", [1e9, 1e12])
def test_public_large_ppml_matches_independent_balanced_group_mle(level):
    groups, x = np.repeat([0, 1], 10), np.tile([-1.0, 1.0], 10)
    mean = level * (1 + groups) * np.exp(0.001 * x)
    y = np.round(mean + np.tile([-2, -1, 0, 1, 2], 4) * np.sqrt(mean))
    frame = pd.DataFrame({"y": y, "group": groups, "x": x})
    result = oe.ppmlhdfe(data=frame, y="y", x=["x"], absorb=["group"],
                        covariance="nonrobust")
    # Balanced +/-1 cells and group intercepts imply a closed-form slope MLE.
    expected_beta = 0.5 * math.log(y[x == 1].sum() / y[x == -1].sum())
    assert result.coefficients[0].estimate == pytest.approx(expected_beta, abs=2e-13)
    fitted = np.empty(len(frame))
    for group in (0, 1):
        subset = groups == group
        baseline = y[subset].mean() / math.cosh(expected_beta)
        fitted[subset] = baseline * np.exp(expected_beta * x[subset])
    actual_mean = np.array([row["fitted"] for row in result.predictions])
    assert_allclose(actual_mean, fitted, rtol=5e-14, atol=0)
    deviance = sum(decimal_poisson(count, fitted_mean)[1]
                   for count, fitted_mean in zip(y, actual_mean, strict=True))
    likelihood = sum(decimal_poisson(count, fitted_mean)[0]
                     for count, fitted_mean in zip(y, actual_mean, strict=True))
    assert result.metrics["deviance"] == pytest.approx(deviance, rel=2e-13)
    assert result.metrics["log_likelihood"] == pytest.approx(likelihood, abs=1e-10)


@pytest.mark.parametrize("dispersion,y", [(1.0, 1e7), (1e-8, 3.0), (1e-6, 9e6)])
def test_fixed_nb_guard_precedes_any_optimizer_iteration(monkeypatch, dispersion, y):
    def forbidden(*args, **kwargs):
        pytest.fail("An unverified fixed-NB likelihood must be rejected before optimization")

    monkeypatch.setattr(optimizer_module, "maximize_newton", forbidden)
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=pd.DataFrame({"y": [y] * 8}), y="y", x=[],
               family="nbinomial", dispersion=dispersion)
    assert caught.value.code == "precision_unsupported"


def test_fixed_nb_raw_objective_rejects_the_same_unsafe_shape():
    y = torch.ones(8, dtype=torch.float64)
    with pytest.raises(KernelError) as caught:
        GlmObjective(y[:, None], y, y, None, NegativeBinomial(1e-8), Log())
    assert caught.value.code == "precision_unsupported"


@pytest.mark.parametrize("form", ["mean", "constant"])
def test_nb_precision_start_and_trial_contract(form):
    y = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
    objective = NegativeBinomialObjective(torch.ones((4, 1), dtype=torch.float64),
                                         y, torch.ones_like(y), None, form)
    unsafe = torch.tensor([math.log(10), -math.log(1e8)], dtype=torch.float64)
    with pytest.raises(KernelError) as caught:
        objective.value(unsafe)
    assert caught.value.code == "precision_unsupported"
    with pytest.raises(AnalysisError) as caught:
        maximize(objective, unsafe, what="negative binomial")
    assert caught.value.code == "precision_unsupported"
    assert not objective.reject_precision_trials
    objective.reject_precision_trials = True
    assert math.isinf(float(objective.value(unsafe)))
    assert float(objective.value(unsafe)) < 0
    assert objective.precision_rejected


def test_optimizer_final_candidate_is_checked_and_trial_mode_is_restored(monkeypatch):
    y = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
    objective = NegativeBinomialObjective(torch.ones((4, 1), dtype=torch.float64),
                                         y, torch.ones_like(y), None, "mean")
    start = torch.tensor([math.log(4), math.log(0.5)], dtype=torch.float64)

    def unsafe_result(*args, **kwargs):
        unsafe = start.clone()
        unsafe[-1] = -math.log(1e8)
        return OptimResult(unsafe, 0.0, torch.zeros(2), -torch.eye(2), 1, True, "fixture")

    monkeypatch.setattr(optimizer_module, "maximize_newton", unsafe_result)
    with pytest.raises(AnalysisError) as caught:
        maximize(objective, start, what="negative binomial")
    assert caught.value.code == "precision_unsupported"
    assert not objective.reject_precision_trials


@pytest.mark.parametrize("form", ["mean", "constant"])
def test_nb_static_large_count_is_typed_before_poisson_start(monkeypatch, form):
    def forbidden(*args, **kwargs):
        pytest.fail("The impossible NB count must be checked before even the Poisson start")

    monkeypatch.setattr(nb_module, "estimate", forbidden)
    with pytest.raises(AnalysisError) as caught:
        oe.nbreg(data=pd.DataFrame({"y": [1e7] * 8}), y="y", x=[], dispersion=form)
    assert caught.value.code == "precision_unsupported"


@pytest.fixture
def overdispersed():
    rng = np.random.default_rng(7821)
    x = rng.normal(size=400)
    mu = np.exp(0.8 + 0.25 * x)
    return pd.DataFrame({"x": x, "y": rng.negative_binomial(2, 2 / (2 + mu))})


def test_valid_nb_survives_rejected_precision_trial_and_optional_null(overdispersed, monkeypatch):
    reference = oe.nbreg(data=overdispersed, y="y", x=["x"])
    real_newton = optimizer_module.maximize_newton

    def inject_trial(objective, start, **kwargs):
        if isinstance(objective, NegativeBinomialObjective):
            unsafe = start.clone()
            unsafe[-1] = -math.log(1e8)
            assert float(objective.value(unsafe)) == -math.inf
        return real_newton(objective, start, **kwargs)

    def unsupported_null(*args, **kwargs):
        raise AnalysisError("precision_unsupported", "Unverified optional comparison shape")

    monkeypatch.setattr(optimizer_module, "maximize_newton", inject_trial)
    monkeypatch.setattr(nb_module, "_null", unsupported_null)
    fitted = oe.nbreg(data=overdispersed, y="y", x=["x"])
    assert_allclose([c.estimate for c in fitted.coefficients],
                    [c.estimate for c in reference.coefficients], rtol=0, atol=0)
    assert_allclose(fitted.covariance_matrix, reference.covariance_matrix, rtol=0, atol=0)
    assert fitted.metrics["log_likelihood"] == reference.metrics["log_likelihood"]
    assert fitted.metrics["pseudo_r_squared"] is None
    assert fitted.extra["null_log_likelihood"] is None
    assert fitted.extra["null_model"] is None
    assert "Wald" in fitted.tests["model"]["label"]
    assert any("precision domain" in warning for warning in fitted.warnings)


def test_nonprecision_optional_null_error_is_not_hidden(overdispersed, monkeypatch):
    def failed_null(*args, **kwargs):
        raise AnalysisError("nonconvergence", "Unrelated optional comparison failure")

    monkeypatch.setattr(nb_module, "_null", failed_null)
    with pytest.raises(AnalysisError) as caught:
        oe.nbreg(data=overdispersed, y="y", x=["x"])
    assert caught.value.code == "nonconvergence"


@pytest.mark.parametrize("precision_rejected,code", [(True, "precision_unsupported"),
                                                     (False, "nonconvergence")])
@pytest.mark.parametrize("tiny_dispersion", [False, True])
def test_failed_nb_iteration_is_not_mislabelled_poisson_boundary(overdispersed, monkeypatch,
                                                               precision_rejected, code,
                                                               tiny_dispersion):
    def failed_nb(objective, start, **kwargs):
        if not isinstance(objective, NegativeBinomialObjective):
            pytest.fail("Only NB optimization should be replaced by this fixture")
        value, gradient, hessian = objective(start)
        # The initial NB likelihood can be below the Poisson likelihood even
        # for truly overdispersed data. That alone is no boundary certificate.
        if tiny_dispersion:
            value = (objective.prior * Poisson().log_likelihood(
                objective.y, objective.mean(start), 1)).sum()
            start = start.clone()
            start[-1] = math.log(1e-7)
            hessian = hessian.clone()
            hessian[-1, -1] = 0
            # Even a tiny dispersion, weak ancillary curvature and likelihood
            # near Poisson cannot certify a boundary when its one-sided
            # dispersion score points into the NB parameter space.
        return OptimResult(start, float(value), gradient, hessian, 0, False, "fixture",
                           {"precision_rejected": precision_rejected, "message": "stalled"})

    monkeypatch.setattr(nb_module, "maximize", failed_nb)
    with pytest.raises(AnalysisError) as caught:
        oe.nbreg(data=overdispersed, y="y", x=["x"])
    assert caught.value.code == code


@pytest.mark.parametrize("form", ["mean", "constant"])
def test_genuine_underdispersed_poisson_boundary_is_preserved(form):
    frame = pd.DataFrame({"y": np.tile([1.0, 2.0, 3.0], 30)})
    with pytest.raises(AnalysisError) as caught:
        oe.nbreg(data=frame, y="y", x=[], dispersion=form)
    assert caught.value.code == "boundary_solution"
    assert "oe.poisson" in str(caught.value)
