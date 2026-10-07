"""High-precision rare-tail/derivative and bounded-work adapter certificates."""

import importlib
import math

import mpmath as mp
import numpy as np
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.mixture_prediction import CountMixture
from openecon.econometrics.postest.prediction import _Design

from test_econ_saved_prediction_mixtures import fitted as fitted


def adapter(estimator, form=None, link=None):
    auxiliary = (2, 3) if estimator in {"zip", "zinb", "hurdle"} else ()
    return CountMixture(estimator, (0, 1), auxiliary, {"x": 1}, {"x": 3} if auxiliary else {},
                        link, form, 4 if form is not None else None, None, 0)


def design(limit):
    return _Design(torch.tensor([[1., .2, 1., .2, 0.]], dtype=torch.float64),
                   torch.zeros(1, dtype=torch.float64),
                   torch.tensor([float(limit)], dtype=torch.float64))


def high_precision(beta, x, ll, form):
    eta = beta[0] + x * beta[1]
    mu = mp.exp(eta)
    if form is None:
        survival = -mp.expm1(-mu) if ll == 0 else mp.gammainc(ll + 1, 0, mu) / mp.factorial(ll)
        first_tail = 1 if ll == 0 else mp.gammainc(ll, 0, mu) / mp.factorial(ll - 1)
    else:
        r = mp.exp(-beta[4]) if form == "mean" else mp.exp(eta - beta[4])
        q = mu / (mu + r)
        survival = mp.betainc(ll + 1, r, 0, q, regularized=True)
        first_tail = 1 if ll == 0 else mp.betainc(ll, r + 1, 0, q, regularized=True)
    return mu * first_tail / survival


@pytest.mark.parametrize("eta,ll,form", [
    (-100., 0, None), (-6.9, 200, None),
    (-100., 0, "mean"), (-6.9, 120, "mean"),
    (-24., 20, "constant"), (3., 0, "mean"),
])
def test_rare_count_tails_values_effects_and_parameter_jacobians(eta, ll, form):
    est = "tpoisson" if form is None else "tnbreg"
    module = adapter(est, form)
    beta = torch.tensor([eta - .08, .4, -.2, .1, math.log(.6)], dtype=torch.float64)
    encoded = design(ll)
    with mp.workdps(100):
        b = [mp.mpf(float(v)) for v in beta]
        x = mp.mpf(float(encoded.x[0, 1]))
        expected = high_precision(b, x, ll, form)
        expected_effect = mp.diff(lambda v: high_precision(b, v, ll, form), x)
        gradients, effect_gradients = [], []
        for index in range(len(b)):
            def point(value):
                changed = list(b)
                changed[index] = value
                return high_precision(changed, x, ll, form)

            def slope(value):
                changed = list(b)
                changed[index] = value
                return mp.diff(lambda v: high_precision(changed, v, ll, form), x)

            gradients.append(float(mp.diff(point, b[index])))
            effect_gradients.append(float(mp.diff(slope, b[index])))
    value = module.values(encoded, beta, "response")
    effect = module.effects(encoded, beta, "x", "response")
    jac = module.jacobian(encoded, beta, "response")[0]
    effect_jac = module.jacobian(encoded, beta, "derivative", "x")[0]
    np.testing.assert_allclose(value, float(expected), rtol=2e-14, atol=0)
    np.testing.assert_allclose(effect.detach(), float(expected_effect), rtol=3e-13, atol=0)
    np.testing.assert_allclose(jac, gradients, rtol=3e-13, atol=1e-100)
    np.testing.assert_allclose(effect_jac, effect_gradients, rtol=5e-12, atol=1e-100)
    if eta == -100:
        assert float(effect.detach()[0]) > 1e-45  # tiny but mathematically nonzero


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_zero_inflated_mean_avoids_unnecessary_overflow(link):
    module = adapter("zip", link=link)
    beta = torch.tensor([800., 0., 800. if link == "logit" else 40., 0., 0.], dtype=torch.float64)
    encoded = design(0)
    with mp.workdps(100):
        z = mp.mpf(float(beta[2]))
        log_survival = -mp.log(1 + mp.exp(z)) if link == "logit" else mp.log(mp.erfc(z / mp.sqrt(2)) / 2)
        expected = mp.exp(800 + log_survival)
        hazard = 1 / (1 + mp.exp(-z)) if link == "logit" else mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi) / (mp.erfc(z / mp.sqrt(2)) / 2)
    value = module.values(encoded, beta, "response")
    jac = module.jacobian(encoded, beta, "response")[0]
    np.testing.assert_allclose(value, float(expected), rtol=3e-13)
    np.testing.assert_allclose(jac, [float(expected), .2 * float(expected), -float(expected * hazard), -.2 * float(expected * hazard), 0.], rtol=4e-13)


def test_shared_predictor_exact_response_effect_cancellation():
    module = adapter("zip", link="logit")
    # At a saturated inflation logit log(mean)=eta-z, and equal slopes cancel
    # exactly. Its parameter gradient still includes the direct slope contrast.
    beta = torch.tensor([800., 1., 800., 1., 0.], dtype=torch.float64)
    encoded = design(0)
    assert float(module.effects(encoded, beta, "x", "response")[0].detach()) == 0
    np.testing.assert_array_equal(module.jacobian(encoded, beta, "derivative", "x"), [[0., 1., 0., -1., 0.]])


@pytest.mark.parametrize("form", [None, "mean"])
@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_hurdle_compensated_tail_keeps_representable_derivatives(form, link):
    module = adapter("hurdle", form, link)
    beta = torch.tensor([600., .4, -800. if link != "probit" else -40., .2, math.log(.6)], dtype=torch.float64)
    encoded = design(0)
    with mp.workdps(100):
        b = [mp.mpf(float(v)) for v in beta]
        x = mp.mpf(.2)

        def mean(parameters, point):
            z = parameters[2] + parameters[3] * point
            probability = (1 / (1 + mp.exp(-z)) if link == "logit" else
                           mp.erfc(-z / mp.sqrt(2)) / 2 if link == "probit" else
                           -mp.expm1(-mp.exp(z)))
            return high_precision(parameters, point, 0, form) * probability

        expected = mean(b, x)
        slope = mp.diff(lambda point: mean(b, point), x)
        jac, slope_jac = [], []
        for index in range(len(b)):
            def value(changed):
                parameters = list(b)
                parameters[index] = changed
                return mean(parameters, x)

            def effect(changed):
                parameters = list(b)
                parameters[index] = changed
                return mp.diff(lambda point: mean(parameters, point), x)

            jac.append(float(mp.diff(value, b[index])))
            slope_jac.append(float(mp.diff(effect, b[index])))
    np.testing.assert_allclose(module.values(encoded, beta, "response"), float(expected), rtol=6e-13)
    np.testing.assert_allclose(module.effects(encoded, beta, "x", "response").detach(), float(slope), rtol=6e-13)
    np.testing.assert_allclose(module.jacobian(encoded, beta, "response"), [jac], rtol=1e-12, atol=1e-180)
    np.testing.assert_allclose(module.jacobian(encoded, beta, "derivative", "x"), [slope_jac], rtol=1e-12, atol=1e-180)


@pytest.mark.parametrize("failure", ["subnormal", "negative_cutoff", "fractional_cutoff", "nb_shape", "nb_work", "series_nonconvergence", "budget"])
def test_precision_and_resource_refusals_are_explicit(failure, monkeypatch):
    module = adapter("tpoisson")
    encoded, beta = design(0), torch.tensor([0., .4, 0., 0., 0.], dtype=torch.float64)
    code = "prediction_precision"
    if failure == "subnormal":
        beta[0] = -740
    elif failure in {"negative_cutoff", "fractional_cutoff"}:
        encoded.scale[0] = -1 if failure == "negative_cutoff" else 1.5
        code = "unsupported_margins_transform"
    elif failure == "nb_shape":
        module = adapter("tnbreg", "mean")
        beta[4] = -math.log(1e8)
        code = "precision_unsupported"
    elif failure == "nb_work":
        module = adapter("tnbreg", "mean")
        encoded.scale[0] = 10001
        code = "prediction_work_limit"
    elif failure == "series_nonconvergence":
        from openecon.econometrics.postest.mixture_prediction import _tail_series
        with pytest.raises(AnalysisError) as exc:
            _tail_series(torch.tensor([1e6]), None, torch.tensor([1e6]), parameters=5)
        assert exc.value.code == code
        return
    else:
        name = importlib.import_module("openecon.econometrics.postest.mixture_prediction")
        monkeypatch.setattr(name, "_MAX_DESIGN_BYTES", 100)
        code = "prediction_memory_limit"
    with pytest.raises(AnalysisError) as exc:
        module.values(encoded, beta, "response")
    assert exc.value.code == code


def test_saved_ancillary_metadata_and_outcome_selection_guard(fitted):
    model, frame, _ = fitted
    if model.spec.estimator in {"zinb", "tnbreg"} or model.spec.estimator == "hurdle" and model.extra["dist"] == "nbinomial":
        bad = model.model_copy(deep=True)
        bad.extra["alpha"]["estimate"] *= 2
        from openecon.econometrics.postest.inference import _parameters
        from openecon.econometrics.postest.mixture_prediction import configure
        with pytest.raises(AnalysisError) as exc:
            configure(bad, _parameters(bad))
        assert exc.value.code == "invalid_result"
    import openecon as oe
    with pytest.raises(AnalysisError) as exc:
        oe.predict(model, frame.iloc[:3], outcome=1)
    assert exc.value.code == "unsupported_prediction_outcome"


def test_saved_cpu_scope_restores_global_meta_device(fitted):
    import openecon as oe
    model, frame, _ = fitted
    with torch.device("meta"):
        result = oe.predict(model, frame.iloc[:3].drop(columns="y"), interval="mean")
        assert torch.ones(1).device.type == "meta"
    assert np.isfinite(result.response).all()


def test_saved_derivative_inference_mode_is_local_and_restored(fitted):
    import openecon as oe
    model, frame, _ = fitted
    new = frame.iloc[:7].drop(columns="y")
    expected = oe.predict(model, new, kind="derivative", term="x", interval="mean")
    with torch.inference_mode():
        actual = oe.predict(model, new, kind="derivative", term="x", interval="mean")
        margins = oe.margins(model, ["x", "g"], data=new)
        assert torch.is_inference_mode_enabled()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    assert np.isfinite(margins.std_error).all()
