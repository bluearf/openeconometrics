"""Saved limited-response predictions checked against independent quadrature."""

import copy
from decimal import Decimal, localcontext
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy import stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.limited_prediction import LimitedNormal
from openecon.models import ResultBundle


def conditional_oracle(mu, sigma, lower, upper):
    """Adaptive integrals of the shifted density, with independent nodes."""
    a = -math.inf if lower is None else (lower - mu) / sigma
    b = math.inf if upper is None else (upper - mu) / sigma
    if a >= 0:
        anchor, sign, endpoint = lower, 1, a
        high = math.inf if upper is None else (upper - lower) / sigma
        scale = max(1., a)
        def density(t):
            return math.exp(-endpoint * t / scale - .5 * (t / scale) ** 2)
        width = high * scale
        norm = quad(density, 0, width, epsabs=1e-13, epsrel=1e-13)[0]
        average = quad(lambda t: t / scale * density(t), 0, width, epsabs=1e-13)[0] / norm
    elif b <= 0:
        return -conditional_oracle(-mu, sigma, None if upper is None else -upper,
                                   None if lower is None else -lower)
    else:
        anchor, sign = mu, 1
        def density(t):
            return math.exp(-.5 * t * t)
        norm = quad(density, a, b, epsabs=1e-13, epsrel=1e-13)[0]
        average = quad(lambda t: t * density(t), a, b, epsabs=1e-13)[0] / norm
    return anchor + sign * sigma * average


def clipped_oracle(mu, sigma, lower, upper):
    a = -math.inf if lower is None else (lower - mu) / sigma
    b = math.inf if upper is None else (upper - mu) / sigma
    result = quad(lambda z: (mu + sigma * z) * stats.norm.pdf(z), a, b,
                  epsabs=1e-12, epsrel=1e-12)[0]
    if lower is not None:
        result += lower * stats.norm.cdf(a)
    if upper is not None:
        result += upper * stats.norm.sf(b)
    return result


@pytest.fixture(scope="module", params=["tobit", "truncreg", "intreg"])
def fitted(request):
    rng = np.random.default_rng(7926)
    n = 450
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": np.resize(["B", "A", "C"], n),
                          "o": rng.uniform(-.2, .2, n), "w": rng.uniform(.4, 2., n)})
    latent = .5 + .7 * frame.x + .35 * (frame.g == "A") - .2 * (frame.g == "C") + frame.o + rng.normal(size=n)
    frame["y"] = latent.clip(0, 2.5) if request.param == "tobit" else latent
    kwargs = {"data": frame, "x": ["x", "g"], "categorical": ["g"],
              "offset": "o", "weights": "w", "weight_type": "iweight", "missing": "drop"}
    if request.param == "intreg":
        frame["lo"], frame["hi"] = np.floor(latent * 2) / 2, np.ceil(latent * 2) / 2
        model = oe.intreg(y_low="lo", y_high="hi", **kwargs)
    else:
        model = getattr(oe, request.param)(y="y", ll=0, ul=2.5, **kwargs)
    return frame, ResultBundle.model_validate_json(model.model_dump_json())


def matrix(frame, terms):
    columns = {"Intercept": np.ones(len(frame)), "x": frame.x.to_numpy(),
               "g[B]": (frame.g == "B").to_numpy(float), "g[C]": (frame.g == "C").to_numpy(float)}
    return np.column_stack([columns.get(name, np.zeros(len(frame))) for name in terms])


def oracle_mean(model, frame, beta, kind="response"):
    terms = [c.term for c in model.coefficients]
    mu = matrix(frame, terms) @ beta + frame.o.to_numpy()
    if kind in {"xb", "latent"} or model.spec.estimator == "intreg":
        return mu
    scale = beta[terms.index("/sigma")]
    lower, upper = model.extra["limits"].values()
    operation = conditional_oracle if kind == "conditional" or model.spec.estimator == "truncreg" else clipped_oracle
    return np.array([operation(value, scale, lower, upper) for value in mu])


def numerical_gradient(function, beta, step=2e-5):
    return np.column_stack([(function(beta + np.eye(len(beta))[i] * step)
                             - function(beta - np.eye(len(beta))[i] * step)) / (2 * step)
                            for i in range(len(beta))])


@pytest.mark.parametrize("kind", ["response", "latent", "conditional"])
def test_saved_new_rows_means_full_covariance_and_coefficient_permutations(fitted, kind):
    _, model = fitted
    if model.spec.estimator == "intreg" and kind == "conditional":
        with pytest.raises(AnalysisError, match="limits"):
            oe.predict(model, pd.DataFrame(), kind=kind)
        return
    new = pd.DataFrame({"x": [-1.2, .3, 1.4], "g": ["B", "C", "B"], "o": [.1, -.2, .05]})
    beta = np.array([c.estimate for c in model.coefficients])
    before = copy.deepcopy(model.model_dump())
    actual = oe.predict(model, new, kind=kind, interval="mean")
    expected = oracle_mean(model, new, beta, kind)
    gradient = numerical_gradient(lambda b: oracle_mean(model, new, b, kind), beta)
    variance = np.einsum("nk,kl,nl->n", gradient, model.covariance_matrix, gradient)
    assert_allclose(actual[kind], expected, atol=2e-12, rtol=2e-11)
    assert_allclose(actual.std_error, np.sqrt(variance), rtol=2e-8, atol=2e-10)
    terms = [c.term for c in model.coefficients]
    scale_index = terms.index("/lnsigma" if model.spec.estimator == "intreg" else "/sigma")
    if kind == "response" and model.spec.estimator != "intreg":
        assert np.max(np.abs(gradient[:, scale_index])) > .02
        diagonal_only = np.sum(gradient ** 2 * np.diag(model.covariance_matrix), axis=1)
        assert not np.allclose(variance, diagonal_only, rtol=1e-6)
    permutation = np.arange(len(beta))[::-1]
    changed = model.model_copy(deep=True)
    changed.coefficients = [changed.coefficients[i] for i in permutation]
    changed.covariance_matrix = np.array(changed.covariance_matrix)[permutation][:, permutation].tolist()
    assert_allclose(oe.predict(changed, new, kind=kind, interval="mean"), actual, atol=2e-12)
    assert model.model_dump() == before


def test_continuous_row_derivative_and_delta_inference(fitted):
    _, model = fitted
    new = pd.DataFrame({"x": [-1., .2, 1.], "g": ["B", "C", "B"], "o": [.05, -.1, .2]})
    beta = np.array([c.estimate for c in model.coefficients])

    def effect(b):
        plus, minus = new.copy(), new.copy()
        plus.x += 1e-4
        minus.x -= 1e-4
        return (oracle_mean(model, plus, b) - oracle_mean(model, minus, b)) / 2e-4

    expected = effect(beta)
    gradient = numerical_gradient(effect, beta, step=1e-4)
    actual = oe.predict(model, new, kind="derivative", term="x", interval="mean")
    assert_allclose(actual["dydx[x]"], expected, atol=3e-9, rtol=3e-8)
    assert_allclose(actual.std_error, np.sqrt(np.einsum("nk,kl,nl->n", gradient,
                    model.covariance_matrix, gradient)), rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize("method", ["ame", "mem"])
@pytest.mark.parametrize("kind", ["response", "conditional", "latent"])
def test_weighted_margins_continuous_categories_full_scale_gradient(fitted, method, kind):
    _, model = fitted
    if model.spec.estimator == "intreg" and kind == "conditional":
        return
    new = pd.DataFrame({"x": [-1., .2, 1.], "g": ["B", "C", "A"],
                        "o": [.05, -.1, .2], "w": [.3, 1., 2.]})
    terms = [c.term for c in model.coefficients]
    beta = np.array([c.estimate for c in model.coefficients])
    weights = new.w.to_numpy() / new.w.sum()

    def means(frame, b):
        if method == "ame":
            return weights @ oracle_mean(model, frame, b, kind)
        # Construct the oracle from the encoded design's weighted mean, including
        # fractional category proportions; do not choose a modal category.
        mu = float(weights @ matrix(frame, terms) @ b + weights @ frame.o)
        if kind == "latent" or model.spec.estimator == "intreg":
            return mu
        operation = conditional_oracle if kind == "conditional" or model.spec.estimator == "truncreg" else clipped_oracle
        return operation(mu, b[terms.index("/sigma")], *model.extra["limits"].values())

    def oracle(b):
        high, low = new.copy(), new.copy()
        high.x += 1e-4
        low.x -= 1e-4
        values = [(means(high, b) - means(low, b)) / 2e-4]
        baseline = new.copy()
        baseline.g = "A"
        for level in ["B", "C"]:
            alternative = new.copy()
            alternative.g = level
            values.append(means(alternative, b) - means(baseline, b))
        return np.array(values)

    actual = oe.margins(model, ["x", "g"], data=new, method=method, kind=kind)
    gradient = numerical_gradient(oracle, beta, step=1e-4)
    assert actual.variable.tolist() == ["x", "g[B]", "g[C]"]
    assert_allclose(actual.estimate, oracle(beta), rtol=3e-8, atol=2e-9)
    assert_allclose(actual.attrs["delta_gradients"], gradient, rtol=3e-5, atol=2e-7)
    assert_allclose(actual.std_error, np.sqrt(np.einsum("nk,kl,nl->n", gradient,
                    model.covariance_matrix, gradient)), rtol=3e-6, atol=2e-8)


def test_missing_duplicate_indexes_unseen_categories_and_outcome_not_required(fitted):
    _, model = fitted
    new = pd.DataFrame({"x": [0., np.nan, .5], "g": ["B", "A", "C"], "o": [.1, .2, -.1]},
                       index=["same", "same", "third"])
    actual = oe.predict(model, new)
    assert actual.index.tolist() == new.index.tolist()
    assert actual.attrs["missing_row_positions"] == [1]
    assert math.isnan(actual.response.iloc[1])
    assert np.isfinite(actual.response.iloc[[0, 2]]).all()
    bad = new.copy()
    bad.loc["third", "g"] = "not fitted"
    with pytest.raises(AnalysisError) as error:
        oe.predict(model, bad)
    assert error.value.code == "unknown_category"
    changed = model.model_copy(deep=True)
    changed.spec.missing = "raise"
    with pytest.raises(AnalysisError) as error:
        oe.predict(changed, new)
    assert error.value.code == "missing_values"
    if model.spec.estimator == "intreg":
        assert actual.attrs["response_definition"] == "latent unconditional normal mean E[Y*|X]"


@pytest.mark.parametrize("lower,upper", [(0., None), (None, 2.), (-1., 2.), (0., .00001)])
@pytest.mark.parametrize("mu", [-40., -4., 0., .000001, 1., 4., 40.])
def test_tail_and_narrow_conditional_means_and_analytic_derivatives(lower, upper, mu):
    normal = LimitedNormal("truncreg", 0, False, lower, upper)
    beta = torch.tensor([1.3], dtype=torch.float64)
    value, first, scale_first, second, cross = [float(t) for t in normal.evaluate(torch.tensor([mu], dtype=torch.float64), beta)]
    def oracle(m, s):
        return conditional_oracle(m, s, lower, upper)
    assert value == pytest.approx(oracle(mu, 1.3), rel=2e-10, abs=2e-11)
    step = 1e-3
    derivative = (oracle(mu + step, 1.3) - oracle(mu - step, 1.3)) / (2 * step)
    scale_derivative = (oracle(mu, 1.3 + step) - oracle(mu, 1.3 - step)) / (2 * step)
    curvature = (oracle(mu + step, 1.3) - 2 * oracle(mu, 1.3) + oracle(mu - step, 1.3)) / step ** 2
    mixed = (oracle(mu + step, 1.3 + step) - oracle(mu - step, 1.3 + step)
             - oracle(mu + step, 1.3 - step) + oracle(mu - step, 1.3 - step)) / (4 * step ** 2)
    assert first == pytest.approx(derivative, rel=3e-6, abs=3e-8)
    assert scale_first == pytest.approx(scale_derivative, rel=4e-6, abs=3e-8)
    assert second == pytest.approx(curvature, rel=2e-4, abs=3e-8)
    assert cross == pytest.approx(mixed, rel=2e-4, abs=3e-8)


@pytest.mark.parametrize("mu", [-50., -2., 0., 1., 2., 50.])
def test_censored_boundaries_and_saturated_tails_remain_differentiable(mu):
    normal = LimitedNormal("tobit", 0, False, 0., 2.)
    beta = torch.tensor([.7], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([mu], dtype=torch.float64, requires_grad=True)
    value, first, scale_first, second, cross = normal.evaluate(eta, beta)
    assert value.item() == pytest.approx(clipped_oracle(mu, .7, 0., 2.), abs=1e-12)
    actual_mu, actual_scale = torch.autograd.grad(value.sum(), (eta, beta), create_graph=True)
    assert_allclose(actual_mu.detach(), first.detach(), atol=3e-14)
    assert_allclose(actual_scale.detach(), scale_first.detach(), atol=3e-14)
    assert torch.isfinite(actual_mu).all() and torch.isfinite(actual_scale).all()
    assert torch.isfinite(second).all() and torch.isfinite(cross).all()


def test_corrupt_normal_scale_and_limits_rejected(fitted):
    _, model = fitted
    changed = model.model_copy(deep=True)
    scale = next(c for c in changed.coefficients if c.term in {"/sigma", "/lnsigma"})
    scale.estimate = 1000. if model.spec.estimator == "intreg" else -1.
    with pytest.raises(AnalysisError) as error:
        oe.predict(changed, pd.DataFrame())
    assert error.value.code == "invalid_result"
    if model.spec.estimator != "intreg":
        for limits in [None, {"lower": False, "upper": 2.5}, {"lower": 3., "upper": 2.5},
                       {"lower": -1., "upper": 2.5}]:
            changed = model.model_copy(deep=True)
            changed.extra["limits"] = limits
            with pytest.raises(AnalysisError) as error:
                oe.predict(changed, pd.DataFrame())
            assert error.value.code == "invalid_result"


def test_unrepresentable_standardized_conditional_domain_has_explicit_guard():
    normal = LimitedNormal("truncreg", 0, False, 0., None)
    with pytest.raises(AnalysisError) as error:
        normal.evaluate(torch.tensor([-1e101], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64))
    assert error.value.code == "prediction_precision"


@pytest.mark.parametrize("lower,upper", [(0., None), (None, 2.), (-1., 2.), (0., .00001)])
@pytest.mark.parametrize("mu", [-40., -4., 0., 1., 4., 40.])
def test_conditional_first_and_second_gradients_match_autograd_through_parameter_moments(lower, upper, mu):
    normal = LimitedNormal("truncreg", 0, False, lower, upper)
    beta = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([mu], dtype=torch.float64, requires_grad=True)
    value, first, scale_first, second, cross = normal.evaluate(eta, beta)
    actual_mu, actual_scale = torch.autograd.grad(value.sum(), (eta, beta), retain_graph=True)
    actual_second, actual_cross = torch.autograd.grad(first.sum(), (eta, beta))
    assert_allclose(actual_mu.detach(), first.detach(), rtol=3e-10, atol=4e-13)
    assert_allclose(actual_scale.detach(), scale_first.detach(), rtol=3e-10, atol=4e-13)
    assert_allclose(actual_second.detach(), second.detach(), rtol=3e-10, atol=4e-13)
    assert_allclose(actual_cross.detach(), cross.detach(), rtol=3e-10, atol=4e-13)


def test_saved_operations_never_call_fitters_or_optimizer(fitted, monkeypatch):
    _, model = fitted
    from openecon.econometrics.limited import censored, truncated

    def fail(*args, **kwargs):
        pytest.fail("Saved prediction must never refit or optimize.")

    for module, names in [(censored, ["fit_tobit", "fit_intreg", "_fit_censored"]),
                          (truncated, ["fit_truncreg", "run_newton"])]:
        for name in names:
            monkeypatch.setattr(module, name, fail)
    new = pd.DataFrame({"x": [0., 1.], "g": ["A", "C"], "o": [0., .1], "w": [1., 2.]})
    assert np.isfinite(oe.predict(model, new, interval="mean")).all().all()
    assert np.isfinite(oe.margins(model, ["x", "g"], data=new).estimate).all()


def test_outcome_derived_limits_persist_after_json_round_trip():
    rng = np.random.default_rng(395)
    frame = pd.DataFrame({"x": rng.normal(size=260)})
    frame["y"] = np.clip(.4 + .8 * frame.x + rng.normal(size=len(frame)), 0., 2.)
    model = oe.tobit(data=frame, y="y", x=["x"], ll="min", ul="max")
    saved = ResultBundle.model_validate_json(model.model_dump_json())
    assert saved.extra["limits"] == {"lower": 0., "upper": 2.}
    new = pd.DataFrame({"x": [-20., 0., 20.]})
    actual = oe.predict(saved, new)
    assert_allclose(actual.response.iloc[[0, 2]], [0., 2.], atol=1e-12)
    assert actual.response.iloc[1] > 0


def test_all_missing_prediction_rows_keep_positions(fitted):
    _, model = fitted
    new = pd.DataFrame({"x": [np.nan, np.nan], "g": ["A", "B"], "o": [0., 0.]},
                       index=[3, 3])
    actual = oe.predict(model, new, interval="mean")
    assert actual.index.tolist() == [3, 3]
    assert actual.isna().all().all()
    assert actual.attrs["missing_row_positions"] == [0, 1]


@pytest.mark.parametrize("distance", [1e8, 1e40, 1e90])
def test_far_conditional_tail_moments_use_rescaled_independent_integrals(distance):
    # u = distance * (Y - lower). Adaptive integration retains O(1) moments
    # independently of the production Legendre quadrature's adaptive scale.
    def density(u):
        return math.exp(-u - .5 * (u / distance) ** 2)

    normalizer = quad(density, 0, math.inf, epsabs=1e-13)[0]
    mean_u = quad(lambda u: u * density(u), 0, math.inf, epsabs=1e-13)[0] / normalizer
    moments = [quad(lambda u: (u - mean_u) ** order * density(u), 0, math.inf,
                    epsabs=1e-13)[0] / normalizer for order in (2, 3, 4)]
    v, m3, m4 = moments
    expected = [mean_u / distance, v / distance ** 2,
                (2 * v + m3 / distance ** 2) / distance,
                m3 / distance ** 3,
                (2 * m3 - 2 * v + (m4 - v * v) / distance ** 2) / distance ** 2]
    normal = LimitedNormal("truncreg", 0, False, 0., None)
    actual = [float(value) for value in normal.evaluate(torch.tensor([-distance], dtype=torch.float64),
                                                        torch.tensor([1.], dtype=torch.float64))]
    assert_allclose(actual, expected, rtol=2e-11, atol=0)


def test_quadrature_cache_primed_in_inference_mode_remains_valid_for_margins_gradients():
    from openecon.econometrics.postest.limited_prediction import _quadrature
    _quadrature.cache_clear()
    normal = LimitedNormal("truncreg", 0, False, 0., None)
    with torch.inference_mode():
        normal.evaluate(torch.tensor([5.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64))
    beta = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([5.], dtype=torch.float64, requires_grad=True)
    first = normal.evaluate(eta, beta)[1]
    gradients = torch.autograd.grad(first.sum(), (eta, beta))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def narrow_censored_decimal_oracle(mu, sigma, lower, upper):
    """260-digit endpoint algebra and integrated normal-density Taylor series.

    Decimal.from_float preserves the actual binary endpoint values; rounded
    decimal strings would erase the interval widths under review.
    """
    with localcontext() as context:
        context.prec = 260
        m, s, low, high = map(Decimal.from_float, (mu, sigma, lower, upper))
        a, width = (low - m) / s, (high - low) / s
        b = a + width
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
        density_a = (-a * a / 2).exp() / (2 * pi).sqrt()
        density_b = (-b * b / 2).exp() / (2 * pi).sqrt()
        previous2, previous = Decimal(0), Decimal(1)
        probability = width
        for order in range(1, 25):
            coefficient = (-a * previous - previous2) / order
            probability += coefficient * width ** (order + 1) / (order + 1)
            previous2, previous = previous, coefficient
        probability *= density_a
        scale_first = density_a - density_b
        mixed = (a * density_a - b * density_b) / s
        return np.array(list(map(float, (probability, scale_first, scale_first / s, mixed))))


@pytest.mark.parametrize("mu,sigma,lower,upper", [
    (0., 1., 1., 1. + 1e-15),
    (0., 1., 1., 1. + 1e-14),
    (0., 1., -1., -1. + 1e-15),
    (0., 1., -.000001, .000001),
    (1e-100, 1., -1e-100, 1e-100),
    (0., 1., 0., 1e-100),
    (.7, 1e100, -1., 1.),
])
def test_narrow_censored_mass_sigma_and_effect_gradients_against_decimal(mu, sigma, lower, upper):
    normal = LimitedNormal("tobit", 0, False, lower, upper)
    eta = torch.tensor([mu], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([sigma], dtype=torch.float64, requires_grad=True)
    mean, first, scale_first, second, mixed = normal.evaluate(eta, beta)
    expected = narrow_censored_decimal_oracle(mu, sigma, lower, upper)
    assert_allclose([first.item(), scale_first.item(), second.item(), mixed.item()], expected,
                    rtol=3e-14, atol=0)
    mean_mu, mean_sigma = torch.autograd.grad(mean.sum(), (eta, beta), create_graph=True, retain_graph=True)
    slope_mu, slope_sigma = torch.autograd.grad(first.sum(), (eta, beta), retain_graph=True)
    double_mu, double_sigma = torch.autograd.grad(mean_mu.sum(), (eta, beta))
    assert_allclose([mean_mu.item(), mean_sigma.item(), slope_mu.item(), slope_sigma.item()], expected,
                    rtol=3e-14, atol=0)
    assert_allclose([double_mu.item(), double_sigma.item()], expected[2:], rtol=3e-14, atol=0)


@pytest.mark.parametrize("sigma,width", [(1e300, 1e-100), (1e200, 1e-200), (1e200, 1e-110)])
def test_conditional_standardized_window_underflow_cannot_return_wrong_endpoint(sigma, width):
    normal = LimitedNormal("truncreg", 0, False, 0., width)
    with pytest.raises(AnalysisError) as error:
        normal.evaluate(torch.tensor([0.], dtype=torch.float64), torch.tensor([sigma], dtype=torch.float64))
    assert error.value.code == "prediction_precision"


def test_saved_requested_boolean_limit_cannot_alias_numeric_zero(fitted):
    _, model = fitted
    if model.spec.estimator == "intreg":
        return
    changed = model.model_copy(deep=True)
    changed.spec.options["ll"] = False
    with pytest.raises(AnalysisError) as error:
        oe.predict(changed, pd.DataFrame())
    assert error.value.code == "invalid_result"
