"""Independent SciPy likelihood, finite-difference information and curve oracles.

Four fixed seeds x four laws x four noninformative coarsening patterns provide
64 reproducible fit cells. SciPy is used only here, never by the runtime.
"""

import copy
import json
import math

import numpy as np
import pytest
from scipy import integrate, optimize, stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext import interval as model
from openecon.econometrics.survival_ext.common import interval_data


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


FAMILIES = ["exponential", "weibull", "lognormal", "loglogistic"]
TIMES = [0.0, 0.4, 1.2, 3.0, 8.0]


def distribution(family, raw):
    if family == "exponential":
        return stats.expon(scale=np.exp(-raw[0]))
    if family == "weibull":
        return stats.weibull_min(np.exp(raw[0]), scale=np.exp(raw[1]))
    if family == "lognormal":
        return stats.lognorm(np.exp(raw[1]), scale=np.exp(raw[0]))
    return stats.fisk(np.exp(raw[0]), scale=np.exp(raw[1]))


def sample(family, seed, pattern, n=80):
    rng = np.random.default_rng(seed)
    truth = {
        "exponential": [math.log(0.3)],
        "weibull": [math.log(1.5), math.log(3.0)],
        "lognormal": [1.0, math.log(0.65)],
        "loglogistic": [math.log(2.0), math.log(3.0)],
    }[family]
    times = distribution(family, truth).rvs(n, random_state=rng)
    lower, upper = [], []
    for i, t in enumerate(times):
        if pattern == "interval" or (pattern == "mixed" and i % 4 == 1):
            lo = math.floor(t / 0.75) * 0.75
            lower.append(lo)
            upper.append(lo + 0.75)
        elif pattern == "right" or (pattern == "mixed" and i % 4 == 2):
            lower.append(min(t, 3.0))
            upper.append(t if t <= 3 else None)
        elif pattern == "mixed" and i % 4 == 0 and t <= 1.5:
            lower.append(0.0)
            upper.append(1.5)
        else:
            lower.append(t)
            upper.append(t)
    return lower, upper


def likelihood(family, raw, lower, upper):
    law = distribution(family, raw)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray([np.inf if v is None else v for v in upper], dtype=float)
    right = np.isinf(hi)
    exact = lo == hi
    left = (lo == 0) & (~right)
    intervals = ~(right | exact | left)
    total = float(
        law.logsf(lo[right]).sum() + law.logpdf(lo[exact]).sum() + law.logcdf(hi[left]).sum()
    )
    if np.any(intervals):
        left_end, h = lo[intervals], hi[intervals]
        lower_tail = law.cdf(h) < 0.5
        a, b = law.logcdf(h[lower_tail]), law.logcdf(left_end[lower_tail])
        total += float(np.sum(a + np.log(-np.expm1(b - a))))
        a, b = law.logsf(left_end[~lower_tail]), law.logsf(h[~lower_tail])
        total += float(np.sum(a + np.log(-np.expm1(b - a))))
    return total


def derivative(function, point, step=1e-5):
    result = []
    for j in range(len(point)):
        delta = np.zeros_like(point)
        delta[j] = step
        result.append((function(point + delta) - function(point - delta)) / (2 * step))
    return np.asarray(result)


def information(function, point):
    k = len(point)

    def hessian(step):
        answer = np.zeros((k, k))
        baseline = function(point)
        for i in range(k):
            ei = np.zeros(k)
            ei[i] = step
            answer[i, i] = (function(point + ei) - 2 * baseline + function(point - ei)) / step**2
            for j in range(i):
                ej = np.zeros(k)
                ej[j] = step
                answer[i, j] = answer[j, i] = (
                    function(point + ei + ej)
                    - function(point + ei - ej)
                    - function(point - ei + ej)
                    + function(point - ei - ej)
                ) / (4 * step**2)
        return answer

    coarse, fine = hessian(1e-3), hessian(5e-4)
    return -(4 * fine - coarse) / 3


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pattern", ["exact", "mixed", "interval", "right"])
@pytest.mark.parametrize("seed", [302111, 302113, 302117, 302119])
def test_independent_likelihood_information_full_covariance_and_curves(family, pattern, seed):
    lower, upper = sample(family, seed, pattern)
    fit = getattr(model, "stinterval_" + family)(lower, upper, times=TIMES)
    state = fit.attrs["prediction_state"]
    raw = np.asarray(state["raw_parameters"])

    def function(point):
        return likelihood(family, point, lower, upper)

    assert fit.attrs["log_likelihood"] == pytest.approx(function(raw), abs=3e-10)
    # A separate derivative-free optimization starts from a perturbed point;
    # no Torch derivatives, fitted Hessian, or optimizer implementation is reused.
    oracle = optimize.minimize(
        lambda point: -function(point),
        raw + 0.12,
        method="Powell",
        options=dict(xtol=1e-10, ftol=1e-12, maxiter=1000),
    )
    assert oracle.success
    np.testing.assert_allclose(raw, oracle.x, atol=2e-6, rtol=2e-6)
    assert np.max(np.abs(derivative(function, raw))) < 1e-4
    observed = information(function, raw)
    expected_cov = np.linalg.inv(observed)
    raw_cov = np.asarray(state["raw_covariance"])
    np.testing.assert_allclose(raw_cov, expected_cov, rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(fit.attrs["observed_information"], observed, rtol=3e-5, atol=3e-5)
    natural = np.asarray(state["natural_parameters"])
    scales = natural.copy()
    if family == "lognormal":
        scales[0] = 1.0
    np.testing.assert_allclose(
        state["natural_covariance"],
        expected_cov * scales[:, None] * scales[None, :],
        rtol=3e-5,
        atol=3e-7,
    )
    law = distribution(family, raw)
    expected_survival = law.sf(TIMES)
    np.testing.assert_allclose(
        fit["survival"]["survival"], expected_survival, atol=2e-12, rtol=2e-12
    )
    jacobian = np.column_stack(
        [derivative(lambda theta: distribution(family, theta).sf(time), raw) for time in TIMES]
    ).T
    expected_joint = jacobian @ raw_cov @ jacobian.T
    np.testing.assert_allclose(
        fit.attrs["survival_covariance"], expected_joint, atol=1e-10, rtol=3e-7
    )
    np.testing.assert_allclose(
        np.asarray(fit["survival"]["se"]) ** 2, np.diag(expected_joint), atol=1e-10, rtol=3e-7
    )
    np.testing.assert_allclose(fit["survival"]["cdf"], 1 - expected_survival, atol=2e-12)
    for i, s in enumerate(expected_survival):
        if s in (0.0, 1.0):
            continue
        transformed = math.log(-math.log(s))
        transformed_se = math.sqrt(expected_joint[i, i]) / abs(s * math.log(s))
        critical = stats.norm.ppf(0.975)
        expected_lower = math.exp(-math.exp(transformed + critical * transformed_se))
        expected_upper = math.exp(-math.exp(transformed - critical * transformed_se))
        assert float(fit["survival"].iloc[i]["lower"]) == pytest.approx(expected_lower, abs=2e-9)
        assert float(fit["survival"].iloc[i]["upper"]) == pytest.approx(expected_upper, abs=2e-9)
    assert fit.attrs["scaled_score"] <= 1e-10
    assert fit.attrs["converged"]
    assert len(fit["observations"]) == len(lower)
    assert len(fit["survival_covariance"]) == len(TIMES) ** 2
    saved = json.loads(json.dumps(fit.attrs, allow_nan=False))
    replay = model.interval_survival_predict(saved, times=TIMES)
    np.testing.assert_array_equal(replay["survival"]["survival"], fit["survival"]["survival"])
    np.testing.assert_array_equal(
        replay.attrs["survival_covariance"], fit.attrs["survival_covariance"]
    )
    assert "survival" in fit.to_latex() and "log\\_covariance" in fit.to_latex()


def test_exponential_exact_right_and_overlapping_interval_analytic_laws():
    fit = model.stinterval_exponential([1, 2, 3, 4], [1, None, 3, None], times=TIMES)
    rate = 2 / 10
    assert fit.attrs["prediction_state"]["natural_parameters"][0] == pytest.approx(rate, abs=1e-11)
    assert fit.attrs["prediction_state"]["raw_covariance"][0][0] == pytest.approx(0.5, abs=1e-10)
    assert fit.attrs["log_likelihood"] == pytest.approx(2 * math.log(rate) - rate * 10, abs=1e-11)
    overlapping = model.stinterval_exponential([1.0] * 8, [2.0] * 8, times=TIMES)
    assert overlapping.attrs["prediction_state"]["natural_parameters"][0] == pytest.approx(
        math.log(2), abs=2e-10
    )
    assert overlapping.attrs["log_likelihood"] == pytest.approx(8 * math.log(0.25), abs=2e-10)


def test_lognormal_complete_data_closed_form_and_time_density_jacobian():
    t = np.array([0.2, 0.5, 1.0, 2.0, 4.0, 9.0])
    fit = model.stinterval_lognormal(t, t, times=TIMES)
    mu = float(np.log(t).mean())
    sd = float(np.log(t).std())
    state = fit.attrs["prediction_state"]
    np.testing.assert_allclose(state["natural_parameters"], [mu, sd], rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(
        state["raw_covariance"], [[sd * sd / len(t), 0], [0, 1 / (2 * len(t))]], atol=2e-9
    )
    assert fit.attrs["log_likelihood"] == pytest.approx(
        stats.lognorm(sd, scale=np.exp(mu)).logpdf(t).sum(), abs=2e-10
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_global_meta_and_permutation_invariance(family):
    lower, upper = sample(family, 12345, "mixed", n=40)
    baseline = getattr(model, "stinterval_" + family)(lower, upper, times=TIMES)
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        actual = getattr(model, "stinterval_" + family)(lower, upper, times=TIMES)
        restored = model.interval_survival_predict(actual, times=TIMES)
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_device(previous)
    np.testing.assert_array_equal(
        actual.attrs["prediction_state"]["raw_parameters"],
        baseline.attrs["prediction_state"]["raw_parameters"],
    )
    np.testing.assert_array_equal(
        restored.attrs["survival_covariance"], baseline.attrs["survival_covariance"]
    )
    perm = np.random.default_rng(12).permutation(len(lower))
    shuffled = getattr(model, "stinterval_" + family)(
        [lower[i] for i in perm], [upper[i] for i in perm], times=TIMES
    )
    np.testing.assert_allclose(
        shuffled.attrs["prediction_state"]["raw_parameters"],
        baseline.attrs["prediction_state"]["raw_parameters"],
        atol=2e-8,
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lower,upper", [([1, 2, 3], [None, None, None]), ([0, 0, 0], [1, 2, 3])])
def test_all_one_sided_censored_no_finite_mle_refused(family, lower, upper):
    with pytest.raises(AnalysisError, match="finite"):
        getattr(model, "stinterval_" + family)(lower, upper, times=TIMES)


@pytest.mark.parametrize("family", ["weibull", "lognormal", "loglogistic"])
@pytest.mark.parametrize(
    "lower,upper", [([1, 1, 1], [2, 2, 2]), ([1, 1, 1], [1, 1, None]), ([0, 1, 1], [2, 2, None])]
)
def test_concentration_boundary_has_no_finite_two_parameter_mle(family, lower, upper):
    with pytest.raises(AnalysisError, match="finite"):
        getattr(model, "stinterval_" + family)(lower, upper, times=TIMES)


@pytest.mark.parametrize(
    "change",
    ["raw_parameter", "raw_covariance", "natural_covariance", "names", "bool", "schema", "extra"],
)
def test_saved_state_tampering_and_inconsistent_coordinates_refused(change):
    lower, upper = sample("weibull", 31415, "mixed", n=40)
    fit = model.stinterval_weibull(lower, upper, times=TIMES)
    saved = copy.deepcopy(fit.attrs)
    state = saved["prediction_state"]
    if change == "raw_parameter":
        state["raw_parameters"][0] += 0.1
    elif change == "raw_covariance":
        state["raw_covariance"][0][1] = 0
        state["checksum"] = model._checksum(state)
    elif change == "natural_covariance":
        state["natural_covariance"][0][1] += 1
        state["checksum"] = model._checksum(state)
    elif change == "names":
        state["raw_names"][0] = "wrong"
        state["checksum"] = model._checksum(state)
    elif change == "bool":
        state["raw_parameters"][0] = True
        state["checksum"] = model._checksum(state)
    elif change == "schema":
        state["schema"] = "unknown"
    else:
        state["unexpected"] = 1
    with pytest.raises(AnalysisError):
        model.interval_survival_predict(saved, times=TIMES)


@pytest.mark.parametrize("family", FAMILIES)
def test_input_options_and_output_budget_refusals(family):
    fn = getattr(model, "stinterval_" + family)
    for kwargs in [
        dict(device="cuda"),
        dict(weights=[1] * 4),
        dict(maxiter=True),
        dict(maxiter=0),
        dict(level=1.0),
        dict(times=[2, 1]),
    ]:
        with pytest.raises(AnalysisError):
            fn([1, 2, 3, 4], [1, 2, 3, 4], **kwargs)
    for lower, upper in [
        ([1, float("nan"), 3], [1, 2, 3]),
        ([0, 2, 3], [None, 2, 3]),
        ([1, 2, 3], [1, 1, 3]),
        ([True, 2, 3], [1, 2, 3]),
    ]:
        with pytest.raises(AnalysisError):
            fn(lower, upper, times=TIMES)
    with pytest.raises(AnalysisError):
        fn([1, 2, 3], [1, 2, 3], times=list(np.linspace(0, 10, 257)))


@pytest.mark.parametrize("family", FAMILIES)
def test_tail_and_narrow_interval_log_likelihoods_against_independent_laws(family):
    raw = {
        "exponential": [0.0],
        "weibull": [math.log(1.5), 0.0],
        "lognormal": [0.0, math.log(0.8)],
        "loglogistic": [math.log(2.0), 0.0],
    }[family]
    lower = [1e-10, 1.0, 10.0, 100.0, 1e6]
    upper = [1e-9, 1.00001, 10.00001, None, 1e6 + 1.0]
    data = interval_data(lower, upper)
    runtime = model._contributions(
        family, torch.tensor(raw, dtype=torch.float64, device="cpu"), data
    )
    assert bool(torch.isfinite(runtime).all())
    if family == "loglogistic":
        # SciPy fisk's generic survival subtraction loses the narrow far-tail
        # mass here; independently integrate its continuous density instead.
        law = distribution(family, raw)
        oracle = sum(
            law.logsf(lo)
            if hi is None
            else math.log(integrate.quad(law.pdf, lo, hi, epsabs=1e-250, epsrel=2e-12)[0])
            for lo, hi in zip(lower, upper)
        )
    else:
        oracle = likelihood(family, raw, lower, upper)
    assert float(runtime.sum()) == pytest.approx(oracle, rel=2e-9, abs=2e-7)


@pytest.mark.parametrize("family", FAMILIES)
def test_one_ulp_endpoint_probability_and_autodiff_are_finite(family):
    hi = 1e12
    lo = np.nextafter(hi, 0.0)
    raw = {
        "exponential": [-math.log(hi)],
        "weibull": [math.log(1.5), math.log(hi)],
        "lognormal": [math.log(hi), math.log(0.8)],
        "loglogistic": [math.log(2.0), math.log(hi)],
    }[family]
    data = interval_data([lo], [hi])
    theta = torch.tensor(raw, dtype=torch.float64, device="cpu", requires_grad=True)

    def function(parameter):
        return model._contributions(family, parameter, data).sum()

    actual = function(theta)
    probability = integrate.quad(
        distribution(family, raw).pdf, lo, hi, epsabs=1e-200, epsrel=1e-12
    )[0]
    assert float(actual.detach()) == pytest.approx(math.log(probability), abs=2e-11)
    assert bool(torch.isfinite(torch.autograd.grad(actual, theta)[0]).all())
    assert bool(torch.isfinite(torch.autograd.functional.hessian(function, theta)).all())


def test_maximum_sample_complete_density_likelihood_and_resource_gate():
    t = np.linspace(0.5, 8.0, 4096)
    result = model.stinterval_exponential(t, t, times=TIMES)
    rate = 1 / float(t.mean())
    assert result.attrs["prediction_state"]["natural_parameters"][0] == pytest.approx(
        rate, rel=2e-10
    )
    assert result.attrs["prediction_state"]["raw_covariance"][0][0] == pytest.approx(
        1 / len(t), rel=2e-10
    )
    assert len(result["observations"]) == len(t)
    assert result.attrs["log_likelihood"] == pytest.approx(
        len(t) * math.log(rate) - rate * t.sum(), abs=2e-8
    )
    with pytest.raises(AnalysisError, match="4096"):
        model.stinterval_exponential([1.0] * 4097, [1.0] * 4097, times=TIMES)


def test_exhausted_iteration_budget_never_returns_fit():
    lower, upper = sample("weibull", 456789, "mixed")
    with pytest.raises(AnalysisError, match="stationary"):
        model.stinterval_weibull(lower, upper, times=TIMES, maxiter=1)


def test_underflowing_exponential_survival_keeps_representable_confidence_limit():
    result = model.stinterval_exponential([0.2, 1.8], [0.2, 1.8], times=[1000.0])
    curve = result["survival"].iloc[0]
    expected = math.exp(-1000 * math.exp(-stats.norm.ppf(0.975) * math.sqrt(0.5)))
    assert curve["survival"] == 0 and curve["se"] == 0
    assert curve["upper"] == pytest.approx(expected, rel=2e-11, abs=0)
    restored = model.interval_survival_predict(json.loads(json.dumps(result.attrs)), times=[1000.0])
    assert restored["survival"].iloc[0]["upper"] == curve["upper"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("tail", ["rounded_one", "underflow"])
def test_direct_transformed_tail_inference_against_analytic_gradient(family, tail):
    if tail == "rounded_one":
        time = 1e-12
        raw = {
            "exponential": [-20.0],
            "weibull": [math.log(1.5), 20.0],
            "lognormal": [20.0, 0.0],
            "loglogistic": [math.log(1.5), 20.0],
        }[family]
    else:
        time = 1e12 if family == "lognormal" else 1000.0
        raw = {
            "exponential": [0.0],
            "weibull": [0.0, 0.0],
            "lognormal": [0.0, math.log(0.5)],
            "loglogistic": [math.log(200.0), 0.0],
        }[family]
    covariance = (
        np.array([[0.1]]) if family == "exponential" else np.array([[0.1, 0.02], [0.02, 0.2]])
    )
    a = raw[0]
    if family == "exponential":
        g = a + math.log(time)
        derivative = np.array([1.0])
    elif family == "weibull":
        shape = math.exp(a)
        g = shape * (math.log(time) - raw[1])
        derivative = np.array([g, -shape])
    elif family == "lognormal":
        sigma = math.exp(raw[1])
        z = (math.log(time) - a) / sigma
        if stats.norm.logcdf(z) < -36:
            g = stats.norm.logcdf(z)
            ratio = math.exp(stats.norm.logpdf(z) - g)
        else:
            h = -stats.norm.logsf(z)
            g = math.log(h)
            ratio = math.exp(stats.norm.logpdf(z) - stats.norm.logsf(z)) / h
        derivative = np.array([-ratio / sigma, -ratio * z])
    else:
        shape = math.exp(a)
        z = shape * (math.log(time) - raw[1])
        if z < -36:
            g = z
            ratio = 1.0
        else:
            h = np.logaddexp(0, z)
            g = math.log(h)
            ratio = stats.logistic.cdf(z) / h
        derivative = ratio * np.array([z, -shape])
    transformed_se = math.sqrt(derivative @ covariance @ derivative)
    critical = stats.norm.ppf(0.975)
    low = math.exp(-math.exp(min(709.0, g + critical * transformed_se)))
    high = math.exp(-math.exp(max(-745.0, g - critical * transformed_se)))
    curves, _, _ = model._curves(
        family,
        torch.tensor(raw, dtype=torch.float64, device="cpu"),
        torch.tensor(covariance, dtype=torch.float64, device="cpu"),
        torch.tensor([time], dtype=torch.float64, device="cpu"),
        0.95,
        critical,
    )
    row = curves[0]
    assert row["survival"] == (1.0 if tail == "rounded_one" else 0.0)
    assert row["log_cumulative_hazard"] == pytest.approx(g, abs=2e-10)
    assert row["log_cumulative_hazard_se"] == pytest.approx(transformed_se, rel=2e-7, abs=1e-10)
    assert row["lower"] == pytest.approx(low, rel=2e-7, abs=1e-14)
    assert row["upper"] == pytest.approx(high, rel=2e-7, abs=1e-14)
