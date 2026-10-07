"""Independent likelihood, derivative and high-precision rare-tail oracles."""
import math

import mpmath as mp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import nbinom, poisson
from scipy.special import gammainc, gammaincc
import torch

from openecon.econometrics.count.censored_kernels import CensoredPieces, _nb_tails, _poisson_tails
from openecon.econometrics.count.kernels import NegBinDensity, PoissonDensity
from openecon.engines.contracts import KernelError


def distribution(theta, form):
    mu = np.exp(theta[0])
    if form == "poisson":
        return poisson(mu)
    alpha = np.exp(theta[1])
    shape = 1 / alpha if form == "mean" else mu / alpha
    return nbinom(shape, shape / (shape + mu))


def log_event(theta, lo, hi, form):
    dist = distribution(theta, form)
    if lo == hi:
        return dist.logpmf(lo)
    if math.isinf(hi):
        return dist.logsf(lo - 1)
    if lo == 0:
        return dist.logcdf(hi)
    return np.log(np.sum(dist.pmf(np.arange(lo, hi + 1))))


def finite_derivatives(function, theta, step=2e-4):
    theta = np.array(theta, float)
    d = len(theta)
    value = function(theta)
    gradient = np.zeros(d)
    hessian = np.zeros((d, d))
    for a in range(d):
        da = np.eye(d)[a] * step
        gradient[a] = (function(theta + da) - function(theta - da)) / (2 * step)
        hessian[a, a] = (function(theta + da) - 2 * value + function(theta - da)) / step**2
        for b in range(a + 1, d):
            db = np.eye(d)[b] * step
            hessian[a, b] = hessian[b, a] = (
                function(theta + da + db) - function(theta + da - db)
                - function(theta - da + db) + function(theta - da - db)) / (4 * step**2)
    return gradient, hessian


@pytest.mark.parametrize("form", ["poisson", "mean", "constant"])
@pytest.mark.parametrize("lo,hi", [(2, 2), (0, 2), (5, math.inf), (2, 6), (1, 95)])
def test_event_values_and_all_analytic_index_derivatives_match_independent_scipy(form, lo, hi):
    theta = [math.log(3.2)] + ([] if form == "poisson" else [math.log(.7)])
    density = PoissonDensity() if form == "poisson" else NegBinDensity(form)
    pieces = CensoredPieces(density, torch.tensor([lo], dtype=torch.float64),
                            torch.tensor([hi], dtype=torch.float64))
    index = [torch.tensor([theta[0]], dtype=torch.float64)]
    if len(theta) == 2:
        index.append(torch.tensor(theta[1], dtype=torch.float64))
    values, scores, curves = pieces(index)
    expected_g, expected_h = finite_derivatives(lambda t: log_event(t, lo, hi, form), theta)
    actual_h = np.zeros_like(expected_h)
    position = 0
    for a in range(len(theta)):
        for b in range(a, len(theta)):
            actual_h[a, b] = actual_h[b, a] = curves[position].item()
            position += 1
    assert values.item() == pytest.approx(log_event(theta, lo, hi, form), rel=2e-11, abs=1e-12)
    assert_allclose([value.item() for value in scores], expected_g, rtol=2e-6, atol=1e-7)
    assert_allclose(actual_h, expected_h, rtol=8e-5, atol=8e-7)
    assert pieces(index, False)[0].item() == pytest.approx(values.item(), rel=1e-12)


@pytest.mark.parametrize("form,mu,alpha,k,tail", [
    ("mean", 1e-8, .2, 100, "sf"),
    ("mean", .0001, .001, 50, "sf"),
    ("constant", 1000., .001, 5, "cdf"),
    ("poisson", .01, None, 500, "sf"),
    ("poisson", 3000., None, 0, "cdf"),
])
def test_rare_tails_remain_finite_when_ordinary_probabilities_underflow(form, mu, alpha, k, tail):
    with mp.workdps(85):
        m = mp.mpf(str(mu))
        if form == "poisson":
            expected = mp.log(mp.gammainc(k + 1, 0, m) / mp.gamma(k + 1)) if tail == "sf" \
                else mp.log(mp.gammainc(k + 1, m, mp.inf) / mp.gamma(k + 1))
            pair = _poisson_tails(torch.tensor([float(k)], dtype=torch.float64),
                                   torch.tensor([math.log(mu)], dtype=torch.float64), True)
        else:
            a = mp.mpf(str(alpha))
            shape = 1 / a if form == "mean" else m / a
            p, q = shape / (shape + m), m / (shape + m)
            expected = mp.log(mp.betainc(k + 1, shape, 0, q, regularized=True)) if tail == "sf" \
                else mp.log(mp.betainc(shape, k + 1, 0, p, regularized=True))
            pair = _nb_tails(torch.tensor([float(k)], dtype=torch.float64),
                             torch.tensor([math.log(mu)], dtype=torch.float64),
                             torch.tensor(math.log(alpha), dtype=torch.float64), form, True)
        result = pair[tail == "sf"]
        assert result.v.item() == pytest.approx(float(expected), rel=8e-13, abs=1e-9)
        assert torch.isfinite(result.g).all() and torch.isfinite(result.h).all()
        assert result.v.item() < -500


@pytest.mark.parametrize("form", ["poisson", "mean", "constant"])
def test_event_whole_support_is_exactly_uninformative(form):
    density = PoissonDensity() if form == "poisson" else NegBinDensity(form)
    pieces = CensoredPieces(density, torch.tensor([0.], dtype=torch.float64),
                            torch.tensor([math.inf], dtype=torch.float64))
    index = [torch.tensor([1.], dtype=torch.float64)]
    if density.size == 2:
        index.append(torch.tensor(0., dtype=torch.float64))
    value, score, hessian = pieces(index)
    assert value.item() == 0 and all(part.item() == 0 for part in [*score, *hessian])


def test_special_function_budget_raises_instead_of_returning_a_capped_tail(monkeypatch):
    import openecon.econometrics.count.censored_kernels as kernels
    from openecon.engines.contracts import KernelError
    monkeypatch.setattr(kernels, "_MAX_ITER", 1)
    with pytest.raises(KernelError) as error:
        _nb_tails(torch.tensor([30.], dtype=torch.float64), torch.tensor([3.], dtype=torch.float64),
                  torch.tensor(0., dtype=torch.float64), "mean", True)
    assert error.value.code == "tail_nonconvergence"


@pytest.mark.parametrize("count", [10**9, 10**12, 2**53 - 1, 2**53])
@pytest.mark.parametrize("standard_deviations", [-2, 0, 2])
def test_large_poisson_point_masses_match_high_precision_with_native_mean_rounding_bound(count, standard_deviations):
    eta = math.log(count + standard_deviations * math.sqrt(count))
    index = torch.tensor([eta], dtype=torch.float64)
    bounds = torch.tensor([float(count)], dtype=torch.float64)
    value, scores, curves = CensoredPieces(PoissonDensity(), bounds, bounds)([index])
    native_mu = torch.exp(index).item()
    with mp.workdps(85):
        mu = mp.exp(mp.mpf(eta))
        expected = count * mp.mpf(eta) - mu - mp.loggamma(count + 1)
        m = mp.mpf(native_mu)
        native_expected = count * mp.log(m) - m - mp.loggamma(count + 1)
    assert value.item() == pytest.approx(float(native_expected), abs=5e-13)
    exponential_rounding = math.ulp(native_mu) * abs(float(count) / native_mu - 1)
    assert abs(value.item() - float(expected)) <= exponential_rounding + 5e-13
    # Derivatives use the actual native float64 exponential, with at most one
    # ulp of mean rounding, rather than a cancellation-prone log-factorial.
    assert scores[0].item() == float(count) - native_mu
    assert curves[0].item() == -native_mu


@pytest.mark.parametrize("count", [10**9, 10**12])
@pytest.mark.parametrize("standard_deviations", [-10, -8, -5, -3, -1, 0, 1, 3, 5, 8, 10])
def test_large_poisson_cumulative_tails_and_hazard_derivatives_match_independent_oracles(count, standard_deviations):
    eta = torch.tensor([math.log(count + standard_deviations * math.sqrt(count))], dtype=torch.float64)
    mu = torch.exp(eta).item()
    left, right = _poisson_tails(torch.tensor([float(count)], dtype=torch.float64), eta, True)
    probabilities = [gammaincc(count + 1, mu), gammainc(count + 1, mu)]
    with mp.workdps(85):
        m = mp.mpf(mu)
        boundary = float(mp.log(m) + count * mp.log(m) - m - mp.loggamma(count + 1))
    for tail, probability, sign in zip((left, right), probabilities, (-1, 1), strict=True):
        assert math.exp(tail.v.item()) == pytest.approx(probability, abs=4e-11)
        assert tail.v.item() == pytest.approx(math.log(probability), abs=1e-9)
        slope = sign * math.exp(boundary - math.log(probability))
        hessian = slope * ((float(count) - mu) + 1) - slope**2
        assert tail.g.item() == pytest.approx(slope, rel=2e-9, abs=1e-13)
        assert tail.h.item() == pytest.approx(hessian, rel=2e-7, abs=1e-8)


def test_large_poisson_underflow_tail_uses_the_stable_prefactor():
    count = 10**9
    eta = torch.tensor([math.log(.9 * count)], dtype=torch.float64)
    mu = torch.exp(eta).item()
    tail = _poisson_tails(torch.tensor([float(count)], dtype=torch.float64), eta, True)[1]
    with mp.workdps(85):
        m = mp.mpf(mu)
        # Independent hypergeometric identity for the lower incomplete gamma;
        # ordinary SciPy survival probability has already underflowed here.
        expected = -m + (count + 1) * mp.log(m) - mp.loggamma(count + 2) \
            + mp.log(mp.hyp1f1(1, count + 2, m))
    assert tail.v.item() == pytest.approx(float(expected), abs=2e-8)
    assert tail.v.item() < -5e6
    assert torch.isfinite(tail.g).all() and torch.isfinite(tail.h).all()


@pytest.mark.parametrize("form,alpha", [("mean", .1), ("constant", .2)])
def test_large_nb_mass_and_derivatives_inside_declared_region_match_high_precision(form, alpha):
    count, eta, tau = 10**6, math.log(10**6), math.log(alpha)
    bounds = torch.tensor([float(count)], dtype=torch.float64)
    value, gradient, curves = CensoredPieces(NegBinDensity(form), bounds, bounds)([
        torch.tensor([eta], dtype=torch.float64), torch.tensor(tau, dtype=torch.float64)])
    with mp.workdps(85):
        def function(e, t):
            mu, dispersion = mp.exp(e), mp.exp(t)
            shape = 1 / dispersion if form == "mean" else mu / dispersion
            return mp.loggamma(count + shape) - mp.loggamma(shape) - mp.loggamma(count + 1) \
                + shape * mp.log(shape / (shape + mu)) + count * mp.log(mu / (shape + mu))
        e, t = mp.mpf(eta), mp.mpf(tau)
        expected = function(e, t)
        scores = [mp.diff(lambda e: function(e, t), e), mp.diff(lambda t: function(e, t), t)]
        hessians = [mp.diff(lambda e: function(e, t), e, 2),
                    mp.diff(lambda e: mp.diff(lambda t: function(e, t), t), e),
                    mp.diff(lambda t: function(e, t), t, 2)]
    assert value.item() == pytest.approx(float(expected), abs=1e-8)
    assert_allclose([part.item() for part in gradient], [float(part) for part in scores], rtol=1e-8, atol=1e-8)
    assert_allclose([part.item() for part in curves], [float(part) for part in hessians], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("form,mu,alpha,count", [
    ("constant", 1e5, .001, 5),  # NB1 shape=1e8, old rare-tail fixture.
    ("mean", 3., 1e-9, 5),
    ("mean", 1e9, .3, 10**9),
    ("constant", 1e12, 1., 10**12),
])
def test_nb_unverified_gamma_scale_fails_explicitly_instead_of_returning_rounded_values(form, mu, alpha, count):
    with pytest.raises(KernelError) as error:
        _nb_tails(torch.tensor([float(count)], dtype=torch.float64),
                  torch.tensor([math.log(mu)], dtype=torch.float64),
                  torch.tensor(math.log(alpha), dtype=torch.float64), form, True)
    assert error.value.code == "precision_unsupported"


def test_exact_poisson_maximum_count_and_short_interval_do_not_require_unsupported_cdf():
    index = [torch.tensor([math.log(2**53)], dtype=torch.float64)]
    pieces = CensoredPieces(PoissonDensity(), torch.tensor([2**53 - 1.], dtype=torch.float64),
                            torch.tensor([2**53], dtype=torch.float64))
    result = pieces(index)
    assert result is not None and torch.isfinite(result[0]).all()
    with pytest.raises(KernelError) as error:
        CensoredPieces(PoissonDensity(), torch.tensor([0.], dtype=torch.float64),
                        torch.tensor([2**53], dtype=torch.float64))
    assert error.value.code == "precision_unsupported"


def test_nb_optimizer_precision_trial_is_rejected_but_direct_evaluation_is_explicit():
    pieces = CensoredPieces(NegBinDensity("constant"), torch.tensor([5.], dtype=torch.float64),
                            torch.tensor([5.], dtype=torch.float64))
    index = [torch.tensor([math.log(1e5)], dtype=torch.float64), torch.tensor(math.log(.001), dtype=torch.float64)]
    with pytest.raises(KernelError) as error:
        pieces(index)
    assert error.value.code == "precision_unsupported"
    pieces.reject_precision_trials = True
    assert pieces(index) is None and pieces.precision_rejected
    with pytest.raises(KernelError) as error:
        pieces.check_precision(index)
    assert error.value.code == "precision_unsupported"


def test_small_dispersion_trial_tracks_precision_before_generic_domain_rejection():
    pieces = CensoredPieces(NegBinDensity("mean"), torch.tensor([2.], dtype=torch.float64),
                            torch.tensor([2.], dtype=torch.float64))
    pieces.reject_precision_trials = True
    index = [torch.tensor([.5], dtype=torch.float64), torch.tensor(-25., dtype=torch.float64)]
    assert pieces(index) is None and pieces.precision_rejected
