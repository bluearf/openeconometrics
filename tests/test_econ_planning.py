"""Independent probability-law oracles and design-inversion boundaries.

SciPy is test-only. The binomial oracle enumerates its own rejection set and
all smaller sizes; the production solver has no SciPy dependency.
"""
import json
import math

import numpy as np
import pytest
from scipy import integrate, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def plan(result):
    return result["plan"].iloc[0]


def normal_power(shift, alpha, alternative):
    cut = stats.norm.isf(alpha / (2 if alternative == "two-sided" else 1))
    if alternative == "upper":
        return stats.norm.sf(cut, loc=shift)
    if alternative == "lower":
        return stats.norm.cdf(-cut, loc=shift)
    return stats.norm.cdf(-cut, loc=shift) + stats.norm.sf(cut, loc=shift)


def exact_region(n, p0, alpha, alternative):
    k = np.arange(n + 1)
    tail = alpha / (2 if alternative == "two-sided" else 1)
    rejection = np.zeros(n + 1, dtype=bool)
    if alternative != "upper":
        rejection |= stats.binom.cdf(k, n, p0) <= tail
    if alternative != "lower":
        rejection |= stats.binom.sf(k - 1, n, p0) <= tail
    return k[rejection]


@pytest.mark.parametrize("alternative", ["upper", "lower", "two-sided"])
@pytest.mark.parametrize("sd2,ratio", [(None, 1.), (2., 1.), (.4, .37), (3., 2.3)])
def test_mean_law_inversion_and_minimal_integer(alternative, sd2, ratio):
    effect = -.7 if alternative == "lower" else .7
    n, sd, alpha = 31, 1.3, .013
    result = oe.power_mean(effect, sd=sd, sd2=sd2, ratio=ratio, n=n,
                           alpha=alpha, alternative=alternative)
    n2 = math.ceil(ratio * n) if sd2 is not None else None
    variance = sd**2 / n + (sd2**2 / n2 if n2 else 0)
    oracle = normal_power(effect / math.sqrt(variance), alpha, alternative)
    assert plan(result).power == pytest.approx(oracle, abs=2e-14)
    assert plan(result).standard_error == pytest.approx(math.sqrt(variance))
    sized = oe.power_mean(effect, sd=sd, sd2=sd2, ratio=ratio, power=.81,
                          alpha=alpha, alternative=alternative)
    required = int(plan(sized).n)
    assert plan(sized).power >= .81
    if required > 1:
        assert plan(oe.power_mean(effect, sd=sd, sd2=sd2, ratio=ratio, n=required - 1,
                                 alpha=alpha, alternative=alternative)).power < .81
    mde = oe.power_mean(sd=sd, sd2=sd2, ratio=ratio, n=n, power=.81,
                        alpha=alpha, alternative=alternative)
    assert plan(mde).power == pytest.approx(.81, abs=2e-14)
    assert plan(mde).effect * effect > 0


def test_published_mean_example_and_probability_integral():
    # Official power onemean example 3: null 15, alternative 40, known SD 40.
    assert plan(oe.power_mean(25, sd=40, power=.8)).n == 21
    output = plan(oe.power_mean(.8, sd=2., n=29))
    # Independently integrate density of sample mean outside null critical interval.
    se = 2 / math.sqrt(29)
    cutoff = stats.norm.isf(.025) * se
    def pdf(x):
        return stats.norm.pdf(x, loc=.8, scale=se)
    probability = integrate.quad(pdf, -np.inf, -cutoff)[0] + integrate.quad(pdf, cutoff, np.inf)[0]
    assert output.power == pytest.approx(probability, abs=3e-13)


@pytest.mark.parametrize("alternative", ["upper", "lower", "two-sided"])
@pytest.mark.parametrize("rho0,rho", [(.2, .55), (-.4, .1), (.8, .9)])
def test_correlation_fisher_law_and_inversions(alternative, rho0, rho):
    if alternative == "lower":
        rho0, rho = rho, rho0
    n = 73
    result = oe.power_correlation(rho0, rho, n=n, alternative=alternative)
    oracle = normal_power((np.arctanh(rho) - np.arctanh(rho0)) * np.sqrt(n - 3),
                          .05, alternative)
    assert plan(result).power == pytest.approx(oracle, abs=3e-14)
    sized = oe.power_correlation(rho0, rho, power=.9, alternative=alternative)
    required = int(plan(sized).n)
    assert plan(sized).power >= .9
    if required > 4:
        assert plan(oe.power_correlation(rho0, rho, n=required - 1,
                                        alternative=alternative)).power < .9
    detectable = oe.power_correlation(rho0, n=n, power=.9, alternative=alternative)
    assert plan(detectable).power == pytest.approx(.9, abs=4e-14)
    assert (plan(detectable).rho - rho0) * (rho - rho0) > 0
    assert "approximation" in detectable.attrs["inference"]


def test_correlation_two_sided_lower_branch_and_symmetry():
    lower = plan(oe.power_correlation(.3, n=100, power=.8, direction="lower"))
    upper = plan(oe.power_correlation(-.3, n=100, power=.8))
    assert lower.rho == pytest.approx(-upper.rho)
    assert plan(oe.power_correlation(0., 0., n=4)).power == pytest.approx(.05)


def test_official_correlation_examples():
    assert plan(oe.power_correlation(0., .5, power=.8, alternative="upper")).n == 24
    assert plan(oe.power_correlation(0., .7, power=.8, alternative="upper")).n == 12
    assert plan(oe.power_correlation(0., .5, n=15, alternative="upper")).power \
        == pytest.approx(.6018, abs=.00005)
    assert plan(oe.power_correlation(0., n=15, power=.8, alternative="upper")).rho \
        == pytest.approx(.6155, abs=.00005)


@pytest.mark.parametrize("alternative", ["upper", "lower", "two-sided"])
@pytest.mark.parametrize("p0,p1,n", [(.5, .7, 23), (.12, .3, 77), (.83, .95, 51),
                                      (.02, .005, 800), (.5, .5, 30)])
def test_binomial_full_rejection_set_oracle(alternative, p0, p1, n):
    result = oe.power_proportion(p0, p1, n=n, alternative=alternative)
    row = plan(result)
    rejection = exact_region(n, p0, .05, alternative)
    assert row.power == pytest.approx(stats.binom.pmf(rejection, n, p1).sum(), abs=2e-12)
    assert row.actual_alpha == pytest.approx(stats.binom.pmf(rejection, n, p0).sum(), abs=2e-12)
    assert row.actual_alpha <= .05 + 1e-13
    expected = np.arange(n + 1)
    expected = expected[(expected <= row.lower_critical) | (expected >= row.upper_critical)]
    np.testing.assert_array_equal(rejection, expected)


@pytest.mark.parametrize("alternative,p0,p1", [("upper", .4, .7), ("lower", .7, .4),
                                               ("two-sided", .3, .6), ("two-sided", .7, .4)])
def test_binomial_exhaustive_minimum_and_detectable_alternative(alternative, p0, p1):
    result = oe.power_proportion(p0, p1, power=.85, alternative=alternative)
    n = int(plan(result).n)
    probabilities = [stats.binom.pmf(exact_region(k, p0, .05, alternative), k, p1).sum()
                     for k in range(1, n + 1)]
    assert max(probabilities[:-1], default=0) < .85
    assert probabilities[-1] >= .85
    assert result.attrs["n_search_evaluations"] == n
    direction = "upper" if p1 > p0 else "lower"
    mde = plan(oe.power_proportion(p0, n=n, power=.85, alternative=alternative,
                                  direction=direction))
    assert mde.power == pytest.approx(.85, abs=3e-13)
    assert (mde.p1 - p0) * (p1 - p0) > 0


def test_binomial_nonmonotone_power_and_empty_rejection_region():
    powers = [plan(oe.power_proportion(.5, .7, n=n)).power for n in range(15, 31)]
    assert any(right < left for left, right in zip(powers, powers[1:]))
    row = plan(oe.power_proportion(.5, .7, n=1))
    assert row.power == row.actual_alpha == 0.
    assert row.lower_critical == -1 and row.upper_critical == 2
    with pytest.raises(AnalysisError, match="unattainable"):
        oe.power_proportion(.5, n=1, power=.8)


@pytest.mark.parametrize("alpha", [1e-8, .0001, .2])
@pytest.mark.parametrize("p0,p1", [(1e-8, .01), (.99999999, .99), (.4, .400001)])
def test_extreme_binomial_tails(alpha, p0, p1):
    row = plan(oe.power_proportion(p0, p1, n=1000, alpha=alpha))
    rejection = exact_region(1000, p0, alpha, "two-sided")
    assert row.power == pytest.approx(stats.binom.pmf(rejection, 1000, p1).sum(), abs=3e-12)
    assert row.actual_alpha == pytest.approx(stats.binom.pmf(rejection, 1000, p0).sum(), abs=3e-12)


def test_cpu_is_explicit_even_with_a_different_default_device():
    with torch.device("meta"):
        output = oe.power_proportion(.5, .7, n=37)
    assert plan(output).power > 0.
    assert output.attrs["device"] == "cpu"


def test_numpy_integer_budget_is_saved_as_json_integer():
    for output in [oe.power_mean(.5, sd=1., n=30, max_n=np.int64(100)),
                   oe.power_proportion(.5, .7, n=30, max_n=np.int64(100)),
                   oe.power_correlation(0., .3, n=30, max_n=np.int64(100)),
                   oe.precision_mean(sd=1., n=30, max_n=np.int64(100))]:
        for key, value in output["settings"].itertuples(index=False, name=None):
            json.loads(value)


def test_detectable_correlation_refuses_loss_of_accuracy_near_boundary():
    with pytest.raises(AnalysisError):
        oe.power_correlation(.999999999, n=10_000_000, power=.8)


@pytest.mark.parametrize("sd2,ratio", [(None, 1.), (3., .27), (.7, 1.), (8., 3.1)])
@pytest.mark.parametrize("confidence", [.8, .95, .999])
def test_precision_law_minimum_and_roundtrip(sd2, ratio, confidence):
    output = oe.precision_mean(sd=2., sd2=sd2, ratio=ratio, width=.6, confidence=confidence)
    row = plan(output)
    n = int(row.n)
    oracle = 2 * stats.norm.isf((1 - confidence) / 2) * np.sqrt(
        4 / n + (sd2**2 / math.ceil(ratio * n) if sd2 else 0))
    assert row.width == pytest.approx(oracle, rel=2e-14)
    assert row.width <= .6
    if n > 1:
        assert plan(oe.precision_mean(sd=2., sd2=sd2, ratio=ratio, n=n - 1,
                                     confidence=confidence)).width > .6
    assert plan(oe.precision_mean(sd=2., sd2=sd2, ratio=ratio, n=n,
                                 confidence=confidence)).width == row.width


def test_precision_official_example_scaling_and_boundary():
    assert plan(oe.precision_mean(sd=.8, width=.5)).n == 40
    assert plan(oe.precision_mean(sd=5.5, sd2=5., width=6.)).n == 24
    small = plan(oe.precision_mean(sd=1e-90, width=5e-91))
    large = plan(oe.precision_mean(sd=1e90, width=5e89))
    assert small.n == large.n
    exact = plan(oe.precision_mean(sd=1., n=17)).width
    assert plan(oe.precision_mean(sd=1., width=exact)).n == 17
    assert plan(oe.precision_mean(sd=1., width=math.nextafter(exact, 0.))).n == 18


@pytest.mark.parametrize("call", [
    lambda: oe.power_mean(sd=1.), lambda: oe.power_mean(1., sd=True, n=30),
    lambda: oe.power_mean(1., sd=1., n=30, power=.8),
    lambda: oe.power_mean(1., sd=1., n=30.0),
    lambda: oe.power_mean(float("nan"), sd=1., n=30),
    lambda: oe.power_mean(1., sd=0., n=30),
    lambda: oe.power_mean(1., sd=1., n=30, ratio=2.),
    lambda: oe.power_mean(1., sd=1., n=30, alpha=1e-10),
    lambda: oe.power_mean(1., sd=1., n=30, alternative="paired"),
    lambda: oe.power_mean(0., sd=1., power=.8),
    lambda: oe.power_mean(-1., sd=1., power=.8, alternative="upper"),
    lambda: oe.power_mean(1., sd=1., power=.8, max_n=1),
    lambda: oe.power_mean(1., sd=1., sd2=1., ratio=3., n=5, max_n=10),
    lambda: oe.power_mean(1., sd=1., n=True),
    lambda: oe.power_mean(sd=1e100, n=1, power=.8),
    lambda: oe.power_correlation(1., .8, n=30),
    lambda: oe.power_correlation(0., .3, n=3),
    lambda: oe.power_correlation(0., .3, n=30, direction="bad"),
    lambda: oe.power_correlation(0., 0., power=.8),
    lambda: oe.power_correlation(.9, .5, power=.8, alternative="upper"),
    lambda: oe.power_correlation(0., .1, n=11, max_n=10),
    lambda: oe.power_proportion(0., .3, n=30),
    lambda: oe.power_proportion(.5, 1., n=30),
    lambda: oe.power_proportion(.5, .3, n=1001),
    lambda: oe.power_proportion(.5, .5, power=.8),
    lambda: oe.power_proportion(.5, .7, power=.8, max_n=1),
    lambda: oe.precision_mean(sd=1.), lambda: oe.precision_mean(sd=1., n=10, width=1.),
    lambda: oe.precision_mean(sd=1., width=0.),
    lambda: oe.precision_mean(sd=1., n=10, confidence=1.),
    lambda: oe.precision_mean(sd=1., width=.0001, max_n=100),
    lambda: oe.precision_mean(sd=1., n=10, max_n=10_000_001),
])
def test_invalid_and_budget_contract(call):
    with pytest.raises(AnalysisError):
        call()


def test_all_metadata_is_in_ordinary_saved_cells_and_rng_untouched():
    before = torch.random.get_rng_state().clone()
    results = [oe.power_mean(.5, sd=1., power=.8),
               oe.power_proportion(.5, .7, power=.8),
               oe.power_correlation(0., .3, power=.8), oe.precision_mean(sd=1., width=.5)]
    for result in results:
        assert list(result) == ["plan", "scenarios", "settings"]
        settings = {name: json.loads(value) for name, value in result["settings"].itertuples(index=False, name=None)}
        assert settings == result.attrs
        assert settings["observations_used"] is False
        assert settings["prospective"] is True
        assert "tabular" in result.to_latex(index=False)
        for frame in result.values():
            assert len(frame) <= 50 and len(frame.columns) <= 30
            assert all(len(str(cell)) < 500 for row in frame.itertuples(index=False, name=None) for cell in row)
    assert torch.equal(before, torch.random.get_rng_state())
