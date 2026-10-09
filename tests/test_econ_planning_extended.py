"""Independent laws, integer minima, all solve modes and scientific refusal.

SciPy/NumPy are validation-only; production uses native distribution kernels.
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


def row(result):
    return result["plan"].iloc[0]


def power_law(shift, alternative, alpha=0.05):
    z = stats.norm.isf(alpha / (2 if alternative == "two-sided" else 1))
    if alternative == "upper":
        return stats.norm.sf(z, loc=shift)
    if alternative == "lower":
        return stats.norm.cdf(-z, loc=shift)
    return stats.norm.sf(z, loc=shift) + stats.norm.cdf(-z, loc=shift)


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("correlation", [-0.8, 0.0, 0.7, 1 - 1e-12])
def test_paired_exact_law_all_modes(alternative, correlation):
    effect = -0.4 if alternative == "lower" else 0.4
    assumptions = dict(
        sd_before=1.2, sd_after=0.8, correlation=correlation, alternative=alternative
    )
    observed = row(oe.power_paired_mean(effect, n=43, **assumptions))
    variance = 1.2**2 + 0.8**2 - 2 * correlation * 1.2 * 0.8
    se = math.sqrt(variance / 43)
    assert observed.standard_error == pytest.approx(se, abs=2e-15)
    assert observed.power == pytest.approx(power_law(effect / se, alternative), abs=3e-14)
    assert observed.pairs == 43 and observed.measurements == 86
    sized = row(oe.power_paired_mean(effect, power=0.8, **assumptions))
    assert sized.power >= 0.8
    if sized.n > 1:
        assert row(oe.power_paired_mean(effect, n=int(sized.n) - 1, **assumptions)).power < 0.8
    mde = row(oe.power_paired_mean(n=43, power=0.8, **assumptions))
    assert row(oe.power_paired_mean(mde.effect, n=43, **assumptions)).power == pytest.approx(
        0.8, abs=1e-13
    )


def test_paired_covariance_cancellation_and_density_integral():
    result = row(oe.power_paired_mean(0.02, sd_before=1, sd_after=1, correlation=1 - 1e-12, n=2))
    assert result.difference_sd == pytest.approx(math.sqrt(2e-12), rel=3e-5)
    se = math.sqrt((4 + 9 - 2 * 0.4 * 2 * 3) / 27)
    cutoff = stats.norm.isf(0.025) * se
    probability = integrate.quad(lambda x: stats.norm.pdf(x, loc=0.6, scale=se), -np.inf, -cutoff)[
        0
    ]
    probability += integrate.quad(lambda x: stats.norm.pdf(x, loc=0.6, scale=se), cutoff, np.inf)[0]
    assert row(
        oe.power_paired_mean(0.6, sd_before=2, sd_after=3, correlation=0.4, n=27)
    ).power == pytest.approx(probability, abs=2e-13)


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("ratio", [0.37, 1, 2.3])
@pytest.mark.parametrize("method", ["power_two_proportions", "power_two_correlations"])
def test_two_group_transform_law_minimum_and_detectable(method, ratio, alternative):
    lower = alternative == "lower"
    first, second = (0.5, 0.25) if lower else (0.25, 0.5)
    fn = getattr(oe, method)
    settings = dict(ratio=ratio, alternative=alternative)
    observed = row(fn(first, second, n=200, **settings))
    n2 = math.ceil(ratio * 200)
    if method == "power_two_proportions":
        shift = (
            2 * (np.arcsin(np.sqrt(second)) - np.arcsin(np.sqrt(first))) / np.sqrt(1 / 200 + 1 / n2)
        )
        effect_key = "p2"
    else:
        shift = (np.arctanh(second) - np.arctanh(first)) / np.sqrt(1 / 197 + 1 / (n2 - 3))
        effect_key = "rho2"
    assert observed.n2 == n2 and observed.total_n == 200 + n2
    assert observed.power == pytest.approx(power_law(shift, alternative), abs=5e-14)
    sized = row(fn(first, second, power=0.85, **settings))
    assert sized.power >= 0.85
    previous = int(sized.n) - 1
    if method == "power_two_proportions":
        admissible = (
            min(first, 1 - first) * previous >= 10
            and min(second, 1 - second) * math.ceil(ratio * previous) >= 10
        )
    else:
        admissible = previous >= 4 and math.ceil(ratio * previous) >= 4
    if admissible:
        assert row(fn(first, second, n=previous, **settings)).power < 0.85
    mde = row(fn(first, n=200, power=0.85, **settings))
    assert row(fn(first, mde[effect_key], n=200, **settings)).power == pytest.approx(
        0.85, abs=1e-12
    )
    assert (mde[effect_key] - first) * (-1 if lower else 1) > 0


def test_two_proportion_declared_domain_and_null_guard():
    assert row(oe.power_two_proportions(0.5, 0.5, n=20)).power == pytest.approx(0.05)
    for kwargs in [
        dict(p1=0.001, p2=0.4, n=10000),
        dict(p1=0.1, p2=0.2, n=99),
        dict(p1=0.4, n=200, power=0.9, ratio=0.01),
        dict(p1=0.5, p2=0.6, power=0.8, max_n=19),
    ]:
        with pytest.raises(AnalysisError):
            oe.power_two_proportions(**kwargs)
    with pytest.raises(AnalysisError):
        oe.power_two_proportions(0.99, n=10000, power=0.99)
    settings = oe.power_two_proportions(0.3, 0.5, power=0.8).attrs
    assert "arcsine" in settings["inference"] and "approximation" in settings["inference"]


def test_two_correlation_no_shared_sample_and_extreme_inversion_refusal():
    with pytest.raises(AnalysisError):
        oe.power_two_correlations(0.2, 0.5, n=4, ratio=0.5)
    with pytest.raises(TypeError):
        oe.power_two_correlations(0.2, 0.5, n=100, shared=True)
    with pytest.raises(AnalysisError):
        oe.power_two_correlations(1 - 1e-12, n=100, power=0.8)


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("design_variance", [0.1, 1.0, 7.0])
def test_fixed_slope_exact_law_and_inversion(alternative, design_variance):
    effect = -0.4 if alternative == "lower" else 0.4
    kwargs = dict(error_sd=1.3, design_variance=design_variance, alternative=alternative)
    observed = row(oe.power_slope(effect, n=37, **kwargs))
    se = 1.3 / math.sqrt(37 * design_variance)
    assert observed.standard_error == pytest.approx(se)
    assert observed.centered_gram == pytest.approx(37 * design_variance)
    assert observed.power == pytest.approx(power_law(effect / se, alternative), abs=4e-14)
    sized = row(oe.power_slope(effect, power=0.8, **kwargs))
    assert sized.n >= 2 and sized.power >= 0.8
    if sized.n > 2:
        assert row(oe.power_slope(effect, n=int(sized.n) - 1, **kwargs)).power < 0.8
    mde = row(oe.power_slope(n=37, power=0.8, **kwargs))
    assert row(oe.power_slope(mde.effect, n=37, **kwargs)).power == pytest.approx(0.8, abs=1e-13)


def test_slope_independent_ols_sampling():
    # Actually generate the fixed-X normal regression, not a shifted-Z simulation.
    rng = np.random.default_rng(9237)
    x = np.linspace(-1, 1, 31)
    x -= x.mean()
    variance = np.dot(x, x) / len(x)
    y = 0.3 * x + rng.normal(size=(40000, len(x)))
    slopes = y @ x / np.dot(x, x)
    se = 1 / math.sqrt(np.dot(x, x))
    empirical = np.mean(np.abs(slopes / se) > stats.norm.isf(0.025))
    predicted = row(oe.power_slope(0.3, error_sd=1, design_variance=variance, n=len(x))).power
    assert abs(empirical - predicted) < 0.009
    assert slopes.std() == pytest.approx(se, rel=0.015)


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.8])
def test_schoenfeld_information_events_and_hr_inversion(alternative, q):
    hr = 0.7 if alternative == "lower" else 1.4
    kwargs = dict(information_fraction=q, event_fraction=0.6, alternative=alternative)
    result = row(oe.power_logrank(hr, events=250, **kwargs))
    shift = math.log(hr) * math.sqrt(250 * q * (1 - q))
    assert result.power == pytest.approx(power_law(shift, alternative), abs=3e-14)
    assert result.expected_enrollment == math.ceil(250 / 0.6)
    sized = row(oe.power_logrank(hr, power=0.8, **kwargs))
    assert sized.power >= 0.8
    minimum = math.ceil(10 / min(q, 1 - q))
    if sized.events > minimum:
        assert row(oe.power_logrank(hr, events=int(sized.events) - 1, **kwargs)).power < 0.8
    mde = row(oe.power_logrank(events=250, power=0.8, **kwargs))
    assert row(oe.power_logrank(mde.hazard_ratio, events=250, **kwargs)).power == pytest.approx(
        0.8, abs=1e-12
    )


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("probability", [0.2, 0.35, 0.65, 0.8])
def test_conditional_exact_mcnemar_full_rejection_enumeration(alternative, probability):
    m = 43
    k = np.arange(m + 1)
    tail = 0.05 / (2 if alternative == "two-sided" else 1)
    rejection = np.zeros(m + 1, dtype=bool)
    if alternative != "upper":
        rejection |= stats.binom.cdf(k, m, 0.5) <= tail
    if alternative != "lower":
        rejection |= stats.binom.sf(k - 1, m, 0.5) <= tail
    result = row(
        oe.power_mcnemar(
            probability, discordant_pairs=m, discordance_fraction=0.3, alternative=alternative
        )
    )
    assert result.power == pytest.approx(
        stats.binom.pmf(k[rejection], m, probability).sum(), abs=3e-14
    )
    assert result.actual_alpha == pytest.approx(
        stats.binom.pmf(k[rejection], m, 0.5).sum(), abs=3e-14
    )
    assert result.expected_total_pairs == math.ceil(m / 0.3)


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
def test_mcnemar_first_discordant_count_and_mde(alternative):
    probability = 0.3 if alternative == "lower" else 0.7
    sized = row(
        oe.power_mcnemar(probability, power=0.8, alternative=alternative, max_discordant_pairs=200)
    )
    for m in range(1, int(sized.discordant_pairs)):
        k = np.arange(m + 1)
        tail = 0.05 / (2 if alternative == "two-sided" else 1)
        rejection = (
            (stats.binom.cdf(k, m, 0.5) <= tail)
            if alternative != "upper"
            else np.zeros(m + 1, dtype=bool)
        )
        if alternative != "lower":
            rejection |= stats.binom.sf(k - 1, m, 0.5) <= tail
        assert stats.binom.pmf(k[rejection], m, probability).sum() < 0.8
    mde = row(oe.power_mcnemar(discordant_pairs=100, power=0.8, alternative=alternative))
    assert row(
        oe.power_mcnemar(mde.probability, discordant_pairs=100, alternative=alternative)
    ).power == pytest.approx(0.8, abs=3e-13)


@pytest.mark.parametrize("method", ["precision_mean_unknown", "precision_variance"])
@pytest.mark.parametrize("confidence", [0.8, 0.95, 0.999])
@pytest.mark.parametrize("assurance", [0.1, 0.5, 0.9, 0.999])
def test_random_width_quantile_probability_and_exhaustive_first_n(method, confidence, assurance):
    n = 29
    parameter = 1.7
    kwargs = {
        "sd" if method == "precision_mean_unknown" else "variance": parameter,
        "confidence": confidence,
        "assurance": assurance,
    }
    fn = getattr(oe, method)

    def oracle(k):
        df = k - 1
        if method == "precision_mean_unknown":
            return (
                2
                * stats.t.isf((1 - confidence) / 2, df)
                * parameter
                * np.sqrt(stats.chi2.ppf(assurance, df) / (k * df))
            )
        return (
            parameter
            * stats.chi2.ppf(assurance, df)
            * (
                1 / stats.chi2.ppf((1 - confidence) / 2, df)
                - 1 / stats.chi2.isf((1 - confidence) / 2, df)
            )
        )

    result = row(fn(n=n, **kwargs))
    assert result.width == pytest.approx(oracle(n), rel=3e-10)
    assert result.width_probability == pytest.approx(assurance, abs=3e-11)
    target = oracle(n) * 1.04
    sized = row(fn(width=target, max_n=80, **kwargs))
    qualifying = next(k for k in range(2, 81) if oracle(k) <= target)
    assert (
        sized.n == qualifying
        and sized.width <= target
        and sized.width_probability >= assurance - 2e-11
    )


def test_random_intervals_actual_coverage_and_width_assurance():
    rng = np.random.default_rng(49921)
    n = 31
    df = n - 1
    sigma = 1.3
    data = rng.normal(0, sigma, size=(50000, n))
    mean = data.mean(axis=1)
    s2 = data.var(axis=1, ddof=1)
    tcritical = stats.t.isf(0.025, df)
    half = tcritical * np.sqrt(s2 / n)
    mean_width = row(oe.precision_mean_unknown(sd=sigma, n=n, assurance=0.9)).width
    lo = df * s2 / stats.chi2.isf(0.025, df)
    hi = df * s2 / stats.chi2.ppf(0.025, df)
    variance_width = row(oe.precision_variance(variance=sigma**2, n=n, assurance=0.9)).width
    assert abs(np.mean(np.abs(mean) <= half) - 0.95) < 0.006
    assert abs(np.mean((lo <= sigma**2) & (hi >= sigma**2)) - 0.95) < 0.006
    assert abs(np.mean(2 * half <= mean_width) - 0.9) < 0.006
    assert abs(np.mean(hi - lo <= variance_width) - 0.9) < 0.006


FACTORIES = {
    "power_paired_mean": dict(effect=0.5, sd_before=1, sd_after=1, correlation=0.5, n=50),
    "power_two_proportions": dict(p1=0.3, p2=0.5, n=100),
    "power_two_correlations": dict(rho1=0.2, rho2=0.5, n=100),
    "power_slope": dict(effect=0.5, error_sd=1, design_variance=1, n=50),
    "power_logrank": dict(hazard_ratio=0.7, events=200),
    "power_mcnemar": dict(probability=0.7, discordant_pairs=50),
    "precision_mean_unknown": dict(sd=1, n=50),
    "precision_variance": dict(variance=1, n=50),
}


@pytest.mark.parametrize("method", list(FACTORIES))
@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "50", 0])
def test_invalid_scalar_count_inputs(method, bad):
    kwargs = FACTORIES[method].copy()
    key = (
        "events"
        if method == "power_logrank"
        else "discordant_pairs"
        if method == "power_mcnemar"
        else "n"
    )
    kwargs[key] = bad
    with pytest.raises(AnalysisError):
        getattr(oe, method)(**kwargs)


@pytest.mark.parametrize("method", list(FACTORIES))
def test_complete_json_contract_and_rng_unchanged(method):
    before = torch.random.get_rng_state().clone()
    result = getattr(oe, method)(**FACTORIES[method])
    assert torch.equal(before, torch.random.get_rng_state())
    settings = {
        key: json.loads(value)
        for key, value in result["settings"].itertuples(index=False, name=None)
    }
    assert settings == result.attrs
    assert (
        settings["observations_used"] is False
        and settings["device"] == "cpu"
        and settings["precision"] == "float64"
    )
    assert settings["stochastic_draws"] == 0
    assert all(len(frame) <= 50 and len(frame.columns) <= 30 for frame in result.values())
    assert max(len(value) for value in result["settings"]["json"]) <= 500


@pytest.mark.parametrize("method", ["precision_mean_unknown", "precision_variance"])
def test_precision_budgets_extremes_and_scale(method):
    fn = getattr(oe, method)
    key = "sd" if method == "precision_mean_unknown" else "variance"
    for bad in [0, -1, True, float("nan"), 1e101]:
        with pytest.raises(AnalysisError):
            fn(**{key: bad}, n=20)
    for kwargs in [
        dict(n=1),
        dict(n=20, max_n=19),
        dict(width=1e-100, max_n=10),
        dict(n=20, max_n=10001),
        dict(n=20, assurance=1),
        dict(n=20, confidence=0.5),
        dict(n=20, width=1),
    ]:
        with pytest.raises(AnalysisError):
            fn(**{key: 1}, **kwargs)
    small = row(fn(**{key: 1e-100}, n=20))
    large = row(fn(**{key: 1e100}, n=20))
    assert large.width / small.width == pytest.approx(1e200, rel=3e-14)
    assert small.width_probability == pytest.approx(0.9, abs=1e-11)
    assert large.width_probability == pytest.approx(0.9, abs=1e-11)


def test_unsupported_designs_unreachable_branches_and_limits():
    calls = [
        lambda: oe.power_slope(0.5, error_sd=1, design_variance=0, n=20),
        lambda: oe.power_slope(error_sd=1e100, design_variance=1e-100, n=20, power=0.8),
        lambda: oe.power_paired_mean(0.5, sd_before=1, sd_after=1, correlation=1, n=20),
        lambda: oe.power_two_correlations(0.2, 0.2, power=0.8),
        lambda: oe.power_two_proportions(0.5, 0.4, power=0.8, alternative="upper"),
        lambda: oe.power_logrank(0.7, power=0.8, alternative="upper"),
        lambda: oe.power_logrank(0.7, events=200, event_fraction=1e-20),
        lambda: oe.power_mcnemar(0.7, discordant_pairs=20, discordance_fraction=1e-20),
        lambda: oe.power_mcnemar(0.7, power=0.8, max_discordant_pairs=1001),
        lambda: oe.power_mcnemar(discordant_pairs=1, power=0.8),
        lambda: oe.power_logrank(0.7, events=1),
        lambda: oe.power_paired_mean(
            0.5, sd_before=1, sd_after=1, correlation=0.5, n=20, power=0.8
        ),
    ]
    for call in calls:
        with pytest.raises(AnalysisError):
            call()


def test_unsupported_options_are_not_silently_accepted():
    with pytest.raises(TypeError):
        oe.power_slope(0.5, error_sd=1, design_variance=1, n=20, weights=[1])


@pytest.mark.parametrize("method", ["precision_mean_unknown", "precision_variance"])
@pytest.mark.parametrize("n", [2, 10000])
@pytest.mark.parametrize("assurance", [1e-6, 1 - 1e-6])
def test_random_width_tail_quantiles_at_hard_domain_edges(method, n, assurance):
    df = n - 1
    confidence = 1 - 1e-8
    if method == "precision_mean_unknown":
        result = row(
            oe.precision_mean_unknown(sd=1, n=n, confidence=confidence, assurance=assurance)
        )
        oracle = (
            2
            * stats.t.isf((1 - confidence) / 2, df)
            * math.sqrt(stats.chi2.ppf(assurance, df) / (n * df))
        )
    else:
        result = row(
            oe.precision_variance(variance=1, n=n, confidence=confidence, assurance=assurance)
        )
        oracle = stats.chi2.ppf(assurance, df) * (
            1 / stats.chi2.ppf((1 - confidence) / 2, df)
            - 1 / stats.chi2.isf((1 - confidence) / 2, df)
        )
    assert result.width == pytest.approx(oracle, rel=1e-10)
    assert result.width_probability == pytest.approx(assurance, abs=1e-11)


def test_expected_count_decimal_edges_and_projection_limits():
    assert row(oe.power_two_proportions(0.9, 0.9, n=100)).n == 100
    assert row(oe.power_logrank(0.7, events=50, information_fraction=0.8)).events == 50
    assert row(oe.power_logrank(0.7, events=200, event_fraction=1)).expected_enrollment == 200
    assert (
        row(oe.power_mcnemar(0.7, discordant_pairs=50, discordance_fraction=1)).expected_total_pairs
        == 50
    )
    logrank = oe.power_logrank(0.7, events=200, event_fraction=2e-7)
    mcnemar = oe.power_mcnemar(0.7, discordant_pairs=50, discordance_fraction=5e-8)
    assert row(logrank).expected_enrollment == 1_000_000_000
    assert row(mcnemar).expected_total_pairs == 1_000_000_000
    assert logrank["scenarios"].events.max() == 200
    assert mcnemar["scenarios"].discordant_pairs.max() == 50


@pytest.mark.parametrize("n,k", [(5, 0), (9, 1), (17, 2), (30, 2)])
@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
def test_conditional_mcnemar_exact_dyadic_rejection_boundary(n, k, alternative):
    # Independent integer enumeration: these fractions are exactly representable.
    cumulative = sum(math.comb(n, j) for j in range(k + 1))
    alpha = (2 if alternative == "two-sided" else 1) * cumulative / 2**n
    result = row(oe.power_mcnemar(0.7, discordant_pairs=n, alpha=alpha, alternative=alternative))
    assert result.lower_critical == (-1 if alternative == "upper" else k)
    assert result.upper_critical == (n + 1 if alternative == "lower" else n - k)
    assert result.actual_alpha == alpha
    lower_alpha = math.nextafter(alpha, 0.0)
    lower = row(
        oe.power_mcnemar(0.7, discordant_pairs=n, alpha=lower_alpha, alternative=alternative)
    )
    assert lower.lower_critical == (-1 if alternative == "upper" else k - 1)
    assert lower.upper_critical == (n + 1 if alternative == "lower" else n - k + 1)
    # Shared prior exact-binomial API receives the same correction.
    prior = row(oe.power_proportion(0.5, 0.7, n=n, alpha=alpha, alternative=alternative))
    assert prior.actual_alpha == alpha and prior.power == result.power
