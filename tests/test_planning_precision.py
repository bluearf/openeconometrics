"""Independent random-width laws, first-integer audits and portable state."""

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats import planning_precision as methods
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def plan(result):
    return result["plan"].iloc[0]


def cp_widths(n, confidence):
    k = np.arange(n + 1)
    alpha = 1 - confidence
    lo = np.zeros(n + 1)
    hi = np.ones(n + 1)
    lo[1:] = stats.beta.ppf(alpha / 2, k[1:], n - k[1:] + 1)
    hi[:-1] = stats.beta.isf(alpha / 2, k[:-1] + 1, n - k[:-1])
    return lo, hi, hi - lo


def garwood(k, exposure, confidence):
    alpha = 1 - confidence
    low = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / (2 * exposure)
    high = stats.chi2.isf(alpha / 2, 2 * (k + 1)) / (2 * exposure)
    return low, high, high - low


@pytest.mark.parametrize("ratio", [0.13, 1.0, 2.3])
@pytest.mark.parametrize("assurance", [0.1, 0.5, 0.9])
def test_pooled_two_mean_continuous_width_law(ratio, assurance):
    result = methods.precision_twomeans_unknown(sd=1.3, n=31, ratio=ratio, assurance=assurance)
    row = plan(result)
    n2 = math.ceil(31 * ratio)
    df = 31 + n2 - 2
    scale = 2 * stats.t.isf(0.025, df) * 1.3 * math.sqrt(1 / 31 + 1 / n2)
    expected = scale * math.sqrt(stats.chi2.ppf(assurance, df) / df)
    assert row.n2 == n2 and row.df == df and row.total_n == 31 + n2
    assert row.width == pytest.approx(expected, rel=2e-12)
    assert row.width_probability == pytest.approx(assurance, abs=2e-12)


def test_pooled_first_n_is_exhaustive_and_group_budget_real():
    result = methods.precision_twomeans_unknown(sd=1.2, width=1.5, ratio=0.6, max_n=100)
    row = plan(result)
    valid = []
    for n1 in range(2, 101):
        n2 = math.ceil(0.6 * n1)
        if n2 < 2:
            continue
        df = n1 + n2 - 2
        w = (
            2
            * stats.t.isf(0.025, df)
            * 1.2
            * math.sqrt((1 / n1 + 1 / n2) * stats.chi2.ppf(0.9, df) / df)
        )
        if w <= 1.5:
            valid.append(n1)
    assert row.n == valid[0]
    assert result.attrs["n_search_evaluations"] == row.n - 2 + 1
    with pytest.raises(AnalysisError) as error:
        methods.precision_twomeans_unknown(sd=1, n=50, ratio=3, max_n=100)
    assert error.value.code == "resource_limit"


def test_decimal_allocation_fixed_n_and_per_arm_boundary():
    out = methods.precision_twomeans_unknown(sd=1, n=50, ratio=0.14)
    assert plan(out).n2 == 7
    assert out.attrs["allocation_ratio_numerator"] == 7
    assert out.attrs["allocation_ratio_denominator"] == 50
    out = methods.precision_twomeans_unknown(sd=1, n=50, ratio=1.1, max_n=55)
    assert plan(out).n2 == 55
    with pytest.raises(AnalysisError) as error:
        methods.precision_twomeans_unknown(sd=1, n=51, ratio=1.1, max_n=55)
    assert error.value.code == "resource_limit"


def test_decimal_allocation_first_n_oracle_at_exact_boundary():
    ratio = Fraction("0.14")
    alpha, assurance, sd = 0.05, 0.9, 1.0
    widths = []
    for n1 in range(2, 101):
        n2 = (n1 * ratio.numerator + ratio.denominator - 1) // ratio.denominator
        if n2 < 2:
            continue
        df = n1 + n2 - 2
        width = (
            2
            * stats.t.isf(alpha / 2, df)
            * sd
            * math.sqrt((1 / n1 + 1 / n2) * stats.chi2.ppf(assurance, df) / df)
        )
        widths.append((n1, n2, width))
    # A target between successive designs makes the decimal allocation at
    # n1=50 decisive without an oracle/kernel equality-at-last-bit comparison.
    target = (
        next(w for n1, _, w in widths if n1 == 49) + next(w for n1, _, w in widths if n1 == 50)
    ) / 2
    expected = next((n1, n2) for n1, n2, width in widths if width <= target)
    out = methods.precision_twomeans_unknown(sd=sd, ratio=0.14, width=target, max_n=100)
    assert (plan(out).n, plan(out).n2) == expected == (50, 7)


@pytest.mark.parametrize("n", [1, 2, 7, 20])
@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 1.0])
def test_complete_binomial_width_distribution_and_inclusive_quantile(n, p):
    result = methods.precision_binomial(p=p, n=n)
    row = plan(result)
    law = result["width_distribution"]
    low, high, widths = cp_widths(n, 0.95)
    probabilities = stats.binom.pmf(np.arange(n + 1), n, p)
    assert list(law["count"]) == list(range(n + 1))
    assert law["ci_low"].to_numpy() == pytest.approx(low, abs=2e-12)
    assert law["ci_high"].to_numpy() == pytest.approx(high, abs=2e-12)
    assert law["width"].to_numpy() == pytest.approx(widths, abs=2e-12)
    assert law["probability"].to_numpy() == pytest.approx(probabilities, abs=3e-14)
    canonical = law["width"].to_numpy()
    assert np.array_equal(canonical, canonical[::-1])
    expected_mass = probabilities[widths <= row.width + 3e-12].sum()
    assert row.width_probability == pytest.approx(expected_mass, abs=2e-13)
    assert row.probability_lower <= expected_mass + 3e-14
    assert row.probability_upper >= expected_mass - 3e-14
    assert row.probability_lower >= 0.9
    lower_width = canonical[canonical < row.width]
    if len(lower_width):
        assert probabilities[canonical <= lower_width.max()].sum() < 0.9
    assert row.omitted_mass == 0 and row.enumerated_mass == pytest.approx(1)


def test_binomial_first_n_reference_not_monotone_shortcut():
    result = methods.precision_binomial(p=0.35, width=0.6, assurance=0.8, max_n=30)
    row = plan(result)
    reference = []
    for n in range(1, 31):
        widths = cp_widths(n, 0.95)[2]
        mass = stats.binom.pmf(np.arange(n + 1), n, 0.35)[widths <= 0.6].sum()
        if mass >= 0.8:
            reference.append(n)
    assert row.n == reference[0]
    assert result.attrs["n_search_evaluations"] == row.n
    assert row.target_width == 0.6 and row.width_probability >= 0.8


def test_exact_dyadic_cdf_equality_and_zero_rate_boundary():
    result = methods.precision_binomial(p=0.5, n=2, assurance=0.5)
    law = result["width_distribution"]
    row = plan(result)
    assert row.width == law.iloc[0].width == law.iloc[2].width
    assert row.width_probability == row.probability_lower == row.probability_upper == 0.5
    zero = plan(methods.precision_poisson(rate=0, n=4, exposure_per_unit=0.25))
    assert zero.width == pytest.approx(-math.log(0.025), rel=2e-13)
    assert zero.width_probability == zero.probability_lower == zero.probability_upper == 1.0


@pytest.mark.parametrize("confidence", [0.6, 0.99999999])
def test_extreme_confidence_count_laws(confidence):
    result = methods.precision_binomial(p=1e-10, n=10, confidence=confidence)
    low, high, _ = cp_widths(10, confidence)
    law = result["width_distribution"]
    assert law["ci_low"].to_numpy() == pytest.approx(low, abs=3e-12)
    assert law["ci_high"].to_numpy() == pytest.approx(high, abs=3e-12)
    result = methods.precision_poisson(rate=0.03, n=4, confidence=confidence)
    law = result["width_distribution"]
    intervals = np.array([garwood(int(k), 4, confidence) for k in law["count"]])
    assert law[["ci_low", "ci_high", "width"]].to_numpy() == pytest.approx(intervals, abs=3e-12)


@pytest.mark.parametrize("mean", [0.0, 0.02, 1.0, 6.0, 30.0])
@pytest.mark.parametrize("assurance", [0.1, 0.9])
def test_poisson_garwood_distribution_full_tail_accounting(mean, assurance):
    result = methods.precision_poisson(
        rate=mean / 5, exposure_per_unit=0.5, n=10, assurance=assurance
    )
    row = plan(result)
    law = result["width_distribution"]
    k = law["count"].to_numpy(dtype=int)
    probabilities = stats.poisson.pmf(k, mean)
    intervals = np.array([garwood(int(count), 5, 0.95) for count in k])
    assert np.array_equal(k, np.arange(row.count_cutoff + 1))
    assert law["probability"].to_numpy() == pytest.approx(probabilities, abs=3e-14)
    assert law[["ci_low", "ci_high", "width"]].to_numpy() == pytest.approx(intervals, abs=2e-12)
    tail = stats.poisson.sf(int(row.count_cutoff), mean)
    assert row.omitted_mass == pytest.approx(tail, abs=2e-15)
    assert tail <= result.attrs["tail_tolerance"]
    assert row.enumerated_mass + row.omitted_mass == pytest.approx(1, abs=row.roundoff_bound)
    expected = probabilities[intervals[:, 2] <= row.width + 3e-12].sum()
    assert row.width_probability == pytest.approx(expected, abs=3e-13)
    assert row.probability_lower >= assurance
    assert law.iloc[0].ci_low == 0.0
    # The upper bound remains conservative even without assuming tail widths.
    assert row.probability_lower <= expected + 3e-14
    assert row.probability_upper >= expected - 3e-14


def test_poisson_first_n_and_exposure_scaling():
    result = methods.precision_poisson(
        rate=1.2, exposure_per_unit=0.5, width=3, assurance=0.8, max_n=20
    )
    row = plan(result)
    reference = []
    for n in range(1, 21):
        k = np.arange(151)
        widths = np.array([garwood(int(count), 0.5 * n, 0.95)[2] for count in k])
        probability = stats.poisson.pmf(k, 0.6 * n)[widths <= 3].sum()
        if probability >= 0.8:
            reference.append(n)
    assert row.n == reference[0]
    base = plan(methods.precision_poisson(rate=1.2, exposure_per_unit=0.5, n=10))
    scaled = plan(methods.precision_poisson(rate=0.12, exposure_per_unit=5, n=10))
    assert scaled.expected_count == base.expected_count
    assert scaled.width == pytest.approx(base.width / 10, rel=2e-13)
    assert scaled.width_probability == pytest.approx(base.width_probability, abs=2e-14)


@pytest.mark.parametrize(
    "function,arguments",
    [
        (methods.precision_twomeans_unknown, {"sd": 1.2, "n": 20, "ratio": 0.6}),
        (methods.precision_binomial, {"p": 0.2, "n": 20}),
        (methods.precision_poisson, {"rate": 1.2, "exposure_per_unit": 0.5, "n": 10}),
    ],
)
def test_full_state_deterministic_rng_isolation_and_float32_default(function, arguments):
    state = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        result = function(**arguments)
        first = summary_state(result)
        second = summary_state(function(**arguments))
    finally:
        torch.set_default_dtype(dtype)
    assert first == second
    assert torch.equal(state, torch.random.get_rng_state())
    restored = restore_summary(first)
    assert summary_state(restored) == first
    json.loads(first)
    assert restored.attrs == result.attrs
    assert result.attrs["solver"] == (
        "fixed-n native width quantile or exhaustive first-n search; no effect inversion"
    )
    assert restored.to_latex() == result.to_latex()
    settings = dict(zip(result["settings"]["setting"], result["settings"]["json"], strict=True))
    assert json.loads(settings["solver"]) == result.attrs["solver"]
    assert json.loads(settings["solver"]) != "bounded integer search / bracketed effect inversion"
    assert json.loads(settings["assurance"]) == 0.9
    assert result.attrs["device"] == "cpu" and result.attrs["precision"] == "float64"


@pytest.mark.parametrize(
    "function,arguments",
    [
        (methods.precision_twomeans_unknown, {"sd": True, "n": 20}),
        (methods.precision_twomeans_unknown, {"sd": 1, "n": 20, "ratio": 0}),
        (methods.precision_twomeans_unknown, {"sd": 1, "width": 0.001, "max_n": 3}),
        (methods.precision_binomial, {"p": True, "n": 20}),
        (methods.precision_binomial, {"p": 1.01, "n": 20}),
        (methods.precision_binomial, {"p": 0.2, "width": 0.001, "max_n": 5}),
        (methods.precision_poisson, {"rate": -1, "n": 10}),
        (methods.precision_poisson, {"rate": 1, "n": 10, "exposure_per_unit": 0}),
        (methods.precision_poisson, {"rate": 1, "n": 10, "max_count": 1}),
        (methods.precision_poisson, {"rate": 1, "n": 10, "tail_tolerance": 0}),
    ],
)
def test_structured_refusal(function, arguments):
    with pytest.raises(AnalysisError):
        function(**arguments)


@pytest.mark.parametrize(
    "function,arguments",
    [
        (methods.precision_twomeans_unknown, {"sd": 1}),
        (methods.precision_binomial, {"p": 0.2}),
        (methods.precision_poisson, {"rate": 1}),
    ],
)
def test_options_finite_domains(function, arguments):
    for more in (
        {},
        {"n": 3, "width": 1},
        {"n": True},
        {"n": 3, "confidence": 1},
        {"n": 3, "assurance": float("nan")},
        {"n": 3, "max_n": True},
    ):
        with pytest.raises(AnalysisError):
            function(**arguments, **more)


def test_count_workspace_and_exhaustive_work_refusal(monkeypatch):
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        methods.precision_poisson(rate=6500, n=1, max_n=1, max_count=10000)
    assert error.value.code == "workspace_limit"
    monkeypatch.setattr(methods, "MAX_WIDTH_EVALUATIONS", 5)
    with pytest.raises(AnalysisError) as error:
        methods.precision_binomial(p=0.2, width=0.01, max_n=50)
    assert error.value.code == "resource_limit"
    with pytest.raises(AnalysisError) as error:
        methods.precision_poisson(rate=1.2, n=10)
    assert error.value.code == "resource_limit"


def test_ambiguous_probability_is_not_skipped():
    with pytest.raises(AnalysisError) as error:
        methods._first_n(lambda n: (0.8 - 1e-13, 0.8 + 1e-13), 1, 20, 0.8)
    assert error.value.code == "unresolved_assurance"


def test_width_simulations_are_independent_of_planning_implementation():
    rng = np.random.default_rng(533535)
    n1, n2, sd = 20, 13, 1.4
    first = rng.normal(scale=sd, size=(40000, n1))
    second = rng.normal(scale=sd, size=(40000, n2))
    pooled = ((n1 - 1) * first.var(axis=1, ddof=1) + (n2 - 1) * second.var(axis=1, ddof=1)) / (
        n1 + n2 - 2
    )
    widths = 2 * stats.t.isf(0.025, n1 + n2 - 2) * np.sqrt(pooled * (1 / n1 + 1 / n2))
    threshold = plan(methods.precision_twomeans_unknown(sd=sd, n=n1, ratio=n2 / n1)).width
    assert abs(np.mean(widths <= threshold) - 0.9) < 0.009
    result = methods.precision_binomial(p=0.2, n=20)
    reference_widths = cp_widths(20, 0.95)[2]
    counts = rng.binomial(20, 0.2, size=50000)
    observed = np.mean(reference_widths[counts] <= plan(result).width + 3e-12)
    assert abs(observed - plan(result).width_probability) < 0.009
    result = methods.precision_poisson(rate=1.2, exposure_per_unit=0.5, n=10)
    counts = rng.poisson(6, size=50000)
    unique = np.array([garwood(k, 5, 0.95)[2] for k in range(int(counts.max()) + 1)])
    observed = np.mean(unique[counts] <= plan(result).width + 3e-12)
    assert abs(observed - plan(result).width_probability) < 0.009
