"""Independent finite-count enumeration and survival-time integration oracles."""

import json
import math
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats import planning_discrete_survival as methods
from openecon.econometrics.stats.planning_discrete_survival import (
    power_mcnemar_unconditional,
    survival_accrual,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def enrollment_oracle(n, allocation):
    arm2 = int((Decimal(str(allocation)) * n).to_integral_value(rounding=ROUND_FLOOR))
    return [n - arm2, arm2]


def rejection_bounds(d, alpha, alternative):
    if d == 0:
        return -1, 1
    tail = Fraction(alpha / (2 if alternative == "two-sided" else 1))
    eligible = [
        k
        for k in range(d + 1)
        if Fraction(sum(math.comb(d, j) for j in range(k + 1)), 2**d) <= tail
    ]
    lower = max(eligible) if eligible and alternative != "upper" else -1
    upper = d - max(eligible) if eligible and alternative != "lower" else d + 1
    return lower, upper


def mixture_oracle(n, p10, p01, alpha=0.05, alternative="two-sided"):
    q = p10 + p01
    if q == 0:
        return 0.0, 0.0
    theta = p10 / q
    rejection, size = [], []
    for d in range(n + 1):
        lo, hi = rejection_bounds(d, alpha, alternative)
        rejection.append(stats.binom.cdf(lo, d, theta) + stats.binom.sf(hi - 1, d, theta))
        size.append(stats.binom.cdf(lo, d, 0.5) + stats.binom.sf(hi - 1, d, 0.5))
    weights = stats.binom.pmf(np.arange(n + 1), n, q)
    return float(weights @ rejection), float(weights @ size)


def multinomial_oracle(n, p10, p01, alpha, alternative):
    probability = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            lo, hi = rejection_bounds(b + c, alpha, alternative)
            if b <= lo or b >= hi:
                rest = n - b - c
                coefficient = math.comb(n, b) * math.comb(n - b, c)
                probability += coefficient * p10**b * p01**c * (1 - p10 - p01) ** rest
    return probability


@pytest.mark.parametrize("n", [1, 6, 11, 17])
@pytest.mark.parametrize("p10,p01", [(0.3, 0.1), (0.1, 0.4), (0.8, 0.2), (0.0, 0.4)])
@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
def test_mcnemar_direct_multinomial_and_mixture_oracles(n, p10, p01, alternative):
    result = power_mcnemar_unconditional(p10, p01, n=n, alternative=alternative)
    plan = result["plan"].iloc[0]
    expected, size = mixture_oracle(n, p10, p01, alternative=alternative)
    direct = multinomial_oracle(n, p10, p01, 0.05, alternative)
    assert plan.power == pytest.approx(expected, abs=2e-14)
    assert plan.power == pytest.approx(direct, abs=2e-14)
    assert plan.actual_alpha == pytest.approx(size, abs=2e-14)
    assert plan.actual_alpha <= 0.05 + 1e-14
    regions = result["conditional_rejection"]
    assert len(regions) == min(n + 1, 1000) + 1
    for entry in regions.to_dict("records"):
        count = int(entry["discordant_pairs"])
        assert entry["discordant_pairs"] == count
        lo, hi = rejection_bounds(count, 0.05, alternative)
        assert entry["lower_critical"] == lo and entry["upper_critical"] == hi


def test_mcnemar_random_discordance_not_projected_conditional_power():
    result = power_mcnemar_unconditional(0.3, 0.1, n=100)
    plan = result["plan"].iloc[0]
    expected, size = mixture_oracle(100, 0.3, 0.1)
    assert plan.power == pytest.approx(expected, abs=4e-14)
    assert plan.actual_alpha == pytest.approx(size, abs=4e-14)
    conditional = oe.power_mcnemar(0.75, discordant_pairs=40)["plan"].iloc[0].power
    assert abs(plan.power - conditional) > 0.005
    assert plan.expected_discordant_pairs == 40


@pytest.mark.parametrize(
    "alternative,p10,p01", [("two-sided", 0.7, 0.3), ("upper", 0.4, 0.1), ("lower", 0.1, 0.4)]
)
def test_mcnemar_exhaustive_search_certifies_every_prior_size(alternative, p10, p01):
    result = power_mcnemar_unconditional(p10, p01, power=0.65, max_n=200, alternative=alternative)
    plan = result["plan"].iloc[0]
    search = result["search"]
    count = int(plan.n)
    assert list(search.n) == list(range(1, count + 1))
    assert list(search.power[:-1] >= 0.65) == [False] * (count - 1)
    assert plan.power >= 0.65
    assert count == next(
        n for n in range(1, 201) if mixture_oracle(n, p10, p01, alternative=alternative)[0] >= 0.65
    )
    assert result.attrs["n_search_evaluations"] == count
    assert len(result["conditional_rejection"]) == min(count + 1, 200) + 1


def test_mcnemar_nonmonotone_power_and_dyadic_boundary():
    result = power_mcnemar_unconditional(0.7, 0.3, power=0.75, max_n=200)
    assert np.any(np.diff(result["search"].power) < -0.005)
    output = power_mcnemar_unconditional(0.8, 0.2, n=4, alpha=0.125)
    conditional = output["conditional_rejection"].set_index("discordant_pairs")
    assert conditional.loc[4, "lower_critical"] == 0
    assert conditional.loc[4, "upper_critical"] == 4
    assert output["plan"].iloc[0].actual_alpha == pytest.approx(0.125)


def test_mcnemar_zero_discordance_null_and_exchange_direction():
    zero = power_mcnemar_unconditional(0, 0, n=100)
    assert zero["plan"].iloc[0].power == zero["plan"].iloc[0].actual_alpha == 0
    assert zero.attrs["conditional_probability"] is None
    null = power_mcnemar_unconditional(0.2, 0.2, n=90)["plan"].iloc[0]
    assert null.power == pytest.approx(null.actual_alpha, abs=3e-15)
    upper = power_mcnemar_unconditional(0.4, 0.1, n=80, alternative="upper")["plan"].iloc[0]
    lower = power_mcnemar_unconditional(0.1, 0.4, n=80, alternative="lower")["plan"].iloc[0]
    assert upper.power == pytest.approx(lower.power, abs=3e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(p10=True, p01=0.2, n=20),
        dict(p10=-0.1, p01=0.2, n=20),
        dict(p10=0.9, p01=0.2, n=20),
        dict(p10=float("nan"), p01=0.2, n=20),
        dict(p10=0.3, p01=0.1),
        dict(p10=0.3, p01=0.1, n=20, power=0.8),
        dict(p10=0.3, p01=0.1, n=True),
        dict(p10=0.3, p01=0.1, n=1001),
        dict(p10=0.3, p01=0.1, power=0.99, max_n=1),
        dict(p10=0, p01=0, power=0.8),
        dict(p10=0.2, p01=0.2, power=0.8),
        dict(p10=0.1, p01=0.4, power=0.8, alternative="upper"),
        dict(p10=0.3, p01=0.1, n=20, alpha=0.5),
        dict(p10=0.3, p01=0.1, n=20, alternative="upper", direction="lower"),
    ],
)
def test_mcnemar_invalid_or_unattainable_inputs(kwargs):
    with pytest.raises(AnalysisError):
        power_mcnemar_unconditional(**kwargs)


def probability_integral(event, dropout, accrual, followup):
    if event == 0 or accrual + followup == 0:
        return 0.0

    def conditional(entry):
        return integrate.quad(
            lambda elapsed: event * math.exp(-(event + dropout) * elapsed),
            0,
            accrual + followup - entry,
            epsabs=1e-13,
        )[0]

    if accrual == 0:
        return conditional(0)
    return integrate.quad(conditional, 0, accrual, epsabs=1e-13)[0] / accrual


@pytest.mark.parametrize("accrual,followup", [(0, 6), (12, 0), (12, 6), (0, 0), (1e-6, 1e-6)])
@pytest.mark.parametrize(
    "rates,dropouts", [((0.12, 0.08), (0.02, 0.01)), ((0.12, 0.08), (0, 0)), ((0, 0.1), (0, 0.03))]
)
def test_survival_nested_time_integral_and_integer_arms(accrual, followup, rates, dropouts):
    result = survival_accrual(
        event_rate1=rates[0],
        event_rate2=rates[1],
        dropout_rate1=dropouts[0],
        dropout_rate2=dropouts[1],
        accrual=accrual,
        followup=followup,
        allocation=0.37,
        n=201,
    )
    plan = result["plan"].iloc[0]
    counts = enrollment_oracle(201, 0.37)
    expected = [probability_integral(r, d, accrual, followup) for r, d in zip(rates, dropouts)]
    assert list(result["arms"].enrollment) == counts
    assert list(result["arms"].event_probability) == pytest.approx(expected, abs=2e-13, rel=1e-12)
    assert plan.expected_events == pytest.approx(np.dot(counts, expected), abs=4e-11)
    assert plan.achieved_allocation == counts[1] / 201
    assert (
        result.attrs["inference"]
        == "analytic expected observed event counts; no power or count assurance"
    )


@pytest.mark.parametrize("allocation", [0.01, 0.37, 0.5, 0.99])
def test_survival_target_minimum_matches_exhaustive_count_oracle(allocation):
    kwargs = dict(
        event_rate1=0.12,
        event_rate2=0.08,
        dropout_rate1=0.02,
        dropout_rate2=0.01,
        accrual=12,
        followup=6,
        allocation=allocation,
    )
    result = survival_accrual(target_events=71.3, **kwargs)
    plan = result["plan"].iloc[0]
    probabilities = [
        probability_integral(0.12, 0.02, 12, 6),
        probability_integral(0.08, 0.01, 12, 6),
    ]
    expected = next(
        n
        for n in range(2, 1000)
        if all(value > 0 for value in enrollment_oracle(n, allocation))
        and np.dot(enrollment_oracle(n, allocation), probabilities)
        >= 71.3
    )
    assert plan.n == expected
    assert plan.expected_events >= 71.3
    if expected > result.attrs["min_n"]:
        assert survival_accrual(n=expected - 1, **kwargs)["plan"].iloc[0].expected_events < 71.3


@pytest.mark.parametrize("allocation,expected", [(0.99, [1, 99]), (0.58, [42, 58])])
def test_survival_decimal_allocation_boundary_and_minimum_target(allocation, expected):
    kwargs = dict(
        event_rate1=0.12, event_rate2=0.08, dropout_rate1=0.02, dropout_rate2=0.01,
        accrual=12, followup=6, allocation=allocation,
    )
    result = survival_accrual(n=100, **kwargs)
    plan = result["plan"].iloc[0]
    assert [plan.n1, plan.n2] == expected == enrollment_oracle(100, allocation)
    probabilities = [
        probability_integral(0.12, 0.02, 12, 6), probability_integral(0.08, 0.01, 12, 6)
    ]
    at99 = np.dot(enrollment_oracle(99, allocation), probabilities)
    at100 = np.dot(expected, probabilities)
    target = float((at99 + at100) / 2)
    solved = survival_accrual(target_events=target, max_n=200, **kwargs)
    assert solved["plan"].iloc[0].n == 100
    assert at99 < target < at100
    assert solved.attrs["allocation_rational"] == {
        "numerator": Fraction(str(allocation)).numerator,
        "denominator": Fraction(str(allocation)).denominator,
    }


def test_survival_stable_tiny_hazards_saturation_units_and_zero_event_law():
    tiny = survival_accrual(event_rate1=1e-100, event_rate2=1e-100, accrual=1e-100, followup=0, n=2)
    assert tiny["plan"].iloc[0].event_probability1 == pytest.approx(5e-201, rel=1e-13, abs=0)
    saturated = survival_accrual(
        event_rate1=1e100, event_rate2=1e100, accrual=1e100, followup=1e100, n=2
    )
    assert saturated["plan"].iloc[0].expected_events == 2
    base = survival_accrual(
        event_rate1=0.12,
        event_rate2=0.08,
        dropout_rate1=0.02,
        dropout_rate2=0.01,
        accrual=12,
        followup=6,
        n=200,
    )
    units = survival_accrual(
        event_rate1=0.12 / 365,
        event_rate2=0.08 / 365,
        dropout_rate1=0.02 / 365,
        dropout_rate2=0.01 / 365,
        accrual=12 * 365,
        followup=6 * 365,
        n=200,
    )
    assert base["plan"].iloc[0].expected_events == pytest.approx(
        units["plan"].iloc[0].expected_events, abs=3e-14
    )
    zero = survival_accrual(event_rate1=0, event_rate2=0, accrual=12, followup=6, n=20)
    assert zero["plan"].iloc[0].expected_events == 0


@pytest.mark.parametrize(
    "patch",
    [
        dict(event_rate1=-0.1),
        dict(event_rate1=True),
        dict(event_rate2=float("inf")),
        dict(dropout_rate1=-0.1),
        dict(accrual=-1),
        dict(followup=False),
        dict(allocation=0),
        dict(allocation=1),
        dict(n=True),
        dict(n=1),
        dict(n=10_000_001),
        dict(n=None),
        dict(target_events=20),
        dict(n=None, target_events=0),
        dict(n=None, target_events=100, max_n=10),
        dict(n=None, target_events=1, event_rate1=0, event_rate2=0),
        dict(n=None, target_events=1, accrual=0, followup=0),
        dict(allocation=0.01, n=2),
    ],
)
def test_survival_invalid_or_unattainable_inputs(patch):
    kwargs = dict(event_rate1=0.12, event_rate2=0.08, accrual=12, followup=6, n=20)
    kwargs.update(patch)
    with pytest.raises(AnalysisError):
        survival_accrual(**kwargs)


@pytest.mark.parametrize(
    "result",
    [
        lambda: power_mcnemar_unconditional(0.3, 0.1, power=0.7, max_n=200),
        lambda: power_mcnemar_unconditional(0.3, 0.1, n=100),
        lambda: power_mcnemar_unconditional(0, 0, n=100),
        lambda: survival_accrual(
            event_rate1=0.12, event_rate2=0.08, accrual=12, followup=6, target_events=80
        ),
        lambda: survival_accrual(
            event_rate1=0.12,
            event_rate2=0.08,
            dropout_rate1=0.02,
            dropout_rate2=0.01,
            accrual=12,
            followup=6,
            allocation=0.4,
            n=200,
        ),
        lambda: survival_accrual(event_rate1=0, event_rate2=0, accrual=0, followup=0, n=200),
    ],
)
def test_complete_state_roundtrip_latex_and_global_rng_isolation(result):
    before = torch.random.get_rng_state().clone()
    output = result()
    assert torch.equal(before, torch.random.get_rng_state())
    state = oe.summary_state(output)
    json.dumps(json.loads(state), allow_nan=False)
    restored = oe.restore_summary(state)
    assert restored.attrs == output.attrs
    assert set(restored) == set(output)
    for name in output:
        assert restored[name].to_dict("split") == output[name].to_dict("split")
    assert "\\begin{tabular}" in output.to_latex()
    assert restored.to_latex() == output.to_latex()
    settings = {row.setting: json.loads(row.json) for row in output["settings"].itertuples()}
    assert settings["solver"] == output.attrs["solver"]
    assert output.attrs["device"] == "cpu" and output.attrs["precision"] == "float64"


def test_early_resource_admission_before_probability_allocation(monkeypatch):
    original = methods.plan_workspace
    monkeypatch.setattr(
        methods,
        "plan_workspace",
        lambda operation, buffers: original(operation, buffers, budget_bytes=1),
    )
    monkeypatch.setattr(
        methods,
        "_conditional_law",
        lambda *args: pytest.fail("probabilities allocated before admission"),
    )
    with pytest.raises(AnalysisError, match="workspace"):
        power_mcnemar_unconditional(0.3, 0.1, n=100)
    monkeypatch.setattr(
        methods,
        "_event_probabilities",
        lambda *args: pytest.fail("probabilities allocated before admission"),
    )
    with pytest.raises(AnalysisError, match="workspace"):
        survival_accrual(event_rate1=0.12, event_rate2=0.08, accrual=12, followup=6, n=200)
