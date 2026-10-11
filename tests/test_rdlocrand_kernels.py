"""rdlocrand kernels against independent oracles.

Exact randomization laws are checked by brute-force enumeration in exact
rational arithmetic (fractions.Fraction, itertools), so ties are decided
exactly; large-sample tests against SciPy (Welch t statistic, rank sums,
Mann-Whitney with tie correction, ks_2samp, kstwobign, binomtest, F) and
statsmodels (WLS HC1/HC2/HC3); the TSLS sandwich against explicit NumPy algebra.
"""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import statsmodels.api as sm
import torch
from scipy import stats

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.rdlocrand import kernels as k

F64 = torch.float64


def _tensor(values):
    return torch.tensor(values, dtype=F64)


# ---- exact rational oracles ----------------------------------------------------------------------


def _mean(values):
    return sum(values, Fraction(0)) / len(values)


def _diffmeans(y, d, w=None):
    if w is None:
        return _mean([v for v, t in zip(y, d) if t]) - _mean([v for v, t in zip(y, d) if not t])
    w1 = sum((wi for wi, t in zip(w, d) if t), Fraction(0))
    w0 = sum((wi for wi, t in zip(w, d) if not t), Fraction(0))
    if w1 == 0 or w0 == 0:
        return None
    return (sum((wi * v for v, wi, t in zip(y, w, d) if t), Fraction(0)) / w1
            - sum((wi * v for v, wi, t in zip(y, w, d) if not t), Fraction(0)) / w0)


def _ranks(y):
    order = sorted(range(len(y)), key=lambda i: y[i])
    ranks = [Fraction(0)] * len(y)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y[order[j + 1]] == y[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = Fraction(i + j + 2, 2)
        i = j + 1
    return ranks


def _ranksum_sq(y, d):
    """Squared standardized control rank sum, exact (the sign is irrelevant to |z|)."""
    r, n = _ranks(y), len(y)
    n1 = sum(d)
    n0 = n - n1
    t0 = sum((ri for ri, t in zip(r, d) if not t), Fraction(0))
    mean = _mean(r)
    s2 = sum(((ri - mean) ** 2 for ri in r), Fraction(0)) / (n - 1)
    return (t0 - Fraction(n0 * (n + 1), 2)) ** 2 / (Fraction(n0 * n1) * s2 / n)


def _ks(y, d):
    n1 = sum(d)
    n0 = len(y) - n1
    best = Fraction(0)
    for v in set(y):
        f1 = Fraction(sum(1 for a, t in zip(y, d) if t and a <= v), n1)
        f0 = Fraction(sum(1 for a, t in zip(y, d) if not t and a <= v), n0)
        best = max(best, abs(f1 - f0))
    return best


def _absolute(name, y, d, w=None):
    if name == "diffmeans":
        value = _diffmeans(y, d, w)
        return None if value is None else abs(value)
    if name == "ranksum":
        return _ranksum_sq(y, d)
    return _ks(y, d)


def fixed_oracle(name, y, d, w=None):
    """Exact fixed-margins p-value over all subsets (Fraction arithmetic)."""
    y = [Fraction(v) for v in y]
    w = None if w is None else [Fraction(v) for v in w]
    n, n1 = len(y), sum(d)
    observed = _absolute(name, y, d, w)
    hits = total = 0
    for chosen in itertools.combinations(range(n), n1):
        a = [i in chosen for i in range(n)]
        value = _absolute(name, y, a, w)
        if value is None:
            continue
        total += 1
        hits += value >= observed
    return hits / total, total


def bernoulli_oracle(name, y, d, prob):
    y = [Fraction(v) for v in y]
    prob = [Fraction(p) for p in prob]
    n = len(y)
    observed = _absolute(name, y, d)
    hits = total = Fraction(0)
    for bits in itertools.product([False, True], repeat=n):
        if not any(bits) or all(bits):
            continue
        weight = Fraction(1)
        for p, b in zip(prob, bits):
            weight *= p if b else 1 - p
        if weight == 0:
            continue
        total += weight
        if _absolute(name, y, list(bits)) >= observed:
            hits += weight
    return float(hits / total)


def _statistic(name, y, d, fixed=True, weights=None):
    y2 = _tensor(y)[:, None]
    if name == "diffmeans":
        return k.DiffMeans(y2, d, weights)
    if name == "ranksum":
        return k.RankSum(y2, d, fixed)
    return k.KolmogorovSmirnov(y2, d, fixed)


# ---- assignment laws -----------------------------------------------------------------------------


@pytest.mark.parametrize("n, n1", [(6, 2), (7, 3), (7, 5), (8, 4), (5, 1), (9, 8)])
def test_colex_unranking_enumerates_every_subset_once(n, n1):
    law = k.Law(torch.tensor([True] * n1 + [False] * (n - n1)))
    kind, total = law.plan(1, "exact")
    assert kind == "exact" and total == math.comb(n, n1)
    seen = [tuple(row.nonzero().flatten().tolist()) for block, _ in k.assignments(law, kind, total, 0, 1)
            for row in block]
    assert len(seen) == len(set(seen)) == math.comb(n, n1)
    assert all(len(row) == n1 for row in seen)


def test_bernoulli_enumeration_weights_sum_to_one_and_fix_degenerate_units():
    prob = _tensor([0.3, 1.0, 0.0, 0.6, 0.5])
    law = k.Law(torch.tensor([False, True, False, True, False]), prob)
    kind, total = law.plan(10, "auto")
    assert kind == "exact" and total == 2 ** 3
    blocks = list(k.assignments(law, kind, total, 0, 1))
    weights = torch.cat([w for _, w in blocks])
    rows = torch.cat([b for b, _ in blocks])
    assert abs(float(weights.sum()) - 1) < 1e-15
    assert bool(rows[:, 1].all()) and not bool(rows[:, 2].any())


def test_plan_switches_between_exact_and_monte_carlo():
    law = k.Law(torch.tensor([True] * 4 + [False] * 6))          # C(10, 4) = 210
    assert law.plan(1000, "auto") == ("exact", 210)
    assert law.plan(100, "auto") == ("monte_carlo", 100)
    assert law.plan(100, "exact") == ("exact", 210)
    big = k.Law(torch.tensor([True] * 30 + [False] * 30))
    with pytest.raises(AnalysisError) as error:
        big.plan(100, "exact")
    assert error.value.code == "work_budget_exceeded"


# ---- exact randomization p-values ------------------------------------------------------------------

TIES_Y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
TIES_D = [True, False, True, False, False, True, False, True, False, False, True]


@pytest.mark.parametrize("name", ["diffmeans", "ranksum", "ksmirnov"])
def test_exact_fixed_margins_match_rational_brute_force_with_ties(name):
    d = torch.tensor(TIES_D)
    outcome = k.randomize([_statistic(name, TIES_Y, d)], k.Law(d), reps=1, seed=0, method="exact")
    p, total = fixed_oracle(name, TIES_Y, TIES_D)
    assert outcome.kind == "exact" and outcome.draws == total
    assert float(outcome.p_values[0][0]) == pytest.approx(p, abs=1e-14)


@pytest.mark.parametrize("name", ["diffmeans", "ranksum", "ksmirnov"])
def test_balanced_design_counts_mirror_assignments_as_ties(name):
    # n1 = n0: the complement assignment has |T| equal to the observed one in exact
    # arithmetic, which float64 rounding must not break.
    y = [0.1, 0.7, 0.3, 1.9, 2.3, 0.2, 1.1, 0.6]
    d = [True, False, True, True, False, False, True, False]
    dt = torch.tensor(d)
    outcome = k.randomize([_statistic(name, y, dt)], k.Law(dt), reps=1, seed=0, method="exact")
    assert float(outcome.p_values[0][0]) == pytest.approx(fixed_oracle(name, y, d)[0], abs=1e-14)


def test_exact_kernel_weighted_difference_excludes_undefined_assignments():
    y = [0.4, 1.3, -0.2, 0.8, 2.1, 1.7, 0.9]
    w = [0.0, 0.5, 0.25, 1.0, 0.0, 0.75, 0.5]
    d = [False, False, True, False, False, True, False]
    dt = torch.tensor(d)
    outcome = k.randomize([_statistic("diffmeans", y, dt, weights=_tensor(w))], k.Law(dt), reps=1,
                          seed=0, method="exact")
    p, total = fixed_oracle("diffmeans", y, d, w)
    assert float(outcome.p_values[0][0]) == pytest.approx(p, abs=1e-14)
    # The assignment treating units 0 and 4 (both zero weight) has no defined statistic.
    assert float(outcome.valid[0][0]) == total == math.comb(7, 2) - 1


@pytest.mark.parametrize("name", ["diffmeans", "ranksum", "ksmirnov"])
def test_exact_bernoulli_matches_rational_brute_force(name):
    y = [0.5, 2.0, 1.5, 0.5, 3.0, 2.5, 1.0, 4.0]
    d = [False, True, False, False, True, True, False, True]
    prob = [0.25, 0.5, 0.75, 0.5, 0.5, 0.25, 0.125, 1.0]
    dt = torch.tensor(d)
    outcome = k.randomize([_statistic(name, y, dt, fixed=False)], k.Law(dt, _tensor(prob)), reps=1,
                          seed=0, method="exact")
    assert float(outcome.p_values[0][0]) == pytest.approx(bernoulli_oracle(name, y, d, prob), abs=1e-13)


def test_monte_carlo_is_seeded_and_converges_to_the_exact_law():
    rng = np.random.default_rng(3)
    y = rng.normal(size=14).round(3).tolist()
    d = torch.tensor([i % 3 == 0 for i in range(14)])
    tests = lambda: [_statistic(name, y, d) for name in ("diffmeans", "ranksum", "ksmirnov")]  # noqa: E731
    exact = k.randomize(tests(), k.Law(d), reps=1, seed=0, method="exact")
    first = k.randomize(tests(), k.Law(d), reps=40000, seed=11, method="monte_carlo")
    again = k.randomize(tests(), k.Law(d), reps=40000, seed=11, method="monte_carlo")
    other = k.randomize(tests(), k.Law(d), reps=40000, seed=12, method="monte_carlo")
    for e, a, b, c in zip(exact.p_values, first.p_values, again.p_values, other.p_values):
        p = float(e[0])
        assert float(a[0]) == float(b[0])
        assert abs(float(a[0]) - p) <= 4.5 * math.sqrt(p * (1 - p) / 40000) + 1e-12
        assert abs(float(c[0]) - p) <= 4.5 * math.sqrt(p * (1 - p) / 40000) + 1e-12
    assert any(float(a[0]) != float(c[0]) for a, c in zip(first.p_values, other.p_values))


def test_random_subsets_are_uniform():
    law = k.Law(torch.tensor([True, True, False, False, False]))
    draws = torch.cat([block for block, _ in k.assignments(law, "monte_carlo", 50000, 4, 1)])
    codes = (draws.to(torch.int64) * (2 ** torch.arange(5))).sum(1)
    counts = torch.bincount(codes, minlength=32)[[sum(2 ** i for i in c) for c in itertools.combinations(range(5), 2)]]
    assert int(counts.sum()) == 50000
    assert stats.chisquare(counts.numpy()).pvalue > 1e-4


def test_hotelling_matches_numpy_and_its_randomization_law():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(12, 2)).round(3)
    d = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=bool)

    def t2(dd):
        x1, x0 = x[dd], x[~dd]
        n1, n0, n = len(x1), len(x0), len(x)
        pooled = (np.cov(x1.T) * (n1 - 1) + np.cov(x0.T) * (n0 - 1)) / (n - 2)
        diff = x1.mean(0) - x0.mean(0)
        return n0 * n1 / n * diff @ np.linalg.solve(pooled, diff)

    test = k.Hotelling(_tensor(x.tolist()), torch.tensor(d.tolist()))
    assert test.observed == pytest.approx(t2(d), rel=1e-12)
    n, p = 12, 2
    f_value = (n - p - 1) / ((n - 2) * p) * t2(d)
    assert k.hotelling_f(test.observed, n, p) == pytest.approx(stats.f.sf(f_value, p, n - 1 - p), rel=1e-10)
    observed = t2(d)
    values = []
    for chosen in itertools.combinations(range(12), 6):
        dd = np.zeros(12, bool)
        dd[list(chosen)] = True
        values.append(t2(dd))
    expected = np.mean(np.array(values) >= observed * (1 - 1e-10))
    outcome = k.randomize([test], k.Law(torch.tensor(d.tolist())), reps=1, seed=0, method="exact")
    assert float(outcome.p_values[0][0]) == pytest.approx(expected, abs=1e-14)


# ---- large-sample tests -----------------------------------------------------------------------


def test_neyman_difference_is_welch_statistic_with_normal_reference():
    rng = np.random.default_rng(8)
    y = rng.normal(size=40) + np.r_[np.zeros(22), 0.6 * np.ones(18)]
    d = np.r_[np.zeros(22, bool), np.ones(18, bool)]
    diff, se = k.neyman(_tensor(y.tolist())[:, None], torch.tensor(d.tolist()))
    welch = stats.ttest_ind(y[d], y[~d], equal_var=False)
    assert float(diff[0] / se[0]) == pytest.approx(welch.statistic, rel=1e-12)
    z, p = k.z_test(float(diff[0]), float(se[0]))
    assert p == pytest.approx(2 * stats.norm.sf(abs(welch.statistic)), rel=1e-12)


def test_rank_sum_matches_scipy_without_and_with_ties():
    rng = np.random.default_rng(9)
    y = rng.normal(size=30)
    d = rng.random(30) < 0.4
    test = _statistic("ranksum", y.tolist(), torch.tensor(d.tolist()))
    assert float(test.observed[0]) == pytest.approx(stats.ranksums(y[~d], y[d]).statistic, rel=1e-12)
    tied = np.round(y * 2) / 2
    test = _statistic("ranksum", tied.tolist(), torch.tensor(d.tolist()))
    z = float(test.observed[0])
    mw = stats.mannwhitneyu(tied[~d], tied[d], use_continuity=False, method="asymptotic")
    assert k.two_sided(z) == pytest.approx(mw.pvalue, rel=1e-10)


def test_kolmogorov_smirnov_large_sample_branches():
    rng = np.random.default_rng(10)
    y = rng.normal(size=60)
    d = rng.random(60) < 0.5
    test = _statistic("ksmirnov", y.tolist(), torch.tensor(d.tolist()))
    reference = stats.ks_2samp(y[~d], y[d], method="exact")
    assert float(test.observed[0]) == pytest.approx(reference.statistic, rel=1e-14)
    assert k.ks_large_sample(test, 0) == pytest.approx(reference.pvalue, rel=1e-10)
    for x in (0.2, 0.5, 0.8, 0.95, 0.999, 1.0, 1.2, 2.0, 4.0):
        assert k.kolmogorov_sf(x) == pytest.approx(stats.kstwobign.sf(x), rel=1e-12, abs=1e-300)
    big = rng.normal(size=260)
    dd = rng.random(260) < 0.5
    test = _statistic("ksmirnov", big.tolist(), torch.tensor(dd.tolist()))
    assert test.n0 * test.n1 >= 10000
    statistic = stats.ks_2samp(big[~dd], big[dd]).statistic
    # R's ks.test limit: Kolmogorov law at sqrt(n0 n1/(n0 + n1)) D (SciPy's 'asymp' uses kstwo).
    assert k.ks_large_sample(test, 0) == pytest.approx(
        stats.kstwobign.sf(math.sqrt(test.n0 * test.n1 / 260) * statistic), rel=1e-11)


def test_tie_aware_smirnov_law_equals_enumeration():
    y = [1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7]
    d = [False, True, False, True, True, False, False, True, False, True, True, False]
    test = _statistic("ksmirnov", y, torch.tensor(d))
    assert k.ks_large_sample(test, 0) == pytest.approx(fixed_oracle("ksmirnov", y, d)[0], abs=1e-14)
    tiny = k.smirnov_sf(1, 3, 3, torch.ones(6, dtype=torch.bool))
    assert tiny == 1.0


def test_binomial_test_matches_scipy():
    for treated, total in ((3, 10), (10, 10), (5, 10), (17, 40), (60, 100), (0, 7)):
        assert k.binomial_half(treated, total) == pytest.approx(
            stats.binomtest(treated, total, 0.5).pvalue, rel=1e-10)


@pytest.mark.parametrize("vce", ["HC1", "HC2", "HC3"])
def test_polynomial_hc_fit_matches_statsmodels(vce):
    rng = np.random.default_rng(12)
    r = rng.uniform(-1, 1, 80)
    d = r >= 0
    w = np.where(rng.random(80) < 0.1, 0.0, rng.uniform(0.2, 1, 80))
    y = 1 + r + 0.4 * d + rng.normal(size=80)
    powers = np.c_[r, r ** 2]
    fit = k.hc_fit(_tensor(y.tolist()), torch.tensor(d.tolist()), _tensor(powers.tolist()),
                   _tensor(w.tolist()), vce)
    keep = w > 0
    z = np.c_[np.ones(80), d, powers, d[:, None] * powers][keep]
    model = sm.WLS(y[keep], z, weights=w[keep]).fit(cov_type=vce)
    assert float(fit.estimate[0]) == pytest.approx(model.params[1], rel=1e-10)
    assert float(fit.se[0]) == pytest.approx(model.bse[1], rel=1e-9)
    assert fit.n == int(keep.sum())


def test_tsls_hc_fit_matches_explicit_algebra():
    rng = np.random.default_rng(13)
    r = rng.uniform(-1, 1, 90)
    d = (r >= 0).astype(float)
    t = ((rng.random(90) < 0.2) | ((r >= 0) & (rng.random(90) < 0.8))).astype(float)
    y = 0.5 + 0.8 * t + r + rng.normal(size=90)
    w = 1 - np.abs(r)
    z = np.c_[np.ones(90), d, r, d * r]
    x = z.copy()
    x[:, 1] = t
    sw = np.sqrt(w)
    projected = z @ np.linalg.lstsq(z * sw[:, None], x * sw[:, None], rcond=None)[0]
    bread = np.linalg.inv((projected * w[:, None]).T @ projected)
    beta = bread @ (projected * w[:, None]).T @ y
    resid = y - x @ beta
    h = np.einsum("ij,jk,ik->i", projected * sw[:, None], bread, projected * sw[:, None])
    loading = projected @ bread[:, 1]
    for vce, scale in (("HC1", np.full(90, math.sqrt(90 / 86))), ("HC3", 1 / (1 - h))):
        fit = k.hc_fit(_tensor(y.tolist()), torch.tensor((d > 0).tolist()), _tensor(r[:, None].tolist()),
                       _tensor(w.tolist()), vce, _tensor(t.tolist()))
        assert float(fit.estimate[0]) == pytest.approx(beta[1], rel=1e-10)
        assert float(fit.se[0]) == pytest.approx(np.sqrt(np.sum((loading * w * resid * scale) ** 2)), rel=1e-9)


def test_hc_fit_refuses_unidentified_polynomials():
    r = torch.linspace(-1, 1, 6, dtype=F64)
    d = r >= 0
    with pytest.raises(AnalysisError) as error:
        k.hc_fit(r.clone(), d, torch.stack([r, r ** 2, r ** 3], 1), None, "HC3")
    assert error.value.code == "insufficient_observations"
