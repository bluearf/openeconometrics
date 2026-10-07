"""Adversarial review tests for the reference-distribution kernel.

Independent oracles only: closed forms, brute-force sums and quadratures, numpy/scipy and
mpmath where they are trustworthy. The regression tests at the top pin down defects found
during review (astronomical degrees of freedom, a continued fraction run on its divergent
side, a cancelling Halley residual next to p = 1/2, underflowing odds, text arguments).
"""

import math
import time

import numpy as np
import pytest
import torch
from scipy import integrate, special, stats

from openecon.engines import distributions as dist
from openecon.engines.contracts import KernelError

RTOL = 1e-10
PROBABILITIES = [1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-10, 1e-6, 1e-3, .01, .05, .2, .4999, .5,
                 .7, .9, .99, 1 - 1e-6, 1 - 1e-10, 1 - 1e-13, 1 - 1e-16]


def close(actual, expected, rtol=RTOL, atol=1e-310):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=rtol, atol=atol)


# --------------------------------------------------------------------------------------
# Regressions for defects found in review
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("df", [1e290, 1e300, 1e308])
def test_astronomical_t_df_reaches_the_normal_limit(df):
    """t_sf(1.5, 1e300) used to return 0: the x < 1e-290 corner dropped an O(b x) term."""
    xs = [1e-8, .05, .3, 1, 1.5, 2, 4, 8, 20, 30, 37]
    close([dist.t_sf(x, df) for x in xs], stats.norm.sf(xs), rtol=1e-10)
    close([dist.t_cdf(x, df) for x in xs], stats.norm.cdf(xs), rtol=1e-10)
    close([dist.t_cdf(-x, df) for x in xs], stats.norm.sf(xs), rtol=1e-10)
    for x in (.3, 2, 8, 30):
        assert abs(dist.t_isf(stats.norm.sf(x), df) - x) <= 1e-9 * x
        assert dist.p_value(x, "t", df) == pytest.approx(dist.p_value(x, "normal"), rel=1e-10)


@pytest.mark.parametrize("df_big", [1e291, 1e300])
@pytest.mark.parametrize("df_small", [1, 3, 7.5])
def test_astronomical_f_df_reaches_the_chi2_limits(df_big, df_small):
    for x in (.01, .5, 2, 10, 50):
        close(dist.f_cdf(x, df_small, df_big), stats.chi2.cdf(df_small * x, df_small))
        close(dist.f_sf(x, df_small, df_big), stats.chi2.sf(df_small * x, df_small))
        close(dist.f_sf(x, df_big, df_small), stats.chi2.cdf(df_small / x, df_small))
        close(dist.f_cdf(x, df_big, df_small), stats.chi2.sf(df_small / x, df_small))
        # Invert the tail that is resolved by a float (the smaller one).
        p, q = stats.chi2.cdf(df_small * x, df_small), stats.chi2.sf(df_small * x, df_small)
        quantile = dist.f_ppf(p, df_small, df_big) if p <= q else dist.f_isf(q, df_small, df_big)
        assert abs(quantile - x) <= 1e-9 * x


def test_huge_numerator_df_keeps_the_continued_fraction_on_its_convergent_side():
    """f_cdf(1e-5, 1e18, 1e-3) returned 1.0000014 and f_sf -1.4e-6: the side of the beta
    continued fraction was decided from w = 1 - 1e-16 rounded to 1 instead of from v."""
    # F(df1, df2) with df1 -> inf is df2 / chi2(df2): P(F < x) = Q(df2 / 2, df2 / (2 x)).
    assert dist.f_cdf(1e-5, 1e18, 1e-3) == pytest.approx(special.gammaincc(5e-4, 50), rel=1e-9)
    for df1 in (2e16, 1e18, 1e20, 1e100):
        for df2 in (1e-3, .5, 2, 30):
            for x in (1e-5, .1, 1, 10, 1e5):
                lower, upper = dist.f_cdf(x, df1, df2), dist.f_sf(x, df1, df2)
                assert 0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0
                if min(lower, upper) > 1e-290:
                    assert lower + upper == pytest.approx(1.0, abs=4e-16)
                limit_lower = stats.chi2.sf(df2 / x, df2)
                limit_upper = stats.chi2.cdf(df2 / x, df2)
                if limit_lower > 1e-300:
                    assert lower == pytest.approx(limit_lower, rel=1e-9)
                if limit_upper > 1e-300:
                    assert upper == pytest.approx(limit_upper, rel=1e-9)


def test_normal_quantile_keeps_relative_accuracy_next_to_one_half():
    """Phi(x) - p cancels for p ~ 1/2, which left Acklam's 1e-9 start unrefined."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(30):
        for delta in (2 ** -53, 2 ** -50, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, .01, .1, .2, .24):
            for p in (.5 - delta, .5 + delta):
                exact = float(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1))
                assert dist.normal_ppf(p) == pytest.approx(exact, rel=1e-15, abs=0)
                assert dist.normal_isf(p) == -dist.normal_ppf(p)
    # Taylor law next to the median: x = s d + s^3 d^3 / 6 + O(d^5), s = sqrt(2 pi).
    s = math.sqrt(2 * math.pi)
    for d in (1e-17, 1e-12, 1e-8, 1e-6):
        assert dist.normal_ppf(.5 + d) == pytest.approx(s * d + (s * d) ** 3 / 6, rel=1e-15)


def test_underflowing_f_odds_are_handled_in_logs():
    """x df1 / df2 below the float range: I_w(a, b) = w^a / (a B(a, b)) from ln w."""
    for x, df1, df2 in ((1e-300, 1e-3, 1e30), (1e-300, 1e-3, 1e300), (1e-10, 1e-6, 1e306)):
        log_w = math.log(x) + math.log(df1) - math.log(df2)
        exact = math.exp(.5 * df1 * log_w - special.betaln(.5 * df1, .5 * df2)) / (.5 * df1)
        assert 0 < exact < 1
        assert dist.f_cdf(x, df1, df2) == pytest.approx(exact, rel=1e-12)
        assert dist.f_sf(x, df1, df2) == pytest.approx(1 - exact, rel=1e-12)
        assert dist.f_sf(1 / x, df2, df1) == pytest.approx(exact, rel=1e-12)


def test_text_arguments_are_rejected_and_integer_like_orders_accepted():
    for call in (lambda: dist.t_sf("1", 2), lambda: dist.t_sf(1, "2"), lambda: dist.chi2_sf(b"1", 2),
                 lambda: dist.normal_ppf("0.5"), lambda: dist.p_value("1.96", "normal"),
                 lambda: dist.critical("0.05", "chi2", 3), lambda: dist.beta_inc(1, 1, "0.5")):
        with pytest.raises(KernelError) as error:
            call()
        assert error.value.code == "invalid_inference"
    for n in (np.int64(5), np.int32(5), torch.tensor(5), torch.tensor(5, dtype=torch.int32)):
        nodes, weights = dist.gauss_hermite(n)
        assert nodes.shape == weights.shape == (5,)
        assert torch.equal(dist.gauss_legendre(n)[0], dist.gauss_legendre(5)[0])
    for bad in (np.float64(4.0), torch.tensor(4.0), np.bool_(True), "4", 4.0, None):
        for rule in (dist.gauss_hermite, dist.gauss_legendre):
            with pytest.raises(KernelError) as error:
                rule(bad)
            assert error.value.code == "invalid_quadrature"


# --------------------------------------------------------------------------------------
# Incomplete gamma: branch boundaries, brute-force oracles, complements
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("a", [.999, 1, 1.001, 999.9, 1000, 1000.1, 1999.9, 2000.1])
def test_incomplete_gamma_is_continuous_across_its_branch_switches(a):
    """a = 1 (small-a series), x = a + 1 (series / fraction) and a = 1000 (quadrature)."""
    points = sorted({a + 1 - 1e-9, a + 1, a + 1 + 1e-9, .5 * a, a, 1.5 * a, .999, 1.0, 1.001,
                     1.0999, 1.1, 1.1001, .4999, .5, .5001, 1e-6, 1e-3, 1e-1, 10, 100})
    close([dist.gamma_p(a, x) for x in points], special.gammainc(a, points))
    close([dist.gamma_q(a, x) for x in points], special.gammaincc(a, points))


def test_small_shape_branch_boundary_is_seamless():
    """a <= 0.75 x (x >= 1/2) or a <= -0.4 / ln x selects the direct small-tail series."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(30):
        for x in (.1, .3, .5, .5000001, .7, 1.0, 1.0999):
            threshold = .75 * x if x >= .5 else -.4 / math.log(x)
            for a in (threshold * (1 - 1e-12), threshold, threshold * (1 + 1e-12), .5 * threshold):
                if not 0 < a < 1:
                    continue
                upper = float(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True))
                lower = float(mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True))
                assert dist.gamma_q(a, x) == pytest.approx(upper, rel=2e-12, abs=0)
                assert dist.gamma_p(a, x) == pytest.approx(lower, rel=2e-12, abs=0)


def test_incomplete_gamma_against_brute_force_quadrature():
    for a, x in ((.3, .2), (3.7, 2.2), (12.5, 20), (.05, 1e-3), (250, 230), (999, 1050)):
        density = lambda t: math.exp((a - 1) * math.log(t) - t - math.lgamma(a))  # noqa: E731
        lower, _ = integrate.quad(density, 0, x, epsabs=0, epsrel=1e-13, limit=400)
        upper, _ = integrate.quad(density, x, math.inf, epsabs=0, epsrel=1e-13, limit=400)
        assert dist.gamma_p(a, x) == pytest.approx(lower, rel=1e-10)
        assert dist.gamma_q(a, x) == pytest.approx(upper, rel=1e-10)


def test_chi2_even_df_matches_the_poisson_sum():
    """Q(x; 2k) = sum_{j<k} e^-x/2 (x/2)^j / j! and P(x; 2k) = the rest of the series."""
    for k in (1, 2, 5, 20, 50, 100, 400):
        df = 2 * k
        for x in (1e-8, .5, 3.84, df / 3, .9 * df, df, 1.2 * df, 2 * df, 5 * df, 1400.0):
            log_terms = [-x / 2 + j * math.log(x / 2) - math.lgamma(j + 1) for j in range(k)]
            upper = math.fsum(math.exp(v) for v in log_terms)
            tail_terms = []
            for j in range(k, k + 5000):
                value = math.exp(-x / 2 + j * math.log(x / 2) - math.lgamma(j + 1))
                tail_terms.append(value)
                if j > x and value < 1e-40 * (sum(tail_terms) or 1):
                    break
            lower = math.fsum(tail_terms)
            if upper > 1e-300:
                assert dist.chi2_sf(x, df) == pytest.approx(upper, rel=1e-12, abs=0)
            if lower > 1e-300:
                assert dist.chi2_cdf(x, df) == pytest.approx(lower, rel=1e-12, abs=0)


def test_incomplete_gamma_complements_and_bounds_on_a_pseudo_random_grid():
    generator = np.random.default_rng(20240925)
    shapes = 10.0 ** generator.uniform(-3, 9, 300)
    arguments = shapes * 10.0 ** generator.uniform(-2, .5, 300)
    for a, x in zip(shapes, arguments):
        p, q = dist.gamma_p(a, x), dist.gamma_q(a, x)
        assert 0.0 <= p <= 1.0 and 0.0 <= q <= 1.0
        if min(p, q) > 1e-290:
            assert p + q == pytest.approx(1.0, abs=4e-16)
        assert (dist.gamma_p(a, x), dist.gamma_q(a, x)) == (p, q)        # deterministic


# --------------------------------------------------------------------------------------
# Incomplete beta: closed forms that reach every branch
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("big", [.5, 3, 59.9, 60.1, 999.9, 1000.1, 1e4, 1e6, 1e9, 1e12])
def test_incomplete_beta_with_a_unit_shape_is_a_power(big):
    """I_x(a, 1) = x^a,  I_x(1, b) = 1 - (1-x)^b,  I_x(2, b) = 1 - (1-x)^b (1 + b x).

    Dyadic x keep 1 - x exact, so the mirrored calls see the same point.
    """
    for x in (2 ** -40, 2 ** -20, 2 ** -10, .03125, .125, .25, .5, .75, .96875, 1 - 2 ** -20):
        y = 1 - x
        log_y = math.log1p(-x)
        assert dist.beta_inc(big, 1, x) == pytest.approx(math.exp(big * math.log(x)), rel=1e-12)
        assert dist.beta_inc(1, big, x) == pytest.approx(-math.expm1(big * log_y), rel=1e-12)
        assert dist.beta_inc(1, big, x) == pytest.approx(1 - dist.beta_inc(big, 1, y), rel=1e-12)
        log_rest = big * log_y + math.log1p(big * x)
        assert dist.beta_inc(2, big, x) == pytest.approx(-math.expm1(log_rest), rel=1e-12)
        assert dist.beta_inc(big, 2, y) == pytest.approx(math.exp(log_rest), rel=1e-12)


def test_incomplete_beta_half_half_is_the_arcsine_law():
    for x in (1e-300, 1e-20, 1e-6, .01, .25, .5, .75, .99, 1 - 1e-6):
        assert dist.beta_inc(.5, .5, x) == pytest.approx(2 / math.pi * math.asin(math.sqrt(x)),
                                                         rel=1e-12)


def test_incomplete_beta_symmetry_with_exact_complements():
    """I_x(a, b) = 1 - I_{1-x}(b, a) where 1 - x is exact (dyadic x), across all branches."""
    xs = [.125, .25, .375, .5, .625, .75, .875, 2 ** -10, 1 - 2 ** -10, 2 ** -30, 1 - 2 ** -30]
    shapes = [.05, .5, 2.5, 30, 59.9, 60.1, 300, 999.9, 1000.1, 5e3, 1e5, 1e8]
    for a in shapes:
        for b in shapes:
            for x in xs:
                lower, upper = dist.beta_inc(a, b, x), dist.beta_inc(b, a, 1 - x)
                assert 0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0
                if min(lower, upper) > 1e-290:
                    # Two independent evaluations: continued-fraction rounding is ~1e-15.
                    assert lower + upper == pytest.approx(1.0, abs=4e-15)


@pytest.mark.parametrize("a,b", [(1000.1, 59.9), (59.9, 1000.1), (1000.1, 60.1), (999.9, 59.9),
                                 (1e4, .05), (.05, 1e4), (3e3, 1.5), (1e6, 59.9)])
def test_incomplete_beta_near_the_branch_thresholds_matches_scipy(a, b):
    mean = a / (a + b)
    sd = math.sqrt(mean * (1 - mean) / (a + b + 1))
    points = sorted(x for x in {mean + z * sd for z in (-8, -3, -1, 0, 1, 3, 8)}
                    | {1e-6, .01, .5, .99} if 0 < x < 1)
    close([dist.beta_inc(a, b, x) for x in points], special.betainc(a, b, points), rtol=1e-10)
    close([dist.beta_inc(b, a, 1 - x) for x in points], special.betainc(b, a, 1 - np.array(points)),
          rtol=1e-10)


# --------------------------------------------------------------------------------------
# Student t, chi-square and F: non-integer df, closed forms, brute force
# --------------------------------------------------------------------------------------

def test_student_t_four_df_closed_form():
    """F(t) = 1/2 + (3/8) u (1 - u^2 / 12),  u = t / sqrt(1 + t^2 / 4)."""
    for t in (1e-9, .3, 1.5, 4., 20., 1e3, 1e6):
        u = t / math.sqrt(1 + t * t / 4)
        lower = .5 - 3 / 8 * u * (1 - u * u / 12)
        assert dist.t_cdf(-t, 4) == pytest.approx(lower, rel=1e-11) if lower > 1e-14 else True
        assert dist.t_cdf(t, 4) == pytest.approx(1 - lower, rel=1e-13)
        # Exact upper tail without cancellation: 1/2 - (3/8) u (1 - u^2/12) = (3/8) (2 - u)... via
        # the F(1, 4) relation instead, which is independent of the t code path.
        assert dist.t_sf(t, 4) == pytest.approx(.5 * dist.f_sf(t * t, 1, 4), rel=1e-13)


@pytest.mark.parametrize("df", [.3, 2.5, 7.3, 33.3, 118.9, 2000.5])
def test_student_t_non_integer_df_matches_scipy_and_brute_force(df):
    values = np.array([1e-8, .01, .5, 1.3, 2.7, 5., 9., 15., 25.])
    close([dist.t_sf(v, df) for v in values], stats.t.sf(values, df))
    close([dist.t_cdf(-v, df) for v in values], stats.t.cdf(-values, df))
    constant = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - .5 * math.log(df * math.pi)
    density = lambda t: math.exp(constant - .5 * (df + 1) * math.log1p(t * t / df))  # noqa: E731
    for v in (.5, 2.7, 9.):
        tail, _ = integrate.quad(density, v, math.inf, epsabs=0, epsrel=1e-13, limit=400)
        assert dist.t_sf(v, df) == pytest.approx(tail, rel=1e-10)


@pytest.mark.parametrize("df1,df2", [(3.3, 7.7), (.4, 2.2), (12.5, .9), (40.5, 60.5)])
def test_f_against_brute_force_quadrature(df1, df2):
    a, b = .5 * df1, .5 * df2
    log_norm = a * math.log(df1 / df2) - special.betaln(a, b)
    density = lambda x: math.exp(log_norm + (a - 1) * math.log(x)  # noqa: E731
                                 - (a + b) * math.log1p(df1 * x / df2))
    for x in (.05, .7, 2.5, 9.):
        lower, _ = integrate.quad(density, 0, x, epsabs=0, epsrel=1e-13, limit=400)
        upper, _ = integrate.quad(density, x, math.inf, epsabs=0, epsrel=1e-13, limit=400)
        assert dist.f_cdf(x, df1, df2) == pytest.approx(lower, rel=1e-10)
        assert dist.f_sf(x, df1, df2) == pytest.approx(upper, rel=1e-10)


def test_f_two_and_two_df_closed_form_and_reciprocal_symmetry():
    for x in (1e-300, 1e-20, 1e-3, .5, 1., 7., 1e4, 1e100, 1e300):
        assert dist.f_cdf(x, 2, 2) == pytest.approx(x / (1 + x), rel=1e-13)
        assert dist.f_sf(x, 2, 2) == pytest.approx(1 / (1 + x), rel=1e-13)
    for df1, df2 in ((1, 7), (3.5, 60), (120, 2.5), (1e6, 3), (7, 1e9)):
        for x in (1e-6, .2, 1., 3., 1e3):
            assert dist.f_sf(x, df1, df2) == pytest.approx(dist.f_cdf(1 / x, df2, df1), rel=1e-12)


def test_cdfs_are_monotone_across_branch_switches():
    xs = np.concatenate([np.geomspace(1e-6, 1e6, 400), [0.0]])
    for df in (.5, 1, 2.5, 30, 1999.9, 2000.1, 1e6):
        values = [dist.chi2_cdf(x, df) for x in sorted(xs)]
        assert all(b >= a for a, b in zip(values, values[1:]))
        values = [dist.t_sf(x, df) for x in sorted(xs)]
        assert all(b <= a for a, b in zip(values, values[1:]))
    for df1, df2 in ((1, 1), (.3, 2000.1), (2000.1, .3), (119, 1e9), (121, 1e9), (3000, 3000)):
        values = [dist.f_cdf(x, df1, df2) for x in sorted(xs)]
        assert all(b >= a for a, b in zip(values, values[1:]))
    # The a >= 1000 quadrature switch must not leave a step in df.
    for x in (900., 1990., 2000., 2010., 2200.):
        left, right = dist.chi2_sf(x, 1999.999), dist.chi2_sf(x, 2000.001)
        close([left, right], [stats.chi2.sf(x, 1999.999), stats.chi2.sf(x, 2000.001)])


# --------------------------------------------------------------------------------------
# Quantiles: the contract |ppf(cdf(x)) - x| <= 1e-9 max(1, |x|), round trips, budget
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("df", [.3, 2.5, 7.3, 33.3, 1e4, 1e9])
def test_student_and_chi2_quantiles_recover_x_on_a_fine_grid(df):
    """|ppf(cdf(x)) - x| <= 1e-9 max(1, |x|) on the side where the float tail resolves x.

    On the other side the quantile only sees 1 - p, and a float p next to 1 pins x down no
    better than ~1e-16 / (density (1 - p)); that side is not a statement about the solver.
    """
    def check(x, lower_tail, upper_tail, ppf, isf):
        p, q = lower_tail(x), upper_tail(x)
        tail, inverse = (p, ppf) if p <= q else (q, isf)
        if tail > 1e-300:
            assert abs(inverse(tail) - x) <= 1e-9 * max(1, abs(x))

    for x in np.geomspace(1e-6, 30, 60):
        for value in (x, -x):
            check(value, lambda v: dist.t_cdf(v, df), lambda v: dist.t_sf(v, df),
                  lambda p: dist.t_ppf(p, df), lambda q: dist.t_isf(q, df))
    for x in np.geomspace(1e-6 * max(1, df), 30 * max(1, df), 60):
        check(x, lambda v: dist.chi2_cdf(v, df), lambda v: dist.chi2_sf(v, df),
              lambda p: dist.chi2_ppf(p, df), lambda q: dist.chi2_isf(q, df))


@pytest.mark.parametrize("df1,df2", [(.3, 7.7), (7.7, .3), (2.5, 1e9), (1e9, 2.5), (1, 1),
                                     (119, 121), (2000.5, 3000.5), (3, math.inf), (math.inf, 3)])
def test_f_quantiles_recover_x_and_invert_both_tails(df1, df2):
    for x in np.geomspace(1e-5, 1e5, 50):
        p, q = dist.f_cdf(x, df1, df2), dist.f_sf(x, df1, df2)
        # The smaller tail is the one a float resolves x through (see the t/chi2 test).
        if p <= q and p > 1e-300:
            assert abs(dist.f_ppf(p, df1, df2) - x) <= 1e-9 * max(1, x)
        if q < p and q > 1e-300:
            assert abs(dist.f_isf(q, df1, df2) - x) <= 1e-9 * max(1, x)
    for p in PROBABILITIES:
        for quantile, tail, other in ((dist.f_ppf(p, df1, df2), dist.f_cdf, dist.f_sf),
                                      (dist.f_isf(p, df1, df2), dist.f_sf, dist.f_cdf)):
            assert not math.isnan(quantile) and quantile >= 0
            if 2.3e-308 < quantile < math.inf:
                if p <= .5:
                    assert tail(quantile, df1, df2) == pytest.approx(p, rel=1e-9, abs=0)
                else:
                    assert other(quantile, df1, df2) == pytest.approx(1 - p, rel=1e-9, abs=0)


def test_quantiles_next_to_the_median_are_finite_small_and_cheap():
    calls = 0
    solve = dist._solve_log_newton

    def counting(residual, start, increasing):
        def wrapped(x):
            nonlocal calls
            calls += 1
            return residual(x)
        return solve(wrapped, start, increasing)

    dist._solve_log_newton = counting
    try:
        for df in (.1, 1, 3, 30, 1e6):
            median = dist.chi2_ppf(.5, df)
            for k in (1, 2, 5):
                for p in (.5 - k * 2 ** -53, .5 + k * 2 ** -53):
                    calls = 0
                    t = dist.t_ppf(p, df)
                    assert math.isfinite(t) and abs(t) < 1e-9 and calls <= 60
                    assert math.copysign(1.0, t) == math.copysign(1.0, p - .5) or t == 0.0
                    calls = 0
                    x = dist.chi2_ppf(p, df)
                    assert abs(x - median) <= 1e-9 * max(1, median) and calls <= 60
                    calls = 0
                    f = dist.f_ppf(p, 4, df)
                    assert math.isfinite(f) and f > 0 and calls <= 60
    finally:
        dist._solve_log_newton = solve


def test_quantile_call_budget_over_a_wide_grid():
    calls = 0
    solve = dist._solve_log_newton

    def counting(residual, start, increasing):
        def wrapped(x):
            nonlocal calls
            calls += 1
            return residual(x)
        return solve(wrapped, start, increasing)

    dist._solve_log_newton = counting
    try:
        worst = 0
        probabilities = (1e-300, 1e-50, 1e-6, .01, .25, .5, .75, .99, 1 - 1e-6, 1 - 1e-13)
        for df in (1e-3, .3, 1, 2.5, 30, 999, 1001, 1e6, 1e12):
            for p in probabilities:
                for quantile in (lambda: dist.chi2_ppf(p, df), lambda: dist.chi2_isf(p, df),
                                 lambda: dist.t_ppf(p, df)):
                    calls = 0
                    quantile()
                    worst = max(worst, calls)
        for df1 in (1e-3, 1, 2.5, 120, 2000, 1e9, math.inf):
            for df2 in (1e-3, 1, 2.5, 120, 2000, 1e9, math.inf):
                if math.isinf(df1) and math.isinf(df2):
                    continue
                for p in probabilities:
                    for quantile in (lambda: dist.f_ppf(p, df1, df2), lambda: dist.f_isf(p, df1, df2)):
                        calls = 0
                        quantile()
                        worst = max(worst, calls)
    finally:
        dist._solve_log_newton = solve
    assert worst <= 40


# --------------------------------------------------------------------------------------
# Limits, invalid arguments, argument shapes, determinism, non-mutation, speed
# --------------------------------------------------------------------------------------

def test_infinite_statistics_and_infinite_df_limits():
    for df in (.3, 4, 1e9, math.inf):
        assert dist.t_cdf(math.inf, df) == 1.0 and dist.t_sf(math.inf, df) == 0.0
        assert dist.t_cdf(-math.inf, df) == 0.0 and dist.t_sf(-math.inf, df) == 1.0
        assert dist.p_value(math.inf, "t", df) == 0.0 == dist.p_value(-math.inf, "t", df)
        assert dist.p_value(-math.inf, "t", df, two_sided=False) == 1.0
    assert dist.p_value(math.inf, "chi2", 7) == 0.0 and dist.p_value(-math.inf, "chi2", 7) == 1.0
    for df1, df2 in ((3, 9), (3, math.inf), (math.inf, 9)):
        assert dist.f_cdf(math.inf, df1, df2) == 1.0 and dist.f_sf(math.inf, df1, df2) == 0.0
        assert dist.f_cdf(-math.inf, df1, df2) == 0.0 and dist.f_sf(-math.inf, df1, df2) == 1.0
        assert dist.p_value(math.inf, "F", df1, df2) == 0.0
    # t with infinite df is exactly the normal, including quantiles and criticals.
    for x in (-3., -1e-6, 0., .4, 2.2, 9.):
        assert dist.t_cdf(x, math.inf) == dist.normal_cdf(x)
        assert dist.t_sf(x, math.inf) == dist.normal_sf(x)
    for p in (1e-300, .001, .3, .5, .7, .999):
        assert dist.t_ppf(p, math.inf) == dist.normal_ppf(p)
        assert dist.critical(p, "t", math.inf) == dist.critical(p, "normal")
    # F with one infinite df is the scaled chi-square, including quantiles.
    for p in (1e-20, .05, .5, .95):
        assert dist.f_ppf(p, 4, math.inf) == pytest.approx(dist.chi2_ppf(p, 4) / 4, rel=1e-12)
        assert dist.f_isf(p, math.inf, 4) == pytest.approx(4 / dist.chi2_ppf(p, 4), rel=1e-12)
    # Negative zero and negative statistics for one-sided families.
    assert dist.chi2_cdf(-0.0, 3) == 0.0 and dist.chi2_sf(-0.0, 3) == 1.0
    assert dist.f_cdf(-0.0, 3, 4) == 0.0 and dist.gamma_p(2, -0.0) == 0.0
    assert dist.t_cdf(-0.0, 3) == .5 == dist.t_sf(-0.0, 3) and dist.t_ppf(.5, 3) == 0.0


@pytest.mark.parametrize("call", [
    lambda: dist.t_cdf(1, math.nan), lambda: dist.t_ppf(math.nan, 3), lambda: dist.t_ppf(.5, -math.inf),
    lambda: dist.chi2_ppf(.5, math.inf), lambda: dist.chi2_isf(.5, math.nan),
    lambda: dist.f_ppf(.5, math.inf, math.inf), lambda: dist.f_isf(.5, math.inf, math.inf),
    lambda: dist.f_cdf(1, math.nan, 3), lambda: dist.f_sf(1, 3, math.nan),
    lambda: dist.gamma_p(1, torch.tensor([1.0, 2.0])), lambda: dist.beta_inc(1, 1, np.array([.5, .6])),
    lambda: dist.p_value(torch.tensor([1.0, 2.0]), "normal"), lambda: dist.p_value(1, "t", df=None),
    lambda: dist.p_value(1, "F", 3, df2=None), lambda: dist.p_value(1, "", 3),
    lambda: dist.p_value(1, b"t", 3), lambda: dist.critical(.05, "T "), lambda: dist.critical(.05, "F", 3),
    lambda: dist.critical(1 + 1e-15, "normal"), lambda: dist.critical(math.nan, "normal"),
    lambda: dist.normal_ppf(torch.tensor(float("nan"))), lambda: dist.normal_cdf(None),
    lambda: dist.normal_cdf([1.0]), lambda: dist.chi2_sf({"x": 1}, 2),
])
def test_more_invalid_arguments_raise_kernel_errors(call):
    with pytest.raises(KernelError) as error:
        call()
    assert error.value.code == "invalid_inference"


def test_scalar_arguments_accept_zero_dim_and_one_element_tensors_without_mutation():
    statistic = torch.tensor(2.3, dtype=torch.float64)
    df = torch.tensor([17], dtype=torch.int64)
    column = torch.tensor([[2.3]], dtype=torch.float32)       # float32 is read as its own value
    alpha = torch.tensor(.05, dtype=torch.float64)
    before = (statistic.clone(), df.clone(), column.clone())
    assert dist.t_sf(statistic, df) == dist.t_sf(2.3, 17)
    assert dist.t_sf(column, df) == dist.t_sf(float(column), 17) != dist.t_sf(2.3, 17)
    assert dist.p_value(statistic, "t", df) == dist.p_value(2.3, "t", 17)
    assert dist.critical(alpha, "F", df, df) == dist.critical(.05, "F", 17, 17)
    assert dist.chi2_sf(np.float32(7.8), np.int64(3)) == dist.chi2_sf(float(np.float32(7.8)), 3)
    assert torch.equal(statistic, before[0]) and torch.equal(df, before[1])
    assert torch.equal(column, before[2])
    assert all(isinstance(v, float) for v in (dist.t_sf(statistic, df), dist.p_value(statistic, "t", df),
                                              dist.critical(alpha, "t", df)))
    # Bools are numbers in Python; keep the behaviour explicit so it does not drift silently.
    assert dist.chi2_sf(True, 2) == dist.chi2_sf(1, 2)


def test_results_are_deterministic_and_quadrature_outputs_are_fresh_tensors():
    first = [dist.t_sf(2.3, 17), dist.chi2_isf(1e-7, 2001), dist.f_ppf(.3, 119, 1e9),
             dist.gamma_q(1e6, 1e6 + 3e3), dist.beta_inc(60.1, 1000.1, .06), dist.normal_ppf(1e-200)]
    second = [dist.t_sf(2.3, 17), dist.chi2_isf(1e-7, 2001), dist.f_ppf(.3, 119, 1e9),
              dist.gamma_q(1e6, 1e6 + 3e3), dist.beta_inc(60.1, 1000.1, .06), dist.normal_ppf(1e-200)]
    assert first == second
    nodes, weights = dist.gauss_legendre(40)
    again_nodes, again_weights = dist.gauss_legendre(40)
    assert torch.equal(nodes, again_nodes) and torch.equal(weights, again_weights)
    assert nodes.data_ptr() != again_nodes.data_ptr() and weights.data_ptr() != again_weights.data_ptr()
    assert nodes.is_contiguous() and nodes.device.type == "cpu" and not nodes.requires_grad
    assert nodes.dtype == weights.dtype == torch.float64


def test_quadrature_small_orders_match_closed_forms_and_odd_orders_have_an_exact_zero():
    nodes, weights = dist.gauss_hermite(2)
    close(nodes, [-1 / math.sqrt(2), 1 / math.sqrt(2)], rtol=1e-15)
    close(weights, [math.sqrt(math.pi) / 2] * 2, rtol=1e-15)
    nodes, weights = dist.gauss_hermite(3)
    close(nodes, [-math.sqrt(1.5), 0, math.sqrt(1.5)], rtol=1e-15, atol=1e-16)
    close(weights, [math.sqrt(math.pi) / 6, 2 * math.sqrt(math.pi) / 3, math.sqrt(math.pi) / 6],
          rtol=1e-15)
    nodes, weights = dist.gauss_legendre(3)
    close(nodes, [-math.sqrt(.6), 0, math.sqrt(.6)], rtol=1e-15, atol=1e-16)
    close(weights, [5 / 9, 8 / 9, 5 / 9], rtol=1e-15)
    for n in (7, 51, 199):
        for rule in (dist.gauss_hermite, dist.gauss_legendre):
            nodes, weights = rule(n)
            assert float(nodes[n // 2]) == 0.0 and torch.equal(nodes, -nodes.flip(0))
            assert bool((nodes[1:] > nodes[:-1]).all()) and bool((weights > 0).all())
    # Exact integrals of non-polynomials converge: int e^-x^2 cos x = sqrt(pi) e^-1/4 and
    # int_-1^1 dx / (1 + x^2) = pi / 2.
    nodes, weights = dist.gauss_hermite(40)
    assert float((weights * torch.cos(nodes)).sum()) == pytest.approx(
        math.sqrt(math.pi) * math.exp(-.25), rel=1e-15)
    nodes, weights = dist.gauss_legendre(60)
    assert float((weights / (1 + nodes * nodes)).sum()) == pytest.approx(math.pi / 2, rel=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 9, 16, 17, 33, 100, 127, 128, 129, 199, 200])
def test_quadrature_matches_numpy_for_awkward_orders(n):
    nodes, weights = dist.gauss_hermite(n)
    expected_nodes, expected_weights = np.polynomial.hermite.hermgauss(n)
    np.testing.assert_allclose(nodes.numpy(), expected_nodes, rtol=0, atol=2e-14 * max(1, n ** .5))
    np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=5e-12, atol=0)
    nodes, weights = dist.gauss_legendre(n)
    expected_nodes, expected_weights = np.polynomial.legendre.leggauss(n)
    np.testing.assert_allclose(nodes.numpy(), expected_nodes, rtol=0, atol=2e-14)
    np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=1e-10, atol=0)
    assert float(weights.sum()) == pytest.approx(2.0, rel=1e-15)


def test_large_quadrature_orders_are_fast_finite_and_exact_for_moments():
    dist._gauss_hermite.cache_clear()
    dist._gauss_legendre.cache_clear()
    start = time.perf_counter()
    nodes, weights = dist.gauss_hermite(1000)
    legendre_nodes, legendre_weights = dist.gauss_legendre(1000)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, elapsed
    assert bool(torch.isfinite(nodes).all()) and bool((weights >= 0).all())
    assert bool((nodes[1:] > nodes[:-1]).all()) and bool((legendre_nodes[1:] > legendre_nodes[:-1]).all())
    assert bool((legendre_weights > 0).all())
    for k in (0, 1, 2, 5, 20):
        assert float((weights * nodes ** (2 * k)).sum()) == pytest.approx(math.gamma(k + .5), rel=1e-12)
        assert float((legendre_weights * legendre_nodes ** (2 * k)).sum()) == pytest.approx(
            2 / (2 * k + 1), rel=1e-12)
    # Beyond n ~ 380 the extreme Hermite weights e^-x^2 underflow to exactly zero; this is a
    # float64 limit of the physicists' weight, and the rule stays usable for expectations.
    assert float(weights.min()) == 0.0 and float(weights.sum()) == pytest.approx(math.sqrt(math.pi))
    # A cached call is a clone: microseconds, not milliseconds.
    start = time.perf_counter()
    for _ in range(100):
        dist.gauss_hermite(1000)
    assert time.perf_counter() - start < .2


def test_many_p_values_are_cheap_enough_for_estimator_tables():
    """20,000 mixed p-values and 2,000 critical values in well under a few seconds."""
    generator = np.random.default_rng(7)
    statistics = generator.standard_normal(20000) * 3
    dfs = 10.0 ** generator.uniform(0, 7, 20000)
    start = time.perf_counter()
    for i, (s, df) in enumerate(zip(statistics, dfs)):
        if i % 3 == 0:
            dist.p_value(s, "t", df)
        elif i % 3 == 1:
            dist.p_value(s * s, "chi2", 1 + (i % 7))
        else:
            dist.p_value(s * s, "F", 1 + (i % 5), df)
    for df in dfs[:2000]:
        dist.critical(.05, "t", df)
    assert time.perf_counter() - start < 5.0
