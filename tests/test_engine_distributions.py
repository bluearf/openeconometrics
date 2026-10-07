"""Independent oracles for the reference-distribution kernel.

SciPy is the primary oracle. Where SciPy itself loses accuracy (documented next to each
case) the oracle is a closed form or an mpmath high-precision evaluation instead.
"""

import math
import warnings

import numpy as np
import pytest
import torch
from scipy import special, stats

from openecon.engines import distributions as dist
from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value, student_t_two_sided

RTOL = 1e-10
# Results below 1e-310 are subnormal and cannot carry ten digits.
ATOL = 1e-310
PROBABILITIES = [1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3, .01, .05, .2, .4999,
                 .5, .7, .9, .99, 1 - 1e-6, 1 - 1e-10, 1 - 1e-13, 1 - 1e-16]


def close(actual, expected, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=rtol, atol=atol)


def mp_tail(log_density, slope, curvature, start, sign, limit):
    """30-digit tail mass of a log-concave density by panelled mpmath quadrature."""
    mp = pytest.importorskip("mpmath")
    left, total = mp.mpf(start), mp.mpf(0)
    for _ in range(400):
        g1, g2 = float(sign * slope(left)), float(curvature(left))
        root = math.sqrt(g1 * g1 + 10 * abs(g2))
        step = (g1 + root) / abs(g2) if g1 > 0 else 10 / (root - g1)
        right = left + sign * mp.mpf(step)
        last = right >= limit if sign > 0 else right <= limit
        if last:
            right = mp.mpf(limit)
        part = abs(mp.quad(lambda t: mp.exp(log_density(t)), mp.linspace(left, right, 9)))
        total += part
        if last or part < total * mp.mpf("1e-32"):
            return total
        left = right
    raise AssertionError("oracle did not converge")


# --------------------------------------------------------------------------------------
# Incomplete gamma
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("a", [1e-3, .05, .3, .5, .9, 1, 1.5, 2.5, 7, 20, 50, 150, 400, 999,
                               1000, 1500, 5000, 1e5, 1e7, 5e8])
def test_incomplete_gamma_matches_scipy(a):
    points = {a * m for m in (1e-12, 1e-6, 1e-3, .01, .1, .5, .9, 1.05, 1.2, 2, 3, 10, 50, 300)}
    points |= {1e-12, 1e-6, 1e-3, .01, .1, .5, .9, 1.05, 1.2, 2, 3, 10, 50, 300, 700}
    points |= {a + z * math.sqrt(a) for z in (-4, -3, -1.5, -.3, -1e-3, 0, 1e-3, .3, 1, 3, 4)}
    # Cephes stops its series/fraction after 2000 terms, so for a > 5000 SciPy is only an
    # oracle within 4 standard deviations of the mean (its Temme branch); the far tails
    # are checked against mpmath below.
    xs = np.array(sorted(x for x in points
                         if x > 0 and (a <= 5000 or abs(x - a) <= 4 * math.sqrt(a))))
    close([dist.gamma_p(a, x) for x in xs], special.gammainc(a, xs))
    close([dist.gamma_q(a, x) for x in xs], special.gammaincc(a, xs))


@pytest.mark.parametrize("a", [1e5, 1e7, 5e8, 1e12])
def test_large_shape_gamma_tails_match_high_precision_quadrature(a):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(40):
        shape = mp.mpf(a)
        log_gamma = mp.loggamma(shape)

        def log_density(t):
            return (shape - 1) * mp.log(t) - t - log_gamma

        def slope(t):
            return (shape - 1) / t - 1

        def curvature(t):
            return -(shape - 1) / (t * t)

        for z in (-30, -6, 6, 37):
            x = a + z * math.sqrt(a)
            if z < 0:
                expected = float(mp_tail(log_density, slope, curvature, x, -1, 0))
                assert dist.gamma_p(a, x) == pytest.approx(expected, rel=5e-12, abs=0)
                assert dist.gamma_q(a, x) == pytest.approx(1 - expected, rel=1e-15)
            else:
                expected = float(mp_tail(log_density, slope, curvature, x, 1, mp.inf))
                assert dist.gamma_q(a, x) == pytest.approx(expected, rel=5e-12, abs=0)
                assert dist.gamma_p(a, x) == pytest.approx(1 - expected, rel=1e-15)


@pytest.mark.parametrize("a", [1e-12, 1e-8, 1e-5, 1e-3, .2])
def test_small_shape_upper_gamma_does_not_cancel(a):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(40):
        for x in (1e-200, 1e-30, 1e-8, 1e-3, .2, .56, .9, 1.09, 1.11, 2.5, 30):
            upper = mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True)
            lower = mp.gammainc(mp.mpf(a), 0, mp.mpf(x), regularized=True)
            assert dist.gamma_q(a, x) == pytest.approx(float(upper), rel=1e-12, abs=0)
            assert dist.gamma_p(a, x) == pytest.approx(float(lower), rel=1e-12, abs=0)


def test_incomplete_gamma_limits_and_complement():
    assert dist.gamma_p(2.5, 0) == 0.0 and dist.gamma_q(2.5, 0) == 1.0
    assert dist.gamma_p(2.5, math.inf) == 1.0 and dist.gamma_q(2.5, math.inf) == 0.0
    assert dist.gamma_q(1.5, 1000) == 0.0                      # e^-1000 underflows
    assert dist.gamma_p(3, 1e-200) == 0.0
    for a, x in [(.3, .2), (4, 2), (4, 9), (2000, 1990), (3e6, 3.001e6)]:
        assert dist.gamma_p(a, x) + dist.gamma_q(a, x) == pytest.approx(1.0, abs=4e-16)


# --------------------------------------------------------------------------------------
# Incomplete beta
# --------------------------------------------------------------------------------------

BETA_SHAPES = [.05, .5, 1, 2.5, 10, 58.5, 300, 999, 1000, 5000, 4e4]


@pytest.mark.parametrize("a", BETA_SHAPES)
@pytest.mark.parametrize("b", BETA_SHAPES)
def test_incomplete_beta_matches_scipy(a, b):
    mean = a / (a + b)
    sd = math.sqrt(mean * (1 - mean) / (a + b + 1))
    points = {mean + z * sd for z in (-30, -12, -6, -3, -1, -.2, 0, .2, 1, 3, 6, 12, 30)}
    points |= {1e-300, 1e-100, 1e-12, 1e-6, 1e-3, .03, .3, .5, .8, .97, 1 - 1e-6, 1 - 1e-12}
    for odds in (1e-12, 1e-6, 1e-3, .03, .3, 3, 30, 1e3):
        points.add(odds * a / (odds * a + b))
    xs = np.array(sorted(x for x in points if 0 < x < 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = special.betainc(a, b, xs)
    close([dist.beta_inc(a, b, x) for x in xs], expected)


@pytest.mark.parametrize("a,b", [(.5, 5e8), (2.5, 5e8), (5, 5e8), (20, 5e8), (50, 5e11),
                                 (59.5, 1e6), (5, 1e3), (.5, 1e12)])
def test_one_large_shape_incomplete_beta_matches_mpmath_series(a, b):
    """Boost (SciPy) is only ~1e-8 accurate here; the 2F1 series in x is exact for mpmath."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(350):                  # 1 - lower must resolve tails down to 1e-300
        for multiple in (1e-6, .01, .3, 1, 2.5, 6, 20, 60):
            x = multiple * a / b
            lower = mp.betainc(mp.mpf(a), mp.mpf(b), 0, mp.mpf(x), regularized=True)
            upper = 1 - lower
            # Both orientations: (a, b, x) and its mirror image (b, a, 1 - x).
            for p, q in (dist._beta_core(a, b, x, 1 - x)[:2],
                         dist._beta_core(b, a, 1 - x, x)[1::-1]):
                assert p == pytest.approx(float(lower), rel=5e-12, abs=ATOL)
                assert q == pytest.approx(float(upper), rel=5e-12, abs=ATOL)


@pytest.mark.parametrize("a,b", [(1e5, 1e5), (5e8, 5e8), (1e6, 1e3), (60, 1e6), (5e3, 5e7)])
def test_both_large_shapes_incomplete_beta_matches_high_precision_quadrature(a, b):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(40):
        s1, s2 = mp.mpf(a), mp.mpf(b)
        log_beta = mp.loggamma(s1) + mp.loggamma(s2) - mp.loggamma(s1 + s2)
        mean = s1 / (s1 + s2)
        sd = mp.sqrt(mean * (1 - mean) / (s1 + s2 + 1))

        def tail(first, second, start):
            def log_density(t):
                return (first - 1) * mp.log(t) + (second - 1) * mp.log1p(-t) - log_beta

            def slope(t):
                return (first - 1) / t - (second - 1) / (1 - t)

            def curvature(t):
                return -(first - 1) / (t * t) - (second - 1) / ((1 - t) ** 2)

            return mp_tail(log_density, slope, curvature, start, -1, 0)

        for z in (-25, -4, 2, 30):
            exact = mean + z * sd
            if not 0 < exact < 1:
                continue
            # The side nearer to zero is the exactly representable input.
            if exact < 0.5:
                x = float(exact)
                y = float(1 - mp.mpf(x))
                exact_x, exact_y = mp.mpf(x), 1 - mp.mpf(x)
            else:
                y = float(1 - exact)
                x = float(1 - mp.mpf(y))
                exact_x, exact_y = 1 - mp.mpf(y), mp.mpf(y)
            p, q, _ = dist._beta_core(a, b, x, y)
            if z < 0:
                assert p == pytest.approx(float(tail(s1, s2, exact_x)), rel=5e-12, abs=0)
            else:
                assert q == pytest.approx(float(tail(s2, s1, exact_y)), rel=5e-12, abs=0)
            assert p + q == pytest.approx(1.0, abs=4e-16)


def test_incomplete_beta_limits():
    assert dist.beta_inc(2, 3, 0) == 0.0 and dist.beta_inc(2, 3, 1) == 1.0
    assert dist.beta_inc(1, 1, .25) == pytest.approx(.25, rel=1e-15)
    assert dist.beta_inc(2, 1, .5) == pytest.approx(.25, rel=1e-15)
    assert dist.beta_inc(.5, .5, .5) == pytest.approx(.5, rel=1e-15)


# --------------------------------------------------------------------------------------
# Normal
# --------------------------------------------------------------------------------------

def test_normal_cdf_sf_pdf_match_scipy():
    xs = np.concatenate([np.linspace(-38, 38, 305), [-1e-300, 0., 1e-300, 1e-9, -1e-9]])
    close([dist.normal_cdf(x) for x in xs], stats.norm.cdf(xs), rtol=1e-12)
    close([dist.normal_sf(x) for x in xs], stats.norm.sf(xs), rtol=1e-12)
    close([dist.normal_pdf(x) for x in xs], stats.norm.pdf(xs), rtol=1e-12)
    # 1 - Phi(38) = 2.885428351e-316 is subnormal (SciPy's sf underflows to zero there).
    assert dist.normal_sf(38) == pytest.approx(2.885428351e-316, rel=1e-7, abs=0)
    assert dist.normal_cdf(-38) == dist.normal_sf(38)
    assert dist.normal_sf(37) == pytest.approx(stats.norm.sf(37), rel=1e-12, abs=0)
    assert dist.normal_cdf(-math.inf) == 0.0 and dist.normal_cdf(math.inf) == 1.0
    assert dist.normal_sf(-math.inf) == 1.0 and dist.normal_sf(math.inf) == 0.0
    assert dist.normal_pdf(math.inf) == 0.0


def test_normal_quantiles_match_scipy_and_invert():
    probabilities = np.array(PROBABILITIES + list(np.linspace(.001, .999, 333))
                             + [10.0 ** -k for k in range(1, 300, 7)])
    close([dist.normal_ppf(p) for p in probabilities], stats.norm.ppf(probabilities),
          rtol=1e-12, atol=1e-15)
    close([dist.normal_isf(p) for p in probabilities], stats.norm.isf(probabilities),
          rtol=1e-12, atol=1e-15)
    for p in probabilities:
        if p <= .5:
            assert dist.normal_cdf(dist.normal_ppf(p)) == pytest.approx(p, rel=1e-12, abs=0)
            assert dist.normal_sf(dist.normal_isf(p)) == pytest.approx(p, rel=1e-12, abs=0)
    # The cdf resolves x only where it is not rounded towards 1, hence x <= 1/2.
    for x in np.linspace(-37, .5, 76):
        assert abs(dist.normal_ppf(dist.normal_cdf(x)) - x) <= 1e-9 * max(1, abs(x))
        assert abs(dist.normal_isf(dist.normal_sf(-x)) + x) <= 1e-9 * max(1, abs(x))
    assert dist.normal_ppf(.5) == 0.0
    # Subnormal probabilities (below the 1e-300 contract) keep the starting approximation.
    assert dist.normal_ppf(5e-324) == pytest.approx(stats.norm.ppf(5e-324), rel=1e-8)
    assert dist.normal_ppf(1e-305) == pytest.approx(stats.norm.ppf(1e-305), rel=1e-12)


# --------------------------------------------------------------------------------------
# Student t
# --------------------------------------------------------------------------------------

T_VALUES = [0., 1e-300, 1e-12, 1e-8, 1e-4, .01, .1, .5, .9, 1., 1.01, 1.5, 1.96, 2., 3., 5.,
            8., 12., 20., 30., 37., 40., 100., 1e3, 1e5, 1e10]


@pytest.mark.parametrize("df", [.1, .5, 1, 2, 3, 5, 11, 30, 117, 1000, 1999, 2000, 2001, 5000,
                                1e5, 1e6, 1e9, 1e12])
def test_student_t_matches_scipy(df):
    values = np.array(T_VALUES + [-v for v in T_VALUES])
    cdf, sf = stats.t.cdf(values, df), stats.t.sf(values, df)
    if df == 1:
        # SciPy's t(1) loses ~3e-9 just next to zero; the Cauchy law is exact.
        cdf, sf = np.arctan2(1, -values) / math.pi, np.arctan2(1, values) / math.pi
    close([dist.t_cdf(v, df) for v in values], cdf)
    close([dist.t_sf(v, df) for v in values], sf)
    close([dist.t_cdf(-v, df) for v in values], [dist.t_sf(v, df) for v in values], rtol=0)


def test_student_t_closed_forms_and_limits():
    for x in (1e-9, .3, 1., 7., 40., 1e4, 1e9, 1e150, 1e200, 1e300):
        root = math.sqrt(2 + x * x) if x < 1e150 else x
        assert dist.t_sf(x, 2) == pytest.approx(1 / (root * (root + x)), rel=RTOL, abs=0)
        assert dist.t_cdf(-x, 2) == pytest.approx(1 / (root * (root + x)), rel=RTOL, abs=0)
        assert dist.t_sf(x, 1) == pytest.approx(math.atan2(1, x) / math.pi, rel=RTOL, abs=0)
    # Heavy tails where even x^2 overflows: P(T > x) = x^-df df^(df/2) / (df B(df/2, 1/2)).
    for df in (1e-3, .1, .7):
        for x in (1e160, 1e300):
            log_tail = (-df * math.log(x) + .5 * df * math.log(df) - math.log(df)
                        - special.betaln(.5 * df, .5))
            assert dist.t_sf(x, df) == pytest.approx(math.exp(log_tail), rel=1e-9)
            assert dist.t_cdf(x, df) == pytest.approx(1 - math.exp(log_tail), rel=1e-12)
    assert dist.t_sf(40, 3) == pytest.approx(stats.t.sf(40, 3), rel=RTOL)
    assert 0 < dist.t_sf(8.3, 117) == pytest.approx(stats.t.sf(8.3, 117), rel=1e-12)
    values = np.linspace(-38, 38, 77)
    close([dist.t_cdf(v, math.inf) for v in values], stats.norm.cdf(values), rtol=1e-12)
    close([dist.t_sf(v, math.inf) for v in values], stats.norm.sf(values), rtol=1e-12)
    for df in (.2, 3, 1e7):
        assert dist.t_cdf(0, df) == .5 == dist.t_sf(0, df)
        assert dist.t_cdf(math.inf, df) == 1.0 and dist.t_sf(math.inf, df) == 0.0
        assert dist.t_cdf(-math.inf, df) == 0.0 and dist.t_sf(-math.inf, df) == 1.0


@pytest.mark.parametrize("df", [1, 5, 117, 1e5, 1e6, 1e9, 1e12])
def test_student_t_agrees_with_existing_inference_kernel(df):
    for value in (0., 1e-8, .9, 1.96, 5., 8., 20., 37.):
        expected = student_t_two_sided(value, df)
        assert dist.p_value(value, "t", df) == pytest.approx(expected, rel=1e-10, abs=0)
        assert dist.p_value(-value, "t", df) == pytest.approx(expected, rel=1e-10, abs=0)
    for alpha in (.5, .05, 1e-4, 1e-12):
        assert dist.critical(alpha, "t", df) == pytest.approx(critical_value(alpha, df), rel=1e-9)
    assert dist.critical(.05, "normal") == pytest.approx(critical_value(.05), rel=1e-13)


# --------------------------------------------------------------------------------------
# Chi-square
# --------------------------------------------------------------------------------------

CHI2_VALUES = [0., 1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 1e-3, .01, .1, .5, 1., 2., 3.84, 5., 10.,
               20., 50., 100., 200., 500., 1000., 1400., 2000., 5000., 1e4, 1e6]


@pytest.mark.parametrize("df", [1e-3, .1, .5, 1, 2, 3, 4, 7, 10, 30, 100, 500, 1000, 1999, 2000,
                                2001, 4000])
def test_chi2_matches_scipy(df):
    values = np.array(CHI2_VALUES + [df * m for m in (.2, .5, .8, .95, 1, 1.05, 1.3, 2, 4)])
    close([dist.chi2_cdf(v, df) for v in values], stats.chi2.cdf(values, df))
    close([dist.chi2_sf(v, df) for v in values], stats.chi2.sf(values, df))


@pytest.mark.parametrize("df", [1e5, 1e6, 1e9])
def test_large_df_chi2_matches_scipy_near_the_mean(df):
    values = np.array([df + z * math.sqrt(2 * df) for z in np.linspace(-4, 4, 33)])
    close([dist.chi2_cdf(v, df) for v in values], stats.chi2.cdf(values, df))
    close([dist.chi2_sf(v, df) for v in values], stats.chi2.sf(values, df))


def test_chi2_closed_forms_tails_and_limits():
    for x in (1e-12, .3, 3.84, 50., 700., 1400.):
        assert dist.chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12, abs=0)
        assert dist.chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2)), rel=1e-12, abs=0)
        assert dist.chi2_sf(x, 4) == pytest.approx(math.exp(-x / 2) * (1 + x / 2), rel=1e-12)
    assert dist.chi2_cdf(1e-12, 2) == pytest.approx(-math.expm1(-5e-13), rel=1e-12)
    assert dist.chi2_sf(2000, 3) == stats.chi2.sf(2000, 3) == 0.0      # e^-1000 underflows
    assert 0 < dist.chi2_sf(1400, 3) == pytest.approx(stats.chi2.sf(1400, 3), rel=RTOL)
    assert dist.chi2_sf(-1, 3) == 1.0 and dist.chi2_cdf(-1, 3) == 0.0
    assert dist.chi2_sf(0, 3) == 1.0 and dist.chi2_sf(math.inf, 3) == 0.0
    assert dist.chi2_cdf(math.inf, 3) == 1.0


# --------------------------------------------------------------------------------------
# F
# --------------------------------------------------------------------------------------

F_VALUES = [0., 1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 1e-3, .01, .1, .5, .9, 1., 1.1, 1.5, 2., 3.,
            5., 10., 30., 100., 1e3, 1e4, 1e6, 1e10, 1e30, 1e100, 1e300]


@pytest.mark.parametrize("df1", [.1, 1, 2, 3, 5, 10, 40, 100, 1000, 1e4, 1e5])
@pytest.mark.parametrize("df2", [.1, 1, 2, 5, 10, 30, 117, 1000, 1999, 2000, 2001, 1e5])
def test_f_matches_scipy(df1, df2):
    values = np.array(F_VALUES)
    close([dist.f_cdf(v, df1, df2) for v in values], stats.f.cdf(values, df1, df2))
    close([dist.f_sf(v, df1, df2) for v in values], stats.f.sf(values, df1, df2))


@pytest.mark.parametrize("df", [.1, 3, 30, 1e3, 1e6, 1e9, 1e12])
def test_f_closed_forms_with_two_degrees_of_freedom(df):
    """F(2, d): sf = (1 + 2x/d)^(-d/2);  F(d, 2): cdf = (1 + 2/(d x))^(-d/2)."""
    for x in (1e-8, .01, .5, 1., 3., 20., 300., 1e4, 1e6):
        upper = math.exp(-.5 * df * math.log1p(2 * x / df))
        assert dist.f_sf(x, 2, df) == pytest.approx(upper, rel=RTOL, abs=ATOL)
        assert dist.f_cdf(x, 2, df) == pytest.approx(-math.expm1(-.5 * df * math.log1p(2 * x / df)),
                                                    rel=RTOL, abs=ATOL)
        lower = math.exp(-.5 * df * math.log1p(2 / (df * x)))
        assert dist.f_cdf(x, df, 2) == pytest.approx(lower, rel=RTOL, abs=ATOL)
        assert dist.f_sf(x, df, 2) == pytest.approx(
            -math.expm1(-.5 * df * math.log1p(2 / (df * x))), rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("df1,df2", [(1, 1e9), (5, 1e9), (10, 1e9), (40, 1e9), (100, 1e12),
                                     (119, 2e6), (7, 3000)])
def test_large_denominator_f_matches_mpmath(df1, df2):
    """SciPy/Boost is ~1e-8 accurate for df2 ~ 1e9; mpmath's series in w is exact."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(350):                  # 1 - lower must resolve tails down to 1e-300
        for x in (.01, .3, 1, 1.5, 3, 8, 30):
            w = mp.mpf(df1) * x / (mp.mpf(df1) * x + mp.mpf(df2))
            lower = mp.betainc(mp.mpf(df1) / 2, mp.mpf(df2) / 2, 0, w, regularized=True)
            upper = 1 - lower
            if lower > mp.mpf("1e-300"):
                assert dist.f_cdf(x, df1, df2) == pytest.approx(float(lower), rel=5e-12, abs=0)
                assert dist.f_sf(1 / x, df2, df1) == pytest.approx(float(lower), rel=5e-12, abs=0)
            if upper > mp.mpf("1e-300"):
                assert dist.f_sf(x, df1, df2) == pytest.approx(float(upper), rel=5e-12, abs=0)
                assert dist.f_cdf(1 / x, df2, df1) == pytest.approx(float(upper), rel=5e-12, abs=0)


def test_f_tails_limits_and_relations():
    assert dist.f_sf(1e6, 2, 5) == pytest.approx(stats.f.sf(1e6, 2, 5), rel=RTOL)
    assert dist.f_sf(1e6, 2, 5) == pytest.approx((1 + 2e6 / 5) ** -2.5, rel=1e-12)
    # F(1, 1): cdf = (2/pi) atan(sqrt(x)); SciPy rounds the sf at 1e-20 to exactly 1.
    for x in (1e-20, 1e-6, .3, 1., 50., 1e12, 1e40):
        assert dist.f_cdf(x, 1, 1) == pytest.approx(2 / math.pi * math.atan(math.sqrt(x)), rel=RTOL)
        assert dist.f_sf(x, 1, 1) == pytest.approx(2 / math.pi * math.atan(1 / math.sqrt(x)),
                                                 rel=RTOL)
    # T^2 ~ F(1, df).
    for df in (.5, 3, 40, 1e7):
        for t in (.01, 1., 2.5, 9., 30.):
            assert dist.f_sf(t * t, 1, df) == pytest.approx(2 * dist.t_sf(t, df), rel=1e-12)
    # df2 = inf: df1 F ~ chi2(df1);  df1 = inf: df2 / F ~ chi2(df2).
    for df in (.3, 1, 4, 25, 3000):
        for x in (.01, .7, 1., 2.2, 9.):
            assert dist.f_cdf(x, df, math.inf) == pytest.approx(stats.chi2.cdf(df * x, df),
                                                              rel=RTOL)
            assert dist.f_sf(x, df, math.inf) == pytest.approx(stats.chi2.sf(df * x, df),
                                                             rel=RTOL, abs=ATOL)
            assert dist.f_sf(x, math.inf, df) == pytest.approx(stats.chi2.cdf(df / x, df), rel=RTOL)
            assert dist.f_cdf(x, math.inf, df) == pytest.approx(stats.chi2.sf(df / x, df),
                                                              rel=RTOL, abs=ATOL)
            assert dist.f_sf(x, df, 1e15) == pytest.approx(dist.f_sf(x, df, math.inf), rel=1e-9)
    # Tiny numerator df: the lower tail is not small even at the edge of the float range,
    # and tiny denominator df keeps a visible upper tail there (SciPy returns 1 and 0).
    lower = math.exp(5e-4 * math.log(1e-300 * 1e-3 / 30) - special.betaln(5e-4, 15)) / 5e-4
    assert dist.f_cdf(1e-300, 1e-3, 30) == pytest.approx(lower, rel=1e-9)
    assert dist.f_sf(1e-300, 1e-3, 30) == pytest.approx(1 - lower, rel=1e-9)
    assert dist.f_sf(1e300, 30, 1e-3) == pytest.approx(lower, rel=1e-9)
    assert dist.f_cdf(0, 3, 9) == 0.0 and dist.f_sf(0, 3, 9) == 1.0
    assert dist.f_cdf(-2, 3, 9) == 0.0 and dist.f_sf(-2, 3, 9) == 1.0
    assert dist.f_cdf(math.inf, 3, 9) == 1.0 and dist.f_sf(math.inf, 3, 9) == 0.0


# --------------------------------------------------------------------------------------
# Quantiles
# --------------------------------------------------------------------------------------

CENTRAL = [1e-10, 1e-6, 1e-3, .01, .05, .2, .5, .8, .95, .99, 1 - 1e-6, 1 - 1e-10]


@pytest.mark.parametrize("df", [.1, .5, 1, 2, 3, 5, 10, 30, 100, 1e3, 1999, 2001, 1e4])
def test_chi2_quantiles_match_scipy(df):
    probabilities = np.array(CENTRAL)
    close([dist.chi2_ppf(p, df) for p in probabilities], stats.chi2.ppf(probabilities, df),
          rtol=1e-9)
    close([dist.chi2_isf(p, df) for p in probabilities], stats.chi2.isf(probabilities, df),
          rtol=1e-9)


@pytest.mark.parametrize("df", [.5, 1, 2, 3, 5, 10, 30, 100, 1e3, 1e4, 1e6, 1e9, 1e12])
def test_student_t_quantiles_match_scipy(df):
    probabilities = np.array(CENTRAL)
    close([dist.t_ppf(p, df) for p in probabilities], stats.t.ppf(probabilities, df),
          rtol=1e-9, atol=1e-15)
    close([dist.t_isf(p, df) for p in probabilities], stats.t.isf(probabilities, df),
          rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("df1", [1, 2, 3, 7, 50, 1000, 1e5])
@pytest.mark.parametrize("df2", [2, 5, 30, 1000, 1e5])
def test_f_quantiles_match_scipy(df1, df2):
    # Boost's inverse beta is itself only ~1e-7 accurate in the far tails, hence the
    # narrower probability range; the round-trip tests below cover 1e-300 .. 1 - 1e-16.
    probabilities = np.array([1e-6, 1e-3, .01, .05, .2, .5, .8, .95, .99, 1 - 1e-6])
    close([dist.f_ppf(p, df1, df2) for p in probabilities], stats.f.ppf(probabilities, df1, df2),
          rtol=1e-9)
    close([dist.f_isf(p, df1, df2) for p in probabilities], stats.f.isf(probabilities, df1, df2),
          rtol=1e-9)


def round_trip(ppf, isf, cdf, sf, tolerance=1e-9):
    """cdf(ppf(p)) = p and sf(isf(p)) = p wherever the quantile is inside the float range."""
    for p in PROBABILITIES:
        for quantile, tail in ((ppf(p), cdf), (isf(p), sf)):
            if 2.3e-308 < abs(quantile) < math.inf:       # representable and not subnormal
                # p is the small, well-conditioned tail for p <= 1/2; above it the
                # complement is what the quantile actually resolves.
                if p <= .5:
                    assert tail(quantile) == pytest.approx(p, rel=tolerance, abs=0)
                else:
                    other = sf if tail is cdf else cdf
                    assert other(quantile) == pytest.approx(1 - p, rel=tolerance, abs=0)


@pytest.mark.parametrize("df", [1e-3, .1, .5, 1, 2, 3, 5, 10, 30, 100, 1e3, 1999, 2000, 2001, 1e4,
                                1e6, 1e9])
def test_chi2_quantiles_invert_both_tails(df):
    round_trip(lambda p: dist.chi2_ppf(p, df), lambda p: dist.chi2_isf(p, df),
               lambda x: dist.chi2_cdf(x, df), lambda x: dist.chi2_sf(x, df))


@pytest.mark.parametrize("df", [.1, .5, 1, 2, 3, 5, 10, 30, 100, 1e3, 1e4, 1e6, 1e9, 1e12,
                                math.inf])
def test_student_t_quantiles_invert_both_tails(df):
    round_trip(lambda p: dist.t_ppf(p, df), lambda p: dist.t_isf(p, df),
               lambda x: dist.t_cdf(x, df), lambda x: dist.t_sf(x, df))
    assert dist.t_ppf(.5, df) == 0.0
    for p in (1e-300, 1e-9, .25, .49):
        assert dist.t_isf(p, df) == -dist.t_ppf(p, df)
    assert dist.t_ppf(.75, df) == -dist.t_ppf(.25, df)          # 1 - p is exact here


@pytest.mark.parametrize("df1", [.1, 1, 2, 3, 7, 50, 1000, 1e5, 1e9, math.inf])
@pytest.mark.parametrize("df2", [.1, 1, 2, 5, 30, 1000, 1e6, 1e9, math.inf])
def test_f_quantiles_invert_both_tails(df1, df2):
    if math.isinf(df1) and math.isinf(df2):
        with pytest.raises(KernelError):
            dist.f_ppf(.5, df1, df2)
        return
    # With both df ~ 1e9 the tail probability moves by 1e-11 per ulp of the quantile.
    tolerance = 1e-9 if min(df1, df2) < 1e8 else 5e-9
    round_trip(lambda p: dist.f_ppf(p, df1, df2), lambda p: dist.f_isf(p, df1, df2),
               lambda x: dist.f_cdf(x, df1, df2), lambda x: dist.f_sf(x, df1, df2), tolerance)


def test_quantiles_recover_the_argument():
    """|ppf(cdf(x)) - x| <= 1e-9 max(1, |x|) on the side where the cdf resolves x."""
    for df in (1, 3, 30, 1e4, 1e9):
        for x in (-1e6, -40., -8., -2., -.3, -1e-6):
            if dist.t_cdf(x, df) > 1e-300:
                assert abs(dist.t_ppf(dist.t_cdf(x, df), df) - x) <= 1e-9 * max(1, abs(x))
                assert abs(dist.t_isf(dist.t_sf(-x, df), df) + x) <= 1e-9 * max(1, abs(x))
    for df in (.5, 1, 4, 60, 3000, 1e7):
        for multiple in (1e-8, .01, .4, .9):
            x = df * multiple
            if dist.chi2_cdf(x, df) > 1e-300:
                assert abs(dist.chi2_ppf(dist.chi2_cdf(x, df), df) - x) <= 1e-9 * max(1, x)
        for extra in (.5, 3, 12, 30):
            x = df + extra * math.sqrt(2 * df) + extra
            if dist.chi2_sf(x, df) > 1e-300:
                assert abs(dist.chi2_isf(dist.chi2_sf(x, df), df) - x) <= 1e-9 * max(1, x)
    for df1, df2 in ((1, 4), (3, 30), (12, 1e6), (2000, 2000), (.7, 9)):
        for x in (1e-6, .02, .4):
            if dist.f_cdf(x, df1, df2) > 1e-300:
                assert abs(dist.f_ppf(dist.f_cdf(x, df1, df2), df1, df2) - x) <= 1e-9
        for x in (1.5, 4., 60., 1e4):
            upper = dist.f_sf(x, df1, df2)
            if upper > 1e-300:
                assert abs(dist.f_isf(upper, df1, df2) - x) <= 1e-9 * x


def test_extreme_quantiles_against_exact_laws():
    # Cauchy and t(2) quantiles are elementary; SciPy returns inf or half the value here.
    for q in (1e-300, 1e-200, 1e-20, .01, .3):
        assert dist.t_isf(q, 1) == pytest.approx(1 / math.tan(math.pi * q), rel=1e-12)
        alpha = 2 * q
        exact = math.sqrt(2) * (1 - alpha) / math.sqrt(alpha * (2 - alpha))
        assert dist.t_isf(q, 2) == pytest.approx(exact, rel=1e-12)
        # F(2, d): x = (d/2) (q^(-2/d) - 1).
        for df in (.5, 4, 300, 1e8):
            exponent = -2 * math.log(q) / df
            exact = .5 * df * math.expm1(exponent) if exponent < 700 else math.inf
            assert dist.f_isf(q, 2, df) == pytest.approx(exact, rel=1e-11)
        assert dist.chi2_isf(q, 2) == pytest.approx(-2 * math.log(q), rel=1e-12)
        assert dist.chi2_ppf(q, 2) == pytest.approx(-2 * math.log1p(-q), rel=1e-12)
    assert dist.t_isf(1e-300, .1) == math.inf                 # true value ~ 1e2999
    assert dist.t_ppf(1e-300, .1) == -math.inf
    assert dist.chi2_ppf(1e-300, 1e-3) == 0.0                 # true value ~ 1e-600000
    assert dist.f_ppf(1e-300, 1e-3, 5) == 0.0
    assert dist.f_isf(1e-300, 5, 1e-3) == math.inf
    assert dist.chi2_isf(1e-300, 1) == pytest.approx(dist.normal_isf(5e-301) ** 2, rel=1e-12)
    assert dist.f_ppf(.3, 4, math.inf) == pytest.approx(stats.chi2.ppf(.3, 4) / 4, rel=1e-10)
    assert dist.f_isf(.3, math.inf, 4) == pytest.approx(4 / stats.chi2.ppf(.3, 4), rel=1e-10)


def test_quantiles_are_monotone_and_cheap():
    """The solver is Newton with a bracket, not a long bisection."""
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
        for df in (.5, 1, 3, 30, 1e3, 1e6):
            previous = [-math.inf] * 3
            for p in (1e-300, 1e-50, 1e-6, .01, .3, .5, .8, .999, 1 - 1e-12):
                current = []
                for quantile in (lambda: dist.chi2_ppf(p, df), lambda: dist.t_ppf(p, df),
                                 lambda: dist.f_ppf(p, 4, df)):
                    calls = 0
                    current.append(quantile())
                    worst = max(worst, calls)
                assert all(new >= old for new, old in zip(current, previous))
                previous = current
    finally:
        dist._solve_log_newton = solve
    assert worst <= 40


# --------------------------------------------------------------------------------------
# p_value / critical
# --------------------------------------------------------------------------------------

def test_p_values_follow_the_reference_distribution():
    assert dist.p_value(1.96, "normal") == pytest.approx(2 * stats.norm.sf(1.96), rel=1e-13)
    assert dist.p_value(-1.96, "normal") == dist.p_value(1.96, "normal")
    assert dist.p_value(1.96, "normal", two_sided=False) == pytest.approx(stats.norm.sf(1.96),
                                                                           rel=1e-13)
    assert dist.p_value(-1.96, "normal", two_sided=False) == pytest.approx(stats.norm.cdf(1.96),
                                                                            rel=1e-13)
    assert dist.p_value(2.3, "t", 17) == pytest.approx(2 * stats.t.sf(2.3, 17), rel=RTOL)
    assert dist.p_value(-2.3, "t", 17) == dist.p_value(2.3, "t", 17)
    assert dist.p_value(-2.3, "t", 17, two_sided=False) == pytest.approx(stats.t.cdf(2.3, 17),
                                                                         rel=RTOL)
    assert dist.p_value(0, "t", 17) == 1.0 == dist.p_value(0, "normal")
    assert dist.p_value(7.8, "chi2", 3) == pytest.approx(stats.chi2.sf(7.8, 3), rel=RTOL)
    assert dist.p_value(7.8, "chi2", 3, two_sided=False) == dist.p_value(7.8, "chi2", 3)
    assert dist.p_value(4.2, "F", 3, 40) == pytest.approx(stats.f.sf(4.2, 3, 40), rel=RTOL)
    assert dist.p_value(4.2, "f", 3, 40) == dist.p_value(4.2, "F", 3, 40)
    assert dist.p_value(4.2, "F", 3, math.inf) == pytest.approx(stats.chi2.sf(12.6, 3), rel=RTOL)
    assert dist.p_value(2.3, "t", math.inf) == pytest.approx(dist.p_value(2.3, "normal"), rel=1e-15)
    assert dist.p_value(math.inf, "normal") == 0.0 == dist.p_value(-math.inf, "t", 5)
    assert dist.p_value(math.inf, "chi2", 2) == 0.0 == dist.p_value(math.inf, "F", 2, 9)
    assert dist.p_value(-math.inf, "normal", two_sided=False) == 1.0
    assert dist.p_value(0, "chi2", 2) == 1.0 == dist.p_value(0, "F", 2, 9)
    # ints and 0-dim tensors are accepted everywhere.
    tensor = torch.tensor(2.3, dtype=torch.float64)
    assert dist.p_value(tensor, "t", torch.tensor(17)) == dist.p_value(2.3, "t", 17)
    assert dist.p_value(4, "F", 3, 40) == dist.p_value(4.0, "F", 3.0, 40.0)
    assert isinstance(dist.p_value(tensor, "normal"), float)


@pytest.mark.parametrize("alpha", [.5, .1, .05, .01, 1e-4, 1e-12, 1e-100])
def test_critical_values_invert_p_values(alpha):
    assert dist.critical(alpha, "normal") == pytest.approx(stats.norm.isf(alpha / 2), rel=1e-12)
    assert dist.critical(alpha, "normal", two_sided=False) == pytest.approx(stats.norm.isf(alpha),
                                                                            rel=1e-12, abs=1e-15)
    assert dist.critical(alpha, "t", 23) == pytest.approx(stats.t.isf(alpha / 2, 23), rel=1e-9)
    assert dist.critical(alpha, "chi2", 6) == pytest.approx(stats.chi2.isf(alpha, 6), rel=1e-9)
    for name, df, df2 in (("normal", None, None), ("t", 23, None), ("t", 1e7, None),
                          ("chi2", 6, None), ("F", 4, 60), ("F", 4, math.inf), ("F", 300, 1e8)):
        for two_sided in (True, False):
            value = dist.critical(alpha, name, df, df2, two_sided=two_sided)
            assert dist.p_value(value, name, df, df2, two_sided=two_sided) == pytest.approx(
                alpha, rel=1e-9, abs=0)
    assert dist.critical(.05, "F", 4, 60) == pytest.approx(stats.f.isf(.05, 4, 60), rel=1e-9)


# --------------------------------------------------------------------------------------
# Invalid arguments
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: dist.gamma_p(0, 1), lambda: dist.gamma_q(-1, 1), lambda: dist.gamma_p(1, -1),
    lambda: dist.gamma_q(math.inf, 1), lambda: dist.gamma_p(math.nan, 1),
    lambda: dist.gamma_q(1, math.nan),
    lambda: dist.beta_inc(0, 1, .5), lambda: dist.beta_inc(1, -2, .5),
    lambda: dist.beta_inc(1, 1, 1.5), lambda: dist.beta_inc(1, 1, -.1),
    lambda: dist.beta_inc(1, 1, math.nan),
    lambda: dist.normal_cdf(math.nan), lambda: dist.normal_sf(math.nan),
    lambda: dist.normal_pdf(math.nan), lambda: dist.normal_cdf("x"),
    lambda: dist.normal_ppf(0), lambda: dist.normal_ppf(1), lambda: dist.normal_ppf(-.1),
    lambda: dist.normal_ppf(1.5), lambda: dist.normal_ppf(math.nan), lambda: dist.normal_isf(0),
    lambda: dist.t_cdf(1, 0), lambda: dist.t_sf(1, -3), lambda: dist.t_cdf(math.nan, 3),
    lambda: dist.t_sf(1, math.nan), lambda: dist.t_ppf(0, 3), lambda: dist.t_ppf(.5, 0),
    lambda: dist.t_isf(1, 3),
    lambda: dist.chi2_cdf(1, 0), lambda: dist.chi2_sf(1, -1), lambda: dist.chi2_sf(math.nan, 2),
    lambda: dist.chi2_sf(1, math.inf), lambda: dist.chi2_ppf(1, 2), lambda: dist.chi2_ppf(.5, 0),
    lambda: dist.chi2_isf(0, 2),
    lambda: dist.f_cdf(1, 0, 3), lambda: dist.f_sf(1, 3, 0), lambda: dist.f_sf(1, 3, -1),
    lambda: dist.f_sf(math.nan, 3, 3), lambda: dist.f_cdf(1, math.inf, math.inf),
    lambda: dist.f_ppf(0, 3, 3), lambda: dist.f_ppf(.5, 3, math.nan), lambda: dist.f_isf(1, 3, 3),
    lambda: dist.p_value(1, "weibull"), lambda: dist.p_value(1, "t"),
    lambda: dist.p_value(1, "chi2"), lambda: dist.p_value(1, "F", 3),
    lambda: dist.p_value(1, "F", None, 3), lambda: dist.p_value(math.nan, "normal"),
    lambda: dist.p_value(1, "t", 0), lambda: dist.p_value(1, None),
    lambda: dist.critical(.05, "gamma"), lambda: dist.critical(.05, "t"),
    lambda: dist.critical(.05, "F", 3), lambda: dist.critical(0, "normal"),
    lambda: dist.critical(1, "chi2", 3), lambda: dist.critical(.05, "chi2", -3),
])
def test_invalid_arguments_raise_kernel_errors(call):
    with pytest.raises(KernelError) as error:
        call()
    assert error.value.code == "invalid_inference"


# --------------------------------------------------------------------------------------
# Gaussian quadrature
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 20, 64, 200])
def test_gauss_hermite_matches_numpy(n):
    nodes, weights = dist.gauss_hermite(n)
    expected_nodes, expected_weights = np.polynomial.hermite.hermgauss(n)
    assert nodes.dtype == weights.dtype == torch.float64 and nodes.shape == weights.shape == (n,)
    assert bool((nodes[1:] > nodes[:-1]).all())
    np.testing.assert_allclose(nodes.numpy(), expected_nodes, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=2e-12, atol=0)
    assert torch.equal(nodes, -nodes.flip(0)) and torch.equal(weights, weights.flip(0))
    assert float(weights.sum()) == pytest.approx(math.sqrt(math.pi), rel=1e-15)
    # Exact for even monomials of degree < 2n:  int e^(-x^2) x^(2k) dx = Gamma(k + 1/2).
    for k in range(0, min(n, 40)):
        assert float((weights * nodes ** (2 * k)).sum()) == pytest.approx(math.gamma(k + .5),
                                                                          rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 64, 200])
def test_gauss_legendre_matches_numpy(n):
    nodes, weights = dist.gauss_legendre(n)
    expected_nodes, expected_weights = np.polynomial.legendre.leggauss(n)
    assert nodes.dtype == weights.dtype == torch.float64 and nodes.shape == weights.shape == (n,)
    assert bool((nodes[1:] > nodes[:-1]).all()) and float(nodes.abs().max()) < 1
    np.testing.assert_allclose(nodes.numpy(), expected_nodes, rtol=0, atol=1e-14)
    # NumPy's own weights are ~2e-11 off for n = 200 (ours agree with mpmath to 1e-13).
    np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=1e-10, atol=0)
    assert float(weights.sum()) == pytest.approx(2.0, rel=1e-15)
    for k in range(0, min(n, 60)):
        assert float((weights * nodes ** (2 * k)).sum()) == pytest.approx(2 / (2 * k + 1),
                                                                          rel=1e-12)


@pytest.mark.parametrize("n", [20, 200])
def test_quadrature_weights_match_high_precision_recurrence(n):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(40):
        nodes, weights = dist.gauss_legendre(n)
        for node, weight in list(zip(nodes.tolist(), weights.tolist()))[:: max(1, n // 10)]:
            x = mp.mpf(node)
            for _ in range(3):
                below, current = mp.mpf(1), x
                for j in range(1, n):
                    below, current = current, ((2 * j + 1) * x * current - j * below) / (j + 1)
                derivative = n * (x * current - below) / (x * x - 1)
                x -= current / derivative
            assert node == pytest.approx(float(x), abs=1e-15)
            assert weight == pytest.approx(float(2 / ((1 - x * x) * derivative ** 2)), rel=1e-12)
        nodes, weights = dist.gauss_hermite(n)
        for node, weight in list(zip(nodes.tolist(), weights.tolist()))[:: max(1, n // 10)]:
            x = mp.mpf(node)
            for _ in range(3):
                below, current = mp.mpf(0), mp.pi ** (-mp.mpf(1) / 4)
                for j in range(n):
                    below, current = current, (x * mp.sqrt(mp.mpf(2) / (j + 1)) * current
                                               - mp.sqrt(mp.mpf(j) / (j + 1)) * below)
                x -= current / (mp.sqrt(2 * n) * below)
            assert node == pytest.approx(float(x), abs=1e-13)
            assert weight == pytest.approx(float(1 / (n * below * below)), rel=5e-12)


def test_quadrature_integrates_random_effects_style_expectations():
    # E[Phi(a + b Z)] = Phi(a / sqrt(1 + b^2)) for Z ~ N(0, 1): the random-effects probit kernel.
    for n, tolerance in ((64, 1e-8), (200, 1e-14)):
        nodes, weights = dist.gauss_hermite(n)
        z = math.sqrt(2) * nodes
        for a, b in ((.3, .8), (-1.2, 2.5), (2., .1)):
            values = 0.5 * torch.special.erfc(-(a + b * z) / math.sqrt(2))
            estimate = float((weights * values).sum()) / math.sqrt(math.pi)
            assert estimate == pytest.approx(stats.norm.cdf(a / math.sqrt(1 + b * b)),
                                             abs=tolerance)
    nodes, weights = dist.gauss_legendre(20)
    assert float((weights * torch.exp(nodes)).sum()) == pytest.approx(2 * math.sinh(1), rel=1e-14)


def test_quadrature_is_cached_safely_and_survives_large_orders():
    nodes, weights = dist.gauss_hermite(12)
    nodes *= 100.0
    weights.zero_()
    again_nodes, again_weights = dist.gauss_hermite(12)
    assert float(again_nodes.abs().max()) < 10 and float(again_weights.sum()) > 1
    assert not again_nodes.requires_grad and not again_nodes.is_inference()
    # NumPy's hermgauss overflows from n ~ 400; the rescaled recurrence does not.
    nodes, weights = dist.gauss_hermite(600)
    assert bool(torch.isfinite(nodes).all()) and bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    assert float((weights * nodes ** 2).sum()) == pytest.approx(math.sqrt(math.pi) / 2, rel=1e-12)
    for bad in (0, -3, 2.5, "4", True):
        for rule in (dist.gauss_hermite, dist.gauss_legendre):
            with pytest.raises(KernelError) as error:
                rule(bad)
            assert error.value.code == "invalid_quadrature"
