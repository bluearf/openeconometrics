"""Independent numerical expectations for the sandwich "meat" builders.

Oracles: brute-force numpy double loops over pairs of observations on small data,
explicit inclusion-exclusion, and statsmodels' sandwich_covariance module.
"""

import itertools
import math

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw
import torch
from numpy.testing import assert_allclose

from openecon.engines import covariance as cv
from openecon.engines import linalg
from openecon.engines.contracts import KernelError


def t(values):
    return torch.tensor(np.asarray(values), dtype=torch.float64)


def codes(values):
    return torch.tensor(np.asarray(values), dtype=torch.int64)


def pair_sum(scores, weight):
    """sum_i sum_j weight(i, j) s_i s_j' by an explicit double loop."""
    n, k = scores.shape
    total = np.zeros((k, k))
    for i in range(n):
        for j in range(n):
            value = weight(i, j)
            if value:
                total += value * np.outer(scores[i], scores[j])
    return total


def kernel(name, distance, lags):
    """Textbook kernel k(d / (L + 1)), written independently of the module."""
    if distance == 0:
        return 1.0
    z = distance / (lags + 1)
    if name == "bartlett":
        return max(0.0, 1 - z)
    if name == "truncated":
        return 1.0 if distance <= lags else 0.0
    if name == "parzen":
        if z <= 0.5:
            return 1 - 6 * z ** 2 + 6 * z ** 3
        return 2 * (1 - z) ** 3 if z < 1 else 0.0
    a = 6 * math.pi * z / 5
    return 25 / (12 * math.pi ** 2 * z ** 2) * (math.sin(a) / a - math.cos(a))


KERNELS = ["bartlett", "truncated", "parzen", "quadratic_spectral"]


@pytest.fixture
def small():
    rng = np.random.default_rng(404)
    n, k = 48, 3
    scores = rng.normal(size=(n, k)) * [1.0, 5.0, 0.1]
    scores[1:] += 0.6 * scores[:-1]
    return scores


def test_group_sums_and_counts(small):
    rng = np.random.default_rng(1)
    group = rng.integers(0, 6, size=len(small))
    group[group == 4] = 0                      # an empty group stays in the output
    sums = cv.group_sums(t(small), codes(group), 6)
    expected = np.stack([small[group == g].sum(axis=0) for g in range(6)])
    assert_allclose(sums.numpy(), expected, rtol=1e-13, atol=1e-13)
    assert_allclose(cv.group_sums(t(small[:, 0]), codes(group), 6).numpy(), expected[:, 0],
                    rtol=1e-13, atol=1e-13)
    counts = cv.group_counts(codes(group), 6)
    assert counts.dtype == torch.int64
    assert counts.tolist() == [int((group == g).sum()) for g in range(6)]
    for bad in (codes(group).float(), codes(group)[:-1], codes(group) + 1, -codes(group) - 1):
        with pytest.raises(KernelError) as error:
            cv.group_sums(t(small), bad, 6)
        assert error.value.code == "invalid_clusters"


def test_hc_residuals():
    resid = t([1.0, -2.0, 0.5, 4.0])
    leverage = t([0.1, 0.5, 0.75, 0.0])
    assert cv.hc_residuals(resid, None, "HC0") is resid
    assert cv.hc_residuals(resid, leverage, "HC1") is resid
    assert_allclose(cv.hc_residuals(resid, leverage, "HC2").numpy(),
                    resid.numpy() / np.sqrt(1 - leverage.numpy()), rtol=1e-15)
    assert_allclose(cv.hc_residuals(resid, leverage, "HC3").numpy(),
                    resid.numpy() / (1 - leverage.numpy()), rtol=1e-15)
    wide = torch.stack([resid, 2 * resid], dim=1)
    assert_allclose(cv.hc_residuals(wide, leverage, "HC3").numpy(),
                    wide.numpy() / (1 - leverage.numpy())[:, None], rtol=1e-15)
    for kind in ("HC2", "HC3"):
        with pytest.raises(KernelError) as error:
            cv.hc_residuals(resid, t([0.1, 1.0, 0.2, 0.3]), kind)
        assert error.value.code == "undefined_leverage_correction"
        with pytest.raises(KernelError) as error:
            cv.hc_residuals(resid, None, kind)
        assert error.value.code == "missing_leverage"
    with pytest.raises(KernelError) as error:
        cv.hc_residuals(resid, leverage, "HC4")
    assert error.value.code == "unsupported_covariance"


def test_white_and_cluster_meats_match_double_loops(small):
    rng = np.random.default_rng(2)
    group = rng.integers(0, 7, size=len(small))
    white = cv.meat_white(t(small))
    assert_allclose(white.numpy(), pair_sum(small, lambda i, j: i == j), rtol=1e-12)
    cluster = cv.meat_cluster(t(small), codes(group), 7)
    assert_allclose(cluster.numpy(), pair_sum(small, lambda i, j: group[i] == group[j]),
                    rtol=1e-11, atol=1e-11)
    assert torch.equal(cluster, cluster.T)
    # One cluster per observation is the White meat.
    own = cv.meat_cluster(t(small), codes(np.arange(len(small))), len(small))
    assert_allclose(own.numpy(), white.numpy(), rtol=1e-13)


def test_ols_robust_covariances_match_statsmodels():
    rng = np.random.default_rng(6)
    n = 160
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    group = rng.integers(0, 12, size=n)
    y = x @ [1.0, 2.0, -1.0, 0.5] + rng.normal(size=n) * (1 + x[:, 1] ** 2) \
        + rng.normal(size=12)[group]
    fit = linalg.least_squares(t(x), t(y), need_leverage=True)
    oracle = sm.OLS(y, x).fit()
    k = x.shape[1]
    for kind, expected in [("HC0", oracle.cov_HC0), ("HC1", oracle.cov_HC1),
                           ("HC2", oracle.cov_HC2), ("HC3", oracle.cov_HC3)]:
        scores = t(x) * cv.hc_residuals(fit.resid, fit.leverage, kind)[:, None]
        covariance = cv.sandwich(fit.xtx_inv, cv.meat_white(scores))
        if kind == "HC1":
            covariance = covariance * n / (n - k)
        assert_allclose(covariance.numpy(), expected, rtol=1e-9, atol=1e-14)
    scores = t(x) * fit.resid[:, None]
    clustered = cv.sandwich(fit.xtx_inv, cv.meat_cluster(scores, codes(group), 12)) \
        * cv.cluster_factor(n, k, 12)
    expected = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": group}).cov_params()
    assert_allclose(clustered.numpy(), expected, rtol=1e-9, atol=1e-14)
    hac = cv.sandwich(fit.xtx_inv, cv.meat_hac(scores, 5))
    expected = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).cov_params()
    assert_allclose(hac.numpy(), expected, rtol=1e-9, atol=1e-14)


@pytest.fixture
def three_way(small):
    rng = np.random.default_rng(3)
    n = len(small)
    firm = rng.integers(0, 6, size=n)
    year = rng.integers(0, 5, size=n)
    state = rng.integers(0, 4, size=n)
    return small, [firm, year, state], [6, 5, 4]


def explicit_inclusion_exclusion(scores, groups, factor):
    """Numpy inclusion-exclusion over crossed labels; factor(G_S) scales each term."""
    k = scores.shape[1]
    total = np.zeros((k, k))
    for size in range(1, len(groups) + 1):
        for subset in itertools.combinations(range(len(groups)), size):
            labels = [tuple(groups[d][i] for d in subset) for i in range(len(scores))]
            cells = sorted(set(labels))
            sums = np.stack([scores[[label == cell for label in labels]].sum(axis=0)
                             for cell in cells])
            total += (-1) ** (size + 1) * factor(len(cells)) * sums.T @ sums
    return total


def test_two_way_meat_is_the_three_term_formula(three_way):
    scores, (firm, year, _), _ = three_way
    result = cv.meat_multiway(t(scores), [(codes(firm), 6), (codes(year), 5)], adjust="none")
    both = np.unique(np.stack([firm, year], axis=1), axis=0, return_inverse=True)[1].ravel()
    expected = (cv.meat_cluster(t(scores), codes(firm), 6)
                + cv.meat_cluster(t(scores), codes(year), 5)
                - cv.meat_cluster(t(scores), codes(both), int(both.max()) + 1)).numpy()
    assert_allclose(result.meat.numpy(), expected, rtol=1e-11, atol=1e-11)
    # CGM: pairs of observations sharing a cluster in ANY dimension enter exactly once.
    brute = pair_sum(scores, lambda i, j: firm[i] == firm[j] or year[i] == year[j])
    assert_allclose(result.meat.numpy(), brute, rtol=1e-10, atol=1e-10)
    assert result.group_counts == [6, 5] and result.min_groups == 5
    assert result.psd_adjusted is False
    assert [(term["dimensions"], term["sign"]) for term in result.terms] == \
        [([0], 1), ([1], 1), ([0, 1], -1)]
    assert result.terms[2]["groups"] == int(both.max()) + 1


def test_three_way_meat_and_small_sample_conventions(three_way):
    scores, groups, sizes = three_way
    n, k = len(scores), 7
    dimensions = [(codes(g), size) for g, size in zip(groups, sizes)]

    plain = cv.meat_multiway(t(scores), dimensions, adjust="none")
    assert_allclose(plain.meat.numpy(), explicit_inclusion_exclusion(scores, groups, lambda g: 1.0),
                    rtol=1e-10, atol=1e-10)
    shares = pair_sum(scores, lambda i, j: any(g[i] == g[j] for g in groups))
    assert_allclose(plain.meat.numpy(), shares, rtol=1e-10, atol=1e-10)
    assert len(plain.terms) == 7 and plain.group_counts == sizes and plain.min_groups == 4
    assert all(term["factor"] == 1.0 for term in plain.terms)

    minimum = cv.meat_multiway(t(scores), dimensions)       # default: reghdfe / ivreg2
    assert_allclose(minimum.meat.numpy(), plain.meat.numpy() * 4 / 3, rtol=1e-13)
    each = cv.meat_multiway(t(scores), dimensions, adjust="each")
    assert_allclose(each.meat.numpy(),
                    explicit_inclusion_exclusion(scores, groups, lambda g: g / (g - 1)),
                    rtol=1e-10, atol=1e-10)
    assert [term["factor"] for term in each.terms] == \
        [term["groups"] / (term["groups"] - 1) for term in each.terms]

    with_df = cv.meat_multiway(t(scores), dimensions, adjust="min", n=n, k=k)
    assert_allclose(with_df.meat.numpy(), plain.meat.numpy() * 4 / 3 * (n - 1) / (n - k),
                    rtol=1e-13)
    one_way = cv.meat_multiway(t(scores), dimensions[:1], adjust="min", n=n, k=k)
    assert_allclose(one_way.meat.numpy(),
                    cv.meat_cluster(t(scores), codes(groups[0]), 6).numpy()
                    * cv.cluster_factor(n, k, 6), rtol=1e-13)


def test_two_way_cluster_covariance_matches_statsmodels():
    rng = np.random.default_rng(9)
    n = 200
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    firm = rng.integers(0, 15, size=n)
    year = rng.integers(0, 8, size=n)
    y = x @ [1.0, -1.0, 2.0] + rng.normal(size=15)[firm] + rng.normal(size=8)[year] \
        + rng.normal(size=n)
    fit = linalg.least_squares(t(x), t(y))
    scores = t(x) * fit.resid[:, None]
    meat = cv.meat_multiway(scores, [(codes(firm), 15), (codes(year), 8)], adjust="each",
                            n=n, k=3)
    expected = sw.cov_cluster_2groups(sm.OLS(y, x).fit(), firm, year)[0]
    assert_allclose(cv.sandwich(fit.xtx_inv, meat.meat).numpy(), expected, rtol=1e-9, atol=1e-14)


def test_multiway_counts_only_nonempty_groups_and_validates(three_way):
    scores, (firm, year, _), _ = three_way
    # Declaring more groups than are present must not change counts or factors.
    padded = cv.meat_multiway(t(scores), [(codes(firm), 40), (codes(year), 9)])
    exact = cv.meat_multiway(t(scores), [(codes(firm), 6), (codes(year), 5)])
    assert padded.group_counts == [6, 5]
    assert_allclose(padded.meat.numpy(), exact.meat.numpy(), rtol=1e-13)
    # Observation-level intersections reuse the White meat.
    unique = np.arange(len(scores))
    crossed = cv.meat_multiway(t(scores), [(codes(firm), 6), (codes(unique), len(scores))],
                               adjust="none")
    assert_allclose(crossed.meat.numpy(), cv.meat_cluster(t(scores), codes(firm), 6).numpy(),
                    rtol=1e-11, atol=1e-11)

    single = codes(np.zeros(len(scores)))
    for code, call in [
        ("insufficient_clusters", lambda: cv.meat_multiway(t(scores), [(single, 1)])),
        ("invalid_cluster_adjustment",
         lambda: cv.meat_multiway(t(scores), [(codes(firm), 6)], adjust="max")),
        ("invalid_cluster_adjustment",
         lambda: cv.meat_multiway(t(scores), [(codes(firm), 6)], n=len(scores))),
        ("invalid_clusters", lambda: cv.meat_multiway(t(scores), [])),
        ("invalid_clusters", lambda: cv.meat_multiway(t(scores), [(codes(firm), 3)])),
    ]:
        with pytest.raises(KernelError) as error:
            call()
        assert error.value.code == code
    assert cv.meat_multiway(t(scores), [(single, 1)], adjust="none").min_groups == 1


def test_force_psd_replaces_negative_eigenvalues():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(12, 3))
    first = rng.integers(0, 3, size=12)
    second = rng.integers(0, 3, size=12)
    dimensions = [(codes(first), 3), (codes(second), 3)]
    raw = cv.meat_multiway(t(scores), dimensions, adjust="none")
    values, vectors = np.linalg.eigh(raw.meat.numpy())
    assert values[0] < -1.0 and raw.psd_adjusted is False      # genuinely indefinite
    fixed = cv.meat_multiway(t(scores), dimensions, adjust="none", force_psd=True)
    expected = (vectors * np.maximum(values, 0)) @ vectors.T
    assert fixed.psd_adjusted is True
    assert_allclose(fixed.meat.numpy(), expected, rtol=1e-11, atol=1e-11)
    assert np.linalg.eigvalsh(fixed.meat.numpy())[0] > -1e-12

    repaired, adjusted = cv.nearest_psd(raw.meat)
    assert adjusted is True
    assert_allclose(repaired.numpy(), expected, rtol=1e-11, atol=1e-11)
    # A PSD matrix, even a rank-deficient one, is returned unchanged.
    low_rank = t(scores[:2].T @ scores[:2])
    same, adjusted = cv.nearest_psd(low_rank)
    assert adjusted is False and torch.equal(same, linalg.symmetrize(low_rank))
    one_way = cv.meat_multiway(t(scores), dimensions[:1], adjust="none", force_psd=True)
    assert one_way.psd_adjusted is False


def test_kernel_weights_formulas():
    assert_allclose(cv.kernel_weights(6).numpy(), sw.weights_bartlett(6)[1:], rtol=1e-15)
    assert_allclose(cv.kernel_weights(4, "bartlett").numpy(), [0.8, 0.6, 0.4, 0.2], rtol=1e-15)
    assert cv.kernel_weights(3, "truncated").tolist() == [1.0, 1.0, 1.0]
    for name in KERNELS:
        for lags in (1, 2, 5, 8):
            weights = cv.kernel_weights(lags, name)
            assert weights.dtype == torch.float64 and weights.shape == (lags,)
            assert_allclose(weights.numpy(), [kernel(name, d, lags) for d in range(1, lags + 1)],
                            rtol=1e-13, atol=1e-15)
        assert cv.kernel_weights(0, name).shape == (0,)
    # Compact kernels vanish beyond L; the quadratic spectral kernel does not.
    assert cv.kernel_weights(3, "parzen", count=6).tolist()[3:] == [0.0, 0.0, 0.0]
    assert cv.kernel_weights(3, "truncated", count=5).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    spectral = cv.kernel_weights(3, "quadratic_spectral", count=9).numpy()
    assert_allclose(spectral, [kernel("quadratic_spectral", d, 3) for d in range(1, 10)],
                    rtol=1e-12)
    assert np.abs(spectral[3:]).min() > 0
    for code, call in [("unsupported_kernel", lambda: cv.kernel_weights(2, "gaussian")),
                       ("invalid_lags", lambda: cv.kernel_weights(-1)),
                       ("invalid_lags", lambda: cv.kernel_weights(2.0))]:
        with pytest.raises(KernelError) as error:
            call()
        assert error.value.code == code


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("lags", [1, 4, 60])
def test_hac_matches_double_loop_for_one_series(small, name, lags):
    meat = cv.meat_hac(t(small), lags, name)
    expected = pair_sum(small, lambda i, j: kernel(name, abs(i - j), lags))
    assert_allclose(meat.numpy(), expected, rtol=1e-10, atol=1e-10)
    assert torch.equal(meat, meat.T)
    # The same series with an explicit period index, in shuffled row order.
    order = np.random.default_rng(lags).permutation(len(small))
    shuffled = cv.meat_hac(t(small[order]), lags, name, time=codes(order + 1990))
    assert_allclose(shuffled.numpy(), expected, rtol=1e-10, atol=1e-10)


def test_hac_matches_statsmodels_newey_west(small):
    for lags in (1, 3, 7):
        expected = sw.S_hac_simple(small, nlags=lags, weights_func=sw.weights_bartlett)
        assert_allclose(cv.meat_hac(t(small), lags).numpy(), expected, rtol=1e-11, atol=1e-11)
        uniform = sw.S_hac_simple(small, nlags=lags, weights_func=sw.weights_uniform)
        assert_allclose(cv.meat_hac(t(small), lags, "truncated").numpy(), uniform,
                        rtol=1e-11, atol=1e-11)
    assert_allclose(cv.meat_hac(t(small), 0).numpy(), cv.meat_white(t(small)).numpy(), rtol=0)
    assert_allclose(cv.meat_hac(t(small), 0, "quadratic_spectral").numpy(),
                    cv.meat_white(t(small)).numpy(), rtol=0)
    assert_allclose(cv.meat_hac(t(small[:1]), 3).numpy(), np.outer(small[0], small[0]), rtol=1e-14)


@pytest.mark.parametrize("name", KERNELS)
def test_hac_respects_period_gaps_in_any_row_order(small, name):
    rng = np.random.default_rng(12)
    # Irregular calendar: gaps of 1, 2, 3 or 9 periods, far from zero, shuffled rows.
    period = 730_000 + np.cumsum(rng.choice([1, 1, 1, 2, 3, 9], size=len(small)))
    order = rng.permutation(len(small))
    scores, period = small[order], period[order]
    for lags in (2, 5):
        meat = cv.meat_hac(t(scores), lags, name, time=codes(period))
        expected = pair_sum(scores, lambda i, j: kernel(name, abs(period[i] - period[j]), lags))
        assert_allclose(meat.numpy(), expected, rtol=1e-10, atol=1e-10)
    # Gaps matter: treating the rows as consecutive gives a different answer.
    assert not np.allclose(cv.meat_hac(t(scores), 2, name, time=codes(period)).numpy(),
                           cv.meat_hac(t(scores[np.argsort(period)]), 2, name).numpy())


@pytest.mark.parametrize("name", KERNELS)
def test_panel_hac_forms_autocovariances_within_units_only(name):
    rng = np.random.default_rng(15)
    rows = [(unit, period) for unit in (3, 4, 9, 20) for period in range(2001, 2013)
            if rng.uniform() > 0.25]                  # unbalanced with gaps
    rows.append((57, 2040))                           # a unit observed once
    unit, period = np.array(rows).T
    scores = rng.normal(size=(len(rows), 3))
    scores[1:] += 0.5 * scores[:-1]
    order = rng.permutation(len(rows))
    unit, period, scores = unit[order], period[order], scores[order]
    for lags in (1, 3, 30):
        meat = cv.meat_hac(t(scores), lags, name, time=codes(period), panel=codes(unit))
        expected = pair_sum(scores, lambda i, j: kernel(name, abs(period[i] - period[j]), lags)
                            if unit[i] == unit[j] else 0.0)
        assert_allclose(meat.numpy(), expected, rtol=1e-10, atol=1e-10)


def test_balanced_panel_hac_matches_statsmodels_panel_newey_west():
    rng = np.random.default_rng(18)
    units, periods, lags = 9, 14, 3
    scores = rng.normal(size=(units * periods, 4))
    unit = np.repeat(np.arange(units), periods)
    period = np.tile(np.arange(periods), units)
    bounds = [(start, start + periods) for start in range(0, units * periods, periods)]
    expected = sw.S_nw_panel(scores, sw.weights_bartlett(lags), bounds)
    meat = cv.meat_hac(t(scores), lags, time=codes(period), panel=codes(unit))
    assert_allclose(meat.numpy(), expected, rtol=1e-11, atol=1e-11)
    # Sorted input takes the no-sort path; a shuffled copy must agree with it.
    order = rng.permutation(len(scores))
    shuffled = cv.meat_hac(t(scores[order]), lags, time=codes(period[order]),
                           panel=codes(unit[order]))
    assert_allclose(shuffled.numpy(), expected, rtol=1e-11, atol=1e-11)
    # More lags than periods: units must still never be linked to each other.
    wide = cv.meat_hac(t(scores), 40, "truncated", time=codes(period), panel=codes(unit))
    assert_allclose(wide.numpy(), cv.meat_cluster(t(scores), codes(unit), units).numpy(),
                    rtol=1e-11, atol=1e-11)


def test_hac_validates_time_and_panel(small):
    n = len(small)
    period = np.arange(n)
    repeated = period.copy()
    repeated[5] = repeated[4]
    unit = np.repeat([0, 1], n // 2)
    cases = [
        ("repeated_time_values", lambda: cv.meat_hac(t(small), 2, time=codes(repeated))),
        ("repeated_time_values",
         lambda: cv.meat_hac(t(small), 2, time=codes(repeated), panel=codes(unit))),
        ("invalid_time", lambda: cv.meat_hac(t(small), 2, panel=codes(unit))),
        ("invalid_time", lambda: cv.meat_hac(t(small), 2, time=t(period))),
        ("invalid_time", lambda: cv.meat_hac(t(small), 2, time=codes(period[:-1]))),
        ("invalid_time", lambda: cv.meat_hac(t(small), 2, time=codes(period), panel=t(unit))),
        ("invalid_time", lambda: cv.meat_hac(t(small), 2, time=codes(period * (1 << 57)))),
        ("invalid_lags", lambda: cv.meat_hac(t(small), -1)),
        ("unsupported_kernel", lambda: cv.meat_hac(t(small), 2, "gaussian")),
        ("invalid_scores", lambda: cv.meat_hac(t(small).float(), 2)),
        ("invalid_scores", lambda: cv.meat_white(t(small[:, 0]))),
    ]
    for code, call in cases:
        with pytest.raises(KernelError) as error:
            call()
        assert error.value.code == code
    # The same period in different units is legitimate panel data.
    period_in_unit = np.tile(np.arange(n // 2), 2)
    expected = cv.meat_hac(t(small), 2, time=codes(period_in_unit), panel=codes(unit))
    # Raw unit identifiers, however wide, are renumbered rather than rejected.
    for width in (10 ** 12, 1 << 61):
        wide = cv.meat_hac(t(small), 2, time=codes(period_in_unit), panel=codes(unit * width))
        assert_allclose(wide.numpy(), expected.numpy(), rtol=1e-13)


@pytest.mark.parametrize("name", KERNELS)
def test_driscoll_kraay_matches_double_loop(name):
    rng = np.random.default_rng(22)
    n = 70
    period = rng.choice([1, 2, 3, 5, 6, 10, 11, 12, 20], size=n)     # gaps between periods
    scores = rng.normal(size=(n, 3)) + rng.normal(size=(21, 3))[period]
    for lags in (1, 4):
        meat = cv.meat_driscoll_kraay(t(scores), codes(period), lags, name)
        expected = pair_sum(scores, lambda i, j: kernel(name, abs(period[i] - period[j]), lags))
        assert_allclose(meat.numpy(), expected, rtol=1e-10, atol=1e-10)
    # lags = 0 is clustering on the period, for every kernel.
    labels = np.unique(period, return_inverse=True)[1]
    clustered = cv.meat_cluster(t(scores), codes(labels), int(labels.max()) + 1).numpy()
    assert_allclose(cv.meat_driscoll_kraay(t(scores), codes(period), 0, name).numpy(), clustered,
                    rtol=1e-12, atol=1e-12)
    assert_allclose(clustered, pair_sum(scores, lambda i, j: period[i] == period[j]),
                    rtol=1e-10, atol=1e-10)


def test_driscoll_kraay_matches_statsmodels_groupsum():
    rng = np.random.default_rng(25)
    units, periods, lags = 11, 16, 3
    period = np.tile(np.arange(periods), units)
    scores = rng.normal(size=(units * periods, 3)) + rng.normal(size=(periods, 3))[period]
    expected = sw.S_hac_groupsum(scores, period, nlags=lags)
    assert_allclose(cv.meat_driscoll_kraay(t(scores), codes(period), lags).numpy(), expected,
                    rtol=1e-11, atol=1e-11)
    with pytest.raises(KernelError) as error:
        cv.meat_driscoll_kraay(t(scores), t(period), lags)
    assert error.value.code == "invalid_time"


def test_lag_rule_factors_and_sandwich():
    for n in (1, 50, 100, 101, 500, 10_000, 1_000_000):
        assert cv.newey_west_lags(n) == int(np.floor(4 * (n / 100) ** (2 / 9)))
    assert cv.newey_west_lags(100) == 4 and cv.newey_west_lags(1000) == 6
    with pytest.raises(KernelError):
        cv.newey_west_lags(0)

    assert_allclose(cv.cluster_factor(100, 5, 10), 10 / 9 * 99 / 95, rtol=1e-15)
    with pytest.raises(KernelError) as error:
        cv.cluster_factor(100, 5, 1)
    assert error.value.code == "insufficient_clusters"
    with pytest.raises(KernelError) as error:
        cv.cluster_factor(5, 5, 3)
    assert error.value.code == "insufficient_observations"

    rng = np.random.default_rng(28)
    bread = rng.normal(size=(3, 3))
    root = rng.normal(size=(3, 3))
    meat = root @ root.T
    result = cv.sandwich(t(bread), t(meat))
    assert_allclose(result.numpy(), bread @ meat @ bread.T, rtol=1e-12, atol=1e-12)
    assert torch.equal(result, result.T)
    wide = rng.normal(size=(2, 3))                    # GMM-style rectangular bread
    assert_allclose(cv.sandwich(t(wide), t(meat)).numpy(), wide @ meat @ wide.T, rtol=1e-12)


def test_meats_are_ordinary_float64_tensors(small):
    period = codes(np.arange(len(small)))
    group = codes(np.arange(len(small)) % 5)
    results = [cv.meat_white(t(small)), cv.meat_cluster(t(small), group, 5),
               cv.meat_hac(t(small), 3), cv.meat_hac(t(small), 3, time=period),
               cv.meat_driscoll_kraay(t(small), period, 2),
               cv.meat_multiway(t(small), [(group, 5), (period, len(small))]).meat]
    for meat in results:
        assert meat.dtype == torch.float64 and meat.shape == (3, 3)
        assert not meat.requires_grad and not meat.is_inference()
