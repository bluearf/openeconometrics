import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.stats import norm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.dependent_resampling import advanced_intervals, multipliers


def pctile(a, p):
    a = np.sort(a)
    k = len(a) * p
    if abs(k - round(k)) < 1e-9 * max(1, k):
        lo, hi = np.clip([round(k), round(k) + 1], 1, len(a)).astype(int)
        return (a[lo - 1] + a[hi - 1]) / 2
    return a[min(max(int(np.ceil(k)), 1), len(a)) - 1]


def data():
    rng = np.random.default_rng(23)
    n = 72
    x = rng.normal(size=n)
    e = rng.normal(size=n)
    for i in range(1, n):
        e[i] += 0.6 * e[i - 1]
    return pd.DataFrame(dict(x=x, y=1 + 0.5 * x + e, t=np.arange(n), g=np.arange(n) // 12))


@pytest.mark.parametrize("scheme", ["moving_block", "residual", "residual_block", "wild_cluster"])
def test_dependence_draws_ols_covariance_against_numpy_reference(scheme):
    df = data()
    fit = oe.ols(data=df, y="y", x=["x"])
    options = (
        dict(cluster="g")
        if scheme == "wild_cluster"
        else dict(time="t", block_length=6)
        if "block" in scheme
        else {}
    )
    boot = oe.bootstrap(fit, df, reps=99, seed=32, scheme=scheme, ci="percentile", **options)
    x = np.column_stack([np.ones(len(df)), df.x])
    y = df.y.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    errors = y - x @ beta
    errors -= errors.mean() if scheme != "wild_cluster" else 0
    gen = torch.Generator().manual_seed(32)
    if scheme == "wild_cluster":
        wild = torch.randint(2, (99, 6), generator=gen).numpy() * 2 - 1
    draws = []
    for i in range(99):
        if "block" in scheme:
            starts = torch.randint(67, (12,), generator=gen).numpy()
            rows = np.concatenate([np.arange(s, s + 6) for s in starts])
        elif scheme == "residual":
            rows = torch.randint(72, (72,), generator=gen).numpy()
        if scheme == "moving_block":
            b = np.linalg.lstsq(x[rows], y[rows], rcond=None)[0]
        else:
            eps = errors * wild[i, df.g] if scheme == "wild_cluster" else errors[rows]
            b = np.linalg.lstsq(x, x @ beta + eps, rcond=None)[0]
        draws.append(b)
    draws = np.array(draws)
    assert_allclose(boot.covariance_matrix, np.cov(draws, rowvar=False), rtol=1e-9, atol=1e-12)
    for j, term in enumerate(["Intercept", "x"]):
        bounds = boot.extra["bootstrap_ci"]["intervals"][term]
        assert_allclose(
            [bounds["ci_low"], bounds["ci_high"]],
            [pctile(draws[:, j], 0.025), pctile(draws[:, j], 0.975)],
            rtol=1e-9,
        )
    assert boot.extra["bootstrap"]["residual_model"] == "unrestricted"


@pytest.mark.parametrize("kind", ["bca", "studentized"])
def test_advanced_intervals_against_independent_formula(kind):
    rng = np.random.default_rng(818)
    y = rng.exponential(size=35)
    indices = rng.integers(len(y), size=(399, len(y)))
    draws = y[indices].mean(axis=1)
    errors = y[indices].std(axis=1, ddof=1) / np.sqrt(len(y))
    observed = y.mean()
    original_error = y.std(ddof=1) / np.sqrt(len(y))
    jack = np.array([np.delete(y, j).mean() for j in range(len(y))])
    intervals, details = advanced_intervals(
        torch.tensor(draws[:, None]),
        torch.tensor([observed]),
        0.05,
        kind,
        errors=torch.tensor(errors[:, None]),
        original_errors=torch.tensor([original_error]),
        jack=torch.tensor(jack[:, None]),
    )
    if kind == "studentized":
        pivots = (draws - observed) / errors
        expected = [
            observed - pctile(pivots, 0.975) * original_error,
            observed - pctile(pivots, 0.025) * original_error,
        ]
    else:
        u = jack.mean() - jack
        acceleration = (u**3).sum() / (6 * (u**2).sum() ** 1.5)
        z0 = norm.ppf(np.mean(draws <= observed))
        z = z0 + norm.ppf([0.025, 0.975])
        prob = norm.cdf(z0 + z / (1 - acceleration * z))
        expected = [pctile(draws, q) for q in prob]
        assert_allclose(details["acceleration"], [acceleration], atol=1e-12)
    assert_allclose(intervals[0], expected, rtol=1e-11)


def test_local_rng_bca_refits_budget_and_generic_iid_time_guard():
    df = data()
    fit = oe.ols(data=df, y="y", x=["x"])
    torch.manual_seed(87)
    before = torch.get_rng_state().clone()
    a = oe.bootstrap(fit, df, reps=30, seed=8, ci="bca")
    assert torch.equal(before, torch.get_rng_state())
    b = oe.bootstrap(fit, df, reps=30, seed=8, ci="bca")
    assert a.covariance_matrix == b.covariance_matrix
    assert a.extra["bootstrap_ci"]["jackknife_refits"] == len(df)
    with pytest.raises(AnalysisError) as exc:
        oe.bootstrap(fit, df, reps=30, seed=8, ci="bca", max_refits=100)
    assert exc.value.code == "bootstrap_too_large"
    series_fit = oe.ols(data=df, y="y", x=["x"], time="t", lags=3, covariance="hac")
    with pytest.raises(AnalysisError) as exc:
        oe.bootstrap(series_fit, df, reps=10, seed=8)
    assert exc.value.code == "bootstrap_unsupported"
    assert (
        oe.bootstrap(
            series_fit, df, reps=20, seed=8, scheme="moving_block", block_length=6, ci="studentized"
        ).extra["bootstrap_ci"]["type"]
        == "studentized"
    )


def test_restricted_wild_bootstrap_t_and_cluster_studentization():
    df = data()
    fit = oe.ols(data=df, y="y", x=["x"], covariance="cluster", cluster="g")
    result = oe.bootstrap(
        fit, df, reps=99, seed=99, scheme="wild_cluster", cluster="g", null={"x": 0}
    )
    assert result.extra["bootstrap"]["residual_model"] == "restricted"
    x = np.column_stack([np.ones(len(df)), df.x])
    y = df.y.to_numpy()
    restricted = np.array([y.mean(), 0.0])
    resid = y - x @ restricted
    gen = torch.Generator().manual_seed(99)
    weights = torch.randint(2, (99, 6), generator=gen).numpy() * 2 - 1
    t = []
    inv = np.linalg.inv(x.T @ x)
    for w in weights:
        yy = x @ restricted + resid * w[df.g]
        b = np.linalg.lstsq(x, yy, rcond=None)[0]
        scores = x * (yy - x @ b)[:, None]
        sums = np.zeros((6, 2))
        np.add.at(sums, df.g, scores)
        # OLS cluster CR1 has both G/(G-1) and (N-1)/(N-K).
        v = inv @ (sums.T @ sums) @ inv * 6 / 5 * 71 / 70
        t.append(b[1] / np.sqrt(v[1, 1]))
    observed = fit.coefficients[1].estimate / fit.coefficients[1].std_error
    assert_allclose(
        result.extra["bootstrap_test"]["tests"]["x"]["p_value"], np.mean(np.abs(t) >= abs(observed))
    )
    assert (
        oe.bootstrap(
            fit,
            df,
            reps=30,
            seed=9,
            scheme="wild_cluster",
            cluster="g",
            wild="webb",
            ci="studentized",
        ).extra["bootstrap"]["wild"]
        == "webb"
    )


def test_studentized_interval_simulation_coverage_matches_reference():
    # Fixed-seed coverage diagnostic, not a claim of universal nominal coverage.
    rng = np.random.default_rng(74)
    native_covered = reference_covered = 0
    for _ in range(120):
        y = rng.normal(size=40)
        sample = y[rng.integers(40, size=(199, 40))]
        b, se = sample.mean(axis=1), sample.std(axis=1, ddof=1) / np.sqrt(40)
        original_se = y.std(ddof=1) / np.sqrt(40)
        bounds, _ = advanced_intervals(
            torch.tensor(b[:, None]),
            torch.tensor([y.mean()]),
            0.05,
            "studentized",
            errors=torch.tensor(se[:, None]),
            original_errors=torch.tensor([original_se]),
        )
        pivots = (b - y.mean()) / se
        lo, hi = (
            y.mean() - pctile(pivots, 0.975) * original_se,
            y.mean() - pctile(pivots, 0.025) * original_se,
        )
        native_covered += bounds[0][0] <= 0 <= bounds[0][1]
        reference_covered += lo <= 0 <= hi
    assert native_covered == reference_covered
    assert 100 <= native_covered <= 120


@pytest.mark.parametrize("kind", ["rademacher", "normal", "webb"])
def test_shared_multiplier_kernel_center_scale_and_bounds(kind):
    draws = multipliers(10, 500, 8, kind)
    assert abs(float(draws.mean())) < 0.06
    assert abs(float(draws.square().mean()) - 1) < 0.06
    with pytest.raises(AnalysisError):
        multipliers(10000, 10000, 1, kind)


def test_serial_block_interval_coverage_matches_independent_ols_simulation():
    rng = np.random.default_rng(804)
    covered = reference_covered = 0
    for world in range(40):
        n, length, reps = 80, 8, 99
        x = rng.normal(size=n)
        errors = rng.normal(size=n)
        for i in range(1, n):
            errors[i] += 0.65 * errors[i - 1]
        df = pd.DataFrame(dict(x=x, y=1 + 0.5 * x + errors, t=np.arange(n)))
        fit = oe.ols(data=df, y="y", x=["x"])
        boot = oe.bootstrap(
            fit,
            df,
            reps=reps,
            seed=world,
            scheme="moving_block",
            time="t",
            block_length=length,
            ci="percentile",
        )
        generator = torch.Generator().manual_seed(world)
        design = np.column_stack([np.ones(n), x])
        draws = []
        for _ in range(reps):
            starts = torch.randint(n - length + 1, (n // length,), generator=generator).numpy()
            positions = np.concatenate([np.arange(start, start + length) for start in starts])
            draws.append(
                np.linalg.lstsq(design[positions], df.y.to_numpy()[positions], rcond=None)[0]
            )
        bounds = boot.extra["bootstrap_ci"]["intervals"]["Intercept"]
        lo, hi = pctile(np.array(draws)[:, 0], 0.025), pctile(np.array(draws)[:, 0], 0.975)
        assert_allclose([bounds["ci_low"], bounds["ci_high"]], [lo, hi], atol=1e-10)
        covered += bounds["ci_low"] <= 1 <= bounds["ci_high"]
        reference_covered += lo <= 1 <= hi
    assert covered == reference_covered
    # This checks reference agreement; it is not a nominal-coverage guarantee.


def test_advanced_failed_replicates_and_second_resampling_guard():
    df = data().iloc[:60].copy()
    df["rare"] = 0.0
    df.loc[[3, 17, 41], "rare"] = 1.0
    fit = oe.ols(data=df, y="y", x=["x", "rare"])
    boot = oe.bootstrap(fit, df, reps=200, seed=31, ci="studentized")
    assert 0 < boot.inference["failed_replicates"] <= 20
    assert boot.inference["failure_reasons"]
    with pytest.raises(AnalysisError) as exc:
        oe.bootstrap(boot, df, reps=20, seed=8, scheme="residual")
    assert exc.value.code == "bootstrap_unsupported"
    df["rare"] = 0.0
    df.loc[5, "rare"] = 1.0
    fit = oe.ols(data=df, y="y", x=["x", "rare"])
    with pytest.raises(AnalysisError) as exc:
        oe.bootstrap(fit, df, reps=50, seed=1, ci="studentized")
    assert exc.value.code == "bootstrap_failed"


def test_few_cluster_wild_t_interval_coverage_matches_numpy_reference():
    rng = np.random.default_rng(502)
    covered = reference_covered = 0
    groups = np.repeat(np.arange(6), 10)
    for world in range(24):
        x = rng.normal(size=60)
        y = 1 + 0.5 * x + rng.normal(size=6)[groups] + rng.normal(size=60)
        df = pd.DataFrame(dict(x=x, y=y, g=groups))
        fit = oe.ols(data=df, y="y", x=["x"], covariance="cluster", cluster="g")
        result = oe.bootstrap(
            fit, df, scheme="wild_cluster", cluster="g", ci="studentized", reps=63, seed=world
        )
        design = np.column_stack([np.ones(60), x])
        inverse = np.linalg.inv(design.T @ design)
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        resid = y - design @ beta

        def se(target, b):
            scores = design * (target - design @ b)[:, None]
            cluster_scores = np.zeros((6, 2))
            np.add.at(cluster_scores, groups, scores)
            variance = inverse @ (cluster_scores.T @ cluster_scores) @ inverse * 6 / 5 * 59 / 58
            return np.sqrt(variance[1, 1])

        original_error = se(y, beta)
        weights = (
            torch.randint(2, (63, 6), generator=torch.Generator().manual_seed(world)).numpy() * 2
            - 1
        )
        pivots = []
        for weight in weights:
            target = design @ beta + resid * weight[groups]
            b = np.linalg.lstsq(design, target, rcond=None)[0]
            pivots.append((b[1] - beta[1]) / se(target, b))
        lo = beta[1] - pctile(pivots, 0.975) * original_error
        hi = beta[1] - pctile(pivots, 0.025) * original_error
        bounds = result.extra["bootstrap_ci"]["intervals"]["x"]
        assert_allclose([bounds["ci_low"], bounds["ci_high"]], [lo, hi], atol=1e-10)
        covered += bounds["ci_low"] <= 0.5 <= bounds["ci_high"]
        reference_covered += lo <= 0.5 <= hi
    assert covered == reference_covered


@pytest.mark.parametrize("error", [None, float("nan"), float("inf"), 0.0])
def test_studentized_original_standard_error_must_be_positive_finite(error):
    df = data()
    fit = oe.ols(data=df, y="y", x=["x"])
    coefficients = [
        fit.coefficients[0].model_copy(update={"std_error": error}),
        fit.coefficients[1],
    ]
    invalid = fit.model_copy(update={"coefficients": coefficients})
    with pytest.raises(AnalysisError) as exc:
        oe.bootstrap(invalid, df, reps=10, seed=8, ci="studentized")
    assert exc.value.code == "bootstrap_failed"
