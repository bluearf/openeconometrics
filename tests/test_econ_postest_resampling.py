"""Independent oracles for oe.bootstrap and oe.jackknife (post-estimation family).

The replicate draws are regenerated from the same seed with the family's draw
kernels; every replicate is then re-estimated independently in NumPy (least
squares, the within estimator with relabelled panels, weighted least squares)
or statsmodels (logit), and the covariance, percentile and bias-corrected
intervals are recomputed from those replicates. The jackknife is compared with
explicit delete-one loops, and the frequency-weighted jackknife with the
delete-one jackknife of the expanded data.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.postest.draws import (
    Clusters, Strata, WeightedStrata, draw_units, draw_weighted, stata_percentile,
)
from openecon.models import ResultBundle


def make_data(seed=11, n=150):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    group = rng.integers(0, 20, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "group": group,
                          "stratum": (np.arange(n) % 3)})
    shock = rng.normal(size=20)[group]
    frame["y"] = 1 + 0.5 * x1 - 0.3 * x2 + shock + rng.normal(size=n) * (1 + 0.5 * np.abs(x1))
    frame["binary"] = (0.3 + x1 - 0.5 * x2 + rng.logistic(size=n) > 0).astype(int)
    frame["f"] = rng.integers(1, 4, size=n)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def ols_numpy(frame, columns, y="y", weights=None):
    x = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(float)])
    target = frame[y].to_numpy(float)
    if weights is not None:
        root = np.sqrt(weights)
        return np.linalg.lstsq(x * root[:, None], target * root, rcond=None)[0]
    return np.linalg.lstsq(x, target, rcond=None)[0]


def observation_draws(n, reps, seed, strata=None, size=None):
    codes = torch.zeros(n, dtype=torch.int64) if strata is None else torch.as_tensor(strata)
    table = Strata.build(codes, int(codes.max()) + 1)
    sizes = table.count if size is None else torch.full_like(table.count, size)
    generator = torch.Generator().manual_seed(seed)
    return [draw_units(table, sizes, generator).numpy() for _ in range(reps)]


def stata_percentile_numpy(values, p):
    ordered = np.sort(values)
    r = len(ordered)
    position = r * p
    if abs(position - round(position)) < 1e-9:
        lower = min(max(int(round(position)), 1), r)
        upper = min(lower + 1 if round(position) >= 1 else 1, r)
        return (ordered[lower - 1] + ordered[upper - 1]) / 2
    return ordered[int(np.ceil(position)) - 1]


def test_bootstrap_ols_matches_numpy_loop_and_intervals(data):
    fit = oe.ols(data=data, y="y", x=["x1", "x2"])
    reps, seed = 120, 2024
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed, ci="percentile")
    replicates = np.array([ols_numpy(data.iloc[rows], ["x1", "x2"])
                           for rows in observation_draws(len(data), reps, seed)])
    covariance = np.cov(replicates, rowvar=False, ddof=1)
    assert_allclose(np.array(boot.covariance_matrix), covariance, rtol=1e-9, atol=1e-14)
    se = np.sqrt(np.diag(covariance))
    estimates = np.array([c.estimate for c in fit.coefficients])
    assert_allclose([c.estimate for c in boot.coefficients], estimates)
    assert_allclose([c.std_error for c in boot.coefficients], se, rtol=1e-9)
    z = estimates / se
    assert_allclose([c.statistic for c in boot.coefficients], z, rtol=1e-9)
    assert_allclose([c.p_value for c in boot.coefficients], 2 * stats.norm.sf(np.abs(z)),
                    rtol=1e-8)
    crit = stats.norm.ppf(0.975)
    assert_allclose([c.ci_low for c in boot.coefficients], estimates - crit * se, rtol=1e-9)
    intervals = boot.extra["bootstrap_ci"]["intervals"]
    assert boot.extra["bootstrap_ci"]["type"] == "percentile"
    for j, term in enumerate(["Intercept", "x1", "x2"]):
        assert_allclose(intervals[term]["ci_low"],
                        stata_percentile_numpy(replicates[:, j], 0.025), rtol=1e-12)
        assert_allclose(intervals[term]["ci_high"],
                        stata_percentile_numpy(replicates[:, j], 0.975), rtol=1e-12)
    bias = boot.extra["bootstrap"]["bias"]
    assert_allclose([bias[t] for t in ["Intercept", "x1", "x2"]],
                    replicates.mean(axis=0) - estimates, atol=1e-12)
    assert boot.inference["covariance"] == "bootstrap"
    assert boot.inference["use_t"] is False and boot.inference["df_inference"] is None
    assert boot.inference["reps"] == reps and boot.inference["seed"] == seed
    assert boot.inference["original_covariance"] == "nonrobust"
    assert boot.provenance["postestimation"]["method"] == "bootstrap"
    assert "inference_details" not in boot.provenance
    assert boot.tests["model"]["distribution"] == "chi2"
    assert type(boot) is ResultBundle


def test_bootstrap_bias_corrected_interval(data):
    fit = oe.ols(data=data, y="y", x=["x1"])
    reps, seed = 150, 9
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed, ci="bc", alpha=0.1)
    replicates = np.array([ols_numpy(data.iloc[rows], ["x1"])
                           for rows in observation_draws(len(data), reps, seed)])
    estimates = np.array([c.estimate for c in fit.coefficients])
    z = stats.norm.ppf(0.95)
    for j, term in enumerate(["Intercept", "x1"]):
        z0 = stats.norm.ppf(np.mean(replicates[:, j] <= estimates[j]))
        p1, p2 = stats.norm.cdf(2 * z0 - z), stats.norm.cdf(2 * z0 + z)
        interval = boot.extra["bootstrap_ci"]["intervals"][term]
        assert_allclose(interval["ci_low"], stata_percentile_numpy(replicates[:, j], p1))
        assert_allclose(interval["ci_high"], stata_percentile_numpy(replicates[:, j], p2))
    assert boot.spec.alpha == 0.1
    assert boot.inference["confidence_level"] == pytest.approx(0.9)
    assert ResultBundle.model_validate_json(boot.model_dump_json()) == boot


def test_bootstrap_is_deterministic_with_a_seed(data):
    fit = oe.logit(data=data, y="binary", x=["x1", "x2"])
    a = oe.bootstrap(fit, data, reps=30, seed=5)
    b = oe.bootstrap(fit, data, reps=30, seed=5)
    c = oe.bootstrap(fit, data, reps=30, seed=6)
    assert a.covariance_matrix == b.covariance_matrix
    assert a.covariance_matrix != c.covariance_matrix
    unseeded = oe.bootstrap(fit, data, reps=10)
    assert isinstance(unseeded.inference["seed"], int)
    replay = oe.bootstrap(fit, data, reps=10, seed=unseeded.inference["seed"])
    assert replay.covariance_matrix == unseeded.covariance_matrix


def test_bootstrap_logit_against_statsmodels(data):
    fit = oe.logit(data=data, y="binary", x=["x1", "x2"])
    reps, seed = 25, 3
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed)
    replicates = []
    for rows in observation_draws(len(data), reps, seed):
        sample = data.iloc[rows]
        model = sm.Logit(sample["binary"], sm.add_constant(sample[["x1", "x2"]]))
        replicates.append(model.fit(disp=0, tol=1e-12, maxiter=100).params.to_numpy())
    assert_allclose(np.array(boot.covariance_matrix), np.cov(np.array(replicates), rowvar=False),
                    rtol=1e-6)


def test_bootstrap_strata_and_size(data):
    fit = oe.ols(data=data, y="y", x=["x1"])
    strata = data["stratum"].to_numpy()
    draws = observation_draws(len(data), 40, 1, strata=strata, size=30)
    for rows in draws:
        assert np.bincount(strata[rows], minlength=3).tolist() == [30, 30, 30]
    boot = oe.bootstrap(fit, data, reps=40, seed=1, strata="stratum", size=30)
    replicates = np.array([ols_numpy(data.iloc[rows], ["x1"]) for rows in draws])
    assert_allclose(np.array(boot.covariance_matrix), np.cov(replicates, rowvar=False),
                    rtol=1e-9)
    assert boot.inference["strata"] == "stratum" and boot.inference["size"] == 30
    with pytest.raises(AnalysisError) as error:
        oe.bootstrap(fit, data, reps=5, seed=1, strata="stratum", size=51)
    assert error.value.code == "invalid_spec"


def cluster_draws(codes, reps, seed):
    codes = torch.as_tensor(codes, dtype=torch.int64)
    count = int(codes.max()) + 1
    table = Strata.build(torch.zeros(count, dtype=torch.int64), 1)
    clusters = Clusters.build(codes, count)
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(reps):
        rows, slot = clusters.expand(draw_units(table, table.count, generator))
        out.append((rows.numpy(), slot.numpy()))
    return out


def test_cluster_draws_keep_clusters_intact(data):
    codes = pd.factorize(data["group"])[0]
    for rows, slot in cluster_draws(codes, 10, 4):
        for s in np.unique(slot):
            members = rows[slot == s]
            cluster = codes[members[0]]
            assert np.all(codes[members] == cluster)
            assert sorted(members) == sorted(np.flatnonzero(codes == cluster))
        assert len(np.unique(slot)) == codes.max() + 1


def test_cluster_bootstrap_ols(data):
    fit = oe.ols(data=data, y="y", x=["x1", "x2"])
    boot = oe.bootstrap(fit, data, reps=60, seed=8, cluster="group")
    codes = pd.factorize(data["group"])[0]
    replicates = np.array([ols_numpy(data.iloc[rows], ["x1", "x2"])
                           for rows, _ in cluster_draws(codes, 60, 8)])
    assert_allclose(np.array(boot.covariance_matrix), np.cov(replicates, rowvar=False),
                    rtol=1e-9)
    assert boot.inference["cluster_count"] == 20
    assert boot.inference["resampling_unit"] == "clusters of 'group'"


def within_numpy(frame, panel, columns, y="y"):
    x = frame[columns].to_numpy(float)
    target = frame[y].to_numpy(float)
    ids = pd.factorize(np.asarray(panel))[0]
    xd = x - pd.DataFrame(x).groupby(ids).transform("mean").to_numpy()
    yd = target - pd.Series(target).groupby(ids).transform("mean").to_numpy()
    beta = np.linalg.lstsq(xd, yd, rcond=None)[0]
    intercept = target.mean() - x.mean(axis=0) @ beta
    return np.r_[intercept, beta]


def test_panel_bootstrap_relabels_resampled_panels(data):
    fit = oe.xtreg(data=data, y="y", x=["x1", "x2"], panel="group")
    boot = oe.bootstrap(fit, data, reps=40, seed=13)
    assert boot.inference["resampling_unit"] == "clusters of 'group'"
    codes = pd.factorize(data["group"])[0]
    replicates = []
    for rows, slot in cluster_draws(codes, 40, 13):
        sample = data.iloc[rows].reset_index(drop=True)
        replicates.append(within_numpy(sample, slot, ["x1", "x2"]))
    assert_allclose(np.array(boot.covariance_matrix), np.cov(np.array(replicates), rowvar=False),
                    rtol=1e-8)
    # With a time variable, unrelabelled copies of a panel would repeat its periods and every
    # refit would fail (repeated time values); relabelled copies are distinct panels.
    timed = data.assign(t=data.groupby("group").cumcount())
    fit_t = oe.xtreg(data=timed, y="y", x=["x1", "x2"], panel="group", time="t")
    boot_t = oe.bootstrap(fit_t, timed, reps=40, seed=13)
    assert boot_t.inference["failed_replicates"] == 0
    assert_allclose(boot_t.covariance_matrix, boot.covariance_matrix, rtol=1e-8)


def test_panel_bootstrap_uses_whole_panels_for_dynamic_models():
    rng = np.random.default_rng(3)
    n_id, periods = 40, 7
    frame = pd.DataFrame({"id": np.repeat(np.arange(n_id), periods),
                          "t": np.tile(np.arange(periods), n_id)})
    frame["x"] = rng.normal(size=len(frame))
    y = np.zeros(len(frame))
    for i in range(n_id):
        for t in range(1, periods):
            k = i * periods + t
            y[k] = 0.5 * y[k - 1] + frame["x"].iloc[k] + rng.normal()
    frame["y"] = y
    fit = oe.xtabond(data=frame, y="y", x=["x"], panel="id", time="t")
    boot = oe.bootstrap(fit, frame, reps=20, seed=1)
    assert boot.inference["failed_replicates"] == 0
    assert boot.extra["bootstrap"]["population"].startswith("all rows of the panels")
    assert all(c.std_error > 0 for c in boot.coefficients)


def test_fweight_bootstrap_draws_expanded_units(data):
    fit = oe.ols(data=data, y="y", x=["x1"], weights="f", weight_type="fweight")
    reps, seed = 40, 21
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed)
    f = torch.as_tensor(data["f"].to_numpy(float))
    table = WeightedStrata.build(torch.zeros(len(data), dtype=torch.int64), 1, f)
    sizes = torch.tensor([int(f.sum())])
    generator = torch.Generator().manual_seed(seed)
    replicates = []
    for _ in range(reps):
        counts = draw_weighted(table, sizes, len(data), generator).numpy()
        assert counts.sum() == int(f.sum())
        keep = counts > 0
        replicates.append(ols_numpy(data[keep], ["x1"], weights=counts[keep].astype(float)))
    assert_allclose(np.array(boot.covariance_matrix), np.cov(np.array(replicates), rowvar=False),
                    rtol=1e-9)
    # Draw probabilities are proportional to the frequency weights.
    generator = torch.Generator().manual_seed(0)
    total = sum(draw_weighted(table, sizes, len(data), generator).numpy() for _ in range(300))
    expected = 300 * data["f"].to_numpy()
    assert stats.chisquare(total, expected * total.sum() / expected.sum()).pvalue > 1e-3
    assert boot.inference["resampling_unit"] == "frequency-weighted observations"


def test_failed_replicates_are_counted(data):
    frame = data.iloc[:60].reset_index(drop=True).copy()
    frame["rare"] = 0.0
    frame.loc[[3, 17, 41], "rare"] = 1.0
    fit = oe.ols(data=frame, y="y", x=["x1", "rare"])
    reps, seed = 200, 31
    boot = oe.bootstrap(fit, frame, reps=reps, seed=seed)
    draws = observation_draws(len(frame), reps, seed)
    degenerate = [frame["rare"].iloc[rows].nunique() < 2 for rows in draws]
    expected_failed = int(np.sum(degenerate))
    assert expected_failed > 0
    assert boot.inference["failed_replicates"] == expected_failed
    assert boot.inference["reps_completed"] == reps - expected_failed
    assert any("bootstrap replications failed" in w for w in boot.warnings)
    good = np.array([ols_numpy(frame.iloc[rows], ["x1", "rare"])
                     for rows, bad in zip(draws, degenerate, strict=True) if not bad])
    assert_allclose(np.array(boot.covariance_matrix), np.cov(good, rowvar=False), rtol=1e-8)


def test_too_many_failures_raise(data):
    frame = data.iloc[:40].reset_index(drop=True).copy()
    frame["rare"] = 0.0
    frame.loc[5, "rare"] = 1.0
    fit = oe.ols(data=frame, y="y", x=["x1", "rare"])
    with pytest.raises(AnalysisError) as error:
        oe.bootstrap(fit, frame, reps=50, seed=1)
    assert error.value.code == "bootstrap_failed"
    assert "failed" in str(error.value)


def test_missing_rows_and_collinear_terms(data):
    frame = data.copy()
    frame.loc[:7, "x2"] = np.nan
    frame["x3"] = frame["x1"] + frame["x2"]
    fit = oe.poisson(data=frame.assign(count=(frame["y"] > 1).astype(int)), y="count",
                     x=["x1", "x2", "x3"], missing="drop")
    assert [c.term for c in fit.coefficients] == ["Intercept", "x1", "x2"]
    boot = oe.bootstrap(fit, frame.assign(count=(frame["y"] > 1).astype(int)), reps=20, seed=2)
    assert boot.nobs == len(frame) - 8
    assert [c.term for c in boot.coefficients] == ["Intercept", "x1", "x2"]
    assert boot.inference["failed_replicates"] == 0


def test_jackknife_ols_matches_delete_one_loop(data):
    fit = oe.ols(data=data, y="y", x=["x1", "x2"])
    jack = oe.jackknife(fit, data)
    n = len(data)
    replicates = np.array([ols_numpy(data.drop(index=i), ["x1", "x2"]) for i in range(n)])
    centered = replicates - replicates.mean(axis=0)
    covariance = (n - 1) / n * centered.T @ centered
    assert_allclose(np.array(jack.covariance_matrix), covariance, rtol=1e-9)
    se = np.sqrt(np.diag(covariance))
    estimates = np.array([c.estimate for c in fit.coefficients])
    t = estimates / se
    assert_allclose([c.p_value for c in jack.coefficients], 2 * stats.t.sf(np.abs(t), n - 1),
                    rtol=1e-8)
    crit = stats.t.ppf(0.975, n - 1)
    assert_allclose([c.ci_high for c in jack.coefficients], estimates + crit * se, rtol=1e-9)
    assert jack.inference["use_t"] is True and jack.inference["df_inference"] == n - 1
    bias = jack.extra["jackknife"]["bias"]
    assert_allclose(bias["x1"], (n - 1) * (replicates[:, 1].mean() - estimates[1]), atol=1e-12)
    assert jack.tests["model"]["distribution"] == "F"
    assert jack.tests["model"]["df2"] == n - 1
    assert ResultBundle.model_validate_json(jack.model_dump_json()) == jack


def test_jackknife_cluster_and_logit(data):
    fit = oe.logit(data=data, y="binary", x=["x1"])
    jack = oe.jackknife(fit, data, cluster="group")
    replicates = []
    groups = pd.unique(data["group"])
    for g in groups:
        sample = data[data["group"] != g]
        model = sm.Logit(sample["binary"], sm.add_constant(sample[["x1"]]))
        replicates.append(model.fit(disp=0, tol=1e-12).params.to_numpy())
    replicates = np.array(replicates)
    g = len(groups)
    centered = replicates - replicates.mean(axis=0)
    assert_allclose(np.array(jack.covariance_matrix), (g - 1) / g * centered.T @ centered,
                    rtol=1e-6)
    assert jack.inference["df_inference"] == g - 1
    assert jack.inference["cluster_count"] == g


def test_jackknife_fweights_equal_expanded_data(data):
    small = data.iloc[:40].reset_index(drop=True)
    fit = oe.ols(data=small, y="y", x=["x1"], weights="f", weight_type="fweight")
    jack = oe.jackknife(fit, small)
    expanded = small.loc[small.index.repeat(small["f"])].reset_index(drop=True)
    n = len(expanded)
    replicates = np.array([ols_numpy(expanded.drop(index=i), ["x1"]) for i in range(n)])
    centered = replicates - replicates.mean(axis=0)
    assert_allclose(np.array(jack.covariance_matrix), (n - 1) / n * centered.T @ centered,
                    rtol=1e-9)
    assert jack.inference["df_inference"] == n - 1
    assert jack.inference["refits"] == len(small)


def test_jackknife_panel_deletes_whole_panels(data):
    fit = oe.xtreg(data=data, y="y", x=["x1"], panel="group")
    jack = oe.jackknife(fit, data)
    groups = pd.unique(data["group"])
    replicates = np.array([within_numpy(data[data["group"] != g], data.loc[data["group"] != g,
                                                                          "group"], ["x1"])
                           for g in groups])
    g = len(groups)
    centered = replicates - replicates.mean(axis=0)
    assert_allclose(np.array(jack.covariance_matrix), (g - 1) / g * centered.T @ centered,
                    rtol=1e-8)


def test_resampling_error_codes(data):
    fit = oe.ols(data=data, y="y", x=["x1"])
    cases = [
        (lambda: oe.bootstrap(fit, data, reps=1), "invalid_spec"),
        (lambda: oe.bootstrap(fit, data, reps=10, ci="unknown"), "invalid_spec"),
        (lambda: oe.bootstrap(fit, data, reps=10, seed=-1), "invalid_spec"),
        (lambda: oe.bootstrap(fit, data, reps=10, alpha=1.5), "invalid_spec"),
        (lambda: oe.bootstrap(fit, data, reps=10, cluster="nope"), "missing_columns"),
        (lambda: oe.bootstrap(fit, data.assign(y=data["y"] + 1), reps=10), "data_mismatch"),
        (lambda: oe.bootstrap(fit, data.iloc[1:], reps=10), "data_mismatch"),
        (lambda: oe.bootstrap(fit, data.drop(columns="x1"), reps=10), "data_mismatch"),
        (lambda: oe.bootstrap("fit", data), "invalid_result"),
        (lambda: oe.jackknife(fit, data, max_refits=100), "jackknife_too_large"),
        (lambda: oe.jackknife(fit, data, max_refits=1), "invalid_spec"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, str(error.value))


def test_time_series_and_resampled_fits_are_refused():
    rng = np.random.default_rng(0)
    series = pd.DataFrame({"y": np.cumsum(rng.normal(size=80)) * 0.1 + rng.normal(size=80),
                           "t": np.arange(80)})
    arima = oe.arima(data=series, y="y", order=[1, 0, 0])
    for procedure in (oe.bootstrap, oe.jackknife):
        with pytest.raises(AnalysisError) as error:
            procedure(arima, series)
        assert error.value.code.endswith("_unsupported")
    frame = make_data(n=60)
    boot_cov = oe.ols(data=frame, y="y", x=["x1"], covariance="bootstrap", reps=20, seed=1)
    with pytest.raises(AnalysisError) as error:
        oe.bootstrap(boot_cov, frame, reps=5)
    assert error.value.code == "bootstrap_unsupported"


def test_percentile_helper_matches_stata_definition():
    values = torch.arange(1.0, 11.0, dtype=torch.float64)
    assert stata_percentile(values, 0.5) == 5.5           # P = 5 integer: average
    assert stata_percentile(values, 0.25) == 3.0          # P = 2.5: ceil -> 3rd value
    assert stata_percentile(values, 0.025) == 1.0
    assert stata_percentile(values, 0.975) == 10.0


def test_summary_latex_and_public_access(data):
    fit = oe.poisson(data=data.assign(c=(data["y"] > 1).astype(int)), y="c", x=["x1"])
    boot = oe.bootstrap(fit, data.assign(c=(data["y"] > 1).astype(int)), reps=20, seed=4)
    text = boot.summary()
    assert "bootstrap standard errors" in text and "Wald chi2" in text
    assert "x1" in str(boot.to_latex())
    assert oe.bootstrap.__module__.endswith("postest.resampling")
    assert "jackknife" in dir(oe)


def test_group_role_is_relabelled_with_resampled_clusters():
    from statsmodels.discrete.conditional_models import ConditionalLogit

    rng = np.random.default_rng(1)
    groups, size = 50, 4
    frame = pd.DataFrame({"grp": np.repeat(np.arange(groups), size)})
    frame["x"] = rng.normal(size=len(frame))
    utility = frame["x"] + rng.gumbel(size=len(frame))
    frame["y"] = (utility == utility.groupby(frame["grp"]).transform("max")).astype(int)
    fit = oe.clogit(data=frame, y="y", x=["x"], group="grp")
    boot = oe.bootstrap(fit, frame, reps=15, seed=3, cluster="grp")
    replicates = []
    for rows, slot in cluster_draws(pd.factorize(frame["grp"])[0], 15, 3):
        sample = frame.iloc[rows].reset_index(drop=True)
        model = ConditionalLogit(sample["y"], sample[["x"]], groups=slot)
        fitted = model.fit(disp=0, method="newton", tol=1e-12, maxiter=100)
        replicates.append(fitted.params.to_numpy())
    replicates = np.array(replicates)
    assert_allclose(boot.covariance_matrix[0][0], np.var(replicates[:, 0], ddof=1), rtol=1e-5)
    plain = oe.bootstrap(fit, frame, reps=5, seed=3)
    assert any("pass cluster=" in w for w in plain.warnings)


def test_streamed_ols_fits_recover_their_rows(data):
    from openecon.dataset import Dataset

    frame = data.copy()
    frame.loc[:4, "x2"] = np.nan
    streamed = oe.ols(data=Dataset.from_frame(frame), y="y", x=["x1", "x2"], missing="drop")
    assert streamed.provenance.get("sample_positions_omitted") and not streamed.sample_positions
    dense = oe.ols(data=frame, y="y", x=["x1", "x2"], missing="drop")
    boot_s = oe.bootstrap(streamed, frame, reps=20, seed=1)
    boot_d = oe.bootstrap(dense, frame, reps=20, seed=1)
    assert_allclose(boot_s.covariance_matrix, boot_d.covariance_matrix, rtol=1e-9)
    jack_s, jack_d = oe.jackknife(streamed, frame), oe.jackknife(dense, frame)
    assert_allclose(jack_s.covariance_matrix, jack_d.covariance_matrix, rtol=1e-9)
    edited = frame.assign(x1=frame["x1"] * 1.01)
    with pytest.raises(AnalysisError) as error:
        oe.bootstrap(streamed, edited, reps=5)
    assert error.value.code == "data_mismatch"


def test_analytic_weights_travel_with_their_rows(data):
    weighted = data.assign(w=1.0 + data["f"] / 2)
    fit = oe.ols(data=weighted, y="y", x=["x1"], weights="w", weight_type="aweight")
    boot = oe.bootstrap(fit, weighted, reps=30, seed=17)
    replicates = np.array([ols_numpy(weighted.iloc[rows], ["x1"],
                                     weights=weighted["w"].to_numpy()[rows])
                           for rows in observation_draws(len(weighted), 30, 17)])
    assert_allclose(np.array(boot.covariance_matrix), np.cov(replicates, rowvar=False),
                    rtol=1e-9)
    assert boot.inference["resampling_unit"] == "observations"
    jack = oe.jackknife(fit, weighted)
    n = len(weighted)
    delete = np.array([ols_numpy(weighted.drop(index=i), ["x1"],
                                 weights=weighted["w"].drop(index=i).to_numpy())
                       for i in range(n)])
    centered = delete - delete.mean(axis=0)
    assert_allclose(np.array(jack.covariance_matrix), (n - 1) / n * centered.T @ centered,
                    rtol=1e-9)
