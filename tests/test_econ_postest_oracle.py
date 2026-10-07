"""Independent oracles for the post-estimation family (verify-and-repair pass).

Nothing here reuses the family's formulas or its draw kernels:

- bootstrap replicates are regenerated from the documented draw contract (one
  ``torch.rand`` vector per replicate from ``torch.Generator().manual_seed``;
  slot j of a stratum takes unit ``floor(u_j n_h)`` of that stratum, units in
  first-appearance order) with NumPy index arithmetic written here, and every
  replicate is re-estimated by statsmodels or explicit NumPy algebra;
- the jackknife is an explicit delete-one / delete-one-cluster loop over
  statsmodels fits or a brute-force SciPy tobit likelihood;
- lrtest / estat_ic are compared with statsmodels log likelihoods and a
  brute-force ordered-logit likelihood maximized by SciPy;
- suest is compared with a stacked-score sandwich built from statsmodels'
  ``score_obs`` / ``hessian`` and from numerically differentiated per-observation
  log likelihoods;
- fcast_eval and dm_test are recomputed from the EViews / Diebold-Mariano /
  Harvey-Leybourne-Newbold definitions in NumPy.

Invariances: frequency weights equal duplicated rows (bootstrap and
jackknife), rescaling a regressor rescales its bootstrap standard error by the
same factor, ols and glm(gaussian) share bootstrap draws and replicates.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import linalg, optimize, stats
from scipy.special import expit, gammaln

import openecon as oe

# ---- independent helpers ------------------------------------------------------------


def uniforms(seed, sizes):
    """The uniform vectors of successive replicates (one torch.rand call each)."""
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(size, generator=generator, dtype=torch.float64).numpy() for size in sizes]


def take(units, u):
    """Equal-probability draws with replacement: unit floor(u * n) of ``units``."""
    units = np.asarray(units)
    return units[np.minimum(np.floor(u * len(units)).astype(int), len(units) - 1)]


def stratified_draws(strata, reps, seed, size=None):
    """Row draws within strata (strata in first-appearance order, rows in data order)."""
    codes, levels = pd.factorize(pd.Series(strata), sort=False)
    members = [np.flatnonzero(codes == h) for h in range(len(levels))]
    counts = [len(m) if size is None else size for m in members]
    draws = []
    for u in uniforms(seed, [sum(counts)] * reps):
        pieces, start = [], 0
        for unit, count in zip(members, counts):
            pieces.append(take(unit, u[start:start + count]))
            start += count
        draws.append(np.concatenate(pieces))
    return draws


def cluster_draws(labels, reps, seed):
    """Whole-cluster draws: (rows in draw order, slot of each row) per replicate."""
    codes, levels = pd.factorize(pd.Series(labels), sort=False)
    members = [np.flatnonzero(codes == g) for g in range(len(levels))]
    out = []
    for u in uniforms(seed, [len(levels)] * reps):
        drawn = take(np.arange(len(levels)), u)
        rows = np.concatenate([members[g] for g in drawn])
        slots = np.concatenate([np.full(len(members[g]), i) for i, g in enumerate(drawn)])
        out.append((rows, slots))
    return out


def stata_pctile(values, p):
    """Stata's _pctile: P = R p; x_(floor(P)+1) if P is fractional, else the mean of x_P and
    x_(P+1) (1-based order statistics)."""
    x = np.sort(values)
    r = len(x)
    position = r * p
    whole = int(np.floor(position + 1e-12))
    if abs(position - whole) < 1e-9:
        lower, upper = max(whole, 1), min(whole + 1, r)
        return 0.5 * (x[lower - 1] + x[upper - 1])
    return x[min(whole + 1, r) - 1]


def lstsq(x, y, w=None):
    x = np.column_stack([np.ones(len(y)), x])
    if w is not None:
        root = np.sqrt(w)
        return np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    return np.linalg.lstsq(x, y, rcond=None)[0]


def within_slope(y, x, panel):
    """The fixed-effects (within) slopes and the intercept Stata's xtreg fe reports."""
    frame = pd.DataFrame({"y": y, "p": panel})
    for j in range(x.shape[1]):
        frame[f"x{j}"] = x[:, j]
    means = frame.groupby("p").transform("mean")
    yd = y - means["y"].to_numpy() + y.mean()
    xd = x - means[[f"x{j}" for j in range(x.shape[1])]].to_numpy() + x.mean(axis=0)
    return lstsq(xd, yd)


def covariance(replicates):
    return np.cov(np.asarray(replicates), rowvar=False, ddof=1)


def jackknife_cov(replicates):
    r = np.asarray(replicates)
    n = len(r)
    centered = r - r.mean(axis=0)
    return (n - 1) / n * centered.T @ centered


def table_of(result):
    return {name: np.array([getattr(c, name) for c in result.coefficients])
            for name in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")}


def check_inference(result, cov, df=None, alpha=0.05):
    """Standard errors, statistics, p-values and intervals implied by ``cov``."""
    rows = table_of(result)
    se = np.sqrt(np.diag(cov))
    stat = rows["estimate"] / se
    if df is None:
        p, crit = 2 * stats.norm.sf(np.abs(stat)), stats.norm.ppf(1 - alpha / 2)
    else:
        p, crit = 2 * stats.t.sf(np.abs(stat), df), stats.t.ppf(1 - alpha / 2, df)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-6, atol=1e-13)
    assert_allclose(rows["std_error"], se, rtol=1e-6)
    assert_allclose(rows["statistic"], stat, rtol=1e-6)
    assert_allclose(rows["p_value"], p, rtol=1e-5, atol=1e-12)
    assert_allclose(rows["ci_low"], rows["estimate"] - crit * se, rtol=1e-6, atol=1e-10)
    assert_allclose(rows["ci_high"], rows["estimate"] + crit * se, rtol=1e-6, atol=1e-10)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261004)
    n = 140
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "c": rng.integers(0, 3, n), "g": rng.integers(0, 12, n),
                          "s": np.where(np.arange(n) % 3 == 0, "a", "b"),
                          "f": rng.integers(1, 4, n)})
    shock = rng.normal(size=12)[frame.g]
    frame["y"] = 1 + 0.5 * frame.x1 - 0.4 * frame.x2 + shock + rng.normal(size=n)
    frame["b"] = (0.2 + frame.x1 - 0.6 * frame.x2 + rng.logistic(size=n) > 0).astype(int)
    frame["cnt"] = rng.poisson(np.exp(0.2 + 0.4 * frame.x1 + 0.2 * (frame.c == 2)))
    frame["ord"] = np.digitize(frame.x1 - 0.5 * frame.x2 + rng.logistic(size=n), [-1, 0.3, 1.4])
    frame["cat"] = np.digitize(frame.x1 + rng.logistic(size=n), [-0.4, 0.8])
    frame.loc[[4, 17, 90], "x2"] = np.nan
    return frame


# ---- bootstrap ------------------------------------------------------------------------


def test_bootstrap_probit_matches_statsmodels_replicates(data):
    fit = oe.probit(data=data, y="b", x=["x1", "x2"], missing="drop")
    reps, seed = 50, 77
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed, ci="percentile", alpha=0.1)
    sample = data.dropna(subset=["b", "x1", "x2"]).reset_index(drop=True)
    x = sm.add_constant(sample[["x1", "x2"]].to_numpy())
    y = sample["b"].to_numpy()
    replicates = []
    for u in uniforms(seed, [len(sample)] * reps):
        rows = take(np.arange(len(sample)), u)
        replicates.append(sm.Probit(y[rows], x[rows]).fit(disp=0, tol=1e-12).params)
    replicates = np.array(replicates)
    check_inference(boot, covariance(replicates), alpha=0.1)
    estimate = table_of(boot)["estimate"]
    assert_allclose(estimate, [c.estimate for c in fit.coefficients], rtol=0, atol=0)
    intervals = boot.extra["bootstrap_ci"]["intervals"]
    for j, term in enumerate(["Intercept", "x1", "x2"]):
        assert_allclose(intervals[term]["ci_low"], stata_pctile(replicates[:, j], 0.05),
                        rtol=1e-6)
        assert_allclose(intervals[term]["ci_high"], stata_pctile(replicates[:, j], 0.95),
                        rtol=1e-6)
        assert_allclose(boot.extra["bootstrap"]["bias"][term],
                        replicates[:, j].mean() - estimate[j], rtol=1e-5, atol=1e-9)
    assert boot.inference["covariance"] == "bootstrap" and boot.inference["seed"] == seed
    assert boot.inference["reps_completed"] == reps and boot.nobs == len(sample)
    # Model test: Wald chi2 of the two slopes with the bootstrap covariance.
    cov = covariance(replicates)[1:, 1:]
    wald = estimate[1:] @ np.linalg.solve(cov, estimate[1:])
    assert boot.tests["model"]["distribution"] == "chi2" and boot.tests["model"]["df"] == 2
    assert_allclose(boot.tests["model"]["statistic"], wald, rtol=1e-5)
    assert_allclose(boot.tests["model"]["p_value"], stats.chi2.sf(wald, 2), rtol=1e-4)


def test_bootstrap_bias_corrected_interval_from_replicates(data):
    fit = oe.poisson(data=data, y="cnt", x=["x1"])
    reps, seed = 80, 5
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed, ci="bc")
    x = sm.add_constant(data[["x1"]].to_numpy())
    y = data["cnt"].to_numpy()
    replicates = np.array([
        sm.Poisson(y[rows], x[rows]).fit(disp=0, tol=1e-12).params
        for rows in (take(np.arange(len(data)), u) for u in uniforms(seed, [len(data)] * reps))])
    estimate = table_of(boot)["estimate"]
    z = stats.norm.ppf(0.975)
    for j, term in enumerate(["Intercept", "x1"]):
        z0 = stats.norm.ppf(np.mean(replicates[:, j] <= estimate[j]))
        low, high = stats.norm.cdf(2 * z0 - z), stats.norm.cdf(2 * z0 + z)
        interval = boot.extra["bootstrap_ci"]["intervals"][term]
        assert_allclose(interval["ci_low"], stata_pctile(replicates[:, j], low), rtol=1e-6)
        assert_allclose(interval["ci_high"], stata_pctile(replicates[:, j], high), rtol=1e-6)
    check_inference(boot, covariance(replicates))


def test_bootstrap_cluster_draws_whole_clusters(data):
    sample = data.dropna(subset=["x2"]).reset_index(drop=True)
    fit = oe.ols(data=data, y="y", x=["x1", "x2"], missing="drop")
    reps, seed = 40, 31
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed, cluster="g")
    x, y = sample[["x1", "x2"]].to_numpy(), sample["y"].to_numpy()
    replicates = [lstsq(x[rows], y[rows]) for rows, _ in cluster_draws(sample.g, reps, seed)]
    check_inference(boot, covariance(replicates))
    assert boot.inference["cluster_count"] == sample.g.nunique()
    assert boot.inference["resampling_unit"] == "clusters of 'g'"


def test_bootstrap_stratified_with_size(data):
    sample = data.dropna(subset=["x2"]).reset_index(drop=True)
    fit = oe.ols(data=data, y="y", x=["x1", "x2"], missing="drop")
    reps, seed, size = 30, 8, 30
    boot = oe.bootstrap(fit, data, reps=reps, seed=seed, strata="s", size=size)
    x, y = sample[["x1", "x2"]].to_numpy(), sample["y"].to_numpy()
    replicates = [lstsq(x[rows], y[rows])
                  for rows in stratified_draws(sample.s, reps, seed, size=size)]
    check_inference(boot, covariance(replicates))


def test_bootstrap_fweights_equal_duplicated_rows(data):
    expanded = data.loc[data.index.repeat(data.f)].reset_index(drop=True)
    weighted = oe.poisson(data=data, y="cnt", x=["x1"], weights="f", weight_type="fweight")
    plain = oe.poisson(data=expanded, y="cnt", x=["x1"])
    assert weighted.nobs == plain.nobs == int(data.f.sum())
    for options in ({}, {"strata": "s"}):
        a = oe.bootstrap(weighted, data, reps=25, seed=12, **options)
        b = oe.bootstrap(plain, expanded, reps=25, seed=12, **options)
        assert_allclose(np.array(a.covariance_matrix), np.array(b.covariance_matrix),
                        rtol=1e-10, atol=1e-16)


def test_bootstrap_panel_within_estimator_on_relabelled_panels():
    rng = np.random.default_rng(3)
    groups, periods = 25, 5
    frame = pd.DataFrame({"pid": np.repeat(np.arange(100, 100 + groups), periods),
                          "t": np.tile(np.arange(periods), groups),
                          "x": rng.normal(size=groups * periods)})
    frame["y"] = (2 + frame.x + rng.normal(size=groups)[frame.pid - 100]
                  + rng.normal(size=len(frame)))
    fit = oe.xtreg(data=frame, y="y", x=["x"], panel="pid", time="t", model="fe")
    reps, seed = 30, 4
    boot = oe.bootstrap(fit, frame, reps=reps, seed=seed)
    replicates = []
    for rows, slots in cluster_draws(frame.pid, reps, seed):
        replicates.append(within_slope(frame.y.to_numpy()[rows], frame[["x"]].to_numpy()[rows],
                                       slots))
    check_inference(boot, covariance(replicates))
    assert boot.inference["cluster_count"] == groups


def test_bootstrap_invariances(data):
    """Rescaling a regressor rescales its bootstrap SE; ols and glm(gaussian) agree."""
    scaled = data.assign(x1=data.x1 * 1e4)
    a = oe.bootstrap(oe.ols(data=data, y="y", x=["x1"]), data, reps=30, seed=2)
    b = oe.bootstrap(oe.ols(data=scaled, y="y", x=["x1"]), scaled, reps=30, seed=2)
    ra, rb = table_of(a), table_of(b)
    assert_allclose(rb["std_error"], ra["std_error"] * np.array([1, 1e-4]), rtol=1e-8)
    assert_allclose(rb["p_value"], ra["p_value"], rtol=1e-7)
    g = oe.bootstrap(oe.glm(data=data, y="y", x=["x1"], family="gaussian"), data, reps=30, seed=2)
    assert_allclose(np.array(g.covariance_matrix), np.array(a.covariance_matrix), rtol=1e-7)


def test_bootstrap_mlogit_keeps_the_base_of_the_original_fit():
    """Regression: a resample whose most frequent outcome differs used to change the base
    category (different terms), failing a third of the replicates."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x1": rng.normal(size=300), "y": rng.integers(0, 3, 300)})
    automatic = oe.mlogit(data=frame, y="y", x=["x1"])
    explicit = oe.mlogit(data=frame, y="y", x=["x1"], base=automatic.extra["base"])
    a = oe.bootstrap(automatic, frame, reps=40, seed=1)
    b = oe.bootstrap(explicit, frame, reps=40, seed=1)
    assert a.inference["failed_replicates"] == 0
    assert_allclose(np.array(a.covariance_matrix), np.array(b.covariance_matrix), rtol=1e-9)
    small = frame.iloc[:60].reset_index(drop=True)
    automatic = oe.mlogit(data=small, y="y", x=["x1"])
    explicit = oe.mlogit(data=small, y="y", x=["x1"], base=automatic.extra["base"])
    assert_allclose(np.array(oe.jackknife(automatic, small).covariance_matrix),
                    np.array(oe.jackknife(explicit, small).covariance_matrix), rtol=1e-9)


# ---- jackknife ------------------------------------------------------------------------


def test_jackknife_poisson_delete_one_with_missing_rows(data):
    fit = oe.poisson(data=data, y="cnt", x=["x1", "x2"], missing="drop")
    jk = oe.jackknife(fit, data)
    sample = data.dropna(subset=["x2"]).reset_index(drop=True)
    x, y = sm.add_constant(sample[["x1", "x2"]].to_numpy()), sample.cnt.to_numpy()
    replicates = [sm.Poisson(np.delete(y, i), np.delete(x, i, 0)).fit(disp=0, tol=1e-12).params
                  for i in range(len(y))]
    cov = jackknife_cov(replicates)
    n = len(y)
    check_inference(jk, cov, df=n - 1)
    assert jk.inference["df_inference"] == n - 1 and jk.inference["use_t"] is True
    estimate = table_of(jk)["estimate"]
    wald = estimate[1:] @ np.linalg.solve(cov[1:, 1:], estimate[1:]) / 2
    test = jk.tests["model"]
    assert test["distribution"] == "F" and test["df"] == 2 and test["df2"] == n - 1
    assert_allclose(test["statistic"], wald, rtol=1e-6)
    assert_allclose(test["p_value"], stats.f.sf(wald, 2, n - 1), rtol=1e-5)
    bias = (n - 1) * (np.mean(replicates, axis=0) - estimate)
    assert_allclose([jk.extra["jackknife"]["bias"][t] for t in ["Intercept", "x1", "x2"]], bias,
                    rtol=1e-5, atol=1e-10)


def test_jackknife_logit_clusters(data):
    fit = oe.logit(data=data, y="b", x=["x1"])
    jk = oe.jackknife(fit, data, cluster="g")
    x, y, g = sm.add_constant(data[["x1"]].to_numpy()), data.b.to_numpy(), data.g.to_numpy()
    labels = pd.unique(data.g)
    replicates = [sm.Logit(y[g != k], x[g != k]).fit(disp=0, tol=1e-12).params for k in labels]
    check_inference(jk, jackknife_cov(replicates), df=len(labels) - 1)
    assert jk.inference["cluster_count"] == len(labels)


def test_jackknife_fweights_equal_duplicated_rows(data):
    expanded = data.loc[data.index.repeat(data.f)].reset_index(drop=True)
    a = oe.jackknife(oe.ols(data=data, y="y", x=["x1"], weights="f", weight_type="fweight"),
                     data)
    b = oe.jackknife(oe.ols(data=expanded, y="y", x=["x1"]), expanded, max_refits=10_000)
    assert_allclose(np.array(a.covariance_matrix), np.array(b.covariance_matrix), rtol=1e-9)
    assert a.inference["df_inference"] == b.inference["df_inference"] == len(expanded) - 1
    # And the delete-one loop of the expanded data written out.
    x, y = expanded[["x1"]].to_numpy(), expanded.y.to_numpy()
    replicates = [lstsq(np.delete(x, i, 0), np.delete(y, i)) for i in range(len(y))]
    assert_allclose(np.array(a.covariance_matrix), jackknife_cov(replicates), rtol=1e-8)


def test_jackknife_panel_deletes_panels():
    rng = np.random.default_rng(8)
    groups, periods = 15, 4
    frame = pd.DataFrame({"pid": np.repeat([f"p{i}" for i in range(groups)], periods),
                          "x": rng.normal(size=groups * periods)})
    frame["y"] = frame.x + rng.normal(size=groups).repeat(periods) + rng.normal(size=len(frame))
    fit = oe.xtreg(data=frame, y="y", x=["x"], panel="pid", model="fe")
    jk = oe.jackknife(fit, frame)
    replicates = []
    for label in frame.pid.unique():
        keep = frame.pid.to_numpy() != label
        replicates.append(within_slope(frame.y.to_numpy()[keep], frame[["x"]].to_numpy()[keep],
                                       frame.pid.to_numpy()[keep]))
    check_inference(jk, jackknife_cov(replicates), df=groups - 1)


def tobit_fit(y, x, low):
    """Brute-force left-censored tobit MLE (parameters b, ln sigma)."""
    def nll(theta):
        b, sigma = theta[:-1], np.exp(theta[-1])
        xb = x @ b
        censored = y <= low
        return -(stats.norm.logcdf((low - xb[censored]) / sigma).sum()
                 + (stats.norm.logpdf((y[~censored] - xb[~censored]) / sigma)
                    - np.log(sigma)).sum())
    start = np.r_[np.linalg.lstsq(x, y, rcond=None)[0], 0.0]
    found = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-10})
    found = optimize.minimize(nll, found.x, method="Nelder-Mead",
                              options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
    return np.r_[found.x[:-1], np.exp(found.x[-1])]


def test_jackknife_tobit_clusters_against_brute_force_likelihood():
    rng = np.random.default_rng(12)
    n = 120
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 6, n)})
    frame["y"] = np.maximum(0.3 + frame.x + rng.normal(size=n), 0.0)
    fit = oe.tobit(data=frame, y="y", x=["x"], ll=0.0)
    jk = oe.jackknife(fit, frame, cluster="g")
    assert [c.term for c in fit.coefficients] == ["Intercept", "x", "/sigma"]
    x = sm.add_constant(frame[["x"]].to_numpy())
    labels = pd.unique(frame.g)
    replicates = [tobit_fit(frame.y.to_numpy()[frame.g != k], x[frame.g.to_numpy() != k], 0.0)
                  for k in labels]
    assert_allclose(np.array(jk.covariance_matrix), jackknife_cov(replicates), rtol=2e-4,
                    atol=1e-9)
    assert jk.inference["df_inference"] == len(labels) - 1


# ---- lrtest and estat_ic --------------------------------------------------------------


def test_lrtest_poisson_categorical_against_statsmodels(data):
    full = oe.poisson(data=data, y="cnt", x=["x1", "c"], categorical=["c"])
    small = oe.poisson(data=data, y="cnt", x=["x1"])
    result = oe.lrtest(full, small)
    x_small = sm.add_constant(data[["x1"]].to_numpy())
    x_full = np.column_stack([x_small, data.c == 1, data.c == 2]).astype(float)
    ll_full = sm.Poisson(data.cnt, x_full).fit(disp=0, tol=1e-12).llf
    ll_small = sm.Poisson(data.cnt, x_small).fit(disp=0, tol=1e-12).llf
    row = result.iloc[0]
    assert_allclose([row.ll_full, row.ll_restricted], [ll_full, ll_small], rtol=1e-9)
    assert_allclose(row.statistic, 2 * (ll_full - ll_small), rtol=1e-6)
    assert row.df == 2
    assert_allclose(row.p_value, stats.chi2.sf(2 * (ll_full - ll_small), 2), rtol=1e-5)


def ologit_ll(y, x, k):
    """Brute-force ordered logit with free cutpoints (maximized log likelihood)."""
    def nll(theta):
        b, cuts = theta[:x.shape[1]], np.r_[-np.inf, np.sort(theta[x.shape[1]:]), np.inf]
        xb = x @ b
        return -np.log(expit(cuts[y + 1] - xb) - expit(cuts[y] - xb)).sum()
    start = np.r_[np.zeros(x.shape[1]), np.linspace(-1, 1, k - 1)]
    found = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-9})
    return -found.fun


def test_lrtest_ologit_against_brute_force_likelihood(data):
    full = oe.ologit(data=data, y="ord", x=["x1", "x2"], missing="drop")
    small = oe.ologit(data=data, y="ord", x=["x1"])
    with pytest.raises(oe.AnalysisError) as error:
        oe.lrtest(full, small)   # the restricted fit also uses the rows with missing x2
    assert error.value.code == "incompatible_models"
    sample = data.dropna(subset=["x2"]).reset_index(drop=True)
    full = oe.ologit(data=sample, y="ord", x=["x1", "x2"])
    small = oe.ologit(data=sample, y="ord", x=["x1"])
    y = sample.ord.to_numpy()
    full_ll = ologit_ll(y, sample[["x1", "x2"]].to_numpy(), 4)
    small_ll = ologit_ll(y, sample[["x1"]].to_numpy(), 4)
    result = oe.lrtest(full, small)
    row = result.iloc[0]
    assert_allclose([row.ll_full, row.ll_restricted], [full_ll, small_ll], rtol=1e-8)
    assert row.df == 1 and result.attrs["k_full"] == 5 and result.attrs["k_restricted"] == 4
    assert_allclose(row.statistic, 2 * (full_ll - small_ll), rtol=1e-5)
    assert_allclose(row.p_value, stats.chi2.sf(2 * (full_ll - small_ll), 1), rtol=1e-4)


def test_lrtest_nbreg_against_poisson_with_chibar2(data):
    poisson = oe.poisson(data=data, y="cnt", x=["x1"])
    nbreg = oe.nbreg(data=data, y="cnt", x=["x1"])
    with pytest.raises(oe.AnalysisError):
        oe.lrtest(nbreg, poisson)
    result = oe.lrtest(nbreg, poisson, force=True)
    x = sm.add_constant(data[["x1"]].to_numpy())
    y = data.cnt.to_numpy()

    def nb2_nll(theta):            # NB2 with ln alpha, written out
        mu, alpha = np.exp(x @ theta[:2]), np.exp(theta[2])
        m = 1 / alpha
        return -(gammaln(y + m) - gammaln(m) - gammaln(y + 1) + m * np.log(m / (m + mu))
                 + y * np.log(mu / (m + mu))).sum()

    found = optimize.minimize(nb2_nll, np.r_[np.log(y.mean() + 0.1), 0.0, -1.0], method="BFGS",
                              options={"gtol": 1e-10})
    ll_nb = -found.fun
    assert_allclose(nbreg.metrics["log_likelihood"], ll_nb, rtol=1e-8)
    ll_p = sm.Poisson(data.cnt, x).fit(disp=0, tol=1e-12).llf
    statistic = max(0.0, 2 * (ll_nb - ll_p))
    row = result.iloc[0]
    assert_allclose(row.statistic, statistic, rtol=1e-5, atol=1e-7)
    assert row.df == 1 and result.attrs["forced"] is True
    assert result.attrs["boundary_terms"] == ["/lnalpha"]
    assert_allclose(result.attrs["p_value_chibar2"], 0.5 * stats.chi2.sf(statistic, 1),
                    rtol=1e-5)


def test_estat_ic_fweights_and_mlogit_against_statsmodels(data):
    weighted = oe.poisson(data=data, y="cnt", x=["x1"], weights="f", weight_type="fweight")
    mlogit = oe.mlogit(data=data, y="cat", x=["x1"], base=0)
    logit = oe.logit(data=data, y="b", x=["x1"])
    table = oe.estat_ic(weighted, mlogit, logit, names=["pw", "ml", "lg"])
    expanded = data.loc[data.index.repeat(data.f)]
    x = sm.add_constant(expanded[["x1"]].to_numpy())
    ll = sm.Poisson(expanded.cnt.to_numpy(), x).fit(disp=0, tol=1e-12).llf
    n = int(data.f.sum())
    row = table.loc[0]
    assert row.nobs == n and row.df == 2
    assert_allclose([row.ll, row.aic, row.bic], [ll, -2 * ll + 4, -2 * ll + 2 * np.log(n)],
                    rtol=1e-9)
    mn = sm.MNLogit(data.cat.to_numpy(), sm.add_constant(data[["x1"]].to_numpy())).fit(
        disp=0, tol=1e-12, method="newton")
    row = table.loc[1]
    assert row.df == 4 and row.nobs == len(data)
    assert_allclose([row.ll, row.ll_null, row.aic, row.bic],
                    [mn.llf, mn.llnull, -2 * mn.llf + 8, -2 * mn.llf + 4 * np.log(len(data))],
                    rtol=1e-8)
    lg = sm.Logit(data.b.to_numpy(), sm.add_constant(data[["x1"]].to_numpy())).fit(
        disp=0, tol=1e-12)
    row = table.loc[2]
    assert_allclose([row.ll, row.ll_null], [lg.llf, lg.llnull], rtol=1e-8)


def test_lrtest_and_estat_ic_after_ols_against_statsmodels(data):
    frame = data.dropna().reset_index(drop=True)
    full = oe.ols(data=frame, y="y", x=["x1", "x2", "c"], categorical=["c"])
    small = oe.ols(data=frame, y="y", x=["x1"])
    x_small = sm.add_constant(frame[["x1"]].to_numpy())
    x_full = np.column_stack([x_small, frame.x2, frame.c == 1, frame.c == 2]).astype(float)
    big, little = sm.OLS(frame.y, x_full).fit(), sm.OLS(frame.y, x_small).fit()
    statistic, p_value, df = big.compare_lr_test(little)
    row = oe.lrtest(full, small).iloc[0]
    assert_allclose([row.statistic, row.p_value], [statistic, p_value], rtol=1e-8)
    assert row.df == df == 3
    table = oe.estat_ic(full, small)
    null = sm.OLS(frame.y, np.ones(len(frame))).fit().llf
    assert_allclose(table.ll, [big.llf, little.llf], rtol=1e-10)
    assert_allclose(table.ll_null.astype(float), [null, null], rtol=1e-10)
    assert_allclose(table.aic, [big.aic, little.aic], rtol=1e-10)   # k = e(rank), sigma excluded
    assert_allclose(table.bic, [big.bic, little.bic], rtol=1e-10)
    weighted = oe.ols(data=frame, y="y", x=["x1"], weights="f", weight_type="fweight")
    expanded = frame.loc[frame.index.repeat(frame.f)]
    null_w = sm.OLS(expanded.y.to_numpy(), np.ones(len(expanded))).fit().llf
    assert_allclose(oe.estat_ic(weighted).ll_null.astype(float), [null_w], rtol=1e-10)


def test_lrtest_nested_seemingly_unrelated_regressions(data):
    """Regression: the systems 'system' role lists regressors too, so nested sureg fits were
    refused as 'different outcomes'."""
    frame = data.dropna().reset_index(drop=True)
    full = oe.sureg(data=frame, equations=[{"y": "y", "x": ["x1", "x2"]},
                                           {"y": "cnt", "x": ["x1"]}], iterate=True)
    small = oe.sureg(data=frame, equations=[{"y": "y", "x": ["x1"]},
                                            {"y": "cnt", "x": ["x1"]}], iterate=True)
    result = oe.lrtest(full, small)
    statistic = 2 * (full.metrics["log_likelihood"] - small.metrics["log_likelihood"])
    assert result.iloc[0].df == 1
    assert_allclose(result.iloc[0].statistic, statistic, rtol=1e-12)
    other = oe.sureg(data=frame, equations=[{"y": "y", "x": ["x1"]}, {"y": "b", "x": ["x1"]}],
                     iterate=True)
    with pytest.raises(oe.AnalysisError) as error:
        oe.lrtest(full, other)
    assert error.value.code == "incompatible_models" and "equation=b" in str(error.value)


def test_lrtest_reml_rules():
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"g": np.repeat(np.arange(30), 6), "x": rng.normal(size=180),
                          "z": rng.normal(size=180)})
    frame["y"] = frame.x + rng.normal(size=30)[frame.g] + rng.normal(size=180) \
        + rng.normal(size=30)[frame.g] * frame.z
    reml_full = oe.mixed(data=frame, y="y", x=["x", "z"], group="g", method="reml")
    reml_small = oe.mixed(data=frame, y="y", x=["x"], group="g", method="reml")
    ml_small = oe.mixed(data=frame, y="y", x=["x"], group="g")
    for full, small in ((reml_full, reml_small), (reml_full, ml_small)):
        with pytest.raises(oe.AnalysisError) as error:
            oe.lrtest(full, small)
        assert error.value.code == "incompatible_models" and "REML" in str(error.value)
    # force passes the REML gate (here the REML criterion even rises when z is dropped,
    # which is why the test is invalid; the order is reversed with df= to show the gate).
    assert reml_small.metrics["log_likelihood"] > reml_full.metrics["log_likelihood"]
    forced = oe.lrtest(reml_small, reml_full, df=1, force=True)
    assert forced.attrs["forced"] is True and any("REML" in n for n in forced.attrs["notes"])
    with pytest.raises(oe.AnalysisError):
        oe.lrtest(reml_full, ml_small, force=True)
    # Same fixed effects, nested random effects: allowed under REML.
    slopes = oe.mixed(data=frame, y="y", x=["x", "z"], group="g", random=["z"], method="reml")
    result = oe.lrtest(slopes, reml_full)
    statistic = 2 * (slopes.metrics["log_likelihood"] - reml_full.metrics["log_likelihood"])
    assert_allclose(result.iloc[0].statistic, statistic, rtol=1e-12)
    assert result.iloc[0].df == 1 and result.attrs["boundary_terms"]
    table = oe.estat_ic(reml_full, reml_small)
    assert any("REML" in note for note in table.attrs["notes"])


def test_lrtest_mixed_against_ols_boundary_and_df_note():
    """mixed (ML) vs ols: the OLS error variance is estimated but not reported, so the
    parameter-count difference overstates df; with df=1 the chibar2 p-value is the
    random-intercept LR test (statsmodels MixedLM log likelihood as the oracle)."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(16)
    frame = pd.DataFrame({"g": np.repeat(np.arange(30), 6), "x": rng.normal(size=180)})
    frame["y"] = frame.x + 0.6 * rng.normal(size=30)[frame.g] + rng.normal(size=180)
    mixed = oe.mixed(data=frame, y="y", x=["x"], group="g")
    ols = oe.ols(data=frame, y="y", x=["x"])
    reference = smf.mixedlm("y ~ x", frame, groups=frame["g"]).fit(reml=False)
    assert_allclose(mixed.metrics["log_likelihood"], reference.llf, rtol=1e-7)
    default = oe.lrtest(mixed, ols, force=True)
    assert default.iloc[0].df == 2 and any("overstates df" in n for n in default.attrs["notes"])
    result = oe.lrtest(mixed, ols, force=True, df=1)
    statistic = 2 * (reference.llf - sm.OLS(frame.y, sm.add_constant(frame.x)).fit().llf)
    assert_allclose(result.iloc[0].statistic, statistic, rtol=1e-6)
    assert result.attrs["boundary_terms"] == ["/var(_cons[g])"]
    assert_allclose(result.attrs["p_value_chibar2"], 0.5 * stats.chi2.sf(statistic, 1),
                    rtol=1e-5)


# ---- suest ----------------------------------------------------------------------------


def test_suest_logit_poisson_ols_against_stacked_statsmodels_scores(data):
    frame = data.copy()
    frame.loc[110:, "cnt"] = np.nan
    m1 = oe.logit(data=frame, y="b", x=["x1", "x2"], missing="drop")
    m2 = oe.poisson(data=frame, y="cnt", x=["x1"], missing="drop")
    m3 = oe.ols(data=frame, y="y", x=["x1"])
    rows1 = frame[["b", "x1", "x2"]].dropna().index.to_numpy()
    rows2 = frame[["cnt", "x1"]].dropna().index.to_numpy()
    rows3 = frame.index.to_numpy()
    lg = sm.Logit(frame.b[rows1].to_numpy(), sm.add_constant(frame.loc[rows1, ["x1", "x2"]]
                                                             .to_numpy())).fit(disp=0, tol=1e-12)
    po = sm.Poisson(frame.cnt[rows2].to_numpy(), sm.add_constant(frame.loc[rows2, ["x1"]]
                                                                 .to_numpy())).fit(disp=0,
                                                                                   tol=1e-12)
    x3, y3 = sm.add_constant(frame[["x1"]].to_numpy()), frame.y.to_numpy()
    beta = np.linalg.lstsq(x3, y3, rcond=None)[0]
    resid = y3 - x3 @ beta
    sigma2 = resid @ resid / len(y3)
    scores3 = np.column_stack([x3 * (resid / sigma2)[:, None], 0.5 * (resid ** 2 / sigma2 - 1)])
    hessian3 = linalg.block_diag(-x3.T @ x3 / sigma2, [[-len(y3) / 2]])
    union = np.union1d(np.union1d(rows1, rows2), rows3)
    stacked = np.zeros((len(union), 8))
    stacked[np.searchsorted(union, rows1), 0:3] = lg.model.score_obs(lg.params)
    stacked[np.searchsorted(union, rows2), 3:5] = po.model.score_obs(po.params)
    stacked[np.searchsorted(union, rows3), 5:8] = scores3
    bread = linalg.block_diag(np.linalg.inv(-lg.model.hessian(lg.params)),
                              np.linalg.inv(-po.model.hessian(po.params)),
                              np.linalg.inv(-hessian3))
    n = len(union)
    robust = oe.suest(m1, m2, m3, data=frame, names=["l", "p", "o"])
    expected = bread @ stacked.T @ stacked @ bread * n / (n - 1)
    check_inference(robust, expected)
    assert_allclose(table_of(robust)["estimate"],
                    np.r_[lg.params, po.params, beta, np.log(sigma2)], rtol=1e-7)
    assert [c.term for c in robust.coefficients][-1] == "o:/lnvar" and robust.nobs == n
    clustered = oe.suest(m1, m2, m3, data=frame, names=["l", "p", "o"], cluster="g")
    labels = frame.g.to_numpy()[union]
    meat = sum(np.outer(stacked[labels == k].sum(0), stacked[labels == k].sum(0))
               for k in np.unique(labels))
    groups = len(np.unique(labels))
    check_inference(clustered, bread @ meat @ bread * groups / (groups - 1))
    # A single model's block is its robust sandwich with the N/(N-1) factor of the union.
    hc0 = sm.OLS(y3, x3).fit(cov_type="HC0").cov_params()
    assert_allclose(np.array(robust.covariance_matrix)[5:7, 5:7], hc0 * n / (n - 1), rtol=1e-8)


def ordered_scores(theta, x, y, cdf, pdf):
    """Per-observation scores of an ordered model with free cutpoints (own derivation):
    P = F(c_y - xb) - F(c_(y-1) - xb); d ln P/db = -x (f_y - f_(y-1)) / P and
    d ln P/dc_j = (1{j = y} f_y - 1{j = y - 1} f_(y-1)) / P."""
    k = x.shape[1]
    cuts = np.r_[-np.inf, theta[k:], np.inf]
    xb = x @ theta[:k]
    upper, lower = cuts[y + 1] - xb, cuts[y] - xb
    prob = cdf(upper) - cdf(lower)
    f_up, f_low = pdf(upper), pdf(lower)
    scores = np.zeros((len(y), len(theta)))
    scores[:, :k] = -x * ((f_up - f_low) / prob)[:, None]
    for j in range(len(theta) - k):
        scores[:, k + j] = ((y == j) * f_up - (y == j + 1) * f_low) / prob
    return scores


def jacobian_of_sum(score, theta, step=1e-5):
    columns = []
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = step
        columns.append((score(theta + e).sum(0) - score(theta - e).sum(0)) / (2 * step))
    hessian = np.column_stack(columns)
    return (hessian + hessian.T) / 2


def test_suest_ordered_and_multinomial_against_own_scores(data):
    frame = data.dropna().reset_index(drop=True)
    m1 = oe.ologit(data=frame, y="ord", x=["x1", "x2"])
    m2 = oe.oprobit(data=frame, y="ord", x=["x1"])
    m3 = oe.mlogit(data=frame, y="cat", x=["x1"], base=0)
    joint = oe.suest(m1, m2, m3, data=frame)
    y = frame.ord.to_numpy()
    logistic_pdf = lambda z: expit(z) * (1 - expit(z))   # noqa: E731
    pieces = []
    for model, x, cdf, pdf in ((m1, frame[["x1", "x2"]].to_numpy(), expit, logistic_pdf),
                               (m2, frame[["x1"]].to_numpy(), stats.norm.cdf, stats.norm.pdf)):
        theta = np.array([c.estimate for c in model.coefficients])
        score = lambda t, x=x, cdf=cdf, pdf=pdf: ordered_scores(t, x, y, cdf, pdf)  # noqa: E731
        assert np.abs(score(theta).sum(0)).max() < 1e-6     # converged estimates
        pieces.append((score(theta), jacobian_of_sum(score, theta)))
    mn = sm.MNLogit(frame.cat.to_numpy(), sm.add_constant(frame[["x1"]].to_numpy())).fit(
        disp=0, tol=1e-12, method="newton")
    theta = mn.params.ravel(order="F")
    pieces.append((mn.model.score_obs(theta), mn.model.hessian(theta)))
    stacked = np.hstack([s for s, _ in pieces])
    bread = linalg.block_diag(*[np.linalg.inv(-h) for _, h in pieces])
    n = len(frame)
    expected = bread @ stacked.T @ stacked @ bread * n / (n - 1)
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=1e-6,
                    atol=1e-8 * np.abs(expected).max())
    assert_allclose(table_of(joint)["estimate"][-4:], theta, rtol=1e-7)
    assert [c.term for c in joint.coefficients][:3] == ["ologit:x1", "ologit:x2", "ologit:/cut1"]


# ---- forecast evaluation ----------------------------------------------------------------


def eviews_statistics(y, f, m=1, naive=None):
    e = y - f
    h = len(y)
    out = {"n": h, "mean_error": e.mean(), "mae": np.abs(e).mean(), "mse": (e ** 2).mean()}
    out["rmse"] = np.sqrt(out["mse"])
    out["mape"] = 100 * np.mean(np.abs(e / y))
    out["smape"] = 100 * np.mean(np.abs(f - y) / ((np.abs(f) + np.abs(y)) / 2))
    out["theil_u1"] = out["rmse"] / (np.sqrt(np.mean(f ** 2)) + np.sqrt(np.mean(y ** 2)))
    num = sum(((f[t + 1] - y[t + 1]) / y[t]) ** 2 for t in range(h - 1))
    den = sum(((y[t + 1] - y[t]) / y[t]) ** 2 for t in range(h - 1))
    out["theil_u2"] = np.sqrt(num / den)
    sf, sy = np.std(f), np.std(y)        # divisor h (biased), as EViews
    r = np.corrcoef(f, y)[0, 1]
    out["bias_proportion"] = (f.mean() - y.mean()) ** 2 / out["mse"]
    out["variance_proportion"] = (sf - sy) ** 2 / out["mse"]
    out["covariance_proportion"] = 2 * (1 - r) * sf * sy / out["mse"]
    scale = (np.mean(np.abs(y - naive)) if naive is not None
             else np.mean(np.abs(y[m:] - y[:-m])))
    out["mase"] = out["mae"] / scale
    return out


def test_fcast_eval_matches_eviews_definitions():
    rng = np.random.default_rng(14)
    y = 10 + np.cumsum(rng.normal(size=40))
    f = y + rng.normal(scale=0.8, size=40) + 0.3
    naive = np.r_[y[0], y[:-1]]
    for options, m, bench in (({}, 1, None), ({"seasonal_period": 4}, 4, None),
                              ({"naive": naive}, 1, naive)):
        table = oe.fcast_eval(y, f, **options)
        expected = eviews_statistics(y, f, m, bench)
        for name, value in expected.items():
            assert_allclose(table.loc[name, "value"], value, rtol=1e-10, err_msg=name)
    total = sum(table.loc[name, "value"] for name in
                ("bias_proportion", "variance_proportion", "covariance_proportion"))
    assert_allclose(total, 1.0, rtol=1e-12)
    # Scale invariance of the relative measures; MSE scales with c^2.
    scaled = oe.fcast_eval(y * 1e6, f * 1e6)
    base = oe.fcast_eval(y, f)
    for name in ("mape", "smape", "theil_u1", "theil_u2", "bias_proportion", "mase"):
        assert_allclose(scaled.loc[name, "value"], base.loc[name, "value"], rtol=1e-9)
    assert_allclose(scaled.loc["mse", "value"], base.loc["mse", "value"] * 1e12, rtol=1e-9)


def dm_numpy(y, f1, f2, h, loss, kernel, harvey):
    e1, e2 = y - f1, y - f2
    d = e1 ** 2 - e2 ** 2 if loss == "squared" else np.abs(e1) - np.abs(e2)
    t = len(d)
    c = d - d.mean()
    gamma = [np.dot(c[k:], c[:t - k]) / t for k in range(h)]
    weights = [1 - k / h if kernel == "bartlett" else 1.0 for k in range(h)]
    lrv = gamma[0] + 2 * sum(w * g for w, g in zip(weights[1:], gamma[1:]))
    if not lrv > 0:
        return None, None, lrv
    dm = d.mean() / np.sqrt(lrv / t)
    if not harvey:
        return dm, 2 * stats.norm.sf(abs(dm)), lrv
    corrected = dm * np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    return corrected, 2 * stats.t.sf(abs(corrected), t - 1), lrv


@pytest.mark.parametrize("h", [1, 3, 90])
@pytest.mark.parametrize("loss", ["squared", "absolute"])
@pytest.mark.parametrize("kernel", ["bartlett", "uniform"])
def test_dm_test_matches_definition(h, loss, kernel):
    rng = np.random.default_rng(h)
    t = 400
    y = rng.normal(size=t).cumsum()
    noise = np.convolve(rng.normal(size=t + h), np.ones(h) / h, mode="valid")[:t]
    f1, f2 = y + noise, y + 1.1 * rng.normal(size=t)
    for harvey in (True, False):
        statistic, p_value, lrv = dm_numpy(y, f1, f2, h, loss, kernel, harvey)
        if not lrv > 0:
            with pytest.raises(oe.AnalysisError):
                oe.dm_test(y, f1, f2, horizon=h, loss=loss, kernel=kernel, harvey=harvey)
            continue
        row = oe.dm_test(y, f1, f2, horizon=h, loss=loss, kernel=kernel, harvey=harvey).iloc[0]
        assert_allclose([row.statistic, row.p_value, row.long_run_variance],
                        [statistic, p_value, lrv], rtol=1e-8)
        assert row.df == (t - 1 if harvey else None) or (not harvey and np.isnan(row.df))
        swapped = oe.dm_test(y, f2, f1, horizon=h, loss=loss, kernel=kernel,
                             harvey=harvey).iloc[0]
        assert_allclose(swapped.statistic, -row.statistic, rtol=1e-12)
        assert_allclose(swapped.p_value, row.p_value, rtol=1e-12)
