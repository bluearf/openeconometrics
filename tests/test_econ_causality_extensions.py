"""Independent NumPy oracles for the bootstrap causality extensions.

Every statistic is recomputed here with explicit NumPy linear algebra (direct
Kronecker-product SUR GLS, uncentred QR/lstsq regressions with an explicit
constant, loops over replications) and the bootstrap distributions are
recomputed from the SAME resampling indices, regenerated from the documented
private-generator draw order. Nothing below calls the OpenEcon kernels for an
expected value.
"""

import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2, norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.var import bootstrap_kernels as bk
from openecon.econometrics.var.asymmetric import asymcausality
from openecon.econometrics.var.fourier_causality import fouriertycausality
from openecon.econometrics.var.panel_bootstrap_causality import ekcausality, konyacausality
from openecon.econometrics.var.rolling_causality import rollingcausality


@pytest.fixture(autouse=True, scope="module")
def owned_test_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


# ---- data ----------------------------------------------------------------------------


def panel(units=4, periods=36, seed=8, rho=.6, causal=.6, level=5.0):
    rng = np.random.default_rng(seed)
    cov = rho * np.ones((units, units)) + (1 - rho) * np.eye(units)
    ch = np.linalg.cholesky(cov)
    e = rng.normal(size=(periods + 50, units)) @ ch.T
    v = rng.normal(size=(periods + 50, units)) @ ch.T
    yv = np.zeros((periods + 50, units))
    xv = np.zeros((periods + 50, units))
    for t in range(2, periods + 50):
        xv[t] = .3 + .5 * xv[t - 1] + .1 * yv[t - 1] + v[t]
        yv[t] = level * .6 + .4 * yv[t - 1] + causal * xv[t - 1] * (np.arange(units) % 2) + e[t]
    frame = pd.DataFrame({"y": yv[-periods:].T.ravel(), "x": xv[-periods:].T.ravel()})
    frame["unit"] = np.repeat([f"u{i}" for i in range(units)], periods)
    frame["t"] = np.tile(np.arange(periods), units)
    return frame


def arrays(frame, effect="y", cause="x"):
    ys, xs = [], []
    for unit in sorted(frame.unit.unique()):
        part = frame[frame.unit == unit].sort_values("t")
        ys.append(part[effect].to_numpy())
        xs.append(part[cause].to_numpy())
    return np.array(ys), np.array(xs)


def lags_of(v, lags, start):
    n = len(v)
    return [v[start - j:n - j] for j in range(1, lags + 1)]


# ---- Kónya SUR oracle ------------------------------------------------------------------


def sur_oracle(ys, xs, own, cross, start):
    """Direct Zellner two-step GLS with explicit intercept columns and Kronecker weights."""
    units, total = ys.shape
    usable = total - start
    blocks, outs, resid = [], [], []
    for yv, xv in zip(ys, xs):
        z = np.column_stack([np.ones(usable), *lags_of(yv, own, start), *lags_of(xv, cross, start)])
        b = np.linalg.lstsq(z, yv[start:], rcond=None)[0]
        resid.append(yv[start:] - z @ b)
        blocks.append(z)
        outs.append(yv[start:])
    e = np.array(resid)
    sigma = e @ e.T / usable
    m = blocks[0].shape[1]
    x = np.zeros((units * usable, units * m))
    for i, z in enumerate(blocks):
        x[i * usable:(i + 1) * usable, i * m:(i + 1) * m] = z
    omega = np.kron(np.linalg.inv(sigma), np.eye(usable))
    v = np.linalg.inv(x.T @ omega @ x)
    b = v @ x.T @ omega @ np.concatenate(outs)
    sur_resid = (np.concatenate(outs) - x @ b).reshape(units, usable)
    wald = []
    for i in range(units):
        idx = [i * m + 1 + own + j for j in range(cross)]
        wald.append(b[idx] @ np.linalg.solve(v[np.ix_(idx, idx)], b[idx]))
    return {"wald": np.array(wald), "b": b, "v": v, "sigma": sigma, "resid": sur_resid, "m": m}


def restricted_sur_oracle(ys, own, start):
    units, total = ys.shape
    usable = total - start
    blocks, outs, resid = [], [], []
    for yv in ys:
        z = np.column_stack([np.ones(usable), *lags_of(yv, own, start)])
        b = np.linalg.lstsq(z, yv[start:], rcond=None)[0]
        resid.append(yv[start:] - z @ b)
        blocks.append(z)
        outs.append(yv[start:])
    e = np.array(resid)
    sigma = e @ e.T / usable
    m = own + 1
    x = np.zeros((units * usable, units * m))
    for i, z in enumerate(blocks):
        x[i * usable:(i + 1) * usable, i * m:(i + 1) * m] = z
    omega = np.kron(np.linalg.inv(sigma), np.eye(usable))
    b = np.linalg.solve(x.T @ omega @ x, x.T @ omega @ np.concatenate(outs))
    resid = (np.concatenate(outs) - x @ b).reshape(units, usable)
    return b.reshape(units, m), resid


def konya_bootstrap_oracle(ys, xs, own, cross, index, burn_in=0):
    start = max(own, cross)
    coef, resid = restricted_sur_oracle(ys, own, start)
    units, total = ys.shape
    out = []
    for draw in index:
        shocks = resid[:, draw]
        if burn_in:
            # Kónya's implementation remark: zero start, keep the last T levels.
            sim = np.zeros((units, start + shocks.shape[1]))
        else:
            sim = ys.astype(float).copy()
        for t in range(start, sim.shape[1]):
            sim[:, t] = coef[:, 0] + sum(coef[:, j] * sim[:, t - j] for j in range(1, own + 1)) \
                + shocks[:, t - start]
        out.append(sur_oracle(sim[:, -total:], xs, own, cross, start)["wald"])
    return np.array(out)


@pytest.mark.parametrize("own,cross", [(1, 1), (2, 1), (1, 3)])
def test_konya_sur_wald_coefficients_and_full_covariance_match_kronecker_gls(own, cross):
    frame = panel()
    result = konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=(own, cross),
                            bootstrap=99, seed=1)
    for direction, (effect, cause) in {"x -> y": ("y", "x"), "y -> x": ("x", "y")}.items():
        ys, xs = arrays(frame, effect, cause)
        expected = sur_oracle(ys, xs, own, cross, max(own, cross))
        tests = result["tests"].query("direction == @direction")
        coefficients = result["coefficients"].query("direction == @direction")
        assert_allclose(tests.statistic, expected["wald"], rtol=1e-9)
        assert_allclose(coefficients.estimate, expected["b"], rtol=1e-9, atol=1e-11)
        assert_allclose(coefficients.std_error, np.sqrt(np.diag(expected["v"])), rtol=1e-9)
        assert_allclose(result.attrs["covariance_matrix"][direction], expected["v"], rtol=1e-8,
                        atol=1e-12 * np.abs(expected["v"]).max())
        assert_allclose(result.attrs["residual_covariance"][direction], expected["sigma"], rtol=1e-10)
        assert_allclose(tests.asymptotic_p_value, chi2.sf(expected["wald"], cross), rtol=1e-8)
        assert (tests.df == cross).all()
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{table}" in result.to_latex()
    assert "Kónya" in str(result)


def test_konya_bootstrap_matches_oracle_with_identical_period_draws():
    frame = panel(units=3, periods=30)
    draws, seed = 99, 17
    result = konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=(2, 1),
                            bootstrap=draws, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    for direction, (effect, cause) in {"x -> y": ("y", "x"), "y -> x": ("x", "y")}.items():
        usable = result.attrs["lags"][direction]["usable_periods"]
        assert result.attrs["chunk_size"][direction] == draws        # one randint call
        index = torch.randint(usable, (draws, usable), generator=gen).numpy()
        assert result.attrs["bootstrap_first_indices"][direction] == index[0].tolist()
        ys, xs = arrays(frame, effect, cause)
        boot = konya_bootstrap_oracle(ys, xs, 2, 1, index)
        observed = sur_oracle(ys, xs, 2, 1, 2)["wald"]
        tests = result["tests"].query("direction == @direction")
        expected_p = [(1 + (boot[:, i] >= observed[i]).sum()) / (draws + 1) for i in range(3)]
        assert_allclose(tests.bootstrap_p_value, expected_p)
        ordered = np.sort(boot, axis=0)
        for level, column in ((.01, "cv_1"), (.05, "cv_5"), (.10, "cv_10")):
            rank = math.ceil((draws + 1) * (1 - level) - 1e-9)
            assert_allclose(tests[column], ordered[rank - 1], rtol=1e-7)
        assert_allclose(tests.mc_std_error,
                        np.sqrt(np.array(expected_p) * (1 - np.array(expected_p)) / draws))


def test_konya_burn_in_scheme_matches_oracle_with_identical_draws():
    frame = panel(units=3, periods=30, seed=12)
    draws, seed, burn = 99, 8, 25
    result = konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1,
                            bootstrap=draws, seed=seed, burn_in=burn, direction="x_to_y")
    ys, xs = arrays(frame)
    usable = ys.shape[1] - 1
    index = torch.randint(usable, (draws, usable + burn),
                          generator=torch.Generator().manual_seed(seed)).numpy()
    boot = konya_bootstrap_oracle(ys, xs, 1, 1, index, burn_in=burn)
    observed = sur_oracle(ys, xs, 1, 1, 1)["wald"]
    expected = [(1 + (boot[:, i] >= observed[i]).sum()) / (draws + 1) for i in range(3)]
    assert_allclose(result["tests"].bootstrap_p_value, expected)
    assert result.attrs["burn_in"] == burn and "burn-in" in result.attrs["null_resampling"]


def test_konya_lag_selection_uses_declared_system_criterion():
    frame = panel(units=3, periods=40, seed=4)
    result = konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=None, maxlag=2,
                            criterion="sic", bootstrap=99, direction="x_to_y")
    ys, xs = arrays(frame)
    grid = {}
    for own in (1, 2):
        for cross in (1, 2):
            fit = sur_oracle(ys, xs, own, cross, 2)
            usable = ys.shape[1] - 2
            sign, logdet = np.linalg.slogdet(fit["resid"] @ fit["resid"].T / usable)
            # Kónya (2004 WP eqs 3-4): ln|W| + N^2 q ln(T) / T with N = 3 equations.
            grid[(own, cross)] = logdet + math.log(usable) * 9 * (1 + own + cross) / usable
    table = result["lag_selection"]
    assert_allclose(table["sic"], [grid[(o, c)] for o, c in zip(table.own_lags, table.cross_lags)],
                    rtol=1e-9)
    best = min(grid, key=grid.get)
    assert (result.attrs["lags"]["x -> y"]["own"], result.attrs["lags"]["x -> y"]["cross"]) == best
    assert result.attrs["lags"]["x -> y"]["start"] == max(best)


def test_konya_dependence_diagnostics_match_formulas():
    frame = panel(units=5, periods=40, rho=.7)
    result = konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1,
                            bootstrap=99, direction="x_to_y")
    ys, xs = arrays(frame)
    sigma = sur_oracle(ys, xs, 1, 1, 1)["sigma"]
    corr = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
    rho = corr[np.triu_indices(5, 1)]
    usable = 39
    lm = usable * (rho ** 2).sum()
    cd = math.sqrt(2 * usable / 20) * rho.sum()
    table = result["dependence"]
    assert_allclose(table.statistic, [lm, cd], rtol=1e-10)
    assert_allclose(table.p_value, [chi2.sf(lm, 10), 2 * norm.sf(abs(cd))], rtol=1e-8)


def test_konya_detects_heterogeneous_causality_with_cross_dependence():
    frame = panel(units=4, periods=60, seed=21, causal=.9)
    result = konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1,
                            bootstrap=199, seed=3, direction="x_to_y")
    p = result["tests"].set_index("unit").bootstrap_p_value
    assert p["u1"] < .05 and p["u3"] < .05
    assert result["dependence"].p_value.iloc[0] < .01


# ---- Emirmahmutoğlu--Köse oracle -------------------------------------------------------


def ty_unit_wald(yv, xv, k, d, start):
    """Toda--Yamamoto b'V^-1 b with ML residual variance, explicit constant column."""
    p = k + d
    usable = len(yv) - start
    z = np.column_stack([np.ones(usable), *lags_of(yv, p, start), *lags_of(xv, p, start)])
    b = np.linalg.lstsq(z, yv[start:], rcond=None)[0]
    u = yv[start:] - z @ b
    v = np.linalg.inv(z.T @ z) * (u @ u / usable)
    idx = [1 + p + j for j in range(k)]
    return b[idx] @ np.linalg.solve(v[np.ix_(idx, idx)], b[idx])


def ek_bootstrap_oracle(ys, xs, ks, ds, index):
    units, total = ys.shape
    start = max(k + d for k, d in zip(ks, ds))
    usable = total - start
    coefs, resids = [], []
    for yv, xv, k, d in zip(ys, xs, ks, ds):
        p = k + d
        xl = lags_of(xv, p, start)[k:]
        z = np.column_stack([np.ones(usable), *lags_of(yv, p, start), *xl])
        b = np.linalg.lstsq(z, yv[start:], rcond=None)[0]
        u = yv[start:] - z @ b
        coefs.append(b)
        resids.append(u - u.mean())
    resids = np.array(resids)
    out = []
    for draw in index:
        shocks = resids[:, draw]
        lam = 0.0
        for i, (yv, xv, k, d) in enumerate(zip(ys, xs, ks, ds)):
            p = k + d
            b = coefs[i]
            sim = yv.astype(float).copy()
            for t in range(start, total):
                sim[t] = b[0] + sum(b[j] * sim[t - j] for j in range(1, p + 1)) \
                    + sum(b[1 + p + (j - k - 1)] * xv[t - j] for j in range(k + 1, p + 1)) \
                    + shocks[i, t - start]
            lam += -2 * chi2.logsf(ty_unit_wald(sim, xv, k, d, start), k)
        out.append(lam)
    return np.array(out)


def test_ek_unit_wald_fisher_and_bootstrap_match_oracle_with_heterogeneous_orders():
    frame = panel(units=3, periods=34, seed=31)
    lags = {"u0": 1, "u1": 2, "u2": 1}
    dmax = {"u0": 1, "u1": 0, "u2": 2}
    draws, seed = 99, 5
    result = ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=lags, dmax=dmax,
                         bootstrap=draws, seed=seed, direction="x_to_y")
    ks, ds = [1, 2, 1], [1, 0, 2]
    ys, xs = arrays(frame)
    start = 3
    walds = np.array([ty_unit_wald(ys[i], xs[i], ks[i], ds[i], start) for i in range(3)])
    individual = result["individual"]
    assert_allclose(individual.statistic, walds, rtol=1e-9)
    assert_allclose(individual.p_value, [chi2.sf(w, k) for w, k in zip(walds, ks)], rtol=1e-9)
    fisher = -2 * sum(chi2.logsf(w, k) for w, k in zip(walds, ks))
    test = result["tests"].iloc[0]
    assert_allclose(test.statistic, fisher, rtol=1e-10)
    assert_allclose(test.asymptotic_p_value, chi2.sf(fisher, 6), rtol=1e-9)
    gen = torch.Generator().manual_seed(seed)
    index = torch.randint(ys.shape[1] - start, (draws, ys.shape[1] - start), generator=gen).numpy()
    assert result.attrs["bootstrap_first_indices"]["x -> y"] == index[0].tolist()
    boot = ek_bootstrap_oracle(ys, xs, ks, ds, index)
    assert_allclose(test.bootstrap_p_value, (1 + (boot >= fisher).sum()) / (draws + 1))
    ordered = np.sort(boot)
    assert_allclose([test.cv_1, test.cv_5, test.cv_10],
                    [ordered[math.ceil(100 * (1 - a) - 1e-9) - 1] for a in (.01, .05, .10)], rtol=1e-8)
    assert result.attrs["common_start"] == start
    roles = result["coefficients"].query("unit == 'u2'").role.tolist()
    assert roles.count("tested") == 1 and roles.count("augmentation") == 4
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{table}" in result.to_latex()


def test_ek_lag_selection_matches_numpy_varsoc_criterion():
    frame = panel(units=3, periods=50, seed=7)
    result = ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=None, maxlag=3,
                         criterion="aic", dmax=1, bootstrap=99, direction="y_to_x")
    ys, xs = arrays(frame)
    for i in range(3):
        values = np.column_stack([ys[i], xs[i]])
        n, top = len(values), 3
        usable = n - top
        best = None
        for lag in range(1, top + 1):
            z = np.column_stack([np.ones(usable)] + [values[top - j:n - j] for j in range(1, lag + 1)])
            u = values[top:] - z @ np.linalg.lstsq(z, values[top:], rcond=None)[0]
            logdet = np.linalg.slogdet(u.T @ u / usable)[1]
            ll = -usable / 2 * (logdet + 2 * math.log(2 * math.pi) + 2)
            aic = (-2 * ll + 2 * 2 * z.shape[1]) / usable
            if best is None or aic < best[0]:
                best = (aic, lag)
        assert result.attrs["unit_lags"][f"u{i}"] == best[1]


def test_ek_independent_fisher_has_correct_asymptotic_reference_under_independence():
    frame = panel(units=6, periods=60, seed=2, rho=0.0, causal=0.0)
    result = ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1, dmax=1,
                         bootstrap=199, seed=1)
    tests = result["tests"]
    assert (tests.df == 12).all()
    assert (np.abs(tests.bootstrap_p_value - tests.asymptotic_p_value) < .2).all()


# ---- Fourier Toda--Yamamoto oracle -----------------------------------------------------


def series(n=160, seed=5, causal=.4):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 3))
    v = np.zeros((n, 3))
    for t in range(1, n):
        shift = 1.5 * np.sin(2 * np.pi * 2 * t / n)
        v[t, 1] = .9 * v[t - 1, 1] + e[t, 1]
        v[t, 2] = .5 * v[t - 1, 2] + e[t, 2]
        v[t, 0] = shift + .5 * v[t - 1, 0] + causal * v[t - 1, 1] + e[t, 0]
    return pd.DataFrame(v, columns=["a", "b", "c"])


def fourier(n, k, cumulative):
    t = np.arange(1, n + 1)
    cols = []
    for j in (range(1, k + 1) if cumulative else [k]):
        cols += [np.sin(2 * np.pi * j * t / n), np.cos(2 * np.pi * j * t / n)]
    return np.column_stack(cols)


def fty_design(values, order, terms):
    n, k = values.shape
    lagged = [values[order - j:n - j, v] for v in range(k) for j in range(1, order + 1)]
    return np.column_stack([*lagged, np.ones(n - order), terms[order:]])


def fty_oracle(values, p, d, k, cumulative, effect, cause):
    n, kv = values.shape
    order = p + d
    z = fty_design(values, order, fourier(n, k, cumulative))
    y = values[order:, effect]
    b = np.linalg.lstsq(z, y, rcond=None)[0]
    u = y - z @ b
    df = len(y) - z.shape[1]
    v = np.linalg.inv(z.T @ z) * (u @ u / df)
    idx = [cause * order + j for j in range(p)]
    wald = b[idx] @ np.linalg.solve(v[np.ix_(idx, idx)], b[idx])
    keep = [j for j in range(z.shape[1]) if j not in idx]
    br = np.linalg.lstsq(z[:, keep], y, rcond=None)[0]
    rss_r = ((y - z[:, keep] @ br) ** 2).sum()
    return wald, (rss_r - u @ u) / (u @ u / df), z, keep, u


@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize("p,d,k", [(1, 1, 1), (2, 1, 2), (2, 2, 3)])
def test_fourier_ty_wald_matches_quadratic_form_and_tspdlib_rss_form(cumulative, p, d, k):
    frame = series()
    result = fouriertycausality(data=frame, y=["a", "b", "c"], lags=p, dmax=d, frequency=k,
                                cumulative=cumulative, bootstrap=0)
    values = frame.to_numpy()
    for row in result["tests"].itertuples():
        e, c = "abc".index(row.effect), "abc".index(row.cause)
        wald, rss_form, *_ = fty_oracle(values, p, d, k, cumulative, e, c)
        assert_allclose(row.statistic, wald, rtol=1e-9)
        assert_allclose(wald, rss_form, rtol=1e-9)
        assert_allclose(row.asymptotic_p_value, chi2.sf(wald, p), rtol=1e-8)
    assert len(result["tests"]) == 6
    z = fty_design(values, p + d, fourier(len(values), k, cumulative))
    b = np.linalg.lstsq(z, values[p + d:], rcond=None)[0]
    assert_allclose(result["coefficients"].estimate, b.T.reshape(-1), rtol=1e-8, atol=1e-10)


def test_fourier_ty_frequency_and_lag_selection_and_fixed_design_bootstrap():
    frame = series()[["a", "b"]]
    draws, seed = 199, 9
    result = fouriertycausality(data=frame, y=["a", "b"], lags=None, maxlag=3, frequency=None,
                                kmax=3, criterion="aic", bootstrap=draws, seed=seed)
    values = frame.to_numpy()
    n, top = len(values), 3
    usable = n - top
    grid = {}
    for k in (1, 2, 3):
        terms = fourier(n, k, False)
        for lag in (1, 2, 3):
            z = np.column_stack([np.ones(usable), terms[top:]]
                                + [values[top - j:n - j] for j in range(1, lag + 1)])
            u = values[top:] - z @ np.linalg.lstsq(z, values[top:], rcond=None)[0]
            logdet = np.linalg.slogdet(u.T @ u / usable)[1]
            ll = -usable / 2 * (logdet + 2 * math.log(2 * math.pi) + 2)
            grid[(k, lag)] = (-2 * ll + 2 * 2 * z.shape[1]) / usable
    selection = result["selection"]
    assert_allclose(selection.aic, [grid[(k, lag)] for k, lag in zip(selection.frequency, selection.lags)],
                    rtol=1e-10)
    chosen = min(grid, key=lambda key: (grid[key], key))
    assert (result.attrs["frequency"], result.attrs["lags"]) == chosen == (2, 1)
    gen = torch.Generator().manual_seed(seed)
    for row in result["tests"].itertuples():
        e, c = "ab".index(row.effect), "ab".index(row.cause)
        wald, _, z, keep, u = fty_oracle(values, chosen[1], 1, chosen[0], False, e, c)
        index = torch.randint(len(u), (draws, len(u)), generator=gen).numpy()
        centred = u - u.mean()
        boot = []
        for draw in index:
            star = centred[draw] - centred[draw].mean()
            # Restricted fitted values added explicitly, as in the authors' bootstrap.
            y = values[1 + chosen[1]:, e]
            restricted = z[:, keep] @ np.linalg.lstsq(z[:, keep], y, rcond=None)[0]
            ystar = restricted + star
            bu = np.linalg.lstsq(z, ystar, rcond=None)[0]
            uu = ystar - z @ bu
            br = np.linalg.lstsq(z[:, keep], ystar, rcond=None)[0]
            ur = ystar - z[:, keep] @ br
            boot.append((ur @ ur - uu @ uu) / (uu @ uu / (len(ystar) - z.shape[1])))
        boot = np.array(boot)
        assert_allclose(row.bootstrap_p_value, (1 + (boot >= wald).sum()) / (draws + 1))
        ordered = np.sort(boot)
        assert_allclose(row.cv_5, ordered[math.ceil(200 * .95 - 1e-9) - 1], rtol=1e-7)
    assert result["tests"].set_index("cause").bootstrap_p_value["b"] < .01
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{table}" in result.to_latex()


# ---- rolling-window oracle --------------------------------------------------------------


def window_lr(w, p):
    n = len(w)
    z = np.column_stack([np.ones(n - p), *lags_of(w[:, 0], p, p), *lags_of(w[:, 1], p, p)])
    y = w[p:, 0]
    uu = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    zr = z[:, :1 + p]
    ur = y - zr @ np.linalg.lstsq(zr, y, rcond=None)[0]
    df = len(y) - z.shape[1]
    return df * math.log((ur @ ur) / (uu @ uu)), df * (ur @ ur - uu @ uu) / (uu @ uu), z


def window_boot(w, p, index):
    n = len(w)
    lr, _, z = window_lr(w, p)
    zr = z[:, :1 + p]
    b0 = np.linalg.lstsq(zr, w[p:, 0], rcond=None)[0]
    b1 = np.linalg.lstsq(z, w[p:, 1], rcond=None)[0]
    res = np.column_stack([w[p:, 0] - zr @ b0, w[p:, 1] - z @ b1])
    res -= res.mean(0)
    out = []
    for draw in index:
        sim = w.copy()
        shocks = res[draw]
        for t in range(p, n):
            sim[t, 0] = b0[0] + sum(b0[j] * sim[t - j, 0] for j in range(1, p + 1)) + shocks[t - p, 0]
            sim[t, 1] = b1[0] + sum(b1[j] * sim[t - j, 0] for j in range(1, p + 1)) \
                + sum(b1[p + j] * sim[t - j, 1] for j in range(1, p + 1)) + shocks[t - p, 1]
        out.append(window_lr(sim, p)[0])
    return lr, np.array(out)


@pytest.mark.parametrize("p", [1, 2])
def test_rolling_windows_statistics_and_bootstrap_sequence_match_oracle(p):
    frame = series(n=90)[["a", "b"]].rename(columns={"a": "y", "b": "x"})
    frame["period"] = np.arange(1000, 1090)
    draws, seed, window, step = 99, 4, 40, 10
    result = rollingcausality(data=frame.sample(frac=1, random_state=3), y="y", x="x",
                              window=window, lags=p, step=step, time="period",
                              bootstrap=draws, seed=seed)
    rows = result["windows"]
    assert len(rows) == (90 - window) // step + 1 == result.attrs["windows"]
    values = frame[["y", "x"]].to_numpy()
    gen = torch.Generator().manual_seed(seed)
    for row in rows.itertuples():
        w = values[row.start_row:row.start_row + window]
        assert row.start == 1000 + row.start_row and row.end == 1000 + row.start_row + window - 1
        index = torch.randint(window - p, (2, draws, window - p), generator=gen).numpy()
        for direction, oriented in (("x_to_y", w), ("y_to_x", w[:, ::-1].copy())):
            slot = 0 if direction == "x_to_y" else 1
            lr, boot = window_boot(oriented, p, index[slot])
            wald = window_lr(oriented, p)[1]
            assert_allclose(getattr(row, f"lr_{direction}"), lr, rtol=1e-9, atol=1e-12)
            assert_allclose(getattr(row, f"wald_{direction}"), wald, rtol=1e-9, atol=1e-12)
            assert_allclose(getattr(row, f"asymptotic_p_{direction}"), chi2.sf(lr, p), rtol=1e-8)
            assert_allclose(getattr(row, f"bootstrap_p_{direction}"),
                            (1 + (boot >= lr - 1e-10 * abs(lr)).sum()) / (draws + 1))
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{table}" in result.to_latex()


def test_rolling_output_and_work_are_bounded_before_allocation():
    frame = series(n=120)[["a", "b"]].rename(columns={"a": "y", "b": "x"})
    with pytest.raises(AnalysisError) as error:
        rollingcausality(data=frame, y="y", x="x", window=30, max_windows=10, bootstrap=99)
    assert error.value.code == "too_many_windows"
    with pytest.raises(AnalysisError) as error:
        rollingcausality(data=frame, y="y", x="x", window=30, step=10, bootstrap=99, max_work=1000)
    assert error.value.code == "work_budget_exceeded"
    with pytest.raises(AnalysisError) as error:
        rollingcausality(data=frame, y="y", x="x", window=8, lags=2, bootstrap=99)
    assert error.value.code == "insufficient_observations"


# ---- Hacker--Hatemi-J leveraged bootstrap oracle ----------------------------------------


def hh_oracle(levels, base, dmax, index, recursive=True):
    """NumPy replica of the authors' GAUSS steps, with observation-specific leverages."""
    p = base + dmax

    def varlags(z, lags):
        return z[lags:], np.column_stack([z[lags - j:len(z) - j] for j in range(1, lags + 1)])

    def wtest(y, x):
        a = np.linalg.lstsq(x, y, rcond=None)[0].T
        res = y - x @ a.T
        est = res.T @ res / (len(y) - a.shape[1])
        vec = a.T.reshape(-1)
        r = np.zeros((base, vec.size))
        for o in range(base):
            r[o, (1 + o * 2 + 1) * 2] = 1
        f1 = r @ vec
        return f1 @ np.linalg.solve(r @ np.kron(np.linalg.inv(x.T @ x), est) @ r.T, f1)

    y_t, xl = varlags(levels, p)
    x_t = np.column_stack([np.ones(len(y_t)), xl])
    stat = wtest(y_t, x_t)
    y_s, xs = varlags(levels, base)
    x_s = np.column_stack([np.ones(len(y_s)), xs])
    x_r = x_s[:, [0] + [1 + 2 * j for j in range(base)]]
    b1 = np.linalg.lstsq(x_r, y_s[:, 0], rcond=None)[0]
    b2 = np.linalg.lstsq(x_s, y_s[:, 1], rcond=None)[0]
    h1 = np.einsum("ij,ji->i", x_r, np.linalg.inv(x_r.T @ x_r) @ x_r.T)
    h2 = np.einsum("ij,ji->i", x_s, np.linalg.inv(x_s.T @ x_s) @ x_s.T)
    a = np.zeros((2, x_t.shape[1]))
    a[0, [0] + [1 + 2 * j for j in range(base)]] = b1
    a[1, :1 + 2 * base] = b2
    res = np.column_stack([y_s[:, 0] - x_r @ b1, y_s[:, 1] - x_s @ b2])
    adj = (res / np.sqrt(1 - np.column_stack([h1, h2])))[dmax:]
    out = []
    for draw in index:
        sim = np.column_stack([adj[draw[:, 0], 0], adj[draw[:, 1], 1]])
        sim -= sim.mean(0)
        if not recursive:
            # Written form Y* = D^ Z + delta*: observed lagged regressors held fixed.
            out.append(wtest(x_t @ a.T + sim, x_t))
            continue
        rows = [x_t[0]]
        for t in range(len(sim)):
            value = rows[t] @ a.T + sim[t]
            rows.append(np.r_[1, value, rows[t][1:1 + 2 * (p - 1)]])
        rows = np.array(rows)
        out.append(wtest(rows[1:, 1:3], rows[:-1]))
    return stat, np.array(out)


@pytest.mark.parametrize("signs,base,dmax", [(("+", "+"), 2, 1), (("-", "+"), 1, 2)])
def test_asymmetric_leveraged_bootstrap_matches_author_algorithm(signs, base, dmax):
    rng = np.random.default_rng(12)
    data = pd.DataFrame(dict(y=rng.normal(size=100).cumsum(), x=rng.normal(size=100).cumsum(),
                             t=np.arange(100)))
    draws, seed = 199, 3
    result = asymcausality(data=data, y="y", x="x", time="t", lags=base, dmax=dmax, signs=signs,
                           bootstrap=draws, seed=seed, resampling="leverage", max_work=10**12)
    diff = np.diff(data[["y", "x"]].to_numpy(), axis=0)
    split = np.column_stack([np.maximum(diff[:, j], 0) if s == "+" else np.minimum(diff[:, j], 0)
                             for j, s in enumerate(signs)])
    levels = np.vstack([np.zeros(2), split.cumsum(0)])
    rows = len(levels) - base - dmax
    index = torch.randint(rows, (draws, rows, 2), generator=torch.Generator().manual_seed(seed)).numpy()
    stat, boot = hh_oracle(levels, base, dmax, index)
    row = result["tests"].iloc[0]
    assert_allclose(row.statistic, stat, rtol=1e-9)
    assert_allclose(result.attrs["bootstrap_statistics"], boot, rtol=1e-7, atol=1e-10)
    ordered = np.sort(boot)
    expected = [(ordered[draws - draws // k - 1] + ordered[draws - draws // k]) / 2 for k in (100, 20, 10)]
    assert_allclose([row.cv_1, row.cv_5, row.cv_10], expected, rtol=1e-7)
    assert_allclose(row.bootstrap_p_value, (1 + (boot >= stat).sum()) / (draws + 1))
    assert result.attrs["resampling"] == "leverage"
    json.dumps(result.attrs, allow_nan=False)


def test_asymmetric_leveraged_fixed_design_matches_written_form():
    rng = np.random.default_rng(14)
    data = pd.DataFrame(dict(y=rng.normal(size=90).cumsum(), x=rng.normal(size=90).cumsum()))
    draws, seed = 199, 6
    result = asymcausality(data=data, y="y", x="x", lags=2, dmax=1, bootstrap=draws, seed=seed,
                           resampling="leverage", leverage_design="fixed", max_work=10**12)
    diff = np.diff(data[["y", "x"]].to_numpy(), axis=0)
    levels = np.vstack([np.zeros(2), np.maximum(diff, 0).cumsum(0)])
    rows = len(levels) - 3
    index = torch.randint(rows, (draws, rows, 2), generator=torch.Generator().manual_seed(seed)).numpy()
    stat, boot = hh_oracle(levels, 2, 1, index, recursive=False)
    assert_allclose(result["tests"].statistic.iloc[0], stat, rtol=1e-9)
    assert_allclose(result.attrs["bootstrap_statistics"], boot, rtol=1e-8, atol=1e-10)
    assert result.attrs["leverage_design"] == "fixed"
    with pytest.raises(AnalysisError):
        asymcausality(data=data, y="y", x="x", bootstrap=99, resampling="leverage",
                      leverage_design="wild")


def test_asymmetric_default_iid_route_is_unchanged_and_option_is_validated():
    rng = np.random.default_rng(12)
    data = pd.DataFrame(dict(y=rng.normal(size=80).cumsum(), x=rng.normal(size=80).cumsum()))
    plain = asymcausality(data=data, y="y", x="x", bootstrap=49, seed=11)
    assert "resampling" not in plain.attrs and len(plain.attrs["bootstrap_statistics"]) == 49
    with pytest.raises(AnalysisError):
        asymcausality(data=data, y="y", x="x", bootstrap=99, resampling="wild")
    with pytest.raises(AnalysisError):
        asymcausality(data=data, y="y", x="x", bootstrap=49, resampling="leverage")


# ---- shared contracts --------------------------------------------------------------------


def test_chi2_log_tail_is_stable_far_beyond_float_underflow():
    x = torch.tensor([0., 1e-9, .5, 3., 40., 900., 2500.], dtype=torch.float64)
    for df in range(1, 13):
        got = bk.chi2_logsf_integer(x, df).numpy()
        finite = np.isfinite(chi2.logsf(x.numpy(), df))
        assert_allclose(got[finite], chi2.logsf(x.numpy(), df)[finite], rtol=1e-11, atol=1e-13)
        assert np.isfinite(got).all() and got[-1] < -1000


def test_seeded_draws_are_reproducible_and_leave_global_rng_untouched():
    frame = panel()
    state = torch.random.get_rng_state().clone()
    one = ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99, seed=3)
    two = ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99, seed=3)
    assert torch.equal(state, torch.random.get_rng_state())
    assert one.attrs["bootstrap_index_sha256"] == two.attrs["bootstrap_index_sha256"]
    pd.testing.assert_frame_equal(one["tests"], two["tests"])
    other = ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99, seed=4)
    assert one.attrs["bootstrap_index_sha256"] != other.attrs["bootstrap_index_sha256"]


@pytest.mark.parametrize("function", [konyacausality, ekcausality])
def test_panel_failure_contract(function):
    frame = panel()
    base = dict(y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99)
    cases = [
        (frame.assign(y=np.where(np.arange(len(frame)) == 5, np.nan, frame.y)), "missing_values"),
        (frame.iloc[1:], "unbalanced_panel"),
        (frame.assign(t=np.where(frame.unit == "u1", frame.t + 1, frame.t)), "unaligned_panel"),
    ]
    for data, code in cases:
        with pytest.raises(AnalysisError) as error:
            function(data=data, **base)
        assert error.value.code == code
    for option in ({"bootstrap": 10}, {"bootstrap": True}, {"direction": "up"}, {"seed": -1},
                   {"alpha": 1.5}, {"y": "x"}, {"lags": 0}):
        with pytest.raises(AnalysisError):
            function(data=frame, **{**base, **option})
    with pytest.raises(AnalysisError) as error:
        function(data=frame, **base, max_work=10)
    assert error.value.code == "work_budget_exceeded"


def test_konya_needs_more_periods_than_units_for_sur_covariance():
    frame = panel(units=12, periods=12)
    with pytest.raises(AnalysisError) as error:
        konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99)
    assert error.value.code == "insufficient_observations"


def test_failed_bootstrap_draw_is_never_silently_dropped(monkeypatch):
    from openecon.econometrics.var import panel_bootstrap_causality as module
    original = module.bk.unit_wald
    calls = {"n": 0}

    def flaky(*args):
        stats, ok = original(*args)
        calls["n"] += 1
        if calls["n"] > 1:                      # observed fit passes, bootstrap fails
            ok = torch.zeros_like(ok)
        return stats, ok

    monkeypatch.setattr(module.bk, "unit_wald", flaky)
    with pytest.raises(AnalysisError) as error:
        konyacausality(data=panel(), y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99,
                       direction="x_to_y")
    assert error.value.code == "bootstrap_failure"


def test_fourier_and_rolling_input_contracts():
    frame = series()
    for option in ({"dmax": 3}, {"kmax": 6}, {"frequency": 0}, {"criterion": "hq"},
                   {"y": ["a"]}, {"lags": 13}, {"bootstrap": 50}, {"cumulative": 1}):
        with pytest.raises(AnalysisError):
            fouriertycausality(data=frame, **{"y": ["a", "b"], "bootstrap": 0, **option})
    with pytest.raises(AnalysisError) as error:
        fouriertycausality(data=frame.assign(a=np.nan), y=["a", "b"], lags=1, frequency=1, bootstrap=0)
    assert error.value.code == "missing_values"
    with pytest.raises(AnalysisError) as error:
        fouriertycausality(data=frame, y=["a", "b"], lags=1, frequency=1, bootstrap=999, max_work=10)
    assert error.value.code == "work_budget_exceeded"


def test_committed_size_power_receipt_is_complete():
    from pathlib import Path
    path = Path(__file__).parent / "fixtures" / "causality_extensions" / "size_power.json"
    record = json.loads(path.read_text())
    script = Path(__file__).parents[1] / "scripts" / "generate_causality_extensions_oracles.py"
    assert script.exists() and len(record["script_sha256"]) == 64
    for cell in record["cells"]:
        assert cell["planned_datasets"] == cell["completed_datasets"] and not cell["failed_calls"]
        low, high = cell["wilson95"]
        if "null" in cell["cell"] and "asymptotic" not in cell["cell"]:
            assert low <= .05 <= high <= .10, cell["cell"]
        if "power" in cell["cell"]:
            assert low > .30, cell["cell"]


def test_portable_summary_state_restores_tables_and_bootstrap_settings():
    frame = panel()
    for result in (konyacausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1,
                                  bootstrap=99, direction="x_to_y"),
                   rollingcausality(data=series(n=70)[["a", "b"]], y="a", x="b", window=40,
                                    step=15, bootstrap=99)):
        restored = oe.restore_summary(oe.summary_state(result))
        assert restored.attrs == json.loads(json.dumps(result.attrs))
        for name, table in result.items():
            pd.testing.assert_frame_equal(restored[name].reset_index(drop=True),
                                          table.reset_index(drop=True), check_dtype=False)


def test_public_exports_and_summary_rendering():
    frame = panel()
    for name in ("konyacausality", "ekcausality", "fouriertycausality", "rollingcausality"):
        assert callable(getattr(oe, name))
    result = oe.ekcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=1, bootstrap=99)
    text = str(result)
    assert "Fisher" in text and "bootstrap" in text
