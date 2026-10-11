"""Break, Fourier and threshold cointegration tests against independent references.

References: explicit NumPy re-implementations below (loops over candidate dates),
the committed R oracles in tests/fixtures/cointegration_breaks/oracles.json
(Bruce Hansen's published Gregory-Hansen procedures, CRAN makicoint 2.0.0,
hatemicoint 1.1.0, cointests 1.1.0, urca ``ur.df``/``ca.jo`` and the
Bayer-Hanck authors' null draws), and the transcription file
critical_values.json. Regenerate with scripts/generate_cointegration_breaks_oracles.py.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot import bayer_hanck_null as bh_null
from openecon.econometrics.unitroot import coint_breaks_tables as tables
from openecon.econometrics.unitroot import fourier_coint
from openecon.econometrics.unitroot.coint_kernels import gauss_round

FIXTURES = Path(__file__).parent / "fixtures" / "cointegration_breaks"
ORACLES = json.loads((FIXTURES / "oracles.json").read_text())
CRITICAL = FIXTURES / "critical_values.json"


def frame(name: str) -> tuple[pd.DataFrame, list[str]]:
    data = ORACLES["datasets"][name]
    columns = {"y": data["y"]}
    for index, values in enumerate(data["x"]):
        columns[f"x{index + 1}"] = values
    return pd.DataFrame(columns), [f"x{i + 1}" for i in range(len(data["x"]))]


def as_list(value):
    return value if isinstance(value, list) else [value]


def simulate(seed: int, n: int, m: int, *, cointegrated: bool = False, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(n, m)), axis=0)
    if cointegrated:
        u = np.zeros(n)
        shocks = rng.normal(size=n)
        for t in range(1, n):
            u[t] = 0.4 * u[t - 1] + shocks[t]
        y = 1.0 + x.sum(axis=1) * 0.7 + shift * (np.arange(n) >= n // 2) + u
    else:
        y = np.cumsum(rng.normal(size=n))
    data = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(m)])
    data.insert(0, "y", y)
    return data, [f"x{i + 1}" for i in range(m)]


# ---- NumPy reference implementations (Hansen's GAUSS procedures) ---------------------------


def np_ols(design, target):
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    df = design.shape[0] - design.shape[1]
    s2 = resid @ resid / df
    cov = s2 * np.linalg.inv(design.T @ design)
    return beta, resid, np.sqrt(np.diag(cov)), cov


def np_adf(e, rule, kmax, critical=1.96, strict=False):
    n = e.shape[0]
    de = np.diff(e)
    results = {}
    for k in range(kmax, -1, -1) if rule != "fixed" else [kmax]:
        target = de[k:]
        cols = [e[k:n - 1]] + [de[k - j:n - 1 - j] for j in range(1, k + 1)]
        beta, resid, se, _ = np_ols(np.column_stack(cols), target)
        rows = n - 1 - k
        t_last = beta[-1] / se[-1]
        results[k] = (beta[0] / se[0], resid @ resid, rows, t_last)
        if rule == "t":
            significant = abs(t_last) > critical if strict else abs(t_last) >= critical
            if k == 0 or significant:
                return results[k][0], k, results[k][1]
    if rule == "fixed":
        return results[kmax][0], kmax, results[kmax][1]
    penalty = (lambda rows: 2.0) if rule == "aic" else math.log
    ic = {k: math.log(v[1] / v[2]) + penalty(v[2]) * (k + 2) / v[2] for k, v in results.items()}
    best = min(sorted(ic), key=lambda k: ic[k])
    return results[best][0], best, results[best][1]


def np_phillips(e):
    n = e.shape[0]
    rho = (e[:-1] @ e[1:]) / (e[:-1] @ e[:-1])
    v = e[1:] - rho * e[:-1]
    nu = v.shape[0]
    bu = (v[:-1] @ v[1:]) / (v[:-1] @ v[:-1])
    a2 = 4 * bu ** 2 / (1 - bu) ** 4
    m = 1.3221 * (a2 * nu) ** 0.2
    lam, j = 0.0, 1
    while j <= m:
        gamma = v[: nu - j] @ v[j:] / nu
        c = j / m
        lam += 75 / (6 * math.pi * c) ** 2 * (math.sin(1.2 * math.pi * c) / (1.2 * math.pi * c)
                                               - math.cos(1.2 * math.pi * c)) * gamma
        j += 1
    p = np.sum(e[:-1] * e[1:] - lam) / (e[:-1] @ e[:-1])
    sigma2 = 2 * lam + v @ v / nu
    return n * (p - 1), (p - 1) / math.sqrt(sigma2 / (e[:-1] @ e[:-1]))


def np_design(y, x, breaks, model):
    n = y.shape[0]
    t = np.arange(1, n + 1, dtype=float)
    cols = [np.ones(n)]
    dummies = [(t > b).astype(float) for b in breaks]
    cols += dummies
    if model in ("trend", "regime_trend"):
        cols.append(t)
    if model == "regime_trend":
        cols += [d * t for d in dummies]
    cols += list(x.T)
    if model in ("regime", "regime_trend"):
        cols += [d * col for d in dummies for col in x.T]
    return np.column_stack(cols)


def np_scan(y, x, model, grid, rule, kmax):
    best = {"adf": (np.inf, None, None), "zt": (np.inf, None), "za": (np.inf, None)}
    for breaks in grid:
        _, e, _, _ = np_ols(np_design(y, x, breaks, model), y)
        adf, lag, _ = np_adf(e, rule, kmax)
        za, zt = np_phillips(e)
        if adf < best["adf"][0]:
            best["adf"] = (adf, tuple(breaks), lag)
        if zt < best["zt"][0]:
            best["zt"] = (zt, tuple(breaks))
        if za < best["za"][0]:
            best["za"] = (za, tuple(breaks))
    return best


def np_maki(y, x, m, model, rule, kmax, trim=0.10):
    n = y.shape[0]
    tb = gauss_round(trim * n)
    located, taus = [], []
    for _ in range(m):
        bounds = [0, *sorted(located), n]
        seg_best = []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            lo, hi = a + tb + 1, b - tb
            if lo > hi:
                continue
            best_tau, best_ssr, best_pos = np.inf, np.inf, None
            for i in range(lo, hi + 1):
                _, e, _, _ = np_ols(np_design(y, x, [*located, i], model), y)
                tau, _, ssr = np_adf(e, rule, kmax, critical=1.654, strict=True)
                best_tau = min(best_tau, tau)
                if ssr < best_ssr:
                    best_ssr, best_pos = ssr, i
            seg_best.append((best_tau, best_pos))
        tau, pos = min(seg_best, key=lambda item: item[0])
        taus.append(tau)
        located.append(pos)
    return min(taus), sorted(located)


# ---- transcriptions -------------------------------------------------------------------------


def test_transcription_file_hash_and_module_tables_agree():
    raw = CRITICAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == tables.TRANSCRIPTION_SHA256
    doc = json.loads(raw)

    def norm(value):
        if isinstance(value, dict):
            return {str(k): norm(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        return value

    assert doc["gregory_hansen"]["table"] == norm(tables.GREGORY_HANSEN)
    assert doc["hatemi_j"]["table"] == norm(tables.HATEMI_J)
    assert doc["maki"]["table"] == norm(tables.MAKI)
    assert doc["bayer_hanck"]["table"] == norm(tables.BAYER_HANCK)
    assert doc["tsong"]["table"] == norm(fourier_coint.TSONG)
    assert doc["fadl"]["table"] == norm(fourier_coint.FADL)
    # Spot values read from the scanned Gregory-Hansen tables.
    assert tables.GREGORY_HANSEN["level"][1] == ((-5.13, -4.83, -4.61, -4.34, -2.25),
                                                 (-50.07, -45.01, -40.48, -36.19, -10.63))
    assert tables.GREGORY_HANSEN["regime_trend"][4][1] == (-100.69, -94.00, -88.47, -82.30, -40.99)
    assert tables.HATEMI_J[1][1][0] is None                       # disputed transcription
    for model in tables.MAKI.values():                              # monotone in the level
        for rows in model.values():
            assert all(a < b < c for a, b, c in rows)


def test_bayer_hanck_null_asset_decodes_to_the_authors_draws():
    draws = bh_null.null_draws()
    assert tuple(draws.shape) == (132, 10000)
    assert bool((draws[:, 1:] >= draws[:, :-1]).all())
    checkpoints = ORACLES["null_checkpoints"]
    stored = draws[:, checkpoints["positions"]].numpy()
    original = np.array(checkpoints["values"])
    assert np.max(np.abs(stored - original)) <= 0.5e-4 + 1e-9
    assert ORACLES["null_payload_sha256"] == bh_null.PAYLOAD_SHA256


# ---- Gregory-Hansen -----------------------------------------------------------------------


@pytest.mark.parametrize("case", ORACLES["oracles"]["gregory_hansen"],
                         ids=lambda c: f"{c['data']}-{c['model']}-{c['rule']}")
def test_gregory_hansen_matches_hansen_published_procedures(case):
    data, names = frame(case["data"])
    lags = case["kmax"] if case["rule"] == "fixed" else case["rule"]
    maxlag = None if case["rule"] == "fixed" else case["kmax"]
    result = oe.gregory_hansen(data, "y", names, model=case["model"], lags=lags, maxlag=maxlag)
    stats, breaks = result.attrs["statistics"], result.attrs["break_index"]
    for name, key in (("ADF*", "adf"), ("Zt*", "zt"), ("Za*", "za")):
        assert stats[name] == pytest.approx(case[key], rel=1e-10, abs=1e-10)
        assert breaks[name] + 1 == as_list(case[f"{key}_breaks"])[0]
    assert result.attrs["lags"] == case["adf_lag"]
    critical = result.attrs["critical_values"]
    m = len(names)
    assert critical["ADF*"]["5%"] == tables.GREGORY_HANSEN[case["model"]][m][0][2]
    assert critical["Za*"]["97.5%"] == tables.GREGORY_HANSEN[case["model"]][m][1][4]


def test_gregory_hansen_numpy_reference_and_break_grid():
    data, names = simulate(11, 60, 2, cointegrated=True, shift=3.0)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    grid = [[b] for b in range(gauss_round(0.15 * 60), gauss_round(0.85 * 60) + 1)]
    for model in ("level", "regime_trend"):
        ref = np_scan(y, x, model, grid, "t", 4)
        out = oe.gregory_hansen(data, "y", names, model=model, maxlag=4)
        assert out.attrs["statistics"]["ADF*"] == pytest.approx(ref["adf"][0], rel=1e-10)
        assert out.attrs["statistics"]["Zt*"] == pytest.approx(ref["zt"][0], rel=1e-10)
        assert out.attrs["statistics"]["Za*"] == pytest.approx(ref["za"][0], rel=1e-10)
        assert out.attrs["break_index"]["ADF*"] + 1 == ref["adf"][1][0]
        assert out.attrs["lags"] == ref["adf"][2]
        assert out.attrs["candidates"] == len(grid)


def test_break_grids_use_the_published_literal_fractions():
    data, names = frame("three_regressors")      # n = 150: 0.15 n = 22.5 and 0.85 n = 127.5
    assert oe.gregory_hansen(data, "y", names).attrs["candidates"] == 128 - 23 + 1
    small, small_names = simulate(2, 105, 1)    # 0.70 n = 73.5: GAUSS rounds half away from 0
    pairs = sum(max(0, 89 - (a + 16) + 1) for a in range(16, 75))
    assert oe.hatemi_j(small, "y", small_names, lags=0).attrs["candidates"] == pairs


def test_gregory_hansen_inference_domain_and_reporting():
    data, names = frame("level_break")
    data["date"] = pd.date_range("2000-01-01", periods=len(data), freq="MS")
    out = oe.gregory_hansen(data, "y", names, time="date", model="regime", regress=True)
    assert out.attrs["break_period"]["ADF*"].startswith("20")
    assert "cointegrating_regression" in out
    assert out.attrs["p_value"] is None and out.attrs["inference_status"].startswith("asymptotic")
    other = oe.gregory_hansen(data, "y", names, trim=0.10)
    assert other.attrs["critical_values"] is None and "0.15" in other.attrs["inference_status"]
    wide, wide_names = simulate(3, 120, 5)
    many = oe.gregory_hansen(wide, "y", wide_names)
    assert many.attrs["critical_values"] is None
    json.dumps(out.attrs, allow_nan=False)
    assert "tabular" in out.to_latex() and "Gregory-Hansen" in str(out)
    restored = oe.restore_summary(oe.summary_state(out))
    assert restored.attrs["statistics"] == out.attrs["statistics"]


# ---- Hatemi-J ------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ORACLES["oracles"]["hatemi_j"],
                         ids=lambda c: f"{c['data']}-{c['rule']}{c['kmax']}")
def test_hatemi_j_matches_author_procedure_and_hatemicoint(case):
    data, names = frame(case["data"])
    lags = case["kmax"] if case["rule"] == "fixed" else case["rule"]
    maxlag = None if case["rule"] == "fixed" else case["kmax"]
    result = oe.hatemi_j(data, "y", names, lags=lags, maxlag=maxlag)
    stats, breaks = result.attrs["statistics"], result.attrs["break_index"]
    for name, key in (("ADF*", "adf"), ("Zt*", "zt"), ("Za*", "za")):
        assert stats[name] == pytest.approx(case[key], rel=1e-10, abs=1e-10)
        assert [b + 1 for b in breaks[name]] == case[f"{key}_breaks"]
    if "hatemicoint_adf" in case:
        assert stats["ADF*"] == pytest.approx(case["hatemicoint_adf"], rel=1e-10)
        assert [b + 1 for b in breaks["ADF*"]] == case["hatemicoint_adf_breaks"]
    critical = result.attrs["critical_values"]
    if len(names) == 1:
        assert critical["Za*"]["1%"] is None and critical["Za*"]["5%"] == -76.003
        assert "withheld" in result.attrs["inference_status"]


def test_hatemi_j_numpy_reference_grid():
    data, names = simulate(5, 44, 1, cointegrated=True, shift=2.0)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    n = 44
    begin = gauss_round(0.15 * n)
    grid = [[a, b] for a in range(begin, gauss_round(0.70 * n) + 1)
            for b in range(a + begin, gauss_round(0.85 * n) + 1)]
    ref = np_scan(y, x, "regime", grid, "aic", 2)
    out = oe.hatemi_j(data, "y", names, lags="aic", maxlag=2)
    assert out.attrs["candidates"] == len(grid)
    assert out.attrs["statistics"]["ADF*"] == pytest.approx(ref["adf"][0], rel=1e-10)
    assert out.attrs["statistics"]["Za*"] == pytest.approx(ref["za"][0], rel=1e-10)
    assert tuple(b + 1 for b in out.attrs["break_index"]["Zt*"]) == ref["zt"][1]


# ---- Maki ----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ORACLES["oracles"]["maki"],
                         ids=lambda c: f"{c['data']}-{c['model']}-m{c['breaks']}")
def test_maki_matches_makicoint_author_engine(case):
    data, names = frame(case["data"])
    lags = case["kmax"] if case["rule"] == "fixed" else "t"
    maxlag = None if case["rule"] == "fixed" else case["kmax"]
    result = oe.maki_coint(data, "y", names, breaks=case["breaks"], model=case["model"],
                           lags=lags, maxlag=maxlag)
    assert result.attrs["statistic"] == pytest.approx(case["statistic"], rel=1e-10, abs=1e-10)
    assert [b + 1 for b in result.attrs["break_index"]] == as_list(case["break_dates"])
    critical = result.attrs["critical_values"]
    assert [critical[level] for level in ("1%", "5%", "10%")] == case["critical"]


def test_maki_numpy_reference_sequential_search():
    data, names = simulate(9, 70, 1, cointegrated=True, shift=2.5)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    statistic, located = np_maki(y, x, 3, "regime", "t", 4)
    out = oe.maki_coint(data, "y", names, breaks=3, model="regime", maxlag=4)
    assert out.attrs["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert [b + 1 for b in out.attrs["break_index"]] == located
    assert [step["step"] for step in out.attrs["steps"]] == [1, 2, 3]


# ---- Bayer-Hanck ---------------------------------------------------------------------------


def np_johansen_max(z, lags, trend):
    n, k = z.shape
    dz = np.diff(z, axis=0)
    rows = np.arange(lags + 1, n)                       # 0-based observations used
    z0 = dz[rows - 1]
    z1 = z[rows - 1]
    pieces = [dz[rows - 1 - i] for i in range(1, lags + 1)]
    if trend == "trend":
        pieces.append((rows + 1.0)[:, None])
    if trend in ("constant", "trend"):
        pieces.append(np.ones((rows.size, 1)))
    if pieces:
        z2 = np.column_stack(pieces)
        r0 = z0 - z2 @ np.linalg.lstsq(z2, z0, rcond=None)[0]
        r1 = z1 - z2 @ np.linalg.lstsq(z2, z1, rcond=None)[0]
    else:
        r0, r1 = z0, z1
    t = rows.size
    s00, s01, s11 = r0.T @ r0 / t, r0.T @ r1 / t, r1.T @ r1 / t
    eig = np.linalg.eigvals(np.linalg.solve(s11, s01.T) @ np.linalg.solve(s00, s01))
    return -t * math.log(1 - max(eig.real))


def np_bayer_hanck(z, lags, trend):
    n = z.shape[0]
    det = {"none": np.zeros((n, 0)), "constant": np.ones((n, 1)),
           "trend": np.column_stack([np.ones(n), np.arange(1, n + 1)])}[trend]
    _, e, _, _ = np_ols(np.column_stack([z[:, 1:], det]), z[:, 0])
    eg, _, _ = np_adf(e, "fixed", lags)
    dz = np.diff(z, axis=0)
    rows = np.arange(lags + 1, n)
    w = [dz[rows - 1, 1:]] + [dz[rows - 1 - i] for i in range(1, lags + 1)]
    d = det[rows]
    wd = np.column_stack([*w, d])
    resid = lambda v: v - wd @ np.linalg.lstsq(wd, v, rcond=None)[0]  # noqa: E731
    r = np.column_stack([resid(z[rows - 1, j]) for j in range(z.shape[1])])
    xf = np.column_stack([r, d])
    beta, _, se, cov = np_ols(xf, resid(dz[rows - 1, 0]))
    return eg, np_johansen_max(z, lags, trend), beta[0] / se[0], beta @ np.linalg.solve(cov, beta)


@pytest.mark.parametrize("case", ORACLES["oracles"]["bayer_hanck"],
                         ids=lambda c: f"{c['data']}-{c['trend']}-L{c['lags']}")
def test_bayer_hanck_components_match_urca_numpy_and_author_null(case):
    data, names = frame(case["data"])
    result = oe.bayer_hanck(data, "y", names, lags=case["lags"], trend=case["trend"])
    comp, pvals = result.attrs["components"], result.attrs["component_p_values"]
    reference = {"Engle-Granger": case["engle_granger"], "Johansen": case["johansen_max"],
                 "Banerjee": case["banerjee"], "Boswijk": case["boswijk"]}
    z = data[["y", *names]].to_numpy()
    eg, jo, ba, bo = np_bayer_hanck(z, case["lags"], case["trend"])
    for value, name in ((eg, "Engle-Granger"), (jo, "Johansen"), (ba, "Banerjee"),
                        (bo, "Boswijk")):
        assert comp[name] == pytest.approx(value, rel=1e-9)
    for j, (name, value) in enumerate(reference.items()):
        if value is None:
            continue
        assert comp[name] == pytest.approx(value, rel=1e-10)
        # Stored draws are rounded to 1e-4: at most one draw changes side.
        assert abs(pvals[name] - case["p_values"][j]) <= 1e-4 + 1e-12
    fisher = -2 * (math.log(pvals["Engle-Granger"]) + math.log(pvals["Johansen"]))
    assert result.attrs["statistics"]["EG-J"] == pytest.approx(fisher, rel=1e-12)
    crit = result.attrs["critical_values"]
    column = ("none", "constant", "trend").index(case["trend"])
    assert crit["EG-J-Ba-Bo"]["5%"] == tables.BAYER_HANCK["5%"][1][len(names) - 1][column]


def test_bayer_hanck_p_value_counts_follow_the_ado_tail_rules():
    draws = bh_null.null_draws()
    column = 44 * 1 + 4 * 0                                   # constant, N = 2, EG
    statistic = float(draws[column, 500]) + 1e-9
    expected = float((draws[column] < statistic).sum()) / 10000 + 1e-12
    assert bh_null.component_p_value(statistic, 0, 2, "constant") == pytest.approx(expected)
    upper = float(draws[column + 1, 9000]) - 1e-9
    expected = float((draws[column + 1] > upper).sum()) / 10000 + 1e-12
    assert bh_null.component_p_value(upper, 1, 2, "constant") == pytest.approx(expected)
    assert bh_null.component_p_value(-100.0, 0, 2, "constant") == pytest.approx(1e-12)


def test_ecm_cointegration_standalone_and_reporting():
    data, names = frame("three_regressors")
    full = oe.bayer_hanck(data, "y", names, lags=2, trend="trend")
    for test, name in (("banerjee", "Banerjee"), ("boswijk", "Boswijk")):
        single = oe.ecm_cointegration(data, "y", names, test=test, lags=2, trend="trend")
        assert single.attrs["statistic"] == full.attrs["components"][name]
        assert single.attrs["p_value"] == full.attrs["component_p_values"][name]
    json.dumps(full.attrs, allow_nan=False)
    assert "tabular" in full.to_latex()
    restored = oe.restore_summary(oe.summary_state(full))
    assert restored.attrs["components"] == full.attrs["components"]


# ---- Fourier tests -------------------------------------------------------------------------


def fourier_terms(n, k):
    t = np.arange(1, n + 1)
    return np.column_stack([np.sin(2 * np.pi * k * t / n), np.cos(2 * np.pi * k * t / n)])


@pytest.mark.parametrize("case", ORACLES["oracles"]["fourier"],
                         ids=lambda c: f"{c['test']}-{c['data']}-{c['model']}")
def test_fourier_tests_match_cointests(case):
    data, names = frame(case["data"])
    if case["test"] == "fadl":
        out = oe.fourier_adl(data, "y", names, model=case["model"],
                             max_frequency=case["max_frequency"], max_lag=case["max_lag"],
                             lags=case["criterion"])
        assert out.attrs["statistic"] == pytest.approx(case["statistic"], rel=1e-10)
        assert out.attrs["frequency"] == case["frequency"] and out.attrs["lags"] == case["lags"]
        assert list(out.attrs["dx_lags"]) == as_list(case["dx_lags"])
        assert out.attrs["nobs"] == case["nobs"]
    else:
        out = oe.tsong_fourier(data, "y", names, model=case["model"],
                               max_frequency=case["max_frequency"],
                               leads_lags=case["leads_lags"], kernel=case["kernel"],
                               bandwidth=case["bandwidth"])
        assert out.attrs["statistic"] == pytest.approx(case["statistic"], rel=1e-10)
        assert out.attrs["fourier_F"] == pytest.approx(case["fourier_F"], rel=1e-9)
        assert out.attrs["frequency"] == case["frequency"] and out.attrs["nobs"] == case["nobs"]
        ols = oe.tsong_fourier(data, "y", names, model=case["model"],
                               max_frequency=case["max_frequency"], estimator="ols",
                               kernel=case["kernel"], bandwidth=case["bandwidth"])
        assert ols.attrs["statistic"] == pytest.approx(case["statistic_ols"], rel=1e-10)
        assert ols.attrs["frequency"] == case["frequency_ols"]


def test_fourier_adl_numpy_reference_fixed_lags():
    data, names = simulate(21, 120, 2)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    n, p, q = 120, 2, 1
    rows = np.arange(max(p, q) + 1, n)
    dz = np.vstack([np.zeros((1, 3)), np.diff(np.column_stack([y, x]), axis=0)])
    best = None
    for k in (1, 2, 3):
        cols = [np.ones(n)[rows], (np.arange(1, n + 1)[rows] - 60.5) / n,
                *fourier_terms(n, k)[rows].T, y[rows - 1], *x[rows - 1].T]
        cols += [dz[rows - j, 1 + i] for i in range(2) for j in range(q + 1)]
        cols += [dz[rows - j, 0] for j in range(1, p + 1)]
        beta, resid, se, _ = np_ols(np.column_stack(cols), dz[rows, 0])
        if best is None or resid @ resid < best[0]:
            best = (resid @ resid, beta[4] / se[4], k)
    out = oe.fourier_adl(data, "y", names, model="trend", lags=p, dx_lags=q)
    assert out.attrs["statistic"] == pytest.approx(best[1], rel=1e-10)
    assert out.attrs["frequency"] == best[2]
    assert out.attrs["critical_values"]["5%"] == min(fourier_coint.FADL["trend"][2][best[2] - 1][1],
                                                     fourier_coint.FADL["trend"][2][best[2] - 1][4])


def test_tsong_numpy_reference_dols_bartlett():
    data, names = simulate(23, 150, 1, cointegrated=True)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    n, q, bw = 150, 3, 5
    dx = np.vstack([np.zeros((1, 1)), np.diff(x, axis=0)])
    rows = np.arange(q + 1, n - q)
    best = None
    for k in (1, 2, 3):
        cols = [np.ones(n)[rows], *x[rows].T, *fourier_terms(n, k)[rows].T]
        cols += [dx[rows + j, 0] for j in range(-q, q + 1)]
        _, e, _, _ = np_ols(np.column_stack(cols), y[rows])
        if best is None or e @ e < best[0]:
            best = (e @ e, e, k)
    e = best[1]
    m = e.size
    omega = e @ e / m + 2 * sum((1 - j / (bw + 1)) * (e[j:] @ e[:-j]) / m for j in range(1, bw + 1))
    expected = np.sum(np.cumsum(e) ** 2) / (m * m * omega)
    out = oe.tsong_fourier(data, "y", names, leads_lags=q, bandwidth=bw)
    assert out.attrs["statistic"] == pytest.approx(expected, rel=1e-10)
    assert out.attrs["frequency"] == best[2]


def test_fourier_critical_value_domains_and_simulation_corroboration():
    assert fourier_coint.fadl_critical("constant", 1, 1, 99)[0] is None
    low, _ = fourier_coint.fadl_critical("constant", 1, 1, 100)
    assert low == {"1%": -4.73, "5%": -4.09, "10%": -3.76}
    large, _ = fourier_coint.fadl_critical("constant", 2, 3, 5000)
    assert large == {"1%": -4.94, "5%": -4.30, "10%": -3.96}
    assert fourier_coint.fadl_critical("trend", 1, 4, 200)[0] is None
    assert fourier_coint.tsong_critical("constant", 3, 4) is None
    assert fourier_coint.tsong_critical("constant", 3, 1)["10%"] is None
    simulation = json.loads((FIXTURES / "fourier_simulation.json").read_text())
    for cell in simulation["tsong"]:
        for simulated, published in zip(cell["simulated"], cell["table"], strict=True):
            if published is not None:
                assert abs(simulated - published) <= 0.13 * published
    for cell in simulation["fadl"]:
        for simulated, published in zip(cell["simulated"], cell["table"], strict=True):
            assert abs(simulated - published) <= 0.05 * abs(published)


# ---- Enders-Siklos -------------------------------------------------------------------------


def np_enders_siklos(y, x, model, threshold, p, trim=0.15):
    n = y.shape[0]
    _, mu, _, _ = np_ols(np.column_stack([np.ones(n), x]), y)
    dmu = np.concatenate([[0.0], np.diff(mu)])
    start = p + 1 if model == "tar" else max(p, 1) + 1
    rows = np.arange(start, n)
    signal = mu[rows - 1] if model == "tar" else dmu[rows - 1]
    hist = [dmu[rows - i] for i in range(1, p + 1)]

    def fit(tau):
        above = (signal >= tau).astype(float)
        design = np.column_stack([above * mu[rows - 1], (1 - above) * mu[rows - 1], *hist])
        return np_ols(design, dmu[rows])

    if threshold == "zero":
        tau = 0.0
    else:
        values = np.unique(signal)
        first = int(trim * values.size)
        grid = values[first - 1:values.size - first + 1]
        ssr = [fit(v)[1] @ fit(v)[1] for v in grid]
        tau = grid[int(np.argmin(ssr))]
    beta, _, se, cov = fit(tau)
    b, v = beta[:2], cov[:2, :2]
    phi = b @ np.linalg.solve(v, b) / 2
    sym = (b[0] - b[1]) ** 2 / (v[0, 0] + v[1, 1] - 2 * v[0, 1])
    return tau, phi, max(beta[0] / se[0], beta[1] / se[1]), sym, rows.size - 2 - p


@pytest.mark.parametrize("model", ["tar", "mtar"])
@pytest.mark.parametrize("threshold", ["zero", "consistent"])
def test_enders_siklos_numpy_reference(model, threshold):
    data, names = simulate(31, 140, 2, cointegrated=True)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    tau, phi, tmax, sym, df = np_enders_siklos(y, x, model, threshold, 2)
    out = oe.enders_siklos(data, "y", names, model=model, threshold=threshold, lags=2)
    assert out.attrs["tau"] == pytest.approx(tau, abs=1e-12)
    assert out.attrs["phi"] == pytest.approx(phi, rel=1e-10)
    assert out.attrs["t_max"] == pytest.approx(tmax, rel=1e-10)
    assert out.attrs["symmetry_F"] == pytest.approx(sym, rel=1e-10)
    assert out.attrs["symmetry_p_value"] == pytest.approx(sps.f.sf(sym, 1, df), rel=1e-9)
    assert out.attrs["critical_values"] is None and "no critical values" in \
        out.attrs["inference_status"]
    cv = out.attrs["cointegrating_vector"]
    ref = np.linalg.lstsq(np.column_stack([np.ones(140), x]), y, rcond=None)[0]
    assert cv["Intercept"] == pytest.approx(ref[0], rel=1e-9)
    json.dumps(out.attrs, allow_nan=False)


def test_enders_siklos_lag_selection_uses_common_sample():
    data, names = simulate(37, 120, 1, cointegrated=True)
    y, x = data["y"].to_numpy(), data[names].to_numpy()
    _, mu, _, _ = np_ols(np.column_stack([np.ones(120), x]), y)
    dmu = np.concatenate([[0.0], np.diff(mu)])
    rows = np.arange(4 + 1, 120)
    ic = []
    for p in range(5):
        above = (mu[rows - 1] >= 0).astype(float)
        design = np.column_stack([above * mu[rows - 1], (1 - above) * mu[rows - 1],
                                  *[dmu[rows - i] for i in range(1, p + 1)]])
        _, e, _, _ = np_ols(design, dmu[rows])
        ic.append(math.log(e @ e / rows.size) + math.log(rows.size) * (p + 2) / rows.size)
    out = oe.enders_siklos(data, "y", names, lags="bic", max_lag=4)
    assert out.attrs["lags"] == int(np.argmin(ic))


# ---- failure contract ---------------------------------------------------------------------


def test_failure_contract():
    data, names = frame("level_break")
    gaps = data.assign(t=range(len(data))).drop(index=5)
    missing = data.copy()
    missing.loc[3, "y"] = np.nan
    cases = [
        (lambda: oe.gregory_hansen(missing, "y", names), "missing_values"),
        (lambda: oe.hatemi_j(gaps, "y", names, time="t"), "time_gaps"),
        (lambda: oe.gregory_hansen(data, "y", names, model="bogus"), "invalid_option"),
        (lambda: oe.gregory_hansen(data, "y", names, trim=0.6), "invalid_option"),
        (lambda: oe.hatemi_j(data, "y", names, trim=0.34), "invalid_option"),
        (lambda: oe.gregory_hansen(data, "y", names, lags=-1), "invalid_lags"),
        (lambda: oe.gregory_hansen(data, "y", names, lags=2, maxlag=3), "invalid_option"),
        (lambda: oe.gregory_hansen(data, "y", "x1"), "invalid_spec"),
        (lambda: oe.gregory_hansen(data.assign(x1=1.0), "y", names), "constant_series"),
        (lambda: oe.gregory_hansen(data.head(12), "y", names), "insufficient_observations"),
        (lambda: oe.gregory_hansen(data.head(25), "y", names, lags="aic", maxlag=12),
         "invalid_lags"),
        (lambda: oe.maki_coint(data, "y", names, breaks=6), "invalid_option"),
        (lambda: oe.maki_coint(data.head(30), "y", names, breaks=5),
         "insufficient_observations"),
        (lambda: oe.bayer_hanck(data, "y", names, trend="quadratic"), "invalid_option"),
        (lambda: oe.bayer_hanck(data, "y", names, lags=-2), "invalid_lags"),
        (lambda: oe.tsong_fourier(data, "y", names, max_frequency=6), "invalid_option"),
        (lambda: oe.fourier_adl(data, "y", names, lags="aic", dx_lags=2), "invalid_option"),
        (lambda: oe.enders_siklos(data, "y", names, threshold="chan"), "invalid_option"),
        (lambda: oe.ecm_cointegration(data, "y", names, test="johansen"), "invalid_option"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code)
    wide, wide_names = simulate(1, 200, 12)
    with pytest.raises(AnalysisError) as error:
        oe.bayer_hanck(wide, "y", wide_names)
    assert error.value.code == "critical_values_unavailable"
    collinear = data.assign(x2=2.0 * data["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.gregory_hansen(collinear, "y", names)
    assert error.value.code == "collinear_regressors"
    with pytest.raises(AnalysisError) as error:
        oe.bayer_hanck(collinear, "y", names)
    assert error.value.code == "collinear_regressors"


# ---- seeded size and power (fast) -----------------------------------------------------------


def test_fast_seeded_size_and_power():
    """Rejection rates at the published 5% points; bands allow the Monte Carlo error.

    The documented larger simulation is in docs/econometrics/cointegration-breaks.md.
    """
    reps = 60
    rejections = {"gh_null": 0, "gh_power": 0, "bh_null": 0, "bh_power": 0}
    for rep in range(reps):
        null, names = simulate(1000 + rep, 100, 1)
        shifted, _ = simulate(5000 + rep, 100, 1, cointegrated=True, shift=4.0)
        stable, _ = simulate(9000 + rep, 100, 1, cointegrated=True)
        for key, data in (("gh_null", null), ("gh_power", shifted)):
            out = oe.gregory_hansen(data, "y", names, model="level", lags=0)
            rejections[key] += out.attrs["statistics"]["Zt*"] < out.attrs["critical_values"]["Zt*"]["5%"]
        for key, data in (("bh_null", null), ("bh_power", stable)):
            out = oe.bayer_hanck(data, "y", names, lags=1)
            rejections[key] += out.attrs["statistics"]["EG-J-Ba-Bo"] > \
                out.attrs["critical_values"]["EG-J-Ba-Bo"]["5%"]
    rates = {key: value / reps for key, value in rejections.items()}
    assert rates["gh_null"] <= 0.20 and rates["bh_null"] <= 0.20
    assert rates["gh_power"] >= 0.8 and rates["bh_power"] >= 0.8


def test_documented_size_power_fixture():
    """The committed seeded simulation reported in the documentation (500 replications)."""
    record = json.loads((FIXTURES / "size_power.json").read_text())
    assert record["reps"] == 500 and record["seed"] == 20261014 and record["n"] == 100
    results = record["results"]
    assert 0.02 <= results["bayer_hanck"]["size"]["EG-J-Ba-Bo"] <= 0.10
    assert results["gregory_hansen_level"]["size"]["Zt*"] <= 0.10
    for name in ("gregory_hansen_level", "gregory_hansen_regime", "hatemi_j", "bayer_hanck"):
        assert min(results[name]["power"].values()) >= 0.8
    assert results["tsong_fourier"]["size"] <= 0.05
    # Documented size distortions of the published procedures at n = 100.
    assert results["fourier_adl"]["size"]["t"] > 0.10
    assert results["hatemi_j"]["size"]["ADF*"] > 0.10


def test_threshold_ssr_profile_equals_brute_force():
    import torch

    from openecon.econometrics.unitroot.threshold_coint import threshold_ssr

    rng = np.random.default_rng(41)
    n = 90
    lagged = rng.normal(size=n)
    signal = np.round(rng.normal(size=n), 1)                    # ties in the threshold variable
    history = rng.normal(size=(n, 2))
    target = 0.3 * lagged + history @ np.array([0.2, -0.1]) + rng.normal(size=n)
    grid = np.unique(signal)
    profile = threshold_ssr(*(torch.tensor(v) for v in (target, lagged, signal, history, grid)))
    for value, got in zip(grid, profile.numpy(), strict=True):
        above = (signal >= value).astype(float)
        if above.all():
            assert got == np.inf
            continue
        design = np.column_stack([above * lagged, (1 - above) * lagged, history])
        _, resid, _, _ = np_ols(design, target)
        assert got == pytest.approx(resid @ resid, rel=1e-10)
