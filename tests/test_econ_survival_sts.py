"""Independent oracles for oe.sts (Kaplan-Meier, Nelson-Aalen, log-rank family).

statsmodels' SurvfuncRight and survdiff serve as oracles where they implement
the same quantity; everything else (Peto-Peto-Prentice and general
Fleming-Harrington weights, the trend test, confidence intervals, percentile
limits, restricted mean) is recomputed by explicit loops over the distinct
failure times in NumPy.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def make_data(seed=1, n=300, entry=False):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    t = np.round(rng.exponential(5 / (1 + 0.4 * g)), 1) + 0.1
    c = np.round(rng.exponential(8, n), 1) + 0.1
    d = (t <= c).astype(float)
    t = np.minimum(t, c)
    frame = pd.DataFrame({"t": t, "d": d, "g": g, "s": rng.integers(0, 2, n),
                          "w": rng.integers(1, 4, n)})
    if entry:
        frame["t0"] = np.where(rng.random(n) < 0.4, t * rng.uniform(0.05, 0.9, n), 0.0)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def km_oracle(t, d, w=None, t0=None):
    """Explicit loop: n at risk, d, S, Greenwood sum, log sum, NA and its variance."""
    w = np.ones_like(t) if w is None else w
    t0 = np.zeros_like(t) if t0 is None else t0
    rows, s, gsum, lsum, h, hv = [], 1.0, 0.0, 0.0, 0.0, 0.0
    for time in np.unique(t):
        at_risk = w[(t0 < time) & (t >= time)].sum()
        dead = w[(t == time) & (d == 1)].sum()
        cens = w[(t == time) & (d == 0)].sum()
        if dead:
            s *= 1 - dead / at_risk
            if at_risk > dead:
                gsum += dead / (at_risk * (at_risk - dead))
                lsum += np.log((at_risk - dead) / at_risk)
            h += dead / at_risk
            hv += dead / at_risk ** 2
        rows.append((time, at_risk, dead, cens, s, gsum, lsum, h, hv))
    return np.array(rows)


def weighted_logrank(t, d, g, weight, strata=None, t0=None, scores=None):
    """sts test formulas by an explicit loop over strata and failure times."""
    strata = np.zeros_like(t) if strata is None else strata
    t0 = np.zeros_like(t) if t0 is None else t0
    levels = np.unique(g)
    u, v = np.zeros(len(levels)), np.zeros((len(levels), len(levels)))
    for s in np.unique(strata):
        m = strata == s
        km, km_plus = 1.0, 1.0
        for time in np.unique(t[m & (d == 1)]):
            risk = m & (t0 < time) & (t >= time)
            fail = m & (t == time) & (d == 1)
            n, dd = risk.sum(), fail.sum()
            ng = np.array([(risk & (g == lv)).sum() for lv in levels])
            dg = np.array([(fail & (g == lv)).sum() for lv in levels])
            km_plus *= 1 - dd / (n + 1)
            weights = {"logrank": 1.0, "wilcoxon": n, "tware": np.sqrt(n), "peto": km_plus,
                       "fh": km ** weight[1] * (1 - km) ** weight[2] if weight[0] == "fh" else 0}
            wj = weights[weight[0]]
            u += wj * (dg - ng * dd / n)
            if n > 1:
                v += wj ** 2 * dd * (n - dd) / (n * (n - 1)) * (
                    np.diag(ng) - np.outer(ng, ng) / n)
            km *= 1 - dd / n
    chi2 = u[:-1] @ np.linalg.solve(v[:-1, :-1], u[:-1])
    trend = None if scores is None else (scores @ u) ** 2 / (scores @ v @ scores)
    return chi2, trend


def test_kaplan_meier_matches_statsmodels_and_explicit_loop(data):
    out = oe.sts(data, "t", failure="d", by="g")
    table = out["survival"]
    for level in range(3):
        sub = data[data.g == level]
        rows = table[table.group == level]
        oracle = km_oracle(sub.t.to_numpy(), sub.d.to_numpy())
        assert_allclose(rows.time, oracle[:, 0])
        assert_allclose(rows.n_risk, oracle[:, 1])
        assert_allclose(rows.n_event, oracle[:, 2])
        assert_allclose(rows.n_censored, oracle[:, 3])
        assert_allclose(rows.survivor, oracle[:, 4], rtol=1e-12)
        assert_allclose(rows.cumulative_hazard, oracle[:, 7], rtol=1e-12)
        assert_allclose(rows.cumulative_hazard_std_error, np.sqrt(oracle[:, 8]), rtol=1e-12)
        sf = SurvfuncRight(sub.t, sub.d)
        failed = rows[rows.n_event > 0]
        assert_allclose(failed.survivor, sf.surv_prob, rtol=1e-12)
        assert_allclose(failed.std_error, sf.surv_prob_se, rtol=1e-10)
        assert_allclose(failed.n_risk, sf.n_risk)


def test_delayed_entry_and_frequency_weights():
    frame = make_data(seed=4, entry=True)
    out = oe.sts(frame, "t", failure="d", entry="t0", weights="w")
    oracle = km_oracle(frame.t.to_numpy(), frame.d.to_numpy(), frame.w.to_numpy(float),
                       frame.t0.to_numpy())
    table = out["survival"]
    assert_allclose(table.n_risk, oracle[:, 1])
    assert_allclose(table.survivor, oracle[:, 4], rtol=1e-12)
    alive = oracle[:, 4] > 0
    assert_allclose(table.std_error[alive], (oracle[:, 4] * np.sqrt(oracle[:, 5]))[alive],
                    rtol=1e-10)
    # fweights equal the duplicated data
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    plain = oe.sts(expanded, "t", failure="d", entry="t0")["survival"]
    assert_allclose(table.survivor, plain.survivor, rtol=1e-12)
    assert_allclose(table.ci_low, plain.ci_low, rtol=1e-12)
    sf = SurvfuncRight(frame.t, frame.d, entry=frame.t0, freq_weights=frame.w)
    failed = table[table.n_event > 0]
    assert_allclose(failed.survivor, sf.surv_prob, rtol=1e-12)


@pytest.mark.parametrize("conftype", ["loglog", "log", "plain"])
def test_confidence_intervals_follow_stata_and_spss_formulas(data, conftype):
    sub = data[data.g == 1]
    table = oe.sts(sub, "t", failure="d", conftype=conftype, alpha=0.1)["survival"]
    oracle = km_oracle(sub.t.to_numpy(), sub.d.to_numpy())
    s, gsum, lsum = oracle[:, 4], oracle[:, 5], oracle[:, 6]
    z = stats.norm.isf(0.05)
    inside = (s > 0) & (s < 1)
    if conftype == "loglog":
        sigma = np.sqrt(gsum[inside]) / np.abs(lsum[inside])
        low, high = s[inside] ** np.exp(z * sigma), s[inside] ** np.exp(-z * sigma)
    elif conftype == "log":
        low = s[inside] * np.exp(-z * np.sqrt(gsum[inside]))
        high = np.minimum(1, s[inside] * np.exp(z * np.sqrt(gsum[inside])))
    else:
        se = s[inside] * np.sqrt(gsum[inside])
        low, high = np.clip(s[inside] - z * se, 0, 1), np.clip(s[inside] + z * se, 0, 1)
    assert_allclose(table.ci_low[inside], low, rtol=1e-10)
    assert_allclose(table.ci_high[inside], high, rtol=1e-10)
    h, hv = oracle[:, 7], oracle[:, 8]
    pos = h > 0
    assert_allclose(table.cumulative_hazard_ci_low[pos], h[pos] * np.exp(-z * np.sqrt(hv[pos])
                                                                          / h[pos]), rtol=1e-10)


def test_percentiles_median_limits_and_restricted_mean(data):
    out = oe.sts(data, "t", failure="d", by="g")
    summary = out["summary"].set_index("group")
    z = stats.norm.isf(0.025)
    for level in range(3):
        sub = data[data.g == level]
        oracle = km_oracle(sub.t.to_numpy(), sub.d.to_numpy())
        failed = oracle[oracle[:, 2] > 0]
        times, s = failed[:, 0], failed[:, 4]
        for name, p in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
            hit = times[s <= 1 - p]
            expected = hit[0] if hit.size else None
            assert summary.loc[level, name] == pytest.approx(expected)
        sigma = np.sqrt(failed[:, 5]) / np.abs(failed[:, 6])
        low, high = s ** np.exp(z * sigma), s ** np.exp(-z * sigma)
        assert summary.loc[level, "median_ci_low"] == pytest.approx(times[low <= 0.5][0])
        hits = times[high <= 0.5]
        if hits.size:
            assert summary.loc[level, "median_ci_high"] == pytest.approx(hits[0])
        # restricted mean: area under the step function up to the largest time
        largest = sub.t.max()
        edges = np.append(times, largest)
        area_pieces = s * np.diff(edges)
        mean = times[0] + area_pieces.sum()
        tail = np.cumsum(area_pieces[::-1])[::-1]
        n_at, dd = failed[:, 1], failed[:, 2]
        ok = n_at > dd
        se = np.sqrt(np.sum(tail[ok] ** 2 * dd[ok] / (n_at[ok] * (n_at[ok] - dd[ok]))))
        assert summary.loc[level, "restricted_mean"] == pytest.approx(mean, rel=1e-12)
        assert summary.loc[level, "restricted_mean_std_error"] == pytest.approx(se, rel=1e-10)
        assert summary.loc[level, "events"] == sub.d.sum()
        assert summary.loc[level, "time_at_risk"] == pytest.approx(sub.t.sum())
    sf = SurvfuncRight(data.t[data.g == 0], data.d[data.g == 0])
    assert summary.loc[0, "median"] == pytest.approx(sf.quantile(0.5))


def test_rank_tests_match_survdiff_and_explicit_formulas(data):
    out = oe.sts(data, "t", failure="d", by="g")
    tests = out["tests"]
    for name, kwargs in (("logrank", {}), ("wilcoxon", {"weight_type": "gb"}),
                         ("tware", {"weight_type": "tw"})):
        chi2, p = survdiff(data.t, data.d, data.g, **kwargs)
        assert tests.loc[name, "statistic"] == pytest.approx(chi2, rel=1e-10)
        assert tests.loc[name, "p_value"] == pytest.approx(p, rel=1e-8)
        assert tests.loc[name, "df"] == 2
    t, d, g = data.t.to_numpy(), data.d.to_numpy(), data.g.to_numpy()
    for name, weight in (("peto", ("peto",)), ("wilcoxon", ("wilcoxon",))):
        chi2, _ = weighted_logrank(t, d, g, weight)
        assert tests.loc[name, "statistic"] == pytest.approx(chi2, rel=1e-10)
    fh = oe.sts(data, "t", failure="d", by="g", test="fh", fh_p=1, fh_q=0.5)
    chi2, _ = weighted_logrank(t, d, g, ("fh", 1, 0.5))
    assert fh.attrs["statistic"] == pytest.approx(chi2, rel=1e-10)
    chi2_sm, _ = survdiff(data.t, data.d, data.g, weight_type="fh", fh_p=1)
    fh1 = oe.sts(data, "t", failure="d", by="g", test="fh", fh_p=1)
    assert fh1.attrs["statistic"] == pytest.approx(chi2_sm, rel=1e-10)
    expected = out["expected"]
    assert_allclose(expected.observed, [d[g == k].sum() for k in range(3)])
    assert expected.expected.sum() == pytest.approx(d.sum())


def test_stratified_and_trend_tests(data):
    out = oe.sts(data, "t", failure="d", by="g", strata="s", test="logrank", trend=True)
    chi2, _ = survdiff(data.t, data.d, data.g, strata=data.s)
    assert out.attrs["statistic"] == pytest.approx(chi2, rel=1e-10)
    t, d, g, s = (data[c].to_numpy() for c in ("t", "d", "g", "s"))
    _, trend = weighted_logrank(t, d, g, ("logrank",), strata=s, scores=np.arange(3.0))
    assert out["tests"].loc["logrank_trend", "statistic"] == pytest.approx(trend, rel=1e-10)
    entry = make_data(seed=8, entry=True)
    tw = oe.sts(entry, "t", failure="d", entry="t0", by="g", test="tware")
    chi2, _ = weighted_logrank(*(entry[c].to_numpy() for c in ("t", "d", "g")), ("tware",),
                               t0=entry.t0.to_numpy())
    assert tw.attrs["statistic"] == pytest.approx(chi2, rel=1e-10)


def test_thinning_rendering_and_failure_contract(data):
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"t": rng.exponential(size=5000), "d": 1.0})
    out = oe.sts(big, "t", failure="d")
    assert len(out["survival"]) == 2000
    assert any("2000 evenly spaced" in note for note in out.attrs["notes"])
    assert out["survival"].time.iloc[-1] == pytest.approx(big.t.max())
    assert "Survivor function" in str(out)
    assert "tabular" in out.to_latex()
    with pytest.raises(AnalysisError) as error:
        oe.sts(data.assign(d=data.d * 2), "t", failure="d")
    assert error.value.code == "invalid_failure_indicator"
    with pytest.raises(AnalysisError) as error:
        oe.sts(data.assign(t=-data.t), "t", failure="d")
    assert error.value.code == "invalid_survival_time"
    with pytest.raises(AnalysisError) as error:
        oe.sts(data, "t", failure="d", test="logrank")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.sts(data, "t", failure="d", conftype="arcsine")
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.sts(data.assign(g=data.g.astype(str)), "t", failure="d", by="g", trend=True)
    assert error.value.code == "invalid_trend"
    missing = data.astype({"t": float}).copy()
    missing.loc[3, "t"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.sts(missing, "t", failure="d", missing="raise")
    assert error.value.code == "missing_values"
    dropped = oe.sts(missing, "t", failure="d")
    assert dropped.attrs["n_dropped"] == 1
    ended = data.assign(t0=np.where(np.arange(len(data)) < 5, data.t + 1, 0.0))
    noted = oe.sts(ended, "t", failure="d", entry="t0")
    assert noted.attrs["n"] == len(data) - 5
    assert any("end on or before" in note for note in noted.attrs["notes"])
