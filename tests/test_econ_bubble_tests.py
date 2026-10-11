"""Independent oracles for oe.psy (Phillips-Shi-Yu SADF/GSADF/BSADF).

Every window statistic is refitted separately with NumPy least squares (a
transcription of psymonitor's ADF with a fixed lag); the Monte Carlo and wild
bootstrap calibrations are replayed draw by draw; the finite-sample critical
values are compared with Phillips, Shi and Yu (CFDP 1843, Table 1(b)).
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.unitroot import bubbles


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def code(function, *args, **kwargs):
    with pytest.raises(AnalysisError) as error:
        function(*args, **kwargs)
    return error.value.code


def adf(y, k):
    """psymonitor::ADF(y, IC = 0, adflag = k): t ratio of y_{t-1} with intercept and k lags."""
    t1 = len(y) - 1
    dy = np.diff(y)
    x = np.column_stack([y[k:t1], np.ones(t1 - k)] + [dy[k - j:t1 - j] for j in range(1, k + 1)])
    beta = np.linalg.lstsq(x, dy[k:], rcond=None)[0]
    resid = dy[k:] - x @ beta
    s2 = resid @ resid / (t1 - k - k - 2)
    return beta[0] / math.sqrt(s2 * np.linalg.inv(x.T @ x)[0, 0])


def psy_by_windows(y, window, k):
    total = len(y)
    bsadf, starts = [], []
    for end in range(window, total + 1):
        values = [adf(y[s:end], k) for s in range(0, end - window + 1)]
        bsadf.append(max(values))
        starts.append(int(np.argmax(values)))
    sadf = max(adf(y[:end], k) for end in range(window, total + 1))
    return np.array(bsadf), np.array(starts), sadf, adf(y, k)


def bubble_series(total=70, start=30, stop=45, rho=1.06, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(total)
    y = np.zeros(total)
    y[0] = 50.0
    for i in range(1, total):
        y[i] = (rho if start <= i < stop else 1.0) * y[i - 1] + e[i]
    if stop < total:
        y[stop:] -= y[stop] - y[start - 1]
    return y


@pytest.mark.parametrize("total,lags", [(60, 0), (64, 1), (70, 2)])
def test_every_window_statistic_matches_separate_least_squares(total, lags):
    y = bubble_series(total)
    window = bubbles.default_window(total)
    assert window == math.floor(total * (0.01 + 1.8 / math.sqrt(total)))
    stats = bubbles.recursive_statistics(torch.tensor(y)[None, :], window, lags)
    bsadf, starts, sadf, full = psy_by_windows(y, window, lags)
    np.testing.assert_allclose(stats["bsadf"][0].numpy(), bsadf, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(stats["start"][0].numpy(), starts)
    assert float(stats["sadf"][0]) == pytest.approx(sadf, rel=1e-9)
    assert float(stats["gsadf"][0]) == pytest.approx(bsadf.max(), rel=1e-9)
    assert float(stats["adf"][0]) == pytest.approx(full, rel=1e-9)
    assert int(stats["undefined"][0]) == 0


def test_public_tables_sequence_and_episodes():
    y = bubble_series(70)
    frame = pd.DataFrame({"price": y, "month": np.arange(500, 570)})
    result = oe.psy(frame, "price", time="month", replications=99, seed=4)
    window = bubbles.default_window(70)
    bsadf, starts, sadf, full = psy_by_windows(y, window, 0)
    tests = result["tests"].set_index("test")
    assert tests.loc["GSADF", "statistic"] == pytest.approx(bsadf.max(), rel=1e-9)
    assert tests.loc["SADF", "statistic"] == pytest.approx(sadf, rel=1e-9)
    sequence = result["sequence"]
    assert len(sequence) == 70 - window + 1
    assert sequence.period.tolist() == list(range(500 + window - 1, 570))
    np.testing.assert_allclose(sequence.bsadf, bsadf, rtol=1e-9)
    assert sequence.window_start.tolist() == [500 + s for s in starts]
    assert (sequence.exceeds == (sequence.bsadf > sequence.critical_95pct)).all()
    episodes = result["episodes"]
    assert len(episodes) >= 1
    first = episodes.iloc[0]
    assert 525 <= first.start_period <= 545
    assert first.duration >= math.ceil(math.log(70))
    json.dumps(result.attrs, allow_nan=False)
    assert "\\caption{episodes}" in result.to_latex()
    str(result)


def test_monte_carlo_calibration_replays_the_null_draws():
    y = bubble_series(48, start=20, stop=30)
    frame = pd.DataFrame({"y": y})
    state = torch.get_rng_state().clone()
    result = oe.psy(frame, "y", replications=39, seed=11, lags=1)
    assert torch.equal(state, torch.get_rng_state())
    again = oe.psy(frame, "y", replications=39, seed=11, lags=1)
    assert result.attrs == again.attrs
    pd.testing.assert_frame_equal(result["sequence"], again["sequence"])
    window = bubbles.default_window(48)
    generator = torch.Generator().manual_seed(11)
    noise = torch.randn((39, 48), dtype=torch.float64, generator=generator)
    paths = (noise + 1 / 48).cumsum(dim=1).numpy()
    null = np.array([psy_by_windows(path, window, 1) for path in paths], dtype=object)
    gsadf = np.array([row[0].max() for row in null])
    sadf = np.array([row[2] for row in null])
    bsadf = np.stack([row[0] for row in null])
    tests = result["tests"].set_index("test")
    observed = tests.loc["GSADF", "statistic"]
    assert tests.loc["GSADF", "p_value"] == pytest.approx(((gsadf >= observed).sum() + 1) / 40)
    np.testing.assert_allclose(tests.loc["GSADF", ["critical_90pct", "critical_95pct", "critical_99pct"]]
                               .to_numpy(dtype=float), np.quantile(gsadf, [.9, .95, .99]), rtol=1e-8)
    np.testing.assert_allclose(tests.loc["SADF", ["critical_90pct", "critical_95pct", "critical_99pct"]]
                               .to_numpy(dtype=float), np.quantile(sadf, [.9, .95, .99]), rtol=1e-8)
    np.testing.assert_allclose(result["sequence"].critical_95pct,
                               np.quantile(bsadf, 0.95, axis=0), rtol=1e-8)
    assert result.attrs["monte_carlo_denominator"] == 40
    assert result.attrs["failed_replications"] == 0


def test_monte_carlo_critical_values_agree_with_phillips_shi_yu_table_1():
    """CFDP 1843 Table 1(b): T = 100, r0 = 0.4, lag 0, 5000 draws.

    Two independent seeds of the same 5000-draw design moved the 95% GSADF quantile
    by up to 0.07 at T = 200, so a 0.12 band is about three Monte Carlo standard errors.
    """
    generator = torch.Generator().manual_seed(20261010)
    paths = bubbles.null_paths(100, 4000, generator)
    stats = bubbles.recursive_statistics(paths, 40, 0)
    levels = torch.tensor([0.90, 0.95, 0.99], dtype=torch.float64)
    sadf = torch.quantile(stats["sadf"], levels).tolist()
    gsadf = torch.quantile(stats["gsadf"], levels).tolist()
    assert sadf[:2] == pytest.approx([0.72, 1.05], abs=0.12)
    assert gsadf[:2] == pytest.approx([1.16, 1.48], abs=0.12)
    assert [sadf[2], gsadf[2]] == pytest.approx([1.66, 2.08], abs=0.25)


def test_wild_bootstrap_replays_the_phillips_shi_recursion():
    y = bubble_series(44, start=24, stop=32, seed=8)
    frame = pd.DataFrame({"y": y})
    lags, draws, horizon = 2, 19, 12
    result = oe.psy(frame, "y", lags=lags, calibration="wild_bootstrap", replications=draws,
                    seed=6, horizon=horizon)
    level = (y - y[0]) / np.abs(y - y[0]).max()
    dy = np.diff(level)
    rows = len(dy) - lags
    x = np.column_stack([np.ones(rows)] + [dy[lags - j:len(dy) - j] for j in range(1, lags + 1)])
    coefficient = np.linalg.lstsq(x, dy[lags:], rcond=None)[0]
    resid = dy[lags:] - x @ coefficient
    generator = torch.Generator().manual_seed(6)
    picks = torch.randint(0, rows, (draws, len(dy) - lags), generator=generator).numpy()
    weights = torch.randn((draws, len(dy) - lags), dtype=torch.float64, generator=generator).numpy()
    window = bubbles.default_window(44)
    maxima, gsadf = [], []
    for b in range(draws):
        increments = list(dy[:lags])
        for i in range(len(dy) - lags):
            ar = sum(coefficient[j] * increments[-j] for j in range(1, lags + 1))
            increments.append(ar + weights[b, i] * resid[picks[b, i]])
        path = np.r_[level[0], level[0] + np.cumsum(increments)]
        bsadf = psy_by_windows(path, window, lags)[0]
        maxima.append(bsadf[:horizon].max())
        gsadf.append(bsadf.max())
    family = result.attrs["family_wise_critical_values"]
    assert list(family.values()) == pytest.approx(np.quantile(maxima, [.9, .95, .99]), rel=1e-8)
    tests = result["tests"].set_index("test")
    assert tests.loc["GSADF", "critical_95pct"] == pytest.approx(np.quantile(gsadf, .95), rel=1e-8)
    assert result.attrs["ar_coefficients"] == pytest.approx(coefficient[1:].tolist(), rel=1e-9)
    stamped = result["sequence"].exceeds.to_numpy(dtype=bool)
    assert (stamped == (result["sequence"].bsadf.to_numpy() > family["95%"])).all()


def test_episode_rules_first_crossing_minimum_duration_and_ongoing():
    bsadf = torch.tensor([0., 2., 0.5, 0.5, 0.5, 2., 2., 0.5, 0., 0., 3., 3., 3.],
                         dtype=torch.float64)
    cv = torch.ones_like(bsadf)
    # Origination at 1; a dip at 2 lies inside the 2.5-observation minimum duration,
    # so termination is the first point >= 1 + 3 below the critical value (4).
    assert bubbles.episodes(bsadf, cv, 2.5) == [(1, 4), (5, 8), (10, None)]
    assert bubbles.episodes(bsadf, cv, 0.0) == [(1, 2), (5, 7), (10, None)]
    assert bubbles.episodes(torch.zeros(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64), 1.0) == []


def test_psy_failure_contract():
    y = bubble_series(60)
    frame = pd.DataFrame({"y": y})
    assert code(oe.psy, frame, "y", calibration="sieve") == "invalid_option"
    assert code(oe.psy, frame, "y", level=0.975) == "invalid_option"
    assert code(oe.psy, frame, "y", min_window=4) == "invalid_option"
    assert code(oe.psy, frame, "y", lags=3, min_window=9) == "invalid_option"
    assert code(oe.psy, frame, "y", horizon=1000) == "invalid_option"
    assert code(oe.psy, frame, "y", min_duration=-1.0) == "invalid_option"
    assert code(oe.psy, frame, "y", replications=10) == "invalid_option"
    assert code(oe.psy, frame, "y", max_work=1000) == "work_limit"
    assert code(oe.psy, frame.assign(y=4.0), "y") == "constant_series"
    holes = frame.copy()
    holes.loc[3, "y"] = np.nan
    assert code(oe.psy, holes, "y") == "missing_values"
    statistics_only = oe.psy(frame, "y", calibration="none")
    assert statistics_only.attrs["p_value"] is None
    assert statistics_only["episodes"].empty
    assert statistics_only["sequence"].critical_95pct.isna().all()
