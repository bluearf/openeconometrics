"""Independent NumPy/SciPy formulas, published moments and common-block nulls."""
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.special import ndtr

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def cube(seed=8, n=7, t=42):
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(n, t, 3)) + rng.normal(size=(1, t, 3)) * .25
    return innovations.cumsum(axis=1)


def frame(v):
    n, t, _ = v.shape
    return pd.DataFrame({'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
                         'y': v[:, :, 0].ravel(), 'x': v[:, :, 1].ravel(), 'z': v[:, :, 2].ravel()})


def fit(x, y):
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    return b, y - x @ b


def lrv(e, lag):
    return (e @ e + 2 * sum((1 - j / (lag + 1)) * (e[j:] @ e[:-j]) for j in range(1, lag + 1))) / len(e)


def independent(v, method, deterministic='constant', ar='panel', lag=2, bandwidth=3):
    n, t, _ = v.shape
    e = []
    for unit in v:
        x = unit[:, 1:]
        if deterministic != 'none':
            x = np.column_stack((np.ones(t), x))
        if deterministic == 'trend':
            x = np.column_stack((np.arange(t), x))
        e.append(fit(x, unit[:, 0])[1])
    e = np.asarray(e)
    if method == 'westerlund':
        s, r = np.sum(np.cumsum(e, axis=1)**2, axis=1), np.sum(e**2, axis=1)
        return {'variance_ratio': np.sqrt(n) * (np.mean(s / r) if ar == 'panel' else s.sum() / r.sum()) / t**2}
    numerators, denominators, variance, weight, a, h, s2 = [], [], [], [], [], [], []
    for row, unit in zip(v, e):
        de = np.diff(unit)
        err = fit(unit[:-1, None], de)[1]
        omega = lrv(err, bandwidth)
        numerators.append(unit[:-1] @ de - (t - 1) * (omega - np.mean(err**2)) / 2)
        denominators.append(unit[:-1] @ unit[:-1])
        variance.append(omega)
        dx = np.diff(row[:, 1:], axis=0)
        if deterministic == 'trend':
            dx = np.column_stack((np.ones(t - 1), dx))
        conditional = fit(dx, np.diff(row[:, 0]))[1]
        weight.append(1 / lrv(conditional, bandwidth))
        # Explicit residual-maker, distinct from the QR replay path.
        z = np.column_stack([de[lag - j:len(de) - j] for j in range(1, lag + 1)])
        proj = np.eye(t - 1 - lag) - z @ np.linalg.pinv(z)
        delta, level = proj @ de[lag:], proj @ unit[lag:-1]
        errors = fit(level[:, None], delta)[1]
        a.append(level @ delta)
        h.append(level @ level)
        s2.append(np.mean(errors**2))
    b, r, o, w, an, den, s = map(np.asarray, (numerators, denominators, variance, weight, a, h, s2))
    if ar == 'panel':
        return {'modified_pp': t / np.sqrt(n) * np.sum(b / r),
                'pp': np.sum(b / np.sqrt(o * r)) / np.sqrt(n),
                'adf': np.sum(an / np.sqrt(s * den)) / np.sqrt(n)}
    return {'modified_variance_ratio': t**2 * n**1.5 / (w @ r),
            'modified_pp': t * np.sqrt(n) * (w @ b) / (w @ r),
            'pp': (w @ b) / np.sqrt(np.mean(w * o) * (w @ r)),
            'adf': (w @ an) / np.sqrt(s.mean() * (w @ den))}


@pytest.mark.parametrize('method', ['pedroni', 'westerlund'])
@pytest.mark.parametrize('deterministic', ['none', 'constant', 'trend'])
@pytest.mark.parametrize('ar', ['panel', 'same'])
def test_raw_statistics_calibration_tails_and_complete_sample(method, deterministic, ar):
    v = cube()
    r = oe.xtcointtest(frame(v), 'y', ['x', 'z'], 'id', 'time', test=method, deterministic=deterministic,
                       ar=ar, lags=2, kernel_lags=3)
    expected = independent(v, method, deterministic, ar)
    assert_allclose(list(r.attrs['raw_statistics'].values()), list(expected.values()), atol=1e-10)
    for name, raw in expected.items():
        mean, variance = r.attrs['moments'][name]
        standardized = (raw - math.sqrt(7) * mean) / math.sqrt(variance)
        assert_allclose(r.loc[name, 'statistic'], standardized, atol=1e-10)
        assert_allclose(r.loc[name, 'p_value'], ndtr(-standardized if name == 'modified_variance_ratio' else standardized), atol=1e-14)
    assert r.attrs['nobs'] == 7 * 42
    # Independently transcribed spot checks cover distinct deterministic/family moments.
    if method == 'pedroni' and deterministic == 'trend' and ar == 'same':
        assert_allclose(r.attrs['moments']['modified_pp'], [-14.011, 64.219])
    if method == 'westerlund' and deterministic == 'constant' and ar == 'panel':
        assert_allclose(r.attrs['moments']['variance_ratio'], [.019406, .000273])


@pytest.mark.parametrize('method', ['pedroni', 'westerlund'])
def test_common_block_bootstrap_against_independent_integrated_null(method):
    v = cube(n=4, t=30)
    before = torch.random.get_rng_state().clone()
    r = oe.xtcointtest(frame(v), 'y', ['x', 'z'], 'id', 'time', test=method, lags=2,
                       kernel_lags=3, bootstrap=19, block_length=3, seed=198)
    assert torch.equal(before, torch.random.get_rng_state())
    gen = torch.Generator().manual_seed(198)
    delta = np.diff(v, axis=1)
    delta -= delta.mean(axis=1, keepdims=True)
    draws = []
    for _ in range(19):
        starts = torch.randint(29, (10,), generator=gen).numpy()
        indices = ((starts[:, None] + np.arange(3)).ravel()[:29] % 29)
        null = np.concatenate((np.zeros((4, 1, 3)), delta[:, indices].cumsum(axis=1)), axis=1)
        draws.append(list(independent(null, method).values()))
    assert_allclose(r.attrs['bootstrap']['raw_statistics'], draws, atol=1e-10)
    draws = np.array(draws)
    observed = list(independent(v, method).values())
    for j, name in enumerate(r.index):
        assert r.loc[name, 'bootstrap_p_value'] == (1 + sum(draws[:, j] <= observed[j])) / 20
        assert_allclose(r.loc[name, 'bootstrap_raw_critical_5pct'], np.quantile(draws[:, j], .05), atol=1e-10)


def test_kao_legacy_preserved_auto_is_explicit_recorded_and_reproducible():
    f = frame(cube())
    default = oe.xtcointtest(f, 'y', ['x', 'z'], 'id', 'time')
    fixed = oe.xtcointtest(f, 'y', ['x', 'z'], 'id', 'time', kernel_lags=int(4 * (.42)**(2/9)))
    assert_allclose(default['statistic'], fixed['statistic'])
    auto = oe.xtcointtest(f, 'y', ['x', 'z'], 'id', 'time', bandwidth='auto')
    assert auto.attrs['bandwidth'] == 'auto'
    assert len(auto.attrs['kernel_lags_by_panel']) == 7
    assert auto.attrs['kernel_lags'] is None
    assert not np.allclose(auto['statistic'], fixed['statistic'])


def test_kao_automatic_bandwidth_and_full_hac_independent():
    v = cube()
    n, t, _ = v.shape
    f = frame(v)
    # Global LSDV by SVD, then per-panel plug-in bandwidth and matrix HAC.
    dummies = np.repeat(np.eye(n), t, axis=0)
    design = np.column_stack((v[:, :, 1:].reshape(-1, 2), dummies))
    e = fit(design, v[:, :, 0].ravel())[1].reshape(n, t)
    rho = np.sum(e[:, :-1] * e[:, 1:]) / np.sum(e[:, :-1]**2)
    selected = []
    omega = np.zeros((3, 3))
    for unit, residual in zip(v, e):
        u = residual[1:] - rho * residual[:-1]
        pilot = int(4 * ((t - 1) / 100)**(2/9))
        gamma = np.array([u[j:] @ u[:len(u)-j] / len(u) for j in range(pilot + 1)])
        s0 = gamma[0] + 2 * gamma[1:].sum()
        s1 = 2 * np.arange(1, pilot + 1) @ gamma[1:]
        bandwidth = min(len(u) - 1, int(1.1447 * ((s1 / s0)**2)**(1/3) * len(u)**(1/3)))
        selected.append(bandwidth)
        dw = np.diff(unit, axis=0)
        hac = dw.T @ dw
        for j in range(1, bandwidth + 1):
            cross = dw[j:].T @ dw[:-j]
            hac += (1 - j / (bandwidth + 1)) * (cross + cross.T)
        omega += hac / (n * (t - 1))
    r = oe.xtcointtest(f, 'y', ['x', 'z'], 'id', 'time', bandwidth='auto')
    assert r.attrs['kernel_lags_by_panel'] == selected
    conditional = omega[0, 0] - omega[0, 1:] @ np.linalg.solve(omega[1:, 1:], omega[1:, 0])
    assert_allclose(r.attrs['sigma2_0v'], conditional, atol=1e-10)


@pytest.mark.parametrize('method,ar', [('pedroni','panel'), ('pedroni','same'),
                                     ('westerlund','panel'), ('westerlund','same')])
def test_published_replication_data_with_frozen_independent_reference(method, ar):
    base = Path(__file__).parent / 'fixtures/panel_cointegration'
    f = pd.read_csv(base / 'xtcoint_model.csv')
    reference = json.loads((base / 'reference.json').read_text())[method][ar]
    r = oe.xtcointtest(f, 'productivity', ['rddomestic', 'rdforeign'], 'id', 'time',
                       test=method, ar=ar, lags=1, kernel_lags=4)
    for key, expected in reference.items():
        assert_allclose(r.attrs['raw_statistics'][key], expected, rtol=1e-8)
    assert r.attrs['n_panels'] == 100
    assert r.attrs['periods'] == 150


def test_domains_and_work_admission():
    f = frame(cube())
    kwargs = dict(y='y', x=['x', 'z'], panel='id', time='time', test='pedroni')
    with pytest.raises(AnalysisError, match='max_work'):
        oe.xtcointtest(f, **kwargs, max_work=1)
    with pytest.raises(AnalysisError, match='balanced'):
        oe.xtcointtest(f.iloc[1:], **kwargs)
    f.loc[0, 'x'] = np.nan
    with pytest.raises(AnalysisError, match='missing'):
        oe.xtcointtest(f, **kwargs)


def test_independent_brownian_moment_simulation():
    # Independent random-walk/SVD replication of HW Table24, T500, two x.
    # Sampling tolerance accounts for 2500 draws vs the paper's 100,000.
    rng = np.random.default_rng(783)
    ratios = []
    for _ in range(2500):
        v = rng.normal(size=(500, 3)).cumsum(axis=0)
        design = np.column_stack((np.ones(500), v[:, 1:]))
        e = fit(design, v[:, 0])[1]
        ratios.append(np.sum(np.cumsum(e)**2) / np.sum(e**2) / 500**2)
    assert_allclose(np.mean(ratios), .019406, rtol=.06)
    assert_allclose(np.var(ratios, ddof=1), .000273, rtol=.15)
