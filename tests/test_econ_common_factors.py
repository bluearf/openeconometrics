"""SVD joint-design oracles and published Eberhardt (2012) macro replication."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def panel(seed=15, g=9, t=46):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=t).cumsum() * .08
    x = rng.normal(size=(g, t, 2)) + rng.normal(size=(g, 1, 2)) * f[None, :, None]
    y = np.zeros((g, t))
    b = rng.normal([.9, -.5], .2, size=(g, 2))
    for period in range(t):
        y[:, period] = .25 * y[:, period - 1] + np.sum(x[:, period] * b, axis=1) + f[period] + rng.normal(size=g)
    return pd.DataFrame({'id': np.repeat(np.arange(g), t), 'time': np.tile(np.arange(t), g),
                         'y': y.ravel(), 'x': x[:, :, 0].ravel(), 'z': x[:, :, 1].ravel()})


def oracle(frame, model, p, q, cl):
    """Fit full nuisance + slope designs by SVD, without residualization kernels."""
    v = frame[['y', 'x', 'z']].to_numpy().reshape(frame.id.nunique(), -1, 3)
    g, t, _ = v.shape
    start = max(p, q, cl)
    avg = v.mean(axis=0)
    process = None
    if model == 'amg':
        dyx = np.diff(v, axis=1)
        # Ordinary period indicators on first differences, integrated afterwards.
        fd = np.column_stack((dyx[:, :, 1:].reshape(-1, 2), np.tile(np.eye(t - 1), (g, 1))))
        b = np.linalg.lstsq(fd, dyx[:, :, 0].ravel(), rcond=None)[0]
        process = np.r_[0, np.cumsum(b[2:])]
    nuisance = [np.ones(t - start)]
    if model == 'amg':
        nuisance.append(process[start:])
    else:
        for j in range(3):
            for lag in range((0 if model == 'csdl' and j == 0 else cl) + 1):
                nuisance.append(avg[start - lag:t - lag, j])
    z = np.column_stack(nuisance)
    zcols = z.shape[1]
    slopes, variances, designs, responses = [], [], [], []
    for row in v:
        columns = [row[start - lag:t - lag, 0] for lag in range(1, p + 1)]
        for j in (1, 2):
            columns.append(row[start:, j])
            for lag in range(q):
                if model == 'csdl':
                    columns.append(row[start - lag:t - lag, j] - row[start - lag - 1:t - lag - 1, j])
                else:
                    columns.append(row[start - lag - 1:t - lag - 1, j])
        design = np.column_stack((z, *columns))
        y = row[start:, 0]
        b = np.linalg.lstsq(design, y, rcond=None)[0]
        e = y - design @ b
        inv = np.linalg.inv(design.T @ design)
        vc = inv * (e @ e) / (len(y) - len(b))
        slopes.append(b[zcols:])
        variances.append(vc[zcols:, zcols:])
        designs.append(design)
        responses.append(y)
    slopes, variances = np.asarray(slopes), np.asarray(variances)
    b = slopes.mean(axis=0)
    vc = np.cov(slopes, rowvar=False, ddof=1) / g
    if model == 'ccep':
        # Global dummy/block nuisance design + shared slopes, then cluster scores.
        n = t - start
        full = np.zeros((g * n, g * zcols + slopes.shape[1]))
        for i, d in enumerate(designs):
            full[i * n:(i + 1) * n, i * zcols:(i + 1) * zcols] = z
            full[i * n:(i + 1) * n, g * zcols:] = d[:, zcols:]
        y = np.concatenate(responses)
        joint = np.linalg.lstsq(full, y, rcond=None)[0]
        e = y - full @ joint
        bread = np.linalg.inv(full.T @ full)
        scores = np.array([full[i * n:(i + 1) * n].T @ e[i * n:(i + 1) * n] for i in range(g)])
        all_v = bread @ (scores.T @ scores) @ bread * g / (g - 1)
        b, vc = joint[g * zcols:], all_v[g * zcols:, g * zcols:]
    return b, np.atleast_2d(vc), slopes, variances


@pytest.mark.parametrize('model,p,q,cl', [('ccemg', 0, 0, 0), ('ccep', 0, 0, 0),
    ('amg', 0, 0, 0), ('dcce', 1, 0, 2), ('csardl', 1, 1, 2), ('csdl', 0, 1, 2)])
def test_joint_svd_all_slopes_covariance_and_unit_uncertainty(model, p, q, cl):
    f = panel()
    r = oe.xtcce(data=f, y='y', x=['x', 'z'], panel='id', time='time', model=model,
                 y_lags=p, x_lags=q, cs_lags=cl)
    b, vc, unit_b, unit_v = oracle(f, model, p, q, cl)
    assert_allclose([c.estimate for c in r.coefficients], b, atol=3e-10)
    assert_allclose(r.covariance_matrix, vc, atol=3e-10)
    assert_allclose(r.extra['unit_coefficients'], unit_b, atol=3e-10)
    assert_allclose(r.extra['unit_covariance'], unit_v, atol=3e-10)
    assert r.nobs == len(f) - f.id.nunique() * max(p, q, cl)
    saved = ResultBundle.model_validate_json(r.model_dump_json())
    assert saved.extra == r.extra
    if model == 'csardl':
        lr, variances = [], []
        for estimate, cov in zip(unit_b, unit_v):
            def ratios(b):
                return np.array([b[1:3].sum(), b[3:5].sum()]) / (1 - b[0])
            # Independent central differences of the nonlinear ratio map.
            jac = np.column_stack([(ratios(estimate + np.eye(5)[j] * 1e-5)
                                    - ratios(estimate - np.eye(5)[j] * 1e-5)) / 2e-5 for j in range(5)])
            lr.append(ratios(estimate))
            variances.append(jac @ cov @ jac.T)
        assert_allclose(r.extra['long_run']['estimate'], np.mean(lr, axis=0), atol=3e-10)
        assert_allclose(r.extra['long_run']['covariance'], np.cov(lr, rowvar=False) / len(lr), atol=3e-10)
        assert_allclose(r.extra['long_run']['unit_delta_covariance'], variances, rtol=2e-8)


def test_published_macro_coefficients_se_interval_and_observed_sample():
    f = pd.read_csv(Path(__file__).parent / 'fixtures/common_factors/manu_prod_model.csv')
    r = oe.xtcce(data=f, y='ly', x=['lk'], panel='nwbcode', time='year', model='ccemg', trend=True, missing='drop')
    c = r.coefficients[0]
    # Published output rounded to seven decimals; original DTA floats explain
    # several units in the last printed decimal after later precision changes.
    assert_allclose([c.estimate, c.std_error, c.ci_low, c.ci_high],
                    [.3124664, .0849231, .1460202, .4789127], atol=5e-7, rtol=0)
    assert r.nobs == 1194
    assert r.metrics['n_groups'] == 48
    assert min(r.extra['unit_nobs']) == 11
    assert max(r.extra['unit_nobs']) == 33
    amg = oe.xtcce(data=f, y='ly', x=['lk'], panel='nwbcode', time='year', model='amg', trend=True, missing='drop')
    assert abs(amg.coefficients[0].estimate - .298) < .0005
    assert abs(amg.coefficients[0].statistic - 3.66) < .005


def test_unbalanced_equal_counts_different_clock_and_rank_rejection():
    f = panel()
    # Equal counts do not imply a balanced shared calendar.
    f = f.drop(f.groupby('id').head(1).index[::2]).drop(f.groupby('id').tail(1).index[1::2])
    r = oe.xtcce(data=f, y='y', x=['x', 'z'], panel='id', time='time')
    assert r.nobs == len(f)
    assert not r.extra['balanced']
    with pytest.raises(AnalysisError, match='balanced|consecutive'):
        oe.xtcce(data=f, y='y', x=['x'], panel='id', time='time', model='dcce')
    f.loc[f.id == 0, 'x'] = 1.
    with pytest.raises(AnalysisError):
        oe.xtcce(data=f, y='y', x=['x'], panel='id', time='time')


def test_missing_order_work_and_default_lag_contract():
    f = panel()
    kwargs = dict(y='y', x=['x', 'z'], panel='id', time='time', model='dcce')
    a = oe.xtcce(data=f, **kwargs)
    b = oe.xtcce(data=f.sample(frac=1, random_state=4), **kwargs)
    assert_allclose(a.covariance_matrix, b.covariance_matrix, atol=1e-12)
    assert a.extra['lags']['cross_sectional'] == int(46 ** (1 / 3))
    with pytest.raises(AnalysisError, match='max_work'):
        oe.xtcce(data=f, **kwargs, max_work=1)
    f.loc[0, 'x'] = np.nan
    with pytest.raises(AnalysisError):
        oe.xtcce(data=f, **kwargs)
    with pytest.raises(AnalysisError, match='balanced'):
        oe.xtcce(data=f, **kwargs, missing='drop')
