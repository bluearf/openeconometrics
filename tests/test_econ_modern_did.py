"""Independent dummy/SVD/cluster and SciPy constrained-QP comparisons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def staggered():
    rng = np.random.default_rng(556)
    n, t = 18, 12
    first = np.repeat([4, 7, t], 6)
    ids, time = np.repeat(np.arange(n), t), np.tile(np.arange(t), n)
    d = (time >= first[ids]).astype(float)
    # Heterogeneous cohort and dynamic effects: TWFE includes contaminated
    # already-treated comparisons whereas BJS/SA target eligible event means.
    y = rng.normal(size=n)[ids] + time * .2 + d * (first[ids] / 4 + (time - first[ids]) * .3) + rng.normal(size=n * t)
    return pd.DataFrame({'id': ids, 'time': time, 'd': d, 'y': y}), first


def fe_design(n, t):
    ids, time = np.repeat(np.arange(n), t), np.tile(np.arange(t), n)
    return np.column_stack((np.ones(n * t), pd.get_dummies(ids).to_numpy(float)[:, 1:],
                            pd.get_dummies(time).to_numpy(float)[:, 1:]))


@pytest.mark.parametrize('model', ['bjs', 'sunab'])
def test_heterogeneous_effects_joint_covariance_against_svd_and_cluster(model):
    f, first = staggered()
    n, t = 18, 12
    r = oe.heterodid(data=f, y='y', treatment='d', panel='id', time='time', model=model)
    fe = fe_design(n, t)
    y, d = f.y.to_numpy(), f.d.to_numpy().astype(bool)
    target = np.zeros((8, len(f)))
    for e in range(8):
        selected = [i * t + c + e for i, c in enumerate(first) if c < t and c + e < t]
        target[e, selected] = 1 / len(selected)
    if model == 'bjs':
        inverse = np.linalg.pinv(fe[~d])
        prediction = fe @ (inverse @ y[~d])
        b = target @ (y - prediction)
        mapping = target.copy()
        mapping[:, ~d] -= target @ fe @ inverse
        residual = y - prediction
        for c in [4, 7]:
            for period in range(c, t):
                rows = np.flatnonzero(first == c) * t + period
                residual[rows] -= np.mean(residual[rows])
        scores = (mapping * residual).reshape(8, n, t).sum(axis=2).T
        v = scores.T @ scores * n / (n - 1)
    else:
        terms, columns = [], []
        for c in [4, 7]:
            for period in range(t):
                if period - c != -1:
                    terms.append((c, period - c))
                    columns.append(((first[f.id] == c) & (f.time == period)).to_numpy(float))
        design = np.column_stack((fe, *columns))
        fitted = sm.OLS(y, design).fit(cov_type='cluster', cov_kwds={'groups': f.id, 'use_correction': True})
        aggregation = np.zeros((8, design.shape[1]))
        for j in range(8):
            eligible = sum((first < t) & (first + j < t))
            for col, (c, event) in enumerate(terms):
                if event == j:
                    aggregation[j, fe.shape[1] + col] = sum(first == c) / eligible
        b, v = aggregation @ fitted.params, aggregation @ fitted.cov_params() @ aggregation.T
    assert_allclose([c.estimate for c in r.coefficients], b, atol=1e-10)
    assert_allclose(r.covariance_matrix, v, atol=1e-10)
    assert ResultBundle.model_validate_json(r.model_dump_json()).extra == r.extra
    assert r.extra['reference_event'] == -1


def test_bacon_exact_weights_2x2_means_and_twfe_contamination():
    f, first = staggered()
    r = oe.bacon(f, 'y', 'd', 'id', 'time')
    # Analytic cohort counts times disjoint interval lengths; these weights
    # follow the decomposition theorem, not the production within transform.
    expected = {(4, 7): 4 * 3, (4, 12): 4 * 8, (7, 4): 3 * 5, (7, 12): 7 * 5}
    total = sum(expected.values())
    y = f.y.to_numpy().reshape(18, 12)
    for _, row in r.iterrows():
        a, b = int(row.treated_cohort_index), int(row.comparison_cohort_index)
        assert row.weight == pytest.approx(expected[(a, b)] / total)
        pre = np.arange(12) < a
        post = np.arange(12) >= a
        domain = np.arange(12) < b if a < b else np.arange(12) >= b
        def means(group, window):
            return y[first == group][:, window & domain].mean()
        direct = means(a, post) - means(a, pre) - means(b, post) + means(b, pre)
        assert row.estimate == pytest.approx(direct)
    direct_twfe = sm.OLS(f.y, np.column_stack((fe_design(18, 12), f.d))).fit().params.iloc[-1]
    assert r.attrs['twfe'] == pytest.approx(direct_twfe)
    assert r.attrs['reconstructed_twfe'] == pytest.approx(direct_twfe)
    assert 'already-treated' in r.attrs['contamination']


def qp(a, b, penalty):
    scale = max(1, np.max(np.abs(a)), np.max(np.abs(b)))
    a, b, penalty = a / scale, b / scale, penalty / scale**2
    def objective(w):
        residual = a @ w - b
        return residual @ residual + penalty * (w @ w)
    def derivative(w):
        return 2 * a.T @ (a @ w - b) + 2 * penalty * w
    fit = minimize(objective, np.full(a.shape[1], 1/a.shape[1]), jac=derivative,
                   method='SLSQP', bounds=[(0, 1)] * a.shape[1],
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(len(w))},
                   options={'ftol': 1e-14, 'maxiter': 10000})
    assert fit.success, fit.message
    return fit.x


def synthetic_oracle(y, n0, t0, model, zo=None, zl=None):
    a, b = y[:n0, :t0].T.copy(), y[n0:, :t0].mean(axis=0)
    sigma = np.diff(y[:n0, :t0], axis=1).std(ddof=1)
    if model == 'sdid':
        zo = ((len(y) - n0) * (y.shape[1] - t0))**.25 * sigma if zo is None else zo
        zl = 1e-6 * sigma if zl is None else zl
        a -= a.mean(axis=0)
        b -= b.mean()
    else:
        zo, zl = 0., 0.
    w = qp(a, b, t0 * zo**2)
    if model == 'sdid':
        a, b = y[:n0, :t0].copy(), y[:n0, t0:].mean(axis=1)
        a -= a.mean(axis=0)
        b -= b.mean()
        lam = qp(a, b, n0 * zl**2)
    else:
        lam = np.zeros(t0)
    curve = y[n0:].mean(axis=0) - w @ y[:n0]
    return curve[t0:].mean() - lam @ curve[:t0], w, lam, zo, zl


@pytest.mark.parametrize('model', ['sc', 'sdid'])
def test_california_published_data_all_weights_point_and_placebo_uncertainty(model):
    base = Path(__file__).parent / 'fixtures/modern_did'
    f = pd.read_csv(base / 'california_prop99.csv', sep=';')
    before = torch.random.get_rng_state().clone()
    r = oe.synthcontrol(data=f, y='PacksPerCapita', treatment='treated', panel='State', time='Year', model=model)
    expected = json.loads((base / 'reference.json').read_text())[model]
    assert_allclose(r.coefficients[0].estimate, expected['att'], atol=.002)
    assert_allclose(r.coefficients[0].std_error, expected['std_error'], atol=.002)
    assert_allclose(r.extra['donor_weights'], expected['omega'], atol=3e-4)
    assert_allclose(r.extra['pre_period_weights'], expected['lambda'], atol=3e-4)
    assert_allclose(r.extra['placebo_effects'], expected['placebo_effects'], atol=.01)
    assert r.extra['unit_qp']['scaled_dual_gap'] <= 1e-10
    assert sum(r.extra['donor_weights']) == pytest.approx(1)
    assert min(r.extra['donor_weights']) >= 0
    if model == 'sdid':
        assert sum(r.extra['pre_period_weights']) == pytest.approx(1)
    assert torch.equal(before, torch.random.get_rng_state())
    assert ResultBundle.model_validate_json(r.model_dump_json()).extra == r.extra


def test_causal_domain_and_resource_rejections():
    f, _ = staggered()
    kwargs = dict(y='y', treatment='d', panel='id', time='time')
    with pytest.raises(AnalysisError, match='max_work'):
        oe.heterodid(data=f, **kwargs, max_work=1)
    with pytest.raises(AnalysisError, match='balanced'):
        oe.heterodid(data=f.iloc[1:], **kwargs)
    with pytest.raises(AnalysisError, match='shared adoption'):
        oe.synthcontrol(data=f, **kwargs)
    f.loc[5, 'd'] = 0
    with pytest.raises(AnalysisError, match='absorbing'):
        oe.heterodid(data=f, **kwargs)
