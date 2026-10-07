"""Heterogeneous residual panel cointegration, with separate moment contracts."""
from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.unitroot.common import (
    KERNELS, check_choice, check_count, check_flag, column_names, load_panel,
)
from openecon.econometrics.unitroot.series import kpss_bandwidth
from openecon.econometrics.unitroot.westerlund_moments import TABLES
from openecon.engines import covariance as cov
from openecon.engines.distributions import normal_cdf, normal_isf
from openecon.engines.linalg import least_squares
from openecon.resources import plan_workspace

# Pedroni (2004) Corollary 1, m=1; Pedroni (1999) Table 2, m=2..7.
# Each row: panel-v, panel-rho, panel-t, group-rho, group-t (mean, variance).
_PEDRONI = {
    'none': [
        [(4.,27.81),(-2.77,24.91),(-1.01,1.50),(-6.84,26.78),(-1.39,.78)],
        [(6.982,81.145),(-6.388,64.288),(-1.662,1.559),(-9.889,41.943),(-1.992,.649)],
        [(10.402,140.804),(-10.191,89.962),(-2.156,1.286),(-13.865,57.801),(-2.440,.600)],
        [(14.254,182.450),(-14.136,103.176),(-2.571,1.028),(-17.834,72.097),(-2.819,.567)],
        [(18.198,217.784),(-18.042,120.787),(-2.926,.928),(-21.805,88.611),(-3.151,.559)],
        [(22.169,256.530),(-21.985,132.499),(-3.244,.820),(-25.750,103.371),(-3.450,.544)],
        [(26.120,277.429),(-25.889,143.561),(-3.533,.750),(-29.627,117.059),(-3.723,.530)],
    ],
    'constant': [
        [(8.62,60.75),(-6.02,31.27),(-1.73,.93),(-9.05,35.98),(-2.03,.66)],
        [(11.754,104.546),(-9.495,57.610),(-2.177,.964),(-12.938,51.49),(-2.453,.618)],
        [(15.197,151.094),(-13.256,81.772),(-2.576,.923),(-16.888,67.123),(-2.827,.585)],
        [(18.910,190.661),(-17.163,99.331),(-2.930,.843),(-20.841,81.835),(-3.157,.560)],
        [(22.715,231.864),(-21.013,119.546),(-3.241,.800),(-24.775,98.278),(-3.452,.553)],
        [(26.603,270.451),(-24.944,134.341),(-3.531,.750),(-28.720,113.131),(-3.726,.542)],
        [(30.457,293.431),(-28.795,144.615),(-3.795,.685),(-32.538,126.059),(-3.976,.525)],
    ],
    'trend': [
        [(17.86,101.68),(-10.54,39.52),(-2.29,.66),(-13.65,50.91),(-2.53,.56)],
        [(21.162,160.249),(-14.011,64.219),(-2.648,.690),(-17.359,66.387),(-2.872,.555)],
        [(24.556,198.167),(-17.600,83.815),(-2.967,.686),(-21.116,81.832),(-3.179,.548)],
        [(28.046,239.425),(-21.287,103.905),(-3.262,.688),(-24.930,97.362),(-3.464,.543)],
        [(31.738,276.997),(-25.130,124.613),(-3.545,.686),(-28.849,113.145),(-3.737,.538)],
        [(35.537,310.982),(-28.981,138.227),(-3.806,.654),(-32.716,127.989),(-3.986,.530)],
        [(39.231,348.217),(-32.756,154.378),(-4.047,.638),(-36.494,140.756),(-4.217,.518)],
    ],
}


def _ols(x, y):
    return kernel_call(least_squares, x.contiguous(), y.contiguous(), drop_collinear=False)


def _lrv(v, lags, kernel):
    scores = v[:, None] if v.ndim == 1 else v
    return kernel_call(cov.meat_hac, scores.contiguous(), lags, kernel) / len(v)


def _raw(cube, test, deterministic, ar, lags, kernel, kernel_lags, bandwidth, demean):
    n, t, dimension = cube.shape
    if demean:
        cube = cube - cube.mean(dim=0, keepdim=True)
    d = [] if deterministic == 'none' else [torch.ones(t, dtype=torch.float64)]
    if deterministic == 'trend':
        d.append(torch.arange(t, dtype=torch.float64))
    residuals, betas = [], []
    for v in cube:
        design = torch.cat((torch.stack(d, dim=1), v[:, 1:]), dim=1) if d else v[:, 1:]
        fit = _ols(design, v[:, 0])
        if float(fit.ssr) <= 1e-22 * float(v[:, 0].square().sum()):
            raise AnalysisError('perfect_fit', 'A unit cointegrating regression fits exactly.')
        residuals.append(fit.resid)
        betas.append(fit.beta.tolist())
    e = torch.stack(residuals)
    if test == 'westerlund':
        r = e.square().sum(dim=1)
        s = e.cumsum(dim=1).square().sum(dim=1)
        raw = math.sqrt(n) * float((s / r).mean() if ar == 'panel' else s.sum() / r.sum()) / t ** 2
        return {'variance_ratio': raw}, {'unit_cointegrating_coefficients': betas, 'kernel_lags_by_panel': []}
    pp_numerators, denominators, omega, weights, adf_num, adf_den, adf_s2, selected = ([] for _ in range(8))
    for v, u in zip(cube, e):
        du = u[1:] - u[:-1]
        fit = _ols(u[:-1, None], du)
        innovation = fit.resid
        lag = kpss_bandwidth(innovation) if bandwidth == 'auto' else kernel_lags
        selected.append(lag)
        s2 = float(innovation.square().mean())
        lr = float(_lrv(innovation, lag, kernel)[0, 0])
        if not lr > 0:
            raise AnalysisError('invalid_long_run_variance', 'A unit innovation long-run variance is not positive.')
        pp_numerators.append(float(u[:-1] @ du) - (t - 1) * (lr - s2) / 2)
        denominators.append(float(u[:-1] @ u[:-1]))
        omega.append(lr)
        # Conditional variance of difference-regression innovations (Pedroni's
        # long-run L11^2); remove deterministic differences as well as Delta X.
        changes = v[1:] - v[:-1]
        dx = changes[:, 1:]
        if deterministic == 'trend':
            dx = torch.cat((torch.ones((t - 1, 1), dtype=torch.float64), dx), dim=1)
        eta = _ols(dx, changes[:, 0]).resid
        condition = float(_lrv(eta, lag, kernel)[0, 0])
        if not condition > 0:
            raise AnalysisError('invalid_long_run_variance', 'Conditional difference long-run variance is not positive.')
        weights.append(1 / condition)
        delta, level = du[lags:], u[lags:-1]
        if lags:
            regressors = torch.stack([du[lags - j:len(du) - j] for j in range(1, lags + 1)], dim=1)
            delta, level = _ols(regressors, delta).resid, _ols(regressors, level).resid
        augmented = _ols(level[:, None], delta)
        adf_num.append(float(level @ delta))
        adf_den.append(float(level @ level))
        adf_s2.append(float(augmented.resid.square().sum()) / len(delta))
    b, r, o, w, a, h, s = [torch.tensor(v, dtype=torch.float64) for v in
                            (pp_numerators, denominators, omega, weights, adf_num, adf_den, adf_s2)]
    if bool((r <= 0).any()) or bool((h <= 0).any()) or bool((s <= 0).any()):
        raise AnalysisError('perfect_fit', 'A residual DF/ADF regression has zero denominator or variance.')
    if ar == 'panel':
        raw = {'modified_pp': t / math.sqrt(n) * float((b / r).sum()),
               'pp': float((b / (o * r).sqrt()).sum()) / math.sqrt(n),
               'adf': float((a / (s * h).sqrt()).sum()) / math.sqrt(n)}
    else:
        weighted_r, weighted_b = float(w @ r), float(w @ b)
        raw = {'modified_variance_ratio': t ** 2 * n ** 1.5 / weighted_r,
               'modified_pp': t * math.sqrt(n) * weighted_b / weighted_r,
               'pp': weighted_b / math.sqrt(float((w * o).mean()) * weighted_r),
               'adf': float(w @ a) / math.sqrt(float(s.mean()) * float(w @ h))}
    return raw, {'unit_cointegrating_coefficients': betas, 'kernel_lags_by_panel': selected}


def _moments(test, deterministic, ar, m):
    if test == 'westerlund':
        number = (20 if ar == 'same' else 23) + ('none', 'constant', 'trend').index(deterministic)
        row = TABLES[str(number)]
        return {'variance_ratio': (row['mean'][m - 1], row['variance'][m - 1])}
    panel_v, panel_r, panel_t, group_r, group_t = _PEDRONI[deterministic][m - 1]
    return ({'modified_pp': group_r, 'pp': group_t, 'adf': group_t} if ar == 'panel' else
            {'modified_variance_ratio': panel_v, 'modified_pp': panel_r, 'pp': panel_t, 'adf': panel_t})


def run(data, y, x, panel, time, *, test, deterministic, ar, lags, kernel,
        kernel_lags, bandwidth, demean, bootstrap, seed, block_length, max_work):
    check_choice(test, 'test', ('pedroni', 'westerlund'))
    check_choice(deterministic, 'deterministic', ('none', 'constant', 'trend'))
    check_choice(ar, 'ar', ('panel', 'same'))
    check_choice(kernel, 'kernel', KERNELS)
    check_choice(bandwidth, 'bandwidth', ('auto', 'legacy', 'fixed'))
    check_flag(demean, 'demean')
    lags = check_count(lags, 'lags')
    bootstrap = check_count(bootstrap, 'bootstrap')
    seed = check_count(seed, 'seed')
    max_work = check_count(max_work, 'max_work', minimum=1)
    if bootstrap > 5000:
        raise AnalysisError('work_budget', 'Bootstrap output is bounded at 5000 replications.')
    if not isinstance(y, str):
        raise AnalysisError('invalid_spec', 'y must name one column.')
    names = column_names(x, 'x')
    if not 1 <= len(names) <= 7:
        raise AnalysisError('calibration_domain', 'Published moment contract supports one to seven regressors.')
    # Admission precedes the sorted resident copy in load_panel.
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError('unsupported_data', 'Heterogeneous panel cointegration currently requires a resident DataFrame; no Dataset collection occurs.')
    size = len(data)
    plan = plan_workspace('panel cointegration input and bootstrap', {
        'selected_sorted_numeric_input': 160 * size * (len(names) + 3),
        'bootstrap_statistics': 128 * bootstrap,
    })
    sample = load_panel(data, [y, *names], panel, time)
    if not sample.balanced:
        raise AnalysisError('unbalanced_panel', 'Heterogeneous panel cointegration requires a shared balanced consecutive clock.')
    cube = sample.cube()
    n, t, m = cube.shape
    if n < 2 or t < max(20, lags + m + 4):
        raise AnalysisError('insufficient_observations', 'Need at least two units and 20 sufficiently long periods.')
    if kernel_lags is not None:
        kernel_lags = check_count(kernel_lags, 'kernel_lags')
        bandwidth = 'fixed'
    elif bandwidth == 'fixed':
        raise AnalysisError('invalid_lags', 'Fixed bandwidth needs kernel_lags.')
    else:
        kernel_lags = int(4 * (t / 100) ** (2 / 9))
    if bandwidth == 'auto' and kernel != 'bartlett':
        raise AnalysisError('unsupported_bandwidth', 'Automatic Newey-West bandwidth is implemented for Bartlett only.')
    if kernel_lags >= t - 1:
        raise AnalysisError('invalid_lags', 'Kernel lags must be smaller than T-1.')
    work = 8 * n * t * (m + lags + 2) ** 2 * (bootstrap + 1)
    if work > max_work:
        raise AnalysisError('work_budget', 'Declared regression/bootstrap work exceeds max_work.')
    args = (test, deterministic, ar, lags, kernel, kernel_lags, bandwidth, demean)
    raw, details = _raw(cube, *args)
    moments = _moments(test, deterministic, ar, len(names))
    standardized = {key: (v - math.sqrt(n) * moments[key][0]) / math.sqrt(moments[key][1]) for key, v in raw.items()}
    def upper(key):
        return key == 'modified_variance_ratio'
    rows = [[v, normal_cdf(-v if upper(key) else v),
             normal_isf(.05) if upper(key) else -normal_isf(.05)] for key, v in standardized.items()]
    boot = None
    if bootstrap:
        block_length = max(1, math.ceil((t - 1) ** (1 / 3))) if block_length is None else check_count(block_length, 'block_length', minimum=1)
        if block_length >= t - 1:
            raise AnalysisError('invalid_block_length', 'Blocks must be shorter than the difference series.')
        # Null: integrated, centered joint differences, with shared circular
        # block indices across units/variables to retain cross-sectional dependence.
        changes = cube[:, 1:] - cube[:, :-1]
        changes = changes - changes.mean(dim=1, keepdim=True)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        draws = []
        blocks = math.ceil((t - 1) / block_length)
        for _ in range(bootstrap):
            starts = torch.randint(t - 1, (blocks,), generator=generator)
            indices = (starts[:, None] + torch.arange(block_length)[None, :]).flatten()[:t - 1] % (t - 1)
            null = torch.cat((torch.zeros((n, 1, m), dtype=torch.float64), changes[:, indices].cumsum(dim=1)), dim=1)
            statistic, _ = _raw(null, *args)
            draws.append([statistic[key] for key in raw])
        draws = torch.tensor(draws, dtype=torch.float64)
        for j, (key, observed) in enumerate(raw.items()):
            extreme = draws[:, j] >= observed if upper(key) else draws[:, j] <= observed
            rows[j].extend([(1 + int(extreme.sum())) / (bootstrap + 1),
                            float(torch.quantile(draws[:, j], .95 if upper(key) else .05))])
        boot = {'replications': bootstrap, 'seed': seed, 'block_length': block_length,
                'null': 'joint centered differences integrated without cointegration; common circular block indices',
                'raw_statistics': draws.tolist(), 'columns': list(raw), 'tail': 'VR upper; all other statistics lower'}
    return table(rows, columns=['statistic', 'p_value', 'critical_5pct'] +
                 (['bootstrap_p_value', 'bootstrap_raw_critical_5pct'] if bootstrap else []), index=list(raw),
                 title=f'{test.title()} panel cointegration', test=test, distribution='normal',
                 deterministic=deterministic, ar=ar, n_panels=n, periods=t, nobs=n * t,
                 panel_labels=sample.labels, lags=lags, kernel=kernel, bandwidth=bandwidth,
                 demean=demean, raw_statistics=raw, moments=moments, bootstrap=boot,
                 workspace_plan=plan.record(), planned_work=work, max_work=max_work,
                 h0='No cointegration', ha='Some panels cointegrated' if test == 'westerlund' and ar == 'panel' else 'All panels cointegrated',
                 calibration=('Pedroni 1999 Table 2 / 2004 Corollary 1, T=1000 numerical moments' if test == 'pedroni' else
                              'Hlouskova-Wagner 2009 Tables 20-25, T=500 numerical large-T approximation'),
                 assumptions='Heterogeneous slopes; independent units for asymptotic normal p; block bootstrap targets cross-dependent stationary joint differences; large N,T',
                 **details)
