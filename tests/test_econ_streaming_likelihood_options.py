"""Native likelihood options must retain their estimator on bounded replay batches."""
import json

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_likelihood import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from test_econ_streaming_likelihood import fixture


def compare(frame, spec, rows=43):
    dense = registry.load_entry(registry.get(spec.estimator))(spec, frame)
    source = Dataset.from_batches(lambda: (frame.iloc[i:i+37].copy() for i in range(0, len(frame), 37)),
                                  frame.columns.tolist(), row_count=len(frame))
    actual = fit_streaming(spec, source, batch_rows=rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in dense.coefficients]
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], [c.estimate for c in dense.coefficients],
                               atol=2e-7, rtol=3e-6)
    np.testing.assert_allclose(actual.covariance_matrix, dense.covariance_matrix, atol=3e-7, rtol=3e-5)
    assert (actual.nobs, actual.nobs_original, actual.dropped_rows) == (dense.nobs, dense.nobs_original, dense.dropped_rows)
    assert actual.provenance['streaming']['maximum_batch_rows'] <= rows
    assert actual.sample_positions == []
    for field in ('rho', 'sigma', 'lambda', 'n_selected', 'n_censored', 'n_truncated', 'n_left_censored',
                  'n_uncensored', 'n_right_censored', 'n_subjects', 'n_failures', 'time_at_risk', 'log_likelihood'):
        if field in dense.metrics:
            assert actual.metrics[field] == pytest.approx(dense.metrics[field], rel=2e-6, abs=3e-7), field
    for name, expected in dense.tests.items():
        assert actual.tests[name]['distribution'] == expected['distribution']
        for field in ('statistic', 'p_value'):
            if expected.get(field) is not None:
                assert actual.tests[name][field] == pytest.approx(expected[field], rel=4e-4, abs=3e-6), (name, field)
    ResultBundle.model_validate_json(actual.model_dump_json())
    json.dumps(actual.model_dump(), allow_nan=False)
    return dense, actual


@pytest.mark.parametrize('name', ['heckman', 'ivprobit', 'ivtobit'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('rows', [17, 83])
def test_twostep_native_full_covariance_and_diagnostics(name, weighted, rows):
    frame, base = fixture(name, n=520)
    spec = base.model_copy(update={'options': {**base.options, 'method': 'twostep'},
                                  'weights': 'w' if weighted else None,
                                  'weight_type': 'fweight' if weighted else None})
    dense, actual = compare(frame, spec, rows)
    assert actual.extra['method'] == 'twostep'
    if name != 'heckman':
        for variable, fields in dense.extra['first_stage'].items():
            for field, value in fields.items():
                assert actual.extra['first_stage'][variable][field] == pytest.approx(value, rel=3e-6, abs=3e-8)
    else:
        assert actual.extra['probit_log_likelihood'] == pytest.approx(dense.extra['probit_log_likelihood'], abs=3e-7)


@pytest.mark.parametrize('name', ['tobit', 'ivtobit'])
@pytest.mark.parametrize('weighted', [False, True])
def test_global_sample_extreme_normal_limits(name, weighted):
    frame, base = fixture(name, n=520)
    spec = base.model_copy(update={'options': {'ll_at_min': True, 'ul_at_max': True},
                                  'weights': 'w' if weighted else None,
                                  'weight_type': 'fweight' if weighted else None})
    # Extreme rows with missing regressors or zero weight must not determine a bound.
    frame.loc[3, ['y', 'x']] = [-900., np.nan]
    if weighted:
        frame.loc[6, ['y', 'w']] = [800., 0.]
    spec = spec.model_copy(update={'missing': 'drop'})
    dense, actual = compare(frame, spec, rows=29)
    assert actual.extra['limits'] == dense.extra['limits']
    assert actual.provenance['data_hash']


@pytest.mark.parametrize('weighted', [False, True])
def test_scalar_truncation_preserves_complete_global_truncation_count(weighted):
    frame, base = fixture('truncreg', n=520)
    spec = base.model_copy(update={'options': {'ll': -.8, 'ul': 2.2},
                                  'weights': 'w' if weighted else None,
                                  'weight_type': 'fweight' if weighted else None, 'missing': 'drop'})
    frame.loc[3, ['y', 'x']] = [-900., np.nan]
    if weighted:
        frame.loc[6, ['y', 'w']] = [800., 0.]
    compare(frame, spec, rows=29)


GLM_CASES = [('gaussian', 'identity'), ('gaussian', 'log'), ('binomial', 'logit'), ('binomial', 'probit'),
             ('poisson', 'log'), ('poisson', 'identity'), ('gamma', 'log'), ('gamma', 'reciprocal'),
             ('inverse_gaussian', 'log'), ('inverse_gaussian', 'inverse_squared'), ('nbinomial', 'log')]


@pytest.mark.parametrize('family,link', GLM_CASES)
@pytest.mark.parametrize('covariance', ['nonrobust', 'robust', 'opg', 'cluster'])
def test_global_irls_preserves_expected_information_covariance(family, link, covariance):
    rng = np.random.default_rng(732)
    n = 480
    x = rng.uniform(-.4, .4, n)
    mu = np.exp(.3+.3*x)
    if family == 'binomial':
        y = rng.binomial(1, 1/(1+np.exp(-(.2+.4*x))))
    elif family == 'poisson':
        y = rng.poisson(mu)
    elif family == 'nbinomial':
        y = rng.negative_binomial(2., 2./(2.+mu))
    elif family in {'gamma', 'inverse_gaussian'}:
        y = rng.gamma(5., mu/5.)
    else:
        y = mu+rng.normal(0., .25, n)
    frame = pd.DataFrame({'x': x, 'y': y, 'group': np.arange(n)%23, 'w': np.arange(n)%3+1})
    spec = ModelSpec(estimator='glm', outcome='y', predictors=['x'], covariance=covariance,
                     cluster='group' if covariance == 'cluster' else None, weights='w', weight_type='fweight',
                     options={'family': family, 'link': link, 'optimizer': 'irls'})
    _, actual = compare(frame, spec, rows=53)
    assert actual.extra['information'] == 'expected'
    assert actual.extra['optimizer'] == 'irls'


@pytest.mark.parametrize('covariance', ['nonrobust', 'robust', 'opg', 'cluster'])
@pytest.mark.parametrize('weighted', [False, True])
def test_survival_subject_intervals_and_id_cluster_covariance(covariance, weighted):
    rng = np.random.default_rng(312)
    n = 260
    x = rng.normal(size=n)
    duration = rng.weibull(1.4, n)*np.exp(-.35*x)+.01
    censoring = rng.exponential(2., n)
    time = np.minimum(duration, censoring)
    first = pd.DataFrame({'x': x, 'time': time/2, 'entry': np.zeros(n), 'event': np.zeros(n),
                          'id': np.arange(n), 'group': np.arange(n)%19, 'w': np.ones(n)})
    last = pd.DataFrame({'x': x, 'time': time, 'entry': time/2, 'event': (duration <= censoring).astype(float),
                         'id': np.arange(n), 'group': np.arange(n)%19, 'w': np.arange(n)%3+1.})
    frame = pd.concat((first, last), ignore_index=True).sample(frac=1., random_state=4).reset_index(drop=True)
    spec = ModelSpec(estimator='streg', outcome='time', predictors=['x'],
                     columns={'entry': 'entry', 'failure': 'event', 'id': 'id'},
                     options={'distribution': 'weibull'}, covariance=covariance,
                     cluster='group' if covariance == 'cluster' else None,
                     weights='w' if weighted else None, weight_type='fweight' if weighted else None)
    _, actual = compare(frame, spec, rows=31)
    assert actual.extra['subject_validation']['overlap_checked']
    if covariance == 'robust':
        assert actual.inference['cluster_count'] == n
        assert actual.inference['covariance'] == 'robust'


def test_cross_batch_survival_overlap_is_rejected_and_scratch_cleaned(tmp_path, monkeypatch):
    frame = pd.DataFrame({'time': [2., 3., 4., 5., 6., 7.], 'entry': [0., 2., 3., 4., 5., 0.],
                          'event': [0., 1., 1., 1., 1., 1.], 'id': [1, 2, 3, 4, 5, 1], 'x': [0., 1., 2., 0., 1., 2.]})
    spec = ModelSpec(estimator='streg', outcome='time', predictors=['x'],
                     columns={'entry': 'entry', 'failure': 'event', 'id': 'id'}, options={'distribution': 'weibull'})
    monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, Dataset.from_frame(frame), batch_rows=2)
    assert error.value.code == 'overlapping_records'
    assert list(tmp_path.iterdir()) == []


def test_global_extreme_discovery_detects_source_change():
    frame, spec = fixture('tobit', n=450)
    spec = spec.model_copy(update={'options': {'ll_at_min': True}})
    calls = 0
    def batches():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls > 1:
            changed.loc[len(changed)-1, 'y'] += .2
        for i in range(0, len(changed), 37):
            yield changed.iloc[i:i+37].copy()
    source = Dataset.from_batches(batches, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, source, batch_rows=43)
    assert error.value.code == 'source_changed'


ALL_LINKS = [(family, link) for family, links in {
    'gaussian': ('identity', 'log', 'power', 'reciprocal', 'inverse_squared', 'logit', 'probit', 'cloglog', 'loglog'),
    'binomial': ('logit', 'probit', 'cloglog', 'loglog', 'log', 'identity'),
    'poisson': ('log', 'identity', 'power', 'reciprocal', 'inverse_squared'),
    'gamma': ('log', 'identity', 'power', 'reciprocal', 'inverse_squared'),
    'inverse_gaussian': ('log', 'identity', 'power', 'reciprocal', 'inverse_squared'),
    'nbinomial': ('log', 'identity', 'power', 'reciprocal', 'inverse_squared', 'nbinomial'),
}.items() for link in links]


@pytest.mark.parametrize('family,link', ALL_LINKS)
def test_irls_all_native_family_link_options(family, link):
    rng = np.random.default_rng(237)
    n = 380
    x = rng.uniform(-.2, .2, n)
    if family == 'binomial':
        y = rng.binomial(1, .45+.1*x)
    elif family == 'gaussian':
        y = .5+.1*x+rng.normal(0., .04, n)
    elif family in {'poisson', 'nbinomial'}:
        y = rng.poisson(1.4+.2*x) if family == 'poisson' else rng.negative_binomial(3., 3./(4.4+.2*x))
    else:
        y = rng.gamma(12., (1.4+.2*x)/12.)
    frame = pd.DataFrame({'y': y, 'x': x})
    options = {'family': family, 'link': link, 'optimizer': 'irls', 'max_iterations': 200}
    if link == 'power':
        options['power'] = .5
    spec = ModelSpec(estimator='glm', outcome='y', predictors=['x'], options=options)
    compare(frame, spec, rows=41)


@pytest.mark.parametrize('name', ['ivprobit', 'ivtobit'])
@pytest.mark.parametrize('intercept', [True, False])
def test_overidentified_two_endogenous_newey_with_categories_and_frequency(name, intercept):
    rng = np.random.default_rng(187)
    n = 760
    x, z1, z2, z3, v1, v2, noise = rng.normal(size=(7, n))
    d1 = .3+.4*x+.8*z1+.5*z3+v1
    d2 = -.2+.2*x+.9*z2+.2*z3+.15*v1+v2
    category = pd.Categorical(np.arange(n)%3)
    latent = .2+.3*x+.5*d1-.3*d2+.2*v1+.25*v2+noise
    y = (latent > 0).astype(float) if name == 'ivprobit' else np.maximum(0., latent)
    frame = pd.DataFrame({'x': x, 'z1': z1, 'z2': z2, 'z3': z3, 'd1': d1, 'd2': d2,
                          'y': y, 'cat': category, 'w': np.arange(n)%3+1.})
    options = {'method': 'twostep', **({'ll': 0.} if name == 'ivtobit' else {})}
    spec = ModelSpec(estimator=name, outcome='y', predictors=['x', 'cat'], categorical=['cat'], intercept=intercept,
                     columns={'endogenous': ['d1', 'd2'], 'instruments': ['z1', 'z2', 'z3']},
                     options=options, weights='w', weight_type='fweight')
    dense, actual = compare(frame, spec, rows=61)
    for variable, fields in dense.extra['first_stage'].items():
        for field, value in fields.items():
            assert actual.extra['first_stage'][variable][field] == pytest.approx(value, rel=3e-6, abs=3e-8)


def test_noncanonical_irls_covariance_is_fisher_information_independently():
    rng = np.random.default_rng(991)
    n = 580
    x = rng.normal(size=n)
    y = np.exp(.3+.2*x)+rng.normal(0., .3, n)
    frame = pd.DataFrame({'x': x, 'y': y})
    spec = ModelSpec(estimator='glm', outcome='y', predictors=['x'],
                     options={'family': 'gaussian', 'link': 'log', 'optimizer': 'irls'})
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=47)
    beta = np.array([c.estimate for c in actual.coefficients])
    design = np.column_stack((np.ones(n), x))
    fitted = np.exp(design@beta)
    phi = np.sum((y-fitted)**2)/(n-2)
    expected = phi*np.linalg.inv(design.T@(design*fitted[:, None]**2))
    observed = phi*np.linalg.inv(design.T@(design*(fitted**2-(y-fitted)*fitted)[:, None]))
    np.testing.assert_allclose(actual.covariance_matrix, expected, atol=2e-11, rtol=2e-9)
    assert np.linalg.norm(expected-observed) > 1e-8


@pytest.mark.parametrize('weight_type', ['aweight', 'iweight', 'pweight'])
@pytest.mark.parametrize('binomial', [False, True])
def test_irls_offset_trials_continuous_weights_and_no_intercept(weight_type, binomial):
    rng = np.random.default_rng(93)
    n = 560
    x, offset = rng.normal(size=n), rng.normal(0., .2, n)
    frame = pd.DataFrame({'x': x, 'offset': offset, 'w': rng.uniform(.3, 3.7, n)})
    roles = {'offset': 'offset'}
    if binomial:
        frame['trials'] = rng.integers(1, 20, n)
        frame['y'] = rng.binomial(frame.trials, 1/(1+np.exp(-(.35*x+offset))))
        roles['trials'] = 'trials'
    else:
        frame['y'] = rng.gamma(3., np.exp(.35*x+offset)/3.)
    spec = ModelSpec(estimator='glm', outcome='y', predictors=['x'], intercept=False, columns=roles,
                     options={'family': 'binomial' if binomial else 'gamma', 'link': 'probit' if binomial else 'log',
                              'optimizer': 'irls', 'max_iterations': 200},
                     covariance='robust', weights='w', weight_type=weight_type)
    compare(frame, spec, rows=71)


def test_survival_native_single_cluster_dimension_contract_remains():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match='at most 1 cluster dimension'):
        ModelSpec(estimator='streg', outcome='time', predictors=['x'], columns={'failure': 'event'},
                  options={'distribution': 'weibull'}, covariance='cluster', cluster=['first', 'second'])


@pytest.mark.parametrize('name', ['heckman', 'ivprobit', 'ivtobit'])
def test_twostep_does_not_fit_preview_and_detects_late_source_changes(name, monkeypatch):
    frame, base = fixture(name, n=520)
    spec = base.model_copy(update={'options': {**base.options, 'method': 'twostep'}})
    def collect(*args, **kwargs):
        raise AssertionError('A two-step replay attempted to fit a preview instead of replaying all rows.')
    monkeypatch.setattr(Dataset, 'head', collect)
    fit_streaming(spec, Dataset.from_frame(frame), batch_rows=47)
    calls = 0
    def batches():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls >= 8:
            changed.loc[len(changed)-1, 'x'] += .4
        for i in range(0, len(changed), 37):
            yield changed.iloc[i:i+37].copy()
    source = Dataset.from_batches(batches, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, source, batch_rows=47)
    assert error.value.code == 'source_changed'


@pytest.mark.parametrize('optimizer', ['ml', 'irls'])
@pytest.mark.parametrize('seed', [1, 3, 7])
def test_binomial_trials_two_way_psd_repair_uses_native_prior_centred_basis(optimizer, seed):
    rng = np.random.default_rng(seed)
    n = 220
    x, trials = rng.normal(size=n), rng.integers(1, 80, n)
    frame = pd.DataFrame({'x': x, 'trials': trials, 'y': rng.binomial(trials, 1/(1+np.exp(-(.2+.4*x)))),
                          'a': np.arange(n)%3, 'b': np.arange(n)%4})
    spec = ModelSpec(estimator='glm', outcome='y', predictors=['x'], columns={'trials': 'trials'},
                     covariance='cluster', cluster=['a', 'b'],
                     options={'family': 'binomial', 'link': 'probit', 'optimizer': optimizer})
    dense, actual = compare(frame, spec, rows=31)
    assert dense.inference['psd_adjusted']
    np.testing.assert_allclose(actual.covariance_matrix, dense.covariance_matrix, atol=2e-12, rtol=3e-9)
