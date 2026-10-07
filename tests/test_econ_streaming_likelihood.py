"""Full-sample native likelihood replay: dense parity and independent identities."""
import builtins
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.replay_sample import ReplaySample
from openecon.econometrics.streaming_likelihood import ReplayObjective, _adapter, _create, fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget

NAMES = ['glm', 'poisson', 'cloglog', 'fracreg', 'nbreg', 'betareg', 'hetprobit',
         'biprobit', 'ologit', 'oprobit', 'mlogit', 'tobit', 'intreg', 'truncreg',
         'heckman', 'heckprobit', 'ivprobit', 'ivtobit', 'cpoisson', 'cnbreg',
         'tpoisson', 'tnbreg', 'zip', 'zinb', 'gnbreg', 'hurdle', 'churdle']


def fixture(name, *, mixed=False, n=850, seed=732):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=(2, n))
    cat = pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3])
    eta = .4+.3*x+(.12*(np.arange(n)%3) if mixed else 0)
    noise = rng.normal(size=n)
    latent = eta+noise
    frame = pd.DataFrame({'x': x, 'z': z, 'cat': cat, 'cluster': np.arange(n)%23,
                          'w': (np.arange(n)%3+1).astype(float)})
    roles, options = {}, {}
    if name == 'glm':
        y = latent
    elif name in {'cloglog', 'hetprobit', 'biprobit', 'heckprobit', 'ivprobit'}:
        y = (latent > 0).astype(float)
    elif name == 'fracreg':
        y = 1/(1+np.exp(-latent))
    elif name == 'betareg':
        mu = 1/(1+np.exp(-eta))
        y = rng.beta(mu*8, (1-mu)*8)
    elif name in {'ologit', 'oprobit', 'mlogit'}:
        y = np.digitize(latent, [-.5, .6])
    elif name in {'tobit', 'ivtobit'}:
        y = np.maximum(0, latent)
        options['ll'] = 0.
    elif name == 'intreg':
        y = np.floor(latent/.4)*.4
        frame['upper'] = y+.4
        roles['upper'] = 'upper'
    elif name == 'truncreg':
        y = latent
        options['ll'] = -1.
    elif name == 'heckman':
        y = latent
    elif name in {'churdle'}:
        y = np.where(rng.random(n) < .55, np.maximum(.15, latent+1.5), 0.)
    else:
        mu = np.exp(eta)
        if name in {'nbreg', 'cnbreg', 'tnbreg', 'zinb', 'gnbreg'}:
            y = rng.negative_binomial(1.4, 1.4/(1.4+mu)).astype(float)
        else:
            y = rng.poisson(mu).astype(float)
        if name in {'cpoisson', 'cnbreg'}:
            y = np.minimum(y, 4)
            options['ul'] = 4
        if name in {'tpoisson', 'tnbreg'}:
            y = np.maximum(y, 1)
            options['ll'] = 0
        if name in {'zip', 'zinb', 'hurdle'}:
            y[rng.random(n) < .35] = 0.
    if name == 'hetprobit':
        y = (eta+np.exp(.25*z)*noise > 0).astype(float)
        roles['het'] = ['z']
    elif name == 'biprobit':
        frame['y2'] = (.3+.25*z+.25*noise+rng.normal(size=n) > 0).astype(float)
        roles.update(outcome2='y2', predictors2=['z'])
    elif name in {'heckman', 'heckprobit'}:
        frame['selected'] = (.3+.7*z+.25*noise+rng.normal(size=n) > 0).astype(float)
        y = np.where(frame.selected == 1, y, np.nan)
        roles.update(select='selected', select_x=['z'])
    elif name in {'ivprobit', 'ivtobit'}:
        v = rng.normal(size=n)
        frame['d'] = .2+.4*x+.8*z+v
        latent = eta+.55*frame.d+.35*v+noise
        y = (latent > 0).astype(float) if name == 'ivprobit' else np.maximum(0, latent)
        roles.update(endogenous=['d'], instruments=['z'])
    elif name in {'zip', 'zinb'}:
        roles['inflate'] = ['z']
    elif name == 'gnbreg':
        roles['lnalpha'] = ['z']
    elif name in {'hurdle', 'churdle'}:
        roles['select_x'] = ['z']
    frame['y'] = y
    if mixed:
        frame.loc[5, 'x'] = np.nan
        frame.loc[11, 'w'] = 0.
    spec = ModelSpec(estimator=name, outcome='y', predictors=['x', *(['cat'] if mixed else [])],
                     categorical=['cat'] if mixed else [], intercept=name not in {'ologit', 'oprobit'}, columns=roles, options=options,
                     covariance='cluster' if mixed else 'nonrobust', cluster='cluster' if mixed else None,
                     weights='w' if mixed else None, weight_type='fweight' if mixed else None,
                     missing='drop' if mixed else 'raise')
    return frame, spec


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('mixed', [False, True])
def test_dense_full_sample_coefficient_covariance_likelihood_parity(name, mixed):
    frame, spec = fixture(name, mixed=mixed)
    dense = registry.load_entry(registry.get(name))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=73)
    assert [c.term for c in dense.coefficients] == [c.term for c in replay.coefficients]
    np.testing.assert_allclose([c.estimate for c in dense.coefficients],
                               [c.estimate for c in replay.coefficients], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, rtol=2e-5, atol=2e-7)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], rel=1e-9, abs=1e-7)
    assert (replay.nobs, replay.nobs_original, replay.dropped_rows) == (dense.nobs, dense.nobs_original, dense.dropped_rows)
    assert replay.provenance['streaming']['maximum_batch_rows'] <= 73
    assert replay.sample_positions == []
    assert len(replay.predictions) <= 400
    restored = ResultBundle.model_validate_json(replay.model_dump_json())
    assert restored.coefficients == replay.coefficients
    assert '\\begin{tabular}' in restored.to_latex()
    json.dumps(replay.model_dump(), allow_nan=False)


@pytest.mark.parametrize('name', ['poisson', 'ologit', 'hetprobit', 'ivprobit', 'intreg', 'hurdle'])
@pytest.mark.parametrize('covariance', ['robust', 'opg'])
@pytest.mark.parametrize('weight_type', ['aweight', 'iweight', 'pweight'])
def test_weight_and_score_covariance_parity(name, covariance, weight_type):
    if weight_type == 'pweight' and covariance == 'opg':
        pytest.skip('The shared registry forbids pweights with OPG covariance.')
    frame, base = fixture(name, n=520)
    frame['w'] = np.linspace(.3, 2.9, len(frame))
    spec = base.model_copy(update={'covariance': covariance, 'weights': 'w', 'weight_type': weight_type})
    dense = registry.load_entry(registry.get(name))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=101)
    np.testing.assert_allclose([c.estimate for c in dense.coefficients], [c.estimate for c in replay.coefficients], rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, rtol=2e-5, atol=2e-7)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], rel=1e-9, abs=1e-7)


@pytest.mark.parametrize('name', ['poisson', 'ologit', 'hetprobit', 'biprobit', 'ivprobit', 'intreg', 'hurdle'])
def test_replayed_gradient_hessian_and_score_identity(name):
    frame, spec = fixture(name, n=140)
    sample = _create(spec, Dataset.from_frame(frame), batch_rows=19)
    builder, start, *_ = _adapter(sample)
    objective = ReplayObjective(sample, builder, len(start))
    value, gradient, hessian = objective(start)
    eps = 1e-5
    numeric_g, numeric_h = [], []
    for column in range(len(start)):
        direction = torch.zeros_like(start)
        direction[column] = eps
        numeric_g.append(float((objective.value(start+direction)-objective.value(start-direction))/(2*eps)))
        numeric_h.append(((objective(start+direction)[1]-objective(start-direction)[1])/(2*eps)).tolist())
    np.testing.assert_allclose(gradient, numeric_g, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(hessian, np.asarray(numeric_h).T, rtol=2e-5, atol=2e-5)
    scores = torch.zeros_like(gradient)
    values = 0.
    for batch in sample.batches():
        native = builder(batch)
        scores += (native.score_rows(start)*batch.weights[:, None]).sum(0)
        values += float(native.value(start))
    np.testing.assert_allclose(gradient, scores, rtol=1e-11, atol=1e-11)
    assert float(value) == pytest.approx(values, abs=1e-11)


def test_batch_permutation_unit_rescaling_meta_and_no_external_solver(monkeypatch):
    frame, spec = fixture('poisson', n=503)
    first = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=31)
    frame['x'] *= 1e120
    frame = frame.sample(frac=1, random_state=4).reset_index(drop=True)
    old_import = builtins.__import__
    def block(name, *args, **kwargs):
        if name.startswith(('scipy', 'statsmodels', 'sklearn', 'linearmodels')):
            raise AssertionError(name)
        return old_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', block)
    with torch.device('meta'):
        other = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=97)
        assert torch.zeros(1).device.type == 'meta'
    assert other.coefficients[0].estimate == pytest.approx(first.coefficients[0].estimate, abs=2e-10)
    assert other.coefficients[1].estimate*1e120 == pytest.approx(first.coefficients[1].estimate, abs=2e-10)
    assert other.metrics['log_likelihood'] == pytest.approx(first.metrics['log_likelihood'], abs=1e-9)


def test_shared_configured_budget_snapshot_precedes_factor_allocation():
    frame, spec = fixture('poisson', n=140)
    with use_workspace_budget(1):
        sample = ReplaySample(spec, Dataset.from_frame(frame), batch_rows=13)
        sample.add_design('mean')
    sample.prepare()
    assert sample.working_bytes == 1024**2
    assert sample.provenance()['streaming']['resource_plan']['estimated_workspace_bytes'] <= 1024**2
    wide = frame.copy()
    for i in range(40):
        wide[f'z{i}'] = np.sin(np.arange(len(wide))+i)
    broad = spec.model_copy(update={'predictors': [f'z{i}' for i in range(40)]})
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            sample = ReplaySample(broad, Dataset.from_frame(wide), batch_rows=13)
            sample.add_design('mean')
            sample.prepare()
    assert error.value.code == 'workspace_limit'


@pytest.mark.parametrize('name', ['frontier', 'streg'])
@pytest.mark.parametrize('mixed', [False, True])
def test_frontier_and_parametric_survival_mixed_native_parity(name, mixed):
    rng = np.random.default_rng(525)
    n = 650
    x = rng.normal(size=n)
    frame = pd.DataFrame({'x': x, 'z': rng.normal(size=n), 'cluster': np.arange(n)%23,
                          'strata': np.arange(n)%3, 'w': np.arange(n)%3+1.})
    if name == 'frontier':
        frame['y'] = .7+.3*x+rng.normal(size=n)*.45-abs(rng.normal(size=n))*1.3
        roles, options = {}, {}
    else:
        t = rng.weibull(1.6, size=n)*np.exp(.6-.2*x)
        censor = rng.uniform(1, 4, size=n)
        frame['y'], frame['failure'] = np.minimum(t, censor), (t < censor).astype(float)
        frame['entry'] = .03
        roles = {'failure': 'failure', 'entry': 'entry'}
        if mixed:
            roles.update(strata='strata', ancillary=['z'])
        options = {}
    spec = ModelSpec(estimator=name, outcome='y', predictors=['x'], columns=roles, options=options,
                     weights='w' if mixed else None, weight_type='fweight' if mixed else None,
                     covariance='cluster' if mixed else 'nonrobust', cluster='cluster' if mixed else None)
    dense = registry.load_entry(registry.get(name))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=71)
    assert [c.term for c in dense.coefficients] == [c.term for c in replay.coefficients]
    np.testing.assert_allclose([c.estimate for c in dense.coefficients], [c.estimate for c in replay.coefficients], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, atol=2e-7, rtol=2e-5)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], abs=1e-7)
    assert (replay.nobs, replay.dropped_rows) == (dense.nobs, dense.dropped_rows)


@pytest.mark.parametrize('distribution', ['exponential', 'weibull', 'gompertz', 'lognormal', 'loglogistic', 'ggamma'])
def test_all_parametric_survival_distributions(distribution):
    rng = np.random.default_rng(525)
    n = 650
    x = rng.normal(size=n)
    t = (np.exp(.8+.2*x+rng.normal(size=n)*.6) if distribution in {'lognormal', 'loglogistic', 'ggamma'}
         else rng.weibull(1.6, size=n)*np.exp(.6-.2*x))
    censor = rng.uniform(1, 4, size=n)
    frame = pd.DataFrame({'y': np.minimum(t, censor), 'x': x, 'failure': (t < censor).astype(float)})
    spec = ModelSpec(estimator='streg', outcome='y', predictors=['x'], columns={'failure': 'failure'},
                     options={'distribution': distribution})
    dense = registry.load_entry(registry.get('streg'))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=113)
    np.testing.assert_allclose([c.estimate for c in dense.coefficients], [c.estimate for c in replay.coefficients], atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, atol=3e-7, rtol=3e-5)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], abs=1e-7)
    json.dumps(replay.model_dump(), allow_nan=False)
    assert '\\begin{tabular}' in replay.to_latex()


@pytest.mark.parametrize('distribution', ['hnormal', 'exponential', 'tnormal'])
@pytest.mark.parametrize('cost', [False, True])
def test_all_frontier_distributions_cost_sign(distribution, cost):
    rng = np.random.default_rng(525)
    n = 650
    x = rng.normal(size=n)
    u = abs(rng.normal(size=n))*1.3
    frame = pd.DataFrame({'x': x, 'y': .7+.3*x+rng.normal(size=n)*.45+(u if cost else -u)})
    spec = ModelSpec(estimator='frontier', outcome='y', predictors=['x'], options={'distribution': distribution, 'cost': cost})
    dense = registry.load_entry(registry.get('frontier'))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=103)
    np.testing.assert_allclose([c.estimate for c in dense.coefficients], [c.estimate for c in replay.coefficients], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, atol=2e-7, rtol=2e-5)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], abs=1e-7)
    for key in ['sigma_v', 'sigma_u', 'sigma2', 'lambda'] if distribution != 'tnormal' else ['sigma2', 'gamma', 'sigma_u2', 'sigma_v2']:
        assert replay.metrics[key] == pytest.approx(dense.metrics[key], rel=2e-6)


@pytest.mark.parametrize('name', ['poisson', 'hetprobit', 'intreg', 'hurdle', 'ivprobit'])
def test_two_way_cluster_covariance_exact_inclusion_exclusion(name):
    frame, base = fixture(name, n=520)
    frame['g2'] = np.arange(len(frame))//23
    spec = base.model_copy(update={'covariance': 'cluster', 'cluster': ['cluster', 'g2']})
    dense = registry.load_entry(registry.get(name))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=57)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, atol=2e-7, rtol=2e-5)
    assert replay.inference['cluster_counts'] == dense.inference['cluster_counts']
    assert replay.inference['psd_adjusted'] == dense.inference['psd_adjusted']


@pytest.mark.parametrize('family,link', [('gaussian', 'identity'), ('poisson', 'log'), ('binomial', 'logit'),
                                        ('gamma', 'log'), ('inverse_gaussian', 'log'), ('nbinomial', 'log')])
def test_glm_family_native_density_and_trials_offsets(family, link):
    rng = np.random.default_rng(327)
    n = 503
    x = rng.normal(size=n)
    offset = .03*np.sin(np.arange(n))
    mu = np.exp(.5+.2*x+offset)
    roles = {'offset': 'offset'}
    if family == 'gaussian':
        y = .5+.2*x+offset+rng.normal(size=n)
    elif family == 'binomial':
        trials = np.arange(n)%5+2
        y = rng.binomial(trials, 1/(1+np.exp(-.5-.2*x-offset)))
        roles['trials'] = 'trials'
    elif family == 'gamma':
        y = rng.gamma(2., mu/2)
    elif family == 'inverse_gaussian':
        y = rng.wald(mu, 2.)
    elif family == 'nbinomial':
        y = rng.negative_binomial(2., 2/(2+mu))
    else:
        y = rng.poisson(mu)
    frame = pd.DataFrame({'y': y, 'x': x, 'offset': offset})
    if family == 'binomial':
        frame['trials'] = trials
    spec = ModelSpec(estimator='glm', outcome='y', predictors=['x'], columns=roles,
                     options={'family': family, 'link': link})
    dense = registry.load_entry(registry.get('glm'))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=59)
    np.testing.assert_allclose([c.estimate for c in dense.coefficients], [c.estimate for c in replay.coefficients], atol=3e-8)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, atol=2e-8, rtol=2e-6)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], abs=1e-7)


@pytest.mark.parametrize('name', ['cpoisson', 'cnbreg'])
def test_row_specific_censored_events_missing_open_limits(name):
    frame, base = fixture(name, n=420)
    rng = np.random.default_rng(302)
    y = rng.negative_binomial(2, 2/(2+np.exp(.4+.3*frame.x))).astype(float) if name == 'cnbreg' else rng.poisson(np.exp(.4+.3*frame.x)).astype(float)
    frame['y'] = y
    frame['lo'], frame['hi'], frame['event'] = np.nan, np.nan, 'exact'
    left = (np.arange(len(frame))%7 == 0)&(y <= 2)
    right = (np.arange(len(frame))%7 == 1)&(y >= 3)
    interval = np.arange(len(frame))%7 == 2
    frame.loc[left, ['lo', 'event']] = [2, 'left']
    frame.loc[right, ['hi', 'event']] = [3, 'right']
    frame.loc[interval, 'lo'] = np.maximum(0, y[interval]-1)
    frame.loc[interval, 'hi'] = y[interval]+1
    frame.loc[interval, 'event'] = 'interval'
    spec = base.model_copy(update={'columns': {'left_limit': 'lo', 'right_limit': 'hi', 'censoring': 'event'}, 'options': {}})
    dense = registry.load_entry(registry.get(name))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=47)
    np.testing.assert_allclose([c.estimate for c in dense.coefficients], [c.estimate for c in replay.coefficients], atol=3e-7)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, atol=3e-7, rtol=3e-5)
    assert replay.metrics['log_likelihood'] == pytest.approx(dense.metrics['log_likelihood'], abs=1e-7)
    assert replay.dropped_rows == 0


@pytest.mark.parametrize('name,change,code', [
    ('poisson', 'negative', 'invalid_count_outcome'), ('hurdle', 'negative', 'invalid_count_outcome'),
    ('cloglog', 'separation', 'separation_detected'), ('hetprobit', 'separation', 'separation_detected'),
    ('hurdle', 'separation', 'separation_detected'), ('poisson', 'zero', 'constant_outcome'),
    ('heckman', 'twostep_robust', 'unsupported_covariance'), ('ivprobit', 'twostep_iweight', 'unsupported_weights'),
    ('glm', 'irls_separation', 'separation_detected'), ('tobit', 'minimum', 'invalid_spec'),
])
def test_global_domains_and_unsupported_choices_refuse(name, change, code):
    frame, spec = fixture(name, n=180)
    if change == 'negative':
        frame.loc[179, 'y'] = -1.
    elif change == 'separation':
        frame['y'] = (frame.x > 0).astype(float)
        if name == 'hurdle':
            spec = spec.model_copy(update={'columns': {'select_x': ['x']}})
    elif change == 'zero':
        frame['y'] = 0.
    elif change == 'twostep_robust':
        spec = spec.model_copy(update={'options': {**spec.options, 'method': 'twostep'}, 'covariance': 'robust'})
    elif change == 'twostep_iweight':
        spec = spec.model_copy(update={'options': {**spec.options, 'method': 'twostep'}, 'weights': 'w', 'weight_type': 'iweight'})
    elif change == 'irls_separation':
        frame['y'] = (frame.x > 0).astype(float)
        spec = spec.model_copy(update={'options': {'family': 'binomial', 'optimizer': 'irls'}})
    else:
        spec = spec.model_copy(update={'options': {**spec.options, 'll_at_min': True}})
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, Dataset.from_frame(frame), batch_rows=23)
    assert error.value.code == code


def test_changed_replay_source_and_disposable_cluster_spill_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENECON_SCRATCH_DIRECTORY', str(tmp_path))
    frame, spec = fixture('poisson', mixed=True, n=180)
    calls = 0
    def batches():
        nonlocal calls
        calls += 1
        copy = frame.copy()
        if calls >= 6:
            copy.loc[143, 'y'] += 1
        yield copy
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, Dataset.from_batches(batches, frame.columns.tolist(), row_count=len(frame)), batch_rows=23)
    assert error.value.code == 'source_changed'
    assert not list(tmp_path.iterdir())


def test_reporting_sample_uses_physical_positions_and_bounds_only400():
    frame, spec = fixture('poisson', mixed=True, n=850)
    result = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=79)
    assert len(result.predictions) == 400
    assert [row['row'] for row in result.predictions][:13] == [i for i in range(15) if i not in {5, 11}]
    assert result.provenance['prediction_sample'] == 'first400 retained observations'
    assert result.provenance['streaming']['dense_observation_matrix'] is False


@pytest.mark.parametrize('name', ['hetprobit', 'tobit', 'intreg', 'truncreg', 'ologit', 'oprobit', 'mlogit'])
def test_saved_outcome_independent_prediction_and_margin_parity(name):
    from openecon.econometrics.postest.prediction import predict, margins
    frame, spec = fixture(name, mixed=True, n=520)
    dense = registry.load_entry(registry.get(name))(spec, frame)
    replay = ResultBundle.model_validate_json(fit_streaming(spec, Dataset.from_frame(frame), batch_rows=81).model_dump_json())
    new = frame.loc[20:26, ['x', 'z', 'cat', 'w']].copy()
    for kind in ['response', 'xb']:
        if name in {'ologit', 'oprobit', 'mlogit'} and kind == 'xb':
            continue
        expected, actual = predict(dense, data=new, kind=kind, interval='mean'), predict(replay, data=new, kind=kind, interval='mean')
        np.testing.assert_allclose(actual.select_dtypes(include='number'), expected.select_dtypes(include='number'), rtol=2e-5, atol=2e-7)
    expected, actual = margins(dense, data=new, variables=['x']), margins(replay, data=new, variables=['x'])
    np.testing.assert_allclose(actual.select_dtypes(include='number'), expected.select_dtypes(include='number'), rtol=3e-5, atol=2e-7)
    if name == 'tobit':
        assert replay.inference['distribution'] == 't'
        assert replay.inference['df_inference'] == replay.nobs-3
        for a, b in zip(replay.coefficients, dense.coefficients, strict=True):
            assert a.p_value == pytest.approx(b.p_value, abs=2e-7)
            assert a.ci_low == pytest.approx(b.ci_low, abs=2e-7)


@pytest.mark.parametrize('name', ['ivprobit', 'ivtobit'])
def test_endogenous_working_coordinates_two_regressors_and_unit_rescaling(name):
    frame, base = fixture(name, n=630)
    rng = np.random.default_rng(5521)
    frame['z2'] = rng.normal(size=len(frame))
    frame['d2'] = -.4+.2*frame.x+.7*frame.z2+.1*frame.d+rng.normal(size=len(frame))
    frame['d'] *= 1e80
    frame['d2'] *= 1e-80
    base = base.model_copy(update={'columns': {'endogenous': ['d', 'd2'], 'instruments': ['z', 'z2']}})
    replay = fit_streaming(base, Dataset.from_frame(frame), batch_rows=59)
    # The dense implementation intentionally refuses these original units;
    # compare with a well-scaled data fit and the exact reporting Jacobian.
    stable = frame.copy()
    stable['d'] /= 1e80
    stable['d2'] /= 1e-80
    dense = registry.load_entry(registry.get(name))(base, stable)
    scales = np.ones(len(dense.coefficients))
    for i, c in enumerate(dense.coefficients):
        if c.term == 'd':
            scales[i] = 1e-80
        elif c.term == 'd2':
            scales[i] = 1e80
        elif c.term.startswith('d:'):
            scales[i] = 1e80
        elif c.term.startswith('d2:'):
            scales[i] = 1e-80
    expected = np.array([c.estimate for c in dense.coefficients])*scales
    for i, c in enumerate(dense.coefficients):
        if c.term == '/lnsigma2':
            expected[i] += np.log(1e80)
        elif c.term == '/lnsigma3':
            expected[i] += np.log(1e-80)
    np.testing.assert_allclose([c.estimate for c in replay.coefficients], expected, rtol=2e-5, atol=2e-7)
    expected_cov = np.asarray(dense.covariance_matrix)*scales[:, None]*scales[None, :]
    # Compare in the common reference units: exact zero cross-covariances
    # otherwise acquire huge-looking units from ordinary rounding noise.
    normalized = np.asarray(replay.covariance_matrix)/scales[:, None]/scales[None, :]
    np.testing.assert_allclose(normalized, expected_cov/scales[:, None]/scales[None, :], rtol=3e-5, atol=2e-7)
