"""All-row replayed treatment effects: independent M-estimation and guards."""
import builtins
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.data import DataError
from openecon.econometrics import registry
from openecon.econometrics.streaming_teffects import _create, _nuisance, _pieces, fit_streaming
from openecon.econometrics.teffects.stacked import _jacobian, _scores
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget
from test_econ_teffects import make_data, oracle


def specification(method='aipw', estimand='ate', omodel=None, tmodel=None, *, mixed=False):
    options = {'method': method, 'estimand': estimand}
    if method != 'ipw' and omodel is not None:
        options['omodel'] = omodel
    if method != 'ra' and tmodel is not None:
        options['tmodel'] = tmodel
    return ModelSpec(estimator='teffects', outcome='y', predictors=['x1', 'x2', *(['cat'] if mixed else [])],
                     categorical=['cat'] if mixed else [], columns={'treatment': 'd'}, options=options,
                     covariance='cluster' if mixed else 'robust', cluster='cl' if mixed else None,
                     weights='f' if mixed else None, weight_type='fweight' if mixed else None,
                     missing='drop' if mixed else 'raise')


@pytest.mark.parametrize('method', ['ra', 'ipw', 'ipwra', 'aipw'])
@pytest.mark.parametrize('mixed', [False, True])
def test_global_dense_parity_categories_missing_frequency_clusters(method, mixed):
    frame = make_data(n=657)
    frame['cat'] = pd.Categorical(np.arange(len(frame))%3, categories=[0, 1, 2, 3])
    if mixed:
        frame.loc[7, 'x1'] = np.nan
        frame.loc[14, 'f'] = 0
    spec = specification(method, mixed=mixed)
    dense = registry.load_entry(registry.get('teffects'))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=73)
    assert [c.term for c in replay.coefficients] == [c.term for c in dense.coefficients]
    np.testing.assert_allclose([c.estimate for c in replay.coefficients], [c.estimate for c in dense.coefficients], rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(replay.covariance_matrix, dense.covariance_matrix, rtol=2e-6, atol=2e-8)
    assert (replay.nobs, replay.nobs_original, replay.dropped_rows) == (dense.nobs, dense.nobs_original, dense.dropped_rows)
    assert replay.extra['observations_by_level'] == dense.extra['observations_by_level']
    assert replay.extra['potential_outcome_means'] == pytest.approx(dense.extra['potential_outcome_means'], abs=2e-8)
    for equation, coefficients in dense.extra['auxiliary_equations'].items():
        actual = replay.extra['auxiliary_equations'][equation]
        assert [c['term'] for c in actual] == [c['term'] for c in coefficients]
        np.testing.assert_allclose([[c['estimate'], c['std_error']] for c in actual],
                                   [[c['estimate'], c['std_error']] for c in coefficients], rtol=2e-6, atol=2e-8)
    assert replay.sample_positions == [] and replay.predictions == []
    assert replay.provenance['streaming']['retained_reporting_rows'] <= 400
    restored = ResultBundle.model_validate_json(replay.model_dump_json())
    assert restored == replay
    assert '\\begin{tabular}' in replay.to_latex()
    json.dumps(replay.model_dump(), allow_nan=False)


CASES = [('ra', 'ate', 'linear', 'logit'), ('ra', 'atet', 'linear', 'logit'),
         ('ipw', 'ate', 'linear', 'probit'), ('ipw', 'atet', 'linear', 'logit'),
         ('ipwra', 'ate', 'linear', 'logit'), ('ipwra', 'atet', 'linear', 'probit'),
         ('aipw', 'ate', 'linear', 'probit'), ('aipw', 'pomeans', 'linear', 'logit'),
         ('ra', 'ate', 'logit', 'logit'), ('ipwra', 'ate', 'probit', 'probit'),
         ('aipw', 'ate', 'poisson', 'logit'), ('ipwra', 'atet', 'poisson', 'logit')]


@pytest.mark.parametrize('method,estimand,omodel,tmodel', CASES)
def test_independent_numpy_finite_jacobian_m_estimation_oracle(method, estimand, omodel, tmodel):
    frame = make_data(n=547)
    y = 'yb' if omodel in {'logit', 'probit'} else 'yc' if omodel == 'poisson' else 'y'
    expected, se, _, _ = oracle(frame, method, estimand, omodel, tmodel, y=y)
    spec = specification(method, estimand, omodel, tmodel).model_copy(update={'outcome': y})
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=61)
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose([c.std_error for c in actual.coefficients], se, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('method', ['ra', 'ipw', 'ipwra', 'aipw'])
@pytest.mark.parametrize('weight_cluster', [False, True])
def test_probability_weight_cluster_oracle_and_global_rescaling(method, weight_cluster):
    frame = make_data(n=513)
    expected, se, _, _ = oracle(frame, method, wcol='pw', wtype='pweight', cluster='cl' if weight_cluster else None)
    spec = specification(method).model_copy(update={'weights': 'pw', 'weight_type': 'pweight',
           'covariance': 'cluster' if weight_cluster else 'robust', 'cluster': 'cl' if weight_cluster else None})
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=67)
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose([c.std_error for c in actual.coefficients], se, atol=2e-7, rtol=2e-5)
    rescaled = fit_streaming(spec, Dataset.from_frame(frame.assign(pw=frame.pw*1e100)), batch_rows=37)
    np.testing.assert_allclose([c.estimate for c in rescaled.coefficients], [c.estimate for c in actual.coefficients], atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(rescaled.covariance_matrix, actual.covariance_matrix, atol=2e-10, rtol=2e-8)


@pytest.mark.parametrize('method', ['ra', 'ipw', 'ipwra', 'aipw'])
def test_multivalued_label_order_control_and_pomeans(method):
    frame = make_data(n=702, levels=3)
    frame['d'] = pd.Categorical(frame.d.map({0: 'C', 1: 'A', 2: 'B'}), categories=['B', 'C', 'A', 'unused'])
    spec = specification(method, 'pomeans').model_copy(update={'options': {'method': method, 'estimand': 'pomeans', 'control': 'B'}})
    dense = registry.load_entry(registry.get('teffects'))(spec, frame)
    replay = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=93)
    assert replay.extra['levels'] == dense.extra['levels']
    np.testing.assert_allclose([c.estimate for c in replay.coefficients], [c.estimate for c in dense.coefficients], atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(replay.covariance_matrix, dense.covariance_matrix, atol=2e-8, rtol=2e-6)


def test_analytic_stacked_jacobian_is_derivative_of_full_sample_scores():
    frame = make_data(n=299)
    sample = _create(specification('ipwra', 'atet', 'probit', 'logit').model_copy(update={'outcome': 'yb'}), Dataset.from_frame(frame), 47)
    nuisance = _nuisance(sample)
    pieces = [_pieces(sample, nuisance, batch) for batch in sample.batches()]
    tau = torch.tensor([.2, .6], dtype=torch.float64)
    analytical = sum(_jacobian('ipwra', layout, y, w, ind, omega, slope, tau, selection)
                     for layout, y, w, ind, omega, slope, selection in pieces)
    # Independent numerical derivative reconstructs the *unweighted* estimating
    # functions from perturbed native nuisance parameters; A is analytic above.
    gamma, betas = nuisance.treatment, nuisance.outcomes
    theta = np.concatenate([gamma.flatten().numpy(), *(b.numpy() for b in betas), tau.numpy()])
    sizes = [gamma.numel(), *(len(b) for b in betas)]
    def score_sum(value):
        from openecon.econometrics.streaming_teffects import Nuisance
        offset = sizes[0]
        g = torch.tensor(value[:offset], dtype=torch.float64).reshape_as(gamma)
        b = []
        for width in sizes[1:]:
            b.append(torch.tensor(value[offset:offset+width], dtype=torch.float64))
            offset += width
        t = torch.tensor(value[offset:], dtype=torch.float64)
        answer = torch.zeros(len(value), dtype=torch.float64)
        for batch in sample.batches():
            layout, y, w, ind, omega, _, selection = _pieces(sample, Nuisance(g, b), batch)
            answer += w@_scores('ipwra', layout, y, ind, omega, t, selection, 0, len(y))
        return answer.numpy()
    numerical = []
    for i in range(len(theta)):
        hi, lo = theta.copy(), theta.copy()
        step = 1e-5*max(1, abs(theta[i]))
        hi[i] += step
        lo[i] -= step
        numerical.append((score_sum(hi)-score_sum(lo))/(2*step))
    np.testing.assert_allclose(analytical.numpy(), np.column_stack(numerical), rtol=1e-6, atol=2e-7)


def test_no_collect_external_estimator_cpu_meta_and_unit_permutation_invariance(monkeypatch):
    frame = make_data(n=430)
    spec = specification('aipw')
    baseline = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=89)
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split('.')[0] in {'statsmodels', 'linearmodels', 'scipy', 'sklearn'}:
            raise AssertionError(f'Runtime estimator import: {name}')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    scaled = frame.sample(frac=1, random_state=441).assign(x1=lambda f: f.x1*1e80, x2=lambda f: f.x2*1e-80)
    yielded = []
    def factory():
        for i in range(0, len(scaled), 19):
            block = scaled.iloc[i:i+19]
            yielded.append(len(block))
            yield block
    source = Dataset.from_batches(factory, list(scaled.columns), row_count=len(scaled))
    with torch.device('meta'):
        replay = fit_streaming(spec, source, batch_rows=31)
        assert torch.ones(1).device.type == 'meta'
    np.testing.assert_allclose([c.estimate for c in replay.coefficients], [c.estimate for c in baseline.coefficients], atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(replay.covariance_matrix, baseline.covariance_matrix, atol=2e-8, rtol=2e-6)
    assert max(yielded) <= 19 and len(yielded) > len(scaled)//19


@pytest.mark.parametrize('change,code', [({'method': 'nnmatch', 'estimand': 'pomeans'}, 'unsupported_estimand'),
    ({'method': 'nnmatch', 'matching_vce': 'iid'}, 'streaming_options_unsupported'),
    ({'method': 'ra', 'estimand': 'atet', 'tlevel': 1}, 'streaming_options_unsupported'),
    ({'method': 'aipw', 'estimand': 'atet'}, 'unsupported_estimand'),
    ({'method': 'ra', 'tmodel': 'probit'}, 'invalid_spec'),
    ({'method': 'ra', 'control': 999}, 'invalid_treatment'),
    ({'method': 'aipw', 'pstolerance': .49}, 'overlap_violation'),
    ({'method': 'ra', 'omodel': 'logit'}, 'invalid_outcome')])
def test_contract_failure_guards(change, code):
    frame = make_data(n=281)
    spec = specification().model_copy(update={'options': change})
    with pytest.raises(AnalysisError) as exc:
        fit_streaming(spec, Dataset.from_frame(frame), batch_rows=47)
    assert exc.value.code == code


def test_global_separation_collinearity_budget_and_source_identity(tmp_path):
    frame = make_data(n=347)
    separated = frame.assign(x1=frame.d+.001*np.random.default_rng(81).normal(size=len(frame)))
    with pytest.raises(AnalysisError, match='separat') as exc:
        fit_streaming(specification('ipw'), Dataset.from_frame(separated), batch_rows=37)
    assert exc.value.code == 'overlap_violation'
    frame['copy'] = frame.x1
    spec = specification('ra').model_copy(update={'predictors': ['x1', 'copy', 'x2']})
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=49)
    assert set(actual.provenance['omitted_terms']) == {'OME0:copy', 'OME1:copy'}
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as exc:
            fit_streaming(specification().model_copy(update={'covariance': 'cluster', 'cluster': 'cl'}), Dataset.from_frame(frame))
    assert exc.value.code in {'workspace_limit', 'batch_too_large'}
    path = tmp_path/'data.csv'
    frame.to_csv(path, index=False)
    source = scan(path)
    sample = _create(specification(), source, 49)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    frame.loc[0, 'y'] += 1
    frame.to_csv(path, index=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != before
    with pytest.raises(DataError) as exc:
        list(sample.batches())
    assert exc.value.code == 'SOURCE_CHANGED'
