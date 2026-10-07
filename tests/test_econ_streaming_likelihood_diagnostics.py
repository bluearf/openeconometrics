"""All-sample restricted likelihoods: dense-native references and independent tests."""
import numpy as np
import pytest

from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_likelihood import fit_streaming
from test_econ_streaming_likelihood import fixture


NAMES = ['glm', 'poisson', 'cloglog', 'fracreg', 'nbreg', 'betareg', 'hetprobit',
         'biprobit', 'ologit', 'oprobit', 'mlogit', 'tobit', 'intreg', 'truncreg',
         'heckman', 'heckprobit', 'ivprobit', 'ivtobit', 'cpoisson', 'cnbreg',
         'tpoisson', 'tnbreg', 'zip', 'zinb', 'gnbreg', 'hurdle', 'churdle']


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('covariance', ['nonrobust', 'robust'])
def test_replayed_restricted_tests_null_and_pseudo_r2_match_dense(name, covariance):
    frame, base = fixture(name, n=850)
    spec = base.model_copy(update={'covariance': covariance})
    dense = registry.load_entry(registry.get(name))(spec, frame)
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=87)
    for key in ['pseudo_r_squared', 'log_likelihood_ols', 'rho', 'sigma']:
        if key in dense.metrics:
            assert key in actual.metrics
            if dense.metrics[key] is None:
                assert actual.metrics[key] is None
            else:
                assert actual.metrics[key] == pytest.approx(dense.metrics[key], rel=2e-5, abs=2e-7), (name, key)
    for key in ['null_log_likelihood', 'comparison_log_likelihood', 'probit_log_likelihood']:
        if key in dense.extra:
            if dense.extra[key] is None:
                assert actual.extra.get(key) is None
            else:
                assert actual.extra[key] == pytest.approx(dense.extra[key], rel=2e-8, abs=2e-7), (name, key)
    for key, expected in dense.tests.items():
        assert key in actual.tests, (name, key)
        observed = actual.tests[key]
        assert observed['distribution'] == expected['distribution'], (name, key)
        for field in ['statistic', 'p_value']:
            if expected[field] is not None:
                assert observed[field] == pytest.approx(expected[field], rel=3e-4, abs=3e-6), (name, key, field)
    if name in {'ivprobit', 'ivtobit'}:
        for variable, values in dense.extra['first_stage'].items():
            for key, value in values.items():
                assert actual.extra['first_stage'][variable][key] == pytest.approx(value, rel=2e-6, abs=2e-8)
    if name == 'gnbreg':
        for key in ('mean', 'min', 'max'):
            assert actual.extra['alpha'][key] == pytest.approx(dense.extra['alpha'][key], rel=2e-6, abs=2e-8)
        assert actual.extra['alpha']['definition'] == dense.extra['alpha']['definition']
    if name == 'betareg':
        for key in ('estimate', 'std_error', 'ci_low', 'ci_high'):
            assert actual.extra['precision'][key] == pytest.approx(dense.extra['precision'][key], rel=2e-6, abs=2e-8)
    assert 'full' in actual.extra['model_test_scope'].lower()


@pytest.mark.parametrize('name', ['poisson', 'cloglog', 'fracreg', 'nbreg', 'mlogit'])
def test_no_intercept_null_fixed_outcome_slopes(name):
    frame, base = fixture(name, n=530)
    spec = base.model_copy(update={'intercept': False})
    dense = registry.load_entry(registry.get(name))(spec, frame)
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=101)
    assert actual.extra['null_log_likelihood'] == pytest.approx(dense.extra['null_log_likelihood'], abs=2e-6)
    assert actual.metrics['pseudo_r_squared'] == pytest.approx(dense.metrics['pseudo_r_squared'], abs=2e-7)


def test_poisson_closed_form_reference_and_chibar_variance_independently():
    frame, spec = fixture('poisson', n=480)
    frame['w'] = np.arange(len(frame))%4+1
    spec = spec.model_copy(update={'weights': 'w', 'weight_type': 'fweight'})
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=79)
    from scipy.special import gammaln
    from scipy.stats import chi2
    y, w = frame.y.to_numpy(), frame.w.to_numpy()
    mean = w@y/w.sum()
    reference = np.sum(w*(y*np.log(mean)-mean-gammaln(y+1)))
    assert actual.extra['null_log_likelihood'] == pytest.approx(reference, abs=1e-7)
    statistic = 2*(actual.metrics['log_likelihood']-reference)
    assert actual.tests['model']['statistic'] == pytest.approx(statistic, abs=1e-7)
    assert actual.tests['model']['p_value'] == pytest.approx(chi2.sf(statistic, 1), abs=1e-10)


@pytest.mark.parametrize('name', ['tobit', 'intreg', 'nbreg', 'hetprobit', 'biprobit', 'heckman', 'gnbreg', 'zip'])
@pytest.mark.parametrize('weighted', [False, True])
def test_restricted_replays_preserve_frequency_and_cluster_sample(name, weighted):
    frame, base = fixture(name, mixed=True, n=850)
    spec = base if weighted else base.model_copy(update={'covariance': 'nonrobust', 'cluster': None})
    dense = registry.load_entry(registry.get(name))(spec, frame)
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=77)
    for key in ['null_log_likelihood', 'comparison_log_likelihood', 'probit_log_likelihood']:
        if dense.extra.get(key) is not None:
            assert actual.extra[key] == pytest.approx(dense.extra[key], rel=2e-8, abs=2e-7)
    if name == 'gnbreg':
        for key in ('mean', 'min', 'max'):
            assert actual.extra['alpha'][key] == pytest.approx(dense.extra['alpha'][key], rel=2e-6, abs=2e-8)
    for name, expected in dense.tests.items():
        assert actual.tests[name]['distribution'] == expected['distribution']
        if expected.get('statistic') is not None:
            assert actual.tests[name]['statistic'] == pytest.approx(expected['statistic'], rel=3e-4, abs=3e-6)
    assert actual.provenance['streaming']['passes'] > 10


@pytest.mark.parametrize('distribution', ['exponential', 'weibull', 'gompertz', 'lognormal', 'loglogistic', 'ggamma'])
def test_parametric_survival_full_null_density(distribution):
    from openecon.models import ModelSpec
    rng = np.random.default_rng(916)
    n = 550
    x, z = rng.normal(size=(2, n))
    import pandas as pd
    time = np.exp(.3+.2*x+.8*rng.normal(size=n))
    event = (rng.random(n) > .25).astype(int)
    frame = pd.DataFrame({'x': x, 'z': z, 'time': time, 'event': event})
    spec = ModelSpec(estimator='streg', outcome='time', predictors=['x'], columns={'failure': 'event'}, options={'distribution': distribution})
    dense = registry.load_entry(registry.get('streg'))(spec, frame)
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=91)
    assert actual.extra['null_log_likelihood'] == pytest.approx(dense.extra['null_log_likelihood'], abs=2e-6)
    assert actual.tests['model']['statistic'] == pytest.approx(dense.tests['model']['statistic'], abs=2e-5)


@pytest.mark.parametrize('distribution', ['hnormal', 'exponential', 'tnormal'])
def test_frontier_normal_boundary_reference_uses_all_rows(distribution):
    from openecon.models import ModelSpec
    import pandas as pd
    rng = np.random.default_rng(949)
    n = 903
    x = rng.normal(size=n)
    inefficiency = np.abs(rng.normal(size=n))*.85
    frame = pd.DataFrame({'x': x, 'y': .7+.3*x+rng.normal(size=n)*.45-inefficiency})
    spec = ModelSpec(estimator='frontier', outcome='y', predictors=['x'], options={'distribution': distribution})
    dense = registry.load_entry(registry.get('frontier'))(spec, frame)
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=73)
    assert actual.metrics['log_likelihood_ols'] == pytest.approx(dense.metrics['log_likelihood_ols'], abs=2e-7)
    assert actual.tests['sigma_u']['p_value'] == pytest.approx(dense.tests['sigma_u']['p_value'], rel=3e-6, abs=2e-8)


def test_constant_dispersion_equation_does_not_invent_zero_df_test():
    frame, spec = fixture('gnbreg', n=850)
    spec = spec.model_copy(update={'columns': {'lnalpha': []}})
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=83)
    dense = registry.load_entry(registry.get('gnbreg'))(spec, frame)
    assert 'lnalpha' not in actual.tests
    assert 'lnalpha' not in dense.tests
    assert actual.extra['null_log_likelihood'] == pytest.approx(dense.extra['null_log_likelihood'], abs=2e-6)


def test_unavailable_optional_restricted_fit_retains_explicit_wald_fallback(monkeypatch):
    from openecon.analysis_contracts import AnalysisError
    from openecon.econometrics import streaming_likelihood_diagnostics as diagnostics
    frame, spec = fixture('poisson', n=470)
    def refusal(*args, **kwargs):
        raise AnalysisError('precision_unsupported', 'Independent injected restricted-domain refusal.')
    monkeypatch.setattr(diagnostics, '_restricted', refusal)
    actual = fit_streaming(spec, Dataset.from_frame(frame), batch_rows=79)
    assert actual.tests['model']['distribution'] == 'chi2'
    assert 'Wald' in actual.tests['model']['label']
    assert actual.extra['null_log_likelihood'] is None
    assert actual.metrics['pseudo_r_squared'] is None
    assert actual.extra['restricted_model_diagnostics']['null']['status'] == 'unavailable'
    assert actual.extra['restricted_model_diagnostics']['null']['reason'] == 'precision_unsupported'


def test_actual_numeric_peak_survives_later_block_plan_shrinkage():
    from openecon.econometrics.replay_sample import ReplaySample
    frame, spec = fixture('poisson', n=370)
    sample = ReplaySample(spec, Dataset.from_frame(frame), batch_rows=127)
    sample.add_design('mean')
    sample.prepare()
    first = max(len(batch.frame) for batch in sample.batches())
    sample.plan_rows('independent smaller-block proof', {}, 10_000_000)
    later = max(len(batch.frame) for batch in sample.batches())
    assert later < first
    proof = sample.provenance()['streaming']
    assert proof['actual_numeric_peak_rows'] == first
    assert proof['maximum_batch_rows'] == first
    assert proof['batch_rows'] == later
