"""Native exact matching is unchanged by source partitioning and disk storage."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset


@pytest.fixture(scope='module')
def frame():
    generator = torch.Generator().manual_seed(71)
    x, z, error = torch.randn((3, 193), generator=generator, dtype=torch.float64)
    treatment = (torch.rand(193, generator=generator) < torch.sigmoid(.3*x-.4*z)).to(torch.int64)
    return pd.DataFrame({'x': x.numpy(), 'z': z.numpy(), 'y': (1+2*treatment+x+.4*z+error).numpy(),
                         'd': treatment.numpy(), 'tied': torch.arange(193).remainder(7).numpy()})


def source(frame, rows):
    return Dataset.from_batches(lambda: (frame.iloc[i:i+rows] for i in range(0, len(frame), rows)),
                                list(frame.columns), row_count=len(frame))


@pytest.mark.parametrize('rows', [17, 81])
@pytest.mark.parametrize('options', [
    {'method': 'nnmatch'}, {'method': 'nnmatch', 'metric': 'ivariance'},
    {'method': 'nnmatch', 'metric': 'euclidean', 'neighbors': 3, 'biasadj': ['x', 'z']},
    {'method': 'nnmatch', 'estimand': 'atet', 'biasadj': ['z']},
    {'method': 'nnmatch', 'x': ['tied'], 'neighbors': 2},
    {'method': 'psmatch'}, {'method': 'psmatch', 'tmodel': 'probit', 'estimand': 'atet', 'neighbors': 2},
    {'method': 'psmatch', 'biasadj': ['x'], 'vce_neighbors': 3},
])
def test_complete_estimate_covariance_and_matching_contract(frame, rows, options):
    arguments = {'y': 'y', 'treatment': 'd', 'x': ['x', 'z'], **options}
    dense = oe.teffects(data=frame, **arguments)
    replay = oe.teffects(data=source(frame, rows), **arguments)
    assert dense.nobs == replay.nobs == len(frame)
    torch.testing.assert_close(torch.tensor([c.estimate for c in dense.coefficients], dtype=torch.float64),
                               torch.tensor([c.estimate for c in replay.coefficients], dtype=torch.float64), rtol=3e-8, atol=1e-10)
    torch.testing.assert_close(torch.tensor(dense.covariance_matrix, dtype=torch.float64),
                               torch.tensor(replay.covariance_matrix, dtype=torch.float64), rtol=3e-7, atol=1e-10)
    assert dense.extra['matches'] == replay.extra['matches']
    assert dense.extra['usage']['units_used'] == replay.extra['usage']['units_used']
    for name in ['abadie_imbens', 'propensity_adjustment']:
        assert replay.extra['variance'][name] == pytest.approx(dense.extra['variance'][name], rel=3e-7, abs=1e-10)
    assert replay.provenance['streaming']['dense_observation_matrix'] is False
    assert replay.sample_positions == []
    if options['method'] == 'psmatch':
        key = 'P(d=1)'
        for level in ['0', '1']:
            for name in ['min', 'p25', 'median', 'mean', 'p75', 'max']:
                assert replay.extra['overlap']['propensity'][key][level][name] == pytest.approx(dense.extra['overlap']['propensity'][key][level][name], rel=1e-8, abs=1e-10)
        equation = 'TME1'
        for actual, expected in zip(replay.extra['auxiliary_equations'][equation], dense.extra['auxiliary_equations'][equation], strict=True):
            assert actual['term'] == expected['term']
            for name in ['estimate', 'std_error']:
                assert actual[name] == pytest.approx(expected[name], rel=1e-8, abs=1e-10)


def test_missing_policy_caliper_and_source_change(frame):
    incomplete = frame.copy()
    incomplete.loc[[11, 150], 'x'] = float('nan')
    arguments = {'y': 'y', 'treatment': 'd', 'x': ['x', 'z'], 'method': 'psmatch', 'missing': 'drop'}
    dense = oe.teffects(data=incomplete, **arguments)
    replay = oe.teffects(data=source(incomplete, 23), **arguments)
    assert replay.dropped_rows == dense.dropped_rows == 2
    assert replay.coefficients[0].estimate == pytest.approx(dense.coefficients[0].estimate, rel=1e-8)
    with pytest.raises(oe.AnalysisError) as caught:
        oe.teffects(data=source(frame, 19), y='y', treatment='d', x=['x'], method='nnmatch', caliper=1e-12)
    assert caught.value.code == 'caliper_violation'
    calls = 0
    def mutated():
        nonlocal calls
        calls += 1
        yield frame.assign(y=frame.y+calls)
    with pytest.raises(oe.AnalysisError) as caught:
        oe.teffects(data=Dataset.from_batches(mutated, list(frame.columns)), **arguments)
    assert caught.value.code == 'source_changed'


@pytest.mark.parametrize('method', ['nnmatch', 'psmatch'])
def test_full_source_tail_and_tiny_internal_tiles(frame, method, monkeypatch, tmp_path):
    from openecon.econometrics.streaming_matching import MatchingStore
    import tempfile
    monkeypatch.setattr(MatchingStore, 'block_rows', 7)
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    large = pd.concat([frame]*5, ignore_index=True)
    large.loc[400:, 'y'] += .013*torch.arange(len(large)-400, dtype=torch.float64).numpy()
    arguments = dict(y='y', treatment='d', x=['tied'] if method == 'nnmatch' else ['x', 'z'],
                     method=method, metric='euclidean' if method == 'nnmatch' else None)
    dense = oe.teffects(data=large, **arguments)
    replay = oe.teffects(data=source(large, 23), **arguments)
    assert replay.nobs == 965
    assert replay.extra['matches'] == dense.extra['matches']
    assert replay.extra['usage']['units_used'] == dense.extra['usage']['units_used']
    assert replay.coefficients[0].estimate == pytest.approx(dense.coefficients[0].estimate, rel=1e-9, abs=1e-10)
    torch.testing.assert_close(torch.tensor(replay.covariance_matrix, dtype=torch.float64),
                               torch.tensor(dense.covariance_matrix, dtype=torch.float64), rtol=1e-7, atol=1e-10)
    assert not list(tmp_path.iterdir())


def test_matching_cleanup_on_caliper_error(frame, monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    with pytest.raises(oe.AnalysisError, match='caliper'):
        oe.teffects(data=source(frame, 17), y='y', treatment='d', x=['x'], method='nnmatch', caliper=1e-15)
    assert not list(tmp_path.iterdir())


def test_temporary_storage_failure_is_an_analysis_error(frame, monkeypatch):
    import tempfile
    def unavailable(*args, **kwargs):
        raise OSError('simulated unavailable owned temporary storage')
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', unavailable)
    with pytest.raises(oe.AnalysisError) as caught:
        oe.teffects(data=source(frame, 17), y='y', treatment='d', x=['x'], method='nnmatch')
    assert caught.value.code == 'matching_spill_failed'
