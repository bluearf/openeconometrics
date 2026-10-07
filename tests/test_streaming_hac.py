"""Finite-lag disk HAC equals the independent existing full-sample tensor kernel."""
import pandas as pd
import pytest
import torch

from openecon.engines.covariance import meat_hac
from openecon.engines.streaming_hac import HACAccumulator
from openecon.engines.contracts import KernelError
from openecon.streaming_design import encode_cluster_labels


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "truncated"])
@pytest.mark.parametrize("timed", [False, True])
def test_all_boundary_pairs_groups_gaps_and_large_integer_periods(kernel, timed):
    gen = torch.Generator().manual_seed(32)
    n = 8317
    scores = torch.randn((n, 3), generator=gen, dtype=torch.float64)
    panel = torch.arange(n).remainder(3)
    time = torch.arange(n).div(3, rounding_mode="floor")*2+(1<<54)
    order = torch.randperm(n, generator=gen) if timed else torch.arange(n)
    scores, panel, time = scores[order], panel[order], time[order]
    with HACAccumulator(3, 7, kernel) as acc:
        path = acc.path.parent
        for start in range(0, n, 357):
            end = min(start+357, n)
            acc.add(scores[start:end], torch.arange(start, end)*3,
                    time=pd.Series(time[start:end].numpy()) if timed else None,
                    units=encode_cluster_labels(pd.Series(panel[start:end].numpy())) if timed else None)
        actual = acc.finish()
        assert acc.diagnostics["hac_rows"] == n
    assert not path.exists()
    expected = meat_hac(scores, 7, kernel, time=time if timed else None, panel=panel if timed else None)
    torch.testing.assert_close(actual, expected, rtol=5e-13, atol=1e-9)


def test_datetimes_rank_globally_and_duplicates_fail():
    dates = pd.to_datetime(["2020-01-01", "2020-03-01", "2020-02-01", "2020-01-01"])
    panel = pd.Series(["a", "a", "b", "b"])
    scores = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=torch.float64)
    with HACAccumulator(2, 2) as acc:
        for start in (0, 2):
            acc.add(scores[start:start+2], torch.arange(start, start+2), time=pd.Series(dates[start:start+2]), units=encode_cluster_labels(panel.iloc[start:start+2]))
        expected = meat_hac(scores, 2, time=torch.tensor([0, 2, 1, 0]), panel=torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(acc.finish(), expected)
    with HACAccumulator(2, 2) as acc:
        acc.add(scores[:1], torch.tensor([0]), time=pd.Series([4]))
        with pytest.raises(KernelError) as caught:
            acc.add(scores[:1], torch.tensor([1]), time=pd.Series([4]))
        assert caught.value.code == "duplicate_time"


def test_guard_before_lag_allocation():
    from openecon.analysis_contracts import AnalysisError
    with pytest.raises(AnalysisError) as caught:
        HACAccumulator(3, 10**12, budget_bytes=32*1024**2)
    assert caught.value.code == "workspace_limit"


@pytest.mark.parametrize("lags", [0, 3, 10**9])
@pytest.mark.parametrize("timed", [False, True])
def test_quadratic_spectral_uses_every_lag_with_bounded_tiles(lags, timed):
    generator = torch.Generator().manual_seed(295)
    n = 1057
    scores = torch.randn((n, 3), generator=generator, dtype=torch.float64)
    panel = torch.arange(n).remainder(2)
    periods = torch.arange(n).div(2, rounding_mode="floor")*3+(1<<54)
    order = torch.randperm(n, generator=generator) if timed else torch.arange(n)
    scores, panel, periods = scores[order], panel[order], periods[order]
    with HACAccumulator(3, lags, "quadratic_spectral") as accumulator:
        path = accumulator.path.parent
        for begin in range(0, n, 91):
            end = min(begin+91, n)
            accumulator.add(scores[begin:end], torch.arange(begin, end),
                            time=pd.Series(periods[begin:end].numpy()) if timed else None,
                            units=encode_cluster_labels(pd.Series(panel[begin:end].numpy())) if timed else None)
        actual = accumulator.finish()
        assert accumulator.diagnostics["hac_pair_count"] == (0 if not lags else sum(k*(k-1)//2 for k in ([529, 528] if timed else [n])))
    assert not path.exists()
    expected = meat_hac(scores, lags, "quadratic_spectral", time=periods if timed else None,
                        panel=panel if timed else None)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-9)


def test_quadratic_spectral_global_date_ranks():
    dates = pd.to_datetime(["2020-01-01", "2020-03-01", "2020-02-01", "2020-01-01"])
    scores = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=torch.float64)
    with HACAccumulator(2, 2, "quadratic_spectral") as accumulator:
        accumulator.add(scores, torch.arange(4), time=pd.Series(dates),
                        units=encode_cluster_labels(pd.Series(["a", "a", "b", "b"])))
        expected = meat_hac(scores, 2, "quadratic_spectral", time=torch.tensor([0, 2, 1, 0]),
                            panel=torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(accumulator.finish(), expected)


@pytest.mark.parametrize('kernel', ['bartlett', 'parzen', 'truncated', 'quadratic_spectral'])
def test_zero_lag_white_allows_duplicate_times_and_unbounded_period_span(kernel):
    scores = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
    times = torch.tensor([-(1<<62), -(1<<62), (1<<62)], dtype=torch.int64)
    with HACAccumulator(2, 0, kernel) as accumulator:
        for i in range(3):
            accumulator.add(scores[i:i+1], torch.tensor([i]), time=pd.Series(times[i:i+1].numpy()))
        expected = meat_hac(scores, 0, kernel, time=times)
        torch.testing.assert_close(accumulator.finish(), expected)
        assert accumulator._connection.execute('SELECT COUNT(*) FROM scores').fetchone()[0] == 0
