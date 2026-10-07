"""Independent many-group cancellation and actual bounded bulk I/O proof."""
import torch

from openecon.engines.streaming_groups import ClusterAccumulator


def test_10007_groups_bulk_io_preserves_exact_cancellation(tmp_path):
    groups = 10_007
    small = torch.arange(groups, dtype=torch.int64).remainder(7)-3
    target = torch.stack((small, small.square()), 1).to(torch.float64)
    expected = torch.tensor([[sum(int(value)**2 for value in small),
                              sum(int(value)**3 for value in small)],
                             [sum(int(value)**3 for value in small),
                              sum(int(value)**4 for value in small)]], dtype=torch.float64)
    query_counts, payloads = [], []

    class TraceConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, query, *args):
            if query.startswith("SELECT key,total,correction"):
                keys = args[0]
                query_counts.append(len(keys))
                payloads.append(sum(len(key)+16*2+64 for key in keys))
            return self.connection.execute(query, *args)

        def executemany(self, *args):
            return self.connection.executemany(*args)

        def close(self):
            self.connection.close()

    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    try:
        with ClusterAccumulator(2, tmp_path) as accumulator:
            accumulator._connection = TraceConnection(accumulator._connection)
            for phase in range(3):
                # A bijection changes every group's arrival order by phase;
                # compensation must survive disk loads, not just adjacent rows.
                ids = torch.arange(groups).roll(137*phase)
                values = target[ids] if phase == 1 else torch.tensor([1e16, -1e16], dtype=torch.float64).expand(groups, 2)*(1 if phase == 0 else -1)
                for start in range(0, groups, 777):
                    stop = min(groups, start+777)
                    keys = [f"group:{int(index)}".encode() for index in ids[start:stop]]
                    accumulator.add(keys, values[start:stop])
            actual, count = accumulator.finish()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert count == groups
            assert max(query_counts) <= 500
            assert max(payloads) <= 1024*1024
            assert len(query_counts) < groups//20
            assert accumulator.diagnostics["cluster_peak_cache_accounted_bytes"] <= 4*1024*1024
    finally:
        torch.set_num_threads(previous)
    assert not list(tmp_path.iterdir())
