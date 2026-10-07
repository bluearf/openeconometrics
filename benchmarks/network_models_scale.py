"""Physical-fixture MRQAP, fixed-K SBM and sparse temporal workflow receipt.

Fixture/import/operation times are separate; one measured call, no warmups.
These are generated topologies and explicit budgets, not supported-size claims.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import torch

import openecon as oe
from network_workflow_scale import cpu_name, peak_rss, write_records

SOURCE_PATHS = (
    'src/openecon/networks.py', 'src/openecon/_network_sparse.py',
    'src/openecon/_network_qap.py', 'src/openecon/_network_mrqap.py',
    'src/openecon/_network_sbm.py', 'src/openecon/_network_temporal.py',
    'benchmarks/network_models_scale.py', 'benchmarks/network_workflow_scale.py',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    fingerprints = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    phases, receipts = [], {}
    memory_mb, max_work = 1024, 2_000_000_000

    def timed(name, function):
        start = time.perf_counter()
        result = function()
        phases.append(dict(name=name, seconds=time.perf_counter()-start))
        return result

    def imported(path, directed):
        return oe.network(oe.scan(path), weight='weight', directed=directed,
                          batch_rows=4096, max_memory_mb=memory_mb)

    with tempfile.TemporaryDirectory(prefix='openecon-network-models-') as owned:
        temporary = Path(owned)
        n = 1000
        def records(kind):
            for u in range(n):
                for step in range(1, 12):
                    a, b = int(step <= 10), int(step <= 9 or step == 11)
                    value = a if kind == 1 else b if kind == 2 else 3*a + 2*b
                    if value:
                        yield u, (u+step) % n, float(value)
        paths = [temporary / f'mrqap-{i}.parquet' for i in range(3)]
        physical = [timed(f'create_mrqap_{i}', lambda i=i: write_records(paths[i], records(i), weighted=True))
                    for i in range(3)]
        graphs = [timed(f'import_mrqap_{i}', lambda i=i: imported(paths[i], True)) for i in range(3)]
        result = timed('qap_regression', lambda: graphs[0].qap_regression(
            {'first':graphs[1], 'second':graphs[2]}, permutations=39, seed=42, max_work=max_work))
        coefficients = dict(zip(result.term, result.coefficient))
        assert math.isclose(coefficients['first'], 3., abs_tol=1e-12)
        assert math.isclose(coefficients['second'], 2., abs_tol=1e-12)
        assert abs(coefficients['Constant']) < 1e-12
        assert result.dyads.iloc[0] == n*(n-1)
        for row in result.iloc[:2].itertuples():
            assert row.pvalue == (row.extreme_permutations+1)/40
        receipts['mrqap'] = dict(physical=physical, nodes=n,
            edges_per_snapshot=[g.edge_count for g in graphs], permutations=39,
            result=result.to_dict(orient='records'), metadata=result.attrs,
            verification='Analytical Y=3X1+2X2 over every eligible ordered dyad; plus-one p-values; no fitted recovery claim')
        # Constant has no permutation test; export nullable scalars as JSON null.
        receipts['mrqap']['result'] = json.loads(result.to_json(orient='records'))
        del graphs, result
        gc.collect()

        n, k, degree = 4000, 4, 8
        path = temporary / 'sbm.parquet'
        def block_records():
            for u in range(n):
                base = u//(n//k)*(n//k)
                for step in range(1, degree+1):
                    yield u, base + (u-base+step)%(n//k), 1.
        physical = timed('create_sbm', lambda: write_records(path, block_records(), weighted=True))
        graph = timed('import_sbm', lambda: imported(path, False))
        fitted = timed('block_model', lambda graph=graph: graph.block_model(k,
            initial={u:u//(n//k) for u in range(n)}, starts=1, max_iter=10, max_work=max_work))
        lookup = dict(zip(fitted['membership'].node, fitted['membership'].block))
        # Re-enumerate imported sparse edges independently, then calculate all
        # possible off-diagonal dyads algebraically for the returned partition.
        sizes = [sum(block == b for block in lookup.values()) for b in range(k)]
        counts = {}
        for u, v, _ in block_records():
            pair = tuple(sorted((lookup[u], lookup[v])))
            counts[pair] = counts.get(pair, 0)+1
        likelihood = 0.
        for row in fitted['blocks'].itertuples():
            a, b = row.source_block, row.target_block
            d = math.comb(sizes[a],2) if a == b else sizes[a]*sizes[b]
            e = counts.get((a,b),0)
            assert row.dyads == d and row.edges == e
            if d and 0 < e < d:
                likelihood += e*math.log(e/d)+(d-e)*math.log1p(-e/d)
        assert math.isclose(likelihood, fitted['metadata']['log_likelihood'], abs_tol=1e-6)
        receipts['sbm'] = dict(physical=physical, nodes=n, edges=graph.edge_count,
            metadata=fitted['metadata'], block_probabilities=fitted['blocks'].to_dict(orient='records'),
            verification='Supplied four-block initial partition; independent edge/potential-dyad counts and profile likelihood readback; no random-start recovery or global-optimum claim')
        del graph, fitted
        gc.collect()

        n, t, degree = 25_000, 4, 4
        def temporal_records(position):
            for u in range(n):
                for step in range(1, degree+1):
                    distance = step if step < degree else step+position
                    yield u, (u+distance)%n, 1.
        paths = [temporary/f'temporal-{i}.parquet' for i in range(t)]
        physical = [timed(f'create_snapshot_{i}', lambda i=i: write_records(paths[i], temporal_records(i), weighted=True)) for i in range(t)]
        graphs = [timed(f'import_snapshot_{i}', lambda i=i: imported(paths[i], True)) for i in range(t)]
        snapshots = timed('network_snapshots', lambda: oe.network_snapshots(dict(enumerate(graphs)), ordered=True, max_work=max_work))
        transitions = timed('transitions', lambda: snapshots.transitions(max_work=max_work))
        assert all(transitions.edges_persisted == n*(degree-1))
        assert all(transitions.edges_added == n) and all(transitions.edges_removed == n)
        persistence = timed('edge_persistence', lambda: snapshots.edge_persistence(max_edges=500_000,max_work=max_work))
        assert len(persistence) == n*(degree-1+t)
        assert all(persistence.eligible_snapshots == t)
        assert sum(persistence.present_snapshots == t) == n*(degree-1)
        assert all(persistence.longest_run == persistence.present_snapshots)
        aggregate = timed('aggregate_sum', lambda: snapshots.aggregate(max_edges=500_000,max_work=max_work))
        assert aggregate.edge_count == len(persistence) and aggregate.node_count == n
        assert aggregate._edges.values().sum().item() == n*degree*t
        path = timed('temporal_path', lambda: snapshots.temporal_path(0,9,max_work=max_work))
        assert path.attrs['reachable'] and path.attrs['arrival_position'] == 1 and len(path) == 2
        assert path.source.iloc[0] == 0 and path.target.iloc[-1] == 9
        assert list(path.position) == [0,1]
        receipts['snapshots'] = dict(physical=physical, nodes=n, layers=t, edges_per_layer=n*degree,
            union_edges=len(persistence), aggregate_edges=aggregate.edge_count,
            transitions=transitions.to_dict(orient='records'), path=path.to_dict(orient='records'),
            path_metadata=path.attrs, metadata=snapshots.metadata,
            verification='Analytical shared band edges, turnover/persistence/aggregate strength and earliest strict two-hop path; full retained layers')
    assert fingerprints == {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}, 'Source changed during measurement.'
    report = dict(status='passed', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),cpu=cpu_name(),torch_version=torch.__version__,torch_threads=torch.get_num_threads(),
        max_memory_mb_per_graph=memory_mb,max_work=max_work,warmups=0,measured_repetitions=1,
        timing_scope='fixture/import/method calls separate; Python/Torch startup excluded',
        rss_scope='whole process, libraries, fixture creation, imports and all operations',
        process_peak_rss_bytes=peak_rss(),source_sha256=fingerprints,phases=phases,fixtures=receipts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',phases=phases,process_peak_rss_mib=report['process_peak_rss_bytes']/1024**2)))


if __name__ == '__main__':
    main()
