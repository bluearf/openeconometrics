"""Compare canonical directed CSR setup against the explicitly pinned .37 source.

Both paths use the same resident graph and physical batched Parquet fixture.
The old function is compiled in an isolated namespace, not installed/imported
over production code. Measurements cover CSR creation only, not graph import.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import openecon as oe
from openecon._network_sparse import csr
from network_advanced import write_fixture, cpu_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=1_000_000)
    parser.add_argument('--nodes', type=int, default=100_000)
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    assert args.repeat > 0
    root = Path(__file__).resolve().parents[1]
    old_ref = 'be4c6ab76e7ecc2fd70aebcfa6a3fbd2819c151b'
    path = 'src/openecon/_network_sparse.py'
    original = subprocess.check_output(['git','show',f'{old_ref}:{path}'], cwd=root)
    namespace = {'__name__':'owned_old_csr_benchmark'}
    exec(compile(original, '<pinned-old-csr>', 'exec'), namespace)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='openecon-csr-benchmark-') as temporary:
        file = Path(temporary) / 'edges.parquet'
        write_fixture(file, args.rows, args.nodes, 83)
        started = time.perf_counter()
        graph = oe.network(oe.scan(file), directed=True, weight='weight', max_memory_mb=1024)
        imported = time.perf_counter()-started
        times = {'before':[], 'after':[]}
        owned = {}
        for label, function in [('before',namespace['csr']), ('after',csr)]:
            for _ in range(args.repeat+1):
                gc.collect()
                started = time.perf_counter(); view = function(graph)
                elapsed = time.perf_counter()-started
                assert torch.equal(view._tensors[1], graph._arcs.indices()[1])
                assert torch.equal(view._tensors[2], graph._arcs.values())
                if times[label] or _:
                    times[label].append(elapsed)
                original_storage = {x.untyped_storage().data_ptr() for x in
                                    (graph._arcs.indices(), graph._arcs.values())}
                owned[label] = sum(x.untyped_storage().nbytes() for x in view._tensors
                                  if x.untyped_storage().data_ptr() not in original_storage)
                del view
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {'status':'measured', 'cpu':cpu_name(), 'platform':platform.platform(),
            'python':platform.python_version(), 'torch':torch.__version__, 'torch_threads':2,
            'device':'cpu', 'dtype':'float64', 'physical_rows':args.rows,
            'seed':83, 'physical_input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'max_memory_mb':1024, 'batch_rows':65536,
            'node_count':graph.node_count, 'edge_count':graph.edge_count,
            'graph_import_seconds':imported, 'csr_seconds':times,
            'csr_median_seconds':{key:statistics.median(value) for key,value in times.items()},
            'new_csr_owned_storage_bytes':owned, 'original_graph_storage_reused':True,
            'peak_process_rss_bytes':rss if sys.platform=='darwin' else rss*1024,
            'memory_scope':'new CSR numeric owned storage; graph resident separately; RSS includes both benchmarks/runtime/input creation',
            'comparison':'same canonical directed forward graph; first setup warmup excluded',
            'baseline_source_commit':old_ref, 'baseline_source_sha256':hashlib.sha256(original).hexdigest(),
            'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                [path,'src/openecon/networks.py','benchmarks/network_csr.py','benchmarks/network_advanced.py']},
            'out_of_core_graph':False, 'universal_speed_guarantee':False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'csr_median_seconds':report['csr_median_seconds'],
                          'new_csr_owned_storage_bytes':owned, 'import_seconds':imported}))


if __name__ == '__main__':
    main()
