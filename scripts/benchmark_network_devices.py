"""Fresh-process CPU/CUDA sparse-kernel comparison; missing CUDA stays missing.

Uses a weighted circulant with known uniform centralities. This measures one
regular workload, not general convergence behavior or an automatic speedup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

METHODS = ('degree', 'pagerank', 'eigenvector', 'hits', 'katz', 'modularity')
ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    import numpy as np
    import torch
    import openecon as oe
    from openecon._network_device import resolve

    torch.set_num_threads(1)
    device = resolve(args.device)
    n, offsets = args.nodes, 8
    start = time.perf_counter()
    g = oe.network((dict(source=u, target=(u + j) % n, weight=float(j + 1))
                    for u in range(n) for j in range(offsets)), nodes=range(n),
                   weight='weight', directed=True, max_memory_mb=1024)
    build_seconds = time.perf_counter() - start
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def run(selected):
        if args.worker == 'modularity':
            return g.modularity([0] * n, device=selected, max_work=2_000_000_000)
        options = dict(device=selected)
        if args.worker in ('hits', 'katz'):
            options['max_work'] = 2_000_000_000
        return getattr(g, args.worker)(**options)

    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    start = time.perf_counter()
    result = run(str(device))
    synchronize()
    cold = time.perf_counter() - start
    timings = []
    for _ in range(3):
        synchronize()
        start = time.perf_counter()
        result = run(str(device))
        synchronize()
        timings.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    # Isolate sparse input transfer and return transfer in synchronized timings.
    synchronize()
    start = time.perf_counter()
    pairs = g._arcs._indices().to(device)
    weights = g._arcs._values().to(device)
    synchronize()
    upload_seconds = time.perf_counter() - start
    start = time.perf_counter()
    returned = weights.cpu()
    synchronize()
    download_seconds = time.perf_counter() - start
    assert returned.numel() == weights.numel() and pairs.shape[1] == n * offsets
    del pairs, weights, returned
    cpu = run('cpu')
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        run(str(device))
        synchronize()
    operators = sorted(profile.key_averages(), key=lambda event: event.cpu_time_total, reverse=True)[:12]
    operator_profile = [dict(operator=event.key, calls=event.count,
                             cpu_total_microseconds=event.cpu_time_total,
                             device_total_microseconds=event.device_time_total) for event in operators]
    if args.worker == 'modularity':
        difference = abs(result - cpu)
        assert abs(result) < 1e-12
        metadata = dict(device=str(device), dtype='float64')
    else:
        assert result.node.tolist() == list(range(n))
        difference = float(np.max(np.abs(result.iloc[:, 1:].to_numpy() - cpu.iloc[:, 1:].to_numpy())))
        expected = {'degree': None, 'pagerank': 1 / n, 'eigenvector': 1 / n**.5,
                    'hits': 1 / n, 'katz': 1 / n**.5}[args.worker]
        if expected is None:
            assert result.in_degree.tolist() == result.out_degree.tolist() == [offsets] * n
            assert result.in_strength.tolist() == result.out_strength.tolist() == [36.] * n
        else:
            np.testing.assert_allclose(result.iloc[:, 1:].to_numpy(), expected, atol=2e-10, rtol=0)
        metadata = result.attrs
    assert difference < 2e-10
    driver = None
    if device.type == 'cuda':
        # Record driver evidence separately; CUDA toolkit version is not driver version.
        process = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15)
        driver = dict(returncode=process.returncode, inventory=process.stdout.strip())
        if process.returncode:
            raise RuntimeError('Real CUDA receipt requires NVIDIA driver inventory')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(method=args.worker, device=str(device), nodes=n, edges=n * offsets, build_seconds=build_seconds,
                cold_seconds=cold, warm_seconds=timings, sparse_upload_seconds=upload_seconds,
                edge_value_download_seconds=download_seconds, cpu_gpu_max_absolute_difference=difference,
                known_circulant_formula_verified=True, metadata=metadata,
                operator_profile=operator_profile,
                gpu_baseline_allocated_bytes=baseline_allocated, gpu_peak_allocated_bytes=peak,
                gpu_peak_reserved_bytes=reserved, driver=driver,
                device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                torch=torch.__version__, cuda_build=torch.version.cuda, threads=1,
                rss_peak_bytes=int(rss if sys.platform == 'darwin' else rss * 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, nargs='+', default=[5000, 20000])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--worker', choices=METHODS)
    args = parser.parse_args()
    if min(args.nodes) < 8:
        parser.error('Use at least eight nodes')
    if args.worker:
        args.nodes = args.nodes[0]
        print(json.dumps(worker(args), allow_nan=False))
        return
    import torch
    available = torch.cuda.is_available()
    if args.require_cuda and not available:
        raise SystemExit('Real CUDA is unavailable; no CPU substitute receipt was written.')
    devices = ['cpu'] + (['cuda'] if available else [])
    runs = []
    for nodes in args.nodes:
        for method in METHODS:
            for device in devices:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', method,
                    '--nodes', str(nodes), '--device', device, '--output', str(args.output)],
                    capture_output=True, text=True, timeout=300, check=True)
                record = json.loads(process.stdout)
                runs.append(record)
                print(json.dumps({k: record[k] for k in ('method', 'nodes', 'device', 'cold_seconds')}), flush=True)
    files = ['src/openecon/' + name for name in ('_network_device.py', 'networks.py',
             '_network_centrality.py', '_network_spectral.py', '_network_communities.py')]
    receipt = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                   python=platform.python_version(), cuda_verified=available, runs=runs,
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
                   scope='Fresh processes, weighted regular circulant only; no general speedup or installed-app claim',
                   cold_scope='First method call after graph creation and device inspection; fresh process per method',
                   warm_scope='Three subsequent complete calls including upload and CPU result conversion',
                   transfer_scope='Separate full sparse index/value upload and edge-value return microbenchmark',
                   memory_scope='Allocator peaks exclude other processes/driver; RSS includes runtime and graph')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
