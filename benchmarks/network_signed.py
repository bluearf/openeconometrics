"""Physical signed graph fixtures; each case runs in a fresh process.

Large strengths/Katz have an independent uniform closed-form solution. Layered
negative-cost DAG paths have exact potential differences. Communities use four
positive cliques with negative cross-group matching; no global optimum claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

import openecon as oe
from openecon import _network_signed


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(case, directory):
    torch.set_num_threads(2)
    phases = {}

    def measured(name, function):
        start = time.perf_counter()
        value = function()
        phases[name] = time.perf_counter() - start
        return value

    if case == "strength_katz":
        n = 100_000
        source = np.repeat(np.arange(n, dtype=np.int64), 10)
        offset = np.tile(np.arange(1, 11, dtype=np.int64), n)
        target = (source + offset) % n
        weight = np.where(offset <= 5, 2., -1.)
        directed = True
    elif case == "communities":
        n = 1_000
        left, right = np.triu_indices(n//4, k=1)
        source = np.concatenate([left + group * (n//4) for group in range(4)] + [np.arange(n)])
        target = np.concatenate([right + group * (n//4) for group in range(4)] + [(np.arange(n) + n//4) % n])
        weight = np.concatenate([np.full(len(left)*4, 2.), np.full(n, -1.)])
        directed = False
    elif case == "negative_dag_paths":
        n, width = 10_001, 1_000
        source = np.concatenate([np.zeros(width, dtype=np.int64)] + [
            np.repeat(np.arange(1 + layer*width, 1+(layer+1)*width, dtype=np.int64), 4)
            for layer in range(9)])
        target = np.concatenate([np.arange(1, width+1, dtype=np.int64)] + [
            1 + (layer+1)*width + (np.repeat(np.arange(width), 4) + np.tile(np.arange(4), width)) % width
            for layer in range(9)])
        potential = np.concatenate([[0]] + [
            (3. if layer % 2 == 0 else -3.) + np.tile(np.array([-2., 3., 1., -4., 5.]), 200)
            for layer in range(10)])
        weight = potential[target] - potential[source]
        # Alternating layer potentials keep every offset-0 arc nonzero.
        # Positive slack on other offsets preserves the exact potential proof.
        weight[source != 0] += np.tile([0., .25, .5, .75], 9_000)
        directed = True
    else:
        raise ValueError(case)
    frame = pd.DataFrame({"source": source, "target": target, "weight": weight})
    path = directory / (case + ".parquet")
    measured("physical_input_write", lambda frame=frame: frame.to_parquet(path, index=False))
    del source, target, weight, frame
    physical_digest = digest(path)
    graph = measured("bounded_import", lambda: oe.signed_network(oe.scan(path), directed=directed,
        nodes=range(n), batch_rows=65_536, max_memory_mb=512))
    result_details = {}
    if case == "strength_katz":
        strengths = measured("signed_strength", graph.signed_strength)
        for column, expected in (("positive_out_strength", 10), ("negative_out_strength", 5),
                                 ("net_out_strength", 5), ("absolute_out_strength", 15),
                                 ("positive_in_strength", 10), ("negative_in_strength", 5)):
            assert bool((strengths[column] == expected).all())
        katz = measured("signed_katz", lambda: graph.signed_katz(1/30, tol=1e-12))
        error = float(np.max(np.abs(katz.signed_katz.to_numpy() - 1.2)))
        assert error < 3e-12
        result_details = dict(katz_max_error_vs_closed_form=error,
            katz_expected=1.2, iterations=katz.attrs["iterations"], residual=katz.attrs["residual_linf"],
            work_used=katz.attrs["work_used"], max_work=katz.attrs["max_work"])
    elif case == "communities":
        groups = measured("signed_communities", graph.signed_communities)
        modularity = measured("signed_modularity", lambda: graph.signed_modularity(groups))
        assert len(groups) == n and groups.community.nunique() == 4
        for group in range(4):
            assert groups.community.iloc[group*250:(group+1)*250].nunique() == 1
        assert abs(modularity - groups.attrs["signed_modularity"]) < 1e-13
        result_details = dict(communities=4, modularity=modularity, sweeps=groups.attrs["sweeps"],
            objective_trace=groups.attrs["objective_trace"], work_used=groups.attrs["work_used"],
            max_work=groups.attrs["max_work"], optimality=groups.attrs["optimality"])
    else:
        distances = measured("signed_shortest_paths", lambda: graph.signed_shortest_paths(0))
        expected = potential
        error = float(np.max(np.abs(distances.distance.to_numpy() - expected)))
        assert error == 0 and distances.reachable.all() and len(distances) == n
        result_details = dict(max_error_vs_potential=error, iterations=distances.attrs["iterations"],
            reachable_nodes=n, work_used=distances.attrs["work_used"], max_work=distances.attrs["max_work"])
    assert digest(path) == physical_digest
    return dict(status="passed", case=case, node_count=n, edge_count=graph.edge_count,
        physical_input_rows=graph.metadata["input_rows"], physical_parquet_sha256=physical_digest,
        phases_seconds=phases, result=result_details, network=graph.metadata,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024),
        process_rss_scope="fresh process lifetime high-water mark; imports, physical input generation, Torch, tables and checks included",
        platform=platform.platform(), architecture=platform.machine(), python=platform.python_version(),
        torch=str(torch.__version__), torch_threads=torch.get_num_threads(), seed=None,
        fixture_randomness="none; deterministic analytic fixtures", source_sha256=digest(_network_signed.__file__),
        benchmark_sha256=digest(__file__), full_graph_analysis=True, sampled=False,
        disk_native=False, device="cpu", dtype="float64")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["strength_katz", "communities", "negative_dag_paths"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    with tempfile.TemporaryDirectory(prefix="openecon-signed-scale-") as temporary:
        result = run(args.case, Path(temporary))
    result["owned_physical_fixture_removed"] = not Path(temporary).exists()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "case", "node_count", "edge_count", "phases_seconds")}))


if __name__ == "__main__":
    main()
