"""Physical count-network fits, explicit pair prediction and bounded charts.

Each method/direction has one measured call and no warmups. The four supplied
blocks are a starting partition; this measures profile fitting and independent
readback, not random-start recovery, global optimality or maximum supported size.
Full sparse snapshots remain resident; import batches do not make fitting
out-of-core. Startup is excluded and whole-process peak RSS is reported.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
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
    "src/openecon/__init__.py", "src/openecon/networks.py",
    "src/openecon/_network_sparse.py", "src/openecon/_network_communities.py",
    "src/openecon/_network_sbm.py", "src/openecon/_network_poisson.py",
    "src/openecon/_network_dc_sbm.py", "src/openecon/_network_block_prediction.py",
    "src/openecon/dataset.py", "src/openecon/frame.py",
    "packages/openecon-charts/src/openecon_charts/charts.py",
    "packages/openecon-charts/src/openecon_charts/network.py",
    "benchmarks/network_count_models_scale.py", "benchmarks/network_workflow_scale.py",
)


def fingerprint(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def records(nodes, groups):
    """Disconnected count-valued band blocks with varying strengths and loops."""
    width = nodes // groups
    for source in range(nodes):
        base = source // width * width
        for step in range(1, 9):
            target = base + (source - base + step) % width
            yield source, target, 1 + (source * 7 + step * 3) % 11
        if source % 57 == 0:
            yield source, source, 1 + source % 5


def write_csv(path, rows):
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "target", "weight"])
        for row in rows:
            writer.writerow(row)
            count += 1
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                physical_rows=count, format="csv")


def verify_fit(fit, fixture_records, nodes, directed):
    """Independent count/degree/profile algebra without fitter private helpers.

    Large-case full likelihood is evaluated directly from observed c log(lambda)
    minus log(c!) and all-dyad expected exposure. Exact rational block/node
    normalization verifies every fitted expected in/out/stub degree without
    allocating an N-by-N adjacency. Tiny full-dyad oracles live in model tests.
    """
    corrected = fit["metadata"]["model"] == "degree_corrected_poisson"
    membership = dict(zip(fit["membership"].node, fit["membership"].block))
    assert set(membership) == set(range(nodes))
    groups = fit["metadata"]["groups"]
    sizes = [0] * groups
    degree_out, degree_in = [0] * nodes, [0] * nodes
    observed, block_counts = {}, {}
    for node, block in membership.items():
        sizes[block] += 1
    for source, target, count in fixture_records:
        if source == target and not corrected:
            continue
        if not directed and source > target:
            source, target = target, source
        pair = source, target
        observed[pair] = observed.get(pair, 0) + count
    for (source, target), count in observed.items():
        degree_out[source] += count
        if directed:
            degree_in[target] += count
        else:
            degree_out[target] += count  # A raw self-loop contributes two stubs.
        a, b = membership[source], membership[target]
        if not directed and a > b:
            a, b = b, a
        block_counts[a, b] = block_counts.get((a, b), 0) + count
    stubs_out, stubs_in = [0] * groups, [0] * groups
    for node, block in membership.items():
        stubs_out[block] += degree_out[node]
        stubs_in[block] += degree_in[node]
    mixing = {}
    for row in fit["blocks"].itertuples(index=False):
        a, b = row.source_block, row.target_block
        count = block_counts.get((a, b), 0)
        if corrected:
            multiplier = 2 if not directed and a == b else 1
            mixing[a, b] = count * multiplier
            assert row.count_strength == count and row.omega == mixing[a, b]
            assert row.expected_count == count
            assert row.stubs_source == stubs_out[a]
            assert row.stubs_target == (stubs_in if directed else stubs_out)[b]
        else:
            dyads = sizes[a] * sizes[b] if a != b else sizes[a] * (sizes[a] - 1)
            if not directed and a == b:
                dyads //= 2
            assert row.edges == count and row.dyads == dyads
            mixing[a, b] = Fraction(count, dyads) if dyads else Fraction(0)
            if dyads:
                assert row.mean_count == float(mixing[a, b])
    def expected(source, target):
        a, b = membership[source], membership[target]
        cell = (a, b) if directed or a <= b else (b, a)
        if not corrected:
            return mixing[cell]
        numerator = degree_out[source] * (degree_in if directed else degree_out)[target] * mixing[cell]
        denominator = stubs_out[a] * (stubs_in if directed else stubs_out)[b]
        if not directed and source == target:
            denominator *= 2
        return Fraction(numerator, denominator) if denominator else Fraction(0)
    total = sum(observed.values())
    actual_ll = math.fsum([count * math.log(float(expected(source, target))) - math.lgamma(count + 1)
                         for (source, target), count in observed.items()]) - total
    ll_error = abs(actual_ll - fit["metadata"]["log_likelihood"])
    assert ll_error < 5e-6
    if corrected:
        # Independently check each group mixing row/column and every exported
        # node theta. This establishes all-node exact expected degree equality.
        for a in range(groups):
            row_sum = sum(value for (x, y), value in mixing.items() if x == a or (not directed and y == a))
            assert row_sum == stubs_out[a]
            if directed:
                assert sum(value for (x, y), value in mixing.items() if y == a) == stubs_in[a]
        for row in fit["membership"].itertuples(index=False):
            node, block = row.node, row.block
            if directed:
                assert row.theta_out == degree_out[node] / stubs_out[block]
                assert row.theta_in == degree_in[node] / stubs_in[block]
            else:
                assert row.theta == degree_out[node] / stubs_out[block]
    return dict(full_log_likelihood_independent=actual_ll, absolute_likelihood_error=ll_error,
                count_strength=total, degree_range_out=[min(degree_out), max(degree_out)],
                degree_range_in=[min(degree_in), max(degree_in)] if directed else None,
                all_node_expected_degrees_checked=corrected), expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nodes < 128 or args.nodes % 4:
        parser.error("Use at least 128 nodes in four equal blocks.")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    hashes = fingerprint(root)
    phases, receipts = [], {}
    max_memory_mb, max_work, groups = 1024, 2_000_000_000, 4
    def timed(name, callback):
        start = time.perf_counter()
        result = callback()
        phases.append(dict(name=name, seconds=time.perf_counter() - start))
        return result
    with tempfile.TemporaryDirectory(prefix="openecon-count-network-") as owned:
        temporary = Path(owned)
        paths = {False: temporary / "undirected-counts.parquet", True: temporary / "directed-counts.csv"}
        physical = {
            False: timed("create_undirected_parquet", lambda: write_records(paths[False],
                records(args.nodes, groups), batch_rows=4096)),
            True: timed("create_directed_csv", lambda: write_csv(paths[True], records(args.nodes, groups))),
        }
        for directed in (False, True):
            direction = "directed" if directed else "undirected"
            graph = timed(f"import_{direction}", lambda: oe.network(oe.scan(paths[directed]), weight="weight",
                directed=directed, nodes=range(args.nodes), batch_rows=4096, max_memory_mb=max_memory_mb))
            for corrected in (False, True):
                model = "degree_corrected_poisson" if corrected else "poisson"
                name = f"{model}_{direction}"
                method = graph.degree_corrected_block_model if corrected else graph.poisson_block_model
                fit = timed(f"fit_{name}", lambda method=method: method(groups,
                    initial={node: node // (args.nodes // groups) for node in range(args.nodes)},
                    starts=1, seed=42, max_iter=10, max_work=max_work))
                verification, expected = timed(f"verify_{name}", lambda fit=fit: verify_fit(fit,
                    records(args.nodes, groups), args.nodes, directed))
                candidates = []
                for index in range(1000):
                    source = index * 17 % args.nodes
                    target = source if corrected and index % 3 == 0 else (source + 1 + index % 97) % args.nodes
                    candidates.append((source, target))
                prediction = timed(f"expected_edges_{name}", lambda fit=fit: fit.expected_edges(candidates,
                    max_pairs=1000, max_memory_mb=max_memory_mb))
                prediction_error = 0.
                for row, pair in zip(prediction.itertuples(index=False), candidates):
                    assert (row.source, row.target) == pair and row.identified
                    mean = float(expected(*pair))
                    prediction_error = max(prediction_error, abs(mean - row.mean_count))
                    assert math.isclose(row.mean_count, mean, rel_tol=3e-15, abs_tol=1e-15)
                    assert math.isclose(row.presence_probability, -math.expm1(-mean), rel_tol=3e-15, abs_tol=1e-15)
                chart = timed(f"chart_{name}", lambda graph=graph, fit=fit: oe.plot.network(graph, groups=fit["membership"],
                    max_nodes=1000, max_edges=5000, title=name))
                serialized = timed(f"serialize_chart_{name}", lambda chart=chart: json.dumps(chart.model_dump(), allow_nan=False))
                payload = chart.model_dump()["config"]["network"]
                assert payload["node_count"] == args.nodes and payload["edge_count"] == graph.edge_count
                assert payload["shown_node_count"] <= 1000 and payload["shown_edge_count"] <= 5000
                assert payload["sampled"] == (payload["shown_node_count"] < args.nodes
                                               or payload["shown_edge_count"] < graph.edge_count)
                assert not fit["metadata"]["sampled"]
                if args.nodes > 1000:
                    assert payload["sampled"]
                receipts[name] = dict(physical=physical[directed], nodes=args.nodes, stored_dyads=graph.edge_count,
                    metadata=fit["metadata"], block_parameters=json.loads(fit["blocks"].to_json(orient="records")),
                    independent_verification=verification, selected_prediction_pairs=len(prediction),
                    selected_prediction_max_absolute_error=prediction_error,
                    chart=dict(shown_nodes=payload["shown_node_count"], shown_edges=payload["shown_edge_count"],
                        sampled=payload["sampled"], grouping=payload["grouping"],
                        finite_json_bytes=len(serialized.encode("utf-8"))),
                    verification_scope="Supplied four-block initial partition; independent full-Poisson likelihood and count algebra; "
                        "exact relevant expected-degree identities for DC fit; 1,000 explicit pair rates/presence; "
                        "bounded real chart JSON, without browser-render or random-start recovery claim")
                del fit, prediction, chart, expected
                gc.collect()
            del graph, method
            gc.collect()
    assert hashes == fingerprint(root), "Sources changed during measurement; receipt is not valid."
    report = dict(status="passed", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=cpu_name(), torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(), max_memory_mb_per_graph=max_memory_mb,
        max_work_per_fit=max_work, starts=1, max_iter=10, warmups=0, measured_repetitions=1,
        source_sha256=hashes, phases=phases, fixtures=receipts,
        timing_scope="fixture creation, import, model fitting, verification, pair prediction and chart serialization separate; Python/Torch startup excluded",
        memory_scope="Sparse fit snapshots resident; import reads physical file batches; not out-of-core model fitting",
        rss_scope="whole process, libraries, fixture creation/import and every measured operation",
        process_peak_rss_bytes=peak_rss())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", phases=phases,
        process_peak_rss_mib=report["process_peak_rss_bytes"] / 1024**2)))


if __name__ == "__main__":
    main()
