"""Advanced, synthetic international production-network laboratory.

Open this file in the OpenEconometrics code panel and press Run. No installation
or files are required. All calculations use the native sparse network SDK and
Torch; pandas-backed oe.DataFrame is used only for labelled result tables.

The defaults simulate 2,400 firms, 12 sectors, 8 countries and 8 monthly directed
count networks, with a matched no-shock counterfactual and a firm/skill layer.
This is a methodological demonstration, not estimated economic damage.

Change DEFAULT_CONFIG, or call run_lab({...}, display_callback=display). The
panel's whole-file entry also accepts an explicit NETWORK_LAB_CONFIG dictionary.
Networks are resident sparse graphs, not out-of-core or analytical GPU jobs.
"""
from __future__ import annotations

import hashlib
import json
import math
from time import perf_counter

import torch
import openecon as oe


DEFAULT_CONFIG = dict(
    firms=2400, months=8, groups=6, seed=731,
    mrqap_nodes=120, permutations=199, betweenness_samples=32,
    link_candidates=6000, block_starts=2, block_iterations=100, skills=96,
)
NAVY = "#14263d"
BLUE = "#4c78a8"
PALETTE = [NAVY, "#244c74", "#356995", BLUE, "#7ca2c4", "#aec9df"]
SECTORS = ["Energy", "Logistics", "Chemicals", "Electronics", "Machinery", "Metals",
           "Food", "Transport", "Textiles", "Construction", "Finance", "Services"]
COUNTRIES = ["US", "China", "Turkey", "Germany", "Japan", "Brazil", "India", "South Africa"]
CENTRES = [(-100, 38), (104, 35), (29, 39), (10, 51), (139, 36),
           (-52, -14), (78, 22), (25, -29)]
MEMORY_MB = 512
WORK = 300_000_000


def _rng(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _digest(pairs):
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def _config(override):
    c = {**DEFAULT_CONFIG, **(override or {})}
    unknown = set(c) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration: {sorted(unknown)}")
    for key, value in c.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["firms"] < 72 or c["firms"] > 5000 or c["months"] != 8 or c["groups"] != 6:
        raise ValueError("This design requires 72–5,000 firms, exactly 8 months and 6 latent groups")
    if c["mrqap_nodes"] > c["firms"] or c["mrqap_nodes"] > 180 or c["mrqap_nodes"] % 12:
        raise ValueError("mrqap_nodes must be a multiple of 12, <= firms and <= 180")
    if not 12 <= c["skills"] <= 120 or c["skills"] % 6:
        raise ValueError("skills must be a multiple of 6 between 12 and 120")
    if c["betweenness_samples"] > c["firms"] or c["link_candidates"] > 10000:
        raise ValueError("Betweenness sources must fit the node universe; candidates <= 10,000")
    return c


def _graph(labels, u, v, weights=None, *, directed=True):
    rows = {"source": [labels[i] for i in u.tolist()],
            "target": [labels[i] for i in v.tolist()]}
    if weights is not None:
        rows["weight"] = weights.tolist()
    return oe.network(rows, nodes=labels, directed=directed,
                      weight="weight" if weights is not None else None,
                      batch_rows=4096, max_memory_mb=MEMORY_MB)


def _edge_map(graph):
    return {(u, v): float(w) for u, v, w in
            graph.edges()[["source", "target", "weight"]].itertuples(index=False, name=None)}


def _count_scores(observed, predicted, identified):
    """Score frozen monthly forecasts on one explicitly shared dyad mask.

    A positive count assigned zero mean has infinite negative log likelihood.
    Expose its count and a missing score instead of clipping the mean or putting
    nonfinite numbers in saved JSON. Empty subsets also keep undefined scores.
    """
    actual, mean = observed[identified], predicted[identified]
    positive = actual > 0
    error = (mean - actual).abs()
    impossible = positive & (mean == 0)
    probability = -torch.expm1(-mean)
    nll = None
    if len(actual) and not bool(impossible.any()):
        terms = mean + torch.lgamma(actual + 1)
        terms[positive] -= actual[positive] * torch.log(mean[positive])
        nll = float(terms.mean())
    return dict(evaluated_pairs=len(actual), exposure_months=1,
        positive_pairs=int(positive.sum()), zero_pairs=int((~positive).sum()),
        observed_count=float(actual.sum()), predicted_count=float(mean.sum()),
        count_bias=float((mean - actual).mean()) if len(actual) else None,
        mae=float(error.mean()) if len(actual) else None,
        positive_mae=float(error[positive].mean()) if bool(positive.any()) else None,
        zero_mae=float(error[~positive].mean()) if bool((~positive).any()) else None,
        presence_brier=float((probability - positive.to(torch.float64)).square().mean()) if len(actual) else None,
        mean_poisson_nll=nll, impossible_positive_pairs=int(impossible.sum()))


def _binary(graph, *, directed=False):
    edges = graph.edges()
    return oe.network(edges[["source", "target"]], nodes=graph.nodes().node.tolist(),
                      directed=directed, max_memory_mb=MEMORY_MB)


def _uniform_pairs(labels, count, seed, excluded=(), *, directed=False):
    """Draw unique pairs without looking at any held-out outcomes."""
    excluded, chosen = set(excluded), set()
    available = len(labels) * (len(labels) - 1) // (1 if directed else 2) - len(excluded)
    count = min(count, available)
    generator = _rng(seed)
    while len(chosen) < count:
        draw = torch.randint(len(labels), (max(64, count - len(chosen)), 2), generator=generator)
        for a, b in draw.tolist():
            if a == b:
                continue
            if not directed and a > b:
                a, b = b, a
            pair = (labels[a], labels[b])
            if pair not in excluded:
                chosen.add(pair)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _ari(first, second):
    """Adjusted Rand index from a small contingency table; no sklearn."""
    n = len(first)
    a = {x: i for i, x in enumerate(sorted(set(first)))}
    b = {x: i for i, x in enumerate(sorted(set(second)))}
    codes = torch.tensor([a[x] * len(b) + b[y] for x, y in zip(first, second)])
    cells = torch.bincount(codes, minlength=len(a) * len(b)).reshape(len(a), len(b))
    def choose(x):
        return float((x.to(torch.float64) * (x - 1) / 2).sum())
    total, rows, columns = n * (n - 1) / 2, choose(cells.sum(1)), choose(cells.sum(0))
    expected = rows * columns / total
    denominator = (rows + columns) / 2 - expected
    return (choose(cells) - expected) / denominator if denominator else 1.


def _ranking_metrics(scores, outcomes, k=100):
    """Threshold-group AP and tie-aware AUC; deterministic top-k tie ordering."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    hits = sum(outcomes[i] for i in order[:min(k, len(order))])
    tp, fp, ap, auc_numerator = 0, 0, 0., 0.
    for _, indices in _score_groups(scores, order):
        p = sum(outcomes[i] for i in indices)
        q = len(indices) - p
        tp += p
        fp += q
        if positives:
            ap += p / positives * tp / (tp + fp)
        auc_numerator += p * (negatives - fp + q / 2)
    return dict(candidates=len(scores), positives=positives,
                prevalence=positives / len(scores) if scores else None,
                precision_at_100=hits / min(k, len(order)) if order else None,
                average_precision=ap if positives else None,
                roc_auc=auc_numerator / (positives * negatives) if positives and negatives else None)


def _score_groups(scores, order):
    current, indices = None, []
    for i in order:
        if indices and scores[i] != current:
            yield current, indices
            indices = []
        current = scores[i]
        indices.append(i)
    if indices:
        yield current, indices


def _simulate(c):
    n, generator = c["firms"], _rng(c["seed"])
    labels = [f"F{i:04d}" for i in range(n)]
    sectors = torch.arange(n) % 12
    blocks = sectors % 6
    noisy = torch.rand(n, generator=generator) < .12
    blocks[noisy] = torch.randint(6, (int(noisy.sum()),), generator=generator)
    countries = torch.randint(8, (n,), generator=generator)
    activity = torch.exp(.65 * torch.randn(n, generator=generator)).clamp(.25, 6)
    longitude = torch.tensor([CENTRES[i][0] for i in countries.tolist()], dtype=torch.float64)
    latitude = torch.tensor([CENTRES[i][1] for i in countries.tolist()], dtype=torch.float64)
    longitude += torch.randn(n, generator=generator, dtype=torch.float64) * 2
    latitude += torch.randn(n, generator=generator, dtype=torch.float64) * 1.5
    hubs = torch.argsort(activity, descending=True)[:max(6, n // 80)]
    # Sparse candidate construction; no firm-by-firm dense matrix.
    senders, receivers = [], []
    for i in range(n):
        for mask, count in ((blocks == blocks[i], 8), (sectors == sectors[i], 4),
                            (countries == countries[i], 3)):
            pool = torch.where(mask)[0]
            picks = pool[torch.randint(len(pool), (count,), generator=generator)]
            senders.extend([i] * count)
            receivers.extend(picks.tolist())
        senders.extend([i] * 3)
        receivers.extend(torch.randint(n, (2,), generator=generator).tolist())
        receivers.append(int(hubs[torch.randint(len(hubs), (1,), generator=generator)]))
    base_u, base_v = torch.tensor(senders), torch.tensor(receivers)
    actual, control, monthly_rows = {}, {}, []
    # Equirectangular coordinates are calculated once, not relaid out per frame.
    positions = {node: {"x": 3 * float(longitude[i]), "y": -3 * float(latitude[i])}
                 for i, node in enumerate(labels)}
    shock_nodes = ((sectors % 6 == 1) | (sectors % 6 == 3)) & ((countries == 1) | (countries == 4))
    if not bool(shock_nodes.any()):
        shock_nodes[hubs[:2]] = True
    # Include a contact backbone for a known, nontrivial time-respecting route.
    for month in range(8):
        name = f"2026-{month + 1:02d}"
        fresh_u = torch.arange(n).repeat_interleave(2)
        fresh_v = torch.randint(n, (2 * n,), generator=generator)
        u, v = torch.cat([base_u, fresh_u]), torch.cat([base_v, fresh_v])
        keep = u != v
        u, v = u[keep], v[keep]
        same_block, same_country = blocks[u] == blocks[v], countries[u] == countries[v]
        rate = (.35 + 1.2 * same_block + .35 * same_country) * torch.sqrt(activity[u] * activity[v])
        rate *= 1 + .025 * month + .08 * math.sin(month * math.pi / 4)
        counts = torch.poisson(rate.to(torch.float64), generator=generator)
        corridor = ((countries[u] == 1) & (countries[v] == 0)) | ((countries[u] == 0) & (countries[v] == 1))
        impacted = shock_nodes[u] | shock_nodes[v] | corridor
        severity = [0, 0, 0, 0, .8, .8, .45, .15][month]
        survival = torch.where(impacted, 1 - severity, 1.).to(torch.float64)
        thinned = torch.binomial(counts, survival, generator=generator)
        # The backbone exists in both matched scenarios, one successive hop/month.
        u = torch.cat([u, torch.tensor([month])])
        v = torch.cat([v, torch.tensor([month + 1])])
        counts, thinned = torch.cat([counts, torch.ones(1)]), torch.cat([thinned, torch.ones(1)])
        base = _graph(labels, u, v, counts)
        observed = _graph(labels, u, v, thinned)
        attrs = {node: dict(label=node, sector=SECTORS[int(sectors[i])], country=COUNTRIES[int(countries[i])],
                           ecosystem=int(blocks[i]), activity=float(activity[i]),
                           longitude=float(longitude[i]), latitude=float(latitude[i]),
                           shock_exposure=float(severity if bool(shock_nodes[i]) else 0))
                 for i, node in enumerate(labels)}
        actual[name] = observed.with_attributes(nodes=attrs).with_positions(positions, fixed=True)
        control[name] = base
        monthly_rows.append(dict(month=name, nodes=n, actual_edges=observed.edge_count,
                                 control_edges=base.edge_count, actual_count=int(thinned.sum()),
                                 control_count=int(counts.sum()), retention=float(thinned.sum() / counts.sum()),
                                 shock_intensity=severity))
    return dict(labels=labels, sectors=sectors, planted=blocks, countries=countries,
                activity=activity, longitude=longitude, latitude=latitude,
                actual=actual, control=control, monthly=oe.DataFrame(monthly_rows))


def run_lab(config=None, *, display_callback=None):
    """Run the full laboratory, returning all tables, charts and validation proof."""
    c, started = _config(config), perf_counter()
    tables, charts, timings = {}, {}, {}
    proof = {}

    def stage(name):
        print(name, flush=True)
        return perf_counter()

    def table(name, frame, title):
        attributes = dict(getattr(frame, "attrs", {}))
        frame = oe.DataFrame(frame)
        frame.attrs.update(attributes)
        frame.attrs.update(title=title, caption=title, synthetic=True)
        tables[name] = frame
        if display_callback is not None:
            display_callback(frame)
        return frame

    def chart(name, spec):
        charts[name] = spec
        if display_callback is not None:
            display_callback(spec)
        return spec

    tick = stage("1/7 · Directed monthly counts and matched shock scenario")
    simulation = _simulate(c)
    labels, months = simulation["labels"], list(simulation["actual"])
    graphs = simulation["actual"]
    series = oe.network_snapshots(graphs, ordered=True, max_work=WORK)
    aggregate = series.aggregate(reducer="sum", max_edges=200000, max_work=WORK)
    table("scenarios", simulation["monthly"], "Matched scenarios · integer counts per month")
    chart("shock_counts", oe.plot.bar(data=simulation["monthly"], x="month",
          y=["control_count", "actual_count"], aggregate=None, palette=[NAVY, BLUE],
          title="Monthly interactions · matched control and shock", unit="Count"))
    timings["simulation"] = perf_counter() - tick

    tick = stage("2/7 · PageRank, sampled brokerage, Leiden and resilience")
    pr = aggregate.pagerank()
    membership = aggregate.communities(method="leiden", seed=c["seed"], max_iter=25, max_work=WORK)
    # Counts are strengths. Brokerage distances deliberately use binary hops.
    topology = _binary(aggregate)
    brokerage = topology.betweenness(samples=c["betweenness_samples"], seed=c["seed"], max_work=WORK)
    core = topology.core_numbers()
    nodes = aggregate.degree().merge(pr, on="node").merge(membership, on="node")
    nodes = nodes.merge(brokerage, on="node").merge(core, on="node")
    node_index = {node: i for i, node in enumerate(labels)}
    nodes["sector"] = [SECTORS[int(simulation["sectors"][node_index[node]])] for node in nodes.node]
    nodes["country"] = [COUNTRIES[int(simulation["countries"][node_index[node]])] for node in nodes.node]
    nodes["planted_group"] = [int(simulation["planted"][node_index[node]]) for node in nodes.node]
    node_attrs = {row.node: dict(pagerank=float(row.pagerank), community=int(row.community),
                               sector=row.sector, country=row.country) for row in nodes.itertuples()}
    aggregate = aggregate.with_attributes(nodes=node_attrs)
    table("hubs", nodes.sort_values("pagerank", ascending=False).head(25),
          f"Top firms · brokerage estimated from {c['betweenness_samples']} source nodes")
    overview = oe.plot.network(aggregate, groups=membership, title="Production system · all firms and links",
        max_nodes=c["firms"], max_edges=200000, seed=c["seed"], layout="forceatlas2",
        layout_options=dict(iterations=160, work_limit=20000000, time_limit_ms=12000,
                            scaling=4, gravity=1, linlog=True),
        width=1100, height=700, palette=PALETTE, opacity=.45,
        node_size=dict(field="attrs.pagerank", scale="sqrt", range=[2, 10]),
        edge_width=dict(field="weight", scale="sqrt", range=[.3, 1.3]),
        labels=dict(show=True, min_zoom=3, max_count=24), legend=True)
    overview = overview.annotate_node("Highest PageRank", node=nodes.sort_values("pagerank", ascending=False).node.iloc[0])
    chart("system", overview)
    assert not overview.config["network"]["sampled"]
    proof["pagerank_sum"] = float(pr.pagerank.sum())
    proof["brokerage"] = dict(samples=c["betweenness_samples"], distance="undirected binary hops", exact=bool(brokerage.attrs["exact"]))
    proof["leiden"] = dict(communities=int(nodes.community.nunique()),
        modularity=float(aggregate.modularity(membership)),
        planted_ari=_ari(nodes.planted_group.tolist(), nodes.community.tolist()))
    order = nodes.sort_values(["pagerank", "node"], ascending=[False, True]).node.tolist()
    random_order = [labels[i] for i in torch.randperm(len(labels), generator=_rng(c["seed"] + 90)).tolist()]
    resilience = []
    for share in [0, .02, .05, .1, .15, .2]:
        row = dict(removed_percent=100 * share)
        count = int(len(labels) * share)
        for name, ranking in (("targeted", order), ("random", random_order)):
            remaining = topology.edit_nodes(remove=ranking[:count])
            comp = remaining.components(connectivity="weak")
            row[name] = float(comp.component.value_counts().max() / len(labels))
        resilience.append(row)
    table("resilience", resilience, "Largest component / original firm count · matched removal fractions")
    chart("resilience", oe.plot.bar(data=oe.DataFrame(resilience), x="removed_percent", y=["targeted", "random"],
          aggregate=None, palette=[NAVY, BLUE], title="Connectivity after targeted and random removal",
          unit="Share of original firms"))
    timings["structure"] = perf_counter() - tick

    tick = stage("3/7 · Temporal turnover, persistence and geographic shock timeline")
    transitions = series.transitions(max_work=WORK)
    persistence = series.edge_persistence(max_edges=200000, max_work=WORK)
    table("turnover", transitions, "Month-to-month directed edge turnover")
    route = series.temporal_path(labels[0], labels[8], start=months[0], max_work=WORK)
    table("temporal_route", route, "Earliest arrival · at most one directed hop per monthly snapshot")
    proof["temporal"] = dict(arrival_layer=route.attrs.get("arrival_layer"),
        reachable=bool(route.attrs["reachable"]), hops=int(route.attrs.get("hops", 0)),
        persistent_edges=int((persistence.present_snapshots == 8).sum()))
    timeline = oe.plot.network(series, title="Geographic shock · fixed coordinates across eight months",
        timeline=True, max_nodes=c["firms"], max_edges=100000,
        layout="fixed", width=1100, height=700, opacity=.35,
        node_size=dict(field="attrs.activity", scale="sqrt", range=[2, 7]),
        node_color=dict(field="attrs.shock_exposure", domain=[0, .8], range=["#aec9df", NAVY]),
        edge_width=dict(field="weight", scale="sqrt", range=[.3, 1.4]),
        labels=dict(show=True, min_zoom=5, max_count=24), palette=PALETTE, legend=True)
    chart("timeline", timeline)
    assert not timeline.config["network"]["sampled"]
    assert all(not frame["network"]["sampled"] for frame in timeline.config["network"]["frames"])
    timings["temporal"] = perf_counter() - tick

    tick = stage("4/7 · Poisson and degree-corrected blocks · next-month holdout")
    train_month, test_month = months[2], months[3]
    fit_graph = graphs[train_month]
    test_pairs = _uniform_pairs(labels, min(3000, c["link_candidates"]), c["seed"] + 30, directed=True)
    train_counts = _edge_map(fit_graph)
    train_offdiagonal_count = sum(count for (source, target), count in train_counts.items() if source != target)
    train_dyads = len(labels) * (len(labels) - 1)
    train_rate = train_offdiagonal_count / train_dyads
    # Same directed non-loop pairs and a one-month exposure for both model families.
    observed_map = _edge_map(graphs[test_month])
    observed = torch.tensor([observed_map.get(pair, 0.) for pair in test_pairs], dtype=torch.float64)
    model_rows, fits, forecasts = [], {}, {}
    for method in ("poisson_block_model", "degree_corrected_block_model"):
        fit = getattr(fit_graph, method)(6, starts=c["block_starts"], max_iter=c["block_iterations"],
                  seed=c["seed"], max_work=WORK)
        fits[method] = fit
        predictions = fit.expected_edges(test_pairs, max_pairs=10000, max_memory_mb=MEMORY_MB)
        predicted = torch.tensor(predictions.mean_count.tolist(), dtype=torch.float64)
        identified = torch.tensor(predictions.identified.tolist())
        forecasts[method] = predicted, identified
    common_identified = forecasts["poisson_block_model"][1] & forecasts["degree_corrected_block_model"][1]
    for method, fit in fits.items():
        predicted, identified = forecasts[method]
        meta = fit["metadata"]
        model_rows.append(dict(model=method, train_month=train_month, test_month=test_month,
            pairs=len(test_pairs), identified_pairs=int(common_identified.sum()),
            model_identified_pairs=int(identified.sum()), converged=bool(meta["converged"]),
            iterations=int(meta["iterations"]), starts=c["block_starts"],
            stopping_reason=meta["stopping_reason"], selected_seed=meta["selected_seed"],
            mae_identified=float((predicted[common_identified] - observed[common_identified]).abs().mean())
                           if bool(common_identified.any()) else None,
            zero_baseline_mae=float(observed[common_identified].abs().mean()) if bool(common_identified.any()) else None))
    table("block_validation", model_rows, "Fixed K=6 local block fits · common next-month dyads, one-month exposure")
    calibration_rows = [dict(model=method, **_count_scores(observed, predicted, common_identified))
                        for method, (predicted, _) in forecasts.items()]
    for name, rate in [("zero_baseline", 0.), ("training_rate_baseline", train_rate)]:
        calibration_rows.append(dict(model=name, **_count_scores(
            observed, torch.full_like(observed, rate), common_identified)))
    table("block_calibration", calibration_rows,
          "Frozen next-month count forecasts · common dyads, positive/zero errors and proper scores")
    blocks = fits["poisson_block_model"]["blocks"]
    wide = []
    for source_block in range(6):
        row = dict(source_block=f"Block {source_block + 1}")
        for target_block in range(6):
            cell = blocks[(blocks.source_block == source_block) & (blocks.target_block == target_block)]
            row[f"to_{target_block + 1}"] = float(cell.mean_count.iloc[0])
        wide.append(row)
    chart("block_intensity", oe.plot.bar(data=oe.DataFrame(wide), x="source_block",
          y=[f"to_{i + 1}" for i in range(6)], aggregate=None, palette=PALETTE,
          title="Poisson blocks · fitted monthly count per directed dyad", unit="Mean count"))
    proof["block_holdout"] = dict(train_month=train_month, test_month=test_month, exposure_months=1,
        pairs_sha256=_digest(test_pairs), pairs=len(test_pairs), future_nonzero=int((observed > 0).sum()),
        known_planted_initialization=False, global_optimum_certified=False,
        likelihoods_compared=False, evaluation_mask="common identified intersection", models=model_rows,
        training_rate_baseline=dict(train_month=train_month, count=train_offdiagonal_count,
                                    dyads=train_dyads, mean_count=train_rate),
        calibration=calibration_rows,
        optimization_traces={method: fit["metadata"]["start_fits"] for method, fit in fits.items()},
        score_scope="conditional Poisson predictions on the same held-out non-loop dyads; not training likelihood comparison",
        nonfinite_score="infinite when a positive count has zero mean; exported as missing with impossible_positive_pairs")
    timings["blocks"] = perf_counter() - tick

    tick = stage("5/7 · Dyadic MRQAP · independent sector-stratified sample")
    generator = _rng(c["seed"] + 40)
    sample = []
    for sector in range(12):
        pool = torch.where(simulation["sectors"] == sector)[0]
        sample.extend(pool[torch.randperm(len(pool), generator=generator)[:c["mrqap_nodes"] // 12]].tolist())
    qlabels = [labels[i] for i in sample]
    def log_counts(graph):
        edges = graph.subgraph(qlabels).edges()
        return oe.network({"source": edges.source.tolist(), "target": edges.target.tolist(),
                           "weight": [math.log1p(w) for w in edges.weight]},
                          weight="weight", nodes=qlabels, directed=True, max_memory_mb=MEMORY_MB)
    qs, qt, proximity, same_sector = [], [], [], []
    for a in sample:
        for b in sample:
            if a == b:
                continue
            # Geographic affinity is a fixed covariate, not an edge capacity.
            lat_a, lat_b = [math.radians(float(simulation["latitude"][i])) for i in (a, b)]
            dlat = lat_b - lat_a
            dlon = math.radians(float(simulation["longitude"][b] - simulation["longitude"][a]))
            haversine = math.sin(dlat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(dlon / 2) ** 2
            distance_km = 6371.0088 * 2 * math.asin(math.sqrt(min(1., max(0., haversine))))
            qs.append(labels[a])
            qt.append(labels[b])
            proximity.append(1 / (1 + distance_km / 1000))
            same_sector.append(int(simulation["sectors"][a] == simulation["sectors"][b]))
    def covariate(values):
        return oe.network(dict(source=qs, target=qt, weight=values), nodes=qlabels,
                          weight="weight", directed=True, max_memory_mb=MEMORY_MB)
    mrqap = log_counts(graphs[months[3]]).qap_regression(
        {"lagged_count": log_counts(graphs[months[2]]), "geographic_affinity": covariate(proximity),
         "same_sector": covariate(same_sector)}, values="weight", permutations=c["permutations"],
        seed=c["seed"], max_work=20_000_000_000)
    table("mrqap", mrqap, "Approximate coefficient-specific Freedman–Lane MRQAP · log(1 + count), sampled firms")
    proof["mrqap"] = dict(nodes=len(qlabels), sample_sha256=_digest(qlabels),
        sample_seed=c["seed"] + 40, sample_selected_without_outcomes=True,
        sample_nodes=qlabels, sample_design="equal-size random within each sector",
        sample_inference_scope="unweighted selected subgraph; not a population-weighted estimate",
        geographic_affinity="1 / (1 + haversine distance_km / 1000)",
        response_month=months[3], lag_month=months[2], permutations=c["permutations"],
        pvalue_resolution=float(mrqap.attrs["pvalue_resolution"]),
        condition_number=float(mrqap.attrs["model_condition_number"]),
        r_squared=float(mrqap.attrs["fit_r_squared"]), work=int(mrqap.attrs["work_used"]),
        inference="approximate conditional Freedman-Lane; residual node exchangeability assumed",
        exchangeability_verified=False, pvalues_exploratory=True,
        multiplicity_adjustment="none", causal=False)
    timings["mrqap"] = perf_counter() - tick

    tick = stage("6/7 · Frozen-candidate future links and certified shock flow")
    training = oe.network_snapshots({m: graphs[m] for m in months[:4]}, ordered=True, max_work=WORK)
    train_union = _binary(training.aggregate(reducer="sum", max_edges=200000, max_work=WORK))
    existing = {tuple(sorted(pair)) for pair in _edge_map(train_union)}
    candidates = _uniform_pairs(labels, c["link_candidates"], c["seed"] + 50, existing)
    # Freeze pairs and scores before any future outcomes are consulted.
    link_scores = {method: train_union.link_prediction(candidates, method=method,
                   max_pairs=10000, max_work=WORK) for method in ("jaccard", "resource_allocation")}
    frozen_candidate_digest = _digest(candidates)
    future = {tuple(sorted(pair)) for m in months[4:] for pair in _edge_map(graphs[m])}
    outcomes = [int(pair in future) for pair in candidates]
    link_rows = [dict(method=method, **_ranking_metrics(frame.score.tolist(), outcomes))
                 for method, frame in link_scores.items()]
    table("link_validation", link_rows, "New weak undirected links · Jan–Apr scores, May–Aug outcomes, uniform nonedges")
    proof["links"] = dict(training_months=months[:4], outcome_months=months[4:],
        candidates_sha256=frozen_candidate_digest, candidates=len(candidates),
        candidate_policy="uniform unique training nonedges, selected before future labels",
        overlap_with_training=len(set(candidates) & existing), future_positives=sum(outcomes),
        scores_ignore_direction_and_counts=True, probabilities_calibrated=False,
        precision_ties="lexicographic pair order", ap_ties="whole score-threshold groups",
        auc_ties="half credit")
    ranked = link_scores["resource_allocation"].copy()
    ranked["appeared_later"] = outcomes
    table("ranked_links", ranked.sort_values(["score", "source", "target"], ascending=[False, True, True]).head(20),
          "Top frozen candidates · resource allocation scores, later outcome shown separately")
    # Super terminals are fixed across matched scenarios and never used as real firms.
    origins = [labels[i] for i in range(len(labels)) if int(simulation["sectors"][i]) in (0, 6)][:32]
    destinations = [labels[i] for i in range(len(labels)) if int(simulation["sectors"][i]) in (3, 9)][:32]
    shock_month = months[4]
    terminal_capacity = float(simulation["monthly"].control_count.iloc[4] + 1)
    flows, flow_rows = {}, []
    for name, graph in (("control", simulation["control"][shock_month]), ("shock", graphs[shock_month])):
        graph = graph.edit_nodes(add=["__SOURCE__", "__SINK__"])
        extra = [dict(source="__SOURCE__", target=node, weight=terminal_capacity) for node in origins]
        extra += [dict(source=node, target="__SINK__", weight=terminal_capacity) for node in destinations]
        result = graph.edit_edges(add=extra).max_flow("__SOURCE__", "__SINK__", max_work=WORK)
        flows[name] = result
        meta = result["metadata"]
        flow_rows.append(dict(scenario=name, month=shock_month, max_flow=float(result["value"]),
            cut_capacity=float(meta["cut_capacity"]), certified=bool(meta["certified"]),
            conservation_error=float(meta["maximum_conservation_error"]), cut_edges=len(result["cut_edges"])))
    assert flows["shock"]["value"] <= flows["control"]["value"] + 1e-8
    assert all(row["certified"] and abs(row["max_flow"] - row["cut_capacity"]) < 1e-8
               and row["conservation_error"] < 1e-8 for row in flow_rows)
    table("flow_certificates", flow_rows, "Maximum-flow/min-cut certificates · count weights used as assumed capacity proxies")
    proof["flow"] = dict(capacity_semantics="monthly interaction count proxy, not economic equilibrium",
        matched_month=shock_month, terminal_capacity=terminal_capacity,
        control=float(flows["control"]["value"]), shock=float(flows["shock"]["value"]),
        monotonic=True, certificates=flow_rows)
    timings["prediction_flow"] = perf_counter() - tick

    tick = stage("7/7 · Bipartite firm/skill projection and final validation")
    skills = [f"Skill-{i:02d}" for i in range(c["skills"])]
    generator = _rng(c["seed"] + 60)
    source, target = [], []
    for i, firm in enumerate(labels):
        own = int(simulation["planted"][i])
        pool = [j for j in range(c["skills"]) if j % 6 == own]
        picks = [pool[j] for j in torch.randint(len(pool), (4,), generator=generator).tolist()]
        picks += torch.randint(c["skills"], (2,), generator=generator).tolist()
        source.extend([firm] * len(set(picks)))
        target.extend(skills[j] for j in sorted(set(picks)))
    two_mode = oe.network(dict(source=source, target=target), nodes=labels + skills, max_memory_mb=MEMORY_MB)
    partition = {**dict.fromkeys(labels, 0), **dict.fromkeys(skills, 1)}
    projection = two_mode.bipartite_projection(partition, onto=1, weight="count", max_work=WORK, max_edges=10000)
    skill_membership = projection.communities(method="leiden", seed=c["seed"], max_work=WORK)
    chart("skill_network", oe.plot.network(projection, groups=skill_membership,
          title="Shared skills · projection onto skills, not the dense firm clique graph",
          max_nodes=c["skills"], max_edges=10000, layout="community", seed=c["seed"],
          layout_options=dict(iterations=100, work_limit=2000000, time_limit_ms=5000, spacing=50, radius=200),
          width=1000, height=650, palette=PALETTE,
          edge_width=dict(field="weight", scale="sqrt", range=[.4, 2]),
          labels=dict(show=True, min_zoom=.8, max_count=120), legend=True))
    proof["bipartite"] = dict(firms=len(labels), skills=len(skills), incidence_edges=two_mode.edge_count,
                              projection_nodes=projection.node_count, projection_edges=projection.edge_count)
    timings["bipartite"] = perf_counter() - tick
    summary_rows = [dict(analysis="Leiden", metric="Modularity", value=proof["leiden"]["modularity"]),
                    dict(analysis="Leiden", metric="ARI vs synthetic planted groups", value=proof["leiden"]["planted_ari"]),
                    dict(analysis="Temporal", metric="Edges present in every month", value=proof["temporal"]["persistent_edges"]),
                    dict(analysis="MRQAP", metric="R-squared on sampled firms", value=proof["mrqap"]["r_squared"]),
                    dict(analysis="Skills", metric="Projected skill links", value=projection.edge_count)]
    table("analysis_summary", summary_rows, "Synthetic laboratory · method and validation summary")
    total_outputs = len(tables) + len(charts)
    assert total_outputs <= 20, "Stay within the panel's displayed-output budget"
    assert abs(proof["pagerank_sum"] - 1) < 1e-8 and proof["links"]["overlap_with_training"] == 0
    metadata = dict(synthetic=True, sdk_version=oe.__version__, config=c, nodes=len(labels),
        months=months, monthly_edges=[graphs[m].edge_count for m in months],
        aggregate_edges=aggregate.edge_count, aggregate_count=int(simulation["monthly"].actual_count.sum()),
        tables=len(tables), charts=len(charts), output_count=total_outputs,
        timing_seconds={name: round(value, 4) for name, value in timings.items()},
        total_seconds=round(perf_counter() - started, 4), analytical_device="cpu", dtype="float64",
        resident_sparse=True, out_of_core=False, default_exports=False,
        analytical_solver_dependencies=["openecon native sparse network kernels", "torch"],
        graph_memory_budget_mb=MEMORY_MB,
        memory_budget_scope="owned operation buffers plus resident snapshots; not whole-process RSS")
    print(f"Completed · {metadata['nodes']:,} firms · {metadata['aggregate_edges']:,} distinct directed links · "
          f"8 months · {len(tables)} tables · {len(charts)} charts · {metadata['total_seconds']:.2f}s")
    return dict(tables=tables, charts=charts, metadata=metadata, proof=proof)


if __name__ == "__main__":
    network_lab_result = run_lab(globals().get("NETWORK_LAB_CONFIG"),
                                display_callback=globals().get("display"))
