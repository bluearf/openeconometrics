"""Explicit observed-dyad Gaussian and conditional mixed-membership models.

No implicit absent-dyad zeros: an observation table states value and split.
Gaussian blocks use bounded hard-label profile likelihood with a declared
variance floor. Mixed memberships are fixed node-simplex parameters, fitted
by exact marginal Bernoulli likelihood (not Bayesian variational inference).
Role draws marginalized out give p_ij = pi_i' B pi_j (Airoldi et al., 2008).
"""

from __future__ import annotations

import math
import time

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _label, _real
from openecon._network_model_common import ModelResult, Work, binary_graph
from openecon._network_sbm import _initial


def _sample(graph, observations, maximum, work, *, binary=False):
    if not isinstance(observations, pd.DataFrame) or any(
        list(observations.columns).count(c) != 1 for c in ["source", "target", "value"]
    ):
        _error(
            "invalid_dyads",
            "Use an explicit source/target/value dataframe; optional split is train/validation/test.",
        )
    if not 1 <= len(observations) <= maximum or list(observations.columns).count("split") > 1:
        _error(
            "observation_budget",
            "Observed dyad table must be nonempty and within max_observations.",
        )
    work.add(len(observations))
    seen, pairs, values, splits = set(), [], [], []
    has_split = "split" in observations
    columns = ["source", "target", "value", *(["split"] if has_split else [])]
    for row in observations.loc[:, columns].itertuples(index=False, name=None):
        left, right = _label(row[0]), _label(row[1])
        if left not in graph._index or right not in graph._index or left == right:
            _error("invalid_dyads", "Observed dyads require two distinct exact known node IDs.")
        i, j = graph._index[left], graph._index[right]
        pair = (i, j) if graph.directed or i < j else (j, i)
        if pair in seen:
            _error(
                "duplicate_dyad",
                "Every dyad has one value and one split; reverse undirected duplicates are invalid.",
            )
        seen.add(pair)
        value = _real(
            row[2],
            "invalid_value",
            "Observed values must be finite real numbers, including explicit zero.",
        )
        if binary and value not in {0.0, 1.0}:
            _error(
                "invalid_value", "Mixed-membership Bernoulli observations must be exactly 0 or 1."
            )
        split = row[3] if has_split else "train"
        if not isinstance(split, str) or split not in {"train", "validation", "test"}:
            _error("invalid_split", "Each split must be train, validation or test.")
        pairs.append(pair)
        values.append(value)
        splits.append(split)
    if "train" not in splits:
        _error("empty_sample", "At least one training dyad is required.")
    return torch.tensor(pairs, dtype=torch.int64), torch.tensor(values, dtype=torch.float64), splits


class LatentBlockResult(ModelResult):
    def predict(self, pairs, *, max_pairs=100_000):
        """Selected typed dyads, preserving order and duplicates; no uncertainty claim."""
        maximum = _integer(max_pairs, "max_pairs", 1_000_000)
        if not isinstance(pairs, (list, tuple)) or len(pairs) > maximum:
            _error("output_budget", "Prediction needs a bounded list of explicit node pairs.")
        meta = self["metadata"]
        labels = self["membership"].node.tolist()
        index = {label: i for i, label in enumerate(labels)}
        if (
            32768 + 2048 * len(labels) * meta["groups"] + 2048 * len(pairs)
            > meta["prediction_memory_budget"]
        ):
            _error(
                "memory_budget",
                "Selected-dyad prediction exceeds the fitted graph memory allowance.",
            )
        if meta["model"] == "Gaussian observed-dyad SBM":
            member = self["membership"].block.tolist()
            cells = {
                (int(r.source_block), int(r.target_block)): (
                    float(r.mean),
                    float(r.variance),
                    bool(r.identified),
                )
                for r in self["blocks"].itertuples(index=False)
            }
        else:
            pi = torch.tensor(
                self["membership"].drop(columns="node").to_numpy(), dtype=torch.float64
            )
            b = torch.tensor(self["blocks"].to_numpy(), dtype=torch.float64)
        rows = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                _error("invalid_pairs", "Each prediction pair contains exactly two typed nodes.")
            left, right = _label(pair[0]), _label(pair[1])
            if left not in index or right not in index or left == right:
                _error("invalid_pairs", "Prediction dyads need distinct fitted node IDs.")
            i, j = index[left], index[right]
            if meta["model"] == "Gaussian observed-dyad SBM":
                left_group, right_group = member[i], member[j]
                if not meta["directed"] and left_group > right_group:
                    left_group, right_group = right_group, left_group
                mean, variance, identified = cells[left_group, right_group]
                if not identified:
                    _error(
                        "unidentified_block",
                        "This block has no training observations; a held-out value cannot identify it.",
                    )
            else:
                mean = float(pi[i] @ b @ pi[j])
                variance = mean * (1 - mean)
            rows.append((left, right, mean, variance))
        result = as_frame(pd.DataFrame(rows, columns=["source", "target", "mean", "variance"]))
        if len(result):
            result["source"], result["target"] = (
                pd.Series([r[0] for r in rows], dtype=object),
                pd.Series([r[1] for r in rows], dtype=object),
            )
        result.attrs.update(
            model=meta["model"],
            uncertainty="conditional distribution variance; parameter uncertainty not estimated",
        )
        return result


def _gaussian_profile(membership, pairs, y, groups, directed, floor):
    a, b = membership[pairs[:, 0]], membership[pairs[:, 1]]
    if not directed:
        a, b = torch.minimum(a, b), torch.maximum(a, b)
    codes = a * groups + b
    counts = torch.bincount(codes, minlength=groups * groups).to(torch.float64)
    sums = torch.zeros(groups * groups, dtype=torch.float64).scatter_add_(0, codes, y)
    mean = sums / counts.clamp_min(1)
    residuals = y - mean[codes]
    ss = torch.zeros_like(sums).scatter_add_(0, codes, residuals.square())
    variance = (ss / counts.clamp_min(1)).clamp_min(floor)
    loglike = (
        -0.5
        * (
            math.log(2 * math.pi) + variance[codes].log() + residuals.square() / variance[codes]
        ).sum()
    )
    if not bool(torch.isfinite(loglike)):
        _error("numerical_failure", "Gaussian block likelihood overflow; rescale observations.")
    return float(loglike), counts, mean, variance


def gaussian_block_model(
    graph,
    groups,
    observations,
    *,
    initial=None,
    seed=0,
    starts=4,
    max_iter=50,
    tol=1e-8,
    variance_floor=1e-6,
    max_observations=100_000,
    max_work=50_000_000,
    timeout=60.0,
    cancelled=None,
):
    """Fixed-K Gaussian profile ascent on observed dyads; negative/zero values valid.

    Graph supplies ONLY typed nodes/direction, not responses or initialization.
    Nonobserved dyads are missing, not zero. Node labels are shared across splits;
    only train values affect fitting. Variances are MLEs constrained >= floor.
    Zero-observation blocks are unidentified. All groups remain nonempty; a
    no-move sweep is only a single-node local optimum, never global recovery.
    """
    binary_graph(graph)
    groups = _integer(groups, "groups", 128)
    starts = _integer(starts, "starts", 64)
    max_iter = _integer(max_iter, "max_iter", 1000, zero=True)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    maximum = _integer(max_observations, "max_observations", 1_000_000)
    floor = _real(variance_floor, "invalid_option", "variance_floor must be finite and positive.")
    tol = _real(tol, "invalid_option", "tol must be finite and nonnegative.")
    if (
        floor <= 0
        or tol < 0
        or groups > graph.node_count
        or (not max_iter and (initial is None or starts != 1))
    ):
        _error(
            "invalid_option",
            "Need positive variance floor, nonnegative tol, K<=N; zero iterations require one explicit partition.",
        )
    count = len(observations) if isinstance(observations, pd.DataFrame) else 0
    if count > maximum:
        _error("observation_budget", "Observed dyads exceed max_observations.")
    work = Work(
        graph,
        workspace=32768 + 2048 * count + 1024 * graph.node_count + 2048 * groups * groups,
        max_work=max_work,
        timeout=timeout,
        cancelled=cancelled,
    )
    pairs, y, splits = _sample(graph, observations, maximum, work)
    mask = torch.tensor([s == "train" for s in splits], dtype=torch.bool)
    train_pairs, train_y = pairs[mask], y[mask]
    # Anchor the data before profile moments, retaining sub-unit differences at large levels.
    anchor = float(train_y[0])
    train_y = train_y - anchor
    rng = torch.Generator(device="cpu").manual_seed(seed)
    n, best, runs = graph.node_count, None, []
    for start in range(starts):
        if start == 0 and initial is not None:
            membership = torch.tensor(_initial(graph, initial, groups), dtype=torch.int64)
        else:
            membership = torch.empty(n, dtype=torch.int64)
            membership[torch.randperm(n, generator=rng)] = torch.arange(n) % groups
        work.add(len(train_y) + n)
        objective, *_ = _gaussian_profile(
            membership, train_pairs, train_y, groups, graph.directed, floor
        )
        history, converged = [objective], groups == 1
        for iteration in range(max_iter):
            changed = 0
            for node in torch.randperm(n, generator=rng).tolist():
                old = int(membership[node])
                if int((membership == old).sum()) == 1:
                    continue
                target, value = old, objective
                for group in range(groups):
                    if group == old:
                        continue
                    work.add(len(train_y) * 8 + groups * groups + n)
                    candidate = membership.clone()
                    candidate[node] = group
                    trial, *_ = _gaussian_profile(
                        candidate, train_pairs, train_y, groups, graph.directed, floor
                    )
                    if trial > value + tol:
                        target, value = group, trial
                if target != old:
                    membership[node], objective = target, value
                    changed += 1
            history.append(objective)
            if not changed:
                converged = True
                break
        record = {
            "start": start,
            "iterations": len(history) - 1,
            "objective": objective,
            "converged": converged,
            "objective_history": history,
        }
        runs.append(record)
        if best is None or objective > best[0]:
            best = objective, membership.clone(), record
    objective, membership, record = best
    _, counts, mean, variance = _gaussian_profile(
        membership, train_pairs, train_y, groups, graph.directed, floor
    )
    blocks = []
    for a in range(groups):
        for b in range(0 if graph.directed else a, groups):
            code = a * groups + b
            identified = counts[code] > 0
            blocks.append(
                (
                    a,
                    b,
                    int(counts[code]),
                    float(mean[code]) + anchor if identified else math.nan,
                    float(variance[code]) if identified else math.nan,
                    bool(identified),
                )
            )
    result = LatentBlockResult(
        membership=graph._frame({"block": membership.tolist()}),
        blocks=as_frame(
            pd.DataFrame(
                blocks,
                columns=[
                    "source_block",
                    "target_block",
                    "observations",
                    "mean",
                    "variance",
                    "identified",
                ],
            )
        ),
        metadata={
            "model": "Gaussian observed-dyad SBM",
            "method": "hard-label constrained profile likelihood",
            "nodes": n,
            "groups": groups,
            "fixed_groups": True,
            "directed": graph.directed,
            "self_loops": False,
            "observations": int(mask.sum()),
            "observed_dyads": len(y),
            "split_counts": {s: splits.count(s) for s in set(splits)},
            "sample_space": "explicit real-valued dyads, missing dyads excluded; zero/negative observed values retained",
            "variance_floor": floor,
            "identification": "blocks require training observations; labels identified only up to permutation",
            "objective": objective,
            "iterations": record["iterations"],
            "converged": record["converged"],
            "convergence_rule": "coordinate label sweep with no improvement; local optimum only",
            "start_fits": runs,
            "seed": seed,
            "starts": starts,
            "global_optimum_certified": False,
            "uncertainty": "not estimated",
            "training_uses": "only train values; graph topology and holdouts excluded",
            **work.metadata(),
        },
    )
    result["metadata"]["prediction_memory_budget"] = graph._budget.limit
    result["evaluation"] = _evaluation(result, pairs, y, splits, graph, train_y + anchor)
    return result


def _evaluation(result, pairs, y, splits, graph, train_y):
    gaussian = result["metadata"]["model"] == "Gaussian observed-dyad SBM"
    baseline = float(train_y.mean())
    variance = max(
        float(((train_y - baseline) ** 2).mean()), result["metadata"].get("variance_floor", 1e-6)
    )
    if not gaussian:
        baseline = min(1 - 1e-12, max(1e-12, baseline))
    rows = []
    for split in ["train", "validation", "test"]:
        selected = [i for i, s in enumerate(splits) if s == split]
        if not selected:
            continue
        requested = [
            (graph._labels[int(pairs[i, 0])], graph._labels[int(pairs[i, 1])]) for i in selected
        ]
        try:
            prediction = result.predict(requested, max_pairs=len(requested))
        except Exception as exc:
            from openecon.analysis_contracts import AnalysisError

            if isinstance(exc, AnalysisError) and exc.code == "network_unidentified_block":
                rows.append((split, len(selected), None, None, None, "unidentified training block"))
                continue
            raise
        observed = y[selected]
        mean = torch.tensor(prediction["mean"].to_numpy(), dtype=torch.float64)
        if gaussian:
            var = torch.tensor(prediction["variance"].to_numpy(), dtype=torch.float64)
            ll = float(
                -0.5 * (math.log(2 * math.pi) + var.log() + (observed - mean).square() / var).sum()
            )
            base = float(
                -0.5
                * (
                    math.log(2 * math.pi * variance) + (observed - baseline).square() / variance
                ).sum()
            )
        else:
            mean = mean.clamp(1e-12, 1 - 1e-12)
            ll = float((observed * mean.log() + (1 - observed) * torch.log1p(-mean)).sum())
            base = float(
                (observed * math.log(baseline) + (1 - observed) * math.log1p(-baseline)).sum()
            )
        rows.append(
            (split, len(selected), ll, base, float((observed - mean).square().mean()), "evaluated")
        )
    return as_frame(
        pd.DataFrame(
            rows,
            columns=[
                "split",
                "observations",
                "log_likelihood",
                "baseline_log_likelihood",
                "mse",
                "status",
            ],
        )
    )


def mixed_membership_block_model(
    graph,
    groups,
    observations,
    *,
    seed=0,
    starts=4,
    max_iter=200,
    batch_size=1024,
    learning_rate=0.03,
    tol=1e-6,
    max_observations=100_000,
    max_work=500_000_000,
    timeout=120.0,
    cancelled=None,
):
    """Conditional mixed-membership Bernoulli likelihood with mini-batch Adam.

    pi_i is a fixed per-node simplex, not a posterior Dirichlet parameter.
    Latent dyad roles are integrated exactly. Undirected B is symmetric.
    Initialization and epochs use a private RNG; graph topology and held-out
    responses do not enter fitting. No unobserved zeros or negative samples
    are manufactured. Parameters are local, potentially unidentified; no SE
    or global-optimum guarantee is provided.
    """
    binary_graph(graph)
    starts = _integer(starts, "starts", 64)
    if starts > 1:
        maximum_work = _integer(max_work, "max_work", 2**63 - 1)
        started, used, best, records = time.monotonic(), 0, None, []
        total_timeout = _real(timeout, "invalid_option", "timeout must be positive and finite.")
        for start in range(starts):
            remaining = total_timeout - (time.monotonic() - started)
            if remaining <= 0:
                _error(
                    "time_budget", "All requested mixed-membership starts must fit within timeout."
                )
            candidate = mixed_membership_block_model(
                graph,
                groups,
                observations,
                seed=(_integer(seed, "seed", 2**63 - 1, zero=True) + start) % 2**63,
                starts=1,
                max_iter=max_iter,
                batch_size=batch_size,
                learning_rate=learning_rate,
                tol=tol,
                max_observations=max_observations,
                max_work=maximum_work - used,
                timeout=remaining,
                cancelled=cancelled,
            )
            meta = candidate["metadata"]
            used += meta["work_used"]
            records.append(
                {
                    "start": start,
                    "seed": meta["seed"],
                    "objective": meta["objective"],
                    "converged": meta["converged"],
                }
            )
            if best is None or meta["objective"] > best["metadata"]["objective"]:
                best = candidate
        best["metadata"].update(
            starts=starts,
            start_fits=records,
            work_used=used,
            max_work=maximum_work,
            timeout_seconds=total_timeout,
        )
        return best
    groups = _integer(groups, "groups", 128)
    max_iter = _integer(max_iter, "max_iter", 10_000)
    batch_size = _integer(batch_size, "batch_size", 100_000)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    maximum = _integer(max_observations, "max_observations", 1_000_000)
    rate = _real(learning_rate, "invalid_option", "learning_rate must be positive and finite.")
    tol = _real(tol, "invalid_option", "tol must be positive and finite.")
    if groups > graph.node_count or rate <= 0 or tol <= 0:
        _error("invalid_option", "Need K<=N and positive learning_rate/tol.")
    count = len(observations) if isinstance(observations, pd.DataFrame) else 0
    if count > maximum:
        _error("observation_budget", "Observed dyads exceed max_observations.")
    n = graph.node_count
    work = Work(
        graph,
        workspace=32768
        + 2048 * count
        + 1024 * n * groups
        + 1024 * groups * groups
        + 512 * batch_size * groups,
        max_work=max_work,
        timeout=timeout,
        cancelled=cancelled,
    )
    pairs, y, splits = _sample(graph, observations, maximum, work, binary=True)
    train = torch.tensor([i for i, s in enumerate(splits) if s == "train"], dtype=torch.int64)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn((n, groups), generator=rng, dtype=torch.float64).requires_grad_()
    block = torch.randn((groups, groups), generator=rng, dtype=torch.float64).requires_grad_()
    optimizer = torch.optim.Adam([logits, block], lr=rate)
    history, converged = [], False

    def probability(ids):
        pi, b = logits.softmax(1), block.sigmoid()
        if not graph.directed:
            b = (b + b.T) / 2
        selected = pairs[ids]
        return ((pi[selected[:, 0]] @ b) * pi[selected[:, 1]]).sum(1).clamp(1e-12, 1 - 1e-12)

    def objective():
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(train), batch_size):
                work.add(n * groups + groups * groups + batch_size * groups * groups)
                ids = train[start : start + batch_size]
                p = probability(ids)
                total += float((y[ids] * p.log() + (1 - y[ids]) * torch.log1p(-p)).sum())
        return total

    for iteration in range(max_iter):
        order = train[torch.randperm(len(train), generator=rng)]
        for start in range(0, len(order), batch_size):
            work.add(n * groups * 8 + groups * groups * 8 + batch_size * groups * groups * 8)
            ids = order[start : start + batch_size]
            optimizer.zero_grad()
            p = probability(ids)
            loss = -(y[ids] * p.log() + (1 - y[ids]) * torch.log1p(-p)).mean()
            loss.backward()
            optimizer.step()
        value = objective()
        history.append(value)
        if len(history) > 5 and max(
            abs(b - a) for a, b in zip(history[-5:], history[-4:])
        ) <= tol * max(1.0, abs(value)):
            converged = True
            break
    pi, b = logits.detach().softmax(1), block.detach().sigmoid()
    if not graph.directed:
        b = (b + b.T) / 2
    trained_nodes = set(pairs[train].reshape(-1).tolist())
    result = LatentBlockResult(
        membership=graph._frame({f"membership_{k}": pi[:, k].tolist() for k in range(groups)}),
        blocks=as_frame(pd.DataFrame(b.numpy(), index=range(groups), columns=range(groups))),
        metadata={
            "model": "Conditional mixed-membership Bernoulli SBM",
            "method": "mini-batch marginal-likelihood Adam",
            "nodes": n,
            "groups": groups,
            "directed": graph.directed,
            "self_loops": False,
            "fixed_groups": True,
            "observations": len(train),
            "observed_dyads": len(y),
            "split_counts": {s: splits.count(s) for s in set(splits)},
            "sample_space": "explicit binary dyads; absent observations are missing, not zero",
            "memberships": "fixed node-simplex parameters; not Bayesian posterior estimates",
            "identification": "block-label permutations and latent factorizations need not be unique; unidentified nodes remain initialized",
            "unobserved_train_nodes": [
                graph._labels[i] for i in range(n) if i not in trained_nodes
            ],
            "objective": history[-1],
            "objective_history": history,
            "iterations": len(history),
            "converged": converged,
            "convergence_rule": "five relative objective changes below tolerance; not a score or global-optimum certificate",
            "seed": seed,
            "starts": 1,
            "batch_size": batch_size,
            "learning_rate": rate,
            "uncertainty": "not estimated",
            "training_uses": "only explicit train observations; no graph topology or held-out values",
            **work.metadata(),
        },
    )
    result["metadata"]["prediction_memory_budget"] = graph._budget.limit
    result["evaluation"] = _evaluation(result, pairs, y, splits, graph, y[train])
    return result
