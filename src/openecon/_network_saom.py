"""Fixed-actor, fully observed binary SAOM with exact small-state panel likelihood.

Constant opportunity rate per actor; choices are no change or one outgoing
tie toggle, with softmax of the actor objective in the resulting state.
This is a continuous-time actor choice process, not an ERGM estimator.
Snijders (2017), Annual Review of Statistics 4, equations (4)-(5).
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key, _real, network
from openecon._network_model_common import ModelResult, Work, binary_graph, coefficients
from openecon._network_temporal import NetworkSnapshots, network_snapshots

TERMS = ("outdegree", "reciprocity", "transitive_triplets")


def _terms(terms):
    if (
        not isinstance(terms, (list, tuple))
        or not terms
        or any(not isinstance(t, str) or t not in TERMS for t in terms)
        or len(set(terms)) != len(terms)
    ):
        _error("unsupported_term", "SAOM terms are outdegree, reciprocity and transitive_triplets.")
    return list(terms)


def actor_statistics(a, actor, terms):
    values = {
        "outdegree": a[actor].sum(),
        "reciprocity": a[actor] @ a[:, actor],
        "transitive_triplets": a[actor] @ (a @ a[actor]),
    }
    return torch.stack([values[term] for term in terms])


def choices(a, actor, terms):
    rows = [actor_statistics(a, actor, terms)]
    targets = [j for j in range(len(a)) if j != actor]
    for target in targets:
        changed = a.clone()
        changed[actor, target] = 1 - changed[actor, target]
        rows.append(actor_statistics(changed, actor, terms))
    return torch.stack(rows), targets


def _state_matrix(state, n, pairs):
    a = torch.zeros((n, n), dtype=torch.float64)
    for bit, (i, j) in enumerate(pairs):
        a[i, j] = float(bool(state & (1 << bit)))
    return a


def _encoding(graph, labels, pairs):
    present = {(graph._labels[i], graph._labels[j]) for i, j in graph._edges.indices().T.tolist()}
    return sum(1 << bit for bit, (i, j) in enumerate(pairs) if (labels[i], labels[j]) in present)


def _scaffold(n, terms, work):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    states = 2 ** len(pairs)
    work.add(states * n * n**3)
    features = torch.empty((states, n, n, len(terms)), dtype=torch.float64)
    destinations = torch.empty((states, n, n - 1), dtype=torch.int64)
    position = {pair: bit for bit, pair in enumerate(pairs)}
    for state in range(states):
        if state % 128 == 0:
            work.add(0)
        a = _state_matrix(state, n, pairs)
        for actor in range(n):
            features[state, actor], targets = choices(a, actor, terms)
            destinations[state, actor] = torch.tensor(
                [state ^ (1 << position[actor, j]) for j in targets]
            )
    return features, destinations, pairs


def generator_matrix(parameters, features, destinations):
    """parameters = [log opportunity rate, objective coefficients]."""
    states, n = features.shape[:2]
    probabilities = (features @ parameters[1:]).softmax(2)
    rates = parameters[0].exp() * probabilities[:, :, 1:]
    rows = torch.arange(states)[:, None, None].expand_as(destinations).reshape(-1)
    q = torch.zeros((states, states), dtype=torch.float64)
    q = q.index_put((rows, destinations.reshape(-1)), rates.reshape(-1), accumulate=True)
    return q - torch.diag(q.sum(1))


class SAOMResult(ModelResult):
    def simulate(self, graph, times, **kwargs):
        return simulate_saom(
            graph,
            times,
            rate=self["metadata"]["opportunity_rate"],
            theta=self["coefficients"].coefficient.iloc[1:].tolist(),
            terms=self["metadata"]["terms"],
            **kwargs,
        )


def _times(times, count=None):
    if (
        not isinstance(times, (list, tuple))
        or len(times) < 2
        or (count is not None and len(times) != count)
    ):
        _error(
            "invalid_times",
            "times must contain one explicit elapsed time per panel wave, with at least two waves.",
        )
    values = [_real(t, "invalid_times", "Panel times must be finite real numbers.") for t in times]
    if any(b <= a or not math.isfinite(b - a) for a, b in zip(values, values[1:])):
        _error("invalid_times", "Panel times must be strictly increasing with finite intervals.")
    return values


def saom(
    panels,
    times,
    *,
    terms=("outdegree",),
    initial=None,
    max_iter=100,
    tol=1e-6,
    max_states=256,
    max_work=500_000_000,
    timeout=120.0,
    cancelled=None,
):
    """Exact CTMC maximum likelihood for a small fully observed fixed-actor panel.

    Pass one ordered NetworkSnapshots or a list of independent replicate panels
    sharing the same explicit wave times and typed actor set. Constant actor
    rate is estimated jointly. Missing dyads/actor turnover must not be encoded
    as absence; partial observation and behavior coevolution are unsupported.
    Exact enumeration is bounded by max_states; no approximate estimator is
    substituted. Parameter domain is explicit: log(rate) in [-8,8], beta in
    [-15,15]. Boundary/flat solutions have no reported covariance.
    """
    terms = _terms(terms)
    max_states = _integer(max_states, "max_states", 4096)
    max_iter = _integer(max_iter, "max_iter", 1000)
    tol = _real(tol, "invalid_option", "tol must be positive.")
    if tol <= 0:
        _error("invalid_option", "tol must be positive.")
    if isinstance(panels, NetworkSnapshots):
        panels = [panels]
    if not isinstance(panels, (list, tuple)) or not panels or len(panels) > 10_000:
        _error("invalid_panel", "Use one panel or a bounded list of independent panels.")
    first = panels[0]
    if not isinstance(first, NetworkSnapshots) or not first.ordered:
        _error("invalid_panel", "SAOM needs explicitly ordered snapshots.")
    times = _times(times, first.snapshot_count)
    labels = list(first._labels)
    n, p = len(labels), len(terms) + 1
    m = n * (n - 1)
    if m >= 63 or 2**m > max_states:
        _error("state_budget", "Exact SAOM transition space exceeds max_states.")
    states = 2**m
    source = first._graphs[0]
    workspace = (
        32768
        + 1024 * states * states * (p * p + len(times))
        + 1024 * states * n * n * p
        + 2048 * len(panels) * len(times)
    )
    work = Work(
        source, workspace=workspace, max_work=max_work, timeout=timeout, cancelled=cancelled
    )
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    counts = [Counter() for _ in times[1:]]
    all_graphs = {}
    for panel in panels:
        if (
            not isinstance(panel, NetworkSnapshots)
            or not panel.ordered
            or panel.snapshot_count != len(times)
            or list(panel._labels) != labels
        ):
            _error(
                "invalid_panel",
                "All replicate panels need identical fully observed typed actors and wave counts.",
            )
        encoded = []
        for graph in panel._graphs:
            binary_graph(graph, directed=True)
            if sorted(graph._labels, key=_key) != labels:
                _error(
                    "actor_turnover",
                    "SAOM requires the same fully observed actors at every wave; isolates must be explicit.",
                )
            all_graphs[id(graph)] = graph
            encoded.append(_encoding(graph, labels, pairs))
        for count, a, b in zip(counts, encoded, encoded[1:]):
            count[a, b] += 1
    resident = sum(graph._base_bytes for graph in all_graphs.values())
    for graph in all_graphs.values():
        graph._budget.check(resident + workspace)
    features, destinations, _ = _scaffold(n, terms, work)
    counts = [
        torch.tensor([[a, b, k] for (a, b), k in count.items()], dtype=torch.int64)
        for count in counts
    ]
    elapsed = [b - a for a, b in zip(times, times[1:])]

    def likelihood(parameters):
        work.add(len(times) * states**3 * 8)
        q = generator_matrix(parameters, features, destinations)
        value = torch.zeros((), dtype=torch.float64)
        for duration, count in zip(elapsed, counts):
            transition = torch.matrix_exp(q * duration)
            probabilities = transition[count[:, 0], count[:, 1]]
            if not bool(torch.isfinite(probabilities).all()) or bool((probabilities <= 0).any()):
                _error(
                    "numerical_failure",
                    "SAOM transition probability is nonpositive; rescale time/rate.",
                )
            value = value + (probabilities.log() * count[:, 2]).sum()
        return value

    bounds = torch.tensor([8.0, *([15.0] * len(terms))], dtype=torch.float64)
    theta = (
        torch.zeros(p, dtype=torch.float64)
        if initial is None
        else torch.tensor(initial, dtype=torch.float64)
    )
    if (
        theta.shape != (p,)
        or not bool(torch.isfinite(theta).all())
        or bool((theta.abs() >= bounds).any())
    ):
        _error(
            "invalid_option",
            "initial is [log rate, one coefficient per term], strictly within explicit parameter bounds.",
        )
    raw = torch.atanh(theta / bounds).requires_grad_()
    optimizer = torch.optim.LBFGS(
        [raw],
        max_iter=max_iter,
        tolerance_grad=tol * 0.1,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    history = []

    def closure():
        optimizer.zero_grad()
        value = likelihood(bounds * raw.tanh())
        history.append(float(value.detach()))
        (-value).backward()
        return -value

    optimizer.step(closure)
    theta = (bounds * raw.tanh()).detach().requires_grad_()
    objective = likelihood(theta)
    score = torch.autograd.grad(objective, theta)[0]
    information = -torch.autograd.functional.hessian(likelihood, theta)
    eigen = torch.linalg.eigvalsh(information)
    identified = float(eigen.min()) > max(1e-10, float(eigen.max()) * 1e-10)
    boundary = bool((theta.detach().abs() > bounds * 0.99).any())
    converged = float(score.abs().max()) < tol * 10 and identified and not boundary
    covariance = torch.linalg.inv(information).detach() if converged else None
    names = ["log_rate", *terms]
    return SAOMResult(
        coefficients=coefficients(names, theta.detach(), covariance),
        covariance=None
        if covariance is None
        else as_frame(pd.DataFrame(covariance.numpy(), index=names, columns=names)),
        metadata={
            "model": "SAOM",
            "method": "exact small-state CTMC panel maximum likelihood",
            "terms": terms,
            "nodes": n,
            "actor_ids": labels,
            "states": states,
            "times": times,
            "independent_panels": len(panels),
            "observations": len(panels) * (len(times) - 1),
            "opportunity_rate": float(theta[0].detach().exp()),
            "rate_unit": "opportunities per actor per elapsed-time unit",
            "choice_space": "no change or one outgoing dyad toggle",
            "missingness": "fully observed only; absent ties are observed zeros",
            "parameter_bounds": {"log_rate": [-8, 8], "objective_coefficients": [-15, 15]},
            "converged": converged,
            "identified": identified,
            "boundary": boundary,
            "iterations": int(optimizer.state[raw].get("n_iter", 0)),
            "objective": float(objective.detach()),
            "score": score.detach().tolist(),
            "score_max": float(score.abs().max()),
            "objective_evaluations": history,
            "uncertainty": "inverse observed panel likelihood information"
            if covariance is not None
            else "unavailable: boundary/unidentified/unconverged",
            "simulation_repetitions": 0,
            "seed": None,
            **work.metadata(),
        },
    )


def simulate_saom(
    graph,
    times,
    *,
    rate=1.0,
    theta=(0.0,),
    terms=("outdegree",),
    seed=0,
    max_events=100_000,
    max_work=50_000_000,
    timeout=60.0,
    cancelled=None,
):
    """Gillespie actor opportunities and softmax single-tie choices, including no-change."""
    binary_graph(graph, directed=True)
    times, terms = _times(times), _terms(terms)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    maximum = _integer(max_events, "max_events", 10_000_000)
    rate = _real(rate, "invalid_option", "rate must be positive and finite.")
    if rate <= 0 or not math.isfinite(rate * graph.node_count):
        _error("invalid_option", "rate must be positive with finite total opportunity rate.")
    theta = torch.tensor(theta, dtype=torch.float64)
    if theta.shape != (len(terms),) or not bool(torch.isfinite(theta).all()):
        _error("invalid_option", "theta must contain one finite objective coefficient per term.")
    n = graph.node_count
    work = Work(
        graph,
        workspace=32768 + 1024 * n * n * (len(times) + len(terms)),
        max_work=max_work,
        timeout=timeout,
        cancelled=cancelled,
    )
    from openecon._network_model_common import adjacency

    a = adjacency(graph)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    current, events, changes = times[0], 0, 0
    layers, diagnostics = {}, []
    # Exponential opportunities may be no-change. Carry residual waiting time
    # across reporting waves; wave boundaries do not reset the random process.
    next_event = current - math.log(max(float(torch.rand((), generator=rng)), 1e-300)) / (n * rate)
    for wave, target_time in enumerate(times):
        while next_event <= target_time:
            work.add(n**4 + len(terms) * n)
            if events >= maximum:
                _error(
                    "event_budget",
                    "SAOM simulation exceeded max_events; no truncated panel is returned.",
                )
            current = next_event
            actor = int(torch.randint(n, (), generator=rng))
            features, targets = choices(a, actor, terms)
            probability = (features @ theta).softmax(0)
            chosen = int(torch.multinomial(probability, 1, generator=rng))
            if chosen:
                j = targets[chosen - 1]
                a[actor, j] = 1 - a[actor, j]
                changes += 1
            events += 1
            next_event = current - math.log(max(float(torch.rand((), generator=rng)), 1e-300)) / (
                n * rate
            )
        edges = (a > 0).nonzero().tolist()
        layers[wave] = network(
            [{"source": graph._labels[i], "target": graph._labels[j]} for i, j in edges],
            nodes=list(graph._labels),
            directed=True,
            max_memory_mb=graph._budget.limit / 1024**2,
        )
        diagnostics.append((wave, target_time, events, changes, len(edges)))
    panel = network_snapshots(layers, ordered=True)
    return ModelResult(
        panel=panel,
        diagnostics=as_frame(
            pd.DataFrame(diagnostics, columns=["wave", "time", "opportunities", "changes", "edges"])
        ),
        metadata={
            "model": "SAOM simulation",
            "method": "Gillespie actor opportunities and softmax choices",
            "nodes": n,
            "times": times,
            "terms": terms,
            "rate": rate,
            "seed": seed,
            "opportunities": events,
            "changes": changes,
            "max_events": maximum,
            "simulation_repetitions": 1,
            "missingness": "fully observed, constant typed actors",
            "rate_unit": "per actor per elapsed-time unit",
            **work.metadata(),
        },
    )
