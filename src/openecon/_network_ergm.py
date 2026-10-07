"""Undirected binary ERGM: exact likelihood, MPLE, and toggle Gibbs simulation.

P(Y=y)=exp(theta'g(y)-log Z(theta)), full loopless labeled graph space.
Terms: edges, unordered two-stars, triangles. Exact likelihood Fisher information
is the covariance of sufficient statistics. MPLE is explicitly a composite
conditional likelihood; its inverse Hessian is not advertised as uncertainty
for dependent dyads. Reference: Hunter et al. (2008), JSS 24(3).
"""

from __future__ import annotations

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _real, network
from openecon._network_model_common import (
    ModelResult,
    Work,
    adjacency,
    binary_graph,
    coefficients,
)

TERMS = ("edges", "twostars", "triangles")


def _terms(terms):
    if (
        not isinstance(terms, (list, tuple))
        or not terms
        or any(not isinstance(term, str) or term not in TERMS for term in terms)
        or len(set(terms)) != len(terms)
    ):
        _error(
            "unsupported_term", "terms must be a nonempty unique list of edges/twostars/triangles."
        )
    return list(terms)


def statistics(a, terms):
    degree = a.sum(1)
    values = {
        "edges": a.sum() / 2,
        "twostars": (degree * (degree - 1)).sum() / 2,
        "triangles": torch.trace(a @ a @ a) / 6,
    }
    return torch.stack([values[name] for name in terms])


def _change(a, i, j, terms):
    # Increment when this dyad is inserted into the graph WITHOUT itself.
    previous = float(a[i, j])
    degree = a.sum(1)
    values = {
        "edges": 1.0,
        "twostars": float(degree[i] + degree[j] - 2 * previous),
        "triangles": float(a[i] @ a[j]),
    }
    return torch.tensor([values[name] for name in terms], dtype=torch.float64)


def _enumerate(n, terms, work, max_states):
    dyads = torch.triu_indices(n, n, 1)
    m = dyads.shape[1]
    if m >= 63 or 2**m > max_states:
        _error(
            "state_budget",
            "Exact ERGM graph enumeration exceeds max_states; use an explicitly selected method.",
        )
    states = 2**m
    work.add(states * (n**3 + m + len(terms)))
    result = torch.empty((states, len(terms)), dtype=torch.float64)
    a = torch.zeros((n, n), dtype=torch.float64)
    # Gray-code toggles leave only one changed dyad at each state.
    previous = 0
    for index in range(states):
        if index % 256 == 0:
            work.add(0)
        gray = index ^ (index >> 1)
        if index:
            bit = (gray ^ previous).bit_length() - 1
            i, j = dyads[:, bit].tolist()
            a[i, j] = a[j, i] = 1 - a[i, j]
        result[index] = statistics(a, terms)
        previous = gray
    return result


def _newton(theta, evaluate, *, max_iter, tol, work):
    history = []
    converged = False
    for iteration in range(max_iter + 1):
        work.add(theta.numel() ** 3)
        objective, score, information = evaluate(theta)
        history.append(float(objective))
        if float(score.abs().max()) <= tol:
            converged = True
            break
        if iteration == max_iter:
            break
        if not bool(torch.isfinite(information).all()):
            _error("numerical_failure", "ERGM information is nonfinite.")
        try:
            step = torch.linalg.solve(information, score)
        except torch.linalg.LinAlgError:
            _error(
                "unidentified_model",
                "ERGM information is singular; terms or sample are unidentified.",
            )
        accepted = False
        for power in range(30):
            candidate = theta + step * 2.0 ** (-power)
            if float(candidate.abs().max()) > 50:
                continue
            value = evaluate(candidate)[0]
            if float(value) >= float(objective) + 1e-4 * 2.0 ** (-power) * float(score @ step):
                theta, accepted = candidate, True
                break
        if not accepted:
            break
    return theta, converged, iteration, history, score, information


def _support_boundary(stats, observed, work):
    """Exact supporting-plane certificate for the integer statistic catalogue.

    At most three statistics: a support face through the observed point has a
    normal perpendicular to p-1 independent support differences. All integer
    products here are exactly representable in float64 at the admitted n<=7.
    """
    differences = torch.unique(stats, dim=0) - observed
    p, count = differences.shape[1], len(differences)
    work.add(count**p * max(1, p))
    if int(torch.linalg.matrix_rank(differences)) < p:
        _error("unidentified_model", "ERGM statistic support is not full dimensional.")
    if p == 1:
        return bool((differences >= 0).all() or (differences <= 0).all())

    def supports(normals):
        normals = normals[normals.ne(0).any(1)]
        if not len(normals):
            return False
        products = normals @ differences.T
        return bool(((products >= 0).all(1) | (products <= 0).all(1)).any())

    if p == 2:
        return supports(torch.stack([-differences[:, 1], differences[:, 0]], 1))
    for i in range(count - 1):
        work.add(0)
        if supports(
            torch.linalg.cross(differences[i].expand(count - i - 1, 3), differences[i + 1 :])
        ):
            return True
    return False


class ERGMResult(ModelResult):
    def simulate(
        self,
        *,
        draws=100,
        burn_in=1000,
        thin=10,
        seed=0,
        max_work=50_000_000,
        timeout=60.0,
        cancelled=None,
    ):
        """Seeded single-toggle Gibbs chain; one sample graph per retained draw."""
        return simulate_ergm(
            self["graph"],
            self["coefficients"].coefficient.tolist(),
            terms=self["coefficients"].term.tolist(),
            draws=draws,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            max_work=max_work,
            timeout=timeout,
            cancelled=cancelled,
        )


def ergm(
    graph,
    *,
    terms=("edges",),
    method="exact",
    values="binary",
    initial=None,
    max_iter=100,
    tol=1e-7,
    max_states=65_536,
    max_work=50_000_000,
    timeout=60.0,
    cancelled=None,
):
    """Fit a loopless undirected ERGM by exact MLE or explicitly approximate MPLE.

    Positive edge presence is selected explicitly with values='binary'. Exact
    enumeration is deliberately bounded; MPLE evaluates every eligible dyad.
    No unsupported term, conditioned edge count, or silent MCMC estimator is
    substituted. Exact covariance uses inverse Fisher information only after
    a finite converged solution; dependent MPLE uncertainty is unavailable.
    """
    binary_graph(graph, directed=False)
    terms = _terms(terms)
    if values != "binary" or not isinstance(method, str) or method not in {"exact", "mple"}:
        _error("invalid_option", "Select values='binary' and method='exact' or 'mple'.")
    max_iter = _integer(max_iter, "max_iter", 10_000)
    max_states = _integer(max_states, "max_states", 1_048_576)
    tol = _real(tol, "invalid_option", "tol must be positive and finite.")
    if tol <= 0:
        _error("invalid_option", "tol must be positive.")
    n, p = graph.node_count, len(terms)
    m = n * (n - 1) // 2
    if method == "exact" and (m >= 63 or 2**m > max_states):
        _error(
            "state_budget",
            "Exact enumeration exceeds max_states; no approximate fit is substituted.",
        )
    states = 2**m if method == "exact" else m
    workspace = 32768 + 128 * n * n + 128 * states * (p + 1)
    work = Work(graph, workspace=workspace, max_work=max_work, timeout=timeout, cancelled=cancelled)
    work.add(n**3 + m * n * n)
    a = adjacency(graph)
    observed = statistics(a, terms)
    theta = torch.zeros(p, dtype=torch.float64)
    if initial is not None:
        try:
            theta = torch.tensor(initial, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError):
            _error("invalid_option", "initial must be a finite coefficient vector.")
        if (
            theta.shape != (p,)
            or not bool(torch.isfinite(theta).all())
            or float(theta.abs().max()) > 50
        ):
            _error(
                "invalid_option", "initial must be one finite coefficient per term, bounded by 50."
            )
    if method == "exact":
        stats = _enumerate(n, terms, work, max_states)
        boundary = _support_boundary(stats, observed, work)

        def evaluate(value):
            work.add(states * (p * p + p))
            logits = stats @ value
            probability = logits.softmax(0)
            expectation = probability @ stats
            centered = stats - expectation
            information = centered.T @ (centered * probability[:, None])
            return (
                value @ observed - torch.logsumexp(logits, 0),
                observed - expectation,
                information,
            )
    else:
        boundary = graph.edge_count in {0, m}
        pairs = torch.triu_indices(n, n, 1)
        design = torch.stack([_change(a, i, j, terms) for i, j in pairs.T.tolist()])
        target = a[pairs[0], pairs[1]]

        def evaluate(value):
            work.add(m * (p * p + p))
            eta = design @ value
            probability = eta.sigmoid()
            return (
                (target * eta - torch.nn.functional.softplus(eta)).sum(),
                design.T @ (target - probability),
                design.T @ (design * (probability * (1 - probability))[:, None]),
            )

    theta, converged, iterations, history, score, information = _newton(
        theta, evaluate, max_iter=max_iter, tol=tol, work=work
    )
    eigen = torch.linalg.eigvalsh(information)
    identified = float(eigen.min()) > max(1e-12, float(eigen.max()) * 1e-12)
    # Empty/full graphs and convex-support boundaries can have only an infinite MLE.
    if boundary or not identified or float(theta.abs().max()) > 30:
        converged = False
    covariance = (
        torch.linalg.inv(information) if method == "exact" and converged and identified else None
    )
    result = ERGMResult(
        graph=graph,
        coefficients=coefficients(terms, theta, covariance),
        statistics=as_frame(pd.DataFrame({"term": terms, "observed": observed.tolist()})),
        covariance=None
        if covariance is None
        else as_frame(pd.DataFrame(covariance.numpy(), index=terms, columns=terms)),
        metadata={
            "model": "ERGM",
            "method": "exact MLE" if method == "exact" else "maximum pseudolikelihood",
            "nodes": n,
            "observations": m,
            "terms": terms,
            "boundary_statistic": boundary,
            "convergence_rule": "score tolerance, nonsingular information and full-support interior certificate for exact likelihood; MPLE score convergence only",
            "sample_space": "all loopless undirected binary graphs on fixed typed nodes",
            "values": values,
            "directed": False,
            "converged": converged,
            "identified": identified,
            "iterations": iterations,
            "objective": history[-1],
            "objective_history": history,
            "score": score.tolist(),
            "score_max": float(score.abs().max()),
            "states": states,
            "log_partition": float(torch.logsumexp(stats @ theta, 0))
            if method == "exact"
            else None,
            "uncertainty": "inverse exact Fisher information"
            if covariance is not None
            else "unavailable; dependent MPLE or nonfinite/unconverged MLE",
            "chain": None,
            "burn_in": None,
            "mixing": "not applicable to deterministic enumeration/MPLE",
            **work.metadata(),
        },
    )
    return result


def simulate_ergm(
    graph,
    theta,
    *,
    terms=("edges",),
    draws=100,
    burn_in=1000,
    thin=10,
    seed=0,
    max_work=50_000_000,
    timeout=60.0,
    cancelled=None,
):
    """Gibbs simulation from stated ERGM; this does not perform MCMC likelihood estimation."""
    binary_graph(graph, directed=False)
    terms = _terms(terms)
    draws = _integer(draws, "draws", 100_000)
    burn_in = _integer(burn_in, "burn_in", 10_000_000, zero=True)
    thin = _integer(thin, "thin", 1_000_000)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    try:
        theta = torch.tensor(theta, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        _error("invalid_option", "theta must be a finite coefficient vector.")
    if theta.shape != (len(terms),) or not bool(torch.isfinite(theta).all()):
        _error("invalid_option", "theta must contain one finite value per term.")
    n, p = graph.node_count, len(terms)
    m, steps = n * (n - 1) // 2, burn_in + draws * thin
    work = Work(
        graph,
        workspace=32768 + 128 * n * n + 1024 * draws * (m + n + p),
        max_work=max_work,
        timeout=timeout,
        cancelled=cancelled,
    )
    work.add(steps * (n * n + p) + draws * n**3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a, pairs = adjacency(graph), torch.triu_indices(n, n, 1)
    samples, history, changes = [], [], 0
    for step in range(steps):
        if step % 128 == 0:
            work.add(0)
        chosen = int(torch.randint(m, (), generator=generator))
        i, j = pairs[:, chosen].tolist()
        probability = float((_change(a, i, j, terms) @ theta).sigmoid())
        new = float(torch.rand((), generator=generator)) < probability
        changes += new != bool(a[i, j])
        a[i, j] = a[j, i] = float(new)
        if step >= burn_in and (step - burn_in + 1) % thin == 0:
            selected = a[pairs[0], pairs[1]] > 0
            edges = pairs[:, selected].T.tolist()
            samples.append(
                network(
                    [{"source": graph._labels[i], "target": graph._labels[j]} for i, j in edges],
                    nodes=list(graph._labels),
                    max_memory_mb=graph._budget.limit / 1024**2,
                )
            )
            history.append(statistics(a, terms).tolist())
    values = torch.tensor(history, dtype=torch.float64)
    mixing = []
    for column, term in enumerate(terms):
        series = values[:, column] - values[:, column].mean()
        denom = float(series.square().sum())
        rho = float(series[:-1] @ series[1:]) / denom if denom and draws > 1 else None
        ess = (
            min(float(draws), draws * (1 - rho) / (1 + rho))
            if rho is not None and -1 < rho < 1
            else None
        )
        mixing.append((term, rho, ess))
    edge_counts = [sample.edge_count for sample in samples]
    return ModelResult(
        graphs=samples,
        statistics=as_frame(pd.DataFrame(history, columns=terms)),
        diagnostics=as_frame(
            pd.DataFrame(mixing, columns=["term", "lag1_autocorrelation", "ess_lag1_approx"])
        ),
        metadata={
            "model": "ERGM simulation",
            "method": "single-dyad Gibbs",
            "nodes": n,
            "draws": draws,
            "chain": 1,
            "burn_in": burn_in,
            "thin": thin,
            "seed": seed,
            "terms": terms,
            "change_fraction": changes / steps,
            "empty_full_fraction": sum(e in {0, m} for e in edge_counts) / draws,
            "mixing": "lag-1 diagnostics are descriptive, not a convergence certificate",
            "uncertainty": "simulation Monte Carlo variation",
            **work.metadata(),
        },
    )
