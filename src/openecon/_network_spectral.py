"""Sparse HITS and Katz iterations without a dense adjacency or external solver.

HITS: Kleinberg, Authoritative sources in a hyperlinked environment (1999).
https://www.cs.cornell.edu/home/kleinber/auth.pdf
Katz uses x = beta + alpha A.T x, with a checked sufficient contraction bound.
"""
from collections.abc import Mapping
import math

import torch

from openecon._network_centrality import _direction
from openecon._network_device import execution, metadata
from openecon.networks import _error, _integer, _label, _real


class _Work:
    def __init__(self, graph, maximum, *, edge_sensitive=True):
        self.limit = _integer(maximum, 'max_work', 2**63 - 1)
        self.used = 0
        arcs = graph._arcs._nnz() if edge_sensitive else 0
        graph._guard(256 * graph.node_count + 96 * arcs + 4096)
        self.tick(graph.node_count + arcs)

    def tick(self, count):
        if self.used + count > self.limit:
            _error('work_budget', 'Sparse iteration exceeds max_work; no incomplete vector is returned.')
        self.used += count


def _options(graph, max_iter, tol, max_work, normalization, *, edge_sensitive=True):
    maximum = _integer(max_iter, 'max_iter', 1_000_000)
    tolerance = _real(tol, 'invalid_option', 'tol must be positive and finite.')
    if tolerance <= 0 or normalization not in ('l1', 'l2', 'none'):
        _error('invalid_option', "Use tol > 0 and normalization='l1', 'l2', or 'none'.")
    return maximum, tolerance, _Work(graph, max_work, edge_sensitive=edge_sensitive)


def _normalize(vector, normalization):
    if normalization == 'none' or not vector.numel():
        return vector
    # Raw squares/sums overflow or underflow even when all normalized scores
    # are representable. Divide by the maximum magnitude before either norm.
    maximum = float(vector.abs().max())
    if not math.isfinite(maximum):
        _error('precision', 'Centrality output contains a nonfinite value.')
    if not maximum:
        return vector
    scaled = vector / maximum
    if bool(((vector != 0) & (scaled == 0)).any()):
        _error('precision', 'A nonzero centrality score underflows during normalization scaling.')
    divisor = float(scaled.abs().sum() if normalization == 'l1' else torch.linalg.vector_norm(scaled))
    if not math.isfinite(divisor):
        _error('precision', 'Centrality output normalization exceeds float64 range.')
    result = scaled / divisor if divisor else scaled
    if bool(((scaled != 0) & (result == 0)).any()):
        _error('precision', 'A nonzero normalized centrality score is not representable in float64.')
    return result


def _scaled(graph, selected):
    raw = graph._arcs._values().to(selected)
    scale = float(raw.max()) if raw.numel() else 1.
    values = raw / scale
    if raw.numel() and bool((values <= 0).any()):
        _error('precision', 'Positive edge strengths cannot be scaled without float64 underflow.')
    return graph._arcs._indices().to(selected), values, scale


def hits(graph, *, max_iter=1000, tol=1e-10, normalization='l1', max_work=50_000_000, device='cpu'):
    maximum, tol, work = _options(graph, max_iter, tol, max_work, normalization)
    if normalization == 'none':
        _error('invalid_option', "HITS singular vectors require normalization='l1' or 'l2'.")
    n, count = graph.node_count, graph._arcs._nnz()
    workspace = 256 * n + 96 * count + 4096
    with execution(graph, device, workspace) as selected:
        pairs, weights, scale = _scaled(graph, selected)
        authority = torch.full((n,), 1 / math.sqrt(n) if n else 0., dtype=torch.float64)
        hub = torch.zeros(n, dtype=torch.float64)
        iteration = 0
        change = residual = singular = 0.

        def product(vector, transpose):
            work.tick(n + count)
            output = torch.zeros(n, dtype=torch.float64)
            u, v = (pairs[0], pairs[1]) if transpose else (pairs[1], pairs[0])
            output.index_add_(0, v, weights * vector[u])
            return output

        if not count:
            authority.zero_()
        else:
            for iteration in range(1, maximum + 1):
                hub = product(authority, False)
                norm = float(torch.linalg.vector_norm(hub))
                if not math.isfinite(norm) or norm <= 0:
                    _error('precision', 'HITS has no representable positive normalization.')
                hub /= norm
                following = product(hub, True)
                magnitude = float(torch.linalg.vector_norm(following))
                if not math.isfinite(magnitude) or magnitude <= 0:
                    _error('precision', 'HITS authority normalization exceeds float64 range.')
                following /= magnitude
                change = float(torch.linalg.vector_norm(following - authority))
                authority = following
                if change <= tol:
                    # Verify both singular equations; a small step alone is insufficient.
                    left = product(authority, False)
                    singular = float(torch.linalg.vector_norm(left))
                    hub = left / singular
                    right = product(hub, True)
                    residual = float(torch.linalg.vector_norm(right - singular * authority)) / singular
                    if math.isfinite(residual) and residual <= tol:
                        break
            else:
                _error('nonconvergence', 'HITS did not meet both step and singular-residual tolerance.')
        return graph._frame({'hub': _normalize(hub, normalization).cpu().numpy(),
                             'authority': _normalize(authority, normalization).cpu().numpy()},
                            kind='network_hits', algorithm='scaled sparse HITS power iteration',
                            normalization=normalization, converged=True, iterations=iteration,
                            l2_change=change, relative_singular_residual=residual,
                            singular_value_scaled=singular, weight_scale=scale,
                            weight_semantics='aggregate strength', edgeless='zero scores',
                            uniqueness='not assumed; positive-start dominant singular vectors',
                            work_used=work.used, max_work=work.limit, **metadata(selected, workspace))


def katz(graph, alpha=None, beta=1., *, direction='in', max_iter=1000, tol=1e-10,
         normalization='l2', max_work=50_000_000, device='cpu'):
    automatic = alpha is None
    original_alpha = None if automatic else _real(alpha, 'invalid_option', 'alpha must be finite and nonnegative.')
    if original_alpha is not None and original_alpha < 0:
        _error('invalid_option', 'alpha must be nonnegative.')
    zero_alpha = original_alpha == 0 if not automatic else False
    maximum, tol, work = _options(graph, max_iter, tol, max_work, normalization,
                                  edge_sensitive=not zero_alpha)
    direction = _direction(direction)
    n, count = graph.node_count, graph._arcs._nnz()
    workspace = 256 * n + (0 if zero_alpha else 96 * count) + 4096
    with execution(graph, device, workspace) as selected:
        baseline = torch.empty(n, dtype=torch.float64, device='cpu')
        if isinstance(beta, Mapping):
            if len(beta) != n:
                _error('invalid_option', 'beta must supply exactly one value for every node.')
            visited = set()
            for label, value in beta.items():
                label = _label(label)
                if label not in graph._index:
                    _error('unknown_node', 'beta contains an unknown node.')
                index = graph._index[label]
                value = _real(value, 'invalid_option', 'beta values must be finite and nonnegative.')
                if value < 0:
                    _error('invalid_option', 'beta values must be nonnegative.')
                baseline[index] = value
                visited.add(index)
            if len(visited) != n:
                _error('invalid_option', 'beta must cover all exact node IDs.')
        else:
            value = _real(beta, 'invalid_option', 'beta must be finite and nonnegative.')
            if value < 0:
                _error('invalid_option', 'beta must be nonnegative.')
            baseline.fill_(value)
        if n and not bool((baseline > 0).any()):
            _error('invalid_option', 'At least one beta value must be positive.')
        baseline = baseline.to(selected)
        baseline_scale = float(baseline.max()) if n else 0.
        if zero_alpha:
            # x=beta is exact and independent of every edge strength. In
            # particular, irrelevant edge scaling must not reject this case.
            scale, scaled_alpha, contraction = None, 0., 0.
            active = False
        else:
            pairs, weights, scale = _scaled(graph, selected)
            u, v = (pairs[0], pairs[1]) if direction == 'in' else (pairs[1], pairs[0])
            sums = torch.zeros(n, dtype=torch.float64)
            sums.index_add_(0, v, weights)
            bound = float(sums.max()) if n else 0.
            if automatic:
                scaled_alpha = .85 / bound if bound else 0.
                original_alpha = scaled_alpha / scale
            else:
                scaled_alpha = original_alpha * scale
                if original_alpha > 0 and count and scaled_alpha == 0:
                    _error('precision', 'Positive Katz coefficients underflow float64.')
            contraction = scaled_alpha * bound
            if not math.isfinite(scaled_alpha) or not math.isfinite(contraction) or contraction >= 1:
                _error('invalid_option', 'alpha does not meet the sufficient contraction bound: '
                       'alpha * maximum incoming/outgoing strength must be below one. '
                       'Use alpha=None to select a checked attenuation automatically.')
            coefficients = weights * scaled_alpha
            if scaled_alpha > 0 and count and bool((coefficients <= 0).any()):
                _error('precision', 'Positive Katz coefficients underflow float64.')
            active = bool(n and count and scaled_alpha)
        if active:
            scaled_baseline = baseline / baseline_scale
            if bool(((baseline > 0) & (scaled_baseline == 0)).any()):
                _error('precision', 'Positive beta values cannot be scaled without float64 underflow.')
            baseline = scaled_baseline
        rank = baseline.clone()
        iteration = 0
        change = error_bound = 0.
        for iteration in range(1, maximum + 1) if active else []:
            work.tick(n + count)
            following = baseline.clone()
            following.index_add_(0, v, coefficients * rank[u])
            if not bool(torch.isfinite(following).all()):
                _error('precision', 'Katz scores exceed float64 range.')
            change = float((following - rank).abs().max())
            error_bound = contraction / (1 - contraction) * change
            rank = following
            if error_bound / float(rank.abs().max()) <= tol:
                break
        else:
            if active:
                _error('nonconvergence', 'Katz did not reach its contraction error tolerance.')
        relative_bound = error_bound / float(rank.abs().max()) if active else 0.
        alpha_representable = math.isfinite(original_alpha) and (original_alpha > 0 or scaled_alpha == 0)
        if active and normalization == 'none':
            restored = rank * baseline_scale
            if not bool(torch.isfinite(restored).all()) or bool(((rank > 0) & (restored == 0)).any()):
                _error('precision', 'Unnormalized Katz scores overflow or underflow float64.')
            rank = restored
        return graph._frame({'katz': _normalize(rank, normalization).cpu().numpy()},
                            kind='network_katz', algorithm='sparse checked-contraction Katz iteration',
                            normalization=normalization, orientation=direction, converged=True,
                            iterations=iteration, alpha=original_alpha if alpha_representable else None,
                            alpha_representable=alpha_representable, alpha_scaled=scaled_alpha,
                            alpha_selected_automatically=automatic, contraction_bound=contraction,
                            relative_error_bound_estimate=relative_bound, tol=tol,
                            bound_scope='iteration bound; not a rigorous floating-point certificate',
                            explicit_alpha_scope='conservative sufficient max-strength bound',
                            baseline_scale=baseline_scale,
                            iteration_scale='beta/max(beta)' if active else 'original beta; no edge iteration',
                            weight_scale=scale, weight_semantics='aggregate strength',
                            work_used=work.used, max_work=work.limit, **metadata(selected, workspace))
