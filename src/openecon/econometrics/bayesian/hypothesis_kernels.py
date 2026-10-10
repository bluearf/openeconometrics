"""Native exact restricted normal–inverse-gamma algebra in a prior-whitened chart."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import t_isf

from .core import _log_gamma_increment, _log_scale_increment, matrices


def _error(message):
    raise AnalysisError("invalid_hypothesis", message)


def _finite(*blocks):
    for block in blocks:
        if isinstance(block, torch.Tensor):
            valid = bool(torch.isfinite(block).all())
        else:
            valid = math.isfinite(block)
        if not valid:
            _error("The restricted posterior exceeds representable CPU float64 range.")


def _affine_chart(c, d, prior_factor, center, fixed_coordinates):
    """Solve an explicit affine chart and factor its proper prior precision.

    Coordinate equalities then have structurally zero loading rows, rather than
    acquiring spurious uncertainty from cancellation in an orthogonal QR basis.
    This constructs the statistical free chart; it does not project or clip a
    covariance estimate. Prior diagonal units keep pivot choices dimensionless.
    """
    r, k = c.shape
    free = k - r
    units = torch.linalg.vector_norm(prior_factor, dim=1)
    normalized = c * units[None, :]
    table = normalized.clone()
    permutation = list(range(k))
    for i in range(r):
        trailing = table[i:, i:].abs()
        index = int(torch.argmax(trailing))
        row, column = i + index // (k - i), i + index % (k - i)
        if float(trailing.flatten()[index]) <= 0:
            _error("The affine constraint chart is not representably full rank.")
        if row != i:
            table[[i, row]] = table[[row, i]].clone()
        if column != i:
            table[:, [i, column]] = table[:, [column, i]].clone()
            permutation[i], permutation[column] = permutation[column], permutation[i]
        if i + 1 < r:
            table[i + 1 :, i + 1 :] -= (
                table[i + 1 :, i : i + 1] / table[i, i] * table[i : i + 1, i + 1 :]
            )
    pivots, remaining = permutation[:r], permutation[r:]
    cp, cf = normalized[:, pivots], normalized[:, remaining]
    center_z = center / units
    if not free:
        center_z[pivots] = torch.linalg.solve(cp, d)
        center = center_z * units
        for coordinate, value in fixed_coordinates:
            center[coordinate] = value
        return center, torch.zeros((k, 0), dtype=torch.float64, device="cpu")
    chart = torch.zeros((k, free), dtype=torch.float64, device="cpu")
    chart[remaining] = torch.eye(free, dtype=torch.float64, device="cpu")
    chart[pivots] = -torch.linalg.solve(cp, cf)
    for coordinate, _ in fixed_coordinates:
        # A declared single-coordinate equality removes this coordinate from
        # the statistical free chart exactly, before prior precision is built.
        chart[coordinate] = 0
    center_z[pivots] = torch.linalg.solve(cp, d - cf @ center_z[remaining])
    normalized_factor = prior_factor / units[:, None]
    prior_precision = torch.cholesky_inverse(normalized_factor)
    precision = chart.T @ prior_precision @ chart
    free_units = torch.diag(precision).sqrt()
    factor = torch.linalg.cholesky(precision / free_units[:, None] / free_units[None, :])
    inverse_factor = torch.linalg.solve_triangular(
        factor.T, torch.eye(free, dtype=torch.float64, device="cpu"), upper=True
    )
    loading = units[:, None] * (chart / free_units[None, :] @ inverse_factor)
    _finite(center_z, loading)
    center = center_z * units
    for coordinate, value in fixed_coordinates:
        center[coordinate] = value
    return center, loading


def geometry(prior, constraints, values):
    """Resolve full row rank after constraint-row and prior-coordinate equilibration.

    An explicit affine chart defines a proper free-dimensional whitened prior.
    No covariance subtraction, pseudoinverse, eigenvalue clipping or projection
    is used to manufacture a rank-deficient covariance.
    """
    c = torch.tensor(constraints, dtype=torch.float64, device="cpu")
    d = torch.tensor(values, dtype=torch.float64, device="cpu")
    fixed_coordinates = []
    for row, value in zip(c, d, strict=True):
        positions = row.nonzero().flatten()
        if len(positions) == 1:
            position = int(positions[0])
            fixed_coordinates.append((position, float(value / row[position])))
    m = torch.tensor(prior.mean, dtype=torch.float64, device="cpu")
    v = torch.tensor(prior.scale_matrix, dtype=torch.float64, device="cpu")
    prior_factor = torch.linalg.cholesky(v)
    r, k = c.shape
    row_units = c.abs().amax(dim=1)
    if bool((row_units <= 0).any()):
        _error("Every constraint row must be nonzero and jointly full row rank.")
    c = c / row_units[:, None]
    d = d / row_units
    whitened = c @ prior_factor
    coordinate_units = whitened.abs().amax(dim=1)
    if bool((coordinate_units <= 0).any()):
        _error("Constraint directions are not representable under the proper prior.")
    whitened = whitened / coordinate_units[:, None]
    norm = torch.linalg.vector_norm(whitened, dim=1)
    c = c / coordinate_units[:, None] / norm[:, None]
    d = d / coordinate_units / norm
    whitened = whitened / norm[:, None]
    _finite(c, d, whitened)
    gram = whitened @ whitened.T
    eig = torch.linalg.eigvalsh(gram)
    # This declared rank-resolution gate is dimensionless and invariant to
    # coefficient units and constraint row units. Near-collinear declarations
    # are refused instead of silently changing the null hypothesis dimension.
    floor = 256 * torch.finfo(torch.float64).eps * r * float(eig[-1])
    if float(eig[0]) <= floor:
        _error("Constraint rows are dependent or numerically unresolved in the prior metric.")
    lg = torch.linalg.cholesky(gram)
    difference = d - c @ m
    displacement = whitened.T @ torch.cholesky_solve(difference[:, None], lg)
    center = m + (prior_factor @ displacement).flatten()
    penalty = float(displacement.square().sum()) / 2
    center, basis = _affine_chart(c, d, prior_factor, center, fixed_coordinates)
    _finite(center, basis, penalty)
    return {
        "constraints_normalized": c,
        "values_normalized": d,
        "prior_constraint_factor": lg,
        "center": center,
        "basis": basis,
        "prior_scale_increment": penalty,
        "rank": r,
        "free_dimension": k - r,
        "rank_resolution_floor": floor,
    }


def _density(shape, scale, mean, conditional, c, d):
    """Exact multivariate Student-t log density, with stable gamma/scale ratios."""
    factor = torch.linalg.cholesky(conditional)
    loading = c @ factor
    gram = loading @ loading.T
    lg, status = torch.linalg.cholesky_ex(gram)
    if int(status):
        _error("Posterior constraint density is not representably positive definite.")
    deviation = d - c @ mean
    whitened = torch.linalg.solve_triangular(lg, deviation[:, None], upper=False)
    increment = float(whitened.square().sum()) / 2
    r = len(d)
    value = math.fsum(
        (
            _log_gamma_increment(shape, r, normalizer=scale),
            -r * math.log(2 * math.pi) / 2,
            -float(torch.log(torch.diag(lg)).sum()),
            -(shape + r / 2) * _log_scale_increment(increment, scale),
        )
    )
    _finite(value, increment)
    return value


def restricted_algebra(alternative, constraints, values, prior_odds, alpha):
    """Condition the complete prior, fit its affine null and compute an exact BF01."""
    with torch.device("cpu"):
        g = geometry(alternative.prior, constraints, values)
        basis, center = g["basis"], g["center"]
        state = alternative.state
        y, x = matrices(state.source_values, state.sample_positions, intercept=state.intercept)
        n, k, r, free = len(y), len(alternative.terms), g["rank"], g["free_dimension"]
        a0 = alternative.prior.shape + r / 2
        b0 = alternative.prior.scale + g["prior_scale_increment"]
        residual0 = y - x @ center
        if free:
            w = x @ basis
            precision = torch.eye(free, dtype=torch.float64, device="cpu") + w.T @ w
            lp = torch.linalg.cholesky(precision)
            v = torch.cholesky_inverse(lp)
            m = torch.cholesky_solve((w.T @ residual0)[:, None], lp).flatten()
            residual = residual0 - w @ m
            increment = float(residual.square().sum() + m.square().sum()) / 2
            mean = center + basis @ m
            conditional = basis @ v @ basis.T
            logdet_half = float(torch.log(torch.diag(lp)).sum())
        else:
            v = torch.empty((0, 0), dtype=torch.float64, device="cpu")
            m = torch.empty((0,), dtype=torch.float64, device="cpu")
            mean = center
            conditional = torch.zeros((k, k), dtype=torch.float64, device="cpu")
            increment = float(residual0.square().sum()) / 2
            logdet_half = 0.0
        an, bn = a0 + n / 2, b0 + increment
        _finite(a0, b0, an, bn, increment, mean, conditional)
        if min(b0, bn) <= 0:
            _error("The conditional prior and posterior inverse-gamma scales must be positive.")
        coefficient_scale = conditional * (bn / an)
        # Retain a tiny proper shape when a0+(n+r)/2 rounds to one: subtracting
        # that rounded shape would invent an unavailable inverse-gamma moment.
        moment_denominator = alternative.prior.shape + (n + r - 2) / 2
        variance_mean = bn / moment_denominator
        _finite(moment_denominator, variance_mean)
        covariance = conditional * variance_mean
        critical = t_isf(alpha / 2, 2 * an)
        intervals = [
            [
                float(mean[i]) - critical * math.sqrt(float(coefficient_scale[i, i])),
                float(mean[i]) + critical * math.sqrt(float(coefficient_scale[i, i])),
            ]
            for i in range(k)
        ]
        prior_conditional = basis @ basis.T
        prior_mean = torch.tensor(alternative.prior.mean, dtype=torch.float64, device="cpu")
        prior_v = torch.tensor(alternative.prior.scale_matrix, dtype=torch.float64, device="cpu")
        posterior_mean = torch.tensor(alternative.mean, dtype=torch.float64, device="cpu")
        posterior_v = torch.tensor(
            alternative.conditional_scale_matrix, dtype=torch.float64, device="cpu"
        )
        prior_log_density = _density(
            alternative.prior.shape,
            alternative.prior.scale,
            prior_mean,
            prior_v,
            g["constraints_normalized"],
            g["values_normalized"],
        )
        posterior_log_density = _density(
            alternative.shape,
            alternative.scale,
            posterior_mean,
            posterior_v,
            g["constraints_normalized"],
            g["values_normalized"],
        )
        log_bf = math.fsum((posterior_log_density, -prior_log_density))
        log_null = math.fsum(
            (
                -n * math.log(2 * math.pi) / 2,
                -logdet_half,
                _log_gamma_increment(a0, n, normalizer=b0),
                -an * _log_scale_increment(increment, b0),
            )
        )
        # A stable density ratio is the primary BF. The restricted marginal
        # evidence is independently retained rather than reconstructed as
        # alternative evidence + log BF (a tautological cache).
        _finite(log_bf, log_null, coefficient_scale)
        if covariance is not None:
            _finite(covariance)
        if log_bf > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            bf, status = None, "overflow"
        elif log_bf < math.log(float.fromhex("0x0.0000000000001p-1022")):
            bf, status = None, "underflow"
        else:
            bf, status = math.exp(log_bf), "finite"
        probability, alternative_probability, log_odds = None, None, None
        if prior_odds is not None:
            log_odds = math.fsum((math.log(prior_odds), log_bf))
            probability = (
                1 / (1 + math.exp(-log_odds))
                if log_odds >= 0
                else math.exp(log_odds) / (1 + math.exp(log_odds))
            )
            alternative_probability = (
                math.exp(-log_odds) / (1 + math.exp(-log_odds))
                if log_odds >= 0
                else 1 / (1 + math.exp(log_odds))
            )
        null = {
            "schema_version": "openecon.restricted_nig_posterior.v1",
            "terms": list(alternative.terms),
            "constraint_rank": r,
            "free_dimension": free,
            "prior_mean": center.tolist(),
            "prior_conditional_scale_matrix": prior_conditional.tolist(),
            "prior_shape": a0,
            "prior_scale": b0,
            "prior_scale_increment": g["prior_scale_increment"],
            "basis": basis.tolist(),
            "free_mean": m.tolist(),
            "free_conditional_scale_matrix": v.tolist(),
            "mean": mean.tolist(),
            "conditional_scale_matrix": conditional.tolist(),
            "shape": an,
            "scale": bn,
            "data_scale_increment": increment,
            "variance_moment_denominator": moment_denominator,
            "degrees_of_freedom": 2 * an,
            "coefficient_scale_matrix": coefficient_scale.tolist(),
            "coefficient_covariance": covariance.tolist(),
            "variance_mean": variance_mean,
            "credible_intervals": intervals,
            "alpha": alpha,
            "log_marginal_likelihood": log_null,
        }
        return {
            "null_posterior": null,
            "log_bayes_factor_null_alternative": log_bf,
            "bayes_factor_null_alternative": bf,
            "bayes_factor_status": status,
            "constraint_prior_log_density": prior_log_density,
            "constraint_posterior_log_density": posterior_log_density,
            "density_coordinate_convention": "prior-equilibrated constraint rows; common row Jacobian cancels",
            "log_posterior_model_odds_null_alternative": log_odds,
            "posterior_probability_null": probability,
            "posterior_probability_alternative": alternative_probability,
        }


def draw_arrays(null, alternative, target, count, seed, query):
    """Exact joint beta/sigma² draws in the affine support; no ambient RNG writes."""
    with torch.device("cpu"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = null if target == "null" else alternative
        gamma = torch._standard_gamma(
            torch.full((count,), selected["shape"], dtype=torch.float64, device="cpu"),
            generator=generator,
        )
        sigma = selected["scale"] / gamma
        mean = torch.tensor(selected["mean"], dtype=torch.float64, device="cpu")
        if target == "null":
            free = null["free_dimension"]
            if free:
                basis = torch.tensor(null["basis"], dtype=torch.float64, device="cpu")
                factor = torch.linalg.cholesky(
                    torch.tensor(
                        null["free_conditional_scale_matrix"], dtype=torch.float64, device="cpu"
                    )
                )
                normal = torch.randn(
                    (count, free), dtype=torch.float64, device="cpu", generator=generator
                )
                beta = mean + sigma.sqrt()[:, None] * (normal @ factor.T @ basis.T)
            else:
                beta = mean.expand(count, len(mean)).clone()
        else:
            factor = torch.linalg.cholesky(
                torch.tensor(
                    selected["conditional_scale_matrix"], dtype=torch.float64, device="cpu"
                )
            )
            normal = torch.randn(
                (count, len(mean)), dtype=torch.float64, device="cpu", generator=generator
            )
            beta = mean + sigma.sqrt()[:, None] * (normal @ factor.T)
        means, outcomes = None, None
        if query is not None:
            design = torch.tensor(query, dtype=torch.float64, device="cpu")
            means = beta @ design.T
            noise = torch.randn(means.shape, dtype=torch.float64, device="cpu", generator=generator)
            outcomes = means + sigma.sqrt()[:, None] * noise
            _finite(means, outcomes)
        _finite(sigma, beta)
        if bool((sigma <= 0).any()):
            _error("A joint variance draw is not representably positive.")
        return {
            "beta": beta.tolist(),
            "sigma_squared": sigma.tolist(),
            "mean_draws": None if means is None else means.tolist(),
            "outcome_draws": None if outcomes is None else outcomes.tolist(),
        }
