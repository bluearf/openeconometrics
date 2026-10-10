"""Moreira/AMS probabilities and Mikusheva whole-line inversion, native CPU64.

The maximum eigenvalue is null-invariant. Inversion searches a bounded LR
critical value, then solves a homogeneous quadratic inequality. No coefficient
grid, finite search window or Wald replacement is involved. Eigen-gap geometry
avoids subtracting two large almost-equal eigenvalues under strong instruments.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import chi2_isf, chi2_sf, gauss_legendre

DTYPE, DEVICE = torch.float64, "cpu"
EPS = torch.finfo(DTYPE).eps


def tensor(value):
    return torch.tensor(value, dtype=DTYPE, device=DEVICE)


def fail(message, code="numerical_failure"):
    raise AnalysisError(code, message)


def probability(statistic, concentration, k, options):
    """Conditional upper tail, with a successive-quadrature error estimate.

    This error estimate is a convergence diagnostic, not a rigorous integration
    bound. The sine transformation removes the angular endpoint singularity.
    The denominator uses cos² + LR/(LR+QT) sin² to remain stable for very large QT.
    """
    if statistic <= 0:
        return dict(value=1.0, error_estimate=0.0, order=0, evaluations=0)
    if concentration < 0 or not math.isfinite(statistic + concentration):
        fail("CLR statistics must be finite and nonnegative.")
    if k == 1 or concentration == 0:
        return dict(
            value=chi2_sf(statistic, k if concentration == 0 else 1),
            error_estimate=0.0,
            order=0,
            evaluations=0,
        )
    a = (k - 1) / 2
    normalization = math.exp(math.lgamma(a + 0.5) - math.lgamma(a) - math.lgamma(0.5))
    fraction = statistic / (statistic + concentration)
    # Near LR=0 the tail changes in a narrow layer around cos(theta)=0.
    # Integrate complementary angle u directly on geometric panels, so a tiny
    # angle remains representable without subtracting it from pi/2.
    boundaries = [0.0, math.pi / 2]
    if statistic < 0.01:
        delta = max(1e-8, math.sqrt(min(statistic, fraction)))
        boundaries = [0.0]
        while delta < math.pi / 2:
            boundaries.append(delta)
            delta *= 4
        boundaries.append(math.pi / 2)
    previous, evaluations = None, 0
    order = 32
    while order <= options["max_order"]:
        nodes, weights = gauss_legendre(order)
        contributions = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            u = left + (nodes + 1) * ((right - left) / 2)
            sine, cosine = u.cos(), u.sin()
            density = 2 * normalization * sine.pow(k - 2)
            arguments = statistic / (cosine.square() + fraction * sine.square())
            values = tensor([chi2_sf(float(v), k) for v in arguments])
            contributions.append(float(weights @ (density * values)) * ((right - left) / 2))
        value = math.fsum(contributions)
        evaluations += order * (len(boundaries) - 1)
        if not -64 * EPS <= value <= 1 + 64 * EPS:
            fail("CLR conditional probability is outside its probability domain.")
        value = min(1.0, max(0.0, value))
        if previous is not None:
            difference = abs(value - previous)
            # Relative tail accuracy also protects confidence levels near one.
            if difference <= options["probability_tolerance"] * max(value, 1e-300):
                return dict(
                    value=value,
                    error_estimate=difference,
                    order=order,
                    evaluations=evaluations,
                )
        previous, order = value, order * 2
    fail("CLR conditional probability quadrature did not converge within max_order.")


def geometry(rows, spec, options):
    """Normalize reporting units, partial included controls, retain full joint Ω."""
    n = len(rows)
    block = tensor(rows)
    responses = block[:, :2]
    c = block[:, 2 : 2 + len(spec["x"])]
    z = block[:, 2 + len(spec["x"]) :]
    if options["intercept"]:
        c = torch.cat([torch.ones((n, 1), dtype=DTYPE, device=DEVICE), c], 1)
    width = c.shape[1] + z.shape[1]
    if n < max(8, width + 2):
        fail(
            "CLR needs at least eight complete rows and two reduced-form residual df.",
            "insufficient_observations",
        )
    w = torch.cat([c, z], 1)
    column_scales = w.abs().amax(0)
    if bool((column_scales == 0).any()):
        fail(
            "Included controls and excluded instruments must have nonzero columns.",
            "singular_design",
        )
    normalized = w / column_scales
    if float(torch.linalg.svdvals(normalized).min()) <= 1e-10 * math.sqrt(n):
        fail(
            "Controls and excluded instruments must have full rank after normalization.",
            "singular_design",
        )
    if c.shape[1]:
        qc = torch.linalg.qr(normalized[:, : c.shape[1]], mode="reduced")[0]
        responses = responses - qc @ (qc.T @ responses)
        z = normalized[:, c.shape[1] :] - qc @ (qc.T @ normalized[:, c.shape[1] :])
    else:
        z = normalized
    qz = torch.linalg.qr(z, mode="reduced")[0]
    if options["omega"] is None:
        scales = responses.abs().amax(0)
        if bool((scales == 0).any()):
            fail(
                "Estimated reduced-form covariance cannot have a zero response variance.",
                "invalid_covariance",
            )
    else:
        scales = tensor([math.sqrt(options["omega"][j][j]) for j in range(2)])
    response = responses / scales
    projected = qz.T @ response
    df = n - width
    if options["omega"] is None:
        innovation = response - qz @ projected
        omega = innovation.T @ innovation / df
    else:
        omega = tensor(options["omega"]) / scales[:, None] / scales[None, :]
    diagonal = omega.diagonal()
    if bool((diagonal <= 0).any()) or not bool(torch.isfinite(omega).all()):
        fail(
            "The full reduced-form covariance must be finite positive definite.",
            "invalid_covariance",
        )
    correlation = omega / diagonal.sqrt()[:, None] / diagonal.sqrt()[None, :]
    if float(torch.linalg.eigvalsh(correlation).min()) <= 1e-10:
        fail(
            "The full reduced-form covariance is singular or numerically unresolved.",
            "invalid_covariance",
        )
    lower = torch.linalg.cholesky(omega)
    inverse_lower = torch.linalg.solve_triangular(
        lower, torch.eye(2, dtype=DTYPE, device=DEVICE), upper=False
    )
    whitened = projected @ inverse_lower.T
    a = whitened.T @ whitened
    gap = math.hypot(float(a[0, 0] - a[1, 1]), 2 * float(a[0, 1]))
    maximum = (float(a.trace()) + gap) / 2
    if not math.isfinite(maximum) or maximum > 1e150:
        fail("CLR normalized concentration exceeds its float64 numerical domain.")
    # An analytic canonical rotation avoids platform-dependent eigenvector signs
    # in saved state; both directions remain full Ω-whitened physical directions.
    angle = 0.5 * math.atan2(2 * float(a[0, 1]), float(a[0, 0] - a[1, 1]))
    sine, cosine = math.sin(angle), math.cos(angle)
    vectors = tensor([[-sine, cosine], [cosine, sine]])
    # Rows map a=(beta_normalized,1) to minimum/maximum whitened eigendirections.
    directions = vectors.T @ inverse_lower
    coefficient_unit = float(scales[0] / scales[1])
    if not math.isfinite(coefficient_unit) or coefficient_unit <= 0:
        fail("CLR coefficient reporting-unit conversion is not representable in float64.")
    gram = projected.T @ projected
    physical_omega = omega * scales[:, None] * scales[None, :]
    physical_gram = gram * scales[:, None] * scales[None, :]
    if not bool(torch.isfinite(physical_omega).all()) or bool(
        (physical_omega.diagonal() <= 0).any()
    ):
        fail("Reduced-form covariance is not representable in original float64 units.")
    return dict(
        n=n,
        k=z.shape[1],
        controls=c.shape[1],
        df_reduced_form=df,
        response_scales=scales.tolist(),
        coefficient_unit=coefficient_unit,
        normalized_covariance=omega.tolist(),
        reduced_form_covariance=physical_omega.tolist(),
        normalized_projected_gram=gram.tolist(),
        projected_gram=physical_gram.tolist(),
        whitened_gram=a.tolist(),
        eigen_directions=directions.tolist(),
        maximum_eigenvalue=maximum,
        eigen_gap=gap,
        omega_convention=("supplied_known" if options["omega"] is not None else "residual_df"),
    )


def critical(g, options):
    """Find LR* in [chi²1(alpha), chi²k(alpha)], instead of subtracting near-M C."""
    alpha, m, k = 1 - options["confidence"], g["maximum_eigenvalue"], g["k"]
    upper = chi2_isf(alpha, k)
    if m <= upper:
        # At C=0, p(M;0)>=alpha; all QT>=0 are accepted, including QT=0.
        return dict(
            branch="all_real_small_maximum",
            lr=None,
            lr_bracket=None,
            conditioning_threshold=None,
            iterations=0,
            evaluations=0,
            trace=[],
        )
    lower = chi2_isf(alpha, 1)
    if k == 1:
        return dict(
            branch="analytic_one_instrument",
            lr=lower,
            lr_bracket=[lower, lower],
            conditioning_threshold=m - lower,
            iterations=0,
            evaluations=0,
            trace=[],
        )
    trace, evaluations = [], 0
    for iteration in range(options["max_iterations"]):
        middle = lower + (upper - lower) / 2
        p = probability(middle, m - middle, k, options)
        evaluations += p["evaluations"]
        trace.append(dict(lr=middle, **p))
        if p["value"] > alpha:
            lower = middle
        else:
            upper = middle
        if upper - lower <= options["root_tolerance"] * max(1.0, abs(middle)):
            value = lower + (upper - lower) / 2
            return dict(
                branch="monotone_conditional_root",
                lr=value,
                lr_bracket=[lower, upper],
                conditioning_threshold=m - value,
                iterations=iteration + 1,
                evaluations=evaluations,
                trace=trace,
            )
    fail("CLR inversion root did not converge within max_iterations.")


def quadratic(g, lr):
    """The normalized inequality a'H a>=0, assembled from stable eigengap."""
    v = tensor(g["eigen_directions"])
    h = v.T @ torch.diag(tensor([lr - g["eigen_gap"], lr])) @ v
    return ((h + h.T) / 2).tolist()


def _roots(h, unit):
    a, b, c = h[0][0], 2 * h[0][1], h[1][1]
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0 or abs(a) <= 128 * EPS * scale:
        fail(
            "CLR leading quadratic coefficient is numerically unresolved; no tail topology asserted.",
            "unresolved_topology",
        )
    a, b, c = a / scale, b / scale, c / scale
    discriminant = math.fsum([b * b, -4 * a * c])
    if discriminant <= 128 * EPS * (b * b + abs(4 * a * c)):
        fail("CLR quadratic roots are numerically unresolved.", "unresolved_topology")
    q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
    roots = sorted([q / a, c / q])
    roots = [r * unit for r in roots]
    if any(not math.isfinite(r) for r in roots):
        fail("CLR confidence endpoints are not representable in original float64 units.")
    return roots, a > 0


def intervals(g, root):
    """Global interval geometry plus endpoint brackets from numerical LR* brackets."""
    if root["lr"] is None:
        return dict(
            topology="all_real",
            intervals=[[None, None]],
            endpoint_brackets=[[None, None]],
            normalized_inequality=None,
        )
    lo, hi = root["lr_bracket"]
    gap = g["eigen_gap"]
    margin = 128 * EPS * max(gap, hi)
    if lo - gap > margin:
        return dict(
            topology="all_real",
            intervals=[[None, None]],
            endpoint_brackets=[[None, None]],
            normalized_inequality=quadratic(g, root["lr"]),
        )
    if hi - gap >= -margin:
        fail(
            "CLR all-real versus interval topology is unresolved at the inversion accuracy.",
            "unresolved_topology",
        )
    matrices = [quadratic(g, x) for x in (lo, root["lr"], hi)]
    solutions = [_roots(h, g["coefficient_unit"]) for h in matrices]
    if len({s[1] for s in solutions}) != 1:
        fail(
            "CLR bounded versus unbounded topology is unresolved at the inversion accuracy.",
            "unresolved_topology",
        )
    roots, outside = solutions[1]
    brackets = [[min(s[0][j] for s in solutions), max(s[0][j] for s in solutions)] for j in (0, 1)]
    return dict(
        topology="two_unbounded_rays" if outside else "bounded_interval",
        intervals=[[None, roots[0]], [roots[1], None]] if outside else [roots],
        endpoint_brackets=[[None, brackets[0]], [brackets[1], None]] if outside else [brackets],
        normalized_inequality=matrices[1],
    )


def test(g, null, options):
    # [beta,unit] is proportional to (beta_normalized,1); normalize first to
    # avoid overflow at large finite nulls or very small original response units.
    unit = g["coefficient_unit"]
    scale = max(abs(null), unit)
    a = tensor([null / scale, unit / scale])
    direction = tensor(g["eigen_directions"]) @ a
    normalization = float(direction.abs().max())
    if not math.isfinite(normalization) or normalization == 0:
        fail("CLR null direction is outside its float64 numerical domain.")
    direction = direction / normalization
    squares = direction.square()
    fraction = float(squares[0] / squares.sum())
    statistic = g["eigen_gap"] * fraction
    conditioning = max(0.0, g["maximum_eigenvalue"] - statistic)
    p = probability(statistic, conditioning, g["k"], options)
    return dict(
        null=null,
        statistic=statistic,
        conditioning_statistic=conditioning,
        conditional_p_value=p["value"],
        probability_error_estimate=p["error_estimate"],
        quadrature_order=p["order"],
        quadrature_evaluations=p["evaluations"],
        accepted=p["value"] >= 1 - options["confidence"],
    )
