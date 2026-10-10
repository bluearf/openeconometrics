"""Exact finite proper-NIG model mixtures; no model search or external runtime."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import chi2_isf, gamma_q, t_cdf, t_ppf
from .core import posterior_algebra

DT = torch.float64
MAX_QUANTILE_STEPS = 2048


def tensor(value):
    return torch.tensor(value, dtype=DT, device="cpu")


def finite(value, name):
    if not math.isfinite(value):
        raise AnalysisError("numerical_failure", f"{name} exceeds finite float64 support.")
    return value


def normalize(logs):
    """Normalize positive model masses without truncating a supported model."""
    maximum = max(logs)
    shifted = [math.exp(v - maximum) for v in logs]
    if any(v == 0 for v in shifted):
        raise AnalysisError(
            "numerical_failure", "A positive model mass underflows float64; no model is discarded."
        )
    total = math.fsum(shifted)
    logtotal = maximum + math.log(total)
    weights = [v / total for v in shifted]
    if any(v == 0 or not math.isfinite(v) for v in weights):
        raise AnalysisError(
            "numerical_failure", "A positive normalized model weight is unrepresentable."
        )
    return weights, [v - logtotal for v in logs], finite(logtotal, "Model normalizer")


def component(y, x, prior, indices, total):
    solved = posterior_algebra(y, x[:, indices], prior)
    # The core's mean field is the t location. Mean existence is a separate law.
    shape = prior.shape
    n = len(y)
    mean_exists = math.fsum((shape, (n - 1) / 2)) > 0
    variance_exists = math.fsum((shape, (n - 2) / 2)) > 0
    sigma_variance_exists = math.fsum((shape, (n - 4) / 2)) > 0
    location = torch.zeros(total, dtype=DT, device="cpu")
    location[indices] = tensor(solved["mean"])
    scale = torch.zeros((total, total), dtype=DT, device="cpu")
    block = tensor(solved["coefficient_scale_matrix"])
    scale[tensor(indices).long()[:, None], tensor(indices).long()[None, :]] = block
    covariance = None
    if variance_exists:
        covariance = torch.zeros((total, total), dtype=DT, device="cpu")
        covariance[tensor(indices).long()[:, None], tensor(indices).long()[None, :]] = tensor(
            solved["coefficient_covariance"]
        )
    sigma_variance = None
    if sigma_variance_exists:
        d1 = math.fsum((shape, (n - 2) / 2))
        d2 = math.fsum((shape, (n - 4) / 2))
        sigma_sd = (solved["scale"] / d1) / math.sqrt(d2)
        sigma_variance = finite(sigma_sd * sigma_sd, "Variance-parameter posterior variance")
        if sigma_variance <= 0:
            raise AnalysisError(
                "numerical_failure",
                "A positive variance-parameter posterior variance is not float64 representable.",
            )
    return dict(
        indices=indices,
        posterior=solved,
        location=location.tolist(),
        scale_matrix=scale.tolist(),
        covariance=None if covariance is None else covariance.tolist(),
        mean_exists=mean_exists,
        variance_exists=variance_exists,
        sigma_variance_exists=sigma_variance_exists,
        sigma_variance=sigma_variance,
    )


def cdf(value, laws, weights, *, left=False):
    """CDF or left CDF for a finite t mixture with explicit point atoms."""
    values = []
    for law, weight in zip(laws, weights, strict=True):
        location, scale, df = law
        if scale == 0:
            probability = float(value > location if left else value >= location)
        else:
            probability = t_cdf((value - location) / scale, df)
        values.append(weight * probability)
    return math.fsum(values)


def ig_cdf(value, laws, weights):
    if value <= 0:
        return 0.0
    return math.fsum(w * gamma_q(a, b / value) for (a, b), w in zip(laws, weights, strict=True))


def quantile(probability, laws, weights, *, inverse_gamma=False):
    """Generalized inverse CDF, retaining atoms and bounded numerical refusals."""
    if inverse_gamma:
        points = [
            finite(2 * b / chi2_isf(probability, 2 * a), "Component IG quantile") for a, b in laws
        ]

        def cumulative(v):
            return ig_cdf(v, laws, weights)
    else:
        atoms = sorted(set(location for location, scale, _ in laws if scale == 0))
        for atom in atoms:
            if cdf(atom, laws, weights, left=True) <= probability <= cdf(atom, laws, weights):
                return atom
        points = [
            location
            if scale == 0
            else finite(location + scale * t_ppf(probability, df), "Component t quantile")
            for location, scale, df in laws
        ]

        def cumulative(v):
            return cdf(v, laws, weights)

    def accepted(value):
        right = cumulative(value)
        left = right if inverse_gamma else cdf(value, laws, weights, left=True)
        if max(0.0, left - probability, probability - right) > 2e-10:
            raise AnalysisError(
                "numerical_failure", "Mixture quantile cannot resolve its CDF target in float64."
            )
        return value

    lo, hi = min(points), max(points)
    if lo == hi:
        return accepted(lo)
    # Every component p-quantile brackets the mixture p-quantile. Expand only
    # for an endpoint CDF roundoff, preserving the all-model mixture target.
    for _ in range(16):
        if cumulative(lo) <= probability:
            break
        lo = math.nextafter(lo, -math.inf)
    for _ in range(16):
        if cumulative(hi) >= probability:
            break
        hi = math.nextafter(hi, math.inf)
    if (
        not math.isfinite(lo)
        or not math.isfinite(hi)
        or cumulative(lo) > probability
        or cumulative(hi) < probability
    ):
        raise AnalysisError("numerical_failure", "Mixture quantile has no resolved finite bracket.")
    for _ in range(MAX_QUANTILE_STEPS):
        middle = lo / 2 + hi / 2
        if middle == lo or middle == hi:
            return accepted(hi)
        probability_at = cumulative(middle)
        if probability_at < probability:
            lo = middle
        else:
            hi = middle
        unit = max(abs(lo), abs(hi))
        if unit and abs(lo / unit - hi / unit) <= 2e-13:
            right = cumulative(hi)
            left = right if inverse_gamma else cdf(hi, laws, weights, left=True)
            if max(0.0, left - probability, probability - right) <= 2e-10:
                return accepted(hi)
    raise AnalysisError(
        "numerical_failure", "Mixture quantile exhausted its bounded CDF inversion work."
    )


def marginal_laws(components, coordinate):
    return [
        (
            v["location"][coordinate],
            math.sqrt(v["scale_matrix"][coordinate][coordinate]),
            v["posterior"]["degrees_of_freedom"],
        )
        for v in components
    ]


def mixture_moments(locations, covariances, weights):
    """Centered full within/between covariance, avoiding raw second-moment cancellation."""
    means = tensor(locations)
    w = tensor(weights)
    mean = (w[:, None] * means).sum(0)
    centered = means - mean
    within = (w[:, None, None] * tensor(covariances)).sum(0)
    between = torch.einsum("m,mi,mj->ij", w, centered, centered)
    total = within + between
    if not all(bool(torch.isfinite(v).all()) for v in (mean, within, between, total)):
        raise AnalysisError(
            "numerical_failure", "Full model-mixture moments exceed float64 support."
        )
    if bool((total.diag() <= 0).any()):
        raise AnalysisError(
            "numerical_failure", "A positive full mixture marginal variance is unrepresentable."
        )
    return mean.tolist(), within.tolist(), between.tolist(), total.tolist()


def aggregate(components, prior_odds, optional, forced, alpha):
    logprior = [math.log(v) for v in prior_odds]
    prior_weights, normalized_prior, _ = normalize(logprior)
    evidence = [v["posterior"]["log_marginal_likelihood"] for v in components]
    weights, logweights, normalizer = normalize(
        [a + b for a, b in zip(evidence, normalized_prior, strict=True)]
    )
    k, p = len(components[0]["location"]), len(optional)
    locations = [v["location"] for v in components]
    present = [[bool(mask & (1 << j)) for j in range(p)] for mask in range(len(components))]
    inclusion = [
        math.fsum(w * row[j] for w, row in zip(weights, present, strict=True)) for j in range(p)
    ]
    prior_inclusion = [
        math.fsum(w * row[j] for w, row in zip(prior_weights, present, strict=True))
        for j in range(p)
    ]
    inclusion_c = [
        [
            math.fsum(
                w * (row[i] - inclusion[i]) * (row[j] - inclusion[j])
                for w, row in zip(weights, present, strict=True)
            )
            for j in range(p)
        ]
        for i in range(p)
    ]
    mean_exists = [
        all(v["mean_exists"] or coordinate not in v["indices"] for v in components)
        for coordinate in range(k)
    ]
    variance_exists = [
        all(v["variance_exists"] or coordinate not in v["indices"] for v in components)
        for coordinate in range(k)
    ]
    means = [
        finite(
            math.fsum(w * v["location"][j] for w, v in zip(weights, components, strict=True)),
            "Mixture coefficient mean",
        )
        if mean_exists[j]
        else None
        for j in range(k)
    ]
    variances = [
        finite(
            math.fsum(
                w
                * (
                    (0.0 if j not in v["indices"] else v["covariance"][j][j])
                    + (v["location"][j] - means[j]) ** 2
                )
                for w, v in zip(weights, components, strict=True)
            ),
            "Mixture coefficient variance",
        )
        if variance_exists[j]
        else None
        for j in range(k)
    ]
    if any(v is not None and v <= 0 for v in variances):
        raise AnalysisError(
            "numerical_failure", "A positive mixture coefficient variance is unrepresentable."
        )
    within = between = covariance = None
    if all(variance_exists):
        _, within, between, covariance = mixture_moments(
            locations, [v["covariance"] for v in components], weights
        )
    sigma_mean_exists = all(v["variance_exists"] for v in components)
    sigma_mean = (
        finite(
            math.fsum(
                w * v["posterior"]["variance_mean"]
                for w, v in zip(weights, components, strict=True)
            ),
            "Mixture variance mean",
        )
        if sigma_mean_exists
        else None
    )
    joint_mean = [*means, sigma_mean]
    joint_within = joint_between = joint_cov = None
    if all(v["sigma_variance_exists"] for v in components):
        full_locations, full_covariances = [], []
        for v in components:
            full_locations.append([*v["location"], v["posterior"]["variance_mean"]])
            cm = torch.zeros((k + 1, k + 1), dtype=DT, device="cpu")
            cm[:k, :k] = tensor(v["covariance"])
            cm[-1, -1] = v["sigma_variance"]
            full_covariances.append(cm.tolist())
        _, joint_within, joint_between, joint_cov = mixture_moments(
            full_locations, full_covariances, weights
        )
    limits = [
        [
            quantile(alpha / 2, marginal_laws(components, j), weights),
            quantile(1 - alpha / 2, marginal_laws(components, j), weights),
        ]
        for j in range(k)
    ]
    atoms = [
        math.fsum(w for w, v in zip(weights, components, strict=True) if j not in v["indices"])
        for j in range(k)
    ]
    sigma_laws = [(v["posterior"]["shape"], v["posterior"]["scale"]) for v in components]
    return dict(
        log_model_evidence=evidence,
        log_model_prior=normalized_prior,
        log_model_weights=logweights,
        model_weights=weights,
        log_mixture_evidence=normalizer,
        prior_inclusion=prior_inclusion,
        posterior_inclusion=inclusion,
        inclusion_covariance=inclusion_c,
        coefficient_mean=means,
        coefficient_variance=variances,
        mean_exists=mean_exists,
        variance_exists=variance_exists,
        within_coefficient_covariance=within,
        between_coefficient_covariance=between,
        coefficient_covariance=covariance,
        joint_mean=joint_mean,
        joint_covariance=joint_cov,
        within_joint_covariance=joint_within,
        between_joint_covariance=joint_between,
        variance_mean=sigma_mean,
        variance_mean_exists=sigma_mean_exists,
        coefficient_quantiles=limits,
        coefficient_atom_zero=atoms,
        sigma2_quantiles=[
            quantile(alpha / 2, sigma_laws, weights, inverse_gamma=True),
            quantile(1 - alpha / 2, sigma_laws, weights, inverse_gamma=True),
        ],
    )
