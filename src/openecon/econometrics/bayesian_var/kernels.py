"""Native float64 MNIW algebra; no equationwise NIG or elliptical-t substitution."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import t_isf


def tensor(value):
    return torch.tensor(value, dtype=torch.float64, device="cpu")


def gram(a):
    """Compute each symmetric inner product once, retaining exactly mirrored entries."""
    k = a.shape[1]
    out = torch.empty((k, k), dtype=torch.float64, device="cpu")
    for i in range(k):
        for j in range(i + 1):
            out[i, j] = out[j, i] = torch.dot(a[:, i], a[:, j])
    return out


def spd(a, name, *, code="invalid_prior"):
    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
        or not bool(torch.isfinite(a).all())
        or not torch.equal(a, a.T)
    ):
        raise AnalysisError(code, f"{name} must be finite, square and exactly symmetric.")
    try:
        return torch.linalg.cholesky(a)
    except RuntimeError as exc:
        raise AnalysisError(
            code, f"{name} must be positive definite; no jitter/repair is applied."
        ) from exc


def inverse_factor(chol):
    return torch.linalg.solve_triangular(
        chol, torch.eye(len(chol), dtype=torch.float64, device="cpu"), upper=False
    )


def logdet(chol):
    return 2 * float(torch.log(torch.diag(chol)).sum())


def loggamma_m(a, m):
    return m * (m - 1) / 4 * math.log(math.pi) + math.fsum(math.lgamma(a - j / 2) for j in range(m))


def vech(a):
    return [float(a[i, j]) for j in range(len(a)) for i in range(j, len(a))]


def covariance_sigma(scale, nu):
    m = len(scale)
    if nu <= m + 3:
        return None
    positions = [(i, j) for j in range(m) for i in range(j, m)]
    den = (nu - m) * (nu - m - 1) ** 2 * (nu - m - 3)
    return [
        [
            (
                2 * float(scale[i, j]) * float(scale[k, other_j])
                + (nu - m - 1)
                * (
                    float(scale[i, k]) * float(scale[j, other_j])
                    + float(scale[i, other_j]) * float(scale[j, k])
                )
            )
            / den
            for k, other_j in positions
        ]
        for i, j in positions
    ]


def posterior_algebra(y, x, prior, alpha):
    n, m = y.shape
    k = x.shape[1]
    m0, v0, s0 = tensor(prior.mean), tensor(prior.row_scale), tensor(prior.innovation_scale)
    if m0.shape != (k, m) or v0.shape != (k, k) or s0.shape != (m, m):
        raise AnalysisError(
            "invalid_prior", "Prior matrices disagree with the complete ordered lag design."
        )
    c0, cs0 = spd(v0, "prior row scale"), spd(s0, "prior innovation scale")
    i0 = inverse_factor(c0)
    precision0 = gram(i0)
    precision = precision0 + gram(x)
    ck = spd(precision, "posterior precision", code="numeric_failure")
    vn = gram(inverse_factor(ck))
    mn = torch.cholesky_solve(precision0 @ m0 + x.T @ y, ck)
    sn = s0 + gram(y - x @ mn) + gram(i0 @ (mn - m0))
    csn = spd(sn, "posterior innovation scale", code="numeric_failure")
    nu = prior.degrees_of_freedom + n
    df = nu - m + 1
    coefficient_scale = torch.outer(torch.diag(vn), torch.diag(sn)) / df
    critical = t_isf(alpha / 2, df)
    ci = torch.stack(
        (mn - critical * coefficient_scale.sqrt(), mn + critical * coefficient_scale.sqrt()), dim=2
    )
    sigma_mean = sn / (nu - m - 1) if nu > m + 1 else None
    coefficient_covariance = torch.kron(sigma_mean, vn).tolist() if sigma_mean is not None else None
    sigma_covariance = covariance_sigma(sn, nu)
    joint_mean = mn.T.reshape(-1).tolist() + vech(sigma_mean) if sigma_mean is not None else None
    joint_covariance = None
    if sigma_covariance is not None:
        q, r = k * m, len(sigma_covariance)
        joint_covariance = [coefficient_covariance[i] + [0.0] * r for i in range(q)]
        joint_covariance += [[0.0] * q + row for row in sigma_covariance]
    log_evidence = (
        -n * m / 2 * math.log(math.pi)
        - m / 2 * (logdet(ck) + logdet(c0))
        + prior.degrees_of_freedom / 2 * logdet(cs0)
        - nu / 2 * logdet(csn)
        + loggamma_m(nu / 2, m)
        - loggamma_m(prior.degrees_of_freedom / 2, m)
    )
    output = {
        "location": mn.tolist(),
        "row_scale": vn.tolist(),
        "precision": precision.tolist(),
        "innovation_scale": sn.tolist(),
        "degrees_of_freedom": nu,
        "scalar_degrees_of_freedom": df,
        "coefficient_mean": mn.tolist() if nu > m else None,
        "coefficient_covariance": coefficient_covariance,
        "innovation_mean": None if sigma_mean is None else sigma_mean.tolist(),
        "innovation_covariance": sigma_covariance,
        "joint_mean": joint_mean,
        "joint_covariance": joint_covariance,
        "coefficient_intervals": ci.tolist(),
        "log_marginal_likelihood": log_evidence,
    }
    from .admission import metadata_size

    try:
        metadata_size(output)
    except AnalysisError as exc:
        raise AnalysisError(
            "numeric_failure",
            "Complete posterior output overflowed; no truncation/repair is applied.",
        ) from exc
    return output


def matrix_t_logpdf(value, location, precision, scale, nu):
    b, mean, k, s = map(tensor, (value, location, precision, scale))
    q, m = b.shape
    delta = b - mean
    ck, cs = spd(k, "row precision"), spd(s, "innovation scale")
    # Whiten by precision Cholesky transpose; this is a matrix-t determinant.
    updated = s + gram(ck.T @ delta)
    cu = spd(updated, "matrix-t updated scale")
    return (
        loggamma_m((nu + q) / 2, m)
        - loggamma_m(nu / 2, m)
        - q * m / 2 * math.log(math.pi)
        + m / 2 * logdet(ck)
        + nu / 2 * logdet(cs)
        - (nu + q) / 2 * logdet(cu)
    )


def close(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(close(a, b) for a, b in zip(actual, expected, strict=True))
        )
    return (
        isinstance(actual, (float, int))
        and not isinstance(actual, bool)
        and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12)
    )
