"""Conservative existence guards and explicitly Monte Carlo summaries."""

from __future__ import annotations

from .admission import metadata_size


def availability(nu, m, horizon, *, orthogonalized=False, impulse=False):
    """Sufficient polynomial-moment bounds, not necessary thresholds."""
    if impulse and not orthogonalized and horizon == 0:
        return {
            "mean": True,
            "second": True,
            "fourth": True,
            "mean_threshold": None,
            "second_threshold": None,
            "fourth_threshold": None,
        }
    degree = horizon + (1 if impulse and orthogonalized else 0)
    # B has conditional scale sqrt(Sigma); orthogonalized Phi_h adds one factor.
    thresholds = {
        "mean": m + degree - 1,
        "second": m + 2 * degree - 1,
        "fourth": m + 4 * degree - 1,
    }
    return {key: nu > threshold for key, threshold in thresholds.items()} | {
        f"{key}_threshold": threshold for key, threshold in thresholds.items()
    }


def summarize(paths, nu, m, horizon, alpha, *, orthogonalized=False, impulse=False):
    import torch

    from .kernels import gram, tensor

    a = tensor(paths)
    count = a.shape[0]
    flat = a.reshape(count, -1)
    status = availability(nu, m, horizon, orthogonalized=orthogonalized, impulse=impulse)
    quantiles = torch.quantile(flat, tensor((alpha / 2, 0.5, 1 - alpha / 2)), dim=0).tolist()
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = gram(centered) / (count - 1) if status["second"] and count >= 2 else None
    mean_se = (covariance.diagonal() / count).sqrt().tolist() if covariance is not None else None
    covariance_se = None
    if status["fourth"] and count >= 2:
        # Plug-in influence-function MCSE for an IID sample covariance estimator.
        d = flat.shape[1]
        covariance_se = torch.empty((d, d), dtype=torch.float64)
        for i in range(d):
            for j in range(i + 1):
                products = centered[:, i] * centered[:, j]
                covariance_se[i, j] = covariance_se[j, i] = products.std(correction=1) / count**0.5
        covariance_se = covariance_se.tolist()
    output = {
        "moment_availability": status,
        "sample_mean": mean.tolist() if status["mean"] else None,
        "pointwise_quantiles": quantiles,
        "covariance_mc_estimate": None if covariance is None else covariance.tolist(),
        "mean_mc_standard_error": mean_se,
        "covariance_mc_standard_error": covariance_se,
        "covariance_label": "IID Monte Carlo estimate; population covariance is not computed analytically",
        "mc_error_label": "sample covariance influence-function plug-in; no error bound/convergence certificate",
        "interval_label": "pointwise equal-tail empirical posterior quantiles; no joint coverage claim",
    }
    metadata_size(output)
    return output
