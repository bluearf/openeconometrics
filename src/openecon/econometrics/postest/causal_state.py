"""Bounded reporting-coordinate joint nuisance and POM state."""

import torch


def stacked_state(size, covariance, tau, tau_start, levels, blocks):
    """blocks: (slice, equation name, feature names, working beta, reporting map)."""
    units = torch.eye(size, dtype=torch.float64)
    theta = torch.zeros(size, dtype=torch.float64)
    terms, equations = [""] * size, {}
    for block, name, features, beta, mapping in blocks:
        units[block, block] = mapping
        theta[block] = beta
        reported = [name + ":" + feature for feature in features]
        terms[block] = reported
        equations[name] = dict(zip(reported, features, strict=True))
    theta[tau_start:] = tau
    terms[tau_start:] = ["POM:" + level for level in levels]
    return {
        "version": 1,
        "terms": terms,
        "parameters": (units @ theta).tolist(),
        "covariance": (units @ covariance @ units.T).tolist(),
        "equations": equations,
        "scope": "full joint stacked nuisance/POM sandwich in reporting coordinates",
    }
