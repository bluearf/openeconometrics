"""Native joint Gaussian MGARCH(1,1) likelihood; no matrix projection."""

from __future__ import annotations

import math

import torch

from openecon.engines.contracts import KernelError
from openecon.engines.optimize import maximize_bfgs, information_inverse


def lower_pairs(d):
    return [(i, j) for i in range(d) for j in range(i + 1)]


def corr_pairs(d):
    return [(i, j) for i in range(d) for j in range(i)]


def parameter_count(kind, d, mean):
    return (d if mean else 0) + (
        len(lower_pairs(d)) + 2 * d * d
        if kind == "bekk"
        else 3 * d + len(corr_pairs(d)) + (2 if kind == "dcc" else 0)
    )


def bekk_radius(a, b):
    return (
        torch.linalg.eigvals(
            torch.kron(a.contiguous(), a.contiguous()) + torch.kron(b.contiguous(), b.contiguous())
        )
        .abs()
        .max()
    )


def _matrix_corr(raw, d):
    identity = torch.eye(d, dtype=torch.float64)
    lower = identity.clone()
    for value, (i, j) in zip(raw, corr_pairs(d), strict=True):
        lower[i, j] = value
    covariance = lower @ lower.T
    return (
        covariance / covariance.diagonal().sqrt()[:, None] / covariance.diagonal().sqrt()[None, :]
    )


def decode(raw, kind, d, mean):
    """Unconstrained optimizer to physical reporting parameters/matrices."""
    index = d if mean else 0
    mu = raw[:d] if mean else torch.zeros(d, dtype=torch.float64)
    if kind == "bekk":
        c = torch.zeros((d, d), dtype=torch.float64)
        for value, (i, j) in zip(
            raw[index : index + len(lower_pairs(d))], lower_pairs(d), strict=True
        ):
            c[i, j] = value.exp() if i == j else value
        index += len(lower_pairs(d))
        a = raw[index : index + d * d].reshape(d, d)
        b = raw[index + d * d : index + 2 * d * d].reshape(d, d)
        # Homogeneity of the full BEKK Kronecker operator gives a radial,
        # invertible map onto its strict covariance-stationary domain.
        denominator = torch.sqrt(1 + bekk_radius(a, b)) / math.sqrt(1 - 1e-6)
        a, b = a / denominator, b / denominator
        values = torch.cat(
            (
                mu if mean else raw[:0],
                torch.stack([c[i, j] for i, j in lower_pairs(d)]),
                a.flatten(),
                b.flatten(),
            )
        )
        return values, {"mu": mu, "C": c, "A": a, "B": b}
    garch = raw[index : index + 3 * d].reshape(d, 3)
    omega = garch[:, 0].exp()
    probabilities = torch.softmax(
        torch.column_stack((garch[:, 1:], torch.zeros(d, dtype=torch.float64))), dim=1
    ) * (1 - 1e-6)
    alpha, beta = probabilities[:, 0], probabilities[:, 1]
    index += 3 * d
    correlation = _matrix_corr(raw[index : index + len(corr_pairs(d))], d)
    values = torch.cat(
        (
            mu if mean else raw[:0],
            torch.column_stack((omega, alpha, beta)).flatten(),
            torch.stack([correlation[i, j] for i, j in corr_pairs(d)]),
        )
    )
    matrices = {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "target": correlation}
    if kind == "dcc":
        probs = torch.softmax(torch.cat((raw[-2:], torch.zeros(1, dtype=torch.float64))), dim=0) * (
            1 - 1e-6
        )
        matrices["dcc_alpha"], matrices["dcc_beta"] = probs[0], probs[1]
        values = torch.cat((values, probs[:2]))
    return values, matrices


def encode(values, kind, d, mean):
    """Physical inverse map, used only for starting values/reference checks."""
    index = d if mean else 0
    prefix = values[:d] if mean else values[:0]
    if kind == "bekk":
        cvalues = values[index : index + len(lower_pairs(d))]
        c = torch.stack(
            [v.log() if i == j else v for v, (i, j) in zip(cvalues, lower_pairs(d), strict=True)]
        )
        ab = values[index + len(lower_pairs(d)) :]
        norm = bekk_radius(ab[: d * d].reshape(d, d), ab[d * d :].reshape(d, d))
        if float(norm) >= 1 - 1e-6:
            raise KernelError(
                "mgarch_stationarity",
                "Full BEKK requires the Kronecker covariance operator spectral radius<1-1e-6.",
            )
        return torch.cat((prefix, c, ab / torch.sqrt((1 - 1e-6) - norm)))
    garch = values[index : index + 3 * d].reshape(d, 3)
    omega, alpha, beta = garch[:, 0], garch[:, 1], garch[:, 2]
    remainder = (1 - 1e-6) - alpha - beta
    g = torch.column_stack(
        (omega.log(), (alpha / remainder).log(), (beta / remainder).log())
    ).flatten()
    index += 3 * d
    correlation = torch.eye(d, dtype=torch.float64)
    for value, (i, j) in zip(
        values[index : index + len(corr_pairs(d))], corr_pairs(d), strict=True
    ):
        correlation[i, j] = correlation[j, i] = value
    lower = torch.linalg.cholesky(correlation)
    raw_corr = torch.stack([lower[i, j] / lower[i, i] for i, j in corr_pairs(d)])
    raw = torch.cat((prefix, g, raw_corr))
    if kind == "dcc":
        remainder = (1 - 1e-6) - values[-2:].sum()
        raw = torch.cat((raw, (values[-2:] / remainder).log()))
    return raw


def physical_matrices(values, kind, d, mean):
    """Reporting-unit matrices for persisted forecasts and independent tests."""
    index = d if mean else 0
    mu = values[:d] if mean else torch.zeros(d, dtype=torch.float64)
    if kind == "bekk":
        c = torch.zeros((d, d), dtype=torch.float64)
        for value, (i, j) in zip(
            values[index : index + len(lower_pairs(d))], lower_pairs(d), strict=True
        ):
            c[i, j] = value
        index += len(lower_pairs(d))
        return {
            "mu": mu,
            "C": c,
            "A": values[index : index + d * d].reshape(d, d),
            "B": values[index + d * d : index + 2 * d * d].reshape(d, d),
        }
    garch = values[index : index + 3 * d].reshape(d, 3)
    index += 3 * d
    corr = torch.eye(d, dtype=torch.float64)
    for value, (i, j) in zip(
        values[index : index + len(corr_pairs(d))], corr_pairs(d), strict=True
    ):
        corr[i, j] = corr[j, i] = value
    out = {
        "mu": mu,
        "omega": garch[:, 0],
        "alpha": garch[:, 1],
        "beta": garch[:, 2],
        "target": corr,
    }
    if kind == "dcc":
        out.update(dcc_alpha=values[-2], dcc_beta=values[-1])
    return out


def validate_matrices(m, kind):
    if any(not bool(torch.isfinite(value).all()) for value in m.values()):
        raise KernelError("mgarch_nonfinite", "MGARCH parameters must be finite.")
    if kind == "bekk":
        if bool((m["C"].diagonal() <= 0).any()):
            raise KernelError("mgarch_positivity", "BEKK C must have positive diagonal.")
        norm = float(
            torch.linalg.matrix_norm(m["A"], ord=2).square()
            + torch.linalg.matrix_norm(m["B"], ord=2).square()
        )
        radius = float(bekk_radius(m["A"], m["B"]))
        if radius >= 1 - 1e-6:
            raise KernelError(
                "mgarch_stationarity",
                "BEKK violates covariance stationarity of the full Kronecker operator.",
            )
        return {"spectral_radius": radius, "spectral_norm_contraction": norm}
    if (
        bool((m["omega"] <= 0).any())
        or bool((m["alpha"] <= 0).any())
        or bool((m["beta"] <= 0).any())
    ):
        raise KernelError(
            "mgarch_positivity", "GARCH omega, alpha and beta must be positive interior values."
        )
    if bool((m["alpha"] + m["beta"] >= 1 - 1e-6).any()):
        raise KernelError(
            "mgarch_stationarity", "Univariate GARCH alpha+beta must be less than 1-1e-6."
        )
    check_spd(m["target"], "correlation target")
    if kind == "dcc" and (
        float(m["dcc_alpha"]) <= 0
        or float(m["dcc_beta"]) <= 0
        or float(m["dcc_alpha"] + m["dcc_beta"]) >= 1 - 1e-6
    ):
        raise KernelError(
            "mgarch_stationarity", "DCC alpha,beta must be positive and sum below 1-1e-6."
        )
    return {
        "variance_persistence": (m["alpha"] + m["beta"]).tolist(),
        "dcc_persistence": float(m["dcc_alpha"] + m["dcc_beta"]) if kind == "dcc" else None,
    }


def check_spd(matrix, label):
    if not bool(torch.isfinite(matrix).all()) or not torch.allclose(
        matrix, matrix.T, rtol=1e-11, atol=1e-12
    ):
        raise KernelError("mgarch_invalid_covariance", f"{label} must be finite and symmetric.")
    chol, info = torch.linalg.cholesky_ex(matrix)
    if int(info) != 0:
        raise KernelError(
            "mgarch_invalid_covariance",
            f"{label} must be strictly positive definite; no projection is performed.",
        )
    return chol


def covariance_path(m, y, initial, kind):
    residuals = y - m["mu"]
    y.shape[1]
    if kind == "bekk":
        h = initial
        path = [h]
        for t in range(1, len(y)):
            shock = torch.outer(residuals[t - 1], residuals[t - 1])
            h = m["C"] @ m["C"].T + m["A"].T @ shock @ m["A"] + m["B"].T @ h @ m["B"]
            path.append(h)
        covariances = torch.stack(path)
        return covariances, residuals, {"H": h, "last_residual": residuals[-1]}
    h = initial.diagonal()
    q = m["target"]
    path = []
    for t in range(len(y)):
        correlation = q / q.diagonal().sqrt()[:, None] / q.diagonal().sqrt()[None, :]
        path.append(h.sqrt()[:, None] * correlation * h.sqrt()[None, :])
        if t < len(y) - 1:
            z = residuals[t] / h.sqrt()
            if kind == "dcc":
                q = (
                    (1 - m["dcc_alpha"] - m["dcc_beta"]) * m["target"]
                    + m["dcc_alpha"] * torch.outer(z, z)
                    + m["dcc_beta"] * q
                )
            h = m["omega"] + m["alpha"] * residuals[t].square() + m["beta"] * h
    covariances = torch.stack(path)
    return (
        covariances,
        residuals,
        {"H": covariances[-1], "h": h, "Q": q, "last_residual": residuals[-1]},
    )


def log_likelihood(raw, y, initial, kind, mean):
    _, m = decode(raw, kind, y.shape[1], mean)
    return matrix_log_likelihood(m, y, initial, kind)


def matrix_log_likelihood(m, y, initial, kind):
    covariance, residuals, state = covariance_path(m, y, initial, kind)
    # Batched Cholesky errors fail; invalid covariance never gets jitter/projected.
    chol = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(chol, residuals[:, :, None], upper=False)[:, :, 0]
    density = -0.5 * (
        y.shape[1] * math.log(2 * math.pi)
        + 2 * torch.log(chol.diagonal(dim1=1, dim2=2)).sum(dim=1)
        + whitened.square().sum(dim=1)
    )
    return density.sum()


def gradient(fn, raw, hessian=False):
    point = raw.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        value = fn(point)
        score = torch.autograd.grad(value, point, create_graph=hessian)[0]
        if hessian:
            info = torch.stack(
                [
                    torch.autograd.grad(score[i], point, retain_graph=True)[0]
                    for i in range(len(point))
                ]
            )
            return value.detach(), score.detach(), info.detach()
    return value.detach(), score.detach()


def fit_joint(y, initial, kind, mean, *, max_iterations=300, tolerance=1e-7):
    d = y.shape[1]
    check_spd(initial, "initial covariance")
    mu = y.mean(dim=0) if mean else y[:0].reshape(-1)
    if kind == "bekk":
        # Begin with genuinely full matrices; signs are not squared elementwise.
        a = torch.eye(d, dtype=torch.float64) * 0.15
        b = torch.eye(d, dtype=torch.float64) * 0.5
        a[0, 1] = 0.03
        b[1, 0] = -0.02
        c = torch.linalg.cholesky(initial * (1 - float(bekk_radius(a, b))))
        start_values = torch.cat(
            (mu, torch.stack([c[i, j] for i, j in lower_pairs(d)]), a.flatten(), b.flatten())
        )
    else:
        target = initial / initial.diagonal().sqrt()[:, None] / initial.diagonal().sqrt()[None, :]
        garch = torch.column_stack(
            (
                initial.diagonal() * 0.1,
                torch.full((d,), 0.1, dtype=torch.float64),
                torch.full((d,), 0.8, dtype=torch.float64),
            )
        )
        start_values = torch.cat(
            (
                mu,
                garch.flatten(),
                torch.stack([target[i, j] for i, j in corr_pairs(d)]),
                torch.tensor([0.05, 0.9], dtype=torch.float64)
                if kind == "dcc"
                else y[:0].reshape(-1),
            )
        )
    start = encode(start_values, kind, d, mean)

    def objective(q):
        return log_likelihood(q, y, initial, kind, mean)

    def evaluation(q):
        try:
            value, score = gradient(objective, q)
            if not bool(torch.isfinite(value)) or not bool(torch.isfinite(score).all()):
                return torch.tensor(float("-inf"), dtype=torch.float64), torch.zeros_like(q)
            return value, score
        except (RuntimeError, OverflowError, FloatingPointError):
            # Infeasible line-search trials are rejected, never projected.
            return torch.tensor(float("-inf"), dtype=torch.float64), torch.zeros_like(q)

    fit = maximize_bfgs(
        evaluation,
        start,
        max_iter=max_iterations,
        gradient_tol=tolerance,
        scaled_gradient_tol=tolerance,
        step_tol=tolerance * 0.01,
        hessian_fn=lambda q: gradient(objective, q, True)[2],
        raise_on_failure=False,
    )
    if not fit.converged:
        error = KernelError(
            "mgarch_nonconvergence",
            "Joint MGARCH likelihood did not converge at a nonsingular interior maximum.",
        )
        error.diagnostics = fit.diagnostics
        raise error
    physical, m = decode(fit.theta, kind, d, mean)
    constraints = validate_matrices(m, kind)

    # Evaluate actual reporting-unit information, rather than treating the
    # optimizer's approximate stationarity as exact under a nonlinear transform.
    def physical_objective(values):
        return matrix_log_likelihood(physical_matrices(values, kind, d, mean), y, initial, kind)

    _, physical_gradient, physical_hessian = gradient(physical_objective, physical, True)
    covariance = information_inverse(-physical_hessian)
    scaled_score = float(physical_gradient @ covariance @ physical_gradient)
    if not math.isfinite(scaled_score) or scaled_score > max(1e-6, 10 * tolerance):
        raise KernelError(
            "mgarch_nonconvergence",
            "MGARCH reporting-unit score has not converged after the parameter transformation.",
        )
    if not bool(torch.isfinite(covariance).all()) or bool((covariance.diagonal() <= 0).any()):
        raise KernelError("mgarch_information", "The full joint reporting covariance is invalid.")
    path, residuals, state = covariance_path(m, y, initial, kind)
    correlation = (
        path
        / path.diagonal(dim1=1, dim2=2).sqrt()[:, :, None]
        / path.diagonal(dim1=1, dim2=2).sqrt()[:, None, :]
    )
    diagnostics = {
        "converged": True,
        "iterations": fit.iterations,
        "gradient_max": float(fit.gradient.abs().max()),
        "physical_gradient_max": float(physical_gradient.abs().max()),
        "physical_scaled_gradient": scaled_score,
        "optimizer": fit.diagnostics,
        "covariance": "full joint observed information evaluated in physical reporting units",
        "derivatives": "float64 native PyTorch first/second autodiff",
        "constraints": constraints,
        "global_maximum_guaranteed": False,
        "estimation": "joint Gaussian conditional ML; all mean, variance and correlation parameters estimated together",
    }
    return (
        physical.detach(),
        covariance.detach(),
        float(fit.value),
        path.detach(),
        correlation.detach(),
        {k: v.detach() for k, v in state.items()},
        diagnostics,
    )


def forecast_path(m, state, kind, steps):
    """Exact one-step; BEKK full expectation, CCC/DCC multistep plug-in."""
    validate_matrices(m, kind)
    h = state["H"]
    check_spd(h, "forecast origin covariance")
    paths = []
    if kind == "bekk":
        shock = torch.outer(state["last_residual"], state["last_residual"])
        for step in range(steps):
            h = m["C"] @ m["C"].T + m["A"].T @ shock @ m["A"] + m["B"].T @ h @ m["B"]
            check_spd(h, "BEKK forecast covariance")
            paths.append(h)
            shock = h
        return torch.stack(paths)
    variance = state["h"]
    q = state["Q"]
    if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
        raise KernelError(
            "mgarch_invalid_covariance", "Forecast origin variances must be finite and positive."
        )
    check_spd(q, "forecast origin correlation state")
    origin_correlation = q / q.diagonal().sqrt()[:, None] / q.diagonal().sqrt()[None, :]
    expected_origin = variance.sqrt()[:, None] * origin_correlation * variance.sqrt()[None, :]
    if not torch.allclose(h, expected_origin, rtol=1e-10, atol=1e-12):
        raise KernelError(
            "mgarch_invalid_covariance",
            "Persisted forecast origin covariance disagrees with the variance/correlation state.",
        )
    if kind == "ccc" and not torch.allclose(q, m["target"], rtol=1e-10, atol=1e-12):
        raise KernelError(
            "mgarch_invalid_covariance",
            "CCC forecast origin correlation must equal the fitted fixed target.",
        )
    shock = state["last_residual"].square()
    z = state["last_residual"] / variance.sqrt()
    for step in range(steps):
        variance = m["omega"] + m["alpha"] * shock + m["beta"] * variance
        if kind == "dcc":
            q = (1 - m["dcc_alpha"] - m["dcc_beta"]) * m["target"] + (
                m["dcc_alpha"] * torch.outer(z, z) + m["dcc_beta"] * q
                if step == 0
                else (m["dcc_alpha"] + m["dcc_beta"]) * q
            )
        correlation = q / q.diagonal().sqrt()[:, None] / q.diagonal().sqrt()[None, :]
        h = variance.sqrt()[:, None] * correlation * variance.sqrt()[None, :]
        check_spd(h, "correlation forecast covariance")
        paths.append(h)
        shock = variance
    return torch.stack(paths)
