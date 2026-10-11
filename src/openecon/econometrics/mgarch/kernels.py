"""Native joint Gaussian MGARCH(1,1) likelihood; no matrix projection.

Kinds: ``ccc``, ``dcc``, full ``bekk`` (MARKET-131) and the MARKET-161 extensions
``dvech`` (diagonal VECH with a positive definite constant and positive diagonal
persistence; off-diagonal coefficients free, positive definiteness enforced on the
sample path),
``dbekk`` (diagonal BEKK) and ``adcc`` (asymmetric DCC of Cappiello, Engle and Sheppard
2006 with the jointly estimated target of ``dcc`` and the sample second moment of the
negative standardized residuals). See docs/econometrics/mgarch.md and
docs/econometrics/garch-extensions.md.
"""

from __future__ import annotations

import math

import torch

from openecon.engines.contracts import KernelError
from openecon.engines.optimize import maximize_bfgs, information_inverse


def lower_pairs(d):
    return [(i, j) for i in range(d) for j in range(i + 1)]


def corr_pairs(d):
    return [(i, j) for i in range(d) for j in range(i)]


_LIMIT = 1 - 1e-6     # strict stationarity margin shared by every kind


def parameter_count(kind, d, mean):
    pairs = len(lower_pairs(d))
    count = {
        "bekk": pairs + 2 * d * d,
        "dvech": 3 * pairs,
        "dbekk": pairs + 2 * d,
        "adcc": 3 * d + len(corr_pairs(d)) + 3,
    }.get(kind, 3 * d + len(corr_pairs(d)) + (2 if kind == "dcc" else 0))
    return (d if mean else 0) + count


def _symmetric(values, d):
    out = torch.zeros((d, d), dtype=torch.float64)
    for value, (i, j) in zip(values, lower_pairs(d), strict=True):
        out[i, j] = value
        out[j, i] = value
    return out


def _lower(values, d, positive_diagonal=True):
    out = torch.zeros((d, d), dtype=torch.float64)
    for value, (i, j) in zip(values, lower_pairs(d), strict=True):
        out[i, j] = value.exp() if (i == j and positive_diagonal) else value
    return out


def _stack_lower(matrix):
    d = matrix.shape[0]
    return torch.stack([matrix[i, j] for i, j in lower_pairs(d)])


def negative_moment(z):
    """N-bar = (1/T) sum_t n_t n_t', n_t = z_t 1(z_t < 0) (Cappiello-Engle-Sheppard eq. 7)."""
    n = z * (z < 0).to(torch.float64)
    return n.T @ n / z.shape[0]


def asymmetry_scale(target, nbar):
    """delta = largest eigenvalue of Qbar^-1/2 Nbar Qbar^-1/2 (via the Cholesky factor)."""
    factor = torch.linalg.cholesky(target)
    inner = torch.linalg.solve_triangular(factor, nbar, upper=False)
    inner = torch.linalg.solve_triangular(factor, inner.T, upper=False)
    return torch.linalg.eigvalsh(0.5 * (inner + inner.T)).max()


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
    if kind == "dvech":
        pairs = len(lower_pairs(d))
        c = _lower(raw[index : index + pairs], d)
        index += pairs
        ra, rb = raw[index : index + pairs], raw[index + pairs : index + 2 * pairs]
        a = torch.zeros((d, d), dtype=torch.float64)
        b = torch.zeros((d, d), dtype=torch.float64)
        for position, (i, j) in enumerate(lower_pairs(d)):
            if i == j:
                # Variance equations: a_ii, b_ii > 0 and a_ii + b_ii < 1 - 1e-6.
                share = torch.softmax(
                    torch.stack((ra[position], rb[position], torch.zeros((), dtype=torch.float64))),
                    dim=0,
                ) * _LIMIT
                a[i, i], b[i, i] = share[0], share[1]
            else:
                a[i, j] = a[j, i] = ra[position]
                b[i, j] = b[j, i] = rb[position]
        matrices = {"mu": mu, "C": c @ c.T, "A": a, "B": b}
        values = torch.cat(
            (
                mu if mean else raw[:0],
                _stack_lower(matrices["C"]),
                _stack_lower(matrices["A"]),
                _stack_lower(matrices["B"]),
            )
        )
        return values, matrices
    if kind == "dbekk":
        pairs = len(lower_pairs(d))
        c = _lower(raw[index : index + pairs], d)
        index += pairs
        ra, rb = raw[index : index + d], raw[index + d : index + 2 * d]
        # The first entries are positive (a and -a give the same model).
        ra = torch.cat((ra[:1].exp(), ra[1:]))
        rb = torch.cat((rb[:1].exp(), rb[1:]))
        scale = torch.sqrt(_LIMIT / (1 + ra.square() + rb.square()))
        a, b = ra * scale, rb * scale
        values = torch.cat((mu if mean else raw[:0], _stack_lower(c), a, b))
        return values, {"mu": mu, "C": c, "a": a, "b": b}
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
    if kind == "adcc":
        # a + b + delta g < 1 - 1e-6: the simplex gives (a, b, delta g); delta depends on
        # the data and is applied in covariance_path ("dcc_gamma_share").
        probs = torch.softmax(torch.cat((raw[-3:], torch.zeros(1, dtype=torch.float64))), dim=0) * (
            1 - 1e-6
        )
        matrices["dcc_alpha"], matrices["dcc_beta"] = probs[0], probs[1]
        matrices["dcc_gamma_share"] = probs[2]
        values = torch.cat((values, probs[:3]))
    return values, matrices


def encode(values, kind, d, mean, delta=None):
    """Physical inverse map, used only for starting values/reference checks.

    ``adcc`` needs ``delta`` (``asymmetry_scale``) to map dcc_gamma onto its share.
    """
    index = d if mean else 0
    prefix = values[:d] if mean else values[:0]
    if kind in {"dvech", "dbekk"}:
        m = physical_matrices(values, kind, d, mean)
        lc = m["C"] if kind == "dbekk" else torch.linalg.cholesky(m["C"])
        raw_c = torch.stack(
            [lc[i, j].log() if i == j else lc[i, j] for i, j in lower_pairs(d)]
        )
        if kind == "dbekk":
            a, b = m["a"], m["b"]
            size = a.square() + b.square()
            if float(size.max()) >= _LIMIT or float(a[0]) <= 0 or float(b[0]) <= 0:
                raise KernelError(
                    "mgarch_stationarity",
                    "Diagonal BEKK requires a_i^2 + b_i^2 < 1-1e-6 and positive a_1, b_1.",
                )
            stretch = torch.sqrt((1 + size / (_LIMIT - size)) / _LIMIT)
            ra, rb = a * stretch, b * stretch
            ra = torch.cat((ra[:1].log(), ra[1:]))
            rb = torch.cat((rb[:1].log(), rb[1:]))
            return torch.cat((prefix, raw_c, ra, rb))
        raw_a, raw_b = [], []
        for i, j in lower_pairs(d):
            a, b = m["A"][i, j], m["B"][i, j]
            if i == j:
                remainder = _LIMIT - a - b
                if float(a) <= 0 or float(b) <= 0 or float(remainder) <= 0:
                    raise KernelError(
                        "mgarch_stationarity",
                        "Diagonal VECH requires a_ii, b_ii > 0 and a_ii + b_ii < 1-1e-6.",
                    )
                a, b = (a / remainder).log(), (b / remainder).log()
            raw_a.append(a)
            raw_b.append(b)
        return torch.cat((prefix, raw_c, torch.stack(raw_a), torch.stack(raw_b)))
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
    if kind == "adcc":
        shares = torch.stack((values[-3], values[-2], values[-1] * delta))
        remainder = (1 - 1e-6) - shares.sum()
        raw = torch.cat((raw, (shares / remainder).log()))
    return raw


def physical_matrices(values, kind, d, mean):
    """Reporting-unit matrices for persisted forecasts and independent tests."""
    index = d if mean else 0
    mu = values[:d] if mean else torch.zeros(d, dtype=torch.float64)
    if kind == "dvech":
        pairs = len(lower_pairs(d))
        blocks = [values[index + k * pairs : index + (k + 1) * pairs] for k in range(3)]
        return {"mu": mu, **{name: _symmetric(block, d) for name, block in zip("CAB", blocks)}}
    if kind == "dbekk":
        pairs = len(lower_pairs(d))
        c = torch.zeros((d, d), dtype=torch.float64)
        for value, (i, j) in zip(values[index : index + pairs], lower_pairs(d), strict=True):
            c[i, j] = value
        index += pairs
        return {"mu": mu, "C": c, "a": values[index : index + d],
                "b": values[index + d : index + 2 * d]}
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
    if kind == "adcc":
        out.update(dcc_alpha=values[-3], dcc_beta=values[-2], dcc_gamma=values[-1])
    return out


def validate_matrices(m, kind):
    if any(not bool(torch.isfinite(value).all()) for value in m.values()):
        raise KernelError("mgarch_nonfinite", "MGARCH parameters must be finite.")
    if kind == "dvech":
        check_spd(m["C"], "diagonal VECH constant C")
        a, b = m["A"], m["B"]
        if bool((a.diagonal() <= 0).any()) or bool((b.diagonal() <= 0).any()):
            raise KernelError(
                "mgarch_positivity", "Diagonal VECH a_ii and b_ii must be positive."
            )
        persistence = a + b
        if float(persistence.abs().max()) >= 1 - 1e-6 or float(b.abs().max()) >= 1 - 1e-6:
            raise KernelError(
                "mgarch_stationarity",
                "Diagonal VECH requires |a_ij + b_ij| < 1-1e-6 and |b_ij| < 1-1e-6 for every "
                "element (each element of H_t is a stationary first-order recursion).",
            )
        smallest = [float(torch.linalg.eigvalsh(m[name]).min()) for name in ("A", "B")]
        unconditional = m["C"] / (1 - persistence)
        unconditional_pd = int(torch.linalg.cholesky_ex(unconditional)[1]) == 0
        return {
            "max_element_persistence": float(persistence.abs().max()),
            "ding_engle_psd": min(smallest) >= -1e-12,
            "min_eigenvalue_A": smallest[0],
            "min_eigenvalue_B": smallest[1],
            "unconditional_covariance_positive_definite": unconditional_pd,
        }
    if kind == "dbekk":
        if bool((m["C"].diagonal() <= 0).any()) or float(m["a"][0]) <= 0 or float(m["b"][0]) <= 0:
            raise KernelError(
                "mgarch_positivity",
                "Diagonal BEKK C must have a positive diagonal and a_1, b_1 must be positive.",
            )
        persistence = (m["a"].square() + m["b"].square()).max()
        if float(persistence) >= 1 - 1e-6:
            raise KernelError(
                "mgarch_stationarity",
                "Diagonal BEKK covariance stationarity requires a_i^2 + b_i^2 < 1-1e-6.",
            )
        return {"spectral_radius": float(persistence)}
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
    if kind == "adcc":
        if float(m["dcc_alpha"]) <= 0 or float(m["dcc_beta"]) <= 0 or float(m["dcc_gamma"]) <= 0:
            raise KernelError(
                "mgarch_stationarity", "aDCC a, b and g must be positive interior values."
            )
        persistence = float(m["dcc_alpha"] + m["dcc_beta"])
        if "nbar" in m:
            delta = float(asymmetry_scale(m["target"], m["nbar"]))
            persistence += delta * float(m["dcc_gamma"])
            if persistence >= 1 - 1e-6:
                raise KernelError(
                    "mgarch_stationarity",
                    "aDCC requires a + b + delta g < 1-1e-6 (positive definite intercept).",
                )
        return {
            "variance_persistence": (m["alpha"] + m["beta"]).tolist(),
            "adcc_persistence": persistence,
        }
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
    if kind in {"dvech", "dbekk"}:
        if kind == "dbekk":
            c, a, b = m["C"] @ m["C"].T, torch.outer(m["a"], m["a"]), torch.outer(m["b"], m["b"])
        else:
            c, a, b = m["C"], m["A"], m["B"]
        h = initial
        path = [h]
        for t in range(1, len(y)):
            h = c + a * torch.outer(residuals[t - 1], residuals[t - 1]) + b * h
            path.append(h)
        return torch.stack(path), residuals, {"H": h, "last_residual": residuals[-1]}
    if kind == "adcc":
        return _adcc_path(m, residuals, initial)
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


def _adcc_path(m, residuals, initial):
    """Asymmetric DCC (Cappiello, Engle and Sheppard 2006, eq. 6-7), scalar version:

    Q_(t+1) = (1 - a - b) Qbar - g Nbar + a z_t z_t' + b Q_t + g n_t n_t',   Q_1 = Qbar,

    with n_t = z_t 1(z_t < 0) and Nbar the sample second moment of n at the current
    variance parameters (``m["nbar"]`` overrides it for reference checks). With the
    share parameterization of ``decode`` g = share / delta.
    """
    T = residuals.shape[0]
    h = initial.diagonal()
    variances = []
    for t in range(T):
        variances.append(h)
        if t < T - 1:
            h = m["omega"] + m["alpha"] * residuals[t].square() + m["beta"] * h
    variances = torch.stack(variances)
    z = residuals / variances.sqrt()
    nbar = m["nbar"] if "nbar" in m else negative_moment(z)
    target = m["target"]
    gamma = m["dcc_gamma"] if "dcc_gamma" in m else m["dcc_gamma_share"] / asymmetry_scale(
        target, nbar
    )
    a, b = m["dcc_alpha"], m["dcc_beta"]
    intercept = (1 - a - b) * target - gamma * nbar
    negative = z * (z < 0).to(torch.float64)
    q = target
    path = []
    for t in range(T):
        correlation = q / q.diagonal().sqrt()[:, None] / q.diagonal().sqrt()[None, :]
        sd = variances[t].sqrt()
        path.append(sd[:, None] * correlation * sd[None, :])
        if t < T - 1:
            q = (
                intercept
                + a * torch.outer(z[t], z[t])
                + b * q
                + gamma * torch.outer(negative[t], negative[t])
            )
    covariances = torch.stack(path)
    state = {"H": covariances[-1], "h": variances[-1], "Q": q, "last_residual": residuals[-1],
             "nbar": nbar}
    return covariances, residuals, state


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
    elif kind == "dvech":
        scale = initial.diagonal().sqrt()
        correlation = initial / scale[:, None] / scale[None, :]
        start_values = torch.cat(
            (mu, _stack_lower(0.05 * initial), _stack_lower(0.05 * correlation),
             _stack_lower(0.9 * correlation))
        )
    elif kind == "dbekk":
        c = torch.linalg.cholesky(0.05 * initial)
        start_values = torch.cat(
            (mu, _stack_lower(c), torch.full((d,), math.sqrt(0.05), dtype=torch.float64),
             torch.full((d,), math.sqrt(0.9), dtype=torch.float64))
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
                else torch.tensor([0.04, 0.9, 0.02], dtype=torch.float64)
                if kind == "adcc"
                else y[:0].reshape(-1),
            )
        )
    delta = None
    if kind == "adcc":
        start_matrices = physical_matrices(start_values, kind, d, mean)
        nbar = covariance_path(start_matrices, y, initial, kind)[2]["nbar"]
        delta = asymmetry_scale(start_matrices["target"], nbar)
        start_values[-1] = min(0.02, 0.05 / float(delta)) if float(delta) > 0 else 0.02
    start = encode(start_values, kind, d, mean, delta)

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
    if kind == "adcc":
        # The physical asymmetry coefficient g = share / delta at the estimates.
        nbar = covariance_path(m, y, initial, kind)[2]["nbar"]
        gamma = m["dcc_gamma_share"] / asymmetry_scale(m["target"], nbar)
        physical = torch.cat((physical[:-1], gamma.reshape(1)))
        m = physical_matrices(physical, kind, d, mean)
        m["nbar"] = nbar
    constraints = validate_matrices(m, kind)
    m.pop("nbar", None)

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
    """Exact one-step; BEKK/diagonal VECH/diagonal BEKK full expectation, CCC/DCC/aDCC
    multistep plug-in."""
    if kind == "adcc":
        m = {**m, "nbar": state["nbar"]}
    validate_matrices(m, kind)
    h = state["H"]
    check_spd(h, "forecast origin covariance")
    paths = []
    if kind in {"dvech", "dbekk"}:
        if kind == "dbekk":
            c, a, b = m["C"] @ m["C"].T, torch.outer(m["a"], m["a"]), torch.outer(m["b"], m["b"])
        else:
            c, a, b = m["C"], m["A"], m["B"]
        # E[e e'] = H turns the recursion into an exact expectation recursion.
        shock = torch.outer(state["last_residual"], state["last_residual"])
        for step in range(steps):
            h = c + a * shock + b * h
            check_spd(h, "diagonal model forecast covariance")
            paths.append(h)
            shock = h
        return torch.stack(paths)
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
        if kind == "adcc":
            negative = z * (z < 0).to(torch.float64)
            a, b, g = m["dcc_alpha"], m["dcc_beta"], m["dcc_gamma"]
            q = (
                (1 - a - b) * m["target"] - g * state["nbar"]
                + a * torch.outer(z, z) + b * q + g * torch.outer(negative, negative)
                if step == 0
                # Engle-Sheppard section 7 with E[n n'] replaced by Nbar.
                else (1 - a - b) * m["target"] + (a + b) * q
            )
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
