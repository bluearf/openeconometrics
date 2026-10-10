"""Native inverse QR, Powell bread and weak-identification test profiles."""
from __future__ import annotations

import math

import torch

from openecon.econometrics.core import kernel_call
from openecon.econometrics.quantile import kernels as qr
from openecon.engines.distributions import chi2_sf, normal_ppf
from .common import fail


def _pd(matrix, name):
    scale = matrix.diagonal().abs().sqrt()
    if not bool(torch.isfinite(matrix).all()) or bool((scale == 0).any()):
        fail(f"{name} is nonfinite or singular.", "invalid_covariance")
    normalized = matrix / scale[:, None] / scale[None, :]
    factor, info = torch.linalg.cholesky_ex((normalized + normalized.T) / 2)
    if int(info) or float(factor.diagonal().min()) <= 1e-6:
        fail(f"{name} is not numerically positive definite.", "invalid_covariance")


def density_width(residual, config):
    """Powell residual-space rectangle bandwidth; no common-density assumption."""
    if config["bandwidth"] is not None:
        return config["bandwidth"]
    spread = min(float(residual.std(unbiased=True)),
                 float(torch.quantile(residual, 0.75) - torch.quantile(residual, 0.25)) / 1.349)
    value = 1.06 * spread * len(residual)**(-0.2)
    if not math.isfinite(value) or not value > 1e-12:
        fail("Residual-space bandwidth is degenerate; supply a justified explicit bandwidth.",
             "degenerate_density")
    return value


def certificate(w, target, beta, basis, tau, *, witness=None):
    """Native LP primal/dual optimality and unique-solution certificate, no refit."""
    n, k = w.shape
    if len(basis) != k or len(set(basis)) != k or any(type(v) is not int or not 0 <= v < n for v in basis):
        fail("The quantile LP basis is invalid.", "invalid_state")
    beta = torch.tensor(beta, dtype=torch.float64, device="cpu")
    if beta.shape != (k,) or not bool(torch.isfinite(beta).all()):
        fail("The quantile coefficients are invalid.", "invalid_state")
    residual = target - w @ beta
    scale = max(float((target - target.mean()).abs().mean()), 1e-12)
    tolerance = max(2e-9 * scale, 64 * torch.finfo(torch.float64).eps *
                    max(float(target.abs().max()), float((w @ beta).abs().max()), 1e-12))
    if float(residual[basis].abs().max()) > tolerance:
        fail("Saved LP basis does not interpolate the adjusted outcome.", "invalid_state")
    nonbasis = torch.ones(n, dtype=torch.bool, device="cpu")
    nonbasis[basis] = False
    if bool((residual[nonbasis].abs() <= tolerance).any()):
        fail("A tied or ambiguous profile QR is outside the continuous unique-LP contract.",
             "ambiguous_quantile")
    psi = torch.full_like(residual, tau)
    psi = torch.where(residual < 0, psi - 1, psi)
    rhs = -(w[nonbasis].T @ psi[nonbasis])
    try:
        lam = torch.linalg.solve(w[basis].T, rhs)
    except torch.linalg.LinAlgError:
        fail("The saved LP basis is singular.", "invalid_state")
    if bool((lam < tau - 1 - 1e-8).any()) or bool((lam > tau + 1e-8).any()):
        fail("The profile QR fails the LP dual certificate.", "invalid_state")
    # Strict interior basic dual values imply a unique coefficient vector.
    if float(torch.minimum(lam - (tau - 1), tau - lam).min()) <= 1e-8:
        fail("The profile QR has no strict unique vertex certificate.", "ambiguous_quantile")
    if witness is not None:
        # Scores live in the dimensionless quantile dual interval [tau-1, tau].
        close(lam, witness, "LP dual witness", units=1., normalized_atol=1e-10)
    psi[basis] = lam
    balance = w.T @ psi
    scaled = balance / torch.linalg.vector_norm(w, dim=0)
    if float(scaled.abs().max()) > 1e-7:
        fail("The LP certificate is not balanced.", "invalid_state")
    residual[basis] = 0
    objective = float(qr.check_loss(residual, tau))
    return beta, residual, lam, objective


def profile_record(y, d, w, a, kx, alpha, config):
    adjusted = y - d * alpha
    fit = kernel_call(qr.solve, kernel_call(qr.prepare, w, adjusted), config["quantile"],
                      max_iterations=200, max_pivots=200 + 20 * w.shape[1])
    if fit.exact_fit or not fit.unique:
        fail("Profile QR must be finite, nonperfect and uniquely certified.", "ambiguous_quantile")
    beta, residual, lam, objective = certificate(w, adjusted, fit.beta.tolist(),
                                                fit.basis.tolist(), config["quantile"])
    value = calculate(y, d, w, a, kx, alpha, beta, residual, config)
    value.update(basis=fit.basis.tolist(), dual_basis=lam.tolist(),
                 qr_objective=objective, qr_iterations=fit.iterations,
                 qr_pivots=fit.pivots, qr_gap=fit.gap)
    return value


def calculate(y, d, w, a, kx, alpha, beta, residual, config):
    h = density_width(residual, config)
    f = (residual.abs() <= h).to(torch.float64) / (2 * h)
    bread = w.T @ (f[:, None] * w)
    _pd(bread, "Powell QR information")
    inverse = torch.linalg.inv(bread)
    meat = config["quantile"] * (1 - config["quantile"]) * (w.T @ w)
    covariance = inverse @ meat @ inverse.T
    covariance = (covariance + covariance.T) / 2
    _pd(covariance, "complete QR covariance")
    gamma = beta[kx:]
    c = covariance[kx:, kx:]
    statistic = float(gamma @ torch.linalg.solve(c, gamma))
    structural_residual = y - d * alpha - w[:, :kx] @ beta[:kx]
    moment = w.T @ (config["quantile"] - (structural_residual <= 0).to(torch.float64)) / len(y)
    return dict(alpha=float(alpha), coefficients=beta.tolist(), bandwidth=h,
                density_support=int((f > 0).sum()), bread=bread.tolist(),
                covariance=covariance.tolist(),
                criterion=float(gamma @ a @ gamma), statistic=statistic,
                p_value=chi2_sf(statistic, w.shape[1] - kx),
                structural_moments=moment.tolist())


def close(actual, expected, name, *, units=None, normalized_atol=None):
    """Replay in declared units, with no universal dimensionful absolute floor.

    Covariance/information entries use coordinate diagonal scales. Other
    dimensionful fields use relative precision unless a source-derived unit is
    supplied. Absolute roundoff is allowed only after unit normalization.
    """
    try:
        pending = [expected]
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError("Invalid saved numeric type.")
        expected = torch.tensor(expected, dtype=torch.float64, device="cpu")
        actual = (actual.to(dtype=torch.float64, device="cpu") if isinstance(actual, torch.Tensor)
                  else torch.tensor(actual, dtype=torch.float64, device="cpu"))
        if (actual.shape != expected.shape or not bool(torch.isfinite(expected).all())
                or not bool(torch.isfinite(actual).all())):
            raise ValueError("Invalid shape/finite value.")
        if name in {"covariance", "bread"}:
            if actual.ndim != 2 or actual.shape[0] != actual.shape[1]:
                raise ValueError("Invalid covariance/information dimensions.")
            coordinate = torch.maximum(actual.diagonal().abs(), expected.diagonal().abs()).sqrt()
            if bool((coordinate <= 0).any()):
                raise ValueError("Invalid covariance/information coordinate scales.")
            actual = actual / coordinate[:, None] / coordinate[None, :]
            expected = expected / coordinate[:, None] / coordinate[None, :]
            roundoff = 128 * torch.finfo(torch.float64).eps
        elif units is not None:
            units = torch.as_tensor(units, dtype=torch.float64, device="cpu")
            if not bool(torch.isfinite(units).all()) or bool((units <= 0).any()):
                raise ValueError("Invalid declared replay units.")
            actual, expected = actual / units, expected / units
            roundoff = (128 * torch.finfo(torch.float64).eps
                        if normalized_atol is None else normalized_atol)
        else:
            roundoff = 0.
        tolerance = 2e-8 * torch.maximum(actual.abs(), expected.abs()) + roundoff
        if not bool(((actual - expected).abs() <= tolerance).all()):
            raise ValueError("Numerical disagreement.")
    except (RuntimeError, TypeError, ValueError):
        fail(f"Saved {name} fails semantic numerical replay.", "invalid_state")


def replay_record(y, d, w, a, kx, record, config):
    fields = {"alpha", "coefficients", "bandwidth", "density_support", "bread", "covariance",
              "criterion", "statistic", "p_value", "structural_moments", "basis", "dual_basis",
              "qr_objective", "qr_iterations", "qr_pivots", "qr_gap"}
    if not isinstance(record, dict) or set(record) != fields:
        fail("Invalid complete profile record schema.", "invalid_state")
    if isinstance(record["alpha"], bool) or not isinstance(record["alpha"], (int, float)) or not math.isfinite(record["alpha"]):
        fail("Invalid saved profile alpha.", "invalid_state")
    if type(record["density_support"]) is not int or not w.shape[1] <= record["density_support"] <= len(y):
        fail("Invalid saved density support.", "invalid_state")
    if type(record["qr_iterations"]) is not int or not 0 <= record["qr_iterations"] <= 200:
        fail("Invalid saved QR iteration diagnostics.", "invalid_state")
    if type(record["qr_pivots"]) is not int or not 0 <= record["qr_pivots"] <= 200 + 20 * w.shape[1]:
        fail("Invalid saved QR pivot diagnostics.", "invalid_state")
    if not isinstance(record["qr_gap"], (int, float)) or not math.isfinite(record["qr_gap"]) or record["qr_gap"] < 0:
        fail("Invalid saved QR gap diagnostics.", "invalid_state")
    beta, residual, _, objective = certificate(w, y - d * record["alpha"],
                                              record["coefficients"], record["basis"],
                                              config["quantile"], witness=record["dual_basis"])
    actual = calculate(y, d, w, a, kx, record["alpha"], beta, residual, config)
    for name in actual:
        if name in {"statistic", "p_value"}:
            close(actual[name], record[name], name, units=1., normalized_atol=1e-10)
        elif name == "structural_moments":
            close(actual[name], record[name], name,
                  units=torch.linalg.vector_norm(w, dim=0) / math.sqrt(len(y)))
        else:
            close(actual[name], record[name], name)
    close(objective, record["qr_objective"], "QR objective")
    return record


def accepted_runs(records, confidence):
    """Connected runs of accepted tested POINTS; edges/tails remain unknown."""
    accepted = [r["p_value"] >= 1 - confidence for r in records]
    runs, start = [], None
    for i, keep in enumerate(accepted + [False]):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            end = i - 1
            runs.append(dict(first_index=start, last_index=end,
                             first_alpha=records[start]["alpha"], last_alpha=records[end]["alpha"],
                             lower_grid_boundary=start == 0,
                             upper_grid_boundary=end == len(records) - 1,
                             between_points="untested", outer_tails="unknown"))
            start = None
    return accepted, runs


def identified_covariance(y, d, x, w, a, selected, config):
    """Joint inverse-QR influence map, with EVERY nuisance/cross block."""
    kx = x.shape[1]
    beta = torch.tensor(selected["coefficients"], dtype=torch.float64, device="cpu")
    residual = y - d * selected["alpha"] - w @ beta
    # Basis residuals are mathematical zero; use the same canonical residual convention.
    residual[selected["basis"]] = 0
    f = (residual.abs() <= selected["bandwidth"]).to(torch.float64) / (2 * selected["bandwidth"])
    bread = torch.tensor(selected["bread"], dtype=torch.float64, device="cpu")
    covariance = torch.tensor(selected["covariance"], dtype=torch.float64, device="cpu")
    derivative = -torch.linalg.solve(bread, w.T @ (f * d))
    t = derivative[kx:]
    precision = float(t @ a @ t)
    ratio = precision / max(float(a.diagonal().sum()) * float(derivative.square().sum()), 1e-300)
    if not precision > 0 or ratio <= 1e-10:
        fail("The structural inverse-QR local Jacobian is unidentified.", "weak_identification")
    row = -(t @ a) / precision
    influence = torch.zeros((kx + 1, w.shape[1]), dtype=torch.float64, device="cpu")
    influence[0, kx:] = row
    influence[1:, :kx] = torch.eye(kx, dtype=torch.float64, device="cpu")
    influence[1:] += derivative[:kx, None] * influence[0, None]
    joint = influence @ covariance @ influence.T
    joint = (joint + joint.T) / 2
    _pd(joint, "joint structural parameter covariance")
    params = torch.cat([torch.tensor([selected["alpha"]], dtype=torch.float64, device="cpu"), beta[:kx]])
    return dict(parameters=params.tolist(), covariance=joint.tolist(),
                profile_derivative=derivative.tolist(), influence_map=influence.tolist(),
                local_jacobian_precision=precision,
                inference="asymptotic normal under declared local strong identification and consistent profile search")


def coefficient_rows(terms, values, covariance, confidence):
    critical = normal_ppf((1 + confidence) / 2)
    rows = []
    for i, (name, value) in enumerate(zip(terms, values)):
        if covariance is None:
            se = z = p = low = high = None
        else:
            se = math.sqrt(covariance[i][i])
            z = value / se
            p = math.erfc(abs(z) / math.sqrt(2))
            low, high = value - critical * se, value + critical * se
        rows.append(dict(term=name, coefficient=value, std_error=se, z=z, p_value=p,
                         ci_lower=low, ci_upper=high))
    return rows
