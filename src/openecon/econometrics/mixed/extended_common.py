"""Bounded native likelihood derivatives and full observed-information inference.

New likelihoods use exact chain-rule differentiation of Torch expressions;
neither finite differences nor external statistical runtimes enter estimation.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call
from openecon.engines import optimize
from openecon.engines.distributions import gauss_hermite


class Objective:
    def __init__(self, value, work, budget):
        self.value = value
        self.work = int(work)
        self.budget = int(budget)
        self.used = 0

    def charge(self, multiplier=1):
        self.used += self.work * multiplier
        if self.used > self.budget:
            raise AnalysisError(
                "work_budget",
                "Likelihood derivative work exceeded max_work; raise the explicit budget or reduce the model.",
            )

    def __call__(self, theta):
        self.charge(2)
        with torch.enable_grad():
            point = theta.detach().requires_grad_()
            value = self.value(point)
            grad = torch.autograd.grad(value, point)[0]
        return value.detach(), grad.detach()

    def hessian(self, theta):
        self.charge(2 * (len(theta) + 1))
        with torch.enable_grad():
            return torch.autograd.functional.hessian(self.value, theta).detach()


def maximize(objective, starts):
    candidates = []
    for initial in starts:
        objective.charge()
        value = float(objective.value(initial))
        if math.isfinite(value):
            candidates.append((value, initial))
    if not candidates:
        raise AnalysisError("invalid_start", "No finite starting likelihood.")
    initial = max(candidates, key=lambda pair: pair[0])[1]
    fitted = kernel_call(
        optimize.maximize_bfgs,
        objective,
        initial,
        hessian_fn=objective.hessian,
        raise_on_failure=False,
        max_iter=500,
        scaled_gradient_tol=1e-10,
    )
    if not fitted.converged:
        raise AnalysisError(
            "nonconvergence", f"Likelihood did not converge: {fitted.diagnostics.get('message')}"
        )
    information = -(fitted.hessian + fitted.hessian.T) / 2
    scale = information.diagonal().abs().rsqrt()
    chol, status = torch.linalg.cholesky_ex(information * scale[:, None] * scale)
    if int(status) or not bool(torch.isfinite(chol).all()) or float(chol.diagonal().min()) < 1e-6:
        raise AnalysisError(
            "boundary_solution",
            "Full observed information is singular or indefinite; ordinary Wald inference is unavailable.",
        )
    covariance = torch.cholesky_inverse(chol) * scale[:, None] * scale
    return fitted, covariance


def transformed(theta, covariance, function):
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(function, theta)
    return function(theta).detach(), jacobian @ covariance @ jacobian.T


def quadrature(points, dimensions):
    nodes, weights = kernel_call(gauss_hermite, points)
    axes = torch.cartesian_prod(*([torch.arange(points)] * dimensions))
    if dimensions == 1:
        axes = axes[:, None]
    return math.sqrt(2) * nodes[axes], torch.log(weights[axes] / math.sqrt(math.pi)).sum(1)


def factor(theta, dimensions, structure):
    if structure == "independent":
        return torch.diag(torch.exp(theta[:dimensions]))
    rows, cols = torch.tril_indices(dimensions, dimensions)
    values = torch.where(rows == cols, theta.exp(), theta)
    return torch.zeros((dimensions, dimensions), dtype=torch.float64).index_put(
        (rows, cols), values
    )


def factor_size(dimensions, structure):
    return dimensions if structure == "independent" else dimensions * (dimensions + 1) // 2


def factor_start(dimensions, structure, sd):
    if structure == "independent":
        return torch.full((dimensions,), math.log(sd), dtype=torch.float64)
    rows, cols = torch.tril_indices(dimensions, dimensions)
    return torch.where(rows == cols, math.log(sd), 0.0).to(torch.float64)


def covariance_report(theta, dimensions, structure):
    lower = factor(theta, dimensions, structure)
    matrix = lower @ lower.T
    values = [matrix[i, i] for i in range(dimensions)]
    if structure == "unstructured":
        values += [matrix[i, j] for i in range(dimensions) for j in range(i)]
    return torch.stack(values)


def check_factor(theta, dimensions, structure):
    lower = factor(theta, dimensions, structure)
    covariance = lower @ lower.T
    sd = covariance.diagonal().sqrt()
    if float(sd.min()) < 1e-4 or not bool(torch.isfinite(sd).all()):
        raise AnalysisError(
            "boundary_solution", "A random-effect standard deviation is at zero or nonfinite."
        )
    corr = covariance / sd[:, None] / sd
    if float(torch.linalg.eigvalsh(corr).min()) < 1e-8:
        raise AnalysisError("boundary_solution", "Random-effect correlation is singular.")


def optimizer_record(fit, objective):
    return {
        "method": fit.method,
        "iterations": fit.iterations,
        "converged": fit.converged,
        "derivatives": "exact Torch chain rule of the declared likelihood",
        "full_observed_information": True,
        "work_used": objective.used,
        "max_work": objective.budget,
        **fit.diagnostics,
    }
