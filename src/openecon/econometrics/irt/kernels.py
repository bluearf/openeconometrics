"""CPU float64 MML, identified item geometry and observed information.

No empirical-Bayes fitting of the latent distribution: mean=0, variance=1.
Gauss-Hermite nodes use the standard-normal Jacobi matrix (Golub-Welsch).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from openecon.analysis_contracts import AnalysisError

DTYPE = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DTYPE, device="cpu")


def quadrature(points):
    matrix = torch.zeros((points, points), dtype=DTYPE, device="cpu")
    index = torch.arange(points-1, device="cpu")
    off = torch.arange(1, points, dtype=DTYPE, device="cpu").sqrt()
    matrix[index, index+1] = off
    matrix[index+1, index] = off
    nodes, vectors = torch.linalg.eigh(matrix)
    weights = vectors[0].square()
    return nodes, weights / weights.sum()


def decode(raw, family, categories):
    """Return item discriminations/steps and natural parameters (RSM sum=0)."""
    discriminations, steps, natural = [], [], []
    cursor = 0
    if family == "rsm":
        j, k = len(categories), categories[0]
        shared = torch.cat((raw[j:], -raw[j:].sum().reshape(1)))
        for i in range(j):
            discriminations.append(raw.new_tensor(1.))
            steps.append(raw[i] + shared)
        return discriminations, steps, torch.cat((raw[:j], shared))
    for k in categories:
        if family in ("2pl", "3pl", "grm"):
            a = raw[cursor].exp()
            cursor += 1
            natural.append(a.reshape(1))
        else:
            a = raw.new_tensor(1.)
        if family == "grm":
            first = raw[cursor:cursor+1]
            gaps = raw[cursor+1:cursor+k-1].exp()
            b = torch.cat((first, first + gaps.cumsum(0)))
        else:
            b = raw[cursor:cursor+k-1]
        cursor += k-1
        discriminations.append(a)
        steps.append(b)
        natural.append(b)
    return discriminations, steps, torch.cat(natural)


def log_probabilities(raw, family, categories, guessing, theta):
    a, thresholds, _ = decode(raw, family, categories)
    output = []
    for i, k in enumerate(categories):
        z = a[i] * (theta[:, None] - thresholds[i][None, :])
        if family in ("rasch", "2pl", "3pl"):
            c = guessing[i]
            if c:
                log_yes = torch.logaddexp(theta.new_tensor(c).log(),
                                         theta.new_tensor(1-c).log() + F.logsigmoid(z[:, 0]))
                log_no = theta.new_tensor(1-c).log() + F.logsigmoid(-z[:, 0])
            else:
                log_yes, log_no = F.logsigmoid(z[:, 0]), F.logsigmoid(-z[:, 0])
            logp = torch.stack((log_no, log_yes), 1)
        elif family == "grm":
            # Stable differences of cumulative logits, even in tail categories.
            middle = (F.logsigmoid(z[:, :-1]) + F.logsigmoid(-z[:, 1:])
                      + torch.log(-torch.expm1(z[:, 1:] - z[:, :-1])))
            logp = torch.cat((F.logsigmoid(-z[:, :1]), middle,
                              F.logsigmoid(z[:, -1:])), 1)
        else:
            logits = torch.cat((torch.zeros_like(theta[:, None]), z.cumsum(1)), 1)
            logp = torch.log_softmax(logits, 1)
        output.append(logp)
    return output


def pattern_loglik(raw, family, categories, guessing, theta, responses):
    logp = log_probabilities(raw, family, categories, guessing, theta)
    total = torch.zeros((len(responses), len(theta)), dtype=DTYPE, device="cpu")
    for j, lp in enumerate(logp):
        observed = responses[:, j] >= 0
        # -1 is an explicitly missing scoring item, contributing no log factor.
        total = total + lp[:, responses[:, j].clamp_min(0)].T * observed[:, None]
    return total


def objective(raw, family, categories, guessing, nodes, weights, responses):
    ll = pattern_loglik(raw, family, categories, guessing, nodes, responses)
    return -torch.logsumexp(ll + weights.log(), 1).sum()


def parameter_count(family, categories):
    if family == "rsm":
        return len(categories) + categories[0] - 2
    return sum(k-1 + int(family in ("2pl", "3pl", "grm")) for k in categories)


def initialize(responses, family, categories, guessing):
    if family == "rsm":
        return torch.zeros(parameter_count(family, categories), dtype=DTYPE, device="cpu")
    values = []
    for j, k in enumerate(categories):
        if family in ("2pl", "3pl", "grm"):
            values.append(0.)
        if family in ("rasch", "2pl", "3pl"):
            mean = float(responses[:, j].double().mean())
            adjusted = min(.95, max(.05, (mean-guessing[j])/(1-guessing[j])))
            values.append(float(-torch.logit(tensor(adjusted))))
        elif family == "grm":
            threshold = tensor([-torch.logit(tensor(float((responses[:, j] >= s).double().mean())))
                                for s in range(1, k)])
            values.extend([float(threshold[0]), *threshold.diff().log().tolist()])
        else:
            values.extend([0.]*(k-1))
    return tensor(values)


def fit(responses, family, categories, guessing, points, max_iter, max_eval, tolerance):
    nodes, weights = quadrature(points)
    raw = initialize(responses, family, categories, guessing).requires_grad_()
    def function(value):
        return objective(value, family, categories, guessing, nodes, weights, responses)
    calls = 0
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=max_iter, max_eval=max_eval,
                                  tolerance_grad=tolerance, tolerance_change=1e-12,
                                  history_size=20, line_search_fn="strong_wolfe")

    def closure():
        nonlocal calls
        calls += 1
        if calls > max_eval:
            raise AnalysisError("work_limit", "IRT likelihood evaluation budget exhausted; no converged result returned.")
        optimizer.zero_grad()
        loss = function(raw)
        if not bool(torch.isfinite(loss)):
            raise AnalysisError("numerical_failure", "Non-finite IRT likelihood; no boundary fit is reported.")
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
        nll = function(raw)
        gradient = torch.autograd.grad(nll, raw)[0]
        gmax = float(gradient.abs().max())
        if not bool(torch.isfinite(raw).all()) or gmax > max(tolerance*10, 2e-5):
            raise AnalysisError("nonconvergence", f"IRT likelihood has not converged (max gradient {gmax:.3g}).")
        hessian = torch.autograd.functional.hessian(function, raw)
        hessian = (hessian+hessian.T)/2
        eigenvalues = torch.linalg.eigvalsh(hessian)
        if not bool(torch.isfinite(hessian).all()) or float(eigenvalues[0]) <= max(1e-7, float(eigenvalues[-1])*1e-9):
            raise AnalysisError("unidentified_model", "IRT observed information is not positive definite or is ill-conditioned.")
        covariance_raw = torch.linalg.inv(hessian)
        def natural(value):
            return decode(value, family, categories)[2]
        jacobian = torch.autograd.functional.jacobian(natural, raw)
        estimates = natural(raw)
        covariance = jacobian @ covariance_raw @ jacobian.T
    a, b, _ = decode(raw.detach(), family, categories)
    if any(not .1 < float(v) < 5. for v in a) or any(bool((v.abs() > 8.).any()) for v in b):
        raise AnalysisError("boundary_fit", "Supported interior requires .1 < discrimination < 5 and |item threshold| <= 8.")
    # Audit a denser normal quadrature at the accepted solution. This is a
    # likelihood integration audit, not a claim about parameter refit accuracy.
    audit_nodes, audit_weights = quadrature(min(121, 2*points+1))
    audit = objective(raw.detach(), family, categories, guessing, audit_nodes, audit_weights, responses)
    error = abs(float(audit-nll.detach()))/len(responses)
    if error > 1e-5:
        raise AnalysisError("quadrature_accuracy", f"IRT log likelihood integration audit exceeds 1e-5 per person ({error:.3g}); increase points.")
    return dict(raw=raw.detach(), estimates=estimates.detach(), covariance=covariance.detach(),
                hessian=hessian.detach(), raw_covariance=covariance_raw.detach(), nodes=nodes, weights=weights,
                log_likelihood=-float(nll.detach()), max_gradient=gmax, evaluations=calls,
                iterations=optimizer.state[raw]["n_iter"], min_information_eigenvalue=float(eigenvalues[0]),
                quadrature_error_per_person=error, audit_points=len(audit_nodes))
