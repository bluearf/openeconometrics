"""Scalar random-intercept likelihood by adaptive Gauss-Hermite quadrature (tensors only).

For group g with rows i, a conditional density f(y_i | eta_i + s v) with linear
predictor eta_i, random-intercept standard deviation s and v ~ N(0, 1),

    L_g = int prod_i f(y_i | eta_i + s v) phi(v) dv.

Mode-curvature adaptive quadrature (Liu and Pierce 1994; ordinal::clmm with
nAGQ > 1; Stata's ``mcaghermite``): with h_g(v) = sum_i ln f + ln phi(v), its
mode v_g, curvature c_g = -h_g''(v_g), scale t_g = c_g^-1/2 and Gauss-Hermite
nodes/weights (x_k, w_k) for e^{-x^2},

    ln L_g = ln(sqrt(2) t_g) + logsumexp_k [ln w_k + x_k^2 + h_g(v_g + sqrt(2) t_g x_k)].

The modes are recomputed at every parameter value, so the approximation is a
smooth function of the parameters (the clmm/lme4 convention, unlike Stata's
fixed-adaptation iterations). The mode is found by damped Newton steps without
a derivative tape; two further Newton steps are then taken on the tape. Since
the Newton map has zero derivative at its fixed point, the taped mode carries
exact first and second parameter derivatives, so Torch's chain rule yields the
exact gradient and Hessian of the quadrature likelihood. ``ghermite`` is the
non-adaptive rule: nodes sqrt(2) x_k around zero for every group.

``pieces(eta) -> (ln f, d ln f/d eta, d2 ln f/d eta2)`` is elementwise and must
be log-concave in eta (ordinal logit/probit and censored normal are), which
makes h_g strictly concave and its mode unique.
"""

from __future__ import annotations

import math

import torch

from openecon.engines.contracts import KernelError
from openecon.engines.distributions import gauss_hermite

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
SQRT2 = math.sqrt(2.0)


def group_sums(values, codes, count):
    """Differentiable sums of rows (first axis) by group code."""
    out = torch.zeros((count, *values.shape[1:]), dtype=values.dtype)
    return out.index_add(0, codes, values)


class RandomInterceptQuadrature:
    def __init__(self, codes, count: int, points: int, adaptive: bool = True):
        if points < 1:
            raise KernelError("invalid_option", "At least one quadrature point is required.")
        self.codes, self.count, self.points, self.adaptive = codes, count, points, adaptive
        nodes, weights = gauss_hermite(points)
        self.nodes = nodes
        self.node_terms = torch.log(weights) + nodes.square()

    def _h(self, pieces, eta, sigma, v):
        logf = pieces(eta + sigma * v[self.codes])[0]
        return group_sums(logf, self.codes, self.count) - v.square() / 2

    def mode(self, pieces, eta, sigma):
        """Posterior mode of every group's standardized random effect (no tape)."""
        with torch.no_grad():
            v = torch.zeros(self.count, dtype=torch.float64)
            current = self._h(pieces, eta, sigma, v)
            for _ in range(200):
                _, d1, d2 = pieces(eta + sigma * v[self.codes])
                gradient = sigma * group_sums(d1, self.codes, self.count) - v
                curvature = 1 - sigma.square() * group_sums(d2, self.codes, self.count)
                if not bool(torch.isfinite(gradient).all()) or not bool((curvature > 0).all()):
                    raise KernelError("numerical_failure", "A group's random-effect posterior "
                                      "is not log-concave or not finite at these parameters.")
                step = gradient / curvature
                factor = torch.ones_like(v)
                for _ in range(60):
                    trial = v + factor * step
                    value = self._h(pieces, eta, sigma, trial)
                    worse = ~(value >= current - 1e-13 * current.abs())
                    if not bool(worse.any()):
                        break
                    factor = torch.where(worse, factor / 2, factor)
                v, current = trial, value
                if float((factor * step).abs().max()) < 1e-13 * (1 + float(v.abs().max())):
                    break
            else:
                raise KernelError("numerical_failure", "The random-effect modes did not converge.")
        return v

    def log_likelihood(self, pieces, eta, sigma):
        """Sum over groups of the (adaptive) quadrature log likelihood; differentiable."""
        if self.adaptive:
            v = self.mode(pieces, eta.detach(), sigma.detach())
            for _ in range(2):
                _, d1, d2 = pieces(eta + sigma * v[self.codes])
                gradient = sigma * group_sums(d1, self.codes, self.count) - v
                curvature = 1 - sigma.square() * group_sums(d2, self.codes, self.count)
                v = v + gradient / curvature
            _, _, d2 = pieces(eta + sigma * v[self.codes])
            curvature = 1 - sigma.square() * group_sums(d2, self.codes, self.count)
            scale = SQRT2 * curvature.rsqrt()
            location = v
        else:
            scale = torch.full((self.count,), SQRT2, dtype=torch.float64)
            location = torch.zeros(self.count, dtype=torch.float64)
        nodes = location[:, None] + scale[:, None] * self.nodes[None, :]          # [S, K]
        logf = pieces(eta[:, None] + sigma * nodes[self.codes])[0]                 # [n, K]
        terms = (group_sums(logf, self.codes, self.count) + self.node_terms[None, :]
                 - nodes.square() / 2 - _HALF_LOG_2PI)
        return (torch.log(scale) + torch.logsumexp(terms, dim=1)).sum()

    def posterior_means(self, pieces, eta, sigma):
        """Empirical Bayes means E[s v | y] of every group (no tape)."""
        with torch.no_grad():
            if self.adaptive:
                v = self.mode(pieces, eta, sigma)
                _, _, d2 = pieces(eta + sigma * v[self.codes])
                curvature = 1 - sigma.square() * group_sums(d2, self.codes, self.count)
                scale = SQRT2 * curvature.rsqrt()
            else:
                v = torch.zeros(self.count, dtype=torch.float64)
                scale = torch.full((self.count,), SQRT2, dtype=torch.float64)
            nodes = v[:, None] + scale[:, None] * self.nodes[None, :]
            logf = pieces(eta[:, None] + sigma * nodes[self.codes])[0]
            terms = (group_sums(logf, self.codes, self.count) + self.node_terms[None, :]
                     - nodes.square() / 2)
            weights = torch.softmax(terms, dim=1)
            return sigma * (weights * nodes).sum(1)
