"""Panel count likelihoods of ``xtpoisson`` (tensors only).

Random effects (gamma, Hausman, Hall and Griliches 1984)
--------------------------------------------------------
y_it | nu_i ~ Poisson(nu_i lambda_it), lambda_it = exp(x_it'b + offset_it), and
nu_i ~ Gamma(mean 1, variance alpha). With a = 1/alpha, Y_i = sum_t y_it and
L_i = sum_t lambda_it the panel likelihood is closed form (negative binomial
in the panel totals):

    ln L_i = sum_t [y_it eta_it - ln y_it!] + ln Gamma(Y_i + a) - ln Gamma(a)
             - a ln(1 + L_i / a) - Y_i ln(L_i + a).

With S_i = sum_t lambda_it x_it and k_i = (Y_i + a) / (L_i + a):

    d/db       = sum_t y_it x_it - k_i S_i
    d2/db db'  = -k_i sum_t lambda_it x_it x_it' + k_i / (L_i + a) S_i S_i'
    d/da       = psi(Y_i + a) - psi(a) - ln(1 + L_i/a) + (L_i - Y_i)/(L_i + a)
    d2/da2     = psi'(Y_i + a) - psi'(a) + L_i/(a (a + L_i)) - (L_i - Y_i)/(L_i + a)^2
    d2/da db   = (Y_i - L_i)/(L_i + a)^2 S_i

and the parameter is ln alpha = -ln a (chain rule d/d ln alpha = -a d/da).

Fixed effects (conditional Poisson)
-----------------------------------
Conditioning on Y_i removes nu_i: (y_i1..y_iT) | Y_i is multinomial with
p_it = lambda_it / L_i, so with pbar_i = sum_t p_it x_it

    ln L_i = ln Y_i! - sum_t ln y_it! + sum_t y_it ln p_it
    d/db   = sum_t y_it x_it - Y_i pbar_i
    d2     = -Y_i [sum_t p_it x_it x_it' - pbar_i pbar_i'].

Panels with Y_i = 0 have L_i = 1 and are dropped by the caller.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.covariance import group_sums


def _rejected(size: int) -> tuple[Tensor, Tensor, Tensor]:
    return (torch.tensor(-math.inf, dtype=torch.float64), torch.zeros(size, dtype=torch.float64),
            -torch.eye(size, dtype=torch.float64))


class GammaPoissonObjective:
    """Random-effects (gamma) Poisson log likelihood in theta = (b, ln alpha)."""

    def __init__(self, x: Tensor, y: Tensor, codes: Tensor, n_groups: int,
                 offset: Tensor | None = None):
        self.x, self.y, self.codes, self.groups = x, y, codes, n_groups
        self.offset = torch.zeros_like(y) if offset is None else offset
        self.p = x.shape[1]
        self.size = self.p + 1
        self.totals = group_sums(y, codes, n_groups)
        self.constant = float((-torch.lgamma(y + 1)).sum())
        self.yx = x * y[:, None]

    def _parts(self, theta: Tensor):
        eta = self.x @ theta[:self.p] + self.offset
        lam = torch.exp(eta)
        a = torch.exp(-theta[self.p])
        big_l = group_sums(lam, self.codes, self.groups)
        y_sum = self.totals
        group = (torch.lgamma(y_sum + a) - torch.lgamma(a) - a * torch.log1p(big_l / a)
                 - y_sum * torch.log(big_l + a))
        values = group_sums(self.y * eta, self.codes, self.groups) + group
        return eta, lam, a, big_l, values

    def value(self, theta: Tensor) -> Tensor:
        value = self._parts(theta)[4].sum() + self.constant
        return value if bool(torch.isfinite(value)) else torch.tensor(-math.inf,
                                                                      dtype=torch.float64)

    def group_log_likelihood(self, theta: Tensor) -> Tensor:
        """Per-panel log likelihood (without the ln y! constants)."""
        return self._parts(theta)[4]

    def _derivatives(self, theta: Tensor, hessian: bool):
        p = self.p
        eta, lam, a, big_l, values = self._parts(theta)
        y_sum = self.totals
        s = group_sums(lam[:, None] * self.x, self.codes, self.groups)          # [G, p]
        k = (y_sum + a) / (big_l + a)
        score_b = group_sums(self.yx, self.codes, self.groups) - k[:, None] * s
        d_a = (torch.special.digamma(y_sum + a) - torch.special.digamma(a)
               - torch.log1p(big_l / a) + (big_l - y_sum) / (big_l + a))
        scores = torch.cat([score_b, (-a * d_a)[:, None]], dim=1)              # [G, p + 1]
        value = values.sum() + self.constant
        if not hessian:
            return value, scores.sum(dim=0), None, scores
        out = torch.zeros((p + 1, p + 1), dtype=torch.float64)
        weight = (k[self.codes] * lam)
        out[:p, :p] = -(self.x * weight[:, None]).T @ self.x + (s * (k / (big_l + a))[:, None]).T @ s
        d_aa = (torch.special.polygamma(1, y_sum + a) - torch.special.polygamma(1, a)
                + big_l / (a * (a + big_l)) - (big_l - y_sum) / (big_l + a).square())
        d_ab = ((y_sum - big_l) / (big_l + a).square())[:, None] * s
        out[p, p] = (a.square() * d_aa + a * d_a).sum()
        out[:p, p] = (-a * d_ab).sum(dim=0)
        out[p, :p] = out[:p, p]
        return value, scores.sum(dim=0), (out + out.T) / 2, scores

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, gradient, hessian, _ = self._derivatives(theta, True)
        if not (bool(torch.isfinite(value)) and bool(torch.isfinite(hessian).all())):
            return _rejected(self.size)
        return value, gradient, hessian

    def group_scores(self, theta: Tensor) -> Tensor:
        return self._derivatives(theta, False)[3]


class ConditionalPoissonObjective:
    """Conditional (fixed-effects) Poisson log likelihood in b."""

    def __init__(self, x: Tensor, y: Tensor, codes: Tensor, n_groups: int,
                 offset: Tensor | None = None):
        self.x, self.y, self.codes, self.groups = x, y, codes, n_groups
        self.offset = torch.zeros_like(y) if offset is None else offset
        self.size = x.shape[1]
        self.totals = group_sums(y, codes, n_groups)
        self.constant = float(torch.lgamma(self.totals + 1).sum() - torch.lgamma(y + 1).sum())
        self.yx = group_sums(x * y[:, None], codes, n_groups)

    def _probabilities(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        eta = self.x @ theta + self.offset
        top = torch.full((self.groups,), -math.inf, dtype=torch.float64).scatter_reduce(
            0, self.codes, eta, "amax")
        shifted = torch.exp(eta - top[self.codes])
        total = group_sums(shifted, self.codes, self.groups)
        log_p = eta - top[self.codes] - torch.log(total)[self.codes]
        return log_p, torch.exp(log_p)

    def value(self, theta: Tensor) -> Tensor:
        log_p, _ = self._probabilities(theta)
        value = (self.y * log_p).sum() + self.constant
        return value if bool(torch.isfinite(value)) else torch.tensor(-math.inf,
                                                                      dtype=torch.float64)

    def _derivatives(self, theta: Tensor, hessian: bool):
        log_p, prob = self._probabilities(theta)
        mean_x = group_sums(prob[:, None] * self.x, self.codes, self.groups)    # pbar_i
        scores = self.yx - self.totals[:, None] * mean_x
        value = (self.y * log_p).sum() + self.constant
        if not hessian:
            return value, scores.sum(dim=0), None, scores
        weight = self.totals[self.codes] * prob
        out = -((self.x * weight[:, None]).T @ self.x
                - (mean_x * self.totals[:, None]).T @ mean_x)
        return value, scores.sum(dim=0), (out + out.T) / 2, scores

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, gradient, hessian, _ = self._derivatives(theta, True)
        if not (bool(torch.isfinite(value)) and bool(torch.isfinite(hessian).all())):
            return _rejected(self.size)
        return value, gradient, hessian

    def group_scores(self, theta: Tensor) -> Tensor:
        return self._derivatives(theta, False)[3]
