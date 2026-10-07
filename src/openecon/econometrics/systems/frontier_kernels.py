"""Likelihoods of the stochastic frontier model with analytic scores and Hessians.

Model: ``y_i = x_i'b + v_i - s u_i`` with ``s = 1`` (production) or ``s = -1``
(cost), ``v ~ N(0, sigma_v^2)`` and ``u >= 0``; ``e_i = y_i - x_i'b``. With
``G = ln Phi``, ``M = phi/Phi`` and ``G'' = -M (M + z)`` from
``discrete.kernels.mills_ratio`` (stable in both tails), the per-observation
log densities are (Stata's [R] frontier parameterizations):

half-normal, ``u ~ N+(0, sigma_u^2)``, ancillaries ``p = ln sigma_v^2``, ``q = ln sigma_u^2``
    ``l = 1/2 ln(2/pi) - 1/2 ln S + G(a) - e^2/(2S)``,
    ``S = e^p + e^q``, ``a = -s e A``, ``A = sigma_u / (sigma_v sqrt(S))``.

exponential, ``u ~ Exp(sigma_u)`` (mean sigma_u), the same ``p, q``
    ``l = -q/2 + c^2/2 + s e / sigma_u + G(-s e / sigma_v - c)``, ``c = sigma_v / sigma_u``.

truncated normal, ``u ~ N+(mu, sigma_u^2)``, ancillaries ``mu``, ``t = ln sigma^2``,
``g = logit(gamma)`` with ``sigma^2 = sigma_u^2 + sigma_v^2``, ``gamma = sigma_u^2 / sigma^2``
    ``l = -1/2 ln(2 pi) - t/2 - (s e + mu)^2 / (2 sigma^2) + G(z_1) - G(z_2)``,
    ``z_1 = mu e^{-(g+t)/2} - s e e^{(g-t)/2}`` (= mu*/sigma*), ``z_2 = mu / sigma_u``.

Every log density depends on the coefficients only through ``e``, so with the
per-observation derivatives ``l_e``, ``l_ee``, ``l_a``, ``l_ea``, ``l_aa`` (``a``
the ancillaries) the score and Hessian are

    dl/db = -X'(w l_e),  d2l/db db' = X' diag(w l_ee) X,  d2l/db da = -X'(w l_ea),
    dl/da = sum w l_a,   d2l/da da' = sum w l_aa,

which costs O(n k^2). The derivatives of the inner quantities are written
out in closed form below; the tests compare them with numerical ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.engines.contracts import KernelError

_HALF_LOG_2_OVER_PI = 0.5 * math.log(2 / math.pi)
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
ANCILLARIES = {"hnormal": ("/lnsig2v", "/lnsig2u"), "exponential": ("/lnsig2v", "/lnsig2u"),
               "tnormal": ("/mu", "/lnsigma2", "/ilgtgamma")}


@dataclass
class Pieces:
    value: Tensor      # [n] log densities
    e1: Tensor         # [n] dl/de
    e2: Tensor         # [n] d2l/de2
    a1: Tensor         # [n, m] dl/da
    ea: Tensor         # [n, m] d2l/de da
    aa: Tensor         # [n, m, m] d2l/da da'


def _outer(left: Tensor, right: Tensor) -> Tensor:
    return left[:, :, None] * right[:, None, :]


def _half_normal(e: Tensor, anc: Tensor, s: float, second: bool) -> Pieces:
    p, q = anc[0], anc[1]
    v_var, u_var = torch.exp(p), torch.exp(q)
    total = v_var + u_var
    big_a = torch.sqrt(u_var / (v_var * total))
    a = -s * e * big_a
    log_cdf, ratio, curve = mills_ratio(a)
    value = _HALF_LOG_2_OVER_PI - 0.5 * torch.log(total) + log_cdf - e.square() / (2 * total)
    n = e.shape[0]
    lp, lq = v_var / total, u_var / total                       # d ln S / dp, dq
    dl_s = torch.stack([lp, lq])                                # [2]
    dln_a = torch.stack([-0.5 - lp / 2, 0.5 - lq / 2])          # d ln A / d(p, q)
    a_e = -s * big_a
    a_t = a[:, None] * dln_a[None, :]                           # [n, 2]
    t_half = e.square() / (2 * total)
    e1 = ratio * a_e - e / total
    a1 = -0.5 * dl_s[None, :] + ratio[:, None] * a_t + t_half[:, None] * dl_s[None, :]
    if not second:
        return Pieces(value, e1, e1, a1, a1, a1[:, :, None])
    e2 = -curve * a_e.square() - 1 / total
    ea = (-curve * a_e)[:, None] * a_t + (ratio * a_e)[:, None] * dln_a[None, :] \
        + (e / total)[:, None] * dl_s[None, :]
    cross = v_var * u_var / total.square()
    d2l_s = torch.stack([torch.stack([cross, -cross]), torch.stack([-cross, cross])])
    d2ln_a = -d2l_s / 2
    a_tt = a[:, None, None] * (_outer(dln_a.expand(n, 2), dln_a.expand(n, 2)) + d2ln_a)
    aa = -0.5 * d2l_s.expand(n, 2, 2) - curve[:, None, None] * _outer(a_t, a_t) \
        + ratio[:, None, None] * a_tt \
        - t_half[:, None, None] * (_outer(dl_s.expand(n, 2), dl_s.expand(n, 2)) - d2l_s)
    return Pieces(value, e1, e2, a1, ea, aa)


def _exponential(e: Tensor, anc: Tensor, s: float, second: bool) -> Pieces:
    p, q = anc[0], anc[1]
    c = torch.exp((p - q) / 2)
    r1, r2 = torch.exp(-q / 2), torch.exp(-p / 2)
    b = -s * e * r2 - c
    log_cdf, ratio, curve = mills_ratio(b)
    value = -q / 2 + c.square() / 2 + s * e * r1 + log_cdf
    n = e.shape[0]
    zeros = torch.zeros(n, dtype=torch.float64)
    b_e = -s * r2
    b_a = torch.stack([s * e * r2 / 2 - c / 2, (c / 2).expand(n)], dim=1)
    pe_e = s * r1
    pe_a = torch.stack([(c.square() / 2).expand(n), -0.5 - c.square() / 2 - s * e * r1 / 2], dim=1)
    e1 = pe_e + ratio * b_e
    a1 = pe_a + ratio[:, None] * b_a
    if not second:
        return Pieces(value, e1, e1, a1, a1, a1[:, :, None])
    e2 = -curve * b_e.square()
    b_ea = torch.stack([(s * r2 / 2).expand(n), zeros], dim=1)
    pe_ea = torch.stack([zeros, (-s * r1 / 2).expand(n)], dim=1)
    ea = pe_ea - (curve * b_e)[:, None] * b_a + ratio[:, None] * b_ea
    b_pp = -s * e * r2 / 4 - c / 4
    b_aa = torch.stack([torch.stack([b_pp, (c / 4).expand(n)], dim=1),
                        torch.stack([(c / 4).expand(n), (-c / 4).expand(n)], dim=1)], dim=1)
    c2 = (c.square() / 2).expand(n)
    pe_aa = torch.stack([torch.stack([c2, -c2], dim=1),
                         torch.stack([-c2, c2 + s * e * r1 / 4], dim=1)], dim=1)
    aa = pe_aa - curve[:, None, None] * _outer(b_a, b_a) + ratio[:, None, None] * b_aa
    return Pieces(value, e1, e2, a1, ea, aa)


def _truncated_normal(e: Tensor, anc: Tensor, s: float, second: bool) -> Pieces:
    mu, t, g = anc[0], anc[1], anc[2]
    n = e.shape[0]
    ones, zeros = torch.ones(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    a_ = torch.exp(-(g + t) / 2)
    b_ = torch.exp((g - t) / 2)
    gamma = torch.sigmoid(g)
    inv = torch.exp(-t)
    z1 = mu * a_ - s * e * b_
    dlog_d = -(1 - gamma) / 2
    c2 = torch.exp(-t / 2) * torch.sqrt(1 + torch.exp(-g))
    z2 = mu * c2
    r = s * e + mu
    big_q = r.square() * inv / 2
    log1, m1, v1 = mills_ratio(z1)
    log2, m2, v2 = mills_ratio(z2.reshape(1))
    m2, v2 = m2[0], v2[0]
    value = -_HALF_LOG_2PI - t / 2 - big_q + log1 - log2[0]
    # Partial derivatives in the order (mu, t, g); e is handled separately.
    z1_e = (-s * b_).expand(n)
    z1_a = torch.stack([a_.expand(n), -z1 / 2, -mu * a_ / 2 - s * e * b_ / 2], dim=1)
    z2_a = torch.stack([c2, -z2 / 2, z2 * dlog_d])
    q_e = s * r * inv
    q_a = torch.stack([r * inv, -big_q, zeros], dim=1)
    unit_t = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
    e1 = -q_e + m1 * z1_e
    a1 = -unit_t[None, :] - q_a + m1[:, None] * z1_a - m2 * z2_a[None, :]
    if not second:
        return Pieces(value, e1, e1, a1, a1, a1[:, :, None])
    e2 = -inv * ones - v1 * z1_e.square()
    z1_ea = torch.stack([zeros, (s * b_ / 2).expand(n), (-s * b_ / 2).expand(n)], dim=1)
    q_ea = torch.stack([(s * inv).expand(n), -s * r * inv, zeros], dim=1)
    ea = -q_ea - (v1 * z1_e)[:, None] * z1_a + m1[:, None] * z1_ea
    z1_g = z1_a[:, 2]
    z1_aa = torch.stack([
        torch.stack([zeros, (-a_ / 2).expand(n), (-a_ / 2).expand(n)], dim=1),
        torch.stack([(-a_ / 2).expand(n), z1 / 4, -z1_g / 2], dim=1),
        torch.stack([(-a_ / 2).expand(n), -z1_g / 2, mu * a_ / 4 - s * e * b_ / 4], dim=1),
    ], dim=1)
    z2_g = z2 * dlog_d
    d2log_d = gamma * (1 - gamma) / 2
    z2_aa = torch.stack([
        torch.stack([torch.zeros((), dtype=torch.float64), -c2 / 2, c2 * dlog_d]),
        torch.stack([-c2 / 2, z2 / 4, -z2_g / 2]),
        torch.stack([c2 * dlog_d, -z2_g / 2, z2 * (dlog_d.square() + d2log_d)]),
    ])
    q_aa = torch.stack([
        torch.stack([inv * ones, -r * inv, zeros], dim=1),
        torch.stack([-r * inv, big_q, zeros], dim=1),
        torch.stack([zeros, zeros, zeros], dim=1),
    ], dim=1)
    aa = -q_aa - v1[:, None, None] * _outer(z1_a, z1_a) + m1[:, None, None] * z1_aa \
        + (v2 * torch.outer(z2_a, z2_a) - m2 * z2_aa)[None, :, :]
    return Pieces(value, e1, e2, a1, ea, aa)


_PIECES = {"hnormal": _half_normal, "exponential": _exponential, "tnormal": _truncated_normal}


class FrontierObjective:
    """Weighted log likelihood of a stochastic frontier, ``theta = (b, ancillaries)``."""

    def __init__(self, x: Tensor, y: Tensor, weights: Tensor, distribution: str, cost: bool):
        if distribution not in _PIECES:
            raise KernelError("invalid_option", f"Unknown inefficiency distribution "
                              f"'{distribution}'.")
        self.x, self.y, self.w = x, y, weights
        self.k = x.shape[1]
        self.m = len(ANCILLARIES[distribution])
        self.sign = -1.0 if cost else 1.0
        self.pieces_fn = _PIECES[distribution]

    def pieces(self, theta: Tensor, second: bool = True) -> Pieces:
        e = self.y - self.x @ theta[:self.k]
        return self.pieces_fn(e, theta[self.k:], self.sign, second)

    def value(self, theta: Tensor) -> Tensor:
        pieces = self.pieces(theta, second=False)
        total = self.w @ pieces.value
        return total if bool(torch.isfinite(total)) else torch.tensor(-math.inf,
                                                                       dtype=torch.float64)

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pc = self.pieces(theta)
        w, x = self.w, self.x
        value = w @ pc.value
        gradient = torch.cat([-(x.T @ (w * pc.e1)), w @ pc.a1])
        bb = (x * (w * pc.e2)[:, None]).T @ x
        ba = -(x.T @ (w[:, None] * pc.ea))
        aa = (w[:, None, None] * pc.aa).sum(dim=0)
        hessian = torch.cat([torch.cat([bb, ba], dim=1), torch.cat([ba.T, aa], dim=1)], dim=0)
        hessian = (hessian + hessian.T) / 2
        if not (bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all())
                and bool(torch.isfinite(hessian).all())):
            size = self.k + self.m
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(size, dtype=torch.float64), -torch.eye(size, dtype=torch.float64))
        return value, gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores before weights, ``[n, k + m]``."""
        pc = self.pieces(theta, second=False)
        return torch.cat([-self.x * pc.e1[:, None], pc.a1], dim=1)


def conditional_inefficiency(e: Tensor, distribution: str, cost: bool, sigma_v2: float,
                             sigma_u2: float, mu: float = 0.0) -> tuple[Tensor, Tensor]:
    """``(mu*_i, sigma*)`` of the distribution of ``u_i`` given ``e_i`` (Jondrow et al. 1982).

    ``u | e ~ N+(mu*, sigma*^2)``: half-normal ``mu* = -s e su2/s2``,
    ``sigma* = su sv / s``; exponential ``mu* = -s e - sv2/su``, ``sigma* = sv``;
    truncated normal ``mu* = (mu sv2 - s e su2)/s2``, ``sigma*`` as half-normal.
    """
    s = -1.0 if cost else 1.0
    total = sigma_u2 + sigma_v2
    if distribution == "exponential":
        return -s * e - sigma_v2 / math.sqrt(sigma_u2), torch.tensor(math.sqrt(sigma_v2),
                                                                     dtype=torch.float64)
    centre = (mu * sigma_v2 - s * e * sigma_u2) / total
    return centre, torch.tensor(math.sqrt(sigma_u2 * sigma_v2 / total), dtype=torch.float64)
