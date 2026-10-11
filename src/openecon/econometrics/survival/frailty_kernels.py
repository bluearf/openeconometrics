"""Shared-frailty likelihood kernels on float64 tensors (no autograd).

Cox shared frailty (penalized partial likelihood)
-------------------------------------------------
Record i of shared group g(i) (q groups) has the linear predictor
``eta_i = x_i'b + w_{g(i)} + offset_i`` with ``w_g`` the log frailty of group
g. The partial likelihood PL(b, w) is the Cox partial likelihood of the design
``Z = [X, G]`` (G the q group indicators) with Breslow or Efron ties, as in
``cox_kernels``. ``G`` is never formed: its blocks are group sums
(``index_add_``) and two [T, q] risk-set blocks, the risk-set totals
``S1w[k, g] = sum_{i in R_k, g(i) = g} u_i`` (suffix sums of per-time
exit/entry contributions) and, for Efron ties, the failure totals
``D1w[k, g]``. Their cost is bounded by ``FRAILTY_CELL_LIMIT`` cells.

For a frailty variance theta the penalized partial likelihood is
``PPL = PL(b, w) - P(w; theta)`` with (Therneau, Grambsch and Pankratz 2003)

* gamma frailty (mean 1, variance theta, nu = 1/theta):
  ``P = nu sum_g (exp(w_g) - w_g)``;
* log-normal frailty (``w_g ~ N(0, theta)``): ``P = sum_g w_g^2 / (2 theta)``.

At the PPL maximum the integrated (marginal) log partial likelihood is

* gamma: ``L(theta) = PL(b, w) + nu sum_g w_g + sum_g c_g(nu)`` with
  ``c_g(nu) = D_g + nu ln nu + ln Gamma(nu + D_g) - ln Gamma(nu)
  - (nu + D_g) ln(nu + D_g)``, D_g the number of failures of group g. This is
  R survival's ``coxph(..., frailty(g, distribution='gamma'))`` I-likelihood
  (``frailty.gammacon``); with Breslow ties it is the exact gamma-frailty
  marginal likelihood at the nonparametric baseline hazard, shifted by the
  constant that makes ``L(0)`` the Cox partial likelihood. ``c_g`` is
  evaluated as ``D_g - nu log1p(D_g/nu) + sum_{j<D_g} log1p((j - D_g)/(nu + D_g))``,
  which is free of cancellation for every nu.
* log-normal: the Laplace approximation of ``ln int exp(PL(b, w)) phi(w; theta) dw``
  at ``(b, w)`` maximizing the PPL, ``L(theta) = PPL - 1/2 ln det(I + theta
  J_ww)``, ``J_ww`` the negative ww block of the PL Hessian (Ripatti and
  Palmgren 2000; the integrated partial likelihood of R ``coxme``).

Both are normalized so that ``L(0)`` is the Cox partial likelihood.

Parametric shared frailty (closed-form marginal likelihood)
-----------------------------------------------------------
With conditional hazard ``z h(t)`` and cumulative hazard ``z H(t)`` the
marginal contribution of group g with D_g failures and ``A_g = sum H(t_i)`` is
``sum_i delta_i ln h(t_i) + ln M_{D_g}(A_g)``, ``M_m(s) = E[Z^m exp(-s Z)]``:

* gamma (mean 1, variance theta, nu = 1/theta):
  ``ln M_m(s) = nu ln nu + ln Gamma(nu + m) - ln Gamma(nu) - (nu + m) ln(nu + s)``;
* inverse Gaussian (mean 1, variance theta), ``u = sqrt(1 + 2 theta s)``:
  ``ln M_m(s) = (1 - u)/theta - m ln u + ln S_n(theta / (2u))`` with
  ``S_n(w) = sum_{k=0}^{n} (n + k)! / (k! (n - k)!) w^k``, ``n = m - 1`` for
  ``m >= 1``, from the half-integer modified Bessel function
  ``K_{n+1/2}`` (finite sum; DLMF 10.49.12).

The first and second derivatives in ``s`` and ``ln theta`` are analytic.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.survival.data import FLOAT, RiskSets
from openecon.engines.contracts import KernelError

# Largest number of cells (failure times x groups) of one [T, q] risk-set block.
FRAILTY_CELL_LIMIT = 10_000_000


def _flip_cumsum(values: Tensor) -> Tensor:
    return values.flip(0).cumsum(0).flip(0)


class FrailtyCox:
    """Cox partial likelihood in ``(b, w)`` with value, gradient and Hessian.

    ``x`` [n, k] (centred), ``groups`` [n] int64 codes 0..q-1, ``risk`` the risk
    sets of the records, ``ties`` ``'breslow'`` or ``'efron'``; no weights.
    """

    def __init__(self, x: Tensor, groups: Tensor, q: int, risk: RiskSets,
                 offset: Tensor | None, ties: str):
        if ties not in {"breslow", "efron"}:
            raise KernelError("invalid_option", f"Unknown ties method {ties!r}.")
        cells = (risk.n_events + 1) * q
        if cells > FRAILTY_CELL_LIMIT:
            raise KernelError(
                "resource_limit",
                f"The frailty risk-set blocks need {cells:,} cells (failure times x groups; "
                f"limit {FRAILTY_CELL_LIMIT:,}). Use fewer shared groups or coarser times.")
        self.x, self.g, self.q, self.risk, self.offset, self.ties = x, groups, q, risk, offset, ties
        self.k = x.shape[1]
        failing = risk.event_of >= 0
        self.failing = failing
        self.fail_float = failing.to(FLOAT)
        self.event_of = risk.event_of
        self.d = risk.at_event(self.fail_float)
        self.group_events = torch.zeros(q, dtype=FLOAT).index_add_(0, groups, self.fail_float)
        self.g_const = torch.cat([(x * self.fail_float[:, None]).sum(0), self.group_events])
        if ties == "efron":
            counts = self.d.round().to(torch.int64)
            times = torch.arange(risk.n_events)
            self.term_time = torch.repeat_interleave(times, counts)
            starts = torch.cumsum(counts, 0) - counts
            position = torch.arange(self.term_time.numel()) - starts[self.term_time]
            self.term_frac = position.to(FLOAT) / counts[self.term_time].to(FLOAT)
            self.counts = counts

    # ---- pieces ------------------------------------------------------------------

    def eta(self, params: Tensor) -> Tensor:
        eta = self.x @ params[:self.k] + params[self.k:][self.g]
        return eta if self.offset is None else eta + self.offset

    def _scores(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta = self.eta(params)
        eta = eta - eta.max()
        v = torch.exp(eta)
        return eta, v, self.risk.at_risk(v)

    def _group_at_risk(self, v: Tensor) -> Tensor:
        """``S1w[k, g] = sum_{i in R_k, g(i) = g} v_i`` as suffix sums ([T, q])."""
        risk, total = self.risk, self.risk.n_events
        table = torch.zeros((total + 1, self.q), dtype=FLOAT)
        table.index_put_((risk.exit_count, self.g), v, accumulate=True)
        table.index_put_((risk.entry_count, self.g), -v, accumulate=True)
        return _flip_cumsum(table)[1:]

    def _group_at_event(self, v: Tensor) -> Tensor:
        fail = self.failing
        out = torch.zeros((self.risk.n_events, self.q), dtype=FLOAT)
        return out.index_put_((self.event_of[fail], self.g[fail]), v[fail], accumulate=True)

    def _efron_phi(self, v: Tensor, s0: Tensor) -> Tensor:
        d0 = self.risk.at_event(v * self.fail_float)
        return s0[self.term_time] - self.term_frac * d0[self.term_time]

    def value(self, params: Tensor) -> Tensor:
        eta, v, s0 = self._scores(params)
        first = (self.fail_float * eta).sum()
        if self.ties == "breslow":
            safe = torch.where(self.d > 0, s0, torch.ones_like(s0))
            return first - (self.d * torch.log(safe)).sum()
        return first - torch.log(self._efron_phi(v, s0)).sum()

    def _zwz(self, weight: Tensor) -> Tensor:
        """``Z' diag(weight) Z`` for ``Z = [X, G]``."""
        x, g, q, k = self.x, self.g, self.q, self.k
        out = torch.zeros((k + q, k + q), dtype=FLOAT)
        out[:k, :k] = (x * weight[:, None]).T @ x
        gx = torch.zeros((q, k), dtype=FLOAT).index_add_(0, g, x * weight[:, None])
        out[k:, :k] = gx
        out[:k, k:] = gx.T
        out[k:, k:] = torch.diag(torch.zeros(q, dtype=FLOAT).index_add_(0, g, weight))
        return out

    @staticmethod
    def _outer(c: Tensor, a: Tensor, b: Tensor) -> Tensor:
        return (a * c[:, None]).T @ b

    def __call__(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, risk, g, q = self.x, self.risk, self.g, self.q
        eta, v, s0 = self._scores(params)
        s1 = torch.cat([risk.at_risk(v[:, None] * x), self._group_at_risk(v)], dim=1)
        first = (self.fail_float * eta).sum()
        if self.ties == "breslow":
            safe = torch.where(self.d > 0, s0, torch.ones_like(s0))
            value = first - (self.d * torch.log(safe)).sum()
            a = self.d / safe
            weight = v * risk.accumulate(a)
            gradient = self.g_const - torch.cat(
                [x.T @ weight, torch.zeros(q, dtype=FLOAT).index_add_(0, g, weight)])
            hessian = -self._zwz(weight) + self._outer(a / safe, s1, s1)
            return value, gradient, (hessian + hessian.T) / 2
        phi = self._efron_phi(v, s0)
        value = first - torch.log(phi).sum()
        inverse, frac, total = 1.0 / phi, self.term_frac, risk.n_events

        def per_time(values: Tensor) -> Tensor:
            return torch.zeros(total, dtype=FLOAT).index_add_(0, self.term_time, values)

        a, b = per_time(inverse), per_time(frac * inverse)
        alpha2, beta2 = per_time(inverse.square()), per_time(frac * inverse.square())
        gamma2 = per_time(frac.square() * inverse.square())
        d1 = torch.cat([risk.at_event(v[:, None] * x * self.fail_float[:, None]),
                        self._group_at_event(v)], dim=1)
        b_rows = torch.where(self.failing, b[self.event_of.clamp_min(0)], torch.zeros_like(v))
        weight = v * risk.accumulate(a) - v * b_rows
        gradient = self.g_const - torch.cat(
            [x.T @ weight, torch.zeros(q, dtype=FLOAT).index_add_(0, g, weight)])
        cross = self._outer(beta2, s1, d1)
        hessian = (-self._zwz(weight) + self._outer(alpha2, s1, s1) - cross - cross.T
                   + self._outer(gamma2, d1, d1))
        return value, gradient, (hessian + hessian.T) / 2


class PenalizedFrailty:
    """``PPL = PL(b, w) - P(w; theta)`` for a fixed frailty variance theta > 0."""

    def __init__(self, cox: FrailtyCox, distribution: str, theta: float):
        if distribution not in {"gamma", "lognormal"}:
            raise KernelError("invalid_option", f"Unknown frailty distribution {distribution!r}.")
        if not theta > 0:
            raise KernelError("invalid_option", "The frailty variance must be positive.")
        self.cox, self.distribution, self.theta = cox, distribution, float(theta)

    def penalty(self, omega: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        theta = self.theta
        if self.distribution == "gamma":
            nu = 1.0 / theta
            e = torch.exp(omega)
            return nu * (e - omega).sum(), nu * (e - 1.0), nu * e
        return omega.square().sum() / (2 * theta), omega / theta, torch.full_like(omega, 1 / theta)

    def value(self, params: Tensor) -> Tensor:
        return self.cox.value(params) - self.penalty(params[self.cox.k:])[0]

    def __call__(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k = self.cox.k
        value, gradient, hessian = self.cox(params)
        p0, p1, p2 = self.penalty(params[k:])
        gradient = gradient.clone()
        gradient[k:] -= p1
        hessian = hessian.clone()
        hessian[k:, k:] -= torch.diag(p2)
        return value - p0, gradient, hessian


def gamma_correction(group_events: Tensor, nu: float) -> tuple[float, float]:
    """``sum_g c_g(nu)`` and its derivative in nu (see the module notes), cancellation free."""
    d = group_events
    counts = d.round().to(torch.int64)
    positive = counts > 0
    part = (d - nu * torch.log1p(d / nu)).sum()
    derivative = (-torch.log1p(d / nu)).sum()
    if bool(positive.any()):
        owner = torch.repeat_interleave(torch.arange(d.numel()), counts)
        starts = torch.cumsum(counts, 0) - counts
        j = (torch.arange(owner.numel()) - starts[owner]).to(FLOAT)
        total = d[owner]
        part = part + torch.log1p((j - total) / (nu + total)).sum()
        derivative = derivative + (1.0 / (nu + j)).sum()
    return float(part), float(derivative)


def gamma_integrated(partial: float, omega: Tensor, group_events: Tensor,
                     theta: float) -> tuple[float, float]:
    """Gamma-frailty integrated log partial likelihood and its derivative in ln theta."""
    nu = 1.0 / theta
    correction, slope = gamma_correction(group_events, nu)
    value = partial + nu * float(omega.sum()) + correction
    # dL/dnu = sum_g w_g + sum_g c_g'(nu) at the PPL maximum (envelope theorem with
    # sum_g exp(w_g) = q); dL/dln(theta) = -nu dL/dnu.
    return value, -nu * (float(omega.sum()) + slope)


def lognormal_integrated(partial: float, omega: Tensor, information: Tensor,
                         theta: float) -> float:
    """Laplace integrated log partial likelihood of the log-normal frailty.

    ``information`` is ``J_ww``, minus the ww block of the PL Hessian at the PPL
    maximum; ``ln det(I + theta J_ww)`` is computed from a Cholesky factor.
    """
    q = omega.numel()
    matrix = torch.eye(q, dtype=FLOAT) + theta * information
    chol, info = torch.linalg.cholesky_ex((matrix + matrix.T) / 2)
    if int(info) != 0:
        raise KernelError("numerical_failure", "The Laplace determinant is not positive definite.")
    logdet = 2.0 * float(torch.log(chol.diagonal()).sum())
    return partial - float(omega.square().sum()) / (2 * theta) - 0.5 * logdet


# ---- parametric frailty: log Laplace-transform derivatives -------------------------

def gamma_log_moment(m: Tensor, s: Tensor, log_theta: float) -> tuple[Tensor, ...]:
    """``ln M_m(s)`` of the gamma frailty and derivatives in (s, ln theta).

    Returns ``(l, l_s, l_ss, l_t, l_tt, l_st)`` with ``t = ln theta``.
    """
    nu = math.exp(-log_theta)
    nu_t = torch.full_like(s, nu)
    a = nu_t + m
    b = nu_t + s
    lgam = torch.lgamma(a) - math.lgamma(nu)
    value = nu * math.log(nu) + lgam - a * torch.log(b)
    l_s = -a / b
    l_ss = a / b.square()
    psi = torch.digamma(a) - torch.digamma(nu_t)
    psi1 = torch.polygamma(1, a) - torch.polygamma(1, nu_t)
    l_nu = -torch.log1p(s / nu) + (s - m) / b + psi
    l_nunu = 1.0 / nu + psi1 - 1.0 / b - (s - m) / b.square()
    l_snu = -(s - m) / b.square()
    l_t = -nu * l_nu
    l_tt = nu * nu * l_nunu + nu * l_nu
    l_st = -nu * l_snu
    return value, l_s, l_ss, l_t, l_tt, l_st


def _bessel_sum(n: Tensor, w: Tensor, order_limit: int) -> tuple[Tensor, Tensor, Tensor]:
    """``ln S_n(w)``, ``d ln S/dw`` and ``d2 ln S/dw2`` for per-row orders n (padded)."""
    k = torch.arange(order_limit + 1, dtype=FLOAT)
    nf = n.to(FLOAT)[:, None]
    valid = k[None, :] <= nf
    log_c = (torch.lgamma(nf + k + 1) - torch.lgamma(k + 1)
             - torch.lgamma((nf - k).clamp_min(0) + 1))
    log_w = torch.log(w)[:, None]
    terms = torch.where(valid, log_c + k * log_w, torch.full_like(log_c, -math.inf))
    log_s = torch.logsumexp(terms, dim=1)
    share = torch.exp(terms - log_s[:, None])            # c_k w^k / S
    first = (share * k).sum(1) / w                        # S'/S
    second = (share * k * (k - 1)).sum(1) / w.square()    # S''/S
    return log_s, first, second - first.square()


def inverse_gaussian_log_moment(m: Tensor, s: Tensor, log_theta: float,
                                order_limit: int) -> tuple[Tensor, ...]:
    """``ln M_m(s)`` of the inverse-Gaussian frailty and derivatives in (s, ln theta)."""
    theta = math.exp(log_theta)
    u = torch.sqrt(1.0 + 2.0 * theta * s)
    order = torch.where(m >= 1, m - 1, -m).round().to(torch.int64)
    w = theta / (2.0 * u)
    log_s, s1, s2 = _bessel_sum(order, w, order_limit)
    one_u = 1.0 + u
    value = -2.0 * s / one_u - m * torch.log(u) + log_s
    # derivatives of u and w
    u_s, u_th = theta / u, s / u
    w_s = -theta * theta / (2 * u ** 3)
    w_th = (1.0 + theta * s) / (2 * u ** 3)
    w_ss = 3 * theta ** 3 / (2 * u ** 5)
    w_thth = s / (2 * u ** 3) - 3 * s * (1 + theta * s) / (2 * u ** 5)
    w_sth = -theta / u ** 3 + 3 * theta ** 2 * s / (2 * u ** 5)
    del u_s, u_th
    # f1 = (1 - u)/theta, f2 = -m ln u, f3 = ln S(w)
    f_s = -1.0 / u - m * theta / u.square() + s1 * w_s
    f_ss = theta / u ** 3 + 2 * m * theta ** 2 / u ** 4 + s2 * w_s.square() + s1 * w_ss
    f_th = 2 * s.square() / (u * one_u.square()) - m * s / u.square() + s1 * w_th
    f_thth = (-2 * s ** 3 * (1 + 3 * u) / (u ** 3 * one_u ** 3) + 2 * m * s.square() / u ** 4
              + s2 * w_th.square() + s1 * w_thth)
    f_sth = (s / u ** 3 - m / u.square() + 2 * m * theta * s / u ** 4
             + s2 * w_s * w_th + s1 * w_sth)
    l_t = theta * f_th
    l_tt = theta * theta * f_thth + theta * f_th
    l_st = theta * f_sth
    return value, f_s, f_ss, l_t, l_tt, l_st


# ---- scalar searches over ln(theta) -----------------------------------------------

_GOLDEN = 0.3819660112501051


def brent_maximize(function, low: float, high: float, *, tol: float = 1e-9,
                   max_iter: int = 200) -> float:
    """Brent's derivative-free maximizer of a scalar function on [low, high].

    Golden-section steps with parabolic interpolation (Brent 1973, ch. 5); a loop
    over iterations only. Returns the abscissa of the best value found.
    """
    a, b = low, high
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = -function(x)
    d = e = 0.0
    for _ in range(max_iter):
        middle = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-12
        tol2 = 2.0 * tol1
        if abs(x - middle) <= tol2 - 0.5 * (b - a):
            break
        parabolic = False
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            if abs(p) < abs(0.5 * q * e) and q * (a - x) < p < q * (b - x):
                e, d = d, p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if middle >= x else -tol1
                parabolic = True
        if not parabolic:
            e = (a if x >= middle else b) - x
            d = _GOLDEN * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = -function(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def zeroin(function, low: float, high: float, f_low: float, f_high: float, *,
           tol: float = 1e-12, max_iter: int = 200) -> float:
    """Brent-Dekker root of a scalar function bracketed by [low, high] (signs differ)."""
    a, b, fa, fb = low, high, f_low, f_high
    if fa * fb > 0:
        raise KernelError("invalid_bracket", "The root is not bracketed.")
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * 2.2e-16 * abs(b) + 0.5 * tol
        middle = 0.5 * (c - b)
        if abs(middle) <= tol1 or fb == 0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2.0 * middle * s, 1.0 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2.0 * middle * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * middle * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = middle
        else:
            d = e = middle
        a, fa = b, fb
        b = b + (d if abs(d) > tol1 else (tol1 if middle > 0 else -tol1))
        fb = function(b)
    return b
