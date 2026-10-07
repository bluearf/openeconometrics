"""Likelihood kernels of the count family on float64 tensors (no autograd, O(n) memory).

Index models
------------
Every likelihood of the family is ``ll(theta) = sum_i w_i l_i(e_1i, ..., e_pi)``
where each index ``e_a = X_a theta_a + offset_a`` is linear in its own block of
parameters, or is a single ancillary parameter such as ``ln(alpha)`` (a *scalar
index*, design ``None``). A model supplies only the per-observation value
``l_i`` with its first and second derivatives in the indices,

    g_a = dl/de_a,    h_ab = d2l/de_a de_b    (a <= b, upper-triangular order),

and ``IndexObjective`` assembles the gradient blocks ``X_a'(w g_a)``, the
Hessian blocks ``X_a' diag(w h_ab) X_b`` and the score rows ``[x_ai g_a]``.

Count densities
---------------
Poisson, index ``eta`` (``mu = exp(eta)``):

    l = y eta - mu - lnG(y+1),   g = y - mu,   h = -mu.

Negative binomial, indices ``(eta, tau)`` with ``tau = ln(alpha)`` (NB2,
``Var = mu + alpha mu^2``, ``m = 1/alpha``) or ``tau = ln(delta)`` (NB1,
``Var = mu (1 + delta)``, ``m = mu/delta``); the formulas are those of the glm
family's ``nbreg`` kernel, written per observation so that ``tau`` may vary
across observations (``gnbreg``):

    NB2  l = lnG(y+m) - lnG(m) - lnG(y+1) - (y+m) ln(1+alpha mu) + y ln(alpha mu)
         c = psi(m) - psi(y+m) + ln(1+alpha mu),   t = psi'(m) - psi'(y+m)
         g_eta = (y-mu)/(1+alpha mu),              g_tau = m c + g_eta
         h_ee  = -mu (1+alpha y)/(1+alpha mu)^2,   h_et = -alpha mu (y-mu)/(1+alpha mu)^2
         h_tt  = -m c - m^2 t + mu/(1+alpha mu) + h_et

Left truncation (``TruncatedPieces``)
-------------------------------------
``l = ln f(y) - ln S`` with ``S = Pr(Y > ll) = 1 - sum_{j<=ll} f(j)``.

* Poisson: ``S = P(ll+1, mu)``, the regularized lower incomplete gamma function
  (``ln S`` is taken as ``log1p(-Q)`` when the complement ``Q`` is the small
  tail). With ``r = mu f(ll) / S``: ``dlnS/deta = r`` and
  ``d2lnS/deta2 = r (1 + ll - mu) - r^2``. For ``ll = 0``: ``S = -expm1(-mu)``,
  ``r = mu / expm1(mu)``.
* Negative binomial: the lower sum is accumulated term by term together with
  ``F_a = sum f_j s_ja`` and ``F_ab = sum f_j (s_ja s_jb + H_jab)`` (``s_j``,
  ``H_j`` the score and Hessian of ``ln f(j)``), so that
  ``dlnS = -F_a / S`` and ``d2lnS = -F_ab / S - F_a F_b / S^2``. The ``j = 0``
  term enters ``S`` through ``-expm1(ln f(0))``, exact for zero truncation. A
  trial point where some ``S_i <= 1e-12`` is rejected: the complement of the
  lower sum has no accuracy there.

Zero inflation (``ZeroInflatedPieces``)
---------------------------------------
With ``a = ln F(g)``, ``b = ln(1 - F(g))`` for the inflation index ``g`` and
``v = ln f(y)``: a zero has ``l = logaddexp(a, b + v)`` and a positive count
``l = b + v``. With the posterior probability of a structural zero
``pi = exp(a - l)`` (0 for positives), ``q = 1 - pi`` and the base score ``s``
and Hessian ``H``:

    dl/dg = pi a' + q b',            d2l/dg2 = pi a'' + q b'' + pi q (a' - b')^2
    dl/dtheta = q s,                 d2l/dtheta2 = q H + pi q s s'
    d2l/dg dtheta = -pi q (a' - b') s

Truncated normal (``TruncatedNormalPieces``, Cragg's hurdle)
-----------------------------------------------------------
Outcome ``t`` (``y`` or ``ln y``) above the limit ``c``, indices ``(xb, s)``
with ``s = ln(sigma)``, ``z = (t - xb)/sigma``, ``A = (xb - c)/sigma``,
``lam = phi(A)/Phi(A)`` and ``v = lam (lam + A)``:

    l = -s - z^2/2 - ln sqrt(2 pi) - ln Phi(A)
    g_b = (z - lam)/sigma,                 g_s = z^2 - 1 + lam A
    h_bb = (v - 1)/sigma^2,   h_bs = (lam - 2 z - v A)/sigma,   h_ss = -2 z^2 + v A^2 - lam A

All derivatives are verified against numerical ones in the tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.engines.contracts import KernelError
from openecon.engines.count_numeric import poisson_logmass, validate_nb_precision
from openecon.engines.linalg import weighted_crossprod

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
# Smallest Pr(Y > ll) for which the complement of a lower sum is still accurate.
_TAIL_FLOOR = 1e-12

Terms = tuple[Tensor, list[Tensor] | None, list[Tensor | None] | None]


def _pairs(size: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(size) for b in range(a, size)]


def _log1mexp(value: Tensor) -> Tensor:
    """``ln(1 - exp(value))`` for ``value < 0``, accurate at both ends."""
    return torch.where(value > -math.log(2.0), torch.log(-torch.expm1(value)),
                       torch.log1p(-torch.exp(value)))


def link_pieces(link: str, g: Tensor, derivatives: bool = True):
    """``a = ln F(g)``, ``b = ln(1 - F(g))`` and their first and second derivatives.

    ``logit``: ``F`` logistic; ``probit``: standard normal (tail-stable Mills
    ratios); ``cloglog``: ``F = 1 - exp(-exp(g))``.
    """
    if link == "logit":
        a, b = logsigmoid(g), logsigmoid(-g)
        if not derivatives:
            return a, b, None, None, None, None
        p, c = torch.sigmoid(g), torch.sigmoid(-g)
        curve = -p * c
        return a, b, c, -p, curve, curve
    if link == "probit":
        a, ratio_a, curve_a = mills_ratio(g)
        b, ratio_b, curve_b = mills_ratio(-g)
        return a, b, ratio_a, -ratio_b, -curve_a, -curve_b
    if link == "cloglog":
        u = torch.exp(g)
        a, b = _log1mexp(-u), -u
        if not derivatives:
            return a, b, None, None, None, None
        slope = u / torch.expm1(u)
        return a, b, slope, -u, slope * (1 + u / torch.expm1(-u)), -u
    raise KernelError("invalid_link", "The link must be 'logit', 'probit' or 'cloglog'.")


# ---- count densities -----------------------------------------------------------------


class PoissonDensity:
    """``ln f(y)`` of Poisson(exp(eta)) and its derivatives in ``eta``."""

    size = 1
    name = "poisson"

    def terms(self, y: Tensor, log_factorial: Tensor | float, index: Sequence[Tensor],
              derivatives: bool = True) -> Terms:
        eta = index[0]
        mu = torch.exp(eta)
        value = poisson_logmass(y, mu, eta=eta, log_factorial=log_factorial)
        if not derivatives:
            return value, None, None
        return value, [y - mu], [-mu]


class NegBinDensity:
    """``ln f(y)`` of the negative binomial in ``(eta, tau)``; ``tau`` may be a vector."""

    size = 2
    name = "nbinomial"

    def __init__(self, form: str = "mean"):
        if form not in ("mean", "constant"):
            raise KernelError("invalid_option", "dispersion must be 'mean' or 'constant'.")
        self.form = form

    def check_precision(self, y, index=None):
        if index is None:
            validate_nb_precision(y)
            return
        eta, tau = index
        shape = torch.exp(-tau) if self.form == "mean" else torch.exp(eta - tau)
        validate_nb_precision(y, shape)

    def terms(self, y: Tensor, log_factorial: Tensor | float, index: Sequence[Tensor],
              derivatives: bool = True) -> Terms:
        eta, tau = index
        self.check_precision(y, index)
        mu = torch.exp(eta)
        if self.form == "mean":
            a, m = torch.exp(tau), torch.exp(-tau)
            am = a * mu
            ratio, log_ratio = 1 + am, torch.log1p(am)
            value = (torch.lgamma(y + m) - torch.lgamma(m) - log_factorial
                     - (y + m) * log_ratio + torch.special.xlogy(y, am))
            if not derivatives:
                return value, None, None
            c = torch.special.digamma(m) - torch.special.digamma(y + m) + log_ratio
            t = torch.special.polygamma(1, m) - torch.special.polygamma(1, y + m)
            b = (y - mu) / ratio
            h_et = -am * (y - mu) / ratio.square()
            return value, [b, m * c + b], [-mu * (1 + a * y) / ratio.square(), h_et,
                                           -m * c - m * m * t + mu / ratio + h_et]
        d = torch.exp(tau)
        m, log1pd = mu / d, torch.log1p(d)
        value = (torch.lgamma(y + m) - torch.lgamma(m) - log_factorial - m * log1pd
                 + y * (tau - log1pd))
        if not derivatives:
            return value, None, None
        big = torch.special.digamma(y + m) - torch.special.digamma(m) - log1pd
        t = torch.special.polygamma(1, y + m) - torch.special.polygamma(1, m)
        curve = m * big + m.square() * t
        share = mu / (1 + d)
        return value, [m * big, -m * big + (y - mu) / (1 + d)], [
            curve, -curve - share, curve + share - (y - mu) * d / (1 + d).square()]


# ---- per-observation models ----------------------------------------------------------


class CountPieces:
    """A plain count density evaluated at the observed outcome."""

    def __init__(self, density, y: Tensor):
        self.density, self.y = density, y
        self.log_factorial = torch.lgamma(y + 1)

    def check_precision(self, index=None):
        if isinstance(self.density, NegBinDensity):
            self.density.check_precision(self.y, index)

    def __call__(self, index: Sequence[Tensor], derivatives: bool = True) -> Terms | None:
        return self.density.terms(self.y, self.log_factorial, index, derivatives)


class TruncatedPieces:
    """Count density conditional on ``y > ll`` (``limit``: an integer or an [n] tensor)."""

    def __init__(self, density, y: Tensor, limit: Tensor | int):
        self.density, self.y = density, y
        self.log_factorial = torch.lgamma(y + 1)
        if isinstance(limit, Tensor):
            limit = limit.to(torch.float64)
            if bool((limit == limit[0]).all()):
                limit = int(limit[0])
        if isinstance(limit, Tensor):
            self.limit, self.top, self.varying = limit, int(limit.max()), True
        else:
            self.limit, self.top, self.varying = float(limit), int(limit), False
        if self.top < 0 or (self.varying and bool((self.limit < 0).any())):
            raise KernelError("invalid_truncation", "The truncation point must be nonnegative.")

    def check_precision(self, index=None):
        if isinstance(self.density, NegBinDensity):
            self.density.check_precision(self.y, index)
        elif self.top > 1e12:
            raise KernelError("precision_unsupported", "Poisson cumulative truncation cutoffs above 1e12 exceed the validated gamma-function region.")

    def _poisson_tail(self, eta: Tensor, derivatives: bool):
        mu = torch.exp(eta)
        if not self.varying and self.top == 0:
            log_s = _log1mexp(-mu)
            if not derivatives:
                return log_s, None, None
            ratio = mu / torch.expm1(mu)
            return log_s, [ratio], [ratio * (1 - mu) - ratio.square()]
        shape = self.limit + 1 if self.varying else torch.full_like(mu, self.limit + 1)
        upper = torch.special.gammaincc(shape, mu)          # Pr(Y <= ll)
        lower = torch.special.gammainc(shape, mu)           # Pr(Y > ll)
        log_s = torch.where(upper < 0.5, torch.log1p(-upper), torch.log(lower))
        if not derivatives:
            return log_s, None, None
        ratio = torch.exp(eta + poisson_logmass(shape - 1, mu, eta=eta) - log_s)
        return log_s, [ratio], [ratio * (shape - mu) - ratio.square()]

    def _summed_tail(self, index: Sequence[Tensor], derivatives: bool):
        size = self.density.size
        pairs = _pairs(size)
        rest = 0.0
        first: list = [0.0] * size
        second: list = [0.0] * len(pairs)
        head = None
        for j in range(self.top + 1):
            count = torch.tensor(float(j), dtype=torch.float64)
            value, g, h = self.density.terms(count, math.lgamma(j + 1), index, derivatives)
            f = torch.exp(value)
            if self.varying and j:
                f = f * (self.limit >= j)
            if j == 0:
                head = value
            else:
                rest = rest + f
            if derivatives:
                for a in range(size):
                    first[a] = first[a] + f * g[a]
                for position, (a, b) in enumerate(pairs):
                    second[position] = second[position] + f * (g[a] * g[b] + h[position])
        survival = -torch.expm1(head) - rest
        if not bool((survival > _TAIL_FLOOR).all()):
            return None
        log_s = _log1mexp(head) if self.top == 0 else torch.log(survival)
        if not derivatives:
            return log_s, None, None
        slopes = [-value / survival for value in first]
        curves = [-second[position] / survival - slopes[a] * slopes[b]
                  for position, (a, b) in enumerate(pairs)]
        return log_s, slopes, curves

    def tail(self, index: Sequence[Tensor], derivatives: bool = True):
        """``ln Pr(Y > ll)`` and its derivatives in the indices (None: not computable)."""
        if isinstance(self.density, PoissonDensity):
            return self._poisson_tail(index[0], derivatives)
        return self._summed_tail(index, derivatives)

    def conditional_mean(self, index: Sequence[Tensor]) -> Tensor:
        """``E[y | y > ll] = (mu - sum_{j<=ll} j f(j)) / Pr(Y > ll)``."""
        tail = self.tail(index, True)
        if tail is None:
            raise KernelError("numerical_failure", "The truncated mean is not computable: "
                              "some observations have Pr(y > ll) below 1e-12.")
        mu = torch.exp(index[0])
        if isinstance(self.density, PoissonDensity):
            return mu + tail[1][0]
        lower = 0.0
        for j in range(1, self.top + 1):
            count = torch.tensor(float(j), dtype=torch.float64)
            f = torch.exp(self.density.terms(count, math.lgamma(j + 1), index, False)[0])
            lower = lower + j * (f * (self.limit >= j) if self.varying else f)
        return (mu - lower) / torch.exp(tail[0])

    def __call__(self, index: Sequence[Tensor], derivatives: bool = True) -> Terms | None:
        tail = self.tail(index, derivatives)
        if tail is None:
            return None
        value, g, h = self.density.terms(self.y, self.log_factorial, index, derivatives)
        log_s, slopes, curves = tail
        if not derivatives:
            return value - log_s, None, None
        return (value - log_s, [u - v for u, v in zip(g, slopes, strict=True)],
                [u - v for u, v in zip(h, curves, strict=True)])


class ZeroInflatedPieces:
    """Zero-inflated count density; indices ``(eta, inflate[, tau])``."""

    def __init__(self, density, y: Tensor, link: str):
        self.density, self.y, self.link = density, y, link
        self.zero = y == 0
        self.log_factorial = torch.lgamma(y + 1)
        self.size = density.size + 1

    def check_precision(self, index=None):
        if isinstance(self.density, NegBinDensity):
            self.density.check_precision(self.y, None if index is None else [index[0], *index[2:]])

    def __call__(self, index: Sequence[Tensor], derivatives: bool = True) -> Terms | None:
        base = [index[0], *index[2:]]
        v, s, hess = self.density.terms(self.y, self.log_factorial, base, derivatives)
        a, b, a1, b1, a2, b2 = link_pieces(self.link, index[1], derivatives)
        count = b + v
        mixture = torch.logaddexp(a, count)
        value = torch.where(self.zero, mixture, count)
        if not derivatives:
            return value, None, None
        pi = torch.where(self.zero, torch.exp(a - mixture), 0.0)
        q = torch.where(self.zero, torch.exp(count - mixture), 1.0)
        both, gap = pi * q, a1 - b1
        g_inflate = pi * a1 + q * b1
        h_inflate = pi * a2 + q * b2 + both * gap.square()
        cross = [-both * gap * value_ for value_ in s]
        h_ee = q * hess[0] + both * s[0].square()
        if self.density.size == 1:
            return value, [q * s[0], g_inflate], [h_ee, cross[0], h_inflate]
        return value, [q * s[0], g_inflate, q * s[1]], [
            h_ee, cross[0], q * hess[1] + both * s[0] * s[1],
            h_inflate, cross[1], q * hess[2] + both * s[1].square()]


class BinaryPieces:
    """Binary outcome ``d`` (bool) with ``Pr(d = 1) = F(g)``."""

    def __init__(self, positive: Tensor, link: str):
        self.positive, self.link = positive, link

    def __call__(self, index: Sequence[Tensor], derivatives: bool = True) -> Terms | None:
        a, b, a1, b1, a2, b2 = link_pieces(self.link, index[0], derivatives)
        value = torch.where(self.positive, a, b)
        if not derivatives:
            return value, None, None
        return value, [torch.where(self.positive, a1, b1)], [torch.where(self.positive, a2, b2)]


class TruncatedNormalPieces:
    """Normal density of ``t`` given ``t > limit`` in ``(xb, ln sigma)``; ``limit`` may be None.

    ``constant`` is added to every log density (the Jacobian ``-ln y`` of the
    exponential hurdle model).
    """

    def __init__(self, t: Tensor, limit: float | None, constant: Tensor | None = None):
        self.t, self.limit, self.constant = t, limit, constant

    def __call__(self, index: Sequence[Tensor], derivatives: bool = True) -> Terms | None:
        xb, log_sigma = index
        sigma = torch.exp(log_sigma)
        z = (self.t - xb) / sigma
        value = -log_sigma - 0.5 * z.square() - _LOG_SQRT_2PI
        if self.constant is not None:
            value = value + self.constant
        if self.limit is None:
            if not derivatives:
                return value, None, None
            return value, [z / sigma, z.square() - 1], [
                torch.full_like(z, -1.0) / sigma.square(), -2 * z / sigma, -2 * z.square()]
        alpha = (xb - self.limit) / sigma
        log_cdf, ratio, curve = mills_ratio(alpha)
        value = value - log_cdf
        if not derivatives:
            return value, None, None
        return value, [(z - ratio) / sigma, z.square() - 1 + ratio * alpha], [
            (curve - 1) / sigma.square(), (ratio - 2 * z - curve * alpha) / sigma,
            -2 * z.square() + curve * alpha.square() - ratio * alpha]


# ---- objective assembly ----------------------------------------------------------------


class IndexObjective:
    """``sum_i w_i l_i(e_1i, ..., e_pi)`` with analytic gradient and Hessian.

    ``designs[a]`` is the float64 [n, k_a] design of index ``a`` or ``None``
    for a scalar index (one parameter shared by every observation);
    ``offsets[a]`` is added to the index. ``pieces(index, derivatives)``
    returns ``(value, g, h)`` per observation (``h`` in upper-triangular
    order, entries ``None`` for identically zero blocks) or ``None`` for a
    point outside the domain.
    """

    def __init__(self, pieces: Callable[..., Terms | None],
                 designs: Sequence[Tensor | None], weights: Tensor,
                 offsets: Sequence[Tensor | None] | None = None):
        n = weights.shape[0]
        for design in designs:
            if design is not None and (not isinstance(design, Tensor) or design.ndim != 2
                                       or design.dtype != torch.float64
                                       or design.shape[0] != n):
                raise KernelError("invalid_design", "Designs must be float64 [n, k] tensors.")
        if weights.dtype != torch.float64 or weights.ndim != 1:
            raise KernelError("invalid_weights", "Weights must be a float64 vector [n].")
        self.pieces, self.designs, self.w = pieces, list(designs), weights
        self.reject_precision_trials = False
        self.precision_rejected = False
        self._precision_check = getattr(pieces, "check_precision", None)
        if self._precision_check is not None:
            self._precision_check()
        self.offsets = list(offsets) if offsets is not None else [None] * len(self.designs)
        sizes = [1 if design is None else design.shape[1] for design in self.designs]
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        self.slices = [slice(bounds[a], bounds[a + 1]) for a in range(len(sizes))]
        self.size = bounds[-1]
        self.pairs = _pairs(len(sizes))

    def check_precision(self, theta):
        if self._precision_check is not None:
            self._precision_check(self.indices(theta))

    def _pieces(self, theta, derivatives):
        try:
            self.check_precision(theta)
            return self.pieces(self.indices(theta), derivatives)
        except KernelError as exc:
            if exc.code != "precision_unsupported" or not self.reject_precision_trials:
                raise
            self.precision_rejected = True
            return None

    def indices(self, theta: Tensor) -> list[Tensor]:
        out = []
        for design, part, offset in zip(self.designs, self.slices, self.offsets, strict=True):
            if design is None:
                out.append(theta[part][0])
            else:
                index = design @ theta[part]
                out.append(index if offset is None else index + offset)
        return out

    def _rejected(self) -> tuple[Tensor, Tensor, Tensor]:
        return (torch.tensor(-math.inf, dtype=torch.float64),
                torch.zeros(self.size, dtype=torch.float64),
                -torch.eye(self.size, dtype=torch.float64))

    def observations(self, theta: Tensor) -> Tensor:
        """Per-observation log likelihood (before weights)."""
        terms = self._pieces(theta, False)
        if terms is None or not bool(torch.isfinite(terms[0]).all()):
            raise KernelError("numerical_failure", "The log likelihood is not finite at the "
                              "requested parameter values.")
        return terms[0]

    def value(self, theta: Tensor) -> Tensor:
        terms = self._pieces(theta, False)
        if terms is None:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return (self.w * terms[0]).sum()

    def _block(self, a: int, b: int, weight: Tensor) -> Tensor:
        left, right = self.designs[a], self.designs[b]
        if left is None and right is None:
            return weight.sum().reshape(1, 1)
        if left is None:
            return (right.T @ weight)[None, :]
        if right is None:
            return (left.T @ weight)[:, None]
        return weighted_crossprod(left, weight, None if a == b else right)

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        terms = self._pieces(theta, True)
        if terms is None:
            return self._rejected()
        value, g, h = terms
        w = self.w
        gradient = torch.empty(self.size, dtype=torch.float64)
        hessian = torch.zeros((self.size, self.size), dtype=torch.float64)
        for a, design in enumerate(self.designs):
            weighted = w * g[a]
            gradient[self.slices[a]] = weighted.sum() if design is None else design.T @ weighted
        for position, (a, b) in enumerate(self.pairs):
            if h[position] is None:
                continue
            block = self._block(a, b, w * h[position])
            hessian[self.slices[a], self.slices[b]] = block
            if a != b:
                hessian[self.slices[b], self.slices[a]] = block.T
        return (w * value).sum(), gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation score rows [n, K] before weights."""
        terms = self._pieces(theta, True)
        if terms is None:
            raise KernelError("numerical_failure", "The scores are not finite at the estimates.")
        columns = [g[:, None] if design is None else design * g[:, None]
                   for design, g in zip(self.designs, terms[1], strict=True)]
        return torch.cat(columns, dim=1)
