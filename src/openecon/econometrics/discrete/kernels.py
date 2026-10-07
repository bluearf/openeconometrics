"""Likelihood kernels of the discrete-choice family on float64 tensors.

No autograd: every objective returns its analytic gradient and Hessian, which
the tests compare with numerical derivatives. All sums are O(n) in memory;
category and group work uses ``index_add_``. ``w_i`` denotes the weight of
observation i in every likelihood sum (ones when unweighted).

Ordered logit / probit (``OrderedObjective``)
---------------------------------------------
Categories are coded ``c_i = 0..J-1`` and separated by the cutpoints
``k_0 < ... < k_{J-2}`` (``k_{-1} = -inf``, ``k_{J-1} = +inf``); with the index
``e_i = x_i'b + offset_i`` and ``F`` the logistic or standard normal
distribution function,

    p_i = F(u_i) - F(l_i),   u_i = k_{c_i} - e_i,   l_i = k_{c_i - 1} - e_i.

The difference is formed on the side where both tails are small: when
``u + l > 0`` the pair is reflected to ``(-u, -l)`` (``F(u) - F(l) =
F(-l) - F(-u)``), and ``ln p = ln F(b) + ln(1 - exp(ln F(a) - ln F(b)))`` for
``a < b`` uses ``log_ndtr`` / ``logsigmoid``. With the density ``f``, its log
derivative ``r = f'/f`` (``-x`` for the normal, ``1 - 2F`` for the logistic)
and the ratios ``A = f(u)/p``, ``B = f(l)/p`` (zero at an infinite limit):

    dl/du = A,  dl/dl = -B,  d2l/du2 = r(u) A - A^2,  d2l/dl2 = -r(l) B - B^2,
    d2l/du dl = A B,

and the chain rule (``du = dk_{c} - de``, ``dl = dk_{c-1} - de``) gives the
score and the Hessian in ``(b, k)``. A trial point whose cutpoints are not
strictly increasing has no likelihood: the value is ``-inf`` and the line
search of the optimizer rejects it.

Multinomial logit (``MultinomialObjective``)
--------------------------------------------
``eta_ij = x_i'b_j`` for the non-base categories and 0 for the base;
``ln p_ij = eta_ij - logsumexp_j(eta_ij)``. With ``d_ij = 1[c_i = j]``:

    gradient_j = X'(w (d_j - p_j)),
    Hessian_jl = -X' diag(w p_j (1[j = l] - p_l)) X.

The Hessian is accumulated over row blocks as
``-(blockdiag_j X' diag(w p_j) X - M'M)`` with ``M = sqrt(w) (p_i kron x_i)``,
one symmetric rank update per block, so no n-by-k(J-1) matrix outlives a block.

Heteroskedastic probit (``HetprobitObjective``)
-----------------------------------------------
``Pr(y = 1) = Phi(x'b / exp(z'g))``. With ``s = exp(z'g)``, ``t = x'b / s``,
``q = 2y - 1``, the inverse Mills ratio ``m = q phi(q t) / Phi(q t)`` and
``v = m (m + t)``:

    dl/db = m x / s,                  dl/dg = -m t z,
    d2l/db db' = -v x x' / s^2,       d2l/db dg' = (v t - m) x z' / s,
    d2l/dg dg' = (m t - v t^2) z z'.

``m`` uses the scaled complementary error function in the lower tail, where
``phi / Phi`` would be a ratio of two underflowing numbers. Without variance
regressors the same objective is the ordinary probit likelihood, which
supplies starting values and the comparison model of the LR tests.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
# Elements of the widest temporary of a blocked accumulation (32 MiB of float64).
_BLOCK_ELEMENTS = 4_000_000


def _rejected(size: int) -> tuple[Tensor, Tensor, Tensor]:
    """A non-finite objective for a trial point outside the parameter space."""
    return (torch.tensor(-math.inf, dtype=torch.float64), torch.zeros(size, dtype=torch.float64),
            -torch.eye(size, dtype=torch.float64))


def mills_ratio(t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``ln Phi(t)``, ``phi(t)/Phi(t)`` and ``v = ratio (ratio + t)`` without cancellation.

    For ``t < 0`` the ratio is ``sqrt(2/pi) / erfcx(-t / sqrt 2)``; beyond
    ``t < -1e4`` the curvature ``v`` switches to its asymptotic series
    ``1 - 1/t^2 + 6/t^4 - 50/t^6`` because ``ratio + t`` cancels there.
    """
    log_cdf = torch.special.log_ndtr(t)
    negative = torch.clamp(t, max=0.0)
    lower = math.sqrt(2 / math.pi) / torch.special.erfcx(-negative / math.sqrt(2))
    upper = torch.exp(-0.5 * t.square() - _LOG_SQRT_2PI - log_cdf)
    ratio = torch.where(t < 0, lower, upper)
    curvature = ratio * (ratio + t)
    inverse = torch.clamp(t, max=-1.0).reciprocal().square()
    series = 1 - inverse + 6 * inverse.square() - 50 * inverse.pow(3)
    return log_cdf, ratio, torch.where(t < -1e4, series, curvature)


class OrderedObjective:
    """Log likelihood of the ordered logit / probit model in ``(b, cutpoints)``."""

    def __init__(self, x: Tensor, codes: Tensor, n_categories: int, weights: Tensor,
                 offset: Tensor | None, link: str):
        if link not in {"logit", "probit"}:
            raise KernelError("invalid_link", "The ordered link must be 'logit' or 'probit'.")
        if n_categories < 2:
            raise KernelError("constant_outcome", "An ordered model needs at least two "
                              "outcome categories.")
        self.x, self.codes, self.w, self.offset, self.link = x, codes, weights, offset, link
        self.n, self.k = x.shape
        self.categories = n_categories
        self.size = self.k + n_categories - 1

    # ---- link pieces -----------------------------------------------------------

    def _log_cdf(self, value: Tensor) -> Tensor:
        return logsigmoid(value) if self.link == "logit" else torch.special.log_ndtr(value)

    def _log_pdf(self, value: Tensor) -> Tensor:
        if self.link == "logit":
            return logsigmoid(value) + logsigmoid(-value)
        return -0.5 * value.square() - _LOG_SQRT_2PI

    def _log_slope(self, value: Tensor) -> Tensor:
        """``f'(x) / f(x)``, with a finite value at infinite limits (where f/p is zero)."""
        if self.link == "logit":
            return -torch.tanh(value / 2)
        return -torch.nan_to_num(value, posinf=0.0, neginf=0.0)

    def _limits(self, theta: Tensor) -> tuple[Tensor, Tensor] | None:
        beta, cuts = theta[:self.k], theta[self.k:]
        if self.categories > 2 and bool((cuts[1:] <= cuts[:-1]).any()):
            return None
        eta = self.x @ beta
        if self.offset is not None:
            eta = eta + self.offset
        infinity = torch.full((1,), math.inf, dtype=torch.float64)
        extended = torch.cat([-infinity, cuts, infinity])
        return extended[self.codes + 1] - eta, extended[self.codes] - eta

    def _log_probability(self, upper: Tensor, lower: Tensor) -> Tensor:
        reflect = (upper + lower) > 0
        low = torch.where(reflect, -upper, lower)
        high = torch.where(reflect, -lower, upper)
        log_high = self._log_cdf(high)
        return log_high + torch.log(-torch.expm1(self._log_cdf(low) - log_high))

    # ---- objective -------------------------------------------------------------

    def value(self, theta: Tensor) -> Tensor:
        limits = self._limits(theta)
        if limits is None:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return (self.w * self._log_probability(*limits)).sum()

    def probabilities(self, theta: Tensor) -> Tensor:
        """Fitted probability of each observation's own category."""
        limits = self._limits(theta)
        if limits is None:
            raise KernelError("numerical_failure", "The cutpoints are not increasing.")
        return torch.exp(self._log_probability(*limits))

    def determined(self, theta: Tensor, threshold: float) -> int:
        """Observations whose side of some cutpoint is predicted with numerical certainty.

        Cutpoint j splits the outcome into ``y <= j`` and ``y > j``. When a regressor
        separates the two sides, the slope and the cutpoint diverge together and the
        fitted probability of the WRONG side, ``1 - F(k_j - e_i)`` for ``y_i <= j`` and
        ``F(k_j - e_i)`` otherwise, vanishes although no observation's own category is
        predicted perfectly. Returns the number of observations for which that
        probability is below ``threshold`` at some cutpoint, or zero when some
        observation is instead put on the wrong side with certainty (a diverging
        iteration, not separation). One pass per cutpoint; used only to explain a
        failed iteration.
        """
        beta, cuts = theta[:self.k], theta[self.k:]
        eta = self.x @ beta
        if self.offset is not None:
            eta = eta + self.offset
        certain = torch.zeros(self.n, dtype=torch.bool)
        for j in range(self.categories - 1):
            gap = cuts[j] - eta
            wrong = torch.exp(self._log_cdf(torch.where(self.codes <= j, -gap, gap)))
            if bool((wrong > 1 - threshold).any()):
                return 0
            certain |= wrong < threshold
        return int(certain.sum())

    def _ratios(self, theta: Tensor) -> tuple[Tensor, ...] | None:
        limits = self._limits(theta)
        if limits is None:
            return None
        upper, lower = limits
        log_p = self._log_probability(upper, lower)
        ratio_u = torch.exp(self._log_pdf(upper) - log_p)       # A = f(u) / p
        ratio_l = torch.exp(self._log_pdf(lower) - log_p)       # B = f(l) / p
        return upper, lower, log_p, ratio_u, ratio_l

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pieces = self._ratios(theta)
        if pieces is None:
            return _rejected(self.size)
        upper, lower, log_p, ratio_u, ratio_l = pieces
        k, cuts, w, codes = self.k, self.categories - 1, self.w, self.codes
        slope_u = ratio_u * self._log_slope(upper)
        slope_l = ratio_l * self._log_slope(lower)
        h_uu = slope_u - ratio_u.square()
        h_ll = -slope_l - ratio_l.square()
        h_ul = ratio_u * ratio_l
        value = (w * log_p).sum()
        gradient = torch.empty(self.size, dtype=torch.float64)
        gradient[:k] = self.x.T @ (w * (ratio_l - ratio_u))
        sum_u = group_sums(w * ratio_u, codes, cuts + 1)
        sum_l = group_sums(w * ratio_l, codes, cuts + 1)
        gradient[k:] = sum_u[:cuts] - sum_l[1:]
        hessian = torch.zeros((self.size, self.size), dtype=torch.float64)
        h_ee = w * (h_uu + 2 * h_ul + h_ll)
        hessian[:k, :k] = self.x.T @ (self.x * h_ee[:, None])
        cross_u = group_sums(self.x * (w * (h_uu + h_ul))[:, None], codes, cuts + 1)
        cross_l = group_sums(self.x * (w * (h_ll + h_ul))[:, None], codes, cuts + 1)
        cross = -(cross_u[:cuts] + cross_l[1:]).T                   # [k, J-1]
        hessian[:k, k:] = cross
        hessian[k:, :k] = cross.T
        diagonal = group_sums(w * h_uu, codes, cuts + 1)[:cuts] \
            + group_sums(w * h_ll, codes, cuts + 1)[1:]
        index = torch.arange(cuts)
        hessian[k + index, k + index] = diagonal
        if cuts > 1:
            adjacent = group_sums(w * h_ul, codes, cuts + 1)[1:cuts]
            hessian[k + index[:-1], k + index[1:]] = adjacent
            hessian[k + index[1:], k + index[:-1]] = adjacent
        return value, gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k + J - 1], before weights."""
        pieces = self._ratios(theta)
        if pieces is None:
            raise KernelError("numerical_failure", "The cutpoints are not increasing.")
        _, _, _, ratio_u, ratio_l = pieces
        k, cuts, codes = self.k, self.categories - 1, self.codes
        rows = torch.zeros((self.n, self.size), dtype=torch.float64)
        rows[:, :k] = self.x * (ratio_l - ratio_u)[:, None]
        position = torch.arange(self.n)
        has_upper, has_lower = codes < cuts, codes > 0
        rows[position[has_upper], k + codes[has_upper]] = ratio_u[has_upper]
        rows[position[has_lower], k + codes[has_lower] - 1] = -ratio_l[has_lower]
        return rows


class MultinomialObjective:
    """Log likelihood of the multinomial logit model, one equation per non-base category.

    ``theta`` stacks the equations: ``theta[j * k:(j + 1) * k]`` are the
    coefficients of the j-th non-base category (categories in code order with
    the base removed).
    """

    def __init__(self, x: Tensor, codes: Tensor, n_categories: int, base: int, weights: Tensor):
        if n_categories < 2:
            raise KernelError("constant_outcome", "A multinomial model needs at least two "
                              "outcome categories.")
        self.x, self.w = x, weights
        self.n, self.k = x.shape
        self.equations = n_categories - 1
        self.size = self.k * self.equations
        # Slot of each observation's category among the equations; the base is slot J-1,
        # whose linear predictor is the appended zero column.
        slot = torch.arange(n_categories)
        slot = torch.where(slot > base, slot - 1, slot)
        slot[base] = self.equations
        self.slot = slot[codes]

    def _eta(self, theta: Tensor) -> Tensor:
        eta = self.x @ theta.reshape(self.equations, self.k).T
        return torch.cat([eta, torch.zeros((self.n, 1), dtype=torch.float64)], dim=1)

    def value(self, theta: Tensor) -> Tensor:
        eta = self._eta(theta)
        own = eta.gather(1, self.slot[:, None])[:, 0]
        return (self.w * (own - torch.logsumexp(eta, dim=1))).sum()

    def probabilities(self, theta: Tensor) -> Tensor:
        """Fitted probability of each observation's own category."""
        eta = self._eta(theta)
        own = eta.gather(1, self.slot[:, None])[:, 0]
        return torch.exp(own - torch.logsumexp(eta, dim=1))

    def excluded(self, theta: Tensor, threshold: float) -> int:
        """Observations for which some category has a fitted probability below ``threshold``.

        A regressor that rules a category out for part of the sample (quasi-complete
        separation) drives that category's probability to zero there without predicting
        any observation's own category perfectly. Returns zero when some observation's
        OWN category is the one ruled out (a diverging iteration, not separation).
        """
        eta = self._eta(theta)
        log_p = eta - torch.logsumexp(eta, dim=1, keepdim=True)
        limit = math.log(threshold)
        if bool((log_p.gather(1, self.slot[:, None]) < limit).any()):
            return 0
        return int((log_p < limit).any(dim=1).sum())

    def _residuals(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta = self._eta(theta)
        log_sum = torch.logsumexp(eta, dim=1)
        own = eta.gather(1, self.slot[:, None])[:, 0]
        probability = torch.exp(eta[:, :self.equations] - log_sum[:, None])
        residual = -probability
        chosen = self.slot < self.equations
        rows = torch.arange(self.n)[chosen]
        residual[rows, self.slot[chosen]] += 1.0
        return (self.w * (own - log_sum)).sum(), probability, residual

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, probability, residual = self._residuals(theta)
        k, m = self.k, self.equations
        gradient = (self.x.T @ (residual * self.w[:, None])).T.reshape(-1)
        outer = torch.zeros((self.size, self.size), dtype=torch.float64)
        diagonal = torch.zeros((m, k, k), dtype=torch.float64)
        block = max(1024, _BLOCK_ELEMENTS // max(1, self.size))
        for start in range(0, self.n, block):
            stop = min(self.n, start + block)
            x, p, w = self.x[start:stop], probability[start:stop], self.w[start:stop]
            stacked = (p[:, :, None] * x[:, None, :]).reshape(stop - start, self.size)
            stacked = stacked * w.sqrt()[:, None]
            outer += stacked.T @ stacked
            weighted = x * w[:, None]
            for j in range(m):
                diagonal[j] += x.T @ (weighted * p[:, j:j + 1])
        hessian = outer
        for j in range(m):
            hessian[j * k:(j + 1) * k, j * k:(j + 1) * k] -= diagonal[j]
        return value, gradient, (hessian + hessian.T) / 2

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k (J - 1)], before weights."""
        _, _, residual = self._residuals(theta)
        return (residual[:, :, None] * self.x[:, None, :]).reshape(self.n, self.size)


class HetprobitObjective:
    """Log likelihood of ``Pr(y = 1) = Phi(x'b / exp(z'g))`` in ``(b, g)``.

    With no variance regressors (``z`` of width zero) this is the probit model.
    """

    def __init__(self, x: Tensor, z: Tensor, y: Tensor, weights: Tensor):
        self.x, self.z, self.w = x, z, weights
        self.sign = 2 * y - 1
        self.k, self.q = x.shape[1], z.shape[1]
        self.size = self.k + self.q

    def _index(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        scale = torch.exp(-(self.z @ theta[self.k:])) if self.q else None
        index = self.x @ theta[:self.k]
        return (index if scale is None else index * scale), scale

    def value(self, theta: Tensor) -> Tensor:
        index, _ = self._index(theta)
        return (self.w * torch.special.log_ndtr(self.sign * index)).sum()

    def probabilities(self, theta: Tensor) -> Tensor:
        """Fitted ``Pr(y = 1)``."""
        return torch.special.ndtr(self._index(theta)[0])

    def outcome_probabilities(self, theta: Tensor) -> Tensor:
        """Fitted probability of each observation's own outcome."""
        return torch.special.ndtr(self.sign * self._index(theta)[0])

    def _terms(self, theta: Tensor) -> tuple[Tensor, ...]:
        index, scale = self._index(theta)
        log_cdf, ratio, curvature = mills_ratio(self.sign * index)
        return index, scale, log_cdf, self.sign * ratio, curvature

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        index, scale, log_cdf, mills, curvature = self._terms(theta)
        k, w = self.k, self.w
        value = (w * log_cdf).sum()
        if not bool(torch.isfinite(index).all()):
            return _rejected(self.size)
        inverse = torch.ones_like(index) if scale is None else scale        # 1 / sigma
        gradient = torch.empty(self.size, dtype=torch.float64)
        gradient[:k] = self.x.T @ (w * mills * inverse)
        hessian = torch.empty((self.size, self.size), dtype=torch.float64)
        hessian[:k, :k] = -(self.x.T @ (self.x * (w * curvature * inverse.square())[:, None]))
        if self.q:
            gradient[k:] = -(self.z.T @ (w * mills * index))
            cross = self.x.T @ (self.z * (w * (curvature * index - mills) * inverse)[:, None])
            hessian[:k, k:] = cross
            hessian[k:, :k] = cross.T
            own = w * (mills * index - curvature * index.square())
            hessian[k:, k:] = self.z.T @ (self.z * own[:, None])
        return value, gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k + q], before weights."""
        index, scale, _, mills, _ = self._terms(theta)
        mean = self.x * (mills if scale is None else mills * scale)[:, None]
        if not self.q:
            return mean
        return torch.cat([mean, self.z * (-mills * index)[:, None]], dim=1)
