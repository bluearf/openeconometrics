"""Sample-selection likelihood kernels on float64 tensors.

No autograd: both objectives return the analytic gradient and Hessian, which
the tests compare with numerical derivatives. ``w_i`` is the weight of
observation i in every likelihood sum. The selected and the nonselected rows
are split once at construction, so each iteration touches every row once.

Heckman selection model (``HeckmanObjective``)
----------------------------------------------
Outcome ``y = x'b + u1`` observed when ``z'g + u2 > 0``; ``u1 ~ N(0, sigma^2)``,
``u2 ~ N(0, 1)``, ``corr(u1, u2) = rho``. Parameters ``(b, g, a, tau)`` with
``rho = tanh(a)`` (Stata's ``/athrho``) and ``sigma = exp(tau)``
(``/lnsigma``). With ``e = (y - x'b) / sigma``, ``c = cosh(a) = (1 - rho^2)^(-1/2)``
and ``h = sinh(a) = rho c``:

    selected:     l = ln Phi(n) - e^2 / 2 - tau - ln sqrt(2 pi),   n = c z'g + h e,
    nonselected:  l = ln Phi(-z'g).

With the inverse Mills ratio ``r = phi(n) / Phi(n)``, ``v = r (r + n)`` and
``n_a = dn/da = h z'g + c e``, the derivatives of a selected row with respect
to ``m = x'b``, ``zg = z'g``, ``a`` and ``tau`` are

    l_m = (e - r h) / sigma,     l_zg = r c,     l_a = r n_a,     l_tau = e^2 - 1 - r h e,
    l_m,m   = -(v h^2 + 1) / sigma^2,          l_m,zg  = v h c / sigma,
    l_m,a   = (v h n_a - r c) / sigma,         l_m,tau = (r h - 2 e - v h^2 e) / sigma,
    l_zg,zg = -v c^2,                          l_zg,a  = r h - v c n_a,
    l_zg,tau = v c h e,                        l_a,a   = r n - v n_a^2,
    l_a,tau = v n_a h e - r c e,               l_tau,tau = r h e - 2 e^2 - v h^2 e^2,

and a nonselected row has ``l_zg = -r0``, ``l_zg,zg = -r0 (r0 - zg)`` with
``r0 = phi(zg) / Phi(-zg)``. Only univariate normal functions are needed.

Probit with sample selection (``HeckprobitObjective``)
-----------------------------------------------------
Binary outcome ``y = 1[x'b + u1 > 0]`` observed when ``z'g + u2 > 0`` with
standard bivariate normal errors of correlation ``rho = tanh(a)``:

    selected, y = 1:   l = ln Phi2(x'b, z'g; rho),
    selected, y = 0:   l = ln Phi2(-x'b, z'g; -rho),
    nonselected:       l = ln Phi(-z'g).

A selected row is exactly a bivariate-probit observation whose second outcome
equals one, so the selected part is the discrete family's
``BiprobitObjective`` (Genz's bivariate normal distribution function with a
tail-accurate logarithm, analytic score and Hessian) evaluated on the selected
rows; the nonselected rows add a probit term in ``g``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.discrete.bivariate import BiprobitObjective
from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import weighted_crossprod

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
# |athrho| beyond which a trial point is rejected: |rho| > 1 - 2e-13.
ATHRHO_LIMIT = 15.0


def _rejected(size: int) -> tuple[Tensor, Tensor, Tensor]:
    return (torch.tensor(-math.inf, dtype=torch.float64), torch.zeros(size, dtype=torch.float64),
            -torch.eye(size, dtype=torch.float64))


def _split(selected: Tensor) -> tuple[Tensor, Tensor]:
    if selected.dtype != torch.bool or selected.ndim != 1:
        raise KernelError("invalid_design", "The selection indicator must be a boolean vector.")
    return selected.nonzero().flatten(), (~selected).nonzero().flatten()


class HeckmanObjective:
    """Log likelihood of the Heckman selection model in ``(b, g, athrho, lnsigma)``.

    ``x`` [n, k] and ``z`` [n, q] are the outcome and selection designs of every
    row, ``y`` the outcome (ignored where not selected) and ``selected`` the
    boolean selection indicator.
    """

    def __init__(self, x: Tensor, z: Tensor, y: Tensor, selected: Tensor, weights: Tensor):
        self._on, self._off = _split(selected)
        self.n, self.k, self.q = x.shape[0], x.shape[1], z.shape[1]
        self.size = self.k + self.q + 2
        self.x1, self.z1, self.y1 = x[self._on], z[self._on], y[self._on]
        self.z0 = z[self._off]
        self.w1, self.w0 = weights[self._on], weights[self._off]
        if not bool(torch.isfinite(self.y1).all()):
            raise KernelError("non_finite_values", "Selected outcomes must be finite.")

    def _valid(self, theta: Tensor) -> bool:
        return abs(float(theta[-2])) <= ATHRHO_LIMIT and abs(float(theta[-1])) <= 300

    def _pieces(self, theta: Tensor) -> tuple[Tensor, ...]:
        k, q = self.k, self.q
        athrho, tau = theta[k + q], theta[k + q + 1]
        cosh, sinh, inverse = torch.cosh(athrho), torch.sinh(athrho), torch.exp(-tau)
        e = (self.y1 - self.x1 @ theta[:k]) * inverse
        zg = self.z1 @ theta[k:k + q]
        index = cosh * zg + sinh * e
        return e, zg, index, cosh, sinh, inverse, tau

    def value(self, theta: Tensor) -> Tensor:
        if not self._valid(theta):
            return torch.tensor(-math.inf, dtype=torch.float64)
        e, _, index, _, _, _, tau = self._pieces(theta)
        on = torch.special.log_ndtr(index) - 0.5 * e.square() - tau - _LOG_SQRT_2PI
        off = torch.special.log_ndtr(-(self.z0 @ theta[self.k:self.k + self.q]))
        return (self.w1 * on).sum() + (self.w0 * off).sum()

    def _first(self, theta: Tensor) -> tuple[Tensor, ...]:
        """Row derivatives with respect to (m, zg, a, tau) and what the Hessian reuses."""
        e, zg, index, cosh, sinh, inverse, tau = self._pieces(theta)
        log_cdf, ratio, curvature = mills_ratio(index)
        slope = sinh * zg + cosh * e                                    # dn/da
        l_m = (e - ratio * sinh) * inverse
        l_zg = ratio * cosh
        l_a = ratio * slope
        l_tau = e.square() - 1 - ratio * sinh * e
        log_off, ratio_off, curvature_off = mills_ratio(
            -(self.z0 @ theta[self.k:self.k + self.q]))
        value = (self.w1 * (log_cdf - 0.5 * e.square() - tau - _LOG_SQRT_2PI)).sum() \
            + (self.w0 * log_off).sum()
        return (value, l_m, l_zg, l_a, l_tau, ratio_off, curvature_off, e, index, ratio,
                curvature, slope, cosh, sinh, inverse)

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self._valid(theta):
            return _rejected(self.size)
        (value, l_m, l_zg, l_a, l_tau, ratio_off, curvature_off, e, index, ratio, curvature,
         slope, cosh, sinh, inverse) = self._first(theta)
        k, q, w1, w0 = self.k, self.q, self.w1, self.w0
        x1, z1, z0 = self.x1, self.z1, self.z0
        a, t = k + q, k + q + 1
        gradient = torch.empty(self.size, dtype=torch.float64)
        gradient[:k] = x1.T @ (w1 * l_m)
        gradient[k:a] = z1.T @ (w1 * l_zg) - z0.T @ (w0 * ratio_off)
        gradient[a] = (w1 * l_a).sum()
        gradient[t] = (w1 * l_tau).sum()
        v, r = curvature, ratio
        h_mm = -(v * sinh.square() + 1) * inverse.square()
        h_mz = v * sinh * cosh * inverse
        h_ma = (v * sinh * slope - r * cosh) * inverse
        h_mt = (r * sinh - 2 * e - v * sinh.square() * e) * inverse
        h_zz = -v * cosh.square()
        h_za = r * sinh - v * cosh * slope
        h_zt = v * cosh * sinh * e
        h_aa = r * index - v * slope.square()
        h_at = v * slope * sinh * e - r * cosh * e
        h_tt = r * sinh * e - 2 * e.square() - v * sinh.square() * e.square()
        hessian = torch.empty((self.size, self.size), dtype=torch.float64)
        hessian[:k, :k] = weighted_crossprod(x1, w1 * h_mm)
        cross = weighted_crossprod(x1, w1 * h_mz, z1)
        hessian[:k, k:a] = cross
        hessian[k:a, :k] = cross.T
        hessian[k:a, k:a] = weighted_crossprod(z1, w1 * h_zz) \
            - weighted_crossprod(z0, w0 * curvature_off)
        for column, with_m, with_z in ((a, h_ma, h_za), (t, h_mt, h_zt)):
            block = torch.cat([x1.T @ (w1 * with_m), z1.T @ (w1 * with_z)])
            hessian[:a, column] = block
            hessian[column, :a] = block
        hessian[a, a] = (w1 * h_aa).sum()
        hessian[t, t] = (w1 * h_tt).sum()
        hessian[a, t] = hessian[t, a] = (w1 * h_at).sum()
        return value, gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k + q + 2] in the original row order, before weights."""
        _, l_m, l_zg, l_a, l_tau, ratio_off, *_ = self._first(theta)
        k, q = self.k, self.q
        rows = torch.zeros((self.n, self.size), dtype=torch.float64)
        rows[self._on] = torch.cat([self.x1 * l_m[:, None], self.z1 * l_zg[:, None],
                                    l_a[:, None], l_tau[:, None]], dim=1)
        rows[self._off, k:k + q] = -self.z0 * ratio_off[:, None]
        return rows

    def selection_probability(self, theta: Tensor, z: Tensor) -> Tensor:
        """Fitted ``Pr(selected) = Phi(z'g)`` for the rows of ``z``."""
        return torch.special.ndtr(z @ theta[self.k:self.k + self.q])


class HeckprobitObjective:
    """Log likelihood of the probit model with sample selection in ``(b, g, athrho)``."""

    def __init__(self, x: Tensor, z: Tensor, y: Tensor, selected: Tensor, weights: Tensor):
        self._on, self._off = _split(selected)
        self.n, self.k, self.q = x.shape[0], x.shape[1], z.shape[1]
        self.size = self.k + self.q + 1
        y1 = y[self._on]
        if bool(((y1 != 0) & (y1 != 1)).any()):
            raise KernelError("invalid_binary_outcome", "Selected outcomes must be coded 0/1.")
        self.z0, self.w0 = z[self._off], weights[self._off]
        self.joint = BiprobitObjective(x[self._on], z[self._on], y1, torch.ones_like(y1),
                                       weights[self._on])

    def _off_index(self, theta: Tensor) -> Tensor:
        return -(self.z0 @ theta[self.k:self.k + self.q])

    def value(self, theta: Tensor) -> Tensor:
        if abs(float(theta[-1])) > ATHRHO_LIMIT:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return self.joint.value(theta) \
            + (self.w0 * torch.special.log_ndtr(self._off_index(theta))).sum()

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if abs(float(theta[-1])) > ATHRHO_LIMIT:
            return _rejected(self.size)
        value, gradient, hessian = self.joint(theta)
        if not bool(torch.isfinite(value)):
            return _rejected(self.size)
        k, q = self.k, self.q
        log_off, ratio_off, curvature_off = mills_ratio(self._off_index(theta))
        gradient = gradient.clone()
        hessian = hessian.clone()
        gradient[k:k + q] -= self.z0.T @ (self.w0 * ratio_off)
        hessian[k:k + q, k:k + q] -= weighted_crossprod(self.z0, self.w0 * curvature_off)
        return value + (self.w0 * log_off).sum(), gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k + q + 1] in the original row order, before weights."""
        k, q = self.k, self.q
        rows = torch.zeros((self.n, self.size), dtype=torch.float64)
        rows[self._on] = self.joint.score_rows(theta)
        _, ratio_off, _ = mills_ratio(self._off_index(theta))
        rows[self._off, k:k + q] = -self.z0 * ratio_off[:, None]
        return rows

    def selected_probabilities(self, theta: Tensor) -> Tensor:
        """Joint probability of each selected row's own outcome and of being selected."""
        return self.joint.outcome_probabilities(theta)
