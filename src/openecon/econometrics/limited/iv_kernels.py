"""Likelihood kernel of probit / tobit models with continuous endogenous regressors.

Model
-----
    y1* = w'd + u,              w = [x1, y2]   (exogenous, then endogenous regressors)
    y2  = Pi z + v,             z = [x1, x2]   (all exogenous variables), p equations
    (u, v) ~ N(0, Sigma),

with ``y1 = 1[y1* > 0]`` (``ivprobit``, ``var(u) = 1``) or ``y1*`` censored at
its limits (``ivtobit``). The joint density factors as
``f(y1 | y2, z) f(y2 | z)``, and ``u | v ~ N(v'a, omega^2)``.

Working parameterization
------------------------
The likelihood is maximized in a recursive (triangular) form in which every
equation is a regression with its own independent error, so the problem is
unconstrained for any number of endogenous regressors:

    r_j = y2_j - g_j'f_j,   g_j = [z, y2_1, ..., y2_{j-1}],   r_j ~ N(0, s_j^2),  j = 1..p,
    index  n = w'd~ + r'k,
    ln L = sum_i w_i { h(n_i) + sum_j [-t_j - r_ij^2 exp(-2 t_j) / 2 - ln sqrt(2 pi)] },

``t_j = ln s_j``. ``h`` is the probit term ``ln Phi(q n)`` (``q = 2 y1 - 1``;
then ``d~ = d / omega`` and ``k = a~ / omega`` are scaled by the conditional
standard deviation ``omega`` of ``u`` given ``v``), or the censored-normal
term of ``kernels.CensoredRows`` with its own scale ``t_0 = ln omega`` (then
``d~ = d``). The residuals ``r = A v`` are the innovations of the Cholesky
(LDL') factorization of ``Var(v)``: ``A`` is unit lower triangular with
``A[j, k] = -c_jk`` and ``c_jk`` the coefficient of ``y2_k`` in ``f_j``.

With ``h'`` and ``h''`` the derivatives of ``h`` in the index, ``P_j =
exp(-2 t_j)`` and ``D_i = dn_i/dtheta = [w, r, -k_1 g_1, ..., -k_p g_p]``:

    dl/dd~ = h' w,        dl/dk_j = h' r_j,     dl/df_j = (-h' k_j + r_j P_j) g_j,
    dl/dt_j = r_j^2 P_j - 1,
    d2l = h'' D D'  +  h' d2n  +  Gaussian part,
    d2n/dk_j df_j = -g_j,
    Gaussian: d2l/df_j df_j' = -P_j g_j g_j',  d2l/df_j dt_j = -2 r_j P_j g_j,
              d2l/dt_j^2 = -2 r_j^2 P_j.

Every block of the Hessian is a sub-block of three weighted cross products of
the fixed matrix ``B = [z, y2]`` and the residuals ``R``, so an iteration costs
O(n (k_z + p)^2) and no n-by-K score or direction matrix is formed.

Reported parameterization (``structural``)
------------------------------------------
Stata's parameters are recovered exactly, with their covariance by the delta
method (analytic Jacobian). With ``B = A^-1``, ``S = diag(s_j^2)``:

    Pi = B Pi*,                      Var(v) = B S B',
    ivprobit: omega = (1 + k'S k)^(-1/2),  d = omega d~,  Cov(v, u) = omega B S k,  Var(u) = 1,
    ivtobit:  omega = exp(t_0),            d = d~,        Cov(v, u) = B S k,
              Var(u) = omega^2 + k'S k,

and the ancillary parameters are ``athrho{i}_{j} = atanh corr`` between the
errors of equations i > j (equation 1 is the outcome equation, equation
``m + 1`` the m-th endogenous regressor) and ``lnsigma{i} = ln sd`` (equation
1 only for ``ivtobit``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import weighted_crossprod

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


class ProbitRows:
    """Row terms ``ln Phi(q n)`` of a probit outcome equation (no scale parameter)."""

    has_scale = False

    def __init__(self, y: Tensor):
        if bool(((y != 0) & (y != 1)).any()):
            raise KernelError("invalid_binary_outcome", "The outcome must be coded 0/1.")
        self.sign = 2 * y - 1

    def log_rows(self, index: Tensor, tau: Tensor | None) -> Tensor:
        return torch.special.log_ndtr(self.sign * index)

    def rows(self, index: Tensor, tau: Tensor | None) -> tuple[Tensor, ...]:
        log_cdf, ratio, curvature = mills_ratio(self.sign * index)
        zero = torch.zeros_like(index)
        return log_cdf, self.sign * ratio, zero, -curvature, zero, zero


class IVObjective:
    """Joint log likelihood of the outcome equation and the reduced forms (see module notes).

    ``z`` [n, kz] holds the exogenous variables with the ``k_exog`` included
    ones first, ``y2`` [n, p] the endogenous regressors and ``outcome`` the row
    terms of the outcome equation (``ProbitRows`` or ``kernels.CensoredRows``).
    ``theta = (d~ [k_exog + p], k [p], f_1 [kz], ..., f_p [kz + p - 1], t [p], t_0)``.
    """

    def __init__(self, z: Tensor, y2: Tensor, k_exog: int, outcome, weights: Tensor):
        if z.ndim != 2 or y2.ndim != 2 or z.shape[0] != y2.shape[0] or y2.shape[1] < 1:
            raise KernelError("invalid_design", "The instrument and endogenous blocks do not "
                              "conform.")
        self.n, self.kz = z.shape
        self.p, self.k1 = y2.shape[1], k_exog
        self.kw = k_exog + self.p
        self.base = torch.cat([z, y2], dim=1)
        self.y2, self.w, self.outcome = y2, weights, outcome
        self.scale = bool(outcome.has_scale)
        self.columns_w = [*range(k_exog), *range(self.kz, self.kz + self.p)]
        self.o_kappa = self.kw
        self.o_phi = []
        position = self.kw + self.p
        for j in range(self.p):
            self.o_phi.append(position)
            position += self.kz + j
        self.o_tau = position
        self.size = position + self.p + int(self.scale)
        self.gram = weighted_crossprod(self.base, weights)
        self.total = weights.sum()

    # ---- unpacking -------------------------------------------------------------

    def _unpack(self, theta: Tensor) -> tuple[Tensor, ...]:
        kz, p = self.kz, self.p
        delta, kappa = theta[:self.kw], theta[self.kw:self.kw + p]
        phi = torch.zeros((kz + p, p), dtype=torch.float64)
        for j in range(p):
            phi[:kz + j, j] = theta[self.o_phi[j]:self.o_phi[j] + kz + j]
        tau = theta[self.o_tau:self.o_tau + p]
        tau0 = theta[self.o_tau + p] if self.scale else None
        return delta, kappa, phi, tau, tau0

    def _valid(self, theta: Tensor) -> bool:
        return bool((theta[self.o_tau:].abs() <= 300).all())

    def residuals(self, theta: Tensor) -> Tensor:
        """Recursive reduced-form residuals ``r`` [n, p]."""
        return self.y2 - self.base @ self._unpack(theta)[2]

    def _index(self, delta: Tensor, kappa: Tensor, resid: Tensor) -> Tensor:
        direction = torch.zeros(self.kz + self.p, dtype=torch.float64)
        direction[self.columns_w] = delta
        return self.base @ direction + resid @ kappa

    def index(self, theta: Tensor) -> Tensor:
        """Index ``w'd~ + r'k`` of the outcome equation given the endogenous regressors."""
        delta, kappa, phi, _, _ = self._unpack(theta)
        return self._index(delta, kappa, self.y2 - self.base @ phi)

    # ---- objective -------------------------------------------------------------

    def value(self, theta: Tensor) -> Tensor:
        if not self._valid(theta):
            return torch.tensor(-math.inf, dtype=torch.float64)
        delta, kappa, phi, tau, tau0 = self._unpack(theta)
        resid = self.y2 - self.base @ phi
        log_l = self.outcome.log_rows(self._index(delta, kappa, resid), tau0)
        squares = (self.w[:, None] * resid.square()).sum(dim=0)
        normal = -(tau + _LOG_SQRT_2PI).sum() * self.total \
            - 0.5 * (squares * torch.exp(-2 * tau)).sum()
        return (self.w * log_l).sum() + normal

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self._valid(theta):
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(self.size, dtype=torch.float64),
                    -torch.eye(self.size, dtype=torch.float64))
        kz, p, kw, w, base = self.kz, self.p, self.kw, self.w, self.base
        delta, kappa, phi, tau, tau0 = self._unpack(theta)
        resid = self.y2 - base @ phi
        log_l, g_m, g_t, h_mm, h_mt, h_tt = self.outcome.rows(
            self._index(delta, kappa, resid), tau0)
        precision = torch.exp(-2 * tau)
        slope, curve = w * g_m, w * h_mm
        s_bb = weighted_crossprod(base, curve)
        s_br = weighted_crossprod(base, curve, resid)
        s_rr = weighted_crossprod(resid, curve)
        b_g, r_g = base.T @ slope, resid.T @ slope
        b_wr = weighted_crossprod(base, w, resid)
        squares = (w[:, None] * resid.square()).sum(dim=0)
        value = (w * log_l).sum() - (tau + _LOG_SQRT_2PI).sum() * self.total \
            - 0.5 * (squares * precision).sum()
        cw, ok, ot = self.columns_w, self.o_kappa, self.o_tau
        gradient = torch.zeros(self.size, dtype=torch.float64)
        hessian = torch.zeros((self.size, self.size), dtype=torch.float64)
        gradient[:kw] = b_g[cw]
        gradient[ok:ok + p] = r_g
        gradient[ot:ot + p] = precision * squares - self.total
        hessian[:kw, :kw] = s_bb[cw][:, cw]
        hessian[:kw, ok:ok + p] = s_br[cw]
        hessian[ok:ok + p, ok:ok + p] = s_rr
        hessian[ot:ot + p, ot:ot + p] = torch.diag(-2 * precision * squares)
        for j in range(p):
            width = kz + j
            rows = slice(self.o_phi[j], self.o_phi[j] + width)
            gradient[rows] = -kappa[j] * b_g[:width] + precision[j] * b_wr[:width, j]
            hessian[:kw, rows] = -kappa[j] * s_bb[cw][:, :width]
            hessian[ok:ok + p, rows] = -kappa[j] * s_br[:width].T
            hessian[ok + j, rows] -= b_g[:width]
            for other in range(j, p):
                span = kz + other
                columns = slice(self.o_phi[other], self.o_phi[other] + span)
                block = kappa[j] * kappa[other] * s_bb[:width, :span]
                if other == j:
                    block = block - precision[j] * self.gram[:width, :width]
                hessian[rows, columns] = block
            hessian[rows, ot + j] = -2 * precision[j] * b_wr[:width, j]
        if self.scale:
            last = ot + p
            b_t, r_t = base.T @ (w * h_mt), resid.T @ (w * h_mt)
            gradient[last] = (w * g_t).sum()
            hessian[:kw, last] = b_t[cw]
            hessian[ok:ok + p, last] = r_t
            for j in range(p):
                width = kz + j
                hessian[self.o_phi[j]:self.o_phi[j] + width, last] = -kappa[j] * b_t[:width]
            hessian[last, last] = (w * h_tt).sum()
        upper = torch.triu(hessian)
        return value, gradient, upper + torch.triu(hessian, 1).T

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, size], before weights."""
        kz, p, kw = self.kz, self.p, self.kw
        delta, kappa, phi, tau, tau0 = self._unpack(theta)
        resid = self.y2 - self.base @ phi
        _, g_m, g_t, _, _, _ = self.outcome.rows(self._index(delta, kappa, resid), tau0)
        precision = torch.exp(-2 * tau)
        rows = torch.empty((self.n, self.size), dtype=torch.float64)
        rows[:, :kw] = self.base[:, self.columns_w] * g_m[:, None]
        rows[:, self.o_kappa:self.o_kappa + p] = resid * g_m[:, None]
        for j in range(p):
            width = kz + j
            factor = -kappa[j] * g_m + precision[j] * resid[:, j]
            rows[:, self.o_phi[j]:self.o_phi[j] + width] = self.base[:, :width] * factor[:, None]
        rows[:, self.o_tau:self.o_tau + p] = resid.square() * precision - 1
        if self.scale:
            rows[:, self.o_tau + p] = g_t
        return rows

    # ---- reported parameters ---------------------------------------------------

    def ancillary_names(self) -> list[str]:
        """Stata's names of the ancillary parameters, in reporting order."""
        names = [f"/athrho{i + 1}_{j + 1}" for i in range(1, self.p + 1) for j in range(i)]
        names.extend(f"/lnsigma{i + 1}" for i in range(0 if self.scale else 1, self.p + 1))
        return names

    def structural(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """Reported parameters ``(d, Pi rows, athrho.., lnsigma..)`` and their Jacobian.

        The Jacobian [size, size] is analytic (see the module notes); the
        covariance of the reported parameters is ``J V J'``.
        """
        kz, p, kw, size = self.kz, self.p, self.kw, self.size
        delta, kappa, phi, tau, tau0 = self._unpack(theta)
        s2 = torch.exp(2 * tau)
        eye = torch.eye(p, dtype=torch.float64)
        inverse = torch.linalg.solve_triangular(eye - phi[kz:].T.tril(-1), eye, upper=False)
        pi = inverse @ phi[:kz].T                                   # [p, kz]
        quad = float((kappa.square() * s2).sum())
        if self.scale:
            omega, psi = math.exp(float(tau0)), 1.0
            var_u = omega ** 2 + quad
        else:
            omega = psi = (1 + quad) ** -0.5
            var_u = 1.0
        var_v = inverse @ torch.diag(s2) @ inverse.T
        direction = inverse @ (s2 * kappa)                          # B S k
        cov_vu = psi * direction

        def ancillary(d_uu: float, d_vu: Tensor, d_vv: Tensor, level: bool) -> list[float]:
            out = []
            for i in range(p):
                for j in range(i + 1):
                    if j == 0:
                        cov, d_cov, var_b, d_var_b = cov_vu[i], d_vu[i], var_u, d_uu
                    else:
                        cov, d_cov = var_v[i, j - 1], d_vv[i, j - 1]
                        var_b, d_var_b = var_v[j - 1, j - 1], d_vv[j - 1, j - 1]
                    var_a, d_var_a = var_v[i, i], d_vv[i, i]
                    scale = math.sqrt(float(var_a) * float(var_b))
                    rho = float(cov) / scale
                    if level:
                        out.append(math.atanh(rho) if abs(rho) < 1 else math.copysign(
                            math.inf, rho))
                        continue
                    d_rho = float(d_cov) / scale - 0.5 * rho * (
                        float(d_var_a) / float(var_a) + float(d_var_b) / float(var_b))
                    out.append(d_rho / (1 - rho * rho))
            if self.scale:
                out.append(0.5 * math.log(var_u) if level else d_uu / (2 * var_u))
            for i in range(p):
                out.append(0.5 * math.log(float(var_v[i, i])) if level
                           else float(d_vv[i, i]) / (2 * float(var_v[i, i])))
            return out

        zero_v, zero_vv = torch.zeros(p, dtype=torch.float64), torch.zeros(
            (p, p), dtype=torch.float64)
        o_anc = kw + p * kz
        reported = torch.cat([delta * (1.0 if self.scale else omega), pi.reshape(-1),
                              torch.tensor(ancillary(0.0, zero_v, zero_vv, True),
                                           dtype=torch.float64)])
        jacobian = torch.zeros((size, size), dtype=torch.float64)
        jacobian[:kw, :kw] = torch.eye(kw, dtype=torch.float64) * (1.0 if self.scale else omega)

        def moments(column: int, d_omega: float, d_uu: float, d_vu: Tensor, d_vv: Tensor) -> None:
            if not self.scale:
                jacobian[:kw, column] = delta * d_omega
            jacobian[o_anc:, column] = torch.tensor(ancillary(d_uu, d_vu, d_vv, False),
                                                    dtype=torch.float64)

        for m in range(p):
            # k_m and t_m change k'S k, hence omega (probit) or Var(u) (tobit).
            share = float(kappa[m] * s2[m])
            for column, d_quad, d_dir, d_vv in (
                (self.o_kappa + m, 2 * share, inverse[:, m] * s2[m], zero_vv),
                (self.o_tau + m, 2 * share * float(kappa[m]), 2 * share * inverse[:, m],
                 2 * s2[m] * torch.outer(inverse[:, m], inverse[:, m])),
            ):
                if self.scale:
                    moments(column, 0.0, d_quad, d_dir, d_vv)
                else:
                    d_omega = -0.5 * omega ** 3 * d_quad
                    moments(column, d_omega, 0.0, omega * d_dir + d_omega * direction, d_vv)
            # Reduced-form coefficients: Pi = B Pi*.
            for i in range(p):
                block = slice(kw + i * kz, kw + (i + 1) * kz)
                jacobian[block, self.o_phi[m]:self.o_phi[m] + kz] = \
                    inverse[i, m] * torch.eye(kz, dtype=torch.float64)
            # c_mk: dB = B E_mk B.
            for k in range(m):
                column = self.o_phi[m] + kz + k
                d_inverse = torch.outer(inverse[:, m], inverse[k])
                half = d_inverse @ torch.diag(s2) @ inverse.T
                moments(column, 0.0, 0.0, psi * (d_inverse @ (s2 * kappa)), half + half.T)
                jacobian[kw:kw + p * kz, column] = torch.outer(inverse[:, m], pi[k]).reshape(-1)
        if self.scale:
            moments(self.o_tau + p, 0.0, 2 * omega ** 2, zero_v, zero_vv)
        return reported, jacobian
