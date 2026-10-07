"""Linear mixed-model likelihood on group sufficient statistics (tensors only).

Model
-----
One or two nested levels. Observations i of the lowest-level group c (the
"class") follow

    y = X b + Z_c u_c + 1 v_s + e,   u_c ~ N(0, G),  v_s ~ N(0, g),  e ~ N(0, s2 I),

where the optional top level s (two-level models only) has a random intercept
v_s and the lowest level has the q random effects u_c with design Z (random
slopes and/or an intercept). Per top group the covariance is

    V_s = s2 (A_s + d 1 1'),     A_s = blockdiag_c(I + Z_c D Z_c'),

with relative parameters D = G / s2 and d = g / s2.

Woodbury on q-by-q matrices
---------------------------
With D = L L' and M_c = I + L' Z_c'Z_c L (batched Cholesky over the classes):

    log|I + Z_c D Z_c'| = log|M_c|
    u' A_c^-1 v         = u'v - (L'Z_c'u)' M_c^-1 (L'Z_c'v)
    Z_c' A_c^-1 v       = Z_c'v - Z_c'Z_c L M_c^-1 L' Z_c'v.

The top level adds one rank-one update per top group: with
a_s = 1'A_s^-1 1 and b_s = 1'A_s^-1 [X y] (sums over the classes of s),

    log|A_s + d 11'| = log|A_s| + log(1 + d a_s)
    [X y]'(A_s + d 11')^-1 [X y] = [X y]'A_s^-1 [X y] - d/(1 + d a_s) b_s' b_s.

Everything therefore depends on the data only through X'X, X'y, y'y (global)
and, per class, Z'Z, Z'[1 X y] and 1'[1 X y]: these are accumulated ONCE by
``index_add_`` (O(N)), after which one likelihood evaluation costs
O(C q^2 (p + q)) and never forms an N-by-N matrix. Frequency weights enter
every cross product (a row counted f times within its class).

Likelihood
----------
For given variance parameters the fixed effects are the GLS estimate
b = (X'V^-1 X)^-1 X'V^-1 y (profiled out), and with p_R = p for REML, 0 for ML,

    l = -(N - p_R)/2 ln(2 pi s2) - 1/2 ln|V~| - r'V~^-1 r / (2 s2) [- 1/2 ln|X'V~^-1 X|],

V~ = V / s2. The REML form is Harville's restricted likelihood without the
ln|X'X| constant (Stata's and statsmodels' convention).

Gradient
--------
Analytic. With C~ = (X'V~^-1 X)^-1 and the envelope theorem for b:

    dl/dD  = sum_c [ -1/2 Z_c'V~^-1 Z_c + (Z_c'V~^-1 r)(.)'/(2 s2) + 1/2 Z_c'V~^-1 X C~ X'V~^-1 Z_c ]
    dl/dd  = sum_s [ -1/2 a_s/(1+d a_s) + (b_r,s/(1+d a_s))^2/(2 s2) + 1/2 b_X,s C~ b_X,s'/(1+d a_s)^2 ]
    dl/ds2 = -(N - p_R)/(2 s2) + r'V~^-1 r/(2 s2^2)          (at fixed D, d)

(the last terms of the first two lines are REML only). The absolute parameters
(G, g, s2) follow by the chain rule D = G/s2, d = g/s2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums
from openecon.engines.linalg import weighted_crossprod

_LOG_2PI = math.log(2 * math.pi)
STRUCTURES = ("independent", "unstructured", "exchangeable", "identity")


class CovStructure:
    """Parameterization of the lowest-level covariance G (q-by-q) by an unconstrained theta.

    identity      G = exp(2 t) I                              (1 parameter)
    independent   G = diag(exp(2 t_a))                        (q parameters)
    exchangeable  G = v [(1 - rho) I + rho 11'],  v = exp(2 t_1),
                  rho = (exp(t_2) - 1)/(exp(t_2) + q - 1)     (2 parameters; rho in (-1/(q-1), 1))
    unstructured  G = L L',  L lower triangular, L_aa = exp(t_aa), off-diagonal free
                  (q (q + 1)/2 parameters: the diagonal first, then the rows of the lower part)

    ``reported`` lists the variances and covariances shown to the user (Stata's
    order: variances, then covariances of pairs a < b); they are entries of G,
    so their Jacobian is a selection of ``jacobian``.
    """

    def __init__(self, kind: str, q: int):
        if kind not in STRUCTURES:
            raise KernelError("invalid_option", f"Unknown covariance structure '{kind}'.")
        if q < 1:
            raise KernelError("invalid_option", "A random-effects level needs at least one effect.")
        if q == 1:
            kind = "independent"            # every structure is one variance
        self.kind, self.q = kind, q
        self.lower = [(a, b) for a in range(q) for b in range(a)]
        self.size = {"identity": 1, "independent": q, "exchangeable": 2,
                     "unstructured": q + len(self.lower)}[kind]
        if kind in {"identity", "independent"}:
            self.entries = [(a, a) for a in range(q)][: 1 if kind == "identity" else q]
        elif kind == "exchangeable":
            self.entries = [(0, 0), (1, 0)]
        else:
            self.entries = [(a, a) for a in range(q)] + [(b, a) for a in range(q)
                                                        for b in range(a + 1, q)]

    def factor(self, theta: Tensor) -> Tensor:
        """A matrix L with G = L L' (lower triangular, or diagonal)."""
        q = self.q
        if self.kind == "identity":
            return torch.eye(q, dtype=torch.float64) * torch.exp(theta[0])
        if self.kind == "independent":
            return torch.diag(torch.exp(theta))
        if self.kind == "unstructured":
            factor = torch.diag(torch.exp(theta[:q]))
            if self.lower:
                rows, cols = zip(*self.lower, strict=True)
                factor[list(rows), list(cols)] = theta[q:]
            return factor
        factor, info = torch.linalg.cholesky_ex(self.matrix(theta))
        if int(info) != 0:
            raise KernelError("numerical_failure", "The exchangeable covariance is not positive "
                              "definite.")
        return factor

    def rho(self, theta: Tensor) -> Tensor:
        e = torch.exp(theta[1].clamp(max=700.0))
        return (e - 1) / (e + self.q - 1)

    def matrix(self, theta: Tensor) -> Tensor:
        if self.kind == "exchangeable":
            v, rho = torch.exp(2 * theta[0]), self.rho(theta)
            eye = torch.eye(self.q, dtype=torch.float64)
            return v * ((1 - rho) * eye + rho * torch.ones_like(eye))
        factor = self.factor(theta)
        return factor @ factor.T

    def jacobian(self, theta: Tensor) -> Tensor:
        """dG/dtheta as an [m, q, q] tensor."""
        q, m = self.q, self.size
        out = torch.zeros((m, q, q), dtype=torch.float64)
        if self.kind == "identity":
            out[0] = 2 * torch.exp(2 * theta[0]) * torch.eye(q, dtype=torch.float64)
        elif self.kind == "independent":
            index = torch.arange(q)
            out[index, index, index] = 2 * torch.exp(2 * theta)
        elif self.kind == "exchangeable":
            eye = torch.eye(q, dtype=torch.float64)
            out[0] = 2 * self.matrix(theta)
            e = torch.exp(theta[1].clamp(max=700.0))
            slope = q * e / (e + q - 1).square()
            out[1] = torch.exp(2 * theta[0]) * (torch.ones_like(eye) - eye) * slope
        else:
            factor = self.factor(theta)
            pairs = [(a, a) for a in range(q)] + self.lower
            for position, (a, b) in enumerate(pairs):
                unit = torch.zeros((q, q), dtype=torch.float64)
                unit[a, b] = factor[a, a] if a == b else 1.0
                out[position] = unit @ factor.T + factor @ unit.T
        return out

    def reported(self, matrix: Tensor) -> Tensor:
        return torch.stack([matrix[a, b] for a, b in self.entries])

    def reported_jacobian(self, theta: Tensor) -> Tensor:
        """d reported / d theta, [r, m]."""
        jac = self.jacobian(theta)
        return torch.stack([jac[:, a, b] for a, b in self.entries])

    def scaled(self, theta: Tensor, factor: Tensor) -> Tensor:
        """theta of the matrix G(theta) * factor (factor > 0)."""
        out = theta.clone()
        half = 0.5 * torch.log(factor)
        if self.kind == "exchangeable":
            out[0] = out[0] + half
        elif self.kind == "unstructured":
            out[:self.q] = out[:self.q] + half
            out[self.q:] = out[self.q:] * torch.sqrt(factor)
        else:
            out = out + half
        return out

    def start(self, diagonal: Tensor) -> Tensor:
        """theta of (approximately) G = diag(diagonal)."""
        logs = 0.5 * torch.log(diagonal)
        if self.kind == "identity":
            return logs.mean().reshape(1)
        if self.kind == "independent":
            return logs
        if self.kind == "exchangeable":
            return torch.stack([logs.mean(), torch.zeros((), dtype=torch.float64)])
        return torch.cat([logs, torch.zeros(len(self.lower), dtype=torch.float64)])


def _factorize(m: Tensor) -> Tensor:
    """Batched Cholesky factor of the [C, q, q] matrices M_c (a square root when q = 1)."""
    if m.shape[-1] == 1:
        if not bool((m > 0).all()):
            raise KernelError("numerical_failure", "A class covariance factor is not positive "
                              "definite.")
        return m.sqrt()
    chol, info = torch.linalg.cholesky_ex(m)
    if bool((info != 0).any()):
        raise KernelError("numerical_failure", "A class covariance factor is not positive "
                          "definite.")
    return chol


def _solve(rhs: Tensor, chol: Tensor) -> Tensor:
    """M^-1 rhs from the batched factor (elementwise when q = 1)."""
    if chol.shape[-1] == 1:
        return rhs / chol.square()
    return torch.cholesky_solve(rhs, chol)


@dataclass
class MixedEvaluation:
    value: Tensor
    beta: Tensor
    xvx: Tensor               # X'V~^-1 X (relative scale; X'V^-1 X = xvx / s2)
    rvr: Tensor               # r'V~^-1 r
    sigma2: Tensor
    gradient: Tensor | None = None
    low_blup: Tensor | None = None     # [C, q] BLUPs of u_c
    top_blup: Tensor | None = None     # [S] BLUPs of v_s


class MixedLikelihood:
    """(Restricted) log likelihood of the linear mixed model in theta = (t_G, [t_g], ln s).

    ``low_codes`` [N] are dense class codes 0..C-1; ``top_of_low`` [C] maps each class
    to its top-level group (two-level models) or is None. ``theta`` holds the
    structure parameters of G, then (two levels) ln sqrt(g), then ln sigma_e.
    """

    def __init__(self, x: Tensor, y: Tensor, z: Tensor, low_codes: Tensor, n_low: int,
                 structure: CovStructure, *, top_of_low: Tensor | None = None, n_top: int = 0,
                 frequency: Tensor | None = None, reml: bool = False):
        n, p = x.shape
        q = z.shape[1]
        if q != structure.q or y.shape != (n,) or z.shape[0] != n or low_codes.shape != (n,):
            raise KernelError("invalid_design", "The mixed-model inputs do not conform.")
        self.p, self.q, self.structure, self.reml = p, q, structure, reml
        self.n_low, self.n_top, self.top = n_low, n_top, top_of_low
        self.two_level = top_of_low is not None
        weights = torch.ones(n, dtype=torch.float64) if frequency is None else frequency
        self.nobs = float(weights.sum())
        data = torch.cat([x, y[:, None]], dim=1)                     # [N, p + 1]
        self.cross = weighted_crossprod(data, frequency)            # [p + 1, p + 1]
        zw = z * weights[:, None]
        self.ztz = torch.stack([group_sums(zw[:, a:a + 1] * z, low_codes, n_low)
                                for a in range(q)], dim=1)          # [C, q, q]
        self.ztz = (self.ztz + self.ztz.transpose(1, 2)) / 2
        self.zd = torch.stack([group_sums(zw[:, a:a + 1] * data, low_codes, n_low)
                               for a in range(q)], dim=1)           # [C, q, p + 1]
        if self.two_level:
            ones = group_sums(zw, low_codes, n_low)                 # Z'1  [C, q]
            self.zd = torch.cat([ones[:, :, None], self.zd], dim=2)  # [C, q, 1 + p + 1]
            self.ones_d = group_sums(weights[:, None] * data, low_codes, n_low)   # [C, p + 1]
            self.sizes = group_sums(weights, low_codes, n_low)       # [C]
        self.size = structure.size + int(self.two_level) + 1
        self.eye = torch.eye(q, dtype=torch.float64)

    # ---- evaluation --------------------------------------------------------------

    def _split(self, theta: Tensor) -> tuple[Tensor, Tensor | None, Tensor]:
        m = self.structure.size
        top = theta[m] if self.two_level else None
        return theta[:m], top, theta[-1]

    def evaluate(self, theta: Tensor, *, gradient: bool = False, blups: bool = False,
                 sigma2: Tensor | None = None, beta: Tensor | None = None) -> MixedEvaluation:
        """Value (and gradient) at theta; ``sigma2`` overrides exp(2 ln s) (profiling).

        ``beta`` fixes the fixed effects instead of profiling them (predictions on
        new data use the estimated coefficients); the gradient is then not valid.
        """
        p = self.p
        t_g, t_top, ln_sigma = self._split(theta)
        s2 = torch.exp(2 * ln_sigma) if sigma2 is None else sigma2
        factor = self.structure.factor(t_g) / torch.sqrt(s2)             # D = L L'
        lt_ztz = factor.T @ self.ztz                                     # [C, q, q]
        m = self.eye + lt_ztz @ factor
        chol = _factorize(m)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum()
        lb = factor.T @ self.zd                                          # [C, q, m']
        solved = _solve(lb, chol)                                        # M^-1 L' B
        u = slice(1, None) if self.two_level else slice(None)
        q_cross = self.cross - torch.einsum("cqa,cqb->ab", lb[:, :, u], solved[:, :, u])
        if self.two_level:
            d = torch.exp(2 * t_top) / s2
            lb0, s0 = lb[:, :, 0], solved[:, :, 0]
            a_c = self.sizes - (lb0 * s0).sum(dim=1)
            b_c = self.ones_d - torch.einsum("cq,cqm->cm", lb0, solved[:, :, 1:])
            a_s = group_sums(a_c, self.top, self.n_top)
            b_s = group_sums(b_c, self.top, self.n_top)
            denom = 1 + d * a_s
            kappa = d / denom
            q_cross = q_cross - torch.einsum("s,sa,sb->ab", kappa, b_s, b_s)
            logdet = logdet + torch.log(denom).sum()
        q_cross = (q_cross + q_cross.T) / 2
        xvx, xvy, yvy = q_cross[:p, :p], q_cross[:p, p], q_cross[p, p]
        xchol = None
        if beta is None or self.reml:
            xchol, info = torch.linalg.cholesky_ex(xvx)
            if int(info) != 0:
                raise KernelError("singular_design", "X'V^-1 X is not positive definite.")
        if beta is None:
            beta = torch.cholesky_solve(xvy[:, None], xchol)[:, 0]
            rvr = yvy - xvy @ beta
        else:
            rvr = yvy - 2 * xvy @ beta + beta @ xvx @ beta
        if not bool(rvr > 0):
            raise KernelError("perfect_fit", "The residual quadratic form is not positive.")
        p_r = p if self.reml else 0
        value = (-(self.nobs - p_r) / 2 * (_LOG_2PI + torch.log(s2)) - logdet / 2
                 - rvr / (2 * s2))
        if self.reml:
            value = value - torch.log(torch.diagonal(xchol)).sum()
        result = MixedEvaluation(value, beta, xvx, rvr, s2)
        if not (gradient or blups):
            return result
        # Z'A^-1 [cols] = B - Z'Z L M^-1 L' B, then the top-level rank-one update.
        t = self.zd - self.ztz @ (factor @ solved)
        if self.two_level:
            t1, tu = t[:, :, 0], t[:, :, 1:]
            kc = kappa[self.top]
            zvu = tu - kc[:, None, None] * t1[:, :, None] * b_s[self.top][:, None, :]
        else:
            zvu = t
        zr = zvu[:, :, p] - zvu[:, :, :p] @ beta                       # Z'V~^-1 r  [C, q]
        if blups:
            # E[u | y] = G Z'V^-1 r = D Z'V~^-1 r;  E[v | y] = g 1'V^-1 r = d b_r/(1 + d a).
            result.low_blup = zr @ (factor @ factor.T)
            if self.two_level:
                result.top_blup = d * (b_s[:, p] - b_s[:, :p] @ beta) / denom
        if not gradient:
            return result
        w_solved = _solve(lt_ztz, chol)                                  # M^-1 L'Z'Z
        zvz = self.ztz - lt_ztz.transpose(1, 2) @ w_solved
        if self.two_level:
            zvz = zvz - kc[:, None, None] * t1[:, :, None] * t1[:, None, :]
        grad_d = -0.5 * zvz.sum(dim=0) + zr.T @ zr / (2 * s2)
        if self.reml:
            cx = torch.cholesky_inverse(xchol)
            zx = zvu[:, :, :p]
            grad_d = grad_d + 0.5 * torch.einsum("cap,pr,cbr->ab", zx, cx, zx)
        grad_d = (grad_d + grad_d.T) / 2
        delta = factor @ factor.T
        trace = (grad_d * delta).sum()
        pieces = []
        grad_g = grad_d / s2
        pieces.append(torch.einsum("mab,ab->m", self.structure.jacobian(t_g), grad_g))
        if self.two_level:
            b_r = b_s[:, p] - b_s[:, :p] @ beta
            grad_top = (-0.5 * a_s / denom + b_r.square() / (2 * s2 * denom.square())).sum()
            if self.reml:
                b_x = b_s[:, :p]
                quad = ((b_x @ cx) * b_x).sum(dim=1)
                grad_top = grad_top + 0.5 * (quad / denom.square()).sum()
            trace = trace + d * grad_top
            g_top = torch.exp(2 * t_top)
            pieces.append((2 * g_top * grad_top / s2).reshape(1))
        grad_s2 = -(self.nobs - p_r) / (2 * s2) + rvr / (2 * s2.square()) - trace / s2
        pieces.append((2 * s2 * grad_s2).reshape(1))
        result.gradient = torch.cat(pieces)
        return result

    def value(self, theta: Tensor) -> Tensor:
        try:
            return self.evaluate(theta).value
        except KernelError:
            return torch.tensor(-math.inf, dtype=torch.float64)

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        try:
            out = self.evaluate(theta, gradient=True)
        except KernelError:
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(self.size, dtype=torch.float64))
        return out.value, out.gradient

    def gradient(self, theta: Tensor) -> Tensor:
        return self.evaluate(theta, gradient=True).gradient

    def profiled_sigma2(self, theta: Tensor) -> Tensor:
        """s2 maximizing the likelihood at the relative parameters implied by theta.

        The relative parameters are D = G(theta)/exp(2 ln s); the result is
        r'V~^-1 r / (N - p_R), the closed-form maximizer.
        """
        out = self.evaluate(theta)
        return out.rvr / (self.nobs - (self.p if self.reml else 0))

    def group_scores(self, x: Tensor, y: Tensor, z: Tensor, low_codes: Tensor, theta: Tensor,
                     frequency: Tensor | None = None) -> Tensor:
        """Per-group scores of the full ML log likelihood in (b, theta), [groups, p + size].

        The groups are the top level (the classes of a one-level model); b is the GLS
        estimate at theta. Columns: d l_s/db = X_s'V_s^-1 r_s, then the partial
        derivatives with respect to theta at fixed b, which are the group terms of the
        analytic gradient (module notes): with A_c = I + Z_c D Z_c' and the top-level
        rank-one update kappa_s = d/(1 + d a_s),

            d l_s/dD = sum_c [-1/2 Z_c'V~^-1 Z_c + (Z_c'V~^-1 r)(Z_c'V~^-1 r)'/(2 s2)]
            d l_s/dd = -1/2 a_s/(1 + d a_s) + b_r,s^2 / (2 s2 (1 + d a_s)^2)
            d l_s/ds2 = -n_s/(2 s2) + r_s'V~_s^-1 r_s/(2 s2^2)          (D, d fixed)

        chained to (G, g, s2) and theta as in ``evaluate``. They sum to the gradient of
        the profiled likelihood (envelope theorem). The restricted likelihood does not
        separate by groups, so REML has no group scores (Stata: no robust VCE with REML).
        Every per-observation quantity is one ``index_add_`` over the rows.
        """
        if self.reml:
            raise KernelError("unsupported_covariance", "The restricted likelihood has no "
                              "group scores; robust covariances need method='ml'.")
        p, q, n_low = self.p, self.q, self.n_low
        t_g, t_top, ln_sigma = self._split(theta)
        s2 = torch.exp(2 * ln_sigma)
        beta = self.evaluate(theta).beta
        weights = torch.ones(len(y), dtype=torch.float64) if frequency is None else frequency
        resid = y - x @ beta
        wr = weights * resid
        zr = group_sums(z * wr[:, None], low_codes, n_low)                  # Z'r  [C, q]
        xr = group_sums(x * wr[:, None], low_codes, n_low)                  # X'r  [C, p]
        rr = group_sums(wr * resid, low_codes, n_low)                       # r'r  [C]
        offset = 1 if self.two_level else 0
        zx = self.zd[:, :, offset:offset + p]                               # Z'X  [C, q, p]
        factor = self.structure.factor(t_g) / torch.sqrt(s2)                # D = L L'
        chol = _factorize(self.eye + factor.T @ self.ztz @ factor)
        columns = [zr[:, :, None], zx, self.ztz]
        if self.two_level:
            columns.append(self.zd[:, :, :1])                               # Z'1
        lb = factor.T @ torch.cat(columns, dim=2)                           # L'Z'[r X Z (1)]
        solved = _solve(lb, chol)                                           # M^-1 L'Z'[...]
        s_r = solved[:, :, 0]
        ztz_l = self.ztz @ factor
        # u'A^-1 v = u'v - (L'Z'u)' M^-1 (L'Z'v);  Z'A^-1 v = Z'v - Z'Z L M^-1 L'Z'v.
        r_a_r = rr - (lb[:, :, 0] * s_r).sum(dim=1)
        x_a_r = xr - torch.einsum("cqp,cq->cp", lb[:, :, 1:1 + p], s_r)
        z_a_r = zr - (ztz_l @ s_r[:, :, None])[:, :, 0]
        z_a_z = self.ztz - ztz_l @ solved[:, :, 1 + p:1 + p + q]
        if self.two_level:
            top, n_top = self.top, self.n_top
            l1, s1 = lb[:, :, -1], solved[:, :, -1]
            a_c = self.sizes - (l1 * s1).sum(dim=1)                         # 1'A^-1 1
            b_r = group_sums(wr, low_codes, n_low) - (l1 * s_r).sum(dim=1)   # 1'A^-1 r
            b_x = self.ones_d[:, :p] - torch.einsum("cq,cqp->cp", l1, solved[:, :, 1:1 + p])
            t1 = self.zd[:, :, 0] - (ztz_l @ s1[:, :, None])[:, :, 0]       # Z'A^-1 1
            a_s = group_sums(a_c, top, n_top)
            br_s = group_sums(b_r, top, n_top)
            d = torch.exp(2 * t_top) / s2
            denom = 1 + d * a_s
            kappa = d / denom
            k_c = kappa[top]
            z_v_r = z_a_r - (k_c * br_s[top])[:, None] * t1
            z_v_z = z_a_z - k_c[:, None, None] * t1[:, :, None] * t1[:, None, :]
            x_v_r = (group_sums(x_a_r, top, n_top)
                     - (kappa * br_s)[:, None] * group_sums(b_x, top, n_top))
            r_v_r = group_sums(r_a_r, top, n_top) - kappa * br_s.square()
            n_s = group_sums(self.sizes, top, n_top)
            per_class = -0.5 * z_v_z + z_v_r[:, :, None] * z_v_r[:, None, :] / (2 * s2)
            grad_d = group_sums(per_class.reshape(n_low, q * q), top, n_top).reshape(-1, q, q)
        else:
            x_v_r, r_v_r = x_a_r, r_a_r
            n_s = group_sums(weights, low_codes, n_low)
            grad_d = -0.5 * z_a_z + z_a_r[:, :, None] * z_a_r[:, None, :] / (2 * s2)
        trace = (grad_d * (factor @ factor.T)).sum(dim=(1, 2))
        pieces = [x_v_r / s2,
                  torch.einsum("mab,sab->sm", self.structure.jacobian(t_g), grad_d / s2)]
        if self.two_level:
            grad_top = -0.5 * a_s / denom + br_s.square() / (2 * s2 * denom.square())
            trace = trace + d * grad_top
            pieces.append((2 * torch.exp(2 * t_top) * grad_top / s2)[:, None])
        grad_s2 = -n_s / (2 * s2) + r_v_r / (2 * s2.square()) - trace / s2
        pieces.append((2 * s2 * grad_s2)[:, None])
        return torch.cat(pieces, dim=1)
