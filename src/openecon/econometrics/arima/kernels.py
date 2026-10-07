"""Gaussian likelihood of a regression with multiplicative seasonal ARMA errors.

Model (after any differencing, done by the caller):

    y_t = x_t'b + u_t,     phi(L) Phi(L^s) u_t = theta(L) Theta(L^s) e_t,     e_t ~ N(0, sigma^2)

with phi(L) = 1 - phi_1 L - ..., theta(L) = 1 + theta_1 L + ... (Stata's signs) and
the expanded polynomials phi*(L) = phi(L) Phi(L^s), theta*(L) = theta(L) Theta(L^s)
of degrees p* and q*.

Conditional residuals. Setting presample u and e to zero (their expectation), the
disturbances are the output of a linear filter,

    v~ = phi*(L) u / theta*(L),

computed for all n observations by ``filters.inverse_filter`` (no time loop). The
conditional (CSS) log likelihood, Stata's ``arima, condition``, is

    ll_c = -n/2 ln(2 pi sigma^2) - v~'v~ / (2 sigma^2).

Exact likelihood. In Harvey's state-space form (state dimension r = max(p*, q* + 1),
transition T with phi* in its first column and ones above the diagonal,
R = (1, theta*_1, ..., theta*_(r-1))') write the state as alpha_t = x_t + R e_t with
x_t = T alpha_(t-1). Then y_t = x_t[0] + e_t and x_(t+1) = T x_t + T R e_t hold for
every t, so given the presample state x_1 the disturbances are

    e = v~ - G x_1,       G[t, j] = psi_(t-j),     psi(L) = 1 / theta*(L),

and x_1 ~ N(0, sigma^2 S_x) is independent of e, with S_x = T P T' = P - R R' and P
the stationary state covariance (P = T P T' + R R', solved by doubling). Integrating
x_1 out of the Gaussian density gives the exact likelihood of the stationary process,
the same number the Kalman filter with the unconditional initial state produces:

    ll = -n/2 ln(2 pi sigma^2) - D/2 - S / (2 sigma^2)
    D  = ln det(I + S_x M),            M = G'G
    S  = v~'v~ - b' W b,               b = G'v~,   W = (S_x^-1 + M)^-1 = C (I + C'MC)^-1 C'

where S_x = C C'. Because psi decays geometrically for an invertible MA part, G
has only as many non-negligible rows as the impulse response is long; for a pure
AR model it is the first r rows of the identity. The cost is O(n) filter passes
plus r-by-r algebra, for any n. The formula holds for any MA polynomial; it is
evaluated only where the MA part is invertible or marginally non-invertible
(largest inverse root rho with rho^n <= 10), which loses nothing because every
non-invertible MA polynomial has an invertible twin with the same likelihood.
The exact likelihood also needs a stationary AR part; other points are infeasible
(the objective returns -inf and the optimizer rejects the step).

Derivatives are analytic. With z = v~ / theta(L) etc. each derivative of v~ is a
shifted filtered series (d v~/d theta_i = -L^i v~/theta(L), d v~/d phi_i = -L^i
Phi(L^s) u / theta*(L), d v~/d b_j = -phi*(L) x_j / theta*(L), ...); dG follows from
d psi/d theta_i = -L^i psi/theta(L); dS_x = dP - d(RR') with dP the solution of the
Lyapunov equation whose right-hand side is dT P T' + T P dT' + d(RR'). Then

    dD = tr(dS_x N) + tr(W dM),            N = M - M W M
    dS = 2 v~'dv~ - 2 m'db - c'dS_x c + m'dM m,      m = W b,  c = b - M m.

The prediction-error decomposition (innovations v_t, variance ratios F_t and the
per-observation scores needed by the OPG and robust covariances) follows from the
same representation: given y_1..y_(t-1) the presample state is N(m_t, sigma^2 W_t)
with W_t = (S_x^-1 + M_t)^-1, M_t and b_t the partial sums over s < t, so

    v_t = v~_t - g_t m_t,      F_t = 1 + g_t W_t g_t',
    ll_t = -1/2 ln(2 pi sigma^2 F_t) - v_t^2 / (2 sigma^2 F_t)

for the rows where g_t is not negligible, and v_t = v~_t, F_t = 1 afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.econometrics.arima.filters import (
    apply_polynomial, inverse_filter, lag_polynomial, levinson, lyapunov, multiply,
    spectral_radius, transition_powers,
)
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares

_LOG_2PI = math.log(2.0 * math.pi)
_DECAYED = 1e-17          # relative size below which the impulse response is treated as zero
_GROWTH_LIMIT = math.log(10.0)   # largest n * ln(rho) of a marginally non-invertible MA part
_RANK_TOL = 1e-14
_CHUNK_ELEMENTS = 2_000_000


@dataclass
class Pieces:
    """Sufficient statistics of one likelihood evaluation."""

    ss: float                     # S: the (corrected) sum of squares
    logdet: float                 # D: ln det(I + S_x M); 0 for the conditional likelihood
    v: Tensor                     # conditional residuals v~ [n]
    d_ss: Tensor | None = None    # [K] derivative of S
    d_logdet: Tensor | None = None
    work: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decomposition:
    """Prediction-error decomposition at given parameters."""

    innovations: Tensor           # v_t [n]
    variance_ratio: Tensor        # F_t [n]; Var(v_t) = sigma^2 F_t
    scores: Tensor | None         # [n, K + 1] per-observation scores (last column: sigma)
    disturbances: Tensor          # E[e_t | y] for the last q* periods (oldest first)
    disturbance_covariance: Tensor  # their covariance / sigma^2, [q*, q*]


def _lagged(sequence: Tensor, columns: int, length: int, offset: int = 0) -> Tensor:
    """[length, columns] matrix with entry (t, j) = sequence[t - j - offset], zero elsewhere."""
    out = torch.zeros((length, columns), dtype=torch.float64)
    for j in range(columns):
        start = j + offset
        if start >= length:
            break
        count = min(sequence.shape[0], length - start)
        out[start:start + count, j] = sequence[:count]
    return out


class ArmaLikelihood:
    """Exact or conditional Gaussian likelihood of regression with (seasonal) ARMA errors.

    ``y`` [n] and ``x`` [n, k] are float64 (already differenced). The parameter
    vector is ``psi = (b [k], phi [p], theta [q], Phi [P], Theta [Q])``;
    ``full`` appends sigma. ``exact=False`` gives the conditional likelihood.
    """

    def __init__(self, y: Tensor, x: Tensor, *, p: int = 0, q: int = 0, seasonal_p: int = 0,
                 seasonal_q: int = 0, period: int = 0, exact: bool = True):
        if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.shape[0] or y.dtype != torch.float64 \
                or x.dtype != torch.float64:
            raise KernelError("invalid_design", "ARMA likelihood needs float64 y [n] and x [n, k].")
        if (seasonal_p or seasonal_q) and period < 2:
            raise KernelError("invalid_spec", "Seasonal ARMA terms need a period of at least 2.")
        self.y, self.xt = y, x.T.contiguous()
        self.n, self.k = int(y.shape[0]), int(x.shape[1])
        self.sizes = (p, q, seasonal_p, seasonal_q)
        self.period = max(int(period), 1)
        self.exact = bool(exact)
        self.n_arma = p + q + seasonal_p + seasonal_q
        self.n_psi = self.k + self.n_arma
        edges = [self.k]
        for size in self.sizes:
            edges.append(edges[-1] + size)
        self.slices = {name: slice(edges[i], edges[i + 1])
                       for i, name in enumerate(("ar", "ma", "sar", "sma"))}
        self.ma_index = [*range(edges[1], edges[2]), *range(edges[3], edges[4])]
        self.p_full = p + seasonal_p * self.period
        self.q_full = q + seasonal_q * self.period
        self.r = max(self.p_full, self.q_full + 1)

    # ---- polynomials and the state-space form ------------------------------------

    def split(self, psi: Tensor) -> tuple[list[float], list[float], list[float], list[float]]:
        values = psi.tolist()
        return tuple(values[self.slices[name]] for name in ("ar", "ma", "sar", "sma"))

    def feasible(self, phi: list[float], theta: list[float], sphi: list[float],
                 stheta: list[float]) -> bool:
        """MA part at most marginally non-invertible; AR part stationary for the exact form."""
        radius = spectral_radius([-value for value in theta])
        if radius > 1.0 and self.n * math.log(radius) > _GROWTH_LIMIT:
            return False
        radius = spectral_radius([-value for value in stheta])
        if radius > 1.0 and self.n / self.period * math.log(radius) > _GROWTH_LIMIT:
            return False
        if self.exact and (spectral_radius(phi) >= 1.0 or spectral_radius(sphi) >= 1.0):
            return False
        return True

    def _state(self, phi: list[float], theta: list[float], sphi: list[float],
               stheta: list[float]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """T, R and the derivatives of T's first column and of R by ARMA parameter."""
        s, r = self.period, self.r
        ar, sar = lag_polynomial(phi, sign=-1.0), lag_polynomial(sphi, unit=s, sign=-1.0)
        ma, sma = lag_polynomial(theta), lag_polynomial(stheta, unit=s)
        ar_full, ma_full = multiply(ar, sar), multiply(ma, sma)
        transition = torch.zeros((r, r), dtype=torch.float64)
        if self.p_full:
            transition[:self.p_full, 0] = -torch.tensor(ar_full[1:], dtype=torch.float64)
        if r > 1:
            transition[:-1, 1:] += torch.eye(r - 1, dtype=torch.float64)
        loading = torch.zeros(r, dtype=torch.float64)
        loading[0] = 1.0
        if self.q_full:
            loading[1:self.q_full + 1] = torch.tensor(ma_full[1:], dtype=torch.float64)
        a_cols = [[0.0] * r for _ in range(self.n_arma)]
        c_cols = [[0.0] * r for _ in range(self.n_arma)]
        row = 0
        for i in range(1, len(phi) + 1):            # d phi*_m / d phi_i = Phi-polynomial[m - i]
            for j, value in enumerate(sar):
                a_cols[row][i + j - 1] = value
            row += 1
        for i in range(1, len(theta) + 1):          # d theta*_m / d theta_i
            for j, value in enumerate(sma):
                c_cols[row][i + j] = value
            row += 1
        for j in range(1, len(sphi) + 1):           # d phi*_m / d Phi_j = phi-polynomial[m - j s]
            for i, value in enumerate(ar):
                a_cols[row][j * s + i - 1] = value
            row += 1
        for j in range(1, len(stheta) + 1):
            for i, value in enumerate(ma):
                c_cols[row][j * s + i] = value
            row += 1
        shape = (self.n_arma, r)
        return (transition, loading,
                torch.tensor(a_cols, dtype=torch.float64).reshape(shape),
                torch.tensor(c_cols, dtype=torch.float64).reshape(shape))

    def _impulse(self, theta: list[float], stheta: list[float]) -> tuple[Tensor, Tensor | None,
                                                                         Tensor | None]:
        """psi = 1/theta*(L) and psi/theta(L), psi/Theta(L^s), cut where they have decayed."""
        if not self.q_full:
            return torch.ones(1, dtype=torch.float64), None, None
        n, s = self.n, self.period
        length = min(n, max(256, 8 * (self.q_full + self.r)))
        while True:
            unit = torch.zeros((1, length), dtype=torch.float64)
            unit[0, 0] = 1.0
            psi = inverse_filter(inverse_filter(unit, theta), stheta, s)
            rows = [psi, inverse_filter(psi, theta) if theta else None,
                    inverse_filter(psi, stheta, s) if stheta else None]
            stack = torch.cat([row for row in rows if row is not None]).abs()
            peak = stack.max(dim=1, keepdim=True).values
            if length == n or bool((stack[:, -self.q_full:] <= _DECAYED * peak).all()):
                break
            length = min(n, 8 * length)
        significant = (stack > _DECAYED * peak).any(dim=0).nonzero()
        keep = int(significant[-1]) + 1 if len(significant) else 1
        return tuple(None if row is None else row[0, :keep] for row in rows)

    # ---- one evaluation ----------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, psi: Tensor, *, derivatives: bool = True) -> Pieces | None:
        """S, D, the conditional residuals and (optionally) dS, dD; None if infeasible."""
        k, n, s = self.k, self.n, self.period
        if not bool(torch.isfinite(psi).all()):
            return None
        phi, theta, sphi, stheta = self.split(psi)
        if not self.feasible(phi, theta, sphi, stheta):
            return None
        ar, sar = lag_polynomial(phi, sign=-1.0), lag_polynomial(sphi, unit=s, sign=-1.0)
        u = self.y - psi[:k] @ self.xt if k else self.y
        head = torch.cat([u[None], self.xt]) if derivatives and k else u[None]
        rows = [apply_polynomial(apply_polynomial(head, ar), sar)]
        if derivatives and phi:
            rows.append(apply_polynomial(u[None], sar))
        if derivatives and sphi:
            rows.append(apply_polynomial(u[None], ar))
        batch = inverse_filter(inverse_filter(torch.cat(rows), theta), stheta, s)
        v = batch[0]
        ss = float(torch.dot(v, v))
        if not math.isfinite(ss):
            return None
        pieces = Pieces(ss=ss, logdet=0.0, v=v)
        if derivatives:
            regressors = batch[1:1 + k]
            sources: list[tuple[Tensor, int]] = []   # d v~/d psi_j = -L^lag z for ARMA terms
            row = 1 + k
            if phi:
                sources += [(batch[row], i) for i in range(1, len(phi) + 1)]
                row += 1
            if theta:
                z = inverse_filter(v[None], theta)[0]
                sources += [(z, i) for i in range(1, len(theta) + 1)]
            if sphi:
                sources += [(batch[row], j * s) for j in range(1, len(sphi) + 1)]
            if stheta:
                z = inverse_filter(v[None], stheta, s)[0]
                sources += [(z, j * s) for j in range(1, len(stheta) + 1)]
            d_ss = torch.zeros(self.n_psi, dtype=torch.float64)
            if k:
                d_ss[:k] = -2.0 * (regressors @ v)
            for j, (z, lag) in enumerate(sources):
                if lag < n:
                    d_ss[k + j] = -2.0 * torch.dot(v[lag:], z[:n - lag])
            pieces.d_ss = d_ss
            pieces.d_logdet = torch.zeros(self.n_psi, dtype=torch.float64)
            pieces.work.update(regressors=regressors, sources=sources)
        if self.exact and not self._exact(pieces, (phi, theta, sphi, stheta), derivatives):
            return None
        return pieces

    def _head(self, work: dict[str, Any], length: int) -> Tensor:
        """[K, length] matrix of d v~_t / d psi for the first ``length`` periods."""
        out = torch.zeros((self.n_psi, length), dtype=torch.float64)
        if self.k:
            out[:self.k] = -work["regressors"][:, :length]
        for j, (z, lag) in enumerate(work["sources"]):
            if lag < length:
                out[self.k + j, lag:] = -z[:length - lag]
        return out

    def _exact(self, pieces: Pieces, polynomials: tuple, derivatives: bool) -> bool:
        """Add the presample-state correction to S and D (and their derivatives)."""
        phi, theta, sphi, stheta = polynomials
        n, k, r, s = self.n, self.k, self.r, self.period
        transition, loading, a_cols, c_cols = self._state(phi, theta, sphi, stheta)
        powers = transition_powers(transition)
        if powers is None:
            return False
        outer = torch.outer(loading, loading)
        state_cov = lyapunov(powers, outer)
        sx = state_cov - outer
        sx = (sx + sx.T) / 2
        if not bool(torch.isfinite(sx).all()):
            return False
        psi_imp, zeta, zeta_s = self._impulse(theta, stheta)
        lg = min(n, psi_imp.shape[0] + r - 1)
        g = _lagged(psi_imp, r, lg)
        v = pieces.v
        m_matrix = g.T @ g
        b = g.T @ v[:lg]
        values, vectors = torch.linalg.eigh(sx)
        keep = values > max(float(values[-1]), 0.0) * _RANK_TOL
        c_factor = vectors[:, keep] * values[keep].sqrt()
        rank = int(keep.sum())
        if rank:
            inner = torch.eye(rank, dtype=torch.float64) + c_factor.T @ m_matrix @ c_factor
            chol, info = torch.linalg.cholesky_ex(inner)
            if int(info) != 0:
                return False
            w = c_factor @ torch.cholesky_solve(c_factor.T, chol)
            pieces.logdet = 2.0 * float(chol.diagonal().log().sum())
        else:
            w = torch.zeros((r, r), dtype=torch.float64)
        m = w @ b
        pieces.ss -= float(torch.dot(b, m))
        if not (math.isfinite(pieces.ss) and math.isfinite(pieces.logdet)) or pieces.ss <= 0.0:
            return False
        pieces.work.update(g=g, lg=lg, c_factor=c_factor, rank=rank, w=w, m=m)
        if not derivatives:
            return True
        db = self._head(pieces.work, lg) @ g                      # G' d v~ per parameter
        n_ma = len(self.ma_index)
        d_g = torch.zeros((n_ma, lg, r), dtype=torch.float64)
        d_m = torch.zeros((n_ma, r, r), dtype=torch.float64)
        lags = [(zeta, i) for i in range(1, len(theta) + 1)] \
            + [(zeta_s, j * s) for j in range(1, len(stheta) + 1)]
        for j, (z, lag) in enumerate(lags):
            d_g[j] = -_lagged(z, r, lg, lag)
            cross = d_g[j].T @ g
            d_m[j] = cross + cross.T
            db[self.ma_index[j]] += d_g[j].T @ v[:lg]
        c = b - m_matrix @ m
        n_matrix = m_matrix - m_matrix @ w @ m_matrix
        if self.n_arma:
            lead = transition @ state_cov[:, 0]
            rhs_t = a_cols[:, :, None] * lead[None, None, :]
            rhs_r = c_cols[:, :, None] * loading[None, None, :]
            rhs_r = rhs_r + rhs_r.transpose(1, 2)
            d_sx = lyapunov(powers, rhs_t + rhs_t.transpose(1, 2) + rhs_r) - rhs_r
        else:
            d_sx = torch.zeros((0, r, r), dtype=torch.float64)
        pieces.d_ss -= 2.0 * (db @ m)
        pieces.d_ss[k:] -= torch.einsum("r,krs,s->k", c, d_sx, c)
        pieces.d_logdet[k:] = (d_sx * n_matrix).sum(dim=(1, 2))
        if n_ma:
            pieces.d_ss[self.ma_index] += torch.einsum("r,krs,s->k", m, d_m, m)
            pieces.d_logdet[self.ma_index] += (d_m * w).sum(dim=(1, 2))
        pieces.work.update(d_g=d_g, d_sx=d_sx)
        return True

    # ---- objectives ------------------------------------------------------------------------

    def _infeasible(self, size: int) -> tuple[float, Tensor]:
        return -math.inf, torch.full((size,), math.nan, dtype=torch.float64)

    def concentrated(self, psi: Tensor) -> tuple[float, Tensor]:
        """Log likelihood with sigma^2 = S/n concentrated out, and its gradient in psi."""
        pieces = self.evaluate(psi)
        if pieces is None or not pieces.ss > 0.0:
            return self._infeasible(self.n_psi)
        n = self.n
        value = -0.5 * n * (_LOG_2PI + 1.0 + math.log(pieces.ss / n)) - 0.5 * pieces.logdet
        return value, -(0.5 * n / pieces.ss) * pieces.d_ss - 0.5 * pieces.d_logdet

    def full(self, theta: Tensor) -> tuple[float, Tensor]:
        """Log likelihood in (psi, sigma) and its gradient."""
        sigma = float(theta[-1])
        pieces = self.evaluate(theta[:-1]) if sigma > 0.0 and math.isfinite(sigma) else None
        if pieces is None or not pieces.ss > 0.0:
            return self._infeasible(self.n_psi + 1)
        n, s2 = self.n, sigma * sigma
        value = -0.5 * n * _LOG_2PI - n * math.log(sigma) - 0.5 * pieces.logdet \
            - pieces.ss / (2.0 * s2)
        gradient = torch.empty(self.n_psi + 1, dtype=torch.float64)
        gradient[:-1] = -0.5 * pieces.d_logdet - pieces.d_ss / (2.0 * s2)
        gradient[-1] = -n / sigma + pieces.ss / (s2 * sigma)
        return value, gradient

    def full_hessian(self, psi: Tensor, concentrated_hessian: Tensor) -> Tensor:
        """Hessian in (psi, sigma) at the maximum from the Hessian of the concentrated form.

        With ll = c - n ln sigma - D/2 - S/(2 sigma^2) and sigma^2 = S/n at the maximum,
        H_ss = -2n/sigma^2, H_ps = S'/sigma^3, and the concentrated Hessian is the Schur
        complement H_pp - H_ps H_ss^-1 H_sp, so H_pp = H_c - S'S''/(2 n sigma^4).
        """
        pieces = self.evaluate(psi)
        if pieces is None:
            raise KernelError("numerical_failure",
                              "The likelihood is not defined at the estimates.")
        n, k = self.n, self.n_psi
        s2 = pieces.ss / n
        sigma = math.sqrt(s2)
        hessian = torch.empty((k + 1, k + 1), dtype=torch.float64)
        hessian[:k, :k] = concentrated_hessian \
            - torch.outer(pieces.d_ss, pieces.d_ss) / (2.0 * n * s2 * s2)
        hessian[:k, k] = hessian[k, :k] = pieces.d_ss / (s2 * sigma)
        hessian[k, k] = -2.0 * n / s2
        return hessian

    @torch.no_grad()
    def gauss_newton(self, psi: Tensor) -> Tensor | None:
        """(n/S) J J' with J = d v~/d psi: the Gauss-Newton information of the concentrated form."""
        pieces = self.evaluate(psi)
        if pieces is None:
            return None
        jacobian = self._head(pieces.work, self.n)
        return (self.n / pieces.ss) * (jacobian @ jacobian.T)

    # ---- prediction-error decomposition and scores --------------------------------------

    @torch.no_grad()
    def decompose(self, theta: Tensor, *, scores: bool = True) -> Decomposition:
        """Innovations, their variance ratios, per-observation scores and the end-of-sample
        disturbance estimates at ``theta = (psi, sigma)`` (see the module docstring)."""
        sigma = float(theta[-1])
        pieces = self.evaluate(theta[:-1], derivatives=scores)
        if pieces is None or not sigma > 0.0:
            raise KernelError("numerical_failure",
                              "The likelihood is not defined at the estimates.")
        n, k, kk, r = self.n, self.k, self.n_psi, self.r
        work = pieces.work
        v = pieces.v.clone()
        f = torch.ones(n, dtype=torch.float64)
        dv = self._head(work, n) if scores else None
        df = None
        tail = min(self.q_full, n)
        disturbances = pieces.v[n - tail:].clone()
        disturbance_cov = torch.zeros((tail, tail), dtype=torch.float64)
        if self.exact and work["rank"]:
            g_all, lg, c_factor = work["g"], work["lg"], work["c_factor"]
            rank = work["rank"]
            ma, n_ma = self.ma_index, len(self.ma_index)
            eye = torch.eye(rank, dtype=torch.float64)
            m_run = torch.zeros((r, r), dtype=torch.float64)
            b_run = torch.zeros(r, dtype=torch.float64)
            if scores:
                d_g_all, d_sx = work["d_g"], work["d_sx"]
                dm_run = torch.zeros((n_ma, r, r), dtype=torch.float64)
                db_run = torch.zeros((kk, r), dtype=torch.float64)
                df = torch.zeros((kk, lg), dtype=torch.float64)
            chunk = max(64, _CHUNK_ELEMENTS // (max(1, n_ma) * r * r))
            for start in range(0, lg, chunk):
                stop = min(lg, start + chunk)
                g, vt = g_all[start:stop], pieces.v[start:stop]
                gg = g[:, :, None] * g[:, None, :]
                m_t = m_run + gg.cumsum(dim=0) - gg
                b_inc = g * vt[:, None]
                b_t = b_run + b_inc.cumsum(dim=0) - b_inc
                chol = torch.linalg.cholesky(eye + c_factor.T @ m_t @ c_factor)
                w_t = c_factor @ torch.cholesky_solve(
                    c_factor.T.expand(stop - start, -1, -1), chol)
                a_t = (w_t @ g[:, :, None])[:, :, 0]
                mean_t = (w_t @ b_t[:, :, None])[:, :, 0]
                f[start:stop] = 1.0 + (g * a_t).sum(dim=1)
                v[start:stop] = vt - (g * mean_t).sum(dim=1)
                if scores:
                    c_t = b_t - (m_t @ mean_t[:, :, None])[:, :, 0]
                    e_t = g - (m_t @ a_t[:, :, None])[:, :, 0]
                    dvt = dv[:, start:stop]
                    inc = g[None] * dvt[:, :, None]
                    if n_ma:
                        d_g = d_g_all[:, start:stop]
                        inc[ma] += d_g * vt[None, :, None]
                    db_t = db_run[:, None] + inc.cumsum(dim=1) - inc
                    gdm = (db_t * a_t[None]).sum(dim=-1)
                    dft = torch.zeros((kk, stop - start), dtype=torch.float64)
                    if self.n_arma:
                        gdm[k:] += torch.einsum("tr,krs,ts->kt", e_t, d_sx, c_t)
                        dft[k:] = torch.einsum("tr,krs,ts->kt", e_t, d_sx, e_t)
                    if n_ma:
                        dgg = d_g[:, :, :, None] * g[None, :, None, :]
                        dgg = dgg + dgg.transpose(-1, -2)
                        dm_t = dm_run[:, None] + dgg.cumsum(dim=1) - dgg
                        gdm[ma] += (d_g * mean_t[None]).sum(dim=-1) \
                            - torch.einsum("tr,ktrs,ts->kt", a_t, dm_t, mean_t)
                        dft[ma] += 2.0 * (d_g * a_t[None]).sum(dim=-1) \
                            - torch.einsum("tr,ktrs,ts->kt", a_t, dm_t, a_t)
                        dm_run = dm_run + dgg.sum(dim=1)
                    dv[:, start:stop] = dvt - gdm
                    df[:, start:stop] = dft
                    db_run = db_run + inc.sum(dim=1)
                m_run = m_run + gg.sum(dim=0)
                b_run = b_run + b_inc.sum(dim=0)
            first = n - tail
            if tail and lg > first:
                g_tail = torch.zeros((tail, r), dtype=torch.float64)
                g_tail[:lg - first] = g_all[first:lg]
                disturbances = disturbances - g_tail @ work["m"]
                disturbance_cov = g_tail @ work["w"] @ g_tail.T
        score = None
        if scores:
            s2 = sigma * sigma
            score = torch.empty((n, kk + 1), dtype=torch.float64)
            block = -(dv * (v / (s2 * f))[None])
            if df is not None:
                lg = df.shape[1]
                block[:, :lg] += df * (0.5 * (v.square() / (s2 * f.square()) - 1.0 / f))[None, :lg]
            score[:, :kk] = block.T
            score[:, kk] = -1.0 / sigma + v.square() / (s2 * sigma * f)
        return Decomposition(innovations=v, variance_ratio=f, scores=score,
                             disturbances=disturbances, disturbance_covariance=disturbance_cov)


def shrink(values: list[float], *, moving_average: bool, radius: float = 0.95) -> list[float]:
    """Scale coefficient i by kappa^i so that every inverse root has modulus <= ``radius``."""
    if not values:
        return values
    current = spectral_radius([-value for value in values] if moving_average else values)
    if not math.isfinite(current):
        return [0.0] * len(values)
    if current <= radius:
        return values
    kappa = radius / current
    return [value * kappa ** (i + 1) for i, value in enumerate(values)]


@torch.no_grad()
def starting_values(y: Tensor, x: Tensor, sizes: tuple[int, int, int, int],
                    period: int) -> Tensor:
    """Hannan-Rissanen starting values for (b, phi, theta, Phi, Theta).

    b is the OLS coefficient vector; its residuals u are fitted by a long
    Yule-Walker autoregression whose residuals stand in for the disturbances;
    u is then regressed on its own lags and the lagged disturbance estimates at
    the lags of the expanded polynomials. The coefficients at lags 1..p and
    s, 2s, .. seed phi and Phi (likewise theta and Theta), and each polynomial
    is pulled inside the unit circle (inverse roots at most 0.95).
    """
    p, q, sp, sq = sizes
    s = max(int(period), 1)
    n, k = x.shape
    start = torch.zeros(k + p + q + sp + sq, dtype=torch.float64)
    u = y
    if k:
        fit = least_squares(x, y)
        start[fit.kept] = fit.beta
        u = fit.resid
    if not p + q + sp + sq:
        return start
    ar_lags = sorted({i + j * s for i in range(p + 1) for j in range(sp + 1)} - {0})
    ma_lags = sorted({i + j * s for i in range(q + 1) for j in range(sq + 1)} - {0})
    longest = max(ar_lags + ma_lags)
    burn = 0
    disturbances = None
    if ma_lags:
        order = min(max(20, 2 * longest), (n - 1) // 3)
        if order < 1:
            return start
        gamma = [float(torch.dot(u[lag:], u[:n - lag])) / n for lag in range(order + 1)]
        long_ar, _ = levinson(gamma, order)
        disturbances = apply_polynomial(u[None], [1.0, *(-value for value in long_ar)])[0]
        burn = order
    first = burn + longest
    if n - first < len(ar_lags) + len(ma_lags) + 2:
        return start
    columns = [u[first - lag:n - lag] for lag in ar_lags] \
        + [disturbances[first - lag:n - lag] for lag in ma_lags]
    try:
        fit = least_squares(torch.stack(columns, dim=1), u[first:].contiguous())
    except KernelError:
        return start
    coefficients = torch.zeros(len(columns), dtype=torch.float64)
    coefficients[fit.kept] = fit.beta
    values = coefficients.tolist()
    ar = dict(zip(ar_lags, values[:len(ar_lags)], strict=True))
    ma = dict(zip(ma_lags, values[len(ar_lags):], strict=True))
    blocks = (
        shrink([ar[i] for i in range(1, p + 1)], moving_average=False),
        shrink([ma[i] for i in range(1, q + 1)], moving_average=True),
        shrink([ar[j * s] if j * s > p else 0.0 for j in range(1, sp + 1)], moving_average=False),
        shrink([ma[j * s] if j * s > q else 0.0 for j in range(1, sq + 1)], moving_average=True),
    )
    start[k:] = torch.tensor([value for block in blocks for value in block], dtype=torch.float64)
    return start
