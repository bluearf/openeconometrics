"""Log likelihood of the ARCH family with analytic scores.

Model. With u_t the structural disturbance and e_t = sqrt(h_t) z_t the innovation,

    y_t = x_t'b + psi g(h_t) + u_t,       u_t = sum_j rho_j u_(t-j) + sum_k theta_k e_(t-k) + e_t,

(g is h, sqrt(h) or ln h; AR and MA terms as in Stata's ``arch``/``arima``) and one of

    garch   h_t = w_t + sum a_i e_(t-i)^2 + sum b_j h_(t-j)
    gjr     h_t = w_t + sum a_i e_(t-i)^2 + sum g_i e_(t-i)^2 1(e_(t-i) < 0) + sum b_j h_(t-j)
    egarch  ln h_t = w_t + sum a_i z_(t-i) + sum g_i (|z_(t-i)| - sqrt(2/pi)) + sum b_j ln h_(t-j)
    parch   s_t = w_t + sum a_i |e_(t-i)|^phi + sum b_j s_(t-j),        s_t = h_t^(phi/2)

where w_t is the constant omega, or with variance regressors ``exp(l_0 + z_t'l)``
(``l_0 + z_t'l`` for EGARCH), Stata's multiplicative heteroskedasticity. The log
likelihood is ``sum_t ln f(e_t / sqrt(h_t)) - ln(h_t)/2`` (``densities``).

Presample values (Stata's defaults ``arma0(zero)`` and ``arch0(xb)``). Presample u and e
of the ARMA recursion are zero. Presample variances and squared innovations equal

    h_0 = (1/T) sum_t e~_t^2,

the mean squared residual of the mean equation with its ARMA terms AT THE CURRENT
PARAMETERS (e~ leaves out the ARCH-in-mean term, which itself depends on h_0); h_0 is
therefore differentiated with the mean parameters. Presample news take their expected
values: e^2 = h_0, e^2 1(e<0) = h_0 / 2, z = 0 and |z| - sqrt(2/pi) = 0, |e|^phi = s =
h_0^(phi/2).

Evaluation. The variance state is v = h, ln h or s. For the models that are linear
filters of known inputs (every model except EGARCH, without ARCH-in-mean) the path and
all derivative paths are solved by ``arima.filters.inverse_filter`` with no time loop
(engine "filter"). Otherwise the state path comes from ``recursions.run_path`` (a loop of
Python floats) and the derivative paths from ``recursions.solve_recurrence`` (engine
"scan"), which handles every model and is used to cross-check the filter engine.

Gradient in reverse mode. For the filter engine the gradient alone (every evaluation of
the optimizer and of the numerical Hessian) needs no derivative path at all: with
q_t = dl_t/dv_t and lambda = G(L)^-T q (one anti-causal filter),

    sum_t q_t (d v_t) = sum_t lambda_t (direct derivative of the forcing)_t,

and the mean parameters, which enter through e_t = (R_t)/theta(L), are collected in the
same way by one anti-causal MA filter. The cost is that of a few vector operations
whatever the number of parameters. Per-observation scores (OPG and robust covariances)
use the forward derivative paths described next, and tests check that both agree.

Derivatives. With G_t the partial derivatives of v_t with respect to the parameters at
fixed lagged states (including the presample values through h_0),

    d v_t = G_t + sum_i A_(t,i) d e_(t-i) + sum_l C_(t,l) d v_(t-l)
    d e_t = E_t - kappa_t d v_t + sum_j rho_j kappa_(t-j) d v_(t-j) - sum_k theta_k d e_(t-k)

where A = dv_t/de_(t-i), C = dv_t/dv_(t-l), kappa_t = psi g'(h_t) dh_t/dv_t and E_t the
direct derivative of e_t. Per-observation scores are
``dl/de * d e_t + dl/dln h * d ln h_t`` plus the direct derivative of the density.

Smoothness. The EGARCH likelihood has a kink wherever a standardized residual is zero
(the |z| term); ``evaluate(signs=...)`` replaces |z_t| by ``signs_t z_t``, the smooth
piece selected by a reference sign pattern, which ``kinks`` uses to obtain reliable
Hessians and to locate a maximum that lies on a kink. The power-ARCH news |e|^phi has
a kink (phi = 1) or a cusp (phi < 1) at e = 0 that cannot be removed this way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.econometrics.arch import densities
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.arch.recursions import (
    adjoint_filter, dense, lagged, lead, run_path, solve_recurrence,
)
from openecon.econometrics.arima.filters import inverse_filter

POWER_BOUNDS = (0.05, 20.0)      # admissible range of the power-ARCH exponent phi


@dataclass
class Evaluation:
    """One evaluation of the likelihood at a full (unrestricted-layout) parameter vector."""

    value: float
    gradient: Tensor | None          # [K]
    scores: Tensor | None            # [T, K] per-observation scores, on request
    loglik: Tensor                   # [T] per-observation log likelihood
    residual: Tensor                 # e_t
    disturbance: Tensor              # u_t = y_t - x_t'b - psi g(h_t)
    variance: Tensor                 # h_t
    presample_variance: float        # h_0
    d_standardized: Tensor | None = None   # [m, K]: d z_t / d theta at the ``active`` periods


def _finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


class ArchLikelihood:
    """The likelihood of one specification on one sample (float64 tensors, time ordered)."""

    def __init__(self, y: Tensor, x: Tensor, z: Tensor, layout: Layout):
        self.y, self.x, self.z, self.layout = y, x, z, layout
        self.n = int(y.shape[0])
        self.xt = x.T.contiguous()
        self.zt = z.T.contiguous()
        # Linear filters of known inputs: no EGARCH news, no feedback of h on the mean.
        self.linear = layout.kind != "egarch" and layout.archm is None

    def scan_lags(self) -> tuple[int, int]:
        """Lags of d v and of d e carried by the derivative recursion of the scan engine.

        The recursion has ``lags_v + lags_e`` state rows, and its transition maps take
        ``8 n (lags_v + lags_e)^2`` bytes.
        """
        layout = self.layout
        lags_v = max([1, *layout.garch_lags,
                      *(layout.arch_lags if layout.kind == "egarch" else ()),
                      *(layout.ar_lags if layout.archm is not None else ())])
        return lags_v, max([1, *layout.arch_lags, *layout.ma_lags])

    # ---- pieces ---------------------------------------------------------------------

    def _mean(self, theta: Tensor, derivatives: bool):
        """Residuals of the mean equation without the ARCH-in-mean term, and h_0.

        Returns ``(m, e, D, h0, dh0)``: m = y - x'b, e the ARMA-filtered residual,
        D = d e / d(b, rho, theta) as ``[k, T]`` and the presample variance h_0 with its
        derivative. ``None`` when the filter leaves the float64 range.
        """
        layout, ix, n = self.layout, self.layout.index, self.n
        m = self.y - self.x @ theta[ix.x] if layout.k_x else self.y
        rho, tma = theta[ix.ar].tolist(), theta[ix.ma].tolist()
        w = m
        for value, lag in zip(rho, layout.ar_lags, strict=True):
            w = w - value * lagged(m, lag)
        ma = dense(tma, layout.ma_lags)
        e = inverse_filter(w, ma) if ma else w
        if not _finite(e):
            return None
        h0 = float(e.square().mean())
        if not (math.isfinite(h0) and h0 > 0.0):
            return None
        if not derivatives:
            return m, e, None, h0, None
        rows = []
        if layout.k_x:
            block = -self.xt
            for value, lag in zip(rho, layout.ar_lags, strict=True):
                block = block + value * lagged(self.xt, lag)
            rows.append(block)
        rows += [-lagged(m, lag)[None] for lag in layout.ar_lags]
        rows += [-lagged(e, lag)[None] for lag in layout.ma_lags]
        if rows:
            d = torch.cat(rows)
            if ma:
                d = inverse_filter(d, ma)
            if not _finite(d):
                return None
        else:
            d = torch.empty((0, n), dtype=torch.float64)
        return m, e, d, h0, (2.0 / n) * (d @ e)

    # ---- evaluation -----------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, theta: Tensor, *, scores: bool = False, derivatives: bool = True,
                 engine: str | None = None, signs: Tensor | None = None,
                 active: list[int] | None = None, flat: Tensor | None = None,
                 ) -> Evaluation | None:
        """Log likelihood, gradient and (on request) per-observation scores at ``theta``.

        ``theta`` is the full parameter vector in the order of ``layout.index``. Returns
        ``None`` when the point is infeasible: a non-positive or non-finite variance, a
        distribution or power parameter outside its domain, or an explosive recursion.
        ``engine`` forces "filter" (linear models only) or "scan"; by default the filter
        engine is used whenever the model allows it.

        EGARCH only: ``signs`` ([T], entries in [-1, 1]) replaces |z_t| by
        ``signs_t z_t`` in the news and in every derivative, and ``active`` lists periods
        whose ``d z_t / d theta`` is returned in ``Evaluation.d_standardized``. ``flat``
        (GED only) masks observations whose density kernel is taken as zero
        (``densities.log_density``).
        """
        layout, ix, n = self.layout, self.layout.index, self.n
        kind = layout.kind
        if not _finite(theta):
            return None
        tau = float(theta[ix.d[0]]) if ix.d else 0.0
        if not densities.feasible(layout.dist, tau):
            return None
        phi = float(theta[ix.p[0]]) if ix.p else 2.0
        if not POWER_BOUNDS[0] < phi < POWER_BOUNDS[1]:
            return None
        psi = float(theta[ix.m[0]]) if ix.m else 0.0
        a, b = theta[ix.a].tolist(), theta[ix.b].tolist()
        g = theta[ix.g].tolist() if ix.g else [0.0] * len(a)
        rho, tma = theta[ix.ar].tolist(), theta[ix.ma].tolist()
        engine = engine or ("filter" if self.linear else "scan")
        if engine == "filter" and not self.linear:
            raise ValueError("The filter engine needs a model that is linear in its lagged state.")

        reverse = engine == "filter" and not scores       # gradient without derivative paths
        base = self._mean(theta, derivatives and not reverse)
        if base is None:
            return None
        m, e0, d0, h0, dh0 = base

        # Variance constant w_t.
        if layout.k_z:
            linear = float(theta[ix.c]) + self.z @ theta[ix.z]
            omega = linear if kind == "egarch" else linear.exp()
            if not _finite(omega):
                return None
        else:
            omega = float(theta[ix.c])

        # Presample news and state, and their derivatives with respect to h_0.
        if kind == "egarch":
            pre1 = pre2 = d_pre1 = d_pre2 = 0.0
            pre_v, d_pre_v = math.log(h0), 1.0 / h0
        elif kind == "parch":
            pre1 = pre_v = h0 ** (0.5 * phi)
            d_pre1 = d_pre_v = 0.5 * phi * pre1 / h0
            pre2 = d_pre2 = 0.0
        else:
            pre1 = pre_v = h0
            pre2, d_pre1, d_pre_v, d_pre2 = 0.5 * h0, 1.0, 1.0, 0.5
        garch_filter = dense(b, layout.garch_lags, -1.0)

        # ---- state path --------------------------------------------------------------
        z_std = None
        if engine == "filter":
            e, u = e0, m
            news1 = e.abs().pow(phi) if kind == "parch" else e.square()
            news2 = news1 * (e < 0) if kind == "gjr" else None
            forcing = omega.clone() if isinstance(omega, Tensor) \
                else torch.full((n,), omega, dtype=torch.float64)
            for i, lag in enumerate(layout.arch_lags):
                forcing += a[i] * lagged(news1, lag, pre1)
                if news2 is not None:
                    forcing += g[i] * lagged(news2, lag, pre2)
            for value, lag in zip(b, layout.garch_lags, strict=True):
                forcing[:lag] += value * pre_v
            v = inverse_filter(forcing, garch_filter) if garch_filter else forcing
            if not _finite(v) or not bool((v > 0.0).all()):
                return None
            h = v.pow(2.0 / phi) if kind == "parch" else v
        else:
            path = run_path(
                m.tolist(), omega.tolist() if isinstance(omega, Tensor) else omega, kind=kind,
                arch=list(zip(a, layout.arch_lags)),
                asymmetry=list(zip(g, layout.arch_lags)) if layout.asymmetric else [],
                garch=list(zip(b, layout.garch_lags)), presample=(pre1, pre2, pre_v),
                power=phi, psi=psi, archm=layout.archm, ar=list(zip(rho, layout.ar_lags)),
                ma=list(zip(tma, layout.ma_lags)), abs_mean=densities.ABS_NORMAL,
                signs=signs.tolist() if signs is not None and kind == "egarch" else None)
            if path is None:
                return None
            e = torch.tensor(path[0], dtype=torch.float64)
            u = e if path[1] is path[0] else torch.tensor(path[1], dtype=torch.float64)
            v = torch.tensor(path[2], dtype=torch.float64)
            h = v.exp() if kind == "egarch" else v.pow(2.0 / phi) if kind == "parch" else v
            if kind == "egarch":
                z_std = e / h.sqrt()
                sign = torch.sign(z_std) if signs is None else signs
                news1, news2 = z_std, sign * z_std - densities.ABS_NORMAL
            elif kind == "parch":
                news1, news2 = e.abs().pow(phi), None
            else:
                news1 = e.square()
                news2 = news1 * (e < 0) if kind == "gjr" else None
        if not _finite(h) or not bool((h > 0.0).all()):
            return None

        loglik, d_e, d_log_h, d_tau = densities.log_density(layout.dist, e, h, tau, derivatives,
                                                            flat)
        value = float(loglik.sum())
        if not math.isfinite(value):
            return None
        if not derivatives:
            return Evaluation(value, None, None, loglik, e, u, h, h0)
        if reverse:
            gradient = self._reverse(theta, m, e, v, omega, news1, news2,
                                     (pre1, pre2, pre_v), (d_pre1, d_pre2, d_pre_v), phi, h0,
                                     d_e, d_log_h, d_tau)
            if not _finite(gradient):
                return None
            return Evaluation(value, gradient, None, loglik, e, u, h, h0)

        # ---- direct derivatives G of the variance state ------------------------------
        k = ix.k
        direct = torch.zeros((k, n), dtype=torch.float64)
        if layout.k_z:
            direct[ix.z] = self.zt if kind == "egarch" else self.zt * omega
            direct[ix.c] = 1.0 if kind == "egarch" else omega
        else:
            direct[ix.c] = 1.0
        for i, lag in enumerate(layout.arch_lags):
            direct[ix.a[i]] = lagged(news1, lag, pre1)
            if ix.g:
                direct[ix.g[i]] = lagged(news2, lag, pre2)
        for j, lag in enumerate(layout.garch_lags):
            direct[ix.b[j]] = lagged(v, lag, pre_v)
        # Presample values depend on h_0, hence on the mean parameters (and on phi).
        reach = min(layout.max_lag, n)
        presample = torch.zeros(reach, dtype=torch.float64)
        for i, lag in enumerate(layout.arch_lags):
            presample[:lag] += a[i] * d_pre1 + g[i] * d_pre2
        for value_b, lag in zip(b, layout.garch_lags, strict=True):
            presample[:lag] += value_b * d_pre_v
        mean_ix = ix.mean
        if mean_ix:
            direct[mean_ix, :reach] += dh0[:, None] * presample[None, :]
        if kind == "parch":
            nonzero = e != 0.0
            log_abs = torch.where(nonzero, e.abs().log(), torch.zeros_like(e))
            pre_phi = 0.5 * pre1 * math.log(h0)
            row = torch.zeros(n, dtype=torch.float64)
            for i, lag in enumerate(layout.arch_lags):
                row += a[i] * lagged(news1 * log_abs, lag, pre_phi)
            for value_b, lag in zip(b, layout.garch_lags, strict=True):
                row[:lag] += value_b * pre_phi
            direct[ix.p[0]] = row

        # A_i: derivative of the news entering v_t with respect to e at its own date.
        if kind == "garch":
            slopes = [2.0 * a[i] * e for i in range(len(a))]
        elif kind == "gjr":
            negative = (e < 0).to(torch.float64)
            slopes = [2.0 * e * (a[i] + g[i] * negative) for i in range(len(a))]
        elif kind == "egarch":
            root = h.sqrt()
            slopes = [(a[i] + g[i] * sign) / root for i in range(len(a))]
        else:
            signed = torch.where(e != 0.0, e.abs().pow(phi - 1.0) * torch.sign(e),
                                 torch.zeros_like(e))
            slopes = [a[i] * phi * signed for i in range(len(a))]

        # ---- derivative paths ----------------------------------------------------------
        if engine == "filter":
            if mean_ix:
                for slope, lag in zip(slopes, layout.arch_lags, strict=True):
                    direct[mean_ix] += lagged(slope[None, :] * d0, lag)
            d_v = inverse_filter(direct, garch_filter) if garch_filter else direct
            d_e_rows, e_rows = d0, mean_ix
        else:
            magnitude = sign * z_std if kind == "egarch" else None
            d_v, d_e_rows = self._scan(theta, direct, slopes, e, u, v, h, z_std, magnitude,
                                       psi, phi)
            e_rows = list(range(k))
        if not _finite(d_v) or not _finite(d_e_rows):
            return None
        d_standardized = None
        if active and kind == "egarch":
            # z = e / sqrt(h): d z = (d e - z sqrt(h) d ln h / 2) / sqrt(h); d ln h = d v.
            at = torch.tensor(active, dtype=torch.int64)
            d_standardized = ((d_e_rows[:, at] - 0.5 * e[at] * d_v[:, at])
                              / h[at].sqrt()).T.contiguous()

        # d ln h from d v.
        if kind == "egarch":
            score = d_v
        elif kind == "parch":
            score = d_v * ((2.0 / phi) / v)
            score[ix.p[0]] -= (2.0 / phi ** 2) * v.log()
        else:
            score = d_v / h
        score *= d_log_h
        if e_rows:
            score[e_rows] += d_e_rows * d_e
        if ix.d:
            score[ix.d[0]] = d_tau
        gradient = score.sum(dim=1)
        if not _finite(gradient):
            return None
        return Evaluation(value, gradient, score.T.contiguous() if scores else None, loglik,
                          e, u, h, h0, d_standardized)

    def _pull(self, theta: Tensor, m: Tensor, e: Tensor, weights: Tensor) -> Tensor:
        """``D weights`` for D = d e / d(b, rho, theta) ([k_mean, T]) without forming D.

        D = theta(L)^-1 R with R the direct derivatives of the mean equation, so
        ``D w = R (theta(L)^-T w)``: one anti-causal filter and a few inner products.
        """
        layout, ix, n = self.layout, self.layout.index, self.n
        rho = theta[ix.ar].tolist()
        weights = adjoint_filter(weights, dense(theta[ix.ma].tolist(), layout.ma_lags))
        parts = []
        if layout.k_x:
            carried = weights
            for value, lag in zip(rho, layout.ar_lags, strict=True):
                carried = carried - value * lead(weights, lag)
            parts.append(-(self.xt @ carried))
        for series, lags in ((m, layout.ar_lags), (e, layout.ma_lags)):
            if lags:
                parts.append(torch.stack([-torch.dot(series[:max(n - lag, 0)], weights[lag:])
                                          for lag in lags]))
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)

    def _reverse(self, theta: Tensor, m: Tensor, e: Tensor, v: Tensor, omega: float | Tensor,
                 news1: Tensor, news2: Tensor | None, presample: tuple[float, float, float],
                 d_presample: tuple[float, float, float], phi: float, h0: float, d_e: Tensor,
                 d_log_h: Tensor, d_tau: Tensor | None) -> Tensor:
        """Gradient of the filter engine in reverse mode (see the module docstring)."""
        layout, ix, n = self.layout, self.layout.index, self.n
        kind = layout.kind
        a, b = theta[ix.a].tolist(), theta[ix.b].tolist()
        g = theta[ix.g].tolist() if ix.g else [0.0] * len(a)
        pre1, pre2, pre_v = presample
        d_pre1, d_pre2, d_pre_v = d_presample
        gradient = torch.zeros(ix.k, dtype=torch.float64)
        # lambda_t: derivative of the log likelihood with respect to the forcing of v_t.
        weight = d_log_h * (2.0 / phi) / v if kind == "parch" else d_log_h / v
        lam = adjoint_filter(weight, dense(b, layout.garch_lags, -1.0))
        if layout.k_z:
            scaled = lam * omega
            gradient[ix.z] = self.zt @ scaled
            gradient[ix.c] = scaled.sum()
        else:
            gradient[ix.c] = lam.sum()
        head = lam[:min(layout.max_lag, n)].cumsum(0)        # sums of lambda over the presample

        def carried(series: Tensor, lag: int, fill: float) -> Tensor:
            """``sum_t lambda_t (L^lag series)_t`` with the presample value ``fill``."""
            return torch.dot(lam[lag:], series[:max(n - lag, 0)]) + fill * head[min(lag, n) - 1]

        from_h0 = 0.0             # d log L / d h_0 through the presample news and states
        for i, lag in enumerate(layout.arch_lags):
            gradient[ix.a[i]] = carried(news1, lag, pre1)
            if ix.g:
                gradient[ix.g[i]] = carried(news2, lag, pre2)
            from_h0 += (a[i] * d_pre1 + g[i] * d_pre2) * float(head[min(lag, n) - 1])
        for j, lag in enumerate(layout.garch_lags):
            gradient[ix.b[j]] = carried(v, lag, pre_v)
            from_h0 += b[j] * d_pre_v * float(head[min(lag, n) - 1])
        if kind == "parch":
            nonzero = e != 0.0
            log_abs = torch.where(nonzero, e.abs().log(), torch.zeros_like(e))
            pre_phi = 0.5 * pre1 * math.log(h0)
            total = -(2.0 / phi ** 2) * torch.dot(d_log_h, v.log())
            for i, lag in enumerate(layout.arch_lags):
                total = total + a[i] * carried(news1 * log_abs, lag, pre_phi)
            for j, lag in enumerate(layout.garch_lags):
                total = total + b[j] * pre_phi * head[min(lag, n) - 1]
            gradient[ix.p[0]] = total
        mean_ix = ix.mean
        if mean_ix:
            # The news of period s enters v_(s+i): collect lambda led by the ARCH lags.
            ahead = torch.zeros(n, dtype=torch.float64)
            ahead_negative = torch.zeros(n, dtype=torch.float64) if kind == "gjr" else None
            for i, lag in enumerate(layout.arch_lags):
                shifted = lead(lam, lag)
                ahead += a[i] * shifted
                if ahead_negative is not None:
                    ahead_negative += g[i] * shifted
            if kind == "parch":
                through_news = phi * ahead * torch.where(
                    e != 0.0, e.abs().pow(phi - 1.0) * torch.sign(e), torch.zeros_like(e))
            elif kind == "gjr":
                through_news = 2.0 * e * (ahead + ahead_negative * (e < 0))
            else:
                through_news = 2.0 * e * ahead
            gradient[mean_ix] = self._pull(theta, m, e,
                                           through_news + d_e + (2.0 * from_h0 / n) * e)
        if ix.d:
            gradient[ix.d[0]] = d_tau.sum()
        return gradient

    def _scan(self, theta: Tensor, direct: Tensor, slopes: list[Tensor], e: Tensor, u: Tensor,
              v: Tensor, h: Tensor, z_std: Tensor | None, magnitude: Tensor | None,
              psi: float, phi: float) -> tuple[Tensor, Tensor]:
        """``(d v, d e)`` as ``[K, T]`` from the time-varying linear recursion (see module)."""
        layout, ix, n = self.layout, self.layout.index, self.n
        kind, k = layout.kind, ix.k
        a, b = theta[ix.a].tolist(), theta[ix.b].tolist()
        g = theta[ix.g].tolist() if ix.g else [0.0] * len(a)
        rho, tma = theta[ix.ar].tolist(), theta[ix.ma].tolist()
        in_mean = layout.archm is not None
        lags_v, lags_e = self.scan_lags()
        size = lags_v + lags_e
        maps = torch.zeros((n, size, size), dtype=torch.float64)
        # Row 0: d v_t on lagged d v and lagged d e.
        for value, lag in zip(b, layout.garch_lags, strict=True):
            maps[:, 0, lag - 1] += value
        if kind == "egarch":
            for i, lag in enumerate(layout.arch_lags):
                maps[:, 0, lag - 1] += lagged(-0.5 * (a[i] * z_std + g[i] * magnitude), lag)
        for slope, lag in zip(slopes, layout.arch_lags, strict=True):
            maps[:, 0, lags_v + lag - 1] += lagged(slope, lag)
        # Row lags_v: d e_t; the remaining rows shift the lags.
        for value, lag in zip(tma, layout.ma_lags, strict=True):
            maps[:, lags_v, lags_v + lag - 1] -= value
        kappa = None
        own = torch.zeros((k, n), dtype=torch.float64)        # E_t: direct derivative of u_t
        if layout.k_x:
            own[ix.x] = -self.xt
        if in_mean:
            if layout.archm == "variance":
                in_mean_value, slope_h = h, torch.ones_like(h)
            elif layout.archm == "sd":
                in_mean_value = h.sqrt()
                slope_h = 0.5 / in_mean_value
            else:
                in_mean_value, slope_h = h.log(), 1.0 / h
            own[ix.m[0]] = -in_mean_value
            if kind == "egarch":
                kappa = psi * slope_h * h
            elif kind == "parch":
                kappa = psi * slope_h * (2.0 / phi) * h / v
                # h = v^(2/phi) also depends on phi at fixed v.
                own[ix.p[0]] = psi * slope_h * h * (2.0 / phi ** 2) * v.log()
            else:
                kappa = psi * slope_h
            maps[:, lags_v, :] -= kappa[:, None] * maps[:, 0, :]
            for value, lag in zip(rho, layout.ar_lags, strict=True):
                maps[:, lags_v, lag - 1] += value * lagged(kappa, lag)
        for row in range(1, size):
            if row != lags_v:
                maps[:, row, row - 1] = 1.0
        d_e_direct = own.clone()
        for value, lag in zip(rho, layout.ar_lags, strict=True):
            d_e_direct -= value * lagged(own, lag)
        for j, lag in enumerate(layout.ar_lags):
            d_e_direct[ix.ar[j]] -= lagged(u, lag)
        for j, lag in enumerate(layout.ma_lags):
            d_e_direct[ix.ma[j]] -= lagged(e, lag)
        if kappa is not None:
            d_e_direct -= kappa[None, :] * direct
        forcing = torch.zeros((n, size, k), dtype=torch.float64)
        forcing[:, 0, :] = direct.T
        forcing[:, lags_v, :] = d_e_direct.T
        states = solve_recurrence(maps, forcing)
        return states[:, 0, :].T.contiguous(), states[:, lags_v, :].T.contiguous()
