"""Exact global ARMA likelihood and analytic derivatives on stateful blocks.

The conditional innovation recursion carries its real history. The stationary
presample correction uses the same native closed-form kernel as dense ARIMA,
with only its finite impulse head retained; all sums use every observation.
"""
from __future__ import annotations

import math

import torch

from openecon.econometrics.arima.filters import inverse_filter, lag_polynomial, levinson
from openecon.econometrics.arima.kernels import ArmaLikelihood, Pieces, _DECAYED, shrink
from openecon.engines.contracts import KernelError
from openecon.engines.streaming_filter import LagState, PolynomialState
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.engines.linalg import least_squares
from openecon.engines.execution import qr_factor
from openecon.resources import plan_workspace


class StreamingArmaLikelihood(ArmaLikelihood):
    def __init__(self, factory, n, k, *, budget_bytes, workspace_reserver=None, **orders):
        super().__init__(torch.zeros(1, dtype=torch.float64), torch.zeros((1, k), dtype=torch.float64), **orders)
        self.factory, self.n, self.budget_bytes = factory, int(n), int(budget_bytes)
        self._cached_impulse = None
        self.last_plan = None
        self.workspace_reserver = workspace_reserver

    def _reserve(self, operation, buffers):
        self.last_plan = self.workspace_reserver(operation, buffers) if self.workspace_reserver is not None else plan_workspace(operation, buffers, budget_bytes=self.budget_bytes)
        return self.last_plan

    def _impulse(self, theta, stheta):
        if self._cached_impulse is not None:
            return self._cached_impulse
        if not self.q_full:
            return torch.ones(1, dtype=torch.float64), None, None
        length = min(self.n, max(256, 8*(self.q_full+self.r)))
        while True:
            self._reserve("exact ARMA finite presample state", {
                "impulse_and_derivatives": 128*length,
                "finite_head_and_derivatives": 32*(self.n_psi+self.r*(1+len(self.ma_index)))*(length+self.r),
                "prefix_decomposition_and_score_state": 128*(length+self.r)*(self.n_psi+self.k+self.r+4),
                "decomposition_state_chunks": 64*min(length, 2_000_000//max(1, len(self.ma_index)*self.r*self.r))*self.r*self.r*(1+len(self.ma_index)),
                "parameter_algebra": 256*(self.r*self.r*(1+self.n_arma)+(self.n_psi+1)**2),
            })
            unit = torch.zeros((1, length), dtype=torch.float64)
            unit[0, 0] = 1.
            psi = inverse_filter(inverse_filter(unit, theta), stheta, self.period)
            rows = [psi, inverse_filter(psi, theta) if theta else None,
                    inverse_filter(psi, stheta, self.period) if stheta else None]
            stack = torch.cat([row for row in rows if row is not None]).abs()
            peak = stack.max(dim=1, keepdim=True).values
            if length == self.n or bool((stack[:, -self.q_full:] <= _DECAYED*peak).all()):
                break
            length = min(self.n, 8*length)
        significant = (stack > _DECAYED*peak).any(dim=0).nonzero()
        keep = int(significant[-1])+1 if len(significant) else 1
        self._cached_impulse = tuple(None if row is None else row[0, :keep].clone() for row in rows)
        return self._cached_impulse

    def _head(self, work, length):
        if length > work["head_jacobian"].shape[1]:
            raise KernelError("invalid_design", "ARMA requested a full-sample Jacobian from a bounded head.")
        return work["head_jacobian"][:, :length]

    def conditional_blocks(self, psi, *, derivatives=True):
        phi, theta, sphi, stheta = self.split(psi)
        ar, sar = lag_polynomial(phi, sign=-1.), lag_polynomial(sphi, unit=self.period, sign=-1.)
        head_ar, head_sar = PolynomialState(ar), PolynomialState(sar)
        phi_sar, sphi_ar = PolynomialState(sar), PolynomialState(ar)
        moving, seasonal = PolynomialState(theta, inverse=True), PolynomialState(stheta, inverse=True, unit=self.period)
        ma_derivative, sma_derivative = PolynomialState(theta, inverse=True), PolynomialState(stheta, inverse=True, unit=self.period)
        histories = [LagState(p) for p in (len(phi), len(theta), len(sphi)*self.period, len(stheta)*self.period)]
        count = 0
        for x, y in self.factory():
            if x.shape != (len(y), self.k) or y.dtype != torch.float64 or x.dtype != torch.float64:
                raise KernelError("invalid_design", "ARMA replay changed its numeric design.")
            u = y-x@psi[:self.k] if self.k else y
            head = torch.cat((u[None], x.T), dim=0) if derivatives else u[None]
            rows = [head_sar(head_ar(head))]
            if derivatives and phi:
                rows.append(phi_sar(u[None]))
            if derivatives and sphi:
                rows.append(sphi_ar(u[None]))
            batch = seasonal(moving(torch.cat(rows, dim=0)))
            v, dv = batch[0], None
            if derivatives:
                values = [-batch[1:1+self.k]]
                row = 1+self.k
                if phi:
                    values.append(-torch.stack(histories[0](batch[row], list(range(1, len(phi)+1)))))
                    row += 1
                if theta:
                    z = ma_derivative(v[None])[0]
                    values.append(-torch.stack(histories[1](z, list(range(1, len(theta)+1)))))
                if sphi:
                    values.append(-torch.stack(histories[2](batch[row], [j*self.period for j in range(1, len(sphi)+1)])))
                if stheta:
                    z = sma_derivative(v[None])[0]
                    values.append(-torch.stack(histories[3](z, [j*self.period for j in range(1, len(stheta)+1)])))
                dv = torch.cat(values, dim=0)
            count += len(y)
            yield v, dv, x, y
        if count != self.n:
            raise KernelError("source_changed", "ARMA replay row count changed between likelihood passes.")

    @torch.no_grad()
    def evaluate(self, psi, *, derivatives=True):
        if not bool(torch.isfinite(psi).all()):
            return None
        polynomials = self.split(psi)
        if not self.feasible(*polynomials):
            return None
        self._cached_impulse = None
        length = min(self.n, len(self._impulse(polynomials[1], polynomials[3])[0])+self.r-1) if self.exact else 0
        ss, dss = _CompensatedSum(()), _CompensatedSum((self.n_psi,))
        retained = {name: size for name, size in (self.last_plan.buffers if self.last_plan else []) if name in {"impulse_and_derivatives", "finite_head_and_derivatives", "decomposition_state_chunks", "prefix_decomposition_and_score_state"}}
        self._reserve("ARMA replay head and retained filter states", {**retained,
            "presample_head": 64*length*(self.n_psi+self.k+3),
            "filter_histories": 128*(self.p_full+self.q_full+self.r+1)*(self.k+4),
            "parameter_algebra": 256*(self.n_psi+1)**2,
        })
        vhead = torch.empty(length, dtype=torch.float64)
        jhead = torch.empty((self.n_psi, length), dtype=torch.float64) if derivatives else None
        xhead, yhead = torch.empty((length, self.k), dtype=torch.float64), torch.empty(length, dtype=torch.float64)
        retained = 0
        for v, dv, x, y in self.conditional_blocks(psi, derivatives=derivatives):
            ss.add(v@v)
            if derivatives:
                dss.add(2.*(dv@v))
            used = min(len(v), length-retained)
            if used > 0:
                vhead[retained:retained+used] = v[:used]
                xhead[retained:retained+used] = x[:used]
                yhead[retained:retained+used] = y[:used]
                if derivatives:
                    jhead[:, retained:retained+used] = dv[:, :used]
                retained += used
        if not math.isfinite(float(ss.value)):
            return None
        pieces = Pieces(ss=float(ss.value), logdet=0., v=vhead)
        if derivatives:
            pieces.d_ss, pieces.d_logdet = dss.value.clone(), torch.zeros(self.n_psi, dtype=torch.float64)
            pieces.work["head_jacobian"] = jhead
        pieces.work.update(head_x=xhead, head_y=yhead)
        if self.exact and not self._exact(pieces, polynomials, derivatives):
            return None
        return pieces

    def gauss_newton(self, psi):
        pieces = self.evaluate(psi, derivatives=False)
        if pieces is None:
            return None
        cross = _CompensatedSum((self.n_psi, self.n_psi))
        for _, dv, _, _ in self.conditional_blocks(psi):
            cross.add(dv@dv.T)
        return (self.n/pieces.ss)*cross.value

    def replay_starting_values(self):
        """The full-source Hannan-Rissanen recipe; no prefix/model subsample."""
        tree, total = _TSQRTree(), _CompensatedSum(())
        for x, y in self.factory():
            tree.add(qr_factor(torch.cat((x, y[:, None]), dim=1)))
            total.add(y@y)
        factor = tree.finish()
        start = torch.zeros(self.n_psi, dtype=torch.float64)
        if self.k:
            fit = least_squares(factor[:, :self.k], factor[:, -1])
            start[fit.kept] = fit.beta
        residual = factor[:, -1]-factor[:, :self.k]@start[:self.k]
        self.start_residual_ss, self.response_ss = float(residual@residual), float(total.value)
        if not self.n_arma:
            return start
        p, q, sp, sq = self.sizes
        ar_lags = sorted({i+j*self.period for i in range(p+1) for j in range(sp+1)}-{0})
        ma_lags = sorted({i+j*self.period for i in range(q+1) for j in range(sq+1)}-{0})
        longest, burn, long_ar = max(ar_lags+ma_lags), 0, []
        if ma_lags:
            burn = min(max(20, 2*longest), (self.n-1)//3)
            if burn < 1:
                return start
            gamma, history = _CompensatedSum((burn+1,)), LagState(burn)
            for x, y in self.factory():
                u = y-x@start[:self.k]
                shifted = history(u, list(range(burn+1)))
                gamma.add(torch.stack([u@row for row in shifted]))
            long_ar, _ = levinson((gamma.value/self.n).tolist(), burn)
        first = burn+longest
        if self.n-first < len(ar_lags)+len(ma_lags)+2:
            return start
        tree, count = _TSQRTree(), 0
        raw_history, disturbance_history = LagState(longest), LagState(longest)
        long_filter = PolynomialState([1., *[-value for value in long_ar]])
        for x, y in self.factory():
            u = y-x@start[:self.k]
            values = raw_history(u, ar_lags)
            if ma_lags:
                e = long_filter(u[None])[0]
                values += disturbance_history(e, ma_lags)
            skipped = min(len(u), max(0, first-count))
            count += len(u)
            if skipped < len(u):
                design = torch.stack(values, dim=1)[skipped:]
                tree.add(qr_factor(torch.cat((design, u[skipped:, None]), dim=1)))
        try:
            factor = tree.finish()
            fit = least_squares(factor[:, :-1], factor[:, -1])
        except KernelError:
            return start
        coefficients = torch.zeros(len(ar_lags)+len(ma_lags), dtype=torch.float64)
        coefficients[fit.kept] = fit.beta
        ar, ma = dict(zip(ar_lags, coefficients[:len(ar_lags)].tolist(), strict=True)), dict(zip(ma_lags, coefficients[len(ar_lags):].tolist(), strict=True))
        blocks = (shrink([ar[i] for i in range(1, p+1)], moving_average=False),
                  shrink([ma[i] for i in range(1, q+1)], moving_average=True),
                  shrink([ar[j*self.period] if j*self.period > p else 0. for j in range(1, sp+1)], moving_average=False),
                  shrink([ma[j*self.period] if j*self.period > q else 0. for j in range(1, sq+1)], moving_average=True))
        start[self.k:] = torch.tensor([value for block in blocks for value in block], dtype=torch.float64)
        return start

    def replay_conditional_clone(self):
        p, q, sp, sq = self.sizes
        return StreamingArmaLikelihood(self.factory, self.n, self.k, budget_bytes=self.budget_bytes,
                                       workspace_reserver=self.workspace_reserver,
                                       p=p, q=q, seasonal_p=sp, seasonal_q=sq, period=self.period, exact=False)

    def decomposition_blocks(self, theta, *, scores=True):
        sigma = float(theta[-1])
        if not sigma > 0.:
            raise KernelError("numerical_failure", "ARMA sigma must be positive.")
        psi, s2 = theta[:-1], sigma*sigma
        pieces = self.evaluate(psi, derivatives=False)
        if pieces is None:
            raise KernelError("numerical_failure", "ARMA likelihood is undefined at estimates.")
        head = None
        if self.exact:
            p, q, sp, sq = self.sizes
            prefix = ArmaLikelihood(pieces.work["head_y"], pieces.work["head_x"],
                                    p=p, q=q, seasonal_p=sp, seasonal_q=sq, period=self.period, exact=True)
            head = prefix.decompose(theta, scores=scores)
        cursor, disturbances = 0, torch.empty(0, dtype=torch.float64)
        for v, dv, x, y in self.conditional_blocks(psi, derivatives=scores):
            innovations, variance = v.clone(), torch.ones(len(v), dtype=torch.float64)
            score = torch.cat((-(dv*(v/s2)[None]).T, (-1./sigma+v.square()/(s2*sigma))[:, None]), dim=1) if scores else None
            used = min(len(v), max(0, 0 if head is None else len(head.innovations)-cursor))
            if used:
                innovations[:used], variance[:used] = head.innovations[cursor:cursor+used], head.variance_ratio[cursor:cursor+used]
                if scores:
                    score[:used] = head.scores[cursor:cursor+used]
            cursor += len(v)
            if self.q_full:
                disturbances = torch.cat((disturbances, v))[-self.q_full:].clone()
            yield innovations, variance, score, x, y, v
        self.end_disturbances = disturbances
        self.end_disturbance_covariance = torch.zeros((len(disturbances), len(disturbances)), dtype=torch.float64)
        if self.exact and pieces.work["rank"] and self.q_full:
            first, lg = self.n-len(disturbances), pieces.work["lg"]
            if first < lg:
                tail_g = torch.zeros((len(disturbances), self.r), dtype=torch.float64)
                tail_g[:lg-first] = pieces.work["g"][first:lg]
                self.end_disturbances = disturbances-tail_g@pieces.work["m"]
                self.end_disturbance_covariance = tail_g@pieces.work["w"]@tail_g.T
