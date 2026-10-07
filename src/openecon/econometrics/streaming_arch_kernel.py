"""Full-source ARCH likelihood with finite native state and analytic tangents."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from openecon.econometrics.arch import densities
from openecon.econometrics.arch.kernels import POWER_BOUNDS
from openecon.engines.arch_state import state_block
from openecon.engines.execution import qr_factor
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.engines.streaming_filter import LagState, PolynomialState
from openecon.engines.linalg import least_squares
from openecon.engines.contracts import KernelError
from openecon.econometrics.arima.filters import spectral_radius
from openecon.econometrics.arch.recursions import dense
from openecon.resources import plan_workspace
from .streaming_arma_kernel import StreamingArmaLikelihood


class StreamingArchLikelihood:
    def __init__(self, factory, n, layout, *, budget_bytes, scratch=None):
        self.factory, self.n, self.layout, self.budget_bytes = factory, n, layout, budget_bytes
        self.linear = layout.kind != "egarch" and layout.archm is None
        self.scratch, self.kink_stores = scratch, []
        p, q = max(layout.ar_lags, default=0), max(layout.ma_lags, default=0)
        self.mean = StreamingArmaLikelihood(lambda: ((x, y) for x, y, _ in self.factory()), n, layout.k_x,
                                            budget_bytes=budget_bytes, p=p, q=q, exact=False)
        self.plan = plan_workspace("finite ARCH state and analytic tangent rings", {
            "variance_and_mean_lag_ring": 128*layout.max_lag*(layout.k+1),
            "parameter_and_score_factors": 512*(layout.k+1)**2,
        }, budget_bytes=budget_bytes)

    def presample(self, theta):
        layout, ix = self.layout, self.layout.index
        point = torch.zeros(self.mean.n_psi, dtype=torch.float64)
        point[:layout.k_x] = theta[ix.x]
        for idx, lag in zip(ix.ar, layout.ar_lags, strict=True):
            point[layout.k_x+lag-1] = theta[idx]
        for idx, lag in zip(ix.ma, layout.ma_lags, strict=True):
            point[layout.k_x+self.mean.sizes[0]+lag-1] = theta[idx]
        ss, dss = _CompensatedSum(()), _CompensatedSum((layout.k,))
        rows = list(range(layout.k_x))+[layout.k_x+lag-1 for lag in layout.ar_lags]+[layout.k_x+self.mean.sizes[0]+lag-1 for lag in layout.ma_lags]
        for residual, derivative, _, _ in self.mean.conditional_blocks(point):
            ss.add(residual@residual)
            partial = torch.zeros(layout.k, dtype=torch.float64)
            if rows:
                partial[ix.mean] = 2.*(derivative[rows]@residual)
            dss.add(partial)
        h0, dh0 = float(ss.value)/self.n, dss.value/self.n
        return (h0, dh0) if math.isfinite(h0) and h0>0. and bool(torch.isfinite(dh0).all()) else None

    def initial_state(self, theta, h0, dh0):
        layout, ix, longest = self.layout, self.layout.index, self.layout.max_lag
        state = torch.zeros((5, longest), dtype=torch.float64)
        derivatives = torch.zeros((5, longest, layout.k), dtype=torch.float64)
        if layout.kind == "egarch":
            state[2] = math.log(h0)
            derivatives[2] = dh0/h0
        elif layout.kind == "parch":
            phi = float(theta[ix.p[0]])
            power = h0**(.5*phi)
            d_power = .5*phi*power/h0*dh0
            d_power[ix.p[0]] += .5*power*math.log(h0)
            state[0], state[2] = power, power
            derivatives[0], derivatives[2] = d_power, d_power
        else:
            state[0], state[1], state[2] = h0, .5*h0, h0
            derivatives[0], derivatives[1], derivatives[2] = dh0, .5*dh0, dh0
        return state, derivatives

    def blocks(self, theta, *, signs=None, flat=None):
        layout, ix = self.layout, self.layout.index
        base = self.presample(theta)
        if base is None:
            yield None
            return
        h0, dh0 = base
        state, tangent = self.initial_state(theta, h0, dh0)
        cursor, offset = 0, 0
        kind = {"garch": 0, "gjr": 1, "egarch": 2, "parch": 3}[layout.kind]
        mean_kind = {None: 0, "variance": 1, "sd": 2, "log": 3}[layout.archm]
        for x, y, z in self.factory():
            pattern = torch.empty(0, dtype=torch.float64) if signs is None else signs.block(offset, len(y))
            values = state_block(y, x, z, theta, state, tangent, cursor, kind, mean_kind, ix.x, ix.z, ix.c,
                                 ix.a, ix.g, ix.b, ix.ar, ix.ma, list(layout.arch_lags), list(layout.garch_lags),
                                 list(layout.ar_lags), list(layout.ma_lags), ix.p[0] if ix.p else -1,
                                 ix.m[0] if ix.m else -1, pattern)
            residual, disturbance, variance, de, dlogh, cursor, valid = values
            if not valid or not all(bool(torch.isfinite(v).all()) for v in (residual, disturbance, variance, de, dlogh)):
                yield None
                return
            mask = None
            if flat:
                mask = torch.zeros(len(y), dtype=torch.bool)
                for position in flat:
                    if offset<=position<offset+len(y):
                        mask[position-offset] = True
            value, dle, dlh, dlt = densities.log_density(layout.dist, residual, variance,
                float(theta[ix.d[0]]) if ix.d else 0., derivatives=True, flat=mask)
            score = de*dle[:, None]+dlogh*dlh[:, None]
            if ix.d:
                score[:, ix.d[0]] += dlt
            if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(score).all()):
                yield None
                return
            yield offset, residual, disturbance, variance, score, value, de/variance.sqrt()[:, None]-.5*(residual/variance.sqrt())[:, None]*dlogh, h0
            offset += len(y)
        if offset != self.n:
            from openecon.analysis_contracts import AnalysisError
            raise AnalysisError("source_changed", "ARCH numerical replay changed its complete row count.")

    def feasible(self, theta):
        ix = self.layout.index
        return bool(torch.isfinite(theta).all()) and densities.feasible(self.layout.dist, float(theta[ix.d[0]]) if ix.d else 0.) and (
            not ix.p or POWER_BOUNDS[0]<float(theta[ix.p[0]])<POWER_BOUNDS[1])

    def evaluate(self, theta, *, scores=False, derivatives=True, signs=None, active=None, flat=None):
        if not self.feasible(theta):
            return None
        value, gradient, tree = _CompensatedSum(()), _CompensatedSum((self.layout.k,)), _TSQRTree()
        preview, tail, selected = [], [], {}
        minimum_variance, minimum_standard = math.inf, math.inf
        for row in self.blocks(theta, signs=signs, flat=flat):
            if row is None:
                return None
            start, e, u, h, score, ll, dz, h0 = row
            value.add(ll.sum())
            if derivatives:
                gradient.add(score.sum(0))
            if scores:
                tree.add(qr_factor(score))
            take = min(len(e), max(0, 400-start))
            if take:
                preview.append(torch.stack((e[:take], u[:take], h[:take]), 1))
            tail.append(torch.stack((e, u, h), 1)[-400:].clone())
            if len(tail)>1:
                tail = [torch.cat(tail)[-400:].clone()]
            minimum_variance = min(minimum_variance, float(h.min()))
            minimum_standard = min(minimum_standard, float((e/h.sqrt()).abs().min()))
            for position in active or []:
                if start<=position<start+len(e):
                    at = position-start
                    selected[position] = (e[at].clone(), h[at].clone(), dz[at].clone())
        head, last = torch.cat(preview), torch.cat(tail)[-400:]
        return SimpleNamespace(value=float(value.value), gradient=gradient.value.clone() if derivatives else None,
            scores=tree.finish() if scores else None, residual=head[:, 0], disturbance=head[:, 1], variance=head[:, 2],
            tail_residual=last[:, 0], tail_disturbance=last[:, 1], tail_variance=last[:, 2],
            presample_variance=h0, selected=selected, minimum_variance=minimum_variance, minimum_standardized=minimum_standard)

    def replay_starting_values(self):
        from openecon.analysis_contracts import AnalysisError
        from openecon.econometrics.arch.maximize import _variance_candidates, _VARIANCE_RANGE, _T_START, _GED_START
        layout, ix = self.layout, self.layout.index
        tree, total = _TSQRTree(), _CompensatedSum(())
        xx, zz, zmean = _CompensatedSum((layout.k_x,)), _CompensatedSum((layout.k_z,)), _CompensatedSum((layout.k_z,))
        for x, y, z in self.factory():
            tree.add(qr_factor(torch.cat((x, y[:, None]), 1)))
            total.add(y@y)
            xx.add(x.square().sum(0))
            zz.add(z.square().sum(0))
            zmean.add(z.sum(0))
        factor = tree.finish()
        beta = torch.zeros(layout.k_x, dtype=torch.float64)
        if layout.k_x:
            fit = least_squares(factor[:, :-1], factor[:, -1])
            beta[fit.kept] = fit.beta
        residual = factor[:, -1]-factor[:, :-1]@beta
        s2 = float(residual@residual)/self.n
        if not s2>1e-20*max(float(total.value)/self.n, 1e-300):
            raise AnalysisError("perfect_fit", "The mean regressors reproduce the outcome exactly.")
        if not _VARIANCE_RANGE[0]<s2<_VARIANCE_RANGE[1]:
            raise AnalysisError("numerical_failure", "ARCH residual variance is outside its float64 range; rescale the outcome.")
        rho, ma = self._arma_start(beta)
        point = torch.zeros(layout.k, dtype=torch.float64)
        point[ix.x], point[ix.ar], point[ix.ma] = beta, torch.tensor(rho, dtype=torch.float64), torch.tensor(ma, dtype=torch.float64)
        base = self.presample(point)
        if base is not None and 0.<base[0]<=s2:
            s2 = base[0]
        else:
            rho, ma = [0.]*len(rho), [0.]*len(ma)
        sd, scale = math.sqrt(s2), torch.ones(ix.k, dtype=torch.float64)
        if layout.k_x:
            rms = (xx.value/self.n).sqrt()
            scale[ix.x] = sd/torch.where(rms>0, rms, torch.ones_like(rms))
        if ix.m:
            scale[ix.m[0]] = {"variance": 1./sd, "sd": 1., "log": sd}[layout.archm]
        if layout.k_z:
            spread = (zz.value/self.n-(zmean.value/self.n).square()).clamp_min(0).sqrt()
            scale[ix.z] = 1./torch.where(spread>0, spread, torch.ones_like(spread))
        elif layout.kind != "egarch":
            scale[ix.c] = s2
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):
            raise AnalysisError("numerical_failure", "ARCH parameter scales are outside finite float64 precision.")
        starts = []
        for candidate in _variance_candidates(layout, s2):
            theta = torch.zeros(layout.k, dtype=torch.float64)
            theta[ix.x], theta[ix.ar], theta[ix.ma] = beta, torch.tensor(rho, dtype=torch.float64), torch.tensor(ma, dtype=torch.float64)
            theta[ix.a], theta[ix.b] = torch.tensor(candidate["a"], dtype=torch.float64), torch.tensor(candidate["b"], dtype=torch.float64)
            if ix.g:
                theta[ix.g] = torch.tensor(candidate["g"], dtype=torch.float64)
            omega = candidate["omega"]
            theta[ix.c] = math.log(omega) if layout.k_z and layout.kind != "egarch" else omega
            if ix.p:
                theta[ix.p[0]] = candidate["power"]
            if ix.d:
                theta[ix.d[0]] = _T_START if layout.dist == "t" else _GED_START
            starts.append(theta)
        return starts, scale

    def _arma_start(self, beta):
        layout = self.layout
        ar_lags, ma_lags = list(layout.ar_lags), list(layout.ma_lags)
        rho, ma = [0.]*len(ar_lags), [0.]*len(ma_lags)
        longest = max([0, *ar_lags, *ma_lags])
        if not longest:
            return rho, ma
        first, order, long_coeff = max(ar_lags, default=0), 0, None
        try:
            if ma_lags:
                order = min(max(longest+4, round(math.log(self.n)**2)), self.n//4)
                if order<1:
                    return rho, ma
                history, tree, seen = LagState(order), _TSQRTree(), 0
                for x, y, _ in self.factory():
                    u = y-x@beta
                    lags = history(u, list(range(1, order+1)))
                    skip = min(len(u), max(0, order-seen))
                    seen += len(u)
                    if skip<len(u):
                        tree.add(qr_factor(torch.cat((torch.stack(lags, 1)[skip:], u[skip:, None]), 1)))
                factor = tree.finish()
                fit = least_squares(factor[:, :-1], factor[:, -1])
                long_coeff = torch.zeros(order, dtype=torch.float64)
                long_coeff[fit.kept] = fit.beta
                first = order+max(ma_lags)
            if self.n-first<2*(len(ar_lags)+len(ma_lags))+2:
                return rho, ma
            history, innovations, tree, seen = LagState(longest), LagState(longest), _TSQRTree(), 0
            filt = PolynomialState([1., *(-long_coeff).tolist()]) if order else None
            for x, y, _ in self.factory():
                u = y-x@beta
                rows = history(u, ar_lags)
                if ma_lags:
                    e = filt(u[None])[0]
                    e[:min(len(e), max(0, order-seen))] = 0.
                    rows += innovations(e, ma_lags)
                skip = min(len(u), max(0, first-seen))
                seen += len(u)
                if skip<len(u):
                    tree.add(qr_factor(torch.cat((torch.stack(rows, 1)[skip:], u[skip:, None]), 1)))
            factor = tree.finish()
            fit = least_squares(factor[:, :-1], factor[:, -1])
            values = torch.zeros(len(ar_lags)+len(ma_lags), dtype=torch.float64)
            values[fit.kept] = fit.beta
            rho, ma = values[:len(ar_lags)].tolist(), values[len(ar_lags):].tolist()
        except KernelError:
            return [0.]*len(ar_lags), [0.]*len(ma_lags)
        for values, lags, sign in ((rho, ar_lags, 1.), (ma, ma_lags, -1.)):
            for _ in range(60):
                if not values or spectral_radius(dense(values, lags, sign))<.95:
                    break
                values[:] = [.8*value for value in values]
        return rho, ma

    def replay_flat_mask(self, active):
        return tuple(active) if self.layout.dist == "ged" and active else None

    def replay_piece_hessian(self, offset, transform, t, signs=None, active=None):
        from .streaming_arch_kinks import piece_hessian
        return piece_hessian(self, offset, transform, t, signs, active)

    def replay_polish(self, offset, transform, start, hessian=None):
        from .streaming_arch_kinks import polish
        return polish(self, offset, transform, start, hessian)

    def close(self):
        for store in self.kink_stores:
            store.close()
        self.kink_stores.clear()

    def replay_diagnose(self, theta):
        out = self.evaluate(theta, derivatives=False)
        if out is None:
            return None
        ix = self.layout.index
        if self.layout.kind == "parch" and float(theta[ix.p[0]])<=1.:
            return "The power-ARCH likelihood has a cusp at zero residuals for power at most1; the data do not identify a regular maximum."
        if self.layout.kind == "egarch" and self.layout.dist == "ged" and math.exp(float(theta[ix.d[0]]))<2. and out.minimum_standardized<1e-8:
            return "The EGARCH/GED maximum lies on a kink with unbounded density curvature; use normal or Student t errors."
        coefficients = torch.cat((theta[ix.a], theta[ix.g], theta[ix.b]))
        if self.layout.kind != "egarch" and (bool((coefficients<0.).any()) or (not self.layout.k_z and float(theta[ix.c])<0.)):
            return "The unrestricted ARCH search left nonnegative variance coefficients without finding a regular maximum; simplify the variance equation."
        return None
