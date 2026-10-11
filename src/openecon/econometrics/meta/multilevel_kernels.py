"""Batched two-component (three-level) meta likelihood on padded study blocks.

The marginal covariance is block diagonal by independent study,
``V_j = S_j + sigma2_study 11' + sigma2_effect I`` (Konstantopoulos 2011;
metafor ``rma.mv(random = ~1|study/effect)``). Every likelihood, profiled
score and observed Hessian is evaluated on padded ``(batch, study, slot)``
tensors; padded slots carry an identity covariance and zero data, so they
never contribute. No per-observation Python loop and no n-by-n matrix.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.optimize import maximize_newton

LOG_2PI = math.log(2*math.pi)
MAX_EVALUATIONS = 50_000
CHUNK_ENTRIES = 1_000_000
GRID_POINTS = 61
SCORE_TOLERANCE = 1e-7


class Packing:
    """Map effects (original order) to padded independent study blocks."""

    def __init__(self, cluster_index, groups):
        n = int(cluster_index.numel())
        counts = torch.bincount(cluster_index, minlength=groups)
        if int(counts.min()) < 1:
            raise AnalysisError("invalid_identifier", "Every declared study must contain at least one effect.")
        self.groups, self.n, self.counts = groups, n, counts
        self.size = int(counts.max())
        order = torch.argsort(cluster_index, stable=True)
        starts = torch.cumsum(counts, 0)-counts
        self.order = order
        self.block = cluster_index[order]
        self.slot = torch.arange(n, dtype=torch.int64)-starts[self.block]
        self.mask = torch.zeros((groups, self.size), dtype=torch.bool)
        self.mask[self.block, self.slot] = True
        self.weight = self.mask.to(torch.float64)

    def vector(self, values):
        out = values.new_zeros((self.groups, self.size, *values.shape[1:]))
        out[self.block, self.slot] = values[self.order]
        return out

    def unvector(self, padded):
        flat = padded[..., self.block, self.slot]
        out = torch.empty_like(flat)
        out[..., self.order] = flat
        return out

    def matrix(self, dense):
        """Gather complete within-study blocks; padded diagonal is one."""
        index = torch.zeros((self.groups, self.size), dtype=torch.int64)
        index[self.block, self.slot] = self.order
        blocks = dense[index[:, :, None], index[:, None, :]]*(self.weight[:, :, None]*self.weight[:, None, :])
        return blocks+torch.diag_embed(1-self.weight)

    def blocks(self, padded):
        """Return the unpadded within-study blocks as nested lists, by study."""
        sizes = self.counts.tolist()
        return [padded[j, :sizes[j], :sizes[j]].tolist() for j in range(self.groups)]


class Likelihood:
    """Normalized two-component profiled ML/REML likelihood with exact derivatives."""

    def __init__(self, x, y, s_blocks, packing, intercept):
        self.packing = packing
        self.n, self.p = x.shape
        diagonal = s_blocks.diagonal(dim1=-2, dim2=-1)[packing.mask]
        self.scale = float(diagonal.median())
        self.origin = float(y.mean()) if intercept else 0.0
        self.x_raw, self.y_raw = x, y
        self.xn, self.transform = _normalize(x, intercept)
        self.x = packing.vector(self.xn)
        self.y = packing.vector((y-self.origin)/math.sqrt(self.scale))
        self.s = (s_blocks-torch.diag_embed(1-packing.weight))/self.scale+torch.diag_embed(1-packing.weight)
        self.jm = packing.weight[:, :, None]*packing.weight[:, None, :]
        self.im = torch.diag_embed(packing.weight)
        sign, self.logdet_xx = torch.linalg.slogdet(self.xn.T@self.xn)
        if float(sign) != 1:
            raise AnalysisError("singular_design", "The moderator design is singular.")
        self.logdet_xx = float(self.logdet_xx)
        self.evaluations = 0

    def evaluate(self, theta, *, derivatives=True):
        """Evaluate a batch of normalized (study, effect) variances, shape (B, 2)."""
        theta = torch.as_tensor(theta, dtype=torch.float64).reshape(-1, 2)
        self.evaluations += theta.shape[0]
        if self.evaluations > MAX_EVALUATIONS:
            raise AnalysisError("work_budget", f"The variance-component solver exceeded {MAX_EVALUATIONS} likelihood evaluations.")
        if bool((theta < 0).any()) or not bool(torch.isfinite(theta).all()):
            raise AnalysisError("numerical_failure", "Variance components must be finite and nonnegative.")
        cell = self.packing.groups*self.packing.size**2
        step = max(1, CHUNK_ENTRIES//cell)
        if theta.shape[0] <= step:
            return self._evaluate(theta, derivatives)
        parts = [self._evaluate(theta[i:i+step], derivatives) for i in range(0, theta.shape[0], step)]
        return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}

    def _evaluate(self, theta, derivatives):
        batch = theta.shape[0]
        v = self.s[None]+theta[:, 0, None, None, None]*self.jm[None]+theta[:, 1, None, None, None]*self.im[None]
        factor, info = torch.linalg.cholesky_ex(v)
        if bool((info != 0).any()):
            raise AnalysisError("numerical_failure", "A marginal study covariance block is not positive definite.")
        logdet = 2*factor.diagonal(dim1=-2, dim2=-1).log().sum((-1, -2))
        x = self.x[None].expand(batch, -1, -1, -1)
        wx = torch.cholesky_solve(x, factor)
        wy = torch.cholesky_solve(self.y[None, :, :, None].expand(batch, -1, -1, -1), factor)[..., 0]
        gram = torch.einsum("gmp,bgmq->bpq", self.x, wx)
        gram = (gram+gram.transpose(-1, -2))/2
        spectrum = torch.linalg.eigvalsh(gram)
        if bool((spectrum[:, 0] <= spectrum[:, -1]*1e-12).any()):
            raise AnalysisError("singular_design", "The weighted scaled moderator Gram matrix exceeds the 1e12 condition limit.")
        gram_factor, info = torch.linalg.cholesky_ex(gram)
        if bool((info != 0).any()):
            raise AnalysisError("singular_design", "The weighted moderator design is singular.")
        bread = torch.cholesky_inverse(gram_factor)
        beta = torch.cholesky_solve(torch.einsum("gmp,bgm->bp", self.x, wy)[..., None], gram_factor)[..., 0]
        q = wy-torch.einsum("bgmp,bp->bgm", wx, beta)
        residual = self.y[None]-torch.einsum("gmp,bp->bgm", self.x, beta)
        rss = (residual*q).sum((-1, -2))
        logdet_gram = 2*gram_factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        ll_ml = -.5*(self.n*LOG_2PI+logdet+rss)
        ll_reml = -.5*((self.n-self.p)*LOG_2PI+logdet+logdet_gram-self.logdet_xx+rss)
        out = {"theta": theta, "beta": beta, "bread": bread, "rss": rss, "ll_ML": ll_ml, "ll_REML": ll_reml,
               "factor": factor, "q": q, "residual": residual}
        if not derivatives:
            _finite(out, ("ll_ML", "ll_REML", "rss"))
            return out
        w = torch.cholesky_inverse(factor)
        w = (w+w.transpose(-1, -2))/2
        structures = (self.jm, self.im)
        wv = [w@a for a in structures]
        tr_wv = torch.stack([m.diagonal(dim1=-2, dim2=-1).sum((-1, -2)) for m in wv], -1)
        tr_wvwv = torch.stack([torch.stack([(wv[a]*wv[b].transpose(-1, -2)).sum((-1, -2, -3)) for b in range(2)], -1)
                               for a in range(2)], -2)
        va_wx = [a@wx for a in structures]
        t = [torch.einsum("bgmp,bgmq->bpq", wx, item) for item in va_wx]
        w_va_wx = [w@item for item in va_wx]
        u = [[torch.einsum("bgmp,bgmq->bpq", va_wx[a], w_va_wx[b]) for b in range(2)] for a in range(2)]
        uq = [(a@q[..., None])[..., 0] for a in structures]
        quad = torch.stack([(q*item).sum((-1, -2)) for item in uq], -1)
        w_uq = [(w@item[..., None])[..., 0] for item in uq]
        xw_uq = [torch.einsum("bgmp,bgm->bp", wx, item) for item in uq]
        qpq = torch.stack([torch.stack([(uq[a]*w_uq[b]).sum((-1, -2))
                                        - torch.einsum("bp,bpq,bq->b", xw_uq[a], bread, xw_uq[b]) for b in range(2)], -1)
                           for a in range(2)], -2)
        trace = lambda left, right: torch.einsum("bpq,bqp->b", left, right)  # noqa: E731
        tr_pv = tr_wv-torch.stack([trace(bread, t[a]) for a in range(2)], -1)
        out["trace_p"] = tr_pv[:, 1]
        tr_pvpv = tr_wvwv-torch.stack([torch.stack([trace(bread, u[a][b])+trace(bread, u[b][a])
                                                    - trace(bread@t[a], bread@t[b]) for b in range(2)], -1)
                                       for a in range(2)], -2)
        out.update(score_ML=-.5*tr_wv+.5*quad, score_REML=-.5*tr_pv+.5*quad,
                   hessian_ML=.5*tr_wvwv-qpq, hessian_REML=.5*tr_pvpv-qpq,
                   fisher_ML=.5*tr_wvwv, fisher_REML=.5*tr_pvpv)
        for name in ("hessian_ML", "hessian_REML", "fisher_ML", "fisher_REML"):
            out[name] = (out[name]+out[name].transpose(-1, -2))/2
        _finite(out, ("ll_ML", "ll_REML", "rss", "score_ML", "score_REML", "hessian_ML", "hessian_REML"))
        return out

    # ---- scalar (one fixed component) maximization ------------------------

    def _line_points(self, which, other, values, method):
        theta = torch.zeros((len(values), 2), dtype=torch.float64)
        theta[:, which] = torch.as_tensor(values, dtype=torch.float64)
        theta[:, 1-which] = other
        out = self.evaluate(theta)
        return (out["ll_"+method], out["score_"+method][:, which], out["hessian_"+method][:, which, which])

    def line_max(self, method, which, other, upper):
        """Maximize one component given the other; boundary and every bracketed root compared."""
        grid = torch.cat([torch.zeros(1, dtype=torch.float64),
                          torch.logspace(-10, math.log10(upper), GRID_POINTS, dtype=torch.float64)])
        ll, score, _ = self._line_points(which, other, grid, method)
        if float(score[-1]) >= 0 or int(torch.argmax(ll)) == len(grid)-1:
            raise AnalysisError("variance_not_converged", "The bounded variance-component profile has no resolved upper tail.")
        candidates = [(0.0, float(ll[0]), float(score[0]))]
        for i in range(len(grid)-1):
            if float(score[i]) > 0 and float(score[i+1]) <= 0:
                value = self._root(method, which, other, float(grid[i]), float(grid[i+1]))
                point = self._line_points(which, other, [value], method)
                candidates.append((value, float(point[0][0]), float(point[1][0])))
        return max(candidates, key=lambda item: item[1])

    def _root(self, method, which, other, lo, hi):
        """Safeguarded Newton/bisection on a positive-to-nonpositive score bracket."""
        x = (lo+hi)/2
        for _ in range(200):
            _, score, hessian = self._line_points(which, other, [x], method)
            score, hessian = float(score[0]), float(hessian[0])
            if score > 0:
                lo = x
            else:
                hi = x
            if hi-lo <= 2e-13*max(1.0, x) or abs(score)*(1+x) <= 1e-14*self.n:
                return x
            step = x-score/hessian if hessian < 0 else math.nan
            x = step if lo < step < hi and math.isfinite(step) else (lo+hi)/2
        raise AnalysisError("variance_not_converged", "A variance-component score root exceeded its iteration budget.")

    # ---- joint maximization -----------------------------------------------

    def upper(self):
        return max(1.0, float(self.y.square().sum()/self.n))*1e4

    def maximize(self, method):
        upper = self.upper()
        candidates = [((0.0, 0.0), float(self.evaluate(torch.zeros((1, 2)), derivatives=False)["ll_"+method][0]), "boundary")]
        for which in (0, 1):
            value, ll, _ = self.line_max(method, which, 0.0, upper)
            if value > 0:
                theta = [0.0, 0.0]
                theta[which] = value
                candidates.append((tuple(theta), ll, "face"))
        # Coarse batched grid supplies interior Newton starts on the log scale.
        axis = torch.logspace(-6, math.log10(upper), 15, dtype=torch.float64)
        grid = torch.cartesian_prod(axis, axis)
        ll = self.evaluate(grid, derivatives=False)["ll_"+method]
        starts = grid[torch.argsort(ll, descending=True)[:3]]
        for start in starts:
            found = self._newton(method, start.log())
            if found is not None:
                candidates.append(found)
        theta, ll, kind = max(candidates, key=lambda item: item[1])
        point = self.evaluate(torch.tensor([theta], dtype=torch.float64))
        score = point["score_"+method][0]
        residual = [float(score[a])*(1+theta[a])/self.n for a in range(2)]
        for a in range(2):
            if (theta[a] == 0 and residual[a] > SCORE_TOLERANCE) or (theta[a] > 0 and abs(residual[a]) > SCORE_TOLERANCE):
                raise AnalysisError("variance_not_converged", "The selected variance components fail the boundary/interior score (KKT) check.")
        return {"theta": theta, "loglik": ll, "kind": kind, "score_residual": residual,
                "candidates": [{"sigma2": [c[0][0]*self.scale, c[0][1]*self.scale], "loglik": c[1], "kind": c[2]}
                               for c in candidates],
                "upper_bound": upper*self.scale}

    def _newton(self, method, start):
        def objective(psi):
            theta = psi.exp()
            point = self.evaluate(theta[None])
            gradient = point["score_"+method][0]*theta
            hessian = point["hessian_"+method][0]*theta[:, None]*theta[None, :]+torch.diag(gradient)
            return point["ll_"+method][0], gradient, hessian
        try:
            result = maximize_newton(objective, start, max_iter=200, raise_on_failure=False)
        except KernelError:
            return None
        theta = result.theta.exp()
        if not result.converged or bool((theta < 1e-8).any()) or not bool(torch.isfinite(theta).all()):
            return None
        value = float(result.value)
        # Polish with full Newton steps in natural scale to rounding-level stationarity.
        for _ in range(20):
            point = self.evaluate(theta[None])
            score, hessian = point["score_"+method][0], point["hessian_"+method][0]
            if float(torch.linalg.eigvalsh(hessian)[-1]) >= 0:
                break
            step = -torch.linalg.solve(hessian, score)
            trial = theta+step
            if bool((trial <= 0).any()):
                break
            trial_value = float(self.evaluate(trial[None], derivatives=False)["ll_"+method][0])
            if trial_value < value-1e-12*max(1.0, abs(value)):
                break
            theta, value = trial, max(value, trial_value)
            if float((step.abs()/theta).max()) <= 1e-13:
                break
        return ((float(theta[0]), float(theta[1])), value, "interior")

    # ---- profile likelihood -------------------------------------------------

    def profile(self, method, which, value, upper):
        """Profile log likelihood with component ``which`` fixed at normalized ``value``."""
        _, ll, _ = self.line_max(method, 1-which, value, upper)
        return ll

    def profile_interval(self, method, which, theta, loglik, crit, upper):
        """Profile-likelihood interval endpoints in normalized scale (metafor confint.rma.mv)."""
        def drop(value):
            return loglik-self.profile(method, which, value, upper)-crit
        estimate = theta[which]
        if estimate == 0 or drop(0.0) <= 0:
            lower = 0.0
        else:
            lower = _bisect(drop, 0.0, estimate)
        hi = max(estimate, 1e-8)*2
        while drop(hi) <= 0:
            if hi > upper:
                # Unbounded within the admitted variance range: report no limit.
                return lower, None
            hi *= 2
        return lower, _bisect(drop, estimate, hi)

    def finish(self, theta):
        """GLS on original scale at fixed normalized components; returns original units."""
        point = self.evaluate(torch.tensor([theta], dtype=torch.float64))
        beta = self.transform@point["beta"][0]*math.sqrt(self.scale)
        if self.origin:
            beta[0] += self.origin
        covariance = self.transform@point["bread"][0]@self.transform.T*self.scale
        covariance = (covariance+covariance.T)/2
        sigma2 = [theta[0]*self.scale, theta[1]*self.scale]
        out = {"beta": beta, "covariance": covariance, "sigma2": sigma2, "rss": float(point["rss"][0]),
               "loglik_ML": float(point["ll_ML"][0])-.5*self.n*math.log(self.scale),
               "loglik_REML": float(point["ll_REML"][0])-.5*(self.n-self.p)*math.log(self.scale),
               "variance_scale": self.scale}
        for method in ("ML", "REML"):
            out["score_"+method] = (point["score_"+method][0]/self.scale).tolist()
            out["hessian_"+method] = (point["hessian_"+method][0]/self.scale**2).tolist()
            out["fisher_"+method] = (point["fisher_"+method][0]/self.scale**2).tolist()
        return out


def _bisect(function, lo, hi):
    """Illinois false position with bisection safeguard on a sign-changing bracket."""
    f_lo, f_hi = function(lo), function(hi)
    if not (f_lo > 0 >= f_hi or f_lo <= 0 < f_hi):
        raise AnalysisError("variance_not_converged", "The profile-likelihood limit is not bracketed.")
    side = 0
    for iteration in range(200):
        if iteration % 4 == 3:
            mid = (lo+hi)/2
        else:
            mid = (lo*f_hi-hi*f_lo)/(f_hi-f_lo)
            if not lo < mid < hi:
                mid = (lo+hi)/2
        value = function(mid)
        if (value > 0) == (f_lo > 0):
            lo, f_lo = mid, value
            if side == -1:
                f_hi /= 2
            side = -1
        else:
            hi, f_hi = mid, value
            if side == 1:
                f_lo /= 2
            side = 1
        if hi-lo <= 1e-12*max(1e-12, abs(mid)) or value == 0:
            return mid
    raise AnalysisError("variance_not_converged", "The profile-likelihood limit exceeded its iteration budget.")


def _finite(out, names):
    for name in names:
        if not bool(torch.isfinite(out[name]).all()):
            raise AnalysisError("numerical_failure", "The three-level likelihood is not finite; rescale effects or variances.")


def _normalize(x, intercept):
    from .dependent_kernels import normalize_design
    return normalize_design(x, intercept)
