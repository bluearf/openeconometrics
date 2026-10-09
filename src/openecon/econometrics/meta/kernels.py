"""Bounded float64 GLS and scalar ML/REML variance likelihood, no oracle imports."""
from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from .common import finite


@torch.no_grad()
def gls(y, v, x, tau):
    """Evaluate many nonnegative tau² values; no study-by-study dense matrix."""
    w = 1/(v[None, :]+tau[:, None])
    a = torch.einsum("ki,ip,iq->kpq", w, x, x)
    spectrum = torch.linalg.eigvalsh(a)
    if bool((spectrum[:, 0] <= spectrum[:, -1]*1e-12).any()):
        raise AnalysisError("singular_design", "The scaled weighted moderator design exceeds the condition limit (Gram ratio 1e12); remove redundant moderators or rescale.")
    factor, info = torch.linalg.cholesky_ex(a)
    if bool((info != 0).any()):
        raise AnalysisError("singular_design", "The weighted moderator design is singular.")
    bread = torch.cholesky_inverse(factor)
    beta = torch.cholesky_solve(torch.einsum("ki,ip,i->kp", w, x, y)[:, :, None], factor).squeeze(2)
    resid = y[None, :]-beta@x.T
    rss = (w*resid.square()).sum(1)
    trace_p = w.sum(1)-torch.einsum("kpq,kpq->k", bread, torch.einsum("ki,ip,iq->kpq", w.square(), x, x))
    logdet = 2*factor.diagonal(dim1=1, dim2=2).log().sum(1)
    return beta, bread, resid, w, rss, trace_p, logdet


@torch.no_grad()
def fit(y, v, x, method, *, max_iterations=100):
    """Variance is profiled after stable moderator centering/scaling.

    Sample variance normalizes yi/vi and tau²; an 81-point log grid brackets
    positive-to-negative score roots. All bracketed likelihood maxima and zero
    are compared. A bounded tail/KKT check rejects unresolved solutions.
    """
    k, p = x.shape
    variance_scale = float(v.median())
    offset = float(y.mean()) if bool((x[:, 0] == 1).all()) else 0.
    ys, vs = (y-offset)/math.sqrt(variance_scale), v/variance_scale
    zero = torch.zeros(1, dtype=torch.float64, device="cpu")
    initial = gls(ys, vs, x, zero)
    q0 = float(initial[4][0])
    c = float(initial[5][0])
    if not math.isfinite(c) or c <= 0:
        raise AnalysisError("singular_design", "The residual projection has no positive information.")
    dl = max(0., (q0-(k-p))/c)
    evaluations = 1

    def evaluate(tau):
        nonlocal evaluations
        result = gls(ys, vs, x, tau)
        evaluations += len(tau)
        rss, trace, logdet = result[4:]
        likelihood = -.5*((vs[None, :]+tau[:, None]).log().sum(1)+rss)
        if method == "REML":
            likelihood -= .5*logdet
        score = .5*((result[3].square()*result[2].square()).sum(1)
                    -(trace if method == "REML" else result[3].sum(1)))
        finite(likelihood, "profile likelihood")
        finite(score, "variance score")
        return likelihood, score

    tau_value, iterations = (0. if method == "common" else dl), 0
    if method in ("ML", "REML"):
        # Includes the boundary and tiny variances relative to median sampling v.
        upper = max(1., float(ys.square().mean()), float(vs.max()))*1024
        grid = torch.cat([zero, torch.logspace(-12, math.log10(upper), 81, dtype=torch.float64, device="cpu")])
        ll, scores = evaluate(grid)
        if float(scores[-1]) >= 0 or int(torch.argmax(ll)) == len(grid)-1:
            raise AnalysisError("variance_not_converged", "The bounded profile likelihood has no resolved upper tail.")
        candidates = [0.]
        for i in range(len(grid)-1):
            if float(scores[i]) > 0 and float(scores[i+1]) <= 0:
                lo, hi = float(grid[i]), float(grid[i+1])
                converged = False
                for iteration in range(max_iterations):
                    mid = (lo+hi)/2
                    _, score = evaluate(torch.tensor([mid], dtype=torch.float64, device="cpu"))
                    if float(score[0]) > 0:
                        lo = mid
                    else:
                        hi = mid
                    if hi-lo <= 2e-11*max(1., mid):
                        converged = True
                        break
                iterations += iteration+1 if max_iterations else 0
                if not converged:
                    raise AnalysisError("variance_not_converged", "ML/REML variance root exceeded its iteration budget.")
                candidates.append((lo+hi)/2)
        candidate_ll, candidate_score = evaluate(torch.tensor(candidates, dtype=torch.float64, device="cpu"))
        selected = int(torch.argmax(candidate_ll))
        tau_value = candidates[selected]
        # Dimensionless score scaled by information: both boundary and interior.
        scaled_score = float(candidate_score[selected])*(1+tau_value)/k
        if (tau_value == 0 and scaled_score > 1e-7) or (tau_value > 0 and abs(scaled_score) > 1e-7):
            raise AnalysisError("variance_not_converged", "The selected variance fails the profile-score optimality check.")
    result = gls(ys, vs, x, torch.tensor([tau_value], dtype=torch.float64, device="cpu"))
    beta, bread, resid, w, rss, _, logdet = [a[0] for a in result]
    beta = beta*math.sqrt(variance_scale)
    if offset:
        beta[0] += offset
    covariance = bread*variance_scale
    residual = resid*math.sqrt(variance_scale)
    tau2 = tau_value*variance_scale
    # Restricted likelihood uses a design-normalized constant; ML uses k.
    loglik_ml = -.5*(float((v+tau2).log().sum())+float(rss)+k*math.log(2*math.pi))
    _, determinant_x = torch.linalg.slogdet(x.T@x)
    loglik_reml = -.5*(float((v+tau2).log().sum())+float(rss)+float(logdet)-p*math.log(variance_scale)
                       -float(determinant_x)+(k-p)*math.log(2*math.pi))
    finite(beta, "coefficients")
    finite(covariance, "covariance")
    return {"beta": beta, "bread": covariance, "resid": residual, "weights": w/variance_scale,
            "rss": float(rss), "tau2": tau2, "q0": q0, "typical_variance": (k-p)/c*variance_scale,
            "loglik_ml": loglik_ml, "loglik_reml": loglik_reml,
            "iterations": iterations, "evaluations": evaluations}
