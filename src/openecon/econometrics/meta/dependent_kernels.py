"""Bounded spectral GLS and one-component dependent-meta likelihoods."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError

MAX_EVALUATIONS = 1024
MAX_WORK = 500_000_000


def work_bound(n, p):
    """Structural arithmetic bound; not elapsed time or process RSS."""
    return 12*n**3 + 12*MAX_EVALUATIONS*(n*p*p+p**3)


def normalize_design(x, intercept):
    normalized = x.clone()
    p = x.shape[1]
    transform = torch.eye(p, dtype=torch.float64, device="cpu")
    for j in range(int(intercept), p):
        center = float(x[:, j].mean()) if intercept else 0.0
        spread = float((x[:, j]-center).square().mean().sqrt())
        if not math.isfinite(spread) or spread <= 0:
            raise AnalysisError("singular_design", "A numeric moderator is constant or unscaled.")
        normalized[:, j] = (x[:, j]-center)/spread
        transform[j, j] = 1/spread
        if intercept:
            transform[0, j] = -center/spread
    singular = torch.linalg.svdvals(normalized)
    if float(singular[-1]) <= 1e-10*float(singular[0]):
        raise AnalysisError("singular_design", "The scaled moderator design exceeds its condition limit.")
    return normalized, transform


class Geometry:
    """Whiten the known sampling covariance once, then diagonalize heterogeneity."""

    def __init__(self, x, y, s, cluster_index, model, intercept):
        self.n, self.p = x.shape
        self.x_raw, self.y_raw = x, y
        self.scale = float(s.diagonal().median())
        self.origin = float(y.mean()) if intercept else 0.0
        self.intercept = intercept
        xn, self.transform = normalize_design(x, intercept)
        factor, info = torch.linalg.cholesky_ex(s/self.scale)
        if int(info) != 0:
            raise AnalysisError("invalid_covariance", "The sampling covariance must be positive definite.")
        self.factor = factor
        self.logdet_s = float(2*factor.diagonal().log().sum())
        if model == "effect":
            a = torch.eye(self.n, dtype=torch.float64, device="cpu")
        elif model == "study":
            a = (cluster_index[:, None] == cluster_index[None, :]).to(torch.float64)
        else:
            a = torch.zeros((self.n, self.n), dtype=torch.float64, device="cpu")
        left = torch.linalg.solve_triangular(factor, a, upper=False)
        whitened = torch.linalg.solve_triangular(factor, left.T, upper=False).T
        eigenvalues, self.rotation = torch.linalg.eigh((whitened+whitened.T)/2)
        # A is structurally PSD (identity, cluster-incidence product, or zero).
        # Only the roundoff-sized negative eigenvalues of that constructed A
        # are removed; an indefinite supplied sampling matrix never reaches here.
        tolerance = 1e-11*max(1.0, float(eigenvalues.abs().max()))
        if float(eigenvalues.min()) < -tolerance:
            raise AnalysisError("numerical_failure", "The constructed heterogeneity geometry is not positive semidefinite.")
        self.eigenvalues = eigenvalues.clamp_min(0)
        if model == "study":
            # The incidence product ZZ' has exactly G positive eigenvalues.
            # Preserve its structural null space instead of treating numerical
            # eigenvalues near zero as additional random-effect directions.
            groups = int(cluster_index.max())+1
            self.eigenvalues[:self.n-groups] = 0
            if bool((self.eigenvalues[self.n-groups:]<=0).any()):
                raise AnalysisError("numerical_failure", "The declared study-incidence rank was not resolved.")
        self.x = self.rotation.T@torch.linalg.solve_triangular(factor, xn, upper=False)
        yn = (y-self.origin)/math.sqrt(self.scale)
        self.y = self.rotation.T@torch.linalg.solve_triangular(factor, yn[:, None], upper=False).flatten()
        sign, self.logdet_x = torch.linalg.slogdet(xn.T@xn)
        if float(sign) != 1:
            raise AnalysisError("singular_design", "The moderator design is singular.")
        self.evaluations = 0

    def evaluate(self, tau):
        self.evaluations += 1
        if self.evaluations > MAX_EVALUATIONS:
            raise AnalysisError("work_budget", "The complete variance solver exceeded 1024 GLS evaluations.")
        denominator = 1+tau*self.eigenvalues
        w = 1/denominator
        gram = self.x.T@(w[:, None]*self.x)
        spectrum = torch.linalg.eigvalsh(gram)
        if float(spectrum[0]) <= float(spectrum[-1])*1e-12:
            raise AnalysisError("singular_design", "The weighted scaled Gram matrix exceeds the 1e12 condition limit.")
        factor, info = torch.linalg.cholesky_ex(gram)
        if int(info) != 0:
            raise AnalysisError("singular_design", "The weighted moderator design is singular.")
        bread = torch.cholesky_inverse(factor)
        beta = torch.cholesky_solve((self.x.T@(w*self.y))[:, None], factor).flatten()
        residual = self.y-self.x@beta
        rss = float((w*residual.square()).sum())
        logdet_gram = float(2*factor.diagonal().log().sum())
        determinant = self.logdet_s+float(denominator.log().sum())
        ll_ml = -.5*(determinant+rss+self.n*math.log(2*math.pi))
        ll_reml = -.5*(determinant+rss+logdet_gram-float(self.logdet_x)+(self.n-self.p)*math.log(2*math.pi))
        derivative_weights = self.eigenvalues*w.square()
        score_ml = .5*(float((derivative_weights*residual.square()).sum())-float((self.eigenvalues*w).sum()))
        correction = float(torch.trace(bread@(self.x.T@(derivative_weights[:, None]*self.x))))
        score_reml = score_ml+.5*correction
        if not all(math.isfinite(v) for v in [rss,ll_ml,ll_reml,score_ml,score_reml]):
            raise AnalysisError("numerical_failure", "The profiled dependent-meta likelihood is not finite.")
        return {"beta_normalized": beta, "bread_normalized": bread, "weights": w,
                "rss": rss, "loglik_ml": ll_ml-.5*self.n*math.log(self.scale),
                "loglik_reml": ll_reml-.5*(self.n-self.p)*math.log(self.scale),
                "score_ml": score_ml, "score_reml": score_reml}

    def finish(self, evaluated, tau):
        beta = self.transform@evaluated["beta_normalized"]*math.sqrt(self.scale)
        if self.intercept:
            beta[0] += self.origin
        covariance = self.transform@evaluated["bread_normalized"]@self.transform.T*self.scale
        inverse_l = torch.linalg.solve_triangular(self.factor, torch.eye(self.n, dtype=torch.float64, device="cpu"), upper=False)
        inverse_v = inverse_l.T@self.rotation@(evaluated["weights"][:, None]*self.rotation.T)@inverse_l/self.scale
        return {"beta": beta, "covariance": (covariance+covariance.T)/2,
                "inverse_v": (inverse_v+inverse_v.T)/2,
                "residual": self.y_raw-self.x_raw@beta, "tau2": tau*self.scale,
                "rss": evaluated["rss"], "loglik_ml": evaluated["loglik_ml"],
                "loglik_reml": evaluated["loglik_reml"], "variance_scale": self.scale}


@torch.no_grad()
def at_variance(x, y, s, cluster_index, model, intercept, tau2):
    """Replay one admitted GLS calculation; never run the variance optimizer."""
    geometry = Geometry(x,y,s,cluster_index,model,intercept)
    tau = tau2/geometry.scale
    evaluated = geometry.evaluate(tau)
    result = geometry.finish(evaluated,tau)
    result["score"] = evaluated["score_reml"]
    result["score_ml"] = evaluated["score_ml"]
    result["score_reml"] = evaluated["score_reml"]
    return result


@torch.no_grad()
def fit(x, y, s, cluster_index, model, method, intercept):
    geometry = Geometry(x,y,s,cluster_index,model,intercept)
    zero = geometry.evaluate(0.0)
    if model == "common":
        result = geometry.finish(zero,0.0)
        result["q0"] = zero["rss"]
        result["solver"] = {"status":"known covariance", "evaluations":1,
                            "max_evaluations":MAX_EVALUATIONS, "candidates":[0.0],
                            "profile_grid":[], "score_residual":None, "upper_bound":None,
                            "boundary":True, "algorithm":"closed-form complete GLS"}
        return result
    score_key = "score_reml" if method == "REML" else "score_ml"
    ll_key = "loglik_reml" if method == "REML" else "loglik_ml"
    upper = max(1.0,float(geometry.y.square().mean()),float(1/geometry.eigenvalues[geometry.eigenvalues>0].min()))*1024
    grid = [0.0, *torch.logspace(-12,math.log10(upper),81,dtype=torch.float64,device="cpu").tolist()]
    evaluated = [zero, *[geometry.evaluate(value) for value in grid[1:]]]
    if evaluated[-1][score_key] >= 0 or max(range(len(evaluated)),key=lambda j:evaluated[j][ll_key]) == len(evaluated)-1:
        raise AnalysisError("variance_not_converged", "The bounded profile likelihood has an unresolved upper tail.")
    candidates = [(0.0,zero)]
    for j in range(len(grid)-1):
        if evaluated[j][score_key] > 0 and evaluated[j+1][score_key] <= 0:
            lo,hi = grid[j],grid[j+1]
            for _ in range(80):
                mid = (lo+hi)/2
                current = geometry.evaluate(mid)
                if current[score_key] > 0:
                    lo = mid
                else:
                    hi = mid
                if hi-lo <= 2e-12*max(1.0,mid):
                    break
            else:
                raise AnalysisError("variance_not_converged", "The variance score root did not converge.")
            value = (lo+hi)/2
            candidates.append((value,geometry.evaluate(value)))
    tau,selected = max(candidates,key=lambda candidate:candidate[1][ll_key])
    score_residual = selected[score_key]*(1+tau)/geometry.n
    if (tau == 0 and score_residual > 1e-7) or (tau > 0 and abs(score_residual) > 1e-7):
        raise AnalysisError("variance_not_converged", "The selected variance fails the boundary/interior profile-score check.")
    result = geometry.finish(selected,tau)
    result["q0"] = zero["rss"]
    result["solver"] = {"status":"converged", "evaluations":geometry.evaluations,
                        "max_evaluations":MAX_EVALUATIONS,
                        "candidates":[{"tau2":value*geometry.scale,"loglik":item[ll_key]} for value,item in candidates],
                        "profile_grid":[{"tau2":value*geometry.scale,"loglik":item[ll_key],"normalized_score":item[score_key]} for value,item in zip(grid,evaluated)],
                        "score_residual":score_residual,"upper_bound":upper*geometry.scale,
                        "boundary":tau == 0,
                        "algorithm":"spectral GLS; all bracketed score maxima and zero compared; finite grid is not arbitrary global-optimization proof"}
    return result
