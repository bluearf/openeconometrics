"""Global Gaussian mixed ML/REML from bounded row and disk-group factors.

The single-intercept fast path profiles fixed effects jointly over all groups;
REML uses the determinant of the global fixed-effect factor. General one/two
level slopes use exact disk-backed joint group QR factors in a separate kernel.
Within-group TSQR and positive between-group terms avoid subtracting nearly
equal N-sized crossproducts.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
import sqlite3
from types import SimpleNamespace

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import information_criteria, wald_test
from .mixed import common
from .mixed.lmm import _BOUNDARY, _reported, _variance_terms
from .mixed.lmm_kernels import CovStructure, MixedEvaluation, MixedLikelihood
from .replay_sample import ReplaySample
from .streaming_linear import _GroupMeans, _Notes, _factor, _finite, _result, _scratch_directory, _solve

_LOG_2PI = math.log(2*math.pi)


class _Likelihood(MixedLikelihood):
    """Native two-variance likelihood with all global small moments shared."""

    def __init__(self, sample, store, within, reml):
        self.sample, self.store, self.within = sample, store, within
        self.p, self.nobs, self.reml = within.shape[1]-1, sample.nobs, reml
        self.structure, self.two_level, self.size = CovStructure("independent", 1), False, 2

    def evaluate(self, theta, *, gradient=False, blups=False, sigma2=None, beta=None):
        if blups:
            raise KernelError("unsupported_prediction", "A replay mixed likelihood does not retain all group BLUPs.")
        s2 = torch.exp(2*theta[-1]) if sigma2 is None else sigma2
        d = torch.exp(2*theta[0])/s2
        if not bool(torch.isfinite(d)) or not bool(s2>0):
            raise KernelError("numerical_failure", "The mixed variance parameters exceed finite float64 precision.")
        cross, logdet = _CompensatedSum((self.p+1, self.p+1)), _CompensatedSum(())
        cross.add(self.within.T@self.within)
        for means, _, mass, _ in self.store.group_batches(self.sample.rows):
            precision = mass/(1+mass*d)
            cross.add(means.T@(means*precision[:, None]))
            logdet.add(torch.log1p(mass*d).sum())
        block = (cross.value+cross.value.T)/2
        xvx, xvy = block[:self.p, :self.p], block[:self.p, self.p]
        chol, code = torch.linalg.cholesky_ex(xvx)
        if int(code):
            raise KernelError("singular_design", "The global mixed fixed-effect information is not positive definite.")
        if beta is None:
            beta = torch.cholesky_solve(xvy[:, None], chol).flatten()
        residual = self.within[:, self.p]-self.within[:, :self.p]@beta
        rss = _CompensatedSum(())
        rss.add(residual.square().sum())
        for means, _, mass, _ in self.store.group_batches(self.sample.rows):
            mean_r = means[:, self.p]-means[:, :self.p]@beta
            rss.add((mass/(1+mass*d)*mean_r.square()).sum())
        rvr = rss.value
        if not bool(rvr>0) or not bool(torch.isfinite(rvr)):
            raise KernelError("perfect_fit", "The mixed residual quadratic form is not positive finite.")
        pr = self.p if self.reml else 0
        value = -(self.nobs-pr)/2*(_LOG_2PI+torch.log(s2))-logdet.value/2-rvr/(2*s2)
        if self.reml:
            value -= torch.log(chol.diagonal()).sum()
        out = MixedEvaluation(value, beta, xvx, rvr, s2)
        if gradient:
            inverse = torch.cholesky_inverse(chol) if self.reml else None
            derivative = _CompensatedSum(())
            for means, _, mass, _ in self.store.group_batches(self.sample.rows):
                precision = mass/(1+mass*d)
                mean_x = means[:, :self.p]
                mean_r = means[:, self.p]-mean_x@beta
                term = -.5*precision+.5*(precision*mean_r).square()/s2
                if self.reml:
                    term += .5*precision.square()*((mean_x@inverse)*mean_x).sum(1)
                derivative.add(term.sum())
            dg = 2*d*derivative.value
            ds = -(self.nobs-pr)+rvr/s2-dg
            out.gradient = torch.stack((dg, ds))
        _finite(out.value, out.beta, out.xvx, out.rvr, out.sigma2)
        if gradient:
            _finite(out.gradient)
        return out


def _start(like):
    best, value = None, -math.inf
    for relative in (.0025, .025, .25, 2.5):
        theta = torch.tensor([.5*math.log(relative), 0.], dtype=torch.float64)
        try:
            sigma = like.profiled_sigma2(theta)
            candidate = theta+.5*torch.log(sigma)
            likelihood = float(like.value(candidate))
        except KernelError:
            continue
        if likelihood>value:
            best, value = candidate, likelihood
    if best is None:
        raise AnalysisError("invalid_start", "The global mixed likelihood has no finite starting value.")
    return best


def _rows(sample, store, scale, center, group, *, means=False):
    for batch in sample.batches():
        data = torch.cat((batch.designs["mean"], ((batch.numeric(sample.spec.outcome)-center)/scale)[:, None]), 1)
        keys = encode_cluster_labels(batch.frame[group])
        if means:
            unique, codes = store._codes(keys)
            values, _, _ = store._load_block(unique)
            if bool((values[:, 0]<=0).any()):
                raise AnalysisError("source_changed", "The mixed groups changed between row passes.")
            yield batch, data, values[codes, 1:], values[codes, 0], keys
        else:
            yield batch, data, keys


def fit_streaming_mixed(spec, source, *, batch_rows=None):
    """Bounded native Gaussian mixed ML/REML; no RAM collection."""
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("singular_information", "The mixed likelihood information is not positive definite.") from error
    except (OSError, sqlite3.Error) as error:
        raise _GroupMeans.storage_error() from error


def _fit(spec, source, *, batch_rows):
    notes = _Notes(spec)
    names, random = registry.role_columns(spec, "group"), registry.role_columns(spec, "random")
    if spec.estimator != "mixed":
        raise AnalysisError("invalid_spec", "This replay kernel requires a mixed specification.")
    if len(names)!=1 or random or not notes.option("random_intercept"):
        from .streaming_mixed_general import fit_general
        return fit_general(spec, source, batch_rows=batch_rows)
    group, reml = names[0], notes.option("method")=="reml"
    if group in {spec.outcome, *spec.predictors}:
        raise AnalysisError("invalid_spec", "The mixed grouping column must differ from the outcome and regressors.")
    if reml and spec.covariance != "nonrobust":
        raise AnalysisError("unsupported_covariance", "Robust/cluster mixed covariance requires ML, not REML.")
    clusters = registry.cluster_columns(spec) if spec.covariance == "cluster" else []
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean")
    sample.prepare()
    p, n = len(design.terms), sample.nobs
    if not p or n-p-2<=0:
        raise AnalysisError("insufficient_observations", "Mixed needs an identified fixed-effect design and more observations than parameters.")
    resource = sample.plan_rows("global random-intercept ML/REML bounded group moments", {
        "group_moments_SQLite_cache_and_decode": 6*1024**2,
        "joint_group_score_SQLite_cache": 6*1024**2,
        "global_TSQR_likelihood_covariance": 2048*(p+3)**2,
    }, 512*(p+3))
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), batch.weights, False)
    variation = float(moments.m2.value[1]/moments.mass)
    if variation<=0:
        raise AnalysisError("constant_outcome", "The mixed outcome does not vary.")
    scale = float(moments.magnitude[1])*math.sqrt(variation)
    center = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1]) if spec.intercept else 0.
    with ExitStack() as stack:
        store = _GroupMeans(p+1, len(clusters), panel=True)
        stack.callback(store.close)
        for batch, values, keys in _rows(sample, store, scale, center, group):
            store.add(keys, values, batch.weights, [encode_cluster_labels(batch.frame[name]) for name in clusters])
        store.finish()
        sizes = store.panel_structure(frequency=spec.weight_type=="fweight")
        if store.groups<2:
            raise AnalysisError("insufficient_groups", "A mixed random intercept requires at least two groups.")
        if sizes["t_max"]<=1:
            raise AnalysisError("insufficient_group_size", "Every mixed group contains one observation, so residual and random-intercept variances cannot be separated.")
        if clusters and not all(store.nested_flags):
            raise AnalysisError("cluster_not_nested", "Mixed groups must nest within the covariance clusters.")
        def within_rows():
            for batch, data, mean, _, _ in _rows(sample, store, scale, center, group, means=True):
                within = data-mean
                yield within[:, :p], within[:, p], batch.weights
        factor, factor_info = _factor(within_rows, p, require_more=False)
        like = _Likelihood(sample, store, factor, reml)
        linear_tree = _TSQRTree()
        linear_tree.add(factor)
        for means, _, mass, _ in store.group_batches(sample.rows):
            _, r = torch.linalg.qr(means*mass.sqrt()[:, None], mode="r")
            linear_tree.add(r)
        pooled = linear_tree.finish()
        linear_beta, _, _ = _solve(pooled, p)
        linear_ssr = float((pooled[:, p]-pooled[:, :p]@linear_beta).square().sum())
        if linear_ssr<=1e-24*n:
            raise AnalysisError("perfect_fit", "The fixed regressors explain the mixed outcome exactly.")
        logdet_transform = float(torch.linalg.slogdet(design.transform)[1])
        pr = p if reml else 0
        correction = -(n-pr)*math.log(scale)+(logdet_transform if reml else 0.)
        linear_ll = -(n-pr)/2*(_LOG_2PI+math.log(linear_ssr/(n-pr))+1)
        if reml:
            linear_ll -= float(torch.log(torch.linalg.cholesky(pooled[:, :p].T@pooled[:, :p]).diagonal()).sum())
        linear_ll += correction
        result = optimize.maximize_bfgs(like, _start(like),
                   hessian_fn=lambda t: optimize.numerical_hessian(like.gradient, t), raise_on_failure=False)
        theta = result.theta
        relative = float(torch.exp(2*(theta[0]-theta[1])))
        if relative<_BOUNDARY:
            raise AnalysisError("boundary_solution", "The mixed random-intercept variance is estimated at zero; remove that random effect.")
        if not result.converged:
            raise AnalysisError("nonconvergence", "The global mixed likelihood did not converge: "+str(result.diagnostics.get("message")))
        final = like.evaluate(theta)
        vtheta = optimize.information_inverse(-result.hessian)
        setup = SimpleNamespace(structure=like.structure, groups=names, z_names=["_cons"], two_level=False)
        reported, jacobian = _reported(setup, theta)
        reported *= scale*scale
        jacobian *= scale*scale
        transform = torch.zeros((p+2, p+2), dtype=torch.float64)
        transform[:p, :p] = scale*design.transform
        transform[p:, p:] = jacobian
        bread = torch.block_diag(optimize.information_inverse(final.xvx/final.sigma2), vtheta)
        predictions = []
        acc = ClusterAccumulator(p+2, scratch_directory=_scratch_directory()) if spec.covariance != "nonrobust" else None
        if acc:
            stack.callback(acc.close)
        d, s2 = torch.exp(2*(theta[0]-theta[1])), final.sigma2
        for batch, data, means, mass, keys in _rows(sample, store, scale, center, group, means=True):
            x, y, mx, my = data[:, :p], data[:, p], means[:, :p], means[:, p]
            residual = y-x@final.beta
            mean_r = my-mx@final.beta
            within_r = residual-mean_r
            if acc:
                a = mass/(1+mass*d)
                score_g = 2*d*(-.5*a+.5*(a*mean_r).square()/s2)
                score_s = -1+(within_r.square()+mean_r.square()/(1+mass*d))/s2-score_g/mass
                score_beta = ((x-mx)*within_r[:, None]+mx*(mean_r/(1+mass*d))[:, None])/s2
                scores = torch.cat((score_beta, (score_g/mass)[:, None], score_s[:, None]), 1)*batch.weights[:, None]
                acc.add(encode_cluster_labels(batch.frame[clusters[0]]) if clusters else keys, scores)
            take = min(400-len(predictions), len(y))
            observed = batch.numeric(spec.outcome)
            fitted = center+scale*(x@final.beta)
            for pos, actual, expected in zip(batch.positions[:take].tolist(), observed[:take].tolist(), fitted[:take].tolist(), strict=True):
                predictions.append({"row": pos, "observed": actual, "fitted": expected, "residual": actual-expected})
        if acc:
            meat, g = acc.finish()
            bread = bread@meat@bread*g/(g-1)
            covariance_text = "joint fixed/variance parameter sandwich from actual group scores; G/(G-1), block-diagonal model information"
        else:
            covariance_text = "model-based GLS fixed-effect information and profiled variance information, block diagonal; delta method"
        covariance = transform@bread@transform.T
        beta = scale*design.transform@final.beta
        if spec.intercept:
            beta[0] += center
        params = torch.cat((beta, reported))
        _finite(params, covariance)
        terms, equations, variance = _variance_terms(setup)
        ll = float(final.value)+correction
        criteria = information_criteria(ll, p+2, n)
        tests = {"model": wald_test(beta, covariance[:p, :p], range(int(spec.intercept), p), label="Wald chi2 test of the fixed slopes")}
        if spec.covariance == "nonrobust":
            tests["lr_vs_linear"] = common.boundary_lr(2*(ll-linear_ll), 1, "linear model")
        metrics = {"log_likelihood": ll, "aic": criteria["aic"], "bic": criteria["bic"],
                   "n_groups": store.groups, "group_size_min": sizes["t_min"], "group_size_avg": sizes["t_avg"],
                   "group_size_max": sizes["t_max"], "icc": float(reported[0]/reported.sum())}
        extra = {"method": "reml" if reml else "ml", "likelihood": "restricted (REML)" if reml else "full (ML)",
                 "covstructure": "independent", "random_effects": {"group": group, "effects": ["_cons"]},
                 "levels": [{"group": group, "n_groups": store.groups, "size_min": sizes["t_min"], "size_avg": sizes["t_avg"], "size_max": sizes["t_max"]}],
                 "G": [[float(reported[0])]], "residual_variance": float(reported[1]), "icc": {group: metrics["icc"]},
                 "theta": (theta+math.log(scale)).tolist(), "theta_std_error": vtheta.diagonal().sqrt().tolist(),
                 "theta_parameterization": "lowest-level ln random-intercept sd, then ln sigma_e", "linear_log_likelihood": linear_ll}
        if acc:
            extra["notes"] = ["theta_std_error is model-based; the reported covariance is robust.", "No LR test vs. the linear model under robust/cluster covariance."]
        info = {"covariance": spec.covariance, "correction": covariance_text, "df_inference": None, "df_resid": None}
        if acc:
            info.update({"n_clusters": g, "cluster_column": clusters[0] if clusters else group})
        bundle = _result(sample, terms=[*design.terms, *terms], beta=params, covariance=covariance,
                  info=info, metrics=metrics, notes=notes, predictions=predictions, tests=tests,
                  solver="native_global_profiled_mixed_disk_group_moments", diagnostics={**factor_info, "converged": True,
                     "iterations": result.iterations, "group_moments": store.diagnostics}, extra=extra,
                  resource=resource.record(), title=f"Mixed-effects {'REML' if reml else 'ML'} regression", use_t=False,
                  provenance_extra={"gradient": "analytic global profiled fixed effects", "hessian": "numerical Ridders derivative of analytic gradient", "likelihood_evaluation": "global within TSQR and positive between-group random-intercept moments"})
        for item, equation in zip(bundle.coefficients, [spec.outcome]*p+equations, strict=True):
            item.equation = equation
        common.variance_intervals(bundle, [p+i for i, flag in enumerate(variance) if flag], spec.alpha)
        return bundle
