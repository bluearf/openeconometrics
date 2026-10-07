"""Global Balestra–Varadharajan-Krishnakumar G2SLS from disk panel means.

Auxiliary within and Ti-weighted between IV fits estimate variance components.
All three projections use exact small global native QR factors. Actual rows,
not the factors, supply final residuals and cluster scores. EC2SLS is separate.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
import sqlite3

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .iv import kernels
from .streaming_fe_iv import _raw, _response, _sample
from .streaming_fd_iv import _check_disk_space
from .streaming_linear import _CrossMoments, _GroupMeans, _Notes, _covariance, _finite, _period_keys, _result, _solve
from .streaming_re import _theta, _theta_summary


def _choose(factor, before, k1, kx, intercept):
    """Silent auxiliary variation/rank screens in original left-to-right order."""
    offset = int(intercept)
    small = factor[:, offset:-1].clone()
    if intercept:
        constant = factor[:, :1]
        small -= constant*((constant[:, 0]@small)/(constant[:, 0]@constant[:, 0]))
    after = small.square().sum(0)
    alive = after>1e-13*before
    def select(candidates, base):
        candidates = [index for index in candidates if bool(alive[index])]
        kept, _ = collinear_columns(small[:, [*base, *candidates]])
        return [candidates[index-len(base)] for index in kept if index>=len(base)]
    exog = select(list(range(k1-1)), [])
    endogenous = select(list(range(k1-1, kx-1)), exog)
    excluded = select(list(range(kx-1, len(before))), exog)
    if len(excluded)<len(endogenous):
        raise AnalysisError("underidentified", "The auxiliary RE IV regression has fewer varying excluded instruments than endogenous regressors.")
    return [*exog, *endogenous], [*exog, *excluded], len(endogenous)


def _auxiliary(factor, xindices, zindices, q, intercept):
    offset = int(intercept)
    xi = [*([0] if intercept else []), *[index+offset for index in xindices]]
    zi = [*([0] if intercept else []), *[index+offset for index in zindices]]
    x, y = factor[:, xi], factor[:, -1]
    if not q:
        if not xi:
            return float(y.square().sum()), torch.empty(0, dtype=torch.float64), None
        beta, inverse, _ = _solve(torch.cat((x, y[:, None]), 1), len(xi))
        return float((y-x@beta).square().sum()), beta, inverse
    z = factor[:, zi]
    projected = kernels.project(y, x[:, :0], x, z)
    fit = kernels.k_class(projected, None, 1.)
    return fit.rss, fit.beta, fit.bread


def fit_streaming_re_iv(spec, source, *, batch_rows=None):
    """Native global xtivreg G2SLS, current unweighted covariance/small options."""
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("singular_design", "The random-effects IV global factors are numerically singular.") from error
    except (OSError, sqlite3.Error) as error:
        raise _GroupMeans.storage_error() from error


def _fit(spec, source, *, batch_rows):
    notes = _Notes(spec)
    if spec.estimator!="xtivreg" or notes.option("model")!="re":
        raise AnalysisError("invalid_spec", "This replay kernel requires xtivreg model='re'.")
    ec2sls = bool(notes.option("ec2sls"))
    if spec.weights is not None:
        raise AnalysisError("unsupported_weights", "xtivreg RE currently does not accept observation weights.")
    if spec.covariance not in {"nonrobust", "robust", "cluster"}:
        raise AnalysisError("unsupported_covariance", "RE IV supports nonrobust, panel robust and one cluster column.")
    clusters = ([spec.panel] if spec.covariance=="robust" else registry.cluster_columns(spec) if spec.covariance=="cluster" else [])
    if len(clusters)>1:
        raise AnalysisError("unsupported_cluster_dimensions", "xtivreg currently supports one covariance cluster column.")
    sample, x, z, k1 = _sample(spec, source, batch_rows)
    k, nz, n = len(x.terms), len(z.terms), sample.nobs
    q = k-k1
    if not q or nz-k1<q:
        raise AnalysisError("underidentified", "RE IV requires retained endogenous regressors and at least as many excluded instruments.")
    omitted = [name for name in z.predictors if name not in z.terms and name not in spec.predictors]
    sample.notes["omitted_terms"] = list(dict.fromkeys(term for term in sample.notes["omitted_terms"] if term not in omitted))
    if omitted:
        notes.warn("Instrument(s) omitted because of collinearity: "+", ".join(omitted)+".")
    width = k+nz-k1
    factor_width = k+(2*nz-1 if ec2sls else nz)
    resource = sample.plan_rows("global random-effects IV panel moments and three native projections", {
        "panel_moments_SQLite_and_bounded_decode": 6*1024**2,
        "bounded_actual_score_cluster_SQLite_cache": 6*1024**2 if clusters else 0,
        "global_auxiliary_and_final_IV_factors_covariance": 6144*(factor_width+2)**2,
    }, 1024*(factor_width+2))
    response, center, scale = _response(sample)
    disk_plan = _check_disk_space(sample, width, clusters, "Random-effects IV panel moments")
    with ExitStack() as stack:
        store = _GroupMeans(width, len(clusters), panel=True)
        stack.callback(store.close)
        before = _CompensatedSum((width-1,))
        for batch in sample.batches():
            values = _raw(batch, response, k1)
            before.add(values[:, 1:].square().sum(0))
            store.add(encode_cluster_labels(batch.frame[spec.panel]), values, batch.weights,
                      [encode_cluster_labels(batch.frame[name]) for name in clusters],
                      _period_keys(batch.frame[spec.time]) if spec.time else None)
        store.finish()
        g, structure = store.groups, store.panel_structure()
        if n<=g:
            raise AnalysisError("insufficient_observations", "Every RE IV panel has one observation; idiosyncratic variance is not identified.")
        if spec.covariance=="cluster" and not all(store.nested_flags):
            notes.warn("The panels are not nested within the cluster variable: the random-effects transformation leaves within-panel correlation that this cluster covariance ignores.")
        def rows():
            for batch in sample.batches():
                raw = _raw(batch, response, k1)
                unique, codes = store._codes(encode_cluster_labels(batch.frame[spec.panel]))
                values, _, _ = store._load_block(unique)
                if bool((values[:, 0]<=0).any()):
                    raise AnalysisError("source_changed", "An RE IV panel changed between passes.")
                yield batch, raw, values[:, 1:][codes], values[:, 0][codes]
        within_tree, between_tree, weighted_tree = _TSQRTree(), _TSQRTree(), _TSQRTree()
        for _, raw, means, _ in rows():
            values = raw-means
            within_tree.add(torch.linalg.qr(torch.cat((values[:, 1:], values[:, :1]), 1), mode="r")[1])
        for means, sizes, _, _ in store.group_batches(sample.rows):
            augmented = torch.cat((torch.ones((len(means), 1), dtype=torch.float64), means[:, 1:], means[:, :1]), 1)
            between_tree.add(torch.linalg.qr(augmented, mode="r")[1])
            weighted_tree.add(torch.linalg.qr(augmented*sizes.sqrt()[:, None], mode="r")[1])
        within, between, weighted = within_tree.finish(), between_tree.finish(), weighted_tree.finish()
        wx, wz, wq = _choose(within, before.value, k1, k, False)
        ssr_within, _, _ = _auxiliary(within, wx, wz, wq, False)
        df_within = n-g-len(wx)
        if df_within<=0:
            raise AnalysisError("insufficient_observations", "The within IV variance regression leaves no residual degrees of freedom.")
        if ssr_within<=1e-24*n:
            raise AnalysisError("perfect_fit", "The within IV regression fits the outcome exactly, so sigma_e is zero.")
        sigma_e2 = ssr_within/df_within
        bx, bz, bq = _choose(between, before.value, k1, k, True)
        kb = len(bx)+1
        df_between = g-kb
        if df_between<=0:
            raise AnalysisError("insufficient_observations", "The between IV variance regression leaves no residual degrees of freedom.")
        ssr_between, _, _ = _auxiliary(weighted, bx, bz, bq, True)
        regressor_indices = [0, *[index+1 for index in bx]]
        weighted_regressors = weighted[:, regressor_indices]
        _, inverse, _ = _solve(torch.cat((weighted_regressors, weighted[:, -1:]), 1), kb)
        trace_cross = _CompensatedSum((kb, kb))
        for means, sizes, _, _ in store.group_batches(sample.rows):
            zmean = torch.cat((torch.ones((len(means), 1), dtype=torch.float64), means[:, [index+1 for index in bx]]), 1)*sizes[:, None]
            trace_cross.add(zmean.T@zmean)
        trace = float(torch.trace(inverse@trace_cross.value))
        if n-trace<=0:
            raise AnalysisError("insufficient_observations", "The unbalanced RE IV variance denominator N-trace is not positive.")
        sigma_u2 = max(0., (ssr_between-df_between*sigma_e2)/(n-trace))
        if sigma_u2==0:
            notes.warn("The estimated sigma_u is zero: the random-effects IV estimator reduces to pooled 2SLS.")
        selected_instruments = None
        reference_sums = _CompensatedSum((2*(nz-1),))
        reference_squares = _CompensatedSum((2*(nz-1),))
        def transformed():
            for batch, raw, means, sizes in rows():
                theta = _theta(sizes, sigma_u2, sigma_e2)
                values = raw-theta[:, None]*means
                constant = (1-theta)[:, None]
                real_x = torch.cat((constant, values[:, 1:k]), 1)
                if ec2sls:
                    levels = torch.cat((raw[:, 1:k1], raw[:, k:]), 1)
                    group_means = torch.cat((means[:, 1:k1], means[:, k:]), 1)
                    candidates = torch.cat((levels-group_means, group_means), 1)
                    real_z = torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), candidates), 1)
                    if selected_instruments is not None:
                        real_z = real_z[:, selected_instruments]
                    else:
                        reference = torch.cat((levels, group_means), 1)
                        reference_sums.add(reference.sum(0))
                        reference_squares.add(reference.square().sum(0))
                else:
                    real_z = torch.cat((real_x[:, :k1], values[:, k:]), 1)
                yield batch, raw, real_x, real_z, values[:, 0]
        tree = _TSQRTree()
        for _, _, real_x, real_z, y in transformed():
            tree.add(torch.linalg.qr(torch.cat((real_z, real_x, y[:, None]), 1), mode="r")[1])
        factor = tree.finish()
        candidate_width = 2*nz-1 if ec2sls else nz
        if ec2sls:
            norm = factor[:, 1:candidate_width].square().sum(0)
            reference = (reference_squares.value-reference_sums.value.square()/n).clamp_min(0)
            alive = ((norm>0)&(norm>1e-13*reference)).nonzero().flatten().add(1).tolist()
            kept, _ = collinear_columns(factor[:, [0, *alive]])
            selected_instruments = [0, *[alive[index-1] for index in kept if index>0]]
        compressed_z = factor[:, selected_instruments] if ec2sls else factor[:, :nz]
        compressed_x, compressed_y = factor[:, candidate_width:candidate_width+k], factor[:, -1]
        # 1-theta is not a literal constant. All columns are instrumented on
        # the exact factor; no artificial constant/norm rotation is applied.
        projection = kernels.project(compressed_y, compressed_x[:, :0], compressed_x, compressed_z)
        fit = kernels.k_class(projection, None, 1.)
        working, bread = fit.beta, fit.bread
        df = n-k
        if df<=0:
            raise AnalysisError("insufficient_observations", "RE IV leaves no residual degrees of freedom.")
        within_corr, overall_corr, between_corr = _CrossMoments(), _CrossMoments(), _CrossMoments()
        def scores():
            for batch, raw, real_x, real_z, y in transformed():
                raw_x = torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), raw[:, 1:k]), 1)
                fitted = raw_x@working
                group = store.lookup(encode_cluster_labels(batch.frame[spec.panel]))
                mean_x = torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), group[:, 1:k]), 1)
                within_corr.add(torch.stack(((raw_x-mean_x)@working, raw[:, 0]-group[:, 0]), 1), batch.weights)
                overall_corr.add(torch.stack((fitted, raw[:, 0]), 1), batch.weights)
                batch.residual_override = y-real_x@working
                batch.prediction_response = center+scale*fitted
                projected = real_z@projection.first.beta[:, :k]
                yield batch, projected, y, batch.numeric(spec.outcome), scale
        covariance, info, ssr, predictions = _covariance(sample, scores, working, bread, df=df, k=k, notes=notes,
             score_basis=torch.linalg.inv(x.transform), kind="cluster" if clusters else "nonrobust", cluster_columns=clusters, small=True)
        for means, sizes, _, _ in store.group_batches(sample.rows):
            xm = torch.cat((torch.ones((len(means), 1), dtype=torch.float64), means[:, 1:k]), 1)
            between_corr.add(torch.stack((xm@working, means[:, 0]), 1), torch.ones_like(sizes))
        transform = scale*x.transform
        beta, covariance = transform@working, transform@covariance@transform.T
        beta[0] += center
        _finite(beta, covariance)
        small = bool(notes.option("small"))
        reference = info["df_inference"] if small else None
        info.update({"small": small, "df_inference": reference, "df_resid": df, "covariance": spec.covariance})
        if spec.covariance=="nonrobust":
            info["correction"] = "2SLS on the GLS-transformed data: RSS*/(N-K) (X*'P X*)^-1"
        elif spec.covariance=="robust":
            info["correction"] = "vce(robust) = vce(cluster panel); "+info["correction"]
        def squared(moment):
            variance = moment.m2.value.diagonal()
            return None if min(float(variance[0]), float(variance[1]))<=1e-28*max(float(variance[0]), float(variance[1])) else moment.squared_correlation()
        theta = _theta_summary(store, sigma_u2, sigma_e2)
        metrics = {"r_squared_within": squared(within_corr), "r_squared_between": squared(between_corr), "r_squared_overall": squared(overall_corr),
            "sigma_u": scale*math.sqrt(sigma_u2), "sigma_e": scale*math.sqrt(sigma_e2), "rho": sigma_u2/(sigma_u2+sigma_e2),
            "theta": theta["min"] if structure["t_min"]==structure["t_max"] else None, "rmse": scale*math.sqrt(ssr/df),
            **structure, "n_instruments": nz-k1, "n_endogenous": q}
        extra = {"model": "re", "estimator": "ec2sls" if ec2sls else "g2sls", "endogenous": x.terms[k1:], "instruments": z.terms[k1:],
            "omitted_instruments": omitted, "instrument_columns_used": compressed_z.shape[1], "variance_components": {
                "sigma_u_squared": scale*scale*sigma_u2, "sigma_e_squared": scale*scale*sigma_e2,
                "ssr_within": scale*scale*ssr_within, "df_within": df_within, "within_regressors": len(wx),
                "ssr_between": scale*scale*ssr_between, "df_between": df_between, "between_regressors": kb,
                "trace": trace, "method": "swamy_arora"}, "theta": theta, "ssr_transformed": scale*scale*ssr}
        tests = {"model": wald_test(beta, covariance, range(1, k), df_resid=reference,
                    label="F test of the slopes" if small else "Wald chi2 test of the slopes")}
        return _result(sample, terms=x.terms, beta=beta, covariance=covariance, info=info, metrics=metrics, notes=notes,
            predictions=predictions, tests=tests, solver="native_global_G2SLS_disk_panel_moments", extra=extra, resource=resource.record(),
            title="EC2SLS random-effects IV regression" if ec2sls else "G2SLS random-effects IV regression", use_t=small,
            diagnostics={**store.diagnostics, "joint_TSQR_depth": tree.depth, "first_stage_condition_number": projection.first.condition_number,
                         "second_stage_condition_number": fit.condition_number},
            provenance_extra={"model": "re", "scratch_disk_plan": disk_plan,
                              "random_effects_IV_scope": "global EC2SLS or G2SLS; exact globally screened within/between instruments and separate IV variance components"})
