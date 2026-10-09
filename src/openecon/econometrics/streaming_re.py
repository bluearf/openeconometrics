"""Global Swamy–Arora random-effects GLS using bounded native row factors.

Panel means and lengths live in owned SQLite storage. Within, between and
quasi-demeaned regressions use global TSQR; no all-panel vector is collected.
The variance-component formulas and covariance conventions match xtreg RE.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
import sqlite3

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .panel.common import EXACT_FIT, WITHIN_TOL
from .panel.xtreg import _validate
from .replay_sample import ReplaySample
from .streaming_linear import (
    _CrossMoments, _GroupMeans, _Notes, _covariance, _factor, _finite,
    _period_keys, _result, _solve,
)


def _rows(sample, store, response):
    """Physical-order rows with stable standardized response and disk means."""
    for batch in sample.batches():
        data = torch.cat((batch.designs["mean"], response(batch)[:, None]), 1)
        keys = encode_cluster_labels(batch.frame[sample.spec.panel])
        unique, codes = store._codes(keys)
        values, _, _ = store._load_block(unique)
        if bool((values[:, 0]<=0).any()):
            raise AnalysisError("source_changed", "A random-effects panel changed between replay passes.")
        yield batch, data, values[:, 1:][codes], values[:, 0][codes]


def _theta(sizes, sigma_u2, sigma_e2):
    return 1-torch.sqrt(sigma_e2/(sizes*sigma_u2+sigma_e2))


def _theta_summary(store, sigma_u2, sigma_e2):
    # Indexed scalar order statistics reproduce torch.quantile's linear
    # interpolation without retaining an O(groups) theta/length vector.
    db = store.connection
    db.execute("CREATE INDEX re_panel_length ON panelmeta(rows)")
    def value(index):
        size = db.execute("SELECT rows FROM panelmeta ORDER BY rows LIMIT 1 OFFSET ?", (index,)).fetchone()[0]
        return 1-math.sqrt(sigma_e2/(size*sigma_u2+sigma_e2))
    out = {"min": value(0), "max": value(store.groups-1)}
    for name, q in (("p5", .05), ("median", .5), ("p95", .95)):
        position = (store.groups-1)*q
        lower, upper = math.floor(position), math.ceil(position)
        a, b = value(lower), value(upper)
        out[name] = a+(b-a)*(position-lower)
    return out


def _reduced_fit(factor, candidates):
    """Select columns on an exact small factor, retaining original term order."""
    if not candidates:
        return [], float(factor[:, -1].square().sum())
    inside, _ = collinear_columns(factor[:, candidates])
    kept = [candidates[index] for index in inside]
    if not kept:
        return [], float(factor[:, -1].square().sum())
    reduced = torch.cat((factor[:, kept], factor[:, -1:]), 1)
    beta, _, _ = _solve(reduced, len(kept))
    return kept, float((reduced[:, -1]-reduced[:, :-1]@beta).square().sum())


def fit_streaming_re(spec, source, *, batch_rows=None):
    """Exact global xtreg RE in its current unweighted option domain."""
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("singular_design", "The random-effects global factors are numerically singular.") from error
    except (OSError, sqlite3.Error) as error:
        raise _GroupMeans.storage_error() from error


def _fit(spec, source, *, batch_rows):
    notes = _Notes(spec)
    if spec.estimator!="xtreg" or notes.option("model")!="re":
        raise AnalysisError("invalid_spec", "This replay kernel requires xtreg model='re'.")
    _validate(notes, "re")
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean", intercept=True)
    sample.prepare()
    p, n = len(design.terms), sample.nobs
    if not p or design.terms[0]!="Intercept":
        raise AnalysisError("singular_design", "Random-effects GLS requires an identified constant.")
    if n<=p:
        raise AnalysisError("insufficient_observations", "Random-effects GLS leaves no residual degrees of freedom.")
    clusters = ([spec.panel] if spec.covariance=="robust" else
                registry.cluster_columns(spec) if spec.covariance=="cluster" else [])
    resource = sample.plan_rows("global random-effects GLS with disk panel moments", {
        "panel_moments_SQLite_and_bounded_decode": 6*1024**2,
        "one_or_two_way_cluster_SQLite_caches": (2**len(clusters)-1)*6*1024**2,
        "global_within_between_GLS_TSQR_and_covariance": 4096*(p+2)**2,
    }, 768*(p+2))
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), batch.weights, False)
    anchor, magnitude = float(moments.anchor[1]), float(moments.magnitude[1])
    mean = float(moments.mean[1])
    scale = magnitude*math.sqrt(float(moments.m2.value[1]/moments.mass))
    if scale<=0 or not math.isfinite(scale):
        raise AnalysisError("no_within_variation", "The random-effects outcome has no finite representable variation.")
    center = anchor+magnitude*mean
    def response(batch):
        return ((batch.numeric(spec.outcome)-anchor)/magnitude-mean)*(magnitude/scale)
    with ExitStack() as stack:
        store = _GroupMeans(p+1, len(clusters), panel=True)
        stack.callback(store.close)
        for batch in sample.batches():
            data = torch.cat((batch.designs["mean"], response(batch)[:, None]), 1)
            labels = [encode_cluster_labels(batch.frame[name]) for name in clusters]
            periods = _period_keys(batch.frame[spec.time]) if spec.time else None
            store.add(encode_cluster_labels(batch.frame[spec.panel]), data, batch.weights, labels, periods)
        store.finish()
        structure = store.panel_structure()
        g = store.groups
        if n<=g:
            raise AnalysisError("insufficient_observations", "Random effects need panels containing repeated observations.")
        if clusters and not all(store.nested_flags):
            raise AnalysisError("cluster_not_nested", "Every covariance cluster must be constant within each panel.")
        def within_rows():
            for batch, data, means, _ in _rows(sample, store, response):
                value = data-means
                yield value[:, :p], value[:, p], batch.weights
        within, within_info = _factor(within_rows, p, require_more=False)
        pooled_tree, between_tree = _TSQRTree(), _TSQRTree()
        pooled_tree.add(within)
        harmonic = _CompensatedSum(())
        for means, sizes, _, _ in store.group_batches(sample.rows):
            harmonic.add(sizes.reciprocal().sum())
            _, factor = torch.linalg.qr(means, mode="r")
            between_tree.add(factor)
            _, factor = torch.linalg.qr(means*sizes.sqrt()[:, None], mode="r")
            pooled_tree.add(factor)
        pooled, between = pooled_tree.finish(), between_tree.finish()
        within_tss = float(within[:, p].square().sum())
        # Normalized response has total centered variance N. The guard measures
        # actual stochastic variation, rather than the arbitrary original offset.
        if within_tss<=EXACT_FIT*n:
            raise AnalysisError("no_within_variation", "The outcome has no within-panel variation for estimating sigma_e.")
        before = pooled[:, 1:p].square().sum(0)
        after = within[:, 1:p].square().sum(0)
        candidates = (after>WITHIN_TOL*before).nonzero().flatten().add(1).tolist()
        within_kept, ssr_within = _reduced_fit(within, candidates)
        if ssr_within<=EXACT_FIT*n:
            raise AnalysisError("no_within_variation", "The regressors explain the within-panel outcome exactly, so sigma_e is zero.")
        between_kept, ssr_between = _reduced_fit(between, list(range(p)))
        df_within, df_between = n-g-len(within_kept), g-len(between_kept)
        if df_within<=0 or df_between<=0:
            raise AnalysisError("insufficient_observations", "Random-effects variance components need positive within and between degrees of freedom.")
        t_bar = g/float(harmonic.value)
        sigma_e2 = ssr_within/df_within
        sigma_u2 = max(0., ssr_between/df_between-sigma_e2/t_bar)
        method = "swamy_arora_harmonic_mean"
        if notes.option("sa"):
            trace_cross = _CompensatedSum((len(between_kept), len(between_kept)))
            tree = _TSQRTree()
            for means, sizes, _, _ in store.group_batches(sample.rows):
                x = means[:, between_kept]
                block = torch.cat((x, means[:, p:]), 1)*sizes.sqrt()[:, None]
                _, factor = torch.linalg.qr(block, mode="r")
                tree.add(factor)
                trace_cross.add((x*sizes[:, None]).T@(x*sizes[:, None]))
            factor = tree.finish()
            b, inverse, _ = _solve(factor, len(between_kept))
            weighted_ssr = float((factor[:, -1]-factor[:, :-1]@b).square().sum())
            denominator = n-float(torch.trace(inverse@trace_cross.value))
            if denominator<=0:
                raise AnalysisError("insufficient_observations", "The Baltagi-Chang variance denominator is not positive.")
            sigma_u2 = max(0., (weighted_ssr-df_between*sigma_e2)/denominator)
            method = "baltagi_chang_sa"
        if sigma_u2==0:
            notes.warn("The estimated sigma_u is zero: the random-effects GLS reduces to pooled OLS.")
        def transformed_rows():
            for batch, data, means, sizes in _rows(sample, store, response):
                value = data-_theta(sizes, sigma_u2, sigma_e2)[:, None]*means
                yield value[:, :p], value[:, p], batch.weights
        gls, gls_info = _factor(transformed_rows, p)
        working, bread, condition = _solve(gls, p)
        within_corr, overall_corr, between_corr = _CrossMoments(), _CrossMoments(), _CrossMoments()
        def scores():
            for batch, data, means, sizes in _rows(sample, store, response):
                value = data-_theta(sizes, sigma_u2, sigma_e2)[:, None]*means
                original_fit = data[:, :p]@working
                within_corr.add(torch.stack(((data[:, :p]-means[:, :p])@working, data[:, p]-means[:, p]), 1), batch.weights)
                overall_corr.add(torch.stack((original_fit, data[:, p]), 1), batch.weights)
                batch.prediction_response = center+scale*original_fit
                yield batch, value[:, :p], value[:, p], batch.numeric(spec.outcome), scale
        kind = "cluster" if clusters else "nonrobust"
        covariance, info, ssr, predictions = _covariance(sample, scores, working, bread,
            df=n-p, k=p, notes=notes, kind=kind, cluster_columns=clusters,
            score_basis=torch.linalg.inv(design.transform))
        info.update({"df_inference": None, "df_resid": n-p, "covariance": spec.covariance, "k_small_sample": p})
        if spec.covariance=="nonrobust":
            info["correction"] = "GLS: rmse^2 (X*'X*)^-1 with rmse^2 = RSS*/(N-K) of the transformed regression (Stata's e(rmse))"
        pooled_beta, _, _ = _solve(pooled, p)
        pooled_ssr = float((pooled[:, p]-pooled[:, :p]@pooled_beta).square().sum())
        lm_numerator, lm_spread = _CompensatedSum(()), _CompensatedSum(())
        for means, sizes, _, _ in store.group_batches(sample.rows):
            between_corr.add(torch.stack((means[:, :p]@working, means[:, p]), 1), torch.ones_like(sizes))
            lm_numerator.add((sizes*(means[:, p]-means[:, :p]@pooled_beta)).square().sum())
            lm_spread.add(sizes.square().sum())
        if pooled_ssr<=0 or float(lm_spread.value)<=n:
            raise AnalysisError("numerical_failure", "The pooled residuals do not define the Breusch-Pagan random-effects statistic.")
        lm = n*n/(2*(float(lm_spread.value)-n))*(float(lm_numerator.value)/pooled_ssr-1)**2
        transform = scale*design.transform
        beta = transform@working
        beta[0] += center
        covariance = transform@covariance@transform.T
        _finite(beta, covariance)
        theta = _theta_summary(store, sigma_u2, sigma_e2)
        metrics = {"r_squared_within": within_corr.squared_correlation(), "r_squared_between": between_corr.squared_correlation(),
            "r_squared_overall": overall_corr.squared_correlation(), "sigma_u": scale*math.sqrt(sigma_u2),
            "sigma_e": scale*math.sqrt(sigma_e2), "rho": sigma_u2/(sigma_u2+sigma_e2),
            "theta": theta["min"] if structure["t_min"]==structure["t_max"] else None,
            "rmse": scale*math.sqrt(ssr/(n-p)), **structure}
        tests = {"model": wald_test(beta, covariance, range(1, p), label="Wald chi2 test of the slopes"),
            "breusch_pagan": {"statistic": lm, "df": 1, "p_value": .5*chi2_sf(lm, 1), "distribution": "chibar2",
                "label": "Breusch-Pagan LM test of sigma_u = 0 (chibar2(01): chi2(1) tail halved)"}}
        extra = {"model": "re", "variance_components": {"method": method, "sigma_u_squared": scale*scale*sigma_u2,
            "sigma_e_squared": scale*scale*sigma_e2, "ssr_within": scale*scale*ssr_within, "df_within": df_within,
            "within_regressors": len(within_kept), "ssr_between": scale*scale*ssr_between, "df_between": df_between,
            "t_bar_harmonic": t_bar}, "theta": theta, "ssr_transformed": scale*scale*ssr}
        return _result(sample, terms=design.terms, beta=beta, covariance=covariance, info=info, metrics=metrics,
            notes=notes, predictions=predictions, tests=tests, solver="native_global_TSQR_random_effects_disk_panel_moments",
            diagnostics={**within_info, **gls_info, **store.diagnostics, "condition_number": condition}, extra=extra,
            resource=resource.record(), title="Random-effects GLS regression", use_t=False,
            provenance_extra={"model": "re", "random_effects_scope": "global Swamy-Arora or Baltagi-Chang, unweighted; bounded physical row and disk panel means",
                              "theta_summary_algorithm": "indexed SQLite scalar length quantiles; linear interpolation"})
